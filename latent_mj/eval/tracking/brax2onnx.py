from typing import Dict, Union, Tuple, Mapping
import functools
from absl import logging
from dataclasses import dataclass
import tyro

# --- Set environment variables ---
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- TensorFlow GPU setup ---
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy("float32")

import numpy as np
import matplotlib.pyplot as plt
import jax
import tf2onnx
import onnxruntime as rt


# --- MLP model definition ---
class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        use_tanh_distribution=True,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.model = tf.keras.Sequential(name="MLP_0")
        self.use_tanh_distribution = use_tanh_distribution

        for i, size in enumerate(layer_sizes):
            self.model.add(
                tf.keras.layers.Dense(
                    size, activation=None, use_bias=bias, kernel_initializer=kernel_init, name=f"hidden_{i}"
                )
            )
            if i != len(layer_sizes) - 1 or activate_final:
                if layer_norm:
                    self.model.add(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))

    def call(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Dense):
                if self.activate_final or not layer.name.endswith(
                    f"{len(self.model.layers) // (2 if self.layer_norm else 1) - 1}"
                ):
                    x = self.activation(x)
        loc, _ = tf.split(x, 2, axis=-1)
        if self.use_tanh_distribution:
            return tf.tanh(loc)
        else:
            return loc


# --- Utility functions ---
def build_tf_policy_network(
    action_size,
    hidden_layer_sizes,
    activation="swish",
    kernel_init="lecun_uniform",
    layer_norm=False,
    use_tanh_distribution=True,
):
    if activation == "swish":
        activation = tf.nn.swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size * 2],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        use_tanh_distribution=use_tanh_distribution,
    )


def transfer_weights(jax_params, tf_model):
    for name, params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=name)
        except ValueError:
            logging.error(f"Layer {name} not found in TF model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            tf_layer.set_weights([np.array(params["kernel"]), np.array(params["bias"])])
        else:
            logging.error(f"Unhandled layer type: {type(tf_layer)}")
    logging.info("Weights transferred successfully.")


def get_latest_ckpt(path):
    from pathlib import Path

    ckpts = [ckpt for ckpt in Path(path).glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def convert_jax2onnx(
    ckpt_dir,
    output_path,
    inference_fn,
    hidden_layer_sizes,
    obs_size: Union[int, Mapping[str, Union[Tuple[int, ...], int]]],
    action_size: int,
    policy_obs_key,
    jax_params,
    activation="swish",
    use_tanh_distribution=True,
):
    rand_obs = {
        "state": np.random.randn(1, obs_size["state"][0]).astype(np.float32),
        "privileged_state": np.random.randn(1, obs_size["privileged_state"][0]).astype(np.float32),
    }

    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    tf_model = build_tf_policy_network(
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        use_tanh_distribution=use_tanh_distribution,
    )

    example_input = tf.ones((1, obs_size[policy_obs_key][0]))
    tf_model(example_input)  # build model

    transfer_weights(jax_params[1]["params"], tf_model)

    test_input = [rand_obs[policy_obs_key].reshape(1, -1)]
    tf_pred = tf_model(test_input)[0][0].numpy()

    tf_model.output_names = ["continuous_actions"]

    # Dynamic shape for ONNX conversion
    spec = (tf.TensorSpec([None, obs_size[policy_obs_key][0]], tf.float32, name="obs"),)
    tf2onnx.convert.from_keras(
        tf_model, input_signature=spec, opset=11, output_path=output_path
    )

    sess = rt.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(["continuous_actions"], {"obs": test_input[0].astype(np.float32)})[0][0]

    logging.info("Predictions:")
    np.set_printoptions(precision=2, suppress=True)
    logging.info(f"\n\tJAX  : {jax_pred}\n\tTF   : {tf_pred}\n\tONNX : {onnx_pred}")
    jax2onnx_mae = np.mean(np.abs(jax_pred - onnx_pred))

    np.testing.assert_allclose(onnx_pred, tf_pred, rtol=1e-03, atol=1e-05)
    logging.info(f"Mean absolute error: {jax2onnx_mae:.2e}")
    logging.info(f"Success! ONNX model saved to {output_path}")