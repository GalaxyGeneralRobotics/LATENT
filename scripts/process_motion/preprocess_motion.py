import os
import jax
from dataclasses import dataclass
import tyro
import latent_mj as lmj

@dataclass
class Args:
    task: str
    num_batches: int

def motion_preprocess(args: Args):
    env_class = lmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = lmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config

    env = env_class(config=env_cfg)
    batch_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    print(args.task, args.num_batches, batch_idx)

    env.preprocess_trajectory(env._config.reference_traj_config.name, batch_idx, args.num_batches)

    return


if __name__ == "__main__":

    motion_preprocess(tyro.cli(Args))
