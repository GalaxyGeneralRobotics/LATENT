# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logger for training metrics."""

import collections
import logging
import numpy as np

class EpisodeMetricsLogger:
    """Logs training metrics for each episode."""

    def __init__(self, devices, buffer_size=500, steps_between_logging=1e5, progress_fn=None):
        self._devices = devices
        self._rollout_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._average_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._training_metrics_buffer = collections.defaultdict(lambda: collections.deque(maxlen=buffer_size))
        self._buffer_size = buffer_size
        self._steps_between_logging = steps_between_logging
        self._num_steps = 0
        self._last_log_steps = 0
        self._log_count = 0
        self._progress_fn = progress_fn

    def update_episode_metrics(self, aggregated_metrics, dones, training_metrics):
        # Convert JAX arrays to numpy FIRST to avoid slow device-to-host transfers during iteration
        dones_np = np.asarray(dones)
        self._num_steps += np.prod(dones_np.shape) * self._devices
        
        if np.sum(dones_np) > 0:
            dones_mask = dones_np.astype(bool)
            for name, metric in aggregated_metrics.items():
                # Convert to numpy before indexing to avoid JAX array iteration
                metric_np = np.asarray(metric)
                done_metrics = metric_np[dones_mask].reshape(-1).tolist()  # Convert to Python list for fast extend
                if "average_" in name:
                    self._average_metrics_buffer[name.replace("average_", "")].extend(done_metrics)
                else:
                    self._rollout_metrics_buffer[name].extend(done_metrics)
        for name, metric in training_metrics.items():
            # Convert to numpy/list for fast deque operations
            metric_list = np.asarray(metric).reshape(-1).tolist()
            self._training_metrics_buffer[name].extend(metric_list)
        
        if self._num_steps - self._last_log_steps >= self._steps_between_logging:
            self.log_metrics()
            self._last_log_steps = self._num_steps

    def log_metrics(self, pad=35):
        """Log metrics to console."""
        self._log_count += 1
        log_string = f"\n{'Steps':>{pad}} Env: {self._num_steps} Log: {self._log_count}\n"
        episode_metrics = {}
        average_metrics = {}
        training_metrics = {}
        for metric_name in self._rollout_metrics_buffer:
            episode_metrics[metric_name] = np.mean(self._rollout_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Episode {metric_name}:':>{pad}}" f" {episode_metrics[metric_name]:.4f}\n"
        for metric_name in self._average_metrics_buffer:
            average_metrics[metric_name] = np.mean(self._average_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Average {metric_name}:':>{pad}}" f" {average_metrics[metric_name]:.4f}\n"
        for metric_name in self._training_metrics_buffer:
            training_metrics[metric_name] = np.mean(self._training_metrics_buffer[metric_name]).astype(np.float64)
            log_string += f"{f'Training {metric_name}:':>{pad}}" f" {training_metrics[metric_name]:.4f}\n"
        logging.info(log_string)

        if self._progress_fn is not None:
            episode_metrics = {f"episode/{name}": value for name, value in episode_metrics.items()}
            average_metrics = {f"average/{name}": value for name, value in average_metrics.items()}
            training_metrics = {f"training/{name}": value for name, value in training_metrics.items()}
            self._progress_fn(
                int(self._num_steps),
                {**episode_metrics, **average_metrics, **training_metrics},
            )
