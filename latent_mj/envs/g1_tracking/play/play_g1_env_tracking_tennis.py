import os
import collections
from typing import Union, List, Any, Dict
from dataclasses import dataclass
import time
import copy
import numpy as np
from tqdm import tqdm
import mujoco
import jax
from mujoco.mjx._src import math
import imageio.v2 as imageio

import latent_mj as lmj
from latent_mj.envs.g1_tracking import g1_tracking_constants_tennis as consts
from latent_mj.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryData,
    recalculate_traj_angular_velocity,
    recalculate_traj_joint_velocity,
    recalculate_traj_linear_velocity,
)
from latent_mj.utils.dataset.traj_handler import TrajectoryHandler, TrajCarry


@dataclass
class State:
    info: dict
    obs: dict

@lmj.registry.register("G1TrackingTennis", "tracking_play_env_class")
class PlayG1TrackingTennisEnv:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData

    def __init__(
        self,
        with_racket=True,
        config=None,
        dt=0.02,
        sim_dt=0.002,
        play_ref_motion=False,
        use_viewer=False,
        use_renderer=False,
        exp_name="debug",
    ):
        xml_path = consts.task_to_xml(with_racket=with_racket)
        if not isinstance(xml_path, str):
            xml_path = str(xml_path)
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mj_model.opt.timestep = sim_dt
        self.ref_mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        self.renderer = None
        self.ref_renderer = None
        self.use_viewer = use_viewer
        self.use_renderer = use_renderer
        self.current_traj_info = None
        self.exp_name = exp_name

        if self.use_viewer:
            import mujoco.viewer as viewer

            self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        if self.use_renderer:
            self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
            self.ref_renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
            self.renderer.update_scene(self.mj_data)
            self.ref_renderer.update_scene(self.ref_mj_data)
            self.writer = None
            os.makedirs(f"storage/videos/track/{self.exp_name}", exist_ok=True)

        self._config = config
        self.dt = dt
        self.sim_dt = sim_dt
        self.play_ref_motion = play_ref_motion
        self._config.reference_traj_config.random_start = False
        self._config.reference_traj_config.fixed_start_frame = 0
        self._post_init()

    def _post_init(self):
        self._num_joints = len(self.mj_data.qpos[7:])
        self._default_qpos = np.array(consts.DEFAULT_QPOS[7:])

        # ======= Build excluded joints/bodies indices =======
        excluded_joints = consts.EXCLUDED_ACTION_JOINTs
        excluded_bodies = consts.EXCLUDED_TRACKING_LINKs
        assert (set(excluded_joints).issubset(set(consts.ACTION_JOINT_NAMES))) and (set(excluded_joints).issubset(set(consts.UPPER_BODY_JOINTs))), \
            f"Excluded joints must be a subset of action joints and upper body joints. Got excluded_joints: {excluded_joints}"
        assert (set(excluded_bodies).issubset(set(consts.UPPER_BODY_LINKs))), \
            f"Excluded bodies must be a subset of upper body links. Got excluded_bodies: {excluded_bodies}"
        
        self._excluded_joint_names = excluded_joints
        self._excluded_body_names = excluded_bodies
        
        self._excluded_qpos_adrs = np.array([
            (self.mj_model.joint(j).qposadr - 7).item() for j in excluded_joints
        ]) if excluded_joints else np.array([], dtype=np.int32)
        
        self._excluded_body_ids = np.array([
            self.mj_model.body(b).id for b in excluded_bodies
        ]) if excluded_bodies else np.array([], dtype=np.int32)
        
        self._excluded_actuator_ids = np.array([
            self.mj_model.actuator(j).id for j in excluded_joints
        ]) if excluded_joints else np.array([], dtype=np.int32)
        
        self._excluded_joint_ids = np.array([
            self.mj_model.joint(j).id for j in excluded_joints
        ]) if excluded_joints else np.array([], dtype=np.int32)

        # ======= Build action joints (full vs active) =======
        self.all_actuator_names = consts.ACTION_JOINT_NAMES.copy()
        self.all_actuator_ids = np.array([
            self.mj_model.actuator(j_name).id for j_name in self.all_actuator_names
        ])
        
        self.active_actuator_names = [j for j in self.all_actuator_names if j not in excluded_joints]
        self.active_actuator_ids = np.array([
            self.mj_model.actuator(j_name).id for j_name in self.active_actuator_names
        ])
        
        self._active_actuator_to_full = np.array([
            self.all_actuator_names.index(j) for j in self.active_actuator_names
        ])

        self._active_qpos_to_full = np.array([
            (self.mj_model.joint(j).qposadr - 7).item() for j in self.active_actuator_names
        ])

        # ======================================================

        self.obs_joint_names = consts.OBS_JOINT_NAMES.copy()
        self.obs_joint_ids = []
        for j_name in self.obs_joint_names:
            self.obs_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.obs_joint_ids = np.array(self.obs_joint_ids)

        self._floor_geom_id = self.mj_model.geom("floor").id
        self._torso_imu_site_id = self.mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        
        self._feet_geom_id = np.array([self.mj_model.geom(name).id for name in consts.FEET_GEOMS])
        self._feet_site_id = np.array([self.mj_model.site(name).id for name in consts.FEET_SITES])
        self._feet_all_site_id = np.array([self.mj_model.site(name).id for name in consts.FEET_ALL_SITES])
        self._hands_site_id = np.array([self.mj_model.site(name).id for name in consts.HAND_SITES])

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self.mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self.mj_model.sensor_adr[sensor_id]
            sensor_dim = self.mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr)

        # ======================================================
        
        def filter_body_ids(body_names):
            return np.array([self.mj_model.body(n).id for n in body_names 
                           if self.mj_model.body(n).id not in self._excluded_body_ids])
        
        def filter_joint_ids(joint_names):
            return np.array([self.mj_model.joint(n).id for n in joint_names 
                           if self.mj_model.joint(n).id not in self._excluded_joint_ids])
        # ======================================================

        self.body_id_pelvis = self.mj_model.body("pelvis").id
        self.body_id_torso = self.mj_model.body("torso_link").id
        self.body_names_left_leg = ["left_knee_link", "left_ankle_roll_link"]
        self.body_ids_left_leg = [self.mj_model.body(n).id for n in self.body_names_left_leg]
        self.body_names_right_leg = ["right_knee_link", "right_ankle_roll_link"]
        self.body_ids_right_leg = [self.mj_model.body(n).id for n in self.body_names_right_leg]

        # filter
        self.upper_body_ids = filter_body_ids(consts.UPPER_BODY_LINKs)
        self.lower_body_ids = np.array([self.mj_model.body(n).id for n in consts.LOWER_BODY_LINKs])
        self.upper_body_joints = filter_joint_ids(consts.UPPER_BODY_JOINTs)
        self.key_body_ids = filter_body_ids(consts.KEY_BODY_LINKs)
        self.feet_ids = np.array([self.mj_model.body(n).id for n in consts.FEET_LINKs])
        self.valid_body_ids = np.concatenate((self.lower_body_ids, self.upper_body_ids))
        
        self._kps = np.array(consts.KPs)
        self._kds = np.array(consts.KDs)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # reference trajectory
        self.th: TrajectoryHandler = None
        if self._config.reference_traj_config.random_start:
            self._th_params = {"random_start": True}
        else:
            self._th_params = {
                "random_start": False,
                "fixed_start_conf": [0, self._config.reference_traj_config.fixed_start_frame],
            }

        # load reference trajectory
        self.prepare_trajectory(self._config.reference_traj_config.name, self._config.recalculate_vel_in_reference_motion)
        self.all_evaluation_metrics = collections.defaultdict(list)

        # ======= Excluded joints random target state =======
        self._excluded_joints_config = getattr(self._config, 'excluded_joints_config', None)
        if self._excluded_joints_config is None:
            # Default config if not provided
            self._excluded_joints_config = {
                'enable_random_targets': True,
                'reset_interval_range': [0.5, 2.0],
                'target_range_scale': 1.0,
            }
        
        # Initialize excluded joints state
        self._excluded_targets = self._default_qpos[self._excluded_qpos_adrs].copy() if len(self._excluded_qpos_adrs) > 0 else np.array([])
        self._excluded_next_reset_step = 0

        # output the dataset and observation info of tennis tracker
        print("=" * 50)
        print(
            f"Tracking {self.th.n_trajectories} trajectories with {self.th.traj.data.qpos.shape[0]} timesteps, fps={1 / self.dt:.1f}"
        )
        print(f"Observation: {self._config.obs_keys}")
        print(f"Privileged state: {self._config.privileged_obs_keys}")
        print(f"Excluded joints: {self._excluded_joint_names}")
        print(f"Active action joints: {len(self.active_actuator_names)} (policy output dim)")
        print("=" * 50)

    def reset(self):
        # reset the trajectory handler
        self.current_traj_info = TrajCarry(key=jax.random.PRNGKey(123), traj_state=self.th.init_state())
        self.current_traj_info = self.th.reset_state(self.current_traj_info)

        # reset the simulator from the current trajectory
        obs, info = self._reset_from_current_traj()

        # view or render
        if self.use_viewer:
            self.viewer.sync()

        if self.use_renderer:
            os.makedirs(f"storage/videos/track/{self.exp_name}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][0]}", exist_ok=True)

            self.writer = imageio.get_writer(
                f"storage/videos/track/{self.exp_name}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][0]}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][1]}.mp4",
                fps=50,
            )

            self.renderer.update_scene(self.mj_data, camera=0)
            pixels = self.renderer.render()

            if not self.play_ref_motion:
                self.ref_renderer.update_scene(self.ref_mj_data, camera=0)
                ref_pixels = self.ref_renderer.render()
                self.writer.append_data(np.concatenate([pixels, ref_pixels], axis=1))
            else:
                self.writer.append_data(pixels)

        return State(info, obs)

    def _reset_from_current_traj(self):
        traj_data = self.th.get_current_traj_data(self.current_traj_info)
        qpos, qvel = traj_data.qpos, traj_data.qvel
        self.evaluation_metrics = collections.defaultdict(list)

        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        self.ref_mj_data.qpos[:] = qpos
        self.ref_mj_data.qvel[:] = qvel
        self.mj_data.ctrl[:] = qpos[7:]

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # get the next trajectory data
        init_traj_data = copy.deepcopy(traj_data)
        self.current_traj_info = self.th.update_state_play(self.current_traj_info)
        traj_data = self.th.get_current_traj_data(self.current_traj_info)

        # Reset excluded joints state
        self._reset_excluded_joints_state()

        info = {
            "step": 0,
            "last_motor_targets": self.mj_data.qpos[7:].copy(),
            "motor_targets": np.zeros(self._num_joints),
            "traj_info": self.current_traj_info,
        }
        obs, history = self.get_obs(traj_data, info)

        # history
        if self._config.history_len > 0:
            _, init_history = self.get_obs(init_traj_data, info)
            info["previous_obs"] = np.stack([init_history] * self._config.history_len, axis=0)
            obs["state"] = np.concatenate([obs["state"], info["previous_obs"].flatten()], axis=0)
            obs["privileged_state"] = np.concatenate([obs["privileged_state"], info["previous_obs"].flatten()], axis=0)
            info["previous_obs"] = np.concatenate([info["previous_obs"][1:], history[None, :]], axis=0)

        return obs, info

    def _reset_excluded_joints_state(self):
        """Reset the random target state for excluded joints."""
        if len(self._excluded_qpos_adrs) == 0:
            return
            
        # Get config values (handle both dict and config_dict)
        if hasattr(self._excluded_joints_config, 'enable_random_targets'):
            enable_random = self._excluded_joints_config.enable_random_targets
            interval_range = self._excluded_joints_config.reset_interval_range
        else:
            enable_random = self._excluded_joints_config.get('enable_random_targets', True)
            interval_range = self._excluded_joints_config.get('reset_interval_range', [0.5, 2.0])
        
        if enable_random:
            # Initialize targets from current reference motion
            self._excluded_targets = self.mj_data.qpos[7:][self._excluded_qpos_adrs].copy()
            # Set first reset step
            self._excluded_next_reset_step = int(np.random.uniform(interval_range[0], interval_range[1]) / self.dt)
        else:
            self._excluded_next_reset_step = 999999  # Never reset

    def _update_excluded_joints_targets(self, step: int):
        """Update random targets for excluded joints if needed."""
        if len(self._excluded_qpos_adrs) == 0:
            return
            
        # Get config values
        if hasattr(self._excluded_joints_config, 'enable_random_targets'):
            enable_random = self._excluded_joints_config.enable_random_targets
            interval_range = self._excluded_joints_config.reset_interval_range
        else:
            enable_random = self._excluded_joints_config.get('enable_random_targets', True)
            interval_range = self._excluded_joints_config.get('reset_interval_range', [0.5, 2.0])
        
        if not enable_random:
            return
            
        if step >= self._excluded_next_reset_step:
            # Generate new random targets within joint limits
            excluded_lowers = self._lowers[self._excluded_qpos_adrs]
            excluded_uppers = self._uppers[self._excluded_qpos_adrs]
            self._excluded_targets = np.random.uniform(excluded_lowers, excluded_uppers)
            
            # Calculate next reset step
            self._excluded_next_reset_step = step + int(np.random.uniform(interval_range[0], interval_range[1]) / self.dt)

    def step(self, state: State, action: np.ndarray):
        # action is defined as the deviation from reference motion
        # action is for active joints only (dim(action) = num_active_actuators, excluding right wrist)
        step_start = time.time()

        if self.play_ref_motion:
            qpos, qvel = self.th.get_current_traj_data_fast(state.info["traj_info"])
            self.mj_data.qpos[:] = qpos
            self.mj_data.qvel[:] = qvel
            mujoco.mj_forward(self.mj_model, self.mj_data)
        else:
            qpos, qvel = self.th.get_current_traj_data_fast(state.info["traj_info"])

            self.ref_mj_data.qpos[:] = qpos
            self.ref_mj_data.qvel[:] = qvel
            mujoco.mj_forward(self.mj_model, self.ref_mj_data)

            # Update excluded joints random targets
            self._update_excluded_joints_targets(state.info["step"])

            ref_joint_qpos = np.array(qpos[7:])
            active_motor_targets = ref_joint_qpos[self.active_actuator_ids] + action * self._config.action_scale

            motor_targets = self._default_qpos.copy()
            motor_targets[self.active_actuator_ids] = active_motor_targets

            # Excluded joints: use random targets (periodically reset)
            if len(self._excluded_actuator_ids) > 0:
                if hasattr(self._excluded_joints_config, 'enable_random_targets'):
                    enable_random = self._excluded_joints_config.enable_random_targets
                else:
                    enable_random = self._excluded_joints_config.get('enable_random_targets', True)
                    
                if enable_random:
                    motor_targets[self._excluded_actuator_ids] = self._excluded_targets
                else:
                    # If random targets disabled, use reference motion for excluded joints
                    pass

            full_action = np.zeros(len(self.all_actuator_ids))
            full_action[self._active_actuator_to_full] = action

            state.info["last_motor_targets"] = motor_targets.copy()
            state.info["last_act"] = full_action.copy()

            for _ in range(int(self.dt / self.sim_dt)):
                torques = consts.KPs * (motor_targets - self.mj_data.qpos[7:]) + consts.KDs * (-self.mj_data.qvel[6:])
                torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
                self.mj_data.ctrl[:] = torques
                mujoco.mj_step(self.mj_model, self.mj_data)

        # view or render
        if self.use_viewer:
            self.viewer.sync()

        if self.use_renderer:
            self.renderer.update_scene(self.mj_data, camera=0)
            pixels = self.renderer.render()

            if not self.play_ref_motion:
                self.ref_renderer.update_scene(self.ref_mj_data, camera=0)
                ref_pixels = self.ref_renderer.render()
                self.writer.append_data(np.concatenate([pixels, ref_pixels], axis=1))
            else:
                self.writer.append_data(pixels)

        state.info["traj_info"] = self.th.update_state_play(state.info["traj_info"], backend=np)

        # trajectory change, reset the environment
        if state.info["traj_info"].traj_state.traj_no != self.current_traj_info.traj_state.traj_no:
            self.current_traj_info = state.info["traj_info"]
            obs, info = self._reset_from_current_traj()
            state.info = info

            if self.use_renderer and self.writer is not None:
                self.writer.close()
                os.makedirs(f"storage/videos/track/{self.exp_name}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][0]}", exist_ok=True)
                self.writer = imageio.get_writer(
                    f"storage/videos/track/{self.exp_name}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][0]}/{self.ref_traj_names[self.current_traj_info.traj_state.traj_no][1]}.mp4",
                    fps=50,
                )
        else:
            self.current_traj_info = state.info["traj_info"]
            traj_data = self.th.get_current_traj_data(state.info["traj_info"])
            qpos, qvel = traj_data.qpos, traj_data.qvel
            obs, history = self.get_obs(traj_data, state.info)

            if self._config.history_len > 0:
                obs["state"] = np.concatenate([obs["state"], state.info["previous_obs"].flatten()], axis=0)
                obs["privileged_state"] = np.concatenate(
                    [obs["privileged_state"], state.info["previous_obs"].flatten()], axis=0
                )
                state.info["previous_obs"] = np.concatenate([state.info["previous_obs"][1:], history[None, :]], axis=0)

            # update buffer
            state.info["step"] += 1
            state.info["last_act"] = action.copy()

        # sleep to wait for the next step
        if self.use_viewer:
            time_until_next_step = self.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        return State(state.info, obs)

    def get_obs(self, traj_data, info):
        # pose
        gyro_pelvis = self.get_gyro("pelvis")
        gvec_pelvis = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])

        # joint
        joint_pos = self.mj_data.qpos[7:]
        joint_vel = self.mj_data.qvel[6:]

        # reference
        dif_joint_pos = traj_data.qpos[7:] - joint_pos
        dif_joint_vel = traj_data.qvel[6:] - joint_vel
        traj_root_rot_mat = quat_to_mat(traj_data.qpos[3:7])

        ref_feet_height = self.ref_mj_data.site_xpos[self._feet_all_site_id, 2]

        state_dict = {
            "gyro_pelvis": gyro_pelvis * self._config.obs_scales_config.joint_vel,
            "gvec_pelvis": gvec_pelvis,
            "joint_pos": (joint_pos - self._default_qpos)[self.obs_joint_ids],
            "joint_vel": joint_vel[self.obs_joint_ids] * self._config.obs_scales_config.joint_vel,
            "last_motor_targets": info["last_motor_targets"],
            "dif_joint_pos": dif_joint_pos,
            "dif_joint_vel": dif_joint_vel * self._config.obs_scales_config.dif_joint_vel,
            "ref_feet_height": ref_feet_height,
            "ref_root_height": traj_data.qpos[2],
            "ref_root_linvel_local": (traj_root_rot_mat.T @ traj_data.qvel[:3]) * self._config.obs_scales_config.joint_vel,
            "ref_root_angvel_local": traj_data.qvel[3:6] * self._config.obs_scales_config.joint_vel,
        }

        state = np.hstack([state_dict[k] for k in self._config.obs_keys])
        history = np.hstack([state_dict[k] for k in self._config.history_keys])

        return {"state": state, "privileged_state": None}, history

    def load_trajectory(self, traj: Trajectory = None, warn: bool = True) -> None:
        th_params = self._th_params if self._th_params is not None else {}
        return TrajectoryHandler(model=self.mj_model, warn=warn, traj=traj, control_dt=self.dt, **th_params)

    def prepare_trajectory(self, dataset_dict: Dict[str, List[str]], recalculate_vel_in_reference_motion: bool) -> Trajectory:
        self.ref_traj_names = []
        all_trajectories = []
        for dataset_name, traj_names in dataset_dict.items():
            print(f"Loading dataset: {dataset_name} with {len(traj_names)} trajectories.")
            dataset_dir = os.path.join(os.getcwd(), "storage", "data", "mocap", dataset_name)
            for idx, t_name in enumerate(tqdm(traj_names)):
                traj_path = os.path.join(dataset_dir, f"{t_name}.npz")
                
                if os.path.exists(traj_path):
                    print(f"Loading trajectory {t_name} from {dataset_dir}")
                    traj = Trajectory.load(traj_path, backend=np)

                    if recalculate_vel_in_reference_motion:
                        traj = recalculate_traj_angular_velocity(traj, frequency=1.0 / self.dt, backend=np)
                        traj = recalculate_traj_linear_velocity(traj, frequency=1.0 / self.dt, backend=np)
                        traj = recalculate_traj_joint_velocity(traj, frequency=1.0 / self.dt, backend=np)
                    all_trajectories.append(traj)
                    self.ref_traj_names.append([dataset_name, t_name])
                    continue
                raise FileNotFoundError(f"Motion file {traj_path} not found.")
            
        assert len(all_trajectories) > 0, "No valid trajectories found in the dataset."

        # concatenate trajectories
        if len(all_trajectories) == 1:
            trajectory = all_trajectories[0]
        else:
            traj_datas = [t.data for t in all_trajectories]
            traj_infos = [t.info for t in all_trajectories]
            traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=np)
            trajectory = Trajectory(traj_info, traj_data)

        # load trajectory again to ensure the latest transformed trajectories is loaded
        self.th = self.load_trajectory(trajectory, warn=False)

        return trajectory

    def close(self):
        if self.use_renderer and self.writer is not None:
            self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.ref_renderer is not None:
            self.ref_renderer.close()

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def get_gyro(self, frame: str) -> np.ndarray:
        """Return the gyroscope readings in the local frame."""
        return self.get_sensor_data(f"{consts.GYRO_SENSOR}_{frame}")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion into a 9-dimensional rotation matrix."""
    q = np.outer(q, q)

    return np.array(
        [
            [
                q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                2 * (q[1, 2] - q[0, 3]),
                2 * (q[1, 3] + q[0, 2]),
            ],
            [
                2 * (q[1, 2] + q[0, 3]),
                q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                2 * (q[2, 3] - q[0, 1]),
            ],
            [
                2 * (q[1, 3] - q[0, 2]),
                2 * (q[2, 3] + q[0, 1]),
                q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
            ],
        ]
    )
