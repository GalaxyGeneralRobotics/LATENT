from copy import deepcopy
from dataclasses import replace
import jax.numpy as jnp
import mujoco
import numpy as np
from types import ModuleType

from latent_mj.utils.dataset.traj_class import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    SingleData)
from latent_mj.utils.dataset.traj_handler import TrajCarry
from latent_mj.envs.g1_tracking import g1_tracking_constants as consts

class ReplayCallback:

    """Base class that can be used to do things while replaying a trajectory."""

    @staticmethod
    def __call__(env, model, data, traj_sample, carry):
        data = env.set_sim_state_from_traj_data(data, traj_sample, carry)
        mujoco.mj_forward(model, data)
        carry = env.th.update_state(carry)
        return model, data, carry


class ExtendTrajData(ReplayCallback):

    def __init__(self, env, n_samples, model, body_names=None, site_names=None):
        self.b_names, self.b_ids = self.get_body_names_and_ids(env._mj_model, body_names)
        self.s_names, self.s_ids = self.get_site_names_and_ids(env._mj_model, site_names)
        dim_qpos, dim_qvel = 0, 0
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                dim_qpos += 7
                dim_qvel += 6
            else:
                dim_qpos += 1
                dim_qvel += 1
        self.recorder = dict(xpos=np.zeros((n_samples, model.nbody, 3)),
                             xquat=np.zeros((n_samples, model.nbody, 4)),
                             cvel=np.zeros((n_samples, model.nbody, 6)),
                             subtree_com=np.zeros((n_samples, model.nbody, 3)),
                             site_xpos=np.zeros((n_samples, model.nsite, 3)),
                             site_xmat=np.zeros((n_samples, model.nsite, 9)),
                             qpos=np.zeros((n_samples, dim_qpos)),
                             qvel=np.zeros((n_samples, dim_qvel)))
        self.traj_model = TrajectoryModel(njnt=model.njnt,
                                          jnt_type=jnp.array(model.jnt_type),
                                          nbody=model.nbody,
                                          body_rootid=jnp.array(model.body_rootid),
                                          body_weldid=jnp.array(model.body_weldid),
                                          body_mocapid=jnp.array(model.body_mocapid),
                                          body_pos=jnp.array(model.body_pos),
                                          body_quat=jnp.array(model.body_quat),
                                          body_ipos=jnp.array(model.body_ipos),
                                          body_iquat=jnp.array(model.body_iquat),
                                          nsite=model.nsite,
                                          site_bodyid=jnp.array(model.site_bodyid),
                                          site_pos=jnp.array(model.site_pos),
                                          site_quat=jnp.array(model.site_quat))
        self.current_length = 0

    def __call__(self, env, model: mujoco.MjModel, data: mujoco.MjData, traj_sample: SingleData, carry: TrajCarry):
        r"""
        Args:
            env: exp env
            traj_sample: get by `traj_handler.get_current_traj_data(traj_carry, np)`, will call Trajectory.data.get(...)
        """
        model, data, carry = super().__call__(env, model, data, traj_sample, carry)

        self.recorder["xpos"][self.current_length] = data.xpos[self.b_ids]
        self.recorder["xquat"][self.current_length] = data.xquat[self.b_ids]
        self.recorder["cvel"][self.current_length] = data.cvel[self.b_ids]
        self.recorder["subtree_com"][self.current_length] = data.subtree_com[self.b_ids]
        self.recorder["site_xpos"][self.current_length] = data.site_xpos[self.s_ids]
        self.recorder["site_xmat"][self.current_length] = data.site_xmat[self.s_ids]

        # add joint properties
        self.recorder["qpos"][self.current_length] = data.qpos
        self.recorder["qvel"][self.current_length] = data.qvel

        self.current_length += 1

        return model, data, carry

    def extend_trajectory_data(self, traj_data: TrajectoryData, traj_info: TrajectoryInfo):
        assert self.current_length == traj_data.qpos.shape[0]
        assert traj_info.model.njnt == self.traj_model.njnt
        converted_data = {}
        for key, value in self.recorder.items():
            converted_data[key] = jnp.array(value)
        return (traj_data.replace(**converted_data),
                replace(traj_info, body_names=self.b_names if len(self.b_names) > 0 else None,
                        site_names=self.s_names if len(self.s_names) > 0 else None,
                        model=self.traj_model))

    @staticmethod
    def get_body_names_and_ids(model, keys=None):
        """
        Get the names of the bodies in the model. If keys is not None, only return the names of the bodies
        that are in keys, otherwise return all body names.

        Args:
            model: mujoco model
            keys: list of body names

        Returns:
            List of body names and list of body ids.
        """
        keys = deepcopy(keys)
        body_names = []
        ids = range(model.nbody)
        for i in ids:
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if (keys is not None) and (b_name in keys):
                keys.remove(b_name)
            body_names.append(b_name)
        assert keys is None or len(keys) == 0, f"Could not find the following body names: {keys}"
        return body_names, list(ids)

    @staticmethod
    def get_site_names_and_ids(model, keys=None):
        """
        Get the names of the sites in the model. If keys is not None, only return the names of the sites
        that are in keys, otherwise return all site names.

        Args:
            model: mujoco model
            keys: list of site names

        Returns:
            List of site names and list of site ids.
        """
        keys = deepcopy(keys)
        site_names = []
        ids = range(model.nsite)
        for i in ids:
            s_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if (keys is not None) and (s_name in keys):
                keys.remove(s_name)
            site_names.append(s_name)
        assert keys is None or len(keys) == 0, f"Could not find the following site names: {keys}"
        return site_names, list(ids)