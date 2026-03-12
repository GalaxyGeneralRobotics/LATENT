import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_dif_rigid_body_pos_local(current_data: mujoco.MjData, trajectory_data: mujoco.MjData):
    """
    Calculate the difference in rigid body positions between the current data and the reference data.
    """
    current_root_pos_g = current_data.qpos[:3]
    current_root_quat_g = current_data.qpos[3:7]
    current_root_rot_g = R.from_quat(current_root_quat_g[[1, 2, 3, 0]])

    ref_root_pos_g = trajectory_data.qpos[:3]
    ref_root_quat_g = trajectory_data.qpos[3:7]
    ref_root_rot_g = R.from_quat(ref_root_quat_g[[1, 2, 3, 0]])

    current_xpos_g = current_data.xpos
    ref_xpos_g = trajectory_data.xpos

    current_xpos_translated = current_xpos_g - current_root_pos_g
    current_xpos_l = current_root_rot_g.inv().apply(current_xpos_translated)

    ref_xpos_translated = ref_xpos_g - ref_root_pos_g
    ref_xpos_l = ref_root_rot_g.inv().apply(ref_xpos_translated)

    dif_rigid_body_pos_local = ref_xpos_l - current_xpos_l

    return dif_rigid_body_pos_local


def linvel_from_pos_diff(pos_curr: np.ndarray, pos_last: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute linear velocity from position difference.
    
    Args:
        pos_curr: Current position, shape (3,)
        pos_last: Last position, shape (3,)
        dt: Time step
        
    Returns:
        Linear velocity in world frame, shape (3,)
    """
    return (pos_curr - pos_last) / dt


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion (scalar-first: w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
