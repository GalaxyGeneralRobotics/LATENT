import mujoco
import numpy as np
import jax.numpy as jnp
from mujoco import MjSpec, MjModel, MjData
from typing import Union, Dict, Tuple


def mj_jnt_name2id(name, model):
    """
    Get the joint ID (in the Mujoco datastructure) from the joint name.
    """
    for i in range(model.njnt):
        j = model.joint(i)
        if j.name == name:
            return i
    raise ValueError(f"Joint name {name} not found in model!")


def mj_jntname2qposid(j_name, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qposid(j_id, model)


def mj_jntname2qvelid(j_name, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_name (str): joint name.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
    if j_id == -1:
        raise ValueError(f"Joint name {j_name} not found in model.")

    return mj_jntid2qvelid(j_id, model)


def mj_jntid2qposid(j_id, model):
    """
    Get qpos index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qpos index in MjData corresponding to that joint.
    """
    start_qpos_id = model.jnt_qposadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_id = [i for i in range(start_qpos_id, start_qpos_id+7)]
    else:
        qpos_id = [start_qpos_id]

    return qpos_id


def mj_jntid2qvelid(j_id, model):
    """
    Get qvel index of a joint in mujoco data structure.

    Args:
        j_id (int): joint id.
        model (mjModel): mujoco model.

    Returns:
        list of qvel index in MjData corresponding to that joint.

    """
    start_qvel_id = model.jnt_dofadr[j_id]
    jnt_type = model.jnt_type[j_id]

    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        qvel_id = [i for i in range(start_qvel_id, start_qvel_id+6)]
    else:
        qvel_id = [start_qvel_id]

    return qvel_id


def mj_spec_find_geom_id(spec, geom_name):
    """
    Find geom id in mujoco specification.

    Args:
        spec (MjSpec): Mujoco specification.
        geom_name (str): geom name.

    Returns:
        int: geom id in Mujoco specification.
    """
    for i, g in enumerate(spec.geoms):
        if g.name == geom_name:
            return i
    raise ValueError(f"Geom {geom_name} not found in spec.")


def mj_get_collision_dist_and_normal(geom_id1, geom_id2, data, backend):
    """
    Get the distance and normal of the collision between two geoms.

    Args:
        geom_id1 (int): geom id in Mujoco model.
        geom_id2 (int): geom id in Mujoco model.
        data: Mujoco data structure.
        backend: np or jnp.

    Returns:
        tuple: distance and normal vector.
    """
    if backend == jnp:
        mask = (jnp.array([geom_id1, geom_id2]) == data.contact.geom).all(axis=1)
        mask |= (jnp.array([geom_id2, geom_id1]) == data.contact.geom).all(axis=1)
        idx = jnp.where(mask, data.contact.dist, 1e4).argmin()
        dist = data.contact.dist[idx] * mask[idx]
        normal = (dist < 0) * data.contact.frame[idx, 0, :3]
    else:
        raise NotImplementedError

    return dist, normal


def mj_check_collisions(geom_id1, geom_id2, data, backend):
    """
    Check if two geoms collide.

    Args:
        geom_id1 (int): geom id in Mujoco model.
        geom_id2 (int): geom id in Mujoco model.
        data: Mujoco data structure.
        backend: np or jnp.

    Returns:
        bool: True if geoms collide, False otherwise.
    """

    def _is_in_contact(con_id, res):
        con = data.contact[con_id]
        is_in_contact = np.logical_or(np.logical_and(con.geom1 == geom_id1, con.geom2 == geom_id2),
                                      np.logical_and(con.geom1 == geom_id2, con.geom2 == geom_id1))
        return np.logical_or(res, is_in_contact)

    if backend == jnp:
        return mj_get_collision_dist_and_normal(geom_id1, geom_id2, data, backend)[0] < 0
    else:
        return np.any([_is_in_contact(i, False) for i in range(data.ncon)])


def load_mujoco(xml_file: Union[str, MjSpec],
                model_option_conf: Dict = None) -> Tuple[MjModel, MjModel, MjData, MjSpec]:
    """
    Loads and compiles the Mujoco model from an XML file or MjSpec object.

    Args:
        xml_file (Union[str, MjSpec]): Path to the XML file or a Mujoco specification object.
        model_option_conf (Dict, optional): Configuration options for the model. Defaults to None.

    Returns:
        Tuple[MjModel, MjModel, MjData, MjSpec]: The compiled Mujoco model, duplicate model, data, and spec.
    """
    if isinstance(xml_file, MjSpec):
        if model_option_conf is not None:
            xml_file = modify_option_spec(xml_file, model_option_conf)
        model = xml_file.compile()
        spec = xml_file
    elif isinstance(xml_file, str):
        spec = mujoco.MjSpec.from_file(xml_file)
        if model_option_conf is not None:
            spec = modify_option_spec(spec, model_option_conf)
        model = spec.compile()
    else:
        raise ValueError(f"Unsupported type for xml_file {type(xml_file)}.")

    data = mujoco.MjData(model)
    return model, model, data, spec


def modify_option_spec(spec: MjSpec, option_config: Dict) -> MjSpec:
    """
    Modifies the Mujoco specification options.

    Args:
        spec (MjSpec): The Mujoco specification.
        option_config (Dict): Dictionary of options to modify.

    Returns:
        MjSpec: The modified Mujoco specification.
    """
    if option_config is not None:
        for key, value in option_config.items():
            setattr(spec.option, key, value)
    return spec