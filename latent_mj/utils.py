import importlib.util
from pathlib import Path
import shutil

import latent_mj
from latent_mj.constant import PATH_ASSET


def create_mujoco_menagerie_soft_link():
    spec = importlib.util.find_spec("mujoco_playground").origin
    mj_play_dir = Path(spec).parent
    tgt_path = Path(mj_play_dir) / "external_deps" / "mujoco_menagerie"

    if tgt_path.exists() and len(list(tgt_path.iterdir())) > 1:
        print(f"{tgt_path} already exists")
        return

    src_path = Path(latent_mj.__file__).parent.parent / PATH_ASSET / "mujoco_menagerie"
    if not src_path.exists():
        raise FileExistsError(src_path, "source path does not exist")

    # remove the target dir if it exists
    if tgt_path.exists():
        shutil.rmtree(tgt_path)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.symlink_to(src_path, target_is_directory=True)
    print(f"Created {tgt_path}")


if __name__ == "__main__":
    create_mujoco_menagerie_soft_link()
