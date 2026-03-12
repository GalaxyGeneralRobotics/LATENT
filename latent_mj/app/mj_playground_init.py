import os
import importlib.util
from pathlib import Path
import shutil

PATH_GLI_STR = os.environ.get("GLI_PATH")
PATH_GLI = Path(PATH_GLI_STR)
PATH_STORAGE = PATH_GLI / "storage"
PATH_ASSET = PATH_STORAGE / "assets"


def create_mujoco_menagerie_soft_link():
    spec = importlib.util.find_spec("mujoco_playground").origin
    mj_play_dir = Path(spec).parent
    tgt_path = Path(mj_play_dir) / "external_deps" / "mujoco_menagerie"

    if tgt_path.exists() and len(list(tgt_path.iterdir())) > 1:
        print(f"{tgt_path} already exists")
        return
    spec = importlib.util.find_spec("latent_mj").origin
    src_path = Path(spec).parent.parent / PATH_ASSET / "mujoco_menagerie"
    if not src_path.exists():
        raise FileExistsError(src_path, "source path does not exist")

    # remove the target dir if it exists
    if tgt_path.exists():
        # remove a softlink
        if tgt_path.is_symlink():
            tgt_path.unlink()
        else:
            shutil.rmtree(tgt_path)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.symlink_to(src_path, target_is_directory=True)
    print(f"Link {src_path} to {tgt_path}")


if __name__ == "__main__":
    create_mujoco_menagerie_soft_link()
