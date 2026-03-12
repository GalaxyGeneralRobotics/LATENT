<div align="center">
  <h1 align="center"> LATENT </h1>
  <h3 align="center"> Tsinghua | Galbot </h3>

📑 [Paper](https://github.com/GalaxyGeneralRobotics/LATENT) | 🏠 [Website](https://zzk273.github.io/LATENT/)
</div>

This is the official implementation of ***Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data***. This repository provides an open-source humanoid robot learning pipeline for motion tracker pre-training, online distillation, and high-level policy learning. The pipeline uses MuJoCo for simulation and supports multi-GPU parallel training.

---
---

# News 🚩

[March 13, 2026] Tracking codebase and a small subset of human tennis motion data released. **Now you can track these motions, with the tracking pipeline described in our paper.**

---
---

# TODOs

- [x] Release motion tracking codebase
- [x] Release a small subset of human tennis motion data
- [ ]Release pretrained tracker specialists to track all released human tennis motion data
- [ ] Release all human tennis motion data we used
- [ ] Release DAgger online distillation codebase
- [ ] Release pretrained latent action model trained on our tennis motion data
- [ ] Release high-level tennis-playing policy training codebase
- [ ] Release sim2real designs for high-level tennis-playing policy
- [ ] Release more pretrained checkpoints

---
---

# Initialization

1. Clone the repository:
   ```shell
   git clone git@github.com:GalaxyGeneralRobotics/LATENT.git
   ```

2. Create a virtual environment and install dependencies:
   ```shell
   uv sync -i https://pypi.org/simple 
   ```

3. Create a `.env` file in the project directory with the following content:
   ```bash
   export GLI_PATH=<absolute_project_path>
   export WANDB_PROJECT=<your_project_name>
   export WANDB_ENTITY=<your_entity_name>
   export WANDB_API_KEY=<your_wandb_api_key>
   ```

4. Download the [retargeted tennis data](https://drive.google.com/file/d/1nBGrph4Yf9wLGRZ1tTpOmtU1k3zEAN_y/view?usp=drive_link) and put them under `storage/data/mocap/Tennis/`.

   The file structure should be like:

   ```
    storage/data
    ├── mocap
    │   └── Tennis
    │       ├──p1
    │       │  ├── High-Hit02_Tennis\ 001.npz
    │       │  └── ...
    │       └── ...
    └── assets
        └── ...
   ```

5. Initialize assets

   ```shell
   python latent_mj/app/mj_playground_init.py
   ```

---
---

# Usage

## Initialize environment

```shell
source .venv/bin/activate; source .env;
```

---

## Motion tracking

The motion tracker training pipeline refers to the implementation in [OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack).


### Train the model

   ```bash
   # Train without DR
   python -m latent_mj.learning.train.train_ppo_track_tennis --task G1TrackingTennis --exp_name <your_exp_name>

   # Train with DR
   python -m latent_mj.learning.train.train_ppo_track_tennis --task G1TrackingTennisDR --exp_name <your_exp_name>
   ```

### Evaluate the model

   First, convert the Brax model checkpoint to ONNX:

   ```shell
   python -m latent_mj.app.brax2onnx_tracking --task G1TrackingTennis --exp_name <your_exp_name>
   ```

   Next, run the evaluation script:
   
   ```shell
   python -m latent_mj.eval.tracking.mj_onnx_video --task G1TrackingTennis --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
   ```

---
---

# Acknowledgement

This repository is build upon `jax`, `brax`, `loco-mujoco`, `mujoco_playground`, and `OpenTrack`.

If you find this repository helpful, please cite our work:

```bibtex
```