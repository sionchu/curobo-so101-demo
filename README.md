# SO-101 + NVIDIA cuRobo demo on Windows (RTX 4070 Ti)

A staged demo built on one Windows 11 PC: cuRobo motion planning for the LeRobot SO-101 arm, driven by
real LeRobot datasets, a depth-reconstructed scene, a perception -> plan -> verify -> recover harness,
Isaac Sim PhysX execution with obstacles, FoundationPose on real RGB-D, and LeRobot policy training
(ACT, SmolVLA) including simulation data.

Everything lives in this folder. Isaac Sim (`C:\isaacsim`, 6.0.1-rc.7) is used read-only through its own
`python.bat`; nothing was installed into it. No system CUDA Toolkit was installed (see "CUDA without admin").

> Commit history note: the commits group the work by stage after the fact; they are not a live record of
> the session.

## Stages and results

Legend: **measured** = observed in a run on this machine; **assumption** = a value we had to choose;
**unverified** = not checked against ground truth.

| Stage | What | Key numbers (measured) | Assumptions / unverified |
|---|---|---|---|
| 1 | cuRobo on Windows (Franka, then SO-101 config built from the URDF) | MPC converges to < 1 mm; pose plans ~0.06-0.1 s | SO-101 is 5-DOF: position-priority goals |
| 2 | Replay of real `lerobot/svla_so101_pickplace` episodes + cuRobo re-plan through recorded EE waypoints | 12/12 segments planned, 0.5 mm at waypoints, 8.1 mm mean gap | joint mapping: LeRobot +-100 -> URDF limits (no calibration file) |
| 3 | Scene from real depth (`CarolinePascal/so101_depth_raw`): plane fit, ICP robot base, TSDF/ESDF, planning | ICP residual 6.5 mm (trimmed mean); plan into bin 0.07-0.1 s, 14 mm clearance | D435 depth intrinsics fx=fy=385 (no intrinsics in dataset); `SANGHO1234/...` depth was unusable (all frames saturated) |
| 4 | Harness: perceive -> cuRobo plan -> verify -> recover (kinematic) | injected 25 mm perception error caught by grasp verify (25.1 mm), recovered, placed 5.0 mm from target | rule-based stub policy, kinematic execution |
| 5 | Isaac Sim PhysX execution + static wall + box spawned mid-transport | real gripper grasp (cube lifted 78 mm), detected new box (-7.1 mm on old path), replanned in 0.137 s, 9.8 mm clearance, 0 robot/obstacle contacts, placed 21.5 mm from target | re-perception after the first attempt = sim pose + 1.6 mm noise (virtual camera) |
| 6 | FoundationPose (model-based) on real RGB-D | vs ICP/PnP: cube 0.9 mm xy / -3.9 mm z / -3.8 deg yaw; bin 4.2 mm xy; tracking jitter < 0.6 mm | depth re-projected into RGB with our estimated extrinsics; FP says the cube is tilted ~17 deg (unverified) |
| 7 | LeRobot ACT (scratch, 25k steps) and SmolVLA (fine-tune, 20k steps) on real data; held-out episodes 45-49 | arm MAE: hold-state 3.19, ACT 3.96, SmolVLA 2.00; gripper close timing: hold 4.0, ACT 3.0, SmolVLA 7.2 frames | Isaac closed-loop grasp from rendered cameras failed for both (visual sim-to-real gap; camera extrinsics guessed) |
| 8 | Simulation data (cuRobo expert in Isaac) + co-training SmolVLA | see "Stage 8" below | |

### Stage 7 offline evaluation (held-out episodes 45-49, 1355 frames, dataset units)

| Method | arm MAE | gripper MAE | close-timing err (frames) |
|---|---|---|---|
| mean action (baseline) | 25.63 | 7.21 | - |
| hold current state (baseline) | 3.19 | 2.28 | 4.0 |
| ACT | 3.96 | 1.37 | 3.0 |
| SmolVLA | 2.00 | 1.25 | 7.2 |

"repeat the previous recorded action" gives 0.74 but needs the ground-truth previous action, so it is not a
reachable baseline.

### Stage 8 (simulation data)

_In progress - filled in when the run finishes._

## Layout

```
so101/        all scripts (planning, replay, depth scene, harness, perception, Isaac client, training/eval, sim data)
isaac/        isaac_server.py - Isaac Sim physics/render server (JSON over localhost TCP)
patches/      our changes to third-party code (see below)
*.ps1 *.bat   training launchers and the CUDA build environment for FoundationPose
```

Generated, not in git: `.venv*`, `data/`, `models/`, `train/`, `logs/`, `videos/*.mp4`, Isaac USD/frames,
and the third-party clones.

## Environments

| venv | Python | What | Notes |
|---|---|---|---|
| `.venv` | 3.11 | cuRobo (V2, `pip install -e ./curobo[cu13]`), torch 2.14.1+cu130, viser, pyav, open3d-free | torch from the PyTorch cu130 index (PyPI Windows torch is CPU-only) |
| `.venv-fp` | 3.11 | FoundationPose, nvdiffrast 0.4.0, pytorch3d (built from source), torch 2.11 cu130 | needs `fp_env.bat` |
| `.venv-lerobot` | 3.12 | LeRobot 0.6.2 (`[smolvla,dataset,training,av-dep]`), torch 2.11 cu130 | video decoding via pyav (torchcodec DLL does not load) |

### CUDA without admin (FoundationPose build)

NVIDIA publishes the CUDA 13.0 toolchain as pip wheels. In `.venv-fp`:
`nvidia-cuda-nvcc==13.0.*`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-cuda-crt`, `nvidia-nvvm==13.0.*`
(pin it: the resolver picks 13.4 otherwise), plus header packages `nvidia-cublas/cusparse/cusolver/curand/cufft`.
`fp_env.bat` sets `CUDA_HOME` to `.venv-fp\Lib\site-packages\nvidia\cu13`, loads MSVC 2022 `vcvars64`, and sets
`VSLANG=1033` + UTF-8 (Korean MSVC messages otherwise crash torch's build output decoding).

### Windows workarounds

- LeRobot checkpoint `last` symlink needs admin -> directory junction fallback (`patches/lerobot-windows-checkpoint-junction.patch`).
- LeRobot turns `lerobot/smolvla_base` into a backslash path -> download to `models/smolvla_base` and pass the local path.
- LeRobot dataloader hung once at 4 workers -> resumed from the 10k checkpoint with 2 workers (`train_resume.ps1`).
- PowerShell `*>` logs are UTF-16 and hard-wrapped; parse accordingly.
- FoundationPose writes to `/tmp` -> `patches/foundationpose-windows-tmpdir.patch`; its C++ `mycpp` (Boost/Eigen) is
  replaced by a NumPy port of the single function it provides (`patches/foundationpose-mycpp-numpy-port.py` ->
  `FoundationPose/mycpp/build/mycpp.py`).
- Isaac Sim: one `simulation_app.update()` advances 1/60 s (2 PhysX substeps at 120 Hz); hidden browser panes do
  not render, so videos use a headless Edge (Playwright, `channel="msedge"`) as the Viser render client.

## Third-party code (cloned, not vendored)

`patches/foundationpose-*` contain small changes to / a port of NVIDIA FoundationPose code, which is under the NVIDIA
Source Code License (non-commercial); keep this repository private or remove those files before sharing publicly.

| Repo | Commit | License |
|---|---|---|
| NVlabs/curobo | 78fd485 | Apache-2.0 |
| huggingface/lerobot | 297774f | Apache-2.0 |
| NVlabs/FoundationPose | a1b694b | NVIDIA Source Code License (non-commercial: research/evaluation only) |
| TheRobotStudio/SO-ARM100 | a758567 | see repo (URDF/meshes) |
| NVlabs/nvdiffrast | 253ac4f | NVIDIA Source Code License (non-commercial) |
| facebookresearch/pytorch3d | 88e182f | BSD-3-Clause |

Models / data: `lerobot/smolvla_base` (Apache-2.0), FoundationPose weights (same NVIDIA non-commercial license),
`lerobot/svla_so101_pickplace` (Apache-2.0), `CarolinePascal/so101_depth_raw` (Apache-2.0),
`SANGHO1234/so101_realsense_depth_task_20260814_104640` (Apache-2.0, depth unusable). Isaac Sim is under NVIDIA's
own license and is not redistributed here.

## Re-running (from this folder)

```bat
:: viewers (Viser)
.venv\Scripts\python.exe so101\so101_demo.py mpc --port 8081
.venv\Scripts\python.exe so101\replay.py --episode 0 --port 8083
.venv\Scripts\python.exe so101\depth_planning.py plan --port 8084
.venv\Scripts\python.exe so101\harness_pickplace.py --port 8089
:: Isaac physics harness with obstacles, then its video
.venv\Scripts\python.exe so101\harness_isaac.py
.venv\Scripts\python.exe so101\make_isaac_video.py
:: FoundationPose
fp_env.bat .venv-fp\Scripts\python.exe so101\fp_run.py
:: training / eval
powershell -File train_all.ps1
.venv-lerobot\Scripts\python.exe so101\eval_policies.py train\act\checkpoints\025000\pretrained_model ACT train\smolvla\checkpoints\020000\pretrained_model SmolVLA
.venv-lerobot\Scripts\python.exe so101\policy_isaac.py --mode policy --policy train\smolvla\checkpoints\020000\pretrained_model --tag smolvla
:: simulation data
.venv\Scripts\python.exe so101\sim_demo_gen.py --episodes 120
.venv-lerobot\Scripts\python.exe so101\sim_to_lerobot.py
```

## Pushing to GitHub (not done)

The repo is local only. To publish privately later (needs the GitHub CLI logged in):

```bat
gh repo create so101-curobo-demo --private --source . --push
```
