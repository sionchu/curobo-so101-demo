# SO-101 + NVIDIA cuRobo demo on Windows (RTX 4070 Ti)

A staged demo built on one Windows 11 PC: cuRobo motion planning for the LeRobot SO-101 arm, driven by
real LeRobot datasets, a depth-reconstructed scene, a perception -> plan -> verify -> recover harness,
Isaac Sim PhysX execution with obstacles, FoundationPose on real RGB-D, and LeRobot policy training
(ACT, SmolVLA) including simulation data. Inspired by what I saw at ROSCon China.

Everything lives in this folder. Isaac Sim (`C:\isaacsim`, 6.0.1-rc.7) is used read-only through its own
`python.bat`; nothing was installed into it. No system CUDA Toolkit was installed (see "CUDA without admin").

> Commit history note: the commits group the work by stage after the fact; they are not a live record of
> the session.

## What was built here (main results)

All of the following was simulated, planned, trained and rendered on this PC (Isaac Sim 6.0.1 + cuRobo + LeRobot):

| | Result (measured, 20 seeded Isaac scenarios) | Video |
|---|---|---|
| Full chain: cuRobo approach -> SmolVLA grasp -> physics verify -> recover -> cuRobo transport around a wall and a box that appears mid-transport | 16/20 delivered (policy grasp 1st try 16, retry 2, cuRobo fallback 2); cuRobo-only 17/20; replan median ~110 ms | `full_chain.mp4` |
| Simulation data: 120 cuRobo expert demos generated in Isaac, SmolVLA co-trained (real + sim) | closed-loop from rest: real-data-only 0/20 -> co-trained 7/20 (expert 20/20) | `sim_finetune_compare.mp4` |
| Planner harness in PhysX with a static wall and a box spawned mid-transport | replanned in 0.137 s, 9.8 mm clearance, 0 contacts | `isaac_harness_obstacles.mp4` |

Details: "Stage 9", "Stage 8" and stage 5 below. `videos/full_journey.mp4` shows these first; the stages built on other
people's recordings (1-4, 6, 7 real-data training) are a short reference section at its end.

## Reference inputs: whose footage is this?

No real robot was run for this project. **External** = someone else's public recording, used as input.
**Made here** = simulated, planned, trained or rendered on this PC. `videos/full_journey.mp4` carries the same
orange (EXTERNAL) / green (MADE HERE) badges on every clip.

| Video | External (not ours) | Made here |
|---|---|---|
| `replay_ep5.mp4` | right panels: real camera video, `lerobot/svla_so101_pickplace` (LeRobot team, Apache-2.0) | left 3D view: recorded joints replayed + cuRobo re-plan |
| `depth_plan.mp4`, `mpc.mp4`, `harness_pickplace.mp4` | right panels: real RGB + depth, `CarolinePascal/so101_depth_raw` (Apache-2.0) | left 3D view: TSDF/ESDF scene, robot placement, cuRobo plans, harness |
| `foundationpose_overlay.mp4` | the RGB video, `CarolinePascal/so101_depth_raw` | pose boxes/axes (FoundationPose run on this PC, NVIDIA model weights) |
| `isaac_harness_obstacles.mp4`, `isaac_policy_grasp.mp4`, `sim_finetune_compare.mp4`, `full_chain.mp4` | - (robot model: SO-101 URDF/meshes from TheRobotStudio/SO-ARM100) | everything: Isaac Sim scenes, physics, rendering, cuRobo, policies |
| plots (`policy_*.png`, `sim_finetune_loss.png`) | reference curves = recorded actions of the external dataset | training runs, predictions |

Inputs we did not make: the two datasets above, `lerobot/smolvla_base` (pretrained weights we fine-tuned),
FoundationPose weights, the SO-101 URDF, and the Isaac Sim / cuRobo software itself.

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
| 8 | Simulation data (cuRobo expert in Isaac) + co-training SmolVLA | Isaac closed-loop success: real-only 0/20, sim co-trained 7/20, expert 20/20; real offline arm MAE 2.01 -> 2.28 | eval scenes share the sim training distribution; no real-robot test |
| 9 | Full chain: cuRobo approach -> SmolVLA grasp -> physics verify -> recover -> cuRobo transport around a static wall + a box appearing mid-transport | delivered 16/20 (policy grasp 1st try 16, retry 2, cuRobo fallback 2); cuRobo-only 17/20; replan median ~110 ms | perception = sim pose + 1.6 mm noise; recovery sometimes uses an unchecked joint move |

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

Data: 120 cuRobo expert pick-and-place episodes generated in Isaac Sim (`sim_demo_gen.py`, 120/120 kept,
170 min, 36,468 frames), with box/brick/camera/lighting sampled around the real dataset's statistics. Merged
with the real dataset (`sim_to_lerobot.py`) -> 170 episodes. SmolVLA was co-trained from the real-data
checkpoint (020000) on real episodes 0-44 + all sim episodes (sim ~77% of frames), 10k steps, batch 8,
66 min, loss 1.06 -> 0.045.

Isaac closed-loop (20 seeded scenarios, seed 1000, different from the training seed; start at rest pose;
15 s; success = brick resting inside the box per physics state):

| Method | Success | Lifted >= 30 mm |
|---|---|---|
| SmolVLA, real data only (020000) | 0/20 | 0/20 |
| SmolVLA + sim co-training (010000) | 7/20 | 12/20 |
| cuRobo expert (upper bound) | 20/20 | 20/20 |

Offline re-evaluation on held-out real episodes 45-49 (same protocol as Stage 7):

| Method | arm MAE | gripper MAE | 10-step chunk MAE | close-timing err (frames) |
|---|---|---|---|---|
| hold current state | 3.19 | 2.28 | - | 4.0 |
| ACT | 3.96 | 1.37 | 4.35 | 3.0 |
| SmolVLA (real only) | 2.01 | 1.16 | 2.89 | 8.0 |
| SmolVLA + sim | 2.28 | 1.26 | 3.09 | 14.6 |

Reading (inference, not verified): the sim data closes the *in-sim* visual gap, but the eval scenes come from
the same distribution as the sim training data, so this says nothing yet about the real robot - and the real
offline metrics got slightly worse. SmolVLA re-runs differ by ~0.01 MAE / ~1 frame (stochastic inference).
The policy is plugged into the cuRobo harness in Stage 9.

Videos: `videos/sim_finetune_compare.mp4` (same 4 scenarios, 3 methods side by side) and
`videos/full_journey.mp4` (all stages, 4.3 min; stages 1-7 are result clips, not screen recordings).

### Stage 9 (full chain in Isaac Sim)

`so101/chain_isaac.py`, same 20 seeded scenarios as Stage 8. Per episode: perceive (virtual sensor = sim brick pose +
1.6 mm noise) -> cuRobo plans to a pre-grasp pose -> the sim co-trained SmolVLA (010000) runs closed-loop at 30 Hz
from the rendered cameras until the brick rises 20 mm or 150 steps -> cuRobo lifts 4 cm -> verify from physics
(both jaws in contact, brick >= 30 mm up) -> on failure: open, back off, re-perceive, policy once more, then the
cuRobo scripted grasp -> cuRobo transport with the brick attached to the planner model -> a 4 x 4 cm box is spawned on
the remaining path after 35 % of it, sensed from the sim's obstacle list, and cuRobo replans from the measured state
-> place -> verify (brick inside the box) -> retreat. The policy runs in its own venv (`policy_server.py`, TCP).

| Metric (20 scenarios) | cuRobo approach + SmolVLA grasp | cuRobo only (scripted grasp) |
|---|---|---|
| delivered (brick in box, physics) | **16/20** | **17/20** |
| grasp verified | 20/20: policy 1st try 16, policy retry 2, cuRobo fallback 2 | 20/20 |
| delivered without the cuRobo fallback (policy grasp + 1 retry only) | 14/20 | - |
| static wall present | 13/20 | 13/20 |
| box appeared mid-transport / replans that found a path | 13 / 9 | 13 / 10 |
| replan time, median (range) | 113 ms (95-974) | 109 ms (93-833) |
| clearance to the new box after replanning | 26-31 mm | 21-30 mm |
| episodes with a robot-obstacle contact (PhysX) | 3 (two grazes <= 0.05 N s, one 7.6 N s) | 2 (grazes <= 0.05 N s) |
| recoveries using the unchecked joint move | 6 | 0 |
| mean wall-clock / sim time per episode | 53 s / 11.3 s | 39 s / 8.7 s |

For reference (Stage 8, no obstacles, start at rest): policy alone end to end 7/20, cuRobo expert 20/20.
"Without fallback" is derived from the same run: those episodes are identical up to the grasp decision.

Every failure in both runs is the same step: no path around the box that appears mid-transport (4 + 3 cases). The
wall contacts are grazes during transport (planned wall clearance ~10 mm; only the dynamic box is inflated by 1.5 cm)
and one 7.6 N s contact in an episode whose recovery used the unchecked joint move (attribution inferred, contacts are
not logged per step). Harness fixes made during the run (each restart re-ran all 20 scenarios):
spawn the box >= 15 mm from the arm's current pose; recovery falls back to a joint move when cuRobo has no valid start;
1.5 cm planning margin around the sensed box (7 mm planned clearance still hit it in physics); relax the place
orientation when the tilt of a policy grasp makes the exact pose unreachable for the 5-DOF arm.

Video: `videos/full_chain.mp4` (38 s: scenario 1 policy grasp + replan, 19 failed grasp -> recovery -> policy retry,
3 replan failure).

### Learning story (what training changed, checkpoint by checkpoint)

`videos/learning_story.mp4` (105 s, Korean captions; `so101/make_learning_story.py`, gripper paths from
`so101/compute_ee.py`): training data grids (9 of 50 real demos, 9 of 120 sim expert demos) and the gripper-tip
paths of all 45 real training + 120 sim episodes (cuRobo FK), loss curves from the train logs, held-out
predictions per checkpoint, then the same Isaac scenarios (4, 6, 10, 14) run by every checkpoint with live
gripper paths, a success chart, and the Stage 9 clip.

| Checkpoint | Isaac closed-loop success / lifted (no obstacles, start at rest) | held-out real arm MAE |
|---|---|---|
| SmolVLA base, never fine-tuned here (cameras fed as camera1/2) | 0/4 / 0/4 (trajectory scenarios only) | - |
| real data only, 20k | 0/20 / 0/20 | 2.02 |
| real + sim, 5k | 5/20 / 9/20 | 2.58 |
| real + sim, 10k | 7/20 / 12/20 | 2.27 |
| cuRobo expert | 20/20 / 20/20 | - |

The 5k row is a new run (`powershell -File pipeline_learning.ps1`); the 20k / 10k / expert rows are the Stage 8
runs. The trajectory reruns of scenarios 4, 6, 10, 14 gave real-only 0/4, 10k 2/4 (lifted 4/4), expert 4/4; SmolVLA
is stochastic, so these do not match the Stage 8 per-scenario outcomes exactly. Expert paths are the commanded cuRobo
trajectories; policy paths are the measured joints after each step.

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

The FoundationPose patches are derived from NVIDIA FoundationPose code (NVIDIA Source Code License, non-commercial),
so they are not in this public repository or its history. FoundationPose itself has to be cloned and set up separately
under its own license.

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
:: full chain (detached; waits for a free GPU; progress in logs\pipeline_chain.txt)
powershell -File pipeline_chain.ps1
.venv\Scripts\python.exe so101\summarize_chain.py
.venv\Scripts\python.exe so101\make_chain_video.py chain_policy 1,19,3
:: co-training + closed-loop/offline evals, detached (progress in logs\pipeline_simft.txt)
powershell -File pipeline_simft.ps1
.venv\Scripts\python.exe so101\make_simft_compare.py
:: learning story: per-checkpoint evals with joint traces (detached; logs\pipeline_learning.txt), then the video
powershell -File pipeline_learning.ps1
.venv\Scripts\python.exe so101\compute_ee.py
.venv-lerobot\Scripts\python.exe so101\make_learning_story.py
.venv\Scripts\python.exe so101\make_full_journey.py
```

## Repository

Public: https://github.com/sionchu/curobo-so101-demo (code, README, the LinkedIn cut videos; data, checkpoints
and the other videos stay local). The FoundationPose-derived patches were removed from the history before publishing.
