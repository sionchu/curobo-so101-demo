"""Closed-loop test of a trained LeRobot policy in Isaac Sim (grasp phase of a held-out episode).

    .venv-lerobot\\Scripts\\python.exe so101\\policy_isaac.py --mode replay
    .venv-lerobot\\Scripts\\python.exe so101\\policy_isaac.py --mode policy --policy train\\act\\checkpoints\\last\\pretrained_model

Setup (all approximate, see report): held-out episode 46 starts 45 frames before the recorded gripper close;
a pink 2x4 LEGO-size brick (31.8 x 15.8 x 11.4 mm, assumed) is placed where the recorded gripper closed
(cuRobo FK of the recorded joints). Two cameras approximate the dataset's 'up' (front) and 'side' views; their
real extrinsics/intrinsics are unknown. Robot painted black, table light grey, to narrow the visual gap.
- replay: the recorded human actions are sent open-loop (control for the sim/joint-mapping setup)
- policy: the policy sees the rendered images + sim joint state at 30 Hz and its actions are executed
Success = the brick is lifted >= 30 mm while both jaws touch it (physics contacts).
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from isaac_client import IsaacSim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIMITS = np.array([[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806],
                   [-2.74385, 2.84121], [-0.174533, 1.74533]], np.float32)  # so101_new_calib.urdf


def to_rad(norm):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    r = (lo + hi) / 2 + np.asarray(norm, np.float32) / 100.0 * (hi - lo) / 2
    r[5] = lo[5] + norm[5] / 100.0 * (hi[5] - lo[5])
    return np.clip(r, lo, hi)


def to_norm(rad):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    n = (np.asarray(rad, np.float32) - (lo + hi) / 2) / ((hi - lo) / 2) * 100.0
    n[5] = (rad[5] - lo[5]) / (hi[5] - lo[5]) * 100.0
    return n


def decode(img):
    return np.frombuffer(base64.b64decode(img["b64"]), np.uint8).reshape(img["shape"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "policy"], required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--episode", type=int, default=46)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    ep = json.loads((ROOT / "logs" / f"policy_sim_ep{a.episode}.json").read_text())
    tag = a.tag or a.mode
    tz = ep["min_tool_z"] - 0.003
    yaw = np.radians(ep["tool_yaw_at_close_deg"])
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    dims = [0.0158, 0.0318, 0.0114]  # short side along the jaw-closing axis (tool x)
    c_xy = np.array(ep["tool_at_close"][:2]) + R2 @ np.array([-(dims[0] / 2 + 0.002), 0.0])
    brick = dict(center=[float(c_xy[0]), float(c_xy[1]), tz + dims[2] / 2], yaw=float(yaw), size=0.02, dims=dims,
                 color=[1.0, 0.35, 0.55], mass=0.003)
    cams = {"up": dict(eye=[0.55, 0.0, 0.26], target=[0.0, 0.03, 0.08]),
            "side": dict(eye=[0.30, -0.42, 0.12], target=[0.02, 0.06, 0.07])}
    frames_dir = ROOT / "isaac" / f"frames_policy_{tag}"
    if frames_dir.exists():
        for f in frames_dir.glob("*.jpg"):
            f.unlink()
    sim = IsaacSim(port=5600, frames_dir=frames_dir, log=ROOT / "logs" / f"isaac_policy_{tag}.log")
    r = sim.call(cmd="setup", scene=dict(table_z=tz, friction=1.0, bin=None, cube=brick, place=[5.0, 5.0],
                                         obstacles=[dict(name="backdrop_x", center=[-0.45, 0.0, 0.3], dims=[0.02, 2.0, 0.8], color=[0.92, 0.92, 0.92]),
                                                    dict(name="backdrop_y", center=[0.0, 0.55, 0.3], dims=[2.0, 0.02, 0.8], color=[0.92, 0.92, 0.92])],
                                         q0=to_rad(np.array(ep["states"][ep["start_frame"]])).tolist(), cameras=cams,
                                         robot_color=[0.05, 0.05, 0.05], table_color=[0.85, 0.85, 0.85],
                                         cam_eye=[0.45, 0.35, 0.35], cam_target=[0.02, 0.08, 0.05],
                                         gains=dict(arm_kp=300.0, arm_kd=15.0, grip_kp=8.0, grip_kd=0.5)))
    z0 = r["cube"][0][2]
    pol = pre = post = None
    if a.mode == "policy":
        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(a.policy)
        cfg.device = "cuda"
        pol = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to("cuda").eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy)
        pol.reset()
    log, views, max_rise = [], [], -1.0
    t0 = time.perf_counter()
    for k in range(a.steps):
        rr = sim.call(cmd="render")
        imgs = {n: decode(v) for n, v in rr["images"].items()}
        if k % 3 == 0:
            views.append(np.concatenate([imgs["up"], imgs["side"]], 1))
        q = np.array(rr["q"], np.float32)
        if a.mode == "replay":
            idx = min(ep["start_frame"] + k, ep["n_frames"] - 1)
            act = np.array(ep["actions"][idx], np.float32)
        else:
            obs = {"observation.images.up": imgs["up"], "observation.images.side": imgs["side"],
                   "observation.state": to_norm(q)}
            act = predict_action(obs, pol, torch.device("cuda"), pre, post, use_amp=False, task=ep["task"],
                                 robot_type="so100_follower").cpu().numpy().reshape(-1)
        res = sim.call(cmd="exec", q=[to_rad(act).tolist()], dt=1 / 30, settle=0.0)
        cz = res["cube"][0][2]
        max_rise = max(max_rise, cz - z0)
        log.append(dict(k=k, action=np.round(act, 2).tolist(), q_norm=np.round(to_norm(q), 2).tolist(),
                        brick_z_rise_mm=round((cz - z0) * 1000, 1), contacts=res["contacts"]))
    st = sim.call(cmd="state")
    cz = st["cube"][0][2]
    jaw = {k_.split("|")[0] for e in log[-20:] for k_ in e["contacts"] if k_.endswith("/World/cube")}
    # the 150-step window also covers the demo's place-into-box phase, so judge by the max lift reached
    lifted = [e for e in log if e["brick_z_rise_mm"] > 30.0]
    jaw_lift = {k_.split("|")[0] for e in lifted for k_ in e["contacts"] if k_.endswith("/World/cube")}
    ok = len(lifted) > 0 and {"gripper_link", "moving_jaw_so101_v1_link"} <= jaw_lift
    summary = dict(mode=a.mode, policy=a.policy, episode=a.episode, steps=a.steps, success=bool(ok),
                   final_brick_rise_mm=round((cz - z0) * 1000, 1), max_brick_rise_mm=round(max_rise * 1000, 1),
                   jaw_contacts_while_lifted=sorted(jaw_lift) if lifted else [], steps_lifted_over_30mm=len(lifted), wall_s=round(time.perf_counter() - t0, 1))
    print(json.dumps(summary), flush=True)
    (ROOT / "logs" / f"policy_isaac_{tag}.json").write_text(json.dumps(dict(summary=summary, log=log), indent=1))
    np.save(ROOT / "logs" / f"policy_views_{tag}.npy", np.stack(views))
    sim.close()


if __name__ == "__main__":
    main()
