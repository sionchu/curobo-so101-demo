"""Closed-loop evaluation in Isaac Sim on fixed, seeded scenarios (same for every method).

    .venv-lerobot\\Scripts\\python.exe so101\\eval_isaac_closedloop.py --make-scenarios
    .venv-lerobot\\Scripts\\python.exe so101\\eval_isaac_closedloop.py --policy train\\smolvla\\checkpoints\\020000\\pretrained_model --tag real_only
    .venv\\Scripts\\python.exe so101\\eval_isaac_closedloop.py --expert --tag curobo_expert

Scenario = box placement (4 sessions) x brick pose (5 each) sampled like the sim training data but with a different
seed (1000). Episode starts at the rest pose; up to 450 frames (15 s) at 30 Hz.
Success (physics): brick resting inside the box footprint at the end. Also reported: grasp+lift (brick >= 30 mm up).
"""

import argparse
import base64
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from isaac_client import IsaacSim  # noqa: E402
from policy_isaac import to_norm, to_rad  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "logs" / "eval_scenarios.json"
TZ, BRICK, BOX = -0.012, [0.0158, 0.0318, 0.0114], dict(L=0.10, W=0.075, H=0.03)
CAM0 = {"up": dict(eye=[0.55, 0.0, 0.26], target=[0.0, 0.03, 0.08]),
        "side": dict(eye=[0.30, -0.42, 0.12], target=[0.02, 0.06, 0.07])}
TASK = "pink lego brick into the transparent box"
MAX_FRAMES = 450


def make_scenarios(n_box=4, per_box=5, seed=1000):
    real = json.loads((ROOT / "logs" / "real_task_stats.json").read_text())
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_box):
        k = int(rng.integers(0, 45))
        box_c = (np.array(real["release"][k][:2]) + rng.normal(0, 0.01, 2)).tolist()
        box_yaw = float(rng.normal(0, np.radians(10)))
        for _ in range(per_box):
            for _ in range(50):
                g = real["grasp"][int(rng.integers(0, 45))]
                bxy = np.array(g[:2]) + rng.normal(0, 0.01, 2)
                if np.linalg.norm(bxy - box_c) > 0.10 and 0.06 < np.linalg.norm(bxy) < 0.26:
                    break
            out.append(dict(box=box_c, box_yaw=box_yaw, brick=bxy.tolist(), brick_yaw=float(np.radians(g[3] + rng.normal(0, 10))),
                            cams={n: dict(eye=(np.array(c["eye"]) + rng.normal(0, 0.02, 3)).tolist(),
                                          target=(np.array(c["target"]) + rng.normal(0, 0.01, 3)).tolist()) for n, c in CAM0.items()},
                            shade=float(rng.uniform(0.78, 0.92)), rest_noise=rng.normal(0, 1.0, 5).tolist()))
    SCEN.write_text(json.dumps(dict(seed=seed, scenarios=out, rest=np.mean(np.array(real["rest"])[:45], 0).tolist()), indent=1))
    print("wrote", len(out), "scenarios")


def session_scene(sc, rest_n):
    return dict(table_z=TZ, friction=1.0, place=[5.0, 5.0], q0=to_rad(np.r_[rest_n[:5], 1.0]).tolist(),
                bin=dict(center=sc["box"], yaw=sc["box_yaw"], L=BOX["L"], W=BOX["W"], H=BOX["H"], color=[0.82, 0.88, 0.93]),
                cube=dict(center=[0.15, 0.05, TZ + BRICK[2] / 2], yaw=0.0, size=0.02, dims=BRICK, color=[1.0, 0.35, 0.55], mass=0.003),
                obstacles=[dict(name="backdrop_x", center=[-0.45, 0.0, 0.3], dims=[0.02, 2.0, 0.8], color=[0.92, 0.92, 0.92]),
                           dict(name="backdrop_y", center=[0.0, 0.55, 0.3], dims=[2.0, 0.02, 0.8], color=[0.92, 0.92, 0.92])],
                cameras=CAM0, robot_color=[0.05, 0.05, 0.05], table_color=[0.85, 0.85, 0.85],
                cam_eye=[0.45, 0.35, 0.35], cam_target=[0.1, 0.0, 0.05],
                gains=dict(arm_kp=300.0, arm_kd=15.0, grip_kp=8.0, grip_kd=0.5))


def judge(cube, sc, z0, max_rise):
    b = np.array(cube[0])
    y = sc["box_yaw"]
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    lb = R.T @ (b[:2] - np.array(sc["box"]))
    inside = abs(lb[0]) < BOX["L"] / 2 - 0.004 and abs(lb[1]) < BOX["W"] / 2 - 0.004 and b[2] < TZ + 0.004 + BRICK[2]
    return dict(success=bool(inside), lifted=bool(max_rise > 0.03), max_rise_mm=round(max_rise * 1000, 1),
                brick_final=np.round(b, 4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-scenarios", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--video-every", type=int, default=5, help="keep viewport frames for every Nth scenario")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.make_scenarios:
        make_scenarios()
        return
    S = json.loads(SCEN.read_text())
    rest_n = np.array(S["rest"])
    pol = None
    if a.policy:
        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(a.policy)
        cfg.device = "cuda"
        pol = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to("cuda").eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy)
    if a.expert:
        import sim_demo_gen as G
    results, sim, cur_box = [], None, None
    t0 = time.perf_counter()
    for i, sc in enumerate(S["scenarios"][: a.limit]):
        if cur_box != sc["box"]:  # new Isaac session per box placement
            if sim is not None:
                sim.close()
            fdir = ROOT / "isaac" / f"frames_eval_{a.tag}_{i:02d}"
            shutil.rmtree(fdir, ignore_errors=True)
            sim = IsaacSim(port=5602, frames_dir=fdir, log=ROOT / "logs" / f"isaac_eval_{a.tag}.log")
            sim.call(cmd="setup", scene=session_scene(sc, rest_n))
            cur_box = sc["box"]
            if a.expert:
                ex = G.Expert(np.array(sc["box"]), sc["box_yaw"])
        te = time.perf_counter()
        if a.expert:
            class _R:  # replay this scenario's sampled values through the generator's episode function
                def __init__(self, sc):
                    self.sc = sc

                def integers(self, lo, hi):
                    return 0

                def normal(self, m, s, n=None):
                    return np.zeros(n) if n else 0.0

                def uniform(self, lo, hi):
                    return self.sc["shade"]
            G.OUT = ROOT / "data" / "eval_tmp"
            real = dict(grasp=[[*sc["brick"], float(np.degrees(sc["brick_yaw"])), 0.0]], release=[[*sc["box"], 0.0]])
            G.CAM0 = sc["cams"]
            ok, meta, rec = G.run_episode(sim, ex, _R(sc), real, i, np.array(sc["box"]), sc["box_yaw"], rest_n + np.r_[sc["rest_noise"], 0])
            st = sim.call(cmd="state")
            r = judge(st["cube"], sc, TZ + BRICK[2] / 2, 0.0)
            # the generator checks the lift from physics (>= 30 mm) before transporting
            lifted = bool(ok) or meta.get("fail") in ("no transport plan", "brick not in box at the end")
            r.update(success=bool(ok), lifted=lifted, max_rise_mm=None, fail=meta.get("fail"), frames=meta.get("frames"))
            if rec is not None:
                shutil.rmtree(rec.dir, ignore_errors=True)
        else:
            rr = sim.call(cmd="reset", q0=to_rad(np.r_[rest_n[:5] + np.array(sc["rest_noise"]), 1.0]).tolist(),
                          cube=dict(center=[*sc["brick"], TZ + BRICK[2] / 2], yaw=sc["brick_yaw"]),
                          cameras=sc["cams"], table_color=[sc["shade"]] * 3)
            z0, max_rise = rr["cube"][0][2], 0.0
            pol.reset()
            for k in range(MAX_FRAMES):
                rd = sim.call(cmd="render")
                im = {n: np.frombuffer(base64.b64decode(v["b64"]), np.uint8).reshape(v["shape"]) for n, v in rd["images"].items()}
                obs = {"observation.images.up": im["up"], "observation.images.side": im["side"],
                       "observation.state": to_norm(np.array(rd["q"], np.float32))}
                act = predict_action(obs, pol, torch.device("cuda"), pre, post, use_amp=False, task=TASK,
                                     robot_type="so100_follower").cpu().numpy().reshape(-1)
                ex_ = sim.call(cmd="exec", q=[to_rad(act).tolist()], dt=1 / 60, settle=0.0)
                max_rise = max(max_rise, ex_["cube"][0][2] - z0)
                if k % 50 == 49:  # per-step timing goes to the log file (no pipes)
                    print(f"   step {k + 1}: {(time.perf_counter() - te) / (k + 1) * 1000:.0f} ms/step, "
                          f"max_rise {max_rise * 1000:.0f} mm", flush=True)
            st = sim.call(cmd="wait", seconds=0.5)
            r = judge(st["cube"], sc, z0, max_rise)
        fe = sim.call(cmd="state")["frames"]
        r.update(scenario=i, seconds=round(time.perf_counter() - te, 1), frame_end=fe,
                 frames_dir=f"frames_eval_{a.tag}_{i - i % 5:02d}")
        results.append(r)
        print(f"[{a.tag}] scenario {i:02d}: {'SUCCESS' if r['success'] else 'fail   '} lifted={r['lifted']} "
              f"max_rise={r['max_rise_mm']}mm ({r['seconds']}s)", flush=True)
        (ROOT / "logs" / f"eval_closedloop_{a.tag}.json").write_text(json.dumps(results, indent=1))
    sim.close()
    n = len(results)
    print(f"[{a.tag}] SUCCESS {sum(r['success'] for r in results)}/{n}, LIFTED {sum(r['lifted'] for r in results)}/{n}, "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
