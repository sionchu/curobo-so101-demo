"""Offline evaluation of trained LeRobot policies on held-out episodes of lerobot/svla_so101_pickplace.

    .venv-lerobot\\Scripts\\python.exe so101\\eval_policies.py train\\act\\checkpoints\\last\\pretrained_model act
                                                          train\\smolvla\\checkpoints\\last\\pretrained_model smolvla

For every frame of episodes 45-49 (never seen in training) the policy predicts an action chunk from the recorded
observation; we compare chunk[0] (and the first 10 steps) with the recorded actions. Units are the dataset's
(LeRobot RANGE_M100_100 for arm joints, RANGE_0_100 for the gripper).
Baselines: hold (action = current state), last (action = previous recorded action), mean (training mean action).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = [45, 46, 47, 48, 49]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
H = 10


def load(path):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = "cuda"
    pol = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path)
    return pol, pre, post


def gripper_events(g, thr):
    """(close, re-open) frames: the gripper starts near-closed, opens to approach, closes on the object."""
    if not (g > thr).any():
        return None, None
    peak = int(np.argmax(np.convolve(g, np.ones(5) / 5, mode="same")))  # middle of the open plateau
    after = g[peak:] < thr  # the close is the first drop below threshold after the open peak
    if not after.any():
        return None, None
    close = int(peak + np.argmax(after))
    rest = g[close:] > thr
    rel = close + int(np.argmax(rest)) if rest.any() else None
    return close, rel


def run(path, name, ds, ep_index):
    pol, pre, post = load(path)
    preds, chunks = [], []
    t0 = time.perf_counter()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    with torch.inference_mode():
        for batch in loader:
            b = pre(dict(batch))
            ch = pol.predict_action_chunk(b)  # (B, chunk, 6) normalized
            B, C, D = ch.shape
            un = post(ch.reshape(B * C, D)).reshape(B, C, D)
            preds.append(un[:, 0].float().cpu().numpy())
            chunks.append(un[:, :H].float().cpu().numpy())
    dt = (time.perf_counter() - t0) / len(ds) * 1000
    return np.concatenate(preds), np.concatenate(chunks), dt


def metrics(pred, act):
    e = pred - act
    return dict(mae=np.abs(e).mean(0).round(2).tolist(), rmse=np.sqrt((e ** 2).mean(0)).round(2).tolist(),
                mae_arm_mean=round(float(np.abs(e[:, :5]).mean()), 2), mae_gripper=round(float(np.abs(e[:, 5]).mean()), 2))


if __name__ == "__main__":
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    ds = LeRobotDataset("lerobot/svla_so101_pickplace", root=ROOT / "data" / "svla_so101_pickplace",
                        episodes=HOLDOUT, video_backend="pyav")
    act = torch.stack([ds.hf_dataset[i]["action"] for i in range(len(ds.hf_dataset))]).numpy()
    st = torch.stack([ds.hf_dataset[i]["observation.state"] for i in range(len(ds.hf_dataset))]).numpy()
    ep = torch.stack([ds.hf_dataset[i]["episode_index"] for i in range(len(ds.hf_dataset))]).numpy().reshape(-1)
    stats = ds.meta.stats["action"]
    mean = np.asarray(stats["mean"]).reshape(-1)
    print(f"held-out frames: {len(act)} in episodes {HOLDOUT}", flush=True)

    last = act.copy()
    for e_ in HOLDOUT:  # previous recorded action within each episode
        idx = np.flatnonzero(ep == e_)
        last[idx[1:]] = act[idx[:-1]]
        last[idx[0]] = st[idx[0]]
    def timing(pred):
        errs = []
        for e_ in HOLDOUT:
            idx = np.flatnonzero(ep == e_)
            g = act[idx, 5]
            thr = (g.max() + g.min()) / 2
            (c0, _), (c1, _) = gripper_events(g, thr), gripper_events(pred[idx, 5], thr)
            if c0 is not None and c1 is not None:
                errs.append(abs(c1 - c0))
        return dict(close_timing_abs_err_frames=round(float(np.mean(errs)), 1) if errs else None, detected=len(errs))

    results = {"hold_state": dict(**metrics(st, act), gripper=timing(st)),
               "last_action": dict(**metrics(last, act), gripper=timing(last)),
               "mean_action (dataset stats)": metrics(np.tile(mean, (len(act), 1)), act)}
    allpred = {}
    for path, name in pairs:
        pred, chunk, ms = run(path, name, ds, ep)
        r = metrics(pred, act)
        # 10-step open-loop chunk error against the next 10 recorded actions (same episode only)
        errs = []
        for i in range(len(act)):
            j = np.arange(i, min(i + H, len(act)))
            j = j[ep[j] == ep[i]]
            errs.append(np.abs(chunk[i, :len(j)] - act[j]).mean())
        r["chunk10_mae"] = round(float(np.mean(errs)), 2)
        r["infer_ms_per_frame(batched)"] = round(ms, 2)
        # gripper close / release timing per episode
        dts = []
        for e_ in HOLDOUT:
            idx = np.flatnonzero(ep == e_)
            g, gp = act[idx, 5], pred[idx, 5]
            thr = (g.max() + g.min()) / 2
            (c0, r0), (c1, r1) = gripper_events(g, thr), gripper_events(gp, thr)
            dts.append(dict(ep=int(e_), close_rec=c0, close_pred=c1, release_rec=r0, release_pred=r1))
        ok = [abs(d["close_pred"] - d["close_rec"]) for d in dts if d["close_pred"] is not None and d["close_rec"] is not None]
        okr = [abs(d["release_pred"] - d["release_rec"]) for d in dts if d["release_pred"] is not None and d["release_rec"] is not None]
        r["gripper_close_timing_abs_err_frames"] = dict(mean=round(float(np.mean(ok)), 1) if ok else None, detected=len(ok),
                                                        release_mean=round(float(np.mean(okr)), 1) if okr else None,
                                                        episodes=dts)
        results[name] = r
        allpred[name] = pred
        print(name, json.dumps({k: v for k, v in r.items() if k != "gripper_close_timing_abs_err_frames"}), flush=True)

    (ROOT / "logs" / "policy_eval.json").write_text(json.dumps(results, indent=1, default=float))
    for e_ in (HOLDOUT[0], HOLDOUT[2]):
        idx = np.flatnonzero(ep == e_)
        fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for j, ax in enumerate(axs.ravel()):
            ax.plot(act[idx, j], "k-", lw=2, label="recorded action")
            for name, p in allpred.items():
                ax.plot(p[idx, j], lw=1.2, label=f"{name} (pred)")
            ax.set_title(JOINTS[j])
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"Held-out episode {e_}: predicted vs recorded actions (offline, from recorded observations)")
        fig.tight_layout()
        out = ROOT / "videos" / f"policy_offline_ep{e_}.png"
        fig.savefig(out, dpi=110)
        print("saved", out)
