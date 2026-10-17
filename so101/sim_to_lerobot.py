"""Convert data/sim_raw episodes into a LeRobot v3 dataset with the same features as lerobot/svla_so101_pickplace,
then merge it with the real dataset for co-training.

    .venv-lerobot\\Scripts\\python.exe so101\\sim_to_lerobot.py

Outputs: data/so101_sim_pickplace (sim only), data/so101_real_plus_sim (real episodes 0-49 first, then sim).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

ROOT = Path(__file__).resolve().parent.parent
_tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "")
RAW = ROOT / "data" / ("sim_raw" + _tag)
SIM = ROOT / "data" / ("so101_sim_pickplace" + _tag)
MERGED = ROOT / "data" / ("so101_real_plus_sim" + _tag)
REAL = ROOT / "data" / "svla_so101_pickplace"
TASK = "pink lego brick into the transparent box"

if __name__ == "__main__":
    info = json.loads((REAL / "meta" / "info.json").read_text())
    feats = {k: v for k, v in info["features"].items()
             if k in ("action", "observation.state", "observation.images.up", "observation.images.side")}
    for k in ("observation.images.up", "observation.images.side"):
        feats[k] = dict(dtype="video", shape=feats[k]["shape"], names=feats[k]["names"])
    eps = sorted(p for p in RAW.glob("ep_*") if (p / "data.npz").exists())
    if "--merge-only" not in sys.argv:
        shutil.rmtree(SIM, ignore_errors=True)
        ds = LeRobotDataset.create("local/so101_sim_pickplace", fps=30, features=feats, root=SIM,
                                   robot_type=info["robot_type"], use_videos=True)
        for e in eps:
            d = np.load(e / "data.npz")
            for i in range(len(d["action"])):
                ds.add_frame({"observation.images.up": np.array(Image.open(e / "up" / f"{i:04d}.jpg")),
                              "observation.images.side": np.array(Image.open(e / "side" / f"{i:04d}.jpg")),
                              "observation.state": d["state"][i].astype(np.float32),
                              "action": d["action"][i].astype(np.float32), "task": TASK})
            ds.save_episode()
            print("saved", e.name, len(d["action"]), "frames", flush=True)
        ds.finalize()
    # metadata alignment for merging: same features/codec (AV1 yuv420p 480x640, same names/units); only the
    # encoder bookkeeping fields differ, so adopt the real dataset's feature dict verbatim
    si = json.loads((SIM / "meta" / "info.json").read_text())
    assert si["features"]["observation.images.up"]["info"]["video.codec"] == "av1"
    si["features"] = info["features"]
    (SIM / "meta" / "info.json").write_text(json.dumps(si, indent=4))
    sim = LeRobotDataset("local/so101_sim_pickplace", root=SIM, video_backend="pyav")
    real = LeRobotDataset("lerobot/svla_so101_pickplace", root=REAL, video_backend="pyav")
    print("sim episodes", sim.meta.total_episodes, "frames", sim.meta.total_frames)
    shutil.rmtree(MERGED, ignore_errors=True)
    m = merge_datasets([real, sim], output_repo_id="local/so101_real_plus_sim", output_dir=MERGED)
    print("merged episodes", m.meta.total_episodes, "frames", m.meta.total_frames)
