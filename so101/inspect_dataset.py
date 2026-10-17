"""Print value ranges and per-episode video offsets for the SO-101 LeRobot dataset."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/svla_so101_pickplace")
data = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
state = np.stack(data["observation.state"].to_numpy())
print("frames:", len(data), "episodes:", data["episode_index"].nunique())
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
for i, n in enumerate(names):
    print(f"  {n:14s} min {state[:, i].min():8.2f}  max {state[:, i].max():8.2f}  mean {state[:, i].mean():8.2f}")
ep0 = state[data["episode_index"].to_numpy() == 0]
print("episode 0 first state:", ep0[0].round(2), "len", len(ep0))

eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
cols = [c for c in eps.columns if "episode_index" in c or "length" in c or "from_timestamp" in c or "to_timestamp" in c or c == "tasks"]
print(eps[cols].head(3).to_string())
