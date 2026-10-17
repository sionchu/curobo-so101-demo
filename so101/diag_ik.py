"""Feasibility of recorded SO-101 states under different checks (graph-planner feasibility)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from replay import ARM, LIMIT_MARGIN, LIMITS, load_episode, to_radians  # noqa: E402

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

state, _, _ = load_episode(0)
q = to_radians(state)[:, :5]
lo, hi = LIMITS[:5, 0] + LIMIT_MARGIN, LIMITS[:5, 1] - LIMIT_MARGIN
frames = [0, 100, 150, 225, 300]
qs = np.clip(q[frames], lo, hi)
print("clipped states:\n", np.round(qs, 3))
tbl = lambda z: {"cuboid": {"table": {"dims": [1.0, 1.0, 0.04], "pose": [0, 0, z - 0.02, 1, 0, 0, 0]}}}
for name, scene, selfc in [
    ("no table, no self", None, False),
    ("no table, self", None, True),
    ("table z=0, no self", tbl(0.0), False),
    ("table z=-0.03, no self", tbl(-0.03), False),
]:
    pl = MotionPlanner(MotionPlannerCfg.create(robot=ARM, scene_model=scene, self_collision_check=selfc))
    m = pl.graph_planner.check_samples_feasibility(torch.tensor(qs, device="cuda", dtype=torch.float32))
    print(f"{name:24s} feasible per frame {dict(zip(frames, m.cpu().numpy().tolist()))}", flush=True)
