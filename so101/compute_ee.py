"""Gripper-tip (cuRobo tool frame) paths for the learning-story video, from recorded joint values.

    .venv\\Scripts\\python.exe so101\\compute_ee.py

Inputs: real training episodes 0-44 (dataset states), the 120 cuRobo-expert sim episodes (data/sim_raw),
and the closed-loop runs that logged joints (logs/traj_<tag>.json). Output: logs/learning_ee.json (metres, robot base).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))
from policy_isaac import to_rad  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(ROOT / "so101" / "so101_pick.yml")))
NJ = len(kin.joint_names)


def ee(q_rad):
    q = torch.as_tensor(np.asarray(q_rad, np.float32)[:, :NJ], device="cuda")
    p = kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names)).tool_poses
    return np.round(p.get_link_pose(kin.tool_frames[0]).position.reshape(-1, 3).cpu().numpy(), 4).tolist()


def norm_to_rad(states):
    return np.stack([to_rad(s) for s in np.asarray(states, np.float32)])


out = dict(real=[], sim=[], runs={})
df = pq.read_table(ROOT / "data" / "svla_so101_pickplace" / "data" / "chunk-000" / "file-000.parquet").to_pandas()
for e in range(45):
    st = np.stack(df[df["episode_index"] == e].sort_values("frame_index")["observation.state"].to_numpy())
    out["real"].append(ee(norm_to_rad(st)[::2]))
for d in sorted((ROOT / "data" / "sim_raw").glob("ep_*")):
    out["sim"].append(ee(norm_to_rad(np.load(d / "data.npz")["state"])[::2]))
for f in sorted((ROOT / "logs").glob("traj_*.json")):
    tag = f.stem[5:]
    out["runs"][tag] = {k: dict(ee=ee(v["q"]) if v["q"] else [], cube=v["cube"], box=v["box"], box_yaw=v["box_yaw"],
                                brick0=v["brick0"], success=v["success"])
                        for k, v in json.loads(f.read_text()).items()}
(ROOT / "logs" / "learning_ee.json").write_text(json.dumps(out))
print(f"real {len(out['real'])} eps, sim {len(out['sim'])} eps, runs {sorted(out['runs'])}")
