import sys, json, numpy as np, torch
sys.path.insert(0, "so101")
from replay import FULL, JOINTS, load_episode, to_radians
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState
from scipy.spatial.transform import Rotation
import pyarrow.parquet as pq
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 46
state, row, task = load_episode(ep)
data = pq.read_table("data/svla_so101_pickplace/data/chunk-000/file-000.parquet").to_pandas()
act = np.stack(data[data["episode_index"] == ep].sort_values("frame_index")["action"].to_numpy())
g = act[:, 5]; thr = (g.max() + g.min()) / 2
opened = np.flatnonzero(g > thr); close = int(opened[0] + np.argmax(g[opened[0]:] < thr))
q = to_radians(state)
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(FULL))
p = kin.compute_kinematics(JointState.from_position(torch.tensor(q, device="cuda"), joint_names=JOINTS)).tool_poses.get_link_pose(kin.tool_frames[0])
pos, quat = p.position.reshape(-1, 3).cpu().numpy(), p.quaternion.reshape(-1, 4).cpu().numpy()
R = Rotation.from_quat(quat[close][[1, 2, 3, 0]]).as_matrix()
start = max(0, close - 45)
out = dict(episode=ep, task=task, close_frame=close, start_frame=start, n_frames=len(q),
           q_start=q[start].tolist(), tool_at_close=pos[close].tolist(), tool_zaxis_at_close=R[:, 2].tolist(),
           tool_yaw_at_close_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
           min_tool_z=float(pos[:, 2].min()), actions=act.tolist(), states=state.tolist())
json.dump(out, open(f"logs/policy_sim_ep{ep}.json", "w"))
print({k: v for k, v in out.items() if k not in ("actions", "states")})
