"""Headless smoke test for the SO-101 cuRobo config: FK, self-collision, pose planning."""

from pathlib import Path

import torch
import yaml

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

HERE = Path(__file__).parent
FULL = HERE / "so101.yml"
ARM = HERE / "so101_arm.yml"

# Arm-only variant: gripper locked so planners only use the 5 arm joints.
cfg = yaml.safe_load(FULL.read_text())
(cfg.get("robot_cfg", cfg))["kinematics"]["lock_joints"] = {"gripper": 0.0}
ARM.write_text(yaml.safe_dump(cfg, sort_keys=False))

kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(FULL)))
print("joints:", kin.joint_names, "tool frames:", kin.tool_frames)
q0 = torch.zeros(1, kin.get_dof(), device="cuda")
st = kin.compute_kinematics(JointState.from_position(q0, joint_names=kin.joint_names))
p = st.tool_poses.get_link_pose(kin.tool_frames[0])
print("FK zero config gripper_frame_link pos:", p.position.cpu().numpy().round(3))

planner = MotionPlanner(MotionPlannerCfg.create(robot=str(ARM), scene_model="collision_table.yml"))
planner.warmup(enable_graph=True, num_warmup_iterations=3)
print("planner joints:", planner.joint_names)
start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)

# Goal = FK of a different, known-reachable config so the full 6D pose is feasible.
q_goal = torch.tensor([[0.6, -0.4, 0.5, 0.8, 0.3]], device="cuda")
g = planner.kinematics.compute_kinematics(
    JointState.from_position(q_goal, joint_names=planner.joint_names)
).tool_poses.to_dict()
print("goal pos:", g[planner.tool_frames[0]].position.cpu().numpy().round(3))
res = planner.plan_pose(GoalToolPose.from_poses(g, num_goalset=1), start, max_attempts=3)
print("plan success:", bool(res is not None and res.success.any()), "status:", getattr(res, "status", None))
if res is not None and res.success.any():
    print("waypoints:", res.get_interpolated_plan().position.shape)
