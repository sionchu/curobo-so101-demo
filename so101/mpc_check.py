"""Headless check: SO-101 MPC tracks a goal offset 8 cm in Y from the start pose."""

import sys
from pathlib import Path

import os
import torch
RUN_IK = os.environ.get('RUN_IK', '1') == '1'

sys.path.insert(0, str(Path(__file__).parent))
from so101_demo import ROBOT, SCENE, _criteria  # noqa: E402

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState

mpc = ModelPredictiveControl(
    ModelPredictiveControlCfg.create(
        robot=ROBOT, scene_model=SCENE, use_cuda_graph=True,
        optimization_dt=0.03, interpolation_steps=4,
    )
)
mpc.update_tool_pose_criteria(_criteria(mpc.tool_frames))
s = JointState.from_position(mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names)
s.velocity = torch.zeros_like(s.position)
s.acceleration = torch.zeros_like(s.position)
mpc.setup(s)
goal = {k: v.clone() for k, v in mpc.compute_kinematics(s).tool_poses.to_dict().items()}
tf = mpc.tool_frames[0]
goal[tf].position[..., 0] -= 0.12
goal[tf].position[..., 2] -= 0.05
goal_pos = goal[tf].position.clone()
ok = mpc.update_goal_tool_poses(GoalToolPose.from_poses(goal, ordered_tool_frames=mpc.tool_frames, num_goalset=1), run_ik=RUN_IK); print("goal update ok:", ok, "run_ik", RUN_IK)
for i in range(150):
    r = mpc.optimize_action_sequence(s)
    if r.action_sequence is not None and r.action_sequence.position.shape[1] > 0:
        s = JointState.from_position(r.action_sequence.position[:, -1, :].clone(), joint_names=mpc.joint_names)
        s.velocity = r.action_sequence.velocity[:, -1, :]
        s.acceleration = r.action_sequence.acceleration[:, -1, :]
    if (i + 1) % 50 == 0:
        ee = mpc.compute_kinematics(s).tool_poses.to_dict()[tf].position
        aseq = None if r.action_sequence is None else tuple(r.action_sequence.position.shape)
        print(f"step {i+1}: reported err {r.position_error.item():.4f}  "
              f"goal {goal_pos.cpu().numpy().round(3)}  ee {ee.cpu().numpy().round(3)}  aseq {aseq}")
print("final q:", s.position.cpu().numpy().round(3))


