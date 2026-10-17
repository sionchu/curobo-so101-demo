"""SO-101 versions of cuRobo's interactive Viser demos (MPC + motion planning).

Adapted from curobo/examples/getting_started/{reactive_control,motion_planning}.py:
same loop, but with the SO-101 robot config, an SO-101-sized scene, and
relaxed orientation weights because the 5-DOF arm cannot reach every 6D pose.

    python so101/so101_demo.py mpc  --port 8081
    python so101/so101_demo.py plan --port 8082
"""

import argparse
import threading
import time
from pathlib import Path

import torch

from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria
from curobo.viewer import ViserVisualizer

HERE = Path(__file__).resolve().parent
ROBOT = str(HERE / "so101_arm.yml")
SCENE = str(HERE / "scene_so101.yml")

# Position-priority: full weight on xyz, small weight on orientation (5-DOF arm).
ORIENT_W = 0.05


def _criteria(tool_frames):
    return {
        k: ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[ORIENT_W] * 3
        )
        for k in tool_frames
    }


def _viewer(port: int) -> ViserVisualizer:
    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=ROBOT),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    @viz._server.on_client_connect
    def _(client):  # start the camera close to the 0.4 m-reach arm
        client.camera.position = (0.6, -0.5, 0.45)
        client.camera.look_at = (0.15, 0.0, 0.1)

    return viz


def _obstacle_poses(frames):
    return {k: Pose.from_numpy(frames[k].position, frames[k].wxyz) for k in frames.keys()}


def run_mpc(port: int):
    viz = _viewer(port)
    config = ModelPredictiveControlCfg.create(
        robot=ROBOT,
        scene_model=SCENE,
        use_cuda_graph=True,
        optimization_dt=0.03,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.02,
    )
    frames = viz.add_scene(config.scene_collision_cfg.scene_model, add_control_frames=True)
    old_obs = _obstacle_poses(frames)
    mpc = ModelPredictiveControl(config)
    mpc.update_tool_pose_criteria(_criteria(mpc.tool_frames))

    state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    mpc.setup(state)
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(
            mpc.compute_kinematics(state).tool_poses.to_dict(),
            ordered_tool_frames=mpc.tool_frames,
            num_goalset=1,
        ),
        run_ik=False,
    )
    print(f"SO-101 MPC running at http://localhost:{port}  (drag the gripper gizmo)")

    prev = None
    changed = False
    while True:
        obs = _obstacle_poses(frames)
        for k in obs:
            if obs[k] != old_obs[k]:
                mpc.scene_collision_checker.update_obstacle_pose(k, obs[k])
                changed = True
        old_obs = {k: v.clone() for k, v in obs.items()}

        targets = viz.get_control_frame_pose()
        if prev is None:
            prev = targets
        elif any(targets[k] != prev[k] for k in targets):
            prev = {k: v.clone() for k, v in targets.items()}
            changed = True

        if changed:
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(
                    {k.replace("target_", ""): v for k, v in targets.items()},
                    ordered_tool_frames=mpc.tool_frames,
                    num_goalset=1,
                ),
                run_ik=False,
            )
            changed = False

        res = mpc.optimize_action_sequence(state)
        if res.action_sequence is not None and res.action_sequence.position.shape[1] > 0:
            nxt = res.action_sequence.position[:, -1, :]
            state = JointState.from_position(nxt.clone(), joint_names=mpc.joint_names)
            state.velocity = res.action_sequence.velocity[:, -1, :]
            state.acceleration = res.action_sequence.acceleration[:, -1, :]
            viz.set_joint_state(state.squeeze(0))
        time.sleep(0.001)


def run_plan(port: int):
    viz = _viewer(port)
    config = MotionPlannerCfg.create(robot=ROBOT, scene_model=SCENE)
    planner = MotionPlanner(config)
    planner.update_tool_pose_criteria(_criteria(planner.tool_frames))
    frames = viz.add_scene(config.scene_collision_cfg.scene_model, add_control_frames=True)
    old_obs = _obstacle_poses(frames)
    current = planner.default_joint_state.clone().unsqueeze(0)
    print("Warming up planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    server = viz._server
    status = server.gui.add_markdown("Drag the gripper gizmo, then press **Move**.")
    busy = False

    def execute(traj):
        nonlocal current
        traj = traj.squeeze(0)
        for i in range(traj.position.shape[-2]):
            viz.set_joint_state(
                JointState.from_position(
                    traj.position[0, i, :].unsqueeze(0), joint_names=traj.joint_names
                ).squeeze(0)
            )
            time.sleep(0.02)
        current = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0), joint_names=traj.joint_names
        )

    def on_move(_):
        nonlocal busy
        if busy:
            return

        def work():
            nonlocal busy
            busy = True
            for k, p in _obstacle_poses(frames).items():
                if p != old_obs[k]:
                    planner.scene_collision_checker.update_obstacle_pose(k, p)
                    old_obs[k] = p.clone()
            res = planner.plan_pose(
                GoalToolPose.from_poses(viz.get_control_frame_pose(), num_goalset=1),
                planner.kinematics.get_active_js(current.clone()),
                use_implicit_goal=True,
                max_attempts=3,
            )
            if res is not None and res.success.any():
                interp = res.get_interpolated_plan()
                status.content = (
                    f"Plan OK: {interp.position.shape[-2]} waypoints, solve {res.total_time:.3f}s"
                )
                execute(interp)
            else:
                status.content = "Plan failed: target unreachable or in collision."
            busy = False

        threading.Thread(target=work, daemon=True).start()

    server.gui.add_button("Move", color="green").on_click(on_move)
    print(f"SO-101 motion planner running at http://localhost:{port}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mpc", "plan"])
    ap.add_argument("--port", type=int, default=8081)
    a = ap.parse_args()
    run_mpc(a.port) if a.mode == "mpc" else run_plan(a.port)
