"""SO-101 planning against a scene reconstructed from real depth (CarolinePascal/so101_depth_raw).

Pipeline: raw depth (m) -> camera pose from table-plane fit + ICP robot base -> mask robot pixels
(cuRobo collision spheres at the recorded joints) -> cuRobo Mapper TSDF -> ESDF -> MotionPlanner / MPC.

    python so101/depth_planning.py plan --port 8084
    python so101/depth_planning.py mpc  --port 8085
    python so101/depth_planning.py test            # headless numbers only

Run scene_analyze.py and fit_robot.py first (they write logs/T_table_cam.npy, logs/T_table_base.npy).
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from depth_scene import CX, CY, FX, FY, H, W, backproject, load_frames, transform  # noqa: E402
from replay import ARM, FULL, JOINTS, to_radians  # noqa: E402
from so101_demo import _criteria  # noqa: E402

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.scene import Scene
from curobo.types import CameraObservation, ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

LOGS = Path(__file__).resolve().parent.parent / "logs"
VOXEL = 0.01
ESDF_VOXEL = 0.01
EXTENT = (1.0, 1.0, 0.5)
GRID_CENTER = (0.15, -0.1, 0.2)  # robot base frame; covers the reachable side (+x) incl. the bin
ROBOT_MASK_MARGIN = 0.03  # m added to each robot sphere when removing robot pixels
FRAME_STRIDE = 4


def poses():
    T_tc = np.load(LOGS / "T_table_cam.npy").astype(np.float64)
    T_tb = np.load(LOGS / "T_table_base.npy").astype(np.float64)
    T_wc = np.linalg.inv(T_tb) @ T_tc  # world = robot base_link frame
    return T_wc, T_tb


def robot_spheres(kin: Kinematics, q6: np.ndarray):
    s = kin.get_robot_as_spheres(torch.tensor(q6[None], device="cuda", dtype=torch.float32))[0]
    c = np.array([sp.position for sp in s], np.float32).reshape(-1, 3)
    r = np.array([sp.radius for sp in s], np.float32)
    return c, r


def mask_robot(depth: np.ndarray, T_wc: np.ndarray, centers, radii, exclude_boxes=()) -> tuple[np.ndarray, int]:
    pts, (v, u) = backproject(depth, 0.05, 2.0)
    pw = transform(T_wc, pts)
    d = np.linalg.norm(pw[:, None, :] - centers[None], axis=-1) - (radii[None] + ROBOT_MASK_MARGIN)
    hit = (d < 0).any(1)
    for c_box, half in exclude_boxes:  # e.g. the object to grasp: removed from the world model
        hit |= (np.abs(pw - np.asarray(c_box)) < np.asarray(half)).all(1)
    out = depth.copy()
    out[v[hit], u[hit]] = 0.0
    return out, int(hit.sum())


def build_map(exclude_boxes=()):
    depth, rgb, state = load_frames(stride=FRAME_STRIDE)
    T_wc, T_tb = poses()
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(FULL))
    q = to_radians(state)
    cfg = MapperCfg(
        voxel_size=VOXEL,
        esdf_voxel_size=ESDF_VOXEL,
        extent_meters_xyz=EXTENT,
        extent_esdf_meters_xyz=EXTENT,
        grid_center=torch.tensor(GRID_CENTER, dtype=torch.float32),
        truncation_distance=VOXEL * 4,
        depth_minimum_distance=0.2,
        depth_maximum_distance=0.75,  # table is ~0.55 m away; drops the floor around it
        minimum_tsdf_weight=2.0,
        num_cameras=1,
        image_height=H,
        image_width=W,
        block_size=8,
    )
    mapper = Mapper(cfg)
    filt = FilterDepth(image_shape=(H, W), depth_minimum_distance=0.2, depth_maximum_distance=0.75,
                       flying_pixel_threshold=0.5, bilateral_kernel_size=3)
    K = torch.tensor([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], device="cuda", dtype=torch.float32)
    cam_pose = Pose.from_matrix(torch.tensor(T_wc, device="cuda", dtype=torch.float32).unsqueeze(0))
    removed = []
    t0 = time.perf_counter()
    for i in range(len(depth)):
        c, r = robot_spheres(kin, q[i])
        dm, n = mask_robot(depth[i], T_wc, c, r, exclude_boxes)
        removed.append(n)
        d_t = torch.tensor(dm, device="cuda")
        d_t, _ = filt(d_t.unsqueeze(0))
        mapper.integrate(CameraObservation(
            depth_image=d_t, rgb_image=torch.tensor(rgb[i], device="cuda").unsqueeze(0),
            pose=cam_pose, intrinsics=K.unsqueeze(0)))
    torch.cuda.synchronize()
    t_int = time.perf_counter() - t0
    t0 = time.perf_counter()
    grid = mapper.compute_esdf()
    torch.cuda.synchronize()
    t_esdf = time.perf_counter() - t0
    info = dict(frames=len(depth), robot_px_removed=int(np.mean(removed)), t_integrate=t_int,
                t_esdf=t_esdf, esdf_shape=tuple(grid.feature_tensor.shape))
    return mapper, grid, info, depth, rgb, q, T_wc, T_tb


def esdf_query(grid, pts: torch.Tensor) -> torch.Tensor:
    """Trilinear ESDF lookup at world points (N,3); positive = free space (metres)."""
    e = grid.feature_tensor
    nx, ny, nz = e.shape
    origin = torch.as_tensor(grid.pose[:3], device=e.device, dtype=torch.float32)
    half = torch.tensor([(nx - 1), (ny - 1), (nz - 1)], device=e.device) * grid.voxel_size / 2
    g = ((pts.float() - origin) / half)[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
    return F.grid_sample(e.float()[None, None], g, align_corners=True, padding_mode="border").view(-1)


def clearance(kin_arm: Kinematics, grid, traj: torch.Tensor) -> float:
    """Min over a trajectory of (ESDF at sphere centre - sphere radius)."""
    best = 1e9
    for qq in traj:
        s = kin_arm.get_robot_as_spheres(qq.view(1, -1))[0]
        c = torch.tensor(np.array([sp.position for sp in s]).reshape(-1, 3), device="cuda", dtype=torch.float32)
        r = torch.tensor([sp.radius for sp in s], device="cuda")
        best = min(best, float((esdf_query(grid, c) - r).min()))
    return best


def make_planner(grid):
    pl = MotionPlanner(MotionPlannerCfg.create(robot=ARM, scene_model=Scene(voxel=[grid]),
                                               optimizer_collision_activation_distance=0.01))
    pl.update_tool_pose_criteria(_criteria(pl.tool_frames))
    pl.warmup(enable_graph=True, num_warmup_iterations=3)
    return pl


def bin_target(T_tb):
    """Goal 8 cm above the centre of the green bin (bin centre from scene_analyze, table frame)."""
    bin_c = np.array([-0.109, 0.101, 0.0, 1.0])
    b = (np.linalg.inv(T_tb) @ bin_c)[:3]
    return b, np.array([b[0], b[1], 0.08], np.float32)


def run_test():
    mapper, grid, info, *_ , T_wc, T_tb = build_map()
    print("map:", info)
    bin_w, goal = bin_target(T_tb)
    print("bin centre (base frame):", bin_w.round(3), "goal:", goal.round(3))
    print("ESDF at bin centre +3cm (m):", float(esdf_query(grid, torch.tensor([[*bin_w[:2], 0.03]], device="cuda"))))
    pl = make_planner(grid)
    start = pl.default_joint_state.clone().unsqueeze(0)
    tf = pl.tool_frames[0]
    gp = Pose(position=torch.tensor(goal[None], device="cuda"),
              quaternion=torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda"))  # gripper pointing down
    res = pl.plan_pose(GoalToolPose.from_poses({tf: gp}, num_goalset=1), start, max_attempts=5)
    ok = res is not None and bool(res.success.any())
    print("plan into the bin: success", ok, "solve %.3fs" % (res.total_time if res is not None else -1))
    if ok:
        plan = res.get_interpolated_plan()
        traj = pl.kinematics.get_active_js(JointState.from_position(
            plan.position.reshape(-1, plan.position.shape[-1]), joint_names=plan.joint_names)).position
        kin_arm = Kinematics(KinematicsCfg.from_robot_yaml_file(ARM))
        print("min clearance to reconstructed scene along plan: %.1f mm" % (clearance(kin_arm, grid, traj) * 1000))
        ee = kin_arm.compute_kinematics(JointState.from_position(traj, joint_names=pl.joint_names)).tool_poses.get_link_pose(tf).position.reshape(-1, 3)
        print("final EE %s (goal %s), max EE height %.3f" % (ee[-1].cpu().numpy().round(3), goal.round(3), ee[:, 2].max().item()))


class _Recorder:
    """Grab 800x720 renders from the connected browser and add real RGB/depth panels (1280x720 MP4)."""

    CENTER = np.array([0.15, -0.08, 0.02])

    def __init__(self, server, path, rgb0, dvis):
        from record_util import Mp4Writer, compose, look_at_wxyz, wait_client

        self._compose, self._look = compose, look_at_wxyz
        self.client = wait_client(server)
        self.out = Mp4Writer(path, fps=30)
        self.panels = [(rgb0, "real RGB (not registered to depth)"),
                       ((np.stack([dvis] * 3, -1) * 255).astype(np.uint8), "real depth")]
        self.angle = np.deg2rad(-110.0)

    def _pose(self):
        p = self.CENTER + np.array([0.7 * np.cos(self.angle), 0.7 * np.sin(self.angle), 0.42])
        return p, self._look(p, self.CENTER)

    def frame(self, lines, repeat=1):
        time.sleep(0.03)
        p, q = self._pose()
        img = self.client.get_render(720, 800, wxyz=q, position=p, fov=np.deg2rad(50))
        f = self._compose(img, self.panels, lines)
        for _ in range(repeat):
            self.out.write(f)

    def hold(self, n, lines):
        for _ in range(n):
            self.frame(lines)

    def orbit(self, n, lines, sweep_deg=60.0):
        for _ in range(n):
            self.angle += np.deg2rad(sweep_deg) / n
            self.frame(lines)

    def close(self):
        self.out.close()


def run_viewer(mode: str, port: int, record: str = None):
    mapper, grid, info, depth, rgb, q, T_wc, T_tb = build_map()
    print("map:", info, flush=True)
    viz = ViserVisualizer(content_path=ContentPath(robot_config_file=ARM), connect_port=port,
                          add_robot_to_scene=True, add_control_frames=True)
    server = viz._server

    @server.on_client_connect
    def _(client):
        client.camera.position = (-0.05, -0.75, 0.45)
        client.camera.look_at = (0.2, -0.12, 0.02)

    vox = mapper.extract_occupied_voxels(surface_only=True, max_points=200_000)
    pts = vox.centers.cpu().numpy()
    h = np.clip((pts[:, 2] + 0.02) / 0.2, 0, 1)
    cols = (np.stack([0.3 + 0.7 * h, 0.5 + 0.2 * (1 - h), 1.0 - 0.8 * h], -1) * 255).astype(np.uint8)
    server.scene.add_point_cloud("/reconstruction", points=pts, colors=cols, point_size=VOXEL)
    cam = Pose.from_matrix(torch.tensor(T_wc, dtype=torch.float32, device="cuda").unsqueeze(0))
    server.scene.add_frame("/camera", position=cam.position.cpu().numpy().ravel(), wxyz=cam.quaternion.cpu().numpy().ravel(), axes_length=0.06, axes_radius=0.003)

    bin_w, goal = bin_target(T_tb)
    dvis = np.clip((depth[0] - 0.45) / 0.3, 0, 1)
    server.gui.add_markdown(
        "**Real depth scene** (CarolinePascal/so101_depth_raw)\n\n"
        f"Mapper TSDF {VOXEL*100:.0f} cm -> ESDF {tuple(info['esdf_shape'])}, {info['frames']} frames, "
        f"robot pixels masked/frame ~{info['robot_px_removed']}\n\n"
        "Points coloured by height (not RGB: depth is not registered to colour)."
    )
    server.gui.add_image(rgb[0], label="RGB (real, unregistered)", format="jpeg")
    server.gui.add_image((np.stack([dvis] * 3, -1) * 255).astype(np.uint8), label="depth (real)", format="jpeg")
    status = server.gui.add_markdown("")
    target = viz._control_frames[viz._vis_frames[0]]

    if mode == "plan":
        pl = make_planner(grid)
        kin_arm = Kinematics(KinematicsCfg.from_robot_yaml_file(ARM))
        current = pl.default_joint_state.clone().unsqueeze(0)
        busy = False

        def go(goal_pose: Pose):
            nonlocal current, busy
            busy = True
            res = pl.plan_pose(GoalToolPose.from_poses({pl.tool_frames[0]: goal_pose}, num_goalset=1),
                               pl.kinematics.get_active_js(current.clone()), max_attempts=5)
            if res is not None and res.success.any():
                plan = res.get_interpolated_plan()
                traj = pl.kinematics.get_active_js(JointState.from_position(
                    plan.position.reshape(-1, plan.position.shape[-1]), joint_names=plan.joint_names)).position
                status.content = (f"Plan OK: {len(traj)} waypoints, solve {res.total_time:.3f}s, "
                                  f"min clearance to real scene {clearance(kin_arm, grid, traj)*1000:.0f} mm")
                for qq in traj:
                    viz.set_joint_state(JointState.from_position(qq.view(1, -1), joint_names=pl.joint_names).squeeze(0))
                    time.sleep(0.02)
                current = JointState.from_position(traj[-1:].clone(), joint_names=pl.joint_names)
            else:
                status.content = "Plan failed (target unreachable or inside reconstructed geometry)."
            busy = False

        def on_move(_):
            if not busy:
                threading.Thread(target=go, args=(Pose.from_numpy(target.position, target.wxyz),), daemon=True).start()

        def on_bin(_):
            if not busy:
                target.position = tuple(goal.tolist())
                target.wxyz = (0.0, 0.0, 1.0, 0.0)
                threading.Thread(target=go, args=(Pose.from_numpy(goal, np.array([0.0, 0.0, 1.0, 0.0])),), daemon=True).start()

        server.gui.add_button("Move to gizmo", color="green").on_click(on_move)
        server.gui.add_button("Reach over the bin (demo)", color="blue").on_click(on_bin)
        print(f"Depth-scene planner at http://localhost:{port}", flush=True)
        if record:
            rec = _Recorder(server, record, rgb[0], dvis)
            rec.orbit(90, ["Real depth scene (CarolinePascal/so101_depth_raw) -> cuRobo TSDF/ESDF",
                           "blue/grey points = reconstructed table + bin (height-coloured)",
                           "SO-101 placed by ICP against the depth point cloud"])
            goals = [("reach into the green bin", goal, (0.0, 0.0, 1.0, 0.0)),
                     ("reach to a free spot on the table", np.array([0.20, 0.10, 0.10], np.float32), (0.0, 0.0, 1.0, 0.0)),
                     ("back over the bin", np.array([goal[0] - 0.04, goal[1], 0.12], np.float32), (0.0, 0.0, 1.0, 0.0))]
            for name, gpos, gq in goals:
                target.position, target.wxyz = tuple(gpos.tolist()), gq
                t0 = time.perf_counter()
                res = pl.plan_pose(GoalToolPose.from_poses({pl.tool_frames[0]: Pose.from_numpy(gpos, np.array(gq))}, num_goalset=1),
                                   pl.kinematics.get_active_js(current.clone()), max_attempts=5)
                if res is None or not res.success.any():
                    rec.hold(30, [f"cuRobo motion planning: {name}", "plan FAILED for this target"])
                    continue
                plan = res.get_interpolated_plan()
                traj = pl.kinematics.get_active_js(JointState.from_position(
                    plan.position.reshape(-1, plan.position.shape[-1]), joint_names=plan.joint_names)).position
                lines = [f"cuRobo motion planning against real-depth ESDF: {name}",
                         f"solve {res.total_time:.3f} s, {len(traj)} waypoints, "
                         f"min clearance to scene {clearance(kin_arm, grid, traj)*1000:.0f} mm",
                         "(trajectory played at 0.5x speed)"]
                for qq in traj:
                    viz.set_joint_state(JointState.from_position(qq.view(1, -1), joint_names=pl.joint_names).squeeze(0))
                    rec.frame(lines, repeat=2)
                rec.hold(25, lines)
                current = JointState.from_position(traj[-1:].clone(), joint_names=pl.joint_names)
            rec.close()
            return
        while True:
            time.sleep(0.1)

    # MPC mode
    mpc = ModelPredictiveControl(ModelPredictiveControlCfg.create(
        robot=ARM, scene_model=Scene(voxel=[grid]), use_cuda_graph=True, optimization_dt=0.03,
        interpolation_steps=4, optimizer_collision_activation_distance=0.02))
    mpc.update_tool_pose_criteria(_criteria(mpc.tool_frames))
    s = JointState.from_position(mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names)
    s.velocity = torch.zeros_like(s.position)
    s.acceleration = torch.zeros_like(s.position)
    mpc.setup(s)
    last = None
    print(f"Depth-scene MPC at http://localhost:{port}", flush=True)
    if record:
        kin_arm = Kinematics(KinematicsCfg.from_robot_yaml_file(ARM))
        tf = mpc.tool_frames[0]
        ee0 = mpc.compute_kinematics(s).tool_poses.get_link_pose(tf).position.view(3).cpu().numpy().copy()
        down = (0.0, 0.0, 1.0, 0.0)
        # scripted target path (the gizmo is moved programmatically, as a user would drag it)
        keys = [ee0, goal, np.array([0.22, 0.12, 0.12]), np.array([0.12, -0.05, 0.25]), goal, ee0]
        rec = _Recorder(server, record, rgb[0], dvis)
        step = 0
        for k in range(len(keys) - 1):
            for j in range(120):  # 90 frames moving, 30 frames holding
                a = min(j / 90.0, 1.0)
                tgt = (1 - a) * keys[k] + a * keys[k + 1]
                target.position, target.wxyz = tuple(tgt.tolist()), down
                mpc.update_goal_tool_poses(GoalToolPose.from_poses(
                    {tf: Pose.from_numpy(tgt.astype(np.float32), np.array(down))},
                    ordered_tool_frames=mpc.tool_frames, num_goalset=1), run_ik=False)
                r = mpc.optimize_action_sequence(s)
                s = JointState.from_position(r.action_sequence.position[:, -1, :].clone(), joint_names=mpc.joint_names)
                s.velocity = r.action_sequence.velocity[:, -1, :]
                s.acceleration = r.action_sequence.acceleration[:, -1, :]
                viz.set_joint_state(s.squeeze(0))
                ee = kin_arm.compute_kinematics(s).tool_poses.get_link_pose(tf).position.view(3).cpu().numpy()
                rec.frame(["cuRobo MPC in the real-depth scene (target gizmo moved by script)",
                           f"gripper-to-target {np.linalg.norm(ee - tgt)*1000:.0f} mm   "
                           f"clearance to scene {clearance(kin_arm, grid, s.position)*1000:.0f} mm",
                           f"t = {step / 30:.1f} s (1 MPC step per video frame, dt 0.03 s)"])
                step += 1
        rec.close()
        return
    while True:
        tp = Pose.from_numpy(target.position, target.wxyz)
        if last is None or tp != last:
            mpc.update_goal_tool_poses(GoalToolPose.from_poses({mpc.tool_frames[0]: tp}, ordered_tool_frames=mpc.tool_frames, num_goalset=1), run_ik=False)
            last = tp.clone()
        r = mpc.optimize_action_sequence(s)
        if r.action_sequence is not None and r.action_sequence.position.shape[1] > 0:
            s = JointState.from_position(r.action_sequence.position[:, -1, :].clone(), joint_names=mpc.joint_names)
            s.velocity = r.action_sequence.velocity[:, -1, :]
            s.acceleration = r.action_sequence.acceleration[:, -1, :]
            viz.set_joint_state(s.squeeze(0))
        time.sleep(0.001)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "mpc", "test"])
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--record", type=str, default=None, help="write an MP4 (plan/mpc) and exit")
    a = ap.parse_args()
    run_test() if a.mode == "test" else run_viewer(a.mode, a.port, a.record)
