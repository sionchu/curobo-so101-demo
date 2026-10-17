"""Modular pick & place on the real-depth SO-101 scene:
perception (model-based 6D pose fallback) -> cuRobo planning -> harness verify -> retry/recover.

    python so101/harness_pickplace.py --port 8089                 # live Viser demo
    python so101/harness_pickplace.py --port 8090 --record videos/harness_pickplace.mp4
    python so101/harness_pickplace.py --headless                  # numbers + log only

What is real vs simulated:
- real: depth/RGB frames, the reconstructed ESDF, perception residuals, cuRobo planning times/clearances.
- simulated (kinematic, no physics): execution of trajectories, the gripper closing, the object
  following the gripper while attached, and the object settling on the table after release.
- the "high-level policy" is a local rule-based stub (no LLM/VLA).
- failure injection: attempt 1 of perceive() adds 25 mm of synthetic position noise to the cube pose.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import depth_planning as dp  # noqa: E402
from perception import perceive, summary  # noqa: E402
from replay import FULL, JOINTS  # noqa: E402

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria
from curobo.viewer import ViserVisualizer

HERE = Path(__file__).resolve().parent
PICK_CFG = HERE / "so101_pick.yml"
CONTACT_LINKS = ["gripper_link", "moving_jaw_so101_v1_link", "attached_object"]
PRE_OFFSET = 0.08  # m above grasp / place poses
GRIP_OPEN, GRIP_CLOSED = 1.2, 0.25  # rad, visual only (planning uses the arm with gripper locked)
# tilt tolerance: a vertical grasp is outside the SO-101 workspace at the cube (0.37 m out); IK gives ~30-35 deg
TOL = dict(pos_mm=8.0, tilt_deg=40.0, yaw_deg=20.0, place_xy_mm=25.0)
ORIENT_W = 0.05  # orientation weight relative to position


def make_pick_cfg():
    """so101_arm.yml + an 'attached_object' link on the tool frame (sphere slots for a held object)."""
    cfg = yaml.safe_load((HERE / "so101_arm.yml").read_text())
    k = cfg.get("robot_cfg", cfg)["kinematics"]
    k["extra_links"] = {"attached_object": {"fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_name": "attach_joint",
                                            "joint_type": "FIXED", "link_name": "attached_object",
                                            "parent_link_name": "gripper_frame_link"}}
    k["extra_collision_spheres"] = {"attached_object": 16}
    k["collision_link_names"] = list(k["collision_link_names"]) + ["attached_object"]
    k["grasp_contact_link_names"] = CONTACT_LINKS
    k["self_collision_buffer"]["attached_object"] = 0.0
    ign = k["self_collision_ignore"]
    ign["attached_object"] = ["gripper_link", "moving_jaw_so101_v1_link", "wrist_link", "gripper_frame_link"]
    for l in ("gripper_link", "moving_jaw_so101_v1_link", "wrist_link"):
        ign.setdefault(l, []).append("attached_object")
    PICK_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def down_quat(yaw: float) -> np.ndarray:
    """wxyz of a gripper pointing straight down (tool z = -world z), rotated by yaw about world z."""
    q = (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi)).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


class World:
    """Kinematic simulation state: the 'true' object pose (from clean perception) and what is held."""

    def __init__(self, cube_center, cube_yaw, size):
        self.cube_c, self.cube_yaw, self.size = np.array(cube_center, float), float(cube_yaw), float(size)
        self.attached_T = None  # tool->object transform while held


class Harness:
    """Rule-based executor (stub policy) over primitives; every step is timed and logged."""

    def __init__(self, ui=None):
        self.log, self.ui, self.t0 = [], ui, time.perf_counter()

    def step(self, name, fn, *a, **kw):
        t = time.perf_counter()
        ok, info = fn(*a, **kw)
        dt = (time.perf_counter() - t) * 1000
        rec = dict(step=name, ok=bool(ok), ms=round(dt, 1), **info)
        self.log.append(rec)
        line = f"[{len(self.log):02d}] {name:<14s} {'OK ' if ok else 'FAIL'} {dt:7.1f} ms  " + \
               ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("traj",))
        print(line, flush=True)
        if self.ui:
            self.ui(self.log)
        return ok, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-fault", action="store_true", help="skip the injected perception error")
    a = ap.parse_args()

    make_pick_cfg()
    # ---- clean perception = the kinematic sim's ground truth for the object ----
    t = time.perf_counter()
    truth = perceive()
    t_truth = time.perf_counter() - t
    print("perception (clean, frames 0-171):", json.dumps(summary(truth)), f"({t_truth:.1f}s)", flush=True)
    c = truth["cube"]
    world = World(c["center"], c["yaw"], c["size"])
    place_xy = truth["place"]["center"][:2]
    table_z = truth["table_z"]

    # ---- world model: depth map with robot and the target object removed ----
    half = np.array([c["size"] / 2 + 0.012] * 2 + [c["size"] / 2 + 0.012])
    mapper, grid, info, depth, rgb, q_rec, T_wc, T_tb = dp.build_map(exclude_boxes=[(c["center"], half)])
    print("map:", info, flush=True)
    planner = MotionPlanner(MotionPlannerCfg.create(robot=str(PICK_CFG), scene_model=Scene(voxel=[grid]),
                                                    optimizer_collision_activation_distance=0.01, max_goalset=4))
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    tf = planner.tool_frames[0]
    # position-priority: a straight-down 6D pose 30 cm out is not reachable for the 5-DOF SO-101
    # trajectory goals below come from IK-achieved (exactly reachable) poses, so track full 6D
    full_6d = {tf: ToolPoseCriteria()}
    planner.update_tool_pose_criteria(full_6d)

    # ---- viewer ----
    viz = server = None
    if not a.headless:
        viz = ViserVisualizer(content_path=ContentPath(robot_config_file=FULL), connect_port=a.port,
                              add_robot_to_scene=True, add_control_frames=False)
        server = viz._server

        @server.on_client_connect
        def _(client):
            client.camera.position = (0.05, -0.7, 0.45)
            client.camera.look_at = (0.22, -0.15, 0.02)

        vox = mapper.extract_occupied_voxels(surface_only=True, max_points=200_000)
        pts = vox.centers.cpu().numpy()
        h = np.clip((pts[:, 2] + 0.02) / 0.2, 0, 1)
        cols = (np.stack([0.3 + 0.7 * h, 0.5 + 0.2 * (1 - h), 1.0 - 0.8 * h], -1) * 255).astype(np.uint8)
        server.scene.add_point_cloud("/reconstruction", points=pts, colors=cols, point_size=dp.VOXEL)
        cube_h = server.scene.add_box("/cube_true", color=(220, 30, 30), dimensions=(world.size,) * 3)
        est_h = server.scene.add_box("/cube_perceived", color=(255, 170, 0), dimensions=(world.size,) * 3,
                                     wireframe=True)
        server.scene.add_box("/place_target", color=(20, 220, 60), opacity=0.9,
                             dimensions=(2 * TOL["place_xy_mm"] / 1000, 2 * TOL["place_xy_mm"] / 1000, 0.002),
                             position=(*place_xy, table_z + 0.012))
        server.gui.add_markdown(
            "**Modular pick & place on real depth** (CarolinePascal/so101_depth_raw)\n\n"
            "perceive -> plan (cuRobo) -> verify -> recover. Red = object (sim truth), orange wire = perceived, "
            "green pad = place region (yellow sticker in RGB).\n\nPolicy = local rule-based stub; execution is kinematic.")
        server.gui.add_image(rgb[0], label="real RGB", format="jpeg")
        log_md = server.gui.add_markdown("")

    grip = [GRIP_OPEN]
    q_now = [planner.default_joint_state.position.clone().view(1, -1)]

    def show(q5):
        if world.attached_T is not None:  # held object follows the tool (kinematic)
            To = tool_T(q5) @ world.attached_T
            world.cube_c = To[:3, 3]
            world.cube_yaw = float(np.arctan2(To[1, 0], To[0, 0]))
        if viz is None:
            return
        q6 = torch.cat([q5.view(1, -1), torch.tensor([[grip[0]]], device="cuda")], 1)
        viz.set_joint_state(JointState.from_position(q6, joint_names=JOINTS).squeeze(0))
        cube_h.position = tuple(world.cube_c)
        cube_h.wxyz = tuple(Rotation.from_euler("z", world.cube_yaw).as_quat()[[3, 0, 1, 2]])

    def tool_T(q5):
        p = planner.kinematics.compute_kinematics(JointState.from_position(q5.view(1, -1), joint_names=planner.joint_names)
                                                  ).tool_poses.get_link_pose(tf)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(p.quaternion.view(4).cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = p.position.view(3).cpu().numpy()
        return T

    rec = None
    frame_cb = [lambda: None]

    def play(traj, speed=1.0):
        q_now[0] = traj[-1:].clone()
        if viz is None:
            show(traj[-1])
            return
        for i in range(0, len(traj), max(1, int(speed))):
            show(traj[i])
            frame_cb[0]()
            if rec is None:
                time.sleep(0.025)

    def clearance_of(traj):
        """Min (ESDF - radius) over all robot spheres incl. gripper and any attached object."""
        best = 1e9
        for qq in traj[::3]:
            s = planner.kinematics.get_robot_as_spheres(qq.view(1, -1), filter_valid=True)[0]
            cc = torch.tensor(np.array([x.position for x in s]).reshape(-1, 3), device="cuda", dtype=torch.float32)
            rr = torch.tensor([x.radius for x in s], device="cuda")
            best = min(best, float((dp.esdf_query(grid, cc) - rr).min()))
        return round(best * 1000, 1)

    def plan_to(pos, quats, linear=False, disable=False):
        """Plan from q_now to a tool pose (goalset over yaw candidates). Returns traj or None, timing."""
        gs = len(quats)
        gp = GoalToolPose(tool_frames=[tf],
                          position=torch.tensor(np.tile(pos, (gs, 1)), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 3),
                          quaternion=torch.tensor(np.array(quats), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 4))
        if linear:
            planner.update_tool_pose_criteria({tf: ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 1],
                non_terminal_pose_axes_weight_factor=[1, 1, 0, 1, 1, 1],
                project_distance_to_goal=True)})  # straight-line motion along tool z
        if disable:
            planner.disable_link_collision(CONTACT_LINKS)
        start = JointState.from_position(q_now[0].clone(), joint_names=planner.joint_names)
        res = planner.plan_pose(gp, start, max_attempts=4)
        if disable:
            planner.enable_link_collision(CONTACT_LINKS)
        if linear:
            planner.update_tool_pose_criteria(full_6d)
        if res is None or not res.success.any():
            return None, None
        plan = res.get_interpolated_plan()
        traj = planner.kinematics.get_active_js(JointState.from_position(
            plan.position.reshape(-1, plan.position.shape[-1]), joint_names=plan.joint_names)).position
        return traj, float(res.total_time)

    # ------------------------------ primitives ------------------------------
    est = {}

    def p_perceive(attempt):
        # injected fault: 25 mm / 10 deg error toward the robot base (keeps the wrong target reachable)
        noise = None if (attempt > 1 or a.no_fault) else dict(xy_m=0.025, yaw_deg=10.0, dir_xy=-world.cube_c[:2])
        frames = (0, 172, 1) if attempt == 1 else (86, 172, 1)  # re-perceive on an independent half
        p = perceive(frames=frames, noise=noise, seed=attempt)
        est.update(center=p["cube"]["center"], yaw=p["cube"]["yaw"], size=p["cube"]["size"])
        if viz is not None:
            est_h.position = tuple(est["center"])
            est_h.wxyz = tuple(Rotation.from_euler("z", est["yaw"]).as_quat()[[3, 0, 1, 2]])
        err = np.linalg.norm(est["center"][:2] - world.cube_c[:2]) * 1000
        return True, dict(frames=f"{frames[0]}-{frames[1]-1}", injected_noise=bool(noise),
                          bin_icp_mm=round(p["bin"]["icp_trimmed_mm"], 1), pnp_px=round(p["rgb_cam"]["reproj_px"], 2),
                          est_vs_truth_xy_mm=round(float(err), 1))

    def grasp_quats(yaw):
        return [down_quat(yaw + k * np.pi / 2) for k in range(4)]

    def T_quat(T):
        return Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def ik_reach(positions, quats):
        """IK (goalset) for the best reachable tool pose; returns achieved 4x4 and position error (mm)."""
        gs = len(quats)
        gp = GoalToolPose(tool_frames=[tf],
                          position=torch.tensor(np.array(positions), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 3),
                          quaternion=torch.tensor(np.array(quats), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 4))
        # orientation-weight schedule: keep the most orientation that still reaches the position
        # rpy weights act on tool-frame axes: x/y = tilt, z = jaw rotation about the approach axis.
        for w in ([1, 1, 1], [0.3, 0.3, 1], [0.05, 0.05, 1], [0.05, 0.05, 0.05]):
            planner.update_tool_pose_criteria({tf: ToolPoseCriteria.track_position_and_orientation(rpy=w)})
            r = planner.ik_solver.solve_pose(gp, return_seeds=1,
                                             current_state=JointState.from_position(q_now[0].clone(), joint_names=planner.joint_names))
            sol = r.solution.reshape(-1, planner.action_dim)[0:1]
            T = tool_T(sol)
            perr = min(np.linalg.norm(T[:3, 3] - np.asarray(p)) for p in positions) * 1000
            if perr <= 5.0:
                break
        planner.update_tool_pose_criteria(full_6d)
        est["ik_w"] = w
        est["ik_q"] = sol
        return T, float(perr)

    def fix_yaw(q5, target_yaw, goal_pos):
        """Sweep wrist_roll (rotates about the approach axis) to align the jaws with the object faces."""
        d = torch.linspace(-np.pi / 2, np.pi / 2, 181, device="cuda")
        qs = q5.repeat(len(d), 1)
        qs[:, 4] = qs[:, 4] + d
        lo, hi = planner.kinematics.get_joint_limits().position
        ok = (qs[:, 4] >= lo[4]) & (qs[:, 4] <= hi[4])
        p = planner.kinematics.compute_kinematics(JointState.from_position(qs, joint_names=planner.joint_names)).tool_poses.get_link_pose(tf)
        R = torch.tensor(Rotation.from_quat(p.quaternion.reshape(-1, 4).cpu().numpy()[:, [1, 2, 3, 0]]).as_matrix())
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).numpy()
        err = np.abs((yaw - target_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        err[~ok.cpu().numpy()] = 9.0
        shift_mm = np.linalg.norm(p.position.reshape(-1, 3).cpu().numpy() - goal_pos, axis=1) * 1000
        k = int(np.argmin(np.degrees(err) + 2.0 * shift_mm))  # roll axis is ~8 mm off the tool point
        return qs[k : k + 1].clone(), float(np.degrees(err[k]))

    def plan_T(T, linear=False, disable=False):
        return plan_to(T[:3, 3], [T_quat(T)], linear=linear, disable=disable)

    def backoff(T, dist=PRE_OFFSET):
        """Same orientation, moved back along the tool approach axis (tool z)."""
        B = T.copy()
        B[:3, 3] = T[:3, 3] - T[:3, 2] * dist
        return B

    def tilt_of(T):
        return float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))

    def p_plan_approach():
        g = est["center"] + np.array([0, 0, -0.005])
        Tg, perr = ik_reach([g] * 4, grasp_quats(est["yaw"]))
        if perr > 5.0:
            return False, dict(reason=f"grasp pose unreachable (IK pos err {perr:.1f} mm)")
        qg, yaw_res = fix_yaw(est["ik_q"], est["yaw"], g)
        Tg = tool_T(qg)
        perr = float(np.linalg.norm(Tg[:3, 3] - g) * 1000)
        est["Tg"] = Tg
        traj, tp = plan_T(backoff(Tg))
        if traj is None:
            return False, dict(reason="no plan to pre-grasp")
        cl = clearance_of(traj)
        grip[0] = GRIP_OPEN
        play(traj)
        return True, dict(plan_s=round(tp, 3), waypoints=len(traj), min_clear_mm=cl,
                          grasp_tilt_deg=round(tilt_of(Tg), 1), ik_pos_err_mm=round(perr, 1), yaw_residual_deg=round(yaw_res, 1))

    def p_grasp():
        traj, tp = plan_T(est["Tg"], linear=True, disable=True)
        if traj is None:
            return False, dict(reason="no straight-line descend plan")
        cl = clearance_of(traj)
        play(traj)
        for v in np.linspace(GRIP_OPEN, GRIP_CLOSED, 12):
            grip[0] = v
            show(q_now[0][0])
            frame_cb[0]()
        return True, dict(plan_s=round(tp, 3), waypoints=len(traj), min_clear_mm=cl,
                          note="gripper/jaw collisions off for the final straight-line descend")

    def p_verify_grasp():
        Tt = tool_T(q_now[0])
        want = world.cube_c + np.array([0, 0, -0.005])
        pos_err = np.linalg.norm(Tt[:3, 3] - want) * 1000
        tilt = tilt_of(Tt)
        yaw_err = np.degrees((np.arctan2(Tt[1, 0], Tt[0, 0]) - world.cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        ok = pos_err < TOL["pos_mm"] and tilt < TOL["tilt_deg"] and abs(yaw_err) < TOL["yaw_deg"]
        if ok:
            world.attached_T = np.linalg.inv(Tt) @ cube_T()
            js = JointState.from_position(q_now[0].clone(), joint_names=planner.joint_names)
            planner.attachment_manager.attach(
                js, [Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[world.size] * 3)],
                link_name="attached_object", num_spheres=8, sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=Pose.from_numpy(world.cube_c.astype(np.float32),
                                                          Rotation.from_euler("z", world.cube_yaw).as_quat()[[3, 0, 1, 2]].astype(np.float32)))
        return ok, dict(tool_to_object_mm=round(float(pos_err), 1), tilt_deg=round(float(tilt), 1),
                        yaw_err_deg=round(float(yaw_err), 1), attached=bool(ok))

    def cube_T():
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", world.cube_yaw).as_matrix()
        T[:3, 3] = world.cube_c
        return T

    def p_recover():
        if world.attached_T is not None:  # failure while holding: put it back down where it was grasped
            traj, _ = plan_T(est["Tg"], linear=True, disable=True)
            if traj is not None:
                play(traj)
            world.attached_T = None
            planner.attachment_manager.detach(link_name="attached_object")
        grip[0] = GRIP_OPEN
        traj, tp = plan_T(backoff(tool_T(q_now[0])), linear=True, disable=True)
        if traj is None:
            return False, dict(reason="retreat plan failed")
        play(traj)
        return True, dict(action="open gripper + back out 8 cm along the approach axis", plan_s=round(tp, 3))

    def p_lift():
        traj, tp = plan_T(backoff(tool_T(q_now[0])), linear=True, disable=True)
        if traj is None:
            return False, dict(reason="lift plan failed")
        cl = clearance_of(traj)
        play(traj)
        return True, dict(plan_s=round(tp, 3), waypoints=len(traj), min_clear_mm=cl, holding=True)

    def p_transport():
        # tool goals that put the held object upright on the target pad (1 cm above the table)
        # the object may be set down at any yaw: rank yaws by how upright the (forced) tool tilt leaves it
        cands = []
        for yaw in np.deg2rad(np.arange(0, 360, 15)):
            P = np.eye(4)
            P[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            P[:3, 3] = (*place_xy, table_z + world.size / 2 + 0.01)
            Tk = P @ np.linalg.inv(world.attached_T)
            T_ach, perr = ik_reach([Tk[:3, 3]], [T_quat(Tk)])
            To = T_ach @ world.attached_T
            obj_tilt = float(np.degrees(np.arccos(np.clip(To[2, 2], -1, 1))))
            xy_err = float(np.linalg.norm(To[:2, 3] - place_xy) * 1000)
            score = obj_tilt + 0.5 * xy_err + (100 if perr > 5.0 else 0)
            cands.append((score, T_ach, perr, obj_tilt))
        cands.sort(key=lambda c: c[0])
        for tries, (_, Tp, perr, _) in enumerate(cands[:4], 1):  # replan with the next candidate on failure
            if perr > 5.0:
                break
            traj, tp = plan_T(backoff(Tp))
            if traj is not None:
                est["Tp"] = Tp
                cl = clearance_of(traj)
                play(traj)
                return True, dict(plan_s=round(tp, 3), waypoints=len(traj), min_clear_mm=cl,
                                  object_attached=True, place_candidates_tried=tries)
        return False, dict(reason="transport plan failed for the best place candidates")

    def p_place():
        traj, tp = plan_T(est["Tp"], linear=True, disable=True)
        if traj is None:
            return False, dict(reason="place descend failed")
        cl = clearance_of(traj)
        play(traj)
        To = tool_T(q_now[0]) @ world.attached_T
        obj_tilt = float(np.degrees(np.arccos(np.clip(To[2, 2], -1, 1))))
        for v in np.linspace(GRIP_CLOSED, GRIP_OPEN, 12):
            grip[0] = v
            show(q_now[0][0])
            frame_cb[0]()
        drop = world.cube_c[2] - (table_z + world.size / 2)
        world.attached_T = None
        world.cube_c[2] = table_z + world.size / 2  # kinematic settle on the table (no physics)
        planner.attachment_manager.detach(link_name="attached_object")
        show(q_now[0][0])
        est["obj_tilt"] = obj_tilt
        return True, dict(plan_s=round(tp, 3), waypoints=len(traj), min_clear_mm=cl,
                          release_height_mm=round(float(drop * 1000), 1), object_tilt_at_release_deg=round(obj_tilt, 1))

    def p_verify_place():
        d = np.linalg.norm(world.cube_c[:2] - place_xy) * 1000
        ok = d < TOL["place_xy_mm"] and est.get("obj_tilt", 90) < 15.0
        return ok, dict(object_to_target_xy_mm=round(float(d), 1), tol_mm=TOL["place_xy_mm"],
                        object_tilt_deg=round(est.get("obj_tilt", float("nan")), 1))

    def p_retreat():
        return p_recover()

    # ------------------------------ policy (rule-based stub) ------------------------------
    def ui(log):
        if viz is None:
            return
        rows = [f"`{i+1:02d} {r['step']:<13s} {'OK  ' if r['ok'] else 'FAIL'} {r['ms']:7.1f} ms`" for i, r in enumerate(log[-14:])]
        log_md.content = "**Step log**\n\n" + "\n\n".join(rows)

    H = Harness(ui)
    lines = ["Modular pick & place: perceive -> cuRobo plan -> verify -> recover"]
    if a.record and viz is not None:
        from depth_planning import _Recorder

        dvis = np.clip((depth[0] - 0.45) / 0.3, 0, 1)
        rec = _Recorder(server, a.record, rgb[0], dvis)
        rec.CENTER = np.array([0.22, -0.15, 0.02])
        rec.panels[1] = (rec.panels[1][0], "real depth")

        keys = dict(perceive=("est_vs_truth_xy_mm", "pose err {} mm"), verify_grasp=("tool_to_object_mm", "gripper-object {} mm"),
                    verify_place=("object_to_target_xy_mm", "object-target {} mm"), plan_approach=("min_clear_mm", "clear {} mm"),
                    transport=("min_clear_mm", "clear {} mm"), recover=("plan_s", "replan {} s"))

        def row(i, r):
            k = keys.get(r["step"])
            extra = ("  " + k[1].format(r[k[0]])) if k and k[0] in r else ""
            inj = "  [injected 25 mm noise]" if r.get("injected_noise") else ""
            return f"{i:02d} {r['step']}: {'OK' if r['ok'] else 'FAIL'}{extra}{inj}"

        def cb():
            rows = [row(i + 1, r) for i, r in enumerate(H.log)][-10:]
            rec.frame(lines + rows)

        frame_cb[0] = cb
        rec.orbit(45, lines + ["real depth scene -> cuRobo ESDF (object masked out of the map)"])
    elif viz is not None:
        print(f"Harness viewer at http://localhost:{a.port}", flush=True)
        from record_util import wait_client
        wait_client(server, timeout=24 * 3600)  # the run starts when the first browser connects

    show(q_now[0][0])
    done = False
    for attempt in range(1, 4):
        H.step("perceive", p_perceive, attempt)
        if not H.step("plan_approach", p_plan_approach)[0]:
            continue
        if not H.step("grasp", p_grasp)[0]:
            H.step("recover", p_recover)
            continue
        if not H.step("verify_grasp", p_verify_grasp)[0]:
            H.step("recover", p_recover)
            continue
        if not (H.step("lift", p_lift)[0] and H.step("transport", p_transport)[0] and H.step("place", p_place)[0]):
            H.step("recover", p_recover)
            continue
        ok = H.step("verify_place", p_verify_place)[0]
        H.step("retreat", p_retreat)
        done = ok
        break
    print("TASK", "SUCCESS" if done else "FAILED", f"after {attempt} attempt(s), wall {time.perf_counter() - H.t0:.1f}s", flush=True)
    (Path(__file__).resolve().parent.parent / "logs" / "harness_log.json").write_text(json.dumps(H.log, indent=1, default=float))
    if rec is not None:
        for _ in range(45):
            cb()
        rec.close()
        return
    while viz is not None:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
