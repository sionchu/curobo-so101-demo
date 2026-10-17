"""Modular pick & place with Isaac Sim PhysX in the loop, plus static and dynamic obstacles.

    python so101/harness_isaac.py                 # runs the task; frames -> isaac/frames_run, log -> logs/
    python so101/harness_isaac.py --no-fault      # without the injected perception error

Loop: perceive -> cuRobo plan -> execute in Isaac (position drives, PhysX contacts, real gripper closing)
      -> verify from physics state -> recover / replan.
- perceive #1: real-data perception (perception.py) + injected 25 mm error (synthetic fault).
- perceive #2+: "virtual camera" = cube pose read from the physics sim + Gaussian noise
  (sigma 1.6 mm = measured repeatability of the real-data estimator). Labelled as such.
- obstacles: a static wall (known at planning time) and a box that is spawned in the sim while the arm
  is transporting; the harness detects it by polling the sim's obstacle list (virtual sensor), checks the
  remaining path against it, and replans from the measured joint state.
- the policy is a local rule-based stub (no LLM/VLA).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import depth_planning as dp  # noqa: E402
from harness_pickplace import CONTACT_LINKS, PICK_CFG, PRE_OFFSET, down_quat, make_pick_cfg  # noqa: E402
from isaac_client import IsaacSim  # noqa: E402
from perception import perceive, summary  # noqa: E402

from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

ROOT = Path(__file__).resolve().parent.parent
GRIP_OPEN, GRIP_CLOSED = 0.5, -0.1  # rad; 0.5 = 4.1 cm jaw gap (measured from the URDF meshes); closed target is past contact so the drive squeezes
# cube centre expressed in the tool frame at grasp: tool point = fixed-jaw tip (inner face at x=0, measured from the
# URDF meshes); moving jaw closes from -x. Cube face 2 mm off the fixed jaw, fingertips 10 mm past the cube centre.
GRASP_OFFSET = None  # set from the cube size at runtime
TOL = dict(grasp_mm=20.0, lift_mm=40.0, slip_mm=15.0, place_xy_mm=25.0, place_tilt_deg=10.0)
DT = 0.025
FAR = [0.0, 0.0, -2.0, 1, 0, 0, 0]


def yaw_quat(yaw):
    return Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]


def box_dist(points, center, dims, yaw):
    """Signed distance from points (N,3) to an oriented box (positive outside)."""
    R = Rotation.from_euler("z", yaw).as_matrix()
    p = (points - np.asarray(center)) @ R
    q = np.abs(p) - np.asarray(dims) / 2
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return out + inside


class Run:
    def __init__(self, a):
        self.a = a
        self.log, self.captions = [], []
        make_pick_cfg()
        t = time.perf_counter()
        self.truth = perceive()
        c = self.truth["cube"]
        print("perception:", json.dumps(summary(self.truth)), f"({time.perf_counter() - t:.1f}s)", flush=True)
        self.size, self.tz = c["size"], self.truth["table_z"]
        self.place_xy = self.truth["place"]["center"][:2]
        b = self.truth["bin"]
        self.bin = dict(center=b["T"][:2, 3].tolist(), yaw=float(np.arctan2(b["T"][1, 0], b["T"][0, 0])),
                        L=b["size"][0], W=b["size"][1], H=b["size"][2])
        # static wall between the bin and the place target, perpendicular to the line joining them
        bc, pc = np.array(self.bin["center"]), np.array(self.place_xy)
        d = (pc - bc) / np.linalg.norm(pc - bc)
        wc = bc + d * (self.bin["L"] / 2 + 0.045)
        self.wall = dict(name="wall", center=[float(wc[0]), float(wc[1]), self.tz + 0.04], dims=[0.02, 0.12, 0.08],
                         yaw=float(np.arctan2(d[1], d[0])))
        half = np.array([self.size / 2 + 0.012] * 3)
        _, self.grid, info, *_ = dp.build_map(exclude_boxes=[(c["center"], half)])
        print("map:", info, flush=True)
        self.obst = {"wall": self.wall}
        scene = Scene(voxel=[self.grid], cuboid=[
            Cuboid(name="wall", pose=[*self.wall["center"], *yaw_quat(self.wall["yaw"])], dims=self.wall["dims"]),
            Cuboid(name="dyn", pose=FAR, dims=[0.05, 0.05, 0.12])])
        self.pl = MotionPlanner(MotionPlannerCfg.create(robot=str(PICK_CFG), scene_model=scene,
                                                        optimizer_collision_activation_distance=0.01, max_goalset=4))
        self.pl.warmup(enable_graph=True, num_warmup_iterations=3)
        self.tf = self.pl.tool_frames[0]
        self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria()})
        self.grip = GRIP_OPEN
        self.attached_T = None
        self.est = {}

        frames = ROOT / "isaac" / "frames_run"
        if frames.exists():
            for f in frames.glob("*.jpg"):
                f.unlink()
        t = time.perf_counter()
        self.sim = IsaacSim(frames_dir=frames)
        q0 = self.pl.default_joint_state.position.view(-1).cpu().numpy().tolist() + [GRIP_OPEN]
        r = self.sim.call(cmd="setup", scene=dict(
            table_z=self.tz, friction=1.0, bin=self.bin,
            cube=dict(center=c["center"].tolist(), yaw=float(c["yaw"]), size=self.size, mass=0.01),
            obstacles=[self.wall], place=self.place_xy.tolist(), q0=q0,
            gains=dict(arm_kp=300.0, arm_kd=15.0, grip_kp=8.0, grip_kd=0.5)))
        print(f"isaac up in {time.perf_counter() - t:.0f}s; cube settled at {np.round(r['cube'][0], 4)}", flush=True)
        self.q = torch.tensor([r["q"][:5]], device="cuda")
        self.frames = r.get("frames", 0)

    # ---------------- helpers ----------------
    def tool_T(self, q5):
        p = self.pl.kinematics.compute_kinematics(JointState.from_position(q5.view(1, -1), joint_names=self.pl.joint_names)
                                                  ).tool_poses.get_link_pose(self.tf)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(p.quaternion.view(4).cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = p.position.view(3).cpu().numpy()
        return T

    def cube_state(self):
        r = self.sim.call(cmd="state")
        p, o = np.array(r["cube"][0]), np.array(r["cube"][1])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(o[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = p
        return T, r

    def caption(self, extra=None):
        rows = []
        for i, r in enumerate(self.log[-9:], start=max(1, len(self.log) - 8)):
            rows.append(f"{i:02d} {r['step']}: {'OK' if r['ok'] else 'FAIL'}  {r.get('brief', '')}")
        self.captions.append(dict(frame=self.frames, lines=rows + ([extra] if extra else [])))

    def execute(self, traj, settle=0.2):
        q6 = [list(map(float, q.tolist())) + [self.grip] for q in traj]
        r = self.sim.call(cmd="exec", q=q6, dt=DT, settle=settle)
        self.q = torch.tensor([r["q"][:5]], device="cuda")
        self.frames = r["frames"]
        return r

    def set_grip(self, value, seconds=0.8):
        steps = int(seconds / DT)
        q5 = self.q.view(-1).cpu().numpy().tolist()
        q6 = [q5 + [float(v)] for v in np.linspace(self.grip, value, steps)]
        self.grip = value
        r = self.sim.call(cmd="exec", q=q6, dt=DT, settle=0.4)
        self.frames = r["frames"]
        return r

    def clearance(self, traj):
        """Min (distance - radius) per obstacle family over the trajectory (all robot spheres incl. held object)."""
        out = {"scene_esdf": 1e9}
        for k in self.obst:
            out[k] = 1e9
        for qq in traj[::2]:
            s = self.pl.kinematics.get_robot_as_spheres(qq.view(1, -1), filter_valid=True)[0]
            c = np.array([x.position for x in s]).reshape(-1, 3)
            r = np.array([x.radius for x in s])
            e = dp.esdf_query(self.grid, torch.tensor(c, device="cuda", dtype=torch.float32)).cpu().numpy()
            out["scene_esdf"] = min(out["scene_esdf"], float((e - r).min()))
            for k, ob in self.obst.items():
                out[k] = min(out[k], float((box_dist(c, ob["center"], ob["dims"], ob["yaw"]) - r).min()))
        return {k: round(v * 1000, 1) for k, v in out.items()}

    def plan(self, T, linear=False, disable=False):
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        gp = GoalToolPose.from_poses({self.tf: Pose.from_numpy(T[:3, 3].astype(np.float32), quat.astype(np.float32))},
                                     num_goalset=1)
        if linear:
            self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 1],
                non_terminal_pose_axes_weight_factor=[1, 1, 0, 1, 1, 1], project_distance_to_goal=True)})
        if disable:
            self.pl.disable_link_collision(CONTACT_LINKS)
        res = self.pl.plan_pose(gp, JointState.from_position(self.q.clone(), joint_names=self.pl.joint_names), max_attempts=4)
        if disable:
            self.pl.enable_link_collision(CONTACT_LINKS)
        if linear:
            self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria()})
        if res is None or not res.success.any():
            return None, None
        p = res.get_interpolated_plan()
        traj = self.pl.kinematics.get_active_js(JointState.from_position(
            p.position.reshape(-1, p.position.shape[-1]), joint_names=p.joint_names)).position
        return traj, float(res.total_time)

    def ik(self, positions, quats):
        gs = len(quats)
        gp = GoalToolPose(tool_frames=[self.tf],
                          position=torch.tensor(np.array(positions), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 3),
                          quaternion=torch.tensor(np.array(quats), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 4))
        best = None
        for w in ([1, 1, 1], [0.3, 0.3, 1], [0.05, 0.05, 1], [0.05, 0.05, 0.05]):
            self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria.track_position_and_orientation(rpy=w)})
            r = self.pl.ik_solver.solve_pose(gp, return_seeds=1, current_state=JointState.from_position(
                self.q.clone(), joint_names=self.pl.joint_names))
            sol = r.solution.reshape(-1, self.pl.action_dim)[0:1]
            T = self.tool_T(sol)
            perr = min(np.linalg.norm(T[:3, 3] - np.asarray(p)) for p in positions) * 1000
            best = (sol, T, perr)
            if perr <= 5.0:
                break
        self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria()})
        return best

    def fix_yaw(self, q5, target_yaw, goal_pos):
        d = torch.linspace(-np.pi / 2, np.pi / 2, 181, device="cuda")
        qs = q5.repeat(len(d), 1)
        qs[:, 4] = qs[:, 4] + d
        lo, hi = self.pl.kinematics.get_joint_limits().position
        ok = ((qs[:, 4] >= lo[4]) & (qs[:, 4] <= hi[4])).cpu().numpy()
        p = self.pl.kinematics.compute_kinematics(JointState.from_position(qs, joint_names=self.pl.joint_names)
                                                  ).tool_poses.get_link_pose(self.tf)
        R = Rotation.from_quat(p.quaternion.reshape(-1, 4).cpu().numpy()[:, [1, 2, 3, 0]]).as_matrix()
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        err = np.abs((yaw - target_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        err[~ok] = 9.0
        shift = np.linalg.norm(p.position.reshape(-1, 3).cpu().numpy() - goal_pos, axis=1) * 1000
        k = int(np.argmin(np.degrees(err) + 2.0 * shift))
        return qs[k:k + 1].clone(), float(np.degrees(err[k]))

    @staticmethod
    def backoff(T, dist=PRE_OFFSET):
        B = T.copy()
        B[:3, 3] = T[:3, 3] - T[:3, 2] * dist
        return B

    # ---------------- primitives ----------------
    def perceive(self, attempt):
        if attempt == 1:
            noise = None if self.a.no_fault else dict(xy_m=0.025, yaw_deg=10.0, dir_xy=-self.truth["cube"]["center"][:2])
            p = perceive(noise=noise, seed=1)
            c, yaw, src = p["cube"]["center"], p["cube"]["yaw"], "real RGB-D perception" + (" + injected 25 mm error" if noise else "")
        else:
            T, _ = self.cube_state()
            rng = np.random.default_rng(attempt)
            c = T[:3, 3] + np.r_[rng.normal(0, 0.0016, 2), 0.0]
            yaw, src = float(np.arctan2(T[1, 0], T[0, 0])), "virtual camera (sim pose + 1.6 mm noise)"
        Tt, _ = self.cube_state()
        self.est = dict(center=np.asarray(c, float), yaw=float(yaw))
        err = np.linalg.norm(self.est["center"][:2] - Tt[:2, 3]) * 1000
        return True, dict(source=src, est_vs_sim_xy_mm=round(float(err), 1), brief=f"pose err {err:.1f} mm ({src.split(' (')[0]})")

    def plan_approach(self):
        c = self.est["center"]
        off = np.array([-(self.size / 2 + 0.002), 0.0, -0.004])
        best = None
        for k in range(4):  # each jaw orientation: tool target so the cube sits between the jaws
            qd = down_quat(self.est["yaw"] + k * np.pi / 2)
            g = c - Rotation.from_quat(qd[[1, 2, 3, 0]]).as_matrix() @ off
            for _ in range(3):  # refine with the orientation actually reached (tilted for the 5-DOF arm)
                sol, Tg, _ = self.ik([g], [qd])
                qg, yres = self.fix_yaw(sol, self.est["yaw"], g)
                Tg = self.tool_T(qg)
                g = c - Tg[:3, :3] @ off
            e = float(np.linalg.norm(Tg[:3, 3] - g) * 1000)
            # clearance of the open moving jaw (tool-frame point, measured at 0.7 rad) to the bin's inner walls
            jaw = (Tg @ np.array([-0.061, 0.0, -0.01, 1.0]))[:3]  # outer face of the open moving jaw
            bc, by = np.array(self.bin["center"]), self.bin["yaw"]
            Rb = np.array([[np.cos(by), -np.sin(by)], [np.sin(by), np.cos(by)]])
            jb = Rb.T @ (jaw[:2] - bc)
            wall_clear = min(self.bin["L"] / 2 - 0.003 - abs(jb[0]), self.bin["W"] / 2 - 0.003 - abs(jb[1]))
            score = (-1000 * wall_clear) if e < 9.0 else (1000 + e)  # reachable first, then most wall clearance
            if best is None or score < best[0]:
                best = (score, Tg, yres, e, wall_clear)
        _, Tg, yres, perr, wall_clear = best
        if perr > 9.0:
            return False, dict(reason=f"grasp unreachable ({perr:.1f} mm)", brief="unreachable")
        self.Tg = Tg
        traj, tp = self.plan(self.backoff(self.Tg))
        if traj is None:
            return False, dict(reason="no plan to pre-grasp", brief="no plan")
        cl = self.clearance(traj)
        r = self.execute(traj)
        return True, dict(plan_s=round(tp, 3), clear_mm=cl, track_err_rad=round(r["track_err"], 4),
                          grasp_ik_err_mm=round(perr, 1), open_jaw_to_bin_wall_mm=round(wall_clear * 1000, 1),
                          contacts=r["contacts"], brief=f"plan {tp:.2f}s, wall clear {cl['wall']} mm")

    def grasp(self):
        traj, tp = self.plan(self.Tg, linear=True, disable=True)
        if traj is None:
            return False, dict(reason="no descend plan", brief="no plan")
        r1 = self.execute(traj)
        r2 = self.set_grip(GRIP_CLOSED)
        return True, dict(plan_s=round(tp, 3), contacts_descend=r1["contacts"], contacts_close=r2["contacts"],
                          brief=f"descend {tp:.2f}s + close gripper (physics)")

    def verify_grasp(self):
        r = self.sim.call(cmd="wait", seconds=0.15)  # fresh contact reading while the drive keeps squeezing
        self.frames = r["frames"]
        Tc, _ = self.cube_state()
        Tt = self.tool_T(self.q)
        off = np.array([-(self.size / 2 + 0.002), 0.0, -0.004])
        local = (np.linalg.inv(Tt) @ Tc)[:3, 3]
        d = np.linalg.norm(local - off) * 1000  # cube centre vs the expected between-the-jaws point
        jaw_links = {k.split("|")[0] for k in r["contacts"] if k.endswith("/World/cube")}
        grip_q = r["q"][5]
        blocked = grip_q > GRIP_CLOSED + 0.05  # the moving jaw was stopped by the cube, not by its target
        ok = d < TOL["grasp_mm"] and {"gripper_link", "moving_jaw_so101_v1_link"} <= jaw_links and blocked
        self.cube_before = Tc
        return ok, dict(cube_vs_jaw_center_mm=round(float(d), 1), jaw_contacts=sorted(jaw_links),
                        gripper_joint_rad=round(grip_q, 3), jaw_blocked_by_cube=bool(blocked),
                        brief=f"gripper-cube {d:.1f} mm (physics)")

    def recover(self):
        if self.attached_T is not None:
            self.pl.attachment_manager.detach(link_name="attached_object")
            self.attached_T = None
        self.set_grip(GRIP_OPEN, 0.6)
        traj, tp = self.plan(self.backoff(self.tool_T(self.q)), linear=True, disable=True)
        if traj is None:
            return False, dict(reason="retreat failed", brief="retreat failed")
        self.execute(traj)
        return True, dict(plan_s=round(tp, 3), brief="open + back out 8 cm")

    def lift(self):
        Tc0 = self.cube_before
        traj, tp = self.plan(self.backoff(self.tool_T(self.q)), linear=True, disable=True)
        if traj is None:
            return False, dict(reason="lift plan failed", brief="no plan")
        self.execute(traj)
        Tc, _ = self.cube_state()
        Tt = self.tool_T(self.q)
        rise = (Tc[2, 3] - Tc0[2, 3]) * 1000
        rel0 = np.linalg.inv(self.tool_T_at_grasp) @ Tc0 if hasattr(self, "tool_T_at_grasp") else None
        drift = np.linalg.norm((np.linalg.inv(Tt) @ Tc)[:3, 3] - rel0[:3, 3]) * 1000 if rel0 is not None else float("nan")
        ok = rise > TOL["lift_mm"] and drift < TOL["slip_mm"]
        if ok:  # collision model only: the held cube becomes part of the robot for planning (physics is untouched)
            self.attached_T = np.linalg.inv(Tt) @ Tc
            q = Rotation.from_matrix(Tc[:3, :3]).as_quat()[[3, 0, 1, 2]]
            self.pl.attachment_manager.attach(
                JointState.from_position(self.q.clone(), joint_names=self.pl.joint_names),
                [Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[self.size] * 3)], link_name="attached_object",
                num_spheres=8, sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=Pose.from_numpy(Tc[:3, 3].astype(np.float32), q.astype(np.float32)))
        return ok, dict(plan_s=round(tp, 3), cube_rise_mm=round(float(rise), 1), slip_mm=round(float(drift), 1),
                        brief=f"cube rose {rise:.0f} mm, slip {drift:.1f} mm (physics)")

    def transport(self):
        cands = []
        for yaw in np.deg2rad(np.arange(0, 360, 15)):
            P = np.eye(4)
            P[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            P[:3, 3] = (*self.place_xy, self.tz + self.size / 2 + 0.01)
            Tk = P @ np.linalg.inv(self.attached_T)
            sol, T_ach, perr = self.ik([Tk[:3, 3]], [Rotation.from_matrix(Tk[:3, :3]).as_quat()[[3, 0, 1, 2]]])
            To = T_ach @ self.attached_T
            tilt = float(np.degrees(np.arccos(np.clip(To[2, 2], -1, 1))))
            cands.append((tilt + 0.5 * np.linalg.norm(To[:2, 3] - self.place_xy) * 1000 + (100 if perr > 5 else 0), T_ach))
        cands.sort(key=lambda c: c[0])
        for tries, (_, Tp) in enumerate(cands[:4], 1):
            traj, tp = self.plan(self.backoff(Tp))
            if traj is None:
                continue
            self.Tp = Tp
            cl0 = self.clearance(traj)
            # execute the first 35 % of the path, then a box appears in the sim (scenario event)
            k = int(0.35 * len(traj))
            self.execute(traj[:k], settle=0.0)
            if not self.a.no_dynamic:
                # scenario event: a 10 cm box lands on the remaining path (>= 6 cm from the place target)
                ee = np.array([self.tool_T(q)[:3, 3] for q in traj[k:]])
                far = [p for p in ee if np.linalg.norm(p[:2] - self.place_xy) > 0.06]
                mid = far[len(far) // 2] if far else ee[len(ee) // 2]
                h = float(min(0.12, mid[2] - self.tz + 0.03))  # top 3 cm above the planned gripper path there
                self.sim.call(cmd="spawn", name="dyn", center=[float(mid[0]), float(mid[1]), self.tz + h / 2],
                              dims=[0.05, 0.05, h], yaw=0.0)
            # virtual sensor: poll the sim's obstacle list; new ones go into cuRobo's world model
            _, st = self.cube_state()
            new = [n for n in st["obstacles"] if n not in self.obst]
            info = dict(plan_s=round(tp, 3), clear_planned_mm=cl0, place_candidates_tried=tries)
            if new:
                ob = st["obstacles"]["dyn"]
                self.obst["dyn"] = dict(name="dyn", **ob)
                # the planner's box was pre-allocated with the max size; scale it to the sensed size
                top = ob["center"][2] + ob["dims"][2] / 2  # align the 12 cm planning box's top with the sensed top
                self.pl.scene_collision_checker.update_obstacle_pose(
                    "dyn", Pose.from_list([ob["center"][0], ob["center"][1], top - 0.06, *yaw_quat(ob["yaw"])]))
                if getattr(self.pl, "graph_planner", None) is not None:
                    self.pl.graph_planner.reset_buffer()
                rest_cl = self.clearance(traj[k:])
                info.update(detected=new, remaining_path_clear_mm=rest_cl)
                self.log.append(dict(step="obstacle_detected", ok=True, ms=0.0,
                                     brief=f"new box in path: clearance {rest_cl['dyn']} mm -> replan", **{"clear": rest_cl}))
                print(f"[--] obstacle_detected  remaining-path clearance to new box {rest_cl['dyn']} mm", flush=True)
                self.caption("obstacle appeared: replanning from measured state")
                if rest_cl["dyn"] < 5.0:
                    t = time.perf_counter()
                    info["start_state_clear_mm"] = self.clearance(self.q.clone())
                    traj2 = tp2 = None
                    for _, Tc_ in [(None, Tp)] + [(None, T_) for _, T_ in cands[:4] if T_ is not Tp]:
                        traj2, tp2 = self.plan(self.backoff(Tc_))
                        if traj2 is not None:
                            self.Tp = Tc_
                            break
                    if traj2 is None:  # escape upwards first, then retry the best candidate
                        up = self.tool_T(self.q)
                        up[2, 3] += 0.08
                        tr_up, _ = self.plan(up)
                        if tr_up is not None:
                            self.execute(tr_up, settle=0.0)
                            traj2, tp2 = self.plan(self.backoff(Tp))
                            info["escape_up"] = True
                    if traj2 is None:
                        return False, dict(reason="replan around new obstacle failed", brief="replan failed", **info)
                    cl2 = self.clearance(traj2)
                    r = self.execute(traj2)
                    info.update(replan_s=round(tp2, 3), replan_wall_ms=round((time.perf_counter() - t) * 1000, 1),
                                clear_replanned_mm=cl2, contacts=r["contacts"])
                    info["brief"] = f"replanned in {tp2:.2f}s, box clear {cl2['dyn']} mm, wall {cl2['wall']} mm"
                    return True, info
            r = self.execute(traj[k:])
            info.update(contacts=r["contacts"], brief=f"plan {tp:.2f}s, wall clear {cl0['wall']} mm")
            return True, info
        return False, dict(reason="transport plan failed", brief="no plan")

    def place(self):
        traj, tp = self.plan(self.Tp, linear=True, disable=True)
        if traj is None:
            return False, dict(reason="place descend failed", brief="no plan")
        r1 = self.execute(traj)
        self.pl.attachment_manager.detach(link_name="attached_object")
        self.attached_T = None
        r2 = self.set_grip(GRIP_OPEN, 0.6)
        r3 = self.sim.call(cmd="wait", seconds=1.0)  # let the cube settle under gravity
        self.frames = r3["frames"]
        return True, dict(plan_s=round(tp, 3), contacts=r1["contacts"], brief=f"descend {tp:.2f}s, open, settle 1 s")

    def verify_place(self):
        Tc, _ = self.cube_state()
        d = np.linalg.norm(Tc[:2, 3] - self.place_xy) * 1000
        tilt = float(np.degrees(np.arccos(np.clip(Tc[2, 2], -1, 1))))
        tilt = min(tilt, abs(90 - tilt), abs(180 - tilt))  # a cube resting on any face is upright
        rest = (Tc[2, 3] - (self.tz + self.size / 2)) * 1000
        ok = d < TOL["place_xy_mm"] and tilt < TOL["place_tilt_deg"] and abs(rest) < 5.0
        return ok, dict(cube_to_target_xy_mm=round(float(d), 1), cube_tilt_deg=round(tilt, 1),
                        height_above_rest_mm=round(float(rest), 1), brief=f"cube-target {d:.1f} mm, tilt {tilt:.1f} deg (physics)")

    def retreat(self):
        traj, tp = self.plan(self.backoff(self.tool_T(self.q)), linear=True, disable=True)
        if traj is None:
            return False, dict(reason="retreat failed", brief="no plan")
        self.execute(traj, settle=0.6)
        return True, dict(plan_s=round(tp, 3), brief="back out")

    # ---------------- policy (rule-based stub) ----------------
    def step(self, name, fn, *args):
        t = time.perf_counter()
        ok, info = fn(*args)
        rec = dict(step=name, ok=bool(ok), ms=round((time.perf_counter() - t) * 1000, 1), **info)
        self.log.append(rec)
        print(f"[{len(self.log):02d}] {name:<14s} {'OK ' if ok else 'FAIL'} {rec['ms']:8.1f} ms  " +
              ", ".join(f"{k}={v}" for k, v in info.items() if k != "brief"), flush=True)
        self.caption()
        return ok

    def run(self):
        t0 = time.perf_counter()
        done = False
        for attempt in range(1, 5):
            self.step("perceive", self.perceive, attempt)
            if not self.step("plan_approach", self.plan_approach):
                continue
            self.step("grasp", self.grasp)
            if not self.step("verify_grasp", self.verify_grasp):
                self.step("recover", self.recover)
                continue
            self.tool_T_at_grasp = self.tool_T(self.q)
            if not self.step("lift+verify", self.lift):
                self.step("recover", self.recover)
                continue
            if not (self.step("transport", self.transport) and self.step("place", self.place)):
                self.step("recover", self.recover)
                continue
            done = self.step("verify_place", self.verify_place)
            self.step("retreat", self.retreat)
            break
        wall = time.perf_counter() - t0
        _, st = self.cube_state()
        print("TASK", "SUCCESS" if done else "FAILED", f"after {attempt} attempt(s), wall {wall:.1f}s, "
              f"sim time {st['sim_time']:.1f}s", flush=True)
        self.caption("TASK " + ("SUCCESS" if done else "FAILED"))
        tag = "nofault" if self.a.no_fault else "fault"
        (ROOT / "logs" / f"isaac_harness_{tag}.json").write_text(json.dumps(
            dict(log=self.log, captions=self.captions, frames=self.frames, success=done), indent=1, default=float))
        self.sim.call(cmd="wait", seconds=1.0)
        self.sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fault", action="store_true")
    ap.add_argument("--no-dynamic", action="store_true")
    Run(ap.parse_args()).run()
