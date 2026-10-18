"""Full chain in Isaac Sim PhysX: cuRobo approach -> SmolVLA grasp -> physics verify -> recover -> cuRobo transport
around a static wall and a box that appears mid-transport -> place -> verify -> retreat.

    .venv\\Scripts\\python.exe so101\\chain_isaac.py --grasp policy --tag chain_policy
    .venv\\Scripts\\python.exe so101\\chain_isaac.py --grasp curobo --tag chain_curobo

Same 20 seeded scenarios as eval_isaac_closedloop.py (logs/eval_scenarios.json), same cameras and scene.
- perceive: virtual sensor = brick pose from the physics sim + Gaussian xy noise (sigma 1.6 mm, the measured
  repeatability of the real-data estimator).
- grasp (policy mode): the sim co-trained SmolVLA runs closed-loop at 30 Hz from the rendered cameras, starting at the
  cuRobo pre-grasp pose, until the brick rises 20 mm or 150 steps (5 s). The policy runs in its own venv
  (policy_server.py) and is queried over localhost TCP.
- verify_grasp: cuRobo lifts 4 cm (gripper held at the last commanded value), then both jaws must touch the brick
  (PhysX contacts) and the brick must be >= 30 mm above its start.
- recovery: open + back up, re-perceive, re-plan the pre-grasp, policy once more; then the cuRobo scripted grasp.
- obstacles: a static wall between brick and box (where there is room: gap >= 9 cm and >= 15 mm from the
  unchecked rest->ready sweep) and a box
  spawned in the sim after 35 % of the transport, detected by polling the sim's obstacle list (virtual sensor).
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import sim_demo_gen as G  # noqa: E402
from eval_isaac_closedloop import BOX, BRICK, TZ, judge, session_scene  # noqa: E402
from harness_pickplace import PICK_CFG, down_quat, make_pick_cfg  # noqa: E402
from isaac_client import IsaacSim  # noqa: E402
from policy_isaac import to_norm, to_rad  # noqa: E402

from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import JointState, Pose, ToolPoseCriteria

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "train" / "smolvla_simft" / "checkpoints" / "010000" / "pretrained_model"
FAR = [0.0, 0.0, -2.0, 1, 0, 0, 0]
WALL = [0.012, 0.06, 0.06]  # thickness (along brick->box), length, height
DYN_XY, DYN_PLAN_H = 0.04, 0.12
MARGIN = 0.015  # the planner sees the sensed box inflated by this on every side (sphere model + PD tracking lag)
OPEN_N, CLOSED_N = G.OPEN_N, G.CLOSED_N
BUDGET, HANDBACK_RISE = 150, 0.02
JAWS = {"gripper_link", "moving_jaw_so101_v1_link"}


def grip_rad(n):
    return float(to_rad(np.r_[0, 0, 0, 0, 0, n])[5])


def box_dist(points, center, dims, yaw):
    R = Rotation.from_euler("z", yaw).as_matrix()
    q = np.abs((points - np.asarray(center)) @ R) - np.asarray(dims) / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)


def rect_dist(p, sc):
    y = sc["box_yaw"]
    R = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    q = np.abs(R.T @ (np.asarray(p[:2]) - sc["box"])) - [BOX["L"] / 2, BOX["W"] / 2]
    return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))


def wall_candidates(sc):
    """Walls between brick and box, best (most centred in the free gap) first; none if the gap is < 9 cm."""
    b, c = np.array(sc["brick"]), np.array(sc["box"])
    if rect_dist(b, sc) < 0.09:
        return []
    d = (c - b) / np.linalg.norm(c - b)
    cand = sorted(((min(s, rect_dist(b + d * s, sc)), s) for s in np.arange(0.045, np.linalg.norm(c - b), 0.005)), reverse=True)
    return [dict(center=[float(p[0]), float(p[1]), TZ + WALL[2] / 2], dims=WALL, yaw=float(np.arctan2(d[1], d[0])))
            for m, s_ in cand if m >= 0.045 for p in [b + d * s_]]


class PolicyClient:
    def __init__(self, path, port=5610):
        log = ROOT / "logs" / "policy_server.log"
        self.logf = open(log, "w")
        self.proc = subprocess.Popen([str(ROOT / ".venv-lerobot" / "Scripts" / "python.exe"), "-u", str(ROOT / "so101" / "policy_server.py"),
                                      "--policy", str(path), "--port", str(port)], stdout=self.logf, stderr=subprocess.STDOUT,
                                     env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
        t0 = time.time()
        while "POLICY_SERVER_READY" not in log.read_text(errors="ignore"):
            if self.proc.poll() is not None or time.time() - t0 > 600:
                raise RuntimeError(f"policy server failed, see {log}")
            time.sleep(2)
        self.f = socket.create_connection(("127.0.0.1", port), timeout=600).makefile("rwb")

    def call(self, **m):
        self.f.write((json.dumps(m) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())

    def close(self):
        try:
            self.call(cmd="quit")
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
        self.logf.close()


class Planner(G.Expert):
    def __init__(self, box_c, box_yaw):  # the generator's expert + a movable wall and a pre-allocated dynamic box
        make_pick_cfg()
        scene = Scene(cuboid=[Cuboid(name="table", pose=[0.15, 0.0, TZ - 0.02, 1, 0, 0, 0], dims=[1.0, 1.0, 0.04])]
                      + G.box_cuboids(box_c, box_yaw)
                      + [Cuboid(name="wall", pose=FAR, dims=WALL), Cuboid(name="dyn", pose=FAR, dims=[DYN_XY + 2 * MARGIN, DYN_XY + 2 * MARGIN, DYN_PLAN_H])])
        self.pl = MotionPlanner(MotionPlannerCfg.create(robot=str(PICK_CFG), scene_model=scene, max_goalset=4,
                                                        optimizer_collision_activation_distance=0.01))
        self.pl.warmup(enable_graph=True, num_warmup_iterations=2)
        self.tf = self.pl.tool_frames[0]
        self.box_c, self.box_yaw = np.asarray(box_c), box_yaw

    def move(self, name, pose):
        self.pl.scene_collision_checker.update_obstacle_pose(name, Pose.from_list(list(map(float, pose))))
        if getattr(self.pl, "graph_planner", None) is not None:
            self.pl.graph_planner.reset_buffer()

    def plan_relaxed(self, q5, goals):
        """Goalset plan; if IK finds no exact pose (a 5-DOF arm cannot match the orientation a tilted grasp implies),
        relax the orientation weights (position first). Returns (traj, weights used)."""
        for w in (None, [0.3, 0.3, 1.0], [0.05, 0.05, 0.2]):
            if w is not None:
                self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria.track_position_and_orientation(rpy=w)})
            try:
                tr = self.plan(q5, goals)
            finally:
                if w is not None:
                    self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria()})
            if tr is not None:
                return tr, w
        return None, None

    def spheres(self, q5):
        s = self.pl.kinematics.get_robot_as_spheres(torch.as_tensor(np.asarray(q5)[None], device="cuda", dtype=torch.float32),
                                                    filter_valid=True)[0]
        return np.array([x.position for x in s]).reshape(-1, 3), np.array([x.radius for x in s])

    def clearance(self, traj, ob):
        out = 1e9
        for q in traj[::2]:
            c, r = self.spheres(q)
            out = min(out, float((box_dist(c, ob["center"], ob["dims"], ob["yaw"]) - r).min()))
        return round(out * 1000, 1)


class Episode:
    def __init__(self, sim, pl, pol, sc, i, a):
        self.sim, self.pl, self.pol, self.sc, self.i, self.a = sim, pl, pol, sc, i, a
        self.log, self.captions, self.contacts = [], [], {}
        self.grip = CLOSED_N
        self.rng = np.random.default_rng(1000 + i)

    # ---------------- io ----------------
    def _acc(self, r):
        for k, v in r.get("contacts", {}).items():
            self.contacts[k] = round(self.contacts.get(k, 0.0) + v, 5)
        self.q = np.array(r["q"], np.float32)
        self.cube = r["cube"]
        self.frames = r.get("frames", getattr(self, "frames", 0))
        return r

    def exec(self, traj, grip_from=None, grip_to=None, settle=0.0):
        g0 = self.grip if grip_from is None else grip_from
        g = np.linspace(g0, g0 if grip_to is None else grip_to, len(traj))
        self.grip = g[-1]
        return self._acc(self.sim.call(cmd="exec", q=[[*map(float, q), grip_rad(gn)] for q, gn in zip(traj, g)], dt=1 / 30, settle=settle))

    def hold(self, g_to, n):
        return self.exec(np.repeat(self.q[None, :5], n, 0), self.grip, g_to)

    def note(self, step, ok, brief, **kw):
        self.log.append(dict(step=step, ok=bool(ok), brief=brief, frame=self.frames, t=round(time.perf_counter() - self.t0, 1), **kw))
        print(f"   [{self.i:02d}] {step:<15s} {'OK  ' if ok else 'FAIL'} {brief}", flush=True)
        return ok

    def rise(self):
        return self.cube[0][2] - self.z0

    # ---------------- steps ----------------
    def perceive(self):
        st = self._acc(self.sim.call(cmd="state"))
        c = np.array(st["cube"][0]) + np.r_[self.rng.normal(0, 0.0016, 2), 0.0]
        R = Rotation.from_quat(np.array(st["cube"][1])[[1, 2, 3, 0]]).as_matrix()
        self.est = dict(c=c, yaw=float(np.arctan2(R[1, 0], R[0, 0])))
        err = np.linalg.norm(c[:2] - np.array(st["cube"][0][:2])) * 1000
        return self.note("perceive", True, f"virtual sensor, err {err:.1f} mm", err_mm=round(err, 1))

    def grasp_goals(self):
        goals = []
        for y in (self.est["yaw"], self.est["yaw"] + np.pi):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(down_quat(y)[[1, 2, 3, 0]]).as_matrix()
            T[:3, 3] = np.r_[self.est["c"][:2], TZ + BRICK[2] / 2] - T[:3, :3] @ G.OFF
            goals.append(T)
        return goals

    def approach(self):
        pre = [T.copy() for T in self.grasp_goals()]
        for P in pre:
            P[:3, 3] -= P[:3, 2] * 0.06
        t = time.perf_counter()
        tr = self.pl.plan(self.q[:5], pre)
        ms = (time.perf_counter() - t) * 1000
        if tr is None:
            return self.note("plan_approach", False, "no pre-grasp plan", plan_ms=round(ms))
        self.exec(tr, self.grip, OPEN_N)
        return self.note("plan_approach", True, f"cuRobo pre-grasp, plan {ms:.0f} ms", plan_ms=round(ms))

    def policy_grasp(self, label):
        self.pol.call(cmd="reset")
        k, t = 0, time.perf_counter()
        for k in range(1, BUDGET + 1):
            rd = self.sim.call(cmd="render")
            act = np.array(self.pol.call(cmd="act", images=rd["images"], state=to_norm(np.array(rd["q"], np.float32)).tolist())["action"])
            self._acc(self.sim.call(cmd="exec", q=[to_rad(act).tolist()], dt=1 / 60, settle=0.0))
            self.grip = float(act[5])
            if self.rise() >= HANDBACK_RISE:
                break
        ms = (time.perf_counter() - t) / max(k, 1) * 1000
        return self.note(label, True, f"SmolVLA {k} steps, brick rise {self.rise() * 1000:.0f} mm", steps=k, ms_per_step=round(ms))

    def scripted_grasp(self, label):
        Tt = self.pl.tool_T(self.q[:5])
        Tg = min(self.grasp_goals(), key=lambda T: np.linalg.norm(T[:3, :3] - Tt[:3, :3]))
        tr = self.pl.plan(self.q[:5], [Tg], linear=True, disable=True)
        if tr is None:
            return self.note(label, False, "no descend plan")
        self.exec(tr, OPEN_N)
        self.hold(CLOSED_N, 18)
        return self.note(label, True, "cuRobo descend + close")

    def up(self, dz):
        """Straight up; else any collision-free path up. Returns the trajectory or None."""
        Tu = self.pl.tool_T(self.q[:5])
        Tu[2, 3] += dz
        tr = self.pl.plan(self.q[:5], [Tu], linear=True, disable=True)
        return tr if tr is not None else self.pl.plan(self.q[:5], [Tu], disable=True)

    def verify_grasp(self):
        tr = self.up(0.04)
        if tr is not None:
            self.exec(tr)
        self.contacts_before = dict(self.contacts)
        r = self._acc(self.sim.call(cmd="wait", seconds=0.15))
        jaws = {k.split("|")[0] for k in r["contacts"] if k.endswith("/World/cube")}
        ok = tr is not None and JAWS <= jaws and self.rise() >= 0.03
        return self.note("verify_grasp", ok, f"lift {self.rise() * 1000:.0f} mm, jaws {'both' if JAWS <= jaws else len(jaws & JAWS)} (physics)",
                         rise_mm=round(self.rise() * 1000, 1), jaw_contacts=sorted(jaws), lift_planned=tr is not None)

    def recover(self):
        self.hold(OPEN_N, 12)
        tr = self.up(0.06)
        if tr is not None:
            self.exec(tr)
            return self.note("recover", True, "open + back up 6 cm (cuRobo)")
        # cuRobo has no valid start (e.g. jaw pressed into the table): joint-interpolate to the ready pose,
        # the same unchecked motion the demos use from rest
        self.exec(np.linspace(self.q[:5], self.ready, 30))
        return self.note("recover", True, "open + joint move to ready (cuRobo start invalid; not collision-checked)", unchecked=True)

    def attach(self, scale):
        Tc = np.eye(4)
        Tc[:3, :3] = Rotation.from_quat(np.array(self.cube[1])[[1, 2, 3, 0]]).as_matrix()
        Tc[:3, 3] = self.cube[0]
        self.pl.pl.attachment_manager.attach(
            JointState.from_position(torch.tensor(self.q[None, :5], device="cuda"), joint_names=self.pl.pl.joint_names),
            [Cuboid(name="brick", pose=[0, 0, 0, 1, 0, 0, 0], dims=[d * scale for d in BRICK])], link_name="attached_object",
            num_spheres=6, sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=Pose.from_numpy(Tc[:3, 3].astype(np.float32), Rotation.from_matrix(Tc[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)))
        return Tc

    def transport(self):
        # the held brick joins the planner's robot model; if its spheres overlap the jaws (off-centre policy grasp),
        # the start is in self-collision -> retry with a shrunk brick, then without it; then escape up once (all logged)
        for escape in (False, True):
            if escape:
                c, r = self.pl.spheres(self.q[:5])
                diag = dict(start_table_clear_mm=round(float((c[:, 2] - r - TZ).min()) * 1000, 1))
                if self.wall:
                    diag["start_wall_clear_mm"] = self.pl.clearance([self.q[:5]], self.wall)
                tr = self.up(0.05)
                if tr is None:
                    return self.note("transport", False, "no transport plan, escape up failed", **diag)
                self.exec(tr)
                self.note("escape_up", True, "moved up 5 cm, retrying transport", **diag)
            Tt = self.pl.tool_T(self.q[:5])
            for mode, scale in (("attached", 1.0), ("attached_70pct", 0.7), ("not_attached", None)):
                Tc = self.attach(scale or 1.0)
                if scale is None:
                    self.pl.pl.attachment_manager.detach(link_name="attached_object")
                self.attach_mode = mode + ("_after_escape" if escape else "")
                try:
                    ok = self._transport(np.linalg.inv(Tt) @ Tc)
                finally:
                    if scale is not None:
                        self.pl.pl.attachment_manager.detach(link_name="attached_object")
                if ok != "no_plan":
                    return ok
        return False

    def _transport(self, attT):
        sc = self.sc
        place = []
        for yo in np.radians([0, 90, 180, 270]):
            P = np.eye(4)
            P[:3, :3] = Rotation.from_euler("z", sc["box_yaw"] + yo).as_matrix()
            P[:3, 3] = [*sc["box"], TZ + 0.004 + BRICK[2] / 2 + BOX["H"] + 0.03]
            place.append(P @ np.linalg.inv(attT))
        t = time.perf_counter()
        tr, w = self.pl.plan_relaxed(self.q[:5], place)
        ms = (time.perf_counter() - t) * 1000
        if tr is None:
            self.note("transport", False, f"no transport plan ({self.attach_mode})", plan_ms=round(ms))
            return "no_plan"
        info = dict(plan_ms=round(ms), attach_mode=self.attach_mode, rpy_weights=w)
        if self.wall:
            info["wall_clear_planned_mm"] = self.pl.clearance(tr, self.wall)
        k = int(0.35 * len(tr))
        self.exec(tr[:k])
        # scenario event: a box lands on the remaining path, away from the target box and the wall
        ee = [self.pl.tool_T(q)[:3, 3] for q in tr[k:]]
        cand = [p for p in ee if rect_dist(p, sc) > 0.045 and np.linalg.norm(p[:2]) > 0.09
                and (self.wall is None or np.linalg.norm(p[:2] - self.wall["center"][:2]) > 0.06)]
        if not cand:
            self.note("obstacle_event", True, "no free spot on the path: no box spawned")
            self.exec(tr[k:])
            return self.note("transport", True, f"plan {ms:.0f} ms", **info)
        # nearest to the middle of the remaining path whose box keeps >= 15 mm from the arm's current pose
        order = sorted(range(len(cand)), key=lambda j: abs(j - len(cand) // 2))
        spot = None
        for j in order:
            p = cand[j]
            h = float(min(DYN_PLAN_H, p[2] - TZ + 0.03))
            box = dict(center=[float(p[0]), float(p[1]), TZ + h / 2], dims=[DYN_XY, DYN_XY, h], yaw=0.0)
            if self.pl.clearance([self.q[:5]], box) >= 15.0:
                spot = box
                break
        if spot is None:
            self.note("obstacle_event", True, "no spot clear of the arm: no box spawned")
            self.exec(tr[k:])
            return self.note("transport", True, f"plan {ms:.0f} ms", **info)
        h = spot["dims"][2]
        self.spawned.append(f"dyn_{self.i:02d}")
        self.sim.call(cmd="spawn", name=f"dyn_{self.i:02d}", center=spot["center"], dims=spot["dims"])
        st = self._acc(self.sim.call(cmd="state"))  # virtual sensor: the sim's obstacle list
        new = {n: o for n, o in st["obstacles"].items() if n.startswith("dyn")}
        if not new:
            return self.note("obstacle_event", False, "spawned box not sensed")
        ob = next(iter(new.values()))
        self.dyn = dict(center=ob["center"], dims=ob["dims"], yaw=ob["yaw"])
        top = ob["center"][2] + ob["dims"][2] / 2
        self.pl.move("dyn", [ob["center"][0], ob["center"][1], top + MARGIN - DYN_PLAN_H / 2, 1, 0, 0, 0])
        rest_cl = self.pl.clearance(tr[k:], self.dyn)
        info.update(dyn_box_h_mm=round(h * 1000), remaining_path_clear_mm=rest_cl)
        self.note("obstacle_event", True, f"box appeared, old path clearance {rest_cl:.0f} mm", clear_mm=rest_cl)
        if rest_cl >= 5.0:
            self.exec(tr[k:])
            return self.note("transport", True, f"path still clear ({rest_cl:.0f} mm), no replan", **info)
        info["start_clear_mm"] = self.pl.clearance([self.q[:5]], self.dyn)
        t = time.perf_counter()
        tr2, _ = self.pl.plan_relaxed(self.q[:5], place)
        if tr2 is None:  # escape upwards first, then retry
            Tu = self.pl.tool_T(self.q[:5])
            Tu[2, 3] += 0.08
            tu = self.pl.plan(self.q[:5], [Tu])
            if tu is not None:
                self.exec(tu)
                tr2, _ = self.pl.plan_relaxed(self.q[:5], place)
                info["escape_up"] = True
        rms = (time.perf_counter() - t) * 1000
        info["replan_ms"] = round(rms)
        if tr2 is None:
            return self.note("replan", False, f"no path around the box ({rms:.0f} ms)", **info)
        info["dyn_clear_replanned_mm"] = self.pl.clearance(tr2, self.dyn)
        if self.wall:
            info["wall_clear_replanned_mm"] = self.pl.clearance(tr2, self.wall)
        self.exec(tr2)
        return self.note("replan", True, f"replanned in {rms:.0f} ms, box clearance {info['dyn_clear_replanned_mm']:.0f} mm", **info)

    def place_and_retreat(self):
        Td = self.pl.tool_T(self.q[:5])
        Td[2, 3] -= 0.025
        tr = self.pl.plan(self.q[:5], [Td], linear=True, disable=True)
        if tr is not None:
            self.exec(tr)
        self.hold(OPEN_N, 12)
        self.hold(OPEN_N, 9)
        self._acc(self.sim.call(cmd="wait", seconds=0.5))
        r = judge(self.cube, self.sc, self.z0, 0.0)
        ok = self.note("verify_place", r["success"], "brick inside the box (physics)" if r["success"] else "brick not in the box",
                       brick_final=r["brick_final"])
        Tu = self.pl.tool_T(self.q[:5])
        Tu[2, 3] += 0.05
        tr = self.pl.plan(self.q[:5], [Tu], linear=True, disable=True)
        if tr is not None:
            self.exec(tr)
        self.note("retreat", tr is not None, "back up 5 cm")
        return ok

    # ---------------- chain ----------------
    def run(self, rest_n):
        sc, self.t0 = self.sc, time.perf_counter()
        r = self._acc(self.sim.call(cmd="reset", q0=to_rad(np.r_[rest_n[:5] + np.array(sc["rest_noise"]), CLOSED_N]).tolist(),
                                    cube=dict(center=[*sc["brick"], TZ + BRICK[2] / 2], yaw=sc["brick_yaw"]),
                                    cameras=sc["cams"], table_color=[sc["shade"]] * 3))
        self.frame0 = self.frames
        self.z0 = r["cube"][0][2]
        # rest -> ready by joint interpolation (as in the demos; not collision-checked), gripper opens
        ready = to_rad(np.array([rest_n[0], -55.0, 60.0, 70.0, rest_n[4], CLOSED_N]))[:5]
        to_ready = np.linspace(self.q[:5], ready, 30)
        self.ready = ready
        # static wall: the most centred candidate that keeps >= 15 mm from that unchecked sweep
        self.wall, self.dyn = next((w for w in wall_candidates(sc) if self.pl.clearance(to_ready, w) >= 15.0), None), None
        self.spawned = []
        if self.wall:
            self.spawned.append(f"wall_{self.i:02d}")
            self.sim.call(cmd="spawn", name=f"wall_{self.i:02d}", color=[0.25, 0.35, 0.85], **self.wall)
            self.pl.move("wall", [*self.wall["center"], *G.yawq(self.wall["yaw"])])
        else:
            self.pl.move("wall", FAR)
        self.pl.move("dyn", FAR)
        self.note("scene", True, "static wall between brick and box" if self.wall else "no static wall (no room)")
        self.exec(to_ready, CLOSED_N, OPEN_N)
        path = None
        attempts = [("policy_grasp", "policy"), ("policy_retry", "policy_retry"), ("curobo_grasp", "curobo_fallback")] \
            if self.a.grasp == "policy" else [("curobo_grasp", "curobo")]
        if self.a.no_fallback:
            attempts = attempts[:2]
        for n, (label, tag) in enumerate(attempts):
            if n:
                if not self.recover():
                    break
            self.perceive()
            if not self.approach():
                continue
            ok = self.policy_grasp(label) if label.startswith("policy") else self.scripted_grasp(label)
            if ok and self.verify_grasp():
                path = tag
                break
        res = dict(scenario=self.i, grasp_path=path, wall=self.wall is not None)
        if path is not None and self.transport() is True:
            res["placed"] = self.place_and_retreat()
        else:
            res["placed"] = False
        res["success"] = bool(res["placed"])
        # contacts with the obstacles (wall / dynamic box): robot links and the brick
        hits = {k: v for k, v in self.contacts.items() if "/World/obstacles/wall" in k or "/World/obstacles/dyn" in k}
        res.update(obstacle_contacts=hits, robot_obstacle_contacts=sum(1 for k in hits if not k.startswith("cube|")),
                   bin_contacts={k: v for k, v in self.contacts.items() if "/World/bin" in k and not k.startswith("cube|")},
                   wall_s=round(time.perf_counter() - self.t0, 1), sim_s=round((self.frames - self.frame0) / 30, 1),
                   frame_start=self.frame0, frame_end=self.frames, log=self.log)
        for n in self.spawned:
            self.sim.call(cmd="remove", name=n)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp", choices=["policy", "curobo"], default="policy")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--policy", default=str(POLICY))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    a = ap.parse_args()
    tag = a.tag or f"chain_{a.grasp}"
    S = json.loads((ROOT / "logs" / "eval_scenarios.json").read_text())
    rest_n = np.array(S["rest"])
    ids = [int(x) for x in a.only.split(",")] if a.only else list(range(len(S["scenarios"])))
    pol = PolicyClient(a.policy) if a.grasp == "policy" else None
    out, sim, cur_box, t0 = [], None, None, time.perf_counter()
    for i in ids:
        sc = S["scenarios"][i]
        if cur_box != sc["box"]:
            if sim is not None:
                sim.close()
            fdir = ROOT / "isaac" / f"frames_{tag}_{i:02d}"
            shutil.rmtree(fdir, ignore_errors=True)
            sim = IsaacSim(port=5603, frames_dir=fdir, log=ROOT / "logs" / f"isaac_{tag}.log")
            sim.call(cmd="setup", scene=session_scene(sc, rest_n))
            pl = Planner(np.array(sc["box"]), sc["box_yaw"])
            cur_box, sdir = sc["box"], fdir.name
        ep = Episode(sim, pl, pol, sc, i, a)
        try:
            r = ep.run(rest_n)
        except Exception as e:  # keep going; record the failure
            import traceback

            traceback.print_exc()
            r = dict(scenario=i, success=False, error=repr(e), log=ep.log, frame_end=getattr(ep, "frames", 0))
        r["frames_dir"] = sdir
        out.append(r)
        print(f"[{tag}] scenario {i:02d}: {'SUCCESS' if r['success'] else 'fail   '} grasp={r.get('grasp_path')} "
              f"obstacle_contacts={r.get('robot_obstacle_contacts')} ({r.get('wall_s')} s)", flush=True)
        (ROOT / "logs" / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    sim.close()
    if pol:
        pol.close()
    n = len(out)
    print(f"[{tag}] SUCCESS {sum(r['success'] for r in out)}/{n} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
