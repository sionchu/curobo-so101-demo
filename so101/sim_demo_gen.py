"""Generate expert pick-and-place demonstrations in Isaac Sim (PhysX) with cuRobo as the expert.

    .venv\\Scripts\\python.exe so101\\sim_demo_gen.py --episodes 120 --per-session 10

Task = svla_so101_pickplace ("pink lego brick into the transparent box"). The task distribution is taken from the
real dataset: each episode picks a random *training* episode (0-44) and uses its grasp point/yaw (cuRobo FK of the
recorded joints) for the brick and its release point for the box, plus jitter. Cameras, table shade and box pose
are randomized lightly. Recording matches the real dataset: up/side 640x480 RGB, observation.state and action in the
dataset's normalized units (LeRobot RANGE_M100_100 / gripper RANGE_0_100), 30 fps. Only physics-verified successes
are kept (brick resting inside the box footprint). Raw output: data/sim_raw/ep_XXXX/{up,side}/*.jpg + data.npz.
"""

import argparse
import base64
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from harness_pickplace import CONTACT_LINKS, PICK_CFG, down_quat, make_pick_cfg  # noqa: E402
from isaac_client import IsaacSim  # noqa: E402
from policy_isaac import to_norm, to_rad  # noqa: E402

from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sim_raw"
TZ = -0.012
BRICK = [0.0158, 0.0318, 0.0114]  # 2x4 LEGO-size (assumed); short side along the jaw-closing axis
BOX = dict(L=0.10, W=0.075, H=0.03)  # transparent box size: assumption
OPEN_N, CLOSED_N = 28.0, 1.0  # gripper command in dataset units (real demos open to ~26, rest ~1.5)
CAM0 = {"up": dict(eye=[0.55, 0.0, 0.26], target=[0.0, 0.03, 0.08]),
        "side": dict(eye=[0.30, -0.42, 0.12], target=[0.02, 0.06, 0.07])}
OFF = np.array([-(BRICK[0] / 2 + 0.002), 0.0, -0.004])  # brick centre in the tool frame at grasp
FPS = 30


def yawq(y):
    return Rotation.from_euler("z", y).as_quat()[[3, 0, 1, 2]]


def box_cuboids(c, yaw):
    th, L, W, H = 0.003, BOX["L"], BOX["W"], BOX["H"]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    out = [Cuboid(name="box_floor", pose=[c[0], c[1], TZ + 0.002, *yawq(yaw)], dims=[L, W, 0.004])]
    for n, off, d in [("bxp", (L / 2 - th / 2, 0), [th, W, H]), ("bxm", (-L / 2 + th / 2, 0), [th, W, H]),
                      ("byp", (0, W / 2 - th / 2), [L, th, H]), ("bym", (0, -W / 2 + th / 2), [L, th, H])]:
        o = R @ np.array(off)
        out.append(Cuboid(name=n, pose=[c[0] + o[0], c[1] + o[1], TZ + H / 2, *yawq(yaw)], dims=d))
    return out


class Expert:
    def __init__(self, box_c, box_yaw):
        make_pick_cfg()
        scene = Scene(cuboid=[Cuboid(name="table", pose=[0.15, 0.0, TZ - 0.02, 1, 0, 0, 0], dims=[1.0, 1.0, 0.04])]
                      + box_cuboids(box_c, box_yaw))
        self.pl = MotionPlanner(MotionPlannerCfg.create(robot=str(PICK_CFG), scene_model=scene, max_goalset=4,
                                                        optimizer_collision_activation_distance=0.01))
        self.pl.warmup(enable_graph=True, num_warmup_iterations=2)
        self.tf = self.pl.tool_frames[0]
        self.box_c, self.box_yaw = np.asarray(box_c), box_yaw

    def tool_T(self, q5):
        p = self.pl.kinematics.compute_kinematics(JointState.from_position(
            torch.as_tensor(q5, device="cuda", dtype=torch.float32).view(1, -1), joint_names=self.pl.joint_names)
        ).tool_poses.get_link_pose(self.tf)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(p.quaternion.view(4).cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = p.position.view(3).cpu().numpy()
        return T

    def plan(self, q5, poses, linear=False, disable=False):
        """poses: list of 4x4 tool goals (goalset). Returns (N,5) joint path resampled to 30 fps, or None."""
        gs = len(poses)
        quats = [Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]] for T in poses]
        gp = GoalToolPose(tool_frames=[self.tf],
                          position=torch.tensor(np.array([T[:3, 3] for T in poses]), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 3),
                          quaternion=torch.tensor(np.array(quats), device="cuda", dtype=torch.float32).view(1, 1, 1, gs, 4))
        if linear:
            self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 1],
                non_terminal_pose_axes_weight_factor=[1, 1, 0, 1, 1, 1], project_distance_to_goal=True)})
        if disable:
            self.pl.disable_link_collision(CONTACT_LINKS)
        res = self.pl.plan_pose(gp, JointState.from_position(torch.tensor(np.asarray(q5)[None], device="cuda", dtype=torch.float32),
                                                            joint_names=self.pl.joint_names), max_attempts=4)
        if disable:
            self.pl.enable_link_collision(CONTACT_LINKS)
        if linear:
            self.pl.update_tool_pose_criteria({self.tf: ToolPoseCriteria()})
        if res is None or not res.success.any():
            return None
        p = res.get_interpolated_plan()
        traj = self.pl.kinematics.get_active_js(JointState.from_position(
            p.position.reshape(-1, p.position.shape[-1]), joint_names=p.joint_names)).position.cpu().numpy()
        dt = self.pl.trajopt_solver.config.interpolation_dt
        t = np.arange(len(traj)) * dt
        tt = np.arange(0, t[-1] + 1e-9, 1 / FPS)
        return np.stack([np.interp(tt, t, traj[:, j]) for j in range(5)], 1)


class Recorder:
    def __init__(self, sim, ep_dir):
        self.sim, self.dir = sim, ep_dir
        for c in ("up", "side"):
            (ep_dir / c).mkdir(parents=True, exist_ok=True)
        self.state, self.action, self.n = [], [], 0
        self.q = None

    def step(self, q5, grip_n):
        rr = self.sim.call(cmd="render")
        for name, img in rr["images"].items():
            a = np.frombuffer(base64.b64decode(img["b64"]), np.uint8).reshape(img["shape"])
            Image.fromarray(a).save(self.dir / name / f"{self.n:04d}.jpg", quality=92)
        act_n = np.r_[to_norm(np.r_[q5, 0.0])[:5], grip_n].astype(np.float32)
        self.state.append(to_norm(np.array(rr["q"], np.float32)))
        self.action.append(act_n)
        r = self.sim.call(cmd="exec", q=[to_rad(act_n).tolist()], dt=1 / 60, settle=0.0)
        self.q = np.array(r["q"], np.float32)
        self.cube = r["cube"]
        self.n += 1

    def path(self, traj, grip_from, grip_to=None):
        g = np.linspace(grip_from, grip_from if grip_to is None else grip_to, len(traj))
        for q5, gn in zip(traj, g):
            self.step(q5, gn)

    def hold(self, q5, grip_from, grip_to, n):
        for gn in np.linspace(grip_from, grip_to, n):
            self.step(q5, gn)


def run_episode(sim, ex, rng, real, i_ep, box_c, box_yaw, rest_n):
    # brick from a random real training episode's grasp point, kept away from the box
    for _ in range(50):
        k = int(rng.integers(0, 45))
        g = real["grasp"][k]
        bxy = np.array(g[:2]) + rng.normal(0, 0.01, 2)
        if np.linalg.norm(bxy - box_c) > 0.10 and 0.06 < np.linalg.norm(bxy) < 0.26:
            break
    yaw = np.radians(g[3] + rng.normal(0, 10))
    cams = {n: dict(eye=(np.array(c["eye"]) + rng.normal(0, 0.02, 3)).tolist(),
                    target=(np.array(c["target"]) + rng.normal(0, 0.01, 3)).tolist()) for n, c in CAM0.items()}
    shade = float(rng.uniform(0.78, 0.92))
    rest_r = to_rad(np.r_[np.array(rest_n[:5]) + rng.normal(0, 1.0, 5), CLOSED_N])
    r = sim.call(cmd="reset", q0=rest_r.tolist(), cube=dict(center=[float(bxy[0]), float(bxy[1]), TZ + BRICK[2] / 2], yaw=float(yaw)),
                 cameras=cams, table_color=[shade, shade, shade])
    z0 = r["cube"][0][2]
    ep_dir = OUT / f"ep_{i_ep:04d}"
    shutil.rmtree(ep_dir, ignore_errors=True)
    rec = Recorder(sim, ep_dir)
    rec.q = np.array(r["q"], np.float32)
    meta = dict(source_real_episode=k, brick=[*bxy.tolist(), float(yaw)], box=[*np.asarray(box_c).tolist(), box_yaw],
                cams=cams, table_shade=shade)

    def fail(why):
        meta["fail"] = why
        return False, meta, rec

    # 1) rest -> ready by joint interpolation (the rest pose is outside cuRobo's feasible set), gripper opens
    ready = to_rad(np.array([rest_n[0], -55.0, 60.0, 70.0, rest_n[4], CLOSED_N]))[:5]
    rec.path(np.linspace(rec.q[:5], ready, 30), CLOSED_N, OPEN_N)
    # 2) pre-grasp (vertical, jaws across the brick's short side; yaw and yaw+pi both work)
    c = np.array([bxy[0], bxy[1], TZ + BRICK[2] / 2])
    goals = []
    for y in (yaw, yaw + np.pi):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(down_quat(y)[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = c - T[:3, :3] @ OFF
        goals.append(T)
    pre = [T.copy() for T in goals]
    for P in pre:
        P[:3, 3] -= P[:3, 2] * 0.06
    tr = ex.plan(rec.q[:5], pre)
    if tr is None:
        return fail("no pre-grasp plan")
    rec.path(tr, OPEN_N)
    Tt = ex.tool_T(rec.q[:5])
    Tg = min(goals, key=lambda T: np.linalg.norm(T[:3, :3] - Tt[:3, :3]))
    # 3) straight-line descend, 4) close
    tr = ex.plan(rec.q[:5], [Tg], linear=True, disable=True)
    if tr is None:
        return fail("no descend plan")
    rec.path(tr, OPEN_N)
    rec.hold(rec.q[:5], OPEN_N, CLOSED_N, 18)
    # 5) lift and verify from physics
    Tl = ex.tool_T(rec.q[:5])
    Tl[:3, 3] = Tl[:3, 3] - Tl[:3, 2] * 0.06
    tr = ex.plan(rec.q[:5], [Tl], linear=True, disable=True)
    if tr is None:
        return fail("no lift plan")
    rec.path(tr, CLOSED_N)
    if rec.cube[0][2] - z0 < 0.03:
        return fail(f"brick not lifted ({(rec.cube[0][2] - z0) * 1000:.0f} mm)")
    # 6) transport with the brick attached to the planner's robot model (collision only)
    Tt = ex.tool_T(rec.q[:5])
    Tc = np.eye(4)
    Tc[:3, :3] = Rotation.from_quat(np.array(rec.cube[1])[[1, 2, 3, 0]]).as_matrix()
    Tc[:3, 3] = rec.cube[0]
    attT = np.linalg.inv(Tt) @ Tc
    ex.pl.attachment_manager.attach(JointState.from_position(torch.tensor(rec.q[None, :5], device="cuda"), joint_names=ex.pl.joint_names),
                                    [Cuboid(name="brick", pose=[0, 0, 0, 1, 0, 0, 0], dims=BRICK)], link_name="attached_object",
                                    num_spheres=6, sphere_fit_type=SphereFitType.VOXEL,
                                    world_objects_pose_offset=Pose.from_numpy(Tc[:3, 3].astype(np.float32), Rotation.from_matrix(Tc[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)))
    place = []
    for yo in np.radians([0, 90, 180, 270]):
        P = np.eye(4)
        P[:3, :3] = Rotation.from_euler("z", box_yaw + yo).as_matrix()  # brick upright, any of 4 yaws
        P[:3, 3] = [box_c[0], box_c[1], TZ + 0.004 + BRICK[2] / 2 + BOX["H"] + 0.03]
        place.append(P @ np.linalg.inv(attT))
    tr = ex.plan(rec.q[:5], place)
    ex.pl.attachment_manager.detach(link_name="attached_object")
    if tr is None:
        return fail("no transport plan")
    rec.path(tr, CLOSED_N)
    # 7) lower 2.5 cm, open, let it settle
    Td = ex.tool_T(rec.q[:5])
    Td[:3, 3][2] -= 0.025
    tr = ex.plan(rec.q[:5], [Td], linear=True, disable=True)
    if tr is not None:
        rec.path(tr, CLOSED_N)
    rec.hold(rec.q[:5], CLOSED_N, OPEN_N, 12)
    rec.hold(rec.q[:5], OPEN_N, OPEN_N, 9)
    # 8) retreat up, back to ready, then rest (gripper closes like the human demos)
    Tu = ex.tool_T(rec.q[:5])
    Tu[:3, 3][2] += 0.05
    tr = ex.plan(rec.q[:5], [Tu], linear=True, disable=True)
    if tr is not None:
        rec.path(tr, OPEN_N)
    rec.path(np.linspace(rec.q[:5], to_rad(np.r_[rest_n[:5], CLOSED_N])[:5], 40), OPEN_N, CLOSED_N)
    rec.hold(rec.q[:5], CLOSED_N, CLOSED_N, 6)
    # success: brick resting inside the box footprint
    b = np.array(rec.cube[0])
    Rb = np.array([[np.cos(box_yaw), -np.sin(box_yaw)], [np.sin(box_yaw), np.cos(box_yaw)]])
    lb = Rb.T @ (b[:2] - box_c)
    inside = abs(lb[0]) < BOX["L"] / 2 - 0.004 and abs(lb[1]) < BOX["W"] / 2 - 0.004 and b[2] < TZ + 0.004 + BRICK[2]
    meta.update(frames=rec.n, brick_final=b.tolist(), brick_in_box_local=lb.tolist())
    if not inside:
        return fail("brick not in box at the end")
    return True, meta, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--per-session", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=5601)
    a = ap.parse_args()
    real = json.loads((ROOT / "logs" / "real_task_stats.json").read_text())
    rest_n = np.mean(np.array(real["rest"])[:45], 0)
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    kept = len(list(OUT.glob("ep_*/data.npz")))
    attempts, t0 = 0, time.perf_counter()
    while kept < a.episodes:
        # one Isaac session per box placement (box = a real episode's release point + jitter)
        k = int(rng.integers(0, 45))
        box_c = np.array(real["release"][k][:2]) + rng.normal(0, 0.01, 2)
        box_yaw = float(rng.normal(0, np.radians(10)))
        ex = Expert(box_c, box_yaw)
        sim = IsaacSim(port=a.port, frames_dir=ROOT / "isaac" / "frames_gen", log=ROOT / "logs" / "isaac_gen.log")
        sim.call(cmd="setup", scene=dict(
            table_z=TZ, friction=1.0, place=[5.0, 5.0], q0=to_rad(np.r_[rest_n[:5], CLOSED_N]).tolist(),
            bin=dict(center=box_c.tolist(), yaw=box_yaw, L=BOX["L"], W=BOX["W"], H=BOX["H"], color=[0.82, 0.88, 0.93]),
            cube=dict(center=[0.15, 0.05, TZ + BRICK[2] / 2], yaw=0.0, size=0.02, dims=BRICK, color=[1.0, 0.35, 0.55], mass=0.003),
            obstacles=[dict(name="backdrop_x", center=[-0.45, 0.0, 0.3], dims=[0.02, 2.0, 0.8], color=[0.92, 0.92, 0.92]),
                       dict(name="backdrop_y", center=[0.0, 0.55, 0.3], dims=[2.0, 0.02, 0.8], color=[0.92, 0.92, 0.92])],
            cameras=CAM0, robot_color=[0.05, 0.05, 0.05], table_color=[0.85, 0.85, 0.85],
            cam_eye=[0.45, 0.35, 0.35], cam_target=[0.1, 0.0, 0.05],
            gains=dict(arm_kp=300.0, arm_kd=15.0, grip_kp=8.0, grip_kd=0.5)))
        for _ in range(a.per_session):
            if kept >= a.episodes:
                break
            attempts += 1
            te = time.perf_counter()
            try:
                ok, meta, rec = run_episode(sim, ex, rng, real, kept, box_c, box_yaw, rest_n)
            except Exception as e:  # keep generating; log the failure
                ok, meta, rec = False, dict(fail=f"exception {e!r}"), None
            meta.update(ok=ok, seconds=round(time.perf_counter() - te, 1), attempt=attempts)
            log.append(meta)
            if ok:
                np.savez(OUT / f"ep_{kept:04d}" / "data.npz", state=np.array(rec.state, np.float32),
                         action=np.array(rec.action, np.float32))
                (OUT / f"ep_{kept:04d}" / "meta.json").write_text(json.dumps(meta))
                kept += 1
            elif rec is not None:
                shutil.rmtree(rec.dir, ignore_errors=True)
            print(f"[{attempts:3d}] {'OK  ' if ok else 'FAIL'} kept={kept} frames={meta.get('frames', '-')} "
                  f"{meta.get('fail', '')} ({meta['seconds']}s, total {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            (ROOT / "logs" / "sim_gen_log.json").write_text(json.dumps(log, indent=1, default=float))
        sim.close()
    print(f"DONE kept {kept} of {attempts} attempts, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
