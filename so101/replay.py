"""Replay a real SO-101 episode (LeRobot dataset) in Viser, next to its camera video,
and compare it with a cuRobo re-plan through the recorded end-effector waypoints.

    python so101/replay.py --episode 0 --port 8083

Joint mapping (assumption, no calibration file ships with the dataset):
LeRobot RANGE_M100_100 puts 0 at the middle of each motor's calibrated range and
+-100 at its ends; we map +-100 onto the so101_new_calib.urdf joint limits.
Gripper uses RANGE_0_100 and is mapped onto [lower, upper] of the URDF gripper joint.
"""

import argparse
import threading
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import yourdfpy
from viser.extras import ViserUrdf

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria
from curobo.viewer import ViserVisualizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FULL = str(HERE / "so101.yml")
ARM = str(HERE / "so101_arm.yml")
DATA = ROOT / "data" / "svla_so101_pickplace"
URDF = ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# URDF limits (rad) from so101_new_calib.urdf
LIMITS = np.array(
    [
        [-1.91986, 1.91986],
        [-1.74533, 1.74533],
        [-1.69, 1.69],
        [-1.65806, 1.65806],
        [-2.74385, 2.84121],
        [-0.174533, 1.74533],
    ],
    dtype=np.float32,
)
LIMIT_MARGIN = 0.08  # rad
# Table top 2 cm below the base plane: in the recording the gripper touches the table
# when grasping, which the conservative sphere model would otherwise flag as collision.
TABLE_ONLY = {"cuboid": {"table": {"dims": [1.0, 1.0, 0.04], "pose": [0, 0, -0.04, 1, 0, 0, 0]}}}
VIDEO_W, VIDEO_H = 320, 240


def to_radians(norm: np.ndarray) -> np.ndarray:
    """Map LeRobot normalized joint values (T, 6) to URDF radians."""
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    rad = mid + norm / 100.0 * half
    rad[:, 5] = lo[5] + norm[:, 5] / 100.0 * (hi[5] - lo[5])
    return np.clip(rad, lo, hi)


def load_episode(ep: int):
    data = pq.read_table(DATA / "data/chunk-000/file-000.parquet").to_pandas()
    data = data[data["episode_index"] == ep].sort_values("frame_index")
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    meta = pq.read_table(DATA / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    row = meta[meta["episode_index"] == ep].iloc[0]
    task = row["tasks"][0] if len(row["tasks"]) else ""
    return state, row, task


def decode_segment(key: str, t0: float, t1: float, n: int) -> np.ndarray:
    """Decode frames in [t0, t1) of a LeRobot v3 concatenated video, resized to VIDEO_WxVIDEO_H."""
    path = DATA / f"videos/{key}/chunk-000/file-000.mp4"
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t0 - 1e-3:
                continue
            if t >= t1 - 1e-3 or len(frames) >= n:
                break
            frames.append(f.reformat(width=VIDEO_W, height=VIDEO_H, format="rgb24").to_ndarray())
    while len(frames) < n:  # pad if the segment came up short
        frames.append(frames[-1])
    return np.stack(frames)


def replan(q_rec: np.ndarray, ee_pos: torch.Tensor, ee_quat: torch.Tensor, step: int):
    """Plan through every `step`-th recorded EE pose; return (T, 5) joints resampled to the recording."""
    planner = MotionPlanner(MotionPlannerCfg.create(robot=ARM, scene_model=TABLE_ONLY))
    tf = planner.tool_frames[0]
    planner.update_tool_pose_criteria(
        {tf: ToolPoseCriteria.track_position_and_orientation(xyz=[1, 1, 1], rpy=[0.05] * 3)}
    )
    planner.warmup(enable_graph=True, num_warmup_iterations=3)

    T = len(q_rec)
    # Keep planner states slightly inside the URDF limits (the recorded rest pose sits on them).
    lo, hi = LIMITS[:5, 0] + LIMIT_MARGIN, LIMITS[:5, 1] - LIMIT_MARGIN
    inside = lambda x: np.clip(x, lo, hi).astype(np.float32)
    js = lambda x: JointState.from_position(torch.tensor(x[None], device="cuda"), joint_names=planner.joint_names)

    # cuRobo's sphere model flags the tightly folded rest pose (start/end of each episode) as
    # self-colliding, so only the span between the first and last feasible frame is re-planned;
    # outside it the recorded joints are copied.
    feas = planner.graph_planner.check_samples_feasibility(
        torch.tensor(inside(q_rec[:, :5]), device="cuda")
    ).cpu().numpy()
    ok_idx = np.flatnonzero(feas)
    out = inside(q_rec[:, :5]).copy()
    counts = {"pose": 0, "cspace": 0, "hold": 0, "fail": 0, "rest_copied_frames": int(T - (ok_idx[-1] - ok_idx[0] + 1))}
    if len(ok_idx) == 0:
        return out, counts, 0.0, (0, 0)
    first, last = int(ok_idx[0]), int(ok_idx[-1])
    idx = [i for i in range(first + step, last, step) if feas[i]] + [last]

    cur = js(out[first])
    prev, solve = first, 0.0
    for i in idx:
        n = i - prev
        res, kind = None, "fail"
        if np.abs(inside(q_rec[i, :5]) - out[prev]).max() < 0.02:
            kind = "hold"  # arm is (nearly) still in the recording
        else:
            goal = Pose(position=ee_pos[i : i + 1].clone(), quaternion=ee_quat[i : i + 1].clone())
            res = planner.plan_pose(GoalToolPose.from_poses({tf: goal}, num_goalset=1), cur, max_attempts=3)
            if res is not None and res.success.any():
                kind = "pose"
            else:  # IK rejected the pose (e.g. on a joint limit): plan in joint space instead
                res = planner.plan_cspace(js(inside(q_rec[i, :5])), cur, max_attempts=3)
                kind = "cspace" if res is not None and res.success.any() else "fail"
        counts[kind] += 1
        if kind in ("pose", "cspace"):
            solve += float(res.total_time)
            plan = res.get_interpolated_plan()
            seg = planner.kinematics.get_active_js(
                JointState.from_position(plan.position.reshape(-1, plan.position.shape[-1]), joint_names=plan.joint_names)
            ).position.cpu().numpy()
        else:
            if kind == "fail":
                print(f"  segment to frame {i} failed")
            seg = np.repeat(out[prev : prev + 1], 2, axis=0)
        src = np.linspace(0, 1, len(seg))
        dst = np.linspace(0, 1, n + 1)[1:]
        out[prev + 1 : i + 1] = np.stack([np.interp(dst, src, seg[:, j]) for j in range(5)], -1)
        cur = js(out[i])
        prev = i
    return out, counts, solve, (first, last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--waypoint-step", type=int, default=15, help="frames between re-plan waypoints")
    ap.add_argument("--record", type=str, default=None, help="write an MP4 of the replay and exit")
    a = ap.parse_args()

    state, row, task = load_episode(a.episode)
    q = to_radians(state)
    T = len(q)
    print(f"episode {a.episode}: {T} frames, task: {task}")

    t0 = float(row["videos/observation.images.up/from_timestamp"])
    t1 = float(row["videos/observation.images.up/to_timestamp"])
    vid_up = decode_segment("observation.images.up", t0, t1, T)
    t0s = float(row["videos/observation.images.side/from_timestamp"])
    t1s = float(row["videos/observation.images.side/to_timestamp"])
    vid_side = decode_segment("observation.images.side", t0s, t1s, T)
    print("decoded video frames:", vid_up.shape, vid_side.shape)

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(FULL))
    tf = kin.tool_frames[0]
    poses = kin.compute_kinematics(
        JointState.from_position(torch.tensor(q, device="cuda"), joint_names=JOINTS)
    ).tool_poses.get_link_pose(tf)
    ee_pos = poses.position.reshape(T, 3).clone()  # FK output buffers are reused
    ee_quat = poses.quaternion.reshape(T, 4).clone()

    t_plan = time.time()
    q_plan5, counts, solve, (p_first, p_last) = replan(q, ee_pos, ee_quat, a.waypoint_step)
    q_plan = np.concatenate([q_plan5, q[:, 5:6]], axis=1)  # gripper follows the recording
    plan_ee = kin.compute_kinematics(
        JointState.from_position(torch.tensor(q_plan, device="cuda"), joint_names=JOINTS)
    ).tool_poses.get_link_pose(tf).position.reshape(T, 3).clone()
    dev = (plan_ee - ee_pos).norm(dim=-1).cpu().numpy()
    wp_idx = np.arange(p_first + a.waypoint_step, p_last, a.waypoint_step)
    print(
        f"re-plan frames {p_first}..{p_last}, segments {counts}, solver {solve:.2f}s total "
        f"(wall {time.time() - t_plan:.1f}s incl. warmup); EE gap mean (planned span) {dev[p_first:p_last + 1].mean()*1000:.1f} mm, "
        f"at waypoints {dev[wp_idx].mean()*1000:.1f} mm"
    )

    # ---------------- Viser ----------------
    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=FULL),
        connect_port=a.port,
        add_robot_to_scene=True,
        add_control_frames=False,
    )
    server = viz._server

    @server.on_client_connect
    def _(client):  # start the camera close to the 0.4 m-reach arm
        client.camera.position = (0.55, -0.45, 0.4)
        client.camera.look_at = (0.15, 0.0, 0.05)

    ghost = ViserUrdf(
        server,
        yourdfpy.URDF.load(str(URDF), load_meshes=True, build_scene_graph=True,
                           filename_handler=lambda fname: str(URDF.parent / fname.replace("package://", ""))),
        root_node_name="/planned",
        mesh_color_override=(0.2, 0.8, 0.3, 0.45),
    )
    ghost_order = [JOINTS.index(n) for n in ghost.get_actuated_joint_names()]
    ee_np = ee_pos.cpu().numpy()
    pe_np = plan_ee.cpu().numpy()
    seg = lambda p: np.stack([p[:-1], p[1:]], axis=1)
    server.scene.add_line_segments("/paths/recorded", seg(ee_np), colors=(220, 40, 40), line_width=3.0)
    server.scene.add_line_segments("/paths/planned", seg(pe_np), colors=(40, 170, 70), line_width=3.0)
    server.scene.add_point_cloud("/paths/waypoints", ee_np[wp_idx], colors=(30, 30, 30), point_size=0.008)

    server.gui.add_markdown(
        f"**Real SO-101 episode {a.episode}** - _{task}_\n\n"
        f"Yellow = recorded joints | green ghost = cuRobo re-plan\n\n"
        f"Red path = recorded gripper | green = re-plan | dots = waypoints\n\n"
        f"Re-plan segments: {counts['pose']} pose-goal, {counts['cspace']} joint-goal, {counts['hold']} hold, {counts['fail']} failed (frames {p_first}-{p_last}; rest pose outside is copied) | EE gap mean {dev[p_first:p_last + 1].mean()*1000:.1f} mm"
    )
    img_up = server.gui.add_image(vid_up[0], label="camera: up (real)", format="jpeg")
    img_side = server.gui.add_image(vid_side[0], label="camera: side (real)", format="jpeg")
    slider = server.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("play", initial_value=True)
    show_ghost = server.gui.add_checkbox("show re-plan ghost", initial_value=True)
    info = server.gui.add_markdown("")

    def render(i: int):
        viz.set_joint_state(JointState.from_position(torch.tensor(q[i : i + 1], device="cuda"), joint_names=JOINTS).squeeze(0))
        ghost.update_cfg(q_plan[i][ghost_order])
        img_up.image = vid_up[i]
        img_side.image = vid_side[i]
        info.content = f"t = {i / 30:.2f}s | EE gap {dev[i]*1000:.0f} mm"

    slider.on_update(lambda _: render(int(slider.value)))
    show_ghost.on_update(lambda _: setattr(ghost, "show_visual", show_ghost.value))

    print(f"Replay viewer at http://localhost:{a.port}", flush=True)
    if a.record:
        from record_util import Mp4Writer, compose, look_at_wxyz, wait_client

        playing.value = False
        client = wait_client(server)
        cam_p, cam_t = (0.42, -0.42, 0.32), (0.12, 0.0, 0.06)
        wxyz = look_at_wxyz(cam_p, cam_t)
        out = Mp4Writer(a.record, fps=15)  # 30 fps recording played at 0.5x
        for i in range(T):
            render(i)
            time.sleep(0.05)
            img = client.get_render(720, 800, wxyz=wxyz, position=cam_p, fov=np.deg2rad(50))
            out.write(compose(
                img, [(vid_up[i], "real camera: up"), (vid_side[i], "real camera: side")],
                [f"Real SO-101 recording: lerobot/svla_so101_pickplace, episode {a.episode}",
                 f"task: {task}",
                 "yellow = recorded joints   green ghost = cuRobo re-plan",
                 "red path = recorded gripper   green path = re-plan",
                 f"t = {i / 30:.2f} s (0.5x speed)   gripper gap {dev[i] * 1000:.0f} mm"],
            ))
        out.close()
        return
    i = 0
    while True:
        if playing.value:
            i = (int(slider.value) + 1) % T
            slider.value = i  # triggers render
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
