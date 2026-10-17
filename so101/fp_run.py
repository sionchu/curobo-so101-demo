"""Run NVlabs FoundationPose (model-based) on the real CarolinePascal/so101_depth_raw RGB-D frames.

Run with the .venv-fp interpreter via fp_env.bat (CUDA 13 pip toolchain + MSVC), cwd = FoundationPose/.
- depth is not registered to RGB in this dataset: we reproject it into the RGB camera using the depth
  camera pose (table-plane fit) and RGB camera pose (PnP) estimated earlier (logs/fp_inputs.npz)
- meshes are built from measured geometry: 32.7 mm cube, 122x126x46 mm open bin (metric .obj)
- init masks: HSV colour masks from the RGB frame (red cube, green bin)
- output: poses, comparison with the ICP/PnP fallback, overlay PNGs in logs/fp_overlay/
"""

import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "FoundationPose"
sys.path.insert(0, str(FP))
os.chdir(FP)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed  # noqa: E402

inp = np.load(ROOT / "logs" / "fp_inputs.npz")
K_rgb, K_d = inp["K_rgb"], inp["K_depth"]
T_w_rgb, T_w_d = inp["T_w_rgb"], inp["T_w_depth"]
OUT = ROOT / "logs" / "fp_overlay"
OUT.mkdir(exist_ok=True)
for f in OUT.glob("*.png"):
    f.unlink()


def load_frames():
    t = pq.read_table(ROOT / "data/so101_depth_raw/data/chunk-000/file-000.parquet").to_pylist()
    img = lambda c: np.array(Image.open(io.BytesIO(c["bytes"])))
    return [img(r["observation.images.front"])[..., :3] for r in t], [img(r["observation.depth.front"]).astype(np.float32) for r in t]


def register_depth(depth):
    """Depth image (depth camera) -> depth image in the RGB camera (z-buffer)."""
    v, u = np.nonzero((depth > 0.15) & (depth < 1.5))
    z = depth[v, u]
    p = np.stack([(u - K_d[0, 2]) * z / K_d[0, 0], (v - K_d[1, 2]) * z / K_d[1, 1], z, np.ones_like(z)])
    T = np.linalg.inv(T_w_rgb) @ T_w_d
    q = T @ p
    uu = np.round(q[0] / q[2] * K_rgb[0, 0] + K_rgb[0, 2]).astype(int)
    vv = np.round(q[1] / q[2] * K_rgb[1, 1] + K_rgb[1, 2]).astype(int)
    ok = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 480) & (q[2] > 0)
    out = np.full((480, 640), np.inf, np.float32)
    np.minimum.at(out, (vv[ok], uu[ok]), q[2][ok].astype(np.float32))
    out[np.isinf(out)] = 0
    # close 1-px holes from the reprojection (forward warping)
    filled = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    out[out == 0] = filled[out == 0]
    return out


def color_mesh(m, rgb):
    m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=0.004)[:2])
    m.visual.vertex_colors = np.tile(np.array([*rgb, 255], np.uint8), (len(m.vertices), 1))
    return m


def cube_mesh(s):
    return color_mesh(trimesh.creation.box(extents=[s, s, s]), (200, 40, 40))


def bin_mesh(L, W, H, th=0.003):
    parts = [trimesh.creation.box(extents=[L, W, th], transform=trimesh.transformations.translation_matrix([0, 0, -H / 2 + th / 2]))]
    for dx, dy, ex in [(L / 2 - th / 2, 0, [th, W, H]), (-L / 2 + th / 2, 0, [th, W, H]),
                       (0, W / 2 - th / 2, [L, th, H]), (0, -W / 2 + th / 2, [L, th, H])]:
        parts.append(trimesh.creation.box(extents=ex, transform=trimesh.transformations.translation_matrix([dx, dy, 0])))
    return color_mesh(trimesh.util.concatenate(parts), (20, 140, 70))


def hsv_mask(rgb, lo, hi, largest=True):
    m = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if largest:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m > 0


def draw_box(img, pose, extents, color):
    e = np.asarray(extents) / 2
    corners = np.array([[sx * e[0], sy * e[1], sz * e[2], 1] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]).T
    pc = pose @ corners
    uv = (K_rgb @ pc[:3]).T
    uv = (uv[:, :2] / uv[:, 2:]).astype(int)
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, 2, cv2.LINE_AA)
    o = (K_rgb @ (pose @ np.array([0, 0, 0, 1.0]))[:3])
    o = (o[:2] / o[2]).astype(int)
    for k, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        a = np.zeros(4)
        a[k], a[3] = 0.03, 1
        p = K_rgb @ (pose @ a)[:3]
        cv2.line(img, tuple(o), tuple((p[:2] / p[2]).astype(int)), c, 2, cv2.LINE_AA)


def yaw_of(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def wrap(a, period):
    return (a + period / 2) % period - period / 2


if __name__ == "__main__":
    set_seed(0)
    rgbs, depths = load_frames()
    s = float(inp["cube_size"])
    L, W, H = inp["bin_size"]
    objs = {"cube": (cube_mesh(s), [s, s, s], (0, 120, 70), (10, 255, 255), (255, 60, 60)),
            "bin": (bin_mesh(L, W, H), [L, W, H], (50, 120, 50), (85, 255, 255), (60, 255, 120))}
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    t0 = time.perf_counter()
    d0 = register_depth(depths[0])
    print(f"registered depth: {np.count_nonzero(d0)} px valid", flush=True)
    est, poses, timing = {}, {}, {}
    for name, (mesh, ext, lo, hi, _) in objs.items():
        e = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                           refiner=refiner, debug_dir=str(ROOT / "logs" / "fp_debug"), debug=0, glctx=glctx)
        mask = hsv_mask(rgbs[0], lo, hi)
        t = time.perf_counter()
        poses[name] = [e.register(K=K_rgb, rgb=rgbs[0], depth=d0, ob_mask=mask, iteration=5)]
        timing[name + "_register_s"] = round(time.perf_counter() - t, 2)
        est[name] = e
        print(f"{name}: registered in {timing[name + '_register_s']} s, mask {mask.sum()} px", flush=True)
    # track the cube (the arm barely moves in this episode, so the object is static: a stability test)
    t = time.perf_counter()
    for i in range(1, len(rgbs)):
        d = register_depth(depths[i])
        poses["cube"].append(est["cube"].track_one(rgb=rgbs[i], depth=d, K=K_rgb, iteration=2))
    timing["cube_track_ms_per_frame"] = round((time.perf_counter() - t) / (len(rgbs) - 1) * 1000, 1)

    # comparison with the ICP / PnP fallback (all in the robot base frame)
    res = {}
    for name in ("cube", "bin"):
        Tw = T_w_rgb @ poses[name][0]
        if name == "cube":
            ref_c, ref_yaw, period = np.asarray(inp["cube_center"]), float(np.degrees(inp["cube_yaw"])), 90.0
        else:
            ref_c, ref_yaw, period = np.r_[inp["bin_T"][:2, 3], inp["table_z"] + H / 2], yaw_of(inp["bin_T"]), 90.0
        up = Tw[:3, :3] @ np.array([0, 0, 1.0])
        res[name] = dict(fp_center_m=np.round(Tw[:3, 3], 4).tolist(), ref_center_m=np.round(ref_c, 4).tolist(),
                         diff_xy_mm=round(float(np.linalg.norm(Tw[:2, 3] - ref_c[:2]) * 1000), 1),
                         diff_z_mm=round(float((Tw[2, 3] - ref_c[2]) * 1000), 1),
                         yaw_diff_deg_mod90=round(float(wrap(yaw_of(Tw) - ref_yaw, period)), 1),
                         fp_up_axis_tilt_deg=round(float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1)))), 1))
    cw = np.array([(T_w_rgb @ P)[:3, 3] for P in poses["cube"]])
    res["cube_tracking_jitter_mm(std over 172 frames)"] = np.round(cw.std(0) * 1000, 2).tolist()
    res["timing"] = timing
    print(json.dumps(res, indent=1), flush=True)
    (ROOT / "logs" / "fp_results.json").write_text(json.dumps(res, indent=1))
    np.save(ROOT / "logs" / "fp_poses_cube_cam.npy", np.array(poses["cube"]))
    np.save(ROOT / "logs" / "fp_pose_bin_cam.npy", poses["bin"][0])

    # overlays: FoundationPose boxes (red cube, green bin) + ICP/PnP reference (white) on the real RGB
    ref_cube = np.linalg.inv(T_w_rgb) @ np.block([[np.array([[np.cos(inp["cube_yaw"]), -np.sin(inp["cube_yaw"]), 0],
                                                             [np.sin(inp["cube_yaw"]), np.cos(inp["cube_yaw"]), 0], [0, 0, 1]]),
                                                   np.asarray(inp["cube_center"]).reshape(3, 1)], [np.zeros((1, 3)), 1]])
    for i in range(len(rgbs)):
        img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
        draw_box(img, ref_cube, [s] * 3, (255, 255, 255))
        draw_box(img, poses["bin"][0], [L, W, H], (60, 255, 120))
        draw_box(img, poses["cube"][i], [s] * 3, (60, 60, 255))
        cv2.putText(img, f"FoundationPose frame {i}: red=cube (tracked) green=bin  white=ICP/PnP ref", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{i:04d}.png"), img)
    print("overlays written:", len(rgbs), "total %.1fs" % (time.perf_counter() - t0))
