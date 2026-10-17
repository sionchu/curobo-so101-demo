"""Model-based 6D perception fallback for the SO-101 harness demo (FoundationPose is not built here).

All poses are in the robot base_link frame ("world"). Real inputs: CarolinePascal/so101_depth_raw
(depth in metres + unregistered RGB). Steps:
  1. bin pose: depth segmentation + ICP of an open-box rim model (x, y, yaw; z from data)
  2. RGB camera pose: PnP of the bin's outer rim corners (RGB green segmentation) vs the bin model
     -> cross-checked against the depth camera position (RealSense colour/depth baseline is ~1.5 cm)
  3. red cube: RGB red segmentation -> ray / plane intersection at the cube-top height
  4. place target: yellow sticker -> ray / table-plane intersection

perceive(frames, noise=None) returns a dict with poses and residuals. `noise` is used by the
harness to inject a perception error (synthetic, labelled as such).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from depth_scene import backproject, load_frames, transform  # noqa: E402

LOGS = Path(__file__).resolve().parent.parent / "logs"
# Assumption: RealSense D435 colour stream defaults at 640x480 (no intrinsics in the dataset)
RGB_K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]])


def _poses():
    T_tc = np.load(LOGS / "T_table_cam.npy").astype(np.float64)
    T_tb = np.load(LOGS / "T_table_base.npy").astype(np.float64)
    T_wc = np.linalg.inv(T_tb) @ T_tc
    table_z = float(np.linalg.inv(T_tb)[2, 3])
    return T_wc, table_z


def _rect_model(L, Wd, z, step=0.003, band=0.006):
    """Points on a rectangular rim ring (outer L x Wd, inner band `band`) at height z, centred at 0."""
    xs = np.arange(-L / 2, L / 2 + 1e-9, step)
    ys = np.arange(-Wd / 2, Wd / 2 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ring = (np.abs(X) > L / 2 - band) | (np.abs(Y) > Wd / 2 - band)
    return np.stack([X[ring], Y[ring], np.full(ring.sum(), z)], -1)


def _se2(x, y, yaw, z=0.0):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (x, y, z)
    return T


def _icp2d(data, model, T, iters=50):
    for _ in range(iters):
        m = transform(T, model)
        d, idx = cKDTree(m).query(data)
        keep = d <= np.quantile(d, 0.85)
        P, Q = data[keep, :2], m[idx[keep], :2]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((Q - qc).T @ (P - pc))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        dT = np.eye(4)
        dT[:2, :2], dT[:2, 3] = R, pc - R @ qc
        T = dT @ T
    d, _ = cKDTree(transform(T, model)).query(data)
    return T, float(np.mean(d[d <= np.quantile(d, 0.85)])), float(np.median(d))


def perceive_bin(depth_med, T_wc, table_z, prior_xy=(0.31, -0.214)):
    pts = transform(T_wc, backproject(depth_med, 0.2, 0.75)[0])
    near = np.linalg.norm(pts[:, :2] - np.asarray(prior_xy), axis=1) < 0.13
    seg = pts[near & (pts[:, 2] > table_z + 0.008)]
    h = np.percentile(seg[:, 2], 97) - table_z
    rim = seg[seg[:, 2] > table_z + 0.75 * h]
    # initial size/yaw from the oriented bounding rectangle of the rim points
    (cx, cy), (L, Wd), ang = cv2.minAreaRect(rim[:, :2].astype(np.float32))
    model = _rect_model(L, Wd, table_z + h)
    best = None
    for dyaw in (0, 90, 180, 270):  # rectangle symmetry; keep the best-scoring
        T0 = _se2(cx, cy, np.deg2rad(ang + dyaw))
        T, trimmed, med = _icp2d(rim, model, T0)
        if best is None or trimmed < best[1]:
            best = (T, trimmed, med)
    T, trimmed, med = best
    return dict(T=T, size=(float(L), float(Wd), float(h)), rim_points=len(rim),
                icp_trimmed_mm=trimmed * 1000, icp_median_mm=med * 1000)


def _green_rect_corners(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (50, 120, 50), (85, 255, 255))  # saturated green only (table is dark, low-S)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64), cv2.contourArea(c)


def rgb_camera_pose(rgb, bin_est, T_wc_depth):
    """PnP of the bin's 4 outer rim corners. Returns T_world_rgbcam and reprojection error (px)."""
    img_pts, area = _green_rect_corners(rgb)
    L, Wd, h = bin_est["size"]
    T = bin_est["T"]
    obj = transform(T, np.array([[L / 2, Wd / 2, h * 0], [-L / 2, Wd / 2, 0], [-L / 2, -Wd / 2, 0], [L / 2, -Wd / 2, 0]]))
    obj[:, 2] = T[2, 3] + bin_est["rim_z"]
    best = None
    for shift in range(4):
        for flip in (False, True):
            o = obj[::-1] if flip else obj
            o = np.roll(o, shift, axis=0)
            # iterative PnP seeded with the depth camera pose (colour and depth sit on one sensor);
            # avoids the mirrored planar solution
            T_cw0 = np.linalg.inv(T_wc_depth)
            rv0, _ = cv2.Rodrigues(T_cw0[:3, :3])
            ok, rv, tv = cv2.solvePnP(o, img_pts, RGB_K, None, rv0.copy(), T_cw0[:3, 3].reshape(3, 1).copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            for rv, tv in ([(rv, tv)] if ok else []):
                proj, _ = cv2.projectPoints(o, rv, tv, RGB_K, None)
                err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
                R, _ = cv2.Rodrigues(rv)
                T_cw = np.eye(4)
                T_cw[:3, :3], T_cw[:3, 3] = R, tv.ravel()
                T_wcam = np.linalg.inv(T_cw)
                gap = float(np.linalg.norm(T_wcam[:3, 3] - T_wc_depth[:3, 3]))
                score = err + 100 * gap  # prefer the solution consistent with the depth camera position
                if best is None or score < best[0]:
                    best = (score, T_wcam, err, gap)
    _, T_wcam, err, gap = best
    return dict(T=T_wcam, reproj_px=err, offset_to_depth_cam_m=gap, green_area_px=area)


def _ray_plane(T_wcam, uv, z):
    d = np.linalg.inv(RGB_K) @ np.array([uv[0], uv[1], 1.0])
    d = T_wcam[:3, :3] @ d
    o = T_wcam[:3, 3]
    t = (z - o[2]) / d[2]
    return o + t * d


def _blob(rgb, lo, hi, near_uv=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cent[k], cv2.minAreaRect(cs[0]), int(stats[k, cv2.CC_STAT_AREA])


def perceive(frames=(0, 172, 1), noise=None, seed=0):
    """Run the full perception chain. noise: optional dict(xy_m=..., yaw_deg=...) added to the cube pose."""
    depth, rgb, _ = load_frames(stride=frames[2])
    depth = depth[frames[0] // frames[2]: frames[1] // frames[2]]
    rgb = rgb[frames[0] // frames[2]: frames[1] // frames[2]]
    med = np.nan_to_num(np.median(np.where(depth > 0, depth, np.nan), axis=0), nan=0.0)
    T_wc, table_z = _poses()
    b = perceive_bin(med, T_wc, table_z)
    b["rim_z"] = b["size"][2] + table_z - b["T"][2, 3]
    cam = rgb_camera_pose(rgb[0], b, T_wc)

    # red cube: centroid + oriented rect in RGB; size from pixel extent at the cube-top plane
    c_uv, rect, area = _blob(rgb[0], (0, 120, 70), (10, 255, 255))
    floor_z = table_z + 0.004  # assumption: bin floor ~4 mm above the table
    s = 0.025
    for _ in range(5):  # fixed-point: cube size <-> top height
        p = _ray_plane(cam["T"], c_uv, floor_z + s)
        dist = np.linalg.norm(p - cam["T"][:3, 3])
        s = float(np.sqrt(area)) * dist / RGB_K[0, 0]
    cube_top = _ray_plane(cam["T"], c_uv, floor_z + s)
    cube_c = cube_top - np.array([0, 0, s / 2])
    # yaw of the cube in world: project the rect's edge direction onto the cube-top plane
    a = np.deg2rad(rect[2])
    p2 = _ray_plane(cam["T"], c_uv + 20 * np.array([np.cos(a), np.sin(a)]), floor_z + s)
    yaw = float(np.arctan2(*(p2 - cube_top)[[1, 0]]) % (np.pi / 2))
    # cross-check against depth: highest depth point within 1.5 cm of the cube centre
    pts = transform(T_wc, backproject(med, 0.2, 0.75)[0])
    near = np.linalg.norm(pts[:, :2] - cube_c[:2], axis=1) < 0.015
    depth_top = float(np.percentile(pts[near, 2], 90)) if near.sum() > 5 else float("nan")

    sticker_uv, _, s_area = _blob(rgb[0], (20, 120, 120), (35, 255, 255))
    place = _ray_plane(cam["T"], sticker_uv, table_z)

    cube = dict(center=cube_c, yaw=yaw, size=s, rgb_area_px=area,
                depth_top_z=depth_top, depth_vs_rgb_top_mm=(depth_top - cube_top[2]) * 1000)
    if noise:
        rng = np.random.default_rng(seed)
        d = noise.get("dir_xy")
        ang = float(np.arctan2(d[1], d[0])) if d is not None else rng.uniform(0, 2 * np.pi)
        cube["center"] = cube["center"] + noise.get("xy_m", 0) * np.array([np.cos(ang), np.sin(ang), 0])
        cube["yaw"] = cube["yaw"] + np.deg2rad(noise.get("yaw_deg", 0))
        cube["injected_noise"] = noise
    return dict(bin=b, rgb_cam=cam, cube=cube, place=dict(center=place, sticker_area_px=s_area), table_z=table_z)


def summary(p):
    b, c, cam = p["bin"], p["cube"], p["rgb_cam"]
    yawb = np.rad2deg(np.arctan2(b["T"][1, 0], b["T"][0, 0]))
    return {
        "bin_pose_xyz_yaw": [*np.round(b["T"][:3, 3], 4).tolist(), round(float(yawb), 1)],
        "bin_size_LWH_m": np.round(b["size"], 4).tolist(),
        "bin_icp_residual_mm(trimmed,median)": [round(b["icp_trimmed_mm"], 1), round(b["icp_median_mm"], 1)],
        "rgb_pnp_reproj_px": round(cam["reproj_px"], 2),
        "rgb_cam_offset_from_depth_cam_m": round(cam["offset_to_depth_cam_m"], 4),
        "cube_center": np.round(c["center"], 4).tolist(),
        "cube_yaw_deg": round(float(np.rad2deg(c["yaw"])), 1),
        "cube_size_m": round(c["size"], 4),
        "cube_top_depth_minus_rgb_mm": round(float(c["depth_vs_rgb_top_mm"]), 1),
        "place_target(sticker)": np.round(p["place"]["center"], 4).tolist(),
    }


if __name__ == "__main__":
    p = perceive()
    print(json.dumps(summary(p), indent=1))
