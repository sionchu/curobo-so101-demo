"""Estimate the SO-101 base pose on the table by ICP of the robot mesh (recorded joints)
against the depth point cloud. Optimises x, y, yaw and a small z offset.

Writes logs/T_table_base.npy and prints the fit residual.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from depth_scene import backproject, load_frames, transform  # noqa: E402
from replay import JOINTS, URDF, to_radians  # noqa: E402


def robot_mesh(q6: np.ndarray) -> trimesh.Trimesh:
    """Robot visual mesh in base_link frame at joint config q6 (radians)."""
    u = yourdfpy.URDF.load(
        str(URDF), load_meshes=True, build_scene_graph=True,
        filename_handler=lambda fname: str(URDF.parent / fname.replace("package://", "")),
    )
    u.update_cfg(dict(zip(JOINTS, q6.tolist())))
    return u.scene.to_mesh() if hasattr(u.scene, "to_mesh") else u.scene.dump(concatenate=True)


def visible_from_top(pts: np.ndarray, cell=0.004) -> np.ndarray:
    """Keep the highest model point per xy cell (what a top-down camera would see)."""
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((-pts[:, 2], key))
    _, first = np.unique(key[order], return_index=True)
    return pts[order[first]]


def se2(x, y, yaw, z=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (x, y, z)
    return T


def icp(data, model_pts, T0, iters=40, trim=0.8):
    """Point-to-point ICP restricted to (x, y, yaw, z). data: observed points (table frame)."""
    T = T0.copy()
    for _ in range(iters):
        m = transform(T, model_pts)
        tree = cKDTree(m)
        dist, idx = tree.query(data)
        keep = dist <= np.quantile(dist, trim)
        P, Q = data[keep], m[idx[keep]]
        # solve 2D rotation + translation in xy, z offset separately
        pc, qc = P[:, :2].mean(0), Q[:, :2].mean(0)
        Hm = (Q[:, :2] - qc).T @ (P[:, :2] - pc)
        U, _, Vt = np.linalg.svd(Hm)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        dT = np.eye(4)
        dT[:2, :2] = R
        dT[:2, 3] = pc - R @ qc
        dT[2, 3] = np.clip(np.median(P[:, 2] - Q[:, 2]), -0.01, 0.01)
        T = dT @ T
    m = transform(T, model_pts)
    dist, _ = cKDTree(m).query(data)
    return T, float(np.mean(dist[dist <= np.quantile(dist, trim)])), float(np.median(dist))


if __name__ == "__main__":
    depth, rgb, state = load_frames(stride=1)
    T_tc = np.load("logs/T_table_cam.npy")
    med = np.nan_to_num(np.median(np.where(depth > 0, depth, np.nan), axis=0), nan=0.0)
    pts_t = transform(T_tc, backproject(med)[0])

    # robot cluster: points >1.5 cm above the table near the robot component found by scene_analyze
    near = np.linalg.norm(pts_t[:, :2] - np.array([0.13, -0.141]), axis=1) < 0.2
    data = pts_t[near & (pts_t[:, 2] > 0.015)]
    print("robot cluster points:", len(data))

    q6 = to_radians(state[:1])[0]
    print("recorded joints (rad, frame 0):", q6.round(3))
    mesh = robot_mesh(q6)
    model = visible_from_top(mesh.sample(60000).astype(np.float32))
    print("model visible points:", len(model), "model height %.3f m" % model[:, 2].max(),
          "| observed height %.3f m" % data[:, 2].max())

    c = data[:, :2].mean(0) - model[:, :2].mean(0)
    best = None
    for yaw in np.deg2rad(np.arange(0, 360, 15)):
        Ry = se2(0, 0, yaw)
        mc = transform(Ry, model)[:, :2].mean(0)
        T0 = se2(*(data[:, :2].mean(0) - mc), yaw)
        T, trimmed, med_d = icp(data, model, T0)
        if best is None or trimmed < best[1]:
            best = (T, trimmed, med_d, np.rad2deg(yaw))
    T, trimmed, med_d, yaw0 = best
    yaw = np.rad2deg(np.arctan2(T[1, 0], T[0, 0]))
    print(f"best fit: base xy {T[:2, 3].round(3)} z {T[2, 3]:.3f} yaw {yaw:.1f} deg "
          f"(init {yaw0:.0f}); residual trimmed-mean {trimmed*1000:.1f} mm, median {med_d*1000:.1f} mm")
    np.save("logs/T_table_base.npy", T.astype(np.float32))
