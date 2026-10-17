"""Shared helpers for the SO-101 raw-depth scene (CarolinePascal/so101_depth_raw).

Assumptions (the dataset ships no intrinsics/extrinsics):
- Depth is float32 metres in the depth sensor frame, *not* registered to the RGB image.
- Depth intrinsics = RealSense D435 depth stream defaults at 640x480 (FX, FY, CX, CY below).
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data" / "so101_depth_raw"
FX = FY = 385.0
CX, CY = 320.0, 240.0
H, W = 480, 640


def _img(cell) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])))


def load_frames(stride: int = 1):
    """Return (depth[N,H,W] m, rgb[N,H,W,3], state[N,6]) for every `stride`-th frame."""
    t = pq.read_table(ROOT / "data/chunk-000/file-000.parquet")
    rows = t.to_pylist()[::stride]
    depth = np.stack([_img(r["observation.depth.front"]).astype(np.float32) for r in rows])
    rgb = np.stack([_img(r["observation.images.front"])[..., :3] for r in rows])
    state = np.stack([np.asarray(r["observation.state"], np.float32) for r in rows])
    return depth, rgb, state


def backproject(depth: np.ndarray, zmin=0.2, zmax=1.5):
    """Depth image (H,W) -> (N,3) camera-frame points (x right, y down, z forward) and pixel idx."""
    v, u = np.nonzero((depth > zmin) & (depth < zmax))
    z = depth[v, u]
    pts = np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], -1)
    return pts.astype(np.float32), (v, u)


def fit_plane(pts: np.ndarray, iters=400, thresh=0.006, seed=0):
    """RANSAC plane; returns (n, d) with n.p + d = 0, normal pointing toward the camera."""
    rng = np.random.default_rng(seed)
    best, best_n, best_d = 0, None, None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        cnt = np.count_nonzero(np.abs(pts @ n + d) < thresh)
        if cnt > best:
            best, best_n, best_d = cnt, n, d
    inl = np.abs(pts @ best_n + best_d) < thresh
    # refine with least squares on inliers
    p = pts[inl]
    c = p.mean(0)
    n = np.linalg.eigh(np.cov((p - c).T))[1][:, 0]  # smallest-variance direction
    d = -n @ c
    if d < 0:  # camera at origin should be on the positive side
        n, d = -n, -d
    return n.astype(np.float32), float(d), inl.mean()


def camera_to_table_frame(n: np.ndarray, d: float) -> np.ndarray:
    """4x4 T_table_cam: table plane z=0, +z toward camera, x roughly along image u."""
    z = n
    x = np.array([1.0, 0, 0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: table axes in camera coords
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = (0.0, 0.0, d)  # origin = camera's foot point on the table; camera at height d
    return T


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]
