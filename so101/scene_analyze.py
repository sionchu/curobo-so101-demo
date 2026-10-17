"""Fit the table plane and map what stands on it; saves a height-map image for inspection."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from depth_scene import backproject, camera_to_table_frame, fit_plane, load_frames, transform  # noqa: E402

depth, rgb, state = load_frames(stride=1)
med = np.median(np.where(depth > 0, depth, np.nan), axis=0)
med = np.nan_to_num(med, nan=0.0)
pts, (v, u) = backproject(med)
n, d, frac = fit_plane(pts)
print(f"plane normal(cam) {n.round(3)}, camera height above table {d:.3f} m, inliers {frac*100:.0f}%")
T = camera_to_table_frame(n, d)
pt = transform(T, pts)
print("table-frame z range p1/p50/p99:", np.percentile(pt[:, 2], [1, 50, 99]).round(3))

# height map, 5 mm cells
res = 0.005
xy_min = pt[:, :2].min(0)
ij = ((pt[:, :2] - xy_min) / res).astype(int)
hm = np.full(ij.max(0) + 1, -1.0, np.float32)
np.maximum.at(hm, (ij[:, 0], ij[:, 1]), pt[:, 2])
above = hm > 0.015
lab, nlab = ndimage.label(ndimage.binary_closing(above, iterations=2))
sizes = ndimage.sum(above, lab, range(1, nlab + 1))
print(f"{nlab} components above 1.5 cm; largest:")
for k in np.argsort(-sizes)[:6]:
    m = lab == k + 1
    cells = np.argwhere(m)
    c = cells.mean(0) * res + xy_min
    print(f"  comp {k+1}: area {sizes[k]*res*res*1e4:.0f} cm2, centre xy {c.round(3)}, "
          f"max h {hm[m].max():.3f} m, bbox {((cells.max(0)-cells.min(0))*res).round(3)}")
vis = np.clip(hm / 0.25, 0, 1)
vis[hm < 0] = 0
img = (np.stack([vis, vis, vis], -1) * 255).astype(np.uint8)
img[above & (lab > 0)] = (img[above & (lab > 0)] * 0.5 + np.array([120, 0, 0])).astype(np.uint8)
Image.fromarray(img.transpose(1, 0, 2)[::-1]).resize((hm.shape[0] * 2, hm.shape[1] * 2), 0).save("logs/heightmap.png")
np.save("logs/T_table_cam.npy", T)
print("saved logs/heightmap.png (x right, y up)")
