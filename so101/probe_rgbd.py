"""Decode one RGB + depth frame from the SO-101 RealSense dataset and save a preview."""

import json
import math
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data" / "so101_rgbd"
info = json.loads((ROOT / "meta/info.json").read_text())
dinfo = info["features"]["observation.images.cam_external_depth"]["info"]
DMIN, DMAX, SHIFT = dinfo["video.depth_min"], dinfo["video.depth_max"], dinfo["video.shift"]
QMAX = 4095


def dequantize(q: np.ndarray) -> np.ndarray:
    """LeRobot log depth quantization inverse (lerobot.datasets.depth_utils.dequantize_depth)."""
    lmin, lmax = math.log(DMIN + SHIFT), math.log(DMAX + SHIFT)
    d = np.exp(q.astype(np.float32) * (lmax - lmin) / QMAX + lmin) - SHIFT
    return np.clip(d, DMIN, DMAX)


def first_frame(path: Path, fmt: str):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        print(path.name, "codec", s.codec_context.name, "pix_fmt", s.codec_context.pix_fmt, "frames", s.frames)
        for f in c.decode(s):
            return f.to_ndarray(format=fmt)


if __name__ == "__main__":
    eps = pq.read_table(ROOT / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    print(eps[[c for c in eps.columns if c in ("episode_index", "length") or "file_index" in c or "from_timestamp" in c]].head(4).to_string())
    rgb = first_frame(ROOT / "videos/observation.images.cam_external/chunk-000/file-000.mp4", "rgb24")
    q = first_frame(ROOT / "videos/observation.images.cam_external_depth/chunk-000/file-000.mp4", "gray12le")
    print("rgb", rgb.shape, rgb.dtype, "| depth codes", q.shape, q.dtype, "min", q.min(), "max", q.max())
    d = dequantize(q)
    valid = (d > DMIN + 1e-3) & (d < DMAX - 1e-3)
    print("depth m: valid %.1f%%, p5 %.3f p50 %.3f p95 %.3f" % (100 * valid.mean(), *np.percentile(d[valid], [5, 50, 95])))
    vis = np.clip((d - 0.2) / (1.2 - 0.2), 0, 1)
    vis = (np.stack([vis] * 3, -1) * 255).astype(np.uint8)
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/rgbd_preview.png")
    Image.fromarray(np.concatenate([rgb, vis], axis=1)).save(out)
    print("saved", out)
