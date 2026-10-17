"""Inspect CarolinePascal/so101_depth_raw: depth dtype/units, ranges, joint states; save previews."""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data" / "so101_depth_raw"
OUT = Path(__file__).resolve().parent.parent / "logs"

t = pq.read_table(ROOT / "data/chunk-000/file-000.parquet")
print("columns:", t.column_names, "rows:", t.num_rows)
stats = json.loads((ROOT / "meta/stats.json").read_text(encoding="utf-8-sig"))
print("depth stats:", {k: str(v)[:80] for k, v in stats.get("observation.depth.front", {}).items()})
row0 = t.slice(0, 1).to_pylist()[0]
d0 = row0["observation.depth.front"]
print("depth cell type:", type(d0), (d0.keys() if isinstance(d0, dict) else ""))


def load_img(cell):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(b)))


for i in (0, 85, 171):
    r = t.slice(i, 1).to_pylist()[0]
    d = load_img(r["observation.depth.front"])
    rgb = load_img(r["observation.images.front"])
    nz = d[d > 0]
    print(f"frame {i}: depth {d.dtype} {d.shape} zero%={100*(d==0).mean():.1f} "
          f"p5/p50/p95={np.percentile(nz,[5,50,95]).round(1)} max={d.max()} | rgb {rgb.shape} "
          f"| state {np.round(r['observation.state'],1)}")
    vis = np.clip((d.astype(np.float32) - np.percentile(nz, 2)) / (np.percentile(nz, 98) - np.percentile(nz, 2)), 0, 1)
    vis = (np.stack([vis] * 3, -1) * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb[..., :3], vis], 1)).save(OUT / f"depth_raw_preview_{i}.png")
