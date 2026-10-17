"""Compose isaac/frames_run/*.jpg + the harness step log into videos/isaac_harness_obstacles.mp4."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "fault"
log = json.loads((ROOT / "logs" / f"isaac_harness_{tag}.json").read_text())
caps = sorted(log["captions"], key=lambda c: c["frame"])
frames = sorted((ROOT / "isaac" / "frames_run").glob("*.jpg"))
title = "Isaac Sim PhysX + cuRobo: perceive -> plan -> execute -> verify (physics) -> recover"
legend = "orange-red box = obstacle that appears mid-transport | blue wall = static obstacle | green disc = place target"
out = Mp4Writer(ROOT / "videos" / f"isaac_harness_obstacles{'' if tag == 'fault' else '_' + tag}.mp4", fps=30)
f_title, f_row = _font(22), _font(18)
ci = 0
for i, fp in enumerate(frames):
    while ci + 1 < len(caps) and caps[ci + 1]["frame"] <= i:
        ci += 1
    im = Image.open(fp).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    rows = (caps[ci]["lines"] if caps and caps[0]["frame"] <= i else ["starting"])[-7:]
    d.rectangle([0, 0, im.width, 58], fill=(255, 255, 255, 190))
    d.text((12, 6), title, fill=(10, 10, 10), font=f_title)
    d.text((12, 34), legend, fill=(60, 60, 60), font=_font(15))
    y0 = im.height - 14 - 22 * len(rows)
    d.rectangle([0, y0 - 8, 640, im.height], fill=(255, 255, 255, 190))
    for k, r in enumerate(rows):
        col = (180, 0, 0) if "FAIL" in r else (0, 110, 0) if "SUCCESS" in r else (20, 20, 20)
        d.text((12, y0 + 22 * k), r, fill=col, font=_font(16))
    d.text((im.width - 190, im.height - 30), f"sim t = {i / 30:5.1f} s", fill=(255, 255, 255), font=f_row)
    out.write(np.asarray(im))
out.close()
