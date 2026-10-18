"""videos/full_chain.mp4: a few full-chain episodes (Isaac viewport) with the harness step log on screen.

    .venv\\Scripts\\python.exe so101\\make_chain_video.py chain_policy 3,7,12
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
tag, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
MAX_S, FPS = 58, 30
res = {r["scenario"]: r for r in json.loads((ROOT / "logs" / f"{tag}.json").read_text())}
GRASP = {"policy": "SmolVLA grasp (1st try)", "policy_retry": "SmolVLA grasp (retry after recovery)",
         "curobo_fallback": "cuRobo fallback grasp", "curobo": "cuRobo scripted grasp", None: "no grasp"}

segs = []
for i in ids:
    r = res[i]
    files = sorted((ROOT / "isaac" / r["frames_dir"]).glob("*.jpg"))[r["frame_start"]:r["frame_end"]]
    segs.append((r, files))
speed = max(1, math.ceil(sum(len(f) for _, f in segs) / (MAX_S * FPS - 45 * len(segs))))

out = Mp4Writer(ROOT / "videos" / "full_chain.mp4", fps=FPS)
for r, files in segs:
    for k in range(0, len(files), speed):
        fr = r["frame_start"] + k
        im = Image.open(files[k]).convert("RGB").resize((1280, 720))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, 1280, 40], fill=(255, 255, 255, 215))
        d.text((10, 8), f"Full chain in Isaac Sim PhysX, scenario {r['scenario']} of 20 ({speed}x speed): "
               "cuRobo approach -> SmolVLA grasp -> verify -> recover -> cuRobo transport around obstacles",
               fill=(10, 10, 10), font=_font(17))
        rows = [e for e in r["log"] if e["frame"] <= fr]
        d.rectangle([790, 44, 1280, 44 + 8 + 19 * 12], fill=(0, 0, 0, 140))
        for j, e in enumerate(rows[-12:]):
            d.text((798, 50 + 19 * j), f"{e['step']:<14s} {'OK  ' if e['ok'] else 'FAIL'} {e['brief']}"[:62],
                   fill=(140, 255, 140) if e["ok"] else (255, 150, 150), font=_font(13))
        if k + speed >= len(files):
            msg = ("SUCCESS: brick in box" if r["success"] else "FAILED") + f" | {GRASP[r.get('grasp_path')]}" \
                  + f" | obstacle contacts {r.get('robot_obstacle_contacts', 0)}"
            for _ in range(45):
                d.rectangle([660, 650, 1270, 700], fill=(20, 120, 40, 220) if r["success"] else (150, 30, 30, 220))
                d.text((672, 662), msg, fill=(255, 255, 255), font=_font(18))
                out.write(np.asarray(im))
        else:
            out.write(np.asarray(im))
out.close()
print("speed", speed)
