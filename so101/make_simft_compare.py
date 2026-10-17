"""videos/sim_finetune_compare.mp4: same Isaac scenarios, real-only SmolVLA vs sim co-trained SmolVLA vs cuRobo expert."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("real_only", "SmolVLA, real data only"), ("sim_ft", "SmolVLA + sim co-training"), ("curobo_expert", "cuRobo expert (upper bound)")]
SCEN = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else "0,5,10,15".split(","))]
SPEED = 2
res = {t: json.loads((ROOT / "logs" / f"eval_closedloop_{t}.json").read_text()) for t, _ in RUNS}


def segment(tag, i):
    r = res[tag][i]
    prev = res[tag][i - 1]["frame_end"] if i % 5 and i > 0 else 0
    d = ROOT / "isaac" / r["frames_dir"]
    files = sorted(d.glob("*.jpg"))
    return files[prev:r["frame_end"]], r


out = Mp4Writer(ROOT / "videos" / "sim_finetune_compare.mp4", fps=30)
for i in SCEN:
    segs = {t: segment(t, i) for t, _ in RUNS}
    n = max(len(s[0]) for s in segs.values())
    for k in range(0, n, SPEED):
        can = Image.new("RGB", (1280, 720), (15, 15, 20))
        for j, (t, title) in enumerate(RUNS):
            files, r = segs[t]
            im = Image.open(files[min(k, len(files) - 1)]).convert("RGB").resize((640, 360)) if files else Image.new("RGB", (640, 360))
            x, y = (j % 2) * 640, (j // 2) * 360
            can.paste(im, (x, y))
            d = ImageDraw.Draw(can, "RGBA")
            d.rectangle([x, y, x + 640, y + 50], fill=(255, 255, 255, 200))
            d.text((x + 8, y + 4), title, fill=(10, 10, 10), font=_font(18))
            ok = r["success"]
            d.text((x + 8, y + 27), ("SUCCESS (brick in box)" if ok else ("lifted, not placed" if r["lifted"] else "no grasp"))
                   + (f" | max lift {r['max_rise_mm']:.0f} mm" if r.get("max_rise_mm") is not None else ""),
                   fill=(0, 120, 0) if ok else (170, 0, 0), font=_font(15))
        d = ImageDraw.Draw(can)
        tot = {t: sum(x["success"] for x in res[t]) for t, _ in RUNS}
        d.text((660, 380), f"Isaac Sim closed-loop, scenario {i} of 20 (2x speed)", fill=(255, 255, 255), font=_font(20))
        for j, (t, title) in enumerate(RUNS):
            d.text((660, 420 + 30 * j), f"{title}: {tot[t]}/{len(res[t])} successes overall", fill=(220, 220, 230), font=_font(17))
        d.text((660, 530), "Same seeded scenarios for every method; start at rest pose.", fill=(160, 160, 170), font=_font(15))
        d.text((660, 552), "Success = brick resting inside the box (physics state).", fill=(160, 160, 170), font=_font(15))
        out.write(np.asarray(can))
out.close()
