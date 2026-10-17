"""Concatenate the recorded clips into videos/demo_highlights.mp4 with a title card per segment."""

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

V = Path(__file__).resolve().parent.parent / "videos"
FPS, W, H = 30, 1280, 720
SEGMENTS = [  # (file, title, subtitle, max seconds)
    ("replay_ep5.mp4", "1. Real SO-101 recording, replayed in cuRobo",
     "LeRobot dataset video + joints; green ghost = cuRobo re-plan through recorded waypoints", 16),
    ("depth_plan.mp4", "2. Planning in a scene rebuilt from real depth",
     "Depth -> cuRobo TSDF/ESDF; SO-101 placed by ICP; planner reaches into the real bin", 22),
    ("mpc.mp4", "3. MPC tracking a moving target in the same scene",
     "Target moved by script; MPC keeps clearance from the reconstructed geometry", 18),
]


def title_card(title, sub):
    im = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(im)
    d.text((80, 300), title, fill=(255, 255, 255), font=_font(44))
    d.text((80, 370), sub, fill=(190, 200, 215), font=_font(24))
    d.text((80, 640), "SO-101 + NVIDIA cuRobo | recorded on RTX 4070 Ti (Windows)", fill=(120, 130, 145), font=_font(18))
    return np.asarray(im)


def frames(path, max_s):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rep = max(1, round(FPS / float(s.average_rate)))  # 15 fps clip -> duplicate to 30 fps
        n = 0
        for f in c.decode(s):
            img = np.asarray(f.to_image().resize((W, H)))
            for _ in range(rep):
                if n >= max_s * FPS:
                    return
                yield img
                n += 1


out = Mp4Writer(V / "demo_highlights.mp4", fps=FPS)
for fname, title, sub, max_s in SEGMENTS:
    if not (V / fname).exists():
        print("missing", fname)
        continue
    card = title_card(title, sub)
    for _ in range(int(1.5 * FPS)):
        out.write(card)
    for img in frames(V / fname, max_s):
        out.write(img)
out.close()
