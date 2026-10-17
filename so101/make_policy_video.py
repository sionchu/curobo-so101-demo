"""videos/isaac_policy_grasp.mp4: Isaac closed-loop grasp-phase runs (replay control, SmolVLA, ACT) with the
policy camera views inset and the physics outcome captioned."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("replay", "Control: recorded human actions replayed open-loop"),
        ("smolvla", "SmolVLA (fine-tuned) closed-loop on rendered images"),
        ("act", "ACT (trained from scratch) closed-loop on rendered images")]
out = Mp4Writer(ROOT / "videos" / "isaac_policy_grasp.mp4", fps=30)
for tag, title in RUNS:
    s = json.loads((ROOT / "logs" / f"policy_isaac_{tag}.json").read_text())["summary"]
    views = np.load(ROOT / "logs" / f"policy_views_{tag}.npy")
    frames = sorted((ROOT / "isaac" / f"frames_policy_{tag}").glob("*.jpg"))
    res = (f"physics result: {'GRASP + LIFT OK' if s['success'] else 'no grasp'}  |  max brick lift "
           f"{s['max_brick_rise_mm']:.0f} mm, steps lifted >30 mm: {s['steps_lifted_over_30mm']}")
    for i, fp in enumerate(frames):
        im = Image.open(fp).convert("RGB")
        v = Image.fromarray(views[min(len(views) - 1, int(i / len(frames) * len(views)))]).resize((480, 180))
        im.paste(v, (im.width - 490, im.height - 190))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, im.width, 64], fill=(255, 255, 255, 200))
        d.text((12, 6), "Held-out episode 46, grasp phase in Isaac Sim (PhysX): " + title, fill=(10, 10, 10), font=_font(20))
        d.text((12, 36), res, fill=(0, 110, 0) if s["success"] else (170, 0, 0), font=_font(18))
        d.text((im.width - 488, im.height - 212), "policy cameras (sim): up | side", fill=(20, 20, 20), font=_font(15))
        out.write(np.asarray(im))
out.close()
