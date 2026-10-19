"""videos/full_journey.mp4: title card per stage + the existing result clips + process footage, 1280x720, 30 fps."""

import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "videos"
W, H, FPS = 1280, 720, 30
out = Mp4Writer(V / "full_journey.mp4", fps=FPS)


def fit(img):
    """Letterbox any image into 1280x720."""
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    im = im.convert("RGB")
    s = min(W / im.width, H / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    can = Image.new("RGB", (W, H), (0, 0, 0))
    can.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.asarray(can)


OURS, EXT = (20, 130, 60), (200, 110, 10)  # green = made on this PC, orange = someone else's recording
SIM = [(0, 1280, OURS, "MADE HERE: simulated + rendered on this PC (Isaac Sim / cuRobo)")]


def tag(f, tags):
    """Source badges at the bottom of screen regions: (x0, x1, colour, text)."""
    im = Image.fromarray(f)
    d = ImageDraw.Draw(im, "RGBA")
    for x0, x1, col, text in tags:
        w = min(x1 - x0 - 8, 11 + 8 * len(text))
        d.rectangle([x0 + 4, H - 30, x0 + 4 + w, H - 6], fill=(*col, 235))
        d.text((x0 + 10, H - 27), text, fill=(255, 255, 255), font=_font(14))
    return np.asarray(im)


def card(title, lines, note=None, seconds=3.5):
    im = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(im)
    d.text((70, 170), title, fill=(255, 255, 255), font=_font(42))
    for i, l in enumerate(lines):
        d.text((70, 250 + 38 * i), l, fill=(200, 210, 225), font=_font(24))
    if note:
        d.text((70, 640), note, fill=(150, 160, 175), font=_font(17))
    f = np.asarray(im)
    for _ in range(int(seconds * FPS)):
        out.write(f)


def clip(name, max_s=None, speed=1.0, tags=SIM):
    p = V / name
    if not p.exists():
        print("missing", name)
        return
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        src_fps = float(s.average_rate)
        step = src_fps / FPS * speed  # source frames per output frame
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    n_out = int(len(frames) / step)
    if max_s:
        n_out = min(n_out, int(max_s * FPS))
    for i in range(n_out):
        out.write(tag(fit(frames[min(len(frames) - 1, int(i * step))]), tags))


def still(name, seconds=4.0, tags=SIM):
    p = V / name
    if p.exists():
        f = tag(fit(Image.open(p)), tags)
        for _ in range(int(seconds * FPS)):
            out.write(f)


def gen_timelapse(seconds=12.0, n=12):
    """Process footage: 12 generated sim episodes (policy 'up' camera) playing side by side at 3x."""
    eps = sorted(p for p in (ROOT / "data" / "sim_raw").glob("ep_*") if (p / "data.npz").exists())[:n]
    if not eps:
        return
    tiles = [sorted((e / "up").glob("*.jpg")) for e in eps]
    cols, rows = 4, 3
    tw, th = W // cols, (H - 40) // rows
    for k in range(int(seconds * FPS)):
        can = Image.new("RGB", (W, H), (10, 10, 10))
        for j, fr in enumerate(tiles):
            im = Image.open(fr[min(len(fr) - 1, k * 3)]).resize((tw, th))
            can.paste(im, ((j % cols) * tw, 40 + (j // cols) * th))
        d = ImageDraw.Draw(can)
        d.text((10, 8), f"Process: cuRobo expert demos generated in Isaac Sim (12 of the episodes, 3x speed)",
               fill=(255, 255, 255), font=_font(20))
        out.write(tag(np.asarray(can), SIM))


summary = ROOT / "logs" / "sim_finetune_summary.json"
s = json.loads(summary.read_text()) if summary.exists() else {}
card("SO-101 pick & place: cuRobo + Isaac Sim + SmolVLA",
     ["Built and run on one Windows PC (RTX 4070 Ti): simulation, planning, data, training, evaluation",
      "Main part: everything simulated and rendered here.  External recordings: reference section at the end."],
     "Badges: green = made here; orange = someone else's public recording (used as reference input).", 5)
if (V / "full_chain.mp4").exists():
    card("Result: full chain in Isaac Sim physics",
         ["cuRobo approach -> SmolVLA grasp -> physics verify -> recover -> cuRobo transport around obstacles",
          s.get("line9a", ""), s.get("line9b", "")])
    clip("full_chain.mp4", max_s=40)
card("How the grasp policy got there: simulation data",
     [s.get("line1", ""), s.get("line2", ""), "Only physics-verified successes kept (120/120 of 118 attempts + 2 tests)"])
gen_timelapse()
still("sim_finetune_loss.png", 4)
clip("sim_finetune_compare.mp4", max_s=32)
if (V / "learning_story.mp4").exists():  # its last ~42 s repeat full_chain.mp4, shown in stage 9
    card("8b. What the training changed, checkpoint by checkpoint",
         ["Training data (real + sim demos and their gripper paths), loss curves from the logs,",
          "held-out predictions, and the same Isaac scenarios run by every checkpoint"])
    clip("learning_story.mp4", max_s=69, tags=[])  # carries its own per-section badges (real grid is EXTERNAL)
card("Planner harness in physics: obstacles and recovery",
     ["Isaac Sim PhysX gripper; a box appears mid-transport -> detected -> replanned in 0.137 s",
      "9.8 mm clearance, 0 contacts; injected 25 mm perception error caught and recovered"])
clip("isaac_harness_obstacles.mp4", max_s=27)
card("Summary", [s.get("final", "See README for all numbers, assumptions and unverified items.")], seconds=4)
card("Reference: inputs from other people's recordings",
     ["Public Hugging Face datasets (not recorded here; no real robot was run for this project):",
      "lerobot/svla_so101_pickplace - real SO-101 demos: task layout + real-data policy training",
      "CarolinePascal/so101_depth_raw - real RGB-D: depth scene + FoundationPose check",
      "Shown briefly: left/overlay = computed here, orange panels = their recording."], seconds=5)
clip("replay_ep5.mp4", max_s=6, tags=[(0, 800, OURS, "MADE HERE: cuRobo 3D view"), (800, 1280, EXT, "REFERENCE: lerobot/svla_so101_pickplace")])
clip("depth_plan.mp4", max_s=6, tags=[(0, 800, OURS, "MADE HERE: cuRobo 3D view"), (800, 1280, EXT, "REFERENCE: CarolinePascal/so101_depth_raw")])
clip("harness_pickplace.mp4", max_s=6, tags=[(0, 800, OURS, "MADE HERE: cuRobo 3D view"), (800, 1280, EXT, "REFERENCE: CarolinePascal/so101_depth_raw")])
clip("foundationpose_overlay.mp4", max_s=5, tags=[(160, 1120, EXT, "REFERENCE video: CarolinePascal/so101_depth_raw  |  MADE HERE: pose overlay")])
out.close()
