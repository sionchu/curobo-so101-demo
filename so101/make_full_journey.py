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


card("SO-101 + NVIDIA cuRobo on one Windows PC",
     ["Real LeRobot data -> cuRobo planning -> Isaac Sim physics -> policy learning",
      "Everything below ran on an RTX 4070 Ti (Windows 11)."],
     "Note: stages 1-7 are shown as result clips; there are no screen recordings of the work itself.", 5)
legend = Image.new("RGB", (W, H), (18, 22, 30))
d_ = ImageDraw.Draw(legend)
d_.text((70, 150), "Whose footage is this?", fill=(255, 255, 255), font=_font(42))
for i_, (col_, t1, t2) in enumerate([
        (EXT, "EXTERNAL", "real robot / camera recordings from public Hugging Face datasets, not made here:"),
        (None, "", "  lerobot/svla_so101_pickplace (LeRobot team)  |  CarolinePascal/so101_depth_raw"),
        (None, "", "  No real robot was run for this project."),
        (OURS, "MADE HERE", "everything simulated, planned, trained or rendered on this PC:"),
        (None, "", "  cuRobo plans, Viser 3D views, Isaac Sim scenes/physics, sim data, trained policies, overlays")]):
    if col_:
        d_.rectangle([70, 250 + 46 * i_, 230, 280 + 46 * i_], fill=col_)
        d_.text((80, 253 + 46 * i_), t1, fill=(255, 255, 255), font=_font(20))
    d_.text((250, 253 + 46 * i_), t2, fill=(210, 220, 232), font=_font(20))
d_.text((70, 640), "Each clip below carries these badges at the bottom of the region they describe.", fill=(150, 160, 175), font=_font(17))
for _ in range(int(6 * FPS)):
    out.write(np.asarray(legend))
card("1-2. Real SO-101 recording, replayed in cuRobo",
     ["lerobot/svla_so101_pickplace: robot joints + 2 real cameras",
      "cuRobo re-plans through recorded waypoints: 0.5 mm at waypoints, 8-10 mm mean gap"])
clip("replay_ep5.mp4", max_s=15, tags=[(0, 800, OURS, "MADE HERE: cuRobo replay + re-plan (3D view)"), (800, 1280, EXT, "EXTERNAL: lerobot/svla_so101_pickplace")])
card("3. Scene rebuilt from real depth",
     ["Plane fit + ICP robot placement (6.5 mm residual), TSDF/ESDF in cuRobo",
      "Plan into the real bin: ~0.1 s, 14 mm clearance; MPC tracks a moving target"])
clip("depth_plan.mp4", max_s=19, tags=[(0, 800, OURS, "MADE HERE: cuRobo scene + plan from that depth"), (800, 1280, EXT, "EXTERNAL: CarolinePascal/so101_depth_raw")])
clip("mpc.mp4", max_s=12, tags=[(0, 800, OURS, "MADE HERE: cuRobo scene + plan from that depth"), (800, 1280, EXT, "EXTERNAL: CarolinePascal/so101_depth_raw")])
card("4. Harness: perceive -> plan -> verify -> recover",
     ["Injected 25 mm perception error caught by grasp verification (25.1 mm)",
      "Re-perceive -> replan -> placed 5.0 mm from target (kinematic execution)"])
clip("harness_pickplace.mp4", max_s=19, tags=[(0, 800, OURS, "MADE HERE: cuRobo scene + plan from that depth"), (800, 1280, EXT, "EXTERNAL: CarolinePascal/so101_depth_raw")])
card("5. Isaac Sim PhysX + obstacles",
     ["Real gripper physics: cube lifted 78 mm; box appears mid-transport",
      "Detected (-7.1 mm on old path) -> replanned in 0.137 s -> 9.8 mm clearance, 0 contacts"])
clip("isaac_harness_obstacles.mp4", max_s=27)
card("6. FoundationPose on real RGB-D",
     ["Built on Windows with a pip-only CUDA 13 toolchain (no admin)",
      "vs ICP/PnP: cube 0.9 mm xy, bin 4.2 mm xy; tracking jitter < 0.6 mm"])
clip("foundationpose_overlay.mp4", max_s=11.5, tags=[(160, 1120, EXT, "EXTERNAL video: CarolinePascal/so101_depth_raw  |  MADE HERE: pose boxes/axes (FoundationPose run on this PC)")])
card("7. Policy learning on real data (ACT, SmolVLA)",
     ["Held-out arm MAE: hold-state 3.19 | ACT 3.96 | SmolVLA 2.00",
      "Closed-loop in Isaac: both failed to grasp (visual sim-to-real gap)"])
still("policy_train_loss.png", 4, tags=[(0, 1280, OURS, "MADE HERE: training on this PC (on the external real dataset)")])
still("policy_offline_ep47.png", 5, tags=[(0, 1280, OURS, "MADE HERE: policy predictions  |  reference curves = EXTERNAL recorded actions (lerobot/svla_so101_pickplace)")])
clip("isaac_policy_grasp.mp4", max_s=24)
summary = ROOT / "logs" / "sim_finetune_summary.json"
s = json.loads(summary.read_text()) if summary.exists() else {}
card("8. Simulation data + co-training",
     [s.get("line1", "cuRobo expert demos in Isaac, mixed with the real data"),
      s.get("line2", "SmolVLA fine-tuned from the real-data checkpoint")])
gen_timelapse()
still("sim_finetune_loss.png", 4)
clip("sim_finetune_compare.mp4", max_s=40)
if (V / "learning_story.mp4").exists():  # its last ~42 s repeat full_chain.mp4, shown in stage 9
    card("8b. What the training changed, checkpoint by checkpoint",
         ["Training data (real + sim demos and their gripper paths), loss curves from the logs,",
          "held-out predictions, and the same Isaac scenarios run by every checkpoint"])
    clip("learning_story.mp4", max_s=63)
if (V / "full_chain.mp4").exists():
    card("9. Full chain: cuRobo + SmolVLA + harness",
         [s.get("line9a", "cuRobo approach -> SmolVLA grasp -> physics verify -> recover -> cuRobo transport"),
          s.get("line9b", "Static wall + a box that appears mid-transport (replan from the measured state)")])
    clip("full_chain.mp4", max_s=40)
card("Summary", [s.get("final", "See README for all numbers, assumptions and unverified items.")], seconds=4)
out.close()
