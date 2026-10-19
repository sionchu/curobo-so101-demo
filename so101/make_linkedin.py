"""videos/linkedin_cut_en.mp4: ~70 s social cut focused on cuRobo planning and policy training (all footage made here)."""

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
out = Mp4Writer(V / "linkedin_cut_en.mp4", fps=FPS)
_cache = {}


def frames(name):
    if name not in _cache:
        with av.open(str(V / name)) as c:
            _cache[name] = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    return _cache[name]


def caption(img, text, step=None):
    im = Image.fromarray(img).convert("RGB").resize((W, H))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, H - 78, W, H], fill=(10, 14, 22, 225))
    if step:
        d.rectangle([0, H - 78, 10, H], fill=(118, 185, 0, 255))
        d.text((28, H - 70), step, fill=(118, 185, 0), font=_font(20))
    d.text((28, H - 44 if step else H - 56), text, fill=(255, 255, 255), font=_font(28))
    return np.asarray(im)


def seg(name, t0, t1, speed, caps, step=None):
    """Source seconds [t0, t1) at `speed`x; caps = [(source_second, text)] (latest one before the frame wins)."""
    fr = frames(name)
    i = t0 * FPS
    while i < min(t1 * FPS, len(fr)):
        txt = [c for s, c in caps if s <= i / FPS][-1]
        out.write(caption(fr[int(i)], txt, step))
        i += speed


def card(lines, seconds, sub=None):
    im = Image.new("RGB", (W, H), (12, 16, 24))
    d = ImageDraw.Draw(im)
    y = 200
    for k, l in enumerate(lines):
        f = _font(46 if k == 0 else 30)
        d.text((80, y), l, fill=(255, 255, 255) if k == 0 else (200, 210, 225), font=f)
        y += 70 if k == 0 else 48
    if sub:
        d.text((80, 600), sub, fill=(118, 185, 0), font=_font(22))
    a = np.asarray(im)
    for _ in range(int(seconds * FPS)):
        out.write(a)


def gen_grid(seconds, step):
    eps = sorted(p for p in (ROOT / "data" / "sim_raw").glob("ep_*") if (p / "data.npz").exists())[:12]
    tiles = [sorted((e / "up").glob("*.jpg")) for e in eps]
    tw, th = W // 4, H // 3
    for k in range(int(seconds * FPS)):
        can = Image.new("RGB", (W, H))
        for j, fr in enumerate(tiles):
            can.paste(Image.open(fr[min(len(fr) - 1, k * 4)]).resize((tw, th)), ((j % 4) * tw, (j // 4) * th))
        out.write(caption(np.asarray(can), "cuRobo as the teacher: 120 physics-verified pick & place demos", step))


card(["NVIDIA cuRobo + a learned VLA policy",
      "SO-101 arm pick & place, simulated in Isaac Sim",
      "Everything built and run on one RTX 4070 Ti"], 3.5,
     "cuRobo GPU motion planning  ·  sim data generation  ·  SmolVLA fine-tuning")
# 1) cuRobo planning in physics
seg("isaac_harness_obstacles.mp4", 0, 27.2, 2.2,
    [(0, "GPU collision-free motion planning: ~0.1 s per plan"),
     (8, "Grasp checked from physics: missed by 16 mm -> back off, re-plan, retry"),
     (17, "Lifted -> an obstacle drops into the path"),
     (21, "Detected -> re-planned from the current state in 0.14 s, 0 contacts")], "1  cuRobo motion planning")
# 2) cuRobo generates training data
gen_grid(7, "2  Simulation data")
# 3) training
loss = np.asarray(Image.open(V / "sim_finetune_loss.png").convert("RGB"))
for _ in range(int(3.5 * FPS)):
    out.write(caption(np.pad(np.asarray(Image.fromarray(loss).resize((1100, 620))), ((20, 80), (90, 90), (0, 0))),
                      "Fine-tuned SmolVLA (vision-language-action) on real + simulated demos", "3  Policy training"))
seg("sim_finetune_compare.mp4", 0, 31.6, 2,
    [(0, "Same 20 unseen test scenes, camera images in -> joint commands out"),
     (8, "Real data only: 0/20   ->   + simulated cuRobo demos: 7/20   (cuRobo: 20/20)")], "3  Policy training")
# 4) full chain
seg("full_chain.mp4", 0, 11.5, 1.3,
    [(0, "cuRobo brings the arm in -> the learned policy grasps"),
     (5.5, "A box appears mid-transport -> cuRobo re-plans around it")], "4  Planner + learned policy")
seg("full_chain.mp4", 11.5, 30.7, 2.2,
    [(11.5, "Grasp failed -> detected from physics -> recovered"),
     (22, "Policy retries and succeeds -> cuRobo delivers")], "4  Planner + learned policy")
card(["Results (simulation, 20 test scenes)",
      "cuRobo + learned grasp: 16/20 delivered   ·   cuRobo alone: 17/20",
      "Policy alone, end to end: 7/20   ·   re-planning: ~0.1 s",
      "Next: real SO-101 hardware (not tested yet)"], 5,
     "cuRobo · Isaac Sim 6 · LeRobot SmolVLA · RTX 4070 Ti")
out.close()
