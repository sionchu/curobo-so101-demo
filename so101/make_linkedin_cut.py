"""videos/linkedin_cut.mp4: ~45 s cut for LinkedIn, cuRobo planning + policy learning only.

    .venv-lerobot\\Scripts\\python.exe so101\\make_linkedin_cut.py

Cuts from the already rendered videos (isaac_harness_obstacles, demo_highlights, learning_story, full_chain);
each clip is shrunk to leave a caption band at the bottom so the burned-in overlays stay readable.
"""

import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V, LOG = ROOT / "videos", ROOT / "logs"
W, H, FPS, BAND = 1280, 720, 30, 84
BG = (12, 14, 20)
EXT, OURS, NV = (200, 110, 10), (20, 130, 60), (118, 185, 0)
out = Mp4Writer(V / "linkedin_cut.mp4", fps=FPS)


def F(size, bold=False):
    for name in (("malgunbd.ttf",) if bold else ()) + ("malgun.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frames(name, a, b):
    with av.open(str(V / name)) as c:
        return [f.to_image() for i, f in enumerate(c.decode(video=0)) if a <= i < b]


def caption_frame(img, tag, text, badge=None):
    """Source frame shrunk into the upper area, caption band below: [tag] text, optional source badge."""
    can = Image.new("RGB", (W, H), BG)
    h = H - BAND
    w = int(img.width * h / img.height)
    can.paste(img.convert("RGB").resize((w, h)), ((W - w) // 2, 0))
    d = ImageDraw.Draw(can)
    tw = d.textlength(tag, font=F(22, True)) + 24
    d.rectangle([24, h + 20, 24 + tw, h + 60], fill=NV if tag.startswith("cuRobo") else (40, 110, 200))
    d.text((36, h + 24), tag, fill=(0, 0, 0) if tag.startswith("cuRobo") else (255, 255, 255), font=F(22, True))
    d.text((44 + tw, h + 22), text, fill=(240, 240, 245), font=F(25))
    if badge:
        bw = d.textlength(badge, font=F(14)) + 16
        d.rectangle([W - 16 - bw, h - 30, W - 16, h - 6], fill=EXT)
        d.text((W - 8 - bw, h - 28), badge, fill=(255, 255, 255), font=F(14))
    return np.asarray(can)


def clip(name, a, b, speed, tag, text, badge=None, hold=0.0):
    fr = frames(name, a, b)
    n = int(len(fr) / speed)
    for i in range(n):
        out.write(caption_frame(fr[min(len(fr) - 1, int(i * speed))], tag, text, badge))
    last = caption_frame(fr[-1], tag, text, badge)
    for _ in range(int(hold * FPS)):
        out.write(last)


def card(lines, sec, foot=None):
    can = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(can)
    y = 230
    for text, size, col in lines:
        d.text((90, y), text, fill=col, font=F(size, size >= 34))
        y += int(size * 1.7)
    if foot:
        d.text((90, 640), foot, fill=(140, 145, 160), font=F(17))
    a = np.asarray(can)
    for _ in range(int(sec * FPS)):
        out.write(a)


cp = json.loads((LOG / "chain_policy.json").read_text())
cc = json.loads((LOG / "chain_curobo.json").read_text())
ok = sum(r["success"] for r in cp)

card([("SO-101 로봇팔, NVIDIA cuRobo + 정책 학습", 46, (255, 255, 255)),
      ("GPU 모션 계획으로 시연을 만들고, 그 시연으로 집기 정책을 학습", 26, (205, 210, 222)),
      ("Isaac Sim 물리 시뮬레이션 · RTX 4070 Ti 한 대", 22, (150, 200, 90))], 2.8)

# ---- cuRobo
clip("isaac_harness_obstacles.mp4", 450, 817, 1.5, "cuRobo",
     "운반 중 나타난 상자를 감지하고 0.14초 만에 다시 계획 (Isaac Sim 물리)")
clip("demo_highlights.mp4", 700, 1000, 2.0, "cuRobo",
     "깊이 영상으로 장면을 복원하고 그 안에서 충돌 없이 계획 (약 0.1초)",
     badge="오른쪽 깊이·RGB: 외부 공개 데이터 CarolinePascal/so101_depth_raw")

# ---- learning (learning_story.mp4 frame ranges: sim demos, loss curves, scenario 14 rollouts, success bars)
clip("learning_story.mp4", 255, 435, 2.0, "학습 데이터",
     "cuRobo를 교사로: Isaac Sim에서 집기·옮기기 시연 120개 자동 생성")
clip("learning_story.mp4", 660, 855, 2.0, "학습",
     "그 시연으로 SmolVLA 정책 학습 (실제 시연 45개와 함께)")
clip("learning_story.mp4", 1720, 1893, 1.0, "학습 결과",
     "같은 장면: 학습 전·실제 데이터만은 실패, 시뮬 시연을 더한 체크포인트는 성공")
clip("learning_story.mp4", 1893, 1968, 1.0, "학습 결과",
     "성공률 0/20 → 7/20 (교사인 cuRobo는 20/20)", hold=2.0)

# ---- together
clip("full_chain.mp4", 0, 400, 1.5, "cuRobo + 학습",
     f"cuRobo 접근·운반 + 학습한 정책 집기: {ok}/{len(cp)} 성공, 장애물 재계획 포함")

card([("NVIDIA cuRobo · Isaac Sim · LeRobot SmolVLA", 34, (255, 255, 255)),
      (f"cuRobo만: {sum(r['success'] for r in cc)}/{len(cc)}  |  cuRobo + 학습한 정책: {ok}/{len(cp)}  |  정책 혼자: 7/20", 23, (205, 210, 222)),
      ("모두 시뮬레이션 결과이며, 실제 로봇에서는 아직 시험하지 않았습니다", 22, (205, 210, 222))], 4,
     foot="학습 데이터: 직접 만든 시뮬 시연 120개 + 공개 데이터 lerobot/svla_so101_pickplace 45개 (LeRobot 팀, Apache-2.0)")
out.close()
