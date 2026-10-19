"""videos/linkedin_cut_{en,ko}.mp4 (arg: en | ko): ~70 s social cut focused on cuRobo planning and policy training (all footage made here)."""

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer, _font as _font_en  # noqa: E402
from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "videos"
W, H, FPS = 1280, 720, 30
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"  # en | ko
KO = {'Inspired by what I saw at ROSCon China': 'ROSCon China에서 본 것에서 영감을 받아 시작', 'Code: github.com/sionchu/curobo-so101-demo': '코드: github.com/sionchu/curobo-so101-demo', 'NVIDIA cuRobo + a learned VLA policy': 'NVIDIA cuRobo + 학습한 VLA 정책', 'SO-101 arm pick & place, simulated in Isaac Sim': 'SO-101 로봇팔 집어서 옮기기, Isaac Sim 시뮬레이션', 'Everything built and run on one RTX 4070 Ti': 'RTX 4070 Ti 한 대로 직접 구축하고 실행', 'cuRobo GPU motion planning  ·  sim data generation  ·  SmolVLA fine-tuning': 'cuRobo GPU 모션 플래닝  ·  시뮬레이션 데이터 생성  ·  SmolVLA 파인튜닝', '1  cuRobo motion planning': '1  cuRobo 모션 플래닝', '2  Simulation data': '2  시뮬레이션 데이터', '3  Policy training': '3  정책 학습', '4  Planner + learned policy': '4  플래너 + 학습한 정책', 'GPU collision-free motion planning: ~0.1 s per plan': 'GPU 충돌 회피 경로 계획: 한 번에 약 0.1초', 'Grasp checked from physics: missed by 16 mm -> back off, re-plan, retry': '물리로 잡기 확인: 16mm 빗나감 → 물러나서 재계획 후 재시도', 'Lifted -> an obstacle drops into the path': '들어 올림 → 경로 위에 장애물이 떨어짐', 'Detected -> re-planned from the current state in 0.14 s, 0 contacts': '감지 → 현재 자세에서 0.14초 만에 재계획, 충돌 0회', 'cuRobo as the teacher: 120 physics-verified pick & place demos': 'cuRobo가 교사: 물리로 검증된 집어 옮기기 시연 120개 생성', 'Fine-tuned SmolVLA (vision-language-action) on real + simulated demos': '실제 + 시뮬레이션 시연으로 SmolVLA(비전-언어-행동 모델) 파인튜닝', 'Same 20 unseen test scenes, camera images in -> joint commands out': '처음 보는 테스트 장면 20개: 카메라 영상 입력 → 관절 명령 출력', 'Real data only: 0/20   ->   + simulated cuRobo demos: 7/20   (cuRobo: 20/20)': '실제 데이터만: 0/20   →   cuRobo 시뮬 시연 추가: 7/20   (cuRobo: 20/20)', 'cuRobo brings the arm in -> the learned policy grasps': 'cuRobo가 팔을 가져가고 → 학습한 정책이 잡기', 'A box appears mid-transport -> cuRobo re-plans around it': '운반 중 상자 등장 → cuRobo가 돌아가는 경로를 재계획', 'Grasp failed -> detected from physics -> recovered': '잡기 실패 → 물리로 감지 → 복구', 'Policy retries and succeeds -> cuRobo delivers': '정책이 다시 잡아 성공 → cuRobo가 상자까지 운반', 'Results (simulation, 20 test scenes)': '결과 (시뮬레이션, 테스트 장면 20개)', 'cuRobo + learned grasp: 16/20 delivered   ·   cuRobo alone: 17/20': 'cuRobo + 학습한 잡기: 16/20 성공   ·   cuRobo 단독: 17/20', 'Policy alone, end to end: 7/20   ·   re-planning: ~0.1 s': '정책 단독(처음부터 끝까지): 7/20   ·   재계획: 약 0.1초', 'Next: real SO-101 hardware (not tested yet)': '다음 단계: 실제 SO-101 하드웨어 (아직 테스트 안 함)'}


def T(text):
    return KO.get(text, text) if LANG == "ko" else text


def _font(size):
    return ImageFont.truetype("malgun.ttf", size) if LANG == "ko" else _font_en(size)


out = Mp4Writer(V / f"linkedin_cut_{LANG}.mp4", fps=FPS)
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
        d.text((28, H - 70), T(step), fill=(118, 185, 0), font=_font(20))
    d.text((28, H - 44 if step else H - 56), T(text), fill=(255, 255, 255), font=_font(28))
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
        d.text((80, y), T(l), fill=(255, 255, 255) if k == 0 else (200, 210, 225), font=f)
        y += 70 if k == 0 else 48
    if sub:
        d.text((80, 600), T(sub), fill=(118, 185, 0), font=_font(22))
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
      "Everything built and run on one RTX 4070 Ti",
      "Inspired by what I saw at ROSCon China"], 4,
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
     "Code: github.com/sionchu/curobo-so101-demo")
out.close()
