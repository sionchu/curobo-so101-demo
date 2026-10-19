"""videos/learning_story.mp4: what the policies learned from, how training went, and what changed per checkpoint.

    .venv\\Scripts\\python.exe so101\\compute_ee.py            (gripper-tip paths, needs cuRobo)
    .venv-lerobot\\Scripts\\python.exe so101\\make_learning_story.py

Every curve and number is read from logs/: train_*.log, policy_eval_ckpts.json/_preds.npz, eval_closedloop_*.json,
traj_*.json, learning_ee.json, chain_summary.json. Clips are the recorded dataset videos and Isaac viewport frames.
"""

import json
import re
import sys
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from record_util import Mp4Writer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs"
W, H, FPS = 1280, 720, 30
BG = (15, 15, 20)
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
out = Mp4Writer(ROOT / "videos" / "learning_story.mp4", fps=FPS)


def F(size):
    for name in ("malgunbd.ttf" if size >= 30 else "malgun.ttf", "malgun.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


EXT, OURS = (200, 110, 10), (20, 130, 60)  # orange = someone else's public recording, green = made on this PC
SRC = []  # (x0, x1, colour, text) source badges drawn at the bottom of every frame until changed


def src(*tags):
    SRC[:] = tags


def emit(img, n=1):
    img = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img)
    if SRC:
        img = img.copy()
        d = ImageDraw.Draw(img, "RGBA")
        for x0, x1, col, text in SRC:
            w = min(x1 - x0 - 8, int(d.textlength(text, font=F(15))) + 16)
            d.rectangle([x0 + 4, H - 32, x0 + 4 + w, H - 5], fill=(*col, 240))
            d.text((x0 + 12, H - 30), text, fill=(255, 255, 255), font=F(15))
    a = np.asarray(img)
    for _ in range(n):
        out.write(a)


def fig_img(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def card(title, lines, sec=3.5):
    src()
    can = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(can)
    d.text((70, 210), title, fill=(255, 255, 255), font=F(42))
    for i, l in enumerate(lines):
        d.text((70, 290 + 42 * i), l, fill=(205, 205, 215), font=F(24))
    emit(can, int(sec * FPS))


def header(can, text, sub=None):
    d = ImageDraw.Draw(can, "RGBA")
    d.rectangle([0, 0, W, 58], fill=(0, 0, 0, 220))
    d.text((16, 6), text, fill=(255, 255, 255), font=F(24))
    if sub:
        d.text((16, 35), sub, fill=(180, 180, 190), font=F(15))


def fit(im, w, h):
    im = im.convert("RGB")
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    bgc = Image.new("RGB", (w, h), (0, 0, 0))
    bgc.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bgc


# ---------------------------------------------------------------- 1. training data
def grid_play(clips, labels, title, sub, stride=4, hold=2, max_n=90, rows=3):
    tw, th = W // 3, (H - 60) // rows
    n = min(max_n, max(len(c) for c in clips) // stride)
    for k in range(n):
        can = Image.new("RGB", (W, H), BG)
        for j, (c, lab) in enumerate(zip(clips, labels)):
            fr = c[min(k * stride, len(c) - 1)]
            x, y = (j % 3) * tw, 60 + (j // 3) * th
            can.paste(fit(fr if isinstance(fr, Image.Image) else Image.open(fr), tw - 4, th - 4), (x + 2, y + 2))
            dd = ImageDraw.Draw(can)
            dd.rectangle([x + 4, y + 4, x + 14 + dd.textlength(lab, font=F(16)), y + 28], fill=SRC[0][2] if SRC else (0, 0, 0))
            dd.text((x + 9, y + 5), lab, fill=(255, 255, 255), font=F(16))
        header(can, title, sub)
        emit(can, hold)


def real_clips(eps):
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset("lerobot/svla_so101_pickplace", root=ROOT / "data" / "svla_so101_pickplace",
                        episodes=eps, video_backend="pyav")
    epi = np.array([int(x) for x in ds.hf_dataset["episode_index"]])
    clips = []
    for e in eps:
        idx = np.flatnonzero(epi == e)[::4][:90]
        fr = []
        for i in idx:
            t = ds[int(i)]["observation.images.up"]  # (3,H,W) float 0..1
            fr.append(Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
        clips.append([f for f in fr for _ in range(4)])  # re-expand so grid_play's stride applies uniformly
    return clips


REAL_EPS = [0, 20, 40]
card("내가 한 것: 시뮬 데이터를 만들고, 학습시키고, 시뮬에서 평가", [
    "SO-101 집어서 옮기기. Isaac Sim + cuRobo로 시연을 직접 만들고 정책을 학습해 바뀐 점을 비교",
    "1) 직접 만든 시뮬 시연  2) 학습 곡선  3) 체크포인트별 예측",
    "4) 같은 시나리오에서 체크포인트별 실행 + 그리퍼 궤적  5) cuRobo와 연결",
    "외부 공개 데이터는 학습에 섞은 참고 자료로만 짧게 보여 줍니다"], 4.5)
leg = Image.new("RGB", (W, H), BG)
d_ = ImageDraw.Draw(leg)
d_.text((70, 150), "화면 아래 색 표시", fill=(255, 255, 255), font=F(42))
for i_, (col_, t1, t2) in enumerate([
        (OURS, "직접 만듦", "본편. 이 PC에서 한 것: Isaac Sim 시뮬 시연 120개, 학습, 평가 실행, 그래프"),
        (EXT, "외부 참고", "참고. 남이 실제 로봇으로 찍어 공개한 데이터 (HF lerobot/svla_so101_pickplace)"),
        (None, "", "학습에 함께 섞었고 예측 비교의 정답으로만 씀. 이 프로젝트에서 실제 로봇은 돌리지 않음")]):
    if col_:
        d_.rectangle([70, 250 + 50 * i_, 240, 284 + 50 * i_], fill=col_)
        d_.text((82, 253 + 50 * i_), t1, fill=(255, 255, 255), font=F(22))
    d_.text((260, 254 + 50 * i_), t2, fill=(210, 215, 228), font=F(21))
emit(leg, 4 * FPS)
sim_eps = sorted((ROOT / "data" / "sim_raw").glob("ep_*"))
pick = sim_eps[:: max(1, len(sim_eps) // 9)][:9]
src((0, W, OURS, f"직접 만듦: 이 PC의 Isaac Sim에서 cuRobo로 생성한 시뮬 시연 {len(sim_eps)}개 중 일부"))
grid_play([sorted((d / "up").glob("*.jpg")) for d in pick], [f"직접 시뮬 {d.name}" for d in pick],
          f"1. 직접 만든 학습 데이터: Isaac Sim + cuRobo 전문가 시연 {len(sim_eps)}개 중 9개", "이 PC에서 생성: Isaac Sim 물리, cuRobo가 계획한 집기/옮기기, 브릭·상자·카메라·조명 무작위, 2배속")
src((0, W, EXT, "외부 참고: 남이 실제 로봇으로 찍은 공개 영상 (HF lerobot/svla_so101_pickplace, LeRobot 팀). 내가 만든 것 아님"))
grid_play(real_clips(REAL_EPS), [f"외부 참고 ep {e}" for e in REAL_EPS],
          "참고: 함께 섞은 외부 실제 시연 45개 중 3개", "남이 공개한 데이터. 시뮬 장면 배치를 이 데이터 통계에 맞췄고, 학습에 섞음", max_n=45, rows=1)

EE = json.loads((LOG / "learning_ee.json").read_text())


def box_xy(c, yaw, L=0.10, Wd=0.075):
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (R @ np.array([[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]]).T / 2).T + np.array(c[:2])


n = 90
src((0, 640, OURS, "주황 선 = 직접 만든 시뮬 시연 (이 PC, cuRobo)"),
    (640, W, EXT, "흐린 파란 선 = 외부 참고 (남이 찍은 실제 시연)"))
for k in range(n + 45):
    f = min(1.0, (k + 1) / n)
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.6), dpi=100)
    for ax, (a, b, ttl) in zip(axs, [(0, 1, "위에서 본 그리퍼 끝 궤적 (x-y)"), (0, 2, "옆에서 본 궤적 (x-z)")]):
        for name, col, al in (("real", "tab:blue", 0.15), ("sim", "tab:orange", 0.45)):
            for p in EE[name]:
                p = np.array(p)
                m = max(2, int(len(p) * f))
                ax.plot(p[:m, a], p[:m, b], color=col, lw=0.6, alpha=al)
        ax.plot([], [], color="tab:orange", label=f"[직접] 시뮬 전문가 시연 {len(EE['sim'])}개")
        ax.plot([], [], color="tab:blue", alpha=0.4, label=f"[외부 참고] 실제 시연 {len(EE['real'])}개")
        ax.set_title(ttl)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)" if b == 1 else "z (m)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("학습 데이터의 궤적 분포: 정책은 이 궤적들을 따라 하도록 학습됨 (cuRobo FK, 로봇 base 기준)")
    fig.tight_layout()
    emit(fit(fig_img(fig), W, H))
    plt.close(fig)


# ---------------------------------------------------------------- 2. training curves
def parse(fn):
    b = (LOG / fn).read_bytes()  # PowerShell '*>' logs are UTF-16
    t = b.decode("utf-16", errors="ignore") if b"\x00" in b[:200] else b.decode("utf-8", errors="ignore")
    t = t.replace("\r", "").replace("\n", "")
    pts = []
    for s, l in re.findall(r"step:(\d+(?:\.\d+)?[KM]?) (?:smpl:\S+ )?ep:\S+ epch:\S+ loss:([\d.]+)", t):
        mul = 1000 if s.endswith("K") else 1_000_000 if s.endswith("M") else 1
        pts.append((float(s.rstrip("KM")) * mul, float(l)))
    return np.array(sorted(set(pts))).reshape(-1, 2)


act = parse("train_act.log")
act_r = parse("train_act_resume.log")
CURVES = [("ACT (외부 실제 데이터)", np.r_[act[act[:, 0] <= 10000], act_r], [25000]),
          ("SmolVLA (외부 실제 데이터)", parse("train_smolvla.log"), [20000]),
          ("SmolVLA 추가학습 (외부 실제+직접 시뮬)", parse("train_smolvla_simft.log"), [5000, 10000])]
n = 150
src((0, W, OURS, "직접 함: 학습은 모두 이 PC에서 돌렸고 곡선은 그 로그 (학습에 쓴 데이터는 괄호 안 표시)"))
for k in range(n + 45):
    f = min(1.0, (k + 1) / n)
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 6.6), dpi=100)
    for ax, (ttl, c, cks) in zip(axs, CURVES):
        m = max(2, int(len(c) * f))
        ax.plot(c[:m, 0], c[:m, 1], color="tab:gray", lw=0.8, alpha=0.5, label="학습 손실 (로그)")
        if m > 10:
            w = max(3, len(c) // 40)
            sm = np.convolve(c[:m, 1], np.ones(w) / w, mode="valid")
            ax.plot(c[w - 1:m, 0], sm, color="tab:red", lw=2, label="이동 평균")
        for s in cks:
            if c[m - 1, 0] >= s:
                ax.axvline(s, color="tab:green", ls="--")
                ax.text(s, np.percentile(c[:, 1], 99), f" 체크포인트 {s // 1000:.0f}k",
                        color="tab:green", fontsize=9, va="top", rotation=90)
        ax.set_xlim(0, c[:, 0].max() * 1.02)
        ax.set_ylim(0, np.percentile(c[:, 1], 99) * 1.1)
        ax.set_title(ttl)
        ax.set_xlabel("학습 step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("학습 곡선 (train 로그의 실제 손실값). 손실이 줄어드는 동안 저장한 체크포인트를 아래에서 비교")
    fig.tight_layout()
    emit(fit(fig_img(fig), W, H))
    plt.close(fig)

# ---------------------------------------------------------------- 3. offline predictions on held-out episodes
P = np.load(LOG / "policy_eval_ckpts_preds.npz")
PE = json.loads((LOG / "policy_eval_ckpts.json").read_text())
names = [k for k in P.files if k not in ("recorded", "state", "episode")]
JN = ["어깨 회전", "어깨 들기", "팔꿈치", "손목 굽힘", "손목 회전", "그리퍼"]
src((0, 720, OURS, "색 선 = 내가 학습시킨 정책이 같은 화면을 보고 낸 예측"),
    (720, W, EXT, "검정 선 = 외부 참고 정답 (남이 조종한 시연)"))
for e in (47,):
    idx = np.flatnonzero(P["episode"] == e)
    n = len(idx)
    step = max(1, n // 200)
    for m in list(range(2, n, step)) + [n] * 45:
        fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.6), dpi=100, sharex=True)
        for j, ax in enumerate(axs.ravel()):
            ax.plot(P["recorded"][idx[:m], j], "k-", lw=2.2, label="[외부] 사람 시연 (정답)")
            for nm in names:
                ax.plot(P[nm][idx[:m], j], lw=1.2, label=nm)
            ax.set_xlim(0, n)
            lo, hi = P["recorded"][idx, j].min(), P["recorded"][idx, j].max()
            pad = (hi - lo) * 0.4 + 2
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(JN[j])
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7, loc="best")
        mae = " | ".join(f"{nm}: {PE[nm]['mae_arm_mean']}" for nm in names)
        fig.suptitle(f"학습에 안 쓴 시연 ep {e}: 같은 화면을 보고 정책이 낸 관절 명령 vs 사람 시연\n팔 관절 평균 오차(작을수록 좋음, 홀드아웃 5개) {mae}"
                     f" | 가만히 있기 기준 {PE['hold_state']['mae_arm_mean']}", fontsize=11)
        fig.tight_layout()
        emit(fit(fig_img(fig), W, H))
        plt.close(fig)

# ---------------------------------------------------------------- 4. same scenario, every checkpoint + gripper paths
src((0, W, OURS, "직접 만듦: 전부 이 PC의 Isaac Sim 시뮬 실행 (실제 로봇 아님). 학습 전 모델만 HF lerobot/smolvla_base 공개 가중치"))
RUNS = [("trj_base", "학습 전 (SmolVLA base)", "tab:gray"), ("trj_real_only", "실제 데이터만 (20k)", "tab:blue"),
        ("sim_ft_5k", "실제+시뮬 5k", "tab:purple"), ("trj_sim_ft", "실제+시뮬 10k", "tab:red"),
        ("trj_expert", "cuRobo 전문가 (교사)", "tab:green")]
RES = {}
for tag, _, _ in RUNS:
    f = LOG / f"eval_closedloop_{tag}.json"
    if f.exists():
        RES[tag] = {r["scenario"]: r for r in json.loads(f.read_text())}
RUNS = [r for r in RUNS if r[0] in RES]


def seg(tag, i):
    r = RES[tag].get(i)
    if r is None:
        return [], None
    files = sorted((ROOT / "isaac" / r["frames_dir"]).glob("*.jpg"))
    return files[r["frame_start"]:r["frame_end"]], r


def verdict(r):
    if r is None:
        return "실행 안 함", (120, 120, 120)
    if r["success"]:
        return "성공: 상자 안에 놓음", (0, 140, 0)
    return ("들었지만 못 놓음", (190, 110, 0)) if r["lifted"] else ("못 집음", (190, 0, 0))


SPEED = 3
tw, th = W // 3, (H - 60) // 2
for i in sorted(set.intersection(*[set(RES[t]) for t, _, _ in RUNS])):
    segs = {t: seg(t, i) for t, _, _ in RUNS}
    n = max(len(s[0]) for s in segs.values())
    for k in list(range(0, n, SPEED)) + [n - 1] * 30:
        can = Image.new("RGB", (W, H), BG)
        for j, (t, ttl, _) in enumerate(RUNS):
            files, r = segs[t]
            im = Image.open(files[min(k, len(files) - 1)]) if files else Image.new("RGB", (16, 9))
            x, y = (j % 3) * tw, 60 + (j // 3) * th
            can.paste(fit(im, tw - 4, th - 34), (x + 2, y + 32))
            d = ImageDraw.Draw(can)
            d.rectangle([x + 2, y + 2, x + tw - 2, y + 32], fill=(245, 245, 245))
            v, col = verdict(r)
            d.text((x + 8, y + 6), ttl, fill=(10, 10, 10), font=F(16))
            v = "결과: " + v
            d.text((x + tw - 8 - d.textlength(v, font=F(14)), y + 8), v, fill=col, font=F(14))
        fig, ax = plt.subplots(figsize=(tw / 100, (th - 4) / 100), dpi=100)
        sc = None
        for t, ttl, col in RUNS:
            tr = EE["runs"].get(t, {}).get(str(i))
            if not tr or not tr["ee"]:
                continue
            sc = tr
            p = np.array(tr["ee"])
            mm = max(2, int(len(p) * min(1.0, k / max(1, len(segs[t][0]) - 1))))
            ax.plot(p[:mm, 0], p[:mm, 1], color=col, lw=1.6)
            ax.plot(p[mm - 1, 0], p[mm - 1, 1], "o", color=col, ms=4)
        if sc:
            ax.add_patch(Polygon(box_xy(sc["box"], sc["box_yaw"]), closed=True, fill=False, ec="k", lw=1.2))
            ax.plot(*sc["brick0"][:2], "s", color="hotpink", ms=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_title("그리퍼 끝 궤적 (위에서, ■ 브릭 시작, □ 상자)", fontsize=8)
        ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)
        j = len(RUNS)
        can.paste(fit(fig_img(fig), tw - 4, th - 4), ((j % 3) * tw + 2, 60 + (j // 3) * th + 2))
        plt.close(fig)
        header(can, f"같은 시나리오 {i}번에서 체크포인트별 실행 (Isaac 물리, {SPEED}배속)",
               "모두 같은 브릭/상자/카메라 배치, 휴지 자세에서 시작. 궤적 선 색 = 타일 순서(회색, 파랑, 보라, 빨강, 초록)")
        emit(can)

# success rate per checkpoint (full 20-scenario runs; the base model only ran the 4 trajectory scenarios)
BARS = []
for tag, ttl in [("trj_base", "학습 전"), ("real_only", "실제만 20k"), ("sim_ft_5k", "실제+시뮬 5k"),
                 ("sim_ft", "실제+시뮬 10k"), ("curobo_expert", "cuRobo 전문가")]:
    f = LOG / f"eval_closedloop_{tag}.json"
    if f.exists():
        rr = json.loads(f.read_text())
        BARS.append((ttl, sum(r["success"] for r in rr), sum(r["lifted"] for r in rr), len(rr)))
for k in range(75):
    f = min(1.0, (k + 1) / 45)
    fig, ax = plt.subplots(figsize=(12.8, 6.6), dpi=100)
    xs = np.arange(len(BARS))
    ax.bar(xs - 0.2, [b[2] / b[3] * 100 * f for b in BARS], 0.4, color="tab:orange", label="브릭 들어 올림 (30mm 이상)")
    ax.bar(xs + 0.2, [b[1] / b[3] * 100 * f for b in BARS], 0.4, color="tab:green", label="상자 안에 놓음 (성공)")
    for x, b in zip(xs, BARS):
        ax.text(x - 0.2, b[2] / b[3] * 100 * f + 1, f"{b[2]}/{b[3]}", ha="center", fontsize=10)
        ax.text(x + 0.2, b[1] / b[3] * 100 * f + 1, f"{b[1]}/{b[3]}", ha="center", fontsize=10)
    ax.set_xticks(xs, [b[0] for b in BARS])
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    ax.set_title("체크포인트별 Isaac 폐루프 결과 (정책 혼자 처음부터 끝까지, 장애물 없음). 학습 전은 시나리오 4개만 실행")
    fig.tight_layout()
    emit(fit(fig_img(fig), W, H))
    plt.close(fig)

# ---------------------------------------------------------------- 5. connected to cuRobo (stage 9 clip)
cp = json.loads((LOG / "chain_policy.json").read_text())
cc = json.loads((LOG / "chain_curobo.json").read_text())
card("학습한 정책을 cuRobo와 연결", [f"cuRobo 접근 + SmolVLA 집기 + 검증/복구 + cuRobo 운반(장애물 회피): {sum(r['success'] for r in cp)}/{len(cp)} 성공",
                                   f"전부 cuRobo로 했을 때: {sum(r['success'] for r in cc)}/{len(cc)}  |  정책 혼자: 위 막대 그래프",
                                   "아래는 그 실행 영상 (videos/full_chain.mp4)"], 4)
src((0, W, OURS, "직접 만듦: 이 PC의 Isaac Sim 시뮬 실행 (cuRobo + 이 PC에서 학습한 SmolVLA)"))
with av.open(str(ROOT / "videos" / "full_chain.mp4")) as c:
    for fr in c.decode(video=0):
        emit(fit(fr.to_image(), W, H))
out.close()
