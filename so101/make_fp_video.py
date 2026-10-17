import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, "so101")
from record_util import Mp4Writer
fr = sorted(Path("logs/fp_overlay").glob("*.png"))
w = Mp4Writer("videos/foundationpose_overlay.mp4", fps=15)
for f in fr:
    im = Image.open(f).convert("RGB").resize((1280, 960))
    w.write(np.asarray(im))
w.close()
