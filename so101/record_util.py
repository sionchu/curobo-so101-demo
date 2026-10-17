"""Record Viser scenes to MP4: grab browser renders via ClientHandle.get_render and composite panels."""

import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def look_at_wxyz(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Viser/OpenCV camera (+z forward, +y down) quaternion looking from position at target."""
    p, t, u = (np.asarray(x, np.float64) for x in (position, target, up))
    z = t - p
    z /= np.linalg.norm(z)
    x = np.cross(z, u)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def wait_client(server, timeout=600.0):
    print("waiting for a browser client to connect (open the viewer URL)...", flush=True)
    t0 = time.time()
    while not server.get_clients():
        if time.time() - t0 > timeout:
            raise TimeoutError("no browser client connected")
        time.sleep(0.5)
    time.sleep(3.0)  # let the scene finish loading in the browser
    return list(server.get_clients().values())[0]


def _font(size):
    for name in ("arial.ttf", "malgun.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(main: np.ndarray, panels: list, lines: list, panel_w=480) -> np.ndarray:
    """main render (H,W,3) on the left, panels stacked on the right, caption lines over the main view."""
    H = main.shape[0]
    canvas = Image.new("RGB", (main.shape[1] + (panel_w if panels else 0), H), (255, 255, 255))
    canvas.paste(Image.fromarray(main[..., :3]), (0, 0))
    if panels:
        ph = H // len(panels)
        for k, (img, label) in enumerate(panels):
            im = Image.fromarray(img[..., :3]).resize((panel_w, ph))
            canvas.paste(im, (main.shape[1], k * ph))
            d = ImageDraw.Draw(canvas)
            d.rectangle([main.shape[1], k * ph, main.shape[1] + 8 + 9 * len(label), k * ph + 24], fill=(0, 0, 0))
            d.text((main.shape[1] + 4, k * ph + 3), label, fill=(255, 255, 255), font=_font(16))
    d = ImageDraw.Draw(canvas)
    f = _font(20)
    for i, line in enumerate(lines):
        d.text((12, 10 + 26 * i), line, fill=(20, 20, 20), font=f)
    return np.asarray(canvas)


class Mp4Writer:
    def __init__(self, path, fps=30):
        self.path, self.fps, self.c, self.s, self.n = str(path), fps, None, None, 0

    def write(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        h, w = h - h % 2, w - w % 2
        if self.c is None:
            self.c = av.open(self.path, "w")
            self.s = self.c.add_stream("libx264", rate=self.fps)
            self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
            self.s.options = {"crf": "20", "preset": "medium"}
        for p in self.s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:h, :w, :3]), format="rgb24")):
            self.c.mux(p)
        self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        print(f"wrote {self.path}: {self.n} frames, {self.n / self.fps:.1f}s", flush=True)
