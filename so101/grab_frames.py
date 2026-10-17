import sys, av
from PIL import Image
path, outp, idxs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
with av.open(path) as c:
    s = c.streams.video[0]
    print(path, f"{s.width}x{s.height}", "frames", s.frames, "dur %.1fs" % float(s.duration * s.time_base) if s.duration else "")
    for i, f in enumerate(c.decode(s)):
        if i in idxs:
            f.to_image().save(f"{outp}_{i:04d}.png")
