"""Contact sheet of chosen frames from an MP4, for checking a rendered video by eye.

    python so101\\contact_sheet.py videos\\x.mp4 out.jpg 60,200,380,500     (frame indices)
"""

import sys

import av
from PIL import Image

src, dst, want = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
got = {}
with av.open(src) as c:
    for i, fr in enumerate(c.decode(video=0)):
        if i in want:
            got[i] = fr.to_image().resize((640, 360))
        if i > max(want):
            break
cols = 2
rows = (len(want) + 1) // cols
sheet = Image.new("RGB", (640 * cols, 360 * rows))
for k, i in enumerate(want):
    if i in got:
        sheet.paste(got[i], ((k % cols) * 640, (k // cols) * 360))
sheet.save(dst, quality=85)
print("frames", sorted(got))
