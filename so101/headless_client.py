"""Keep a headless Edge tab open on a Viser URL so the server can grab renders (get_render)."""
import sys, time
from playwright.sync_api import sync_playwright
url, secs = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 900
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True,
                          args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    pg.goto(url)
    print("connected", url, flush=True)
    t0 = time.time()
    while time.time() - t0 < secs:
        time.sleep(1)
        if pg.is_closed():
            break
    b.close()
