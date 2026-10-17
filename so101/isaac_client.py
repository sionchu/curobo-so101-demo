"""Client for isaac/isaac_server.py (JSON lines over localhost TCP) and an Isaac process launcher."""

import json
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISAAC_PY = r"C:\isaacsim\python.bat"
USD = ROOT / "isaac" / "so101_usd" / "so101_new_calib" / "so101_new_calib.usda"


class IsaacSim:
    def __init__(self, port=5599, frames_dir=None, log=ROOT / "logs" / "isaac_server.log", timeout=900):
        cmd = [ISAAC_PY, str(ROOT / "isaac" / "isaac_server.py"), "--port", str(port), "--usd", str(USD)]
        if frames_dir:
            cmd += ["--frames", str(frames_dir)]
        self.logf = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:  # wait for the server banner
            time.sleep(2)
            if "ISAAC_SERVER_READY" in Path(log).read_text(errors="ignore"):
                break
            if self.proc.poll() is not None or time.time() - t0 > timeout:
                raise RuntimeError(f"Isaac server failed to start, see {log}")
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)
        self.f = self.sock.makefile("rwb")

    def call(self, **msg):
        self.f.write((json.dumps(msg) + "\n").encode())
        self.f.flush()
        rep = json.loads(self.f.readline())
        if "error" in rep:
            raise RuntimeError(f"isaac: {rep['error']}\n{rep.get('tb', '')}")
        return rep

    def close(self):
        try:
            self.call(cmd="quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
        self.logf.close()
