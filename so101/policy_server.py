"""Serve a LeRobot policy over localhost TCP (JSON lines) so a cuRobo process (other venv) can query it.

    .venv-lerobot\\Scripts\\python.exe so101\\policy_server.py --policy train\\smolvla_simft\\checkpoints\\010000\\pretrained_model --port 5610

  {"cmd": "reset"}                                             clear the policy's action queue
  {"cmd": "act", "images": {name: {shape, b64}}, "state": [6]}  -> {"action": [6]} (dataset units)
  {"cmd": "quit"}
"""

import argparse
import base64
import json
import socket

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

TASK = "pink lego brick into the transparent box"

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--port", type=int, default=5610)
a = ap.parse_args()
cfg = PreTrainedConfig.from_pretrained(a.policy)
cfg.device = "cuda"
pol = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", a.port))
srv.listen(1)
print(f"POLICY_SERVER_READY port={a.port}", flush=True)
conn, _ = srv.accept()
f = conn.makefile("rwb")
while True:
    line = f.readline()
    if not line:
        break
    m = json.loads(line)
    if m["cmd"] == "reset":
        pol.reset()
        rep = dict(ok=True)
    elif m["cmd"] == "act":
        im = {n: np.frombuffer(base64.b64decode(v["b64"]), np.uint8).reshape(v["shape"]) for n, v in m["images"].items()}
        obs = {"observation.images.up": im["up"], "observation.images.side": im["side"],
               "observation.state": np.array(m["state"], np.float32)}
        act = predict_action(obs, pol, torch.device("cuda"), pre, post, use_amp=False, task=TASK,
                             robot_type="so100_follower").cpu().numpy().reshape(-1)
        rep = dict(action=act.tolist())
    else:
        rep = dict(ok=True)
    f.write((json.dumps(rep) + "\n").encode())
    f.flush()
    if m["cmd"] == "quit":
        break
conn.close()
