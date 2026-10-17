import sys, time, numpy as np
sys.path.insert(0, "so101")
from isaac_client import IsaacSim
t0 = time.time()
sim = IsaacSim(frames_dir="isaac/frames_smoke")
print("server up %.0fs" % (time.time() - t0))
sc = dict(table_z=-0.0168, friction=1.0,
          bin=dict(center=[0.3004, -0.2027], yaw=np.deg2rad(-177.1), L=0.1221, W=0.1264, H=0.0457),
          cube=dict(center=[0.3227, -0.1826, 0.0035], yaw=np.deg2rad(19.9), size=0.0327, mass=0.01),
          obstacles=[dict(name="wall", center=[0.20, -0.17, -0.0168 + 0.04], dims=[0.02, 0.12, 0.08], yaw=np.deg2rad(73.6))],
          place=[0.1293, -0.1508], q0=[0, 0, 0, 0, 0, 1.2])
r = sim.call(cmd="setup", scene=sc)
print("setup:", r)
q = [[0, -0.3 * a, 0.3 * a, 0.5 * a, 0, 1.2 - a] for a in np.linspace(0, 1, 40)]
r = sim.call(cmd="exec", q=q, dt=0.025, settle=0.5)
print("exec:", {k: v for k, v in r.items()})
sim.close()
print("total %.0fs" % (time.time() - t0))
