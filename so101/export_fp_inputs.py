import sys, numpy as np
sys.path.insert(0, "so101")
import perception as P
from depth_scene import FX, FY, CX, CY
p = P.perceive()
T_wc, tz = P._poses()
c = p["cube"]
np.savez("logs/fp_inputs.npz", T_w_depth=T_wc, T_w_rgb=p["rgb_cam"]["T"], K_rgb=P.RGB_K,
         K_depth=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]), table_z=tz,
         cube_center=c["center"], cube_yaw=c["yaw"], cube_size=c["size"],
         bin_T=p["bin"]["T"], bin_size=np.array(p["bin"]["size"]))
print("saved; rgb cam offset from depth cam %.4f m" % p["rgb_cam"]["offset_to_depth_cam_m"])
