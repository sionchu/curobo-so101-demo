"""Pure-NumPy port of FoundationPose's mycpp.cluster_poses (mycpp/src/app/pybind_api.cpp).

The C++ extension needs Boost + Eigen + OpenMP, which are not installed on this Windows machine.
Same greedy algorithm: keep a pose if no kept pose is within dist_diff (m) and, under any symmetry
transform, within angle_diff (deg) geodesic rotation distance.
"""

import numpy as np


def _geodesic(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs):
    poses_in = [np.asarray(p, np.float32) for p in poses_in]
    sym = np.asarray(symmetry_tfs, np.float32).reshape(-1, 4, 4)
    print(f"num original candidates = {len(poses_in)}")
    thr = angle_diff / 180.0 * np.pi
    out = [poses_in[0]]
    for cur in poses_in[1:]:
        isnew = True
        for cl in out:
            if np.linalg.norm(cl[:3, 3] - cur[:3, 3]) >= dist_diff:
                continue
            cur_sym = cur[None] @ sym  # (S, 4, 4)
            if np.any(_geodesic(cur_sym[:, :3, :3], np.broadcast_to(cl[:3, :3], (len(sym), 3, 3))) < thr):
                isnew = False
                break
        if isnew:
            out.append(cur)
    print(f"num of pose after clustering: {len(out)}")
    return out
