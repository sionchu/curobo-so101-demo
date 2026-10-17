"""Isaac Sim physics backend for the SO-101 harness (run with C:\\isaacsim\\python.bat, read-only use).

A JSON-lines TCP server on localhost. The cuRobo harness (separate venv) plans; this process executes
joint trajectories through the articulation's position drives with PhysX and reports physics state.

Commands (one JSON object per line, one JSON reply per line):
  {"cmd": "setup", "scene": {...}}            build table / bin / cube / obstacles, reset robot
  {"cmd": "exec", "q": [[6 joints]...], "dt": 0.025, "settle": 0.3}   follow a trajectory (arm + gripper)
  {"cmd": "state"}                            joints, cube pose, contacts since the last call
  {"cmd": "spawn", "name": .., "center": [..], "dims": [..], "yaw": ..}   add a static obstacle
  {"cmd": "wait", "seconds": s}               step physics without new targets
  {"cmd": "quit"}
Frames are written to --frames as JPEGs (one per rendered step) for the video.
"""

import argparse
import json
import math
import os
import socket

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5599)
ap.add_argument("--usd", required=True)
ap.add_argument("--frames", default=None)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--frame-stride", type=int, default=2, help="save every Nth app update (60 Hz / 2 = 30 fps)")
args, _ = ap.parse_known_args()

simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PHYS_HZ = 120  # PhysX steps per second
APP_HZ = 60  # one simulation_app.update() advances the timeline by 1/60 s (2 physics substeps), measured

stage_utils.create_new_stage(template="sunlight")
simulation_app.update()
stage = stage_utils.get_current_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateTimeStepsPerSecondAttr(PHYS_HZ)
px.CreateEnableCCDAttr(True)
stage_utils.add_reference_to_stage(usd_path=args.usd, path="/World/so101")
simulation_app.update()

_contacts = []  # (a, b, impulse) since last read


def _on_contacts(headers, data):
    for h in headers:
        a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        imp = 0.0
        for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            imp += float(np.linalg.norm(np.array(data[i].impulse)))
        _contacts.append((a, b, imp))


_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(_on_contacts)


def _material(path, mu):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    return m


def _box(path, center, dims, yaw=0.0, color=(0.5, 0.5, 0.5), dynamic=False, mass=None, material=None, report=False):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    cube = UsdGeom.Cube.Define(stage, path + "/geom")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, dims)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    if dynamic or report:
        rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        if not dynamic:
            rb.CreateKinematicEnabledAttr(True)
        if mass is not None:
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(mass))
        cr = PhysxSchema.PhysxContactReportAPI.Apply(xf.GetPrim())
        cr.CreateThresholdAttr(0.0)
    return xf


robot = None
cube_rb = None
_cams = {}
rp = annot = None
frame_i = 0


_step_i = 0


def _render_frame():
    global frame_i, _step_i
    _step_i += 1
    if annot is None or _step_i % args.frame_stride:
        return
    data = annot.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return
    Image.fromarray(np.asarray(data)[..., :3]).save(os.path.join(args.frames, f"{frame_i:05d}.jpg"), quality=90)
    frame_i += 1


def _step(n=1):
    for _ in range(n):
        simulation_app.update()
        _render_frame()


def _dof_order():
    names = list(robot.dof_names)
    return [names.index(j) for j in JOINTS]


def _set_targets(q6):
    idx = _dof_order()
    t = np.zeros((1, len(idx)), np.float32)
    for k, i in enumerate(idx):
        t[0, i] = q6[k]
    robot.set_dof_position_targets(t)


def _joints():
    q = robot.get_dof_positions().numpy()[0]
    return [float(q[i]) for i in _dof_order()]


def _cube_pose():
    p, o = cube_rb.get_world_poses()
    return p.numpy()[0].tolist(), o.numpy()[0].tolist()  # wxyz


def setup(sc):
    global robot, cube_rb, rp, annot
    mu = _material("/World/Looks/rubber", sc.get("friction", 1.0))
    tz = sc["table_z"]
    _box("/World/table", (0.15, -0.1, tz - 0.02), (1.0, 1.0, 0.04), color=tuple(sc.get("table_color", (0.15, 0.17, 0.2))), material=mu)
    if sc.get("robot_color"):  # override the robot's visual material (e.g. black like the real SO-101)
        mat = UsdShade.Material.Define(stage, "/World/Looks/robot_paint")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/robot_paint/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*sc["robot_color"]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/so101")).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    b = sc.get("bin")  # open box: floor + 4 walls (measured size and pose)
    if b is None:
        b = dict(L=0.02, W=0.02, H=0.02, center=[50.0, 50.0], yaw=0.0)  # no bin in this scene (placed far away)
    L, W, H, th = b["L"], b["W"], b["H"], 0.003
    c, yaw = np.array(b["center"]), b["yaw"]
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    green = tuple(b.get("color", (0.05, 0.55, 0.25)))
    _box("/World/bin/floor", (c[0], c[1], tz + 0.002), (L, W, 0.004), yaw, green, material=mu)
    for name, off, dims in [("wxp", (L / 2 - th / 2, 0), (th, W, H)), ("wxm", (-L / 2 + th / 2, 0), (th, W, H)),
                            ("wyp", (0, W / 2 - th / 2), (L, th, H)), ("wym", (0, -W / 2 + th / 2), (L, th, H))]:
        o = R @ np.array(off)
        _box(f"/World/bin/{name}", (c[0] + o[0], c[1] + o[1], tz + H / 2), dims, yaw, green, material=mu, report=True)
    cb = sc["cube"]
    _box("/World/cube", cb["center"], cb.get("dims", (cb["size"],) * 3), cb["yaw"], tuple(cb.get("color", (0.85, 0.1, 0.1))),
         dynamic=True, mass=cb.get("mass", 0.01), material=mu)
    for name, cam in sc.get("cameras", {}).items():  # extra RGB cameras for policies (rendered on demand)
        path = f"/World/cams/{name}"
        c_ = UsdGeom.Camera.Define(stage, path)
        c_.CreateFocalLengthAttr(cam.get("focal_mm", 18.0))
        c_.CreateHorizontalApertureAttr(cam.get("aperture_mm", 20.955))
        c_.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        set_camera_view(eye=cam["eye"], target=cam["target"], camera_prim_path=path)
        rp_ = rep.create.render_product(path, (cam.get("w", 640), cam.get("h", 480)))
        an_ = rep.AnnotatorRegistry.get_annotator("rgb")
        an_.attach([rp_])
        _cams[name] = an_
    for ob in sc.get("obstacles", []):
        _box(f"/World/obstacles/{ob['name']}", ob["center"], ob["dims"], ob.get("yaw", 0.0), tuple(ob.get("color", (0.25, 0.35, 0.85))),
             report=True)
    tp = sc["place"]
    pad = UsdGeom.Cylinder.Define(stage, "/World/place_pad")
    pad.CreateRadiusAttr(0.025)
    pad.CreateHeightAttr(0.001)
    pad.AddTranslateOp().Set(Gf.Vec3d(tp[0], tp[1], tz + 0.0006))
    pad.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.3)])
    # robot links report contacts
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/so101") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    simulation_app.update()

    robot = Articulation("/World/so101")
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    cube_rb = RigidPrim("/World/cube")
    n = len(robot.dof_names)
    g = sc.get("gains", {})
    stiff = np.full((1, n), g.get("arm_kp", 300.0), np.float32)
    damp = np.full((1, n), g.get("arm_kd", 15.0), np.float32)
    gi = list(robot.dof_names).index("gripper")
    stiff[0, gi], damp[0, gi] = g.get("grip_kp", 40.0), g.get("grip_kd", 2.0)
    robot.set_dof_gains(stiffnesses=stiff, dampings=damp)
    q0 = sc["q0"]
    idx = _dof_order()
    t = np.zeros((1, n), np.float32)
    for k, i in enumerate(idx):
        t[0, i] = q0[k]
    robot.set_dof_positions(t)
    robot.set_dof_position_targets(t)
    if args.frames:
        os.makedirs(args.frames, exist_ok=True)
        set_camera_view(eye=sc.get("cam_eye", [-0.05, -0.80, 0.55]), target=sc.get("cam_target", [0.17, -0.12, 0.05]),
                        camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (args.width, args.height))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
    _contacts.clear()
    _step(int(APP_HZ * 0.5))  # let the cube settle
    return dict(dof_names=list(robot.dof_names), cube=_cube_pose(), q=_joints())


def _summarize_contacts():
    out = {}
    for a, b, imp in _contacts:
        ra = a.startswith("/World/so101")
        rb_ = b.startswith("/World/so101")
        if ra == rb_:
            if not (a.startswith("/World/cube") or b.startswith("/World/cube")):
                continue
        other = b if ra else a
        link = (a if ra else b).split("/")[-1]
        if not (ra or rb_):
            link, other = "cube", (b if a.startswith("/World/cube") else a)
        key = f"{link}|{'/'.join(other.split('/')[:4])}"
        out[key] = round(out.get(key, 0.0) + imp, 5)
    _contacts.clear()
    return out


def handle(m):
    c = m["cmd"]
    if c == "setup":
        return setup(m["scene"])
    if c == "exec":
        sub = max(1, int(round(m.get("dt", 0.025) * APP_HZ)))
        for q6 in m["q"]:
            _set_targets(q6)
            _step(sub)
        _step(int(APP_HZ * m.get("settle", 0.2)))
        q = _joints()
        return dict(q=q, cube=_cube_pose(), track_err=float(np.max(np.abs(np.array(q) - np.array(m["q"][-1])))),
                    contacts=_summarize_contacts(), frames=frame_i)
    if c == "wait":
        _step(int(APP_HZ * m["seconds"]))
        return dict(q=_joints(), cube=_cube_pose(), contacts=_summarize_contacts(), frames=frame_i)
    if c == "state":  # includes a 'virtual sensor' listing of obstacle prims present in the sim
        obs = {}
        root = stage.GetPrimAtPath("/World/obstacles")
        if root:
            for ch in root.GetChildren():
                m = UsdGeom.Xformable(ch).ComputeLocalToWorldTransform(0)
                g = UsdGeom.Cube(stage.GetPrimAtPath(ch.GetPath().AppendChild("geom")))
                sc_ = [float(v) for v in g.GetPrim().GetAttribute("xformOp:scale").Get()]
                obs[ch.GetName()] = dict(center=list(m.ExtractTranslation()), dims=sc_,
                                         yaw=math.atan2(m[0][1], m[0][0]))
        return dict(q=_joints(), cube=_cube_pose(), contacts=_summarize_contacts(), frames=frame_i,
                    sim_time=float(omni.timeline.get_timeline_interface().get_current_time()), obstacles=obs)
    if c == "spawn":
        _box(f"/World/obstacles/{m['name']}", m["center"], m["dims"], m.get("yaw", 0.0), (0.95, 0.3, 0.1), report=True)
        _step(1)
        return dict(ok=True, frames=frame_i)
    if c == "reset":  # new episode without restarting: brick pose, robot joints, camera poses, table shade
        idx = _dof_order()
        n = len(robot.dof_names)
        t = np.zeros((1, n), np.float32)
        for k, i in enumerate(idx):
            t[0, i] = m["q0"][k]
        robot.set_dof_positions(t)
        robot.set_dof_position_targets(t)
        robot.set_dof_velocities(np.zeros((1, n), np.float32))
        cb = m["cube"]
        h = cb["yaw"] / 2
        cube_rb.set_world_poses(positions=np.array([cb["center"]], np.float32),
                                orientations=np.array([[math.cos(h), 0, 0, math.sin(h)]], np.float32))
        cube_rb.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
        for name, cam in m.get("cameras", {}).items():
            set_camera_view(eye=cam["eye"], target=cam["target"], camera_prim_path=f"/World/cams/{name}")
        if "table_color" in m:
            UsdGeom.Cube(stage.GetPrimAtPath("/World/table/geom")).GetDisplayColorAttr().Set([Gf.Vec3f(*m["table_color"])])
        _step(int(APP_HZ * 0.3))
        _contacts.clear()
        return dict(q=_joints(), cube=_cube_pose(), frames=frame_i)
    if c == "render":  # images from the policy cameras (raw RGB, base64)
        import base64

        _step(1)
        out = {}
        for name, an_ in _cams.items():
            img = np.ascontiguousarray(np.asarray(an_.get_data())[..., :3])
            out[name] = dict(shape=list(img.shape), b64=base64.b64encode(img.tobytes()).decode())
        return dict(images=out, q=_joints(), cube=_cube_pose(), frames=frame_i)
    if c == "quit":
        return dict(ok=True, frames=frame_i)
    return dict(error=f"unknown cmd {c}")


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
print(f"ISAAC_SERVER_READY port={args.port}", flush=True)
conn, _ = srv.accept()
f = conn.makefile("rwb")
running = True
while running:
    line = f.readline()
    if not line:
        break
    msg = json.loads(line)
    try:
        rep_ = handle(msg)
    except Exception as e:  # report errors to the client instead of dying silently
        import traceback

        rep_ = dict(error=repr(e), tb=traceback.format_exc()[-1500:])
    f.write((json.dumps(rep_) + "\n").encode())
    f.flush()
    running = msg.get("cmd") != "quit"
conn.close()
simulation_app.close()
