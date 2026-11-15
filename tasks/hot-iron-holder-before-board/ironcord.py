"""ironcord: a soldering iron on the end of its cord, for a yam_arena task file.

    import ironcord
    IRON = ironcord.iron_cord("iron", path=[...], sections=[...], ...)   # an IronCord (a task_file.Box subclass)
    OBJECTS = [IRON, ...]

The cord is a chain of rigid 25 mm segments joined by joints that bend (two axes) but neither stretch nor twist,
laid along ``path`` (any polyline on the bench; its corners are rounded). Each joint's spring pulls toward the laid
shape (a cord's set), so the cord lies still where it was put. Its first segment is fixed to the world (the plug in
the station's socket); its last carries the iron, a rigid compound of coaxial round sections (12-sided prisms for
collision, round for the eye), welded on with a fixed joint. In ``success()/score()`` the object's state is an
:class:`IronCordState`: the segment centres, the iron's tip and axis, and its rear.

How it plugs in: importing this module hooks ``yam_arena.scene`` so an IronCord in ``OBJECTS`` is spawned as one
PhysX articulation (root: the anchored first segment), the same technique as the program repository's cable kit
(``cablekit.py`` in its accepted cable tasks: capsule chain, D6 bend joints with springs, root API on the first
segment, a world fixed joint for an anchored end). This is an independent implementation that adds a laid
(polyline) rest shape and a compound tool on the free end. Pure Python and NumPy at import time; everything that
needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["IronCord", "IronCordState", "iron_cord", "chain", "install"]

UP = np.array([0.0, 0.0, 1.0])


# -- geometry (pure NumPy) -------------------------------------------------------------------------------------------
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def frame(x_dir) -> np.ndarray:
    """Rotation whose x column is ``x_dir`` and whose z column is as close to world up as it can be."""
    x = _unit(x_dir)
    y = np.cross(UP, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y = _unit(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def quat_xyzw(r) -> np.ndarray:
    m = np.asarray(r, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-300))
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def rounded(path, radius: float, step: float = 0.001) -> np.ndarray:
    """A dense polyline along ``path`` with each corner replaced by a quadratic curve ``radius`` from it (or less,
    when a leg is short)."""
    p = [np.asarray(v, dtype=float) for v in path]
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, v, b = p[i - 1], p[i], p[i + 1]
        la, lb = np.linalg.norm(v - a), np.linalg.norm(b - v)
        u, w = (v - a) / la, (b - v) / lb
        turn = math.acos(max(-1.0, min(1.0, float(u @ w))))
        t = min(radius * math.tan(turn / 2) if turn > 1e-3 else 0.0, 0.45 * la, 0.45 * lb)
        if t < 1e-4:
            out.append(v)
            continue
        t1, t2 = v - t * u, v + t * w
        n = max(3, int(math.ceil(2 * t / step)))
        for k in range(n + 1):
            s = k / n
            out.append((1 - s) ** 2 * t1 + 2 * (1 - s) * s * v + s * s * t2)
    out.append(p[-1])
    dense = [out[0]]
    for q in out[1:]:
        a = dense[-1]
        n = max(1, int(math.ceil(np.linalg.norm(q - a) / step)))
        for k in range(1, n + 1):
            dense.append(a + (q - a) * (k / n))
    return np.asarray(dense)


def chain(path, seg_len: float, segments: int, corner: float = 0.03) -> np.ndarray:
    """``segments + 1`` points 1 seg_len apart (straight-line distance) walking along ``path`` (corners rounded to
    ``corner``) from its first point. Raises if the path is too short."""
    dense = rounded(path, corner)
    pts = [dense[0]]
    i = 0
    while len(pts) < segments + 1:
        p = pts[-1]
        found = False
        while i < len(dense) - 1:
            a, b = dense[i], dense[i + 1]
            da, db = np.linalg.norm(a - p), np.linalg.norm(b - p)
            if da <= seg_len <= db:
                f = (seg_len - da) / max(db - da, 1e-12)
                q = a + f * (b - a)
                for _ in range(20):  # refine onto the sphere |q - p| = seg_len along a..b
                    d = np.linalg.norm(q - p)
                    f += (seg_len - d) / max(np.linalg.norm(b - a), 1e-12)
                    f = min(max(f, 0.0), 1.0)
                    q = a + f * (b - a)
                pts.append(q)
                found = True
                break
            i += 1
        if not found:
            raise ValueError(f"the path is too short for {segments} x {seg_len} m")
    return np.asarray(pts)


@dataclass(frozen=True)
class Section:
    name: str
    s0: float
    s1: float
    radius: float
    color: tuple


@dataclass(frozen=True)
class IronCord(Box):
    """A tool on a cord, laid along ``ends`` (segment end points, the anchored end first). Build with iron_cord()."""

    ends: tuple = ()  # (segments + 1) points, world, anchor first, the tool's rear last
    radius: float = 0.0025
    cord_mass: float = 0.03
    bend_stiffness: float = 0.002
    bend_damping: float = 0.0005
    armature: float = 5e-6
    max_bend: float = 1.2
    friction: float = 0.8
    cord_color: tuple = (0.2, 0.2, 0.2)
    sections: tuple = ()  # Section rows along the tool, s from its tip (the tool's frame origin) to its rear
    tool_len: float = 0.2
    tool_mass: float = 0.05
    tool_friction: float = 0.8
    sides: int = 12
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def free(self) -> bool:  # never jittered by yam_arena (its reset writes rigid poses only)
        return False

    @property
    def segments(self) -> int:
        return len(self.ends) - 1

    @property
    def seg_len(self) -> float:
        e = np.asarray(self.ends)
        return float(np.mean(np.linalg.norm(np.diff(e, axis=0), axis=1)))

    def _check(self) -> None:
        super()._check()
        if self.segments < 2 or not self.sections:
            raise ValueError(f"{self.name}: needs at least 2 segments and a tool")
        if self.seg_len <= 2 * self.radius:
            raise ValueError(f"{self.name}: segments must be longer than the cord's diameter")

    # -- the laid layout ---------------------------------------------------------------------------------------
    def segment_frames(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(centre, rotation) of every segment at spawn; x along the cord toward the tool."""
        e = np.asarray(self.ends, dtype=float)
        return [((e[i] + e[i + 1]) / 2, frame(e[i + 1] - e[i])) for i in range(self.segments)]

    def tool_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(tip, rotation) of the tool at spawn: x from the tip toward the rear (the cord continues it straight)."""
        e = np.asarray(self.ends, dtype=float)
        d = _unit(e[-1] - e[-2])  # the last segment points into the tool's rear
        tip = e[-1] + self.tool_len * d
        return tip, frame(-d)

    def anchor(self) -> np.ndarray:
        return np.asarray(self.ends[0], dtype=float)

    def rest_state(self) -> "IronCordState":
        tip, r = self.tool_frame()
        cs = np.asarray([c for c, _ in self.segment_frames()])
        return IronCordState.build(self, cs, tip, r[:, 0], quat_xyzw(r), source="spec")

    # tools/offline/check_task.py: the tool's own box (the cord is thin and lies on the bench), flagged approximate
    def offline_box(self):
        tip, r = self.tool_frame()
        rear = tip + self.tool_len * r[:, 0]
        rmax = max(s.radius for s in self.sections)
        lo, hi = np.minimum(tip, rear) - rmax, np.maximum(tip, rear) + rmax
        return tuple(((lo + hi) / 2).tolist()), tuple((hi - lo).tolist()), 0.0

    def offline_state(self, state) -> "IronCordState":
        return self.rest_state()


def iron_cord(name: str, path, *, segments: int, seg_len: float, radius: float, sections, tool_len: float,
              tool_mass: float, corner: float = 0.03, **kwargs) -> IronCord:
    """An IronCord whose cord follows ``path`` from the tool's rear (path[0]) to the anchored end (path[-1])."""
    pts = chain(path, seg_len, segments, corner)[::-1]  # anchor first
    secs = tuple(Section(*row) if not isinstance(row, Section) else row for row in sections)
    e = np.asarray(pts)
    d = _unit(e[-1] - e[-2])
    tip = e[-1] + tool_len * d
    rmax = max(s.radius for s in secs)
    lo = np.minimum(e.min(axis=0), tip) - max(radius, rmax)
    hi = np.maximum(e.max(axis=0), tip) + max(radius, rmax)
    cc = tuple(kwargs.pop("cord_color", (0.2, 0.2, 0.2)))
    spec = IronCord(name=name, pos=tuple(((lo + hi) / 2).tolist()), size=tuple((hi - lo).tolist()),
                    color=cc, cord_color=cc, ends=tuple(tuple(map(float, p)) for p in pts),
                    radius=radius, sections=secs, tool_len=tool_len, tool_mass=tool_mass, **kwargs)
    spec._check()
    return spec


@dataclass(frozen=True)
class IronCordState(ObjectState):
    """An ObjectState for an IronCord: ``points`` the segment centres (anchor first), ``tip`` the tool's tip,
    ``axis`` the unit vector from the tip toward the rear, ``rear`` the tool's rear (the cord's end)."""

    points: np.ndarray = None  # type: ignore[assignment]
    tip: np.ndarray = None  # type: ignore[assignment]
    axis: np.ndarray = None  # type: ignore[assignment]
    rear: np.ndarray = None  # type: ignore[assignment]
    tool_quat: np.ndarray = None  # type: ignore[assignment]
    anchor: np.ndarray = None  # type: ignore[assignment]
    seg_len: float = 0.0
    source: str = "sim"

    @staticmethod
    def build(spec: IronCord, points, tip, axis, tool_quat=None, lin_vel=None, ang_vel=None, source="sim"):
        pts = np.asarray(points, dtype=float)
        tip = np.asarray(tip, dtype=float)
        axis = _unit(axis)
        rear = tip + spec.tool_len * axis
        rmax = max(s.radius for s in spec.sections)
        lo = np.minimum(pts.min(axis=0) - spec.radius, np.minimum(tip, rear) - rmax)
        hi = np.maximum(pts.max(axis=0) + spec.radius, np.maximum(tip, rear) + rmax)
        return IronCordState(pos=(tip + rear) / 2, quat_xyzw=np.asarray(tool_quat if tool_quat is not None
                                                                        else (0.0, 0.0, 0.0, 1.0), dtype=float),
                             lin_vel=np.zeros(3) if lin_vel is None else np.asarray(lin_vel, dtype=float),
                             ang_vel=np.zeros(3) if ang_vel is None else np.asarray(ang_vel, dtype=float),
                             low=lo, high=hi, points=pts, tip=tip, axis=axis, rear=rear,
                             tool_quat=None if tool_quat is None else np.asarray(tool_quat, dtype=float),
                             anchor=spec.anchor(), seg_len=spec.seg_len, source=source)

    def at(self, s: float) -> np.ndarray:
        """The point on the tool's axis ``s`` from its tip."""
        return self.tip + float(s) * self.axis

    def cord_polyline(self) -> np.ndarray:
        """Anchor, segment centres, the tool's rear."""
        return np.vstack([self.anchor, self.points, self.rear])


# -- Isaac Sim side ------------------------------------------------------------------------------------------------
def _prism_points(r: float, s0: float, s1: float, sides: int):
    """A convex n-gon prism along x from s0 to s1 with its flats at radius r, one flat facing -z (lies flat)."""
    from pxr import Gf

    rv = r / math.cos(math.pi / sides)
    pts = []
    for x in (s0, s1):
        for k in range(sides):
            a = 2 * math.pi * (k + 0.5) / sides - math.pi / 2
            pts.append(Gf.Vec3f(float(x), float(rv * math.cos(a)), float(rv * math.sin(a))))
    return pts


def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the articulation under ``prim_path`` (pxr only). ``cfg.cord_name`` keys the IronCord spec."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: IronCord = _SPECS[cfg.cord_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    def look(name, color, rough=0.5):
        path = f"{prim_path}/Looks/{name}"
        c = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=rough)
        c.func(path, c)
        return path

    def material(name, mu):
        path = f"{prim_path}/Looks/{name}"
        m = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu)
        m.func(path, m)
        return path

    cord_look, cord_phys = look("cord", spec.cord_color), material("cord_physics", spec.friction)
    tool_phys = material("tool_physics", spec.tool_friction)
    frames = spec.segment_frames()
    c0, r0 = frames[0]
    L, rad, n = spec.seg_len, spec.radius, spec.segments

    def local(c, r):
        """World (centre, rotation) -> (translation, rotation) in the root frame (seg_0's frame)."""
        return r0.T @ (np.asarray(c) - c0), r0.T @ r

    def gquat(r):
        qx, qy, qz, qw = quat_xyzw(r)
        return Gf.Quatf(float(qw), float(qx), float(qy), float(qz))

    def body(path, c, r, mass):
        t, rl = local(c, r)
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in t)))
        qx, qy, qz, qw = quat_xyzw(rl)
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(qw), float(qx), float(qy),
                                                                               float(qz)))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateLinearDampingAttr(0.5)
        rb.CreateAngularDampingAttr(2.0)
        return prim

    def collide(prim, phys, vis):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.002)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), vis, stage=stage)

    def prism(path, r, s0, s1, phys, vis):
        mesh = UsdGeom.Mesh.Define(stage, path)
        pts = _prism_points(r, s0, s1, spec.sides)
        mesh.CreatePointsAttr(pts)
        m = spec.sides
        mesh.CreateFaceVertexCountsAttr([m, m] + [4] * m)
        idx = list(range(m)) + list(range(2 * m - 1, m - 1, -1))
        for k in range(m):
            idx += [k, (k + 1) % m, m + (k + 1) % m, m + k]
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreatePurposeAttr("guide")  # collision only, not rendered
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        collide(mesh.GetPrim(), phys, vis)

    seg_mass = spec.cord_mass / n
    seg_paths = []
    for i, (c, r) in enumerate(frames):
        path = f"{prim_path}/seg_{i}"
        body(path, c, r, seg_mass)
        cap = UsdGeom.Capsule.Define(stage, f"{path}/geom")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(rad)
        cap.CreateHeightAttr(max(L - 2 * rad, 1e-4) + 2 * rad * 0.9)
        bind_visual_material(str(cap.GetPath()), cord_look, stage=stage)
        prism(f"{path}/collider", rad, -L / 2 + 0.3 * rad, L / 2 - 0.3 * rad, cord_phys, cord_look)
        seg_paths.append(path)

    def joint(parent, child, name, pos0, pos1, rot1, bend):
        jp = f"{child}/{name}"
        j = UsdPhysics.Joint.Define(stage, jp) if bend else UsdPhysics.FixedJoint.Define(stage, jp)
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pos0)))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in pos1)))
        j.CreateLocalRot1Attr(gquat(rot1))  # the joint frame (the parent's) seen from the child: zero as laid
        if not bend:
            return
        prim = j.GetPrim()
        for dof in ("transX", "transY", "transZ", "rotX"):  # no stretch, no twist
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        for dof in ("rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(-math.degrees(spec.max_bend))
            lim.CreateHighAttr(math.degrees(spec.max_bend))
            drv = UsdPhysics.DriveAPI.Apply(prim, dof)
            drv.CreateTypeAttr("force")
            drv.CreateStiffnessAttr(spec.bend_stiffness * math.pi / 180)  # USD angular gains are per degree
            drv.CreateDampingAttr(spec.bend_damping * math.pi / 180)
            drv.CreateTargetPositionAttr(0.0)

    for i in range(1, n):
        joint(seg_paths[i - 1], seg_paths[i], f"bend_{i}", (L / 2, 0, 0), (-L / 2, 0, 0),
              frames[i][1].T @ frames[i - 1][1], bend=True)

    # the tool: its frame's origin at its tip, x toward the rear
    tip, rt = spec.tool_frame()
    tool_path = f"{prim_path}/tool"
    body(tool_path, tip, rt, spec.tool_mass)
    for sec in spec.sections:
        vis = look(f"tool_{sec.name}", sec.color, 0.35)
        cyl = UsdGeom.Cylinder.Define(stage, f"{tool_path}/{sec.name}_visual")
        cyl.CreateAxisAttr("X")
        cyl.CreateRadiusAttr(float(sec.radius))
        cyl.CreateHeightAttr(float(sec.s1 - sec.s0))
        cyl.AddTranslateOp().Set(Gf.Vec3d((sec.s0 + sec.s1) / 2, 0.0, 0.0))
        bind_visual_material(str(cyl.GetPath()), vis, stage=stage)
        prism(f"{tool_path}/{sec.name}_collider", sec.radius, sec.s0, sec.s1, tool_phys, vis)
    joint(seg_paths[-1], tool_path, "weld", (L / 2, 0, 0), (spec.tool_len, 0, 0), rt.T @ frames[-1][1], bend=False)

    root_link = stage.GetPrimAtPath(seg_paths[0])
    UsdPhysics.ArticulationRootAPI.Apply(root_link)  # seg_0 is the root link (its pose is the articulation's)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_link)
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    fj = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor")  # the plug in the station's socket
    fj.CreateBody1Rel().SetTargets([Sdf.Path(seg_paths[0])])
    wt = UsdGeom.Xformable(root_link).ComputeLocalToWorldTransform(0)
    t, q = wt.ExtractTranslation(), wt.ExtractRotationQuat()
    fj.CreateLocalPos0Attr(Gf.Vec3f(*t))
    fj.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root.GetPrim()


def _arena_object(spec: IronCord):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class IronCordSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn)
        cord_name: str = ""

    class IronCordObject(Object):
        def _generate_articulation_cfg(self):
            cfg = ArticulationCfg(
                prim_path=self.prim_path,
                spawn=self.spawner_cfg,
                actuators={"bend": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=spec.bend_stiffness,
                                                       damping=spec.bend_damping, armature=spec.armature)},
            )
            return self._add_initial_pose_to_cfg(cfg)

    c0, r0 = spec.segment_frames()[0]
    return IronCordObject(name=spec.name, object_type=ObjectType.ARTICULATION,
                          spawner_cfg=IronCordSpawnerCfg(cord_name=spec.name),
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in c0),
                                            rotation_xyzw=tuple(float(v) for v in quat_xyzw(r0))))


def _first(data, *names):
    for n in names:
        v = getattr(data, n, None)
        if v is not None:
            return v
    return None


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def read_state(env, spec: IronCord) -> IronCordState:
    base = env.unwrapped
    art = base.scene[spec.name]
    origin = _np(base.scene.env_origins)[0]
    names = list(art.body_names)
    pose7 = _first(art.data, "body_link_pose_w", "body_pose_w")
    if pose7 is not None:
        poses = _np(pose7)[0]
        pos, quat = poses[:, :3] - origin, poses[:, 3:7]
    else:
        pos = _np(art.data.body_pos_w)[0] - origin
        quat = _np(art.data.body_quat_w)[0]
    vel = _first(art.data, "body_link_vel_w", "body_vel_w", "body_com_vel_w")
    vel = _np(vel)[0] if vel is not None else None
    idx = [names.index(f"seg_{i}") for i in range(spec.segments)]
    k = names.index("tool")
    pts, tip = pos[idx], pos[k]
    # the tool continues the last segment straight (a fixed joint): its axis from positions, whatever the quat order
    axis = pts[-1] - tip
    return IronCordState.build(spec, pts, tip, axis, quat[k], None if vel is None else vel[k, :3],
                               None if vel is None else vel[k, 3:6], source="sim")


# -- hooking yam_arena.scene ---------------------------------------------------------------------------------------
_SPECS: dict[str, IronCord] = {}


def _patch_scene(scene) -> None:
    if getattr(scene, "_ironcord_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, IronCord):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = read_state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._ironcord_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            found = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if found is None or found.loader is None:
            return None
        loader = found.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_scene(module)

        loader.exec_module = exec_module
        return found


def install() -> None:
    """Hook yam_arena.scene (now if already imported, else when it is). Idempotent."""
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
