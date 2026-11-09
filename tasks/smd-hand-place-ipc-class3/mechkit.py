"""mechkit: compound rigid parts and jointed mechanisms for yam_arena task files, and their state for scoring.

A yam_arena task file can only declare single primitives (Box, upright Cylinder, Sphere, library Asset). This module
adds one more kind of object, an :class:`Assembly`, built from several boxes and cylinders:

- a **compound rigid body** (no joints): loose with a ``mass`` (an open tote, an L-bracket, a plate with a square hole,
  a T-handle, a tray with a rim) or fixed without one;
- a **mechanism**: that base body, fixed (no ``mass``) or loose, plus moving links on joints, each with limits, an
  optional spring toward its rest position, damping and joint friction: :func:`slider` (prismatic: drawer, sliding
  door, sash, gate) and :func:`hinge` (revolute: lid, flap, door, lever).

``success()``/``score()`` receive an :class:`AssemblyState` for it: a normal ObjectState (``pos`` is the centre of the
world-aligned bounding box, as for every other object) plus the base frame's ``origin``, ``joints`` and ``joint_vel``
(by link name) and every link's pose, with helpers (``to_local``, ``link_bounds``, ``in_body_box``, ``tilt``).

Use it from a task file (copy this file next to task.py)::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import mechkit as mk                     # importing it hooks yam_arena; nothing else to call

    TOTE = mk.assembly("tote", origin=(0.20, 0.27, TOP), mass=0.12,
                       shapes=mk.open_box((0, 0, 0.0225), (0.13, 0.075, 0.045), wall=0.006))
    CABINET = mk.assembly("cabinet", origin=(0.33, -0.05, TOP), shapes=CARCASS,      # no mass: fixed in place
                          joints=[mk.slider("drawer", "-X", DRAWER, upper=0.10, friction=1.0),
                                  mk.hinge("flap", "Y", (0.058, 0.082, 0.066), FLAP, upper=1.4, spring=0.15)])
    OBJECTS = [CABINET, TOTE]

    def success(objects):
        return objects["cabinet"].joints["drawer"] > 0.06 and objects["tote"].tilt() < 0.2

Conventions. Metres, kilograms, radians, seconds; world x forward, y left, z up (yam_arena's). Every shape centre,
of the base AND of every link, is given in the assembly's base frame with all joints at 0 (draw the closed cabinet
once). ``origin`` places that frame in the world and ``yaw`` turns it about the vertical. A joint's ``axis`` is
"X", "Y", "Z", "-X", "-Y" or "-Z" of the base frame at rest; its position is 0 as drawn, and positive moves the link
along the axis (slider) or turns it right-handedly about the axis through ``anchor`` (hinge). Limits must contain 0.
A spring (``spring``, N/m or N m/rad) pulls back toward 0; ``damping`` is N s/m or N m s/rad; ``friction`` is passed
to Isaac Lab's ``ImplicitActuatorCfg.friction``, the same field yam_arena fills with the arms' joint friction (an
effort in N or N m in Isaac Sim 5+: see MECHKIT.md). Links of one assembly never collide with each other (a drawer
slides inside its cabinet on its joint alone); they collide with everything else.

What importing does. It hooks ``yam_arena.scene.task_object`` (an Assembly is spawned from USD prims this module
writes), ``yam_arena.scene.object_states`` (it reports an AssemblyState) and ``YamArenaEmbodiment._task_step`` /
``reset`` (episode and step counters, an optional per-step trace, a one-time printout of what Isaac Lab built, and a
check that every Assembly was really built by this module; the run stops with an error if one was not).
``yam_arena.scene`` is imported only after Isaac Sim starts, so the scene hook is applied from a ``sys.meta_path``
finder when that import happens. yam_arena itself is not modified. Hooks are idempotent, survive this module being
imported twice, and coexist with ``tools/kit/rigstate.py`` (both wrap the same two embodiment methods).

Everything that needs Isaac Sim, Isaac Lab, Arena, torch or pxr is imported inside functions: importing this module
needs only numpy and the pure-Python ``yam_arena.task_file`` (the real one or tools/offline's stub). Offline, the
checker in tools/offline sees an Assembly as its bounding box at rest and hands the scorer an AssemblyState with every
joint at 0 (``Assembly.offline_box``, ``Assembly.offline_state``).

Environment: ``MECHKIT_TRACE=<file>`` appends one JSON line per control step (every assembly's joints and pose, and
the step's info); ``MECHKIT_PRINT_EVERY=<n>`` prints a one-line summary to stderr every n steps (``configure()``
sets defaults for both).

The techniques (subclassing ``task_file.Box`` so the loader accepts the object, patching ``yam_arena.scene`` from a
``sys.meta_path`` hook at the task file's import, authoring the bodies and joints with pxr inside a custom spawner,
springs as implicit actuators, anchoring a fixed base with a world joint) follow those first shown in the program
repository's tasks/bearing-press (partkit) and used by later tasks (keyed-plug-loose-socket, laptop-under-stack,
lock-before-loading, the cable tasks); this is an independent implementation. Written for yam_arena on Isaac
Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1.
"""

from __future__ import annotations

import functools
import importlib.abc
import json
import math
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from yam_arena.task_file import GREY, Box, ObjectState

__all__ = [
    "Assembly",
    "AssemblyState",
    "Joint",
    "Shape",
    "assembly",
    "box",
    "configure",
    "cylinder",
    "env",
    "episode",
    "hinge",
    "install",
    "offline_reset",
    "open_box",
    "plate_with_hole",
    "read_state",
    "slider",
    "state_from_poses",
    "status",
    "step",
    "usd_plan",
]

VERSION = "1"
AXES = ("X", "Y", "Z", "-X", "-Y", "-Z")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")  # link names become USD prim names and Isaac Lab regexes
RESERVED_LINK_NAMES = frozenset({"base", "joints", "Looks"})  # prims an assembly already owns under its root

# Physics settings for every assembly (overridable per assembly where a field exists).
FIXED_BASE_MASS = 1.0  # kg: a fixed base is still an articulation link; its mass does not matter
DEFAULT_CONTACT_OFFSET = 0.005  # m, as yam_arena gives its loose primitives (scene._contact_props)
REST_OFFSET = 0.0
DEFAULT_SOLVER_ITERATIONS = 16  # position iterations, as yam_arena's arms
SOLVER_VELOCITY_ITERATIONS = 4
MAX_DEPENETRATION_VELOCITY = 1.0  # m/s
LOOK_ROUGHNESS = 0.5
DEFAULT_SURFACE_FRICTION = (1.2, 1.0)  # static, dynamic: yam_arena.constants.GRIPPY_FRICTION, every native object's
# Where a FIXED-base articulation's ArticulationRootAPI goes: "xform" (the Xform above the base, which PhysX reads as
# a fixed-base articulation together with the world joint; the default) or "base" (on the base body itself, the
# layout of the URDF-converted USD in tasks/lock-before-loading). MECHKIT_FIXED_ROOT=base switches, for an A/B test.
FIXED_ROOT_MODES = ("xform", "base")


# == small helpers ====================================================================================================


def _vec(value, n: int, what: str) -> tuple[float, ...]:
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be {n} numbers, got {value!r}") from None
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what} must be {n} finite numbers, got {value!r}")
    return out


def _vec3(value, what: str) -> tuple[float, float, float]:
    return _vec(value, 3, what)  # type: ignore[return-value]


def _rgb(value, what: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    rgb = _vec3(value, what)
    if not all(0.0 <= c <= 1.0 for c in rgb):
        raise ValueError(f"{what} must be sRGB in 0..1, got {value!r}")
    return rgb


def _number(value, what: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(x) or (positive and not x > 0) or (nonnegative and not x >= 0):
        need = "positive" if positive else "non-negative" if nonnegative else "finite"
        raise ValueError(f"{what} must be {need}, got {value!r}")
    return x


def _try_tuple(value):
    """``value`` as a tuple of floats when it is a sequence of numbers, else unchanged (validation reports it)."""
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return value


def quat_to_matrix(quat_xyzw) -> np.ndarray:
    """3x3 rotation of a unit quaternion (x, y, z, w)."""
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not n > 0:
        raise ValueError(f"not a rotation quaternion: {quat_xyzw!r}")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r) -> np.ndarray:
    """Unit quaternion (x, y, z, w), w >= 0, of a 3x3 rotation."""
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


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    """Rotation by ``angle`` (rad) about the unit vector ``axis`` (right hand)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def axis_vector(axis: str) -> np.ndarray:
    """Unit vector of a signed axis name ("X" .. "-Z")."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    v = np.zeros(3)
    v[_AXIS_INDEX[axis[-1]]] = -1.0 if axis.startswith("-") else 1.0
    return v


def _joint_frame_wxyz(axis: str) -> tuple[float, float, float, float]:
    """USD joints turn about / slide along +X, +Y or +Z of their joint frame. For a negative axis the joint frame is
    the body frame turned half a turn about another axis, so that the joint's positive axis points the other way."""
    if not axis.startswith("-"):
        return (1.0, 0.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0, 0.0) if axis == "-Z" else (0.0, 0.0, 0.0, 1.0)  # about X for -Z, about Z for -X / -Y


def _np(value) -> np.ndarray:
    """A float64 array from an Isaac Lab 3 data buffer (``.torch``), a torch tensor, or anything array-like."""
    t = getattr(value, "torch", None)
    if t is not None and not callable(t):
        value = t
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _first_attr(obj, *names):
    """The first of ``names`` that ``obj`` has (and is not None), as (name, value); (None, None) if none."""
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - lazy Isaac Lab properties may raise
            continue
        if value is not None:
            return name, value
    return None, None


# == declarations =====================================================================================================


@dataclass(frozen=True)
class Shape:
    """One collider (and its look) of an assembly body: a ``box`` (edge lengths ``size``) or a ``cylinder``
    (``radius``, ``height`` along ``axis`` "X", "Y" or "Z"). ``center`` is in the assembly's base frame with every
    joint at 0. ``color`` (sRGB) defaults to its link's, then the assembly's. Build with :func:`box` /
    :func:`cylinder`."""

    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    height: float = 0.0
    axis: str = "Z"
    color: tuple[float, float, float] | None = None

    def check(self, what: str = "shape") -> None:
        if self.kind not in ("box", "cylinder"):
            raise ValueError(f"{what}: kind must be 'box' or 'cylinder', got {self.kind!r}")
        _vec3(self.center, f"{what} center")
        if self.kind == "box":
            if not all(s > 0 for s in _vec3(self.size, f"{what} size")):
                raise ValueError(f"{what}: box size must be positive, got {self.size!r}")
        else:
            _number(self.radius, f"{what} radius", positive=True)
            _number(self.height, f"{what} height", positive=True)
            if self.axis not in _AXIS_INDEX:
                raise ValueError(f"{what}: cylinder axis must be 'X', 'Y' or 'Z', got {self.axis!r}")
        _rgb(self.color, f"{what} color")

    def half_extents(self) -> np.ndarray:
        """Half its extent along the frame's x, y, z (it is axis-aligned in the base frame)."""
        if self.kind == "box":
            return np.asarray(self.size, dtype=float) / 2.0
        half = np.full(3, float(self.radius))
        half[_AXIS_INDEX[self.axis]] = self.height / 2.0
        return half

    def volume(self) -> float:
        if self.kind == "box":
            return float(np.prod(self.size))
        return math.pi * self.radius**2 * self.height

    def aabb_half(self, rotation) -> np.ndarray:
        """Half extents of its world-aligned bounding box once its frame is turned by ``rotation`` (exact)."""
        r = np.asarray(rotation, dtype=float)
        if self.kind == "box":
            return np.abs(r) @ (np.asarray(self.size, dtype=float) / 2.0)
        a = r[:, _AXIS_INDEX[self.axis]]
        return self.radius * np.sqrt(np.clip(1.0 - a * a, 0.0, 1.0)) + self.height / 2.0 * np.abs(a)


def box(center, size, color=None) -> Shape:
    """A box collider: ``center`` and ``size`` (x, y, z edge lengths) in metres, base frame at rest."""
    s = Shape("box", _vec3(center, "box center"), _vec3(size, "box size"), color=_rgb(color, "box color"))
    s.check("box")
    return s


def cylinder(center, radius: float, height: float, axis: str = "Z", color=None) -> Shape:
    """A cylinder collider along ``axis`` ("X", "Y" or "Z": a lying cylinder is fine here)."""
    s = Shape("cylinder", _vec3(center, "cylinder center"), radius=float(radius), height=float(height),
              axis=str(axis).upper(), color=_rgb(color, "cylinder color"))
    s.check("cylinder")
    return s


def open_box(center, outer, wall: float, floor: float | None = None, color=None) -> tuple[Shape, ...]:
    """An open-top box (a tote, a tray with a rim, a bin): a floor and four walls filling the outer box ``outer``
    (x, y, z) centred on ``center``. ``wall`` is the wall thickness, ``floor`` the floor's (default ``wall``)."""
    cx, cy, cz = _vec3(center, "open_box center")
    sx, sy, sz = _vec3(outer, "open_box outer")
    t = _number(wall, "open_box wall", positive=True)
    f = t if floor is None else _number(floor, "open_box floor", positive=True)
    if not (2 * t < min(sx, sy) and f < sz):
        raise ValueError(f"open_box: walls {t} / floor {f} do not fit in the outer size {outer!r}")
    h = sz - f
    zc = cz - sz / 2 + f + h / 2
    return (
        box((cx, cy, cz - sz / 2 + f / 2), (sx, sy, f), color),
        box((cx + sx / 2 - t / 2, cy, zc), (t, sy, h), color),
        box((cx - sx / 2 + t / 2, cy, zc), (t, sy, h), color),
        box((cx, cy + sy / 2 - t / 2, zc), (sx - 2 * t, t, h), color),
        box((cx, cy - sy / 2 + t / 2, zc), (sx - 2 * t, t, h), color),
    )


def plate_with_hole(center, outer_xy, hole_xy, thickness: float, hole_offset=(0.0, 0.0), color=None
                    ) -> tuple[Shape, ...]:
    """A flat plate (``outer_xy`` by ``thickness``, centred on ``center``) with a rectangular through-hole
    ``hole_xy`` whose centre is ``hole_offset`` from the plate's: up to four boxes around the hole."""
    cx, cy, cz = _vec3(center, "plate center")
    ox, oy = _vec(outer_xy, 2, "plate outer_xy")
    hx, hy = _vec(hole_xy, 2, "plate hole_xy")
    dx, dy = _vec(hole_offset, 2, "plate hole_offset")
    t = _number(thickness, "plate thickness", positive=True)
    x0, x1, y0, y1 = cx - ox / 2, cx + ox / 2, cy - oy / 2, cy + oy / 2
    hx0, hx1, hy0, hy1 = cx + dx - hx / 2, cx + dx + hx / 2, cy + dy - hy / 2, cy + dy + hy / 2
    if not (hx > 0 and hy > 0 and x0 <= hx0 < hx1 <= x1 and y0 <= hy0 < hy1 <= y1) or (hx0, hx1, hy0, hy1) == (
            x0, x1, y0, y1):
        raise ValueError("plate_with_hole: the hole must lie inside the plate and leave some of it")
    out = []
    for a, b in ((x0, hx0), (hx1, x1)):  # strips left and right of the hole, full depth
        if b - a > 1e-9:
            out.append(box(((a + b) / 2, cy, cz), (b - a, oy, t), color))
    for a, b in ((y0, hy0), (hy1, y1)):  # strips in front of and behind the hole, hole width
        if b - a > 1e-9:
            out.append(box(((hx0 + hx1) / 2, (a + b) / 2, cz), (hx1 - hx0, b - a, t), color))
    return tuple(out)


@dataclass(frozen=True)
class Joint:
    """A moving link of an assembly and the joint that carries it. Build with :func:`slider` or :func:`hinge`.

    ``kind`` "prismatic" or "revolute"; ``axis`` a signed base-frame axis; ``anchor`` a point on the joint axis and
    the link frame's origin (base frame, at rest); ``shapes`` in the base frame at rest; ``lower``/``upper`` limits
    (m or rad) around 0; ``mass`` kg; ``spring`` toward 0, ``damping``, ``friction`` (see the module docstring);
    ``parent`` "base" or an earlier link's name."""

    name: str
    kind: str
    axis: str
    anchor: tuple[float, float, float]
    shapes: tuple[Shape, ...]
    lower: float
    upper: float
    mass: float = 0.05
    spring: float = 0.0
    damping: float = 0.0
    friction: float = 0.0
    parent: str = "base"
    color: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _try_tuple(self.anchor))
        object.__setattr__(self, "shapes", tuple(self.shapes))
        object.__setattr__(self, "axis", str(self.axis).upper())
        if self.color is not None:
            object.__setattr__(self, "color", _try_tuple(self.color))

    @property
    def unit(self) -> str:
        return "m" if self.kind == "prismatic" else "rad"

    def axis_vector(self) -> np.ndarray:
        return axis_vector(self.axis)

    def check(self, owner: str = "assembly") -> None:
        what = f"{owner}: link {self.name!r}"
        if not isinstance(self.name, str) or not _NAME.match(self.name) or self.name in RESERVED_LINK_NAMES:
            raise ValueError(f"{what}: a link name must be a letter then letters, digits or _, and not one of "
                             f"{sorted(RESERVED_LINK_NAMES)}")
        if self.kind not in ("prismatic", "revolute"):
            raise ValueError(f"{what}: kind must be 'prismatic' or 'revolute', got {self.kind!r}")
        if self.axis not in AXES:
            raise ValueError(f"{what}: axis must be one of {AXES}, got {self.axis!r}")
        _vec3(self.anchor, f"{what} anchor")
        if not self.shapes:
            raise ValueError(f"{what}: a link needs at least one shape")
        for i, s in enumerate(self.shapes):
            if not isinstance(s, Shape) and type(s).__name__ != "Shape":
                raise TypeError(f"{what}: shape {i} is {s!r}; build shapes with box() or cylinder()")
            s.check(f"{what} shape {i}")
        lo = _number(self.lower, f"{what} lower")
        hi = _number(self.upper, f"{what} upper")
        if not lo <= 0.0 <= hi or not lo < hi:
            raise ValueError(f"{what}: limits [{lo}, {hi}] must contain 0 (the pose as drawn) and lower < upper")
        bound = 2 * math.pi if self.kind == "revolute" else 2.0
        if max(-lo, hi) > bound:
            raise ValueError(f"{what}: limits [{lo}, {hi}] exceed +-{bound:.3f} {self.unit}")
        _number(self.mass, f"{what} mass", positive=True)
        for label in ("spring", "damping", "friction"):
            _number(getattr(self, label), f"{what} {label}", nonnegative=True)
        if not isinstance(self.parent, str):
            raise ValueError(f"{what}: parent must be a link name, got {self.parent!r}")
        _rgb(self.color, f"{what} color")


def _shapes(shapes, what: str) -> tuple[Shape, ...]:
    if isinstance(shapes, Shape):
        shapes = (shapes,)
    try:
        out = tuple(shapes)
    except TypeError:
        raise TypeError(f"{what}: shapes must be a list of box()/cylinder() shapes, got {shapes!r}") from None
    return out


def _bbox_center(shapes) -> tuple[float, float, float]:
    lo = np.min([np.asarray(s.center) - s.half_extents() for s in shapes], axis=0)
    hi = np.max([np.asarray(s.center) + s.half_extents() for s in shapes], axis=0)
    return tuple(float(v) for v in (lo + hi) / 2)


def slider(name: str, axis: str, shapes, *, upper: float, lower: float = 0.0, anchor=None, mass: float = 0.1,
           spring: float = 0.0, damping: float = 1.0, friction: float = 0.0, parent: str = "base",
           color=None) -> Joint:
    """A link on a prismatic joint (drawer, sliding door, sash, gate) moving along ``axis`` from ``lower`` to
    ``upper`` metres (0 = as drawn). ``anchor`` (the link frame's origin, reported as its pose) defaults to the
    centre of its shapes. ``spring`` N/m toward 0, ``damping`` N s/m."""
    shapes = _shapes(shapes, f"slider {name!r}")
    if anchor is None and shapes and all(isinstance(s, Shape) for s in shapes):
        anchor = _bbox_center(shapes)
    j = Joint(name, "prismatic", axis, anchor if anchor is not None else (0.0, 0.0, 0.0), shapes, lower, upper,
              mass, spring, damping, friction, parent, color)
    j.check("slider")
    return j


def hinge(name: str, axis: str, anchor, shapes, *, upper: float, lower: float = 0.0, mass: float = 0.05,
          spring: float = 0.0, damping: float = 0.01, friction: float = 0.0, parent: str = "base",
          color=None) -> Joint:
    """A link on a revolute joint (lid, flap, door, lever) turning about ``axis`` through ``anchor`` from ``lower``
    to ``upper`` radians (0 = as drawn; positive is right-handed about the axis). ``spring`` N m/rad toward 0,
    ``damping`` N m s/rad."""
    j = Joint(name, "revolute", axis, anchor, _shapes(shapes, f"hinge {name!r}"), lower, upper, mass, spring,
              damping, friction, parent, color)
    j.check("hinge")
    return j


@dataclass(frozen=True)
class Assembly(Box):
    """A compound rigid body or a mechanism, as one task-file object. It subclasses ``task_file.Box`` only so
    yam_arena's loader accepts it. ``pos`` is the BASE FRAME'S ORIGIN in the world (not a bounding-box centre) and
    ``yaw`` turns that frame about the vertical; ``size`` is filled in as the extent of every shape at rest (base
    frame). ``mass`` None: the base is fixed in place; otherwise it is loose. ``friction`` is the surface friction
    of every collider, a number or (static, dynamic); None gives yam_arena's (1.2, 1.0). Build with
    :func:`assembly`."""

    shapes: tuple[Shape, ...] = ()
    joints: tuple[Joint, ...] = ()
    friction: Any = None
    contact_offset: float = DEFAULT_CONTACT_OFFSET
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS

    _mechkit_assembly = True  # a class attribute, not a field: how the hooks recognise an Assembly across reloads

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _try_tuple(self.pos))
        object.__setattr__(self, "color", _try_tuple(self.color))
        object.__setattr__(self, "shapes", tuple(self.shapes))
        object.__setattr__(self, "joints", tuple(self.joints))
        if isinstance(self.friction, (list, tuple)):
            object.__setattr__(self, "friction", _try_tuple(self.friction))
        try:
            lo, hi = self.local_bounds()
            object.__setattr__(self, "size", tuple(float(v) for v in hi - lo))
        except Exception:  # noqa: BLE001 - invalid shapes: _check() reports them
            pass

    # -- validation --------------------------------------------------------------------------------------------------

    def _check(self) -> None:
        super()._check()  # name, pos, color, mass, size
        if not math.isfinite(float(self.yaw)):
            raise ValueError(f"{self.name}.yaw must be finite, got {self.yaw!r}")
        if not self.shapes:
            raise ValueError(f"{self.name}: an Assembly needs at least one base shape")
        for i, s in enumerate(self.shapes):
            if not isinstance(s, Shape) and type(s).__name__ != "Shape":
                raise TypeError(f"{self.name}: base shape {i} is {s!r}; build shapes with box() or cylinder()")
            s.check(f"{self.name}: base shape {i}")
        seen: list[str] = []
        for j in self.joints:
            if not isinstance(j, Joint) and type(j).__name__ != "Joint":
                raise TypeError(f"{self.name}: joints holds {j!r}; build links with slider() or hinge()")
            j.check(self.name)
            if j.name in seen:
                raise ValueError(f"{self.name}: link names must be unique, {j.name!r} repeats")
            if j.parent != "base" and j.parent not in seen:
                raise ValueError(f"{self.name}: link {j.name!r} has parent {j.parent!r}, which must be 'base' or a "
                                 f"link listed before it")
            seen.append(j.name)
        self.surface_friction()
        _number(self.contact_offset, f"{self.name}.contact_offset", positive=True)
        if not self.contact_offset <= 0.05:
            raise ValueError(f"{self.name}.contact_offset must be at most 0.05 m, got {self.contact_offset!r}")
        if not (isinstance(self.solver_iterations, int) and 1 <= self.solver_iterations <= 255):
            raise ValueError(f"{self.name}.solver_iterations must be an int in 1..255, got {self.solver_iterations!r}")

    def surface_friction(self) -> tuple[float, float]:
        f = self.friction
        if f is None:
            return DEFAULT_SURFACE_FRICTION
        if isinstance(f, (int, float)):
            v = _number(f, f"{self.name}.friction", nonnegative=True)
            return (v, v)
        static, dynamic = _vec(f, 2, f"{self.name}.friction")
        if min(static, dynamic) < 0:
            raise ValueError(f"{self.name}.friction must be non-negative, got {f!r}")
        return (static, dynamic)

    # -- structure ---------------------------------------------------------------------------------------------------

    @property
    def kind(self) -> str:
        """"static" (fixed, no joints: colliders only), "rigid" (one loose body) or "articulation"."""
        if self.joints:
            return "articulation"
        return "rigid" if self.mass is not None else "static"

    @property
    def fixed_base(self) -> bool:
        return self.mass is None

    @property
    def body_names(self) -> tuple[str, ...]:
        return ("base", *(j.name for j in self.joints))

    def link(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"{self.name} has no link {name!r} (links: {[j.name for j in self.joints]})")

    def body_offset(self, body: str) -> np.ndarray:
        """Where a body's frame origin sits in the base frame at rest (the base's: 0; a link's: its anchor)."""
        return np.zeros(3) if body == "base" else np.asarray(self.link(body).anchor, dtype=float)

    def body_shapes(self, body: str) -> tuple[Shape, ...]:
        """A body's shapes, centres in the base frame at rest."""
        return self.shapes if body == "base" else self.link(body).shapes

    def body_color(self, body: str, shape: Shape) -> tuple[float, float, float]:
        if shape.color is not None:
            return tuple(shape.color)
        if body != "base" and self.link(body).color is not None:
            return tuple(self.link(body).color)
        return tuple(self.color)

    def body_mass(self, body: str) -> float:
        if body != "base":
            return float(self.link(body).mass)
        return float(self.mass) if self.mass is not None else FIXED_BASE_MASS

    # -- geometry (base frame unless stated) -------------------------------------------------------------------------

    def link_transforms(self, joints: dict | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each body's frame (rotation, origin) in the base frame, with the joints at ``joints`` (link name ->
        position; missing ones at 0)."""
        q = joints or {}
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {"base": (np.eye(3), np.zeros(3))}
        for j in self.joints:  # parents come first (checked)
            rp, pp = out[j.parent]
            rel = np.asarray(j.anchor, dtype=float) - self.body_offset(j.parent)
            a = j.axis_vector()
            v = float(q.get(j.name, 0.0))
            if j.kind == "prismatic":
                out[j.name] = (rp, pp + rp @ (rel + a * v))
            else:
                out[j.name] = (rp @ axis_angle_matrix(a, v), pp + rp @ rel)
        return out

    def local_bounds(self, joints: dict | None = None, bodies=None) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) of the base-frame-aligned box around ``bodies`` (default all) with the joints at ``joints``."""
        frames = self.link_transforms(joints) if joints else None
        los, his = [], []
        for body in bodies or self.body_names:
            off = self.body_offset(body)
            for s in self.body_shapes(body):
                if frames is None:  # at rest every shape is axis-aligned where it was drawn
                    c, h = np.asarray(s.center, dtype=float), s.half_extents()
                else:
                    r, p = frames[body]
                    c, h = p + r @ (np.asarray(s.center, dtype=float) - off), s.aabb_half(r)
                los.append(c - h)
                his.append(c + h)
        if not los:
            raise ValueError(f"{self.name}: no shapes")
        return np.min(los, axis=0), np.max(his, axis=0)

    def center_of_mass(self, body: str = "base") -> np.ndarray:
        """A body's centre of mass at rest (base frame), its mass spread evenly over its shapes' volume; this is how
        PhysX derives it from the colliders when only a mass is given (overlapping volume counts twice there too)."""
        shapes = self.body_shapes(body)
        vols = np.array([s.volume() for s in shapes])
        centres = np.array([s.center for s in shapes], dtype=float)
        return (vols[:, None] * centres).sum(axis=0) / vols.sum()

    def total_center_of_mass(self, joints: dict | None = None) -> np.ndarray:
        """Centre of mass of the base and all links together (base frame), joints at ``joints``. For a fixed base
        the base counts with its nominal mass; use it for loose assemblies."""
        frames = self.link_transforms(joints)
        total, acc = 0.0, np.zeros(3)
        for body in self.body_names:
            r, p = frames[body]
            m = self.body_mass(body)
            acc += m * (p + r @ (self.center_of_mass(body) - self.body_offset(body)))
            total += m
        return acc / total

    def spawn_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The base frame's world origin and orientation (x, y, z, w) at spawn."""
        return np.asarray(self.pos, dtype=float), yaw_quat(float(self.yaw))

    def world_bounds(self, joints: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World-aligned (low, high) at the spawn pose with the joints at ``joints``."""
        s = self.rest_state() if not joints else state_from_poses(self, (*self.pos, *yaw_quat(self.yaw)),
                                                                  joints=joints, source="spec")
        return s.low, s.high

    # -- offline (tools/offline/check_task.py) and spec-only states ------------------------------------------------

    def offline_box(self) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """(centre, size, yaw) of the yawed box around every shape at rest: what the offline checker draws."""
        lo, hi = self.local_bounds()
        origin, quat = self.spawn_pose()
        centre = origin + quat_to_matrix(quat) @ ((lo + hi) / 2)
        return tuple(float(v) for v in centre), tuple(float(v) for v in hi - lo), float(self.yaw)

    def rest_state(self) -> AssemblyState:
        """The state at spawn, every joint at 0 (also what a fixed compound reports in the simulator: it never
        moves)."""
        origin, quat = self.spawn_pose()
        return state_from_poses(self, (*origin, *quat), source="spec")

    def offline_state(self, state) -> AssemblyState:
        """An AssemblyState from the plain ObjectState the offline checker computes for this object's box (its
        ``pos`` is that box's centre): same pose and velocities, every joint at 0."""
        lo, hi = self.local_bounds()
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - quat_to_matrix(quat) @ ((lo + hi) / 2)
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return state_from_poses(self, (*origin, *quat), vel, source="offline")


def assembly(name: str, origin, shapes, *, mass: float | None = None, yaw: float = 0.0, joints=(), color=GREY,
             friction=None, contact_offset: float = DEFAULT_CONTACT_OFFSET,
             solver_iterations: int = DEFAULT_SOLVER_ITERATIONS) -> Assembly:
    """Declare an assembly: ``shapes`` (the base body) and ``joints`` (links from :func:`slider` / :func:`hinge`)
    in the base frame, which sits at ``origin`` (world) turned ``yaw`` about the vertical. Without ``mass`` the
    base is fixed in place; with it, loose. Checked now, and again by yam_arena's loader."""
    spec = Assembly(name=name, pos=origin, color=color, mass=mass, yaw=yaw, shapes=_shapes(shapes, name),
                    joints=tuple(joints), friction=friction, contact_offset=contact_offset,
                    solver_iterations=solver_iterations)
    spec._check()
    return spec


# == state ============================================================================================================


@dataclass(frozen=True)
class AssemblyState(ObjectState):
    """An Assembly after a control step (world frame, the env origin subtracted). The ObjectState fields mean what
    they mean for every object: ``pos`` the centre of the world-aligned box ``low``..``high`` around all its shapes,
    ``quat_xyzw``/``lin_vel``/``ang_vel`` the base body's (zero velocities for a fixed base). Added: ``origin``, the
    base frame's origin; ``joints`` / ``joint_vel``, link name -> position (m or rad, 0 as drawn) and velocity;
    ``links``, link name -> its frame's pose (x, y, z, qx, qy, qz, qw); ``source`` "sim", "spec" (computed from the
    declaration: a fixed compound) or "offline"."""

    origin: np.ndarray = None  # type: ignore[assignment]
    joints: dict = field(default_factory=dict)
    joint_vel: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    source: str = "sim"
    spec: Any = field(default=None, repr=False, compare=False)

    def rotation(self) -> np.ndarray:
        """3x3 matrix taking base-frame vectors to the world."""
        return quat_to_matrix(self.quat_xyzw)

    def to_world(self, point) -> np.ndarray:
        """Base-frame point(s) (3,) or (N, 3) in the world."""
        return np.asarray(point, dtype=float) @ self.rotation().T + np.asarray(self.origin, dtype=float)

    def to_local(self, point) -> np.ndarray:
        """World point(s) (3,) or (N, 3) in the base frame."""
        return (np.asarray(point, dtype=float) - np.asarray(self.origin, dtype=float)) @ self.rotation()

    def tilt(self) -> float:
        """Angle (rad) between the base frame's z axis and the vertical."""
        return math.acos(max(-1.0, min(1.0, float(self.rotation()[2, 2]))))

    def joint(self, name: str) -> float:
        if name not in self.joints:
            raise KeyError(f"no joint {name!r}; joints: {sorted(self.joints)}")
        return float(self.joints[name])

    def link_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """(rotation, origin) of a body's frame in the world ("base" or a link name)."""
        if body == "base":
            return self.rotation(), np.asarray(self.origin, dtype=float)
        if body not in self.links:
            raise KeyError(f"no link {body!r}; links: {sorted(self.links)}")
        pose = np.asarray(self.links[body], dtype=float)
        return quat_to_matrix(pose[3:7]), pose[:3]

    def link_to_world(self, body: str, point) -> np.ndarray:
        """A point given in the base frame AT REST that rides on ``body`` (e.g. a spot on the drawer's floor as
        drawn), where it is now in the world."""
        r, p = self.link_pose(body)
        off = self.spec.body_offset(body) if self.spec is not None else np.zeros(3)
        return (np.asarray(point, dtype=float) - off) @ r.T + p

    def link_bounds(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """World-aligned (low, high) around one body's shapes as they are now."""
        if self.spec is None:
            raise RuntimeError("this state carries no Assembly spec")
        r, p = self.link_pose(body)
        off = self.spec.body_offset(body)
        los, his = [], []
        for s in self.spec.body_shapes(body):
            c, h = p + r @ (np.asarray(s.center, dtype=float) - off), s.aabb_half(r)
            los.append(c - h)
            his.append(c + h)
        return np.min(los, axis=0), np.max(his, axis=0)

    def in_body_box(self, body: str, point, margin: float = 0.0) -> bool:
        """Whether a world point is inside the box around ``body``'s shapes, taken in that body's own frame (so it
        turns with a lid and slides with a drawer), grown by ``margin`` (m; negative shrinks it). E.g. "the cube's
        centre is in the drawer": ``s.in_body_box("drawer", cube.pos)``."""
        if self.spec is None:
            raise RuntimeError("this state carries no Assembly spec")
        r, p = self.link_pose(body)
        off = self.spec.body_offset(body)
        local = (np.asarray(point, dtype=float) - p) @ r + off  # back in the base frame at rest
        lo, hi = self.spec.local_bounds(bodies=(body,))
        return bool(np.all(local >= lo - margin) and np.all(local <= hi + margin))


def state_from_poses(spec: Assembly, base_pose, base_vel=None, link_poses: dict | None = None,
                     joints: dict | None = None, joint_vel: dict | None = None, source: str = "sim") -> AssemblyState:
    """Assemble an AssemblyState. ``base_pose`` (x, y, z, qx, qy, qz, qw) and ``base_vel`` (vx, vy, vz, wx, wy, wz)
    are the base body's (world, env origin subtracted); ``link_poses`` each link's frame pose (or None: computed
    from ``joints`` by forward kinematics)."""
    pose = np.asarray(base_pose, dtype=float).reshape(7)
    origin, quat = pose[:3].copy(), pose[3:7] / np.linalg.norm(pose[3:7])
    rb = quat_to_matrix(quat)
    q = {j.name: float((joints or {}).get(j.name, 0.0)) for j in spec.joints}
    qd = {j.name: float((joint_vel or {}).get(j.name, 0.0)) for j in spec.joints}
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if link_poses is None:
        for body, (r, p) in spec.link_transforms(q).items():
            frames[body] = (rb @ r, origin + rb @ p)
    else:
        frames["base"] = (rb, origin)
        for j in spec.joints:
            if j.name not in link_poses:
                raise ValueError(f"{spec.name}: no pose for link {j.name!r}")
            lp = np.asarray(link_poses[j.name], dtype=float).reshape(7)
            frames[j.name] = (quat_to_matrix(lp[3:7]), lp[:3].copy())
    los, his = [], []
    for body in spec.body_names:
        r, p = frames[body]
        off = spec.body_offset(body)
        for s in spec.body_shapes(body):
            c, h = p + r @ (np.asarray(s.center, dtype=float) - off), s.aabb_half(r)
            los.append(c - h)
            his.append(c + h)
    low, high = np.min(los, axis=0), np.max(his, axis=0)
    links = {j.name: np.concatenate([frames[j.name][1], matrix_to_quat(frames[j.name][0])]) for j in spec.joints}
    vel = np.zeros(6) if base_vel is None else np.asarray(base_vel, dtype=float).reshape(6)
    return AssemblyState(pos=(low + high) / 2, quat_xyzw=quat, lin_vel=vel[:3].copy(), ang_vel=vel[3:].copy(),
                         low=low, high=high, origin=origin, joints=q, joint_vel=qd, links=links, source=source,
                         spec=spec)


# == the USD layout (pure Python, so it is testable offline) ==========================================================


def usd_plan(spec: Assembly, root: str) -> dict:
    """What :func:`_author` writes under ``root`` for ``spec``, in USD units (degrees and per-degree gains for
    revolute joints, as UsdPhysics wants them)::

        <root>                 Xform at the spawn pose; ArticulationRootAPI if a FIXED-base articulation
          base                 Xform; RigidBodyAPI + MassAPI unless "static"; ArticulationRootAPI if a LOOSE one
            box_0, cyl_1 ...   colliders (Cube scaled to the box / Cylinder), physics material and look bound
          <link>               Xform at the link's anchor; RigidBodyAPI + MassAPI; its colliders
          joints/<link>        Prismatic/RevoluteJoint parent -> link, limits, DriveAPI (the spring), target 0
          joints/_world_anchor FixedJoint world -> base (fixed-base articulations only)
          Looks/...            materials

    Joint names in Isaac Lab are the link names; body names are "base" and the link names."""
    articulated = bool(spec.joints)
    fixed = spec.mass is None
    base_path = f"{root}/base"
    static, dynamic = spec.surface_friction()
    mode = os.environ.get("MECHKIT_FIXED_ROOT", "xform").strip().lower() or "xform"
    if mode not in FIXED_ROOT_MODES:
        _loud(f"WARNING: MECHKIT_FIXED_ROOT={mode!r} is not one of {FIXED_ROOT_MODES}; using 'xform'")
        mode = "xform"
    plan: dict = {
        "root": root,
        "kind": spec.kind,
        "fixed_base": fixed,
        "articulation_root": (root if fixed and mode == "xform" else base_path) if articulated else None,
        "bodies": [],
        "shapes": [],
        "joints": [],
        "world_anchor": {"path": f"{root}/joints/_world_anchor", "body1": base_path} if articulated and fixed else None,
        "friction": (static, dynamic),
        "contact_offset": float(spec.contact_offset),
        "rest_offset": REST_OFFSET,
        "solver_position_iterations": int(spec.solver_iterations),
        "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
        "max_depenetration_velocity": MAX_DEPENETRATION_VELOCITY,
        "self_collisions": False,
    }
    for body in spec.body_names:
        path = base_path if body == "base" else f"{root}/{body}"
        rigid = spec.kind != "static"
        off = spec.body_offset(body)
        plan["bodies"].append({"name": body, "path": path, "translate": tuple(float(v) for v in off), "rigid": rigid,
                               "mass": spec.body_mass(body) if rigid else None})
        for i, s in enumerate(spec.body_shapes(body)):
            plan["shapes"].append({
                "path": f"{path}/{'box' if s.kind == 'box' else 'cyl'}_{i}",
                "body": body,
                "kind": s.kind,
                "translate": tuple(float(v) for v in np.asarray(s.center, dtype=float) - off),
                "size": tuple(float(v) for v in s.size),
                "radius": float(s.radius),
                "height": float(s.height),
                "axis": s.axis,
                "color": spec.body_color(body, s),
            })
    for j in spec.joints:
        frame = _joint_frame_wxyz(j.axis)
        deg = j.kind == "revolute"
        scale = math.pi / 180.0 if deg else 1.0  # USD angular drive gains are per degree
        plan["joints"].append({
            "name": j.name,
            "path": f"{root}/joints/{j.name}",
            "type": j.kind,
            "body0": base_path if j.parent == "base" else f"{root}/{j.parent}",
            "body1": f"{root}/{j.name}",
            "axis": j.axis[-1],
            "local_pos0": tuple(float(v) for v in np.asarray(j.anchor, dtype=float) - spec.body_offset(j.parent)),
            "local_rot0_wxyz": frame,
            "local_pos1": (0.0, 0.0, 0.0),
            "local_rot1_wxyz": frame,
            "lower": math.degrees(j.lower) if deg else float(j.lower),
            "upper": math.degrees(j.upper) if deg else float(j.upper),
            "drive": {"api": "angular" if deg else "linear", "stiffness": float(j.spring) * scale,
                      "damping": float(j.damping) * scale, "target": 0.0},
            "actuator": {"stiffness": float(j.spring), "damping": float(j.damping), "friction": float(j.friction)},
        })
    return plan


# == Isaac Sim side (lazy imports only) ===============================================================================


class _PhysxProps:
    """Writes the PhysX schema attributes, through ``pxr.PhysxSchema`` when it imports (Isaac Sim), else as raw
    applied-schema tokens and attributes."""

    def __init__(self) -> None:
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
            _loud("pxr.PhysxSchema did not import: writing PhysX attributes as raw USD")
        self.schema = PhysxSchema

    def _raw(self, prim, api: str, prefix: str, values: dict) -> None:
        from pxr import Sdf

        prim.AddAppliedSchema(api)
        types_ = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
        for name, value in values.items():
            prim.CreateAttribute(f"{prefix}:{name}", types_[type(value)]).Set(value)

    def rigid_body(self, prim, plan: dict) -> None:
        if self.schema is None:
            self._raw(prim, "PhysxRigidBodyAPI", "physxRigidBody", {
                "maxDepenetrationVelocity": float(plan["max_depenetration_velocity"]),
                "solverPositionIterationCount": int(plan["solver_position_iterations"])})
            return
        api = self.schema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateMaxDepenetrationVelocityAttr(float(plan["max_depenetration_velocity"]))
        api.CreateSolverPositionIterationCountAttr(int(plan["solver_position_iterations"]))

    def collider(self, prim, plan: dict) -> None:
        if self.schema is None:
            self._raw(prim, "PhysxCollisionAPI", "physxCollision", {
                "contactOffset": float(plan["contact_offset"]), "restOffset": float(plan["rest_offset"])})
            return
        api = self.schema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(float(plan["contact_offset"]))
        api.CreateRestOffsetAttr(float(plan["rest_offset"]))

    def articulation(self, prim, plan: dict) -> None:
        values = {"enabledSelfCollisions": bool(plan["self_collisions"]),
                  "solverPositionIterationCount": int(plan["solver_position_iterations"]),
                  "solverVelocityIterationCount": int(plan["solver_velocity_iterations"])}
        if self.schema is None:
            self._raw(prim, "PhysxArticulationAPI", "physxArticulation", values)
            return
        api = self.schema.PhysxArticulationAPI.Apply(prim)
        api.CreateEnabledSelfCollisionsAttr(values["enabledSelfCollisions"])
        api.CreateSolverPositionIterationCountAttr(values["solverPositionIterationCount"])
        api.CreateSolverVelocityIterationCountAttr(values["solverVelocityIterationCount"])


def _author(stage, prim_path: str, spec: Assembly, translation=None, orientation=None):
    """Write ``spec`` under ``prim_path`` (layout: :func:`usd_plan`) and return the root prim. ``orientation`` is
    (x, y, z, w), as Isaac Lab 3.0 passes it to spawners."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    plan = usd_plan(spec, prim_path)
    physx = _PhysxProps()
    double = UsdGeom.XformOp.PrecisionDouble

    def place(xform, position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
        x, y, z, w = (float(v) for v in quat_xyzw)
        xform.AddTranslateOp(precision=double).Set(Gf.Vec3d(*(float(v) for v in position)))
        xform.AddOrientOp(precision=double).Set(Gf.Quatd(w, x, y, z))

    root = UsdGeom.Xform.Define(stage, prim_path)
    place(root, tuple(translation) if translation is not None else (0.0, 0.0, 0.0),
          tuple(orientation) if orientation is not None else (0.0, 0.0, 0.0, 1.0))

    physics_material = f"{prim_path}/Looks/physics"
    static, dynamic = plan["friction"]
    material = sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic)
    material.func(physics_material, material)
    looks: dict[tuple, str] = {}

    def look(color) -> str:
        key = tuple(float(c) for c in color)
        if key not in looks:
            path = f"{prim_path}/Looks/color_{len(looks)}"
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(key), roughness=LOOK_ROUGHNESS)
            cfg.func(path, cfg)
            looks[key] = path
        return looks[key]

    prims = {}
    for body in plan["bodies"]:
        xform = UsdGeom.Xform.Define(stage, body["path"])
        place(xform, body["translate"])
        prim = xform.GetPrim()
        if body["rigid"]:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(body["mass"]))
            physx.rigid_body(prim, plan)
        prims[body["name"]] = prim

    for s in plan["shapes"]:
        if s["kind"] == "box":
            geom = UsdGeom.Cube.Define(stage, s["path"])
            geom.CreateSizeAttr(1.0)  # a unit cube scaled to the box
            geom.AddTranslateOp().Set(Gf.Vec3d(*s["translate"]))
            geom.AddScaleOp().Set(Gf.Vec3f(*s["size"]))
        else:
            geom = UsdGeom.Cylinder.Define(stage, s["path"])
            geom.CreateAxisAttr(s["axis"])
            geom.CreateRadiusAttr(s["radius"])
            geom.CreateHeightAttr(s["height"])
            geom.AddTranslateOp().Set(Gf.Vec3d(*s["translate"]))
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        physx.collider(prim, plan)
        bind_physics_material(s["path"], physics_material, stage=stage)
        bind_visual_material(s["path"], look(s["color"]), stage=stage)

    if plan["joints"]:
        UsdGeom.Xform.Define(stage, f"{prim_path}/joints")
    for j in plan["joints"]:
        cls = UsdPhysics.PrismaticJoint if j["type"] == "prismatic" else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, j["path"])
        joint.CreateAxisAttr(j["axis"])
        joint.CreateBody0Rel().SetTargets([Sdf.Path(j["body0"])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(j["body1"])])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*j["local_pos0"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(*j["local_rot0_wxyz"]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*j["local_pos1"]))
        joint.CreateLocalRot1Attr(Gf.Quatf(*j["local_rot1_wxyz"]))
        joint.CreateLowerLimitAttr(float(j["lower"]))
        joint.CreateUpperLimitAttr(float(j["upper"]))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), j["drive"]["api"])
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(j["drive"]["stiffness"])
        drive.CreateDampingAttr(j["drive"]["damping"])
        drive.CreateTargetPositionAttr(j["drive"]["target"])

    anchor = plan["world_anchor"]
    if anchor is not None:
        # A fixed joint from the world (no body0) to the base, at the base's world pose as spawned.
        world = UsdGeom.Xformable(prims["base"]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
        fixed = UsdPhysics.FixedJoint.Define(stage, anchor["path"])
        fixed.CreateBody1Rel().SetTargets([Sdf.Path(anchor["body1"])])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        fixed.CreateLocalRot0Attr(Gf.Quatf(float(q.GetReal()), *(float(v) for v in q.GetImaginary())))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    if plan["articulation_root"] is not None:
        # Fixed base: the root API on the Xform above the base (the base is held by the world joint), so PhysX
        # makes a fixed-base articulation. Loose: on the base body itself, so the base is the root link.
        art = stage.GetPrimAtPath(plan["articulation_root"])
        UsdPhysics.ArticulationRootAPI.Apply(art)
        physx.articulation(art, plan)
    return root.GetPrim()


def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """The Isaac Lab spawner (wrapped with ``clone`` in :func:`_arena_object`): builds the registered spec."""
    from isaaclab.sim.utils.stage import get_current_stage

    hub = _hub()
    spec = hub.specs.get(cfg.mechkit_name)
    if spec is None:
        raise RuntimeError(f"mechkit: no Assembly named {cfg.mechkit_name!r} was registered by the scene hook")
    prim = hub.impl.author(get_current_stage(), prim_path, spec, translation, orientation)
    hub.spawned.add(spec.name)
    return prim


def _arena_object(spec: Assembly):
    """The Arena object for an Assembly: a static collider set (BASE), a rigid object, or an articulation whose
    joints carry implicit actuators (the springs, damping and friction)."""
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class MechkitSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn)
        mechkit_name: str = ""

    joints = tuple(spec.joints)

    class MechkitObject(Object):
        def _generate_articulation_cfg(self):
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import ArticulationCfg

            actuators = {
                j.name: ImplicitActuatorCfg(joint_names_expr=[j.name], stiffness=float(j.spring),
                                            damping=float(j.damping), friction=float(j.friction))
                for j in joints
            }
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            cfg.init_state.joint_pos = {j.name: 0.0 for j in joints}
            return self._add_initial_pose_to_cfg(cfg)

    kind = {"articulation": ObjectType.ARTICULATION, "rigid": ObjectType.RIGID, "static": ObjectType.BASE}[spec.kind]
    origin, quat = spec.spawn_pose()
    return MechkitObject(name=spec.name, object_type=kind, spawner_cfg=MechkitSpawnerCfg(mechkit_name=spec.name),
                         initial_pose=Pose(position_xyz=tuple(float(v) for v in origin),
                                           rotation_xyzw=tuple(float(v) for v in quat)))


def read_state(environment, spec: Assembly) -> AssemblyState:
    """An Assembly's state from the running simulator (env 0). A static compound never moves: its declared pose."""
    if spec.kind == "static":
        return spec.rest_state()
    base = environment.unwrapped
    try:
        asset = base.scene[spec.name]
    except KeyError as exc:
        raise RuntimeError(f"mechkit: the scene has no object {spec.name!r}") from exc
    origin = _np(base.scene.env_origins)[0]
    data = asset.data
    if not spec.joints:
        _, pose = _first_attr(data, "root_link_pose_w", "root_pose_w")
        if pose is None:
            raise RuntimeError(f"mechkit: {spec.name}: no root_link_pose_w / root_pose_w on {type(data).__name__}")
        pose = _np(pose)[0].copy()
        _, vel = _first_attr(data, "root_link_vel_w", "root_vel_w", "root_com_vel_w")
        vel = _np(vel)[0] if vel is not None else np.zeros(6)
        pose[:3] -= origin
        return state_from_poses(spec, pose, vel, source="sim")
    bodies = list(asset.body_names)
    _, poses = _first_attr(data, "body_link_pose_w", "body_pose_w")
    if poses is None:
        raise RuntimeError(f"mechkit: {spec.name}: no body_link_pose_w / body_pose_w on {type(data).__name__}")
    poses = _np(poses)[0].copy()
    poses[:, :3] -= origin
    missing = [b for b in spec.body_names if b not in bodies]
    if missing:
        raise RuntimeError(f"mechkit: {spec.name}: the articulation has bodies {bodies}, not {missing}; was it built "
                           f"by mechkit?")
    _, vels = _first_attr(data, "body_link_vel_w", "body_vel_w", "body_com_vel_w")
    base_vel = _np(vels)[0][bodies.index("base")] if vels is not None and not spec.fixed_base else np.zeros(6)
    names = list(asset.joint_names)
    q = _np(data.joint_pos)[0]
    _, qd_raw = _first_attr(data, "joint_vel")
    qd = _np(qd_raw)[0] if qd_raw is not None else np.zeros_like(q)
    joints, joint_vel = {}, {}
    for j in spec.joints:
        if j.name not in names:
            raise RuntimeError(f"mechkit: {spec.name}: no joint {j.name!r} in the simulator (joints: {names})")
        k = names.index(j.name)
        joints[j.name], joint_vel[j.name] = float(q[k]), float(qd[k])
    link_poses = {j.name: poses[bodies.index(j.name)] for j in spec.joints}
    return state_from_poses(spec, poses[bodies.index("base")], base_vel, link_poses, joints, joint_vel, "sim")


def _describe_sim(environment, spec: Assembly) -> list[str]:
    """What Isaac Lab built for ``spec``, as printable lines (never raises): checked once per run."""
    plan = usd_plan(spec, "<root>")
    lines = [f"{spec.name}: {spec.kind}{', fixed base' if spec.fixed_base else ', loose'}; declared mass {spec.mass}; "
             f"base centre of mass (base frame) {np.round(spec.center_of_mass(), 4).tolist()}; articulation root on "
             f"{plan['articulation_root']} (mechkit {VERSION})"]
    if spec.kind == "static":
        return lines
    try:
        asset = environment.unwrapped.scene[spec.name]
        data = asset.data
        lines.append(f"  isaac lab: {type(asset).__name__}, fixed base {getattr(asset, 'is_fixed_base', 'n/a')}, "
                     f"bodies {list(getattr(asset, 'body_names', []))}")
        for label, names in (("mass", ("default_mass",)), ("com (body frame)", ("body_com_pos_b",))):
            got, value = _first_attr(data, *names)
            if value is not None:
                lines.append(f"  {label} [{got}]: {np.round(_np(value)[0], 4).tolist()}")
        if spec.joints:
            names = list(asset.joint_names)
            lines.append(f"  joints {names}")
            columns = {}
            for label, candidates in (("limits", ("joint_pos_limits",)), ("stiffness", ("joint_stiffness",)),
                                      ("damping", ("joint_damping",)),
                                      ("friction", ("joint_friction_coeff", "joint_friction", "joint_static_friction")),
                                      ("default", ("default_joint_pos",))):
                got, value = _first_attr(data, *candidates)
                columns[label] = (got, None if value is None else _np(value)[0])
            for j in spec.joints:
                k = names.index(j.name) if j.name in names else None
                sim = []
                for label, (got, arr) in columns.items():
                    if arr is not None and k is not None:
                        sim.append(f"{label}={np.round(arr[k], 5).tolist()}")
                lines.append(f"  {j.name}: {j.kind} {j.axis} limits [{j.lower:.4f}, {j.upper:.4f}] {j.unit}, spring "
                             f"{j.spring}, damping {j.damping}, friction {j.friction} | sim: {', '.join(sim) or 'n/a'}")
        lines.append(f"  state now: {_summary(read_state(environment, spec))}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never stop the run
        lines.append(f"  (could not read it: {exc!r})")
    return lines


def _summary(s: AssemblyState) -> str:
    parts = [f"origin {np.round(s.origin, 4).tolist()} tilt {math.degrees(s.tilt()):.1f} deg"]
    if np.linalg.norm(s.lin_vel) > 0:
        parts.append(f"v {np.linalg.norm(s.lin_vel):.3f} m/s")
    for name, q in s.joints.items():
        unit = s.spec.link(name).unit if s.spec is not None else ""
        parts.append(f"{name} {q:+.4f} {unit} ({s.joint_vel.get(name, 0.0):+.3f}/s)")
    return ", ".join(parts)


# == hooking yam_arena ================================================================================================

_HUB_NAME = "_mechkit_hub"  # one process-wide namespace, shared by every copy/import of this module
_MARK = "_mechkit_wrapped"
_SCENE, _EMBODIMENT = "yam_arena.scene", "yam_arena.inspect_embodiment"


def _hub():
    hub = sys.modules.get(_HUB_NAME)
    if hub is None:
        hub = types.ModuleType(_HUB_NAME, "mechkit's process-wide state (see mechkit.py)")
        hub.__dict__.update(
            specs={},  # name -> Assembly, as the scene hook was asked to build them
            spawned=set(),  # names whose USD this module wrote
            last={},  # name -> (step, AssemblyState) from the latest object_states call
            hooked={},  # module name -> status text
            described=set(),
            warned=set(),
            env=None, episode=0, step=0, in_step=False, in_reset=False,
            trace_path=None, print_every=0, impl=None,
        )
        sys.modules[_HUB_NAME] = hub
    return hub


def _loud(message: str, *, once: bool = True) -> None:
    hub = _hub()
    if once and message in hub.warned:
        return
    hub.warned.add(message)
    print(f"[mechkit] {message}", file=sys.stderr, flush=True)


def _is_ours(fn) -> bool:
    seen = 0
    while fn is not None and seen < 64:  # functools.wraps copies the mark up the chain; walk it anyway
        if getattr(fn, _MARK, False):
            return True
        fn, seen = getattr(fn, "__wrapped__", None), seen + 1
    return False


def _wrap_task_object(original):
    @functools.wraps(original)
    def task_object(spec):
        return _hub().impl.task_object(original, spec)

    setattr(task_object, _MARK, True)
    return task_object


def _wrap_object_states(original):
    @functools.wraps(original)
    def object_states(environment, names):
        return _hub().impl.object_states(original, environment, names)

    setattr(object_states, _MARK, True)
    return object_states


def _wrap_task_step(original):
    @functools.wraps(original)
    def _task_step(self, *args, **kwargs):
        hub = _hub()
        if hub.in_step:  # a second copy of this wrapper further down the chain
            return original(self, *args, **kwargs)
        hub.in_step = True
        try:
            hub.impl.before_step(self)
            result = original(self, *args, **kwargs)
            hub.impl.after_step(self, result)
            return result
        finally:
            hub.in_step = False

    setattr(_task_step, _MARK, True)
    return _task_step


def _wrap_reset(original):
    @functools.wraps(original)
    def reset(self, *args, **kwargs):
        hub = _hub()
        if hub.in_reset:
            return original(self, *args, **kwargs)
        hub.in_reset = True
        try:
            observation = original(self, *args, **kwargs)
            hub.impl.after_reset(self)
            return observation
        finally:
            hub.in_reset = False

    setattr(reset, _MARK, True)
    return reset


def _patch_scene(scene) -> None:
    hub = _hub()
    missing = [a for a in ("task_object", "object_states") if not callable(getattr(scene, a, None))]
    if missing:
        hub.hooked[_SCENE] = f"FAILED: no {missing}"
        message = (f"ERROR: {scene.__name__} has no {missing}; this mechkit was written for yam_arena 0.3.1 and "
                   f"cannot build its assemblies (they would be spawned as plain boxes)")
        _loud(message)
        raise RuntimeError(f"mechkit: {message}")
    if not _is_ours(scene.task_object):
        scene.task_object = _wrap_task_object(scene.task_object)
    if not _is_ours(scene.object_states):
        scene.object_states = _wrap_object_states(scene.object_states)
    hub.hooked[_SCENE] = "hooked"


def _patch_embodiment(module) -> None:
    hub = _hub()
    cls = getattr(module, "YamArenaEmbodiment", None)
    if cls is None or not callable(getattr(cls, "_task_step", None)) or not callable(getattr(cls, "reset", None)):
        hub.hooked[_EMBODIMENT] = "missing YamArenaEmbodiment._task_step/reset"
        _loud(f"WARNING: {module.__name__} has no YamArenaEmbodiment with _task_step() and reset(): the episode and "
              f"step counters, the trace and the spawn check are off (assemblies are still built and scored)")
        return
    if not _is_ours(cls._task_step):
        cls._task_step = _wrap_task_step(cls._task_step)
    if not _is_ours(cls.reset):
        cls.reset = _wrap_reset(cls.reset)
    hub.hooked[_EMBODIMENT] = "hooked"


_PATCHERS = {_SCENE: _patch_scene, _EMBODIMENT: _patch_embodiment}


class _PatchingLoader(importlib.abc.Loader):
    """The real loader of a hooked module, then the patch on the finished module."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        _hub().impl.patch_module(module.__name__, module)

    def __getattr__(self, name):  # get_code, get_source, is_package, ... stay the real loader's
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """First on ``sys.meta_path``: finds the hooked modules through the other finders and wraps their loader."""

    _mechkit_finder = True

    def __init__(self, names) -> None:
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        for finder in list(sys.meta_path):
            if getattr(finder, "_mechkit_finder", False) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                _loud(f"ERROR: cannot hook {fullname}: its loader {spec.loader!r} has no exec_module()")
                return spec
            spec.loader = _PatchingLoader(spec.loader)
            return spec
        return None


def _patch_module(name: str, module) -> None:
    patch = _PATCHERS.get(name)
    if patch is not None:
        patch(module)


def _impl_task_object(original, spec):
    if not getattr(spec, "_mechkit_assembly", False):
        return original(spec)
    _hub().specs[spec.name] = spec
    try:
        return _arena_object(spec)
    except Exception as exc:
        _loud(f"ERROR: building {spec.name!r} failed: {exc!r}")
        raise


def _impl_object_states(original, environment, names):
    hub = _hub()
    names = list(names)
    ours = [n for n in names if n in hub.specs]
    plain = [n for n in names if n not in hub.specs]
    out = dict(original(environment, plain)) if plain else {}
    for n in ours:
        try:
            state = read_state(environment, hub.specs[n])
        except Exception as exc:
            _loud(f"ERROR: reading {n!r} failed: {exc!r}")
            raise
        hub.last[n] = (hub.step, state)
        out[n] = state
    hub.env = environment
    return {n: out[n] for n in names}


def _impl_before_step(embodiment) -> None:
    hub = _hub()
    hub.step += 1
    hub.env = getattr(embodiment, "_env", hub.env)


def _impl_after_step(embodiment, result) -> None:
    hub = _hub()
    try:
        path = os.environ.get("MECHKIT_TRACE") or hub.trace_path
        every = int(os.environ.get("MECHKIT_PRINT_EVERY") or hub.print_every or 0)
        if not path and not (every > 0 and hub.step % every == 0):
            return
        states = {}
        for name in sorted(hub.spawned):
            step, state = hub.last.get(name, (None, None))
            if step != hub.step:
                state = read_state(hub.env, hub.specs[name])
            states[name] = state
        t = float(getattr(embodiment, "_t", float("nan")))
        if every > 0 and hub.step % every == 0:
            text = " | ".join(f"{n}: {_summary(s)}" for n, s in states.items())
            print(f"[mechkit] ep {hub.episode} step {hub.step} t {t:.2f}s | {text}", file=sys.stderr, flush=True)
        if path:
            info = getattr(result, "info", None) or {}
            row = {"episode": hub.episode, "step": hub.step, "t": round(t, 4),
                   "info": {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str))},
                   "assemblies": {n: _trace_row(s) for n, s in states.items()}}
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 - tracing must not stop a run
        _loud(f"WARNING: the per-step trace failed and is off for this message: {exc!r}")


def _trace_row(s: AssemblyState) -> dict:
    r = lambda v: [round(float(x), 5) for x in np.asarray(v).reshape(-1)]  # noqa: E731
    return {"joints": {k: round(v, 6) for k, v in s.joints.items()},
            "joint_vel": {k: round(v, 5) for k, v in s.joint_vel.items()},
            "origin": r(s.origin), "quat_xyzw": r(s.quat_xyzw), "pos": r(s.pos), "low": r(s.low), "high": r(s.high),
            "lin_vel": r(s.lin_vel), "links": {k: r(v) for k, v in s.links.items()}}


def _impl_after_reset(embodiment) -> None:
    hub = _hub()
    hub.episode += 1
    hub.step = 0
    hub.last.clear()
    hub.env = getattr(embodiment, "_env", hub.env)
    task = getattr(embodiment, "task", None)
    wanted = [o.name for o in getattr(task, "objects", ()) if getattr(o, "_mechkit_assembly", False)]
    missing = [n for n in wanted if n not in hub.spawned]
    if missing:
        message = (f"ERROR: {missing} were not built by mechkit (the yam_arena.scene hook did not run: "
                   f"{hub.hooked.get(_SCENE, 'never imported')}); they would be plain boxes. Stopping.")
        _loud(message)
        raise RuntimeError(f"mechkit: {message}")
    for name in sorted(hub.spawned):
        if name in hub.described or name not in hub.specs:
            continue
        hub.described.add(name)
        for line in _describe_sim(hub.env, hub.specs[name]):
            print(f"[mechkit] {line}", file=sys.stderr, flush=True)


# == public runtime API ===============================================================================================


def install() -> dict:
    """Hook yam_arena: now for the modules already imported, and from a ``sys.meta_path`` finder for the ones imported
    later. Idempotent; importing this module calls it. Returns :func:`status`."""
    hub = _hub()
    hub.impl = types.SimpleNamespace(
        task_object=_impl_task_object, object_states=_impl_object_states, before_step=_impl_before_step,
        after_step=_impl_after_step, after_reset=_impl_after_reset, patch_module=_patch_module, author=_author,
        version=VERSION, module_file=__file__,
    )
    for name in _PATCHERS:
        module = sys.modules.get(name)
        if module is not None:
            _patch_module(name, module)
        else:
            hub.hooked.setdefault(name, "pending (patched when it is imported)")
    if not any(getattr(f, "_mechkit_finder", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(_PATCHERS))
    return status()


def status() -> dict:
    """Which hooks are in place: {module: "hooked" | "pending ..." | "FAILED ..."}, plus "finder" and "spawned"."""
    hub = _hub()
    return {**hub.hooked, "finder": any(getattr(f, "_mechkit_finder", False) for f in sys.meta_path),
            "spawned": sorted(hub.spawned)}


def configure(trace: str | None = None, print_every: int | None = None) -> None:
    """Defaults for the per-step trace file and the stderr summary interval (steps). ``MECHKIT_TRACE`` and
    ``MECHKIT_PRINT_EVERY`` override them."""
    hub = _hub()
    if trace is not None:
        hub.trace_path = str(trace) or None
    if print_every is not None:
        hub.print_every = max(0, int(print_every))


def env():
    """The Isaac Lab env (the gym env; ``.unwrapped`` is Isaac Lab's) of the current run, or None before it exists."""
    return _hub().env


def episode() -> int:
    """Increments after every reset (sim) or :func:`offline_reset`; 0 before the first. Use it to clear latches."""
    return _hub().episode


def step() -> int:
    """Control steps since the last reset (the one being scored now, from 1)."""
    return _hub().step


def offline_reset() -> None:
    """Called by the offline checker before each playback: a new episode for latches keyed on :func:`episode`."""
    hub = _hub()
    hub.episode += 1
    hub.step = 0
    hub.last.clear()


install()
