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

Version 2 adds (each still to be verified on the GPU by tasks/_mechkit2-smoke; see MECHKIT.md):

- shapes turned by any rotation: ``box(..., quat=...)`` / ``box(..., ypr=(yaw, pitch, roll))``, the same for
  ``cylinder`` (chamfers, ramps, a disc turned 15 deg), and :func:`plate_with_holes` (a plate with N holes);
- tight-fit physics per assembly: ``assembly(..., tight_fit=True)`` (contact offset 0.5 mm, rest offset 0, 32 solver
  position iterations, angular damping 1, max depenetration 1 m/s) or each of those fields on its own;
- per-step rules, run at the start of every control step before the action and the physics (sim only):
  :func:`on_step` / :func:`on_reset` callbacks, :func:`weld` (a part's pose rewritten relative to a base once a
  condition fires; released at reset), :func:`bond` (a FixedJoint between two bodies that breaks above a force or
  torque, can be released or armed at run time, and is re-armed at every reset) and :func:`set_targets` (joint drive
  targets for rule-based detents; SI units: m or rad).

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
    "Bond",
    "Joint",
    "Shape",
    "TIGHT_FIT",
    "Weld",
    "assembly",
    "bond",
    "bond_frames",
    "box",
    "configure",
    "cylinder",
    "detent_target",
    "env",
    "episode",
    "hinge",
    "install",
    "keep_yaw",
    "offline_reset",
    "on_reset",
    "on_step",
    "open_box",
    "plate_with_hole",
    "plate_with_holes",
    "quat_from_ypr",
    "read_state",
    "rules",
    "set_targets",
    "slider",
    "state_from_poses",
    "status",
    "step",
    "usd_plan",
    "weld",
]

VERSION = "2"
AXES = ("X", "Y", "Z", "-X", "-Y", "-Z")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")  # link names become USD prim names and Isaac Lab regexes
RESERVED_LINK_NAMES = frozenset({"base", "joints", "bonds", "Looks"})  # prims an assembly already owns under its root

# Physics settings for every assembly (overridable per assembly where a field exists).
FIXED_BASE_MASS = 1.0  # kg: a fixed base is still an articulation link; its mass does not matter
DEFAULT_CONTACT_OFFSET = 0.005  # m, as yam_arena gives its loose primitives (scene._contact_props)
REST_OFFSET = 0.0
DEFAULT_SOLVER_ITERATIONS = 16  # position iterations, as yam_arena's arms
SOLVER_VELOCITY_ITERATIONS = 4
MAX_DEPENETRATION_VELOCITY = 1.0  # m/s
# Tight fits (about 1 mm of play), as tasks/bayonet-mate partkit._spawn_part sets them for its small stiff parts:
# a contact offset below the play so the faces do not all touch at once, 32 position iterations, and light angular
# damping ("small stiff parts otherwise chatter in tight fits"). ``assembly(..., tight_fit=True)`` applies these to
# every field not given explicitly.
TIGHT_FIT = {"contact_offset": 0.0005, "rest_offset": 0.0, "solver_iterations": 32, "angular_damping": 1.0,
             "max_depenetration_velocity": 1.0}
UNBREAKABLE = 3.0e38  # N or N m written for an infinite break force/torque (PhysX "unbreakable", as forceps' _BIG)
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


def quat_from_ypr(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of yaw about z, then pitch about the new y, then roll about the newest x
    (radians; R = Rz(yaw) Ry(pitch) Rx(roll)). A box pitched +0.3 tips its +x end down."""
    r = (axis_angle_matrix((0, 0, 1), float(yaw)) @ axis_angle_matrix((0, 1, 0), float(pitch))
         @ axis_angle_matrix((1, 0, 0), float(roll)))
    return matrix_to_quat(r)


def _orientation(quat, ypr, what: str) -> tuple[float, float, float, float] | None:
    """A shape's turn as a unit quaternion (x, y, z, w) with w >= 0, or None for none. Give at most one of
    ``quat`` (x, y, z, w) and ``ypr`` (yaw, pitch, roll in radians)."""
    if quat is not None and ypr is not None:
        raise ValueError(f"{what}: give quat or ypr, not both")
    if ypr is not None:
        q = quat_from_ypr(*_vec(ypr, 3, f"{what} ypr"))
    elif quat is not None:
        q = np.asarray(_vec(quat, 4, f"{what} quat (x, y, z, w)"), dtype=float)
        n = float(np.linalg.norm(q))
        if not n > 1e-9:
            raise ValueError(f"{what}: quat {quat!r} is not a rotation")
        q = q / n
        if q[3] < 0:
            q = -q
    else:
        return None
    if np.allclose(q, (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-12):
        return None
    return tuple(float(v) for v in q)


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
    joint at 0. ``color`` (sRGB) defaults to its link's, then the assembly's. ``quat`` (x, y, z, w; None: not turned)
    turns the shape about its own centre, in the base frame (a box's edges, a cylinder's axis). Build with
    :func:`box` / :func:`cylinder`."""

    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    height: float = 0.0
    axis: str = "Z"
    color: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None

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
        if self.quat is not None:
            q = _vec(self.quat, 4, f"{what} quat")
            if not abs(math.sqrt(sum(v * v for v in q)) - 1.0) < 1e-6:
                raise ValueError(f"{what}: quat must be a unit quaternion (x, y, z, w), got {self.quat!r}")

    def rotation(self) -> np.ndarray:
        """The shape's own turn (3x3, base frame): identity unless it was given ``quat`` / ``ypr``."""
        return np.eye(3) if self.quat is None else quat_to_matrix(self.quat)

    def half_extents(self) -> np.ndarray:
        """Half its extent along the base frame's x, y, z (exact; for a turned shape, of its bounding box)."""
        if self.quat is not None:
            return self.aabb_half(np.eye(3))
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
        """Half extents of its world-aligned bounding box once its frame is turned by ``rotation`` (exact; the
        shape's own ``quat`` is included)."""
        r = np.asarray(rotation, dtype=float)
        if self.quat is not None:
            r = r @ quat_to_matrix(self.quat)
        if self.kind == "box":
            return np.abs(r) @ (np.asarray(self.size, dtype=float) / 2.0)
        a = r[:, _AXIS_INDEX[self.axis]]
        return self.radius * np.sqrt(np.clip(1.0 - a * a, 0.0, 1.0)) + self.height / 2.0 * np.abs(a)


def box(center, size, color=None, *, quat=None, ypr=None) -> Shape:
    """A box collider: ``center`` and ``size`` (x, y, z edge lengths) in metres, base frame at rest. ``quat``
    (x, y, z, w) or ``ypr`` (yaw, pitch, roll, rad: :func:`quat_from_ypr`) turns it about its centre: a chamfer, a
    ramp, a key turned 15 deg (tasks/din-rail-mount partkit ``_box`` writes the same with ``AddOrientOp``)."""
    s = Shape("box", _vec3(center, "box center"), _vec3(size, "box size"), color=_rgb(color, "box color"),
              quat=_orientation(quat, ypr, "box"))
    s.check("box")
    return s


def cylinder(center, radius: float, height: float, axis: str = "Z", color=None, *, quat=None, ypr=None) -> Shape:
    """A cylinder collider along ``axis`` ("X", "Y" or "Z": a lying cylinder is fine here), optionally turned
    about its centre by ``quat`` / ``ypr`` (a disc tilted on a slope, a pin at an angle)."""
    s = Shape("cylinder", _vec3(center, "cylinder center"), radius=float(radius), height=float(height),
              axis=str(axis).upper(), color=_rgb(color, "cylinder color"), quat=_orientation(quat, ypr, "cylinder"))
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


def plate_with_holes(center, outer_xy, holes, thickness: float, color=None) -> tuple[Shape, ...]:
    """A flat plate (``outer_xy`` by ``thickness``, centred on ``center``) with any number of rectangular
    through-holes, each ``(x, y, size_x, size_y)``: its centre's offset from the plate's centre and its full size
    (the same meaning as ``plate_with_hole``'s ``hole_offset`` and ``hole_xy``). Holes may touch the plate's edge
    (a notch) or each other, but not overlap. Built as tasks/replace-soft-gripper-finger ``_boxes_minus_holes``
    does: the plate is cut into a grid along every hole edge, the cells inside a hole are left out, and cells
    next to each other along y are merged, so each box is one solid strip."""
    cx, cy, cz = _vec3(center, "plate center")
    ox, oy = _vec(outer_xy, 2, "plate outer_xy")
    t = _number(thickness, "plate thickness", positive=True)
    if not (ox > 0 and oy > 0):
        raise ValueError(f"plate_with_holes: outer_xy must be positive, got {outer_xy!r}")
    try:
        holes = [tuple(_vec(h, 4, f"plate_with_holes hole {i}")) for i, h in enumerate(holes)]
    except TypeError:
        raise ValueError(f"plate_with_holes: holes must be a list of (x, y, size_x, size_y), got {holes!r}") from None
    eps = 1e-9
    rects = []
    for i, (dx, dy, sx, sy) in enumerate(holes):
        if not (sx > 0 and sy > 0):
            raise ValueError(f"plate_with_holes: hole {i} needs a positive size, got {(sx, sy)}")
        r = (dx - sx / 2, dx + sx / 2, dy - sy / 2, dy + sy / 2)  # in the plate's own x, y (centre 0, 0)
        if r[0] < -ox / 2 - eps or r[1] > ox / 2 + eps or r[2] < -oy / 2 - eps or r[3] > oy / 2 + eps:
            raise ValueError(f"plate_with_holes: hole {i} {holes[i]} does not lie inside the {ox} x {oy} plate")
        for j, q in enumerate(rects):
            if min(r[1], q[1]) - max(r[0], q[0]) > eps and min(r[3], q[3]) - max(r[2], q[2]) > eps:
                raise ValueError(f"plate_with_holes: holes {j} and {i} overlap")
        rects.append(r)
    xs = sorted({-ox / 2, ox / 2, *(min(max(v, -ox / 2), ox / 2) for r in rects for v in r[:2])})
    ys = sorted({-oy / 2, oy / 2, *(min(max(v, -oy / 2), oy / 2) for r in rects for v in r[2:])})
    xs = [v for k, v in enumerate(xs) if k == 0 or v - xs[k - 1] > eps]
    ys = [v for k, v in enumerate(ys) if k == 0 or v - ys[k - 1] > eps]
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        mx = (a + b) / 2
        run = None  # the current solid strip along y in this column: (start, end)
        for c, d in zip(ys[:-1], ys[1:]):
            my = (c + d) / 2
            solid = not any(r[0] < mx < r[1] and r[2] < my < r[3] for r in rects)
            if solid:
                run = (run[0], d) if run is not None else (c, d)
            elif run is not None:
                out.append((a, b, *run))
                run = None
        if run is not None:
            out.append((a, b, *run))
    if not out:
        raise ValueError("plate_with_holes: the holes leave nothing of the plate")
    return tuple(box((cx + (a + b) / 2, cy + (c + d) / 2, cz), (b - a, d - c, t), color) for a, b, c, d in out)


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
    of every collider, a number or (static, dynamic); None gives yam_arena's (1.2, 1.0). The physics fields
    (contact and rest offset, solver iterations, damping, max depenetration velocity) apply to every body and
    collider of the assembly; ``angular_damping`` / ``linear_damping`` None leave PhysX's defaults. Build with
    :func:`assembly`."""

    shapes: tuple[Shape, ...] = ()
    joints: tuple[Joint, ...] = ()
    friction: Any = None
    contact_offset: float = DEFAULT_CONTACT_OFFSET
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS
    rest_offset: float = REST_OFFSET
    angular_damping: float | None = None
    linear_damping: float | None = None
    max_depenetration_velocity: float = MAX_DEPENETRATION_VELOCITY

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
        rest = _number(self.rest_offset, f"{self.name}.rest_offset", nonnegative=True)
        if not rest < self.contact_offset:
            raise ValueError(f"{self.name}.rest_offset {rest!r} must be below contact_offset {self.contact_offset!r}")
        for label in ("angular_damping", "linear_damping"):
            if getattr(self, label) is not None:
                _number(getattr(self, label), f"{self.name}.{label}", nonnegative=True)
        _number(self.max_depenetration_velocity, f"{self.name}.max_depenetration_velocity", positive=True)

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
             friction=None, contact_offset: float | None = None, solver_iterations: int | None = None,
             tight_fit: bool = False, rest_offset: float | None = None, angular_damping: float | None = None,
             linear_damping: float | None = None, max_depenetration_velocity: float | None = None) -> Assembly:
    """Declare an assembly: ``shapes`` (the base body) and ``joints`` (links from :func:`slider` / :func:`hinge`)
    in the base frame, which sits at ``origin`` (world) turned ``yaw`` about the vertical. Without ``mass`` the
    base is fixed in place; with it, loose. Checked now, and again by yam_arena's loader.

    Physics: by default yam_arena's loose-primitive settings (contact offset 5 mm, rest offset 0, 16 position
    iterations, PhysX's default damping). ``tight_fit=True`` uses :data:`TIGHT_FIT` instead (0.5 mm, 0, 32, angular
    damping 1.0, max depenetration 1 m/s) for parts with about 1 mm of play; any field given explicitly wins."""
    base = TIGHT_FIT if tight_fit else {"contact_offset": DEFAULT_CONTACT_OFFSET, "rest_offset": REST_OFFSET,
                                        "solver_iterations": DEFAULT_SOLVER_ITERATIONS, "angular_damping": None,
                                        "max_depenetration_velocity": MAX_DEPENETRATION_VELOCITY}
    given = {"contact_offset": contact_offset, "rest_offset": rest_offset, "solver_iterations": solver_iterations,
             "angular_damping": angular_damping, "max_depenetration_velocity": max_depenetration_velocity}
    phys = {k: (v if v is not None else base.get(k)) for k, v in given.items()}
    spec = Assembly(name=name, pos=origin, color=color, mass=mass, yaw=yaw, shapes=_shapes(shapes, name),
                    joints=tuple(joints), friction=friction, linear_damping=linear_damping, **phys)
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

    def distance(self, points, bodies=None) -> np.ndarray:
        """Distance (m) from each world point (3,) or (N, 3) to the nearest shape of ``bodies`` (default every body)
        as they are now: exact for boxes and cylinders, turned shapes included; 0 inside a shape. For release and
        keep-out rules: ``rigstate.released(arms, state.distance)``."""
        if self.spec is None:
            raise RuntimeError("this state carries no Assembly spec")
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        best = np.full(len(pts), np.inf)
        for body in bodies or self.spec.body_names:
            r, p = self.link_pose(body)
            off = self.spec.body_offset(body)
            for s in self.spec.body_shapes(body):
                rw = r @ s.rotation()
                local = (pts - (p + r @ (np.asarray(s.center, dtype=float) - off))) @ rw
                if s.kind == "box":
                    q = np.abs(local) - np.asarray(s.size, dtype=float) / 2
                    d = np.linalg.norm(np.maximum(q, 0.0), axis=1)
                else:
                    k = _AXIS_INDEX[s.axis]
                    radial = np.linalg.norm(np.delete(local, k, axis=1), axis=1) - s.radius
                    along = np.abs(local[:, k]) - s.height / 2
                    d = np.hypot(np.maximum(radial, 0.0), np.maximum(along, 0.0))
                best = np.minimum(best, d)
        return best


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
        "rest_offset": float(spec.rest_offset),
        "solver_position_iterations": int(spec.solver_iterations),
        "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
        "max_depenetration_velocity": float(spec.max_depenetration_velocity),
        "angular_damping": None if spec.angular_damping is None else float(spec.angular_damping),
        "linear_damping": None if spec.linear_damping is None else float(spec.linear_damping),
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
                # USD orient op (w, x, y, z), written between translate and scale; None: no orient op
                "orient_wxyz": None if s.quat is None else (s.quat[3], s.quat[0], s.quat[1], s.quat[2]),
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
        damping = {k: float(plan[p]) for k, p in (("angularDamping", "angular_damping"),
                                                   ("linearDamping", "linear_damping")) if plan.get(p) is not None}
        if self.schema is None:
            self._raw(prim, "PhysxRigidBodyAPI", "physxRigidBody", {
                "maxDepenetrationVelocity": float(plan["max_depenetration_velocity"]),
                "solverPositionIterationCount": int(plan["solver_position_iterations"]), **damping})
            return
        api = self.schema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateMaxDepenetrationVelocityAttr(float(plan["max_depenetration_velocity"]))
        api.CreateSolverPositionIterationCountAttr(int(plan["solver_position_iterations"]))
        if "angularDamping" in damping:  # tasks/bayonet-mate partkit: CreateAngularDampingAttr(1.0)
            api.CreateAngularDampingAttr(damping["angularDamping"])
        if "linearDamping" in damping:
            api.CreateLinearDampingAttr(damping["linearDamping"])

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
        # Op order translate, orient, scale (tasks/din-rail-mount partkit _box): the unit cube is scaled to the box,
        # turned about its own centre, then placed.
        if s["kind"] == "box":
            geom = UsdGeom.Cube.Define(stage, s["path"])
            geom.CreateSizeAttr(1.0)  # a unit cube scaled to the box
            geom.AddTranslateOp().Set(Gf.Vec3d(*s["translate"]))
            if s["orient_wxyz"] is not None:
                geom.AddOrientOp().Set(Gf.Quatf(*(float(v) for v in s["orient_wxyz"])))
            geom.AddScaleOp().Set(Gf.Vec3f(*s["size"]))
        else:
            geom = UsdGeom.Cylinder.Define(stage, s["path"])
            geom.CreateAxisAttr(s["axis"])
            geom.CreateRadiusAttr(s["radius"])
            geom.CreateHeightAttr(s["height"])
            geom.AddTranslateOp().Set(Gf.Vec3d(*s["translate"]))
            if s["orient_wxyz"] is not None:
                geom.AddOrientOp().Set(Gf.Quatf(*(float(v) for v in s["orient_wxyz"])))
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
    stage = get_current_stage()
    prim = hub.impl.author(stage, prim_path, spec, translation, orientation)
    hub.spawned.add(spec.name)
    hub.paths[spec.name] = str(prim_path)
    if hub.bonds:
        author_bonds = getattr(hub.impl, "author_spawn_bonds", None)
        for name in (author_bonds(stage) if author_bonds is not None else ()):
            _loud(f"bond {name!r} glued at spawn ({hub.bonds[name].path})", once=False)
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


# == per-step rules: welds, bonds, drive targets (pure logic here; the simulator writes are in the next section) ======
#
# Every rule runs at the START of a control step (before the action and the physics), from the step hook, in this
# order: the task's on_step() callbacks, bond break checks, welds (fire, then rewrite), drive targets. That is where
# tasks/bayonet-mate partkit runs its on_step functions (snapfit-close's weld) and forceps its bonds. Offline (the
# checker in tools/offline) no rule runs: the declarations are checked and the logic is unit-tested on its own.


def _body_ref(value, what: str) -> tuple[str, str]:
    """(assembly name, body name) of an Assembly, an (Assembly, link name) pair or an assembly name string."""
    link = "base"
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        value, link = value
    if getattr(value, "_mechkit_assembly", False):
        if link != "base":
            value.link(link)  # KeyError when the assembly has no such link
        return value.name, link
    if isinstance(value, str) and _NAME.match(value):
        return value, link
    raise TypeError(f"{what} must be an Assembly, (Assembly, link name) or an object name, got {value!r}")


def state_frame(state) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of an object's frame from its state: an AssemblyState's base frame origin, or a plain
    ObjectState's ``pos`` (a native primitive's frame is its centre)."""
    origin = getattr(state, "origin", None)
    o = np.asarray(state.pos if origin is None else origin, dtype=float).reshape(3)
    return o, quat_to_matrix(state.quat_xyzw)


def body_frame(state, body: str = "base") -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of one body of an object ("base" or a link name of an AssemblyState)."""
    if body == "base":
        return state_frame(state)
    r, p = state.link_pose(body)
    return np.asarray(p, dtype=float), r


def _finite_state(state) -> bool:
    try:
        o, r = state_frame(state)
        v = np.concatenate([np.asarray(state.lin_vel, float).ravel(), np.asarray(state.ang_vel, float).ravel()])
    except Exception:  # noqa: BLE001 - a missing field is as bad as a NaN
        return False
    return bool(np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and np.all(np.isfinite(v)))


def keep_yaw(z: float | None = None):
    """An adjustment for :func:`weld` ``rel_pose``: keep only the captured turn about the base's z axis (the part
    sits square on its seat) and, if given, set the height above the base frame to ``z`` (snapfit-close: the lid
    held at the floor plus half its thickness)."""

    def adjust(rel_pos, rel_rot):
        yaw = math.atan2(rel_rot[1, 0], rel_rot[0, 0])
        pos = np.array(rel_pos, dtype=float)
        if z is not None:
            pos[2] = float(z)
        return pos, axis_angle_matrix((0.0, 0.0, 1.0), yaw)

    return adjust


@dataclass(eq=False)
class Weld:
    """A part held at a fixed pose relative to a base (or the world) once a condition fires, by rewriting its root
    pose and velocity every control step (tasks/snapfit-close ``_on_step``; tasks/bayonet-mate partkit
    ``write_pose``). Released at every reset. Build with :func:`weld`. State: ``active``, ``rel`` (position, 3x3
    rotation of the part in the base frame), ``fired_step``, ``fired_episode``, ``writes``. Say so in the task's
    fidelity text: "held seated ..., rewritten every control step"."""

    name: str
    part: str
    base: str | None
    rel_pose: Any = None  # None: captured when it fires; (pos, quat_xyzw): fixed; callable(pos, rot) -> (pos, rot)
    when: Any = None  # callable(states) -> bool, checked every step until it fires; None: only fire() fires it
    active: bool = False
    rel: tuple | None = None
    fired_step: int | None = None
    fired_episode: int | None = None
    writes: int = 0

    def _base_frame(self, states):
        if self.base is None:
            return np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3)
        b = states[self.base]
        o, r = state_frame(b)
        return o, r, np.asarray(b.lin_vel, float).reshape(3), np.asarray(b.ang_vel, float).reshape(3)

    def fire(self, states=None, rel_pose=None) -> bool:
        """Weld now. The relative pose is ``rel_pose`` if given here, else the weld's own ``rel_pose`` (fixed, or an
        adjustment of the captured one), else the pose the part has now relative to the base (needs ``states``).
        True if it was not active before."""
        if self.active:
            return False
        rel = self.rel_pose if rel_pose is None else rel_pose
        if rel is None or callable(rel):
            if states is None:
                raise ValueError(f"weld {self.name!r}: capturing the pose needs the states")
            o, r, _, _ = self._base_frame(states)
            po, pr = state_frame(states[self.part])
            pos, rot = r.T @ (po - o), r.T @ pr
            if callable(rel):
                pos, rot = rel(pos, rot)
                rot = np.asarray(rot, dtype=float)
                if rot.shape == (4,):
                    rot = quat_to_matrix(rot)
        else:
            pos = np.asarray(_vec(rel[0], 3, f"weld {self.name!r} rel_pose position"), dtype=float)
            rot = quat_to_matrix(_vec(rel[1], 4, f"weld {self.name!r} rel_pose quat"))
        self.rel = (np.asarray(pos, dtype=float).reshape(3), np.asarray(rot, dtype=float).reshape(3, 3))
        self.active = True
        self.fired_step, self.fired_episode = _hub().step, _hub().episode
        return True

    def release(self) -> None:
        """Stop rewriting the part's pose (it is a free body again)."""
        self.active = False
        self.rel = None

    def target(self, states) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(position, quat_xyzw, lin_vel, ang_vel) to write for the part this step (env frame): the welded pose on
        the base as the base is now, moving with the base (v = v_base + w_base x r)."""
        if self.rel is None:
            raise RuntimeError(f"weld {self.name!r} has not fired")
        o, r, v, w = self._base_frame(states)
        pos = o + r @ self.rel[0]
        return pos, matrix_to_quat(r @ self.rel[1]), v + np.cross(w, pos - o), w.copy()

    def reset(self) -> None:
        self.release()
        self.fired_step = self.fired_episode = None
        self.writes = 0

    def status(self) -> dict:
        return {"active": self.active, "fired_step": self.fired_step, "writes": self.writes}


def weld(part, base=None, rel_pose=None, *, when=None, name: str | None = None) -> Weld:
    """Hold ``part`` (a loose Assembly or a native object's name) at a pose relative to ``base`` (an object, or None
    for the world) from the step ``when(states)`` first returns True (or from ``Weld.fire()``) until the episode
    resets. ``rel_pose``: None captures the pose the part has when it fires; ``(pos, quat_xyzw)`` in the base frame
    snaps it there; a callable ``(pos, rot) -> (pos, rot)`` adjusts the captured one (:func:`keep_yaw`). ``states``
    passed to ``when`` holds every task object's state (AssemblyStates for assemblies). Sim only: offline nothing
    is rewritten."""
    p_name, p_body = _body_ref(part, "weld part")
    if p_body != "base":
        raise ValueError("weld part: only a whole object's root can be welded, not a link")
    if getattr(part, "_mechkit_assembly", False) and part.mass is None:
        raise ValueError(f"weld part {p_name!r} is fixed in place (no mass): nothing to weld")
    b_name = None
    if base is not None:
        b_name, b_body = _body_ref(base, "weld base")
        if b_body != "base":
            raise ValueError("weld base: welds are relative to a whole object's base frame, not a link")
        if b_name == p_name:
            raise ValueError(f"weld: part and base are both {p_name!r}")
    if rel_pose is not None and not callable(rel_pose):
        try:
            _vec(rel_pose[0], 3, "weld rel_pose position")
            q = _vec(rel_pose[1], 4, "weld rel_pose quat")
        except (TypeError, IndexError):
            raise ValueError(f"weld rel_pose must be None, (pos, quat_xyzw) or a callable, got {rel_pose!r}") from None
        if not math.sqrt(sum(v * v for v in q)) > 1e-9:
            raise ValueError("weld rel_pose quat is not a rotation")
    if when is not None and not callable(when):
        raise TypeError(f"weld when must be a callable(states) -> bool, got {when!r}")
    w = Weld(name or f"{p_name}_to_{b_name or 'world'}", p_name, b_name, rel_pose, when)
    if not _NAME.match(w.name):
        raise ValueError(f"weld name {w.name!r} must be a letter then letters, digits or _")
    _hub().welds[w.name] = w  # a task file loaded twice replaces, never stacks
    return w


def bond_frames(child_pose, parent_pose=None, anchor=(0.0, 0.0, 0.0)) -> dict:
    """The FixedJoint frames that glue two bodies where they are now. ``child_pose`` / ``parent_pose``: (position,
    quat_xyzw) in one frame (the world, for USD); ``parent_pose`` None: the world. ``anchor``: where the glue sits,
    in the child's frame. body0 is the child (PhysX measures a joint's torque about body0: forceps mechkit puts the
    child there so ``break_torque`` is about it); frame 0 is the anchor with the child's axes; frame 1 the same
    world frame seen from the parent (snap-board-out-of-panel ``_joint``: rot1 = R_parent^T R_child). Returns
    local_pos0, local_rot0_wxyz, local_pos1, local_rot1_wxyz."""
    cp, cq = np.asarray(child_pose[0], float).reshape(3), np.asarray(child_pose[1], float).reshape(4)
    rc = quat_to_matrix(cq)
    a_world = cp + rc @ np.asarray(anchor, dtype=float).reshape(3)
    if parent_pose is None:
        pos1, rot1 = a_world, rc
    else:
        pp, pq = np.asarray(parent_pose[0], float).reshape(3), np.asarray(parent_pose[1], float).reshape(4)
        rp = quat_to_matrix(pq)
        pos1, rot1 = rp.T @ (a_world - pp), rp.T @ rc
    q1 = matrix_to_quat(rot1)
    return {"local_pos0": tuple(float(v) for v in np.asarray(anchor, dtype=float).reshape(3)),
            "local_rot0_wxyz": (1.0, 0.0, 0.0, 0.0),
            "local_pos1": tuple(float(v) for v in pos1),
            "local_rot1_wxyz": (float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2]))}


def _relative(child_frame, parent_frame) -> tuple[np.ndarray, np.ndarray]:
    (pc, rc), (pp, rp) = child_frame, parent_frame
    return rp.T @ (pc - pp), rp.T @ rc


def _rotation_angle(r) -> float:
    return math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))


@dataclass(eq=False)
class Bond:
    """A FixedJoint between two bodies (glue, a break-away web, a magnet that can be switched off). It breaks when
    its constraint force exceeds ``break_force`` (N) or its torque ``break_torque`` (N m, about the child); a
    broken PhysX joint stays broken, so it is re-armed at every reset. Build with :func:`bond`.

    State: ``enabled`` (the joint is on in USD), ``broken`` (latched this episode: the pair moved apart by more than
    ``slip`` m or ``slip_deg`` while it was on: geometry, the way forceps' ``broken()`` reports it), ``released``
    (switched off by :meth:`release`), ``at_rest_frames`` (its frames are those of the reset pose; False after
    :meth:`arm`), ``path`` (its USD prim), ``generation`` (how many times it was authored anew)."""

    name: str
    child: str
    child_body: str
    parent: str | None
    parent_body: str
    parent_static: bool
    break_force: float
    break_torque: float
    anchor: tuple[float, float, float]
    armed: bool
    slip: float
    slip_deg: float
    enabled: bool = False
    broken: bool = False
    released: bool = False
    at_rest_frames: bool = True
    path: str | None = None
    generation: int = 0
    rel: tuple | None = None  # the child's pose in the parent frame (env frame for the world) when it was glued
    broke_step: int | None = None
    stepped: bool = False  # a control step has run since it was last prepared at a reset
    pending: list = field(default_factory=list)  # runtime requests made outside a step: "arm" / "release"
    child_spec: Any = field(default=None, repr=False)
    parent_spec: Any = field(default=None, repr=False)

    def frames_of(self, states) -> tuple[tuple, tuple]:
        """(child frame, parent frame) as (origin, rotation) in the env frame; the world (or a fixed part, which
        never moves) is the identity."""
        child = body_frame(states[self.child], self.child_body)
        if self.parent is None or self.parent_static:
            return child, (np.zeros(3), np.eye(3))
        return child, body_frame(states[self.parent], self.parent_body)

    def glued_at(self, states) -> None:
        """Remember the pair's relative pose now (after the joint was authored, or after a reset)."""
        self.rel = _relative(*self.frames_of(states))

    def check_break(self, states) -> bool:
        """While on: has the pair moved apart beyond the slip limits since it was glued? Latches ``broken``;
        True on the step it breaks."""
        if not self.enabled or self.broken or self.rel is None:
            return False
        pos, rot = _relative(*self.frames_of(states))
        moved = float(np.linalg.norm(pos - self.rel[0]))
        turned = math.degrees(_rotation_angle(self.rel[1].T @ rot))
        if moved > self.slip or turned > self.slip_deg:
            self.broken, self.enabled, self.broke_step = True, False, _hub().step
            return True
        return False

    def release(self) -> None:
        """Switch it off now (sim: at the start of the next control step); re-armed at the next reset if ``armed``."""
        self.pending.append("release")

    def arm(self) -> None:
        """Glue the pair where they are now (sim: a new joint authored from the live poses at the start of the next
        control step). Until the next reset its frames are not the reset pose's."""
        self.pending.append("arm")

    def reset_plan(self) -> str:
        """What a reset does with it, from this episode's history: "new" (remove the prim and author a fresh joint,
        enabled, with frames from the declared spawn poses: it broke, was released or was armed at run time; a
        broken PhysX joint stays broken; forceps ``_reglue``; also a BREAKABLE bond after any step of an episode,
        because PhysX may have broken it without the pair slipping far enough to see, e.g. a glued part that rests
        on a support), "keep" (an unbreakable one, intact at its spawn frames: the reset puts both bodies back where
        it glued them), "off" (declared ``armed=False``: switched off if it was armed) or "author" (no prim yet).
        Idempotent within one reset: preparing it clears ``stepped``."""
        if not self.armed:
            return "off"
        if self.path is None:
            return "author"
        if self.broken or self.released or not self.at_rest_frames:
            return "new"
        if self.stepped and self.breakable:
            return "new"
        return "keep"

    @property
    def breakable(self) -> bool:
        """A finite break force or break torque."""
        return math.isfinite(self.break_force) or math.isfinite(self.break_torque)

    def status(self) -> dict:
        return {"enabled": self.enabled, "broken": self.broken, "released": self.released,
                "generation": self.generation, "broke_step": self.broke_step}


def bond(name: str, child, parent=None, *, break_force: float = math.inf, break_torque: float = math.inf,
         anchor=None, armed: bool = True, slip_mm: float = 2.0, slip_deg: float = 2.0) -> Bond:
    """Glue ``child`` (a loose Assembly, or ``(Assembly, link)``) to ``parent`` (another assembly body, or None for
    the world) with a FixedJoint authored where they are at spawn (and at every reset). ``break_force`` N /
    ``break_torque`` N m: PhysX breaks it above either (inf: never). ``anchor``: the glue point in the child body's
    frame (default its origin); the torque is measured about it. ``armed=False``: declared off, switched on with
    ``Bond.arm()``. A fixed part without joints (a static assembly) as parent is the world plus a filtered
    collision pair, as forceps glues to fixed parts. Sim only: offline it is a declaration."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"bond name {name!r} must be a letter then letters, digits or _")
    c_name, c_body = _body_ref(child, "bond child")
    c_spec = child[0] if isinstance(child, tuple) else child
    if not getattr(c_spec, "_mechkit_assembly", False):
        raise TypeError(f"bond {name!r}: the child must be a mechkit Assembly (its USD prims are written here)")
    if c_spec.kind == "static" or (c_body == "base" and c_spec.fixed_base):
        raise ValueError(f"bond {name!r}: the child {c_name!r} {c_body!r} is fixed in place; glue a moving body")
    p_name, p_body, p_static = None, "base", False
    if parent is not None:
        p_spec = parent[0] if isinstance(parent, tuple) else parent
        if not getattr(p_spec, "_mechkit_assembly", False):
            raise TypeError(f"bond {name!r}: the parent must be a mechkit Assembly or None (the world)")
        p_name, p_body = _body_ref(parent, "bond parent")
        p_static = p_spec.kind == "static"
        if (p_name, p_body) == (c_name, c_body):
            raise ValueError(f"bond {name!r}: child and parent are the same body")
    limits = []
    for label, value in (("break_force", break_force), ("break_torque", break_torque)):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = math.nan
        if math.isnan(v) or not v > 0:
            raise ValueError(f"bond {name!r}: {label} must be positive (inf: unbreakable), got {value!r}")
        limits.append(v)
    bf, bt = limits
    a = (0.0, 0.0, 0.0) if anchor is None else _vec3(anchor, f"bond {name!r} anchor")
    b = Bond(name, c_name, c_body, p_name, p_body, p_static, bf, bt, a, bool(armed),
             _number(slip_mm, "slip_mm", positive=True) / 1000.0, _number(slip_deg, "slip_deg", positive=True),
             child_spec=c_spec, parent_spec=None if parent is None else p_spec)
    b.enabled = b.armed
    _hub().bonds[name] = b  # a task file loaded twice replaces, never stacks
    return b


def set_targets(assembly, joint: str, value: float) -> float:
    """Set the drive target of one joint of an articulated Assembly (or its name), in SI units: metres for a
    slider, RADIANS for a hinge (Isaac Lab's ``set_joint_position_target``; the USD attribute would be degrees).
    The spring then pulls toward ``value`` instead of 0. Written at the start of every control step from then on
    (sim), so a task's :func:`on_step` rule can move it each step: a rule-based detent, as tasks/meter-test-switch
    does (``target = where(in_jaw, 0, q - bias)``). Every target goes back to 0 at reset, so set a lasting one from
    an ``on_step`` (or ``on_reset``) callback, not once at import."""
    name = assembly.name if getattr(assembly, "_mechkit_assembly", False) else assembly
    hub = _hub()
    spec = hub.specs.get(name) or (assembly if getattr(assembly, "_mechkit_assembly", False) else None)
    if spec is None:
        raise KeyError(f"set_targets: no Assembly named {name!r}")
    j = spec.link(joint)  # KeyError for an unknown joint
    v = _number(value, f"set_targets {name}.{joint}")
    bound = 2 * math.pi if j.kind == "revolute" else 2.0
    if abs(v) > bound:
        hint = " (degrees? give radians)" if j.kind == "revolute" else ""
        raise ValueError(f"set_targets {name}.{joint}: {v} {j.unit} is beyond +-{bound:.3f} {j.unit}{hint}")
    if j.spring <= 0:
        _loud(f"WARNING: set_targets {name}.{joint}: the joint has no spring, so a target does nothing")
    if not j.lower - 1e-9 <= v <= j.upper + 1e-9:
        _loud(f"WARNING: set_targets {name}.{joint}: a target outside the joint's limits [{j.lower:g}, {j.upper:g}] "
              f"{j.unit} (a preload against a stop is fine; a hinge target in degrees is not)")
    hub.specs.setdefault(name, spec)
    hub.targets.setdefault(name, {})[joint] = v
    hub.targeted.setdefault(name, set()).add(joint)
    return v


def detent_target(q: float, notches, capture: float, bias: float = 0.0) -> float:
    """A rule-based detent's drive target: the nearest notch when ``q`` is within ``capture`` of it (the spring
    snaps the joint there), else ``q - bias`` (it holds where it was left; tasks/meter-test-switch biases toward
    closed with ``q - bias``). Same units as the joint (m or rad)."""
    q = float(q)
    near = min((float(n) for n in notches), key=lambda n: abs(n - q), default=None)
    if near is not None and abs(near - q) <= float(capture):
        return near
    return q - float(bias)


def on_step(fn):
    """Run ``fn(states)`` at the start of every control step in the simulator (before the action, the physics and
    the scoring), with every task object's state; it may fire welds, arm or release bonds and set targets.
    Re-registering the same function (a task file loaded twice) replaces it. Sim only."""
    _hub().step_fns[_fn_key(fn)] = fn
    return fn


def on_reset(fn):
    """Run ``fn()`` after every episode reset (sim) and every :func:`offline_reset`, after the kit's own reset
    (welds released, bonds re-armed, targets back to 0)."""
    _hub().reset_fns[_fn_key(fn)] = fn
    return fn


def _fn_key(fn) -> tuple:
    """The same source function gives the same key, whatever module name the task file was loaded under (the
    offline checker and tests load it under fresh names), so a reload replaces its callbacks instead of stacking
    a stale copy that still holds the old module's welds and bonds."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None), id(fn))
    return (getattr(fn, "__qualname__", None), code.co_filename, code.co_firstlineno)


def rules() -> dict:
    """Every weld's, bond's and target's state now (also written to the MECHKIT_TRACE rows as ``rules``)."""
    hub = _hub()
    return {"welds": {n: w.status() for n, w in hub.welds.items()},
            "bonds": {n: b.status() for n, b in hub.bonds.items()},
            "targets": {n: dict(t) for n, t in hub.targets.items()}}


def _has_rules(hub) -> bool:
    return bool(hub.welds or hub.bonds or hub.targets or hub.step_fns)


def _rule_states(environment, names) -> dict:
    """Every named object's state, read without touching the scorer's cache: assemblies through read_state, the
    rest through yam_arena.scene.object_states as yam_arena wrote it."""
    hub = _hub()
    out = {n: read_state(environment, hub.specs[n]) for n in names if n in hub.specs}
    plain = [n for n in names if n not in hub.specs]
    if plain:
        scene = sys.modules.get(_SCENE)
        fn = getattr(scene, "object_states", None)
        while fn is not None and _is_ours(fn) and getattr(fn, "__wrapped__", None) is not None:
            fn = fn.__wrapped__
        if fn is None:
            raise RuntimeError("mechkit: yam_arena.scene.object_states is not available for the rules")
        out.update(fn(environment, plain))
    return out


def _run_rules(environment, states: dict) -> None:
    """One control step of rules (see the section header for the order). ``hub.impl`` holds the simulator writes,
    so this runs offline on fake ones in the tests."""
    hub = _hub()
    impl = hub.impl
    for fn in list(hub.step_fns.values()):
        fn(states)
    for b in hub.bonds.values():
        b.stepped = True
        for request in b.pending:
            if request == "release" and b.enabled:
                impl.bond_enable(environment, b, False)
                b.enabled, b.released = False, True
            elif request == "arm":
                impl.bond_author(environment, b, states, new=b.path is not None, enabled=True)
                b.enabled, b.broken, b.at_rest_frames = True, False, False
                b.glued_at(states)
        b.pending.clear()
        if not (b.child in states and (b.parent is None or b.parent_static or b.parent in states)):
            _loud(f"ERROR: bond {b.name!r}: no state for {b.child!r} / {b.parent!r}")
            continue
        if b.check_break(states):
            _loud(f"bond {b.name!r} broke at episode {hub.episode} step {hub.step} (the pair moved more than "
                  f"{b.slip * 1000:.1f} mm / {b.slip_deg:.1f} deg apart)", once=False)
    for w in hub.welds.values():
        if w.part not in states or (w.base is not None and w.base not in states):
            _loud(f"ERROR: weld {w.name!r}: no state for {w.part!r} / {w.base!r}")
            continue
        if not w.active and w.when is not None:
            try:
                fire = bool(w.when(states))
            except Exception as exc:  # noqa: BLE001 - a broken condition never welds (fails closed), loudly
                _loud(f"ERROR: weld {w.name!r}: its condition raised {exc!r}; not welding")
                fire = False
            if fire and w.fire(states):
                _loud(f"weld {w.name!r} fired at episode {hub.episode} step {hub.step}", once=False)
        if w.active:
            if not (_finite_state(states[w.part]) and (w.base is None or _finite_state(states[w.base]))):
                _loud(f"ERROR: weld {w.name!r}: a non-finite state; not rewriting this step")
                continue
            impl.write_pose(environment, w.part, *w.target(states))
            w.writes += 1
    for name, targets in hub.targets.items():
        if targets:
            impl.write_targets(environment, hub.specs[name], dict(targets))


def _rest_states(b: Bond) -> dict:
    """The declared spawn states (env frame) of a bond's two ends: what a reset puts them back to (forceps re-glues
    from its declared spawn poses too, never from wherever a fallen part lies)."""
    out = {b.child: b.child_spec.rest_state()}
    if b.parent is not None:
        out[b.parent] = b.parent_spec.rest_state()
    return out


def _prepare_bond(environment, b: Bond) -> str:
    """Bring one bond to its declared state for the next episode (see :meth:`Bond.reset_plan`); idempotent, so it
    runs before the reset (the joint must hold a hanging part through the reset's settle steps, with frames from
    the declared spawn poses the reset teleports to) and again after it. ``environment`` None: bookkeeping only."""
    hub = _hub()
    plan = b.reset_plan()
    if environment is not None:
        if plan == "new" or (plan == "author" and b.child in hub.paths
                             and (b.parent is None or b.parent in hub.paths)):
            hub.impl.bond_author(environment, b, _rest_states(b), new=plan == "new", enabled=True)
            if plan == "new":
                _loud(f"bond {b.name!r} re-armed for the next episode (new joint {b.path})", once=False)
        elif plan == "off" and b.path is not None and b.enabled:
            hub.impl.bond_enable(environment, b, False)
    b.pending.clear()
    b.enabled = b.armed
    b.broken = b.released = b.stepped = False
    b.broke_step = None
    b.at_rest_frames = True
    return plan


def _warn_off_spawn(b: Bond) -> None:
    """Say so (once) when a bond's pair, after a reset, is not where its re-armed frames put it: the joint, authored
    from the declared spawn poses, then pulls the pair back together (``object_jitter`` > 0 moved a glued part)."""
    try:
        pos, rot = _relative(*b.frames_of(_rest_states(b)))
    except Exception:  # noqa: BLE001 - a diagnostic only
        return
    moved = float(np.linalg.norm(b.rel[0] - pos)) if b.rel is not None else 0.0
    turned = math.degrees(_rotation_angle(rot.T @ b.rel[1])) if b.rel is not None else 0.0
    if moved > b.slip or turned > b.slip_deg:
        _loud(f"WARNING: bond {b.name!r}: after the reset its pair is {moved * 1000:.1f} mm / {turned:.1f} deg from "
              f"the declared spawn pose its joint was authored at (object_jitter > 0?); the joint pulls it back. Run "
              f"with -E object_jitter=0")


def _zero_targets(environment) -> None:
    hub = _hub()
    if environment is not None:
        for name, joints in hub.targeted.items():
            if joints and name in hub.specs:
                hub.impl.write_targets(environment, hub.specs[name], dict.fromkeys(sorted(joints), 0.0))
    hub.targets.clear()


def _before_reset_rules(environment) -> None:
    """Before a reset teleports the bodies and lets them settle: welds released, drive targets back to 0 (a latched
    plunger must not stay down through the settle), bonds re-armed from their declared spawn poses."""
    hub = _hub()
    for w in hub.welds.values():
        w.reset()
    _zero_targets(environment)
    if environment is None:  # nothing to write to (the first reset builds the sim): keep each bond's history, so
        return  # the after-reset pass, which has the env, still re-arms what broke or was released
    for b in hub.bonds.values():
        _prepare_bond(environment, b)


def _reset_rules(environment, states: dict | None) -> None:
    """After a reset: the same as before it (idempotent: a wrapper that missed the before-reset hook still re-arms
    here), then each bond's glued pose from the live states (for its break check), then the on_reset callbacks.
    ``environment`` None (offline): only the bookkeeping."""
    hub = _hub()
    for w in hub.welds.values():
        w.reset()
    _zero_targets(environment)
    hub.targeted.clear()
    for b in hub.bonds.values():
        _prepare_bond(environment, b)
        b.rel = None
        if environment is not None and states is not None and b.enabled and b.path is not None:
            b.glued_at(states)
            _warn_off_spawn(b)
    for fn in list(hub.reset_fns.values()):
        fn()


# == per-step rules: the simulator side (lazy imports; copied from the accepted tasks named in each docstring) =======


def _t(x):
    """A torch tensor from an Isaac Lab 3 buffer (``.torch``) or a tensor (tasks/bayonet-mate partkit ``_t``)."""
    return getattr(x, "torch", x)


def _write_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> None:
    """Teleport an object's root (env 0) to a pose in the env frame with the given velocity: tasks/bayonet-mate
    partkit ``write_pose``, unchanged."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = _t(base.scene.env_origins[0])
    pose = torch.tensor([*(float(v) for v in pos), *(float(v) for v in quat_xyzw)], dtype=origin.dtype,
                        device=origin.device).unsqueeze(0)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    v = torch.tensor([*(float(x) for x in lin_vel), *(float(x) for x in ang_vel)], dtype=origin.dtype,
                     device=origin.device).unsqueeze(0)
    asset.write_root_velocity_to_sim_index(root_velocity=v)


def _write_targets(environment, spec: Assembly, targets: dict) -> None:
    """Joint position targets (m / rad) for some joints of an articulated Assembly (env 0): Isaac Lab's
    ``set_joint_position_target(target, joint_ids=...)`` as tasks/meter-test-switch calls it every step; the env
    writes the buffer to PhysX at every physics substep."""
    import torch

    base = environment.unwrapped
    asset = base.scene[spec.name]
    names = list(asset.joint_names)
    missing = [j for j in targets if j not in names]
    if missing:
        raise RuntimeError(f"mechkit: {spec.name}: no joints {missing} in the simulator (joints: {names})")
    ids = [names.index(j) for j in targets]
    origin = _t(base.scene.env_origins[0])
    value = torch.tensor([[float(targets[j]) for j in targets]], dtype=origin.dtype, device=origin.device)
    asset.set_joint_position_target(value, joint_ids=ids)


def _body_path(name: str, body: str) -> str:
    hub = _hub()
    if name not in hub.paths:
        raise RuntimeError(f"mechkit: {name!r} was not spawned by mechkit (no prim path)")
    return f"{hub.paths[name]}/{body}"


def _prim_world_pose(stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, quat_xyzw) of a prim in the world, from USD (as the fixed-base world anchor is placed)."""
    from pxr import Usd, UsdGeom

    world = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t, q = world.ExtractTranslation(), world.ExtractRotationQuat()
    return (np.array([float(t[0]), float(t[1]), float(t[2])]),
            np.array([*(float(v) for v in q.GetImaginary()), float(q.GetReal())]))


def _prim_valid(stage, path: str) -> bool:
    try:
        return bool(stage.GetPrimAtPath(path).IsValid())
    except Exception:  # noqa: BLE001 - a stage without IsValid (a test double): define it
        return False


def _author_bond_prim(stage, b: Bond, child_pose, parent_pose, enabled: bool, new: bool) -> str:
    """Write the bond's FixedJoint: disabled first, bodies and frames, collisions between the pair off, excluded
    from any articulation, break force and torque, then ``jointEnabled`` LAST (tasks/snap-board-out-of-panel
    ``_joint``). ``new``: remove the old prim and write a fresh one under a new name (a broken PhysX joint stays
    broken; tasks/forceps-open-for-steam ``_reglue``)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if new and b.path is not None:
        stage.RemovePrim(b.path)
        b.generation += 1
        b.path = None
    if b.path is None:
        folder = f"{_hub().paths[b.child]}/bonds"
        if not _prim_valid(stage, folder):
            UsdGeom.Xform.Define(stage, folder)
        b.path = f"{folder}/{b.name}" + (f"_r{b.generation}" if b.generation else "")
    f = bond_frames(child_pose, parent_pose, b.anchor)
    joint = UsdPhysics.FixedJoint.Define(stage, b.path)
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(_body_path(b.child, b.child_body))])
    if parent_pose is not None:
        joint.CreateBody1Rel().SetTargets([Sdf.Path(_body_path(b.parent, b.parent_body))])
    (x0, y0, z0), (w0, i0, j0, k0) = f["local_pos0"], f["local_rot0_wxyz"]
    (x1, y1, z1), (w1, i1, j1, k1) = f["local_pos1"], f["local_rot1_wxyz"]
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x0, y0, z0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(w0, Gf.Vec3f(i0, j0, k0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(x1, y1, z1))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(w1, Gf.Vec3f(i1, j1, k1)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateBreakForceAttr().Set(float(min(b.break_force, UNBREAKABLE)))
    joint.CreateBreakTorqueAttr().Set(float(min(b.break_torque, UNBREAKABLE)))
    if b.parent_static:  # glued to a fixed part = to the world; keep the pair from colliding (forceps)
        pair = (_body_path(b.child, b.child_body), _body_path(b.parent, "base"))  # the Xform over its colliders
        filtered = _hub().filtered
        if pair not in filtered:  # once: the API stays on the child when a joint prim is removed, and a re-arm must
            # not edit a live rigid body's applied schemas (a schema change makes omni.physx re-parse the body)
            fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pair[0]))
            fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(pair[1]))
            filtered[pair] = True
    joint.CreateJointEnabledAttr().Set(bool(enabled))  # the last write
    return b.path


def _bond_author(environment, b: Bond, states: dict, *, new: bool, enabled: bool) -> None:
    """Author a bond at run time with frames from ``states`` (env frame; plus the env origin = world): the live
    states for :meth:`Bond.arm`, the declared spawn states for a re-arm at reset."""
    from isaaclab.sim.utils.stage import get_current_stage

    origin = _np(environment.unwrapped.scene.env_origins)[0]
    (pc, rc), (pp, rp) = b.frames_of(states)
    child = (pc + origin, matrix_to_quat(rc))
    parent = None if (b.parent is None or b.parent_static) else (pp + origin, matrix_to_quat(rp))
    _author_bond_prim(get_current_stage(), b, child, parent, enabled, new)


def _bond_enable(environment, b: Bond, on: bool) -> None:
    """Switch a bond's joint on or off at run time (tasks/forceps-open-for-steam ``_set_joint_enabled``)."""
    if b.path is None:
        return
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    UsdPhysics.Joint(get_current_stage().GetPrimAtPath(b.path)).GetJointEnabledAttr().Set(bool(on))


def _author_spawn_bonds(stage) -> list[str]:
    """In the spawner, once both ends of a bond exist: glue them where USD puts them at spawn (snap-board authors
    its joint in the second panel's spawner). Returns the bonds written."""
    hub = _hub()
    done = []
    for b in hub.bonds.values():
        if b.path is not None or b.child not in hub.paths or (b.parent is not None and b.parent not in hub.paths):
            continue
        child = _prim_world_pose(stage, _body_path(b.child, b.child_body))
        parent = None if (b.parent is None or b.parent_static) else _prim_world_pose(
            stage, _body_path(b.parent, b.parent_body))
        _author_bond_prim(stage, b, child, parent, b.armed, new=False)
        b.enabled = b.armed
        done.append(b.name)
    return done


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
    for key, factory in _HUB_V2.items():  # a hub made by an older copy of this module gets the new fields
        if key not in hub.__dict__:
            setattr(hub, key, factory())
    return hub


_HUB_V2 = {
    "welds": dict,  # name -> Weld
    "bonds": dict,  # name -> Bond
    "targets": dict,  # assembly name -> {joint: target (m / rad)}, written every step
    "targeted": dict,  # assembly name -> joints given a target this episode (set back to 0 at reset)
    "step_fns": dict,  # on_step callbacks
    "reset_fns": dict,  # on_reset callbacks
    "paths": dict,  # assembly name -> its root prim path, from the spawner
    "filtered": dict,  # (child body path, fixed parent path) -> True once its FilteredPairs entry is written
}


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
            before = getattr(hub.impl, "before_reset", None)
            if before is not None:
                before(self)
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


def _task_names(embodiment) -> list[str]:
    return [o.name for o in getattr(getattr(embodiment, "task", None), "objects", ())]


def _impl_before_step(embodiment) -> None:
    hub = _hub()
    hub.step += 1
    hub.env = getattr(embodiment, "_env", hub.env)
    if _has_rules(hub) and hub.env is not None and not hub.in_reset:
        _run_rules(hub.env, _rule_states(hub.env, _task_names(embodiment)))


def _impl_before_reset(embodiment) -> None:
    hub = _hub()
    if hub.welds or hub.bonds or hub.targeted:
        environment = getattr(embodiment, "_env", None)
        _before_reset_rules(hub.env if environment is None else environment)


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
            if _has_rules(hub) or hub.targeted:
                row["rules"] = rules()
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
    if hub.welds or hub.bonds or hub.targeted or hub.reset_fns:
        names = _task_names(embodiment)
        states = _rule_states(hub.env, names) if hub.bonds and hub.env is not None else None
        _reset_rules(hub.env, states)
        if hub.bonds or hub.welds:
            print(f"[mechkit] episode {hub.episode} rules after reset: {json.dumps(rules())}; bond prims "
                  f"{ {n: b.path for n, b in hub.bonds.items()} }", file=sys.stderr, flush=True)


# == public runtime API ===============================================================================================


def install() -> dict:
    """Hook yam_arena: now for the modules already imported, and from a ``sys.meta_path`` finder for the ones imported
    later. Idempotent; importing this module calls it. Returns :func:`status`."""
    hub = _hub()
    hub.impl = types.SimpleNamespace(
        task_object=_impl_task_object, object_states=_impl_object_states, before_step=_impl_before_step,
        after_step=_impl_after_step, after_reset=_impl_after_reset, before_reset=_impl_before_reset,
        patch_module=_patch_module, author=_author, author_spawn_bonds=_author_spawn_bonds,
        write_pose=_write_pose, write_targets=_write_targets, bond_author=_bond_author, bond_enable=_bond_enable,
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
    """Called by the offline checker before each playback: a new episode for latches keyed on :func:`episode`.
    The rules' bookkeeping resets too (welds released, bonds as declared, targets cleared), then on_reset."""
    hub = _hub()
    hub.episode += 1
    hub.step = 0
    hub.last.clear()
    _reset_rules(None, None)


install()
