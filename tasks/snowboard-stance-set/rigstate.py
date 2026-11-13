"""The arms' state for a task file's scorer: where each TCP and its pad tips are, and how far each gripper is open.

``yam_arena`` hands a task file's ``success()``/``score()`` only the objects' states. Some rules are about the arms
("both grippers let go at the end", "this gripper never entered the keep-out box", "the part was held at its
handle"). This module wraps ``YamArenaEmbodiment._task_step`` and ``reset`` once, from the task file's import, so
that before every scoring call it takes a snapshot of both arms, and so a task can run code when an episode resets.
It changes nothing the embodiment does and reads only state the simulator already has.

    import rigstate
    rigstate.install()                 # once, at import time of task.py
    arms = rigstate.arms()             # {"left": Arm, "right": Arm} for the current control step, or None
    arms["left"].tcp, arms["left"].tips, arms["left"].opening   # env-frame metres; 0 closed .. 1 open
    rigstate.on_reset(fn)              # fn() runs after every episode reset
    rigstate.episode()                 # increments at every reset

Positions are in the same frame as ``ObjectState.pos`` (the env's origin subtracted). Offline (no simulator), the
checker in ``tools/offline`` feeds the same snapshot from its kinematics through ``offline_feed``.

Fail-closed helpers for scorers (version 2):

    rigstate.readable(arms)                       # both arms present, every number finite
    rigstate.released(arms, part, clearance=0.03) # both grippers' finger zones at least 3 cm from the part
    rigstate.keep_out_breached(arms, zone, margin=0.0)
    rigstate.hands_gap(arms, part)                # the nearer finger zone's distance to the part, None if unreadable

THE RULE: a step whose arms are not ``readable`` (None, a side missing, a non-finite number: the snapshot failed or
the simulator blew up), or whose distance to the part cannot be computed (the distance function raised or returned a
non-finite number), is NOT released and IS a breach of every keep-out rule. ``released`` returns False and
``keep_out_breached`` returns True on such a step, so a scorer built on them can never be credited by an arm it could
not see. A scorer that needs "released for the last 0.5 s" must see ``released`` True on every one of those steps.

``part`` (or ``zone``) is either a function from (M, 3) world points to (M,) distances to the part (0 inside), such
as ``mechkit.AssemblyState.distance`` (exact for boxes and cylinders) or ``boxes_distance(boxes)`` (prefer these), or
an (N, 3) array of world points on the part's surface. Points are only as good as their sampling: the distance to
the nearest sample over-states the true gap (the 8 corners of a 40 mm cube put a finger touching a face centre 28 mm
away), so sample faces at about 1 mm or use a distance function.

The finger zone is the one tasks/snowboard-stance-set's scorer uses (gripper heading unknown, so the fingers may be
anywhere round the gripper's axis): every point within ``zone_radius(opening)`` (half the finger gap plus a pad's
8 mm thickness, and half its 27 mm width: 56.6 mm fully open, 15.7 mm closed) of the 38 mm segment from the pad-tip
point back toward the TCP. A finger resting on any part of the object, open or closed, is therefore not released.

The technique (wrapping the embodiment's step and reset from a task file) follows the one first used in the program
repository's ``tasks/bearing-press``; this is an independent implementation. Written for yam_arena 0.3.1.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ARMS = ("left", "right")
FINGER_JOINTS = ("joint7", "joint8")
FINGER_OPEN = -0.04695  # finger joint value with the pads 94 mm apart (yam_arena.constants.FINGER_OPEN)
_MARK = "_rigstate_wrapped"

# The finger zone (tasks/snowboard-stance-set/layout.py): metres.
FULL_GAP = 0.0939  # the pads' gap fully open (opening 1.0)
FINGER_THICK = 0.008
FINGER_HALF_WIDTH = 0.0135
FINGER_LEN = 0.038  # the zone's axis: from the pad-tip point this far back toward the TCP
ZONE_SAMPLES = 39  # points along that axis, 1 mm apart
DEFAULT_CLEARANCE = 0.030  # "let go": every finger zone at least 3 cm from the part (the review bar: 2 to 3 cm)


@dataclass(frozen=True)
class Arm:
    tcp: np.ndarray  # TCP position (m)
    tips: np.ndarray  # pad-tip point, 13 mm past the TCP along the approach axis (m)
    opening: float  # 0 fingers closed .. 1 fully open (mean of the two finger joints)

    def is_open(self, threshold: float = 0.9) -> bool:
        return self.opening >= threshold


_STATE: dict = {"arms": None, "episode": 0, "steps": 0, "error": None, "installed": False}
_RESET_CALLBACKS: dict[tuple, Callable[[], None]] = {}


def arms() -> dict[str, Arm] | None:
    """The arms at the current control step (None before the first step, or if they could not be read)."""
    return _STATE["arms"]


def episode() -> int:
    return _STATE["episode"]


def error() -> str | None:
    """Why the last snapshot failed, if it did."""
    return _STATE["error"]


def on_reset(fn: Callable[[], None]) -> Callable[[], None]:
    """Run ``fn()`` after every episode reset. Re-registering the same function (same module, name and line, as
    when a task file is loaded twice) replaces it instead of stacking."""
    code = getattr(fn, "__code__", None)
    key = (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None),
           *((code.co_filename, code.co_firstlineno) if code else (id(fn),)))
    _RESET_CALLBACKS[key] = fn
    return fn


def _np(x) -> np.ndarray:
    t = getattr(x, "torch", None)  # Isaac Lab 3 data buffers expose .torch
    if t is not None and not callable(t):
        x = t
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _snapshot(env) -> dict[str, Arm]:
    from yam_arena import runtime  # needs the running simulator

    base = env.unwrapped
    origin = _np(base.scene.env_origins)[0]
    out = {}
    for arm in ARMS:
        robot = base.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        q = _np(robot.data.joint_pos)[0]
        fingers = [float(q[names.index(j)]) for j in FINGER_JOINTS]
        opening = float(np.clip(np.mean(fingers) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = Arm(
            tcp=np.asarray(runtime.tcp_world(robot), dtype=float) - origin,
            tips=np.asarray(runtime.pad_tips_world(robot), dtype=float) - origin,
            opening=opening,
        )
    return out


def _take(env) -> None:
    try:
        _STATE["arms"] = _snapshot(env)
        _STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 - reported, never silently guessed
        _STATE["arms"] = None
        if _STATE["error"] is None:
            print(f"[rigstate] could not read the arms: {exc!r}", file=sys.stderr)
        _STATE["error"] = repr(exc)


def _reset_happened() -> None:
    _STATE["episode"] += 1
    _STATE["steps"] = 0
    for fn in list(_RESET_CALLBACKS.values()):
        fn()


def install() -> bool:
    """Wrap the embodiment's ``_task_step`` and ``reset`` (idempotent). False when yam_arena's embodiment is not
    loaded (offline checks): then ``offline_feed`` supplies the arms instead."""
    if _STATE["installed"]:
        return True
    module = sys.modules.get("yam_arena.inspect_embodiment")
    cls = getattr(module, "YamArenaEmbodiment", None) if module else None
    if cls is None:
        return False
    if not getattr(cls._task_step, _MARK, False):
        original_step = cls._task_step

        @functools.wraps(original_step)
        def _task_step(self, *args, **kwargs):
            _take(self._env)
            _STATE["steps"] += 1
            return original_step(self, *args, **kwargs)

        setattr(_task_step, _MARK, True)
        cls._task_step = _task_step
    if not getattr(cls.reset, _MARK, False):
        original_reset = cls.reset

        @functools.wraps(original_reset)
        def reset(self, *args, **kwargs):
            observation = original_reset(self, *args, **kwargs)
            _reset_happened()
            _take(self._env)
            return observation

        setattr(reset, _MARK, True)
        cls.reset = reset
    _STATE["installed"] = True
    return True


# -- offline (tools/offline/check_task.py) ---------------------------------------------------------------------


def offline_reset() -> None:
    _STATE["arms"] = None
    _reset_happened()


def offline_feed(snapshot: dict[str, dict]) -> None:
    """Called by the offline checker once per control step, before the task's hooks."""
    _STATE["arms"] = {
        a: Arm(tcp=np.asarray(s["tcp"], float), tips=np.asarray(s["tips"], float), opening=float(s["opening"]))
        for a, s in snapshot.items()
    }
    _STATE["steps"] += 1


# -- fail-closed helpers for scorers ------------------------------------------------------------------------------


def read_arms(arms) -> list[tuple[np.ndarray, np.ndarray, float]] | None:
    """Both grippers as (tcp, tips, opening) in ARMS order, or None when the arms are unreadable: not a dict, a side
    missing, a vector not 3 numbers, or any number not finite."""
    if not isinstance(arms, dict):
        return None
    out = []
    try:
        for side in ARMS:
            a = arms[side]
            tcp = np.asarray(a.tcp, dtype=float).ravel()
            tips = np.asarray(a.tips, dtype=float).ravel()
            opening = float(a.opening)
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return None
            if not np.isfinite(opening):
                return None
            out.append((tcp, tips, opening))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return out


def readable(arms) -> bool:
    """Both arms present and every number finite. When False: not released, and a breach of any keep-out rule."""
    return read_arms(arms) is not None


def zone_radius(opening: float) -> float:
    """From the gripper's axis to a finger pad's outer corner (m) at ``opening`` (0 closed .. 1 open)."""
    gap = float(np.clip(opening, 0.0, 1.0)) * FULL_GAP
    return float(np.hypot(gap / 2 + FINGER_THICK, FINGER_HALF_WIDTH))


def zone_axis(tcp, tips) -> np.ndarray:
    """The ZONE_SAMPLES points of the zone's axis: from the pad-tip point FINGER_LEN back toward the TCP."""
    tcp, tips = np.asarray(tcp, dtype=float).reshape(3), np.asarray(tips, dtype=float).reshape(3)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    return tips - np.outer(np.linspace(0.0, FINGER_LEN, ZONE_SAMPLES), u)


def _distance_fn(part):
    if callable(part):
        return part
    pts = np.atleast_2d(np.asarray(part, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError(f"part must be (N, 3) world points or a distance function, got shape {pts.shape}")

    def distance(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        return np.min(np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2), axis=1)

    return distance


def zone_gap(arm, part) -> float:
    """How far one gripper's finger zone is from the part (m; negative: they overlap). ``arm`` is an Arm or a
    (tcp, tips, opening) tuple."""
    tcp, tips, opening = (arm.tcp, arm.tips, arm.opening) if isinstance(arm, Arm) else arm
    distance = _distance_fn(part)  # a malformed ``part`` is a bug in the scorer: it raises here
    try:
        d = np.asarray(distance(zone_axis(tcp, tips)), dtype=float)
    except Exception as exc:  # noqa: BLE001 - a distance that cannot be computed is as close as it gets (fail closed)
        if _STATE.get("distance_error") != repr(exc):
            _STATE["distance_error"] = repr(exc)
            print(f"[rigstate] a part's distance raised {exc!r}: counted as touching (not released)", file=sys.stderr)
        return -np.inf
    if d.size == 0 or not np.all(np.isfinite(d)):
        return -np.inf  # a distance we cannot trust is as close as it gets (fail closed)
    return float(d.min()) - zone_radius(opening)


def hands_gap(arms, part) -> float | None:
    """The nearer of the two grippers' finger zones to the part (m), or None when the arms are unreadable."""
    read = read_arms(arms)
    if read is None:
        return None
    return min(zone_gap(a, part) for a in read)


def released(arms, part, clearance: float = DEFAULT_CLEARANCE) -> bool:
    """Both grippers have let go of the part: arms readable and every finger zone at least ``clearance`` from it.
    Unreadable arms: False (never released)."""
    gap = hands_gap(arms, part)
    return gap is not None and gap >= clearance - 1e-9


def keep_out_breached(arms, zone, margin: float = 0.0) -> bool:
    """A finger zone within ``margin`` of a keep-out ``zone`` (points or a distance function). Unreadable arms:
    True (always a breach)."""
    gap = hands_gap(arms, zone)
    return gap is None or gap < margin - 1e-9


def box_distances(points, center, size, rotation=None) -> np.ndarray:
    """Distance from each world point to a box (``center``, edge lengths ``size``, axes the columns of
    ``rotation``, default the world's); 0 inside."""
    r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    local = (np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(center, dtype=float)) @ r
    q = np.abs(local) - np.asarray(size, dtype=float) / 2
    return np.linalg.norm(np.maximum(q, 0.0), axis=1)


def boxes_distance(boxes):
    """A distance function to the union of boxes [(center, size) or (center, size, rotation)], for ``released``."""
    boxes = [tuple(b) for b in boxes]
    if not boxes:
        raise ValueError("boxes_distance needs at least one box")

    def distance(points):
        return np.min([box_distances(points, *b) for b in boxes], axis=0)

    return distance

