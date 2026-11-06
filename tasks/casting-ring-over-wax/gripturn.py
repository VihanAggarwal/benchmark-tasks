"""Which way each gripper's fingers face, for the scorer's finger test: the pose of each arm's `gripper` link.

``rigstate.py`` (the program kit's file, copied unchanged) reports each gripper's tool point, finger-tip point and
opening, but not its turn about its own axis, so on its own the scorer could only judge a finger by everywhere it
could be at some turn. This module reads, for each arm, the world pose of the link the tool point is defined in
(``gripper``: the tool point is 0.134 m along its z, the fingers close along its x), once per control step and at the
same moment as rigstate (before the embodiment steps the physics), by wrapping the embodiment's ``_task_step`` and
``reset`` the same way rigstate does. It changes nothing the embodiment does and reads only state the simulator
already has.

    gripturn.install()   # once, at import time of task.py, after rigstate.install()
    gripturn.bodies()    # {"left": [(pos, quat4), ...], "right": [...]}: candidate link poses, or None
    gripturn.error()     # why the last reading failed, if it did

Nothing here decides anything: ``scoring.gripper_turn`` accepts a candidate only if it puts the tool point within
3 mm of rigstate's and the link's z axis within 3 degrees of rigstate's tool-point-to-tips axis, trying both
quaternion orders (w, x, y, z) and (x, y, z, w). A reading that fails (another link, another frame convention, an
exception) leaves the turn unknown and the scorer falls back to the turn-free zone, which is never more lenient.
Offline (no simulator) nothing is read and the turn is unknown.
"""

from __future__ import annotations

import functools
import sys

import numpy as np

ARMS = ("left", "right")
BODY = "gripper"  # the URDF link the tool point is defined in
# (position, quaternion) buffers in the order tried; Isaac Lab versions name them differently
_SOURCES = (("body_link_pos_w", "body_link_quat_w"), ("body_pos_w", "body_quat_w"))
_POSE7 = "body_link_pose_w"
_MARK = "_gripturn_wrapped"

_STATE: dict = {"bodies": None, "error": None, "installed": False}


def bodies() -> dict | None:
    """{arm: [(position, quaternion), ...]} for the current control step (None before the first step or on error)."""
    return _STATE["bodies"]


def error() -> str | None:
    return _STATE["error"]


def _np(x) -> np.ndarray:
    t = getattr(x, "torch", None)  # Isaac Lab 3 data buffers expose .torch
    if t is not None and not callable(t):
        x = t
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _indices(robot) -> list[int]:
    names = [str(n) for n in robot.body_names]
    if BODY in names:
        return [names.index(BODY)]
    return [i for i, n in enumerate(names) if BODY in n.lower()]


def _read(data, idx, origin, source) -> list:
    if isinstance(source, str):
        pose = _np(getattr(data, source))[0]
        return [(pose[i, :3] - origin, pose[i, 3:7]) for i in idx]
    pos, quat = _np(getattr(data, source[0]))[0], _np(getattr(data, source[1]))[0]
    return [(pos[i] - origin, quat[i]) for i in idx]


def _snapshot(env) -> dict:
    base = env.unwrapped
    origin = _np(base.scene.env_origins)[0]
    out, errors = {}, []
    for arm in ARMS:
        robot = base.scene[f"{arm}_robot"]
        idx = _indices(robot)
        if not idx:
            raise LookupError(f"{arm}_robot has no body named like {BODY!r}: {list(robot.body_names)}")
        cands = []
        for source in (*_SOURCES, _POSE7):
            try:
                cands += _read(robot.data, idx, origin, source)
            except Exception as exc:  # noqa: BLE001 - this buffer is missing in this Isaac Lab version: try the next
                errors.append(f"{source}: {exc!r}")
        if not cands:
            raise LookupError(f"{arm}_robot: no body pose buffer could be read ({'; '.join(errors)})")
        out[arm] = cands
    return out


def _take(env) -> None:
    try:
        _STATE["bodies"] = _snapshot(env)
        _STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 - reported once, and the scorer then uses the turn-free zone
        _STATE["bodies"] = None
        if _STATE["error"] is None:
            print(f"[gripturn] could not read the gripper links: {exc!r} (the finger test uses the turn-free zone)",
                  file=sys.stderr)
        _STATE["error"] = repr(exc)


def _is_ours(fn) -> bool:
    seen = 0
    while fn is not None and seen < 64:  # functools.wraps copies marks up the chain; walk it anyway
        if getattr(fn, _MARK, False):
            return True
        fn, seen = getattr(fn, "__wrapped__", None), seen + 1
    return False


def install() -> bool:
    """Wrap the embodiment's ``_task_step`` and ``reset`` (idempotent). False when yam_arena's embodiment is not
    loaded (offline checks): then the turn stays unknown."""
    if _STATE["installed"]:
        return True
    module = sys.modules.get("yam_arena.inspect_embodiment")
    cls = getattr(module, "YamArenaEmbodiment", None) if module else None
    if cls is None:
        return False
    if not _is_ours(cls._task_step):
        original_step = cls._task_step

        @functools.wraps(original_step)
        def _task_step(self, *args, **kwargs):
            _take(self._env)
            return original_step(self, *args, **kwargs)

        setattr(_task_step, _MARK, True)
        cls._task_step = _task_step
    if not _is_ours(cls.reset):
        original_reset = cls.reset

        @functools.wraps(original_reset)
        def reset(self, *args, **kwargs):
            observation = original_reset(self, *args, **kwargs)
            _take(self._env)
            return observation

        setattr(reset, _MARK, True)
        cls.reset = reset
    _STATE["installed"] = True
    return True


def offline_reset() -> None:
    _STATE["bodies"] = None
    _STATE["error"] = None
