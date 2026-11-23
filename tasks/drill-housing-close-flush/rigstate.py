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
