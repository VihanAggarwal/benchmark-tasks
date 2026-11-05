"""The scorer, as plain functions of the four loose objects' states and the arms' state. NumPy only; no simulator.

The hot-work rule is about the whole trial, not only its end: every combustible must be outside the 21 cm radius
and the fire watch posted (the extinguisher held upright over the hot-work mat) at every moment of hot work; once
hot work has started no combustible may come back inside the radius; after the last moment of hot work the watch
must stay posted for 10 s; and neither the torch nor the extinguisher may ever be put down anywhere but in its
holder. So the scorer keeps a little state (what has happened, and for how long) and is fed one observation per
control step by ``task.success()``.

    Tracker.update(torch, extinguisher, carton, rag, arms) -> Result(score, success, details)

The objects are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then 'held' is taken as met for that step (the rules that
need it are kept) and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

'Held': a gripper is closed on an upright cylinder (the torch or the extinguisher) when its finger gap is no more
than 6 mm wider than the cylinder's diameter and the point between its finger tips lies within 12 mm of the
cylinder's axis, between its two ends. Either gripper may hold either object; one gripper cannot hold both.

The torch's 'tip' is the centre of its lower end (the end of its axis that is lower).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
SIDES = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a unit quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in SIDES:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def axis(s) -> np.ndarray:
    """The body's z axis in the world (a cylinder's axis)."""
    return rotation(s.quat_xyzw)[:, 2]


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's z axis and the world's vertical (degrees, 0..180; upside down is 180)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def lower_end(s, height: float) -> np.ndarray:
    """Centre of a cylinder's lower end (whichever end of its axis is lower)."""
    c, u = np.asarray(s.pos, dtype=float), axis(s)
    a, b = c + u * height / 2, c - u * height / 2
    return a if a[2] < b[2] else b


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def held_by(arm, s, radius: float, height: float) -> bool:
    """This gripper is closed on the cylinder ``s``: finger gap at most 6 mm wider than its diameter, and the point
    between the finger tips within 12 mm of its axis, between its two ends."""
    if finger_gap(arm) > 2 * radius + L.HELD_SLACK:
        return False
    c, u = np.asarray(s.pos, dtype=float), axis(s)
    d = np.asarray(arm.tips, dtype=float) - c
    t = float(d @ u)
    if abs(t) > height / 2:
        return False
    return bool(np.linalg.norm(d - t * u) <= L.HELD_AXIS)


def holder(arms, s, radius: float, height: float) -> str | None:
    """Which gripper holds the cylinder ``s`` ('left', 'right' or None). Arms must be valid."""
    for side in SIDES:
        if held_by(arms[side], s, radius, height):
            return side
    return None


# -- the objects -------------------------------------------------------------------------------------------------
def outline_gap(s) -> float:
    """Horizontal distance from the weld mark's centre to the object's outline seen from above (its world-aligned
    bounding box); 0 if the mark's centre is under it."""
    lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
    dx = max(0.0, lo[0] - L.MARK_X, L.MARK_X - hi[0])
    dy = max(0.0, lo[1] - L.MARK_Y, L.MARK_Y - hi[1])
    return float(math.hypot(dx, dy))


def area_clear(combustibles) -> bool:
    """No part of any combustible's outline within 21 cm of the mark's centre (none of it over the mat)."""
    return all(outline_gap(c) > L.RADIUS for c in combustibles)


def in_holder(s, radius: float, height: float, cup_x: float, cup_y: float, inner: float) -> bool:
    """The cylinder's lower end over its cup's floor and its lowest point below the cup's rim: it is in the holder
    (standing or leaning)."""
    e = lower_end(s, height)
    inside = abs(e[0] - cup_x) <= inner / 2 and abs(e[1] - cup_y) <= inner / 2
    return bool(inside and float(s.low[2]) <= L.TOP + L.HOLDER_HEIGHT)


def parked(s, radius: float, height: float, cup_x: float, cup_y: float, inner: float) -> bool:
    """Standing upright in its holder: in it, tilted under 10 degrees, its lowest point within 5 mm of the floor."""
    return bool(in_holder(s, radius, height, cup_x, cup_y, inner) and tilt_deg(s.quat_xyzw) < L.PARKED_TILT_DEG
                and float(s.low[2]) <= L.TOP + L.PARKED_FLOOR)


def torch_in_holder(t) -> bool:
    return in_holder(t, L.TORCH_RADIUS, L.TORCH_HEIGHT, L.TORCH_X, L.TORCH_Y, L.TORCH_CUP_INNER)


def ext_in_holder(e) -> bool:
    return in_holder(e, L.EXT_RADIUS, L.EXT_HEIGHT, L.EXT_X, L.EXT_Y, L.EXT_CUP_INNER)


def torch_parked(t) -> bool:
    return parked(t, L.TORCH_RADIUS, L.TORCH_HEIGHT, L.TORCH_X, L.TORCH_Y, L.TORCH_CUP_INNER)


def ext_parked(e) -> bool:
    return parked(e, L.EXT_RADIUS, L.EXT_HEIGHT, L.EXT_X, L.EXT_Y, L.EXT_CUP_INNER)


def tip(t) -> np.ndarray:
    return lower_end(t, L.TORCH_HEIGHT)


def tip_height(t) -> float:
    """Height of the torch's tip above the plate's top."""
    return float(tip(t)[2]) - L.PLATE_TOP


def tip_off_mark(t) -> float:
    p = tip(t)
    return float(math.hypot(p[0] - L.MARK_X, p[1] - L.MARK_Y))


def is_hot(t) -> bool:
    """Hot work: the torch's tip over the plate (inside its 120 x 120 mm outline) and at most 25 mm above its top."""
    p = tip(t)
    over = abs(p[0] - L.MARK_X) <= L.PLATE_HALF and abs(p[1] - L.MARK_Y) <= L.PLATE_HALF
    return bool(over and tip_height(t) <= L.HOT_HEIGHT)


def on_mark(t) -> bool:
    """Welding the mark: hot, with the tip within 15 mm of the mark's centre horizontally."""
    return bool(is_hot(t) and tip_off_mark(t) <= L.ON_MARK)


def watch_spot(e) -> bool:
    """The extinguisher where the fire watch needs it: upright within 20 degrees, its centre within 21 cm of the
    mark's centre horizontally (over the mat). Whether it is held is checked separately."""
    return bool(tilt_deg(e.quat_xyzw) < L.WATCH_TILT_DEG
                and L.dist_to_mark(float(e.pos[0]), float(e.pos[1])) <= L.RADIUS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "hot_not_clear": "hot work while a combustible was within 21 cm of the mark",
    "hot_unwatched": "hot work while the fire watch was not posted",
    "combustible_returned": "a combustible within 21 cm of the mark after hot work had started",
    "torch_set_down": "the torch out of its holder and not held for more than 0.25 s",
    "extinguisher_set_down": "the extinguisher out of its holder and not held for more than 0.25 s",
    "off_table": "a loose object fell off the table",
}

STARTS = (  # (x, y, lowest point) of the torch, extinguisher, carton and rag at the start
    (L.TORCH_X, L.TORCH_Y, L.TOP),
    (L.EXT_X, L.EXT_Y, L.TOP),
    (L.CARTON_X, L.CARTON_Y, L.MAT_TOP),
    (L.RAG_X, L.RAG_Y, L.MAT_TOP),
)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.loose_limit = int(round(L.LOOSE_S * control_hz))  # 5 steps: breach on the 6th in a row
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.hot_steps = 0  # control steps of hot work (tip over the plate, within 25 mm of it)
        self.weld_steps = 0  # ... of them, welding the mark
        self.last_hot_step: int | None = None
        self.post_steps = 0  # control steps the watch has stayed posted, without a break, since the last hot step
        self.post_open = False  # still counting (no break since the last hot step)
        self.ready_run = 0  # control steps in a row with the area clear and the watch posted
        self.ready_best = 0
        self.torch_loose = 0  # control steps in a row with the torch out of its holder and not held
        self.ext_loose = 0
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.min_gap = math.inf  # m: the closest any combustible's outline came to the mark after hot work began
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of the four objects at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @property
    def weld_s(self) -> float:
        return self.weld_steps * self.dt

    @property
    def post_watch_s(self) -> float:
        return self.post_steps * self.dt

    @property
    def ready_best_s(self) -> float:
        return self.ready_best * self.dt

    @staticmethod
    def at_start(*objs) -> bool:
        """The starting layout: each object within START_TOL of its starting place, at its starting height, upright
        and still (START_TOL covers a seeded reset's jitter)."""
        for s, (x, y, z) in zip(objs, STARTS):
            if not (math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
                    and abs(float(s.low[2]) - z) < 0.003 and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def _posted(self, ext, arms, have_arms: bool) -> bool:
        held = holder(arms, ext, L.EXT_RADIUS, L.EXT_HEIGHT) is not None if have_arms else True
        return bool(held and watch_spot(ext))

    def update(self, torch, ext, carton, rag, arms=None) -> Result:
        objs = (torch, ext, carton, rag)
        if not all(valid(s) for s in objs):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(*objs, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in objs]
        if self.prev is not None and self._teleported(now) and self.at_start(*objs):
            self.reset()  # the simulator put everything back: a new episode (rigstate's on_reset hook normally does
            #               this first; this is the fallback)
        self.prev = now
        self.steps += 1

        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1
        t_held = holder(arms, torch, L.TORCH_RADIUS, L.TORCH_HEIGHT) is not None if have_arms else True
        e_held = holder(arms, ext, L.EXT_RADIUS, L.EXT_HEIGHT) is not None if have_arms else True

        self.torch_loose = 0 if (t_held or torch_in_holder(torch)) else self.torch_loose + 1
        self.ext_loose = 0 if (e_held or ext_in_holder(ext)) else self.ext_loose + 1
        if self.torch_loose > self.loose_limit:
            self._flag("torch_set_down")
        if self.ext_loose > self.loose_limit:
            self._flag("extinguisher_set_down")
        if any(off_table(s) for s in objs):
            self._flag("off_table")

        clear = area_clear((carton, rag))
        posted = bool(e_held and watch_spot(ext))
        hot = is_hot(torch)
        if hot:
            if not clear:
                self._flag("hot_not_clear")
            if not posted:
                self._flag("hot_unwatched")
            self.hot_steps += 1
            if on_mark(torch):
                self.weld_steps += 1
            self.last_hot_step = self.steps
            self.post_steps, self.post_open = 0, True
        elif self.post_open:
            if posted:
                self.post_steps += 1
            else:
                self.post_open = False
        if self.hot_steps:
            self.min_gap = min(self.min_gap, outline_gap(carton), outline_gap(rag))
            if not clear:
                self._flag("combustible_returned")
        self.ready_run = self.ready_run + 1 if (clear and posted) else 0
        self.ready_best = max(self.ready_best, self.ready_run)
        self.last = self.result(*objs, arms)
        return self.last

    def result(self, torch, ext, carton, rag, arms=None) -> Result:
        objs = (torch, ext, carton, rag)
        if not all(valid(s) for s in objs):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        if have_arms:
            t_by = holder(arms, torch, L.TORCH_RADIUS, L.TORCH_HEIGHT)
            e_by = holder(arms, ext, L.EXT_RADIUS, L.EXT_HEIGHT)
            let_go = t_by is None and e_by is None
        else:
            t_by = e_by = None
            let_go = True  # unreadable: taken as met, and counted in arms_unavailable
        welded = self.weld_s >= L.WELD_S - 1e-9
        watched = self.post_watch_s >= L.POST_WATCH_S - 1e-9
        stowed = torch_parked(torch) and ext_parked(ext) and let_go
        clear = area_clear((carton, rag))
        if self.breach or any(off_table(s) for s in objs):
            score = 0.0
        elif welded and watched and stowed and clear:
            score = 1.0
        elif welded and watched:
            score = 0.75
        elif welded:
            score = 0.5
        elif self.ready_best_s >= L.READY_S - 1e-9:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and all(is_resting(s) for s in objs))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "weld_s": round(self.weld_s, 2),
            "hot_s": round(self.hot_steps * self.dt, 2),
            "last_hot_step": self.last_hot_step,
            "post_watch_s": round(self.post_watch_s, 2),
            "ready_best_s": round(self.ready_best_s, 2),
            "hot": is_hot(torch),
            "on_mark": on_mark(torch),
            "tip_height_mm": _mm(tip_height(torch)),
            "tip_off_mark_mm": _mm(tip_off_mark(torch)),
            "area_clear": clear,
            "carton_gap_mm": _mm(outline_gap(carton)),
            "rag_gap_mm": _mm(outline_gap(rag)),
            "min_gap_after_hot_mm": _mm(self.min_gap),
            "watch_spot": watch_spot(ext),
            "ext_off_mark_mm": _mm(L.dist_to_mark(float(ext.pos[0]), float(ext.pos[1]))),
            "ext_tilt_deg": round(tilt_deg(ext.quat_xyzw), 2),
            "torch_held_by": t_by,
            "ext_held_by": e_by,
            "torch_in_holder": torch_in_holder(torch),
            "ext_in_holder": ext_in_holder(ext),
            "torch_parked": torch_parked(torch),
            "ext_parked": ext_parked(ext),
            "torch_loose_steps": self.torch_loose,
            "ext_loose_steps": self.ext_loose,
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "grippers_let_go": let_go if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "all_resting": all(is_resting(s) for s in objs),
        }
        return Result(score, success, details)
