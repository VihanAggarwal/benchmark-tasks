"""The cut rule and the scorer, as plain functions of the parts' states and the arms' state. NumPy only.

Two pieces, both fed one observation per control step:

``Cutter.update(sharpener, pencil)``: the cut. The pencil is PRESSED on the cutting front when its end has gone
through both bore rings (within 1.5 mm of the bore axis, past the carriage ring), is within 1 mm of the front's disc
and pushes the disc's feeler in at least 2 mm (2 N at 1,000 N/m). The crank's angle is integrated unwrapped; GAIN is
the forward travel past the furthest angle reached within the last turn (the furthest angle is never more than one
turn ahead of the crank, so a backlog longer than a turn is forgiven after one turn). While pressed, the cut depth D
grows by 1.5 mm per turn of gain, to 24 mm (the point complete: the pencil stop), and never shrinks. In the
simulator task.py runs this from a mechkit ``on_step`` rule (before the physics) and drives the cutting slide to D;
offline it runs from ``success()``. ``crank_target`` turns the cutting drag into the crank drive's target: a
forward-only friction of 0.15 N m (full above 0.5 rad/s, less below) while pressed and the point is unfinished.

``Tracker.update(states, arms, cutter)`` -> Result(score, success, details), judged every step:

    0.25  D reached 6 mm (four pressed forward turns)
    0.5   D reached 12 mm
    0.75  D reached 24 mm: the point is complete
    1.0   D is 24 mm, and the pencil is at least 10 mm clear of the sharpener's outer box; its axis lies in both
          notches of its rest (within 3 mm sideways and 2 mm in height of the notch seat at both V blocks); both
          grippers released it (every finger zone 30 mm or more from it) on every step of the last 1 s; and it
          moved 2 mm or less over that 1 s. An unreadable arm or a distance that cannot be computed is NOT
          released (rigstate's fail-closed rule).
    0     otherwise (doing nothing; cranking with no pressed pencil; pushing with no crank; a step with a broken state)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TWO_PI = 2.0 * math.pi


def wrap_to_pi(a: float) -> float:
    return math.remainder(float(a), TWO_PI)


def _finite(*values) -> bool:
    try:
        return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)
    except (TypeError, ValueError):
        return False


def valid(s) -> bool:
    """A part state the scorer can use: present with a finite pose (a blow-up gives NaN)."""
    try:
        return _finite(s.origin, s.quat_xyzw, s.pos)
    except AttributeError:
        return False


# -- the pencil's geometry ----------------------------------------------------------------------------------------


def pencil_ends(pencil) -> tuple[np.ndarray, np.ndarray]:
    """(the end that is in or toward the sharpener: the one with the smaller y, the other end), world points on the
    pencil's axis."""
    a = np.asarray(pencil.to_world((0.0, L.HALF, 0.0)), dtype=float)
    b = np.asarray(pencil.to_world((0.0, -L.HALF, 0.0)), dtype=float)
    return (a, b) if a[1] < b[1] else (b, a)


def axis_at_y(pencil, y: float) -> np.ndarray | None:
    """The point of the pencil's axis at world ``y``, or None when the axis does not reach that y (or runs across)."""
    a, b = pencil_ends(pencil)
    dy = float(b[1] - a[1])
    if dy < 1e-6:
        return None
    t = (float(y) - float(a[1])) / dy
    if not 0.0 <= t <= 1.0:
        return None
    return a + t * (b - a)


def box_distance(points, lo, hi) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    d = np.maximum(np.maximum(np.asarray(lo) - pts, pts - np.asarray(hi)), 0.0)
    return np.linalg.norm(d, axis=1)


_RING = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)])


def pencil_surface_points(pencil) -> np.ndarray:
    """Points on the pencil's surface (world): its side at 1 mm along the axis and 24 places round it, and both end
    faces at three radii. A flat face's distance from them is within 0.05 mm of the true distance."""
    a, b = pencil_ends(pencil)
    u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
    helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    ring = _RING[:, :1] * e1 + _RING[:, 1:] * e2  # (24, 3) unit vectors round the axis
    t = np.linspace(0.0, 1.0, int(L.PENCIL_LEN * 1000) + 1)[:, None]
    axis = a + t * (b - a)
    side = (axis[:, None, :] + L.PENCIL_R * ring[None, :, :]).reshape(-1, 3)
    caps = [end + f * L.PENCIL_R * ring for end in (a, b) for f in (0.0, 0.5, 1.0)]
    return np.vstack([side, *caps])


def clearance_to_sharpener(pencil) -> float:
    """The pencil surface's distance to the sharpener's outer box (m; 0 when touching or inside), from points on
    its side and end faces."""
    pts = pencil_surface_points(pencil)
    lo, hi = np.array(L.SHARPENER_BOX[0], dtype=float), np.array(L.SHARPENER_BOX[1], dtype=float)
    lo[2] += L.TOP
    hi[2] += L.TOP
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    if np.any(inside):
        return 0.0
    return float(np.min(box_distance(pts, lo, hi)))


# -- the cut ------------------------------------------------------------------------------------------------------


def front_face_y(qs: float, qf: float) -> float:
    """The cutting front's face, base-frame y (the base frame is not turned, so world y too)."""
    return L.FRONT_Y0 - float(qs) - float(qf)


class Cutter:
    """The cut depth D and what decided it this step."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.depth = 0.0  # D, m
        self.th_prev: float | None = None
        self.th_un = 0.0  # the crank's angle, unwrapped (rad, forward positive)
        self.th_max = 0.0  # the furthest angle reached, never more than one turn ahead of th_un
        self.pressed = False
        self.drag = 0.0
        self.gain = 0.0
        self.turns_pressed = 0.0  # pressed forward turns that cut (for the trace)
        self.info: dict = {}
        self.steps = 0

    def rebase(self, th_now: float) -> None:
        """The simulator's crank angle was rewritten by a whole turn (the wrap): the next reading is relative to it."""
        self.th_prev = float(th_now)

    def update(self, sharpener, pencil) -> dict:
        """One control step. ``sharpener`` carries the joints slide, front, crank (and their velocities); ``pencil``
        its pose. Returns what it saw and decided; a broken state changes nothing."""
        self.steps += 1
        try:
            qs = float(sharpener.joints["slide"])
            qf = float(sharpener.joints["front"])
            th = float(sharpener.joints["crank"])
            w = float(sharpener.joint_vel.get("crank", 0.0))
            if not (_finite(qs, qf, th, w) and valid(pencil) and valid(sharpener)):
                raise ValueError("non-finite state")
            end, _ = pencil_ends(pencil)
            e = np.asarray(sharpener.to_local(end), dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.pressed, self.drag, self.gain = False, 0.0, 0.0
            self.info = {"valid": False, "D_mm": round(self.depth * 1000, 3)}
            return self.info
        dth = 0.0 if self.th_prev is None else wrap_to_pi(th - self.th_prev)
        self.th_prev = th
        self.th_un += dth
        self.th_max = min(self.th_max, self.th_un + TWO_PI)  # a backlog of at most one turn
        gain = max(0.0, self.th_un - self.th_max)  # forward travel past the furthest angle within the last turn
        self.th_max = max(self.th_max, self.th_un)
        face = front_face_y(qs, qf)
        off_axis = math.hypot(float(e[0]) - L.AXIS_X, float(e[2]) - L.AXIS_Z)
        on_axis = off_axis < L.ON_AXIS and float(e[1]) < L.PAST_RINGS_Y
        touch = float(e[1]) - face
        pressed = bool(on_axis and touch < L.TOUCH and qf >= L.PRESS - L.EPS)
        if pressed and gain > 0.0 and self.depth < L.CUT_DEPTH:
            add = L.CUT_PER_TURN * gain / TWO_PI
            self.depth = min(L.CUT_DEPTH, self.depth + add)
            self.turns_pressed += gain / TWO_PI
        cutting = pressed and w > 0.0 and self.depth < L.CUT_DEPTH - L.EPS
        self.drag = L.T_CUT * min(1.0, w / L.W_SAT) if cutting else 0.0
        self.pressed, self.gain = pressed, gain
        self.info = {"valid": True, "D_mm": round(self.depth * 1000, 3), "pressed": pressed, "on_axis": on_axis,
                     "off_axis_mm": round(off_axis * 1000, 2), "end_y_mm": round(float(e[1]) * 1000, 2),
                     "touch_mm": round(touch * 1000, 2), "slide_mm": round(qs * 1000, 3),
                     "feeler_mm": round(qf * 1000, 3), "crank_rad": round(th, 4), "crank_w": round(w, 3),
                     "turns": round(self.th_un / TWO_PI, 3), "gain_rad": round(gain, 4),
                     "turns_cut": round(self.turns_pressed, 3), "drag_nm": round(self.drag, 4)}
        return self.info


def crank_target(th: float, w: float, drag: float) -> float:
    """The crank drive's target (rad) for one control step: torque = CRANK_K (target - q) - CRANK_DAMP q'. Leading
    the crank by w DT / 2 makes the spring's torque average zero over the step at steady speed, the CRANK_DAMP w /
    CRANK_K term cancels the drive's damping on average, and -drag / CRANK_K leaves a mean resisting torque of
    ``drag``. With drag 0 the free crank feels only its bearing friction. Kept inside the joint's limits."""
    t = float(th) + float(w) * L.DT / 2 + (L.CRANK_DAMP * float(w) - float(drag)) / L.CRANK_K
    return max(-L.CRANK_LIMIT, min(L.CRANK_LIMIT, t))


def tier_of_depth(depth: float) -> float:
    """The stage of the cut: 0.75 at the full 24 mm, 0.5 from 12 mm, 0.25 from 6 mm, else 0."""
    for tier, d in L.TIER_DEPTHS:
        if depth >= d - L.EPS:
            return tier
    return 0.0


def wrap_shift(q: float) -> float:
    """The whole turn (rad) that brings a hinge angle beyond +-pi back inside it, else 0: the rule rewrites the
    crank's and the knob's angles by it (a whole turn leaves every part in the same pose)."""
    q = float(q)
    if q > L.WRAP_AT:
        return -TWO_PI
    if q < -L.WRAP_AT:
        return TWO_PI
    return 0.0


# -- the scorer ---------------------------------------------------------------------------------------------------


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


def on_rest(pencil) -> tuple[bool, list]:
    """Whether the pencil's axis lies in both notches: at each V block's y, within 3 mm of the notch's x and 2 mm of
    its seat height. Also returns the offsets (mm) per block, None where the axis does not reach that block."""
    ok, offs = True, []
    for y in L.V_BLOCK_YS:
        p = axis_at_y(pencil, y)
        if p is None:
            ok = False
            offs.append(None)
            continue
        dx = float(p[0]) - L.REST_XY[0]
        dz = float(p[2]) - (L.TOP + L.SEAT_Z)
        offs.append([_mm(dx), _mm(dz)])
        if abs(dx) > L.SEAT_SIDE_TOL + L.EPS or abs(dz) > L.SEAT_HEIGHT_TOL + L.EPS:
            ok = False
    return ok, offs


class Tracker:
    """The last 1 s of the pencil's centre (stillness) and of the release check."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0  # fallback resets seen (the pencil teleported back to its start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)  # pencil centre per valid step
        self.released: deque = deque(maxlen=self.window)
        self.prev = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(centre) -> bool:
        start = np.array([L.PENCIL_START[0], L.PENCIL_START[1], L.TOP + L.PENCIL_START[2]])
        return float(np.linalg.norm(np.asarray(centre) - start)) <= L.RESET_START_TOL

    def teleported(self, centre) -> bool:
        return (self.prev is not None and self.at_start(centre)
                and float(np.linalg.norm(np.asarray(centre) - self.prev)) >= L.RESET_JUMP)

    def update(self, states: dict, arms, cutter: Cutter) -> Result:
        """One control step: record the pencil's centre and the release check, then judge."""
        self.steps += 1
        pencil = states.get("pencil")
        if pencil is None or not valid(pencil) or states.get("sharpener") is None:
            self.bad_steps += 1  # a blow-up or a missing part: 0, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "D_mm": round(cutter.depth * 1000, 3)})
            return self.last
        centre = np.asarray(pencil.origin, dtype=float).copy()
        if self.teleported(centre):
            self.reset()
            cutter.reset()  # a new trial the reset hooks missed: the cut starts again too
            self.steps = 1
            self.resets += 1
        self.hist.append(centre)
        self.prev = centre
        try:
            dist = pencil.distance
        except AttributeError:  # no distance: rigstate's rule, not released
            dist = None
        self.released.append(dist is not None and rigstate.released(arms, dist, L.LET_GO))
        self.last = self.judge(pencil, arms, cutter)
        return self.last

    def path(self) -> float | None:
        """The pencil centre's path over the last 1 s (m), or None before 1 s of valid states."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return sum(float(np.linalg.norm(b - a)) for a, b in zip(h, h[1:]))

    def judge(self, pencil, arms, cutter: Cutter) -> Result:
        depth = cutter.depth
        level = tier_of_depth(depth)
        clear = clearance_to_sharpener(pencil)
        is_clear = clear >= L.CLEAR - L.EPS
        resting, offs = on_rest(pencil)
        released_now = bool(self.released and self.released[-1])
        released_1s = len(self.released) == self.window and all(self.released)
        path = self.path()
        still = path is not None and path <= L.STILL_PATH + L.EPS
        top = level == 0.75 and is_clear and resting and released_1s and still
        score = 1.0 if top else level
        gap = None
        try:
            gap = rigstate.hands_gap(arms, pencil.distance)
        except Exception:  # noqa: BLE001 - trace only
            gap = None
        details = {"valid_state": True, "D_mm": round(depth * 1000, 3), "tier": level, "score": score,
                   "clear_mm": _mm(clear), "clear": bool(is_clear), "on_rest": bool(resting), "notch_mm": offs,
                   "released": released_now, "released_1s": bool(released_1s), "path_1s_mm": _mm(path),
                   "still_1s": bool(still), "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms)}
        return Result(score, bool(score == 1.0), details)
