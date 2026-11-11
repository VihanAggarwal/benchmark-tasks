"""The scorer, as plain functions of the two tyres' states and the grippers' state. NumPy only; no simulator needed.

The score is read from where the tyres end up, as a tyre fitter checks a match-mounted wheel: is a tyre down on each
wheel's turntable round the wheel, let go and at rest, and on each wheel is the right dot at the right mark. The rule
(tyre makers' match-mounting instructions): on a wheel with a low-point (minimum radial runout) mark, the tyre's red
dot (its uniformity high point) goes to that mark; on a wheel without one, the tyre's yellow dot (its light point)
goes to the valve stem. Either tyre may go on either wheel. The scorer keeps only the tyres' last second of centres
and headings (to tell a tyre at rest from one still moving) and is fed one observation per control step by
``task.success()``:

    Tracker.update(tyres, arms) -> Result(score, success, details)

``tyres`` maps each tyre's name to its state as mechkit reports it (``AssemblyState``): ``origin`` (its base frame's
origin: the centre of the ring's bottom face), ``quat_xyzw`` (that frame's orientation: local +z up through the
dotted sidewall, local +x toward the red dot) and the ``ObjectState`` fields. A step where either tyre's state is
missing, has a non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up) scores 0 and leaves the
tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then no tyre counts as let
go on that step (the rule cannot be checked) and the step is counted in ``arms_unavailable``. rigstate gives no
gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point
within reach of a pad's outer corner (half the current finger gap plus the 8 mm pad thickness, and half the fingers'
27 mm width across; 57 mm fully open) of the 38 mm segment from the point between its pad tips back up the gripper.

A dot's angle is the direction from the tyre's own centre to the dot's centre, seen from above (counter-clockwise from
+x): the tyre's turn, which is what match-mounting sets. A seated tyre sits within 5 mm of its wheel's axis; once a
real tyre is inflated its bead centres it, so the angle about the tyre's centre is the angle it keeps.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

TT_TOP_W = L.TOP + L.TT_TOP  # the turntables' tops (world z)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrap180(a: float) -> float:
    """An angle in degrees folded into (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def valid(t) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    if t is None:
        return False
    try:
        parts = [np.asarray(getattr(t, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def all_valid(tyres) -> bool:
    try:
        return all(valid(tyres.get(n)) for n in L.TYRE_NAMES)
    except AttributeError:
        return False


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- one tyre's pose ----------------------------------------------------------------------------------------------
def to_world(t, local) -> np.ndarray:
    return np.asarray(t.origin, dtype=float) + rotation(t.quat_xyzw) @ np.asarray(local, dtype=float)


def tilt_deg(t) -> float:
    """Angle between the tyre's axis (local +z, out of the dotted sidewall) and straight up: 0 lying flat dots up,
    180 lying flat upside down (dots down), 90 standing on its tread."""
    return float(np.degrees(np.arccos(np.clip(rotation(t.quat_xyzw)[2, 2], -1.0, 1.0))))


def heading_deg(t) -> float:
    """The tyre's turn seen from above: the direction of its local +x (toward the red dot), degrees from +x."""
    r = rotation(t.quat_xyzw)
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def dot_local(colour: str) -> tuple[float, float, float]:
    """The centre of a dot's top in the tyre's own frame."""
    a = L.RED_LOCAL_DEG if colour == "red" else L.YELLOW_LOCAL_DEG
    dx, dy = L.on_top(L.DOT_R, a)
    return dx, dy, L.TYRE_TOP_H


def dot_angle_deg(t, colour: str) -> float:
    """Direction from the tyre's centre to the dot's centre, seen from above, degrees counter-clockwise from +x."""
    c = to_world(t, (0.0, 0.0, L.TYRE_TOP_H))
    d = to_world(t, dot_local(colour)) - c
    return float(math.degrees(math.atan2(d[1], d[0])))


def nearest_wheel(t) -> tuple[str, float]:
    """(name, horizontal distance (m) from the tyre's centre to that wheel's axis)."""
    o = np.asarray(t.origin, dtype=float)
    best = min(L.WHEEL_NAMES, key=lambda w: math.hypot(o[0] - L.wheel_axis(w)[0], o[1] - L.wheel_axis(w)[1]))
    ax = L.wheel_axis(best)
    return best, float(math.hypot(o[0] - ax[0], o[1] - ax[1]))


def pose_seated(t) -> tuple[str | None, str, dict]:
    """(wheel it sits on, or None; why not; diagnostics) from the pose alone: down on that wheel's turntable round
    the wheel (underside within 3 mm of the turntable's top, centre within 6 mm of the axis), lying flat dots up
    (axis within 5 degrees of vertical)."""
    w, off = nearest_wheel(t)
    bottom = float(np.asarray(t.origin, dtype=float)[2] - TT_TOP_W)
    tilt = tilt_deg(t)
    d = {"wheel": w, "axis_offset_mm": round(off * 1000, 2), "bottom_mm": round(bottom * 1000, 2),
         "tilt_deg": round(tilt, 2), "heading_deg": round(heading_deg(t), 2),
         "red_deg": round(dot_angle_deg(t, "red"), 2), "yellow_deg": round(dot_angle_deg(t, "yellow"), 2)}
    if off > L.AXIS_TOL:
        return None, "not_round_a_wheel", d
    if tilt > L.TILT_MAX_DEG:
        return None, ("upside_down" if tilt > 90.0 else "tilted"), d
    if abs(bottom) > L.SEAT_TOL:
        return None, ("not_down_on_the_turntable" if bottom > 0 else "below_the_turntable"), d
    return w, "", d


def rule_error_deg(t, wheel: str) -> float:
    """How far (degrees, 0..180) the rule's dot is from the rule's mark on ``wheel``."""
    target, colour = L.TARGETS[wheel]
    return abs(wrap180(dot_angle_deg(t, colour) - target))


# -- the grippers ---------------------------------------------------------------------------------------------------
def point_distance(p, t) -> float:
    """Distance from a point to the tyre's ring (0 inside the rubber), in the tyre's own frame."""
    q = rotation(t.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(t.origin, dtype=float))
    rho = math.hypot(q[0], q[1])
    dr = max(L.TYRE_R_IN - rho, rho - L.TYRE_R_OUT, 0.0)
    dz = max(-q[2], q[2] - L.TYRE_TOP_H, 0.0)
    return float(math.hypot(dr, dz))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, t, samples: int = 20) -> float:
    """How far a gripper's finger zone is from the tyre (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return min(point_distance(p, t) for p in pts) - zone_radius(arm)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each tyre's centre and heading over the last second, and how many
    steps the arms could not be read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_STEPS * control_hz / L.CONTROL_HZ)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=2 * self.window + 1) for n in L.TYRE_NAMES}  # (centre, heading) per step
        self.prev_origin = {n: None for n in L.TYRE_NAMES}
        self.arms_unavailable = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, t) -> bool:
        """A tyre lying still on its starting spot, flat, dots up, at its starting turn."""
        x, y, yaw = L.TYRES[name]
        o = np.asarray(t.origin, dtype=float)
        on_spot = math.hypot(o[0] - x, o[1] - y) < L.START_TOL and abs(o[2] - L.TOP) < 0.003
        turned = abs(wrap180(heading_deg(t) - yaw)) < 5.0
        return bool(on_spot and turned and tilt_deg(t) < 2.0)

    def still(self, name: str) -> tuple[bool, dict]:
        """(at rest, diagnostics): in each of the last two 0.5 s windows the centre travelled at most 1.5 mm, and over
        the last 1 s the heading changed at most 2 degrees. False until 1 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < 2 * self.window + 1:
            return False, {"path_mm": None, "yaw_range_deg": None}
        cs = [c for c, _ in h]
        steps = [float(np.linalg.norm(b - a)) for a, b in zip(cs, cs[1:])]
        paths = [sum(steps[: self.window]), sum(steps[self.window:])]
        yaws = [y for _, y in h]
        spread = max(abs(wrap180(y - yaws[-1])) for y in yaws)
        ok = max(paths) <= L.STILL_PATH and spread <= L.STILL_YAW_DEG
        return bool(ok), {"path_mm": [round(p * 1000, 2) for p in paths], "yaw_range_deg": round(spread, 2)}

    def update(self, tyres, arms=None) -> Result:
        if not all_valid(tyres):  # a blow-up or a missing tyre: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(tyres, arms)
            return self.last
        jumped = False
        for n in L.TYRE_NAMES:
            o = np.asarray(tyres[n].origin, dtype=float)
            prev = self.prev_origin[n]
            if prev is not None and self.at_start(n, tyres[n]) and float(np.linalg.norm(o - prev)) > L.RESET_JUMP:
                jumped = True  # the simulator put it back: a new trial
        if jumped:
            self.reset()
        self.steps += 1
        for n in L.TYRE_NAMES:
            t = tyres[n]
            self.history[n].append((to_world(t, (0.0, 0.0, L.TYRE_H / 2)), heading_deg(t)))
            self.prev_origin[n] = np.asarray(t.origin, dtype=float)
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(tyres, arms)
        return self.last

    def result(self, tyres, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not all_valid(tyres):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        readable = arms_valid(arms)
        per: dict = {}
        on_wheel: dict = {w: [] for w in L.WHEEL_NAMES}
        for n in L.TYRE_NAMES:
            t = tyres[n]
            wheel, why, d = pose_seated(t)
            at_rest, sd = self.still(n)
            if readable:
                zl, zr = zone_gap(arms["left"], t), zone_gap(arms["right"], t)
                let_go = min(zl, zr) >= L.HANDS_OFF
                zones = [round(zl * 1000, 1), round(zr * 1000, 1)]
            else:  # not readable: the let-go rule cannot be checked, so not let go
                let_go, zones = False, None
            if wheel is not None and not let_go:
                why = "gripper_on_or_near_it" if readable else "arms_unreadable"
            elif wheel is not None and not at_rest:
                why = "still_moving"
            seated = wheel if (wheel is not None and let_go and at_rest) else None
            if seated is not None:
                on_wheel[seated].append(n)
            target, colour = L.TARGETS[d["wheel"]]
            d.update(seated_on=seated, why_not_seated=why, let_go=let_go, at_rest=at_rest, zone_gap_mm=zones,
                     rule_dot=colour, rule_target_deg=target, rule_error_deg=round(rule_error_deg(t, d["wheel"]), 2),
                     **sd)
            per[n] = d
        correct = {}
        for w in L.WHEEL_NAMES:
            ts = on_wheel[w]
            # one tyre on a wheel, its rule dot within 10 degrees of the rule mark (two rings cannot both be down
            # on one turntable; if a state ever says so, that wheel is not correct)
            correct[w] = bool(len(ts) == 1 and rule_error_deg(tyres[ts[0]], w) <= L.ANGLE_TOL_DEG)
        n_correct = sum(correct.values())
        n_seated = sum(len(v) for v in on_wheel.values())
        if n_correct == 2:
            score = 1.0
        elif n_correct == 1:
            score = 0.5
        elif n_seated >= 1:
            score = 0.25
        else:
            score = 0.0
        details = {"valid_state": True, "n_seated": n_seated, "n_correct": n_correct, "correct": correct,
                   "on_wheel": on_wheel, "arms_unavailable": self.arms_unavailable, "tyres": per}
        return Result(score, bool(score == 1.0), details)
