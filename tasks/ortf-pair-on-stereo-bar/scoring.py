"""The scorer, as plain functions of the two microphones' states and the grippers' state. NumPy only; no simulator.

The rule is about the finished pair, read from the state at the last control step, as a recording engineer checks an
ORTF pair with a ruler and a protractor: are the two capsule points 170 mm apart, are the two capsule axes 110 degrees
apart and splayed outward (the capsules farther apart than the tails), is the pair centred on the source axis and does
its centre line point at the centre of the choir (the white mark on the riser). Every score above 0 also needs both
mics let go (every gripper's finger zone at least 10 mm from each mic), still, lying flat on the bar plate itself
(not on the other mic, not on the table) with their centres over it.

    Tracker.update(mics, arms) -> Result(score, success, details)

``mics`` is {"mic_a": state, "mic_b": state}, each as mechkit reports it (``AssemblyState``): ``origin`` (its base
frame's origin: the middle of its bottom face), ``quat_xyzw`` (that frame's orientation: local +x along the mic toward
its grille end, local +z up) and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. The two mics are alike:
either may be either one of the pair, and every quantity below is the same with the two swapped. A state with a
non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up, or a mic not built as an assembly) scores
0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the let-go condition is
taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so a
gripper's fingers are taken to be anywhere round its axis: its finger zone is every point within a radius of the
38 mm segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer corner (half
the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).

'Still' is judged from positions, not the reported velocity (which chatters while a part is gripped): in each of the
last two 0.5 s windows the mic's centre travelled under 1.5 mm, and its heading changed less than 2 degrees over the
last 1 s.

Ordered decision (first match wins): gate not met: 0; A and S and C: 1.0; A and S: 0.67; A: 0.33; otherwise 0.
A: included angle 110 +- 8 degrees, splayed outward, bisector within 30 degrees of the direction from the capsules'
midpoint to the mark. S: capsule separation 170 +- 10 mm. C: the midpoint within 10 mm of y = 0 and the bisector
within 5 degrees of the direction to the mark.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_EPS = 1e-9
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
_BOX_C = np.array(L.CENTER_LOCAL, dtype=float)
_BOX_H = np.array(L.MIC_HALF, dtype=float)
SIDES = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world.
    (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the right sizes, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos", "lin_vel")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool([p.size for p in parts] == [3, 4, 3, 3] and np.linalg.norm(parts[1]) > 0.5)


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


# -- one mic -----------------------------------------------------------------------------------------------------
def to_world(s, local) -> np.ndarray:
    return np.asarray(s.origin, dtype=float) + rotation(s.quat_xyzw) @ np.asarray(local, dtype=float)


def centre(s) -> np.ndarray:
    """The mic's centre (world)."""
    return to_world(s, L.CENTER_LOCAL)


def capsule(s) -> np.ndarray:
    """The capsule point: the grille's centre, 7.5 mm behind the front face along the mic's axis (world)."""
    return to_world(s, L.CAPSULE_LOCAL)


def axis(s) -> np.ndarray:
    """The capsule axis u: the unit vector from the tail toward the grille end (world, 3D)."""
    return rotation(s.quat_xyzw)[:, 0]


def plan_axis(s) -> np.ndarray | None:
    """u seen from above, as a unit 2D vector (None if the mic stands on an end)."""
    a = axis(s)[:2]
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-6 else None


def heading_deg(s) -> float:
    """Which way the grille end points seen from above, counter-clockwise from +x (degrees, -180..180)."""
    a = axis(s)
    return float(math.degrees(math.atan2(a[1], a[0])))


def tilt_deg(s) -> float:
    """Angle between the mic's long axis and the horizontal (degrees, 0..90): 0 lying flat, 90 standing on an end."""
    return float(math.degrees(math.asin(min(1.0, abs(float(axis(s)[2]))))))


def corners(s) -> np.ndarray:
    """The mic's eight corners (world, 8 x 3)."""
    r = rotation(s.quat_xyzw)
    return np.asarray(s.origin, dtype=float) + ((_SIGNS * _BOX_H) + _BOX_C) @ r.T


def bottom_gap(s) -> float:
    """Height of the mic's lowest point above the plate top (m; negative = below it)."""
    return float(corners(s)[:, 2].min() - L.PLATE_TOP_W)


def over_plate(s) -> bool:
    """The mic's centre, seen from above, is over the bar plate."""
    c = centre(s)
    return bool(L.PLATE_X0 - _EPS <= c[0] <= L.PLATE_X1 + _EPS and L.PLATE_Y0 - _EPS <= c[1] <= L.PLATE_Y1 + _EPS)


def point_distance(p, s) -> float:
    """Distance from a point to the mic's box (0 inside it), in the mic's own frame."""
    q = rotation(s.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(s.origin, dtype=float))
    d = np.abs(q - _BOX_C) - _BOX_H
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, s, samples: int = 20) -> float:
    """How far a gripper's finger zone is from the mic (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return min(point_distance(p, s) for p in pts) - zone_radius(arm)


def mic_gate(s, still: bool, hands_off: bool) -> tuple[bool, str, dict]:
    """(gate met for this mic, why not, diagnostics): let go, still, long axis within 10 degrees of horizontal, its
    centre over the plate and its lowest point within 5 mm of the plate top."""
    c = centre(s)
    d = {"centre_mm": [round(float(v) * 1000, 1) for v in c], "yaw_deg": round(heading_deg(s), 2),
         "tilt_deg": round(tilt_deg(s), 2), "bottom_mm": round(bottom_gap(s) * 1000, 2),
         "over_plate": over_plate(s), "still": bool(still), "hands_off": bool(hands_off)}
    if not hands_off:
        return False, "held_or_touched", d
    if not still:
        return False, "still_moving", d
    if tilt_deg(s) > L.FLAT_DEG + _EPS:
        return False, "not_lying_flat", d
    if not d["over_plate"]:
        return False, "not_over_the_plate", d
    if abs(bottom_gap(s)) > L.ON_PLATE_TOL + _EPS:
        return False, "not_resting_on_the_plate", d
    return True, "", d


# -- the pair ----------------------------------------------------------------------------------------------------
def _angle_deg(a, b) -> float:
    """Angle between two 2D vectors (degrees, 0..180)."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))))


def pair(sa, sb) -> dict:
    """The pair's measures, seen from above (the same with the two mics swapped)."""
    ca, cb = capsule(sa)[:2], capsule(sb)[:2]
    ua, ub = plan_axis(sa), plan_axis(sb)
    mid = (ca + cb) / 2
    sep = float(np.linalg.norm(ca - cb))
    to_mark = np.asarray(L.MARK, dtype=float) - mid
    out = {"separation_mm": round(sep * 1000, 2), "mid_mm": [round(float(v) * 1000, 2) for v in mid],
           "mid_off_axis_mm": round(float(mid[1] - L.SOURCE_AXIS_Y) * 1000, 2), "included_deg": None,
           "splay": None, "splay_outward": False, "bisector_deg": None, "facing_err_deg": None}
    if ua is None or ub is None:
        out.update(A=False, S=False, C=False, why_not_a="a_mic_standing_on_an_end")
        return out
    included = _angle_deg(ua, ub)
    splay = float(np.dot(ca - cb, ua - ub))  # > 0: the capsules are farther apart than the tails
    bis = ua + ub
    nb = float(np.linalg.norm(bis))
    if nb > 1e-9:
        facing = _angle_deg(bis, to_mark)
        bis_deg = round(math.degrees(math.atan2(bis[1], bis[0])), 2)
    else:  # axes opposite: no centre line
        facing, bis_deg = 180.0, None
    a_ok = bool(abs(included - L.ANGLE_DEG) <= L.ANGLE_TOL_DEG + _EPS and splay > 0.0
                and facing <= L.FACING_DEG + _EPS)
    s_ok = bool(abs(sep - L.SPACING) <= L.SPACING_TOL + _EPS)
    c_ok = bool(abs(float(mid[1]) - L.SOURCE_AXIS_Y) <= L.CENTRE_TOL + _EPS and facing <= L.AIM_DEG + _EPS)
    why_not_a = None
    if abs(included - L.ANGLE_DEG) > L.ANGLE_TOL_DEG + _EPS:
        why_not_a = "included_angle_not_110"
    elif splay <= 0.0:
        why_not_a = "crossed_not_splayed_outward"
    elif facing > L.FACING_DEG + _EPS:
        why_not_a = "not_facing_the_choir"
    out.update(included_deg=round(included, 2), splay=round(splay * 1e6, 1), splay_outward=splay > 0.0,
               bisector_deg=bis_deg, facing_err_deg=round(facing, 2), A=a_ok, S=s_ok, C=c_ok,
               why_not_a=why_not_a)
    return out


def tier(p: dict) -> float:
    """The ordered list, after the gate: A and S and C 1.0, A and S 0.67, A 0.33, otherwise 0."""
    if p["A"] and p["S"] and p["C"]:
        return 1.0
    if p["A"] and p["S"]:
        return 0.67
    if p["A"]:
        return 0.33
    return 0.0


def evaluate(mics: dict, still: dict, hands_off: dict) -> tuple[float, dict]:
    """The score of a state (both mics valid) and the details for the trace."""
    gates, reasons, per = {}, {}, {}
    for n in L.MICS:
        ok, why, d = mic_gate(mics[n], still.get(n, False), hands_off.get(n, True))
        gates[n], reasons[n], per[n] = ok, why, d
    p = pair(mics[L.MICS[0]], mics[L.MICS[1]])
    details = {"mics": per, "pair": p, "gate": all(gates.values())}
    if not details["gate"]:
        first = next(n for n in L.MICS if not gates[n])
        return 0.0, {**details, "zero_reason": f"{first}:{reasons[first]}"}
    score = tier(p)
    return score, {**details, "zero_reason": None if score > 0 else p["why_not_a"]}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Tracker:
    """What the scorer remembers within a trial: each mic's centre and heading over the last second, how many steps
    were broken or had the arms unreadable, and when each score was first reached."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.history: dict[str, deque] = {n: deque(maxlen=L.STILL_WINDOWS * self.window + 1) for n in L.MICS}
        self.prev = None  # {mic: (centre, velocity)} at the previous valid step
        self.first_at: dict[float, int] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(mics: dict) -> bool:
        """The starting layout: each mic lying still in a case pad, on a starting mark (either mic on either mark),
        its grille end pointing the way that mark's mic does."""
        free = dict(L.MIC_START)
        for s in mics.values():
            c = centre(s)
            hit = None
            for mark, (sx, sy, syaw) in free.items():
                if (math.hypot(float(c[0]) - sx, float(c[1]) - sy) < L.START_TOL
                        and abs(_wrap(heading_deg(s) - syaw)) < L.START_YAW_TOL_DEG):
                    hit = mark
                    break
            if hit is None:
                return False
            free.pop(hit)
            lying = abs(float(corners(s)[:, 2].min()) - (L.TOP + L.PAD_T)) < 0.003 and tilt_deg(s) < 2.0
            slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.05
            if not (lying and slow):
                return False
        return True

    def _teleported(self, now: dict) -> bool:
        """Some mic moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could have
        carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for n, (p0, v0) in self.prev.items() if n in now
        )

    def motion(self, name: str) -> tuple[list[float] | None, float | None]:
        """The mic centre's path (m) in each of the last two 0.5 s windows and its heading's range (degrees) over the
        last 1 s; (None, None) until 1 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < h.maxlen:
            return None, None
        pts = [p for p, _ in h]
        paths = []
        for k in range(L.STILL_WINDOWS):
            seg = pts[k * self.window:(k + 1) * self.window + 1]
            paths.append(float(sum(np.linalg.norm(b - a) for a, b in zip(seg, seg[1:]))))
        yaws = [y for _, y in h]
        run, lo, hi = 0.0, 0.0, 0.0
        for a, b in zip(yaws, yaws[1:]):
            run += _wrap(b - a)
            lo, hi = min(lo, run), max(hi, run)
        return paths, float(hi - lo)

    def is_still(self, name: str) -> bool:
        paths, yaw_range = self.motion(name)
        return bool(paths is not None and max(paths) < L.STILL_PATH and yaw_range < L.STILL_YAW_DEG)

    def update(self, mics: dict, arms=None) -> Result:
        self.steps += 1
        if not all(valid(mics.get(n)) for n in L.MICS):  # a blow-up: score 0 for this step, keep the record
            self.bad_steps += 1
            self.last = self.result(mics, arms)
            return self.last
        now = {n: (centre(mics[n]), np.asarray(mics[n].lin_vel, dtype=float)) for n in L.MICS}
        if self.prev is not None and self._teleported(now) and self.at_start(mics):
            bad = self.bad_steps
            self.reset()  # the simulator put both mics back on their marks: a new episode
            self.steps, self.bad_steps = 1, bad
        self.prev = now
        for n in L.MICS:
            self.history[n].append((now[n][0], heading_deg(mics[n])))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(mics, arms)
        for t in L.TIERS:
            if self.last.score >= t - _EPS and t not in self.first_at:
                self.first_at[t] = self.steps
        return self.last

    def result(self, mics: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"step": self.steps, "bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = [n for n in L.MICS if not valid(mics.get(n))]
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_mics": bad, "zero_reason": "invalid_state",
                                       **trial})
        still = {n: self.is_still(n) for n in L.MICS}
        if arms_valid(arms):
            gaps = {n: min(zone_gap(arms[a], mics[n]) for a in SIDES) for n in L.MICS}
            hands = {n: gaps[n] >= L.HANDS_OFF for n in L.MICS}
            zone_mm = {n: round(gaps[n] * 1000, 1) for n in L.MICS}
        else:  # not readable: taken as met, and counted
            hands, zone_mm = {n: True for n in L.MICS}, None
        score, details = evaluate(mics, still, hands)
        motion = {}
        for n in L.MICS:
            paths, yr = self.motion(n)
            motion[n] = {"path_mm": None if paths is None else [round(p * 1000, 3) for p in paths],
                         "yaw_range_deg": None if yr is None else round(yr, 3)}
        details.update(zone_gap_mm=zone_mm, motion=motion)
        return Result(score, bool(score == 1.0), {"valid_state": True, **details, **trial})
