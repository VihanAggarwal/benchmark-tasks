"""The scorer, as plain functions of the two stop-ends' states and the grippers' state. NumPy only; no simulator
needed.

The rule is about where the construction joints will be: each girder gets one stop-end standing across its trough,
wholly inside the middle third of its span and at least twice the cross beam's width clear of the cross beam's face
(ACI 318). So the score is read from the end state; the scorer is fed one observation per control step by
``task.success()`` and keeps only what it needs for that (each stop-end's end points over the last 0.5 s, to tell
that both are still, and counters for the trace).

    Tracker.update(stops, arms) -> Result(score, success, details)

``stops`` maps each stop-end's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; each is a box of known size
(``layout.STOP_SIZE``), so its corners follow from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then the let-go condition is taken as met for that step
and the step is counted in ``arms_unavailable``. A step on which either stop-end has a non-finite number or a zero
quaternion (a simulator blow-up), or is missing, scores 0 and leaves the tracker as it was.

A stop-end is a plain box, the same either end up and either way round, so 'upright' is measured from whichever end
is nearer to facing down and 'square' only by the line of its length.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a finger's outer corner (half the current finger gap plus
the finger's 10 mm thickness, and half its 40 mm width across; 60 mm fully open, 22 mm closed).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
HALF = np.array(L.STOP_HALF, dtype=float)
ARMS = ("left", "right")
GIRDERS = {"left": 1.0, "right": -1.0}  # each girder's side of the midline


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
    if s is None:
        return False
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
        for side in ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- one stop-end ---------------------------------------------------------------------------------------------------
def corners(s) -> np.ndarray:
    """The stop-end's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * HALF) @ rotation(s.quat_xyzw).T


def lowest(s) -> float:
    """Height of the stop-end's lowest point above the table top (m)."""
    return float(corners(s)[:, 2].min() - L.TOP)


def tilt_deg(s) -> float:
    """Angle between the vertical and the stop-end's 60 mm side (degrees, 0..90): either end may be down."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def square_err_deg(s) -> float:
    """How far the stop-end's 40 mm length, seen from above, is turned from y, the line square across a girder
    (degrees, 0..90; either end may point either way). A length standing on end has no line seen from above: 90."""
    u = rotation(s.quat_xyzw)[:, 1]
    n = math.hypot(float(u[0]), float(u[1]))
    if n < 1e-9:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(abs(float(u[1])) / n, 0.0, 1.0))))


def ends(s) -> np.ndarray:
    """The centres of the stop-end's top and bottom faces (2 x 3)."""
    c, u = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)[:, 2] * HALF[2]
    return np.array([c + u, c - u])


def girder_of(s) -> str | None:
    """The girder whose trough the stop-end's centre is over, seen from above (between the columns' faces and the
    side forms' inside faces), at any height; None if over neither."""
    x, y = float(s.pos[0]), float(s.pos[1])
    if not (L.COL_FACE_X[0] <= x <= L.COL_FACE_X[1]):
        return None
    for g, sgn in GIRDERS.items():
        if L.TROUGH_IN <= sgn * y <= L.TROUGH_OUT:
            return g
    return None


def beam_clearance(xmin: float, xmax: float) -> float:
    """Distance along x from an outline spanning xmin..xmax to the cross beam's nearer side face (0 if it reaches
    the cross beam's width)."""
    if xmin >= L.BEAM_FAR_FACE:
        return xmin - L.BEAM_FAR_FACE
    if xmax <= L.BEAM_NEAR_FACE:
        return L.BEAM_NEAR_FACE - xmax
    return 0.0


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the gap plus the finger's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def zone_gap(arm, s) -> float:
    """How far a gripper's finger zone is from the stop-end (negative: they overlap)."""
    pts, r = finger_zone(arm)
    d = (np.asarray(pts, dtype=float) - np.asarray(s.pos, dtype=float)) @ rotation(s.quat_xyzw)
    return float(np.linalg.norm(np.maximum(np.abs(d) - HALF, 0.0), axis=1).min()) - r


# -- the rules -----------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


def stop_info(s, arms=None) -> dict:
    """Everything the rules ask of one stop-end, at this step. ``arms`` unreadable: let go is taken as met."""
    pts = corners(s)[:, :2]
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    low, tilt, sq = lowest(s), tilt_deg(s), square_err_deg(s)
    g = girder_of(s)
    in_trough = False
    if g is not None:
        sgn = GIRDERS[g]
        lo, hi = sorted((sgn * L.TROUGH_IN, sgn * L.TROUGH_OUT))
        in_trough = bool(xmin >= L.COL_FACE_X[0] - L.TROUGH_TOL - L.EPS and xmax <= L.COL_FACE_X[1] + L.TROUGH_TOL
                         + L.EPS and ymin >= lo - L.TROUGH_TOL - L.EPS and ymax <= hi + L.TROUGH_TOL + L.EPS)
    have_arms = arms_valid(arms)
    zones = {side: zone_gap(arms[side], s) for side in ARMS} if have_arms else None
    let_go = all(z >= L.HANDS_OFF - L.EPS for z in zones.values()) if have_arms else True
    upright = tilt <= L.UPRIGHT_DEG + 1e-9
    on_floor = low <= L.ON_FLOOR + L.EPS
    square = sq <= L.SQUARE_DEG + 1e-9
    stands = bool(g is not None and upright and on_floor and square and in_trough and let_go)
    middle_third = bool(xmin >= L.MID_LO - L.EPS and xmax <= L.MID_HI + L.EPS)
    clear = beam_clearance(xmin, xmax)
    clear_ok = clear >= L.OFFSET - L.EPS
    if stands and middle_third and clear_ok:
        score, why = L.FULL, "in the band"
    elif stands and middle_third:
        score, why = L.PART, "in the middle third, too near the cross beam"
    elif g is None:
        score, why = 0.0, "not over a girder"
    elif not stands:
        why = ("not upright" if not upright else "not on the floor" if not on_floor else "not square"
               if not square else "outside the trough" if not in_trough else "not let go")
        score = 0.0
    else:
        score, why = 0.0, "outside the middle third"
    return {"girder": g, "stands": stands, "score": score, "why": why, "xmin": xmin, "xmax": xmax, "ymin": ymin,
            "ymax": ymax, "low": low, "tilt": tilt, "square_err": sq, "in_trough": in_trough, "let_go": bool(let_go),
            "middle_third": middle_third, "beam_clear": clear, "zones": zones}


def evaluate(stops: dict, arms=None) -> tuple[float, dict]:
    """The score of the bay as it stands, and what it was read from. ``stops`` must all be valid."""
    infos = {n: stop_info(s, arms) for n, s in stops.items()}
    girders = {}
    for g in GIRDERS:
        members = [n for n in L.STOP_NAMES if n in infos and infos[n]["girder"] == g]
        girders[g] = min(infos[n]["score"] for n in members) if members else 0.0
    score = float(sum(girders.values()))
    have_arms = arms_valid(arms)
    details = {
        "girder_left": girders["left"],
        "girder_right": girders["right"],
        "arms_available": bool(have_arms),
    }
    for n, i in infos.items():
        details[n] = {
            "girder": i["girder"],
            "score": i["score"],
            "why": i["why"],
            "stands": i["stands"],
            "x_mm": _mm(float(stops[n].pos[0])),
            "y_mm": _mm(float(stops[n].pos[1])),
            "xmin_mm": _mm(i["xmin"]),
            "xmax_mm": _mm(i["xmax"]),
            "beam_clear_mm": _mm(i["beam_clear"]),
            "middle_third": i["middle_third"],
            "in_trough": i["in_trough"],
            "low_mm": _mm(i["low"]),
            "tilt_deg": round(i["tilt"], 2),
            "square_err_deg": round(i["square_err"], 2),
            "let_go": i["let_go"],
            "left_zone_mm": _mm(i["zones"]["left"]) if i["zones"] else None,
            "right_zone_mm": _mm(i["zones"]["right"]) if i["zones"] else None,
        }
    return score, details


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer keeps over a trial: each stop-end's end points over the last 0.5 s, and counters."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # steps with a stop-end missing or non-finite (scored 0, tracker untouched)
        self.arms_unavailable = 0  # steps on which rigstate could not read the arms
        self.history: dict[str, deque] = {}  # stop-end -> end points at this step and the 10 before it
        self.prev: dict[str, np.ndarray] | None = None  # each stop-end's centre at the previous valid step
        self.first_full_step: int | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(stops: dict) -> bool:
        """The starting layout: both stop-ends upright on the table on their marks (within 45 mm)."""
        for n, (x, y, _) in L.STOP_START.items():
            s = stops.get(n)
            if s is None or not valid(s):
                return False
            on_mark = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            if not (on_mark and abs(lowest(s)) < 0.003 and tilt_deg(s) < 2.0):
                return False
        return True

    def moved(self) -> dict[str, float | None]:
        """How far each stop-end's ends moved at most over the last 0.5 s (None until 0.5 s has been seen)."""
        out = {}
        for n in L.STOP_NAMES:
            h = self.history.get(n)
            if h is None or len(h) < self.window + 1:
                out[n] = None
                continue
            pts = np.array(h)  # (k, 2, 3)
            out[n] = float(np.linalg.norm(pts - pts[-1], axis=2).max())
        return out

    def update(self, stops: dict, arms=None) -> Result:
        if any(not valid(stops.get(n)) for n in L.STOP_NAMES):
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        centres = {n: np.asarray(stops[n].pos, dtype=float) for n in L.STOP_NAMES}
        if self.prev is not None and self.at_start(stops):
            jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in L.STOP_NAMES)
            if jump > L.RESET_JUMP:  # the simulator put the stop-ends back: a new trial
                self.reset()
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        for n in L.STOP_NAMES:
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(ends(stops[n]))
        self.prev = centres
        self.last = self.result(stops, arms)
        if self.first_full_step is None and self.last.score == 1.0:
            self.first_full_step = self.steps
        self.last.details.update(first_full_step=self.first_full_step)
        return self.last

    def result(self, stops: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (it changes nothing)."""
        if any(not valid(stops.get(n)) for n in L.STOP_NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        score, details = evaluate({n: stops[n] for n in L.STOP_NAMES}, arms)
        moved = self.moved()
        still = all(m is not None and m <= L.STILL_TOL + L.EPS for m in moved.values())
        details.update(valid_state=True, still=still, moved_mm={n: _mm(m) for n, m in moved.items()},
                       steps=self.steps, bad_steps=self.bad_steps, arms_unavailable=self.arms_unavailable)
        return Result(score, bool(score == 1.0 and still), details)
