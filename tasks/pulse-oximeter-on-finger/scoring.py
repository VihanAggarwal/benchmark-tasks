"""The scorer, as plain functions of the oximeter's state and the arms' state. NumPy only.

Every control step ``Tracker.update(states, arms)`` reads the oximeter's two shells (its base pose and the pivot
angle ``q`` of its ``jaw`` link), the fixed fingertip P and both grippers (``rigstate``), and returns
Result(score, success, details).

Readings, in the centre-line frame (origin at the pivot; ``m`` = the base frame's X turned by q/2 about its Z, so it
points at the mouth along the centre line; ``n`` = its Y turned the same way, toward the display shell):

- depth: how far the fingertip is past the line joining the two lips, along ``m`` (mm). On the stop it is 31.0 mm
  shut, 30.0 mm with the cushions on the finger (16.3 deg) and 28.9 mm with the jaws fully open (28 deg).
- in the chamber: depth 4 to 32.5 mm (between the lips and the stop) and the fingertip's centre within 5 mm of the
  centre line (both across the shells, ``v``, and along the pivot, ``w``).
- over the finger: in the chamber at depth 10 mm or more, ``m`` within 20 deg of the finger's axis (+x) and ``n``
  within 45 deg of the nail side (+y).
- jaws open: q of 22 deg or more.
- slid on open: a record of the way in. Emptied at a reset and on every step with depth below 0 (the fingertip right
  out past the lips: a fresh try). On every step in the chamber: with the jaws open d_open = max(d_open, depth), else
  d_shut = max(d_shut, depth). Slid on open = d_open is set and d_shut is empty or at most d_open + 1.5 mm
  (closing on the stop adds 30.0 - 28.9 = 1.1 mm by itself; 0.4 mm is the margin).
- gripping: over the finger with q within 1.8 deg of 16.3 deg (both cushions on the 15 mm finger).
- let go: on every step of the last 1 s, ``rigstate.released`` with clearance 30 mm against the oximeter's exact
  shapes (both shells): every finger zone of both grippers 30 mm or more from it. Unreadable arms, or a distance that
  cannot be computed, are not let go (fail closed).
- clipped on: let go; no part of the oximeter within 20 mm of the table; and through the last 1 s gripping on every
  step, the base moved under 2 mm and turned under 2 deg (from the window's first pose).

Records (cleared only by a reset): opened in the air (jaws open with every part of the oximeter 20 mm or more above
the table for 10 consecutive steps, 0.5 s) and pushed home (over the finger at depth 28 mm or more for 10 consecutive
steps). Only the first is credited while the oximeter is still held; 0.5 and above need a released state.

Score, the highest line that holds:

    1.0  clipped on at depth 28 mm or more, and slid on open
    0.75 clipped on (depth 10 mm or more), short of 28 mm or not slid on open
    0.5  pushed home (record), now over the finger (depth 10 mm or more) and let go for the last 1 s, but not
         clipped on (not gripping, not still, or touching the table): released, short of a held on state
    0.25 opened in the air (record): a minor first stage, earned while still held; nothing on the finger is needed
    0    otherwise
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

OXIMETER = "oximeter"
JAW = "jaw"
MANIKIN = "manikin"
P_WORLD = np.array([L.P[0], L.P[1], L.TOP + L.P[2]])  # the fingertip point (world)
FINGER_AXIS = np.array([1.0, 0.0, 0.0])
NAIL_SIDE = np.array([0.0, 1.0, 0.0])
WINDOW = int(round(L.WINDOW_S * L.CONTROL_HZ))  # 20 steps
LATCH = int(round(L.LATCH_S * L.CONTROL_HZ))  # 10 steps
START_ORIGIN = np.array([L.START[0], L.START[1], L.TOP + L.START[2]])


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_between(u, v) -> float:
    """Degrees between two vectors."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    c = float(u @ v) / (float(np.linalg.norm(u) * np.linalg.norm(v)) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rotation_angle_deg(r: np.ndarray) -> float:
    """The angle of a rotation matrix, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))))


@dataclass(frozen=True)
class Reading:
    """What one step's oximeter state says (distances in mm, angles in degrees)."""
    origin: np.ndarray
    r: np.ndarray
    q_deg: float
    s: float
    v: float
    w: float
    depth: float
    align_deg: float
    nail_deg: float
    low: float  # lowest point of both shells above the table (m)
    in_chamber: bool
    over_finger: bool
    jaws_open: bool
    gripping: bool


def read(state) -> Reading | None:
    """The readings of an oximeter state (a mechkit AssemblyState), or None when it cannot be read: a missing pose
    or pivot angle, a non-finite number or a degenerate quaternion."""
    if state is None:
        return None
    try:
        o = getattr(state, "origin", None)
        origin = np.asarray(state.pos if o is None else o, dtype=float).reshape(3)
        quat = np.asarray(state.quat_xyzw, dtype=float).reshape(4)
        q = float(state.joints[JAW])
        low = float(np.asarray(state.low, dtype=float).reshape(3)[2]) - L.TOP
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(quat)) and math.isfinite(q) and math.isfinite(low)):
        return None
    if float(np.linalg.norm(quat)) < 0.5:
        return None
    r = rotation(quat)
    a = q / 2
    rc = r @ rz(a)  # the centre-line frame
    m, n, zax = rc[:, 0], rc[:, 1], rc[:, 2]
    pivot = origin + r @ np.array([L.PIVOT_X, 0.0, 0.0])
    d = (P_WORLD - pivot) * 1000.0
    s, v, w = float(d @ m), float(d @ n), float(d @ zax)
    depth = L.lip_line(a) - s
    align = angle_between(m, FINGER_AXIS)
    nail = angle_between(n, NAIL_SIDE)
    q_deg = math.degrees(q)
    in_chamber = (L.CHAMBER_MIN_MM - L.EPS <= depth <= L.CHAMBER_MAX_MM + L.EPS
                  and abs(v) <= L.CENTRE_TOL_MM + L.EPS and abs(w) <= L.CENTRE_TOL_MM + L.EPS)
    over = (in_chamber and depth >= L.OVER_DEPTH_MM - L.EPS and align <= L.ALIGN_DEG + L.EPS_DEG
            and nail <= L.NAIL_SIDE_DEG + L.EPS_DEG)
    jaws_open = q_deg >= L.OPEN_DEG - L.EPS_DEG
    gripping = over and abs(q_deg - L.HOLD_DEG) <= L.HOLD_BAND_DEG + L.EPS_DEG
    return Reading(origin, r, q_deg, s, v, w, depth, align, nail, low, bool(in_chamber), bool(over),
                   bool(jaws_open), bool(gripping))


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


class Tracker:
    """The whole trial's records (slide, the two latches) and the last 1 s of poses and of the release check."""

    def __init__(self) -> None:
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=WINDOW)  # (origin, rotation, gripping) per readable step
        self.released: deque = deque(maxlen=WINDOW)
        self.d_open: float | None = None
        self.d_shut: float | None = None
        self.air_run = 0
        self.home_run = 0
        self.opened_in_air = False
        self.pushed_home = False
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    def _clear_windows(self) -> None:
        self.hist.clear()
        self.released.clear()
        self.air_run = 0
        self.home_run = 0

    def slid_on_open(self) -> bool:
        if self.d_open is None:
            return False
        return self.d_shut is None or self.d_shut <= self.d_open + L.SLIDE_SLACK_MM + L.EPS

    @staticmethod
    def at_start(rd: Reading) -> bool:
        return (float(np.linalg.norm(rd.origin - START_ORIGIN)) <= L.RESET_START_TOL
                and rotation_angle_deg(rd.r) <= L.RESET_START_DEG and abs(rd.q_deg) <= L.RESET_START_DEG)

    def latched(self) -> float:
        return 0.25 if self.opened_in_air else 0.0

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        state = states.get(OXIMETER) if isinstance(states, dict) else None
        rd = read(state)
        if rd is None:
            self.bad_steps += 1  # a blow-up or a missing part: the records stand, the 1 s windows start again
            self._clear_windows()
            self.prev = None
            level = self.latched()
            self.last = Result(level, False, {"valid_state": False, "bad_steps": self.bad_steps, "tier": level})
            return self.last
        if (self.prev is not None and self.at_start(rd)
                and float(np.linalg.norm(rd.origin - self.prev.origin)) >= L.RESET_JUMP):
            self.reset()  # put back on its start pose in one step: a new trial
            self.steps = 1
            self.resets += 1
        self.prev = rd
        # the way in
        if rd.depth < 0.0:
            self.d_open = self.d_shut = None
        elif rd.in_chamber:
            if rd.jaws_open:
                self.d_open = rd.depth if self.d_open is None else max(self.d_open, rd.depth)
            else:
                self.d_shut = rd.depth if self.d_shut is None else max(self.d_shut, rd.depth)
        # the two records
        in_air = rd.jaws_open and rd.low >= L.TABLE_CLEAR - L.EPS
        self.air_run = self.air_run + 1 if in_air else 0
        home = rd.over_finger and rd.depth >= L.HOME_MM - L.EPS
        self.home_run = self.home_run + 1 if home else 0
        if self.air_run >= LATCH:
            self.opened_in_air = True
        if self.home_run >= LATCH:
            self.pushed_home = True
        # let go and still
        try:
            dist = state.distance
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not released (rigstate's rule)
            ok = False
        self.released.append(ok)
        self.hist.append((rd.origin.copy(), rd.r.copy(), rd.gripping))
        let_go = len(self.released) == WINDOW and all(self.released)
        moved = turned = None
        still = False
        held_on = False
        if len(self.hist) == WINDOW:
            o0, r0, _ = self.hist[0]
            moved = max(float(np.linalg.norm(o - o0)) for o, _, _ in self.hist) * 1000.0
            turned = max(rotation_angle_deg(r0.T @ r) for _, r, _ in self.hist)
            still = moved < L.STILL_MM and turned < L.STILL_DEG
            held_on = all(g for _, _, g in self.hist)
        off_table = rd.low >= L.TABLE_CLEAR - L.EPS
        clipped = bool(let_go and off_table and held_on and still)
        slid = self.slid_on_open()
        if clipped and rd.depth >= L.HOME_MM - L.EPS and slid:
            level = 1.0
        elif clipped:
            level = 0.75
        elif self.pushed_home and let_go and rd.over_finger:
            level = 0.5  # released on the finger after being pushed home, but not clipped on
        else:
            level = self.latched()
        gap = None
        try:
            gap = rigstate.hands_gap(arms, state.distance)
        except Exception:  # noqa: BLE001 - trace only
            gap = None
        details = {
            "valid_state": True, "tier": level, "q_deg": _r(rd.q_deg), "depth_mm": _r(rd.depth),
            "s_mm": _r(rd.s), "v_mm": _r(rd.v), "w_mm": _r(rd.w), "align_deg": _r(rd.align_deg),
            "nail_deg": _r(rd.nail_deg), "low_mm": _r(rd.low * 1000.0), "in_chamber": rd.in_chamber,
            "over_finger": rd.over_finger, "jaws_open": rd.jaws_open, "gripping": rd.gripping,
            "d_open_mm": _r(self.d_open), "d_shut_mm": _r(self.d_shut), "slid_on_open": slid,
            "released": ok, "let_go_1s": let_go, "moved_mm": _r(moved), "turned_deg": _r(turned),
            "still_1s": still, "gripping_1s": held_on, "clipped_on": clipped,
            "opened_in_air": self.opened_in_air, "pushed_home": self.pushed_home,
            "zone_mm": None if gap is None or not math.isfinite(gap) else round(gap * 1000.0, 1),
            "arms_readable": rigstate.readable(arms),
            "origin": [round(float(x), 4) for x in rd.origin],
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
