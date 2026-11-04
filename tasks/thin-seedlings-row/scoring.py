"""The scorer, as plain functions of the ten seedlings' states. NumPy only; no simulator needed.

The rule is about the end state and about the whole trial: the seedlings left on the bed must be exactly the ones the
thinning rule keeps (going from the tallest down, keep a seedling if it is at least 10 cm from every one kept), the
others must be in a compost tray, and no seedling left on the bed may have been disturbed at any moment (tipped,
shoved or pulled and set back). So the scorer keeps a little state (how far each seedling has ever moved from its
planting spot and how far it has ever tilted) and is fed one observation per control step by ``task.success()``.

    Tracker.update(states) -> Result(score, success, details)

``states`` maps each seedling's name (``layout.NAMES``) to anything with ``pos`` (bounding-box centre),
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
``yam_arena.task_file.ObjectState`` has. A state with a non-finite number or a zero quaternion (a simulator blow-up),
or a missing seedling, scores 0 for that step and leaves the tracker as it was.

A seedling is a plain cylinder, the same either end up, so 'tilt' is the angle between its axis and the vertical
folded to 0..90 degrees. Spacing is judged on the planting spots of the seedlings left on the bed (they may each
have moved at most 15 mm, or the trial already scores 0).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
START = {n: np.array(L.start_pos(n)) for n in L.NAMES}


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


def all_valid(states) -> bool:
    try:
        return all(valid(states[n]) for n in L.NAMES)
    except (KeyError, TypeError):
        return False


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the seedling's axis (degrees, 0..90): a plain cylinder is the same
    either end up, so whichever end is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- where a seedling is -----------------------------------------------------------------------------------------
def moved(name: str, s) -> float:
    """How far the seedling's centre is from where it started, upright on its planting spot (m, in 3D, so a seedling
    lifted counts as moved)."""
    return float(np.linalg.norm(np.asarray(s.pos, dtype=float) - START[name]))


def over_bed_xy(p) -> bool:
    return bool(abs(p[0] - L.BED_X) <= L.BED_SIZE[0] / 2 and abs(p[1] - L.BED_Y) <= L.BED_SIZE[1] / 2)


def on_bed(s) -> bool:
    """Standing or lying on the seed bed: its centre over the bed and its lowest point within 5 mm of the bed's top
    (not held above it)."""
    return bool(over_bed_xy(s.pos) and abs(float(s.low[2]) - L.BED_TOP) <= L.ON_BED_Z_TOL)


def tray_of(s) -> str | None:
    """Which compost tray the seedling is in ('left' or 'right'), or None: its centre over the tray's 100 x 100 mm
    floor and its lowest point below the tray's rim (standing, lying or leaning, but not held above it)."""
    for side, (tx, ty) in L.TRAYS.items():
        if (abs(s.pos[0] - tx) <= L.TRAY_INNER_HALF and abs(s.pos[1] - ty) <= L.TRAY_INNER_HALF
                and float(s.low[2]) < L.TRAY_TOP_Z):
            return side
    return None


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def place(s) -> str:
    """'bed', 'tray_left', 'tray_right', 'off_table' or 'elsewhere' (on the table, held, on a rim, ...)."""
    if off_table(s):
        return "off_table"
    if on_bed(s):
        return "bed"
    t = tray_of(s)
    return f"tray_{t}" if t else "elsewhere"


# -- the thinning rule on the planting spots ---------------------------------------------------------------------
def spacing_ok(names) -> bool:
    """No two of these seedlings planted closer than 10 cm."""
    return all(abs(L.START_Y[a] - L.START_Y[b]) >= L.SPACING - 1e-9 for a, b in itertools.combinations(names, 2))


def one_wrong_choice(names) -> tuple[str, str] | None:
    """(keeper, stand-in) if ``names`` is the rule's keep set with one keeper replaced by a shorter seedling planted
    within 10 cm of it, the 10 cm spacing still kept; else None."""
    names = frozenset(names)
    missing, extra = L.KEEP - names, names - L.KEEP
    if len(missing) != 1 or len(extra) != 1:
        return None
    (k,), (x,) = tuple(missing), tuple(extra)
    if L.HEIGHT[x] < L.HEIGHT[k] and abs(L.START_Y[x] - L.START_Y[k]) < L.SPACING and spacing_ok(names):
        return k, x
    return None


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.peak_move = {n: 0.0 for n in L.NAMES}  # m: the furthest each seedling's centre has been from its start
        self.peak_tilt = {n: 0.0 for n in L.NAMES}  # degrees: the most each seedling has tilted
        self.bad_steps = 0  # control steps with a non-finite or missing state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: every seedling upright and still on its planting spot."""
        for n in L.NAMES:
            s = states[n]
            if not (float(np.hypot(*(np.asarray(s.pos[:2], dtype=float) - START[n][:2]))) < L.START_TOL
                    and abs(float(s.low[2]) - L.BED_TOP) < 0.003 and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some seedling moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - self.prev[n][0]))
            > L.RESET_JUMP + float(np.linalg.norm(self.prev[n][1])) * self.dt + fall
            for n in L.NAMES
        )

    def update(self, states) -> Result:
        if not all_valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states)
            return self.last
        now = {n: (np.array(states[n].pos, dtype=float), np.array(states[n].lin_vel, dtype=float)) for n in L.NAMES}
        if self.prev is not None and self._teleported(now) and self.at_start(states):
            self.reset()  # the simulator put every seedling back on its spot: a new episode
        self.prev = now
        self.steps += 1
        for n in L.NAMES:
            self.peak_move[n] = max(self.peak_move[n], moved(n, states[n]))
            self.peak_tilt[n] = max(self.peak_tilt[n], tilt_deg(states[n].quat_xyzw))
        self.last = self.result(states)
        return self.last

    def result(self, states) -> Result:
        if not all_valid(states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        where = {n: place(states[n]) for n in L.NAMES}
        move = {n: max(self.peak_move[n], moved(n, states[n])) for n in L.NAMES}
        tilt = {n: max(self.peak_tilt[n], tilt_deg(states[n].quat_xyzw)) for n in L.NAMES}
        left = frozenset(n for n in L.NAMES if where[n] == "bed")
        in_tray = frozenset(n for n in L.NAMES if where[n].startswith("tray_"))
        disturbed = sorted(n for n in left if move[n] > L.DISTURB_TOL or tilt[n] > L.TILT_MAX_DEG)
        nudged = sorted(n for n in left if move[n] > L.KEEP_TOL)
        off = sorted(n for n in L.NAMES if where[n] == "off_table")
        spaced = spacing_ok(left)
        rest_composted = all(n in in_tray for n in L.NAMES if n not in left)
        swap = one_wrong_choice(left)
        pulled_ok = len(L.PULL & in_tray)

        if off:
            zero = "off_table"
        elif disturbed:
            zero = "disturbed"
        elif len(left) < L.MIN_LEFT:
            zero = "over_thinned"
        else:
            zero = None
        if zero:
            score = 0.0
        elif left == L.KEEP and not nudged and rest_composted:
            score = 1.0
        elif spaced and rest_composted and ((left == L.KEEP and len(nudged) == 1) or (swap and not nudged)):
            score = 0.75
        elif spaced:
            score = 0.5
        elif pulled_ok >= L.MIN_PULLED_PARTIAL and L.KEEP <= left:
            score = 0.25
        else:
            score = 0.0
        still = all(is_resting(states[n]) for n in L.NAMES)
        success = bool(score == 1.0 and still)
        details = {
            "valid_state": True,
            "zero_reason": zero,
            "left_on_bed": sorted(left),
            "keep_set_left": left == L.KEEP,
            "spacing_ok": spaced,
            "wrong_choice": list(swap) if swap else None,
            "rest_composted": rest_composted,
            "to_pull_in_trays": pulled_ok,
            "disturbed": disturbed,
            "nudged": nudged,
            "off_table": off,
            "where": where,
            "peak_move_mm": {n: _mm(move[n]) for n in L.NAMES},
            "peak_tilt_deg": {n: round(tilt[n], 1) for n in L.NAMES},
            "max_keeper_move_mm": _mm(max(move[n] for n in L.KEEP)),
            "max_keeper_tilt_deg": round(max(tilt[n] for n in L.KEEP), 2),
            "all_still": still,
            "moving": sorted(n for n in L.NAMES if not is_resting(states[n])),
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
