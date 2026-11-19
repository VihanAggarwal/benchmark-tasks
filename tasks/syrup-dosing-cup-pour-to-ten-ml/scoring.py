"""The scorer: the syrup ledger and the dose rules, as plain functions of the bottle's and the cup's poses. NumPy only.

The simulator has no liquids. The syrup is a volume ledger driven by the bottle's and the cup's real poses, once per
control step (20 Hz), fed from ``task.success()``:

  1. The bottle holds ``v_bottle`` of syrup, 118 mL at the start. Its free surface is the horizontal plane that has
     exactly that volume below it inside the bottle's bore (a 45 mm round body, then a 22 mm neck), found from the
     bottle's pose (``surface_height``).
  2. Syrup leaves over the lip's lowest point when the surface stands above it, by the depth ``h`` of the surface
     over that point. The rate is a laminar film over the lip, ``Q = g w h^3 / (3 nu)`` (w the chord of the lip circle
     that the surface cuts, nu = 1.15e-4 m^2/s for a 150 mPa s, 1.31 g/mL syrup), but never more than what the neck
     passes, ``Q_CAP`` = 3.0 mL/s. No delay and no tail are written in: the stream stops because tilting the bottle
     back drops the surface under the lip, and because syrup leaving lowers the surface.
  3. The stream falls straight down from the lip's lowest point. It goes into the cup if that point is over the cup's
     opening (4 mm inside its round wall) and above its rim, and the cup is tilted 30 degrees or less; otherwise it is
     spilled onto the table or the scale. A cup tilted more than 45 degrees pours out what it holds.
  4. The time the stream spends falling into the cup while the cup is not held up and level (its lowest point under
     20 mm over the pan, or tilted over 5 degrees) is counted as 'unread' seconds, with its volume: the label's rule
     is to read the line with the cup held level at eye height. More than 0.15 s of it caps the score at 0.5.

    Tracker.update(bottle, cup, hands_clear) -> Result(score, success, details)

``bottle`` and ``cup`` are anything with ``origin`` (the middle of the part's bottom face), ``quat_xyzw``, ``pos``,
``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``mechkit.AssemblyState`` has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

N_SLICES = 160  # slices per cylinder in the volume integral: the volume is exact to under 0.01 mL in this bottle
_S_BODY = L.BORE_Z0 + (np.arange(N_SLICES) + 0.5) * (L.BORE_Z1 - L.BORE_Z0) / N_SLICES
_S_NECK = L.BORE_Z1 + (np.arange(N_SLICES) + 0.5) * (L.NECK_TOP - L.BORE_Z1) / N_SLICES
_DS_BODY = (L.BORE_Z1 - L.BORE_Z0) / N_SLICES
_DS_NECK = (L.NECK_TOP - L.BORE_Z1) / N_SLICES


# -- poses ----------------------------------------------------------------------------------------------------------------
def quat_matrix(quat_xyzw) -> np.ndarray:
    """3x3 rotation of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 0:
        raise ValueError("not a rotation")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's z axis and the world's vertical (degrees)."""
    r = quat_matrix(quat_xyzw)
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the syrup ------------------------------------------------------------------------------------------------------------
def _segment_area(d, r):
    """Area of the part of a disc of radius r that lies below a chord at signed distance d from the centre."""
    d = np.clip(d, -r, r)
    return r * r * np.arccos(-d / r) + d * np.sqrt(np.maximum(r * r - d * d, 0.0))


def volume_below(zs: float, origin, axis) -> float:
    """Volume (m^3) of the bore (body then neck) that lies below the horizontal plane z = zs, the bottle's bottom
    face centred at ``origin`` and its axis along the unit vector ``axis``."""
    sin = max(math.sqrt(max(1.0 - float(axis[2]) ** 2, 0.0)), 1e-9)
    total = 0.0
    for r, s, ds in ((L.BORE_R, _S_BODY, _DS_BODY), (L.LIP_R, _S_NECK, _DS_NECK)):
        cz = origin[2] + axis[2] * s  # each slice's centre height
        total += float(np.sum(_segment_area((zs - cz) / sin, r))) * ds
    return total


def surface_height(v: float, origin, axis) -> float:
    """The height z of the free surface of ``v`` m^3 of syrup in the bottle (bisection on ``volume_below``)."""
    reach = L.NECK_TOP + L.BORE_R
    lo, hi = float(origin[2]) - reach, float(origin[2]) + reach
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if volume_below(mid, origin, axis) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lip_low_point(origin, axis) -> np.ndarray:
    """The lowest point of the lip circle (world), where syrup goes over."""
    origin, axis = np.asarray(origin, dtype=float), np.asarray(axis, dtype=float)
    centre = origin + axis * L.NECK_TOP
    sin = math.sqrt(max(1.0 - axis[2] ** 2, 0.0))
    if sin < 1e-9:  # upright or inverted: the whole rim is level; take the point under the axis
        return centre
    down = -(np.array([0.0, 0.0, 1.0]) - axis * axis[2]) / sin
    return centre + L.LIP_R * down


def pour_rate(v: float, origin, axis) -> tuple[float, float, float]:
    """(Q m^3/s, depth h m of the surface over the lip's lowest point, chord w m) for ``v`` m^3 in the bottle."""
    if not v > 0.0:
        return 0.0, -1.0, 0.0
    zs = surface_height(v, origin, axis)
    low = lip_low_point(origin, axis)
    h = zs - float(low[2])
    if not h > 0.0:
        return 0.0, h, 0.0
    sin = max(math.sqrt(max(1.0 - float(axis[2]) ** 2, 0.0)), 1e-9)
    centre_z = float(origin[2]) + float(axis[2]) * L.NECK_TOP
    d = (zs - centre_z) / sin
    w = 2.0 * math.sqrt(max(L.LIP_R ** 2 - d ** 2, 0.0)) if d < L.LIP_R else 2.0 * L.LIP_R
    film = L.GRAV * w * h ** 3 / (3.0 * L.NU)
    return min(film, L.Q_CAP), h, w


def stream_in_cup(point, cup_origin, cup_rot) -> bool:
    """A stream falling from ``point`` lands in the round cup: over its opening (within CATCH_MARGIN of the circle
    inside its wall panels, so 14 mm of its axis), above its rim, the cup tilted CATCH_TILT_DEG or less."""
    if float(cup_rot[2, 2]) < math.cos(math.radians(L.CATCH_TILT_DEG)):
        return False
    local = cup_rot.T @ (np.asarray(point, dtype=float) - np.asarray(cup_origin, dtype=float))
    radius = L.CUP_INNER_HALF - L.CATCH_MARGIN
    return bool(math.hypot(float(local[0]), float(local[1])) <= radius and local[2] >= L.CUP_H)


# -- rule predicates ------------------------------------------------------------------------------------------------------
def cup_held_level(cup) -> bool:
    """Held up (lowest point 20 mm or more over the pan) and level (5 degrees or less)."""
    return bool(float(cup.low[2]) >= L.PAN_TOP + L.HELD_CLEAR and tilt_deg(cup.quat_xyzw) <= L.HELD_TILT_DEG)


def cup_on_scale(cup) -> bool:
    off = math.hypot(float(cup.origin[0]) - L.PAN_X, float(cup.origin[1]) - L.PAN_Y)
    return bool(off <= L.CUP_SET_XY and abs(float(cup.low[2]) - L.PAN_TOP) <= L.CUP_SET_Z
                and tilt_deg(cup.quat_xyzw) < L.CUP_SET_TILT_DEG)


def bottle_set_down(bottle) -> bool:
    off = math.hypot(float(bottle.origin[0]) - L.BOTTLE_X, float(bottle.origin[1]) - L.BOTTLE_Y)
    return bool(off <= L.BOTTLE_SET_XY and abs(float(bottle.low[2]) - L.TOP) <= L.BOTTLE_SET_Z
                and tilt_deg(bottle.quat_xyzw) < L.BOTTLE_SET_TILT_DEG)


def _finite(*states) -> bool:
    try:
        for s in states:
            for v in (s.origin, s.quat_xyzw, s.pos, s.lin_vel, s.ang_vel, s.low, s.high):
                if not np.all(np.isfinite(np.asarray(v, dtype=float))):
                    return False
            if not float(np.linalg.norm(np.asarray(s.quat_xyzw, dtype=float))) > 0.0:
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def ml(v: float) -> float:
    return v * 1e6


class Tracker:
    """The syrup ledger and what has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.v_bottle = L.FILL
        self.v_cup = 0.0
        self.v_peak = 0.0
        self.spilled = 0.0
        self.unread = 0.0
        self.unread_s = 0.0
        self.dumped = False
        self.poured = 0.0  # all that has left the bottle
        self.flow = 0.0  # m^3/s at the latest step
        self.depth = -1.0  # m: the surface over the lip's lowest point at the latest step
        self.prev = None  # (bottle origin, cup origin) at the previous valid step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(bottle, cup) -> bool:
        b = math.hypot(float(bottle.origin[0]) - L.BOTTLE_X, float(bottle.origin[1]) - L.BOTTLE_Y)
        c = math.hypot(float(cup.origin[0]) - L.CUP_X, float(cup.origin[1]) - L.CUP_Y)
        return bool(b <= L.START_TOL and c <= L.START_TOL and tilt_deg(bottle.quat_xyzw) < 2.0
                    and abs(float(bottle.low[2]) - L.TOP) <= L.BOTTLE_SET_Z
                    and abs(float(cup.low[2]) - L.PAN_TOP) <= L.CUP_SET_Z and is_resting(bottle) and is_resting(cup))

    def update(self, bottle, cup, hands_clear: bool) -> Result:
        """Advance the ledger one control step and score the state as it stands. ``hands_clear``: both grippers'
        finger zones are LET_GO or more from both parts (False if the arms could not be read)."""
        self.steps += 1
        if not _finite(bottle, cup):  # a blown-up or missing state: nothing is credited and no latch moves
            self.bad_steps += 1
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        now = (np.array(bottle.origin, dtype=float), np.array(cup.origin, dtype=float))
        if self.prev is not None and self.at_start(bottle, cup):
            jump = max(float(np.linalg.norm(now[i] - self.prev[i])) for i in range(2))
            if jump >= L.RESET_JUMP:  # the simulator put both back on their marks: a new episode
                self.reset()
                self.resets += 1
                self.steps = 1
        self.prev = now
        self._pour(bottle, cup)
        self.last = self.result(bottle, cup, hands_clear)
        return self.last

    # -- the ledger ----------------------------------------------------------------------------------------------------
    def _pour(self, bottle, cup) -> None:
        rb, rc = quat_matrix(bottle.quat_xyzw), quat_matrix(cup.quat_xyzw)
        axis = rb[:, 2]
        origin = np.asarray(bottle.origin, dtype=float)
        q, h, _ = pour_rate(self.v_bottle, origin, axis)
        self.flow = q
        self.depth = h
        dv = min(q * self.dt, self.v_bottle)
        if dv > 0.0:
            self.v_bottle -= dv
            self.poured += dv
            if stream_in_cup(lip_low_point(origin, axis), cup.origin, rc):
                self.v_cup += dv
                if not cup_held_level(cup):
                    self.unread += dv
                    self.unread_s += self.dt
            else:
                self.spilled += dv
        if tilt_deg(cup.quat_xyzw) > L.DUMP_TILT_DEG:
            if self.v_cup > 0.0:
                self.spilled += self.v_cup
                self.v_cup = 0.0
            self.dumped = True
        self.v_peak = max(self.v_peak, self.v_cup)

    # -- the score -----------------------------------------------------------------------------------------------------
    def tier(self, placed: bool) -> float:
        """The score from the ledger; ``placed``: cup on the scale, bottle set down, hands clear."""
        dose = ml(self.v_cup)
        spilled, unread = ml(self.spilled), round(self.unread_s, 6)
        if ml(self.v_peak) > L.OVERDOSE_ML or self.dumped or spilled > L.SPILL_ZERO_ML:
            return 0.0
        if L.WINDOW[0] <= dose <= L.WINDOW[1]:
            clean = spilled <= L.CLEAN_SPILL_ML and unread <= L.UNREAD_S and placed
            score = 1.0 if clean else 0.75
            if unread > L.UNREAD_S or spilled > L.SPILL_CAP_ML:
                score = min(score, 0.5)
            return score
        if L.WIDE[0] <= dose <= L.WIDE[1]:
            return 0.5
        if dose >= L.PARTIAL_ML:
            return 0.25
        return 0.0

    def result(self, bottle, cup, hands_clear: bool) -> Result:
        placed = bool(cup_on_scale(cup) and bottle_set_down(bottle) and hands_clear)
        score = self.tier(placed)
        resting = is_resting(bottle) and is_resting(cup)
        success = bool(score == 1.0 and resting)
        details = {
            "valid_state": True,
            "score": score,
            "cup_ml": round(ml(self.v_cup), 3),
            "peak_ml": round(ml(self.v_peak), 3),
            "spilled_ml": round(ml(self.spilled), 3),
            "unread_ml": round(ml(self.unread), 3),
            "unread_s": round(self.unread_s, 3),
            "bottle_ml": round(ml(self.v_bottle), 3),
            "flow_ml_s": round(ml(self.flow), 3),
            "depth_mm": round(self.depth * 1000, 2),
            "dumped": self.dumped,
            "cup_held_level": cup_held_level(cup),
            "cup_on_scale": cup_on_scale(cup),
            "bottle_set_down": bottle_set_down(bottle),
            "hands_clear": bool(hands_clear),
            "bottle_tilt_deg": round(tilt_deg(bottle.quat_xyzw), 2),
            "cup_tilt_deg": round(tilt_deg(cup.quat_xyzw), 2),
            "resting": resting,
        }
        return Result(score, success, details)
