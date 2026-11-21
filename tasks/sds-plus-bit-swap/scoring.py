"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
hammer, the 6 mm bit in its chuck and the 10 mm bit on the bench (loose bodies: pose and velocity) and both grippers'
TCPs, pad tips and openings, and works in the hammer's own frame (layout.py), so it doesn't matter where the hammer has
been nudged to.

A bit's frame has its origin at the end of its shank and +y pointing from the tip to that end (the way it goes into
the chuck); its two drive slots lie along its +-x (the 6 mm bit) or 45 degrees round from it (the 10 mm bit).

    IN THE BORE  a bit's shank end within 3 mm of the chuck's axis, the bit within 6 degrees of it (the bore's play
                 lets a part-inserted bit lean about 4); its DEPTH is how far its shank end
                 is past the nose (home: 40 mm)
    OLD OUT      the 6 mm bit's shank end at least 12 mm in front of the nose (clear of the dust cap), or 20 mm or
                 more off the chuck's axis
    RELEASED     old out and no gripper within 30 mm of it
    STARTED      the 10 mm bit in the bore at least 10 mm deep
    LOCKED       the 10 mm bit home (within 2 mm of 40 mm deep) with its slots on the keys and the sleeve let go (under
                 6 mm back): the locking balls hold it (task.py)
    CLEAR        no gripper within 30 mm of the hammer or either bit, all three still (under 5 mm/s)

A gripper is the segment from its TCP to its pad tips, swollen by half its pads' current gap plus 8 mm.

    1.0   RELEASED, LOCKED and CLEAR (success, held 1 s)     0.75  RELEASED and LOCKED
    0.5   RELEASED and STARTED                                0.25  RELEASED
    0     anything else (the untouched start), and any trial in which the hammer goes off the bench
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

HAMMER, OLD_BIT, NEW_BIT = "rotary_hammer", "bit_6mm", "bit_10mm"


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=float).reshape(4))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < L.EPS:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass(frozen=True)
class Body:
    origin: np.ndarray
    rot: np.ndarray
    vel: np.ndarray

    def local(self, p) -> np.ndarray:
        return self.rot.T @ (np.asarray(p, dtype=float) - self.origin)


def body(state) -> Body | None:
    try:
        o = np.asarray(state.origin, dtype=float).ravel()
        r = quat_matrix(state.quat_xyzw)
        v = np.asarray(state.lin_vel, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return None
    if o.size != 3 or v.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(v))):
        return None
    return Body(o, r, v)


def read(states):
    try:
        bs = tuple(body(states[n]) for n in (HAMMER, OLD_BIT, NEW_BIT))
    except (KeyError, TypeError):
        return None
    return None if any(b is None for b in bs) else bs


@dataclass(frozen=True)
class Seat:
    in_bore: bool
    depth: float  # how far the shank end is past the nose (m); negative: still outside
    roll: float  # the slots' angle off the drive keys (rad, folded to [0, pi/2])


def slot_angle(hammer: Body, bit: Body, slot_offset: float = 0.0) -> float:
    """The angle of a bit's slots round the chuck's axis (hammer x-z plane, from +x toward +z), unfolded."""
    slots = (hammer.rot.T @ bit.rot) @ np.array([math.cos(slot_offset), 0.0, math.sin(slot_offset)])
    return math.atan2(slots[2], slots[0])


def seat(hammer: Body, bit: Body, slot_offset: float = 0.0) -> Seat:
    """``slot_offset``: where the bit's slots lie round its own axis, from its frame's x."""
    p = hammer.local(bit.origin)
    rel = hammer.rot.T @ bit.rot
    axis = rel[:, 1]
    off_axis = math.hypot(p[0], p[2] - L.AXIS_Z)
    in_bore = off_axis <= L.ON_AXIS and axis[1] >= math.cos(math.radians(L.ON_AXIS_DEG))
    a = slot_angle(hammer, bit, slot_offset) % math.pi
    roll = min(a, math.pi - a)
    return Seat(bool(in_bore), float(p[1]), float(roll))


def bit_samples(bit: Body, shank_r: float, flute_r: float, flute_l: float, n: int = 9):
    """Points along a bit's axis with its radius there (for distances from the grippers)."""
    out = []
    for k in range(n):
        y = -(L.SHANK_L + flute_l) * k / (n - 1)
        r = shank_r if -y <= L.SHANK_L else flute_r
        out.append((bit.origin + bit.rot @ np.array([0.0, y, 0.0]), r))
    return out


HAMMER_BOXES = [  # hammer frame: the housing with its nose and chuck, and the pistol grip
    (np.array([L.BODY_X[0], -0.005, L.BODY_Z[0]]), np.array([L.BODY_X[1], L.LENGTH, L.BODY_Z[1]])),
    (np.array([L.GRIP_X[0], L.GRIP_Y[0], 0.010]), np.array([L.GRIP_X[1], L.GRIP_Y[1], 0.066])),
]


def gripper_samples(arms, n: int = 6) -> list[tuple[np.ndarray, float]]:
    if not rigstate.readable(arms):
        return []
    out = []
    for s in ("left", "right"):
        a = arms[s]
        tcp, tips = np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float)
        rad = max(0.0, min(1.0, float(a.opening))) * L.FINGER_GAP / 2 + L.PAD_MARGIN
        out += [(tcp + (tips - tcp) * k / (n - 1), rad) for k in range(n)]
    return out


def tier(released: bool, started: bool, locked: bool, clear: bool, foul: bool) -> float:
    if foul or not released:
        return 0.0
    if locked:
        return 1.0 if clear else 0.75
    return 0.5 if started else 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.fouls: list[str] = []
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        bodies = read(states) if isinstance(states, dict) else None
        if bodies is None:
            self.bad_steps += 1
            self.last = Result(0.0 if self.fouls else self.last.score, False,
                               {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        hammer, old, new = bodies
        so, sn = seat(hammer, old), seat(hammer, new, L.NEW_ROLL)
        pe = hammer.local(old.origin)  # its shank end, tilted or not
        old_out = pe[1] <= -L.OUT or math.hypot(pe[0], pe[2] - L.AXIS_Z) > 0.020
        started = sn.in_bore and sn.depth >= L.STARTED
        try:
            sleeve = float(states[HAMMER].joints["sleeve"])
        except (AttributeError, KeyError, TypeError, ValueError):
            sleeve = math.inf  # unreadable: the lock can't be confirmed
        locked = (sn.in_bore and sn.depth >= L.DEPTH - L.HOME_TOL and sn.roll <= L.KEY_TOL
                  and sleeve < L.SLEEVE_FREE)
        gaps = {"old": math.inf, "kit": math.inf}
        points = gripper_samples(arms)
        old_pts = bit_samples(old, L.SHANK_R, L.OLD_R, L.OLD_L)
        new_pts = bit_samples(new, L.SHANK_R, L.NEW_R, L.NEW_L)
        for pt, rad in points:
            d_old = min(math.dist(pt, q) - r for q, r in old_pts) - rad
            d_new = min(math.dist(pt, q) - r for q, r in new_pts) - rad
            lp = hammer.local(pt)
            d_ham = min(L.box_distance(lp, lo, hi) for lo, hi in HAMMER_BOXES) - rad
            gaps["old"] = min(gaps["old"], d_old)
            gaps["kit"] = min(gaps["kit"], d_old, d_new, d_ham)
        if hammer.origin[2] < L.TOP - 0.05 and "off the bench" not in self.fouls:
            self.fouls.append("off the bench")
        released = old_out and gaps["old"] >= L.CLEAR
        still = max(float(np.linalg.norm(b.vel)) for b in bodies) < L.STILL_SPEED
        clear = bool(points) and gaps["kit"] >= L.CLEAR and still
        level = tier(released, started, locked, clear, bool(self.fouls))
        details = {
            "valid_state": True, "score": level, "old_out": old_out, "released": released, "started": started,
            "locked": locked, "clear": clear, "sleeve_mm": None if math.isinf(sleeve) else round(sleeve * 1000, 2), "still": still, "fouls": list(self.fouls),
            "old_depth_mm": round(so.depth * 1000, 1) if so.in_bore else None,
            "new_depth_mm": round(sn.depth * 1000, 1) if sn.in_bore else None,
            "new_roll_deg": round(math.degrees(sn.roll), 1),
            "new_local_mm": [round(v * 1000, 1) for v in hammer.local(new.origin)],
            "old_local_mm": [round(v * 1000, 1) for v in hammer.local(old.origin)],
            "new_axis_deg": [round(math.degrees(math.asin(max(-1.0, min(1.0, v)))), 1)
                             for v in ((hammer.rot.T @ new.rot)[0, 1], (hammer.rot.T @ new.rot)[2, 1])],
            "hammer_mm": [round(v * 1000, 1) for v in hammer.origin[:2]],
            "hammer_yaw_deg": round(math.degrees(math.atan2(hammer.rot[1, 0], hammer.rot[0, 0])), 1),
            "gaps_mm": {k: (None if math.isinf(v) else round(v * 1000, 1)) for k, v in gaps.items()},
            "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
