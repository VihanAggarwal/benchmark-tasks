"""The scorer, as plain functions of the four air movers' states. NumPy only; no simulator needed.

The rule is the drying set-up water-damage restoration crews are taught: air movers along the wet walls, each with
its outlet at the base of the wall and blowing into it at 15 to 45 degrees so the air sweeps along it, about one per
10 to 16 ft of wall, and all of them blowing the same way round the room so the air turns in one vortex. All of it is
read from the movers' poses, as a top photo with a tape and a protractor would read it. The only memory the scorer
keeps is each mover's last half second of positions, to tell whether it is still.

    Tracker.update(movers) -> Result(score, success, details)

``movers`` maps each name in ``layout.MOVERS`` to anything with ``origin`` (the mover frame's origin: the middle of
the body's base), ``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high``
(world-aligned bounding box), as mechkit's ``AssemblyState`` has. A missing mover or a state with a non-finite number
or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
FIELDS = ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")
SENSES = ("clockwise", "anticlockwise")
EPS = 1e-9  # an airflow component smaller than this is zero (straight along or straight across the room)
WALL_SIDES = ("left", "right")


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


def valid(m) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(m, k), dtype=float).ravel() for k in FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(m) -> float:
    """Angle between the mover's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(m.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def airflow(m) -> tuple[float, float] | None:
    """The airflow direction (the body axis toward the outlet) seen from above, as a unit 2-D vector; None if it
    points straight up or down."""
    a = rotation(m.quat_xyzw)[:, 0]
    n = math.hypot(float(a[0]), float(a[1]))
    if n < 1e-9:
        return None
    return float(a[0]) / n, float(a[1]) / n


def heading_rad(m) -> float:
    """Which way the airflow points seen from above (rad, counter-clockwise from +x); NaN if straight up or down."""
    f = airflow(m)
    return math.atan2(f[1], f[0]) if f is not None else float("nan")


def wall_of(m) -> str | None:
    """The side wall the mover blows toward: 'left' (+y) if its airflow has any +y part, 'right' if any -y part,
    None if it blows straight along the room (or straight up or down)."""
    f = airflow(m)
    if f is None or abs(f[1]) < EPS:
        return None
    return "left" if f[1] > 0.0 else "right"


def angle_into_wall_deg(m) -> float:
    """The angle between the airflow and its side wall, seen from above (degrees): 0 along the wall, 90 straight at
    it. NaN if the airflow points straight up or down."""
    f = airflow(m)
    if f is None:
        return float("nan")
    return math.degrees(math.atan2(abs(f[1]), abs(f[0])))


def sense_of(m) -> str | None:
    """Which way round the room the mover blows, seen from above: 'clockwise' along the left wall away from the arms
    (+x) or along the right wall toward them (-x); 'anticlockwise' the reverse. None if it blows straight across or
    straight along the room."""
    f = airflow(m)
    if f is None or abs(f[0]) < EPS or abs(f[1]) < EPS:
        return None
    return "clockwise" if f[0] * f[1] > 0.0 else "anticlockwise"


def _corners(m, box) -> np.ndarray:
    """World corners (8, 3) of a box given in the mover frame as ((u0, u1), (v0, v1), (z0, z1))."""
    r = rotation(m.quat_xyzw)
    o = np.asarray(m.origin, dtype=float)
    (u0, u1), (v0, v1), (z0, z1) = box
    return np.array([o + r @ np.array([u, v, z]) for u in (u0, u1) for v in (v0, v1) for z in (z0, z1)])


def outlet_gap(m, wall: str | None = None) -> float:
    """Distance (m, seen from above, square to the wall) from the outlet's nearest point to the line of the face of the
    side wall the mover blows toward (or ``wall``): negative if the outlet is past the face. Infinite if there is no such
    wall."""
    wall = wall_of(m) if wall is None else wall
    if wall is None:
        return math.inf
    c = _corners(m, L.OUTLET_BOX)
    if wall == "left":
        return float(L.LEFT_FACE - c[:, 1].max())
    return float(c[:, 1].min() - L.RIGHT_FACE)


def lowest(m) -> float:
    """Height of its lowest point above the carpet (m)."""
    return float(np.asarray(m.low, dtype=float)[2] - (L.TOP + L.FLOOR))


def in_room(m) -> bool:
    """Its centre (the origin, the middle of its base) inside the room, over the carpet."""
    x, y = (float(v) for v in np.asarray(m.origin, dtype=float)[:2])
    return bool(L.ROOM_X[0] <= x <= L.ROOM_X[1] and L.ROOM_Y[0] <= y <= L.ROOM_Y[1])


def wall_credit(xs: list[float]) -> tuple[int, float | None]:
    """How many movers one wall credits for one sense, from the centres' x (along the wall) of the movers set on it:
    none 0, one 1, two or more 2 if some pair is 190-305 mm apart, else 1. Also the spacing used (the pair nearest the
    band's middle when one is in band, else the pair nearest the band; None with fewer than two)."""
    if not xs:
        return 0, None
    if len(xs) == 1:
        return 1, None
    lo, hi = L.SPACING
    gaps = [abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    inside = [g for g in gaps if lo <= g <= hi]
    if inside:
        return L.PER_WALL, min(inside, key=lambda g: abs(g - (lo + hi) / 2))
    return 1, min(gaps, key=lambda g: min(abs(g - lo), abs(g - hi)))


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each mover's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {name: deque(maxlen=WINDOW + 1) for name in L.MOVERS}  # (origin, heading) per valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, m) -> bool:
        """The starting layout for this mover: upright on the carpet on its mark, heading as it starts."""
        x, y, h = L.START[name]
        on_mark = float(np.hypot(m.origin[0] - x, m.origin[1] - y)) < L.START_TOL
        hd = heading_rad(m)
        turn = abs(math.remainder(hd - h, 2 * math.pi)) if math.isfinite(hd) else math.pi
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(m) < 2.0 and abs(lowest(m)) < L.FLOOR_TOL)

    def path(self, name: str) -> float:
        """How far the mover moved over the last 0.5 s: its origin's path plus its turns about the vertical counted
        at its outline's corner (m). Infinite until half a second has been seen."""
        h = list(self.history[name])
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (p0, a0), (p1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.linalg.norm(p1 - p0)) + L.turn_radius() * turn
        return total

    def update(self, movers: dict) -> Result:
        self.steps += 1
        ms = {n: (movers or {}).get(n) for n in L.MOVERS}
        if not all(valid(m) for m in ms.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(movers)
            return self.last
        if all(self.history[n] for n in L.MOVERS) and all(self.at_start(n, m) for n, m in ms.items()):
            jump = max(float(np.linalg.norm(np.asarray(m.origin, dtype=float) - self.history[n][-1][0]))
                       for n, m in ms.items())
            if jump > L.RESET_JUMP:  # the simulator put them back: a new episode
                self.reset()
                self.steps = 1
        for n, m in ms.items():
            self.history[n].append((np.asarray(m.origin, dtype=float).copy(), heading_rad(m)))
        self.last = self.result(movers)
        return self.last

    def _placed(self, name: str, m) -> tuple[str | None, dict]:
        """None if placed (upright, on the carpet, centre inside the room, still), else why not."""
        tilt, low, path = tilt_deg(m), lowest(m), self.path(name)
        d = {"tilt_deg": round(tilt, 2), "lowest_mm": round(low * 1000, 1),
             "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
             "origin_mm": [round(float(c) * 1000, 1) for c in np.asarray(m.origin, dtype=float)]}
        if tilt > L.TILT_PLACED_DEG:
            return "not_upright", d
        if abs(low) > L.FLOOR_TOL:
            return "not_on_the_carpet", d
        if not in_room(m):
            return "not_inside_the_room", d
        if path >= L.STILL_PATH:
            return "moving", d
        return None, d

    def mover_result(self, name: str, m) -> dict:
        """One mover: placed or not, the wall it blows toward, the angle into it, the outlet's gap, its sense, and
        whether it is set (placed, 15-45 degrees into the wall, outlet within 15 mm of it)."""
        why, d = self._placed(name, m)
        wall, ang, sense = wall_of(m), angle_into_wall_deg(m), sense_of(m)
        gap = outlet_gap(m, wall)
        d.update(heading_deg=round(math.degrees(heading_rad(m)), 2) if math.isfinite(heading_rad(m)) else None,
                 wall=wall, angle_deg=round(ang, 2) if math.isfinite(ang) else None,
                 gap_mm=round(gap * 1000, 1) if math.isfinite(gap) else None, sense=sense)
        if why is None:
            if wall is None:
                why = "not_blowing_into_a_side_wall"
            elif not L.ANGLE_DEG[0] <= ang <= L.ANGLE_DEG[1] or sense is None:  # in band, sense is never None
                why = "angle_out_of_band"
            elif gap > L.GAP_MAX:
                why = "outlet_more_than_15_mm_from_the_wall"
        d.update(set=why is None, why=why)
        return d

    def result(self, movers) -> Result:
        ms = {n: (movers or {}).get(n) for n in L.MOVERS}
        if not all(valid(m) for m in ms.values()):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        per = {n: self.mover_result(n, m) for n, m in ms.items()}
        totals, walls = {}, {}
        for sense in SENSES:
            total, w = 0, {}
            for side in WALL_SIDES:
                names = [n for n, d in per.items() if d["set"] and d["wall"] == side and d["sense"] == sense]
                credit, spacing = wall_credit([float(np.asarray(ms[n].origin, dtype=float)[0]) for n in names])
                w[side] = {"set": names, "credit": credit,
                           "spacing_mm": round(spacing * 1000, 1) if spacing is not None else None}
                total += credit
            totals[sense], walls[sense] = total, w
        best = max(SENSES, key=lambda s: totals[s])  # ties: clockwise (the same count either way)
        score = L.CREDIT * totals[best]
        details = {"valid_state": True, "zero_reason": None if totals[best] else "no_mover_set",
                   "credited": totals, "best_sense": best if totals[best] else None, "walls": walls, **per}
        return Result(score, bool(score == 1.0), details)
