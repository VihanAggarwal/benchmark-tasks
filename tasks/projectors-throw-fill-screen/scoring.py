"""The scorer, as plain functions of the two projectors' states. NumPy only; no simulator needed.

The rule is the AV technician's set-up of a fixed-lens projector: the throw distance (from the front of the lens to
the screen) is the projector's throw ratio times the picture width, so to fill a 1.6 m screen (160 mm here) the 1.5:1
projector stands 240 mm from it and the 1.0:1 projector 160 mm; each must face its screen square, its lens on the
screen's centre line. All of it is read from the projectors' poses, as a tape from the wall, a set square and a top
photo would read it, together with whether the grippers have let go of them. The only memory the scorer keeps is each
projector's last half second of positions, to tell whether it is still.

    Tracker.update(black, white, arms) -> Result(score, success, details)

Each projector is anything with ``origin`` (the projector frame's origin: the middle of its body's bottom face),
``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding
box), as mechkit's ``AssemblyState`` has. A state with a non-finite number or a zero quaternion (a simulator blow-up),
or a missing projector, scores 0 for that step and leaves the tracker as it was. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then no projector counts as let go for that step.

A gripper has let go of a projector when it is at least 90 % open (a gripper that open holds nothing) or its 'finger
zone' is more than 10 mm from the projector's two boxes (body and lens). rigstate gives no gripper heading, so the
zone is round: every point within a radius of the 38 mm segment from the point between the finger tips back up the
gripper, the radius reaching a pad's outer corner (half the finger gap plus the 8 mm pad, and half its 27 mm width).

A projector's picture width is its throw distance divided by its own throw ratio; the throw distance is how far the
centre of its lens's front face is in front of the screens' plane (x 0.468), along x. Either projector may serve
either screen (the screens are the same width): each is judged on the distance its own ratio needs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
FIELDS = ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")
ARMS = ("left", "right")
LENS_LOCAL = np.array([L.LENS_FRONT_U, 0.0, L.LENS_Z])  # the centre of the lens's front face, projector frame


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


def valid(p) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(p, k), dtype=float).ravel() for k in FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(q.size and np.all(np.isfinite(q)) for q in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(p) -> float:
    """Angle between the projector's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(p.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def heading_rad(p) -> float:
    """Which way its lens points seen from above (rad, counter-clockwise from +x); NaN if straight up or down."""
    f = rotation(p.quat_xyzw)[:, 0]
    if math.hypot(f[0], f[1]) < 1e-9:
        return float("nan")
    return math.atan2(float(f[1]), float(f[0]))


def square_err_deg(p) -> float:
    """How far its lens's axis is turned from straight at the wall (+x, square to the screens), degrees 0..180."""
    h = heading_rad(p)
    return abs(math.degrees(h)) if math.isfinite(h) else 180.0


def lens_point(p) -> np.ndarray:
    """The centre of the lens's front face (world)."""
    return np.asarray(p.origin, dtype=float) + rotation(p.quat_xyzw) @ LENS_LOCAL


def throw_distance(p) -> float:
    """From the centre of the lens's front face to the screens' plane, along x (m; negative = past the wall's face)."""
    return float(L.SCREEN_FACE_X - lens_point(p)[0])


def picture_width(name: str, p) -> float:
    """The picture's width on the wall: throw distance / throw ratio (m)."""
    return throw_distance(p) / L.RATIO[name]


def width_err(name: str, p) -> float:
    """The picture's width against the screen's 160 mm, as a fraction (+ = wider than the screen)."""
    return picture_width(name, p) / L.SCREEN_W - 1.0


def read_arms(arms) -> list[tuple[np.ndarray, np.ndarray, float]] | None:
    """Both grippers as (tcp, tips, opening), or None when the arms are unreadable (None, a side missing, or a
    number missing or not finite)."""
    if not isinstance(arms, dict):
        return None
    out = []
    try:
        for side in ARMS:
            a = arms[side]
            tcp = np.asarray(a.tcp, dtype=float).ravel()
            tips = np.asarray(a.tips, dtype=float).ravel()
            opening = float(a.opening)
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return None
            if not math.isfinite(opening):
                return None
            out.append((tcp, tips, opening))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return out


def shape_distances(points, p) -> np.ndarray:
    """Distance from each world point to the nearer of the projector's two boxes (body and lens); 0 inside one."""
    r = rotation(p.quat_xyzw)
    local = (np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(p.origin, dtype=float)) @ r  # r.T @ (x - o)
    best = np.full(len(local), math.inf)
    for _, centre, size, _ in L.SHAPES:
        q = np.abs(local - np.asarray(centre, dtype=float)) - np.asarray(size, dtype=float) / 2
        best = np.minimum(best, np.linalg.norm(np.maximum(q, 0.0), axis=1))
    return best


def zone_radius(opening: float) -> float:
    """From the gripper's axis to a finger pad's outer corner (m): half the gap plus the pad's thickness, and half
    its width across. 56.6 mm fully open, 15.7 mm closed."""
    gap = float(np.clip(opening, 0.0, 1.0)) * L.FULL_GAP
    return math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH)


def zone_gap(arm, p) -> float:
    """How far one gripper's finger zone is from the projector (m; negative: they overlap). The zone's axis runs
    from the finger-tip point 38 mm back toward the TCP."""
    tcp, tips, opening = arm
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, L.ZONE_SAMPLES), u)
    return float(shape_distances(pts, p).min()) - zone_radius(opening)


def let_go(arm, p) -> bool:
    """This gripper has let go of the projector: at least 90 % open, or its finger zone more than 10 mm away."""
    return bool(arm[2] >= L.GRIPPER_OPEN - L.EPS or zone_gap(arm, p) > L.LET_GO_GAP + L.EPS)


def released(read, p) -> bool:
    """Both grippers have let go of the projector (``read`` from read_arms; None, unreadable arms: not let go)."""
    return bool(read is not None and all(let_go(a, p) for a in read))


def lowest(p) -> float:
    """Height of its lowest point above the projector table's top (m)."""
    return float(np.asarray(p.low, dtype=float)[2] - L.TABLE_TOP_W)


def over_table(p) -> bool:
    """Its origin (the middle of its base), seen from above, is over the projector table."""
    x, y = float(p.origin[0]), float(p.origin[1])
    return bool(L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1])


def nearest_screen(p) -> tuple[str, float]:
    """The screen whose centre line the lens's centre is nearest to, and how far off it it is sideways (m)."""
    y = float(lens_point(p)[1])
    best = min(L.SCREENS, key=lambda s: abs(y - L.SCREEN_Y[s]))
    return best, abs(y - L.SCREEN_Y[best])


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each projector's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {name: deque(maxlen=WINDOW + 1) for name in L.PROJECTORS}  # (origin, heading) per valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, p) -> bool:
        """The starting layout for this projector: upright on the table on its mark, turned sideways as it starts."""
        x, y, h = L.START[name]
        on_mark = float(np.hypot(p.origin[0] - x, p.origin[1] - y)) < L.START_TOL
        hd = heading_rad(p)
        turn = abs(math.remainder(hd - h, 2 * math.pi)) if math.isfinite(hd) else math.pi
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(p) < 2.0 and abs(lowest(p)) < L.TABLE_TOL)

    def path(self, name: str) -> float:
        """How far the projector moved over the last 0.5 s: its origin's path plus its turns about the vertical
        counted at the outline's far corner (m). Infinite until half a second has been seen."""
        h = list(self.history[name])
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (p0, a0), (p1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.linalg.norm(p1 - p0)) + L.TURN_RADIUS * turn
        return total

    @staticmethod
    def _states(black, white) -> dict | None:
        states = {"projector_black": black, "projector_white": white}
        return states if all(valid(s) for s in states.values()) else None

    def update(self, black, white, arms) -> Result:
        self.steps += 1
        states = self._states(black, white)
        if states is None:  # a blow-up or a missing projector: score 0 for this step, keep everything as it was
            self.last = self.result(black, white, arms)
            return self.last
        if all(self.history[n] for n in L.PROJECTORS) and all(self.at_start(n, s) for n, s in states.items()):
            jump = max(float(np.linalg.norm(np.asarray(s.origin, dtype=float) - self.history[n][-1][0]))
                       for n, s in states.items())
            if jump > L.RESET_JUMP:  # the simulator put them back: a new episode
                self.reset()
                self.steps = 1
        for n, s in states.items():
            self.history[n].append((np.asarray(s.origin, dtype=float).copy(), heading_rad(s)))
        self.last = self.result(black, white, arms)
        return self.last

    def projector_result(self, name: str, p, read=None) -> tuple[float, str | None, dict]:
        """One projector's credit (0, 0.25 or 0.5), the screen it is set up for (None if none) and what it was read
        from. ``read`` is read_arms(arms): None (unreadable arms) means not let go."""
        tilt, low, path = tilt_deg(p), lowest(p), self.path(name)
        free = released(read, p)
        sq, throw = square_err_deg(p), throw_distance(p)
        err = width_err(name, p)
        screen, off = nearest_screen(p)
        upright = tilt <= L.TILT_MAX_DEG + L.EPS
        on_table = over_table(p) and abs(low) <= L.TABLE_TOL + L.EPS
        still = path < L.STILL_PATH
        aimed = sq <= L.AIM_DEG + L.EPS
        at_line = off <= L.CENTRE_LOOSE + L.EPS
        loose = abs(err) <= L.LOOSE_TOL + L.EPS
        full = (abs(err) <= L.FULL_TOL + L.EPS and sq <= L.SQUARE_DEG + L.EPS and off <= L.CENTRE_TIGHT + L.EPS)
        if not upright:
            why = "not_upright"
        elif not on_table:
            why = "not_on_the_table"
        elif not still:
            why = "moving"
        elif not free:
            why = "not_let_go"
        elif not aimed:
            why = "lens_not_toward_the_wall"
        elif not at_line:
            why = "not_on_a_centre_line"
        elif not loose:
            why = "picture_too_wide" if err > 0 else "picture_too_narrow"
        elif not full:
            why = ("picture_off_by_over_5pc" if abs(err) > L.FULL_TOL + L.EPS
                   else "off_square" if sq > L.SQUARE_DEG + L.EPS else "off_centre")
        else:
            why = None
        if why is None:
            credit = 0.5
        elif why in ("picture_off_by_over_5pc", "off_square", "off_centre"):
            credit = 0.25
        else:
            credit = 0.0
        details = {
            "credit": credit,
            "why": why,
            "screen": screen if credit > 0 else None,
            "throw_mm": round(throw * 1000, 1),
            "picture_mm": round(picture_width(name, p) * 1000, 1),
            "width_err_pc": round(err * 100, 2),
            "square_err_deg": round(sq, 2),
            "nearest_screen": screen,
            "centre_off_mm": round(off * 1000, 1),
            "origin_mm": [round(float(v) * 1000, 1) for v in np.asarray(p.origin, dtype=float)],
            "tilt_deg": round(tilt, 2),
            "lowest_mm": round(low * 1000, 1),
            "on_table": on_table,
            "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
            "let_go": free,
            "zone_gap_mm": ({a: round(zone_gap(arm, p) * 1000, 1) for a, arm in zip(ARMS, read)}
                            if read is not None else None),
        }
        return credit, (screen if credit > 0 else None), details

    def result(self, black, white, arms) -> Result:
        states = self._states(black, white)
        if states is None:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        read = read_arms(arms)
        details: dict = {"valid_state": True, "zero_reason": None, "arms_read": read is not None,
                         "openings": ({a: round(arm[2], 3) for a, arm in zip(ARMS, read)} if read else None)}
        per_screen = {s: (0.0, None) for s in L.SCREENS}  # each screen counts its best projector, once
        for n, s in states.items():
            credit, screen, d = self.projector_result(n, s, read)
            details[n] = d
            if screen is not None and credit > per_screen[screen][0]:
                per_screen[screen] = (credit, n)
        for s, (credit, n) in per_screen.items():
            details[s] = {"credit": credit, "projector": n}
        for n in L.PROJECTORS:  # a second projector on a screen already served earns nothing
            scr = details[n]["screen"]
            details[n]["counted"] = bool(scr is not None and per_screen[scr][1] == n)
        score = float(sum(c for c, _ in per_screen.values()))
        return Result(score, bool(score == 1.0), details)
