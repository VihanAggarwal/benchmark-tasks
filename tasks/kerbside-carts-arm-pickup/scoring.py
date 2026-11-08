"""The scorer, as plain functions of the two carts' states. NumPy only; no simulator needed.

The rule is what an automated side-loader needs to pick a cart up: the cart standing at the kerb (its front 0 to 30
mm behind the kerb face), its lid opening toward the street (front within 25 degrees of square to the kerb), off the
driveway, and 91 mm (3 ft at 1:10) clear of the other cart, the mailbox and its post, the hydrant and the parked car.
All of it is read from the carts' poses, as a top photo with a tape would read it. The only memory the scorer keeps
is each cart's last half second of positions, to tell whether it is still.

    Tracker.update(left, right) -> Result(score, success, details)

Each cart is anything with ``origin`` (the cart frame's origin: the middle of its body's bottom face), ``quat_xyzw``,
``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
mechkit's ``AssemblyState`` has. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0
for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
FIELDS = ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")


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


def valid(cart) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(cart, k), dtype=float).ravel() for k in FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(cart) -> float:
    """Angle between the cart's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(cart.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def heading_rad(cart) -> float:
    """Which way its front points seen from above (rad, counter-clockwise from +x); NaN if straight up or down."""
    f = rotation(cart.quat_xyzw)[:, 0]
    if math.hypot(f[0], f[1]) < 1e-9:
        return float("nan")
    return math.atan2(float(f[1]), float(f[0]))


def facing_err_deg(cart) -> float:
    """How far its front is turned from straight at the street (+x, square to the kerb), degrees 0..180."""
    h = heading_rad(cart)
    return abs(math.degrees(h)) if math.isfinite(h) else 180.0


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def footprint(cart) -> np.ndarray:
    """The cart's outline seen from above, (n, 2): the shadow of the box it spans (79 mm front to back with the
    handle and wheels, 58 mm across, up to the lid's top), turned and tilted as it is. Standing upright it is the
    79 x 58 mm rectangle."""
    r = rotation(cart.quat_xyzw)
    o = np.asarray(cart.origin, dtype=float)
    corners = [o + r @ np.array([u, v, z]) for u in (-L.FOOT_BACK, L.FOOT_FRONT) for v in (-L.FOOT_V, L.FOOT_V)
               for z in (0.0, L.OUTLINE_Z)]
    return _hull(np.array(corners)[:, :2])


def lowest(cart) -> float:
    """Height of its lowest point above the verge (m)."""
    return float(np.asarray(cart.low, dtype=float)[2] - L.TOP)


def setback(cart) -> float:
    """From the footprint's most forward point back to the kerb face (m; negative = past the kerb face)."""
    return float(L.KERB_FACE_X - footprint(cart)[:, 0].max())


# -- 2-D distances (top view) ------------------------------------------------------------------------------------------


def _rect_poly(x0, x1, y0, y1) -> np.ndarray:
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _point_segment(p, a, b) -> float:
    ab = b - a
    t = 0.0 if float(ab @ ab) == 0.0 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def _inside(poly: np.ndarray, p) -> bool:
    """Is p inside (or on) a counter-clockwise convex polygon?"""
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and o1 * o2 < 0 and o3 * o4 < 0


def poly_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Shortest distance between two convex polygons seen from above (0 if they touch or overlap)."""
    if any(_inside(q, v) for v in p) or any(_inside(p, v) for v in q):
        return 0.0
    ep = [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    eq = [(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
    if any(_segments_cross(a, b, c, d) for a, b in ep for c, d in eq):
        return 0.0
    best = math.inf
    for a, b in ep:
        for v in q:
            best = min(best, _point_segment(v, a, b))
    for a, b in eq:
        for v in p:
            best = min(best, _point_segment(v, a, b))
    return float(best)


def circle_distance(p: np.ndarray, centre, radius: float) -> float:
    """Shortest distance from a convex polygon to a circle seen from above (0 if they touch or overlap)."""
    c = np.asarray(centre, dtype=float)
    if _inside(p, c):
        return 0.0
    d = min(_point_segment(c, p[i], p[(i + 1) % len(p)]) for i in range(len(p)))
    return max(d - radius, 0.0)


MAILBOX_POLY = _rect_poly(*L.MAILBOX_RECT)
CAR_POLY = _rect_poly(*L.CAR_RECT)
DRIVE_POLY = _rect_poly(*L.DRIVE_RECT)


def clearances(cart, other) -> dict:
    """Top-view distance (m) from the cart's footprint to each thing it must stand 91 mm clear of."""
    fp = footprint(cart)
    return {
        "other_cart": poly_distance(fp, footprint(other)) if valid(other) else math.inf,
        "mailbox": poly_distance(fp, MAILBOX_POLY),
        "hydrant": circle_distance(fp, L.HYDRANT_XY, L.HYDRANT_R),
        "car": poly_distance(fp, CAR_POLY),
    }


def on_driveway(cart) -> bool:
    """Any part of its footprint over the driveway (where the kerb is dropped)."""
    return poly_distance(footprint(cart), DRIVE_POLY) == 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each cart's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {name: deque(maxlen=WINDOW + 1) for name in L.CARTS}  # (origin, heading) per valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, cart) -> bool:
        """The starting layout for this cart: upright on the verge on its mark, side-on as it starts."""
        x, y, h = L.START[name]
        on_mark = float(np.hypot(cart.origin[0] - x, cart.origin[1] - y)) < L.START_TOL
        turn = abs(math.remainder(heading_rad(cart) - h, 2 * math.pi))
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(cart) < 2.0 and abs(lowest(cart)) < L.FLOOR_TOL)

    def path(self, name: str) -> float:
        """How far the cart moved over the last 0.5 s: its origin's path plus its turns about the vertical counted
        at the footprint's corner (m). Infinite until half a second has been seen."""
        h = list(self.history[name])
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (p0, a0), (p1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.linalg.norm(p1 - p0)) + L.TURN_RADIUS * turn
        return total

    def update(self, left, right) -> Result:
        self.steps += 1
        carts = {"cart_left": left, "cart_right": right}
        if not all(valid(c) for c in carts.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(left, right)
            return self.last
        if all(self.history[n] for n in L.CARTS) and all(self.at_start(n, c) for n, c in carts.items()):
            jump = max(float(np.linalg.norm(np.asarray(c.origin, dtype=float) - self.history[n][-1][0]))
                       for n, c in carts.items())
            if jump > L.RESET_JUMP:  # the simulator put them back: a new episode
                self.reset()
                self.steps = 1
        for n, c in carts.items():
            self.history[n].append((np.asarray(c.origin, dtype=float).copy(), heading_rad(c)))
        self.last = self.result(left, right)
        return self.last

    def cart_result(self, name: str, cart, other) -> tuple[float, dict]:
        """One cart's credit (0, 0.25 or 0.5) and what it was read from."""
        tilt, low, path = tilt_deg(cart), lowest(cart), self.path(name)
        face, sb, drive = facing_err_deg(cart), setback(cart), on_driveway(cart)
        clear = clearances(cart, other)
        nearest = min(clear, key=clear.get)
        upright = tilt <= L.TILT_STAND_DEG
        on_verge = abs(low) <= L.FLOOR_TOL
        still = path < L.STILL_PATH
        standing = upright and on_verge and still
        facing = face <= L.FACING_DEG
        at_kerb = -L.KERB_TOL <= sb <= L.SETBACK_MAX
        is_clear = clear[nearest] >= L.CLEAR
        if not upright:
            why = "not_upright"
        elif not on_verge:
            why = "not_on_the_verge"
        elif not still:
            why = "moving"
        elif drive:
            why = "on_the_driveway"
        elif not facing:
            why = "not_facing_the_street"
        elif not at_kerb:
            why = "not_at_the_kerb"
        elif not is_clear:
            why = "too_close_to_" + nearest
        else:
            why = None
        if why is None:
            credit = 0.5
        elif why.startswith("too_close_to_"):
            credit = 0.25
        else:
            credit = 0.0
        details = {
            "credit": credit,
            "why": why,
            "standing": standing,
            "facing_err_deg": round(face, 2),
            "setback_mm": round(sb * 1000, 1),
            "on_driveway": drive,
            "clear_mm": round(clear[nearest] * 1000, 1) if math.isfinite(clear[nearest]) else None,
            "nearest": nearest,
            "origin_mm": [round(float(v) * 1000, 1) for v in np.asarray(cart.origin, dtype=float)],
            "tilt_deg": round(tilt, 2),
            "lowest_mm": round(low * 1000, 1),
            "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
        }
        return credit, details

    def result(self, left, right) -> Result:
        carts = {"cart_left": left, "cart_right": right}
        if not all(valid(c) for c in carts.values()):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        reason = None
        if any(tilt_deg(c) > L.TIPPED_DEG for c in carts.values()):
            reason = "a_cart_knocked_over"
        elif any(setback(c) < -L.KERB_TOL for c in carts.values()):
            reason = "a_cart_on_the_kerb_or_in_the_road"
        details = {"valid_state": True, "zero_reason": reason}
        total = 0.0
        for n, c in carts.items():
            other = carts["cart_right" if n == "cart_left" else "cart_left"]
            credit, d = self.cart_result(n, c, other)
            details[n] = d
            total += credit
        score = 0.0 if reason is not None else total
        return Result(score, bool(score == 1.0), details)
