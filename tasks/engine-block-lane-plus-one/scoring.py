"""The scorer, as plain functions of the two vehicles' states. NumPy only; no simulator needed.

The rule is the traffic-incident set-up: the fire engine parked upstream of the crashed car, angled 30 to 60
degrees to the lanes so that it blocks the crash lane and the next one ("lane + 1"), its pump-panel side facing
downstream (toward the car), 50 to 125 mm (4 to 10 m) short of the car; the ambulance parked downstream, straight in
either lane with its rear doors toward the car, 50 to 125 mm beyond it. All of it is read from the two vehicles'
poses, as a top photo with a tape and a protractor would read it. The only memory the scorer keeps is each vehicle's
last half second of positions, to tell whether it is still.

    Tracker.update(engine, ambulance) -> Result(score, success, details)

Each vehicle is anything with ``origin`` (the vehicle frame's origin: the middle of its length, on its centreline,
at the bottom), ``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high``
(world-aligned bounding box), as mechkit's ``AssemblyState`` has. A state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
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


def valid(v) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(v, k), dtype=float).ravel() for k in FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(v) -> float:
    """Angle between the vehicle's up axis and the vertical (degrees, 0..180: on its roof is 180)."""
    zz = rotation(v.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _flat(axis) -> tuple[float, float] | None:
    """A body axis seen from above, as a unit 2-D vector; None if it points straight up or down."""
    n = math.hypot(float(axis[0]), float(axis[1]))
    if n < 1e-9:
        return None
    return float(axis[0]) / n, float(axis[1]) / n


def heading_rad(v) -> float:
    """Which way its front points seen from above (rad, counter-clockwise from +x); NaN if straight up or down."""
    f = _flat(rotation(v.quat_xyzw)[:, 0])
    return math.atan2(f[1], f[0]) if f is not None else float("nan")


def angle_to_lanes_deg(v) -> float:
    """The angle between its long axis and the lanes (y), seen from above: 0 in line with the traffic, 90 square
    across the road. NaN if its long axis points straight up or down."""
    f = _flat(rotation(v.quat_xyzw)[:, 0])
    if f is None:
        return float("nan")
    return math.degrees(math.atan2(abs(f[0]), abs(f[1])))


def left_side_y(v) -> float:
    """The y part of its left side's outward direction seen from above (the fire engine's pump-panel side):
    negative = facing downstream (toward the car); NaN if that side faces straight up or down."""
    s = _flat(rotation(v.quat_xyzw)[:, 1])
    return s[1] if s is not None else float("nan")


def front_off_downstream_deg(v) -> float:
    """How far its front is turned from straight downstream (-y), degrees 0..180 (180 = facing upstream)."""
    h = heading_rad(v)
    if not math.isfinite(h):
        return 180.0
    return abs(math.degrees(math.remainder(h + math.pi / 2, 2 * math.pi)))


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


def footprint(name: str, v) -> np.ndarray:
    """The vehicle's outline seen from above, (n, 2): the shadow of the box it spans (the fire engine 125 mm long
    and 32 mm across at its pump panel, the ambulance 88 x 30 mm, up to the light bar's top), turned and tilted as
    it is. Parked upright it is that rectangle."""
    back, front, right, left, top = L.OUTLINE[name]
    r = rotation(v.quat_xyzw)
    o = np.asarray(v.origin, dtype=float)
    corners = [o + r @ np.array([u, w, z]) for u in (-back, front) for w in (-right, left) for z in (0.0, top)]
    return _hull(np.array(corners)[:, :2])


def lowest(v) -> float:
    """Height of its lowest point above the road surface (m)."""
    return float(np.asarray(v.low, dtype=float)[2] - (L.TOP + L.ROAD_TOP))


# -- 2-D distances (top view) ------------------------------------------------------------------------------------------


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
    return o1 * o2 < 0 and o3 * o4 < 0


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


CAR_POLY = _hull(np.array(L.car_corners()))


def engine_measures(v) -> dict:
    """What a tape and a protractor read off the fire engine (m and degrees)."""
    fp = footprint("fire_engine", v)
    x_lo, x_hi = float(fp[:, 0].min()), float(fp[:, 0].max())
    return {
        "buffer": float(fp[:, 1].min()) - L.CAR_UP_Y,  # from its most downstream point to the car's upstream end
        "x_lo": x_lo,
        "x_hi": x_hi,
        "crash_lane_cover": max(0.0, min(x_hi, L.LANE1[1]) - max(x_lo, L.LANE1[0])),
        "angle": angle_to_lanes_deg(v),
        "panel_y": left_side_y(v),
    }


def ambulance_measures(v) -> dict:
    fp = footprint("ambulance", v)
    return {
        "gap": L.CAR_DOWN_Y - float(fp[:, 1].max()),  # from the car's downstream end to its most upstream point
        "centre_x": float(np.asarray(v.origin, dtype=float)[0]),
        "front_off": front_off_downstream_deg(v),
    }


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each vehicle's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {name: deque(maxlen=WINDOW + 1) for name in L.VEHICLES}  # (origin, heading) per valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, v) -> bool:
        """The starting layout for this vehicle: upright on the road on its mark, heading as it starts."""
        x, y, h = L.START[name]
        on_mark = float(np.hypot(v.origin[0] - x, v.origin[1] - y)) < L.START_TOL
        hd = heading_rad(v)
        turn = abs(math.remainder(hd - h, 2 * math.pi)) if math.isfinite(hd) else math.pi
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(v) < 2.0 and abs(lowest(v)) < L.FLOOR_TOL)

    def path(self, name: str) -> float:
        """How far the vehicle moved over the last 0.5 s: its origin's path plus its turns about the vertical counted
        at its outline's corner (m). Infinite until half a second has been seen."""
        h = list(self.history[name])
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (p0, a0), (p1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.linalg.norm(p1 - p0)) + L.turn_radius(name) * turn
        return total

    def update(self, engine, ambulance) -> Result:
        self.steps += 1
        vs = {"fire_engine": engine, "ambulance": ambulance}
        if not all(valid(v) for v in vs.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(engine, ambulance)
            return self.last
        if all(self.history[n] for n in L.VEHICLES) and all(self.at_start(n, v) for n, v in vs.items()):
            jump = max(float(np.linalg.norm(np.asarray(v.origin, dtype=float) - self.history[n][-1][0]))
                       for n, v in vs.items())
            if jump > L.RESET_JUMP:  # the simulator put them back: a new episode
                self.reset()
                self.steps = 1
        for n, v in vs.items():
            self.history[n].append((np.asarray(v.origin, dtype=float).copy(), heading_rad(v)))
        self.last = self.result(engine, ambulance)
        return self.last

    def _parked(self, name: str, v) -> tuple[str | None, dict]:
        """None if parked (upright, on the road surface, centre over the road, still), else why not."""
        tilt, low, path = tilt_deg(v), lowest(v), self.path(name)
        cx = float(np.asarray(v.origin, dtype=float)[0])
        d = {"tilt_deg": round(tilt, 2), "lowest_mm": round(low * 1000, 1),
             "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
             "origin_mm": [round(float(c) * 1000, 1) for c in np.asarray(v.origin, dtype=float)]}
        if tilt > L.TILT_PARK_DEG:
            return "not_upright", d
        if abs(low) > L.FLOOR_TOL:
            return "not_on_the_road_surface", d
        if not L.ROAD_CENTRE_X[0] <= cx <= L.ROAD_CENTRE_X[1]:
            return "not_over_the_road", d
        if path >= L.STILL_PATH:
            return "moving", d
        return None, d

    def engine_result(self, v) -> tuple[float, dict]:
        """The fire engine's credit (0, 0.25 or 0.5) and what it was read from."""
        why, d = self._parked("fire_engine", v)
        m = engine_measures(v)
        ang, pan = m["angle"], m["panel_y"]
        d.update(buffer_mm=round(m["buffer"] * 1000, 1), x_lo_mm=round(m["x_lo"] * 1000, 1),
                 x_hi_mm=round(m["x_hi"] * 1000, 1), crash_lane_cover_mm=round(m["crash_lane_cover"] * 1000, 1),
                 angle_deg=round(ang, 2) if math.isfinite(ang) else None,
                 panel_y=round(pan, 3) if math.isfinite(pan) else None)
        if why is None:
            if m["buffer"] <= 0.0:
                why = "not_upstream"
            elif m["buffer"] > L.FAR_MAX:
                why = "more_than_250_mm_back"
            elif m["crash_lane_cover"] < L.CRASH_LANE_COVER:
                why = "not_blocking_the_crash_lane"
        if why is not None:
            credit = 0.0
        else:
            if not (math.isfinite(ang) and L.ANGLE_DEG[0] <= ang <= L.ANGLE_DEG[1]):
                why = "angle_out_of_band"
            elif not (m["x_lo"] <= L.BOTH_LANES[0] and m["x_hi"] >= L.BOTH_LANES[1]):
                why = "not_across_both_lanes"
            elif not (math.isfinite(pan) and pan < 0.0):
                why = "panel_toward_traffic"
            elif m["buffer"] < L.BAND[0]:
                why = "buffer_under_50_mm"
            elif m["buffer"] > L.BAND[1]:
                why = "buffer_over_125_mm"
            credit = 0.5 if why is None else 0.25
        d.update(credit=credit, why=why)
        return credit, d

    def ambulance_result(self, v) -> tuple[float, dict]:
        """The ambulance's credit (0, 0.25 or 0.5) and what it was read from."""
        why, d = self._parked("ambulance", v)
        m = ambulance_measures(v)
        d.update(gap_mm=round(m["gap"] * 1000, 1), centre_x_mm=round(m["centre_x"] * 1000, 1),
                 front_off_deg=round(m["front_off"], 2))
        if why is None:
            if m["gap"] <= 0.0:
                why = "not_downstream"
            elif m["gap"] > L.FAR_MAX:
                why = "more_than_250_mm_on"
            elif not L.LANES[0] <= m["centre_x"] <= L.LANES[1]:
                why = "not_in_a_lane"
        if why is not None:
            credit = 0.0
        else:
            if m["front_off"] > L.STRAIGHT_DEG:
                why = "not_straight_rear_to_the_car"
            elif m["gap"] < L.BAND[0]:
                why = "gap_under_50_mm"
            elif m["gap"] > L.BAND[1]:
                why = "gap_over_125_mm"
            credit = 0.5 if why is None else 0.25
        d.update(credit=credit, why=why)
        return credit, d

    def result(self, engine, ambulance) -> Result:
        vs = {"fire_engine": engine, "ambulance": ambulance}
        if not all(valid(v) for v in vs.values()):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        fps = {n: footprint(n, v) for n, v in vs.items()}
        to_car = {n: poly_distance(fp, CAR_POLY) for n, fp in fps.items()}
        between = poly_distance(fps["fire_engine"], fps["ambulance"])
        reason = None
        if any(tilt_deg(v) > L.TIPPED_DEG for v in vs.values()):
            reason = "a_vehicle_on_its_side_or_roof"
        elif min(to_car.values()) <= L.TOUCH_GAP:
            reason = "a_vehicle_touching_the_car"
        elif between <= L.TOUCH_GAP:
            reason = "the_vehicles_touching"
        details = {"valid_state": True, "zero_reason": reason,
                   "to_car_mm": {n: round(d * 1000, 1) for n, d in to_car.items()},
                   "between_mm": round(between * 1000, 1)}
        ce, de = self.engine_result(engine)
        ca, da = self.ambulance_result(ambulance)
        details["fire_engine"], details["ambulance"] = de, da
        score = 0.0 if reason is not None else ce + ca
        return Result(score, bool(score == 1.0), details)
