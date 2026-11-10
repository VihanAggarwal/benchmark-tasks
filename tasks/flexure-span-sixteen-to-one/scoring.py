"""The scorer, as plain functions of the specimen's and the two support blocks' states and the grippers' state.
NumPy only; no simulator needed.

The score is read from the end state, as a technician checks a three-point bend set-up before starting a flexural
test: do both supports stand on the fixture base with their rollers parallel to the centre line; how far apart are
the roller axes (the span, read with a steel rule) and how far is the point midway between them from the red centre
line (half the difference of the two rule readings from the line to each roller); does the specimen lie flat across
both rollers; how far is its middle from the span's middle (half the difference of its two overhangs) and is it
square to the rollers. Every tier also needs everything let go and at rest. The scorer keeps only the last half
second of the specimen's end points and the rollers' centres (to tell a set-up at rest from one still moving) and is
fed one observation per control step by ``task.success()``:

    Tracker.update(specimen, supports, arms) -> Result(score, success, details)

``specimen`` is the bar's ObjectState: ``pos`` (its centre), ``quat_xyzw`` (local y along its length, local z through
its 10 mm thickness), ``lin_vel``, ``ang_vel``. ``supports`` is the pair of support blocks' states as mechkit reports
them (``AssemblyState``): ``origin`` (the middle of the block's underside) and ``quat_xyzw`` (local x along the
roller, local z up), in a fixed order (the task passes them by name); the scorer tells the left and right support
apart by where their rollers are. A state with a non-finite number, a zero quaternion or a missing field (a simulator
blow-up, or a support not built as an assembly) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the hands-off condition
is taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so
a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a radius of the
38 mm segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer corner (half
the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).

The specimen is the same either face up and either end first, and a support is the same either way round (its roller
runs its full length), so those are not told apart.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

EPS = 1e-9  # a limit is met when the measure is within it to a nanometre (so exact edges are not lost to rounding)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _finite(obj, names: tuple[str, ...], sizes: tuple[int, ...]) -> bool:
    try:
        parts = [np.asarray(getattr(obj, k), dtype=float).ravel() for k in names]
    except (AttributeError, TypeError, ValueError):
        return False
    if any(p.size != n or not np.all(np.isfinite(p)) for p, n in zip(parts, sizes)):
        return False
    return bool(np.linalg.norm(parts[names.index("quat_xyzw")]) > 0.5)


def sp_valid(sp) -> bool:
    """The specimen's centre and orientation finite, the quaternion not degenerate."""
    return _finite(sp, ("pos", "quat_xyzw"), (3, 4))


def sup_valid(sup) -> bool:
    """A support's base-frame origin, orientation and box centre finite, the quaternion not degenerate."""
    return _finite(sup, ("origin", "quat_xyzw", "pos"), (3, 4, 3))


def all_valid(sp, sups) -> bool:
    try:
        return sp_valid(sp) and len(sups) == 2 and all(sup_valid(s) for s in sups)
    except TypeError:
        return False


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def _horizontal(v) -> np.ndarray | None:
    """Unit horizontal projection of a direction (None when it is vertical)."""
    h = np.array([float(v[0]), float(v[1])])
    n = float(np.linalg.norm(h))
    return None if n < 1e-9 else h / n


def _angle_from(v, axis: int) -> float:
    """Angle (0..90 degrees) between a direction seen from above and the world's x (axis 0) or y (axis 1), either
    way round; 90 when the direction is vertical."""
    h = _horizontal(v)
    if h is None:
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(h[axis])))))


# -- the specimen -----------------------------------------------------------------------------------------------------
def sp_ends(sp) -> tuple[np.ndarray, np.ndarray]:
    """The centres of the bar's two end faces (its local -y end, then its +y end), in the world."""
    c, r = np.asarray(sp.pos, dtype=float), rotation(sp.quat_xyzw)
    return c - r[:, 1] * (L.SP_LEN / 2), c + r[:, 1] * (L.SP_LEN / 2)


def sp_tilt_deg(sp) -> float:
    """Angle between the vertical and the bar's 10 mm thickness (local z), either face down (0..90 degrees): 0 lying
    flat on a 20 x 210 mm face; 90 on its edge or standing on an end."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(sp.quat_xyzw)[2, 2]), 0.0, 1.0))))


def sp_heading_err_deg(sp) -> float:
    """Angle between the bar's length seen from above and y (the line across the centre line), 0..90 degrees."""
    return _angle_from(rotation(sp.quat_xyzw)[:, 1], 1)


def underside_line(sp) -> tuple[np.ndarray, np.ndarray]:
    """(a point, the unit direction) of the centre line of the bar's underside, whichever face is down."""
    c, r = np.asarray(sp.pos, dtype=float), rotation(sp.quat_xyzw)
    down = -r[:, 2] if r[2, 2] >= 0.0 else r[:, 2]
    return c + down * (L.SP_DEPTH / 2), r[:, 1]


# -- the supports -----------------------------------------------------------------------------------------------------
def roller_centre(sup) -> np.ndarray:
    """The middle of the support's roller axis, in the world."""
    return np.asarray(sup.origin, dtype=float) + rotation(sup.quat_xyzw) @ np.array([0.0, 0.0, L.ROLLER_Z])


def roller_axis(sup) -> np.ndarray:
    return rotation(sup.quat_xyzw)[:, 0]


def sup_tilt_deg(sup) -> float:
    """Angle between the vertical and the support's own vertical axis (0 upright .. 180 upside down)."""
    return float(np.degrees(np.arccos(np.clip(rotation(sup.quat_xyzw)[2, 2], -1.0, 1.0))))


def roller_heading_err_deg(sup) -> float:
    """Angle between the roller seen from above and x (the centre line), either way round (0..90 degrees)."""
    return _angle_from(roller_axis(sup), 0)


def standing(sup) -> tuple[bool, str, dict]:
    """Is the support standing on the base (upright, the middle of its underside on the base's top and inside its
    outline) with its roller parallel to the centre line? (ok, why not, diagnostics)."""
    o = np.asarray(sup.origin, dtype=float)
    tilt, head = sup_tilt_deg(sup), roller_heading_err_deg(sup)
    height = float(o[2] - L.BASE_TOP_W)
    inside = bool(L.BASE_X0 - EPS <= o[0] <= L.BASE_X1 + EPS and abs(o[1]) <= L.BASE_HALF_Y + EPS)
    d = {"tilt_deg": round(tilt, 2), "roller_heading_deg": round(head, 2), "height_mm": round(height * 1000, 2),
         "inside_base": inside}
    if tilt > L.SUP_TILT_MAX_DEG + EPS or abs(height) > L.SUP_ON_BASE_TOL + EPS or not inside:
        return False, "support_not_standing_on_base", d
    if head > L.ROLLER_HEADING_MAX_DEG + EPS:
        return False, "roller_not_parallel_to_centre_line", d
    return True, "", d


def left_right(sups) -> tuple:
    """The two supports as (left, right): the one whose roller is further toward +y first."""
    a, b = sups
    return (a, b) if roller_centre(a)[1] >= roller_centre(b)[1] else (b, a)


def span_and_offset(sups) -> tuple[float, float]:
    """(span s, offset e): the distance along y between the two roller axes (left minus right, so never negative),
    and how far the point midway between them is from the centre line (m). On a real rig: a steel rule between the
    roller axes, and half the difference of the two readings from the red line to each roller's axis."""
    lft, rgt = left_right(sups)
    yl, yr = float(roller_centre(lft)[1]), float(roller_centre(rgt)[1])
    return yl - yr, abs((yl + yr) / 2 - L.MID_Y)


# -- the specimen on the rollers ---------------------------------------------------------------------------------------
def crossing(sp, sup) -> dict | None:
    """Where the centre line of the bar's underside passes over the roller's axis, seen from above: ``tau`` its
    distance along the bar from the bar's middle, ``along`` its distance along the roller from the roller's middle,
    ``gap`` the bar's underside there above the roller's top (m), ``square_err_deg`` how far the bar is from square
    to the roller. None when the bar runs parallel to the roller."""
    p0, d = underside_line(sp)
    rc, a = roller_centre(sup), roller_axis(sup)
    m = np.array([[d[0], -a[0]], [d[1], -a[1]]])
    if abs(float(np.linalg.det(m))) < 1e-9:
        return None
    tau, along = np.linalg.solve(m, np.array([rc[0] - p0[0], rc[1] - p0[1]]))
    under = float(p0[2] + tau * d[2])
    top = float(rc[2] + along * a[2] + L.ROLLER_R)
    hd, ha = _horizontal(d), _horizontal(a)
    if hd is None or ha is None:
        sq = 90.0
    else:
        sq = 90.0 - float(np.degrees(math.acos(min(1.0, abs(float(hd @ ha))))))
    return {"tau": float(tau), "along": float(along), "gap": under - top, "square_err_deg": sq}


def on_rollers(sp, sups) -> tuple[bool, str, dict, tuple[float, float] | None]:
    """Is the bar lying flat across both rollers, resting on them? (ok, why not, diagnostics, (c, square)).
    Diagnostics, at each roller (left, then right): the bar's square error, the distance along the roller, how far in
    from the bar's end and the underside gap. c is the bar's offset from the span's middle measured along the bar (its
    middle to the point midway between the two crossings: half the difference of its two overhangs), square the worse
    of its two square errors (degrees); None when it does not cross both rollers."""
    tilt = sp_tilt_deg(sp)
    d = {"sp_tilt_deg": round(tilt, 2), "sp_heading_err_deg": round(sp_heading_err_deg(sp), 2),
         "square_err_deg": [], "along_roller_mm": [], "in_from_end_mm": [], "underside_mm": [], "sp_offset_mm": None}
    crosses = [crossing(sp, s) for s in left_right(sups)]
    for c in crosses:
        if c is None:
            for k in ("square_err_deg", "along_roller_mm", "in_from_end_mm", "underside_mm"):
                d[k].append(None)
            continue
        d["square_err_deg"].append(round(c["square_err_deg"], 2))
        d["along_roller_mm"].append(round(c["along"] * 1000, 2))
        d["in_from_end_mm"].append(round((L.SP_LEN / 2 - abs(c["tau"])) * 1000, 2))
        d["underside_mm"].append(round(c["gap"] * 1000, 2))
    raw = None
    if all(c is not None for c in crosses):
        raw = (abs(crosses[0]["tau"] + crosses[1]["tau"]) / 2, max(c["square_err_deg"] for c in crosses))
        d["sp_offset_mm"] = round(raw[0] * 1000, 2)
    if tilt > L.SP_TILT_MAX_DEG + EPS:
        return False, "specimen_not_flat", d, raw
    if raw is None or raw[1] > L.SP_ACROSS_MAX_DEG + EPS:
        return False, "specimen_not_across_the_rollers", d, raw
    for c in crosses:
        if (abs(c["along"]) > L.ALONG_ROLLER_MAX + EPS or L.SP_LEN / 2 - abs(c["tau"]) < L.END_MIN - EPS
                or abs(c["gap"]) > L.ON_ROLLER_TOL + EPS):
            return False, "specimen_not_on_both_rollers", d, raw
    if crosses[0]["tau"] * crosses[1]["tau"] >= 0.0:  # both rollers on one side of its middle: not across the span
        return False, "specimen_not_on_both_rollers", d, raw
    return True, "", d, raw


def pose_tier(sp, sups) -> tuple[float, str, dict]:
    """(score from the poses alone, why it is 0 if it is, diagnostics). Every pose with both supports standing, the
    specimen on both rollers and the span within the standards' band maps to exactly one of 0.25, 0.5, 0.75, 1.0."""
    lft, rgt = left_right(sups)
    s, e = span_and_offset(sups)
    d = {"span_mm": round(s * 1000, 2), "span_error_mm": round((s - L.SPAN_NOMINAL) * 1000, 2),
         "centre_off_mm": round(e * 1000, 2),
         "roller_y_mm": [round(float(roller_centre(x)[1]) * 1000, 2) for x in (lft, rgt)],
         "roller_x_mm": [round(float(roller_centre(x)[0]) * 1000, 2) for x in (lft, rgt)],
         "span_tier": 0.0, "sp_tier": 0.0}
    reason = ""
    for name, sup in (("left", lft), ("right", rgt)):
        ok, why, sd = standing(sup)
        d.update({f"{name}_{k}": v for k, v in sd.items()})
        if not ok and not reason:
            reason = why
    ok, why, rd, raw = on_rollers(sp, sups)
    d.update(rd)
    if reason:
        return 0.0, reason, d
    if not ok:
        return 0.0, why, d
    err = abs(s - L.SPAN_NOMINAL)
    if err <= L.SPAN_TIGHT + EPS and e <= L.CENTRE_TIGHT + EPS:
        span_tier = 0.5
    elif err <= L.SPAN_LOOSE + EPS and e <= L.CENTRE_LOOSE + EPS:
        span_tier = 0.25
    else:
        return 0.0, ("span_out_of_band" if err > L.SPAN_LOOSE + EPS else "span_off_centre"), d
    c, square = raw
    if c <= L.SP_OFF_TIGHT + EPS and square <= L.SQUARE_TIGHT_DEG + EPS:
        sp_tier = 0.5
    elif c <= L.SP_OFF_LOOSE + EPS:
        sp_tier = 0.25
    else:
        sp_tier = 0.0
    d.update(span_tier=span_tier, sp_tier=sp_tier)
    return span_tier + sp_tier, "", d


# -- the grippers -------------------------------------------------------------------------------------------------------
def _box_distance(p, origin, quat_xyzw, centre_local, half) -> float:
    q = rotation(quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(origin, dtype=float))
    d = np.abs(q - np.asarray(centre_local, dtype=float)) - np.asarray(half, dtype=float)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def sp_point_distance(p, sp) -> float:
    """Distance from a point to the bar's box (0 inside it)."""
    return _box_distance(p, sp.pos, sp.quat_xyzw, (0.0, 0.0, 0.0), L.SP_HALF)


def sup_point_distance(p, sup) -> float:
    """Distance from a point to a support's outline (block and roller, 0 inside it)."""
    return _box_distance(p, sup.origin, sup.quat_xyzw, L.SUP_BOX_CENTER, L.SUP_HALF)


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_points(arm, samples: int = 20) -> np.ndarray:
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    return tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)


def zone_gap(arm, sp, sups) -> float:
    """How far a gripper's finger zone is from the nearest of the bar and the two supports (negative: overlap)."""
    pts = zone_points(arm)
    near = min(min([sp_point_distance(p, sp), *(sup_point_distance(p, s) for s in sups)]) for p in pts)
    return near - zone_radius(arm)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the bar's end points and the two rollers' centres over the last half
    second, where the parts were at the previous step (to spot a reset), and how many steps the arms could not be
    read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # (end a, end b, roller 1, roller 2), this step and 10
        self.prev = None
        self.arms_unavailable = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(sp) -> bool:
        """The specimen lying still and flat on the mat on its mark, its length along y."""
        c = np.asarray(sp.pos, dtype=float)
        on_mark = (math.hypot(c[0] - L.SP_START[0], c[1] - L.SP_START[1]) < L.START_TOL
                   and abs(c[2] - (L.TOP + L.SP_START[2])) < 0.003)
        try:
            slow = float(np.linalg.norm(np.asarray(sp.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and sp_tilt_deg(sp) < 2.0 and sp_heading_err_deg(sp) < 5.0 and slow)

    @staticmethod
    def sup_at_start(sup) -> bool:
        """A support standing upright on the base on one of the two starting marks."""
        o = np.asarray(sup.origin, dtype=float)
        on_mark = any(math.hypot(o[0] - x, o[1] - y) < L.SUP_START_TOL for x, y in L.SUP_START.values())
        return bool(on_mark and abs(o[2] - L.BASE_TOP_W) < 0.003 and sup_tilt_deg(sup) < 2.0)

    def moved(self) -> float | None:
        """The farthest any tracked point (bar ends, roller centres) has been, over the last 0.5 s, from where it is
        now (m); None until 0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        now = self.history[-1]
        return float(max(max(np.linalg.norm(p - q) for p, q in zip(past, now)) for past in self.history))

    def update(self, sp, sups, arms=None) -> Result:
        if not all_valid(sp, sups):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(sp, sups, arms)
            return self.last
        cur = (np.asarray(sp.pos, dtype=float), *(np.asarray(s.origin, dtype=float) for s in sups))
        if self.prev is not None:
            limits = (L.RESET_JUMP, L.SUP_RESET_JUMP, L.SUP_RESET_JUMP)
            jumped = [float(np.linalg.norm(a - b)) > j for a, b, j in zip(cur, self.prev, limits)]
            back = [self.at_start(sp), *(self.sup_at_start(s) for s in sups)]
            if any(j and b for j, b in zip(jumped, back)):  # the simulator put a part back: a new episode
                self.reset()
        self.steps += 1
        a, b = sp_ends(sp)
        self.history.append((a, b, *(roller_centre(s) for s in sups)))
        self.prev = cur
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(sp, sups, arms)
        return self.last

    def result(self, sp, sups, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not all_valid(sp, sups):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", "tier": 0.0})
        tier, reason, d = pose_tier(sp, sups)
        moved = self.moved()
        at_rest = moved is not None and moved <= L.STILL_TOL + EPS
        if arms_valid(arms):
            zl, zr = zone_gap(arms["left"], sp, sups), zone_gap(arms["right"], sp, sups)
            off = min(zl, zr) >= L.HANDS_OFF - EPS
            zones = [round(zl * 1000, 1), round(zr * 1000, 1)]
        else:  # not readable: taken as met, and counted
            off, zones = True, None
        score = tier if (at_rest and off) else 0.0
        if tier > 0 and not off:
            reason = "gripper_on_or_near_a_part"
        elif tier > 0 and not at_rest:
            reason = "still_moving"
        details = {"valid_state": True, "tier": tier, "zero_reason": reason, "at_rest": at_rest,
                   "moved_mm": None if moved is None else round(moved * 1000, 2), "hands_off": off,
                   "zone_gap_mm": zones, "arms_unavailable": self.arms_unavailable, **d}
        return Result(score, bool(score == 1.0), details)
