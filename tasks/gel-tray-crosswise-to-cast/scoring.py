"""The scorer, as plain functions of the tray's and the comb's states and the grippers' state. NumPy only; no simulator
needed.

The score is read from where the tray and the comb are at the end, as a technician checks a gel box set up to cast
before pouring the agarose: is the tray flat on the platform, turned crosswise between the tank's side walls so that
its two open, gasketed ends face those walls (a top photo and a rule), is the comb standing in a slot (its bar down in
the notches of both side walls of the tray), and which slot: one near an end, or the middle one. Every tier needs both
grippers clear of the tray (let go of it, and not holding the comb in it) and the tray at rest; the comb counts as
seated only at rest and with both grippers clear of it too. The scorer keeps the last half second of a few tracked
points of each part (to tell a part at rest from one still moving) and is fed one observation per control step by
``task.success()``:

    Tracker.update(tray, comb, arms) -> Result(score, success, details)

``tray`` and ``comb`` are the parts' states as mechkit reports them (``AssemblyState``): ``origin`` (the base frame's
origin) and ``quat_xyzw`` (its orientation), plus the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. The
tray's frame: origin at the middle of its underside, local x along its length (gasket to gasket), y across, z up.
The comb's frame: origin at the middle of its bar's underside, local y along the bar, x across it, z up (the teeth
hang below). A state with a non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up, or a part not
built as an assembly) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the hands-clear
conditions are taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper
heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a
radius of the 38 mm segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer
corner (half the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm
fully open).

The tray is the same either way round (a gasket on each end, slots placed symmetrically), and so is the comb: only
the tray's open side must face up and the comb's teeth down.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

PLAT_TOP_W = L.TOP + L.PLAT_TOP  # the platform's top (world z)
_TRAY_CORNERS = [(sx * L.TRAY_LEN / 2, sy * L.TRAY_W / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
_GASKET_FACES = [(sx * L.TRAY_LEN / 2, 0.0, L.GASKET_SIZE[2] / 2) for sx in (-1, 1)]
_COMB_SEATS = [(0.0, sy * L.WALL_Y, 0.0) for sy in (-1, 1)]  # the bar's underside over each tray side wall
_COMB_ENDS = [(0.0, sy * L.COMB_BAR[1] / 2, L.COMB_BAR[2] / 2) for sy in (-1, 1)]  # the bar's end faces' centres


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


def valid(part) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(part, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


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


def to_world(part, local) -> np.ndarray:
    return np.asarray(part.origin, dtype=float) + rotation(part.quat_xyzw) @ np.asarray(local, dtype=float)


def to_local(part, world) -> np.ndarray:
    return rotation(part.quat_xyzw).T @ (np.asarray(world, dtype=float) - np.asarray(part.origin, dtype=float))


# -- the tray -------------------------------------------------------------------------------------------------------
def tray_corners(tray) -> list[np.ndarray]:
    """The four corners of the tray's underside, in the world."""
    return [to_world(tray, c) for c in _TRAY_CORNERS]


def tray_tilt_deg(tray) -> float:
    """Angle between the vertical and the tray's up axis (0..180 degrees: 0 level and open side up, 180 upside
    down)."""
    return float(np.degrees(np.arccos(np.clip(rotation(tray.quat_xyzw)[2, 2], -1.0, 1.0))))


def cross_err_deg(tray) -> float:
    """Angle between the tray's length seen from above and the line between the side walls (x), either way round
    (0..90 degrees): 0 crosswise, 90 lengthwise (its open ends toward the electrodes)."""
    a = rotation(tray.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # standing on an end
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(a[0]) / h))))


def gasket_gaps(tray) -> tuple[float, float]:
    """(near, far): along x, from the near side wall's inner face (x 0.273) to the nearer gasket end's face centre,
    and from the farther gasket end's face centre to the far side wall's inner face (x 0.367). Both between 0 and
    10 mm for a crosswise tray down between the walls; negative is a gasket pressed into a wall (solver penetration)."""
    xs = sorted(float(to_world(tray, g)[0]) for g in _GASKET_FACES)
    return xs[0] - L.SIDE_IN[0], L.SIDE_IN[1] - xs[1]


def tray_check(tray) -> tuple[bool, str, dict]:
    """Is the tray set to cast (its pose alone)? (ok, why not, diagnostics). On the platform: open side up, tilted
    3 degrees or less, every corner of its underside within 2 mm of the platform's top height and over the platform
    (within 55 mm of y 0); crosswise within 12 degrees; each gasketed end within 10 mm of the side wall it faces."""
    tilt, cross = tray_tilt_deg(tray), cross_err_deg(tray)
    cw = tray_corners(tray)
    corners = [float(c[2] - PLAT_TOP_W) for c in cw]
    overhang = max(abs(float(c[1]) - L.TANK_Y) for c in cw) - L.PLAT_HALF_Y  # > 0: a corner past the platform's end
    near, far = gasket_gaps(tray)
    o = np.asarray(tray.origin, dtype=float)
    d = {"tray_tilt_deg": round(tilt, 2), "cross_err_deg": round(cross, 2),
         "corner_gap_mm": [round(c * 1000, 2) for c in corners], "gasket_gap_mm": [round(near * 1000, 2),
                                                                                    round(far * 1000, 2)],
         "tray_centre_mm": [round(float(o[0]) * 1000, 1), round(float(o[1]) * 1000, 1)],
         "tray_overhang_mm": round(overhang * 1000, 1)}
    if tilt > L.TRAY_TILT_DEG:
        return False, ("tray_upside_down" if tilt > 90.0 else "tray_tilted"), d
    if max(abs(c) for c in corners) > L.CORNER_TOL:
        return False, "tray_not_on_platform", d
    if cross > L.CROSS_DEG:
        return False, ("tray_lengthwise" if cross > 45.0 else "tray_not_crosswise"), d
    if not (L.GASKET_MIN <= near <= L.GASKET_MAX and L.GASKET_MIN <= far <= L.GASKET_MAX):
        return False, "tray_not_between_side_walls", d
    if overhang > 0.0:
        return False, "tray_off_platform", d
    return True, "", d


# -- the comb -------------------------------------------------------------------------------------------------------
def comb_in_tray(comb, tray) -> dict:
    """Where the comb is in the tray's frame: its bar's underside over each of the tray's side walls (x along the
    tray, z above the tray's underside), its middle's offset across the tray, its heading and tilt relative to it."""
    rt, rc = rotation(tray.quat_xyzw), rotation(comb.quat_xyzw)
    seats = [to_local(tray, to_world(comb, p)) for p in _COMB_SEATS]
    mid = to_local(tray, comb.origin)
    v = rt.T @ rc[:, 1]  # the bar's length in the tray's frame
    heading = float(np.degrees(math.atan2(abs(v[0]), abs(v[1])))) if math.hypot(v[0], v[1]) > 1e-9 else 90.0
    up = float(np.clip(rt[:, 2] @ rc[:, 2], -1.0, 1.0))
    return {"seat_x": [float(s[0]) for s in seats], "seat_z": [float(s[2]) for s in seats], "mid_y": float(mid[1]),
            "heading_deg": heading, "tilt_deg": float(np.degrees(math.acos(up)))}


def comb_slot(comb, tray) -> tuple[str | None, str, dict]:
    """(the slot the comb is seated in, or None; why not; diagnostics). Seated: upright relative to the tray (teeth
    down), its bar within 10 degrees of the tray's width, its middle within 6 mm of the tray's centre line (teeth
    between the side walls), and at both side walls its bar's underside within 1.5 mm of the notch bottoms (13 mm above
    the tray's underside) and within 6 mm of the same slot's centre along the tray."""
    w = comb_in_tray(comb, tray)
    d = {"comb_seat_x_mm": [round(x * 1000, 2) for x in w["seat_x"]],
         "comb_seat_z_mm": [round((z - L.NOTCH_BOTTOM) * 1000, 2) for z in w["seat_z"]],
         "comb_mid_y_mm": round(w["mid_y"] * 1000, 2), "comb_heading_deg": round(w["heading_deg"], 2),
         "comb_tilt_deg": round(w["tilt_deg"], 2)}
    if w["tilt_deg"] > L.COMB_TILT_DEG:
        return None, "comb_not_upright", d
    if w["heading_deg"] > L.COMB_HEADING_DEG:
        return None, "comb_not_across_tray", d
    if abs(w["mid_y"]) > L.COMB_Y_TOL:
        return None, "comb_not_between_side_walls", d
    if max(abs(z - L.NOTCH_BOTTOM) for z in w["seat_z"]) > L.SEAT_Z_TOL:
        return None, "comb_not_down_in_notches", d
    for name, sx in L.SLOTS.items():
        if all(abs(x - sx) <= L.SLOT_X_TOL for x in w["seat_x"]):
            return name, "", d
    return None, "comb_not_in_a_slot", d


# -- the grippers ---------------------------------------------------------------------------------------------------
def point_distance(p, part, box_center, box_half) -> float:
    """Distance from a point to a part's bounding box (0 inside it), in the part's own frame."""
    q = to_local(part, p)
    d = np.abs(q - np.asarray(box_center)) - np.asarray(box_half)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, part, box_center, box_half, samples: int = 20) -> float:
    """How far a gripper's finger zone is from a part's box (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return min(point_distance(p, part, box_center, box_half) for p in pts) - zone_radius(arm)


def tray_zone_gap(arm, tray) -> float:
    return zone_gap(arm, tray, L.TRAY_BOX_CENTER, L.TRAY_BOX_HALF)


def comb_zone_gap(arm, comb) -> float:
    return zone_gap(arm, comb, L.COMB_BOX_CENTER, L.COMB_BOX_HALF)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the tray's four bottom corners and the comb's two bar ends over the
    last half second, and how many steps the arms could not be read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.tray_hist: deque = deque(maxlen=self.window + 1)
        self.comb_hist: deque = deque(maxlen=self.window + 1)
        self.prev_origin = None
        self.arms_unavailable = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(tray) -> bool:
        """The starting layout: the tray standing still on the bench on its mark, open side up, lengthwise."""
        o = np.asarray(tray.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.TRAY_START[0], o[1] - L.TRAY_START[1]) < L.START_TOL
                   and abs(o[2] - L.TOP) < 0.003)
        try:
            slow = float(np.linalg.norm(np.asarray(tray.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and tray_tilt_deg(tray) < 2.0 and cross_err_deg(tray) > 85.0 and slow)

    @staticmethod
    def _moved(hist, window: int) -> float | None:
        """The farthest any tracked point has been, over the last 0.5 s, from where it is now (m); None until 0.5 s
        of this trial has been seen."""
        if len(hist) < window + 1:
            return None
        now = hist[-1]
        return float(max(max(np.linalg.norm(p - q) for p, q in zip(past, now)) for past in hist))

    def update(self, tray, comb, arms=None) -> Result:
        if not (valid(tray) and valid(comb)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(tray, comb, arms)
            return self.last
        o = np.asarray(tray.origin, dtype=float)
        if self.prev_origin is not None and self.at_start(tray):
            if float(np.linalg.norm(o - self.prev_origin)) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
        self.steps += 1
        self.tray_hist.append(tray_corners(tray))
        self.comb_hist.append([to_world(comb, p) for p in _COMB_ENDS])
        self.prev_origin = o
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(tray, comb, arms)
        return self.last

    def result(self, tray, comb, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not (valid(tray) and valid(comb)):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state",
                                       "tray_valid": valid(tray), "comb_valid": valid(comb)})
        tray_ok, reason, d = tray_check(tray)
        slot, comb_reason, dc = comb_slot(comb, tray)
        tm, cm = self._moved(self.tray_hist, self.window), self._moved(self.comb_hist, self.window)
        tray_rest = tm is not None and tm <= L.STILL_TOL
        comb_rest = cm is not None and cm <= L.STILL_TOL
        if arms_valid(arms):
            tz = [tray_zone_gap(arms[s], tray) for s in ("left", "right")]
            cz = [comb_zone_gap(arms[s], comb) for s in ("left", "right")]
            clear_tray, clear_comb = min(tz) >= L.HANDS_OFF, min(cz) >= L.HANDS_OFF
            zones = {"tray": [round(v * 1000, 1) for v in tz], "comb": [round(v * 1000, 1) for v in cz]}
        else:  # not readable: taken as met, and counted
            clear_tray, clear_comb, zones = True, True, None
        tray_set = tray_ok and tray_rest and clear_tray
        if not tray_ok:
            zero = reason
        elif not clear_tray:
            zero = "gripper_on_or_near_tray"
        elif not tray_rest:
            zero = "tray_still_moving"
        else:
            zero = ""
        seated = slot is not None and comb_rest and clear_comb
        if slot is not None and not clear_comb:
            comb_reason = "comb_held_or_touched"
        elif slot is not None and not comb_rest:
            comb_reason = "comb_still_moving"
        if not tray_set:
            score = 0.0
        elif seated:
            score = 1.0 if slot in L.END_SLOTS else 0.75
        else:
            score = 0.5
        details = {"valid_state": True, "zero_reason": zero, "tray_set": tray_set, "tray_pose_ok": tray_ok,
                   "tray_at_rest": tray_rest, "tray_moved_mm": None if tm is None else round(tm * 1000, 2),
                   "comb_slot": slot if seated else None, "comb_slot_pose": slot, "comb_reason": comb_reason,
                   "comb_at_rest": comb_rest, "comb_moved_mm": None if cm is None else round(cm * 1000, 2),
                   "clear_of_tray": clear_tray, "clear_of_comb": clear_comb, "zone_gap_mm": zones,
                   "arms_unavailable": self.arms_unavailable, **d, **dc}
        return Result(score, bool(score == 1.0), details)
