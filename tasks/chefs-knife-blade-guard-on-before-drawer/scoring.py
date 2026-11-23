"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step.
Works in the knife's own frame for guard coverage/seating and in world frame for the fixed utensil drawer.

    PARTIAL     guard aligned on the blade line covering at least 10 cm of the blade
    SEATED      guard aligned on the blade line covering the ENTIRE cutting edge, heel to tip: open mouth at or past
                the heel and closed end at or past the tip (0.5 mm measurement tolerance, no exposed edge)
    IN_DRAWER   guard seated and the sheathed knife resting flat inside the open utensil drawer interior
    CLEAR       both grippers at least 4.0 cm clear of the knife and guard
    STILL       knife and guard linear speed < 5 mm/s and displacement <= 2 mm over the clear hold window

    Latched hard-0 violations (score 0 for the rest of the trial):
    unsheathed_in_drawer   any part of the unsheathed blade enters the utensil drawer interior
                           (x-y inside drawer, z < 10 cm above bench) before the guard is seated
    edge_contact           any gripper finger zone comes within 8 mm of an exposed (uncovered) cutting edge

    1.0   SEATED, IN_DRAWER, CLEAR, and STILL (success, held 1.0 s)
    0.75  SEATED and IN_DRAWER (not yet clear/still)
    0.50  SEATED (on board / not yet in drawer)
    0.25  PARTIAL (guard slid >= 10 cm onto blade)
    0.0   anything else (untouched start) or any latched violation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

KNIFE, GUARD = "knife", "guard"
BOARD, DRAWER = "cutting_board", "utensil_drawer"


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

    def world(self, lp) -> np.ndarray:
        return self.origin + self.rot @ np.asarray(lp, dtype=float)

    def tilt(self) -> float:
        cos_t = max(-1.0, min(1.0, float(self.rot[2, 2])))
        return math.acos(cos_t)


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
        bs = tuple(body(states[n]) for n in (KNIFE, GUARD))
    except (KeyError, TypeError):
        return None
    return None if any(b is None for b in bs) else bs


def guard_on(knife: Body, guard: Body) -> tuple[float, float | None]:
    """(length of blade covered in m, guard center y in knife frame or None when off the blade line)."""
    c = knife.local(guard.origin)
    axis_y = knife.rot.T @ guard.rot[:, 1]
    axis_z = knife.rot.T @ guard.rot[:, 2]
    if (abs(c[0] - L.GUARD_SEAT_X) > L.ALIGN_X
            or abs(c[2] - L.GUARD_SEAT_Z) > L.ALIGN_Z
            or axis_y[1] < L.ALIGN_COS
            or abs(axis_z[2]) < L.ALIGN_COS):
        return 0.0, None
    lo = max(L.BLADE_Y[0], c[1] - L.GUARD_L / 2)
    hi = min(L.BLADE_Y[1], c[1] + L.GUARD_L / 2)
    return max(0.0, hi - lo), float(c[1])


def catch_ready(covered: float, gy: float | None, on_from_tip: bool) -> tuple[bool, bool]:
    """Guard heel retention catch rule: engages once the guard has slid onto the blade from the tip side (covering
    >= 10 cm while short of the seat) and the robot has pushed it to within CATCH_TOL (1 mm) of the heel seat, or
    against the handle face that stops it at the seat, while still on the blade (the window is two-sided, so a
    guard beyond the handle or off the blade never engages). The catch then locks it flush with the heel, moving it at most
    CATCH_TOL, so the robot itself must push the guard fully home."""
    if gy is None:
        return False, False
    if covered >= L.PARTIAL_COVER and gy < L.GUARD_SEAT_Y - 0.005:
        on_from_tip = True
    # Two-sided window: the guard must be on the blade (still covering >= 10 cm) and within CATCH_TOL of the seat
    # on either side, so a guard lifted clear of the blade or carried past the handle can never be snapped onto it.
    near_seat = abs(gy - L.GUARD_SEAT_Y) <= L.CATCH_TOL + 1e-9
    return bool(on_from_tip and near_seat and covered >= L.PARTIAL_COVER), on_from_tip


def exposed_edge(gy: float | None) -> float:
    """Length (m) of cutting edge left outside the guard sleeve: short of the heel plus short of the tip."""
    if gy is None:
        return L.BLADE_Y[1] - L.BLADE_Y[0]
    heel_gap = max(0.0, L.BLADE_Y[1] - (gy + L.GUARD_L / 2))
    tip_gap = max(0.0, (gy - L.GUARD_L / 2) - L.BLADE_Y[0])
    return heel_gap + tip_gap


def fully_covered(gy: float | None) -> bool:
    """Seated: guard aligned on the blade line and covering the entire cutting edge from heel to tip."""
    return gy is not None and exposed_edge(gy) <= L.FULL_COVER_TOL


# Precomputed blade sample points in knife frame
_EDGE_YS = np.linspace(L.BLADE_Y[0], L.BLADE_Y[1], 41)
_EDGE_POINTS = [np.array([L.blade_edge_x(float(y)), float(y), L.BLADE_Z]) for y in _EDGE_YS]

_BLADE_CHECK_POINTS = []
for _y in np.linspace(L.BLADE_Y[0], L.BLADE_Y[1], 11):
    _yf = float(_y)
    _BLADE_CHECK_POINTS.append(np.array([L.BLADE_SPINE_X, _yf, L.BLADE_Z]))
    _BLADE_CHECK_POINTS.append(np.array([L.blade_edge_x(_yf), _yf, L.BLADE_Z]))

# Key corners of sheathed knife (handle + seated guard) for drawer-containment check
_SHEATHED_CORNERS = [
    np.array([L.HANDLE_X[0], L.HANDLE_Y[0], 0.0]),
    np.array([L.HANDLE_X[1], L.HANDLE_Y[0], 0.0]),
    np.array([L.HANDLE_X[0], L.HANDLE_Y[1], L.HANDLE_H]),
    np.array([L.HANDLE_X[1], L.HANDLE_Y[1], L.HANDLE_H]),
    np.array([L.GUARD_SEAT_X - L.GUARD_W / 2, L.GUARD_SEAT_Y - L.GUARD_L / 2, L.GUARD_SEAT_Z]),
    np.array([L.GUARD_SEAT_X + L.GUARD_W / 2, L.GUARD_SEAT_Y - L.GUARD_L / 2, L.GUARD_SEAT_Z]),
    np.array([L.GUARD_SEAT_X - L.GUARD_W / 2, L.GUARD_SEAT_Y + L.GUARD_L / 2, L.GUARD_SEAT_Z]),
    np.array([L.GUARD_SEAT_X + L.GUARD_W / 2, L.GUARD_SEAT_Y + L.GUARD_L / 2, L.GUARD_SEAT_Z]),
]

HANDLE_BOX = (np.array([L.HANDLE_X[0], L.HANDLE_Y[0], 0.0]),
              np.array([L.HANDLE_X[1], L.HANDLE_Y[1], L.HANDLE_H]))
BLADE_BOX = (np.array([L.BLADE_EDGE_HEEL_X, L.BLADE_Y[0], L.BLADE_Z - L.BLADE_T / 2]),
             np.array([L.BLADE_SPINE_X, L.BLADE_Y[1], L.BLADE_Z + L.BLADE_T / 2]))
GUARD_HALF = np.array([L.GUARD_W / 2, L.GUARD_L / 2, L.GUARD_H / 2])


def bare_edge_points(gy: float | None) -> list[np.ndarray]:
    """Points along the cutting edge not currently covered by the blade guard."""
    if gy is None:
        return list(_EDGE_POINTS)
    lo, hi = gy - L.GUARD_L / 2, gy + L.GUARD_L / 2
    return [p for p in _EDGE_POINTS if not (lo - L.EPS <= p[1] <= hi + L.EPS)]


def edge_distance(lp: np.ndarray, bare_pts: list[np.ndarray]) -> float:
    if not bare_pts:
        return math.inf
    return min(float(np.linalg.norm(lp - ep)) for ep in bare_pts)


def knife_distance(lp: np.ndarray) -> float:
    return min(L.box_distance(lp, *HANDLE_BOX), L.box_distance(lp, *BLADE_BOX))


def guard_distance(p_world: np.ndarray, guard: Body) -> float:
    return L.box_distance(guard.local(p_world), -GUARD_HALF, GUARD_HALF)


def unsheathed_blade_in_drawer(knife: Body, seated: bool) -> bool:
    """True if any part of the unsheathed blade enters the utensil drawer interior below DRAWER_HAZARD_Z."""
    if seated:
        return False
    x0, x1 = L.DRAWER_INNER_X
    y0, y1 = L.DRAWER_INNER_Y
    for lp in _BLADE_CHECK_POINTS:
        wp = knife.world(lp)
        if x0 <= wp[0] <= x1 and y0 <= wp[1] <= y1 and wp[2] < L.DRAWER_HAZARD_Z:
            return True
    return False


def sheathed_in_drawer(knife: Body, guard: Body, seated: bool) -> bool:
    """True when the guard is seated and the entire sheathed knife rests flat inside the utensil drawer."""
    if not seated:
        return False
    if knife.tilt() > L.FLAT_TILT_RAD:
        return False
    if abs(float(knife.origin[2]) - L.DRAWER_FLOOR_Z) > L.DRAWER_FLOOR_TOL:
        return False
    x0, x1 = L.DRAWER_INNER_X
    y0, y1 = L.DRAWER_INNER_Y
    for lp in _SHEATHED_CORNERS:
        wp = knife.world(lp)
        if not (x0 - 0.002 <= wp[0] <= x1 + 0.002
                and y0 - 0.002 <= wp[1] <= y1 + 0.002
                and L.DRAWER_FLOOR_Z - 0.005 <= wp[2] <= L.DRAWER_RIM_Z + 0.005):
            return False
    return True


def gripper_samples(arms, n: int = 6) -> list[tuple[np.ndarray, float]]:
    """Points along each gripper's TCP-to-tips segment with the radius covering its pads."""
    if not rigstate.readable(arms):
        return []
    out = []
    for s in ("left", "right"):
        a = arms[s]
        tcp, tips = np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float)
        rad = max(0.0, min(1.0, float(a.opening))) * L.FINGER_GAP / 2 + L.PAD_MARGIN
        out += [(tcp + (tips - tcp) * k / (n - 1), rad) for k in range(n)]
    return out


def tier(partial: bool, seated: bool, in_drawer: bool, clear_still: bool, foul: bool) -> float:
    if foul:
        return 0.0
    if seated and in_drawer:
        return 1.0 if clear_still else 0.75
    if seated:
        return 0.50
    if partial:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Per-episode state tracker. Only an episode reset clears latched fouls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.fouls: list[str] = []
        self._hold_ref: tuple[np.ndarray, np.ndarray] | None = None
        self._hold_steps = 0
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        bodies = read(states) if isinstance(states, dict) else None
        if bodies is None:
            self.bad_steps += 1
            self.last = Result(0.0 if self.fouls else self.last.score, False,
                               {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        knife, guard = bodies
        covered, gy = guard_on(knife, guard)
        partial = covered >= L.PARTIAL_COVER
        seated = fully_covered(gy)
        in_drawer = sheathed_in_drawer(knife, guard, seated)

        bare_pts = bare_edge_points(gy)
        points = gripper_samples(arms)
        edge_gap = math.inf
        kit_gap = math.inf
        for pt, rad in points:
            lp = knife.local(pt)
            if bare_pts:
                edge_gap = min(edge_gap, edge_distance(lp, bare_pts) - rad)
            kit_gap = min(kit_gap, knife_distance(lp) - rad, guard_distance(pt, guard) - rad)

        # Latched hard-0 violations
        if unsheathed_blade_in_drawer(knife, seated) and "unsheathed_in_drawer" not in self.fouls:
            self.fouls.append("unsheathed_in_drawer")
        if bare_pts and edge_gap < L.EDGE_CLEAR and "edge_contact" not in self.fouls:
            self.fouls.append("edge_contact")

        clear = bool(points) and kit_gap >= L.CLEAR_DIST
        speed_ok = max(float(np.linalg.norm(knife.vel)), float(np.linalg.norm(guard.vel))) < L.STILL_SPEED
        if seated and in_drawer and clear and speed_ok:
            if self._hold_ref is None:
                self._hold_ref = (knife.origin.copy(), guard.origin.copy())
                self._hold_steps = 1
                drift = 0.0
            else:
                drift = max(float(np.linalg.norm(knife.origin - self._hold_ref[0])),
                            float(np.linalg.norm(guard.origin - self._hold_ref[1])))
                if drift <= L.STILL_DRIFT:
                    self._hold_steps += 1
                else:
                    self._hold_ref = (knife.origin.copy(), guard.origin.copy())
                    self._hold_steps = 1
        else:
            self._hold_ref = None
            self._hold_steps = 0
            drift = 0.0

        still = self._hold_steps >= L.HOLD_STEPS
        clear_still = clear and still
        level = tier(partial, seated, in_drawer, clear_still, bool(self.fouls))
        details = {
            "valid_state": True,
            "score": level,
            "knife_mm": [round(float(v) * 1000, 1) for v in knife.origin],
            "knife_tilt_deg": round(math.degrees(knife.tilt()), 1),
            "guard_local_mm": [round(float(v) * 1000, 1) for v in knife.local(guard.origin)],
            "covered_mm": round(covered * 1000, 1),
            "guard_y_mm": None if gy is None else round(gy * 1000, 1),
            "exposed_edge_mm": round(exposed_edge(gy) * 1000, 1),
            "partial": partial,
            "seated": seated,
            "in_drawer": in_drawer,
            "clear": clear,
            "still": still,
            "hold_steps": self._hold_steps,
            "drift_mm": round(drift * 1000, 2),
            "fouls": list(self.fouls),
            "arms_readable": rigstate.readable(arms),
            "edge_gap_mm": None if math.isinf(edge_gap) else round(edge_gap * 1000, 1),
            "kit_gap_mm": None if math.isinf(kit_gap) else round(kit_gap * 1000, 1),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
