"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads
the pruner, the battery and the scabbard (loose bodies: pose and velocity) and both grippers' TCPs and pad tips, and
works in the pruner's own frame (layout.py), so it doesn't matter where the pruner has been nudged to.

    BATTERY OUT  the battery's centre at least 65 mm from its home: clear of the 60 mm it sits in the handle
    RELEASED     the battery out and no gripper within 30 mm of it
    ON           the scabbard on the guide bar's line (within 4 mm across and 3 mm in height, open end toward the
                 housing) covering at least 40 mm of the bar
    SEATED       on, with its open end within 4 mm of the bar's root (its catch clicked)
    CLEAR        no gripper within 30 mm of the pruner, the battery or the scabbard, and all three still
                 (under 5 mm/s)

    A gripper's reach: the segment from its TCP to its pad tips, swollen by half the pads' current gap plus 8 mm (the
    pads themselves), so the fingers' whole span counts, not two points.

    fouls (score 0 for the rest of the trial):
    TOOTH        a gripper within 10 mm of a bare tooth (the chain's edges not covered by the scabbard)
    BAR          a gripper within 20 mm of the guide bar or chain while the battery is still in
    TRIGGER      a gripper within 10 mm of the trigger while the battery is in
    ORDER        the scabbard 10 mm or more onto the bar while the battery is still in (fitted first)

    1.0   SEATED and CLEAR (success, held 1 s)     0.75  SEATED
    0.5   ON                                         0.25  RELEASED
    0     anything else (the untouched start), and every trial with a foul
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PRUNER, BATTERY, GUARD = "pruner", "battery", "scabbard"


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
        bs = tuple(body(states[n]) for n in (PRUNER, BATTERY, GUARD))
    except (KeyError, TypeError):
        return None
    return None if any(b is None for b in bs) else bs


def guard_on(pruner: Body, guard: Body) -> tuple[float, float | None]:
    """(length of bar covered, the scabbard's centre along the bar or None when it is off the bar's line)."""
    c = pruner.local(guard.origin)
    axis = pruner.rot.T @ guard.rot[:, 1]  # the scabbard's open end points along its +y
    if abs(c[0]) > L.ALIGN_X or abs(c[2] - L.BAR_Z) > L.ALIGN_Z or axis[1] < math.cos(math.radians(5.0)):
        return 0.0, None
    lo, hi = max(L.BAR_Y[0], c[1] - L.GUARD_L / 2), min(L.BAR_Y[1], c[1] + L.GUARD_L / 2)
    return max(0.0, hi - lo), float(c[1])


def bare_spans(centre_y: float | None) -> list[tuple[float, float]]:
    """The stretches of the bar (pruner frame y) whose teeth the scabbard does not cover."""
    if centre_y is None:
        return [L.BAR_Y]
    lo, hi = centre_y - L.GUARD_L / 2, centre_y + L.GUARD_L / 2
    spans = [(L.BAR_Y[0], min(L.BAR_Y[1], lo)), (max(L.BAR_Y[0], hi), L.BAR_Y[1])]
    return [(a, b) for a, b in spans if b - a > L.EPS]


NOSE_TEETH = [np.array([L.BAR_X * math.cos(a), L.NOSE_Y - L.BAR_X * math.sin(a), L.BAR_Z])
              for a in np.linspace(0.0, math.pi, 25)]  # round the nose, from one edge to the other


def tooth_distance(p_local, spans) -> float:
    """Distance from a point (pruner frame) to the nearest bare tooth: the bar's two straight edges and the half
    circle round its nose, wherever the scabbard doesn't cover them."""
    best = math.inf
    for a, b in spans:
        lo, hi = max(a, L.NOSE_Y), b
        if hi > lo:
            y = min(max(p_local[1], lo), hi)
            for sx in (-1.0, 1.0):
                best = min(best, math.dist(p_local, (sx * L.BAR_X, y, L.BAR_Z)))
        for t in NOSE_TEETH:
            if a - L.EPS <= t[1] <= b + L.EPS:
                best = min(best, math.dist(p_local, t))
    return best


BAR_BOX = (np.array([-L.BAR_X, L.BAR_Y[0], L.BAR_Z - L.BAR_T / 2]),
           np.array([L.BAR_X, L.BAR_Y[1], L.BAR_Z + L.BAR_T / 2]))
TRIGGER_BOX = tuple(np.array(v) for v in zip(*L.TRIGGER))  # handle frame
BODY_BOX = (np.array([L.HOUSING[0][0], L.BAR_Y[0], 0.0]), np.array([L.HOUSING[0][1], L.HOUSING[1][1], L.HOUSING[2][1]]))
HANDLE_HALF_X = L.BAT_X / 2 + L.HANDLE_CLEAR + L.HANDLE_WALL
HANDLE_BOX = (np.array([-HANDLE_HALF_X, L.HANDLE_Y[0], 0.0]), np.array([HANDLE_HALF_X, L.HANDLE_Y[1], 0.05]))  # handle frame


def pruner_distance(lp) -> float:
    """Distance from a point (pruner frame) to the pruner: its housing and bar, or its angled handle."""
    return min(L.box_distance(lp, *BODY_BOX), L.box_distance(L.pruner_to_handle(lp), *HANDLE_BOX))
BATTERY_HALF = np.array([L.BAT_X / 2 + L.HOOK_SIZE[0], L.BAT_L / 2, L.BAT_Z / 2])
GUARD_HALF = np.array([L.GUARD_W / 2, L.GUARD_L / 2, L.GUARD_H / 2])


def near_box(p, b: Body, half) -> float:
    return L.box_distance(b.local(p), -half, half)


def gripper_samples(arms, n: int = 6) -> list[tuple[np.ndarray, float]]:
    """Points along each gripper's TCP-to-tips segment with the radius that covers its pads there."""
    if not rigstate.readable(arms):
        return []
    out = []
    for s in ("left", "right"):
        a = arms[s]
        tcp, tips = np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float)
        rad = max(0.0, min(1.0, float(a.opening))) * L.FINGER_GAP / 2 + L.PAD_MARGIN
        out += [(tcp + (tips - tcp) * k / (n - 1), rad) for k in range(n)]
    return out


def latch_step(latched: bool, pressed: bool, disp: float, max_disp: float) -> str:
    """The battery latch's rule for one step: "free" (let it go), "catch" (hold it where it is), "click" (hold it at
    home) or "" (no change). ``disp``: how far the battery is from home now; ``max_disp``: the farthest it has been
    since it was last freed."""
    if latched:
        return "free" if pressed else ""
    if pressed:
        return ""
    if disp <= L.HOME_CLICK:
        return "click"
    if max_disp < L.HOOK_CATCH:
        return "catch"
    return ""


def tier(released: bool, on: bool, seated: bool, clear: bool, foul: bool) -> float:
    if foul or not released:
        return 0.0
    if seated:
        return 1.0 if clear else 0.75
    return 0.5 if on else 0.25


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
        pruner, battery, guard = bodies
        bat_off = float(np.linalg.norm(pruner.local(battery.origin) - L.BAT_HOME))
        battery_out = bat_off >= L.BATTERY_OUT
        covered, gy = guard_on(pruner, guard)
        on = covered >= L.ON_BAR
        seated = gy is not None and abs(gy - L.GUARD_SEAT_Y) <= L.SEAT_TOL
        spans = bare_spans(gy)
        points = gripper_samples(arms)
        gaps = {"tooth": math.inf, "bar": math.inf, "trigger": math.inf, "battery": math.inf, "kit": math.inf}
        for pt, rad in points:
            lp = pruner.local(pt)
            gaps["tooth"] = min(gaps["tooth"], tooth_distance(lp, spans) - rad)
            gaps["bar"] = min(gaps["bar"], L.box_distance(lp, *BAR_BOX) - rad)
            gaps["trigger"] = min(gaps["trigger"], L.box_distance(L.pruner_to_handle(lp), *TRIGGER_BOX) - rad)
            b = near_box(pt, battery, BATTERY_HALF) - rad
            gaps["battery"] = min(gaps["battery"], b)
            gaps["kit"] = min(gaps["kit"], b, pruner_distance(lp) - rad,
                              near_box(pt, guard, GUARD_HALF) - rad)
        for name, hit in (("order", not battery_out and covered >= L.ENGAGED),
                          ("tooth", gaps["tooth"] < L.TOOTH_CLEAR),
                          ("bar", not battery_out and gaps["bar"] < L.BAR_CLEAR),
                          ("trigger", not battery_out and gaps["trigger"] < L.TRIGGER_CLEAR)):
            if hit and name not in self.fouls:
                self.fouls.append(name)
        released = battery_out and gaps["battery"] >= L.CLEAR
        still = max(float(np.linalg.norm(b.vel)) for b in bodies) < L.STILL_SPEED
        clear = bool(points) and gaps["kit"] >= L.CLEAR and still
        level = tier(released, on, seated, clear, bool(self.fouls))
        details = {
            "valid_state": True, "score": level, "battery_off_mm": round(bat_off * 1000, 1),
            "pruner_mm": [round(v * 1000, 1) for v in pruner.origin[:2]],
            "pruner_yaw_deg": round(math.degrees(math.atan2(pruner.rot[1, 0], pruner.rot[0, 0])), 1),
            "guard_local_mm": [round(v * 1000, 1) for v in pruner.local(guard.origin)],
            "battery_out": battery_out, "released": released, "covered_mm": round(covered * 1000, 1),
            "guard_y_mm": None if gy is None else round(gy * 1000, 1), "on": on, "seated": seated, "clear": clear,
            "still": still, "fouls": list(self.fouls), "arms_readable": rigstate.readable(arms),
            "gaps_mm": {k: (None if math.isinf(v) else round(v * 1000, 1)) for k, v in gaps.items()},
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
