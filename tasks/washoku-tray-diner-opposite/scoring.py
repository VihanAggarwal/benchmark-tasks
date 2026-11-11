"""The scorer, as plain functions of the four dishes' states (and the grippers', for 'let go'). NumPy only; no
simulator needed.

The rule is the basic washoku layout read from the diner's side of the tray. The guest sits beyond the tray's far edge
facing the robot, so the guest's left is the robot's right (-y) and the guest's front is the tray's far half (x above
its centre line). Each dish earns 0.25 when it is placed in its quadrant: rice front left, soup front right, the fish
plate back right, the side dish back left; and the fish plate also lies side to side with the fish's head to the
guest's left and its belly toward the guest. Two dishes whose footprints overlap both earn 0, and a dish that ever
tipped past 45 degrees earns 0 for the rest of the trial. The score is read from the state at the last control step.

    Tracker.update(dishes, arms) -> Result(score, success, details)

``dishes`` maps each name in ``layout.DISHES`` ("rice", "soup", "side", "fish") to its state as mechkit reports it
(``AssemblyState``): ``origin`` (the base frame's origin: the middle of the dish's bottom face), ``quat_xyzw`` (that
frame's orientation: local +z up through the dish's top; for the fish plate local +x toward the fish's head and local
+y toward its belly) and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. A step on which any dish has a
non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up) scores 0 and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point, 13 mm past the
TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms could not
be read. An arm that cannot be read is skipped by the let-go rule (the other is still checked) and the step is
counted in ``arms_unavailable``.

The tracker remembers, within one trial, each dish's centre over the last second (for 'still') and whether it ever
tipped past 45 degrees (a latch). It starts a new trial on ``reset()`` (task.py calls it at every reset rigstate or
mechkit sees) or when the simulator puts every dish back on its starting mark in one step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

SIDES = ("left", "right")
_EPS = 1e-9
TRAY_TOP_W = L.TOP + L.TRAY_TOP
BOARD_TOP_W = L.TOP + L.BOARD_TOP


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world.
    (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[2].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def velocity(s) -> np.ndarray:
    """The reported linear velocity, or zeros when missing or non-finite (it is used only to spot a teleport)."""
    try:
        v = np.asarray(s.lin_vel, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return np.zeros(3)
    return v if v.size == 3 and np.all(np.isfinite(v)) else np.zeros(3)


def centre(s) -> np.ndarray:
    """The middle of the dish's bottom face (its base frame's origin), in the world."""
    return np.asarray(s.origin, dtype=float)


def tilt_deg(s) -> float:
    """Angle between the dish's own up axis (local +z) and the vertical, 0..180 degrees (180: upside down)."""
    return float(np.degrees(np.arccos(np.clip(rotation(s.quat_xyzw)[2, 2], -1.0, 1.0))))


def _plate_corners(s, top: float) -> np.ndarray:
    """The 8 corners of the fish plate's bounding box (plate and fish), in the world."""
    hx, hy = L.PLATE_L / 2, L.PLATE_W / 2
    local = np.array([(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, top)])
    return local @ rotation(s.quat_xyzw).T + centre(s)


def extremes(name: str, s) -> tuple[float, float]:
    """(lowest, highest) world z of the dish, from its pose. A bowl is a cylinder (radius, height); the fish plate
    is the box around the plate and the fish."""
    if name == "fish":
        z = _plate_corners(s, L.FISH_TOP)[:, 2]
        return float(z.min()), float(z.max())
    r, h = L.BOWL[name]
    az = float(rotation(s.quat_xyzw)[2, 2])
    side = r * math.sqrt(max(0.0, 1.0 - az * az))
    z0 = float(centre(s)[2])
    ends = (z0, z0 + h * az)  # the bottom and top faces' centres
    return min(ends) - side, max(ends) + side


def heading_deg(vec_xy) -> float:
    return float(math.degrees(math.atan2(vec_xy[1], vec_xy[0])))


def angle_to_deg(vec, want) -> float:
    """Angle (0..180 degrees) seen from above between a body axis (projected) and a wanted direction; 180 when the
    axis is vertical (no direction seen from above)."""
    v = np.asarray(vec, dtype=float)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(float(v @ np.asarray(want, dtype=float)) / n, -1.0, 1.0))))


def fish_directions(s) -> tuple[float, float]:
    """(head error, belly error), degrees: the fish's head from the diner's left (-y), its belly from the diner (+x),
    seen from above."""
    r = rotation(s.quat_xyzw)
    return (angle_to_deg(r @ np.asarray(L.HEAD_LOCAL), L.HEAD_WANT),
            angle_to_deg(r @ np.asarray(L.BELLY_LOCAL), L.BELLY_WANT))


# -- footprints (seen from above) ------------------------------------------------------------------------------------
def footprint(name: str, s) -> dict:
    """A dish's footprint: a bowl's base circle (centre, radius) or the fish plate's rectangle (centre, half sizes,
    axes), seen from above."""
    c = centre(s)[:2]
    if name == "fish":
        r = rotation(s.quat_xyzw)
        ax = r[:2, 0]
        n = float(np.linalg.norm(ax))
        ax = ax / n if n > 1e-6 else np.array([1.0, 0.0])
        ay = np.array([-ax[1], ax[0]])
        return {"kind": "rect", "c": c, "half": (L.PLATE_L / 2, L.PLATE_W / 2), "ax": ax, "ay": ay}
    return {"kind": "circle", "c": c, "r": L.BOWL[name][0]}


def _rect_corners(f) -> np.ndarray:
    hx, hy = f["half"]
    return np.array([f["c"] + sx * hx * f["ax"] + sy * hy * f["ay"] for sx in (-1, 1) for sy in (-1, 1)])


def point_to_footprint(p, f) -> float:
    """Horizontal distance from a point to a footprint (0 on or inside it)."""
    p = np.asarray(p, dtype=float)[:2]
    if f["kind"] == "circle":
        return max(0.0, float(np.linalg.norm(p - f["c"])) - f["r"])
    d = p - f["c"]
    u = np.array([abs(float(d @ f["ax"])) - f["half"][0], abs(float(d @ f["ay"])) - f["half"][1]])
    return float(np.linalg.norm(np.maximum(u, 0.0)))


def overlap_depth(fa, fb) -> float:
    """How far two footprints cross (m; 0 or less: they do not). Circle-circle exactly; circle-rectangle by the
    circle's centre's signed distance to the rectangle; there is only one rectangle."""
    if fa["kind"] == "rect":
        fa, fb = fb, fa
    if fb["kind"] == "circle":
        return fa["r"] + fb["r"] - float(np.linalg.norm(fa["c"] - fb["c"]))
    d = fa["c"] - fb["c"]
    u = np.array([abs(float(d @ fb["ax"])) - fb["half"][0], abs(float(d @ fb["ay"])) - fb["half"][1]])
    outside = float(np.linalg.norm(np.maximum(u, 0.0)))
    signed = outside if outside > 0.0 else float(max(u[0], u[1]))
    return fa["r"] - signed


def tray_overhang(f) -> float:
    """How far the footprint sticks out past the tray's outline (m; 0 or less: wholly on it)."""
    if f["kind"] == "circle":
        x, y = f["c"]
        r = f["r"]
        return float(max(L.TRAY_X0 - (x - r), (x + r) - L.TRAY_X1, L.TRAY_Y0 - (y - r), (y + r) - L.TRAY_Y1))
    pts = _rect_corners(f)
    return float(max(L.TRAY_X0 - pts[:, 0].min(), pts[:, 0].max() - L.TRAY_X1,
                     L.TRAY_Y0 - pts[:, 1].min(), pts[:, 1].max() - L.TRAY_Y1))


def quadrant_inset(name: str, s) -> float:
    """How far the dish's centre is inside its quadrant from the nearer centre line (m; negative: in another
    quadrant or that far short of its own)."""
    sx, sy = L.QUADRANT[name]
    c = centre(s)
    return float(min(sx * (c[0] - L.MID_X), sy * (c[1] - L.MID_Y)))


def quadrant_of(s) -> str:
    """Which quadrant (diner's view) the dish's centre is in, for the trace."""
    c = centre(s)
    return ("front" if c[0] > L.MID_X else "back") + " " + ("left" if c[1] < L.MID_Y else "right")


# -- let go: the arms (rigstate) ---------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    """One arm's pad-tip point, or None if it cannot be read (no snapshot, the arm missing, a non-finite value)."""
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm) -> float:
    """How far from this gripper's pad-tip point (horizontally) a finger can touch, over every yaw, plus 5 mm."""
    return L.release_reach(opening(arm))


def held(name: str, s, arms) -> bool:
    """Not let go: some readable gripper's pad-tip point lower than 10 mm above the dish's top while within that
    gripper's finger reach of the dish's footprint horizontally (a finger may be holding, pressing or propping it,
    whatever the gripper's yaw)."""
    top = extremes(name, s)[1]
    f = footprint(name, s)
    for side in readable_sides(arms):
        tips = arm_tips(arms, side)
        if point_to_footprint(tips, f) < reach(arms[side]) and tips[2] - top < L.RELEASE_ABOVE:
            return True
    return False


# -- one dish ---------------------------------------------------------------------------------------------------------
def dish_checks(name: str, s, arms=None) -> dict:
    """Every placement condition of one dish except stillness, overlap and the tip latch (which need the other dishes
    or the trial's history), with the numbers behind them (JSON-clean)."""
    f = footprint(name, s)
    low, high = extremes(name, s)
    inset = quadrant_inset(name, s)
    over = tray_overhang(f)
    rest = low - TRAY_TOP_W
    tilt = tilt_deg(s)
    is_held = held(name, s, arms)
    d = {"centre_mm": [round(float(v) * 1000, 1) for v in centre(s)], "quadrant": quadrant_of(s),
         "inset_mm": round(inset * 1000, 1), "overhang_mm": round(over * 1000, 1), "rest_mm": round(rest * 1000, 1),
         "tilt_deg": round(tilt, 2), "held": is_held,
         "in_quadrant": bool(inset >= L.INSET - _EPS), "on_tray": bool(over <= L.OVERHANG + _EPS),
         "resting": bool(abs(rest) <= L.REST_TOL + _EPS), "upright": bool(tilt <= L.UPRIGHT_DEG + _EPS)}
    ok = d["in_quadrant"] and d["on_tray"] and d["resting"] and d["upright"] and not is_held
    if name == "fish":
        head, belly = fish_directions(s)
        d.update(head_err_deg=round(head, 2), belly_err_deg=round(belly, 2),
                 head_heading_deg=round(heading_deg((rotation(s.quat_xyzw) @ np.asarray(L.HEAD_LOCAL))[:2]), 2),
                 fish_turned_right=bool(head <= L.FISH_DIR_DEG + _EPS and belly <= L.FISH_DIR_DEG + _EPS))
        ok = ok and d["fish_turned_right"]
    d["pose_ok"] = bool(ok)
    return d


def overlaps(dishes) -> dict[str, list[str]]:
    """For each dish, the other dishes whose footprints cross its own by more than 1 mm."""
    fps = {n: footprint(n, dishes[n]) for n in L.DISHES}
    out: dict[str, list[str]] = {n: [] for n in L.DISHES}
    for i, a in enumerate(L.DISHES):
        for b in L.DISHES[i + 1:]:
            if overlap_depth(fps[a], fps[b]) > L.OVERLAP + _EPS:
                out[a].append(b)
                out[b].append(a)
    return out


def evaluate(dishes, arms=None, tipped=None, paths=None) -> tuple[float, dict]:
    """The score of the dishes as they stand. ``tipped``: the names latched as tipped this trial; ``paths``: each
    dish's centre path over the last second (None: not yet a second of history, so nothing is still)."""
    tipped = set(tipped or ())
    ov = overlaps(dishes)
    per: dict = {}
    total = 0.0
    for n in L.DISHES:
        d = dish_checks(n, dishes[n], arms)
        path = None if paths is None else paths.get(n)
        d["path_mm"] = None if path is None else round(path * 1000, 2)
        d["still"] = bool(path is not None and path < L.STILL_PATH)
        d["tipped_latched"] = n in tipped
        d["overlaps"] = ov[n]
        if d["tipped_latched"]:
            reason = "tipped"
        elif ov[n]:
            reason = "overlaps_" + "_".join(ov[n])
        elif not d["pose_ok"]:
            reason = next(k for k, bad in (("wrong_quadrant", not d["in_quadrant"]), ("off_tray", not d["on_tray"]),
                                           ("not_on_tray_top", not d["resting"]), ("not_upright", not d["upright"]),
                                           ("fish_turned_wrong", not d.get("fish_turned_right", True)),
                                           ("held", d["held"])) if bad)
        elif not d["still"]:
            reason = "moving"
        else:
            reason = ""
        d["zero_reason"] = reason
        d["points"] = 0.0 if reason else L.POINTS
        total += d["points"]
        per[n] = d
    return round(total, 4), per


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers about one trial: each dish's centre over the last second, the tip latch, and
    diagnostics for the trace."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_S * control_hz))  # 20 control steps in 1 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.tipped: dict[str, int] = {}  # dish -> the step it first tipped past 45 degrees
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.prev = None  # {name: (centre, velocity)} at the previous valid step
        self.history: deque = deque(maxlen=self.window + 1)  # {name: centre} at the last valid steps
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(dishes) -> bool:
        """The starting layout: every dish still and upright on its serving board, on its own mark."""
        for n in L.DISHES:
            s = dishes[n]
            x0, y0, _ = L.START[n]
            c = centre(s)
            on_mark = math.hypot(c[0] - x0, c[1] - y0) < L.START_TOL
            seated = abs(extremes(n, s)[0] - BOARD_TOP_W) < L.REST_TOL
            if not (on_mark and seated and tilt_deg(s) < 2.0 and float(np.linalg.norm(velocity(s))) < 0.01):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(now[n][0] - self.prev[n][0]))
                   > L.RESET_JUMP + float(np.linalg.norm(self.prev[n][1])) * self.dt + fall for n in L.DISHES)

    def paths(self) -> dict[str, float] | None:
        """Each dish's centre path (m) over the last second, or None before a second of valid steps."""
        if len(self.history) < self.window + 1:
            return None
        h = list(self.history)
        return {n: float(sum(np.linalg.norm(b[n] - a[n]) for a, b in zip(h, h[1:]))) for n in L.DISHES}

    def update(self, dishes, arms=None) -> Result:
        self.steps += 1
        if not all(n in dishes and valid(dishes[n]) for n in L.DISHES):  # a blow-up: 0, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(dishes, arms)
            return self.last
        now = {n: (centre(dishes[n]).copy(), velocity(dishes[n])) for n in L.DISHES}
        if self.prev is not None and self._teleported(now) and self.at_start(dishes):
            self.reset()  # the simulator put every dish back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append({n: c for n, (c, _) in now.items()})
        for n in L.DISHES:
            if n not in self.tipped and tilt_deg(dishes[n]) > L.TIP_DEG + _EPS:
                self.tipped[n] = self.steps
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(dishes, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, dishes, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"bad_steps": self.bad_steps, "best_score": self.best, "arms_readable": list(readable_sides(arms)),
                 "arms_unavailable": self.arms_unavailable, "tipped_at_step": dict(self.tipped),
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = [n for n in L.DISHES if n not in dishes or not valid(dishes[n])]
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_dishes": bad, **trial})
        score, per = evaluate(dishes, arms, self.tipped, self.paths())
        return Result(score, bool(score >= 1.0 - _EPS), {"valid_state": True, "dishes": per, **trial})
