"""The scorer, as plain functions of the six propellers' states. NumPy only; no simulator needed.

The score is read from what stands on the drone's motors at the end, as a technician checks a quad before its first
flight: which motors carry a propeller, is each propeller standing upright on its motor's top, and is it of the hand
that motor needs (its raised leading edges leading in the direction of the motor's arrow). A wrong-hand propeller,
or one upside down, on any motor makes the drone flip at take-off: the trial scores 0. The scorer keeps only each
propeller's last half second of positions (to tell a propeller at rest from one still moving) and is fed one
observation per control step by ``task.success()``:

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps the names in ``layout.PROPS`` to the propellers' states as mechkit reports them
(``AssemblyState``): ``origin`` (the base frame's origin: the centre of the hub's bottom face) and ``quat_xyzw``
(that frame's orientation: local +z up the hub's axis, local +x along the blades). A propeller's hand is fixed by its
shape (``layout.HAND``): no turn about the vertical changes it. A step on which any propeller's state has a
non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up, or a propeller not built as a compound)
scores 0 and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L


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


def valid(st) -> bool:
    """Every number the scorer reads is finite, the quaternion is not degenerate and the base frame's origin is
    there (it is missing if the propeller was not built as a compound)."""
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in ("origin", "quat_xyzw")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def points(st) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(foot, top, tip_a, tip_b): the centre of the hub's bottom face, the centre of its top face and the two blade
    tips, in the world."""
    r = rotation(st.quat_xyzw)
    o = np.asarray(st.origin, dtype=float)
    return (o, o + r @ np.array(L.HUB_TOP_LOCAL), o + r @ np.array(L.TIP_LOCAL[0]), o + r @ np.array(L.TIP_LOCAL[1]))


def tilt_deg(st) -> float:
    """Angle between the hub's axis (bottom face to top face) and straight up, 0..180 degrees: upside down is 180."""
    return float(np.degrees(np.arccos(np.clip(rotation(st.quat_xyzw)[2, 2], -1.0, 1.0))))


def seat_on(point, motor: str) -> tuple[float, float]:
    """(horizontal distance from the motor's axis, height above the motor's top) of a point (m)."""
    mx, my = L.MOTOR_XY[motor]
    return float(math.hypot(point[0] - mx, point[1] - my)), float(point[2] - (L.TOP + L.MOTOR_TOP))


def placement(st) -> dict:
    """Where the propeller's pose alone puts it: ``motor`` (the motor it stands on, or None) and ``way`` ('up':
    upright on it, hub's bottom face on the motor's top; 'down': upside down on it, the hub's top face on the motor's
    top; None), with the numbers behind them."""
    tilt = tilt_deg(st)
    foot, top, _, _ = points(st)
    if tilt <= L.UPRIGHT_DEG:
        way, seat = "up", foot
    elif tilt >= 180.0 - L.UPRIGHT_DEG:
        way, seat = "down", top
    else:
        way, seat = None, foot
    best = None
    for name, _, _, _ in L.MOTORS:
        off, up = seat_on(seat, name)
        if best is None or off < best[1]:
            best = (name, off, up)
    name, off, up = best
    on = way is not None and off <= L.CENTRE_TOL and abs(up) <= L.SEAT_TOL
    return {"motor": name if on else None, "way": way if on else None, "nearest": name,
            "centre_off_mm": round(off * 1000, 2), "seat_mm": round(up * 1000, 2), "tilt_deg": round(tilt, 2)}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each propeller's hub foot and top and blade tips over the last
    half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in L.PROPS}
        self.prev_foot: dict[str, np.ndarray] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every propeller lying upright on its mark on its mat."""
        for n in L.PROPS:
            o = np.asarray(objects[n].origin, dtype=float)
            mx, my = L.PROP_START[n]
            if not (math.hypot(o[0] - mx, o[1] - my) < L.START_TOL and abs(o[2] - (L.TOP + L.MAT_TOP)) < 0.003
                    and tilt_deg(objects[n]) < 2.0):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest the propeller's hub foot, hub top or either blade tip has been, over the last 0.5 s, from where
        it is now (m); None until 0.5 s of this trial has been seen. A propeller spinning in place moves its tips."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        now = h[-1]
        return float(max(max(np.linalg.norm(p0 - p) for p0, p in zip(old, now)) for old in h))

    def update(self, objects) -> Result:
        if not all(valid(objects.get(n)) for n in L.PROPS):  # a blow-up: score 0, keep everything as it was
            self.steps += 1
            self.last = self.result(objects)
            return self.last
        pts = {n: points(objects[n]) for n in L.PROPS}
        if self.prev_foot and self.at_start(objects):
            jumped = any(float(np.linalg.norm(pts[n][0] - self.prev_foot[n])) > L.RESET_JUMP for n in L.PROPS)
            if jumped:  # the simulator put the propellers back on the mats: a new episode
                self.reset()
        self.steps += 1
        for n in L.PROPS:
            self.history[n].append(pts[n])
            self.prev_foot[n] = pts[n][0]
        self.last = self.result(objects)
        return self.last

    def result(self, objects) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        bad = [n for n in L.PROPS if not valid(objects.get(n))]
        if bad:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", "invalid": bad})
        props, wrong, upside_down = {}, [], []
        fitted = {name: [] for name, _, _, _ in L.MOTORS}  # motor -> right-hand propellers on it at rest
        for n in L.PROPS:
            p = placement(objects[n])
            moved = self.moved(n)
            rest = moved is not None and moved <= L.STILL_TOL
            right = p["motor"] is not None and L.HAND[n] == L.MOTOR_DIR[p["motor"]]
            if p["motor"] is None:
                status = "not_on_a_motor"
            elif p["way"] == "down":
                status = "upside_down"
                upside_down.append(f"{n}@{p['motor']}")
            elif not right:
                status = "wrong_hand"
                wrong.append(f"{n}@{p['motor']}")
            elif not rest:
                status = "still_moving"
            else:
                status = "fitted"
                fitted[p["motor"]].append(n)
            props[n] = {"hand": L.HAND[n], "motor": p["motor"], "status": status, "nearest": p["nearest"],
                        "centre_off_mm": p["centre_off_mm"], "seat_mm": p["seat_mm"], "tilt_deg": p["tilt_deg"],
                        "at_rest": rest, "moved_mm": None if moved is None else round(moved * 1000, 2)}
        motors_ok = sum(1 for m in fitted if fitted[m])
        if wrong:
            score, zero = 0.0, "wrong_hand_on_a_motor"
        elif upside_down:
            score, zero = 0.0, "upside_down_on_a_motor"
        else:
            score, zero = L.PER_MOTOR * motors_ok, ""
        success = bool(score == 1.0)
        details = {"valid_state": True, "motors_fitted": motors_ok, "zero_reason": zero, "wrong": wrong,
                   "upside_down": upside_down, "fitted": {m: fitted[m] for m in fitted}, "props": props}
        return Result(score, success, details)
