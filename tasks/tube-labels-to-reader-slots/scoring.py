"""The scorer, as plain functions of the four sample tubes' states. NumPy only; no simulator needed.

The score is read from where the tubes stand at the end, as a laboratory technician checks a loaded rack before it
goes onto the analyser: which positions hold a tube, is each one standing upright on the rack's floor, and does its
barcode label face out through that position's reader slot (toward the arms, -x) so the reader can see it, and has
the gripper let go of it. The scorer keeps only each tube's last half second of positions (to tell a tube at rest
from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` maps the names in ``layout.TUBES`` to the tubes' states as mechkit reports them (``AssemblyState``):
``origin`` (the base frame's origin: the centre of the tube's bottom) and ``quat_xyzw`` (that frame's orientation:
local +z up the tube's axis, local +x out through the middle of the label). A step on which any tube's state has a
non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up, or a tube not built as a compound) scores
0 and leaves the tracker as it was.

``arms`` is ``rigstate.arms()`` (both grippers' tool point, pad-tip point and opening; fed by the offline checker
too). A tube counts only once both grippers' finger zones are at least ``layout.RELEASE_CLEAR`` (20 mm) from it. The
rule fails closed: arms that cannot be read (None, a side missing, a non-finite number) release nothing, so no tube
counts on that step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate


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
    there (it is missing if the tube was not built as a compound)."""
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in ("origin", "quat_xyzw")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def points(st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bottom, top, label): the centre of the tube's bottom, the middle of its cap's top and the middle of its
    label's face, in the world."""
    r = rotation(st.quat_xyzw)
    o = np.asarray(st.origin, dtype=float)
    return o, o + r @ np.array(L.TOP_LOCAL), o + r @ np.array(L.LABEL_POINT)


def tube_distance(st):
    """A distance function (rigstate's ``part``) to the tube as it stands: (M, 3) world points -> (M,) distances (m)
    to the nearest of its body cylinder, its closure cylinder and its label box; 0 inside."""
    r = rotation(st.quat_xyzw)
    o = np.asarray(st.origin, dtype=float)

    def distance(points):
        local = (np.atleast_2d(np.asarray(points, dtype=float)) - o) @ r  # into the tube's frame (columns = axes)
        radial = np.hypot(local[:, 0], local[:, 1])
        out = []
        for rad, z0, z1 in ((L.TUBE_R, 0.0, L.BODY_H), (L.CAP_R, L.BODY_H, L.TUBE_H)):
            along = np.maximum(np.maximum(z0 - local[:, 2], local[:, 2] - z1), 0.0)
            out.append(np.hypot(np.maximum(radial - rad, 0.0), along))
        q = np.abs(local - np.array(L.LABEL_CENTER)) - np.array(L.LABEL_SIZE) / 2
        out.append(np.linalg.norm(np.maximum(q, 0.0), axis=1))
        return np.min(out, axis=0)

    return distance


def released(arms, st) -> bool:
    """Both grippers have let go of this tube: the arms are readable and each finger zone is at least
    RELEASE_CLEAR from it (rigstate's fail-closed rule: unreadable arms release nothing)."""
    return rigstate.released(arms, tube_distance(st), L.RELEASE_CLEAR)


def tilt_deg(st) -> float:
    """Angle between the tube's axis (bottom to cap) and the vertical, 0..180 degrees: upside down is 180."""
    return float(np.degrees(np.arccos(np.clip(rotation(st.quat_xyzw)[2, 2], -1.0, 1.0))))


def bearing_err_deg(st) -> float:
    """Angle, seen from above, between the direction the label faces (its outward normal) and straight out through
    the reader slot (-x, toward the arms): 0..180 degrees. A label facing straight up or down (a tube lying on its
    side, label up) has no direction seen from above and reads 180."""
    n = rotation(st.quat_xyzw)[:, 0]
    h = math.hypot(n[0], n[1])
    if h < 1e-6:
        return 180.0
    c = (n[0] * L.READER_DIR[0] + n[1] * L.READER_DIR[1]) / h
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


def label_edges_y(st) -> tuple[float, float]:
    """World y of the label face's two vertical edges (the reader looks along +x through the slot, so y is what
    the slot's posts can hide): the face's middle plus or minus half the label's width along the face."""
    r = rotation(st.quat_xyzw)
    o = np.asarray(st.origin, dtype=float)
    ys = [float((o + r @ np.array([L.LABEL_OUT, s * L.LABEL_W / 2, L.LABEL_POINT[2]]))[1]) for s in (-1, 1)]
    return min(ys), max(ys)


def label_in_slot(st, k: int) -> bool:
    """Both vertical edges of the label's face lie inside the 18 mm reader slot of position ``k``, so no post hides
    part of the barcode from the reader."""
    lo, hi = label_edges_y(st)
    half = L.SLOT_W / 2 + L.LABEL_EDGE_TOL
    return bool(lo >= L.POS_Y[k - 1] - half and hi <= L.POS_Y[k - 1] + half)


def position_of(st) -> int | None:
    """The analyser rack position (1..5, P1 on the robot's left) whose 24 x 24 mm opening holds the centre of the
    tube's bottom, seen from above; None if it is in none of them."""
    o = np.asarray(st.origin, dtype=float)
    if abs(o[0] - L.RACK_X) > L.POS_TOL:
        return None
    for k, py in enumerate(L.POS_Y, start=1):
        if abs(o[1] - py) <= L.POS_TOL:
            return k
    return None


def pose(st) -> dict:
    """What the tube's pose alone says: the position it stands in (its bottom's centre inside the opening and within
    5 mm of the table), its tilt, its bottom's height and its label's bearing error."""
    o = np.asarray(st.origin, dtype=float)
    bottom = float(o[2] - L.TOP)
    k = position_of(st)
    down = abs(bottom) <= L.BOTTOM_TOL
    return {"pos": k if (k is not None and down) else None, "over_pos": k, "bottom_mm": round(bottom * 1000, 2),
            "tilt_deg": round(tilt_deg(st), 2), "bearing_err_deg": round(bearing_err_deg(st), 2),
            "upright": tilt_deg(st) <= L.TILT_MAX_DEG, "facing": bearing_err_deg(st) <= L.BEARING_MAX_DEG,
            "in_slot": k is not None and label_in_slot(st, k)}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each tube's bottom, top and label face over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in L.TUBES}
        self.prev_bottom: dict[str, np.ndarray] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every tube standing upright on its holding-rack mark."""
        for n in L.TUBES:
            o = np.asarray(objects[n].origin, dtype=float)
            mx, my = L.TUBE_START[n]
            if not (math.hypot(o[0] - mx, o[1] - my) < L.START_TOL and abs(o[2] - L.TOP) < 0.003
                    and tilt_deg(objects[n]) < 2.0):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest the tube's bottom, top or label face has been, over the last 0.5 s, from where it is now (m);
        None until 0.5 s of this trial has been seen. A tube spinning in place moves its label face."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        now = h[-1]
        return float(max(max(np.linalg.norm(p0 - p) for p0, p in zip(old, now)) for old in h))

    def update(self, objects, arms) -> Result:
        """One control step: ``objects`` (the four tubes' states) and ``arms`` (``rigstate.arms()``)."""
        if not all(valid(objects.get(n)) for n in L.TUBES):  # a blow-up: score 0, keep everything as it was
            self.steps += 1
            self.last = self.result(objects, arms)
            return self.last
        pts = {n: points(objects[n]) for n in L.TUBES}
        if self.prev_bottom and self.at_start(objects):
            jumped = any(float(np.linalg.norm(pts[n][0] - self.prev_bottom[n])) > L.RESET_JUMP for n in L.TUBES)
            if jumped:  # the simulator put the tubes back in the bench racks: a new episode
                self.reset()
        self.steps += 1
        for n in L.TUBES:
            self.history[n].append(pts[n])
            self.prev_bottom[n] = pts[n][0]
        self.last = self.result(objects, arms)
        return self.last

    def result(self, objects, arms) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        bad = [n for n in L.TUBES if not valid(objects.get(n))]
        if bad:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", "invalid": bad})
        poses = {n: pose(objects[n]) for n in L.TUBES}
        moved = {n: self.moved(n) for n in L.TUBES}
        rest = {n: moved[n] is not None and moved[n] <= L.STILL_TOL for n in L.TUBES}
        free = {n: released(arms, objects[n]) for n in L.TUBES}
        # how many tubes stand in each position (bottom in the opening, on the floor), at rest or not
        count = {k: sum(1 for n in L.TUBES if poses[n]["pos"] == k) for k in range(1, L.N_POS + 1)}
        tubes, readable = {}, 0
        for n in L.TUBES:
            p = poses[n]
            if p["over_pos"] is None:
                why = "not_in_the_analyser_rack"
            elif p["pos"] is None:
                why = "not_down_on_the_rack_floor"
            elif count[p["pos"]] > 1:
                why = "two_tubes_in_one_position"
            elif not p["upright"]:
                why = "tilted"
            elif not rest[n]:
                why = "still_moving"
            elif not free[n]:
                why = "gripper_not_clear"
            elif not p["facing"]:
                why = "label_turned_away"
            elif not p["in_slot"]:
                why = "label_behind_slot_post"
            else:
                why = ""
            racked = why in ("", "label_turned_away", "gripper_not_clear", "label_behind_slot_post")
            ok = why == ""
            readable += ok
            tubes[n] = {"pos": p["pos"], "racked": racked, "readable": ok, "why": why,
                        "bearing_err_deg": p["bearing_err_deg"], "tilt_deg": p["tilt_deg"],
                        "bottom_mm": p["bottom_mm"], "at_rest": rest[n], "released": free[n],
                        "moved_mm": None if moved[n] is None else round(moved[n] * 1000, 2)}
        score = L.PER_TUBE * readable
        success = bool(readable == len(L.TUBES))
        details = {"valid_state": True, "readable": readable, "arms_readable": rigstate.readable(arms),
                   "racked": sum(1 for n in L.TUBES if tubes[n]["racked"]), "tubes": tubes}
        return Result(score, success, details)
