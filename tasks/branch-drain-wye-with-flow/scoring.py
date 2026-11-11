"""The scorer, as plain functions of the three loose parts' states and the grippers' state. NumPy only; no simulator.

The score is read from the end state, as a plumber checks a dry fit before cementing it: which fitting is in the gap
of the drain, is it seated (flat on the bench, its run in line with the drain and centred in the gap), which way does
the wye's branch point, and does the branch pipe lie in line with the branch, its end at the opening. Nothing is
latched: dry-fitting the wrong fitting and swapping it leaves no trace, and the end state is all that matters. The
tracker keeps only each part's last second of positions and orientations (to tell a part at rest from one moving)
and whether a gripper was near it, and is fed one observation per control step by ``task.success()``:

    Tracker.update(parts, arms) -> Result(score, success, details)

``parts`` maps "wye", "tee" and "pipe" to their states. The two fittings are mechkit compounds (``AssemblyState``):
``origin`` is the fitting frame's origin (the middle of the run's bottom face), ``quat_xyzw`` that frame's
orientation (+y along the run, +z up when it stands upright). The pipe is a plain box (``ObjectState``): ``pos`` is
its centre, ``quat_xyzw`` its orientation (+y along its length). A state with a non-finite number, a zero quaternion
or a missing field (a simulator blow-up, a part not built as an assembly) scores 0 for that step and leaves the
tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'released' is taken as
met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so a gripper's
fingers are taken to be anywhere round its axis: its finger zone is every point within reach of a pad's outer corner
(half the current finger gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully open) around the
38 mm stretch from the point between its pad tips back up the gripper.

Decision list (first match wins), every condition read from parts that are counted (released and settled):
  1. the sanitary tee seated in the gap: 0
  2. no wye seated in the gap: 0
  3. the wye seated, its branch not 135 +- 7 degrees (front and leaning upstream): 0.25
  4. the wye seated and correct, the pipe on line (fine), the tee off the drain (resting on the bench on any face,
     its run's centre more than 60 mm from the gap's centre), all three still for 1 s: 1.0
  5. the wye seated and correct, the pipe on line (coarse): 0.75
  6. the wye seated and correct, the pipe anywhere else: 0.5
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

PARTS = ("wye", "tee", "pipe")
_RUN_CORNERS = np.array([[sx * L.RUN_SIZE[0] / 2, sy * L.RUN_SIZE[1] / 2, L.RUN_CENTRE[2] + sz * L.RUN_SIZE[2] / 2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
_PIPE_CORNERS = np.array([[sx * L.PIPE_SIZE[0] / 2, sy * L.PIPE_SIZE[1] / 2, sz * L.PIPE_SIZE[2] / 2]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
# each part's solid shapes in its own frame, for the grippers' distance: ("box", centre, half size) or
# ("disc", centre, radius, half height) (an upright cylinder along the frame's z)
SHAPES = {
    "wye": [("box", np.array(L.RUN_CENTRE), np.array(L.RUN_SIZE) / 2)]
    + [("disc", np.array(c), L.BEAD_R, L.SEC / 2) for c in L.WYE_BEADS],
    "tee": [("box", np.array(L.RUN_CENTRE), np.array(L.RUN_SIZE) / 2),
            ("box", np.array(L.TEE_BRANCH_CENTRE), np.array(L.TEE_BRANCH_SIZE) / 2)]
    + [("box", np.array(c), np.array(s) / 2) for c, s in L.TEE_SWEEP],
    "pipe": [("box", np.zeros(3), np.array(L.PIPE_SIZE) / 2)],
}


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


def _vec(st, key: str, n: int) -> np.ndarray | None:
    try:
        v = np.asarray(getattr(st, key), dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return None
    return v if v.size == n and np.all(np.isfinite(v)) else None


def valid(st, fitting: bool) -> bool:
    """Every number finite, the quaternion not degenerate; a fitting also needs its frame's ``origin``."""
    if st is None:
        return False
    q = _vec(st, "quat_xyzw", 4)
    if q is None or np.linalg.norm(q) < 0.5 or _vec(st, "pos", 3) is None:
        return False
    return _vec(st, "origin", 3) is not None if fitting else True


def frame(st, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of a part's frame: a fitting's base frame, or the pipe's centre."""
    key = "pos" if name == "pipe" else "origin"
    return np.asarray(getattr(st, key), dtype=float), rotation(st.quat_xyzw)


def to_world(st, name: str, local) -> np.ndarray:
    o, r = frame(st, name)
    return o + r @ np.asarray(local, dtype=float)


def centre(st, name: str) -> np.ndarray:
    """A fitting's run centre (mid-height), or the pipe's centre: the point tracked for stillness."""
    return to_world(st, name, (0.0, 0.0, 0.0) if name == "pipe" else L.RUN_CENTRE)


def lowest(st, name: str) -> float:
    """Height of the part's lowest corner (the run's, for a fitting) above the table top (m)."""
    o, r = frame(st, name)
    corners = _PIPE_CORNERS if name == "pipe" else _RUN_CORNERS
    return float(np.min(o[2] + corners @ r[2, :]) - L.TOP)


def up_tilt_deg(st) -> float:
    """Angle between a fitting's up axis and the vertical, either way up (0 upright or upside down, 90 on its side)."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(st.quat_xyzw)[2, 2]), 0.0, 1.0))))


def pipe_tilt_deg(st) -> float:
    """Angle between the pipe's length and the horizontal (its square section may lie on any of its four faces)."""
    return float(np.degrees(np.arcsin(np.clip(abs(rotation(st.quat_xyzw)[2, 1]), 0.0, 1.0))))


def _heading(v) -> np.ndarray | None:
    h = np.array([float(v[0]), float(v[1])])
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-9 else None


def run_axis_err_deg(st) -> float:
    """Angle between a fitting's run seen from above and the drain's axis (y), either way round (0..90)."""
    h = _heading(rotation(st.quat_xyzw)[:, 1])
    return 90.0 if h is None else float(np.degrees(math.acos(min(1.0, abs(h[1])))))


def flat_on_bench(st, name: str) -> bool:
    """Lying flat on the table: tilt under 5 degrees and the lowest point within 3 mm of the table top."""
    tilt = pipe_tilt_deg(st) if name == "pipe" else up_tilt_deg(st)
    return bool(tilt < L.FLAT_DEG and abs(lowest(st, name)) <= L.ON_TABLE)


def seat(st, name: str) -> dict:
    """Where a fitting is relative to the gap: flat on the bench, run along the drain, run centre in the gap."""
    c = centre(st, name)
    dx, dy = float(c[0] - L.GAP_CENTRE[0]), float(c[1] - L.GAP_CENTRE[1])
    tilt, axis, low = up_tilt_deg(st), run_axis_err_deg(st), lowest(st, name)
    ok = (tilt < L.FLAT_DEG and abs(low) <= L.ON_TABLE and axis <= L.SEAT_AXIS_DEG
          and abs(dx) <= L.SEAT_CENTRE and abs(dy) <= L.SEAT_CENTRE)
    return {"seated": bool(ok), "tilt_deg": round(tilt, 2), "lowest_mm": round(low * 1000, 2),
            "run_axis_err_deg": round(axis, 2), "off_x_mm": round(dx * 1000, 2), "off_y_mm": round(dy * 1000, 2)}


def branch(st) -> tuple[np.ndarray, np.ndarray, float]:
    """The wye's branch from its pose (flips included): (E, the opening's centre in the world; u, the branch's
    outward axis seen from above, unit 2D; theta_b, u's yaw in degrees, -180..180)."""
    r = rotation(st.quat_xyzw)
    e = to_world(st, "wye", L.WYE_TIP)
    u = _heading(r @ np.array([L.WYE_DIR[0], L.WYE_DIR[1], 0.0]))
    if u is None:  # the branch pointing straight up or down: no direction
        return e, np.array([0.0, 0.0]), float("nan")
    return e, u, float(math.degrees(math.atan2(u[1], u[0])))


def branch_err_deg(theta_b: float) -> float:
    """How far theta_b is from the correct 135 degrees (0..180; NaN counts as 180)."""
    if not math.isfinite(theta_b):
        return 180.0
    return abs((theta_b - L.BRANCH_OK_DEG + 180.0) % 360.0 - 180.0)


def pipe_line(pipe, wye) -> dict:
    """The branch pipe against the wye's branch axis: the angle between the pipe's length and u (either way round),
    the near end's centre (the end nearer the opening) measured from E along u (gap) and across it (lateral), and how
    far the pipe's centre lies beyond E along u."""
    e, u, _ = branch(wye)
    if not np.any(u):
        return {"angle_deg": 90.0, "gap_mm": None, "lateral_mm": None, "outward_mm": None, "g": math.inf,
                "lat": math.inf, "out": -math.inf}
    r = rotation(pipe.quat_xyzw)
    p = np.asarray(pipe.pos, dtype=float)
    a = _heading(r[:, 1])
    ang = 90.0 if a is None else float(np.degrees(math.acos(min(1.0, abs(float(a @ u))))))
    ends = [p + s * L.PIPE_HALF_LEN * r[:, 1] for s in (-1.0, 1.0)]
    near = min(ends, key=lambda q: float(np.linalg.norm(q[:2] - e[:2])))
    d = near[:2] - e[:2]
    n = np.array([u[1], -u[0]])
    g, lat, out = float(d @ u), float(d @ n), float((p[:2] - e[:2]) @ u)
    return {"angle_deg": round(ang, 2), "gap_mm": round(g * 1000, 2), "lateral_mm": round(lat * 1000, 2),
            "outward_mm": round(out * 1000, 2), "g": g, "lat": lat, "out": out}


def on_line(line: dict, tol: dict) -> bool:
    lo, hi = tol["gap"]
    return bool(line["angle_deg"] <= tol["angle_deg"] and abs(line["lat"]) <= tol["lateral"]
                and lo <= line["g"] <= hi and line["out"] >= L.OUTWARD)


def tee_lowest(st) -> float:
    """Height of the tee's lowest corner, over all its boxes (run, branch and sweep), above the table top (m)."""
    o, r = frame(st, "tee")
    corners = np.array([shape[1] + np.array([sx, sy, sz]) * shape[2] for shape in SHAPES["tee"]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return float(np.min(o[2] + corners @ r[2, :]) - L.TOP)


def tee_on_bench(st) -> bool:
    """The tee rests on the bench on any of its faces: one of its frame's axes within 5 degrees of vertical and its
    lowest corner within 3 mm of the table top (so a tee on the drain, on another part or leaning on one never
    counts; upright, upside down, on its side or on its end all do)."""
    r = rotation(st.quat_xyzw)
    square = float(np.max(np.abs(r[2, :]))) >= math.cos(math.radians(L.FLAT_DEG))
    return bool(square and abs(tee_lowest(st)) <= L.ON_TABLE)


def tee_clear(tee) -> tuple[bool, float]:
    """(the tee rests on the bench, on any face, with its run's centre more than 60 mm from the gap's centre, seen
    from above; that distance)."""
    c = centre(tee, "tee")
    dist = float(math.hypot(c[0] - L.GAP_CENTRE[0], c[1] - L.GAP_CENTRE[1]))
    return bool(tee_on_bench(tee) and dist > L.TEE_CLEAR), dist


# -- the grippers -------------------------------------------------------------------------------------------------
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


def distances(points, st, name: str) -> np.ndarray:
    """Distance from each of the points (N x 3) to the part's solid (0 inside it), from its shapes in its own
    frame."""
    o, r = frame(st, name)
    q = (np.atleast_2d(np.asarray(points, dtype=float)) - o) @ r  # each point in the part's frame
    best = np.full(len(q), np.inf)
    for shape in SHAPES[name]:
        if shape[0] == "box":
            d = np.linalg.norm(np.maximum(np.abs(q - shape[1]) - shape[2], 0.0), axis=1)
        else:
            c, rad, hh = shape[1], shape[2], shape[3]
            radial = np.maximum(0.0, np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) - rad)
            vertical = np.maximum(0.0, np.abs(q[:, 2] - c[2]) - hh)
            d = np.hypot(radial, vertical)
        best = np.minimum(best, d)
    return best


def point_distance(p, st, name: str) -> float:
    """Distance from a point to the part's solid (0 inside it)."""
    return float(distances(p, st, name)[0])


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, st, name: str, samples: int = 20) -> float:
    """How far a gripper's finger zone is from the part (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return float(np.min(distances(pts, st, name))) - zone_radius(arm)


def turn_deg(r0, r1) -> float:
    """The angle of the rotation from orientation r0 to r1 (degrees)."""
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return float(np.degrees(math.acos(min(1.0, max(-1.0, c)))))


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each part's centre, orientation and whether a gripper's finger zone
    was within 15 mm of it, over the last second (21 observations at 20 Hz), and how many steps the arms could not be
    read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=2 * self.window + 1)  # {part: (centre, rotation, hands_off)}
        self.arms_unavailable = 0
        self.jumps = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def states_valid(parts) -> bool:
        try:
            return all(valid(parts[n], n != "pipe") for n in PARTS)
        except (KeyError, TypeError):
            return False

    @staticmethod
    def hands_off(parts, arms) -> dict:
        """Per part: both grippers' finger zones at least 15 mm from it (True for every part if the arms cannot be
        read)."""
        if not arms_valid(arms):
            return {n: True for n in PARTS}
        return {n: min(zone_gap(arms[s], parts[n], n) for s in ("left", "right")) >= L.HANDS_OFF for n in PARTS}

    def _observe(self, parts, arms) -> dict:
        off = self.hands_off(parts, arms)
        return {n: (centre(parts[n], n), rotation(parts[n].quat_xyzw), off[n]) for n in PARTS}

    def update(self, parts, arms=None) -> Result:
        if not self.states_valid(parts):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(parts, arms)
            return self.last
        obs = self._observe(parts, arms)
        if self.history:
            prev = self.history[-1]
            if any(float(np.linalg.norm(obs[n][0] - prev[n][0])) > L.RESET_JUMP for n in PARTS):
                self.history.clear()  # the simulator put a part back (a new episode): forget the last second
                self.jumps += 1
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.history.append(obs)
        self.last = self.result(parts, arms)
        return self.last

    def window_motion(self, name: str, k: int) -> tuple[float, float, bool] | None:
        """Over the k-th 0.5 s window back (0 the last, 1 the one before): the part's centre path (m), its turning
        (degrees, summed step by step) and whether the grippers stayed clear at every step; None if not yet seen."""
        w = self.window
        h = list(self.history)
        if len(h) < (k + 1) * w + 1:
            return None
        seg = h[len(h) - (k + 1) * w - 1: len(h) - k * w]
        path = sum(float(np.linalg.norm(b[name][0] - a[name][0])) for a, b in zip(seg, seg[1:]))
        turn = sum(turn_deg(a[name][1], b[name][1]) for a, b in zip(seg, seg[1:]))
        return path, turn, all(o[name][2] for o in seg)

    def status(self, name: str, off_now: bool = True) -> dict:
        """counted (released over the last 0.5 s and at the step at hand, and settled over the last 0.5 s) and
        still_1s (counted, and settled in each of the last two 0.5 s windows), with the numbers."""
        m0, m1 = self.window_motion(name, 0), self.window_motion(name, 1)

        def settled(m):
            return m is not None and m[0] < L.STILL_PATH and m[1] < L.STILL_TURN_DEG

        counted = settled(m0) and bool(m0[2]) and bool(off_now)
        return {"counted": bool(counted), "still_1s": bool(counted and settled(m1)),
                "released": None if m0 is None else bool(m0[2]),
                "path_mm": None if m0 is None else round(m0[0] * 1000, 2),
                "turn_deg": None if m0 is None else round(m0[1], 2)}

    def result(self, parts, arms=None) -> Result:
        """The score of the state at hand with the history fed so far (idempotent: it changes nothing)."""
        if not self.states_valid(parts):
            return Result(0.0, False, {"valid_state": False, "line": 0, "zero_reason": "invalid_state"})
        wye, tee, pipe = parts["wye"], parts["tee"], parts["pipe"]
        off_now = self.hands_off(parts, arms)
        st = {n: self.status(n, off_now[n]) for n in PARTS}
        ws, ts = seat(wye, "wye"), seat(tee, "tee")
        e, u, theta_b = branch(wye)
        line = pipe_line(pipe, wye)
        clear, tee_dist = tee_clear(tee)
        pipe_flat = flat_on_bench(pipe, "pipe")
        fine = pipe_flat and on_line(line, L.FINE)
        coarse = pipe_flat and on_line(line, L.COARSE)
        d = {"valid_state": True, "arms_unavailable": self.arms_unavailable, "jumps": self.jumps,
             "wye": {**ws, **st["wye"]}, "tee": {**ts, **st["tee"]}, "pipe": {**st["pipe"]},
             "theta_b_deg": None if not math.isfinite(theta_b) else round(theta_b, 2),
             "branch_err_deg": round(branch_err_deg(theta_b), 2), "tip_E": [round(float(v), 4) for v in e],
             "u": [round(float(v), 4) for v in u], "pipe_flat": pipe_flat,
             "pipe_tilt_deg": round(pipe_tilt_deg(pipe), 2), "pipe_lowest_mm": round(lowest(pipe, "pipe") * 1000, 2),
             **{k: v for k, v in line.items() if k not in ("g", "lat", "out")},
             "pipe_fine": fine, "pipe_coarse": coarse, "tee_clear": clear, "tee_dist_mm": round(tee_dist * 1000, 1)}
        if ts["seated"] and st["tee"]["counted"]:
            score, n, why = 0.0, 1, "sanitary_tee_in_the_gap"
        elif not (ws["seated"] and st["wye"]["counted"]):
            score, n = 0.0, 2
            why = "wye_held_or_moving" if ws["seated"] else "no_wye_seated"
        elif branch_err_deg(theta_b) > L.BRANCH_TOL_DEG:
            score, n, why = 0.25, 3, "wye_branch_wrong_way"
        elif fine and st["pipe"]["counted"] and clear and all(st[k]["still_1s"] for k in PARTS):
            score, n, why = 1.0, 4, ""
        elif coarse and st["pipe"]["counted"]:
            score, n, why = 0.75, 5, ""
        else:
            score, n, why = 0.5, 6, ""
        d.update(line=n, zero_reason=why)
        return Result(score, bool(score == 1.0), d)
