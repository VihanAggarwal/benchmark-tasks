"""The scorer, as plain functions of the packers' and the beam's states. NumPy only; no simulator needed.

The task is judged on its end state: each pier packed up to the datum with packers lying flat and fully on it,
then the beam seated level across both. The scorer keeps only a short memory (where each loose object was over
the last 0.5 s, to tell whether it is still) and is fed one observation per control step by ``task.success()``.

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps names to anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has; the six packers and
the beam must be there (fixed objects are not read). A state with a non-finite number or a zero quaternion (a
simulator blow-up) scores 0 for that step and leaves the tracker as it was.

A packer is a 50 x 50 mm block whose two 50 x 50 faces are alike, so either may face down; its thickness axis is
its body z. The beam's length is its body y; it may lie on any of its four long faces.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_FIELDS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")


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


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def packer_size(name: str) -> np.ndarray:
    t = int(name.split("_")[1])
    return np.array([L.PACKER_SIDE, L.PACKER_SIDE, L.PACKERS[t][0]])


def size_of(name: str) -> np.ndarray:
    return np.array(L.BEAM_SIZE) if name == "beam" else packer_size(name)


def thickness_tilt_deg(s) -> float:
    """Angle between the vertical and a packer's thickness axis (0..90 degrees; either 50 x 50 face may be down)."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(s.quat_xyzw)[2, 2]), 0.0, 1.0))))


def outline(s, size) -> np.ndarray:
    """The object's real outline seen from above, as the shadow of its eight corners, (8, 2)."""
    r = rotation(s.quat_xyzw)
    half = np.asarray(size) / 2
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return (np.asarray(s.pos, dtype=float) + (signs * half) @ r.T)[:, :2]


def face(s, size, up: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """The face nearest to facing up (``up``) or down: (centre, unit normal pointing up, axis e1, axis e2,
    half-length along e1, half-length along e2)."""
    r = rotation(s.quat_xyzw)
    half = np.asarray(size, dtype=float) / 2
    i = int(np.argmax(np.abs(r[2, :])))
    n = r[:, i] if r[2, i] >= 0.0 else -r[:, i]
    j, k = [a for a in range(3) if a != i]
    c = np.asarray(s.pos, dtype=float) + (n if up else -n) * half[i]
    return c, n, r[:, j], r[:, k], float(half[j]), float(half[k])


def plane_z(f, x: float, y: float) -> float:
    """Height of a face's plane above the point (x, y)."""
    c, n = f[0], f[1]
    return float(c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2])


def pier_top(side: str) -> float:
    return L.TOP + L.PIER_H[side]


def pier_face(side: str):
    c = np.array([L.PIER_X, L.SIGN[side] * L.PIER_Y, pier_top(side)])
    return c, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), L.PIER_HALF, L.PIER_HALF


def face_samples(f, n: int = L.FACE_SAMPLES) -> np.ndarray:
    """An n x n grid of points over a face, edges and corners included, (n*n, 3)."""
    c, _, e1, e2, h1, h2 = f
    u = np.linspace(-1.0, 1.0, n)
    a, b = np.meshgrid(u * h1, u * h2, indexing="ij")
    return c + a.reshape(-1, 1) * e1 + b.reshape(-1, 1) * e2


def on_pier(s, side: str) -> bool:
    """A packer is on a pier: its centre seen from above over the pier's top or within 15 mm of it, and its lowest
    point more than 10 mm above the table (not lying on the table beside the pier)."""
    m = L.PIER_HALF + L.ON_PIER_MARGIN
    dx, dy = s.pos[0] - L.PIER_X, s.pos[1] - L.SIGN[side] * L.PIER_Y
    return bool(abs(dx) <= m and abs(dy) <= m and s.low[2] > L.TOP + L.ON_PIER_MIN_LOW)


def overhang(s, size, side: str) -> float:
    """How far (m) the packer's outline seen from above reaches past the pier's 60 x 60 mm top (0 if inside)."""
    pts = outline(s, size)
    dx = np.abs(pts[:, 0] - L.PIER_X) - L.PIER_HALF
    dy = np.abs(pts[:, 1] - L.SIGN[side] * L.PIER_Y) - L.PIER_HALF
    return float(max(0.0, dx.max(), dy.max()))


def over_face(x: float, y: float, f) -> bool:
    """Is the point (x, y) seen from above inside the face's outline?"""
    c, n, e1, e2, h1, h2 = f
    d = np.array([x - c[0], y - c[1]])
    m = np.array([[e1[0], e2[0]], [e1[1], e2[1]]])
    if abs(np.linalg.det(m)) < 1e-9:
        return False
    s, t = np.linalg.solve(m, d)
    return bool(abs(s) <= h1 + 1e-9 and abs(t) <= h2 + 1e-9)


# -- the beam -------------------------------------------------------------------------------------------------


def beam_angles(beam) -> dict:
    """level_deg: its length from horizontal; face_deg: the axis through its nearest-to-vertical pair of long faces
    from vertical; heading_deg: its length seen from above from the line through the piers (y), 0..90 (reported
    only: a centre line that crosses both top packers is within 9.2 deg of y)."""
    r = rotation(beam.quat_xyzw)
    a = r[:, 1]
    level = math.degrees(math.asin(min(1.0, abs(float(a[2])))))
    face_ax = max(abs(float(r[2, 0])), abs(float(r[2, 2])))
    face_deg = math.degrees(math.acos(min(1.0, face_ax)))
    heading = math.degrees(math.atan2(abs(float(a[0])), abs(float(a[1])))) if math.hypot(a[0], a[1]) > 1e-9 else 90.0
    return {"level_deg": level, "face_deg": face_deg, "heading_deg": heading}


def beam_gap(beam, f) -> float | None:
    """Vertical gap (m) between the beam's underside and a support face where they meet: the smallest gap between
    the underside's plane and any sampled point of the face that lies under the beam (seen from above). None if no
    part of the face lies under the beam."""
    c, n, e1, e2, h1, h2 = face(beam, L.BEAM_SIZE, up=False)
    pts = face_samples(f)
    m = np.array([[e1[0], e2[0]], [e1[1], e2[1]]])
    if abs(np.linalg.det(m)) < 1e-9:  # the beam stands on end: no long face down
        return None
    st = np.linalg.solve(m, (pts[:, :2] - c[:2]).T).T
    inside = (np.abs(st[:, 0]) <= h1 + 1e-9) & (np.abs(st[:, 1]) <= h2 + 1e-9)
    if not inside.any():
        return None
    z_under = c[2] + st[inside, 0] * e1[2] + st[inside, 1] * e2[2]
    return float(np.min(z_under - pts[inside, 2]))


def rests(gap: float | None) -> bool:
    return gap is not None and -L.BEAM_SINK_TOL <= gap <= L.BEAM_REST_TOL


def bearing(beam, f) -> tuple[float, bool]:
    """(length of the beam over the face along its length (m), whether the beam's centre line crosses the face)."""
    r = rotation(beam.quat_xyzw)
    a = r[:, 1]
    ah = np.array([a[0], a[1]])
    nh = float(np.hypot(*ah))
    if nh < 1e-9:
        return 0.0, False
    ah = ah / nh
    wh = np.array([-ah[1], ah[0]])
    c, _, e1, e2, h1, h2 = f
    corners = np.array([c + sx * h1 * e1 + sy * h2 * e2 for sx in (-1, 1) for sy in (-1, 1)])[:, :2]
    d = corners - np.asarray(beam.pos, dtype=float)[:2]
    along = d @ ah / nh  # horizontal distance along the beam, as a distance along its (possibly tilted) length
    across = d @ wh
    half = L.BEAM_LEN / 2
    over = min(float(along.max()), half) - max(float(along.min()), -half)
    return max(0.0, over), bool(across.min() <= 0.0 <= across.max())


# -- the tracker ----------------------------------------------------------------------------------------------


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _angle_deg(q1, q2) -> float:
    a, b = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    d = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


class Tracker:
    """Where each loose object has been over the last 0.5 s (to tell whether it is still)."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = int(round(L.STILL_WINDOW_S * control_hz)) + 1  # this step and the previous 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hist: dict[str, deque] = {n: deque(maxlen=self.window) for n in L.LOOSE}
        self.prev: dict[str, np.ndarray] | None = None
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every loose object flat on the table within 4.5 cm of its mark."""
        for side in L.SIDES:
            for t in L.PACKERS:
                s = objects[L.packer_name(t, side)]
                mark = (L.SET_X[t], L.SIGN[side] * L.SET_Y)
                if (math.hypot(s.pos[0] - mark[0], s.pos[1] - mark[1]) > L.START_TOL or abs(s.low[2] - L.TOP) > 0.003
                        or thickness_tilt_deg(s) > 1.0):
                    return False
        b = objects["beam"]
        ang = beam_angles(b)
        return bool(math.hypot(b.pos[0] - L.BEAM_X, b.pos[1] - L.BEAM_Y) <= L.START_TOL
                    and abs(b.low[2] - L.TOP) <= 0.003 and ang["level_deg"] < 1.0 and ang["face_deg"] < 1.0)

    def still(self, name: str) -> bool:
        """Moved under 3 mm and turned under 2 degrees over the last 0.5 s (a full window of observations)."""
        h = self.hist[name]
        if len(h) < self.window:
            return False
        p_now, q_now = h[-1]
        return all(float(np.linalg.norm(p_now - p)) < L.STILL_MOVE and _angle_deg(q_now, q) < L.STILL_TURN_DEG
                   for p, q in h)

    def update(self, objects) -> Result:
        self.steps += 1
        if not all(n in objects and valid(objects[n]) for n in L.LOOSE):  # a blow-up: 0, keep the record
            self.last = Result(0.0, False, {"valid_state": False})
            return self.last
        now = {n: np.array(objects[n].pos, dtype=float) for n in L.LOOSE}
        if self.prev is not None and self.at_start(objects):
            if max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.LOOSE) > L.RESET_JUMP:
                steps = self.steps
                self.reset()  # the simulator put everything back: a new episode
                self.steps = steps
        for n in L.LOOSE:
            self.hist[n].append((now[n], np.array(objects[n].quat_xyzw, dtype=float)))
        self.prev = now
        self.last = self.result(objects)
        return self.last

    # -- the rule ---------------------------------------------------------------------------------------------
    def pier(self, objects, side: str) -> dict:
        """What is on one pier: which packers, which of them count, the packing's top, whether it is packed."""
        top = pier_top(side)
        on = [n for n in L.PACKER_NAMES if on_pier(objects[n], side)]
        on.sort(key=lambda n: float(objects[n].low[2]))
        counted: list[str] = []
        rows = {}
        for n in on:
            s, size = objects[n], packer_size(n)
            tilt = thickness_tilt_deg(s)
            hang = overhang(s, size, side)
            bottom = face(s, size, up=False)
            x, y = float(s.pos[0]), float(s.pos[1])
            under = plane_z(bottom, x, y)
            rest_on = "pier" if abs(under - top) <= L.REST_TOL else None
            if rest_on is None:
                for m in counted:
                    tf = face(objects[m], packer_size(m), up=True)
                    if over_face(x, y, tf) and abs(under - plane_z(tf, x, y)) <= L.REST_TOL:
                        rest_on = m
                        break
            still = self.still(n)
            ok = tilt <= L.FLAT_DEG and hang <= L.OVERHANG + 1e-9 and rest_on is not None and still
            if ok:
                counted.append(n)
            rows[n] = {"tilt_deg": round(tilt, 2), "overhang_mm": round(hang * 1000, 1), "rests_on": rest_on,
                       "still": still, "counts": ok, "on_edge": tilt > L.EDGE_DEG}
        if on:
            highest = max(on, key=lambda n: float(objects[n].high[2]))
            level = face(objects[highest], packer_size(highest), up=True)[0][2]
        else:
            highest, level = None, top
        packed = bool(on) and len(counted) == len(on) and abs(level - (L.TOP + L.DATUM)) <= L.DATUM_TOL
        return {"on": on, "counted": counted, "rows": rows, "top_packer": highest, "level": float(level),
                "packed": packed, "on_edge": any(r["on_edge"] for r in rows.values())}

    def beam_on(self, objects, side: str, pier: dict) -> dict:
        """Where the beam meets one pier: does it rest on the pier or on a packer there, on the top packer, and how
        much of its length is over the top packer."""
        beam = objects["beam"]
        supports = [("pier", pier_face(side))] + [(n, face(objects[n], packer_size(n), up=True)) for n in pier["on"]]
        gaps = {k: beam_gap(beam, f) for k, f in supports}
        rest_any = any(rests(g) for g in gaps.values())
        top = pier["top_packer"]
        if top is not None:
            over, crosses = bearing(beam, supports[[k for k, _ in supports].index(top)][1])
            on_top = rests(gaps[top])
        else:
            over, crosses, on_top = 0.0, False, False
        g = gaps.get(top) if top is not None else gaps["pier"]
        return {"rests": rest_any, "on_top_packer": on_top, "bearing": over, "centre_line_over": crosses,
                "gap": g}

    def result(self, objects) -> Result:
        if not all(n in objects and valid(objects[n]) for n in L.LOOSE):
            return Result(0.0, False, {"valid_state": False})
        piers = {side: self.pier(objects, side) for side in L.SIDES}
        meets = {side: self.beam_on(objects, side, piers[side]) for side in L.SIDES}
        ang = beam_angles(objects["beam"])
        beam_still = self.still("beam")
        across = all(meets[s]["rests"] for s in L.SIDES)
        tilted_across = across and ang["level_deg"] > L.BEAM_LEVEL_DEG
        on_edge = any(piers[s]["on_edge"] for s in L.SIDES)
        both = all(piers[s]["packed"] for s in L.SIDES)
        seated = bool(
            both and ang["face_deg"] <= L.BEAM_FACE_DEG and ang["level_deg"] <= L.BEAM_LEVEL_DEG and beam_still
            and all(meets[s]["on_top_packer"] and meets[s]["bearing"] >= L.BEARING_MIN - 1e-9
                    and meets[s]["centre_line_over"] for s in L.SIDES))
        if on_edge or tilted_across:
            score = 0.0
        else:
            score = 0.25 * sum(piers[s]["packed"] for s in L.SIDES) + (0.5 if seated else 0.0)
        details = {"valid_state": True, "zero_on_edge": on_edge, "zero_beam_tilted_across": tilted_across,
                   "beam_level_deg": round(ang["level_deg"], 2), "beam_face_deg": round(ang["face_deg"], 2),
                   "beam_heading_deg": round(ang["heading_deg"], 2), "beam_still": beam_still,
                   "beam_across_both": across, "beam_seated": seated}
        for s in L.SIDES:
            p, m = piers[s], meets[s]
            details[f"{s}_packers"] = p["on"]
            details[f"{s}_counted"] = p["counted"]
            details[f"{s}_level_mm"] = round((p["level"] - L.TOP) * 1000, 2)
            details[f"{s}_packed"] = p["packed"]
            details[f"{s}_rows"] = p["rows"]
            details[f"{s}_beam_rests"] = m["rests"]
            details[f"{s}_beam_gap_mm"] = None if m["gap"] is None else round(m["gap"] * 1000, 2)
            details[f"{s}_bearing_mm"] = round(m["bearing"] * 1000, 1)
            details[f"{s}_centre_line_over"] = m["centre_line_over"]
        return Result(score, bool(score == 1.0), details)
