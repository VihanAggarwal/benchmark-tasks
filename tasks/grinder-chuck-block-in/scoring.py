"""The scorer, as plain functions of the loose objects' states and the arms' state. NumPy only; no simulator needed.

The score is the end state a setter checks before starting the grinder's table: the part lying flat on the chuck
against the rail, across a brass line, and blocked at both ends by pieces lower than the part, with no piece taller
than the part left on the chuck and nothing still held. The scorer keeps a little state (the last 0.5 s of each
object's path, for 'still', and counts for the trace) and is fed one observation per control step by
``task.success()``.

    Tracker.update(states, arms) -> Result(score, success, details)

``states`` maps each loose object's name (``layout.LOOSE``: the part and the four blocking pieces) to anything with
``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has; every object is a box of ``layout.SIZES[name]``, so its corners follow
from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``,
``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could
not be read; then every item is taken as let go for that step and the step is counted in ``arms_unavailable``. A
state with a non-finite number or a zero quaternion (a simulator blow-up), or a missing object, scores 0 for that
step and leaves the tracker as it was.

The blocking pieces are told apart only by their size, never by their names: any piece lying flat on the chuck
face with its top at least 1 mm below the part's top can block an end, and any piece on the chuck whose top is
more than 21 mm above the face (the part is 20 mm tall) scores 0. Every object is a plain box, the same either way
up, so 'tilt' is measured from whichever face is nearest to facing down.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is a solid cylinder about the axis from the point between its finger tips 38 mm back up the gripper, flat at
the tips (the fingers end there: a gripper hovering above an item is as far from it as its finger tips are); the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across; 57 mm fully open). An item within 2 mm of a finger zone is still
held and does not count (a finger resting on an item counts as holding it).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 control steps


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
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def all_valid(states) -> bool:
    """Every loose object present and valid."""
    try:
        return all(valid(states[n]) for n in L.LOOSE)
    except (KeyError, TypeError):
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


# -- boxes -----------------------------------------------------------------------------------------------------
def half(name: str) -> np.ndarray:
    return np.array(L.SIZES[name], dtype=float) / 2


def corners(s, name: str) -> np.ndarray:
    """The box's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half(name)) @ rotation(s.quat_xyzw).T


def lowest(s, name: str) -> float:
    return float(corners(s, name)[:, 2].min())


def highest(s, name: str) -> float:
    return float(corners(s, name)[:, 2].max())


def down_axis(s) -> tuple[int, float]:
    """Which body axis is nearest to vertical (0 x, 1 y, 2 z) and its angle from the vertical (degrees): the box
    rests on the pair of faces across that axis, either way up."""
    r = rotation(s.quat_xyzw)
    k = int(np.argmax(np.abs(r[2, :])))
    return k, float(np.degrees(np.arccos(np.clip(abs(r[2, k]), 0.0, 1.0))))


def flat_axes(s, name: str) -> list[tuple[np.ndarray, float]]:
    """The box's two horizontal body axes (unit vectors seen from above) with its half sizes along them."""
    r = rotation(s.quat_xyzw)
    k, _ = down_axis(s)
    out = []
    for j in range(3):
        if j == k:
            continue
        v = r[:2, j]
        n = float(np.linalg.norm(v))
        out.append((v / n if n > 1e-9 else np.array([1.0, 0.0]), float(half(name)[j])))
    return out


def footprint(s, name: str) -> np.ndarray:
    """The box's outline seen from above: the convex hull of its eight corners' shadows, counter-clockwise."""
    return _hull(corners(s, name)[:, :2])


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def side(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = side(pts), side(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1])


def overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes.)"""
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for a in axes:
        p, q = poly @ a, rect @ a
        if p.max() < q.min() - 1e-12 or q.max() < p.min() - 1e-12:
            return False
    return True


def over_chuck(s, name: str) -> bool:
    """Some of the box's outline seen from above is over the chuck."""
    return overlaps_rect(footprint(s, name), L.CHUCK_X0, L.CHUCK_X1, L.CHUCK_Y0, L.CHUCK_Y1)


def wholly_over_chuck(s, name: str) -> bool:
    c = corners(s, name)
    return bool(np.all((c[:, 0] >= L.CHUCK_X0) & (c[:, 0] <= L.CHUCK_X1)
                       & (c[:, 1] >= L.CHUCK_Y0) & (c[:, 1] <= L.CHUCK_Y1)))


def point_box_distance(p, s, name: str) -> float:
    """Distance from a point to the box (0 inside it), in the box's own frame."""
    r = rotation(s.quat_xyzw)
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - half(name), 0.0)))


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_axis(arm) -> tuple[np.ndarray, np.ndarray]:
    """The point between the finger tips and the unit vector from there back up the gripper (toward the TCP)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    return tips, (u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0]))


def _perpendiculars(u) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to each other and to the unit vector u."""
    e1 = np.cross(u, np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(u, e1)


_AXIS_S = np.linspace(0.0, L.FINGER_LEN, 20)  # sample points along the zone's axis (from the tips up)
_RING = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)  # and round the rims of its two flat ends
_COS, _SIN = np.cos(_RING), np.sin(_RING)


def _cylinder_gap(tips, u, r, q) -> np.ndarray:
    """Distances from the points q (n x 3) to the solid cylinder of radius r about the axis from ``tips`` along
    ``u`` for FINGER_LEN (negative inside it, by the radial depth). Below the tips' plane the distance is measured
    from that flat end, not from a rounded cap: the fingers end at their tips."""
    v = q - tips
    a = v @ u
    rho = np.linalg.norm(v - np.outer(a, u), axis=1)
    da = np.maximum(0.0, np.maximum(-a, a - L.FINGER_LEN))
    dr = rho - r
    return np.where((da == 0.0) & (dr <= 0.0), dr, np.hypot(da, np.maximum(0.0, dr)))


def finger_zone(arm) -> tuple[np.ndarray, float]:
    """Points on the zone's axis and on the rims of its two flat ends (the tips' plane and 38 mm up), and its
    radius."""
    tips, u = finger_axis(arm)
    r = zone_radius(arm)
    e1, e2 = _perpendiculars(u)
    circle = r * (np.outer(_COS, e1) + np.outer(_SIN, e2))
    return np.vstack([tips + np.outer(_AXIS_S, u), tips + circle, tips + L.FINGER_LEN * u + circle]), r


def _closest_on_box(s, name: str, pts) -> np.ndarray:
    """The point of the box nearest to each of the points (n x 3); a point inside it is its own nearest point."""
    c, rot = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)
    local = (np.asarray(pts, dtype=float) - c) @ rot  # each row: the point in the box's frame
    h = half(name)
    return np.clip(local, -h, h) @ rot.T + c


def zone_gap(arm, s, name: str) -> float:
    """How far a gripper's finger zone is from a box (negative: they overlap). The zone is a flat-ended cylinder
    (the fingers end at their tips), so a gripper whose finger tips are 30 mm straight above a box is 30 mm from it
    whatever its opening. The smallest distance is searched over the box's points nearest to the zone's axis and
    rims, and the box's corners."""
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([_closest_on_box(s, name, pts), corners(s, name)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


def released(arms, s, name: str) -> bool:
    """Neither gripper's finger zone comes within 2 mm of the box (unreadable arms: taken as let go)."""
    if not arms_valid(arms):
        return True
    return bool(all(zone_gap(arms[a], s, name) >= L.HANDS_OFF for a in ("left", "right")))


# -- the part ---------------------------------------------------------------------------------------------------
def part_axes(part) -> tuple[np.ndarray, np.ndarray]:
    """The part's long (40 mm) and short (30 mm) body axes seen from above, unit vectors; the long one points to
    +x (the far end), the short one to +y (the rail's side)."""
    r = rotation(part.quat_xyzw)
    u, v = r[:2, 0].copy(), r[:2, 1].copy()
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    if u[0] < 0:
        u = -u
    if v[1] < 0:
        v = -v
    return u, v


def part_flat(part) -> bool:
    """Lying on one of its 40 x 30 mm faces (its 20 mm axis vertical within 3 degrees), its lowest point within
    2 mm of the chuck face."""
    k, tilt = down_axis(part)
    return bool(k == 2 and tilt < L.PART_TILT_DEG and abs(lowest(part, "part") - L.FACE_Z) <= L.PART_FACE_TOL)


def part_square_deg(part) -> float:
    """How far the part's long side is turned from the rail's direction (x), seen from above (degrees, 0..90)."""
    u, _ = part_axes(part)
    return float(np.degrees(np.arccos(np.clip(abs(u[0]), 0.0, 1.0))))


def rail_gaps(part) -> tuple[float, float]:
    """From the rail's face to each of the two lower corners of the part's side facing the rail (m; + = in front
    of the rail, - = past its face), near (smaller x) first."""
    c = corners(part, "part")
    low4 = c[np.argsort(c[:, 2])[:4]]  # the bottom face (tilted at most a few degrees)
    back = low4[np.argsort(low4[:, 1])[2:]]  # its two corners nearest the rail
    back = back[np.argsort(back[:, 0])]
    return float(L.RAIL_FACE_Y - back[0, 1]), float(L.RAIL_FACE_Y - back[1, 1])


def against_rail(part) -> bool:
    """R: flat on the face, wholly over the chuck, its long side within 5 degrees of the rail and both corners of
    the side facing the rail within 3 mm of the rail's face."""
    if not (part_flat(part) and wholly_over_chuck(part, "part") and part_square_deg(part) <= L.SQUARE_DEG):
        return False
    return all(-L.RAIL_PEN <= g <= L.RAIL_GAP for g in rail_gaps(part))


def part_ends_x(part) -> tuple[float, float]:
    """x of the centres of the part's two end faces (its 30 x 20 mm faces), near end first."""
    u, _ = part_axes(part)
    c = np.asarray(part.pos, dtype=float)[:2]
    a, b = float((c - u * L.PART_HALF[0])[0]), float((c + u * L.PART_HALF[0])[0])
    return min(a, b), max(a, b)


def line_steel(part) -> tuple[float | None, float, float]:
    """The brass line the part bridges best and how much of its length lies on steel either side of that line
    (m; near side, far side): from the line's edge to the end face's centre, along the rail."""
    near, far = part_ends_x(part)
    best = (None, -math.inf, -math.inf)
    for x in L.LINES_X:
        s_near, s_far = (x - L.LINE_W / 2) - near, far - (x + L.LINE_W / 2)
        if min(s_near, s_far) > min(best[1], best[2]):
            best = (x, s_near, s_far)
    return best


def across_line(part) -> bool:
    """B: a brass line under the part with at least 8 mm of its length on steel on each side."""
    _, a, b = line_steel(part)
    return bool(min(a, b) >= L.LINE_STEEL)


# -- the blocking pieces -----------------------------------------------------------------------------------------
def piece_on_chuck(s, name: str) -> bool:
    """On the chuck: some of its outline over the chuck and its lowest point at most 21 mm above the face (resting
    on the face, on the rail, or on something else on the chuck)."""
    return bool(over_chuck(s, name) and lowest(s, name) - L.FACE_Z <= L.ON_CHUCK)


def too_tall(s, name: str) -> bool:
    """On the chuck with its top more than 21 mm above the face: higher than the part, in the wheel's path."""
    return bool(piece_on_chuck(s, name) and highest(s, name) - L.FACE_Z > L.TOO_TALL)


def piece_flat_low(s, name: str) -> bool:
    """Lying flat on the chuck face (resting on one of its faces, tilted under 5 degrees, its lowest point within
    2 mm of the face) with its top at most 19 mm above the face (lower than the part)."""
    _, tilt = down_axis(s)
    return bool(tilt < L.PIECE_TILT_DEG and abs(lowest(s, name) - L.FACE_Z) <= L.PIECE_FACE_TOL
                and highest(s, name) - L.FACE_Z <= L.LOW_MAX and over_chuck(s, name))


def end_fit(part, s, name: str, sign: int) -> dict:
    """How a piece sits against one end of the part (sign -1 the near end, +1 the far end): 'gap' from the end
    face to the piece's nearest point along the part's length (m), 'angle' between the end face and the piece's
    side facing it (degrees), 'overlap' of the piece with the end face across the part's depth (m)."""
    u, v = part_axes(part)
    n = u * sign
    c = np.asarray(part.pos, dtype=float)[:2]
    e = c + n * L.PART_HALF[0]
    (a, ha), (b, hb) = flat_axes(s, name)
    p = np.asarray(s.pos, dtype=float)[:2]
    pts = np.array([p + sa * a * ha + sb * b * hb for sa in (-1, 1) for sb in (-1, 1)])  # its outline's corners
    gap = float(min((q - e) @ n for q in pts))
    angle = float(np.degrees(np.arccos(np.clip(max(abs(a @ n), abs(b @ n)), 0.0, 1.0))))
    w = np.array([-n[1], n[0]])
    depth = L.PART_HALF[1] * abs(float(v @ w))  # the end face's half depth seen along w
    span = [float((q - c) @ w) for q in pts]
    overlap = float(min(max(span), depth) - max(min(span), -depth))
    return {"gap": gap, "angle": angle, "overlap": overlap}


def blocks_end(part, s, name: str, sign: int, arms=None) -> bool:
    """E: a piece lying flat on the face, lower than the part, let go, its nearest point within 3 mm of the end
    face (and not 2 mm into it), its side facing the part parallel to the end face within 10 degrees, covering at
    least 10 mm of the end face's 30 mm depth."""
    if not piece_flat_low(s, name):
        return False
    f = end_fit(part, s, name, sign)
    return bool(-L.END_PEN <= f["gap"] <= L.END_GAP and f["angle"] <= L.END_PARALLEL_DEG
                and f["overlap"] >= L.END_OVERLAP and released(arms, s, name))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: the last 0.5 s of each object's path, and counts for the trace."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.paths: dict[str, deque] = {n: deque(maxlen=_WINDOW + 1) for n in L.LOOSE}  # recent centres
        self.speeds: dict[str, float] = {n: 0.0 for n in L.LOOSE}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite or missing object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, velocity) of the part at the previous valid control step

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: the part flat on the table near its starting mark, still."""
        part = states["part"]
        near = math.hypot(float(part.pos[0]) - L.PART_X0, float(part.pos[1]) - L.PART_Y0) < L.START_TOL
        k, tilt = down_axis(part)
        on_table = abs(lowest(part, "part") - L.TOP) < 0.003 and k == 2 and tilt < 1.0
        return bool(near and on_table and np.linalg.norm(part.lin_vel) < L.STILL_SPEED)

    def _teleported(self, c) -> bool:
        """The part moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        p0, v0 = self.prev
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall

    def update(self, states, arms=None) -> Result:
        if not all_valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states, arms)
            return self.last
        c = np.array(states["part"].pos, dtype=float)
        if self.prev is not None and self._teleported(c) and self.at_start(states):
            self.reset()  # the simulator put the part back at the start: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = (c, np.array(states["part"].lin_vel, dtype=float))
        for n in L.LOOSE:
            self.paths[n].append(np.array(states[n].pos, dtype=float))
            self.speeds[n] = float(np.linalg.norm(np.asarray(states[n].lin_vel, dtype=float)))
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(states, arms)
        return self.last

    def still(self, name: str) -> bool:
        """Moved no more than 2 mm (path length) over the last 0.5 s and slower than 5 cm/s now."""
        path = self.paths[name]
        if len(path) < _WINDOW + 1:
            return False
        pts = np.array(path)
        travel = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        return bool(travel <= L.STILL_PATH and self.speeds[name] < L.STILL_SPEED)

    def result(self, states, arms=None) -> Result:
        if not all_valid(states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        part = states["part"]
        have_arms = arms_valid(arms)
        tall = [n for n in L.PIECE_NAMES if too_tall(states[n], n)]
        part_free = released(arms, part, "part")
        r = against_rail(part) and part_free
        b = r and across_line(part)
        blockers: dict[str, str | None] = {"near": None, "far": None}
        fits = {}
        for end, sign in (("near", -1), ("far", 1)):
            rows = []  # every piece against this end: a blocking one first, then the one nearest to blocking it
            for n in L.PIECE_NAMES:
                f = end_fit(part, states[n], n, sign)
                ok = r and blocks_end(part, states[n], n, sign, arms)
                rows.append((not ok, abs(f["gap"]) + (0.0 if f["overlap"] > 0 else 1.0), n, f, ok))
            rows.sort(key=lambda t: (t[0], t[1]))
            fits[end] = (rows[0][2], rows[0][3])
            blockers[end] = rows[0][2] if rows[0][4] else None
        e_near, e_far = blockers["near"] is not None, blockers["far"] is not None
        if tall:
            score = 0.0
        else:
            score = 0.25 * r + 0.25 * b + 0.25 * e_near + 0.25 * e_far
        counted = ["part"] + [n for n in blockers.values() if n is not None]
        still = all(self.still(n) for n in counted)
        success = bool(score == 1.0 and still)
        gaps = rail_gaps(part)
        line_x, s_near, s_far = line_steel(part)
        k, tilt = down_axis(part)
        details = {
            "valid_state": True,
            "against_rail": r,
            "across_line": b,
            "near_end_blocked": e_near,
            "far_end_blocked": e_far,
            "near_blocker": blockers["near"],
            "far_blocker": blockers["far"],
            "too_tall_on_chuck": tall,
            "part_flat": part_flat(part),
            "part_released": part_free,
            "part_tilt_deg": round(tilt, 2),
            "part_down_axis": k,
            "part_bottom_mm": _mm(lowest(part, "part") - L.FACE_Z),
            "part_square_deg": round(part_square_deg(part), 2),
            "rail_gap_near_mm": _mm(gaps[0]),
            "rail_gap_far_mm": _mm(gaps[1]),
            "line_x": line_x,
            "line_steel_near_mm": _mm(s_near),
            "line_steel_far_mm": _mm(s_far),
            "part_ends_x": [round(x, 4) for x in part_ends_x(part)],
        }
        for end in ("near", "far"):
            n, f = fits[end]
            details[f"{end}_piece"] = n
            details[f"{end}_gap_mm"] = _mm(f["gap"])
            details[f"{end}_angle_deg"] = round(f["angle"], 2)
            details[f"{end}_overlap_mm"] = _mm(f["overlap"])
        details.update({
            "pieces": {n: {"flat_low": piece_flat_low(states[n], n), "on_chuck": piece_on_chuck(states[n], n),
                           "top_mm": _mm(highest(states[n], n) - L.FACE_Z)} for n in L.PIECE_NAMES},
            "zones_mm": ({a: {n: _mm(zone_gap(arms[a], states[n], n)) for n in L.LOOSE} for a in ("left", "right")}
                         if have_arms else None),
            "opening": ({a: round(float(arms[a].opening), 3) for a in ("left", "right")} if have_arms else None),
            "still": still,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        })
        return Result(score, success, details)
