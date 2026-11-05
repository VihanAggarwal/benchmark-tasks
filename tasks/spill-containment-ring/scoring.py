"""The scorer, as plain functions of the four socks' states and the arms' state. NumPy only; no simulator needed.

Part of the rule is about the whole trial, not only its end: no sock and no gripper's fingers may ever come over
the spill or within 5 mm of its edge (seen from above, at any height). So the scorer keeps a little state (which
of those has happened, and when) and is fed one observation per control step by ``task.success()``.

    Tracker.update(socks, arms) -> Result(score, success, details)

``socks`` maps each sock's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``
and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the gripper rule is taken as kept
for that step and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Shapes seen from above: a sock's 'outline' is the outline of its real (turned, perhaps tilted) box seen from above,
the convex hull of its eight corners; distances between socks, and from the spill's centre to a sock, use these
outlines, not bounding boxes (a sock carried at 45 degrees has a bounding box far larger than itself).

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a finger's outer corner (half the current finger gap plus the fingers'
10 mm thickness, and half their 40 mm width across: their size 19-38 mm above the tips on yam_arena's collision
meshes, a little more than at the tips). That holds the lowest 38 mm of both fingers whatever the gripper's turn.
The gripper's body and the rest of the arm, higher up, are not judged.

The ring: the placed socks close round the spill if all of them can be taken in a cycle, each within the joint limit
of the next, such that the loop through them (each sock's centre, its nearest point to the next sock, that sock's
nearest point, its centre, ...) goes round the spill's centre. Every segment of that loop lies inside a sock or
crosses a joint, so any way out from the spill crosses a sock or a joint no wider than the limit. The 0.5 tier's
'group' is a connected one: socks linked one to another by gaps of 25 mm or less.

The socks are plain bars with a square section, so any of their four long faces may be down: 'tilt' is measured
from whichever of the two cross axes is nearer to vertical (which also catches a sock tilted along its length).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
CENTRE = np.array([L.SPILL_X, L.SPILL_Y])
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_HALF = np.array(L.SOCK_SIZE, dtype=float) / 2


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


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and whichever of the sock's two cross axes is nearer to it (degrees,
    0..90). A square bar lies the same on any of its four long faces; standing on end it is 90."""
    r = rotation(quat_xyzw)
    up = max(abs(r[2, 1]), abs(r[2, 2]))
    return float(np.degrees(np.arccos(np.clip(up, 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- plan-view geometry ---------------------------------------------------------------------------------------
def corners(s) -> np.ndarray:
    """The eight corners of a sock (world, 8 x 3) from its centre and orientation."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(s.quat_xyzw).T


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points) -> np.ndarray:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain); duplicates and collinear points go."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def outline(s) -> np.ndarray:
    """The sock's outline seen from above: the convex hull of its corners (k x 2, counter-clockwise)."""
    return convex_hull(corners(s)[:, :2])


def _closest_on_segment(p, a, b) -> np.ndarray:
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-18 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return a + t * ab


def inside(poly, p) -> bool:
    """Is the point inside (or on) a counter-clockwise convex polygon?"""
    k = len(poly)
    if k < 3:
        return False
    return all(_cross(poly[i], poly[(i + 1) % k], p) >= -1e-12 for i in range(k))


def point_outline_distance(poly, p) -> tuple[float, np.ndarray]:
    """Distance from a point to a convex outline (0 inside it) and the outline's nearest point."""
    p = np.asarray(p, dtype=float)
    if inside(poly, p):
        return 0.0, p.copy()
    best = (math.inf, p)
    k = len(poly)
    for i in range(k):
        q = _closest_on_segment(p, poly[i], poly[(i + 1) % k])
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            best = (d, q)
    return best


def _crossing(a, b, c, d) -> np.ndarray | None:
    """Where segments ab and cd properly cross, or None."""
    d1, d2 = _cross(c, d, a), _cross(c, d, b)
    d3, d4 = _cross(a, b, c), _cross(a, b, d)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return a + (b - a) * (d1 / (d1 - d2))
    return None


def outline_gap(pa, pb) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance between two convex outlines (0 when they touch or overlap) and a nearest point on each."""
    for v in pa:
        if inside(pb, v):
            return 0.0, v.copy(), v.copy()
    for v in pb:
        if inside(pa, v):
            return 0.0, v.copy(), v.copy()
    ka, kb = len(pa), len(pb)
    for i in range(ka):
        for j in range(kb):
            x = _crossing(pa[i], pa[(i + 1) % ka], pb[j], pb[(j + 1) % kb])
            if x is not None:  # on both outlines
                return 0.0, x, x.copy()
    best = (math.inf, pa[0], pb[0])
    for v in pa:  # for two disjoint convex polygons the nearest pair includes a vertex of one of them
        d, q = point_outline_distance(pb, v)
        if d < best[0]:
            best = (d, v, q)
    for v in pb:
        d, q = point_outline_distance(pa, v)
        if d < best[0]:
            best = (d, q, v)
    return float(best[0]), np.asarray(best[1], float), np.asarray(best[2], float)


def spill_clearance(s) -> float:
    """From the spill's centre to the sock's outline, seen from above (m); under ZONE_R (65 mm) it has entered."""
    return point_outline_distance(outline(s), CENTRE)[0]


def angular_span(poly) -> tuple[float, float]:
    """The directions, seen from the spill's centre, that cross the outline: (start, width) in radians."""
    rel = poly - CENTRE
    mid = math.atan2(float(rel[:, 1].mean()), float(rel[:, 0].mean()))
    angs = [math.remainder(math.atan2(float(y), float(x)) - mid, 2 * math.pi) for x, y in rel]
    lo, hi = min(angs), max(angs)
    return (mid + lo) % (2 * math.pi), hi - lo


def covered_deg(spans) -> float:
    """Total angle (degrees) covered by the union of a set of (start, width) direction intervals."""
    pieces = []
    for start, width in spans:
        width = min(width, 2 * math.pi)
        end = start + width
        if end > 2 * math.pi:
            pieces += [(start, 2 * math.pi), (0.0, end - 2 * math.pi)]
        else:
            pieces.append((start, end))
    pieces.sort()
    total, cur = 0.0, None
    for a, b in pieces:
        if cur is None or a > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur is not None:
        total += cur[1] - cur[0]
    return math.degrees(total)


def winding(loop) -> float:
    """How many times a closed polygon goes round the spill's centre (about +-1 round it, 0 not)."""
    total = 0.0
    rel = [np.asarray(p, float) - CENTRE for p in loop]
    for a, b in zip(rel, rel[1:] + rel[:1]):
        total += math.remainder(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]), 2 * math.pi)
    return total / (2 * math.pi)


# -- the socks ------------------------------------------------------------------------------------------------
def on_table(s) -> bool:
    """The sock's lowest point within 3 mm of the table top (not on a tray, not on another sock)."""
    return bool(abs(float(s.low[2]) - L.TOP) <= L.ON_TABLE_TOL)


def placed(s) -> bool:
    """Flat on the table (a long face down within 5 degrees, lowest point within 3 mm of the table top) with its
    outline within 60 mm of the spill's edge (120 mm of its centre)."""
    return bool(tilt_deg(s.quat_xyzw) < L.FLAT_DEG and on_table(s) and spill_clearance(s) <= L.PLACED_R)


@dataclass
class Ring:
    tight: bool = False  # all joints <= 10 mm round the spill
    one_loose: bool = False  # round the spill, joints <= 25 mm, at most one over 10 mm
    joints_mm: list | None = None  # the joints of the best cycle found round the spill
    group: int = 0  # the largest connected group of placed socks, linked one to another by gaps of 25 mm or less
    span_deg: float = 0.0  # the largest angle round the spill a group of 3 or more covers


def analyse_ring(outlines: dict) -> Ring:
    """``outlines``: name -> outline of every placed sock. A ring goes through every one of them (with all four
    placed, all four are in it: three socks closing a ring with a fourth lying elsewhere do not count)."""
    names = sorted(outlines)
    ring = Ring()
    if not names:
        return ring
    gap = {}
    for a, b in itertools.combinations(names, 2):
        d, pa, pb = outline_gap(outlines[a], outlines[b])
        gap[(a, b)] = (d, pa, pb)
        gap[(b, a)] = (d, pb, pa)
    centre = {n: outlines[n].mean(axis=0) for n in names}
    best = None
    k = len(names)
    if k >= 3:
        first, rest = names[0], names[1:]
        for perm in itertools.permutations(rest):
            cyc = (first, *perm)
            joints = [gap[(cyc[i], cyc[(i + 1) % k])][0] for i in range(k)]
            if max(joints) > L.GAP_LOOSE:
                continue
            loop = []
            for i in range(k):
                a, b = cyc[i], cyc[(i + 1) % k]
                _, pa, pb = gap[(a, b)]
                loop += [centre[a], pa, pb]
            if abs(winding(loop)) < 0.5:
                continue
            wide = sum(j > L.GAP_TIGHT for j in joints)
            key = (wide, max(joints))
            if best is None or key < best[0]:
                best = (key, joints)
    if best is not None:
        ring.tight = best[0][0] == 0
        ring.one_loose = best[0][0] <= 1
        ring.joints_mm = [round(j * 1000, 1) for j in best[1]]
    # groups: connected sets of placed socks, neighbours within GAP_LOOSE
    left = set(names)
    while left:
        stack, grp = [left.pop()], set()
        while stack:
            n = stack.pop()
            grp.add(n)
            for m in list(left):
                if gap[(n, m)][0] <= L.GAP_LOOSE:
                    left.discard(m)
                    stack.append(m)
        ring.group = max(ring.group, len(grp))
        if len(grp) >= 3:
            ring.span_deg = max(ring.span_deg, covered_deg([angular_span(outlines[n]) for n in grp]))
    return ring


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def finger_clearance(arm) -> float:
    """From the spill's centre to the gripper's finger zone, seen from above (m); under ZONE_R it has entered."""
    pts, r = finger_zone(arm)
    return float(np.min(np.hypot(pts[:, 0] - CENTRE[0], pts[:, 1] - CENTRE[1]))) - r


def point_sock_distance(p, s) -> float:
    """Distance from a point to a sock's real (turned) box (0 inside it)."""
    local = rotation(s.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(local) - _HALF, 0.0)))


def zone_gap(arm, s) -> float:
    """How far a gripper's finger zone is from a sock (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_sock_distance(p, s) for p in pts) - r


def let_go(arm, socks) -> bool:
    """The gripper holds no sock: it is open (opening 0.9 or more), or its finger zone is more than 10 mm from
    every sock."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or all(zone_gap(arm, s) > L.LET_GO_GAP for s in socks))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def breach_text(key: str) -> str:
    if key.endswith("_fingers_entered"):
        return f"the {key.split('_')[0]} gripper's fingers came over the spill or within 5 mm of its edge"
    return f"{key[:-len('_entered')]} came over the spill or within 5 mm of its edge"


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.min_sock_clear = math.inf  # m: closest any sock's outline came to the spill's centre (65 mm: entered)
        self.min_finger_clear = {"left": math.inf, "right": math.inf}  # m: the same for each finger zone
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of each sock] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(socks: dict) -> bool:
        """The starting layout: every sock flat on its tray within START_TOL of its starting place, all still."""
        for name, (x, y) in L.SOCK_START.items():
            s = socks.get(name)
            if s is None or not valid(s):
                return False
            if not (np.hypot(s.pos[0] - x, s.pos[1] - y) < L.START_TOL and abs(float(s.low[2]) - L.TRAY_TOP_Z) < 0.003
                    and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some sock moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    @staticmethod
    def _ordered(socks: dict) -> list:
        return [socks.get(n) for n in L.SOCKS]

    def update(self, socks: dict, arms=None) -> Result:
        states = self._ordered(socks)
        if not all(valid(s) for s in states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(socks, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in states]
        if self.prev is not None and self._teleported(now) and self.at_start(socks):
            self.reset()  # the simulator put every sock back on its tray: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        for name, s in zip(L.SOCKS, states):
            d = spill_clearance(s)
            self.min_sock_clear = min(self.min_sock_clear, d)
            if d < L.ZONE_R:
                self._flag(f"{name}_entered")
        if arms_valid(arms):
            for side in ("left", "right"):
                d = finger_clearance(arms[side])
                self.min_finger_clear[side] = min(self.min_finger_clear[side], d)
                if d < L.ZONE_R:
                    self._flag(f"{side}_fingers_entered")
        else:
            self.arms_unavailable += 1
        self.last = self.result(socks, arms)
        return self.last

    def result(self, socks: dict, arms=None) -> Result:
        states = self._ordered(socks)
        if not all(valid(s) for s in states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        released = (all(let_go(arms[side], states) for side in ("left", "right")) if have_arms
                    else True)  # unreadable: taken as met, counted
        outlines = {n: outline(s) for n, s in zip(L.SOCKS, states)}
        clear = {n: point_outline_distance(outlines[n], CENTRE)[0] for n in L.SOCKS}
        is_placed = {n: bool(tilt_deg(s.quat_xyzw) < L.FLAT_DEG and on_table(s) and clear[n] <= L.PLACED_R)
                     for n, s in zip(L.SOCKS, states)}
        ring = analyse_ring({n: outlines[n] for n in L.SOCKS if is_placed[n]})
        n_placed = sum(is_placed.values())
        all_placed = n_placed == len(L.SOCKS)
        if self.breach:
            score = 0.0
        elif all_placed and ring.tight and released:
            score = 1.0
        elif all_placed and ring.one_loose:
            score = 0.75
        elif ring.group >= 3 and ring.span_deg >= L.CHAIN_SPAN_DEG:
            score = 0.5
        elif n_placed >= 2:
            score = 0.25
        else:
            score = 0.0
        resting = {n: is_resting(s) for n, s in zip(L.SOCKS, states)}
        success = bool(score == 1.0 and all(resting.values()))
        fingers = {side: finger_clearance(arms[side]) for side in ("left", "right")} if have_arms else None
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "placed": sorted(n for n, v in is_placed.items() if v),
            "n_placed": n_placed,
            "ring_closed": ring.tight,
            "ring_one_loose_joint": ring.one_loose and not ring.tight,
            "ring_joints_mm": ring.joints_mm,
            "group_size": ring.group,
            "group_span_deg": round(ring.span_deg, 1),
            "sock_clear_mm": {n: _mm(v) for n, v in clear.items()},
            "sock_tilt_deg": {n: round(tilt_deg(s.quat_xyzw), 2) for n, s in zip(L.SOCKS, states)},
            "sock_low_mm": {n: _mm(float(s.low[2]) - L.TOP) for n, s in zip(L.SOCKS, states)},
            "min_sock_clear_mm": _mm(self.min_sock_clear),
            "finger_clear_mm": {k: _mm(v) for k, v in fingers.items()} if fingers else None,
            "min_finger_clear_mm": {k: _mm(v) for k, v in self.min_finger_clear.items()},
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "resting": sorted(n for n, v in resting.items() if v),
        }
        return Result(score, success, details)
