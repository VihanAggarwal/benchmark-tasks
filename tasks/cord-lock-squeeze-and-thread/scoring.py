"""The scorer, as plain functions of the parts' states and the arms' state. NumPy only.

Every control step ``Tracker.update(states, arms)`` reads the cord's centre line (its two end points and its 16 segment
centres, from cablekit's CableState), the lock's pose and the plunger's travel (mechkit's AssemblyState), and both
arms (rigstate), all in the lock's frame (layout.py):

- the cord is THROUGH when its centre line crosses the lock's mid-plane (Y = 0) exactly once inside the scoring window,
  the body hole's outline (X -7.7 to 3.7 mm, Z 1.75 to 18.25 mm), which contains the 11.4 x 13 mm passage; a 7.9 mm
  cord fits through the passage only once;
- OUT, for each side of a through cord: the length along the cord from where it first reaches that side's face plane
  (Y = +-6.5 mm) to its end on that side (none: an end still inside the lock); the cord's out is the smaller side;
- IN: how far a cord end lying inside the lock (within the outline) is from the face it entered by.

RELEASED: both grippers' finger zones at least 30 mm from the lock and from the cord (an unreadable arm or a distance
that cannot be computed is not released). LET GO: released on every step of the last 1 s (20 steps).

PRESSED (latched; every tier needs it): at some step the plunger was at least 5 mm in while no cord end was 5 mm in
and the cord was not through. A straight cord end wedged into the 4.4 mm slot can push the plunger only 3.5 mm in (the
passage is then the cord's 7.9 mm), so the instruction's "press in the button" cannot be skipped by wedging.
PRESS HELD: from the first step a cord end is 5 mm in until the first step the cord is through, the plunger must be at
least 5 mm in on every step. If it drops below that in this window (press_lost, latched until a reset) the cord cannot
earn the through tiers (0.5, 0.75, 1.0): the score is at most the first stage, 0.25. A press released before the end
goes in, with the end then wedged in, is caught the same way.

Tiers, the highest line that holds at this step (each also needs PRESSED; the lines from 0.5 up also need that the
press was not lost):

    1.0  (success) let go, and on every step of that second, all at once: through with at least 40 mm out; the lock's
         lowest point within 2 mm of the table; the plunger at least 1 mm in from its free position; no lock corner
         or cord point moved more than 2 mm
    0.75 let go, and through with at least 40 mm out
    0.5  let go, and through with at least 10 mm out
    0.25 at some point in the trial a cord end was at least 5 mm in (latched): the only credit a held cord or lock
         can earn
    0    otherwise

There is no whole-trial zero. A step whose lock or cord state cannot be read scores the latched 0.25 (or 0) and
restarts the 1 s windows.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

LOCK = "cordlock"
CORD = "cord"
PLUNGER = "plunger"
WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 20 steps
NEG_INF = -math.inf

# Points on the lock whose motion is the lock's stillness: the body's eight corners (base frame) and the button's four
# end corners (on the plunger link, at rest).
BODY_CORNERS = np.array([(x, y, z) for x in (-0.017, 0.017) for y in (-L.HALF_DEPTH, L.HALF_DEPTH)
                         for z in (0.0, L.WIDTH)])
BUTTON_CORNERS = np.array([(0.024, y, z) for y in (-0.0045, 0.0045) for z in (0.002, 0.018)])


# -- reading the states (fail closed: None when a state is missing, malformed or not finite) --------------------------


def cord_line(cord) -> np.ndarray | None:
    """The cord's centre line, world frame: end 0, the segment centres, end 1 ((N + 2, 3)), or None."""
    try:
        pts = np.asarray(cord.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            return None
        half = float(cord.seg_len) / 2
        sd = np.asarray(cord.start_dir, dtype=float).reshape(3)
        ed = np.asarray(cord.end_dir, dtype=float).reshape(3)
    except (AttributeError, TypeError, ValueError):
        return None
    line = np.vstack([pts[0] + sd * half, pts, pts[-1] + ed * half])
    if not np.all(np.isfinite(line)) or not half > 0:
        return None
    return line


def lock_pose(lock) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(origin, rotation, plunger travel) of the lock's base frame, or None."""
    try:
        o = np.asarray(lock.origin, dtype=float).reshape(3)
        q = np.asarray(lock.quat_xyzw, dtype=float).reshape(4)
        travel = float(lock.joints[PLUNGER])
        lowest = float(np.asarray(lock.low, dtype=float).reshape(3)[2])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(q)) and math.isfinite(travel) and math.isfinite(lowest)):
        return None
    n = float(np.linalg.norm(q))
    if n < 0.5:
        return None
    x, y, z, w = q / n
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return o, r, travel


def to_local(points, origin, r) -> np.ndarray:
    return (np.asarray(points, dtype=float) - origin) @ r


# -- the cord against the lock (all in the lock's frame) ----------------------------------------------------------------


def in_outline(c) -> bool:
    return bool(L.FACE_HOLE_X[0] - L.EPS <= c[0] <= L.FACE_HOLE_X[1] + L.EPS
                and L.FACE_HOLE_Z[0] - L.EPS <= c[2] <= L.FACE_HOLE_Z[1] + L.EPS)


def crossings(line) -> list[tuple[int, np.ndarray, bool]]:
    """Every crossing of the mid-plane Y = 0 by the polyline: (segment index i, crossing point, inside the outline).
    A segment crosses when its ends are on different sides (Y <= 0 < Y' or Y' <= 0 < Y), so each crossing counts
    once."""
    out = []
    for i in range(len(line) - 1):
        p, q = line[i], line[i + 1]
        if (p[1] <= 0.0 < q[1]) or (q[1] <= 0.0 < p[1]):
            t = (0.0 - p[1]) / (q[1] - p[1])
            c = p + t * (q - p)
            out.append((i, c, in_outline(c)))
    return out


def arc_past(part, side: float) -> float:
    """``part``: points from the crossing outward to one cord end; ``side`` +1 or -1. The length along the cord from
    where it first reaches the face plane Y = side * 6.5 mm to the end; -inf when it never gets there."""
    length, past = 0.0, False
    for p, q in zip(part, part[1:]):
        seg = float(np.linalg.norm(q - p))
        if past:
            length += seg
        elif side * q[1] >= L.HALF_DEPTH - L.EPS:
            dy = side * q[1] - side * p[1]
            t = 0.0 if dy <= 0 else min(1.0, max(0.0, (L.HALF_DEPTH - side * p[1]) / dy))
            length += (1.0 - t) * seg
            past = True
    return length if past else NEG_INF


def through_out(line) -> float | None:
    """The cord's out (m, the smaller side) when it is through (exactly one crossing inside the outline), else None.
    A side whose end has not left the lock gives -inf."""
    inside = [(i, c) for i, c, ok in crossings(line) if ok]
    if len(inside) != 1:
        return None
    i, c = inside[0]
    a = [c, *line[:i + 1][::-1]]  # crossing -> end 0
    b = [c, *line[i + 1:]]  # crossing -> end 1
    sa = 1.0 if line[i][1] > 0 else -1.0
    sb = 1.0 if line[i + 1][1] > 0 else -1.0
    return min(arc_past(a, sa), arc_past(b, sb))


def depth_in(line) -> float:
    """The deeper of the two ends lying inside the lock (between the face planes, within the outline), measured from
    the face it entered by (the side its neighbouring segment centre is on); -inf when neither is."""
    best = NEG_INF
    for end, nb in ((line[0], line[1]), (line[-1], line[-2])):
        if abs(end[1]) <= L.HALF_DEPTH + L.EPS and in_outline(end):
            side = 1.0 if nb[1] >= 0 else -1.0
            best = max(best, L.HALF_DEPTH - side * float(end[1]))
    return best


def line_distance(line, radius: float = L.CORD_R):
    """A distance function (world points -> distance to the cord's surface, 0 inside) for rigstate's ``released``."""
    a = np.asarray(line[:-1], dtype=float)
    d = np.asarray(line[1:], dtype=float) - a
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-18)

    def dist(points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        rel = pts[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("mij,ij->mi", rel, d) / dd, 0.0, 1.0)
        near = a[None, :, :] + t[:, :, None] * d[None, :, :]
        return np.maximum(np.min(np.linalg.norm(pts[:, None, :] - near, axis=2), axis=1) - radius, 0.0)

    return dist


def lock_points(lock, origin, r) -> np.ndarray:
    """The lock's stillness points, world frame: body corners and the button's end corners."""
    body = BODY_CORNERS @ r.T + origin
    try:
        button = np.asarray(lock.link_to_world(PLUNGER, BUTTON_CORNERS), dtype=float)
    except Exception:  # noqa: BLE001 - no link pose: the base frame plus the travel along -X
        travel = float(lock.joints[PLUNGER])
        button = (BUTTON_CORNERS - np.array([travel, 0.0, 0.0])) @ r.T + origin
    return np.vstack([body, button])


# -- the tracker ---------------------------------------------------------------------------------------------------------


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return round(v * 1000, 2) if math.isfinite(v) else None


class Tracker:
    """The latched first stage, the release and end-state windows, and the tiers."""

    def __init__(self) -> None:
        self.resets = 0  # fallback resets seen (the lock teleported back to its start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.t25 = False  # a cord end 5 mm in at some point (latched)
        self.pressed = False  # the button 5 mm in before any cord end was 5 mm in (latched)
        self.armed = False  # a cord end has been 5 mm in (the press-held window is open or has been)
        self.closed = False  # the cord has been through (the window is over)
        self.press_lost = False  # the plunger was below 5 mm in during the window (latched)
        self.ever10 = self.ever40 = False  # through with 10 / 40 mm out at some point (trace only)
        self.hist: deque = deque(maxlen=WINDOW + 1)  # stillness points per valid step
        self.rel: deque = deque(maxlen=WINDOW)  # released from both parts, per step
        self.ends: deque = deque(maxlen=WINDOW)  # the end state per step
        self.prev_origin = None
        self.last = Result(0.0, False, {})

    def floor(self) -> float:
        """What a step earns without being let go: the latched first stage (after the press)."""
        return 0.25 if (self.t25 and self.pressed) else 0.0

    def _teleported(self, origin) -> bool:
        start = np.array([L.LOCK_ORIGIN[0], L.LOCK_ORIGIN[1], L.TOP])
        if self.prev_origin is None or float(np.linalg.norm(origin - start)) > L.RESET_START_TOL:
            return False
        return float(np.linalg.norm(origin - self.prev_origin)) > L.RESET_JUMP

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        lock, cord = states.get(LOCK), states.get(CORD)
        pose = None if lock is None else lock_pose(lock)
        line = None if cord is None else cord_line(cord)
        if pose is None or line is None:
            self.hist.clear()
            self.rel.clear()
            self.ends.clear()
            self.last = Result(self.floor(), False, {"valid_state": False, "tier": self.floor()})
            return self.last
        origin, r, travel = pose
        if self._teleported(origin):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev_origin = origin
        local = to_local(line, origin, r)
        depth = depth_in(local)
        out = through_out(local)
        through = out is not None
        out10 = bool(through and out >= L.OUT_HALF - L.EPS)
        out40 = bool(through and out >= L.OUT_FULL - L.EPS)
        cross = crossings(local)
        cord_in = bool(depth >= L.IN_MIN - L.EPS or any(ok for _, _, ok in cross))
        if travel >= L.PRESSED - L.EPS and not cord_in:
            self.pressed = True  # before the end goes in: a wedged end cannot do it
        if depth >= L.IN_MIN - L.EPS or out10:
            self.t25 = True
        if depth >= L.IN_MIN - L.EPS and not self.closed:
            self.armed = True
        if self.armed and not self.closed and not through and travel < L.PRESSED - L.EPS:
            self.press_lost = True  # the press let go while the end was going in and before it was through
        if through:
            self.closed = True
        held = bool(self.pressed and not self.press_lost)
        self.ever10 |= out10
        self.ever40 |= out40
        lowest = float(np.asarray(lock.low, dtype=float).reshape(3)[2]) - L.TOP
        on_table = lowest <= L.SUPPORT + L.EPS
        held_in = travel >= L.PLUNGER_HELD - L.EPS
        try:
            lock_dist = lock.distance
            rel_lock = rigstate.released(arms, lock_dist, L.LET_GO)
        except Exception:  # noqa: BLE001 - no distance: rigstate's rule, not released
            lock_dist, rel_lock = None, False
        cord_dist = line_distance(line)
        rel_cord = rigstate.released(arms, cord_dist, L.LET_GO)
        released = bool(rel_lock and rel_cord)
        end_ok = bool(out40 and on_table and held_in and released)
        self.hist.append(np.vstack([lock_points(lock, origin, r), line]))
        self.rel.append(released)
        self.ends.append(end_ok)
        let_go = len(self.rel) == WINDOW and all(self.rel)
        moved = None
        if len(self.hist) == WINDOW + 1:
            first = self.hist[0]
            moved = max(float(np.max(np.linalg.norm(h - first, axis=1))) for h in list(self.hist)[1:])
        still = moved is not None and moved <= L.STILL_MM + L.EPS
        success = bool(held and let_go and still and len(self.ends) == WINDOW and all(self.ends))
        if success:
            level = 1.0
        elif held and let_go and out40:
            level = 0.75
        elif held and let_go and out10:
            level = 0.5
        else:
            level = self.floor()
        score = level
        gaps = {}
        for name, fn in (("lock", lock_dist), ("cord", cord_dist)):
            try:
                gaps[name] = None if fn is None else rigstate.hands_gap(arms, fn)
            except Exception:  # noqa: BLE001 - trace only
                gaps[name] = None
        details = {
            "valid_state": True, "tier": level, "pressed": self.pressed, "press_lost": self.press_lost, "first_stage": self.t25,
            "ever_out10": self.ever10,
            "ever_out40": self.ever40, "through": through, "out_mm": _mm(out),
            "in_mm": _mm(depth), "crossings": [(i, bool(ok)) for i, _, ok in cross],
            "travel_mm": round(travel * 1000, 2), "lowest_mm": round(lowest * 1000, 2), "on_table": bool(on_table),
            "plunger_held": bool(held_in), "released_lock": bool(rel_lock), "released_cord": bool(rel_cord),
            "zone_lock_mm": _mm(gaps["lock"]), "zone_cord_mm": _mm(gaps["cord"]), "let_go_1s": bool(let_go),
            "end_state": end_ok,
            "end_state_1s": bool(len(self.ends) == WINDOW and all(self.ends)), "moved_1s_mm": _mm(moved),
            "still_1s": bool(still), "arms_readable": rigstate.readable(arms),
            "lock_origin": [round(float(v), 4) for v in origin],
            "end0_local_mm": [round(float(v) * 1000, 1) for v in local[0]],
            "end1_local_mm": [round(float(v) * 1000, 1) for v in local[-1]],
        }
        self.last = Result(score, success, details)
        return self.last
