"""The scorer, as plain functions of the two pads' states and the arms' state. NumPy only; no simulator needed.

The rule is about where the pads end up: each of the two anterolateral positions on the manikin's chest earns
credit for a pad lying flat on the chest with its centre close to it, once both grippers are clear of that pad,
and the trial scores 0 if the two pads end up touching or nearly touching (closer than 10 mm, seen from above).
``task.success()`` feeds the scorer one observation per control step; it keeps only diagnostics about the trial
(steps, unreadable steps) and recognises a new episode.

    Tracker.update(pads, arms) -> Result(score, success, details)

``pads`` maps each pad's name (``layout.PADS``) to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; each pad is a box of
``layout.PAD_SIZE``, so its corners and outline follow from ``pos`` and ``quat_xyzw``. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then both pads are taken as released
for that step and the step is counted in ``arms_unavailable``. A pad state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The pads are identical plain blocks whose two large faces are alike, so either pad may go in either position, in
any heading, either face down: 'tilt' is measured from whichever large face is nearer to facing down.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a finger pad's outer corner (half the current finger gap
plus the 8 mm finger thickness, and half the finger's 27 mm width across; 57 mm fully open).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_HALF = np.array(L.PAD_HALF, dtype=float)
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
ARMS = ("left", "right")


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
        for side in ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- one pad ---------------------------------------------------------------------------------------------------
def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the pad's two large (110 x 70 mm) faces (degrees,
    0..90): the pad is the same either way up, so whichever face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def corners(pad) -> np.ndarray:
    """The pad's eight corners (world, 8 x 3)."""
    return np.asarray(pad.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(pad.quat_xyzw).T


def bottom_gap(pad) -> float:
    """Height of the pad's lowest point above the chest top (m; negative = below it)."""
    return float(corners(pad)[:, 2].min() - L.CHEST_TOP)


def over_chest(pad) -> bool:
    """The pad's centre, seen from above, is over the torso's top (x 0.150 .. 0.470, y -0.190 .. +0.190)."""
    x, y = float(pad.pos[0]), float(pad.pos[1])
    return bool(L.TORSO_X0 <= x <= L.TORSO_X1 and abs(y) <= L.TORSO_HALF_Y)


def on_chest(pad) -> bool:
    """'Lying on the chest': centre over the chest top, tilted at most 10 degrees, lowest point within 5 mm of the
    chest top (above or below)."""
    return bool(over_chest(pad) and tilt_deg(pad.quat_xyzw) <= L.TILT_MAX_DEG
                and abs(bottom_gap(pad)) <= L.ON_CHEST_TOL)


def target_distance(pad, target: str) -> float:
    """Horizontal distance from the pad's centre to a target (m)."""
    tx, ty = L.TARGETS[target]
    return float(math.hypot(float(pad.pos[0]) - tx, float(pad.pos[1]) - ty))


def credit(distance: float) -> float:
    """A target's credit for a pad lying on the chest this far from it: 0.5 within 25 mm, 0.25 within 40 mm."""
    if distance <= L.NEAR:
        return 0.5
    if distance <= L.FAR:
        return 0.25
    return 0.0


# -- outlines seen from above ----------------------------------------------------------------------------------
def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1])


def footprint(pad) -> np.ndarray:
    """The pad's real outline seen from above, (n, 2): the shadow of its eight corners, turned and tilted as it is
    (for a flat pad, its 110 x 70 mm face)."""
    return _hull(corners(pad)[:, :2])


def _overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Do two convex polygons overlap (or touch)? Separating axes."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            if math.hypot(*e) < 1e-12:
                continue
            a = np.array([-e[1], e[0]])
            pa, qa = p @ a, q @ a
            if pa.max() < qa.min() or qa.max() < pa.min():
                return False
    return True


def _point_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    n = float(ab @ ab)
    s = 0.0 if n < 1e-18 else float(np.clip((pt - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + s * ab)))


def outline_gap(p: np.ndarray, q: np.ndarray) -> float:
    """Shortest distance between two convex outlines seen from above (m); 0 when they touch or overlap."""
    if _overlap(p, q):
        return 0.0
    best = math.inf
    for poly, other in ((p, q), (q, p)):
        for pt in poly:
            for i in range(len(other)):
                best = min(best, _point_segment(pt, other[i], other[(i + 1) % len(other)]))
    return float(best)


def pads_gap(pads: dict) -> float:
    """How far apart the two pads' outlines are, seen from above (m; 0 touching, overlapping or one on the other)."""
    a, b = (pads[n] for n in L.PADS)
    return outline_gap(footprint(a), footprint(b))


# -- the arms ---------------------------------------------------------------------------------------------------
def point_pad_distance(p, pad) -> float:
    """Distance from a point to the pad (0 inside it), in the pad's own frame."""
    r = rotation(pad.quat_xyzw)
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(pad.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - _HALF, 0.0)))


def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the finger's thickness, and half
    its width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def zone_gap(arm, pad) -> float:
    """How far a gripper's finger zone is from a pad (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_pad_distance(p, pad) for p in pts) - r


def released(arms, pad) -> bool:
    """Both grippers' finger zones at least 10 mm from the pad: neither can be holding, pressing or touching it."""
    return bool(all(zone_gap(arms[s], pad) >= L.RELEASE_CLEAR for s in ARMS))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The trial as it goes: the score is the end state's, the tracker keeps diagnostics and spots resets."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite pad state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.first_full_step: int | None = None  # the first step that scored 1.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {pad: (centre, velocity)} at the previous valid control step

    @staticmethod
    def at_start(pads: dict) -> bool:
        """The starting layout: both pads flat on the table near their starting marks, still."""
        for name in L.PADS:
            s = pads[name]
            x, y, _ = L.PAD_START[name]
            near = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            low = float(corners(s)[:, 2].min()) - L.TOP
            if not (near and abs(low) < 0.003 and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, pads: dict) -> bool:
        """A pad moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could have
        carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        for name in L.PADS:
            p0, v0 = self.prev[name]
            c = np.asarray(pads[name].pos, dtype=float)
            if float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall:
                return True
        return False

    def update(self, pads: dict, arms=None) -> Result:
        if not all(valid(pads.get(n)) for n in L.PADS):  # a blow-up: score 0 for this step, keep the rest
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(pads, arms)
            return self.last
        if self.prev is not None and self._teleported(pads) and self.at_start(pads):
            self.reset()  # the simulator put the pads back: a new episode (rigstate's on_reset hook normally does
            #               this first; this is the fallback)
        self.prev = {n: (np.array(pads[n].pos, dtype=float), np.array(pads[n].lin_vel, dtype=float))
                     for n in L.PADS}
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(pads, arms)
        if self.last.score == 1.0 and self.first_full_step is None:
            self.first_full_step = self.steps
        return self.last

    def result(self, pads: dict, arms=None) -> Result:
        if not all(valid(pads.get(n)) for n in L.PADS):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        have_arms = arms_valid(arms)
        gap = pads_gap(pads)
        details: dict = {"valid_state": True}
        credits = {t: 0.0 for t in L.TARGETS}
        for name in L.PADS:
            s = pads[name]
            lying = on_chest(s)
            free = released(arms, s) if have_arms else True  # unreadable: taken as released, counted
            counts = lying and free
            dist = {t: target_distance(s, t) for t in L.TARGETS}
            if counts:
                for t in L.TARGETS:
                    credits[t] = max(credits[t], credit(dist[t]))
            details.update({
                f"{name}_pos": [round(float(v), 4) for v in s.pos],
                f"{name}_to_sternal_mm": _mm(dist["sternal"]),
                f"{name}_to_apical_mm": _mm(dist["apical"]),
                f"{name}_on_chest": lying,
                f"{name}_tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                f"{name}_bottom_gap_mm": _mm(bottom_gap(s)),
                f"{name}_released": free if have_arms else None,
                f"{name}_left_zone_mm": _mm(zone_gap(arms["left"], s)) if have_arms else None,
                f"{name}_right_zone_mm": _mm(zone_gap(arms["right"], s)) if have_arms else None,
                f"{name}_counts": counts,
                f"{name}_resting": is_resting(s),
            })
        touching = gap < L.PAD_GAP_MIN
        score = 0.0 if touching else float(sum(credits.values()))
        success = bool(score == 1.0 and all(is_resting(pads[n]) for n in L.PADS))
        details.update({
            "pads_gap_mm": _mm(gap),
            "pads_touching": touching,
            "sternal_credit": credits["sternal"],
            "apical_credit": credits["apical"],
            "first_full_step": self.first_full_step,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        })
        return Result(score, success, details)
