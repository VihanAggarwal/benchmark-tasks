"""The scorer, as plain functions of the two floor wedges' states. NumPy only; no simulator needed.

The score is read from where the wedges stand, as a sound engineer checks a stage before the line check, with a tape
and a protractor centred on each mic stand (or a top photo): is each wedge standing base down on the deck; how far is
it from its singer's mic stand; how far round from the mic's front is it, against that mic's null (180 degrees for
the left singer's cardioid, 126 degrees either side for the right singer's supercardioid); and is its
grille aimed at the singer, at her stool. The scorer keeps only the last half second of each wedge's positions (to
tell a wedge at rest from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(states) -> Result(score, success, details)

``states`` maps ``wedge_left`` and ``wedge_right`` to what mechkit reports (``AssemblyState``): ``origin`` (its base
frame's origin, the middle of its footprint on its base plane) and ``quat_xyzw`` (that frame: local +x the way the
grille faces, local +z up), with the ObjectState fields. A state with a non-finite number, a zero quaternion or a
wedge without ``origin`` (a simulator blow-up, or a wedge not built as an assembly) scores 0 for that step and leaves
the tracker as it was.

The two wedges are identical: either may serve either singer. A wedge serves the singer whose mic stand is nearer to
it, and each singer scores from the best wedge serving her, so two wedges at one singer count once.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WEDGES = ("wedge_left", "wedge_right")
NAMES = WEDGES
SINGERS = tuple(L.SINGERS)  # ("left", "right")


def _box_corners(center, size) -> list[tuple[float, float, float]]:
    return [(center[0] + sx * size[0] / 2, center[1] + sy * size[1] / 2, center[2] + sz * size[2] / 2)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


_CORNERS = np.array([c for _, center, size, _ in L.WEDGE_SHAPES for c in _box_corners(center, size)])  # base frame
_FRONT = np.array([L.X_FRONT, 0.0, L.LOW_H / 2])  # the middle of the front face (under the grille)
_BACK = np.array([L.X_BACK, 0.0, L.HEIGHT / 2])  # the middle of the back face


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


def _finite(obj, keys) -> list | None:
    try:
        parts = [np.asarray(getattr(obj, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return None
    return parts


def valid_wedge(w) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    parts = _finite(w, ("origin", "quat_xyzw", "pos"))
    return bool(parts and parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid(states) -> bool:
    try:
        return all(valid_wedge(states[n]) for n in WEDGES)
    except (KeyError, TypeError):
        return False


# -- one wedge -------------------------------------------------------------------------------------------------------


def wedge_corners(w) -> np.ndarray:
    """The corners of the wedge's four boxes (cabinet slab, rear block, two grille plates) in the world, (32, 3)."""
    return np.asarray(w.origin, dtype=float) + _CORNERS @ rotation(w.quat_xyzw).T


def wedge_ends(w) -> tuple[np.ndarray, np.ndarray]:
    """(the middle of its front face, the middle of its back face) in the world: two points 45 mm apart."""
    r = rotation(w.quat_xyzw)
    o = np.asarray(w.origin, dtype=float)
    return o + r @ _FRONT, o + r @ _BACK


def tilt_deg(w) -> float:
    """How far the wedge's up axis is from vertical (0 base down, 90 on its back or side, 180 upside down)."""
    return float(np.degrees(math.acos(max(-1.0, min(1.0, float(rotation(w.quat_xyzw)[2, 2]))))))


def facing_rad(w) -> float | None:
    """The way the grille faces, seen from above (rad, 0 = +x, upstage); None when it faces straight up or down."""
    a = rotation(w.quat_xyzw)[:, 0]
    if math.hypot(a[0], a[1]) < 1e-6:
        return None
    return math.atan2(float(a[1]), float(a[0]))


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

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def footprint(w) -> np.ndarray:
    """The wedge's outline seen from above, (n, 2): the shadow of all its corners, turned and tilted as it is."""
    return _hull(wedge_corners(w)[:, :2])


def deck_margin(points_xy) -> float:
    """How far inside the deck's outline the outermost of the points is, seen from above (m; negative: past an edge)."""
    p = np.asarray(points_xy, dtype=float)
    return float(min((p[:, 0] - L.DECK_X0).min(), (L.DECK_X1 - p[:, 0]).min(),
                     (p[:, 1] - L.DECK_Y0).min(), (L.DECK_Y1 - p[:, 1]).min()))


def nearest_singer(w) -> str:
    """The singer whose mic stand's centre is nearest the middle of the wedge's footprint, seen from above."""
    o = np.asarray(w.origin, dtype=float)
    return min(SINGERS, key=lambda s: math.hypot(o[0] - L.SINGERS[s][0], o[1] - L.SINGERS[s][1]))


def placement(w, singer: str) -> tuple[float, float, float]:
    """(r, beta, aim) of the wedge for that singer: r the distance (m) from her mic stand's centre to the middle of
    the wedge's footprint; beta the angle (degrees, 0..180) at the stand between the mic's front axis (+x) and the line
    to the wedge; aim the angle (degrees, 0..180) between the way the grille faces and the line from the wedge to her
    stool's centre (180 when the grille faces straight up or down, or the wedge's middle is over the stool's centre)."""
    sx, sy, _, _ = L.SINGERS[singer]
    o = np.asarray(w.origin, dtype=float)
    dx, dy = o[0] - sx, o[1] - sy
    r = math.hypot(dx, dy)
    beta = 0.0 if r < 1e-9 else abs(math.degrees(math.remainder(math.atan2(dy, dx) - L.MIC_FRONT, 2 * math.pi)))
    px, py = L.STOOLS[singer]
    face = facing_rad(w)
    if face is None or math.hypot(px - o[0], py - o[1]) < 1e-3:
        return r, beta, 180.0
    aim = abs(math.degrees(math.remainder(math.atan2(py - o[1], px - o[0]) - face, 2 * math.pi)))
    return r, beta, aim


def null_off_deg(singer: str, beta: float) -> float:
    """How far round from that singer's mic null the wedge is (degrees). The supercardioid has a null either side,
    so beta (0..180, either side) is compared with 126."""
    return abs(beta - L.SINGERS[singer][3])


def credit(r: float, null_off: float, aim: float) -> float:
    """A singer's credit for a wedge standing on the deck: 0.5 at 80-150 mm from her stand, within 15 degrees of the
    null and aimed within 20 degrees of her stool; 0.25 at 60-200 mm, within 30 degrees and 45 degrees; else 0."""
    if L.R_FULL[0] <= r <= L.R_FULL[1] and null_off <= L.NULL_FULL_DEG and aim <= L.AIM_FULL_DEG:
        return L.SINGER_FULL
    if L.R_PART[0] <= r <= L.R_PART[1] and null_off <= L.NULL_PART_DEG and aim <= L.AIM_PART_DEG:
        return L.SINGER_PART
    return 0.0


def wedge_pose(w) -> dict:
    """What a tape, a protractor and a look from the side tell about one wedge (no rest test): whom it serves, its
    distance and angle from her stand, its aim, its tilt, its lowest point above the deck and how far inside the
    deck's outline it is. (A wedge that earns credit, 60 mm or more from its stand and at least 96 degrees round from
    the mic's front, cannot touch the stand's 30 mm base or the stool: its outline reaches at most 37.5 mm from its
    middle, which is then at least 60 mm from the stand's centre and 75 mm from the stool's.)"""
    c = wedge_corners(w)
    s = nearest_singer(w)
    r, beta, aim = placement(w, s)
    off = null_off_deg(s, beta)
    gap = float(c[:, 2].min() - L.DECK_TOP_W)
    tilt = tilt_deg(w)
    margin = deck_margin(c[:, :2])
    on_pose = bool(abs(gap) <= L.REST_GAP_TOL and tilt <= L.LEVEL_DEG and margin >= 0.0)
    return {"singer": s, "r_mm": round(r * 1000, 2), "beta_deg": round(beta, 2), "null_off_deg": round(off, 2),
            "aim_deg": round(aim, 2), "tilt_deg": round(tilt, 2), "gap_mm": round(gap * 1000, 2),
            "deck_margin_mm": round(margin * 1000, 2), "on_pose": on_pose,
            "fit": credit(r, off, aim)}


# -- the tiers -------------------------------------------------------------------------------------------------------


def evaluate(states, rest: dict) -> tuple[float, str, dict]:
    """(score, why it is not higher, diagnostics) of a valid state, given which wedges are at rest.

    A wedge is placed when its lowest point is within 3 mm of the deck's top, it is base down within 10 degrees, its
    whole outline is over the deck and it is at rest. It serves the singer
    whose stand is nearer; each singer scores from the best placed wedge serving her (0.5, 0.25 or 0). Score = left +
    right."""
    d: dict = {}
    for n in WEDGES:
        p = wedge_pose(states[n])
        p["at_rest"] = bool(rest.get(n, False))
        p["placed"] = bool(p["on_pose"] and p["at_rest"])
        p["credit"] = p["fit"] if p["placed"] else 0.0
        d[n] = p
    total, why = 0.0, []
    for s in SINGERS:
        mine = [n for n in WEDGES if d[n]["placed"] and d[n]["singer"] == s]
        best = max(mine, key=lambda n: d[n]["credit"]) if mine else None
        got = d[best]["credit"] if best else 0.0
        d[s] = {"wedges": mine, "best": best, "credit": got}
        total += got
        if not mine:
            why.append(f"{s}_no_wedge")
        elif got == 0.0:
            why.append(f"{s}_wedge_off_the_null_or_misaimed")
        elif got < L.SINGER_FULL:
            why.append(f"{s}_wedge_near_the_null")
    return round(total, 2), ",".join(why), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each wedge's two end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # every wedge's origin at the previous valid step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: both wedges standing in the wings on their starting marks, as they start."""
        for n, (x, y, heading) in L.WEDGES.items():
            s = states[n]
            o = np.asarray(s.origin, dtype=float)
            if math.hypot(o[0] - x, o[1] - y) > L.START_TOL:
                return False
            if abs(float(wedge_corners(s)[:, 2].min()) - L.TOP) > 0.003:
                return False
            a = rotation(s.quat_xyzw)[:, 0]
            if float(a[0] * math.cos(heading) + a[1] * math.sin(heading)) < math.cos(math.radians(5.0)):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest the wedge's two end points have been, over the last 0.5 s, from where they are now (m);
        None until 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        p, q = h[-1]
        return float(max(max(np.linalg.norm(a - p), np.linalg.norm(b - q)) for a, b in h))

    def update(self, states) -> Result:
        if not valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(states)
            return self.last
        now = {n: np.asarray(states[n].origin, dtype=float) for n in NAMES}
        if self.prev is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.history[n].append(wedge_ends(states[n]))
        self.prev = now
        self.last = self.result(states)
        return self.last

    def result(self, states) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(states):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        moved = {n: self.moved(n) for n in NAMES}
        rest = {n: (m is not None and m <= L.STILL_TOL) for n, m in moved.items()}
        score, reason, d = evaluate(states, rest)
        details = {"valid_state": True, "zero_reason": reason,
                   "moved_mm": {n: (None if m is None else round(m * 1000, 2)) for n, m in moved.items()}, **d}
        return Result(score, bool(score == 1.0), details)
