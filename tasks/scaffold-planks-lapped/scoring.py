"""The scorer, as plain functions of the two boards' states. NumPy only; no simulator needed.

The rule is about the finished platform, read from the state at the last control step: each bay planked by a board
whose two ends both reach 19 to 38 mm past the centre lines of the putlogs under them (6 to 12 in at 1:8), the two
boards in one straight run, lapped over the middle putlog. A board end more than 38 mm past its putlog is a trap
end and scores 0. Either board may go in either bay and either may be the upper one. The putlogs are fixed, so
their tops come from ``layout``. A small tracker is fed once per control step by ``task.success()``: it keeps the
boards' end points over the last 0.5 s (a board is 'still' when neither end moved more than 2 mm in that time;
a gripped board's reported velocity chatters, so its path is used, not its velocity) and recognises a new episode
when the simulator puts both boards back on their starting marks in one step.

    Tracker.update(boards) -> Result(score, success, details)

``boards`` is a dict {name: state} of the two boards, each anything with ``pos`` (bounding-box centre, which for a
box is its centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
``yam_arena.task_file.ObjectState`` has. The boards are plain boxes (``layout.BOARD_SIZE``, long side along the
body's x axis), so their faces follow from ``pos`` and ``quat_xyzw``; either broad face may be down. A state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it
was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

HALF = np.array(L.BOARD_SIZE, dtype=float) / 2
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the
    world. (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(s) -> float:
    """Angle between the world's vertical and the axis through the board's two broad faces (degrees, 0..90): a
    plain board is the same either way up. Lying flat 0; the lapped board about 5.4; on an edge or end 90."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def long_axis(s) -> np.ndarray:
    """Unit vector along the board's length (world)."""
    return rotation(s.quat_xyzw)[:, 0]


def heading_err_deg(s) -> float:
    """How far the board's long side, seen from above, is turned from the run (world y), degrees 0..90 (either
    way round). A board standing on its end has no heading: 90."""
    u = long_axis(s)
    if math.hypot(float(u[0]), float(u[1])) < 1e-9:
        return 90.0
    ang = math.degrees(math.atan2(float(u[1]), float(u[0]))) % 180.0  # 0 = along x
    ang = (ang - 90.0) % 180.0  # 0 = along y
    return float(min(ang, 180.0 - ang))


def ends(s) -> tuple[np.ndarray, np.ndarray]:
    """The centres of the board's two end faces (world)."""
    c, u = np.asarray(s.pos, dtype=float), long_axis(s)
    return c - u * HALF[0], c + u * HALF[0]


def plan_corners(s) -> np.ndarray:
    """The board's eight corners (world), (8, 3)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    return np.array([c + r @ (HALF * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])


# -- faces and 'lies on' ------------------------------------------------------------------------------------------
class Face:
    """A flat rectangular face: its centre, its normal (pointing up), its four corners (world)."""

    def __init__(self, centre, normal, corners) -> None:
        self.c = np.asarray(centre, dtype=float)
        self.n = np.asarray(normal, dtype=float)
        self.corners = np.asarray(corners, dtype=float)

    def z_at(self, xy) -> float:
        """Height of the face's plane above the point ``xy``."""
        n, c = self.n, self.c
        if abs(n[2]) < 1e-6:
            return float(c[2])
        return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])

    def outline(self) -> np.ndarray:
        """The face's outline seen from above, counter-clockwise, (4, 2)."""
        return _ccw(self.corners[:, :2])


def broad_face(s, which: str) -> Face:
    """The board's upper ('top') or lower ('bottom') broad face, whichever of the two faces up / down."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    up = r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]
    centre = c + up * HALF[2] * (1.0 if which == "top" else -1.0)
    corners = [centre + r[:, 0] * i * HALF[0] + r[:, 1] * j * HALF[1] for i, j in _SIGNS]
    return Face(centre, up, corners)


def putlog_top(name: str) -> Face:
    """The top face of a fixed putlog."""
    y, a, b = L.PUTLOG_Y[name], L.PUTLOG_SIZE[0] / 2, L.PUTLOG_HALF_W
    return Face((L.PUTLOG_X, y, L.PUTLOG_TOP), (0.0, 0.0, 1.0),
                [(L.PUTLOG_X + i * a, y + j * b, L.PUTLOG_TOP) for i, j in _SIGNS])


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex polygon ``clip`` (both CCW)."""
    out = [np.asarray(v, dtype=float) for v in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            return None
        prev = inp[-1]
        for cur in inp:
            sp, sc = side(prev), side(cur)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
            prev = cur
    return np.array(out) if len(out) >= 3 else None


def contact(lower: Face, upper: Face) -> tuple[float, float | None, np.ndarray | None]:
    """Where ``upper``'s underside overlaps ``lower``'s top, seen from above: (the overlap's area, the smallest
    height of ``upper``'s underside above ``lower``'s top over the overlap (both are planes, so it is at a corner
    of the overlap; negative = below), the overlap polygon). (0, None, None) if they do not overlap."""
    part = _clip(upper.outline(), lower.outline())
    if part is None:
        return 0.0, None, None
    area = _area(part)
    if not np.isfinite(area) or area <= 1e-12:
        return 0.0, None, None
    gap = min(upper.z_at(v) - lower.z_at(v) for v in part)
    return float(area), float(gap), part


def _touches(area: float, gap: float | None) -> bool:
    return bool(gap is not None and area >= L.REST_MIN_AREA and abs(gap) <= L.REST_TOL)


def on_putlog_directly(s, name: str) -> bool:
    """The board's underside overlaps the putlog's top (seen from above) and comes within 2 mm of it there."""
    area, gap, _ = contact(putlog_top(name), broad_face(s, "bottom"))
    return _touches(area, gap)


def on_board(upper, lower) -> tuple[bool, np.ndarray | None]:
    """``upper`` rests on ``lower``: its underside comes within 2 mm of ``lower``'s top where they overlap (seen
    from above). Returns (rests, the overlap polygon)."""
    area, gap, part = contact(broad_face(lower, "top"), broad_face(upper, "bottom"))
    return _touches(area, gap), part


def supports(boards: dict) -> dict[str, dict[str, str]]:
    """For each board, the putlogs it lies on and how: 'direct' (on the putlog's top) or 'through' (resting on the
    other board where that board lies directly on the putlog and the putlog's centre line passes under the place
    where the two overlap)."""
    names = list(boards)
    direct = {n: {p for p, _ in L.PUTLOGS if on_putlog_directly(boards[n], p)} for n in names}
    out: dict[str, dict[str, str]] = {}
    for n in names:
        found = {p: "direct" for p in direct[n]}
        for m in names:
            if m == n:
                continue
            rests, part = on_board(boards[n], boards[m])
            if not rests or part is None:
                continue
            y0, y1 = float(part[:, 1].min()), float(part[:, 1].max())
            for p in direct[m]:
                if p not in found and y0 <= L.PUTLOG_Y[p] <= y1:
                    found[p] = "through"
        out[n] = found
    return out


def crossing(s, y: float) -> float | None:
    """Where the board's centre line crosses the line y = ``y`` (seen from above), as the distance along the board
    from its centre (signed along its long axis); None if the board lies parallel to that line."""
    u = long_axis(s)
    if abs(float(u[1])) < 1e-6:
        return None
    return float((y - float(s.pos[1])) / float(u[1]))


def overhangs(s, putlogs) -> dict[str, float] | None:
    """The overhang of each end of the board (m): the distance along the board from where its centre line crosses
    the centre line of the outermost putlog it lies on toward that end, to that end face. Keyed by which end ('low'
    for the end at the smaller y, 'high' for the other). None if it lies on no putlog it crosses."""
    params = [t for t in (crossing(s, L.PUTLOG_Y[p]) for p in putlogs) if t is not None]
    if not params:
        return None
    plus, minus = HALF[0] - max(params), HALF[0] + min(params)  # the ends at +L/2 and -L/2 along the long axis
    plus_is_high = float(long_axis(s)[1]) >= 0.0
    return {"high": plus, "low": minus} if plus_is_high else {"high": minus, "low": plus}


def lowest(s) -> float:
    """Height of the board's lowest point above the table top (m)."""
    return float(plan_corners(s)[:, 2].min() - L.TOP)


def within_platform(s) -> bool:
    """The board's outline, seen from above, lies within the putlogs' length (x 0.200 to 0.320)."""
    xs = plan_corners(s)[:, 0]
    return bool(xs.min() >= L.PUTLOG_X0 - 1e-9 and xs.max() <= L.PUTLOG_X1 + 1e-9)


def x_at_middle(s) -> float | None:
    """Where the board's centre line, seen from above, crosses the middle putlog's centre line (its x)."""
    t = crossing(s, L.PUTLOG_Y[L.MIDDLE])
    if t is None:
        return None
    return float(s.pos[0] + t * long_axis(s)[0])


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(boards: dict, still: dict[str, bool]) -> tuple[float, dict]:
    """The score of the platform as it stands (``still``: board name -> neither end moved more than 2 mm in the
    last 0.5 s), and what was found (JSON-clean)."""
    sup = supports(boards)
    per: dict[str, dict] = {}
    trap: list[str] = []
    bears: dict[str, list[str]] = {bay: [] for bay in L.BAYS}
    for n, s in boards.items():
        lies = sup[n]
        head, tilt, low = heading_err_deg(s), tilt_deg(s), lowest(s)
        scaffold = bool(lies) and head <= L.MEASURE_HEADING_DEG and low >= L.CLEAR_OF_TABLE
        oh = overhangs(s, lies) if scaffold else None
        info = {"lies_on": dict(sorted(lies.items())), "on_scaffold": scaffold,
                "overhang_mm": None if oh is None else {k: _mm(v) for k, v in oh.items()},
                "heading_err_deg": round(head, 2), "tilt_deg": round(tilt, 2), "lowest_mm": _mm(low),
                "within_platform": within_platform(s), "still": bool(still.get(n, False)), "bears_in": None}
        if oh is not None and max(oh.values()) > L.OVERHANG_MAX + L.LIMIT_EPS:
            trap.append(n)
        if scaffold and oh is not None:
            ok = (head <= L.HEADING_DEG and tilt <= L.TILT_MAX_DEG and info["within_platform"] and info["still"]
                  and all(L.OVERHANG_MIN - L.LIMIT_EPS <= v <= L.OVERHANG_MAX + L.LIMIT_EPS for v in oh.values()))
            for bay, pair in L.BAYS.items():
                if ok and all(p in lies for p in pair):
                    bears[bay].append(n)
                    info["bears_in"] = bay
        per[n] = info

    d: dict = {"boards": per, "trap_end": trap, "bays_planked": [b for b in L.BAYS if bears[b]],
               "in_line_mm": None, "lap_mm": None, "lapped_in_line": False, "upper": None, "zero_reason": ""}
    if trap:
        d["zero_reason"] = "trap_end"
        return 0.0, d

    left, right = bears["left"], bears["right"]
    pair = next(((a, b) for a in left for b in right if a != b), None)
    if pair is not None:
        a, b = pair  # a planks the left bay, b the right
        xa, xb = x_at_middle(boards[a]), x_at_middle(boards[b])
        off = None if xa is None or xb is None else abs(xa - xb)
        d["in_line_mm"] = _mm(off)
        ohs_a, ohs_b = overhangs(boards[a], sup[a]), overhangs(boards[b], sup[b])
        lap = ohs_a["low"] + ohs_b["high"]  # each board's inner end past the middle putlog's centre line
        d["lap_mm"] = _mm(lap)
        upper = next((n for n in (a, b) if sup[n].get(L.MIDDLE) == "through"), None)
        d["upper"] = upper
        lapped = (upper is not None and off is not None and off <= L.IN_LINE_TOL and lap >= L.LAP_MIN
                  and ohs_a["low"] > 0.0 and ohs_b["high"] > 0.0)
        d["lapped_in_line"] = bool(lapped)
        return (1.0 if lapped else 0.5), d
    if left or right:
        return 0.25, d
    d["zero_reason"] = "nothing_bears"
    return 0.0, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: both boards' end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.history: dict[str, deque] = {}  # board -> deque of (end, end) for this step and the 10 before it
        self.prev = None  # {board: (centre, velocity)} at the previous valid control step
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(boards: dict) -> bool:
        """The starting layout: both boards flat on the table on their marks, lying along x, still."""
        for n, s in boards.items():
            sx, sy = L.BOARD_START_X, L.BOARD_START_Y.get(n, float("nan"))
            on_mark = math.hypot(float(s.pos[0]) - sx, float(s.pos[1]) - sy) < L.START_TOL
            on_table = abs(float(s.low[2]) - L.TOP) < 0.003
            along_x = heading_err_deg(s) > 90.0 - L.START_HEADING_DEG
            slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.05
            if not (on_mark and on_table and along_x and tilt_deg(s) < 2.0 and slow):
                return False
        return True

    def _teleported(self, now: dict) -> bool:
        """Some board moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for n, (p0, v0) in self.prev.items() if n in now
        )

    def moved(self, name: str) -> float | None:
        """The farthest either end of the board has been, over the last 0.5 s, from where it is now (m); None until
        0.5 s of this trial has been seen."""
        h = self.history.get(name)
        if h is None or len(h) < self.window + 1:
            return None
        a, b = h[-1]
        return float(max(max(np.linalg.norm(p - a), np.linalg.norm(q - b)) for p, q in h))

    def update(self, boards: dict) -> Result:
        self.steps += 1
        if not all(valid(s) for s in boards.values()):  # a blow-up: score 0 for this step, keep everything else
            self.bad_steps += 1
            self.last = self.result(boards)
            return self.last
        now = {n: (np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for n, s in boards.items()}
        if self.prev is not None and self._teleported(now) and self.at_start(boards):
            bad = self.bad_steps
            self.reset()  # the simulator put both boards back on their marks: a new episode
            self.steps, self.bad_steps = 1, bad
        self.prev = now
        for n, s in boards.items():
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(ends(s))
        self.last = self.result(boards)
        if self.last.score > 0 and self.last.score not in self.first_at:
            self.first_at[self.last.score] = self.steps
        return self.last

    def result(self, boards: dict) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"step": self.steps, "bad_steps": self.bad_steps,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = [n for n, s in boards.items() if not valid(s)]
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_boards": bad, "zero_reason": "invalid_state",
                                       **trial})
        moved = {n: self.moved(n) for n in boards}
        still = {n: m is not None and m <= L.STILL_TOL for n, m in moved.items()}
        score, details = evaluate(boards, still)
        details["moved_mm"] = {n: None if m is None else round(m * 1000, 2) for n, m in moved.items()}
        return Result(score, bool(score == 1.0), {"valid_state": True, **details, **trial})
