"""The scorer, as plain functions of the four boards' and the two stickers' states. NumPy only; no simulator needed.

The rule is about the finished stack, read from the state at the last control step: two boards on the bearers
with an air gap between them, a sticker on them directly over each bearer, two boards on the stickers right above
the first two. Any board may go in any place and either sticker over either bearer. The bearers are fixed, so
their tops come from ``layout``. A small tracker is still fed once per control step by ``task.success()``: it
keeps what the trace needs (when each score was first reached, how many observations were broken) and recognises
a new episode when the simulator puts every piece back on its starting mark in one step.

    Tracker.update(boards, stickers) -> Result(score, success, details)

``boards`` (4) and ``stickers`` (2) are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. All
are plain boxes whose sizes are in ``layout``, so their faces follow from ``pos`` and ``quat_xyzw``. Each has two
identical broad faces, so either may face down. A state with a non-finite number or a zero quaternion (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

import layout as L

BOARD_HALF = np.array(L.BOARD_SIZE, dtype=float) / 2
STICKER_HALF = np.array(L.STICKER_SIZE, dtype=float) / 2
BOARD_W = L.BOARD_SIZE[1]  # the boards' width (their short horizontal side)
STICKER_W = L.STICKER_SIZE[0]
BEARER_W = L.BEARER_SIZE[0]
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


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the piece's two broad faces (degrees, 0..90):
    a plain board or sticker is the same either way up, so whichever broad face is nearer to facing down counts.
    Lying flat 0; standing on an edge or an end 90."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def heading_err_deg(s, half, axis: str) -> float:
    """How far the piece's long side, seen from above, is turned from the world's ``axis`` ('x' or 'y'), in
    degrees 0..90 (a board or sticker is the same either way round)."""
    r = rotation(s.quat_xyzw)
    long_axis = r[:, 0] if half[0] >= half[1] else r[:, 1]
    hx, hy = float(long_axis[0]), float(long_axis[1])
    if math.hypot(hx, hy) < 1e-9:  # standing on its end: no heading
        return 90.0
    ang = math.degrees(math.atan2(hy, hx)) % 180.0  # 0 = along x
    if axis == "y":
        ang = (ang - 90.0) % 180.0
    return float(min(ang, 180.0 - ang))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def flat(s) -> bool:
    return tilt_deg(s.quat_xyzw) < L.FLAT_DEG


def laid_board(b) -> bool:
    """A board lies 'laid': flat (either broad face down) and along x within 5 degrees, either way round."""
    return bool(flat(b) and heading_err_deg(b, BOARD_HALF, "x") <= L.HEADING_DEG)


def laid_sticker(k) -> bool:
    """A sticker lies flat and along y within 5 degrees, either way round."""
    return bool(flat(k) and heading_err_deg(k, STICKER_HALF, "y") <= L.HEADING_DEG)


def off_table(s) -> bool:
    """The piece's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def on_edge(s) -> bool:
    """Standing on an edge or an end (tilted more than 45 degrees)."""
    return tilt_deg(s.quat_xyzw) > L.EDGE_DEG


# -- faces, footprints and 'rests on' ------------------------------------------------------------------------------
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


def broad_face(s, half, which: str) -> Face:
    """The piece's upper ('top') or lower ('bottom') broad face (whichever of the two faces up / down)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    up = r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]
    centre = c + up * half[2] * (1.0 if which == "top" else -1.0)
    corners = [centre + r[:, 0] * i * half[0] + r[:, 1] * j * half[1] for i, j in _SIGNS]
    return Face(centre, up, corners)


def bearer_top(i: int) -> Face:
    """The top face of fixed bearer ``i`` (0: x 0.20, 1: x 0.36)."""
    bx, a, b = L.BEARER_XS[i], L.BEARER_SIZE[0] / 2, L.BEARER_SIZE[1] / 2
    centre = (bx, L.BEARER_Y, L.BEARER_TOP)
    return Face(centre, (0.0, 0.0, 1.0), [(bx + i_ * a, L.BEARER_Y + j * b, L.BEARER_TOP) for i_, j in _SIGNS])


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-15:
        return p.mean(axis=0)
    cx = float(((x + np.roll(x, -1)) * cross).sum()) / (6 * a)
    cy = float(((y + np.roll(y, -1)) * cross).sum()) / (6 * a)
    return np.array([cx, cy])


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


def crossing(lower: Face, upper: Face) -> tuple[float, float | None]:
    """Where ``upper``'s underside crosses ``lower``'s top, seen from above: (the area of the overlap, the height
    of ``upper``'s underside above ``lower``'s top at the overlap's centroid; None if they do not overlap)."""
    part = _clip(upper.outline(), lower.outline())
    if part is None:
        return 0.0, None
    area = _area(part)
    if not np.isfinite(area) or area <= 1e-12:
        return 0.0, None
    at = _centroid(part)
    return float(area), upper.z_at(at) - lower.z_at(at)


def rests(lower: Face, upper: Face, w_lower: float, w_upper: float) -> bool:
    """``upper`` rests on ``lower``: their outlines overlap by at least half of a square crossing of the two
    widths, and ``upper``'s underside is within 3 mm of ``lower``'s top there (above or below it)."""
    area, gap = crossing(lower, upper)
    return bool(gap is not None and area >= L.REST_COVER * w_lower * w_upper and abs(gap) <= L.REST_Z_TOL)


def board_on_bearers(b) -> bool:
    """A board 'on the bearers': laid, and resting on both bearers."""
    if not laid_board(b):
        return False
    under = broad_face(b, BOARD_HALF, "bottom")
    return all(rests(bearer_top(i), under, BEARER_W, BOARD_W) for i in range(len(L.BEARER_XS)))


def sticker_on_boards(k, lower_boards) -> bool:
    """A sticker laid (flat, along y) and resting on every one of ``lower_boards``."""
    if not laid_sticker(k):
        return False
    under = broad_face(k, STICKER_HALF, "bottom")
    return all(rests(broad_face(b, BOARD_HALF, "top"), under, BOARD_W, STICKER_W) for b in lower_boards)


def board_on_stickers(b, stickers) -> bool:
    """A board laid and resting on every one of ``stickers``."""
    if not laid_board(b):
        return False
    under = broad_face(b, BOARD_HALF, "bottom")
    return all(rests(broad_face(k, STICKER_HALF, "top"), under, STICKER_W, BOARD_W) for k in stickers)


def layer_gap(a, b) -> float:
    """The air gap between two boards lying along x: their centres' distance across (in y) minus a board's
    width."""
    return float(abs(float(a.pos[1]) - float(b.pos[1])) - BOARD_W)


def nearest_bearer(k) -> tuple[int, float]:
    """(index, distance in x) of the bearer nearest to the sticker's centre."""
    d = [abs(float(k.pos[0]) - bx) for bx in L.BEARER_XS]
    i = int(np.argmin(d))
    return i, float(d[i])


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(boards, stickers) -> tuple[float, dict]:
    """The score of the stack as it stands, and what was found (JSON-clean)."""
    pieces = list(boards) + list(stickers)
    fell = [i for i, s in enumerate(pieces) if off_table(s)]
    edge = [i for i, s in enumerate(pieces) if on_edge(s)]
    d: dict = {
        "off_table": fell, "on_edge": edge,
        "max_tilt_deg": round(max(tilt_deg(s.quat_xyzw) for s in pieces), 2),
        "layer1": [], "layer1_gap_mm": None, "layer1_ok": False,
        "sticker_bearer": [None, None], "sticker_off_mm": [None, None],
        "stickers_tight": False, "stickers_loose": False,
        "layer2": [], "layer2_off_mm": [], "layer2_tight": False, "layer2_loose": False,
    }
    if fell or edge:
        return 0.0, d

    # Layer 1: exactly two boards on both bearers, with an air gap of 30-90 mm between them.
    layer1 = [i for i, b in enumerate(boards) if board_on_bearers(b)]
    d["layer1"] = layer1
    if len(layer1) == 2:
        gap = layer_gap(boards[layer1[0]], boards[layer1[1]])
        d["layer1_gap_mm"] = _mm(gap)
        d["layer1_ok"] = bool(L.GAP_MIN <= gap <= L.GAP_MAX)
    if not d["layer1_ok"]:
        return 0.0, d
    lower = [boards[i] for i in layer1]

    # Stickers: each laid along y on both layer-1 boards, its centre over a bearer (each over a different one).
    offs: list[float | None] = []
    which: list[int | None] = []
    for j, k in enumerate(stickers):
        if sticker_on_boards(k, lower):
            i, off = nearest_bearer(k)
            which.append(i)
            offs.append(off)
        else:
            which.append(None)
            offs.append(None)
    d["sticker_bearer"] = which
    d["sticker_off_mm"] = [_mm(o) for o in offs]
    apart = None not in which and len(set(which)) == len(which)
    d["stickers_loose"] = bool(apart and all(o <= L.STICKER_LOOSE for o in offs))
    d["stickers_tight"] = bool(apart and all(o <= L.STICKER_TIGHT for o in offs))

    # Layer 2: the other two boards laid on both stickers, each right above a different layer-1 board.
    if d["stickers_loose"]:
        upper = [i for i, b in enumerate(boards) if i not in layer1 and board_on_stickers(b, stickers)]
        d["layer2"] = upper
        if len(upper) == 2:
            best = None
            for perm in permutations(range(2)):
                o = [abs(float(boards[upper[m]].pos[1]) - float(lower[perm[m]].pos[1])) for m in range(2)]
                if best is None or max(o) < max(best):
                    best = o
            d["layer2_off_mm"] = [_mm(o) for o in best]
            d["layer2_loose"] = bool(max(best) <= L.STACK_LOOSE)
            d["layer2_tight"] = bool(max(best) <= L.STACK_TIGHT)

    if d["stickers_loose"] and d["layer2_loose"]:
        score = 1.0 if (d["stickers_tight"] and d["layer2_tight"]) else 0.75
    elif d["stickers_tight"]:
        score = 0.5
    else:
        score = 0.25
    return score, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the trace needs about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every piece at the previous valid control step

    @staticmethod
    def at_start(boards, stickers) -> bool:
        """The starting layout: every piece still, flat on the table, square, on its own starting mark."""
        marks = [(L.BOARD_START_X, y, "x", BOARD_HALF) for _, y in L.BOARDS]
        marks += [(x, y, "x", STICKER_HALF) for _, x, y in L.STICKERS]  # the stickers start along x
        for s, (mx, my, axis, half) in zip(list(boards) + list(stickers), marks, strict=True):
            on_mark = math.hypot(float(s.pos[0]) - mx, float(s.pos[1]) - my) < L.START_TOL
            on_table = abs(float(s.low[2]) - L.TOP) < 0.003
            square = tilt_deg(s.quat_xyzw) < 1.0 and heading_err_deg(s, half, axis) < 1.0
            if not (on_mark and on_table and square and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some piece moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def update(self, boards, stickers) -> Result:
        pieces = list(boards) + list(stickers)
        self.steps += 1
        if not all(valid(s) for s in pieces):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(boards, stickers)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in pieces]
        if self.prev is not None and self._teleported(now) and self.at_start(boards, stickers):
            self.reset()  # the simulator put every piece back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.last = self.result(boards, stickers)
        for tier in self.TIERS:
            if self.last.score >= tier and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, boards, stickers) -> Result:
        pieces = list(boards) + list(stickers)
        bad = [i for i, s in enumerate(pieces) if not valid(s)]
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_pieces": bad, **trial})
        score, details = evaluate(boards, stickers)
        still = all(is_resting(s) for s in pieces)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
