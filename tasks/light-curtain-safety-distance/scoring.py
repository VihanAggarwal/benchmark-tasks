"""The scorer, as plain functions of the two posts' states. NumPy only; no simulator needed.

The score is read from where the two light-curtain posts stand at the end, as an installer checks them with a tape
from the die line and a look from above: is each post standing upright on the opening's floor against its side
guard, is the beam (the line through the posts' centres) at least the ISO 13855 safety distance S = 168 mm from the
die line and no more than 20 mm beyond it, and do the posts' lens faces face each other along that beam. The scorer
keeps only each post's last half second of positions (to tell a post at rest from one still moving) and is fed one
observation per control step by ``task.success()``:

    Tracker.update({"emitter_post": state, "receiver_post": state}) -> Result(score, success, details)

Each state is anything with ``pos`` (bounding-box centre, which is the box's centre), ``quat_xyzw``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has (``lin_vel`` and ``ang_vel`` are
not used: a gripped body's reported velocity chatters, so rest is judged from positions). A state with a
non-finite number or a zero quaternion (a simulator blow-up), or a missing post, scores 0 for that step and leaves
the tracker as it was.

A post is a plain box: its two broad faces are alike, so either one is taken as its lens window, and either end
may be down.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

POSTS = ("emitter_post", "receiver_post")
START = {"emitter_post": L.POST_START["left"], "receiver_post": L.POST_START["right"]}


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


def valid(post) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(post, k), dtype=float).ravel() for k in ("pos", "quat_xyzw", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[2].size == 3 and parts[3].size == 3
                and np.linalg.norm(parts[1]) > 0.5)


def corners(post) -> np.ndarray:
    """The post's eight corners in the world, (8, 3)."""
    r = rotation(post.quat_xyzw)
    half = np.array(L.POST_HALF)
    c = np.asarray(post.pos, dtype=float)
    return np.array([c + r @ (half * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def tilt_deg(post) -> float:
    """Angle between the vertical and the post's long axis (0..90 degrees; either end may be down)."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(post.quat_xyzw)[2, 2]), 0.0, 1.0))))


def beam_distance(post) -> float:
    """D: how far the post's centre line (the beam passes through it) is in front of the die line's near edge,
    along x (m; negative = past the line, under the press)."""
    return float(L.DIE_X - float(post.pos[0]))


def side(post) -> str:
    """Which guard the post is nearer: 'left' (+y) or 'right' (-y)."""
    return "left" if float(post.pos[1]) >= 0.0 else "right"


def guard_gap(post) -> float:
    """Gap between the post's outer face (its farthest point toward its nearer guard) and that guard's inner face
    (m; negative = into the guard)."""
    sgn = 1.0 if side(post) == "left" else -1.0
    return float(L.GUARD_INNER_Y - (sgn * corners(post)[:, 1]).max())


def bottom_gap(post) -> float:
    """Height of the post's lowest point above the opening's floor plate (m)."""
    return float(post.low[2] - L.FLOOR_TOP)


def over_floor(post) -> bool:
    """The post's centre, seen from above, is over the opening's floor plate (between the guards' inner faces,
    from the opening's front edge to the die line)."""
    x, y = float(post.pos[0]), float(post.pos[1])
    return bool(L.OPEN_X0 <= x <= L.DIE_X and abs(y) < L.GUARD_INNER_Y)


def too_close(post) -> bool:
    """Score 0: the post's centre is between the guards (up to their outer faces) and less than S from the die line
    (including past it, over the press), at any height."""
    return bool(abs(float(post.pos[1])) <= L.GUARD_OUTER_Y and beam_distance(post) < L.S_MIN - L.EPS)


def facing_deg(post, other) -> float:
    """Angle, seen from above, between the post's lens axis (through its two broad faces) and the line from its
    centre to the other post's centre (0..90 degrees: either broad face may be the lens)."""
    a = rotation(post.quat_xyzw)[:2, 0]
    line = np.asarray(other.pos, dtype=float)[:2] - np.asarray(post.pos, dtype=float)[:2]
    na, nl = float(np.hypot(*a)), float(np.hypot(*line))
    if na < 1e-9 or nl < 1e-9:  # lying with its broad faces level, or both posts in one place
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(float(a @ line)) / (na * nl)))))


def pose_check(post) -> dict:
    """Everything 'placed' needs except rest, each as a fact with its measurement."""
    d, gap, tilt, bot = beam_distance(post), guard_gap(post), tilt_deg(post), bottom_gap(post)
    facts = {
        "upright": tilt <= L.TILT_MAX_DEG + L.EPS,
        "on_floor": over_floor(post) and abs(bot) <= L.FLOOR_TOL + L.EPS,
        "at_guard": -L.GAP_PEN - L.EPS <= gap <= L.GAP_MAX + L.EPS,
        "in_window": L.S_MIN - L.EPS <= d <= L.S_MAX + L.EPS,
    }
    facts["pose_ok"] = all(facts.values())
    facts.update({"side": side(post), "d_mm": round(d * 1000, 2), "gap_mm": round(gap * 1000, 2),
                  "tilt_deg": round(tilt, 2), "bottom_mm": round(bot * 1000, 2), "too_close": too_close(post)})
    return facts


def tracked_points(post) -> np.ndarray:
    """Three points of the post whose motion shows any move, turn or tilt: its centre, the centre of its top and
    the centre of a broad face. (3, 3)."""
    r = rotation(post.quat_xyzw)
    c = np.asarray(post.pos, dtype=float)
    return np.array([c, c + r[:, 2] * L.POST_HALF[2], c + r[:, 0] * L.POST_HALF[0]])


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each post's tracked points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # {name: points} for this step and the 10 before it
        self.prev = None  # {name: centre} at the previous valid step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, post) -> bool:
        """The starting layout for one post: standing upright on its mat, on its mark."""
        x0, y0 = START[name]
        on_mark = (math.hypot(float(post.pos[0]) - x0, float(post.pos[1]) - y0) < L.START_TOL
                   and abs(float(post.low[2]) - (L.TOP + L.MAT_SIZE[2])) < 0.003)
        return bool(on_mark and tilt_deg(post) < 2.0)

    def moved(self, name: str) -> float | None:
        """The farthest any tracked point of the post has been, over the last 0.5 s, from where it is now (m); None
        until 0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        now = self.history[-1][name]
        return float(max(np.linalg.norm(h[name] - now, axis=1).max() for h in self.history))

    @staticmethod
    def _states(posts) -> dict | None:
        try:
            states = {n: posts[n] for n in POSTS}
        except (KeyError, TypeError):
            return None
        return states if all(valid(s) for s in states.values()) else None

    def update(self, posts) -> Result:
        states = self._states(posts)
        if states is None:  # a blow-up or a missing post: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(posts)
            return self.last
        centres = {n: np.asarray(s.pos, dtype=float) for n, s in states.items()}
        if self.prev is not None and all(self.at_start(n, s) for n, s in states.items()):
            if max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in POSTS) > L.RESET_JUMP:
                self.reset()  # the simulator put them back: a new episode
        self.steps += 1
        self.history.append({n: tracked_points(s) for n, s in states.items()})
        self.prev = centres
        self.last = self.result(posts)
        return self.last

    def result(self, posts) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        states = self._states(posts)
        if states is None:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        per: dict[str, dict] = {}
        for n, s in states.items():
            f = pose_check(s)
            m = self.moved(n)
            f["at_rest"] = m is not None and m <= L.STILL_TOL
            f["moved_mm"] = None if m is None else round(m * 1000, 2)
            f["placed"] = bool(f["pose_ok"] and f["at_rest"])
            per[n] = f
        details: dict = {"valid_state": True, "zero_reason": ""}
        if any(f["too_close"] for f in per.values()):
            details["zero_reason"] = "closer_than_S"
            score = 0.0
        else:
            sides = {per[n]["side"] for n in POSTS if per[n]["placed"]}  # each guard counts once
            score = 0.25 * len(sides)
            if len(sides) == 2:  # one post placed against each guard: is each lens facing the other post?
                for n, o in (POSTS, POSTS[::-1]):
                    ang = facing_deg(states[n], states[o])
                    per[n]["facing_deg"] = round(ang, 2)
                    per[n]["facing"] = ang <= L.FACING_MAX_DEG
                    score += 0.25 if per[n]["facing"] else 0.0
        details["placed_sides"] = sorted({per[n]["side"] for n in POSTS if per[n]["placed"]})
        for n in POSTS:
            details[n] = per[n]
        return Result(score, bool(score == 1.0), details)
