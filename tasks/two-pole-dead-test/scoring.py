"""The scorer, as plain functions of the two probes' states and the arms' state. NumPy only; no simulator needed.

The procedure is about the whole trial, not only its end: the tester must be proved on the proving unit before any
supply test counts, every pair of the five supply terminals must be tested, and the tester proved again after the
last of them; and at no moment may the two metal tips come close together, a metal tip touch two terminals at once,
a probe that is not held touch a terminal, or a probe be put down anywhere but a rest. So the scorer keeps a record
(which pairs were tested and when, which rule was broken and when) and is fed one observation per control step by
``task.success()``.

    Tracker.update(red, black, arms) -> Result(score, success, details)

``red`` and ``black`` are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then 'held' is taken as met for that step (both probes held,
by different grippers, and at the end neither held) and the step is counted in ``arms_unavailable``. An object state
with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as
it was.

Probe geometry (see layout.py). Each probe is a 26 x 300 x 26 mm bar; its 'metal tip' is the 20 mm at its tip end and
its 'contact point' is the centre of the tip's underside, 10 mm from the end. The bar is square, so it may be gripped
with any of its four long faces down: the 'underside' is whichever of them faces most nearly down.

- **On a post**: the contact point within the post's 8 mm radius of its axis horizontally, and between 3 mm below
  and 4 mm above the post's top.
- **Level**: the underside faces within 10 degrees of straight down (this bounds both pitch and roll).
- **Steady**: the contact point moves at under 1 cm/s.
- **Held**: a gripper's finger gap is at most 32 mm (the probe's 26 mm plus 6 mm) and the point between its finger
  tips is within 15 mm of the probe's long axis, on the insulated handle: between the far end and 40 mm from the tip
  end. A gripper can hold only one probe this way.
- **A test** of a pair of terminals: both probes held by different grippers, each tip on a different post, both level
  and steady, the same pair for 20 control steps in a row (1 s). Each run of 20 steps is one test.
- **Gap between the tips**: the distance between the two 26 x 20 x 26 mm metal tips, surface to surface (computed from
  98 points on each tip's surface against the other tip's box, which is within 1 mm of the exact distance).
- **On a rest**: both ends of the probe's axis over a rest's 60 x 320 mm board, and its lowest point between 3 mm
  below and 4 mm above the board's top.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
TIP_HALF = np.array([L.PROBE_W / 2, L.TIP_LEN / 2, L.PROBE_HALF_H])  # the metal tip's half extents (body axes)
_g = np.linspace(-1.0, 1.0, 5)
_GRID = np.array([p for p in itertools.product(_g, _g, _g) if max(abs(v) for v in p) == 1.0])  # 98 surface points
POST_NAMES = tuple(L.POSTS)
POST_XY = np.array([L.POSTS[n] for n in POST_NAMES])
SUPPLY_SET = frozenset(L.SUPPLY_PAIRS)


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


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- one probe's geometry --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    centre: np.ndarray
    R: np.ndarray  # body axes (columns)
    axis: np.ndarray  # unit vector from the centre toward the tip end
    down: np.ndarray  # outward normal of the long face that faces most nearly down
    tip_centre: np.ndarray  # centre of the 20 mm metal tip
    pin: np.ndarray  # contact point: centre of the tip's underside
    pin_vel: np.ndarray
    level_deg: float  # angle between the underside's normal and straight down


def probe(s, name: str) -> Probe:
    c = np.asarray(s.pos, dtype=float)
    R = rotation(s.quat_xyzw)
    axis = R[:, 1] * L.TIP_SIGN[name]
    k = 0 if abs(R[2, 0]) > abs(R[2, 2]) else 2  # body x or body z: whichever is nearer vertical
    down = R[:, k] * (-1.0 if R[2, k] > 0 else 1.0)
    tip_centre = c + axis * (L.PROBE_LEN / 2 - L.TIP_LEN / 2)
    pin = tip_centre + down * L.PROBE_HALF_H
    w = np.asarray(s.ang_vel, dtype=float)
    pin_vel = np.asarray(s.lin_vel, dtype=float) + np.cross(w, pin - c)
    level = float(np.degrees(np.arccos(np.clip(-down[2], -1.0, 1.0))))
    return Probe(c, R, axis, down, tip_centre, pin, pin_vel, level)


def tip_points(p: Probe) -> np.ndarray:
    """98 points on the metal tip's surface (world, n x 3)."""
    return p.tip_centre + (_GRID * TIP_HALF) @ p.R.T


def point_tip_distance(pts, p: Probe) -> np.ndarray:
    """Distance from each point to the metal tip of ``p`` (0 inside it)."""
    local = (np.atleast_2d(pts) - p.tip_centre) @ p.R
    return np.linalg.norm(np.maximum(np.abs(local) - TIP_HALF, 0.0), axis=1)


def tip_gap(a: Probe, b: Probe) -> float:
    """Surface-to-surface distance between the two metal tips (0 when they touch or overlap)."""
    return float(min(point_tip_distance(tip_points(a), b).min(), point_tip_distance(tip_points(b), a).min()))


def post_distances(pts) -> np.ndarray:
    """Distance from each point (rows) to each post (columns, in POST_NAMES order), every post a 16 mm cylinder from
    its base's top to its contact cap's top."""
    pts = np.atleast_2d(pts)
    d_xy = np.linalg.norm(pts[:, None, :2] - POST_XY[None, :, :], axis=2)
    radial = np.maximum(d_xy - L.POST_RADIUS, 0.0)
    z = pts[:, None, 2]
    vertical = np.maximum(np.maximum(z - L.POST_TOP, L.BASE_TOP - z), 0.0)
    return np.hypot(radial, vertical)


def posts_touched(p: Probe) -> list[str]:
    """The posts the metal tip is within 2 mm of (touching)."""
    d = post_distances(tip_points(p)).min(axis=0)
    return [n for n, v in zip(POST_NAMES, d) if v < L.TOUCH_GAP]


def on_post(p: Probe) -> str | None:
    """The post whose top the tip rests on (contact point within 8 mm of its axis, -3..+4 mm of its top), or None."""
    dz = float(p.pin[2] - L.POST_TOP)
    if not -L.ON_POST_BELOW <= dz <= L.ON_POST_ABOVE:
        return None
    d = np.linalg.norm(POST_XY - p.pin[:2], axis=1)
    i = int(np.argmin(d))
    return POST_NAMES[i] if d[i] <= L.ON_POST_R else None


def is_level(p: Probe) -> bool:
    return p.level_deg <= L.LEVEL_DEG


def is_steady(p: Probe) -> bool:
    return float(np.linalg.norm(p.pin_vel)) < L.STEADY_SPEED


def holds(arm, p: Probe) -> bool:
    """The gripper is closed on the probe's insulated handle: finger gap at most 32 mm, the point between its finger
    tips within 15 mm of the probe's axis, between the far end and 40 mm from the tip end."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    if gap > L.PROBE_W + L.HELD_GAP_SLACK:
        return False
    r = np.asarray(arm.tips, dtype=float) - p.centre
    along = float(r @ p.axis)
    off = float(np.linalg.norm(r - along * p.axis))
    return bool(off <= L.HELD_AXIS and -L.PROBE_LEN / 2 <= along <= L.PROBE_LEN / 2 - L.GRIP_FROM_END_MIN)


def holders(arms, p: Probe) -> set[str]:
    return {side for side in ("left", "right") if holds(arms[side], p)}


def held_apart(h_red: set, h_black: set) -> bool:
    """Each probe held, by different grippers."""
    return any(a != b for a in h_red for b in h_black)


def on_rest(s, p: Probe, rest_y: float) -> bool:
    """Both ends of the axis over the rest's board, the lowest point within -3..+4 mm of the board's top."""
    ends = (p.centre + p.axis * L.PROBE_LEN / 2, p.centre - p.axis * L.PROBE_LEN / 2)
    over = all(abs(e[0] - L.REST_X) <= L.REST_SIZE[0] / 2 and abs(e[1] - rest_y) <= L.REST_SIZE[1] / 2 for e in ends)
    low = float(s.low[2]) - L.REST_TOP
    return bool(over and -L.REST_LOW_BELOW <= low <= L.REST_LOW_TOL)


def on_any_rest(s, p: Probe) -> bool:
    return any(on_rest(s, p, y) for y in L.REST_Y.values())


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def pair_name(pair) -> str:
    return "-".join(n for n in POST_NAMES if n in pair)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "tips_short": "the two metal tips came within 15 mm of each other",
    "tip_bridged": "a metal tip touched two terminals at once",
    "loose_on_terminal": "a probe that was not held touched a terminal with its metal tip",
    "probe_set_down": "a probe was neither held nor on a rest for more than 0.25 s",
    "off_table": "a probe fell off the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.hold_steps = int(round(L.TEST_HOLD_S * control_hz))
        self.loose_steps = int(round(L.LOOSE_GRACE_S * control_hz))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.first_prove: int | None = None  # control step of the first prove
        self.reprove: int | None = None  # control step of the prove after all 10 pairs (None if none, or cancelled)
        self.tested: dict[frozenset, int] = {}  # supply pair -> the step it was first tested after the first prove
        self.tests: list[tuple[str, int]] = []  # every test made, in order: (pair, step)
        self.early_tests: set[frozenset] = set()  # supply pairs tested before the first prove (they count nothing)
        self.cancelled_reproves = 0  # re-proves undone by a later supply test
        self.run_pair: frozenset | None = None  # the pair currently being held, and for how many steps
        self.run_len = 0
        self.loose_run = {n: 0 for n in L.PROBES}  # steps in a row neither held nor on a rest
        self.min_tip_gap = math.inf
        self.arms_unavailable = 0
        self.bad_steps = 0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of both probes at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(red, black) -> bool:
        """Both probes lying still on their own rests, on their starting marks."""
        for name, s in (("red_probe", red), ("black_probe", black)):
            if np.hypot(s.pos[0] - L.REST_X, s.pos[1] - L.REST_Y[name]) > L.START_TOL:
                return False
            if abs(float(s.low[2]) - L.REST_TOP) > 0.003 or not is_resting(s):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some probe moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def _test(self, pair: frozenset) -> None:
        """A test of ``pair`` has just been completed (held 1 s)."""
        self.tests.append((pair_name(pair), self.steps))
        if pair == L.PROVE_PAIR:
            if self.first_prove is None:
                self.first_prove = self.steps
            elif len(self.tested) == len(L.SUPPLY_PAIRS) and self.reprove is None:
                self.reprove = self.steps
        elif pair in SUPPLY_SET:
            if self.first_prove is None:
                self.early_tests.add(pair)
            else:
                self.tested.setdefault(pair, self.steps)
                if self.reprove is not None:  # testing again after the re-prove: it must be proved again
                    self.reprove = None
                    self.cancelled_reproves += 1
        # a proving terminal paired with a supply terminal is not a test of anything: it counts for nothing

    def update(self, red, black, arms=None) -> Result:
        if not (valid(red) and valid(black)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(red, black, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (red, black)]
        if self.prev is not None and self._teleported(now) and self.at_start(red, black):
            self.reset()  # the simulator put both probes back on their rests: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1
        states = {"red_probe": red, "black_probe": black}
        pr = {n: probe(s, n) for n, s in states.items()}
        held = {n: (holders(arms, p) if have_arms else {"left", "right"}) for n, p in pr.items()}

        gap = tip_gap(pr["red_probe"], pr["black_probe"])
        self.min_tip_gap = min(self.min_tip_gap, gap)
        if gap < L.SHORT_GAP:
            self._flag("tips_short")
        for n, s in states.items():
            touched = posts_touched(pr[n])
            if len(touched) >= 2:
                self._flag("tip_bridged")
            if touched and not held[n]:
                self._flag("loose_on_terminal")
            if held[n] or on_any_rest(s, pr[n]):
                self.loose_run[n] = 0
            else:
                self.loose_run[n] += 1
                if self.loose_run[n] > self.loose_steps:
                    self._flag("probe_set_down")
            if off_table(s):
                self._flag("off_table")

        posts = {n: on_post(p) for n, p in pr.items()}
        ok = (held_apart(held["red_probe"], held["black_probe"])
              and None not in posts.values() and posts["red_probe"] != posts["black_probe"]
              and all(is_level(p) and is_steady(p) for p in pr.values()))
        if ok:
            pair = frozenset(posts.values())
            if pair == self.run_pair:
                self.run_len += 1
            else:
                self.run_pair, self.run_len = pair, 1
            if self.run_len == self.hold_steps:
                self._test(pair)
        else:
            self.run_pair, self.run_len = None, 0

        self.last = self.result(red, black, arms)
        return self.last

    def result(self, red, black, arms=None) -> Result:
        if not (valid(red) and valid(black)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        states = {"red_probe": red, "black_probe": black}
        pr = {n: probe(s, n) for n, s in states.items()}
        held = {n: (sorted(holders(arms, p)) if have_arms else None) for n, p in pr.items()}
        let_go = all(not h for h in held.values()) if have_arms else True  # unreadable: taken as met, counted
        home = {n: on_rest(states[n], pr[n], L.REST_Y[n]) for n in states}
        n_pairs = len(self.tested)
        if self.breach or self.first_prove is None:
            score = 0.0
        elif self.reprove is not None and all(home.values()) and let_go:
            score = 1.0
        elif self.reprove is not None:
            score = 0.9
        else:
            score = round(0.25 + 0.05 * n_pairs, 2)
        success = bool(score == 1.0 and all(is_resting(s) for s in states.values()))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "first_prove_step": self.first_prove,
            "reprove_step": self.reprove,
            "n_pairs": n_pairs,
            "pairs_tested": sorted(pair_name(p) for p in self.tested),
            "pairs_missing": sorted(pair_name(p) for p in L.SUPPLY_PAIRS if p not in self.tested),
            "early_tests": sorted(pair_name(p) for p in self.early_tests),
            "cancelled_reproves": self.cancelled_reproves,
            "n_tests": len(self.tests),
            "holding": pair_name(self.run_pair) if self.run_pair else None,
            "holding_s": round(self.run_len * self.dt, 2),
            "red_on_post": on_post(pr["red_probe"]),
            "black_on_post": on_post(pr["black_probe"]),
            "red_pin_mm": [_mm(v) for v in pr["red_probe"].pin],
            "black_pin_mm": [_mm(v) for v in pr["black_probe"].pin],
            "red_pin_above_post_mm": _mm(pr["red_probe"].pin[2] - L.POST_TOP),
            "black_pin_above_post_mm": _mm(pr["black_probe"].pin[2] - L.POST_TOP),
            "red_level_deg": round(pr["red_probe"].level_deg, 2),
            "black_level_deg": round(pr["black_probe"].level_deg, 2),
            "tip_gap_mm": _mm(tip_gap(pr["red_probe"], pr["black_probe"])),
            "min_tip_gap_mm": _mm(self.min_tip_gap),
            "red_held_by": held["red_probe"],
            "black_held_by": held["black_probe"],
            "probes_on_own_rests": home,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "probes_resting": {n: is_resting(s) for n, s in states.items()},
        }
        return Result(score, success, details)
