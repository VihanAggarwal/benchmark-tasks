"""The scorer, as plain functions of the manikin's two slider positions and the arms' state. NumPy only; no
simulator needed.

CPR is a rhythm: nothing in the final state says whether it was done well. So the scorer is fed one observation
per control step by ``task.success()`` and keeps a record of every compression (its peak depth and when it came,
the recoil before it, how far the nearest gripper low enough to be pressing was from the mark at the peak) and
every breath (when it began and ended, how deep the cap went, whether a gripper was pressing the cap at its
deepest, how deep the chest was meanwhile, whether a gripper waited over the mark throughout). From that record it
groups the compressions into cycles and the breaths into the pauses between cycles, and scores the trial.

    Tracker.update(joints, arms) -> Result(score, success, details)

``joints`` is the manikin's joint positions, ``{"chest": m, "bag": m}``, positive pressed down and 0 as drawn, as
mechkit's ``AssemblyState.joints`` reports them. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}``
with ``tcp`` (tool point), ``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or
``None`` when the arms could not be read; then the arm rules (a gripper pressing the mark at each peak, one
pressing the cap at each breath's deepest and one waiting over the mark during each breath, both grippers clear at
the end) are taken as met for that step and the step is counted in ``arms_unavailable``. A joint reading
that is missing or not finite (a simulator blow-up) scores 0 for that step and leaves the record as it was.

Definitions (depths in mm, d = chest, b = bag; times are control steps x 1/20 s):
- compression: d rises through 20; it ends when d falls below 10. Its depth is its peak d, its time the step of
  the peak, its recoil the lowest d between the previous compression's end (or the trial's start) and its start;
- good compression: depth 50-60, recoil <= 5, at the peak some gripper pressing the mark (its tool point within
  25 mm of the mark horizontally and no more than 40 mm above the mark's top where it is then: a gripper hovering
  over the mark while the other presses the plate elsewhere does not count), and 0.43-0.70 s after the previous
  peak of its cycle (the first of a cycle is exempt);
- cycle: a run of compressions whose consecutive peaks are less than 2.0 s apart; rate 60 (n - 1) / (last peak
  - first peak) per minute;
- breath: a run of steps with b over 15 in which b reaches 25; it counts if it lasts 0.6-2.0 s, at its deepest a
  gripper is pressing the cap (a tool point within 60 mm of the cap's centre horizontally and no more than 40 mm
  above its top where it then is), d stays at or under 10 throughout, and throughout some gripper waits over the
  mark (tool point within 25 mm of it horizontally and no more than 60 mm above it): the compressor's hands stay in
  place while the other gripper gives the breath, so one arm cannot do both jobs (not even by leaning a link on the
  cap while its own gripper waits over the mark, 39 cm away). A breath
  starting between two peaks of one cycle is during that cycle; otherwise it belongs to the pause after the last
  cycle that ended before it (or comes before the first cycle);
- pause: from a cycle's last peak to the next cycle's first peak; after the last cycle, to the end of its second
  counted breath (its only one if it has one; no pause if it has none).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
_PLATE_HALF = L.PLATE_SIZE[0] / 2
_CAP_HALF = L.CAP_SIZE[0] / 2


@dataclass(frozen=True)
class Compression:
    start: float  # s: d rose through 20 mm
    end: float  # s: d fell below 10 mm
    depth: float  # mm: its peak d
    peak_t: float  # s: the step of the peak
    recoil: float  # mm: the lowest d between the previous compression's end and this one's start
    mark_dist: float | None  # m: at the peak, the horizontal distance from the mark to the nearest tool point no
    #                          more than 40 mm above the mark's top (inf: none that low; None: arms unreadable then)


@dataclass(frozen=True)
class Breath:
    start: float  # s: the first step with b over 15 mm
    end: float  # s: the step on which b was back at or under 15 mm
    depth: float  # mm: the cap's deepest
    chest_max: float  # mm: the chest's deepest meanwhile
    hand_at_mark: bool = True  # some gripper waited over the mark on every step of it (unreadable steps count)
    by_gripper: bool = True  # at the cap's deepest a gripper was pressing it (an unreadable step counts)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def counted(self) -> bool:
        return bool(L.BREATH_MIN_S - 1e-9 <= self.duration <= L.BREATH_MAX_S + 1e-9
                    and self.chest_max <= L.BREATH_CHEST_MAX_MM and self.hand_at_mark and self.by_gripper)


# -- readings ----------------------------------------------------------------------------------------------------
def read_joints(joints) -> tuple[float, float] | None:
    """(chest, bag) in mm, or None when either is missing or not finite."""
    try:
        d, b = float(joints["chest"]) * 1000.0, float(joints["bag"]) * 1000.0
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not (math.isfinite(d) and math.isfinite(b)):
        return None
    return d, b


def arms_valid(arms) -> bool:
    """Both arms present with a finite tool point, tip point and opening (rigstate.arms() can be None)."""
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


def mark_distance(arms) -> float:
    """Horizontal distance (m) from the mark's centre to the nearer gripper's tool point."""
    return min(float(math.hypot(arms[s].tcp[0] - L.MARK_X, arms[s].tcp[1] - L.MARK_Y)) for s in ARMS)


def press_distance(arms, d_mm: float) -> float:
    """Horizontal distance (m) from the mark's centre to the nearest tool point that is low enough to be pressing it:
    no more than PRESS_ABOVE above the mark's top where it is with the chest ``d_mm`` down. inf when none is."""
    top = L.TOP + L.MARK_TOP - d_mm / 1000.0 + L.PRESS_ABOVE
    dists = [float(math.hypot(arms[s].tcp[0] - L.MARK_X, arms[s].tcp[1] - L.MARK_Y)) for s in ARMS
             if float(arms[s].tcp[2]) <= top]
    return min(dists, default=math.inf)


def bag_press_distance(arms, b_mm: float) -> float:
    """Horizontal distance (m) from the cap's centre to the nearest tool point that is low enough to be pressing it:
    no more than PRESS_ABOVE above the cap's top where it is with the cap ``b_mm`` down. inf when none is."""
    top = L.TOP + L.CAP_TOP - b_mm / 1000.0 + L.PRESS_ABOVE
    dists = [float(math.hypot(arms[s].tcp[0] - L.BAG_X, arms[s].tcp[1] - L.BAG_Y)) for s in ARMS
             if float(arms[s].tcp[2]) <= top]
    return min(dists, default=math.inf)


def pressing_bag(arms, b_mm: float) -> bool:
    """A gripper is pressing the cap: its tool point within BAG_TOL of the cap's centre horizontally and low enough."""
    return bag_press_distance(arms, b_mm) <= L.BAG_TOL


def hand_at_mark(arms) -> bool:
    """Some gripper waits over the mark: its tool point within 25 mm of the mark's centre horizontally and no more
    than 60 mm above the mark's top as drawn (it may touch the mark; the chest's depth rule is separate)."""
    top = L.TOP + L.MARK_TOP + L.HAND_ABOVE_MARK
    return any(math.hypot(arms[s].tcp[0] - L.MARK_X, arms[s].tcp[1] - L.MARK_Y) <= L.MARK_TOL
               and float(arms[s].tcp[2]) <= top for s in ARMS)


def point_box_distance(p, lo, hi) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))))


def plate_box(depth_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """World box around the chest plate and its mark, pushed down ``depth_mm``."""
    dz = depth_mm / 1000.0
    lo = np.array([L.MARK_X - _PLATE_HALF, L.MARK_Y - _PLATE_HALF, L.TOP + L.PLATE_TOP - L.PLATE_SIZE[2] - dz])
    hi = np.array([L.MARK_X + _PLATE_HALF, L.MARK_Y + _PLATE_HALF, L.TOP + L.MARK_TOP - dz])
    return lo, hi


def cap_box(depth_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """World box around the bag's cap, pushed down ``depth_mm``."""
    dz = depth_mm / 1000.0
    lo = np.array([L.BAG_X - _CAP_HALF, L.BAG_Y - _CAP_HALF, L.TOP + L.CAP_TOP - L.CAP_SIZE[2] - dz])
    hi = np.array([L.BAG_X + _CAP_HALF, L.BAG_Y + _CAP_HALF, L.TOP + L.CAP_TOP - dz])
    return lo, hi


def clearance(arms, d_mm: float, b_mm: float) -> float:
    """The smallest distance (m) from either gripper's tool point or finger-tip point to the plate or the cap."""
    boxes = (plate_box(d_mm), cap_box(b_mm))
    return min(point_box_distance(p, lo, hi) for s in ARMS for p in (arms[s].tcp, arms[s].tips) for lo, hi in boxes)


# -- the record, analysed ----------------------------------------------------------------------------------------
def group_cycles(compressions) -> list[list[Compression]]:
    """Runs of compressions whose consecutive peaks are less than 2.0 s apart."""
    cycles: list[list[Compression]] = []
    for c in compressions:
        if cycles and c.peak_t - cycles[-1][-1].peak_t < L.CYCLE_GAP_S:
            cycles[-1].append(c)
        else:
            cycles.append([c])
    return cycles


def cycle_rate(cycle) -> float | None:
    """Compressions per minute over the cycle (None for a single compression)."""
    span = cycle[-1].peak_t - cycle[0].peak_t
    return 60.0 * (len(cycle) - 1) / span if len(cycle) > 1 and span > 0 else None


def is_good(c: Compression, interval: float | None) -> bool:
    return bool(
        L.DEPTH_MIN_MM <= c.depth <= L.DEPTH_MAX_MM
        and c.recoil <= L.RECOIL_MAX_MM
        and (c.mark_dist is None or c.mark_dist <= L.MARK_TOL)
        and (interval is None or L.INTERVAL_MIN_S - 1e-9 <= interval <= L.INTERVAL_MAX_S + 1e-9)
    )


@dataclass
class Analysis:
    cycles: list
    good: list  # one bool per compression, in order
    intervals: list  # one per compression: seconds since the previous peak of its cycle, or None (first)
    pause_breaths: list  # per cycle: the counted breaths in the pause after it
    breaths_during: int  # breaths (counted or not) starting between two peaks of one cycle
    breaths_before: int  # breaths before the first cycle (or with no compression yet)
    pauses: list  # per cycle: the pause after it (s), or None (the last cycle with no counted breath after it)

    @property
    def n_good(self) -> int:
        return int(sum(self.good))

    @property
    def share_good(self) -> float:
        return self.n_good / len(self.good) if self.good else 0.0


def analyse(compressions, breaths) -> Analysis:
    cycles = group_cycles(compressions)
    good, intervals = [], []
    for cyc in cycles:
        for i, c in enumerate(cyc):
            iv = None if i == 0 else c.peak_t - cyc[i - 1].peak_t
            intervals.append(iv)
            good.append(is_good(c, iv))
    pause_breaths: list[list[Breath]] = [[] for _ in cycles]
    during = before = 0
    for br in breaths:
        if any(cyc[0].peak_t < br.start < cyc[-1].peak_t for cyc in cycles):
            during += 1
            continue
        k = max((i for i, cyc in enumerate(cycles) if cyc[-1].peak_t < br.start), default=None)
        if k is None:
            before += 1
        elif br.counted:
            pause_breaths[k].append(br)
    pauses: list[float | None] = []
    for k, cyc in enumerate(cycles):
        if k + 1 < len(cycles):
            pauses.append(cycles[k + 1][0].peak_t - cyc[-1].peak_t)
        elif pause_breaths[k]:
            closing = pause_breaths[k][min(len(pause_breaths[k]), L.FULL_BREATHS) - 1]
            pauses.append(closing.end - cyc[-1].peak_t)
        else:
            pauses.append(None)
    return Analysis(cycles, good, intervals, pause_breaths, during, before, pauses)


def _within(n: int, bounds) -> bool:
    return bounds[0] <= n <= bounds[1]


def _pauses_ok(pauses, limit: float) -> bool:
    return all(p is None or p <= limit + 1e-9 for p in pauses)


def tier(a: Analysis, zero: bool, hands_off: bool) -> float:
    """The score from the analysed record (see task.json 'scoring')."""
    if zero or a.n_good < L.MIN_GOOD:
        return 0.0
    cyc, n_breaths = a.cycles, [len(b) for b in a.pause_breaths]
    if (len(cyc) == 2 and all(_within(len(c), L.FULL_COUNT) for c in cyc)
            and all(r is not None and L.FULL_RATE[0] <= r <= L.FULL_RATE[1] for r in map(cycle_rate, cyc))
            and a.share_good >= L.FULL_GOOD - 1e-9 and n_breaths == [L.FULL_BREATHS] * 2
            and a.breaths_during == 0 and a.breaths_before == 0 and _pauses_ok(a.pauses, L.FULL_PAUSE_S)
            and hands_off):
        return 1.0
    if (len(cyc) >= 2 and all(_within(len(c), L.PART_COUNT) for c in cyc[:2]) and a.share_good >= L.PART_GOOD - 1e-9
            and sum(n_breaths[:2]) >= L.PART_BREATHS and _pauses_ok(a.pauses[:2], L.FULL_PAUSE_S)):
        return 0.75
    if (any(_within(len(c), L.PART_COUNT) and n >= 1 for c, n in zip(cyc, n_breaths))
            and a.share_good >= L.HALF_GOOD - 1e-9 and _pauses_ok(a.pauses, L.HALF_PAUSE_S)):
        return 0.5
    return 0.25


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


def _r(v, n=2):
    return round(float(v), n) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.compressions: list[Compression] = []
        self.breaths: list[Breath] = []
        self.zero: dict[str, int] = {}  # zero rule -> the control step it first happened
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a missing or non-finite joint reading (ignored)
        self.max_depth = 0.0  # mm, the chest's deepest so far (diagnostic)
        self._comp = None  # the compression in progress: dict(start, depth, peak_t, mark, recoil)
        self._recoil_min = math.inf  # the lowest d since the last compression ended (or since the start)
        self._run = None  # the run of b over 15 mm in progress: dict(start, depth, chest_max)
        self.prev = None  # (d, b) at the previous valid step, mm
        self.speed = (0.0, 0.0)  # mm/s of the chest and the cap over the last step
        self.prev_tcp = None  # both tool points at the previous valid step (the reset fallback)
        self.last: Result = Result(0.0, False, {})

    @property
    def t(self) -> float:
        return self.steps * self.dt

    def _jumped_to_ready(self, arms, d: float, b: float) -> bool:
        """The fallback reset signal: a gripper jumped RESET_JUMP or more in one control step, to its ready pose,
        both grippers open and both sliders at rest. A policy cannot move an arm that far in one step."""
        tcp = {s: np.asarray(arms[s].tcp, dtype=float) for s in ARMS}
        jumped = any(float(np.linalg.norm(tcp[s] - self.prev_tcp[s])) >= L.RESET_JUMP for s in ARMS)
        at_ready = all(float(np.linalg.norm(tcp[s] - np.asarray(L.READY_TCP[s]))) <= L.READY_TOL
                       and float(arms[s].opening) >= 0.9 for s in ARMS)
        return bool(jumped and at_ready and d <= L.REST_TOL_MM and b <= L.REST_TOL_MM)

    def update(self, joints, arms=None) -> Result:
        reading = read_joints(joints)
        if reading is None:  # a blow-up: score 0 for this step, keep the record as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(joints, arms)
            return self.last
        d, b = reading
        have_arms = arms_valid(arms)
        if have_arms and self.prev_tcp is not None and self._jumped_to_ready(arms, d, b):
            self.reset()  # the simulator put the arms back: a new episode (rigstate's on_reset normally does this
            #               first; this is the fallback)
        self.steps += 1
        t = self.t
        if self.prev is not None:
            self.speed = ((d - self.prev[0]) / self.dt, (b - self.prev[1]) / self.dt)
        self.prev = (d, b)
        self.prev_tcp = {s: np.asarray(arms[s].tcp, dtype=float).copy() for s in ARMS} if have_arms else None
        if not have_arms:
            self.arms_unavailable += 1
        mark = press_distance(arms, d) if have_arms else None
        self.max_depth = max(self.max_depth, d)

        if d > L.ZERO_CHEST_MM and b > L.ZERO_BAG_MM:
            self.zero.setdefault("chest_pressed_during_breath", self.steps)

        if self._comp is not None:
            if d > self._comp["depth"]:
                self._comp.update(depth=d, peak_t=t, mark=mark)
            if d < L.COMP_END_MM:
                c = self._comp
                self.compressions.append(Compression(c["start"], t, c["depth"], c["peak_t"], c["recoil"], c["mark"]))
                self._comp = None
                self._recoil_min = d
        elif d >= L.COMP_START_MM:
            self._comp = {"start": t, "depth": d, "peak_t": t, "mark": mark, "recoil": self._recoil_min}
        else:
            self._recoil_min = min(self._recoil_min, d)

        if b > L.BREATH_ON_MM:
            if self._run is None:
                self._run = {"start": t, "depth": -math.inf, "chest_max": d, "hand": True, "by_gripper": True}
            if b > self._run["depth"]:  # the cap's deepest so far: is a gripper the one pressing it?
                self._run.update(depth=b, by_gripper=pressing_bag(arms, b) if have_arms else True)
            self._run["chest_max"] = max(self._run["chest_max"], d)
            if have_arms and not hand_at_mark(arms):
                self._run["hand"] = False
        elif self._run is not None:
            if self._run["depth"] >= L.BREATH_PEAK_MM:  # a light touch that never reached 25 mm is no breath
                r = self._run
                self.breaths.append(Breath(r["start"], t, r["depth"], r["chest_max"], r["hand"], r["by_gripper"]))
            self._run = None

        self.last = self.result(joints, arms)
        return self.last

    def result(self, joints, arms=None) -> Result:
        reading = read_joints(joints)
        if reading is None:
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        d, b = reading
        have_arms = arms_valid(arms)
        clear = clearance(arms, d, b) if have_arms else None
        hands_off = True if clear is None else clear >= L.CLEAR  # unreadable: taken as met, counted
        a = analyse(self.compressions, self.breaths)
        zero = bool(self.zero)
        score = tier(a, zero, hands_off)
        still = abs(self.speed[0]) < L.STILL_MM_S and abs(self.speed[1]) < L.STILL_MM_S
        success = bool(score == 1.0 and still and self._comp is None and self._run is None)
        last = self.compressions[-1] if self.compressions else None
        details = {
            "valid_state": True,
            "t": round(self.t, 3),
            "chest_mm": round(d, 2),
            "bag_mm": round(b, 2),
            "chest_speed_mm_s": round(self.speed[0], 1),
            "bag_speed_mm_s": round(self.speed[1], 1),
            "in_compression": self._comp is not None,
            "in_breath": self._run is not None,
            "zero_rules": sorted(self.zero),
            "n_compressions": len(self.compressions),
            "n_good": a.n_good,
            "good_share": round(a.share_good, 3),
            "cycles": [{"n": len(c), "rate_per_min": _r(cycle_rate(c), 1), "first_peak_s": round(c[0].peak_t, 2),
                        "last_peak_s": round(c[-1].peak_t, 2), "breaths_after": len(a.pause_breaths[k]),
                        "pause_after_s": _r(a.pauses[k])} for k, c in enumerate(a.cycles)],
            "breaths": len(self.breaths),
            "breaths_counted": sum(br.counted for br in self.breaths),
            "breaths_during_cycles": a.breaths_during,
            "breaths_before_first_cycle": a.breaths_before,
            "last_compression": None if last is None else {
                "depth_mm": round(last.depth, 1), "recoil_mm": round(last.recoil, 1),
                "interval_s": _r(a.intervals[-1]), "mark_mm": _mm(last.mark_dist),
                "pressed_at_mark": last.mark_dist is None or last.mark_dist <= L.MARK_TOL, "good": a.good[-1],
                "peak_s": round(last.peak_t, 2)},
            "last_breath": None if not self.breaths else {
                "start_s": round(self.breaths[-1].start, 2), "duration_s": round(self.breaths[-1].duration, 2),
                "depth_mm": round(self.breaths[-1].depth, 1), "chest_max_mm": round(self.breaths[-1].chest_max, 1),
                "hand_at_mark": self.breaths[-1].hand_at_mark, "by_gripper": self.breaths[-1].by_gripper,
                "counted": self.breaths[-1].counted},
            "max_chest_mm": round(self.max_depth, 1),
            "mark_mm": _mm(mark_distance(arms)) if have_arms else None,
            "hand_at_mark": hand_at_mark(arms) if have_arms else None,
            "clear_mm": _mm(clear),
            "hands_off": hands_off,
            "still": still,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
