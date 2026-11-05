"""The scorer, as plain functions of the two pads' states and the arms' state. NumPy only; no simulator needed.

Two-rescuer CPR with an AED is about order and timing over the whole trial, not only its end: compressions start at
once and go on while the pads are put on; everyone is clear while the AED analyses and while the shock button is
pressed; the pause around the shock is short and compressions restart at once. So the scorer keeps a small record of
the trial (the compressions so far, when each pad arrived on its site, the AED's state, the shock) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(pad_a, pad_b, arms) -> Result(score, success, details)

``pad_a`` and ``pad_b`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read. On such a step no compression and no
button press can be seen, the area counts as clear (the rules about the arms are taken as kept) and the step is
counted in ``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is a stack of discs along the 38 mm from the point between its finger tips back up the gripper, each of a
radius reaching a finger pad's outer corner (half the current finger gap plus the 8 mm pad thickness, and half the
pad's 27 mm width across). Nothing of the fingers is below the tips.

The AED, as the scorer models it:
  * it analyses once both pads are on their sites and both grippers are clear; 3 s of that without a break completes
    the analysis and charges it; any break (a gripper back at the patient, a pad off its site) restarts it;
  * once charged, it stays charged until the shock, until a pad leaves its site, or until a compression is counted;
  * a press is a closed gripper's tips on the button; a press that began after the AED was charged and is held 0.5 s
    delivers the shock (a press already under way when the AED charges does nothing);
  * it delivers one shock per trial; after it the AED does not analyse again (the next analysis would be two minutes
    later, beyond the run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
SIDES = ("left", "right")

# The manikin's two fixed boxes (torso, head) as world-aligned (lo, hi).
TORSO_BOX = (np.array([L.TORSO_X - L.TORSO_SIZE[0] / 2, L.TORSO_Y - L.TORSO_SIZE[1] / 2, L.TOP]),
             np.array([L.TORSO_X + L.TORSO_SIZE[0] / 2, L.TORSO_Y + L.TORSO_SIZE[1] / 2, L.TORSO_TOP]))
HEAD_BOX = (np.array([L.HEAD_X - L.HEAD_SIZE[0] / 2, L.HEAD_Y - L.HEAD_SIZE[1] / 2, L.TOP]),
            np.array([L.HEAD_X + L.HEAD_SIZE[0] / 2, L.HEAD_Y + L.HEAD_SIZE[1] / 2, L.HEAD_TOP]))
MANIKIN = (TORSO_BOX, HEAD_BOX)
CPR_FOOT = (np.array([L.CPR_X - L.CPR_MARK / 2, L.CPR_Y - L.CPR_MARK / 2]),
            np.array([L.CPR_X + L.CPR_MARK / 2, L.CPR_Y + L.CPR_MARK / 2]))


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
        for side in SIDES:
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
    """Angle between the world's vertical and the pad's thin (24 mm) axis, through its two 40 x 40 mm faces
    (degrees, 0..90): either face down counts as flat."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def _hdist_to_rect(p, lo, hi, grow: float = 0.0) -> float:
    """Horizontal distance from point ``p`` to the rectangle [lo, hi] (x, y) grown by ``grow`` (0 inside it)."""
    dx = max(0.0, (lo[0] - grow) - p[0], p[0] - (hi[0] + grow))
    dy = max(0.0, (lo[1] - grow) - p[1], p[1] - (hi[1] + grow))
    return float(math.hypot(dx, dy))


def _hdist(p, x, y) -> float:
    return float(math.hypot(float(p[0]) - x, float(p[1]) - y))


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across (57 mm fully open, 21 mm closed to 16 mm)."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_axis(arm, samples: int = 20) -> np.ndarray:
    """Points along the fingers' axis, from the point between the tips back up the gripper (FINGER_LEN)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u)


def at_patient(arm) -> bool:
    """Some part of the gripper's finger zone is inside the clear-zone box of the torso or the head: the box's outline
    seen from above grown by 50 mm on every side, from the table up to 50 mm above the box's top."""
    r = zone_radius(arm)
    for lo, hi in MANIKIN:
        top = float(hi[2]) + L.CLEAR_Z
        for p in finger_axis(arm):
            if float(p[2]) < top and _hdist_to_rect(p, lo, hi, L.CLEAR_XY) < r:
                return True
    return False


def clear_margin(arm) -> float:
    """How far the gripper's finger zone is outside the clear-zone boxes (m; negative: inside). For the trace."""
    r = zone_radius(arm)
    best = math.inf
    for lo, hi in MANIKIN:
        top = float(hi[2]) + L.CLEAR_Z
        for p in finger_axis(arm):
            best = min(best, max(float(p[2]) - top, _hdist_to_rect(p, lo, hi, L.CLEAR_XY) - r))
    return best


def area_clear(arms) -> bool:
    return not (at_patient(arms["left"]) or at_patient(arms["right"]))


def closed(arm) -> bool:
    """Finger gap 30 mm or less."""
    return finger_gap(arm) <= L.CLOSED_GAP + 1e-9


def tips_above_mark(arm) -> float:
    """Height of the point between the finger tips above the compression mark's top."""
    return float(np.asarray(arm.tips, dtype=float)[2]) - L.MARK_TOP


def over_mark(arm) -> bool:
    return _hdist(arm.tips, L.CPR_X, L.CPR_Y) <= L.ON_CHEST_XY


def on_chest(arm) -> bool:
    """Closed, its tip point within 20 mm of the compression mark's centre horizontally and at most 45 mm above it."""
    return bool(closed(arm) and over_mark(arm) and tips_above_mark(arm) <= L.ON_CHEST_HIGH)


def on_button(arm) -> bool:
    """Pressing the shock button: closed, its tip point within 15 mm of the button's centre horizontally and at most
    6 mm above the button's top."""
    tips = np.asarray(arm.tips, dtype=float)
    return bool(closed(arm) and _hdist(tips, L.AED_X, L.AED_Y) <= L.PRESS_XY
                and float(tips[2]) - L.BUTTON_TOP <= L.PRESS_Z)


# -- the pads ---------------------------------------------------------------------------------------------------
def site_of(pad) -> str | None:
    """The site the pad is on, or None: flat (tilted under 10 degrees), its lowest point within 3 mm of the chest's
    top, its centre within 20 mm of the site's centre horizontally."""
    if tilt_deg(pad.quat_xyzw) >= L.FLAT_DEG or abs(float(pad.low[2]) - L.TORSO_TOP) > L.ON_SITE_Z:
        return None
    for name, (x, y) in L.SITES.items():
        if _hdist(pad.pos, x, y) <= L.ON_SITE_XY:
            return name
    return None


def both_on(pad_a, pad_b) -> bool:
    """One pad on each site (the pads are identical; either may go on either site)."""
    return {site_of(pad_a), site_of(pad_b)} == {"sternal", "apex"}


def on_compression_point(pad) -> bool:
    """The pad resting on the chest (lowest point within 5 mm of its top) with its outline seen from above (its
    world-aligned bounding box) overlapping the 30 x 30 mm compression mark."""
    lo, hi = np.asarray(pad.low, dtype=float), np.asarray(pad.high, dtype=float)
    a, b = CPR_FOOT
    overlap = lo[0] < b[0] and a[0] < hi[0] and lo[1] < b[1] and a[1] < hi[1]
    return bool(overlap and float(lo[2]) <= L.TORSO_TOP + L.RESTING_Z)


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "pressed_while_not_clear": "a press of the shock button began while a gripper was at the patient",
    "touched_during_shock": "a gripper was at the patient while the button was pressed on the charged AED",
    "pad_on_compression_point": "a pad rested on the chest over the compression mark",
    "off_table": "a pad fell off the table",
}

STARTS = ((L.PAD_A_X, L.PAD_A_Y), (L.PAD_B_X, L.PAD_B_Y))


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.gap_steps = int(round(L.CPR_GAP_S * control_hz))  # 30: compressions at most 1.5 s apart
        self.analysis_steps = int(round(L.ANALYSIS_S * control_hz))  # 60
        self.press_steps = int(round(L.PRESS_S * control_hz))  # 10
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        # compressions
        self.armed = False  # the right tips have been 30 mm up over the mark since the last compression
        self.compressions: list[int] = []  # control steps at which a compression was counted
        self.in_progress = False
        self.min_tip_mm = math.inf  # the right gripper's lowest tip height over the mark while closed over it
        # pads
        self.site_now = {"pad_a": None, "pad_b": None}
        self.arrived_in_cpr = {"pad_a": False, "pad_b": False}  # at its latest arrival on a site
        self.ever_both_on = False
        # the AED
        self.analysis_run = 0  # control steps in a row with both pads on and the area clear (while not charged)
        self.charged = False
        self.charged_qualified = False  # the charge came from an analysis with both pads put on during compressions
        self.charges = 0
        self.qualified_analysis = False
        self.press_run = 0  # control steps in a row with a gripper on the button
        self.press_live = False  # the current press began while the AED was charged
        self.shock_step: int | None = None
        self.shock_qualified = False
        self.pre_pause_s: float | None = None
        self.max_pause_steps = 0  # the longest pause in compressions before the shock: from the start to the first
        #                           compression, between compressions, and from the last one to the shock
        self.first_post: int | None = None  # the first compression after the shock
        self.run_end: int | None = None  # the last compression of the unbroken run that began there
        self.run_open = False
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of pad_a, of pad_b] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @property
    def first_compression_s(self) -> float | None:
        return (self.compressions[0] - 1) * self.dt if self.compressions else None

    @property
    def post_pause_s(self) -> float | None:
        if self.shock_step is None or self.first_post is None:
            return None
        return (self.first_post - self.shock_step) * self.dt

    @property
    def post_run_s(self) -> float:
        if self.first_post is None or self.run_end is None:
            return 0.0
        return (self.run_end - self.first_post) * self.dt

    @staticmethod
    def at_start(pad_a, pad_b) -> bool:
        """The starting layout: each pad flat on the table within START_TOL of its starting place, still."""
        for s, (x, y) in zip((pad_a, pad_b), STARTS):
            if not (_hdist(s.pos, x, y) < L.START_TOL and abs(float(s.low[2]) - L.TOP) < 0.003
                    and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some pad moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could have
        carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    # -- one control step ----------------------------------------------------------------------------------------
    def _compressions(self, right) -> bool:
        """Count a compression if the right gripper's tips, closed over the mark, came down from 30 mm or more to
        15 mm or less. Returns whether one was counted at this step."""
        if not (closed(right) and over_mark(right)):
            self.armed = False
            return False
        h = tips_above_mark(right)
        self.min_tip_mm = min(self.min_tip_mm, h * 1000)
        if h >= L.STROKE_UP:
            self.armed = True
        if self.armed and h <= L.STROKE_DOWN:
            self.armed = False
            self.compressions.append(self.steps)
            return True
        return False

    def _pauses(self, step: int) -> None:
        """A compression was counted at ``step``: before the shock, record the pause that it ends."""
        if self.shock_step is not None:
            return
        prev = self.compressions[-2] if len(self.compressions) > 1 else 1  # the first: from the trial's first step
        self.max_pause_steps = max(self.max_pause_steps, step - prev)

    def _post_shock(self, step: int) -> None:
        if self.shock_step is None or step <= self.shock_step:
            return
        if self.first_post is None:
            self.first_post = self.run_end = step
            self.run_open = True
        elif self.run_open:
            prev = self.compressions[-2]
            if step - prev <= self.gap_steps:
                self.run_end = step
            else:
                self.run_open = False

    def update(self, pad_a, pad_b, arms=None) -> Result:
        pads = {"pad_a": pad_a, "pad_b": pad_b}
        if not (valid(pad_a) and valid(pad_b)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(pad_a, pad_b, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (pad_a, pad_b)]
        if self.prev is not None and self._teleported(now) and self.at_start(pad_a, pad_b):
            self.reset()  # the simulator put both pads back: a new episode (rigstate's on_reset hook normally does
            #               this first; this is the fallback)
        self.prev = now
        self.steps += 1

        have_arms = arms_valid(arms)
        compressed = False
        if have_arms:
            clear = area_clear(arms)
            compressed = self._compressions(arms["right"])
            if compressed:
                self._pauses(self.steps)
                self._post_shock(self.steps)
            last = self.compressions[-1] if self.compressions else None
            self.in_progress = bool(on_chest(arms["right"]) and last is not None
                                    and self.steps - last <= self.gap_steps)
            pressing = on_button(arms["left"]) or on_button(arms["right"])
        else:
            self.arms_unavailable += 1
            clear, pressing = True, False
            self.armed = self.in_progress = False

        # the pads
        for name, s in pads.items():
            site = site_of(s)
            if site is not None and site != self.site_now[name]:
                self.arrived_in_cpr[name] = self.in_progress
            self.site_now[name] = site
            if on_compression_point(s):
                self._flag("pad_on_compression_point")
            if off_table(s):
                self._flag("off_table")
        on = both_on(pad_a, pad_b)
        self.ever_both_on = self.ever_both_on or on
        qualified = on and all(self.arrived_in_cpr.values())

        # the shock button (before this step's analysis: a press that starts as the AED charges is not live)
        if pressing:
            if self.press_run == 0:
                self.press_live = self.charged
                if not clear:
                    self._flag("pressed_while_not_clear")
            self.press_run += 1
            if self.press_live and self.charged:
                if not clear:
                    self._flag("touched_during_shock")
                if self.press_run >= self.press_steps:
                    self.shock_step = self.steps
                    self.shock_qualified = self.charged_qualified
                    last = [c for c in self.compressions if c < self.steps]
                    self.pre_pause_s = (self.steps - last[-1]) * self.dt if last else None
                    if last:
                        self.max_pause_steps = max(self.max_pause_steps, self.steps - last[-1])
                    self.charged = False
                    self.press_live = False
        else:
            self.press_run = 0
            self.press_live = False

        # the AED's charge is cancelled when a pad leaves its site or compressions resume
        if self.charged and (not on or compressed):
            self.charged = False
        # the analysis: 3 s of both pads on and the area clear, without a break (one shock per trial)
        if not self.charged and self.shock_step is None:
            self.analysis_run = self.analysis_run + 1 if (on and clear) else 0
            if self.analysis_run >= self.analysis_steps:
                self.charged = True
                self.charged_qualified = qualified
                self.qualified_analysis = self.qualified_analysis or qualified
                self.charges += 1
                self.analysis_run = 0
        else:
            self.analysis_run = 0

        self.last = self.result(pad_a, pad_b, arms)
        return self.last

    @property
    def max_pause_s(self) -> float | None:
        """The longest pause in compressions before the shock (None before the shock, or with no compression)."""
        if self.shock_step is None or self.pre_pause_s is None:
            return None
        return self.max_pause_steps * self.dt

    def timing(self) -> dict:
        """The three timing conditions of the 1.0 tier (each True or False)."""
        pause = self.max_pause_s
        post = self.post_pause_s
        return {
            "pauses_ok": pause is not None and pause <= L.MAX_PAUSE_S + 1e-9,
            "post_shock_ok": post is not None and post <= L.POST_SHOCK_S + 1e-9,
            "kept_up_ok": self.post_run_s >= L.POST_RUN_S - 1e-9,
        }

    def result(self, pad_a, pad_b, arms=None) -> Result:
        if not (valid(pad_a) and valid(pad_b)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        left_off = (not at_patient(arms["left"])) if have_arms else True  # unreadable: taken as met, counted
        on = both_on(pad_a, pad_b)
        timing = self.timing()
        if self.breach:
            score = 0.0
        elif self.shock_step is not None and self.shock_qualified:
            score = 1.0 if (all(timing.values()) and on and left_off) else 0.75
        elif self.qualified_analysis:
            score = 0.5
        elif self.ever_both_on:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(pad_a) and is_resting(pad_b))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "compressions": len(self.compressions),
            "first_compression_s": None if self.first_compression_s is None else round(self.first_compression_s, 2),
            "last_compression_step": self.compressions[-1] if self.compressions else None,
            "in_progress": self.in_progress,
            "min_tip_mm": None if not math.isfinite(self.min_tip_mm) else round(self.min_tip_mm, 1),
            "sites": dict(self.site_now),
            "arrived_in_cpr": dict(self.arrived_in_cpr),
            "both_on": on,
            "ever_both_on": self.ever_both_on,
            "analysis_s": round(self.analysis_run * self.dt, 2),
            "charged": self.charged,
            "charges": self.charges,
            "qualified_analysis": self.qualified_analysis,
            "press_steps": self.press_run,
            "shock_step": self.shock_step,
            "shock_qualified": self.shock_qualified,
            "pre_shock_pause_s": None if self.pre_pause_s is None else round(self.pre_pause_s, 2),
            "max_pause_s": None if self.max_pause_s is None else round(self.max_pause_s, 2),
            "post_shock_pause_s": None if self.post_pause_s is None else round(self.post_pause_s, 2),
            "post_shock_run_s": round(self.post_run_s, 2),
            **timing,
            "left_off_manikin": left_off if have_arms else None,
            "area_clear": area_clear(arms) if have_arms else None,
            "clear_margin_mm": ({s: _mm(clear_margin(arms[s])) for s in SIDES} if have_arms else None),
            "right_tip_above_mark_mm": _mm(tips_above_mark(arms["right"])) if have_arms else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "pad_tilt_deg": [round(tilt_deg(p.quat_xyzw), 2) for p in (pad_a, pad_b)],
            "pad_low_mm": [_mm(float(p.low[2]) - L.TORSO_TOP) for p in (pad_a, pad_b)],
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "pads_resting": bool(is_resting(pad_a) and is_resting(pad_b)),
        }
        return Result(score, success, details)
