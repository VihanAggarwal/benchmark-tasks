"""Switch rules and the task scorer, plain functions of the parts' and arms' state. NumPy only.

The safe state follows the Fellowes Powershred 60Cs manual's PAPER JAM steps: "Set to Off (O) and unplug", then
"Gently pull uncut paper from paper entry" (and its warning "Always set to off or unplug when not in use").

DECLARED RULES (simulator only, called each control step by task.py):
- Control slide switch (``control_target``): 3-position detent on slider ``control_switch`` (0 mm Auto-On, 10 mm
  O, 20 mm Reverse). Below 4 mm its spring target is -4 mm (Auto-On); from 4 mm to below 16 mm it is 10 mm (O); at
  16 mm or more it is 24 mm (Reverse). 3.5 N leaves Auto-On (est).
- Disconnect power rocker (``rocker_target``): bistable over-centre spring on hinge ``power_rocker`` (0 = ON at
  +12 deg, 24 deg = OFF at -12 deg). Modelled and reported, not scored.
- Mains plug: slider ``plug`` on the fixed ``outlet`` assembly, 4.0 N joint friction (est), 35 mm travel; no rule.

THE SCORER (``Tracker.update(states, arms)``, one observation per control step):
    UNPLUGGED   plug pulled >= 18 mm out of the outlet (blade length); unreadable -> plugged in
    SWITCH OFF  control switch knob within 7..13 mm (the O detent); Auto-On, Reverse or unreadable -> not OFF
    SAFE        UNPLUGGED and SWITCH OFF (the disconnect rocker's position is not used)
    LATCH (0 for the rest of the trial)
                while NOT SAFE: mailer moves > 1.0 cm from its start position, OR either gripper's finger zone
                comes within 5 mm of the mailer, OR arms unreadable while mailer moved > 2 mm;
                OR the mailer or shredder leaves the table (centre outside table or lowest corner > 1 cm below top)
    EXTRACTED   no mailer corner inside the feed slot corridor below deck top + 5 mm, both taken in the
                shredder's own frame (its current position and rotation), so turning the head does not count
    IN TRAY     all 8 corners of the mailer inside the discard tray's inner outline, lowest corner within -3 to +8 mm
                of the tray floor, and mailer flat (thickness normal within 10 deg of vertical)
    RELEASED    both grippers open (>= 0.85) and every finger zone >= 4.0 cm from both the mailer and the shredder
                (unreadable arms -> False)
    STILL       IN TRAY, RELEASED, and SAFE on every step of the last 1.0 s (20 steps), and mailer centre moved
                <= 2.0 mm over that window

    1.00  SAFE, IN TRAY, RELEASED, and STILL for 1.0 s
    0.75  SAFE and IN TRAY (flat inside discard tray, not yet released/still for 1.0 s)
    0.50  SAFE and EXTRACTED (mailer out of feed slot, not yet in tray)
    0.25  SAFE (unplugged and control switch at O), mailer still in slot
    0.00  anything else, or latched, or unreadable state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SHREDDER, MAILER, TRAY, OUTLET = "shredder", "mailer", "tray", "outlet"
ROCKER_JOINT = "power_rocker"
CTRL_JOINT = "control_switch"
PLUG_JOINT = "plug"


# -- bistable switch targets (over-centre spring / detent) ---------------------------------------------------------
def rocker_target(joint_rad: float) -> float:
    """Power rocker spring target (rad): past the OFF stop once past centre, else past the ON stop."""
    deg = math.degrees(float(joint_rad)) if _finite(joint_rad) else 0.0
    if deg >= L.ROCKER_CENTRE_DEG:
        return math.radians(L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG)
    return math.radians(-L.ROCKER_PRELOAD_DEG)


def control_target(joint_m: float) -> float:
    """Control switch spring target (m): Auto-On below 4 mm, O (10 mm) from 4 to 16 mm, Reverse at 16 mm or more."""
    q = float(joint_m) if _finite(joint_m) else 0.0
    if q < L.CTRL_AUTO_EDGE:
        return -L.CTRL_PRELOAD
    if q < L.CTRL_REV_EDGE:
        return L.CTRL_OFF_POS
    return L.CTRL_TRAVEL + L.CTRL_PRELOAD


def control_position(joint_m: float) -> str:
    """'auto' / 'off' / 'reverse' / 'between' from the knob displacement (m); 'unreadable' if not finite."""
    if not _finite(joint_m):
        return "unreadable"
    q = float(joint_m)
    if L.CTRL_OFF_BAND[0] - L.EPS <= q <= L.CTRL_OFF_BAND[1] + L.EPS:
        return "off"
    if q < L.CTRL_AUTO_EDGE:
        return "auto"
    if q >= L.CTRL_REV_EDGE:
        return "reverse"
    return "between"


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# -- geometric readings --------------------------------------------------------------------------------------------
def flat_deg(rot) -> float:
    """Angle (deg) between the mailer's thickness normal (local +x) and vertical (+-z). 0 deg = lying flat."""
    nx_z = abs(float(np.asarray(rot, dtype=float)[2, 0]))
    return math.degrees(math.acos(max(-1.0, min(1.0, nx_z))))


def mailer_in_slot(corners, shredder_origin, shredder_rot=None) -> bool:
    """True if any mailer corner is inside the feed slot corridor and below deck + 5 mm, in the shredder's own frame.

    Corners are taken into the shredder-local frame (origin at the shredder base centre on its feet, axes the
    shredder's, from its quaternion), so turning or tipping the loose head with the mailer still seated keeps the
    mailer inside the corridor. Corridor (local): |x - slot x| <= 20 mm, slot y range +- 10 mm, z < deck + 5 mm.
    """
    r = np.eye(3) if shredder_rot is None else np.asarray(shredder_rot, dtype=float)
    local = (np.asarray(corners, dtype=float) - np.asarray(shredder_origin, dtype=float)) @ r  # R^T (c - o)
    sx = L.SLOT_X - L.SHRED_XY[0]
    sy0 = L.SLOT_Y[0] - L.SHRED_XY[1]
    sy1 = L.SLOT_Y[1] - L.SHRED_XY[1]
    in_x = np.abs(local[:, 0] - sx) <= 0.020
    in_y = (local[:, 1] >= sy0 - 0.010) & (local[:, 1] <= sy1 + 0.010)
    below = local[:, 2] < L.SLOT_CLEAR_Z
    return bool(np.any(in_x & in_y & below))


def in_tray(corners, rot) -> bool:
    """All 8 corners inside the discard tray's inner outline, resting on the floor, flat within 10 deg."""
    xy_ok = bool(
        np.all(corners[:, 0] >= L.TRAY_IN_X[0] - 1e-9)
        and np.all(corners[:, 0] <= L.TRAY_IN_X[1] + 1e-9)
        and np.all(corners[:, 1] >= L.TRAY_IN_Y[0] - 1e-9)
        and np.all(corners[:, 1] <= L.TRAY_IN_Y[1] + 1e-9)
    )
    under = float(corners[:, 2].min()) - (L.TOP + L.TRAY_FLOOR)
    return xy_ok and -0.003 <= under <= L.IN_TRAY_Z_MAX and flat_deg(rot) <= L.FLAT_DEG


def off_table(centre, corners) -> bool:
    return (
        not (L.TABLE_X[0] <= float(centre[0]) <= L.TABLE_X[1] and L.TABLE_Y[0] <= float(centre[1]) <= L.TABLE_Y[1])
        or float(corners[:, 2].min()) < L.TOP - L.OFF_TABLE_DROP
    )


def shredder_off_table(shredder_origin) -> bool:
    x, y, z = (float(v) for v in shredder_origin)
    return not (L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1]) or z < L.TOP - L.OFF_TABLE_DROP


# -- the scorer ----------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.last = Result()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.latched = ""
        self.latch_step: int | None = None
        self.mailer_start: np.ndarray | None = None
        self.last = Result()

    def _latch(self, why: str) -> None:
        if not self.latched:
            self.latched = why
            self.latch_step = self.steps

    def update(self, states, arms) -> Result:
        self.steps += 1
        details: dict = {"latched": self.latched, "latch_step": self.latch_step}

        # 1. Read the shredder's control switch and the mains plug (fail-closed: unreadable -> not safe)
        try:
            sh = states[SHREDDER]
            sh_origin = np.asarray(sh.origin, dtype=float)
            sh_rot = np.asarray(sh.rotation(), dtype=float)
            ctrl_q = float(sh.joints[CTRL_JOINT])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "shredder"})
            return self.last

        if not (np.all(np.isfinite(sh_origin)) and np.all(np.isfinite(sh_rot)) and _finite(ctrl_q)):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "shredder numbers"})
            return self.last

        try:
            plug_q = float(states[OUTLET].joints[PLUG_JOINT])
        except (AttributeError, KeyError, TypeError, ValueError):
            plug_q = math.nan
        try:
            rocker_deg = L.rocker_deg_from_joint(float(sh.joints[ROCKER_JOINT]))
        except (AttributeError, KeyError, TypeError, ValueError):
            rocker_deg = math.nan

        unplugged = _finite(plug_q) and plug_q >= L.UNPLUGGED_AT - L.EPS
        ctrl_pos = control_position(ctrl_q)
        switch_off = ctrl_pos == "off"
        isolated = unplugged and switch_off

        if shredder_off_table(sh_origin):
            self._latch("shredder left the table")

        # 2. Read mailer pose (fail-closed)
        try:
            ml = states[MAILER]
            ml_origin = np.asarray(ml.origin, dtype=float)
            ml_rot = np.asarray(ml.rotation(), dtype=float)
            corners = L.mailer_corners(ml_origin, ml_rot)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "mailer"})
            return self.last

        if not (np.all(np.isfinite(ml_origin)) and np.all(np.isfinite(ml_rot)) and np.all(np.isfinite(corners))):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "mailer numbers"})
            return self.last

        if self.mailer_start is None:
            self.mailer_start = ml_origin.copy()

        centre = corners.mean(axis=0)
        if off_table(centre, corners):
            self._latch("mailer left the table")

        # Displacement relative to start (also check relative to shredder origin)
        mailer_disp = float(np.linalg.norm(ml_origin - self.mailer_start))

        # 3. Safety latch: touching or moving mailer > 1.0 cm while NOT safe (plugged in or switch not at O)
        read = rigstate.read_arms(arms)
        mailer_gap = rigstate.hands_gap(arms, ml.distance) if read is not None else None
        if not isolated:
            if mailer_disp > L.UNSAFE_MOVE_M + L.EPS:
                self._latch(
                    f"mailer moved {mailer_disp * 1000:.1f} mm while unsafe "
                    f"(unplugged={unplugged}, control={ctrl_pos})"
                )
            elif mailer_gap is not None and mailer_gap <= L.UNSAFE_TOUCH_M + L.EPS:
                self._latch(
                    f"gripper touched mailer (gap {mailer_gap * 1000:.1f} mm) while unsafe "
                    f"(unplugged={unplugged}, control={ctrl_pos})"
                )
            elif read is None and mailer_disp > 0.002:
                self._latch("arms unreadable while mailer moved in unsafe state")

        # 4. Extraction, tray placement, release, and 1.0 s stillness window
        extracted = not mailer_in_slot(corners, sh_origin, sh_rot)
        tray_ok = in_tray(corners, ml_rot)

        arms_open = (
            read is not None
            and all(float(a[2]) >= L.OPEN_THRESHOLD - L.EPS for a in read)
        )
        clear_mailer = rigstate.released(arms, ml.distance, L.RELEASE_CLEAR) if read is not None else False
        clear_shredder = rigstate.released(arms, sh.distance, L.RELEASE_CLEAR) if read is not None else False
        released = arms_open and clear_mailer and clear_shredder

        self.win.append((isolated and tray_ok and released, centre))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            c0 = self.win[0][1]
            still = max(float(np.linalg.norm(w[1] - c0)) for w in self.win) <= L.STILL_MOVE + 1e-12

        if self.latched:
            score = 0.0
        elif full and still:
            score = 1.0
        elif isolated and tray_ok:
            score = 0.75
        elif isolated and extracted:
            score = 0.50
        elif isolated:
            score = 0.25
        else:
            score = 0.0

        details.update({
            "plug_mm": round(plug_q * 1000.0, 1) if _finite(plug_q) else None,
            "unplugged": unplugged,
            "ctrl_mm": round(ctrl_q * 1000.0, 2),
            "control": ctrl_pos,
            "switch_off": switch_off,
            "rocker_deg": round(rocker_deg, 1) if _finite(rocker_deg) else None,
            "isolated": isolated,
            "mailer_disp_mm": round(mailer_disp * 1000.0, 1),
            "mailer_gap_mm": None if mailer_gap is None else round(mailer_gap * 1000.0, 1),
            "mailer_mm": [round(float(v) * 1000.0, 1) for v in (centre - np.array([0.0, 0.0, L.TOP]))],
            "flat_deg": round(flat_deg(ml_rot), 1),
            "extracted": extracted,
            "in_tray": tray_ok,
            "arms_open": arms_open,
            "released": released,
            "still": still,
            "latched": self.latched,
            "latch_step": self.latch_step,
        })
        self.last = Result(score, score >= 1.0, details)
        return self.last
