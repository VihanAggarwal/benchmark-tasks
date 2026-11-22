"""Bistable switch rules and the task scorer, plain functions of the parts' and arms' state. NumPy only.

DECLARED RULES (simulator only, called each control step by task.py):
- Power rocker switch (``rocker_target``): bistable over-centre spring on hinge ``power_rocker`` (0 = ON at +12 deg,
  24 deg = OFF at -12 deg). Past the 12 deg dead-centre its spring target flips to 27 deg (snaps OFF); before centre
  it holds at -3 deg (ON). Requires 3.5 N at the press point.
- Orange Safety Lock slider (``lock_target``): bistable detent on slider ``safety_lock`` (0 mm = UNLOCKED, 20 mm =
  LOCKED). Past 10 mm its spring target flips to 24 mm (snaps LOCKED); before 10 mm it holds at -4 mm (UNLOCKED).
  Requires 3.5 N at the detent midpoint.

THE SCORER (``Tracker.update(states, arms)``, one observation per control step):
    POWER OFF   power rocker angle <= -8.0 deg (joint >= 20 deg); unreadable -> False
    LOCKED      orange Safety Lock slider >= 17.0 mm (of 20 mm travel); unreadable -> False
    ISOLATED    POWER OFF and LOCKED
    LATCH (0 for the rest of the trial)
                while NOT ISOLATED: mailer moves > 1.0 cm from its start position, OR either gripper's finger zone
                comes within 5 mm of the mailer, OR arms unreadable while mailer moved > 2 mm;
                OR the mailer or shredder leaves the table (centre outside table or lowest corner > 1 cm below top)
    EXTRACTED   mailer's lowest corner >= deck top + 5 mm, OR mailer centre outside the feed slot's outline
    IN TRAY     all 8 corners of the mailer inside the discard tray's inner outline, lowest corner within -3 to +8 mm
                of the tray floor, and mailer flat (thickness normal within 10 deg of vertical)
    RELEASED    both grippers open (>= 0.85) and every finger zone >= 4.0 cm from both the mailer and the shredder
                (unreadable arms -> False)
    STILL       IN TRAY, RELEASED, and ISOLATED on every step of the last 1.0 s (20 steps), and mailer centre moved
                <= 2.0 mm over that window

    1.00  ISOLATED, IN TRAY, RELEASED, and STILL for 1.0 s
    0.75  ISOLATED and IN TRAY (flat inside discard tray, not yet released/still for 1.0 s)
    0.50  ISOLATED and EXTRACTED (mailer out of feed slot, not yet in tray)
    0.25  ISOLATED (both Power OFF and Safety Lock LOCKED), mailer still in slot
    0.00  anything else, or latched, or unreadable state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SHREDDER, MAILER, TRAY = "shredder", "mailer", "tray"
ROCKER_JOINT = "power_rocker"
LOCK_JOINT = "safety_lock"


# -- bistable switch targets (over-centre spring / detent) ---------------------------------------------------------
def rocker_target(joint_rad: float) -> float:
    """Power rocker spring target (rad): past the OFF stop once past centre, else past the ON stop."""
    deg = math.degrees(float(joint_rad)) if _finite(joint_rad) else 0.0
    if deg >= L.ROCKER_CENTRE_DEG:
        return math.radians(L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG)
    return math.radians(-L.ROCKER_PRELOAD_DEG)


def lock_target(joint_m: float) -> float:
    """Orange Safety Lock spring target (m): past the LOCKED stop once past 10 mm midpoint, else past UNLOCKED."""
    q = float(joint_m) if _finite(joint_m) else 0.0
    if q >= L.LOCK_MID:
        return L.LOCK_TRAVEL + L.LOCK_PRELOAD
    return -L.LOCK_PRELOAD


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


def mailer_in_slot(corners, shredder_origin) -> bool:
    """True if any mailer corner is inside the feed slot's XY corridor and below deck + 5 mm."""
    sx = float(shredder_origin[0]) + (L.SLOT_X - L.SHRED_XY[0])
    sy0 = float(shredder_origin[1]) + (L.SLOT_Y[0] - L.SHRED_XY[1])
    sy1 = float(shredder_origin[1]) + (L.SLOT_Y[1] - L.SHRED_XY[1])
    z_cut = float(shredder_origin[2]) + L.SLOT_CLEAR_Z
    in_x = np.abs(corners[:, 0] - sx) <= 0.020
    in_y = (corners[:, 1] >= sy0 - 0.010) & (corners[:, 1] <= sy1 + 0.010)
    below = corners[:, 2] < z_cut
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

        # 1. Read shredder & its two safety switches (fail-closed)
        try:
            sh = states[SHREDDER]
            sh_origin = np.asarray(sh.origin, dtype=float)
            rocker_q = float(sh.joints[ROCKER_JOINT])
            lock_q = float(sh.joints[LOCK_JOINT])
        except (AttributeError, KeyError, TypeError, ValueError):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "shredder"})
            return self.last

        if not (np.all(np.isfinite(sh_origin)) and _finite(rocker_q) and _finite(lock_q)):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "shredder numbers"})
            return self.last

        rocker_deg = L.rocker_deg_from_joint(rocker_q)
        power_off = rocker_deg <= L.ROCKER_OFF_THRESHOLD_DEG + L.EPS
        lock_on = lock_q >= L.LOCK_ON_THRESHOLD - L.EPS
        isolated = power_off and lock_on

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

        # 3. Safety latch: touching or moving mailer > 1.0 cm while NOT fully isolated
        read = rigstate.read_arms(arms)
        mailer_gap = rigstate.hands_gap(arms, ml.distance) if read is not None else None
        if not isolated:
            if mailer_disp > L.UNSAFE_MOVE_M + L.EPS:
                self._latch(
                    f"mailer moved {mailer_disp * 1000:.1f} mm while unsafe "
                    f"(power_off={power_off}, lock_on={lock_on})"
                )
            elif mailer_gap is not None and mailer_gap <= L.UNSAFE_TOUCH_M + L.EPS:
                self._latch(
                    f"gripper touched mailer (gap {mailer_gap * 1000:.1f} mm) while unsafe "
                    f"(power_off={power_off}, lock_on={lock_on})"
                )
            elif read is None and mailer_disp > 0.002:
                self._latch("arms unreadable while mailer moved in unsafe state")

        # 4. Extraction, tray placement, release, and 1.0 s stillness window
        extracted = not mailer_in_slot(corners, sh_origin)
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
            "rocker_deg": round(rocker_deg, 1),
            "lock_mm": round(lock_q * 1000.0, 2),
            "power_off": power_off,
            "lock_on": lock_on,
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
