"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
stamp (a loose mechkit mechanism: the base, and the handle on a sprung slider down over it), the delivery note (a loose
compound: an A4 sheet on its card backing, the box printed on it) and both arms through rigstate.

    PRINT       the stamp prints once its handle is within 2 mm of the bottom of its 22 mm travel while its base lies
                flat on the sheet (sole within 1.5 mm of the sheet's top, within 3 deg of the sheet's plane, its middle
                over the sheet), held 4 steps (0.2 s): the die has flipped down onto the paper. One press prints once;
                the handle has to come back up past 10 mm of travel before another press can print. Recorded: where the
                die's 47 x 18 mm footprint lay on the sheet, its angle to the box, which way up, and how far the sheet
                moved from the press's start to its end
    IN_BOX      every point of the footprint inside the box's 30 x 60 mm inside
    SQUARE      the footprint within 5 deg of square to the box, the right way up (the stamp's labelled front toward the
                sheet's lower edge: RECEIVED reads left to right for the robot)
    HOME        the stamp standing upright (5 deg) within 30 mm of its spot, both grippers' finger zones 30 mm or more
                from it on each of the last 20 steps, and it still (2 mm)
    FAILED      latched: the stamp tipped past 45 deg at any moment (dropped)

    1.0   one print, IN_BOX, SQUARE, the sheet moved 5 mm or less during it, and HOME             (success)
    0.75  that print, the stamp not (yet) back
    0.5   one print with at least half the footprint inside the box, within 15 deg of square, the right way up
    0.25  one print on the sheet, the right way up
    0     no print, two or more prints, a print upside down, and every FAILED trial
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

STAMP, SHEET, IMPRINT = "stamp", "sheet", "imprint"
HANDLE = "handle"
FOOT = L.footprint()


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


@dataclass(frozen=True)
class Reading:
    stamp_origin: np.ndarray
    stamp_rot: np.ndarray
    travel: float
    sheet_origin: np.ndarray
    sheet_rot: np.ndarray


def read(states) -> Reading | None:
    try:
        st, sh = states[STAMP], states[SHEET]
        r = Reading(np.asarray(st.origin, dtype=float).ravel(), np.asarray(st.rotation(), dtype=float),
                    float(st.joints[HANDLE]), np.asarray(sh.origin, dtype=float).ravel(),
                    np.asarray(sh.rotation(), dtype=float))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    arrays = (r.stamp_origin, r.stamp_rot, r.sheet_origin, r.sheet_rot)
    if any(not np.all(np.isfinite(a)) for a in arrays) or not math.isfinite(r.travel):
        return None
    if r.stamp_origin.size != 3 or r.sheet_origin.size != 3:
        return None
    return r


def to_sheet(r: Reading, pts_stamp: np.ndarray) -> np.ndarray:
    """Stamp-frame (x, y) points on the sole -> sheet-frame (x, y)."""
    p = np.c_[pts_stamp, np.zeros(len(pts_stamp))]
    world = r.stamp_origin + p @ r.stamp_rot.T
    return ((world - r.sheet_origin) @ r.sheet_rot)[:, :2]


def rel_yaw_deg(r: Reading) -> float:
    """The stamp's x axis (its front points along -x) in the sheet frame, as an angle: 0 = square and the right way
    up, +-180 = upside down."""
    a = r.sheet_rot.T @ r.stamp_rot[:, 0]
    return math.degrees(math.atan2(a[1], a[0]))


def sole_gap(r: Reading) -> float:
    sheet_top = r.sheet_origin + r.sheet_rot[:, 2] * L.SHEET[2]
    return float(np.dot(r.stamp_origin - sheet_top, r.sheet_rot[:, 2]))


def flat_on_sheet(r: Reading) -> bool:
    c = float(np.dot(r.stamp_rot[:, 2], r.sheet_rot[:, 2]))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    mid = to_sheet(r, np.zeros((1, 2)))[0]
    over = abs(mid[0]) <= L.SHEET[0] / 2 and abs(mid[1]) <= L.SHEET[1] / 2
    return bool(angle <= L.FLAT_DEG + L.EPS and abs(sole_gap(r)) <= L.ON_SHEET + L.EPS and over)


def in_box_fraction(pts_sheet: np.ndarray) -> float:
    d = np.abs(pts_sheet - np.asarray(L.BOX_CENTRE))
    inside = (d[:, 0] <= L.BOX[0] / 2 + L.EPS) & (d[:, 1] <= L.BOX[1] / 2 + L.EPS)
    return float(np.mean(inside))


@dataclass
class Print:
    step: int
    centre: np.ndarray  # sheet frame (x, y)
    yaw_deg: float  # rel_yaw_deg at the print
    in_box: float  # fraction of the footprint inside the box
    corners_in: bool
    sheet_slip: float = 0.0  # m, the sheet's travel from the press's start to its end
    on_sheet: bool = True

    @property
    def upright(self) -> bool:
        return abs(self.yaw_deg) < 90.0


def tier(prints: list, home: bool, failed: bool) -> float:
    if failed or len(prints) != 1:
        return 0.0
    p = prints[0]
    if not p.upright:
        return 0.0
    if p.corners_in and abs(p.yaw_deg) <= L.SQUARE_OK + L.EPS and p.sheet_slip <= L.SHEET_SLIP + L.EPS:
        return 1.0 if home else 0.75
    if p.in_box >= 0.5 and abs(p.yaw_deg) <= L.SQUARE_HALF + L.EPS:
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.failed: str | None = None
        self.prints: list[Print] = []
        self.armed = True
        self.pressed = 0  # consecutive steps bottomed and flat
        self.press_sheet0: np.ndarray | None = None  # the sheet's origin when this press first bottomed
        self.hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    def _press(self, r: Reading) -> None:
        bottomed = r.travel >= L.BOTTOMED - L.EPS
        if r.travel <= L.REARM:
            self.armed = True
        if bottomed and flat_on_sheet(r):
            if self.pressed == 0:
                self.press_sheet0 = r.sheet_origin.copy()
            self.pressed += 1
            if self.armed and self.pressed >= L.PRINT_STEPS:
                pts = to_sheet(r, FOOT)
                cs = to_sheet(r, L.corners())
                self.prints.append(Print(self.steps, to_sheet(r, np.zeros((1, 2)))[0], rel_yaw_deg(r),
                                         in_box_fraction(pts), in_box_fraction(cs) == 1.0))
                self.armed = False
            if self.prints and not self.armed and self.press_sheet0 is not None:
                slip = float(np.linalg.norm(r.sheet_origin[:2] - self.press_sheet0[:2]))
                self.prints[-1].sheet_slip = max(self.prints[-1].sheet_slip, slip)
        else:
            self.pressed = 0

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.ok.clear()
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.failed is None and tilt_deg(r.stamp_rot) > L.DROP_DEG:
            self.failed = "the stamp tipped over (dropped)"
        self._press(r)
        self.hist.append(r.stamp_origin.copy())
        x, y, _ = L.stamp_start()
        at_home = (math.hypot(r.stamp_origin[0] - x, r.stamp_origin[1] - y) <= L.HOME_OK + L.EPS
                   and tilt_deg(r.stamp_rot) <= L.UPRIGHT_DEG + L.EPS and abs(r.stamp_origin[2] - L.TOP) <= 0.005)
        try:
            clear = bool(rigstate.released(arms, states[STAMP].distance, L.LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not clear
            clear = False
        self.ok.append(bool(at_home and clear))
        m = self.motion()
        home = bool(len(self.ok) == self.window and all(self.ok) and m is not None and m <= L.STILL_MOVE + L.EPS)
        level = tier(self.prints, home, self.failed is not None)
        p = self.prints[0] if self.prints else None
        details = {
            "valid_state": True, "score": level, "prints": len(self.prints), "travel_mm": _mm(r.travel),
            "flat_on_sheet": flat_on_sheet(r), "sole_gap_mm": _mm(sole_gap(r)),
            "print_centre_mm": None if p is None else [round(float(v) * 1000, 2) for v in p.centre],
            "print_yaw_deg": None if p is None else round(p.yaw_deg, 2),
            "print_in_box": None if p is None else round(p.in_box, 3),
            "print_corners_in": None if p is None else p.corners_in,
            "sheet_slip_mm": None if p is None else _mm(p.sheet_slip),
            "at_home": at_home, "clear_now": clear, "home_20": home, "failed": self.failed,
            "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last

    def motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(p - h[0])) for p in h[1:])
