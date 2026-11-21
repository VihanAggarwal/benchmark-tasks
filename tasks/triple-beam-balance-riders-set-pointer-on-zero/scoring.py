"""The scorer, as plain functions of the balance's state, the specimen's state and the arms. NumPy only.

``Tracker.update(bal, rock, arms)`` -> Result(score, success, details), fed one observation per control step. ``bal``
is the mechkit AssemblyState of the balance (joints "beam" (rad about +x, positive lifts the pan side), "r100", "r10",
"poise" (m the weight has moved to the right of its zero notch)); ``rock`` the AssemblyState of the specimen (frame
origin on the table under its centre). The arms are read only for the let-go and hands-clear checks.

    ON PAN      the specimen's centre within 66 mm of the pan's axis and 3 to 45 mm over the plate (measured in the
                beam's own frame, so it follows the pan up and down)
    NOTCHES     the 100 g rider within 4 mm of notch 2 (200 g) and the 10 g rider within 4 mm of notch 3 (30 g)
    READING     100 x (100 g rider's notch) + 10 x (10 g rider's notch) + the poise's reading, in grams, from the three
                weights' positions (the poise's 10 g spans 100 mm: 1 g = 10 mm)
    POINTER     the beam's angle from zero, in degrees (the scale post has a mark for every degree)
    AT REST     over the last 1 s the beam turned by 0.5 degree or less
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the specimen, which moved 5 mm or
                less over that second (still)
    HANDS CLEAR on every step of the last 1 s both finger zones are 30 mm or more from the beam, the riders and the poise
    FELL        latched: the specimen's lowest point ever more than 20 mm under the table top

    0      FELL, or the specimen not released and still, or not on the pan
    1.0    on the pan; both riders in their notches; reading within 0.5 g of the specimen's 236.4 g; pointer within
           1 degree of zero; beam at rest; hands clear of the instrument
    0.75   on the pan; both riders in their notches; reading within 2 g; pointer within 3 degrees; beam at rest
    0.5    on the pan; both riders in their notches
    0.25   on the pan
    (the rules are tested in this order: the first that holds sets the score)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BALANCE = "balance"
ROCK = "rock"
BEAM, R100, R10, POISE = "beam", "r100", "r10", "poise"
MOVING = (BEAM, R100, R10, POISE)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def joint(bal, name: str) -> float | None:
    try:
        v = float(bal.joints[name])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def reading_g(q100: float, q10: float, qp: float) -> float:
    """What the three beams read, in grams, from the weights' positions."""
    return 100.0 * q100 / L.R100_STEP + 10.0 * q10 / L.R10_STEP + 10.0 * qp / L.POISE_SPAN


def notches_ok(q100: float, q10: float) -> bool:
    return bool(abs(q100 - L.K100_ANS * L.R100_STEP) <= L.NOTCH_TOL + L.EPS
                and abs(q10 - L.K10_ANS * L.R10_STEP) <= L.NOTCH_TOL + L.EPS)


def specimen_in_beam_frame(bal, rock) -> np.ndarray | None:
    """The specimen's centre in the beam's own frame, as drawn at rest (the balance's base frame, beam at 0)."""
    try:
        r, p = bal.link_pose(BEAM)
        off = bal.spec.body_offset(BEAM)
        c = np.asarray(rock.pos, dtype=float)
        loc = (c - p) @ r + off
        return loc if np.all(np.isfinite(loc)) else None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def on_pan(loc: np.ndarray | None) -> bool:
    if loc is None:
        return False
    d = math.hypot(float(loc[0]), float(loc[1]) - L.LA)
    h = float(loc[2]) - L.PAN_Z
    return bool(d <= L.ON_PAN_R + L.EPS and 0.003 <= h <= 0.045)


def fell(rock) -> bool:
    try:
        return bool(float(rock.low[2]) - L.TOP < -L.FALL_DROP)
    except (AttributeError, TypeError, ValueError):
        return True


def tier(latched: bool, released_still: bool, pan: bool, notches: bool, mid: bool, exact: bool) -> float:
    if latched or not released_still or not pan:
        return 0.0
    if not notches:
        return 0.25
    if mid and exact:
        return 1.0
    if mid:
        return 0.75
    return 0.5


class Tracker:
    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"FELL": None}
        self.rock_hist: deque = deque(maxlen=self.window + 1)
        self.theta_hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.clear: deque = deque(maxlen=self.window)
        self.prev = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["FELL"] is not None

    @staticmethod
    def at_start(pos) -> bool:
        return bool(float(np.linalg.norm(np.asarray(pos)[:2] - np.array([L.SPEC_X, L.SPEC_Y]))) <= 0.04)

    def update(self, bal, rock, arms) -> Result:
        self.steps += 1
        try:
            pos = np.asarray(rock.origin, dtype=float).ravel()
            ok = pos.size == 3 and bool(np.all(np.isfinite(pos)))
        except (AttributeError, TypeError, ValueError):
            ok, pos = False, None
        th, q100, q10, qp = (joint(bal, n) for n in MOVING) if bal is not None else (None,) * 4
        if not ok or None in (th, q100, q10, qp):
            self.bad_steps += 1
            for h in (self.rock_hist, self.theta_hist, self.released, self.clear):
                h.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.at_start(pos) and float(np.linalg.norm(pos - self.prev)) >= L.RESET_JUMP:
            self.reset()  # the specimen put back on its mark: a new episode
            self.steps = 1
            self.resets += 1
        self.prev = pos
        self.rock_hist.append(pos)
        self.theta_hist.append(th)
        if self.latch["FELL"] is None and fell(rock):
            self.latch["FELL"] = self.steps
        gap = None
        try:
            rel = bool(rigstate.released(arms, rock.distance, L.LET_GO))
            gap = rigstate.hands_gap(arms, rock.distance)
        except Exception:  # noqa: BLE001 - no distance or no arms: not released
            rel = False
        self.released.append(rel)
        try:
            clr = bool(rigstate.released(arms, lambda p: bal.distance(p, bodies=MOVING), L.LET_GO))
        except Exception:  # noqa: BLE001
            clr = False
        self.clear.append(clr)
        self.last = self.judge(bal, rock, th, q100, q10, qp, gap)
        return self.last

    def judge(self, bal, rock, th, q100, q10, qp, gap) -> Result:
        full = len(self.rock_hist) == self.window + 1
        h = list(self.rock_hist)
        still = bool(full and max(float(np.linalg.norm(x - h[0])) for x in h[1:]) <= L.STILL_MOVE + L.EPS)
        released = len(self.released) == self.window and all(self.released)
        rest = bool(len(self.theta_hist) == self.window + 1
                    and math.degrees(max(self.theta_hist) - min(self.theta_hist)) <= L.REST_DEG + L.EPS)
        clear = len(self.clear) == self.window and all(self.clear)
        loc = specimen_in_beam_frame(bal, rock)
        pan = on_pan(loc)
        notch = notches_ok(q100, q10)
        read = reading_g(q100, q10, qp)
        err = read - L.SPEC_G
        ang = abs(math.degrees(th))
        mid = bool(rest and abs(err) <= L.READ_TOL_G_MID + L.EPS and ang <= L.MID_DEG + L.EPS)
        exact = bool(clear and abs(err) <= L.READ_TOL_G + L.EPS and ang <= L.ZERO_DEG + L.EPS)
        score = tier(self.latched, released and still, pan, notch, mid, exact)
        d = {
            "valid_state": True, "score": score, "latch": dict(self.latch),
            "on_pan": pan, "notches_ok": notch, "reading_g": round(read, 3), "error_g": round(err, 3),
            "beam_deg": round(math.degrees(th), 3), "q100_mm": round(q100 * 1000, 2), "q10_mm": round(q10 * 1000, 2),
            "poise_mm": round(qp * 1000, 2), "rest_1s": rest, "released_1s": bool(released), "still_1s": still,
            "hands_clear_1s": clear, "hands_gap_mm": None if gap is None else round(float(gap) * 1000, 2),
            "rock_mm": [round(float(v) * 1000, 2) for v in rock.origin],
        }
        return Result(score, bool(score == 1.0), d)
