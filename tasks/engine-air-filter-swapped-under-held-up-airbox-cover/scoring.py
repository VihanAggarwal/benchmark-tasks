"""The scorer, as plain functions of the airbox's, the cover's and the two filters' state and of the arms. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the airbox (the clips' joint angles), the cover (its three duct joints: dx, dy, dz from
its seated place) and the old and new filter (bottom-centre origin, yaw). Every reading is a pose; the arms are read
only for the let-go check.

    OLD OUT       the old filter's whole outline (top view) clear of the case's footprint, its lowest point within
                  6 mm of the table (a mat is 2 mm), tilted under 10 degrees
    NEW SEATED    the new filter flat on the seat (all four bottom corners within 2 mm of the seat face), long side
                  along the pocket either way round (yaw within 5 degrees of 0 or 180), its centre within 6 mm of
                  the pocket's centre, tilted under 3 degrees
    COVER SEATED  rim within 2 mm of the case's rim (dz), tabs in their slots (dx within 4 mm, dy within 5 mm: both
                  tabs at least 5 mm into the slots, the lugs under the clips' hooks)
    CLIPPED       both clips' levers within 5 degrees of closed (and not past it: a hook with no lug under it drops
                  to its stop), with the cover seated
    HANDS OFF     on every step of the last 1 s both finger zones are 30 mm or more from the airbox, the cover and
                  both filters, and over that 1 s nothing moves more than 5 mm or turns more than 5 degrees
    ZERO (latched) DUCT: the cover pulled past the duct's travel (dx over 35 mm, dy under -35 or over 280 mm, dz over
                  70 mm: pulled off its stub); FELL: a filter's lowest point more than 20 mm under the table top.
                  A clip shut for a while on an empty or misseated box is no zero (the engine is off: no harm); the
                  end state scores 0 unless the new filter is seated

    0      a zero latched, or not hands off
    1.0    old out, new seated, cover seated, both clips closed
    0.5    old out, new seated, cover seated, a clip open
    0.25   old out and new seated, the cover not seated
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

AIRBOX, COVER, OLD, NEW = "airbox", "cover", "old_filter", "new_filter"
PARTS = (AIRBOX, COVER, OLD, NEW)
FILTERS = (OLD, NEW)
_FILTER_PTS = np.array([(x, y, 0.0) for x, y in L.outline_corners()] + [(x, y, L.F_H) for x, y in L.outline_corners()])
_BOTTOM = _FILTER_PTS[:4]


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def yaw_of(r: np.ndarray) -> float:
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Reading:
    """One step (m), the world frame with the table top at z 0."""
    pos: dict  # filter name -> bottom-centre origin
    rot: dict  # filter name -> rotation
    pts: dict  # filter name -> (8, 3) outline corners (4 bottom, 4 top)
    duct: tuple  # (dx, dy, dz)
    clips: tuple  # (q_right, q_left)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, pts = {}, {}, {}
    try:
        for name in FILTERS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name], pts[name] = o, r, o + _FILTER_PTS @ r.T
        cov, box = states.get(COVER), states.get(AIRBOX)
        if cov is None or box is None:
            return None
        duct = tuple(float(cov.joints[n]) for n in L.COVER_JOINTS)
        clips = tuple(float(box.joints[n]) for n in L.CLIP_NAMES)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(v) for v in duct + clips):
        return None
    return Reading(pos, rot, pts, duct, clips)


# -- the readings ------------------------------------------------------------------------------------------------------
def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def _rect_overlap(poly, x0, x1, y0, y1) -> bool:
    """Whether a convex polygon (N, 2) overlaps the axis rectangle (separating axis test, touching counts)."""
    poly = np.asarray(poly, dtype=float)
    if poly[:, 0].max() < x0 or poly[:, 0].min() > x1 or poly[:, 1].max() < y0 or poly[:, 1].min() > y1:
        return False
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        n = np.array([-e[1], e[0]])
        if float(np.linalg.norm(n)) < 1e-12:
            continue
        a, b = poly @ n, rect @ n
        if a.max() < b.min() - 1e-12 or b.max() < a.min() - 1e-12:
            return False
    return True


def old_out(r: Reading, name: str = OLD) -> bool:
    """The filter's whole outline outside the case's footprint (top view), resting on the table or a mat."""
    p = r.pts[name]
    poly = p[:4, :2]
    if _rect_overlap(poly, L.CASE_X0 - L.SEAT_CLEAR_X, L.CASE_X1 + L.SEAT_CLEAR_X, L.CASE_Y0, L.CASE_Y1):
        return False
    low = float(np.min(p[:, 2]))
    return bool(-0.003 <= low <= L.OLD_REST_HIGH and tilt_deg(r.rot[name]) <= L.OLD_TILT_DEG)


def seated(r: Reading, name: str) -> bool:
    """Flat on the seat (each bottom corner within 2 mm of the seat face), long side along the pocket either way
    round (yaw within 5 degrees of 0 or 180), centred in the pocket within 6 mm, tilted under 3 degrees."""
    p = r.pts[name]
    flat = bool(np.all(np.abs(p[:4, 2] - L.SEAT_Z) <= L.SEAT_TOL + L.EPS))
    aligned = abs(wrap(2.0 * yaw_of(r.rot[name]))) / 2.0 <= L.YAW_TOL + L.EPS
    c = r.pos[name]
    centred = abs(c[0] - L.CX) <= L.SEAT_XY_TOL + L.EPS and abs(c[1] - L.CY) <= L.SEAT_XY_TOL + L.EPS
    return bool(flat and aligned and centred and tilt_deg(r.rot[name]) <= 3.0)


def tab_in(r: Reading) -> float:
    """How far both tabs are into their slots (m): the tab's length less the cover's rise, 0 when a tab is not over
    its slot (it rests on the rear wall's top)."""
    dx, dy, dz = r.duct
    if abs(dx) <= L.COVER_DX_TOL + L.EPS and abs(dy) <= L.TAB_SLACK_Y + L.EPS:
        return max(0.0, L.TAB_DROP - dz)
    return 0.0


def cover_seated(r: Reading) -> bool:
    dx, dy, dz = r.duct
    return bool(dz <= L.COVER_DZ_TOL + L.EPS and abs(dx) <= L.COVER_DX_TOL + L.EPS
                and abs(dy) <= L.COVER_DY_TOL + L.EPS and tab_in(r) >= L.TAB_IN_MIN)


def clip_closed(q: float, r: Reading) -> bool:
    """Within 5 degrees of closed, resting on a lug (not dropped to its stop), the cover seated under it."""
    return bool(L.CLIP_Q_CLOSED_MIN - L.EPS <= q <= L.CLIP_Q_CLOSED_MAX + L.EPS and cover_seated(r))


def duct_torn(r: Reading) -> bool:
    dx, dy, dz = r.duct
    return bool(abs(dx) > L.DUCT_TEAR_X or dy < L.DUCT_TEAR_Y[0] or dy > L.DUCT_TEAR_Y[1] or dz > L.DUCT_TEAR_Z)


def fell(r: Reading) -> bool:
    return any(float(np.min(r.pts[n][:, 2])) < -L.FALL_DROP for n in FILTERS)


def tier(latched: bool, settled: bool, out: bool, new: bool, cover: bool, clips: int) -> float:
    if latched or not settled or not (out and new):
        return 0.0
    if cover:
        return 1.0 if clips == 2 else 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The latches (DUCT, FELL), the last 1 s of readings (stillness) and of the let-go check, and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"DUCT": None, "FELL": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        o = np.array([L.OLD_START[0], L.OLD_START[1], L.OLD_START[2]])
        n = np.array([L.NEW_START[0], L.NEW_START[1], L.NEW_START[2]])
        return (float(np.linalg.norm(r.pos[OLD] - o)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[NEW] - n)) <= L.START_TOL
                and max(abs(v) for v in r.duct) <= 0.003 and all(abs(q) <= 0.1 for q in r.clips))

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in FILTERS)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.latch["DUCT"] is None and duct_torn(r):
            self.latch["DUCT"] = self.steps
        if self.latch["FELL"] is None and fell(r):
            self.latch["FELL"] = self.steps
        try:
            dists = [states[n].distance for n in PARTS]

            def dist(points):
                return np.min([d(points) for d in dists], axis=0)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of either filter, the largest move of the cover
        and the largest swing of a clip (rad) from where each was at the window's start; None before 1 s of valid
        readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = cover = clip = 0.0
        for x in h[1:]:
            for n in FILTERS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
            cover = max(cover, float(np.linalg.norm(np.asarray(x.duct) - np.asarray(first.duct))))
            clip = max(clip, max(abs(a - b) for a, b in zip(x.clips, first.clips)))
        return {"move": move, "turn": turn, "cover": cover, "clip": clip}

    def judge(self, r: Reading, gap) -> Result:
        out, new = old_out(r), seated(r, NEW)
        cover = cover_seated(r)
        n_clips = sum(clip_closed(q, r) for q in r.clips)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["cover"] <= L.STILL_MOVE + L.EPS
                     and m["turn"] <= L.STILL_TURN_DEG + L.EPS and m["clip"] <= L.STILL_CLIP_RAD + L.EPS)
        level = tier(self.latched, released and still, out, new, cover, n_clips)
        dx, dy, dz = r.duct
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "old_out": out, "new_seated": new, "old_seated": seated(r, OLD), "cover_seated": cover,
            "clips_closed": n_clips, "clip_q_deg": [round(math.degrees(q), 1) for q in r.clips],
            "cover_dx_mm": _mm(dx), "cover_dy_mm": _mm(dy), "cover_dz_mm": _mm(dz), "tab_in_mm": _mm(tab_in(r)),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "old_mm": [round(float(c) * 1000.0, 1) for c in r.pos[OLD]],
            "new_mm": [round(float(c) * 1000.0, 1) for c in r.pos[NEW]],
            "old_yaw_deg": round(math.degrees(yaw_of(r.rot[OLD])), 1),
            "new_yaw_deg": round(math.degrees(yaw_of(r.rot[NEW])), 1),
            "new_corner_dz_mm": [round(float(z) * 1000.0, 1) for z in r.pts[NEW][:4, 2] - L.SEAT_Z],
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "cover_mm": _mm(m["cover"]),
                                              "clip_deg": round(math.degrees(m["clip"]), 2)},
        }
        return Result(level, bool(level == 1.0), details)
