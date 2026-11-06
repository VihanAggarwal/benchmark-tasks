"""The scorer, as plain functions of the clamp's and the cover's states and the arms' state. NumPy only; no simulator.

The score is judged at the end of the trial from the end state: where the thermometer clamp sits on the suction line,
whether the foam cover sits over it, and whether the grippers have let go (of both parts for 1.0, of the clamp for the
lower tiers). The only memory is a short ring buffer
per part (its frame origin over the last 10 control steps, 0.5 s) to tell 'at rest' by the path it travelled, never
by a velocity (resting parts' reported velocities chatter in the simulator). It is fed one observation per control
step by ``task.success()``:

    Tracker.update(clamp, cover, arms, offline=False) -> Result(score, success, details)

``clamp`` and ``cover`` are mechkit ``AssemblyState``s (or anything with ``origin``, ``quat_xyzw``, ``low`` and
``high``): ``origin`` is the part's frame origin in the world (the clamp's: the centre of its top block's underside;
the cover's: the centre of its bottom), ``low``/``high`` its world-aligned bounding box. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp`` and ``tips`` (the point between the pad tips), or
None. A state with a missing field, a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that
step and leaves the tracker as it was.

The clamp is the same turned 180 degrees about the vertical, so its heading is folded (mod 180 degrees). It is not
the same upside down (display on top, legs below), so its tilt is not folded; nor is the cover's.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_TOP = np.array([0.0, 0.0, L.TOP])
_BLOCK_HALF = np.array(L.BLOCK_S[:2]) / 2
_UNDER = np.array([(sx * _BLOCK_HALF[0], sy * _BLOCK_HALF[1], 0.0) for sx in (-1, 1) for sy in (-1, 1)])
_BLOCK_TOP = _UNDER + np.array([0.0, 0.0, L.BLOCK_S[2]])
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)


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
    """Every field present, every number finite and the quaternion not degenerate."""
    try:
        o, q, lo, hi = (np.asarray(getattr(s, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "low", "high"))
    except (AttributeError, TypeError, ValueError):
        return False
    if not (o.size == 3 and q.size == 4 and lo.size == 3 and hi.size == 3):
        return False
    if not all(np.all(np.isfinite(p)) for p in (o, q, lo, hi)):
        return False
    return bool(np.linalg.norm(q) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP and pad-tip point (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            for p in (arms[side].tcp, arms[side].tips):
                p = np.asarray(p, dtype=float).ravel()
                if p.size != 3 or not np.all(np.isfinite(p)):
                    return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def origin(s) -> np.ndarray:
    """A part's frame origin, its height above the table top (the scorer works in heights, like layout)."""
    return np.asarray(s.origin, dtype=float).ravel() - _TOP


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's +z axis and the world's vertical (degrees, 0..180)."""
    return float(np.degrees(np.arccos(np.clip(rotation(quat_xyzw)[2, 2], -1.0, 1.0))))


def heading_err_deg(quat_xyzw) -> float:
    """How far the body's y axis, seen from above, is turned from the world's y axis (the line), either way round
    (0..90 degrees)."""
    by = rotation(quat_xyzw)[:, 1]
    return float(np.degrees(math.atan2(abs(by[0]), abs(by[1]))))


def to_world(s, local) -> np.ndarray:
    """Points given in a part's frame, in the world (heights above the table top)."""
    return np.asarray(local, dtype=float) @ rotation(s.quat_xyzw).T + origin(s)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo, dtype=float) - p, p - np.asarray(hi, dtype=float)))
    return float(np.linalg.norm(d))


def part_bounds(shapes, origin_world, quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """World-aligned (low, high) around a part's boxes (layout's CLAMP_SHAPES / COVER_SHAPES) at a pose."""
    r = rotation(quat_xyzw)
    pts = [np.asarray(origin_world, dtype=float) + r @ (np.asarray(c) + _SIGNS[i] * np.asarray(s) / 2)
           for c, s, _ in shapes for i in range(8)]
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


# -- the clamp --------------------------------------------------------------------------------------------------
def d_along(clamp) -> float:
    """d: from the suction service valve's body face (y 0.190) along the line to the clamp's origin (m)."""
    return float(L.VALVE_FACE_Y - origin(clamp)[1])


def block_corner_ys(clamp) -> tuple[float, float]:
    """The lowest and highest y of the four bottom corners of the clamp's 49 x 40 mm top block."""
    ys = to_world(clamp, _UNDER)[:, 1]
    return float(ys.min()), float(ys.max())


def on_bare_copper(clamp) -> bool:
    """The top block's whole 40 mm length lies along the bare copper: every bottom corner between the brazed
    coupling's end (y 0.165) and the foam's end (y -0.080)."""
    lo, hi = block_corner_ys(clamp)
    return bool(lo >= L.FOAM_START_Y and hi <= L.COUPLING_END_Y)


def seat_checks(clamp) -> dict:
    """The geometric parts of 'seated' (everything but being at rest)."""
    o = origin(clamp)
    return {
        "x": bool(abs(o[0] - L.LINE_X) <= L.SEAT_X_TOL),
        "z": bool(L.SEAT_Z_LO <= o[2] <= L.SEAT_Z_HI),
        "tilt": bool(tilt_deg(clamp.quat_xyzw) <= L.SEAT_TILT_DEG),
        "heading": bool(heading_err_deg(clamp.quat_xyzw) <= L.SEAT_YAW_DEG),
        "bare": on_bare_copper(clamp),
    }


# -- the cover --------------------------------------------------------------------------------------------------
def block_in_cover(clamp, cover) -> bool:
    """Seen from above, the four top corners of the clamp's top block, taken into the cover's frame, lie inside the
    cover's 77 x 70 mm outline."""
    local = (to_world(clamp, _BLOCK_TOP) - origin(cover)) @ rotation(cover.quat_xyzw)
    return bool(np.all(np.abs(local[:, 0]) <= L.COVER_FOOT_X) and np.all(np.abs(local[:, 1]) <= L.COVER_FOOT_Y))


def ceiling_gap(clamp, cover) -> float:
    """Height of the cover's channel ceiling (at its centre) above the top of the clamp's display (m)."""
    ceiling = to_world(cover, [0.0, 0.0, L.CEILING])
    display = to_world(clamp, [0.0, 0.0, L.DISPLAY_TOP])
    return float(ceiling[2] - display[2])


def cover_checks(clamp, cover) -> dict:
    """The geometric parts of 'insulated' (everything but being at rest)."""
    g = ceiling_gap(clamp, cover)
    return {
        "inside": block_in_cover(clamp, cover),
        "ceiling": bool(L.CEIL_GAP_LO <= g <= L.CEIL_GAP_HI),
        "tilt": bool(tilt_deg(cover.quat_xyzw) <= L.COVER_TILT_DEG),
    }


# -- the arms ---------------------------------------------------------------------------------------------------
def gripper_margin(arm, part) -> float:
    """How clear one gripper is of one part (m; >= 0 means clear). Clear: its TCP and pad-tip point both at least
    4 cm above the top of the part's bounding box, or both at least 6.5 cm from that box seen from above (an open
    finger reaches 57 mm sideways from the TCP, its corners about 60 mm: checking only the two points on the
    gripper's axis would miss a finger resting on the part). The margin is the better of the two ways, each taken
    at the worse of the two points."""
    lo, hi = np.asarray(part.low, dtype=float), np.asarray(part.high, dtype=float)
    above = side = math.inf
    for p in (arm.tcp, arm.tips):
        p = np.asarray(p, dtype=float).ravel()
        above = min(above, float(p[2] - hi[2]))
        side = min(side, point_box_distance(p[:2], lo[:2], hi[:2]))
    return max(above - L.CLEAR_ABOVE, side - L.CLEAR_SIDE)


def clear_margin(arms, part) -> float:
    """The worse of the two grippers' margins for one part (m; >= 0: both grippers clear of it)."""
    return min(gripper_margin(arms[side], part) for side in ("left", "right"))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The last 0.5 s of both parts' positions, for 'at rest'."""

    def __init__(self) -> None:
        self.fallback_resets = 0  # trials started by the fallback (a teleport back to the start), never cleared
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a broken part state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.trail = {"clamp": deque(maxlen=L.STILL_STEPS + 1), "cover": deque(maxlen=L.STILL_STEPS + 1)}
        self.last: Result = Result(0.0, False, {})

    # -- at rest ---------------------------------------------------------------------------------------------
    def path(self, part: str) -> float | None:
        """How far the part's origin travelled over the last 10 control steps (m), or None before 11 samples."""
        t = self.trail[part]
        if len(t) < t.maxlen:
            return None
        pts = np.array(t)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def at_rest(self, part: str) -> bool:
        p = self.path(part)
        return p is not None and p < L.STILL_PATH

    # -- reset fallback --------------------------------------------------------------------------------------
    @staticmethod
    def at_start(clamp, cover) -> bool:
        """Both parts upright on their pads, each within 4.5 cm of its start (a seeded reset's jitter)."""
        for s, start in ((clamp, L.CLAMP_START), (cover, L.COVER_START)):
            o = origin(s)
            if not (math.hypot(o[0] - start[0], o[1] - start[1]) <= L.START_TOL
                    and abs(o[2] - start[2]) <= L.START_Z_TOL and tilt_deg(s.quat_xyzw) <= L.START_TILT_DEG):
                return False
        return True

    def _teleported(self, clamp, cover) -> bool:
        """Some part jumped more than 5 cm in one control step and both now stand on their start places."""
        jumped = False
        for name, s in (("clamp", clamp), ("cover", cover)):
            t = self.trail[name]
            if t and float(np.linalg.norm(origin(s) - t[-1])) > L.RESET_JUMP:
                jumped = True
        return jumped and self.at_start(clamp, cover)

    # -- feeding ---------------------------------------------------------------------------------------------
    def update(self, clamp, cover, arms=None, offline: bool = False) -> Result:
        if not (valid(clamp) and valid(cover)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(clamp, cover, arms, offline)
            return self.last
        if self._teleported(clamp, cover):
            self.reset()  # the simulator put both parts back on their pads: a new trial (rigstate's reset hook
            self.fallback_resets += 1  # normally does this first; this is the fallback)
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.trail["clamp"].append(origin(clamp))
        self.trail["cover"].append(origin(cover))
        self.last = self.result(clamp, cover, arms, offline)
        return self.last

    def result(self, clamp, cover, arms=None, offline: bool = False) -> Result:
        if not (valid(clamp) and valid(cover)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        have_arms = arms_valid(arms)
        m_clamp = clear_margin(arms, clamp) if have_arms else None
        m_cover = clear_margin(arms, cover) if have_arms else None
        if have_arms:
            clear_clamp = bool(m_clamp >= 0.0)
            clear_both = bool(clear_clamp and m_cover >= 0.0)
        else:  # unreadable arms: clear only offline (the checker's first call)
            clear_clamp = clear_both = bool(offline)
        seat = seat_checks(clamp)
        clamp_rest, cover_rest = self.at_rest("clamp"), self.at_rest("cover")
        seated = bool(all(seat.values()) and clamp_rest)
        cov = cover_checks(clamp, cover)
        insulated = bool(all(cov.values()) and cover_rest)
        d = d_along(clamp)
        near = bool(d <= L.NEAR_D)
        if seated and near and insulated and clear_both:
            score = 1.0
        elif seated and near and clear_clamp:
            score = 0.5
        elif seated and clear_clamp:
            score = 0.25
        else:
            score = 0.0
        o_c, o_v = origin(clamp), origin(cover)
        ys = block_corner_ys(clamp)
        details = {
            "valid_state": True,
            "seated": seated,
            "near": near,
            "insulated": insulated,
            "clear_of_clamp": clear_clamp,
            "clear_of_both": clear_both,
            "seat_checks": seat,
            "cover_checks": cov,
            "d_mm": _mm(d),
            "clamp_x_off_mm": _mm(o_c[0] - L.LINE_X),
            "clamp_above_copper_mm": _mm(o_c[2] - L.LINE_TOP),
            "clamp_tilt_deg": round(tilt_deg(clamp.quat_xyzw), 2),
            "clamp_heading_deg": round(heading_err_deg(clamp.quat_xyzw), 2),
            "clamp_ends_y_mm": [_mm(ys[0]), _mm(ys[1])],
            "clamp_path_mm": _mm(self.path("clamp")),
            "clamp_at_rest": clamp_rest,
            "ceiling_gap_mm": _mm(ceiling_gap(clamp, cover)),
            "cover_tilt_deg": round(tilt_deg(cover.quat_xyzw), 2),
            "cover_off_mm": [_mm(o_v[0] - o_c[0]), _mm(o_v[1] - o_c[1])],
            "cover_path_mm": _mm(self.path("cover")),
            "cover_at_rest": cover_rest,
            "clear_margin_mm": {"clamp": _mm(m_clamp), "cover": _mm(m_cover)},
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
