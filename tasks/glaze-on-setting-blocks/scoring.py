"""The scorer, as plain functions of the glass unit's and the two setting blocks' states. NumPy only; no simulator.

The rule is about the whole trial, not only its end: the glass must never touch the sill. So the scorer keeps a
little state (whether any corner of the unit has ever come down to the sill inside the channel) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(unit, block_1, block_2) -> Result(score, success, details)

Each state is anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. All three are boxes whose sizes
are in ``layout``, so their corners and faces follow from ``pos`` and ``quat_xyzw``. A state with a non-finite
number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The unit and the blocks are plain boxes, the same either way up: the unit may stand on either long edge and a block
may lie on either broad face, so their tilts are folded to 0..90 degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_UH = np.array(L.UNIT_SIZE) / 2
_BH = np.array(L.BLOCK_SIZE) / 2
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it
_CHANNEL = (L.FRAME_X - L.CHANNEL_HALF_W, L.FRAME_X + L.CHANNEL_HALF_W,
            L.FRAME_Y - L.CHANNEL_HALF_L, L.FRAME_Y + L.CHANNEL_HALF_L)  # x0, x1, y0, y1 (inner faces)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(*states) -> bool:
    """Every number finite and every quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    for s in states:
        try:
            parts = [np.asarray(getattr(s, k), dtype=float).ravel()
                     for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
        except (AttributeError, TypeError, ValueError):
            return False
        if not all(p.size and np.all(np.isfinite(p)) for p in parts):
            return False
        if parts[1].size != 4 or np.linalg.norm(parts[1]) < 0.5:
            return False
    return True


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def folded_tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the body's z axis, either way up (0..90 degrees)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def block_yaw_err_deg(quat_xyzw) -> float:
    """How far a block's sides are turned from the channel's (0..90 degrees; a rectangle has 2 ways round)."""
    r = rotation(quat_xyzw)
    a = math.degrees(math.atan2(r[1, 0], r[0, 0])) % 180.0
    return float(min(a, 180.0 - a))


def _up(r: np.ndarray) -> np.ndarray:
    """The body z axis, turned to point up (a plain box is the same either way up)."""
    n = r[:, 2]
    return n if n[2] >= 0 else -n


def corners(s, half) -> np.ndarray:
    """The eight corners of a box (world), (8, 3)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    return np.array([c + r @ (half * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])


def _face(s, half, up: bool) -> np.ndarray:
    """The four corners (world) of a box's top (``up``) or bottom face, whichever z face faces that way."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    n = _up(r) * (1.0 if up else -1.0)
    return np.array([c + n * half[2] + r[:, 0] * (i * half[0]) + r[:, 1] * (j * half[1]) for i, j in _SIGNS])


# -- polygons seen from above -----------------------------------------------------------------------------------
def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex polygon ``clip`` (both CCW)."""
    out = [np.asarray(v, dtype=float) for v in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            return None
        prev = inp[-1]
        for cur in inp:
            sp, sc = side(prev), side(cur)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
            prev = cur
    return np.array(out) if len(out) >= 3 else None


def _inside_channel(xy, slack: float) -> bool:
    x0, x1, y0, y1 = _CHANNEL
    return bool(x0 - slack <= xy[0] <= x1 + slack and y0 - slack <= xy[1] <= y1 + slack)


# -- the glass unit -------------------------------------------------------------------------------------------------
def unit_tilt_deg(unit) -> float:
    """How far the unit's 80 mm height axis is from vertical (0..90; either long edge down)."""
    return folded_tilt_deg(unit.quat_xyzw)


def unit_low(unit) -> float:
    """Height of the unit's lowest corner (z)."""
    return float(corners(unit, _UH)[:, 2].min())


def in_channel(unit) -> bool:
    """The unit's whole outline seen from above (all eight corners) lies inside the channel, between the stops'
    and the jambs' inner faces (1 mm slack)."""
    return all(_inside_channel(p[:2], L.CHANNEL_SLACK) for p in corners(unit, _UH))


def touches_sill(unit) -> bool:
    """A corner of the unit inside the channel's outline (2 mm slack) is less than 1 mm above the marks' top
    (2 mm above the sill's): it is down on the sill or on a mark, not on a block (8 mm tall). A box's lowest
    point is always at a corner, so this catches any part of it coming down there."""
    return any(p[2] < L.SILL_TOUCH_Z and _inside_channel(p[:2], L.TOUCH_SLACK) for p in corners(unit, _UH))


def standing(unit) -> bool:
    """Standing in the channel: inside it, height axis within 3 degrees of vertical."""
    return bool(in_channel(unit) and unit_tilt_deg(unit) < L.STAND_TILT_DEG)


def centre_off(unit) -> float:
    """How far the unit's centre is from the channel's centre along the channel (y), m."""
    return float(abs(unit.pos[1] - L.FRAME_Y))


def off_table(s) -> bool:
    return bool(s.low[2] < L.TOP - L.OFF_TABLE_DROP)


def _underside_z(unit, xy) -> float:
    """Height of the plane of the unit's underside (the 24 x 200 mm face nearer facing down) above (x, y)."""
    r = rotation(unit.quat_xyzw)
    n = _up(r)
    c = np.asarray(unit.pos, dtype=float) - n * _UH[2]
    if n[2] < 1e-6:
        return float(c[2])
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


# -- the setting blocks -----------------------------------------------------------------------------------------------
def block_top(block) -> np.ndarray:
    """Centre of the block's upper broad face (world)."""
    return np.asarray(block.pos, dtype=float) + _up(rotation(block.quat_xyzw)) * _BH[2]


def block_on_sill(block) -> bool:
    """Lying flat on the sill (or on a mark) inside the channel: its 8 mm axis within 10 degrees of vertical, its
    lowest corner within 1 mm below to 3 mm above the sill's top, its centre inside the channel's outline."""
    low = float(corners(block, _BH)[:, 2].min())
    return bool(
        folded_tilt_deg(block.quat_xyzw) < L.BLOCK_FLAT_DEG
        and L.BLOCK_LOW_BAND[0] <= low <= L.BLOCK_LOW_BAND[1]
        and _inside_channel(np.asarray(block.pos, dtype=float)[:2], 0.0)
    )


def mark_offsets(block) -> tuple[float, float]:
    """Horizontal distance (m) of the block's centre from the right (y -0.05) and the left (y +0.05) mark's centre."""
    return tuple(float(np.hypot(block.pos[0] - L.FRAME_X, block.pos[1] - my)) for my in L.MARK_YS)


def block_set_on(block, tol: float) -> int | None:
    """Which mark (0 right, 1 left) the block is set on: on the sill in the channel, square within 15 degrees,
    centre within ``tol`` of the mark's centre. None if neither (the marks are 100 mm apart: never both)."""
    if not block_on_sill(block) or block_yaw_err_deg(block.quat_xyzw) > L.BLOCK_YAW_DEG:
        return None
    for i, d in enumerate(mark_offsets(block)):
        if d <= tol:
            return i
    return None


def both_set(blocks, tol: float) -> bool:
    """One block set on each mark."""
    return {block_set_on(b, tol) for b in blocks} == {0, 1}


def share_under(unit, block) -> float:
    """Share (0..1) of the block's top face, seen from above, that lies under the unit's underside."""
    top = _ccw(_face(block, _BH, up=True)[:, :2])
    whole = _area(top)
    if not np.isfinite(whole) or whole <= 1e-12:
        return 0.0
    part = _clip(top, _ccw(_face(unit, _UH, up=False)[:, :2]))
    if part is None:
        return 0.0
    share = _area(part) / whole
    return float(np.clip(share, 0.0, 1.0)) if np.isfinite(share) else 0.0


def gap_to_block(unit, block) -> float:
    """Height of the unit's underside above the block's top, at the centre of the block's top (m)."""
    t = block_top(block)
    return _underside_z(unit, t[:2]) - float(t[2])


def on_block(unit, block) -> bool:
    """The unit rests on the block: the block lies flat on the sill in the channel, at least half of its top is
    under the unit, and the unit's underside is within 2 mm of the block's top there."""
    return bool(
        block_on_sill(block)
        and share_under(unit, block) >= L.ON_BLOCK_SHARE
        and abs(gap_to_block(unit, block)) <= L.ON_BLOCK_GAP
    )


def _mm(v: float) -> float | None:
    return round(v * 1000, 1) if np.isfinite(v) else None


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
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them (ignored)
        self.touched = False  # the unit has touched the sill inside the channel
        self.touched_at = None  # the step it first did
        self.min_low = math.inf  # the lowest any corner of the unit has been inside the channel (z)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of the unit and both blocks at the previous valid control step

    @staticmethod
    def at_start(unit, blocks) -> bool:
        """The starting layout: the unit standing still on the table on its mark, each block lying still and
        square on the tray near a starting spot."""
        u_mark = (np.hypot(unit.pos[0] - L.UNIT_X, unit.pos[1] - L.UNIT_Y) < L.START_TOL
                  and abs(unit_low(unit) - L.TOP) < 0.003 and unit_tilt_deg(unit) < 1.0 and is_resting(unit))
        if not u_mark:
            return False
        for b in blocks:
            near = min(np.hypot(b.pos[0] - x, b.pos[1] - y) for x, y in L.BLOCK_STARTS) < L.START_TOL
            on_tray = abs(float(corners(b, _BH)[:, 2].min()) - L.TRAY_TOP) < 0.003
            if not (near and on_tray and folded_tilt_deg(b.quat_xyzw) < 1.0 and is_resting(b)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def update(self, unit, block_1, block_2) -> Result:
        self.steps += 1
        blocks = (block_1, block_2)
        if not valid(unit, *blocks):  # a broken observation: count it, change nothing, score it 0
            self.bad_steps += 1
            self.last = self.result(unit, *blocks)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (unit, *blocks)]
        if self.prev is not None and self._teleported(now) and self.at_start(unit, blocks):
            self.reset()  # the simulator put everything back on its starting spots: a new episode
            self.steps = 1
        self.prev = now
        inside = [p for p in corners(unit, _UH) if _inside_channel(p[:2], L.TOUCH_SLACK)]
        if inside:
            self.min_low = min(self.min_low, min(float(p[2]) for p in inside))
        if touches_sill(unit) and not self.touched:
            self.touched, self.touched_at = True, self.steps
        self.last = self.result(unit, *blocks)
        return self.last

    def result(self, unit, block_1, block_2) -> Result:
        blocks = (block_1, block_2)
        if not valid(unit, *blocks):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "touched": self.touched})
        tilt, off = unit_tilt_deg(unit), centre_off(unit)
        stand = standing(unit)
        on = [on_block(unit, b) for b in blocks]
        set_tight, set_loose = both_set(blocks, L.BLOCK_TIGHT), both_set(blocks, L.BLOCK_LOOSE)
        down = tilt > L.DOWN_TILT_DEG
        fell = off_table(unit)
        if self.touched or down or fell:
            score = 0.0
        elif stand and all(on) and set_tight and off <= L.UNIT_CENTRE_TIGHT:
            score = 1.0
        elif stand and all(on) and set_loose and off <= L.UNIT_CENTRE_LOOSE:
            score = 0.75
        elif stand and any(on):
            score = 0.5
        elif set_tight:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(unit) and all(is_resting(b) for b in blocks))
        details = {
            "valid_state": True,
            "touched": self.touched,
            "touched_at": self.touched_at,
            "in_channel": in_channel(unit),
            "standing": stand,
            "unit_tilt_deg": round(tilt, 2),
            "unit_off_mm": _mm(off),
            "unit_low_mm": _mm(unit_low(unit) - L.SILL_TOP),  # lowest corner above the sill's top
            "min_low_in_channel_mm": _mm(self.min_low - L.SILL_TOP),
            "on_block": on,
            "gap_mm": [_mm(gap_to_block(unit, b)) for b in blocks],
            "share_under": [round(share_under(unit, b), 3) for b in blocks],
            "block_on_sill": [block_on_sill(b) for b in blocks],
            "block_mark_mm": [[_mm(d) for d in mark_offsets(b)] for b in blocks],
            "block_yaw_deg": [round(block_yaw_err_deg(b.quat_xyzw), 2) for b in blocks],
            "blocks_set": set_tight,
            "blocks_set_loose": set_loose,
            "lying_down": down,
            "off_table": fell,
        }
        return Result(score, success, details)
