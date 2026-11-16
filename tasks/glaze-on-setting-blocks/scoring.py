"""The scorer, as plain functions of the glass unit's and the two setting blocks' states and of the arms (as
``rigstate.arms()`` gives them). NumPy only; no simulator.

The rule is about the whole trial, not only its end: the glass must never touch the sill. 'Touch' is contact: the
unit and the sill (or a white mark on it) touching or pressed into each other, their separation 0.0 mm or less. Any
air gap between them, however small, is not a touch: that is the event a chalk print shows on the bench. Contact
between two boxes is decided exactly (the separating-axis test over all 15 axes), so every face, edge and corner of
the unit counts against every face of the sill and the marks. The scorer keeps a little state (whether the unit has ever
touched the sill, and for each of the three loose objects how many control steps in a row it has been let go and its
recent poses) and is fed one observation per control step by ``task.success()``.

    Tracker.update(unit, block_1, block_2, arms) -> Result(score, success, details)

Resting is never judged from the objects alone (a gripper can hold glass still just above the blocks): every rule
that credits the unit or a block as resting needs it LET GO (both grippers' finger zones at least 30 mm from it on
each of the last 20 control steps, 1 s) and STILL (each of its last 20 poses within 2 mm and 2 degrees of the
current one), besides its support within 1 mm. This fails closed: a step on which the arms cannot be read (no
snapshot, a side missing, a non-finite number) or a distance cannot be computed counts as NOT let go for every
object, and restarts its count.

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
import rigstate

_UH = np.array(L.UNIT_SIZE) / 2
_BH = np.array(L.BLOCK_SIZE) / 2
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it
_OPENING = (L.FRAME_ROOM_X, L.STOP_X, L.FRAME_Y - L.OPEN_HALF_L, L.FRAME_Y + L.OPEN_HALF_L)  # the rebate floor
_SILL_PARTS = [L.SILL_BOX] + [((L.MARK_X, my, L.SILL_TOP + L.MARK_SIZE[2] / 2), L.MARK_SIZE) for my in L.MARK_YS]
_TOUCH_EPS = 1e-9  # 1 nm: rounding, so a computed separation of 0 is a touch whichever way it rounds


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
    """How far a block's sides are turned from the opening's (0..45 degrees). The blocks are square (30 x 30 mm), so
    a quarter turn leaves them looking and sitting the same: the turn is taken modulo 90 degrees."""
    r = rotation(quat_xyzw)
    a = math.degrees(math.atan2(r[1, 0], r[0, 0])) % 90.0
    return float(min(a, 90.0 - a))


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


def _inside_opening(xy, slack: float) -> bool:
    """Inside the rebate floor's outline seen from above: from the frame's room face to the outside stop, between
    the jambs."""
    x0, x1, y0, y1 = _OPENING
    return bool(x0 - slack <= xy[0] <= x1 + slack and y0 - slack <= xy[1] <= y1 + slack)


# -- contact between two boxes ----------------------------------------------------------------------------------
def box_separation(c1, r1, h1, c2, r2, h2) -> float:
    """Signed separation of two boxes (centre, rotation whose columns are the axes, half edges), by the separating
    axis test over their 15 axes: positive, apart by a gap at least that wide; zero, touching; negative, pressed
    into each other (by the smallest overlap). Two boxes touch or overlap exactly when it is 0 or less."""
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    h1, h2 = np.asarray(h1, dtype=float), np.asarray(h2, dtype=float)
    axes = [r1[:, i] for i in range(3)] + [r2[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            n = np.cross(r1[:, i], r2[:, j])
            if float(np.linalg.norm(n)) > 1e-6:
                axes.append(n / np.linalg.norm(n))
    d = c2 - c1
    best = -math.inf
    for n in axes:
        e1 = float(np.sum(h1 * np.abs(r1.T @ n)))
        e2 = float(np.sum(h2 * np.abs(r2.T @ n)))
        best = max(best, abs(float(d @ n)) - e1 - e2)
    return best


def sill_gap(unit) -> float:
    """The unit's separation (m) from the sill (any face of it) and from each mark, the smallest: 0 or less is
    contact."""
    r, c = rotation(unit.quat_xyzw), np.asarray(unit.pos, dtype=float)
    return min(box_separation(c, r, _UH, centre, np.eye(3), np.asarray(size) / 2) for centre, size in _SILL_PARTS)


def touches_sill(unit) -> bool:
    """The unit is in contact with the sill or a mark: touching or pressed in, separation 0.0 mm or less (to 1 nm of
    rounding). Any air gap is not a touch: a unit standing on blocks set on the marks is 8 mm over them."""
    return bool(sill_gap(unit) <= L.SILL_TOUCH_GAP + _TOUCH_EPS)


# -- the glass unit -------------------------------------------------------------------------------------------------
def unit_tilt_deg(unit) -> float:
    """How far the unit's 80 mm height axis is from vertical (0..90; either long edge down)."""
    return folded_tilt_deg(unit.quat_xyzw)


def unit_low(unit) -> float:
    """Height of the unit's lowest corner (z)."""
    return float(corners(unit, _UH)[:, 2].min())


def stop_gap(unit) -> float:
    """How far the unit's outside face (its corner furthest out, +x) is from the outside stop's face (m)."""
    return float(L.STOP_X - corners(unit, _UH)[:, 0].max())


def in_rebate(unit) -> bool:
    """The unit's whole outline seen from above (all eight corners) lies inside the opening, between the frame's room
    face and the outside stop and between the jambs (1 mm slack), with its outside face within 3 mm of the stop."""
    return bool(all(_inside_opening(p[:2], L.REBATE_SLACK) for p in corners(unit, _UH))
                and stop_gap(unit) <= L.STOP_NEAR)


def standing(unit) -> bool:
    """Standing in the rebate: in it (against or within 3 mm of the outside stop), height axis within 3 degrees of
    vertical."""
    return bool(in_rebate(unit) and unit_tilt_deg(unit) < L.STAND_TILT_DEG)


def centre_off(unit) -> float:
    """How far the unit's centre is from the opening's centre along the opening (y), m."""
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
    """Lying flat on the sill (or on a mark) inside the opening: its 8 mm axis within 10 degrees of vertical, its
    lowest corner within 1 mm of the sill's or a mark's top (1 mm below to 2 mm above the sill's top), its centre
    over the rebate floor. Geometry only: the tracker adds 'let go' and 'still'."""
    low = float(corners(block, _BH)[:, 2].min())
    return bool(
        folded_tilt_deg(block.quat_xyzw) < L.BLOCK_FLAT_DEG
        and L.BLOCK_LOW_BAND[0] <= low <= L.BLOCK_LOW_BAND[1]
        and _inside_opening(np.asarray(block.pos, dtype=float)[:2], 0.0)
    )


def mark_offsets(block) -> tuple[float, float]:
    """Horizontal distance (m) of the block's centre from the right (y -0.05) and the left (y +0.05) mark's centre."""
    return tuple(float(np.hypot(block.pos[0] - L.MARK_X, block.pos[1] - my)) for my in L.MARK_YS)


def block_set_on(block, tol: float, settled: bool = True) -> int | None:
    """Which mark (0 right, 1 left) the block is set on: on the sill in the opening, square within 15 degrees,
    centre within ``tol`` of the mark's centre, and ``settled`` (let go and still, from the tracker). None if
    neither (the marks are 100 mm apart: never both)."""
    if not settled or not block_on_sill(block) or block_yaw_err_deg(block.quat_xyzw) > L.BLOCK_YAW_DEG:
        return None
    for i, d in enumerate(mark_offsets(block)):
        if d <= tol:
            return i
    return None


def both_set(blocks, tol: float, settled=(True, True)) -> bool:
    """One block set on each mark (each let go and still)."""
    return {block_set_on(b, tol, ok) for b, ok in zip(blocks, settled)} == {0, 1}


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
    """The unit's geometry on the block: the block lies flat on the sill in the opening, at least half of its top is
    under the unit, and the unit's underside is within 1 mm of the block's top there. Geometry only: the tracker
    credits it as RESTING only with the unit and the block both let go and still."""
    return bool(
        block_on_sill(block)
        and share_under(unit, block) >= L.ON_BLOCK_SHARE
        and abs(gap_to_block(unit, block)) <= L.ON_BLOCK_GAP
    )


def box_distance(s, size):
    """A distance function (world points -> metres, 0 inside) to the box of state ``s`` (for rigstate.released)."""
    return rigstate.boxes_distance([(np.asarray(s.pos, dtype=float), np.asarray(size, dtype=float),
                                     rotation(s.quat_xyzw))])


def hand_gap(arms, s, size) -> float | None:
    """The nearer gripper's finger zone's distance to the box of ``s`` (m; negative: overlapping), None when the
    arms are unreadable; -inf when the distance cannot be computed."""
    try:
        return rigstate.hands_gap(arms, box_distance(s, size))
    except Exception:  # noqa: BLE001 - a malformed state or arm: as close as it gets (fail closed)
        return -math.inf


def let_go_now(arms, s, size) -> bool:
    """Both grippers' finger zones at least LET_GO_CLEAR from the box of ``s`` on this step. Unreadable arms, or a
    distance that cannot be computed: False (fail closed)."""
    gap = hand_gap(arms, s, size)
    return bool(gap is not None and math.isfinite(gap) and gap >= L.LET_GO_CLEAR - 1e-9)


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
        self.arms_unreadable = 0  # control steps on which the arms could not be read (nothing let go)
        self.free = [0, 0, 0]  # control steps in a row each of unit, block_1, block_2 has been let go
        self.poses: list[list[tuple[np.ndarray, np.ndarray]]] = [[], [], []]  # their last WINDOW_STEPS poses
        self.still = [False, False, False]
        self.gaps: list[float | None] = [None, None, None]  # this step's nearer finger zone gap to each (m)
        self.touched = False  # the unit has been in contact with the sill or a mark
        self.touched_at = None  # the step it first was
        self.min_sill_gap = math.inf  # the smallest separation between the unit and the sill or a mark so far (m)
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

    def _hold(self, objs, arms) -> None:
        """Count, for each loose object, the steps in a row it has been let go, and judge it still from its poses."""
        if not rigstate.readable(arms):
            self.arms_unreadable += 1
        for i, (s, size) in enumerate(zip(objs, (L.UNIT_SIZE, L.BLOCK_SIZE, L.BLOCK_SIZE))):
            gap = hand_gap(arms, s, size)
            self.gaps[i] = gap
            free = gap is not None and math.isfinite(gap) and gap >= L.LET_GO_CLEAR - 1e-9
            self.free[i] = self.free[i] + 1 if free else 0
            window = (self.poses[i] + [(np.array(s.pos, dtype=float), rotation(s.quat_xyzw))])[-L.WINDOW_STEPS:]
            self.poses[i] = window
            p, r = window[-1]
            if len(window) < L.WINDOW_STEPS:
                self.still[i] = False
                continue
            ps = np.array([w[0] for w in window])
            rs = np.array([w[1] for w in window])
            moved = np.linalg.norm(ps - p, axis=1)
            cos = (np.einsum("nij,ij->n", rs, r) - 1.0) / 2.0  # trace(R0^T R): the angle between the two poses
            turned = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            self.still[i] = bool(np.all(moved <= L.STILL_MOVE) and np.all(turned <= L.STILL_TURN_DEG))

    def _forget_hold(self) -> None:
        self.free, self.poses, self.still = [0, 0, 0], [[], [], []], [False, False, False]
        self.gaps = [None, None, None]

    def settled(self) -> list[bool]:
        """Unit, block_1, block_2: each let go for the last WINDOW_STEPS steps and still over them."""
        return [self.free[i] >= L.WINDOW_STEPS and self.still[i] for i in range(3)]

    def _sill(self, unit) -> None:
        gap = sill_gap(unit)
        self.min_sill_gap = min(self.min_sill_gap, gap)
        if gap <= L.SILL_TOUCH_GAP + _TOUCH_EPS and not self.touched:
            self.touched, self.touched_at = True, self.steps

    def update(self, unit, block_1, block_2, arms=None) -> Result:
        """One control step. ``arms`` is ``rigstate.arms()``: None (unreadable) lets go of nothing."""
        self.steps += 1
        blocks = (block_1, block_2)
        if not valid(unit, *blocks):  # a broken observation: count it, score it 0, keep the sill record, and
            self.bad_steps += 1  # restart the let-go and stillness counts (nothing can be judged at rest from it)
            self._forget_hold()
            if valid(unit):  # the unit alone is readable: a touch still counts
                self._sill(unit)
            self.last = self.result(unit, *blocks)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (unit, *blocks)]
        if self.prev is not None and self._teleported(now) and self.at_start(unit, blocks):
            self.reset()  # the simulator put everything back on its starting spots: a new episode
            self.steps = 1
        self.prev = now
        self._hold((unit, *blocks), arms)
        self._sill(unit)
        self.last = self.result(unit, *blocks)
        return self.last

    def result(self, unit, block_1, block_2) -> Result:
        blocks = (block_1, block_2)
        if not valid(unit, *blocks):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "touched": self.touched})
        tilt, off = unit_tilt_deg(unit), centre_off(unit)
        stand = standing(unit)
        settled = self.settled()  # unit, block_1, block_2: let go and still for the last 1 s
        # resting on a block: the geometry, with the unit AND that block let go and still (a held or hovering unit,
        # or one on a held block, rests on nothing)
        on = [bool(settled[0] and settled[1 + i] and on_block(unit, b)) for i, b in enumerate(blocks)]
        set_tight = both_set(blocks, L.BLOCK_TIGHT, settled[1:])
        set_loose = both_set(blocks, L.BLOCK_LOOSE, settled[1:])
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
        success = bool(score == 1.0)  # the 1.0 conditions already need everything let go and still for 1 s
        details = {
            "valid_state": True,
            "let_go_steps": list(self.free),
            "still": list(self.still),
            "settled": settled,
            "arms_unreadable": self.arms_unreadable,
            "hand_gap_mm": [None if g is None else _mm(g) for g in self.gaps],
            "touched": self.touched,
            "touched_at": self.touched_at,
            "sill_gap_mm": _mm(sill_gap(unit)),
            "min_sill_gap_mm": _mm(self.min_sill_gap),
            "in_rebate": in_rebate(unit),
            "stop_gap_mm": _mm(stop_gap(unit)),
            "standing": stand,
            "unit_tilt_deg": round(tilt, 2),
            "unit_off_mm": _mm(off),
            "unit_low_mm": _mm(unit_low(unit) - L.SILL_TOP),  # lowest corner above the sill's top
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
