"""The scorer, as plain functions of the glass unit's and the two setting blocks' states and of the arms (as
``rigstate.arms()`` gives them). NumPy only; no simulator.

The rule is about the whole trial, not only its end: the glass must never touch the sill. 'Touch' is contact: a part
of the unit inside the channel 0.5 mm or less above the surface under it (the sill's top, or a mark's top where the
unit is over a mark), the same event a chalk print shows on the bench. So the scorer keeps a
little state (whether the unit has ever touched the sill inside the channel, and for each of the
three loose objects how many control steps in a row it has been let go and its recent poses) and is fed one
observation per control step by ``task.success()``.

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


def _edge_samples() -> np.ndarray:
    """Points along the unit's twelve edges (corners included), in units of the half sizes, at most TOUCH_STEP apart."""
    pts = []
    for a in range(3):
        n = int(math.ceil(2 * _UH[a] / L.TOUCH_STEP)) + 1
        t = np.linspace(-1.0, 1.0, n)
        o1, o2 = [b for b in range(3) if b != a]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                p = np.zeros((n, 3))
                p[:, a], p[:, o1], p[:, o2] = t, s1, s2
                pts.append(p)
    return np.vstack(pts)


_EDGE_LOCAL = _edge_samples() * _UH  # (N, 3), the unit's body frame


def _surface_under(xy: np.ndarray) -> np.ndarray:
    """Height (z) of the surface under each of the points (N, 2): a mark's top over a mark, else the sill's top."""
    on_mark = np.zeros(len(xy), dtype=bool)
    eps = 1e-6  # a bottom edge exactly on a mark's edge (a unit as wide as the mark) is over the mark
    for my in L.MARK_YS:
        on_mark |= ((np.abs(xy[:, 0] - L.FRAME_X) <= L.MARK_SIZE[0] / 2 + eps)
                    & (np.abs(xy[:, 1] - my) <= L.MARK_SIZE[1] / 2 + eps))
    return np.where(on_mark, L.MARK_TOP, L.SILL_TOP)


_CORNER_LOCAL = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * _UH
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]  # the twelve edges


def _lowest_above(xy, r: np.ndarray, c: np.ndarray) -> float:
    """z of the unit's lowest point on the vertical line through (x, y); inf when the line misses the box."""
    b = r[2, :]  # the vertical direction in the body frame
    a = np.array([xy[0] - c[0], xy[1] - c[1], -c[2]]) @ r  # the point (x, y, 0) in the body frame
    lo, hi = -math.inf, math.inf
    for i in range(3):
        if abs(b[i]) < 1e-12:
            if abs(a[i]) > _UH[i] + 1e-10:
                return math.inf
            continue
        t1, t2 = (-_UH[i] - a[i]) / b[i], (_UH[i] - a[i]) / b[i]
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    return lo if lo <= hi else math.inf  # z = t along the vertical (the line is x, y fixed, z = t)


def _mark_gaps(unit) -> np.ndarray:
    """Exact contact candidates at the outline of each mark, where the surface under the unit steps up by 1 mm: the
    points where each of the unit's twelve edges crosses the vertical plane of each of the mark's four sides (inside
    that side), and the unit's lowest point straight above each of the mark's four corners. Gap = z - mark's top."""
    r, c = rotation(unit.quat_xyzw), np.asarray(unit.pos, dtype=float)
    v = c + _CORNER_LOCAL @ r.T
    pa = v[[e[0] for e in _EDGES]]
    pb = v[[e[1] for e in _EDGES]]
    hx, hy = L.MARK_SIZE[0] / 2, L.MARK_SIZE[1] / 2
    out = []
    for my in L.MARK_YS:
        sides = ((0, L.FRAME_X - hx, 1, my - hy, my + hy), (0, L.FRAME_X + hx, 1, my - hy, my + hy),
                 (1, my - hy, 0, L.FRAME_X - hx, L.FRAME_X + hx), (1, my + hy, 0, L.FRAME_X - hx, L.FRAME_X + hx))
        for ax, val, other, lo, hi in sides:
            d = pb[:, ax] - pa[:, ax]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (val - pa[:, ax]) / d
            ok = np.isfinite(t) & (t >= 0.0) & (t <= 1.0)
            p = pa[ok] + t[ok, None] * (pb[ok] - pa[ok])
            p = p[(p[:, other] >= lo - 1e-9) & (p[:, other] <= hi + 1e-9)]
            out.extend((p[:, 2] - L.MARK_TOP).tolist())
        for sx in (-1, 1):
            for sy in (-1, 1):
                z = _lowest_above((L.FRAME_X + sx * hx, my + sy * hy), r, c)
                if math.isfinite(z):
                    out.append(z - L.MARK_TOP)
    return np.array(out, dtype=float)


def sill_gaps(unit) -> np.ndarray:
    """Height above the surface under it (the sill's top, or a mark's top over a mark) of every sampled point of the
    unit's edges (its eight corners and points every 5 mm or finer along the twelve edges) that lies inside the
    channel's outline (2 mm slack), plus the exact candidates at each mark's outline (where its 4 sides cross the
    unit's edges, and the unit's lowest point above each mark corner; ``_mark_gaps``). Empty when none exists."""
    r, c = rotation(unit.quat_xyzw), np.asarray(unit.pos, dtype=float)
    pts = c + _EDGE_LOCAL @ r.T
    x0, x1, y0, y1 = _CHANNEL
    k = L.TOUCH_SLACK
    keep = (pts[:, 0] >= x0 - k) & (pts[:, 0] <= x1 + k) & (pts[:, 1] >= y0 - k) & (pts[:, 1] <= y1 + k)
    pts = pts[keep]
    return np.concatenate([pts[:, 2] - _surface_under(pts[:, :2]), _mark_gaps(unit)])


def touches_sill(unit) -> bool:
    """The unit is in contact with the sill: some point of its corners or edges inside the channel's outline (2 mm
    slack) is 0.5 mm or less above the surface under it (a mark's top over a mark, else the sill's top), counting
    the exact points at each mark's outline. Edges, not only corners: a unit lowered with no blocks rests on the two
    1 mm marks along its bottom edges while its corners hang 1 mm above the bare sill, and that is a touch. A unit
    standing on blocks set on the marks is about 9 mm up."""
    g = sill_gaps(unit)
    return bool(g.size and g.min() <= L.SILL_TOUCH_GAP)


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
    lowest corner within 1 mm of the sill's or a mark's top (1 mm below to 2 mm above the sill's top), its centre
    inside the channel's outline. Geometry only: the tracker adds 'let go' and 'still'."""
    low = float(corners(block, _BH)[:, 2].min())
    return bool(
        folded_tilt_deg(block.quat_xyzw) < L.BLOCK_FLAT_DEG
        and L.BLOCK_LOW_BAND[0] <= low <= L.BLOCK_LOW_BAND[1]
        and _inside_channel(np.asarray(block.pos, dtype=float)[:2], 0.0)
    )


def mark_offsets(block) -> tuple[float, float]:
    """Horizontal distance (m) of the block's centre from the right (y -0.05) and the left (y +0.05) mark's centre."""
    return tuple(float(np.hypot(block.pos[0] - L.FRAME_X, block.pos[1] - my)) for my in L.MARK_YS)


def block_set_on(block, tol: float, settled: bool = True) -> int | None:
    """Which mark (0 right, 1 left) the block is set on: on the sill in the channel, square within 15 degrees,
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
    """The unit's geometry on the block: the block lies flat on the sill in the channel, at least half of its top is
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
        self.touched = False  # the unit has been in contact with the sill (or a mark) inside the channel
        self.touched_at = None  # the step it first did
        self.min_low = math.inf  # the lowest any corner of the unit has been inside the channel (z)
        self.min_gap = math.inf  # the lowest any sampled point of its edges has been above the surface under it
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

    def update(self, unit, block_1, block_2, arms=None) -> Result:
        """One control step. ``arms`` is ``rigstate.arms()``: None (unreadable) lets go of nothing."""
        self.steps += 1
        blocks = (block_1, block_2)
        if not valid(unit, *blocks):  # a broken observation: count it, score it 0, keep the sill record, and
            self.bad_steps += 1  # restart the let-go and stillness counts (nothing can be judged at rest from it)
            self._forget_hold()
            if valid(unit) and touches_sill(unit) and not self.touched:  # the unit alone is readable: still a touch
                self.touched, self.touched_at = True, self.steps
            self.last = self.result(unit, *blocks)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (unit, *blocks)]
        if self.prev is not None and self._teleported(now) and self.at_start(unit, blocks):
            self.reset()  # the simulator put everything back on its starting spots: a new episode
            self.steps = 1
        self.prev = now
        self._hold((unit, *blocks), arms)
        inside = [p for p in corners(unit, _UH) if _inside_channel(p[:2], L.TOUCH_SLACK)]
        if inside:
            self.min_low = min(self.min_low, min(float(p[2]) for p in inside))
        gaps = sill_gaps(unit)
        if gaps.size:
            self.min_gap = min(self.min_gap, float(gaps.min()))
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
            "in_channel": in_channel(unit),
            "standing": stand,
            "unit_tilt_deg": round(tilt, 2),
            "unit_off_mm": _mm(off),
            "unit_low_mm": _mm(unit_low(unit) - L.SILL_TOP),  # lowest corner above the sill's top
            "min_low_in_channel_mm": _mm(self.min_low - L.SILL_TOP),
            "min_gap_in_channel_mm": _mm(self.min_gap),
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
