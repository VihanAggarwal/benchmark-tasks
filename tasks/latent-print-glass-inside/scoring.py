"""The scorer, as plain functions of the glass's state and the arms' state. NumPy only; no simulator needed.

Most of the rule is about the whole trial, not only its end: the glass may be touched only on its inside, its
outside may never touch anything, and it may be set down only on its start mark, the exam sheet or the box floor.
So the scorer keeps a little state (which rule was broken and when; whether the glass was ever lifted and ever stood
on the exam sheet) and is fed one observation per control step by ``task.success()``.

    Tracker.update(glass, arms) -> Result(score, success, details)

``glass`` is anything with ``pos`` (the centre of its world-aligned bounding box, which for this square open box is
its geometric centre at any orientation), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` (and mechkit's AssemblyState) has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open, pads 93.9 mm apart), or ``None`` when the arms could not be read; then the rules about the grippers are
not applied for that step, 'both grippers let go' is taken as met, and the step is counted in ``arms_unavailable``;
if that is more than 1 % of the steps, the trial scores 0 (the finger rule could not be judged: fail closed).
An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves
the tracker as it was.

The fingers. Each finger's lowest 38 mm is modelled as a block 10 mm thick outside the gap between the pads and 40 mm
wide (the fingers' size 19-38 mm above their tips on yam_arena's collision meshes; 6.4 mm and 27 mm at the tips),
running from the point between the finger tips back up the gripper's axis, tilted as the gripper is (rigstate gives
the tool point and the tip point, so the axis is known). rigstate gives no turn of the gripper about that axis, so
the scorer judges both fingers' blocks at every turn from 0 to 175 degrees in 5 degree steps (relative to the
glass's sides; the pair looks the same turned half a turn), in the glass's own frame:

- 'above': at no turn can any part of a block reach below the rim (the axis's lowest point more than r sin(tilt)
  above it; r: the zone's radius about the axis): not judged;
- 'away': at every turn every part of both blocks is at least 10 mm from the glass (outside a box 10 mm bigger than
  the glass on every side and 10 mm under its bottom, up to the rim). Tested on the round zone the two fingers sweep
  about the axis, so it holds whatever the real turn;
- 'inside': both fingers, wherever they are below the rim, lie in the opening (within 5 mm of the inner walls, which
  fingers pressing them reach, and above the inner floor): at some turn if the tip point is in the glass (below the
  rim, in the opening: the fingers came in through the mouth and its walls hold them to a turn that fits), else at
  every turn. For this test each finger's block runs on
  up to its carriage, 70 mm above the tips: a finger can only be in the glass through its mouth, so all of it that
  is below the rim must be in the opening (a gripper lying on its side cannot hook one finger in over the rim). A
  finger cannot be in the acrylic, so fingers below the rim in the opening's column are in the glass. For a tilted
  gripper the fingers are judged up to r sin(tilt) above the rim too, so that no turn hides one swinging down over
  the outside;
- otherwise 'outside': a finger is on the glass's outside, or within 10 mm of it below the rim: the trial scores 0.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_HALF = np.array([L.GLASS_HALF, L.GLASS_HALF, L.GLASS_HALF_H])
HEADINGS = np.radians(np.arange(0.0, 180.0 - 1e-9, L.HEADING_STEP_DEG))  # 0, 5, .., 175 deg
ARMS = ("left", "right")


def _rect(cx: float, cy: float, sx: float, sy: float) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a rectangle of size (sx, sy) centred on (cx, cy)."""
    return (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2)


MARK = _rect(L.START_X, L.START_Y, L.MARK_SIZE[0], L.MARK_SIZE[1])
SHEET = _rect(L.SHEET_X, L.SHEET_Y, L.SHEET_SIZE[0], L.SHEET_SIZE[1])
BOX_FLOOR = _rect(L.BOX_X, L.BOX_Y, L.BOX_INNER, L.BOX_INNER)
_BI, _BO = L.BOX_INNER_HALF, L.BOX_OUTER_HALF
BOX_WALLS = {  # plan-view rectangles of the evidence box's four walls (front and back run its full width)
    "front": (L.BOX_X + _BI, L.BOX_X + _BO, L.BOX_Y - _BO, L.BOX_Y + _BO),
    "back": (L.BOX_X - _BO, L.BOX_X - _BI, L.BOX_Y - _BO, L.BOX_Y + _BO),
    "left": (L.BOX_X - _BI, L.BOX_X + _BI, L.BOX_Y + _BI, L.BOX_Y + _BO),
    "right": (L.BOX_X - _BI, L.BOX_X + _BI, L.BOX_Y - _BO, L.BOX_Y - _BI),
}


# -- states ---------------------------------------------------------------------------------------------------
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
        for side in ARMS:
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
    """Angle between the glass's own axis (bottom to rim) and the vertical (degrees, 0..180: upside down is 180).
    Not folded: the glass is open at the top, so it has one right way up; it is square, so its turn about its axis
    does not matter anywhere in the rules."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the glass's shape ------------------------------------------------------------------------------------------
def corners(g) -> np.ndarray:
    """The glass's eight outer corners (world, 8 x 3) from its centre and orientation."""
    return np.asarray(g.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(g.quat_xyzw).T


def lowest(g) -> float:
    """Height (world z) of the glass's lowest point."""
    return float(np.min(corners(g)[:, 2]))


def to_local(g, points) -> np.ndarray:
    """World point(s) in the glass's own frame: origin at its centre, z from its bottom face (-40 mm) to its rim
    (+40 mm), x and y along its sides."""
    return (np.asarray(points, dtype=float) - np.asarray(g.pos, dtype=float)) @ rotation(g.quat_xyzw)


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points) -> np.ndarray:
    """Counter-clockwise convex hull of 2-D points (Andrew's monotone chain); duplicates and collinear points go."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def outline(g) -> np.ndarray:
    """The glass's outline seen from above (its 'footprint' when standing): the hull of its corners."""
    return convex_hull(corners(g)[:, :2])


def rect_poly(r) -> np.ndarray:
    x0, x1, y0, y1 = r
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float)


def wholly_on(poly, r) -> bool:
    """Every point of a (convex) outline inside the rectangle r = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = r
    p = np.asarray(poly, dtype=float)
    return bool(np.all(p[:, 0] >= x0) and np.all(p[:, 0] <= x1) and np.all(p[:, 1] >= y0) and np.all(p[:, 1] <= y1))


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(p, q) -> bool:
    """Separating-axis test for two convex polygons (counter-clockwise or clockwise vertex arrays)."""
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]])
            if not np.any(axis):
                continue
            a, b = p @ axis, q @ axis
            if a.max() < b.min() or b.max() < a.min():
                return True
    return False


def poly_gap(p, q) -> float:
    """Distance between two convex polygons seen from above (0 when they overlap)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if not _separated(p, q):
        return 0.0
    best = math.inf
    for a, b in ((p, q), (q, p)):
        for v in a:
            for i in range(len(b)):
                best = min(best, _seg_dist(v, b[i], b[(i + 1) % len(b)]))
    return best


# -- surfaces and the box ---------------------------------------------------------------------------------------
def standing_on(g, r, top: float, poly=None) -> bool:
    """Upright (under 5 degrees), its outline wholly on the rectangle r, its lowest point within 3 mm of ``top``."""
    poly = outline(g) if poly is None else poly
    return bool(tilt_deg(g.quat_xyzw) < L.UPRIGHT_DEG and wholly_on(poly, r) and abs(lowest(g) - top) <= L.REST_TOL)


def in_box(g, poly=None) -> bool:
    return standing_on(g, BOX_FLOOR, L.TOP, poly)


def on_sheet(g, poly=None) -> bool:
    return standing_on(g, SHEET, L.SHEET_TOP, poly)


def box_gap(g, poly=None) -> float:
    """Shortest plan-view distance from the glass's outline to any of the box's walls (0 if over one)."""
    poly = outline(g) if poly is None else poly
    return min(poly_gap(poly, rect_poly(r)) for r in BOX_WALLS.values())


def touches_box(g, poly=None) -> bool:
    """The glass within 2 mm of a box wall seen from above, while its lowest point is less than 2 mm above the
    wall's top: brushing a wall, or resting on one."""
    return bool(lowest(g) < L.BOX_TOP + L.BOX_ABOVE and box_gap(g, poly) < L.BOX_TOUCH)


def set_down_off_limits(g, poly=None) -> bool:
    """The glass's lowest point under 4 mm above the table top while its outline is not wholly on the start mark,
    the exam sheet or the box floor: standing (or being dragged) on the bare table or across an edge."""
    if lowest(g) >= L.TOP + L.SET_DOWN:
        return False
    poly = outline(g) if poly is None else poly
    return not any(wholly_on(poly, r) for r in (MARK, SHEET, BOX_FLOOR))


def off_table(g) -> bool:
    x, y = float(g.pos[0]), float(g.pos[1])
    return bool(lowest(g) < L.TOP - L.OFF_TABLE_DROP or not (L.TABLE_X[0] <= x <= L.TABLE_X[1])
                or not (L.TABLE_Y[0] <= y <= L.TABLE_Y[1]))


# -- the grippers -----------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    """The gap between the pads (m) from rigstate's opening."""
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def axis_of(arm) -> np.ndarray:
    """Unit vector from the tool point to the tip point (straight down for an upright gripper)."""
    u = np.asarray(arm.tips, dtype=float) - np.asarray(arm.tcp, dtype=float)
    n = float(np.linalg.norm(u))
    return u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])


_CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # (8, 3): signs along u, v, a
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(_CORNERS[i] != _CORNERS[j]) == 1]  # 12


def finger_blocks(arm, g) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Both fingers at every turn, in the glass's frame: block corners (turns, 2, 8, 3), each block 10 x 40 mm in
    section and running from the tips 70 mm back up the gripper to the finger carriages (the whole finger: one in
    the glass must be in it through its mouth); the finger zone's axis segment (tip point and 38 mm up) and the
    gripper's axis direction, also in the glass's frame; and the zone's radius about the axis (half the gap plus
    10 mm, and 20 mm across, to the outer corner)."""
    r = rotation(g.quat_xyzw)
    tips = (np.asarray(arm.tips, dtype=float) - np.asarray(g.pos, dtype=float)) @ r
    a = axis_of(arm) @ r  # tool point to tips, glass frame
    ex = np.array([1.0, 0.0, 0.0]) - a * a[0]  # the glass's x side, seen across the gripper's axis
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0.0, 1.0, 0.0]) - a * a[1]
    ex /= np.linalg.norm(ex)
    ey = np.cross(a, ex)
    c, s = np.cos(HEADINGS), np.sin(HEADINGS)
    u = c[:, None] * ex + s[:, None] * ey  # (m, 3) closing axis at each turn
    v = np.cross(a, u)  # across it
    gap = finger_gap(arm)
    t, hw, ln = L.FINGER_THICK, L.FINGER_HALF_WIDTH, L.FINGER_REACH
    corners = []
    for sign in (-1.0, 1.0):
        centre = tips + sign * (gap / 2 + t / 2) * u - (ln / 2) * a  # (m, 3)
        k = (_CORNERS[:, 0, None] * (t / 2)) * u[:, None, :] + (_CORNERS[:, 1, None] * hw) * v[:, None, :] \
            + (_CORNERS[:, 2, None] * (ln / 2)) * a[None, None, :]  # (m, 8, 3)
        corners.append(centre[:, None, :] + k)
    seg = np.array([tips, tips - L.FINGER_LEN * a])
    return np.stack(corners, axis=1), seg, a, math.hypot(gap / 2 + t, hw)


def _below(corners, z_cut) -> tuple[np.ndarray, np.ndarray]:
    """The vertices of each block's part at or below the plane z = z_cut: (points (m, 2, 20, 3), valid mask)."""
    z = corners[..., 2]
    pts = [corners]
    mask = [z <= z_cut]
    for i, j in _EDGES:
        zi, zj = z[..., i], z[..., j]
        cross = (zi - z_cut) * (zj - z_cut) < 0
        f = np.where(cross, (z_cut - zi) / np.where(cross, zj - zi, 1.0), 0.0)
        pts.append((corners[..., i, :] + f[..., None] * (corners[..., j, :] - corners[..., i, :]))[..., None, :])
        mask.append(cross[..., None])
    return np.concatenate(pts, axis=-2), np.concatenate(mask, axis=-1)


def finger_check(arm, g, tol: float = L.INSIDE_TOL) -> dict:
    """Where this gripper's fingers are relative to the glass: 'above', 'away', 'inside' or 'outside' (the rule
    broken); see the module's docstring. ``tol``: how far 'inside' blocks may reach into the inner walls (fingers
    pressing them); offline_model passes 0 to ask whether fingers fit through the mouth."""
    corners, seg, a, radius = finger_blocks(arm, g)
    rim = L.GLASS_HALF_H
    drop = radius * math.sqrt(max(0.0, 1.0 - a[2] * a[2]))  # how far below the axis the fingers can reach, any turn
    if float(np.min(seg[:, 2])) - drop > rim:
        return {"status": "above"}
    # away: the round zone (axis segment, radius) clear of the glass grown by 10 mm (up to its rim)
    big = L.GLASS_HALF + L.OUTSIDE_CLEAR
    pts = seg[0] + np.linspace(0.0, 1.0, 20)[:, None] * (seg[1] - seg[0])
    dx = np.maximum(np.abs(pts[:, 0]) - big, 0.0)
    dy = np.maximum(np.abs(pts[:, 1]) - big, 0.0)
    dz = np.maximum(np.maximum(pts[:, 2] - rim, -rim - L.BELOW_BASE - pts[:, 2]), 0.0)
    clear = float(np.min(np.sqrt(dx * dx + dy * dy + dz * dz))) - radius
    if clear >= 0.0:
        return {"status": "away", "clear_mm": round(clear * 1000, 1)}
    # inside: at some turn both blocks' parts below the rim (plus r sin(tilt) for a tilted gripper) in the opening
    p, m = _below(corners, rim + drop)
    inner = L.INNER_HALF + tol
    floor = -rim + L.GLASS_FLOOR - L.INSIDE_TOL
    ok = (np.abs(p[..., 0]) <= inner) & (np.abs(p[..., 1]) <= inner) & (p[..., 2] >= floor)
    fits = np.all(ok | ~m, axis=(1, 2))  # (m,): every part of both blocks in the opening at that turn
    tip = seg[0]
    tip_in = bool(tip[2] < rim and abs(tip[0]) < L.INNER_HALF and abs(tip[1]) < L.INNER_HALF)
    # fingers whose tip point is in the glass entered through its mouth, so its walls hold them to a turn that fits:
    # one turn that fits is enough. Otherwise (tip point above the rim or off the opening) they must fit at every turn.
    if (np.any(fits) if tip_in else np.all(fits)):
        return {"status": "inside", "turns_ok": int(np.sum(fits))}
    return {"status": "outside"}


def tips_inside(arm, g) -> bool:
    """The point between the finger tips is in the glass: below its rim, above its floor, within the opening."""
    p = to_local(g, arm.tips)
    return bool(-L.GLASS_HALF_H + L.GLASS_FLOOR - 0.002 <= p[2] < L.GLASS_HALF_H
                and abs(p[0]) < L.INNER_HALF and abs(p[1]) < L.INNER_HALF)


def holding(arm, g) -> bool:
    """For the trace only: the gripper's tips in the glass and its fingers opened to within 2 mm per side of the
    inner walls (pads at least 38 mm apart: the outer faces touch the walls at 42 mm)."""
    return bool(tips_inside(arm, g) and finger_gap(arm) >= L.CONTACT_GAP - 0.004)


def carried(g, arms, prev_tips, dt: float) -> bool:
    """Some gripper's tip point is in the glass and moving down with it (its downward speed no more than 0.15 m/s
    short of the glass's): the glass is being lowered, not falling or slipping off the fingers."""
    fall = -float(g.lin_vel[2])
    for side in ARMS:
        a = arms[side]
        down = -(float(np.asarray(a.tips, dtype=float)[2]) - float(prev_tips[side][2])) / dt
        if tips_inside(a, g) and down >= fall - L.FALL_SLACK:
            return True
    return False


def let_go(arm, g) -> bool:
    """The gripper has let go: its tip point 20 mm or more above the rim, or 60 mm or more from the glass's axis."""
    p = to_local(g, arm.tips)
    return bool(p[2] >= L.GLASS_HALF_H + L.LET_GO_ABOVE or math.hypot(p[0], p[1]) >= L.LET_GO_AWAY)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACH_TEXT = {
    "left_fingers_outside": "the left gripper's fingers were on the glass's outside (or within 1 cm of it)",
    "right_fingers_outside": "the right gripper's fingers were on the glass's outside (or within 1 cm of it)",
    "touched_box": "the glass touched the evidence box's walls",
    "tipped": "the glass tilted more than 10 degrees",
    "set_down_off_limits": "the glass was set down (or dragged) off its mark, the exam sheet and the box floor",
    "dropped": "the glass fell faster than 0.45 m/s with no gripper inside it moving down with it",
    "off_table": "the glass left the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # rule -> the control step it was first broken
        self.lifted = False  # the glass's lowest point was 20 mm or more above the table at some step
        self.examined = False  # the glass stood upright and still on the exam sheet at some step
        self.min_box_gap = math.inf  # m: closest the glass came to a box wall while not above the walls
        self.max_fall = 0.0  # m/s: fastest downward speed while not carried down by a gripper
        self.prev_tips = None  # both arms' tip points at the previous control step (None: unknown)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite glass state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, velocity) at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(g) -> bool:
        """The starting layout: the glass upright and still on its mark, within 4.5 cm of its start."""
        return bool(math.hypot(float(g.pos[0]) - L.START_X, float(g.pos[1]) - L.START_Y) < L.START_TOL
                    and abs(lowest(g) - L.MARK_TOP) < 0.003 and tilt_deg(g.quat_xyzw) < 1.0 and is_resting(g))

    def _teleported(self, now) -> bool:
        """The glass moved, in one control step, 5 cm further than its last velocity plus a free fall could have
        carried it: the simulator put it there."""
        p, _ = now
        p0, v0 = self.prev
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def update(self, g, arms=None) -> Result:
        if g is None or not valid(g):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(g, arms)
            return self.last
        now = (np.array(g.pos, dtype=float), np.array(g.lin_vel, dtype=float))
        if self.prev is not None and self._teleported(now) and self.at_start(g):
            self.reset()  # the simulator put the glass back on its mark: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        poly = outline(g)
        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1
        else:
            for side in ARMS:
                if finger_check(arms[side], g)["status"] == "outside":
                    self._flag(f"{side}_fingers_outside")
        if lowest(g) < L.BOX_TOP + L.BOX_ABOVE:
            self.min_box_gap = min(self.min_box_gap, box_gap(g, poly))
        if touches_box(g, poly):
            self._flag("touched_box")
        if tilt_deg(g.quat_xyzw) > L.TILT_MAX_DEG:
            self._flag("tipped")
        if set_down_off_limits(g, poly):
            self._flag("set_down_off_limits")
        if have_arms and self.prev_tips is not None and not carried(g, arms, self.prev_tips, self.dt):
            fall = -float(g.lin_vel[2])
            self.max_fall = max(self.max_fall, fall)
            if fall > L.DROP_SPEED:
                self._flag("dropped")
        self.prev_tips = ({s: np.array(arms[s].tips, dtype=float) for s in ARMS} if have_arms else None)
        if off_table(g):
            self._flag("off_table")
        if lowest(g) >= L.TOP + L.LIFT_HEIGHT:
            self.lifted = True
        if not self.breach and on_sheet(g, poly) and is_resting(g):
            self.examined = True
        self.last = self.result(g, arms, poly)
        return self.last

    def result(self, g, arms=None, poly=None) -> Result:
        if g is None or not valid(g):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        poly = outline(g) if poly is None else poly
        have_arms = arms_valid(arms)
        released = all(let_go(arms[s], g) for s in ARMS) if have_arms else True  # unreadable: taken as met
        boxed, sheeted, resting = in_box(g, poly), on_sheet(g, poly), is_resting(g)
        blind = self.arms_unavailable > L.ARMS_MISSING_MAX * max(self.steps, 1)
        if self.breach or blind:
            score = 0.0
        elif boxed and resting:
            score = 1.0 if (released and self.examined) else 0.75
        elif sheeted and resting:
            score = 0.5
        elif self.lifted:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "in_box": boxed,
            "on_sheet": sheeted,
            "examined": self.examined,
            "lifted": self.lifted,
            "resting": resting,
            "tilt_deg": round(tilt_deg(g.quat_xyzw), 2),
            "low_mm": _mm(lowest(g) - L.TOP),
            "box_gap_mm": _mm(box_gap(g, poly)),
            "min_box_gap_mm": _mm(self.min_box_gap),
            "max_fall_mps": round(self.max_fall, 3),
            "fingers": ({s: finger_check(arms[s], g)["status"] for s in ARMS} if have_arms else None),
            "holding": ({s: holding(arms[s], g) for s in ARMS} if have_arms else None),
            "let_go": ({s: let_go(arms[s], g) for s in ARMS} if have_arms else None),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "arms_blind": blind,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
