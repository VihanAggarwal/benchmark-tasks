"""Every dimension and threshold of the task, in one place, and the lock's geometry as plain functions.

OXO Good Grips 9-inch locking tongs (no. 28481): 230 mm long with the tab in and 135 g (catalogue figures). The other
sizes, masses, forces and the lock's travel below are our estimates of that pair (said once, in task.json fidelity).

World frame (station): x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at
x 0, y +-0.305 and the worktop (the station's table top) is at z 0.7475.

Tongs frame, in millimetres: u along the arm that carries the pin (toward the tips), w across the two arms (the way
they open), v along the hinge pin. The tongs lie on their side, so v is up. The pin is at u 0, w 19. Mechkit's base
frame of the assembly: x_b = w - 19, y_b = u, z_b = v + 15 (all in mm here, metres in the kit), so the pin axis is the
base z axis and the worktop is z_b = 0. Every shape is given at the SHUT pose (u, v, w ranges); the free arm is then
drawn OPEN by turning its shapes 27.5 deg about the pin (opening is a negative turn about base z).
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station worktop
MM = 0.001

# -- the tongs ------------------------------------------------------------------------------------------------------
LENGTH_TAB_IN = 0.230  # catalogue: 9 in
MASS = 0.135  # catalogue: 4 3/4 oz
PIN_ARM_MASS, FREE_ARM_MASS, TAB_MASS = 0.061, 0.054, 0.020
PIN_W = 19.0  # mm: the pin is at u 0, w 19
PIN_TO_TIP = 176.0  # mm: the pin to the tips (the arms' back ends are 24 mm behind the pin)
OPEN_DEG = 27.5  # the spring holds the arms this far open (heads meeting = shut)
OPEN = math.radians(OPEN_DEG)  # 0.480 rad: the hinge's travel (q 0 open as drawn, +OPEN shut)
# Shapes at the SHUT pose: (u0, u1, v0, v1, w0, w1) in mm.
PIN_ARM_BOXES = {
    "bar": (-24.0, 146.0, -10.0, 10.0, 0.0, 6.0),  # the channel arm, 20 x 6 mm
    "head": (146.0, 176.0, -15.0, 15.0, 0.0, 14.0),  # the scalloped head
    "ear_up": (-6.0, 6.0, 10.0, 15.0, 0.0, 23.0),  # the two hinge ears round the pin
    "ear_down": (-6.0, 6.0, -15.0, -10.0, 0.0, 23.0),
}
FREE_ARM_BOXES = {
    "bar": (-24.0, 146.0, -10.0, 10.0, 22.0, 28.0),
    "head": (146.0, 176.0, -15.0, 15.0, 14.0, 28.0),
    "lug_up": (-24.0, -14.0, 5.5, 10.0, 17.6, 22.0),  # the two heel lugs the lock block slides under
    "lug_down": (-24.0, -14.0, -10.0, -5.5, 17.6, 22.0),
}
TAB_BOXES = {  # tab in
    "block": (-8.0, 2.0, -9.0, 9.0, 6.0, 16.6),  # the steel lock block
    "stem": (-29.0, -8.0, -3.0, 3.0, 6.0, 9.0),
    "ring_far": (-54.0, -48.0, -6.0, 6.0, 4.0, 12.0),  # the ring: two ends and two sides round a 13 x 12 hole
    "ring_near": (-35.0, -29.0, -6.0, 6.0, 4.0, 12.0),
    "ring_side_up": (-54.0, -29.0, 6.0, 12.0, 4.0, 12.0),
    "ring_side_down": (-54.0, -29.0, -12.0, -6.0, 4.0, 12.0),
}
RING_U = (-54.0, -29.0)  # tab in
RING_W = (4.0, 12.0)
RING_V = (-12.0, 12.0)
RING_CENTRE_UW = (-41.5, 8.0)
BLOCK_U = (-8.0, 2.0)  # tab in
BLOCK_W = (6.0, 16.6)
LUG_U = (-24.0, -14.0)  # shut
LUG_W = (17.6, 22.0)
TAB_TRAVEL = 0.012  # m: the tab's stroke, hard stops at 0 and 12 mm
TAB_GAP_IN = 0.005  # m: from the arms' hinge end to the ring, tab in (17 mm at the stop)

# -- joints ---------------------------------------------------------------------------------------------------------
SPRING_RATE = 0.20 / OPEN  # 0.417 N m/rad
SPRING_TARGET = -OPEN  # rad: 0.20 N m at the open stop, 0.40 N m shut
HINGE_DAMPING = 0.03  # N m s/rad
TAB_FRICTION = 1.0  # N
TAB_DAMPING = 2.0  # N s/m
FRICTION = 0.6  # every tongs surface (static and dynamic)

# -- the start pose -------------------------------------------------------------------------------------------------
# The tongs lie with their tips toward the robot (-x) and the tab away from it, so that the gripper on the arms works
# near its own base line and the one on the ring 14 cm further out: two straight-down grippers 14 cm apart along the
# tongs keep their wrists apart only this way round (the arms' collision capsules stay 25 mm or more apart while both
# are down; with the tongs along y they overlap).
ORIGIN = (0.190, 0.005, TOP)  # the assembly's base frame: on the pin axis, at the worktop
YAW_HELD = math.pi / 2  # held shut: the pin arm along -x, the arms' centre plane on y 0
YAW = YAW_HELD + OPEN / 2  # open as it lies: the V symmetric about y 0 (103.75 deg)

# -- the scorer (scoring.py) ----------------------------------------------------------------------------------------
CONTROL_HZ = 20.0
SHUT_TIP_GAP = 0.014  # m: arms shut (tip gap at most 14 mm; 84 open, 9 resting on the lock)
SQUEEZE_S = 1.0  # shut for this long at some point of the trial: SQUEEZED SHUT (latched)
ENGAGED_TAB_GAP = 0.016  # m: the tab within 1 mm of its stop (tab gap 17 mm at the stop)
LET_GO = 0.030  # m: every finger zone this far from every part of the tongs
STILL_WINDOW_S = 1.0
STILL_PATH = 0.002  # m: the root body's path over the last 1 s
STILL_TURN_DEG = 2.0  # its summed turn over the last 1 s
STILL_GAP = 0.001  # m: max - min of the tip gap and of the tab gap over the last 1 s
ON_TOP_TOL = 0.001  # m: the lowest point within 1 mm of the worktop
FALLEN_DEPTH = 0.010  # m: a part more than 1 cm below the worktop level has fallen
EPS = 1e-9
EPS_DEG = 1e-6
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): the tongs back on their start pose,
# open, tab in, after a jump no trial can make in one control step.
RESET_START_TOL = 0.005
RESET_START_TURN_DEG = 2.0
RESET_TAB_JUMP = 0.005
RESET_JUMP = 0.020
RESET_TURN_DEG = 10.0

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
STEEL = (0.78, 0.78, 0.80)
BLACK = (0.06, 0.06, 0.06)
DARK_STEEL = (0.45, 0.45, 0.48)


# == geometry in the tongs frame (mm) =================================================================================


def tip_gap(theta: float) -> float:
    """Gap (m) between the inner faces of the two tips with the arms ``theta`` rad from shut."""
    return 2.0 * PIN_TO_TIP * MM * math.sin(max(0.0, theta) / 2.0)


def theta_of(q_hinge: float) -> float:
    """Angle from shut (rad) for a hinge reading (0 open as drawn, +OPEN shut)."""
    return OPEN - float(q_hinge)


def tab_gap(q_tab: float) -> float:
    """Gap (m) from the arms' hinge end to the ring for a tab reading (0 in, 0.012 at the stop)."""
    return TAB_GAP_IN + float(q_tab)


def open_free(points_uw, theta: float) -> np.ndarray:
    """Points of the free arm (shut-pose u, w in mm) with the arms ``theta`` rad from shut."""
    p = np.atleast_2d(np.asarray(points_uw, dtype=float)) - np.array([0.0, PIN_W])
    c, s = math.cos(theta), math.sin(theta)
    return np.array([0.0, PIN_W]) + np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]]


def rect(u0, u1, w0, w1) -> np.ndarray:
    return np.array([(u0, w0), (u1, w0), (u1, w1), (u0, w1)], dtype=float)


def lug_polygon(theta: float) -> np.ndarray:
    """The heel lugs' outline (side view, u-w, mm) with the arms ``theta`` rad from shut."""
    return open_free(rect(*LUG_U, *LUG_W), theta)


def block_rect(t_mm: float) -> np.ndarray:
    """The lock block's outline (side view, u-w, mm) with the tab ``t_mm`` out."""
    return rect(BLOCK_U[0] - t_mm, BLOCK_U[1] - t_mm, *BLOCK_W)


def _clip(poly: np.ndarray, axis: int, bound: float, keep_below: bool) -> np.ndarray:
    """Sutherland-Hodgman: the part of a convex polygon with coordinate ``axis`` below (or above) ``bound``."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina = a[axis] <= bound if keep_below else a[axis] >= bound
        inb = b[axis] <= bound if keep_below else b[axis] >= bound
        if ina:
            out.append(a)
        if ina != inb:
            f = (bound - a[axis]) / (b[axis] - a[axis])
            out.append(a + f * (b - a))
    return np.array(out) if out else np.zeros((0, 2))


def travel_limit(theta: float) -> float:
    """How far (m) the tab can be drawn with the arms ``theta`` rad from shut: until the block's back face meets the
    part of a lug that hangs below the block's top. 0.012 when nothing is in the way."""
    part = _clip(lug_polygon(theta), 1, BLOCK_W[1], keep_below=True)
    part = _clip(part, 1, BLOCK_W[0], keep_below=False) if len(part) else part
    if not len(part):
        return TAB_TRAVEL
    u_max = float(part[:, 0].max())
    return float(min(TAB_TRAVEL, max(0.0, (BLOCK_U[0] - u_max) * MM)))


def _overlap(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """Two convex polygons overlap with positive depth (separating-axis test)."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() + tol or pb.max() <= pa.min() + tol:
                return False
    return True


_REST_CACHE: dict[float, tuple[float, bool]] = {}


def rest_angle(t: float, step_deg: float = 0.05) -> tuple[float, bool]:
    """Where arms let go come to rest with the tab ``t`` m out: (theta rad from shut, on the block). A lug that comes
    down on the block's top stops the arms there (the lock); one that meets the block's back face pushes the tab in
    (its 1 N friction against about 12 N) and the arms spring fully open. The block's edge is reached from 5.9 mm.
    A scan in ``step_deg`` steps finds the first overlap, then 40 bisections place the contact."""
    key = round(float(t), 6)
    if key in _REST_CACHE:
        return _REST_CACHE[key]
    blk = block_rect(key / MM)
    n = int(math.ceil(OPEN_DEG / step_deg))
    result = (OPEN, False)
    for k in range(1, n + 1):
        th = math.radians(min(OPEN_DEG, k * step_deg))
        if not _overlap(lug_polygon(th), blk):
            continue
        lo, hi = math.radians((k - 1) * step_deg), th
        for _ in range(40):
            mid = (lo + hi) / 2
            if _overlap(lug_polygon(mid), blk):
                hi = mid
            else:
                lo = mid
        # On the block when the part of the lug over the block's length is above its top at first contact.
        lug = lug_polygon(hi)
        over = _clip(_clip(lug, 0, blk[0, 0], keep_below=False), 0, blk[1, 0], keep_below=True)
        on_top = len(over) > 0 and float(over[:, 1].min()) >= BLOCK_W[1] - 0.05
        result = (lo, True) if on_top else (OPEN, False)
        break
    _REST_CACHE[key] = result
    return result


def lock_angle() -> float:
    """The rest angle with the tab at its stop (rad from shut): 2.86 deg."""
    return rest_angle(TAB_TRAVEL)[0]


def heel_clearance_open() -> tuple[float, float]:
    """Open arms over the parked block (tab in): (vertical gap mm over the block, normal gap mm from the block's
    back top corner to the free arm's underside)."""
    bar = open_free(np.array([(u, 22.0) for u in np.linspace(-24.0, 30.0, 5401)]), OPEN)
    over = bar[(bar[:, 0] >= BLOCK_U[0]) & (bar[:, 0] <= BLOCK_U[1])]
    vertical = float(over[:, 1].min() - BLOCK_W[1])
    a, b = open_free(np.array([(-24.0, 22.0), (30.0, 22.0)]), OPEN)
    n = np.array([-(b - a)[1], (b - a)[0]])
    n /= np.linalg.norm(n)
    normal = float(abs((np.array([BLOCK_U[0], BLOCK_W[1]]) - a) @ n))
    return vertical, normal


def outer_span(u: float, theta: float) -> float:
    """Distance (mm) from the pin arm's outer face (w 0) to the free arm's outer face, measured across the pin arm at
    distance ``u`` from the pin, arms ``theta`` rad from shut (28 mm shut)."""
    p = open_free(np.array([(u, 28.0)]), theta)[0]
    return float(p[1])


def theta_for_span(u: float, gap_mm: float) -> float:
    """The widest opening (rad from shut) whose outer span at ``u`` fits between pads ``gap_mm`` apart (0: shut)."""
    if gap_mm >= outer_span(u, OPEN):
        return OPEN
    lo, hi = 0.0, OPEN
    for _ in range(50):
        mid = (lo + hi) / 2
        if outer_span(u, mid) <= gap_mm:
            lo = mid
        else:
            hi = mid
    return lo


# == world poses =====================================================================================================


def base_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_world(uw, origin_xy=ORIGIN[:2], yaw=YAW) -> np.ndarray:
    """Pin-arm (root) frame points (u, w in mm) to world (x, y) in m."""
    uw = np.atleast_2d(np.asarray(uw, dtype=float))
    xb = np.c_[(uw[:, 1] - PIN_W) * MM, uw[:, 0] * MM]
    return np.asarray(origin_xy, dtype=float) + xb @ base_rotation(yaw).T


def to_root(xy, origin_xy=ORIGIN[:2], yaw=YAW) -> np.ndarray:
    """World (x, y) in m to root-frame (u, w) in mm."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float)) - np.asarray(origin_xy, dtype=float)
    b = xy @ base_rotation(yaw)
    return np.c_[b[:, 1] / MM, b[:, 0] / MM + PIN_W]


# == the demo's geometry (task.py) ====================================================================================

HOLD_U = 90.0  # mm from the pin: where the right gripper pinches the arms (4.4 N against the spring there)
RING_GRIP_U = -50.0  # mm (tab in): where the left pads are centred on the ring, 8.5 mm past its centre
CENTRE_W = 14.0  # mm: the tongs' centre plane (the arms' inner faces meet there when shut)
HOLD_XY = tuple(float(v) for v in to_world([(HOLD_U, CENTRE_W)], yaw=YAW_HELD)[0])  # (0.100, 0.000)
RING_XY_HELD = tuple(float(v) for v in to_world([RING_CENTRE_UW], yaw=YAW_HELD)[0])  # (0.2315, -0.006), tab in
RING_GRIP_XY = tuple(float(v) for v in to_world([(RING_GRIP_U, RING_CENTRE_UW[1])], yaw=YAW_HELD)[0])  # (0.240, -0.006)
PAD_W = 27.0  # mm along the tongs (the pads' width)
GRIPPER_OPEN_GAP = 94.0  # mm
