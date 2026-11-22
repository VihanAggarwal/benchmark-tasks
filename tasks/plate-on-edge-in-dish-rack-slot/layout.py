"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top (the kitchen counter) is at z 0.7475. Heights below are above the counter unless
named *_W.

On the counter: a fold-flat plastic draining board (Joseph Joseph Flip-Up, 319 x 400 mm) with its plate rack folded
down flat; behind it the sink unit's steel back upstand and the tiled splashback; to its right the sink rim and the
mixer tap. Two washed side plates (Corelle 6.75 in, real dished shape) lean on the upstand to the left of the board,
put there because the rack was still folded. The job: lift the plate rack until it locks upright, then stand each
plate on edge in its own slot between the rack's prongs, and let go.

Figures are sourced (the product listings named in the README) or estimates (est, said in task.json's fidelity): see
the comments.
"""

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the plates (loose) ------------------------------------------------------------------------------------------
# Corelle Livingware Winter Frost White bread and butter plate: 6.75 in (171 mm) across, 0.88 in (22 mm) tall
# (retailer listings), 4.6 oz (0.130 kg, Corelle's listing for the 6.75 in Winter Frost White plate). The dished
# profile below is est: a flat well 110 mm across, a sloping wall, a flat rim (marly) 14 mm wide; 4 mm walls.
PL_D = 0.171
PL_R = PL_D / 2  # 0.0855
PL_DEPTH = 0.022  # back of the well to the front face of the rim
PL_MASS = 0.130
PL_FRICTION = (0.50, 0.40)  # glazed glass on plastic, steel and the counter (est)
PL_WHITE = (0.95, 0.95, 0.94)
WELL_R = 0.055
WELL_T = 0.004  # the well's floor, w 0 .. 0.004
MARLY_R0 = 0.0715  # the flat rim, radius 71.5 .. 85.5 mm, w 0.018 .. 0.022
MARLY_W0, MARLY_W1 = 0.018, 0.022
MARLY_W = (MARLY_W0 + MARLY_W1) / 2  # 0.020: the rim's mid-plane, where the pads pinch
PL_WALL_T = 0.004  # the sloping wall's thickness (not the splashback's WALL_T below)
N_WALL = 16  # boxes round the sloping wall
N_MARLY = 24  # boxes round the rim
W_ORIGIN = PL_DEPTH / 2  # the plate's body frame sits on its axis at mid-depth (w 0.011)
LEAN_DEG = 12.0  # as found: leaning back on the upstand
LEAN = math.radians(LEAN_DEG)
PL_NAMES = ("plate_1", "plate_2")
PL_Y = {"plate_1": 0.265, "plate_2": 0.450}  # centre lines along y at the start (outlines 14 mm apart)

# Plate coordinates: u (in the plane, toward -y as drawn), v (in the plane, up), w (along the axis, from the back of
# the well toward the food side). Upright and facing the robot, (u, v, w) about the origin maps to (x, y, z) =
# (-(w - W_ORIGIN), -u, v): a proper rotation (det +1), so its quaternion is the same turn in the simulator. (Until
# 22 Nov this was (-(w - W_ORIGIN), u, v), a mirror: the round point cloud did not notice, but every collider's
# quaternion was a wrong turn on the GPU, the plates spawned with the well disc across the upstand and were thrown
# flat.) The plates are drawn leaning back LEAN about y (top toward +x): R_LEAN.


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


M_UPRIGHT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (u, v, w) -> (x, y, z), det +1
R_LEAN = _rot_y(LEAN)
R_DRAWN = R_LEAN @ M_UPRIGHT  # (u, v, w) -> the base frame at spawn
AXIS0 = R_DRAWN @ np.array([0.0, 0.0, 1.0])  # the food side's normal at spawn
UP0 = R_DRAWN @ np.array([0.0, 1.0, 0.0])  # the plate's 'up' in its plane at spawn


def profile_front(r):
    """w of the plate's food-side face at radius r (the well's floor, the sloping wall, the rim)."""
    return np.interp(r, [0.0, WELL_R, MARLY_R0, PL_R], [WELL_T, WELL_T, MARLY_W1, MARLY_W1])


def profile_back(r):
    """w of the plate's underside at radius r."""
    return np.interp(r, [0.0, WELL_R, MARLY_R0, PL_R], [0.0, 0.0, MARLY_W0, MARLY_W0])


def _plate_cloud(n_ring: int = 48, radii=(0.0, 0.02, 0.04, WELL_R, 0.0635, MARLY_R0, 0.0785, PL_R)) -> np.ndarray:
    """Points on both faces of the plate in (u, v, w) about the origin: what the scorer uses for its lowest point,
    its bottom and its extents (both faces, rings of 48)."""
    pts = []
    for r in radii:
        k = 1 if r == 0.0 else n_ring
        for i in range(k):
            a = 2 * math.pi * i / k
            u, v = -r * math.cos(a), r * math.sin(a)  # -u: the same world points as before the det fix
            for w in (float(profile_back(r)), float(profile_front(r))):
                pts.append((u, v, w - W_ORIGIN))
    return np.array(pts)


CLOUD_UVW = _plate_cloud()
CLOUD0 = CLOUD_UVW @ R_DRAWN.T  # in the base frame at spawn (state rotation identity)


def plate_points(origin, rot) -> np.ndarray:
    """The plate's surface points in the world (table top z 0) for a body frame at ``origin`` turned ``rot``."""
    return CLOUD0 @ np.asarray(rot, dtype=float).T + np.asarray(origin, dtype=float)


# -- the draining board and its plate rack (fixed board, hinged rack) -----------------------------------------------
# Joseph Joseph Flip-Up draining board (SKU 85139): H 2.4 x W 31.9 x D 40 cm folded (the maker's listing); its plate
# rack 'lifts to hold up to 6 dinner plates', locks into place and folds flat. Laid with its 319 mm width along x and
# its 400 mm depth along y, centred on the midline. The deck height, the prongs' size, pitch and length are est.
BD_X0, BD_X1 = 0.026, 0.345  # 319 mm (26 mm clear of the arm bases' line)
BD_Y0, BD_Y1 = -0.200, 0.200  # 400 mm
DECK = 0.012  # the deck's top over the counter (est; the folded rack lies on it, 20 mm, under the 24 mm listed)
BD_GREY = (0.47, 0.49, 0.52)
RACK_WHITE = (0.86, 0.87, 0.88)
BD_FRICTION = (0.55, 0.45)  # polypropylene (est)
N_PRONGS = 7  # six slots, 'up to 6 plates'
PRONG_PITCH = 0.040
PRONG_T = 0.008  # square section
PRONG_L = 0.095  # from the hinge bar
HINGE_Y = 0.080  # the hinge line (along x) where the prongs rise
HINGE_Z = DECK + PRONG_T / 2  # 0.016
BAR_T = 0.008
RACK_X = 0.210  # the middle prong
PRONG_X = tuple(RACK_X + (k - (N_PRONGS - 1) / 2) * PRONG_PITCH for k in range(N_PRONGS))  # 0.090 .. 0.330
BAR_X0, BAR_X1 = PRONG_X[0] - 0.010, PRONG_X[-1] + 0.010
N_SLOTS = N_PRONGS - 1
SLOT_X = tuple((PRONG_X[k] + PRONG_X[k + 1]) / 2 for k in range(N_SLOTS))  # 0.110 .. 0.310
SLOT_LO = tuple(PRONG_X[k] + PRONG_T / 2 for k in range(N_SLOTS))  # a slot's inner faces
SLOT_HI = tuple(PRONG_X[k + 1] - PRONG_T / 2 for k in range(N_SLOTS))
SLOT_W = PRONG_PITCH - PRONG_T  # 32 mm for a 22 mm deep plate
PRONG_TOP = HINGE_Z + BAR_T / 2 + PRONG_L  # 0.115: a raised prong's top over the counter
RACK_MASS = 0.060  # the rack (bar and seven prongs), est
Q_UP = math.pi / 2  # upright: the lock's notch
Q_STOP = math.radians(92.0)  # the moulded stop, 2 degrees past upright
LOCK_CAPTURE = math.radians(5.0)  # within 5 degrees of upright the lock takes it to upright
LOCK_SPRING = 2.0  # N m/rad: the lock's hold (a rule: see task.py)
LOCK_BIAS = 0.02  # rad: unlocked, the drive target sits this far below the rack's angle, so it sinks back flat
HINGE_DAMPING = 0.02

# -- the upstand, the splashback, the sink rim and the tap (fixed) ----------------------------------------------------
UP_H = 0.075  # 75 mm tall (est)
UP_T = 0.010
UP_FRONT_X = 0.351  # its face toward the robot, 6 mm behind the board
UP_Y0, UP_Y1 = -0.200, 0.600
UP_X = UP_FRONT_X + UP_T / 2
UP_Y = (UP_Y0 + UP_Y1) / 2
UP_STEEL = (0.72, 0.74, 0.76)
WALL_X0 = 0.450  # the tiled splashback's face toward the robot
WALL_T = 0.012
WALL_H = 0.300
WALL_Y0, WALL_Y1 = -0.650, 0.650
WALL_TILE = (0.90, 0.92, 0.93)
SINK_X0, SINK_X1 = 0.100, 0.395  # the sink's steel rim on the counter, right of the board (its bowl is not modelled)
SINK_Y0, SINK_Y1 = -0.620, -0.240
SINK_H = 0.003
SINK_HOLE = (0.030, 0.030)  # rim width (x, y) round the dark bowl opening
SINK_DARK = (0.30, 0.31, 0.33)
TAP_X, TAP_Y = 0.420, -0.430  # mixer tap on the rim's back edge (est 250 mm tall, 25 mm column)
TAP_R = 0.025
TAP_H = 0.250
TAP_SPOUT = 0.150  # its spout reaches this far toward the robot at the top

# -- the rest pose as found -----------------------------------------------------------------------------------------


def lean_rest():
    """(x, z) of a leaning plate's body origin: its underside (the well's back, w 0) through the upstand's top
    front edge, its lowest point on the counter."""
    cz = -float(CLOUD0[:, 2].min())
    a = AXIS0
    # (E - O) . a = -W_ORIGIN for the edge E on the back plane
    cx = UP_FRONT_X - (-W_ORIGIN - (UP_H - cz) * a[2]) / a[0]
    return cx, cz


LEAN_X, LEAN_Z = lean_rest()
_low = CLOUD0[np.argmin(CLOUD0[:, 2])]
FOOT_X = LEAN_X + float(_low[0])  # x of the lowest point (the rim's back corner at the bottom)
_top = CLOUD0[np.argmax(CLOUD0[:, 2])]
TOP_X = LEAN_X + float(_top[0])
TOP_Z = LEAN_Z + float(_top[2])


def upright_rot(lean_deg: float = 0.0) -> np.ndarray:
    """The body rotation (state) of a plate standing facing the robot, its top leaning ``lean_deg`` toward -x
    (negative: toward +x)."""
    return _rot_y(-math.radians(lean_deg)) @ R_LEAN.T


def standing_origin(foot_x: float, y: float, lean_deg: float = 0.0, floor: float = DECK) -> np.ndarray:
    """The body origin of a plate standing on ``floor`` with its lowest point at x ``foot_x``, centre line y."""
    pts = CLOUD0 @ upright_rot(lean_deg).T
    low = pts[np.argmin(pts[:, 2])]
    return np.array([foot_x - low[0], y, floor - low[2]])


def prong_penetration(points, k: int) -> float:
    """How far (m) any of ``points`` is inside raised prong ``k`` (0 if none), the prong a box x_k +- 4 mm, y the
    hinge line +- 4 mm, z from the bar's top to its tip."""
    p = np.asarray(points)
    m = (np.abs(p[:, 1] - HINGE_Y) <= PRONG_T / 2) & (p[:, 2] >= HINGE_Z + BAR_T / 2) & (p[:, 2] <= PRONG_TOP)
    if not np.any(m):
        return 0.0
    dx = PRONG_T / 2 - np.abs(p[m, 0] - PRONG_X[k])
    return float(max(0.0, dx.max()))


def _dense_cloud():
    """A finer grid on the plate's faces (for the lean-in-slot contact only)."""
    pts = []
    for r in np.linspace(0.0, PL_R, 35):
        k = max(1, int(round(2 * math.pi * r / 0.002)))
        for i in range(k):
            a = 2 * math.pi * i / k
            for w in (float(profile_back(r)), float(profile_front(r))):
                pts.append((-r * math.cos(a), r * math.sin(a), w - W_ORIGIN))  # -u: as _plate_cloud
    return np.array(pts) @ R_DRAWN.T


_DENSE = _dense_cloud()


def lean_in_slot(slot: int, foot_x: float, y: float) -> float:
    """The angle (deg, top toward -x) at which a plate standing in ``slot`` with its lowest point at ``foot_x`` comes
    to rest on the near prong (prong ``slot``): the first contact as it tips."""
    def pen(deg):
        o = standing_origin(foot_x, y, deg)
        pts = _DENSE @ upright_rot(deg).T + o
        return prong_penetration(pts, slot)

    lo, hi = 0.0, 45.0
    if pen(lo) > 0:
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if pen(mid) > 0:
            hi = mid
        else:
            lo = mid
    return lo


PLACE_Y = HINGE_Y + 0.050  # the plates' centre lines in the rack: the prongs cross each plate 50 mm off its centre
FOOT_BIAS = -0.007  # the demo sets a plate's lowest point 7 mm toward the robot from the slot's middle: the hanging
# plate's whole 22 mm depth (4 mm in front of its foot, 18 mm behind it) then has 5 mm to each prong


def _lean_check():
    """Each slot's lean at the demo's foot position (all equal by symmetry)."""
    return lean_in_slot(1, SLOT_X[1] + FOOT_BIAS, PLACE_Y)


SLOT_LEAN_DEG = round(_lean_check(), 2)

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
RACK_UP_DEG = 85.0  # the rack counts as up and locked at 85 degrees or more (the lock holds it at 90)
PLACED_LOW = 0.005  # the plate's lowest point within 5 mm of the deck
PLACED_TILT_DEG = 25.0  # its plane at most this far from vertical (it rests on a prong at up to 20.7)
PLACED_Y0, PLACED_Y1 = HINGE_Y + 0.030, HINGE_Y + 0.070  # its centre line 30 to 70 mm beyond the prong row
SLOT_TOL = 0.0005  # its bottom points between a slot's inner faces (0.5 mm allowed)
BOTTOM_BAND = 0.010  # 'its bottom': every surface point within 10 mm of its lowest point
UPRIGHT_DEG = 40.0  # 'on its rim': at most this far from vertical
FALL_SPEED = 0.9  # latched: downward speed of a plate at a step (reached after 41 mm of free fall; read at
# 20 Hz, every free fall of 100 mm or more shows it at some step)
OFF_TABLE = 0.020  # latched: a plate's lowest point this far below the counter
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002  # over the last 1 s no point of a plate or of the rack moves this far (a dot on each plate's top
# rim and on a prong tip, filmed at 30 fps against a mm rule)
RACK_TIP_R = BAR_T / 2 + PRONG_L  # 0.099: a prong tip's distance from the hinge line
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
