"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

Six USDA large eggs stand pointed end down in a 30-pocket egg flat on the left (a flat partly emptied: only six pockets
hold an egg). In the middle stands a half-dozen (2 x 3) moulded-pulp egg carton, held in a pack-out cradle (fixed), its
lid hinged along the side nearest the arms and standing open at 110 degrees against its hinge stop. The job: stand
each egg pointed end down in a carton cell, close the lid, let go.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the egg (USDA large, loose) ------------------------------------------------------------------------------------
EGG_L = 0.057  # length (USDA large average, about 2.25 in)
EGG_B = 0.022  # greatest radius: 44 mm wide
EGG_MASS = 0.057  # a large egg weighs about 2 oz
EGG_SLICES = 12  # the shell is a stack of 12 discs turned on the egg's axis (a stepped egg)
EGG_SKEW = 0.12  # the greatest girth sits 32 mm from the pointed end (blunt end larger): est from a profile curve
EGG_FRICTION = (0.45, 0.40)  # shell on pulp and on the grip pads' rubber (est)
EGG_COLOR = (0.96, 0.93, 0.86)  # white-shelled
EGG_TOP_COLOR = (0.93, 0.87, 0.74)  # the blunt-end half is tinted so the two ends read apart in the screenshot


def _profile(tau: float) -> float:
    return math.sqrt(max(0.0, 1.0 - tau * tau)) * (1.0 + EGG_SKEW * tau)


_PMAX = max(_profile(-1.0 + 2.0 * i / 4000) for i in range(4001))


def egg_radius(u: float) -> float:
    """The egg's radius (m) at height u above its pointed end (0..EGG_L)."""
    return EGG_B * _profile(2.0 * u / EGG_L - 1.0) / _PMAX


SLICE_H = EGG_L / EGG_SLICES
SLICE_U0 = [k * SLICE_H for k in range(EGG_SLICES)]  # bottom of each disc, from the pointed end
SLICE_R = [egg_radius((k + 0.5) * SLICE_H) for k in range(EGG_SLICES)]
SLICE_Z = [(k + 0.5) * SLICE_H - EGG_L / 2 for k in range(EGG_SLICES)]  # disc centres in the egg's frame
# the egg's frame sits at mid-length; its +z axis runs from the pointed end to the blunt end

# -- a moulded-pulp pocket: a square funnel of four 2 mm slabs -----------------------------------------------------
PULP_T = 0.002  # wall thickness (est)
PULP_COLOR = (0.62, 0.55, 0.45)  # grey-brown pulp
PULP_FRICTION = (0.50, 0.45)  # pulp (est)
FLOOR_T = 0.002  # the pocket floors sit on a 2 mm sheet


def funnel_slabs(cx: float, cy: float, R: float, r: float, D: float, zf: float, T: float = PULP_T):
    """The four slabs of one square funnel: inner half width r at height zf (its floor), R at zf + D (its rim).
    Returns [(centre, size, ypr)] in the caller's frame, for mechkit boxes."""
    a = math.atan2(R - r, D)
    ls = math.hypot(D, R - r)
    w = 2.0 * (R + 0.003)
    rm = (R + r) / 2.0
    zc = zf + D / 2.0 - (T / 2.0) * math.sin(a)
    off = rm + (T / 2.0) * math.cos(a)
    return [
        ((cx + off, cy, zc), (T, w, ls), (0.0, a, 0.0)),
        ((cx - off, cy, zc), (T, w, ls), (0.0, -a, 0.0)),
        ((cx, cy + off, zc), (w, T, ls), (0.0, 0.0, -a)),
        ((cx, cy - off, zc), (w, T, ls), (0.0, 0.0, a)),
    ]


def seat_height(R: float, r: float, D: float) -> float:
    """How high the egg's pointed end stands above the pocket floor (m): the egg drops until the lower edge of one of
    its discs meets the funnel wall (a 4-sided funnel: inner half width r + (R - r) h / D at height h), else it
    rests on the floor."""
    m = (R - r) / D
    h0 = 0.0
    for k in range(EGG_SLICES):
        h0 = max(h0, (SLICE_R[k] - r) / m - SLICE_U0[k])
    return h0


# -- the carton (fixed in its cradle), half-dozen 2 x 3 -------------------------------------------------------------
CX, CY = 0.26, 0.05  # the carton's centre (x, y)
C_PITCH = 0.052  # cell pitch: closed carton about 108 x 160 mm (est from a typical 6-count carton: measure yours)
C_LX, C_LY = 0.108, 0.160  # outer size, x (short) by y (long)
C_R, C_r, C_D = 0.026, 0.010, 0.032  # cell: 52 mm square at the rim, 20 mm at the floor, 32 mm deep
C_RIM = FLOOR_T + C_D  # rim height 34 mm
C_CELL_X = (-C_PITCH / 2, C_PITCH / 2)  # the two rows (near, far), in the carton's frame
C_CELL_Y = (-C_PITCH, 0.0, C_PITCH)  # the three columns
CELLS = [(CX + dx, CY + dy) for dx in C_CELL_X for dy in C_CELL_Y]  # world (x, y), near row first
C_H0 = seat_height(C_R, C_r, C_D)  # the pointed end stands this high above the cell floor
EGG_Z_SEATED = FLOOR_T + C_H0 + EGG_L / 2  # a seated egg's centre height (about the rim)
CARTON_MASS = None  # fixed in a cradle: a real one rests in a pack-out cradle with a back stop

# the lid: 111 mm deep from its hinge, 36 mm clear inside, hinged on the near (-x) long side, 10 g
HINGE_X = -C_LX / 2  # the hinge line, in the carton's frame (x), at the carton's rim height, along y
LID_LEN = 0.111
LID_CLEAR = 0.036  # inside height: the lid plate's underside above the rim when closed
LID_T = 0.002
LID_SKIRT = 0.008  # the lid's end and front walls reach this far below the rim
LID_MASS = 0.010
LID_OPEN_DEG = 110.0  # open, past vertical, against its stop
LID_OPEN = math.radians(LID_OPEN_DEG)
LID_DAMP = 0.004  # N m s/rad at the hinge (est): the lid closes under its own weight in a few seconds
LID_CLOSED_TOL = 0.005  # closed: the lid's free edge within 5 mm of its closed place
LID_Q_TOL = LID_CLOSED_TOL / LID_LEN  # 0.045 rad (2.6 degrees)
LID_Y_HALF = C_LY / 2 + 0.001  # inside half width of the lid's end walls


def lid_boxes():
    """The lid's boxes, drawn open (joint at 0): [(centre, size, ypr)] in the carton's frame. The lid is first laid out
    closed about its hinge H = (HINGE_X, 0, C_RIM) and then turned by LID_OPEN about +y (up and back)."""
    hx, hz = HINGE_X, C_RIM
    closed = []  # (centre rel H as (u, v), size (lx, ly, lz)) with u along the lid, v up from the rim plane
    top = LID_CLEAR + LID_T / 2
    closed.append(((LID_LEN / 2, 0.0, top), (LID_LEN, 2 * LID_Y_HALF + 2 * LID_T, LID_T)))
    mid_z = (LID_CLEAR + LID_T - LID_SKIRT) / 2
    h = LID_CLEAR + LID_T + LID_SKIRT
    closed.append(((LID_LEN - LID_T / 2, 0.0, mid_z), (LID_T, 2 * LID_Y_HALF + 2 * LID_T, h)))  # free-edge wall
    for s in (-1, 1):
        closed.append(((LID_LEN / 2, s * (LID_Y_HALF + LID_T / 2), mid_z), (LID_LEN, LID_T, h)))  # end walls
    th = -LID_OPEN  # R_y(-110 deg): +x goes up and back
    c, s_ = math.cos(th), math.sin(th)
    out = []
    for (u, y, v), size in closed:
        x2 = u * c + v * s_
        z2 = -u * s_ + v * c
        out.append(((hx + x2, y, hz + z2), size, (0.0, th, 0.0)))
    return out


def lid_point_world(q: float, u: float, v: float) -> tuple:
    """World (x, z) of the lid point at distance u along the lid and v up from its underside plane's rim level, with the
    lid turned q rad from open toward closed (0 open .. LID_OPEN closed). Used by the offline stand-in and tests."""
    psi = LID_OPEN - q  # the lid's angle above horizontal-closed: 110 open, 0 closed
    return (CX + HINGE_X + u * math.cos(psi) - v * math.sin(psi), C_RIM + u * math.sin(psi) + v * math.cos(psi))


# -- the egg flat (fixed), 30 pockets in 5 rows (x) by 6 columns (y) -------------------------------------------------
F_PITCH = 0.051  # the flat's pocket pitch (30-count, about 255 x 306 mm: est)
F_R, F_r, F_D = 0.0255, 0.011, 0.030
F_ROWS, F_COLS = 5, 6
FX, FY = 0.300, 0.295  # the flat's centre (its near edge 8 mm clear of the carton's cradle)
F_RIM = FLOOR_T + F_D
F_H0 = seat_height(F_R, F_r, F_D)
F_PLATE = (F_ROWS * F_PITCH + 0.006, F_COLS * F_PITCH + 0.006)


def flat_pocket(i: int, j: int) -> tuple:
    """World (x, y) of the pocket in row i (0..4, x) and column j (0..5, y)."""
    return (FX + (i - (F_ROWS - 1) / 2.0) * F_PITCH, FY + (j - (F_COLS - 1) / 2.0) * F_PITCH)


# the six eggs: rows 1..3, columns 1 and 4 (a column 3 pockets away, so no egg stands beside another along the
# gripper's closing axis)
EGG_POCKETS = [(1, 4), (2, 4), (3, 4), (1, 1), (2, 1), (3, 1)]
EGGS = tuple(f"e{k}" for k in range(6))
EGG_START = [flat_pocket(i, j) for (i, j) in EGG_POCKETS]
EGG_START_Z = FLOOR_T + F_H0 + EGG_L / 2 + 0.0004  # the eggs' centres when standing in the flat (0.4 mm to settle)
CARTON_NAME = "carton"
FLAT_NAME = "flat"

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
CELL_XY = 0.012  # an egg's centre within this of a cell's centre (top view)
SEAT_TOL = 0.006  # and its centre within this of the seated height (blunt end no more than 6 mm high)
POINT_DOWN_DEG = 25.0  # the egg's long axis within this of vertical, pointed end down
CRACK_SPEED = 0.8  # an egg moving this fast (about a 33 mm free fall, or a throw) hits hard enough to crack
FALL_Z = -0.010  # an egg whose centre is lower than this (off the table) is lost
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible motion: an egg moves no more than 5 mm over the last second
STILL_TURN_DEG = 5.0  # and turns no more than 5 degrees
STILL_LID = math.radians(5.0)  # rad: the lid turns no more than 5 degrees over the last second
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

CELL_PICK_ORDER = (0, 2, 3, 5, 1, 4)  # the order the demo fills cells (CELLS indices)
