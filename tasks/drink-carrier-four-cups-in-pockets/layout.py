"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights are above the table top unless named *_W.

A molded-pulp four-cup drink carrier stands loose on the counter (the table) between the arms, 22 cm out. Four
lidded 8 oz hot cups, filled, stand on the counter two to a side, out beside it. The job: stand each cup upright in
its own pocket, bottomed on the pocket's floor, and take both hands off.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the cup (loose): a lidded 8 oz paper hot cup, filled ---------------------------------------------------------
# The real cup tapers from an 80 mm rim to a 56 mm base over its 88 mm body; its rolled rim is capped by a press-on
# sip lid (an 82 mm skirt 5 mm deep, a 72 mm dome 4 mm high). In the model it is one rigid compound: the body as 11
# stacked discs 8 mm tall (each as wide as the cone at its mid-height, so 1.1 mm steps), the lid's skirt and its dome.
# Its frame's origin is the centre of its underside.
CUP_BASE_D = 0.056
CUP_RIM_D = 0.080
CUP_BODY_H = 0.088
CUP_DISC_H = 0.008
CUP_N_DISCS = 11
LID_SKIRT_D, LID_SKIRT_Z0, LID_SKIRT_Z1 = 0.082, 0.086, 0.091  # the lid's skirt clamps over the rolled rim
LID_DOME_D, LID_DOME_Z1 = 0.072, 0.095  # a low (4 mm) sip-lid dome
CUP_H = LID_DOME_Z1  # 95 mm, underside to the top of the lid
CUP_R = LID_SKIRT_D / 2  # the widest radius (the lid's skirt), for clearances
CUP_MASS = 0.25  # 237 g of water, an 8 g cup and a 4 g lid (est)
CUP_COLOR = (0.94, 0.93, 0.88)  # white paper
LID_COLOR = (0.15, 0.15, 0.16)  # black polystyrene sip lid
CUP_NAMES = ("cup_a", "cup_b", "cup_c", "cup_d")  # a, b on the left arm's side (+y), c, d on the right's (-y)


def cup_d(h: float) -> float:
    """The real cup's outside diameter at height h over its underside (the cone, 56 mm to 80 mm)."""
    return CUP_BASE_D + (CUP_RIM_D - CUP_BASE_D) * min(max(h, 0.0), CUP_BODY_H) / CUP_BODY_H


def cup_parts():
    """[(kind, centre z, diameter, height)] of the cup's shapes in its own frame (origin at its underside)."""
    out = []
    for k in range(CUP_N_DISCS):
        z0 = k * CUP_DISC_H
        out.append(("body", z0 + CUP_DISC_H / 2, cup_d(z0 + CUP_DISC_H / 2), CUP_DISC_H))
    out.append(("lid", (LID_SKIRT_Z0 + LID_SKIRT_Z1) / 2, LID_SKIRT_D, LID_SKIRT_Z1 - LID_SKIRT_Z0))
    out.append(("lid", (LID_SKIRT_Z1 + LID_DOME_Z1) / 2, LID_DOME_D, LID_DOME_Z1 - LID_SKIRT_Z1))
    return out


def cup_radius_at(z: float) -> float:
    """The model cup's outer radius at height z over its underside (0 outside 0..CUP_H)."""
    r = 0.0
    for _, zc, d, h in cup_parts():
        if zc - h / 2 - 1e-12 <= z <= zc + h / 2 + 1e-12:
            r = max(r, d / 2)
    return r


# -- the carrier (loose): a 2 x 2 molded-pulp tray of four round tapered pockets ------------------------------------
PITCH = 0.104  # pocket pitch, both ways (centre to centre)
CAR_X, CAR_Y = 0.22, 0.0  # the carrier's centre, on the table
N_SIDES = 12  # each pocket is a 12-sided funnel (the pulp's cone, drawn in flat slabs)
POCKET_R_FLOOR = 0.031  # inner half-width (apothem) at the pocket's floor: 62 mm across (3 mm round the 56 mm base)
POCKET_R_RIM = 0.038  # inner half-width at its mouth: 76 mm across (the cup is 65.8 mm there when seated)
FLOOR_TOP = 0.007  # the pockets' floors, above the table: a 7 mm base plate
RIM_H = 0.043  # the pockets' mouths, above the table
POCKET_D = RIM_H - FLOOR_TOP  # 36 mm deep
WALL_T = 0.004  # slab thickness (real pulp is about 2 mm; 4 mm is the simulator's contact margin)
EDGE = 0.004  # the base plate reaches this far past the outer faces of the outermost walls
PLATE = PITCH + 2 * (POCKET_R_RIM + WALL_T + EDGE)  # 0.196: the carrier's footprint, a square
CAR_MASS = 0.10  # (est: a 45 g tray, 100 g in the model so a cup's push does not slide it far)
CAR_FRICTION = (1.0, 0.9)  # pulp on a laminate counter (est)
PULP_COLOR = (0.62, 0.55, 0.45)  # grey-brown pulp
POCKET_XY = {  # name -> (x, y) of the pocket's axis, world
    "near_left": (CAR_X - PITCH / 2, CAR_Y + PITCH / 2),
    "far_left": (CAR_X + PITCH / 2, CAR_Y + PITCH / 2),
    "near_right": (CAR_X - PITCH / 2, CAR_Y - PITCH / 2),
    "far_right": (CAR_X + PITCH / 2, CAR_Y - PITCH / 2),
}
POCKETS = tuple(POCKET_XY)
# a seated cup's underside rests on FLOOR_TOP; its lid stands CUP_H + FLOOR_TOP - RIM_H = 59 mm above the carrier's top
SEATED_TOP = FLOOR_TOP + CUP_H  # the lid's top above the table when seated: 102 mm
FOOTPRINT_X = (CAR_X - PLATE / 2, CAR_X + PLATE / 2)

# the cups' start places: two a side, out beside the carrier, 12 cm apart along x (38 mm between the lids)
CUP_START = {
    "cup_a": (0.19, 0.34),
    "cup_b": (0.31, 0.34),
    "cup_c": (0.19, -0.34),
    "cup_d": (0.31, -0.34),
}
# which pocket each arm fills, and in which order (the near pocket first, so the second cup's path does not cross it)
ASSIGN = {"cup_a": "near_left", "cup_b": "far_left", "cup_c": "near_right", "cup_d": "far_right"}


def funnel_slabs(cx: float, cy: float, R: float, r: float, D: float, zf: float, T: float = WALL_T,
                 n: int = N_SIDES):
    """The n slabs of one funnel: inner half width (apothem) r at height zf (its floor), R at zf + D (its mouth).
    Returns [(centre, size, ypr)] in the caller's frame, for mechkit boxes (slab k faces direction k * 360/n)."""
    a = math.atan2(R - r, D)
    ls = math.hypot(D, R - r)
    w = 2.0 * (R + T) * math.tan(math.pi / n) + 0.003
    rm = (R + r) / 2.0
    zc = zf + D / 2.0 - (T / 2.0) * math.sin(a)
    off = rm + (T / 2.0) * math.cos(a)
    out = []
    for k in range(n):
        th = 2.0 * math.pi * k / n
        out.append(((cx + off * math.cos(th), cy + off * math.sin(th), zc), (T, w, ls), (th, a, 0.0)))
    return out


# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
SEAT_Z_TOL = 0.002  # seated: the cup's underside within 2 mm of the pocket's floor
SEAT_TILT_DEG = 2.0  # its axis within 2 degrees of the carrier's vertical
SEAT_XY_TOL = 0.008  # its centre within 8 mm of the pocket's axis (a cup on the floor is within 3.6 mm anyway)
SPILL_DEG = 15.0  # a cup tilted more than this at any step: spilled (latched)
LYING_DEG = 45.0  # a cup ending tilted more than this: lying
DROP_SPEED = math.sqrt(2.0 * G * 0.040)  # 0.886 m/s: a free fall of 40 mm; a cup falling faster than this (latched)
CAR_MOVE = 0.030  # the carrier's centre moved more than 30 mm from its mark (latched)
CAR_TILT_DEG = 5.0  # the carrier tilted more than 5 degrees (latched)
OFF_TABLE = 0.050  # a cup's or the carrier's lowest point more than 50 mm below the table top: fell off
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # over the last 1 s every cup and the carrier moves under 5 mm (visible motion)
STILL_TURN_DEG = 5.0  # and turns (its axis) under 5 degrees
LET_GO = 0.030  # both grippers' finger zones this far from every cup and the carrier
CONTROL_HZ = 20.0
RESET_JUMP = 0.05  # a part that lands on its mark having moved this much further in one step than its velocity can
START_TOL = 0.01
EPS = 1e-9
