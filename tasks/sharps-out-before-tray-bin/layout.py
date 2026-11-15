"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A used procedure tray (thermoformed, 215 x 130 x 35 mm, loose) lies on the midline. A divider rib runs across it,
splitting it into two compartments. In the left one its foam needle block (glued in as supplied) holds a used 3 mL
Luer-lock syringe upright, its 25 G x 5/8 in needle buried in the block. Plastic forceps lie in the right one. A 1 qt
sharps container (Fisherbrand 22730457 class: 3.9 x 3.8 x 6.1 in, a 1.6 x 1.25 in opening in a flip-top lid, flap
open) is taped down on the left, an open waste bin on the right.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

import mechkit as mk

MM = 0.001
TOP = 0.7475  # station table top (world z)
IN = 0.0254

# -- the procedure tray (loose) ----------------------------------------------------------------------------------------
# Base frame at the centre of the underside, local x across the tray (130 mm), local y along it (215 mm), z up.
TRAY_X, TRAY_Y = 0.220, 0.000  # start: centre on the midline, 22 cm out, long side along y
TRAY_LX, TRAY_LY, TRAY_H = 0.130, 0.215, 0.035  # est (a laceration or lumbar puncture kit tray)
TRAY_WALL = 0.002  # model thickness; the real thermoformed wall is about 0.5 mm (est, fidelity)
TRAY_FLOOR = 0.002
TRAY_MASS = 0.030  # est
RIB_W, RIB_TOP = 0.012, 0.030  # divider rib across the tray at its middle: 12 mm wide, top 30 mm up (est)
BLOCK_SIZE = (0.040, 0.025, 0.025)  # foam needle block (x, y, z), glued in the left compartment (est)
BLOCK_Y = 0.075  # its centre, from the tray's centre along +y (left): the syringe 75 mm from the rib
BLOCK_MASS = 0.003  # est, part of the tray body (glued)
BORE = 0.004  # square bore the needle stands in (the foam is pierced in real life)
BORE_DEPTH = 0.018
BLOCK_TOP = TRAY_FLOOR + BLOCK_SIZE[2]  # 27 mm above the tray's underside
FORCEPS_Y = -0.055  # forceps lie across the right compartment, centre 55 mm right of the tray's centre
TRAY_GREY = (0.82, 0.85, 0.88)  # clear-white thermoformed PET
BLOCK_GREEN = (0.30, 0.62, 0.42)  # the foam block
RIB_GRIP_Z = 0.012  # pad tips 12 mm above the tray's underside (18 mm down the rib) when the tray is gripped

# -- the syringe (loose) -----------------------------------------------------------------------------------------------
# BD 309570 class: 3 mL Luer-Lok syringe with a 25 G x 5/8 in needle (sourced); its sizes below are est (measure one).
# Base frame at the needle's tip, local z up the axis, local x along the flange's long side.
NEEDLE_L, NEEDLE_R = 0.016, 0.001  # 5/8 in; collider 2 mm across (the real 25 G needle is 0.5 mm)
HUB_L, HUB_R = 0.012, 0.0045  # needle hub plus the syringe's Luer-lock collar
BARREL_L, BARREL_R = 0.056, 0.00585  # 11.7 mm across, 56 mm long (a listed 3 mL Luer-lock barrel)
FLANGE = (0.022, 0.013, 0.002)
ROD = (0.007, 0.007, 0.010)  # plunger rod above the flange, plunger pushed home
THUMB_R, THUMB_T = 0.008, 0.002
HUB_Z0 = NEEDLE_L  # 16 mm: where the bare needle ends and the hub starts
BARREL_Z0 = HUB_Z0 + HUB_L  # 28 mm
FLANGE_Z0 = BARREL_Z0 + BARREL_L  # 84 mm
ROD_Z0 = FLANGE_Z0 + FLANGE[2]
THUMB_Z0 = ROD_Z0 + ROD[2]
SYR_LEN = THUMB_Z0 + THUMB_T  # 98 mm
SYR_MASS = 0.004  # est
SYR_X, SYR_Y = TRAY_X, TRAY_Y + BLOCK_Y  # standing in the block's bore
SYR_TIP_Z = BLOCK_TOP - HUB_Z0  # 11 mm: the hub rests on the block's top, the needle 16 mm into the foam
SYRINGE_CLEAR = (0.85, 0.88, 0.90)
NEEDLE_STEEL = (0.75, 0.76, 0.78)
HUB_BLUE = (0.20, 0.40, 0.85)  # 25 G hubs are blue
PLUNGER_WHITE = (0.95, 0.95, 0.95)
GRIP_Z = 0.036  # pad tips 36 mm up the syringe from the needle's tip: 8 mm above the hub, 20 mm above the needle

# -- the forceps (loose) -----------------------------------------------------------------------------------------------
FORCEPS = (0.110, 0.010, 0.008)  # disposable plastic forceps, closed (110 mm sourced length class; section est)
FORCEPS_MASS = 0.005  # est
FORCEPS_BLUE = (0.25, 0.55, 0.85)

# -- the sharps container (fixed, taped) -------------------------------------------------------------------------------
# Fisherbrand 22730457: 1 qt, flip-top lid, 3.9 x 3.8 x 6.1 in, opening 1.6 x 1.25 in. Base frame at the centre of its
# footprint, local axes the world's.
CON_X, CON_Y = 0.250, 0.270
CON_LX, CON_LY, CON_H = round(3.9 * IN, 4), round(3.8 * IN, 4), round(6.1 * IN, 4)  # 99.1 x 96.5 x 154.9 mm
CON_WALL = 0.003  # est
CON_FLOOR = 0.003
LID_T = 0.003
OPEN_X, OPEN_Y = round(1.6 * IN, 4), round(1.25 * IN, 4)  # 40.6 mm along x, 31.8 mm along y, in the lid's centre
LID_UNDER = CON_H - LID_T  # 151.9 mm: below this plane the syringe is inside
FLAP_SET = 0.015  # the open flap is hinged this far beyond the opening's far (+x) edge (est)
FLAP = (0.003, 0.038, 0.045)  # thickness (x), width (y), height (z) (est)
FLAP_LEAN = math.radians(30.0)  # opened past upright, leaning back away from the opening (est)
CON_RED = (0.78, 0.12, 0.12)
CON_INNER = (CON_LX - 2 * CON_WALL, CON_LY - 2 * CON_WALL)

# -- the waste bin (fixed, taped) --------------------------------------------------------------------------------------
BIN_X, BIN_Y = 0.270, -0.290
BIN_IN_X, BIN_IN_Y, BIN_DEPTH = 0.190, 0.270, 0.150  # inside (est: a low open bin, see fidelity)
BIN_WALL = 0.004
BIN_FLOOR = 0.004
BIN_H = BIN_FLOOR + BIN_DEPTH  # rim 154 mm above the table
BIN_GREY = (0.45, 0.47, 0.50)

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
NEEDLE_TOUCH = 0.002  # H3: a finger region within 2 mm of the needle's collider
FINGER_LEN = 0.038  # the finger region: from the pad tips 38 mm back toward the TCP (rigstate's finger zone)
WALL_TOL = 0.002  # "inside the inner walls" allows 2 mm into a wall (as the floor does): a part resting on a wall
# reads a few microns into it (GPU 2026-11-14: the syringe lay in the container's corner, its box 0.00001 mm past y0)
TRAY_FLOOR_TOL = 0.020  # the tray "on the bin floor": its lowest point within 20 mm of the floor's top
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.003  # tray and forceps move under 3 mm and turn under 3 degrees in the last 1 s
STILL_TURN_DEG = 3.0
LET_GO = 0.025  # both grippers' finger zones at least 25 mm from the tray, the forceps and the syringe
OFF_TABLE_DROP = 0.010
TABLE_X0, TABLE_X1 = 0.36 - 0.55, 0.36 + 0.55
TABLE_Y0, TABLE_Y1 = -0.65, 0.65
CONTROL_HZ = 20.0
RESET_JUMP = 0.08
START_TOL = 0.02
EPS = 1e-9


# -- shape builders ----------------------------------------------------------------------------------------------------
def tray_shapes():
    """The tray in its own frame: an open box with a divider rib across its middle, the foam block (with its bore) in
    the left compartment."""
    shapes = list(mk.open_box((0, 0, TRAY_H / 2), (TRAY_LX, TRAY_LY, TRAY_H), TRAY_WALL, TRAY_FLOOR, TRAY_GREY))
    rib_h = RIB_TOP - TRAY_FLOOR
    shapes.append(mk.box((0, 0, TRAY_FLOOR + rib_h / 2), (TRAY_LX - 2 * TRAY_WALL, RIB_W, rib_h), TRAY_GREY))
    bx, by, bz = BLOCK_SIZE
    solid = bz - BORE_DEPTH
    shapes.append(mk.box((0, BLOCK_Y, TRAY_FLOOR + solid / 2), (bx, by, solid), BLOCK_GREEN))
    shapes += mk.plate_with_holes((0, BLOCK_Y, TRAY_FLOOR + solid + BORE_DEPTH / 2), (bx, by), [(0, 0, BORE, BORE)],
                                  BORE_DEPTH, BLOCK_GREEN)
    return shapes


def rib_box():
    """(centre, size) in the tray's frame of the rib with the floor under it: what the offline checker grips."""
    return (0.0, 0.0, RIB_TOP / 2), (TRAY_LX - 2 * TRAY_WALL, RIB_W, RIB_TOP)


def syringe_shapes():
    return [
        mk.cylinder((0, 0, NEEDLE_L / 2), NEEDLE_R, NEEDLE_L, "Z", NEEDLE_STEEL),
        mk.cylinder((0, 0, HUB_Z0 + HUB_L / 2), HUB_R, HUB_L, "Z", HUB_BLUE),
        mk.cylinder((0, 0, BARREL_Z0 + BARREL_L / 2), BARREL_R, BARREL_L, "Z", SYRINGE_CLEAR),
        mk.box((0, 0, FLANGE_Z0 + FLANGE[2] / 2), FLANGE, SYRINGE_CLEAR),
        mk.box((0, 0, ROD_Z0 + ROD[2] / 2), ROD, PLUNGER_WHITE),
        mk.cylinder((0, 0, THUMB_Z0 + THUMB_T / 2), THUMB_R, THUMB_T, "Z", PLUNGER_WHITE),
    ]


def forceps_shapes():
    return [mk.box((0, 0, FORCEPS[2] / 2), FORCEPS, FORCEPS_BLUE)]


def container_shapes():
    """The sharps container in its own frame (origin at the centre of its footprint): an open box, the lid with the
    opening at its centre, and the open flap hinged beyond the opening's far edge, leaning back 30 degrees."""
    body_h = CON_H - LID_T
    shapes = list(mk.open_box((0, 0, body_h / 2), (CON_LX, CON_LY, body_h), CON_WALL, CON_FLOOR, CON_RED))
    shapes += mk.plate_with_holes((0, 0, CON_H - LID_T / 2), (CON_LX, CON_LY), [(0, 0, OPEN_X, OPEN_Y)], LID_T,
                                  CON_RED)
    c, s = math.cos(FLAP_LEAN), math.sin(FLAP_LEAN)
    hx = OPEN_X / 2 + FLAP_SET  # the hinge line, at the lid's top
    centre = (hx + FLAP[0] / 2 * c + FLAP[2] / 2 * s, 0.0, CON_H - FLAP[0] / 2 * s + FLAP[2] / 2 * c)
    shapes.append(mk.box(centre, FLAP, CON_RED, ypr=(0.0, FLAP_LEAN, 0.0)))
    return shapes


def bin_shapes():
    out = (BIN_IN_X + 2 * BIN_WALL, BIN_IN_Y + 2 * BIN_WALL, BIN_H)
    return list(mk.open_box((0, 0, BIN_H / 2), out, BIN_WALL, BIN_FLOOR, BIN_GREY))


def container_inner():
    """(x0, x1, y0, y1, z_floor, z_lid_under) of the container's inside, above the table."""
    hx, hy = CON_INNER[0] / 2, CON_INNER[1] / 2
    return CON_X - hx, CON_X + hx, CON_Y - hy, CON_Y + hy, CON_FLOOR, LID_UNDER


def bin_inner():
    """(x0, x1, y0, y1, z_floor, z_rim) of the bin's inside, above the table."""
    return (BIN_X - BIN_IN_X / 2, BIN_X + BIN_IN_X / 2, BIN_Y - BIN_IN_Y / 2, BIN_Y + BIN_IN_Y / 2, BIN_FLOOR, BIN_H)


def opening_world():
    """(x0, x1, y0, y1, z) of the lid's opening."""
    return CON_X - OPEN_X / 2, CON_X + OPEN_X / 2, CON_Y - OPEN_Y / 2, CON_Y + OPEN_Y / 2, CON_H


def tray_mass() -> float:
    return TRAY_MASS + BLOCK_MASS
