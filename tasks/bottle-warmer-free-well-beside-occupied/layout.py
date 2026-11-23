"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475 and plays the nursery's counter. Heights below are above the table
top unless named *_W.

A two-well bottle warmer (cream plastic, 110 x 320 x 60 mm, two round 72 mm wells 50 mm deep, one push button per well
on its top) stands on the counter against a tiled splashback. In its +y well stands a child's feeding bottle that
is already warming. A second child's bottle waits in a shallow bottle tray on the counter. The job: stand the second
bottle in the free well without disturbing the first, then start that well (press its button).

The wells are 16-sided (inradius 36 mm), built from turned boxes; the bottle is 62 mm across, so it has 5 mm
each side to the flats. Figures are sourced (README) or estimates (est, said in task.json's fidelity).
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the warmer (fixed: non-slip feet on the counter, taped down in a trial) ------------------------------------------
WX, WY = 0.260, 0.0  # centre of the body
W_LX, W_LY, W_H = 0.110, 0.320, 0.060
W_X0, W_X1 = WX - W_LX / 2, WX + W_LX / 2  # 0.205 .. 0.315
W_Y0, W_Y1 = WY - W_LY / 2, WY + W_LY / 2
FLOOR = 0.010  # a well's floor top
WELL_IN = 0.036  # inradius of the 16-sided well (72 mm across the flats)
WELL_OUT = 0.055  # outer radius of the wall ring (reaches the body's front and back faces)
WELL_N = 16
WELL_DEPTH = W_H - FLOOR  # 50 mm
WELL_Y = 0.055
WELL_A = (WX, +WELL_Y)  # occupied: the other child's bottle, already warming
WELL_B = (WX, -WELL_Y)  # free
PLATE_T = 0.005  # the top plate over the corners between the rings
SKIN = 0.004  # the body's outer wall
CREAM = (0.93, 0.90, 0.82)
# push buttons on the top, one per well, each near its well's end of the warmer (at y +-143): 18 mm across, 6 mm proud of
# the top, 5 mm of travel (est). Pushed straight down from above: a closed pad block over a cap, as on the conveyor task.
BTN_R, BTN_LEN = 0.009, 0.010
BTN_DX = -0.030  # 30 mm toward the front of the warmer's centre (the arm's elbow stays off the splashback)
BTN_Y = 0.143  # each button sits near its end of the top, clear of the standing bottle's side
BTN_PROUD = 0.006
BTN_TOP = W_H + BTN_PROUD  # 66 mm over the counter: where a button's face is at rest
BTN_TRAVEL = 0.005
BTN_SPRING = 250.0  # N/m: 1.25 N at the stop (est; a light switch)
BTN_DAMP = 2.0
BTN_GREEN = (0.30, 0.68, 0.40)

# -- the splashback (fixed): the counter stands against a wall ---------------------------------------------------------
WALL_X0, WALL_X1 = 0.320, 0.340
WALL_Y0, WALL_Y1 = -0.200, 0.200
WALL_H = 0.200
TILE = (0.78, 0.82, 0.84)

# -- the bottle tray (fixed): where the second child's bottle waits ---------------------------------------------------
TRAY_X, TRAY_Y = 0.270, -0.250
TRAY_L, TRAY_H, TRAY_WALL, TRAY_FLOOR = 0.100, 0.012, 0.004, 0.004
TRAY_GREY = (0.62, 0.66, 0.72)

# -- the feeding bottles (loose): 260 mL polypropylene, 0.20 kg with a 150 mL feed (water in a trial) ---------------------------------
# Base frame at the centre of the base's underside, z up the axis. Body, screw collar (coloured: whose it is), cap hood.
B_R = 0.031  # 62 mm across
B_BODY_H = 0.110
B_COLLAR_Z0, B_COLLAR_Z1 = 0.110, 0.120
B_HOOD_R = 0.026
B_H = 0.135
B_MASS = 0.20
B_BODY = (0.80, 0.90, 0.95)
B_HOOD = (0.96, 0.96, 0.96)
COLLAR_A = (0.95, 0.55, 0.70)  # the neighbour's: pink
COLLAR_B = (0.35, 0.60, 0.95)  # the one to place: blue
A_X0, A_Y0 = WELL_A  # the neighbour's start
BOT_X0, BOT_Y0 = TRAY_X, TRAY_Y  # the loose bottle's start, on the tray's floor
BOT_Z0 = TRAY_FLOOR

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
SEAT_XY = 0.006  # centre within 6 mm of the well's axis
SEAT_Z = 0.002  # base within 2 mm of the floor top
UPRIGHT_DEG = 3.0  # seated and the neighbour: axis within 3 degrees of vertical
NB_MOVE = 0.003  # the neighbour's centre moved 3 mm or more at any time
NB_LIFT = 0.002  # or its base lifted 2 mm
BTN_A_TOUCH = 0.001  # the neighbour's own button pushed 1 mm
NB_OUT_XY = 0.012  # latched 0: the neighbour left its well (centre 12 mm off its axis) ...
NB_OUT_LIFT = 0.030  # ... or its base 30 mm up
SPILL_DEG = 30.0  # latched 0: either bottle tilted over 30 degrees
FALL_Z = -0.050  # latched 0: either bottle's lowest point 50 mm below the counter's top
PRESSED = 0.003  # the free well's button 3 mm in (of 5): a push
REARM = 0.001  # back out to 1 mm before the next push counts
REST_Z = 0.003  # resting on the warmer's top: lowest point within 3 mm of it
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible-motion rule: each bottle moves under 5 mm over the last 1 s ...
STILL_TURN_DEG = 5.0  # ... and turns under 5 degrees; a button's state is its latched pushes, not sub-mm stillness
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.08
START_TOL = 0.01
EPS = 1e-9


def ring_segments():
    """(centre offset (x, y) from a well's axis, tangent length, yaw) of the 16 wall boxes round one well."""
    out = []
    t = WELL_OUT - WELL_IN
    r_mid = WELL_IN + t / 2
    tang = 2 * WELL_OUT * math.tan(math.pi / WELL_N) + 0.0005
    for k in range(WELL_N):
        a = 2 * math.pi * k / WELL_N
        out.append(((r_mid * math.cos(a), r_mid * math.sin(a)), tang, a))
    return out


def rim_local(r=B_R, z=0.0, n=16):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def bottle_pts_local():
    """Base rim points, then top (hood) rim points, in a bottle's frame: (32, 3)."""
    return rim_local(B_R, 0.0) + rim_local(B_HOOD_R, B_H)
