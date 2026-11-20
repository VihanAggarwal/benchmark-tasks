"""Every dimension of the task, in one place (metres, kilograms, radians, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305 and the station's table top is at z 0.7475. Heights below are above the table top unless named otherwise.

The game is Hasbro Gaming Perfection (25 shapes, 60 s timer; item F7697): one square game unit, its pop-up tray of
25 shape wells, a timer dial and a STOP/START switch on the front border. The unit stands on the table centred at
x 330 mm, y 0, its dial corner toward the robot on the right. Figures marked est are estimates to be measured on the
bought game (the README gives the bench check for each).

The base frame (the unit) has its origin on the table top under the case's centre; local x forward (world +x), y
left, z up; unturned. Joint values: "tray" the tray's travel down in metres (0 popped up, positive pressed down);
"rock" the tray's rock about the x axis in radians (positive lifts its left side); "dial" the timer dial's turn in
radians, positive clockwise seen from above (0 = run out, DIAL_FULL = 60 s); "switch" the switch knob's slide in metres
(0 = START, SWITCH_STOP = STOP, toward +y). The two catches (pawls) are a separate fixed part in the same frame,
joints "pawl_l" and "pawl_r", the pawl's swing outward in radians (0 resting over the tray's barb line); they must
collide with the tray's barbs, and links of one assembly never collide with each other.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the unit: Hasbro Perfection game unit (est where not published) -------------------------------------------------
UX, UY = 0.330, 0.000
ORIGIN = (UX, UY, TOP)
CASE = 0.285  # square case, 285 x 285 (est, from the 11 in box)
DECK_Z = 0.070  # the case's top deck, 70 mm above the table (est)
DECK_T = 0.006
WALL_T = 0.004
TRAY_XC = 0.020  # the tray's centre 20 mm behind the case's centre: the front border carries the dial and switch
TRAY = 0.230  # tray 230 x 230 (est)
TRAY_T = 0.006
HOLE = 0.234  # the deck's opening round the tray (2 mm play a side)

# -- the tray: on a slide down (springs) with a small rock about x -----------------------------------------------------
TRAY_TOP0 = 0.095  # tray top popped: 25 mm proud of the deck
TRAY_MASS = 0.15  # est
TRAVEL_LATCH = 0.025  # pressed 25 mm: the tray top flush with the deck
TRAVEL_MAX = 0.030  # the bottom stop, 5 mm past flush (the tray's skirt on the case floor ribs, est)
SPRING_K = 320.0  # N/m, four coil springs together (est, 80 N/m each)
SPRING_PRE = 0.012  # m: the springs' preload, as the drive target 12 mm above the popped stop (3.8 N popped, est)
SPRING_DAMP = 8.0  # N s/m: a popping tray stops in about 0.1 s against its top stop (est)
ROCK_MAX = math.radians(4.0)  # the tray rocks up to 4 deg on its guides when pressed on one side (est)
ROCK_K = 0.5  # N m/rad: the springs' levelling moment (est)
ROCK_DAMP = 0.05
ROCK_ANCHOR = (TRAY_XC, 0.0, TRAY_TOP0 - TRAY_T / 2)


def spring_force(travel: float) -> float:
    """The springs' total push (N) at a travel (m)."""
    return SPRING_K * (travel + SPRING_PRE)


# The barbs: one on each side skirt of the tray, the catches' pawls hook over them.
BARB_Y_IN, BARB_Y_OUT = 0.110, 0.118  # |y| of the barb block's inner and outer faces
BARB_Z0, BARB_TOP0 = 0.052, 0.056  # the barb block, tray popped (its top is the latch face)
BARB_X = 0.040  # long along x, centred on the tray
BEVEL_HALF = 0.0055  # the bevel under the barb: a square ridge turned 45 deg, half-diagonal 5.5 mm, centred at
# (|y| 112.5, z 52): its lower outer face rises at 45 deg from (112.5, 46.5) to the block's outer bottom corner
# (118, 52), so the tooth's inner top corner (|y| 114) meets a face that pushes it outward
BEVEL_Y = 0.1125
SKIRT_Y_IN, SKIRT_Y_OUT = 0.106, 0.110
SKIRT_Z0 = 0.049

# -- the catches: a sprung pawl inside each side wall, hinged along x ------------------------------------------------
PAWL_HINGE_Z = 0.060
PAWL_HINGE_Y = 0.122
TOOTH_Y_IN, TOOTH_Y_OUT = 0.114, 0.120  # the tooth overlaps the barb's outer 4 mm
TOOTH_Z0, TOOTH_Z1 = 0.032, 0.037  # tooth underside 1 mm above the barb's top at the 25 mm travel
BAR_Y_IN, BAR_Y_OUT = 0.120, 0.124
BAR_Z1 = 0.062
PAWL_MASS = 0.004
PAWL_UPPER = 0.30  # rad, the pawl's outward stop
PAWL_K = 0.15  # N m/rad (est): about 0.7 N more on each side to ride the barb past the tooth
PAWL_DAMP = 0.002
PAWL_KICK = 0.6  # rad: the run-out cam's target (held against the 0.30 stop): the pawls swing clear
TRAVEL_SNAP = 0.0245  # side travel at which the barb's top is under the tooth (the pawl snaps in)
TRAVEL_HOLD = 0.024  # where the springs hold a latched tray, its barb's top against the tooth


def side_travel(travel: float, rock: float, side: int) -> float:
    """The travel (m) of the tray's barb line on one side (+1 left, -1 right) at a centre travel and rock."""
    y = side * (BARB_Y_IN + BARB_Y_OUT) / 2
    return travel - y * math.sin(rock)


def rock_for(q_left: float, q_right: float) -> float:
    y = (BARB_Y_IN + BARB_Y_OUT) / 2
    s = (q_right - q_left) / (2 * y)
    return max(-ROCK_MAX, min(ROCK_MAX, math.asin(max(-1.0, min(1.0, s)))))


# -- the timer dial and the switch, on the front border ----------------------------------------------------------------
DIAL_X, DIAL_Y = -0.119, -0.090  # base frame; world (211, -90) mm
DIAL_R = 0.020  # 40 mm dial (est)
DIAL_HUB_H = 0.008
RIDGE_L, RIDGE_W, RIDGE_H = 0.036, 0.006, 0.010  # the raised grip bar across the dial (est)
DIAL_FULL = math.radians(300.0)  # 60 s at the dial's clockwise stop (est)
DIAL_RATE = DIAL_FULL / 60.0  # rad/s, the clockwork's run-down with the switch at START
DIAL_SPRING = 0.20  # N m/rad: holds the dial where it is left (the clockwork's escapement)
DIAL_FRICTION = 0.015  # N m (est): about 1.1 N pushed at the grip bar 14 mm out
DIAL_DAMP = 0.002
DIAL_MASS = 0.01
SWITCH_X, SWITCH_Y = -0.119, -0.040  # base frame, knob centre at START
KNOB = (0.008, 0.008, 0.009)
SWITCH_STOP = 0.012  # m: the knob slides 12 mm toward +y to STOP
SWITCH_K = 100.0  # N/m (est): its two detents snap within 4 mm
SWITCH_CAPTURE = 0.004
SWITCH_DAMP = 2.0
SWITCH_MASS = 0.005
SWITCH_START_TOL = 0.003  # the knob within 3 mm of START

LID_ORIGIN = (0.660, 0.000, TOP)  # the box lid with the 25 shapes, behind the unit (out of the work)

# -- colours (sRGB 0..1) -------------------------------------------------------------------------------------------------
CASE_BLUE = (0.10, 0.33, 0.72)
TRAY_YELLOW = (0.96, 0.78, 0.10)
WELL_DARK = (0.12, 0.12, 0.14)
DIAL_RED = (0.85, 0.15, 0.12)
WHITE = (0.95, 0.95, 0.93)
GREY = (0.55, 0.56, 0.58)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
LATCH_OVERLAP = 0.002  # the tooth at least 2 mm over the barb along y
LATCH_DROP = 0.006  # and the barb's top no more than 6 mm under the tooth
FORCE_OVERLAP = 0.002  # the barb more than 2 mm into the tooth in both y and z: forced through (broken in a real trial)
PRESSED = 0.010  # the tray pressed 10 mm down at its centre
DIAL_SET_S = 50.0  # the dial shows 50 s or more
RUN_WINDOW_S = 1.0
RUN_MIN = math.radians(2.0)  # the dial ran down at least 2 deg over the last second
LET_GO = 0.050  # both finger zones 50 mm or more from the game
HOLD_WINDOW_S = 1.0
RESET_JUMP = 0.015
EPS = 1e-9


def dial_seconds(q: float) -> float:
    return 60.0 * q / DIAL_FULL
