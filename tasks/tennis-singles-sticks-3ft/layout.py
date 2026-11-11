"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A tennis court painted at 1:15, only the 300 mm of it either side of the net: the net line runs along y at x 0.350,
the court's centre line is y 0. ITF Rules of Tennis, Rule 1: the singles court is 8.23 m wide and the doubles court
10.97 m, both measured to the outside edges of the lines, so each doubles alley is 1.37 m wide; when a doubles net is
used for singles it is held up by two singles sticks whose centres are 0.914 m (3 ft) outside the singles court on
each side, two thirds of the way across the alley; the net posts' centres are 0.914 m outside the doubles court; the
net is 1.07 m high at the posts and sticks. At 1:15 those are 0.2743 m, 0.3657 m, 0.3353 m, 0.4266 m and 71 mm.
The net itself is not modelled: a dark grey band painted along the net line marks where it hangs.
"""

import math

TOP = 0.7475  # station table top (world z)
SCALE = 15.0
M = 1.0 / SCALE  # one real metre at 1:15

# -- the court rule (ITF Rules of Tennis, Rule 1), at scale ----------------------------------------------------------
SINGLES_HALF = 8.23 / 2 * M  # 0.27433: the singles sideline's outside edge, from the centre line
DOUBLES_HALF = 10.97 / 2 * M  # 0.36567: the doubles sideline's outside edge
STICK_OUT = 0.914 * M  # 0.06093: a singles stick's centre this far outside the singles court (3 ft)
STICK_Y = SINGLES_HALF + STICK_OUT  # 0.33527: where each singles stick's centre goes (|y|)
NET_POST_Y = DOUBLES_HALF + STICK_OUT  # 0.42660: the net posts' centres (|y|)
NET_HEIGHT = 1.07 * M  # 0.0713: the net's height at the posts and sticks

# -- the painted court: fixed 1 mm mats, their tops flush (COURT_TOP), the lines 0.4 mm proud on top -----------------
MAT_H = 0.001
COURT_TOP = TOP + MAT_H  # the playing surface: the reference for 'standing on the court'
COURT_X = (0.200, 0.500)  # the part of the court shown: 150 mm (2.25 m) either side of the net
NET_X = 0.350  # the net line (along y)
NET_BAND_W = 0.010  # the painted net band, 10 mm wide along x (15 cm at scale) ...
NET_BAND_HALF = 0.427  # ... from y -0.427 to +0.427 (post centre to post centre)
SURROUND_Y = 0.450  # the green surround outside the doubles sidelines runs to |y| 0.450
NET_X0, NET_X1 = NET_X - NET_BAND_W / 2, NET_X + NET_BAND_W / 2  # 0.345 .. 0.355
LINE_W = 0.003  # every painted line 3 mm wide (4.5 cm at scale; the ITF allows 2.5 to 5 cm)
LINE_T = 0.0004
SINGLES_LINE_Y = SINGLES_HALF - LINE_W / 2  # 0.27283: the singles sidelines' centre lines (|y|)
DOUBLES_LINE_Y = DOUBLES_HALF - LINE_W / 2  # 0.36417: the doubles sidelines' centre lines (|y|)
CENTRE_LINE_Y = 0.0  # the centre service line
# The lines stop at the net band (they are painted on the court mats either side of it).

# -- the net posts (fixed, over-scale: 25 mm across, 1.07 m tall at scale) -------------------------------------------
NET_POST_RADIUS = 0.0125
NET_POST_H = 0.071

# -- the two singles sticks: one loose rigid body each (mechkit compound), white ABS, uniform density ---------------
# Base frame: origin at the middle of the foot's underside, local +z up the post's axis.
FOOT_SIZE = (0.040, 0.040, 0.008)  # a square foot
POST_SIZE = (0.020, 0.020, 0.063)  # a 20 mm square post on it (0.30 m at scale; the ITF limit is 7.5 cm)
STICK_H = FOOT_SIZE[2] + POST_SIZE[2]  # 0.071: the post's top at the net's height
FOOT_CENTER = (0.0, 0.0, FOOT_SIZE[2] / 2)
POST_CENTER = (0.0, 0.0, FOOT_SIZE[2] + POST_SIZE[2] / 2)
STICK_MASS = 0.040  # 38.0 cm3 of ABS (1.05 g/cm3)
STICK_VOLUME = (FOOT_SIZE[0] * FOOT_SIZE[1] * FOOT_SIZE[2] + POST_SIZE[0] * POST_SIZE[1] * POST_SIZE[2])
CENTRE_LOCAL = (0.0, 0.0, STICK_H / 2)  # the stick's centre on its axis (its stillness is read here)
STICK_REACH_RADIUS = math.hypot(FOOT_SIZE[0] / 2, FOOT_SIZE[1] / 2)  # 0.0283: the foot's corner from the axis
NAMES = ("stick_a", "stick_b")
START = {"stick_a": (0.240, 0.150), "stick_b": (0.240, -0.150)}  # standing on the near court, either side
START_YAW = 0.0

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
NET_TOL = 0.006  # 'on the net line': the stick's centre within 6 mm of it along x
ITF_TOL = 0.008  # 0.5: | |y| - 0.3353 | within 8 mm
ALLEY = (0.270, 0.370)  # 0.25: |y| from the singles sideline's inner edge to beyond the doubles sideline's outer edge
UPRIGHT_DEG = 10.0  # 'upright': the post's axis within 10 degrees of vertical, foot down
SEAT_TOL = 0.003  # 'standing on the court': its lowest point within 3 mm of the court surface
STILL_WINDOW_S = 0.5  # 'still': its axis midpoint's path under STILL_PATH in each of the last two 0.5 s windows
STILL_WINDOWS = 2
STILL_PATH = 0.0015
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # every stick back on its starting mark having moved this much further in one control step than
#                    its last velocity and gravity could carry it: the simulator put it there, a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# 'Let go': rigstate reports each gripper's pad-tip point (midway between its finger tips, 13 mm past the TCP) and
# how far it is open, but not its yaw, so a finger may be on any side of that point. With the fingers FULL_GAP x
# opening apart, a finger (FINGER_THICK thick outside the gap, 2 x FINGER_HALF_WIDTH wide across it) can touch a stick
# only if the stick's axis is within release_reach(opening) of that point horizontally (the foot's corner is 28.3 mm
# from the axis): 94 mm fully open, 62 mm closed on the post. A gripper whose pad-tip point is lower than RELEASE_ABOVE
# over the stick's top while within that reach may be holding, pressing or propping it: the stick is not released.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
COURT_BLUE = (0.12, 0.30, 0.58)
SURROUND_GREEN = (0.16, 0.45, 0.30)
LINE_WHITE = (0.96, 0.96, 0.96)
NET_GREY = (0.12, 0.12, 0.13)
POST_GREEN = (0.05, 0.22, 0.12)
STICK_WHITE = (0.97, 0.97, 0.95)


def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the foot's
    corner distance from the stick's axis and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + STICK_REACH_RADIUS + RELEASE_MARGIN


assert abs(SINGLES_HALF - 0.2743) < 0.00005 and abs(DOUBLES_HALF - 0.3657) < 0.00005
assert abs(STICK_Y - 0.3353) < 0.00005 and abs(NET_POST_Y - 0.4266) < 0.00005
assert abs(STICK_H - NET_HEIGHT) < 0.0005
assert abs(STICK_OUT / (DOUBLES_HALF - SINGLES_HALF) - 2.0 / 3.0) < 0.002  # two thirds of the alley
assert abs(STICK_MASS / STICK_VOLUME - 1050.0) < 15.0  # ABS
