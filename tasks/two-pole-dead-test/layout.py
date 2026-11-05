"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The terminals all stand on or near the midline (y 0), in one row along x. Each probe is a long square bar that lies
along y and is held at its middle, so its tip reaches the midline while its gripper stays 14 cm out to the side:
the two grippers are always at least 28 cm apart in y, and neither ever passes over a terminal.
"""

TOP = 0.7475  # station table top

# -- the terminals ---------------------------------------------------------------------------------------------------
POST_RADIUS = 0.008  # every terminal post is 16 mm across
BASE_H = 0.008  # both insulating bases (the proving unit's and the terminal block's) are 8 mm tall
BASE_TOP = TOP + BASE_H
COLLAR_H = 0.013  # each post: a coloured insulating collar on the base ...
CAP_H = 0.004  # ... topped by a 4 mm brass contact cap
POST_TOP = BASE_TOP + COLLAR_H + CAP_H  # 25 mm above the table: the surface a probe tip rests on
COLLAR_TOP = BASE_TOP + COLLAR_H

# The proving unit (a known voltage source), nearest the arms, across the midline: its two terminals side by side in y,
# so that the two probes reach them from opposite sides with their tips 40 mm apart end to end.
PROVE_X = 0.13
PROVE_POS_Y, PROVE_NEG_Y = 0.030, -0.030  # P+ on the left of the midline, P- on the right
PROVE_BASE_SIZE = (0.030, 0.100, BASE_H)  # yellow; x 0.115..0.145, y -0.05..+0.05

# The supply's terminal block: L1, L2, L3, N, PE in a row along x on the midline, 50 mm apart. 50 mm keeps two probe
# tips on neighbouring posts 24 mm apart (50 mm less the 26 mm width of a tip), 9 mm more than the 15 mm short limit.
SUPPLY_NAMES = ("L1", "L2", "L3", "N", "PE")
POST_PITCH = 0.050
SUPPLY_X0 = 0.18  # L1; PE at 0.38
SUPPLY_X = tuple(SUPPLY_X0 + i * POST_PITCH for i in range(5))
BLOCK_SIZE = (4 * POST_PITCH + 0.050, 0.040, BASE_H)  # dark grey; x 0.155..0.405 (10 mm clear of the proving unit)
BLOCK_X = (SUPPLY_X[0] + SUPPLY_X[-1]) / 2

# Every terminal: name -> (x, y) of its axis. P+ and P- are the proving unit's.
POSTS = {"P+": (PROVE_X, PROVE_POS_Y), "P-": (PROVE_X, PROVE_NEG_Y),
         **{n: (x, 0.0) for n, x in zip(SUPPLY_NAMES, SUPPLY_X)}}
PROVE_PAIR = frozenset(("P+", "P-"))
SUPPLY_PAIRS = tuple(frozenset((a, b)) for i, a in enumerate(SUPPLY_NAMES) for b in SUPPLY_NAMES[i + 1:])  # 10

# -- the probes --------------------------------------------------------------------------------------------------------
# A hollow square plastic bar with a brass tip cap on one end. 26 mm square: the gripper's pads need at least 24 mm of
# height to grip with their tips 4 mm above the bar's bottom, and it is gripped across 26 mm (18 mm commanded).
# 300 mm long, gripped at its middle (its centre of mass, so its weight puts no torque on the grip), puts the gripper
# 140 mm out from the tip's contact point: far enough out that the grippers stay 28 cm apart and the reach limit at
# the far post is met with margin.
PROBE_SIZE = (0.026, 0.300, 0.026)  # x (across), y (long axis), z, lying along y
PROBE_W = PROBE_SIZE[0]
PROBE_LEN = PROBE_SIZE[1]
PROBE_HALF_H = PROBE_SIZE[2] / 2
PROBE_MASS = 0.060  # a 2 mm-wall plastic tube with a brass cap
TIP_LEN = 0.020  # the last 20 mm toward the tip end is the metal tip; the rest is insulated
PIN_FROM_END = TIP_LEN / 2  # the contact point: the centre of the tip's underside, 10 mm from the tip end
PIN_FROM_CENTRE = PROBE_LEN / 2 - PIN_FROM_END  # 0.140
GRIP_FROM_END_MIN = 0.040  # 'held' needs the grip point at least this far from the tip end: on the insulated handle,
#                            the 27 mm pads wholly behind the 20 mm metal tip with 6.5 mm to spare
# Which way each probe's tip points along its body's y axis (both start with the tip toward the midline).
TIP_SIGN = {"red_probe": -1.0, "black_probe": 1.0}
PROBES = ("red_probe", "black_probe")

# -- the probe rests ---------------------------------------------------------------------------------------------------
REST_X = 0.22
REST_Y = {"red_probe": 0.22, "black_probe": -0.22}  # the red probe on the left, the black on the right
REST_SIZE = (0.060, 0.320, 0.004)  # light grey boards; the probe lies centred on its board, tip 7 cm from the midline
REST_TOP = TOP + REST_SIZE[2]
PROBE_START_Z = REST_TOP + PROBE_HALF_H

# -- the grippers (yam_arena) ----------------------------------------------------------------------------------------
FULL_GAP = 0.0939  # pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
ON_POST_R = POST_RADIUS  # a tip's contact point over a post's top: within the post's 8 mm radius horizontally
ON_POST_ABOVE = 0.004  # ... and the tip's underside at most 4 mm above the post top (a 4 mm spring contact pin) ...
ON_POST_BELOW = 0.003  # ... and at most 3 mm below it (contact penetration and a slightly tilted cap's centre)
LEVEL_DEG = 10.0  # a probe in a test: its tip's underside facing within 10 degrees of straight down
STEADY_SPEED = 0.01  # a probe in a test: its contact point moving under 1 cm/s
TEST_HOLD_S = 1.0  # a test: the same pair held, both tips on their posts, both steady, for 1 s (20 control steps)
SHORT_GAP = 0.015  # the two metal tips closer than this (surface to surface): a short (score 0)
TOUCH_GAP = 0.002  # a metal tip this close to a post: touching it
LOOSE_GRACE_S = 0.25  # a probe neither held nor on a rest for longer than this: set down or dropped (score 0)
HELD_GAP_SLACK = 0.006  # 'held': the finger gap at most 6 mm wider than the probe's 26 mm ...
HELD_AXIS = 0.015  # ... and the point between the finger tips within 15 mm of the probe's long axis
REST_LOW_TOL = 0.004  # 'on a rest': the probe's lowest point at most 4 mm above the rest's top (and at most 3 mm
#                      below it), and both ends of its axis over the rest's 60 x 320 mm board
REST_LOW_BELOW = 0.003
OFF_TABLE_DROP = 0.010  # a probe whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s, for success
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.05  # a probe moving this much further in one step than its velocity allows was teleported
START_TOL = 0.02  # ... onto its starting place: within this of it
RUN_LIMIT_STEPS = 2400

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
RED = (0.85, 0.10, 0.10)
BLACK = (0.08, 0.08, 0.08)
BRASS = (0.80, 0.65, 0.25)
YELLOW = (0.95, 0.80, 0.10)
BLOCK_GREY = (0.25, 0.25, 0.28)
REST_GREY = (0.75, 0.75, 0.75)
WHITE = (0.95, 0.95, 0.95)
COLLAR = {  # the conductor colours of IEC 60445 (brown, black, grey, blue, green-and-yellow shown as green)
    "P+": RED,
    "P-": WHITE,
    "L1": (0.45, 0.25, 0.10),
    "L2": BLACK,
    "L3": (0.55, 0.55, 0.55),
    "N": (0.10, 0.30, 0.85),
    "PE": (0.15, 0.65, 0.20),
}
