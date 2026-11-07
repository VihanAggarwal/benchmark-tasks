"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top (the bench) is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top: the bench

# The air track: a fixed aluminium beam along y, centred on the midline 28 cm in front of the arm bases.
TRACK_X = 0.280
TRACK_W = 0.030  # across (x)
TRACK_LEN = 0.440  # along y, from -0.220 to +0.220
TRACK_H = 0.025
TRACK_Y0, TRACK_Y1 = -TRACK_LEN / 2, TRACK_LEN / 2
TRACK_HALF_W = TRACK_W / 2

# End buffers: fixed black rubber blocks against each end of the track.
BUFFER_SIZE = (0.040, 0.010, 0.035)
BUFFER_Y = TRACK_Y1 + BUFFER_SIZE[1] / 2  # +-0.225: they span |y| 0.220 .. 0.230

# The parked glider: fixed (the air is off), resting on the track at its +y (left) end, 1 mm short of the buffer.
# A top plate on the track, two skirts hanging down beside the track's sides, and a black interrupt card standing
# on top along the track. It runs toward -y when the air is on.
GLIDER_Y = 0.184
GLIDER_LEN = 0.070  # along y: it spans y 0.149 .. 0.219
GLIDER_PLATE = (0.041, GLIDER_LEN, 0.005)  # on the track top: 25-30 mm above the bench
GLIDER_SKIRT = (0.005, GLIDER_LEN, 0.015)  # 10-25 mm above the bench, inner faces 0.5 mm off the track's sides
GLIDER_SKIRT_X = TRACK_HALF_W + 0.0005 + GLIDER_SKIRT[0] / 2  # +-0.018 from the track's centre line
GLIDER_SKIRT_Z0 = 0.010
GLIDER_CARD = (0.002, 0.040, 0.025)  # 30-55 mm above the bench, centred on the plate
GLIDER_HALF_W = GLIDER_SKIRT_X + GLIDER_SKIRT[0] / 2  # 0.0205
GLIDER_FRONT_Y = GLIDER_Y - GLIDER_LEN / 2  # 0.149: the end it runs toward
CARD_TOP = TRACK_H + GLIDER_PLATE[2] + GLIDER_CARD[2]  # 0.055 above the bench

# The glider's path: the space it sweeps on its run, grown by a small margin. Nothing of a gate may be in it.
PATH_HALF_W = 0.022  # from the track's centre line, across (the glider is 41 mm wide: 1.5 mm each side)
PATH_TOP = 0.057  # above the bench (the card's top is 55 mm up)
PATH_Y0, PATH_Y1 = TRACK_Y0, TRACK_Y1  # between the buffers

# The timing section: where a gate's centre must be so the glider, starting from where it is parked, runs
# through it and its card passes right through the beam. In front of the parked glider (crossbar at least 9 mm
# clear of its front) and far enough from the far buffer that, when the glider stops against it (glider y -0.220
# to -0.150, card -0.205 to -0.165), the card's back edge is 10 mm past the gate's windows (a gate nearer the
# buffer would have the card come to rest in its beam).
SECTION_Y0, SECTION_Y1 = -0.150, 0.130
CARD_BACK_AT_STOP = TRACK_Y0 + GLIDER_LEN / 2 + GLIDER_CARD[1] / 2  # -0.165: the card's +y edge, glider at the stop

# A light gate: a free-standing inverted U of black ABS, its base frame at the centre between its two feet, on the
# bench (z 0 local), local x along the crossbar. Two feet, two legs, a crossbar and a red sensor window on each
# leg's inner face (the beam crosses between them, 40-50 mm up: above the glider's 30 mm top plate, below the
# card's 55 mm top, so the card cuts it).
LEG_X = 0.040  # leg and foot centres at local x +-0.040
FOOT_SIZE = (0.012, 0.040, 0.004)  # x 0.034 .. 0.046, along the track +-0.020, 0-4 mm up
LEG_SIZE = (0.010, 0.020, 0.070)  # x 0.035 .. 0.045, 4-74 mm up
CROSSBAR_SIZE = (0.090, 0.020, 0.010)  # x -0.045 .. 0.045, 74-84 mm up
WINDOW_SIZE = (0.001, 0.010, 0.010)  # 1 mm proud of each leg's inner face, 40-50 mm up
FOOT_Z = FOOT_SIZE[2] / 2
LEG_Z = FOOT_SIZE[2] + LEG_SIZE[2] / 2  # 0.039
CROSSBAR_Z0 = FOOT_SIZE[2] + LEG_SIZE[2]  # 0.074: the crossbar's underside
CROSSBAR_Z = CROSSBAR_Z0 + CROSSBAR_SIZE[2] / 2  # 0.079
GATE_H = CROSSBAR_Z0 + CROSSBAR_SIZE[2]  # 0.084
WINDOW_X = LEG_X - LEG_SIZE[0] / 2 - WINDOW_SIZE[0] / 2  # 0.0345
WINDOW_Z = 0.045
INNER_HALF = LEG_X - LEG_SIZE[0] / 2 - WINDOW_SIZE[0]  # 0.034: feet and windows, from the gate's centre
GATE_MASS = 0.08
# Local shapes (centre, size) of the gate's base body, at rest: what the scorer tests against the glider's path.
GATE_BOXES = (
    ((-LEG_X, 0.0, FOOT_Z), FOOT_SIZE),
    ((LEG_X, 0.0, FOOT_Z), FOOT_SIZE),
    ((-LEG_X, 0.0, LEG_Z), LEG_SIZE),
    ((LEG_X, 0.0, LEG_Z), LEG_SIZE),
    ((0.0, 0.0, CROSSBAR_Z), CROSSBAR_SIZE),
    ((-WINDOW_X, 0.0, WINDOW_Z), WINDOW_SIZE),
    ((WINDOW_X, 0.0, WINDOW_Z), WINDOW_SIZE),
)
GATE_BBOX_CENTRE = (0.0, 0.0, GATE_H / 2)  # local centre of the box around every shape (what offline carries)

# Starting places: each gate standing on the bench beside its own arm, turned a quarter turn (crossbar along y,
# along the track's direction), so it must be turned square to the track before it is set over it.
GATES = ("gate_left", "gate_right")
START = {"gate_left": (0.200, 0.260), "gate_right": (0.200, -0.260)}
START_YAW = math.pi / 2

# Rule thresholds (scoring.py).
STAND_TOL = 0.003  # 'standing on the bench': every foot-bottom corner within this of the bench
SQUARE_DEG = 10.0  # 'square to the track': the crossbar within this of straight across it
SPACING_FULL = (0.190, 0.210)  # centres this far apart along the track: +0.5
SPACING_PART = (0.180, 0.220)  # +0.25 (outside the full band)
SPACING_AIM = 0.200
REST_WINDOW_S = 0.5  # 'at rest': neither foot moved more than REST_MOVE in the last 0.5 s
REST_MOVE = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a gate moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
ALUMINIUM = (0.78, 0.79, 0.81)
RUBBER_BLACK = (0.06, 0.06, 0.06)
GLIDER_BLUE = (0.15, 0.35, 0.75)
CARD_BLACK = (0.04, 0.04, 0.04)
GATE_BLACK = (0.10, 0.10, 0.11)
SENSOR_RED = (0.85, 0.12, 0.10)
