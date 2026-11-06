"""Every dimension, position, mass, colour and threshold of the task, in one place (metres, kilograms, degrees where a
name says so; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left of their midpoint, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# (1) The planter: a fixed open trough on the midline, built from a floor and four walls (all fixed boxes).
PLANTER_X, PLANTER_Y = 0.270, 0.000
INNER_X0, INNER_X1 = 0.190, 0.350  # inside, along x: 160 mm
INNER_HALF_Y = 0.170  # inside, along y: 340 mm (y -0.170 .. +0.170)
WALL = 0.008  # wall thickness
FLOOR_T = 0.005  # floor thickness: the floor's top is 5 mm up
RIM = 0.086  # the walls' top, the planter's rim, 86 mm up
OUTER_X = INNER_X1 - INNER_X0 + 2 * WALL  # 0.176
OUTER_Y = 2 * INNER_HALF_Y + 2 * WALL  # 0.356
WALL_H = RIM - FLOOR_T  # 0.081: the walls stand on the floor
FLOOR_TOP = FLOOR_T

# (2) The two station discs: thin fixed cylinders on the planter floor, one for each plant.
DISC_R = 0.045  # 90 mm across
DISC_T = 0.001
DISC_X = 0.270
DISC_Y = 0.110  # discs at y +0.110 (left) and -0.110 (right)
DISC_TOP = FLOOR_TOP + DISC_T  # 6 mm up
DISCS = {"disc_left": (DISC_X, +DISC_Y), "disc_right": (DISC_X, -DISC_Y)}

# (3) The two pots: filled plastic grow pots, closed straight-sided cylinders 70 mm across (plant left out).
POT_R = 0.035
POTS = {
    # name: (height, mass, x, y)
    "pot_a": (0.035, 0.100, 0.200, +0.385),  # a bulb pan, on the left arm's side
    "pot_b": (0.046, 0.130, 0.200, -0.385),  # an azalea pot, on the right arm's side
}

# (4) The six risers: white high-density rigid foam pucks, 60 mm across, three heights, standing upright.
RISER_R = 0.030
RISERS = {
    # name: (height, mass, x, y); the 100 mm pitch along x leaves room for the open fingers between them
    "riser_l20": (0.020, 0.015, 0.170, +0.260),
    "riser_l30": (0.030, 0.022, 0.270, +0.260),
    "riser_l42": (0.042, 0.030, 0.370, +0.260),
    "riser_r30": (0.030, 0.022, 0.170, -0.260),
    "riser_r42": (0.042, 0.030, 0.270, -0.260),
    "riser_r20": (0.020, 0.015, 0.370, -0.260),
}
LOOSE = (*POTS, *RISERS)  # every loose object, in a fixed order

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939

# Rule thresholds (scoring.py). Every bound stated as inclusive is compared with a 1e-9 tolerance.
# The tolerances are set so that, stacked every way the scorer allows, no wrong riser can bring a pot's rim into the
# 10-20 mm band (closest: pot A on the 20 mm riser 20.2 mm down; tests/test_scoring.py checks every corner).
POT_UPRIGHT_DEG = 3.0  # a set pot's axis within this of vertical (either end up)
DISC_TOL = 0.020  # a set pot's centre, seen from above, within this of a disc's centre
STEP_TOL = 0.0015  # its bottom face's centre within this (in height) of a riser's top face's centre
AXIS_TOL = 0.035  # ... and the two faces' centres within this of each other, seen from above
RISER_UPRIGHT_DEG = 2.0  # that riser's axis within this of vertical (either end up)
RISER_LOW_MIN, RISER_LOW_MAX = 0.004, 0.0075  # the centre of that riser's bottom face this high: floor (5) or disc (6;
# 7 when tilted 2 degrees on its edge)
STILL_STEPS = 10  # 'still' is judged over the last 0.5 s (10 control steps) ...
STILL_PATH = 0.003  # ... the pot's centre moved along a path shorter than this
RELEASE_DIST = 0.060  # 'let go': each gripper's TCP at least this far from the pot's centre
GAP_MIN, GAP_MAX = 0.010, 0.020  # a staged pot's rim gap (planter rim minus the pot's highest rim point)
RESET_JUMP = 0.08  # an object moving this far in one control step was put back by the simulator: a new episode ...
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
CHARCOAL = (0.20, 0.21, 0.22)
DISC_GREY = (0.55, 0.55, 0.55)
TERRACOTTA = (0.74, 0.40, 0.25)
WHITE = (0.95, 0.95, 0.95)
