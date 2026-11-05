"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# -- the enabling station (left of the midline, in front of the left arm only) --------------------------------
# A square sleeve of four fixed walls round a 30 x 30 mm opening (the table is its floor), 80 mm tall.
SLEEVE_X, SLEEVE_Y = 0.220, 0.200
SLEEVE_INNER = 0.030
SLEEVE_WALL = 0.008
SLEEVE_HEIGHT = 0.080
SLEEVE_HALF = SLEEVE_INNER / 2
SLEEVE_TOP_Z = TOP + SLEEVE_HEIGHT

# The enabling plunger: a loose upright cylinder standing in the sleeve on the table, 40 mm proud of the sleeve.
# A closed black-anodised aluminium tube: a solid aluminium rod this size (54 cm3) would weigh 147 g.
PLUNGER_RADIUS = 0.012
PLUNGER_HEIGHT = 0.120
PLUNGER_MASS = 0.10
PLUNGER_GRIP_BELOW_TOP = 0.020  # its grip point: on its axis, 20 mm below its top (TCP z 0.100 above the table)

# The scale post beside the sleeve (fixed, 10 x 10 mm): white up to the plunger's resting top, then a red band
# (plunger top 0-15 mm up: not enabled), a green band (15-30 mm up: the enable band) and a red band (pulled too
# high). The bands are 12 x 12 mm, stacked, each 15 mm tall.
POST_X, POST_Y = 0.220, 0.243
POST_WIDTH = 0.010
BAND_WIDTH = 0.012
POST_WHITE_TOP = 0.120  # above the table top
BAND_HEIGHT = 0.015

# -- the stage (right of the midline, in front of the right arm only) -----------------------------------------
PLATE_X, PLATE_Y = 0.360, -0.160
PLATE_SIZE = 0.200  # square
PLATE_THICK = 0.006
PLATE_TOP_Z = TOP + PLATE_THICK
FRAME_WALL = 0.010  # the frame's walls stand on the plate's edge: 180 x 180 mm inside
FRAME_HEIGHT = 0.020
FRAME_INNER_HALF = PLATE_SIZE / 2 - FRAME_WALL  # 0.090
FRAME_TOP_Z = PLATE_TOP_Z + FRAME_HEIGHT
MARK_SIZE = 0.010  # 10 x 10 mm, 1 mm thick, on the wall tops
MARK_THICK = 0.001

# The targets (the carriage's centre, horizontally) and the home mark where it starts.
HOME = (0.360, -0.160)
RED = (0.400, -0.190)
BLUE = (0.330, -0.110)
TARGETS = {"red": RED, "blue": BLUE, "home": HOME}

# The carriage: a loose white PLA cube printed solid (64 cm3 at 1.24 g/cm3: 79 g) on the plate at home.
CARRIAGE_SIZE = 0.040
CARRIAGE_MASS = 0.08
CARRIAGE_HALF = CARRIAGE_SIZE / 2

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports 0 (closed) .. 1 (open) -----------
FULL_GAP = 0.0939

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------
ENABLE_LO, ENABLE_HI = 0.015, 0.030  # the enable band: the plunger's lowest point this high above the table top
# 'Upright' while enabled: tilted under this (degrees). To jam across its 30 mm sleeve with its foot 15-30 mm up, the
# 24 mm plunger must tilt at least 5.4 degrees, so an upright plunger in its band cannot be staying up on its own.
ENABLE_TILT_MAX = 4.0
HELD_RADIUS = 0.025  # 'held': a gripper's TCP within this of the plunger's grip point ...
HELD_OPENING = 0.5  # ... with the gripper closed past this opening (47 mm apart)
PLUNGER_DOWN = 0.002  # 'let down': the plunger's lowest point within this of the table top
PLUNGER_TILT_MAX = 15.0  # degrees
CARRIAGE_TILT_MAX = 10.0  # degrees (any face down)
LIFT_MAX = 0.005  # the carriage's lowest point this far above the plate's top: carried, not guided
IDLE_TRAVEL_MAX = 0.002  # horizontal travel of the carriage while not enabled
RULER = 0.0002  # travel is followed in steps of 0.2 mm (a shiver inside 0.2 mm adds nothing)
SPEED_CAP = 0.050  # m/s, the carriage's horizontal speed ...
SPEED_STEPS = 2  # ... over the last 2 control steps (0.1 s)
STOP_TOL = 0.003  # 'stopped at a point': centre within 3 mm of the target, horizontally ...
STOP_SPEED = 0.002  # ... speed under 2 mm/s ...
STOP_S = 1.0  # ... for 1 s, enabled throughout
HOME_END_TOL = 0.005  # at the end, the carriage's centre within this of the home mark
GUIDE_MIN = 0.020  # 0.25: the carriage's centre at least 20 mm from where it was when the stretch began ...
GUIDE_S = 3.0  # ... in one continuous enabled stretch of at least 3 s
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart)
AWAY = 0.030  # at the end each gripper's tip point and TCP at least this far from the carriage and the plunger
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.008  # an object moving this much further in one step than its velocity and a free fall allow ...
START_TOL = 0.010  # ... onto its starting place (within this of it), everything at rest: a reset

# Colours (sRGB).
GREY = (0.55, 0.55, 0.58)
STAGE_GREY = (0.80, 0.80, 0.80)
WHITE = (0.97, 0.97, 0.97)
BLACK = (0.08, 0.08, 0.08)
RED_C = (0.85, 0.15, 0.12)
GREEN_C = (0.15, 0.70, 0.25)
BLUE_C = (0.15, 0.35, 0.90)
