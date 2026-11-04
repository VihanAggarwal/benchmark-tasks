"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The press nest (the die): four fixed dark grey steel locating rails round a 36 x 36 mm floor (the table top),
# centred on the midline 30 cm in front of the arm bases. The 30 mm part fits with 3 mm of slack each way.
NEST_X, NEST_Y = 0.300, 0.0
NEST_INNER = 0.036  # the floor between the rails' inner faces, both ways
RAIL_THICK = 0.008  # each rail, across
RAIL_HEIGHT = 0.008  # each rail, tall
RAIL_TOP_Z = TOP + RAIL_HEIGHT
# The rails' inner faces are at x 0.282 / 0.318 and y -0.018 / +0.018; their outer faces 8 mm further out.

# The press frame (the press body): one fixed grey steel wall standing just behind the nest, 10 mm along x,
# 160 mm along y, 120 mm tall. The ram, which would sit above the nest, is not in the scene (nothing moves when the
# cycle runs), so the part can be loaded and unloaded straight down.
FRAME_X, FRAME_Y = 0.365, 0.0
FRAME_SIZE = (0.010, 0.160, 0.120)  # along x, along y, tall: its near face at x 0.36, 34 mm behind the back rail
FRAME_TOP_Z = TOP + FRAME_SIZE[2]

# The 'die space' the scorer watches: the nest's surroundings, x 0.25..0.35, y -0.05..+0.05, from the table top up to
# 20 cm above it. A gripper whose finger zone is wholly above DIE_TOP_Z, or outside that footprint, is out of it.
DIE_X0, DIE_X1 = 0.250, 0.350
DIE_Y0, DIE_Y1 = -0.050, 0.050
DIE_HEIGHT = 0.200
DIE_TOP_Z = TOP + DIE_HEIGHT

# The part: a loose blue aluminium block, a 30 mm cube, standing on the feed mark with its sides parallel to the axes.
PART_SIZE = (0.030, 0.030, 0.030)
PART_MASS = 0.070  # 27 cm3 of aluminium is 73 g; a lightly chamfered block
PART_HALF = PART_SIZE[0] / 2

# The feed mark: a fixed thin white plate the part starts on, at the left of the nest, in reach of both arms (the
# right arm reaches to about y +0.17 at this x).
PLATE_X, PLATE_Y = 0.240, 0.120
PLATE_SIZE = (0.040, 0.040, 0.002)  # along x, along y, thickness
PLATE_TOP_Z = TOP + PLATE_SIZE[2]

# The two run buttons: fixed yellow cylinders standing on the table at the far left and the far right of the bench,
# each in reach of one arm only (the right arm reaches no further left than about y +0.17 at this x, the left arm no
# further right than about y -0.18). 42 cm apart: no single gripper can be on both.
BUTTON_X = 0.200
BUTTON_Y = {"left": 0.210, "right": -0.210}  # which arm's button is which
BUTTON_RADIUS = 0.020
BUTTON_HEIGHT = 0.020
BUTTON_TOP_Z = TOP + BUTTON_HEIGHT

# The output tray: four fixed grey walls round a floor 80 mm along x and 140 mm along y (the table top), in front of
# the nest on the midline, in reach of both arms. 140 mm along y, the fingers' closing axis at yaw 0, leaves the open
# fingers (94 mm apart plus two 8 mm pads) 15 mm to each end wall.
TRAY_X, TRAY_Y = 0.160, 0.0
TRAY_INNER_X, TRAY_INNER_Y = 0.080, 0.140
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.015
TRAY_TOP_Z = TOP + TRAY_HEIGHT

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py).
PRESS_XY = 0.016  # 'pressed': a gripper's tip point within this of its button's axis horizontally,
PRESS_Z_BELOW = 0.005  # between this far below the button's top
PRESS_Z_ABOVE = 0.010  # and this far above it,
PRESS_OPENING = 0.5  # with the fingers at most this open (47 mm apart)
RELEASE_Z = 0.030  # 'released': the tip point more than this above the button's top,
RELEASE_XY = 0.030  # or more than this from its axis horizontally (clearly off the button, not hovering on it)
SEAT_XY_TOL = 0.004  # 'seated': the part's centre within this of the nest's centre horizontally,
SEAT_Z_TOL = 0.003  # its lowest point within this of the table top,
SEAT_TILT_DEG = 5.0  # and tilted less than this
SYNC_S = 0.5  # 'synchronous': the two presses began within this of each other (ISO 13851 type IIIC: 0.5 s)
CYCLE_S = 2.0  # 'cycle': both buttons held pressed, the part seated, for at least this long without a break
IN_TRAY_Z_TOL = 0.003  # 'in the tray': the part's lowest point within this of the table top, its centre over the floor
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from the part: it let go
OFF_TABLE_DROP = 0.010  # a part whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
SYNC_STEPS = int(round(SYNC_S * CONTROL_HZ))  # 10
CYCLE_STEPS = int(round(CYCLE_S * CONTROL_HZ))  # 40
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
START_Z_TOL = 0.003  # and its lowest point within this of the feed mark's top

# Colours (sRGB).
RAIL_GREY = (0.22, 0.22, 0.24)  # dark grey steel rails
FRAME_GREY = (0.42, 0.42, 0.45)  # the press frame
TRAY_GREY = (0.60, 0.60, 0.63)  # the output tray
PLATE_WHITE = (0.93, 0.93, 0.93)
PART_BLUE = (0.15, 0.35, 0.85)
BUTTON_YELLOW = (1.00, 0.90, 0.20)
