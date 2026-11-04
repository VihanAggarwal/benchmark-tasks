"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The wound mark: a thin pink pad fixed to the table on the midline (the skin round the wound, to be dressed).
MARK_X, MARK_Y = 0.30, 0.0
MARK_SIZE = (0.070, 0.070, 0.002)
MARK_HALF = MARK_SIZE[0] / 2
MARK_TOP = TOP + MARK_SIZE[2]

# The used dressing: a loose soiled pad lying centred on the wound mark (right gripper only).
DRESSING_SIZE = (0.050, 0.050, 0.016)
DRESSING_MASS = 0.020  # a 40 cm3 absorbent pad holding exudate (0.5 g/cm3)
DRESSING_X, DRESSING_Y = MARK_X, MARK_Y

# The clean pad: a loose sterile pad lying in the sterile tray, in front of the left arm (left gripper only).
PAD_SIZE = (0.050, 0.050, 0.016)
PAD_MASS = 0.010  # a dry, tightly folded 40 cm3 gauze pad (0.25 g/cm3)
PAD_HALF = PAD_SIZE[0] / 2

# The sterile tray: four fixed walls round a 140 x 70 mm floor (the table top), long side along x. A 70 mm floor
# would leave no room for the open fingers (94 mm apart plus 8 mm pads) inside 15 mm walls; 140 mm along x, the
# fingers' closing axis, gives them 15 mm to each end wall.
TRAY_X, TRAY_Y = 0.24, 0.22
TRAY_INNER_X, TRAY_INNER_Y = 0.140, 0.070
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.015
TRAY_TOP_Z = TOP + TRAY_HEIGHT

# The waste bin: four fixed walls round a 90 x 90 mm floor (the table top), in front of the right arm.
BIN_X, BIN_Y = 0.22, -0.24
BIN_INNER = 0.090
BIN_WALL = 0.008
BIN_HEIGHT = 0.040
BIN_INNER_HALF = BIN_INNER / 2
BIN_OUTER_HALF = BIN_INNER / 2 + BIN_WALL  # 0.053
BIN_TOP_Z = TOP + BIN_HEIGHT

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py).
TOUCH_GAP = 0.005  # the left gripper's finger zone this close to the used dressing's box: touching it
KEEP_OUT = 0.030  # the right gripper's finger zone must stay this far from the clean pad's box
PAD_DRESSING_GAP = 0.003  # the two pads' boxes this close: touching
BARE_TABLE_TOL = 0.001  # a corner of the clean pad this close to the table top (or below it): touching the table
TRAY_FLOOR_SLACK = 0.002  # a corner within this of the tray's inner walls still counts as over its floor
PAD_BIN_GAP = 0.003  # the clean pad's box this close to the bin's outer box: touching the bin
FLAT_DEG = 5.0  # 'flat': tilted less than this (either 50 x 50 mm face down)
ON_MARK_Z_TOL = 0.003  # 'on the wound mark': the clean pad's lowest point within this of the mark's top
CENTRE_TIGHT = 0.010  # clean pad centre to the wound mark's centre, horizontally (1.0)
CENTRE_LOOSE = 0.025  # (0.75)
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from the item it handles: it let go
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
PINK = (0.90, 0.60, 0.60)
STAINED = (0.75, 0.65, 0.45)
WHITE = (0.97, 0.97, 0.97)
TRAY_BLUE = (0.30, 0.50, 0.85)
BIN_YELLOW = (0.95, 0.80, 0.10)
