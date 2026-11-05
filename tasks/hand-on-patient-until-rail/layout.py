"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The bed: a fixed white platform standing in for the bed's deck, between the arms and a little to the right, so
# that both arms reach it (straight down, the left arm reaches no further right than about y -0.18 and the right
# no further left than about y +0.17 at x 0.18; about -0.13 and +0.12 at x 0.28).
BED_SIZE = (0.200, 0.300, 0.030)  # along x, along y, height
BED_X, BED_Y = 0.28, -0.03
DECK_TOP = TOP + BED_SIZE[2]  # the deck's top face
DECK_X0, DECK_X1 = BED_X - BED_SIZE[0] / 2, BED_X + BED_SIZE[0] / 2  # 0.18 .. 0.38
DECK_Y0, DECK_Y1 = BED_Y - BED_SIZE[1] / 2, BED_Y + BED_SIZE[1] / 2  # -0.18 .. +0.12

# The bed mark: a thin fixed pale green plate on the deck's top, where the patient goes (its left half, within the
# left arm's reach and 20 cm from the rail's place).
MARK_SIZE = (0.080, 0.070, 0.001)
MARK_X, MARK_Y = 0.28, 0.05
MARK_TOP = DECK_TOP + MARK_SIZE[2]
MARK_X0, MARK_X1 = MARK_X - MARK_SIZE[0] / 2, MARK_X + MARK_SIZE[0] / 2  # 0.24 .. 0.32
MARK_Y0, MARK_Y1 = MARK_Y - MARK_SIZE[1] / 2, MARK_Y + MARK_SIZE[1] / 2  # 0.015 .. 0.085

# The rail's place: along the deck's right (-y) edge, the open side of the bed. A rail 30 mm wide centred here
# spans y -0.18 .. -0.15, its outer face flush with the deck's edge.
RAIL_PLACE_X, RAIL_PLACE_Y = 0.28, -0.165

# The trolley: a fixed dark grey platform standing in for the trolley's deck, at the left, 40 mm from the bed's
# left edge (y 0.12 .. 0.16 is the gap), out of the right arm's straight-down reach (a grip rolled 20-40 degrees
# sideways could just pinch the patient there: see the README).
TROLLEY_SIZE = (0.200, 0.160, 0.030)
TROLLEY_X, TROLLEY_Y = 0.28, 0.24
TROLLEY_TOP = TOP + TROLLEY_SIZE[2]  # the same height as the bed's deck
TROLLEY_X0, TROLLEY_X1 = TROLLEY_X - TROLLEY_SIZE[0] / 2, TROLLEY_X + TROLLEY_SIZE[0] / 2  # 0.18 .. 0.38
TROLLEY_Y0, TROLLEY_Y1 = TROLLEY_Y - TROLLEY_SIZE[1] / 2, TROLLEY_Y + TROLLEY_SIZE[1] / 2  # 0.16 .. 0.32

# The patient: a loose skin-pink block standing in for the patient's torso, lying centred on the trolley. Only the
# left arm reaches it there with a straight-down grip; it is gripped across its 70 mm (the fingers closing along
# x), 36 of its 40 mm held.
PATIENT_SIZE = (0.070, 0.060, 0.040)  # along x, along y, height
PATIENT_MASS = 0.20
PATIENT_X, PATIENT_Y = TROLLEY_X, TROLLEY_Y
PATIENT_HALF = tuple(v / 2 for v in PATIENT_SIZE)

# The rail: a loose light grey bar (hollow aluminium) lying on the table at the right, long side along x, out of
# the left arm's straight-down reach; gripped at its centre across its 30 mm (the fingers closing along y).
RAIL_SIZE = (0.200, 0.030, 0.040)
RAIL_MASS = 0.10
RAIL_X, RAIL_Y = 0.28, -0.27
RAIL_HALF = tuple(v / 2 for v in RAIL_SIZE)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap (the real finger widens to about 10 mm thick and 40 mm
#                       wide near its base, up to 7 mm beyond this zone: inside HAND_GAP, so a touching finger is
#                       never 'let go'; 'a hand on' uses the pads' own size so that a gripper merely near does not count)
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py).
ON_BED_Z_TOL = 0.010  # 'on the bed' (the unattended rule): any part of the patient's outline over the deck and its
#                       lowest point within this of the deck's top, above or below (resting, just above, or half on)
REST_Z_TOL = 0.003  # 'resting on the bed': its lowest point within this of the deck's top (on the mark it is 1 mm up)
FLAT_DEG = 5.0  # 'flat': tilted less than this
TIP_DEG = 15.0  # the patient tilted more than this at any moment: 0
DROP_SPEED = 0.15  # m/s: the patient descending faster than this at any moment: dropped, 0
REST_BAND = 0.003  # within this of the surface under it only the step-to-step descent counts (a contact transient
#                    as it lands is not a drop); above it its downward velocity counts as well
DRAG_MAX = 0.015  # the patient's centre travelling more than this horizontally while low off the bed (slid along
#                   the trolley or the table instead of lifted): 0
DRAG_BAND = 0.010  # 'low': its lowest point less than this above the surface under it (resting, or skimming), its
#                    centre not over the bed deck (sliding the patient on the bed to position them is allowed)
DRAG_STEP = 0.0005  # drag is followed in steps of this: a move counts once the centre is this far from where drag
#                     was last counted, so a resting body's sub-millimetre contact jitter does not add up
CENTRE_TIGHT = 0.015  # 1.0: the patient's centre within this of the mark's centre, horizontally
CENTRE_LOOSE = 0.030  # 0.75
RAIL_XY_TOL = 0.010  # 'rail up': its centre within this of its place, horizontally ...
RAIL_HEADING_DEG = 5.0  # ... its long axis along x within this ...
RAIL_TILT_DEG = 5.0  # ... tilted less than this ...
RAIL_Z_TOL = 0.003  # ... and its lowest point within this of the deck's top
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
HAND_GAP = 0.010  # a gripper less open than that with its finger zone overlapping the patient's box and its finger
#                   tips no higher than this above the patient's top: a hand on them; a gripper open, or its zone
#                   further than this from an item, has let go of it
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
ON_TABLE_TOL = 0.010  # the patient's lowest point within this of the table top (or below it) over neither platform:
#                       it ended on the table, off both
LIFT_CLEAR = 0.020  # 'lifted off the trolley' (traced, not scored): its lowest point this far above the trolley's top
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
START_Z_TOL = 0.003  # and its lowest point within this of the surface it starts on

# Colours (sRGB).
BED_WHITE = (0.95, 0.95, 0.95)
MARK_GREEN = (0.70, 0.90, 0.72)
TROLLEY_GREY = (0.35, 0.36, 0.38)
SKIN_PINK = (0.93, 0.72, 0.62)
RAIL_GREY = (0.72, 0.73, 0.75)
