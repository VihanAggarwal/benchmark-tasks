"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The litter: a loose flat board of pale grey polymer standing in for a two-bearer litter, lying flat on the table
# across the midline, long side along y, so that each arm is at one end. Its outer 100 mm at each end (|y| from
# 0.15 to 0.25) are the handles; the middle 300 mm are covered by the casualty.
LITTER_SIZE = (0.040, 0.500, 0.020)  # along x, along y, thickness
LITTER_MASS = 0.20
LITTER_X, LITTER_Y = 0.24, 0.0
LITTER_TOP = TOP + LITTER_SIZE[2]  # its top face at the start, where the casualty's underside rests
HANDLE_LEN = 0.100  # each end's handle section (|y| 0.15..0.25)
HANDLE_Y = 0.21  # where each gripper takes its handle, measured from the midline (centre of the 27 mm pad: |y|
#                  0.1965..0.2235, 46 mm clear of the casualty and 26 mm short of the litter's end)

# The casualty: a loose orange box standing in for the patient, lying centred on the litter. It is wider than the
# litter (70 mm against 40 mm) and covers its middle 300 mm, so a gripper can take the litter only by its ends.
CASUALTY_SIZE = (0.070, 0.300, 0.040)  # along x, along y, height
CASUALTY_MASS = 0.30
CASUALTY_X, CASUALTY_Y = LITTER_X, LITTER_Y
CASUALTY_BOTTOM = LITTER_TOP  # its underside at the start

# The trolley: a fixed dark grey platform standing in for the ambulance trolley's deck, 14 cm in front of the
# litter. The litter's gripped ends (|y| 0.21) and the pads (27 mm wide along y) stay clear of its y extent
# (|y| <= 0.18) when the litter is set down centred on it.
TROLLEY_SIZE = (0.120, 0.360, 0.030)  # along x, along y, height
TROLLEY_X, TROLLEY_Y = 0.38, 0.0
TROLLEY_TOP = TOP + TROLLEY_SIZE[2]
TROLLEY_HALF_X, TROLLEY_HALF_Y = TROLLEY_SIZE[0] / 2, TROLLEY_SIZE[1] / 2  # it spans x 0.32..0.44, y -0.18..0.18

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it; the flat pad faces are about 22 mm of that)

# Rule thresholds (scoring.py).
REST_TOL = 0.003  # 'resting': the litter's lowest point within this of the surface it is over (table or trolley)
LIFT_CLEAR = 0.010  # 'lifted': its lowest point at least this far above that surface
END_UP = 0.010  # 'an end is up': either end's bottom edge at least this far above the table top (the litter is
#                 being handled: lifted, lifted at one end, or on the trolley); the end-difference and tilt records
#                 are kept over these steps
CARRY_CLEAR = 0.030  # 'carried level' (0.25): the lowest point at least this far above the table top ...
LEVEL_DEG = 5.0  # ... with the long axis within this of horizontal ...
END_DIFF_FULL = 0.020  # ... and the ends' heights differing by no more than this (also the 1.0 limit, over the trial)
END_DIFF_MAX = 0.040  # the ends' heights differing by more than this while an end is up (lifted at one end): 0
TILT_FULL = 5.0  # the litter tilted more than this while an end is up: 0.75 at most
TILT_MAX = 10.0  # the litter tilted more than this at any moment: 0
DRAG_MAX = 0.015  # the litter's centre travelling more than this horizontally while low (dragged): 0
DRAG_BAND = LIFT_CLEAR  # 'low': its lowest point less than this above the surface it is over (resting, or skimming)
DRAG_STEP = 0.0005  # drag is followed in steps of this: a move counts once the centre is this far from where drag
#                     was last counted, so a resting body's sub-millimetre contact jitter does not add up
CASUALTY_GAP = 0.006  # 'casualty on the litter': its underside within this of the litter's top ...
CASUALTY_OFF = 0.030  # ... its centre within this of the litter's centre horizontally ...
CASUALTY_TILT_ON = 10.0  # ... and tilted less than this
CASUALTY_TIP_DEG = 15.0  # the casualty tilted more than this at any moment: 0
CASUALTY_OFF_STEPS = 5  # 'not on the litter' must hold this many control steps in a row (0.25 s) to count: a
#                         one-step contact hop as the litter is lifted or set down is not the casualty leaving it
FLAT_DEG = 3.0  # 'rests flat on the trolley': its broad face tilted less than this (covers a roll as well)
HEADING_DEG = 10.0  # ... and its long axis along the trolley's (y) within this
ON_TROLLEY_Z_TOL = REST_TOL  # ... its lowest point within this of the trolley's top
SETDOWN_WINDOW = LIFT_CLEAR  # the set-down speed is watched over the trolley in the last 10 mm above it ...
SETDOWN_MAX = 0.10  # ... and a descent faster than this (m/s) before it comes to rest is a drop: 0.75 at most
CENTRE_X_FULL, CENTRE_Y_FULL = 0.015, 0.020  # 1.0: the litter's centre within these of the trolley's, in x and y
CENTRE_LOOSE = 0.040  # 0.75: within this in x and in y
ON_TROLLEY_TILT = TILT_MAX  # 0.5: resting on the trolley, tilted less than this, its centre over the trolley
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from the litter: it let go
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
PALE_GREY = (0.82, 0.83, 0.85)
ORANGE = (0.95, 0.50, 0.12)
DARK_GREY = (0.28, 0.29, 0.31)
