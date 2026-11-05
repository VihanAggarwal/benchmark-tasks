"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. The station table runs from x -0.19 to x 0.91 and from
y -0.65 to y +0.65.

The beam path the demo sets up (seen from above):

    laser (0.12, +0.24) --> +x --> block spot (0.22) --> M1 (0.33, +0.24, yaw +45 deg) --> -y
    M1 --> -y --> M2 (0.33, -0.16, yaw -45 deg) --> -x --> detector face (x 0.21, y -0.16)

The right arm cannot reach the beam (y +0.24) and the left arm cannot reach the right mirror's starting place,
so a beam off both mirrors needs both arms.
"""

TOP = 0.7475  # station table top

# The station table's outline (a beam that leaves it without being stopped is stray).
TABLE_X = (-0.19, 0.91)
TABLE_Y = (-0.65, 0.65)

# The beam: horizontal, 40 mm above the table top, leaving the laser's front face along +x.
BEAM_H = 0.040
BEAM_Z = TOP + BEAM_H

# The laser: a fixed housing, its aperture on the +x face, centred on the beam.
LASER_X, LASER_Y = 0.12, 0.24
LASER_SIZE = (0.040, 0.040, 0.060)
APERTURE_X = LASER_X + LASER_SIZE[0] / 2  # 0.14

# The beam dump: a fixed black box at the far end of the beam's starting path.
DUMP_X, DUMP_Y = 0.50, 0.24
DUMP_SIZE = (0.030, 0.060, 0.060)

# Two thin red lines either side of the beam's starting path: a corridor, not a line under the beam, because a
# mirror standing on a line would tilt. Centres 55 mm either side of the beam, 10 mm wide (2.5 pixels in the
# 320-pixel camera view), 2 mm tall; a mirror at 45 degrees with the beam on the middle of its face clears them by
# 15 mm.
BEAM_LINE_X0, BEAM_LINE_X1 = 0.145, 0.480
LINE_OFFSET = 0.055
LINE_WIDTH = 0.010
LINE_HEIGHT = 0.002

# The detector: a fixed yellow sensor head, its sensitive face toward +x; its centre is the point on that face at
# beam height (40 mm), on the face's vertical centre line. 20 cm in front of the right arm's base line, so the
# path from the first mirror to it is 0.51 m (a mirror turned 0.1 degree moves the spot 2 mm per metre of path).
DET_X, DET_Y = 0.20, -0.16
DET_SIZE = (0.020, 0.040, 0.060)
DET_FACE_X = DET_X + DET_SIZE[0] / 2  # 0.21
DET_HALF_W = DET_SIZE[1] / 2  # 0.02

# Two thin black lines either side of the detector's axis (the line along x through its centre).
DET_LINE_X0, DET_LINE_X1 = 0.215, 0.400

# The backstop: a fixed black board behind the detector, so a beam that misses the detector ends on the table.
# 15 mm behind the detector.
BACK_X, BACK_Y = 0.170, -0.16
BACK_SIZE = (0.010, 0.200, 0.060)

# The mirrors: loose, a double-sided flat mirror in a 20 mm mount. Each 60 x 60 mm face has a 40 x 40 mm mirror
# window in its middle, framed by the mount's 10 mm rim: only the windows reflect; the rim and the 20 mm edges stop
# the beam. The rim is what makes 'slide the mirror into the live beam' visible to rule 2: the beam meets the rim
# first, and the mirror has to move at least 10 mm x cos(incidence) further (7.1 mm at 45 degrees) while lit before
# the beam reaches the window. Local x (the 20 mm side) is the faces' normal, local y the 60 mm width, local z the
# 60 mm height.
MIRROR_SIZE = (0.020, 0.060, 0.060)
MIRROR_FRAME = 0.010  # the rim round each 40 x 40 mm window
MIRROR_MASS = 0.080  # two 40 x 40 x 2 mm glass mirrors (16 g) in a plastic mount
MIRROR_L_X, MIRROR_L_Y = 0.32, 0.38  # in front of the left arm, outside the beam corridor; faces toward x
MIRROR_R_X, MIRROR_R_Y = 0.24, -0.36  # in front of the right arm

# The beam block: a loose orange box.
BLOCK_SIZE = (0.040, 0.040, 0.060)
BLOCK_MASS = 0.060  # painted softwood (0.6 g/cm3)
BLOCK_X, BLOCK_Y = 0.20, 0.36

LOOSE = ("mirror_left", "mirror_right", "block")
MIRRORS = ("mirror_left", "mirror_right")
SIZES = {"mirror_left": MIRROR_SIZE, "mirror_right": MIRROR_SIZE, "block": BLOCK_SIZE}
STARTS = {"mirror_left": (MIRROR_L_X, MIRROR_L_Y), "mirror_right": (MIRROR_R_X, MIRROR_R_Y),
          "block": (BLOCK_X, BLOCK_Y)}

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Rule thresholds (scoring.py).
CENTRE_TIGHT = 0.010  # beam spot to the detector's centre on its face (1.0)
LIT_MOVE = 0.003  # a mirror the beam is on moving this far (m) ...
LIT_TURN_DEG = 1.0  # ... or turning this much from its last place in the dark: moved in the live beam
MAX_BOUNCES = 12  # more reflections than this: the beam is trapped between the mirrors (stopped on the table)
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from both mirrors and the block
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
LASER_BLUE = (0.15, 0.25, 0.60)
BEAM_RED = (0.85, 0.10, 0.10)
DUMP_BLACK = (0.06, 0.06, 0.06)
DET_YELLOW = (0.95, 0.80, 0.10)
LINE_BLACK = (0.05, 0.05, 0.05)
MIRROR_SILVER = (0.78, 0.80, 0.84)
BLOCK_ORANGE = (0.95, 0.45, 0.10)
