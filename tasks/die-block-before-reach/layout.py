"""Every dimension of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights named *_Z below are above the table top (add TOP for
the world z).

The die is one fixed mechkit assembly whose base frame sits at DIE_ORIGIN on the table top, axes along the world's,
so a world point (x, y, TOP + h) is (x - 0.310, y, h) in it. Its base holds the bed's top layer (with the part's
pocket and the white seat), the fixed back platen; its one link, the moving platen, slides toward -x (the arms) on
a spring-return prismatic joint 'platen' whose position j is the die's opening (0 shut, as drawn). The bed's bottom
slab, on which the part stands in its pocket, is a plain fixed box. The safety block is a loose mechkit assembly (a
bar and a handle arm, one rigid body) whose base frame sits at the middle of the bar's bottom face.
"""

import math

TOP = 0.7475  # station table top

# -- the die --------------------------------------------------------------------------------------------------------
DIE_ORIGIN = (0.310, 0.000, TOP)
BED_X0, BED_X1 = 0.225, 0.395
BED_Y0, BED_Y1 = -0.110, 0.110
BED_SLAB_Z = 0.008  # the plain fixed slab under everything: the pocket's floor
BED_TOP_Z = 0.018  # the top of the bed's upper layer (the die's base), where the platens and the seated block stand
# The pocket in the upper layer, round the part: 48 x 60 mm, 10 mm deep, its x walls in line with the two faces.
POCKET_X0, POCKET_X1 = 0.311, 0.359
POCKET_Y0, POCKET_Y1 = -0.030, 0.030
# The white seat, flush with the bed's top: where the safety block's bar lies when the die is blocked open.
SEAT_X0, SEAT_X1 = 0.261, 0.359
SEAT_Y0, SEAT_Y1 = -0.095, -0.065
# The fixed back platen (its face toward the arms at x 0.359).
FIXED_FACE_X = 0.359
FIXED_PLATEN = ((0.359, 0.379), (-0.100, 0.100), (0.018, 0.078))  # (x0, x1), (y0, y1), (z0, z1)
# The moving platen, drawn shut (j = 0): face plate, two cheeks either side of the part, the handle bracket and the
# black handle post. j moves every part of it by -j along x.
MOVING_FACE_X0 = 0.311  # its face (toward the fixed platen) at j = 0
FACE_PLATE = ((0.291, 0.311), (-0.100, 0.100), (0.018, 0.078))
CHEEK_LEFT = ((0.311, 0.355), (0.024, 0.034), (0.018, 0.058))
CHEEK_RIGHT = ((0.311, 0.355), (-0.034, -0.024), (0.018, 0.058))
BRACKET = ((0.261, 0.291), (0.045, 0.075), (0.063, 0.078))
POST = ((0.2605, 0.2855), (0.0475, 0.0725), (0.078, 0.118))
POST_X0, POST_Y = 0.273, 0.060  # the post's centre at j = 0
POST_SIZE = (0.025, 0.025, 0.040)
POST_Z0, POST_Z1 = 0.078, 0.118
PLATEN_TRAVEL = 0.075  # m: the joint's upper limit (lower 0: shut)
PLATEN_SPRING = 40.0  # N/m toward shut: 1.6 N at 40 mm open, 3.0 N at 75 mm
PLATEN_DAMPING = 3.0  # N s/m
PLATEN_MASS = 0.12  # kg: a hollow plastic platen (face plate, cheeks, bracket and post together)

# -- the part -------------------------------------------------------------------------------------------------------
PART_SIZE = (0.040, 0.040, 0.040)
PART_MASS = 0.080  # a solid 40 mm PLA cube (1.24 g/cm3: 79 g)
PART_X, PART_Y = 0.335, 0.000  # in its pocket: 4 mm to each face and to each cheek when the die is shut
PART_Z = BED_SLAB_Z + PART_SIZE[2] / 2  # its centre, standing on the pocket's floor

# -- the safety block -----------------------------------------------------------------------------------------------
BAR_SIZE = (0.098, 0.030, 0.030)  # the bar: exactly the face-to-face gap at j = 0.050
ARM_SIZE = (0.030, 0.060, 0.030)  # the handle arm, from the bar's middle toward -y
BAR_CENTER = (0.0, 0.0, 0.015)  # block base frame: origin at the middle of the bar's bottom face
ARM_CENTER = (0.0, -0.045, 0.015)
BLOCK_MASS = 0.15  # solid yellow ABS: 142 cm3 at 1.05 g/cm3
BLOCK_HEIGHT = 0.030
GRIP_FROM_BAR = 0.060  # the grip point on the handle arm: 60 mm from the bar's middle along -y (15 mm from its end)
# The stand: a fixed grey pad the block starts on and must end on.
STAND_X, STAND_Y = 0.200, -0.205
STAND_SIZE = (0.130, 0.110, 0.005)
STAND_TOP_Z = STAND_SIZE[2]
BLOCK_ORIGIN = (0.200, -0.180, TOP + STAND_TOP_Z)  # bar centre (0.200, -0.180); handle arm y -0.255..-0.195

# -- the tray -------------------------------------------------------------------------------------------------------
TRAY_X, TRAY_Y = 0.150, -0.070
TRAY_INNER = (0.100, 0.130)  # x, y: 130 mm across y, where the open fingers (yaw 0) span 110 mm with their pads
TRAY_FLOOR_Z = 0.003
TRAY_WALL = 0.004
TRAY_WALL_Z = 0.015

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------------
# The pinch zone (for the grippers' tool points): from 10 mm in front of the moving face to 10 mm into the fixed
# platen, 10 mm beyond the faces' ends, from the table up to 140 mm.
ZONE_MARGIN = 0.010
ZONE_X1 = FIXED_FACE_X + ZONE_MARGIN  # 0.369
ZONE_Y0, ZONE_Y1 = -0.110, 0.110
ZONE_Z1 = 0.140
# 'Blocked': the bar on the bed in its seat, square, the platen resting on it, no tool point near the handle post.
SEAT_TOL = 0.008  # every corner of the bar within 8 mm of the seat's outline (seen from above): a bar resting
#                   5 mm short of the fixed face (the demo sets it there) sticks out 5 mm at the near end
BAR_ON_BED_TOL = 0.003  # the bar's lowest point within this of the bed's top
BAR_YAW_DEG = 10.0  # the bar's long axis within this of x
BAR_TILT_DEG = 10.0  # the block lying flat (either broad face down)
BLOCKED_J = (0.045, 0.060)  # the opening while the platen rests on a 98 mm bar
FACE_ON_BAR_TOL = 0.004  # the moving face within this of the bar's near end: the spring presses it on the block
HAND_NEAR_POST = 0.060  # a tool point or finger-tip point this close to the handle post's box: a hand may be
#                         holding the die open (an open finger's outer face is 57 mm from the tool point)
HELD_POST_DIST = 0.010  # a gripper closed (opening < GRIP_OPEN) with its tool point this close to the post: holding
HELD_PLATEN_DIST = 0.010  # ... or this close to the moving platen's face plate, bracket or post (not the cheeks,
#                           inside the pinch zone beside the part): a closed gripper there holds the die, not blocked
PULLED_OPEN_J = 0.045  # the die counts as pulled open by hand once a gripper has held the post with j at least
#                        this, since the die was last shut (j <= CLOSED_J): pushing it open with the block
#                        itself (one arm turning the bar between the faces) never blocks it
SLAM_SPEED = 0.10  # m/s: the platen closing faster than this while nobody holds the post
CLOSED_J = 0.003  # the die shut
CLOSED_ON_J = (0.003, 0.045)  # at the end, open this much with nobody holding the post: shut on something
BLOCKED_HOLD_S = 1.0  # 'blocked for 1 s': this many seconds in a row
FALL_SPEED = 0.25  # m/s: the part or the block moving down faster than this has fallen (or was slammed down)
OFF_TABLE = 0.010  # an object whose lowest point is this far below the table top has left it
LIFT_MIN = 0.030  # 0.5: the part's bottom raised this much above where it started ...
# ... and its centre outside the die's footprint (the bed's outline seen from above) at some point.
IN_TRAY_Z_TOL = 0.005  # the part's lowest point within this of the tray's floor
STAND_Z_TOL = 0.003  # the block's lowest point within this of the stand's top
GRIP_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
CLEAR_DIST = 0.030  # 1.0: each gripper's tool point and finger-tip point this far from the part, the block and the
#                     die's moving platen
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
STILL_JOINT = 0.005  # m/s: the platen
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # the part or the block moving this much further in one step than its velocity allows: teleported
START_TOL = 0.045  # ... back onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# -- the offline stand-in (standin.py; no physics offline: tools/offline/check_task.py) -------------------------------
STANDIN_GRIP = 0.6  # a gripper closing past this opening takes hold of what its tool point is at
STANDIN_REACH = 0.006  # ... its tool point within this of the post's box, or of the handle arm's box
STANDIN_TAU = PLATEN_DAMPING / PLATEN_SPRING  # 0.075 s: a released platen closes like a first-order lag

# Colours (sRGB).
STEEL = (0.50, 0.52, 0.55)
PLATEN_GREY = (0.42, 0.44, 0.47)
POST_BLACK = (0.06, 0.06, 0.07)
SEAT_WHITE = (0.95, 0.95, 0.95)
PART_GREEN = (0.20, 0.65, 0.25)
BLOCK_YELLOW = (0.95, 0.80, 0.10)
STAND_GREY = (0.33, 0.33, 0.35)
TRAY_GREY = (0.70, 0.70, 0.72)

DEG = math.pi / 180.0
