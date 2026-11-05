"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The ring is a 'pinwheel' of four identical socks round the spill: each sock's end stops just short of the inner
side of the next one, so the four close a square with no corner gaps. With socks of length l and width w and a
joint gap g, the inner faces sit (l - w + g) / 2 from the spill's centre: 87.5 mm here, 27.5 mm outside the spill's
edge and 22.5 mm outside the keep-out circle. (Socks 150 mm long, as first sketched, would leave a 120 mm hole,
inside the 130 mm keep-out circle: a closed ring of four needs socks of about 200 mm.)

Each sock is gripped 40 mm from its centre toward its 'corner' end (the end flush with the next sock's outer side),
so a gripper working on the ring stays 117.5 mm from the spill's centre: its finger zone, opened to 50 mm, keeps
12 mm outside the keep-out circle, and its fingers 10 mm from the neighbouring sock's inner side.
"""

import math

TOP = 0.7475  # station table top

# The spill: a thin dark disc fixed to the table, the hazard nothing may come over.
SPILL_X, SPILL_Y = 0.28, 0.0
SPILL_R = 0.060
SPILL_H = 0.001
KEEP_OUT = 0.005  # nothing may come within this of the spill's edge (seen from above, at any height)
ZONE_R = SPILL_R + KEEP_OUT  # 0.065: the keep-out circle
NEAR = 0.060  # a placed sock lies within this of the spill's edge (seen from above)
PLACED_R = SPILL_R + NEAR  # 0.120: ... so within this of its centre

# The absorbent socks: four loose identical bars, square in section.
SOCK_LEN, SOCK_W, SOCK_H = 0.200, 0.030, 0.030
SOCK_SIZE = (SOCK_LEN, SOCK_W, SOCK_H)  # before yaw: long along x
SOCK_MASS = 0.050  # 180 cm3: polypropylene absorbent packed firmly in a fabric sleeve (0.28 g/cm3)

# The ring (pinwheel, counter-clockwise seen from above): sock centres relative to the spill's centre.
JOINT = 0.005  # nominal gap at each joint in the demo (1.0 allows 10 mm)
RING_A = (SOCK_LEN - SOCK_W + JOINT) / 2  # 0.0875: spill centre to each sock's inner face
_OFF = RING_A + SOCK_W / 2  # 0.1025: spill centre to each sock's centre line
_ALONG = SOCK_LEN / 2 + JOINT - RING_A  # 0.0175: each sock's centre shifted along its length
RING = {  # place -> (x, y, yaw) in the world
    "left": (SPILL_X + _ALONG, SPILL_Y + _OFF, 0.0),  # along x, on the left (+y) of the spill
    "front": (SPILL_X + _OFF, SPILL_Y - _ALONG, math.pi / 2),  # along y, in front (+x)
    "right": (SPILL_X - _ALONG, SPILL_Y - _OFF, 0.0),  # along x, on the right (-y)
    "back": (SPILL_X - _OFF, SPILL_Y + _ALONG, math.pi / 2),  # along y, behind (-x)
}

# Where the demo grips each sock as it lays it: 40 mm from the sock's centre toward its corner end.
GRIP_OFF = 0.040
RING_GRIP = {  # place -> (x, y) of the TCP
    "left": (RING["left"][0] + GRIP_OFF, RING["left"][1]),  # 0.3375, +0.1025: toward the front-left corner
    "front": (RING["front"][0], RING["front"][1] - GRIP_OFF),  # 0.3825, -0.0575: toward the front-right corner
    "right": (RING["right"][0] - GRIP_OFF, RING["right"][1]),  # 0.2225, -0.1025: toward the back-right corner
    "back": (RING["back"][0], RING["back"][1] + GRIP_OFF),  # 0.1775, +0.0575: toward the back-left corner
}

# The spill-kit trays: fixed flat boards, one in front of each arm, two socks lying along x on each.
TRAY_T = 0.008
TRAY_TOP_Z = TOP + TRAY_T
# Starting places of the socks' centres, all along x (yaw 0). sock_1 and sock_2 are on the left tray, sock_3 and
# sock_4 on the right one. sock_1 and sock_3 start in line with the ring's left and right sides (moved straight
# across); sock_2 and sock_4 start with their grip point over the back and front grip points' x (lifted, turned a
# quarter turn about the grip, then moved straight across). The trays sit far enough out that neither arm reaches
# the other's socks: straddling the nearest one (its centre line 23.5 cm off the midline) needs the open gripper's
# axis 20.5 cm past the midline, where the straight-down reach falls short by 9 mm or more (tools/offline/yamkin).
SOCK_START = {
    "sock_1": (RING["left"][0], 0.235),  # (0.2975, +0.235)
    "sock_2": (RING_GRIP["back"][0] + GRIP_OFF, 0.305),  # (0.2175, +0.305): gripped 40 mm behind its centre
    "sock_3": (RING["right"][0], -0.235),  # (0.2625, -0.235)
    "sock_4": (RING_GRIP["front"][0] + GRIP_OFF, -0.305),  # (0.4225, -0.305): gripped 40 mm behind its centre
}
SOCKS = tuple(SOCK_START)
# 15 mm of tray beyond the socks on every side.
LEFT_TRAY = ((0.2575, 0.270), (0.310, 0.130))  # centre (x, y), size (x, y): x 0.1025..0.4125, y 0.205..0.335
RIGHT_TRAY = ((0.3425, -0.270), (0.390, 0.130))  # x 0.1475..0.5375, y -0.335..-0.205

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
# The fingers' zone covers the lowest 38 mm of each finger, measured on yam_arena's finger collision meshes: at the
# tips a finger is 6.4 mm thick (outside the gap) and 27 mm wide, 19-38 mm up it is 10 mm thick and 40 mm wide, so
# the zone uses 10 mm and 40 mm over its whole length. Higher up the fingers' roots, their carriages and the
# gripper's body are wider still (up to about 58 mm from the axis with the fingers 50 mm apart): not judged.
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap, lowest 38 mm (6.4 mm at the tips, 10 mm at 38 mm up)
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there (27 mm at the tips, 40 mm higher)
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Rule thresholds (scoring.py).
GAP_TIGHT = 0.010  # a closed joint: neighbouring socks at most this far apart (1.0)
GAP_LOOSE = 0.025  # a loose joint (0.75 allows one; the 0.5 chain may use them)
CHAIN_SPAN_DEG = 250.0  # 0.5: three or more socks in a chain wrapping this far round the spill (3 sides: ~277)
FLAT_DEG = 5.0  # 'flat': a long face down within this (any of the four: the section is square)
ON_TABLE_TOL = 0.003  # 'on the table': the sock's lowest point within this of the table top
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010  # ... or, if less open, its finger zone at least this far from every sock: it let go
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.05  # a sock moving this much further in one step than its velocity allows was teleported
#                    (a sock laid in the ring is only 8-10 cm from a jittered start on its tray)
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
SPILL_DARK = (0.10, 0.09, 0.08)
SOCK_WHITE = (0.96, 0.96, 0.94)
TRAY_YELLOW = (0.95, 0.80, 0.10)
