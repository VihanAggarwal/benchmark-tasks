"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A steel door stands on edge, hinge edge up, along y in front of both arms: its top end (the head) at +y, its bottom
end at -y. Lengths along the door are to scale, 1 in = 5 mm: an 80 in door is 400 mm, a 4-1/2 in hinge 22.5 mm, a
3/4 in undercut 3.75 mm. The door is 36 mm thick and 80 mm tall, not to scale, so the edge is a stable shelf.

The hollow metal door trade's hinge locations, '5, 10 and equal' (NAAMM/HMMA): the top hinge 5 in from the frame's
head rabbet to the top of the hinge, the bottom hinge 10 in from the finished floor to the bottom of the hinge, a
third hinge centred between them. The red HEAD line stands for the head rabbet: its door-side edge is level with
the door's top end (the 1/8 in head clearance is ignored). The blue FLOOR line stands for the finished floor: its
door-side edge is 3/4 in (3.75 mm) past the door's bottom end, the undercut. So, along y:

    top leaf's centre    = HEAD_Y  - 5 in  - half a leaf = 0.200 - 0.025 - 0.01125 = +0.16375
    bottom leaf's centre = FLOOR_Y + 10 in + half a leaf = -0.20375 + 0.050 + 0.01125 = -0.14250
    middle leaf's centre = halfway between the top and bottom leaves as hung (nominally +0.010625)
"""

import math

TOP = 0.7475  # station table top
IN = 0.005  # one inch along the door, at scale

# -- the door: a fixed box standing on its long edge, hinge edge up ------------------------------------------
DOOR_X = 0.300  # the hinge edge's centreline (and the door's centre), along y
DOOR_THICK = 0.036  # along x (not to scale)
DOOR_LEN = 80 * IN  # 0.400 along y: an 80 in door
DOOR_H = 0.080  # along z (the door's width; not to scale)
DOOR_SIZE = (DOOR_THICK, DOOR_LEN, DOOR_H)
DOOR_TOP_END = DOOR_LEN / 2  # +0.200: the door's top end (the head)
DOOR_BOTTOM_END = -DOOR_LEN / 2  # -0.200: its bottom end
EDGE_TOP = TOP + DOOR_H  # the hinge edge's surface (world z)

# -- the two reference lines: fixed painted strips on the table, across the door's two ends -----------------
LINE_W = 0.006  # across (along y)
LINE_LEN = 0.080  # along x, centred on the door: x 0.260 .. 0.340, 22 mm past each face
LINE_T = 0.0005
HEAD_Y = DOOR_TOP_END  # the HEAD line's door-side edge (the head rabbet); the strip runs y 0.200 .. 0.206
UNDERCUT = 0.75 * IN  # 3.75 mm
FLOOR_Y = DOOR_BOTTOM_END - UNDERCUT  # -0.20375: the FLOOR line's door-side edge; the strip runs to -0.20975

# -- the hinge leaves: loose brass blocks, the hinge folded (leaf plus knuckle) -------------------------------
LEAF_LEN = 4.5 * IN  # 0.0225 along the door edge (a 4-1/2 in hinge): the block's y side at yaw 0
LEAF_W = 0.016  # across the edge (x at yaw 0): clearly narrower than long, so 'lengthwise' is plain to see
LEAF_H = 0.010  # clearly thinner than wide, so 'flat' is plain to see
LEAF_SIZE = (LEAF_W, LEAF_LEN, LEAF_H)
LEAF_MASS = 0.031  # solid brass, 8.5 g/cm3 x 3.6 cm3 (30.6 g)
LEAF_NAMES = ("hinge_leaf_a", "hinge_leaf_b", "hinge_leaf_c")

# -- the parts tray: a fixed plate in front of the door; the leaves start on it ------------------------------
TRAY_X = 0.200
TRAY_SIZE = (0.060, 0.320, 0.003)  # x 0.170 .. 0.230, y -0.160 .. +0.160
TRAY_TOP = TOP + TRAY_SIZE[2]
LEAF_START = {  # name -> (x, y, yaw in degrees): flat on the tray, long side along y
    "hinge_leaf_a": (TRAY_X, 0.130, 0.0),
    "hinge_leaf_b": (TRAY_X, 0.050, 0.0),
    "hinge_leaf_c": (TRAY_X, -0.130, 0.0),
}

# -- the rule ------------------------------------------------------------------------------------------------
TOP_FROM_HEAD = 5 * IN  # head rabbet to the top of the top hinge
BOTTOM_FROM_FLOOR = 10 * IN  # finished floor to the bottom of the bottom hinge
TOP_Y = round(HEAD_Y - TOP_FROM_HEAD - LEAF_LEN / 2, 9)  # +0.16375
BOTTOM_Y = round(FLOOR_Y + BOTTOM_FROM_FLOOR + LEAF_LEN / 2, 9)  # -0.14250
MID_NOMINAL = round((TOP_Y + BOTTOM_Y) / 2, 9)  # +0.010625
WINDOW = 0.005  # each location: the leaf's centre within this along the edge (+-1 in at full size)

# 'On the edge': flat, lengthwise, centred across the edge, resting on it, wholly within the door's length.
SEAT_TOL = 0.003  # its lowest point within this of the edge top
FLAT_DEG = 5.0  # its 22.5 x 16 mm faces within this of level (either face up)
ALONG_DEG = 5.0  # its 22.5 mm side within this of the door's length (either way round)
ACROSS_TOL = 0.008  # its centre within this of the edge's centreline: all of it on the 36 mm edge, even turned
#                     5 deg and tilted 5 deg (its half-extent across is then at most 9.4 mm: 8 + 9.4 < 18)
ON_DOOR = DOOR_TOP_END - LEAF_LEN / 2  # |centre y| at most this: no part of it past either end (0.18875)

# 'Settled': centre moved at most STILL_TOL and long axis turned at most STILL_YAW_DEG over the last 0.5 s, judged
# from the path (a gripped part's reported velocity chatters). 'Still for 1 s': settled in each of the last two
# 0.5 s windows. 'Released': not held at any control step of the last 0.5 s.
STILL_WINDOW_S = 0.5
STILL_TOL = 0.002
STILL_YAW_DEG = 1.0

# 'Held' (rigstate). rigstate reports each gripper's pad-tip point (midway between its finger tips, 13 mm past the
# TCP) and how far it is open, not its yaw, so a finger may be on any side of that point. With the fingers
# FULL_GAP x opening apart, a finger (FINGER_THICK outside the gap, 2 x FINGER_HALF_WIDTH wide) can touch a leaf only
# if the leaf's centre is within release_reach(opening) of that point horizontally (the leaf's half-diagonal plus
# RELEASE_MARGIN included): 79 mm fully open, 56.5 mm at the demo's 44 mm opening. A gripper whose pad-tip point is
# lower than RELEASE_ABOVE over the leaf's top while within that reach may be holding, pressing or propping it.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010
LEAF_HALF_DIAG = math.hypot(LEAF_LEN / 2, LEAF_W / 2)

# -- scores --------------------------------------------------------------------------------------------------
S_FULL, S_ENDS, S_TWO, S_ONE = 1.0, 0.75, 0.5, 0.25

CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a leaf landing on its starting mark having moved this much further in one control step than its
#                    last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.035  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) -----------------------------------------------------------------------------------------
STEEL_GREY = (0.55, 0.57, 0.60)
HEAD_RED = (0.85, 0.08, 0.06)
FLOOR_BLUE = (0.08, 0.30, 0.90)
TRAY_GREY = (0.22, 0.22, 0.24)
BRASS = (0.80, 0.63, 0.25)


def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the leaf's
    half-diagonal and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + LEAF_HALF_DIAG + RELEASE_MARGIN


def start_yaw(name: str) -> float:
    return math.radians(LEAF_START[name][2])


assert abs(TOP_Y - 0.16375) < 1e-9 and abs(BOTTOM_Y + 0.1425) < 1e-9 and abs(MID_NOMINAL - 0.010625) < 1e-9
assert DOOR_THICK / 2 - ACROSS_TOL >= LEAF_W / 2 - 1e-12  # a leaf inside the across window is wholly on the edge
assert LEAF_LEN / LEAF_W >= 1.4 and LEAF_W / LEAF_H >= 1.4  # lengthwise and flat both plain to see
