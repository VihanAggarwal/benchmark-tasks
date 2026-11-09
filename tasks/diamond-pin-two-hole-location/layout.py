"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A fixture plate on the midline has two square sockets on a line along y, 100 mm apart: A on the robot's left, B on
its right. A pin stands in a socket on the table (the socket's floor). The bracket has two holes 124 mm apart, 24 mm
more than the sockets: the exaggerated hole-spacing error of a part at the edge of its tolerance. Over a round pin a
hole has 8 mm of clearance all round; the diamond pin's head is a 48 x 8 mm bar, so with its flats facing the other
socket it leaves 28 mm of clearance along the line between the sockets and 8 mm across it.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the fixture plate: five fixed boxes, 100 x 250 x 15 mm outside, two 55 x 55 mm sockets through it -------------
FIX_X, FIX_Y = 0.300, 0.000
FIX_LEN_X, FIX_LEN_Y = 0.100, 0.250
FIX_H = 0.015
FIX_TOP = FIX_H  # 15 mm above the table
SOCKET = 0.055  # square opening; the table is the socket's floor
SOCKET_HALF = SOCKET / 2
PIN_SPACING = 0.100
SOCKETS = {"A": (FIX_X, FIX_Y + PIN_SPACING / 2), "B": (FIX_X, FIX_Y - PIN_SPACING / 2)}  # A left, B right
FIX_X0, FIX_X1 = FIX_X - FIX_LEN_X / 2, FIX_X + FIX_LEN_X / 2  # 0.250, 0.350
FIX_Y0, FIX_Y1 = FIX_Y - FIX_LEN_Y / 2, FIX_Y + FIX_LEN_Y / 2  # -0.125, 0.125
SOCK_X0, SOCK_X1 = FIX_X - SOCKET_HALF, FIX_X + SOCKET_HALF  # 0.2725, 0.3275
FIXTURE_PARTS = {  # name -> (x0, x1, y0, y1): the plate's five boxes, each 15 mm tall on the table
    "fixture_near": (FIX_X0, SOCK_X0, FIX_Y0, FIX_Y1),
    "fixture_far": (SOCK_X1, FIX_X1, FIX_Y0, FIX_Y1),
    "fixture_middle": (SOCK_X0, SOCK_X1, SOCKETS["B"][1] + SOCKET_HALF, SOCKETS["A"][1] - SOCKET_HALF),
    "fixture_end_left": (SOCK_X0, SOCK_X1, SOCKETS["A"][1] + SOCKET_HALF, FIX_Y1),
    "fixture_end_right": (SOCK_X0, SOCK_X1, FIX_Y0, SOCKETS["B"][1] - SOCKET_HALF),
}

# -- the pins -------------------------------------------------------------------------------------------------------
PIN_R = 0.024  # 48 mm across: 3.5 mm of play either side in a 55 mm socket
PIN_H = 0.055
PIN_MASS = 0.10  # round pins: hollow (a solid steel pin this size would weigh 780 g)
ROUND = ("round_pin_1", "round_pin_2")
DIAMOND = "diamond_pin"
PIN_NAMES = (*ROUND, DIAMOND)
# The diamond pin: one loose rigid body (mechkit compound), base frame at the centre of the shank's bottom face,
# local +z up its axis, local +x along the head's long side. A round shank under a relieved head: a 48 x 8 mm bar
# whose two long faces are the flats and whose two 8 mm ends are the lands.
SHANK_H = 0.015  # exactly the plate's thickness: in a socket the head starts at the plate's top
HEAD_SIZE = (0.048, 0.008, PIN_H - SHANK_H)  # 48 (long side) x 8 (across the flats) x 40 mm
HEAD_HALF_LONG, HEAD_HALF_FLATS = HEAD_SIZE[0] / 2, HEAD_SIZE[1] / 2
SHANK_CENTER = (0.0, 0.0, SHANK_H / 2)
HEAD_CENTER = (0.0, 0.0, SHANK_H + HEAD_SIZE[2] / 2)
DIAMOND_MASS = 0.08

# -- the trays and the starting marks -------------------------------------------------------------------------------
TRAY_H = 0.003
TRAY_TOP = TRAY_H
RIGHT_TRAY_X, RIGHT_TRAY_Y = 0.220, -0.265
RIGHT_TRAY_SIZE = (0.090, 0.150, TRAY_H)
LEFT_TRAY_X, LEFT_TRAY_Y = 0.235, 0.250
LEFT_TRAY_SIZE = (0.090, 0.090, TRAY_H)
ROUND_START = {"round_pin_1": (0.220, -0.235), "round_pin_2": (0.220, -0.295)}  # 60 mm apart along y
DIAMOND_START = (LEFT_TRAY_X, LEFT_TRAY_Y)
DIAMOND_START_YAW = math.pi / 2  # head's long side along y: along the line between the sockets (turned wrong)

# -- the bracket: one loose rigid body (mechkit compound) ----------------------------------------------------------
# Base frame at the centre of its underside, local +z up through the boss, local +y along its length. A 10 mm plate
# 88 x 212 mm with two 64 x 64 mm through-holes centred 62 mm either side of its middle (124 mm apart), and a
# 30 x 40 x 25 mm boss on top at its centre (where it is gripped).
BR_LEN_X, BR_LEN_Y, BR_T = 0.088, 0.212, 0.010
HOLE = 0.064
HOLE_HALF = HOLE / 2
HOLE_Y = 0.062  # hole centres at local y +-62 mm
HOLE_SPACING = 2 * HOLE_Y  # 124 mm: 24 mm more than the pins
BOSS_SIZE = (0.030, 0.040, 0.025)
BR_MASS = 0.19  # acetal (1.41 g/cm3): 135 cm3
BR_START = (0.135, 0.110)  # lying flat on the table left of the midline, long side along y
_SIDE_W = (BR_LEN_X - HOLE) / 2  # 12 mm strips along the long sides
_END_W = BR_LEN_Y / 2 - (HOLE_Y + HOLE_HALF)  # 12 mm bars across the ends
_WEB_W = 2 * (HOLE_Y - HOLE_HALF)  # 60 mm web between the holes
BRACKET_SHAPES = (  # (centre, size) in the bracket's frame
    ((BR_LEN_X / 2 - _SIDE_W / 2, 0.0, BR_T / 2), (_SIDE_W, BR_LEN_Y, BR_T)),  # front strip
    ((-(BR_LEN_X / 2 - _SIDE_W / 2), 0.0, BR_T / 2), (_SIDE_W, BR_LEN_Y, BR_T)),  # back strip
    ((0.0, 0.0, BR_T / 2), (HOLE, _WEB_W, BR_T)),  # web between the holes
    ((0.0, BR_LEN_Y / 2 - _END_W / 2, BR_T / 2), (HOLE, _END_W, BR_T)),  # left end
    ((0.0, -(BR_LEN_Y / 2 - _END_W / 2), BR_T / 2), (HOLE, _END_W, BR_T)),  # right end
    ((0.0, 0.0, BR_T + BOSS_SIZE[2] / 2), BOSS_SIZE),  # grip boss
)
BR_CORNERS = tuple((sx * BR_LEN_X / 2, sy * BR_LEN_Y / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1))
BR_HOLES = ((0.0, HOLE_Y, 0.0), (0.0, -HOLE_Y, 0.0))  # hole centres on the underside, local

# -- what the geometry allows (the bracket's centre along y, measured from the fixture's centre) --------------------
# A round pin in B lets a hole over it only with the bracket's centre at y 4..20 mm (hole 62 mm from the bracket's
# centre, pin 50 mm from the fixture's, 8 mm of clearance); the diamond in A with its flats facing B allows -40..16;
# together 4..16 mm (12 mm wide). The diamond turned a quarter turn (48 mm along the line) or a second round pin in
# A allows only -20..-4: 8 mm short of 4, so the bracket cannot come down over both and rests on the pin heads.
# With the round pin in A and the diamond in B, the mirror image: -16..-4.
ROUND_CLEAR = HOLE_HALF - PIN_R  # 8 mm
SPACING_ERROR = HOLE_SPACING - PIN_SPACING  # 24 mm
WINDOW_LO = SPACING_ERROR / 2 - ROUND_CLEAR  # 4 mm
WINDOW_HI = -SPACING_ERROR / 2 + HOLE_HALF - HEAD_HALF_FLATS  # 16 mm
BLOCKED_HI = -SPACING_ERROR / 2 + ROUND_CLEAR  # -4 mm: the most a round-only (or turned-diamond) socket A allows
BR_AIM_Y = (WINDOW_LO + WINDOW_HI) / 2  # 10 mm: the demo aims at the middle of the window

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
FIT_TILT_DEG = 5.0  # a fitted pin is upright within 5 degrees
FIT_FLOOR_TOL = 0.003  # ... and the centre of its bottom face within 3 mm of the table (the socket floor)
TURN_TOL_DEG = 15.0  # the diamond's head within 15 degrees of square to the line between the sockets
SEAT_TILT_DEG = 2.0  # a seated bracket is level within 2 degrees
SEAT_TOL = 0.002  # ... with every corner of its underside within 2 mm of the plate's top
STILL_WINDOW_S = 0.5  # 'at rest': no tracked point moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
OFF_TABLE_DROP = 0.010  # 'off the table': lowest point more than 10 mm below the table top, or centre outside it
TABLE_X0, TABLE_X1 = 0.36 - 0.55, 0.36 + 0.55  # the station table's outline (1.10 x 1.30 m)
TABLE_Y0, TABLE_Y1 = -0.65, 0.65
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a pin or the bracket moving this far in one control step was teleported: a new episode
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
FIXTURE_GREY = (0.40, 0.42, 0.45)
PIN_STEEL = (0.78, 0.79, 0.81)
HEAD_BLACK = (0.08, 0.08, 0.09)
BRACKET_BLUE = (0.16, 0.36, 0.78)
TRAY_BLACK = (0.13, 0.13, 0.14)
