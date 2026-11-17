"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475.

The scene is the top shelf of a small open bookcase, standing on the station table in front of the left arm: a shelf
board between two side panels, with a back panel, and no board above it. Six upright paperbacks (A to F) fill it
from panel to panel, spines toward the robot, except for a 40 mm gap between C and D. The new book, a taller and
thicker paperback, stands on the table (a book truck's top) to the left of the case, in front of it.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the bookcase (fixed): shelf board, two side panels, back panel ------------------------------------------------
Y0 = 0.305  # the gap's centre line: the left arm's base line
SHELF_T = 0.012  # shelf board thickness (12 mm melamine chipboard, est)
SHELF_TOP = TOP + SHELF_T  # the books stand on this
ROW_HALF = 0.095  # the row's six books are 190 mm long in all (6 x 25 + the 40 mm gap): half of it
PANEL_GAP = 0.0005  # the end books stand this far from the panels (0.5 mm)
PANEL_T = 0.018  # side panels: 18 mm
PANEL_H = 0.240  # side and back panels stand this high above the table top (225 mm over the shelf)
BACK_T = 0.012
XS = 0.265  # the row's spine line: the neighbours' spines (the faces toward the robot)
BOOK_D = 0.105  # every book is 105 mm along x (front to back)
XB = XS + BOOK_D  # 0.370: the books' backs
BACK_GAP = 0.0005
SHELF_X0 = XS - 0.030  # the shelf board's front edge, 30 mm ahead of the spines
SHELF_X1 = XB + BACK_GAP  # 0.3705: the back panel's front face
CASE_X1 = SHELF_X1 + BACK_T  # 0.3825
PANEL_Y_IN = ROW_HALF + PANEL_GAP  # the side panels' inner faces: y0 +- 0.0955

# -- the neighbours A to F (loose): mass-market (A6 pocket) paperbacks -----------------------------------------------
NB_T = 0.025  # thickness (along y)
NB_H = 0.148  # height (A6 is 105 x 148 mm)
NB_MASS = 0.16  # est
GAP = 0.040  # between C and D
NEIGHBOURS = ("book_a", "book_b", "book_c", "book_d", "book_e", "book_f")
NB_DY = {  # centre offsets from Y0
    "book_a": -(ROW_HALF - NB_T / 2),
    "book_b": -(ROW_HALF - NB_T * 1.5),
    "book_c": -(GAP / 2 + NB_T / 2),
    "book_d": GAP / 2 + NB_T / 2,
    "book_e": ROW_HALF - NB_T * 1.5,
    "book_f": ROW_HALF - NB_T / 2,
}
NB_X = XS + BOOK_D / 2  # 0.3175
NB_START = {n: (NB_X, Y0 + dy) for n, dy in NB_DY.items()}
NB_TOP = SHELF_TOP + NB_H  # the neighbours' tops

# -- the new book (loose): a tall trade paperback --------------------------------------------------------------------
NEW = "book_new"
NEW_T = 0.030
NEW_H = 0.185
NEW_MASS = 0.26  # est
NEW_START = (0.170, Y0 + 0.130)  # on the table, left of the case, in front of its end panel (its spine toward the robot)
NEW_END_X = XS + BOOK_D / 2  # its centre when seated: spine on the row's line
BOOKS = (*NEIGHBOURS, NEW)
SIZE = {n: (BOOK_D, NB_T, NB_H) for n in NEIGHBOURS}
SIZE[NEW] = (BOOK_D, NEW_T, NEW_H)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_Y_TOL = 0.003  # seated: centre within this of the gap's centre line (5 mm clear each side when centred)
SEAT_TILT_DEG = 3.0  # and tilted less than this
SEAT_YAW_DEG = 3.0  # spine out: turned less than this from its start heading
SEAT_CLEAR_MIN = 0.002  # and every bottom corner this far or more from C's face and from D's face
SEAT_SPINE_TOL = 0.005  # spine within this of the row's spine line
SEAT_Z_TOL = 0.003  # underside (at its centre) within this of the shelf top
GAP_TILT_DEG = 10.0  # 'in the gap': the underside's four corners between the neighbours' faces, tilt under this,
GAP_YAW_DEG = 15.0  # turned under this,
GAP_SPINE_TOL = 0.040  # spine within this of the spine line
ROW_TILT_END_DEG = 5.0  # row upright: each neighbour tilted less than this at the end,
ROW_TILT_MAX_DEG = 20.0  # and never more than this,
ROW_SHIFT = 0.006  # and its centre (horizontally) within this of its starting place
ROW_LEAN_END_DEG = 20.0  # 0.25 tier (in the gap, the row disturbed): a neighbour leaning up to this at the end
ROW_LEAN_SHIFT = 0.020
FALL_DEG = 45.0  # a neighbour tilted more than this at any moment, or the new book tilted more than this at any
#                  moment when NOT held: fell: 0. The new book held in a hand may be tilted more than this on the way
HOLD_GAP = 0.010  # the new book is held when a gripper's finger zone is within this of it ...
HOLD_WIDTH = 0.075  # ... with its pads less than this apart (the 30 mm book fits between them, an open hand, 94 mm, does not)
OFF_TABLE = 0.010  # the new book's lowest point more than this below the table top (it fell off the table, or through
#                    it) at any moment: 0. The shelf board is 12 mm over the table top: a book on it is 12 mm higher
OFF_SHELF_DROP = 0.010  # a neighbour's underside this far below the shelf top, or its centre off the board: 0
HANDS_S = 1.0  # last second: both grippers' finger zones clear, every book still
HANDS_CLEAR = 0.030  # finger zones at least this from every book
STILL_MOVE = 0.005  # over the last second no book's centre moves more than this (visible: 5 mm)
STILL_TURN_DEG = 5.0  # and none turns more than this (visible: 5 degrees)
CONTROL_HZ = 20.0
RESET_JUMP = 0.08  # a book that lands on its starting mark having moved this much more in one step than its last
#                    velocity and gravity could carry it was teleported: a new episode began

# Colours (sRGB).
CASE_PINE = (0.80, 0.66, 0.46)
SHELF_WHITE = (0.88, 0.86, 0.80)
NEW_RED = (0.72, 0.20, 0.18)
BOOK_COLOURS = {
    "book_a": (0.25, 0.38, 0.60), "book_b": (0.80, 0.72, 0.30), "book_c": (0.30, 0.52, 0.38),
    "book_d": (0.55, 0.30, 0.55), "book_e": (0.82, 0.50, 0.25), "book_f": (0.35, 0.35, 0.40),
}
