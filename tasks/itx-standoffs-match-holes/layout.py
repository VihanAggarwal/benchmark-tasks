"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A steel case tray (the motherboard tray of a PC case, lying flat) is fixed on the midline. It has seven threaded
holes, drawn as dark discs: four are the mini-ITX board's (H1-H4) and three are only for larger boards (A1-A3).
Brass standoffs stand loose on five of them, as a case ships for a larger board: on H1, H2, A1, A2 and A3. A1 and A2
lie inside the outline the mini-ITX board covers when mounted, where the board has no hole; A3 lies outside it, so a
standoff may stay on it (any other standoff over the tray must stand on H1-H4: one loose in the case is a stray). The
board's four holes are not on a rectangle (its front-right hole is 30 mm further back than its front-left one), so
it fits its holes only one way round, with its I/O block at the rear (+x, the far edge).
"""

TOP = 0.7475  # station table top (world z)

# -- the case tray and its holes -----------------------------------------------------------------------------------
CASE_X, CASE_Y = 0.300, 0.000
CASE_SIZE = (0.180, 0.200, 0.003)  # 180 mm along x, 200 mm along y, 3 mm steel
TRAY_TOP = CASE_SIZE[2]  # 3 mm above the table
DISC_R = 0.014  # each threaded hole is drawn as a dark disc 28 mm across
DISC_H = 0.001  # ... standing 1 mm proud of the tray's top
DISC_TOP = TRAY_TOP + DISC_H
ITX_HOLES = {  # the mini-ITX board's four holes (world x, y)
    "H1": (0.335, 0.035),  # rear left
    "H2": (0.335, -0.035),  # rear right
    "H3": (0.265, 0.035),  # front left
    "H4": (0.295, -0.035),  # front right: 30 mm behind the front-left hole
}
ATX_HOLES = {  # holes only a larger (ATX) board uses
    "A1": (0.300, 0.035),  # inside the mini-ITX outline, where the board has no hole
    "A2": (0.265, -0.035),  # inside it too: where a rectangular pattern would put the fourth hole
    "A3": (0.335, -0.075),  # outside it (31 mm past its right edge)
}
HOLES = {**ITX_HOLES, **ATX_HOLES}
ITX = tuple(ITX_HOLES)
SPARE_HOLES = ("A3",)  # a larger board's hole clear of the mini-ITX outline: a standoff may stay standing on it
TRAY_OUTLINE = (CASE_X - CASE_SIZE[0] / 2, CASE_X + CASE_SIZE[0] / 2,
                CASE_Y - CASE_SIZE[1] / 2, CASE_Y + CASE_SIZE[1] / 2)  # x0 x1 y0 y1: x 0.21-0.39, y -0.10-0.10

# -- the standoffs: loose brass cylinders ---------------------------------------------------------------------------
STANDOFF_R = 0.009  # 18 mm across
STANDOFF_H = 0.020  # 20 mm tall
STANDOFF_MASS = 0.020  # hollow brass (a solid one this size would weigh 43 g)
STANDOFF_TOP = DISC_TOP + STANDOFF_H  # 24 mm above the table, standing on a disc
STANDOFFS = ("standoff_1", "standoff_2", "standoff_3", "standoff_4", "standoff_5")
STANDOFF_START = {"standoff_1": "H1", "standoff_2": "H2", "standoff_3": "A1", "standoff_4": "A2", "standoff_5": "A3"}

# -- the mini-ITX board: one loose rigid body (mechkit compound) ---------------------------------------------------
# Base frame at the centre of the board's underside, local +z up through its components, local +x toward its I/O
# edge. A green 88 x 88 x 3 mm board (about half the size of a real mini-ITX board) with a low CPU cooler fitted at its
# centre (the grip), the I/O connector block along its +x edge, and light gold pads around its four holes.
BOARD = "board"
BOARD_LEN = 0.088
BOARD_HALF = BOARD_LEN / 2
BOARD_T = 0.003
BOARD_HOLES = {  # hole centres on the underside, local (x, y); hole k sits over tray hole H<k> when mounted
    "H1": (0.035, 0.035),
    "H2": (0.035, -0.035),
    "H3": (-0.035, 0.035),
    "H4": (-0.005, -0.035),
}
COOLER_SIZE = (0.040, 0.040, 0.030)
COOLER_CENTER = (0.0, 0.0, BOARD_T + COOLER_SIZE[2] / 2)
IO_SIZE = (0.012, 0.048, 0.015)  # along the rear edge, 4 mm clear of the rear holes' pads
IO_CENTER = (BOARD_HALF - IO_SIZE[0] / 2, 0.0, BOARD_T + IO_SIZE[2] / 2)
PAD_R = 0.007  # the gold pad around each hole, 14 mm across
PAD_H = 0.001
BOARD_MASS = 0.15
MOUNT_X, MOUNT_Y = CASE_X, CASE_Y  # the board's centre when mounted: its holes over H1-H4
MOUNT_Z = STANDOFF_TOP  # its underside on the standoffs' tops
OUTLINE = (MOUNT_X - BOARD_HALF, MOUNT_X + BOARD_HALF, MOUNT_Y - BOARD_HALF, MOUNT_Y + BOARD_HALF)  # x0 x1 y0 y1

# -- the anti-static mat the board starts on, and the parts dish ----------------------------------------------------
MAT_X, MAT_Y = 0.180, -0.200
MAT_SIZE = (0.110, 0.110, 0.005)
MAT_TOP = MAT_SIZE[2]
BOARD_START = (MAT_X, MAT_Y)  # its centre; I/O block toward +x, as it goes in
DISH_X, DISH_Y = 0.190, 0.210
DISH_SIZE = (0.080, 0.080, 0.003)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
UPRIGHT_DEG = 10.0  # a standoff stands upright: its axis within 10 degrees of vertical (either end down)
FLOOR_TOL = 0.003  # ... on the tray: the centre of its bottom face within 3 mm of the tray's top
ON_HOLE_TOL = 0.005  # ... on a hole: that centre within 5 mm of the hole's centre (its base wholly on the disc)
STRAY_MARGIN = STANDOFF_R  # 'in the case': centre over the tray or less than 9 mm (a radius) outside its edge
BOARD_FLAT_DEG = 3.0  # a mounted board is level within 3 degrees
BOARD_HOLE_TOL = 0.007  # ... each of its holes within 7 mm (its pad's radius) of its standoff's axis
BOARD_SEAT_TOL = 0.002  # ... and its underside there within 2 mm of that standoff's top
STILL_WINDOW_S = 0.5  # 'at rest': no tracked point moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
OFF_TABLE_DROP = 0.010  # 'off the table': lowest point more than 10 mm below the table top, or centre outside it
TABLE_X0, TABLE_X1 = 0.36 - 0.55, 0.36 + 0.55  # the station table's outline (1.10 x 1.30 m)
TABLE_Y0, TABLE_Y1 = -0.65, 0.65
PER_HOLE = 0.2  # score for each mini-ITX hole with a standoff on it
MOUNT_CREDIT = 0.2  # ... and for the board mounted on all four
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a part moving this far in one control step was teleported: a new episode
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
TRAY_GREY = (0.74, 0.75, 0.77)
HOLE_DARK = (0.22, 0.22, 0.24)
BRASS = (0.80, 0.64, 0.26)
PCB_GREEN = (0.10, 0.40, 0.20)
COOLER_BLACK = (0.08, 0.08, 0.09)
IO_SILVER = (0.72, 0.74, 0.78)
PAD_GOLD = (0.93, 0.80, 0.45)
MAT_PINK = (0.86, 0.56, 0.66)
DISH_BLACK = (0.13, 0.13, 0.14)
