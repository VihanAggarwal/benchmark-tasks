"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A herbarium mounting sheet (the standard 11.5 x 16.5 in sheet, 292.1 x 419.1 mm, at 1:1.5: 194.7 x 279.4 mm) lies
portrait on the bench, its long side along x. Its foot (bottom edge) is the edge nearest the arms, x 0.205; its
head is x 0.4844. Read upright from the foot (a viewer at the foot looks along +x), the sheet's right edge is its
-y edge (y -0.09735, on the right arm's side) and its left edge its +y edge (y +0.09735). The accession barcode is
already stuck on, in the top right corner.

The rule (two herbarium mounting guides, see README): the collection label goes in the bottom right corner a few mm
from both edges (scored as 2-12 mm, 3-18 mm full size), reading upright from the foot; a determinavit slip goes
just above the label (ANBG) or above or to its left (Florida Museum), so directly above or directly left of it,
0-10 mm away and lined up within 10 mm, also reading upright; the plant is laid with its roots at the bottom
(toward the foot), on the sheet, clear of the label, the slip and the barcode.

Item frames (mechkit compounds): origin at the middle of the item's bottom face, local z up through its printed
top. The label and slip have their long side along local y; the specimen's long axis is local x, local +x toward
its root end.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the mounting sheet (fixed, printed on the table) ----------------------------------------------------------
SCALE = 1.5  # model = full size / 1.5
INCH = 0.0254
SHEET_LEN = round(16.5 * INCH / SCALE, 5)  # 0.2794 along x (portrait)
SHEET_W = 0.1947  # 11.5 in / 1.5 = 194.73 mm, to 0.1 mm, along y
SHEET_X0 = 0.205  # the foot (bottom edge), nearest the arms
SHEET_X1 = round(SHEET_X0 + SHEET_LEN, 5)  # 0.4844, the head
SHEET_Y0 = -SHEET_W / 2  # -0.09735: the sheet's right edge (read from the foot)
SHEET_Y1 = SHEET_W / 2  # its left edge
SHEET_T = 0.0005  # white mounting card, 0.5 mm proud of the table
OUTLINE_W = 0.002  # a 2 mm grey line drawn on the bench round the sheet so its edge shows in the camera
OUTLINE_T = 0.0005

# -- the accession barcode (fixed sticker on the sheet, top right corner) --------------------------------------
BARCODE_X0, BARCODE_X1 = 0.459, 0.479  # 20 mm along x: 5.4 mm below the head
BARCODE_Y0, BARCODE_Y1 = -0.090, -0.040  # 50 mm along y: 7.35 mm in from the right edge
BARCODE_T = 0.0003  # white paper sticker on the sheet
BAR_T = 0.0003  # printed black bars on the sticker
BAR_X0, BAR_X1 = 0.462, 0.476  # each bar runs 14 mm along x
BARS = [(-0.0860 + i * 0.00525, w) for i, w in enumerate((0.0010, 0.0020, 0.0010, 0.0015, 0.0010, 0.0020, 0.0010,
                                                          0.0015, 0.0010))]  # (centre y, width along y)

# -- the collection label (loose): a white pine tile with printed text lines ----------------------------------
LABEL_SIZE = (0.047, 0.067, 0.012)  # local (x, y, z): 67 mm long side along local y, 12 mm thick
LABEL_MASS = 0.020  # pine, about 0.53 g/cm3 (37.8 cm3)
LABEL_LINES = [  # (centre x, length along y, width along x) of each printed line; left-aligned at local y +0.025
    (0.0130, 0.050, 0.004),  # the herbarium's name, bold
    (0.0040, 0.044, 0.003),
    (-0.0040, 0.050, 0.003),
    (-0.0120, 0.030, 0.003),
]
LABEL_START = (0.250, -0.220, 0.0)  # origin x, y (m) and yaw (deg) on the bare table, in front of the right arm

# -- the determinavit slip (loose): a cream pine tile with "Det." and a name line ------------------------------
SLIP_SIZE = (0.020, 0.060, 0.012)  # 60 mm long side along local y
SLIP_MASS = 0.008  # pine (14.4 cm3)
SLIP_MARKS = [  # (centre x, centre y, width along x, length along y)
    (0.0040, 0.0200, 0.0035, 0.0120),  # "Det."
    (0.0040, -0.0060, 0.0025, 0.0320),  # the determiner's name for the taxon
    (-0.0045, 0.0050, 0.0025, 0.0400),  # the name and the date
]
SLIP_START = (0.350, -0.220, 0.0)

PRINT_T = 0.0005  # printed lines on the tiles

# -- the pressed specimen (loose): a basswood board with the plant printed on top -----------------------------
BOARD_SIZE = (0.200, 0.070, 0.010)  # local x along the plant (root end +x), y across, 10 mm thick
SPEC_MASS = 0.060  # basswood, about 0.43 g/cm3 (140 cm3)
PLANT_T = 0.001  # the printed plant, 1 mm proud of the board
PLANT = {  # name -> list of (centre x, centre y, size x, size y) in the board's frame, on its top face
    "roots": [(0.0840, 0.0, 0.024, 0.044)],  # dark brown root cluster at the +x end
    "stem": [(-0.0020, 0.0, 0.148, 0.005)],  # green stem, x -0.076 .. +0.072
    "leaves": [(0.0400, 0.0105, 0.024, 0.016), (0.0400, -0.0105, 0.024, 0.016),
               (0.0000, 0.0105, 0.022, 0.016), (0.0000, -0.0105, 0.022, 0.016),
               (-0.0360, 0.0095, 0.018, 0.014), (-0.0360, -0.0095, 0.018, 0.014)],
    "flowers": [(-0.0860, 0.0, 0.016, 0.016), (-0.0740, 0.0160, 0.012, 0.012), (-0.0740, -0.0160, 0.012, 0.012)],
}
SPEC_START = (0.300, 0.250, 90.0)  # lying along y, roots toward +y, in front of the left arm

ITEMS = ("label", "slip", "specimen")
BODY_SIZE = {"label": LABEL_SIZE, "slip": SLIP_SIZE, "specimen": BOARD_SIZE}  # the main body: its footprint
TOP_PRINT = {"label": PRINT_T, "slip": PRINT_T, "specimen": PLANT_T}
START = {"label": LABEL_START, "slip": SLIP_START, "specimen": SPEC_START}


def box_center_local(name: str) -> tuple[float, float, float]:
    """Centre of the item's bounding box (body plus its print) in its own frame."""
    return (0.0, 0.0, (BODY_SIZE[name][2] + TOP_PRINT[name]) / 2)


def box_half(name: str) -> tuple[float, float, float]:
    sx, sy, sz = BODY_SIZE[name]
    return (sx / 2, sy / 2, (sz + TOP_PRINT[name]) / 2)


# -- where the demo lays them ---------------------------------------------------------------------------------
EDGE_TARGET = 0.007  # the label 7 mm in from the right edge and from the foot (the middle of the 2-12 mm band)
LABEL_TARGET = (round(SHEET_X0 + EDGE_TARGET + LABEL_SIZE[0] / 2, 6),  # (0.2355, -0.05685), yaw 0
                round(SHEET_Y0 + EDGE_TARGET + LABEL_SIZE[1] / 2, 6))
SLIP_GAP_TARGET = 0.006  # the slip 6 mm above the label (0-10 mm), flush with its right edge
SLIP_TARGET = (round(LABEL_TARGET[0] + LABEL_SIZE[0] / 2 + SLIP_GAP_TARGET + SLIP_SIZE[0] / 2, 6),  # (0.275, ...)
               round(SHEET_Y0 + EDGE_TARGET + SLIP_SIZE[1] / 2, 6))  # -0.06035
SPEC_TARGET = (0.355, 0.045, 180.0)  # roots toward the foot: footprint x 0.255-0.455, y 0.010-0.080

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------
FLAT_DEG = 5.0  # 'flat': the item's printed face up, within 5 degrees of level
ON_TABLE = 0.003  # 'lying on the table': its lowest point within 3 mm of the table top (the sheet is 0.5 mm)
SQUARE_DEG = 5.0  # label and slip: long side along the sheet's width (y) within 5 degrees, and reading upright
#                   from the foot (local +x, toward the top text line, within 5 degrees of the head, +x)
EDGE_IN_MIN, EDGE_IN_MAX = 0.002, 0.012  # label: right and bottom edges 2-12 mm inside the sheet's edges
SLIP_GAP_MAX = 0.010  # slip: 0-10 mm from the label, directly above or directly left of it
SLIP_ALIGN = 0.010  # ... its matching edge lined up with the label's within 10 mm
SPEC_MARGIN = 0.002  # specimen: whole footprint at least 2 mm inside every sheet edge
ROOT_DEG = 15.0  # specimen: root end (local +x) within 15 degrees of -x (toward the foot)
OVERLAP_EPS = 1e-6  # m: footprints sharing only an edge do not overlap
EPS = 1e-9  # m: a reading exactly on a limit counts as inside it (no floating-point rounding decides a tier)
ANG_EPS = 1e-6  # degrees: the same for angle limits
STILL_WINDOW_S = 0.5  # 'still for 1 s': in each of the last two 0.5 s windows, no top corner's path over 1.5 mm
STILL_WINDOWS = 2
STILL_PATH = 0.0015
HANDS_OFF = 0.020  # 'let go': both grippers' finger zones at least 20 mm from the item
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an item landing on its starting mark having moved this much further in one control step than
#                    its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)
START_YAW_TOL_DEG = 10.0

WEIGHTS = {"label": 0.4, "slip": 0.2, "specimen": 0.4}

# -- the grippers (the let-go rule; rigstate reports the TCP, the point between the pad tips and the opening) -----
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) --------------------------------------------------------------------------------------------
SHEET_WHITE = (0.97, 0.97, 0.95)
OUTLINE_GREY = (0.45, 0.45, 0.47)
BARCODE_WHITE = (1.0, 1.0, 1.0)
INK_BLACK = (0.05, 0.05, 0.05)
LABEL_WHITE = (0.99, 0.99, 0.99)
TEXT_GREY = (0.20, 0.20, 0.22)
SLIP_CREAM = (0.93, 0.86, 0.62)
BOARD_BUFF = (0.80, 0.76, 0.66)
ROOT_BROWN = (0.36, 0.20, 0.07)
LEAF_GREEN = (0.13, 0.50, 0.14)
FLOWER_YELLOW = (0.98, 0.84, 0.08)


def start_yaw(name: str) -> float:
    return math.radians(START[name][2])
