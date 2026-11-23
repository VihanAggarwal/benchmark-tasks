"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP.

Scene ("est" marks a figure not read at a source):
- a 9 x 6 x 4 cm block of dry ice in its paper wrap, labelled DRY ICE (0.33 kg: 216 cm3 at 1.55 g/cm3), on a plate
  on the midline;
- on the robot's right, the insulated meal-kit box it came in: a corrugated box (outside 17 x 25 x 8 cm, est) lined
  with a foil-faced insulation liner, its two long top flaps standing open, leaning 10 degrees outward; not airtight;
- on the robot's left, a clear airtight food tub with an attached gasketed lid (pet-food or bulk-food bin style,
  2 L, 12 x 20 x 9 cm, est), its lid hinged along the far edge and standing open; shut, the gasket seals it.
The box and the tub stand on a non-slip mat: fixed. The block is loose.
"""

from __future__ import annotations

import math

TOP = 0.7475
CONTROL_HZ = 20.0
EPS = 1e-9

# -- the block of dry ice (loose) -------------------------------------------------------------------------------------
BLOCK_SIZE = (0.090, 0.060, 0.040)  # x, y, z: the 6 cm side along y, where the fingers close
BLOCK_MASS = 0.33
BLOCK_FRICTION = (0.45, 0.40)  # paper wrap on plate, est
PLATE_XY = (0.300, 0.000)
PLATE_R = 0.080
PLATE_T = 0.015  # plate top 15 mm above the table
BLOCK_XY0 = PLATE_XY
BLOCK_Z0 = PLATE_T + BLOCK_SIZE[2] / 2  # block centre height at the start

# -- the insulated box (fixed) ----------------------------------------------------------------------------------------
BOX_C = (0.300, -0.215)  # centre of the box's outline
BOX_OUT = (0.170, 0.250, 0.080)  # x, y, height
BOX_WALL = 0.015  # corrugated wall and liner
BOX_FLOOR = 0.015
BOX_X0, BOX_X1 = BOX_C[0] - BOX_OUT[0] / 2, BOX_C[0] + BOX_OUT[0] / 2  # 0.215, 0.385
BOX_Y0, BOX_Y1 = BOX_C[1] - BOX_OUT[1] / 2, BOX_C[1] + BOX_OUT[1] / 2  # -0.340, -0.090
BOX_IN_X = (BOX_X0 + BOX_WALL, BOX_X1 - BOX_WALL)
BOX_IN_Y = (BOX_Y0 + BOX_WALL, BOX_Y1 - BOX_WALL)
RIM = BOX_OUT[2]

FLAP_LEN = 0.083  # each long flap reaches 83 mm in: the two meet 4 mm apart at the middle
FLAP_T = 0.004
FLAP_W = BOX_OUT[1]  # runs the box's full length along y
FLAP_MASS = 0.030
FLAP_OPEN_DEG = 100.0  # as drawn: 10 degrees past upright, leaning outward, on its stop
FLAP_BALANCE_DEG = 90.0  # past upright it falls shut under its own weight
FLAP_DAMP = 0.002  # N m s/rad, crease friction (est)
FLAP_HINGE_Z = RIM + FLAP_T / 2  # a closed flap rests on the rim
FLAP_SHUT_DEG = 20.0  # closed: within 20 degrees of flat
FLAP_NAMES = ("near_flap", "far_flap")  # hinged on the box's near (x0) and far (x1) long edges

# -- the airtight tub (fixed) with its attached lid, hinged along the far edge -----------------------------------------
TUB_C = (0.315, 0.215)
TUB_OUT = (0.120, 0.200, 0.090)  # x, y, height
TUB_WALL = 0.004
TUB_FLOOR = 0.004
TUB_IN_X = (TUB_C[0] - TUB_OUT[0] / 2 + TUB_WALL, TUB_C[0] + TUB_OUT[0] / 2 - TUB_WALL)
TUB_IN_Y = (TUB_C[1] - TUB_OUT[1] / 2 + TUB_WALL, TUB_C[1] + TUB_OUT[1] / 2 - TUB_WALL)
TUB_RIM = TUB_OUT[2]
TUB_X0, TUB_X1 = TUB_C[0] - TUB_OUT[0] / 2, TUB_C[0] + TUB_OUT[0] / 2
TUB_Y0, TUB_Y1 = TUB_C[1] - TUB_OUT[1] / 2, TUB_C[1] + TUB_OUT[1] / 2
LID_LEN = TUB_OUT[0]  # reaches across the tub from its far-edge hinge
LID_T = 0.010
LID_MASS = 0.080
LID_OPEN_DEG = 100.0
LID_HINGE_Z = TUB_RIM + LID_T / 2
LID_SEALED_DEG = 5.0  # shut on its gasket: within 5 degrees of flat
LID_NAME = "tub_lid"
LATCH_SIZE = (0.030, 0.004, 0.018)  # the lid's front lip (visual)
# the press-to-seal button on the tub's front wall: pressed with the lid shut, it locks the lid down on its gasket
BUTTON = "seal_button"
BUTTON_C = (TUB_X0 - 0.012, TUB_C[1])  # a tab standing out 12 mm in front of the near wall, at the rim
BUTTON_SIZE = (0.020, 0.030, 0.012)
BUTTON_TOP = TUB_RIM + 0.004
BUTTON_TRAVEL = 0.004
BUTTON_SEAL_AT = 0.0025  # pressed 2.5 mm: sealed (with the lid shut)
BUTTON_K = 1700.0  # N/m: 4.2 N at the seal point (est)
BUTTON_MASS = 0.010

# -- scoring thresholds -----------------------------------------------------------------------------------------------
IN_ALLOW = 0.002  # into a wall or the floor: contact
LIFT_MOVE = 0.050  # the block moved 5 cm from its start counts as handled
RELEASE_CLEAR = 0.040
OPEN_THRESHOLD = 0.85
STILL_WINDOW_S = 1.0
WIN_STEPS = int(round(STILL_WINDOW_S * CONTROL_HZ))
STILL_MOVE = 0.003
OFF_TABLE_DROP = 0.010
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
KRAFT = (0.72, 0.56, 0.38)
FOIL = (0.80, 0.81, 0.83)
ICE_PAPER = (0.93, 0.93, 0.90)
LABEL_BLACK = (0.05, 0.05, 0.05)
PLATE_WHITE = (0.95, 0.95, 0.95)
TUB_CLEAR = (0.82, 0.88, 0.92)
LATCH_BLUE = (0.20, 0.42, 0.75)
MAT_GREY = (0.25, 0.26, 0.28)


# every hinged panel the closed pads push shut: name -> (hinge x, inward sign along x, hinge z, length, y range, open)
PANELS = {
    FLAP_NAMES[0]: (BOX_X0, +1.0, FLAP_HINGE_Z, FLAP_LEN, (BOX_Y0, BOX_Y1), FLAP_OPEN_DEG),
    FLAP_NAMES[1]: (BOX_X1, -1.0, FLAP_HINGE_Z, FLAP_LEN, (BOX_Y0, BOX_Y1), FLAP_OPEN_DEG),
    LID_NAME: (TUB_X1, -1.0, LID_HINGE_Z, LID_LEN, (TUB_Y0, TUB_Y1), LID_OPEN_DEG),
}
BALANCE_DEG = 90.0


def flap_dir(name: str, theta_deg: float) -> tuple[float, float]:
    """A panel's direction (x, z) from its hinge at opening ``theta_deg`` (0 flat, as drawn when open)."""
    t = math.radians(theta_deg)
    return PANELS[name][1] * math.cos(t), math.sin(t)


def flap_hinge_x(name: str) -> float:
    return PANELS[name][0]


# -- the DRY ICE label: block letters as strokes (cell coordinates 0..1 across, 0..1 up) ------------------------------
_D = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.7, 1.0)), ((0.0, 0.0), (0.7, 0.0)), ((1.0, 0.2), (1.0, 0.8)),
      ((0.7, 1.0), (1.0, 0.8)), ((0.7, 0.0), (1.0, 0.2))]
_R = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0)), ((1.0, 0.5), (1.0, 1.0)), ((0.0, 0.5), (1.0, 0.5)),
      ((0.3, 0.5), (1.0, 0.0))]
_Y = [((0.0, 1.0), (0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5)), ((0.5, 0.0), (0.5, 0.5))]
_I = [((0.5, 0.0), (0.5, 1.0)), ((0.15, 1.0), (0.85, 1.0)), ((0.15, 0.0), (0.85, 0.0))]
_C = [((0.0, 0.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 0.0))]
_E = _C + [((0.0, 0.5), (0.8, 0.5))]
LABEL_TEXT = [_D, _R, _Y, None, _I, _C, _E]  # None: a space
LETTER_W, LETTER_H, LETTER_GAP, STROKE = 0.0062, 0.011, 0.0018, 0.0013
LABEL_PATCH = (0.020, 0.058)  # white label patch on the block's top (x, y), the letters on it


def label_strokes():
    """(centre (x, y), length, yaw) of each stroke on the block's top face, in the block's frame. The text reads
    left to right from the robot's side: along -y, letters standing up toward +x."""
    n = len(LABEL_TEXT)
    total = n * LETTER_W + (n - 1) * LETTER_GAP
    out = []
    for i, letter in enumerate(LABEL_TEXT):
        if letter is None:
            continue
        y_left = total / 2 - i * (LETTER_W + LETTER_GAP)  # the letter's left edge (as read) in block y
        for (u0, v0), (u1, v1) in letter:
            a = (-LETTER_H / 2 + v0 * LETTER_H, y_left - u0 * LETTER_W)
            b = (-LETTER_H / 2 + v1 * LETTER_H, y_left - u1 * LETTER_W)
            cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            length = math.hypot(b[0] - a[0], b[1] - a[1]) + STROKE
            out.append(((cx, cy), length, math.atan2(b[1] - a[1], b[0] - a[0])))
    return out
