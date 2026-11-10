"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A 1:5 model of one steam-table section: four full-size (GN 1/1, 530 x 325 mm) pans in a row across the table in front
of the arms, long sides along x, hanging in a stainless deck that is flush with their rims. The server's side is the
arms' side (-x). Pans 1 and 3 each hold a serving utensil, standing in the food near the front wall with its handle
rising out of the pan and running out over the front rim toward the server. Four loose pan covers (lids) lie
crosswise on the counters beside the deck, two on each side: one notched and one solid on each counter.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the pans (fixed) ------------------------------------------------------------------------------------------------
PAN_X = 0.330  # every pan's centre x
PANS = {  # name: (centre y, has a utensil)
    "pan1": (0.1275, True),  # chilli, a ladle (left of the midline, outermost)
    "pan2": (0.0425, False),  # rice
    "pan3": (-0.0425, True),  # sweetcorn, a slotted spoon
    "pan4": (-0.1275, False),  # green beans (right, outermost)
}
PAN_NAMES = tuple(PANS)
PAN_L, PAN_W = 0.106, 0.065  # outer length (along x) and width (along y)
PAN_H = 0.030  # rim height above the table (the deck's top is flush with it)
WALL = 0.003  # wall thickness
RIM_TOP_W = TOP + PAN_H
FOOD_TOP = 0.022  # food fills each pan to 8 mm below the rim
FOOD_COLOURS = {
    "pan1": (0.42, 0.14, 0.08),  # chilli, dark red-brown
    "pan2": (0.95, 0.94, 0.89),  # rice, white
    "pan3": (0.96, 0.80, 0.20),  # sweetcorn, yellow
    "pan4": (0.24, 0.52, 0.18),  # green beans, green
}

# -- the steam-table deck (fixed): strips flush with the rims, round and between the pans -------------------------------
DECK_X0, DECK_X1 = PAN_X - PAN_L / 2 - 0.020, PAN_X + PAN_L / 2 + 0.020  # 0.257 .. 0.403
DECK_Y = 0.180  # the deck spans y -0.180 .. +0.180
DECK_STRIPS = (  # (name, x0, x1, y0, y1)
    ("deck_front", DECK_X0, PAN_X - PAN_L / 2, -DECK_Y, DECK_Y),
    ("deck_back", PAN_X + PAN_L / 2, DECK_X1, -DECK_Y, DECK_Y),
    ("deck_left", PAN_X - PAN_L / 2, PAN_X + PAN_L / 2, 0.160, DECK_Y),
    ("deck_1_2", PAN_X - PAN_L / 2, PAN_X + PAN_L / 2, 0.075, 0.095),
    ("deck_2_3", PAN_X - PAN_L / 2, PAN_X + PAN_L / 2, -0.010, 0.010),
    ("deck_3_4", PAN_X - PAN_L / 2, PAN_X + PAN_L / 2, -0.095, -0.075),
    ("deck_right", PAN_X - PAN_L / 2, PAN_X + PAN_L / 2, -DECK_Y, -0.160),
)

# -- the two serving utensils (fixed): a stepped stand-in for a ladle leaning on the pan's front ------------------------
# A bowl on the food near the front wall, a straight stem rising from it 26 mm past the rim, and the handle running from
# the stem's top out over the front rim toward the server (-x). Every part above the rim lies over the notch of a
# centred, correctly turned notched lid: across the pan the notch leaves 8 mm either side of the 10 mm stem and handle;
# along it, 12 mm between the stem's back face and the notch's end.
UTENSIL_W = 0.010  # stem and handle width (y)
STEM_X0, STEM_X1 = 0.284, 0.290  # 4 mm behind the front wall's inner face (0.280)
STEM_Z0, STEM_Z1 = FOOD_TOP, 0.056
BOWL_X0, BOWL_X1, BOWL_W = STEM_X1, STEM_X1 + 0.024, 0.024
BOWL_Z0, BOWL_Z1 = FOOD_TOP, FOOD_TOP + 0.004
HANDLE_X0, HANDLE_X1 = 0.232, STEM_X1  # 45 mm out past the pan's front (0.277)
HANDLE_Z0, HANDLE_Z1 = STEM_Z1, 0.064
UTENSILS = {  # pan: (name prefix, handle colour)
    "pan1": ("ladle", (0.06, 0.06, 0.07)),  # black handle
    "pan3": ("spoon", (0.78, 0.10, 0.10)),  # red handle
}
UTENSIL_PARTS = ("bowl", "stem", "handle")
UTENSIL_MOVE_TOL = 0.005  # a utensil part more than this from where it was set: the trial scores 0

# -- the four lids: each one loose rigid body (mechkit compound) --------------------------------------------------------
# Base frame at the middle of the plate's outline, on its underside; local +x along the long side; local +z up. A
# notched lid's notch is cut into its local -x end, centred across it.
LID_L, LID_W, LID_T = 0.106, 0.065, 0.003  # the same outline as a pan (530 x 325 mm at 1:5), 3 mm plate
NOTCH_W, NOTCH_D = 0.026, 0.025  # across (local y) and deep (local x)
BAR = (0.050, 0.020, 0.014)  # handle bar on top at the middle: 50 long (local x), 20 wide, 14 tall
BAR_Z = LID_T + BAR[2] / 2
_TAB_W = (LID_W - NOTCH_W) / 2  # 19.5 mm: the plate on either side of the notch
LID_SHAPES = {  # kind: ((part, local centre, size, colour key), ...)
    "notched": (
        ("plate", (NOTCH_D / 2, 0.0, LID_T / 2), (LID_L - NOTCH_D, LID_W, LID_T), "plate"),
        ("tab_left", (-LID_L / 2 + NOTCH_D / 2, NOTCH_W / 2 + _TAB_W / 2, LID_T / 2), (NOTCH_D, _TAB_W, LID_T), "plate"),
        ("tab_right", (-LID_L / 2 + NOTCH_D / 2, -NOTCH_W / 2 - _TAB_W / 2, LID_T / 2), (NOTCH_D, _TAB_W, LID_T),
         "plate"),
        ("bar", (0.0, 0.0, BAR_Z), BAR, "bar"),
    ),
    "solid": (
        ("plate", (0.0, 0.0, LID_T / 2), (LID_L, LID_W, LID_T), "plate"),
        ("bar", (0.0, 0.0, BAR_Z), BAR, "bar"),
    ),
}
LID_MASS = {"notched": 0.046, "solid": 0.048}
LIDS = {  # name: (kind, origin x, y at the start, yaw at the start (rad): local +x's heading)
    # Left counter: the notched lid further out, notch toward +y (away from the pans); the solid lid nearer the arms.
    "lid_notched_left": ("notched", 0.370, 0.270, -math.pi / 2),
    "lid_solid_left": ("solid", 0.250, 0.270, -math.pi / 2),
    # Right counter: the solid lid further out; the notched lid nearer the arms, notch toward -y.
    "lid_solid_right": ("solid", 0.370, -0.270, math.pi / 2),
    "lid_notched_right": ("notched", 0.250, -0.270, math.pi / 2),
}
LID_NAMES = tuple(LIDS)
# The plate's four outer corners on its underside (base frame): what a ruler at each corner checks.
PLATE_CORNERS = tuple((sx * LID_L / 2, sy * LID_W / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1))

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_Z_TOL = 0.002  # every plate corner's underside within 2 mm of the rim top (a lid on a lid is at least 3 mm up)
TILT_MAX_DEG = 3.0  # lying flat
CENTRE_TOL = 0.008  # the plate's middle within 8 mm of the pan's centre, seen from above
YAW_TOL_DEG = 6.0  # its long side within 6 degrees of the pan's (for a notched lid on a utensil pan: notch to the front)
PAN_CREDIT = 0.25
STILL_WINDOW_S = 0.5  # 'still': no plate corner moved more than STILL_TOL over the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a lid moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
PAN_STEEL = (0.74, 0.75, 0.77)
DECK_STEEL = (0.50, 0.51, 0.53)
UTENSIL_STEEL = (0.80, 0.81, 0.83)
LID_STEEL = (0.84, 0.85, 0.87)
BAR_BLACK = (0.08, 0.08, 0.09)
