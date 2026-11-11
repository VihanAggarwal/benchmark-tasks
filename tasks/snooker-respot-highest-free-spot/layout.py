"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A snooker table at 1:6, its long axis along y, the baulk end at -y (the right arm's side), the top end at +y.
WPBSA Official Rules (May 2022), Section 1: playing area 3569 x 1778 mm; the baulk line 737 mm from the face of the
bottom cushion; the D a semicircle of radius 292 mm on the baulk line; viewed from the baulk end the yellow spot is
the right-hand corner of the D, the green the left-hand corner, the brown the middle of the baulk line; the blue
spot is the centre of the table, the pink midway between the centre and the face of the top cushion, the black
324 mm from the face of the top cushion. Viewed from the baulk end (looking toward +y) the right hand is +x, so the
yellow spot is at +x. Spot values: yellow 2, green 3, brown 4, blue 5, pink 6, black 7.

In this layout reds sit on the black and the pink spots; the blue and the brown are in play away from their spots
and the green is on its own. Section 3 Rule 7(e): a colour whose own spot is occupied goes on the highest-value
spot available, so the black goes on the blue spot; the yellow's own spot is free.
"""

import math

TOP = 0.7475  # station table top
SCALE = 6.0

# -- the table -----------------------------------------------------------------------------------------------------
PLAY_W = 1.778 / SCALE  # 296.3 mm across (along x)
PLAY_L = 3.569 / SCALE  # 594.8 mm long (along y)
CX = 0.350  # the table's centre (the blue spot)
CY = 0.000
BED_X0, BED_X1 = CX - PLAY_W / 2, CX + PLAY_W / 2  # 0.2018 .. 0.4982
BED_Y0, BED_Y1 = CY - PLAY_L / 2, CY + PLAY_L / 2  # -0.2974 .. +0.2974 (the bottom cushion face at -y)
BED_H = 0.010  # the bed: a fixed 10 mm slab on the table
BED_TOP = TOP + BED_H
RAIL_W = 0.012  # cushion rails round the bed, 12 mm wide, standing on the table ...
RAIL_H = 0.020  # ... 20 mm tall: their tops 10 mm above the playing surface

# Markings: thin white stickers, 0.5 mm thick, sunk 0.3 mm into the bed so they stand 0.2 mm proud (a puck on a
# spot sits level within 1 degree). Every spot is the same 8 mm white disc, unlabelled.
MARK_T = 0.0005
MARK_PROUD = 0.0002
MARK_TOP = BED_TOP + MARK_PROUD
LINE_W = 0.004  # the baulk line and the D, 4 mm wide so the top camera shows them (positions are their centres)
SPOT_R = 0.004  # 8 mm spots
D_SEGMENTS = 12  # the D drawn as 12 straight strips of 15 degrees

BAULK_Y = BED_Y0 + 0.737 / SCALE  # -0.17458
D_R = 0.292 / SCALE  # 48.67 mm
SPOTS = {  # name -> (x, y)
    "yellow": (CX + D_R, BAULK_Y),  # right-hand corner of the D seen from the baulk end (+x)
    "green": (CX - D_R, BAULK_Y),
    "brown": (CX, BAULK_Y),
    "blue": (CX, CY),
    "pink": (CX, (CY + BED_Y1) / 2),
    "black": (CX, BED_Y1 - 0.324 / SCALE),
}
VALUE = {"yellow": 2, "green": 3, "brown": 4, "blue": 5, "pink": 6, "black": 7}

# -- the balls (pucks) ------------------------------------------------------------------------------------------
PUCK_R = 0.015  # 30 mm across
PUCK_H = 0.020
PUCK_MASS = 0.025
RESPOT = ("black", "yellow")  # the two colours to re-spot, each on its tray
TABLE = ("red_on_black", "red_on_pink", "red", "pink", "blue", "green", "brown")  # already on the table
NAMES = (*RESPOT, *TABLE)
TRAY = 0.060  # trays: fixed 60 x 60 x 10 mm wooden slabs beside the table's two ends
TRAY_H = 0.010
TRAY_TOP = TOP + TRAY_H
TRAYS = {"black": (0.240, 0.370), "yellow": (0.240, -0.370)}
START = {  # (x, y) of each puck's centre
    "black": TRAYS["black"],
    "yellow": TRAYS["yellow"],
    "red_on_black": SPOTS["black"],
    "red_on_pink": SPOTS["pink"],
    "red": (0.300, 0.070),
    "pink": (0.400, 0.090),
    "blue": (0.300, -0.070),
    "green": SPOTS["green"],
    "brown": (0.420, -0.100),
}
ON_MARKS = ("red_on_black", "red_on_pink", "green")  # pucks that start standing on a spot (0.2 mm proud)


def start_bottom(name: str) -> float:
    """World z of a puck's bottom face at the start."""
    if name in RESPOT:
        return TRAY_TOP
    return MARK_TOP if name in ON_MARKS else BED_TOP


# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------
SPOT_TOL = 0.008  # a colour is on a spot when its centre is within 8 mm of it (spots are 48.7 mm apart at least)
TOUCH = 2 * PUCK_R + 0.001  # 31 mm: two pucks whose centres are closer than this touch (also 'spot occupied')
DISTURB = 0.008  # a table puck more than 8 mm from where it stood at the start: latched 0
FLAT_DEG = 10.0  # a colour counts only if tilted less than this
SEAT_TOL = 0.003  # ... its lowest point within 3 mm of the bed top (on the bed, not stacked or held up)
STILL_WINDOW_S = 0.5  # 'still': the centre's path in each of the last two 0.5 s windows ...
STILL_PATH = 0.0015  # ... under 1.5 mm (a path, not the reported velocity, which chatters while a part is gripped)
STILL_WINDOWS = 2
BLACK_FULL = "blue"  # the black's spot under Rule 7(e) in this layout
BLACK_PART = 0.25  # the black on any other spot (in this layout only the brown or the yellow is reachable)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a puck that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.010  # ... 'on its starting mark': within this of it

# 'Released' (rigstate): each gripper's pad-tip point (midway between its two finger tips, 13 mm past the TCP)
# and how far it is open are read, but not its yaw, so a finger may be on any side of that point. With the fingers
# FULL_GAP x opening apart, a finger (FINGER_THICK thick outside the gap, 2 x FINGER_HALF_WIDTH wide across it) can
# touch a puck only if the puck's axis is within release_reach(opening) of that point horizontally. A gripper whose
# pad-tip point is lower than RELEASE_ABOVE over the puck's top while within that reach may be holding, pressing or
# propping it: the puck is not released.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010

# -- colours (sRGB) -----------------------------------------------------------------------------------------------
CLOTH = (0.16, 0.56, 0.20)
RAIL_GREEN = (0.05, 0.28, 0.10)
WHITE = (0.97, 0.97, 0.97)
WOOD = (0.62, 0.45, 0.26)
COLOUR = {
    "black": (0.04, 0.04, 0.04),
    "yellow": (0.98, 0.85, 0.08),
    "red_on_black": (0.80, 0.04, 0.04),
    "red_on_pink": (0.80, 0.04, 0.04),
    "red": (0.80, 0.04, 0.04),
    "pink": (1.00, 0.55, 0.75),
    "blue": (0.10, 0.30, 0.90),
    "green": (0.02, 0.25, 0.08),
    "brown": (0.45, 0.24, 0.08),
}


def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the puck's
    radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + PUCK_R + RELEASE_MARGIN


def d_strips():
    """The D's strips: (centre x, centre y, length, yaw) for 12 chords from the green spot round the baulk side
    (-y) to the yellow spot, each 1 line-width longer than its chord so the joints close."""
    out = []
    for k in range(D_SEGMENTS):
        a0 = math.pi + k * math.pi / D_SEGMENTS
        a1 = a0 + math.pi / D_SEGMENTS
        am = (a0 + a1) / 2
        chord = 2 * D_R * math.sin(math.pi / D_SEGMENTS / 2)
        r_mid = D_R * math.cos(math.pi / D_SEGMENTS / 2)
        out.append((CX + r_mid * math.cos(am), BAULK_Y + r_mid * math.sin(am), chord + LINE_W, am + math.pi / 2))
    return out


assert abs(BED_X0 - 0.2018) < 0.0001 and abs(BED_Y1 - 0.2974) < 0.0001
assert abs(SPOTS["yellow"][0] - 0.3987) < 0.0001 and abs(BAULK_Y + 0.1746) < 0.0001
assert abs(SPOTS["pink"][1] - 0.1487) < 0.0001 and abs(SPOTS["black"][1] - 0.2434) < 0.0001
