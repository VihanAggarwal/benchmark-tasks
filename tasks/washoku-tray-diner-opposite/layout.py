"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A Japanese set meal (teishoku, the ichiju-nisai form: rice, one soup, a main and a side dish) is laid on a tray for a
guest who sits beyond the tray's far edge, on an indigo cushion, facing the robot (facing -x). The basic washoku
layout, seen from the diner: rice front left, soup front right (to the right of the rice), the main dish back right
(beyond the soup), the side dish back left (beyond the rice); a whole fish lies with its head to the diner's left and
its belly toward the diner. The diner's left is the robot's right (-y) and the diner's front is the tray's far half
(x above the centre line). Laid from the robot's own side, every dish lands in the opposite quadrant.

Every dish is a loose mechkit compound (one rigid body): a bowl is a cylinder with a thin disc of food on top, the fish
plate is a slab with a rigid fish model on it. Each dish's base frame has its origin at the middle of its bottom face
and +z up through its top; the fish's head is along the plate's local +x, its belly along local +y.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the tray (fixed, lacquered wood): the meal is laid on it ------------------------------------------------------
TRAY_X0, TRAY_X1 = 0.220, 0.440  # 220 mm deep along x
TRAY_Y0, TRAY_Y1 = -0.150, 0.150  # 300 mm wide along y
TRAY_H = 0.006
TRAY_TOP = TRAY_H  # the dishes stand on it
MID_X = (TRAY_X0 + TRAY_X1) / 2  # 0.330: the tray's centre line across (front half beyond it, seen from the diner)
MID_Y = 0.0  # its centre line from front to back (the diner's left is -y)

# -- the diner's cushion (fixed, a flat indigo zabuton beyond the tray's far edge: where the guest sits) -----------
CUSHION_X0, CUSHION_X1 = 0.455, 0.575  # 120 mm
CUSHION_Y0, CUSHION_Y1 = -0.100, 0.100  # 200 mm
CUSHION_H = 0.002

# -- two serving boards (fixed, light wood), one each side, where the dishes wait ----------------------------------
BOARD_X0, BOARD_X1 = 0.160, 0.405  # 245 mm
BOARD_Y_IN, BOARD_Y_OUT = 0.245, 0.355  # 110 mm; the left board at +y, the right one mirrored at -y
BOARD_H = 0.002
BOARD_TOP = BOARD_H

# -- the four dishes ------------------------------------------------------------------------------------------------
DISHES = ("rice", "soup", "side", "fish")  # the trace and the tests use these names; task.py names the objects

# Bowls: (body radius, body height, food-disc radius, extra band height on top of the body (soup's red rim), mass).
RICE_R, RICE_BODY_H, RICE_FOOD_R = 0.033, 0.044, 0.027  # 66 mm porcelain rice bowl (chawan)
SOUP_R, SOUP_BODY_H, SOUP_RIM_H, SOUP_FOOD_R = 0.032, 0.045, 0.004, 0.026  # 64 mm lacquer soup bowl (owan)
SIDE_R, SIDE_BODY_H, SIDE_FOOD_R = 0.0275, 0.034, 0.022  # 55 mm side-dish bowl (kobachi)
FOOD_H = 0.001  # the disc of food, 1 mm proud of the bowl's rim
RICE_H = RICE_BODY_H + FOOD_H  # 45 mm in all
SOUP_H = SOUP_BODY_H + SOUP_RIM_H + FOOD_H  # 50 mm
SIDE_H = SIDE_BODY_H + FOOD_H  # 35 mm
RICE_MASS, SOUP_MASS, SIDE_MASS = 0.080, 0.070, 0.050
BOWL = {"rice": (RICE_R, RICE_H), "soup": (SOUP_R, SOUP_H), "side": (SIDE_R, SIDE_H)}  # radius, height

# The fish plate: a rectangular glazed plate with a painted resin food-replica fish glued on it (local frame: origin at
# the middle of the plate's underside, +x toward the fish's head, +y toward its belly).
PLATE_L, PLATE_W, PLATE_T = 0.130, 0.060, 0.012
FISH_BODY_CENTER, FISH_BODY_SIZE = (-0.010, 0.0, PLATE_T + 0.009), (0.080, 0.035, 0.018)  # x -0.050 .. +0.030
FISH_HEAD_CENTER, FISH_HEAD_SIZE = (0.040, 0.0, PLATE_T + 0.008), (0.020, 0.028, 0.016)  # x +0.030 .. +0.050
FISH_TAIL_CENTER, FISH_TAIL_SIZE = (-0.055, 0.0, PLATE_T + 0.002), (0.010, 0.024, 0.004)  # x -0.060 .. -0.050
FISH_BELLY_CENTER, FISH_BELLY_SIZE = (-0.010, 0.0135, PLATE_T + 0.0185), (0.070, 0.008, 0.001)  # white, +y edge
FISH_RIDGE_CENTER, FISH_RIDGE_SIZE = (-0.010, -0.0135, PLATE_T + 0.0195), (0.060, 0.008, 0.003)  # dark, -y edge
FISH_LEN = 0.110  # tail tip to head
FISH_TOP = PLATE_T + 0.018 + 0.003  # 33 mm: the dorsal ridge's top
FISH_MASS = 0.120  # plate and fish together
HEAD_LOCAL = (1.0, 0.0, 0.0)  # the fish's head, along local +x
BELLY_LOCAL = (0.0, 1.0, 0.0)  # its belly, along local +y

# Where each dish starts: on the serving boards, out of the other arm's reach. The bowls stand upright; the fish
# plate lies along x with the fish's head away from the robot (+x) and its belly toward +y.
START = {  # (x, y, yaw in rad)
    "rice": (0.330, -0.300, 0.0),
    "side": (0.215, -0.300, 0.0),
    "soup": (0.205, 0.300, 0.0),
    "fish": (0.330, 0.300, 0.0),
}

# -- the rule (scoring.py) ------------------------------------------------------------------------------------------
# Each dish's quadrant as signs of (x - MID_X, y - MID_Y): front is +x (toward the diner), the diner's left is -y.
QUADRANT = {"rice": (+1, -1), "soup": (+1, +1), "side": (-1, -1), "fish": (-1, +1)}
QUADRANT_NAME = {"rice": "front left", "soup": "front right", "side": "back left", "fish": "back right"}
INSET = 0.010  # a dish's centre at least 10 mm inside its quadrant from both centre lines (none astride a line)
OVERHANG = 0.005  # its footprint may stick out past the tray's edge by at most 5 mm
REST_TOL = 0.003  # its lowest point within 3 mm of the tray top (standing on the tray, not on a dish or held up)
UPRIGHT_DEG = 10.0  # tilted at most 10 degrees
TIP_DEG = 45.0  # tilted past 45 degrees at any step: tipped (food spilled), 0 for the rest of the trial
OVERLAP = 0.001  # two footprints crossing by more than 1 mm: both dishes 0
FISH_DIR_DEG = 15.0  # head within 15 degrees of the diner's left (-y), belly within 15 degrees of the diner (+x)
HEAD_WANT = (0.0, -1.0)  # seen from above
BELLY_WANT = (1.0, 0.0)
STILL_S = 1.0  # 'still': its centre's path over the last 1 s ...
STILL_PATH = 0.003  # ... under 3 mm (a path, not the reported velocity, which chatters while a part is gripped)
POINTS = 0.25  # each dish
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a dish that lands on its starting mark having moved this much further in one control step than
#                    its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# 'let go': rigstate reports each gripper's pad-tip point (midway between its finger tips, 13 mm past the TCP) and how
# far it is open, not its yaw. A finger (FINGER_THICK outside the gap, 2 x FINGER_HALF_WIDTH across it) can touch a
# dish only if the dish's footprint is within release_reach(opening) of the pad-tip point horizontally; a gripper whose
# pad-tip point is also lower than RELEASE_ABOVE over the dish's top may be holding, pressing or propping it.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
TRAY_RED = (0.42, 0.07, 0.06)  # vermilion-brown lacquer
CUSHION_INDIGO = (0.14, 0.17, 0.42)
BOARD_WOOD = (0.80, 0.66, 0.45)
PORCELAIN = (0.90, 0.88, 0.82)  # the rice bowl, cream-white
RICE_WHITE = (0.99, 0.99, 0.97)
LACQUER_BLACK = (0.06, 0.05, 0.05)
LACQUER_RED = (0.66, 0.09, 0.06)
MISO_BROWN = (0.58, 0.40, 0.20)
SIDE_BLUE = (0.16, 0.30, 0.66)
GREENS = (0.25, 0.55, 0.20)
PLATE_WHITE = (0.96, 0.96, 0.94)
FISH_GRILLED = (0.52, 0.46, 0.38)
FISH_DARK = (0.20, 0.17, 0.14)
FISH_BELLY = (0.94, 0.93, 0.90)


def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the margin
    (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + RELEASE_MARGIN


def footprint_half(name: str) -> tuple[float, float]:
    """Half extents (along local x, local y) of a dish's footprint: a bowl's radius both ways, the plate's half sizes."""
    if name == "fish":
        return PLATE_L / 2, PLATE_W / 2
    r = BOWL[name][0]
    return r, r


assert -PLATE_L / 2 <= FISH_TAIL_CENTER[0] - FISH_TAIL_SIZE[0] / 2 and FISH_HEAD_CENTER[0] + FISH_HEAD_SIZE[0] / 2 \
    <= PLATE_L / 2  # the fish lies within the plate
assert abs((FISH_HEAD_CENTER[0] + FISH_HEAD_SIZE[0] / 2) - (FISH_TAIL_CENTER[0] - FISH_TAIL_SIZE[0] / 2)
           - FISH_LEN) < 1e-9
