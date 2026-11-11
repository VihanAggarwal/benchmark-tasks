"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A 1:100 deck model of an 80 m power-driven cargo ship, 13 m in the beam: 800 x 130 mm. Its bow points toward
+y (the left arm) and its stern toward -y. Facing forward (+y) with z up, the right hand points +x, so starboard is
+x (the side away from the arms) and port is -x (the near side).

The rule (COLREG 1972, Rules 21 and 23 and Annex I section 3, with IMO MSC.1/Circ.1260/Rev.1 for 'near the side'):
- a vessel of 50 m or more shows two masthead lights over the fore and aft centreline;
- the forward one not more than one quarter of the length from the stem (20 m: 200 mm);
- the two at least half the length apart horizontally (40 m: 400 mm; the 100 m cap does not bind at 80 m);
- the sidelights (red to port, green to starboard) not in front of the forward masthead light, at or near the side:
  not more than 10 per cent of the beam inboard, up to 1 m (here the 1 m cap governs: 10 mm).
"""

import math

TOP = 0.7475  # station table top
SCALE = 100.0
M = 1.0 / SCALE  # one metre at 1:100 (10 mm)

# -- the ship ------------------------------------------------------------------------------------------------
SHIP_L = 80.0 * M  # 0.800: length overall, transom to stem
BEAM = 13.0 * M  # 0.130
CL_X = 0.350  # the fore and aft centreline
PORT_X = CL_X - BEAM / 2  # 0.285: the port side (near the arms)
STBD_X = CL_X + BEAM / 2  # 0.415: the starboard side
TRANSOM_Y = -SHIP_L / 2  # -0.400: the square stern
STEM_Y = SHIP_L / 2  # +0.400: the stem, the bow's point
SHOULDER_Y = STEM_Y - 16.0 * M  # 0.240: the parallel sides run from the transom to here, then the last 16 m
# taper straight in to the stem
DECK_H = 0.010  # the deck plate is 10 mm thick, lying on the table
DECK_TOP = TOP + DECK_H
LINE_W = 0.002  # the white centreline, flush with the deck (a separate strip of the plate)
BOW_STEPS = 15  # in simulation the bow's taper is built from this many flat steps (see bow_steps())

# The deck outline as a convex polygon, counter-clockwise seen from above, and which edges are which side.
OUTLINE = (
    (PORT_X, TRANSOM_Y),  # 0 -> 1: transom
    (STBD_X, TRANSOM_Y),  # 1 -> 2: starboard side (parallel)
    (STBD_X, SHOULDER_Y),  # 2 -> 3: starboard bow
    (CL_X, STEM_Y),  # 3 -> 4: port bow
    (PORT_X, SHOULDER_Y),  # 4 -> 0: port side (parallel)
)
PORT_EDGES = (3, 4)
STBD_EDGES = (1, 2)

# The sternlight, already fitted at the stern on the centreline (fixed).
STERN_X, STERN_Y = CL_X, TRANSOM_Y + 1.3 * M  # -0.387: 1.3 m from the transom
STERN_RADIUS = 0.006
STERN_H = 0.015

# -- the four loose light units ----------------------------------------------------------------------------
# Masthead units: white aluminium cylinders 20 mm across and 20 mm tall (2 m at scale), ends alike.
MAST_RADIUS = 0.010
MAST_H = 0.020
MAST_MASS = 0.017  # aluminium, 2.7 g/cm3
# Sidelight units: 20 mm aluminium cubes, red and green; the outboard face stands for the lamp.
SIDE = 0.020
SIDE_MASS = 0.022
WHITE_UNITS = ("masthead_1", "masthead_2")
RED, GREEN = "sidelight_red", "sidelight_green"
NAMES = (*WHITE_UNITS, RED, GREEN)
KIND = {"masthead_1": "cyl", "masthead_2": "cyl", RED: "cube", GREEN: "cube"}
START = {  # (x, y) on the table in front of the deck, every unit upright and square to the table's edges
    "masthead_1": (0.230, 0.150),
    "sidelight_green": (0.230, 0.030),
    "sidelight_red": (0.230, -0.030),
    "masthead_2": (0.230, -0.090),
}

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------
F_MIN_Y = STEM_Y - SHIP_L / 4  # 0.200: the forward masthead light not more than L/4 (20 m) from the stem
MIN_SEPARATION = SHIP_L / 2  # 0.400: the after one at least L/2 (40 m) abaft the forward one
CL_TOL = 0.006  # 'over the centreline': the unit's centre within 6 mm (0.6 m) of it
SIDE_BAND = min(0.10 * BEAM, 1.0 * M)  # 0.010: outboard face at most 1 m inboard of the side
UPRIGHT_DEG = 10.0  # 'standing': tilted at most this
SEAT_TOL = 0.003  # 'standing': its lowest point within this of the deck top
STILL_WINDOW_S = 0.5  # 'still for 1 s': the unit's centre path ...
STILL_WINDOWS = 2  # ... in each of the last two 0.5 s windows ...
STILL_PATH = 0.0015  # ... under 1.5 mm (a path, not the reported velocity, which chatters while gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # every unit back on its starting mark, one moved this much further in one step than it could
START_TOL = 0.030  # ... 'on its starting mark': within this of it

# 'Let go' (rigstate reports each gripper's pad-tip point and opening, not its yaw): a finger can touch a unit only
# if the unit's axis is within release_reach(opening, unit) of that point horizontally. A gripper whose pad-tip point
# is lower than RELEASE_ABOVE over the unit's top while within that reach may be holding or propping it.
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010

# Colours (sRGB).
DECK_GREY = (0.46, 0.48, 0.50)
LINE_WHITE = (0.97, 0.97, 0.97)
UNIT_WHITE = (0.95, 0.95, 0.93)
LAMP_RED = (0.85, 0.06, 0.05)
LAMP_GREEN = (0.05, 0.65, 0.20)


def unit_radius(name: str) -> float:
    """The unit's horizontal reach from its axis: a cylinder's radius, a cube's half-diagonal (m)."""
    return MAST_RADIUS if KIND[name] == "cyl" else SIDE / math.sqrt(2)


def unit_height(name: str) -> float:
    return MAST_H if KIND[name] == "cyl" else SIDE


def release_reach(opening: float, name: str) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the unit's
    horizontal reach and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + unit_radius(name) + RELEASE_MARGIN


def half_width(y: float) -> float:
    """The deck's half-width (m) at y: the full half-beam aft of the shoulder, tapering straight to 0 at the stem."""
    if y <= SHOULDER_Y:
        return BEAM / 2
    return max(0.0, BEAM / 2 * (STEM_Y - y) / (STEM_Y - SHOULDER_Y))


def bow_steps():
    """The simulated bow: BOW_STEPS strips along y from the shoulder to the stem, each as wide as the true outline
    at its middle (so the steps sit within 2.2 mm either side of the straight taper). Yields (y0, y1, half_width)."""
    h = (STEM_Y - SHOULDER_Y) / BOW_STEPS
    for k in range(BOW_STEPS):
        y0, y1 = SHOULDER_Y + k * h, SHOULDER_Y + (k + 1) * h
        yield y0, y1, half_width((y0 + y1) / 2)


assert abs(SHIP_L - 0.800) < 1e-9 and abs(BEAM - 0.130) < 1e-9
assert abs(F_MIN_Y - 0.200) < 1e-9 and abs(MIN_SEPARATION - 0.400) < 1e-9
assert abs(SIDE_BAND - 0.010) < 1e-9  # the 1 m cap governs (10 % of 13 m is 1.3 m)
assert MIN_SEPARATION < 100.0 * M  # the 100 m cap does not bind
