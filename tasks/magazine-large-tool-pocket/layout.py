"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

The tool magazine is a bench-scale ring: a fixed round plate with eight square sockets on a circle around its
centre, every 45 degrees, numbered P1 (straight ahead, marked by a red tab) to P8 counter-clockwise seen from above
(P3 on the robot's left, P7 on its right). Tools stand shank-down in the sockets.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the magazine plate (fixed round plate) --------------------------------------------------------------------------
MAG_X, MAG_Y = 0.300, 0.000  # centre of the ring
DISC_R = 0.135  # 270 mm across: holds every socket's corners (132 mm out at P2, P4, P6, P8)
DISC_H = 0.010
DISC_TOP = DISC_H  # 10 mm above the table

# -- the eight pockets: square sockets of four fixed walls, sides parallel to x and y -------------------------------
RING_R = 0.090  # pocket centres 90 mm from the ring's centre: 68.9 mm apart (centre to centre) around the ring
N_POCKETS = 8
POCKET_IN = 0.052  # 52 x 52 mm inside: 6 mm of play either side of a 40 mm shank
WALL_T = 0.004
WALL_H = 0.025  # 25 mm deep sockets
WALL_TOP = DISC_TOP + WALL_H  # 35 mm above the table
POCKET_OUT = POCKET_IN + 2 * WALL_T  # 60 mm outside; diagonal neighbours' walls are 3.6 mm apart


def pocket_xy(k: int) -> tuple[float, float]:
    """Centre of pocket Pk (k = 1..8), seen from above: 45 degrees apart counter-clockwise from straight ahead."""
    a = math.radians(45.0 * (k - 1))
    return (MAG_X + RING_R * math.cos(a), MAG_Y + RING_R * math.sin(a))


POCKETS = {k: pocket_xy(k) for k in range(1, N_POCKETS + 1)}


def neighbours(k: int) -> tuple[int, int]:
    """The two pockets next to Pk around the ring (P8 and P1 are neighbours)."""
    return ((k - 2) % N_POCKETS + 1, k % N_POCKETS + 1)


PITCH = 2 * RING_R * math.sin(math.pi / N_POCKETS)  # 0.0689

# The red tab marking P1: a thin red block on the plate just outside P1's front wall.
TAB_SIZE = (0.008, 0.020, 0.003)
TAB_X = POCKETS[1][0] + POCKET_OUT / 2 + 0.002 + TAB_SIZE[0] / 2  # 0.426
TAB_Y = 0.0

# -- the tools -------------------------------------------------------------------------------------------------------
# Small tools (drills and end mills in their holders): plain upright cylinders, either end down.
TOOL_R = 0.020
TOOL_H = 0.070
TOOL_MASS = 0.15
LOADED = {"tool_p1": 1, "tool_p2": 2, "tool_p5": 5}  # already in the magazine: name -> its own pocket
NEW_SMALL = ("new_tool_a", "new_tool_b")  # on the left tray
FACE = "face_mill"  # on the right tray
TOOL_NAMES = (*LOADED, *NEW_SMALL, FACE)

# The face mill: one loose rigid body (mechkit compound), base frame at the centre of the shank's bottom face, local
# +z up the tool's axis. A 40 mm shank (the holder) under a 76 mm (3 inch) cutter body.
SHANK_R, SHANK_H = 0.020, 0.040
BODY_R, BODY_H = 0.038, 0.030
FM_H = SHANK_H + BODY_H  # 0.070
FM_MASS = 0.28
SHANK_CENTER = (0.0, 0.0, SHANK_H / 2)
BODY_CENTER = (0.0, 0.0, SHANK_H + BODY_H / 2)
# Rims of the face mill's outline along its axis (distance from the bottom face, radius): its lowest point is the
# lowest of these circles.
FM_RIMS = ((0.0, SHANK_R), (SHANK_H, SHANK_R), (SHANK_H, BODY_R), (FM_H, BODY_R))

# -- the trays and the starting marks --------------------------------------------------------------------------------
TRAY_H = 0.003
TRAY_TOP = TRAY_H
LEFT_TRAY_X, LEFT_TRAY_Y = 0.210, 0.230
LEFT_TRAY_SIZE = (0.200, 0.090, TRAY_H)
RIGHT_TRAY_X, RIGHT_TRAY_Y = 0.200, -0.220
RIGHT_TRAY_SIZE = (0.110, 0.110, TRAY_H)
NEW_START = {"new_tool_a": (0.260, 0.230), "new_tool_b": (0.160, 0.230)}  # 100 mm apart along x
FM_START = (RIGHT_TRAY_X, RIGHT_TRAY_Y)

# -- what the rule makes of this layout -------------------------------------------------------------------------------
# With P1, P2 and P5 loaded, P7 is the only pocket whose two neighbours are empty (P3 is next to P2, P4 and P6 next
# to P5, P8 next to P1). Once the face mill has P7, P6 and P8 must stay empty, so the small tools go in P3 and P4.
FACE_POCKET = 7
SMALL_POCKETS = (3, 4)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_HALF = POCKET_IN / 2  # 'in pocket k': the centre of the tool's bottom face inside the 52 mm opening
SEAT_TILT_DEG = 5.0  # seated: upright within 5 degrees
SEAT_FLOOR_TOL = 0.003  # seated: the bottom face's centre within 3 mm of the plate's top
ON_MAGAZINE_H = 0.075  # 'on the magazine': lowest point under 75 mm above the plate (a seated tool's top is at 70)
STILL_WINDOW_S = 0.5  # 'at rest': neither end of the tool's axis moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a new tool or the face mill moving this far in one control step was teleported: a new episode
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
PLATE_GREY = (0.36, 0.38, 0.41)
WALL_DARK = (0.20, 0.21, 0.23)
TAB_RED = (0.80, 0.10, 0.10)
STEEL = (0.66, 0.68, 0.71)  # the tools already loaded
NEW_BLUE = (0.18, 0.38, 0.78)  # the three new tools' holders (the face mill's shank too)
CUTTER_BLACK = (0.12, 0.12, 0.13)  # the face mill's body
TRAY_GREY = (0.78, 0.78, 0.80)
