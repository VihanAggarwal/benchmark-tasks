"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475, and the table top is the floor of both rooms.

A 1:5 model of two rooms' floors: 300 mm floor tiles with 10 mm joints become 60 mm tiles with 2 mm joints (a 62 mm
module), and a 1.70 x 1.39 m room becomes 340 x 278 mm. The left room is 278 mm front to back (x) and 340 mm side to
side (y); the right room is the other way round, 340 mm front to back and 278 mm side to side.

The set-out arithmetic. Across a centre line with D of room across it, a tile whose centre is u from the line has
a grid of tiles and joints that, extended to the two walls, leaves border cuts of

    (D/2 - u - 32 mm) mod 62 mm   and   (D/2 + u - 32 mm) mod 62 mm

(wall to the edge of the last whole tile, minus one 2 mm joint). The two always add up to either 28 mm or 90 mm,
so both are at least half a tile (30 mm) only when they add up to 90. For D = 278 mm that is the tile's centre within
15 mm of the line (centred on it: cuts 45 and 45); for D = 340 mm it is the centre 16 to 46 mm from the line (a
joint on the line, u = 31: cuts 45 and 45), which the 45 mm 'at the crossing' limit caps at 45 mm. Centred on a
340 mm line, or with a joint on a 278 mm line, the cuts are 14 and 14 mm: slivers.
"""

import math

TOP = 0.7475  # station table top: the rooms' floor

# -- tiles and joints -----------------------------------------------------------------------------------------
TILE = 0.060  # a tile's side
JOINT = 0.002  # the joint between tiles
MODULE = TILE + JOINT  # 0.062: one tile and one joint
HALF_TILE = TILE / 2  # 0.030: the narrowest border cut allowed
TILE_T = 0.010  # thickness (a gripper needs the height; 1:5 of a 10 mm tile would be 2 mm)
TILE_SIZE = (TILE, TILE, TILE_T)
TILE_MASS = 0.080  # unglazed terracotta, about 2.2 g/cm3

# -- the two rooms --------------------------------------------------------------------------------------------
WALL_T = 0.010  # wall thickness
WALL_H = 0.025  # wall height


class Room:
    """A room's inside faces (x0 near wall, x1 far wall, y0 right-hand wall, y1 left-hand wall) and its chalked
    centre lines: the front-to-back line runs along x at y = cy, the side-to-side line along y at x = cx."""

    def __init__(self, name: str, x0: float, x1: float, y0: float, y1: float) -> None:
        self.name, self.x0, self.x1, self.y0, self.y1 = name, x0, x1, y0, y1

    @property
    def cx(self) -> float:  # the side-to-side centre line (along y) is at this x
        return round((self.x0 + self.x1) / 2, 9)

    @property
    def cy(self) -> float:  # the front-to-back centre line (along x) is at this y
        return round((self.y0 + self.y1) / 2, 9)

    @property
    def depth(self) -> float:  # front to back (x): the room across the side-to-side line
        return round(self.x1 - self.x0, 9)

    @property
    def width(self) -> float:  # side to side (y): the room across the front-to-back line
        return round(self.y1 - self.y0, 9)


ROOMS = {
    # in front of the left arm: 278 mm front to back, 340 mm side to side; crossing at (0.289, 0.240)
    "left": Room("left", 0.150, 0.428, 0.070, 0.410),
    # in front of the right arm: 340 mm front to back, 278 mm side to side; crossing at (0.290, -0.209)
    "right": Room("right", 0.120, 0.460, -0.348, -0.070),
}

# Chalk lines: fixed blue plastic strips 4 mm wide and 0.5 mm proud of the floor, wall to wall through the room's
# centre. 4 mm (not a chalk line's 2 mm) so the setup screenshot shows them: a 2 mm line is about one pixel there.
# Offsets are measured to a line's centre, so its width does not enter the scoring.
LINE_W = 0.004
LINE_T = 0.0005

# -- the tiles' starting places: inside each room, away from the crossing, turned off square --------------------
TILE_START = {  # name -> (x, y, yaw in degrees)
    "tile_left": (0.205, 0.335, 20.0),  # in the left room, 84 mm from its side-to-side line
    "tile_right": (0.195, -0.275, -25.0),  # in the right room, 95 mm from its side-to-side line
}

# -- where the demo lays them (each direction 15 mm inside its good window either way, except 14 mm outward from a
# joint line, where the 45 mm crossing limit ends the window) -------------------------------------------------
# Left room: centred on the side-to-side line (278 mm across it), a joint on the front-to-back line (340 mm across
# it), the tile on the arm's side of it. Right room: centred on the front-to-back line (278 mm), a joint on the
# side-to-side line (340 mm), the tile on the near side of it.
TARGET = {
    "left": (ROOMS["left"].cx, ROOMS["left"].cy + MODULE / 2),  # (0.289, 0.271)
    "right": (ROOMS["right"].cx - MODULE / 2, ROOMS["right"].cy),  # (0.259, -0.209)
}

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------
AT_CROSSING = 0.045  # 'at the crossing': the tile's centre within this of both of the room's centre lines
FLAT_DEG = 5.0  # 'flat': its 60 x 60 mm faces tilted at most this from level (either face up)
SQUARE_DEG = 5.0  # 'square': its edges turned at most this from the walls
ON_FLOOR = 0.002  # 'on the floor': its lowest point within this of the table top (above or below it)
CUT_EPS = 1e-9  # m: comparisons of a cut with half a tile allow this much floating-point rounding
STILL_WINDOW_S = 0.5  # 'still': no corner of the tile moved more than STILL_TOL over the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a tile landing on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)
START_YAW_TOL_DEG = 10.0

# -- colours (sRGB) --------------------------------------------------------------------------------------------
WALL_WHITE = (0.93, 0.93, 0.90)  # painted plaster
CHALK_BLUE = (0.10, 0.35, 0.85)  # chalk-line blue
TERRACOTTA = (0.74, 0.34, 0.20)  # unglazed quarry tile


def start_yaw(name: str) -> float:
    return math.radians(TILE_START[name][2])
