"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W. The
room is at 1:16 scale.

A wet room stands on the table: a carpet floor, a side wall on the left (+y), a side wall on the right (-y) and a back
wall (far from the arms); the fourth wall, nearest the arms, is left out so they can reach in, and the three walls are
cut down to their bottom 16 cm so the camera sees the floor. Four air movers stand in the middle of the room as the crew
wheeled them in, two on each side of the room, all blowing toward the right-hand wall.
"""

import math

TOP = 0.7475  # station table top (world z)
SCALE = 16  # 1:16

# -- the room (fixed) --------------------------------------------------------------------------------------------------
CARPET_T = 0.001
FLOOR = CARPET_T  # the carpet's top, above the table top: the movers stand on it
ROOM_X = (0.120, 0.470)  # the carpet: from the open side (nearest the arms) to the back wall's face
ROOM_Y = (-0.185, 0.185)  # between the side walls' faces
WALL_T = 0.010  # walls 10 mm thick ...
WALL_H = 0.010  # ... and 10 mm (16 cm) tall: the rest of each wall is cut away
LEFT_FACE = ROOM_Y[1]  # the left wall's face (the room side), y +0.185
RIGHT_FACE = ROOM_Y[0]  # the right wall's face, y -0.185
BACK_FACE = ROOM_X[1]  # the back wall's face, x 0.470
WALLS = (  # (name, (x0, x1), (y0, y1)): fixed boxes from the table top to WALL_H, side by side (no overlaps)
    ("left_wall", (ROOM_X[0], ROOM_X[1] + WALL_T), (LEFT_FACE, LEFT_FACE + WALL_T)),
    ("right_wall", (ROOM_X[0], ROOM_X[1] + WALL_T), (RIGHT_FACE - WALL_T, RIGHT_FACE)),
    ("back_wall", (BACK_FACE, BACK_FACE + WALL_T), ROOM_Y),
)

# -- the air movers: loose rigid bodies (mechkit compounds) -----------------------------------------------------------
# Mover frame: origin at the middle of the body's base, +u the airflow (toward the outlet), +v its own left, +z up.
# Each entry: (name, centre (u, v, z), size (u, v, z), colour key). No two shapes overlap (mechkit spreads the mass by
# volume). A snail-shell air mover about 48 cm long, 42 cm wide and 48 cm tall, its outlet at the bottom of the front.
MOVER_MASS = 0.025
# The housing is two boxes, blue behind and black over its front 5 mm (the outlet end), so the top camera shows which
# way a mover blows whichever way it faces (the outlet alone sits low at the front and hides behind the housing when it
# faces away from the camera). Together they are the 30 x 26 x 30 mm housing; the pads (u -10 .. 10) hold the blue box.
MOVER_SHAPES = (
    ("body", (-0.0025, 0.0, 0.015), (0.025, 0.026, 0.030), "mover_blue"),  # u -15 .. 10, v -13 .. 13, z 0-30
    ("front", (0.0125, 0.0, 0.015), (0.005, 0.026, 0.030), "black"),  # the housing's outlet end: u 10-15, z 0-30
    ("outlet", (0.0175, 0.0, 0.006), (0.005, 0.026, 0.012), "black"),  # front face, bottom: u 15-20, z 0-12
    ("handle", (0.0, 0.0, 0.033), (0.020, 0.010, 0.006), "handle_grey"),  # on top, along the airflow: z 30-36
)
OUTLET_BOX = ((0.015, 0.020), (-0.013, 0.013), (0.0, 0.012))  # the outlet's extent (u, v, z) in the mover frame
# The outline the scorer uses: (back, front, right, left, top) in the mover frame, m.
OUTLINE = (0.015, 0.020, 0.013, 0.013, 0.036)
MOVERS = ("air_mover_1", "air_mover_2", "air_mover_3", "air_mover_4")

# Start (origin x, y, heading): two on each side of the middle of the room, all four blowing toward the right wall
# (-y). Movers 1 and 2 are on the left (the left arm's side), 3 and 4 on the right.
START = {
    "air_mover_1": (0.330, 0.130, -math.pi / 2),
    "air_mover_2": (0.420, 0.130, -math.pi / 2),
    "air_mover_3": (0.330, -0.130, -math.pi / 2),
    "air_mover_4": (0.420, -0.130, -math.pi / 2),
}

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
TILT_PLACED_DEG = 10.0  # 'placed': upright within 10 degrees ...
FLOOR_TOL = 0.003  # ... its lowest point within 3 mm of the carpet ...
STILL_WINDOW_S = 0.5  # ... its centre inside the room, and still: its origin's path over the last 0.5 s, turns
STILL_PATH = 0.005  # counted at its outline's corner, under 5 mm (positions: gripped velocities chatter)
GAP_MAX = 0.015  # 'set': the outlet within 15 mm (24 cm) of the wall it blows into ...
ANGLE_DEG = (15.0, 45.0)  # ... blowing into that wall at 15 to 45 degrees
SPACING = (0.190, 0.305)  # two set movers on one wall count only if their centres are 190 to 305 mm (10-16 ft) apart
PER_WALL = 2  # a wall credits at most two movers
CREDIT = 0.25  # per credited mover
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a mover moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def turn_radius() -> float:
    """Turns about the vertical are counted at the outline's farthest corner from the origin."""
    back, front, right, left, _ = OUTLINE
    return math.hypot(max(back, front), max(right, left))


# -- where the demo puts them (the scorer does not use these): clockwise seen from above, 230 mm apart ----------------
# On the left wall the airflow points +x (away from the arms) and 30 degrees into the wall; on the right wall it points
# -x (toward the arms) and 30 degrees into the wall. The outlet's wall-side corner is 21.3 mm from the centre toward the
# wall (20 sin 30 + 13 cos 30), so a centre 156 mm out leaves a 7.7 mm gap (the band is 0 to 15).
TARGET = {
    "air_mover_1": (0.190, 0.156, math.pi / 6),
    "air_mover_2": (0.420, 0.156, math.pi / 6),
    "air_mover_3": (0.190, -0.156, -5 * math.pi / 6),
    "air_mover_4": (0.420, -0.156, -5 * math.pi / 6),
}

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "carpet": (0.50, 0.51, 0.53),  # mid-grey felt (the carpet): neutral against the beige table, the blue housings,
    # the black front ends and outlets and the off-white walls
    "wall": (0.93, 0.92, 0.88),  # off-white painted MDF
    "mover_blue": (0.10, 0.35, 0.75),
    "black": (0.05, 0.05, 0.05),
    "handle_grey": (0.55, 0.56, 0.58),
}
