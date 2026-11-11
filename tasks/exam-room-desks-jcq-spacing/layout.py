"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W. The
room is at 1:10 scale.

A classroom stands on the table: a grey vinyl floor and, at its far end (away from the arms), the front wall with the
whiteboard and the exam clock on it. Four single desks, each with its chair fixed behind it, stand as they were left
after a seminar: two rows facing each other across the middle of the room, every candidate's seat facing sideways.
"""

import math

TOP = 0.7475  # station table top (world z)
SCALE = 10  # 1:10

# -- the room (fixed) --------------------------------------------------------------------------------------------------
FLOOR_T = 0.001
FLOOR = FLOOR_T  # the floor's top, above the table top: the desks stand on it
ROOM_X = (0.150, 0.460)  # the grey floor: from the side nearest the arms to the front wall's face
ROOM_Y = (-0.280, 0.280)
FRONT_FACE = ROOM_X[1]  # the front wall's face (the room side), x 0.460
WALL_T = 0.010  # the front wall: 10 mm thick ...
WALL_H = 0.060  # ... and 60 mm (60 cm) tall: the rest of it is cut away so the camera sees the room
WALL = ((FRONT_FACE, FRONT_FACE + WALL_T), ROOM_Y, (0.0, WALL_H))  # (x0, x1), (y0, y1), (z0, z1)
# On the wall's face (fixed, touching it): the whiteboard, and the exam clock to its left (+y).
WHITEBOARD = ((FRONT_FACE - 0.002, FRONT_FACE), (-0.150, 0.150), (0.015, 0.055))
CLOCK_RIM = ((FRONT_FACE - 0.003, FRONT_FACE), (0.185, 0.215), (0.020, 0.050))
CLOCK_FACE = ((FRONT_FACE - 0.004, FRONT_FACE - 0.003), (0.188, 0.212), (0.023, 0.047))
FRONT = (1.0, 0.0)  # the way every candidate must face, seen from above: toward the front wall (+x)

# -- the desks: loose rigid bodies (mechkit compounds) ----------------------------------------------------------------
# Desk frame: origin on the floor under the middle of the desk top, +u the way the candidate faces (from the chair to
# the desk), +v the candidate's left, +z up. A single exam desk 600 x 450 mm, 720 mm tall, on two steel side frames;
# a chair (seat 360 x 320 mm at 460 mm, back to 860 mm) fixed behind it by a steel rail between the side frames.
# Each entry: (name, centre (u, v, z), size (u, v, z), colour key). No two shapes overlap (mechkit spreads the mass by
# volume); every shape touches the next, so the unit is one solid piece.
DESK_MASS = 0.080
DESK_SHAPES = (
    ("top", (0.0, 0.0, 0.069), (0.045, 0.060, 0.006), "beech"),  # u -22.5 .. 22.5, v -30 .. 30, z 66-72
    ("frame_left", (0.0, 0.027, 0.033), (0.045, 0.006, 0.066), "steel"),  # v 24-30, z 0-66
    ("frame_right", (0.0, -0.027, 0.033), (0.045, 0.006, 0.066), "steel"),  # v -30 .. -24
    ("rail", (-0.0205, 0.0, 0.043), (0.004, 0.048, 0.006), "steel"),  # between the frames' back ends, z 40-46
    ("seat", (-0.0385, 0.0, 0.043), (0.032, 0.036, 0.006), "chair_blue"),  # u -54.5 .. -22.5, z 40-46
    ("chair_left", (-0.0385, 0.016, 0.020), (0.032, 0.004, 0.040), "steel"),  # the chair's side frames, z 0-40
    ("chair_right", (-0.0385, -0.016, 0.020), (0.032, 0.004, 0.040), "steel"),
    ("back", (-0.052, 0.0, 0.066), (0.005, 0.036, 0.040), "chair_blue"),  # u -54.5 .. -49.5, z 46-86
)
# The outline the scorer uses: (back, front, right, left, top) in the desk frame, m.
OUTLINE = (0.0545, 0.0225, 0.030, 0.030, 0.086)
DESK_TOP_CENTRE = (0.0, 0.0, 0.069)  # the desk top's centre in the desk frame: the point the spacing is measured from
DESKS = ("desk_1", "desk_2", "desk_3", "desk_4")

# Start (origin x, y, heading): a seminar layout, two rows facing each other across the middle of the room. Desks 1
# and 2 (the left arm's side) face the right of the room (-y), their chairs toward +y; desks 3 and 4 face the left
# (+y). In each row the two desks stand 100 mm apart centre to centre (40 mm between their side frames).
START = {
    "desk_1": (0.260, 0.200, -math.pi / 2),
    "desk_2": (0.360, 0.200, -math.pi / 2),
    "desk_3": (0.260, -0.200, math.pi / 2),
    "desk_4": (0.360, -0.200, math.pi / 2),
}

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
TILT_MAX_DEG = 10.0  # 'set': upright within 10 degrees ...
FLOOR_TOL = 0.003  # ... its lowest point within 3 mm of the floor ...
STILL_WINDOW_S = 0.5  # ... its desk-top centre over the floor, and still: its origin's path over the last 0.5 s,
STILL_PATH = 0.005  # turns counted at its outline's farthest corner, under 5 mm (positions: gripped velocities chatter)
FACING_DEG = 15.0  # 'facing the front': the chair-to-desk direction within 15 degrees of +x (square to the front wall)
SPACING_MIN = 0.125  # 'spaced': its desk-top centre at least 125 mm (1.25 m) from every other desk's, seen from above
CREDIT = 0.25  # per desk that is set, faces the front and is spaced
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.040  # a desk moving this far in one control step (0.8 m/s) was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def turn_radius() -> float:
    """Turns about the vertical are counted at the outline's farthest corner from the origin."""
    back, front, right, left, _ = OUTLINE
    return math.hypot(max(back, front), max(right, left))


# -- where the demo puts them (the scorer does not use these): a 2 x 2 grid, every desk facing the front -------------
# 160 mm apart front to back, 300 mm across the room, 340 mm on the diagonals (the rule asks for 125 mm).
TARGET = {
    "desk_1": (0.230, 0.150, 0.0),
    "desk_2": (0.390, 0.150, 0.0),
    "desk_3": (0.230, -0.150, 0.0),
    "desk_4": (0.390, -0.150, 0.0),
}

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "floor": (0.50, 0.51, 0.53),  # mid-grey vinyl: neutral against the beige table, the beech tops and blue chairs
    "wall": (0.93, 0.92, 0.88),  # off-white painted plaster
    "whiteboard": (0.98, 0.98, 0.98),
    "clock_rim": (0.05, 0.05, 0.05),
    "clock_face": (0.97, 0.97, 0.95),
    "beech": (0.85, 0.68, 0.42),  # beech-effect laminate desk top
    "steel": (0.30, 0.31, 0.33),  # grey powder-coated steel
    "chair_blue": (0.12, 0.32, 0.78),  # blue polypropylene
}
