"""Every dimension and threshold of the task, in one place (metres, kilograms, radians unless a name says _MM or _DEG),
station frame: x forward from the line through the two arm bases, y to the left, z up; the station's table top is at
z 0.7475 and every height below is above it.

The scene: a loose spring cord lock (a Ravenox Extra Large cord lock: 41 x 20 x 13 mm with its button out, a
0.65 x 0.45 in. hole, black Delrin, stainless spring) lying on one 34 x 13 mm long side, and 320 mm of loose 5/16 in.
shock cord lying straight along y in front of it.

The lock's frame (the assembly's base frame): origin on the table under the body's centre; X along the lock's length
toward the button, Y along the hole's axis (the cord's direction through it), Z up across the lock's 20 mm width. It
lies with Z up and yaw 0, so its axes are the world's. The body is a box 34 x 13 x 20 mm (X, Y, Z) with two 2 mm
faces (Y -6.5 to -4.5 and 4.5 to 6.5) holding the catalogue hole, 2 mm bottom and top walls, a closed end at -X and
an open end at +X (9 x 16 mm inside). The plunger (link ``plunger``, a slider along -X, 0 to 7 mm: positive is
pressed in) fills the open end; its last 7 mm stands out of the body as the button. The passage through the lock is
the overlap of the face hole (X -7.7 to 3.7, Z 1.75 to 18.25) and the plunger's own hole (X -0.7 - q to 10.7 - q,
Z 3.5 to 16.5): 4.4 mm + q wide along X and 13 mm along Z.

The cord's ends: end 0 (the cablekit cable's start) lies at y -0.060 pointing to +y, toward the lock; end 1 at y
-0.380. Either end may be threaded.
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
MM = 0.001

# -- the cord lock (loose articulation) ------------------------------------------------------------------------------
DELRIN = 1410.0  # kg/m3, acetal
BODY_LEN = 0.034  # X, without the button
DEPTH = 0.013  # Y: the lock's thickness, along the hole's axis
WIDTH = 0.020  # Z
FACE_T = 0.002  # each face (Y)
WALL_T = 0.002  # bottom, top and closed-end walls
HALF_DEPTH = DEPTH / 2  # faces' outer planes at Y = +-6.5 mm
FACE_HOLE_X = (-0.0077, 0.0037)  # the catalogue hole (11.4 x 16.5 mm), its centre 15 mm from the closed end
FACE_HOLE_Z = (0.00175, 0.01825)
HOLE_CX, HOLE_CZ = -0.002, 0.010
PLUNGER_HOLE_X = (-0.0007, 0.0107)  # at rest (q = 0); it moves -X with the travel
PLUNGER_HOLE_Z = (0.0035, 0.0165)  # 13 mm
TRAVEL = 0.007  # free position to flush; the body's end face is the stop
BUTTON = 0.007  # how far the button stands out of the body at rest
SPRING = 1500.0  # N/m from the free position (no preload): 10.5 N at flush
PLUNGER_DAMPING = 1.0  # N s/m
LOCK_FRICTION = (0.4, 0.4)
BODY_MASS = 0.0049  # 3,480 mm3 of Delrin
PLUNGER_MASS = 0.0039  # 2,770 mm3
LOCK_MASS = BODY_MASS + PLUNGER_MASS  # 0.0088
LOCK_ORIGIN = (0.302, 0.130)  # body centre on the table; the button toward +x, the hole's axis along y
LOCK_LEN_REST = BODY_LEN + BUTTON  # 0.041
BLACK = (0.05, 0.05, 0.05)
BUTTON_BLACK = (0.09, 0.09, 0.09)  # the same black plastic; a shade lighter so the button reads in a render

# Body shapes (centre, size), base frame, metres: (name, centre, size)
BODY_SHAPES = (
    ("near_face_closed_side", (-0.01235, -0.0055, 0.010), (0.0093, 0.002, 0.020)),
    ("near_face_button_side", (0.01035, -0.0055, 0.010), (0.0133, 0.002, 0.020)),
    ("near_face_below_hole", (-0.002, -0.0055, 0.000875), (0.0114, 0.002, 0.00175)),
    ("near_face_above_hole", (-0.002, -0.0055, 0.019125), (0.0114, 0.002, 0.00175)),
    ("far_face_closed_side", (-0.01235, 0.0055, 0.010), (0.0093, 0.002, 0.020)),
    ("far_face_button_side", (0.01035, 0.0055, 0.010), (0.0133, 0.002, 0.020)),
    ("far_face_below_hole", (-0.002, 0.0055, 0.000875), (0.0114, 0.002, 0.00175)),
    ("far_face_above_hole", (-0.002, 0.0055, 0.019125), (0.0114, 0.002, 0.00175)),
    ("bottom_wall", (0.0, 0.0, 0.001), (0.034, 0.009, 0.002)),
    ("top_wall", (0.0, 0.0, 0.019), (0.034, 0.009, 0.002)),
    ("closed_end", (-0.016, 0.0, 0.010), (0.002, 0.009, 0.016)),
)
PLUNGER_SHAPES = (
    ("web_closed_side", (-0.0026, 0.0, 0.010), (0.0038, 0.009, 0.016)),
    ("web_button_side", (0.01385, 0.0, 0.010), (0.0063, 0.009, 0.016)),
    ("rail_below_hole", (0.005, 0.0, 0.00275), (0.0114, 0.009, 0.0015)),
    ("rail_above_hole", (0.005, 0.0, 0.01725), (0.0114, 0.009, 0.0015)),
    ("button", (0.0205, 0.0, 0.010), (0.007, 0.009, 0.016)),
)
LOCK_EXTENT_X = (-0.017, 0.024)  # at rest (the button end moves to 0.017 at flush)


def passage_x(q: float) -> tuple[float, float]:
    """The passage along X (base frame) at travel ``q``: the overlap of the face hole and the plunger's hole."""
    return (max(FACE_HOLE_X[0], PLUNGER_HOLE_X[0] - q), min(FACE_HOLE_X[1], PLUNGER_HOLE_X[1] - q))


def passage_width(q: float) -> float:
    lo, hi = passage_x(q)
    return max(0.0, hi - lo)


# -- the shock cord (loose cablekit cable) ----------------------------------------------------------------------------
CORD_D = 0.0079  # 5/16 in.
CORD_R = CORD_D / 2
CORD_LEN = 0.320
CORD_SEGMENT = 0.020  # 16 capsules
CORD_MASS = 0.0147  # 46 g/m
CORD_BEND = 0.05  # N m/rad per joint: EI 1e-3 N m2 over 20 mm
CORD_BEND_DAMPING = 0.002
CORD_SIDES = 16  # the collider's section: a 16-sided prism, flats at 3.95 mm, corners at 4.03 mm
CORD_ARMATURE = 5e-6
CORD_MAX_BEND = 1.2
CORD_FRICTION = 0.6
CORD_X = 0.300
CORD_END0_Y = -0.060  # end 0, pointing to +y
CORD_SPAWN_LIFT = 0.0005  # spawned 0.5 mm above the table: it settles onto a flat
WHITE = (0.93, 0.93, 0.90)
RING_MM = (5.0, 23.0, 53.0)  # pen rings from each end, for reading a real trial: 5 mm; 13 + 10 mm; 13 + 40 mm

# Cord pinched in the lock: the spring brings the plunger back until the passage is the cord's width
PINCH_Q = CORD_D - passage_width(0.0)  # 3.5 mm

# -- the scorer ----------------------------------------------------------------------------------------------------
IN_MIN = 0.005  # 0.25: a cord end at least 5 mm into the passage
OUT_HALF = 0.010  # 0.5: through, at least 10 mm out on both sides (along the cord)
OUT_FULL = 0.040  # 0.75 and 1.0: at least 40 mm out on both sides
PLUNGER_HELD = 0.001  # 1.0: the plunger held at least 1 mm in (the button at most 6 mm out)
PRESSED = 0.005  # every tier: the button pressed at least 5 mm in (at most 2 mm out) before a cord end is 5 mm
# in, and held that far in on every step from then until the cord is through; a straight cord end can push the
# plunger only PINCH_Q (3.5 mm) in, so wedging the end in never counts as a press
SUPPORT = 0.002  # 1.0: the lock's lowest point within 2 mm of the table top
LET_GO = 0.030  # every finger zone at least 30 mm from the lock and from the cord
STILL_WINDOW_S = 1.0  # 20 steps
STILL_MM = 0.002  # no lock corner or cord point moves more than 2 mm in that second
EPS = 1e-9
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): the lock put back at its start (within
# 5 mm) in one control step after jumping 50 mm or more.
RESET_START_TOL = 0.005
RESET_JUMP = 0.050
