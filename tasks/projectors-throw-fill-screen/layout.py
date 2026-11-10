"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the station's table top unless named
*_W. The classroom is at 1:10 scale.

At the far end of the table stands the front wall of a classroom with two projection screens side by side on it (each
160 mm wide: a 1.6 m screen). In front of the wall stands the projector table (classroom tables pushed together,
75 mm tall: 0.75 m), and on it two portable projectors, turned sideways with their lenses toward the middle, as they
were unpacked. Each must stand on the table facing its screen, square to the wall, its lens on the screen's centre
line, at the throw distance its own throw ratio needs for a picture as wide as the screen: throw distance = throw
ratio x picture width.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the room: fixed boxes ---------------------------------------------------------------------------------------------
WALL_X = (0.470, 0.480)  # the front wall, 10 mm thick
WALL_Y = (-0.250, 0.250)
WALL_H = 0.190
SCREEN_FACE_X = 0.468  # the screens' faces (the plane the throw distance is measured to)
SCREEN_T = 0.002  # each screen is a 2 mm sheet on the wall's face
SCREEN_X = (SCREEN_FACE_X, SCREEN_FACE_X + SCREEN_T)
SCREEN_W = 0.160  # 1.6 m wide at 1:10: the picture width that fills it
SCREEN_Z = (0.080, 0.180)  # 100 mm tall, its bottom 5 mm above the projector table's top
SCREENS = ("screen_left", "screen_right")
SCREEN_Y = {"screen_left": 0.120, "screen_right": -0.120}  # centre lines 240 mm apart; 80 mm of wall between them
MARK_W = 0.004  # a black centre mark on the wall below each screen's centre (a strip of tape)
MARK_Z = (0.015, 0.075)
TABLE_X = (0.170, 0.370)  # the projector table (classroom tables pushed together): 200 mm deep, 400 mm wide,
#                           75 mm tall (2.0 x 4.0 m, 0.75 m tall); its far edge 98 mm (0.98 m) from the screens
TABLE_Y = (-0.200, 0.200)
TABLE_H = 0.075
TABLE_TOP_W = TOP + TABLE_H  # world z of the projector table's top

# -- the projectors: two loose rigid bodies (mechkit compounds) ------------------------------------------------------
# Projector frame: origin at the middle of the body's bottom face, local +x (u) the lens's direction (the front),
# +y (v) the projector's own left, +z up.
BODY_D, BODY_W, BODY_H = 0.026, 0.034, 0.014  # body: 26 mm front to back, 34 mm across, 14 mm tall
LENS_OUT, LENS_W, LENS_H = 0.005, 0.014, 0.010  # the lens barrel: 5 mm proud of the front face, 14 mm across, 10 tall
LENS_Z = BODY_H / 2  # the lens's axis at the body's mid-height (7 mm up)
LENS_FRONT_U = BODY_D / 2 + LENS_OUT  # 0.018: the lens's front face, forward of the origin
PROJ_MASS = 0.050
PROJECTORS = ("projector_black", "projector_white")
RATIO = {"projector_black": 1.5, "projector_white": 1.0}  # throw ratio = throw distance / picture width
# (name, centre (u, v, z), size (u, v, z), colour key); the two shapes touch but do not overlap
SHAPES = (
    ("body", (0.0, 0.0, BODY_H / 2), (BODY_D, BODY_W, BODY_H), "body"),
    ("lens", (BODY_D / 2 + LENS_OUT / 2, 0.0, LENS_Z), (LENS_OUT, LENS_W, LENS_H), "lens"),
)
FOOT_FRONT, FOOT_BACK, FOOT_V = LENS_FRONT_U, BODY_D / 2, BODY_W / 2  # outline seen from above: 31 x 34 mm
TURN_RADIUS = math.hypot(FOOT_FRONT, FOOT_V)  # 25 mm: turns about the vertical counted at the outline's far corner

# Start: on the table, turned sideways, lenses toward the middle (as they were unpacked)
START = {
    "projector_black": (0.300, 0.150, -math.pi / 2),  # origin x, y, heading: lens toward -y
    "projector_white": (0.300, -0.150, math.pi / 2),  # lens toward +y
}

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
FULL_TOL = 0.05  # 'fills the screen': picture width within 5 % of 160 mm (152-168 mm)
LOOSE_TOL = 0.15  # partial credit: within 15 % (136-184 mm)
SQUARE_DEG = 5.0  # 'square to the wall': the lens's axis within 5 degrees of straight at the wall (+x)
AIM_DEG = 10.0  # 'facing its screen' at all: within 10 degrees
CENTRE_TIGHT = 0.006  # 'on the centre line': the lens's centre within 6 mm of the screen's centre line (sideways)
CENTRE_LOOSE = 0.012  # 'at' a screen at all: within 12 mm
TILT_MAX_DEG = 5.0  # 'standing': upright within 5 degrees ...
TABLE_TOL = 0.003  # ... its lowest point within 3 mm of the projector table's top, its origin over the table ...
STILL_WINDOW_S = 0.5  # ... and still: its origin's path over the last 0.5 s, turns counted at TURN_RADIUS, under
STILL_PATH = 0.005  # 5 mm (positions, not the simulator's velocities, which chatter while a body is gripped)
# ... and let go: each gripper (read through rigstate.py) is at least 90 % open, or its finger zone is more than
# 10 mm from the projector. The finger zone is every point within a radius of the segment from the point between
# its finger tips 38 mm back up the gripper; the radius reaches a pad's outer corner (half the finger gap plus the
# 8 mm pad, and half the pad's 27 mm width across): 15.7 mm closed, 56.6 mm fully open. rigstate gives no gripper
# heading, so the zone is round. Arms that cannot be read: no projector counts as let go.
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
LET_GO_GAP = 0.010
FULL_GAP = 0.0939  # the pads' gap fully open (rigstate opening 1.0)
FINGER_THICK = 0.008
FINGER_HALF_WIDTH = 0.0135
FINGER_LEN = 0.038
ZONE_SAMPLES = 39  # points along the 38 mm segment, 1 mm apart
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a projector moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)
EPS = 1e-9  # float slack on the bands' edges

# -- where the demo puts them (the scorer does not use these) ----------------------------------------------------------


def throw_for(name: str) -> float:
    """The throw distance (lens to screen) that fills a 160 mm screen with this projector's picture (m)."""
    return RATIO[name] * SCREEN_W


TARGET = {  # origin x, y, heading: the lens's front face at the throw distance, its centre on the centre line
    "projector_black": (SCREEN_FACE_X - throw_for("projector_black") - LENS_FRONT_U, SCREEN_Y["screen_left"], 0.0),
    "projector_white": (SCREEN_FACE_X - throw_for("projector_white") - LENS_FRONT_U, SCREEN_Y["screen_right"], 0.0),
}  # black (0.210, +0.120): lens 240 mm from the left screen; white (0.290, -0.120): lens 160 mm from the right

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "wall": (0.60, 0.68, 0.60),  # pale green painted plaster
    "screen": (0.96, 0.96, 0.95),  # matt white
    "mark": (0.05, 0.05, 0.05),  # black tape
    "table": (0.70, 0.55, 0.38),  # beech-effect laminate
    "black": (0.08, 0.08, 0.09),
    "white": (0.92, 0.92, 0.90),
    "silver": (0.76, 0.77, 0.80),
    "lens_dark": (0.06, 0.06, 0.07),
}
BODY_COLOUR = {"projector_black": "black", "projector_white": "white"}
LENS_COLOUR = {"projector_black": "silver", "projector_white": "lens_dark"}  # each lens contrasts with its body
