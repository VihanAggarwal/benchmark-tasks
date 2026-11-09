"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475 and stands for the floor of the room. Heights below are
above the table top unless named *_W. The room is at 1:8 scale.

An armchair is fixed across the far side of the table, facing the arms. Mr Okafor, a rigid seated manikin, sits in
it facing the arms (facing -x), his feet flat on the floor, his toes on the line x 0.350, his centre line on y 0.
His walking frame, one loose rigid body (a mechkit compound), stands side-on to his left front (the robot's right),
out of his reach.
"""

import math

TOP = 0.7475  # station table top (world z): the floor of the room

# -- the armchair: fixed boxes, green upholstery (x0, x1), (y0, y1), (z0, z1) -------------------------------------------
CHAIR_BOXES = (
    ("chair_seat", (0.395, 0.455), (-0.034, 0.034), (0.000, 0.055)),
    ("chair_back", (0.455, 0.470), (-0.044, 0.044), (0.000, 0.110)),
    ("chair_arm_right", (0.395, 0.455), (0.034, 0.044), (0.000, 0.075)),  # his right: +y (he faces -x)
    ("chair_arm_left", (0.395, 0.455), (-0.044, -0.034), (0.000, 0.075)),
)

# -- Mr Okafor: fixed boxes and a sphere, sitting in the chair facing the arms ----------------------------------------
Y_C = 0.000  # his centre line
TOE_X = 0.350  # the line of his toes (the front of his slippers): distances are measured from it
RESIDENT_BOXES = (  # name, (x0, x1), (y0, y1), (z0, z1), colour key
    ("right_slipper", (0.350, 0.385), (0.003, 0.018), (0.000, 0.009), "slipper"),  # his right (+y): left and
    ("left_slipper", (0.350, 0.385), (-0.018, -0.003), (0.000, 0.009), "slipper"),  # right are his own
    ("right_shin", (0.373, 0.385), (0.004, 0.017), (0.009, 0.055), "trousers"),
    ("left_shin", (0.373, 0.385), (-0.017, -0.004), (0.009, 0.055), "trousers"),
    ("thighs", (0.373, 0.430), (-0.019, 0.019), (0.055, 0.069), "trousers"),
    ("torso", (0.430, 0.455), (-0.022, 0.022), (0.055, 0.115), "shirt"),
    ("right_upper_arm", (0.443, 0.455), (0.022, 0.034), (0.083, 0.112), "shirt"),
    ("left_upper_arm", (0.443, 0.455), (-0.034, -0.022), (0.083, 0.112), "shirt"),
    ("right_forearm", (0.405, 0.450), (0.035, 0.043), (0.075, 0.083), "skin"),  # resting on the chair's arms
    ("left_forearm", (0.405, 0.450), (-0.043, -0.035), (0.075, 0.083), "skin"),
)
HEAD_CENTER, HEAD_RADIUS = (0.442, Y_C, 0.128), 0.013

# -- the walking frame: one loose rigid body. Its frame: origin at the middle of its footprint on the floor, local +x
#    (u) forward, from its open side to its crossbar (the way he walks in it), local +y (v) across, +z up.
#    Footprint 56 (u) x 70 (v) mm, 106 mm tall. --------------------------------------------------------------------
FRAME_MASS = 0.060
FOOT_U, FOOT_V = 0.028, 0.035  # footprint half-depth (u) and half-width (v): the outer faces of the four legs
FRAME_H = 0.106
TUBE = 0.008  # every tube is 8 x 8 mm
LEG_U, LEG_V = FOOT_U - TUBE / 2, FOOT_V - TUBE / 2  # leg centres (+-0.024, +-0.031)
FERRULE_H = 0.010
# (name, centre (u, v, z), size (u, v, z), colour key); no two shapes overlap (mechkit spreads the mass by volume)
FRAME_SHAPES = []
for _su, _fb in ((1, "front"), (-1, "back")):
    for _sv, _lr in ((1, "left"), (-1, "right")):
        FRAME_SHAPES.append((f"ferrule_{_fb}_{_lr}", (_su * LEG_U, _sv * LEG_V, FERRULE_H / 2),
                             (TUBE, TUBE, FERRULE_H), "rubber"))
        FRAME_SHAPES.append((f"leg_{_fb}_{_lr}", (_su * LEG_U, _sv * LEG_V, (FERRULE_H + FRAME_H - TUBE) / 2),
                             (TUBE, TUBE, FRAME_H - TUBE - FERRULE_H), "alu"))  # z 0.010-0.098
for _sv, _lr in ((1, "left"), (-1, "right")):
    _z = FRAME_H - TUBE / 2  # the top rails, z 0.098-0.106, over the legs: aluminium ends, black handgrip between
    FRAME_SHAPES.append((f"top_front_{_lr}", (0.022, _sv * LEG_V, _z), (0.012, TUBE, TUBE), "alu"))  # u 0.016-0.028
    FRAME_SHAPES.append((f"handgrip_{_lr}", (0.0, _sv * LEG_V, _z), (0.032, TUBE, TUBE), "grip"))  # u -0.016-0.016
    FRAME_SHAPES.append((f"top_back_{_lr}", (-0.022, _sv * LEG_V, _z), (0.012, TUBE, TUBE), "alu"))
    FRAME_SHAPES.append((f"side_rail_{_lr}", (0.0, _sv * LEG_V, 0.054), (2 * (LEG_U - TUBE / 2), TUBE, TUBE),
                         "alu"))  # between the legs, u -0.020..0.020, z 0.050-0.058
FRAME_SHAPES.append(("crossbar", (LEG_U, 0.0, 0.074), (TUBE, 2 * (LEG_V - TUBE / 2), TUBE), "alu"))  # z 0.070-0.078
FRAME_SHAPES = tuple(FRAME_SHAPES)
GRIP_U = 0.002  # the grip: across the two handgrips (70 mm outside), at the frame's centre of mass along its depth
FRAME_START = (0.300, -0.170)  # footprint centre: to his left front, side-on, 170 mm from his centre line
FRAME_START_HEADING = math.radians(-90.0)  # its crossbar side toward -y (away from the midline): side-on to him

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
# Three conditions, each with a full band and a wider loose band; outside the loose band the score is 0.
SQUARE_FULL_DEG, SQUARE_LOOSE_DEG = 15.0, 30.0  # its forward direction (crossbar side) within this of -x
CENTRE_FULL, CENTRE_LOOSE = 0.020, 0.050  # its footprint centre within this of his centre line (y 0)
DIST_FULL = (0.010, 0.050)  # toe gap d: from the toe line to the frame's nearest leg, along x (8-40 cm at 1:8)
DIST_LOOSE = (0.000, 0.100)  # d below 0 (a leg past the toe line: around his feet) or over 100 mm: 0
SCORE_BY_LOOSE = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.0}  # by how many conditions are only in their loose band
FLOOR_TOL = 0.003  # 'on its four feet': the bottom of each foot within 3 mm of the floor
STILL_WINDOW_S = 0.5  # 'still': its centre moved less than STILL_PATH over the last 0.5 s (path length, turns
STILL_PATH = 0.005  # counted at the footprint's corner radius), so gripped-velocity chatter cannot matter
TURN_RADIUS = math.hypot(FOOT_U, FOOT_V)  # 45 mm
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the frame moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def heading_of(turn_deg: float) -> float:
    """The frame's heading (rad, from +x counter-clockwise) for a frame turned ``turn_deg`` from square (its
    forward direction at -x, toward the arms, open side toward him)."""
    return math.radians(180.0 + turn_deg)


def corner_offsets(heading: float) -> list:
    """World (dx, dy) of the footprint's four corners from its centre for a frame facing ``heading`` (rad):
    front-left, front-right, back-left, back-right (front = its crossbar side)."""
    c, s = math.cos(heading), math.sin(heading)
    out = []
    for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        u, v = su * FOOT_U, sv * FOOT_V
        out.append((u * c - v * s, u * s + v * c))
    return out


# -- where the demo puts it (the scorer does not use these) ------------------------------------------------------------
TARGET_GAP = 0.030  # 30 mm from his toes (24 cm at scale): the middle of the 10-50 mm band
TARGET_CENTER = (TOE_X - TARGET_GAP - FOOT_U, Y_C)  # (0.292, 0.000)
TARGET_HEADING = heading_of(0.0)  # 180 deg: square, crossbar toward the arms, open side toward him

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "upholstery": (0.22, 0.42, 0.30),  # green
    "skin": (0.55, 0.38, 0.27),  # brown PLA manikin
    "shirt": (0.85, 0.80, 0.62),  # beige
    "trousers": (0.30, 0.30, 0.36),  # charcoal
    "slipper": (0.48, 0.12, 0.12),  # maroon
    "alu": (0.74, 0.76, 0.79),  # anodised aluminium
    "grip": (0.10, 0.10, 0.11),  # black foam handgrips
    "rubber": (0.36, 0.36, 0.38),  # grey rubber ferrules
}
