"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475 and stands for the floor of the room. Heights below are
above the table top unless named *_W. The room is at 1:10 scale.

A bed is fixed across the far side of the table, its long side along y, its head end at +y. Mrs Hale, a rigid seated
manikin, sits on the bed's near edge facing the arms (facing -x), so her right side is +y, the robot's left. Her left
side is the weak one (a sling on her left forearm). The wheelchair, one loose rigid body (a mechkit compound), starts
past the foot of the bed on her weak side, angled toward the arms.
"""

import math

TOP = 0.7475  # station table top (world z): the floor of the room

# -- the bed: one fixed box, mattress top 48 mm up ---------------------------------------------------------------------
BED_X0, BED_X1 = 0.400, 0.490  # near (side) face at x 0.400: the line the gap is measured from
BED_Y0, BED_Y1 = 0.000, 0.240  # foot end at y 0, head end at y 0.240
BED_H = 0.048
PILLOW = (0.415, 0.475, 0.195, 0.235, BED_H, BED_H + 0.008)  # x0, x1, y0, y1, z0, z1: fixed, at the head end

# -- Mrs Hale: fixed boxes and a sphere, sitting on the near edge, her centre line at y 0.100 -----------------------
Y_P = 0.100  # her centre line
RESIDENT_BOXES = (  # name, (x0, x1), (y0, y1), (z0, z1), colour key
    ("thighs", (0.390, 0.445), (0.080, 0.120), (0.048, 0.063), "skin"),
    ("torso", (0.430, 0.455), (0.080, 0.120), (0.063, 0.113), "skin"),
    ("lower_legs", (0.388, 0.400), (0.082, 0.118), (0.000, 0.048), "skin"),
    ("slippers", (0.363, 0.388), (0.082, 0.118), (0.000, 0.008), "slipper"),
    ("right_arm", (0.425, 0.455), (0.120, 0.128), (0.065, 0.105), "skin"),  # her strong side (+y)
    ("left_forearm_sling", (0.418, 0.430), (0.074, 0.102), (0.078, 0.092), "sling"),  # her weak side
)
HEAD_CENTER, HEAD_RADIUS = (0.442, Y_P, 0.124), 0.011

# -- the wheelchair: one loose rigid body. Its frame: origin at the middle of its footprint on the floor, local +x
#    (u) forward toward the footplates, local +y (v) to the chair's own left, +z up. Footprint 95 (u) x 63 (v) mm. ----
CHAIR_MASS = 0.120
FOOT_U, FOOT_V = 0.0475, 0.0315  # footprint half-length (u) and half-width (v)
CHAIR_H = 0.085  # backrest top
# (name, centre (u, v, z), size (u, v, z), colour key); no two shapes overlap (mechkit spreads the mass by volume)
CHAIR_SHAPES = (
    ("wheel_left", (-0.0175, 0.0305, 0.030), (0.060, 0.002, 0.060), "tyre"),  # rear wheels: square plates, the grip
    ("wheel_right", (-0.0175, -0.0305, 0.030), (0.060, 0.002, 0.060), "tyre"),
    ("seat", (-0.0175, 0.0, 0.046), (0.050, 0.059, 0.008), "vinyl"),  # u -0.0425..0.0075, z 0.042-0.050
    ("backrest", (-0.045, 0.0, 0.0675), (0.005, 0.059, 0.035), "vinyl"),  # at the rear, z 0.050-0.085
    ("rail_left", (0.02625, 0.0295, 0.039), (0.0275, 0.004, 0.006), "frame"),  # u 0.0125-0.040, z 0.036-0.042
    ("rail_right", (0.02625, -0.0295, 0.039), (0.0275, 0.004, 0.006), "frame"),
    ("leg_left", (0.043, 0.0295, 0.027), (0.006, 0.004, 0.030), "frame"),  # front posts, z 0.012-0.042
    ("leg_right", (0.043, -0.0295, 0.027), (0.006, 0.004, 0.030), "frame"),
    ("castor_left", (0.0425, 0.0285, 0.006), (0.010, 0.006, 0.012), "tyre"),  # u 0.0375-0.0475, z 0-0.012
    ("castor_right", (0.0425, -0.0285, 0.006), (0.010, 0.006, 0.012), "tyre"),
    ("footplate_left", (0.036, 0.0145, 0.010), (0.016, 0.022, 0.004), "frame"),  # u 0.028-0.044, v 0.0035-0.0255
    ("footplate_right", (0.036, -0.0145, 0.010), (0.016, 0.022, 0.004), "frame"),  # z 0.008-0.012
)
GRIP_U = -0.010  # the grip: across the two wheel plates (63 mm outside), 32.5 mm in front of the backrest
CHAIR_START = (0.300, -0.200)  # footprint centre: past the foot of the bed, on her weak side
CHAIR_START_HEADING = math.radians(135.0)  # its front (footplates) toward (-0.71, +0.71): toward the arms and +y

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
Y_STRONG = Y_P + 0.030  # 0.130: the line of her right hip and right arm. Every footprint corner must be at or past it
Y_BESIDE = Y_STRONG + 0.040  # 0.170: ... and the nearest corner no further along the bed than this (beside her: 40 mm,
#                             0.4 m at scale). For any angle 0-90 the centre is then at most 57 mm further, short of
#                             the bed's head end, so no separate head-end rule is needed
BED_SIDE_X = BED_X0  # the gap g = BED_SIDE_X minus the footprint's largest x
HEAD_END_Y = BED_Y1  # the bed's head end (the chair, held beside her by Y_BESIDE, never reaches it)
BED_LINE_TOL = 0.003  # a corner more than 3 mm past the bed's side line (into the bed: a broken state): 0
GAP_CLOSE = 0.020  # closeness 0.5: g <= 20 mm (0.2 m at 1:10)
GAP_MAX = 0.080  # g over 80 mm (0.8 m): 0
ANGLE_FULL_DEG = (30.0, 50.0)  # angle 0.5
ANGLE_PART_DEG = (15.0, 65.0)  # angle 0.25 outside the full band but inside this one
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 0.0, 90.0  # below 0 (turned in to the bed) or above 90 (facing the head end): 0
TILT_MAX_DEG = 10.0  # 'upright'
FLOOR_TOL = 0.003  # 'on the floor': its lowest point within 3 mm of the floor
STILL_WINDOW_S = 0.5  # 'still': its centre moved less than STILL_PATH over the last 0.5 s (path length, turns
STILL_PATH = 0.005  # counted at the footprint's corner radius), so gripped-velocity chatter cannot matter
TURN_RADIUS = math.hypot(FOOT_U, FOOT_V)  # 57 mm
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the chair moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def corner_offsets(alpha_deg: float) -> dict:
    """World (dx, dy) of the footprint's corners from its centre for a chair at angle alpha (degrees) to the bed:
    its front f = (-sin a, -cos a), its left v = (cos a, -sin a). Keys: front_left, front_right, rear_left (the
    bed-side corner for 0..90 degrees), rear_right."""
    a = math.radians(alpha_deg)
    f = (-math.sin(a), -math.cos(a))
    v = (math.cos(a), -math.sin(a))
    out = {}
    for name, su, sv in (("front_left", 1, 1), ("front_right", 1, -1), ("rear_left", -1, 1), ("rear_right", -1, -1)):
        out[name] = (su * FOOT_U * f[0] + sv * FOOT_V * v[0], su * FOOT_U * f[1] + sv * FOOT_V * v[1])
    return out


def heading_of(alpha_deg: float) -> float:
    """The chair's heading (rad, from +x counter-clockwise) for an angle alpha (degrees) to the bed."""
    return math.radians(270.0 - alpha_deg)


# -- where the demo puts it (the scorer does not use these) ------------------------------------------------------------
TARGET_ALPHA_DEG = 40.0  # the middle of the 30-50 band
TARGET_GAP = 0.010  # 10 mm from the bed's side: 10 mm inside the 20 mm band, 10 mm from touching the bed
_C = corner_offsets(TARGET_ALPHA_DEG)
TARGET_CENTER = (BED_SIDE_X - TARGET_GAP - max(dx for dx, _ in _C.values()), 0.205)  # (0.3353, 0.2050): its
#   nearest corner at y 0.1483, 18 mm past the strong-side line and 22 mm short of Y_BESIDE
HANDOVER_GRIP = (0.2575, 0.000)  # the relay point on the midline; the chair faces the arms there (heading 180 deg)

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "bed": (0.93, 0.93, 0.91),  # white-painted wood
    "pillow": (0.98, 0.98, 0.98),
    "skin": (0.82, 0.66, 0.52),  # tan PLA manikin
    "slipper": (0.20, 0.20, 0.22),  # dark grey
    "sling": (0.20, 0.42, 0.80),  # blue
    "tyre": (0.12, 0.12, 0.13),  # dark grey rubber
    "vinyl": (0.10, 0.17, 0.42),  # navy vinyl
    "frame": (0.78, 0.79, 0.81),  # chromed steel
}
