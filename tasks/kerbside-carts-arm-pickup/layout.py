"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W. The
street is at 1:10 scale.

The table top is the verge (the strip of pavement in front of the house). Across the far side runs the street: a
raised kerb whose face is at x 0.390, and the road beyond it. On the verge at the kerb stand a mailbox on its post
(on the midline) and a fire hydrant (to the left); a car is parked in the road at the kerb to the right, and to the
left of the hydrant the house's driveway crosses the verge (the kerb is dropped there). The two wheeled carts start
near the house, side-on, their fronts toward each other.
"""

import math

TOP = 0.7475  # station table top (world z): the verge

# -- the street: fixed boxes -------------------------------------------------------------------------------------------
KERB_FACE_X = 0.390  # the kerb's verge-side face: the line the set-back is measured from
KERB_X = (KERB_FACE_X, 0.400)  # 10 mm wide
KERB_H = 0.012  # 12 mm tall (0.12 m)
KERB_Y = (-0.640, 0.360)  # the raised kerb; at the driveway (y over 0.360) it is dropped
ROAD_X = (0.400, 0.600)
ROAD_Y = (-0.640, 0.640)
ROAD_H = 0.001  # a 1 mm asphalt sheet
DRIVE_X = (0.240, 0.400)  # the driveway: a 1 mm concrete sheet across the verge to the road
DRIVE_Y = (0.360, 0.640)
DRIVE_H = 0.001

# -- street furniture (fixed) ------------------------------------------------------------------------------------------
POST_XY = (0.370, 0.000)  # the mailbox post: a steel cylinder, 16 mm across, 105 mm tall
POST_R, POST_H = 0.008, 0.105
MAILBOX = (0.350, 0.390, -0.008, 0.008, 0.105, 0.125)  # x0, x1, y0, y1, z0, z1: a black box on the post, door to the road
HYDRANT_XY = (0.365, 0.300)  # the fire hydrant: a red cylinder 24 mm across, 55 mm tall, with a bonnet on top
HYDRANT_R, HYDRANT_H = 0.012, 0.055
HYDRANT_CAP_R, HYDRANT_CAP_H = 0.009, 0.010  # the bonnet: a cylinder 18 mm across, 10 mm tall (65 mm in all)
CAR_BODY = (0.410, 0.590, -0.640, -0.300, ROAD_H, 0.081)  # parked 10 mm (0.1 m) off the kerb, its near end at y -0.300
CAR_CABIN = (0.420, 0.580, -0.570, -0.370, 0.081, 0.141)

# Top-view outlines the clearance is measured to (scoring.py): the mailbox with its post (the box covers the post's
# circle), the hydrant's circle, the car's body.
MAILBOX_RECT = (MAILBOX[0], MAILBOX[1], MAILBOX[2], MAILBOX[3])
CAR_RECT = (CAR_BODY[0], CAR_BODY[1], CAR_BODY[2], CAR_BODY[3])
DRIVE_RECT = (DRIVE_X[0], DRIVE_X[1], DRIVE_Y[0], DRIVE_Y[1])

# -- the carts: two loose rigid bodies (mechkit compounds), 1:10 models of a 64-gallon (240 L) automated-collection
#    cart. Cart frame: origin at the middle of the body's bottom face, local +x (u) the front (the side the lid opens
#    on, where the lid arrow points), +y (v) the cart's own left, +z up. ------------------------------------------------
CART_MASS = 0.120
BODY_U, BODY_V, BODY_H = 0.0365, 0.029, 0.095  # body half-depth, half-width; height
LID_T = 0.006  # lid on top of the body, flush with it on every side: top 101 mm up
BACK_U = 0.0425  # the handle bar and the wheels stand 6 mm proud of the body's back face
CART_H = BODY_H + LID_T  # 0.101
ARROW_T = 0.001
# (name, centre (u, v, z), size (u, v, z), colour key); no two shapes overlap (mechkit spreads the mass by volume)
CART_SHAPES = (
    ("body", (0.0, 0.0, BODY_H / 2), (2 * BODY_U, 2 * BODY_V, BODY_H), "body"),
    ("lid", (0.0, 0.0, BODY_H + LID_T / 2), (2 * BODY_U, 2 * BODY_V, LID_T), "lid"),
    ("handle", (-(BODY_U + BACK_U) / 2, 0.0, 0.090), (BACK_U - BODY_U, 2 * BODY_V, 0.010), "black"),  # z 85-95
    ("wheel_left", (-(BODY_U + BACK_U) / 2, 0.020, 0.012), (BACK_U - BODY_U, 0.016, 0.024), "black"),  # v 12-28
    ("wheel_right", (-(BODY_U + BACK_U) / 2, -0.020, 0.012), (BACK_U - BODY_U, 0.016, 0.024), "black"),
    # the white arrow on the lid, pointing to the front: a shaft and a stepped head
    ("arrow_shaft", (-0.003, 0.0, CART_H + ARROW_T / 2), (0.022, 0.006, ARROW_T), "white"),  # u -0.014..0.008
    ("arrow_head_1", (0.011, 0.0, CART_H + ARROW_T / 2), (0.006, 0.020, ARROW_T), "white"),  # u 0.008..0.014
    ("arrow_head_2", (0.017, 0.0, CART_H + ARROW_T / 2), (0.006, 0.012, ARROW_T), "white"),  # u 0.014..0.020
    ("arrow_head_3", (0.0225, 0.0, CART_H + ARROW_T / 2), (0.005, 0.004, ARROW_T), "white"),  # u 0.020..0.025
)
# The footprint the scorer uses: the cart's outline seen from above, 79 mm front to back (handle and wheels
# included) and 58 mm across. For the shadow of a tilted cart, the box it spans up to the lid's top.
FOOT_FRONT, FOOT_BACK, FOOT_V = BODY_U, BACK_U, BODY_V
OUTLINE_Z = CART_H
TURN_RADIUS = math.hypot(FOOT_BACK, FOOT_V)  # 51 mm: turns about the vertical counted at the footprint's corner

# Start: near the house, side-on, fronts toward each other (as they were wheeled out of the garage)
CARTS = ("cart_left", "cart_right")
START = {
    "cart_left": (0.220, 0.130, -math.pi / 2),  # origin x, y, heading: front toward -y
    "cart_right": (0.220, -0.130, math.pi / 2),  # front toward +y
}

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
CLEAR = 0.091  # 3 ft (0.91 m) at 1:10: from the other cart, the mailbox and its post, the hydrant and the car
SETBACK_MAX = 0.030  # 'at the kerb': the footprint's most forward point 0-30 mm (0-0.3 m) behind the kerb face
KERB_TOL = 0.002  # a footprint more than 2 mm past the kerb face is on the kerb or in the road: the trial scores 0
FACING_DEG = 25.0  # 'facing the street': its front within 25 degrees of +x (square to the kerb, toward the road)
TILT_STAND_DEG = 10.0  # 'standing': upright within 10 degrees ...
FLOOR_TOL = 0.003  # ... its lowest point within 3 mm of the verge ...
STILL_WINDOW_S = 0.5  # ... and still: its origin's path over the last 0.5 s, turns counted at TURN_RADIUS, under
STILL_PATH = 0.005  # 5 mm (positions, not the simulator's velocities, which chatter while a body is gripped)
TIPPED_DEG = 45.0  # a cart tilted more than this is knocked over: the trial scores 0
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a cart moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- where the demo puts them (the scorer does not use these) ----------------------------------------------------------
TARGET_SETBACK = 0.012  # front 12 mm behind the kerb face: 12 mm inside the band's near end, 18 mm from its far end
TARGET_X = KERB_FACE_X - TARGET_SETBACK - FOOT_FRONT  # 0.3415: the origin (the body's middle)
TARGET = {
    "cart_left": (TARGET_X, 0.150, 0.0),  # between the mailbox and the hydrant, facing the street
    "cart_right": (TARGET_X, -0.150, 0.0),  # between the mailbox and the car
}


def stretch(side: str) -> tuple[float, float]:
    """The range of origin y over which a cart facing the street with its front TARGET_SETBACK behind the kerb is
    clear of the street furniture on that side of the mailbox (left: mailbox and hydrant; right: mailbox and car)."""
    near = MAILBOX[3] + CLEAR + FOOT_V  # 0.128 from the mailbox's side face
    if side == "left":
        return near, HYDRANT_XY[1] - HYDRANT_R - CLEAR - FOOT_V  # 0.128 .. 0.168
    front = KERB_FACE_X - TARGET_SETBACK
    dx = CAR_RECT[0] - front
    return CAR_RECT[3] + math.sqrt(CLEAR ** 2 - dx ** 2) + FOOT_V, -near  # -0.1858 .. -0.128


# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "kerb": (0.72, 0.72, 0.70),  # concrete
    "road": (0.17, 0.17, 0.18),  # asphalt
    "drive": (0.83, 0.82, 0.78),  # pale concrete
    "post": (0.62, 0.63, 0.65),  # galvanised steel
    "mailbox": (0.08, 0.08, 0.09),  # black
    "hydrant": (0.80, 0.10, 0.08),  # red
    "car": (0.10, 0.20, 0.45),  # dark blue
    "car_glass": (0.30, 0.38, 0.48),
    "body": (0.22, 0.23, 0.24),  # dark grey HDPE
    "lid_left": (0.15, 0.55, 0.20),  # green lid (organics)
    "lid_right": (0.12, 0.35, 0.80),  # blue lid (recycling)
    "black": (0.06, 0.06, 0.06),
    "white": (0.97, 0.97, 0.97),
}
LID_COLOUR = {"cart_left": "lid_left", "cart_right": "lid_right"}
