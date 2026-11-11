"""Every dimension of the task, in one place (metres, kilograms, degrees, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475 and is the surgery's floor.

A table-top dental surgery. The patient chair is a fixed block with a headrest at its -y end, so a patient lying
supine in it has the top of the head toward -y and the feet toward +y. The painted mouth mark on the headrest is
the centre of the four-handed clock (zones of activity): seen from above with the patient's face as the clock and
the top of the head at 12, a right-handed operator works at 7 to 12, the assistant at 2 to 4, instruments pass in
the transfer zone at 4 to 7 and the static zone at 12 to 2 holds the set-up. For a left-handed operator every zone
mirrors about the 12 to 6 axis: operator 12 to 5, transfer 5 to 8, assistant 8 to 10, static 10 to 12.

Clock angle phi of a point (dx, dy) from the mouth mark, clockwise seen from above from 12 o'clock (-y), 30 degrees
per hour: phi = atan2(-dx, -dy). So 12 is -y (behind the head), 3 is -x (the supine patient's left, toward the
arms' bases), 6 is +y (the feet) and 9 is +x (the patient's right).
"""

import math

TOP = 0.7475  # station table top: the surgery floor

# The patient chair (fixed): a body and a headrest at its -y end, both 40 mm tall, touching at y 0.070.
CHAIR_H = 0.040
CHAIR = (0.220, 0.300, 0.070, 0.370)  # x0, x1, y0, y1: 80 x 300 mm
HEADREST = (0.230, 0.290, 0.020, 0.070)  # 60 x 50 mm
MOUTH = (0.260, 0.050)  # the painted mouth mark on the headrest's top, the clock's centre
MOUTH_RADIUS = 0.006  # a 12 mm red disc ...
MARK_H = 0.0005  # ... 0.5 mm thick (paint), on the headrest's top

# The three loose pieces of equipment.
D_STOOL = "dentist_stool"
A_STOOL = "assistant_stool"
CART = "mobile_cart"
NAMES = (D_STOOL, A_STOOL, CART)
STOOL_RADIUS = 0.025  # both stools 50 mm across
D_STOOL_H = 0.040  # the dentist's stool is the lower one ...
A_STOOL_H = 0.060  # ... the assistant sits higher
CART_SIZE = (0.060, 0.045, 0.045)  # x, y, z before yaw
MASS = {D_STOOL: 0.12, A_STOOL: 0.15, CART: 0.20}
HEIGHT = {D_STOOL: D_STOOL_H, A_STOOL: A_STOOL_H, CART: CART_SIZE[2]}
START = {  # (x, y, yaw deg): the two the right arm fetches at its side, the assistant's stool at the foot end
    D_STOOL: (0.280, -0.330, 0.0),
    A_STOOL: (0.400, 0.330, 0.0),
    CART: (0.360, -0.340, 0.0),
}

# The rule (scoring.py). Distances are horizontal, from the mouth mark to an object's centre.
R_IN, R_OUT = 0.080, 0.200  # 'at the chair': 80 to 200 mm from the mouth mark
LH_DENTIST = (15.0, 105.0)  # 12:30 to 3:30, inside the left-handed operator zone (12 to 5)
LH_ASSISTANT = (247.5, 292.5)  # 8:15 to 9:45, inside the left-handed assistant zone (8 to 10)
LH_CART = (247.5, 352.5)  # 8:15 to 11:45: the left-handed assistant's side (assistant or static zone, 8 to 12)
RH_DENTIST = (255.0, 345.0)  # 8:30 to 11:30, inside the right-handed operator zone (7 to 12)
RH_ASSISTANT = (67.5, 112.5)  # 2:15 to 3:45, inside the right-handed assistant zone (2 to 4)
TRANSFER = (150.0, 240.0)  # the left-handed transfer zone, 5 to 8: no object's centre within R_OUT of the mouth
UPRIGHT_DEG = 10.0  # 'standing': its vertical axis within this of vertical (either end up) ...
FLOOR_TOL = 0.005  # ... and its lowest point within this of the floor (not on the chair, another object or held up)
# 'Released': no gripper holds it over the last 0.5 s. rigstate reports each gripper's pad-tip point (midway between
# the finger tips, 13 mm past the TCP) and how far it is open, but not its yaw, so a finger may be on any side of that
# point. With the fingers FULL_GAP x opening apart, a finger (FINGER_THICK thick outside the gap and
# 2 x FINGER_HALF_WIDTH wide across it, as on yam_arena's finger collision meshes) can touch an object only if its
# centre is within release_reach(opening, name) of that point horizontally. A gripper whose pad-tip point is within
# that reach and lower than RELEASE_ABOVE over the object's top may be holding, pressing or propping it.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010
STILL_WINDOW_S = 0.5  # 'settled': its centre's path over the last 0.5 s ...
STILL_PATH = 0.002  # ... under 2 mm (a path, not the reported velocity, which chatters while a part is gripped)
STILL_WINDOWS_FOR_FULL = 2  # 1.0 also needs every object settled in each of the last two 0.5 s windows (1 s)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it

# The demo's set-down points (scoring needs none of them): each 80 mm or more inside its window's edges.
DEMO_AT = {  # name -> (clock angle deg, distance from the mouth m)
    D_STOOL: (45.0, 0.100),  # 1:30, 100 mm
    A_STOOL: (270.0, 0.100),  # 9:00, 100 mm
    CART: (330.0, 0.130),  # 11:00, 130 mm
}

# Colours (sRGB).
CHAIR_BLUE = (0.72, 0.86, 0.96)
MOUTH_RED = (0.85, 0.05, 0.05)
STOOL_BLUE = (0.05, 0.20, 0.85)
STOOL_GREEN = (0.10, 0.65, 0.15)
CART_GREY = (0.45, 0.45, 0.47)


def clock_xy(phi_deg: float, r: float) -> tuple[float, float]:
    """The point at clock angle ``phi_deg`` and distance ``r`` from the mouth mark."""
    p = math.radians(phi_deg)
    return MOUTH[0] - r * math.sin(p), MOUTH[1] - r * math.cos(p)


def clock_of(x: float, y: float) -> tuple[float, float]:
    """(phi in degrees 0..360, r in m) of a point seen from above, from the mouth mark."""
    dx, dy = x - MOUTH[0], y - MOUTH[1]
    return math.degrees(math.atan2(-dx, -dy)) % 360.0, math.hypot(dx, dy)


def hours(phi_deg: float) -> str:
    """A clock angle as h:mm (12:00 at 0 degrees)."""
    minutes = round(phi_deg % 360.0 * 2) % 720
    h, m = divmod(minutes, 60)
    return f"{h or 12}:{m:02d}"


def half_width(name: str) -> float:
    """The farthest an object's side reaches horizontally from its centre (its radius, or a box's half diagonal)."""
    return STOOL_RADIUS if name != CART else math.hypot(CART_SIZE[0] / 2, CART_SIZE[1] / 2)


def release_reach(opening: float, name: str) -> float:
    """The farthest an object's centre can be, horizontally, from a gripper's pad-tip point while a finger touches
    it, over every yaw, plus 5 mm. ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + half_width(name) + RELEASE_MARGIN


assert abs(clock_of(*clock_xy(90.0, 0.1))[0] - 90.0) < 1e-9 and clock_xy(90.0, 0.1)[0] < MOUTH[0]  # 3 is -x
assert abs(clock_of(*clock_xy(180.0, 0.1))[0] - 180.0) < 1e-9 and clock_xy(180.0, 0.1)[1] > MOUTH[1]  # 6 is +y
