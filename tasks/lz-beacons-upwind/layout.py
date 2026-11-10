"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A helicopter landing zone at 1:110. Air-medical services commonly ask for a zone of about 100 x 100 ft on open
ground, a beacon at each corner and a fifth on the upwind side, so the pilot can read the wind. At 1:110, 100 ft
(30.48 m) is 277 mm, 90 ft (27.43 m) is 249 mm and 1 to 6 m is 10 to 55 mm. The grass field is 335 x 360 mm
(37 x 40 m). The windsock streams toward -y, so the wind blows from +y: the zone's upwind side is its left side.
"""

import math

TOP = 0.7475  # station table top
SCALE = 110.0  # the field, the zone and the distances (the beacons and the windsock are over-scale)
FT = 0.3048 / SCALE  # one foot at 1:110 (2.77 mm)

# Three flat mats, 1 mm thick, fixed to the table: the grass field in the middle, the road along its left edge,
# the parking strip along its right edge. Their tops are flush.
MAT_H = 0.001
MAT_TOP = TOP + MAT_H
GRASS = (0.130, 0.465, -0.170, 0.190)  # x0, x1, y0, y1: 335 x 360 mm
ROAD = (0.100, 0.480, 0.190, 0.260)  # 380 x 70 mm, left of the field
STRIP = (0.100, 0.480, -0.250, -0.170)  # 380 x 80 mm, right of the field

# A fire engine parked on the strip: a fixed red block (about 10 x 2.6 x 3 m at scale).
ENGINE_SIZE = (0.090, 0.024, 0.027)
ENGINE_X, ENGINE_Y = 0.330, -0.196  # x 0.285 .. 0.375, y -0.208 .. -0.184

# The windsock: a fixed white mast on the table beyond the road, and at its top a stiffened sleeve of five
# tapering bands (orange, white, orange, white, orange) streaming from the mast toward -y (downwind). It is over-scale
# (60 mm long) so its direction reads in the top camera: about 33 px long in the 960 x 528 setup screenshot.
MAST_X, MAST_Y = 0.450, 0.330
MAST_RADIUS = 0.005  # 10 mm across
MAST_H = 0.090
SLEEVE_Z = 0.080  # the sleeve's axis, above the table top (the mast's top is at 0.090)
SLEEVE_LEN = 0.012  # each band, along y
SLEEVE_SIDES = (0.018, 0.016, 0.014, 0.012, 0.010)  # square bands, tapering away from the mast
SLEEVE_Y0 = MAST_Y - MAST_RADIUS  # 0.325: the mouth, against the mast; the tip is at 0.265, beyond the road
WIND_FROM = (0.0, 1.0)  # the wind comes from +y (the sleeve points the way it blows: -y)

# Beacons: loose upright cylinders, alike in all but colour; the two ends of each are alike.
BEACON_RADIUS = 0.014
BEACON_H = 0.028
BEACON_MASS = 0.040
ORANGE = ("orange_1", "orange_2", "orange_3", "orange_4")
BLUE = "blue"
NAMES = (*ORANGE, BLUE)
START = {  # (x, y): two oranges and the blue on the road, two oranges on the parking strip
    "orange_1": (0.200, 0.225),
    "orange_2": (0.260, 0.225),
    "orange_3": (0.200, -0.215),
    "orange_4": (0.240, -0.215),
    "blue": (0.320, 0.225),
}

# Rule thresholds (scoring.py).
SIDE_FULL = 0.277  # 100 ft: every side at least this (0.5)
SIDE_PART = 0.249  # 90 ft (0.25)
ANGLE_FULL = (80.0, 100.0)  # every corner (degrees), 0.5
ANGLE_PART = (70.0, 110.0)  # 0.25
MARK_NEAR, MARK_FAR = 0.010, 0.055  # the blue beacon this far outside the upwind side's line (0.5) ...
MARK_T = (1.0 / 3.0, 2.0 / 3.0)  # ... and along its middle third
MARK_LOOSE = 0.100  # 0.25: outside the upwind side, up to this far out, anywhere along it
UPRIGHT_DEG = 10.0  # 'placed': tilted at most this
SEAT_TOL = 0.003  # 'placed': its lowest point within this of the grass top
# 'placed' also needs the beacon let go. rigstate reports each gripper's pad-tip point (midway between its two finger
# tips, 13 mm past the TCP) and how far it is open, but not its yaw, so a finger may be on any side of that point.
# With the fingers FULL_GAP x opening apart, a finger (FINGER_THICK thick outside the gap and 2 x FINGER_HALF_WIDTH
# wide across it, as measured on yam_arena's finger collision meshes) can touch a beacon only if the beacon's axis is
# within release_reach(opening) of that point horizontally: 79 mm fully open, 50 mm closed on a beacon. A gripper
# whose pad-tip point is lower than RELEASE_ABOVE over the beacon's top while within that reach may be holding,
# pressing or propping it: the beacon is not placed.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010  # a finger's thickness outside the gap
FINGER_HALF_WIDTH = 0.020  # half a finger's width across the closing axis
RELEASE_MARGIN = 0.005  # beyond the beacon's surface; also covers the wider gripper body above the fingers
RELEASE_ABOVE = 0.010  # held if the pad-tip point is within reach and lower than this above the beacon's top
STILL_WINDOW_S = 0.5  # 'still': its centre's path over the last 0.5 s ...
STILL_PATH = 0.005  # ... under 5 mm (a path, not the reported velocity, which chatters while a part is gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a beacon that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it

# Colours (sRGB).
GRASS_GREEN = (0.20, 0.55, 0.20)
ASPHALT = (0.22, 0.22, 0.24)
CONCRETE = (0.65, 0.65, 0.63)
ENGINE_RED = (0.80, 0.08, 0.06)
WHITE = (0.95, 0.95, 0.95)
SOCK_ORANGE = (1.00, 0.45, 0.05)
BEACON_ORANGE = (1.00, 0.50, 0.00)
BEACON_BLUE = (0.10, 0.30, 0.90)



def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the beacon's
    radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + BEACON_RADIUS + RELEASE_MARGIN


assert abs(SIDE_FULL - 100 * FT) < 0.0005 and abs(SIDE_PART - 90 * FT) < 0.0005
assert abs(MARK_NEAR - 1.0 / SCALE) < 0.001 and abs(MARK_FAR - 6.0 / SCALE) < 0.001
assert math.isclose(sum(WIND_FROM), 1.0)
