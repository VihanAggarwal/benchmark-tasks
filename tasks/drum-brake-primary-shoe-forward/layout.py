"""Every dimension of the task, in one place (metres, kilograms, degrees where named *_DEG, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A bench model, about half size, of the two rear duo-servo drum brakes of one car (a 9 inch, 229 mm drum at full
size). Each backing plate lies shoe side up, its anchor post on the far side (+x, the top of the brake) and its
adjuster block on the near side (-x). Seen this way, the front of the car is to the left (+y) of the LEFT rear brake
and to the right (-y) of the RIGHT rear brake (left and right brakes are mirror images); an arrow painted on each
plate, and a larger one on the bench beyond it, points that way. Each plate has two painted seats, one on each side
of the line from the anchor to the adjuster. On a duo-servo brake the primary shoe (shorter lining) goes on the seat
toward the front of the car and the secondary (longer lining) on the rear one.

Shoe body frame: origin at the mid-arc point of the shoe's centre line (46 mm from its centre of curvature) on its
bottom face; body +x radially outward (the convex, lined side), z up through the painted lining on its top. The
centre of curvature is at body (-0.046, 0, 0). A shoe sits in a seat when that centre is on the plate's centre and the
shoe's mid-arc direction points across the plate, toward +y or -y.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the two backing plates (fixed): steel discs, 120 mm across, 6 mm thick ----------------------------------------
PLATE_R = 0.060
PLATE_H = 0.006
PLATE_X = 0.350
PLATE_Y = 0.170
PAINT_H = 0.0005  # painted seats and arrows: 0.5 mm proud of what they are painted on
PLATE_TOP = PLATE_H  # what a fitted shoe rests on (the seat paint adds 0.5 mm, inside the 3 mm tolerance)
# plate name -> (centre x, centre y, FRONT direction along y: +1 toward +y, -1 toward -y)
PLATES = {"left_plate": (PLATE_X, PLATE_Y, 1.0), "right_plate": (PLATE_X, -PLATE_Y, -1.0)}

# -- on each plate: an anchor post at the top (+x) and an adjuster block at the bottom (-x), both fixed ---------------
R_MID = 0.046  # a fitted shoe's centre line: 46 mm from the plate's centre
ANCHOR_R = 0.005  # 10 mm across
ANCHOR_H = 0.008  # 8 mm tall on the plate
ANCHOR_DX = R_MID  # the anchor's centre, from the plate's centre, along +x
ADJ_SIZE = (0.012, 0.014, 0.008)  # 12 (x) x 14 (y) x 8 mm tall on the plate
ADJ_DX = -R_MID  # the adjuster block's centre, from the plate's centre, along x

# -- the shoes: four loose rigid bodies (mechkit compounds), all one shape ------------------------------------------
# A curved bar 20 mm wide (radially, inner radius 36, outer 56 mm) and 15 mm tall, its centre line a 46 mm radius arc,
# with rounded ends: 140 degrees end to end along its centre line. Built from a 20 x 16 x 15 mm block at mid-arc (the
# part a gripper closes on: flat faces 20 mm apart) and twelve upright 20 mm discs centred on the centre line at
# +-10, 20, 30, 40, 50 and 57.5 degrees from mid-arc; the last disc's edge is 70 degrees round (on the centre line).
SHOE_HALF_W = 0.010  # half the radial width
SHOE_H = 0.015
SHOE_SPAN_DEG = 140.0
END_DISC_DEG = SHOE_SPAN_DEG / 2 - math.degrees(math.asin(SHOE_HALF_W / R_MID))  # 57.44: its edge at 70 degrees
DISC_DEGS = (10.0, 20.0, 30.0, 40.0, 50.0, END_DISC_DEG)  # either side of mid-arc
GRIP_BLOCK = (0.020, 0.016, SHOE_H)  # at body (0, 0, SHOE_H / 2)
SHOE_MASS = 0.04  # 3D-printed solid PETG, painted (32 cm3 at 1.27 g/cm3: 41 g)
# The lining, painted dark brown on the shoe's top along its outer edge: a band 6 mm wide (radius 50 to 56 mm on
# the drawing; built from 1 mm thick, 6 mm discs centred 52.5 mm out), over the middle 90 degrees of a primary shoe
# and the middle 130 degrees of a secondary.
LINING_R = 0.0525
LINING_DOT_R = 0.003
LINING_H = 0.001
LINING_DEG = {"primary": 90.0, "secondary": 130.0}
LINING_STEP_DEG = 3.5  # the discs' spacing along the band (3.2 mm at 52.5 mm: they overlap)
SHOE_TOP = SHOE_H + LINING_H  # 16 mm with the lining
CURVE_C = (-R_MID, 0.0)  # the shoe's centre of curvature in its body frame

# Shoe name -> (kind, mid-arc point x, y, yaw in degrees). They lie on the bench in a row in front of the plates,
# convex sides toward the plates (yaw 0), secondary, primary, primary, secondary from -y to +y: each arm has one of
# each on its own side, and the row is a mirror image about the midline.
SHOES = {
    "secondary_1": ("secondary", 0.250, -0.330, 0.0),
    "primary_1": ("primary", 0.250, -0.110, 0.0),
    "primary_2": ("primary", 0.250, 0.110, 0.0),
    "secondary_2": ("secondary", 0.250, 0.330, 0.0),
}
NAMES = tuple(SHOES)
PRIMARIES = tuple(n for n in NAMES if SHOES[n][0] == "primary")
SECONDARIES = tuple(n for n in NAMES if SHOES[n][0] == "secondary")

# -- seats: where a shoe's mid-arc point goes, and its yaw (the mid-arc direction across the plate) -------------------
# (plate, side) -> (x, y, yaw in degrees); side "front" is the seat toward the plate's FRONT arrow.
SEATS = {}
for _plate, (_x, _y, _f) in PLATES.items():
    SEATS[(_plate, "front")] = (_x, _y + _f * R_MID, 90.0 * _f)
    SEATS[(_plate, "rear")] = (_x, _y - _f * R_MID, -90.0 * _f)
SEAT_KEYS = tuple(SEATS)

# -- painted arrows (fixed, white): one on each plate's centre and a larger one on the bench beyond its far edge ----
# Each is a shaft and a stepped head of three bars, pointing along y toward the front of the car (FRONT, from the
# plate's centre). (along-y start, along-y end, width across) in the plate's frame, y multiplied by its FRONT sign.
PLATE_ARROW = ((-0.020, 0.012, 0.006), (0.012, 0.018, 0.016), (0.018, 0.023, 0.010), (0.023, 0.027, 0.004))
BENCH_ARROW_DX = 0.090  # its centre line: 90 mm beyond the plate's centre (x 0.440), on the table
BENCH_ARROW = ((-0.045, 0.025, 0.010), (0.025, 0.035, 0.030), (0.035, 0.045, 0.020), (0.045, 0.055, 0.010))

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
FINE_POS = 0.008  # 'seated (fine)': mid-arc point within 8 mm of the seat's ...
FINE_YAW_DEG = 8.0  # ... and yaw within 8 degrees of the seat's
COARSE_POS = 0.014  # 'seated (coarse)'
COARSE_YAW_DEG = 15.0
TILT_MAX_DEG = 5.0  # 'flat': the shoe's up axis within 5 degrees of vertical (upside down is not flat)
SEAT_TOL = 0.003  # 'on the plate': its lowest point within 3 mm of the plate's top
STILL_WINDOW_S = 0.5  # 'settled': over the last 0.5 s ...
STILL_PATH = 0.002  # ... its mid-arc point's path under 2 mm ...
STILL_YAW_DEG = 1.0  # ... and its yaw within a 1 degree range
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a shoe that lands on its starting mark further than its velocity and gravity could carry it
#                    in one control step was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# 'let go': rigstate reports each gripper's pad-tip point (midway between its finger tips, 13 mm past the TCP) and
# how far it is open, not its yaw, so a finger may be on any side of that point. With the pads FULL_GAP x opening
# apart, a finger (FINGER_THICK thick outside the gap, 2 x FINGER_HALF_WIDTH wide across it, as measured on
# yam_arena's finger collision meshes) can touch a shoe only if the shoe's footprint comes within finger_reach of the
# pad-tip point horizontally. A gripper whose pad-tip point is lower than RELEASE_ABOVE over the shoe's top while
# within that reach may be holding, pressing or propping it: the shoe is not let go.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
PLATE_GREY = (0.22, 0.23, 0.25)
SEAT_YELLOW = (0.85, 0.72, 0.20)
ARROW_WHITE = (0.95, 0.95, 0.95)
ANCHOR_STEEL = (0.62, 0.64, 0.67)
ADJ_BRASS = (0.78, 0.62, 0.25)
SHOE_GREY = (0.74, 0.75, 0.77)
LINING_BROWN = (0.28, 0.16, 0.08)


def finger_reach(opening: float) -> float:
    """How far from a gripper's pad-tip point (horizontally) a finger can reach, over every yaw, plus the margin (m).
    ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + RELEASE_MARGIN


def shoe_disc_centres() -> list[tuple[float, float]]:
    """(x, y) of the shoe's twelve discs in its body frame (the block at mid-arc is separate)."""
    out = []
    for d in DISC_DEGS:
        for s in (-1.0, 1.0):
            a = math.radians(s * d)
            out.append((CURVE_C[0] + R_MID * math.cos(a), CURVE_C[1] + R_MID * math.sin(a)))
    return out


def lining_angles(kind: str) -> list[float]:
    """The lining discs' angles (degrees from mid-arc): the band's ends at +-half its length on the 52.5 mm line."""
    half = LINING_DEG[kind] / 2 - math.degrees(math.asin(LINING_DOT_R / LINING_R))
    n = max(1, math.ceil(2 * half / LINING_STEP_DEG))
    return [-half + 2 * half * i / n for i in range(n + 1)]


def lining_centres(kind: str) -> list[tuple[float, float]]:
    return [(CURVE_C[0] + LINING_R * math.cos(math.radians(a)), CURVE_C[1] + LINING_R * math.sin(math.radians(a)))
            for a in lining_angles(kind)]


assert abs(END_DISC_DEG + math.degrees(math.asin(SHOE_HALF_W / R_MID)) - SHOE_SPAN_DEG / 2) < 1e-9
assert all(abs(abs(v[2]) - 90.0) < 1e-9 for v in SEATS.values())
