"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A new below-ground foul drain drawn at 1:50 (1 m = 20 mm) on a plan sheet fixed to the table. The run starts at its
head H (its free end), runs as 100 mm pipe to P, where it becomes 150 mm pipe, on to the bend B1, turns toward the
robot to the bend B2 and turns again to the existing chamber at the site boundary. It has no junctions and is laid
at one fall. Approved Document H (England), H1 Section 2, paragraph 2.49 (2015 edition): access on or near the head
of each run, at a bend and at a change of gradient, at a change of pipe size, at a junction. So this run needs
access at exactly four points: H, P, B1 and B2. The lengths (7.5, 8, 7 and 12.5 m) are all under Table 13's
spacing limits (22 m from the start of a drain to an inspection chamber, 45 m between inspection chambers), so
spacing adds none. The four points are not marked on the plan.
"""

import math

TOP = 0.7475  # station table top
SCALE = 50.0  # the plan: 1 m = 20 mm

# The plan sheet: a pale cream plastic sheet 1 mm thick fixed to the table; everything stands on it.
SHEET_H = 0.001
SHEET_TOP = TOP + SHEET_H
SHEET = (0.170, 0.420, -0.310, 0.350)  # x0, x1, y0, y1: 250 x 660 mm (12.5 x 33 m at 1:50)

# The four required access points (not marked) and the existing boundary chamber, in the order of flow.
H = (0.370, -0.260)  # head of the run (its free end)
P = (0.370, -0.110)  # the pipe changes from 100 mm to 150 mm
B1 = (0.370, 0.050)  # first bend (90 degrees, toward the robot)
B2 = (0.230, 0.050)  # second bend (90 degrees, toward the boundary)
EXIST = (0.230, 0.300)  # centre of the existing chamber at the boundary
POINTS = {"head": H, "size_change": P, "bend_1": B1, "bend_2": B2}
POINT_NAMES = tuple(POINTS)

# The drawn route: fixed dark blue vinyl strips 0.5 mm thick on the sheet; the width shows the pipe size.
LINE_T = 0.0005
LINE_TOP = SHEET_TOP + LINE_T
W100 = 0.010  # the 100 mm pipe is drawn 10 mm wide (about 5 px in the setup screenshot)
W150 = 0.020  # the 150 mm pipe 20 mm wide
EXIST_RADIUS = 0.035  # the existing chamber: 70 mm across ...
EXIST_H = 0.040  # ... 40 mm tall, fixed
COVER_RADIUS = 0.028  # its cover: a lighter disc 56 mm across, 1 mm thick, on top
COVER_H = 0.001
# The strips as (name, x0, x1, y0, y1): they abut and do not overlap; the corners are filled by the strip that
# runs along y (B1) and the one that runs along x (B2). The last strip stops at the existing chamber's side.
ROUTE = (
    ("pipe_100", H[0] - W100 / 2, H[0] + W100 / 2, H[1], P[1]),
    ("pipe_150_a", P[0] - W150 / 2, P[0] + W150 / 2, P[1], B1[1] + W150 / 2),
    ("pipe_150_b", B2[0] - W150 / 2, B1[0] - W150 / 2, B1[1] - W150 / 2, B1[1] + W150 / 2),
    ("pipe_150_c", B2[0] - W150 / 2, B2[0] + W150 / 2, B2[1] + W150 / 2, EXIST[1] - EXIST_RADIUS),
)
# The site boundary: a fixed red strip 3 mm wide, 0.5 mm thick, across the sheet just beyond the existing chamber.
BOUNDARY_Y = 0.340
BOUNDARY_W = 0.003

# The lengths between points, centre line (m on the table; x 50 for full size).
LENGTHS = {"H-P": 0.150, "P-B1": 0.160, "B1-B2": 0.140, "B2-existing": 0.250}
ROUTE_LEN = sum(LENGTHS.values())  # 0.700 m: 35 m at full size
TABLE13_FROM_START = 22.0  # m: start of an external drain to an inspection chamber
TABLE13_CHAMBERS = 45.0  # m: inspection chamber to inspection chamber

# Four loose chamber bases: black plastic cylinders, alike; any may go to any point.
CHAMBER_RADIUS = 0.030  # 60 mm across
CHAMBER_H = 0.040  # 40 mm tall
CHAMBER_MASS = 0.10
NAMES = ("chamber_1", "chamber_2", "chamber_3", "chamber_4")
START = {  # (x, y), each at least 60 mm from the route's centre line and 120 mm from the existing chamber
    "chamber_1": (0.230, -0.250),
    "chamber_2": (0.310, -0.190),
    "chamber_3": (0.340, 0.130),
    "chamber_4": (0.350, 0.260),
}

# Rule thresholds (scoring.py).
COVER_TOL = 0.015  # a chamber covers a point when its centre is within 15 mm (0.75 m at 1:50) of it, seen from above
UPRIGHT_DEG = 10.0  # 'counted': tilted at most this
SEAT_TOL = 0.003  # 'counted': its lowest point within this of the sheet top (standing on the plan, not stacked)
STILL_WINDOW_S = 0.5  # 'settled': its centre's path over the last 0.5 s ...
STILL_PATH = 0.002  # ... under 2 mm (a path, not the reported velocity, which chatters while a part is gripped)
STILL_WINDOWS_FULL = 2  # 1.0 needs every chamber settled in each of the last two 0.5 s windows (still for 1 s)
POINT_SCORE = 0.25
# 'counted' also needs the chamber let go. rigstate reports each gripper's pad-tip point (midway between its two
# finger tips, 13 mm past the TCP) and how far it is open, but not its yaw, so a finger may be on any side of that
# point. With the fingers FULL_GAP x opening apart, a finger (FINGER_THICK thick outside the gap and
# 2 x FINGER_HALF_WIDTH wide across it, as measured on yam_arena's finger collision meshes) can touch a chamber only
# if the chamber's axis is within release_reach(opening) of that point horizontally: 95 mm fully open, 80 mm
# closed on a chamber, 57 mm closed on nothing. A gripper whose pad-tip point is lower than RELEASE_ABOVE over the
# chamber's top while within that reach may be holding, pressing or propping it: it is not counted.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010  # a finger's thickness outside the gap
FINGER_HALF_WIDTH = 0.020  # half a finger's width across the closing axis
RELEASE_MARGIN = 0.005  # beyond the chamber's surface; also covers the wider gripper body above the fingers
RELEASE_ABOVE = 0.010  # held if the pad-tip point is within reach and lower than this above the chamber's top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a chamber that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.030  # ... 'on its starting mark': within this of it

# Colours (sRGB).
SHEET_CREAM = (0.93, 0.91, 0.84)
ROUTE_BLUE = (0.05, 0.12, 0.45)
BOUNDARY_RED = (0.80, 0.10, 0.10)
CONCRETE = (0.45, 0.45, 0.46)
COVER_GREY = (0.70, 0.70, 0.68)
CHAMBER_BLACK = (0.06, 0.06, 0.06)


def release_reach(opening: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the
    chamber's radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + CHAMBER_RADIUS + RELEASE_MARGIN


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# The arithmetic the scene rests on.
assert abs(_dist(H, P) - LENGTHS["H-P"]) < 1e-9 and abs(_dist(P, B1) - LENGTHS["P-B1"]) < 1e-9
assert abs(_dist(B1, B2) - LENGTHS["B1-B2"]) < 1e-9 and abs(_dist(B2, EXIST) - LENGTHS["B2-existing"]) < 1e-9
assert all(v * SCALE < TABLE13_FROM_START for v in LENGTHS.values())
assert min(_dist(a, b) for i, a in enumerate(POINTS.values()) for b in list(POINTS.values())[i + 1:]) >= 0.140 - 1e-9
assert all(_dist(s, EXIST) >= 0.120 for s in START.values())
