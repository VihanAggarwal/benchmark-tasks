"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Depth down the soil core is measured along -y from the
core's top end (the ground surface) at y +0.280: depth = 0.280 - y.
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# The core tray: a flat matte black board fixed to the table, long side along y.
TRAY_X0, TRAY_X1 = 0.320, 0.500
TRAY_Y_LEFT, TRAY_Y_RIGHT = 0.310, -0.410  # its +y end (beside the core's top) and its -y end
TRAY_H = 0.006
TRAY_X = (TRAY_X0 + TRAY_X1) / 2  # 0.410
TRAY_Y = (TRAY_Y_LEFT + TRAY_Y_RIGHT) / 2  # -0.050
TRAY_SIZE = (TRAY_X1 - TRAY_X0, TRAY_Y_LEFT - TRAY_Y_RIGHT, TRAY_H)  # 180 x 720 x 6 mm
TRAY_TOP = TOP + TRAY_H

# The soil core: a straight flat-topped band lying on the tray, 50 mm wide in x and 30 mm tall, from the ground
# surface at y +0.280 to 560 mm deep at y -0.280.
CORE_X0, CORE_X1 = 0.385, 0.435  # its two long edges: near side (toward the arms) and far side
CORE_X = (CORE_X0 + CORE_X1) / 2  # 0.410
CORE_W = CORE_X1 - CORE_X0  # 0.050
CORE_H = 0.030
CORE_TOP_Z = TRAY_TOP + CORE_H  # 36 mm above the table
CORE_Y_TOP = 0.280  # the ground surface (depth 0)
CORE_LEN = 0.560  # to y -0.280


def y_at(depth: float) -> float:
    """World y of a depth (m) down the core."""
    return CORE_Y_TOP - depth


# The horizons, top down: (name, top depth, bottom depth, sRGB). Matte.
HORIZONS = (
    ("A", 0.000, 0.046, (0.36, 0.25, 0.16)),  # dark brown
    ("E", 0.046, 0.173, (0.75, 0.70, 0.60)),  # light brownish grey
    ("Bt", 0.173, 0.314, (0.60, 0.30, 0.18)),  # reddish brown
    ("BC", 0.314, 0.434, (0.78, 0.60, 0.32)),  # yellowish brown
    ("C", 0.434, 0.560, (0.50, 0.52, 0.42)),  # olive grey
)
# In simulation the core is seven fixed boxes that abut exactly, with equal tops: Bt and BC are each two boxes of
# the same colour (extra joints at 245 and 374 mm deep), so a joint between boxes is never a sign of a boundary.
SEGMENTS = (  # (object name, top depth, bottom depth, horizon)
    ("core_A", 0.000, 0.046, "A"),
    ("core_E", 0.046, 0.173, "E"),
    ("core_Bt_upper", 0.173, 0.245, "Bt"),
    ("core_Bt_lower", 0.245, 0.314, "Bt"),
    ("core_BC_upper", 0.314, 0.374, "BC"),
    ("core_BC_lower", 0.374, 0.434, "BC"),
    ("core_C", 0.434, 0.560, "C"),
)
HORIZON_RGB = {name: rgb for name, _, _, rgb in HORIZONS}

# The four boundaries to tag: (name, depth, world y). At least 120 mm apart.
BOUNDARY_DEPTHS = (0.046, 0.173, 0.314, 0.434)
BOUNDARIES = tuple((f"B{i + 1}", d, y_at(d)) for i, d in enumerate(BOUNDARY_DEPTHS))
BOUNDARY_Y = tuple(b[2] for b in BOUNDARIES)  # +0.234, +0.107, -0.034, -0.154

# The four depth tags: loose white plastic cylinders standing upright on the table, a pair at each end of the tray
# (as on a long logging bench with a box of tags at each end).
TAG_RADIUS = 0.010
TAG_HEIGHT = 0.040
TAG_MASS = 0.015
TAG_STARTS = (  # (name, x, y): the left arm's two beyond the right arm's reach and the right arm's beyond the left's
    ("tag_l1", 0.200, 0.400),
    ("tag_l2", 0.280, 0.400),
    ("tag_r1", 0.200, -0.400),
    ("tag_r2", 0.280, -0.400),
)
TAG_NAMES = tuple(t[0] for t in TAG_STARTS)

# Scoring thresholds (scoring.py).
UPRIGHT_DEG = 10.0  # (a) its axis within this of vertical, either end down
ON_TRAY_TOL = 0.003  # (b) its lowest point within this of the tray top (3 to 9 mm above the table)
REST_WINDOW = 10  # (c) control steps (0.5 s at 20 Hz) over which ...
REST_PATH = 0.003  # ... its centre's path length must stay under this
BESIDE_MAX = 0.045  # (d) its centre at most this far outside the nearer long edge of the core
NEAR_BAND = (CORE_X0 - BESIDE_MAX, CORE_X0)  # x 0.340 .. 0.385 (the near side)
FAR_BAND = (CORE_X1, CORE_X1 + BESIDE_MAX)  # x 0.435 .. 0.480 (the far side)
# (e) Let go and clear. rigstate reports each gripper's pad-tip point (midway between its two finger tips) and how far
# it is open, but not its yaw, so a finger may be on any side of that point. With the fingers FULL_GAP x opening apart,
# a finger (FINGER_THICK thick outside the gap and 2 x FINGER_HALF_WIDTH wide across it over its lowest 38 mm, as
# measured on yam_arena's finger collision meshes) can touch a tag only if the tag's axis is within
# release_reach(opening) of that point horizontally: 75 mm fully open, 43 mm closed on a tag, 37 mm fully closed.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010  # a finger's thickness outside the gap (6.4 mm at the tip, 10 mm at 38 mm up)
FINGER_HALF_WIDTH = 0.020  # half a finger's width across the closing axis (27 mm wide at the tip, 40 mm higher)
RELEASE_MARGIN = 0.005  # beyond the tag's surface; also covers the finger above its lowest 38 mm (about 2 mm more
#   finger beside a 40 mm tag when the pad tips are at tray level) and the wider gripper body higher up
RELEASE_ABOVE = 0.010  # held if the pad-tip point is within reach and lower than this above the tag's top


def release_reach(opening: float) -> float:
    """(e) The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the
    tag's radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return ((half_gap + FINGER_THICK) ** 2 + FINGER_HALF_WIDTH ** 2) ** 0.5 + TAG_RADIUS + RELEASE_MARGIN


TAG_TOL = 0.010  # a placed tag tags boundary k when its centre is within this of it along the core
POINTS_PER_BOUNDARY = 0.25

# A new episode: every tag back on its start mark, standing on the table, and one of them moved this far in a step.
START_TOL = 0.035  # within this of its mark (covers a seeded reset's jitter; the marks are 80 mm apart)
RESET_JUMP = 0.05

# Colours (sRGB).
TRAY_BLACK = (0.12, 0.12, 0.12)
TAG_WHITE = (0.95, 0.95, 0.95)
