"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Rectangles are (centre x, centre y, size x, size y).

The site: a light grey ground sheet with three soil stains on it. Each stain is three tones of 1 mm vinyl sheet:
a pale tan halo (two abutting lobes) on the ground sheet, and on the halo a mid-brown streak and a near-black core
that abut. The core, set off-centre at one end of the streak, is the stain's darkest, most impacted spot (a drip
point and its run-off). Four orange pin flags stand on the bare table beside the sheet, two on each side.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
LAYER = 0.001  # every vinyl sheet is 1 mm thick

# The site sheet: light grey, fixed, lying on the table (top 1 mm up). x 0.080..0.460, y -0.310..+0.310.
SHEET = (0.270, 0.000, 0.380, 0.620)
SHEET_TOP = TOP + LAYER

# The three stains. Halo lobes lie on the site sheet (top 2 mm up); streak and core lie on the halo (top 3 mm up).
# The two lobes of a halo abut (edges touching, never overlapping); the streak and the core abut, and both lie
# wholly inside the halo's larger lobe.
# GPU check before Dream-1: S3's halo reaches y -0.280 and S1's core y +0.250, close to the ready-pose arms at
# y +-0.305 (pad tips at x 0.323, 25 cm up). In the top camera's first frame all three cores and every halo outline
# must be in view. If an arm hides one, move the stains 2 to 3 cm inboard (the pocket, gap and reach tests
# recompute) before running Dream-1.
STAINS = {
    "S1": {  # left of the midline
        "halo": ((0.325, 0.205, 0.150, 0.110),  # x 0.250..0.400, y 0.150..0.260
                 (0.215, 0.200, 0.070, 0.060)),  # x 0.180..0.250, y 0.170..0.230
        "streak": (0.3075, 0.225, 0.075, 0.040),  # x 0.270..0.345, y 0.205..0.245
        "core": (0.370, 0.225, 0.050, 0.050),  # x 0.345..0.395, y 0.200..0.250
    },
    "S2": {  # in the middle
        "halo": ((0.365, -0.025, 0.120, 0.160),  # x 0.305..0.425, y -0.105..+0.055
                 (0.390, 0.080, 0.060, 0.050)),  # x 0.360..0.420, y 0.055..0.105
        "streak": (0.340, -0.005, 0.040, 0.080),  # x 0.320..0.360, y -0.045..+0.035
        "core": (0.340, -0.070, 0.050, 0.050),  # x 0.315..0.365, y -0.095..-0.045
    },
    "S3": {  # right of the midline
        "halo": ((0.290, -0.230, 0.160, 0.100),  # x 0.210..0.370, y -0.280..-0.180
                 (0.395, -0.250, 0.050, 0.060)),  # x 0.370..0.420, y -0.280..-0.220
        "streak": (0.305, -0.210, 0.070, 0.040),  # x 0.270..0.340, y -0.230..-0.190
        "core": (0.245, -0.210, 0.050, 0.050),  # x 0.220..0.270, y -0.235..-0.185
    },
}
STAIN_NAMES = tuple(STAINS)
HALO_TOP = TOP + 2 * LAYER
CORE_TOP = TOP + 3 * LAYER  # the streak's top too

# The four pin flags: loose orange plastic cylinders standing upright on the bare table, 77.5 mm outside the
# sheet's side edges, two on each side. Each arm reaches only the two on its own side (the other arm's base is
# 0.705 m away in y).
FLAG_RADIUS = 0.0125  # 25 mm across
FLAG_HEIGHT = 0.060
FLAG_MASS = 0.027  # a cut polypropylene rod: 29.5 cm3 at 900-910 kg/m3 is 26.5-26.8 g
FLAG_STARTS = (  # (name, x, y)
    ("flag_l1", 0.200, 0.400),
    ("flag_l2", 0.290, 0.400),
    ("flag_r1", 0.200, -0.400),
    ("flag_r2", 0.290, -0.400),
)
FLAG_NAMES = tuple(f[0] for f in FLAG_STARTS)

# Scoring thresholds (scoring.py).
# A flag is PLANTED when (a) .. (d) all hold:
UPRIGHT_DEG = 10.0  # (a) its axis within this of vertical (either end down)
BOTTOM_TOL = 0.005  # (b) its lowest point within this of the table top: on the table or the 1-3 mm sheets, not on a flag
REST_WINDOW = 10  # (c) control steps (0.5 s at 20 Hz) over which ...
REST_PATH = 0.002  # ... its base centre's path must stay under this
# (d) Let go and clear. rigstate reports each gripper's pad-tip point (midway between its two finger tips) and how far
# it is open, but not its yaw, so a finger may be on any side of that point. With the fingers FULL_GAP x opening apart,
# a finger (FINGER_THICK thick outside the gap and 2 x FINGER_HALF_WIDTH wide across it over its lowest 38 mm, as
# measured on yam_arena's finger collision meshes) can touch a flag only if the flag's axis is within
# release_reach(opening) of that point horizontally: 78 mm fully open, 58 mm open to 50 mm (the demo), 48 mm closed
# on a flag. A gripper whose pad-tip point is lower than RELEASE_ABOVE over the flag's top while within that reach
# may be holding, pressing or propping it.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010  # a finger's thickness outside the gap (6.4 mm at the tip, 10 mm at 38 mm up)
FINGER_HALF_WIDTH = 0.020  # half a finger's width across the closing axis (27 mm wide at the tip, 40 mm higher)
RELEASE_MARGIN = 0.005  # beyond the flag's surface; also covers the wider gripper body above the fingers
RELEASE_ABOVE = 0.010  # held if the pad-tip point is within reach and lower than this above the flag's top
CORE_INSET = 0.005  # HOT SPOT: base centre inside the core at least this far in from all four edges (40 x 40 mm)
SHEET_INSET = 0.015  # BACKGROUND: base centre on the sheet at least this far inside its edges ...
BACKGROUND_MIN = 0.150  # ... and at least this far from every halo lobe (each stain's outline)
POINTS_PER_ITEM = 0.25  # three hot spots and one background point

# A new episode: every flag back upright on the table on its start mark, and one of them moved this far in a step.
START_TOL = 0.035  # within this of its mark (covers a seeded reset's jitter; the marks are 90 mm apart)
RESET_JUMP = 0.05

# Colours (sRGB).
SHEET_GREY = (0.78, 0.77, 0.74)
HALO_TAN = (0.72, 0.60, 0.44)
STREAK_BROWN = (0.45, 0.33, 0.22)
CORE_BLACK = (0.13, 0.10, 0.08)
FLAG_ORANGE = (0.95, 0.45, 0.08)


def release_reach(opening: float) -> float:
    """(d) The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the
    flag's radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + FLAG_RADIUS + RELEASE_MARGIN


def rect_bounds(r) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a rectangle given as (centre x, centre y, size x, size y)."""
    cx, cy, sx, sy = r
    return cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2


def rect_distance(px: float, py: float, r) -> float:
    """Distance from a point to a rectangle seen from above (0 inside it)."""
    cx, cy, sx, sy = r
    return math.hypot(max(abs(px - cx) - sx / 2, 0.0), max(abs(py - cy) - sy / 2, 0.0))


def target(stain: str) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a stain's hot-spot target: its core inset 5 mm on every side (40 x 40 mm)."""
    x0, x1, y0, y1 = rect_bounds(STAINS[stain]["core"])
    return x0 + CORE_INSET, x1 - CORE_INSET, y0 + CORE_INSET, y1 - CORE_INSET


def stain_distance(px: float, py: float, stain: str) -> float:
    """Distance from a point to a stain's outline (its nearest halo lobe; 0 on the stain)."""
    return min(rect_distance(px, py, lobe) for lobe in STAINS[stain]["halo"])


def nearest_stain_distance(px: float, py: float) -> float:
    return min(stain_distance(px, py, s) for s in STAIN_NAMES)


# The demo's background point: the middle of the clean pocket by the sheet's near edge, 177 mm from S1's outline,
# 190 mm from S2's and 208 mm from S3's.
BACKGROUND_POINT = (0.115, 0.005)
