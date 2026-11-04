"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# The seed bed: a fixed strip of soil along y, in front of both arms. 520 mm long so the seedlings at the ends
# (centres at y +-0.240, 10 mm radius) stand 10 mm inside its ends.
BED_X, BED_Y = 0.25, 0.0
BED_SIZE = (0.060, 0.520, 0.010)
BED_TOP = TOP + BED_SIZE[2]

# Ten seedlings: loose upright cylinders 20 mm across standing in a row on the bed, numbered from the left end
# (+y) to the right end. (name, y, height). Height stands for strength: the taller, the stronger.
SEEDLING_RADIUS = 0.010
SEEDLINGS = (
    ("seedling_01", +0.240, 0.070),
    ("seedling_02", +0.195, 0.085),
    ("seedling_03", +0.130, 0.050),
    ("seedling_04", +0.090, 0.075),
    ("seedling_05", +0.020, 0.065),
    ("seedling_06", -0.040, 0.090),
    ("seedling_07", -0.095, 0.055),
    ("seedling_08", -0.150, 0.060),
    ("seedling_09", -0.190, 0.045),
    ("seedling_10", -0.240, 0.080),
)
NAMES = tuple(n for n, _, _ in SEEDLINGS)
START_Y = {n: y for n, y, _ in SEEDLINGS}
HEIGHT = {n: h for n, _, h in SEEDLINGS}
SEEDLING_DENSITY = 800.0  # kg/m3: a seedling with its moist soil plug, as one uniform cylinder
MASS = {n: round(SEEDLING_DENSITY * math.pi * SEEDLING_RADIUS ** 2 * h, 4) for n, _, h in SEEDLINGS}  # 11-23 g


def start_pos(name: str) -> tuple[float, float, float]:
    """Where a seedling's centre is at the start: upright on the bed at its planting spot."""
    return (BED_X, START_Y[name], BED_TOP + HEIGHT[name] / 2)


# The thinning rule: going from the tallest down, keep a seedling if it is at least SPACING from every one kept.
SPACING = 0.100


def keep_set(names=NAMES) -> frozenset:
    """The seedlings the rule keeps (greedy by height over the planting spots)."""
    kept: list[str] = []
    for n in sorted(names, key=lambda m: -HEIGHT[m]):
        if all(abs(START_Y[n] - START_Y[k]) >= SPACING - 1e-9 for k in kept):
            kept.append(n)
    return frozenset(kept)


KEEP = keep_set()  # seedling_02, _04, _06, _10 (y +195, +90, -40, -240 mm)
PULL = frozenset(NAMES) - KEEP  # seedling_01, _03, _05, _07, _08, _09

# Two compost trays: four fixed walls round a 100 x 100 mm floor (the table top), one beside each arm's base.
TRAY_INNER = 0.100
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.040
TRAY_TOP_Z = TOP + TRAY_HEIGHT
TRAY_INNER_HALF = TRAY_INNER / 2
TRAY_OUTER_HALF = TRAY_INNER / 2 + TRAY_WALL
TRAYS = {"left": (0.14, +0.40), "right": (0.14, -0.40)}

# Scoring thresholds (scoring.py).
ON_BED_Z_TOL = 0.005  # 'on the bed': lowest point within this of the bed's top, centre over the bed
KEEP_TOL = 0.005  # a seedling kept in place: its centre never more than this from where it started
DISTURB_TOL = 0.015  # ... more than this (at any moment): disturbed
TILT_MAX_DEG = 15.0  # ... or tilted more than this at any moment: disturbed
MIN_LEFT = 3  # fewer seedlings than this left on the bed: over-thinned
MIN_PULLED_PARTIAL = 3  # 0.25: at least this many of the six to pull in a tray
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.010  # ... onto its starting place: within this of it (the layout is fixed: object_jitter=0)

# Colours (sRGB).
SOIL = (0.40, 0.27, 0.16)
LEAF = (0.35, 0.68, 0.25)
TRAY_BLACK = (0.13, 0.13, 0.13)
