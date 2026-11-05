"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

Distances that come from the hot-work rule are scaled 1:50: the 35 ft (10.7 m) clearance radius becomes 21 cm.
The objects themselves are tabletop stand-ins, not to that scale.
"""

import math

TOP = 0.7475  # station table top

# The hot-work area: a thin pale-yellow mat (fixed disc) centred on the weld mark. Its edge is the clearance radius,
# so 'no part of a combustible within 21 cm of the mark' is 'no part of it over the mat'.
MARK_X, MARK_Y = 0.27, 0.0  # the weld mark's centre (and the mat's and the plate's)
RADIUS = 0.21  # 35 ft at 1:50 is 21.3 cm
MAT_THICK = 0.001
MAT_TOP = TOP + MAT_THICK

# The work plate: a fixed steel plate lying on the mat, with the weld mark (a fixed orange square) at its centre.
PLATE_SIZE = (0.120, 0.120, 0.006)
PLATE_HALF = PLATE_SIZE[0] / 2
PLATE_TOP = MAT_TOP + PLATE_SIZE[2]  # TOP + 0.007
MARK_SIZE = (0.020, 0.020, 0.001)
MARK_TOP = PLATE_TOP + MARK_SIZE[2]

# Two holders: open square cups of four fixed walls; the table top is each cup's floor.
HOLDER_WALL = 0.006  # wall thickness
HOLDER_HEIGHT = 0.025  # wall height: a parked tool stands in the cup with its bottom 25 mm below the rim

# The torch: a loose upright dark-blue cylinder standing in its holder on the right; its lower end is the tip.
TORCH_RADIUS = 0.014  # 28 mm across
TORCH_HEIGHT = 0.100
TORCH_MASS = 0.08  # a 62 cm3 PVC rod (1.3 g/cm3)
TORCH_X, TORCH_Y = 0.26, -0.27
TORCH_CUP_INNER = 0.040  # 6 mm of play all round the 28 mm torch

# The extinguisher: a loose upright red cylinder standing in its holder on the left.
EXT_RADIUS = 0.022  # 44 mm across
EXT_HEIGHT = 0.110
EXT_MASS = 0.20  # a capped plastic tube weighted with sand
EXT_X, EXT_Y = 0.26, 0.27
EXT_CUP_INNER = 0.056  # 6 mm of play all round the 44 mm extinguisher

# The combustibles, inside the radius at the start, one in each arm's half of the table.
CARTON_SIZE = (0.040, 0.060, 0.040)  # along x, along y, tall: an empty cardboard carton
CARTON_MASS = 0.020
CARTON_X, CARTON_Y = 0.33, 0.11  # centre 12.5 cm from the mark, its nearest edge 8.9 cm
RAG_SIZE = (0.050, 0.050, 0.030)  # a folded oily rag
RAG_MASS = 0.020
RAG_X, RAG_Y = 0.25, -0.13  # centre 13.2 cm from the mark, its nearest edge 10.5 cm

# Rule thresholds (scoring.py).
HOT_HEIGHT = 0.025  # 'hot': the torch tip over the plate and at most this far above its top (working distance)
ON_MARK = 0.015  # 'welding the mark': hot, with the tip within this of the mark's centre horizontally
WELD_S = 5.0  # the weld is done after this much welding-the-mark time in total
POST_WATCH_S = 10.0  # the watch must stay posted this long after the last hot step (stands in for 30 min)
READY_S = 2.0  # 0.25: area clear and watch posted together for this long without a break
LOOSE_S = 0.25  # the torch or extinguisher out of its holder and not held for longer than this: set down (0)
WATCH_TILT_DEG = 20.0  # the extinguisher held within this of upright to count as ready
PARKED_TILT_DEG = 10.0  # a parked torch or extinguisher stands within this of upright
PARKED_FLOOR = 0.005  # ... with its lowest point within this of the table top (the cup's floor)
HELD_SLACK = 0.006  # 'held': the gripper's finger gap no more than this wider than the part,
HELD_AXIS = 0.012  # ... the point between its finger tips within this of the part's axis, inside its length
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939

# Run limit (task.json "run_limit"): 1,800 control steps, 90 s at 20 Hz.
RUN_LIMIT_STEPS = 1800

# Colours (sRGB).
MAT_YELLOW = (0.95, 0.86, 0.45)
STEEL = (0.55, 0.57, 0.60)
ORANGE = (1.00, 0.45, 0.05)
HOLDER_GREY = (0.30, 0.30, 0.32)
TORCH_BLUE = (0.12, 0.20, 0.50)
RED = (0.85, 0.08, 0.08)
CARDBOARD = (0.66, 0.50, 0.32)
RAG_DARK = (0.28, 0.24, 0.20)


def dist_to_mark(x: float, y: float) -> float:
    return math.hypot(x - MARK_X, y - MARK_Y)
