"""Every dimension and threshold of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81

# -- the crucible former (fixed) -----------------------------------------------------------------------------
# A square black rubber block (the former's base) with a round centring step on top; its axis is the former's axis.
FORMER_X, FORMER_Y = 0.300, 0.080
BASE_SIZE = (0.088, 0.088, 0.010)
BASE_HALF = BASE_SIZE[0] / 2  # 0.044
BASE_TOP = TOP + BASE_SIZE[2]  # 10 mm up
STEP_RADIUS = 0.018  # the centring step: 36 mm across ...
STEP_HEIGHT = 0.006  # ... 6 mm tall
STEP_TOP = BASE_TOP + STEP_HEIGHT  # 16 mm up

# -- the sprued wax pattern (fixed) --------------------------------------------------------------------------
# A red wax sprue (a 4 mm rod) on the step, 6 mm left (+y) of the former's axis, and a blue wax crown on it.
WAX_X, WAX_Y = FORMER_X, FORMER_Y + 0.006
SPRUE_RADIUS = 0.002
SPRUE_HEIGHT = 0.010
SPRUE_BOTTOM = STEP_TOP  # 16 mm up
SPRUE_TOP = SPRUE_BOTTOM + SPRUE_HEIGHT  # 26 mm up
CROWN_SIZE = (0.010, 0.012, 0.010)  # x, y, z
CROWN_BOTTOM = SPRUE_TOP  # 26 mm up
CROWN_TOP = CROWN_BOTTOM + CROWN_SIZE[2]  # 36 mm up: the cover is measured from here
CROWN_CENTRE = (WAX_X, WAX_Y, CROWN_BOTTOM + CROWN_SIZE[2] / 2)
# For the ring-wall test the round sprue counts as the 4 x 4 mm square post around it (never smaller).
SPRUE_POST_HALF = (SPRUE_RADIUS, SPRUE_RADIUS, SPRUE_HEIGHT / 2)
SPRUE_CENTRE = (WAX_X, WAX_Y, SPRUE_BOTTOM + SPRUE_HEIGHT / 2)
WAX_SAMPLE = 0.002  # the finger test samples the crown's and the sprue's surfaces at this spacing (or finer)

# -- the casting rings (loose) -------------------------------------------------------------------------------
# Square stainless-steel tubes, open at both ends: four 4 mm walls, 60 x 60 mm outside, a 52 x 52 mm opening. Each is
# a mechkit assembly whose base frame sits at the centre of its bottom face. Mass: steel (7.9 g/cm3) of that volume.
RING_OUTER = 0.060
RING_WALL = 0.004
RING_INNER = RING_OUTER - 2 * RING_WALL  # 0.052
RING_OUTER_HALF = RING_OUTER / 2  # 0.030
RING_INNER_HALF = RING_INNER / 2  # 0.026
RINGS = {  # name: (height, mass, start x, start y); neutral names so nothing in the scene says which one is right
    "ring_a": (0.040, 0.28, 0.200, 0.330),
    "ring_b": (0.024, 0.17, 0.290, 0.320),
    "ring_c": (0.032, 0.23, 0.390, 0.290),
}
RING_NAMES = tuple(RINGS)
RIGHT_RING = "ring_c"  # seated, its top is 42 mm up: 6 mm over the crown's top (ring_b: 34 mm, -2; ring_a: 50, +14)
COVER_MIN, COVER_MAX = 0.005, 0.008  # the instruction's band for the ring's top over the crown's top


def ring_height(name: str) -> float:
    return RINGS[name][0]


def cover(name: str) -> float:
    """The cover a ring leaves over the crown's top when it stands on the former's base (m)."""
    return (BASE_TOP + ring_height(name)) - CROWN_TOP


# -- the grippers (yam_arena; rigstate reports the opening as 0 closed .. 1 open) --------------------------------
FULL_GAP = 0.0939  # pads 93.9 mm apart fully open
# Each finger, for the touch rule: a box 10 mm thick outside the gap, 40 mm wide and 70 mm long from the tips up to
# the finger carriage (the real finger is 10 mm by 40 mm 19-38 mm above the tips, 6.4 mm by 27 mm at the tips), posed
# as the gripper is and grown 1 mm on every side.
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
FINGER_LEN = 0.070
ZONE_MARGIN = 0.001
# The gripper's turn about its axis comes from its `gripper` link's pose (gripturn.py). A reading is used only if it
# puts the tool point (TCP_OFFSET along the link's z) within TURN_POS_TOL of rigstate's and the link's z within
# TURN_AXIS_DEG of rigstate's tool-point-to-tips axis; otherwise the finger box is swept round the axis (any turn).
TCP_OFFSET = 0.134
TURN_POS_TOL = 0.003
TURN_AXIS_DEG = 3.0

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------
WAX_GAP = 0.0005  # a ring wall within 0.5 mm of the crown or the sprue post: touching the wax
ENCLOSE_TILT_DEG = 10.0  # 'over the wax': tilted at most this ...
# ... with every corner of the crown and of the sprue post inside the 52 x 52 mm opening (in the ring's frame) and
# the crown's top above the ring's bottom face.
SEAT_LOW, SEAT_HIGH = 0.009, 0.012  # 'level on the base': all four lower outer corners 9 to 12 mm up (on the base,
#                                     not on the step); within 3 mm over the 60 mm sides, so tilted under 2.9 degrees
HOLD_GAP = 0.088  # a gripper with its fingers under 88 mm apart (the ring's 84.9 mm diagonal plus 3 mm: a grip at
#                   any turn stops under it) ...
HOLD_RADIUS = 0.045  # ... its tip point within 45 mm of the ring's axis (horizontally; the corners are at 42.4 mm)
#                      and below the ring's top holds it
STILL_STEPS = 10  # 'still': the ring's centre moved at most STILL_PATH in total over the last 10 steps (0.5 s) ...
STILL_PATH = 0.002  # ... (positions, never velocities: a resting part's velocity chatters on the GPU)
SECOND_RING_RADIUS = 0.045  # a ring is on the former if its centre is this close to the former's axis, seen from
ON_FORMER_Z = 0.005  # above, or a point of its lower end's outer edge (sampled every EDGE_SAMPLE) is over the base's
EDGE_SAMPLE = 0.005  # outline and at least ON_FORMER_Z above the table. Two or more rings on the former at the end: 0
BLIND_LIMIT = 0.01  # the arms unreadable on more than 1% of the steps: 0 (the touch rule could not be judged)
RESET_JUMP = 0.08  # a ring moving this much further in one step than its velocity allows was teleported ...
START_TOL = 0.045  # ... back onto its start: within this (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
RUBBER_BLACK = (0.08, 0.08, 0.08)
SPRUE_RED = (0.80, 0.15, 0.12)
WAX_BLUE = (0.12, 0.30, 0.80)
STEEL = (0.62, 0.63, 0.65)

# Sanity: the right ring is the only one in the band; the step keeps a seated ring within 8 mm of the axis and the
# sprue stands on it; a grip across the ring's corners counts as held.
assert sum(COVER_MIN <= cover(n) <= COVER_MAX for n in RINGS) == 1 and COVER_MIN <= cover(RIGHT_RING) <= COVER_MAX
assert math.isclose(RING_INNER_HALF - STEP_RADIUS, 0.008)
assert math.hypot(WAX_X - FORMER_X, WAX_Y - FORMER_Y) + SPRUE_RADIUS < STEP_RADIUS
assert math.hypot(RING_OUTER, RING_OUTER) < HOLD_GAP < FULL_GAP and math.hypot(RING_OUTER_HALF, RING_OUTER_HALF) < HOLD_RADIUS
