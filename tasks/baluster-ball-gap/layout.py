"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The plan layout is a residential guard at 1:2: the 4 in (102 mm) sphere of the guard-opening rule becomes a 51 mm
ball, 1-1/2 in balusters become 20 mm, and 712 mm between newel posts becomes 356 mm. The heights are not to scale
(a real baluster is 34-38 in tall): only the openings along the rail matter to the rule, and balusters are
vertical, so an opening is the same at any height.
"""

TOP = 0.7475  # station table top

# The shoe rail: a fixed oak strip on the table along y that the balusters stand on, running face to face between
# the two newel posts.
RAIL_X = 0.300  # its centre line
RAIL_W = 0.040  # along x
RAIL_H = 0.010
POST_FACE = 0.178  # the newel posts' inner faces, at y +-POST_FACE (356 mm clear between them)
RAIL_LEN = 2 * POST_FACE
RAIL_TOP = TOP + RAIL_H

# Two newel posts: fixed oak blocks at the rail's ends.
POST_SIZE = (0.040, 0.040, 0.100)
POST_Y = POST_FACE + POST_SIZE[1] / 2  # 0.198
POSTS = (("newel_left", POST_Y), ("newel_right", -POST_Y))

# Six balusters: loose white-painted maple cylinders standing upright on the table, three within each arm's reach.
BAL_R = 0.010  # 20 mm across
BAL_D = 2 * BAL_R
BAL_H = 0.080
BAL_MASS = 0.018  # maple, about 700 kg/m3: 25 cm3
BALUSTERS = (  # (name, x, y) at the start
    ("baluster_l1", 0.260, 0.270),
    ("baluster_l2", 0.200, 0.270),
    ("baluster_l3", 0.140, 0.270),
    ("baluster_r1", 0.260, -0.270),
    ("baluster_r2", 0.200, -0.270),
    ("baluster_r3", 0.140, -0.270),
)
NAMES = tuple(n for n, _, _ in BALUSTERS)
START = {n: (x, y) for n, x, y in BALUSTERS}

# The rule: no opening may pass the test ball, and the openings should be even.
BALL = 0.051  # the test ball's diameter (4 in at 1:2); an opening of BALL or more passes it
SPREAD_MAX = 0.010  # 'evenly spaced': widest opening minus narrowest at most this (1.0)
MIN_FOR_QUARTER = 4  # 0.25 needs at least this many balusters standing on the rail (one opening passing)

# 'Stands on the rail' and 'down on the guard' (scoring.py).
UPRIGHT_DEG = 5.0  # axis within this of vertical (either end down)
SEAT_TOL = 0.002  # its bottom within this of the rail's top (above or below)
OVER_RAIL = RAIL_W / 2  # its centre over the rail's top: within this of the centre line (not past its edge)
DOWN_CLEAR = 0.020  # a baluster not standing, tilted, over the rail or a post, lowest point at most this above
#                     the rail's top: lying on or leaning against the guard (0)
DOWN_TOUCH = 0.002  # ... 'over the rail or a post': its outline overlaps their tops grown by this (touching a face)
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
OFF_TABLE_DROP = 0.010  # a baluster this far below the table top fell off the table (it need not be still)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a baluster that lands on its starting mark having moved this much further in one control
#                    step than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)


def even_centres(n: int) -> tuple[float, ...]:
    """The y of n balusters spread evenly between the posts, left to right: every opening (post to baluster and
    baluster to baluster) equal, (356 - 20 n) / (n + 1) mm."""
    gap = (RAIL_LEN - n * BAL_D) / (n + 1)
    return tuple(round(POST_FACE - (i + 1) * gap - (i + 0.5) * BAL_D, 9) + 0.0 for i in range(n))


# The guard the demo builds: five balusters, every opening 42.7 mm (four would leave 55.2 mm, wide enough for the
# ball however they were spaced).
DEMO_YS = even_centres(5)  # +0.1253, +0.0627, 0, -0.0627, -0.1253

# Colours (sRGB).
OAK = (0.55, 0.36, 0.20)
WHITE = (0.93, 0.93, 0.90)
