"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

Two short vineyard trellis rows at about 1:12. A row ends in a tall end post; the next post along the row, a
shorter line post, stands farther from the arms. The end post is held against the trellis wires' pull by a brace
wire from its top down to an anchor in the ground, set in line with the row beyond the end post, as far from the
end post as the end post stands tall (so the brace runs at about 45 degrees). The two end posts are different
heights (100 and 150 mm here: 1.2 and 1.8 m), so each row's distance is read from its own end post. Nothing is
marked: the line is extended past the end post from the row's two posts.
"""

import math

TOP = 0.7475  # station table top

# Four posts: fixed square wooden posts, 25 x 25 mm (thicker than scale, so they show in the top photo), each turned
# so that two of its sides run along its row.
POST_SIDE = 0.025
# Each row's line post stands 20 mm (0.24 m at 1:12) lower than its end post, which rises above the wires to carry
# the brace: 80 mm on the left row, 130 mm on the right. Neither is the anchor distance (20 mm short on each row).
LINE_POST_H = {"left": 0.080, "right": 0.130}
# In each row the line post stands 126 mm farther from the arms and 34 mm further right than the end post: the two
# rows are parallel, running forward and to the right at 15.1 degrees to x, posts 130.5 mm apart centre to centre.
ROW_STEP = (0.126, -0.034)
ROWS = (  # (row, end post centre (x, y), end post height H, line post centre (x, y))
    ("left", (0.320, 0.200), 0.100, (0.320 + ROW_STEP[0], 0.200 + ROW_STEP[1])),
    ("right", (0.370, -0.180), 0.150, (0.370 + ROW_STEP[0], -0.180 + ROW_STEP[1])),
)
ROW_NAMES = tuple(r for r, *_ in ROWS)
POST_GAP = math.hypot(*ROW_STEP)  # 0.130507: end post to line post, centre to centre
ROW_YAW = math.atan2(ROW_STEP[1], ROW_STEP[0])  # -0.2635 rad (-15.10 deg): the posts' and wires' heading
ROW_ANGLE_DEG = -math.degrees(ROW_YAW)  # 15.10


def row_axis(row: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    """(end post centre, unit vector along the row's line pointing away from the line post, end post height H)."""
    for name, end, h, line in ROWS:
        if name == row:
            d = (end[0] - line[0], end[1] - line[1])
            n = math.hypot(*d)
            return end, (d[0] / n, d[1] / n), h
    raise KeyError(row)


def anchor_point(row: str) -> tuple[float, float]:
    """Where the row's anchor goes (not marked): on the row's line, H beyond the end post's centre."""
    end, u, h = row_axis(row)
    return end[0] + h * u[0], end[1] + h * u[1]


TARGETS = {r: anchor_point(r) for r in ROW_NAMES}  # left (0.22345, 0.22605), right (0.22518, -0.14092)

# Two trellis wires: thin fixed boxes along each row between its two posts' facing sides, 70 mm above the table
# (under both posts' tops), 0.5 mm short of each post so they do not overlap them.
WIRE_SECTION = 0.005
WIRE_Z = 0.070  # the wire's centre above the table
WIRE_LEN = POST_GAP - POST_SIDE - 2 * 0.0005  # 0.104507

# Two anchor blocks: loose concrete-grey blocks standing on a 30 x 30 mm face, one in front of each arm. A
# 30 x 30 x 40 mm block of concrete (about 2.2 g/cm3) weighs 80 g. Their top and bottom faces are alike.
ANCHOR_SIZE = (0.030, 0.030, 0.040)
ANCHOR_HALF = tuple(s / 2 for s in ANCHOR_SIZE)
ANCHOR_MASS = 0.080
ANCHORS = (("anchor_left", (0.150, 0.330)), ("anchor_right", (0.150, -0.330)))
NAMES = tuple(n for n, _ in ANCHORS)
START = dict(ANCHORS)

# Rule thresholds (scoring.py).
SEAT_TOL = 0.003  # 'set': the block's lowest point within this of the table top ...
UPRIGHT_DEG = 10.0  # ... and its 40 mm axis tilted at most this from vertical (either end up)
TIGHT_OFFSET = 0.010  # 0.5 per row: centre within this of the row's line ...
TIGHT_REACH = 0.010  # ... and its distance beyond the end post within this of H
LOOSE_OFFSET = 0.020  # 0.25 per row: within this of the line (the 'straight back along x' mistake is 26 / 39 mm)
LOOSE_REACH = 0.015  # ... and within this of H (every wrong height named in the scoring is 20 mm or more off)
OFF_TABLE_DROP = 0.010  # a block whose lowest point is this far below the table top fell off the table: 0
STILL_WINDOW_S = 0.5  # 'still': its centre's path over the last 0.5 s ...
STILL_PATH = 0.005  # ... under 5 mm (a path, not the reported velocity, which chatters while a block is gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a block that lands on its starting spot having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting spot': within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
WOOD = (0.36, 0.27, 0.18)  # weathered post wood
WIRE = (0.72, 0.73, 0.75)  # galvanised wire
CONCRETE = (0.55, 0.55, 0.55)
