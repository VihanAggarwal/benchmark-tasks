"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The glass is gripped from the inside: both grippers enter it with their fingers nearly closed, then open them
until the fingers' outer faces press its inner walls. The numbers that decide whether that works are the
fingers' (yam_arena's finger collision meshes, as measured for the program's earlier finger-zone tasks): outside
the gap between the pads a finger is 6.4 mm thick at its tips and 10 mm thick 19 to 38 mm above them, 27 mm wide at
the tips and 40 mm wide 19 to 38 mm up; the finger carriages stick out 48 mm (closed) to 71 mm (open) from the
gripper's axis 70 to 94 mm above the tips, and the gripper's body starts about 83 mm above them. So:

- across the closing axis the 40 mm fingers need the 62 mm opening: 11 mm clear on each side;
- along it, nearly closed (10 mm apart) the fingers span 2 x (5 + 10) = 30 mm: 16 mm clear of each inner wall;
- their outer faces (10 mm outside the gap) touch walls 62 mm apart at a 42 mm gap; commanding 50 mm (4 mm more per
  finger) presses each wall with about 20 N (5000 N/m x 4 mm), against 1.8 N of weight;
- with the tips 42 mm below the rim, the fingers' 10 mm-thick band (19-38 mm up) is 4-23 mm inside the glass and
  only 4 mm of the unmeasured finger roots above it, the carriages stay 28 mm above the rim and the gripper's body
  41 mm above it, and the tips 32 mm above the inner floor.
"""

TOP = 0.7475  # station table top

# The glass (a mechkit open box with a floor, loose): a square acrylic rocks glass, open at the top.
GLASS_OUTER = 0.074  # outside width, both ways
GLASS_H = 0.080  # height, bottom face to rim
GLASS_WALL = 0.006
GLASS_FLOOR = 0.006
GLASS_MASS = 0.18  # 154 cm3 of acrylic (1.19 g/cm3)
GLASS_HALF = GLASS_OUTER / 2  # 0.037
INNER_HALF = GLASS_HALF - GLASS_WALL  # 0.031: the opening is 62 x 62 mm, 74 mm deep
GLASS_HALF_H = GLASS_H / 2

# Places (centres in x, y). Each arm works only on its own side: the glass starts 38 cm left of the midline, 7.5 cm
# outside the left arm's base line, and the evidence box stands 38 cm right of it, 7.5 cm outside the right arm's.
START_X, START_Y = 0.26, 0.38
SHEET_X, SHEET_Y = 0.27, 0.00
BOX_X, BOX_Y = 0.26, -0.38

# The start mark: a thin yellow card the glass stands on at the start (fixed).
MARK_SIZE = (0.110, 0.110, 0.001)
MARK_TOP = TOP + MARK_SIZE[2]
# The exam sheet: a white card on the midline (fixed), the only place both arms reach.
SHEET_SIZE = (0.140, 0.140, 0.002)
SHEET_TOP = TOP + SHEET_SIZE[2]
# The evidence box: four fixed brown cardboard walls, the table top its floor.
BOX_INNER = 0.105  # 15.5 mm round the glass on every side
BOX_WALL = 0.005
BOX_HEIGHT = 0.055
BOX_INNER_HALF = BOX_INNER / 2
BOX_OUTER_HALF = BOX_INNER_HALF + BOX_WALL
BOX_TOP = TOP + BOX_HEIGHT

# The grippers (yam_arena): rigstate reports the opening as 0 (closed) .. 1 (pads 93.9 mm apart).
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap, lowest 38 mm (6.4 mm at the tips, 10 mm 19-38 mm up)
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there (27 mm at the tips, 40 mm higher)
FINGER_LEN = 0.038  # the finger zone runs from the tips this far back up the gripper
FINGER_REACH = 0.070  # tips to the carriages: a finger in the glass is in it this whole length wherever below the rim
CONTACT_GAP = 2 * (INNER_HALF - FINGER_THICK)  # 0.042: the fingers' outer faces touch both inner walls
GRIP_GAP = CONTACT_GAP + 0.008  # 0.050: commanded 4 mm further per finger, about 20 N on each wall
NEAR_CLOSED_GAP = 0.010  # how far apart the fingers are when they go in and come out

# Rule thresholds (scoring.py).
HEADING_STEP_DEG = 5.0  # the scorer cannot see a gripper's turn about its axis: it tries 0, 5, .., 175 deg
INSIDE_TOL = 0.005  # fingers 'inside the opening' may reach this far into the 6 mm inner walls (fingers pressing them)
OUTSIDE_CLEAR = 0.010  # ... or, at every turn, they must keep this far from the glass (a box 20 mm wider): 'away'
BELOW_BASE = 0.010  # ... and this far under its bottom face (the rule covers the glass up to its rim)
BOX_TOUCH = 0.002  # the glass within this of a box wall (plan view) while not above it: touching the box
BOX_ABOVE = 0.002  # 'above a wall': the glass's lowest point at least this far over the wall's top
SET_DOWN = 0.004  # the glass's lowest point under this above the table top: it is standing on (or near) something
TILT_MAX_DEG = 10.0  # tipped
UPRIGHT_DEG = 5.0  # standing upright (the 1.0, 0.75 and 0.5 tiers)
REST_TOL = 0.003  # 'standing on' a surface: the glass's lowest point within this of its top
DROP_SPEED = 0.45  # m/s down without a gripper carrying it down: dropped (a 1 cm fall reaches 0.44 m/s, 55 mm 1.04)
FALL_SLACK = 0.15  # 'carrying it down': a gripper's tips in the glass, moving down no more than this slower than it
LIFT_HEIGHT = 0.020  # 'lifted': the glass's lowest point this far above the table top
LET_GO_ABOVE = 0.020  # a gripper has let go: its tip point this far above the rim ...
LET_GO_AWAY = 0.060  # ... or this far from the glass's axis horizontally
OFF_TABLE_DROP = 0.010  # lowest point this far below the table top: off the table
TABLE_X = (-0.19, 0.91)  # the station table's extent (yamkin.TABLE_BOX)
TABLE_Y = (-0.65, 0.65)
ARMS_MISSING_MAX = 0.01  # the arms unreadable on more than this share of the steps: the trial scores 0
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.05  # the glass moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its start: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
GLASS_BLUE = (0.70, 0.85, 0.95)  # pale blue tinted clear acrylic
MARK_YELLOW = (0.95, 0.80, 0.10)  # an evidence marker card
SHEET_WHITE = (0.99, 0.99, 0.99)
BOX_BROWN = (0.60, 0.44, 0.28)
