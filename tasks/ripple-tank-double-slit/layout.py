"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
'world'.

A small ripple tank stands on the table, centred on the midline: a pale acrylic floor (260 x 200 mm, 3 mm thick)
inside four grey PVC walls 15 mm tall, a fixed white dipper bar lying across it near the arms and a grey foam beach
along the far wall. Three loose black barrier bars stand on the table outside it: two long ones (70 mm), one in front
of each arm, and a short one (30 mm) in front of the left arm. Set up, the three stand in one line across the tank,
120 mm beyond the dipper and parallel to it, the long ones closed against the side walls and the short one in the
middle: 70 + 30 + 70 mm of bar in the tank's 200 mm leaves 30 mm, two slits of 15 mm whose centres are 45 mm apart.
"""

TOP = 0.7475  # station table top (world z)

# -- the tank (all fixed) --------------------------------------------------------------------------------------
TANK_X0, TANK_X1 = 0.150, 0.410  # the end walls' inside faces (near, far)
TANK_HALF = 0.100  # the side walls' inside faces at y +-0.100: the tank is 200 mm wide inside
TANK_X = (TANK_X0 + TANK_X1) / 2  # 0.280
FLOOR_T = 0.003  # the floor sheet: x 0.150-0.410, y +-0.100, 3 mm thick, lying on the table
FLOOR_TOP = FLOOR_T  # the floor's top, above the table top
WALL_T = 0.008  # every wall's thickness
WALL_H = 0.015  # every wall's height above the table (12 mm above the floor)
SIDE_WALL_LEN = TANK_X1 - TANK_X0  # 0.260: the side walls run between the end walls' inside faces
SIDE_WALL_Y = TANK_HALF + WALL_T / 2  # 0.104
END_WALL_LEN = 2 * (TANK_HALF + WALL_T)  # 0.216: across both side walls' ends
END_WALL_X = (TANK_X0 - WALL_T / 2, TANK_X1 + WALL_T / 2)  # 0.146, 0.414
# The dipper (the vibrating bar that makes straight waves): fixed, lying on the floor along y.
DIPPER_X = 0.180
DIPPER_SIZE = (0.010, 0.180, 0.012)
# The beach (foam that soaks up the waves at the far wall): fixed, lying on the floor along the far wall.
BEACH_SIZE = (0.025, 2 * TANK_HALF, 0.006)
BEACH_X = TANK_X1 - BEACH_SIZE[0] / 2  # 0.3975: x 0.385-0.410

# -- the barrier bars (loose, black-painted aluminium) -----------------------------------------------------------
LONG_SIZE = (0.020, 0.070, 0.025)  # thickness (x), length (y), height (z) as they start
SHORT_SIZE = (0.020, 0.030, 0.025)
LONG_MASS = 0.095  # 35 cm3 of aluminium at 2.7 g/cm3
SHORT_MASS = 0.041  # 15 cm3
BAR_NAMES = ("bar_a", "bar_b", "bar_c")
BAR_SIZE = {"bar_a": LONG_SIZE, "bar_b": LONG_SIZE, "bar_c": SHORT_SIZE}
BAR_MASS = {"bar_a": LONG_MASS, "bar_b": LONG_MASS, "bar_c": SHORT_MASS}
# Where each starts: standing on the table on a 20 mm x length face, its length along y (yaw 0). a and c in front
# of the left arm, b in front of the right arm.
BAR_START = {
    "bar_a": (0.220, 0.240),
    "bar_b": (0.220, -0.240),
    "bar_c": (0.140, 0.200),
}

# -- the rules (scoring.py) --------------------------------------------------------------------------------------
LINE_DIST_MIN, LINE_DIST_MAX = 0.110, 0.130  # every bar's centre this far beyond the dipper's centre line (x 0.29-0.31)
INLINE_MAX = 0.006  # the three bars' centres within this of each other along x
AXIS_DEG = 5.0  # each bar's length within this of y (across the tank, parallel to the dipper)
TILT_DEG = 10.0  # each bar's 25 mm axis within this of vertical (standing on a 20 mm x length face)
ON_FLOOR = 0.002  # each bar's lowest point within this of the floor's top (above or below)
INSIDE_TOL = 0.003  # each bar's outline inside the tank's inside faces, with this much slack (contact, not a prop)
SLIT_MIN, SLIT_MAX = 0.010, 0.020  # S: both slits this wide
SEP_MIN, SEP_MAX = 0.040, 0.050  # C: the slits' centres this far apart ...
MID_MAX = 0.005  # ... and their midpoint within this of the tank's centre line (y 0)
WALL_GAP_MAX = 0.005  # W: both wall gaps at most this
EPS = 1e-6  # m: limits are compared with a micrometre of slack (a slit of exactly 20.000 mm is in the window)
STILL_TOL = 0.002  # success: neither end of any bar moved more than this ...
STILL_WINDOW_S = 0.5  # ... over the last 0.5 s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a bar moving this far in one control step was teleported: a new trial began ...
START_TOL = 0.045  # ... if every bar is then within this of its start (covers a seeded reset's jitter)

# Where the demo sets the bars: every centre on x 0.300 and the short bar centred. Each long bar is first set down
# 6 mm in from its wall (clear of the wall's top whatever the arm's sideways error), then slid out, its bottom below
# the wall's top, toward a point 1.5 mm past the wall's inside face, so the wall's face stops it: it ends against the
# wall (wall gap 0) unless the arm falls more than 6.5 mm short, and an arm that overshoots only presses it on the
# wall (about 0.5 N at the arm's compliance). Against both walls the slits are 15 mm wide, 45 mm apart centre to
# centre, and the short bar may land up to 5 mm off the centre line and still score 1.0. Offline there are no
# contacts: the bar is drawn 1.5 mm into the wall, which the scorer reads as touching it (INSIDE_TOL 3 mm) and the
# offline checker still drops onto the floor (it lands a part on what it overlaps by more than 2 mm).
LINE_X = DIPPER_X + 0.120  # 0.300
SET_WALL_GAP = 0.006
PUSH_PAST = 0.0015
SET_Y = {
    "bar_a": TANK_HALF - SET_WALL_GAP - LONG_SIZE[1] / 2,  # +0.059
    "bar_b": -(TANK_HALF - SET_WALL_GAP - LONG_SIZE[1] / 2),  # -0.059
    "bar_c": 0.0,
}
PLACE_Y = {  # where the demo's tool point ends each bar's move (the bar's centre if nothing stopped it)
    "bar_a": TANK_HALF + PUSH_PAST - LONG_SIZE[1] / 2,  # +0.0665
    "bar_b": -(TANK_HALF + PUSH_PAST - LONG_SIZE[1] / 2),  # -0.0665
    "bar_c": 0.0,
}

# Colours (sRGB).
ACRYLIC_PALE = (0.86, 0.91, 0.95)  # clear acrylic over a white screen sheet
PVC_GREY = (0.42, 0.44, 0.47)
FOAM_GREY = (0.28, 0.28, 0.30)
DIPPER_WHITE = (0.96, 0.96, 0.96)
BAR_BLACK = (0.06, 0.06, 0.07)
