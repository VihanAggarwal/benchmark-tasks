"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A strawberry plasticulture bed at 1:3 (100 mm here is about 12 in). Strawberries go on raised beds under black
plastic mulch as a double row either side of the drip tape on the bed's centre line, the plants of one row level
with the middle of the gaps in the other (staggered), so every plant has the most room. The transplanter set the far
row (six plugs 100 mm apart, 50 mm beyond the tape); the near row is planted by hand: one plug level with the middle
of each of the five gaps, mirrored across the tape, so 50 mm on this side of it. None of the five places is marked:
along the row they follow from the far row's gaps, across the bed from the tape.
"""

TOP = 0.7475  # station table top

# The bed: black plastic mulch over soil, 220 mm across (x) and 600 mm long (y), 25 mm tall, centred at (0.300, 0).
# Built as three fixed boxes with one flush top: the near half, the drip tape along the centre line (16 mm wide,
# shown blue so it can be seen; a real tape runs under the mulch) and the far half.
BED_X = 0.300  # the bed's centre line, where the drip tape runs
BED_W = 0.220  # across, along x: x 0.190 .. 0.410
BED_HALF_LEN = 0.300  # along y: -0.300 .. +0.300
BED_H = 0.025
BED_TOP = TOP + BED_H
TAPE_W = 0.016  # x 0.292 .. 0.308

# Two plug trays: flat light-grey plastic pads fixed to the table, one beside each arm's base.
PAD_SIZE = (0.160, 0.050, 0.005)
PAD_X, PAD_Y = 0.200, 0.360  # centres at (0.200, +0.360) (left) and (0.200, -0.360) (right)
PAD_TOP = TOP + PAD_SIZE[2]
PADS = (("tray_left", PAD_Y), ("tray_right", -PAD_Y))

# A plug: a strawberry plug plant (its root ball and crown), an upright cylinder 30 mm across and 45 mm tall, 25 g
# (a 30 x 45 mm hardwood dowel at about 0.78 g/cm3, the real build). A plain cylinder: the same either end up.
PLUG_R = 0.015
PLUG_H = 0.045
PLUG_MASS = 0.025

# The far row, set by the machine: six plugs on the bed 50 mm beyond the tape, 100 mm apart.
FAR_X = 0.350
FAR = (  # (name, y), left to right
    ("far_1", 0.25),
    ("far_2", 0.15),
    ("far_3", 0.05),
    ("far_4", -0.05),
    ("far_5", -0.15),
    ("far_6", -0.25),
)
# The supply: three plugs on each tray (six for five places: one spare).
SUPPLY_X = (0.140, 0.200, 0.260)
SUPPLY = (
    ("plug_l1", SUPPLY_X[0], PAD_Y),
    ("plug_l2", SUPPLY_X[1], PAD_Y),
    ("plug_l3", SUPPLY_X[2], PAD_Y),
    ("plug_r1", SUPPLY_X[0], -PAD_Y),
    ("plug_r2", SUPPLY_X[1], -PAD_Y),
    ("plug_r3", SUPPLY_X[2], -PAD_Y),
)
FAR_NAMES = tuple(n for n, _ in FAR)
FAR_SPOTS = tuple((FAR_X, y) for _, y in FAR)  # where the far row stands: each must still have a plug on it
SUPPLY_NAMES = tuple(n for n, _, _ in SUPPLY)
NAMES = FAR_NAMES + SUPPLY_NAMES  # the order the scorer reads them in
START = {**{n: (FAR_X, y, BED_TOP + PLUG_H / 2) for n, y in FAR},
         **{n: (x, y, PAD_TOP + PLUG_H / 2) for n, x, y in SUPPLY}}

# The near row: level with the middle of each gap in the far row, mirrored across the tape.
NEAR_X = 2 * BED_X - FAR_X  # 0.250
_far_ys = sorted(y for _, y in FAR)
TARGET_YS = tuple(sorted(((a + b) / 2 for a, b in zip(_far_ys, _far_ys[1:])), reverse=True))  # +0.20 .. -0.20
TARGETS = tuple((NEAR_X, y) for y in TARGET_YS)
POINTS_EACH = 0.2

# Rule thresholds (scoring.py).
SEAT_TOL = 0.003  # 'planted': the plug's lowest point within 3 mm of the bed's top
UPRIGHT_DEG = 10.0  # 'planted': its axis within 10 degrees of vertical (either end up)
FILL_TOL = 0.015  # a target is filled by a planted plug whose centre is within 15 mm of it (in plan)
FAR_SPOT_TOL = 0.010  # a far-row spot is kept by a planted plug whose centre is within 10 mm of it (in plan);
#                       a spot left without one (a far plant knocked over, shoved, lifted or taken) scores 0
OFF_TABLE_DROP = 0.010  # a plug whose lowest point is this far below the table top fell off the table: 0
STILL_WINDOW_S = 0.5  # 'still' (for the early end): each plug's centre path over the last 0.5 s ...
STILL_PATH = 0.005  # ... under 5 mm (a path, not the reported velocity, which chatters while a plug is gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a plug that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it

# Colours (sRGB).
MULCH = (0.08, 0.08, 0.09)
TAPE_BLUE = (0.15, 0.35, 0.80)
TRAY_GREY = (0.62, 0.62, 0.64)
LEAF = (0.20, 0.55, 0.20)
