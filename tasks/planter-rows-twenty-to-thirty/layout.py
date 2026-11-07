"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A row-crop planter at 1:4. Its row units clamp to a toolbar, and the row spacing is measured along the toolbar from
its centre (the hitch) with a tape. Six rows at 20 in (508 mm) become four rows at 30 in (762 mm): both layouts
span 120 in, so the same toolbar carries them. At 1:4 the old pitch is 127 mm and the new one 190.5 mm. Rows are
symmetric about the centre; with an even row count no unit sits on the centre line, so the two middle rows are half
a spacing either side of it: the four rows go at +-95.25 and +-285.75 mm (the six started at +-63.5, +-190.5 and
+-317.5 mm). The two units left over are spares and go onto their stands.
"""

TOP = 0.7475  # station table top

# The toolbar: a fixed dark-grey steel bar along y in front of both arms, built as three boxes with one flush top:
# two grey halves and the red centre mark between them (10 mm wide, painted on a real bar).
BAR_X = 0.300  # its centre line
BAR_W = 0.040  # along x
BAR_H = 0.030
BAR_HALF_LEN = 0.350  # it runs y -0.350 .. +0.350
MARK_HALF = 0.005  # the red mark covers y -0.005 .. +0.005
BAR_TOP = TOP + BAR_H

# The hitch tongue: a fixed grey box on the midline butting the toolbar's face nearest the arms (where the tractor
# is). It only shows where the centre is: nothing goes on it.
HITCH_SIZE = (0.100, 0.030, 0.025)  # x 0.180 .. 0.280, y -0.015 .. +0.015
HITCH_X = BAR_X - BAR_W / 2 - HITCH_SIZE[0] / 2  # 0.230

# Two spare-unit stands: fixed light-grey boxes, one on each side, in front of the toolbar.
STAND_SIZE = (0.090, 0.090, 0.010)
STAND_X, STAND_Y = 0.170, 0.250  # centres at (0.170, +0.250) (left) and (0.170, -0.250) (right)
STAND_TOP = TOP + STAND_SIZE[2]
STANDS = (("stand_left", STAND_Y), ("stand_right", -STAND_Y))

# Six row units: loose green-painted wood blocks, long side along x (fore and aft, square to the bar), standing on
# the toolbar's top and overhanging it 15 mm front and back like a clamp bracket. 36 x 70 x 45 mm of wood at about
# 700 kg/m3 is 80 g.
UNIT_SIZE = (0.070, 0.036, 0.045)  # x (fore and aft), y (across: along the bar), z
UNIT_HALF = tuple(s / 2 for s in UNIT_SIZE)
UNIT_MASS = 0.080
OLD_PITCH = 0.127  # 20 in at 1:4
NEW_PITCH = 0.1905  # 30 in at 1:4
UNITS = (  # (name, y at the start): six rows at 20 in, symmetric about the mark
    ("unit_l1", 0.0635),
    ("unit_l2", 0.1905),
    ("unit_l3", 0.3175),
    ("unit_r1", -0.0635),
    ("unit_r2", -0.1905),
    ("unit_r3", -0.3175),
)
NAMES = tuple(n for n, _ in UNITS)
START = {n: (BAR_X, y) for n, y in UNITS}

# The four-row layout: on each side one unit half a spacing from the mark, the other one and a half spacings.
TARGETS = (NEW_PITCH / 2, 3 * NEW_PITCH / 2)  # 0.09525, 0.28575 from the mark's centre (y 0)

# Rule thresholds (scoring.py).
SEAT_TOL = 0.003  # 'resting on' the toolbar or a stand: the unit's lowest point within this of its top
UPRIGHT_DEG = 5.0  # 'set' on the bar / 'stands' on a stand: tilted at most this
SQUARE_DEG = 8.0  # 'set': its long side within this of x (square to the bar)
LINE_TOL = 0.012  # 'set': its centre within this of the bar's centre line in x
TIGHT = 0.010  # 0.5 per side: each unit's centre within this of its target distance from the mark
LOOSE = 0.025  # 0.25 per side: within this (under the 31.75 mm between the old and new positions)
MARK_COVER = MARK_HALF + UNIT_HALF[1]  # 0.023: a unit on the toolbar with its centre this close to the mark's
#                                        centre covers part of the mark: a row on the centre line, 0
OFF_TABLE_DROP = 0.010  # a unit whose lowest point is this far below the table top fell off the table: 0
STILL_WINDOW_S = 0.5  # 'still': its centre's path over the last 0.5 s ...
STILL_PATH = 0.005  # ... under 5 mm (a path, not the reported velocity, which chatters while a unit is gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a unit that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it

# Colours (sRGB).
STEEL = (0.25, 0.26, 0.28)
RED = (0.85, 0.10, 0.10)
STAND_GREY = (0.60, 0.60, 0.62)
GREEN = (0.15, 0.45, 0.20)
