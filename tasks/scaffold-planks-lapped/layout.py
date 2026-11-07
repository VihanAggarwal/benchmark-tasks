"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A 1:8 scale model of two bays of a scaffold's working platform: 6 in = 19 mm, 12 in = 38 mm, a 1.52 m (5 ft) bay =
190 mm, a 2.0 m (6.6 ft) board = 250 mm. The run of the platform is along y.
"""

TOP = 0.7475  # station table top

# The three putlogs (transoms): fixed galvanised steel bars, long side along x, 190 mm apart on centre along y.
PUTLOG_SIZE = (0.120, 0.020, 0.020)  # along x, along y, height
PUTLOG_X = 0.260  # each spans x 0.200 to 0.320
PUTLOGS = (  # (name, y of its centre line)
    ("putlog_left", 0.190),
    ("putlog_middle", 0.000),
    ("putlog_right", -0.190),
)
PUTLOG_Y = {name: y for name, y in PUTLOGS}
PUTLOG_HALF_W = PUTLOG_SIZE[1] / 2
PUTLOG_X0 = PUTLOG_X - PUTLOG_SIZE[0] / 2  # 0.200: the putlogs' ends, the platform's width
PUTLOG_X1 = PUTLOG_X + PUTLOG_SIZE[0] / 2  # 0.320
PUTLOG_TOP = TOP + PUTLOG_SIZE[2]  # 20 mm above the table
BAYS = {  # bay -> its two putlogs
    "left": ("putlog_left", "putlog_middle"),
    "right": ("putlog_middle", "putlog_right"),
}
MIDDLE = "putlog_middle"

# The two scaffold boards: loose, orange-painted pine, lying flat on the table along x at the start, one on each
# arm's side. Box size is given as they lie at the start (long side along the body's x axis).
BOARD_SIZE = (0.250, 0.040, 0.016)  # length, width, thickness
BOARD_LEN, BOARD_W, BOARD_T = BOARD_SIZE
BOARD_MASS = 0.080  # pine, about 500 kg/m3
BOARD_START_X = 0.245
BOARDS = (  # (name, y of its centre at the start)
    ("board_left", 0.320),
    ("board_right", -0.320),
)
BOARD_START_Y = {name: y for name, y in BOARDS}

# The nominal end state (what the demo builds; the scorer accepts either board in either bay, either on top):
# the right board flat on the right and middle putlogs, centred at y -0.095 (ends at y -0.220 and +0.030, 30 mm past
# each centre line); the left board centred at y +0.095, its outer end on the left putlog and its inner end on the
# right board's end, sloping 5.4 degrees (16 mm down over the 170 mm from the right board's end to the left putlog's
# outer edge). The two overlap by 60 mm along the run, with the middle putlog's centre line inside the lap.
RUN_X = PUTLOG_X  # both boards' centre lines over the putlogs' middle
LOWER_Y = -0.095
UPPER_Y = 0.095
LOWER_BOTTOM = PUTLOG_TOP  # the lower board's underside
LOWER_TOP = LOWER_BOTTOM + BOARD_T  # 36 mm above the table: where the upper board's inner end rests

# Rule thresholds (scoring.py).
REST_TOL = 0.002  # 'lies on': the board's underside within this of the putlog's (or the other board's) top,
REST_MIN_AREA = 2.0e-5  # over an overlap (seen from above) of at least 20 square mm: a real crossing, not a touch
OVERHANG_MIN = 0.019  # 6 in at 1:8 (29 CFR 1926.451(b)(4))
OVERHANG_MAX = 0.038  # 12 in at 1:8 ((b)(5)(i)); more is a trap end: 0 for the trial
LIMIT_EPS = 1e-6  # comparisons at the overhang limits allow 0.001 mm of floating-point rounding
LAP_MIN = 0.038  # 12 in ((b)(7)); automatic when both boards bear (each passes the middle centre line by >= 19 mm)
HEADING_DEG = 5.0  # 'along the run': long side within this of y
MEASURE_HEADING_DEG = 45.0  # a board turned further than this from the run (along a putlog) planks nothing
TILT_MAX_DEG = 8.0  # bearing boards are tilted at most this (the lapped board slopes about 5.4 deg)
IN_LINE_TOL = 0.010  # the two boards' centre lines within this of each other across the run (x), at the middle
CLEAR_OF_TABLE = 0.005  # 'on the scaffold': the board's lowest point at least this above the table top
STILL_WINDOW_S = 0.5  # 'still': neither end of the board moved more than STILL_TOL over the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a board that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)
START_HEADING_DEG = 10.0

# Colours (sRGB).
GALVANISED = (0.60, 0.62, 0.64)  # dull grey steel
BOARD_ORANGE = (0.90, 0.45, 0.10)  # stands out from the oak-coloured station table and the grey putlogs
