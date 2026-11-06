"""Every dimension and threshold of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below marked "above the table" are added to TOP.

The work runs along y, 40 cm in front of the arm bases: a long frame plate (two plies of a seam, laminated, as in a
quilting frame moved by hand under a needle with the feed dogs dropped) on a fixed machine bed, sliding toward -y
(toward the right arm) on its rail. Its handle starts just outside the left arm's base line and three pins stand in
it near its leading end, in front of a fixed needle head on the right arm's side.
'Feed' t is how far the plate has slid (0 at the start, positive toward -y). A pin's 'd' is its centre's y minus the
needle line's y: positive before the needle, falling by the feed.

Why the layout is this long (tools/offline/yamkin, position and orientation IK, arm capsules clear of the table and
the own base): with its gripper tilted up to 40 degrees from straight down in any direction, an arm's tool point
reaches about 0.68 m from its base at grip height (at x 0.40 the left arm reaches y -0.248 at most, straight down
only about the midline; the arms are mirror images). So the pull window (y -0.33 .. -0.29) and the dish are out of
the left arm's reach, and the handle's starting place (y +0.38) out of the right arm's, even tilted 60 degrees.
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81

# -- the machine bed (the work assembly's fixed base) ---------------------------------------------------------------
X_SEAM = 0.40  # the seam line, the bed's and the plate's centre line (x)
BED_W = 0.120  # along x
BED_T = 0.012
BED_Y0, BED_Y1 = -0.555, 0.415  # the bed's ends (y): the plate stays on it over its whole travel
BED_TOP = BED_T  # above the table

# -- the work: a feed plate on a prismatic joint along -y (travel 0 .. FEED_MAX) -------------------------------------
PLATE_W = 0.070  # along x
PLATE_T = 0.010
PLATE_TOP = BED_TOP + PLATE_T  # 0.022 above the table
PLATE_Y0, PLATE_Y1 = -0.355, 0.405  # leading (toward the needle) and trailing ends at feed 0: 760 mm long
PLATE_MASS = 0.30  # 517 cm3 of PVC foam board (0.5 g/cm3, 0.26 kg) and the 36 cm3 handle (ABS, 0.04 kg)
FEED_MAX = 0.180  # the rail's travel
FEED_DAMPING = 2.0  # N s/m, no spring, no joint friction
POCKET = 0.028  # square pockets for the pins, 6 mm deep in the 10 mm plate: 5 mm clear of a pin on every side
POCKET_DEPTH = 0.006
POCKET_FLOOR = PLATE_TOP - POCKET_DEPTH  # 0.016 above the table
SEAM_LINE_W = 0.003  # the blue seam line (x), 0.5 mm proud of the plate
SEAM_LINE_T = 0.0005

# -- the needle line and the pull window ------------------------------------------------------------------------------
Y_NEEDLE = -0.360
WINDOW = (0.030, 0.070)  # a pin is pulled in time when d at the pull is within this (and it never came closer)
STRIKE_GAP = 0.006  # any part of a pin this close to the needle line (an upright pin's centre 15 mm from it) is a strike

# -- the handle: a black block on the plate's trailing end (part of the plate) ----------------------------------------
HANDLE = (0.030, 0.030, 0.040)
HANDLE_Y = 0.380  # its centre at feed 0
HANDLE_Z = PLATE_TOP + HANDLE[2] / 2  # 0.042 above the table
HOLD_MARGIN = 0.005  # a gripper holds the handle: its finger-tip point over the handle's footprint grown by this,
#                      and below the handle's top ...
HOLD_OPENING = 0.6  # ... with the gripper less open than this (fingers under 56 mm apart)

# -- the pins: loose upright cylinders, one standing in each pocket on the seam line ---------------------------------
PIN_R = 0.009
PIN_H = 0.040
PIN_MASS = 0.027  # 10.2 cm3 of aluminium rod, painted red
PIN_D0 = {"pin_1": 0.055, "pin_2": 0.105, "pin_3": 0.155}  # d at feed 0 (pin 1 starts inside the window)
PINS = tuple(PIN_D0)
PIN_START = {n: (X_SEAM, Y_NEEDLE + d, TOP + POCKET_FLOOR + PIN_H / 2) for n, d in PIN_D0.items()}  # world centres, feed 0
OUT_CLEAR = 0.003  # a pin is 'in the plate' while over it with its lowest point under this above the plate's top
PIN_HOLD_OPENING = 0.30  # a gripper holds a pin: less open than this (fingers under 28 mm apart) ...
PIN_HOLD_GAP = 0.004  # ... with its finger-tip point within this of the pin's box (closed fingers pushing a pin from
#                       the side keep that point 6 mm or more from it: they are 6.4 mm thick at the tips)

# -- the needle head: a fixed bridge over the bed with the needle hanging from its cross bar's near edge -------------
# The bridge stands just beyond the needle line (its near face on the line), so the right gripper working on a pin
# 55 mm before the line keeps its fingers 30 mm or so clear of it (the arm's collision capsules, which are rounder and
# larger than the fingers, 12 mm).
POST = (0.010, 0.010)  # x, y
POST_GAP = 0.002  # each post 2 mm outside the bed's side
POST_DX = BED_W / 2 + POST_GAP + POST[0] / 2  # 0.067 from the seam line
HEAD_Y = Y_NEEDLE - POST[1] / 2  # the posts' and the bar's centre line (y)
BAR_Z0, BAR_Z1 = 0.075, 0.090  # the cross bar's underside and top, above the table
BAR_LEN = 2 * POST_DX + POST[0]  # 0.144: over both posts
HEAD_HALF = BAR_LEN / 2  # 0.072: the needle head spans the seam line +- this (the posts' outer faces)
NEEDLE_R = 0.002
NEEDLE_TIP = PLATE_TOP + 0.008  # 0.030: 8 mm above the plate

# -- marks on the bed's two side strips: the needle line (red) and the pull window (a yellow band, 40 mm long) --------
MARK_W = 0.020  # across each 25 mm strip of bed beside the plate (x)
MARK_T = 0.001
NEEDLE_MARK = 0.006  # the red mark's length along y, centred on the needle line
MARK_DX = (BED_W / 2 + PLATE_W / 2) / 2  # 0.0475: centred on each strip

# -- the pin dish: a fixed open tray on the right arm's side, beyond the needle line, long along x ------------------
# The demo stands the pins in it 45 mm apart along x and lets go of each with the fingers closing along y (yaw 0): a
# gripper opens beside a pin without touching its neighbours (its fingers are 40 mm wide across the closing axis:
# 16 mm clear of the next pin). Its floor's nearest corner is 0.69 m from the left arm's base (the demo's places 0.73
# to 0.76 m), out of that arm's reach tilted up to 40 degrees.
DISH_X, DISH_Y = 0.240, -0.400
DISH_INNER = (0.140, 0.070)  # x, y
DISH_WALL = 0.006
DISH_FLOOR_T = 0.004
DISH_RIM = 0.015  # wall height above the table
DISH_HALF = (DISH_INNER[0] / 2, DISH_INNER[1] / 2)
DISH_SPOTS = {"pin_1": -0.045, "pin_2": 0.0, "pin_3": 0.045}  # where the demo stands each pin (x from the centre)

# -- the table (station) ----------------------------------------------------------------------------------------------
TABLE_X = (-0.19, 0.91)
TABLE_Y = (-0.65, 0.65)
OFF_TABLE_DROP = 0.010  # a pin whose lowest point is this far below the table top has fallen off it

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening 0 (closed) .. 1 (open) --------
FULL_GAP = 0.0939
RELEASE_OPENING = 0.9  # a gripper this open (fingers 84.5 mm apart) holds nothing
LET_GO_GAP = 0.020  # ... or its finger-tip point this far from the handle and every pin

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------------
FEED_START = 0.045  # the seam has started (pin 1 is struck from feed 40 mm: this needs it out of the plate)
FEED_DONE = 0.160  # the seam is complete: 5 mm past where pin 3 started (the demo feeds 0.178)
UNHELD_TRAVEL = 0.005  # the work may move at most this, in total, while no gripper holds its handle
TRAVEL_STEP = 0.0002  # travel is counted in steps of this (a ruler's reading; resting jitter adds nothing)
STILL_SPEED = 0.010  # m/s for the pins
STILL_SPIN = 0.5  # rad/s
FEED_STILL = 0.005  # m/s for the work
RESET_JUMP = 0.05  # a pin moving this much further in one step than its velocity allows was teleported
START_TOL = 0.010  # ... onto its starting place: within this of it (the pins are not jittered: object_jitter=0)

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
BED_GREY = (0.62, 0.64, 0.66)  # aluminium
PLATE_CREAM = (0.93, 0.90, 0.80)  # PVC foam board standing in for two plies of cloth
SEAM_BLUE = (0.15, 0.35, 0.85)
HANDLE_BLACK = (0.08, 0.08, 0.09)
PIN_RED = (0.85, 0.12, 0.12)
HEAD_GREY = (0.35, 0.36, 0.38)
NEEDLE_STEEL = (0.80, 0.81, 0.83)
DISH_BLACK = (0.12, 0.12, 0.13)
MARK_RED = (0.90, 0.10, 0.10)
MARK_YELLOW = (0.95, 0.80, 0.10)
