"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top.

Everything is fixed to the table except the ten key pins. Every fixed part is a box; boxes meet face to face
(butt joints), none overlaps another.
"""

TOP = 0.7475  # station table top

# -- the two training plugs ------------------------------------------------------------------------------------
# An enlarged model of a pin-tumbler plug taken out of its shell: a 186 x 50 x 60 mm block along x with four
# open-topped 34 x 34 mm chambers in a row. Chamber 1 is nearest the robot (and the key's bow).
SIDES = ("left", "right")
PLUG_Y = {"left": 0.240, "right": -0.240}  # plug centre lines (the left plug in front of the left arm)
PLUG_X0, PLUG_X1 = 0.207, 0.393  # near and far ends
PLUG_LEN = PLUG_X1 - PLUG_X0  # 0.186
PLUG_X = (PLUG_X0 + PLUG_X1) / 2  # 0.300
PLUG_W = 0.050
PLUG_H = 0.060
PLUG_TOP_Z = TOP + PLUG_H  # the shear line: every key pin's top must end flush with it
SIDE_WALL_T = 0.008  # side walls 186 x 8 x 60 mm, centred 21 mm either side of the plug's centre line
SIDE_WALL_OFF = PLUG_W / 2 - SIDE_WALL_T / 2  # 0.021
CROSS_WALL_T = 0.010  # end walls and the walls between chambers: 10 (x) x 34 (y) x 60 mm
CROSS_WALL_X = (0.212, 0.256, 0.300, 0.344, 0.388)
CHAMBER = 0.034  # chamber opening, both ways: 5 mm clearance a side round a 24 mm pin
# GPU watch (review round 1): the carried pin enters with 5 mm a side, the program's minimum for a loaded place.
# If the first GPU demo shows any pin landing on a wall top or left tilted over 8 degrees (TASK_TRACE: tilt_deg,
# top_mm, path_mm per chamber, pin_tilt_deg and pin_pos per pin), widen CHAMBER to 36 mm (6 mm a side): side walls
# 7 mm so the plug stays 50 mm wide and cross walls 8 mm so the 44 mm pitch stays (plug 184 mm long); the pads'
# outer faces, 20 mm from the pin axis, still cannot enter (18 mm half-width). Then re-derive the lean caps (15.7
# to 36.9 deg; sizes 4-5 in the 28 mm deep chamber could then pass their tipping angle) and the caught leans (to
# 36.5 / 24.5 / 19.5 / 16.0 deg for sizes 1-4), and update every number in task.json, README and the tests.
CHAMBER_HALF = CHAMBER / 2
CHAMBER_X = (0.234, 0.278, 0.322, 0.366)  # chambers 1-4

# -- the new key, shown by its cuts ---------------------------------------------------------------------------
# Each chamber's floor is a 34 x 34 mm nickel-silver block standing for the key's cut under that chamber.
CODE = (3, 1, 4, 2)  # the bitting code, chamber 1 first
FLOOR_H = (0.020, 0.032, 0.014, 0.026)  # floor heights: chamber depths 40, 28, 46, 34 mm
BOW_SIZE = (0.012, 0.030, 0.030)  # the key's bow, touching the plug's near end: the key is in
BOW_X = PLUG_X0 - BOW_SIZE[0] / 2  # 0.201

# -- the pins -----------------------------------------------------------------------------------------------------
# Gold-anodised aluminium cylinders, 24 mm across; size n is 28 + 6 (n - 1) mm long. Sizes differ by 6 mm, so a
# pin one size off stands 6 mm proud of the shear line or sits 6 mm low.
PIN_R = 0.012
PIN_LEN = {1: 0.028, 2: 0.034, 3: 0.040, 4: 0.046, 5: 0.052}
PIN_MASS = {1: 0.034, 2: 0.041, 3: 0.049, 4: 0.056, 5: 0.064}  # 6061 aluminium, 2.70 g/cm3
SIZES = (1, 2, 3, 4, 5)

# -- the pinning kit ------------------------------------------------------------------------------------------
# A floorless tray of grey plastic on the table: ten open cells, 5 rows along x by 2 columns along y, walls 4 mm
# thick and 6 mm tall, cells 42 (x) x 56 (y) mm inside. Row n of each column holds one size-n pin, standing on the
# table at the cell's centre. The left column (y +3.0 cm) is nearer the left arm.
KIT_WALL_T = 0.004
KIT_WALL_H = 0.006
KIT_ROW_X = (0.130, 0.176, 0.222, 0.268, 0.314)  # rows 1-5 (sizes 1-5)
KIT_COL_Y = {"left": 0.030, "right": -0.030}
KIT_CELL = (0.042, 0.056)
KIT_X0, KIT_X1 = 0.105, 0.339
KIT_LONG_Y = (0.060, 0.0, -0.060)  # three full-length walls along x, 234 mm
KIT_CROSS_X = (0.107, 0.153, 0.199, 0.245, 0.291, 0.337)  # 56 mm segments across y, two per line


def pin_name(column: str, size: int) -> str:
    """'pin_l3': the size-3 pin of the left column."""
    return f"pin_{column[0]}{size}"


PINS = {pin_name(c, n): (c, n) for c in SIDES for n in SIZES}  # name -> (kit column, size)
PIN_START = {name: (KIT_ROW_X[n - 1], KIT_COL_Y[c]) for name, (c, n) in PINS.items()}

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------
FLUSH_TOL = 0.0025  # the pin's top-face centre within this of the shear line, above or below (a 2.5 mm feeler)
TILT_MAX_DEG = 8.0  # the pin's axis at most this far from vertical (either end up)
STILL_WINDOW_S = 0.5  # 'still': its centre's path over the last 0.5 s ...
STILL_PATH = 0.002  # ... under 2 mm (positions, never velocities: a gripped part's velocity chatters on the GPU)
# GPU watch (review round 1): a summed path adds up frame noise (0.1 mm a step about a fixed point sums past 2 mm).
# If the first GPU demo's released, resting pins show path_mm over about 0.5 mm, rule (c) becomes net displacement
# or spread under 2 mm over the same window (the trace already carries net_mm and spread_mm, Tracker.drift).
LET_GO_R = 0.050  # 'let go': neither gripper's TCP within 50 mm of the pin's top-face centre
SHADOW_SAMPLES = 24  # points on each end circle when a pin's outline seen from above is drawn
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a pin moving this far in one control step was teleported: a new episode began ...
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) -----------------------------------------------------------------------------------------------
PLUG_GREY = (0.52, 0.54, 0.57)  # aluminium
KEY_SILVER = (0.86, 0.87, 0.89)  # nickel silver
KIT_GREY = (0.32, 0.33, 0.35)  # plastic
PIN_GOLD = (0.86, 0.68, 0.24)  # gold anodised
