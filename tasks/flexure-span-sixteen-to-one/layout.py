"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A three-point bend (flexural) fixture stands on the midline: a steel base with a red centre line across it (where the
testing machine's loading nose comes down) and a scale of white ticks along its near edge, and two loose support
blocks on it, each carrying a steel roller along x. The rollers are 64 mm apart, as set for the last specimen (a 4 mm
thick ISO bar). Set for the 10 mm thick specimen on the mat, the rollers are 16 x 10 = 160 mm apart, symmetric about
the centre line, and the specimen lies flat across both, square to them, its middle midway between them.
"""

TOP = 0.7475  # station table top (world z)

# -- the fixture base (fixed, steel): x 0.265-0.355, y -0.150..+0.150, 10 mm tall ------------------------------------
BASE_X0, BASE_X1 = 0.265, 0.355  # 90 mm along x
BASE_X = (BASE_X0 + BASE_X1) / 2  # 0.310
BASE_HALF_Y = 0.150  # 300 mm along y
BASE_H = 0.010
BASE_TOP_W = TOP + BASE_H

# -- markings on the base's top (fixed, painted, 0.5 mm thick) -------------------------------------------------------
MARK_T = 0.0005
CENTRE_LINE_W = 0.002  # the red centre line: the full 90 mm depth of the base, 2 mm wide, at y 0
MID_Y = 0.0  # the loading nose's line: the span must be symmetric about it
TICK_W = 0.002  # white scale ticks along the near edge, every 10 mm from the centre line out to 140 mm each side
TICK_X0 = BASE_X0 + 0.001  # from 1 mm in from the near edge ...
TICK_LEN_SHORT = 0.008  # ... 8 mm long (x 0.266-0.274)
TICK_LEN_LONG = 0.012  # ... 12 mm at 50 and 100 mm (x 0.266-0.278)
TICK_PITCH = 0.010
TICK_COUNT = 14  # each side: 10, 20, ... 140 mm
TICK_LONG_EVERY = 5  # every fifth tick (50, 100 mm) is long

# -- a support block (two, identical; loose mechkit compounds). Base frame: origin at the middle of its underside,
#    local x along the roller, local z up. An aluminium block with a steel roller fixed along the middle of its top,
#    sunk 2 mm into it -----------------------------------------------------------------------------------------------
BLOCK_SIZE = (0.050, 0.030, 0.025)  # x along the roller, y across it, z
BLOCK_CENTER = (0.0, 0.0, BLOCK_SIZE[2] / 2)
ROLLER_R = 0.005  # 10 mm diameter
ROLLER_LEN = 0.050  # along x, the block's full length
ROLLER_Z = 0.028  # the roller's axis above the block's underside (its bottom 2 mm into the block)
ROLLER_TOP = ROLLER_Z + ROLLER_R  # 0.033 above the block's underside
SUP_MASS = 0.13  # aluminium block 37.5 cm3 (0.10 kg) plus a steel roller 3.9 cm3 (0.03 kg)
SUP_HALF = (BLOCK_SIZE[0] / 2, BLOCK_SIZE[1] / 2, ROLLER_TOP / 2)  # its outline (hands-off rule)
SUP_BOX_CENTER = (0.0, 0.0, ROLLER_TOP / 2)
ROLLER_TOP_W = BASE_TOP_W + ROLLER_TOP  # a roller's top with its block standing on the base (table + 43 mm)
START_SPAN = 0.064  # as set for a 4 mm thick bar (16 x 4 mm)
SUP_START = {"left": (BASE_X, START_SPAN / 2), "right": (BASE_X, -START_SPAN / 2)}  # rollers along x
SUP_START_YAW = 0.0

# -- the specimen (loose Box): a glass-filled nylon bar lying flat; its local y along its length ----------------------
SP_W, SP_LEN, SP_DEPTH = 0.020, 0.210, 0.010  # 20 mm wide (x), 210 mm long (y), 10 mm thick (the depth)
SP_SIZE = (SP_W, SP_LEN, SP_DEPTH)
SP_MASS = 0.057  # PA66-GF30, 42 cm3 at 1.36 g/cm3
SP_HALF = (SP_W / 2, SP_LEN / 2, SP_DEPTH / 2)

# -- the mat and where the specimen starts ----------------------------------------------------------------------------
MAT_X, MAT_Y = 0.190, 0.230
MAT_SIZE = (0.040, 0.240, 0.003)
MAT_TOP = MAT_SIZE[2]
SP_START = (MAT_X, MAT_Y, MAT_TOP + SP_DEPTH / 2)  # the specimen's centre, lying flat along y

# -- the rule (scoring.py) ----------------------------------------------------------------------------------------------
SPAN_RATIO = 16.0  # ASTM D790 / ISO 178: support span 16 (+-1) times the specimen's depth
SPAN_NOMINAL = SPAN_RATIO * SP_DEPTH  # 0.160
SPAN_TIGHT = 0.005  # 0.5 for the span: within 5 mm of 160 (16 +- 0.5 times the depth) ...
CENTRE_TIGHT = 0.005  # ... and its middle within 5 mm of the centre line
SPAN_LOOSE = 0.010  # 0.25: within 10 mm (16 +- 1 times the depth, the standards' tolerance) ...
CENTRE_LOOSE = 0.008  # ... and its middle within 8 mm of the centre line; otherwise 0
SP_OFF_TIGHT = 0.005  # +0.5 for the specimen: its middle within 5 mm of the span's middle ...
SQUARE_TIGHT_DEG = 3.0  # ... and square to both rollers within 3 degrees
SP_OFF_LOOSE = 0.010  # +0.25: within 10 mm (at any angle the 'across' check allows); +0 beyond

# -- the conditions every score needs ----------------------------------------------------------------------------------
SUP_TILT_MAX_DEG = 5.0  # 'standing': the block's vertical axis within 5 degrees of the vertical (upright)
SUP_ON_BASE_TOL = 0.002  # 'on the base': the middle of its underside within 2 mm of the base's top, inside its outline
ROLLER_HEADING_MAX_DEG = 5.0  # its roller within 5 degrees of x (parallel to the centre line), seen from above
SP_TILT_MAX_DEG = 5.0  # 'flat': the specimen's 10 mm thickness within 5 degrees of the vertical (either face down)
SP_ACROSS_MAX_DEG = 10.0  # 'across': its length within 10 degrees of square to each roller, seen from above
ON_ROLLER_TOL = 0.002  # at each roller: its underside within 2 mm of the roller's top ...
ALONG_ROLLER_MAX = ROLLER_LEN / 2 - SP_W / 2  # ... its full 20 mm width on the roller's 50 mm (centre line 15 mm) ...
END_MIN = 0.005  # ... and the roller at least 5 mm in from the specimen's end
HANDS_OFF = 0.020  # every tier: both grippers' finger zones at least 20 mm from the specimen and both supports
STILL_WINDOW_S = 0.5  # 'at rest': no end of the specimen and no roller moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the specimen moving this far in one control step was teleported: a new episode began ...
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
SUP_RESET_JUMP = 0.030  # a support jumping this far in one step (0.6 m/s; carried, it moves under 3 mm a step) ...
SUP_START_TOL = 0.015  # ... onto a support's starting mark (the set place, 48 mm out, is far from it): a new episode

# -- the grippers (the hands-off rule; rigstate reports the TCP, the point between the pad tips and the opening) -----
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
BASE_STEEL = (0.26, 0.27, 0.29)
LINE_RED = (0.85, 0.10, 0.08)
TICK_WHITE = (0.95, 0.95, 0.95)
BLOCK_BLUE = (0.12, 0.35, 0.75)  # blue-anodised aluminium
ROLLER_STEEL = (0.86, 0.87, 0.89)
SPECIMEN_CREAM = (0.93, 0.90, 0.78)  # natural glass-filled nylon
MAT_BLACK = (0.10, 0.10, 0.10)
