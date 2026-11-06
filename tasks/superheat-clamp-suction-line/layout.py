"""Every dimension and threshold of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say TOP.

The fixed rig is a bench mock-up of a condensing unit's service corner: the unit's corner, two brass service valves
on its right (-y) face, and the two copper lines running from them along -y. The lines are copper bars of square
section as wide as the tubes (19 mm suction line, 9.5 mm liquid line), so the rigid clamp sits level on top of the
suction line (a real clamp's spring jaws centre it on a round tube; see fidelity in task.json).

Two parts are loose, each a mechkit compound body (mechkit.py): the thermometer clamp, a saddle (a top block with a
display, two legs under its x edges) whose frame origin is the centre of the top block's underside, and the foam
cover, a half-shell (a roof and two walls along y) whose frame origin is the centre of its bottom.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# Colours (sRGB).
STEEL_GREY = (0.55, 0.57, 0.58)
BRASS = (0.78, 0.62, 0.25)
COPPER = (0.72, 0.45, 0.20)
COUPLING_COPPER = (0.62, 0.38, 0.18)
FOAM_BLACK = (0.08, 0.08, 0.08)
STAND_GREY = (0.45, 0.45, 0.45)  # painted wood
PAD_GREY = (0.62, 0.62, 0.62)  # plastic
CLAMP_BLUE = (0.15, 0.35, 0.75)
DISPLAY_BLACK = (0.03, 0.03, 0.03)
COVER_BLACK = (0.10, 0.10, 0.10)

# -- the fixed rig (centre (x, y, height of the centre above the table), size (x, y, z)) ---------------------------
UNIT_C, UNIT_S = (0.290, 0.250, 0.080), (0.120, 0.060, 0.160)  # x 0.23-0.35, y 0.22-0.28, 16 cm tall

LINE_X = 0.250  # the suction line's axis, along y
LINE_Z = 0.070  # ... its height
LINE_W = 0.019  # 3/4 in tube, as a 19 x 19 mm bar
LINE_HALF = LINE_W / 2
LINE_TOP = LINE_Z + LINE_HALF  # 0.0795: the copper's top
VALVE_FACE_Y = 0.190  # the suction service valve's body face (its -y side), where d is measured from
COUPLING_END_Y = 0.165  # the brazed coupling runs from the valve face to here
FOAM_START_Y = -0.080  # the foam insulation starts here and runs to the line's end
LINE_END_Y = -0.150

SVALVE_C, SVALVE_S = (LINE_X, 0.205, LINE_Z), (0.030, 0.030, 0.030)  # y 0.190-0.220: on the unit's face
SVALVE_CAP_C, SVALVE_CAP_R, SVALVE_CAP_H = (LINE_X, 0.205, 0.090), 0.005, 0.010  # port cap on its top
COUPLING_W = 0.025
COUPLING_C = (LINE_X, (COUPLING_END_Y + VALVE_FACE_Y) / 2, LINE_Z)
COUPLING_S = (COUPLING_W, VALVE_FACE_Y - COUPLING_END_Y, COUPLING_W)  # top 0.0825, 3 mm above the bare copper
BARE_C = (LINE_X, (FOAM_START_Y + COUPLING_END_Y) / 2, LINE_Z)
BARE_S = (LINE_W, COUPLING_END_Y - FOAM_START_Y, LINE_W)  # y -0.080 to 0.165: 24.5 cm of bare copper
FOAM_W = 0.045
FOAM_C = (LINE_X, (LINE_END_Y + FOAM_START_Y) / 2, LINE_Z)
FOAM_S = (FOAM_W, FOAM_START_Y - LINE_END_Y, FOAM_W)  # top 0.0925, bottom 0.0475

LIQ_X, LIQ_Z = 0.320, 0.0653  # the liquid line's axis, along y
LIQ_W = 0.0095  # 3/8 in tube, as a 9.5 x 9.5 mm bar
LIQ_TOP = LIQ_Z + LIQ_W / 2  # 0.07005
LVALVE_C, LVALVE_S = (LIQ_X, 0.208, LIQ_Z), (0.024, 0.024, 0.024)  # y 0.196-0.220: on the unit's face
LVALVE_CAP_C, LVALVE_CAP_R, LVALVE_CAP_H = (LIQ_X, 0.208, 0.0813), 0.004, 0.008
LIQ_START_Y = 0.196
LIQ_C = (LIQ_X, (LINE_END_Y + LIQ_START_Y) / 2, LIQ_Z)
LIQ_S = (LIQ_W, LIQ_START_Y - LINE_END_Y, LIQ_W)

STAND_Y = -0.120  # two blocks hold the lines up near their ends
FOAM_STAND_S = (FOAM_W, 0.030, LINE_Z - FOAM_W / 2)  # 0 to 0.0475, under the foam
FOAM_STAND_C = (LINE_X, STAND_Y, FOAM_STAND_S[2] / 2)
LIQ_STAND_S = (0.020, 0.030, LIQ_Z - LIQ_W / 2)  # 0 to 0.06055, under the liquid line
LIQ_STAND_C = (LIQ_X, STAND_Y, LIQ_STAND_S[2] / 2)

PAD_T = 0.003  # the two start pads' thickness
CLAMP_PAD_C, CLAMP_PAD_S = (0.150, 0.200, PAD_T / 2), (0.080, 0.080, PAD_T)
COVER_PAD_C, COVER_PAD_S = (0.150, 0.085, PAD_T / 2), (0.090, 0.090, PAD_T)

# The lines the clamp's legs can straddle (offline model): name -> (axis x, half width, top, y from, y to)
STRADDLE_LINES = {
    "suction_line": (LINE_X, LINE_HALF, LINE_TOP, FOAM_START_Y, COUPLING_END_Y),
    "brazed_coupling": (LINE_X, COUPLING_W / 2, LINE_Z + COUPLING_W / 2, COUPLING_END_Y, VALVE_FACE_Y),
    "liquid_line": (LIQ_X, LIQ_W / 2, LIQ_TOP, LINE_END_Y, LIQ_START_Y),
}
# ... and those that fit inside the cover's channel (the 45 mm foam too)
CHANNEL_LINES = {**STRADDLE_LINES, "suction_foam": (LINE_X, FOAM_W / 2, LINE_Z + FOAM_W / 2, LINE_END_Y, FOAM_START_Y)}

# -- the thermometer clamp (frame origin: the centre of the top block's underside) ---------------------------------
BLOCK_S = (0.049, 0.040, 0.016)  # 49 mm across the line (x), 40 mm along it (y)
BLOCK_C = (0.0, 0.0, BLOCK_S[2] / 2)
LEG_S = (0.008, 0.040, 0.020)
LEG_X = 0.0205  # legs' centres at +-x: inner faces at +-0.0165, a 33 mm gap (7 mm each side of the 19 mm line)
LEG_C = (LEG_X, 0.0, -LEG_S[2] / 2)
LEG_INNER = LEG_X - LEG_S[0] / 2  # 0.0165
LEG_LEN = LEG_S[2]  # 0.020: the legs reach 1 mm below the line's bottom
DISPLAY_S = (0.024, 0.020, 0.004)  # 4 mm thick, the program's least wall
DISPLAY_C = (0.0, 0.0, BLOCK_S[2] + DISPLAY_S[2] / 2)
DISPLAY_TOP = BLOCK_S[2] + DISPLAY_S[2]  # 0.020 above the origin: 'the clamp's top'
CLAMP_SHAPES = (  # (centre, size, colour) in the clamp's frame
    (BLOCK_C, BLOCK_S, CLAMP_BLUE),
    ((LEG_X, 0.0, LEG_C[2]), LEG_S, CLAMP_BLUE),
    ((-LEG_X, 0.0, LEG_C[2]), LEG_S, CLAMP_BLUE),
    (DISPLAY_C, DISPLAY_S, DISPLAY_BLACK),
)
CLAMP_MASS = 0.06
CLAMP_START = (0.150, 0.200, PAD_T + LEG_LEN)  # standing on its legs on its pad, long side along y

# -- the foam cover (frame origin: the centre of its bottom) --------------------------------------------------------
COVER_W, COVER_L = 0.077, 0.070  # across its walls (x), along its channel (y)
CEILING = 0.036  # the channel's ceiling above the cover's bottom (the walls' height)
ROOF_T = 0.010
WALL_T = 0.006
ROOF_S = (COVER_W, COVER_L, ROOF_T)
ROOF_C = (0.0, 0.0, CEILING + ROOF_T / 2)
WALL_S = (WALL_T, COVER_L, CEILING)
WALL_X = COVER_W / 2 - WALL_T / 2  # 0.0355
WALL_C = (WALL_X, 0.0, CEILING / 2)
CHANNEL_HALF = COVER_W / 2 - WALL_T  # 0.0325: a 65 mm channel, 8 mm each side of the clamp's 49 mm block
COVER_HEIGHT = CEILING + ROOF_T  # 0.046
COVER_SHAPES = (  # (centre, size, colour) in the cover's frame
    (ROOF_C, ROOF_S, COVER_BLACK),
    ((WALL_X, 0.0, WALL_C[2]), WALL_S, COVER_BLACK),
    ((-WALL_X, 0.0, WALL_C[2]), WALL_S, COVER_BLACK),
)
COVER_MASS = 0.02
COVER_START = (0.150, 0.085, PAD_T)  # standing on its walls on its pad, channel along y

# -- scoring (scoring.py) -------------------------------------------------------------------------------------------
STILL_STEPS = 10  # 'at rest': the part's origin moved, over the last 10 control steps (0.5 s) ...
STILL_PATH = 0.003  # ... less than 3 mm in all
SEAT_X_TOL = 0.008  # seated: the clamp's origin within 8 mm of the line's axis, across the line (the legs leave
#                      7 mm of play a side, so a clamp resting astride always passes; further off, a leg is on the
#                      copper and the clamp sits 20 mm high)
SEAT_Z_LO, SEAT_Z_HI = LINE_TOP - 0.001, LINE_TOP + 0.002  # ... its underside 1 mm below to 2 mm above the copper top
SEAT_TILT_DEG = 4.0  # ... tilted at most this
SEAT_YAW_DEG = 20.0  # ... its long side within this of the line (either way round; past about 18 degrees the legs'
#                       inner corners would cut into the copper, so a clamp resting astride always passes)
NEAR_D = 0.150  # d at most this: 'within 15 cm of the suction service valve'
COVER_FOOT_X, COVER_FOOT_Y = COVER_W / 2, COVER_L / 2  # insulated: the block's top corners inside the cover's outline
CEIL_GAP_LO, CEIL_GAP_HI = -0.001, 0.003  # ... the ceiling 1 mm below to 3 mm above the display's top
COVER_TILT_DEG = 10.0
# Clear of a part: a gripper's TCP and pad-tip point both at least CLEAR_ABOVE above the top of the part's bounding
# box, or both at least CLEAR_SIDE from that box seen from above. An open finger's outer face is 57 mm from the TCP
# (47 mm half opening + 10 mm finger) and its corners about 60 mm (the finger is 40 mm wide), so a gripper 6.5 cm to
# the side cannot touch the part with any finger, open or closed, and one 4 cm above it cannot either.
CLEAR_ABOVE = 0.040
CLEAR_SIDE = 0.065
FINGER_INNER_OPEN = 0.047  # an open finger's inner face from the TCP, along the closing axis (tests)
FINGER_OUTER_OPEN = 0.057  # ... its outer face
RESET_JUMP = 0.05  # a part moving this far in one control step was teleported ...
START_TOL = 0.045  # ... and back within this of its start (covers a seeded reset's +-3 cm jitter): a new trial
START_Z_TOL = 0.003
START_TILT_DEG = 5.0

# -- the offline model (offline_saddle.py; never used in the simulator) -----------------------------------------------
OFFLINE_DROP_TOL = 0.003  # the checker dropped a part's box onto a top: its bottom within 3 mm of that top

DEG = math.pi / 180.0
