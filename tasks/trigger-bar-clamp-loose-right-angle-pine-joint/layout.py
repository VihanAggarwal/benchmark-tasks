"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.
Sourced figures are marked (src), estimates (est); README, Sources says how to check each estimate.

The scene: a bench top with two loose pieces of pine, cut from 1x2 furring (actual 3/4 x 1-1/2 in, 19.05 x 38.1 mm,
modelled 19 x 38 mm), lying flat: board B, 300 mm long, along y; stub A, 100 mm long, along x. A person has dry-fitted
them into the corner of a small frame (A's end set against B's face) but roughly: A is 8 mm short of flush with B's
end, 4 mm off B's face and 2.5 degrees out of square. An IRWIN QUICK-GRIP Mini bar clamp, 6 in, lies on its side
on the bench, open, hung loosely over the corner (its pads straddle the two boards without touching them). The job:
square and flush the joint, squeeze the clamp's trigger until it bites, let go.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the pine (loose) --------------------------------------------------------------------------------------------------
BOARD_T = 0.019  # thickness: 3/4 in nominal, 19.05 actual (src: 1x2 furring, dressed size)
BOARD_W = 0.038  # width: 1-1/2 in, 38.1 actual (src)
B_LEN = 0.300  # board B (the rail), along y (est: a cut-off)
A_LEN = 0.100  # stub A, along x (est)
PINE_DENSITY = 500.0  # kg/m3, dry pine (est: 0.45 to 0.55 g/cm3)
B_MASS = round(PINE_DENSITY * BOARD_T * BOARD_W * B_LEN, 3)  # 0.108
A_MASS = round(PINE_DENSITY * BOARD_T * BOARD_W * A_LEN, 3)  # 0.036
PINE_FRICTION = (0.35, 0.30)  # pine on a laminate bench top (est; wet glue would be lower)
PINE = (0.86, 0.72, 0.49)

# the joint as it must end: A's end face against B's front face, A's outer face flush with B's end face
B_X0 = 0.225  # B's back face (the face the fixed jaw meets)
B_X1 = B_X0 + BOARD_W  # 0.263: B's front face, where A's end goes
B_CX = (B_X0 + B_X1) / 2
B_Y0 = 0.0  # B's end face (the outer end of the corner)
B_CY = B_Y0 + B_LEN / 2
A_X0 = B_X1  # A's end face when seated
A_X1 = A_X0 + A_LEN  # 0.363: A's free end, where the moving jaw meets
A_CX = (A_X0 + A_X1) / 2  # 0.313
A_CY = B_Y0 + BOARD_W / 2  # 0.019: flush with B's end face
# how a person leaves it after a rough dry fit (the start)
A_START_DX = 0.004  # A's end 4 mm off B's face
A_START_DY = 0.008  # A's outer face 8 mm inside B's end face (not flush)
A_START_YAW = math.radians(2.5)  # and turned out of square
A_HOLD_X = 0.285  # where the left gripper takes A (its axis), on the end of A nearer B (A's pads clear the moving pad by 24 mm)
# measured on A's end and outer face against B (scoring.py)
GAP_TIGHT = 0.001  # A's end face to B's face (feeler gauge): 1 mm (penetration over 2 mm is a glitch)
FLUSH_TIGHT = 0.0015  # A's outer face against B's end face (straightedge and feeler): 1.5 mm
SQUARE_TIGHT = 0.0025  # A's far end off the square's blade over its 100 mm: 2.5 mm (1.4 degrees)
GAP_LOOSE = 0.003
FLUSH_LOOSE = 0.003
SQUARE_LOOSE = 0.005  # 5 mm over 100 mm (2.9 degrees)
LEVEL_Z = 0.003  # the boards' tops within 3 mm of each other and of the table (lying flat)
SIDE_OK = 0.010  # A on B's front side: its centre at least 10 mm toward +x from B's front face line

# -- the clamp (loose): IRWIN QUICK-GRIP Mini one-handed bar clamp, 6 in capacity -------------------------------------------
# base frame: origin on the bench under the fixed jaw's pad face (x 0), the bar's centreline (y 0); x along the bar toward
# the moving jaw, y from the bar across the throat toward the work, z up. The clamp lies on its side: the jaws are 26 mm
# tall, the bar between them 8 mm.
JAW_H = 0.026  # jaw and handle height (est)
THROAT = 0.051  # 2 in throat: the pads reach 51 mm from the bar's centre (est from the maker's listing: measure the one used)
PAD_Y0 = -0.010
PAD_Y1 = THROAT
CAPACITY = 0.150  # 6 in (152 mm) opening: the pads' gap fully open (the clamp's name; est 150 mm as built), rounded down
G0 = CAPACITY  # the moving pad's face at x = G0 with the jaw at 0
BAR_Z0, BAR_Z1 = 0.009, 0.017
BAR_Y = 0.010  # the bar 20 mm wide
BAR_X0, BAR_X1 = -0.012, 0.236
JAW_BODY = 0.011  # jaw body behind its pad (x)
PAD_T = 0.003  # the black rubber pad
HEAD_X0, HEAD_X1 = G0 + JAW_BODY + PAD_T, G0 + JAW_BODY + PAD_T + 0.062  # 0.164 .. 0.226
HEAD_Y = 0.012
# the pistol grip: the handle on the head, extending away from the work (-y); the trigger behind it (+x)
HANDLE_X0, HANDLE_X1 = 0.170, 0.192  # 22 mm thick (est)
HANDLE_Y0, HANDLE_Y1 = -0.122, -HEAD_Y
TRIG_X0, TRIG_X1 = 0.220, 0.234  # the trigger lever at rest, 14 mm thick; 28 mm from the handle (est)
TRIG_Y0, TRIG_Y1 = -0.122, -HEAD_Y
TRIG_PIVOT = (0.2275, -0.014)  # the trigger's hinge (about z), at its head end
SPAN_REST = TRIG_X1 - HANDLE_X0  # 0.064: pad to pad when the gripper holds handle and trigger, trigger at rest
SPAN_STOP = 0.046  # ... with the trigger squeezed to its stop (18 mm of travel at the grip)
GRIP_LX = (HANDLE_X0 + TRIG_X1) / 2  # 0.202: the gripper's centre along x (clamp frame, jaw at 0)
GRIP_LY = -0.088  # the gripper's centre along y (clamp frame)
TRIG_ARM = GRIP_LY - TRIG_PIVOT[1]  # -0.074: the grip point's distance from the hinge
TRIG_TRAVEL = SPAN_REST - SPAN_STOP  # 0.018
TRIG_MAX = round(TRIG_TRAVEL / abs(TRIG_ARM), 3)  # 0.243 rad at the stop
TRIG_UPPER = 0.27  # the joint's limit (a little past the stop the gripper reaches)
TRIG_SPRING = 1.2  # N m/rad: 3.9 N at the grip point at full travel (est; the gripper's 5 N must beat it)
TRIG_ON = 0.8  # the ratchet takes a step when the trigger passes 80% of its travel ...
TRIG_OFF = 0.3  # ... and arms again when it is back under 30%
# the ratchet (the real mechanism: a pawl on the bar; each full squeeze pulls the moving jaw one step toward the fixed one)
STEP = 0.003  # jaw advance per full squeeze: 3 mm (est; measure with a ruler, README)
JAW_UPPER = 0.140
K_JAW = 20000.0  # N/m: the pads' and the bar's give; clamp force = K_JAW x (the ratchet's pull past the contact) (est)
F_MIN = 40.0  # clamped: at least 40 N (about 9 lbf: a snug glue-up clamp is 50 to 150 N; gauge, README) (est)
# the release lever (a small tab on the head's far end; pressing it frees the pawl and the jaw springs back)
REL_ANCHOR = (0.2285, 0.022)
REL_X0, REL_X1 = 0.2285, 0.2485
REL_Y0, REL_Y1 = 0.002, 0.026
REL_ON = 0.6  # the pawl is free when the tab has turned 60% of its travel (GPU run 2026-11-21: the tab swung to 49% on its own)
REL_SPRING = 1.0  # N m/rad: the tab's return spring (about 9 N at its tip to press it past 60%; 0.05 let it swing)
REL_DAMPING = 0.05
REL_UPPER = 0.45
CLAMP_MASS = 0.28  # est: the 6 in Mini weighs under 0.3 kg (weigh the one used)
CLAMP_FRICTION = (0.45, 0.40)
CLAMP_YELLOW = (0.96, 0.74, 0.07)
CLAMP_BLACK = (0.08, 0.08, 0.09)
STEEL = (0.62, 0.63, 0.65)
# where the clamp is as found, and where it goes: the fixed pad's face 3 mm behind B's back face, the bar's centreline
# 26 mm outside B's end face (so that the bar clears the corner and the left gripper's pad), the pads reaching 25 mm over A
CLAMP_XF = B_X0 - 0.003  # 0.222
CLAMP_YC = B_Y0 - 0.026  # -0.026 (placed)
# GPU run 2026-11-20: dragging the clamp 70 mm by its pistol grip failed (it turned 10.5 degrees about the pinch and
# lagged the hand by 43 mm), so the clamp starts hung over the corner where it is to stay: SLIDE is 0.
SLIDE = 0.0
CLAMP_YC0 = CLAMP_YC - SLIDE  # as found
# GPU run 2026-11-21 (run 2): with the clamp hung loose, every ratchet step (the jaw's 20 N/mm drive) pulled the base toward the
# hand-held jaw and turned it about the hand: the bar slid 21 mm toward -y and 6 degrees, so the pads lost the rail (cover 10 mm).
# A bench stop (a strip of pine screwed to the bench) lies 2.5 mm off the clamp's outer side, along the fixed jaw and the bar.
STOP_X0, STOP_X1 = 0.200, 0.350
STOP_Y1 = CLAMP_YC + PAD_Y0 - 0.0025  # -0.0385: 2.5 mm from the jaw's and the bar's outer faces
STOP_Y0 = STOP_Y1 - 0.020
PAD_OVERLAP_MIN = 0.015  # the pads cover at least 15 mm of A's width and of B's width (they reach 25 mm)
PAD_TOUCH = 0.0015  # a pad's face within 1.5 mm of the board's face: on the work
SPAN_BOARDS = A_LEN + BOARD_W  # 0.138: both boards between the pad faces, fully seated
JAW_STOP = G0 - SPAN_BOARDS  # 0.012: the jaw's pull when the boards are seated and the pads touch
GRIPPER_FULL = 0.0939

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001  # still: nothing moved over 1 mm or turned over 1 degree in the last 1 s (a real trial: no visible creep)
STILL_TURN_DEG = 1.0
LET_GO = 0.030
FALL_DROP = 0.020
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
