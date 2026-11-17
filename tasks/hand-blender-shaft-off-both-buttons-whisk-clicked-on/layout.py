"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305 and the station's table top is at z 0.7475. Heights below are above the table top unless named otherwise.

A Braun MultiQuick 5 Vario hand blender (MQ5235 class) lies unplugged on its back along y at x 300 mm, its coupling
end toward -y. Its blending shaft is clicked into that end; a wire whisk (with its gearbox) lies at x 200 mm, its
gearbox toward +y; a cork mat lies at x 440 mm, y -170 mm. The coupling is a push-on fit: the attachment's stub (a
core and a lip) goes into a square socket in the motor part, two sprung release buttons (one on each side, facing
+x and -x) each move a latch hook on the socket's opposite wall that springs in behind the lip (pressing a button
moves its hook out). Both buttons in: the attachment slides out.

Figures are the draft's (est) and are to be measured on a bought unit (README). Local frames:
  motor    origin = the centre of its coupling face (the socket's mouth), axis along +y into the body;
  shaft    origin = the centre of its shoulder face (the collar's face that meets the coupling face); the stub runs
           from there to +y (u 0..21), the collar behind it (u 0..-12), the stem and bell to -y;
  whisk    the same: the gearbox's top face is the shoulder, the stub the same 21 mm, the wires to -y.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the motor part (loose, one floating articulation: the body and two sprung button links) ----------------------
M_X = 0.300
M_Y0 = 0.020  # the coupling face (y)
M_AXIS_Z = 0.030  # the axis' height: the body is 60 mm deep and lies on its back
M_W, M_H = 0.064, 0.060  # across (x) and deep (z); the oval is built as an octagon (10 mm chamfers)
M_LEN = 0.228  # y 20 to 248 mm
M_CHAMFER = 0.010
M_BASE_MASS = 0.600
M_MASS = 0.620  # with the two button links (0.010 each)
BORE = 0.0205  # the socket is a square 41 mm across, 24 mm deep
BORE_DEPTH = 0.024
FUNNEL = 0.005  # a 45 degree lead-in 5 mm wide at the mouth
M_ORIGIN = (M_X, M_Y0, TOP + M_AXIS_Z)
M_FRICTION = (0.35, 0.30)  # plastic on the table (est)
M_WHITE = (0.93, 0.93, 0.91)
M_GREY = (0.45, 0.46, 0.48)
BTN_BLACK = (0.15, 0.15, 0.17)

# -- the two release buttons and their latch hooks (one link each, a sprung slider frame along x) -----------------------
BTN_Y = 0.045  # 45 mm from the coupling end (centre; est)
BTN_Y_LEN, BTN_Z_LEN = 0.014, 0.022  # 14 x 22 mm
BTN_PROUD = 0.003  # stands 3 mm proud of the body's flank (est)
BTN_TRAVEL = 0.003  # flush when pressed 3 mm
BTN_SPRING = 667.0  # N/m: 2 N at the full 3 mm (est)
BTN_DAMP = 3.0
BTN_MASS = 0.010
HOOK_TIP_X = 0.0172  # the hook's tip, from the axis, at rest: 1.8 mm inside the lip's 19 mm
HOOK_Y0, HOOK_Y1 = 0.0120, 0.0156  # the hook's flat tip face: its back face is 0.4 mm short of the lip's trailing face
HOOK_RAMP_DEG = 25.0  # the ramp's angle to the push axis
HOOK_FLAT_RECESS = 0.0005  # the flat tip face sits 0.5 mm further out than the ramp's crest (17.7 mm): the lip slides
# off the crest onto nothing. Flush with the crest, the flat's front face met the lip's leading face edge-on at the seam of
# the two convex pieces and stopped it (GPU run 17 Nov: whisk stuck 9 mm short, lip front at the crest, arm 16 mm behind)
HOOK_Z = 0.014
HOOK_OUT = 0.0005  # the upper limit of the slider (a hook cannot go further out)
HOOK_IN_MAX = 0.0004  # a hook counts as behind the lip with its button no more than 0.4 mm in
HOOK_CLEARS = 0.0018  # a button in 1.8 mm or more has drawn its hook clear of the lip (the draft's 1.8 mm)

# -- the attachments' stub (the same on the shaft and the whisk) -------------------------------------------------
STUB_LEN = 0.021
CORE_HALF_X, CORE_HALF_Z = 0.0165, 0.019  # u 0..16, with the groove the hooks sit in
CORE_LEN = 0.016
LIP_HALF = 0.019  # u 16..21, 38 mm square: 1.5 mm of play in the 41 mm socket
LIP_U0 = 0.016

# -- the shaft (loose): stainless stem and bell, plastic collar and stub -------------------------------------------
SH_COLLAR_R, SH_COLLAR_LEN = 0.028, 0.012  # the collar behind the shoulder
SH_STEM_R = 0.012
SH_STEM_U0, SH_STEM_U1 = -0.146, -SH_COLLAR_LEN
SH_BELL_R = 0.030  # the 64 mm bell of the draft is built 60 mm so it lies level with the body (est)
SH_BELL_U0, SH_BELL_U1 = -0.191, -0.146
SH_LEN = STUB_LEN - SH_BELL_U0  # 212 mm from the stub's end to the bell's end
SH_MASS = 0.25
SH_AXIS_Z = M_AXIS_Z  # clicked in, level with the motor part
SH_FRICTION = (0.30, 0.25)  # stainless and plastic (est)
SH_STEEL = (0.74, 0.76, 0.78)
SH_PLASTIC = (0.20, 0.20, 0.22)

# -- the whisk (loose): gearbox and wire balloon ---------------------------------------------------------------------
WK_BOX_R, WK_BOX_LEN = 0.029, 0.060  # the gearbox, 58 mm across its flats, 60 long; its top face is the shoulder
WK_BOX_CHAMFER = 0.008  # the rounded-square housing's corner chamfers
WK_PARTS = (  # (radius, u0, u1) of the wire balloon's hulls behind the gearbox
    (0.015, -0.075, -0.060),
    (0.022, -0.100, -0.075),
    (0.029, -0.190, -0.100),
    (0.023, -0.215, -0.190),
    (0.015, -0.235, -0.215),
)
WK_LEN = 0.175 + WK_BOX_LEN  # 235 mm from the shoulder
WK_MASS = 0.110
WK_X = 0.200
WK_SHOULDER_Y = -0.040
WK_AXIS_Z = WK_BOX_R  # lying on the table
WK_ORIGIN = (WK_X, WK_SHOULDER_Y, TOP + WK_AXIS_Z)
WK_FRICTION = (0.30, 0.25)
WK_GREY = (0.80, 0.80, 0.82)
WK_WIRE = (0.78, 0.80, 0.84)

# -- the cork mat (fixed) ----------------------------------------------------------------------------------------------
MAT_X, MAT_Y = 0.440, -0.170  # its near edge at x 340 mm, 10 mm clear of the shaft's bell at the start
MAT_SIZE = 0.200
MAT_T = 0.010
MAT_CORK = (0.72, 0.56, 0.36)
MAT_FRICTION = (0.60, 0.50)

# -- the shaft's start pose: clicked in ----------------------------------------------------------------------------------
SH_ORIGIN = M_ORIGIN

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
HOME_TOL = 0.001  # the shoulder face within 1 mm of the coupling face
HOME_LAT = 0.003  # the attachment's axis within 3 mm of the socket's
HOME_DEG = 3.0  # and square within 3 degrees
INSERT_LAT = 0.030  # the stub counts as in the socket when its axis is within 30 mm of it and its tip is past the face
INSERT_DEG = 30.0
INSERT_MIN = 0.0005  # the stub's tip more than 0.5 mm past the coupling face
MAT_HALF = MAT_SIZE / 2  # the shaft's centre over the mat, 100 mm each way from the mat's centre
LOW_TOL = 0.005  # the shaft's lowest point within 5 mm of the mat's top (or the table)
FALL_DROP = 0.030  # latched: any part's lowest point more than 30 mm below the table top (off the table)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible motion: 5 mm and 5 degrees over the last second
STILL_TURN_DEG = 5.0
LET_GO = 0.030
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
