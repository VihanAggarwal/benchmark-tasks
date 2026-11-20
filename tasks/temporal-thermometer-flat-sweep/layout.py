"""Every dimension and threshold of the task, in one place (metres, kilograms, radians, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights named ``Z_`` are above the table top.

The scene: an adult care-training manikin head stands upright on its neck block on the table, facing the robot (-x),
on a sprung neck (it nods back when the forehead is pushed). Its forehead curves round toward each temple (a vertical
cylinder, radius 75 mm, in top view). An Exergen TemporalScanner TAT-5000S stands on its long edge in a two-block foam
rest to the robot's right of the head, its sensing cup toward the head.
"""

import math

TOP = 0.7475  # station table top

# -- the manikin head (design values; see task.json fidelity) ----------------------------------------------------------
FX, FY = 0.360, -0.060  # the forehead's centre of curvature (top view): a vertical axis through the head
RF = 0.075  # forehead radius in top view (the skin's curve from the centre toward each temple)
Z_LINE = 0.170  # the scan line: mid-forehead, between the brows (top 0.146) and the hairline (0.205)
Z_BROW = 0.146
Z_HAIR = 0.205  # the hairline above the forehead (crown hair from here up)
FOREHEAD_Z = (0.110, 0.235)  # the skull cylinder's span
HAIR_PHI = math.radians(55.0)  # the hairline at each temple: hair (3 mm proud) from 55 degrees round the curve
NECK_Z = 0.010  # the neck hinge (axis along y) above the table
H_ARM = Z_LINE - NECK_Z  # 0.160: the scan line's lever about the neck hinge
HEAD_K = 500.0  # N/m: pushed straight back at the scan line, the head gives 0.5 N per mm (est.)
HEAD_LINK_MASS = 0.80  # kg: the hollow vinyl head that nods on the neck (est.)
HEAD_DAMP = 0.30  # N m s/rad
HEAD_UPPER = 0.080  # rad: the neck's back stop (12.8 mm at the scan line)
HEAD_LOWER = -0.010

# -- the thermometer: Exergen TemporalScanner TAT-5000S (datasheet P/N 818063: 5 x 20 x 3 cm, 213 g) ---------------------
# Its own (body) frame: origin at the centre of the sensing cup's face (the lens), +x out of the back of the body, +y
# toward the probe-head end (the body runs along -y), z across its 50 mm width. The probe head is angled: the cup
# faces CUP_TILT from the back toward the head end (est. from the maker's photos), so with the cup flush on the skin
# the body leans 30 degrees away from it, as in the maker's pictures.
PR_MASS = 0.213
CUP_R = 0.016  # the probe head: a 32 mm dome, its cup facing +x
CUP_DEPTH = 0.030  # front face (x -0.030) to cup face (x 0)
BODY_X = (-0.030, -0.003)  # the body is 27 mm thick; the cup stands 3 mm proud of its back
BODY_Y = (-0.184, -0.037)  # 147 mm of body below a neck; 200 mm from the dome's top to the bottom end
BODY_W = 0.050
NECK_Y = (-0.040, -0.010)
NECK_SIZE_XZ = (0.021, 0.022)
BTN_Y = -0.079  # the SCAN button: on the front face, 79 mm below the cup's centre, under the thumb
BTN_R = 0.008
BTN_PROUD = 0.002
BTN_TRAVEL = 0.002
BTN_K = 750.0  # N/m (1.5 N at the bottom, est.)
BTN_MASS = 0.003
BTN_DAMP = 3.0
BTN_PRESSED = 0.0015  # 'held': pressed at least 1.5 mm of its 2 mm
DISPLAY_Y = (-0.140, -0.105)
CUP_TILT = math.radians(30.0)
CUP_DIR = (math.cos(CUP_TILT), math.sin(CUP_TILT), 0.0)  # the way the cup faces, body frame
GRIP_LOCAL = (-0.0165, BTN_Y, 0.0)  # the TCP in the body frame when the pads close across button and back

# -- the foam rest (two slotted blocks under the body; the grip zone between them is open) ------------------------------
HOLD_X, HOLD_Y = 0.240, -0.170  # the cup's centre when it stands in the rest (body frame lined up with the world)
REST_YAW = CUP_TILT  # the way the cup faces in the rest: 30 degrees left of +x
REST_FLOOR = 0.012  # slot floor above the table
REST_TOP = 0.024
SLOT_W = 0.034
REST_OUT_X = 0.060
REST_BLOCKS = ((-0.060, -0.040), (-0.170, -0.100))  # y spans in the body frame
PR_Z0 = REST_FLOOR + BODY_W / 2  # the cup's centre height when it stands in the rest
PR_X0, PR_Y0 = HOLD_X, HOLD_Y

# -- the scan (the forehead's curve; s is arc length from the centre toward the robot's right, the patient's left) ------
S_START = 0.0
S_BEGIN = 0.005  # the timed stretch starts as the cup's centre leaves 5 mm round the forehead centre ...
S_FINISH = 0.050  # ... and the scan ends 50 mm round the curve (the cup's edge within 6 mm of the hairline)
S_MIN, S_MAX = -0.010, 0.060  # arc a pass may use
START_ZONE = 0.005  # a pass starts within 10 mm before to 5 mm past the forehead centre
FRICTION = (0.5, 0.45)  # thermometer, skin, foam (est.)

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------------
FACE_TOL = 0.002  # 'touching': the cup's centre within 2 mm of the skin
FLUSH_DEG = 10.0  # the cup faces the skin within 10 degrees of square (the skin's normal there)
LINE_TOL = 0.008  # the cup's centre within 8 mm above or below the scan line
PUSH_MIN, PUSH_MAX = 0.0010, 0.0040  # the head pushed back 1 to 4 mm at the scan line (0.5 to 2.0 N straight on)
PUSH_SHOVE = 0.008  # latched zero: the head shoved back 8 mm or more (4 N)
LIFT_TOL = 0.006  # contact may break by up to 6 mm within a pass
BACKSLIP = 0.003
T_MIN, T_MAX = 1.5, 4.0  # seconds for the 45 mm from S_BEGIN to S_FINISH (11 to 30 mm/s; the maker says 2 to 3 s)
TOUCH_S = 0.5
REST_XY = 0.010  # put back: the cup's centre within 10 mm of where it started (top view)
REST_Z = 0.004  # and within 4 mm of its height in the rest
REST_TILT_DEG = 10.0  # standing on its edge within 10 degrees
REST_YAW_DEG = 15.0  # turned within 15 degrees of how it stood
HEAD_BACK = 0.001  # the head back within 1 mm of upright
LET_GO = 0.030
HANDS_OFF_HEAD = 0.020  # touching needs both grippers' finger zones this far from the head: the cup presses it
STILL_WINDOW_S = 1.0
STILL_V, STILL_W = 0.01, 0.5
HEAD_STILL_W = 0.02  # rad/s
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.006
EPS = 1e-9

ARM_K = 550.0  # N/m: how far an arm gives when pushed sideways (about 1.8 mm per N); offline model only


def head_spring(link_mass_height: float) -> float:
    """The neck spring (N m/rad) that makes the head give HEAD_K at the scan line, net of gravity's tipping torque
    from the nodding head's centre of mass ``link_mass_height`` metres above the hinge."""
    return HEAD_K * H_ARM ** 2 + HEAD_LINK_MASS * 9.81 * link_mass_height


def arc_point(s: float, r: float = RF) -> tuple[float, float]:
    """World (x, y) of the point ``s`` along the forehead's curve from its centre, at radius ``r`` from the axis."""
    phi = s / RF
    return FX - r * math.cos(phi), FY - r * math.sin(phi)
