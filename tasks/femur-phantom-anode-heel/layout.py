"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the station table top unless named
*_W. The bare station table around the slabs stands for the room's floor.

A bench mock-up of a radiography room at about 40% scale. An x-ray table (a dark grey slab) lies on the midline with
its long axis along x; a pale yellow light field (the collimated field) with a black central-ray cross is painted on
it. Along its right-hand long edge (-y) stands a cream strip that stands for the x-ray tube's housing: the tube's
cathode-anode axis runs along x, its cathode end ("-", a black plate) at the far end (+x) and its anode end ("+",
a red plate) at the near end. The anode heel effect makes the beam stronger on the cathode side, so the thick (hip)
end of a femur goes toward the cathode: here toward +x. A femur phantom (a right femur, about 40% of real size) lies
crosswise on a side cart on the right; two side markers (red R, blue L) lie on a tray on the left.
"""

import math

TOP = 0.7475  # station table top (world z)
FLOOR_Z = 0.0  # the bare station table: the floor of the mock-up

# -- the x-ray table (fixed slab) and what is painted on it ---------------------------------------------------------
TABLE_X = (0.170, 0.430)  # 260 mm along x: its long axis is the tube's cathode-anode axis
TABLE_Y = (-0.080, 0.080)  # 160 mm across
TABLE_H = 0.010
FIELD_X = (0.195, 0.405)  # the light field: 210 mm along x ...
FIELD_Y = (-0.065, 0.065)  # ... by 130 mm across, centred on the cross
FIELD_H = 0.0005  # paint
FIELD_TOP = TABLE_H + FIELD_H  # what the phantom and the marker lie on (10.5 mm above the station table)
CROSS = (0.300, 0.000)  # the central ray
CROSS_LEN = 0.060  # each line 60 mm long ...
CROSS_W = 0.002  # ... and 2 mm wide
# the lines are paint flush with the field (the same 0.5 mm coat)

# -- the tube-housing strip (fixed) along the table's -y edge, with its two end plates ------------------------------
STRIP_X = TABLE_X  # 260 mm, the table's length
STRIP_Y = (-0.100, -0.080)  # 20 mm deep, against the table's -y edge
STRIP_H = 0.030  # 30 mm tall: its top 20 mm above the table top
PLATE_LEN = 0.040  # each end plate 40 mm (x) by the strip's 20 mm, 1 mm thick, on the strip's top
PLATE_H = 0.001
CATHODE_PLATE_X = (0.385, 0.425)  # black "-" plate at the far (+x) end
ANODE_PLATE_X = (0.175, 0.215)  # red "+" plate at the near end
SIGN_LONG, SIGN_W, SIGN_SHORT = 0.024, 0.005, 0.016  # white signs: a 24 x 5 mm bar ("-"); "+" adds a 5 x 16 mm bar
SIGN_H = 0.0005
CATHODE_DIR = 1.0  # the cathode is toward +x

# -- the side cart (fixed) and the phantom's start --------------------------------------------------------------------
CART_X = (0.230, 0.310)  # 80 mm
CART_Y = (-0.430, -0.210)  # 220 mm
CART_H = 0.010
PH_START = (0.270, -0.320)  # the phantom's centre (its frame's origin) on the cart ...
PH_START_YAW = math.pi / 2  # ... lying along y, hip end toward +y (crosswise to the tube axis)

# -- the femur phantom: one loose rigid body (mechkit compound). Its frame: origin at the middle of its bottom face
#    half-way along its 180 mm length, local +x from the knee end toward the hip end, local +z up (anterior face up).
#    Constant 40 mm width; its height steps from 24 mm at the knee end to 48 mm at the hip end, and a rounded femoral
#    head (an upright 40 mm cylinder) caps the hip end. Two grey painted condyles mark the knee end's top.
PH_LEN, PH_W = 0.180, 0.040
PH_MASS = 0.22
PH_BOXES = (  # (centre, size) in the phantom's frame
    ((-0.060, 0.0, 0.012), (0.060, PH_W, 0.024)),  # knee section: x -90..-30 mm, 24 mm tall
    ((0.005, 0.0, 0.018), (0.070, PH_W, 0.036)),  # shaft: x -30..+40 mm, 36 mm tall (the grip is here)
    ((0.055, 0.0, 0.024), (0.030, PH_W, 0.048)),  # hip section: x +40..+70 mm, 48 mm tall
)
PH_HEAD = ((0.070, 0.0, 0.024), 0.020, 0.048)  # (centre, radius, height): the head, x +50..+90 mm, 48 mm tall
CONDYLE_SIZE = (0.020, 0.014, 0.0005)  # painted on the knee section's top
CONDYLE_CENTERS = ((-0.077, 0.010, 0.024 + CONDYLE_SIZE[2] / 2), (-0.077, -0.010, 0.024 + CONDYLE_SIZE[2] / 2))
PH_CENTRE_LOCAL = (0.0, 0.0, 0.0)  # its centre: half-way along its length (what is centred on the cross)


def phantom_volumes():
    """(volume, centre) of each collider, the paint included (PhysX spreads the mass evenly over the colliders)."""
    out = [(s[0] * s[1] * s[2], c) for c, s in PH_BOXES]
    (hc, r, h) = PH_HEAD
    out.append((math.pi * r * r * h, hc))
    out += [(CONDYLE_SIZE[0] * CONDYLE_SIZE[1] * CONDYLE_SIZE[2], c) for c in CONDYLE_CENTERS]
    return out


def phantom_com_x() -> float:
    """The phantom's centre of mass along its length (local x, m): 16 mm toward the hip end."""
    v = phantom_volumes()
    return sum(vol * c[0] for vol, c in v) / sum(vol for vol, _ in v)


PH_COM_X = phantom_com_x()

# -- the side markers (loose boxes) on their tray (fixed) ------------------------------------------------------------
TRAY_X = (0.225, 0.275)  # 50 mm
TRAY_Y = (0.270, 0.370)  # 100 mm
TRAY_H = 0.010
MARKER_SIZE = (0.024, 0.024, 0.016)
MARKER_MASS = 0.030
R_START = (0.250, 0.300)
L_START = (0.250, 0.340)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
FLOOR_TOUCH = 0.005  # any point of the phantom below this height (above the floor) at any step: dropped, 0 (latched)
SEAT_TOL = 0.003  # 'lying flat on the table': its lowest point within 3 mm of the field's top ...
FLAT_TILT_DEG = 5.0  # ... and its local z (anterior face) within 5 degrees of straight up
ALIGN_DEG = 10.0  # 'aligned': its length within 10 degrees of the tube's cathode-anode axis (x) seen from above
CENTRE_TOL = 0.015  # 'centred': its centre within 15 mm of the cross
MARKER_TILT_DEG = 10.0  # the marker lying flat (either face up), its lowest point within SEAT_TOL of the field top
MARKER_CLEAR = 0.005  # the R marker's outline at least 5 mm clear of the phantom's outline, seen from above
HANDS_OFF = 0.005  # 'released': every gripper's finger zone at least 5 mm from the object
STILL_WINDOW_S = 0.5  # 'still': over each of the last two 0.5 s windows ...
STILL_PATH = 0.0015  # ... its centre's path under 1.5 mm (a path, not the chattering reported velocity) ...
STILL_YAW_DEG = 2.0  # ... and its heading within 2 degrees of its heading 1 s ago (and at every step since)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the phantom moving this far in one control step back onto its start mark: a new episode
START_TOL = 0.045  # ... 'on its start mark': within this of it (covers a seeded reset's +-3 cm jitter)

# -- the grippers (rigstate reports the TCP, the point between the pad tips and the opening; no heading) -------------
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
TABLE_GREY = (0.22, 0.23, 0.25)
FIELD_YELLOW = (0.98, 0.93, 0.55)
CROSS_BLACK = (0.03, 0.03, 0.03)
STRIP_CREAM = (0.93, 0.89, 0.76)
CATHODE_BLACK = (0.04, 0.04, 0.04)
ANODE_RED = (0.85, 0.08, 0.06)
SIGN_WHITE = (0.98, 0.98, 0.98)
CART_WHITE = (0.90, 0.90, 0.92)
TRAY_GREY = (0.55, 0.56, 0.58)
BONE_WHITE = (0.95, 0.92, 0.84)
CONDYLE_GREY = (0.45, 0.43, 0.40)
MARKER_RED = (0.85, 0.10, 0.08)
MARKER_BLUE = (0.10, 0.30, 0.90)

assert FIELD_X[1] - FIELD_X[0] > PH_LEN and FIELD_Y[1] - FIELD_Y[0] > PH_W
assert TABLE_X[0] < FIELD_X[0] and FIELD_X[1] < TABLE_X[1] and TABLE_Y[0] < FIELD_Y[0] and FIELD_Y[1] < TABLE_Y[1]
