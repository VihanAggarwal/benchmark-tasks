"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP (the counter). The cooktop's frame has its origin on the counter under the middle of its
front edge. The pan's frame has its origin at the centre of its base, z up, its handle along +x. The lid's frame has
its origin at the centre of the underside of its glass, z up. Knob angles are seen from above, counterclockwise
positive, 0 = pointing away from the robot (+x).

Sources (README.md): Karinear 12 in two-burner plug-in ceramic cooktop with knob control (520 x 290 x 54 mm; burners
165 mm / 600 W at the front and 200 mm / 1400 W at the rear; two knobs on the glass at the front, 0 to 9);
Tramontina Professional 8 in fry pan (8 in across the top, 1.3 lb); Libbey 16 oz pint glass. Burner and knob
positions are measured off the maker's top-view photo; the knobs' size and sweep, the lid, the pan's base, depth and
handle, and every friction and mass not listed are estimates (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
INCH = 0.0254
TOP = 0.7475  # station table top (the counter)
CONTROL_HZ = 20.0

# -- the cooktop (fixed; world) ---------------------------------------------------------------------------------------
CT_D = 0.520  # deep (along x)
CT_W = 0.290  # wide (along y)
CT_T = 0.054  # the unit's thickness
CT_FEET = 0.006  # on its countertop brackets: the glass 60 mm above the counter
CT_FRONT_X = 0.130  # its front edge, 130 mm in front of the arm bases' line
CT_CY = 0.100  # its middle, 100 mm to the robot's left of centre
GLASS_TOP = CT_FEET + CT_T  # the glass, above the counter
BURNERS = {  # name -> (x from the front edge, y from the middle, radius); off the maker's photo
    "front": (0.184, -0.006, 0.0825),  # 165 mm, 600 W: on, the pan on it
    "rear": (0.387, 0.000, 0.100),  # 200 mm, 1400 W: off
}
CERAMIC = (0.04, 0.04, 0.05)
BURNER_HOT = (0.70, 0.12, 0.06)
BURNER_OFF = (0.12, 0.12, 0.13)

# Knobs on the glass along the front edge, 57 mm behind it: the right one (seen by the robot) works the front burner
# (a burner symbol printed above each knob shows which). Each is a 38 mm round skirt with a grip bar across its top
# and a pointer at one end of the bar. 0 (OFF) points away from the robot; the knob turns clockwise, seen from above,
# through 1 to 9, 270 deg (est.), with a stop at each end. The front burner's knob is at 8, 230 deg from OFF.
KNOBS = {  # name -> (x from the front edge, y from the middle)
    "front_knob": (0.057, -0.056),
    "rear_knob": (0.057, +0.065),
}
KNOB_R = 0.019
SKIRT_H = 0.012  # the knob's round skirt
BAR_L = 0.034
BAR_W = 0.010
BAR_H = 0.014  # the grip bar stands on the skirt
KNOB_SWEEP = math.radians(270.0)  # 0 to 9
KNOB_AT_8 = math.radians(230.0)  # where the front burner's knob starts
KNOB_FRICTION = 0.10  # N m (est.): the switch turns stiffly and stays where it is put
KNOB_STEEL = (0.62, 0.63, 0.65)
POINTER = (0.10, 0.10, 0.10)
OFF_OK_DEG = 10.0  # a knob counts as OFF within 10 deg of OFF


def burner_xy(name: str) -> tuple[float, float]:
    bx, by, _ = BURNERS[name]
    return (CT_FRONT_X + bx, CT_CY + by)


def knob_xy(name: str) -> tuple[float, float]:
    kx, ky = KNOBS[name]
    return (CT_FRONT_X + kx, CT_CY + ky)


# -- the frying pan (pan frame) ----------------------------------------------------------------------------------------
PAN_TOP_R = 4.0 * INCH  # 102 mm: 8 in across the top
PAN_BASE_R = 0.075  # base 150 mm across (est.)
PAN_DEPTH = 0.040  # rim 40 mm above the base (est.)
PAN_WALL = 0.003
PAN_FLOOR = 0.004
HANDLE_LEN = 0.170  # from the rim to the end (est.)
HANDLE_RISE_DEG = 12.0
HANDLE_W = 0.022
HANDLE_T = 0.015
HANDLE_Z0 = PAN_DEPTH - 0.009
M_PAN = 1.3 * 0.4536  # 0.59 kg empty
M_OIL = 0.055  # 60 ml of cooking oil
ALUMINIUM = (0.20, 0.20, 0.22)
SLEEVE = (0.55, 0.06, 0.06)
OIL = (0.62, 0.42, 0.10)
FLAME_OUTER = (1.00, 0.42, 0.04)
FLAME_INNER = (1.00, 0.80, 0.15)
PAN_FRICTION = (0.30, 0.25)  # aluminium on glass ceramic (est.)
WALL_SEGMENTS = 20
PAN_START_YAW = math.radians(-90.0)  # the handle points to the robot's right, out past the cooktop's right edge
# The flame prop (an orange silicone flame prop of this size in a real trial): three crossed tongues standing on the
# pan's floor, the tallest 22 mm, so a lid seated on the rim (its centring skirt 32 mm above the base) clears it.
FLAMES = ((0.0, 0.022, 0.060), (math.pi / 3, 0.019, 0.050), (-math.pi / 3, 0.016, 0.044))  # (yaw, height, width)
FLAME_T = 0.006


def pan_start() -> tuple[float, float, float]:
    x, y = burner_xy("front")
    return (x, y, TOP + GLASS_TOP)


def rim_z() -> float:
    """The pan's rim, world z, when it sits on its burner."""
    return TOP + GLASS_TOP + PAN_DEPTH


def handle_dir() -> np.ndarray:
    a = math.radians(HANDLE_RISE_DEG)
    return np.array([math.cos(a), 0.0, math.sin(a)])


def handle_root() -> np.ndarray:
    r = PAN_BASE_R + (PAN_TOP_R - PAN_BASE_R) * HANDLE_Z0 / PAN_DEPTH
    return np.array([r, 0.0, HANDLE_Z0])


# -- the lid (lid frame) ------------------------------------------------------------------------------------------------
LID_R = 0.106  # 212 mm across its rim: it rests on the pan's 203 mm rim
LID_T = 0.005  # the steel at the rim
DOME = ((0.092, 0.005), (0.068, 0.005))  # (radius, thickness) of the lid's domed top, two steps above the rim
SKIRT_R = 0.086  # the centring skirt under the rim, 5 mm inside the pan's flared wall where the skirt reaches it
LID_SKIRT_H = 0.008
SKIRT_T = 0.003
SKIRT_SEGMENTS = 16
KNOB_NECK = (0.010, 0.010)  # (radius, height) of the lid knob's neck
LID_KNOB_R = 0.019  # the lid's knob, 38 mm across
LID_KNOB_H = 0.024
M_LID = 0.30  # kg (est.): a stainless steel lid
LID_GLASS = (0.66, 0.67, 0.69)  # brushed stainless (the lid is all steel)
STEEL = (0.72, 0.73, 0.75)
LID_FRICTION = (0.5, 0.4)
LID_START = (0.300, 0.380)  # world x, y: lying dome-up on the counter to the left of the cooktop


def lid_knob_z0() -> float:
    """The bottom of the lid's knob (lid frame)."""
    return LID_T + sum(t for _, t in DOME) + KNOB_NECK[1]


def lid_start() -> tuple[float, float, float]:
    return (LID_START[0], LID_START[1], TOP + LID_SKIRT_H)  # resting on its skirt


# -- the glass of water ----------------------------------------------------------------------------------------------
GLASS_R = 1.75 * INCH  # 89 mm across the top
GLASS_BASE_R = 0.032  # 64 mm across the base (est.)
GLASS_H = 5.75 * INCH  # 146 mm tall
GLASS_BOTTOM = 0.012  # the thick glass bottom
GLASS_WALL = 0.003
GLASS_SEGMENTS = 20
WATER_H = 0.118  # water 130 mm up from the table, 16 mm below the rim (about 440 ml)
M_GLASS = 0.28 + 0.44  # the glass and 440 ml of water
WATER = (0.25, 0.55, 0.95)
WATER_TOP = (0.55, 0.78, 1.00)
GLASS_START = (0.080, 0.110)  # on the counter right in front of the pan, 5 mm short of the cooktop's front edge
GLASS_COLOUR = (0.86, 0.90, 0.94)

# -- scoring ------------------------------------------------------------------------------------------------------------
LID_CENTRE_OK = 0.004  # m: the lid's centre within 4 mm (in plan) of the pan's, so its 106 mm-radius rim overlaps the
# pan's 101.6 mm-radius rim all the way round
LID_TILT_OK = 3.0  # deg
LID_SEAT_Z = 0.003  # m: the underside of the lid's glass within 3 mm of the pan's rim (no open gap)
PAN_MOVE_FAIL = 0.030  # m: the pan's base centre more than 30 mm from where it started, at any moment (latched 0)
PAN_LIFT_FAIL = 0.020  # m: its base more than 20 mm above the glass, at any moment (latched 0)
PAN_TILT_FAIL = 10.0  # deg (latched 0)
GLASS_TILT_FAIL = 45.0  # deg: the glass of water tipped past 45 deg, at any moment (latched 0)
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
EPS = 1e-9


def cooktop_bounds():
    """(x0, x1, y0, y1) of the cooktop's outline (world)."""
    return CT_FRONT_X, CT_FRONT_X + CT_D, CT_CY - CT_W / 2, CT_CY + CT_W / 2


def over_cooktop(xy) -> bool:
    x0, x1, y0, y1 = cooktop_bounds()
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
