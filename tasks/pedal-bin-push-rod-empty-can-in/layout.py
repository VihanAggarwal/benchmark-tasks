"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. The bin's own frame (parts.py, scoring.py) has its origin on the
table under the bin's axis, x toward the hinge (away from the arms), z up; heights are above the table top.

The bin is a Brabantia NewIcon 3 L pedal bin (model 233760): published 264 mm high, 168 mm across, 235 mm deep with
the pedal and the hinge, 411 mm high with the lid open, 0.83 kg, soft-closing lid (README: source). Everything inside
it is an estimate (est in task.json) with a measurement step in the README. Its linkage is real geometry: the pedal is
one end of a lever under the bucket floor; the lever's other end, behind the bucket, lifts a loose push rod in a slot in
the back wall; the rod's flat top lifts a pin on an arm under the lid's hinge, so the lid opens. Let go, the lid's own
weight pushes the rod and the lever back (the pedal comes up); the hinge's damping is the soft close. No spring and no
software rule.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the bin: placement and body ---------------------------------------------------------------------------------------
BIN_X, BIN_Y = 0.330, 0.000  # the bin's axis on the table (world)
OUT_R = 0.084  # 168 mm across (published)
IN_R = 0.072  # the clear opening inside the rim, 144 mm (est: steel body, gap and plastic inner bucket in one wall)
RIM = 0.248  # the rim, where the lid sits (est: 264 mm published to the top of the lid, a 15.5 mm lid)
SEGMENTS = 16  # the round wall as 16 flat slabs
FLOOR_TOP = 0.050  # the inner bucket's floor, 50 mm above the table (est); the lever runs under it
FLOOR_T = 0.008
BIN_MASS = 0.83  # published: the whole bin with its inner bucket
LID_MASS = 0.12  # est
LEVER_MASS = 0.06  # est: the pedal and its lever
ROD_MASS = 0.015  # est: the push rod (a separate loose part)
BODY_MASS = BIN_MASS - LID_MASS - LEVER_MASS - ROD_MASS
BIN_FRICTION = (0.8, 0.7)  # the published non-slip base (est), used for every surface of the bin
ROD_FRICTION = (0.15, 0.1)  # a smooth rod (est)
GREY = (0.47, 0.48, 0.48)  # the listing's concrete grey
LID_GREY = (0.43, 0.44, 0.44)
PEDAL_GREY = (0.20, 0.20, 0.21)
ROD_COLOR = (0.70, 0.71, 0.72)

# -- the slot in the back wall and the push rod ----------------------------------------------------------------------
SLOT_X0, SLOT_X1 = 0.07325, 0.08325  # the slot through the back wall, 10 mm (x) by 9 mm (y), table to rim
SLOT_HY = 0.0045
ROD_X = 0.07825  # the rod's centre line (bin frame), 8 x 7 mm: 1 mm of play each way in the slot
ROD_TX, ROD_TY = 0.008, 0.007

# -- the lid ---------------------------------------------------------------------------------------------------------
HINGE_X = 0.0915  # the hinge pin, 7.5 mm behind the body (bin frame); 235 mm deep in all with the pedal
HINGE_Z = RIM + 0.0005  # the lid's underside plane runs through the hinge axis; drawn 0.5 mm over the rim
LID_R = OUT_R
LID_T = 0.008  # the lid's flat plate
CAP_R, CAP_H = 0.070, 0.007  # its raised middle: the top of the lid 263.5 mm over the table (264 published)
LID_UPPER_DEG = 75.0  # the hinge's own stop (the lever's stop comes first)
LID_UPPER = math.radians(LID_UPPER_DEG)
LID_DAMP = 0.10  # N m s/rad: the soft close, about 1.4 s from fully open (est)
PIN_DX, PIN_DZ = 0.012, 0.0081  # the pin under the hinge: 12 mm in front of the hinge axis, 8.1 mm below it
PIN_X, PIN_Z = HINGE_X - PIN_DX, HINGE_Z - PIN_DZ
PIN_R, PIN_LEN = 0.003, 0.006
PIN_D = math.hypot(PIN_DX, PIN_DZ)  # 14.5 mm from the hinge axis
PIN_PHI0 = -math.atan2(PIN_DZ, PIN_DX)  # -34 degrees: the pin's angle below the lid plane, in front of the hinge
TOP_GAP = 0.0005  # the rod's top 0.5 mm under the pin with the lid shut

# -- the pedal lever -------------------------------------------------------------------------------------------------
ROLLER_X = 0.0793  # the lever's rear end: a round nose (r 3 mm) under the rod
ROLLER_R = 0.003
LEVER_REAR = 0.069  # pivot to the nose
PIVOT_X = ROLLER_X - LEVER_REAR  # 10.3 mm behind the bin's axis, under the floor
PIVOT_Z = 0.020
PLATE_X0, PLATE_X1 = -0.138, -0.084  # the pedal plate: 54 mm out in front of the body, 70 mm wide
PLATE_HY = 0.035
PLATE_TOP = 0.048  # its top 48 mm over the table at rest (est)
PLATE_T = 0.005
PRESS_X = (PLATE_X0 + PLATE_X1) / 2  # where the demo presses: the plate's middle, 121 mm in front of the pivot
LEVER_DAMP = 0.005
ROD_BOT = PIVOT_Z + ROLLER_R  # the rod stands on the nose
ROD_TOP = PIN_Z - PIN_R - TOP_GAP
ROD_LEN = ROD_TOP - ROD_BOT


def pin_rise(lid_rad: float) -> float:
    """How far the pin's lowest point rises (m) when the lid opens to ``lid_rad``."""
    return PIN_D * (math.sin(lid_rad + PIN_PHI0) - math.sin(PIN_PHI0))


def lid_from_rod(rise: float) -> float:
    """The lid angle (rad) a rod raised ``rise`` metres holds the lid at (the lid resting on it; 0 below the gap)."""
    s = math.sin(PIN_PHI0) + max(0.0, float(rise) - TOP_GAP) / PIN_D
    if s >= 1.0:
        return math.pi / 2 - PIN_PHI0
    return max(0.0, math.asin(s) - PIN_PHI0)


def rod_from_lever(lever_rad: float) -> float:
    """How far the lever's nose lifts the rod (m) at lever angle ``lever_rad`` (positive: pedal down)."""
    return LEVER_REAR * math.sin(max(0.0, float(lever_rad)))


LID_OPEN_DEG = 67.0  # fully pressed: the lid's front edge 413 mm over the table (411 mm published)
LEVER_UPPER = math.asin((pin_rise(math.radians(LID_OPEN_DEG)) + TOP_GAP) / LEVER_REAR)  # the pedal's stop, 14 deg


def plate_point(x: float, lever_rad: float) -> tuple[float, float]:
    """Where the plate's top-surface point drawn at bin-frame ``x`` is (x, z) with the lever at ``lever_rad``."""
    dx, dz = x - PIVOT_X, PLATE_TOP - PIVOT_Z
    c, s = math.cos(lever_rad), math.sin(lever_rad)
    return PIVOT_X + dx * c - dz * s, PIVOT_Z + dz * c + dx * s


PEDAL_TRAVEL = PLATE_TOP - plate_point(PRESS_X, LEVER_UPPER)[1]  # about 30 mm at the plate's middle


def lever_from_press(tip_z: float) -> float:
    """The lever angle (rad, 0..LEVER_UPPER) at which the plate's middle is at height ``tip_z`` (bin frame)."""
    lo, hi = 0.0, LEVER_UPPER
    if plate_point(PRESS_X, 0.0)[1] <= tip_z:
        return 0.0
    if plate_point(PRESS_X, hi)[1] >= tip_z:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if plate_point(PRESS_X, mid)[1] > tip_z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -- the pedal force (est; README) -------------------------------------------------------------------------------------
LID_CG_X = -HINGE_X  # the lid's centre of mass, over the bin's axis: 91.5 mm in front of the hinge (est)
LID_CG_Z = 0.004
LEVER_CG_TORQUE = 0.046  # N m: the lever's own weight about its pivot (60 g, its centre 79 mm in front), pedal down


def pedal_force_range() -> tuple[float, float]:
    """The pedal force (N) at the plate's middle: at the start of its travel and at its stop."""
    return pedal_force(0.0, LEVER_CG_TORQUE), pedal_force(math.radians(LID_OPEN_DEG), LEVER_CG_TORQUE)


def pedal_force(lid_rad: float, lever_cg_torque: float) -> float:
    """The vertical force (N) at the plate's middle that holds the lid at ``lid_rad`` (no friction; the lever's own
    weight ``lever_cg_torque`` N m helps)."""
    torque = LID_MASS * G * (-LID_CG_X * math.cos(lid_rad) - LID_CG_Z * math.sin(lid_rad))
    dz_dth = PIN_D * math.cos(lid_rad + PIN_PHI0)  # m of pin rise per rad of lid
    rod = torque / dz_dth + ROD_MASS * G
    rise = pin_rise(lid_rad) + TOP_GAP
    lever = math.asin(min(1.0, rise / LEVER_REAR))
    lever_torque = rod * LEVER_REAR * math.cos(lever) - lever_cg_torque * math.cos(lever)
    return max(0.0, lever_torque / ((PIVOT_X - PRESS_X) * math.cos(lever)))


# -- the can (loose) -------------------------------------------------------------------------------------------------
CAN_R = 0.033  # a 330 ml drink can: 66 mm across, 115 mm tall (README: source); empty, about 14 g (est)
CAN_H = 0.115
CAN_MASS = 0.014
CAN_X, CAN_Y = 0.260, 0.240  # standing on the table in front of the left arm, left of the bin
CAN_FRICTION = (0.6, 0.5)  # aluminium on the pads and the table (est)
CAN_COLOR = (0.76, 0.12, 0.14)  # a plain red can, no print

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
IN_DEPTH = 0.010  # in the bin: the can's highest point at least 10 mm below the rim ...
IN_R_SCORE = IN_R  # ... its centre inside the 144 mm opening (top view) ...
FLOOR_TOL = 0.004  # ... and its lowest point no lower than 4 mm under the bucket floor
SHUT_DEG = 5.0  # shut: the lid within 5 degrees of closed on every step of the last second
KEPT_DIST = 0.030  # the bin ended within 30 mm of where it started ...
KEPT_TILT = 5.0  # ... and tilted under 5 degrees
SLID_DIST = 0.100  # latched zero: the bin slid more than 100 mm
TIP_DEG = 30.0  # latched zero: the bin tipped more than 30 degrees
TOUCH = 0.001  # a finger zone within 1 mm of the lid counts as touching it
CAN_TOUCH = 0.001  # the can within 1 mm of the lid (before it is in the bin) counts as hitting it
# Pushed (latched, scores 0): the lid moved by anything but the pedal linkage. LINK = the lower of the lid angle the rod
# holds and the one the pedal's angle gives through the linkage (GPU 17 Nov demo: the lid within 0.12 degrees of it).
FORCED_DEG = 5.0  # the lid 5 degrees above LINK and rising (lifted), or 5 above the rod's angle and not closing ...
HELD_STEPS = 3  # ... on 3 steps running (held open: a free lid falls 1.1 degrees a step or faster, layout + task.py)
RISE_DEG = 0.3  # rising: the lid up more than 0.3 degrees in one step; not closing: down less than that
CAN_LIFT_DEG = 2.0  # the can within 1 mm of the lid while the lid rises 2 degrees above LINK: the can lifts it
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible motion: 5 mm and 5 degrees over 1 s
STILL_TURN_DEG = 5.0
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
