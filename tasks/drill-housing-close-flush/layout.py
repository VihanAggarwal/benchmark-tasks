"""Every dimension, position, mass, colour and threshold of the task, in one place (metres, kilograms, radians
unless a name says otherwise; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left of their midpoint, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z 0.7475.

The product is a compact 20 V cordless drill/driver's clamshell housing (two moulded halves meeting on a parting
plane through the motor's axis), its motor-gearbox unit and its trigger switch. Sizes marked MODEL are this file's
model of the real parts, estimated, NOT yet measured on a unit: they must be measured and corrected before real
trials (README "Sizes and sources"). The RS-550 motor can (35.8 mm across, 57.0 mm long) is the maker's published
size, rounded to 36 x 57 mm.

Housing frame ("local"): origin O on the motor's axis at the joint between motor can and gearbox, in the parting
plane; x along the drill's axis toward the chuck end, y across (the handle runs toward -y), z up from the parting
plane (the table is at z = -PARTING). The open left half lies fixed with its parting plane PARTING above the table;
the right half (the cover) closes on it. Seated, the motor-gearbox unit's frame and the closed cover's frame are both
at O; the switch's frame (its body centre) is at SWITCH_SEAT.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the housing frame in the world -----------------------------------------------------------------------------------
OX, OY = 0.300, 0.040  # O seen from above
PARTING = 0.028  # MODEL: the parting plane 28 mm above the table (half of a 56 mm wide barrel)
WALL = 0.0025  # MODEL: wall and shell thickness


def world(p):
    """A housing-frame point (x, y, z) in the world."""
    return (OX + p[0], OY + p[1], TOP + PARTING + p[2])


# -- outline (top view, housing frame): the barrel along x, the handle toward -y under the motor, the battery foot ---
BARREL_X = (-0.068, 0.063)  # MODEL: rear end wall outside .. nose wall outside (131 mm)
BARREL_Y = (-0.028, 0.028)  # MODEL: 56 mm across
HANDLE_X = (-0.045, 0.001)  # MODEL: 46 mm across, its front face (trigger side) toward +x
HANDLE_Y = (-0.128, -0.028)  # MODEL: 100 mm from the barrel to the battery foot
FOOT_X = (-0.090, 0.030)  # MODEL: the battery foot at the handle's end, 120 mm along the drill (the pack fits here)
FOOT_Y = (-0.158, -0.128)  # MODEL: 30 mm deep
OUTLINE = (("barrel", BARREL_X, BARREL_Y), ("handle", HANDLE_X, HANDLE_Y), ("foot", FOOT_X, FOOT_Y))
# Outline corners where a feeler gauge and a rule check the closure (the barrel's four, the foot's two far ones).
CORNERS = ((BARREL_X[0], BARREL_Y[0]), (BARREL_X[0], BARREL_Y[1]), (BARREL_X[1], BARREL_Y[0]),
           (BARREL_X[1], BARREL_Y[1]), (FOOT_X[0], FOOT_Y[0]), (FOOT_X[1], FOOT_Y[0]))

# -- the motor-gearbox unit (loose): unit frame on the axis at the can / gearbox joint, axis along +x (chuck end) -----
R_UNIT = 0.018  # can and gearbox both 36 mm across (the can: RS-550, 35.8 mm, published; the gearbox: MODEL)
UNIT_PARTS = (  # (name, x0, x1, radius)
    ("spindle", 0.045, 0.069, 0.005),  # MODEL: 10 mm output spindle, 24 mm out of the gearbox (the chuck goes on later)
    ("gearbox", 0.000, 0.045, R_UNIT),  # MODEL: planetary gearbox and clutch housing, 45 mm long
    ("can", -0.057, 0.000, R_UNIT),  # RS-550 can, 57 mm long (published)
    ("boss", -0.062, -0.057, 0.006),  # MODEL: rear bearing boss, 12 mm across, 5 mm long
)
UNIT_MASS = 0.315  # MODEL: can about 0.19 kg, gearbox 0.11 kg, spindle 0.015 kg (estimates; weigh the real unit)
UNIT_START = (0.300, 0.260, TOP + 0.006 + math.sqrt(R_UNIT**2 - 0.008**2))  # on the kitting tray's two rails
# Kitting tray (fixed): two rails along x, 6 mm tall, inner edges 16 mm apart, under the unit's middle.
TRAY_RAILS = (((0.270, 0.330), (0.246, 0.252)), ((0.270, 0.330), (0.268, 0.274)))
TRAY_H = 0.006

# -- the trigger switch (loose): frame at its body's centre -----------------------------------------------------------
SWITCH_BODY = (0.030, 0.024, 0.022)  # MODEL: x (along the barrel) x y x z (across the drill: it straddles the plane)
TRIGGER = ((0.0225, -0.003, 0.0), (0.015, 0.010, 0.012))  # MODEL: centre (switch frame) and size; it points +x
SWITCH_MASS = 0.040  # MODEL
SWITCH_SEAT = (-0.0185, -0.050, 0.0)  # housing frame
SWITCH_START = (0.280, -0.260, TOP + SWITCH_BODY[2] / 2)  # standing on the bench, trigger toward +x

# -- the cover (loose): the right half, frame at O when closed ---------------------------------------------------------
COVER_MASS = 0.110  # MODEL: moulded ABS half shell (its boxes: 102 cm3, 107 g of ABS)
COVER_START = (OX - 0.155, OY, TOP + 0.003)  # open side down on the bench in front of the housing, on its tongue
TONGUE_H = 0.003  # the cover's locating tongue hangs 3 mm below its rim, 2 mm inside the left half's walls
TONGUE_CLEAR = 0.002  # MODEL: real tongues fit within about 0.5 mm; 2 mm here (fidelity note)


def _b(x, y, z):
    """A box as ((x0, x1), (y0, y1), (z0, z1)) in a part's own frame."""
    return (tuple(x), tuple(y), tuple(z))


Z0 = -PARTING  # the table, in the housing frame
F = Z0 + WALL  # the shell's inside floor
W0, W1 = BARREL_Y[0] + WALL, BARREL_Y[1] - WALL  # barrel inside, -0.0255 .. +0.0255
REAR_IN, NOSE_IN = BARREL_X[0] + WALL, BARREL_X[1] - WALL  # -0.0655, +0.0605
HX_IN = (HANDLE_X[0] + WALL, HANDLE_X[1] - WALL)  # -0.0425 .. -0.0015
FX_IN = (FOOT_X[0] + WALL, FOOT_X[1] - WALL)  # -0.0875 .. +0.0275
FY_IN = (FOOT_Y[0] + WALL, FOOT_Y[1] - WALL)  # -0.1555 .. -0.1305 (the handle opens into the foot)
SPINDLE_NOTCH = 0.006  # half-width of the nose wall's spindle notch
SPINDLE_NOTCH_LOW = -0.0065  # its bottom (the spindle's underside is at -0.005)
SLOT_Y = (-0.0605, -0.0455)  # the trigger slot in the handle's front wall (the trigger spans y -0.058 .. -0.048)
SLOT_LOW = -0.0085  # its bottom (the trigger's underside is at -0.006)
# The cover's own notch and slot leave 3.5 mm round the spindle and the trigger, more than the tongue's 2 mm, so the
# tongue is what locates the cover.
C_NOTCH, C_NOTCH_HIGH = 0.0085, 0.0085  # half-width and top of the cover's spindle notch (spindle top at +0.005)
C_SLOT_Y, C_SLOT_HIGH = (-0.0615, -0.0445), 0.0095  # the cover's trigger slot (the trigger's top is at +0.006)


def _rib(x0, x1, r):
    """A cradle rib across the barrel at x0..x1 with a stepped notch for a cylinder of radius r lying on the axis:
    the part rests on the floor piece (top at -r) and is located by two steps per side (0.5 mm clear of it) and two
    cheeks (1.5 mm clear) that stop 4 mm below the parting plane, under the cover's tongue."""
    a1, a2, c = 0.5 * r, 0.8 * r, r + 0.0015
    h1 = r - math.sqrt(r * r - a1 * a1) - 0.0005
    h2 = r - math.sqrt(r * r - a2 * a2) - 0.0005
    out = [_b((x0, x1), (-a1, a1), (F, -r))]
    for s in (-1, 1):
        out.append(_b((x0, x1), sorted((s * a1, s * a2)), (F, -r + h1)))
        out.append(_b((x0, x1), sorted((s * a2, s * c)), (F, -r + h2)))
        out.append(_b((x0, x1), sorted((s * c, s * W1)), (F, -0.004)))
    return out


def _foot_walls():
    """The battery foot's walls (top view): its far end, rear and front, and the two pieces of its near side either
    side of the handle, which opens into it."""
    return {
        "foot_end": (FOOT_X, (FOOT_Y[0], FY_IN[0])),
        "foot_rear": ((FOOT_X[0], FX_IN[0]), (FY_IN[0], FY_IN[1])),
        "foot_front": ((FX_IN[1], FOOT_X[1]), (FY_IN[0], FY_IN[1])),
        "foot_near_rear": ((FOOT_X[0], HX_IN[0]), (FY_IN[1], FOOT_Y[1])),
        "foot_near_front": ((HX_IN[1], FOOT_X[1]), (FY_IN[1], FOOT_Y[1])),
    }


FOOT_WALLS = _foot_walls()

# The left half (fixed): shell floor, walls up to the parting plane, cradle ribs, the switch pocket, all one moulding.
LOWER = {
    "floor_barrel": _b(BARREL_X, BARREL_Y, (Z0, F)),
    "floor_handle": _b(HANDLE_X, HANDLE_Y, (Z0, F)),
    "floor_foot": _b(FOOT_X, FOOT_Y, (Z0, F)),
    "wall_top_side": _b(BARREL_X, (W1, BARREL_Y[1]), (F, 0.0)),
    "wall_low_rear": _b((BARREL_X[0], HANDLE_X[0]), (BARREL_Y[0], W0), (F, 0.0)),
    "wall_low_front": _b((HANDLE_X[1], BARREL_X[1]), (BARREL_Y[0], W0), (F, 0.0)),
    "wall_rear": _b((BARREL_X[0], REAR_IN), (W0, W1), (F, 0.0)),
    "wall_nose_a": _b((NOSE_IN, BARREL_X[1]), (W0, -SPINDLE_NOTCH), (F, 0.0)),
    "wall_nose_b": _b((NOSE_IN, BARREL_X[1]), (SPINDLE_NOTCH, W1), (F, 0.0)),
    "wall_nose_c": _b((NOSE_IN, BARREL_X[1]), (-SPINDLE_NOTCH, SPINDLE_NOTCH), (F, SPINDLE_NOTCH_LOW)),
    "handle_back": _b((HANDLE_X[0], HX_IN[0]), (HANDLE_Y[0], BARREL_Y[0]), (F, 0.0)),
    "handle_front_a": _b((HX_IN[1], HANDLE_X[1]), (HANDLE_Y[0], SLOT_Y[0]), (F, 0.0)),
    "handle_front_b": _b((HX_IN[1], HANDLE_X[1]), (SLOT_Y[1], BARREL_Y[0]), (F, 0.0)),
    "handle_front_c": _b((HX_IN[1], HANDLE_X[1]), SLOT_Y, (F, SLOT_LOW)),
    **{k: _b(x, y, (F, 0.0)) for k, (x, y) in FOOT_WALLS.items()},
    # axial stops: the gearbox's front face and the can's rear face each 3.5 mm from a low rib
    "stop_front": _b((0.0485, 0.0505), (W0, W1), (F, -0.012)),
    "stop_rear": _b((-0.0625, -0.0605), (W0, W1), (F, -0.010)),
    # the switch pocket: a standoff under the body, a rib behind it and one on each side, 3 mm clear
    "switch_standoff": _b((-0.030, -0.007), (-0.058, -0.042), (F, -SWITCH_BODY[2] / 2)),
    "switch_rib_back": _b((-0.0385, -0.0365), (-0.066, -0.034), (F, -0.003)),
    "switch_rib_low": _b((-0.0365, HX_IN[1]), (-0.067, -0.065), (F, -0.003)),
    "switch_rib_high": _b((-0.0365, HX_IN[1]), (-0.035, -0.033), (F, -0.003)),
}
for _name, _x in (("rib_gearbox", (0.0215, 0.0235)), ("rib_can", (-0.0375, -0.0355))):
    for _i, _piece in enumerate(_rib(*_x, R_UNIT)):
        LOWER[f"{_name}_{_i}"] = _piece

# The cover (the right half), in its own frame (= the housing frame when closed), open side down: rim at z 0, shell
# 25.5 .. 28 mm up, the same outline, matching notches at the parting plane, ribs that come down over the motor and
# the switch with 1.5 mm to spare, and the locating tongue.
C_TOP = PARTING - WALL  # 0.0255: underside of the cover's shell
T_IN = W1 - TONGUE_CLEAR  # 0.0235: the tongue's outer face across the barrel
COVER = {
    "shell_barrel": _b(BARREL_X, BARREL_Y, (C_TOP, PARTING)),
    "shell_handle": _b(HANDLE_X, HANDLE_Y, (C_TOP, PARTING)),
    "shell_foot": _b(FOOT_X, FOOT_Y, (C_TOP, PARTING)),
    "wall_top_side": _b(BARREL_X, (W1, BARREL_Y[1]), (0.0, C_TOP)),
    "wall_low_rear": _b((BARREL_X[0], HANDLE_X[0]), (BARREL_Y[0], W0), (0.0, C_TOP)),
    "wall_low_front": _b((HANDLE_X[1], BARREL_X[1]), (BARREL_Y[0], W0), (0.0, C_TOP)),
    "wall_rear": _b((BARREL_X[0], REAR_IN), (W0, W1), (0.0, C_TOP)),
    "wall_nose_a": _b((NOSE_IN, BARREL_X[1]), (W0, -C_NOTCH), (0.0, C_TOP)),
    "wall_nose_b": _b((NOSE_IN, BARREL_X[1]), (C_NOTCH, W1), (0.0, C_TOP)),
    "wall_nose_c": _b((NOSE_IN, BARREL_X[1]), (-C_NOTCH, C_NOTCH), (C_NOTCH_HIGH, C_TOP)),
    "handle_back": _b((HANDLE_X[0], HX_IN[0]), (HANDLE_Y[0], BARREL_Y[0]), (0.0, C_TOP)),
    "handle_front_a": _b((HX_IN[1], HANDLE_X[1]), (HANDLE_Y[0], C_SLOT_Y[0]), (0.0, C_TOP)),
    "handle_front_b": _b((HX_IN[1], HANDLE_X[1]), (C_SLOT_Y[1], BARREL_Y[0]), (0.0, C_TOP)),
    "handle_front_c": _b((HX_IN[1], HANDLE_X[1]), C_SLOT_Y, (C_SLOT_HIGH, C_TOP)),
    **{k: _b(x, y, (0.0, C_TOP)) for k, (x, y) in FOOT_WALLS.items()},
    # ribs over the motor (at the left half's rib stations): over the unit 1.5 mm above its top, cheeks outside it
    "rib_gearbox_top": _b((0.0215, 0.0235), (-0.021, 0.021), (R_UNIT + 0.0015, C_TOP)),
    "rib_gearbox_side_a": _b((0.0215, 0.0235), (W0, -0.021), (0.004, C_TOP)),
    "rib_gearbox_side_b": _b((0.0215, 0.0235), (0.021, W1), (0.004, C_TOP)),
    "rib_can_top": _b((-0.0375, -0.0355), (-0.021, 0.021), (R_UNIT + 0.0015, C_TOP)),
    "rib_can_side_a": _b((-0.0375, -0.0355), (W0, -0.021), (0.004, C_TOP)),
    "rib_can_side_b": _b((-0.0375, -0.0355), (0.021, W1), (0.004, C_TOP)),
    "switch_cap": _b((-0.034, -0.003), (-0.063, -0.037), (SWITCH_BODY[2] / 2 + 0.0015, C_TOP)),
    # the locating tongue: 2 mm strips hanging 3 mm below the rim, TONGUE_CLEAR inside the left half's walls
    "tongue_top": _b((-0.060, 0.056), (T_IN - 0.002, T_IN), (-TONGUE_H, 0.0)),
    "tongue_low_rear": _b((-0.060, -0.048), (-T_IN, -T_IN + 0.002), (-TONGUE_H, 0.0)),
    "tongue_low_front": _b((0.004, 0.056), (-T_IN, -T_IN + 0.002), (-TONGUE_H, 0.0)),
    # (the rear strip is split round the motor's rear boss, 6 mm across the axis)
    "tongue_rear_a": _b((REAR_IN + TONGUE_CLEAR, REAR_IN + TONGUE_CLEAR + 0.002), (-0.018, -0.0085), (-TONGUE_H, 0.0)),
    "tongue_rear_b": _b((REAR_IN + TONGUE_CLEAR, REAR_IN + TONGUE_CLEAR + 0.002), (0.0085, 0.018), (-TONGUE_H, 0.0)),
    "tongue_handle_back": _b((HX_IN[0] + TONGUE_CLEAR, HX_IN[0] + TONGUE_CLEAR + 0.002), (-0.120, -0.070),
                             (-TONGUE_H, 0.0)),
    "tongue_handle_front": _b((HX_IN[1] - TONGUE_CLEAR - 0.002, HX_IN[1] - TONGUE_CLEAR), (-0.120, -0.070),
                              (-TONGUE_H, 0.0)),
    "tongue_foot_end": _b((FX_IN[0] + 0.010, FX_IN[1] - 0.010),
                          (FY_IN[0] + TONGUE_CLEAR, FY_IN[0] + TONGUE_CLEAR + 0.002), (-TONGUE_H, 0.0)),
    "tongue_foot_rear": _b((FX_IN[0] + TONGUE_CLEAR, FX_IN[0] + TONGUE_CLEAR + 0.002), (-0.150, -0.136),
                           (-TONGUE_H, 0.0)),
    "tongue_foot_front": _b((FX_IN[1] - TONGUE_CLEAR - 0.002, FX_IN[1] - TONGUE_CLEAR), (-0.150, -0.136),
                            (-TONGUE_H, 0.0)),
}

# The nest (fixed): three aluminium locating blocks round the left half, 10 mm tall, 1 mm off its outline.
NEST = {
    "nest_nose": _b((BARREL_X[1] + 0.001, BARREL_X[1] + 0.011), (-0.020, 0.020), (Z0, Z0 + 0.010)),
    "nest_top": _b((-0.040, 0.040), (BARREL_Y[1] + 0.001, BARREL_Y[1] + 0.011), (Z0, Z0 + 0.010)),
    "nest_end": _b((FOOT_X[0] + 0.020, FOOT_X[1] - 0.020), (FOOT_Y[0] - 0.011, FOOT_Y[0] - 0.001), (Z0, Z0 + 0.010)),
}

# -- the grippers (rigstate: opening 0 closed .. 1 fully open, 93.9 mm) ------------------------------------------------
FULL_GAP = 0.0939
HELD_OPENING = 0.90  # 'held': a gripper less than 90 % open (pads under 85 mm apart) ...
HELD_REACH = 0.015  # ... with its pad-tip point within 15 mm of the object's surface, and no nearer another's
UNREADABLE = "unreadable"  # on a step where the arms cannot be read, every loose object counts as held

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------------
PARTS = ("motor", "switch")
LOOSE = PARTS + ("cover",)
STILL_STEPS = 10  # 'at rest' is judged over the last 0.5 s (10 control steps) ...
STILL_PATH = 0.003  # ... the frame origin's path over that window shorter than 3 mm
MOTOR_AXIAL_TOL = 0.004  # seated motor: frame within 4 mm of O along the axis (the stops leave 3.5 mm each way) ...
MOTOR_SIDE_TOL = 0.002  # ... within 2 mm across ...
MOTOR_HEIGHT_TOL = 0.0015  # ... and within 1.5 mm in height (its axis on the parting plane) ...
MOTOR_AXIS_DEG = 3.0  # ... its axis within 3 degrees of the housing's, spindle toward the nose
SWITCH_XY_TOL = 0.003  # seated switch: body centre within 3 mm of its seat seen from above ...
SWITCH_HEIGHT_TOL = 0.0015  # ... and 1.5 mm in height (on its standoff) ...
SWITCH_TILT_DEG = 5.0  # ... upright within 5 degrees ...
SWITCH_YAW_DEG = 8.0  # ... trigger within 8 degrees of +x (through its slot)
RIM_GAP = 0.0015  # closed: every outline corner of the cover within 1.5 mm of the parting plane in height ...
CORNER_STEP = 0.004  # ... and within 4 mm of the left half's matching corner seen from above
MOTOR_SCORE = 0.3
SWITCH_SCORE = 0.2
COVER_ON_SCORE = 0.7  # both seated and the cover lying on the housing, not closed
RESET_JUMP = 0.08  # a loose object moving this far in one control step was put back by the simulator ...
START_TOL = 0.045  # ... onto its starting place: within this of it

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
HOUSING_ORANGE = (0.95, 0.45, 0.08)
CAN_SILVER = (0.72, 0.72, 0.74)
GEARBOX_BLACK = (0.10, 0.10, 0.11)
SPINDLE_STEEL = (0.55, 0.56, 0.58)
SWITCH_BLACK = (0.12, 0.12, 0.13)
TRIGGER_GREY = (0.35, 0.35, 0.37)
NEST_GREY = (0.62, 0.63, 0.65)
TRAY_GREY = (0.45, 0.46, 0.48)
