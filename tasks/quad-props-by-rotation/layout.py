"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A quadcopter frame (no propellers) is fixed on the midline in front of the arms, its front (+x) away from them. On
an X quad neighbouring motors turn opposite ways: front left and rear right turn clockwise (CW) seen from above,
front right and rear left counter-clockwise (CCW). A red arrow painted beside each motor shows its direction. Six
propellers lie on two foam mats in front of the arms: three CW and three CCW. A propeller's hand is fixed by its
blades: the raised leading edge of each blade must lead in the motor's direction. No turn about the vertical makes
a CW propeller into a CCW one, so each motor needs a propeller of its own hand.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the quadcopter frame: all fixed ----------------------------------------------------------------------------------
CENTER = (0.300, 0.0)  # the middle of the frame
MOTOR_DX, MOTOR_DY = 0.080, 0.100  # each motor's axis this far from the middle along x and y
MOTOR_R = 0.014  # motor bell: 28 mm across,
MOTOR_H = 0.030  # 30 mm tall, standing on the end of its arm
ARM_W, ARM_H = 0.016, 0.008  # each frame arm: 16 mm wide, 8 mm thick, flat on the table
MOTOR_Z0 = ARM_H  # the motors stand on the arms' ends
MOTOR_TOP = MOTOR_Z0 + MOTOR_H  # 0.038: the motor's top face, where a propeller's hub sits
DIAG = math.hypot(MOTOR_DX, MOTOR_DY)  # 0.128: the middle to each motor's axis
ARM_R0 = 0.020  # each arm runs along its diagonal from 20 mm from the middle (under the centre plate)
ARM_R1 = DIAG + MOTOR_R  # ... to the far side of its motor (142 mm)
ARM_LEN = ARM_R1 - ARM_R0
PLATE_SIZE = (0.070, 0.070, 0.012)  # the centre plate (flight-controller stack) on top of the arms' inner ends
PLATE_Z0 = ARM_H

# (name, x, y, direction seen from above). 'front' is +x (away from the arms), 'left' is +y (the robot's left).
MOTORS = (
    ("front_left", CENTER[0] + MOTOR_DX, CENTER[1] + MOTOR_DY, "CW"),
    ("front_right", CENTER[0] + MOTOR_DX, CENTER[1] - MOTOR_DY, "CCW"),
    ("rear_left", CENTER[0] - MOTOR_DX, CENTER[1] + MOTOR_DY, "CCW"),
    ("rear_right", CENTER[0] - MOTOR_DX, CENTER[1] - MOTOR_DY, "CW"),
)
MOTOR_XY = {n: (x, y) for n, x, y, _ in MOTORS}
MOTOR_DIR = {n: d for n, _, _, d in MOTORS}


def outboard(name: str) -> tuple[float, float]:
    """Unit vector from the frame's middle out through a motor's axis (along its arm)."""
    x, y = MOTOR_XY[name]
    return ((x - CENTER[0]) / DIAG, (y - CENTER[1]) / DIAG)


def turning(name: str) -> tuple[float, float]:
    """Unit vector of the motor's direction of turn at its outboard side, seen from above (tangent to a circle round
    its axis): clockwise is (u_y, -u_x) for the outboard vector u, counter-clockwise the opposite."""
    ux, uy = outboard(name)
    return (uy, -ux) if MOTOR_DIR[name] == "CW" else (-uy, ux)


# -- the rotation arrows: red paint on the table round each motor's outboard side: a curved shaft (an arc of a circle
#    round the motor's axis, drawn as three straight plates) ending in a two-barb head that points the way it turns ---
ARROW_R = 0.032  # the arc's centre line 32 mm from the motor's axis,
ARC_DEG = 100.0  # spanning 100 degrees, centred on the arm's line (outboard), in
ARC_SEGMENTS = 3  # three straight plates (chords of 33.3 degrees each),
ARROW_W = 0.005  # 5 mm wide
BARB_LEN = 0.012  # two 12 mm barbs from the arc's leading end make the head,
BARB_DEG = 40.0  # each swept back 40 degrees from the direction of turn
LAYER = 0.001  # paint plates 1 mm thick; where two would cross, the later one lies on the one below


def arrow_parts(name: str) -> list[tuple[str, tuple[float, float], float, tuple[float, float, float], float]]:
    """(part, centre xy, yaw, size, bottom height) of the five plates of a motor's arrow: three arc plates from the
    trailing end to the leading end, then the two barbs at the leading end."""
    mx, my = MOTOR_XY[name]
    ux, uy = outboard(name)
    sense = -1.0 if MOTOR_DIR[name] == "CW" else 1.0  # clockwise seen from above: the angle decreases
    phi0 = math.atan2(uy, ux) - sense * math.radians(ARC_DEG) / 2  # the trailing end
    step = sense * math.radians(ARC_DEG) / ARC_SEGMENTS
    chord = 2 * ARROW_R * math.sin(abs(step) / 2)
    parts = []
    for k in range(ARC_SEGMENTS):
        a0, a1 = phi0 + k * step, phi0 + (k + 1) * step
        p0 = (mx + ARROW_R * math.cos(a0), my + ARROW_R * math.sin(a0))
        p1 = (mx + ARROW_R * math.cos(a1), my + ARROW_R * math.sin(a1))
        parts.append((f"arc{k + 1}", ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
                      math.atan2(p1[1] - p0[1], p1[0] - p0[0]), (chord + ARROW_W, ARROW_W, LAYER), LAYER * (k % 2)))
    a_end = phi0 + ARC_SEGMENTS * step
    tip = (mx + ARROW_R * math.cos(a_end), my + ARROW_R * math.sin(a_end))
    ahead = a_end + sense * math.pi / 2  # the direction of turn at the leading end
    for k, side in enumerate((+1.0, -1.0)):
        a = ahead + math.pi + side * math.radians(BARB_DEG)
        c = (tip[0] + BARB_LEN / 2 * math.cos(a), tip[1] + BARB_LEN / 2 * math.sin(a))
        parts.append((f"barb{k + 1}", c, a, (BARB_LEN, ARROW_W, LAYER), LAYER * (2 + k)))
    return parts


# -- the two foam mats the propellers lie on --------------------------------------------------------------------------
MAT_X = 0.200
MAT_Y = 0.285  # the left mat centred on y +0.285, the right on -0.285
MAT_SIZE = (0.140, 0.180, 0.003)  # 140 mm along x (the blades lie along x), 180 mm along y
MAT_TOP = MAT_SIZE[2]

# -- the propellers: one loose rigid body each (mechkit compound). Base frame at the centre of the hub's bottom face,
#    local +z up the hub's axis, local +x along the blades ------------------------------------------------------------
HUB_R = 0.011  # hub: 22 mm across,
HUB_H = 0.034  # 34 mm tall (tall enough to grip above the blades)
BLADE_IN, BLADE_OUT = 0.008, 0.063  # each blade from 8 mm (inside the hub) to 63 mm from the axis: a 126 mm span
BLADE_W, BLADE_H = 0.018, 0.004  # 18 mm wide (chord), 4 mm thick, at the hub's foot
EDGE_IN = HUB_R  # the raised leading edge: a strip along the blade's leading side, from the hub to the tip,
EDGE_W, EDGE_H = 0.006, 0.003  # 6 mm wide and 3 mm tall on top of the blade
EDGE_Y = BLADE_W / 2 - EDGE_W / 2  # 6 mm off the blade's centre line, flush with its leading side
PROP_MASS = 0.020
HUB_CENTER = (0.0, 0.0, HUB_H / 2)
BLADE_CENTERS = (((BLADE_IN + BLADE_OUT) / 2, 0.0, BLADE_H / 2), (-(BLADE_IN + BLADE_OUT) / 2, 0.0, BLADE_H / 2))
BLADE_SIZE = (BLADE_OUT - BLADE_IN, BLADE_W, BLADE_H)
EDGE_SIZE = (BLADE_OUT - EDGE_IN, EDGE_W, EDGE_H)
EDGE_X = (EDGE_IN + BLADE_OUT) / 2
EDGE_Z = BLADE_H + EDGE_H / 2


def edge_centers(hand: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The two leading-edge strips' centres in the propeller's frame. Turning clockwise seen from above, the +x blade
    moves toward -y and the -x blade toward +y, so a CW propeller's leading edges are on those sides; a CCW one is
    its mirror image."""
    s = -1.0 if hand == "CW" else 1.0
    return ((EDGE_X, s * EDGE_Y, EDGE_Z), (-EDGE_X, -s * EDGE_Y, EDGE_Z))


TIP_LOCAL = ((BLADE_OUT, 0.0, BLADE_H / 2), (-BLADE_OUT, 0.0, BLADE_H / 2))  # the blade tips (stillness check)
HUB_TOP_LOCAL = (0.0, 0.0, HUB_H)

PROPS = ("prop_1", "prop_2", "prop_3", "prop_4", "prop_5", "prop_6")
# Each mat: the hub nearest the midline, the middle one, the outer one (y +-0.230, +-0.285, +-0.340), blades along x.
# The left mat holds a CW and a CCW propeller for the left side's two motors and a spare CCW one; the right mat its
# mirror image (a CCW, a CW and a spare CW). Each arm's far rear motor needs the hand its own mat has only one of.
PROP_START = {
    "prop_1": (MAT_X, +0.230),
    "prop_2": (MAT_X, +0.285),
    "prop_3": (MAT_X, +0.340),
    "prop_4": (MAT_X, -0.230),
    "prop_5": (MAT_X, -0.285),
    "prop_6": (MAT_X, -0.340),
}
HAND = {"prop_1": "CW", "prop_2": "CCW", "prop_3": "CCW", "prop_4": "CCW", "prop_5": "CW", "prop_6": "CW"}

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
CENTRE_TOL = 0.006  # 'on a motor': the hub's axis within 6 mm of the motor's axis (at the hub's bottom face),
SEAT_TOL = 0.003  # the hub's bottom face within 3 mm of the motor's top (above or below it),
UPRIGHT_DEG = 8.0  # and the hub's axis within 8 degrees of straight up (upside down: within 8 degrees of down)
STILL_WINDOW_S = 0.5  # at rest: the hub's bottom and top and both blade tips each moved at most STILL_TOL in 0.5 s
STILL_TOL = 0.002
PER_MOTOR = 0.25
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # propellers moving this far in one control step were teleported: a new episode began
START_TOL = 0.025  # ... onto their starting marks: within this of them (the marks are 55 mm apart)

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
CARBON = (0.06, 0.06, 0.07)  # frame arms and centre plate
MOTOR_GREY = (0.30, 0.31, 0.33)  # anodised motor bells
ARROW_RED = (0.80, 0.08, 0.08)
MAT_GREY = (0.72, 0.73, 0.75)  # light grey foam
HUB_GREY = (0.48, 0.49, 0.50)
BLADE_BLACK = (0.05, 0.05, 0.05)
EDGE_WHITE = (0.96, 0.96, 0.95)
