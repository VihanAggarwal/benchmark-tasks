"""Every dimension and threshold of the task, in one place (metres, kilograms, degrees where named, the station's
world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.
Angles about a wheel's axis are measured seen from above, counter-clockwise from +x (toward the far side of the
table), in degrees.

A 1:10 model of a tyre fitter's bench: two wheels, each standing on its own changer turntable, and two tyres lying
beside them, one in front of each wheel. Wheel A (left, +y) carries a white low-point (minimum radial runout) mark;
wheel B (right, -y) carries none. Each wheel has a valve stem. Each tyre carries a red dot (uniformity high point)
and a yellow dot (light point) on its top sidewall, 100 degrees apart.
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the changer turntables (fixed, painted steel discs) ---------------------------------------------------------------
TURNTABLE_RADIUS = 0.027  # 54 mm across: inside the tyre's 64 mm, so fingers holding a tyre from outside never touch it
TURNTABLE_H = 0.006
TT_TOP = TURNTABLE_H  # 6 mm up: a seated tyre's underside rests here

# -- the wheels (fixed, painted steel), one on each turntable ----------------------------------------------------------
WHEELS = {  # name: (axis x, axis y, has the white low-point mark)
    "wheel_a": (0.340, 0.150, True),
    "wheel_b": (0.340, -0.150, False),
}
WHEEL_NAMES = tuple(WHEELS)
MARKED_WHEEL, PLAIN_WHEEL = "wheel_a", "wheel_b"
WHEEL_RADIUS = 0.018  # 36 mm across: 5 mm of radial clearance inside the tyre's 46 mm bore
WHEEL_H = 0.024  # on the turntable: its top face is 30 mm up, 2 mm above a seated tyre's top
WHEEL_TOP = TT_TOP + WHEEL_H  # 0.030
# valve stem: a black post on the wheel's top face, 10 mm off the axis toward +x (angle 0)
VALVE_ANGLE = 0.0
VALVE_R = 0.010
VALVE_RADIUS = 0.003  # 6 mm across: it reaches 13 mm from the axis, well inside the tyre's 23 mm bore radius
VALVE_H = 0.008  # its top is 38 mm up
VALVE_TOP = WHEEL_TOP + VALVE_H
# low-point mark: a white painted dot on wheel A's top face, 10 mm off the axis at 120 degrees
MARK_ANGLE = 120.0
MARK_R = 0.010
MARK_RADIUS = 0.004  # 8 mm across
MARK_H = 0.001

# -- the tyres (loose; each one rigid mechkit body: a ring of 36 overlapping upright rubber ribs and two painted dots) --
# mechkit's shapes are boxes and cylinders aligned with the body's frame, so the ring is built from 36 upright
# cylinders (ribs, 9 mm across) on a 27.5 mm radius, 10 degrees apart: together a 64 mm ring with a 46 mm bore whose
# outside is scalloped 0.7 mm deep (a ribbed tread) and whose bore is scalloped 0.6 mm. Base frame: origin at the
# centre of the ring's bottom face, local +z up through the dotted (top) sidewall, local +x toward the red dot. The
# ribs sit at local 5, 15, 25, ... degrees, so across local 0/180 and 90/270 a flat finger pad rests on two ribs (a
# stable grip that cannot roll the ring toward a lower width).
TYRE_OD, TYRE_ID, TYRE_H = 0.064, 0.046, 0.022
TYRE_R_OUT, TYRE_R_IN = TYRE_OD / 2, TYRE_ID / 2
N_RIBS = 36
RIB_RADIUS = 0.0045
RIB_R = 0.0275  # rib centres: 27.5 mm from the axis (outer crest 32 mm, inner crest 23 mm)
RIB_OFFSET_DEG = 5.0
RIB_STEP_DEG = 360.0 / N_RIBS
TREAD_DIP = RIB_RADIUS - math.sqrt(RIB_RADIUS ** 2 - (RIB_R * math.sin(math.radians(RIB_STEP_DEG / 2))) ** 2)
GRIP_WIDTH = 2 * (RIB_R * math.cos(math.radians(RIB_OFFSET_DEG)) + RIB_RADIUS)  # 63.8 mm across two pairs of ribs
DOT_RADIUS = 0.0035  # 7 mm across
DOT_R = 0.0275  # on the sidewall's mid-line
DOT_H = 0.0006  # paint, 0.6 mm proud of the sidewall
RED_LOCAL_DEG = 0.0  # the red dot's angle in the tyre's own frame
YELLOW_LOCAL_DEG = 100.0  # the yellow dot, 100 degrees counter-clockwise of the red one
TYRE_MASS = 0.040  # solid rubber (about 1.2 g/cm3) of the ring's 33 cm3
TYRE_TOP_H = TYRE_H + DOT_H  # the tyre's bounding box height
TYRES = {  # name: (start x, start y, start yaw in degrees: the red dot's angle about the tyre's centre)
    "tyre_a": (0.240, 0.150, 60.0),  # red dot at 60, yellow at 160
    "tyre_b": (0.240, -0.150, -40.0),  # red dot at -40, yellow at 60
}
TYRE_NAMES = tuple(TYRES)

# -- the rule ------------------------------------------------------------------------------------------------------
# On the marked wheel the red dot goes to the white mark; on the wheel without a mark the yellow dot goes to the valve.
# (target, which dot)
TARGETS = {
    MARKED_WHEEL: (MARK_ANGLE, "red"),
    PLAIN_WHEEL: (VALVE_ANGLE, "yellow"),
}

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
ANGLE_TOL_DEG = 10.0  # the relevant dot within 10 degrees of its mark (4.8 mm of arc at the dot's 27.5 mm radius)
SEAT_TOL = 0.003  # 'seated': the tyre's underside (its bottom face's centre) within 3 mm of the turntable's top ...
TILT_MAX_DEG = 5.0  # ... its axis within 5 degrees of vertical, dotted sidewall up ...
AXIS_TOL = 0.006  # ... its centre within 6 mm of the wheel's axis, seen from above (the 5 mm clearance sets the most)
HANDS_OFF = 0.010  # 'let go': both grippers' finger zones at least 10 mm from the tyre
STILL_WINDOW_STEPS = 10  # 'still': in each of the last two 10-step (0.5 s) windows ...
STILL_PATH = 0.0015  # ... the tyre's centre travelled at most 1.5 mm (path length, positions, never velocities) ...
STILL_YAW_DEG = 2.0  # ... and over the last 1 s its heading changed at most 2 degrees
RESET_JUMP = 0.08  # a tyre moving this far in one control step back onto its start: a new trial ...
START_TOL = 0.045  # ... 'onto its start': within this of it (covers a seeded reset's +-3 cm jitter)

# -- the grippers (the let-go rule; rigstate reports the TCP, the point between the pad tips and the opening) --------
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) ---------------------------------------------------------------------------------------------------
TYRE_BLACK = (0.07, 0.07, 0.07)
DOT_RED = (0.86, 0.10, 0.08)
DOT_YELLOW = (0.96, 0.82, 0.10)
WHEEL_SILVER = (0.58, 0.59, 0.61)
TURNTABLE_GREY = (0.28, 0.30, 0.33)
VALVE_BLACK = (0.04, 0.04, 0.04)
MARK_WHITE = (0.97, 0.97, 0.95)


def wheel_axis(name: str) -> tuple[float, float]:
    x, y, _ = WHEELS[name]
    return x, y


def on_top(r: float, angle_deg: float) -> tuple[float, float]:
    """(dx, dy) of a point r from an axis at angle_deg (counter-clockwise from +x)."""
    a = math.radians(angle_deg)
    return r * math.cos(a), r * math.sin(a)


def required_yaw(wheel: str) -> float:
    """The tyre heading (red dot's angle, degrees) that puts the rule's dot exactly on the rule's mark."""
    target, dot = TARGETS[wheel]
    return target - (RED_LOCAL_DEG if dot == "red" else YELLOW_LOCAL_DEG)


# Sanity: the tyre clears the wheel, the valve and the mark pass through its bore, the fingers miss the turntable.
assert TYRE_R_IN - WHEEL_RADIUS >= 0.005 - 1e-12
assert VALVE_R + VALVE_RADIUS < TYRE_R_IN and MARK_R + MARK_RADIUS < WHEEL_RADIUS
assert TURNTABLE_RADIUS < TYRE_R_OUT
assert TYRE_R_IN < DOT_R - DOT_RADIUS and DOT_R + DOT_RADIUS < TYRE_R_OUT
assert math.isclose(RIB_R + RIB_RADIUS, TYRE_R_OUT) and math.isclose(RIB_R - RIB_RADIUS, TYRE_R_IN)
assert RIB_R * 2 * math.sin(math.radians(RIB_STEP_DEG / 2)) < 2 * RIB_RADIUS  # neighbouring ribs overlap
assert WHEEL_TOP > TYRE_H + TT_TOP
