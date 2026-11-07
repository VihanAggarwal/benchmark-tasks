"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The board's four quarters are named from the arms' side of the table: "near" is the half nearer the arm bases
(x < 0.300), "far" the other half; "left" is y > 0, "right" y < 0.
"""

import math

TOP = 0.7475  # station table top

# The support plate: a fixed grey steel plate between the two conveyor rails, under where the board will stop.
PLATE_X, PLATE_Y = 0.300, 0.000
PLATE_SIZE = (0.124, 0.220, 0.005)
PLATE_X0, PLATE_X1 = PLATE_X - PLATE_SIZE[0] / 2, PLATE_X + PLATE_SIZE[0] / 2  # 0.238 .. 0.362
PLATE_Y0, PLATE_Y1 = PLATE_Y - PLATE_SIZE[1] / 2, PLATE_Y + PLATE_SIZE[1] / 2  # -0.110 .. 0.110
PLATE_TOP = TOP + PLATE_SIZE[2]

# The conveyor rails: fixed blue aluminium bars along y, standing on the table just outside the plate's two long
# edges. The board's two long edges ride on them (here they are shown low: 20 mm).
RAIL_W, RAIL_LEN, RAIL_H = 0.006, 0.240, 0.020
RAILS = (  # (name, x of the centre)
    ("rail_near", PLATE_X0 - 0.001 - RAIL_W / 2),  # x 0.231 .. 0.237
    ("rail_far", PLATE_X1 + 0.001 + RAIL_W / 2),  # x 0.363 .. 0.369
)
RAIL_TOP = TOP + RAIL_H


def rail_rect(x: float) -> tuple[float, float, float, float]:
    """A rail's outline seen from above: (x0, x1, y0, y1)."""
    return (x - RAIL_W / 2, x + RAIL_W / 2, PLATE_Y - RAIL_LEN / 2, PLATE_Y + RAIL_LEN / 2)


# The board outline: the edges of the board that will stop over the plate, 120 mm along x by 160 mm along y,
# marked by white lines 3 mm wide centred on the edges; two more white lines split it into quarters.
BOARD_X0, BOARD_X1 = 0.240, 0.360
BOARD_Y0, BOARD_Y1 = -0.080, 0.080
MID_X, MID_Y = 0.300, 0.000  # the quarter lines
LINE_W = 0.003
MARK_H = 0.001  # white lines and red areas are 1 mm thick on the plate's top
MARK_TOP = PLATE_TOP + MARK_H

QUARTERS = {  # name: (x0, x1, y0, y1) of the quarter, edge to edge
    "far_left": (MID_X, BOARD_X1, MID_Y, BOARD_Y1),
    "near_left": (BOARD_X0, MID_X, MID_Y, BOARD_Y1),
    "near_right": (BOARD_X0, MID_X, BOARD_Y0, MID_Y),
    "far_right": (MID_X, BOARD_X1, BOARD_Y0, MID_Y),
}
QUARTER_NAMES = tuple(QUARTERS)


def quarter_of(x: float, y: float) -> str | None:
    """The quarter holding the point (x, y) seen from above (a point on a quarter line goes to the far / left
    side), or None outside the outline."""
    if not (BOARD_X0 <= x <= BOARD_X1 and BOARD_Y0 <= y <= BOARD_Y1):
        return None
    return ("far" if x >= MID_X else "near") + ("_left" if y >= MID_Y else "_right")


def lines() -> list[tuple[str, float, float, float, float]]:
    """The white lines as (name, x0, x1, y0, y1) strips that touch but do not overlap."""
    h = LINE_W / 2
    return [
        ("outline_near", BOARD_X0 - h, BOARD_X0 + h, BOARD_Y0 - h, BOARD_Y1 + h),
        ("outline_far", BOARD_X1 - h, BOARD_X1 + h, BOARD_Y0 - h, BOARD_Y1 + h),
        ("outline_right", BOARD_X0 + h, BOARD_X1 - h, BOARD_Y0 - h, BOARD_Y0 + h),
        ("outline_left", BOARD_X0 + h, BOARD_X1 - h, BOARD_Y1 - h, BOARD_Y1 + h),
        ("quarter_line_x", MID_X - h, MID_X + h, BOARD_Y0 + h, BOARD_Y1 - h),
        ("quarter_line_y_near", BOARD_X0 + h, MID_X - h, MID_Y - h, MID_Y + h),
        ("quarter_line_y_far", MID_X + h, BOARD_X1 - h, MID_Y - h, MID_Y + h),
    ]


# Red areas: where the board has parts on its underside (a pin there would crush them), as (name, x0, x1, y0, y1).
# Each stays at least 2 mm clear of the white lines. In each quarter they leave one clear window big enough for a
# pin with 3 mm to spare, in a different place each time: far left at the board's centre corner, near left at the
# board's near-left corner, near right in the middle of the quarter, far right in the middle of the board's right
# edge. Everywhere else the gaps between red areas and lines are at most 7 mm wide.
RED = (
    # far left (x 0.300-0.360, y 0-0.080): a processor and a connector; clear by the board's centre
    ("red_fl_processor", 0.3035, 0.3565, 0.0390, 0.0765),
    ("red_fl_connector", 0.3370, 0.3565, 0.0035, 0.0350),
    # near left (x 0.240-0.300, y 0-0.080): a BGA and a row of capacitors; clear in the near-left corner
    ("red_nl_bga", 0.2435, 0.2965, 0.0035, 0.0360),
    ("red_nl_capacitors", 0.2840, 0.2965, 0.0400, 0.0765),
    # near right (x 0.240-0.300, y -0.080-0): two IC rows and two resistor columns round a clear middle
    ("red_nr_ics_inner", 0.2435, 0.2965, -0.0190, -0.0035),
    ("red_nr_ics_outer", 0.2435, 0.2965, -0.0765, -0.0610),
    ("red_nr_resistors_near", 0.2435, 0.2500, -0.0570, -0.0230),
    ("red_nr_resistors_far", 0.2900, 0.2965, -0.0570, -0.0230),
    # far right (x 0.300-0.360, y -0.080-0): a connector and two resistor columns; clear at the right edge's middle
    ("red_fr_connector", 0.3035, 0.3565, -0.0360, -0.0035),
    ("red_fr_resistors_near", 0.3035, 0.3100, -0.0765, -0.0400),
    ("red_fr_resistors_far", 0.3500, 0.3565, -0.0765, -0.0400),
)
RED_NAMES = tuple(r[0] for r in RED)

# Support pins: loose steel posts (a magnetic-base support pin simplified to a plain solid post), 20 mm across and
# 45 mm tall, 110 g (a solid steel post this size is 111 g). Three stand in a black tray on each side, in a row
# along x; each arm can reach only its own side's tray.
PIN_R, PIN_H = 0.010, 0.045
PIN_MASS = 0.110
TRAY_SIZE = (0.140, 0.040, 0.003)
TRAY_X, TRAY_Y = 0.190, 0.230  # tray centres at (0.190, +-0.230)
TRAY_TOP = TOP + TRAY_SIZE[2]
PIN_XS = (0.230, 0.190, 0.150)  # in each tray, nearest the board first
PINS = tuple(
    [(f"pin_left_{i + 1}", x, TRAY_Y) for i, x in enumerate(PIN_XS)]
    + [(f"pin_right_{i + 1}", x, -TRAY_Y) for i, x in enumerate(PIN_XS)]
)
PIN_NAMES = tuple(p[0] for p in PINS)
STARTS = {name: (x, y) for name, x, y in PINS}

# Rule thresholds (scoring.py).
EDGE_IN = 0.005  # a supporting pin's whole outline at least this far inside the board's edge (the outline's centre line)
RED_CLEAR = 0.003  # ... and at least this far from every red area
TILT_MAX_DEG = 10.0  # ... standing upright: its axis within this of the vertical
BOTTOM_TOL = 0.003  # ... and its lowest point within this of the plate's top
TOUCH_LOW = 0.005  # 'touching a red area': outline over it with the pin's lowest point within this of the plate
RAIL_NEAR = 0.005  # 'at a rail': outline within this of a rail's outline ...
RAIL_ABOVE = 0.005  # ... with the pin's lowest point at most this far above the rail's top
HANDS_OFF = 0.0  # a pin a gripper's finger zone reaches is still held: it does not count
STILL_WINDOW_S = 0.5  # 'still': moved no more than STILL_PATH (path length) over the last 0.5 s ...
STILL_PATH = 0.002
STILL_SPEED = 0.05  # ... and slower than this now (m/s); grippers' reported velocities chatter, so the path rules
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a pin moving this much further in one step than its velocity allows was teleported ...
START_TOL = 0.045  # ... and every pin within this of its own starting mark: a new episode (+-3 cm seeded jitter)
HULL_SIDES = 48  # a pin's outline seen from above is the hull of its two end circles, 48 points each

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Colours (sRGB).
PLATE_GREY = (0.36, 0.37, 0.39)
RAIL_BLUE = (0.20, 0.35, 0.66)
LINE_WHITE = (0.95, 0.95, 0.95)
RED_AREA = (0.80, 0.12, 0.12)
TRAY_BLACK = (0.07, 0.07, 0.08)
PIN_STEEL = (0.78, 0.79, 0.81)

assert math.isclose(PLATE_X0, 0.238) and math.isclose(PLATE_X1, 0.362)
