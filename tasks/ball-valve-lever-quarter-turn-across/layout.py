"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top.

A utility alcove (a laundry tub's hot and cold supply stubs) stands on the table: a back panel (a 1/2 in plywood wall)
400 mm in front of the arm bases, two side panels, and through the back panel two 1/2 in copper supply stubs 70 mm
above the floor, 203 mm (8 in) apart. Each stub carries a 1/2 in brass full-port ball valve with a flat steel lever
and a vinyl grip. Seen from above, the lever starts pointing toward the robot (open, along the pipe) and shuts
clockwise a quarter turn to its stop, ending across the pipe toward +y. The left stub (y +101.5 mm) is the cold
supply, blue grip: the one to shut. The right stub (y -101.5 mm) is the hot supply, red grip: it stays open.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)

# -- the cabinet (fixed) ---------------------------------------------------------------------------------------------
BACK_X0 = 0.400  # the back panel's front face
BACK_T = round(0.5 * IN, 4)  # 12.7 mm: 1/2 in plywood
BACK_X1 = BACK_X0 + BACK_T
BACK_HALF_Y = 0.438  # the back panel spans y -438 .. +438 mm (flush with the side panels' outer faces)
BACK_H = 0.300
SIDE_T = 0.018  # 18 mm plywood side panels
SIDE_IN_Y = 0.420  # their inner faces at y +-420 mm: wide enough that neither arm's ready pose is inside
SIDE_X0 = 0.200  # their front edges
SIDE_H = 0.300
WOOD = (0.80, 0.66, 0.46)
CHROME = (0.78, 0.78, 0.80)

# -- the two stubs and valves (fixed bodies) ---------------------------------------------------------------------
COLD_Y = 0.1015  # the left stub: cold, blue grip (the one to shut). 8 in (203 mm) between the stubs (est)
HOT_Y = -0.1015  # the right stub: hot, red grip (left open)
PIPE_Z = 0.070  # stub axis above the floor
PIPE_R = round(0.625 * IN / 2, 5)  # 1/2 in copper tube, 5/8 in OD: 7.94 mm radius (ASTM B88 type L)
PIPE_X0 = 0.140  # the stub's open end
COPPER = (0.72, 0.45, 0.28)
BRASS = (0.78, 0.60, 0.20)
STEM_X = 0.270  # the stem's axis: 130 mm in front of the back panel
BODY_L = 0.052  # valve body, end to end
BODY_R = 0.0145  # 30 mm across the ball housing
HOUSING_L = 0.030
END_R = 0.0112
END_L = 0.011  # each end sleeve (52 = 30 + 2 x 11)
NECK_R = 0.0075
NECK_Z0 = PIPE_Z + BODY_R  # 84.5 mm: the housing's top
PLATE_Z = 0.0932  # the stop plate's centre (1.5 mm thick, its top at 93.95 mm, under the bar's 94.0)
PLATE_SIZE = 0.024
TAB = (0.004, 0.004, 0.008)  # the two stop tabs on the plate's edge (the limit is the hinge's)
CLIP_X = (0.345, 0.205)  # the two pipe clips on each stub: supply side and outlet side
CLIP_PLASTIC = (0.80, 0.80, 0.78)
CLIP_POST = (0.016, 0.030)  # the clip's standoff post (x, y), floor to the pipe's underside

# -- the lever (the hinge links) -----------------------------------------------------------------------------------
LEVER_Z = 0.097  # the bar's centre above the floor (stem 97 mm up)
BAR_L = 0.130  # from the stem's axis to the lever's tip (the tip end)
BAR_TAIL = 0.012  # the bar's tail behind the stem
BAR_W = 0.018
BAR_T = 0.006
GRIP_L = 0.060  # the vinyl grip over the outer 60 mm
GRIP_R = 0.0105  # 21 mm across (a round sleeve)
GRIP_C = BAR_L - GRIP_L / 2  # 100 mm from the stem: where a hand pushes
STEEL = (0.62, 0.64, 0.68)
BLUE = (0.10, 0.28, 0.85)
RED = (0.85, 0.12, 0.10)
LEVER_MASS = 0.05
VALVE_FRICTION_TORQUE = 0.20  # N m, run torque = breakaway (est: a cleaned, lubricated valve on a drained line)
PUSH_FORCE = VALVE_FRICTION_TORQUE / GRIP_C  # 2.0 N at the middle of the grip
LEVER_DAMPING = 0.01
Q_CLOSED = math.pi / 2  # the closing stop: a quarter turn, clockwise seen from above

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------
FULL_DEG = 88.0  # 1.0: the lever 88 to 90 degrees (on its stop, within the protractor card's reading)
HALF_DEG = 45.0  # 0.5: 45 up to 88
QUARTER_DEG = 15.0  # 0.25: 15 up to 45; under 15: 0
OVER_DEG = 95.0  # a lever past this (a failed stop) is not a valid trial: 0
HOT_DEG = 10.0  # latched: the hot lever ever turned more than 10 degrees
MOVED_MM = 5.0  # latched: the cabinet, a stub or a valve body moved more than 5 mm
STILL_WINDOW_S = 1.0
STILL_DEG = 2.0  # the cold lever's angle changes under 2 degrees over the last second (about 6 mm on the card)
LET_GO = 0.030  # both finger zones 30 mm or more from the cold lever and valve body, every step of the last second
RESET_DEG = 25.0  # a cold lever that unwinds this far in one step with both back at open: a reset
CONTROL_HZ = 20.0
EPS = 1e-9

FRICTION = (0.5, 0.4)  # wood and metal fixtures (est)


def valve_y(which: str) -> float:
    return {"cold": COLD_Y, "hot": HOT_Y}[which]


def lever_dir(q: float):
    """Unit vector in plan from the stem toward the lever's tip with the hinge at ``q`` (rad, clockwise from open):
    open (q 0) points to -x (toward the robot), closed (q = pi/2) to +y."""
    return (-math.cos(q), math.sin(q))


def push_dir(q: float):
    """Unit vector in plan of the grip's motion when the lever turns clockwise at ``q``."""
    return (math.sin(q), math.cos(q))
