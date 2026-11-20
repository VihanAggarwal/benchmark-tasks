"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The machine's frame: origin on the table under the middle of the cabinet's front edge, x
away from the robot, y to the robot's left, z up.

Sources (README.md): desktop "mini claw machine" toys (about 26 x 26 x 44 cm, a bat-top joystick and a drop button,
small boxed prizes about 50 mm). The machine's program (claw speed, the drop cycle, the timer) and the capsules' spots are
the ones stated in task.json; the claw's grip is a rule (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

ORIGIN = np.array([0.330, 0.000])  # the machine's frame origin (x, y), on the table

# -- the cabinet (fixed: clamped to the table) ------------------------------------------------------------------------
CAB_D, CAB_W = 0.260, 0.260  # depth (x) and width (y)
BASE_H = 0.100  # the pit's floor
LIP_H = 0.035  # a low wall round the pit (the clear walls, drawn low so the cameras see in)
WALL_T = 0.006
POST = 0.012  # corner posts
TOP_Z = 0.420  # the gantry housing's underside
HOUSING_H = 0.030
DECK_D, DECK_H = 0.110, 0.070  # the control deck in front of the cabinet
CHUTE = (0.020, 0.090, 0.040, 0.110)  # the prize chute's opening in the floor: x from, x to, y from, y to (front-left)
CAB_COLOUR = (0.85, 0.20, 0.45)
DECK_COLOUR = (0.15, 0.15, 0.18)
FLOOR_COLOUR = (0.95, 0.85, 0.30)
POST_COLOUR = (0.80, 0.82, 0.86)

# -- the controls (on the deck) ---------------------------------------------------------------------------------------
STICK = np.array([-0.070, -0.065])  # the joystick's pivot (machine frame, x y), on the deck's top
SHAFT_L = 0.045  # the shaft above the deck to the bat top's foot
BAT_R, BAT_L = 0.012, 0.040  # the bat-top handle
STICK_LIMIT = math.radians(22.0)
STICK_SWITCH = math.radians(8.0)  # tilted past this, the direction's switch closes
STICK_SPRING = 0.6  # N m/rad, centring
BUTTON = np.array([-0.070, 0.065])
BUTTON_R, BUTTON_TRAVEL, BUTTON_ON = 0.016, 0.005, 0.003
BUTTON_SPRING = 400.0

# -- the claw and its program (a rule) ---------------------------------------------------------------------------------
HOME = np.array([0.055, 0.075])  # the claw's start and return point (x, y), over the chute
CLAW_X = (0.030, 0.230)  # its travel
CLAW_Y = (-0.105, 0.105)
SPEED = 0.050  # m/s while a direction's switch is closed
HUB_Z = 0.360  # the claw hub's bottom at rest (machine frame z)
PRIZE = 0.050  # the prizes: 50 mm boxes
CAPSULE_R = PRIZE / 2  # half a prize's height
DROP_TO = BASE_H + PRIZE + 0.004  # the hub's bottom at the bottom of its drop, just above a prize's top
DROP_SPEED = 0.080
CLOSE_S = 0.6  # closing at the bottom
GRAB_R = 0.012  # the claw holds a prize whose centre is within this of its axis when it closes
NEAR_R = 0.025  # a near miss
GAME_S = 30.0  # the timer, from the joystick's first move
OPEN_WAIT_S = 1.0
CLAW_COLOUR = (0.80, 0.80, 0.82)
DARK = (0.08, 0.08, 0.08)
BAT_COLOUR = (0.90, 0.10, 0.10)
BUTTON_COLOUR = (0.10, 0.60, 0.95)

# -- the prizes (loose boxes) ---------------------------------------------------------------------------------------------
CAPSULES = [np.array([0.170, -0.050]), np.array([0.200, 0.040]), np.array([0.120, -0.010])]  # machine frame (x, y)
CAPSULE_COLOURS = [(0.90, 0.15, 0.15), (0.15, 0.45, 0.90), (0.95, 0.75, 0.10)]
M_CAPSULE = 0.030

# -- scoring -----------------------------------------------------------------------------------------------------------
MOVED = 0.050  # the claw moved at least this far from its start before dropping: a real attempt
EPS = 1e-9


def world(p_machine) -> np.ndarray:
    p = np.asarray(p_machine, dtype=float)
    return np.array([ORIGIN[0] + p[0], ORIGIN[1] + p[1], TOP + (p[2] if p.size > 2 else 0.0)])


def in_chute(p_machine) -> bool:
    x0, x1, y0, y1 = CHUTE
    return bool(x0 <= p_machine[0] <= x1 and y0 <= p_machine[1] <= y1)


def in_cabinet(p_world) -> bool:
    """A point inside the cabinet's volume above its floor (where a hand must not go)."""
    p = np.asarray(p_world, dtype=float) - np.array([ORIGIN[0], ORIGIN[1], TOP])
    return bool(0.0 < p[0] < CAB_D and abs(p[1]) < CAB_W / 2 and BASE_H < p[2] < TOP_Z)
