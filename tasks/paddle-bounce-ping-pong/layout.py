"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The paddle's frame: origin at the blade's centre, mid-thickness; x along the handle's line
away from the handle (the handle points to -x), z out of the red face. It starts turned PADDLE_YAW: the handle points to
the robot's right (-y).

Sources (README.md): a standard table tennis bat (blade about 158 x 150 mm, 6 mm thick with rubbers, handle about
100 mm, 160 to 190 g) and an ITTF 40+ ball (40 mm, 2.7 g). The ball's bounce off the rubber and every friction are
estimates (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
ARM_BASE = {"left": np.array([0.0, 0.305, TOP]), "right": np.array([0.0, -0.305, TOP])}

# -- the paddle ------------------------------------------------------------------------------------------------------
BLADE_R = 0.077  # the blade, drawn round (real about 158 x 150 mm)
BLADE_T = 0.006  # wood and both rubbers
HANDLE_L = 0.100
HANDLE_W = 0.026  # across the face
HANDLE_T = 0.022  # through the face
HANDLE_IN = 0.012  # the handle runs this far under the blade's edge
HANDLE_X = -(BLADE_R + HANDLE_L / 2 - HANDLE_IN)  # the handle's centre, paddle frame
M_PADDLE = 0.180
RESTITUTION = 0.85  # the ball off the rubber (est.)
PADDLE_FRICTION = (0.8, 0.7)
RED = (0.78, 0.10, 0.10)
WOOD = (0.76, 0.58, 0.36)

# -- where things start (world) --------------------------------------------------------------------------------------
PADDLE_YAW = math.pi / 2  # the paddle frame's x along world +y: the handle points to the robot's right
BLOCK = np.array([0.360, 0.000])  # the foam block's centre (x, y), under the blade
BLOCK_SIZE = (0.110, 0.110, 0.100)
FOAM = (0.20, 0.30, 0.55)
PADDLE_START = np.array([BLOCK[0], BLOCK[1], TOP + BLOCK_SIZE[2] + BLADE_T / 2 + 0.003])  # the blade's centre (spawned 3 mm up: it settles onto the block)
PADDLE_REST = PADDLE_START - np.array([0.0, 0.0, 0.003])  # where it lies, on the block

BALL_R = 0.020
M_BALL = 0.0027
BALL_COLOUR = (0.97, 0.97, 0.94)
STAND = np.array([0.300, 0.150])  # the ball stand's centre (x, y)
STAND_OUTER = 0.030  # a square cup, 30 mm across, 22 mm tall, 3 mm walls
STAND_H = 0.022
STAND_WALL = 0.003
STAND_COLOUR = (0.25, 0.25, 0.28)
BALL_START = np.array([STAND[0], STAND[1], TOP + STAND_H + math.sqrt(BALL_R ** 2 - ((STAND_OUTER - 2 * STAND_WALL) / 2) ** 2)])

# -- scoring -----------------------------------------------------------------------------------------------------------
MIN_RISE = 0.100  # a bounce: the ball's bottom rises at least this far above the red face between touches
MAX_RISE = 0.250  # and no further (after a higher flight a still paddle could add several passive bounces)
MAX_DROP = 0.150  # the ball let go no higher than this (its bottom above the face)
HAND_CLEAR = 0.030  # no gripper within this of the ball's surface during a bounce
HOLD_CLEAR = 0.010  # a gripper this close to the ball's surface has it: the drop is measured from the last such step
TOUCH_BAND = 0.050  # a touch: the ball's bottom within this of the face at one of the two steps around it (at 20 Hz a
# ball meeting the face at 2 m/s is at most 50 mm from it at the nearer step)
KICK = 0.5  # m/s: and the ball's own velocity along the face's normal jumps up by more than this (a contact's impulse;
# gravity alone takes 0.49 m/s off it per step)
LIFTED = BLOCK_SIZE[2] + BLADE_T / 2 + 0.015  # the paddle held up: its blade's centre this far above the table (15 mm off its block)
TIERS = ((10, 1.0), (6, 0.5), (4, 0.25))  # bounces in a row -> score (a still paddle under the highest allowed flight
# adds at most two more over 100 mm: passive rebounds alone never reach a tier)
EPS = 1e-9


def paddle_frame(pos, quat_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of the paddle frame from its base pose."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.asarray(pos, dtype=float), r


def in_paddle(p_world, origin, rot) -> np.ndarray:
    """A world point in the paddle frame."""
    return rot.T @ (np.asarray(p_world, dtype=float) - origin)
