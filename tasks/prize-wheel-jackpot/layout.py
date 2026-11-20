"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The wheel stands upright facing the robot on a horizontal axle along x.

Angles on the wheel are measured clockwise from 12 o'clock as the robot sees it (looking along +x): 3 o'clock is -y.
The wheel's joint turns the same way (a positive turn about +x is clockwise to the robot).

Sources (README.md): tabletop prize wheels (8 in, 12 segments, a peg standing out of the rim at each segment
boundary, a flexible pointer at the top, on a clamp-on stand). The bearing's damping is an estimate (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0

CENTRE = np.array([0.360, 0.000, 0.130])  # the wheel's centre (x, y, height above the table)
R = 0.100  # the wheel's radius (20 cm across)
THICK = 0.012
FACE_X = CENTRE[0] - THICK / 2  # its front face
SEGMENTS = 12
SEG = 2 * math.pi / SEGMENTS
JACKPOT = 9  # segment k is centred at k * 30 deg: the gold JACKPOT starts at 9 o'clock, a quarter turn
# counterclockwise of the pointer
PEG_L = 0.020  # pegs on the segment boundaries (k * 30 + 15 deg), standing out of the rim's edge this far
PEG_R = R + PEG_L / 2  # their centres' radius
PEG_D = 0.010  # 10 mm square
M_WHEEL = 0.30
DAMPING = 0.00117  # N m s/rad: the bearing's drag (est.: a 20 cm wheel spun briskly coasts a turn or two)
COLOURS = [(0.90, 0.15, 0.15), (0.97, 0.97, 0.95), (0.15, 0.40, 0.90), (0.97, 0.97, 0.95)]
GOLD = (0.95, 0.75, 0.10)
STAND_COLOUR = (0.12, 0.12, 0.14)
POINTER_COLOUR = (0.85, 0.10, 0.10)

# -- scoring -----------------------------------------------------------------------------------------------------------
FULL_TURN = 2 * math.pi  # the free coast since the last touch must be at least this
REST_SPEED = 0.02  # rad/s: at rest below this ...
REST_STEPS = 20  # ... for 1 s
TOUCH_IN_FRONT = 0.040  # a gripper point within this in front of the face ...
TOUCH_OUT = 0.035  # ... and within the wheel's radius plus this touches the wheel or its pegs
EPS = 1e-9


def point(a: float, r: float, x: float = FACE_X) -> np.ndarray:
    """The point at angle ``a`` (clockwise from 12 o'clock, as the robot sees it) and radius ``r`` on the plane
    ``x`` (world)."""
    return np.array([x, CENTRE[1] - r * math.sin(a), TOP + CENTRE[2] + r * math.cos(a)])


def tangent(a: float) -> np.ndarray:
    """The direction a point at angle ``a`` moves when the wheel turns clockwise."""
    return np.array([0.0, -math.cos(a), -math.sin(a)])


def segment_under_pointer(turn: float) -> int:
    """The segment at the pointer (12 o'clock) with the wheel turned ``turn`` rad clockwise from its start."""
    return int(round(((-turn) % (2 * math.pi)) / SEG)) % SEGMENTS


def peg_angles() -> list[float]:
    return [k * SEG + SEG / 2 for k in range(SEGMENTS)]
