"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The stamp's frame has its origin at the middle of the underside of its base (where the
die prints), z up, its long side along its own y and its labelled front face toward its own -x. The sheet's frame has
its origin at the middle of its underside, z up, its long side along its own x.

Sources (README.md): Trodat Printy 4912 self-inking stamp (imprint 47 x 18 mm); A4 paper (297 x 210 mm). The stamp's
outer sizes, mass, travel and spring force, the card backing, the printed box and every friction are estimates
(task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top
CONTROL_HZ = 20.0

# -- the stamp (stamp frame) ------------------------------------------------------------------------------------------
IMPRINT = (0.047, 0.018)  # (along y, along x): the die's printing area, 47 x 18 mm
BASE = (0.024, 0.060, 0.026)  # the lower housing (x, y, z) that stands on the paper; the die prints through its sole
HANDLE = (0.030, 0.066)  # the handle's outer footprint (x, y): a cap and four skirt walls sliding down over the base
CAP_T = 0.012  # the cap's thickness
SKIRT_WALL = 0.002
HANDLE_REST_Z = 0.024  # the skirt's lower edge above the base's sole, at rest
HANDLE_TOP = 0.058  # the stamp's overall height at rest
TRAVEL = 0.022  # how far the handle goes down to print
SPRING = 12.0 / TRAVEL  # N/m: about 12 N at the bottom of the travel (est.)
SPRING_DAMPING = 2.0
M_BASE = 0.045
M_HANDLE = 0.020
STAMP_FRICTION = (0.6, 0.5)  # the base's rubber-edged sole on paper (est.)
BLACK = (0.08, 0.08, 0.09)
LABEL = (0.92, 0.92, 0.90)
INK = (0.80, 0.07, 0.07)
STAMP_SPOT = (0.200, -0.180)  # world x, y: standing upright, front label toward the robot (the stamp's yaw 0)

# -- the sheet (sheet frame) -------------------------------------------------------------------------------------------
SHEET = (0.297, 0.210, 0.003)  # an A4 delivery note on a 3 mm card backing (it lies flat), long side along x
M_SHEET = 0.060
SHEET_FRICTION = (0.45, 0.40)  # card on the laminate table top (est.)
PAPER = (0.96, 0.96, 0.94)
PRINT_GREY = (0.45, 0.45, 0.47)
SHEET_CENTRE = (0.320, 0.000)  # world x, y
BOX = (0.030, 0.060)  # the printed 'GOODS RECEIVED' box's inside (along x, along y): 30 x 60 mm
BOX_CENTRE = (-0.070, -0.060)  # in the sheet frame: toward the robot and to its right (the sheet's lower right)
BOX_LINE = 0.0012
FILM = 0.0002  # printed lines, 0.2 mm proud of the sheet

# -- the imprint (shown once the stamp has printed and lifted away) ---------------------------------------------------
IMPRINT_PARK = (-1.50, 0.00)  # world x, y on the floor behind the robot, out of every camera's view, until placed
M_IMPRINT = 0.002

# -- scoring ------------------------------------------------------------------------------------------------------------
BOTTOMED = TRAVEL - 0.002  # the handle within 2 mm of the bottom of its travel
FLAT_DEG = 3.0  # the base within 3 deg of the sheet's plane while it prints
ON_SHEET = 0.0015  # m: the base's sole within 1.5 mm of the sheet's top (resting on it)
PRINT_STEPS = 4  # held bottomed and flat for 0.2 s
REARM = 0.010  # the handle back above 10 mm of travel before another press can print
SQUARE_OK = 5.0  # deg, imprint to the box
SQUARE_HALF = 15.0
SHEET_SLIP = 0.005  # m: the sheet moved during the press
HOME_OK = 0.030  # m: the stamp back within 30 mm of its spot
UPRIGHT_DEG = 5.0
DROP_DEG = 45.0  # the stamp tipped past 45 deg at any moment: dropped (latched 0)
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
EPS = 1e-9


def stamp_start() -> tuple[float, float, float]:
    return (STAMP_SPOT[0], STAMP_SPOT[1], TOP)


def sheet_start() -> tuple[float, float, float]:
    return (SHEET_CENTRE[0], SHEET_CENTRE[1], TOP)


def box_world() -> tuple[float, float]:
    return (SHEET_CENTRE[0] + BOX_CENTRE[0], SHEET_CENTRE[1] + BOX_CENTRE[1])


def footprint(n_y: int = 9, n_x: int = 5) -> np.ndarray:
    """(n_y * n_x, 2) sample points over the imprint, in the stamp frame (x, y)."""
    ys = np.linspace(-IMPRINT[0] / 2, IMPRINT[0] / 2, n_y)
    xs = np.linspace(-IMPRINT[1] / 2, IMPRINT[1] / 2, n_x)
    return np.array([(x, y) for y in ys for x in xs])


def corners() -> np.ndarray:
    a, b = IMPRINT[1] / 2, IMPRINT[0] / 2
    return np.array([(-a, -b), (-a, b), (a, b), (a, -b)])
