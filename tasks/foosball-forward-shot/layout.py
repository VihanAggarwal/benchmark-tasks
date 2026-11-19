"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The foosball table's long side faces the robot: its rods run along x (away from the
robot), its goals are at its left (+y) and right (-y) ends. The robot plays the team whose handles are on the near
side and attacks the right-hand goal.

Sources (README.md): Best Choice Products 40 in tabletop foosball table (40 x 22.5 x 6 in overall, 23.5 lb; two balls).
Inside sizes, rod positions and spacing, the players' shape, the goal mouth, the ball and every friction are estimates
(task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
INCH = 0.0254
TOP = 0.7475
CONTROL_HZ = 20.0

# -- the cabinet (fixed; world) -----------------------------------------------------------------------------------------
CAB_L = 40.0 * INCH  # 1016 mm along y
CAB_W = 22.5 * INCH  # 572 mm along x
CAB_H = 6.0 * INCH  # 152 mm tall
WALL = 0.015
FLOOR_Z = 0.030  # the playfield, 30 mm above the table
NEAR_X = 0.200  # the cabinet's near side, 200 mm in front of the arm bases' line
CAB_CY = 0.000
FIELD_X0, FIELD_X1 = NEAR_X + WALL, NEAR_X + CAB_W - WALL  # the playfield inside the side walls (along x)
FIELD_Y0, FIELD_Y1 = CAB_CY - CAB_L / 2 + WALL, CAB_CY + CAB_L / 2 - WALL  # inside the end walls (along y)
FIELD_CX = (FIELD_X0 + FIELD_X1) / 2
GOAL_W = 0.130  # the goal mouth, 130 mm across (along x), centred on the field
GOAL_H = 0.045  # and 45 mm tall above the playfield
GOAL_DEPTH = 0.060  # the catch box behind each end wall
WOOD = (0.55, 0.36, 0.20)
FIELD_GREEN = (0.13, 0.45, 0.18)
LINE_WHITE = (0.92, 0.92, 0.90)
GOAL_BLACK = (0.08, 0.08, 0.08)

# -- rods and players ---------------------------------------------------------------------------------------------------
ROD_Z = FLOOR_Z + 0.072  # the rods' axis, 72 mm above the playfield
ROD_R = 0.0064  # 1/2 in rods
ROD_PITCH = CAB_L / 8  # eight rods, evenly spaced along the table (est.)
HANDLE_R = 0.016  # rubber handles, 32 mm across
HANDLE_L = 0.100  # sticking 100 mm out of the cabinet's side
PLAYER_DROP = 0.066  # from the rod's axis to the bottom of a player's foot (6 mm above the playfield)
PLAYER = (0.022, 0.020)  # a player's body thickness (along y, the kicking direction) and width (along x)
FOOT = (0.026, 0.024, 0.016)  # the foot block (along y, along x, tall): the striking face
M_PLAYER = 0.012
# Rods from the left end (+y) to the right end (-y): (name, side, men, spacing along x). Side "near": the robot's team,
# handles toward the robot; "far": the opponent, handles on the far side.
RODS = (
    ("goalie", "near", 1, 0.0),
    ("defence", "near", 2, 0.200),
    ("opp_forwards", "far", 3, 0.170),
    ("midfield", "near", 5, 0.105),
    ("opp_midfield", "far", 5, 0.105),
    ("forwards", "near", 3, 0.170),
    ("opp_defence", "far", 2, 0.200),
    ("opp_goalie", "far", 1, 0.0),
)
SHOOTER = "forwards"
SLIDE = 0.060  # the forward rod slides +-60 mm between its bumpers (est.)
ROD_FRICTION = 0.004  # N m: the rod turns freely in its bushings (est.)
TABLE_FRICTION = (0.25, 0.20)  # plastic players, playfield and walls (est.)
SLIDE_FRICTION = 3.0  # N: the rod's bushings resist sliding (est.)
CHROME = (0.78, 0.79, 0.81)
RED = (0.72, 0.10, 0.10)  # the robot's team
BLUE = (0.12, 0.24, 0.72)  # the opponent
HANDLE_BLACK = (0.06, 0.06, 0.06)

# -- the ball -------------------------------------------------------------------------------------------------------------
BALL_R = 0.018  # a 36 mm plastic table-football ball (est.)
M_BALL = 0.017
BALL_COLOUR = (0.96, 0.96, 0.92)
BALL_GAP = 0.015  # m: the ball's surface starts 15 mm in front of (toward -y of) the centre forward's foot face
BALL_OFFSET_X = 0.038  # m: and 38 mm beyond the field's centre line (away from the robot), in the lane past the goalie

# -- scoring --------------------------------------------------------------------------------------------------------------
SPIN_LIMIT = math.radians(360.0)  # the forward rod turned more than a full turn one way at any moment: a spin
NEAR_MISS = 0.100  # m: an on-target miss reaches the right end wall within 100 mm of the goal mouth
PROGRESS = 0.150  # m: the ball moved this far toward the right goal
STRUCK_MOVE = 0.010  # m: the ball has been struck once it has moved 10 mm
EPS = 1e-9


def rod_y(index: int) -> float:
    return CAB_CY + CAB_L / 2 - ROD_PITCH * (index + 0.5)


def rod_index(name: str) -> int:
    return [r[0] for r in RODS].index(name)


def men_x(men: int, spacing: float) -> list[float]:
    """The players' x on a rod, about the field's centre line."""
    return [FIELD_CX + spacing * (k - (men - 1) / 2) for k in range(men)]


def ball_start() -> tuple[float, float, float]:
    y = rod_y(rod_index(SHOOTER)) - FOOT[0] / 2 - BALL_GAP - BALL_R
    return (FIELD_CX + BALL_OFFSET_X, y, TOP + FLOOR_Z + BALL_R)


def goal_line_y() -> float:
    return FIELD_Y0  # the right-hand goal's line, the inside face of the right end wall


def in_goal(p) -> bool:
    """The ball wholly past the right-hand goal line, inside the goal mouth."""
    return bool(p[1] < goal_line_y() - BALL_R and abs(p[0] - FIELD_CX) <= GOAL_W / 2
                and p[2] < TOP + FLOOR_Z + GOAL_H + 0.01)
