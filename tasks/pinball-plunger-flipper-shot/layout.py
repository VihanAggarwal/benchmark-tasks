"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The pinball machine's frame has its origin on the table under the middle of its front
panel; its playfield rises 6 deg away from the robot (+x). Playfield coordinates (u, v): u up the slope from the
playfield's front edge, v across it (+v to the robot's left), both in metres.

Sources (README.md): EastPoint Sports tabletop arcade pinball (about 19 x 13 x 6 in, plunger launch, two flipper
buttons). Its playfield layout, flipper size and travel, plunger travel and spring, the ball and every friction are
estimates (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
INCH = 0.0254
TOP = 0.7475
CONTROL_HZ = 20.0

# -- the cabinet (fixed: held to the table by its feet and two clamps; world) -------------------------------------------
CAB_L = 19.0 * INCH  # 483 mm along x
CAB_W = 13.0 * INCH  # 330 mm along y
CAB_FRONT_X = 0.200  # its front panel, 200 mm in front of the arm bases' line
CAB_CY = 0.000
BASE_H = 0.035  # the plinth the playfield sits on, at the front
FRONT_H = 0.075  # the front panel's top, above the table
WALL_T = 0.012
SLOPE = math.radians(6.0)
PF_U0 = 0.050  # the playfield starts 50 mm behind the front panel (behind the front deck the buttons sit on)
DECK_D = 0.050  # the front deck: 50 mm deep, its top 100 mm above the table
DECK_H = 0.100
SIDE_H = 0.125  # the side and back panels' top, above the table
PF_LEN = 0.420  # up the slope
PF_W = CAB_W - 2 * WALL_T  # 306 mm across
WALL_H = 0.035  # the playfield's walls stand 35 mm above it
CAB_COLOUR = (0.12, 0.16, 0.42)
FIELD = (0.93, 0.83, 0.38)
WALL_COLOUR = (0.85, 0.12, 0.12)
RUBBER = (0.08, 0.08, 0.08)
FLIPPER_COLOUR = (0.96, 0.96, 0.96)


def pf_point(u: float, v: float, h: float = 0.0) -> np.ndarray:
    """A point (cabinet frame) at playfield coordinates (u, v), ``h`` above the playfield along its normal."""
    x = PF_U0 + u * math.cos(SLOPE) - h * math.sin(SLOPE)
    z = BASE_H + u * math.sin(SLOPE) + h * math.cos(SLOPE)
    return np.array([x, v, z])


def world(p_cab) -> np.ndarray:
    p = np.asarray(p_cab, dtype=float)
    return np.array([CAB_FRONT_X + p[0], CAB_CY + p[1], TOP + p[2]])


def to_pf(p_world) -> tuple[float, float, float]:
    """World point -> playfield (u, v, h)."""
    p = np.asarray(p_world, dtype=float) - np.array([CAB_FRONT_X + PF_U0, CAB_CY, TOP + BASE_H])
    u = p[0] * math.cos(SLOPE) + p[2] * math.sin(SLOPE)
    h = -p[0] * math.sin(SLOPE) + p[2] * math.cos(SLOPE)
    return float(u), float(p[1]), float(h)


# -- plunger lane and plunger (right side) ----------------------------------------------------------------------------
LANE_W = 0.026  # the plunger lane, 26 mm wide, along the right wall
LANE_V = -PF_W / 2 + LANE_W / 2  # its centre line
LANE_WALL_U = (0.000, 0.330)  # the lane's inner wall runs up the slope this far
PLUNGER_REST_U = 0.035  # the plunger rod's tip, at rest
PLUNGER_TRAVEL = 0.045  # pulled back up to 45 mm
PLUNGER_SPRING = 300.0  # N/m (est.)
PLUNGER_KNOB_R = 0.013
PLUNGER_KNOB_L = 0.022
PLUNGER_SHAFT_OUT = 0.035  # the shaft stands 35 mm out of the front panel before the knob

# -- flippers and their buttons ------------------------------------------------------------------------------------------
FLIPPER_U = 0.075  # the flippers' pivots, up the slope from the playfield's front
FLIPPER_V = 0.068  # either side of the centre line (the right one inside the plunger lane's wall)
FLIPPER_L = 0.060
FLIPPER_W = 0.012
FLIPPER_H = 0.018  # tall, standing 3 mm off the playfield
FLIPPER_REST = math.radians(-28.0)  # each tip angled down toward the centre (drain gap between them)
FLIPPER_UP = math.radians(28.0)  # raised
FLIPPER_SPRING = 1.0  # N m/rad: the flipper's coil drives it toward its target (a rule stands in for the switch)
BUTTON_TRAVEL = 0.006  # the buttons on the cabinet's top front corners press down 6 mm
BUTTON_ON = 0.0035  # past 3.5 mm the switch closes and the flipper flies up
BUTTON_SPRING = 400.0  # N/m (est.)
BUTTON_R = 0.011
BUTTON_X = 0.025  # on the front deck's top, from the front panel
BUTTON_Y = CAB_W / 2 - 0.022

# -- the ball --------------------------------------------------------------------------------------------------------------
BALL_R = 0.0095  # a 19 mm steel ball
M_BALL = 0.028
BALL_COLOUR = (0.80, 0.81, 0.83)

# -- scoring ---------------------------------------------------------------------------------------------------------------
TOP_U = 0.330  # the top lanes: the ball reaching this far up the slope after a flipper shot
FLIP_SPEED = 0.30  # m/s up the slope: a flipper shot
FLIP_MEMORY = 4  # steps: a shot counts if the ball was in the flippers' reach up to 0.2 s before
FIRE_MEMORY = 6  # steps: and only within 0.3 s after that flipper fired (rose past half way): no passive rebounds
DRAIN_U = 0.030  # the ball below this: drained, the ball lost
LAUNCH_V = LANE_V + LANE_W / 2 + BALL_R  # the ball out of the plunger lane, onto the playfield
EPS = 1e-9


def ball_start() -> np.ndarray:
    """The ball resting in the plunger lane against the plunger's tip (world)."""
    return world(pf_point(PLUNGER_REST_U + BALL_R, LANE_V, BALL_R))
