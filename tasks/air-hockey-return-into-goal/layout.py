"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The game's playfield runs along x: the robot's goal at the near end, the far goal at the
far end.

Sources (README.md): battery tabletop air hockey games (a playfield about 60 x 32 cm with rails, a goal slot at each
end, 50 mm pucks and 60 mm mallets). The air cushion is drawn as a nearly frictionless surface; the practice kicker
that sends the puck is part of the setup (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the game (fixed) ----------------------------------------------------------------------------------------------------
X0, X1 = 0.220, 0.820  # the playfield's near and far ends (x)
HALF_W = 0.160  # y from -HALF_W to +HALF_W
SURFACE = 0.060  # the playfield's top, above the table
RAIL_H, RAIL_T = 0.020, 0.015
GOAL_HALF = 0.045  # each goal slot is 90 mm wide, centred on y 0
CENTRE_X = (X0 + X1) / 2  # the halfway line
SURFACE_FRICTION = (0.02, 0.02)  # the air cushion
BODY = (0.10, 0.25, 0.55)
FIELD = (0.92, 0.94, 0.97)
RAIL = (0.85, 0.15, 0.15)
GOAL_BOX = (0.95, 0.95, 0.95)  # white, so the black puck shows in it

# -- the puck and the mallet (loose) -----------------------------------------------------------------------------------
PUCK_R, PUCK_H, M_PUCK = 0.025, 0.006, 0.010
MALLET_R, MALLET_H, M_MALLET = 0.030, 0.012, 0.050
HANDLE_R, HANDLE_H = 0.012, 0.035
RESTITUTION = 0.75  # puck off the mallet's rim (est.)
PUCK_COLOUR = (0.10, 0.10, 0.10)
MALLET_COLOUR = (0.95, 0.80, 0.10)
PUCK_START = np.array([0.740, 0.080])
MALLET_START = np.array([0.260, 0.000])  # in front of the robot's goal, 25 mm clear of the puck's line

# -- the practice kicker (a fixed mechanism at the far end, off centre) --------------------------------------------------
KICKER_Y = PUCK_START[1]
KICKER_X = PUCK_START[0] + PUCK_R + 0.015  # its face, 15 mm behind the puck
KICK_AT = 7.0  # s
KICK_SPEED = 0.55  # m/s: its face's speed through the stroke
KICK_TRAVEL = 0.040

# -- scoring -----------------------------------------------------------------------------------------------------------
HIT_GAP = 0.030  # the mallet within this of the puck (rim to rim, at the nearer of the two steps) when the puck
# turns round: a hit (at 20 Hz the two are a few cm apart either side of the contact)
HIT_SPEED = 0.05  # m/s: the puck has to be moving away after the hit, not just stopped
STRIKE_SPEED = 0.20  # m/s: the mallet's own speed over the 0.1 s up to the hit (a strike, not a block)
HELD_R = 0.020  # held: a gripper's TCP within this of the mallet's axis (horizontally),
HELD_DZ = (0.0, 0.060)  # this far above the mallet disc's centre (the handle's height, m),
HELD_OPENING = 0.5  # and its fingers closed past half open (onto the 24 mm handle: about 0.26)
RAIL_CLEAR = 0.010  # a hit needs the puck's edge at least this far from the near end rail
HAND_CLEAR = 0.015  # no gripper point within this of the puck's rim (and within 30 mm of its height)
EPS = 1e-9


def at(x: float, y: float, h: float = 0.0) -> np.ndarray:
    """A point on the playfield (world), ``h`` above its surface."""
    return np.array([x, y, TOP + SURFACE + h])


def in_far_goal(p) -> bool:
    return bool(p[0] > X1 and abs(p[1]) < GOAL_HALF)


def in_own_goal(p) -> bool:
    return bool(p[0] < X0 and abs(p[1]) < GOAL_HALF)
