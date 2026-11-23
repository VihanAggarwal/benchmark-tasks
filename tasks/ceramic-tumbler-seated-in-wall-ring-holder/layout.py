"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights named h_* or *_H are above the table top.

A ceramic tumbler (100 mm tall, 60 mm across at its base widening evenly to 76 mm at the rim, 5 mm wall, 0.28 kg)
stands on the vanity (the table). A wall ring holder, a chrome ring (68 mm inside, 8 mm wide band, 3 mm thick) on a
88 mm strap from a flange screwed to a plywood backboard, hangs 90 mm over the vanity. Lowered mouth up through the
ring, the tumbler stops where its taper is as wide as the ring (50 mm up from its base): it hangs by that wedge with
its base 40 mm over the vanity. The ring is mounted low so that both the placing arm's reach and its wrist limits
cover the hang (the arm's straight-down reach ends near 0.20 m over the table at this distance).
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the ring holder (fixed, screwed down) ---------------------------------------------------------------------------
RING_X, RING_Y = 0.24, -0.14  # the ring's centre (the holder's base frame origin, on the table top)
RING_ID = 0.068  # 68 mm inside diameter
RING_R_IN = RING_ID / 2  # 34 mm
RING_BAND = 0.008  # 8 mm wide band (radial)
RING_R_OUT = RING_R_IN + RING_BAND  # 42 mm
RING_T = 0.003  # 3 mm thick
RING_H = 0.090  # the ring's top face 90 mm over the table (see the module note on reach)
RING_SIDES = 32  # the ring (and the tumbler's wall) are 32-sided: a 0.5 % flat-to-corner difference
# The arm: 88 mm from the ring's outer edge to the flange. The placing arm's wrist (link 4) reaches about 0.10 m past
# its TCP toward the wall, so the backboard must stand well back (32 mm of clearance at the lowest point of the demo).
STRAP_LEN = 0.088
STRAP_W = 0.016
FLANGE_X = RING_X + RING_R_OUT + STRAP_LEN  # 0.370 the flange's front face
FLANGE_T, FLANGE_W, FLANGE_H = 0.003, 0.036, 0.060  # the plate the strap is screwed through
BOARD_X0 = FLANGE_X + FLANGE_T  # the plywood backboard's front face (the flange stands on it)
BOARD_T, BOARD_W, BOARD_H = 0.018, 0.400, 0.300  # 18 mm plywood, 400 mm wide, 300 mm high, screwed to the table
BOARD_Y = RING_Y  # centred on the ring
CHROME = (0.78, 0.79, 0.82)
BOARD_WHITE = (0.90, 0.90, 0.88)

# -- the tumbler (loose) ---------------------------------------------------------------------------------------------
TUM_START = (0.22, -0.30)  # stands on the table in front of the placing arm's base (y -0.305)
TUM_H = 0.100  # tall
TUM_D_BASE, TUM_D_RIM = 0.060, 0.076  # outside diameters at the base and the rim
TUM_R_BASE = TUM_D_BASE / 2  # 30 mm
TUM_TAN = (TUM_D_RIM - TUM_D_BASE) / 2 / TUM_H  # 0.08: the wall leans out 8 mm per 100 mm (4.57 degrees)
TUM_ALPHA = math.atan(TUM_TAN)
TUM_WALL = 0.005
TUM_FLOOR = 0.005
TUM_MASS = 0.28  # measured on one set
TUM_FRICTION = (0.40, 0.35)  # glazed ceramic (est)
RING_FRICTION = (0.30, 0.25)  # chrome-plated zinc (est)
TUM_WHITE = (0.94, 0.94, 0.91)
H_SEAT_ON_TUM = (RING_R_IN - TUM_R_BASE) / TUM_TAN  # 0.050: the height on the tumbler where its width is the ring's
H_SEAT = RING_H - H_SEAT_ON_TUM  # 0.040: the base's height over the table when hung
H_RIM_SEAT = H_SEAT + TUM_H  # 0.140

# -- thresholds (scoring.py) -----------------------------------------------------------------------------------------
HUNG_TOL = 0.008  # hung: the base within 8 mm of H_SEAT (32 to 48 mm over the table)
TILT_FULL_DEG = 6.0  # 1.0: axis within 6 degrees of vertical
TILT_HALF_DEG = 15.0  # 0.5: within 15 degrees
OFF_FULL = 0.006  # 1.0: the axis within 6 mm of the ring's centre (at the ring's height)
OFF_HALF = 0.012  # 0.5: within 12 mm
CAUGHT_BASE_MIN = 0.020  # 0.25: base at least 20 mm over the table (not standing on it) ...
CAUGHT_NEAR = 0.060  # ... with the base within 60 mm of the ring's axis (top view) ...
CAUGHT_TILT_DEG = 60.0  # ... and the mouth up: the axis within 60 degrees of vertical
FALL_DROP = 0.020  # latched (an impact): the tumbler's base came down 20 mm or more ...
FALL_STEPS = 3  # ... within 3 control steps (0.15 s; a free fall of 20 mm takes 0.064 s) ...
IMPACT_STEP = 0.0065  # ... still coming down 6.5 mm or more in the last of those steps (130 mm/s) ...
STOP_MOVE = 0.002  # ... then stopped dead: under 2 mm further down in the next step ...
SUPPORT_TOL = 0.005  # ... on a support: its lowest point within 5 mm of the table (or below it: off the table), or
ON_RING_NEAR = RING_R_OUT + 0.029  # on or in the ring: base within 71 mm of the ring's axis (the base disc still
# overlaps the band) with its lowest point no higher than 5 mm over the ring's top face. A fast hand that stops in
# free air is not an impact.
HOLDER_MOVE = 0.005  # latched: the holder moved more than 5 mm
LET_GO = 0.030  # hands off: every finger zone at least 30 mm from the tumbler and the holder
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001  # still: the tumbler moves under 1 mm ...
STILL_TURN_DEG = 0.5  # ... and turns under 0.5 degrees over the last 1 s
CONTROL_HZ = 20.0
RESET_JUMP = 0.05  # the tumbler back on its start mark having jumped this far in one step: a new episode began
START_TOL = 0.01
EPS = 1e-9


def seat_floor(offset: float) -> float | None:
    """The lowest height (m, base over the table) the tumbler can reach through the ring with its axis ``offset`` m
    from the ring's axis, a vertical wall of the cone against the ring's inner edge; None when the 30 mm base cannot
    enter the 34 mm hole (offset over 4 mm: it rests on the ring)."""
    room = RING_R_IN - TUM_R_BASE - offset
    if room < 0.0:
        return None
    return RING_H - room / TUM_TAN
