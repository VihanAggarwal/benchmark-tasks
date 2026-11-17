"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A pick cart's bed (a 6 mm steel deck with a tote locator frame round each of two bays and a rear frame with a push
handle) stands on the table. Two loose stacking totes (400 x 300 x 74 mm, 0.70 kg each with the picked goods in them)
stand in the two bays, side by side along y. The tote in the +y bay (the picked one) is carried across by its two end
hand holds and set on the tote in the -y bay: its foot nests inside the lower tote's walls and its ledge rests on the
lower walls' top.

Tote frame: origin at the centre of the foot's underside, x across the 300 mm width, y along the 400 mm length, z up.
The hand holds are in the two end walls (at y +-0.200). Figures are sourced (README) or estimates (est, said in
task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the tote (loose) ------------------------------------------------------------------------------------------------
# A stacking container on the Euro-container footprint, 400 x 300 mm, 74 mm tall, polypropylene, thin walls: the
# published size of a Euronorm stacking bin (Transoplast 9227.V00.522: 400 x 300 x 74 mm, PP, 0.48 kg, closed hand
# grips on the short sides, 10 kg rated load). The foot, ledge and grip sizes are not published: modelled (est).
TOTE_L = 0.400  # along y, outer
TOTE_W = 0.300  # along x, outer
TOTE_H = 0.074  # rim height above the foot's underside (published external height of the 400 x 300 x 74 mm bin)
WALL = 0.003
TOTE_EMPTY_MASS = 0.48  # kg, published weight of the 400 x 300 x 74 mm bin
GOODS_MASS = 0.22  # kg of picked goods (est)
TOTE_MASS = TOTE_EMPTY_MASS + GOODS_MASS  # 0.70 kg; a rigid compound in the model
TOTE_FRICTION = (0.30, 0.27)  # polypropylene on polypropylene (est); the pads' rubber sets the grip
TOTE_BLUE = (0.16, 0.36, 0.64)
CARTON_TAN = (0.80, 0.64, 0.40)

# The foot: what drops into the lower tote's opening. A 386 x 286 mm ring 12 mm deep, 4 mm inside the lower wall's
# inner face on every side, with a 45 degree lead-in 6 mm deep on its underside.
FOOT_L = 0.386
FOOT_W = 0.286
FOOT_H = 0.012
FOOT_CH = 0.006  # the lead-in's depth and width
FLOOR_L = FOOT_L - 2 * FOOT_CH  # 374 mm: the floor plate's edge at the lead-in's foot
FLOOR_W = FOOT_W - 2 * FOOT_CH  # 274 mm
FLOOR_T = 0.003
LEDGE_Z0 = 0.009  # the main walls' and ledge's underside: the plane the stack seats on
LEDGE_Z1 = 0.012
CLEAR = (TOTE_L - 2 * WALL - FOOT_L) / 2  # 4 mm each side between the foot and the lower wall's inner face
SEAT_DZ = TOTE_H - LEDGE_Z0  # 65 mm: the upper tote's origin above the lower's when seated
RIDE_DZ = TOTE_H  # 74 mm: its origin above the lower's when its foot's underside rides on the wall tops
CAPTURE = FOOT_CH + CLEAR  # 10 mm: the offset the lead-in still takes in

# The hand holds: in each end wall a 110 mm wide, 16 mm tall slot with a 10 mm thick, 12 mm tall grip bar above it.
HH_SLOT_W = 0.110
HH_SLOT_Z0, HH_SLOT_Z1 = 0.024, 0.040
BAR_Z0, BAR_Z1 = 0.040, 0.052
BAR_T = 0.010  # thickness (inward from the wall's outer face)
BAR_Y = TOTE_L / 2 - BAR_T / 2  # 0.195: the bar's centre line from the tote's centre
GRIP_Z = (BAR_Z0 + BAR_Z1) / 2  # 0.046: the bar's centre height in the tote frame

# The picked goods: a carton on the floor, part of the rigid compound (the goods do not move in the model).
CARTON = (0.200, 0.260, 0.030)  # x, y, z

# -- the cart (fixed) -------------------------------------------------------------------------------------------------
DECK_T = 0.006  # steel deck, 6 mm
BAY_X = 0.24
BAY_DY = 0.23  # the bays' centres at y +-0.23: the totes stand 60 mm apart end to end
PICK_Y = +BAY_DY  # the picked tote's bay (the left arm's side)
BASE_Y = -BAY_DY  # the tote already on the cart (the right arm's side)
DECK_X0, DECK_X1 = 0.08, 0.49
DECK_Y = 0.90
RAIL = 0.006  # the locator frame: 6 mm wide, 6 mm tall, 4 mm clear of the foot all round
RAIL_H = 0.006
POST = 0.025
POST_X = 0.47
POST_Y = 0.44
POST_H = 0.30  # rear uprights and a push handle across them
CART_GREY = (0.52, 0.55, 0.60)
CART_FRICTION = (0.35, 0.30)  # steel (est)

# -- the rest poses (tote origins, heights above the table top) --------------------------------------------------------
BASE_ORIGIN = (BAY_X, BASE_Y, DECK_T)
PICK_ORIGIN = (BAY_X, PICK_Y, DECK_T)
SEAT_Z = DECK_T + SEAT_DZ  # 71 mm: the picked tote's origin when seated
GRIP_TCP_PICK = DECK_T + GRIP_Z + 0.002  # 54 mm: TCP at the bar's centre (+2 mm) with the tote on the deck

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_TOL_Z = 0.001  # 1.0: its height within 1 mm of seated
SEAT_TOL_XY = 0.004  # centre within 4 mm of the lower tote's in x and in y
SEAT_TOL_YAW = 2.0  # long axis within 2 degrees of the lower's (top view)
SEAT_TOL_TILT = 1.0  # tilt under 1 degree
NEAR_XY = 0.020  # 0.5: resting on the lower tote, within 20 mm of centre
NEAR_YAW = 4.0
NEAR_TILT = 3.0
NEAR_Z0, NEAR_Z1 = 0.062, 0.077  # and between 52 and 67 mm above it: seated or riding on the rim, not hovering
ON_XY = 0.060  # 0.25: on the lower tote: its centre within 60 mm in x and y, between 45 and 80 mm above it
ON_Z0, ON_Z1 = 0.055, 0.090
ON_TILT = 20.0
ON_YAW = 20.0  # and its long axis within 20 degrees of the lower's: a tote across the rim is not a stack
MOVED_XY = 0.010  # latched: the lower tote pushed more than 10 mm from where it stood
MOVED_YAW = 5.0  # or turned more than 5 degrees (or tilted more than 5 degrees)
FALL_Z = 0.003  # latched: a tote's foot underside lower than 3 mm above the table (off the deck)
TIP_DEG = 30.0  # latched: a tote tilted more than 30 degrees (on its side)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visibly still: under 5 mm over the last 1 s (a person checks it on video)
STILL_TURN_DEG = 5.0  # and under 5 degrees
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

# -- the tote's own points (tote frame), for the scorer ----------------------------------------------------------------
FOOT_PTS = [(sx * FLOOR_W / 2, sy * FLOOR_L / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
RIM_PTS = [(sx * TOTE_W / 2, sy * TOTE_L / 2, TOTE_H) for sx in (-1, 1) for sy in (-1, 1)]


def seat_offset_ok(dx: float, dy: float) -> bool:
    """Whether a foot dropped ``dx``, ``dy`` off centre goes in (the lead-in takes up to CAPTURE)."""
    return abs(dx) <= CAPTURE and abs(dy) <= CAPTURE


def seated_offsets(dx: float, dy: float) -> tuple[float, float]:
    """Where a foot that went in ends: centred to within the 4 mm of clearance."""
    return (max(-CLEAR, min(CLEAR, dx)), max(-CLEAR, min(CLEAR, dy)))


def load_per_hand() -> float:
    """N each hand carries with both hands on the hand holds."""
    return TOTE_MASS * G / 2.0


def tilt_clearance_deg() -> float:
    """The turn about the vertical (degrees) at which the foot's corner meets the lower wall's inner face."""
    return math.degrees(math.atan2(CLEAR, FOOT_L / 2))
