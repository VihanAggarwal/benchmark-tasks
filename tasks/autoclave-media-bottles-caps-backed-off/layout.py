"""Every dimension of the task, in one place (metres, kilograms, rad, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

Two DURAN-type 250 mL laboratory bottles (70 mm across, 143 mm to the top of the GL45 neck), each with 150 mL of
water for media and a blue GL45 screw cap closed finger tight, stand on the bench, one each side of the pan. A
polypropylene autoclave pan (380 x 240 mm, 75 mm deep) stands between them and the front of a bench-top
autoclave stands behind it with its door open. The job: stand both bottles in the pan,
each with its cap backed off from tight by a quarter to one turn but still on its thread, so that no sealed
bottle goes into the cycle. As at the bench, one hand holds the bottle while the other turns its cap: the cap's
torque is more than the bottle's base friction can hold.

Figures are listing figures (README) or estimates (est, said in task.json's fidelity).
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0

# -- the autoclave pan (fixed) ---------------------------------------------------------------------------------------
PAN_CX, PAN_CY = 0.290, 0.0
PAN_OX, PAN_OY, PAN_H = 0.240, 0.380, 0.075  # outer 240 (x) x 380 (y), 75 deep: the long side across the table
PAN_WALL = 0.003
PAN_FLOOR = 0.003  # the floor's top above the table
PAN_X0, PAN_X1 = PAN_CX - PAN_OX / 2 + PAN_WALL, PAN_CX + PAN_OX / 2 - PAN_WALL  # inner walls: x 173 to 407 mm
PAN_Y0, PAN_Y1 = PAN_CY - PAN_OY / 2 + PAN_WALL, PAN_CY + PAN_OY / 2 - PAN_WALL  # y -187 to +187 mm
PAN_FRICTION = 0.5  # polypropylene under glass (est)
PAN_WHITE = (0.82, 0.86, 0.88)

# -- the autoclave (fixed, behind the pan): the front of a bench-top steriliser with its door swung open --------------
AUTO_X0, AUTO_X1 = 0.600, 0.900  # the modelled front section, 300 deep
AUTO_Y = 0.440  # 440 wide, centred on the midline
AUTO_H = 0.360
AUTO_WALL = 0.040  # side, floor and ceiling thickness
CHAMBER_Z0 = 0.045  # the chamber floor above the table
CHAMBER_W = 0.360  # the chamber opening, 360 wide and 275 tall
AUTO_GREY = (0.62, 0.64, 0.66)
AUTO_STEEL = (0.76, 0.77, 0.78)

# -- the bottles (loose, each an articulation: base, a rise slider, a turn hinge) -------------------------------------
BODY_R = 0.035  # 70 mm across
BODY_H = 0.100  # the straight body, to here
SHOULDER_R = 0.031  # the shoulder, up to the cap's seat
NECK_R = 0.0235  # the GL45 neck, 47 mm across (the thread)
NECK_TOP = 0.143  # 143 mm to the top of the neck
CAP_R = 0.027  # the screw cap, 54 across (est)
CAP_SEAT = 0.118  # the skirt's underside above the base when tight (est)
CAP_H = 0.027  # 27 tall (est): skirt, and a plate 2 mm over the neck's top
CAP_TOP = CAP_SEAT + CAP_H  # 145 mm tight
BASE_MASS, SLIDER_MASS, CAP_MASS = 0.310, 0.004, 0.012  # 0.326 kg: bottle 0.16 (est), 150 mL of water, the cap
BOTTLE_FRICTION = 0.8  # the glass, for the grippers' pads too (est)
GLASS = (0.74, 0.86, 0.92)
CAP_BLUE = (0.10, 0.28, 0.75)
MARK_BLACK = (0.04, 0.04, 0.04)
TAPE_COLOUR = (0.90, 0.78, 0.12)
MARK_THICK = 0.0012  # the black line on the shoulder and on the cap: a raised 1.2 mm strip facing -x
MARK_W = 0.004
TAPE_R, TAPE_Z0, TAPE_H = 0.0352, 0.078, 0.012  # the autoclave indicator tape round the body

# start poses (the base's centre) and where the demo stands them in the pan
A0 = (0.400, 0.260)  # bottle a: on the bench left of the pan
B0 = (0.400, -0.260)  # bottle b: on the bench right of the pan
A1 = (0.340, 0.100)  # the demo's spots in the pan
B1 = (0.340, -0.100)

# the thread (est): one full turn lifts the cap PITCH; it comes free after RELEASE_DEG of unscrewing from tight
PITCH = 0.006
RELEASE_DEG = 540.0
STOP_DEG = 355.0  # in the simulator the cap's thread ends in a stop here (a mechkit hinge turns less than 2 pi)
STOP_RAD = math.radians(STOP_DEG)
TIGHTEN_RAD = 0.12  # a cap can be driven this far tighter than the drawn tight (7 degrees)
RISE_MAX = PITCH * STOP_DEG / 360.0 + 0.002  # the slider's travel
CAP_FRICTION = 0.25  # N m of joint friction: the torque to turn a finger-tight cap (est), breaking loose and running
# the bottle's base friction on the pan: mu m g (2/3) r = 0.5 x 0.326 x 9.81 x 0.023 = 0.04 N m, a sixth of that, so
# a cap turned with nothing holding the bottle turns the whole bottle
K_RISE = 3000.0  # N/m: the slider's spring (the thread's lift follows the turn through a declared rule)
D_RISE = 25.0  # N s/m
D_TURN = 0.005  # N m s/rad

# -- scoring figures -------------------------------------------------------------------------------------------------
SEALED_DEG = 90.0  # a cap backed off under a quarter turn from tight: still a sealed bottle
ONE_TURN_DEG = 360.0  # backed off more than one turn: too loose (on its thread, but not "about one turn at most")
THREAD_TOL = 0.002  # a cap's rise must match its turn (6 mm a turn) within 2 mm: it is on its thread
CAP_LATERAL = 0.008  # a cap's axis within 8 mm of its bottle's
STAND_TILT_DEG = 10.0
FLOOR_TOL = 0.003  # the base within 3 mm of the pan's floor
TIP_DEG = 45.0  # a tip: tilt over 45 degrees (water would run out), latched
FALL_DROP = 0.050  # a base more than 50 mm below the table top: off the table
LET_GO = 0.030  # finger zones 30 mm or more from both bottles and caps
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # 5 mm and 5 degrees over 1 s: visible motion on a video
STILL_TURN_DEG = 5.0
RESET_JUMP = 0.05
RESET_TURN = 1.2  # rad in one control step
START_TOL = 0.01
EPS = 1e-9


def gap_for_turn(turn_rad: float) -> float:
    """The rise of a cap on its thread (m) for a turn from tight (rad)."""
    return PITCH * turn_rad / (2.0 * math.pi)


def turn_for_gap(gap: float) -> float:
    """The turn from tight (degrees) that lifts a cap on its thread by ``gap`` metres."""
    return 360.0 * gap / PITCH
