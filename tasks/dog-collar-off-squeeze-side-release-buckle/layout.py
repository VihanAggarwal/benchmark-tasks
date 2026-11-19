"""Every number of the scene, in metres, kg, N and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP. "Table frame" in the
comments is the world with z measured from the table top. The buckle frame B has its origin at (BX, BY, TOP + BZ0),
the buckle's underside on top of the neck, axes the world's: B's +y runs from the female's mouth toward its back.
The female (base of the collar articulation) and the male (its own small articulation) both use B as their base
frame, so at rest a point has the same coordinates on either.

Figures marked est in task.json are design values the simulation is built to; the setup's measure-and-accept checks
say how the real parts are matched to them.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
TABLE_X = (-0.19, 0.91)  # the station table (yamkin.TABLE_BOX)
TABLE_Y = (-0.65, 0.65)

# -- the dog manikin (Simulaids CasPeR, 711 x 584 x 254 mm overall, 6.8 kg), lying on its right side ---------------
# Head toward the arms (nose at x 88), back toward -y, legs toward +y. Drawn from cylinders and boxes to that outline;
# the neck is a cylinder along x, 104 mm across (327 mm round), held 25 mm off the table by the head and the chest.
DOG_MASS = 6.8
DOG_FRICTION = (0.6, 0.5)  # vinyl skin, est
DOG_TAN = (0.70, 0.50, 0.30)  # a darker tan than the table top, so the outline reads in the top camera
DOG_DARK = (0.30, 0.18, 0.10)  # ears
DOG_BLACK = (0.05, 0.05, 0.05)  # nose, eyes
NECK_X0, NECK_X1 = 0.265, 0.380
NECK_R = 0.052
NECK_Z = 0.077  # axis height above the table: underside 25 mm up, top 129 mm up
NECK_GAP = NECK_Z - NECK_R
NECK_TOP = NECK_Z + NECK_R
HEAD_R = 0.066  # behind the ears: 415 mm round, 35 mm more than the collar loop
HEAD_X0, HEAD_X1 = 0.175, 0.265
DOG_ORIGIN = (0.3225, 0.0, TOP)  # the manikin's base frame: under the middle of the neck, on the table
NOSE_PT = (0.088, 0.008, 0.048)  # table frame: the nose's tip (the tape mark)
RUMP_PT = (0.800, -0.020, 0.110)  # the rump's back (the second tape mark)

# -- the collar --------------------------------------------------------------------------------------------------
# Blueberry Pet Classic nylon collar, size M, 3/4 in (19 mm) webbing, red, with a black 3/4 in side-release buckle.
RED = (0.72, 0.05, 0.08)
BLACK = (0.06, 0.06, 0.07)  # the male half, the tri-glide
F_GREY = (0.36, 0.37, 0.40)  # the female half: dark grey, so the seam with the black male shows
STEEL = (0.90, 0.90, 0.92)  # the D-ring: light silver
COLLAR_FRICTION = (0.45, 0.4)  # nylon and acetal on vinyl, est
BX, BY = 0.320, 0.0  # the collar's plane x and the buckle frame's y
BZ0 = NECK_TOP + 0.0005  # the buckle's underside, 0.5 mm over the neck's top
B_ORIGIN = (BX, BY, TOP + BZ0)

# Female half (buckle frame, m): 29 wide (x), from the mouth at y -4 mm to its back at y +30 mm, 11 mm tall.
F_HALF_W = 0.0145
F_Y0, F_Y1 = -0.004, 0.030
F_H = 0.011
F_WALL_IN = 0.0115  # the side walls' inner faces (x); the channel is 23 mm wide
F_FLOOR, F_ROOF = 0.0015, 0.0095  # the channel's floor top and roof underside (z)
NOTCH_Z0, NOTCH_Z1 = 0.003, 0.008  # each side wall is cut open from the mouth to y 15 mm between these heights
NOTCH_Y1 = 0.015
RIB_X0 = 0.0100  # the catch shoulder: a rib on each side wall's inner face, x 10.0 to 11.5 mm,
RIB_Y0, RIB_Y1 = 0.015, 0.018  # y 15 to 18 mm, across the notch's height
F_MASS = 0.006

# Male half: the exposed body (25 wide, y -26 to -4.3 mm), a centre tongue and two sprung prongs on hinges.
M_HALF_W = 0.0125
M_Y0, M_Y1 = -0.026, -0.0043  # exposed body; its back (y -26) is the webbing bar
M_Z0, M_Z1 = 0.0005, 0.0105
TONGUE_HALF_W, TONGUE_Y1, TONGUE_Z = 0.0045, 0.022, (0.0025, 0.0085)
PRONG_ANCHOR_X, PRONG_ANCHOR_Y = 0.0085, -0.008  # hinge axis (vertical) of the +x prong; the -x one mirrors it
PRONG_X = (0.0075, 0.0095)  # the prong's beam
PRONG_Y1 = 0.0223  # its tip
PRONG_Z = (0.0035, 0.0075)
TAB_X1 = 0.0160  # the press tab stands 1.5 mm proud of the female's 14.5 mm side
TAB_Y = (0.006, 0.014)  # centred 14 mm in from the mouth
HOOK_X1 = 0.0113  # the hook: from the beam out to x 11.3 mm, 1.3 mm over the rib (the catch)
HOOK_Y = (0.0183, PRONG_Y1)  # 0.3 mm of play behind the rib
TAB_PROUD = TAB_X1 - F_HALF_W  # 1.5 mm
TAB_LEVER = (TAB_Y[0] + TAB_Y[1]) / 2 - PRONG_ANCHOR_Y  # 18 mm from the hinge to the tab's centre
HOOK_LEVER = HOOK_Y[0] - PRONG_ANCHOR_Y  # 26.3 mm to the hook's nearest corner
CATCH = HOOK_X1 - RIB_X0  # 1.3 mm
FLUSH_RAD = TAB_PROUD / TAB_LEVER  # the prong's turn with its tab pressed flush
RELEASE_RAD = CATCH / HOOK_LEVER  # the turn at which the hook clears the rib
TAB_FORCE_FLUSH = 2.5  # N at the tab, pressed flush (est)
PRONG_SPRING = TAB_FORCE_FLUSH * TAB_LEVER / FLUSH_RAD  # N m / rad, toward the drawn (sprung-out) pose
PRONG_UPPER = 0.16  # rad; the fingers stop on the housing at FLUSH_RAD long before this
PRONG_DAMP = 0.0005
PRONG_MASS = 0.0007
M_MASS = 0.004
PLAY = HOOK_Y[0] - RIB_Y1  # 0.3 mm: how far a clicked male can move out
M_TIP_Y = PRONG_Y1  # the male's +y end (hooks)
M_LEN = M_TIP_Y - M_Y0  # 48.3 mm, bar to hook tips
F_LEN = F_Y1 - F_Y0  # 34 mm
OUT_FULL = M_TIP_Y - F_Y0  # 26.3 mm: the male's travel until no part of it is inside the female

# Webbing: 13 rigid links (25 mm pitch, 24.5 x 19 x 2 mm) on hinges along x, from the female's back (y +30 mm) round
# the neck to the male's bar (y -26 mm), drawn on the circle through those two points that makes the loop 381 mm.
N_LINKS = 13
PITCH = 0.025
LINK_LEN = 0.0245
WEB_W = 0.019
WEB_T = 0.002
LINK_MASS = 0.0012
WEB_DAMP = 0.0002  # N m s / rad: nylon webbing, nearly free to bend (est)
WEB_LIMIT = 2.6  # rad either way
DRING_LINK = 13  # the D-ring lies flat on the last link, next to the male
DRING_MASS = 0.003
TRIGLIDE_LINK = 2  # the tri-glide on the second link, on the neck's +y side
TRIGLIDE_MASS = 0.0025
END_Z = 0.0055  # the webbing meets both buckle halves at mid height (buckle frame)
P_FEMALE = np.array([BX, F_Y1, BZ0 + END_Z])  # table frame
P_MALE = np.array([BX, M_Y0, BZ0 + END_Z])
LOOP_LEN = N_LINKS * PITCH + (F_Y1 - M_Y0)  # 381 mm round the webbing's middle


def _loop_radius() -> float:
    half = (F_Y1 - M_Y0) / 2
    lo, hi = 0.05, 0.08
    for _ in range(100):
        r = (lo + hi) / 2
        total = N_LINKS * 2 * math.asin(PITCH / 2 / r) + 2 * math.asin(half / r)
        lo, hi = (r, hi) if total > 2 * math.pi else (lo, r)
    return (lo + hi) / 2


LOOP_R = _loop_radius()
LOOP_C = np.array([BX, (F_Y1 + M_Y0) / 2, P_FEMALE[2] - math.sqrt(LOOP_R**2 - ((F_Y1 - M_Y0) / 2) ** 2)])


def chain_vertices() -> np.ndarray:
    """(N_LINKS + 1, 3) table-frame points: the female's back, the 12 hinges, the male's bar; equal 25 mm chords
    round LOOP_C, going from the female down the +y side, under the neck and up the -y side."""
    a0 = math.atan2(P_FEMALE[2] - LOOP_C[2], P_FEMALE[1] - LOOP_C[1])
    step = 2 * math.asin(PITCH / 2 / LOOP_R)
    pts = []
    for i in range(N_LINKS + 1):
        a = a0 - i * step  # clockwise seen from +x: +y side first, then the bottom, then -y
        pts.append((BX, LOOP_C[1] + LOOP_R * math.cos(a), LOOP_C[2] + LOOP_R * math.sin(a)))
    pts = np.array(pts)
    pts[-1] = P_MALE  # the last vertex is the male's bar (the radius makes them meet to well under 1 micron)
    return pts


VERTS = chain_vertices()


def link_angle(p0, p1) -> float:
    """The roll about x that turns +y onto the chord p0 -> p1."""
    return math.atan2(float(p1[2] - p0[2]), float(p1[1] - p0[1]))


# -- the arms ------------------------------------------------------------------------------------------------------
FINGER_STIFF = 5000.0  # N/m per finger: the sim's finger drive
OPEN_GAP_FEMALE = 0.044
PINCH_GAP = 0.027  # 2 mm under the 29 mm housing: 5 N a finger at the stop (tab 2.5 N + housing 2.5 N)
OPEN_GAP_MALE = 0.036
MALE_GAP = 0.023  # 2 mm under the 25 mm body: 5 N a finger
ROLL_L, ROLL_R = -0.35, 0.85  # rad about world x: the wrists lean apart (left tips toward -y, right toward +y)

# -- the scorer ----------------------------------------------------------------------------------------------------
OFF_DOG = 0.025  # every collar part at least 25 mm from the manikin (a 2 mm link under the 25 mm neck gap is 23)
REST_HIGH = 0.040
STILL_SPEED = 0.005
HANDS_OFF = 0.030
UNLATCH_OUT = 0.003
APART_OUT = 0.005
DOG_NUDGED = 0.020  # the dog's nose or rump moved this far at any step: the score is capped at 0.75
DOG_MOVED = 0.100  # moved this far (dragged, rolled or its head lifted clear): latch B, 0
BELOW_TABLE = 0.003
WINDOW = 20  # steps: the last 1 s at 20 Hz


def pinch_force(gap: float, part_width: float) -> float:
    """N per finger when commanded to ``gap`` across a part ``part_width`` wide (the finger drive's spring)."""
    return FINGER_STIFF * max(0.0, (part_width - gap) / 2)
