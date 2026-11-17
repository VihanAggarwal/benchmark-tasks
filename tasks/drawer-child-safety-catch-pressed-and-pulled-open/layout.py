"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305 and the station's table top is at z 0.7475. Heights below are above the table top unless named otherwise.

The unit is a stock MUJI stacking chest, half, 1 drawer, oak veneer (JAN 4550512381105; W37 x D28 x H18.5 cm,
drawer inside W33.5 x D24.5 x H15.5 cm, about 5 kg, particle board, MDF and white oak veneer: MUJI's product pages).
It stands on the table, its front at x 300 mm, centred on y 0, screwed down. Its drawer front is inset, flush with the
carcass (est), with an IKEA BAGGANAS bar handle (143 mm, 128 mm hole centres, 31 mm deep) screwed on. A Safety 1st
cabinet and drawer latch (model 48390) is fitted inside the drawer as its maker's guide says: the latch screwed to
the back of the drawer front at the top, 140 mm left of centre, the catch screwed under the chest's top so that the
latch's hook meets it when the drawer front's back face is about 1.5 in (38 mm) out of the carcass.

Published: the outside (370 x 280 x 185) and the drawer's inside (335 x 245 x 155). Everything else about the chest
(panel thicknesses, the inset front, the drawer's sides, back and bottom, gaps) is an estimate (est), chosen to fit
those published sizes; the README gives the bench check for each.

The base frame (the unit) has its origin at the carcass's front face, bottom, centre; local x points back (world +x),
y left, z up; the base sits at world (UX, UY, TOP) unturned. Joint values: "drawer" the slider's travel in metres (0
shut, positive out, toward world -x); "latch" the latch arm's angle in radians about the hinge line along y (0 at
rest, positive presses the hook end down). The inset front's back face is FRONT_BACK (15 mm) inside the carcass when
shut, so the opening the scorer reads (the back of the front past the carcass's front edge) is the travel minus 15 mm.
The catch is a separate fixed part in the same frame (it must collide with the latch: links of one assembly never
collide with each other).
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the unit (carcass): fixed; MUJI stacking chest half 1 drawer, oak veneer -----------------------------------------
UX, UY = 0.300, 0.000
ORIGIN = (UX, UY, TOP)
W, D, H = 0.370, 0.280, 0.185  # outside: 370 wide (y), 280 deep (x), 185 tall (z) (published)
T = 0.012  # top and bottom panels (est)
T_SIDE = 0.009  # side panels (est: the drawer's published 335 mm inside leaves 17.5 mm a side for panel, gap, side)
T_BACK = 0.004  # back panel (est)
TOP_Z0 = H - T  # 0.173: the underside of the top
SIDE_Y0 = W / 2 - T_SIDE  # 0.176: the sides' inner faces
PLY = (0.80, 0.68, 0.50)  # white oak veneer
PLY_EDGE = (0.74, 0.62, 0.45)

# -- the drawer (one prismatic link along -x of the base frame, i.e. toward the arms) ----------------------------------
FRONT_T = 0.015  # the inset front, 15 mm (est), its face flush with the carcass's front face when shut
FRONT_BACK = FRONT_T  # the front's back face, 15 mm inside the carcass's front edge when shut
FRONT_W, FRONT_Z0, FRONT_Z1 = 0.346, 0.014, 0.171  # 346 x 157 mm in the 352 x 161 mm opening (est: 3 and 2 mm gaps)
BOX_SIDE_T = 0.006  # the drawer's sides and back (est)
BOX_Y0 = 0.1675  # the drawer sides' inner faces: 335 mm between them (published), outer faces 2.5 mm off the carcass
BOX_LEN = 0.251  # back face of the front to the back of the drawer box: 245 mm inside (published) and a 6 mm back
BOX_Z0, BOX_Z1 = 0.012, 0.171  # sides 159 mm tall, riding on the carcass bottom, 2 mm under the top
BOTTOM_T = 0.004
BOTTOM_Z0 = 0.012  # the bottom's top face at 16 mm: 155 mm inside (published) up to the sides' top edges
TRAVEL = 0.200  # the slider's end (the chest has no stop of its own: past about 250 mm the drawer comes out)
DRAWER_MASS = 1.4  # kg (est): the drawer, empty (the whole chest is about 5 kg, published)
DRAWER_DAMPING = 60.0  # N s/m: 1.8 N of drag at the demo's 30 mm/s (wood on wood, est 1 to 3 N)

# -- the handle: IKEA BAGGANAS 143 mm, stainless steel (length 143, width 9, depth 31, holes 128 mm; IKEA spec page) --
HANDLE_LEN = 0.143
HANDLE_W = 0.009  # the bar's face width (vertical here)
HANDLE_DEPTH = 0.031  # front face of the drawer to the bar's front face
HANDLE_BAR_X = 0.009  # the bar's section along x (est; calipers on the bought handle)
HANDLE_HOLES = 0.128
HANDLE_POST_R = 0.004  # the legs (est)
HANDLE_Z = 0.140  # 31 mm below the front's top edge, 45 mm below the chest's top (a top-mounted pull: the gripping
# pads' wider upper part, which reaches over the chest's top edge, passes 15 mm above it)
HANDLE_X0 = -HANDLE_DEPTH  # -0.031: the bar's front face (base frame, drawer shut; the inset front is flush at x 0)
HANDLE_X1 = HANDLE_X0 + HANDLE_BAR_X  # -0.022: its back face
HANDLE_XC = (HANDLE_X0 + HANDLE_X1) / 2  # -0.0265
BEHIND_BAR = -HANDLE_X1  # 22 mm between the bar's back face and the drawer front: room for a finger
STEEL = (0.74, 0.75, 0.77)

# -- the latch: Safety 1st cabinet and drawer latch 48390 (83 x 30 x 21 mm on the Target listing) --------------------
# Every latch and catch figure below is placed from the back of the drawer front (x) and the underside of the chest's
# top (z) exactly as on the earlier build that scored 1.0 on the GPU: only the frame moved (15 mm back, 25 mm up).
LATCH_Y = 0.140  # its centre line, 140 mm left of the drawer's centre (any position along the top edge per the
# guide; 140 keeps the pressing pads 14 mm off the drawer's side and the two forearms 34 mm apart)
LATCH_HALF_W = 0.0105  # 21 mm wide
LATCH_LEN = 0.083  # 83 mm from the back of the drawer front to the back of the hook
TAB_X1 = FRONT_BACK + 0.005  # the mounting foot, 5 mm thick, screwed to the back of the drawer front
TAB_Z0, TAB_Z1 = 0.135, 0.165  # 30 mm tall (the listing's 1.18 in), its top 6 mm below the front's top edge
ARM_Z0, ARM_Z1 = 0.147, 0.151  # the flexible arm, 4 mm thick (est)
HOOK_X0, HOOK_X1 = FRONT_BACK + 0.077, FRONT_BACK + 0.083  # the hook at the arm's end: its front face blocks
HOOK_Z1 = 0.165  # the hook stands 14 mm above the arm's underside
PIVOT_RISE = 0.004  # the flexing root is modelled as a pin 4 mm above the foot's top, so the hook's contact patch
# (z 159 to 165 mm) is below the pivot line and a pull on the drawer turns the hook up into the catch, not out of it
HINGE = (TAB_X1, LATCH_Y, TAB_Z1 + PIVOT_RISE)
LATCH_UPPER = 0.25  # rad: the arm bottoms out (18 mm of hook travel)
LATCH_LOWER = -0.02
LATCH_SPRING = 0.30  # N m/rad (est): about 2 N at the press point 12.5 mm from the pivot draws the hook clear
LATCH_DAMPING = 0.002
LATCH_MASS = 0.012  # kg (est)
LATCH_WHITE = (0.94, 0.94, 0.92)

# -- the catch: Safety 1st catch, screwed under the top "1 in from the inside edge of the frame" (guide) --------------
CATCH_X0 = 0.025  # its front edge, 25 mm in from the front of the top
CATCH_X1 = 0.039  # the strike face (the lip's back face)
CATCH_HALF_W = 0.0125  # 25 mm wide (est)
CATCH_PLATE_Z0 = 0.169  # 4 mm mounting plate against the underside of the top
LIP_X0 = 0.033  # the lip, 6 mm thick, hangs to 159 mm
LIP_Z0 = 0.159

# -- derived: where the hook meets the catch (travel = the slider's joint; gap = travel - FRONT_BACK) ---------------
ENGAGE_AT = HOOK_X0 - CATCH_X1  # 0.053 travel: the drawer stops with its front's back face 38 mm out
ENGAGE_GAP = ENGAGE_AT - FRONT_BACK  # 0.038: the guide's "about 1.5 in"
PAST_AT = HOOK_X1 - LIP_X0  # 0.065 travel: the hook is all the way past the lip from here
PAST_GAP = PAST_AT - FRONT_BACK  # 0.050
BITE = HOOK_Z1 - LIP_Z0  # 6 mm of hook behind the lip
HOOK_R = HOOK_X0 - HINGE[0]  # 0.072: the hook's top front corner from the pivot line, along x
HOOK_DZ = HOOK_Z1 - HINGE[2]  # -0.004: and along z


def corner_z(angle: float) -> float:
    """The hook's top front corner's height (base frame) with the arm turned ``angle`` (positive presses it down)."""
    return HINGE[2] - HOOK_R * math.sin(angle) + HOOK_DZ * math.cos(angle)


def _clear_rad() -> float:
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if corner_z(mid) <= LIP_Z0 else (mid, hi)
    return hi


CLEAR_RAD = _clear_rad()  # 0.084 rad: the arm pressed this far draws the hook clear of the lip


def arm_top_z(r: float, angle: float) -> float:
    """The arm's top face's height (base frame) at ``r`` along x from the pivot line, the arm turned ``angle``."""
    h = ARM_Z1 - HINGE[2]  # -0.018: the arm's top face below the pivot line
    return HINGE[2] - r * math.tan(angle) + h / math.cos(angle)


def press_angle(r: float, z: float) -> float:
    """The angle at which the arm's top face at ``r`` from the pivot is down at height ``z`` (0 if it is not reached),
    capped at the arm's stop."""
    if r <= 0.0 or z >= arm_top_z(r, 0.0):
        return 0.0
    if z <= arm_top_z(r, LATCH_UPPER):
        return LATCH_UPPER
    lo, hi = 0.0, LATCH_UPPER
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if arm_top_z(r, mid) > z else (lo, mid)
    return (lo + hi) / 2

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
OPEN_FULL = 0.100  # the front's back face 100 mm out of the carcass: a hand goes in
OPEN_PAST = 0.060  # 60 mm: the hook 10 mm past the catch (the latch has been worked)
FORCE_OVERLAP = 0.002  # the hook more than 2 mm into the lip in both x and z: forced through (broken in a real trial)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # the drawer moves under 5 mm over the last second
LET_GO = 0.030
RESET_JUMP = 0.05
START_TOL = 0.002
EPS = 1e-9
