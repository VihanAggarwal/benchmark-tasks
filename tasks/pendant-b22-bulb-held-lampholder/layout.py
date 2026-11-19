"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights here are above the table top unless named *_W.

A plain plastic pendant set hangs from a rigid stand that stands in for the ceiling: a ceiling rose, 240 mm of round
flex and a B22d lampholder, mouth down. A new A60 B22 LED bulb stands cap-up in a ring stand on the table. The bulb
goes in as a B22 bayonet does: its two pins go up the vertical legs of two J-slots against two sprung plungers, the
bulb turns about 15 degrees and drops 2 mm into the hooks, where the plungers hold it.

The holder's own frame (parts.py, scoring.py) has its origin at the centre of the mouth, z up along the holder's
axis, x toward the hook of J-slot A. "h" is height above the mouth in that frame. The bulb's frame has its origin at
the centre of the cap's end face, z along the bulb's axis toward the cap, x through pin A.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the stand that stands in for the ceiling (fixed to the table) --------------------------------------------------
STAND = "ceiling_stand"
POST_X, POST_Y = 0.450, 0.000  # a 30 mm square steel post
POST_W = 0.030
BEAM_Z0, BEAM_Z1 = 0.500, 0.520  # its top arm, underside 500 mm up
BEAM_W = 0.030
BEAM_X0 = 0.260  # the arm reaches back past the rose to x 260 mm
FOOT = (0.160, 0.140, 0.010)  # a steel foot plate under the post, screwed to the table
STEEL = (0.30, 0.31, 0.33)

# -- the pendant set ------------------------------------------------------------------------------------------------
PENDANT = "pendant"
PEND_X, PEND_Y = 0.300, 0.000  # the rose's (and the holder's) axis, under the arm
ROSE_R = 0.0375  # 75 mm across (est)
ROSE_Z0, ROSE_Z1 = 0.470, 0.4995  # 30 mm deep (0.5 mm under the arm, both fixed)
FLEX_R = 0.003  # 6 mm round two-core flex (est)
FLEX_LEN = 0.240  # rose to holder (est)
SWING_Z = ROSE_Z0  # the flex's two swing joints at the rose
HOLDER_TOP = SWING_Z - FLEX_LEN  # 230 mm
HOLDER_LEN = 0.052  # est
MOUTH_Z = HOLDER_TOP - HOLDER_LEN  # 178 mm: the mouth's height as it hangs at the start
HOLDER_R = 0.019  # 38 mm across (est)
WHITE = (0.93, 0.93, 0.91)
FLEX_WHITE = (0.90, 0.90, 0.88)

# The flex as a joint chain: two swing hinges at the rose (x, then y), a free slack slide up the flex (the flex buckles
# when the holder is pushed up), a stiff stretch spring down it (the flex pulled tight against the rose), and the
# holder's twist hinge about its own axis (the flex's weak twist spring).
SWING_LIMIT = 1.2  # rad each way
SWING_DAMP = 0.01  # N m s/rad (est: light)
SLACK_UP = 0.040  # the holder can be pushed up 40 mm before the flex resists
SLACK_DAMP = 0.1  # N s/m
STRETCH_MAX = 0.020
STRETCH_K = 600.0  # N/m (est): 15 mm takes 9 N
STRETCH_DAMP = 3.0
TWIST_LIMIT = 3.0  # rad each way (172 degrees)
TWIST_K_PER_DEG = 0.0005  # N m per degree (est)
TWIST_K = TWIST_K_PER_DEG * 180.0 / math.pi  # N m/rad
TWIST_DAMP = 0.0005
FLEX_MASS = 0.010
LINK_MASS = 0.005  # the small links (swing hub, slack and stretch collars, plungers, catches)
HOLDER_MASS = 0.030  # est
HOLDER_FRICTION = (0.5, 0.4)  # est: white plastic
HANGING_MASS = HOLDER_MASS + 2 * LINK_MASS + 4 * LINK_MASS  # what hangs below the slack slide: 60 g

# -- the holder (heights h above the mouth) --------------------------------------------------------------------------
WALL_SEGS = 12
WALL_IN = 0.014  # the bore the pins run in: 28 mm (est)
MOUTH_H = 0.005  # the mouth: a 45-degree funnel from 38 mm to the 28 mm bore
GUIDE_IN = 0.01275  # a 25.5 mm collar at the top of the bore keeps the cap's end within 1.75 mm of the axis (est)
GUIDE_H0, GUIDE_H1 = 0.02375, 0.025  # its chamfer, above the pins' highest path (their tops at 23.5 mm at c 6 mm)
CEIL_H = 0.032  # the bore's roof (the plungers' housing above it): 7 mm over the plungers' tips
# GPU run 1 (2026-11-18): with a 23 mm collar (0.5 mm of play round the 22 mm cap) whose chamfer ended at the
# plungers' tips, the cap arrived 0.6 mm off the axis and 1.4 degrees tilted, sat on the collar's edge at c = 0 and
# never reached the plungers: the push went into the holder through the chamfer. The cap now has 1.75 mm of play,
# and nothing rigid on the holder is in its path from the plungers' rest to 6 mm up (tests: Clearance).

# The two spring plungers in the roof, 11 mm apart: long soft springs (est 5 mm travel, 0.4 N each at rest, 0.45 N at
# full travel). GPU run 2 (2026-11-18): with 500 N/m plungers (1.5 N each at the hooks, 3 N at the stop) the cap rose
# 0.3 mm. The arms are soft (about 150 to 250 N/m at the TCP, tests: ArmGive): the right stalled 9 mm under its command
# at 1.4 N and the left rose 5 mm with the holder. Soft plungers need under 1 N, and their 0.8 N preload is still
# more than the 0.59 N hanging below the slack slide, so with no hand on it the holder rides up instead.
PL_R = 0.002
PL_OFF = 0.0055
H_PL = 0.025  # the plungers' tips at rest: the cap's end meets them here
PL_LEN = 0.010
PL_TRAVEL = 0.005
PL_K = 10.0  # N/m
PL_PRELOAD = 0.040  # drive target 40 mm below the rest stop: 0.4 N each at rest
PL_DAMP = 0.3

# The hooks: in each J-slot a catch (the hook's floor) with two 1 mm lips either side of the pin's notch. The J's
# shape is declared, not drawn (scoring.Bayonet): the catches slide in under the pins when the J rule says the pins
# are in the hooks and back out when they leave.
CATCH_R0, CATCH_R1 = 0.014, 0.0175  # radial extent at rest (out of the pins' way)
CATCH_T = 0.006  # tangential width
CATCH_H0, CATCH_H1 = 0.0135, 0.0165  # CATCH_H1 is the hook floor
CATCH_TRAVEL = 0.0025  # in: its inner face at 11.5 mm, 1.5 mm under the pins
CATCH_K = 2000.0
CATCH_DAMP = 2.0
CATCH_IN, CATCH_OUT = 0.003, -0.0005  # drive targets (a preload against each stop)
LIP_W, LIP_C, LIP_H = 0.001, 0.0025, 0.001  # 1 mm lips centred 2.5 mm either side of the notch, 1 mm tall

# -- the bulb (heights z in its own frame: 0 at the cap's end, negative toward the glass) ----------------------------
BULB = "bulb"
BULB_MASS = 0.040  # est
BULB_LEN = 0.104  # est (A60)
GLOBE_R = 0.030  # 60 mm across (published)
GLOBE_C = -0.074  # the globe's centre
CAP_R, CAP_LEN = 0.011, 0.018  # B22d cap, 22 mm (published)
PIN_FROM_END = 0.009
PIN_R = 0.0015
PIN_OUT = 0.013  # pin tips 26 mm apart
NECK_R = 0.0115
NECK_Z = (-0.030, -CAP_LEN)
GLASS = (  # (z0, z1, radius): the neck's taper and the globe as stacked cylinders
    (-0.036, -0.030, 0.016),
    (-0.042, -0.036, 0.019),
    (-0.050, -0.042, 0.0225),
    (-0.058, -0.050, 0.026),
    (-0.066, -0.058, 0.029),
    (-0.078, -0.066, GLOBE_R),
    (-0.086, -0.078, 0.0287),
    (-0.094, -0.086, 0.025),
    (-0.100, -0.094, 0.019),
    (-0.104, -0.100, 0.0105),
)
GRIP_FROM_END = 0.084  # the right gripper's TCP on the axis 84 mm below the cap's end (globe radius 28.7 mm there)
GRIP_R = 0.0287
FROSTED = (0.95, 0.95, 0.93)
CAP_METAL = (0.72, 0.70, 0.62)
BULB_FRICTION = (0.8, 0.7)  # est: glass on the gripper pads

# The bayonet's geometry (c = how far the cap's end has pushed the plungers up)
C_LOCK = 0.002  # in the hooks: the pins on the hook floor, the plungers 2 mm in
C_HOOK = 0.003  # the pins clear of the lips: they can turn into the hooks from here up
C_MAX = PL_TRAVEL  # the plungers' stop (the top of the J's leg)
H_LOCK_END = H_PL + C_LOCK
ENTRY_DEG = -15.0  # the J's vertical leg is 15 degrees round from the hook
HOOK_DEG = 15.0  # turn from the leg to the hook
LEG_TOL_DEG = 5.0
TOOTH_DEG = 5.0  # between 5 and 13 degrees of turn the pins are under the J's tooth (needs c >= C_HOOK)
HOOK_LO_DEG, HOOK_HI_DEG = 13.0, 17.0  # in the hook's notch
HOLD_LO_DEG, HOLD_HI_DEG = 9.0, 21.0  # once in, the lips keep it within this
OVER_DEG = 20.0  # past the J's end: forced
INSIDE_LAT = 0.002  # the cap counts as in the bore while its axis is within 2 mm and 6 degrees of the holder's
INSIDE_TILT_DEG = 6.0

# -- the ring stand ---------------------------------------------------------------------------------------------------
RING_STAND = "bulb_stand"
RING_X, RING_Y = 0.250, -0.130
RING_BASE_R, RING_BASE_H = 0.040, 0.006
RING_POST_R, RING_POST_TOP = 0.012, 0.040
RING_IN, RING_OUT = 0.015, 0.020
RING_TOP = 0.050  # 50 mm tall
RING_SEGS = 12
BULB_START_Z = RING_TOP + 0.100  # the cap's end, the bulb resting on the ring by its globe's lower part
BULB_START_YAW_DEG = ENTRY_DEG  # pins lined up with the J's legs (paint marks at the entry angle)
STAND_GREY = (0.25, 0.25, 0.26)

# -- scoring --------------------------------------------------------------------------------------------------------
STILL_WINDOW_S = 1.0
LOCK_CATCH = 0.8  # each catch at least 80% of its travel in
LOCK_TURN = (13.0, 17.0)
LOCK_C = (0.0015, 0.0025)  # within 0.5 mm of the hook floor
LOCK_TILT_DEG = 5.0
LOCK_LAT = 0.0015
RELEASE_GAP = 0.002  # every finger zone 2 mm or more from the bulb and the holder
HANG_MOVE = 0.010
HANG_PLAN = 0.030
HANG_DZ = 0.015
STRAIN_DZ = 0.015
WOUND_DEG = 90.0
LEFT_STAND = 0.008  # the bulb has left its stand once its cap's end is 8 mm over where it started
DROP_TABLE = 0.002
DROP_TOUCH = 0.001
START_TOL = 0.010
RESET_JUMP = 0.050
EPS = 1e-6


def plunger_force(c: float) -> float:
    """N on each plunger with the cap's end ``c`` (m) above the plungers' rest."""
    return PL_K * (PL_PRELOAD + min(max(c, 0.0), PL_TRAVEL))


def holder_grip_point() -> tuple[float, float, float]:
    """The left gripper's TCP on the holder's axis, 40 mm above the mouth (table-relative)."""
    return (PEND_X, PEND_Y, MOUTH_Z + 0.040)
