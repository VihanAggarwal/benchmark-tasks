"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305 and the station's table top is at z 0.7475. Heights below are above the table top unless named otherwise.

A Canon EOS 2000D (Rebel T7) body with its kit lens, the EF-S 18-55mm f/3.5-5.6 IS II, sits on its base on the table
with the lens pointing toward -y (toward the right arm). Its centre-pinch front cap, a Canon E-58 II, lies face up beside
it. The cap goes back on: pinch its two tabs (which pulls its two latches in), press it square onto the lens's 58 mm
front, let go (the latches spring out behind the lip of the filter thread), set the camera down.

Published figures (Canon specification pages, README): body 129.0 x 101.3 x 77.6 mm, 475 g with battery and card; lens
68.5 mm across, 70 mm long, 200 g, 58 mm filter thread; cap E-58 II, 58 mm, centre pinch. Everything else is (est), to be
measured on the bought units (README, "Measure before the first trial").

Local frames:
  camera  origin = the centre of the lens's front face (where the cap's seat face sits); +y into the lens toward the
          body, z up, x as the world's (the camera sits square to the table, lens toward -y);
  cap     origin = the centre of its seat face (the rear face of its front plate, which sits on the lens's front face);
          +z out of the front (the tabs' side), the latches on local +-x, the two fixed lugs on local +-y. Lying face up
          it is world-aligned; held for capping it is turned 90 degrees about world x so its front faces -y.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the camera (loose, one rigid body) ----------------------------------------------------------------------------
CAM_X = 0.340  # the lens axis (x)
CAM_Y0 = -0.030  # the lens's front face (y)
CAM_AXIS_Z = 0.044  # the lens axis above the table, the camera on its base (est)
CAM_ORIGIN = (CAM_X, CAM_Y0, TOP + CAM_AXIS_Z)
CAM_MASS = 0.675  # body 475 g with battery and card + lens 200 g (published)
CAM_FRICTION = (0.60, 0.50)  # textured plastic and rubber (est)
CAM_BLACK = (0.07, 0.07, 0.08)
LENS_BLACK = (0.10, 0.10, 0.11)
RING_GREY = (0.22, 0.22, 0.24)

LENS_R = 0.03425  # 68.5 mm across (published)
LENS_LEN = 0.070  # front face to the mount (published, retracted)
LIP_R = 0.0280  # the front lip's inner radius: the filter thread's crests (58 mm thread, est 56.0 mm across the lip)
LIP_T = 0.0015  # the lip's depth along the axis (est)
GROOVE_R = 0.0300  # behind the lip the thread's root, 60.0 mm across (est): the latches catch there
GROOVE_DEPTH = 0.009  # the front element's retaining ring sits 9 mm in (est)
RING_N = 24  # the front ring as a 24-sided polygon of boxes, one flat facing each of +-x and +-z

BODY_W = 0.129  # x (published)
BODY_Y0, BODY_Y1 = LENS_LEN, LENS_LEN + 0.062  # the main body 62 mm deep behind the mount (est; 77.6 mm with grip)
BODY_Z0 = -CAM_AXIS_Z  # the base
TOP_PLATE_Z = 0.076 - CAM_AXIS_Z  # the top plate 76 mm above the base (est)
HUMP_W = 0.058  # the pentaprism hump, 58 mm across (x, est)
HUMP_Y0, HUMP_Y1 = 0.066, 0.112  # 46 mm deep (est)
HUMP_Z1 = 0.1013 - CAM_AXIS_Z  # 101.3 mm to the top (published height)
GRIP_X0, GRIP_X1 = -BODY_W / 2, -0.036  # the hand grip on the photographer's right (world -x here)
GRIP_Y0 = BODY_Y0 - 0.0156  # it stands 15.6 mm proud of the front (77.6 - 62 mm, published depth)
GRIP_Z1 = 0.030
EYECUP = (0.036, 0.012, 0.032)  # x, y, z size behind the hump (est)
EYECUP_Z0 = 0.008

# -- the cap (loose, one floating articulation: the cap and two sprung latch links) -------------------------------
CAP_X, CAP_Y = 0.250, -0.130  # lying face up beside the camera
CAP_R = 0.031  # 62 mm across its front plate (est; it covers the 58 mm thread)
PLATE_T = 0.0025
CAP_BASE_MASS = 0.008
LATCH_MASS = 0.002
CAP_MASS = CAP_BASE_MASS + 2 * LATCH_MASS  # 12 g (est)
CAP_FRICTION = (0.45, 0.40)  # ABS (est)
CAP_BLACK = (0.05, 0.05, 0.06)
TAB_BLACK = (0.16, 0.16, 0.18)

# the two finger tabs on the front, each on its latch link (sprung slider along x)
TAB_X0, TAB_X1 = 0.015, 0.019  # outer faces 38 mm apart (est)
TAB_HALF_Y = 0.012  # 24 mm long (est)
TAB_Z1 = PLATE_T + 0.010  # standing 10 mm proud of the plate (est)
TRAVEL = 0.0025  # a tab and its latch move 2.5 mm in (est)
LATCH_SPRING = 800.0  # N/m: 2 N at the full 2.5 mm (est)
LATCH_DAMP = 2.0
LATCH_OUT = 0.0002  # the slider's outer limit (the crest then 0.2 mm inside the groove)

# the latches on the back (one on each latch link) and two fixed lugs; the teeth reach 6 mm behind the seat face
TEETH_Z = -0.0060  # the teeth's lowest point (the cap lying face up rests on them)
LATCH_CATCH_Z = -0.0018  # the latch's catch face: 0.3 mm behind the lip's rear face when the cap is seated
LATCH_FLAT_Z1 = -0.0032  # the flat runs 1.4 mm along the axis to the ramp's crest (the ramp's inner corner stays
# 0.2 mm under the catch face)
LATCH_CREST_R = 0.0296  # the ramp's crest, from the axis: 1.6 mm outside the lip, 0.4 mm inside the groove
LATCH_FLAT_R = 0.0294  # the flat behind it: its catch face bites 1.4 mm behind the lip; a latch out has 0.6 mm
# to the groove's wall, so the cap cannot slide across far enough for either catch to lose its bite
LATCH_LEAD_R = 0.0255
LATCH_IN_R = 0.0240
LUG_TIP_R = 0.0270  # the fixed lugs: 1.0 mm inside the lip, they centre the cap
LUG_FLAT_R = 0.0268
LUG_Z1 = -0.0030
LUG_LEAD_R = 0.0245
LUG_IN_R = 0.0230
TOOTH_W = 0.010  # each tooth 10 mm wide (tangential)
RAMP_T = 0.0015

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
HOME_OUT = 0.001  # the cap's seat face within 1 mm of the lens's front face
HOME_IN = 0.0005  # (and not more than 0.5 mm past it: the plate cannot go further)
HOME_LAT = 0.002  # its axis within 2 mm of the lens's
HOME_DEG = 3.0  # and square within 3 degrees
LATCH_IN_MAX = 0.0004  # a latch counts as out (behind the lip) within 0.4 mm of rest
LATCH_CLEARS = LATCH_CREST_R - LIP_R  # 1.6 mm: a latch pinched this far clears the lip
UPRIGHT_DEG = 10.0  # the camera on its base: its up axis within 10 degrees of vertical
ON_TABLE = 0.005  # and its lowest point within 5 mm of the table top
FALL_DROP = 0.030  # latched: any part's lowest point more than 30 mm below the table top (off the table)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005
STILL_TURN_DEG = 5.0
LET_GO = 0.030
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

# the cap's start pose
CAP_ORIGIN = (CAP_X, CAP_Y, TOP - TEETH_Z)


def ring_seg_width(r_out: float, n: int = RING_N) -> float:
    """The tangential width of one box of an n-sided ring whose outer polygon has apothem ``r_out``."""
    return 2.0 * r_out * math.tan(math.pi / n) + 0.0004
