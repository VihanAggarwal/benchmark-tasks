"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top.

An engine bay's air cleaner, real size (est until measured on a 2014 to 2019 Toyota Corolla 1.8 L, Toyota 17700
series, or an aftermarket copy): a lower case on a steel stand, its cover on a flexible intake duct, a FRAM CA10171
panel filter in the case's seat, two spring clips on the front wall. Figures are sourced (README) or estimates
(est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the airbox's lower case (fixed) ------------------------------------------------------------------------------
CX, CY = 0.360, 0.010  # centre of the case in top view
CASE_LX, CASE_LY = 0.300, 0.240  # outer size, long side along x (est)
CASE_X0, CASE_X1 = CX - CASE_LX / 2, CX + CASE_LX / 2  # 0.210 .. 0.510
CASE_Y0, CASE_Y1 = CY - CASE_LY / 2, CY + CASE_LY / 2  # -0.110 .. 0.130
STAND_H = 0.020  # the steel stand under the case, on rubber grommets (the grommets are not drawn)
SEAT_Z = 0.045  # the seat face: the pocket floor
RIM_Z = 0.075  # the case rim (top of the walls): the pocket is 30 mm deep
POCKET_LX, POCKET_LY = 0.252, 0.185  # the seat pocket: the filter plus 6 mm each side
POCKET_X0, POCKET_X1 = CX - POCKET_LX / 2, CX + POCKET_LX / 2  # 0.234 .. 0.486
POCKET_Y0, POCKET_Y1 = CY - POCKET_LY / 2, CY + POCKET_LY / 2  # -0.0825 .. 0.1025
CASE_GREY = (0.16, 0.16, 0.18)
STAND_GREY = (0.45, 0.47, 0.50)

# the two rear slots take the cover's tabs: holes in the rear wall's top layer
TAB_Y = (CY - 0.070, CY + 0.070)  # tab centres
TAB_X = 0.497  # centre of the tab (and of the slot) in x
TAB_W, TAB_T, TAB_DROP = 0.012, 0.004, 0.012  # tab width (y), thickness (x), length below the cover's rim
SLOT_X, SLOT_Y, SLOT_DEPTH = 0.012, 0.026, 0.017  # a slot's size and depth below the rim
TAB_SLACK_X = (SLOT_X - TAB_T) / 2  # 4 mm each way
TAB_SLACK_Y = (SLOT_Y - TAB_W) / 2  # 7 mm each way

# -- the clips (two rocker levers on the front wall's brackets) ----------------------------------------------------
CLIP_Y = (CY - 0.060, CY + 0.060)  # right, left (-0.050, +0.070): 0.12 m apart
CLIP_NAMES = ("clip_r", "clip_l")
CLIP_PIV_X, CLIP_PIV_Z = 0.145, 0.096  # the pivot pin
CLIP_TAB_X = 0.105  # the thumb tab's outer end (40 mm before the pivot)
CLIP_BAR_END = 0.206  # the hook end, over the cover's lug
CLIP_BAR_W, CLIP_BAR_T = 0.012, 0.006
CLIP_TAB_PLATE = (0.105, 0.125, 0.024)  # thumb plate: x0, x1, width (y)
CLIP_Q_OPEN = 1.35  # rad: released, the hook end up 60 mm, the tab down 39 mm
CLIP_Q_CLOSED_MAX = math.radians(5.0)  # 'closed' means the lever within 5 degrees of its closed place
CLIP_Q_CLOSED_MIN = -0.05  # lower than this: no lug under the hook (it dropped to its stop)
CLIP_Q_LOWER = -0.10
CLIP_SPRING = 0.15  # N m/rad toward the detent notch (2.2 N at the thumb tab at the snap point, est)
CLIP_CAPTURE = CLIP_Q_OPEN / 2  # the notches (0, Q_OPEN) each capture half the travel: over-centre at 0.6 rad
CLIP_MASS = 0.04
CLIP_STEEL = (0.62, 0.64, 0.68)
BRACKET_Y_HALF = 0.014

# -- the cover (a sprung chain on the duct's stub) -------------------------------------------------------------------
COVER_X0, COVER_X1 = CASE_X0, CASE_X1
COVER_Y0, COVER_Y1 = CASE_Y0, CASE_Y1
COVER_SKIRT_Z = RIM_Z  # the skirt's lower edge on the case rim
COVER_ROOF_TOP = 0.115  # 40 mm deep
COVER_WALL = 0.003  # polypropylene shell, 3 mm
COVER_MASS = 0.28  # est: 0.126 m2 of 2.5 to 3 mm PP shell
LUG_X0, LUG_X1 = 0.216, 0.236  # the lift lug on the roof's front edge: 20 mm thick (along x)
LUG_Y0, LUG_Y1 = CY - 0.005, CY + 0.025  # 30 mm wide
LUG_H = 0.020  # 20 mm proud of the roof
LUG_TOP = COVER_ROOF_TOP + LUG_H  # 0.145
CLIP_LUG_X0, CLIP_LUG_X1 = 0.198, 0.213  # a catch lug on the front skirt (one per clip), 12 mm wide
CLIP_LUG_Z0, CLIP_LUG_Z1 = 0.082, 0.092
CLIP_LUG_W = 0.012
BAR_Z0 = 0.093  # the hook bar's underside when closed: 1 mm over the lug's top
STUB_BOX = ((0.590, 0.650), (0.070, 0.120), (0.070, 0.120))  # the throttle body's stub (x, y, z)
SPIGOT_Y, SPIGOT_Z, SPIGOT_R = 0.095, 0.095, 0.020
SPIGOT_X0, SPIGOT_X1 = CASE_X1, CASE_X1 + 0.040
DUCT_K = 10.0  # N/m: the duct as a spring, 0.1 N/cm (est)
DUCT_DAMP = 4.0  # N s/m
DUCT_LIM_X = (-0.04, 0.04)  # joint limits (the duct's travel)
DUCT_LIM_Y = (-0.04, 0.30)
DUCT_LIM_Z = (0.0, 0.08)
DUCT_TEAR_X = 0.035  # over this the duct is pulled off its stub (zero)
DUCT_TEAR_Y = (-0.035, 0.28)
DUCT_TEAR_Z = 0.07
COVER_PP = (0.78, 0.78, 0.80)
COVER_JOINTS = ("dx", "dy", "dz")

# -- the filter (loose): FRAM CA10171 class panel filter ---------------------------------------------------------------
F_LX, F_LY, F_H = 0.240, 0.173, 0.049  # est
F_WALL = 0.020  # the urethane frame's width (also where the gripper pinches it)
F_PLEAT_H = 0.026  # the pleat pack's top 23 mm under the frame's top (est)
F_MASS = 0.15
F_FRAME = (0.30, 0.30, 0.32)
F_OLD = (0.42, 0.36, 0.30)  # a dirty old filter
F_NEW = (0.93, 0.93, 0.90)
F_PLEATS_OLD = (0.30, 0.26, 0.22)
F_PLEATS_NEW = (0.96, 0.95, 0.88)
OLD_START = (CX, CY, SEAT_Z)  # origin = bottom centre
FILTER_FRICTION = 0.6
SEAT_TOL = 0.002  # 'flat on the seat': every bottom corner within 2 mm of the seat face
YAW_TOL = math.radians(5.0)  # long side along x, either way round (the filter is a plain rectangle)
SEAT_XY_TOL = 0.006  # the filter's centre within 6 mm of the pocket's (the pocket leaves it 6 mm of play each way)

# -- mats and the engine bay -------------------------------------------------------------------------------------------
MAT_T = 0.002
MAT_NEW = (0.40, -0.30)  # shop mat with the new filter, long side along y (clear of the right arm's start pose)
MAT_NEW_SIZE = (0.200, 0.250)
MAT_OLD = (0.33, -0.52)  # drop zone for the old filter
MAT_OLD_SIZE = (0.260, 0.190)
MAT_COLOR = (0.15, 0.30, 0.52)
NEW_START = (MAT_NEW[0], MAT_NEW[1], MAT_T)  # at yaw -90 degrees: long side along y, a quarter turn from the seat
NEW_START_YAW = -math.pi / 2
BATTERY = ((0.645, -0.260), (0.230, 0.175), 0.225)  # centre, size (x, y), height: group 35 battery (est)
BATTERY_COLOR = (0.10, 0.12, 0.14)
LIP = ((0.100, 0.0), (0.020, 0.900), 0.030)  # the radiator support's top edge: 30 mm high along y at x 0.09 to 0.11
LIP_COLOR = (0.55, 0.12, 0.10)
FRICTION = 0.5

# -- the gripper facts ------------------------------------------------------------------------------------------------
PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP
FINGER_OPEN = -0.04695

# -- thresholds (scoring.py) -----------------------------------------------------------------------------------------
OLD_REST_HIGH = 0.006  # the old filter's lowest point within 6 mm of the table (a mat is 2 mm)
OLD_TILT_DEG = 10.0
SEAT_CLEAR_X = 0.0  # the old filter's outline must not overlap the case's footprint at all
COVER_DZ_TOL = 0.002  # the cover's rim within 2 mm of the case rim all round
COVER_DX_TOL = TAB_SLACK_X  # a tab in its slot: 4 mm in x
COVER_DY_TOL = 0.005  # and 5 mm in y (the lug under the hook, a tab in its slot)
TAB_IN_MIN = 0.005  # both tabs at least 5 mm into their slots
FALL_DROP = 0.020  # a filter whose lowest point is more than 20 mm below the table top has fallen off the table
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible motion: 5 mm and 5 degrees over 1 s
STILL_TURN_DEG = 5.0
STILL_CLIP_RAD = math.radians(5.0)
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9


def outline_corners():
    """The four extreme corners of the filter's rectangle (for the seat test)."""
    hx, hy = F_LX / 2, F_LY / 2
    return [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]


# -- the demo's grasp points -------------------------------------------------------------------------------------------
GRASP_DX = -0.030  # the gripper pinches the filter's frame 30 mm toward its front (-x) from the centre
FRAME_GRASP_TIP = F_H - 0.013  # pad tips 13 mm under the frame's top
