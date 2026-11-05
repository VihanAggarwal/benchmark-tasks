"""Every dimension of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say TOP.

The manikin is one fixed mechkit assembly: its base (torso shell, neck, head and the bag's post) never moves, and
it carries two spring-return sliders that move straight down: the chest plate and the bag's cap. Its base frame
sits at ORIGIN; the shapes are declared in that frame, so world x, y = ORIGIN x, y + local x, y.
"""

TOP = 0.7475  # station table top
ORIGIN = (0.300, -0.140, TOP)

# The torso: a hollow shell standing on the table, 170 mm along x, 300 mm along y, 80 mm tall, 8 mm walls and an
# 8 mm top deck with a 120 x 120 mm square opening over the chest. World x 0.215..0.385, y -0.290..+0.010. The
# opening is wide enough for the closed gripper to follow the plate 65 mm down without meeting the deck.
TORSO_SIZE = (0.170, 0.300, 0.080)
TORSO_Y = -0.140  # its centre (world)
WALL = 0.008
DECK = 0.008
OPENING = 0.120
# The neck (a short block) and the head, fixed, on the robot's left of the torso.
NECK_SIZE = (0.050, 0.010, 0.050)
NECK_Y = 0.015  # its centre (world): y 0.010..0.020, from the torso's end (y 0.010) to the head
HEAD_SIZE = (0.100, 0.090, 0.070)
HEAD_Y = 0.065  # its centre (world): y 0.020..0.110

# The chest: a plate filling the deck's opening (2.5 mm clear all round) on a vertical spring slider, its top level
# with the deck as drawn, with a dark mark (a disc 30 mm across, 1 mm thick) at its centre. Positive travel is
# downward (compression).
MARK_X, MARK_Y = 0.300, -0.170
PLATE_SIZE = (0.115, 0.115, 0.012)
PLATE_TOP = 0.080
MARK_RADIUS = 0.015
MARK_THICK = 0.001
MARK_TOP = PLATE_TOP + MARK_THICK  # 0.081: where pressing finger tips meet it as drawn
CHEST_TRAVEL = 0.065  # its stop: the plate's underside then 3 mm above the table
CHEST_SPRING = 75.0  # N/m (4.1 N at 55 mm)
CHEST_DAMPING = 1.0  # N s/m
PLATE_MASS = 0.015  # a hollow plastic plate

# The bag (it stands in for a bag-valve-mask's bag): a fixed dark post with a blue cap over it, a hollow sleeve
# (an 8 mm top plate and 4 mm side walls) on a vertical spring slider; pressing its top pushes it down over the post.
BAG_X, BAG_Y = 0.300, 0.220
POST_SIZE = (0.040, 0.040, 0.050)
CAP_SIZE = (0.060, 0.060, 0.060)  # outside
CAP_TOP = 0.110
CAP_PLATE = 0.008
CAP_WALL = 0.004
BAG_TRAVEL = 0.048  # its stop: the sleeve's lower edge then 2 mm above the table
BAG_SPRING = 40.0  # N/m
BAG_DAMPING = 0.5  # N s/m
CAP_MASS = 0.015  # a thin plastic shell

GRAVITY = 9.81
# Under its own weight each slider rests a little below its drawn pose (spring and gravity balance): about 2.0 mm
# for the chest, 3.7 mm for the cap. The scorer reads the joints as they are (0 = as drawn).
CHEST_REST = PLATE_MASS * GRAVITY / CHEST_SPRING
BAG_REST = CAP_MASS * GRAVITY / BAG_SPRING

# -- scoring (scoring.py); depths in millimetres, times in seconds ----------------------------------------------
COMP_START_MM = 20.0  # a compression begins when the chest's depth rises through this
COMP_END_MM = 10.0  # ... and ends when it falls below this
DEPTH_MIN_MM, DEPTH_MAX_MM = 50.0, 60.0  # a good compression's peak depth
RECOIL_MAX_MM = 5.0  # a good compression follows a recoil (the lowest depth since the previous one) within this
MARK_TOL = 0.025  # m: some gripper's tool point within this of the mark (horizontally) at the peak ...
PRESS_ABOVE = 0.040  # m: ... and no more than this above the mark's top where it is then (the gripper pressing it,
#                      13 mm above it when its closed finger tips are on the mark; not one hovering over the plate
#                      while the other presses it somewhere else)
INTERVAL_MIN_S, INTERVAL_MAX_S = 0.43, 0.70  # from the previous peak in the same cycle (the first is exempt)
CYCLE_GAP_S = 2.0  # peaks closer than this belong to one cycle
BREATH_PEAK_MM = 25.0  # a breath: the cap pressed past this ...
BREATH_ON_MM = 15.0  # ... and held past this for BREATH_MIN_S .. BREATH_MAX_S
BREATH_MIN_S, BREATH_MAX_S = 0.6, 2.0
BAG_TOL = 0.060  # m: it counts only if, at its deepest, a gripper is pressing the cap: a tool point within this of
#                  the cap's centre horizontally (the cap is 60 mm across) and no more than PRESS_ABOVE above its top
#                  where it then is (not an arm's link pushing it while the same arm's gripper waits over the mark),
BREATH_CHEST_MAX_MM = 10.0  # ... the chest stays within this throughout,
HAND_ABOVE_MARK = 0.060  # m: ... and some gripper waits over the mark (tool point within MARK_TOL of it
#                          horizontally, no more than this above its top as drawn) throughout
ZERO_CHEST_MM, ZERO_BAG_MM = 20.0, 15.0  # 0: at any step the chest past 20 mm while the cap is past 15 mm
MIN_GOOD = 20  # 0: fewer good compressions than this
FULL_COUNT = (28, 32)  # 1.0: compressions in each of the two cycles
FULL_RATE = (100.0, 120.0)  # 1.0: each cycle's rate, per minute
FULL_GOOD = 0.90
FULL_BREATHS = 2  # 1.0: exactly this many counted breaths after each cycle
FULL_PAUSE_S = 10.0
PART_COUNT = (25, 35)  # 0.75 and 0.5
PART_GOOD = 0.80  # 0.75
PART_BREATHS = 3  # 0.75: counted breaths in the first two cycles' pauses together
HALF_GOOD = 0.60  # 0.5
HALF_PAUSE_S = 15.0
CLEAR = 0.030  # m: 1.0 ends with both grippers' tool point and finger tips this far from the plate and the cap
STILL_MM_S = 5.0  # 'still': each slider moving slower than this
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
# Reset fallback (rigstate's reset hook normally comes first): a gripper jumps this far in one control step to
# within READY_TOL of its ready pose, open, with both sliders at rest.
RESET_JUMP = 0.12
READY_TCP = {"left": (0.3234, 0.305, TOP + 0.2606), "right": (0.3234, -0.305, TOP + 0.2606)}
READY_TOL = 0.010
REST_TOL_MM = 6.0

# -- the offline stand-in (no physics offline: tools/offline/check_task.py) ---------------------------------------
# The arm's vertical give at the finger tips at both presses, from J K^-1 J^T of the YAM's joint gains (80 N m/rad
# shoulder and elbow, 10 N m/rad wrist): about 1.9 mm per newton, so a slider and the arm act as two springs in series.
ARM_STIFFNESS = 530.0  # N/m

# Colours (sRGB).
SKIN = (0.85, 0.70, 0.58)
MARK_COLOR = (0.40, 0.16, 0.12)  # dark red-brown
POST_GREY = (0.25, 0.25, 0.27)
CAP_BLUE = (0.20, 0.40, 0.85)
