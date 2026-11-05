"""Every dimension of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The manikin lies face up across the table, its head toward +y (the robot's left), its feet toward -y. Lying face up
with the head toward +y, the patient's RIGHT side is toward -x (toward the robot) and the patient's LEFT side is
toward +x (away from it): the two grippers work from the patient's right side, as two rescuers kneeling there would.
The torso is a flattened stand-in for an adult chest at about 0.7 scale; the pad stand-ins are smaller than real
pads (see the README).
"""

TOP = 0.7475  # station table top

# -- the manikin (fixed) ---------------------------------------------------------------------------------------
# The compression point: the centre of the chest (lower half of the breastbone), marked by a red square.
CPR_X, CPR_Y = 0.205, -0.008
MARK_THICK = 0.001  # every mark on the chest is a 1 mm thick printed square, fixed on the torso's top
CPR_MARK = 0.030  # the compression mark is 30 x 30 mm

# Torso: a flat skin-coloured block, 240 mm across the chest (along x), 300 mm from the shoulders (+y end) to the
# waist (-y end), 50 mm thick. The shoulder line is 140 mm headward of the compression point.
TORSO_SIZE = (0.240, 0.300, 0.050)
TORSO_X = CPR_X
TORSO_Y = CPR_Y + 0.140 - TORSO_SIZE[1] / 2  # -0.018: y from -0.168 (waist) to +0.132 (shoulders)
TORSO_TOP = TOP + TORSO_SIZE[2]
MARK_TOP = TORSO_TOP + MARK_THICK  # the top of every mark on the chest

# Head: a skin-coloured block beyond the shoulders, 10 mm of neck gap, its top level with the chest.
HEAD_SIZE = (0.100, 0.110, 0.050)
HEAD_X = CPR_X
HEAD_Y = TORSO_Y + TORSO_SIZE[1] / 2 + 0.010 + HEAD_SIZE[1] / 2  # 0.197: y from +0.142 to +0.252
HEAD_TOP = TOP + HEAD_SIZE[2]

# The two pad sites (anterolateral placement), each a blue 40 x 40 mm square the size of a pad:
# sternal site: below the patient's right collarbone, right of the breastbone: 75 mm toward the patient's right
#   (-x) and 110 mm toward the head (+y) of the compression point;
# apex site: the patient's left side, in line with the armpit: 95 mm toward the patient's left (+x) and 20 mm
#   toward the head. On a real chest the apex pad's centre is level with the compression point or 1-3 cm lower;
#   it is 20 mm higher here so that the left arm reaches it without crossing the midline (y 0).
SITE_MARK = 0.040
STERNAL_X, STERNAL_Y = CPR_X - 0.075, CPR_Y + 0.110  # (0.130, 0.102)
APEX_X, APEX_Y = CPR_X + 0.095, CPR_Y + 0.020  # (0.300, 0.012)
SITES = {"sternal": (STERNAL_X, STERNAL_Y), "apex": (APEX_X, APEX_Y)}

# -- the AED (fixed) -------------------------------------------------------------------------------------------
# A yellow-green box beside the patient's head on the left arm's side, with an orange shock button on its top.
AED_X, AED_Y = 0.420, 0.250
AED_SIZE = (0.100, 0.100, 0.045)
AED_TOP = TOP + AED_SIZE[2]
BUTTON_RADIUS = 0.015  # 30 mm across
BUTTON_HEIGHT = 0.006
BUTTON_TOP = AED_TOP + BUTTON_HEIGHT

# -- the electrode pads (loose) --------------------------------------------------------------------------------
# Two identical white blocks standing in for a pad with its foam backing (a real pad is about 120 x 100 mm and
# 2 mm thick; a block is needed for the gripper to hold). Lying flat on the table beside the manikin's head.
PAD_SIZE = (0.040, 0.040, 0.024)
PAD_MASS = 0.015
PAD_HALF = PAD_SIZE[0] / 2
PAD_A_X, PAD_A_Y = 0.130, 0.285  # goes to the sternal site in the demo (either pad may go on either site)
PAD_B_X, PAD_B_Y = 0.280, 0.285  # goes to the apex site in the demo
PADS = ("pad_a", "pad_b")

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open) --
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the finger pads are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------
# Compressions (right gripper only). Heights are of the point between the right gripper's finger tips above the
# compression mark's top; 'over the mark' is that point within ON_CHEST_XY of the mark's centre horizontally.
CLOSED_GAP = 0.030  # 'closed': finger gap 30 mm or less (rigstate opening <= 0.32): the two pads act as one block
ON_CHEST_XY = 0.020
ON_CHEST_HIGH = 0.045  # 'on the chest': closed, over the mark, tips at most 45 mm above it
STROKE_UP = 0.030  # a compression: the tips, closed over the mark, come down from at least 30 mm ...
STROKE_DOWN = 0.015  # ... to 15 mm or less above the mark (a dip of at least 15 mm)
CPR_GAP_S = 1.5  # compressions 'in progress': on the chest, the last compression at most 1.5 s ago
# Clear: no part of either gripper's fingers inside the box that extends CLEAR_XY beyond the manikin (torso or head)
# on every side, seen from above, and CLEAR_Z above its top.
CLEAR_XY = 0.050
CLEAR_Z = 0.050
# Pads.
ON_SITE_XY = 0.020  # 'on its site': the pad's centre within 20 mm of the site's centre (it covers the centre) ...
ON_SITE_Z = 0.003  # ... its lowest point within 3 mm of the chest's top (the torso top; the 1 mm mark counts) ...
FLAT_DEG = 10.0  # ... and tilted less than 10 degrees (a 40 x 40 mm face down)
RESTING_Z = 0.005  # 'resting on the chest' (for the compression-point rule): lowest point within 5 mm of the top
# The AED.
ANALYSIS_S = 3.0  # both pads on their sites and everyone clear for 3 s without a break: analysis complete, charged
PRESS_XY = BUTTON_RADIUS  # 'on the button': closed, tips within 15 mm of the button's centre horizontally ...
PRESS_Z = 0.006  # ... and at most 6 mm above its top
PRESS_S = 0.5  # a press that began after the AED was charged, held 0.5 s, delivers the shock
MAX_PAUSE_S = 10.0  # 1.0: before the shock, compressions never pause longer than 10 s: from the start to the first
#                     compression, between compressions, and from the last compression to the shock (pre-shock pause)
POST_SHOCK_S = 3.0  # 1.0: the first compression within 3 s after the shock ...
POST_RUN_S = 5.0  # ... and compressions kept up from it (no gap over 1.5 s) for at least 5 s
OFF_TABLE_DROP = 0.010  # a pad whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a pad moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Run limit (task.json "run_limit"): 1,200 control steps, 60 s at 20 Hz.
RUN_LIMIT_STEPS = 1200

# -- colours (sRGB) --------------------------------------------------------------------------------------------
SKIN = (0.85, 0.70, 0.58)
RED = (0.80, 0.10, 0.10)
BLUE = (0.15, 0.35, 0.85)
AED_GREEN = (0.55, 0.75, 0.20)
ORANGE = (1.00, 0.45, 0.05)
WHITE = (0.95, 0.95, 0.95)
