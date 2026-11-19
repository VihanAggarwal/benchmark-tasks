"""Every number of the scene, in one place. NumPy only, no simulator.

World frame (yam_arena's): metres, x forward from the line through the two arm bases, y to the robot's left, z up;
``TOP`` is the station table's top. In the comments sizes are millimetres and heights are over the table top.

The scene, real size: a changing mat on the table; an infant care manikin (newborn size, 52 cm; "the doll" in the
code) on its back on the mat, head to the robot's left, its right arm on the robot's side lying along its body; a
digital stick thermometer (Vicks SpeedRead V912US class, switched off, display up) on the mat on the robot's side of
the doll, beside its right hand. Sizes marked (est) are estimates to be measured on the real items (README,
"Measure before a real trial").
"""

from __future__ import annotations

import math

MM = 0.001
TOP = 0.7475  # the station table's top (world z)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- changing mat (loose): mid-blue PVC over foam --------------------------------------------------------------------
MAT_X = (0.060, 0.520)  # 460 mm (est size)
MAT_Y = (-0.300, 0.300)  # 600 mm
MAT_H = 0.015  # its top 15 mm over the table
MAT_MASS = 0.60  # (est)
MAT_FRICTION = 0.8  # PVC on the table (est)

# -- the doll: one rigid body (base) plus its right arm on the shoulder joint ----------------------------------------
# Base frame: origin on the mat top under the doll's midline (world x 300, y 0); all doll shapes below are given in
# WORLD millimetres (x, y) and height over the MAT TOP (z), converted in parts.py.
DOLL_X = 0.300
DOLL_MASS = 2.20  # the 2.3 kg doll minus its right arm (listed 5 lb)
ARM_MASS = 0.10  # the right arm (est)
DOLL_FRICTION = 0.5  # vinyl (est; doll on mat 0.5)
TORSO_X = (238.0, 362.0)  # 124 mm wide (est)
TORSO_Y = (30.0, 145.0)
TORSO_DEPTH = 85.0  # chest 85 mm over the mat (est)
TORSO_SIDE_H = 45.0  # the flat side of the chest under the arm, mat to 45 mm; rounded above
# Newborn proportions (an infant care manikin, not a toy): the head about a quarter of the body's length (120 of
# 522 mm), rounded limbs (cylinders with rounded ends), short chubby legs, rounded hands and feet, no drawn face.
HEAD_C = (300.0, 224.0)  # head centre (top view); the crown at y 284
HEAD_R = 60.0
NECK_Y = (140.0, 165.0)
DIAPER_Y = (-15.0, 45.0)
LEG_X = (276.0, 324.0)
LEG_R = 22.0  # the thigh (its near side x 254, as before: the left gripper's open fingers pass beside it)
SHIN_R = 19.0
KNEE_Y = -120.0
LEG_Y = (-205.0, -15.0)  # hip to ankle
FOOT_C_Y = -221.0  # the foot: a round heel-to-toe block, toes up; the soles at y -238
FOOT_R = 17.0
FOOT_H = 44.0

# Shoulder joint (the doll's disc joint): about the doll's left-right axis (world x), at world y 135, 20.5 mm over the
# mat; the arm lifts up off its side toward the ceiling (0 deg = lying along the body, 90 deg = pointing straight up).
SHOULDER_Y = 135.0
SHOULDER_Z = 20.5
SHOULDER_RANGE_DEG = 180.0
SHOULDER_FRICTION = 0.02  # N m (est; the arm sinks back from any lift under about 70 deg)
SHOULDER_DAMPING = 0.005  # N m s/rad (est)
UPPER_ARM_X = 216.5  # its axis; radius 17.5: x 199 to 234, its inner face 3.5 mm from the sensor film (x 237.5)
UPPER_ARM_R = 17.5
UPPER_ARM_Y = (70.0, 140.0)
SHOULDER_DISC_X = (199.0, 238.0)
ELBOW = (216.5, 72.0)
FOREARM_R = 15.0
FOREARM_LEN = 84.0
FOREARM_YAW_DEG = -8.0  # elbow slightly bent: the forearm angles 8 deg out from the body toward the hand
HAND_R = 13.0
HAND_LEN = 26.0
ARM_GAP_UNDER = 3.0  # the arm's underside 3 mm over the mat

# Armpit sensor: Interlink FSR 402 thin-film force sensor, taped flat on the torso side under the shoulder.
FSR_R = 7.34  # 14.68 mm active disc (listed)
FSR_T = 0.46  # 0.46 mm thick (listed)
# The disc's centre on the torso side: world x 238 (the film's back), y 90 (45 mm toward the feet from the shoulder
# axis), 10 mm over the mat.
FSR_C = (238.0, 90.0, 10.0)

# -- the thermometer (loose): base frame at its tail end, on the underside's centre line, x toward the tip ---------
# Millimetres in that frame. White plastic, about 114 mm long and 13 mm thick (listed); 22 mm wide (est).
TH_LEN = 114.0
TH_W = 22.0
TH_H = 13.0
TH_LOWER_H = 9.0  # the body under the top plate
TH_BODY_END = 84.0  # the front 30 mm tapers to the tip
BUTTON_X = 28.0  # On/Off button, 10 mm across, 28 mm from the tail (est)
BUTTON_R = 5.0
BUTTON_PROUD = 1.0  # its top 1 mm over the body (est)
BUTTON_TRAVEL = 0.0008  # 0.8 mm (est)
BUTTON_K = 2500.0  # N/m: 2 N at full travel (est)
BUTTON_DAMP = 1.0
BUTTON_MASS = 0.0015
WINDOW_X = (36.0, 58.0)  # display window 22 x 10 mm (est)
WINDOW_W = 10.0
DISPLAY_TRAVEL = 0.022  # the display's lit face slides under the window when the unit is on (a drawn stand-in)
DISPLAY_K = 2.0  # N/m: soft, so its push never moves the 15 g stick on the mat
DISPLAY_DAMP = 0.05
DISPLAY_MASS = 0.0005
TIP_R = 2.0  # 4 mm metal tip (est)
TIP_X = (106.0, 114.0)
TH_AXIS_Z = 6.5  # the tip's axis over the underside
TH_MASS = 0.015 - BUTTON_MASS - DISPLAY_MASS  # 15 g in all (est)
TH_FRICTION = 0.4  # plastic on the mat (est; 0.5 on vinyl)
TH_START = (100.0, -140.0)  # tail end (world x, y) at the start; tip toward +x, display up

# The thermometer in the armpit (the demo's target): along x at the sensor's y, tip end on the film.
PLACE_Y = FSR_C[1]
PLACE_TAIL_X = FSR_C[0] - FSR_T - TH_LEN  # 123.54

# -- the device's rules (manual, V912 series; declared, not simulated) --------------------------------------------------
PRESS_M = 0.0003  # a press: the button in 0.3 mm or more
HOLD_SETTING_S = 5.0  # held 5 s: the C/F setting opens instead of switching
READY_S = 2.0  # self-test after switching on
PEAK_S = 11.0  # underarm: "usually 11 seconds"
AUTO_OFF_S = 90.0  # it turns itself off after about 1.5 min

# -- scoring ------------------------------------------------------------------------------------------------------------
TOUCH_M = 0.0005  # "touching": surfaces 0.5 mm or less apart (the engine's rest offset is 0, so resting surfaces
# read 0 to 0.2 mm: GPU 19 Nov, tip 0.02 to 0.11 mm and arm 0.00 mm through the whole count)
CLOSED_MAX_DEG = 22.0  # the arm down on the thermometer (it rests about 13 to 16 deg up on it)
ARM_BACK_DEG = 5.0
SHOWN_TIP_M = 0.030  # out of the armpit: the tip 30 mm or more from the sensor disc
SHOWN_ARM_M = 0.010  # and no part within 10 mm of the doll's arm
SHOWN_UP_DEG = 30.0
SHOWN_S = 2.0
ON_MAT_M = 0.004  # lying on the mat: its lowest point within 4 mm of the mat top
OFF_MAT_M = 0.005  # fell off the mat: a part 5 mm or more below the mat top
RELEASE_M = 0.030  # hands off: every finger zone 30 mm or more from doll and thermometer
STILL_M = 0.005  # over the last 1 s: moved 5 mm or less
DOLL_LIFT_M = 0.030  # unsafe handling: the doll's back lifted more than 30 mm off the mat top
DOLL_ROLL_DEG = 45.0  # or rolled (tipped) more than 45 deg from lying on its back, or off the mat
WIN_STEPS = int(round(1.0 * CONTROL_HZ))
EPS = 1e-9

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
SKIN = (0.55, 0.37, 0.26)  # medium brown vinyl
DIAPER = (0.95, 0.95, 0.93)
MAT_BLUE = (0.20, 0.42, 0.75)
FSR_TAN = (0.80, 0.66, 0.36)  # the sensor's polyester disc
TH_WHITE = (0.96, 0.96, 0.95)
TH_BUTTON = (0.55, 0.62, 0.72)
TIP_METAL = (0.78, 0.79, 0.80)
LCD_OFF = (0.30, 0.34, 0.33)  # blank display
LCD_LIT = (0.55, 0.88, 0.45)  # green backlight
LCD_INK = (0.05, 0.07, 0.06)


def deg(rad: float) -> float:
    return math.degrees(rad)
