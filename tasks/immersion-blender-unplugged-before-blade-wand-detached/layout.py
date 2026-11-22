"""Every number of the scene, in metres, kg, N, N m and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP.
"Table frame" in comments is the world with z measured from the table top. The blending wand's frame has its origin
at the coupling seam on the blender's centerline, axes aligned with world at spawn (the wand extends along -y).
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0
HOLD_STEPS = 20  # 1.0 s at 20 Hz: the end state must hold this long

# -- colours (sRGB 0..1) ------------------------------------------------------------------------------------------
MOTOR_DARK = (0.14, 0.15, 0.17)  # Cuisinart CSB-175 brushed dark housing / grip
MOTOR_SILVER = (0.76, 0.77, 0.79)  # stainless accents and wand shaft
STEEL_DARK = (0.45, 0.47, 0.50)  # coupling collar and recessed blunt blade hub
BUTTON_RED = (0.82, 0.18, 0.16)  # release button cap (high contrast on dark housing)
CRADLE_WOOD = (0.62, 0.48, 0.34)  # countertop motor cradle
OUTLET_WHITE = (0.93, 0.93, 0.91)  # unpowered benchtop outlet block
PLUG_BLACK = (0.09, 0.09, 0.10)  # molded NEMA 1-15P plug body
BRASS_BLADE = (0.80, 0.66, 0.30)  # plug prongs
TRAY_BLUE = (0.18, 0.44, 0.76)  # blue sink wash tray

# -- immersion blender (Cuisinart Smart Stick CSB-175): fixed motor in cradle + loose blending wand ---------------
BLENDER_X = 0.260  # blender centerline x (table frame)
SEAM_Y = 0.040  # coupling seam y between motor handle (+y) and blending wand (-y)
AXIS_Z = 0.050  # blender centerline height above table top

MOTOR_LEN = 0.180  # 18.0 cm motor handle, y in [SEAM_Y, SEAM_Y + MOTOR_LEN]
MOTOR_R = 0.026  # 5.2 cm diameter motor body
MOTOR_TOP_Z = AXIS_Z + MOTOR_R  # 0.076 m

# Low countertop cradle supporting the motor handle (part of fixed motor assembly)
CRADLE_Y = (SEAM_Y + 0.015, SEAM_Y + MOTOR_LEN)  # y in [0.055, 0.220]
CRADLE_HALF_X = 0.032  # 64 mm outer width across x
CRADLE_BASE_H = AXIS_Z - MOTOR_R  # 0.024 m under the motor body
CRADLE_WALL_H = AXIS_Z  # side walls cradle up to the centerline

# Spring-loaded wand release button on top of the motor handle (slider along -Z)
BUTTON_X = BLENDER_X
BUTTON_Y = SEAM_Y + 0.095  # 0.135 m (mid of motor handle, 200 mm from right hand wand grip)
BUTTON_SIZE = (0.018, 0.018, 0.010)  # x, y, z cap dimensions
BUTTON_TOP_Z = MOTOR_TOP_Z + BUTTON_SIZE[2]  # 0.086 m at rest; 0.080 m at full 6 mm stop
BUTTON_TRAVEL = 0.006  # 6 mm stroke
BUTTON_SPRING = 3.5 / BUTTON_TRAVEL  # 583.33 N/m -> 3.5 N at full 6 mm stroke
BUTTON_DAMPING = 2.0
BUTTON_RELEASE_AT = 0.004  # 4 mm press releases the wand lock
BUTTON_PREMATURE_AT = 0.002  # 2 mm press while plugged in triggers hard 0

# Stainless blending wand (loose assembly, origin at coupling seam (BLENDER_X, SEAM_Y, TOP + AXIS_Z))
WAND_LEN = 0.176  # 17.6 cm total length, local y in [-0.176, 0.0]
COLLAR_LEN = 0.025  # coupling collar: local y in [-0.025, 0.0]
COLLAR_R = 0.022  # 44 mm diameter
SHAFT_Y = (-0.130, -0.025)  # stainless shaft: 105 mm long
SHAFT_R = 0.009  # 18 mm diameter (pinched by right gripper)
BELL_Y = (-0.176, -0.130)  # bell guard: 46 mm long
BELL_R = 0.032  # 64 mm diameter (octagonal shroud so it rests flat without rolling)
WAND_MASS = 0.185  # 185 g
WAND_FRICTION = (1.6, 1.4)
WAND_GRIP_LOCAL_Y = -0.100  # grip point on the shaft (world y -0.060), 16 mm short of the bell guard; keeps the right wrist clear of the left
WAND_SEATED_ORIGIN = np.array([BLENDER_X, SEAM_Y, TOP + AXIS_Z], dtype=float)
# Axial coupling: an 18 mm long, 27 mm diameter spigot on the wand (local y in [0, SPIGOT_LEN]) sits in a 12-sided
# socket in the front of the motor handle (inscribed radius 15 mm, 1.5 mm radial play, 18.5 mm deep). With the latch
# released the wand can only leave the socket by being pulled straight out along its axis (-y).
SPIGOT_LEN = 0.018  # 18 mm engagement length
SPIGOT_R = 0.0135  # 27 mm diameter spigot
SOCKET_APOTHEM = 0.015  # socket inscribed radius (1.5 mm radial play)
SOCKET_DEPTH = 0.0185  # socket bottom 0.5 mm past the seated spigot tip
SOCKET_SIDES = 12
WAND_PULL_FORCE = 3.0  # N, real pull-off force with the release held (est; check with a spring scale)
WAND_CLEAR_LATCH = 0.005  # once withdrawn >= 5 mm along the axis, button release no longer re-engages the latch
WAND_PREMATURE_DISP = 0.005  # >= 5 mm wand motion while plugged in triggers hard 0
WAND_SEPARATED_AT = SPIGOT_LEN  # axial withdrawal >= the 18 mm engagement (spigot tip out of the socket mouth)
SOCKET_AXIS_TOL = 0.006  # spigot tip within 6 mm of the socket axis when it leaves = pulled out along the axis

# -- unpowered benchtop outlet block & NEMA 1-15P plug (fixed bench + plug slider along +Y) -----------------------
OUTLET_C = np.array([0.340, 0.205], dtype=float)  # block centre (x 34 cm, y 20.5 cm)
OUTLET_SIZE = (0.040, 0.070, 0.052)  # x, y, z
OUTLET_FACE_Y = float(OUTLET_C[1] + OUTLET_SIZE[1] / 2)  # 0.240 m (+y face)
PLUG_SIZE = (0.026, 0.028, 0.030)  # x (pinched width), y (pull depth), z (height)
PLUG_Z = 0.025  # plug body centre height above table top
PLUG_FRICTION = 4.0  # N: NEMA 1-15P receptacle retention force
PLUG_DAMPING = 5.0
PLUG_UPPER = 0.035  # 35 mm slider travel along +Y
BLADE_LEN = 0.018  # 18 mm (1.8 cm) NEMA 1-15P blade length
UNPLUGGED_AT = BLADE_LEN  # plug must be pulled >= 1.8 cm free of the outlet block
CORD_R = 0.0035  # 7 mm diameter power cord

# -- blue sink wash tray (part of fixed bench assembly) -----------------------------------------------------------
TRAY_C = np.array([0.260, -0.260], dtype=float)  # centre at x 26 cm, y -26 cm (19 mm clear of the pulled wand)
TRAY_OUTER = (0.240, 0.140, 0.040)  # 24 x 14 x 4 cm (long axis along x)
TRAY_WALL = 0.006
TRAY_FLOOR = 0.006
TRAY_INNER_X = (float(TRAY_C[0] - TRAY_OUTER[0] / 2 + TRAY_WALL),
                float(TRAY_C[0] + TRAY_OUTER[0] / 2 - TRAY_WALL))
TRAY_INNER_Y = (float(TRAY_C[1] - TRAY_OUTER[1] / 2 + TRAY_WALL),
                float(TRAY_C[1] + TRAY_OUTER[1] / 2 - TRAY_WALL))

# -- arms & gripper geometry --------------------------------------------------------------------------------------
FULL_GAP = 0.0939
PAD_TIP = 0.013
PAD_HALF_W = 0.0135
OPEN_GAP = 0.060
PLUG_GAP = 0.017  # pinch 26 mm plug across x
WAND_GAP = 0.002  # pinch 18 mm stainless shaft firmly (~40 N finger clamp)

# -- scoring thresholds -------------------------------------------------------------------------------------------
LIVE_BELL_GAP = 0.010  # while plugged in, gripper finger zone within 10 mm of the blade-bearing bell triggers hard 0
TRAY_FLAT_TILT = math.radians(15.0)  # wand axis within 15 deg of horizontal
TRAY_FLOOR_TOL = 0.012  # wand lowest point within 12 mm of tray floor top
HANDS_AWAY = 0.040  # both grippers >= 4.0 cm clear of wand during 1.0 s hold
STILL = 0.005  # wand origin moves < 5 mm over the 1.0 s hold window
FELL_Z = -0.020  # wand fell off bench
