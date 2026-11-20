"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named otherwise.

A low media-style cabinet (laminated chipboard, 16 mm) stands on the table with its open front at x 0.320. Under its
top panel is a double aluminium top track (two grooves, 38 mm wide, 40 mm deep); on its floor (40 mm up) are the two
matching bottom rails (6 mm wide, 6 mm high). The rear door is already hung in the rear groove and rail. The loose front
door (identical, but with a bar pull on its front face) stands on the table in front of the cabinet. The job: lean its
top toward the cabinet, lift it up into the front groove, swing its foot in over the floor, lower it onto the rail,
let go. Figures are sourced (README) or estimates (est, said in task.json's fidelity).
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the cabinet (fixed) -------------------------------------------------------------------------------------------
CAB_X0 = 0.360  # the open front (top panel, floor and track all end here): clear of the arms' ready-pose fingertips
CAB_DEPTH = 0.318
CAB_X1 = CAB_X0 + CAB_DEPTH  # the back panel's rear face
BOARD = 0.016  # laminated chipboard (sides, back); est, a common furniture board
OPEN_HALF = 0.310  # the opening is 620 mm wide: y -310 to +310
SIDE_Y0 = OPEN_HALF  # the side panels' inner faces
FLOOR_TOP = 0.040  # the floor's top (a plinth and the floor board): 40 mm above the table top
TRACK_BOTTOM = 0.310  # the underside of the top track: 270 mm clear above the floor
GROOVE_DEPTH = 0.040
ROOF = TRACK_BOTTOM + GROOVE_DEPTH  # 0.350: the grooves' roof is the underside of the top board
TOP_PANEL_TOP = ROOF + 0.018  # the 18 mm top board
CLEAR_H = TRACK_BOTTOM - FLOOR_TOP  # 270 mm

GROOVE_W = 0.038  # each groove: 38 mm for a 22 mm door (8 mm play each side, est)
X_F = CAB_X0 + 0.027  # the front track's centre line (groove and rail): 0.387
X_R = X_F + 0.046  # the rear track's centre line: 46 mm behind
WALL = 0.008  # the track's walls
FRONT_GROOVE = (X_F - GROOVE_W / 2, X_F + GROOVE_W / 2)  # 0.368 .. 0.406
REAR_GROOVE = (X_R - GROOVE_W / 2, X_R + GROOVE_W / 2)  # 0.414 .. 0.452
TRACK_X0 = FRONT_GROOVE[0] - WALL  # 0.360
TRACK_X1 = REAR_GROOVE[1] + WALL  # 0.460
DIVIDER = (FRONT_GROOVE[1], REAR_GROOVE[0])  # 0.406 .. 0.414

RAIL_W = 0.006
RAIL_H = 0.006
RAIL_TOP = FLOOR_TOP + RAIL_H  # 0.046

# -- the doors (loose) ---------------------------------------------------------------------------------------------
# Honeycomb-core laminated panel, 22 mm (est), 320 x 290 mm, 0.29 kg (about 3.1 kg/m2, est). The foot has a groove
# over the rail: 14 mm wide, 8 mm deep (4 mm strips each side).
DOOR_T = 0.022
DOOR_W = 0.320  # along y
DOOR_H = 0.290
DOOR_MASS = 0.29
FOOT_GROOVE_W = 0.014
FOOT_GROOVE_D = 0.008
STRIP = (DOOR_T - FOOT_GROOVE_W) / 2  # 4 mm
# base frame: x thickness (0 at the centre plane), y along the door (0 at its middle), z up from the underside.

# The bar pull on the front door's front face: a 20 x 20 mm aluminium square tube, 140 mm long (along y), on two
# 12 mm square posts that stand it 30 mm off the face. Its centre is 170 mm above the foot.
PULL_SIZE = (0.020, 0.140, 0.020)  # along x, along y, along z
PULL_OFF = 0.030  # the gap between the door's face and the bar
PULL_H = 0.170  # the bar's centre above the foot
PULL_X = -(DOOR_T / 2 + PULL_OFF + PULL_SIZE[0] / 2)  # -0.051 from the door's centre plane (toward the robot)
POST = (PULL_OFF, 0.012, 0.012)
POST_Y = 0.050

# Where the doors are.
REAR_X, REAR_Y = X_R, -0.140  # hung: y -300 to +20 mm
FRONT_HUNG_Y = 0.140  # y -20 to +300 mm
FRONT_START_X = X_F - 0.050  # centre plane of the front door as it stands on the table: 0.337
FRONT_START_Y = 0.020  # y -140 to +180 mm: clear of the left arm's open fingers at the ready pose (to y 190)

# Heights of the lifted door (underside) for the geometry of the hang.
FOOT_RAISED = ROOF - DOOR_H  # 0.060: the top touching the roof: the foot is 14 mm over the rail top
FOOT_HUNG = FLOOR_TOP  # 0.040: the foot on the floor, the rail in its groove
TOP_HUNG = FLOOR_TOP + DOOR_H  # 0.330: 20 mm into the groove

# -- the rules (scoring.py) ----------------------------------------------------------------------------------------
IN_GROOVE = 0.006  # 'top inside the groove': the top edge at least 6 mm above the groove's mouth (z 0.316)
FOOT_DOWN = 0.003  # 'foot down': every point of the underside within 3 mm of the floor (z 0.043 or lower)
FOOT_RAIL_TOP = 0.009  # 'foot on the rail top': the underside within 9 mm of the floor (the rail top is 6 mm up)
FACE_TOL = 0.007  # the door's centre plane at mid-height (11 mm behind its face) within 7 mm of the track's centre line
RAIL_IN = 0.005  # 'rail in the foot groove': both ends of the foot groove's centre line within 5 mm of the rail's
#                  centre line (the strips stop it at 4 mm; a foot standing beside the rail is 14 mm or more off)
FACING_COS = 0.5  # 'pull facing out': the door's outward normal within 60 degrees of facing the robot (-x)
TILT_FULL = 3.0  # degrees
TILT_PART = 8.0
REAR_Y_TOL = 0.010  # the rear door moved no more than 10 mm along y
REAR_X_TOL = 0.005
REAR_TILT_KEPT = 4.0  # degrees: the rear door's own play in its track is about 2 degrees
REAR_TILT_FELL = 10.0  # ... over this at any moment: knocked out of its track
FALL_Z = 0.030  # a door's lowest point this low (z 30 mm) while it should be in the cabinet: it fell out
LET_GO = 0.030  # 'hands off': every finger zone 30 mm or more from the front door
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visible motion: a door moves under 5 mm over the window
STILL_TURN_DEG = 5.0
RESET_JUMP = 0.08
START_TOL = 0.010
EPS = 1e-9

# Colours (sRGB).
CAB_WOOD = (0.62, 0.50, 0.36)
ALUMINIUM = (0.78, 0.78, 0.80)
DOOR_WHITE = (0.90, 0.88, 0.82)
PULL_STEEL = (0.55, 0.57, 0.60)
FRICTION = 0.4  # laminate on chipboard and aluminium (est)
