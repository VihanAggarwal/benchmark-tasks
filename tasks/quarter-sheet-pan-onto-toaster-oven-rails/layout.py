"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A fixed toaster oven (a small countertop oven that takes a quarter sheet pan) stands on the table with its open front
face toward the arms at x 0.425. Its cavity has two pairs of rails (ledges) on the side walls; the pan goes on the
lower pair. A loose quarter sheet pan (9-1/2 x 13 x 1 in, rolled rim) lies on the table in front of the oven. The
oven's drop-down door lies open flat in front of it (a static obstacle); heating elements and the wire rack are left
out (README, Simplifications).

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the oven (fixed) ------------------------------------------------------------------------------------------------
OVEN_X = 0.425  # the open front face of the cavity (the plane the door shuts against)
CW = 0.344  # cavity inner width (y): 6.9 mm each side of the pan (est: a small oven that takes a quarter sheet pan)
CD = 0.245  # cavity depth (x): the back wall's inner face is at OVEN_X + CD, 3.7 mm more than the pan is deep
WALL_T = 0.015  # side walls, roof and back wall
FLOOR_H = 0.020  # the floor's top (the oven stands on its feet, solid in the model)
ROOF_H = 0.190  # the cavity roof (inner face): the cavity is 0.17 m tall over its floor
ROOF_T = 0.010
RAIL_T = 0.004  # a rail (ledge) is a 4 mm plate (a formed rib on a real oven, est)
RAIL_D = 0.018  # 18 mm deep from the side wall: the pan's edge overlaps it by 11 mm each side
RAIL1_TOP = 0.075  # the lower pair's top (the pan goes here): 55 mm over the floor (est)
RAIL2_TOP = 0.140  # the upper pair's top: 65 mm higher (est)
DOOR_L = 0.170  # the drop-down door, open: hinged at the front bottom of the opening, lying flat toward the arms
DOOR_T = 0.020  # its thickness; its inner face is level with the oven floor's top (20 mm over the table)
BACK_X = OVEN_X + CD  # the back wall's inner face: the back stop
RAIL_Y = CW / 2 - RAIL_D  # 0.154: the rails' inner edges at y = +-RAIL_Y
OVEN_BLACK = (0.10, 0.10, 0.11)
RAIL_GREY = (0.42, 0.42, 0.44)

# -- the pan (loose) -------------------------------------------------------------------------------------------------
# Quarter sheet pan: 9-1/2 x 13 in, 1 in deep (the standard "quarter sheet" size; a rolled rim). Aluminium. Mass is an
# estimate. Walls are 4 mm in the model (a real pan's are about 1 mm), and the rolled rim is a solid bead.
PAN_D = round(9.5 * IN, 4)  # 241.3 mm along x (into the oven)
PAN_W = round(13.0 * IN, 4)  # 330.2 mm along y
PAN_H = round(1.0 * IN, 4)  # 25.4 mm tall
PAN_MASS = 0.20  # est (a quarter sheet pan weighs about 0.2 to 0.25 kg)
PAN_FT = 0.004  # floor thickness (model)
PAN_WT = 0.004  # wall thickness (model)
BEAD_W = 0.016  # the rolled rim as a solid bead 16 mm wide (est; the grippers close on it)
BEAD_H = 0.010
PAN_FRICTION = (0.60, 0.47)  # aluminium on steel, dry: static 0.61, sliding 0.47 (engineering tables)
ALUMINIUM = (0.78, 0.78, 0.80)
PAN_X0 = 0.008  # the pan's trailing edge (the lip nearest the arms) as found: on the table, its leading edge 5.7 mm
# short of the open door's free edge (x 0.255), so the pan starts clear of the door
PAN_Y = 0.0
PAN_CX = PAN_X0 + PAN_D / 2  # the base frame is the centre of the underside
END_GRIP = 0.040  # the hands take the two short ends (the side rims along x) 40 mm from the trailing edge; the
# nearer to the centre, the less the pan's weight turns it in the pads (30 mm: 17 degrees nose down in 4 s on the GPU)
GRIP_X = -PAN_D / 2 + END_GRIP  # that grip point in the pan frame (x)
SIDE_Y = PAN_W / 2 - BEAD_W / 2  # the side rims' bead centres, in the pan frame (y): +-0.157

# -- where the pan comes to rest -------------------------------------------------------------------------------------
HOME_X0 = BACK_X - PAN_D  # the pan's trailing edge with its leading edge on the back wall: 0.4087
HOME_CX = BACK_X - PAN_D / 2
CARRY_STOP = 0.095  # the hands let go with the trailing edge this far outside the face: the hands (TCP 40 mm in from
# that edge, the gripper body about 40 mm each side of it along x) stay 15 mm outside the face, and the pan's centre is
# 26 mm inside it, on the rails

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
LEVEL_DEG = 2.0  # level: the pan's up axis within 2 degrees of vertical
HOME_OUT = 0.0  # home: no part of the pan in front of the face (the door could shut); then its leading edge is
# within 3.7 mm of the back wall, since the cavity is only that much deeper than the pan
NEAR_OUT = 0.040  # 0.5: the trailing edge no more than 40 mm out of the face
IN_CENTRE = 0.030  # any score: the pan's centre at least 30 mm inside the face (it is carried by the rails)
RAIL_LOW = 0.003  # on the rails: no corner of the pan's underside more than 3 mm below the lower rails' top
RAIL_HIGH = 0.012  # ... and none more than 12 mm above it
RAIL_OVERLAP = 0.002  # the pan's edges pass this far over each rail's inner edge
FALL_DROP = 0.020  # latched: the pan's lowest point lower than 20 mm below the table top (off the table)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # still: under 5 mm and 5 degrees over the last 1 s (a real trial: no visible motion in video)
STILL_TURN_DEG = 5.0
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
