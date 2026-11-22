"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A five-position analyser sample rack (positions in a row along y) is held in place on the midline 30 cm in front of
the arm bases. Its wall nearest the arms has one reader slot per position: the analyser's barcode reader looks
through it, along +x, at the tube in that position. A tube is readable when its label faces the slot, i.e. points
along -x (toward the arms). The positions are 28 mm apart: neighbouring 16 mm tubes stand 12 mm apart, so a gripper
setting a tube down next to one already standing has room for its 8 mm pads only front and back (fingers across
the rack, along x), not along the row. Two bench tube racks, one in front of each arm, hold the four tubes at the
start, every label facing sideways.

Revised 2026-11-19 (from 30 mm positions at 36 mm pitch): 24 mm positions at 28 mm pitch, 4 mm dividers, the label
1 mm proud instead of 2, and the release rule (RELEASE_CLEAR). Earlier GPU results are for the old geometry.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the analyser rack: fixed walls on the table, the table is the floor of every position ---------------------------
RACK_X = 0.300  # centre line of the five positions (x)
POS_IN = 0.024  # each position is 24 x 24 mm inside (a 16 mm tube has 4 mm of play each way)
WALL_T = 0.006  # outer walls 6 mm thick
DIV_T = 0.004  # dividers between positions 4 mm thick
PITCH = POS_IN + DIV_T  # 28 mm centre to centre: neighbouring 16 mm tubes stand 12 mm apart
N_POS = 5
POS_Y = tuple(PITCH * (2 - k) for k in range(N_POS))  # P1 (robot's left, +y) .. P5: +0.056 .. -0.056
RACK_H = 0.040  # walls 40 mm tall
SLOT_W = 0.018  # each reader slot: 18 mm wide, centred on its position,
SLOT_Z0 = 0.006  # from 6 mm above the table to the wall's top (40 mm)
NEAR_X0, NEAR_X1 = RACK_X - POS_IN / 2 - WALL_T, RACK_X - POS_IN / 2  # 0.282 .. 0.288: the wall with the slots
FAR_X0, FAR_X1 = RACK_X + POS_IN / 2, RACK_X + POS_IN / 2 + WALL_T  # 0.312 .. 0.318
RACK_HALF_Y = POS_Y[0] + POS_IN / 2 + WALL_T  # 0.074: the rack is 148 mm long (y) and 36 mm deep (x)

# -- the two bench tube racks ('holding racks'): two 30 x 30 mm cells each, along x, 20 mm walls -------------------------------------------
HOLD_Y = 0.200  # the left rack's cells on y +0.200, the right rack's on y -0.200
HOLD_X = (0.200, 0.260)  # the two cells' centres (x)
HOLD_IN = 0.030
HOLD_WALL = 0.006
HOLD_H = 0.020
HOLD_X0 = HOLD_X[0] - HOLD_IN / 2 - HOLD_WALL  # 0.179
HOLD_X1 = HOLD_X[1] + HOLD_IN / 2 + HOLD_WALL  # 0.281

# -- the sample tubes: one loose rigid body each (mechkit compound). Base frame at the centre of the tube's bottom,
#    local +z up its axis, local +x out through the middle of the barcode label -------------------------------------
TUBE_R = 0.008  # a 16 x 100 mm tube: 16 mm across,
BODY_H = 0.082  # the tube body 82 mm tall below the cap,
CAP_R = 0.00825  # a 16.5 mm lavender closure,
CAP_H = 0.018  # 18 mm tall: 100 mm overall
TUBE_H = BODY_H + CAP_H
TUBE_MASS = 0.020  # our estimate: empty tube and closure about 9 g, plus about 11 g of liquid (a 10 mL draw)
LABEL_W = 0.010  # the label: 10 mm wide (across the tube),
LABEL_Z0, LABEL_Z1 = 0.010, 0.050  # from 10 to 50 mm up the body,
LABEL_OUT = 0.009  # its face 9 mm from the axis (1 mm proud of the body, so it shows from above); its outer corners
#                    are 10.3 mm from the axis, inside a position's 12 mm half-width, so a tube turns freely in one
LABEL_IN = math.sqrt(TUBE_R ** 2 - (LABEL_W / 2) ** 2)  # 6.2 mm: its inner face meets the body at its edges
BODY_CENTER = (0.0, 0.0, BODY_H / 2)
CAP_CENTER = (0.0, 0.0, BODY_H + CAP_H / 2)
LABEL_CENTER = ((LABEL_IN + LABEL_OUT) / 2, 0.0, (LABEL_Z0 + LABEL_Z1) / 2)
LABEL_SIZE = (LABEL_OUT - LABEL_IN, LABEL_W, LABEL_Z1 - LABEL_Z0)
LABEL_POINT = (LABEL_OUT, 0.0, (LABEL_Z0 + LABEL_Z1) / 2)  # the middle of the label's face (stillness check)
TOP_LOCAL = (0.0, 0.0, TUBE_H)  # the middle of the cap's top

TUBES = ("tube_1", "tube_2", "tube_3", "tube_4")
# Where each tube starts (its bottom centre) and which way its label faces (yaw of local +x, rad): every label
# faces along the holding rack's length's normal (+y or -y), a quarter turn from the reader direction (-x), and the
# two tubes in each rack need opposite turns.
TUBE_START = {
    "tube_1": (HOLD_X[0], +HOLD_Y),
    "tube_2": (HOLD_X[1], +HOLD_Y),
    "tube_3": (HOLD_X[0], -HOLD_Y),
    "tube_4": (HOLD_X[1], -HOLD_Y),
}
TUBE_START_YAW = {
    "tube_1": +math.pi / 2,  # label toward +y (the robot's left): needs a quarter turn anticlockwise seen from above
    "tube_2": -math.pi / 2,  # label toward -y: a quarter turn clockwise
    "tube_3": -math.pi / 2,  # label toward -y: clockwise
    "tube_4": +math.pi / 2,  # label toward +y: anticlockwise
}
READER_DIR = (-1.0, 0.0)  # through each slot, from the tube toward the arms

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
POS_TOL = POS_IN / 2  # 'in a position': the tube's bottom centre inside the position's 24 x 24 mm opening
BOTTOM_TOL = 0.005  # its bottom centre within 5 mm of the table
TILT_MAX_DEG = 5.0  # its axis within 5 degrees of upright: standing on its own. A tube tips at about 8.9 to 9.6
#                     degrees (its centre of mass is 49 mm up), so one resting against a rack wall leans past that
#                     (at most 11.3 degrees in a 24 mm position with 40 mm walls) and never counts; a free-standing
#                     one stands at about 0
BEARING_MAX_DEG = 30.0  # readable: the label's facing direction, seen from above, within 30 degrees of -x
STILL_WINDOW_S = 0.5  # at rest: the tube's bottom, top and label face each moved at most STILL_TOL over 0.5 s
STILL_TOL = 0.002
LABEL_EDGE_TOL = 0.0  # visible: both vertical edges of the label's face, seen from the reader (y), inside the slot
RELEASE_CLEAR = 0.020  # let go: both grippers' finger zones (rigstate.py: pads round the gripper's axis, from the
#                       pad tips 38 mm back) at least 20 mm from the tube; a tube a gripper still holds earns nothing
PER_TUBE = 0.25
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # tubes moving this far in one control step were teleported: a new episode began
START_TOL = 0.045  # ... onto their starting marks: within this of them (covers a seeded reset's jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
RACK_GREY = (0.22, 0.23, 0.25)  # the analyser sample rack (dark grey ABS)
HOLD_GREY = (0.70, 0.71, 0.72)  # the bench tube racks (light grey)
BLOOD = (0.42, 0.05, 0.07)  # the tube body: clear PET with dark red blood seen through it
CAP_LAVENDER = (0.62, 0.50, 0.80)  # EDTA cap
LABEL_WHITE = (0.97, 0.97, 0.96)
