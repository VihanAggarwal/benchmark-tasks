"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A plumber's bench mock-up of a horizontal drain (drain-waste-vent pipe) running across the table along y, 38 cm in
front of the arms. Waste flows toward -y (painted black arrows on its top). A 134 mm section has been cut out of it
around y 0 to take a new branch: a horizontal branch drain from a fixture on the front side (-x, toward the arms).
Two loose white fittings stand on the bench, a wye and a sanitary tee, and a grey branch pipe lies on the left.

Fitting frame (both fittings): origin at the middle of the run's bottom face, +y along the run, +z up. The wye's
branch leaves the run's middle at 45 degrees, its outward axis along (-0.7071, +0.7071); the tee's branch leaves it
at 90 degrees, along -x. Shapes are axis-aligned in that frame (mechkit's rule), so the wye's 45 degree branch is a
row of six overlapping upright round discs (40 mm across, 40 mm tall) along its axis: a smooth 40 mm wide branch
with a rounded end.
"""

import math

TOP = 0.7475  # station table top (world z)
SQ2 = math.sqrt(0.5)

# -- the drain (fixed, grey PVC): two square-section lengths either side of the cut-out --------------------------------
DRAIN_X = 0.380  # its centre line along y
SEC = 0.040  # 40 mm square section (pipes and fittings): a stand-in for 1-1/2 in DWV pipe that cannot roll
DRAIN_H = SEC
GAP_HALF = 0.067  # the cut-out: y -0.067 .. +0.067 (134 mm)
DRAIN_END = 0.280  # each length runs from the gap out to |y| 0.280 (213 mm long)
GAP_CENTRE = (DRAIN_X, 0.0)
FLOW = (0.0, -1.0)  # the way waste flows: toward -y (downstream); upstream is +y
FLOW_YAW_DEG = -90.0

# -- the flow arrows (fixed, black paint on the drain's top), each 30 mm long, pointing -y -----------------------------
ARROW_Y = (-0.240, -0.120, 0.120, 0.240)  # arrow centres along the drain
ARROW_T = 0.0005  # 0.5 mm thick
# (y offset of the part's centre from the arrow's centre, size along x, size along y): a 4 mm shaft and a stepped head
ARROW_PARTS = (
    (+0.0060, 0.004, 0.018),  # shaft: y -0.003 .. +0.015
    (-0.0050, 0.015, 0.004),  # head, widest step: y -0.007 .. -0.003
    (-0.0090, 0.010, 0.004),  # y -0.011 .. -0.007
    (-0.0130, 0.005, 0.004),  # the tip: y -0.015 .. -0.011
)

# -- the two fittings (loose mechkit compounds, white PVC, 0.10 kg each) -----------------------------------------------
RUN_LEN = 0.118  # the run: 40 x 118 x 40 mm, along the fitting's y; 8 mm clearance at each end of the 134 mm gap
RUN_CENTRE = (0.0, 0.0, SEC / 2)
RUN_SIZE = (SEC, RUN_LEN, SEC)
BRANCH_REACH = 0.075  # each branch's end is 75 mm from the run's axis, measured along the branch's own axis
FITTING_MASS = 0.10
# wye: six upright discs, radius 20 mm, centred on the branch axis 20 to 55 mm out from the run's centre
WYE_DIR = (-SQ2, SQ2)  # the branch's outward axis in the fitting's frame (yaw of 135 degrees)
BEAD_R = SEC / 2
BEAD_S = (0.020, 0.027, 0.034, 0.041, 0.048, 0.055)  # the last disc's rim reaches 75 mm on the axis
WYE_BEADS = tuple((s * WYE_DIR[0], s * WYE_DIR[1], SEC / 2) for s in BEAD_S)
WYE_TIP = (BRANCH_REACH * WYE_DIR[0], BRANCH_REACH * WYE_DIR[1], SEC / 2)  # (-0.0530, +0.0530): the opening's centre
WYE_BRANCH_YAW_DEG = 135.0  # the outward axis's yaw in the fitting's frame: theta_b = fitting yaw + 135 (upright)
# sanitary tee: a square branch along -x, 55 mm out from the run's face, and a stepped sweep (20 mm legs) in its
# (-x, -y) inside corner
TEE_DIR = (-1.0, 0.0)
TEE_BRANCH_CENTRE = (-(SEC / 2 + (BRANCH_REACH - SEC / 2) / 2), 0.0, SEC / 2)  # x -0.075 .. -0.020
TEE_BRANCH_SIZE = (BRANCH_REACH - SEC / 2, SEC, SEC)
TEE_SWEEP = (  # (centre, size): three steps filling the corner between the run's face and the branch's -y side
    ((-0.0250, -0.0250, SEC / 2), (0.010, 0.010, SEC)),
    ((-0.0225, -0.0350, SEC / 2), (0.005, 0.010, SEC)),
    ((-0.0350, -0.0225, SEC / 2), (0.010, 0.005, SEC)),
)
TEE_TIP = (-BRANCH_REACH, 0.0, SEC / 2)
# where the fittings start (fitting-frame origin x, y and yaw in degrees)
WYE_START = (0.280, -0.210, -90.0)  # run along x, branch toward +x and +y (theta_b 45: back and upstream)
TEE_START = (0.240, -0.040, 180.0)  # run along y, branch toward +x (at the gap)

# -- the branch pipe (loose box, grey PVC): 40 x 40 mm section, 140 mm long along its own y ---------------------------
PIPE_SIZE = (SEC, 0.140, SEC)
PIPE_HALF_LEN = PIPE_SIZE[1] / 2
PIPE_MASS = 0.08
PIPE_START = (0.280, 0.360, 90.0)  # centre x, y; yaw 90: its length along x (footprint x 0.21-0.35, y 0.34-0.38)

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
FLAT_DEG = 5.0  # 'flat': the fitting's up axis (or the pipe's length) within 5 degrees of vertical (horizontal)
ON_TABLE = 0.003  # ... and its lowest point within 3 mm of the table top
SEAT_AXIS_DEG = 5.0  # seated: the run within 5 degrees of the drain's axis (y), either way round
SEAT_CENTRE = 0.006  # ... and the run's centre within 6 mm of the gap's centre in x and in y
BRANCH_OK_DEG = 135.0  # the correct branch direction: front (-x) and leaning upstream (+y)
BRANCH_TOL_DEG = 7.0
FINE = {"angle_deg": 7.0, "lateral": 0.006, "gap": (-0.003, 0.010)}  # branch pipe on line, fine
COARSE = {"angle_deg": 15.0, "lateral": 0.015, "gap": (-0.003, 0.040)}  # ... coarse
OUTWARD = 0.050  # the pipe lies outward of the opening: its centre at least 50 mm beyond it along the branch axis
TEE_CLEAR = 0.060  # 'the tee off the drain': resting on the bench (any face), its run's centre > 60 mm from the gap's
STILL_WINDOW_S = 0.5  # 'settled': over the last 0.5 s the centre's path is under 2 mm and it turned under 1 degree
STILL_PATH = 0.002
STILL_TURN_DEG = 1.0
HANDS_OFF = 0.015  # 'released': both grippers' finger zones at least 15 mm from it at every step of the last 0.5 s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a part moving this far in one control step was teleported: a new episode began

# -- the grippers (rigstate reports the TCP, the point between the pad tips and the opening) ---------------------------
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
PVC_GREY = (0.50, 0.52, 0.55)
PVC_WHITE = (0.93, 0.93, 0.90)
ARROW_BLACK = (0.04, 0.04, 0.04)
