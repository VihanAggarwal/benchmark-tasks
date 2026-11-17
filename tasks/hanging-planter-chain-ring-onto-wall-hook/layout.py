"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top.

A 4 in galvanized pail planter (100 mm across the rim, 80 mm tall, 0.25 kg with 170 g of sand) stands on the
table. Its 4 mm wire bail is hinged on two ears at the rim; a 2 in zinc ring is linked through the bail's apex.
On a plywood wall panel 430 mm in front of the arm bases is a steel plant bracket: a plate, an 8 mm bar projecting
150 mm toward the robot and an upturned lip at its free end. The planter has to go on the bar by its ring (over
the lip), hang from it clear of the table, and be let go of.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the wall panel and the bracket (fixed) ---------------------------------------------------------------------------
WALL_X0, WALL_X1 = 0.430, 0.450  # a 20 mm plywood panel standing on the table, its face toward the robot at x 430 mm
WALL_Y0, WALL_Y1 = -0.050, 0.450  # 500 mm wide
WALL_H = 0.320
PLATE_T = 0.006  # the bracket's plate, 6 mm steel, screwed to the panel's face
PLATE_X0 = WALL_X0 - PLATE_T
PLATE_W, PLATE_H = 0.050, 0.080  # across (y), tall (z)
HOOK_Y = 0.200  # the bar's centre line
HOOK_Z = 0.190  # the bar's axis above the table (a 4 in pot's worth of hanging room under it, see HANG_*)
BAR_R = 0.004  # 8 mm round bar
BAR_X0 = 0.274  # the bar's free end
BAR_X1 = PLATE_X0  # 424 mm: the bar runs 150 mm out from the plate
LIP_R = 0.004
LIP_X = BAR_X0 + LIP_R  # the upturned tip, a round post 8 mm thick, centred 4 mm in from the free end
LIP_UP = 0.010  # its top stands 10 mm above the bar's top (the tip curls up)
LIP_Z0 = HOOK_Z - BAR_R
LIP_Z1 = HOOK_Z + BAR_R + LIP_UP
BAR_BEHIND_LIP = LIP_X + LIP_R + 0.005  # the ring's hole must cross the bar this far behind the lip: x 287 mm
BAR_END_MARGIN = 0.005  # and no nearer than this to the plate
STEEL = (0.45, 0.47, 0.50)
PLYWOOD = (0.82, 0.68, 0.46)

# -- the planter (loose compound: pot, bail and ring) ----------------------------------------------------------------
# Frame of the base body (the pot): origin at the centre of the pot's bottom face, z up, bail hinge along y.
POT_R = 0.046  # the pot's body radius (a 4 in pail: 100 mm at the rim)
POT_BODY_H = 0.072
RIM_R = 0.050
RIM_H = 0.008
POT_H = POT_BODY_H + RIM_H  # 80 mm
POT_MASS = 0.25  # pot 0.08 kg + 170 g of dry sand
EAR_Z = 0.076  # the bail's hinge axis (through both ears, along y), 4 mm below the rim's top
BAIL_WIRE = 0.004
BAIL_Y = 0.053  # the side wires' centre lines, clear of the rim
BAIL_RISE = 0.048  # ear axis to the apex
APEX_Z = EAR_Z + BAIL_RISE  # 124 mm: the bail's top bar, where the ring is linked
BAIL_MASS = 0.020
RING_APOTHEM = 0.0225  # centre line of the wire, from the ring's centre (hole 40 mm across flats, 50 mm outside)
RING_IN = 0.020  # the hole's inscribed radius
RING_WIRE = 0.005
RING_MASS = 0.022
RING_DZ = RING_APOTHEM  # the ring's centre above the apex (its lower wire is linked on the bail)
RING_Z0 = APEX_Z + RING_DZ  # 146.5 mm: the ring's centre with the planter standing on the table
BAIL_SPRING, BAIL_DAMP = 0.03, 0.015  # N m/rad toward upright (the bail's own stiffness; a real bail has stops), N m s/rad
RING_SPRING, RING_DAMP = 0.01, 0.006
JOINT_LIMIT = 0.9  # rad each way
FRICTION = None  # the simulator's default for loose objects (1.2, 1.0)
GALV = (0.62, 0.64, 0.67)
ZINC = (0.80, 0.82, 0.85)

POT_X, POT_Y = 0.250, 0.100  # where the pot stands: its ring starts at (250, 100) mm, 100 mm from the hook's line
POT_START = (POT_X, POT_Y, 0.0)

# -- the hung pose ----------------------------------------------------------------------------------------------------
RING_HUNG_DZ = BAR_R - RING_IN  # -16 mm: the ring's upper wire rests on the bar's top, its centre 16 mm below the axis
RING_HUNG_Z = HOOK_Z + RING_HUNG_DZ  # 174 mm
HANG_BASE_Z = RING_HUNG_Z - RING_Z0  # 27.5 mm: the pot's bottom face above the table when hung and plumb
RING_THREAD_Z = HOOK_Z + BAR_R  # 194 mm: the ring's centre while the lip and bar are threaded through its hole

# -- rule thresholds (scoring.py) ---------------------------------------------------------------------------------------
N_MIN = 0.5  # on the bar: the ring's hole axis within 60 degrees of the bar's (x)
THROUGH_R = RING_IN  # the bar's axis crosses the ring's plane within 20 mm of the ring's centre (hung it is 16 mm)
HUNG_BELOW = 0.008  # and the ring's centre is at least 8 mm below the bar's axis (hanging from it, not held up beside it)
CLEAR = 0.020  # the pot's lowest point at least this far above the table top
PLUMB_DEG = 10.0  # the pot's axis within this of vertical
WALL_GAP = 0.003  # nothing else touches the pot: its rim this far short of the wall panel
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # over the last 1 s the pot's origin and the ring's centre each move under 5 mm
STILL_TURN_DEG = 5.0  # and the pot and the ring each turn under 5 degrees
LET_GO = 0.030  # both grippers' finger zones at least this far from the planter, on every step of that second
FALL_DROP = 0.020  # the pot's lowest point this far below the table top: it fell off (latched, score 0)
NEAR_START_TOL = 0.015  # the pot's origin within this of its starting mark (new-episode detection)
RESET_JUMP = 0.08  # ... having moved this much further in one control step than falling could carry it: a new episode
EPS = 1e-9
CONTROL_HZ = 20.0
