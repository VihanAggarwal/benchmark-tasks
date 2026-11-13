"""Every dimension and threshold of the task, in one place (metres, kilograms, radians unless named, station frame).

Real size throughout: an X-ACTO KS 1031 manual crank sharpener screwed to a pine 2x4 offcut, a Dixon Ticonderoga
My First pencil without eraser (33036), new and unsharpened, and a two-notch pine pen rest.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named.

The sharpener's base frame sits on the table at (0.300, 0, TOP), not turned, so its local y IS the world y and its
local z the height above the table. The bore axis runs along y at x 0.300, 0.118 above the table; the pencil goes in
from +y (the dial face at y +0.0405) toward -y (the crank on the far face at y -0.0805). The crank turns about +y;
positive (forward) carries the knob from the top toward +x, away from the robot.
"""

import math

TOP = 0.7475  # station table top

# -- the sharpener: body (outer envelope of the KS, 121 x 70 x 108 mm) on a pine 2x4 offcut ----------------------
SH_X = 0.300  # base frame origin, on the table (local y = world y)
AXIS_X, AXIS_Z = 0.0, 0.118  # the bore axis in the base frame (x 0.300 in the world, 118 mm above the table)
BLOCK = ((0.0, -0.020, 0.019), (0.089, 0.150, 0.038))  # pine 2x4 offcut 38 x 89 x 150, lying flat (centre, size)
BLOCK_MASS_NOTE = 0.254  # kg, 0.51 L of pine at 0.5 g/cm3 (fixed: the mass is not used)
BODY_X = (-0.035, 0.035)  # 70 mm across
BODY_Y = (-0.0805, 0.0405)  # 121 mm along the bore: crank face to dial face
BODY_Z = (0.038, 0.146)  # 108 mm tall, on the offcut's top
SHARPENER_MASS_NOTE = 0.295  # kg, the KS 1031's listed 10.4 oz (fixed: the mass is not used)
TUNNEL = 0.0095  # the body is solid round a 19 x 19 mm square passage along the bore axis ...
TUNNEL_Y0 = -0.0605  # ... from the crank end wall (20 mm thick) to the dial face
# The bore: octagonal rings 12.3 mm across flats (the dial's 7/16 in, 11.1 mm round hole, built 1.0 mm a side for
# the solver's fit range), each a ring of 8 boxes, 3 mm thick radially
BORE_APOTHEM = 0.00615  # half of 12.3 mm across flats
RING_T = 0.003  # radial thickness: inner face 6.15 mm, outer 9.15 mm off the axis
RING_FACE_W = 0.0076  # each box's width round the ring (the octagon's side at its outer radius, 7.58 mm)
DIAL_RING_Y = (0.0345, 0.0375)  # the dial's hole (narrow part) ...
DIAL_LEAD_Y = (0.0375, 0.0405)  # ... with a 45 deg lead-in 3 mm deep (9.15 mm off the axis at the face to 6.15)
CARRIAGE_RING_Y = (0.0125, 0.0165)  # the cutter carriage's entry ring ...
CARRIAGE_LEAD_Y = (0.0165, 0.0180)  # ... with a 1.5 mm, 45 deg lead-in on its +y side (7.65 mm off the axis to 6.15)
LEAD_T = 0.002  # lead-in plate thickness
PAST_RINGS_Y = CARRIAGE_RING_Y[0]  # a pencil end below this y has gone through both rings
RING_SPAN = DIAL_RING_Y[1] - CARRIAGE_RING_Y[0]  # 25 mm between the rings' outer ends

# Cutting front (a disc on a sprung feeler) on the cutting slide (the cut depth)
FRONT_Y0 = 0.0105  # the disc's face at rest
FRONT_R, FRONT_LEN = 0.007, 0.004  # 14 mm disc, 4 mm thick (back at y 0.0065)
FEELER_TRAVEL = 0.005  # 5 mm of feeler travel, then it bottoms on the slide (its joint limit)
FEELER_K = 1000.0  # N/m: a 2 N press pushes it in 2 mm
FEELER_DAMP = 20.0  # N s/m (critical 6.3: overdamped)
FEELER_MASS = 0.010
SLIDE_ANCHOR_Y = 0.0015
SLIDE_ROD = ((0.0, -0.006, AXIS_Z), (0.004, 0.015, 0.004))  # rod y -0.0135 .. 0.0015, centre and size
SLIDE_TRAVEL = 0.026
SLIDE_K = 50000.0  # N/m: held at the cut depth (1 mm per 50 N past it)
SLIDE_DAMP = 200.0  # N s/m (critical 63: overdamped)
SLIDE_MASS = 0.020

# Crank: steel hub and arm on the far face, black plastic knob spinning freely on its pin
CRANK_ANCHOR = (0.0, -0.0855, AXIS_Z)
HUB_R, HUB_LEN = 0.008, 0.010  # y -0.0805 .. -0.0905
CRANK_ARM = ((0.0, -0.0935, 0.1405), (0.014, 0.006, 0.059))  # y -0.0905 .. -0.0965, from 7 mm below the axis
CRANK_R = 0.045  # axis to knob pin (the circle the knob's centre runs on is 90 mm across)
KNOB_R, KNOB_LEN = 0.010, 0.050  # 20 mm across, 50 mm long: y -0.0965 .. -0.1465
KNOB_PIN = (0.0, -0.0965, AXIS_Z + CRANK_R)
KNOB_CENTRE_Y = KNOB_PIN[1] - KNOB_LEN / 2  # -0.1215
CRANK_MASS = 0.055  # steel arm 4.96 cm3 (39 g) + hub 2.01 cm3 (16 g)
KNOB_MASS = 0.014  # solid polypropylene, 15.7 cm3 x 0.9
CRANK_LIMIT = 6.0  # rad either way (inside the kit's 2 pi); the rule turns the angle back by 2 pi past +-pi
KNOB_LIMIT = 6.0
WRAP_AT = math.pi  # the rule rewrites a joint angle beyond +-pi by a whole turn (the same pose)
BEARING_FRICTION = 0.05  # N m, the crank's own bearing
KNOB_FRICTION = 0.002  # N m on its pin
KNOB_DAMP = 0.0005
CRANK_K = 2.0  # N m/rad: the crank's drive spring, whose target the cutting-drag rule sets every step
CRANK_DAMP = 0.02  # N m s/rad (critical about 0.022 for the crank and knob, 6.0e-5 kg m2)
CRANK_INERTIA = 6.0e-5  # kg m2 about the axis: arm 3.2e-5, knob 2.8e-5, hub 5e-7

# -- the cut (rule, scoring.py Cutter) ---------------------------------------------------------------------------
PRESS = 0.002  # the pencil pushes the feeler in at least 2 mm (2 N) ...
TOUCH = 0.001  # ... with its end within 1 mm of the disc's face ...
ON_AXIS = 0.0015  # ... and within 1.5 mm of the bore axis, past both rings
CUT_PER_TURN = 0.0015  # the slide's target moves 1.5 mm deeper per pressed forward turn ...
CUT_DEPTH = 0.024  # ... to 24 mm, where the point is complete (the pencil stop): 16 turns
TURNS_TO_POINT = CUT_DEPTH / CUT_PER_TURN  # 16
T_CUT = 0.15  # N m of cutting drag while pressed and the point is unfinished ...
W_SAT = 0.5  # rad/s: ... rising with crank speed to its full value at 0.5 rad/s and above
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
# The tiers: stages of the cut (D, m)
TIER_DEPTHS = ((0.75, CUT_DEPTH), (0.5, 0.012), (0.25, 0.006))

# -- the pencil: one lying cylinder along y ---------------------------------------------------------------------
PENCIL_R = 0.00515  # 13/32 in (10.3 mm) across
PENCIL_LEN = 0.166  # 6.52 in
HALF = PENCIL_LEN / 2
PENCIL_MASS = 0.009  # 13.8 cm3: incense cedar about 4.4 g plus graphite about 4.2 g
PENCIL_FRICTION = (0.5, 0.4)  # lacquer
SHARPENER_FRICTION = (0.4, 0.3)  # steel and plastic
# -- the pen rest: pine, a base strip and two V blocks with 90 deg notches -------------------------------------
REST_XY = (0.220, 0.220)
REST_BASE = (0.030, 0.130, 0.008)
V_BLOCK_YS = (0.175, 0.265)  # world y of the two V blocks (local -0.045 and +0.045)
V_BLOCK = (0.030, 0.016, 0.022)  # each block 30 x 16 x 22 mm (on the base strip: top 30 mm above the table)
V_APEX_Z = 0.023  # the notch's apex, 23 mm above the table; 90 deg, 7 mm deep, top edges at x 213 and 227 mm
V_TOP_Z = 0.030
V_HALF_W = V_TOP_Z - V_APEX_Z  # 7 mm: the notch is 14 mm wide at the top
SEAT_Z = V_APEX_Z + PENCIL_R * math.sqrt(2.0)  # 0.03028: the pencil's axis when it lies in both notches
PENCIL_START = (REST_XY[0], REST_XY[1], SEAT_Z)  # the pencil's centre at the start
REST_MASS_NOTE = 0.025  # kg, 50.8 cm3 of pine (fixed: the mass is not used)

# -- the end state (scoring.py Tracker) -------------------------------------------------------------------------
CLEAR = 0.010  # the whole pencil at least 10 mm from the sharpener's outer box ...
SHARPENER_BOX = ((SH_X - 0.035, -0.0805, 0.038), (SH_X + 0.035, 0.0405, 0.146))  # ... x, y, z (world, z above table)
SEAT_SIDE_TOL = 0.003  # the pencil's axis within 3 mm sideways (x) of the notch's seat at both V blocks ...
SEAT_HEIGHT_TOL = 0.002  # ... and within 2 mm in height
LET_GO = 0.030  # every finger zone of both grippers at least 30 mm from the pencil
STILL_WINDOW_S = 1.0  # released on every step and moved 2 mm or less over the last 1 s
STILL_PATH = 0.002
EPS = 1e-9
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): the pencil back on its start pose
# (within 3 mm) in one step after a jump of 50 mm or more.
RESET_START_TOL = 0.003
RESET_JUMP = 0.050

# -- colours (sRGB) ---------------------------------------------------------------------------------------------
BLACK = (0.06, 0.06, 0.07)
STEEL = (0.72, 0.72, 0.75)
DIAL_GREY = (0.45, 0.45, 0.48)
PINE = (0.86, 0.72, 0.50)
YELLOW = (0.98, 0.80, 0.10)
CUTTER_GREY = (0.55, 0.55, 0.58)
