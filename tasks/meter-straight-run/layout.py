"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.

The run lies along y on the line x 0.300, between two low rails: water comes in through the copper elbow at the
left (+y) end and leaves through the brass valve at the right (-y) end. Pipe size 25 mm, so 10 pipe diameters is
250 mm and 5 is 125 mm.
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the run ----------------------------------------------------------------------------------------------------
RUN_X = 0.300  # the channel's centre line
ELBOW_FACE_Y = 0.290  # the elbow's -y face: where the run starts
VALVE_FACE_Y = -0.290  # the valve's +y face: where it ends (580 mm of run)
PIPE_W = 0.025  # every part's square section (the meter's body too)
HALF_W = PIPE_W / 2

# Fixed parts (no mass). (centre x, centre y, centre height), size.
ELBOW_C, ELBOW_SIZE = (0.300, 0.310, 0.020), (0.040, 0.040, 0.040)
INLET_TILE_C, INLET_TILE_SIZE = (0.300, 0.310, 0.0405), (0.020, 0.020, 0.001)  # white mark on the elbow's top
SUPPLY_C, SUPPLY_SIZE = (0.350, 0.310, 0.0125), (0.060, 0.025, 0.025)  # the supply pipe into the elbow, along x
VALVE_C, VALVE_SIZE = (0.300, -0.310, 0.020), (0.040, 0.040, 0.040)
LEVER_C, LEVER_SIZE = (0.300, -0.310, 0.043), (0.010, 0.040, 0.006)  # red lever along the pipe: valve open
RAIL_SIZE = (0.006, 0.580, 0.008)
RAIL_NEAR_C = (0.2785, 0.0, 0.004)  # inner face x 0.2815
RAIL_FAR_C = (0.3215, 0.0, 0.004)  # inner face x 0.3185: 37 mm between the rails, 6 mm of play per side

# Loose parts, all lying flat along x at the start. name -> (centre x, centre y, length, mass).
# In the setup image (the first top-camera frame) the arms' start pose covers the table outboard of |y| 0.45 and,
# farther out than x 0.40, outboard of |y| 0.34 (the grippers): every part starts inside |y| 0.4325 and short of
# x 0.40 except the 250's far 85 mm. The short pipes lie nearest the bases, 22.5 mm short of the meter and the 250.
PIPE_START = {
    "pipe_250": (0.360, -0.410, 0.250, 0.12),
    "pipe_125": (0.150, -0.420, 0.125, 0.06),
    "pipe_75": (0.150, 0.420, 0.075, 0.035),
}
PIPES = tuple(PIPE_START)
LENGTH = {n: v[2] for n, v in PIPE_START.items()}

# The meter: one rigid compound (mechkit), base frame at the body's centre, local +x along the flow arrow, +z up
# through the register. Shapes in that frame: (centre, size).
METER_START = (0.260, 0.410)  # body centre; body along x, arrow toward +x (away from the arms)
METER_MASS = 0.20
METER_LEN = 0.100
BODY = ((0.0, 0.0, 0.0), (METER_LEN, PIPE_W, PIPE_W))
REGISTER = ((0.0, 0.0, HALF_W + 0.006), (0.040, 0.016, 0.012))  # top 24.5 mm above the body's centre
ARROW_Z = HALF_W + 0.012 + 0.0015  # arrow strips 3 mm thick on the register's top
ARROW = (  # a shaft 18 x 4 mm and a head of three 4 mm steps, 12, 8 and 4 mm wide, pointing to local +x
    ((-0.006, 0.0, ARROW_Z), (0.018, 0.004, 0.003)),
    ((0.005, 0.0, ARROW_Z), (0.004, 0.012, 0.003)),
    ((0.009, 0.0, ARROW_Z), (0.004, 0.008, 0.003)),
    ((0.013, 0.0, ARROW_Z), (0.004, 0.004, 0.003)),
)
METER_TOP_LOCAL = ARROW_Z + 0.0015  # 27.5 mm above the body's centre: the meter's outline box top
METER_GRIP_LOCAL = -0.025  # picked 25 mm behind the body's centre, toward its inlet (-x) end: the 27 mm wide pads
#   then end 11.5 mm short of the inlet end (more than the pads' 8 mm offset), so neither catches the 250 while the
#   meter is slid to butt; they close on the 25 mm body's sides, 4.5 mm outside the 16 mm register
PAD_WIDTH = 0.027  # a finger pad's width across the closing direction, at the TCP
PAD_OFFSET = 0.008  # how far the pads can sit off the commanded point (measured on the GPU)
PARTS = (*PIPES, "meter")
LENGTH["meter"] = METER_LEN

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------
FLAT_DEG = 5.0  # (a) a straight's nearest long face, or the meter's register, within this of facing up/down
HEADING_DEG = 5.0  # (b) the long axis within this of y, seen from above
CENTRE_TOL = 0.008  # (c) the centre within this of x 0.300 (between the rails, not on one)
BOTTOM_TOL = 0.003  # (d) the lowest point within this of the table top (not on a rail or another part)
JOINT_MAX = 0.005  # a checked joint is closed at 5 mm or less ...
JOINT_MIN = -0.002  # ... and not below -2 mm (contact penetration; a part past the elbow or on another is not)
U_MIN = 0.245  # elbow face to the meter's inlet face: 10 pipe diameters (250 mm), less 5 mm
D_MIN = 0.120  # the meter's outlet face to the valve face: 5 pipe diameters (125 mm), less 5 mm
EPS = 1e-6  # 1 micrometre: the thresholds above are inclusive, floating-point round-off aside
FINGER_LEN = 0.038  # a gripper's 'finger line': from its pad-tip point this far back up the gripper (the pads and
#   the fingers' 40 mm wide section, 19-38 mm up, which is all of a finger that a part lying on the table can touch)
OPEN_FINGER_REACH = (0.04695 + 0.010, 0.020)  # an open finger's outer corner: (out along the closing axis, across)
RELEASE = 0.065  # let go: every point of each gripper's finger line this far from every part's box. An open finger
#   reaches 60.4 mm out from that line (hypot(57, 20) mm), so at any tilt no finger of a let-go gripper touches a part
STILL_WINDOW_S = 0.5  # stillness: every part's end points travelled ...
STILL_PATH = 0.002  # ... at most this far in all over the last 0.5 s (path, never velocity)
RESET_JUMP = 0.08  # a part moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)
GRAVITY = 9.81

# Colours (sRGB).
COPPER = (0.72, 0.45, 0.20)
BRASS = (0.78, 0.62, 0.30)
RAIL_GREY = (0.60, 0.60, 0.62)
WHITE = (0.95, 0.95, 0.95)
LEVER_RED = (0.80, 0.10, 0.10)
REGISTER_BLUE = (0.15, 0.30, 0.70)
