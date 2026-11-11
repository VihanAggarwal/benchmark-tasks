"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475, and the table top is the floor of the toilet room.

A floor plan of an accessible single-user toilet room at 1 in = 4 mm (1:6.35). The rear wall (behind the toilet)
is a grey kerb whose face is at x 0.455, the side wall (beside the toilet) a grey kerb whose face is at y -0.315;
the room is open toward -x and +y. The toilet is painted on the floor: its centreline is 17 in (68 mm) from the
side wall, at y -0.247 (2010 ADA Standards 604.2 allow 16 to 18 in).

The rule, 2010 ADA Standards for Accessible Design:
- 604.5.1 side wall bar: 42 in long minimum, its rear end 12 in maximum from the rear wall, extending 54 in
  minimum from the rear wall. Here: a bar at least 168 mm long, rear end at x >= 0.407, front end at x <= 0.239.
- 604.5.2 rear wall bar: 36 in long minimum, extending from the centreline 12 in minimum on the side closest to
  the side wall and 24 in minimum on the other (open, transfer) side. Here: a bar at least 144 mm long, its
  side-wall end at y <= -0.295 and its open end at y >= -0.151.
- 609.3: 1-1/2 in between a bar and the wall (6 mm here); the scorer accepts a wall-side face 0 to 15 mm off.
The 48 in (192 mm) bar is the only one that can meet the side-wall rule (the 40 in bar is under 42 in), and a rear
bar centred on the toilet fails the 24 in open-side minimum: pushed into the corner against the side wall the
40 in bar reaches only 23 in past the centreline on the open side.
"""

import math

TOP = 0.7475  # station table top: the room's floor
IN = 0.004  # one inch on the plan (1 in = 4 mm)

# -- walls: fixed kerbs on the floor ------------------------------------------------------------------------------
KERB_W = 0.010  # kerb thickness (10 mm, 2.5 in)
KERB_H = 0.004  # kerb height: low enough that a finger passes over it
REAR_FACE_X = 0.455  # the rear wall's face (toward -x, the room)
SIDE_FACE_Y = -0.315  # the side wall's face (toward +y, the room)
REAR_KERB = (REAR_FACE_X, REAR_FACE_X + KERB_W, -0.325, 0.005)  # x0, x1, y0, y1: 10 x 330 mm
SIDE_KERB = (0.205, REAR_FACE_X, SIDE_FACE_Y - KERB_W, SIDE_FACE_Y)  # 250 x 10 mm, ending at the rear wall's face
# The stretch of each wall face a bar can lie along (the side kerb's face runs from x 0.205 to the rear wall's face;
# the rear kerb's face from the side wall's face to y +0.005).
SIDE_SPAN = (SIDE_KERB[0], REAR_FACE_X)  # x
REAR_SPAN = (SIDE_FACE_Y, REAR_KERB[3])  # y

# -- the toilet, painted on the floor -----------------------------------------------------------------------------
CENTRELINE_IN = 17.0  # from the side wall (604.2: 16 to 18 in)
CL_Y = round(SIDE_FACE_Y + CENTRELINE_IN * IN, 9)  # -0.247
TOILET_W = 18 * IN  # 72 mm: the bowl's and the tank's width
TANK_D = 10 * IN  # 40 mm: the tank, against the rear wall
TANK = (REAR_FACE_X - TANK_D, REAR_FACE_X, CL_Y - TOILET_W / 2, CL_Y + TOILET_W / 2)  # x 0.415..0.455
BOWL_R = TOILET_W / 2  # a round bowl 18 in across, touching the tank's front edge
BOWL_X = TANK[0] - BOWL_R  # 0.379: the bowl spans x 0.343..0.415 (the toilet is 28 in deep)
PAINT_T = 0.0005  # the painted shapes are 0.5 mm thick

# -- grab bars: loose ---------------------------------------------------------------------------------------------
BAR_W = 0.020  # square section, 20 mm (over-scale, so a gripper can hold it)
BAR_LEN = {"bar_48": 48 * IN, "bar_40": 40 * IN}  # 0.192, 0.160
BAR_MASS = {"bar_48": 0.058, "bar_40": 0.048}  # aluminium square tube 20 x 20 x 1.5 mm, capped
BAR_START = {  # name -> (x, y, yaw in degrees); yaw 0 = the bar's length along x
    "bar_48": (0.240, -0.120, 90.0),  # parallel to the rear wall: x 0.230..0.250, y -0.216..-0.024
    "bar_40": (0.350, -0.060, 0.0),  # parallel to the side wall: x 0.270..0.430, y -0.070..-0.050
}
NAMES = ("bar_48", "bar_40")

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
SIDE_LEN_MIN = 42 * IN  # 0.168: 604.5.1
SIDE_REAR_MAX = 12 * IN  # 0.048: the side bar's rear end at most this from the rear wall's face
SIDE_EXTENT_MIN = 54 * IN  # 0.216: its front end at least this from the rear wall's face
REAR_LEN_MIN = 36 * IN  # 0.144: 604.5.2
REAR_WALL_SIDE_MIN = 12 * IN  # 0.048: the rear bar past the centreline toward the side wall
REAR_OPEN_SIDE_MIN = 24 * IN  # 0.096: ... and toward the open side
WALL_GAP = (-0.001, 0.015)  # 'along a wall': the wall-side face 0 to 15 mm off the kerb's face at the bar's
#                             midpoint (-1 mm: a bar pressed against the kerb counts; 609.3's 1-1/2 in is 6 mm)
SPAN_TOL = 0.005  # ... and both ends within the kerb face's stretch, with this much over
PARALLEL_DEG = 5.0  # ... its length within this of the wall's direction, seen from above
FLAT_DEG = 5.0  # 'flat': one of its four long faces within this of facing straight down
ON_FLOOR = 0.003  # ... and its lowest point within this of the floor (above or below)
HANDS_OFF = 0.010  # 'let go': both grippers' finger zones at least this far from the bar ...
STILL_WINDOW_S = 0.5  # ... at every control step of the last 0.5 s
STILL_PATH = 0.002  # 'settled': the bar's centre travelled under 2 mm in the last 0.5 s ...
STILL_YAW_DEG = 1.0  # ... and its heading changed under 1 degree
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a bar landing on its starting mark having moved this much further in one control step than its
#                    last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)
START_YAW_TOL_DEG = 10.0

# -- the grippers (the let-go rule; rigstate reports the TCP, the point between the pad tips and the opening) -------
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- where the demo lays the bars (inside the windows, trimmed in task.py) ------------------------------------------
# Side bar: centre y -0.297 (wall-side face 8 mm off: 2 mm wide of 609.3's 1-1/2 in, so the trimmed aim, which moves
# the bar toward the kerb, still keeps its corners >= 2.5 mm clear of the kerb), rear end 38 mm from the rear wall
# (window 24..48 mm for a 192 mm bar), front end 230 mm out (>= 216). Rear bar: centre x 0.437 (8 mm off, likewise),
# side-wall end 12 mm off the side wall (window 4..20 mm for a 160 mm bar), open end 104 mm past the centreline (>= 96).
TARGET = {"bar_48": (REAR_FACE_X - 0.038 - BAR_LEN["bar_48"] / 2, SIDE_FACE_Y + 0.008 + BAR_W / 2),  # (0.321, -0.297)
          "bar_40": (REAR_FACE_X - 0.008 - BAR_W / 2, SIDE_FACE_Y + 0.012 + BAR_LEN["bar_40"] / 2)}  # (0.437, -0.223)

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
KERB_GREY = (0.42, 0.42, 0.44)  # painted wall kerbs
PAINT_WHITE = (0.97, 0.97, 0.97)  # the toilet's bowl
TANK_WHITE = (0.86, 0.87, 0.89)  # the toilet's tank (a shade greyer, so the two shapes read apart)
STEEL = (0.74, 0.76, 0.79)  # brushed aluminium grab bars


def start_yaw(name: str) -> float:
    return math.radians(BAR_START[name][2])


assert abs(CL_Y - (-0.247)) < 1e-9
assert abs(SIDE_LEN_MIN - 0.168) < 1e-9 and abs(SIDE_EXTENT_MIN - 0.216) < 1e-9
assert abs(REAR_OPEN_SIDE_MIN - 0.096) < 1e-9 and abs(REAR_WALL_SIDE_MIN - 0.048) < 1e-9
