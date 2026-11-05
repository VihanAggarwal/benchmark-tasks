"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The live panel: a dark mounting plate (fixed, 3 mm) with a yellow border (four strips on its edges, 2 mm tall),
# 160 x 160 mm, in front of the left arm. Only the left arm reaches its socket with a straight-down grip.
PANEL_X, PANEL_Y = 0.30, 0.15
PANEL_SIZE = 0.160  # the outline's outer edge, square
PANEL_HALF = PANEL_SIZE / 2
PLATE_H = 0.003
PLATE_TOP = TOP + PLATE_H
BORDER_W = 0.012  # the yellow strips' width
BORDER_H = 0.002
ZONE_HEIGHT = 0.15  # a gripper is inside the panel when over its outline and less than this above the table top
ZONE_TOP_Z = TOP + ZONE_HEIGHT

# The fuse base (socket): four fixed walls round a 30 x 30 mm opening at the panel's centre; its floor is the plate.
# 25 mm walls leave the fuse's top 25 mm standing above the rim, so the pads close on it 8 mm above the porcelain.
SOCKET_X, SOCKET_Y = PANEL_X, PANEL_Y
SOCKET_INNER = 0.030  # 5 mm round a 20 mm fuse on each side
SOCKET_WALL = 0.008
SOCKET_HEIGHT = 0.025
SOCKET_FLOOR = PLATE_TOP
SOCKET_RIM = PLATE_TOP + SOCKET_HEIGHT
SOCKET_OUTER_HALF = SOCKET_INNER / 2 + SOCKET_WALL  # 0.023

# Two copper busbars (fixed, on the plate) running from the socket's front and back walls toward the border.
BUS_LEN = 0.040  # along x
BUS_W = 0.012  # along y
BUS_H = 0.006
BUS_GAP = 0.001  # from the socket's outer wall
BUS_OFFSET = SOCKET_OUTER_HALF + BUS_GAP + BUS_LEN / 2  # busbar centre to the socket centre, along x (0.044)

# The fuses: a cartridge fuse is a ceramic tube with brass end caps, 20 mm across and 50 mm long, standing upright.
FUSE_RADIUS = 0.010
FUSE_HEIGHT = 0.050
FUSE_MASS = 0.030  # 15.7 cm3 of sand-filled ceramic and brass (1.9 g/cm3)
RED_X, RED_Y = SOCKET_X, SOCKET_Y  # the blown fuse, seated in the socket

# The spare's cup: four fixed walls round a 30 x 30 mm opening, 20 mm tall, in front of the right arm (only the
# right arm reaches it with a straight-down grip); its floor is the table.
CUP_X, CUP_Y = 0.25, -0.35
CUP_INNER = 0.030
CUP_WALL = 0.008
CUP_HEIGHT = 0.020

# The hand-off pad: a thin white pad (fixed) on the midline, outside the panel; both arms reach it.
HANDOFF_X, HANDOFF_Y = 0.18, 0.0
HANDOFF_SIZE = (0.070, 0.070, 0.003)
HANDOFF_TOP = TOP + HANDOFF_SIZE[2]

# The discard tray: four fixed walls round an 80 x 80 mm floor (the table top), in front of the left arm.
TRAY_X, TRAY_Y = 0.15, 0.30
TRAY_INNER = 0.080
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.025
TRAY_INNER_HALF = TRAY_INNER / 2

# The parking squares: thin grey pads (fixed), one in front of each arm's base, well away from the panel.
PARK_X = 0.12
PARK_Y = {"left": 0.42, "right": -0.42}
PARK_SIZE = 0.100
PARK_HALF = PARK_SIZE / 2
PARK_H = 0.002
PARK_HEIGHT = 0.10  # parked: the TCP over the square's outline and less than this above the table top

# The grippers' finger zone (scoring.py): rigstate gives each gripper's TCP, the point between its finger tips
# (13 mm past the TCP) and its opening, not its heading, so the fingers are taken to fill a flat-ended round
# cylinder about the gripper's axis, from the tip point FINGER_LEN back up, out to a finger's outer corner.
FULL_GAP = 0.0939  # rigstate opening 1.0: the fingers 94 mm apart
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis)
FINGER_LEN = 0.038  # from the tip point 38 mm back up the gripper (13 mm to the TCP and 25 mm past it)

# Rule and tier thresholds (scoring.py).
SEATED_TILT_DEG = 10.0  # seated: the fuse's axis within this of vertical (either end down) ...
SEATED_Z_TOL = 0.003  # ... its lowest point within this of the socket floor ...
SEATED_CENTRE_TOL = 0.008  # ... and its centre within this of the socket's centre, horizontally (inside the opening)
IN_SOCKET_DEPTH = 0.005  # still in the socket: centre over the opening, lowest point this far below the rim or more
TRAY_LOW_TOL = 0.010  # in the tray: centre over its floor, lowest point within this of the table top
OFF_TABLE_DROP = 0.010  # a fuse whose lowest point is this far below the table top has left the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a fuse moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
PLATE_GREY = (0.28, 0.28, 0.30)
PANEL_YELLOW = (0.95, 0.80, 0.10)
IVORY = (0.90, 0.87, 0.78)  # porcelain fuse base
COPPER = (0.78, 0.48, 0.26)
RED = (0.85, 0.12, 0.10)
GREEN = (0.12, 0.62, 0.22)
CUP_BLUE = (0.25, 0.45, 0.85)
WHITE = (0.97, 0.97, 0.97)
TRAY_BLACK = (0.10, 0.10, 0.10)
PARK_GREY = (0.55, 0.55, 0.57)
