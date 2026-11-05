"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say z.
"""

TOP = 0.7475  # station table top

# Floor bearers: three fixed dark-grey timbers, 36 mm tall, running along x under the crate. The two gaps between
# them, centred on y +-0.120 and 40 mm wide, are the fork pockets (the table top is their floor). The pockets are
# 0.24 m apart so that two arms working at them side by side, close in to their bases, keep 26 mm between their
# wrists (capsule proxy, yamkin); at 0.20 m apart the wrists would overlap by 17 mm at the bars' rear ends.
BEARER_H = 0.036
FLOOR_X0, FLOOR_X1 = 0.270, 0.380
CENTRE_BEARER_Y = 0.100  # the centre bearer spans y -0.100 .. +0.100
OUTER_BEARER_Y0, OUTER_BEARER_Y1 = 0.140, 0.160  # the outer bearers span |y| 0.140 .. 0.160
POCKET_Y = 0.120  # pocket centre lines, y +-0.120 (in the crate's own frame too: the crate is centred on y 0)
POCKET_W = OUTER_BEARER_Y0 - CENTRE_BEARER_Y  # 40 mm
FLOOR_TOP = BEARER_H  # 0.036: the crate's underside at the start

# The crate: a closed, empty, rigid crate of 3 mm poplar plywood (about 0.47 g/cm3), too big for a gripper on every
# side. It rests on the centre bearer and on 10 mm of each outer bearer.
CRATE_SIZE = (0.110, 0.300, 0.050)  # along x, along y, height
CRATE_MASS = 0.15
CRATE_X, CRATE_Y = 0.325, 0.0  # resting square on the floor bearers, front face (toward the arms) at x 0.270
CRATE_HALF = tuple(s / 2 for s in CRATE_SIZE)

# The fork bars: two loose square aluminium tubes (1.5 mm wall), lying on the table along x, each in line with a
# pocket, their tips at the pocket mouths (x 0.270).
BAR_SIZE = (0.160, 0.024, 0.024)
BAR_MASS = 0.06
BAR_X = 0.190  # centres; rear ends at x 0.110, tips at x 0.270
BAR_GRIP_FROM_REAR = 0.015  # the demo grips each bar 15 mm from its rear end (x 0.125 at the start)
BAR_HALF_LEN = BAR_SIZE[0] / 2
BAR_HALF_H = BAR_SIZE[2] / 2

# Lines taped on the table beside each bar's path (fixed, 1 mm thick, 10 mm along x, 30 mm along y, centred on
# y +-0.170, 23 mm outside the bar's side): a bar's rear end level with the white line puts its tip 100 mm into
# its pocket (fully in); level with the yellow line, 73 mm (two-thirds).
LINE_SIZE = (0.010, 0.030, 0.001)
LINE_Y = 0.170
FULL_LINE_X = 0.210
TWO_THIRDS_LINE_X = 0.183

# The rack: a fixed blue platform 36 mm tall with three bearers on it in the floor bearers' plan, so the crate can
# be set down on them and the bars slid out from the gaps below it.
RACK_X0, RACK_X1 = 0.400, 0.550
RACK_HALF_Y = 0.170
RACK_PLATFORM_H = 0.036
RACK_BEARER_X0, RACK_BEARER_X1 = 0.410, 0.520
RACK_TOP = RACK_PLATFORM_H + BEARER_H  # 0.072: the rack bearers' tops

# Rule thresholds (scoring.py).
FULL_IN = 0.095  # a bar is fully in its pocket: at least this much of it under the crate (100 mm at the line)
TWO_THIRDS_IN = 0.073  # lifting with less than this under the crate on either side is lifting on the tips: 0
POCKET_BAND = 0.015  # a bar counts as in a pocket while its centre line under the crate is within this of the
#                      pocket's centre line (crate frame, across; the pocket leaves the bar 8 mm each side)
UNDER_GAP = 0.020  # ... and its top is no more than this below the crate's underside (a bar resting on the
#                    table in a pocket is 12 mm below it)
LIFT_TOL = 0.003  # 'lifted': the crate's lowest point more than this above the bearers it stands on
RACK_SUPPORT_X = 0.400  # the crate stands on the rack's bearers (top 0.072) once its centre is at or past this x;
#                         before it, on the floor bearers (top 0.036)
TILT_MAX_DEG = 5.0  # the crate tilted more than this at any moment: 0
HEIGHT_MAX = 0.095  # the crate's lowest point higher than this above the table at any moment: 0
FALL_SPEED = 0.25  # m/s: the crate's lowest point dropping faster than this is a fall: 0
OFF_TABLE_DROP = 0.010  # the crate's lowest point this far below the table top: it left the table: 0
TOUCH_GAP = 0.005  # a gripper's finger zone this close to the crate: the gripper touched it: 0
ON_RACK_X0, ON_RACK_X1 = 0.400, 0.530  # 'on the rack': the crate's footprint inside this x range (the platform's
#                                        front edge to 10 mm past the bearers' far ends: +-10 mm for arm sag),
ON_RACK_HALF_Y = 0.165  # and within this of y 0,
ON_RACK_Z_TOL = 0.003  # its lowest point within this of the rack bearers' top,
ON_RACK_TILT_DEG = 3.0  # tilted no more than this,
ON_RACK_YAW_DEG = 5.0  # and turned no more than this from its start (either way round)
CARRY_LIFT = 0.020  # 0.5: lifted at least this far above the floor bearers on both bars ...
CARRY_ADVANCE = 0.050  # ... with its centre at least this far toward the rack (+x) from its start
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart): a gripper this open holds nothing
CLEAR_DIST = 0.030  # at the end each gripper's tool point and finger-tip point at least this far from the bars
#                     and the crate
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper (13 mm past the TCP to 25 mm
#                     behind it)
ZONE_RADIUS = 0.024  # the zone's radius, the same at every opening: a finger's outer corner when closed on the 24 mm
#                      bar is hypot(12 + 6, 13.5) = 22.5 mm from the axis (24.1 mm counting 8 mm fingers), whatever
#                      the gripper's turn, which is not read. Fingers opened wider spread along an unknown direction
#                      and are not followed further out (an open gripper 45 mm in front of the crate, its fingers
#                      spread along y, is not touching it)

# Colours (sRGB).
BEARER_GREY = (0.30, 0.30, 0.32)
CRATE_BROWN = (0.55, 0.40, 0.25)
BAR_GREY = (0.60, 0.62, 0.65)
WHITE = (0.95, 0.95, 0.95)
YELLOW = (0.95, 0.80, 0.10)
RACK_BLUE = (0.20, 0.40, 0.80)
RACK_BEARER_BLUE = (0.12, 0.25, 0.55)
