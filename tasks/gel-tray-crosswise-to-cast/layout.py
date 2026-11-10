"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A horizontal gel electrophoresis tank stands on the midline with its electrodes at its two ends (y -0.100 black,
y +0.100 red): the run direction is y. Its two side walls run along y; their inner faces are 94 mm apart (x 0.273 and
0.367). A gel tray open at both ends, with a black gasket on each open end, is cast in the tank itself: set crosswise
(its length along x) on the grey platform between the side walls, so its gasketed ends face those walls, and a comb
stands in one of three slots across it (notches in both of its side walls).
"""

TOP = 0.7475  # station table top (world z)

# -- the tank (fixed, clear acrylic) ---------------------------------------------------------------------------------
TANK_X, TANK_Y = 0.320, 0.0
TANK_X0, TANK_X1 = 0.267, 0.373  # outside, along x: 106 mm
TANK_HALF_Y = 0.120  # outside, along y: 240 mm
TANK_WALL = 0.006
TANK_H = 0.045  # walls 45 mm tall
SIDE_IN = (TANK_X0 + TANK_WALL, TANK_X1 - TANK_WALL)  # (0.273, 0.367): the side walls' inner faces, 94 mm apart
# (the 84 mm tray has 5 mm of play at each end: the arms land a part 1-3 mm off its aim)
END_IN = TANK_HALF_Y - TANK_WALL  # 0.114: the end walls' inner faces at y +-0.114
TANK_FLOOR = 0.004

# -- the casting platform (fixed, grey acrylic): the raised middle of the tank floor --------------------------------
PLAT_X0, PLAT_X1 = SIDE_IN  # the full inner width
PLAT_HALF_Y = 0.055  # 110 mm along y
PLAT_TOP = 0.016  # 12 mm tall on the 4 mm floor

# -- the electrodes (fixed): platinum wires on the floor at the two ends, shown as coloured bars ---------------------
ELECTRODE_SIZE = (0.080, 0.004, 0.004)
ELECTRODE_Y = 0.100  # black (cathode) at y -0.100, red (anode) at y +0.100

# -- the gel tray: one loose rigid body (mechkit compound). Base frame: origin at the middle of its underside, local x
#    along its length (gasket to gasket), local y across its width, z up. Open at both ends (local +-x) ------------------
TRAY_LEN = 0.084  # gasket face to gasket face
TRAY_W = 0.064
TRAY_H = 0.018
TRAY_FLOOR = 0.004
TRAY_BODY_LEN = 0.080  # floor and side walls; a 2 mm gasket on each end makes 84
TRAY_WALL = 0.005
WALL_Y = TRAY_W / 2 - TRAY_WALL / 2  # 0.0295: each side wall's centre line
NOTCH_DEPTH = 0.005
NOTCH_BOTTOM = TRAY_H - NOTCH_DEPTH  # 0.013 above the tray's underside
NOTCH_W = 0.022  # the 12 mm comb bar has 5 mm of play a side in it
SLOTS = {"end_minus": -0.025, "middle": 0.0, "end_plus": 0.025}  # slot centres along local x (17 mm in from the
# gasket faces; 3 mm of wall between neighbouring notches, 4 mm beyond the end ones)
END_SLOTS = ("end_minus", "end_plus")
GASKET_SIZE = (0.002, TRAY_W, 0.014)
TRAY_MASS = 0.038  # UV-clear acrylic body 33.8 g + two rubber gaskets 4.3 g
TRAY_START = (0.180, 0.270)  # its centre on the bench, lying lengthwise (length along y)
TRAY_START_YAW = 1.5707963267948966  # local x along world +y

# -- the comb: one loose rigid body (mechkit compound). Base frame: origin at the middle of its bar's underside, local
#    y along the bar, local x across it, z up; the teeth (one plate) hang below the bar's middle ----------------------
COMB_BAR = (0.012, 0.072, 0.018)
COMB_TEETH = (0.003, 0.044, 0.007)
COMB_MASS = 0.020  # black acrylic, 16.5 cm3
COMB_BOX_CENTER = (0.0, 0.0, (COMB_BAR[2] - COMB_TEETH[2]) / 2)  # its bounding box in its own frame
COMB_BOX_HALF = (COMB_BAR[0] / 2, COMB_BAR[1] / 2, (COMB_BAR[2] + COMB_TEETH[2]) / 2)
TRAY_BOX_CENTER = (0.0, 0.0, TRAY_H / 2)
TRAY_BOX_HALF = (TRAY_LEN / 2, TRAY_W / 2, TRAY_H / 2)

# -- the comb stand (fixed, white plastic): two plates with a gap for the teeth; the comb's bar rests on both ---------
STAND_X, STAND_Y = 0.180, -0.270
STAND_PLATE = (0.005, 0.050, 0.018)
STAND_GAP = 0.006  # between the plates' inner faces; the 3 mm teeth stand in it
STAND_PLATE_X = (STAND_X - STAND_GAP / 2 - STAND_PLATE[0] / 2, STAND_X + STAND_GAP / 2 + STAND_PLATE[0] / 2)
COMB_START = (STAND_X, STAND_Y, STAND_PLATE[2])  # the comb's base-frame origin: bar on the plates' tops

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
TRAY_TILT_DEG = 3.0  # 'flat': the tray's floor within 3 degrees of level (open side up)
CORNER_TOL = 0.002  # each bottom corner within 2 mm of the platform's top height
CROSS_DEG = 12.0  # 'crosswise': its length within 12 degrees of x, the line between the side walls (between the
# walls the tray can turn at most about 10 degrees, so every tray down between them passes)
GASKET_MAX = 0.010  # each gasket end within 10 mm of its side wall: the mock-up's whole play (94 - 84 mm)
GASKET_MIN = -0.001  # (a gasket 'into' a wall is solver penetration)
SEAT_Z_TOL = 0.0015  # the bar's underside within 1.5 mm of the notch bottoms, at both side walls
SLOT_X_TOL = 0.006  # the bar within 6 mm of the slot's centre at both walls (the notch leaves 5 mm a side)
COMB_Y_TOL = 0.006  # the comb's middle within 6 mm of the tray's centre line (teeth between the walls: 5 mm a side)
COMB_HEADING_DEG = 10.0  # the bar within 10 degrees of the tray's width
COMB_TILT_DEG = 5.0  # the comb upright relative to the tray
HANDS_OFF = 0.020  # both grippers' finger zones at least 20 mm from a part for it to count as let go
STILL_WINDOW_S = 0.5  # 'at rest': no tracked point of a part moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # the tray moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)
OFFLINE_DROP_TOL = 0.003  # offline_nest: a comb box set within 3 mm of the tray box's top was dropped onto it

# -- the grippers (the hands-off rule; rigstate reports the TCP, the point between the pad tips and the opening) -----
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
CLEAR_ACRYLIC = (0.72, 0.80, 0.86)  # the tank, rendered pale blue-grey
PLATFORM_GREY = (0.45, 0.47, 0.50)
TRAY_CLEAR = (0.86, 0.88, 0.90)  # UV-clear acrylic, rendered pale grey
GASKET_BLACK = (0.05, 0.05, 0.05)
COMB_BLACK = (0.12, 0.12, 0.13)
STAND_WHITE = (0.93, 0.93, 0.91)
CATHODE_BLACK = (0.08, 0.08, 0.08)
ANODE_RED = (0.85, 0.10, 0.08)
