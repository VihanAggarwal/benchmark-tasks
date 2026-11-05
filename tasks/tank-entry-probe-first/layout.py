"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The tank is a scale model of a small confined space (a sump or process tank about 2 m deep, at roughly 1:25): its
depth is split into three equal test bands, the probe is long enough to reach the bottom band while the gripper
holding it stays well above the rim, and the inside is wide enough for an open gripper to work on its floor.
"""

TOP = 0.7475  # station table top

# The tank: four fixed steel walls round a 200 x 360 mm floor (the table top), long side along y, on the midline.
# The entrant (right) gripper works in it at yaw +pi/2, which closes the fingers along x (the GPU-measured convention
# of the reference tasks: yaw 0 closes along y, yaw +-pi/2 along x). Fully open, its two 8 mm pads span 110 mm across
# their outer faces along x and 27 mm along y, so on the part's mark it has 45 mm to each long wall (front and back)
# and 81 mm to the right-hand end wall. 80 mm deep: a gripper on the floor (TCP 17 mm up) has 63 mm of itself below
# the rim. Centred at x 0.26 so the part's mark is 6.7 cm inside the right arm's reach toward the midline at height.
TANK_X, TANK_Y = 0.26, 0.0
TANK_INNER_X, TANK_INNER_Y = 0.200, 0.360
TANK_WALL = 0.008
TANK_DEPTH = 0.080
RIM_Z = TOP + TANK_DEPTH
TANK_IN_HX, TANK_IN_HY = TANK_INNER_X / 2, TANK_INNER_Y / 2  # 0.100, 0.180
TANK_OUT_HX, TANK_OUT_HY = TANK_IN_HX + TANK_WALL, TANK_IN_HY + TANK_WALL  # 0.108, 0.188

# The dropped part: a loose steel cube on the tank floor, toward its right-hand end (22.5 cm from the probe's
# sampling point along y, so the two grippers never come within 20 cm of each other in y).
PART_EDGE = 0.030
PART_MASS = 0.21  # 27 cm3 of steel (7.85 g/cm3)
PART_X, PART_Y = TANK_X, -0.085

# The probe wand: a loose upright cylinder (aluminium tube, the sensor head at its lower end), standing in a fixed
# square holder collar outside the tank on the left. It is gripped with the pad tips 30 mm below its top, so with
# its lower end in the bottom band the pad tips are 120 mm above that end: 55 mm above the rim.
PROBE_RADIUS = 0.015
PROBE_LEN = 0.150
PROBE_MASS = 0.12  # a 30 mm tube with 2 mm walls (70 g) and a 50 g sensor head
PROBE_GRIP_DOWN = 0.030  # pad tips this far below the probe's top when it is gripped
PROBE_X, PROBE_Y = 0.20, 0.25  # the holder's centre (the probe's starting place)
SAMPLE_X, SAMPLE_Y = TANK_X, 0.14  # where the demo lowers the probe: 25 mm of floor between it and the left wall

# The probe holder: four fixed walls round a 50 x 50 mm opening (10 mm of play round the probe), 30 mm tall.
HOLDER_INNER = 0.050
HOLDER_WALL = 0.006
HOLDER_HEIGHT = 0.030

# The parts tray: four fixed walls round an 80 x 80 mm floor (the table top), outside the tank on the right.
TRAY_X, TRAY_Y = 0.24, -0.29
TRAY_INNER = 0.080
TRAY_WALL = 0.006
TRAY_HEIGHT = 0.025
TRAY_TOP_Z = TOP + TRAY_HEIGHT

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis)
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper
TIPS_BELOW_TCP = 0.013

# Rule thresholds (scoring.py). Heights of the probe's lower end ('tip') are measured from the tank floor.
BAND_TOP = (0.055, 0.075)  # 'top': 5 to 25 mm below the rim
BAND_MID = (0.030, 0.050)  # 'middle': within 10 mm of half depth
BAND_BOTTOM = (0.005, 0.025)  # 'bottom': 5 to 25 mm above the floor, not resting on it
BANDS = (BAND_TOP, BAND_MID, BAND_BOTTOM)
BAND_NAMES = ("top", "middle", "bottom")
DWELL_S = 2.0  # the tip held in a band this long ...
DWELL_BOX = 0.010  # ... staying inside a 10 mm box (its height range and its x and y ranges each <= 10 mm)
LOWER_HALF = TANK_DEPTH / 2  # while the right gripper is in the tank the tip must be below half depth (40 mm)
IN_TANK_MARGIN = 0.010  # a gripper is 'in the tank' when its fingers are below the rim within 10 mm of the outline
HOLD_OPENING = 0.5  # 'held': fingers closed to 47 mm or less ...
HOLD_SLACK = 0.005  # ... with the point between the pad tips inside the object's box grown by 5 mm
GRIPPER_OPEN = 0.9  # 'let go': opening 0.9 (84.5 mm) or more, ...
LET_GO_GAP = 0.010  # ... or the finger zone more than 10 mm from the object
OUT_CLEAR = 0.010  # the part is 'out of the tank' once its lowest point is 10 mm above the rim (or it is off the outline)
DROP_HEIGHT = 0.010  # the probe free of both grippers, out of the tank, its lowest point higher than this: dropped
TOPPLE_DEG = 45.0  # the probe tilted more than this at any moment: it fell over (or was laid down)
UPRIGHT_DEG = 10.0  # 'standing': tilted less than this ...
REST_Z_TOL = 0.005  # ... with its lowest point within 5 mm of the table top
HOLDER_TOL = 0.012  # 'in its holder': the probe's centre within 12 mm of the holder's centre along x and along y
PART_START_TOL = 0.030  # the part within this of its mark counts as 'at the start' (reset detection)
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported

# Colours (sRGB).
TANK_GREY = (0.55, 0.58, 0.60)
STEEL = (0.35, 0.36, 0.38)
PROBE_ORANGE = (0.95, 0.50, 0.10)
HOLDER_BLACK = (0.12, 0.12, 0.12)
TRAY_BLUE = (0.20, 0.40, 0.85)
