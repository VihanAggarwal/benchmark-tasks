"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
'world'.

A bay walled on three sides (two side walls along x and a far wall along y, grey plywood, 40 mm tall) is open
toward the arms; its floor is the table. Three loose dark-stained pine floorboards (300 x 50 x 16 mm) lie outside
it, two on the left arm's side and one on the right arm's side, lengthwise along x. Laid, they run along y across
the bay, side by side, tight against each other, with a 10-20 mm expansion gap to both side walls and to the far
wall. A 300 mm board in the 330 mm bay is in the window at both side walls only within 5 mm of centred.
"""

TOP = 0.7475  # station table top (world z)

# -- the bay (fixed walls) --------------------------------------------------------------------------------------
WALL_H = 0.040  # every wall's height
WALL_T = 0.020  # every wall's thickness
BAY_HALF = 0.165  # the side walls' inside faces at y +-0.165: the bay is 330 mm wide
BAY_X0 = 0.160  # the bay's open edge: the side walls' near ends
FAR_X = 0.360  # the far wall's inside face
SIDE_WALL_LEN = FAR_X - BAY_X0  # 0.200: each side wall runs from x 0.160 to the far wall's inside face
SIDE_WALL_X = (BAY_X0 + FAR_X) / 2  # 0.260
SIDE_WALL_Y = BAY_HALF + WALL_T / 2  # 0.175
FAR_WALL_X = FAR_X + WALL_T / 2  # 0.370
FAR_WALL_LEN = 2 * (BAY_HALF + WALL_T)  # 0.370: across both side walls' ends
# The walls' inside faces seen from above, as segments ((x0, y0), (x1, y1)): what the expansion gaps are measured to.
FACES = {
    "left_wall": ((BAY_X0, BAY_HALF), (FAR_X, BAY_HALF)),
    "right_wall": ((BAY_X0, -BAY_HALF), (FAR_X, -BAY_HALF)),
    "far_wall": ((FAR_X, -BAY_HALF), (FAR_X, BAY_HALF)),
}

# -- the floorboards (loose, pine stained dark brown) ---------------------------------------------------------
BOARD_SIZE = (0.300, 0.050, 0.016)  # length (the board's own x), width, thickness
BOARD_HALF = tuple(v / 2 for v in BOARD_SIZE)
BOARD_MASS = 0.11  # 240 cm3 of pine at 0.46 g/cm3
BOARD_NAMES = ("board_a", "board_b", "board_c")
# Where each starts: flat on the table, its length along x (yaw 0). a and b in front of the left arm, c in front of
# the right arm; the demo lays a along the far wall, b next to it, c nearest the arms.
BOARD_START = {
    "board_a": (0.250, 0.235),
    "board_b": (0.250, 0.310),
    "board_c": (0.250, -0.235),
}

# -- the rules (scoring.py) --------------------------------------------------------------------------------------
GAP_MIN, GAP_MAX = 0.010, 0.020  # every expansion gap (side walls, far wall), inclusive
WALL_ZERO = 0.003  # a board whose outline comes this close to a wall's inside face (or crosses it): 0
WALL_ZONE_Z = 0.042  # ... counted only while the board's lowest point is at most this high (the walls' tops + 2 mm)
FLAT_DEG = 3.0  # laid: broad faces tilted at most this from level
ON_FLOOR = 0.002  # laid: lowest point at most this above the table
AXIS_DEG = 3.0  # laid: long side within this of y (across the bay)
JOINT_MAX = 0.003  # 1.0: every joint between neighbouring laid boards at most this wide
JOINT_LINES_Y = (0.140, -0.140)  # ... measured along x on these two lines (10 mm in from a centred board's ends)
EPS = 1e-6  # m: limits are compared with a micrometre of slack (a gap of exactly 20.000 mm is in the window)
HANDS_OFF = 0.030  # 1.0: each gripper's finger zone at least this far from every board
STILL_TOL = 0.002  # success: neither end of any board moved more than this ...
STILL_WINDOW_S = 0.5  # ... over the last 0.5 s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a board moving this far in one control step was teleported: a new trial began ...
START_TOL = 0.045  # ... if every board is then within this of its start (covers a seeded reset's jitter)
OFF_TABLE = 0.010  # a board's lowest point this far below the table top: it fell off the table

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open) --
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap over its lowest 38 mm
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there
FINGER_LEN = 0.038  # the finger zone runs from the tips this far back up the gripper
# Where the closed fingers bear on a board when pushing (their lowest 25 mm), for the offline push model:
PAD_THICK = 0.008
PAD_HALF_WIDTH = 0.010
PAD_ABOVE_TCP = 0.025

# -- the offline push model only (offline_push.py; the simulator never uses these) -------------------------------
FRICTION = 1.0  # the simulator's default dynamic friction
GRAVITY = 9.81
PUSH_COMPLIANCE = 0.006  # m/N along x at the push poses (J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad: 5.8-6.4)
PUSH_OPENING_MAX = 0.15  # only fingers this nearly closed push (rigstate's opening, 0 closed .. 1 open)

# Colours (sRGB).
PLYWOOD_GREY = (0.62, 0.62, 0.60)
WALNUT_STAIN = (0.33, 0.20, 0.11)  # dark-stained pine: unlike the oak-coloured station table
