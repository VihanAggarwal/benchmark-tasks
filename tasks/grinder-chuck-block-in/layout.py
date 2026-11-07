"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# The magnetic chuck: a fixed block 200 mm long (along x), 100 mm wide and 25 mm tall, built from its own face
# strips: steel pole pieces separated by three 6 mm non-magnetic brass lines running across it (along y), one every
# 60 mm. A coarse-pole chuck: each inner pole is 54 mm wide, wider than the 40 mm part, so the part can sit on a
# single pole (barely held on a real chuck) or bridge a brass line (held).
CHUCK_X0, CHUCK_X1 = 0.170, 0.370
CHUCK_Y0, CHUCK_Y1 = -0.070, 0.030
CHUCK_H = 0.025
FACE_Z = TOP + CHUCK_H  # the chuck's face
LINE_W = 0.006
LINES_X = (0.210, 0.270, 0.330)  # brass line centres
CHUCK_X = (CHUCK_X0 + CHUCK_X1) / 2
CHUCK_Y = (CHUCK_Y0 + CHUCK_Y1) / 2
CHUCK_W = CHUCK_Y1 - CHUCK_Y0


def pole_spans() -> list[tuple[float, float]]:
    """The steel pole pieces' x spans, between the chuck's ends and the brass lines."""
    edges = [CHUCK_X0]
    for x in LINES_X:
        edges += [x - LINE_W / 2, x + LINE_W / 2]
    edges.append(CHUCK_X1)
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]


# The rail (on a real grinder, the chuck's back rail): a fixed steel bar along the chuck's left long edge, on its
# face, lower than the part.
RAIL_W, RAIL_H = 0.010, 0.005
RAIL_FACE_Y = CHUCK_Y1 - RAIL_W  # y 0.020: the rail's face toward the chuck's middle
RAIL_Y = CHUCK_Y1 - RAIL_W / 2

# The part: a loose steel block, 40 mm long, 30 mm deep, 20 mm tall (body x, y, z), 188 g, blue with layout dye.
# It starts on the table at the left, its long side along y (yaw 90 deg): it has to be turned to lie along the rail.
PART_SIZE = (0.040, 0.030, 0.020)
PART_HALF = tuple(s / 2 for s in PART_SIZE)
PART_MASS = 0.188  # solid steel, 24 cm^3
PART_X0, PART_Y0 = 0.200, 0.260
PART_YAW0 = math.pi / 2

# Blocking pieces: loose steel blocks on the table at the right, long side along x. Two are lower than the part
# (15 mm tall lying flat, 18 mm on their side); two are taller than it whichever way they lie (24 mm at least).
LOW_SIZE = (0.040, 0.018, 0.015)
LOW_MASS = 0.085  # solid steel, 10.8 cm^3
TALL_SIZE = (0.040, 0.024, 0.028)
TALL_MASS = 0.211  # solid steel, 26.9 cm^3
PIECES = (  # (name, kind, x, y)
    ("block_low_1", "low", 0.250, -0.250),
    ("block_low_2", "low", 0.170, -0.310),
    ("block_tall_1", "tall", 0.170, -0.250),
    ("block_tall_2", "tall", 0.250, -0.310),
)
SIZES = {name: (LOW_SIZE if kind == "low" else TALL_SIZE) for name, kind, _, _ in PIECES}
SIZES["part"] = PART_SIZE
PIECE_NAMES = tuple(p[0] for p in PIECES)
LOOSE = ("part", *PIECE_NAMES)
STARTS = {"part": (PART_X0, PART_Y0), **{name: (x, y) for name, _, x, y in PIECES}}

# Rule thresholds (scoring.py).
PART_FACE_TOL = 0.002  # the part on the face: its lowest point within 2 mm of it
PART_TILT_DEG = 3.0  # ... lying on a 40 x 30 mm face, tilted less than this
SQUARE_DEG = 5.0  # its long side within this of the rail's direction
RAIL_GAP = 0.003  # 'against the rail': both corners of the side facing the rail within 3 mm of the rail's face
RAIL_PEN = 0.002  # (a corner more than 2 mm past the rail's face is not against it: it is on the rail)
LINE_STEEL = 0.008  # 'across a brass line': at least this much of the part's length on steel each side of the line
PIECE_FACE_TOL = 0.002  # a blocking piece flat on the face: lowest point within 2 mm of it
PIECE_TILT_DEG = 5.0  # ... resting on one of its faces, tilted less than this
LOW_MAX = 0.019  # a piece counts as blocking only if its top is at most 19 mm above the face (lower than the part)
ON_CHUCK = 0.021  # 'on the chuck': lowest point at most 21 mm above the face (on it, or on something on it)
TOO_TALL = 0.021  # a piece on the chuck whose top is more than 21 mm above the face (the part is 20 mm): 0
END_GAP = 0.003  # 'blocks an end': its nearest point within 3 mm of the part's end face
END_PEN = 0.002  # (and not more than 2 mm into it: it cannot be)
END_PARALLEL_DEG = 10.0  # its face toward the part parallel to the end face within this
END_OVERLAP = 0.010  # covering at least this much of the end face's 30 mm depth
HANDS_OFF = 0.002  # an item a gripper's finger zone comes within this of is still held: it does not count. The
#                    zone is a cylinder round the fingers, flat at their tips, covering both pads at any heading; the
#                    2 mm makes a finger resting on an item count as holding it
STILL_WINDOW_S = 0.5  # 'still': moved no more than STILL_PATH (path length) over the last 0.5 s ...
STILL_PATH = 0.002
STILL_SPEED = 0.05  # ... and slower than this now (m/s); grippers' reported velocities chatter, so the path rules
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # the part moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Colours (sRGB).
POLE_STEEL = (0.62, 0.63, 0.65)
BRASS = (0.62, 0.42, 0.08)  # darker than the steel (about half its luminance): the 6 mm lines show in small frames
RAIL_STEEL = (0.30, 0.31, 0.33)
LAYOUT_BLUE = (0.15, 0.30, 0.72)
BLACK_OXIDE = (0.13, 0.13, 0.14)
