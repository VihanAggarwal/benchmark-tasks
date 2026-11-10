"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. The scene is a 1:5 model of a kerb line at a driveway
entrance: the kerb runs along y, the road lies in front of it (+x, away from the robot) and the footway behind it.
The table top is the kerbs' bed.
"""

import math

TOP = 0.7475  # station table top: the bed every kerb unit stands on

# The kerb line: every kerb unit is 25 mm deep (along x), its back face on the line x 0.3725 and its battered face
# toward the road at x 0.3975.
UNIT_DEPTH = 0.025
BACK_X = 0.3725
FACE_X = BACK_X + UNIT_DEPTH  # 0.3975
LINE_X = BACK_X + UNIT_DEPTH / 2  # 0.385: the kerb units' middle

# Fixed parts.
# The road: a thin dark grey asphalt sheet in front of the kerb line, its edge 5 mm in front of the kerb faces.
ROAD_X0, ROAD_X1 = FACE_X + 0.005, FACE_X + 0.085  # 0.4025 .. 0.4825
ROAD_Y = 0.280
ROAD_H = 0.001
# Two full-height kerbs either side of the gap, each 120 mm long: a 19 mm deep back part 30 mm tall and a 6 mm deep
# battered face 26 mm tall (darker), the face on the road side.
LIP_DEPTH = 0.006  # the battered (darker) face strip along the road side of every kerb unit
BACK_DEPTH = UNIT_DEPTH - LIP_DEPTH  # 0.019
LIP_DROP = 0.004  # the face strip is this much lower than the unit's top above it
FULL_H = 0.030
FULL_LEN = 0.120
GAP_HALF = 0.135  # the gap in the kerb line between the two full kerbs: y -0.135 .. +0.135 (270 mm)
FULL_KERBS = {"left": 1.0, "right": -1.0}  # full kerb on that side of the gap: y from +-0.135 to +-0.255
# The crossover's footway paving: a thin red-brown block-paving strip behind the gap's middle two slots, marking the
# width of the driveway entrance; it stops 7.5 mm short of the kerb backs.
PAVING_X0, PAVING_X1 = BACK_X - 0.0575, BACK_X - 0.0075  # 0.315 .. 0.365
PAVING_Y = 0.066
PAVING_H = 0.001

# The four slots in the gap: four 60 mm units with five 6 mm joints (full kerb | S1 | S2 | S3 | S4 | full kerb).
UNIT_LEN = 0.060
JOINT = 0.006
SLOTS = {  # name -> y of the slot's centre
    "S1": GAP_HALF - JOINT - UNIT_LEN / 2,  # +0.099, beside the left full kerb
    "S2": UNIT_LEN / 2 + JOINT / 2,  # +0.033
    "S3": -(UNIT_LEN / 2 + JOINT / 2),  # -0.033
    "S4": -(GAP_HALF - JOINT - UNIT_LEN / 2),  # -0.099, beside the right full kerb
}
# What each slot takes: a low kerb (S2, S3), or a transition rising toward the full kerb beside it (+1: toward +y).
SLOT_WANTS = {"S1": ("transition", 1.0), "S2": ("low", 0.0), "S3": ("low", 0.0), "S4": ("transition", -1.0)}

# Loose kerb units (1:5 models, printed in grey PLA, the battered face strip darker). Each is a mechkit compound,
# one rigid body. Every unit's own frame: origin at the centre of its underside (60 x 25 mm footprint), +x along its
# length, +y toward its battered face, +z up. Laid right, a unit's +y points at the road (+x in the world), so its
# own +x points at -y in the world: its heading (the world angle of its own +x) is -90 degrees.
LOW_H = 0.012  # the dropped (low) kerb: 12 mm tall, face strip 8 mm
STEP_HEIGHTS = (0.012, 0.021, 0.030)  # a transition's three 20 mm segments, low end to high end (back part)
SEG_LEN = UNIT_LEN / 3
LOW_MASS = 0.021  # PLA, 1.24 g/cm3: 16.6 cm3
TRANSITION_MASS = 0.037  # 30.1 cm3
UNITS = ("low_kerb_1", "low_kerb_2", "transition_left", "transition_right")
KIND = {"low_kerb_1": "low", "low_kerb_2": "low", "transition_left": "transition", "transition_right": "transition"}
# Which end of its own x a transition's high (30 mm) end is at: the two transitions are mirror images. Laid with
# its face to the road (heading -90 deg), 'transition_left' rises toward +y (the left full kerb), and
# 'transition_right' toward -y. Turned to rise the other way, each has its face to the footway.
HIGH_END = {"transition_left": -1.0, "transition_right": 1.0}


def unit_shapes(name: str):
    """(centre, size, colour) of each box of this unit, in its own frame; the back part(s) first."""
    back_y = -UNIT_DEPTH / 2 + BACK_DEPTH / 2  # -0.003
    lip_y = UNIT_DEPTH / 2 - LIP_DEPTH / 2  # 0.0095
    if KIND[name] == "low":
        return (
            ((0.0, back_y, LOW_H / 2), (UNIT_LEN, BACK_DEPTH, LOW_H), CONCRETE),
            ((0.0, lip_y, (LOW_H - LIP_DROP) / 2), (UNIT_LEN, LIP_DEPTH, LOW_H - LIP_DROP), FACE_GREY),
        )
    k = HIGH_END[name]
    backs, lips = [], []
    for i, h in enumerate(STEP_HEIGHTS):  # i = 0 at the low end
        cx = k * (-UNIT_LEN / 2 + SEG_LEN / 2 + i * SEG_LEN)
        backs.append(((cx, back_y, h / 2), (SEG_LEN, BACK_DEPTH, h), CONCRETE))
        lips.append(((cx, lip_y, (h - LIP_DROP) / 2), (SEG_LEN, LIP_DEPTH, h - LIP_DROP), FACE_GREY))
    return tuple(backs + lips)


def unit_height(name: str) -> float:
    return LOW_H if KIND[name] == "low" else STEP_HEIGHTS[-1]


# Start: in the stock area near the robot, every unit long side along x. heading = the world angle of the unit's
# own +x. The two low kerbs lie near the arms' own sides, faces outward; the two transitions lie either side of the
# midline, tall ends toward the robot and faces toward each other: each on the side of the OTHER end of the gap.
UNIT_START = {  # name -> (x, y, heading) of the unit's frame origin
    "low_kerb_1": (0.180, 0.240, 0.0),  # face toward +y
    "low_kerb_2": (0.180, -0.240, math.pi),  # face toward -y
    "transition_left": (0.160, -0.050, 0.0),  # tall end toward -x (the robot), face toward +y
    "transition_right": (0.160, 0.050, math.pi),  # tall end toward -x, face toward -y
}
LAID_HEADING = -math.pi / 2  # every unit laid right: face to the road

# Rule thresholds (scoring.py).
TILT_MAX_DEG = 5.0  # 'flat': the unit's up axis within this of vertical
BED_TOL = 0.003  # ... and its lowest point within this of the bed (the table top)
LINE_TOL = 0.005  # both back corners within this of the kerb line (x 0.3725), along x
ALONG_TOL = 0.008  # the unit's middle within this of the slot's centre, along y
FACE_MAX_DEG = 30.0  # the battered face pointing within this of straight at the road (+x)
REST_WINDOW_S = 0.5  # 'still': no tracked point moved more than REST_MOVE in this long
REST_MOVE = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.040  # a unit moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... back onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
ASPHALT = (0.18, 0.18, 0.19)
CONCRETE = (0.72, 0.71, 0.68)
FACE_GREY = (0.42, 0.42, 0.41)
PAVING_RED = (0.62, 0.25, 0.18)
