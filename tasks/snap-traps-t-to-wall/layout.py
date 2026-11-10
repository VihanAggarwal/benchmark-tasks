"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. The scene is a passage at real size (1:1): a vinyl floor
sheet between two skirting boards, the wall above them cut away.
"""

import math

TOP = 0.7475  # station table top

# The passage floor: a thin fixed vinyl sheet between the two skirting boards.
FLOOR_X0, FLOOR_X1 = 0.140, 0.480
FACE_Y = 0.250  # the skirting boards' faces (the walls) at y +-0.250: the floor runs from face to face
FLOOR_SIZE = (FLOOR_X1 - FLOOR_X0, 2 * FACE_Y, 0.002)
FLOOR_X = (FLOOR_X0 + FLOOR_X1) / 2
FLOOR_TOP = TOP + FLOOR_SIZE[2]  # the floor the traps stand on: 2 mm above the table

# Two skirting boards: fixed white boxes along x, standing on the table against the floor sheet's long edges.
SKIRT_THICK = 0.015  # along y: y 0.250 .. 0.265 (left), -0.265 .. -0.250 (right)
SKIRT_H = 0.062  # 60 mm above the floor
SKIRT_Y = FACE_Y + SKIRT_THICK / 2  # centres at y +-0.2575
WALLS = {"left": 1.0, "right": -1.0}  # which way (sign of y) each wall lies from the room

# Droppings patches: thin fixed dark brown sheets on the floor, 20 mm along the wall, 10 mm deep, 5-15 mm from the
# wall's face (centres 10 mm from it). Each wall has two; the x positions differ between walls.
PATCH_SIZE = (0.020, 0.010, 0.0005)
PATCH_FROM_WALL = 0.010  # the patch's centre from the wall's face
PATCHES = {  # name -> (x, wall)
    "patch_left_near": (0.340, "left"),
    "patch_left_far": (0.430, "left"),
    "patch_right_near": (0.330, "right"),
    "patch_right_far": (0.440, "right"),
}


def patch_xy(name: str) -> tuple[float, float]:
    x, wall = PATCHES[name]
    return x, WALLS[wall] * (FACE_Y - PATCH_FROM_WALL)


# The snap traps: loose mechkit compounds (one rigid body each), all alike, at real size (a standard wooden mouse
# snap trap, unarmed: its bail has sprung and lies over the pedal end). Shapes in the trap's own frame: origin at the
# centre of the base's underside, +x along its length toward the end WITHOUT the pedal, +z up out of the base.
TRAP_LEN, TRAP_W, TRAP_BASE_H = 0.098, 0.045, 0.010
TRAP_HALF_LEN = TRAP_LEN / 2
BASE_CENTER, BASE_SIZE = (0.0, 0.0, TRAP_BASE_H / 2), (TRAP_LEN, TRAP_W, TRAP_BASE_H)  # pine
PEDAL_CENTER, PEDAL_SIZE = (-0.0315, 0.0, 0.0115), (0.025, 0.022, 0.003)  # yellow plastic: x -0.044 .. -0.019
SPRING_CENTER, SPRING_SIZE = (0.0, 0.0, 0.0135), (0.008, 0.036, 0.007)  # steel coils and pivot across mid-length
BAIL_SIDE_CENTERS = ((-0.0255, 0.0185, 0.011), (-0.0255, -0.0185, 0.011))  # sprung bail: two wires ...
BAIL_SIDE_SIZE = (0.043, 0.002, 0.002)  # ... x -0.047 .. -0.004
BAIL_BAR_CENTER, BAIL_BAR_SIZE = (-0.046, 0.0, 0.011), (0.002, 0.039, 0.002)  # ... and the cross bar at the pedal end
TRAP_H = SPRING_CENTER[2] + SPRING_SIZE[2] / 2  # 17 mm overall
TRAP_MASS = 0.025
PEDAL_END = (-TRAP_HALF_LEN, 0.0, TRAP_BASE_H / 2)  # the midpoint of the base's end face at the pedal end
TRAPS = ("trap_1", "trap_2", "trap_3", "trap_4")
# Start: lying flat in the middle of the passage, long sides along x, two on each side of the midline. 'yaw' is the
# heading of the trap's +x (its plain end): 0 puts the pedal end toward the arms (-x), pi away from them (+x). The
# nearer two point their pedals toward the arms, the farther two away from them, so the two traps on a side need
# opposite quarter turns to bring their pedal ends round to the wall.
TRAP_START = {  # name -> (x, y, yaw) of the trap's origin
    "trap_1": (0.300, 0.130, 0.0),
    "trap_2": (0.420, 0.130, math.pi),
    "trap_3": (0.300, -0.130, 0.0),
    "trap_4": (0.420, -0.130, math.pi),
}

# Rule thresholds (scoring.py).
TILT_MAX_DEG = 10.0  # 'base down': the base's up axis within this of vertical
FLOOR_TOL = 0.003  # ... and its lowest point within this of the floor's top
SQUARE_MAX_DEG = 15.0  # the pedal end's direction within this of straight at the wall
END_GAP_MAX = 0.008  # the pedal end's nearest point at most this far from the wall's face ...
END_GAP_MIN = -0.003  # ... and not more than this into it (contact overlap counts as touching)
ALONG_MAX = 0.025  # the pedal end's midpoint within this of the patch's centre, along the wall
REST_WINDOW_S = 0.5  # 'still': no tracked point moved more than REST_MOVE in this long
REST_MOVE = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.040  # a trap moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... back onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
VINYL_GREY = (0.72, 0.73, 0.72)
SKIRT_WHITE = (0.95, 0.95, 0.93)
DROPPINGS_BROWN = (0.22, 0.13, 0.07)
PINE = (0.86, 0.70, 0.46)
PEDAL_YELLOW = (0.98, 0.82, 0.08)
STEEL = (0.55, 0.56, 0.58)

# The trap's shapes (centre, size, colour), in its own frame; the base first.
TRAP_SHAPES = (
    (BASE_CENTER, BASE_SIZE, PINE),
    (PEDAL_CENTER, PEDAL_SIZE, PEDAL_YELLOW),
    (SPRING_CENTER, SPRING_SIZE, STEEL),
    (BAIL_SIDE_CENTERS[0], BAIL_SIDE_SIZE, STEEL),
    (BAIL_SIDE_CENTERS[1], BAIL_SIDE_SIZE, STEEL),
    (BAIL_BAR_CENTER, BAIL_BAR_SIZE, STEEL),
)
