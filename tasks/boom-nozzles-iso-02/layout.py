"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

The six nozzle bodies are named from the robot's side of the table: L3, L2, L1 on the left (y > 0, the outer one
first), R1, R2, R3 on the right (the inner one first).
"""

TOP = 0.7475  # station table top

# The sprayer boom: a fixed aluminium bar along y, a bench section of a boom with six nozzle bodies 125 mm apart
# (a real boom spaces them 500 mm: 1:4 spacing).
BOOM_X, BOOM_Y = 0.320, 0.000
BOOM_SIZE = (0.044, 0.680, 0.025)  # 44 mm along x, 680 mm along y, 25 mm tall
BOOM_TOP = TOP + BOOM_SIZE[2]

# Nozzle bodies: black square sockets fixed on the boom's top, four walls 5 mm thick and 8 mm tall round a
# 32 x 32 mm opening (42 x 42 mm outside), centred on the boom's centre line.
BODY_OPEN = 0.032
BODY_WALL = 0.005
BODY_WALL_H = 0.008
BODY_OUT = BODY_OPEN + 2 * BODY_WALL  # 42 mm
WALL_TOP = BOOM_TOP + BODY_WALL_H
BODIES = {  # name: y of the centre (x is BOOM_X)
    "L3": 0.3125,
    "L2": 0.1875,
    "L1": 0.0625,
    "R1": -0.0625,
    "R2": -0.1875,
    "R3": -0.3125,
}
BODY_NAMES = tuple(BODIES)
KEEP = ("L3", "R2")  # the two bodies that start with an 02 fitted: they must end with one
WORK = ("L2", "L1", "R1", "R3")  # the four that need work: 0.25 each


def body_walls(name: str) -> list[tuple[str, float, float, float, float]]:
    """A body's four walls as (object name, x centre, y centre, size x, size y): the front and back walls run the
    full 42 mm along y, the two side walls fit between them."""
    y = BODIES[name]
    off = BODY_OPEN / 2 + BODY_WALL / 2  # 18.5 mm from the body's centre to a wall's centre
    return [
        (f"body_{name}_front", BOOM_X + off, y, BODY_WALL, BODY_OUT),
        (f"body_{name}_back", BOOM_X - off, y, BODY_WALL, BODY_OUT),
        (f"body_{name}_left", BOOM_X, y + off, BODY_OPEN, BODY_WALL),
        (f"body_{name}_right", BOOM_X, y - off, BODY_OPEN, BODY_WALL),
    ]


# Parts trays: fixed white open boxes for the nozzles taken off the boom, one each side: a 100 x 100 x 5 mm floor
# and four walls 5 mm thick standing on it, 25 mm tall overall, so the inside is 90 x 90 mm and 20 mm deep.
TRAY_X = 0.200
TRAYS = {"left": 0.110, "right": -0.110}  # side: y of the centre
TRAY_OUT = 0.100
TRAY_WALL = 0.005
TRAY_FLOOR = 0.005
TRAY_H = 0.025
TRAY_IN = TRAY_OUT - 2 * TRAY_WALL  # 90 mm
TRAY_FLOOR_TOP = TOP + TRAY_FLOOR
TRAY_RIM = TOP + TRAY_H


def tray_parts(side: str) -> list[tuple[str, float, float, float, float, float, float]]:
    """A tray's floor and walls as (object name, x, y, z centre, size x, size y, size z)."""
    y = TRAYS[side]
    wall_h = TRAY_H - TRAY_FLOOR
    zc = TRAY_FLOOR_TOP + wall_h / 2
    off = TRAY_OUT / 2 - TRAY_WALL / 2
    return [
        (f"tray_{side}_floor", TRAY_X, y, TOP + TRAY_FLOOR / 2, TRAY_OUT, TRAY_OUT, TRAY_FLOOR),
        (f"tray_{side}_front", TRAY_X + off, y, zc, TRAY_WALL, TRAY_OUT, wall_h),
        (f"tray_{side}_back", TRAY_X - off, y, zc, TRAY_WALL, TRAY_OUT, wall_h),
        (f"tray_{side}_left", TRAY_X, y + off, zc, TRAY_IN, TRAY_WALL, wall_h),
        (f"tray_{side}_right", TRAY_X, y - off, zc, TRAY_IN, TRAY_WALL, wall_h),
    ]


# Spares pads: fixed grey plates the spare nozzles stand on, one each side.
PAD_X = 0.1775
PADS = {"left": 0.265, "right": -0.265}  # side: y of the centre
PAD_SIZE = (0.100, 0.110, 0.005)
PAD_TOP = TOP + PAD_SIZE[2]

# Nozzles: loose solid acetal (POM) cylinders standing in for nozzle tips, 22 mm across and 32 mm tall, 17 g,
# coloured by ISO 10625 size. Only the colours below appear; 02 is yellow.
NOZZLE_R = 0.011
NOZZLE_H = 0.032
NOZZLE_MASS = 0.017
SIZE_OF = {"orange": "01", "green": "015", "yellow": "02", "lilac": "025", "blue": "03", "red": "04"}
TARGET_SIZE = "02"
SPARE_XS = (0.150, 0.205)  # the two rows on each pad, nearer the arm bases first
SPARE_YS = (0.235, 0.295)  # |y| of the two columns on each pad, nearer the midline first

# (name, colour, x, y, where it starts: a body name or "pad_left" / "pad_right")
NOZZLES = (
    ("yellow_L3", "yellow", BOOM_X, BODIES["L3"], "L3"),
    ("blue_L2", "blue", BOOM_X, BODIES["L2"], "L2"),
    ("lilac_R1", "lilac", BOOM_X, BODIES["R1"], "R1"),
    ("yellow_R2", "yellow", BOOM_X, BODIES["R2"], "R2"),
    ("blue_R3", "blue", BOOM_X, BODIES["R3"], "R3"),
    ("yellow_left_a", "yellow", SPARE_XS[0], SPARE_YS[0], "pad_left"),
    ("red_left", "red", SPARE_XS[0], SPARE_YS[1], "pad_left"),
    ("green_left", "green", SPARE_XS[1], SPARE_YS[0], "pad_left"),
    ("yellow_left_b", "yellow", SPARE_XS[1], SPARE_YS[1], "pad_left"),
    ("orange_right", "orange", SPARE_XS[0], -SPARE_YS[0], "pad_right"),
    ("yellow_right_a", "yellow", SPARE_XS[0], -SPARE_YS[1], "pad_right"),
    ("yellow_right_b", "yellow", SPARE_XS[1], -SPARE_YS[0], "pad_right"),
    ("red_right", "red", SPARE_XS[1], -SPARE_YS[1], "pad_right"),
)
NOZZLE_NAMES = tuple(n[0] for n in NOZZLES)
COLOUR = {n: c for n, c, *_ in NOZZLES}
STARTS = {n: (x, y) for n, _, x, y, _ in NOZZLES}
START_PLACE = {n: w for n, *_, w in NOZZLES}
# Nozzles of a wrong size that start fitted: once off the boom they belong in a tray, not among the spares. (No
# spare is blue or lilac, so in a photo any blue or lilac nozzle on a pad came off the boom.)
OFF_THE_BOOM = ("blue_L2", "lilac_R1", "blue_R3")


def start_bottom(name: str) -> float:
    """Height of a nozzle's bottom at the start: the boom's top in a body, the pad's top on a pad."""
    return PAD_TOP if START_PLACE[name].startswith("pad") else BOOM_TOP


# Rule thresholds (scoring.py).
HOLD_ABOVE = 0.040  # 'in a body / tray / on a pad': lowest point less than this above its floor (boom top, ...)
FIT_TILT_DEG = 10.0  # 'fitted': axis within this of the vertical ...
FIT_BOTTOM = 0.003  # ... lowest point within this of the boom's top ...
STILL_WINDOW_S = 0.5  # ... and still: moved no more than STILL_PATH (path length) over the last 0.5 s
STILL_PATH = 0.002
CAP_MIXED = 0.5  # at most this while any body holds a nozzle of another size
CAP_LOOSE = 0.75  # at most this while any nozzle is anywhere else (or a removed one is back among the spares)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a nozzle moving this much further in one step than its velocity allows was teleported ...
START_TOL = 0.045  # ... and every nozzle within this of its own starting mark: a new episode (+-3 cm seeded jitter)

# The table (yam_arena station): a nozzle whose centre is below its top or outside it has fallen off.
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)

# Colours (sRGB).
RGB = {
    "yellow": (0.95, 0.82, 0.10),
    "blue": (0.10, 0.35, 0.85),
    "lilac": (0.70, 0.55, 0.90),
    "red": (0.85, 0.12, 0.12),
    "green": (0.15, 0.65, 0.25),
    "orange": (0.98, 0.50, 0.08),
}
BOOM_GREY = (0.65, 0.66, 0.68)
BODY_BLACK = (0.07, 0.07, 0.08)
TRAY_WHITE = (0.88, 0.88, 0.86)
PAD_GREY = (0.55, 0.56, 0.58)
