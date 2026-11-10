"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. The optical bench runs along y, so every distance along it
(object to lens, lens to screen) is a difference in y.
"""

TOP = 0.7475  # station table top

# The optical bench: a fixed strip of grey anodised aluminium along y, 30 cm in front of the arm bases. It holds the
# screen's rider up to 469 mm from the arrow (the rider ends 31 mm beyond the face: y -0.281 with the face at 450).
BENCH_X = 0.300  # its centre line
BENCH_W = 0.060  # across (x): 0.270 .. 0.330
BENCH_H = 0.006
BENCH_Y0, BENCH_Y1 = -0.300, 0.260  # along y (560 mm); the lamp box stands on its +y end
BENCH_TOP = TOP + BENCH_H

# The lens's focal length (marked on its holder), and the optical axis: the height of the arrow's centre, the lens's
# centre and the image's centre above the bench top.
F = 0.100
AXIS_Z = 0.045

# The lamp box (the object): fixed on the bench's +y end. A white slide on its -y face carries a black upward arrow
# 30 mm tall, centred on the axis; the slide's face is the object plane.
OBJ_Y = 0.200
LAMP_SIZE = (0.060, 0.059, 0.076)  # y 0.201 .. 0.260, 76 mm tall on the bench
LAMP_Y = OBJ_Y + 0.001 + LAMP_SIZE[1] / 2
SLIDE_SIZE = (0.040, 0.001, 0.040)  # y 0.200 .. 0.201, z 25-65 mm above the bench
SLIDE_Y = OBJ_Y + SLIDE_SIZE[1] / 2
ARROW_T = 0.0005  # the arrow is printed 0.5 mm proud of the slide
ARROW_PARTS = (  # (name, width along x, bottom, top above the bench top), centred on x: a shaft and a stepped head
    ("arrow_shaft", 0.004, 0.030, 0.052),
    ("arrow_head", 0.012, 0.052, 0.056),
    ("arrow_tip", 0.006, 0.056, 0.060),
)
ARROW_H = 0.030

# The scale: black ticks on the table beside the bench (its -x side), every 50 mm from the object plane, the ones at
# every 100 mm longer, as on a bench's rule. Tick k is k x 50 mm from the object plane.
TICK_STEP = 0.050
TICK_COUNT = 11  # 0 .. 500 mm: y +0.200 .. -0.300
TICK_X1 = BENCH_X - BENCH_W / 2 - 0.005  # 0.265: 5 mm short of the bench's side
TICK_LONG, TICK_SHORT = 0.016, 0.010  # along x
TICK_SIZE_YZ = (0.002, 0.001)

# The lens in its holder: a loose compound on a rider, base frame at the centre of the rider's underside, local x
# across the bench (the holder's width), local y along the optical axis. Rider 50 x 40 x 12; above it a holder
# frame 50 mm wide and 22 mm thick, 80 mm tall (12-92 mm up), with a square window 38 x 38 mm (26-64 mm up)
# filled by the lens (pale blue glass, 0.5 mm proud of each face); the frame's top bar (64-92 mm up) is where it is
# held, above the glass.
RIDER_LENS = (0.050, 0.040, 0.012)
HOLDER_T = 0.022  # thickness along the axis
HOLDER_W = 0.050
HOLDER_Z0, HOLDER_Z1 = 0.012, 0.092
LENS_Z = AXIS_Z  # the lens's centre: on the axis
LENS_SQ = 0.038
LENS_T = 0.023
LENS_Z0, LENS_Z1 = LENS_Z - LENS_SQ / 2, LENS_Z + LENS_SQ / 2  # 0.026 .. 0.064
POST_W = (HOLDER_W - LENS_SQ) / 2  # 0.006: the frame's side posts beside the window
LENS_BOXES = (  # (centre, size) in the lens's base frame; none overlap (PhysX spreads the mass by volume)
    ((0.0, 0.0, RIDER_LENS[2] / 2), RIDER_LENS),  # rider
    ((0.0, 0.0, (HOLDER_Z0 + LENS_Z0) / 2), (HOLDER_W, HOLDER_T, LENS_Z0 - HOLDER_Z0)),  # bottom bar
    ((-(LENS_SQ + POST_W) / 2, 0.0, LENS_Z), (POST_W, HOLDER_T, LENS_SQ)),  # side post
    (((LENS_SQ + POST_W) / 2, 0.0, LENS_Z), (POST_W, HOLDER_T, LENS_SQ)),  # side post
    ((0.0, 0.0, (LENS_Z1 + HOLDER_Z1) / 2), (HOLDER_W, HOLDER_T, HOLDER_Z1 - LENS_Z1)),  # top bar
    ((0.0, 0.0, LENS_Z), (LENS_SQ, LENS_T, LENS_SQ)),  # the lens
)
LENS_GLASS_INDEX = 5
LENS_H = HOLDER_Z1
LENS_MASS = 0.080

# The screen: a loose compound on a rider, base frame at the centre of the rider's underside. Rider 60 x 40 x 12; a
# board 100 mm wide and 100 mm tall (12-112 mm up), 22 mm thick: white-painted with a black 1 mm back. Its white
# face is at local y +0.011 and must face the lamp (+y).
RIDER_SCREEN = (0.060, 0.040, 0.012)
BOARD_W, BOARD_T, BOARD_Z0, BOARD_Z1 = 0.100, 0.022, 0.012, 0.112
BACK_T = 0.001
FACE_Y = BOARD_T / 2  # local y of the white face
SCREEN_BOXES = (
    ((0.0, 0.0, RIDER_SCREEN[2] / 2), RIDER_SCREEN),  # rider
    ((0.0, BACK_T / 2, (BOARD_Z0 + BOARD_Z1) / 2), (BOARD_W, BOARD_T - BACK_T, BOARD_Z1 - BOARD_Z0)),  # white board
    ((0.0, -(BOARD_T - BACK_T) / 2, (BOARD_Z0 + BOARD_Z1) / 2), (BOARD_W, BACK_T, BOARD_Z1 - BOARD_Z0)),  # back
)
SCREEN_WHITE_INDEX = 1
SCREEN_H = BOARD_Z1
SCREEN_MASS = 0.090

ELEMENTS = ("lens", "screen")
RIDERS = {"lens": RIDER_LENS, "screen": RIDER_SCREEN}
BOXES = {"lens": LENS_BOXES, "screen": SCREEN_BOXES}
HEIGHTS = {"lens": LENS_H, "screen": SCREEN_H}

# Starting places (base-frame origins on the bench top, square to it, the screen's white face toward the lamp). The
# lens stands 50 mm from the arrow (inside its focal length: a magnifying-glass position, no real image) and the
# screen's face 280 mm from it (under 4f = 400 mm, the least object-to-screen distance at which a lens of f = 100 mm
# gives a real image on a screen, so no lens position focuses on it).
LENS_START_U = 0.050
SCREEN_START_D = 0.280
START = {"lens": (BENCH_X, OBJ_Y - LENS_START_U), "screen": (BENCH_X, OBJ_Y - SCREEN_START_D - FACE_Y)}
START_YAW = 0.0

# Where the demo sets them: the lens 2f from the arrow, the screen's face 2f beyond the lens.
AIM_U = 2 * F
AIM_D = 4 * F
AIM = {"lens": (BENCH_X, OBJ_Y - AIM_U), "screen": (BENCH_X, OBJ_Y - AIM_D - FACE_Y)}

# Rule thresholds (scoring.py).
STAND_TOL = 0.003  # 'on the bench': the lowest corner of the rider's underside within this of the bench top
TILT_COUNT_DEG = 5.0  # upright
TILT_ZERO_DEG = 10.0  # tipped: score 0
SQUARE_DEG = 5.0  # its plane within this of square across the bench (the screen's white face toward the lamp)
CENTRE_TOL = 0.008  # its centre within this of the bench's centre line (x)
OFF_BENCH_DROP = 0.003  # its centre this far below the bench top: off the bench (on the table): score 0
SHARP_TOL = 0.010  # the screen within this of the image distance v* = f u / (u - f)
SAME_SIZE = (0.9, 1.1)  # magnification v / u
LENS_MARK = (0.190, 0.210)  # the lens's distance from the arrow, at its same-size mark (2f)
SCREEN_MARK = (0.390, 0.410)  # the screen's distance from the arrow, at its same-size mark (4f)
REST_WINDOW_S = 0.5  # 'at rest': no corner of the rider and not its top moved more than REST_MOVE in 0.5 s
REST_MOVE = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.05  # an element moving this far in one control step was teleported: a new episode began
START_TOL = 0.030  # ... onto its starting mark: within this of it

# Colours (sRGB).
ALUMINIUM_GREY = (0.62, 0.64, 0.67)
LAMP_BLACK = (0.08, 0.08, 0.09)
SLIDE_WHITE = (0.95, 0.95, 0.93)
ARROW_BLACK = (0.03, 0.03, 0.03)
TICK_BLACK = (0.05, 0.05, 0.05)
RIDER_BLACK = (0.10, 0.10, 0.11)
HOLDER_GREY = (0.28, 0.29, 0.31)
GLASS_BLUE = (0.62, 0.80, 0.92)
SCREEN_WHITE = (0.96, 0.96, 0.94)
BACK_BLACK = (0.08, 0.08, 0.08)


def image_distance(u: float) -> float:
    """v* = f u / (u - f): where a sharp image forms, for an object u in front of the lens (u > f)."""
    return F * u / (u - F)

