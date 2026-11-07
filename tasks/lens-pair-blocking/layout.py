"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. 'h' below is a height above the table top.

Everything is fixed to the table except the two edger blocks. Headings are angles seen from above, measured from +x
anticlockwise (toward +y).
"""

import math

TOP = 0.7475  # station table top
SIDES = ("left", "right")

# -- the job ------------------------------------------------------------------------------------------------------
# Block at the frame's boxing centre: (A + DBL - PD) / 2 temporal of the optical centre (OC), along the 180 line.
EYE_SIZE = 0.054  # A, the frame's eye size (boxed lens width)
BRIDGE = 0.020  # DBL, the distance between lenses
PD = 0.058  # the wearer's distance PD
DECENTRATION = (EYE_SIZE + BRIDGE - PD) / 2  # 0.008

# -- the two layout pads (the blocker's lens tables): fixed white acrylic -------------------------------------
PAD_SIZE = (0.120, 0.120, 0.008)
OC = {"left": (0.300, 0.240), "right": (0.300, -0.240)}  # pad centres = lens centres = optical centres

# -- the two uncut lenses: fixed flat discs on the pads, held as a blocker's lens holder holds them ----------
LENS_R = 0.035  # 70 mm uncut
LENS_T = 0.004
LENS_Z0 = PAD_SIZE[2]  # standing on its pad: bottom 8 mm up
LENS_TOP_H = LENS_Z0 + LENS_T  # 0.012
LENS_TOP_Z = TOP + LENS_TOP_H

# -- the lensmeter marks: thin fixed black inlays, bottom 0.3 mm into the lens, top 0.1 mm proud -------------
# 0.1 mm proud so the ink never shares a plane with the lens top (no flicker in the cameras). A placed block rests on
# the 4 mm line under its long axis (and the dot at its nasal end): it can rock sideways by at most about 0.8 deg
# (atan(0.1 / 7)), inside the 3 deg flatness band. GPU watch: after the release, lenses.<side>.turn_deg and tilt_deg
# in TASK_TRACE must settle (turn under 2 deg per 0.5 s). If they chatter, set MARK_H = LENS_TOP_H - MARK_T / 2
# (marks flush with the lens top) and update the setup text and the tests' REST_Z.
MARK_T = 0.0004
MARK_H = LENS_TOP_H + 0.0001 - MARK_T / 2  # centre h 0.0119 (bottom 11.7 mm, top 12.1 mm)
MARK_TOP_H = MARK_H + MARK_T / 2  # 0.0121
DOT_R = 0.004  # the OC dot, 8 mm across (enlarged for the cameras)
LINE_SIZE = (0.060, 0.004, MARK_T)  # the 180 line, 60 mm long and 4 mm wide, centred on the OC
NASAL_SIZE = (0.010, 0.010, MARK_T)  # the nasal square
NASAL_OFF = 0.026  # its centre this far nasal of the OC along the line (it spans 21 to 31 mm)

# Each lens's 180 line is turned on its pad, its nasal end toward the midline (the two lenses lie as a pair with
# their nasal sides facing). TEMPORAL is the heading of the line's temporal direction: the left lens's line is
# turned 25 deg anticlockwise from +y (heading 115 deg), the right lens's 30 deg clockwise from -y (heading -120 deg).
TEMPORAL_DEG = {"left": 115.0, "right": -120.0}


def temporal(side: str) -> tuple[float, float]:
    """Unit vector seen from above, from the OC toward the lens's temporal edge along its 180 line."""
    a = math.radians(TEMPORAL_DEG[side])
    return (math.cos(a), math.sin(a))


def line_heading_deg(side: str) -> float:
    """The 180 line's heading, folded to 0 .. 180 deg (a line has no direction): left 115, right 60."""
    return TEMPORAL_DEG[side] % 180.0


def along(side: str, dist: float) -> tuple[float, float]:
    """The point ``dist`` temporal of the OC along the line (negative: nasal)."""
    (ox, oy), (ux, uy) = OC[side], temporal(side)
    return (ox + dist * ux, oy + dist * uy)


TARGET = {s: along(s, DECENTRATION) for s in SIDES}  # left (0.2966, 0.2473), right (0.2960, -0.2469)
NASAL_XY = {s: along(s, -NASAL_OFF) for s in SIDES}  # left (0.3110, 0.2164), right (0.3130, -0.2175)

# -- the two edger blocks: loose blue acetal boxes standing on the table, long side along y -------------------
BLOCK_SIZE = (0.018, 0.024, 0.016)  # x, y, z at rest: the 24 mm side stands for the block's locating flat
BLOCK_HALF = tuple(v / 2 for v in BLOCK_SIZE)
BLOCK_MASS = 0.010  # acetal, 1.41 g/cm3: 6.9 cm3 is 9.7 g
# GPU watch: if the demo's blocks land more than 2 mm off with a scatter that PLACE_TRIM (task.py, a fixed
# per-arm bias) cannot remove, raise BLOCK_MASS first (0.03: a weighted block; update task.json's setup and
# the README), before touching the bands.
BLOCKS = {"block_left": "left", "block_right": "right"}  # name -> the side it starts on
BLOCK_START = {"block_left": (0.160, 0.310), "block_right": (0.160, -0.310)}

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------
ON_RADIUS = LENS_R  # 'on L': the block's centre, seen from above, within 35 mm of L's OC ...
ON_HEIGHT = 0.060  # ... and less than 60 mm above L's top
FLAT_DEG = 3.0  # 'flat': its 16 mm axis within this of vertical (either 18 x 24 face down)
SEAT_TOL = 0.002  # 'seated': its lowest corner within this of L's top
STILL_WINDOW_S = 0.5  # 'still': over the last 0.5 s its centre's path ...
STILL_PATH = 0.002  # ... is at most 2 mm and ...
STILL_TURN_DEG = 2.0  # ... it turned at most 2 deg in all (positions and orientations, never velocities)
LET_GO_R = 0.040  # 'let go': no gripper whose TCP is within 40 mm of the block's centre ...
LET_GO_OPENING = 0.5  # ... is less than half open (pads under 47 mm apart)
D_TIGHT, A_TIGHT_DEG = 0.002, 3.0  # 0.5 for the lens: centre within 2 mm of the target, long side within 3 deg
# 0.25 for the lens. 3 mm, not 4: measuring 8 mm along y instead of along the turned line lands 3.46 mm (left,
# 25 deg) or 4.14 mm (right, 30 deg) off, and measuring from the dot's edge instead of its centre 4 mm off; each
# scores 0 on both lenses alike.
D_LOOSE, A_LOOSE_DEG = 0.003, 6.0
LENS_TIGHT, LENS_LOOSE = 0.5, 0.25
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a block moving this far in one control step was teleported ...
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) -----------------------------------------------------------------------------------------------
PAD_WHITE = (0.95, 0.95, 0.95)
LENS_BLUE = (0.78, 0.86, 0.95)
INK = (0.05, 0.05, 0.05)
BLOCK_BLUE = (0.15, 0.35, 0.85)
