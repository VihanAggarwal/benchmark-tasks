"""Every dimension of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

Parts. (1) Syrup bottle: a 4 fl oz (118 mL) amber HDPE children's liquid-medicine bottle, cap off, 118 mL of syrup in
it, a plain white label round its body. Outside: a round body 49 mm across and 82 mm tall, a neck 28 mm across up to 105
mm; inside 2 mm walls, a 45 mm bore up to 80 mm and a 22 mm opening to the lip at 105 mm (133.6 mL to the brim). (2) Its
child-resistant cap is already off and put away (not on the table, not modelled). (3) Dosing cup: a 30 mL (1 fl oz)
clear, see-through round cup with printed mL graduations and numerals, 40 mm across outside and 38 mm tall, 2 mm walls
and floor (36 mm inside), drawn as 24 flat wall panels in a ring (CUP_SIDES) on a round floor. The clear wall is drawn
by not rendering it at all (each panel's lower part is a hidden collider; nothing has an opacity below 1), with its
rolled lip (the top RIM_H of every panel) drawn opaque pale grey-blue and its 2 mm floor opaque pale clear-plastic
blue; the marks are drawn bolder than real print so the cameras resolve them: rings at 10, 20 and 30 mL (the 10 mL
line red and 6 mm tall, 20 and 30 mL near black and 4 mm tall), each drawn once round the cup (outside on the six
panels of the top camera's half that neither the hand nor a numeral covers, inside on the rest), with a gap at the
four numeral panels; printed seven-segment numerals 10, 20 and 30 (7 mm tall, 10 red, 20 and 30 near black) beside
the lines on the two panels facing each arm (+-y); short near black ticks 1.5 mm tall at 5, 15 and 25 mL outside on
three panels each side (TICK_PANELS_POS). (4) Scale: a pocket scale's white 100 x 100 mm platform, 14 mm above the
table, fixed.
"""

import math

TOP = 0.7475  # station table top

# -- the scale ---------------------------------------------------------------------------------------------------------
PAN_X, PAN_Y = 0.30, 0.0
PAN_SIZE = (0.100, 0.100, 0.014)
PAN_TOP = TOP + PAN_SIZE[2]  # the pan's top face (world z)

# -- the dosing cup (origin: the middle of the cup's bottom face) ---------------------------------------------------------
# Round, as a ring of CUP_SIDES flat wall panels (a 24-gon): the panel faces stand CUP_INNER / 2 (inside) and
# CUP_OUTER / 2 (outside) from the axis, panel 0 faces +x (the hand grips the panels facing +-x).
CUP_SIDES = 24
CUP_OUTER = 0.040  # across the outside faces of opposite panels
CUP_H = 0.038
CUP_WALL = 0.002
CUP_FLOOR = 0.002
CUP_MASS = 0.008
CUP_INNER = CUP_OUTER - 2 * CUP_WALL  # 0.036 across the inside faces
CUP_INNER_HALF = CUP_INNER / 2  # 18 mm: the inside's inscribed radius (the catch test's circle)
CUP_AREA = CUP_SIDES * CUP_INNER_HALF ** 2 * math.tan(math.pi / CUP_SIDES)  # m^2: 10.24 cm^2, 1 mm is 1.024 mL
CUP_FLOOR_R = CUP_OUTER / 2  # the round floor under the ring (20 mm)
CUP_X, CUP_Y, CUP_Z = PAN_X, PAN_Y, PAN_TOP

# -- the syrup bottle (origin: the middle of the bottle's bottom face; axis +z) ----------------------------------------
BODY_R, BODY_H = 0.0245, 0.082
NECK_R, NECK_TOP = 0.014, 0.105
BORE_R, BORE_Z0, BORE_Z1 = 0.0225, 0.002, 0.080  # the body's inside: radius, floor, shoulder
LIP_R = 0.011  # the neck's opening (inside radius); the lip is the circle at NECK_TOP
BOTTLE_MASS = 0.17  # 118 mL x 1.31 g/mL = 155 g of syrup + 16 g of bottle = 171 g (170 g, rounded; fixed in the sim)
BOTTLE_X, BOTTLE_Y = 0.30, -0.18
FILL = 118.0e-6  # m^3 of syrup in the bottle at the start
LABEL_Z0, LABEL_Z1 = 0.016, 0.066  # a plain white paper label round the body (look; 0.3 mm proud, sunk into the body)
LABEL_T = 0.0003
OPENING_T = 0.0006  # the 22 mm opening drawn as a dark disc in the lip, its top 0.1 mm over the lip (look only)


# -- the syrup -----------------------------------------------------------------------------------------------------------
RHO = 1310.0  # kg/m^3: Simple Syrup NF (85 g sucrose per 100 mL; 1.313 g/mL), a sealed, shelf-stable pharmacy syrup
MU = 0.15  # Pa s (150 mPa s)
NU = MU / RHO  # m^2/s
GRAV = 9.81
Q_CAP = 3.0e-6  # m^3/s: what the 22 mm neck passes however deep the syrup stands at the lip (3.0 mL/s)

# -- rules (scoring.py) ----------------------------------------------------------------------------------------------------
DOSE_ML = 10.0
WINDOW = (9.0, 11.0)  # mL in the cup: the real 2 mm print line, +-1 mL (drawn 6 mm tall: LINE_W, window its middle third)
WIDE = (8.0, 12.0)  # mL: a 20 percent dose error
PARTIAL_ML = 4.0
OVERDOSE_ML = 15.0  # more than this in the cup at any moment: 0 (a dose cannot be taken back out)
CLEAN_SPILL_ML = 0.5
SPILL_CAP_ML = 3.0  # above: at most 0.5
SPILL_ZERO_ML = 10.0  # above: 0 (a bottle emptied onto the table)
UNREAD_S = 0.15  # seconds of stream into the cup while it was not held up and level
HELD_CLEAR = 0.020  # 'held up': the cup's lowest point this far above the pan's top
HELD_TILT_DEG = 5.0  # 'level'
CATCH_MARGIN = 0.004  # the stream must fall this far inside the cup's inner walls (within 14 mm of its axis)
CATCH_TILT_DEG = 30.0  # a cup tilted more than this catches nothing
DUMP_TILT_DEG = 45.0  # a cup tilted more than this pours out what it holds (a 10 mL dose spills at 62 degrees)
CUP_SET_XY = 0.015  # the cup 'on the scale': centre within 15 mm of the pan's centre
CUP_SET_Z = 0.003  # ... its bottom within 3 mm of the pan's top
CUP_SET_TILT_DEG = 3.0
BOTTLE_SET_XY = 0.030  # the bottle 'put back': centre within 30 mm of its start mark
BOTTLE_SET_Z = 0.003
BOTTLE_SET_TILT_DEG = 5.0
LET_GO = 0.030  # both grippers' finger zones this far from both parts
START_TOL = 0.004  # a part within 4 mm of its start mark counts as 'at start'
RESET_JUMP = 0.08  # a part moving this far in one control step was teleported: a new episode began
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# The syrup's level in the cup is drawn as a thin opaque golden disc that rides at the ledger's level (display only).
# The cup's wall is drawn clear by not rendering it: each of the 24 panels is two colliders of the same radial span
# and width, a lower part (floor to RIM_H under the rim) that is hidden (mechkit ``hidden=True``: the collider is kept,
# nothing is drawn) and the rolled lip (the top RIM_H) drawn opaque pale grey-blue (CLEAR_EDGE). Nothing in the scene
# has an opacity below 1: fractional opacity (2b, 0.25 per face) rendered as a dithered grey speckle on this stack.
# If a renderer ignores the hide, the lower parts draw in CUP_WALL_FALLBACK, the pan's white, so the cup reads as a
# white cup with crisp rings, never as a grey disc. The 2 mm floor is opaque pale clear-plastic blue (CUP_BASE): the
# cup's base ellipse on the white pan, and it hides the empty syrup disc parked inside it, so an empty cup shows no
# syrup. The board's top camera sees the cup from behind the arms (from -x), about 64 degrees above the table: about
# 0.6 px/mm at 960 x 528 and 1.8 px/mm in the 2880 x 1584 screenshot this task asks for (pipeline/cam), a vertical
# wall about 0.27 (0.8) px per mm of its height. So the graduations it can see are rings round the cup: the 10 mL line
# red and 6 mm tall (the 9 to 11 mL window its middle third), 20 and 30 mL near black and 4 mm tall. Each ring is drawn
# ONCE round the cup, on one face of each panel: outside on the six panels of the top camera's (-x) half that the hand
# does not grip and that carry no numeral (OUTER_RING_PANELS), inside on the other fourteen; the four numeral panels
# carry no ring on either face, so no line strikes through or shows behind a numeral. Drawn on both faces, a ring
# showed twice and the stacked dark lines read as a grey band. Printed numerals 10, 20 and 30 (seven-segment, 7 mm
# tall, 4 mm wide, 1 mm strokes, 10 red, 20 and 30 near black) stand beside the lines on the two panels facing each
# arm (NUMERAL_PANELS: tens digit, units digit, in reading order for a reader outside the cup), for the wrist cameras.
# The 5, 15 and 25 mL ticks are near black strips 1.5 mm tall outside on TICK_PANELS_POS. Every strip is one panel's
# width or less, sunk MARK_EMBED into its (hidden) panel, so a ring reads as a ring of short segments round the curve.
# Display only: the strips are thin colliders of the cup's own body; the disc is a link of the same articulation,
# which never collides with its own body (mechkit: self collisions off). No two faces of the cup coincide: the panels
# stand CUP_SEAM clear of the floor, of each other and of their lips.
CUP_SEAM = 0.0001  # 0.1 mm between the cup's panels and floor, neighbouring panels' inner corners, panels and lips
RIM_H = 0.0025  # the rolled lip: the top 2.5 mm of every panel (z 35.5 to 38.0 mm over the base), drawn opaque
PANEL_L = 2 * CUP_INNER_HALF * math.tan(math.pi / CUP_SIDES) - CUP_SEAM  # 4.64 mm: each panel's width
SYRUP_T = 0.0018  # disc thickness (m): a visible edge; parked, it fits inside the 2 mm floor (0.15 to 1.95 mm)
SYRUP_SINK = 0.00005  # the empty disc's top: 0.05 mm under the floor's top face, hidden in the solid floor
SYRUP_MASS = 0.0005
SYRUP_SPRING = 30.0  # N/m toward the target; sag under its own weight 0.16 mm
SYRUP_RANGE = CUP_H - CUP_SEAM - CUP_FLOOR + SYRUP_SINK  # the disc's travel: its top up to 0.1 mm under the brim
LINE_Z = CUP_FLOOR + DOSE_ML * 1e-6 / CUP_AREA  # the 10 mL line over the cup's base: 11.8 mm
LINE_W, LINE_T = 0.006, 0.0006  # the 10 mL line: drawn 6 mm tall (the real print is 2 mm); outside marks 0.6 mm proud
GRAD_ML = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)  # the printed graduations (mL)
GRAD_Z = tuple(CUP_FLOOR + ml * 1e-6 / CUP_AREA for ml in GRAD_ML)  # over the cup's base: 6.9 to 31.3 mm
RING_ML = (10.0, 20.0, 30.0)  # drawn as whole rings (the top camera's graduations)
TICK_ML = (5.0, 15.0, 25.0)  # drawn as short ticks facing the arms (the wrist cameras' graduations)
RING_W = 0.004  # the 20 and 30 mL rings: drawn 4 mm tall (19.5 to 23.5 mm and 29.3 to 33.3 mm)
TICK_W = 0.0015  # the 5, 15 and 25 mL ticks: drawn 1.5 mm tall (1.1 mm clear of the 6 mm red line)
MARK_L = 0.0044  # every strip's length along its panel (4.4 mm of the 4.64 mm panel: neighbours never touch)
TICK_PANELS = 3  # a tick spans this many panels (45 degrees, about 15 mm of arc outside)
TICK_PANELS_POS = (3, 4, 5, 19, 20, 21)  # the panels carrying the ticks, outside (clear of the numeral panels)
OUTER_RING_SKIP = 1  # outside, the rings leave out the gripped panel and this many on each side of it
MARK_IN_T = 0.0003  # inside marks stand 0.3 mm proud of the inner wall
MARK_EMBED = 0.0003  # every mark is sunk 0.3 mm into its wall: its back face is inside the wall, not on its face
SYRUP_R = CUP_INNER_HALF - MARK_IN_T - CUP_SEAM  # the disc: 17.6 mm radius, 0.1 mm clear of every inside mark
# Printed numerals 10, 20 and 30 beside the lines: seven-segment digits (1 mm strokes), each digit on one panel.
NUM_H = 0.007  # a numeral's height (horizontal strokes centred at zc + 3, zc, zc - 3 mm, each 1 mm tall)
NUM_W = 0.004  # a digit's width (a horizontal stroke's length; the panel is 4.64 mm wide)
NUM_STROKE = 0.001  # stroke width
NUM_PROUD = LINE_T  # 0.6 mm proud of the outside face, sunk MARK_EMBED (0.3 mm) into the panel
NUM_SEG_V = 0.0018  # a vertical stroke's height (0.1 mm clear of the horizontals above and below it)
NUM_SEG_T = 0.0015  # a vertical stroke's centre, this far each side of the digit's centre line
NUM_Z = {10: GRAD_Z[1], 20: GRAD_Z[3], 30: GRAD_Z[5]}  # each numeral centred on its line: 11.8, 21.5, 31.3 mm
# (tens digit panel, units digit panel) on the side facing each arm. For a reader outside the cup the reader's right
# is the panel's +tangential axis (-sin phi, cos phi): from +y, 7 lies toward -x (the reader's right), from -y, 18
# lies toward +x (the reader's right), so each pair reads "10", "20", "30" left to right.
NUMERAL_PANELS = {"+y": (6, 7), "-y": (17, 18)}

# Colours (sRGB).
AMBER = (0.60, 0.30, 0.06)  # the bottle: dark amber plastic (opaque)
SYRUP_GOLD = (1.00, 0.68, 0.10)  # the syrup disc: bright golden, far lighter than the bottle (opaque)
CLEAR_EDGE = (0.62, 0.70, 0.78)  # the cup's rolled lip: pale cool grey-blue, opaque
CUP_BASE = (0.84, 0.88, 0.93)  # the cup's 2 mm floor: pale clear-plastic tint, opaque (hides the parked disc)
INK = (0.05, 0.05, 0.05)  # the 20 and 30 mL rings and numerals, the 5, 15 and 25 mL ticks: near black
LINE_RED = (0.85, 0.05, 0.05)  # the 10 mL line and numeral: a strong red, against the white pan and the syrup
PAN_WHITE = (0.93, 0.93, 0.91)  # the scale's white platform: the clear cup and its marks read against it
CUP_WALL_FALLBACK = PAN_WHITE  # the hidden wall's colour if a renderer ignores the hide: a white cup, never grey
LABEL_WHITE = (0.95, 0.95, 0.92)  # the bottle's plain paper label
OPENING_DARK = (0.22, 0.10, 0.03)  # the bottle's opening seen from above: dark syrup in the neck
