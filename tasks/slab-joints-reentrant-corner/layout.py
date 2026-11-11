"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
'world'.

A 1:25 model of a 100 mm ground slab ready to pour: an L-shaped slab form (a long leg 120 x 460 mm, 3 x 11.5 m, and
a return leg 120 x 100 mm, 3 x 2.5 m, on its left end toward the arms) on a thin stone sub-base, and four loose joint
strips (crack inducers) beside it, two on each side. Rule (contraction joints): lay the strips so that every panel
they mark out is a rectangle no longer than 132 mm (3.3 m) and no more than 1.5 times as long as it is wide. The L has
an inside (re-entrant) corner at (0.260, 0.120): a panel that contains it is L-shaped, so one joint must run out of
that corner and another must cross between the legs; the other two must split the rest of the long leg.
"""

TOP = 0.7475  # station table top (world z)
SCALE = 25  # 1:25

# -- the slab outline: the forms' inside faces ----------------------------------------------------------------------
LONG = (0.260, 0.380, -0.240, 0.220)  # (x0, x1, y0, y1): the long leg, 120 mm along x by 460 mm along y
RETURN = (0.140, 0.260, 0.120, 0.220)  # the return leg, 120 mm along x by 100 mm along y, toward the arms
LEGS = (LONG, RETURN)
CORNER = (0.260, 0.120)  # the inside (re-entrant) corner
BBOX = (0.140, 0.380, -0.240, 0.220)  # the outline's bounding box
AREA = (LONG[1] - LONG[0]) * (LONG[3] - LONG[2]) + (RETURN[1] - RETURN[0]) * (RETURN[3] - RETURN[2])  # 0.0672 m2

# The forms' inside faces as straight walls. FACES_X: faces square to x, (x, y0, y1); FACES_Y: square to y, (y, x0, x1).
FACES_X = (
    (0.380, -0.240, 0.220),  # far form
    (0.260, -0.240, 0.120),  # the long leg's near side, up to the inside corner
    (0.140, 0.120, 0.220),  # the return leg's end
)
FACES_Y = (
    (0.220, 0.140, 0.380),  # left form, along both legs
    (-0.240, 0.260, 0.380),  # right end
    (0.120, 0.140, 0.260),  # the return leg's side, up to the inside corner
)

# -- the forms (fixed boards standing on the table just outside the outline) ----------------------------------------
FORM_T = 0.010  # every board's thickness
FORM_H = 0.015  # every board's height
FORMS = (  # (name, x0, x1, y0, y1): they meet without overlapping
    ("form_far", 0.380, 0.390, -0.250, 0.230),
    ("form_left", 0.130, 0.380, 0.220, 0.230),
    ("form_return_end", 0.130, 0.140, 0.110, 0.220),
    ("form_return_side", 0.140, 0.250, 0.110, 0.120),
    ("form_near", 0.250, 0.260, -0.250, 0.120),
    ("form_right_end", 0.260, 0.380, -0.250, -0.240),
)

# -- the sub-base (fixed, compacted stone, filling the outline) ----------------------------------------------------
SUBBASE_H = 0.001
SUBBASES = (("sub_base_long", *LONG), ("sub_base_return", *RETURN))

# -- the joint strips (loose, rigid PVC bars) ------------------------------------------------------------------------
# Size along their own axes: x the length, y the width, z the thickness. They start lying flat along x.
STRIP_W = 0.018
STRIP_T = 0.012
LONG_LEN = 0.112  # 8 mm shorter than the long leg is wide (120 mm): 4 mm of play at each end
SHORT_LEN = 0.083  # between a strip laid out of the corner and the left form there is 91 mm: 4 mm at each end.
# It falls 18.5 mm short of a form at each end across the long leg (120 mm) and 8.5 mm across the return (100 mm);
# a long strip (112 mm) does not fit across the return leg at all.
LONG_MASS = 0.033  # rigid PVC, 1.38 g/cm3: 112 x 18 x 12 mm is 24.2 cm3
SHORT_MASS = 0.025  # 83 x 18 x 12 mm is 17.9 cm3
STRIPS = {  # name: (length, mass, x, y), each lying flat with its length along x
    "strip_left_long": (LONG_LEN, LONG_MASS, 0.230, 0.340),
    "strip_left_short": (SHORT_LEN, SHORT_MASS, 0.230, 0.400),
    "strip_right_inner": (LONG_LEN, LONG_MASS, 0.230, -0.340),
    "strip_right_outer": (LONG_LEN, LONG_MASS, 0.230, -0.400),
}
STRIP_NAMES = tuple(STRIPS)

# -- the rules (scoring.py) ----------------------------------------------------------------------------------------
FLAT_DEG = 5.0  # counts: lying on an 18 mm face, its 12 mm thickness within this of vertical
ON_BASE = 0.003  # counts: its lowest point at most this above the sub-base's top
SQUARE_DEG = 5.0  # counts: its length, seen from above, within this of x or y
INSIDE_TOL = 0.002  # counts: its outline inside the slab outline, within this
END_TOL = 0.008  # counts: each end within this of a form's inside face or of another counted strip, along the strip
END_PEN = 0.002  # ... an end may be this far into what it meets (contact)
SNAP = 0.006  # a centre line within this of the inside corner's lines is taken to lie on them
MAX_SIDE = 0.132  # a good panel: its longer side at most this (3.0 m plus 10 %: 3.3 m at 1:25) ...
MAX_ASPECT = 1.5  # ... and at most this times its shorter side
PER_PANEL = 0.2  # score per good panel (four strips can make at most five panels)
HANDS_OFF = 0.020  # counts: both grippers' finger zones at least this far from it (let go)
EPS = 1e-6  # m: limits are compared with a micrometre of slack
STILL_TOL = 0.002  # success: no end of any strip moved more than this ...
STILL_WINDOW_S = 0.5  # ... over the last 0.5 s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a strip moving this far in one control step was teleported: a new trial began ...
START_TOL = 0.045  # ... if all four are then within this of their starts (covers a seeded reset's jitter)

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open) --
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap over its lowest 38 mm
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there
FINGER_LEN = 0.038  # the finger zone runs from the tips this far back up the gripper

# Colours (sRGB).
FORM_PLY = (0.36, 0.17, 0.10)  # film-faced formwork plywood, dark red-brown: unlike the oak-coloured table
STONE = (0.80, 0.77, 0.70)  # compacted stone sub-base, light grey-buff
STRIP_GREY = (0.22, 0.24, 0.27)  # dark grey PVC
