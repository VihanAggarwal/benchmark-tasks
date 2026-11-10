"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A 1:10 model of a small stage set for a seated acoustic duo. A dark grey stage deck is fixed on the midline in front
of the arms; the audience is on the arms' side (-x), the back of the stage away from them (+x). Two singers' stools
stand on the deck, the left singer's at y +0.15 and the right singer's at y -0.15, and in front of each (toward the
audience) her mic stand: the left one with a cardioid mic, the right one with a supercardioid, each mic pointing
upstage (+x) at its singer's stool. Two floor wedges (stage monitors) wait in the wings, on the table beside the deck,
one on each side, each with its grille facing its own wing.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the stage deck (fixed) --------------------------------------------------------------------------------------------
DECK_X0, DECK_X1 = 0.140, 0.470  # downstage edge (toward the arms) .. upstage edge
DECK_Y0, DECK_Y1 = -0.340, 0.340  # wide enough for the whole full band on either side of either null
DECK_T = 0.003  # 3 mm hardboard; its top is the stage floor
DECK_TOP_W = TOP + DECK_T

# -- the two mic stands (fixed), the mics and the singers' stools -----------------------------------------------------
# Each stand: a round base, a straight pole and a mic clipped on top, its ball head straight above the pole and its
# handle running back toward the audience (-x), so the mic's front axis points +x, at the singer's stool.
# Distances and angles are measured from the stand's centre (the pole, straight under the mic's head).
# The mics are at a seated singer's mouth height (1.15 m on the real stage). This also keeps them clear of the arm:
# the YAM's wrist housing (link 4) juts about 65 mm ahead of the gripper, 150-300 mm above the table, when the gripper
# sets a wedge straight behind a stand; a mic at a standing singer's height (about 155 mm) would be in its way.
STAND_BASE_R, STAND_BASE_H = 0.015, 0.004
POLE_R = 0.004
CLIP_SIZE = (0.014, 0.014, 0.006)  # a small clip block on the pole's top
HEAD_R = 0.008  # the mic's ball head (grille), straight above the pole
HEAD_Z = DECK_T + 0.115  # the head's centre 115 mm above the deck (0.118 above the table)
POLE_TOP = HEAD_Z - HEAD_R - CLIP_SIZE[2]  # 0.104 above the table
POLE_H = POLE_TOP - DECK_T - STAND_BASE_H  # 0.097
HANDLE_LEN, HANDLE_W = 0.032, 0.009  # the handle runs from the back of the head toward -x
SINGERS = {  # name: (stand centre x, y, the mic's polar pattern, its null(s) in degrees off the mic's front axis)
    "left": (0.390, 0.150, "cardioid", 180.0),
    "right": (0.390, -0.150, "supercardioid", 126.0),
}
MIC_FRONT = 0.0  # both mics point +x (heading 0), at their singers' stools
STOOL_DX = 0.040  # each stool's centre 40 mm upstage of its stand (0.4 m on the real stage)
STOOLS = {name: (x + STOOL_DX, y) for name, (x, y, _, _) in SINGERS.items()}  # (0.430, +-0.150)
STOOL_R, STOOL_H = 0.016, 0.045  # a round wooden stool, 32 mm across and 45 mm tall (a 0.45 m seat)

# -- the two floor wedges: each one loose rigid body (mechkit compound) of four boxes ---------------------------------
# Base frame at the middle of the footprint on the base plane; local +x is the way the grille faces (toward whoever
# should hear it), local +z up. A real wedge's sloped front is stepped here: a full-depth low cabinet slab, a tall
# rear block, and the grey grille covering the step's top and the rear block's front face.
DEPTH, WIDTH, HEIGHT = 0.045, 0.060, 0.030  # 45 deep (along the facing), 60 wide, 30 tall: a 0.45 x 0.6 x 0.3 m wedge
LOW_H = 0.012  # the slab's height (the front of the wedge)
REAR_D = 0.020  # the tall rear block's depth
GRILLE_T = 0.002  # grille thickness
GRILLE_INSET = 0.002  # the grille is 2 mm narrower than the cabinet on each side
X_BACK, X_FRONT = -DEPTH / 2, DEPTH / 2  # -0.0225, +0.0225
X_STEP = X_BACK + REAR_D  # -0.0025: where the rear block ends and the step begins
WEDGE_SHAPES = (  # (part, local centre, size, colour key)
    ("slab", (0.0, 0.0, LOW_H / 2), (DEPTH, WIDTH, LOW_H), "cabinet"),
    ("rear", ((X_BACK + X_STEP) / 2, 0.0, (LOW_H + HEIGHT) / 2), (REAR_D, WIDTH, HEIGHT - LOW_H), "cabinet"),
    ("grille_top", ((X_STEP + X_FRONT) / 2, 0.0, LOW_H + GRILLE_T / 2),
     (X_FRONT - X_STEP, WIDTH - 2 * GRILLE_INSET, GRILLE_T), "grille"),
    ("grille_face", (X_STEP + GRILLE_T / 2, 0.0, (LOW_H + GRILLE_T + HEIGHT - GRILLE_T) / 2),
     (GRILLE_T, WIDTH - 2 * GRILLE_INSET, HEIGHT - 2 * GRILLE_T - LOW_H), "grille"),
)
WEDGE_MASS = 0.08
WEDGES = {  # name: (footprint middle x, y at the start, heading of the grille at the start, rad)
    "wedge_left": (0.220, 0.390, math.pi / 2),  # in the left wing, on the table, grille facing the left wing (+y)
    "wedge_right": (0.220, -0.390, -math.pi / 2),  # in the right wing, grille facing the right wing (-y)
}

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
REST_GAP_TOL = 0.003  # 'standing on the deck': lowest point within 3 mm of the deck's top
LEVEL_DEG = 10.0  # ... base down: its up axis within 10 degrees of vertical
R_FULL = (0.080, 0.150)  # distance from the stand: 0.8-1.5 m for full credit
R_PART = (0.060, 0.200)  # 0.6-2.0 m for part
NULL_FULL_DEG = 15.0  # angle off the null for full credit
NULL_PART_DEG = 30.0
AIM_FULL_DEG = 20.0  # grille's facing to the line to the stool's centre
AIM_PART_DEG = 45.0
SINGER_FULL, SINGER_PART = 0.5, 0.25
STILL_WINDOW_S = 0.5  # 'at rest': neither end of the wedge moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a wedge moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- colours (sRGB) --------------------------------------------------------------------------------------------------
DECK_GREY = (0.28, 0.28, 0.30)
STAND_BLACK = (0.05, 0.05, 0.05)
MIC_COLOURS = {  # (handle, head)
    "left": ((0.06, 0.06, 0.07), (0.36, 0.36, 0.38)),  # black cardioid, dark mesh head
    "right": ((0.72, 0.73, 0.75), (0.84, 0.84, 0.86)),  # silver supercardioid
}
STOOL_WOOD = (0.62, 0.42, 0.22)
CABINET_BLACK = (0.06, 0.06, 0.07)
GRILLE_GREY = (0.62, 0.63, 0.65)
