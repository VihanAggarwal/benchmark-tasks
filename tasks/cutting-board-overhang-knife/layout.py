"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A fixed laminate-topped worktop stands on the table; its right-hand end (y -0.20) is the walkway edge, with open air
beside it down to the table. A maple cutting board (John Boos 18 x 13 x 0.75 in, 3.75 lb) lies on the worktop
pulled 130 mm out past that edge, as it was left after scraps were swept off it into a bowl held under the edge. An
8 in chef's knife (Victorinox Fibrox Pro class: 335 mm overall, 200 mm blade 53 mm deep, 165 g) lies flat at the
back of the worktop, its edge facing away from the walkway.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
IN = 0.0254
LB = 0.45359237

# -- the worktop (fixed) -----------------------------------------------------------------------------------------------
WT_X, WT_Y = 0.300, 0.140  # centre of its footprint
WT_LX, WT_LY, WT_H = 0.400, 0.680, 0.090  # 400 mm deep (x), 680 mm long (y), top 90 mm above the table (est)
EDGE_Y = WT_Y - WT_LY / 2  # -0.200: the walkway edge (the worktop's right-hand end)
WT_X0, WT_X1 = WT_X - WT_LX / 2, WT_X + WT_LX / 2
WT_Y1 = WT_Y + WT_LY / 2  # 0.480: the left-hand end (against the wall in the story; not a walkway)
WT_GREY = (0.78, 0.77, 0.74)  # laminate

# -- the cutting board (loose) -----------------------------------------------------------------------------------------
# John Boos maple board 18 x 13 x 0.75 in, 3.75 lb (Sur La Table listing). Long side along y.
# Base frame at the centre of its underside.
BOARD_LX, BOARD_LY, BOARD_H = round(13 * IN, 4), round(18 * IN, 4), round(0.75 * IN, 5)  # 330.2 x 457.2 x 19.05 mm
BOARD_MASS = round(3.75 * LB, 3)  # 1.701 kg
OVERHANG = 0.130  # the board's end hangs 130 mm past the walkway edge at the start
BOARD_X = WT_X
BOARD_Y0 = EDGE_Y - OVERHANG + BOARD_LY / 2  # start centre y: its end at y -0.330
BOARD_TOP = WT_H + BOARD_H  # 109.1 mm above the table
MAPLE = (0.86, 0.72, 0.52)

# -- the chef's knife (loose) ------------------------------------------------------------------------------------------
# Victorinox Fibrox Pro 8 in (5.2063.20 class): blade 200 mm, overall 335 mm, blade height 53 mm, spine 2.3 mm, 165 g
# (Cutlery and More listing). The handle's section (26 mm across, 20 mm thick) and the blade's outline are est.
# Base frame at the heel of the blade (where blade meets handle), on the surface it lies on; local x along the knife
# toward the tip, local y from the spine side (-) toward the edge (+), z up. Lying flat on its side, both the handle
# and the blade rest on the surface (est: a real knife's blade sits a few mm off the surface at the heel).
BLADE_L, BLADE_H, BLADE_T = 0.200, 0.053, 0.0023
HANDLE_L, HANDLE_W, HANDLE_T = 0.135, 0.026, 0.020  # 335 overall minus the 200 mm blade; section est
KNIFE_L = HANDLE_L + BLADE_L  # 335 mm
KNIFE_MASS = 0.165
SPINE_Y = -HANDLE_W / 2  # the blade's spine runs in line with the handle's spine side
HEEL_LEN, TIP_LEN, TIP_H = 0.120, 0.080, 0.030  # the blade as two boxes: the full-depth heel, a shallower tip (est)
KNIFE_BOXES = (  # (name, centre, size) in the knife's frame
    ("handle", (-HANDLE_L / 2, 0.0, HANDLE_T / 2), (HANDLE_L, HANDLE_W, HANDLE_T)),
    ("heel", (HEEL_LEN / 2, SPINE_Y + BLADE_H / 2, BLADE_T / 2), (HEEL_LEN, BLADE_H, BLADE_T)),
    ("tip", (HEEL_LEN + TIP_LEN / 2, SPINE_Y + TIP_H / 2, BLADE_T / 2), (TIP_LEN, TIP_H, BLADE_T)),
)
HANDLE_BLACK = (0.10, 0.10, 0.11)
STEEL = (0.76, 0.77, 0.79)
# where the knife starts: lying along x, tip pointing +x (away from the robot), edge facing +y (away from the walkway)
KNIFE_HEEL_X, KNIFE_HANDLE_Y = 0.265, 0.400


def knife_volume_com_x() -> float:
    """The knife's centre of mass along its axis (from the heel, m) with mass spread by volume, as PhysX spreads it."""
    vols = [(c[0], s[0] * s[1] * s[2]) for _, c, s in KNIFE_BOXES]
    return sum(x * v for x, v in vols) / sum(v for _, v in vols)


GRIP_FROM_HEEL = -0.033  # the demo grips the handle here: at the simulated balance point (knife_volume_com_x)
GRIP_PAD_Z = 0.006  # pad tips 6 mm above the surface the knife lies on

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
EDGE_MARGIN = 0.030  # every part of the knife and of the board at least 30 mm inside the walkway edge (top view)
SAFE_Y = EDGE_Y + EDGE_MARGIN  # -0.170
ON_TOL = 0.002  # the knife's outline may read 2 mm past the board's outline (contact, rounding)
REST_GAP = 0.005  # "resting on": the lowest point within 5 mm above the support's top
FALL_DROP = 0.010  # latched: a part's lowest point more than 10 mm below the worktop's top (it went over an edge)
TIP_DEG = 20.0  # latched: the board tilted more than 20 degrees
KNIFE_FLAT_DEG = 10.0  # the knife lies flat on the board: tilt under 10 degrees
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.003
STILL_TURN_DEG = 3.0
LET_GO = 0.030  # both grippers' finger zones at least 30 mm from the knife and the board
CONTROL_HZ = 20.0
RESET_JUMP = 0.08
START_TOL = 0.02
EPS = 1e-9


def knife_corners_local():
    """The 8 corners of each knife box in the knife's frame: (24, 3)."""
    pts = []
    for _, c, s in KNIFE_BOXES:
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    pts.append((c[0] + sx * s[0], c[1] + sy * s[1], c[2] + sz * s[2]))
    return pts


def board_corners_local():
    return [(sx * BOARD_LX, sy * BOARD_LY, (sz + 0.5) * BOARD_H)
            for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]


def knife_outline_halfwidths():
    """(x0, x1, y0, y1) of the knife's outline in its own frame."""
    xs = [c[0] + s * sz[0] / 2 for _, c, sz in KNIFE_BOXES for s in (-1, 1)]
    ys = [c[1] + s * sz[1] / 2 for _, c, sz in KNIFE_BOXES for s in (-1, 1)]
    return min(xs), max(xs), min(ys), max(ys)


def deg(r):
    return math.degrees(r)
