"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A taped corrugated carton (a stock 7 x 4 x 3 inch box set on its 7 x 3 inch side: 178 mm along y, 76 mm along x,
102 mm tall, 0.30 kg) stands on the table. A glazed 5 x 7 inch picture frame (outer 205 x 155 x 18 mm, landscape,
0.25 kg) stands on its bottom edge in front of it, glass toward the robot, its back leaning 15 degrees from vertical
on the carton's top near edge. Behind them a fixed board stands for the room's wall; to the right a low fixed shelf.

The frame's own frame ("U", upright): origin at the middle of its bottom BACK edge, x_U from the glass toward the
back (the back face is x_U = 0, the glass side x_U = -T), y_U along its width, z_U up its height. The task's frame
body is declared upright in the world with its shapes drawn turned by LEAN0 about y (top toward +x), so at spawn
the body's origin is the bottom back edge, on the table, and its rotation is the identity.
"""

import math

import numpy as np

import mechkit as mk

TOP = 0.7475  # station table top (world z)
DEG = math.pi / 180.0

# -- the carton --------------------------------------------------------------------------------------------------------
# Base frame at the centre of its bottom face, local x across its 76 mm, local y along its 178 mm, z up.
CARTON_X, CARTON_Y, CARTON_H = 0.076, 0.178, 0.102  # Uline S-22632, 7 x 4 x 3 in, set on its 7 x 3 in side
CARTON_MASS = 0.30  # est: a small carton of packed household items
CARTON_START = (0.310, 0.000)
CARTON_NEAR_X = CARTON_START[0] - CARTON_X / 2  # 0.272: its face toward the robot

# -- the picture frame ----------------------------------------------------------------------------------------------
FR_W, FR_H, FR_T = 0.205, 0.155, 0.018  # outer width (y), height, depth: a compact 5 x 7 in frame (about 6 x 8 in)
FR_RAIL = 0.014  # moulding width: leaves a 177 x 127 mm opening
FR_MASS = 0.25  # 0.50 to 0.80 lb listed for 5 x 7 frames (0.23 to 0.36 kg)
GLASS_T = 0.002
GLASS_X = (-0.009, -0.007)  # x_U span of the glass (9 mm behind the moulding's front lip)
BACKING_X = (-0.007, 0.0)  # mat, photo and hardboard back, its back flush with the moulding's
LEAN0 = 15.0 * DEG  # start: leaning 15 degrees from vertical on the carton's top near edge
START_GAP = 0.0005  # the back face spawns 0.5 mm off the carton's edge
FRAME_Y = 0.070  # the frame's middle along y (it overhangs the carton's left end)
PIVOT_X = CARTON_NEAR_X - CARTON_H * math.tan(LEAN0) - START_GAP / math.cos(LEAN0)  # its bottom back edge, about 0.2447

# -- the wall and the shelf (fixed) -------------------------------------------------------------------------------
WALL_FACE_X = 0.420
WALL_T = 0.012
WALL_Y = (0.080, 0.420)
WALL_H = 0.180  # the modelled part of the wall: kept below the gripper's finger carriages
SHELF_C = (0.300, -0.300)
SHELF_LX, SHELF_LY, SHELF_T = 0.150, 0.250, 0.018
SHELF_TOP = 0.060
SHELF_BLOCK_Y = 0.030  # the two end blocks under the board
SHELF_X = (SHELF_C[0] - SHELF_LX / 2, SHELF_C[0] + SHELF_LX / 2)
SHELF_YR = (SHELF_C[1] - SHELF_LY / 2, SHELF_C[1] + SHELF_LY / 2)

# -- where the demo puts things ------------------------------------------------------------------------------------
REST_Y = 0.250  # the picture's middle when it leans on the wall
REST_BACK_X = 0.385  # its bottom back edge, stood upright 35 mm in front of the wall, then tipped onto it

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
FALL_SPEED = 0.3  # m/s: the picture's centre of mass dropping faster than this between two control steps (15 mm in
# 50 ms). A fall from the carton peaks at 1.0 m/s and gives 0.53 to 0.77 m/s over 50 ms whatever its timing; a hand
# setting it down, or tipping it the last few degrees onto the wall, stays under 0.1 m/s.
CONTROL_HZ = 20.0
LEAN_MIN, LEAN_MAX = 5.0, 25.0  # deg: leaning on the wall
WALL_TOUCH = 0.010  # its top back edge within 10 mm of the wall's face
FLAT_DEG = 10.0  # lying flat: its face within 10 degrees of horizontal
ON_TABLE = 0.005  # its lowest point within 5 mm of the table
CLEAR_OF_CARTON = 0.005  # not touching the carton: 5 mm or more between them
SHELF_Z_TOL = 0.005  # the carton's base corners within 5 mm of the shelf's top
SHELF_TILT = 10.0
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
STILL_TURN = 1.0
LET_GO = 0.030  # both grippers' finger zones at least 30 mm from the picture and the carton
OFF_TABLE_DROP = 0.010
TABLE_X0, TABLE_X1 = 0.36 - 0.55, 0.36 + 0.55
TABLE_Y0, TABLE_Y1 = -0.65, 0.65
RESET_JUMP = 0.05
START_TOL = 0.02
EPS = 1e-9

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
KRAFT = (0.70, 0.55, 0.36)
TAPE = (0.80, 0.66, 0.44)
WALNUT = (0.30, 0.19, 0.11)
GLASS = (0.74, 0.83, 0.86)
BACKING = (0.42, 0.33, 0.26)
WALL_WHITE = (0.90, 0.89, 0.86)
SHELF_WOOD = (0.62, 0.47, 0.30)


def ry(a: float) -> np.ndarray:
    """Rotation about +y by ``a`` (a positive angle takes +z toward +x: the top leans back, toward the wall)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


R0 = ry(LEAN0)  # U frame -> the frame body's frame


def frame_boxes_u():
    """The frame as boxes in its U frame: (centre, size, colour). Four moulding rails the full 18 mm deep, the glass,
    and the photo with its hardboard back."""
    w2, h, t, r = FR_W / 2, FR_H, FR_T, FR_RAIL
    inner_w, inner_h = FR_W - 2 * r, FR_H - 2 * r
    out = [
        ((-t / 2, 0.0, h - r / 2), (t, FR_W, r), WALNUT),  # top rail
        ((-t / 2, 0.0, r / 2), (t, FR_W, r), WALNUT),  # bottom rail
        ((-t / 2, w2 - r / 2, h / 2), (t, r, inner_h), WALNUT),  # left rail
        ((-t / 2, -w2 + r / 2, h / 2), (t, r, inner_h), WALNUT),  # right rail
        ((sum(GLASS_X) / 2, 0.0, h / 2), (GLASS_X[1] - GLASS_X[0], inner_w, inner_h), GLASS),
        ((sum(BACKING_X) / 2, 0.0, h / 2), (BACKING_X[1] - BACKING_X[0], inner_w, inner_h), BACKING),
    ]
    return out


def frame_shapes():
    """The frame body's shapes: the U boxes turned by LEAN0 about y (pitch, top toward +x)."""
    return [mk.box(tuple(R0 @ np.array(c)), size, col, ypr=(0.0, LEAN0, 0.0)) for c, size, col in frame_boxes_u()]


def frame_com_u() -> np.ndarray:
    """Centre of mass in the U frame, the mass spread over the boxes' volume (as PhysX does)."""
    vols = np.array([s[1][0] * s[1][1] * s[1][2] for s in frame_boxes_u()])
    cs = np.array([s[0] for s in frame_boxes_u()])
    return (vols[:, None] * cs).sum(axis=0) / vols.sum()


COM_U = frame_com_u()  # about (-6.5 mm, 0, 77.5 mm): the back panel puts it 2.5 mm behind the middle
TILT_BALANCE = math.atan2(-COM_U[0], COM_U[2])  # 4.8 deg: tipped back past this it falls over its bottom back edge


def frame_corners_u() -> np.ndarray:
    xs, ys, zs = (-FR_T, 0.0), (-FR_W / 2, FR_W / 2), (0.0, FR_H)
    return np.array([(x, y, z) for x in xs for y in ys for z in zs])


def frame_samples_u() -> np.ndarray:
    """Points over the frame's outer box surface (U frame), 4 mm apart up its height, for the gap to the carton."""
    pts = []
    for x in (-FR_T, -FR_T / 2, 0.0):
        for y in np.linspace(-FR_W / 2, FR_W / 2, 11):
            for z in np.linspace(0.0, FR_H, 40):
                pts.append((x, y, z))
    return np.array(pts)


def carton_shapes():
    """The carton in its own frame (origin at the middle of its bottom): one box, and a strip of tape over its top
    seam (0.5 mm proud: drawn, not felt; it is part of the same body)."""
    return [
        mk.box((0.0, 0.0, CARTON_H / 2), (CARTON_X, CARTON_Y, CARTON_H), KRAFT),
        mk.box((0.0, 0.0, CARTON_H + 0.00025), (0.048, CARTON_Y, 0.0005), TAPE),
    ]


def carton_corners_local(z: float = 0.0) -> np.ndarray:
    return np.array([(sx * CARTON_X / 2, sy * CARTON_Y / 2, z) for sx in (-1, 1) for sy in (-1, 1)])


def carton_samples_local() -> np.ndarray:
    """Points over the carton's outer surface (its own frame), about 5 mm apart round its outline, for the gap."""
    pts = []
    for x in np.linspace(-CARTON_X / 2, CARTON_X / 2, 16):
        for y in np.linspace(-CARTON_Y / 2, CARTON_Y / 2, 9):
            for z in np.linspace(0.0, CARTON_H, 21):
                pts.append((x, y, z))
    return np.array(pts)
