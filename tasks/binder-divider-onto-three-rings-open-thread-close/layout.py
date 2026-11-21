"""Every number of the scene, in metres, kg, N and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP ("table frame" in
the comments: z measured from the table top). The binder lies OPEN and flat on the table with its spine along y; the
front cover is on the arms' side (x below the spine), the back cover, which carries the ring mechanism, beyond it.
The rings stand along y (their axis); each ring is drawn in the x-z plane.

Figures marked est in task.json are design values the simulation is built to; the setup's measure-and-accept checks
say how the real parts are matched to them.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station table top (world z)
CONTROL_HZ = 20.0
TABLE_X = (-0.19, 0.91)
TABLE_Y = (-0.65, 0.65)
BELOW_TABLE = 0.003
WINDOW = 20  # steps of the last second
HANDS_OFF = 0.030
STILL_SPEED = 0.005  # m/s

# -- the binder: Avery Durable View 1 in slant-ring binder 17011 (black), open flat -------------------------------
BLACK = (0.05, 0.05, 0.06)
STEEL = (0.78, 0.80, 0.83)
BINDER_FRICTION = (0.5, 0.4)  # vinyl on the table, est
XS = 0.400  # spine centre line (world x)
SPINE_HALF = 0.021  # spine 42 mm wide (est: 1 in rings)
COVER_W = 0.254  # a cover 254 mm wide (est: the 9.88 in insert plus 20 mm of margin)
HALF_LEN = 0.146  # the binder is 292 mm (11.5 in) along the spine: y -0.146 .. +0.146
COVER_T = 0.003  # cover board with its vinyl, est
PLATE_T = 0.004  # the ring mechanism's base plate (steel), est
PLATE_HALF_X = 0.016
PLATE_HALF_Y = 0.127  # 254 mm long, est
ZB = COVER_T + PLATE_T  # ring bases: top of the plate, 7 mm over the table
X_C = XS + 0.030  # the rings' centre line (world x)
RING_Y = (-0.10795, 0.0, 0.10795)  # 4.25 in pitch, the standard US letter punch (ANSI)
HALF_W = 0.013  # a ring is a half-ellipse 26 mm wide (wire centre lines) ...
HALF_H = 0.027  # ... and 27 mm tall over the plate (inside height about 25 mm: the "1 inch" capacity)
WIRE = 0.004  # wire 4 mm (est)
SEG = 7  # boxes per half ring
XB = X_C - HALF_W  # the movable half's foot: the gate's hinge line (parallel to y, at z = ZB)
GATE_MASS = 0.06
OPEN = 0.80  # the gate's open stop (rad): the movable tips 23 mm clear of the fixed tips
GATE_UPPER = 0.86
OPEN_MIN = 0.55  # "open": detent catches the gate beyond this
CLOSED_MAX = 0.10  # "closed": the movable tips are home against the fixed ones
DETENT_CAPTURE = 0.25
GATE_SPRING = 0.05  # N m/rad toward the detent target
GATE_DAMP = 0.05
# the ring-opening thumb lever at the -y end of the plate: a stalk and a bar over the plate's -x edge, on the gate
LEVER_Y = (-0.1455, -0.1245)
LEVER_X = (XB - 0.030, XB - 0.002)  # the bar's x range
LEVER_H = 0.0265  # the bar's top over the hinge line (rest)
LEVER_PRESS_X = XB - 0.022  # where the pad presses
PRESS_Z = 0.0097  # table-frame height of the bar's top at that x once the gate is fully open

# -- the divider: Avery 8-tab plastic divider for US letter sheets ----------------------------------------------
SHEET_U = 0.2794  # 11 in along the punched edge (the hole row), ANSI letter
SHEET_V = 0.2159  # 8.5 in the other way
SHEET_T = 0.002  # 2 mm (real about 0.2 to 0.3 mm), est
SHEET_MASS = 0.05  # est
SHEET_COLOR = (0.95, 0.55, 0.12)
SHEET_FRICTION = (0.35, 0.3)
HOLE_V = 0.012  # hole centres 12 mm in from the punched edge, est
HOLE_SIZE = 0.008  # square 8 mm hole (real: round about 6.4 mm)
HOLE_U = RING_Y
BETA = math.pi / 2 - OPEN  # the sheet's tilt from horizontal when it stands in the holder and when it is threaded
D_VEC = np.array([-math.cos(BETA), 0.0, math.sin(BETA)])  # the sheet's v axis (from the punched edge to the far edge)
N_VEC = np.array([math.sin(BETA), 0.0, math.cos(BETA)])  # the sheet's normal (w axis)
U_VEC = np.array([0.0, 1.0, 0.0])
SHEET_R0 = np.column_stack([U_VEC, D_VEC, N_VEC])  # sheet axes in the base frame at rest
GRIP_V = SHEET_V - 0.012  # the pad tips: 12 mm below the far edge
GRIP_U = 0.120  # the two pinches at u = -120 mm and +120 mm
# Carry sag, measured on the GPU (run of 2026-11-21, job 20261121T091738): held in the two pinches at the 0.80 rad
# tilt, the divider rode 6 mm lower and 1 mm nearer the arms than the commanded pad tips put it (arm sag under load
# plus give in the pinch). Modelled as CARRY_SAG_Z down along world z plus CARRY_SAG_N against the sheet's normal; the
# demo aims the hands that much higher (task.trim) and the offline model sags the held divider by the same amount.
CARRY_SAG_N = 0.0014
CARRY_SAG_Z = 0.0050
SHEET_O = (0.340, 0.0, COVER_T + 0.0107)  # the sheet frame's origin at rest: the middle of the punched edge, table frame

# -- the holder: a slanted desk sorter, 290 mm wide (est) -------------------------------------------------------
HOLDER_COLOR = (0.20, 0.22, 0.28)
HOLDER_FRICTION = (0.5, 0.4)
HOLDER_HALF_Y = 0.150


def frame_to_base(u, v, w) -> np.ndarray:
    """Sheet-frame coordinates (u, v, w) -> the base-frame vector (x, y, z) at rest."""
    return SHEET_R0 @ np.array([u, v, w], dtype=float)


def ellipse_pt(t: float) -> tuple[float, float]:
    """A point of the ring's wire centre line (world x, table-frame z), t=0 the fixed foot, pi the movable foot."""
    return X_C + HALF_W * math.cos(t), ZB + HALF_H * math.sin(t)


def half_points(half: str, n: int = 25) -> np.ndarray:
    """(n, 2) points (x, z) from the half's FOOT to its TIP at the rest pose. 'movable': t pi -> pi/2."""
    ts = np.linspace(math.pi, math.pi / 2, n) if half == "movable" else np.linspace(0.0, math.pi / 2, n)
    return np.array([ellipse_pt(t) for t in ts])


def rotate_gate(xz: np.ndarray, q: float) -> np.ndarray:
    """Points (x, z) of the gate turned q rad opening (counter-clockwise in the x-z view) about the hinge line."""
    c, s = math.cos(q), math.sin(q)
    p = np.asarray(xz, dtype=float) - np.array([XB, ZB])
    return np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]) + np.array([XB, ZB])


def lever_top(q: float) -> float:
    """Table-frame height of the lever bar's top at the pad's x, with the gate at q."""
    x, h = LEVER_PRESS_X - XB, LEVER_H
    return ZB + x * math.sin(q) + h * math.cos(q)


def gate_for_press(tip_z: float) -> float:
    """The gate angle that lets the lever bar's top at the pad's x sit at table-frame height tip_z (monotone)."""
    if tip_z >= lever_top(0.0):
        return 0.0
    lo, hi = 0.0, GATE_UPPER
    if tip_z <= lever_top(hi):
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lever_top(mid) > tip_z else (lo, mid)
    return (lo + hi) / 2


REST_DEPTH = 0.020  # the holes hang this far along the wire from its tip once the divider has slid down the half rings
REST_TILT = math.radians(25.0)  # and the divider leans back this far from horizontal, its far part on the sorter (offline model)


def wire_at(q: float, s: float) -> np.ndarray:
    """The point of the movable half's wire centre line ``s`` m along it from its tip, the gate at q: (x, 0, z) in the
    table frame."""
    pts = rotate_gate(half_points("movable", 120), q)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    acc = 0.0
    for i in range(len(seg) - 1, -1, -1):
        if acc + seg[i] >= s:
            f = (s - acc) / seg[i]
            p = pts[i + 1] + f * (pts[i] - pts[i + 1])
            return np.array([p[0], 0.0, p[1]])
        acc += seg[i]
    return np.array([pts[0][0], 0.0, pts[0][1]])
