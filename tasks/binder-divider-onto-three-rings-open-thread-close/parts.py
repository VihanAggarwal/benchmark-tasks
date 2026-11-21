"""The shapes of the scene (mechkit boxes), built from layout.py. Base frames: the binder's and the holder's origin
is (XS, 0, TOP) and (SHEET_O x, 0, TOP) with the world's axes; the divider's base frame is the sheet at rest in the
holder (tilted by BETA, baked into its boxes) with its origin at the middle of the punched edge."""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

BIN_O = (L.XS, 0.0, L.TOP)
HOLD_O = (L.SHEET_O[0], 0.0, L.TOP + L.COVER_T)
DIV_O = (L.SHEET_O[0], 0.0, L.TOP + L.SHEET_O[2])


def _quat(rot: np.ndarray):
    return tuple(float(v) for v in mk.matrix_to_quat(rot))


def seg_box(p0, p1, y: float, color=None, wire: float = L.WIRE, grow: float = 0.0008):
    """A wire segment from (x, z) p0 to p1 at height-plane y: a box long along the segment."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dx, dz = p1 - p0
    length = math.hypot(dx, dz) + grow
    ang = math.atan2(dz, dx)
    c = (p0 + p1) / 2
    # a segment rising toward +x needs a negative pitch (positive pitch tips the +x end down)
    return mk.box((float(c[0] - L.XS), y, float(c[1])), (length, wire, wire), color=color, ypr=(0.0, -ang, 0.0))


def half_boxes(half: str, y: float):
    pts = L.half_points(half, L.SEG + 1)
    return [seg_box(pts[i], pts[i + 1], y, L.STEEL) for i in range(L.SEG)]


def bx(x0, x1, y0, y1, z0, z1, color=None):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color=color)


def binder_shapes():
    s = []
    for x0, x1 in ((-L.SPINE_HALF - L.COVER_W, -L.SPINE_HALF), (-L.SPINE_HALF, L.SPINE_HALF),
                   (L.SPINE_HALF, L.SPINE_HALF + L.COVER_W)):
        s.append(bx(x0, x1, -L.HALF_LEN, L.HALF_LEN, 0.0, L.COVER_T, L.BLACK))
    xc = L.X_C - L.XS
    s.append(bx(xc - L.PLATE_HALF_X, xc + L.PLATE_HALF_X, -L.PLATE_HALF_Y, L.PLATE_HALF_Y, L.COVER_T, L.ZB, L.STEEL))
    for y in L.RING_Y:
        s += half_boxes("fixed", y)
        # the movable half's foot block on the plate
        s.append(bx(L.XB - L.XS - 0.004, L.XB - L.XS + 0.004, y - 0.004, y + 0.004, L.ZB - 0.001, L.ZB + 0.002, L.STEEL))
    return s


def gate_shapes():
    s = []
    for y in L.RING_Y:
        s += half_boxes("movable", y)
    x0, x1 = L.LEVER_X[0] - L.XS, L.LEVER_X[1] - L.XS
    y0, y1 = L.LEVER_Y
    xb = L.XB - L.XS
    s.append(bx(xb - 0.006, xb - 0.002, y0, y1, L.ZB, L.ZB + L.LEVER_H - 0.003, L.STEEL))  # the stalk
    s.append(bx(x0, x1, y0, y1, L.ZB + L.LEVER_H - 0.003, L.ZB + L.LEVER_H, L.STEEL))  # the bar (3 mm)
    return s


def gate_joint():
    return mk.hinge("gate", "-Y", (L.XB - L.XS, 0.0, L.ZB), gate_shapes(), upper=L.GATE_UPPER, lower=0.0,
                    mass=L.GATE_MASS, spring=L.GATE_SPRING, damping=L.GATE_DAMP)


def frame_box(u0, u1, v0, v1, w0, w1, color=None):
    """A box given by its extents in the sheet frame, turned into the base frame (at rest)."""
    c = L.frame_to_base((u0 + u1) / 2, (v0 + v1) / 2, (w0 + w1) / 2)
    return mk.box(tuple(float(x) for x in c), (u1 - u0, v1 - v0, w1 - w0), color=color, quat=_quat(L.SHEET_R0))


def divider_shapes():
    h = L.HOLE_SIZE / 2
    hw = L.SHEET_T / 2
    u_half = L.SHEET_U / 2
    hv0, hv1 = L.HOLE_V - h, L.HOLE_V + h
    s = [frame_box(-u_half, u_half, 0.0, hv0, -hw, hw, L.SHEET_COLOR),
         frame_box(-u_half, u_half, hv1, L.SHEET_V, -hw, hw, L.SHEET_COLOR)]
    edges = [-u_half] + sum([[u - h, u + h] for u in L.HOLE_U], []) + [u_half]
    for a, b in zip(edges[0::2], edges[1::2]):
        s.append(frame_box(a, b, hv0, hv1, -hw, hw, L.SHEET_COLOR))
    return s


def holder_shapes():
    """The slanted desk sorter, in the holder frame (origin on the table under the sheet's punched edge)."""
    ox, _, oz = L.SHEET_O
    oz -= L.COVER_T  # the holder's frame sits on the cover
    c = L.HOLDER_COLOR
    hy = L.HOLDER_HALF_Y
    s = [bx(-0.060, 0.030, -hy, hy, 0.0, 0.010, c),  # base plate
         bx(0.003, 0.011, -hy, hy, 0.0, 0.026, c)]  # the lip beyond the sheet's lower edge
    # the backrest: a plate behind the sheet, from just under its lower edge to 140 mm up it
    c_b = L.frame_to_base(0.0, 0.069, -0.004) + np.array([0.0, 0.0, oz])
    s.append(mk.box(tuple(float(x) for x in c_b), (2 * hy, 0.142, 0.005), color=c, quat=_quat(L.SHEET_R0)))
    for u in (-0.140, 0.140):  # two posts under its upper end
        top = float((L.frame_to_base(0.0, 0.120, -0.0065) + np.array([0.0, 0.0, oz]))[2])
        xp = float(L.frame_to_base(0.0, 0.120, -0.004)[0])
        s.append(bx(xp - 0.005, xp + 0.005, u - 0.006, u + 0.006, 0.0, top, c))
    return s
