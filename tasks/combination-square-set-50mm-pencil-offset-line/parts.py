"""The parts' shapes (mechkit boxes and cylinders), each in its own base frame (see layout.py). NumPy only."""

from __future__ import annotations

import layout as L
import mechkit as mk


def board_shapes() -> list:
    """The 1-by-6: one 19.05 x 139.7 x 610 mm piece of pine."""
    return [mk.box((0.0, 0.0, L.BOARD_H / 2), (L.BOARD_W, L.BOARD_L, L.BOARD_H), color=L.PINE)]


def square_shapes() -> list:
    """Head (stock) with its locking screw, the blade through it with graduation marks along the blade's top."""
    hd, hl, hh = L.SQ_HEAD_DEPTH, L.SQ_HEAD_LEN, L.SQ_HEAD_H
    shapes = [mk.box((-hd / 2, 0.0, hh / 2), (hd, hl, hh), color=L.HEAD_COLOR),
              mk.cylinder((L.SQ_KNOB_X, 0.0, hh + L.SQ_KNOB_H / 2), L.SQ_KNOB_R, L.SQ_KNOB_H, color=L.KNOB_COLOR)]
    x0, x1 = L.SQ_EXT - L.SQ_BLADE_L, L.SQ_EXT  # the blade's tail end and its end, local x
    zc = L.SQ_BLADE_Z + L.SQ_BLADE_T / 2
    shapes.append(mk.box(((x0 + x1) / 2, 0.0, zc), (L.SQ_BLADE_L, L.SQ_BLADE_W, L.SQ_BLADE_T), color=L.STEEL))
    ztop = L.SQ_BLADE_Z + L.SQ_BLADE_T + 0.00015
    n = int(round(L.SQ_BLADE_L / L.SQ_GRAD))
    for i in range(1, n):
        x = L.SQ_EXT - i * L.SQ_GRAD  # marks count back from the blade's end (the zero of the head's setting is the face)
        if x < x0 + 0.004 or -hd - 0.001 < x < 0.001:
            continue  # nothing under the head
        ln = 0.010 if i % 2 == 0 else 0.006
        shapes.append(mk.box((x, -L.SQ_BLADE_W / 2 + ln / 2 + 0.0005, ztop), (0.0008, ln, 0.0003), color=L.INK_BLACK))
    return shapes


def pencil_shapes() -> list:
    """Flat carpenter's pencil: the lead's stub, the point stepping out on the +x side, the red body."""
    out = []
    t = L.PN_T
    for k in range(4):  # 2 mm steps: the -x face stays flat at local x -1 mm
        width = L.PN_STUB + 0.004 * k
        z0 = k * L.PN_STEP
        color = L.PN_LEAD if k < 2 else L.PN_RED
        out.append(mk.box((L.PN_X0 + width / 2, 0.0, z0 + L.PN_STEP / 2), (width, t, L.PN_STEP), color=color))
    z0 = 4 * L.PN_STEP
    out.append(mk.box((L.PN_CENTER_X, 0.0, (z0 + L.PN_LEN) / 2), (L.PN_W, t, L.PN_LEN - z0), color=L.PN_RED))
    return out


def block_shapes() -> list:
    """The pencil block: floor, four walls round a 21 x 10 mm slot, 45 mm tall overall."""
    hx, hy = L.BL_HALF_X, L.BL_HALF_Y
    sx, sy = L.SLOT_HX, L.SLOT_HY
    zc = L.BL_FLOOR + L.BL_DEPTH / 2
    out = [mk.box((0.0, 0.0, L.BL_FLOOR / 2), (2 * hx, 2 * hy, L.BL_FLOOR), color=L.BLOCK_WOOD)]
    for s in (-1, 1):
        out.append(mk.box((s * (sx + hx) / 2, 0.0, zc), (hx - sx, 2 * hy, L.BL_DEPTH), color=L.BLOCK_WOOD))
        out.append(mk.box((0.0, s * (sy + hy) / 2, zc), (2 * sx, hy - sy, L.BL_DEPTH), color=L.BLOCK_WOOD))
    return out
