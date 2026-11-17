"""The parts as lists of mechkit shapes (base frames as described in layout.py). No simulator needed.

The motor part is an octagonal body (64 x 60 mm, 10 mm chamfers, standing for the real part's oval) with a square
socket in its coupling end (41 mm across, 24 mm deep, a 5 mm lead-in funnel at the mouth). Each release button is one
sprung link, a frame round the socket: a cap on its flank and a latch hook on the socket's opposite wall (a wedge with
a ramp on the side the attachment comes from and a flat back face), so pressing the cap moves the hook out of the
lip's way.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk

_R2 = math.sqrt(2.0)


def _walls(y0: float, y1: float, color) -> list:
    """The socket's four walls between y0 and y1: two flank walls (full height) and a top and bottom wall."""
    yc, ln = (y0 + y1) / 2, y1 - y0
    out = []
    for s in (-1, 1):
        out.append(mk.box((s * (L.BORE + L.M_W / 2) / 2, yc, 0.0), (L.M_W / 2 - L.BORE, ln, L.M_H), color))
        out.append(mk.box((0.0, yc, s * (L.BORE + L.M_H / 2) / 2), (2 * L.BORE, ln, L.M_H / 2 - L.BORE), color))
    return out


def _mouth(color) -> list:
    """The socket's mouth, y 0..5 mm: the outer blocks and four 45 degree turned fillers that leave a lead-in funnel
    from a 51 mm square at the face down to the 41 mm bore."""
    f, b = L.FUNNEL, L.BORE
    fc = b + f / 2 + f / 4  # a filler's centre from the axis (24.25 mm): the slope's midpoint pushed into the solid
    fy = f / 2 + f / 4  # and along y (3.75 mm)
    out = []
    for s in (-1, 1):
        out.append(mk.box((s * ((b + f) + L.M_W / 2) / 2, f / 2, 0.0), (L.M_W / 2 - b - f, f, L.M_H), color))
        out.append(mk.box((s * fc, fy, 0.0), (f * _R2, f / _R2, L.M_H), color,
                          ypr=(math.radians(-45.0 * s), 0.0, 0.0)))
        out.append(mk.box((0.0, f / 2, s * ((b + f) + L.M_H / 2) / 2), (2 * b, f, L.M_H / 2 - b - f), color))
        out.append(mk.box((0.0, fy, s * fc), (2 * b, f * _R2, f / _R2), color,
                          ypr=(0.0, 0.0, math.radians(-45.0 * s))))
    return out


def _octagon(y0: float, y1: float, color) -> list:
    """The body behind the socket: an octagon 64 x 60 with 10 mm chamfers (the oval of the real part)."""
    yc, ln = (y0 + y1) / 2, y1 - y0
    c = L.M_CHAMFER
    out = [mk.box((0.0, yc, 0.0), (L.M_W, ln, L.M_H - 2 * c), color),
           mk.box((0.0, yc, 0.0), (L.M_W - 2 * c, ln, L.M_H), color)]
    cx, cz = L.M_W / 2 - c / 2 - c / 4, L.M_H / 2 - c / 2 - c / 4  # a chamfer filler's centre: 24.5 and 22.5 mm
    for sx in (-1, 1):
        for sz in (-1, 1):
            out.append(mk.box((sx * cx, yc, sz * cz), (c * _R2, ln, c / _R2), color,
                              ypr=(0.0, math.radians(45.0 * sx * sz), 0.0)))
    return out


def motor_shapes() -> list:
    return (_mouth(L.M_WHITE) + _walls(L.FUNNEL, L.BORE_DEPTH, L.M_WHITE)
            + _octagon(L.BORE_DEPTH, 0.130, L.M_WHITE) + _octagon(0.130, L.M_LEN, L.M_GREY))


def _hook(side: int) -> list:
    """A latch hook on the socket's ``side`` wall (+1 the +x wall, -1 the -x wall), drawn at rest: a wedge with a ramp
    on the side the attachment comes from (the lip's leading face pushes it out toward the wall) and a flat back face
    that stops the lip's trailing face. The ramp's crest (17.2 mm from the axis) is the tip; the flat block behind it
    stands 0.5 mm further out, so the back face bites 1.3 mm into the lip's 19 mm."""
    b, tip = L.BORE, L.HOOK_TIP_X
    ang = math.radians(L.HOOK_RAMP_DEG)
    run = (b - tip) / math.tan(ang)  # the ramp's length along y
    y_wall = L.HOOK_Y0 - run  # where the ramp meets the socket wall
    ramp_len = math.hypot(b - tip, run)
    t = 0.0025
    n = (math.cos(ang), math.sin(ang))  # the ramp's normal into the hook (x, y) on the +x side
    mid = ((b + tip) / 2, (L.HOOK_Y0 + y_wall) / 2)
    # the flat block behind the crest: its tip face 0.5 mm further out than the crest, so the ramp's crest is the hook's
    # innermost point and the lip passes from the ramp to the hook's flank with no edge to meet (two convex pieces flush
    # at the crest make a seam that the solver can resolve along y, a ledge)
    f0 = tip + L.HOOK_FLAT_RECESS
    flat = mk.box((side * (f0 + 0.024) / 2, (L.HOOK_Y0 + L.HOOK_Y1) / 2, 0.0),
                  (0.024 - f0, L.HOOK_Y1 - L.HOOK_Y0, L.HOOK_Z), L.BTN_BLACK)
    ramp = mk.box((side * (mid[0] + n[0] * t / 2), mid[1] + n[1] * t / 2, 0.0), (t, ramp_len, L.HOOK_Z),
                  L.BTN_BLACK, ypr=(side * ang, 0.0, 0.0))
    return [flat, ramp]


def button_shapes(s: int) -> list:
    """One release button and its latch hook, as one sliding frame round the socket; ``s`` +1 for the button on the
    +x flank, -1 for the one on the -x flank. Drawn at rest (button out, hook in). The frame carries the cap on its
    own flank, a bar inside that flank's wall, a cross bar inside the top wall (+x button) or the bottom wall (-x
    button) and a post inside the opposite flank's wall that carries the hook on the OPPOSITE wall of the socket.
    Pressing the cap moves the whole frame toward the far side, so the hook moves out toward its wall, out of the
    lip's way (a press of 1.8 mm clears it); a stub pushed in moves the hook out the same way (and the cap in), and
    the spring brings both back. Every part of the frame outside the hook stays inside the body's walls (more than
    20.5 mm from the axis), so only the hook ever touches a stub; the frame does not collide with its own body."""
    zc = s * 0.0225  # the cross bar's centre height: z 21..24 mm, inside the top (or bottom) wall
    cap_c = L.M_W / 2 - 0.002 + 0.0025  # the cap is 5 mm thick: 2 mm inside the flank, 3 mm proud
    cap = mk.box((s * cap_c, L.BTN_Y, 0.0), (0.005, L.BTN_Y_LEN, L.BTN_Z_LEN), L.BTN_BLACK)
    bar_y0, bar_y1 = L.HOOK_Y0, L.BTN_Y + 0.010
    bar = mk.box((s * 0.027, (bar_y0 + bar_y1) / 2, 0.0), (0.006, bar_y1 - bar_y0, 0.010), L.BTN_BLACK)
    hy, hl = (L.HOOK_Y0 + L.HOOK_Y1) / 2, L.HOOK_Y1 - L.HOOK_Y0
    x_far, x_near = -s * 0.0215, s * 0.027  # the cross bar runs from the far post (|x| 21.5 mm) to the near bar
    cross = mk.box(((x_far + x_near) / 2, hy, zc), (abs(x_near - x_far), hl, 0.003), L.BTN_BLACK)
    bar_up = mk.box((s * 0.027, hy, (0.0 + zc) / 2), (0.006, hl, abs(zc) + 0.0015), L.BTN_BLACK)
    post = mk.box((-s * 0.02325, hy, (-s * L.HOOK_Z / 2 + zc) / 2), (0.0035, hl, abs(zc) + L.HOOK_Z / 2 + 0.0015),
                  L.BTN_BLACK)
    return [cap, bar, bar_up, cross, post] + _hook(-s)


def stub_shapes(color) -> list:
    """The coupling stub, u 0..21 mm from the shoulder: a 33 mm core (the hooks' groove) and a 38 mm lip."""
    return [mk.box((0.0, L.CORE_LEN / 2, 0.0), (2 * L.CORE_HALF_X, L.CORE_LEN, 2 * L.CORE_HALF_Z), color),
            mk.box((0.0, (L.LIP_U0 + L.STUB_LEN) / 2, 0.0), (2 * L.LIP_HALF, L.STUB_LEN - L.LIP_U0, 2 * L.LIP_HALF),
                   color)]


def _cyl(r, u0, u1, color):
    return mk.cylinder((0.0, (u0 + u1) / 2, 0.0), r, u1 - u0, "Y", color)


def shaft_shapes() -> list:
    return (stub_shapes(L.SH_PLASTIC)
            + [_cyl(L.SH_COLLAR_R, -L.SH_COLLAR_LEN, 0.0, L.SH_PLASTIC),
               _cyl(L.SH_STEM_R, L.SH_STEM_U0, L.SH_STEM_U1, L.SH_STEEL),
               _cyl(L.SH_BELL_R, L.SH_BELL_U0, L.SH_BELL_U1, L.SH_STEEL)])


def _gearbox(color) -> list:
    """The whisk's gearbox, u -60..0 mm: a rounded-square housing 58 mm across its flats (8 mm chamfers on the four long
    edges), so it lies on a flat and the pads close on two flats: the square stub keeps its turn about the axis."""
    w, c = 2 * L.WK_BOX_R, L.WK_BOX_CHAMFER
    yc, ln = -L.WK_BOX_LEN / 2, L.WK_BOX_LEN
    out = [mk.box((0.0, yc, 0.0), (w, ln, w - 2 * c), color), mk.box((0.0, yc, 0.0), (w - 2 * c, ln, w), color)]
    k = w / 2 - c / 2 - c / 4  # a chamfer filler's centre from the axis, in x and in z
    for sx in (-1, 1):
        for sz in (-1, 1):
            out.append(mk.box((sx * k, yc, sz * k), (c * _R2, ln, c / _R2), color,
                              ypr=(0.0, math.radians(45.0 * sx * sz), 0.0)))
    return out


def whisk_shapes() -> list:
    out = stub_shapes(L.WK_GREY) + _gearbox(L.WK_GREY)
    out += [_cyl(r, u0, u1, L.WK_WIRE) for r, u0, u1 in L.WK_PARTS]
    return out


def mat_shapes() -> list:
    return [mk.box((0.0, 0.0, L.MAT_T / 2), (L.MAT_SIZE, L.MAT_SIZE, L.MAT_T), L.MAT_CORK)]
