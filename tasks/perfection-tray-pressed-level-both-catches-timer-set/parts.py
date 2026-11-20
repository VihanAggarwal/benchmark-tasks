"""The parts as lists of mechkit shapes in the unit's base frame (layout.py), every joint at 0: the tray popped up,
level, the dial run out, the switch at START, the pawls resting. No simulator.

The case is a deck with the tray's opening, four walls and the printed marks on the front border; the tray is a
plate with its 25 shape wells drawn on it, two side skirts and a barb on each; the dial is a hub with a grip bar; the
switch is a knob. The catches (a sprung pawl inside each side wall) are a separate fixed part.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


def _bx(x0, x1, y0, y1, z0, z1, color, **kw):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, **kw)


def case_shapes() -> list:
    """Deck 285 x 285 x 6 with a 234 mm opening, four 4 mm walls, the dial's and the switch's printed marks."""
    c2, t, zt = L.CASE / 2, L.WALL_T, L.DECK_Z
    out = list(mk.plate_with_hole((0.0, 0.0, zt - L.DECK_T / 2), (L.CASE, L.CASE), (L.HOLE, L.HOLE), L.DECK_T,
                                  (L.TRAY_XC, 0.0), color=L.CASE_BLUE))
    out += [
        _bx(-c2, c2, c2 - t, c2, 0.0, zt - L.DECK_T, L.CASE_BLUE),  # left wall
        _bx(-c2, c2, -c2, -c2 + t, 0.0, zt - L.DECK_T, L.CASE_BLUE),  # right wall
        _bx(-c2, -c2 + t, -c2 + t, c2 - t, 0.0, zt - L.DECK_T, L.CASE_BLUE),  # front wall
        _bx(c2 - t, c2, -c2 + t, c2 - t, 0.0, zt - L.DECK_T, L.CASE_BLUE),  # back wall
        # printed marks, 0.5 mm proud: the dial's white ring with its 0 mark toward the front, STOP and START tabs
        mk.cylinder((L.DIAL_X, L.DIAL_Y, zt + 0.00025), L.DIAL_R + 0.004, 0.0005, color=L.WHITE),
        _bx(L.DIAL_X - L.DIAL_R - 0.010, L.DIAL_X - L.DIAL_R - 0.005, L.DIAL_Y - 0.002, L.DIAL_Y + 0.002,
            zt, zt + 0.0005, L.WELL_DARK),
        _bx(L.SWITCH_X - 0.012, L.SWITCH_X - 0.007, L.SWITCH_Y + L.SWITCH_STOP - 0.004,
            L.SWITCH_Y + L.SWITCH_STOP + 0.004, zt, zt + 0.0005, L.DIAL_RED),  # STOP
        _bx(L.SWITCH_X - 0.012, L.SWITCH_X - 0.007, L.SWITCH_Y - 0.004, L.SWITCH_Y + 0.004, zt, zt + 0.0005,
            L.WHITE),  # START
        _bx(L.SWITCH_X - 0.004, L.SWITCH_X + 0.004, L.SWITCH_Y - 0.006, L.SWITCH_Y + L.SWITCH_STOP + 0.006,
            zt, zt + 0.0005, L.GREY),  # the switch's slot
    ]
    return out


def slide_shapes() -> list:
    """The slide's carrier: a small block under the tray's centre (the guides), hidden inside the case."""
    x, z = L.TRAY_XC, L.TRAY_TOP0 - L.TRAY_T - 0.010
    return [_bx(x - 0.008, x + 0.008, -0.008, 0.008, z - 0.004, z + 0.004, L.GREY)]


def wells() -> list:
    """The 25 shape wells drawn on the tray, 5 x 5 at 38 mm, alternately a square and a round outline (dark)."""
    out = []
    z = L.TRAY_TOP0 + 0.0003
    for i in range(5):
        for j in range(5):
            x, y = L.TRAY_XC + (i - 2) * 0.038, (j - 2) * 0.038
            if (i + j) % 2 == 0:
                out.append(_bx(x - 0.012, x + 0.012, y - 0.012, y + 0.012, z - 0.0003, z + 0.0003, L.WELL_DARK))
            else:
                out.append(mk.cylinder((x, y, z), 0.013, 0.0006, color=L.WELL_DARK))
    return out


def tray_shapes() -> list:
    """Plate 230 x 230 x 6 with its wells; a skirt down each side, 4 mm thick; a barb on each skirt with a 45 deg
    bevel under it (rides the pawl out) and a flat top (the pawl's tooth holds it)."""
    x0, x1 = L.TRAY_XC - L.TRAY / 2, L.TRAY_XC + L.TRAY / 2
    h = L.TRAY / 2
    top = L.TRAY_TOP0
    out = [_bx(x0, x1, -h, h, top - L.TRAY_T, top, L.TRAY_YELLOW)]
    out += wells()
    bx0, bx1 = L.TRAY_XC - L.BARB_X / 2, L.TRAY_XC + L.BARB_X / 2
    d = 2 * L.BEVEL_HALF / math.sqrt(2.0)  # the bevel: a square ridge turned 45 deg about x
    for s in (1, -1):
        ys = sorted((s * L.SKIRT_Y_IN, s * L.SKIRT_Y_OUT))
        yb = sorted((s * L.BARB_Y_IN, s * L.BARB_Y_OUT))
        out.append(_bx(bx0 - 0.010, bx1 + 0.010, ys[0], ys[1], L.SKIRT_Z0, top - L.TRAY_T, L.TRAY_YELLOW))
        out.append(_bx(bx0, bx1, yb[0], yb[1], L.BARB_Z0, L.BARB_TOP0, L.TRAY_YELLOW))
        yc = s * L.BEVEL_Y
        out.append(mk.box((L.TRAY_XC, yc, L.BARB_Z0), (L.BARB_X, d, d), L.TRAY_YELLOW,
                          ypr=(0.0, 0.0, math.radians(45.0))))
    return out


def barb_box(side: int):
    """(centre, size) of one barb's latch block in the tray link's frame at rest (base frame, tray popped)."""
    yb = sorted((side * L.BARB_Y_IN, side * L.BARB_Y_OUT))
    return ((L.TRAY_XC, (yb[0] + yb[1]) / 2, (L.BARB_Z0 + L.BARB_TOP0) / 2),
            (L.BARB_X, yb[1] - yb[0], L.BARB_TOP0 - L.BARB_Z0))


def dial_shapes() -> list:
    """The dial: a red hub 40 mm across, 8 mm high, a white grip bar across it pointing at 0 (toward the front)."""
    z = L.DECK_Z
    return [mk.cylinder((L.DIAL_X, L.DIAL_Y, z + L.DIAL_HUB_H / 2), L.DIAL_R, L.DIAL_HUB_H, color=L.DIAL_RED),
            mk.box((L.DIAL_X, L.DIAL_Y, z + L.DIAL_HUB_H + L.RIDGE_H / 2), (L.RIDGE_L, L.RIDGE_W, L.RIDGE_H),
                   L.WHITE)]


def switch_shapes() -> list:
    kx, ky, kz = L.KNOB
    return [mk.box((L.SWITCH_X, L.SWITCH_Y, L.DECK_Z + kz / 2), (kx, ky, kz), L.WHITE)]


def pawl_shapes(side: int) -> list:
    """A pawl: a bar hanging from its hinge inside the side wall and a tooth reaching in over the barb line."""
    bx0, bx1 = L.TRAY_XC - L.BARB_X / 2, L.TRAY_XC + L.BARB_X / 2
    yb = sorted((side * L.BAR_Y_IN, side * L.BAR_Y_OUT))
    yt = sorted((side * L.TOOTH_Y_IN, side * L.TOOTH_Y_OUT))
    return [_bx(bx0, bx1, yb[0], yb[1], L.TOOTH_Z0, L.BAR_Z1, L.GREY),
            _bx(bx0, bx1, yt[0], yt[1], L.TOOTH_Z0, L.TOOTH_Z1, L.GREY)]


def tooth_box(side: int):
    yt = sorted((side * L.TOOTH_Y_IN, side * L.TOOTH_Y_OUT))
    return ((L.TRAY_XC, (yt[0] + yt[1]) / 2, (L.TOOTH_Z0 + L.TOOTH_Z1) / 2),
            (L.BARB_X, yt[1] - yt[0], L.TOOTH_Z1 - L.TOOTH_Z0))


def catch_mount_shapes() -> list:
    """The catches' mounting posts under the deck (the fixed part's own base), clear of the tray."""
    out = []
    for s in (1, -1):
        y = sorted((s * 0.126, s * 0.132))
        out.append(_bx(L.TRAY_XC - 0.010, L.TRAY_XC + 0.010, y[0], y[1], 0.0, L.DECK_Z - L.DECK_T - 0.001, L.GREY))
    return out


def lid_shapes() -> list:
    """The game box's lid (open side up, 280 x 280 x 40 est) behind the unit, the 25 shapes in it handles up, ready
    for the next turn: a small plate for each (squares and rounds, as the drawn wells) with a peg handle. Not used."""
    out = list(mk.open_box((0.0, 0.0, 0.020), (0.280, 0.280, 0.040), wall=0.004))
    out = [mk.box(s.center, s.size, L.CASE_BLUE) for s in out]
    colours = (L.DIAL_RED, L.TRAY_YELLOW, L.WHITE, (0.20, 0.65, 0.30), (0.95, 0.55, 0.10))
    for i in range(5):
        for j in range(5):
            x, y = (i - 2) * 0.050, (j - 2) * 0.050
            c = colours[(i + 2 * j) % 5]
            if (i + j) % 2 == 0:
                out.append(mk.box((x, y, 0.0075), (0.022, 0.022, 0.007), c))
            else:
                out.append(mk.cylinder((x, y, 0.0075), 0.012, 0.007, color=c))
            out.append(mk.cylinder((x, y, 0.016), 0.003, 0.010, color=c))
    return out
