"""The parts as lists of mechkit shapes in the unit's base frame (layout.py), drawer shut, latch at rest. No simulator.

The carcass (the MUJI chest) is five panels; the drawer is an inset front, two sides, a back and a bottom, with the
bar handle on its front and the latch's mounting foot on the back of the front; the latch arm and its hook are their own
sprung link on the drawer; the catch (a mounting plate and a hanging lip) is a separate fixed part under the top.
"""

from __future__ import annotations

import layout as L
import mechkit as mk


def _bx(x0, x1, y0, y1, z0, z1, color):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def carcass_shapes() -> list:
    """Top and bottom 280 x 370 x 12, two sides 280 x 161 x 9, a back 352 x 161 x 4 (thicknesses est)."""
    w2, t, ts, h, d = L.W / 2, L.T, L.T_SIDE, L.H, L.D
    return [
        _bx(0.0, d, -w2, w2, 0.0, t, L.PLY),  # bottom
        _bx(0.0, d, -w2, w2, h - t, h, L.PLY),  # top
        _bx(0.0, d, w2 - ts, w2, t, h - t, L.PLY_EDGE),  # left side
        _bx(0.0, d, -w2, -w2 + ts, t, h - t, L.PLY_EDGE),  # right side
        _bx(d - L.T_BACK, d, -w2 + ts, w2 - ts, t, h - t, L.PLY),  # back
    ]


def handle_shapes() -> list:
    """IKEA BAGGANAS: a 143 mm bar 9 mm wide standing 31 mm off the front on two legs at 128 mm centres."""
    bar = _bx(L.HANDLE_X0, L.HANDLE_X1, -L.HANDLE_LEN / 2, L.HANDLE_LEN / 2,
              L.HANDLE_Z - L.HANDLE_W / 2, L.HANDLE_Z + L.HANDLE_W / 2, L.STEEL)
    leg_len = -L.HANDLE_X1
    legs = [mk.cylinder((L.HANDLE_X1 / 2, s * L.HANDLE_HOLES / 2, L.HANDLE_Z), L.HANDLE_POST_R,
                        leg_len, "X", L.STEEL) for s in (-1, 1)]
    return [bar] + legs


def drawer_shapes() -> list:
    """The drawer as drawn shut: inset front 346 x 157 x 15 flush with the carcass, sides 251 x 159 x 6, back
    335 x 159 x 6, 4 mm bottom (inside 335 x 245 x 155, published), the handle, and the latch's mounting foot (the
    part of the latch screwed to the front)."""
    fy, by, st, fb = L.FRONT_W / 2, L.BOX_Y0, L.BOX_SIDE_T, L.FRONT_BACK
    x1 = fb + L.BOX_LEN
    out = [
        _bx(0.0, fb, -fy, fy, L.FRONT_Z0, L.FRONT_Z1, L.PLY),  # front
        _bx(fb, x1, by, by + st, L.BOX_Z0, L.BOX_Z1, L.PLY_EDGE),  # left side
        _bx(fb, x1, -by - st, -by, L.BOX_Z0, L.BOX_Z1, L.PLY_EDGE),  # right side
        _bx(x1 - st, x1, -by, by, L.BOX_Z0, L.BOX_Z1, L.PLY_EDGE),  # back
        _bx(fb, x1 - st, -by, by, L.BOTTOM_Z0, L.BOTTOM_Z0 + L.BOTTOM_T, L.PLY),  # bottom
        _bx(fb, L.TAB_X1, L.LATCH_Y - L.LATCH_HALF_W, L.LATCH_Y + L.LATCH_HALF_W, L.TAB_Z0, L.TAB_Z1,
            L.LATCH_WHITE),  # the latch's foot
    ]
    return out + handle_shapes()


def latch_shapes() -> list:
    """The latch's arm (78 mm, 4 mm thick) and its hook (6 mm long, standing 14 mm above the arm's underside)."""
    y0, y1 = L.LATCH_Y - L.LATCH_HALF_W, L.LATCH_Y + L.LATCH_HALF_W
    return [
        _bx(L.TAB_X1, L.HOOK_X1, y0, y1, L.ARM_Z0, L.ARM_Z1, L.LATCH_WHITE),
        _bx(L.HOOK_X0, L.HOOK_X1, y0, y1, L.ARM_Z1, L.HOOK_Z1, L.LATCH_WHITE),
    ]


def hook_box() -> tuple[tuple, tuple]:
    """The hook's box (centre, size) as drawn: what the forced-through check tests against the lip."""
    y0, y1 = L.LATCH_Y - L.LATCH_HALF_W, L.LATCH_Y + L.LATCH_HALF_W
    return (((L.HOOK_X0 + L.HOOK_X1) / 2, (y0 + y1) / 2, (L.ARM_Z1 + L.HOOK_Z1) / 2),
            (L.HOOK_X1 - L.HOOK_X0, y1 - y0, L.HOOK_Z1 - L.ARM_Z1))


def catch_shapes() -> list:
    """The catch: a 4 mm plate under the top (x 25 to 39 mm) and a 6 mm lip hanging to 159 mm at its back edge."""
    y0, y1 = L.LATCH_Y - L.CATCH_HALF_W, L.LATCH_Y + L.CATCH_HALF_W
    return [
        _bx(L.CATCH_X0, L.CATCH_X1, y0, y1, L.CATCH_PLATE_Z0, L.TOP_Z0, L.LATCH_WHITE),
        _bx(L.LIP_X0, L.CATCH_X1, y0, y1, L.LIP_Z0, L.CATCH_PLATE_Z0, L.LATCH_WHITE),
    ]


def lip_bounds() -> tuple[tuple, tuple]:
    y0, y1 = L.LATCH_Y - L.CATCH_HALF_W, L.LATCH_Y + L.CATCH_HALF_W
    return (L.LIP_X0, y0, L.LIP_Z0), (L.CATCH_X1, y1, L.CATCH_PLATE_Z0)
