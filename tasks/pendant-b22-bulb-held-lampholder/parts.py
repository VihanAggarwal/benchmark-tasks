"""The parts as lists of mechkit shapes. No simulator needed.

The pendant's shapes are in its base frame (origin on the table under the rose's axis, z up, heights above the table
top), drawn at rest: the holder hanging straight, mouth down, its J-slot A's hook toward +x. The other parts are in
their own frames: the bulb's origin at the centre of its cap's end (z toward the cap, x through pin A), the stands'
origins on the table under their posts.

The holder is a round wall of 12 flat slabs (38 mm across outside, a 28 mm bore) with a 45-degree funnel at the mouth,
a 25.5 mm collar at the top of the bore (chamfered underneath, 1.75 mm of play round the 22 mm cap) that keeps the
cap's end near the axis, and a solid roof 7 mm over the plungers' tips. The
two plungers hang from the roof, 11 mm apart. Each J-slot's hook is a catch (a small block, the hook's floor) with two
1 mm lips either side of the pin's notch; at rest the catch sits back in the wall, out of the pins' way.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


def _ring(r_in: float, r_out: float, z0: float, z1: float, n: int, color, cx=0.0, cy=0.0, phase=0.0) -> list:
    """A round wall of ``n`` flat slabs, ``r_in`` to ``r_out`` from the axis (cx, cy), from z0 to z1."""
    width = 2.0 * r_out * math.tan(math.pi / n) + 0.0005  # closes the outside corners
    rc = (r_in + r_out) / 2
    out = []
    for i in range(n):
        phi = phase + 2.0 * math.pi * i / n
        out.append(mk.box((cx + rc * math.cos(phi), cy + rc * math.sin(phi), (z0 + z1) / 2),
                          (r_out - r_in, width, z1 - z0), color, ypr=(phi, 0.0, 0.0)))
    return out


def _funnel(r_low: float, z_low: float, rise: float, n: int, color, t: float = 0.002) -> list:
    """A 45-degree chamfer ring: its inner face runs from radius ``r_low`` at ``z_low`` inward and up by ``rise``
    (both ways), the material behind it (outward and up)."""
    run = rise * math.sqrt(2.0)
    rm, zm = r_low - rise / 2, z_low + rise / 2  # the inner face's middle
    rc, zc = rm + t / 2 / math.sqrt(2.0), zm + t / 2 / math.sqrt(2.0)
    width = 2.0 * (r_low + t) * math.tan(math.pi / n) + 0.0005
    out = []
    for i in range(n):
        phi = 2.0 * math.pi * i / n
        out.append(mk.box((rc * math.cos(phi), rc * math.sin(phi), zc), (run, width, t), color,
                          ypr=(phi, math.radians(45.0), 0.0)))
    return out


# -- the stand that stands in for the ceiling ----------------------------------------------------------------------
def stand_shapes() -> list:
    """In the stand's frame: origin on the table under the post."""
    x0 = L.BEAM_X0 - L.POST_X
    x1 = L.POST_W / 2
    return [
        mk.box((0.0, 0.0, L.FOOT[2] / 2), L.FOOT, L.STEEL),
        mk.box((0.0, 0.0, L.BEAM_Z1 / 2), (L.POST_W, L.POST_W, L.BEAM_Z1), L.STEEL),
        mk.box(((x0 + x1) / 2, 0.0, (L.BEAM_Z0 + L.BEAM_Z1) / 2), (x1 - x0, L.BEAM_W, L.BEAM_Z1 - L.BEAM_Z0),
               L.STEEL),
    ]


# -- the pendant (base frame: under the rose's axis) ---------------------------------------------------------------
def rose_shapes() -> list:
    return [mk.cylinder((0.0, 0.0, (L.ROSE_Z0 + L.ROSE_Z1) / 2), L.ROSE_R, L.ROSE_Z1 - L.ROSE_Z0, "Z", L.WHITE)]


SWING_HUB_Z = (L.SWING_Z - 0.004, L.SWING_Z)
FLEX_Z = (L.HOLDER_TOP + 0.020, L.SWING_Z - 0.004)  # the swinging flex
SLACK_Z = (L.HOLDER_TOP + 0.010, L.HOLDER_TOP + 0.020)  # its lowest bit, which slides up when the holder is pushed
GRIP_Z = (L.HOLDER_TOP, L.HOLDER_TOP + 0.010)  # the holder's cord grip


def _cyl(z0: float, z1: float, r: float, color) -> object:
    return mk.cylinder((0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, "Z", color)


def swing_hub_shapes() -> list:
    return [_cyl(*SWING_HUB_Z, 0.004, L.FLEX_WHITE)]


def flex_shapes() -> list:
    return [_cyl(*FLEX_Z, L.FLEX_R, L.FLEX_WHITE)]


def slack_shapes() -> list:
    return [_cyl(*SLACK_Z, L.FLEX_R, L.FLEX_WHITE)]


def cord_grip_shapes() -> list:
    return [_cyl(*GRIP_Z, 0.006, L.WHITE)]


def holder_shapes() -> list:
    """The holder's body as hanging at rest (h above the mouth at z MOUTH_Z)."""
    m = L.MOUTH_Z
    out = _ring(L.WALL_IN, L.HOLDER_R, m + L.MOUTH_H, m + L.CEIL_H, L.WALL_SEGS, L.WHITE)
    out += _funnel(L.HOLDER_R, m, L.HOLDER_R - L.WALL_IN, L.WALL_SEGS, L.WHITE)  # the mouth's funnel, 19 -> 14 mm
    out += _funnel(L.WALL_IN, m + L.GUIDE_H0, L.WALL_IN - L.GUIDE_IN, L.WALL_SEGS, L.WHITE)  # the collar's chamfer
    out += _ring(L.GUIDE_IN, L.WALL_IN, m + L.GUIDE_H1, m + L.CEIL_H, L.WALL_SEGS, L.WHITE)  # the collar
    out.append(_cyl(m + L.CEIL_H, m + L.HOLDER_LEN, L.HOLDER_R, L.WHITE))  # the roof and the terminal housing
    return out


def plunger_shapes(side: int) -> list:
    """Plunger A (side +1, +y) or B (-1): a 4 mm pin, its tip H_PL over the mouth."""
    z0 = L.MOUTH_Z + L.H_PL
    return [mk.cylinder((0.0, side * L.PL_OFF, z0 + L.PL_LEN / 2), L.PL_R, L.PL_LEN, "Z", L.CAP_METAL)]


def catch_shapes(side: int) -> list:
    """The hook of J-slot A (side +1, at +x) or B (-1, at -x) at rest, back in the wall: the catch (the hook's floor)
    and its two lips."""
    rc = (L.CATCH_R0 + L.CATCH_R1) / 2
    depth = L.CATCH_R1 - L.CATCH_R0
    m = L.MOUTH_Z
    x = side * rc
    out = [mk.box((x, 0.0, m + (L.CATCH_H0 + L.CATCH_H1) / 2), (depth, L.CATCH_T, L.CATCH_H1 - L.CATCH_H0), L.WHITE)]
    for t in (-L.LIP_C, L.LIP_C):
        out.append(mk.box((x, t, m + L.CATCH_H1 + L.LIP_H / 2), (depth, L.LIP_W, L.LIP_H), L.WHITE))
    return out


# -- the bulb (its own frame) ------------------------------------------------------------------------------------------
def bulb_shapes() -> list:
    out = [mk.cylinder((0.0, 0.0, -L.CAP_LEN / 2), L.CAP_R, L.CAP_LEN, "Z", L.CAP_METAL)]
    pin_len = L.PIN_OUT - (L.CAP_R - 0.0005)
    xc = L.PIN_OUT - pin_len / 2
    for s in (1.0, -1.0):
        out.append(mk.cylinder((s * xc, 0.0, -L.PIN_FROM_END), L.PIN_R, pin_len, "X", L.CAP_METAL))
    out.append(mk.cylinder((0.0, 0.0, sum(L.NECK_Z) / 2), L.NECK_R, L.NECK_Z[1] - L.NECK_Z[0], "Z", L.FROSTED))
    for z0, z1, r in L.GLASS:
        out.append(mk.cylinder((0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, "Z", L.FROSTED))
    return out


# -- the ring stand (its own frame) -------------------------------------------------------------------------------------
def ring_stand_shapes() -> list:
    c = L.STAND_GREY
    shelf = L.RING_POST_TOP - 0.002
    return [
        mk.cylinder((0.0, 0.0, L.RING_BASE_H / 2), L.RING_BASE_R, L.RING_BASE_H, "Z", c),
        mk.cylinder((0.0, 0.0, (L.RING_BASE_H + shelf) / 2), L.RING_POST_R, shelf - L.RING_BASE_H, "Z", c),
        mk.cylinder((0.0, 0.0, (shelf + L.RING_POST_TOP) / 2), L.RING_OUT, L.RING_POST_TOP - shelf, "Z", c),
    ] + _ring(L.RING_IN, L.RING_OUT, L.RING_POST_TOP, L.RING_TOP, L.RING_SEGS, c)
