"""The parts as lists of mechkit shapes, in the unit's frame (origin on the table under the shaft, x toward the lid
hinge) or in each loose part's own frame. No simulator needed.

The body is a base slab on four rubber feet, the rotor chamber (a deck frame round a 124 mm opening, its corners
filled by 12 slabs on the 16 sides of a polygon), a front panel under the lid's tab and a back strip with the hinge
knuckles. The motor shaft stands in the chamber: a collar (the seat), the 8 mm shaft with a narrowed end and four
45-degree lead-in faces, and the cross pin that drives a rotor through the keyways under its hub. Two plungers in the
shaft (links on sliders along +y and -y, sprung outward) carry noses with 30-degree faces: the snap. The lid is a
top plate and two side skirts on a hinge at the back; its tab is a sprung slider on the lid.

Each rotor has the same hub: a 22 mm square with a 9 mm square bore, two keyway slots under its +-x walls (the pin
passes only there) and a snap groove inside its +-y walls (the noses sit in it seated). The tube rotor is a 16-sided
disc under the hub with 8 tube holes drawn on it; the strip rotor is a plate with four parallel slots along x, each
with a floor. A strip is a cap row over eight tubes with tapered tips. The rack is a top plate with the two loaded
columns cut through, its walls, two support bars under those columns and the other 80 wells drawn as dark discs.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


def slab(a, b, t: float, w: float, phi: float, color=None):
    """A thin box whose outer face spans a -> b in the half-plane at azimuth ``phi`` ((r, z) points; the outward
    normal is the direction a -> b turned a quarter turn clockwise), ``t`` thick inward, ``w`` wide."""
    dr, dz = b[0] - a[0], b[1] - a[1]
    n_len = math.hypot(dr, dz)
    dr, dz = dr / n_len, dz / n_len
    nr, nz = dz, -dr
    cr = (a[0] + b[0]) / 2 - nr * t / 2
    cz = (a[1] + b[1]) / 2 - nz * t / 2
    pitch = math.atan2(-nz, nr)
    return mk.box((cr * math.cos(phi), cr * math.sin(phi), cz), (t, w, n_len), color, ypr=(phi, pitch, 0.0))


def turned(shapes, yaw: float) -> list:
    """Boxes turned ``yaw`` about the frame's z axis (centres and edges)."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sh in shapes:
        x, y, z = sh.center
        out.append(mk.box((c * x - s * y, s * x + c * y, z), sh.size, sh.color, ypr=(yaw, 0.0, 0.0)))
    return out


# -- the unit ---------------------------------------------------------------------------------------------------------
def body_shapes() -> list:
    w = L.WHITE
    out = [mk.cylinder((sx * 0.060, sy * 0.072, L.FOOT_H / 2), L.FOOT_R, L.FOOT_H, "Z", L.FOOT_BLACK)
           for sx in (-1, 1) for sy in (-1, 1)]
    out.append(mk.box((0.0, 0.0, (L.FOOT_H + L.BOWL_FLOOR) / 2), (2 * L.HX, 2 * L.HY, L.BOWL_FLOOR - L.FOOT_H), w))
    out.append(mk.cylinder((0.0, 0.0, L.BOWL_FLOOR + 0.0005), L.BOWL_R - 0.001, 0.001, "Z", L.CHAMBER_GREY))
    h = L.DECK - L.BOWL_FLOOR
    zc = (L.DECK + L.BOWL_FLOOR) / 2
    out += mk.plate_with_hole((0.0, 0.0, zc), (2 * L.HX, 2 * L.HY), (2 * L.BOWL_R, 2 * L.BOWL_R), h, color=w)
    for k in range(16):  # the chamber's corners: slabs on the other 12 sides of a 16-sided polygon
        if k % 4 == 0:
            continue  # the frame's own edges
        phi = 2 * math.pi * k / 16
        c, s = math.cos(phi), math.sin(phi)
        half_w = 0.015
        t = 0.004
        while t < 0.030:  # as deep as the body's outline allows (1 mm inside it)
            r1 = L.BOWL_R + t + 0.001
            far = [(r1 * c - q * s, r1 * s + q * c) for q in (-half_w, half_w)]
            if any(abs(x) > L.HX - 0.001 or abs(y) > L.HY - 0.001 for x, y in far):
                break
            t += 0.001
        r = L.BOWL_R + t / 2
        out.append(mk.box((r * c, r * s, zc), (t, 2 * half_w, h), w, ypr=(phi, 0.0, 0.0)))
    out.append(mk.box(((L.COWL_X0 + L.COWL_X1) / 2, 0.0, (L.DECK + L.COWL_TOP) / 2),
                      (L.COWL_X1 - L.COWL_X0, 2 * L.HY, L.COWL_TOP - L.DECK), w))
    out.append(mk.box(((0.064 + L.HX) / 2, 0.0, L.DECK + 0.002), (L.HX - 0.064, 2 * L.HY, 0.004), w))
    for y in (-0.055, 0.055):
        out.append(mk.cylinder((L.HINGE_X, y, L.HINGE_Z), 0.0035, 0.020, "Y", w))
        out.append(mk.box((L.HINGE_X, y, (L.DECK + L.HINGE_Z) / 2), (0.006, 0.020, L.HINGE_Z - L.DECK), w))
    out += shaft_shapes()
    return out


def shaft_shapes() -> list:
    st = L.STEEL
    z1 = L.SHAFT_TOP - L.PILOT_H - (L.SHAFT_R - L.PILOT_R)  # where the lead-in faces start
    out = [
        mk.cylinder((0.0, 0.0, (L.BOWL_FLOOR + L.SEAT_Z) / 2), L.COLLAR_R, L.SEAT_Z - L.BOWL_FLOOR, "Z", st),
        mk.cylinder((0.0, 0.0, (L.SEAT_Z + z1) / 2), L.SHAFT_R, z1 - L.SEAT_Z, "Z", st),
        mk.cylinder((0.0, 0.0, (z1 + L.SHAFT_TOP) / 2), L.PILOT_R, L.SHAFT_TOP - z1, "Z", st),
        mk.box((0.0, 0.0, (L.KEY_Z0 + L.KEY_Z1) / 2), (2 * L.KEY_HX, 2 * L.KEY_HY, L.KEY_Z1 - L.KEY_Z0), st),
    ]
    a, b = (L.SHAFT_R, z1), (L.PILOT_R, z1 + (L.SHAFT_R - L.PILOT_R))
    for k in range(4):
        out.append(slab(a, b, 0.0012, 0.004, k * math.pi / 2, st))
    return out


def nose_shapes(phi: float) -> list:
    """One plunger's nose at azimuth ``phi``: two slabs meeting at the tip, faces 30 degrees from vertical."""
    f = math.radians(L.NOSE_FACE_DEG)
    tip = (L.NOSE_TIP_R, L.NOSE_TIP_Z)
    up = (tip[0] - L.NOSE_LEN * math.sin(f), tip[1] + L.NOSE_LEN * math.cos(f))
    down = (tip[0] - L.NOSE_LEN * math.sin(f), tip[1] - L.NOSE_LEN * math.cos(f))
    return [slab(tip, up, L.NOSE_T, L.NOSE_W, phi, L.STEEL), slab(down, tip, L.NOSE_T, L.NOSE_W, phi, L.STEEL)]


def lid_shapes() -> list:
    c = L.LID_TINT
    x0, x1 = L.LID_X0, L.LID_X1
    out = [mk.box(((x0 + x1) / 2, 0.0, (L.LID_Z0 + L.LID_Z1) / 2), (x1 - x0, 2 * L.HY, L.LID_Z1 - L.LID_Z0), c)]
    sx1 = 0.066
    for sy in (-1, 1):
        out.append(mk.box(((x0 + sx1) / 2, sy * (L.HY - L.SKIRT_T / 2), (L.SKIRT_Z0 + L.LID_Z0) / 2),
                          (sx1 - x0, L.SKIRT_T, L.LID_Z0 - L.SKIRT_Z0), L.LID_RIM))
    return out


def tab_shapes() -> list:
    return [mk.box(((L.TAB_U0 + L.TAB_U1) / 2, 0.0, (L.LID_Z0 + L.LID_Z1) / 2),
                   (L.TAB_U1 - L.TAB_U0, 2 * L.TAB_HY, L.LID_Z1 - L.LID_Z0), L.LID_RIM)]


# -- the rotors ---------------------------------------------------------------------------------------------------------
def hub_shapes(color) -> list:
    """The hub in the rotor's frame (origin at the middle of its underside)."""
    B, H, Z = L.BORE_HALF, L.HUB_HALF, L.HUB_H
    ky, kz, kx = L.KEYWAY_HY, L.KEYWAY_Z, L.KEYWAY_X
    g0, g1, gr = L.GROOVE_Z0, L.GROOVE_Z1, L.GROOVE_R
    out = []

    def bx(xa, xb, ya, yb, za, zb):
        out.append(mk.box(((xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2), (xb - xa, yb - ya, zb - za), color))

    for sx in (-1, 1):  # the +-x walls, with the keyway slot open at the bottom
        def xs(a, b):
            return (a, b) if sx > 0 else (-b, -a)
        bx(*xs(B, H), ky, H, 0.0, Z)
        bx(*xs(B, H), -H, -ky, 0.0, Z)
        bx(*xs(B, kx), -ky, ky, kz, Z)
        bx(*xs(kx, H), -ky, ky, 0.0, Z)
    for sy in (-1, 1):  # the +-y walls, with the snap groove inside
        def ys(a, b):
            return (a, b) if sy > 0 else (-b, -a)
        bx(-B, B, *ys(B, gr), 0.0, g0)
        bx(-B, B, *ys(B, gr), g1, Z)
        bx(-B, B, *ys(gr, H), 0.0, Z)
    return out


def tube_rotor_disc() -> list:
    """The tube rotor's disc: a 16-sided plate 96 mm across its flats (corners 48.9 mm from the axis) round an 11 mm
    clear radius, as 16 radial boxes, each from the hole to one flat and as wide as that flat (neighbours overlap
    toward the middle, so the 16-gon is filled). An earlier build stacked four turned 96 mm SQUARES: their union is a
    16-pointed star whose points reach 67.9 mm, deep into the 62 mm chamber wall, which jammed the rotor on the
    first GPU run (2026-11-19)."""
    c = L.TUBE_ROTOR_GREY
    t = L.TUBE_ROTOR_T
    r0, r1 = L.HUB_HALF, L.TUBE_ROTOR_R
    w = 2 * r1 * math.tan(math.pi / 16)
    out = []
    for k in range(16):
        phi = 2 * math.pi * k / 16
        r = (r0 + r1) / 2
        out.append(mk.box((r * math.cos(phi), r * math.sin(phi), -t / 2), (r1 - r0, w, t), c, ypr=(phi, 0.0, 0.0)))
    return out


def tube_rotor_shapes() -> list:
    c = L.TUBE_ROTOR_GREY
    out = tube_rotor_disc()
    for k in range(8):
        a = math.pi / 8 + k * math.pi / 4
        out.append(mk.cylinder((L.TUBE_HOLE_AT * math.cos(a), L.TUBE_HOLE_AT * math.sin(a), 0.0002),
                               L.TUBE_HOLE_R, 0.0004, "Z", (0.06, 0.06, 0.07)))
    return out + hub_shapes(c)


def strip_rotor_shapes() -> list:
    c = L.STRIP_ROTOR_BLUE
    t = L.STRIP_ROTOR_T
    holes = [(0.0, 0.0, 2 * L.HUB_HALF, 2 * L.HUB_HALF)] + [(0.0, y, L.SLOT_LX, L.SLOT_W) for y in L.SLOT_Y]
    out = list(mk.plate_with_holes((0.0, 0.0, -t / 2), (L.STRIP_ROTOR_X, L.STRIP_ROTOR_Y), holes, t, color=c))
    floor = t - L.SLOT_DEPTH
    for y in L.SLOT_Y:
        out.append(mk.box((0.0, y, -t + floor / 2), (L.SLOT_LX, L.SLOT_W, floor), c))
    return out + hub_shapes(c)


# -- the strips and the rack --------------------------------------------------------------------------------------------
def strip_shapes() -> list:
    c = L.STRIP_CLEAR
    out = [mk.box((0.0, 0.0, L.STRIP_H - L.CAP_T / 2), (L.STRIP_L, L.STRIP_W, L.CAP_T), c)]
    body = L.STRIP_H - L.CAP_T - L.TIP_H
    for i in range(8):
        x = (i - 3.5) * L.PITCH
        out.append(mk.cylinder((x, 0.0, L.TIP_H + body / 2), L.TUBE_R, body, "Z", c))
        out.append(mk.cylinder((x, 0.0, L.TIP_H / 2), L.TIP_R, L.TIP_H, "Z", c))
    return out


def rack_shapes() -> list:
    w = L.RACK_WHITE
    lx, ly, h, t = L.RACK_LX, L.RACK_LY, L.RACK_H, 0.002
    holes = [(0.0, L.rack_col_y(c), 8 * L.PITCH + 0.003, L.RACK_HOLE_W) for c in L.RACK_COLS]
    out = list(mk.plate_with_holes((0.0, 0.0, h - t / 2), (lx, ly), holes, t, color=w))
    wall_h = h - t
    for sx in (-1, 1):
        out.append(mk.box((sx * (lx / 2 - L.RACK_WALL / 2), 0.0, wall_h / 2), (L.RACK_WALL, ly, wall_h), w))
    for sy in (-1, 1):
        out.append(mk.box((0.0, sy * (ly / 2 - L.RACK_WALL / 2), wall_h / 2), (lx - 2 * L.RACK_WALL, L.RACK_WALL,
                                                                             wall_h), w))
    for col in L.RACK_COLS:
        out.append(mk.box((0.0, L.rack_col_y(col), L.RACK_FLOOR / 2), (8 * L.PITCH, 0.006, L.RACK_FLOOR), w))
    for col in range(1, 13):
        if col in L.RACK_COLS:
            continue
        for i in range(8):
            out.append(mk.cylinder(((i - 3.5) * L.PITCH, L.rack_col_y(col), h + 0.0002), 0.0028, 0.0004, "Z",
                                   L.WELL_DARK))
    return out
