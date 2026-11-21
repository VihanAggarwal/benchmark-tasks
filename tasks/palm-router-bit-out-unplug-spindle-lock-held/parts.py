"""The parts' shapes and joints (mechkit), from layout.py. NumPy and mechkit only.

Five task objects:
- ``router`` (fixed): the motor housing with its lock band (the base), the spindle (hinge about z, with the lock disc's
  six teeth) and the collet nut (hinge about z on the spindle, joint friction: breaking it free needs the spindle held);
- ``lock`` (fixed): the spindle-lock button, a sprung slider along +x carrying the pin that enters a gap between the
  teeth. It is its own assembly so that its pin collides with the spindle's teeth (links of one assembly never
  collide with each other);
- ``bit`` (loose): the stand-in bit, glued to the spindle by a bond that is released once the nut is 0.7 rad (40 deg) loose;
- ``wrench`` (loose): the open-jaw 17 mm collet wrench;
- ``bench`` (fixed): the power strip with the plug on a slider (friction: the receptacle's hold), the cord, the bit
  block with three holes, the wrench's rest block and the bench well's rim.
"""

from __future__ import annotations

import math

import mechkit as mk
import layout as L


def _ring_box(angle: float, r_mid: float, radial: float, tangential: float, z: tuple, color=None):
    c = (r_mid * math.cos(angle), r_mid * math.sin(angle), (z[0] + z[1]) / 2)
    return mk.box(c, (radial, tangential, z[1] - z[0]), color, ypr=(angle, 0.0, 0.0))


# -- the router ---------------------------------------------------------------------------------------------------
def housing_shapes() -> list:
    """R frame. The housing's lower part (bench to the lock band) and upper part (band to the shoulder) are
    cylinders; the band is cut away round the pin's channel toward -x."""
    r, (b0, b1) = L.HOUSING_R, L.BAND_Z
    s = [
        mk.cylinder((0, 0, b0 / 2), r, b0, color=L.BOSCH_BLUE),
        mk.cylinder((0, 0, (b1 + L.HOUSING_TOP) / 2), r, L.HOUSING_TOP - b1, color=L.BOSCH_BLUE),
    ]
    h = b1 - b0
    zc = (b0 + b1) / 2
    w = 0.026  # the band's half extent in x and y (a squared black collar)
    s.append(mk.box((0.0, (L.CHANNEL_HALF_Y + w) / 2, zc), (2 * w, w - L.CHANNEL_HALF_Y, h), L.BLACK))
    s.append(mk.box((0.0, -(L.CHANNEL_HALF_Y + w) / 2, zc), (2 * w, w - L.CHANNEL_HALF_Y, h), L.BLACK))
    s.append(mk.box(((L.CHANNEL_X_IN + w) / 2, 0.0, zc), (w - L.CHANNEL_X_IN, 2 * L.CHANNEL_HALF_Y, h), L.BLACK))
    # the bench well's rim, flush on the bench round the housing (drawn 1 mm high)
    for k in range(8):
        a = k * math.pi / 4
        s.append(_ring_box(a, r + 0.010, 0.018, 0.034, (0.0, 0.001), L.WELL_DARK))
    return s


def spindle_joint() -> mk.Joint:
    shapes = [mk.cylinder((0, 0, (L.SHAFT_Z[0] + L.SHAFT_Z[1]) / 2), L.SHAFT_R, L.SHAFT_Z[1] - L.SHAFT_Z[0],
                          color=L.STEEL)]
    rm = (L.TOOTH_R[0] + L.TOOTH_R[1]) / 2
    for k in range(L.TEETH):
        a = math.pi + math.pi / L.TEETH + k * 2 * math.pi / L.TEETH  # teeth between the gaps at 180 + k 60 deg
        shapes.append(_ring_box(a, rm, L.TOOTH_R[1] - L.TOOTH_R[0], L.TOOTH_W, L.TOOTH_Z, L.STEEL))
    return mk.hinge("spindle", "Z", (0.0, 0.0, 0.0), shapes, lower=-2 * math.pi, upper=2 * math.pi,
                    mass=L.SPINDLE_MASS, damping=L.SPINDLE_DAMPING)


def nut_shapes() -> list:
    """The hex ring: six side slabs, flats L.NUT_FLATS across, bore L.NUT_BORE_R, flat normals at the phase."""
    apothem = L.NUT_FLATS / 2
    side = 2 * apothem / math.sqrt(3)
    radial = apothem - L.NUT_BORE_R
    out = []
    for k in range(6):
        a = L.NUT_FLAT_PHASE + k * math.pi / 3
        out.append(_ring_box(a, L.NUT_BORE_R + radial / 2, radial, side, L.NUT_Z, L.STEEL))
    # the white reference line on the nut's top face (read against the line on the bit's cutter top)
    out.append(_ring_box(L.NUT_FLAT_PHASE, L.NUT_BORE_R + radial / 2, radial, 0.0012, (L.NUT_Z[1], L.NUT_Z[1] + 0.0004),
                         (0.97, 0.97, 0.97)))
    return out


def nut_joint() -> mk.Joint:
    return mk.hinge("nut", "Z", (0.0, 0.0, 0.0), nut_shapes(), lower=-0.2, upper=L.NUT_UPPER, mass=L.NUT_MASS,
                    damping=0.0, friction=L.NUT_FRICTION, parent="spindle")


# -- the lock button ----------------------------------------------------------------------------------------------
def lock_guide_shapes() -> list:
    """R frame: a thin rubber boot ring round the button on the band's face (the base; it never touches the cap)."""
    bx = -0.027  # inside the channel, clear of the cap's travel
    return [mk.box((bx, 0.0, L.BUTTON_Z + 0.0075), (0.002, 0.016, 0.002), L.BLACK)]


def button_joint() -> mk.Joint:
    sx, sy, sz = L.BUTTON_SIZE
    cap = mk.box((L.BUTTON_FACE_X + sx / 2, 0.0, L.BUTTON_Z), (sx, sy, sz), L.BLACK)
    px = L.PIN_SIZE[0]
    pin_x0 = L.BUTTON_FACE_X + sx
    pin = mk.box(((pin_x0 + L.PIN_TIP_X) / 2, 0.0, L.BUTTON_Z), (L.PIN_TIP_X - pin_x0, L.PIN_SIZE[1], L.PIN_SIZE[2]),
                 L.STEEL)
    assert abs((L.PIN_TIP_X - pin_x0) - px) < 1e-9, "the pin's length and its tip must agree"
    return mk.slider("button", "X", [cap, pin], upper=L.BUTTON_TRAVEL, mass=0.01, spring=L.BUTTON_SPRING,
                     damping=L.BUTTON_DAMPING)


# -- the bit --------------------------------------------------------------------------------------------------------
def bit_shapes() -> list:
    """Bit frame: origin on its axis at the shank's bottom end."""
    s0, s1 = L.SHANK_Z
    c0, c1 = L.CUTTER_Z
    return [
        mk.cylinder((0, 0, (s1 - s0) / 2), L.SHANK_R, s1 - s0, color=L.STEEL),
        mk.cylinder((0, 0, (c0 - s0) + (c1 - c0) / 2), L.CUTTER_R, c1 - c0, color=L.BRASS),
        mk.box((0, 0, (c1 - s0) + 0.0002), (2 * L.CUTTER_R - 0.002, 0.0012, 0.0004), (0.97, 0.97, 0.97)),  # white line
    ]


# -- the wrench -----------------------------------------------------------------------------------------------------
def wrench_shapes() -> list:
    t = L.WRENCH_T
    y = L.JAW_GAP / 2 + L.PRONG_W / 2
    p0, p1 = L.PRONG_X
    h0, h1 = L.HEAD_X
    g0, g1 = L.HANDLE_X
    return [
        mk.box(((p0 + p1) / 2, y, 0.0), (p1 - p0, L.PRONG_W, t), L.STEEL),
        mk.box(((p0 + p1) / 2, -y, 0.0), (p1 - p0, L.PRONG_W, t), L.STEEL),
        mk.box(((h0 + h1) / 2, 0.0, 0.0), (h1 - h0, 2 * L.HEAD_HALF_Y, t), L.STEEL),
        mk.box(((g0 + g1) / 2, 0.0, 0.0), (g1 - g0, L.HANDLE_W, t), L.STEEL),
    ]


# -- the bench: strip, plug, cord, bit block, wrench rest -------------------------------------------------------------
def strip_shapes() -> list:
    """Bench frame (table frame, fixed): the strip on its back edge, its outlet face at x STRIP_X[0]."""
    x0, x1 = L.STRIP_X
    y0, y1 = L.STRIP_Y
    s = [mk.box(((x0 + x1) / 2, (y0 + y1) / 2, L.STRIP_H / 2), (x1 - x0, y1 - y0, L.STRIP_H), L.STRIP_WHITE)]
    # The other outlet faces and the switch are on the strip (y -0.290 to -0.090). They were drawn at +y until 20 Nov,
    # where the outlet at +0.240 and the switch overlapped the wrench's jaw and shoved it 7 mm at spawn (GPU trace).
    for k, oy in enumerate((-0.150, -0.195, -0.240)):  # outlet faces (a hair proud of the strip, dark)
        if abs(oy - L.PLUG_Y) > 1e-9:
            s.append(mk.box((x0 - 0.0005, oy, L.PLUG_Z), (0.001, 0.024, 0.030), (0.80, 0.80, 0.78)))
    s.append(mk.box((x0 - 0.0005, -0.275, 0.030), (0.001, 0.012, 0.008), L.RED))  # the lit switch
    return s


def plug_joint() -> mk.Joint:
    """The plug body in front of the face plus the cord's first 20 mm out of its back, sliding along -x."""
    px, py, pz = L.PLUG_SIZE
    x0 = L.STRIP_X[0]
    body = mk.box((x0 - px / 2, L.PLUG_Y, L.PLUG_Z), (px, py, pz), L.BLACK)
    tail = mk.cylinder((x0 - px - 0.010, L.PLUG_Y, L.PLUG_Z), L.CORD_R, 0.020, axis="X", color=L.BLACK)
    drop = mk.box((x0 - px - 0.022, L.PLUG_Y, (L.PLUG_Z + 0.004) / 2 + 0.001), (0.007, 0.007, L.PLUG_Z - 0.004),
                  L.BLACK)
    return mk.slider("plug", "-X", [body, tail, drop], upper=L.PLUG_UPPER, mass=0.03, damping=5.0,
                     friction=L.PLUG_FRICTION)


def cord_shapes() -> list:
    """The cord's fixed run on the bench, from beside the plug's tail to the router's robot-facing side, under the
    lock button (drawn, fixed)."""
    r = L.CORD_R
    z = r + 0.0005
    side = 1.0 if L.PLUG_Y > L.RY else -1.0
    x_start = L.STRIP_X[0] - L.PLUG_SIZE[0] - 0.030
    xa, ya = 0.120, L.PLUG_Y - side * 0.007
    yb = L.RY + side * 0.012
    xr = L.RX - L.HOUSING_R - 0.012
    return [
        mk.box(((x_start + xa) / 2, ya, z), (x_start - xa, 2 * r, 2 * r), L.BLACK),
        mk.box((xa, (ya + yb) / 2, z), (2 * r, abs(ya - yb) + 2 * r, 2 * r), L.BLACK),
        mk.box(((xa + xr) / 2, yb, z), (xr - xa, 2 * r, 2 * r), L.BLACK),
    ]


def block_shapes() -> list:
    bx, by, bz = L.BLOCK_SIZE
    holes = [(float(h[0] - L.BLOCK_C[0]), float(h[1] - L.BLOCK_C[1]), L.HOLE_D, L.HOLE_D) for h in L.HOLES]
    return list(mk.plate_with_holes((float(L.BLOCK_C[0]), float(L.BLOCK_C[1]), bz / 2), (bx, by), holes, bz,
                                    L.BENCH_WOOD))


def rest_shapes() -> list:
    x0, x1 = L.REST_X
    return [mk.box(((x0 + x1) / 2, float(L.WRENCH0[1]), L.REST_H / 2), (x1 - x0, 0.034, L.REST_H), L.BENCH_WOOD)]


def bench_shapes() -> list:
    return strip_shapes() + cord_shapes() + block_shapes() + rest_shapes()
