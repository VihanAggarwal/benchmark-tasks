"""The parts as lists of mechkit shapes (base frames as described in layout.py). No simulator needed.

The camera is one rigid body: the body, the pentaprism hump, the hand grip, the eyecup and the lens. The lens's front
is the part the cap meets: a 1.5 mm lip whose inner edge is 56 mm across (the filter thread's crests) and, behind it,
a 7.5 mm deep groove 60 mm across (the thread's root, where the latches catch), each a 24-sided ring of boxes, then
the solid barrel.

The cap is one articulation: a base (the front plate and two fixed centring lugs on its local +-y) and two sprung
slider links (local +-x), each carrying a finger tab on the front and a latch tooth on the back. A tooth is a flat
block (its top face is the catch that sits behind the lip) and a ramp below it. Pinching the two tabs pulls both teeth
in by 2.5 mm, enough to clear the lip; let go, the springs push them out behind it.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


# -- the camera ------------------------------------------------------------------------------------------------------
def _ring(r_in: float, r_out: float, y0: float, y1: float, color, n: int = L.RING_N) -> list:
    """An n-sided ring round the lens axis (camera y), between y0 and y1: inner apothem ``r_in``, outer ``r_out``.
    One box per side, turned about y; a side at angle phi faces (cos phi, 0, -sin phi)."""
    t, w = r_out - r_in, L.ring_seg_width(r_out, n)
    rc, yc, ln = (r_in + r_out) / 2, (y0 + y1) / 2, y1 - y0
    out = []
    for k in range(n):
        phi = 2 * math.pi * k / n
        out.append(mk.box((rc * math.cos(phi), yc, -rc * math.sin(phi)), (t, ln, w), color, ypr=(0.0, phi, 0.0)))
    return out


def lens_shapes() -> list:
    return (_ring(L.LIP_R, L.LENS_R, 0.0, L.LIP_T, L.RING_GREY)
            + _ring(L.GROOVE_R, L.LENS_R, L.LIP_T, L.GROOVE_DEPTH, L.LENS_BLACK)
            + [mk.cylinder((0.0, (L.GROOVE_DEPTH + L.LENS_LEN) / 2, 0.0), L.LENS_R, L.LENS_LEN - L.GROOVE_DEPTH, "Y",
                           L.LENS_BLACK)])


def body_shapes() -> list:
    def span(a, b):
        return (a + b) / 2, b - a

    by, bl = span(L.BODY_Y0, L.BODY_Y1)
    bz, bh = span(L.BODY_Z0, L.TOP_PLATE_Z)
    hy, hl = span(L.HUMP_Y0, L.HUMP_Y1)
    hz, hh = span(L.TOP_PLATE_Z - 0.006, L.HUMP_Z1)
    gx, gw = span(L.GRIP_X0, L.GRIP_X1)
    gy, gl = span(L.GRIP_Y0, L.BODY_Y0 + 0.002)
    gz, gh = span(L.BODY_Z0, L.GRIP_Z1)
    ey = L.BODY_Y1 + L.EYECUP[1] / 2 - 0.001
    return [mk.box((0.0, by, bz), (L.BODY_W, bl, bh), L.CAM_BLACK),
            mk.box((0.0, hy, hz), (L.HUMP_W, hl, hh), L.CAM_BLACK),
            mk.box((gx, gy, gz), (gw, gl, gh), L.CAM_BLACK),
            mk.box((0.0, ey, L.EYECUP_Z0 + L.EYECUP[2] / 2), L.EYECUP, L.CAM_BLACK)]


def camera_shapes() -> list:
    return lens_shapes() + body_shapes()


# -- the cap ---------------------------------------------------------------------------------------------------------
def _ramp(side: str, r_tip: float, z1: float, r_lead: float, z2: float, color) -> mk.Shape:
    """A ramp slab on a tooth: its outer face runs from the crest (r_tip, z1) down to the lead (r_lead, z2), r measured
    from the cap's axis toward ``side`` ("+x", "-x", "+y", "-y"), z along the cap's axis (negative behind the seat)."""
    a, b = r_tip - r_lead, z1 - z2
    n = math.hypot(a, b)
    t = L.RAMP_T
    nr, nz = b / n, -a / n  # the outer face's normal in (r, z)
    rc = (r_tip + r_lead) / 2 - nr * t / 2
    zc = (z1 + z2) / 2 - nz * t / 2
    w = L.TOOTH_W
    if side in ("+x", "-x"):
        s = 1.0 if side == "+x" else -1.0
        return mk.box((s * rc, 0.0, zc), (t, w, n), color, ypr=(0.0, s * math.atan2(a, b), 0.0))
    s = 1.0 if side == "+y" else -1.0
    return mk.box((0.0, s * rc, zc), (w, t, n), color, ypr=(0.0, 0.0, -s * math.atan2(a, b)))


def _flat(side: str, r_in: float, r_out: float, z_top: float, z_bot: float, color) -> mk.Shape:
    rc, rt = (r_in + r_out) / 2, r_out - r_in
    zc, zh = (z_top + z_bot) / 2, z_top - z_bot
    if side in ("+x", "-x"):
        s = 1.0 if side == "+x" else -1.0
        return mk.box((s * rc, 0.0, zc), (rt, L.TOOTH_W, zh), color)
    s = 1.0 if side == "+y" else -1.0
    return mk.box((0.0, s * rc, zc), (L.TOOTH_W, rt, zh), color)


def cap_base_shapes() -> list:
    """The front plate (seat face at z 0) and the two fixed centring lugs on local +-y."""
    out = [mk.cylinder((0.0, 0.0, L.PLATE_T / 2), L.CAP_R, L.PLATE_T, "Z", L.CAP_BLACK)]
    for side in ("+y", "-y"):
        out.append(_flat(side, L.LUG_IN_R, L.LUG_FLAT_R, 0.0, L.LUG_Z1, L.CAP_BLACK))
        out.append(_ramp(side, L.LUG_TIP_R, L.LUG_Z1, L.LUG_LEAD_R, L.TEETH_Z, L.CAP_BLACK))
    return out


def latch_shapes(s: int) -> list:
    """One latch link, drawn at rest (out): its finger tab on the front and its latch tooth on the back, on local +x
    (``s`` +1) or -x (``s`` -1). The tab and the tooth are one part that slides along x; the link does not collide
    with the cap's own plate."""
    side = "+x" if s > 0 else "-x"
    tab = mk.box((s * (L.TAB_X0 + L.TAB_X1) / 2, 0.0, (L.PLATE_T + L.TAB_Z1) / 2),
                 (L.TAB_X1 - L.TAB_X0, 2 * L.TAB_HALF_Y, L.TAB_Z1 - L.PLATE_T), L.TAB_BLACK)
    return [tab, _flat(side, L.LATCH_IN_R, L.LATCH_FLAT_R, L.LATCH_CATCH_Z, L.LATCH_FLAT_Z1, L.TAB_BLACK),
            _ramp(side, L.LATCH_CREST_R, L.LATCH_FLAT_Z1, L.LATCH_LEAD_R, L.TEETH_Z, L.TAB_BLACK)]


def lowest(shapes) -> float:
    """The lowest point (local z) of a list of shapes at rest."""
    return min(float(s.center[2]) - float(s.half_extents()[2]) for s in shapes)
