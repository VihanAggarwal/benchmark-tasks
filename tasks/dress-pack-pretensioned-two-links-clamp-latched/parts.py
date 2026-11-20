"""The parts' shapes in their base frames at rest (mechkit conventions; base frame: layout's line frame). NumPy only.

Two assemblies, because links of one assembly never collide with each other and the protector must meet both the
hose (its rib) and the unit's exit face: the robot (fixed: forearm, wrist, flange and tool, the retraction unit's
exit end, the saddles and the protector's cradle) with the hose on a sprung slide, and the protector (loose: its U
shell as the base, the lid on its hinge).
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


def robot_shapes():
    out = []
    y0, y1 = L.ARM_Y
    out.append(mk.cylinder((0.0, (y0 + y1) / 2, L.ARM_Z), L.ARM_R, y1 - y0, axis="Y", color=L.GREY))
    for ys in (-0.120, 0.200):  # two stands under the parked forearm
        h = L.ARM_Z - L.ARM_R + 0.004
        out.append(mk.box((0.0, ys, h / 2), (0.040, 0.030, h), L.DARK))
    wy = L.WRIST_Y  # the wrist housing, the tool flange and a two-finger tool, all along -y
    out.append(mk.cylinder((0.0, wy - 0.020, L.ARM_Z), 0.042, 0.040, axis="Y", color=L.BLUE))
    out.append(mk.cylinder((0.0, wy - 0.048, L.ARM_Z), 0.031, 0.016, axis="Y", color=L.DARK))
    out.append(mk.box((0.0, wy - 0.068, L.ARM_Z), (0.060, 0.024, 0.044), L.DARK))
    for sx in (-1, 1):
        out.append(mk.box((sx * 0.018, wy - 0.095, L.ARM_Z), (0.008, 0.030, 0.020), L.GREY))
    # the retraction unit: four walls round the hose's passage (open along y), on a foot on the forearm
    uy0, uy1 = L.UNIT_Y
    a, w = L.UNIT_HALF, L.UNIT_WALL
    cy, ly, z = (uy0 + uy1) / 2, uy1 - uy0, L.HOSE_Z
    out += [mk.box((sx * (a - w / 2), cy, z), (w, ly, 2 * a), L.ORANGE) for sx in (-1, 1)]
    out += [mk.box((0.0, cy, z + sz * (a - w / 2)), (2 * a, ly, w), L.ORANGE) for sz in (-1, 1)]
    # saddles between the forearm and the hose
    zs0, zs1 = L.ARM_Z + L.ARM_R - 0.004, L.HOSE_Z - L.HOSE_R - 0.001
    for ys in (-0.075, 0.010, 0.090):
        out.append(mk.box((0.0, ys, (zs0 + zs1) / 2), (0.030, 0.010, zs1 - zs0), L.DARK))
    # the protector's cradle: a floor under it and two low guides either side (it slides along y only)
    c0, c1 = L.CRADLE_Y
    floor_top = L.HOSE_Z - L.RING_OUT - 0.0005
    fz0 = L.ARM_Z + L.ARM_R - 0.006
    out.append(mk.box((0.0, (c0 + c1) / 2, (fz0 + floor_top) / 2), (2 * L.RING_OUT + 0.016, c1 - c0, floor_top - fz0),
                      L.DARK))
    for sx in (-1, 1):
        out.append(mk.box((sx * (L.RING_OUT + 0.0045), (c0 + c1) / 2, floor_top + 0.006), (0.006, c1 - c0, 0.012),
                          L.DARK))
    return out


def hose_shapes():
    """Crests and waists along y at rest; the yellow band is the crest that sits just outside the protector's
    wrist-side face once the hose is set."""
    shapes = []
    hz = L.HOSE_Z
    band = L.band_rest_y()
    for wc in L.waists(0.0):
        w0, w1 = max(wc - L.GROOVE_W / 2, L.HOSE_Y1), min(wc + L.GROOVE_W / 2, L.HOSE_Y0)
        if w1 - w0 > 0.001:
            shapes.append(mk.cylinder((0.0, (w0 + w1) / 2, hz), L.GROOVE_R, w1 - w0, axis="Y", color=L.BLACK))
        c0, c1 = max(wc + L.GROOVE_W / 2, L.HOSE_Y1), min(wc + L.PITCH - L.GROOVE_W / 2, L.HOSE_Y0)
        if c1 - c0 > 0.001:
            col = L.YELLOW if abs((c0 + c1) / 2 - band) < 1e-6 else L.BLACK
            shapes.append(mk.cylinder((0.0, (c0 + c1) / 2, hz), L.HOSE_R, c1 - c0, axis="Y", color=col))
    return shapes


def hose_link():
    return mk.slider("pull", "-Y", hose_shapes(), upper=L.PULL_MAX, anchor=(0.0, L.WAIST0_Y, L.HOSE_Z),
                     mass=L.HOSE_MASS, spring=L.PULL_K, damping=L.PULL_DAMP)


def shell_shapes():
    """The protector's lower part: a square U round the hose (floor and two walls up to the lid), 1 mm clear of the
    crests, centred on the rib's plane y RING_Y."""
    a, b, w, z = L.RING_OUT, L.RING_IN, L.RING_W, L.HOSE_Z
    t = a - b
    return [mk.box((0.0, L.RING_Y, z - (a + b) / 2), (2 * a, w, t), L.ORANGE),
            mk.box((-(a + b) / 2, L.RING_Y, z), (t, w, 2 * b), L.ORANGE),
            mk.box(((a + b) / 2, L.RING_Y, z), (t, w, 2 * b), L.ORANGE)]


def _turned(center_xz, phi):
    """A shut-pose point (x, z) turned ``phi`` open about the hinge (+y axis)."""
    x, z = center_xz[0] - L.HINGE_X, center_xz[1] - L.HINGE_Z
    c, s = math.cos(phi), math.sin(phi)
    return (L.HINGE_X + x * c + z * s, L.HINGE_Z - x * s + z * c)


def lid_shapes_shut():
    """(centre (x, z), size) of the plate and the rib with the lid shut."""
    plate = ((L.HINGE_X - L.LID_LEN / 2, L.HINGE_Z), (L.LID_LEN, L.RING_W, L.LID_T))
    rib = ((0.0, L.HINGE_Z - L.LID_T / 2 - L.RIB_DEPTH / 2), (0.018, L.RIB_W, L.RIB_DEPTH))
    return plate, rib


def lid_link():
    """Drawn open (joint 0, on its open stop): the shut shapes turned LID_OPEN about the hinge. Shut is
    q = -LID_OPEN (a rotation about +y by +phi opens it)."""
    shapes = []
    for (cxz, size), col in zip(lid_shapes_shut(), (L.ORANGE, L.DARK)):
        x, z = _turned(cxz, L.LID_OPEN)
        shapes.append(mk.box((x, L.RING_Y, z), size, col, ypr=(0.0, L.LID_OPEN, 0.0)))
    return mk.hinge("lid", "Y", (L.HINGE_X, L.RING_Y, L.HINGE_Z), shapes, lower=-L.LID_OPEN, upper=0.0,
                    mass=L.LID_MASS, spring=L.LID_SPRING, damping=0.01)
