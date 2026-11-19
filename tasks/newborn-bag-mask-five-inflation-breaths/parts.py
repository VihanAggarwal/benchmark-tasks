"""The parts as lists of mechkit shapes. No simulator needed.

The manikin's head, the manikin's body and the tray are fixed; their frames have the origin on the table top at
(0, 0) of the station (x forward, y left), so their shapes are drawn in table coordinates. The bag and mask is one
loose articulation drawn in the mask frame (layout.py): origin at the centre of the cushion's lower face, z up the
mask's axis, the valve housing and the bag toward -y.

Bag and mask links: the base is the mask's rigid shell (rim, dome, flat top); 8 cushion pads on vertical sprung
slides under the rim (``pad_0`` .. ``pad_7``, pressed up into the mask); the valve on the 360 degree swivel
(``valve``, a free Z hinge: the elbow and the housing); the bag on a free bending neck (``bag``, an X hinge with
stops at +-0.5 rad; positive: the bag's far end down); and on the bag's two sides the squeeze panels (``squeeze_a``
on +x, ``squeeze_b`` on -x, sprung slides pressed inward).
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


def _cyl(z0: float, z1: float, r: float, color, cx=0.0, cy=0.0) -> mk.Shape:
    return mk.cylinder((cx, cy, (z0 + z1) / 2), r, z1 - z0, "Z", color)


def _ellipse_x(x0: float, x1: float, r_mid: float, r_end: float, axis_y: float, axis_z: float, n: int, color) -> list:
    """A lying body of revolution along x: ``n`` X-cylinders whose radii follow an ellipse from ``r_end`` at the
    ends to ``r_mid`` in the middle."""
    out = []
    step = (x1 - x0) / n
    for i in range(n):
        xc = x0 + (i + 0.5) * step
        u = (2 * (xc - x0) / (x1 - x0)) - 1.0
        r = r_end + (r_mid - r_end) * math.sqrt(max(0.0, 1.0 - u * u))
        out.append(mk.cylinder((xc, axis_y, axis_z), r, step + 0.0002, "X", color))
    return out


def _ellipse_y(y0: float, y1: float, r_mid: float, r_end: float, axis_x: float, axis_z: float, n: int, color) -> list:
    """The same along y (y0 > y1 allowed)."""
    lo, hi = min(y0, y1), max(y0, y1)
    out = []
    step = (hi - lo) / n
    for i in range(n):
        yc = lo + (i + 0.5) * step
        u = (2 * (yc - lo) / (hi - lo)) - 1.0
        r = r_end + (r_mid - r_end) * math.sqrt(max(0.0, 1.0 - u * u))
        out.append(mk.cylinder((axis_x, yc, axis_z), r, step + 0.0002, "Y", color))
    return out


# -- the manikin -----------------------------------------------------------------------------------------------------
def skull_radius(x: float) -> float:
    """The skull's radius at x (the ellipse the slices follow); 0 off its ends."""
    u = (x - L.SKULL_X) / L.SKULL_HALF
    return L.SKULL_R * math.sqrt(max(0.0, 1.0 - u * u))


def head_shapes() -> list:
    """The head: the skull (lying X-cylinders on an ellipse), the face's level seat under the mask, the nose, the
    mouth and the two eyes (dark discs)."""
    x0, x1 = L.SKULL_X - L.SKULL_HALF, L.SKULL_X + L.SKULL_HALF
    n = L.SKULL_SLICES
    step = (x1 - x0) / n
    out = []
    for i in range(n):
        xc = x0 + (i + 0.5) * step
        r = max(0.012, skull_radius(xc))
        out.append(mk.cylinder((xc, 0.0, L.SKULL_AXIS_Z), r, step + 0.0002, "X", L.SKIN))
    out.append(_cyl(L.FACE_Z0, L.FACE_Z, L.FACE_R, L.SKIN, L.FACE_X, L.FACE_Y))
    (nx0, nx1), nh, nz = L.NOSE
    out.append(mk.box((L.FACE_X + (nx0 + nx1) / 2, L.FACE_Y, L.FACE_Z + nz / 2), (nx1 - nx0, 2 * nh, nz), L.SKIN))
    (mx0, mx1), mh, mz = L.MOUTH
    out.append(mk.box((L.FACE_X + (mx0 + mx1) / 2, L.FACE_Y, L.FACE_Z + mz / 2), (mx1 - mx0, 2 * mh, mz), L.LIP))
    for ex, ey, ez in L.EYES:
        out.append(mk.cylinder((ex, ey, ez - 0.0005), L.EYE_DISC_R, 0.002, "Z", L.EYE_DARK))
    return out


def body_shapes() -> list:
    """The neck, the torso, the arms and the legs (the chest plate is its own link)."""
    out = [mk.cylinder(((L.NECK_X[0] + L.NECK_X[1]) / 2, 0.0, L.NECK_AXIS_Z), L.NECK_R, L.NECK_X[1] - L.NECK_X[0],
                       "X", L.SKIN),
           mk.cylinder(((L.TORSO_X[0] + L.TORSO_X[1]) / 2, 0.0, L.TORSO_AXIS_Z), L.TORSO_R,
                       L.TORSO_X[1] - L.TORSO_X[0], "X", L.SKIN)]
    for s in (1, -1):
        out.append(mk.cylinder(((L.ARM_X[0] + L.ARM_X[1]) / 2, s * L.ARM_Y, L.ARM_AXIS_Z), L.ARM_R,
                               L.ARM_X[1] - L.ARM_X[0], "X", L.SKIN))
        out.append(mk.cylinder(((L.LEG_X[0] + L.LEG_X[1]) / 2, s * L.LEG_Y, L.LEG_AXIS_Z), L.LEG_R,
                               L.LEG_X[1] - L.LEG_X[0], "X", L.SKIN))
    return out


def chest_shapes() -> list:
    return [mk.box((L.CHEST_X, 0.0, L.CHEST_Z0 + L.CHEST_SIZE[2] / 2), L.CHEST_SIZE, L.SKIN)]


def tray_shapes() -> list:
    x0, x1 = L.TRAY_X
    y0, y1 = L.TRAY_Y
    return [mk.box(((x0 + x1) / 2, (y0 + y1) / 2, L.TRAY_Z / 2), (x1 - x0, y1 - y0, L.TRAY_Z), L.TRAY_COLOR)]


# -- the bag and mask (mask frame) -------------------------------------------------------------------------------------
def mask_shapes() -> list:
    """The mask's rigid shell: the rim over the cushion, the dome and its flat top round the stem."""
    return (_ring(L.RIM_IN, L.RIM_OUT, L.RIM_Z[0], L.RIM_Z[1], 12, L.CLEAR_SILICONE)
            + [_cyl(L.DOME_Z[0], L.DOME_Z[1], L.DOME_R, L.CLEAR_SILICONE),
               _cyl(L.TOP_Z[0], L.TOP_Z[1], L.TOP_R, L.CLEAR_SILICONE)])


def pad_angle(i: int) -> float:
    return 2.0 * math.pi * i / L.PAD_N


def pad_centre(i: int) -> tuple[float, float, float]:
    phi = pad_angle(i)
    return (L.PAD_RC * math.cos(phi), L.PAD_RC * math.sin(phi), L.PAD_H / 2)


def pad_shapes(i: int) -> list:
    """One cushion pad: a block under the rim, its lower face the cushion's lower face (z 0)."""
    return [mk.box(pad_centre(i), (L.PAD_RADIAL, L.PAD_TAN, L.PAD_H), L.CLEAR_SILICONE, ypr=(pad_angle(i), 0.0, 0.0))]


def valve_shapes() -> list:
    """The elbow up from the swivel and the housing out toward -y."""
    y0, y1 = L.HOUSING_Y
    return [_cyl(L.SWIVEL_Z, L.ELBOW_TOP, L.ELBOW_R, L.VALVE_BLUE),
            mk.cylinder((0.0, (y0 + y1) / 2, L.HOUSING_Z), L.ELBOW_R, abs(y1 - y0), "Y", L.VALVE_BLUE)]


def bag_shapes() -> list:
    """The bag's neck, body (an ellipsoid of stacked Y-cylinders) and air intake."""
    out = [mk.cylinder((0.0, L.NECK_Y - L.NECK_LEN / 2, L.HOUSING_Z), L.NECK_RAD, L.NECK_LEN, "Y", L.BAG_GREEN)]
    out += _ellipse_y(L.BAG_Y[0], L.BAG_Y[1], L.BAG_R, L.BAG_END_R, 0.0, L.HOUSING_Z, L.BAG_SLICES, L.BAG_GREEN)
    out.append(mk.cylinder((0.0, L.BAG_Y[1] - L.INTAKE_LEN / 2, L.HOUSING_Z), L.INTAKE_R, L.INTAKE_LEN, "Y",
                           L.BAG_GREEN))
    return out


def panel_centre(side: int) -> tuple[float, float, float]:
    """A squeeze panel's centre at rest (side +1: +x, -1: -x)."""
    return (side * (L.PANEL_OUT - L.PANEL_T / 2), L.PANEL_Y, L.HOUSING_Z)


def panel_shapes(side: int) -> list:
    return [mk.box(panel_centre(side), (L.PANEL_T, L.PANEL_LEN, L.PANEL_H), L.BAG_GREEN)]


def bag_low_z() -> float:
    """The bag's lowest point (mask frame, at rest): over the cushion's lower face."""
    return L.HOUSING_Z - max(L.BAG_R, L.PANEL_H / 2)
