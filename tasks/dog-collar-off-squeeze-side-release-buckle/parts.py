"""The parts as mechkit shapes and joints (base frames in layout.py's docstring). NumPy only."""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

DOX, DOY = L.DOG_ORIGIN[0], L.DOG_ORIGIN[1]


def _d(x, y, z):
    """Table frame -> the manikin's base frame."""
    return (x - DOX, y - DOY, z)


def _rod(a, b, radius: float, color) -> "mk.Shape":
    """A cylinder lying from table-frame point a to b (both on the same height), turned about z only."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    return mk.cylinder(_d(*((a + b) / 2)), radius, float(np.linalg.norm(d[:2])), axis="X", color=color,
                       quat=mk.quat_from_ypr(math.atan2(d[1], d[0])))


def _upper_ear(color) -> list:
    """The upper (left) floppy ear, lying on the skull: from a narrow root at the back (-y) of the skull it widens
    over the top toward +y and slightly forward and hangs down the +y side to a rounded tip. Four 4 mm plates
    on chords round the skull's axis (angles from the top, positive toward +y), 1 mm proud at each chord's middle,
    then a disc for the tip. x 206 to 257, 8 mm in front of the neck (x 265); the collar's plane is x 320."""
    r_mid, t = L.HEAD_R + 0.003, 0.004  # plate mid-plane radius at each chord's middle: underside 1 mm proud
    plates = ((-50.0, -25.0, 0.243, 0.026), (-25.0, 0.0, 0.238, 0.038), (0.0, 25.0, 0.234, 0.046),
              (25.0, 50.0, 0.232, 0.048))  # deg, x, width
    s = []
    for a0, a1, xc, w in plates:
        a0, a1 = math.radians(a0), math.radians(a1)
        am = (a0 + a1) / 2
        chord = 2 * r_mid * math.tan((a1 - a0) / 2) + 0.002  # 1 mm overlap at each joint
        c = (xc, r_mid * math.sin(am), L.HEAD_R + r_mid * math.cos(am))
        s.append(mk.box(_d(*c), (w, chord, t), color=color, quat=mk.quat_from_ypr(0.0, 0.0, -am)))
    a_end, tip_r = math.radians(plates[-1][1]), 0.025
    r_end = r_mid / math.cos(math.radians(12.5))
    c = (np.array([r_end * math.sin(a_end), L.HEAD_R + r_end * math.cos(a_end)])
         + np.array([math.cos(a_end), -math.sin(a_end)]) * (0.3 * tip_r))
    s.append(mk.cylinder(_d(0.231, c[0], c[1]), tip_r, t, axis="Z", color=color,
                         quat=mk.quat_from_ypr(0.0, 0.0, -a_end)))
    return s


def _eye(color) -> "mk.Shape":
    """The upper eye: a 20 mm disc on the skull at x 192, 22 mm toward the back (-y) from the top, square to the
    skull's surface there and 3 mm proud of it."""
    a = math.asin(-0.022 / L.HEAD_R)  # its angle from the skull's top, positive toward +y
    r = L.HEAD_R + 0.001
    return mk.cylinder(_d(0.192, r * math.sin(a), L.HEAD_R + r * math.cos(a)), 0.010, 0.004, color=color,
                       quat=mk.quat_from_ypr(0.0, 0.0, -a))


def _lower_ear(color) -> list:
    """The lower (right) ear, flipped out flat on the table from under the back of the head and lying forward: a
    plate and a rounded tip, 1 to 4 mm up (clear of the table, so the manikin does not stand on it), x 166 to 218,
    y -55 to -124; its root is hidden under the skull. x 218 at most keeps it 87 mm from the female's side (x 305.5
    when the collar lies in its plane x 320), so a collar laid out beside the head stays 25 mm clear of it even with
    the manikin nudged 60 mm toward +x (tests: the nudged manikin still scores 0.75)."""
    yaw = math.radians(-10.0)  # the ear points toward -y and slightly toward the muzzle (-x)
    c, half_len, tip_r = np.array([0.192, -0.085]), 0.022, 0.023
    tip = c + (half_len - 0.006) * np.array([math.sin(yaw), -math.cos(yaw)])
    return [mk.box(_d(c[0], c[1], 0.0025), (0.046, 2 * half_len, 0.003), color=color, quat=mk.quat_from_ypr(yaw)),
            mk.cylinder(_d(tip[0], tip[1], 0.0025), tip_r, 0.003, color=color)]


def dog_shapes() -> list:
    """The lying dog: muzzle, cheeks, head, neck, chest with a deeper shoulder, the flank it lies on, rump, four legs,
    a tail curled along the table behind the hind legs; floppy ears, a large nose and an eye. Every shape collides.
    The look-only parts (nose, eye, ears, cheeks, shoulder, tail) are all at x 257 mm or less, or x 420 mm or more,
    so none is beside the neck or in the collar's plane: nothing the collar, the buckle or a gripper meets (the collar's plane
    is x 320, the neck x 265 to 380; README, "The model")."""
    t, dk, bl = L.DOG_TAN, L.DOG_DARK, L.DOG_BLACK
    s = [
        mk.cylinder(_d(0.135, 0.008, 0.040), 0.040, 0.080, axis="X", color=t),  # muzzle x 95..175
        mk.cylinder(_d(0.095, 0.008, 0.048), 0.024, 0.014, axis="X", color=bl),  # nose x 88..102, 48 mm across
        mk.box(_d(0.102, 0.008, 0.074), (0.024, 0.036, 0.016), color=bl),  # nose leather over the muzzle tip
        mk.cylinder(_d(0.1775, 0.004, 0.053), 0.053, 0.025, axis="X", color=t),  # cheeks x 165..190: muzzle to skull
        mk.cylinder(_d((L.HEAD_X0 + L.HEAD_X1) / 2, 0.0, L.HEAD_R), L.HEAD_R, L.HEAD_X1 - L.HEAD_X0, axis="X",
                    color=t),  # skull x 175..265
        _eye(bl),
        *_upper_ear(dk),
        *_lower_ear(dk),
        mk.cylinder(_d((L.NECK_X0 + L.NECK_X1) / 2, 0.0, L.NECK_Z), L.NECK_R, L.NECK_X1 - L.NECK_X0, axis="X",
                    color=t),  # neck x 265..380
        mk.cylinder(_d(0.575, -0.020, 0.127), 0.127, 0.390, axis="X", color=t),  # chest and belly x 380..770
        mk.cylinder(_d(0.495, 0.006, 0.128), 0.126, 0.150, axis="X", color=t),  # deeper shoulder x 420..570, to +y 132
        mk.box(_d(0.575, -0.020, 0.020), (0.360, 0.170, 0.040), color=t),  # the flank it lies on
        mk.cylinder(_d(0.785, -0.020, 0.105), 0.105, 0.030, axis="X", color=t),  # rump x 770..800
        _rod((0.795, 0.010, 0.016), (0.850, 0.115, 0.016), 0.015, t),  # tail: out of the rump toward the legs,
        mk.cylinder(_d(0.850, 0.115, 0.0145), 0.0135, 0.027, color=t),  # a bend,
        _rod((0.850, 0.115, 0.013), (0.868, 0.225, 0.013), 0.012, t),  # and on along the table behind the hind legs
        mk.cylinder(_d(0.430, 0.270, 0.030), 0.030, 0.360, axis="Y", color=t),  # front legs, toward +y
        mk.cylinder(_d(0.500, 0.250, 0.030), 0.030, 0.320, axis="Y", color=t),
        mk.cylinder(_d(0.690, 0.250, 0.032), 0.032, 0.320, axis="Y", color=t),  # hind legs
        mk.cylinder(_d(0.750, 0.270, 0.032), 0.032, 0.360, axis="Y", color=t),
    ]
    return s


def _bx(x0, x1, y0, y1, z0, z1, color=None):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color=color)


def female_shapes() -> list:
    """The female housing (buckle frame): floor, roof, back block, and each side wall with its notch (open from the
    mouth to y 15 mm between z 3 and 8 mm) and the catch rib on its inner face behind the notch."""
    w, wi = L.F_HALF_W, L.F_WALL_IN
    y0, y1 = L.F_Y0, L.F_Y1
    s = [_bx(-w, w, y0, y1, 0.0, L.F_FLOOR), _bx(-w, w, y0, y1, L.F_ROOF, L.F_H),
         _bx(-wi, wi, 0.026, y1, L.F_FLOOR, L.F_ROOF)]
    for sg in (1.0, -1.0):
        def side(a, b):
            return (min(sg * a, sg * b), max(sg * a, sg * b))
        xa = side(wi, w)
        s += [_bx(*xa, L.NOTCH_Y1, 0.026, L.F_FLOOR, L.F_ROOF),  # the wall behind the notch
              _bx(*xa, y0, L.NOTCH_Y1, L.F_FLOOR, L.NOTCH_Z0),  # rail under the notch
              _bx(*xa, y0, L.NOTCH_Y1, L.NOTCH_Z1, L.F_ROOF),  # rail over it
              _bx(*side(L.RIB_X0, wi), L.RIB_Y0, L.RIB_Y1, L.NOTCH_Z0, L.NOTCH_Z1)]  # the catch rib
    return s


def male_shapes() -> list:
    """The male's body (exposed, y -26 to -4.3 mm, its back the webbing bar) and centre tongue."""
    return [_bx(-L.M_HALF_W, L.M_HALF_W, L.M_Y0, L.M_Y1, L.M_Z0, L.M_Z1),
            _bx(-L.TONGUE_HALF_W, L.TONGUE_HALF_W, L.M_Y1 - 0.001, L.TONGUE_Y1, *L.TONGUE_Z)]


def prong(sign: float):
    """One sprung prong (+x side: sign 1): beam, press tab and hook. Positive turns it inward."""
    def bx(x0, x1, y0, y1):
        a, b = sorted((sign * x0, sign * x1))
        return _bx(a, b, y0, y1, *L.PRONG_Z)
    shapes = [bx(*L.PRONG_X, L.PRONG_ANCHOR_Y, L.PRONG_Y1),
              bx(L.PRONG_X[1], L.TAB_X1, *L.TAB_Y),
              bx(L.PRONG_X[1], L.HOOK_X1, *L.HOOK_Y)]
    return mk.hinge("prong_p" if sign > 0 else "prong_n", "Z" if sign > 0 else "-Z",
                    (sign * L.PRONG_ANCHOR_X, L.PRONG_ANCHOR_Y, sum(L.PRONG_Z) / 2), shapes,
                    upper=L.PRONG_UPPER, mass=L.PRONG_MASS, spring=L.PRONG_SPRING, damping=L.PRONG_DAMP)


def to_buckle(p) -> tuple:
    """Table frame -> buckle frame."""
    return (float(p[0] - L.BX), float(p[1] - L.BY), float(p[2] - L.BZ0))


def link_shapes(i: int) -> list:
    """Link i (1..13) at rest: the webbing piece on the chord VERTS[i-1] -> VERTS[i], and its fittings."""
    p0, p1 = L.VERTS[i - 1], L.VERTS[i]
    roll = L.link_angle(p0, p1)
    q = mk.quat_from_ypr(0.0, 0.0, roll)
    mid = to_buckle((p0 + p1) / 2)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([0.0, -u[2], u[1]])  # the chord's normal in the y-z plane
    out = n if float(np.dot(n, np.asarray(mid) - (L.LOOP_C - np.array([L.BX, L.BY, L.BZ0])))) > 0 else -n
    s = [mk.box(mid, (L.WEB_W, L.LINK_LEN, L.WEB_T), color=L.RED, quat=q)]
    if i == L.TRIGLIDE_LINK:  # the tri-glide round the doubled webbing: 23 x 14 x 5 mm, black acetal
        s.append(mk.box(mid, (0.023, 0.014, 0.005), color=L.BLACK, quat=q))
    if i == L.DRING_LINK:  # the D-ring lying flat on the webbing's outer face: a 25 x 20 mm ring of 2.5 mm wire
        c = np.asarray(mid) + out * (L.WEB_T / 2 + 0.00125)
        for dy, size in ((0.0085, (0.025, 0.0025, 0.0025)), (-0.0085, (0.018, 0.0025, 0.0025))):
            s.append(mk.box(tuple(c + u * dy), size, color=L.STEEL, quat=q))
        for dx in (0.01125, -0.01125):
            s.append(mk.box(tuple(c + np.array([dx, 0.0, 0.0])), (0.0025, 0.0195, 0.0025), color=L.STEEL, quat=q))
    return s


def link_joints() -> list:
    out = []
    for i in range(1, L.N_LINKS + 1):
        mass = L.LINK_MASS + (L.DRING_MASS if i == L.DRING_LINK else 0.0) + (
            L.TRIGLIDE_MASS if i == L.TRIGLIDE_LINK else 0.0)
        out.append(mk.hinge(f"w{i}", "X", to_buckle(L.VERTS[i - 1]), link_shapes(i), lower=-L.WEB_LIMIT,
                            upper=L.WEB_LIMIT, mass=mass, damping=L.WEB_DAMP, color=L.RED,
                            parent="base" if i == 1 else f"w{i - 1}"))
    return out


def collar_mass() -> float:
    return (L.F_MASS + L.M_MASS + 2 * L.PRONG_MASS + L.N_LINKS * L.LINK_MASS + L.DRING_MASS + L.TRIGLIDE_MASS)
