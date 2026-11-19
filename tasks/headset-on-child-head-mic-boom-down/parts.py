"""The parts as mechkit shapes and joints (frames in layout.py's docstring). NumPy only."""

from __future__ import annotations

import layout as L
import mechkit as mk


def head_shapes() -> list:
    """The head form in its own frame (origin on the table under its axis): turned sections, two ears, a nose, two
    eyes and the mouth line drawn on the face (toward -x)."""
    s = []
    for z0, z1, r in L.HEAD_SECTIONS:
        s.append(mk.cylinder((0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, color=L.FOAM))
    ey = (L.EAR_Y0 + L.EAR_Y1) / 2
    for sign in (1.0, -1.0):
        s.append(mk.box((0.0, sign * ey, L.EAR_ZC), (L.EAR_X, L.EAR_Y1 - L.EAR_Y0, L.EAR_Z[1] - L.EAR_Z[0]),
                        color=L.FOAM))
        s.append(mk.box((-0.0655, sign * 0.024, 0.135), (0.004, 0.012, 0.006), color=L.EYE))
    s.append(mk.box((-0.074, 0.0, 0.110), (0.012, 0.016, 0.022), color=L.FOAM))  # nose
    s.append(mk.box((L.MOUTH_X - 0.0015, 0.0, L.MOUTH_Z), (0.004, 2 * L.MOUTH_HALF_W, 0.004), color=L.LIP))
    return s


def stand_shapes() -> list:
    """Desk headphone stand: a weighted square base, a round post and a padded cradle along y on top."""
    return [
        mk.box((0.0, 0.0, 0.004), (0.100, 0.100, 0.008), color=L.SILVER),
        mk.cylinder((0.0, 0.0, 0.102), 0.008, 0.188, color=L.SILVER),
        mk.box((0.0, 0.0, L.STAND_TOP - 0.006), (0.030, 0.090, 0.012), color=L.BLACK),
    ]


def band_shapes() -> list:
    """The headband (the articulation's base), H frame: a top span and two stepped sides down to the yokes."""
    w, t = L.BAND_W, L.BAND_T
    s = [mk.box((0.0, 0.0, t / 2), (w, 0.100, t), color=L.BLACK)]
    for sign in (1.0, -1.0):
        s.append(mk.box((0.0, sign * 0.0625, -0.006 + t / 2), (w, 0.025, t), color=L.BLACK))  # y 50..75, z -12..0
        s.append(mk.box((0.0, sign * 0.085, -0.024), (w, 0.020, 0.032), color=L.BLACK))  # y 75..95, z -40..-8
    return s


def cup_shapes(sign: float) -> list:
    """A yoke and its ear cup (shell and cushion), H frame at rest. ``sign`` +1: the +y cup."""
    sy0, sy1 = L.SHELL_Y
    cy0, cy1 = L.CUSH_Y
    return [
        mk.box((0.0, sign * L.PIVOT_Y, L.PIVOT_Z - 0.010), (0.012, 0.008, 0.030), color=L.BLACK),
        mk.cylinder((0.0, sign * (sy0 + sy1) / 2, L.CUP_ZC), L.CUP_R, sy1 - sy0, axis="Y", color=L.BLACK),
        mk.cylinder((0.0, sign * (cy0 + cy1) / 2, L.CUP_ZC), L.CUSH_R, cy1 - cy0, axis="Y", color=L.GREY),
    ]


def boom_shapes() -> list:
    """The microphone boom, stowed straight up beside cup_n: the arm, the tip arm in toward the face and the
    microphone housing."""
    bx, by, z0 = L.BOOM_BAR_X, L.BOOM_Y, L.CUP_ZC
    top = z0 + L.BOOM_LEN
    return [
        mk.box((bx, by, (z0 + 0.010 + top) / 2), (0.006, 0.006, top - z0 - 0.010), color=L.BLACK),
        mk.box((bx, (by + L.TIP_Y1) / 2, top), (0.008, by - L.TIP_Y1 if by > L.TIP_Y1 else L.TIP_Y1 - by, 0.008),
               color=L.BLACK),
        mk.box((bx, L.TIP_Y1, top), (0.012, 0.014, 0.012), color=L.BLACK),
    ]


def joints() -> list:
    cp = mk.slider("cup_p", "Y", cup_shapes(1.0), upper=L.CUP_UPPER, mass=L.CUP_MASS, spring=L.CUP_SPRING,
                   damping=L.CUP_DAMP)
    cn = mk.slider("cup_n", "-Y", cup_shapes(-1.0), upper=L.CUP_UPPER, mass=L.CUP_MASS, spring=L.CUP_SPRING,
                   damping=L.CUP_DAMP)
    bm = mk.hinge("boom", "Y", anchor=L.BOOM_PIVOT, shapes=boom_shapes(), lower=L.BOOM_DOWN, upper=0.0,
                  mass=L.BOOM_MASS, damping=L.BOOM_DAMP, parent="cup_n")
    return [cp, cn, bm]
