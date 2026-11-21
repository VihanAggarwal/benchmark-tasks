"""The metronome's shapes (base frame M at rest, layout.py) and its two joints. NumPy only.

The case: three stacked black tiers (the pyramid), a white scale strip on the front of the upper two with a dark
mark at every Maelzel tempo (long marks at 40, 60, 80, 100, 120, 144, 160, 208), the winding key on the right side
(-y) and four feet. The pendulum: a steel rod on a hinge about x 55 mm up, its counterweight inside the case (links
of one assembly never collide with it, as the real bob swings in the hollow case). The weight: a brass block on a
slider along the rod, with a red pointer line on its top front edge. The slider has no PhysX joint friction: its
spring clip is a stiff spring toward a target that clip.py moves (held below 3 N, sliding at 3 N).
"""

from __future__ import annotations

import layout as L
import mechkit as mk
import pendulum as P

PIVOT = (L.ROD_X, 0.0, L.PIVOT_Z)
SCALE_X = 0.0015  # the strip's thickness on the front faces


def case_shapes() -> list:
    out = []
    for x0, x1, hw, z0, z1 in L.TIERS:
        out.append(mk.box(((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (x1 - x0, 2 * hw, z1 - z0), L.BLACK))
    # the scale strip, 14 mm wide, on the fronts of the middle and upper tiers, behind the rod
    for x0, _x1, _hw, z0, z1 in L.TIERS[1:]:
        out.append(mk.box((x0 - SCALE_X / 2, 0.0, (z0 + z1) / 2), (SCALE_X, 0.014, z1 - z0), L.WHITE))
    for b, r in P.MARK_R.items():
        z = L.PIVOT_Z + r + L.W_SIZE[2] / 2  # level with the weight's pointer (its top front edge) when set to b
        tier = L.TIERS[1] if z < L.TIERS[1][4] else L.TIERS[2]
        long = b in L.SCALE_LABELLED
        out.append(mk.box((tier[0] - SCALE_X - 0.0003, 0.0, z), (0.0006, 0.012 if long else 0.006, 0.0007), L.INK))
    # the winding key on the right side (-y), low on the front half (clear of the pinch on the back half)
    hw0 = L.TIERS[0][2]
    out.append(mk.cylinder((-0.010, -hw0 - 0.003, 0.030), 0.004, 0.006, "Y", L.BLACK))
    out.append(mk.box((-0.010, -hw0 - 0.009, 0.030), (0.004, 0.006, 0.018), L.BLACK))
    # the lower front's window round the pivot (drawn: a white disc behind the pivot)
    out.append(mk.cylinder((L.FRONT_X - 0.0005, 0.0, L.PIVOT_Z), 0.006, 0.001, "X", L.WHITE))
    return out


def rod_shapes() -> list:
    rz = (L.ROD_Z0 + L.ROD_Z1) / 2
    return [
        mk.box((L.ROD_X, 0.0, L.PIVOT_Z + rz), (L.ROD_D, L.ROD_W, L.ROD_Z1 - L.ROD_Z0), L.STEEL),
        mk.cylinder((L.ROD_X + 0.013, 0.0, L.PIVOT_Z - L.CW_D), L.CW_R, L.CW_H, "X", L.STEEL),
    ]


def weight_center(r: float) -> tuple:
    """The weight's centre (base frame, rest) with its centre r above the pivot."""
    wx = L.ROD_X + L.W_X0 + L.W_SIZE[0] / 2
    return (wx, 0.0, L.PIVOT_Z + r)


def weight_shapes() -> list:
    c = weight_center(P.R_START)
    return [
        mk.box(c, L.W_SIZE, L.BRASS),
        mk.box((L.ROD_X + L.W_X0 + 0.0015, 0.0, c[2] + L.W_SIZE[2] / 2 + 0.0003), (0.003, L.W_SIZE[1], 0.0006),
               L.RED),
    ]


def joints() -> list:
    rod = mk.hinge("rod", "X", PIVOT, rod_shapes(), lower=-L.SWING_LIMIT, upper=L.SWING_LIMIT,
                   mass=L.ROD_LINK_MASS, damping=0.0, friction=0.0, color=L.STEEL)
    weight = mk.slider("weight", "Z", weight_shapes(), lower=L.W_R_MIN - P.R_START, upper=L.W_R_MAX - P.R_START,
                       anchor=weight_center(P.R_START), mass=L.W_MASS, spring=L.W_CLIP_K,
                       damping=L.W_CLIP_DAMPING, friction=0.0,
                       parent="rod", color=L.BRASS)
    return [rod, weight]
