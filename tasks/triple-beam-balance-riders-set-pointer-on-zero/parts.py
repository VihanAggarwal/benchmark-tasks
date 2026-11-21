"""The balance and the specimen as mechkit assemblies (shapes in each assembly's base frame: origin on the table at the
knife edge, x forward from the middle beam, y to the left (the pan side), z up, every joint at 0 as drawn)."""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402

BEAM, R100, R10, POISE = "beam", "r100", "r10", "poise"
BALANCE_NAME, SPECIMEN_NAME = "balance", "rock"


def _b(lo, hi, color):
    c = tuple((a + b) / 2 for a, b in zip(lo, hi))
    s = tuple(b - a for a, b in zip(lo, hi))
    return mk.box(c, s, color)


def base_shapes():
    out = [_b((-L.BASE_W / 2, L.BASE_Y1, 0.0), (L.BASE_W / 2, L.BASE_Y0, L.BASE_H), L.BLACK)]
    px, py, pz = L.PILLAR
    out.append(_b((-px / 2, -py / 2, L.BASE_H), (px / 2, py / 2, pz), L.STEEL))  # the pillar under the knife edge
    # the scale post, with a mark for every degree of beam angle and a longer red one at zero
    sy = -L.SCALE_D
    out.append(_b((-0.025, sy - 0.004, 0.0), (0.025, sy, 0.150), L.CREAM))
    r = L.POINTER_D + 0.0  # pointer to the beam's axis
    for n in range(-4, 5):
        z = L.ZERO_Z + n * math.radians(L.MARK_DEG) * L.POINTER_D
        w = 0.040 if n == 0 else 0.022
        out.append(_b((-w / 2, sy + 0.0, z - 0.0009), (w / 2, sy + 0.0015, z + 0.0009), L.RED if n == 0 else L.BLACK))
    return out


def beam_shapes(cw_d: float, cw_size=(0.020, 0.040, 0.036)):
    """The beam: three bars, the cross piece, the left arm, the pan on its stem, the pointer and the tare weight at
    ``cw_d`` right of the knife edge."""
    z0, z1 = L.HZ - L.BAR_H / 2, L.HZ + L.BAR_H / 2
    out = []
    for name, dx in L.BEAM_DX.items():
        col = L.GOLD if name == "poise" else L.STEEL
        out.append(_b((dx - L.BAR_W / 2, -L.BEAM_LEN, z0), (dx + L.BAR_W / 2, 0.0, z1), col))
    out.append(_b((-0.044, -0.012, z0), (0.044, 0.012, z1), L.STEEL))  # the cross piece over the knife edge
    out.append(_b((-L.BAR_W / 2, 0.0, z0), (L.BAR_W / 2, L.LA, z1), L.STEEL))  # the left arm
    out.append(mk.cylinder((0.0, L.LA, (L.HZ + L.PAN_Z - L.PAN_T) / 2), 0.005, L.PAN_Z - L.PAN_T - L.HZ, "Z", L.STEEL))
    out.append(mk.cylinder((0.0, L.LA, L.PAN_Z - L.PAN_T / 2), L.PAN_R, L.PAN_T, "Z", L.STEEL))  # the pan plate
    # notch marks along the 500 g and 100 g beams (the rule-based detent is in task.py; these are what is drawn)
    zt = z1
    for k in range(L.R100_N + 1):
        d = L.ZERO_D + k * L.R100_STEP
        out.append(_b((L.BEAM_DX["b500"] - 0.004, -d - 0.001, zt), (L.BEAM_DX["b500"] + 0.004, -d + 0.001, zt + 0.0015),
                      L.BLACK))
    for k in range(L.R10_N + 1):
        d = L.ZERO_D + k * L.R10_STEP
        out.append(_b((L.BEAM_DX["b100"] - 0.004, -d - 0.001, zt), (L.BEAM_DX["b100"] + 0.004, -d + 0.001, zt + 0.0015),
                      L.BLACK))
    # the pointer, hanging from the right end, and the tare weight under the left arm
    out.append(_b((-0.003, -L.POINTER_D - 0.002, L.ZERO_Z), (0.003, -L.POINTER_D + 0.002, z0), L.RED))
    cx, cy, cz = cw_size
    for sx in (-1, 1):  # two blocks, one each side of the pillar
        out.append(_b((sx * 0.032 - cx / 2, cw_d - cy / 2, z0 - cz), (sx * 0.032 + cx / 2, cw_d + cy / 2, z0), L.GREY))
    return out


def rider_shapes(beam: str, d0: float, length: float, top: float, color):
    """A rider astride its beam at ``d0`` right of the knife edge: a top block and two cheeks."""
    dx = L.BEAM_DX[beam]
    y0, y1 = -d0 - length / 2, -d0 + length / 2
    hw, bw = L.RIDER_W / 2, L.BAR_W / 2
    zb = L.HZ + L.BAR_H / 2
    return [
        _b((dx - hw, y0, zb + 0.0005), (dx + hw, y1, L.HZ + top), color),
        _b((dx - hw, y0, L.HZ - 0.008), (dx - bw - 0.0005, y1, zb + 0.0005), color),
        _b((dx + bw + 0.0005, y0, L.HZ - 0.008), (dx + hw, y1, zb + 0.0005), color),
    ]


def _cg_z(shapes) -> float:
    """Height of the volume centre of ``shapes`` (PhysX spreads a body's mass evenly over its colliders)."""
    vols = [s.volume() for s in shapes]
    return sum(v * float(s.center[2]) for v, s in zip(vols, shapes)) / sum(vols)


def gravity_stiffness(cw_d: float) -> float:
    """N m/rad by which gravity tips the beam over (positive = destabilising): the beam, the three weights and the
    rock lying on the pan, each mass times g times its centre of mass's height over the knife edge. The beam's spring
    is K_BEAM plus this, so the net self-righting with the rock on the pan is K_BEAM (0.5 g at the pan per degree).
    GPU run 1 measured the empty beam's net stiffness at -0.062 N m/rad (growth 1.16 per second with the damper and
    the beam's 0.0074 kg m2), the value this gives without the rock's term against the old spring alone."""
    t = L.BEAM_MASS * (_cg_z(beam_shapes(cw_d)) - L.HZ)
    t += L.R100_MASS * (_cg_z(rider_shapes("b500", L.ZERO_D, L.RIDER_L, L.RIDER_TOP, L.BLUE)) - L.HZ)
    t += L.R10_MASS * (_cg_z(rider_shapes("b100", L.ZERO_D, L.RIDER_L, L.RIDER_TOP, L.RED)) - L.HZ)
    t += L.POISE_MASS * (_cg_z(rider_shapes("poise", L.ZERO_D, L.POISE_L, L.POISE_TOP, L.BLACK)) - L.HZ)
    t += L.SPEC_MASS * (L.PAN_Z + _cg_z(specimen_shapes()) - L.HZ)
    return t * L.G


def beam_spring(cw_d: float) -> float:
    return L.K_BEAM + gravity_stiffness(cw_d)


def joints(cw_d: float):
    beam = mk.hinge(BEAM, "X", (0.0, 0.0, L.HZ), beam_shapes(cw_d), upper=math.radians(L.STOP_DEG),
                    lower=-math.radians(L.STOP_DEG), mass=L.BEAM_MASS, spring=beam_spring(cw_d), damping=L.C_BEAM)
    r100 = mk.slider(R100, "-Y", rider_shapes("b500", L.ZERO_D, L.RIDER_L, L.RIDER_TOP, L.BLUE), upper=L.R100_STEP * L.R100_N,
                     mass=L.R100_MASS, spring=L.RIDER_DETENT_K[R100], damping=L.RIDER_DAMPING[R100],
                     friction=L.RIDER_FRICTION, parent=BEAM)
    r10 = mk.slider(R10, "-Y", rider_shapes("b100", L.ZERO_D, L.RIDER_L, L.RIDER_TOP, L.RED), upper=L.R10_STEP * L.R10_N,
                    mass=L.R10_MASS, spring=L.RIDER_DETENT_K[R10], damping=L.RIDER_DAMPING[R10],
                    friction=L.RIDER_FRICTION, parent=BEAM)
    poise = mk.slider(POISE, "-Y", rider_shapes("poise", L.ZERO_D, L.POISE_L, L.POISE_TOP, L.BLACK), upper=L.POISE_SPAN,
                      mass=L.POISE_MASS, spring=0.0, damping=0.5, friction=L.POISE_FRICTION, parent=BEAM)
    return [beam, r100, r10, poise]


def build(cw_d: float):
    return mk.assembly(BALANCE_NAME, origin=(L.X0, L.YP, L.TOP), shapes=base_shapes(), joints=joints(cw_d),
                       color=L.STEEL, friction=(0.6, 0.5))


def tare_torque(asm) -> float:
    """Gravity torque (N m about the knife edge, positive = pan side down) of the beam and the three weights at their
    zero notches, from the colliders' centres of mass (PhysX derives them the same way)."""
    t = 0.0
    for name in (BEAM, R100, R10, POISE):
        t += asm.body_mass(name) * float(asm.center_of_mass(name)[1]) * L.G
    return t


def tune(lo: float = 0.03, hi: float = 0.09) -> float:
    """Where the tare weight sits (m left of the knife edge) so that the empty beam is balanced."""
    f_lo, f_hi = tare_torque(build(lo)), tare_torque(build(hi))
    if f_lo * f_hi > 0:
        raise ValueError(f"the tare weight cannot balance the beam: {f_lo:.4f} and {f_hi:.4f} N m")
    for _ in range(60):
        mid = (lo + hi) / 2
        f = tare_torque(build(mid))
        if f * f_lo > 0:
            lo, f_lo = mid, f
        else:
            hi = mid
    return round((lo + hi) / 2, 5)


if __name__ == "__main__":
    d = tune()
    a = build(d)
    print("tare weight at", d, "torque", tare_torque(a))


def specimen_shapes():
    """A granite specimen in its own frame: origin on the table under its centre, an irregular chunk of two boxes (the
    cap turned 15 degrees), 62 x 48 x 26 mm and a 40 x 30 x 8 mm cap: 87 cm3, 0.2364 kg (2.7 g/cm3, granite)."""
    bx, by, bz = L.SPEC_BODY
    cx, cy, cz = L.SPEC_CAP
    return [
        mk.box((0.0, 0.0, bz / 2), (bx, by, bz), L.GRANITE),
        mk.box((0.004, 0.003, bz + cz / 2), (cx, cy, cz), L.GRANITE, ypr=(math.radians(15.0), 0.0, 0.0)),
    ]


def specimen():
    return mk.assembly(SPECIMEN_NAME, origin=(L.SPEC_X, L.SPEC_Y, L.TOP), mass=L.SPEC_MASS, shapes=specimen_shapes(),
                       color=L.GRANITE)
