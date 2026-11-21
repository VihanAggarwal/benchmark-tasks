"""The pendulum's physics, from the parts' shapes and masses (layout.py): its moment of inertia about the pivot,
its gravity stiffness, its period at a swing amplitude, and so the tempo it beats with the weight at a height.

A metronome ticks twice per period (once each way), so beats per minute = 120 / period. The escapement only keeps the
swing going; the rate is the pendulum's own. The same masses spread over the same shapes (as PhysX spreads a link's
mass over its colliders) are what the simulator uses, so these numbers are the simulator's, not a separate model.
NumPy only.
"""

from __future__ import annotations

import math

import layout as L


def _rod_parts():
    v_rod = L.ROD_W * L.ROD_D * (L.ROD_Z1 - L.ROD_Z0)
    v_cw = math.pi * L.CW_R ** 2 * L.CW_H
    rho = L.ROD_LINK_MASS / (v_rod + v_cw)
    return rho * v_rod, rho * v_cw


M_ROD, M_CW = _rod_parts()
ROD_C = (L.ROD_Z0 + L.ROD_Z1) / 2
_ROD_LEN = L.ROD_Z1 - L.ROD_Z0
# About the pivot's x axis: the rod (a box, y and z sides), the counterweight (a disc whose axis is x), parallel axes.
I_ROD = M_ROD * ((_ROD_LEN ** 2 + L.ROD_W ** 2) / 12 + ROD_C ** 2)
I_CW = M_CW * (L.CW_R ** 2 / 2 + L.CW_D ** 2)
I_W0 = L.W_MASS * (L.W_SIZE[1] ** 2 + L.W_SIZE[2] ** 2) / 12


def inertia(r: float) -> float:
    """kg m^2 about the pivot, the weight's centre r above it."""
    return I_ROD + I_CW + I_W0 + L.W_MASS * r * r


def stiffness(r: float) -> float:
    """N m per unit sin(angle): the counterweight's moment under the pivot less the rod's and the weight's above."""
    return L.G * (M_CW * L.CW_D - M_ROD * ROD_C - L.W_MASS * r)


def period(r: float, amp: float = 0.0) -> float:
    """Seconds per full swing (there and back) at amplitude ``amp`` (rad), the exact pendulum period (arithmetic-
    geometric mean form)."""
    k = stiffness(r)
    if k <= 0:
        return math.inf
    t0 = 2 * math.pi * math.sqrt(inertia(r) / k)
    a, b = 1.0, math.cos(amp / 2)
    for _ in range(30):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return t0 / a


def bpm(r: float, amp: float = 0.0) -> float:
    t = period(r, amp)
    return 0.0 if not math.isfinite(t) else 120.0 / t


def r_of_bpm(b: float, amp: float = 0.0) -> float:
    """The weight's centre height above the pivot that beats ``b`` per minute (bisection over its travel)."""
    lo, hi = 0.0, 0.095
    for _ in range(60):
        m = (lo + hi) / 2
        if bpm(m, amp) > b:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def accel(theta: float, omega: float, r: float) -> float:
    """Angular acceleration (rad/s^2) of the free pendulum (no damping)."""
    return -stiffness(r) * math.sin(theta) / inertia(r)


# The weight's centre height above the pivot (m) for each tempo, at the running swing (layout.RUN_AMP, 20 deg). The
# scale's mark for a tempo is drawn level with the weight's pointer (its top front edge, 7 mm above its centre) when
# it is set there (parts.case_shapes).
MARK_R = {b: round(r_of_bpm(b, L.RUN_AMP), 4) for b in L.MAELZEL}
R_START = MARK_R[L.START_BPM]
R_GOAL = MARK_R[L.GOAL_BPM]
