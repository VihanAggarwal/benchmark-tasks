"""Every dimension of the task, in one place. NumPy only.

The saw is a DeWalt DCS331 20 V MAX cordless jigsaw, bare tool (battery removed, no blade), lying on its LEFT side on
a rubber bench mat: its length along world x (front, the blade end, away from the robot), the shoe toward the robot's
right (-y), the top handle toward its left (+y). The bevel lever swings out to the saw's right, which is up.

Saw frame B (mm in this file, metres where named _M): x along the saw (+ toward its front), y from the shoe's outer
(working) face toward the handle, z up from the mat's top face. B is world-aligned (yaw 0) and its origin is at
ORIGIN_M. The bevel pivot is on the shoe's working face at the blade's line: (0, 0, Z_C).

Sourced (README, Sources): bare-tool mass 5.4 lb (2.45 kg, US listings; the maker's EU manual gives 2.3 kg); the
height, shoe sole to handle top, 215 mm (maker's EU datasheet), which the model has; body width 3.9 in (99 mm, retailer
listings, "height" there: it is the width across the motor housing); an aluminium shoe (maker's manual); detents at 15,
30 and 45 degrees each way, the lever pulled to the side to unlock and pushed back under the body to lock, and the shoe
slid forward to leave the 0 degree positive stop (DCS331 manual). Length: listings disagree (286 mm US, 305 mm on the
maker's EU sheet); 286 mm is kept. Checked against the manual's Fig. A and Fig. G (README, Revision): the lever lies in
the gap between shoe and motor housing, ends at the rear in a knob and swings to the side in the shoe's plane, as here;
Fig. A scaled by the 215 mm height gives a shoe about 170 mm long, as here. Everything else is an estimate (fidelity
text): shoe width and mass, the split of the mass, the lever's exact size, place and travel, the slide needed to leave
the stop and its travel, the handle's shape, the detent and clamp feel.
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # the station bench top (m)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- mat and placement ---------------------------------------------------------------------------------------------
MAT_T = 3.0  # mm, rubber bench mat, fixed
MAT_X = (-260.0, 100.0)  # B frame
MAT_Y = (-80.0, 240.0)
ORIGIN_M = (0.275, -0.060, TOP + MAT_T * MM)  # B's origin in the world: the pivot's x, the shoe's face, the mat's top
MAT_FRICTION = (1.0, 0.9)  # rubber
SAW_FRICTION = (0.8, 0.7)  # glass-filled nylon housing on rubber; aluminium shoe

# -- sourced figures -------------------------------------------------------------------------------------------------
LB = 0.45359237
MASS_TOTAL = 5.4 * LB  # 2.449 kg, bare tool (US listings; maker's EU manual: 2.3 kg)
LENGTH = 11.25 * 25.4  # 285.75 mm overall (US listings; maker's EU sheet: 305 mm)
BODY_T = 3.9 * 25.4  # 99.06 mm across the motor housing (retailer listings)
HEIGHT = 215.0  # mm, shoe sole to handle top (maker's EU datasheet): the handle box's far face, below
DETENTS_DEG = (0.0, 15.0, 30.0, 45.0)  # each way (manual); 45 is also the end of travel
BEVEL_MAX_DEG = 45.0

# -- estimates: masses -----------------------------------------------------------------------------------------------
SHOE_MASS = 0.250  # kg
CARRIAGE_MASS = 0.030
LEVER_MASS = 0.020
BODY_MASS = MASS_TOTAL - SHOE_MASS - CARRIAGE_MASS - LEVER_MASS

Z_C = BODY_T / 2  # 49.53: the blade's plane, the pivot's height over the mat (the saw lies on its side)


def zc(half):
    """(z0, z1) of a part ``half`` mm either side of the blade's plane."""
    return (Z_C - half, Z_C + half)


# -- the body (base, fixed to nothing: the whole saw is loose) --------------------------------------------------------
X_REAR = -226.0  # back of the battery mount
X_SHOE_FRONT = 60.0  # front of the shoe; LENGTH = X_SHOE_FRONT - X_REAR (to 0.25 mm)
X_NOSE = 20.0  # front of the gear housing (blade clamp, roller guide)
BODY_BOXES = {  # name: ((x0, x1), (y0, y1), (z0, z1))
    "housing": ((-175.0, -10.0), (40.0, 120.0), (0.0, BODY_T)),  # motor housing, lies on the mat
    "shoulder": ((-10.0, X_NOSE), (40.0, 120.0), zc(30.0)),
    "nose": ((-30.0, X_NOSE), (24.0, 40.0), zc(15.0)),  # gear housing over the shoe
    "battery_mount": ((X_REAR, -175.0), (40.0, 175.0), zc(40.0)),
    "rear_post": ((-175.0, -145.0), (120.0, 185.0), zc(17.5)),
    "front_post": ((-15.0, 10.0), (120.0, 185.0), zc(17.5)),
    "handle": ((-175.0, 10.0), (185.0, 215.0), zc(18.0)),  # the top handle's grip
    "lever_boss": ((-100.0, -88.0), (18.0, 40.0), (80.0, 95.0)),
}
HANDLE_TOP_Z = zc(18.0)[1]  # 67.5: the handle's upper side, where the left hand rests

# -- the shoe: carriage (slider) and shoe (hinge on the carriage) ------------------------------------------------------
SHOE_X = (-110.0, X_SHOE_FRONT)  # 170 mm long
SHOE_T = 4.0  # plate
SHOE_HALF_W = 45.0  # 90 mm wide (vertical as the saw lies)
BEAD = (6.0, 10.0)  # the cast edge's rib: 6 mm proud of the plate's inner face, 10 mm deep; the edge is 10 mm across
SHOE_BOXES = {
    "plate": (SHOE_X, (0.0, SHOE_T), zc(SHOE_HALF_W)),
    "top_rib": (SHOE_X, (SHOE_T, SHOE_T + BEAD[0]), (Z_C + SHOE_HALF_W - BEAD[1], Z_C + SHOE_HALF_W)),
    "bottom_rib": (SHOE_X, (SHOE_T, SHOE_T + BEAD[0]), (Z_C - SHOE_HALF_W, Z_C - SHOE_HALF_W + BEAD[1])),
}
EDGE_W = SHOE_T + BEAD[0]  # 10 mm: the shoe's edge across, where it is gripped
CARRIAGE_BOX = ((-40.0, 10.0), (SHOE_T, 24.0), zc(12.0))  # the trunnion saddle between shoe and nose
PIVOT = (0.0, 0.0, Z_C)
SLIDE_REL = 6.0  # mm forward to leave the 0 degree positive stop
SLIDE_TRAVEL = 10.0  # mm, the slot's length

# -- the bevel lever ---------------------------------------------------------------------------------------------------
LEVER_Y = (12.0, 18.0)
LEVER_PIVOT = (-95.0, 15.0, 87.5)
LEVER_BOXES = {
    "bar": ((-152.0, -90.0), LEVER_Y, (80.5, 94.5)),
    "tab": ((-152.0, -142.0), LEVER_Y, (94.5, 110.0)),  # finger tab, 11 mm proud of the housing's side
}
LEVER_OPEN_DEG = 60.0  # where it rests when pulled out
LEVER_MAX_DEG = 70.0
LEVER_SNAP_DEG = 25.0  # cam over-centre: below it falls shut, above it opens to LEVER_OPEN_DEG
LOCK_DEG = 10.0  # lever within this of flush: the clamp holds the shoe

# -- the rule's springs (one declared software model: stop, detents and clamp) -----------------------------------------
SLIDE_K = 10000.0  # N/m (pliers task, GPU-verified approach)
SLIDE_D = 60.0  # N s/m
BEVEL_K = 20.0  # N m/rad: 5 N on the shoe's edge 45 mm out turns a clamped shoe 0.6 deg
BEVEL_D = 0.2  # N m s/rad
LEVER_K = 1.0  # N m/rad
LEVER_D = 0.02
FOLLOW_GAIN = 0.9  # free joints: target = q + 0.9 x the last step's travel (pliers task)
FOLLOW_LEAD_SLIDE = 0.001  # m per step, at most
FOLLOW_LEAD_BEVEL = math.radians(1.0)  # rad per step, at most
DETENT_CAPTURE_DEG = 3.0
DETENT_PULL = 0.03 / BEVEL_K  # rad: a detent pulls with at most 0.03 N m (0.7 N at the push height)
LEVER_PULL = 0.05 / LEVER_K  # rad: the cam spring pulls with at most 0.05 N m
TILTED_DEG = 2.0  # past this the shoe cannot slide back into the stop

# -- scoring -----------------------------------------------------------------------------------------------------------
TARGET_DEG = 45.0
TARGET_TOL_DEG = 2.0
HALF_DEG = 28.0  # "30 degrees or more": the 30 detent less its 2 degree band
CLEAR_M = 0.050  # both grippers' finger zones this far from the saw
WINDOW_S = 2.0
WIN_STEPS = int(round(WINDOW_S * CONTROL_HZ))
STILL_BEVEL_DEG = 0.5
STILL_LEVER_DEG = 2.0
STILL_BODY_MM = 2.0
# Latched zero only for a real drop or fall, or the saw leaving the bench. Moving or turning it, however far, is not
# one (revision 2026-11-15b: the "fallen over at 60 degrees" latch is gone; a saw set upright on its shoe is fine).
DROP_M = 0.05  # any part this far below the bench top
FALL_SPEED_M_S = 1.0  # a body corner moving down faster than this (a free fall of 5 cm reaches 0.99 m/s): falling
BENCH_X_M = (-0.19, 0.91)  # the station table's top (world, m): the saw's reference point past it is off the bench
BENCH_Y_M = (-0.65, 0.65)
RESET_JUMP_M = 0.03
EPS = 1e-9


# -- helpers -----------------------------------------------------------------------------------------------------------
def box_centre_size(b):
    (x0, x1), (y0, y1), (z0, z1) = b
    return ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0)


def world_mm(p):
    """A B-frame point (mm) in the world (m), the saw at rest where it starts."""
    return tuple(ORIGIN_M[i] + p[i] * MM for i in range(3))


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def box_points(b, n: int = 4) -> np.ndarray:
    """Points on the surface grid of a box (mm), n per edge: what the offline contact model samples."""
    (x0, x1), (y0, y1), (z0, z1) = b
    xs, ys, zs = np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.linspace(z0, z1, n)
    g = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    on = (np.isclose(g[:, 0], x0) | np.isclose(g[:, 0], x1) | np.isclose(g[:, 1], y0) | np.isclose(g[:, 1], y1)
          | np.isclose(g[:, 2], z0) | np.isclose(g[:, 2], z1))
    return g[on]


def shoe_points(bevel_rad: float, slide_mm: float, n: int = 9) -> np.ndarray:
    """The shoe's sample points (B frame, mm) at a bevel and slide."""
    pts = np.vstack([box_points(b, n) for b in SHOE_BOXES.values()])
    piv = np.array(PIVOT)
    return (pts - piv) @ rot_x(bevel_rad).T + piv + np.array([slide_mm, 0.0, 0.0])


def lever_points(lever_rad: float, n: int = 7) -> np.ndarray:
    pts = np.vstack([box_points(b, n) for b in LEVER_BOXES.values()])
    piv = np.array(LEVER_PIVOT)
    return (pts - piv) @ rot_y(lever_rad).T + piv


def nearest_detent_deg(a_deg: float) -> float:
    cands = [s * d for d in DETENTS_DEG for s in (1.0, -1.0)]
    return min(cands, key=lambda d: abs(d - a_deg))


def bevel_band(a_deg: float) -> str:
    a = abs(a_deg)
    if abs(a - TARGET_DEG) <= TARGET_TOL_DEG + EPS:
        return "45"
    if a >= HALF_DEG - EPS:
        return "30+"
    return "low"
