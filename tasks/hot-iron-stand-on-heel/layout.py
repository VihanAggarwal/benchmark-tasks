"""Every dimension and threshold of the task in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases (away from the robot), y to the left, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z TOP. Heights are above the table top unless named *_W.

On the table: a table-top ironing mat (Joseph Joseph Swift, 90 x 47 cm, the maker's figure; 4 mm thick, est), long
side left to right; on it, toward the right of the arms' midline, a loose silicone iron rest (Encasa, 27 x 14.5 cm,
the seller's figure; 5 mm thick, est) taped down, long side fore and aft. A steam iron (Philips GC1426 Featherlight
Plus, 0.78 kg, the maker's leaflet) lies soleplate down on the rest, nose away from the robot. The iron is cold on a
real bench and declared on: the instruction says so.

IRON frame (the loose ``iron``): origin at the middle of the soleplate's rear bottom edge (the heel corner), x forward
along the sole toward the nose, y to the iron's left, z up from the sole. Lying flat with yaw 0: nose toward +x. Every
iron size below is an estimate (the maker publishes only the mass and the package, 270 x 145 x 114 mm; a retail
listing gives 26 x 11 x 13 cm for the iron), drawn to the maker's photographs: a 250 mm soleplate, a shell 70 mm
tall at the heel, a closed-loop handle whose bar is 28 mm wide and 25 mm thick with a 35 mm finger gap under it. The
flat rear face (the heel rest, the shell's back and the handle's rear post) is what it stands on.
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

MAT_COLOUR = (0.86, 0.82, 0.72)  # ecru cotton cover
REST_COLOUR = (0.42, 0.44, 0.47)  # grey silicone
SOLE_COLOUR = (0.70, 0.71, 0.73)
BODY_COLOUR = (0.20, 0.42, 0.72)
HANDLE_COLOUR = (0.12, 0.13, 0.15)

# -- the ironing mat (fixed): Joseph Joseph Swift table-top mat, 90 x 47 cm (maker); 4 mm thick (est) ----------------
MAT_LEN, MAT_DEPTH, MAT_T = 0.900, 0.470, 0.004
MAT_CX, MAT_CY = 0.360, -0.050
MAT_X0, MAT_X1 = MAT_CX - MAT_DEPTH / 2, MAT_CX + MAT_DEPTH / 2  # 0.125 .. 0.595
MAT_Y0, MAT_Y1 = MAT_CY - MAT_LEN / 2, MAT_CY + MAT_LEN / 2  # -0.500 (its right-hand end) .. 0.400
MAT_TOP = MAT_T
MAT_FRICTION = 0.8  # quilted cotton cover (est)

# -- the silicone iron rest (fixed: taped down on a real bench): Encasa, 27 x 14.5 cm (seller); 5 mm thick (est) -----
REST_LEN, REST_W, REST_T = 0.270, 0.145, 0.005
REST_CX, REST_CY = 0.330, -0.100
REST_X0, REST_X1 = REST_CX - REST_LEN / 2, REST_CX + REST_LEN / 2  # 0.195 .. 0.465
REST_Y0, REST_Y1 = REST_CY - REST_W / 2, REST_CY + REST_W / 2  # -0.1725 .. -0.0275
REST_TOP = MAT_TOP + REST_T  # 0.009

# -- the iron (loose): Philips GC1426, 0.78 kg (maker); every size an estimate ---------------------------------------
IRON_MASS = 0.78
IRON_FRICTION = 0.5  # plastic shell and a coated soleplate on cotton (est)
SOLE_T = 8.0
SOLE_STEPS = ((0.0, 165.0, 105.0), (165.0, 205.0, 85.0), (205.0, 232.0, 55.0), (232.0, 250.0, 25.0))  # (x0, x1, w)
SOLE_LEN = SOLE_STEPS[-1][1]
SHELL = (0.0, 175.0, 100.0, SOLE_T, 70.0)  # x0, x1, width, z0, z1: the water tank and housing
NOSE = (175.0, 220.0, 60.0, SOLE_T, 45.0)
HANDLE_W = 28.0
REAR_POST = (0.0, 35.0, 70.0, 130.0)  # x0, x1, z0, z1 (flush with the heel face, part of the heel rest)
BAR = (35.0, 175.0, 105.0, 130.0)  # the handle's grip bar: x0, x1, z0, z1 (25 thick, a 35 mm finger gap under it)
FRONT_POST = (175.0, 200.0, 45.0, 130.0)
IRON_H = 130.0
HEEL_H = SHELL[4]  # the heel face's broad part: 0 .. 70 mm up from the sole, 100 to 105 wide
HEEL_CENTRE = np.array([0.0, 0.0, HEEL_H / 2])  # the point read as "the heel" (iron frame, mm)

START_O = (REST_CX - SOLE_LEN * MM / 2, REST_CY, REST_TOP)  # the heel corner: the sole centred on the rest


def iron_rows():
    """The iron's boxes in its own frame, mm: (centre, size, colour)."""
    rows = []
    for x0, x1, w in SOLE_STEPS:
        rows.append((((x0 + x1) / 2, 0.0, SOLE_T / 2), (x1 - x0, w, SOLE_T), SOLE_COLOUR))
    for x0, x1, w, z0, z1 in (SHELL, NOSE):
        rows.append((((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (x1 - x0, w, z1 - z0), BODY_COLOUR))
    for x0, x1, z0, z1 in (REAR_POST, BAR, FRONT_POST):
        rows.append((((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (x1 - x0, HANDLE_W, z1 - z0), HANDLE_COLOUR))
    return rows


def iron_com_mm() -> np.ndarray:
    """The centre of mass (iron frame, mm) with the mass spread evenly over the boxes, as PhysX spreads it."""
    vols = np.array([s[0] * s[1] * s[2] for _, s, _ in iron_rows()])
    cs = np.array([c for c, _, _ in iron_rows()])
    return (vols[:, None] * cs).sum(axis=0) / vols.sum()


COM = iron_com_mm()  # about (99, 0, 43) mm
BALANCE_DEG = math.degrees(math.atan2(COM[0], COM[2]))  # nose-up pitch about the heel corner at which it balances


def sole_corners_mm() -> np.ndarray:
    """The soleplate's outline, iron frame (mm): the rear rectangle's and each step's front corners, z 0."""
    pts = []
    for x0, x1, w in SOLE_STEPS:
        for x in (x0, x1):
            for y in (-w / 2, w / 2):
                pts.append((x, y, 0.0))
    return np.array(pts)


def iron_corners_mm() -> np.ndarray:
    pts = []
    for c, s, _ in iron_rows():
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    pts.append((c[0] + sx * s[0], c[1] + sy * s[1], c[2] + sz * s[2]))
    return np.array(pts)


# -- the right-hand end of the mat (the scorer's zone, not marked on the mat) -----------------------------------------
ZONE_LEN = 0.250
ZONE_Y0, ZONE_Y1 = MAT_Y0, MAT_Y0 + ZONE_LEN  # -0.500 .. -0.250
EDGE_KEEP = 0.050  # and the heel at least 50 mm in from the mat's front and back edges
ZONE_X0, ZONE_X1 = MAT_X0 + EDGE_KEEP, MAT_X1 - EDGE_KEEP  # 0.175 .. 0.545

# -- scoring --------------------------------------------------------------------------------------------------------
UPRIGHT_DEG = 15.0  # standing on its heel: the sole within 15 degrees of vertical, nose up
FLAT_DEG = 20.0  # face down: the sole within 20 degrees of face down
NEAR_MM = 10.0  # resting on a surface: the lowest point (or the heel) within 10 mm of it
REST_SPILL_MM = 20.0  # on the bare mat: a corner of the soleplate more than 20 mm outside the rest's outline
HAZARD_S = 5.0  # left face down on the bare mat: still that long (latched)
HAZARD_MM, HAZARD_DEG = 5.0, 2.0  # still: moved less than this over HAZARD_S
END_S = 2.0  # the end state: the iron still and both hands 5 cm clear for the last 2 s
END_MM, END_DEG = 2.0, 1.0
CLEAR_M = 0.050
DROP_M = 0.050  # its lowest point 5 cm or more under the table top: fell off the table (latched)


def ry(a: float) -> np.ndarray:
    """Rotation about world y by ``a`` rad (+a turns +z toward +x)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pitched(phi_deg: float, yaw: float = 0.0) -> np.ndarray:
    """The iron's rotation with its nose raised ``phi_deg`` about its heel corner (0 flat, 90 standing on its heel)."""
    return rz(yaw) @ ry(-math.radians(phi_deg))


def support_top(x: float, y: float) -> float:
    """What a soleplate centred at (x, y) lies on: the rest's top, the mat's top or the table (heights over TOP)."""
    if REST_X0 <= x <= REST_X1 and REST_Y0 <= y <= REST_Y1:
        return REST_TOP
    if MAT_X0 <= x <= MAT_X1 and MAT_Y0 <= y <= MAT_Y1:
        return MAT_TOP
    return 0.0
