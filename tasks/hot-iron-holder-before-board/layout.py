"""Every dimension and threshold of the task in one place. Metres, kilograms, radians unless a name says _MM or _DEG.
Station frame: x forward from the line through the two arm bases, y to the robot's left, z up; the table top is at
world z 0.7475 and every height ``z`` below is above it (``world()`` adds TOP).

The scene (est = estimate, see README for the sources of the others):
- a grey silicone soldering mat 300 (y) x 200 (x) x 1 mm (the ElectroPeak TE-509 size), fixed, centred at x 0.28;
- a Hakko FX-8801 iron (217 mm with its tip, 46 g without its cord: Hakko FX-888D manual) lying on the mat along y,
  tip toward -y; its sections are est: tip 5 mm across, heater barrel 10 mm, nut 14 mm (the metal front, 95 mm),
  grip 20 mm, strain relief 10 mm;
- its 1.2 m cord (manual), 5 mm across est, from the iron's rear to the FX-888D station (100 x 120 x 120 mm, 1.2 kg:
  manual), which stands fixed behind the mat; the cord's surplus lies in a loop behind the station;
- an iron holder in the shape of the Hakko FH800-03BY (its figures are not published: every number est): a cast base
  80 x 150 x 10 mm, 0.4 kg, loose, right of the mat; an octagonal cage sloping 40 deg down to the right (-y), its mouth
  toward the mat: 20 mm of 30 mm bore at the mouth, then 105 mm of 18 mm bore; the step between them stops the grip;
- an 80 x 60 x 1.6 mm FR4 prototype board on four M3 x 10 mm nylon standoffs, 20 g est, on the bench left of the mat.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
MM = 0.001
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ


def world(p) -> tuple[float, float, float]:
    """(x, y, z over the table) -> world point."""
    return (float(p[0]), float(p[1]), TOP + float(p[2]))


# -- the soldering mat (fixed) ------------------------------------------------------------------------------------
MAT_C = (0.28, 0.0)  # centre (x, y)
MAT_SIZE = (0.200, 0.300, 0.001)  # x, y, z (ElectroPeak TE-509: 300 x 200 x 1 mm)
MAT_TOP = MAT_SIZE[2]
MAT_GREY = (0.35, 0.38, 0.40)


def over_mat(x: float, y: float, margin: float = 0.0) -> bool:
    return (abs(x - MAT_C[0]) <= MAT_SIZE[0] / 2 - margin) and (abs(y - MAT_C[1]) <= MAT_SIZE[1] / 2 - margin)


# -- the iron (loose, welded to the cord's end) -------------------------------------------------------------------
# Sections along the iron's axis, s from the tip toward the rear: (name, s0, s1, radius, colour).
IRON_LEN = 0.217  # manual: total length without cord, with the 1.6D tip
IRON_MASS = 0.046  # manual: weight without cord
SILVER = (0.78, 0.78, 0.80)
STEEL = (0.60, 0.61, 0.63)
GRIP_BLUE = (0.10, 0.18, 0.45)
RELIEF_BLACK = (0.08, 0.08, 0.09)
IRON_SECTIONS = (
    ("tip", 0.000, 0.030, 0.0025, SILVER),
    ("barrel", 0.030, 0.080, 0.0050, STEEL),
    ("nut", 0.080, 0.095, 0.0070, STEEL),
    ("grip", 0.095, 0.207, 0.0100, GRIP_BLUE),
    ("relief", 0.207, 0.217, 0.0050, RELIEF_BLACK),
)
METAL_END = 0.095  # the visible metal front: s 0 .. 95 mm
METAL_SECTIONS = tuple(sec for sec in IRON_SECTIONS if sec[2] <= METAL_END + 1e-9)
GRIP_S = (0.095, 0.207)
GRIP_R = 0.010
GRIP_POINT_S = 0.150  # where the demo grips the handle: 55 mm behind the grip's front, 128 mm of pad edge to metal
IRON_FRICTION = 0.8  # rubber grip est
PRISM_SIDES = 12  # round parts collide as 12-sided prisms (flats at the radius) so a lying iron does not roll forever

# The iron lies on the mat through its middle, along y, tip toward -y; it rests on its grip.
IRON_AXIS_Z = MAT_TOP + GRIP_R  # 11 mm
IRON_GRIP_POINT0 = (0.28, 0.020)  # the demo's grip point on the mat (x, y)
IRON_TIP0 = (0.28, IRON_GRIP_POINT0[1] - GRIP_POINT_S, IRON_AXIS_Z)  # y -0.130: 20 mm inside the mat's edge
IRON_REAR0 = (0.28, IRON_TIP0[1] + IRON_LEN, IRON_AXIS_Z)  # y +0.087

# -- the cord ------------------------------------------------------------------------------------------------------
CORD_LEN = 1.2  # manual
CORD_R = 0.0025  # est
CORD_SEG = 0.025
CORD_SEGMENTS = 48  # 48 x 25 mm = 1.2 m
CORD_MASS = 0.030  # est: about 25 g/m
CORD_BEND = 0.002  # N m/rad per joint (est; a limp silicone cord, EI about 5e-5 N m2)
CORD_BEND_DAMPING = 0.0005
CORD_ARMATURE = 5e-6
CORD_MAX_BEND = 1.2
CORD_FRICTION = 0.8
CORD_GREY = (0.20, 0.20, 0.22)
CORNER_R = 0.030  # the laid cord's bends

# -- the station (fixed) -------------------------------------------------------------------------------------------
STATION_C = (0.50, -0.14)
STATION_SIZE = (0.120, 0.100, 0.120)  # x (depth), y (width), z: manual 100 W x 120 H x 120 D mm
STATION_FRONT_X = STATION_C[0] - STATION_SIZE[0] / 2  # 0.44
STATION_BLUE = (0.12, 0.30, 0.62)
CORD_PLUG = (STATION_FRONT_X - CORD_R, -0.110, 0.012)  # where the cord's first segment ends (the station's socket)


LOOP_X = 0.6700656  # the loop's far leg: set so 48 segments of 25 mm end exactly at the socket (tests check it)


def cord_path(loop_x: float = LOOP_X) -> list[tuple[float, float, float]]:
    """The laid cord's centreline corners, from the iron's rear to the station's socket (x, y, z over the table)."""
    rz_mat, rz_bench = MAT_TOP + CORD_R, CORD_R
    rear = IRON_REAR0
    return [
        rear,
        (0.280, rear[1] + 0.032, IRON_AXIS_Z),  # straight out of the strain relief (the first 25 mm chord is level)
        (0.280, 0.135, rz_mat),  # down onto the mat
        (0.372, 0.135, rz_mat),  # back along the mat's left part
        (0.395, 0.135, rz_bench),  # off the mat's back edge
        (0.395, -0.050, rz_bench),  # along the bench behind the mat
        (loop_x, -0.050, rz_bench),  # the surplus loop out behind the station
        (loop_x, -0.250, rz_bench),
        (0.400, -0.250, rz_bench),
        (0.400, -0.110, rz_bench),
        (0.425, -0.110, rz_bench),
        CORD_PLUG,
    ]


# -- the iron holder (loose) ---------------------------------------------------------------------------------------
HOLDER_ORIGIN = (0.28, -0.27)  # base centre (x, y); its frame's z is the table top
HOLDER_MASS = 0.40  # est
HOLDER_FRICTION = (0.5, 0.5)  # chromed steel coil est (the iron then slides onto its stop: 40 deg > atan 0.65)
BASE_SIZE = (0.080, 0.150, 0.010)  # est
CAGE_DEG = 40.0  # est
CAGE_COS, CAGE_SIN = math.cos(math.radians(CAGE_DEG)), math.sin(math.radians(CAGE_DEG))
CAGE_AXIS = np.array([0.0, -CAGE_COS, -CAGE_SIN])  # from the mouth into the cage (the tip's direction when in)
CAGE_UP = np.array([0.0, -CAGE_SIN, CAGE_COS])  # perpendicular to the axis, upward
CAGE_DOWN = -CAGE_UP
MOUTH = np.array([0.28, -0.205, 0.107])  # mouth centre (x, y, z over the table)
MOUTH_LEN, DEEP_LEN = 0.020, 0.105
CAGE_LEN = MOUTH_LEN + DEEP_LEN
MOUTH_R_IN, DEEP_R_IN, CAGE_R_OUT = 0.015, 0.009, 0.017  # across-flats radii of the octagonal bores and outside
STEP_U = MOUTH_LEN  # the grip's front stops here (u along CAGE_AXIS from the mouth)
CAGE_STAVES = 8
CHROME = (0.80, 0.80, 0.82)
BASE_BLACK = (0.12, 0.12, 0.12)
POST_U = (0.040, 0.110)  # posts under the cage at these u
POST_W = 0.010


def cage_point(u: float, perp: float = 0.0) -> np.ndarray:
    """A point on (or ``perp`` above, perpendicular) the cage axis, ``u`` from the mouth (x, y, z over the table)."""
    return MOUTH + u * CAGE_AXIS + perp * CAGE_UP


def holder_rows() -> list[tuple]:
    """The holder's shapes in its own frame (origin on the table under the base centre):
    ("box", centre, size, (yaw, pitch, roll), colour, part)."""
    ox, oy = HOLDER_ORIGIN
    rows = [("box", (0.0, 0.0, BASE_SIZE[2] / 2), BASE_SIZE, (0.0, 0.0, 0.0), BASE_BLACK, "base")]
    # octagonal staves: box x along the cage axis (yaw -90 deg turns x to -y, pitch +40 deg tips it down)
    yaw, pitch = -math.pi / 2, math.radians(CAGE_DEG)
    for u0, u1, r_in in ((0.0, MOUTH_LEN, MOUTH_R_IN), (MOUTH_LEN, CAGE_LEN, DEEP_R_IN)):
        t = CAGE_R_OUT - r_in
        w = 2 * CAGE_R_OUT * math.tan(math.pi / CAGE_STAVES)
        for k in range(CAGE_STAVES):
            roll = 2 * math.pi * k / CAGE_STAVES
            # the stave's local +z (before roll) is the box's outward direction; after yaw and pitch, local y is
            # +x of the world and local z is CAGE_UP; rolling by ``roll`` turns the outward direction about the axis
            out = math.cos(roll) * CAGE_UP + math.sin(roll) * np.array([-1.0, 0.0, 0.0])
            c = cage_point((u0 + u1) / 2) + out * (r_in + t / 2)
            rows.append(("box", (c[0] - ox, c[1] - oy, c[2]), (u1 - u0, w, t), (yaw, pitch, roll), CHROME,
                         "cage"))
    for u in POST_U:
        under = cage_point(u) + CAGE_DOWN * CAGE_R_OUT
        z0, z1 = BASE_SIZE[2], under[2] + 0.001
        rows.append(("box", (under[0] - ox, under[1] - oy, (z0 + z1) / 2), (POST_W, POST_W, z1 - z0),
                     (0.0, 0.0, 0.0), CHROME, "post"))
    return rows


# -- the circuit board (loose) -------------------------------------------------------------------------------------
BOARD_ORIGIN0 = (0.26, 0.30)  # start, on the bench, its long side along y
BOARD_SIZE = (0.060, 0.080, 0.0016)  # x, y, thickness (FR4)
STANDOFF_H, STANDOFF_R = 0.010, 0.00275  # M3 x 10 mm nylon, 5.5 mm hex as a round
STANDOFF_XY = tuple((sx * 0.026, sy * 0.036) for sx in (-1, 1) for sy in (-1, 1))  # 4 mm in from each edge
BOARD_MASS = 0.020  # est: 14 g of 1.6 mm FR4 plus standoffs and screws
BOARD_GREEN = (0.10, 0.45, 0.20)
NYLON = (0.92, 0.92, 0.88)
BOARD_PLATE_Z = STANDOFF_H + BOARD_SIZE[2] / 2  # centre of the plate over the standoffs' feet


def board_rows() -> list[tuple]:
    rows = [("box", (0.0, 0.0, BOARD_PLATE_Z), BOARD_SIZE, (0.0, 0.0, 0.0), BOARD_GREEN)]
    for sx, sy in STANDOFF_XY:
        rows.append(("cyl", (sx, sy, STANDOFF_H / 2), STANDOFF_R, STANDOFF_H, NYLON))
    return rows


# -- scoring thresholds --------------------------------------------------------------------------------------------
CONTACT_M = 0.001  # the hot metal "touches" something within 1 mm (a visible contact on the side video)
HELD_M = 0.005  # a finger zone within 5 mm of the iron: held, not resting in its holder
CLEAR_M = 0.030  # both grippers 3 cm or more from iron, cord and board
AXIS_DEG = 10.0  # iron axis within 10 deg of the cage axis
IN_CAGE_R = MOUTH_R_IN  # the metal front's axis points inside the bore
STANDOFF_TOL = 0.002  # each standoff foot within 2 mm of the mat
CENTRE_OK, CENTRE_NEAR = 0.040, 0.100  # board centre from the mat centre
HOLDER_TILT_DEG, HOLDER_MOVE_M = 20.0, 0.020
OFF_TABLE_M = 0.050
CORD_TAUT = 0.95  # anchor to iron rear 95 % of the cord's length: pulled
STILL_M = 0.001
WIN_STEPS = 20  # 1 s at 20 Hz
