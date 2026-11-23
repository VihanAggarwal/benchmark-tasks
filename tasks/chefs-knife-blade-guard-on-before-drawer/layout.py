"""Every dimension of the scene, in one place. Pure Python (numpy for geometry helpers).

World frame (yam_arena): x forward from the line through the arm bases, y to the robot's left, z up, metres.
The bench top is TOP.

Products (README.md):
- Victorinox Fibrox 8-inch Chef's Knife (5.2063.20): 33.5 cm overall, 20.0 cm blade (4.5 cm wide at heel),
  13.5 cm TPE handle, 180 g.
- Mercer Culinary M33183P 8 x 2 in (20.5 x 5.4 cm nominal) black polypropylene slide-on blade guard, 45 g (est.).
  Modeled as a rigid sleeve 20.5 x 6.4 x 1.4 cm outer (6.0 x 1.0 cm inner slot) so rigid-body simulation provides
  at least 7.5 mm capture clearance per side around the 4.5 x 0.2 cm blade.

Both the knife and the blade guard start flat on a 19.0 x 42.0 x 1.5 cm cutting board with at least 2.1 cm margin
inside every board edge. An open kitchen utensil drawer (18.0 x 38.0 x 5.0 cm interior) sits directly behind the
cutting board.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- cutting board (fixed) ----------------------------------------------------------------------------------------------
BOARD_CENTER = np.array([0.205, 0.000])
BOARD_SIZE = (0.190, 0.420, 0.015)  # x in [0.110, 0.300], y in [-0.210, +0.210]
BOARD_TOP = TOP + BOARD_SIZE[2]
BOARD_COLOR = (0.78, 0.62, 0.42)

# -- open kitchen utensil drawer (fixed) --------------------------------------------------------------------------------
DRAWER_CENTER = np.array([0.410, 0.000])
DRAWER_INNER = (0.180, 0.380, 0.050)  # 18 cm deep (x), 38 cm wide (y), 5 cm tall interior
DRAWER_WALL = 0.010
DRAWER_FLOOR = 0.008
DRAWER_OUTER = (DRAWER_INNER[0] + 2 * DRAWER_WALL,
                DRAWER_INNER[1] + 2 * DRAWER_WALL,
                DRAWER_INNER[2] + DRAWER_FLOOR)  # 20.0 x 40.0 x 5.8 cm; x in [0.310, 0.510], y in [-0.200, +0.200]
DRAWER_FLOOR_Z = TOP + DRAWER_FLOOR
DRAWER_RIM_Z = TOP + DRAWER_OUTER[2]
DRAWER_INNER_X = (float(DRAWER_CENTER[0] - DRAWER_INNER[0] / 2),
                  float(DRAWER_CENTER[0] + DRAWER_INNER[0] / 2))  # (0.320, 0.500)
DRAWER_INNER_Y = (float(DRAWER_CENTER[1] - DRAWER_INNER[1] / 2),
                  float(DRAWER_CENTER[1] + DRAWER_INNER[1] / 2))  # (-0.190, 0.190)
DRAWER_HAZARD_Z = TOP + 0.100  # unsheathed blade entering drawer interior below 10 cm above bench triggers hard 0
DRAWER_COLOR = (0.86, 0.83, 0.78)
DRAWER_FRONT_COLOR = (0.72, 0.66, 0.58)

# -- Victorinox Fibrox 8-inch chef's knife (loose) ----------------------------------------------------------------------
# Base frame: origin at bolster/heel junction on the bottom plane (z=0 is handle bottom).
# Handle runs along +y (0 to +0.135); blade runs along -y (-0.200 tip to 0 heel).
# Spine along +x (+0.014); cutting edge along -x (down to -0.031 at heel).
KNIFE_START = np.array([0.258, 0.0325])  # world x in [0.227, 0.272], y in [-0.1675, +0.1675]
M_KNIFE = 0.180  # kg (maker spec)
KNIFE_FRICTION = (0.5, 0.4)
KNIFE_LIN_DAMPING = 0.5  # 1/s, PhysX body damping: settles contact jitter of the knife resting in the drawer
KNIFE_ANG_DAMPING = 1.0
HANDLE_Y = (0.000, 0.135)
HANDLE_X = (-0.014, 0.014)
HANDLE_H = 0.022
HANDLE_COLOR = (0.14, 0.14, 0.15)
BLADE_Y = (-0.200, 0.000)
BLADE_SPINE_X = 0.014
BLADE_HEEL_W = 0.045
BLADE_EDGE_HEEL_X = BLADE_SPINE_X - BLADE_HEEL_W  # -0.031
BLADE_CX = (BLADE_SPINE_X + BLADE_EDGE_HEEL_X) / 2  # -0.0085
BLADE_Z = HANDLE_H / 2  # 0.011 above board/floor
BLADE_T = 0.002
BLADE_COLOR = (0.76, 0.78, 0.80)
# Five stepped segments (y_lo, y_hi, width) tapering from 4.5 cm heel to 1.0 cm tip along -y
BLADE_SEGMENTS = (
    (-0.050, 0.000, 0.045),
    (-0.100, -0.050, 0.042),
    (-0.145, -0.100, 0.034),
    (-0.178, -0.145, 0.022),
    (-0.200, -0.178, 0.010),
)
# Collision only: each width step is joined by a 12 mm lead-in taper (both sides), drawn inside the stepped outline
BLADE_TAPER_LEN = 0.012
BLADE_TAPER_T = 0.003

# -- Mercer Culinary M33183P slide-on blade guard (loose) ---------------------------------------------------------------
# Base frame: origin at guard center; open mouth at +y, closed tip at -y.
GUARD_L = 0.205
GUARD_W = 0.064  # inner width 0.060 m over 0.045 m blade heel (7.5 mm clearance each side)
GUARD_H = 0.014  # inner height 0.010 m over 0.002 m blade (4.0 mm clearance top and bottom)
GUARD_PLATE = 0.002
GUARD_SIDE = 0.002
GUARD_START = np.array([0.164, -0.075])  # world x in [0.132, 0.196], y in [-0.1775, +0.0275]
M_GUARD = 0.045  # kg (est.)
GUARD_FRICTION = (0.4, 0.3)
GUARD_COLOR = (0.16, 0.17, 0.19)
GUARD_ACCENT_COLOR = (0.32, 0.34, 0.38)
# Sprung internal polypropylene friction-fit retention leaf (hinged inside top plate near mouth)
LEAF_PIVOT_Y = 0.072  # 3.05 cm inside the open mouth (+0.1025)
LEAF_PIVOT_Z = 0.0045  # just under the top plate (inner top face at +0.005); leaf's top face passes through the pivot
LEAF_LEN = 0.032
LEAF_W = 0.030
LEAF_T = 0.0015
# Drawn angled down toward the closed end (a lead-in ramp, like the ribbed tongue of a friction-fit sheath): its tip
# underside sits at guard-local z -2.5 mm, 3.5 mm below the top face of a centred blade (+1.0 mm). The blade cams it
# up through 0.07-0.11 rad; the spring (zero at the drawn angle) then presses the blade with ~1.8-2.8 N.
LEAF_TIP_Z = -0.0025
LEAF_TILT = float(np.arcsin((LEAF_PIVOT_Z - LEAF_TIP_Z - LEAF_T * 0.99) / LEAF_LEN))  # ~0.173 rad (9.9 deg)
LEAF_SPRING = 0.8  # N m/rad toward the drawn angle
LEAF_DAMPING = 0.005
LEAF_LOWER = -LEAF_TILT  # pushed up at most to flat under the top plate
LEAF_UPPER = 0.10  # rad further down (gravity sag only)
LEAF_MASS = 0.008
LEAF_HOLD_LIFT = 0.03  # rad: once the heel catch locks, the leaf is held this far up from where it was (off the blade)

# Seated pose of guard in knife frame (open mouth flush with heel y=0)
GUARD_SEAT_X = BLADE_CX  # -0.0085
GUARD_SEAT_Y = -GUARD_L / 2  # -0.1025
GUARD_SEAT_Z = BLADE_Z  # 0.011
# Heel catch: engages only once the robot has pushed the guard's open mouth to within 1 mm of the heel (or against the
# handle face, which stops it at the seat), then locks it there for transport. It never pulls the guard more than
# 1 mm; earlier versions used 10 mm, which let the catch finish the sheathing for the robot.
CATCH_TOL = 0.001
# Seated (scoring) means the guard covers the ENTIRE cutting edge, heel (y 0) to tip (y -0.200): the open mouth at or
# past the heel and the closed end at or past the tip (the 20.5 cm sleeve is 5 mm longer than the 20.0 cm edge, so
# mouth flush with the heel leaves 5 mm spare beyond the tip). FULL_COVER_TOL is only a pose-measurement tolerance.
FULL_COVER_TOL = 0.0005  # 0.5 mm
ALIGN_X = 0.008  # lateral tolerance on blade line
ALIGN_Z = 0.006  # vertical tolerance on blade line
ALIGN_COS = math.cos(math.radians(8.0))

# -- scoring constants --------------------------------------------------------------------------------------------------
PARTIAL_COVER = 0.100  # 10 cm slid onto blade -> 0.25 tier
CLEAR_DIST = 0.040  # 4 cm gripper clearance from knife and guard
EDGE_CLEAR = 0.008  # 8 mm minimum clearance from any exposed cutting edge
STILL_SPEED = 0.005  # m/s
STILL_DRIFT = 0.002  # 2 mm max displacement during 1.0 s hold
HOLD_S = 1.0  # required consecutive clear + still duration (s)
HOLD_STEPS = int(round(HOLD_S * CONTROL_HZ))  # 20 steps at 20 Hz
FLAT_TILT_RAD = math.radians(10.0)
DRAWER_FLOOR_TOL = 0.015  # handle bottom within 1.5 cm of drawer floor
FINGER_GAP = 0.094  # m between pads when fully open (rigstate opening 1.0)
PAD_MARGIN = 0.008  # each pad's thickness/width beyond TCP-to-tips centerline
EPS = 1e-9


def at(x: float, y: float, h: float = 0.0) -> np.ndarray:
    """Point at (x, y) and height h above bench top TOP."""
    return np.array([float(x), float(y), TOP + float(h)])


def blade_edge_x(y: float) -> float:
    """x-coordinate (knife frame) of the cutting edge at blade coordinate y in [-0.200, 0.000]."""
    for y_lo, y_hi, w in BLADE_SEGMENTS:
        if y_lo - EPS <= y <= y_hi + EPS:
            return BLADE_CX - w / 2
    return BLADE_CX - BLADE_SEGMENTS[-1][2] / 2


def box_distance(p, lo, hi) -> float:
    """Distance from point p to the axis-aligned box [lo, hi] (0 inside)."""
    p, lo, hi = (np.asarray(v, dtype=float) for v in (p, lo, hi))
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))
