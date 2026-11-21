"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
bench (the station table) top is TOP.

Sources (README.md): the Bosch 11224VSRC 7/8" SDS-plus Bulldog rotary hammer (15 in long, 7.9 lb) and SDS-plus bits
(a 10 mm shank with two open drive slots and two closed locking grooves; inserted "while twisting and pushing inward
until it locks automatically", removed by pulling the locking sleeve back). Shapes and the sizes Bosch does not publish
are estimates from photographs (task.json "fidelity").

The hammer lies on its left side on the bench with its chuck pointing at the robot: the chuck's axis runs along x, the
housing stretches away from the robot, its top faces the robot's right (-y) and its pistol grip, at the back, points to
the robot's left (+y).
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the rotary hammer (loose) ------------------------------------------------------------------------------------------
# Its base frame: origin on the bench under the chuck's nose (the dust cap's face), axes as the world's; the chuck's
# axis runs along +y from the nose into the tool, AXIS_Z above the bench.
HAMMER_AT = np.array([0.300, 0.000])
HAMMER_YAW = -math.pi / 2  # its frame's +y (the chuck's axis, into the tool) points away from the robot (+x)
M_HAMMER = 3.58  # kg (7.9 lb, maker)
HAMMER_FRICTION = (0.6, 0.5)  # rubber overmould on the bench (est.)
AXIS_Z = 0.038
LENGTH = 0.381  # 15 in, nose to the back of the housing (maker)
BODY_X = (-0.060, 0.034)  # the housing's underside (toward the robot) to its top
BODY_Z = (0.0, 0.076)
NOSE_Y = (0.040, 0.085)  # the gear-case nose, a cylinder round the chuck's axis
NOSE_R = 0.027
GRIP_Y = (0.300, 0.345)  # the pistol grip, from the housing's underside toward the robot
GRIP_X = (-0.190, -0.060)
BORE_R = 0.006  # the tool holder's bore (shank 10 mm: 1 mm play each side; SDS shanks sit loose)
BARREL_R = 0.012
BARREL_Y = (0.004, 0.045)  # the tool holder's barrel, behind the dust cap
CAP_Y = (-0.012, 0.002)  # the rubber dust cap: a cone in front of the nose that leads a shank into the bore
CAP_R = 0.014  # its outside
CAP_MOUTH = 0.012  # its inside at the front, narrowing toward the bore
CAP_EXIT = 0.0054  # and at its back (a rubber lip round the 10 mm shank)
DEPTH = 0.040  # a shank's home depth from the nose (its end against the bore's bottom)
SLEEVE_Y = (0.008, 0.040)  # the locking sleeve round the barrel
SLEEVE_R = (0.0135, 0.0225)
SLEEVE_TRAVEL = 0.008  # pulled back toward the housing (+y) this far
SLEEVE_FREE = 0.006  # past this the locking balls let a bit go
SLEEVE_SPRING = 600.0  # N/m: 4.8 N at full travel (est.)
KEY_DEPTH = 0.012  # the drive keys: a shank passes this depth only with its slots lined up on them
KEY_TOL = math.radians(8.0)
KEY_TRAVEL = 0.014  # the stand-in gate sits this far off the axis when open: wholly clear of the bore
BODY = (0.06, 0.29, 0.55)  # Bosch blue
DARK = (0.12, 0.12, 0.13)
STEEL = (0.62, 0.63, 0.66)

# -- the bits (loose) ---------------------------------------------------------------------------------------------------
SHANK_R, SHANK_L = 0.005, 0.040
OLD_R, OLD_L = 0.003, 0.170  # the 6 mm bit in the chuck: 6 x 210 mm
NEW_R, NEW_L = 0.005, 0.120  # the 10 mm bit on the bench: 10 x 160 mm
M_OLD, M_NEW = 0.035, 0.080
NEW_START = np.array([0.200, -0.110])  # its centre on the bench, lying along x, shank away from the robot
NEW_ROLL = math.radians(45.0)  # its slots lie this far round from its frame's x (so, as it lies, 45 deg off the keys)

# -- scoring ------------------------------------------------------------------------------------------------------------
OUT = 0.012  # a bit's shank end this far in front of the nose (clear of the dust cap): out of the tool holder
STARTED = 0.010  # the new bit's shank this far into the bore
HOME_TOL = 0.002  # within this of home depth: home (it locks)
ON_AXIS = 0.003  # a shank's end within this of the chuck's axis to count as in the bore
ON_AXIS_DEG = 6.0  # and the bit within this of the axis
CLEAR = 0.030
STILL_SPEED = 0.005
FINGER_GAP = 0.094
PAD_MARGIN = 0.008
EPS = 1e-9


def at(x: float, y: float, h: float = 0.0) -> np.ndarray:
    return np.array([x, y, TOP + h])


def box_distance(p, lo, hi) -> float:
    p, lo, hi = (np.asarray(v, dtype=float) for v in (p, lo, hi))
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))
