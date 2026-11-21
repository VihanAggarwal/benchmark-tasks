"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
bench (the station table) top is TOP.

Sources (README.md): the STIHL GTA 26 cordless garden pruner (1.2 kg without battery, 10 cm guide bar, 1/4" P
chain), its STIHL AS 2 battery (pushed into the end of the handle until it clicks; "Depress both locking hooks" to
take it out) and the scabbard that covers the guide bar. Shapes and the sizes the maker does not publish are
estimates from photographs (task.json "fidelity").

The pruner lies on its left side, so its guide bar is flat (parallel to the bench) and points to the robot's right;
its handle, with the battery in its end, points to the robot's left.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the pruner (loose) -------------------------------------------------------------------------------------------------
# Its base frame: origin on the bench under the housing's centre, axes as the world's.
PRUNER_AT = np.array([0.280, 0.020])
M_PRUNER = 1.20  # kg without the battery (maker)
PRUNER_FRICTION = (0.5, 0.4)  # rubberised housing on the bench (est.)
HOUSING = ((-0.045, 0.045), (-0.080, 0.030), (0.0, 0.070))  # x, y, z extents (base frame)
# The control handle is a pistol grip: it leaves the housing's back face at HANDLE_ROOT and runs back and down, 25 deg
# below the bar's line (est.). The battery goes into its end along its axis. HANDLE_Y, BATTERY_STOP_Y, TRIGGER, BAT_HOME
# and HOOK_Y are written along the handle as if it were straight (handle frame: rotate by HANDLE_YAW about z through
# HANDLE_ROOT to get the pruner frame). Lying on its left side, "down" is toward the robot (-x).
HANDLE_YAW = math.radians(25.0)
HANDLE_ROOT = np.array([0.0, 0.030, 0.0])
HANDLE_Y = (0.030, 0.110)  # the tubular handle that holds the battery
HANDLE_CLEAR, HANDLE_WALL = 0.001, 0.004  # the battery's play in the handle; the handle's wall
BATTERY_STOP_Y = 0.050  # the battery's inner end rests against a stop here
TRIGGER = ((-0.036, -0.026), (0.050, 0.075), (0.016, 0.036))  # on the handle's underside, which faces the robot
BAR_Y = (-0.180, -0.080)  # guide bar: tip and root (the housing's face)
BAR_X = 0.017  # half its width over the chain's teeth
BAR_Z, BAR_T = 0.035, 0.005  # centre height; thickness over the chain
PLATE_X, PLATE_T = 0.0135, 0.0011  # the steel bar plate under the chain: half width, thickness (1.1 mm gauge)
PITCH = 0.00635  # 1/4" chain: a cutter every pitch, alternately on the bar's top and bottom face
TOOTH = (0.003, 0.0035, 0.002)  # a cutter: along the edge, across it (from the plate's edge), its height off the plate
NOSE_Y = BAR_Y[0] + PLATE_X + 0.0035  # the nose's centre: round the nose the teeth run on a half circle of radius BAR_X
# The retractable chain guard ("hood"): a plate along the bar's upper edge (+x, the side away from the robot), hinged
# at the housing about z, sprung shut against the teeth; anything pushed along the bar swings it outward.
HOOD_PIVOT = (0.030, -0.082)  # outboard, so its arm swings clear of a scabbard when open
HOOD_Y = (-0.100, -0.150)  # its plate's extent along the bar
HOOD_T, HOOD_H = 0.002, 0.016  # thickness, height (centred on the bar)
HOOD_OPEN = 0.6  # rad, its stop
HOOD_SPRING = 0.03  # N m/rad toward shut (est.)
HOOD_LEAD = math.radians(35.0)  # its free end is turned outward this much: the scabbard rides up it
BODY = (0.95, 0.45, 0.10)  # STIHL orange
DARK = (0.15, 0.15, 0.16)
CHAIN = (0.55, 0.56, 0.58)

# -- the AS 2 battery (loose; held in the handle until both locking hooks are pressed) ---------------------------------
BAT_L, BAT_X, BAT_Z = 0.110, 0.040, 0.036  # length (along y), width, height
BAT_HOME_H = np.array([0.0, BATTERY_STOP_Y + BAT_L / 2, 0.026])  # its centre, handle frame
M_BATTERY = 0.20  # kg (1.4 kg with battery, maker)
HOOK_Y = 0.030  # the two locking hooks' buttons: on its side faces, this far out from its centre
HOOK_SIZE = (0.004, 0.018, 0.016)
HOOK_TRAVEL = 0.0025  # pressed in this far frees it
HOOK_FREE = 0.0018
HOOK_SPRING = 1500.0  # N/m (3.75 N at full travel; est.)
HOOK_CATCH = 0.008  # the hooks catch the battery again if let go before it is drawn this far
HOME_CLICK = 0.002  # pushed back within this of home with the hooks let go, it clicks in
BATTERY_COLOUR = (0.12, 0.12, 0.13)

# -- the scabbard (loose) -----------------------------------------------------------------------------------------------
GUARD_L, GUARD_W, GUARD_H = 0.140, 0.046, 0.018  # length (along y), width, thickness
GUARD_PLATE, GUARD_SIDE = 0.003, 0.002  # inside: 42 x 12 mm over the chain's 34 x 5 mm
GUARD_START = np.array([0.280, -0.300])  # its centre on the bench, open end toward the bar (+y)
M_GUARD = 0.030
GUARD_COLOUR = (0.20, 0.20, 0.22)
GUARD_SEAT_Y = BAR_Y[1] - GUARD_L / 2  # its centre (pruner frame) when its open end is at the bar's root
SEAT_TOL = 0.004  # the catch clicks within this of the seat
ALIGN_X, ALIGN_Z = 0.005, 0.004  # how far off the bar's line the scabbard may be and still count as on it

# -- scoring ------------------------------------------------------------------------------------------------------------
BATTERY_OUT = 0.065  # the battery this far from home: clear of the handle's 60 mm (out, not just unlatched)
CLEAR = 0.030  # grippers this far from the pruner, the battery and the scabbard at the end
TOOTH_CLEAR = 0.010  # no gripper point this close to a bare tooth, ever
BAR_CLEAR = 0.020  # nor this close to the bar or chain while the battery is in
TRIGGER_CLEAR = 0.010
FINGER_GAP = 0.094  # m between the pads when fully open (rigstate opening 1.0)
PAD_MARGIN = 0.008  # each pad's thickness and width beyond the line between the TCP and the pad tips
ON_BAR = 0.040  # the scabbard counts as on the bar once it covers this much of it
ENGAGED = 0.010  # the scabbard has started onto the bar once it covers this much (with the battery in: a foul)
STILL_SPEED = 0.005  # m/s
EPS = 1e-9


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def handle_to_pruner(p) -> np.ndarray:
    return HANDLE_ROOT + rot_z(HANDLE_YAW) @ (np.asarray(p, dtype=float) - HANDLE_ROOT)


def pruner_to_handle(p) -> np.ndarray:
    return HANDLE_ROOT + rot_z(-HANDLE_YAW) @ (np.asarray(p, dtype=float) - HANDLE_ROOT)


BAT_HOME = handle_to_pruner(BAT_HOME_H)  # its centre, pruner frame (turned HANDLE_YAW with the handle)
HANDLE_AXIS = rot_z(HANDLE_YAW) @ np.array([0.0, 1.0, 0.0])  # the way the battery comes out


def at(x: float, y: float, h: float = 0.0) -> np.ndarray:
    return np.array([x, y, TOP + h])


def box_distance(p, lo, hi) -> float:
    """Distance from point p to the axis-aligned box [lo, hi] (0 inside)."""
    p, lo, hi = (np.asarray(v, dtype=float) for v in (p, lo, hi))
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))
