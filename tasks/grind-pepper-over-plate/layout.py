"""Every dimension of the scene, in one place. Pure Python.

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is at TOP. The mill's own frame has its origin at the centre of its base (the outlet) and z up
its axis; the head turns about that axis.

Sources (README.md): Peugeot Paris u'Select 12 cm pepper mill, natural beech (Peugeot Saveurs; retailers list it 12
cm tall, 5.9 cm across, 190 g); IKEA OFTAST plate 25 cm (art. 302.589.13: 25 cm across, 2 cm high, 0.36 kg). Every
other figure (the mill's profile, the split between body and head, the grinding torque, the peppercorn load) is an
estimate, stated in task.json "fidelity".
"""

from __future__ import annotations

import math

MM = 0.001
TOP = 0.7475  # station table top (yam_arena.task_file.TABLE_TOP["station"])
CONTROL_HZ = 20.0

# -- the mill (mill frame, mm) ------------------------------------------------------------------------------------
# Peugeot Paris profile, turned beech: a flared foot, a straight lower body, a waist, a shoulder; the crown (the head)
# above a 1 mm gap, and the adjustment knob on top. 120 mm overall.
#   name           radius  z0     z1
BODY_PROFILE = (
    ("foot", 29.5, 0.0, 12.0),
    ("lower", 26.0, 12.0, 52.0),
    ("waist", 22.0, 52.0, 72.0),
    ("shoulder", 26.0, 72.0, 79.0),
)
HEAD_PROFILE = (
    ("crown", 29.0, 80.0, 110.0),
    ("knob", 11.0, 110.0, 120.0),
)
HEIGHT = 120.0
HEAD_PIVOT_Z = 80.0  # the head turns about the mill's axis; its frame sits at the crown's underside
M_BODY = 0.170  # kg: the beech body with its grinding mechanism and about 25 g of peppercorns (est.)
M_HEAD = 0.045  # kg: crown and knob (est.); 190 g empty overall (retailer listing)
BEECH = (0.80, 0.64, 0.44)  # natural lacquered beech
MARK = (0.15, 0.15, 0.15)  # a dark dot on the crown and a line on the shoulder, to count turns on video

# The head's resistance: grinding peppercorns between the burrs. No published figure; a firm wrist twist,
# estimated at 0.15 N m of joint friction, with light viscous damping.
GRIND_FRICTION = 0.15  # N m
GRIND_DAMPING = 0.005  # N m s/rad
GRIND_SENSE = -1.0  # grinding is clockwise seen from above: the head turns NEGATIVE about the mill's +z

MILL_START = (0.205, 0.165)  # world x, y of the mill's base centre at the start: on the table, left of the plate
MILL_FRICTION = (0.6, 0.5)  # lacquered beech on the table and in the pads (est.)

# -- the plate ------------------------------------------------------------------------------------------------------
PLATE_R = 125.0  # mm: IKEA OFTAST 25 cm
PLATE_H = 20.0  # mm
PLATE_MASS = 0.36  # kg
PLATE_CENTRE = (0.225, -0.060)  # world x, y
PLATE_COLOUR = (0.95, 0.95, 0.93)  # white opal glass
PLATE_FRICTION = (0.5, 0.4)  # glass on the table (est.)

# -- scoring -----------------------------------------------------------------------------------------------------------
TURNS_NEEDED = 5
TARGET_R = 0.080  # m: the outlet within 80 mm (in plan) of the plate's centre: over the plate's flat well
CLEARANCE = 0.005  # m: the outlet at least 5 mm above the plate's top (held clear, not resting on the plate)
GRIND_TILT_DEG = 30.0  # the mill within 30 deg of upright while it grinds (the outlet facing down)
STAND_TILT_DEG = 10.0  # stood back up: within 10 deg of upright
STAND_Z_TOL = 0.005  # m: its base within 5 mm of the table top
STAND_OFF_PLATE = 0.160  # m: its axis at least 160 mm from the plate's centre (off the plate: 125 + 29.5 + 5.5)
DROP_TILT_DEG = 60.0  # tipped past 60 deg at any moment: dropped (latched 0)
FALL_Z = 0.020  # m: the mill's base more than 20 mm below the table top: off the table (latched 0)
PLATE_MOVE = 0.030  # m: the plate pushed more than 30 mm, or tipped more than 10 deg (latched 0)
PLATE_TILT_DEG = 10.0
LET_GO = 0.030  # m: both grippers' finger zones at least 30 mm from the mill (rigstate.released)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002  # m: over the last second the base centre and the knob top moved at most 2 mm
EPS = 1e-9

PER_TURN = 0.15
STAND_BONUS = 0.25

# Reset fallback (a trial that starts again without rigstate seeing the reset): the mill back on its start spot,
# upright, with the plate on its own.
RESET_START_TOL = 0.004
RESET_JUMP = 0.05


def mill_shapes(kind: str):
    """(radius m, centre z m, height m) of each cylinder of the body ("body") or the head ("head"), mill frame."""
    rows = BODY_PROFILE if kind == "body" else HEAD_PROFILE
    return [(r * MM, (z0 + z1) / 2 * MM, (z1 - z0) * MM) for _, r, z0, z1 in rows]


def crown_centre_z() -> float:
    """Height (m) of the crown's middle above the mill's base."""
    z0, z1 = HEAD_PROFILE[0][2], HEAD_PROFILE[0][3]
    return (z0 + z1) / 2 * MM


def body_grip_z() -> float:
    """Height (m) above the mill's base where the left gripper holds the body: low on the straight lower body (12 to
    52 mm), so the left gripper stays well below the right hand turning the crown."""
    return 26.0 * MM


def turns(angle_rad: float) -> float:
    return angle_rad / (2.0 * math.pi)
