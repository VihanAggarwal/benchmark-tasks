"""Every dimension and threshold of the task, in one place (metres, kilograms, radians unless named, station frame).

The foot end of a manual hospital bed: the crank's gearbox, the crank stub with its fold-away handle, and the head
section's deck on its hinge.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named.

The bed's base frame sits on the table at (0, 0, TOP), not turned, so its local axes ARE the world axes and a local z
is the height above the table. The crank shaft runs along +x at (y SHAFT_Y, z SHAFT_Z), out of the gearbox's front face
(x 0.340) toward the robot. Positive crank angle turns right-handed about +x, which is clockwise seen from the robot:
it raises the deck. The handle's pin is vertical when the crank angle is 0, so the handle folds and unfolds in the
horizontal plane; unfolded, the arm points to -y at crank angle 0 and the knob projects toward the robot (-x).
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the gearbox, stub and handle ----------------------------------------------------------------------------------
SHAFT_Y, SHAFT_Z = -0.100, 0.115  # the shaft's axis (y, height above the table)
BOX_X = (0.340, 0.430)  # fixed grey steel gearbox 90 x 90 x 160 mm, front face at x 340
BOX_Y = (SHAFT_Y - 0.045, SHAFT_Y + 0.045)
BOX_Z = (0.0, 0.160)
STUB_R = 0.006  # 12 mm steel shaft
STUB_X = (0.210, 0.340)  # projecting 130 mm from the face; the tip, where the handle is pinned, is at x 210
PIN_X = STUB_X[0]  # the fold pin: vertical, through (x 210, y SHAFT_Y), at the shaft's height
PIN_R, PIN_LEN = 0.004, 0.030
ARM_LEN = 0.070  # pin to knob axis (estimate: the real lever is measured on the bed)
ARM_W, ARM_T = 0.020, 0.005  # steel flat 20 x 5 mm; the 5 mm is along the pin
ARM_BACK = 0.010  # the flat extends this far behind the pin and beyond the knob axis (round ends)
KNOB_R, KNOB_LEN = 0.0125, 0.070  # red nylon, 25 mm across, 70 mm long, spinning freely on its pin
KNOB_FACE = ARM_W / 2  # the knob starts at the arm's face, 10 mm off the arm's centre line
KNOB_OFF = KNOB_FACE + KNOB_LEN / 2  # arm's centre line to the knob's centre, along the knob axis: 45 mm
GRIP_OFF = 0.010  # the gripper's TCP: this far from the knob's centre toward its free end (pads 31 mm off the arm)
FOLD_OPEN = math.pi / 2  # unfolded: radial, 90 degrees out

CRANK_MASS = 0.115  # steel stub 130 x 12 mm: 14.7 cm3 at 7.85 g/cm3
ARM_MASS = 0.100  # the flat 95 x 20 x 5 mm (75 g) and the pin (12 g), plus the hinge hardware
KNOB_MASS = 0.039  # red nylon, 34.4 cm3 at 1.14 g/cm3
CRANK_LIMIT = 6.0  # rad either way (inside the kit's 2 pi); the rule turns the angle back by 2 pi past +-pi
KNOB_LIMIT = 6.0
WRAP_AT = math.pi
BEARING_FRICTION = 0.05  # N m, the shaft's bearing and the worm's own friction (estimate)
KNOB_FRICTION = 0.002  # N m on its pin
KNOB_DAMP = 0.0005
FOLD_FRICTION = 0.04  # N m: the hinge's spring-ball detent, modelled as joint friction with hard stops (estimate)
FOLD_DAMP = 0.002
CRANK_K = 2.0  # N m/rad: the crank's drive spring, whose target the load rule sets every step
CRANK_DAMP = 0.04  # N m s/rad
T_LOAD = 0.15  # N m of load while the deck is being raised (see README: 3 kg deck, worm drive)
W_SAT = 0.5  # rad/s: the load is full at this crank speed and above, proportional below

# -- the head deck ---------------------------------------------------------------------------------------------------
DECK_HINGE_X = 0.500
DECK_HINGE_Z = 0.270  # the hinge pin's height above the table
DECK_LEN, DECK_WID, DECK_T = 0.450, 0.550, 0.020  # head deck panel 450 x 550 x 20 mm
DECK_Y = SHAFT_Y  # centred on the shaft's y
DECK_MASS = 3.0
POST_X = (0.480, 0.520)  # the frame post under the deck's hinge edge: 40 mm in x
POST_Y = (DECK_Y - 0.300, DECK_Y + 0.300)
POST_Z = (0.0, DECK_HINGE_Z - DECK_T / 2)
DECK_LIMIT = math.radians(80.0)
DECK_K = 1000.0  # N m/rad: the deck's drive spring toward the rule's target angle
DECK_DAMP = 40.0  # N m s/rad
DECK_PER_TURN = math.radians(5.0)  # the declared worm ratio: 5.0 degrees of deck per full crank turn

# -- the end state (scoring.py Tracker) --------------------------------------------------------------------------------
ANGLE_OK = (math.radians(30.0), math.radians(45.0))  # 1.0 and 0.5 tier
ANGLE_NEAR = (math.radians(15.0), math.radians(60.0))  # 0.25 tier (folded, outside the OK range)
ANGLE_GATE = math.radians(60.0)  # more than this at any time: 0
FOLDED_MAX = math.radians(15.0)  # the arm within 15 degrees of the shaft axis (fold angle)
LET_GO = 0.030  # every finger zone of both grippers at least 30 mm from the shaft, arm and knob
STILL_WINDOW_S = 1.0
STILL_ANGLE = math.radians(0.5)  # the deck's angle changes under 0.5 degrees over the last second
EPS = 1e-9
RELEASE_BODIES = ("crank", "fold", "knob")
RESET_JUMP = math.radians(8.0)  # fallback reset detection: the deck falls back to flat from at least this far

# -- colours (sRGB) -----------------------------------------------------------------------------------------------------
GREY = (0.55, 0.57, 0.60)
STEEL = (0.72, 0.72, 0.75)
BLACK = (0.06, 0.06, 0.07)
RED = (0.80, 0.10, 0.10)
DECK_GREY = (0.78, 0.80, 0.82)
FRAME_GREY = (0.40, 0.42, 0.45)

TURNS_MIN, TURNS_MAX = 6.0, 9.0  # the deck's 30 and 45 degrees at 5 degrees a turn


def knob_centre_local(phi: float, fold: float) -> tuple[float, float, float]:
    """The knob's centre in the base frame (m, z above the table) at crank angle ``phi`` and fold angle ``fold``."""
    # in the shaft-turned frame: the pin at (PIN_X, SHAFT_Y); the arm points (cos f, -sin f); the knob sits KNOB_OFF
    # from the arm's line toward (-sin f, -cos f)
    ax, ay = math.cos(fold), -math.sin(fold)
    kx, ky = -math.sin(fold), -math.cos(fold)
    x = PIN_X + ARM_LEN * ax + KNOB_OFF * kx
    dy = ARM_LEN * ay + KNOB_OFF * ky  # offset from the shaft's y
    c, s = math.cos(phi), math.sin(phi)
    return (x, SHAFT_Y + c * dy, SHAFT_Z + s * dy)


def knob_axis_local(phi: float, fold: float) -> tuple[float, float, float]:
    """The knob's axis (toward its free end) in the base frame."""
    kx, ky = -math.sin(fold), -math.cos(fold)
    c, s = math.cos(phi), math.sin(phi)
    return (kx, c * ky, s * ky)


def grip_point_local(phi: float, fold: float) -> tuple[float, float, float]:
    """Where the right TCP goes to hold the knob: its centre, GRIP_OFF toward the free end."""
    c = knob_centre_local(phi, fold)
    a = knob_axis_local(phi, fold)
    return tuple(c[i] + GRIP_OFF * a[i] for i in range(3))
