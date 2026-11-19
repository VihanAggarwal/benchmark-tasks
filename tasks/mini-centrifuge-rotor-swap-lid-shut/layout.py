"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. The centrifuge's own frame (parts.py, scoring.py) has its origin
on the table under the motor shaft, x toward the lid hinge (away from the arms), z up: heights are above the table.
A rotor's frame has its origin at the middle of its hub's underside, z along the shaft. A strip's frame has its
origin at the middle of its underside, x along the strip.

The unit is a Corning LSE mini microcentrifuge, cat. 6770 (maker's size 174 x 148 x 122 mm, 1.1 kg; 8-tube rotor
6770-RT and 4-strip rotor 6770-RTS on a tool-free quick-release shaft). Everything inside it (shaft key, detent, hub
sizes, lid travel, opening angle, rotor and lid masses) is an estimate (est in task.json) with a measurement step in
the README. The shaft key, the snap and the strip slots are real contact geometry. The lid's catch is the one
declared rule (task.py ``_lid_rule``): the latch state below drives the lid hinge's spring target.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
EPS = 1e-9

# -- the centrifuge: placement and body --------------------------------------------------------------------------------
UX, UY = 0.150, 0.000  # the motor shaft on the table (world); the 148 mm side along x, the release tab toward the arms
HX, HY = 0.074, 0.087  # 148 x 174 mm (maker)
UNIT_MASS = 1.1  # maker, the whole unit
LID_MASS = 0.12  # est: the tinted lid
TAB_MASS = 0.004
PLUNGER_MASS = 0.004
BODY_MASS = UNIT_MASS - LID_MASS - TAB_MASS - 2 * PLUNGER_MASS
UNIT_FRICTION = (0.5, 0.4)  # est: rubber feet on the table, a smooth moulded shaft (one value per articulation)
FOOT_R, FOOT_H = 0.008, 0.004
BOWL_FLOOR = 0.050  # the rotor chamber's floor
DECK = 0.092  # the chamber's rim and the body's top deck
BOWL_R = 0.062  # the chamber, 124 mm across
COWL_X0, COWL_X1, COWL_TOP = -HX, -0.054, 0.112  # the front panel under the lid's tab
WHITE = (0.93, 0.93, 0.91)
CHAMBER_GREY = (0.30, 0.31, 0.32)
FOOT_BLACK = (0.08, 0.08, 0.08)
STEEL = (0.62, 0.63, 0.65)

# -- the lid and its release tab -------------------------------------------------------------------------------------
HINGE_X, HINGE_Z = 0.070, 0.100  # the hinge axis along y at the back top edge (unit frame)
LID_X0, LID_X1 = COWL_X1, HX  # the top plate, front to back
LID_Z0, LID_Z1 = 0.118, 0.122  # its underside and top: the unit 122 mm tall (maker)
SKIRT_Z0 = 0.094  # the side skirts hang to 2 mm over the deck
SKIRT_T = 0.004
LID_TINT = (0.17, 0.15, 0.14)  # the smoked see-through lid, drawn opaque (fractional opacity is not proven on the GPU)
LID_RIM = (0.09, 0.09, 0.10)  # its side skirts and release tab, a shade darker (colour only)
LID_UPPER_DEG = 75.0  # the back stop (est)
LID_UPPER = math.radians(LID_UPPER_DEG)
LID_SPRING = 0.15  # N m/rad: the pop-up spring (est)
LID_DAMP = 0.10  # N m s/rad: about 0.7 s from shut to its stop (est)
LID_OPEN_TARGET = 2.0  # rad: freed, the spring is preloaded toward 2 rad: 0.30 N m shut, 0.10 N m at the stop
LID_LATCH_TARGET = -2.0  # rad: latched, the catch's hold: 0.30 N m pressing it shut
LID_CG = (-0.055, 0.020)  # the lid's centre of mass from the hinge axis (x, z), closed (est, from the shapes)
TAB_U0, TAB_U1 = -0.068, LID_X0  # the tab: a 14 mm lip in front of the lid's front edge, 20 mm wide
TAB_HY = 0.010
TAB_TRAVEL = 0.003  # pressed down 3 mm to its stop (est)
TAB_SPRING = 1500.0  # N/m: 3 N at the 2 mm release point, 4.5 N at its stop (est)
TAB_RELEASE = 0.002  # pressed this far, the catch lets go
TAB_UP = 0.001  # the catch can engage again only with the tab back up within this
CATCH_DEG = 0.6  # the catch engages with the lid within this of shut ...
OPENED_DEG = 3.0  # ... once the lid has left its stop by this since it was freed
FORCED_DEG = 2.0  # latched, the lid more than this off its stop: the catch forced (scored 0)

# -- the shaft, its drive key and the snap ---------------------------------------------------------------------------
SEAT_Z = 0.084  # a seated hub's underside, on the collar's top (unit frame)
COLLAR_R = 0.008
SHAFT_R = 0.004
SHAFT_TOP = SEAT_Z + 0.016  # flush with a seated hub's top
PILOT_R, PILOT_H = 0.0025, 0.0025  # the shaft end, narrowed, with four 45-degree lead-in faces
KEY_HX, KEY_HY = 0.008, 0.0015  # the cross pin: 16 x 3 x 3 mm along x, 5 to 8 mm over the seat (est)
KEY_Z0, KEY_Z1 = SEAT_Z + 0.005, SEAT_Z + 0.008
NOSE_TIP_R = 0.00525  # each plunger's nose reaches 0.75 mm into the hub's bore (half width 4.5 mm)
NOSE_TIP_Z = SEAT_Z + 0.0035
NOSE_FACE_DEG = 30.0  # its two faces 30 degrees from vertical
NOSE_LEN, NOSE_T, NOSE_W = 0.003, 0.001, 0.004
PLUNGER_TRAVEL = 0.0009  # each nose can be pushed back 0.9 mm into the shaft
PLUNGER_SPRING = 1300.0  # N/m: about 1 N each pushed back 0.75 mm; push-on and pull-off about 3 N in all (est)
PLUNGER_DAMP = 0.5

# -- the rotors (the same hub on both) -------------------------------------------------------------------------------
HUB_HALF = 0.011  # 22 mm square hub (est)
HUB_H = 0.016
BORE_HALF = 0.0045  # 9 mm square bore round the 8 mm shaft
KEYWAY_HY, KEYWAY_Z, KEYWAY_X = 0.003, 0.0085, 0.0095  # the slots under the hub's +-x walls that take the pin
GROOVE_Z0, GROOVE_Z1, GROOVE_R = 0.002, 0.005, 0.006  # the snap groove in the +-y walls
ROTOR_FRICTION = (0.6, 0.5)  # est
NOSE_REST = 0.0048  # keyed and let go, a hub rests on the noses this far over its seat
PIN_REST = KEY_Z1 - SEAT_Z  # off the key it rests on the pin, 8 mm over its seat
KEY_TOL_DEG = 10.0  # the pin passes the keyway within this of the key (1.5 mm of play at 8 mm)
CLICK_MM = 0.5  # seated and clicked: the hub's underside within 0.5 mm of its seat ...
PLUNGER_OUT = 0.0002  # ... and both noses back out within 0.2 mm

TUBE_ROTOR_R = 0.048  # 6770-RT: 8 places for 1.5/2.0 mL tubes, 96 mm across flats (est)
TUBE_ROTOR_T = 0.014  # its disc hangs 14 mm under the hub's underside
TUBE_ROTOR_MASS = 0.040
TUBE_HOLE_R, TUBE_HOLE_AT = 0.0055, 0.035
TUBE_ROTOR_GREY = (0.20, 0.21, 0.23)

STRIP_ROTOR_X, STRIP_ROTOR_Y = 0.080, 0.076  # 6770-RTS: 4 strip slots, 110 mm across its corners (est)
STRIP_ROTOR_T = 0.020
STRIP_ROTOR_MASS = 0.030
SLOT_Y = (-0.031, -0.019, 0.019, 0.031)  # four parallel slots along x; opposite pairs (1, 2) and (0, 3)
SLOT_PAIRS = ((1, 2), (0, 3))
SLOT_LX, SLOT_W, SLOT_DEPTH = 0.075, 0.010, 0.017
STRIP_ROTOR_BLUE = (0.44, 0.45, 0.47)  # name kept; drawn mid grey (colour only)

# -- the strips and the rack -----------------------------------------------------------------------------------------
STRIP_L = 0.072  # Axygen PCR-0208-AF-C: 8 x 0.2 mL tubes at 9 mm pitch, attached flat caps (cap row 72 mm, est)
STRIP_W = 0.007
STRIP_H = 0.021
CAP_T = 0.0025
TUBE_R, TIP_R, TIP_H = 0.003, 0.0016, 0.004
PITCH = 0.009
STRIP_MASS = 0.00166  # 1.5 g empty (est) and 8 x 20 uL of water sealed under the caps
STRIP_FRICTION = (0.6, 0.5)
STRIP_CLEAR = (0.97, 0.97, 0.96)  # clear polypropylene, drawn opaque white
SEATED_Z = -SLOT_DEPTH  # a seated strip's underside on the slot floor (rotor frame)

RACK_X, RACK_Y = 0.200, -0.200  # a 96-place 0.2 mL rack, long side along y (8 wells of a column along x)
RACK_LX, RACK_LY, RACK_H = 0.086, 0.127, 0.020
RACK_MASS = 0.045
RACK_WHITE = (0.96, 0.96, 0.95)
WELL_DARK = (0.42, 0.43, 0.45)
RACK_COLS = (4, 9)  # the two loaded columns
RACK_WALL = 0.0015
RACK_FLOOR = 0.007  # a strip's underside on the support bar, 7 mm over the table
RACK_HOLE_W = 0.009


def rack_col_y(col: int) -> float:
    """A column's offset along y from the rack's centre (12 columns at 9 mm)."""
    return (col - 6.5) * PITCH


STRIP_STARTS = tuple((RACK_X, RACK_Y + rack_col_y(c), RACK_FLOOR) for c in RACK_COLS)  # (x, y, underside z)

# -- the starting places of the loose rotors --------------------------------------------------------------------------
STRIP_ROTOR_START = (0.250, 0.200, STRIP_ROTOR_T)  # flat on the table, hub up (origin: the hub's underside)
STRIP_ROTOR_START_YAW = math.pi / 2  # its key a quarter turn from the shaft's
TUBE_ROTOR_START = (UX, UY, SEAT_Z)  # seated on the shaft
TUBE_STORE = (0.100, 0.200)  # where the demo sets the tube rotor down

# -- the scorer's numbers --------------------------------------------------------------------------------------------
UNIT_MOVED = 0.015  # the unit slid 15 mm or more (3-D) at any step: traced only, no effect on the score
UNIT_TILT_DEG = 10.0  # the unit tilted over 10 degrees at any step (tipped): 0
DROP_Z = -0.020  # a rotor's or strip's lowest point 20 mm under the table top: off the table: 0
HANDS_OFF = 0.030  # every finger zone 30 mm from the unit, the rotors and the strips ...
HANDS_WINDOW_S = 1.0  # ... over the last second
STRIP_TOL_XY = 0.0025  # a strip seated: its middle within 2.5 mm of a slot's middle line (across) ...
STRIP_TOL_X = 0.004  # ... and 4 mm along it, ...
STRIP_TOL_Z = 0.002  # ... its underside within 2 mm of the slot floor ...
STRIP_TILT_DEG = 10.0  # ... tilted under 10 degrees from the rotor and turned under 15 from the slot
STRIP_TURN_DEG = 15.0
ROTOR_OFF_AXIS = 0.001  # the strip rotor seated: its hub within 1 mm of the shaft's axis, tilted under 3 degrees
ROTOR_TILT_DEG = 3.0
ON_SHAFT_AXIS = 0.0025  # on the shaft (0.25): within 2.5 mm of the axis and its underside under 10 mm over its seat
ON_SHAFT_RISE = 0.010
ON_SHAFT_TILT_DEG = 8.0
STORED_LOW = 0.004  # the tube rotor stored: its lowest point within 4 mm of the table top, ...
STORED_TILT_DEG = 10.0  # ... tilted under 10 degrees, ...
STORED_CLEAR = 0.005  # ... and 5 mm or more clear of the unit's footprint
START_TOL = 0.004  # the parts on their starting marks (the reset fallback) ...
RESET_JUMP = 0.050  # ... reached by a jump no trial makes in one step


# -- the lid's latch: one state machine, run by the rule in the simulator and by the scorer ------------------------
class Latch:
    """The catch on the lid's front edge. Shut and latched at the start. Pressing the tab 2 mm frees it (the spring
    then lifts the lid as soon as nothing holds it down); it catches again only after the lid has opened 3 degrees
    and come back within 0.6 degrees of shut with the tab up. ``frees`` counts releases, ``latches`` catches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.latched = True
        self.opened = False
        self.frees = 0
        self.latches = 0
        self.reopened = False  # freed again after a catch that followed an opening (a run would have started)

    def step(self, lid_rad: float, tab_m: float) -> bool:
        """Feed one step's lid angle (rad, 0 shut) and tab travel (m, + pressed); returns latched."""
        if not (math.isfinite(lid_rad) and math.isfinite(tab_m)):
            return self.latched
        if self.latched:
            if tab_m >= TAB_RELEASE - EPS:
                self.latched = False
                self.opened = False
                if self.latches > 0:
                    self.reopened = True
                self.frees += 1
        else:
            if lid_rad > math.radians(OPENED_DEG):
                self.opened = True
            if self.opened and lid_rad <= math.radians(CATCH_DEG) + EPS and tab_m < TAB_UP:
                self.latched = True
                self.latches += 1
        return self.latched

    def target(self) -> float:
        """The lid hinge's spring target for this state (rad, outside the limits: a preload)."""
        return LID_LATCH_TARGET if self.latched else LID_OPEN_TARGET

    def snapshot(self) -> dict:
        return {"latched": self.latched, "opened": self.opened, "frees": self.frees, "latches": self.latches,
                "reopened": self.reopened}


def lid_spring_torque(lid_rad: float, latched: bool) -> float:
    """N m opening (+) from the spring at ``lid_rad``."""
    target = LID_LATCH_TARGET if latched else LID_OPEN_TARGET
    return LID_SPRING * (target - lid_rad)


def lid_gravity_torque(lid_rad: float) -> float:
    """N m opening (+) from the lid's weight (negative while its centre is in front of the hinge)."""
    cx, cz = LID_CG
    x = cx * math.cos(lid_rad) + cz * math.sin(lid_rad)  # the centre's x from the hinge when open lid_rad
    return LID_MASS * G * x


def lid_point(rho: float, lid_rad: float, w: float = LID_Z1 - HINGE_Z) -> tuple[float, float]:
    """A point of the lid ``rho`` in front of the hinge and ``w`` over it when shut, at ``lid_rad`` (unit frame x, z)."""
    c, s = math.cos(lid_rad), math.sin(lid_rad)
    u = -rho
    return HINGE_X + u * c + w * s, HINGE_Z - u * s + w * c
