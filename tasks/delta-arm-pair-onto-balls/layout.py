"""Every dimension, mass, threshold, pose and shape of the task in one place, and the geometry the scorer, the demo,
the offline model and the tests share, as plain functions. Pure Python and NumPy: importable without a simulator.

Sources: the task spec (research/specs/v3/delta-arm-pair-onto-balls.md, "spec 3.2" and so on) and its overrides
(delta-arm-pair-onto-balls.overrides.md, "V2" and so on), which win where the two differ. Every value here is a
design value or computed from design values; nothing below was measured on the simulator.

Frames. Table frame: origin at the midpoint between the arm bases on the table top, x forward, y to the robot's
left, z up (spec 3.1). World z = TOP + table z. Shape tables below are in millimetres in each assembly's own base
frame (yaw 0, so its axes are the table's), as mechkit wants them: every joint at 0, the drawn pose.

The arm pair (spec 3.2). Arm coordinates (u outboard from the pair's centre line, v along the arm from its pivot
toward the free end, w up, origin at the pivot) are mapped into the slider's base frame and then turned ARM_TURN
about the pivot so the free end stands 5.0 mm nearer the centre line. ``arm_a`` sits on the -x ball (table x 0.272),
hinged to the base; ``arm_b`` on the +x ball (table x 0.328), hinged to ``arm_a``. The arms run toward -y (the
robot's right), so v points along -y.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station's table top (world z); the setup's z is above it
MM = 0.001
CONTROL_HZ = 20.0
PHYSICS_HZ = 240.0
RUN_LIMIT_STEPS = 900  # spec 1: 900 steps (45 s at 20 Hz)

# -- ball studs and stud pairs (spec 1 setup, 3.2, 3.3) -------------------------------------------------------------
BALL_R = 5.0  # mm: DIN 71803, 10 mm ball
STUD_SPACING = 56.0  # mm between the ball centres of one stud pair
BALL_Z = 13.0  # mm: 8 mm plate + half of the 10 mm bar, on slider and effector alike
STUD_BAR = (34.0, 10.0, 10.0)  # mm: along the pair, across, high
STUD_HEX = (4.0, 4.0, 19.0)  # radius, length, offset from the bar's centre along the pair (mm)
STUD_NECK = (3.0, 3.5, 22.75)
# V3: each ball is three crossed discs of radius 5 about its centre, in its stud pair's frame (a along the bar, n
# level and square to it, z up). Along a 4 mm long, so its flat ends stay 1.0 mm inside a seated mouth plane. Along n
# and z only 2 mm long: each gives a sphere's round profile in the plane that holds the socket's axis, and its flat
# ends overhang the 8 mm hole's edge only once the ball is 4 - 1 = 3 mm off the axis (6 mm discs did at 1 mm).
BALL_CYL_LEN_A = 4.0
BALL_CYL_LEN_N = 2.0
BALL_CYL_LEN_Z = 2.0

# -- slider (V2 replaces spec 3.2's base table) ---------------------------------------------------------------------
SLIDER_ORIGIN = (0.300, 0.275)  # table frame, m (spec 3.2)
SLIDER_PLATE = ((50.0, 30.0, 8.0), (0.0, 10.0, 4.0))  # size, centre (mm): V2, 50 mm wide (was 100)
KEEPER_PIN_R = 2.0  # mm, 4 mm steel dowel (V2)
KEEPER_PIN_LEN = 22.0  # mm: standing in the bench from z 0 to 22 (V2)
KEEPER_PIN_X = 44.5  # mm either side of the centre line (spec 1 numbers table: 28 + 14 + 0.5 + 2)

# -- arm pair (spec 3.2) --------------------------------------------------------------------------------------------
ARM_LEN = 315.0  # mm between socket centres (the arm length published Klipper configurations for the printer use)
SOCKET_REST_GAP = 52.0  # mm between the free sockets at rest (stop screw)
ARM_TURN = math.asin((STUD_SPACING + 2 * 3.0 - SOCKET_REST_GAP) / 2.0 / ARM_LEN)  # 0.01587 rad: free end 5.0 mm in
TUBE = (8.0, 329.0, 8.0)  # mm, u x v x w (spec 3.2)
TUBE_CENTRE = (10.0, 157.5, 0.0)  # inner face at u 6.0, outer at 14.0
TUBE_INNER_U = 6.0
TUBE_OUTER_U = 14.0
BLOCK_PROUD = 3.0  # mm proud of the tube's inner face
BLOCK_SIZE = 14.0  # mm square
SOCKET_HOLE = 8.0  # mm square hole
BLOCK_U = 4.5  # block centre: u 3.0 to 6.0
MOUTH_U = 3.0  # the mouth plane
FREE_V = ARM_LEN  # the free socket's centre along the arm
SEAT_DEPTH = BALL_R - math.sqrt(BALL_R ** 2 - (SOCKET_HOLE / 2) ** 2)  # 2.0 mm over the ball (spec 1 numbers)
SEAT_E = math.sqrt(BALL_R ** 2 - (SOCKET_HOLE / 2) ** 2)  # 3.0 mm: a seated ball's centre inboard of the mouth
STRUT_V = (30.0, 285.0)  # mm from each pivot: where the springs and eye bolts are (spec 3.2)
STRUT_LEN = 50.0  # drawn struts, from the inner face inward
STRUT_R = {"arm_a": 3.3, "arm_b": 1.5}
STOP_SCREW_V = 295.0  # mm: 20 mm from the free socket (spec 3.2, 3.7)
STOP_SCREW_R = 2.0
STOP_SCREW_LEN = 58.6  # mm inboard of arm_a's inner face: its tip meets arm_b's inner face at rest
ARM_MASS = {"arm_a": 0.0241, "arm_b": 0.0229}  # kg: spec 1 numbers table (arm_a carries the stop screw)
ARM_FRICTION = (0.25, 0.20)  # surface friction, steel and aluminium on nylon (spec 3.2)

# joints (spec 3.2)
ARM_A = "arm_a"
ARM_B = "arm_b"
ARMS_PAIR = (ARM_A, ARM_B)
ARM_A_LIMITS = (-0.15, 0.15)  # rad: never reached; the rest-bar pins stop the swing at +-0.050
ARM_A_DAMPING = 0.02  # N m s/rad
ARM_A_FRICTION = 0.002  # N m
ARM_B_LIMITS = (0.0, 0.146)  # rad: lower = the stop screw, upper never reached
SPRING_AT_SOCKETS = 0.2  # N/mm: the two springs' pull, as one rate at the sockets (spec 3.2)
ARM_B_SPRING = SPRING_AT_SOCKETS * 1000.0 * (ARM_LEN * MM) ** 2  # 19.85 N m/rad
PRELOAD_N = 1.2  # N at the sockets to start opening
ARM_B_TARGET = -PRELOAD_N * (ARM_LEN * MM) / ARM_B_SPRING  # -0.01905 rad, written every step (on_step)
ARM_B_DAMPING = 0.10  # N m s/rad
ARM_B_FRICTION = 0.0
ARM_A_VOID_MARGIN = 0.02  # rad: a run with arm_a this near a limit is void (spec 3.2)

# -- real springs (spec 3.7; the scorer uses no force) --------------------------------------------------------------
WIRE_D = 0.6  # mm, music wire
SPRING_OD = 6.6  # mm
SPRING_COILS = 25
SHEAR_G = 79300.0  # N/mm2
SPRING_INIT_TENSION = 1.0  # N
SPRING_FREE_LEN = 26.4  # mm inside the loops: (25 + 1) x 0.6 + 2 x (6.6 - 1.2)
EYE_BOLT_IN = 16.28  # mm from the tube's inner face to the inside of the eye
FAR_SPRING_REST_STRETCH = 8.1  # mm (spec 3.7)

# -- rest bar and stop pins (spec 3.4) ------------------------------------------------------------------------------
REST_BAR_ORIGIN = (0.300, 0.100)  # table frame, m
REST_BAR = ((120.0, 15.0, 8.5), (0.0, 0.0, 4.25))  # 0.5 mm lower than the real 9 mm bar
STOP_PIN_R = 2.0
STOP_PIN_LEN = 21.0
STOP_PIN_X = 50.0  # mm either side of x 300
STOP_PIN_Z = 10.5
REST_BAR_V = 175.0  # mm from the slider's balls
REST_BAR_FRICTION = (0.20, 0.15)

# -- effector (spec 3.3) --------------------------------------------------------------------------------------------
EFFECTOR_ORIGIN = (0.300, -0.108)  # table frame, m: the plate's centre on the table
EFFECTOR_END = (0.300, -0.068)  # where the demo leaves it: front balls on the socket line (spec 3.3)
EFFECTOR_MASS = 0.139  # kg: plate 38.3 g + post 25.4 g + three stud pairs at 25.1 g (spec 1 numbers)
# As the kit spreads it over the colliders by volume (overlaps counted twice, as PhysX does; computed with
# center_of_mass): plate 87.3 g, post 28.9 g, stud pairs 22.8 g; centre of mass 11.5 mm up, 2.1 mm behind the centre
# (bench: 14.2 mm up, 1.8 mm behind).
EFFECTOR_FRICTION = (0.30, 0.25)
HEX_FLATS = 66.0  # mm across flats
HEX_THICK = 8.0
HEX_BOX = (38.11, 66.0, 8.0)  # three of them, yaw 0 and +-60 deg
HEX_CORNER_R = HEX_FLATS / 2.0 / math.cos(math.radians(30.0))  # 38.1 mm
POST = ((20.0, 20.0, 50.0), (0.0, -10.0, 33.0))  # size, centre (mm): z 8 to 58, 10 mm behind the centre
STUD_NORMALS_DEG = (90.0, 210.0, 330.0)  # edge normals that carry a stud pair; 90 faces the arms
STUD_PAIR_OUT = 28.0  # mm from the centre along the normal (bar flush with the 33 mm flat)

# -- scorer (spec 4.1 as V4 to V7 change it; scoring text numbers) --------------------------------------------------
WIDTH_EXTRA = 2.0 * (BLOCK_PROUD + TUBE[0])  # 22.0 mm: sockets' distance to width over the tubes' outer faces
WIDTH_REST = SOCKET_REST_GAP + WIDTH_EXTRA  # 74.0
WIDTH_ON = STUD_SPACING + 2.0 * SEAT_E + WIDTH_EXTRA  # 84.0
WIDTH_PASS = STUD_SPACING + 2.0 * BALL_R + WIDTH_EXTRA  # 88.0: the sockets 66 mm apart pass the crowns
WIDTH_TOL = 1.5  # mm
RHO_TOL = 1.5  # mm: ball centre to the socket's axis line, in 3D (V4)
E_MIN = 0.0  # mm: the ball's centre at or inboard of the mouth plane
CLEAR = 0.030  # m: a gripper's zone gap to the arm pair and to the effector (V6; rigstate.zone_gap)
FINGER_LEN_TEXT = 38.0  # mm, the scoring text (rigstate.FINGER_LEN)
ZONE_R_CLOSED_TEXT = 16.0  # mm, the scoring text (rigstate.zone_radius(0) = 15.7)
ZONE_R_OPEN_TEXT = 57.0  # mm, the scoring text (rigstate.zone_radius(1) = 56.6)
FLAT_TOL = 1.0  # mm: each bottom corner of the plate within 1 mm of the table top (V5)
STILL_MOVE = 1.0  # mm: none of the eight points moved more (V7)
STILL_WINDOW_S = 1.0
EPS = 1e-9
TIERS = (0.25, 0.5, 1.0)
RESET_START_TOL = 0.002  # m: the effector back on its start pose ...
RESET_JUMP = 0.020  # m: ... after a jump no trial makes in one control step: a new episode

# -- catch window (spec 2, 5.3; computed from the built shapes with the kit's distance(), not measured) ----------------
# Seat depth e (ball centre inboard of the mouth plane) at which a closing socket first touches a three-disc ball
# that is d mm off its axis: equal to a sphere's sqrt(25 - (4 - d)^2) from d 0 to 2.75 mm along the arm and in height
# (3.00 / 4.00 / 4.77 / 4.84 mm at d 0 / 1.0 / 2.5 / 2.75), then 5.0 from 3.0 mm (a disc's flat end on the block's
# face: no centring). Off both ways at once e follows the larger offset. At 2.75 mm the contact's slope (1.25 / 4.84
# = 0.26) is down to the 0.25 friction of nylon on steel, so the window used is 2.5 mm (slope 0.31). At 2.5 mm the two
# sockets push a released effector with 0.3 to 0.5 N against 0.41 N of table friction (marginal); at 2.0 mm with 1.1
# to 1.3 N.
CATCH_ALONG = 2.5  # mm: each ball off its socket's axis along the arms (and in height) when the sockets close
CATCH_X = 2.0  # mm: open pair's centre line against the stud pair's: 2.0 clear at W 92, no lead-in counted
# Yaw, the pair as built (kit distance on the built shapes). Turned about the effector's centre so that arm_b's
# ball moves off its axis, that ball is 2.2 mm off at 2 deg, 2.5 mm at 2.5 deg and 2.8 mm at 3 deg: 28 sin(yaw)
# plus the stagger where the pair stops, 1.2 to 1.35 mm (arm_a follows the stud pair sideways). Turned about the
# stud pair's midpoint: 2.3 mm at 2.5 deg, 2.6 mm at 3 deg (stagger 1.1 mm). One socket alone takes 5 deg (2.44 mm,
# a sphere's seat 4.75 mm), and so does the other sense (2.3 and 1.8 mm off at -5 deg).
CATCH_YAW_DEG = 2.5
JAM_SHORT = 12.0  # mm: a ball pushed at the closed pair stops this far short of the socket line

# -- the arms' grips (spec 5.1, 5.2; demo only, never the scorer) ---------------------------------------------------
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # pads close along world x
ROLL = {"left": -0.5, "right": 0.6}  # rad about world x (tiltgrip.py): the wrists lean apart
FINGER_THICK = 8.0  # mm (rigstate.FINGER_THICK)
FINGER_STIFFNESS = 5000.0  # N/m: the gripper's position control, 50 N cap (spec 7 O4)
LEFT_STATION = (0.300, 0.030)  # TCP, table frame: 245 mm from the pivots, 70 mm from the free sockets
STATION_V = (SLIDER_ORIGIN[1] - LEFT_STATION[1]) / MM  # 245 mm
LEFT_Z = 0.022  # TCP height: pad tips at 10.6 mm, lowest pad corner 5.8 mm
LEFT_ENTER_GAP = 0.008  # m pad gap entering between the tubes
SPREAD_F = -0.03375  # finger target: 67.5 mm pad gap (GPU C4, see below)
# GPU C4: W 80.9, 85.0, 89.2 at commanded gaps 56, 60, 64 (model 86.7, 91.2, 95.8); W = 80.9 + 1.0375 * (gap - 56)
# gives 92.8 at 67.5, and W creeps down about 1 mm over 4 s. The offline model reads about 6 mm of gap high.
GPU_GAP_LOSS_MM = 6.0  # mm of command added to the model's pad gap (task.spread_finger); offline_delta removes it
HOLD_WIDTH = 92.0  # mm
RIGHT_PICK = (0.300, -0.118)  # TCP over the post's centre (post centre at effector y -10)
RIGHT_Z = 0.040  # pad tips at 29 mm, lowest pad corner 23.6 mm: above the studs' tops at 18 mm
RIGHT_OPEN = -0.030  # 60 mm
RIGHT_GRIP = -0.006  # 12 mm command on the 20 mm post (20 N a finger)
SLIDE_DY = 0.040  # m: the effector slides +y 40 mm to its end pose
# The nested hinge's stagger (computed from the built link poses, arm_a at -phi / 2): arm_b's free socket stands
# behind arm_a's along the arms by 0.9 mm at W 84, 1.2 mm at W 88, 1.6 mm at the W 92 hold, 2.1 mm at W 98 (N4),
# inside the 2.5 mm catch window. GPU C6: while the right gripper still holds the effector square, W reads about 84.9
# and fit_mm[0] about [3.0, 0.0, 3.9, 0.9]; once it opens, the sockets' pull should turn the effector about 0.9 deg
# (asin(0.889 / 56): the balls are 56 mm apart; effector_yaw_deg about -0.9), its origin moving about 0.45 mm toward
# -x and -y, and W should go to 84.0. Held square, on needs the slide within -1.3 / +0.3 mm of nominal (kit distance
# on the built shapes); midway between the two axes is 0.44 mm short. When C5 sets TRIM_RIGHT, aim y 0.4 mm short
# (TRIM_RIGHT = (dx, dy_from_c5 - 0.0004, 0.0)): the held band becomes about +-0.8 mm.
CLOSE_S = 2.5  # s for the left to close from the hold to 8 mm
TRIM_LEFT = (-0.005, 0.0, 0.0)  # m, demo only: C4 centre line x 300.7, 302.6, 304.1 at W 81, 85, 89
TRIM_RIGHT = (0.002, 0.0008, 0.0)  # m, demo only: C5 effector 1.9 mm short in x, 1.2 mm in y; aim 0.4 short
LEFT_AWAY = (0.300, 0.200, 0.120)
RIGHT_AWAY = (0.300, -0.200, 0.120)

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
ALUMINIUM = (0.72, 0.73, 0.75)
STEEL = (0.55, 0.56, 0.58)
BLACK = (0.08, 0.08, 0.09)
NYLON = (0.93, 0.62, 0.25)
ACETAL = (0.92, 0.92, 0.90)
BLUE = (0.15, 0.35, 0.80)

# -- fallback text (V10) ---------------------------------------------------------------------------------------------
# The fidelity as shipped in task.json (471 characters):
#   Simplified: the far sockets are hinges (the keeper pins' job), one arm hinged on the other, so the pair swings
#   as one rigid piece just above the bar; a real pair stays a parallelogram and lies on it. One spring gives both
#   springs' pull at the sockets. Sockets are square holes, balls three crossed 10 mm discs, tubes solid bars. A
#   printer's pair hangs from its slider, held on only by its springs. FLSUN SuperRacer's: layout, method, 315 mm
#   arms; the rest is the build's.
# The fidelity to ship only if the no-preload fallback of spec 3.5 ships (alt.fidelity_no_preload, 474 characters):
#   Simplified: the far sockets are hinges (the keeper pins' job), one arm hinged on the other, so the pair swings
#   as one rigid piece just above the bar; a real pair stays a parallelogram and lies on it. One spring, 0.2 N/mm at
#   the sockets, no 1.2 N preload. Sockets are square holes, balls three crossed 10 mm discs, tubes solid bars. A
#   printer's pair hangs from its slider, held on only by its springs. FLSUN SuperRacer's: layout, method, 315 mm
#   arms; the rest is the build's.


# == geometry helpers ===============================================================================================
def m(v) -> np.ndarray:
    """mm to m."""
    return np.asarray(v, dtype=float) * MM


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pivot(arm: str) -> np.ndarray:
    """The arm's hinge axis point (its far ball's centre), slider base frame, mm."""
    return np.array([-STUD_SPACING / 2 if arm == ARM_A else STUD_SPACING / 2, 0.0, BALL_Z])


def arm_sign(arm: str) -> float:
    """-1 for arm_a (outboard is -x), +1 for arm_b."""
    return -1.0 if arm == ARM_A else 1.0


def arm_rest_turn(arm: str) -> float:
    """The drawn turn about the pivot (rad, about +z): each free end 5 mm toward the centre line."""
    return -arm_sign(arm) * ARM_TURN


def arm_point(arm: str, u: float, v: float, w: float) -> np.ndarray:
    """Arm coordinates (mm) to the slider's base frame at rest (mm)."""
    p = np.array([arm_sign(arm) * u, -v, w])
    return pivot(arm) + rz(arm_rest_turn(arm)) @ p


def arm_dir(arm: str, du: float, dv: float, dw: float = 0.0) -> np.ndarray:
    """An arm-coordinate direction in the base frame at rest (unit length if the input is)."""
    return rz(arm_rest_turn(arm)) @ np.array([arm_sign(arm) * du, -dv, dw])


def mouth_rest(arm: str) -> np.ndarray:
    """The free socket's mouth centre, base frame at rest (mm)."""
    return arm_point(arm, MOUTH_U, FREE_V, 0.0)


def inboard_rest(arm: str) -> np.ndarray:
    """The free socket's inboard normal (minus u), base frame at rest."""
    return arm_dir(arm, -1.0, 0.0)


def socket_distance(phi: float) -> float:
    """Free socket mouth centres' distance (mm) with arm_b turned ``phi`` rad against arm_a (small-angle, spec 3.2)."""
    return SOCKET_REST_GAP + ARM_LEN * phi


def width_of(phi: float) -> float:
    return socket_distance(phi) + WIDTH_EXTRA


def phi_for_width(w_mm: float) -> float:
    return (w_mm - WIDTH_EXTRA - SOCKET_REST_GAP) / ARM_LEN


def inner_gap_at(v: float, phi: float) -> float:
    """The tubes' inner faces' distance (mm) at ``v`` mm from the pivots, the pair centred (spec 3.2)."""
    return STUD_SPACING + 2 * TUBE_INNER_U - 2 * v * math.tan(ARM_TURN) + v * phi


def sim_pull_at_sockets(phi: float) -> float:
    """The simulated hinge spring's pull at the free sockets (N): K (phi - target) / 315 mm."""
    return ARM_B_SPRING * (phi - ARM_B_TARGET) / (ARM_LEN * MM)


def spring_rate() -> float:
    """N/mm: G d^4 / (8 D^3 n), D the mean coil diameter."""
    d_mean = SPRING_OD - WIRE_D
    return SHEAR_G * WIRE_D ** 4 / (8.0 * d_mean ** 3 * SPRING_COILS)


def real_pull_at_sockets(d_mm: float) -> tuple[float, float, float]:
    """The real pair (two springs) with the free sockets ``d_mm`` apart: (free-end spring N, far-end spring N, pull
    at the sockets N), spec 3.7."""
    k = spring_rate()
    open_mm = max(0.0, d_mm - SOCKET_REST_GAP)
    s1 = open_mm * STRUT_V[1] / ARM_LEN
    s2 = FAR_SPRING_REST_STRETCH + open_mm * STRUT_V[0] / ARM_LEN
    f1 = SPRING_INIT_TENSION + k * s1 if open_mm > 0 else SPRING_INIT_TENSION
    f2 = SPRING_INIT_TENSION + k * s2
    return f1, f2, (f1 * STRUT_V[1] + f2 * STRUT_V[0]) / ARM_LEN


def finger_give(force_n: float) -> float:
    """mm a position-controlled finger gives under ``force_n``."""
    return force_n / FINGER_STIFFNESS / MM


def finger_force_at_station(phi: float) -> float:
    """N on each spreading finger at the station (simulated spring), spec 2."""
    return sim_pull_at_sockets(phi) * ARM_LEN / STATION_V


def width_for_pad_gap(gap_mm: float, v: float = STATION_V) -> float:
    """Width (mm) the left hand holds at a pad gap (mm), the fingers giving under the spring (fixed point). Never
    under the rest width: the fingers then do not touch both tubes."""
    phi = 0.0
    for _ in range(30):
        s = gap_mm + 2 * FINGER_THICK - 2 * finger_give(finger_force_at_station(phi))
        phi = max(0.0, (s - inner_gap_at(v, 0.0)) / v)
    return width_of(phi)


def pad_gap_for_width(w_mm: float, v: float = STATION_V) -> float:
    """The pad gap (mm) that holds width ``w_mm`` at the station."""
    phi = phi_for_width(w_mm)
    return inner_gap_at(v, phi) - 2 * FINGER_THICK + 2 * finger_give(finger_force_at_station(phi))


def pin_stop_angle() -> float:
    """arm_a's swing (rad) at which a tube's outer face meets its rest-bar pin (spec 3.4)."""
    face = STUD_SPACING / 2 + TUBE_OUTER_U - REST_BAR_V * math.tan(ARM_TURN)  # 39.2 mm
    pin_face = STOP_PIN_X - STOP_PIN_R  # 48.0 mm
    return (pin_face - face) / REST_BAR_V


# -- stud pairs and the effector -------------------------------------------------------------------------------------
def stud_pair_frames() -> list[tuple[np.ndarray, float]]:
    """The effector's three stud pairs: (bar centre in the effector frame (mm), the bar's long-axis angle (rad))."""
    out = []
    for nd in STUD_NORMALS_DEG:
        n = math.radians(nd)
        out.append((np.array([STUD_PAIR_OUT * math.cos(n), STUD_PAIR_OUT * math.sin(n), BALL_Z]), n - math.pi / 2))
    return out


def effector_balls_local() -> list[tuple[np.ndarray, np.ndarray]]:
    """The six ball centres, in pairs, effector frame (mm). First pair: the front one, facing the arms."""
    out = []
    for c, a in stud_pair_frames():
        d = np.array([math.cos(a), math.sin(a), 0.0]) * STUD_SPACING / 2
        out.append((c - d, c + d))
    return out


def plate_corners_local() -> np.ndarray:
    """The six bottom corners of the hexagonal plate, effector frame (mm)."""
    return np.array([[HEX_CORNER_R * math.cos(math.radians(a)), HEX_CORNER_R * math.sin(math.radians(a)), 0.0]
                     for a in range(0, 360, 60)])


def slider_balls_local() -> list[np.ndarray]:
    return [np.array([-STUD_SPACING / 2, 0.0, BALL_Z]), np.array([STUD_SPACING / 2, 0.0, BALL_Z])]


# == shape tables (plain tuples; task.py turns them into mechkit shapes) ===============================================
# ("box", centre mm, size mm, yaw rad, colour) or ("cyl", centre mm, radius mm, length mm, axis, yaw rad, colour)
def stud_pair_shapes(centre, angle: float) -> list[tuple]:
    """A stud pair (bar, two hexagons, two necks, two balls of three crossed discs) whose bar is centred at
    ``centre`` (mm) with its long axis at ``angle`` (rad about z)."""
    c = np.asarray(centre, dtype=float)
    a = np.array([math.cos(angle), math.sin(angle), 0.0])
    out = [("box", tuple(c), STUD_BAR, angle, ALUMINIUM)]
    for sgn in (-1.0, 1.0):
        out.append(("cyl", tuple(c + sgn * STUD_HEX[2] * a), STUD_HEX[0], STUD_HEX[1], "X", angle, STEEL))
        out.append(("cyl", tuple(c + sgn * STUD_NECK[2] * a), STUD_NECK[0], STUD_NECK[1], "X", angle, STEEL))
        b = tuple(c + sgn * STUD_SPACING / 2 * a)
        out += ball_shapes(b, angle)
    return out


def ball_shapes(centre, angle: float) -> list[tuple]:
    """V3: one ball as three crossed discs of radius 5 (along a 4 mm, along n 2 mm, along z 2 mm)."""
    return [("cyl", tuple(centre), BALL_R, BALL_CYL_LEN_A, "X", angle, STEEL),
            ("cyl", tuple(centre), BALL_R, BALL_CYL_LEN_N, "X", angle + math.pi / 2, STEEL),
            ("cyl", tuple(centre), BALL_R, BALL_CYL_LEN_Z, "Z", 0.0, STEEL)]


def slider_base_shapes() -> list[tuple]:
    size, centre = SLIDER_PLATE
    out = [("box", centre, size, 0.0, ALUMINIUM)]
    out += stud_pair_shapes((0.0, 0.0, BALL_Z), 0.0)
    for sgn in (-1.0, 1.0):
        out.append(("cyl", (sgn * KEEPER_PIN_X, 0.0, KEEPER_PIN_LEN / 2), KEEPER_PIN_R, KEEPER_PIN_LEN, "Z", 0.0,
                    STEEL))
    return out


def _socket_block(arm: str, v: float) -> list[tuple]:
    """Four boxes round the 8 x 8 mm hole, the hole's axis along u."""
    yaw = arm_rest_turn(arm)
    wall = (BLOCK_SIZE - SOCKET_HOLE) / 2  # 3.0
    mid = SOCKET_HOLE / 2 + wall / 2  # 5.5
    out = []
    for dv, dw, sv, sw in ((0.0, mid, BLOCK_SIZE, wall), (0.0, -mid, BLOCK_SIZE, wall),
                           (mid, 0.0, wall, SOCKET_HOLE), (-mid, 0.0, wall, SOCKET_HOLE)):
        c = arm_point(arm, BLOCK_U, v + dv, dw)
        out.append(("box", tuple(c), (BLOCK_PROUD, sv, sw), yaw, NYLON))
    return out


def arm_shapes(arm: str) -> list[tuple]:
    """One arm link: tube, two socket blocks, two drawn spring struts, and (arm_a) the stop screw."""
    yaw = arm_rest_turn(arm)
    out = [("box", tuple(arm_point(arm, *TUBE_CENTRE)), (TUBE[0], TUBE[1], TUBE[2]), yaw, BLACK)]
    out += _socket_block(arm, 0.0)
    out += _socket_block(arm, FREE_V)
    for v in STRUT_V:
        u = TUBE_INNER_U - STRUT_LEN / 2
        out.append(("cyl", tuple(arm_point(arm, u, v, 0.0)), STRUT_R[arm], STRUT_LEN, "X", yaw, STEEL))
    if arm == ARM_A:
        u = TUBE_INNER_U - STOP_SCREW_LEN / 2
        out.append(("cyl", tuple(arm_point(arm, u, STOP_SCREW_V, 0.0)), STOP_SCREW_R, STOP_SCREW_LEN, "X", yaw,
                    ACETAL))
    return out


def rest_bar_shapes() -> list[tuple]:
    size, centre = REST_BAR
    out = [("box", centre, size, 0.0, ACETAL)]
    for sgn in (-1.0, 1.0):
        out.append(("cyl", (sgn * STOP_PIN_X, 0.0, STOP_PIN_Z), STOP_PIN_R, STOP_PIN_LEN, "Z", 0.0, STEEL))
    return out


def effector_shapes() -> list[tuple]:
    out = [("box", (0.0, 0.0, HEX_THICK / 2), HEX_BOX, math.radians(a), BLUE) for a in (0.0, 60.0, -60.0)]
    size, centre = POST
    out.append(("box", centre, size, 0.0, BLUE))
    for c, a in stud_pair_frames():
        out += stud_pair_shapes(c, a)
    return out


# == scorer geometry =================================================================================================
def socket_frames(link_to_world) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two free sockets' (mouth centre, inboard normal) in the world (m), given ``link_to_world(body, point)``
    for base-frame points at rest (mechkit's AssemblyState.link_to_world). The normal is the difference of two
    carried points, so it turns with the link."""
    out = []
    for arm in ARMS_PAIR:
        mo = link_to_world(arm, m(mouth_rest(arm)))
        tip = link_to_world(arm, m(mouth_rest(arm) + 10.0 * inboard_rest(arm)))
        n = np.asarray(tip, dtype=float) - np.asarray(mo, dtype=float)
        out.append((np.asarray(mo, dtype=float), n / np.linalg.norm(n)))
    return out


def ball_in_socket(ball, mouth, normal) -> tuple[float, float]:
    """(e, rho) in mm: e how far the ball's centre is inboard of the mouth plane, rho its distance from the socket's
    axis line (through the mouth centre along the inboard normal), in 3D (V4)."""
    d = (np.asarray(ball, dtype=float) - np.asarray(mouth, dtype=float)) / MM
    e = float(d @ normal)
    rho = float(np.linalg.norm(d - e * np.asarray(normal, dtype=float)))
    return e, rho
