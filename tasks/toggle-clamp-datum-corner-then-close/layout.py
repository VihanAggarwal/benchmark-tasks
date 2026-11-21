"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A steel fixture plate (160 x 320 x 12 mm) stands on the table with two datum stops on it (S1 across the far side, S2
along the right side) and two hold-down toggle clamps (C1 on the near side, C2 on the left side). An aluminium
housing (the casting, 60 x 100 x 30 mm) lies on a staging block to the left of the plate. The job: carry the casting
into the fixture, push it hard against both stops, then close both clamps over it, hands off. A splash guard stands
behind the plate.

Each clamp is one lever (arm + handle) on a hinge, drawn in its OPEN pose: tilted back LEAN (30 degrees) past vertical,
away from the casting, resting on its stop. Closing turns it Q_FLAT (120 degrees) about its pivot: the arm lies
horizontal over the casting with the red pad on its top face and the handle lies flat above the arm. A clamp stays
shut, pressing the casting, once its handle has passed the over-centre point (Q_OC).

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the fixture (fixed): plate, two datum stops, splash guard ---------------------------------------------------------
PLATE_X0, PLATE_X1 = 0.220, 0.380
PLATE_Y0, PLATE_Y1 = -0.160, 0.160
PLATE_H = 0.012  # steel, 12 mm (est)
PLATE_X, PLATE_Y = (PLATE_X0 + PLATE_X1) / 2, (PLATE_Y0 + PLATE_Y1) / 2
STOP_H = 0.012  # both stops stand 12 mm proud of the plate
S1_X0, S1_X1 = 0.330, 0.345  # across the far side: 15 mm deep along x, 90 mm long along y
S1_Y0, S1_Y1 = -0.045, 0.045
S2_X0, S2_X1 = 0.280, 0.320  # along the right side: 40 mm long along x, 10 mm thick
S2_Y0, S2_Y1 = -0.060, -0.050
GUARD_X0, GUARD_X1 = 0.395, 0.403  # splash guard behind the plate: 8 mm steel, 400 wide, 140 tall
GUARD_Y0, GUARD_Y1 = -0.200, 0.200
GUARD_H = 0.140
STEEL = (0.45, 0.47, 0.50)
FIXTURE_FRICTION = (0.30, 0.25)  # aluminium on steel (est): plate and stops

# -- the casting (loose) ---------------------------------------------------------------------------------------------
CS_L = 0.060  # along x
CS_W = 0.100  # along y
CS_H = 0.030
CS_MASS = 0.18  # 6061 block pocketed from below to 5 mm walls and top: 67.5 cm3 at 2.70 g/cm3 (solid: 0.49 kg)
CS_COLOR = (0.74, 0.76, 0.80)
CS_FRICTION = (0.30, 0.25)
SEAT_X, SEAT_Y = 0.300, 0.0  # centre when seated: far face on S1 (x 0.330), right face on S2 (y -0.050)
SEAT_FAR_X = SEAT_X + CS_L / 2  # 0.330
SEAT_RIGHT_Y = SEAT_Y - CS_W / 2  # -0.050
CS_TOP = PLATE_H + CS_H  # 0.042: the top face when seated on the plate

# -- the staging block (fixed) ---------------------------------------------------------------------------------------
ST_X, ST_Y = 0.260, 0.245
ST_L, ST_W, ST_H = 0.100, 0.150, 0.020  # HDPE block, 100 (x) x 150 (y) x 20 mm
ST_COLOR = (0.88, 0.88, 0.86)
START_X, START_Y = ST_X, ST_Y  # the casting lies centred on it, its 100 mm side along y
START_Z = ST_H + CS_H / 2

# -- the clamps (fixed bases, one hinged lever each) -----------------------------------------------------------------
LEAN = math.radians(30.0)  # open: the lever stands 30 degrees back past vertical (as drawn, q = 0)
Q_FLAT = math.radians(120.0)  # q at which the arm lies horizontal (LEAN + 90 degrees)
Q_UPPER = math.radians(135.0)  # the lever's far stop: 15 degrees past flat (a clamp shut on nothing ends here)
Q_OC = math.radians(85.0)  # over-centre: past it (55 degrees from vertical) the toggle drives the lever shut
Q_RELEASE = Q_OC - math.radians(5.0)  # a shut lever forced back below this (by a hard lift) is open again
# The toggle's behaviour (a declared rule, task.py ``toggle_target``) acts through a stiff joint drive (PRESS_K, per
# clamp): short of over-centre the drive's target trails the lever by OPEN_BIAS, a constant pull back toward the open
# stop that the handle must be pushed against (OPEN_FORCE at the grip's centre); past over-centre the target runs on
# at SNAP_RATE to Q_TARGET_SHUT, beyond the lever's far stop, so a lever that lands on the casting at flat presses it
# with PRESS_FORCE through its red pad (the force a spindle set for a light press gives; est).
PRESS_FORCE = 25.0  # N, each pad on the casting's top when shut on it (est)
OPEN_FORCE = 1.5  # N at the grip's centre, the handle's resistance short of over-centre (est)
Q_TARGET_SHUT = math.radians(160.0)  # 40 degrees past flat: the drive's target once past over-centre
SNAP_RATE = 1.5  # rad/s: how fast the target runs on toward Q_TARGET_SHUT past over-centre
DAMPING = 0.05  # N m s / rad
LEVER_MASS = 0.04  # kg
Z_PIVOT = 0.054  # the pivot axis (arm's centre line), above the table: the pad then sits on the casting's top (0.042)
ARM_T = 0.008  # the arm bar's thickness (along t) and its width across (along the pivot axis)
ARM_WIDTH = 0.012
HANDLE_T = 0.008
HANDLE_WIDTH = 0.012
HANDLE_OFFSET = 0.012  # the handle bar's centre line lies this far above the arm's when shut (t)
GRIP_L = 0.045  # the handle's rubber grip: its last 45 mm
GRIP_T = 0.014
GRIP_WIDTH = 0.016
PAD_SIZE = 0.012  # red pad, 12 x 12 mm, 8 mm tall, hanging under the arm's end
PAD_H = 0.008
PAD_T_BOTTOM = -(ARM_T / 2 + PAD_H)  # pad's bottom face, t = -12 mm
PAD_RED = (0.80, 0.12, 0.10)
LEVER_COLOR = (0.15, 0.30, 0.65)
GRIP_COLOR = (0.08, 0.08, 0.08)
BASE_COLOR = (0.30, 0.32, 0.35)
LEVER_FRICTION = (0.30, 0.25)
FACE_T = HANDLE_OFFSET + HANDLE_T / 2  # the handle bar's back face (the side an operator pushes): t = 16 mm
FACE_T_GRIP = HANDLE_OFFSET + GRIP_T / 2  # on the grip: t = 19 mm


class Clamp:
    """One toggle clamp: pivot position (plan), direction toward the casting (plan unit vector), the axis its positive
    turn closes about (so the arm swings toward the casting), arm length to its end, the pad's distance along the arm,
    handle length."""

    def __init__(self, name, px, py, direction, axis, arm, pad_s, handle):
        self.name = name
        self.px, self.py = px, py
        self.dir = np.array(direction, dtype=float)
        self.axis = np.array(axis, dtype=float)
        self.arm = arm
        self.pad_s = pad_s
        self.handle = handle

    @property
    def pivot(self) -> np.ndarray:
        return np.array([self.px, self.py, Z_PIVOT])

    def across(self) -> np.ndarray:
        """The in-plan unit vector across the arm (along the pivot axis)."""
        return np.array([abs(self.axis[0]), abs(self.axis[1]), 0.0])

    def closed_frame(self) -> np.ndarray:
        """Columns: along the arm (toward the casting), across, up, with the lever shut (q = Q_FLAT)."""
        d = np.array([self.dir[0], self.dir[1], 0.0])
        up = np.array([0.0, 0.0, 1.0])
        a = np.cross(up, d)
        return np.column_stack([d, a, up])

    def turn(self, q: float) -> np.ndarray:
        """Rotation taking the lever's SHUT pose to its pose at joint angle ``q`` (positive about ``axis``)."""
        return mk_axis_angle(self.axis, q - Q_FLAT)

    def point(self, s: float, t: float, q: float) -> np.ndarray:
        """World point (z above the table) of the lever point at distance ``s`` along the arm and ``t`` up from its
        centre line (as when shut), with the lever at joint angle ``q``."""
        closed = self.pivot + s * np.array([self.dir[0], self.dir[1], 0.0]) + t * np.array([0.0, 0.0, 1.0])
        return self.pivot + self.turn(q) @ (closed - self.pivot)

    def pad_bottom(self, q: float) -> np.ndarray:
        return self.point(self.pad_s, PAD_T_BOTTOM, q)

    @property
    def grip_r(self) -> float:
        """The grip's centre: its distance from the pivot along the handle."""
        return self.handle - GRIP_L / 2

    @property
    def press_k(self) -> float:
        """The joint drive's stiffness (N m / rad) that gives PRESS_FORCE at the pad with the lever flat."""
        return PRESS_FORCE * self.pad_s / (Q_TARGET_SHUT - Q_FLAT)

    @property
    def open_bias(self) -> float:
        """How far the drive's target trails the lever short of over-centre (rad): OPEN_FORCE at the grip."""
        return OPEN_FORCE * self.grip_r / self.press_k

    def press_force(self, q: float = None) -> float:
        """The pad's force on the casting (N) with the lever held at ``q`` (default flat) by the part."""
        q = Q_FLAT if q is None else q
        return self.press_k * (Q_TARGET_SHUT - q) / self.pad_s

    def pad_xy_range(self) -> tuple[np.ndarray, np.ndarray]:
        """The pad's footprint centre when shut (plan)."""
        p = self.point(self.pad_s, 0.0, Q_FLAT)
        return p[:2], p[:2]


def mk_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


C1 = Clamp("clamp1", px=0.236, py=-0.025, direction=(1.0, 0.0), axis=(0.0, 1.0, 0.0), arm=0.062, pad_s=0.052,
           handle=0.070)  # near side, arm reaching +x over the casting's near edge
C2 = Clamp("clamp2", px=0.300, py=0.125, direction=(0.0, -1.0), axis=(1.0, 0.0, 0.0), arm=0.099, pad_s=0.091,
           handle=0.090)  # left side, arm reaching -y over the casting's left edge
CLAMPS = (C1, C2)
BASE_HALF_ALONG = 0.014  # base post: 28 mm along the arm, 34 mm across, up to 14 mm under the pivot
BASE_HALF_ACROSS = 0.017
FOOT_H = 0.004  # foot plate on the plate, 4 mm

# -- the starting poses of the clamps (as drawn: open, resting on the open stop) ---------------------------------------
Q_OPEN = 0.0

# -- the push geometry: the closed fingers' block (GPU-measured on an earlier task, see README) ---------------------
PAD_FACE = 0.006  # a closed gripper's block: its face this far ahead of the TCP along the closing direction
FIST_HALF = 0.0135  # and this wide (half) across it
TIPS_BELOW = 0.013  # pad tips below the TCP
FIST_H = 0.022  # flat length of the pads above the tips

# -- the placing and pushing numbers -----------------------------------------------------------------------------------
PLACE_GAP_X = 0.012  # the casting is set down 12 mm short of S1 and 12 mm off S2
PLACE_GAP_Y = 0.012
PLACE_X, PLACE_Y = SEAT_X - PLACE_GAP_X, SEAT_Y + PLACE_GAP_Y
PLACE_NEAR_X = PLACE_X - CS_L / 2  # 0.258: its near face
PLACE_LEFT_Y = PLACE_Y + CS_W / 2  # 0.062: its left face

# -- scoring thresholds (scoring.py) ---------------------------------------------------------------------------------
SEAT_TOL = 0.0010  # a stop's face within 1.0 mm of the casting's face
SEAT_TILT_DEG = 1.0
SEAT_FLAT = 0.0015  # lowest point of the casting within 1.5 mm of the plate's top
LATCH_TOL = 0.0015  # a clamp locked on a casting more than 1.5 mm off a stop: latched zero
SHUT_DEG = 8.0  # handle within 8 degrees of flat: q >= Q_FLAT - 8 degrees
Q_SHUT = Q_FLAT - math.radians(SHUT_DEG)
PAD_TOUCH_LOW = -0.0015  # the pad's bottom face between 1.5 mm into and 0.5 mm above the casting's top face
PAD_TOUCH_HIGH = 0.0005  # pressed: a lever resting on the casting is short of its spring target, so it presses
PAD_INSET = 0.004  # the pad's centre this far inside the casting's top face
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 1.0
STILL_HANDLE_DEG = 2.0
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9


# -- the lever's push contact (offline model and demo) -----------------------------------------------------------------
def plan_ab(c: Clamp, xy) -> float:
    """Distance of a plan point from the pivot along the clamp's direction (toward the casting)."""
    return float((np.asarray(xy, dtype=float)[:2] - np.array([c.px, c.py])) @ c.dir)


def face_t(c: Clamp, s: float) -> float:
    return FACE_T_GRIP if s >= c.handle - GRIP_L else FACE_T


def psi_deg(q: float) -> float:
    """The lever's angle from vertical toward the casting (degrees): -30 open, 0 upright, 90 flat."""
    return math.degrees(q - LEAN)


def block_corner(c: Clamp, psi: float, s: float) -> tuple[float, float]:
    """(a, z): along-the-clamp distance from the pivot and height of the point on the handle's back face at distance
    ``s`` along the handle, with the lever ``psi`` radians from vertical toward the casting."""
    t = face_t(c, s)
    a = s * math.sin(psi) - t * math.cos(psi)
    z = Z_PIVOT + s * math.cos(psi) + t * math.sin(psi)
    return a, z


def contact_psi(c: Clamp, a: float, z: float) -> tuple[float | None, float]:
    """The lever angle psi (rad from vertical) at which a block's bottom front corner at (a, z) (a along the clamp
    from the pivot, z height) lies on the handle's back face, and the distance ``s`` along the handle there; psi None
    when the corner is not on the handle's length (no contact)."""
    dz = z - Z_PIVOT
    r = math.hypot(a, dz)
    if r < 1e-6:
        return None, 0.0
    for tf in (FACE_T_GRIP, FACE_T):
        if tf / r >= 1.0:
            continue
        psi = math.atan2(a, dz) + math.asin(tf / r)
        s = a * math.sin(psi) + dz * math.cos(psi)
        if 0.0 <= s <= c.handle + 0.004 and (tf == FACE_T_GRIP) == (s >= c.handle - GRIP_L):
            return psi, s
    return None, 0.0
