"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A steel fixture plate (160 x 320 x 12 mm) stands on the table with two datum stops on it (S1 across the far side, S2
along the right side) and two hold-down toggle clamps (C1 on the near side, C2 on the left side). An aluminium
housing (the casting, 60 x 100 x 30 mm) lies on a staging block to the left of the plate. The job: carry the casting
into the fixture, push it hard against both stops, then close both clamps over it, hands off. A splash guard stands
behind the plate.

Each clamp is a vertical hold-down toggle clamp: a handle and a clamp arm, each on its own pivot on the base, joined
by a connecting link (a four-bar linkage, class Clamp below). Drawn in its OPEN pose: the handle leans 57 degrees back
from vertical (away from the casting) and the arm stands 99 degrees up from flat, both on their stops. Closing, the
robot pushes the handle forward to vertical; the link swings the arm down until its red pad lands on the casting's
top. In the last few degrees the link and the handle's crank line up (dead centre, handle vertical): the spindle is
pressed about 0.7 mm and the pad presses the casting with 25 to 29 N; 1.5 degrees past dead centre the handle rests on
its stop, the pad still presses with 20 N, and that force, passed back through the link, holds the handle there
(over-centre lock).

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the fixture (fixed): plate, two datum stops, splash guard ---------------------------------------------------------
PLATE_X0, PLATE_X1 = 0.200, 0.380
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
ST_X, ST_Y = 0.220, 0.245
ST_L, ST_W, ST_H = 0.100, 0.150, 0.020  # HDPE block, 100 (x) x 150 (y) x 20 mm
ST_COLOR = (0.88, 0.88, 0.86)
START_X, START_Y = ST_X, ST_Y  # the casting lies centred on it, its 100 mm side along y
START_Z = ST_H + CS_H / 2


# -- the clamps (fixed bases; handle, connecting link and clamp arm on pivots) --------------------------------------
# Each clamp works in its own vertical plane: ``a`` along the clamp's direction (toward the casting) from the arm
# pivot's plan position, ``z`` above the table. The arm pivot B is at (0, Z_PIVOT), the handle pivot A behind and below
# it. Joint angles (radians) are 0 as drawn (open) and grow as the clamp closes:
#   handle  0 = 57 degrees back from vertical (open stop), QH_DC = vertical (dead centre), QH_UPPER = 1.5 degrees
#           past it (the over-centre stop)
#   arm     0 = 99 degrees up (open), QA_FLAT = flat (pad on the casting's top), QA_FLAT + delta = where the link puts
#           it at dead centre with nothing under the pad (the spindle's travel lower)
#   link    on the handle at C, turned so that its far end stays on the arm's tail D (a function of the handle angle)
HANDLE_OPEN = math.radians(57.0)  # handle opening (DESTACO 207-U catalogue: 57 degrees)
ARM_OPEN = math.radians(99.0)  # clamp bar opening (207-U catalogue: 99 degrees)
OVER_CENTRE = math.radians(1.5)  # the handle's stop past dead centre
QH_DC = HANDLE_OPEN
QH_UPPER = HANDLE_OPEN + OVER_CENTRE
QH_SHUT = QH_DC + math.radians(0.5)  # over centre: 0.5 degree or more past dead centre (the stop is at 1.5)
QA_FLAT = ARM_OPEN
Z_PIVOT = 0.054  # the arm pivot B (arm's centre line) above the table: the pad then sits on the casting's top (0.042)
PIVOT_A = (-0.032, 0.026)  # the handle pivot A: 32 mm behind B, 26 mm above the table
TAIL_D = (0.0015, 0.0064)  # the arm's tail pin D, from B, with the arm at QA_FLAT + delta (dead centre, no part)
PRESS_FORCE = 20.0  # N, each pad on the casting's top with the handle on its over-centre stop (est: a light press)
SPINDLE_K = 40000.0  # N/m at the pad: the spindle's rubber cap, 0.5 mm under 20 N (est)
HANDLE_K_FRACTION = 0.6  # the handle joint's spring over the linkage's stiffness at touch (Clamp.handle_k)
HANDLE_LEAD = 0.5  # the handle spring's anchor leads by this many control steps of its speed (no drag when moving)
HANDLE_LEAD_MAX = 0.05  # rad: at most
HANDLE_TARGET_MAX = 0.5  # rad: the handle's target at most this far from the handle (the rule's torque is bounded)
HANDLE_DAMPING = 0.02  # N m s / rad (est)
LINK_K = 0.5  # N m / rad: the link's joint spring that keeps it on the arm's tail (rule)
LINK_DAMPING = 0.004
HANDLE_MASS = 0.06  # kg (est)
ARM_MASS = 0.05
LINK_MASS = 0.01
ARM_T = 0.008  # the arm bar's thickness (in the clamp's plane) and its width across (along the pivot axis)
ARM_WIDTH = 0.012
HANDLE_T = 0.008
HANDLE_WIDTH = 0.012
GRIP_L = 0.045  # the handle's rubber grip: its last 45 mm, 14 mm thick, 24 mm wide across
GRIP_T = 0.014
GRIP_WIDTH = 0.024
LINK_T = 0.006  # the connecting link: a 6 mm bar 4 mm thick, beside the arm and the handle
LINK_W = 0.004
LINK_OFFSET = 0.011  # its centre plane this far across from the clamp's centre plane
PAD_SIZE = 0.012  # red pad, 12 x 12 mm, 8 mm tall, hanging under the arm's end
PAD_H = 0.008
PAD_T_BOTTOM = -(ARM_T / 2 + PAD_H)  # pad's bottom face, 12 mm under the arm's centre line
PAD_RED = (0.80, 0.12, 0.10)
LEVER_COLOR = (0.15, 0.30, 0.65)
LINK_COLOR = (0.55, 0.57, 0.60)
GRIP_COLOR = (0.08, 0.08, 0.08)
BASE_COLOR = (0.30, 0.32, 0.35)
LEVER_FRICTION = (0.30, 0.25)


def _rot2(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _circles(p0, r0, p1, r1):
    """The intersection of two circles (plane), the branch on the right of p0 -> p1 (the linkage's assembly)."""
    d = p1 - p0
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-12 or n > r0 + r1 or n < abs(r0 - r1):
        return None
    a = (r0 * r0 - r1 * r1 + n * n) / (2 * n)
    h = math.sqrt(max(r0 * r0 - a * a, 0.0))
    m = p0 + a * d / n
    perp = np.array([-d[1], d[0]]) / n
    return m - h * perp


class Clamp:
    """One vertical hold-down toggle clamp: arm pivot position (plan), direction toward the casting (plan unit
    vector), the axis its positive turn closes about, arm length, the pad's distance along the arm, handle length.

    The linkage (four-bar: base, handle crank A-C, link C-D, arm B-D): at dead centre (handle vertical) A, C and D lie
    on one line, so D is as far from A as it can get and the arm is at its lowest; the crank length is solved so that
    the handle at its open stop puts the arm at its open stop."""

    def __init__(self, name, px, py, direction, axis, arm, pad_s, handle):
        self.name = name
        self.px, self.py = px, py
        self.dir = np.array(direction, dtype=float)
        self.axis = np.array(axis, dtype=float)
        self.arm = arm
        self.pad_s = pad_s
        self.handle = handle
        self.delta = PRESS_FORCE / SPINDLE_K / pad_s  # first guess; set below so the stop presses PRESS_FORCE
        self.k_arm = SPINDLE_K * pad_s * pad_s  # N m / rad: the arm's joint spring (the spindle's stiffness)
        self.arm_damping = 2.0 * math.sqrt(self.k_arm * ARM_MASS * arm * arm / 3.0)  # about critical
        self.A = np.array(PIVOT_A, dtype=float)
        self.B = np.array([0.0, Z_PIVOT])
        self.Dc = self.B + np.array(TAIL_D)
        v = self.Dc - self.A
        self.span = float(np.hypot(v[0], v[1]))  # crank + link
        self.beta = math.atan2(v[0], v[1])  # the crank's angle from the handle's line (toward the casting)
        self.rd = float(np.hypot(*TAIL_D))
        self.dc_angle = math.atan2(TAIL_D[1], TAIL_D[0])
        for k in range(4):  # the spindle's travel at dead centre includes the arm's rise to the over-centre stop
            want = ARM_OPEN + self.delta
            f = self._open_phi
            grid = [0.006 + 0.0005 * k for k in range(int((self.span - 0.012) / 0.0005))]
            lo = hi = None
            for r0, r1 in zip(grid, grid[1:]):  # the first bracket (the open angle grows with the crank)
                v0, v1 = f(r0), f(r1)
                if v0 is not None and v1 is not None and v0 <= want <= v1:
                    lo, hi = r0, r1
                    break
            if lo is None:
                raise ValueError(f"{name}: no crank length gives the arm's open angle")
            for _ in range(60):  # bisection on the crank length
                mid = (lo + hi) / 2
                v = f(mid)
                if v is not None and v < want:
                    lo = mid
                else:
                    hi = mid
            self.crank = (lo + hi) / 2
            if k < 3:
                rise = ARM_OPEN + self.delta - self._arm_target_for(self.crank, QH_UPPER)
                self.delta = PRESS_FORCE / SPINDLE_K / pad_s + rise
        self.link = self.span - self.crank
        self.link_open = self.link_angle(0.0)

    def _phi_for(self, crank, psi):
        """Arm turn up from its dead-centre position (rad, + = opening) with the handle at ``psi`` from vertical."""
        c = self.A + crank * np.array([math.sin(psi + self.beta), math.cos(psi + self.beta)])
        d = _circles(self.B, self.rd, c, self.span - crank)
        if d is None:
            return None
        ang = math.atan2(d[1] - self.B[1], d[0] - self.B[0]) - self.dc_angle
        return (ang + math.pi) % (2 * math.pi) - math.pi

    def _arm_target_for(self, crank, qh):
        p = self._phi_for(crank, self.psi(qh))
        return ARM_OPEN + self.delta - (0.0 if p is None else p)

    def _open_phi(self, crank):
        return self._phi_for(crank, -HANDLE_OPEN)

    # -- the linkage as functions of the handle's joint angle qh ------------------------------------------------
    @staticmethod
    def psi(qh: float) -> float:
        """The handle's angle from vertical (rad, + toward the casting): -57 degrees open, 0 at dead centre."""
        return qh - HANDLE_OPEN

    def crank_pin(self, qh: float) -> np.ndarray:
        p = self.psi(qh)
        return self.A + self.crank * np.array([math.sin(p + self.beta), math.cos(p + self.beta)])

    def arm_target(self, qh: float) -> float:
        """The arm's joint angle where the linkage puts it with the handle at ``qh`` and nothing under the pad."""
        p = self._phi_for(self.crank, self.psi(qh))
        if p is None:
            p = 0.0 if qh >= QH_DC else ARM_OPEN + self.delta
        return ARM_OPEN + self.delta - p

    def arm_ratio(self, qh: float, h: float = 1e-4) -> float:
        """d(arm angle) / d(handle angle): the linkage's velocity ratio (0 at dead centre, negative past it)."""
        return (self.arm_target(qh + h) - self.arm_target(qh - h)) / (2 * h)

    def tail_pin(self, qa: float) -> np.ndarray:
        return self.B + _rot2(np.array(TAIL_D), ARM_OPEN + self.delta - qa)

    def link_angle(self, qh: float) -> float:
        """The link's direction C -> D in the plane (rad, counter-clockwise from +a) with the arm on the linkage."""
        c, d = self.crank_pin(qh), self.tail_pin(self.arm_target(qh))
        return math.atan2(d[1] - c[1], d[0] - c[0])

    def link_joint(self, qh: float) -> float:
        """The link's joint angle on the handle (closing positive = clockwise in the plane)."""
        return -((self.link_angle(qh) - self.link_open) + qh)

    def handle_torque(self, qh: float, qa: float, arm_target: float) -> float:
        """The linkage's reaction on the handle (N m, + closing) from the arm's spring load now: the pad's force (or
        anything else holding the arm off its linkage angle) passed back through the link, by virtual work. Short of
        dead centre it pushes the handle open; past dead centre it pushes it onto its over-centre stop."""
        return -self.k_arm * (arm_target - qa) * self.arm_ratio(qh)

    def press_force(self, qh: float) -> float:
        """The pad's force (N) on a casting under it (arm held at flat) with the handle at ``qh``."""
        return max(0.0, self.k_arm * (self.arm_target(qh) - QA_FLAT) / self.pad_s)

    def handle_force(self, qh: float, s: float) -> float:
        """The push (N, + = the robot must push it closed) needed at distance ``s`` along the handle to hold it at
        ``qh`` with the pad on the casting (the linkage's reaction only; the handle's weight is extra)."""
        at = self.arm_target(qh)
        return -self.handle_torque(qh, min(at, QA_FLAT), at) / s

    def weight_torque(self, qh: float) -> float:
        """The weights' torque on the handle (N m, + closing) with nothing under the pad: the handle's own (it leans
        back, so open until vertical) plus the arm's, passed through the linkage (it closes)."""
        theta = QA_FLAT - self.arm_target(qh)  # the arm's angle up from flat
        arm = ARM_MASS * G * (self.arm / 2) * math.cos(theta) * self.arm_ratio(qh)
        return arm + HANDLE_MASS * G * (self.handle / 2) * math.sin(self.psi(qh))

    def balance_qh(self) -> float:
        """The handle angle short of dead centre beyond which the weights close the clamp: let go there, the arm
        settles onto whatever is under the pad; let go nearer its open stop, it falls back open."""
        lo, hi = 0.0, QH_DC - math.radians(5.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.weight_torque(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @property
    def arm_upper(self) -> float:
        """The arm joint's far stop: 1 degree beyond where the link takes it with nothing under the pad."""
        return ARM_OPEN + self.delta + math.radians(1.0)

    @property
    def handle_k(self) -> float:
        """N m / rad: the handle joint's spring, through which the rule applies the linkage's reaction each control
        step: HANDLE_K_FRACTION of the reaction's own stiffness where the pad first touches. Stiff enough that the
        20 Hz rule does not chatter (with the gripper's own give it needs more than half), soft enough that the
        spring's anchor, moved once a step, tugs a quickly swung handle by under 1 N."""
        return HANDLE_K_FRACTION * self.k_arm * self.arm_ratio(self.touch_qh()) ** 2

    def touch_qh(self) -> float:
        """The handle angle at which the pad first touches a seated casting's top."""
        lo, hi = 0.0, QH_DC
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.arm_target(mid) < QA_FLAT:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # -- the clamp's plane and world points -----------------------------------------------------------------------
    @property
    def pivot(self) -> np.ndarray:
        return np.array([self.px, self.py, Z_PIVOT])

    def across(self) -> np.ndarray:
        """The in-plan unit vector across the clamp (along the pivot axis)."""
        return np.array([abs(self.axis[0]), abs(self.axis[1]), 0.0])

    def world(self, a: float, z: float, w: float = 0.0) -> np.ndarray:
        """World point (z above the table) of plane point (a, z), ``w`` across."""
        d = np.array([self.dir[0], self.dir[1], 0.0])
        return np.array([self.px, self.py, 0.0]) + a * d + w * self.across() + np.array([0.0, 0.0, z])

    def plane_frame(self, ang: float) -> np.ndarray:
        """Rotation whose columns are: the in-plane direction at ``ang`` (counter-clockwise from +a toward +z), across,
        and their normal (right-handed)."""
        d = np.array([self.dir[0], self.dir[1], 0.0])
        e1 = math.cos(ang) * d + math.sin(ang) * np.array([0.0, 0.0, 1.0])
        e2 = self.across()
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    def handle_point(self, s: float, t: float, qh: float) -> np.ndarray:
        """Plane point (a, z) at ``s`` along the handle from A and ``t`` off its centre line (+ = its back, away from
        the casting when vertical), with the handle at ``qh``."""
        p = self.psi(qh)
        return self.A + s * np.array([math.sin(p), math.cos(p)]) + t * np.array([-math.cos(p), math.sin(p)])

    def grasp_point(self, s: float, qh: float) -> np.ndarray:
        """World point on the handle's centre line ``s`` from A (z above the table)."""
        a, z = self.handle_point(s, 0.0, qh)
        return self.world(a, z)

    def arm_angle(self, qa: float) -> float:
        """The arm bar's direction in the plane (rad, counter-clockwise from +a): 99 degrees open, 0 flat."""
        return ARM_OPEN - qa

    def arm_point(self, s: float, t: float, qa: float) -> np.ndarray:
        ang = self.arm_angle(qa)
        return self.B + s * np.array([math.cos(ang), math.sin(ang)]) + t * np.array([-math.sin(ang), math.cos(ang)])

    def pad_bottom(self, qa: float) -> np.ndarray:
        """World point of the pad's bottom-face centre with the arm at ``qa``."""
        a, z = self.arm_point(self.pad_s, PAD_T_BOTTOM, qa)
        return self.world(a, z)

    @property
    def grip_r(self) -> float:
        """The grip's centre: its distance from the handle pivot along the handle."""
        return self.handle - GRIP_L / 2


C1 = Clamp("clamp1", px=0.236, py=-0.025, direction=(1.0, 0.0), axis=(0.0, 1.0, 0.0), arm=0.062, pad_s=0.052,
           handle=0.110)  # near side, arm reaching +x over the casting's near edge
C2 = Clamp("clamp2", px=0.300, py=0.125, direction=(0.0, -1.0), axis=(1.0, 0.0, 0.0), arm=0.099, pad_s=0.091,
           handle=0.145)  # left side, arm reaching -y over the casting's left edge
CLAMPS = (C1, C2)
BASE_HALF_ALONG = 0.014  # base post under B: 28 mm along, 34 mm across, up to 14 mm under B
BASE_HALF_ACROSS = 0.017
BASE_REAR = -0.036  # the base reaches back under the handle pivot to here (a)
FOOT_H = 0.004  # foot plate on the plate, 4 mm
CHEEK_T = 0.003  # the base's side cheeks round the handle pivot, 3 mm steel
GRASP_BACK = 0.015  # the demo grasps each handle's grip 15 mm short of its end

# -- the push geometry: the closed fingers' block (GPU-measured on an earlier task, see README) ---------------------
PAD_FACE = 0.006  # a closed gripper's block: its face this far ahead of the TCP along the closing direction
FIST_HALF = 0.0135  # and this wide (half) across it
TIPS_BELOW = 0.013  # pad tips below the TCP
FIST_H = 0.022  # flat length of the pads above the tips

# -- the placing and pushing numbers -----------------------------------------------------------------------------------
PLACE_GAP_X = 0.006  # the casting is set down 6 mm short of S1 and 12 mm off S2
PLACE_GAP_Y = 0.012
PLACE_X, PLACE_Y = SEAT_X - PLACE_GAP_X, SEAT_Y + PLACE_GAP_Y
PLACE_NEAR_X = PLACE_X - CS_L / 2  # 0.264: its near face


# -- the grasp on a handle (offline model and demo) --------------------------------------------------------------------
GRASP_TCP_UP = 0.004  # TCP this far above the handle's centre line at the grasp point (pad tips 9 mm under it)
GRASP_TOL = 0.010  # offline: a closed gripper this close to the grasp point holds the handle
GRASP_OPENING = (0.05, 0.40)  # offline: the opening of a gripper closed on the 24 mm grip (commanded 12 mm: 0.13)

# -- scoring thresholds (scoring.py) ---------------------------------------------------------------------------------
SEAT_TOL = 0.0010  # a stop's face within 1.0 mm of the casting's face
SEAT_TILT_DEG = 1.0
SEAT_FLAT = 0.0015  # lowest point of the casting within 1.5 mm of the plate's top
LATCH_TOL = 0.0015  # a clamp locked on a casting more than 1.5 mm off a stop: latched zero
PAD_TOUCH_LOW = -0.0015  # the pad's bottom face between 1.5 mm into and 0.5 mm above the casting's top face
PAD_TOUCH_HIGH = 0.0005  # pressed: a shut arm resting on the casting is short of its linkage angle, so it presses
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


