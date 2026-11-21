"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_deg, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475 and stands for the floor of a corridor. Heights below are above
the table top.

The corridor runs along y; its side wall stands beyond the work area (x 600 to 640 mm). A common 24 in two-leaf A-frame
wet-floor sign stands folded at the middle of the table, its ridge (the hinge, 610 mm up) along x, so its faces look
along the corridor: leaf A faces -y (the right arm's side), leaf B faces +y (the left arm's side). Each leaf has a
moulded hand grip on its outer face. The top hinge opens to a moulded stop at 25 degrees between the leaves (12.5
degrees each, feet 264 mm apart). On the +y side lies a flat blue spill mat.

The sign is one articulation. The base is leaf A with its foot pad, grip and printed panel. Leaf B hangs on the hinge
at the top (axis +x: a positive angle swings its foot toward +y), stopped at 25 degrees.

Sign coordinates (as folded, in the base frame): x along the ridge, y the leaf normal (leaf A's outer face is -y),
z up from the foot's underside. Functions below return (n, z): n along y from the hinge line, z above the table.

Figures are estimates (est, said in task.json's fidelity) unless the README gives a source.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the sign (one articulation; base frame at the hinge's foot on the table, every joint 0 = folded) -------------
SX, SY = 0.330, 0.010  # the folded sign's hinge line: its middle at x 330 mm, y +10 mm (the ridge along x)
H = 0.610  # leaf height from the foot's underside to the hinge axis (a 24 in sign)
W = 0.280  # leaf width (along the ridge, x)
T = 0.012  # leaf panel thickness (moulded polypropylene, est)
PAD_D, PAD_W, PAD_H = 0.030, 0.260, 0.008  # a rubber foot pad on each leaf (est): 30 deep, 260 wide, 8 tall
RIB_P = 0.140  # the moulded grab handle (a solid fin) stands 140 mm proud of the leaf's outer face, so a hand
# closed on it keeps the wrist clear of the leaf
RIB_W = 0.020  # 20 mm across (along x): the grip width
RIB_Z0, RIB_Z1 = 0.030, 0.120  # its height range above the foot's underside
U_GRIP = 0.120  # the pads close on the grip's outer part: 120 mm out from the leaf's inner face (pads 27 mm wide)
Z_GRIP = 0.075  # and 75 mm up (TCP height: the pad tips sit 13 mm lower)
PANEL_W, PANEL_H, PANEL_Z = 0.200, 0.150, 0.440  # the printed warning (pictogram and CAUTION WET FLOOR), its middle
PANEL_T = 0.001

# the hinge's moulded stop
THETA_LOCK_DEG = 12.5  # each leaf off vertical at the stop
THETA_LOCK = math.radians(THETA_LOCK_DEG)
Q_STOP = 2.0 * THETA_LOCK  # 25 degrees between the leaves
Q_STOP_DEG = math.degrees(Q_STOP)


def spread(q: float) -> float:
    """The distance between the feet (m) with the leaves ``q`` rad apart and both feet on the table."""
    return 2.0 * H * math.sin(q / 2.0)


SPREAD_LOCK = spread(Q_STOP)  # 264.1 mm between the feet at the stop

# masses (0.55 kg in all, est: a light 24 in sign)
MASS_A = 0.275
MASS_B = 0.275
MASS_TOTAL = MASS_A + MASS_B
FRICTION = (0.6, 0.5)  # the rubber pads on a smooth floor (est); the table's own friction averages in
HINGE_FRICTION = 0.02  # N m (est)
HINGE_DAMPING = 0.02  # N m s/rad

# colours
YELLOW = (0.96, 0.80, 0.08)
RIB_YELLOW = (0.90, 0.62, 0.05)
PAD_GREY = (0.15, 0.15, 0.16)
PANEL_BLACK = (0.05, 0.05, 0.05)
PATCH_BLUE = (0.15, 0.33, 0.78)
WALL_BEIGE = (0.84, 0.80, 0.70)

# -- the spill patch and the wall (fixed) ------------------------------------------------------------------------------
GAP_TARGET = 0.090  # the open sign's +y foot ends 90 mm from the patch's near edge when opened in place
PATCH_X0, PATCH_X1 = 0.200, 0.500  # 300 mm along the corridor's width (x), its long edge parallel to the ridge
PATCH_Y0 = SY + SPREAD_LOCK / 2.0 + GAP_TARGET  # 222 mm: the near edge (toward the sign)
PATCH_Y1 = PATCH_Y0 + 0.200  # 200 mm deep (along y)
PATCH_T = 0.001
PATCH_XC = (PATCH_X0 + PATCH_X1) / 2.0
WALL_X0, WALL_X1 = 0.600, 0.640
WALL_Y0, WALL_Y1 = -0.600, 0.600
WALL_H = 0.300

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------------
SPREAD_LO, SPREAD_HI = 0.244, 0.270  # feet apart: 244 to 270 mm (the stop gives 264 mm; 244 mm is 1.9 degrees short)
LEAN_LO_DEG, LEAN_HI_DEG = 9.0, 16.0  # each leaf off vertical
GAP_LO, GAP_HI = 0.040, 0.140  # the nearer foot from the patch's near edge (along y)
BESIDE_HALF = (PATCH_X1 - PATCH_X0) / 2.0  # the hinge's middle within the patch's 300 mm length (along x)
YAW_OK_DEG = 10.0  # the ridge within 10 degrees of the patch's long edge
YAW_NEAR_DEG = 30.0
OPENED_SPREAD = 0.070  # a sign counts as opened (0.25) from 70 mm of spread
FOOT_Z = 0.010  # both foot lines within 10 mm of the table top
GATE_DEG = 45.0  # either leaf more than 45 degrees off vertical at the end: lying over, a trip hazard
STILL_WINDOW_S = 2.0
STILL_MOVE = 0.003  # no tracked point moves more than 3 mm over the window
STILL_SPREAD = 0.002  # the spread changes under 2 mm
STILL_TURN_DEG = 1.0
LET_GO = 0.050  # every finger zone at least 50 mm from the sign
CONTROL_HZ = 20.0
RESET_JUMP = 0.04
START_TOL = 0.01
EPS = 1e-9


# -- the sign's kinematics (sign coordinates: n along y from the hinge line, z above the table) ------------------
def rib_point(leaf: str, theta: float, u: float = U_GRIP, z: float = Z_GRIP) -> tuple[float, float]:
    """Where the point ``u`` out from a leaf's inner face and ``z`` up the leaf (as folded) is, relative to the hinge's
    foot on the table (n, z), with the sign opened symmetrically to ``theta`` per leaf and both feet on the table. Leaf
    A swings toward -y, leaf B toward +y."""
    w = z - H  # along the leaf from the hinge (negative: below it)
    s = -1.0 if leaf == "A" else 1.0
    n = s * u * math.cos(theta) - s * w * math.sin(theta)
    zz = u * math.sin(theta) + w * math.cos(theta)
    return n, H * math.cos(theta) + zz


def foot_n(leaf: str, theta: float) -> float:
    """A leaf's foot line (its inner bottom edge) relative to the hinge line's y (m)."""
    return (-1.0 if leaf == "A" else 1.0) * H * math.sin(theta)


# gap from the nearer foot (leaf B's) to the patch's near edge when the sign opens in place
GAP_SYMMETRIC = PATCH_Y0 - (SY + foot_n("B", THETA_LOCK))
