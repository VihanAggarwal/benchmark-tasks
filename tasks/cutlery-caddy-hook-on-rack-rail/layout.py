"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top.

A dish rack's side wall stands on the table: a round top rail (10 mm wire) and two lower wires (6 mm), all with their
robot-side tangent plane at x = X_PLANE, carried by two end posts; behind it a draining tray holds two plates on edge.
A slim cutlery caddy (a hanging, slotted holder) stands on the table in front of the wall with three forks standing in
its three slots. Its back wall carries two hooks (an inverted U each: the wall's own tab, a bridge, a rear leg with a
flared tip). Hung, the bridges rest on the top rail, the back wall leans on the lower wires and the caddy's base is
14 mm above the table.

Figures are estimates (est, said in task.json's fidelity): no listing was used, see the README.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the rack's side wall (fixed) -----------------------------------------------------------------------------------
R_X = 0.330  # the top rail's axis
R_Y = -0.110  # the wall's centre line (the rail's middle): its -y end post 70 mm inboard of the right arm's base line
RAIL_R = 0.005  # a 10 mm plastic-coated wire (real rails are 6 to 12 mm: est)
RAIL_Z = 0.135  # its axis
RAIL_TOP = RAIL_Z + RAIL_R  # 0.140
RAIL_L = 0.260  # from y -0.24 to +0.02
X_PLANE = R_X - RAIL_R  # 0.325: the robot-side tangent plane of the rail and of the lower wires
WIRE_R = 0.003  # 6 mm lower wires and end posts
WIRE_X = X_PLANE + WIRE_R
WIRE_ZS = (0.040, 0.085)
POST_DY = 0.125  # the end posts' offset from the wall's centre line
POST_H = RAIL_Z
RACK_GREY = (0.34, 0.36, 0.40)
RACK_FRICTION = (0.35, 0.30)  # PVC-coated wire against PP (est)

# -- the draining tray and the plates (fixed) ----------------------------------------------------------------------
TRAY_X0 = R_X + RAIL_R  # 0.335: starts behind the wall's rail
TRAY_LX = 0.230
TRAY_LY = 0.300
TRAY_H = 0.012
TRAY_GREY = (0.62, 0.68, 0.74)
PLATE_R = 0.090  # two 180 mm side plates on edge in the tray
PLATE_T = 0.008
PLATE_X = 0.460  # their centre line (their near edge at x 0.370)
PLATE_DY = 0.045
PLATE_WHITE = (0.95, 0.95, 0.92)

# -- the caddy (loose) ---------------------------------------------------------------------------------------------
# Frame: origin at the middle of the caddy's outer footprint, at the underside of its base; x toward the back wall.
C_W = 0.076  # along y (the grip: across its two side walls)
C_D = 0.018  # along x: slim, so a fork stands in it 2 mm from each wall
C_H = 0.100  # rim height
C_T = 0.003  # walls
C_BT = 0.003  # base
DIV_T = 0.002  # the two dividers
DIV_H = 0.090
POCKET = round((C_W - 2 * C_T - 2 * DIV_T) / 3, 6)  # 22 mm wide slots
POCKET_Y = (-(POCKET + DIV_T), 0.0, POCKET + DIV_T)  # slot centres: -24, 0, +24 mm
INNER_D = C_D - 2 * C_T  # 12 mm
CADDY_BLUE = (0.30, 0.55, 0.80)
CADDY_DENSITY = 900.0  # polypropylene (est)
CADDY_FRICTION = (0.35, 0.30)  # PP on PP / PVC (est)

# the two hooks on the back wall
HK_Y = 0.024  # each hook's centre line from the caddy's middle
HK_W = 0.012  # each hook's width along y
HK_TAB = 0.026  # the back wall's own tab above the rim: the near leg
U = C_H + HK_TAB  # 0.126: the bridge's underside above the base's underside
HK_BR_T = 0.004  # the bridge
HK_GAP = 0.018  # clear gap between the near leg and the rear leg (the rail is 10 mm)
HK_LEG_T = 0.004  # the rear leg
HK_LEG_H = 0.030  # the rear leg hangs this far below the bridge's underside
HK_FLARE_L = 0.012  # the rear leg's tip is bent outward (away from the rail) for a lead-in
HK_FLARE_T = 0.003
HK_FLARE_DEG = 42.0
NEAR_FACE = C_D / 2  # 0.009: the back wall's outer face = the near leg's inner face
REAR_INNER = NEAR_FACE + HK_GAP  # 0.027
REAR_OUTER = REAR_INNER + HK_LEG_T  # 0.031
FLARE_DX = HK_FLARE_L * math.cos(math.radians(HK_FLARE_DEG))  # reach of the lead-in toward +x (9 mm)
FLARE_DZ = HK_FLARE_L * math.sin(math.radians(HK_FLARE_DEG))  # and its drop (8 mm)

# hung: the near leg's inner face on the wall's tangent plane, the bridges on the rail's top
HUNG_X = X_PLANE - NEAR_FACE  # 0.316
HUNG_Z = RAIL_TOP - U  # 0.014: the base 14 mm above the table
HUNG_Y = R_Y
QX_HUNG = R_X - HUNG_X  # 0.014: the rail's axis in the caddy frame, hung
QX_MIN = NEAR_FACE + RAIL_R  # 0.014: rail axis touching the near leg
QX_MAX = REAR_INNER - RAIL_R  # 0.022: rail axis touching the rear leg
QX_MID = (QX_MIN + QX_MAX) / 2  # 0.018

# start: standing on the table (the demo's x and y)
START_X, START_Y, START_Z = 0.200, -0.210, 0.0

# -- the forks (loose) ----------------------------------------------------------------------------------------------
# Three small forks, 120 mm: tines down, handles up, one in each slot. Frame: origin at the fork's middle.
FORK_L = 0.120
FORK_T = 0.008  # along x (a real fork is 2 to 3 mm thick: solid slabs here, so they stand in a 12 mm slot)
FORK_HEAD_W = 0.016
FORK_MASS = 0.040
FORK_Z0 = C_BT + FORK_L / 2  # 0.063 above the base's underside
FORK_STEEL = (0.76, 0.78, 0.82)
FORK_FRICTION = (0.30, 0.25)  # steel on PP (est)
FORK_TOP_ABOVE_RIM = C_BT + FORK_L - C_H  # 23 mm
FORK_NAMES = ("fork_a", "fork_b", "fork_c")  # slots at y -24, 0, +24 mm

# -- the gripping ----------------------------------------------------------------------------------------------------
GRIP_X = -0.006  # the pads centred 6 mm behind the caddy's middle: 5 mm clear of the rail when hung

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
SEAT_X_TOL = 0.001  # the rail's surface may sit this far inside a leg's face
SEAT_Z_LO = 0.002  # the bridge's underside this far below the rail's top (interpenetration allowance)
SEAT_Z_HI = 0.003  # and at most this far above it
SEAT_TURN_DEG = 30.0  # a hook is seated only with the caddy facing the rack and upright within this
YAW_DEG = 5.0
TILT_DEG = 5.0
END_MARGIN = HK_W  # each hook's centre this far inside the rail's ends
FORK_Z_LO = C_BT - 0.003  # a fork's middle: above the floor, below the rim
FALL_DROP = 0.050
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # visibly motionless: under 5 mm and 5 degrees over the last 1 s
STILL_TURN_DEG = 5.0
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9


def fork_start(i: int) -> tuple:
    """World (x, y, z above the table) of fork ``i``'s middle at the start."""
    return (START_X, START_Y + POCKET_Y[i], START_Z + FORK_Z0)


def hook_shapes_volume() -> float:
    """Volume (m3) of the two hooks' parts (for the caddy's mass)."""
    tab = C_T * HK_W * HK_TAB
    bridge = (C_T + HK_GAP + HK_LEG_T) * HK_W * HK_BR_T
    leg = HK_LEG_T * HK_W * HK_LEG_H
    flare = HK_FLARE_L * HK_W * HK_FLARE_T
    return 2 * (tab + bridge + leg + flare)


def caddy_volume() -> float:
    base = C_D * C_W * C_BT
    walls = 2 * (C_T * C_W * C_H) + 2 * (INNER_D * C_T * C_H)
    divs = 2 * (INNER_D * DIV_T * DIV_H)
    return base + walls + divs + hook_shapes_volume()


CADDY_MASS = round(caddy_volume() * CADDY_DENSITY, 3)  # about 0.06 kg
