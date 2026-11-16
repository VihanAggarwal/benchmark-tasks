"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP unless named *_W.

Scene (sources in README.md; "est" marks a figure not read at a source):
- A grey polycarbonate wall-mount control enclosure lying on its back on the bench (bench wiring and testing),
  fixed: outer 250 (x) x 200 (y) x 120 (z) mm with its cover, walls and floor 3 mm (est, a common catalogue size
  class). Inside: a mounting plate, a 35 mm top-hat DIN rail along x and 16 grey 2.5 mm2 feed-through terminal
  blocks (5.2 mm each) clipped on it, modelled as one block (est sizes).
- Its clear polycarbonate cover (4 mm, 300 g, est) hinged along the box's LEFT long edge, resting open 97 degrees on
  its stop: 7 degrees past vertical, leaning away from the box. Two snap catches on its right edge (the one declared
  rule, catchrule.py).
- A terminal screwdriver (insulated, 3.5 x 100 mm class: 210 mm overall, handle 28 mm across and 110 mm long, 60 g,
  est) left lying along x on top of the terminal row: the handle's top is 25 mm under the shut cover's inner face, so
  a cover shut over it closes and latches.
- A bench test supply (fixed grey box, est) with a 16 A 250 V rocker power switch on its top (1550 series class:
  actuating force 3 to 5 N, sourced), joined to the enclosure's right wall by a rigid grey conduit (est).
- A grey tool tray (est) beyond the supply on the robot's right: a natural place to put the tool, not required.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MM = 0.001
EPS = 1e-9
G = 9.81

# -- the control box (fixed) ----------------------------------------------------------------------------------------
BOX_X = (0.120, 0.370)
BOX_Y = (-0.100, 0.100)
WALL = 0.003
FLOOR = 0.003
RIM = 0.116  # the walls' top: the shut cover's inner face
COVER_T = 0.004
BOX_H = RIM + COVER_T  # 120 mm with the cover shut
BOX_ORIGIN = ((BOX_X[0] + BOX_X[1]) / 2, 0.0)  # the box's base frame: x 0.245, y 0, on the table
IN_X = (BOX_X[0] + WALL, BOX_X[1] - WALL)  # 0.123 .. 0.367
IN_Y = (BOX_Y[0] + WALL, BOX_Y[1] - WALL)  # -0.097 .. 0.097
IN_Z = (FLOOR, RIM)
PLATE_T = 0.003  # mounting plate (est)
PLATE_TOP = FLOOR + PLATE_T  # 6 mm
RAIL = (0.140, 0.350, 0.035, 0.0075)  # x0, x1, width (y), height: TH35-7.5 top-hat rail (EN 60715 size)
RAIL_TOP = PLATE_TOP + RAIL[3]  # 13.5 mm
N_BLOCKS = 16
BLOCK_W = 0.0052  # one 2.5 mm2 feed-through block's width along the rail (est, common catalogue width)
BLOCK_L = 0.045  # across the rail (y) (est)
ROW_TOP = 0.063  # the blocks' tops (est: 49.5 mm over the rail's top)
ROW_CX = 0.270
ROW_X = (ROW_CX - N_BLOCKS * BLOCK_W / 2, ROW_CX + N_BLOCKS * BLOCK_W / 2)  # 0.2284 .. 0.3116
RAIL_Y = -0.025  # the rail runs along x 25 mm right of the box's middle (est layout)
ROW_Y = (RAIL_Y - BLOCK_L / 2, RAIL_Y + BLOCK_L / 2)

# the cover: hinged along the left long edge (the hinge line is the cover's top outer edge, y +0.100, z 0.120)
HINGE_Y, HINGE_Z = BOX_Y[1], BOX_H
COVER_OPEN_DEG = 97.0  # resting on its stop (est; the spec's 95, tuned toward its stated 93 to 98 range)
COVER_MASS = 0.300  # est (a 250 x 200 x 4 mm polycarbonate plate is 240 g, plus hinges and catches)
COVER_DAMP = 0.08  # N m s / rad (est: the cover falls shut in about 1 s without slamming)
CATCH_TAB = (0.030, 0.003, 0.010)  # x, y, z: a catch tab on the cover's right edge (visual; est)
CATCH_XS = (0.170, 0.320)  # the two catches' centres along x
KEEPER = (0.030, 0.004, 0.006)  # the keeper on the right wall's outside under each tab (visual; est)
KEEPER_Z = (0.098, 0.104)

# -- the screwdriver (loose) ----------------------------------------------------------------------------------------
HANDLE_R = 0.014
HANDLE_X = (0.240, 0.350)
SHAFT_R = 0.0025  # 3.5 mm blade in its insulating sleeve (est 5 mm across)
SHAFT_X = (0.140, 0.240)
TOOL_AXIS_Z = ROW_TOP + HANDLE_R  # 77 mm: the handle lies on the terminal row
TOOL_MASS = 0.060
TOOL_ORIGIN = ((HANDLE_X[0] + HANDLE_X[1]) / 2, RAIL_Y, TOOL_AXIS_Z)  # the handle's centre (its base frame)
TOOL_LEN = SHAFT_X[1] - SHAFT_X[0] + HANDLE_X[1] - HANDLE_X[0]  # 210 mm
TOOL_FRICTION = (0.6, 0.5)


def tool_com_x() -> float:
    """The tool's centre of mass along x as PhysX spreads its mass (by volume over the two cylinders)."""
    vh = math.pi * HANDLE_R ** 2 * (HANDLE_X[1] - HANDLE_X[0])
    vs = math.pi * SHAFT_R ** 2 * (SHAFT_X[1] - SHAFT_X[0])
    return (vh * sum(HANDLE_X) / 2 + vs * sum(SHAFT_X) / 2) / (vh + vs)


def tool_axis_points(origin, rot, step: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Points along the tool's axis (world) and the tool's radius at each, from its base-frame origin and rotation."""
    xs = np.arange(SHAFT_X[0], HANDLE_X[1] + 1e-9, step) - TOOL_ORIGIN[0]
    local = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    radii = np.where(xs + TOOL_ORIGIN[0] >= HANDLE_X[0], HANDLE_R, SHAFT_R)
    return np.asarray(origin, dtype=float) + local @ np.asarray(rot, dtype=float).T, radii


# -- the test supply and its rocker switch (fixed) ------------------------------------------------------------------
SUP_XY = (0.220, -0.250)
SUP_SIZE = (0.120, 0.080, 0.060)  # est
SUP_TOP = SUP_SIZE[2]
BEZEL_OUT = (0.034, 0.026)
BEZEL_IN = (0.031, 0.023)
BEZEL_H = 0.0015  # proud of the supply's top (est)
ROCKER = (0.030, 0.022, 0.006)  # the rocker's top (est from the 1550 series' 30 x 22 mm panel size class)
PIVOT_Z = 0.0575  # the rocker's pivot axis (along y) over the table: its bottom at rest
ROCKER_TILT_DEG = 10.0  # either way from level (est)
ROCKER_TRAVEL_DEG = 2 * ROCKER_TILT_DEG  # joint 0 = OFF (the +x ON end raised), 20 deg = ON (ON end down)
ROCKER_CENTRE_DEG = ROCKER_TILT_DEG  # the over-centre point
ROCKER_PRELOAD_DEG = 3.0  # the spring's target lies 3 deg past the stop it holds the rocker on
ROCKER_ON_DEG = ROCKER_CENTRE_DEG + 2.0  # read ON 2 deg past the centre ...
ROCKER_OFF_DEG = ROCKER_CENTRE_DEG - 2.0  # ... and OFF 2 deg before it (hysteresis)
SWITCH_FORCE = 4.0  # N at the ON end's 14 mm press point at the centre: the sheet's 3 to 5 N
PRESS_U = 0.014  # the press point from the pivot along +x
ROCKER_K = SWITCH_FORCE * PRESS_U / math.radians(ROCKER_TILT_DEG + ROCKER_PRELOAD_DEG)  # 0.247 N m / rad
ROCKER_MASS = 0.004
ROCKER_DAMP = 0.0005
CONDUIT_R = 0.010  # 20 mm rigid conduit (est)
CONDUIT_Z = 0.030
CONDUIT_X = SUP_XY[0]
CONDUIT_Y = (BOX_Y[0], SUP_XY[1] + SUP_SIZE[1] / 2)  # -0.100 .. -0.210


def rocker_top_z(u: float, joint_deg: float) -> float:
    """Height of the rocker's top surface at ``u`` (m along +x from the pivot) with the joint at ``joint_deg``."""
    a = math.radians(joint_deg - ROCKER_TILT_DEG)  # the rocker's tilt from level, + tips the +x end down
    return PIVOT_Z + ROCKER[2] * math.cos(a) - u * math.sin(a)


# -- the tool tray (fixed) ------------------------------------------------------------------------------------------
TRAY_XY = (0.250, -0.420)
TRAY_OUT = (0.250, 0.080, 0.020)
TRAY_WALL = 0.005
TRAY_FLOOR = 0.003
TRAY_IN_X = (TRAY_XY[0] - TRAY_OUT[0] / 2 + TRAY_WALL, TRAY_XY[0] + TRAY_OUT[0] / 2 - TRAY_WALL)
TRAY_IN_Y = (TRAY_XY[1] - TRAY_OUT[1] / 2 + TRAY_WALL, TRAY_XY[1] + TRAY_OUT[1] / 2 - TRAY_WALL)

# -- the catch rule ---------------------------------------------------------------------------------------------------
SHUT_DEG = 2.0  # the cover within 2 deg of shut
CATCH_ZONE_Y = (BOX_Y[0], BOX_Y[0] + 0.060)  # a press within 6 cm of the catch edge
CATCH_ZONE_X = (BOX_X[0] + 0.010, BOX_X[1] - 0.010)
PRESS_ABOVE = 0.003  # the pad tips no more than 3 mm over the cover's top: resting on it
PRESS_STEPS = 5  # for 0.25 s
CATCH_FORCE = 5.0  # N (est, a light enclosure snap catch): what the demo's press is commanded past
UNLATCH_DEG = 5.0  # the catches are undone once the cover is 5 deg open

# -- scoring thresholds ---------------------------------------------------------------------------------------------
OPEN_LIVE_DEG = 5.0  # the cover opened 5 deg or more while the supply is ON: latched 0
GATE_TOP = 0.140  # at switch-on, no finger inside the box's outline below 14 cm
CLEAR_M = 0.050  # hands at least 5 cm from the box at the end
WINDOW_S = 1.0
WIN_STEPS = int(round(WINDOW_S * CONTROL_HZ))
STILL_M = 0.002
OFF_TABLE_DROP = 0.010
MAX_REST_HEIGHT = 0.060  # no part of the tool over 6 cm above the table top: lying on it or in the tray
RELEASE_M = 0.030  # every finger zone at least 3 cm from the tool: let go
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)
RESET_JUMP_M = 0.05

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
BOX_GREY = (0.62, 0.64, 0.66)
PLATE_ORANGE = (0.85, 0.55, 0.20)
RAIL_ZINC = (0.78, 0.79, 0.80)
BLOCK_GREY = (0.55, 0.57, 0.58)
COVER_CLEAR = (0.80, 0.88, 0.93)
CATCH_DARK = (0.25, 0.26, 0.28)
HANDLE_RED = (0.85, 0.12, 0.10)
SLEEVE_YELLOW = (0.95, 0.80, 0.10)
SUPPLY_GREY = (0.42, 0.44, 0.47)
BEZEL_BLACK = (0.10, 0.10, 0.11)
ROCKER_RED = (0.80, 0.10, 0.08)
MARK_WHITE = (0.96, 0.96, 0.96)
CONDUIT_GREY = (0.55, 0.56, 0.58)
TRAY_GREY = (0.35, 0.37, 0.40)


# -- the cover's geometry ---------------------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def cover_point(p_shut, open_deg: float) -> np.ndarray:
    """Where a point of the cover (given with the cover shut; world x, y and height over the table) is with the cover
    ``open_deg`` open. Opening turns the cover about -x through the hinge line (its free edge rises toward +y)."""
    h = np.array([0.0, HINGE_Y, HINGE_Z])
    p = np.atleast_2d(np.asarray(p_shut, dtype=float)) - h
    p = p @ rot_x(-math.radians(open_deg)).T + h
    return p if np.ndim(p_shut) > 1 else p[0]


def cover_shut_shapes() -> list:
    """(centre, size) of the cover's shapes with the cover shut (world x, y, height over the table)."""
    cx = (BOX_X[0] + BOX_X[1]) / 2
    out = [((cx, 0.0, RIM + COVER_T / 2), (BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0], COVER_T))]
    for x in CATCH_XS:
        out.append(((x, BOX_Y[0] - CATCH_TAB[1] / 2, BOX_H - CATCH_TAB[2] / 2), CATCH_TAB))
    return out


def cover_plate_points(open_deg: float, step: float = 0.005) -> np.ndarray:
    """Points on the cover plate's two faces with the cover ``open_deg`` open (world x, y, height)."""
    xs = np.arange(BOX_X[0], BOX_X[1] + 1e-9, step)
    ys = np.arange(BOX_Y[0], BOX_Y[1] + 1e-9, step)
    g = np.array([(x, y, z) for x in xs for y in ys for z in (RIM, BOX_H)])
    return cover_point(g, open_deg)


def cover_balance_deg() -> float:
    """The open angle at which the cover's centre of mass stands over its hinge: below it the cover falls shut."""
    (c, _s) = cover_shut_shapes()[0]
    ry, rz = c[1] - HINGE_Y, c[2] - HINGE_Z  # (-0.100, -0.002)
    # y(open) = ry cos a + rz sin a = 0 (rot_x(-a))
    return math.degrees(math.atan2(-ry, rz)) % 360.0 if rz else 90.0


def tool_inside(points, radii) -> bool:
    """Any part of the tool inside the box's inner volume (to the rim)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    lo = np.array([IN_X[0], IN_Y[0], IN_Z[0]])
    hi = np.array([IN_X[1], IN_Y[1], IN_Z[1]])
    q = np.maximum(lo - p, p - hi)
    d = np.linalg.norm(np.maximum(q, 0.0), axis=1)  # 0 for a point inside
    return bool(np.any(d < np.asarray(radii, dtype=float) - 1e-9))


def gate_box() -> tuple:
    """(centre, size) world of the keep-out at switch-on: the box's outline from the table to 14 cm."""
    return (((BOX_X[0] + BOX_X[1]) / 2, (BOX_Y[0] + BOX_Y[1]) / 2, TOP + GATE_TOP / 2),
            (BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0], GATE_TOP))
