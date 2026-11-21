"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP unless named *_W.

Scene (sources in README.md; "est" marks a figure not read at a source):
- A compact slide-pack tool battery charger (DeWalt compact charger class, 145 x 115 x 65 mm, retailer listing),
  fixed: a flat deck between two low side rails, a terminal tongue with two contact strips near its far end and an
  end stop at the far end. Its cord runs to a plug in a switched power strip. The charger has no switch.
- A 10 mm 3/8 in drive chrome socket (17 mm across, 28 mm long, 25 g, est from catalogue ranges) lying across the
  two contact strips, its axis along y.
- A 20 V 2 Ah compact slide pack (113 x 76 x 58 mm, 0.40 kg, est: retail listings give 0.34 to 0.41 kg) lying flat
  on the table, its terminal pocket (underneath, open at its +x end) toward +x.
- A fixed switched power strip (est) along y with a rocker switch at its right end: ON at the start.
- A fixed grey parts tray (est): a natural place for the socket, not required.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
EPS = 1e-9

# -- the charger (fixed) --------------------------------------------------------------------------------------------
CH_X = (0.205, 0.350)  # 145 mm long, along x
CH_Y = (-0.2025, -0.0875)  # 115 mm wide
SEAT_YC = (CH_Y[0] + CH_Y[1]) / 2  # -0.145: the seat's centre line (along x)
CH_ORIGIN = ((CH_X[0] + CH_X[1]) / 2, SEAT_YC)  # the charger's base frame on the table
DECK = 0.052  # the deck the pack slides on (est)
STOP_X = 0.340  # the end stop's face: a seated pack's +x face touches it
STOP_TOP = 0.065  # the end stop block (x 0.340 .. 0.350), the charger's 65 mm overall height
RAIL_IN = 0.040  # the rails' inner faces at y SEAT_YC +- 40 mm: 2 mm play each side of the 76 mm pack
RAIL_TOP = DECK + 0.006  # 6 mm low guide rails (est)
TONGUE_X = (0.290, 0.320)  # the terminal tongue: 30 x 20 mm, 20 mm in front of the end stop (est)
TONGUE_HALF_Y = 0.010
TONGUE_TOP = DECK + 0.010  # 62 mm: the contact strips' tops
CONTACT_T = 0.0005  # the two contact strips (part of the tongue's top, 0.5 mm)
CONTACT_Y = (0.004, 0.007)  # each strip from 4 to 7 mm either side of the centre line
CONTACT_X = (TONGUE_X[0] + 0.002, TONGUE_X[1] - 0.002)

# -- the socket (loose) ---------------------------------------------------------------------------------------------
SOCK_R = 0.0085  # 17 mm across (est, a 10 mm 3/8 in drive socket's outside diameter)
SOCK_L = 0.028  # 28 mm long (est)
SOCK_MASS = 0.025  # est
SOCK_START = ((TONGUE_X[0] + TONGUE_X[1]) / 2, SEAT_YC, TONGUE_TOP + SOCK_R)  # (0.305, -0.145, 0.0705)

# -- the battery pack (loose) ---------------------------------------------------------------------------------------
BAT = (0.113, 0.076, 0.058)  # x, y, z (est, 20 V 2 Ah compact pack class)
BAT_MASS = 0.40  # est (retail listings 0.34 to 0.41 kg)
POCKET = (0.060, 0.030, 0.013)  # the terminal pocket under the pack's +x end: long, wide, tall (est)
TOP_BAND = 0.010  # a yellow band on the pack's top (visual)
BAT_START = (0.180, -0.340)  # the pack's centre on the table (x, y)
BAT_FRICTION = (0.6, 0.5)
SOCK_FRICTION = (0.6, 0.5)


def bat_shapes_local() -> list:
    """(centre, size) of the pack's boxes in its own frame (origin at its bounding box's centre)."""
    hx, hy, hz = BAT[0] / 2, BAT[1] / 2, BAT[2] / 2
    low = POCKET[2]
    upper_h = BAT[2] - low - TOP_BAND
    rear_l = BAT[0] - POCKET[0]
    side_w = (BAT[1] - POCKET[1]) / 2
    out = [((0.0, 0.0, -hz + low + upper_h / 2), (BAT[0], BAT[1], upper_h)),
           ((0.0, 0.0, hz - TOP_BAND / 2), (BAT[0], BAT[1], TOP_BAND)),
           ((-hx + rear_l / 2, 0.0, -hz + low / 2), (rear_l, BAT[1], low))]
    for s in (1.0, -1.0):
        out.append(((hx - POCKET[0] / 2, s * (hy - side_w / 2), -hz + low / 2), (POCKET[0], side_w, low)))
    return out


def bat_com_local() -> np.ndarray:
    """The pack's centre of mass in its frame as PhysX spreads its mass (by volume)."""
    c = np.array([s[0] for s in bat_shapes_local()])
    v = np.array([np.prod(s[1]) for s in bat_shapes_local()])
    return (c * v[:, None]).sum(axis=0) / v.sum()


BAT_ORIGIN = (BAT_START[0], BAT_START[1], BAT[2] / 2)
SEATED_X = STOP_X - BAT[0] / 2  # the pack's centre seated against the stop (0.2835)
SHORT_OF_STOP = 0.025  # the demo lowers the pack 25 mm short of the stop, then slides it home

# -- the power strip and its rocker (fixed) -------------------------------------------------------------------------
STRIP_XY = (0.240, 0.200)
STRIP = (0.050, 0.280, 0.040)  # x, y, z (est)
STRIP_TOP = STRIP[2]
ROCK_XY = (STRIP_XY[0], STRIP_XY[1] - STRIP[1] / 2 + 0.020)  # (0.240, 0.080): 2 cm in from the strip's right end
ROCKER = (0.020, 0.014, 0.006)  # its top block (est)
TILT_DEG = 8.0  # either way from level (est)
TRAVEL_DEG = 2 * TILT_DEG  # joint 0 = ON (drawn: the -x "I" end down), 16 deg = OFF (the +x "O" end down)
CENTRE_DEG = TILT_DEG
PRELOAD_DEG = 3.0
OFF_DEG = CENTRE_DEG + 2.0  # reads OFF 2 deg past the centre toward OFF ...
ON_DEG = CENTRE_DEG - 2.0  # ... and ON again 2 deg before it (hysteresis)
SWITCH_FORCE = 2.0  # N at the press point at the centre (est)
PRESS_U = 0.006  # the press point 6 mm from the pivot along x
ROCK_K = SWITCH_FORCE * PRESS_U / math.radians(TILT_DEG + PRELOAD_DEG)
ROCK_MASS = 0.003
ROCK_DAMP = 0.0003
_RAISED = 0.008  # the raised end's top 8 mm proud of the strip's top
PIVOT_Z = STRIP_TOP + _RAISED - ROCKER[2] * math.cos(math.radians(TILT_DEG)) - (ROCKER[0] / 2) * math.sin(
    math.radians(TILT_DEG))
PLUG = ((0.265, 0.290), (0.140, 0.160), (0.008, 0.032))  # the charger's plug in the strip's side outlet (visual)
CORD_R = 0.003
CORD_X = 0.293
CORD_Y = (CH_Y[1], 0.150)


def rocker_top_z(u: float, joint_deg: float) -> float:
    """Height of the rocker's top surface at ``u`` (m along +x from the pivot) with the joint at ``joint_deg``."""
    a = math.radians(joint_deg - TILT_DEG)  # + tips the +x end down
    return PIVOT_Z + ROCKER[2] * math.cos(a) - u * math.sin(a)


# -- the parts tray (fixed) -----------------------------------------------------------------------------------------
TRAY_XY = (0.330, -0.340)
TRAY_OUT = (0.150, 0.110, 0.020)
TRAY_WALL = 0.004
TRAY_FLOOR = 0.003
TRAY_IN_X = (TRAY_XY[0] - TRAY_OUT[0] / 2 + TRAY_WALL, TRAY_XY[0] + TRAY_OUT[0] / 2 - TRAY_WALL)
TRAY_IN_Y = (TRAY_XY[1] - TRAY_OUT[1] / 2 + TRAY_WALL, TRAY_XY[1] + TRAY_OUT[1] / 2 - TRAY_WALL)

# -- scoring thresholds ---------------------------------------------------------------------------------------------
ZONE_X = TONGUE_X  # the terminal zone: over the tongue's 30 x 20 mm top, from it to 15 mm up
ZONE_Y = (SEAT_YC - TONGUE_HALF_Y, SEAT_YC + TONGUE_HALF_Y)
ZONE_Z = (TONGUE_TOP, TONGUE_TOP + 0.015)
RAILS_BOTTOM = 0.004  # "on the rails": the pack's bottom within 4 mm of the deck ...
RAILS_Y = 0.010  # ... its centre within 10 mm of the seat's centre line ...
RAILS_FRONT = (STOP_X - 0.050, STOP_X + 0.004)  # ... and its +x face from 50 mm short of the stop to the stop
SEAT_MM = 0.003  # seated: the +x face within 3 mm of the stop
SEAT_TILT_DEG = 3.0
LIVE_MM = 0.005  # live cleaning: a finger zone within 5 mm of the socket while it lies in the zone, power ON
DISTURB_M = 0.002  # live disturbance: the socket moved 2 mm from where it lay in the zone, power ON (by anything)
CLEAR_M = 0.030  # hands 3 cm off the pack at the end
WINDOW_S = 1.0
WIN_STEPS = int(round(WINDOW_S * CONTROL_HZ))
STILL_M = 0.002
OFF_TABLE_DROP = 0.010
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)
RESET_JUMP_M = 0.05

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
CH_BLACK = (0.12, 0.12, 0.13)
CH_YELLOW = (0.95, 0.75, 0.10)
CONTACT_BRASS = (0.80, 0.65, 0.30)
CHROME = (0.80, 0.81, 0.83)
BAT_BLACK = (0.15, 0.15, 0.16)
BAT_YELLOW = (0.95, 0.75, 0.10)
STRIP_GREY = (0.70, 0.71, 0.72)
ROCKER_RED = (0.80, 0.10, 0.08)
MARK_WHITE = (0.96, 0.96, 0.96)
PLUG_BLACK = (0.10, 0.10, 0.11)
TRAY_GREY = (0.35, 0.37, 0.40)


def in_zone(p) -> bool:
    """A point (world) inside the terminal zone."""
    x, y, z = float(p[0]), float(p[1]), float(p[2]) - TOP
    return (ZONE_X[0] - EPS <= x <= ZONE_X[1] + EPS and ZONE_Y[0] - EPS <= y <= ZONE_Y[1] + EPS
            and ZONE_Z[0] - EPS <= z <= ZONE_Z[1] + EPS)


def on_charger_xy(p) -> bool:
    """A point (world) over the charger's footprint."""
    return CH_X[0] <= float(p[0]) <= CH_X[1] and CH_Y[0] <= float(p[1]) <= CH_Y[1]
