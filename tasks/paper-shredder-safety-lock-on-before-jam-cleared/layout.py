"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP unless named *_W.

Scene (sources in README.md; "est" marks a figure not read at a source):
- A desktop cross-cut paper shredder head assembly (Fellowes Powershred 60Cs class: 230 mm / 9 in feed throat,
  manual Safety Lock slide switch, top power rocker switch) resting on a 6 cm benchtop riser collar (est), loose on
  four rubber feet (1.4 kg est): 210 mm (x) x 320 mm (y) x 115 mm (z) main deck.
- A recessed feed slot (230 mm along y, 8 mm wide along x, 45 mm deep) at x 29.5 cm, lined with silicone friction pads
  calibrated to 4.5 N vertical extraction resistance.
- A green-lit bistable power rocker switch (30 x 22 mm, hinged about y, starts ON at +12 deg, snaps OFF to -12 deg
  with 3.5 N at the press point) on the left-front top deck at (x 21.5, y +10.0 cm).
- An orange bistable Safety Lock slide switch (14 x 24 x 14 mm thumb tab, 20 mm travel along +x, 3.5 N detent) on the
  right-front top deck at (x 20.5, y -9.0 cm), starting at 0 mm (UNLOCKED) and sliding to +20 mm (LOCKED).
- A rigid 3-ply corrugated cardboard mailer (4 mm thick x 160 mm wide x 120 mm tall, 45 g, est) jammed 45 mm deep
  in the feed slot (bottom at z 7.0 cm, top at z 19.0 cm, protruding 75 mm above the deck).
- A grey discard tray (outer 220 x 160 x 20 mm, 4 mm walls, 3 mm floor, est) fixed at (x 26.0, y -27.5 cm) on the
  robot's right.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MM = 0.001
EPS = 1e-9

# -- shredder head assembly (loose, 1.4 kg est on rubber feet) -----------------------------------------------------
SHRED_XY = (0.245, 0.010)
SHRED_SIZE = (0.210, 0.320, 0.095)  # x (depth), y (width), z (deck height above table)
SHRED_X = (SHRED_XY[0] - SHRED_SIZE[0] / 2, SHRED_XY[0] + SHRED_SIZE[0] / 2)  # 0.140 .. 0.350
SHRED_Y = (SHRED_XY[1] - SHRED_SIZE[1] / 2, SHRED_XY[1] + SHRED_SIZE[1] / 2)  # -0.150 .. +0.170
SHRED_FOOT_H = 0.005  # rubber feet height (est)
SHRED_FOOT_W = 0.028  # square rubber foot width (est)
SHRED_DECK_Z = SHRED_SIZE[2]  # 0.095 m above table
SHRED_MASS = 1.40  # kg (head + benchtop riser collar, est)
SHRED_FRICTION = (0.85, 0.75)  # rubber feet on bench (est)

# feed slot (Fellowes Powershred 60Cs: 230 mm / 9 in throat width)
SLOT_X = 0.275
SLOT_W = 0.008  # 8 mm wide along x (2 mm clearance each side of 4 mm mailer)
SLOT_LEN = 0.230  # 230 mm along y
SLOT_Y = (SHRED_XY[1] - SLOT_LEN / 2, SHRED_XY[1] + SLOT_LEN / 2)  # -0.105 .. +0.125
SLOT_DEPTH = 0.040  # 40 mm deep below deck
SLOT_FLOOR_Z = SHRED_DECK_Z - SLOT_DEPTH  # 0.055 m above table

# -- green-lit power rocker switch (hinged about -Y on shredder base; starts ON at +12 deg) ------------------------
ROCKER_XY = (0.195, 0.105)  # left-front top deck
ROCKER_BEZEL_OUT = (0.034, 0.026)
ROCKER_BEZEL_IN = (0.031, 0.023)
ROCKER_BEZEL_H = 0.002
ROCKER_SIZE = (0.030, 0.022, 0.006)  # x, y, z (1550-series 30 x 22 mm rocker class)
ROCKER_PIVOT_Z = SHRED_DECK_Z - 0.0025  # 0.0925 m above table
ROCKER_TILT_DEG = 12.0  # starts at +12 deg (ON); snaps to -12 deg (OFF)
ROCKER_TRAVEL_DEG = 2.0 * ROCKER_TILT_DEG  # 24 deg total joint travel (0 = ON, 24 deg = OFF)
ROCKER_CENTRE_DEG = ROCKER_TILT_DEG  # 12 deg dead-centre
ROCKER_PRELOAD_DEG = 3.0
ROCKER_OFF_THRESHOLD_DEG = -8.0  # rocker_deg <= -8 deg counts as OFF (joint >= 20 deg)
ROCKER_PRESS_U = 0.011  # press point 11 mm toward -x (OFF end) from pivot
ROCKER_FORCE_N = 3.5  # N at the press point to snap past centre (1550-series class)
ROCKER_K = ROCKER_FORCE_N * ROCKER_PRESS_U / math.radians(ROCKER_TILT_DEG + ROCKER_PRELOAD_DEG)
ROCKER_MASS = 0.006
ROCKER_DAMP = 0.0005

# -- orange Safety Lock slide switch (slider along +X on shredder base; 0 mm UNLOCKED -> 20 mm LOCKED) -------------
LOCK_XY = (0.190, -0.090)  # right-front top deck (rest / UNLOCKED centre)
LOCK_TRAVEL = 0.020  # 20 mm along +X to LOCKED
LOCK_MID = 0.010  # 10 mm bistable detent midpoint
LOCK_PRELOAD = 0.004  # 4 mm preload past stop
LOCK_FORCE_N = 3.5  # N at the detent midpoint
LOCK_K = LOCK_FORCE_N / (LOCK_MID + LOCK_PRELOAD)  # 250 N/m
LOCK_MASS = 0.008
LOCK_DAMP = 4.0  # N s / m
LOCK_TAB = (0.014, 0.024, 0.014)  # x, y, z thumb slider tab
LOCK_TAB_Z0 = SHRED_DECK_Z + 0.001  # bottom 1 mm above deck
LOCK_TAB_Z1 = LOCK_TAB_Z0 + LOCK_TAB[2]  # 0.110 m above table
LOCK_TRACK_X = (LOCK_XY[0] - 0.012, LOCK_XY[0] + LOCK_TRAVEL + 0.012)  # 0.178 .. 0.222
LOCK_TRACK_Y = (LOCK_XY[1] - 0.016, LOCK_XY[1] + 0.016)  # -0.106 .. -0.074
LOCK_TRACK_H = 0.003
LOCK_ON_THRESHOLD = 0.017  # >= 17 mm counts as LOCKED

# -- brace area on left front deck ---------------------------------------------------------------------------------
BRACE_XY = (0.190, 0.150)  # left arm braces here during mailer extraction

# -- jammed corrugated cardboard mailer (loose rigid body, 45 g) ---------------------------------------------------
MAILER_SIZE = (0.004, 0.160, 0.105)  # x (thickness 4 mm), y (width 160 mm), z (height 105 mm)
MAILER_MASS = 0.045  # kg (rigid 3-ply corrugated mailer, est)
MAILER_FRICTION = (0.65, 0.55)
MAILER_X = SLOT_X  # 0.275
MAILER_Y = -0.025  # spans y -0.105 .. +0.055 (inside slot -0.105 .. +0.125)
MAILER_Z0 = SLOT_FLOOR_Z  # 0.055 m (seated 40 mm deep in slot)
MAILER_Z1 = MAILER_Z0 + MAILER_SIZE[2]  # 0.160 m above table (protrudes 65 mm above deck)
MAILER_ORIGIN = (MAILER_X, MAILER_Y, MAILER_Z0 + MAILER_SIZE[2] / 2)  # (0.275, -0.025, 0.1075)
JAM_FORCE_N = 4.5  # N calibrated vertical extraction force
JAM_TORQUE_NM = 0.45  # N m break torque for the jam bond

# -- discard tray (fixed, grey, 22 x 16 x 2 cm outer) --------------------------------------------------------------
TRAY_XY = (0.260, -0.275)
TRAY_OUT = (0.220, 0.160, 0.020)  # x, y, z outer
TRAY_WALL = 0.004
TRAY_FLOOR = 0.003
TRAY_IN_X = (TRAY_XY[0] - TRAY_OUT[0] / 2 + TRAY_WALL, TRAY_XY[0] + TRAY_OUT[0] / 2 - TRAY_WALL)  # 0.154 .. 0.366
TRAY_IN_Y = (TRAY_XY[1] - TRAY_OUT[1] / 2 + TRAY_WALL, TRAY_XY[1] + TRAY_OUT[1] / 2 - TRAY_WALL)  # -0.351 .. -0.199

# -- scoring thresholds --------------------------------------------------------------------------------------------
UNSAFE_MOVE_M = 0.010  # > 1.0 cm mailer movement while unsafe latches 0
UNSAFE_TOUCH_M = 0.005  # gripper finger zone within 5 mm of mailer while unsafe latches 0
SLOT_CLEAR_Z = SHRED_DECK_Z + 0.005  # lowest mailer corner above this -> extracted from slot
IN_TRAY_Z_MAX = 0.008  # lowest corner within 8 mm above tray floor
FLAT_DEG = 10.0  # mailer thickness axis within 10 deg of vertical
RELEASE_CLEAR = 0.040  # >= 4.0 cm clear from mailer and shredder
OPEN_THRESHOLD = 0.85  # both grippers open
STILL_WINDOW_S = 1.0
WIN_STEPS = int(round(STILL_WINDOW_S * CONTROL_HZ))  # 20 steps
STILL_MOVE = 0.002  # <= 2 mm centre movement over the 1.0 s window
OFF_TABLE_DROP = 0.010
TABLE_X = (0.36 - 0.55, 0.36 + 0.55)
TABLE_Y = (-0.65, 0.65)

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
SHRED_DARK = (0.14, 0.15, 0.17)
SHRED_TRIM = (0.24, 0.25, 0.28)
RUBBER_BLACK = (0.07, 0.07, 0.08)
CUTTER_STEEL = (0.68, 0.70, 0.73)
WARN_YELLOW = (0.93, 0.76, 0.10)
BEZEL_BLACK = (0.09, 0.09, 0.10)
ROCKER_GREEN = (0.12, 0.78, 0.26)
MARK_WHITE = (0.96, 0.96, 0.96)
LOCK_ORANGE = (0.95, 0.42, 0.06)
LOCK_TRACK = (0.20, 0.21, 0.23)
LOCK_RED = (0.85, 0.10, 0.10)
KRAFT_BROWN = (0.68, 0.52, 0.34)
LABEL_WHITE = (0.95, 0.94, 0.91)
TRAY_GREY = (0.52, 0.54, 0.57)


def rocker_deg_from_joint(joint_rad: float) -> float:
    """Rocker angle in degrees: +12 deg = ON (joint 0), 0 deg = centre (joint 12 deg), -12 deg = OFF (joint 24 deg)."""
    return ROCKER_TILT_DEG - math.degrees(float(joint_rad))


def rocker_top_z(u_off: float, joint_deg: float) -> float:
    """Height above table of the rocker's top surface at ``u_off`` (m toward -x from pivot) at ``joint_deg``."""
    # At joint_deg = 0 (ON), -x end is raised by +12 deg. At joint_deg = 24 (OFF), -x end is depressed by -12 deg.
    a = math.radians(ROCKER_TILT_DEG - joint_deg)
    return ROCKER_PIVOT_Z + (ROCKER_SIZE[2] / 2) * math.cos(a) + u_off * math.sin(a)


def mailer_corners(origin, rot) -> np.ndarray:
    """The mailer's 8 corners in world coordinates from its centre ``origin`` and 3x3 ``rot``."""
    hx, hy, hz = MAILER_SIZE[0] / 2, MAILER_SIZE[1] / 2, MAILER_SIZE[2] / 2
    local = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return np.asarray(origin, dtype=float) + local @ np.asarray(rot, dtype=float).T
