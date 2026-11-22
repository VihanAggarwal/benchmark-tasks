"""Every dimension of the task in one place: metres, kilograms, seconds, real size. NumPy only.

World frame: x forward from the line through the two arm bases, y to the robot's left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are ABOVE THE TABLE TOP unless named *_W.

Scene (sources in README.md; "est" marks a figure not read at a source):
- A compact 6-inch clip-on desk fan (Igenix DF0006 class: 20 cm H x 20 cm D x 13.5 cm W, 0.64 kg, 152 mm guard
  diameter, 3-blade impeller, rear motor housing with top 2-speed rocker power switch), clipped onto a fixed 18 mm
  bench mounting block so the fan body does not slide or tip.
- The fan is centred at (x 25.0 cm, y -5.0 cm) and faces +y (toward the robot's left arm). Its spoked circular wire
  front guard (152 mm outer diameter) is hinged along its bottom lip (y -0.5 cm, z 1.8 cm) and rests ajar at 22 deg
  open against a seating spring (~2.5 N at the push point, ~1.2 N at the top rim, est), exposing the 3-blade
  impeller inside the shroud.
- A red spring-steel top retaining toggle clip (hinged along x at the top of the shroud rim, y -1.4 cm, z 17.4 cm,
  with a 42 mm rear press lever carrying an 8 mm downward latch pawl tooth on its underside at u 16 mm rearward of the
  pivot, y -3.0 cm; JW Winco GN 832-style top toggle latch) extends rearward across the shroud top, standing open
  12 deg above horizontal and rotating 24 deg down to -12 deg so its underside latch pawl drops straight down into the
  keeper slot in front of the rear keeper crossbar (y -3.2 to -3.6 cm) of the front guard's top keeper plate, physically
  blocking the keeper crossbar from moving forward toward +y when the guard tries to open (~3.0 N at the 32 mm press
  point, est). It stays latched only while the front guard is seated within 3 degrees of closed; with the guard ajar it
  springs back open.
- A 2-speed rocker power switch sits on top of the rear motor housing (x 25.0 cm, y -12.8 cm, z 12.6 cm), rocking
  about y. Its ON end (+x, marked white) starts raised at +12 degrees (OFF); pressing it down needs ~3.5 N at the
  11 mm press point (est), snapping ON at <= -8 degrees (-12 degrees stop).
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

# -- fan base, mounting block and rear shroud (fixed) ---------------------------------------------------------------
FAN_ORIGIN = (0.250, -0.050)  # base frame (x, y) on the table top
FX, FY = FAN_ORIGIN
BLOCK_X = (FX - 0.060, FX + 0.060)  # 120 mm wooden desk-edge mounting block bolted to the bench
BLOCK_Y = (FY - 0.065, FY + 0.035)  # 100 mm
BLOCK_H = 0.018  # 18 mm thick desk-edge block

SHROUD_CX = FX  # 0.250
SHROUD_R = 0.076  # 76 mm radius -> 152 mm (6 in) outer wire guard diameter
GUARD_HINGE_Y = FY + 0.045  # -0.005 m: front rim of rear shroud
GUARD_HINGE_Z = BLOCK_H  # 0.018 m over the table
SHROUD_CZ = GUARD_HINGE_Z + SHROUD_R  # 0.094 m: fan axis height (top rim at 0.170 m)
SHROUD_REAR_Y = FY - 0.020  # -0.070 m: back plane of rear wire cage
SHROUD_RIM_T = 0.006  # rim ring radial width
SHROUD_RIM_Y = 0.006  # rim ring thickness along y

# 3-blade impeller inside the shroud cavity
HUB_R = 0.022
HUB_Y = (FY - 0.015, FY + 0.030)
BLADE_Y = FY + 0.016
BLADE_R_OUT = 0.068  # 136 mm blade sweep inside 152 mm guard
BLADE_W = 0.026
BLADE_T = 0.004

# Impeller cavity keep-out zone (world box relative to table top): hard-0 if any finger zone enters while guard open
CAVITY_X = (SHROUD_CX - 0.068, SHROUD_CX + 0.068)  # 0.182 .. 0.318
CAVITY_Y = (FY - 0.016, FY + 0.038)  # -0.066 .. -0.012 (strictly behind shroud rim -0.005 and guard face +0.006)
CAVITY_Z = (SHROUD_CZ - 0.068, SHROUD_CZ + 0.068)  # 0.026 .. 0.162

# Rear motor housing pod (extends in -y behind the rear shroud)
MOTOR_X = (SHROUD_CX - 0.038, SHROUD_CX + 0.038)  # 76 mm wide
MOTOR_Y = (FY - 0.114, SHROUD_REAR_Y)  # -0.164 .. -0.070 (94 mm deep)
MOTOR_Z = (0.062, 0.128)  # top at 12.8 cm

# -- hinged front wire guard (joint "guard", revolute about +X) -----------------------------------------------------
# Joint 0 = ajar at GUARD_OPEN_DEG (22 deg tilted outward toward +y); joint GUARD_OPEN_DEG = 0 deg (flush closed).
GUARD_OPEN_DEG = 22.0
GUARD_SHUT_Y0 = GUARD_HINGE_Y  # -0.005 m: inner face at shroud rim
GUARD_SHUT_Y1 = FY + 0.056  # +0.006 m: outer domed front face (11 mm proud of rim, 18 mm proud of impeller cavity)
GUARD_MASS = 0.140  # kg (est for 152 mm wire guard)
GUARD_PUSH_X = SHROUD_CX + 0.068  # 0.318: left arm pushes on the +x rim of the guard, clear of right arm
GUARD_PUSH_Z = 0.088  # 70 mm above bottom hinge
GUARD_PUSH_ARM = GUARD_PUSH_Z - GUARD_HINGE_Z  # 0.070 m
GUARD_PUSH_FORCE = 2.5  # N at the push point when seated flush (~1.2 N at top rim, est)
GUARD_K = GUARD_PUSH_FORCE * GUARD_PUSH_ARM / math.radians(GUARD_OPEN_DEG)  # ~0.456 N m / rad
GUARD_DAMP = 0.025  # N m s / rad
GUARD_HOLD_PRELOAD_DEG = 8.0  # when latched by the top clip, drive target sits 8 deg past the closed stop
GUARD_RIM_SEGS = 12  # 12-segment circular wire rim

# -- top retaining toggle clip (joint "clip", revolute about +X, lower = 0, upper = +24 deg) ------------------------
# Extends rearward (-y) from its hinge at the top-rear of the shroud; drawn at joint 0 = OPEN (press tab raised +12 deg).
# Positive rotation about +X tips the -y press tab AND its underside latch pawl DOWN by 24 deg to -12 deg, dropping the
# pawl at y = -0.030 m into the keeper slot in front (+y side) of the rear keeper crossbar (y = -0.036..-0.032 m), so any
# +y opening motion of the front guard is physically blocked by the pawl.
CLIP_X = SHROUD_CX - 0.028  # 0.222: mounted at -28 mm on top rim (22 deg from crown) for generous arm-arm clearance
CLIP_PIVOT_Y = FY + 0.036  # -0.014 m: top of rear shroud just behind the guard rim
CLIP_PIVOT_Z = 0.174  # 4 mm above guard top (0.170): clears shroud and guard at every angle
CLIP_LEN = 0.042  # press tab extends in -y from pivot (-0.014 m) to y = -0.056 m
CLIP_W = 0.028  # along x
CLIP_T = 0.005
CLIP_OPEN_TILT_DEG = 12.0  # press tab angle above horizontal when open (joint = 0)
CLIP_TRAVEL_DEG = 24.0  # rotates down 24 deg to -12 deg below horizontal
CLIP_CENTRE_DEG = CLIP_TRAVEL_DEG / 2.0  # 12.0 deg over-centre point
CLIP_PRELOAD_DEG = 3.0
CLIP_ON_DEG = CLIP_CENTRE_DEG + 3.0  # reads closed at >= 15.0 deg
CLIP_OFF_DEG = CLIP_CENTRE_DEG - 3.0  # reads open at <= 9.0 deg
CLIP_PRESS_U = 0.032  # press point 32 mm rearward (-y) from pivot (y = -0.046 m)
CLIP_FORCE = 3.0  # N at the press point at the centre (est)
CLIP_K = CLIP_FORCE * CLIP_PRESS_U / math.radians(CLIP_CENTRE_DEG + CLIP_PRELOAD_DEG)  # ~0.367 N m / rad
CLIP_MASS = 0.006
CLIP_DAMP = 0.0008
CLIP_BRACKET = (0.032, 0.014, 0.010)  # bracket connecting shroud top to clip hinge
CLIP_PAWL_U = 0.016  # underside latch pawl 16 mm rearward (-y) from pivot (y = -0.030 m, same -y lever side as press tab)
CLIP_PAWL_DROP = 0.008  # 8 mm downward latch pawl tooth on underside of clip lever (z 16.92 -> 16.25 cm when pressed down)
CLIP_PAWL_T = 0.004  # 4 mm y-thickness of the underside latch pawl (y = -0.032..-0.028 m)
GUARD_TANG_Y = (-0.036, 0.002)  # slotted top keeper plate on front guard extending rearward under clip to y = -0.036 m
GUARD_TANG_BAR_Y = (-0.036, -0.032)  # rear keeper crossbar (y = -0.036..-0.032 m) caught on its +y face by pawl at y = -0.030 m
GUARD_TANG_Z = (0.161, 0.166)  # keeper plate height when guard is shut
GUARD_TANG_W = 0.024  # 24 mm outer width along x centred at CLIP_X (4 mm side rails around 16 mm x 18 mm slot)
GUARD_TANG_RAIL = 0.004  # 4 mm side rail width along x

# -- rear 2-speed rocker power switch (joint "rocker", revolute about +Y, lower = 0, upper = +24 deg) ---------------
# Joint 0 = OFF (+12 deg tilt, +x ON end raised); joint 24 deg = ON (-12 deg tilt, +x ON end down).
# Tilt from level: tilt_deg = ROCKER_TILT_DEG - joint_deg -> +12 deg OFF, <= -8 deg ON (joint >= 20 deg).
ROCKER_XY = (SHROUD_CX, FY - 0.078)  # (0.250, -0.128)
BEZEL_OUT = (0.032, 0.024)
BEZEL_IN = (0.028, 0.020)
BEZEL_H = 0.002
ROCKER = (0.026, 0.018, 0.006)
ROCKER_PIVOT_Z = MOTOR_Z[1] - 0.002  # 0.126 m
ROCKER_TILT_DEG = 12.0  # +12 deg OFF, -12 deg ON
ROCKER_TRAVEL_DEG = 2.0 * ROCKER_TILT_DEG  # 24 deg total travel
ROCKER_CENTRE_DEG = ROCKER_TILT_DEG  # 12 deg over-centre point
ROCKER_PRELOAD_DEG = 3.0
ROCKER_ON_DEG = ROCKER_TILT_DEG + 8.0  # joint >= 20 deg -> tilt <= -8 deg (reads ON)
ROCKER_OFF_DEG = ROCKER_TILT_DEG - 4.0  # joint <= 8 deg -> tilt >= +4 deg (reads OFF)
SWITCH_FORCE = 3.5  # N at the 11 mm press point at the centre (est)
PRESS_U = 0.011  # press point from pivot along +x
ROCKER_K = SWITCH_FORCE * PRESS_U / math.radians(ROCKER_TILT_DEG + ROCKER_PRELOAD_DEG)  # ~0.147 N m / rad
ROCKER_MASS = 0.004
ROCKER_DAMP = 0.0005

# -- interlock & scoring thresholds ---------------------------------------------------------------------------------
SHUT_DEG = 2.0  # front guard within 2 deg of flush closed
LATCH_SEAT_DEG = 3.0  # top clip can lock/stay locked only while front guard <= 3 deg from closed
OPEN_LIVE_DEG = 5.0  # front guard open >= 5 deg while power is ON -> latched 0
CLEAR_M = 0.050  # both grippers >= 5 cm from the fan over the last 1 s
WINDOW_S = 1.0
WIN_STEPS = int(round(WINDOW_S * CONTROL_HZ))

# -- colours (sRGB) -------------------------------------------------------------------------------------------------
BLOCK_WOOD = (0.68, 0.54, 0.38)
CLAMP_DARK = (0.18, 0.19, 0.21)
FAN_WHITE = (0.90, 0.91, 0.92)
WIRE_GREY = (0.76, 0.78, 0.80)
GUARD_SILVER = (0.82, 0.84, 0.86)
BADGE_BLUE = (0.18, 0.36, 0.62)
BLADE_BLUE = (0.28, 0.52, 0.78)
HUB_GREY = (0.45, 0.47, 0.50)
MOTOR_WHITE = (0.86, 0.87, 0.89)
CLIP_RED = (0.78, 0.16, 0.14)
BEZEL_BLACK = (0.10, 0.10, 0.11)
ROCKER_BLACK = (0.16, 0.16, 0.18)
MARK_WHITE = (0.96, 0.96, 0.96)


# -- geometry helpers -----------------------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def guard_point(p_shut, open_deg: float) -> np.ndarray:
    """World (x, y, height over table) of a point on the front guard when open by ``open_deg`` degrees.
    Opening tilts the guard outward toward +y (rotation by -open_deg about +X through the bottom hinge)."""
    h = np.array([0.0, GUARD_HINGE_Y, GUARD_HINGE_Z])
    p = np.atleast_2d(np.asarray(p_shut, dtype=float)) - h
    p = p @ rot_x(-math.radians(open_deg)).T + h
    return p if np.ndim(p_shut) > 1 else p[0]


def guard_outer_y(z_shut: float, open_deg: float) -> float:
    """World y of the guard's outer face at shut-height ``z_shut`` when open by ``open_deg``."""
    return float(guard_point((SHROUD_CX, GUARD_SHUT_Y1, z_shut), open_deg)[1])


def clip_top_z(u: float, closed_deg: float) -> float:
    """Height (over the table) of the top retaining clip's top surface at ``u`` rearward (-y) from its hinge."""
    a = math.radians(closed_deg - CLIP_OPEN_TILT_DEG)  # + tips the -y end down
    return CLIP_PIVOT_Z + (CLIP_T / 2.0) * math.cos(a) - u * math.sin(a)


def rocker_top_z(u: float, joint_deg: float) -> float:
    """Height (over the table) of the rocker's top surface at ``u`` along +x from its pivot."""
    a = math.radians(joint_deg - ROCKER_TILT_DEG)  # + tips the +x ON end down
    return ROCKER_PIVOT_Z + ROCKER[2] * math.cos(a) - u * math.sin(a)


def rocker_tilt_deg(joint_deg: float) -> float:
    """Rocker tilt angle from level: +12 deg = OFF (ON end raised), -12 deg = ON (ON end down)."""
    return ROCKER_TILT_DEG - joint_deg


def guard_face_points(open_deg: float, step: float = 0.012) -> np.ndarray:
    """Sample points on the front wire guard disc (world x, y, height over table) for offline contact checks."""
    xs = np.arange(SHROUD_CX - SHROUD_R, SHROUD_CX + SHROUD_R + 1e-9, step)
    zs = np.arange(SHROUD_CZ - SHROUD_R, SHROUD_CZ + SHROUD_R + 1e-9, step)
    pts = []
    r2 = SHROUD_R ** 2 + 1e-9
    for x in xs:
        dx2 = (x - SHROUD_CX) ** 2
        for z in zs:
            if dx2 + (z - SHROUD_CZ) ** 2 <= r2:
                pts.append((x, GUARD_SHUT_Y1, z))
    return guard_point(np.array(pts, dtype=float), open_deg)


def cavity_box() -> tuple:
    """(centre, size) world of the impeller cavity keep-out zone (hard-0 if entered while guard is open)."""
    cx = (CAVITY_X[0] + CAVITY_X[1]) / 2.0
    cy = (CAVITY_Y[0] + CAVITY_Y[1]) / 2.0
    cz = TOP + (CAVITY_Z[0] + CAVITY_Z[1]) / 2.0
    return ((cx, cy, cz), (CAVITY_X[1] - CAVITY_X[0], CAVITY_Y[1] - CAVITY_Y[0], CAVITY_Z[1] - CAVITY_Z[0]))
