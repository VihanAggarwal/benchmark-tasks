"""Every dimension and threshold of the task in one place, and the shears' and tote's geometry. NumPy only.

World frame (station): x forward from the line through the two arm bases, y to the robot's left, z up.
The arm bases stand at x 0, y +-0.305 m; the station bench top is at z = 0.7475 m. Heights below are
above the bench top unless named *_W.

Product reference: Fiskars SingleStep P26 Bypass Pruner (model 111260; retailer spec:
20.8 cm overall length, 7.3 cm closed width (est), 2.2 cm thickness (est), 200 g mass, 22 mm cutting capacity,
top-mounted linear sliding thumb safety catch, geometry and travel est).

Shears local frame (base body = lower handle + anvil + pivot housing):
- Origin (0, 0, 0) on the bench top directly under the pivot pin axis.
- +x forward along the closed shears toward the blade tips (+63 mm), -x back toward the handle ends (-145 mm).
  Total closed length: 145 + 63 = 208 mm (20.8 cm).
- +y to the robot's left when yaw = 0, z up from the bench top.
- Spawn pose: pivot at (0.260, +0.015, TOP), yaw = 0.0 rad (handles back along -x to x = 0.115, blades
  forward along +x to x = 0.323).
- Hinge joint ``jaw`` about +Z at (0, 0, 0):
  Drawn at the open starting pose (THETA_OPEN_DEG = 20.5 deg open, blade tip gap ~22 mm matching Fiskars P26
  cutting capacity, outer handle span at grip zone x_local = -112 mm is 76.5 mm open -> 38.0 mm closed,
  overall closed width across bolsters is 73.0 mm).
  q_jaw = 0.0 rad -> open 20.5 deg; q_jaw = +0.3578 rad (+20.5 deg) -> closed 0.0 deg.
- Slider joint ``catch`` along -Y on top of the pivot housing at anchor (0.014, +0.006, 0.022):
  q_catch = 0.0 m (unlocked) -> q_catch = 0.012 m (locked, 12 mm linear travel). Its 12 x 12 x 10 mm tab is drawn
  in the BASE frame centred at (0.014, 0.006, 0.027), on the catch rail (mechkit draws link shapes in the base
  frame; GPU runs 1-5 drew it at (0, 0, 0.005), buried in the pivot boss).

Garden tool tote (fixed, open-top green polypropylene):
- Centred at (0.280, -0.200) m on the robot's right.
- Interior: 26.0 cm (x: 0.150 to 0.410) x 18.0 cm (y: -0.290 to -0.110) x 6.0 cm high (rim at z = 0.060 m).
- Floor 4 mm thick (z = 0.000 to 0.004 m), walls 4 mm thick (z = 0.000 to 0.060 m).
"""

from __future__ import annotations

import math
import numpy as np

TOP = 0.7475  # station bench top (world z, m)
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MM = 0.001
EPS = 1e-9

# -- colours (sRGB 0..1) -------------------------------------------------------------------------------------------
HANDLE_BLACK = (0.16, 0.17, 0.19)
ACCENT_ORANGE = (0.90, 0.42, 0.08)
STEEL_BRIGHT = (0.76, 0.78, 0.80)
STEEL_DARK = (0.28, 0.30, 0.33)
TOTE_GREEN = (0.16, 0.45, 0.22)

# -- shears spawn & dimensions (metres, degrees, kg) ---------------------------------------------------------------
SHEARS_ORIGIN = (0.260, 0.015, TOP)  # 0.330 until 22 Nov: 7 cm nearer the robot gives the arms ~1.5 cm more room
SHEARS_YAW = 0.0

LENGTH_BACK = 0.145  # pivot to handle tail (m)
LENGTH_FRONT = 0.063  # pivot to blade tip (m), total 0.208 m (20.8 cm, Fiskars P26 spec)
HANDLE_W = 0.012  # single handle bar width (m)
HANDLE_H = 0.016  # single handle bar height (m)
HANDLE_Y_OFF = 0.013  # closed centre-line offset from midline (+y upper, -y lower) -> 38 mm closed grip span
BOLSTER_W_HALF = 0.0365  # 73 mm closed overall width across bolsters (Fiskars P26 spec)
PIVOT_H = 0.022  # 22 mm thickness at pivot (Fiskars P26 spec)

# Handle grip zone (for squeezing and carrying)
GRIP_X_LOCAL = -0.112  # 112 mm behind pivot (33 mm from handle tail), x_world = 0.148 m

# Hinge joint ("jaw"): 0 rad = open 20.5 deg, +20.5 deg = closed 0.0 deg
THETA_OPEN_DEG = 20.5
JAW_LOWER = 0.0
JAW_UPPER = math.radians(THETA_OPEN_DEG)
JAW_SPRING = 1.40  # N m / rad -> 4.5 N return force at r = 0.112 m when closed (est)
JAW_DAMPING = 0.08  # N m s / rad
JAW_FRICTION = 0.02  # N m

# Thumb safety catch slider ("catch"): slides 12 mm along -Y on top of pivot housing
CATCH_ANCHOR = (0.014, 0.006, 0.022)  # base frame (m)
CATCH_SIZE = (0.012, 0.012, 0.010)  # 12 x 12 x 10 mm orange sliding thumb lock tab
CATCH_TRAVEL = 0.012  # 12 mm linear travel along -Y (est)
CATCH_SPRING = 166.7  # N/m -> 2.0 N detent spring across 12 mm (est)
CATCH_DAMPING = 2.0  # N s / m
CATCH_FRICTION = 0.20  # N

# Masses (total = 0.200 kg = 200 g, Fiskars P26 manufacturer spec)
BASE_MASS = 0.105
UPPER_MASS = 0.085
CATCH_MASS = 0.010
FRICTION = 0.75  # textured FiberComp handles + bench contact

# Interlock thresholds
CLOSED_MAX_DEG = 5.0  # handles must be <= 5 deg from closed
CATCH_ENGAGE_M = 0.009  # catch must be slid >= 9 mm (of 12 mm) to lock
CATCH_RELEASE_M = 0.006  # catch pushed back below 6 mm unlocks

# -- garden tool tote (fixed, open-top green polypropylene) --------------------------------------------------------
TOTE_CENTRE = (0.280, -0.200)
TOTE_IN_X = (0.150, 0.410)  # 26.0 cm interior length
TOTE_IN_Y = (-0.290, -0.110)  # 18.0 cm interior width
TOTE_WALL = 0.004  # 4 mm polypropylene walls
TOTE_FLOOR = 0.004  # 4 mm floor
TOTE_RIM = 0.060  # 6.0 cm rim height above bench top
TOTE_OUT_X = (TOTE_IN_X[0] - TOTE_WALL, TOTE_IN_X[1] + TOTE_WALL)
TOTE_OUT_Y = (TOTE_IN_Y[0] - TOTE_WALL, TOTE_IN_Y[1] + TOTE_WALL)

# -- scoring thresholds --------------------------------------------------------------------------------------------
WIN_STEPS = int(round(1.0 * CONTROL_HZ))  # 1.0 s window (20 steps)
CLEAR_M = 0.040  # >= 4.0 cm clear of shears for 1.0 s
STILL_POS_M = 0.002  # <= 2 mm translation over 1.0 s
STILL_ANG_DEG = 2.0  # <= 2 deg rotation over 1.0 s
TILT_MAX_DEG = 10.0  # lying flat in tote
OFF_BENCH_DROP = 0.050  # 5 cm below bench top
RESET_JUMP_M = 0.050  # teleport detection threshold


# -- 2D / 3D helpers -----------------------------------------------------------------------------------------------
def rot2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def open_deg_of(q_jaw: float) -> float:
    """Handle opening angle in degrees: 22.0 at spawn (q_jaw = 0), 0.0 when closed (q_jaw = JAW_UPPER)."""
    return THETA_OPEN_DEG - math.degrees(float(q_jaw))


def q_jaw_of(open_deg: float) -> float:
    """Inverse of open_deg_of: joint angle (rad) for a given opening angle (deg)."""
    return math.radians(THETA_OPEN_DEG - float(open_deg))


def handle_outer_span_at(x_local: float, open_deg: float) -> tuple[float, float, float]:
    """Return (y_mid, span_y, span_perp) in metres at local x_local (negative behind pivot) for a given open_deg.
    Lower handle outer edge is at y = -(HANDLE_Y_OFF + HANDLE_W/2).
    Upper handle centre line at closed is y = +HANDLE_Y_OFF; when open by open_deg it rotates by -open_deg about Z.
    """
    th = math.radians(open_deg)
    # Closed point on upper handle at (x_local, +HANDLE_Y_OFF) rotated by -th around (0, 0)
    # Wait: x_local < 0, so rotating by -th moves it to:
    # x' = x_local*cos(th) + HANDLE_Y_OFF*sin(th)
    # y' = -x_local*sin(th) + HANDLE_Y_OFF*cos(th)
    y_upper_c = -x_local * math.sin(th) + HANDLE_Y_OFF * math.cos(th)
    y_upper_out = y_upper_c + (HANDLE_W / 2.0) / max(math.cos(th), 0.5)
    y_lower_out = -(HANDLE_Y_OFF + HANDLE_W / 2.0)
    span_y = y_upper_out - y_lower_out
    y_mid = 0.5 * (y_upper_out + y_lower_out)
    # Perpendicular span along bisector (-th/2)
    half = th / 2.0
    span_perp = 2.0 * (-x_local * math.sin(half) + HANDLE_Y_OFF * math.cos(half) + HANDLE_W / 2.0)
    return float(y_mid), float(span_y), float(span_perp)


# -- shape lists: (name, centre_m, size_m, yaw_deg, color) ---------------------------------------------------------
def base_box_specs() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], float, tuple]]:
    """Boxes on the base body (lower handle + bolster + pivot base + anvil + catch rail) in base frame (m)."""
    return [
        # Lower handle: x in [-0.145, -0.015], y in [-0.019, -0.007], z in [0, 0.016]
        ("handle_lower", (-0.080, -HANDLE_Y_OFF, HANDLE_H / 2), (0.130, HANDLE_W, HANDLE_H), 0.0, HANDLE_BLACK),
        # Lower palm bolster / guard: x in [-0.038, -0.010], y in [-0.0365, -0.014], z in [0, 0.016]
        ("bolster_lower", (-0.024, -0.02525, HANDLE_H / 2), (0.028, 0.0225, HANDLE_H), 0.0, ACCENT_ORANGE),
        # Pivot housing lower boss: x in [-0.015, +0.022], y in [-0.016, +0.016], z in [0, 0.010]
        ("pivot_base", (0.0035, 0.0, 0.005), (0.037, 0.032, 0.010), 0.0, STEEL_DARK),
        # Catch rail housing on top of pivot boss: x in [+0.006, +0.022], y in [-0.014, +0.014], z in [0.018, 0.022]
        ("catch_rail", (0.014, 0.0, 0.020), (0.016, 0.028, 0.004), 0.0, STEEL_DARK),
        # Anvil / counter-blade (two segments forming curved profile): z in [0, 0.010]
        ("anvil_root", (0.032, 0.004, 0.005), (0.024, 0.012, 0.010), 0.0, STEEL_DARK),
        ("anvil_tip", (0.052, 0.002, 0.005), (0.020, 0.009, 0.010), -10.0, STEEL_DARK),
    ]


def upper_box_specs_closed() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], float, tuple]]:
    """Boxes on the hinged upper body in its CLOSED pose (open_deg = 0)."""
    return [
        # Upper handle: x in [-0.145, -0.015], y in [+0.007, +0.019], z in [0, 0.016]
        ("handle_upper", (-0.080, HANDLE_Y_OFF, HANDLE_H / 2), (0.130, HANDLE_W, HANDLE_H), 0.0, HANDLE_BLACK),
        # Upper palm bolster / guard: x in [-0.038, -0.010], y in [+0.014, +0.0365], z in [0, 0.016]
        ("bolster_upper", (-0.024, 0.02525, HANDLE_H / 2), (0.028, 0.0225, HANDLE_H), 0.0, ACCENT_ORANGE),
        # Upper pivot plate: x in [-0.015, +0.022], y in [-0.012, +0.012], z in [0.011, 0.017]
        ("pivot_upper", (0.0035, 0.0, 0.014), (0.037, 0.024, 0.006), 0.0, STEEL_BRIGHT),
        # Curved bypass blade (passes above anvil, z in [0.0105, 0.0145])
        ("blade_root", (0.032, -0.002, 0.0125), (0.026, 0.010, 0.004), 0.0, STEEL_BRIGHT),
        ("blade_tip", (0.052, 0.001, 0.0125), (0.022, 0.008, 0.004), 12.0, STEEL_BRIGHT),
    ]


def upper_box_specs(open_deg: float = THETA_OPEN_DEG) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], float, tuple]]:
    """Boxes on the hinged upper body rotated by -open_deg about Z (spawn pose = THETA_OPEN_DEG)."""
    R = rot2(-open_deg)
    out = []
    for name, c, s, yaw, col in upper_box_specs_closed():
        xy = R @ np.array([c[0], c[1]], dtype=float)
        out.append((name, (float(xy[0]), float(xy[1]), c[2]), s, yaw - open_deg, col))
    return out


def tote_box_specs() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple]]:
    """Boxes forming the fixed green polypropylene garden tool tote in its local frame (origin at TOTE_CENTRE, TOP)."""
    ix = TOTE_IN_X[1] - TOTE_IN_X[0]  # 0.260
    iy = TOTE_IN_Y[1] - TOTE_IN_Y[0]  # 0.180
    ox = ix + 2 * TOTE_WALL  # 0.268
    oy = iy + 2 * TOTE_WALL  # 0.188
    wh = TOTE_RIM - TOTE_FLOOR  # 0.056
    wz = TOTE_FLOOR + wh / 2  # 0.032
    return [
        ("tote_floor", (0.0, 0.0, TOTE_FLOOR / 2), (ox, oy, TOTE_FLOOR), TOTE_GREEN),
        ("tote_wall_back", (-(ix / 2 + TOTE_WALL / 2), 0.0, wz), (TOTE_WALL, oy, wh), TOTE_GREEN),
        ("tote_wall_front", (+(ix / 2 + TOTE_WALL / 2), 0.0, wz), (TOTE_WALL, oy, wh), TOTE_GREEN),
        ("tote_wall_right", (0.0, -(iy / 2 + TOTE_WALL / 2), wz), (ix, TOTE_WALL, wh), TOTE_GREEN),
        ("tote_wall_left", (0.0, +(iy / 2 + TOTE_WALL / 2), wz), (ix, TOTE_WALL, wh), TOTE_GREEN),
    ]


# -- body sample points and exposed blade edge zone (world frame) --------------------------------------------------
def _box_corners(centre, size, yaw_deg) -> np.ndarray:
    cx, cy, cz = centre
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    local = np.array([
        [-hx, -hy, -hz], [-hx, -hy, +hz], [-hx, +hy, -hz], [-hx, +hy, +hz],
        [+hx, -hy, -hz], [+hx, -hy, +hz], [+hx, +hy, -hz], [+hx, +hy, +hz],
    ], dtype=float)
    R = rot2(yaw_deg)
    xy = local[:, :2] @ R.T + np.array([cx, cy])
    return np.column_stack([xy, local[:, 2] + cz])


def shears_world_points(origin: np.ndarray, rot: np.ndarray, open_deg: float, catch_m: float) -> np.ndarray:
    """All bounding-box corners of the shears (base + upper + catch) in world coordinates (N, 3)."""
    pts = []
    for _, c, s, y, _ in base_box_specs():
        pts.append(_box_corners(c, s, y))
    for _, c, s, y, _ in upper_box_specs(open_deg):
        pts.append(_box_corners(c, s, y))
    cc = (CATCH_ANCHOR[0], CATCH_ANCHOR[1] - catch_m, CATCH_ANCHOR[2] + CATCH_SIZE[2] / 2.0)
    pts.append(_box_corners(cc, CATCH_SIZE, 0.0))
    local = np.vstack(pts)
    return origin[None, :] + local @ rot.T


def blade_edge_world_box(origin: np.ndarray, rot: np.ndarray, open_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centre_world, size_m, rot_world) of the exposed bypass blade cutting-edge box.
    In the closed upper-body frame, the cutting edge runs along the +y (inner) side of the bypass blade
    from x = +0.022 to +0.065 m, y = -0.002 to +0.006 m, z = 0.010 to 0.015 m.
    Rotated by -open_deg about Z in the shears' base frame, then transformed into world frame.
    """
    c_closed = np.array([0.0435, 0.002, 0.0125], dtype=float)
    size = np.array([0.043, 0.008, 0.005], dtype=float)
    a = math.radians(-open_deg)
    ca, sa = math.cos(a), math.sin(a)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c_local = Rz @ c_closed
    c_world = origin + rot @ c_local
    rot_world = rot @ Rz
    return c_world, size, rot_world


def point_in_tote_interior(pts_world: np.ndarray) -> np.ndarray:
    """Boolean mask: True for world points inside the 26 x 18 cm tote interior AND below the 6.0 cm rim."""
    z_rel = pts_world[:, 2] - TOP
    return ((pts_world[:, 0] >= TOTE_IN_X[0] - EPS) & (pts_world[:, 0] <= TOTE_IN_X[1] + EPS)
            & (pts_world[:, 1] >= TOTE_IN_Y[0] - EPS) & (pts_world[:, 1] <= TOTE_IN_Y[1] + EPS)
            & (z_rel < TOTE_RIM - EPS))


def all_in_tote(pts_world: np.ndarray) -> bool:
    """True if every sampled corner of the shears is horizontally inside the tote interior, resting on/above
    the floor (within 2 mm) and below the 6.0 cm rim."""
    z_rel = pts_world[:, 2] - TOP
    return bool(
        np.all(pts_world[:, 0] >= TOTE_IN_X[0] - 0.002)
        and np.all(pts_world[:, 0] <= TOTE_IN_X[1] + 0.002)
        and np.all(pts_world[:, 1] >= TOTE_IN_Y[0] - 0.002)
        and np.all(pts_world[:, 1] <= TOTE_IN_Y[1] + 0.002)
        and float(np.min(z_rel)) >= TOTE_FLOOR - 0.003
        and float(np.max(z_rel)) <= TOTE_RIM + 0.002
    )
