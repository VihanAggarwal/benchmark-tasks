"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP; the cooktop is set into it (its glass top COOKTOP_TOP above the table). The pan's own
frame has its origin at the centre of its base, z up; its handle points along the pan frame's +x.

Sources (README.md): Frigidaire FFEC3025UB 30 in electric cooktop (30 5/8 x 21 3/8 in; elements: left front 6 in,
left rear 9 in, right rear 6 in, right front 6/9 in expandable); Tramontina Professional 10 in fry pan
(80114/535DS: 10 in across the top, 18 3/4 in overall with the handle, 2.05 lb). Element centres, the pan's base
diameter, wall angle and handle section, the saucepan, and every friction are estimates (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
INCH = 0.0254
TOP = 0.7475  # station table top
CONTROL_HZ = 20.0

# -- the cooktop (fixed; world) ---------------------------------------------------------------------------------------
COOK_W = 30.625 * INCH  # 0.778 m across (along y)
COOK_D = 21.375 * INCH  # 0.543 m deep (along x)
COOK_T = 0.006  # glass proud of the counter (the table top)
COOK_FRONT_X = 0.070  # its front edge, 70 mm in front of the arm bases' line
COOK_CY = 0.0
COOKTOP_TOP = COOK_T  # above the table top
GLASS = (0.06, 0.06, 0.07)


def cook_bounds():
    """(x0, x1, y0, y1) of the cooktop's outline (world)."""
    return COOK_FRONT_X, COOK_FRONT_X + COOK_D, COOK_CY - COOK_W / 2, COOK_CY + COOK_W / 2


# Elements: name -> (centre x, centre y, radius, state). Robot's left is +y (the cook's left facing the cooktop).
# Centres estimated from the maker's layout drawing proportions: front elements 150 mm behind the front edge, rear
# elements 380 mm behind it; left and right pairs about 200 mm either side of the centre line.
ELEMENTS = {
    "left_front": (COOK_FRONT_X + 0.150, +0.200, 3.0 * INCH, "pan"),  # 6 in: the frying pan's
    "left_rear": (COOK_FRONT_X + 0.380, +0.195, 4.5 * INCH, "hot"),  # 9 in: just switched off, still hot
    "right_front": (COOK_FRONT_X + 0.150, -0.195, 4.5 * INCH, "lit"),  # 6/9 in: lit, a saucepan simmering
    "right_rear": (COOK_FRONT_X + 0.380, -0.200, 3.0 * INCH, "off"),  # 6 in: off and cold
}
ELEMENT_OFF = (0.16, 0.16, 0.17)  # the printed ring of a cold element
ELEMENT_HOT = (0.85, 0.20, 0.06)  # glowing red-orange
HOT_STATES = ("hot", "lit")
HOT_MARGIN = 0.020  # m: the handle must keep 20 mm (in plan) outside a hot element's ring

# The saucepan simmering on the right front element: a lidded 2 qt stainless saucepan (fixed: it is not to be
# moved), body 180 mm across and 95 mm tall, lid knob on top.
SAUCEPAN_R = 0.090
SAUCEPAN_H = 0.095
STEEL = (0.72, 0.73, 0.75)

# -- the frying pan (pan frame) --------------------------------------------------------------------------------------
PAN_TOP_R = 5.0 * INCH  # 127 mm: 10 in across the top
PAN_BASE_R = 0.096  # base 192 mm across (est.)
PAN_DEPTH = 0.048  # rim 48 mm above the base (est.)
PAN_WALL = 0.003
PAN_FLOOR = 0.004
HANDLE_LEN = 18.75 * INCH - 2 * PAN_TOP_R  # 222 mm from the rim to the end
HANDLE_RISE_DEG = 12.0  # rising away from the pan (est.)
HANDLE_W = 0.024  # across (est.)
HANDLE_T = 0.016  # thick (est.)
HANDLE_Z0 = PAN_DEPTH - 0.010  # where the handle leaves the wall, 10 mm under the rim
GRIP_FROM_RIM = 0.090  # m along the handle where the demo grips it (the silicone sleeve's middle)
M_PAN = 2.05 * 0.4536  # 0.93 kg empty
M_OIL = 0.092  # 100 ml of cooking oil
ALUMINIUM = (0.20, 0.20, 0.22)  # black non-stick inside and out
SLEEVE = (0.55, 0.06, 0.06)  # red silicone handle sleeve
PAN_FRICTION = (0.30, 0.25)  # aluminium base on glass ceramic (est.)
WALL_SEGMENTS = 20

PAN_ELEMENT = "left_front"
PAN_START_YAW = math.radians(90.0)  # the handle points to the robot's left, out past the cooktop's left edge

# -- scoring -----------------------------------------------------------------------------------------------------------
HANDLE_SAMPLES = 12  # points along the handle's centre line, rim to end
EDGE_MARGIN = 0.0  # m: no part of the handle's centre line past any edge of the cooktop (seen from above)
CENTRE_OK = 0.030  # m: the pan's base centre within 30 mm of its element's centre at the end (on its burner)
CENTRE_FAIL = 0.060  # m: more than 60 mm at any moment: slid off its burner (latched 0)
TILT_FAIL_DEG = 10.0  # the pan tilted past 10 deg at any moment: hot oil spilled (latched 0)
LIFT_FAIL = 0.025  # m: the pan's base more than 25 mm above the glass at any moment (latched 0)
FLAT_DEG = 3.0  # at the end: resting flat on the glass
FLAT_Z = 0.004  # m: and its base within 4 mm of the glass
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
EPS = 1e-9
RESET_START_TOL = 0.004
RESET_JUMP = 0.05


def element(name: str):
    return ELEMENTS[name]


def pan_start() -> tuple[float, float, float]:
    x, y, _, _ = ELEMENTS[PAN_ELEMENT]
    return (x, y, TOP + COOKTOP_TOP)


def handle_dir() -> np.ndarray:
    """The handle's direction (pan frame), rim to end."""
    a = math.radians(HANDLE_RISE_DEG)
    return np.array([math.cos(a), 0.0, math.sin(a)])


def handle_root() -> np.ndarray:
    """Where the handle's centre line leaves the pan's wall (pan frame)."""
    r = PAN_BASE_R + (PAN_TOP_R - PAN_BASE_R) * HANDLE_Z0 / PAN_DEPTH
    return np.array([r, 0.0, HANDLE_Z0])


def handle_points() -> np.ndarray:
    """(HANDLE_SAMPLES, 3) points on the handle's centre line, pan frame."""
    s = np.linspace(0.0, HANDLE_LEN, HANDLE_SAMPLES)
    return handle_root()[None, :] + s[:, None] * handle_dir()[None, :]


def grip_point() -> np.ndarray:
    return handle_root() + GRIP_FROM_RIM * handle_dir()


def safe_plan(points_xy: np.ndarray) -> tuple[bool, bool, str | None]:
    """For the handle's centre-line points seen from above (world x, y): (inside the cooktop's outline, clear of
    every hot element, the first problem found)."""
    x0, x1, y0, y1 = cook_bounds()
    pts = np.asarray(points_xy, dtype=float)
    inside = bool(np.all((pts[:, 0] >= x0 + EDGE_MARGIN) & (pts[:, 0] <= x1 - EDGE_MARGIN)
                         & (pts[:, 1] >= y0 + EDGE_MARGIN) & (pts[:, 1] <= y1 - EDGE_MARGIN)))
    problem = None if inside else "handle past the cooktop's edge"
    clear = True
    for name, (cx, cy, r, state) in ELEMENTS.items():
        if state not in HOT_STATES:
            continue
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        if np.any(d < r + HOT_MARGIN):
            clear = False
            problem = problem or f"handle over the {state} {name.replace('_', ' ')} element"
    return inside, clear, problem


def handle_world_xy(origin_xy, yaw: float) -> np.ndarray:
    """The handle's centre-line points seen from above for a flat pan whose base centre is at ``origin_xy`` and
    whose handle points at ``yaw`` (world)."""
    c, s = math.cos(yaw), math.sin(yaw)
    p = handle_points()
    return np.stack([origin_xy[0] + c * p[:, 0] - s * p[:, 1], origin_xy[1] + s * p[:, 0] + c * p[:, 1]], axis=1)


def safe_yaws(step_deg: float = 1.0) -> list[float]:
    """Every handle direction (deg, world) that is safe with the pan centred on its element."""
    x, y, _ = pan_start()
    out = []
    for d in np.arange(-180.0, 180.0, step_deg):
        inside, clear, _ = safe_plan(handle_world_xy((x, y), math.radians(d)))
        if inside and clear:
            out.append(float(d))
    return out
