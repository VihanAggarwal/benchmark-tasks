"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A two-drawer tool chest (a riser or middle chest: one row of two drawers side by side) stands loose on rubber feet,
its front face 36.0 cm in front of the arm bases. Each drawer runs on full-extension slides (190 mm of travel to a hard
stop) and is loaded to 5.0 kg. The LEFT drawer (robot's left) stands fully open at the start; the RIGHT one is shut and
has an empty bay in its foam organiser. A screwdriver lies on the table to the right of the chest.

Chest frame (``CHEST`` in task.py): origin on the table at the middle of the chest's front face, x back into the chest,
y left, z up. The drawers are drawn in that frame SHUT, except the left drawer, which is drawn fully open (its joint
reads 0 there and -0.19 shut; see ``left_open``).

Statics: the chest tips forward about the front edges of its front feet (the tip line, 5 mm behind the front face)
when the moments of its parts about that line, weight times lever arm, sum to less than zero (``moment``). Masses are
spread evenly over each body's colliders, as PhysX does when only a body's mass is given; the centres of mass below are
computed from the shapes (``center_of_mass``), not assumed.
"""

from __future__ import annotations

import math

import numpy as np

import mechkit as mk

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the chest ---------------------------------------------------------------------------------------------------------
CHEST_X = 0.360  # the front face, world x
CHEST_D, CHEST_W, CHEST_H = 0.220, 0.600, 0.100  # depth (x), width (y), height including the feet (z)
WALL = 0.005  # collider thickness of the shell's sheets
FOOT = (0.020, 0.020, 0.008)  # rubber feet
FOOT_IN = 0.005  # each foot's outer edges 5 mm in from the shell's front (or back) face and its side
TIP_X = FOOT_IN  # the tip line in the chest frame: the front feet's front edges
BOTTOM_Z = (FOOT[2], FOOT[2] + WALL)  # the bottom sheet, 8 to 13 mm
TOP_Z = (CHEST_H - WALL, CHEST_H)  # the top sheet, 95 to 100 mm
SHELL_MASS = 3.0  # est: 0.8 mm sheet steel about 2.4 kg, plus feet and the slides' outer members

# -- the drawers -------------------------------------------------------------------------------------------------------
DRAWERS = ("left", "right")
DRAWER_Y = {"left": 0.150, "right": -0.150}  # centre lines
TRAVEL = 0.190  # full extension, to a hard stop
DRAWER_MASS = 5.0  # each, loaded: box about 1 kg, steel plate ballast and hand tools in a foam organiser
FRONT_T, FRONT_W = 0.015, 0.290
FRONT_Z = (0.016, 0.091)  # front panel, 75 mm tall
BOX_X = (FRONT_T, FRONT_T + 0.200)  # the box behind the front: 200 mm deep (x 15 to 215 mm, shut)
BOX_W = 0.280
SHEET = 0.003  # drawer sheet collider thickness
FLOOR_Z = (0.018, 0.021)
SIDE_TOP = 0.073  # side and back walls 55 mm above the floor's underside
BALLAST_Z = (0.021, 0.027)  # 6 mm steel plate over the whole floor
FOAM_Z = (0.027, 0.052)  # 25 mm foam organiser
INNER_X = (BOX_X[0], BOX_X[1] - SHEET)  # 15 to 212 mm
INNER_HALF_Y = BOX_W / 2 - SHEET  # 137 mm
# The right drawer's empty bay: 240 (y) by 70 (x) mm, down to the ballast plate. Its centre sits 77.5 mm behind the
# front panel's back face, which puts it in the middle of the opening when the drawer is out 170 mm.
BAY_X = (0.0575, 0.1275)
BAY_HALF_Y = 0.120
# Pull: a steel bar 120 x 12 x 12 mm on two 8 mm stems, 35 mm clear of the front, its centre 53 mm up.
BAR_LEN, BAR_SEC, BAR_STANDOFF, BAR_Z = 0.120, 0.012, 0.035, 0.053
BAR_FRONT = -(BAR_STANDOFF + BAR_SEC)  # chest frame x of the bar's front face, drawer shut: -47 mm
BAR_CENTRE = -(BAR_STANDOFF + BAR_SEC / 2)  # -41 mm
STEM, STEM_DY = 0.008, 0.050
START_OPEN = {"left": TRAVEL, "right": 0.0}

# -- the screwdriver ---------------------------------------------------------------------------------------------------
# Tool frame: origin on the handle's axis at its middle, local y along the tool with the blade toward -y, z up.
HANDLE_ACROSS, HANDLE_LEN = 0.032, 0.100  # anti-roll flats: an octagon 32 mm across its flats
SHAFT_D, SHAFT_LEN = 0.006, 0.107
BLADE = (0.006, 0.008, 0.0012)  # the slotted tip: 6 mm wide, 1.2 mm thick
TOOL_LEN = HANDLE_LEN + SHAFT_LEN + BLADE[1]  # 215 mm
TOOL_MASS = 0.12
TOOL_MID_Y = (HANDLE_LEN / 2 - (HANDLE_LEN / 2 + SHAFT_LEN + BLADE[1])) / 2  # -57.5 mm: the middle of its length
TOOL_START = (0.250, -0.370)  # handle centre on the table, the blade toward -y (away from the chest)
TOOL_R = HANDLE_ACROSS / 2

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
TIP_LIFT = 0.010  # TIP: any foot's bottom 10 mm or more above the table
DRAG = 0.030  # the chest moved more than 30 mm off its outline
SHUT = 0.010  # the left drawer 10 mm or less open
HALF = 0.095  # the left drawer pushed at least half way: 95 mm or less open
RIGHT_OPEN = 0.150  # the right drawer 150 mm or more open
CLEAR = 0.050  # both grippers' finger zones 5 cm or more from the chest and the screwdriver
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002  # the screwdriver moves under 2 mm and turns under 2 degrees over the last 1 s
STILL_TURN_DEG = 2.0
IN_MARGIN = 0.001
OFF_TABLE_DROP = 0.010
TABLE_X0, TABLE_X1 = 0.36 - 0.55, 0.36 + 0.55
TABLE_Y0, TABLE_Y1 = -0.65, 0.65
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
EPS = 1e-9

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
SHELL_RED = (0.62, 0.08, 0.07)
DRAWER_RED = (0.70, 0.10, 0.08)
CHROME = (0.80, 0.81, 0.83)
FOAM = (0.12, 0.12, 0.13)
BAY_GREY = (0.45, 0.46, 0.48)  # the ballast plate seen through the bay
TOOL_DARK = (0.20, 0.21, 0.23)
RUBBER = (0.05, 0.05, 0.05)
HANDLE_YELLOW = (0.90, 0.70, 0.08)


# -- shapes ------------------------------------------------------------------------------------------------------------
def _b(x0, x1, y0, y1, z0, z1, color=None):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def shell_shapes():
    """The chest's shell in the chest frame: four feet, bottom and top sheets, two sides, the back and a divider."""
    d, hw = CHEST_D, CHEST_W / 2
    fx, fy, fz = FOOT
    out = []
    for x0 in (FOOT_IN, d - FOOT_IN - fx):
        for s in (1, -1):
            y_out = s * (hw - FOOT_IN)
            out.append(_b(x0, x0 + fx, min(y_out, y_out - s * fy), max(y_out, y_out - s * fy), 0.0, fz, RUBBER))
    out.append(_b(0.0, d, -hw, hw, *BOTTOM_Z, SHELL_RED))
    out.append(_b(0.0, d, -hw, hw, *TOP_Z, SHELL_RED))
    for s in (1, -1):
        out.append(_b(0.0, d, min(s * hw, s * (hw - WALL)), max(s * hw, s * (hw - WALL)), BOTTOM_Z[1], TOP_Z[0],
                      SHELL_RED))
    out.append(_b(d - WALL, d, -hw + WALL, hw - WALL, BOTTOM_Z[1], TOP_Z[0], SHELL_RED))
    out.append(_b(0.0, d - WALL, -WALL / 2, WALL / 2, BOTTOM_Z[1], TOP_Z[0], SHELL_RED))
    return out


def drawer_shapes(name: str, dx: float = 0.0):
    """One drawer in the chest frame, shut, moved ``dx`` along x (the left one is drawn open: dx = -TRAVEL)."""
    c = DRAWER_Y[name]
    hw, iw = BOX_W / 2, INNER_HALF_Y

    def b(x0, x1, y0, y1, z0, z1, color=None):
        return _b(x0 + dx, x1 + dx, c + y0, c + y1, z0, z1, color)

    out = [
        b(0.0, FRONT_T, -FRONT_W / 2, FRONT_W / 2, *FRONT_Z, DRAWER_RED),  # front panel
        b(BOX_X[0], BOX_X[1], -hw, hw, *FLOOR_Z, DRAWER_RED),  # floor
        b(BOX_X[0], BOX_X[1], iw, hw, FLOOR_Z[1], SIDE_TOP, DRAWER_RED),  # side walls
        b(BOX_X[0], BOX_X[1], -hw, -iw, FLOOR_Z[1], SIDE_TOP, DRAWER_RED),
        b(INNER_X[1], BOX_X[1], -iw, iw, FLOOR_Z[1], SIDE_TOP, DRAWER_RED),  # back wall
        b(*INNER_X, -iw, iw, *BALLAST_Z, BAY_GREY),  # steel plate ballast
        b(BAR_FRONT, BAR_FRONT + BAR_SEC, -BAR_LEN / 2, BAR_LEN / 2, BAR_Z - BAR_SEC / 2, BAR_Z + BAR_SEC / 2, CHROME),
    ]
    for s in (1, -1):  # the bar's two stems
        out.append(b(BAR_FRONT + BAR_SEC, 0.0, s * STEM_DY - STEM / 2, s * STEM_DY + STEM / 2, BAR_Z - STEM / 2,
                     BAR_Z + STEM / 2, CHROME))
    if name == "right":  # the organiser with its empty bay
        out += [
            b(INNER_X[0], BAY_X[0], -iw, iw, *FOAM_Z, FOAM),
            b(BAY_X[1], INNER_X[1], -iw, iw, *FOAM_Z, FOAM),
            b(BAY_X[0], BAY_X[1], BAY_HALF_Y, iw, *FOAM_Z, FOAM),
            b(BAY_X[0], BAY_X[1], -iw, -BAY_HALF_Y, *FOAM_Z, FOAM),
        ]
    else:  # the open drawer's organiser is full
        out.append(b(*INNER_X, -iw, iw, *FOAM_Z, FOAM))
    # hand tools lying in the foam's back half (spanners and a socket rail), for the look; their mass is in the 5.0 kg
    for k, yy in enumerate((-0.08, -0.02, 0.04, 0.10)):
        out.append(b(0.150, 0.168, yy - 0.002, yy + 0.002 + 0.03 * (k % 2), FOAM_Z[1], FOAM_Z[1] + 0.004, CHROME))
    return out


def tool_shapes():
    """The screwdriver in its own frame: an octagonal handle (four turned boxes), the shaft and the blade."""
    flat = HANDLE_ACROSS * math.tan(math.pi / 8)  # 13.25 mm: the octagon's side
    out = [mk.box((0.0, 0.0, 0.0), (HANDLE_ACROSS, HANDLE_LEN, flat), HANDLE_YELLOW,
                  ypr=(0.0, math.radians(a), 0.0)) for a in (0.0, 45.0, 90.0, 135.0)]
    y_shaft = -(HANDLE_LEN / 2 + SHAFT_LEN / 2)
    out.append(mk.cylinder((0.0, y_shaft, 0.0), SHAFT_D / 2, SHAFT_LEN, "Y", CHROME))
    out.append(mk.box((0.0, -(HANDLE_LEN / 2 + SHAFT_LEN + BLADE[1] / 2), 0.0), BLADE, CHROME))
    return out


def tool_corners() -> np.ndarray:
    """The 8 corners of the box round the screwdriver in its own frame (the octagon's square, the blade's end)."""
    h = HANDLE_ACROSS / 2
    y0, y1 = -(HANDLE_LEN / 2 + SHAFT_LEN + BLADE[1]), HANDLE_LEN / 2
    return np.array([(x, y, z) for x in (-h, h) for y in (y0, y1) for z in (-h, h)], dtype=float)


# -- statics (pure NumPy, the same numbers task.json states) -----------------------------------------------------------
def com_x(shapes) -> float:
    vols = np.array([s.volume() for s in shapes])
    cx = np.array([s.center[0] for s in shapes])
    return float((vols * cx).sum() / vols.sum())


SHELL_COM_X = com_x(shell_shapes())  # about 113 mm behind the front face
DRAWER_COM_X = {n: com_x(drawer_shapes(n)) for n in DRAWERS}  # shut, chest frame


def moment(left_open: float, right_open: float, press: float = 0.0, press_x: float = 0.0, pull: float = 0.0,
           pull_z: float = BAR_Z, tool_x: float | None = None) -> float:
    """Sum of the moments about the tip line (kg m; weight times arm over g): positive stands, negative tips.
    ``press`` N pushing down at chest-frame x ``press_x``; ``pull`` N pulling a bar toward the robot at height
    ``pull_z``; ``tool_x`` the screwdriver's centre (chest frame) if it rides in the chest."""
    m = SHELL_MASS * (SHELL_COM_X - TIP_X)
    for name, q in (("left", left_open), ("right", right_open)):
        m += DRAWER_MASS * (DRAWER_COM_X[name] - q - TIP_X)
    m += press / G * (press_x - TIP_X)
    m -= pull / G * pull_z
    if tool_x is not None:
        m += TOOL_MASS * (tool_x - TIP_X)
    return float(m)


def holding_tipping(left_open: float, right_open: float) -> tuple[float, float]:
    """(holding, tipping) moments about the tip line (kg m), both positive."""
    parts = [SHELL_MASS * (SHELL_COM_X - TIP_X)]
    parts += [DRAWER_MASS * (DRAWER_COM_X[n] - q - TIP_X) for n, q in (("left", left_open), ("right", right_open))]
    hold = sum(p for p in parts if p > 0)
    tip = -sum(p for p in parts if p < 0)
    return float(hold), float(tip)


def tip_onset(other_open: float = TRAVEL, pull: float = 0.0) -> float:
    """How far (m) the right drawer can come out with the left one at ``other_open`` before the chest tips."""
    lo, hi = 0.0, TRAVEL
    if moment(other_open, hi, pull=pull) > 0:
        return math.inf
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if moment(other_open, mid, pull=pull) > 0 else (lo, mid)
    return lo


def rest_angle(left_open: float, right_open: float) -> float:
    """The forward tip (rad) at which the tipped chest comes to rest on the lowest edge of an open drawer's front."""
    best = math.inf
    for q in (left_open, right_open):
        if q <= 0.002:
            continue
        for x, z in ((-q, FRONT_Z[0]), (BAR_FRONT - q, BAR_Z - BAR_SEC / 2)):
            best = min(best, math.atan2(z, TIP_X - x))
    return best if math.isfinite(best) else math.radians(30.0)


def foot_corners() -> np.ndarray:
    """The bottom corners of the four feet's outline (chest frame)."""
    hw = CHEST_W / 2 - FOOT_IN
    return np.array([(x, y, 0.0) for x in (TIP_X, CHEST_D - FOOT_IN) for y in (hw, -hw)], dtype=float)
