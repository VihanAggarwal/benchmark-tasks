"""The three parts as mechkit assemblies (shapes, joints, physics) and the geometry the scorer and the offline model
use: points on the thermometer, the armpit sensor disc, the doll's arm. NumPy and mechkit only (no simulator).

Doll (loose, 2.3 kg): one rigid body (head, torso, diaper, legs, its left arm, the sensor film) and its right arm as
one rigid link on the shoulder joint, a revolute about the doll's left-right axis (world x) with friction and no
spring: 0 is the arm lying along the body on the joint's stop, positive lifts it up off its side. Thermometer (loose,
15 g): its body, the sprung On/Off button (a slider, 0.8 mm) and the display's face (a slider moved by the device
rule: blank under the window when off, the lit face showing Lo when on). Mat (loose).
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402

DOLL, THERMO, MAT = "baby_doll", "thermometer", "changing_mat"
PARTS = (DOLL, THERMO, MAT)
ARM_LINK = "shoulder"


# == the doll ========================================================================================================
def d(x, y, z):
    """World millimetres (x, y) and height over the mat top (z) -> the doll's base frame (m)."""
    return ((x - L.DOLL_X * 1000.0) * L.MM, y * L.MM, z * L.MM)


def dbox(x0, x1, y0, y1, z0, z1, color=L.SKIN, **turn):
    return mk.box(d((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), ((x1 - x0) * L.MM, (y1 - y0) * L.MM,
                                                                  (z1 - z0) * L.MM), color, **turn)


def dcyl(x, y, z, r, h, axis, color=L.SKIN, **turn):
    return mk.cylinder(d(x, y, z), r * L.MM, h * L.MM, axis, color, **turn)


def _head():
    """A ball (radius 60 mm, about a quarter of the body's length, as on a newborn) from twelve stacked slices, lying
    on the back of the head; no drawn face."""
    cx, cy = L.HEAD_C
    n = 12
    dz = 2 * L.HEAD_R / n
    out = []
    for k in range(n):
        zm = (k + 0.5) * dz
        r = math.sqrt(max(L.HEAD_R ** 2 - (zm - L.HEAD_R) ** 2, 100.0))
        out.append(dcyl(cx, cy, zm, r, dz, "Z"))
    return out


def _torso():
    x0, x1 = L.TORSO_X
    y0, y1 = L.TORSO_Y
    h, top = L.TORSO_SIDE_H, L.TORSO_DEPTH
    r = top - h  # 40: the chest's rounded upper half
    return [dbox(x0, x1, y0, y1, 0.0, h),
            dcyl(x0 + r, (y0 + y1) / 2, h, r, y1 - y0, "Y"),
            dcyl(x1 - r, (y0 + y1) / 2, h, r, y1 - y0, "Y"),
            dbox(x0 + r, x1 - r, y0, y1, h, top),
            dcyl(300.0, sum(L.NECK_Y) / 2, 40.0, 32.0, L.NECK_Y[1] - L.NECK_Y[0], "Y")]


def _diaper():
    """The diaper's box and its rounded top; the rounded part's underside sits exactly on the base plane (GPU 19 Nov:
    at radius 42 it stuck 2 mm under the doll's back, so the doll lay on that line, 2.4 mm high, and rocked; the arm
    resting on the thermometer lifted its shoulder and rolled it about 3 deg)."""
    y0, y1 = L.DIAPER_Y
    return [dbox(L.TORSO_X[0], L.TORSO_X[1], y0, y1, 0.0, 48.0, L.DIAPER),
            dcyl(300.0, (y0 + y1) / 2, 40.0, 40.0, y1 - y0, "Y", L.DIAPER)]


def _legs():
    """Short, chubby newborn legs as capsules: thigh (radius 22) to the knee, shin (19) to the ankle, a rounded knee
    and a rounded foot, toes up."""
    out = []
    y0, y1 = L.LEG_Y  # ankle, hip
    for x in L.LEG_X:
        out.append(dcyl(x, (L.KNEE_Y + y1) / 2, L.LEG_R, L.LEG_R, y1 - L.KNEE_Y, "Y"))
        out.append(dcyl(x, L.KNEE_Y, L.LEG_R, L.LEG_R - 1.0, 2 * L.LEG_R - 2.0, "Z"))  # the knee, round from above
        out.append(dcyl(x, (y0 + L.KNEE_Y) / 2, L.SHIN_R, L.SHIN_R, L.KNEE_Y - y0, "Y"))
        out.append(dcyl(x, L.FOOT_C_Y, L.FOOT_H / 2, L.FOOT_R, L.FOOT_H, "Z"))  # the foot, round from above
        out.append(dcyl(x, L.FOOT_C_Y, L.FOOT_H + 3.0, L.FOOT_R - 5.0, 6.0, "Z"))  # its rounded top (the toes)
        out.append(dcyl(x, L.FOOT_C_Y, L.FOOT_R, L.FOOT_R, 2 * L.FOOT_R, "Y"))  # heel and sole, round from the side
    return out


def _arm_shapes(mirror: bool = False):
    """The doll's arm (shoulder disc, upper arm, forearm angled 8 deg out at the elbow, hand), drawn lying along the
    body; ``mirror``: the doll's left arm (the other side of the midline)."""
    sx = -1.0 if mirror else 1.0

    def X(x):
        return 2 * L.DOLL_X * 1000.0 - x if mirror else x

    a = math.radians(L.FOREARM_YAW_DEG) * sx
    u = np.array([math.sin(a), -math.cos(a)])  # down the forearm, toward the hand
    e = np.array(L.ELBOW)
    fore_c = e + u * L.FOREARM_LEN / 2
    hand_end = e + u * L.FOREARM_LEN
    hand_c = hand_end + u * L.HAND_LEN / 2
    g = L.ARM_GAP_UNDER
    turn = {"ypr": (a, 0.0, 0.0)}
    disc_x0, disc_x1 = (X(v) for v in L.SHOULDER_DISC_X)
    out = [
        dcyl((disc_x0 + disc_x1) / 2, L.SHOULDER_Y, L.SHOULDER_Z, L.UPPER_ARM_R, abs(disc_x1 - disc_x0), "X"),
        dcyl(X(L.UPPER_ARM_X), sum(L.UPPER_ARM_Y) / 2, L.SHOULDER_Z, L.UPPER_ARM_R,
             L.UPPER_ARM_Y[1] - L.UPPER_ARM_Y[0], "Y"),
        dcyl(X(e[0]), e[1], g + 16.0, 16.0, 32.0, "X"),
        dcyl(X(fore_c[0]), fore_c[1], g + L.FOREARM_R, L.FOREARM_R, L.FOREARM_LEN, "Y", **turn),
        dcyl(X(hand_c[0]), hand_c[1], g + L.HAND_R, L.HAND_R, L.HAND_LEN, "Y", **turn),
    ]
    if mirror:  # the doll's left arm (look only, on the far side): a rounded hand and elbow, round from above
        tip = hand_end + u * L.HAND_LEN
        out += [dcyl(X(tip[0]), tip[1], g + L.HAND_R, L.HAND_R, 2 * L.HAND_R, "Z"),
                dcyl(X(e[0]), e[1], g + 16.0, 16.0, 32.0, "Z")]
    return out


def hand_centre_world_mm() -> np.ndarray:
    """The hand's centre at rest (world mm, z over the TABLE): where the left gripper pinches the arm."""
    a = math.radians(L.FOREARM_YAW_DEG)
    u = np.array([math.sin(a), -math.cos(a)])
    c = np.array(L.ELBOW) + u * (L.FOREARM_LEN + L.HAND_LEN / 2)
    return np.array([c[0], c[1], L.MAT_H * 1000.0 + L.ARM_GAP_UNDER + L.HAND_R])


def _fsr():
    x, y, z = L.FSR_C
    disc = dcyl(x - L.FSR_T / 2, y, z, L.FSR_R, L.FSR_T, "X", L.FSR_TAN)
    tail = dbox(x - L.FSR_T, x, y - 3.5, y + 3.5, 0.0, z - L.FSR_R + 0.5, L.FSR_TAN)
    return [disc, tail]


class Doll(mk.Assembly):
    """Offline (tools/offline/check_task.py) the checker draws the torso and head only; the arm is the offline
    model's (offline_model.py)."""

    def offline_box(self):
        x0, x1 = L.TORSO_X
        y0, y1 = L.TORSO_Y[0], L.HEAD_C[1] + L.HEAD_R
        c = ((x0 + x1) / 2 * L.MM, (y0 + y1) / 2 * L.MM, L.TOP + L.MAT_H + L.TORSO_DEPTH / 2 * L.MM)
        return c, ((x1 - x0) * L.MM, (y1 - y0) * L.MM, L.TORSO_DEPTH * L.MM), 0.0

    def offline_state(self, state):
        return self.rest_state()


SHOULDER = mk.hinge(ARM_LINK, "-X", d(L.UPPER_ARM_X, L.SHOULDER_Y, L.SHOULDER_Z), _arm_shapes(),
                    upper=math.radians(L.SHOULDER_RANGE_DEG), mass=L.ARM_MASS, friction=L.SHOULDER_FRICTION,
                    damping=L.SHOULDER_DAMPING, color=L.SKIN)
DOLL_SHAPES = tuple(_torso() + _head() + _diaper() + _legs() + _arm_shapes(mirror=True) + _fsr())
DOLL_ORIGIN = (L.DOLL_X, 0.0, L.TOP + L.MAT_H)
DOLL_SPEC = Doll(name=DOLL, pos=DOLL_ORIGIN, color=L.SKIN, mass=L.DOLL_MASS, yaw=0.0, shapes=DOLL_SHAPES,
                 joints=(SHOULDER,), friction=L.DOLL_FRICTION)
DOLL_SPEC._check()


# == the thermometer ==================================================================================================
def t(x, y, z):
    return (x * L.MM, y * L.MM, z * L.MM)


def tbox(x0, x1, y0, y1, z0, z1, color=L.TH_WHITE):
    return mk.box(t((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), ((x1 - x0) * L.MM, (y1 - y0) * L.MM,
                                                                  (z1 - z0) * L.MM), color)


HW = L.TH_W / 2
_R_TAIL = HW


def _thermo_body():
    x0 = _R_TAIL
    x1 = L.TH_BODY_END
    lo = L.TH_LOWER_H
    plate = mk.plate_with_holes(t((x0 + x1) / 2, 0.0, (lo + L.TH_H) / 2), ((x1 - x0) * L.MM, L.TH_W * L.MM),
                                [((L.BUTTON_X - (x0 + x1) / 2) * L.MM, 0.0, 11.0 * L.MM, 11.0 * L.MM),
                                 ((sum(L.WINDOW_X) / 2 - (x0 + x1) / 2) * L.MM, 0.0,
                                  (L.WINDOW_X[1] - L.WINDOW_X[0]) * L.MM, L.WINDOW_W * L.MM)],
                                (L.TH_H - lo) * L.MM, L.TH_WHITE)
    a = L.TH_AXIS_Z
    return [mk.cylinder(t(_R_TAIL, 0.0, L.TH_H / 2), _R_TAIL * L.MM, L.TH_H * L.MM, "Z", L.TH_WHITE),
            tbox(x0, x1, -HW, HW, 0.0, lo),
            *plate,
            tbox(84.0, 94.0, -9.0, 9.0, a - 5.0, a + 5.0),
            tbox(94.0, 104.0, -6.5, 6.5, a - 3.75, a + 3.75),
            tbox(104.0, 110.0, -4.0, 4.0, a - 3.0, a + 3.0),
            mk.cylinder(t(sum(L.TIP_X) / 2, 0.0, a), L.TIP_R * L.MM, (L.TIP_X[1] - L.TIP_X[0]) * L.MM, "X",
                        L.TIP_METAL)]


def _display_shapes():
    """The display's face under the window, drawn OFF: the blank half under the window, the lit half (green, with
    Lo in black) under the body toward the tip; switched on it slides 22 mm toward the tail."""
    w0, w1 = L.WINDOW_X
    span = w1 - w0
    hw = L.WINDOW_W / 2
    z0, z1 = L.TH_LOWER_H + 0.3, L.TH_LOWER_H + 1.5
    lit0 = w1
    ink = []
    # "L": an upright stroke and its foot; "o": a small square ring (seven-segment style), 6 mm tall
    lx = lit0 + 6.0
    ink.append(tbox(lx, lx + 1.0, -3.0, 3.0, z1, z1 + 0.2, L.LCD_INK))
    ink.append(tbox(lx, lx + 4.0, -3.0, -2.0, z1, z1 + 0.2, L.LCD_INK))
    ox = lx + 6.0
    ink += [tbox(ox, ox + 4.0, -3.0, -2.0, z1, z1 + 0.2, L.LCD_INK),
            tbox(ox, ox + 4.0, -0.5, 0.5, z1, z1 + 0.2, L.LCD_INK),
            tbox(ox, ox + 1.0, -3.0, 0.5, z1, z1 + 0.2, L.LCD_INK),
            tbox(ox + 3.0, ox + 4.0, -3.0, 0.5, z1, z1 + 0.2, L.LCD_INK)]
    return [tbox(w0, w1, -hw, hw, z0, z1, L.LCD_OFF), tbox(lit0, lit0 + span, -hw, hw, z0, z1, L.LCD_LIT), *ink]


BUTTON = mk.slider("button", "-Z", [mk.cylinder(t(L.BUTTON_X, 0.0, (L.TH_LOWER_H + L.TH_H + L.BUTTON_PROUD) / 2),
                                                L.BUTTON_R * L.MM,
                                                (L.TH_H + L.BUTTON_PROUD - L.TH_LOWER_H) * L.MM, "Z", L.TH_BUTTON)],
                   upper=L.BUTTON_TRAVEL, mass=L.BUTTON_MASS, spring=L.BUTTON_K, damping=L.BUTTON_DAMP,
                   color=L.TH_BUTTON)
DISPLAY = mk.slider("display", "-X", _display_shapes(), upper=L.DISPLAY_TRAVEL, mass=L.DISPLAY_MASS,
                    spring=L.DISPLAY_K, damping=L.DISPLAY_DAMP, color=L.LCD_OFF)
TH_ORIGIN = (L.TH_START[0] * L.MM, L.TH_START[1] * L.MM, L.TOP + L.MAT_H)
THERMO_SPEC = mk.assembly(THERMO, origin=TH_ORIGIN, shapes=_thermo_body(), mass=L.TH_MASS,
                          joints=[BUTTON, DISPLAY], color=L.TH_WHITE, friction=L.TH_FRICTION)

# == the mat ==========================================================================================================
MAT_ORIGIN = (sum(L.MAT_X) / 2, sum(L.MAT_Y) / 2, L.TOP)
MAT_SPEC = mk.assembly(MAT, origin=MAT_ORIGIN, mass=L.MAT_MASS, friction=L.MAT_FRICTION, color=L.MAT_BLUE,
                       shapes=[mk.box((0.0, 0.0, L.MAT_H / 2), (L.MAT_X[1] - L.MAT_X[0], L.MAT_Y[1] - L.MAT_Y[0],
                                                                L.MAT_H), L.MAT_BLUE)])

SPECS = {DOLL: DOLL_SPEC, THERMO: THERMO_SPEC, MAT: MAT_SPEC}


# == geometry for the scorer and the offline model ===================================================================
def _surface_points(shapes, step: float = 0.002) -> np.ndarray:
    """Points on the surfaces of boxes and cylinders (their own frames), about ``step`` apart."""
    pts = []
    for s in shapes:
        rot = s.rotation()
        c = np.asarray(s.center, dtype=float)
        if s.kind == "box":
            size = np.asarray(s.size, dtype=float)
            axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / step)) + 1)) for h in size / 2]
            for k in range(3):
                for sign in (-1.0, 1.0):
                    g = np.meshgrid(*[axes[i] if i != k else np.array([sign * size[k] / 2]) for i in range(3)],
                                    indexing="ij")
                    pts.append(np.stack([v.ravel() for v in g], axis=1) @ rot.T + c)
        else:
            k = "XYZ".index(s.axis)
            r, h = s.radius, s.height
            n_a = max(8, int(math.ceil(2 * math.pi * r / step)))
            ang = np.linspace(0.0, 2 * math.pi, n_a, endpoint=False)
            along = np.linspace(-h / 2, h / 2, max(2, int(math.ceil(h / step)) + 1))
            loc = []
            for a_ in along:
                for an in ang:
                    loc.append((a_, r * math.cos(an), r * math.sin(an)))
            for sign in (-1.0, 1.0):
                for rr in np.linspace(0.0, r, max(2, int(math.ceil(r / step)) + 1)):
                    for an in ang:
                        loc.append((sign * h / 2, rr * math.cos(an), rr * math.sin(an)))
            loc = np.asarray(loc)
            p = np.zeros_like(loc)
            if k == 0:
                p[:, 0], p[:, 1], p[:, 2] = loc[:, 0], loc[:, 1], loc[:, 2]
            elif k == 1:
                p[:, 1], p[:, 2], p[:, 0] = loc[:, 0], loc[:, 1], loc[:, 2]
            else:
                p[:, 2], p[:, 0], p[:, 1] = loc[:, 0], loc[:, 1], loc[:, 2]
            pts.append(p @ rot.T + c)
    return np.concatenate(pts, axis=0)


# Points on the thermometer's body (its base shapes; the button and the display face are inside or flush) and on the
# metal tip (its front 8 mm).
TH_POINTS = _surface_points(THERMO_SPEC.shapes, 0.002)
TIP_POINTS = _surface_points([THERMO_SPEC.shapes[-1]], 0.0007)
TH_LOWEST = 0.0  # its underside is the base frame's z 0


def th_world(state, local) -> np.ndarray:
    return state.to_world(local)


def fsr_distance(doll_state, points) -> np.ndarray:
    """Distance (m) from world points to the sensor's active disc (a 0.46 mm thick, 14.68 mm cylinder on the torso
    side), in the doll's frame (it moves with the doll)."""
    loc = doll_state.to_local(np.atleast_2d(points))
    c = np.array(d(L.FSR_C[0] - L.FSR_T / 2, L.FSR_C[1], L.FSR_C[2]))
    q = loc - c
    radial = np.hypot(q[:, 1], q[:, 2]) - L.FSR_R * L.MM
    along = np.abs(q[:, 0]) - L.FSR_T * L.MM / 2
    return np.hypot(np.maximum(radial, 0.0), np.maximum(along, 0.0))


def tip_to_fsr(doll_state, th_state) -> float:
    return float(np.min(fsr_distance(doll_state, th_state.to_world(TIP_POINTS))))


def thermo_to_arm(doll_state, th_state) -> float:
    """The gap (m) between the thermometer's body and the doll's right arm (0: touching or overlapping)."""
    return float(np.min(doll_state.distance(th_state.to_world(TH_POINTS), bodies=(ARM_LINK,))))


def thermo_to_torso(doll_state, th_state) -> float:
    return float(np.min(doll_state.distance(th_state.to_world(TH_POINTS), bodies=("base",))))


def doll_state(theta: float = 0.0, base_pose=None, source: str = "offline"):
    pose = base_pose if base_pose is not None else (*DOLL_ORIGIN, 0.0, 0.0, 0.0, 1.0)
    return mk.state_from_poses(DOLL_SPEC, pose, joints={ARM_LINK: theta}, source=source)


def thermo_state(origin, yaw: float = 0.0, button: float = 0.0, display: float = 0.0, source: str = "offline"):
    return mk.state_from_poses(THERMO_SPEC, (*origin, *mk.yaw_quat(yaw)), joints={"button": button,
                                                                                  "display": display},
                               source=source)


def placed_origin() -> tuple[float, float, float]:
    """The thermometer's origin with its tip end on the sensor film, along x at the sensor's y, lying on the mat."""
    return (L.PLACE_TAIL_X * L.MM, L.PLACE_Y * L.MM, L.TOP + L.MAT_H)


def support_angle(th_state, base_pose=None, hi: float = math.radians(80.0), clear: float = 0.0005) -> float:
    """The lowest shoulder angle (rad) at which the doll's arm is ``clear`` off the thermometer: where the arm comes to
    rest on it when let down (0 if it reaches its stop). The demo plans with 0.5 mm; the offline model lays the arm
    0.1 mm off it, as resting surfaces read on the GPU."""
    if thermo_to_arm(doll_state(0.0, base_pose), th_state) > clear:
        return 0.0
    lo = 0.0
    if thermo_to_arm(doll_state(hi, base_pose), th_state) <= clear:
        return hi
    for _ in range(30):
        mid = (lo + hi) / 2
        if thermo_to_arm(doll_state(mid, base_pose), th_state) > clear:
            hi = mid
        else:
            lo = mid
    return hi


AXIS_WORLD = np.array([L.UPPER_ARM_X * L.MM, L.SHOULDER_Y * L.MM, L.TOP + L.MAT_H + L.SHOULDER_Z * L.MM])


def rotate_about_axis(p, theta: float) -> np.ndarray:
    """A world point riding on the doll's arm, at rest at ``p``, with the shoulder at ``theta`` (the doll at its start
    pose): a turn about -x through the shoulder axis."""
    q = np.asarray(p, dtype=float) - AXIS_WORLD
    c, s = math.cos(theta), math.sin(theta)
    return AXIS_WORLD + np.array([q[0], q[1] * c + q[2] * s, -q[1] * s + q[2] * c])


HAND_C = np.array([hand_centre_world_mm()[0] * L.MM, hand_centre_world_mm()[1] * L.MM,
                   L.TOP + hand_centre_world_mm()[2] * L.MM])


def hand_at(theta: float) -> np.ndarray:
    return rotate_about_axis(HAND_C, theta)


def angle_of(point) -> float:
    """The shoulder angle (rad) that puts the hand's centre nearest ``point`` (top of the turn: y, z only)."""
    q = np.asarray(point, dtype=float) - AXIS_WORLD
    q0 = HAND_C - AXIS_WORLD
    return math.atan2(q0[1] * q[2] - q0[2] * q[1], q0[1] * q[1] + q0[2] * q[2]) * -1.0


HAND_LOCAL = HAND_C - np.asarray(DOLL_ORIGIN)  # the hand's centre in the doll's base frame at rest (on the arm link)
