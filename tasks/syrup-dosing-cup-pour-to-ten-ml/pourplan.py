"""The demo's geometry in one place, NumPy only (so the tests can run the whole pour through the scorer).

Right hand (bottle): a side grip, the approach axis pitched PITCH below horizontal toward +x (fingers close along y,
across the bottle's 49 mm), the pad-tip point 8 mm past the bottle's axis at height GRIP_Z over the bottle's bottom.
Tilting the bottle to pour is a roll of the hand about world x (the bottle's top turns toward +y, toward the cup).
Left hand (cup): a side grip, the approach axis pitched ROLL_L below horizontal toward -y (fingers close along x,
across the cup's 36 mm).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import scoring as S

PITCH = math.radians(64.0)  # right hand: approach this far from straight down (20 degrees below horizontal)
GRIP_Z = 0.022  # the bottle's axis point the pads grip, over its bottom
AXIS_PAST_TCP = 0.005  # that point is this far past the TCP along the approach axis (the tip point is 13 mm)
ROLL_L = math.radians(75.0)  # left hand: approach this far from straight down (15 degrees below horizontal)
CUP_GRIP_Z = 0.019  # the cup's wall point the pads grip, over its bottom (mid-height)
CUP_TIP_OFF = 0.0  # the pad-tip point this far toward the hand's side of the cup's axis
THETA_HOLD = math.radians(76.0)  # the bottle's tilt for the pour
Z_LIP = 0.135  # the lip's lowest point over the table at the pour (cup rim at 0.097 over it)
CUP_LIFT = 0.045  # the cup is held this far over the pan while the syrup goes in


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def axis_of(theta: float) -> np.ndarray:
    """The bottle's axis when it is tilted ``theta`` toward +y."""
    return rx(-theta) @ np.array([0.0, 0.0, 1.0])


def bottle_pose(a_point, theta: float):
    """(origin, rotation) of the bottle tilted ``theta`` toward +y about its grip point ``a_point`` (world)."""
    r = rx(-theta)
    return np.asarray(a_point, dtype=float) - r @ np.array([0.0, 0.0, GRIP_Z]), r


def right_approach(theta: float) -> np.ndarray:
    """The right hand's approach axis (unit) with the bottle tilted ``theta``."""
    return rx(-theta) @ np.array([math.sin(PITCH), 0.0, -math.cos(PITCH)])


def right_tcp(a_point, theta: float) -> np.ndarray:
    return np.asarray(a_point, dtype=float) - AXIS_PAST_TCP * right_approach(theta)


def left_approach() -> np.ndarray:
    return rx(-ROLL_L) @ np.array([0.0, 0.0, -1.0])  # (0, -sin, -cos): toward -y and down


def left_tcp(cup_origin) -> np.ndarray:
    """The left TCP gripping a cup whose bottom-face centre is ``cup_origin``."""
    tip = np.asarray(cup_origin, dtype=float) + np.array([0.0, -CUP_TIP_OFF, CUP_GRIP_Z])
    return tip - 0.013 * left_approach()


def pour_anchor(theta: float = THETA_HOLD, cup=(L.CUP_X, L.CUP_Y), z_lip: float = Z_LIP) -> np.ndarray:
    """Where the bottle's grip point is when the lip's lowest point, at tilt ``theta``, is over the cup's centre at
    height ``z_lip`` over the table."""
    origin, r = bottle_pose(np.zeros(3), theta)
    low = S.lip_low_point(origin, r[:, 2])  # relative to the grip point
    return np.array([cup[0] - low[0], cup[1] - low[1], L.TOP + z_lip - low[2]])


def grip_start() -> np.ndarray:
    """The bottle's grip point at the start."""
    return np.array([L.BOTTLE_X, L.BOTTLE_Y, L.TOP + GRIP_Z])


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Pose:
    """A part's state as the scorer reads it (the fields of mechkit.AssemblyState it uses)."""

    def __init__(self, origin, rot, size_low_high, vel=(0, 0, 0), spin=(0, 0, 0)):
        self.origin = np.asarray(origin, dtype=float)
        self.rot = np.asarray(rot, dtype=float)
        self.quat_xyzw = matrix_quat(self.rot)
        low, high = size_low_high
        self.low, self.high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        self.pos = (self.low + self.high) / 2
        self.lin_vel = np.asarray(vel, dtype=float)
        self.ang_vel = np.asarray(spin, dtype=float)


def matrix_quat(r: np.ndarray) -> np.ndarray:
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return q


def _corners(origin, rot, shapes):
    """World bounding box of cylinders/boxes given as ('cyl', radius, z0, z1) or ('box', hx, hy, z0, z1)."""
    pts = []
    for sh in shapes:
        if sh[0] == "cyl":
            _, r, z0, z1 = sh
            hx = hy = r
        else:
            _, hx, hy, z0, z1 = sh
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for z in (z0, z1):
                    pts.append(origin + rot @ np.array([sx, sy, z]))
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


def bottle_state(origin, rot, vel=(0, 0, 0), spin=(0, 0, 0)) -> Pose:
    """A bottle pose with a box-corner bounding box (a safe over-estimate for a cylinder; used by the tests and the
    offline demo run; the simulator's states have exact bounds)."""
    # radial extents of a tilted cylinder: use its true disc extents
    lows, highs = [], []
    for r, z0, z1 in ((L.BODY_R, 0.0, L.BODY_H), (L.NECK_R, L.BODY_H, L.NECK_TOP)):
        for z in (z0, z1):
            c = np.asarray(origin, dtype=float) + rot @ np.array([0.0, 0.0, z])
            # extent of a disc of radius r (normal = rot[:, 2]) along world x, y, z
            n = rot[:, 2]
            e = r * np.sqrt(np.maximum(1.0 - n ** 2, 0.0))
            lows.append(c - e)
            highs.append(c + e)
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    p = Pose(origin, rot, (low, high), vel, spin)
    return p


def cup_state(origin, rot, vel=(0, 0, 0), spin=(0, 0, 0)) -> Pose:
    lows, highs = [], []
    n = rot[:, 2]
    for z in (0.0, L.CUP_H):
        for sx in (-1, 1):
            for sy in (-1, 1):
                c = np.asarray(origin, dtype=float) + rot @ np.array([sx * L.CUP_OUTER / 2, sy * L.CUP_OUTER / 2, z])
                lows.append(c)
                highs.append(c)
    return Pose(origin, rot, (np.min(lows, axis=0), np.max(highs, axis=0)), vel, spin)
