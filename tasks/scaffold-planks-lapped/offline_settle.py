"""How the upper board of a lap comes to rest, for a kinematic checker without physics. The simulator never runs
this: ``task.py`` attaches it only when the offline stand-in for ``yam_arena`` (which has ``OFFLINE_STUB = True``)
is loaded instead of the real package.

A checker without physics lets go of an object by dropping it straight down, level, onto the highest thing under
it. The upper board of the lap would then lie level on the lower board's end with its outer end hanging 16 mm
above its putlog. No real board does that: its centre is 65 mm beyond the lower board's end, so it tips about that
end until its outer end comes down on the putlog, 5.4 degrees. Here each board's spec gets an
``offline_state(state)`` (the checker's hook for objects that build their own state offline): it returns the
board's state unchanged unless the board lies level on the other board with its centre beyond that board's end, in
which case it turns it about that end until its underside meets the highest putlog top (or the table) under its
outer part. Nothing else about the checker's states is changed, and the scorer is the same one the simulator runs.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
from scoring import HALF, long_axis, plan_corners, rotation, tilt_deg

LEVEL_DEG = 0.5  # both boards level (the checker keeps every object level)
PARALLEL_DEG = 10.0  # and lying the same way
ON_TOL = 0.0005  # the upper board's underside this close to the lower board's top: lying on it


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a rotation matrix."""
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12))
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _footprint(s) -> np.ndarray:
    """The four corners of a level board's outline seen from above, (4, 2)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    return np.array([(c + r[:, 0] * i * HALF[0] + r[:, 1] * j * HALF[1])[:2] for i, j in
                     ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def _inside(p, quad: np.ndarray) -> bool:
    """Is point ``p`` inside the convex quadrilateral ``quad`` (either winding)?"""
    def cross(a, b) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    sides = [cross(quad[(i + 1) % 4] - quad[i], np.asarray(p, dtype=float) - quad[i]) for i in range(4)]
    return all(v >= -1e-12 for v in sides) or all(v <= 1e-12 for v in sides)


def tipped(upper, lower):
    """The settled state of ``upper`` if it lies level on ``lower`` with its centre beyond ``lower``'s end;
    otherwise None."""
    if tilt_deg(upper) > LEVEL_DEG or tilt_deg(lower) > LEVEL_DEG:
        return None
    ua, la = long_axis(upper), long_axis(lower)
    cos_par = abs(float(ua[0] * la[0] + ua[1] * la[1]))
    if cos_par < math.cos(math.radians(PARALLEL_DEG)):
        return None
    if abs(float(upper.low[2]) - float(lower.high[2])) > ON_TOL:
        return None
    c = np.asarray(upper.pos, dtype=float)
    quad = _footprint(lower)
    if _inside(c[:2], quad):
        return None  # centre over the lower board: it stays where it is
    d = np.array([ua[0], ua[1]], dtype=float)
    d /= np.linalg.norm(d)
    if float(np.dot(np.asarray(lower.pos, dtype=float)[:2] - c[:2], d)) < 0.0:
        d = -d  # d points from the upper board's centre toward the lower board
    lat = np.array([-d[1], d[0]])
    proj = (quad - c[:2]) @ d
    near = float(proj.min())  # distance from the upper board's centre to the lower board's near end
    if not (0.0 < near < HALF[0]):
        return None  # not overlapping along its length
    lat_proj = (quad - c[:2]) @ lat
    if lat_proj.max() < -HALF[1] or lat_proj.min() > HALF[1]:
        return None  # side by side, not on it
    pivot = np.array([c[0] + near * d[0], c[1] + near * d[1], float(lower.high[2])])
    r_end = near + HALF[0]  # from the pivot to the upper board's outer end, along -d

    # The first support the outer part comes down on: each putlog top under it, or the table.
    angles = [math.atan2(pivot[2] - L.TOP, r_end)]
    for name, y in L.PUTLOGS:
        rect = np.array([(L.PUTLOG_X0, y - L.PUTLOG_HALF_W), (L.PUTLOG_X1, y - L.PUTLOG_HALF_W),
                         (L.PUTLOG_X1, y + L.PUTLOG_HALF_W), (L.PUTLOG_X0, y + L.PUTLOG_HALF_W)])
        r = (rect - pivot[:2]) @ (-d)
        w = (rect - c[:2]) @ lat
        if r.max() <= 0.0 or r.min() >= r_end or w.max() < -HALF[1] or w.min() > HALF[1]:
            continue
        drop = pivot[2] - L.PUTLOG_TOP
        if drop > 0.0:
            angles.append(math.atan2(drop, min(float(r.max()), r_end)))
    theta = min(angles)

    axis = np.array([d[1], -d[0], 0.0])  # horizontal, across the board
    rot = _axis_rotation(axis, theta)
    if float((rot @ np.array([-d[0], -d[1], 0.0]))[2]) > 0.0:  # turn the outer part down, not up
        rot = _axis_rotation(axis, -theta)
    centre = pivot + rot @ (c - pivot)
    r_new = rot @ rotation(upper.quat_xyzw)

    class _Pose:  # the turned board, for its corners
        pos, quat_xyzw = centre, _quat_from_matrix(r_new)

    corners = plan_corners(_Pose)
    return type(upper)(pos=centre, quat_xyzw=_Pose.quat_xyzw, lin_vel=np.asarray(upper.lin_vel, dtype=float),
                       ang_vel=np.asarray(upper.ang_vel, dtype=float), low=corners.min(axis=0),
                       high=corners.max(axis=0))


class Settler:
    """Keeps each board's latest state as the checker hands it over, and settles one on the other."""

    def __init__(self) -> None:
        self.raw: dict[str, object] = {}

    def state_for(self, name: str, state):
        self.raw[name] = state
        for other, o in self.raw.items():
            if other != name:
                t = tipped(state, o)
                if t is not None:
                    return t
        return state


def attach(specs) -> Settler:
    """Give each board spec an ``offline_state`` (the specs are frozen dataclasses, hence object.__setattr__)."""
    settler = Settler()
    for spec in specs:
        object.__setattr__(spec, "offline_state", lambda state, _n=spec.name: settler.state_for(_n, state))
    return settler
