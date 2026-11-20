"""Plane geometry of the hang, in the x-z plane (every part of the cabinet runs the whole way along y, and the door is
only ever turned about y or lifted, so a door's section is a set of rectangles). NumPy only.

Used by task.py (the parts' shapes, the demo's door keyframes, the offline stand-in for the physics) and the tests.
A door pose here is (bx, bz, theta): the centre of the door's underside, and the lean of its top toward +x (rad).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

H = L.DOOR_H
HALF_T = L.DOOR_T / 2


def up_vec(theta: float) -> np.ndarray:
    """The door's up axis in the x-z plane: leaning toward +x by theta."""
    return np.array([math.sin(theta), math.cos(theta)])


def back_vec(theta: float) -> np.ndarray:
    """The door's thickness axis, toward its rear face (away from the robot)."""
    return np.array([math.cos(theta), -math.sin(theta)])


def to_world(b, theta: float, lx: float, lz: float) -> np.ndarray:
    """A point (lx across the thickness, lz up from the underside) of a door at pose (b, theta), in the plane."""
    return np.asarray(b, dtype=float) + lx * back_vec(theta) + lz * up_vec(theta)


def door_rects(front: bool = True) -> list[tuple[float, float, float, float]]:
    """The door's section as rectangles in its own frame (lx0, lx1, lz0, lz1): the panel above the foot groove, the
    two foot strips, and (front door only) the pull's bar and posts."""
    g = L.FOOT_GROOVE_W / 2
    out = [(-HALF_T, HALF_T, L.FOOT_GROOVE_D, H),
           (-HALF_T, -g, 0.0, L.FOOT_GROOVE_D),
           (g, HALF_T, 0.0, L.FOOT_GROOVE_D)]
    if front:
        bx, bz = L.PULL_X, L.PULL_H
        out.append((bx - L.PULL_SIZE[0] / 2, bx + L.PULL_SIZE[0] / 2, bz - L.PULL_SIZE[2] / 2, bz + L.PULL_SIZE[2] / 2))
        out.append((-HALF_T - L.PULL_OFF, -HALF_T, bz - L.POST[2] / 2, bz + L.POST[2] / 2))
    return out


def door_polys(b, theta: float, front: bool = True) -> list[np.ndarray]:
    return [np.array([to_world(b, theta, x, z) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))])
            for x0, x1, z0, z1 in door_rects(front)]


def _rect(x0, x1, z0, z1) -> np.ndarray:
    return np.array([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], dtype=float)


def cabinet_polys(rear_door: bool = True) -> list[np.ndarray]:
    """The fixed parts in the plane (and the rear door, hung, when asked): table, floor, rails, track walls, top board,
    back panel."""
    f, r = L.FRONT_GROOVE, L.REAR_GROOVE
    out = [
        _rect(-1.0, 2.0, -1.0, 0.0),  # the table
        _rect(L.CAB_X0, L.CAB_X1, 0.0, L.FLOOR_TOP),
        _rect(L.X_F - L.RAIL_W / 2, L.X_F + L.RAIL_W / 2, L.FLOOR_TOP, L.RAIL_TOP),
        _rect(L.X_R - L.RAIL_W / 2, L.X_R + L.RAIL_W / 2, L.FLOOR_TOP, L.RAIL_TOP),
        _rect(L.TRACK_X0, f[0], L.TRACK_BOTTOM, L.ROOF),
        _rect(f[1], r[0], L.TRACK_BOTTOM, L.ROOF),
        _rect(r[1], L.TRACK_X1, L.TRACK_BOTTOM, L.ROOF),
        _rect(L.CAB_X0, L.CAB_X1, L.ROOF, L.TOP_PANEL_TOP),
        _rect(L.CAB_X1 - L.BOARD, L.CAB_X1, L.FLOOR_TOP, L.ROOF),
    ]
    if rear_door:
        out += door_polys((L.REAR_X, L.FLOOR_TOP), 0.0, front=False)
    return out


CABINET = cabinet_polys()


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Separating-axis gap between two convex polygons: the largest gap along an edge normal (positive: apart, a lower
    bound on the true distance; negative: overlapping by that depth along the best axis)."""
    best = -math.inf
    for poly, other in ((a, b), (b, a)):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            ax = np.array([e[1], -e[0]])
            ln = float(np.hypot(*ax))
            if ln < 1e-12:
                continue
            ax = ax / ln
            pa, pb = poly @ ax, other @ ax
            gap = max(pa.min() - pb.max(), pb.min() - pa.max())
            best = max(best, float(gap))
    return best


def clearance(b, theta: float, front: bool = True, solids=None) -> float:
    """The smallest separation between the door (at pose b, theta) and the fixed solids (the cabinet, the rear door
    and the table by default): negative when it is in collision."""
    solids = CABINET if solids is None else solids
    worst = math.inf
    for p in door_polys(b, theta, front):
        for s in solids:
            worst = min(worst, separation(p, s))
    return float(worst)


# -- the pull and the hand --------------------------------------------------------------------------------------
def tcp_for(b, theta: float) -> np.ndarray:
    """Where the TCP is when the gripper's pads hold the pull's bar (x, z in the plane)."""
    return to_world(b, theta, L.PULL_X, L.PULL_H)


def pose_for_tcp(tcp, theta: float) -> np.ndarray:
    """The door's underside centre for a TCP at the bar and a lean theta."""
    return np.asarray(tcp, dtype=float) - (L.PULL_X * back_vec(theta) + L.PULL_H * up_vec(theta))


def pitch_for(theta: float) -> float:
    """tiltgrip pitch of the grip that holds a door leaning theta: the approach axis (toward the door) is
    (cos theta, -sin theta), and tiltgrip's approach is (sin p, -cos p): p = pi/2 - theta."""
    return math.pi / 2 - theta


def lean_from_approach(a) -> float:
    """The lean of a door held by a gripper whose approach axis (x, z) is ``a``: atan2(-a_z, a_x)."""
    return math.atan2(-float(a[1]), float(a[0]))


def top_for(t, theta: float) -> np.ndarray:
    """The underside centre of a door whose top-edge centre is at ``t``."""
    return np.asarray(t, dtype=float) - H * up_vec(theta)


def top_corners_x(b, theta: float) -> tuple[float, float, float]:
    """(front-top corner x, rear-top corner x, the top-edge z) of the door in the plane."""
    f = to_world(b, theta, -HALF_T, H)
    r = to_world(b, theta, HALF_T, H)
    return float(f[0]), float(r[0]), float(0.5 * (f[1] + r[1]))
