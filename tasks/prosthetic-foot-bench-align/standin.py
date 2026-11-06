"""An offline stand-in for how the foot comes to rest, for tools/offline/check_task.py only.

The offline checker simulates nothing. It draws and grips the foot as one column (the adapter's 46 x 46 mm outline,
from the sole to the adapter's top: ``task.Foot.offline_box``), carries it level and turned with the gripper, and
when the gripper opens drops it straight down onto the highest thing under that column: the plate, never the heel
block, which lies under the heel, not under the adapter. ``task.py`` therefore hands the scorer this module's pose
instead, and only when mechkit marks the foot's state as offline (``source == "offline"``); in the simulator the
scorer reads the real foot and this module does nothing.

What it models, from the foot's pose (level, as carried or dropped) and the block's pose alone:
- while the foot's sole is in the air (more than 1 mm above the table, the plate's top and the block's top), it is
  carried: the pose is left as it is;
- resting, the sole is a flat 220 mm line along the foot that settles on the supports under its outline: the plate
  (or the table, if the foot's middle is not over the plate) and the block's top over the part of the sole the
  block's outline covers (from b0 to b1 mm along the foot from the heel end). The centre of mass is 93.9 mm from the
  heel end. With the block wholly behind it (b1 below 93.9), the foot tips toe-down about the block's front edge
  until its toe end is on the plate, turning about that edge without slipping: sin(pitch) = block height /
  (220 - b1), the heel end rises to the block height plus b1 sin(pitch) and moves b1 (1 - cos(pitch)) forward.
  With the block wholly in front of it (b0 above 93.9), it tips heel-down about the block's rear edge onto its heel
  end. With the block under it, it lies flat on the block. Without the block under it, it lies flat on the plate.

This is a geometric picture of the settling, not physics: no bounce, no slip, no push on the block.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

RESTING_TOL = 0.001  # the sole this close to a support's height: resting, not carried


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat(r) -> np.ndarray:
    """(x, y, z, w) of a rotation matrix (Shepperd's method)."""
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


def _clip(poly: list, a: np.ndarray, b: float) -> list:
    """Sutherland-Hodgman: the part of a convex polygon (list of 2-D points) where a . p <= b."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        dp, dq = float(a @ p) - b, float(a @ q) - b
        if dp <= 0:
            out.append(p)
        if dp * dq < 0:
            out.append(p + (q - p) * (dp / (dp - dq)))
    return out


def block_span(heel_xy, u, block) -> tuple[float, float] | None:
    """(b0, b1): how far along the foot (from the heel end, m) the block's outline seen from above covers the
    sole's 220 x 70 mm outline, or None if they do not overlap."""
    n = np.array([-u[1], u[0]])
    r = _rot(block.quat_xyzw)
    c = np.asarray(block.pos, dtype=float)
    hx, hy = L.BLOCK_HALF[0], L.BLOCK_HALF[1]
    poly = [c[:2] + r[:2, 0] * sx * hx + r[:2, 1] * sy * hy for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # the sole's outline as four half-planes in the foot's frame (along u: 0..L, across n: -W/2..W/2)
    h = np.asarray(heel_xy, dtype=float)
    for a, b in ((-u, -float(u @ h)), (u, L.FOOT_LEN + float(u @ h)),
                 (-n, L.FOOT_W / 2 - float(n @ h)), (n, L.FOOT_W / 2 + float(n @ h))):
        poly = _clip(poly, a, b)
        if len(poly) < 3:
            return None
    s = [float((p - h) @ u) for p in poly]
    return max(min(s), 0.0), min(max(s), L.FOOT_LEN)


def rest_pose(origin, quat_xyzw, block) -> tuple[np.ndarray, np.ndarray]:
    """(origin, quat_xyzw) of the foot as it would come to rest from a level pose, or the pose unchanged while it
    is carried."""
    o = np.asarray(origin, dtype=float)
    r = _rot(quat_xyzw)
    u3 = r[:, 0]
    heading = math.atan2(u3[1], u3[0])
    u = np.array([math.cos(heading), math.sin(heading)])
    mid = o[:2] + u * L.FOOT_LEN / 2
    base = L.PLATE_TOP if (L.PLATE_X0 <= mid[0] <= L.PLATE_X1 and abs(mid[1]) <= L.PLATE_HALF_Y) else L.TOP
    rb = _rot(block.quat_xyzw)
    block_top = float(np.asarray(block.pos, dtype=float)[2] + np.abs(rb[2]) @ np.array(L.BLOCK_HALF))
    sole = float(o[2])
    if min(abs(sole - L.TOP), abs(sole - L.PLATE_TOP), abs(sole - block_top)) > RESTING_TOL:
        return o, np.asarray(quat_xyzw, dtype=float)  # carried
    hb = block_top - base
    span = block_span(o[:2], u, block) if hb > RESTING_TOL else None
    along, height, pitch = 0.0, base, 0.0
    if span is not None:
        b0, b1 = span
        if b1 < L.FOOT_COM_S:  # block behind the centre of mass: toe down about its front edge
            pitch = math.asin(min(1.0, hb / max(L.FOOT_LEN - b1, 1e-6)))
            along, height = b1 * (1 - math.cos(pitch)), base + hb + b1 * math.sin(pitch)
        elif b0 > L.FOOT_COM_S:  # block in front of it: heel down about its rear edge
            tip = math.asin(min(1.0, hb / max(b0, 1e-6)))
            along, height, pitch = b0 * (1 - math.cos(tip)), base, -tip
        else:  # block under it: flat on the block
            height = base + hb
    c, s = math.cos(heading), math.sin(heading)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])  # +pitch: toe down
    new_o = np.array([o[0] + u[0] * along, o[1] + u[1] * along, height])
    return new_o, _quat(rz @ ry)
