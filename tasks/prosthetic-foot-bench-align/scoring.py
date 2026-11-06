"""The scorer, as plain functions of the foot's and the heel block's states and the arms' state. NumPy only; no
simulator needed.

The score is the end state a prosthetist checks on the alignment jig before building the socket up over the laser
line: the foot sole down on the plate, toes forward, heel point on the centre line, let go with both grippers moved
clear, and still (the gate, else 0), then 0.25 for the gate and 0.25 for each of three independent checks: (A) the
middle of the foot's length 25 to 35 mm in front of the laser line, (B) toes turned out 2 to 8 degrees toward the
robot's right (-y), (C) the heel resting on the heel block, which lies flat on the plate, with the toe on the plate.

    Tracker.update(foot, block, arms) -> Result(score, success, details)

``foot`` is the foot's mechkit AssemblyState (or anything with ``origin``, the base frame's origin at the heel end
of the sole, mid-width; ``quat_xyzw``; and ``lin_vel``): every point is computed from that pose, never from a
bounding box. ``block`` is anything with ``pos`` (its centre), ``quat_xyzw`` and ``lin_vel``, as
``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with
``tcp`` and ``tips`` (the point between the pad tips), or None when the arms could not be read; then nothing counts
as let go on that step (a hold cannot be ruled out) and the step is counted in ``arms_unavailable``. Only those two
points are read (rigstate gives no gripper heading), so a wide-open finger can touch the foot unseen; the corner
limit in (C) (``block_front``) means a foot resting heel on the block and toe on the plate never needs a finger to
stay there. A state with a non-finite number, a zero quaternion or a missing field (a simulator blow-up) scores 0
for that step and leaves the tracker as it was.

Points: H and T are the midpoints of the bottom edges of the heel end and the toe end of the sole, M = (H + T) / 2.
The block is a plain box, the same either 60 x 70 mm face down, so its tilt is folded. The foot is not symmetric:
upside down is 180 degrees from upright and backwards is 180 degrees from toes forward.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 control steps: 11 samples span 0.5 s
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
_BLOCK_HALF = np.array(L.BLOCK_HALF, dtype=float)
_FOOT_BOXES = tuple((np.array(c, dtype=float), np.array(s, dtype=float) / 2) for c, s in L.FOOT_SHAPES)
_SOLE_CORNERS = np.array([(s, w, 0.0) for s in (0.0, L.FOOT_LEN) for w in (-L.FOOT_W / 2, L.FOOT_W / 2)])
ARMS = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a unit quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vec(s, key: str, n: int) -> np.ndarray | None:
    try:
        v = np.asarray(getattr(s, key), dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return None
    return v if v.size == n and np.all(np.isfinite(v)) else None


def _valid(s, where: str) -> bool:
    keys = (where, "quat_xyzw", "lin_vel")
    vs = [_vec(s, k, 4 if k == "quat_xyzw" else 3) for k in keys]
    return all(v is not None for v in vs) and bool(np.linalg.norm(vs[1]) > 0.5)


def foot_valid(foot) -> bool:
    """The foot's origin, orientation and velocity finite, the quaternion not degenerate."""
    return _valid(foot, "origin")


def block_valid(block) -> bool:
    return _valid(block, "pos")


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP and pad-tip point (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            for key in ("tcp", "tips"):
                v = np.asarray(getattr(arms[side], key), dtype=float).ravel()
                if v.size != 3 or not np.all(np.isfinite(v)):
                    return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- the foot -------------------------------------------------------------------------------------------------------
def foot_frame(foot) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of the foot's base frame in the world."""
    return np.asarray(foot.origin, dtype=float), rotation(foot.quat_xyzw)


def heel_toe(foot) -> tuple[np.ndarray, np.ndarray]:
    """H and T: the midpoints of the bottom edges of the heel end and the toe end (world)."""
    o, r = foot_frame(foot)
    return o.copy(), o + r[:, 0] * L.FOOT_LEN


def sole_corners(foot) -> np.ndarray:
    """The sole's four corners (world, 4 x 3)."""
    o, r = foot_frame(foot)
    return o + _SOLE_CORNERS @ r.T


def foot_tilt_deg(foot) -> float:
    """Angle between the foot's up axis and the vertical (0 sole down, 180 upside down)."""
    zz = rotation(foot.quat_xyzw)[2, 2]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, zz)))))


def heading_deg(foot) -> float:
    """Direction of H -> T seen from above, degrees from +x (counter-clockwise +, so toward -y is negative)."""
    h, t = heel_toe(foot)
    return float(math.degrees(math.atan2(t[1] - h[1], t[0] - h[0])))


def toe_out_deg(foot) -> float:
    """How far H -> T is turned from +x toward -y (the robot's right), degrees: the right foot's toe-out."""
    return -heading_deg(foot)


def ap_offset(foot) -> float:
    """The foot's middle M in front of the laser line's centre (m; negative behind it)."""
    h, t = heel_toe(foot)
    return float((h[0] + t[0]) / 2 - L.LASER_X)


def over_plate_xy(p) -> bool:
    return bool(L.PLATE_X0 <= float(p[0]) <= L.PLATE_X1 and abs(float(p[1])) <= L.PLATE_HALF_Y)


def sole_over_plate(foot) -> bool:
    """(G1) All four sole corners over the plate, seen from above."""
    return all(over_plate_xy(c) for c in sole_corners(foot))


# -- the block ------------------------------------------------------------------------------------------------------
def block_corners(block) -> np.ndarray:
    return np.asarray(block.pos, dtype=float) + (_SIGNS * _BLOCK_HALF) @ rotation(block.quat_xyzw).T


def block_tilt_deg(block) -> float:
    """Angle between the block's 20 mm axis and the vertical, folded (either 60 x 70 mm face down): 0..90."""
    zz = rotation(block.quat_xyzw)[2, 2]
    return float(math.degrees(math.acos(min(1.0, abs(zz)))))


def block_bottom(block) -> float:
    """Height of the block's lowest corner above the plate's top (m)."""
    return float(block_corners(block)[:, 2].min() - L.PLATE_TOP)


def block_flat_on_plate(block) -> bool:
    return bool(block_tilt_deg(block) <= L.BLOCK_TILT_MAX_DEG and abs(block_bottom(block)) <= L.BLOCK_BOTTOM_TOL)


def block_rel_heel(foot, block) -> tuple[float, float]:
    """The block's centre from H, seen from above, in the foot's frame: (along H -> T, sideways) in m."""
    h, t = heel_toe(foot)
    d = t[:2] - h[:2]
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-9 else np.array([1.0, 0.0])
    c = np.asarray(block.pos, dtype=float)[:2] - h[:2]
    return float(c @ u), float(c @ np.array([-u[1], u[0]]))


def block_front(foot, block) -> float:
    """How far in front of H, along H -> T seen from above, the block's furthest corner is (m). (C) needs this at
    most 90 mm, 4 mm behind the foot's centre of mass (93.9 mm from H): then a foot resting heel on the block and toe
    on the plate stays so by itself, and no finger (which the scorer cannot see) is needed to hold it there. A block
    square to the foot may have its centre up to 60 mm in front of H; one turned on the plate reaches further, so it
    must sit further back (turned 90 deg, its centre at most 55 mm)."""
    h, t = heel_toe(foot)
    d = t[:2] - h[:2]
    n = float(np.linalg.norm(d))
    u = d / n if n > 1e-9 else np.array([1.0, 0.0])
    return float(((block_corners(block)[:, :2] - h[:2]) @ u).max())


# -- the arms -------------------------------------------------------------------------------------------------------
def point_box_distance(p, centre, half, r) -> float:
    """Distance from a world point to an oriented box (centre, half sizes, rotation): 0 inside it."""
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(centre, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - half, 0.0)))


def arm_clearance(arm, foot, block) -> float:
    """The smaller distance from the arm's TCP and its pad-tip point to any box of the foot or to the block (m)."""
    o, r = foot_frame(foot)
    rb, cb = rotation(block.quat_xyzw), np.asarray(block.pos, dtype=float)
    best = math.inf
    for p in (arm.tcp, arm.tips):
        for c, half in _FOOT_BOXES:
            best = min(best, point_box_distance(p, o + r @ c, half, r))
        best = min(best, point_box_distance(p, cb, _BLOCK_HALF, rb))
    return best


def released(arms, foot, block) -> bool:
    """(G5) Both arms' TCPs and pad-tip points at least 20 mm from every box of the foot and from the block: let go
    and moved clear, as the instruction asks. Arms that cannot be read count as not released."""
    if not arms_valid(arms):
        return False
    return all(arm_clearance(arms[s], foot, block) >= L.RELEASE_CLEAR for s in ARMS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: the last 0.5 s of the heel point's, the toe point's and the block's
    paths, the previous step (for spotting a reset by teleport), and counts for the trace."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.paths = {k: deque(maxlen=_WINDOW + 1) for k in ("heel", "toe", "block")}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite or missing state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(point, velocity) of the foot's origin, of the block's centre] at the last valid step

    # -- resets -----------------------------------------------------------------------------------------------------
    @staticmethod
    def at_start(foot, block) -> bool:
        """The starting layout: the foot sole down on the table near its start (heel end at x 0.190, y -0.380) and
        the block flat on the table near its start (x 0.250, y +0.400), within 4.5 cm (a seeded reset's jitter)."""
        o = np.asarray(foot.origin, dtype=float)
        f_ok = (math.hypot(o[0] - L.FOOT_ORIGIN[0], o[1] - L.FOOT_ORIGIN[1]) < L.START_TOL
                and abs(o[2] - L.TOP) < L.START_Z_TOL and foot_tilt_deg(foot) < L.START_TILT_DEG)
        c = np.asarray(block.pos, dtype=float)
        b_ok = (math.hypot(c[0] - L.BLOCK_START[0], c[1] - L.BLOCK_START[1]) < L.START_TOL
                and abs(float(block_corners(block)[:, 2].min()) - L.TOP) < L.START_Z_TOL
                and block_tilt_deg(block) < L.START_TILT_DEG)
        return bool(f_ok and b_ok)

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for (p, _), (p0, v0) in zip(now, self.prev))

    # -- one control step -----------------------------------------------------------------------------------------------
    def update(self, foot, block, arms=None) -> Result:
        if not (foot_valid(foot) and block_valid(block)):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(foot, block, arms)
            return self.last
        now = [(np.array(foot.origin, dtype=float), np.array(foot.lin_vel, dtype=float)),
               (np.array(block.pos, dtype=float), np.array(block.lin_vel, dtype=float))]
        if self.prev is not None and self._teleported(now) and self.at_start(foot, block):
            self.reset()  # the simulator put both back at the start: a new episode (rigstate's and mechkit's reset
            #               hooks normally do this first; this is the fallback)
        self.prev = now
        h, t = heel_toe(foot)
        self.paths["heel"].append(h)
        self.paths["toe"].append(t)
        self.paths["block"].append(np.array(block.pos, dtype=float))
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(foot, block, arms)
        return self.last

    def path(self, key: str) -> float | None:
        """Path length over the last 0.5 s (m), or None before 0.5 s of this trial has been seen."""
        pts = self.paths[key]
        if len(pts) < _WINDOW + 1:
            return None
        return float(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1).sum())

    def still(self) -> bool:
        """(G6) The heel point, the toe point and the block's centre each moved less than 2 mm over the last 0.5 s."""
        paths = [self.path(k) for k in self.paths]
        return all(p is not None and p < L.STILL_PATH for p in paths)

    # -- the score --------------------------------------------------------------------------------------------------
    def result(self, foot, block, arms=None) -> Result:
        if not (foot_valid(foot) and block_valid(block)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        h, t = heel_toe(foot)
        tilt, heading = foot_tilt_deg(foot), heading_deg(foot)
        gates = {
            "g1_sole_over_plate": sole_over_plate(foot),
            "g2_sole_down": tilt <= L.FOOT_TILT_MAX_DEG,
            "g3_toes_forward": abs(heading) <= L.HEADING_MAX_DEG,
            "g4_heel_on_line": abs(float(h[1])) <= L.HEEL_SIDE_MAX,
            "g5_released": released(arms, foot, block),
            "g6_still": self.still(),
        }
        ap, toe_out = ap_offset(foot), -heading
        along, side = block_rel_heel(foot, block)
        front = block_front(foot, block)
        heel_h, toe_t = float(h[2] - L.PLATE_TOP), float(t[2] - L.PLATE_TOP)
        checks = {
            "a_forward": abs(ap - L.AP_TARGET) <= L.AP_TOL,
            "b_toe_out": L.TOE_OUT_MIN_DEG <= toe_out <= L.TOE_OUT_MAX_DEG,
            "c_heel_on_block": bool(block_flat_on_plate(block) and abs(side) <= L.BLOCK_SIDE_MAX
                                    and along >= L.BLOCK_ALONG_MIN and front <= L.BLOCK_FRONT_MAX
                                    and L.HEEL_H_MIN <= heel_h <= L.HEEL_H_MAX and abs(toe_t) <= L.TOE_T_TOL),
        }
        gate = all(gates.values())
        score = 0.25 * (1 + sum(checks.values())) if gate else 0.0
        clear = {s: arm_clearance(arms[s], foot, block) for s in ARMS} if arms_valid(arms) else None
        details = {
            "valid_state": True,
            "gate": gate,
            **gates,
            **checks,
            "ap_mm": _mm(ap),
            "toe_out_deg": round(toe_out, 2),
            "heel_y_mm": _mm(h[1]),
            "heel_h_mm": _mm(heel_h),
            "toe_t_mm": _mm(toe_t),
            "foot_tilt_deg": round(tilt, 2),
            "block_along_mm": _mm(along),
            "block_side_mm": _mm(side),
            "block_front_mm": _mm(front),
            "block_tilt_deg": round(block_tilt_deg(block), 2),
            "block_bottom_mm": _mm(block_bottom(block)),
            "grip_clear_mm": {s: _mm(v) for s, v in clear.items()} if clear else None,
            "heel_path_mm": _mm(self.path("heel")),
            "toe_path_mm": _mm(self.path("toe")),
            "block_path_mm": _mm(self.path("block")),
            "still": gates["g6_still"],
            "arms_available": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "heel": [round(float(v), 4) for v in h],
            "toe": [round(float(v), 4) for v in t],
        }
        return Result(score, bool(score == 1.0), details)
