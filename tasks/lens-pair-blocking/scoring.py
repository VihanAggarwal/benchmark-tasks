"""The scorer, as plain functions of the two edger blocks' states and the arms' state. NumPy only; no simulator.

The same rule applies to each lens L (left and right). A block is ON L when its centre, seen from above, is within
35 mm of L's optical centre (OC) and less than 60 mm above L's top. L is BLOCKED when all of these hold:

  (a) exactly one block is on L;
  (b) it is flat: its 16 mm axis is within 3 deg of vertical (either 18 x 24 mm face down);
  (c) it is seated: its lowest corner is within 2 mm of L's top;
  (d) it is still: over the last 0.5 s its centre's path is at most 2 mm and it turned at most 2 deg in all
      (positions and orientations, never the reported velocities, which chatter in the simulator);
  (e) it is let go: no gripper whose TCP is within 40 mm of the block's centre is less than half open (rigstate
      opening under 0.5, pads under 47 mm apart). When the arms cannot be read, nothing counts as let go.

d is the horizontal distance from that block's centre to L's target, the point 8 mm temporal of the OC along the
ink line (away from the nasal square); a is the angle seen from above between the block's 24 mm axis and the line,
folded to 0..90 deg. L scores 0.5 if blocked with d <= 2 mm and a <= 3 deg, else 0.25 if blocked with d <= 3 mm and
a <= 6 deg, else 0. The score is the sum over the two lenses: 0, 0.25, 0.5, 0.75 or 1.0, read at the last step.

The scorer keeps only each block's recent poses (for (d)) and is fed one observation per control step by
``task.success()``:

    Tracker.update(blocks, arms) -> Result(score, success, details)

``blocks`` maps each block's name (``layout.BLOCKS``) to anything with ``pos`` (centre), ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp`` and ``opening``, or ``None``. A block state with a
non-finite number or a zero quaternion (a simulator blow-up), or a missing block, scores 0 for that step and leaves
the tracker as it was. The blocks are identical, so either may go on either lens.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_HALF = np.array(L.BLOCK_HALF, dtype=float)


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


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def blocks_valid(blocks) -> bool:
    """Both blocks present with valid states."""
    try:
        return all(valid(blocks[name]) for name in L.BLOCKS)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            tcp = np.asarray(arms[side].tcp, dtype=float).ravel()
            if tcp.size != 3 or not np.all(np.isfinite(tcp)) or not math.isfinite(float(arms[side].opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def unit_quat(quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float)
    return q / (np.linalg.norm(q) or 1.0)


# -- one block ----------------------------------------------------------------------------------------------------
def tilt_deg(s) -> float:
    """Angle between the block's 16 mm axis and the vertical (deg, 0..90): either 18 x 24 mm face may be down."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def corners(s) -> np.ndarray:
    """The block's eight corners (world, 8 x 3) from its centre and orientation."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(s.quat_xyzw).T


def low_above_lens(s) -> float:
    """Height of the block's lowest corner above a lens's top (m; the two lenses' tops are at one height)."""
    return float(corners(s)[:, 2].min() - L.LENS_TOP_Z)


def long_heading_deg(s) -> float | None:
    """Heading of the block's 24 mm axis seen from above, folded to 0 .. 180 deg (None if it points straight up)."""
    v = rotation(s.quat_xyzw)[:, 1]
    if math.hypot(v[0], v[1]) < 1e-6:
        return None
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def angle_to_line_deg(s, side: str) -> float:
    """a: the angle seen from above between the block's 24 mm axis and lens ``side``'s ink line (0..90 deg)."""
    h = long_heading_deg(s)
    if h is None:
        return 90.0
    diff = (h - L.line_heading_deg(side)) % 180.0
    return float(min(diff, 180.0 - diff))


def oc_distance(s, side: str) -> float:
    """Horizontal distance from the block's centre to lens ``side``'s OC (m)."""
    ox, oy = L.OC[side]
    return float(math.hypot(float(s.pos[0]) - ox, float(s.pos[1]) - oy))


def target_distance(s, side: str) -> float:
    """d: horizontal distance from the block's centre to lens ``side``'s target, 8 mm temporal of the OC (m)."""
    tx, ty = L.TARGET[side]
    return float(math.hypot(float(s.pos[0]) - tx, float(s.pos[1]) - ty))


def on_lens(s, side: str) -> bool:
    """The block's centre, seen from above, within 35 mm of the OC, and less than 60 mm above the lens's top."""
    return bool(oc_distance(s, side) <= L.ON_RADIUS and float(s.pos[2]) - L.LENS_TOP_Z < L.ON_HEIGHT)


def let_go(s, arms) -> bool:
    """No gripper whose TCP is within 40 mm of the block's centre is less than half open. Unreadable arms: False
    (no credit without evidence)."""
    if not arms_valid(arms):
        return False
    c = np.asarray(s.pos, dtype=float)
    for side in ("left", "right"):
        near = float(np.linalg.norm(np.asarray(arms[side].tcp, dtype=float) - c)) <= L.LET_GO_R
        if near and float(arms[side].opening) < L.LET_GO_OPENING:
            return False
    return True


def nearest_tcp(s, arms) -> float | None:
    if not arms_valid(arms):
        return None
    c = np.asarray(s.pos, dtype=float)
    return min(float(np.linalg.norm(np.asarray(arms[a].tcp, dtype=float) - c)) for a in ("left", "right"))


def quat_angle(q0, q1) -> float:
    """The angle of the rotation from orientation q0 to q1 (rad)."""
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(unit_quat(q0), unit_quat(q1)))))))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each block's recent poses (for stillness), and the episode."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a missing or non-finite block state (score 0, ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.hist: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in L.BLOCKS}  # (centre, quat)
        self.prev: dict[str, np.ndarray] | None = None  # block centres at the previous valid control step
        self.best = 0.0  # highest score so far (a diagnostic; the score is read at the end)
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(blocks) -> bool:
        """The starting layout: each block standing flat on the table near its starting mark."""
        for name, (sx, sy) in L.BLOCK_START.items():
            s = blocks[name]
            if (math.hypot(float(s.pos[0]) - sx, float(s.pos[1]) - sy) >= L.START_TOL
                    or abs(float(corners(s)[:, 2].min()) - L.TOP) >= 0.003 or tilt_deg(s) >= 2.0):
                return False
        return True

    def path(self, name: str) -> tuple[float, float]:
        """The block's centre path (m) and total turn (deg) over the last 0.5 s; inf until 0.5 s has been seen."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return math.inf, math.inf
        pts = np.array([p for p, _ in h])
        dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        turn = sum(quat_angle(h[i][1], h[i + 1][1]) for i in range(len(h) - 1))
        return dist, float(math.degrees(turn))

    def update(self, blocks, arms=None) -> Result:
        if not blocks_valid(blocks):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        now = {n: np.array(blocks[n].pos, dtype=float) for n in L.BLOCKS}
        if self.prev is not None and self.at_start(blocks):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.BLOCKS)
            if jump > L.RESET_JUMP:  # the simulator put the blocks back on their marks: a new episode (rigstate's
                self.reset()  #       reset hook normally does this first; this is the fallback)
        self.prev = now
        for n in L.BLOCKS:
            self.hist[n].append((now[n], unit_quat(blocks[n].quat_xyzw)))
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(blocks, arms)
        self.best = max(self.best, self.last.score)
        return self.last

    def lens(self, blocks, arms, side: str) -> dict:
        """One lens's state, its points (0, 0.25 or 0.5) and every quantity that decides it."""
        on = [n for n in L.BLOCKS if on_lens(blocks[n], side)]
        row: dict = {"n_on": len(on), "blocks_on": on, "blocked": False, "lens_score": 0.0}
        if len(on) != 1:
            row["why"] = "no block" if not on else "two blocks"
            return row
        name = on[0]
        s = blocks[name]
        path, turn = self.path(name)
        tilt, low = tilt_deg(s), low_above_lens(s)
        d, a = target_distance(s, side), angle_to_line_deg(s, side)
        checks = {
            "flat": tilt <= L.FLAT_DEG,
            "seated": abs(low) <= L.SEAT_TOL,
            "still": path <= L.STILL_PATH and turn <= L.STILL_TURN_DEG,
            "let_go": let_go(s, arms),
        }
        tcp = nearest_tcp(s, arms)
        row.update(block=name, d_mm=_mm(d), a_deg=round(a, 2), tilt_deg=round(tilt, 2), low_mm=_mm(low),
                   oc_mm=_mm(oc_distance(s, side)), path_mm=_mm(path), turn_deg=round(turn, 2) if math.isfinite(turn)
                   else None, tcp_mm=_mm(tcp), held=not checks["let_go"], **checks)
        row["blocked"] = all(checks.values())
        if not row["blocked"]:
            row["why"] = ",".join(k for k, v in checks.items() if not v)
        elif d <= L.D_TIGHT and a <= L.A_TIGHT_DEG:
            row["lens_score"] = L.LENS_TIGHT
        elif d <= L.D_LOOSE and a <= L.A_LOOSE_DEG:
            row["lens_score"] = L.LENS_LOOSE
        else:
            row["why"] = "off target"
        return row

    def result(self, blocks, arms=None) -> Result:
        if not blocks_valid(blocks):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        rows = {side: self.lens(blocks, arms, side) for side in L.SIDES}
        score = float(sum(r["lens_score"] for r in rows.values()))
        details = {
            "valid_state": True,
            "lenses": rows,
            "arms_available": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "best": max(self.best, score),
        }
        return Result(score, bool(score == 1.0), details)
