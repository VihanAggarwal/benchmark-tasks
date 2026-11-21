"""The scorer, as plain functions of the three parts' states and the arms. NumPy only; no simulator.

``Tracker.update(parts, arms, pull)`` -> Result(score, success, details), fed one observation per control step.
``parts`` maps the names below to mechkit AssemblyStates: stub A (100 mm, lying along x), rail B (300 mm, along y) and the
clamp (joints "jaw", "trigger", "release"). ``pull`` is the ratchet's pull on the moving jaw (m), the clamp's own pawl
state: the clamp's force is K_JAW x (pull - the jaw's position) while the pawl is engaged.

    JOINT      A's end face against B's front face (feeler gauge), A's outer face flush with B's end face (straightedge
               and feeler), A square to B (A's far end off a try square's blade, in mm over A's 100 mm), A on B's front
               side, both boards flat on the bench
    ON THE WORK  the fixed pad's face within 1.5 mm of B's back face, the moving pad's face within 1.5 mm of A's free end,
               both pads over at least 15 mm of A's width and of B's width, the clamp flat on the bench
    CLAMPED    on the work and the clamp force (K_JAW x (pull - jaw)) at least 40 N with the pawl engaged
    RELEASED   on every step of the last 1 s both finger zones are 30 mm or more from A, B and the clamp
    STILL      over the last 1 s nothing moved over 1 mm or turned over 1 degree
    FELL       latched: any part's lowest point ever more than 20 mm below the table top

    0      FELL, or not clamped
    0.25   clamped, the joint outside the loose limits (gap 3 mm, flush 3 mm, square 5 mm)
    0.5    clamped, inside the loose limits, outside the tight ones (gap 1, flush 1.5, square 2.5 mm)
    0.75   clamped, inside the tight limits, but a hand still on it, or not still for 1 s
    1.0    clamped, inside the tight limits, both hands clear of everything and everything still for 1 s
    (the rules are tested in this order: the first that holds sets the score)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

A = "stub_a"
B = "rail_b"
CLAMP = "clamp"
PARTS = (A, B, CLAMP)


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def yaw(r: np.ndarray) -> float:
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def wrap_pi(a: float) -> float:
    """``a`` into (-pi/2, pi/2]: a board is the same board turned end for end."""
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return a if a > -math.pi / 2 + 1e-12 else math.pi / 2


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray  # origin, world, table top at z 0
    rot: np.ndarray


def read_pose(state) -> Pose | None:
    try:
        o = np.asarray(state.origin, dtype=float).ravel()
        r = rotation(state.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        return Pose(o - np.array([0.0, 0.0, L.TOP]), r)
    except (AttributeError, TypeError, ValueError):
        return None


def joint_values(state) -> dict | None:
    try:
        j = state.joints
        out = {k: float(j[k]) for k in ("jaw", "trigger", "release")}
        return out if all(math.isfinite(v) for v in out.values()) else None
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


# -- the joint -------------------------------------------------------------------------------------------------------
def joint_geometry(a: Pose, b: Pose) -> dict:
    """A against B, in B's frame (x across B, y along it from its end face, z up): ``gap`` (A's end face to B's front
    face, m), ``flush`` (A's outer face's corner at the joint, in from B's end face, m; negative: sticking out),
    ``square_mm`` (A's far end off the perpendicular over its length, m), ``side`` (A's centre in front of B's front
    face, m), ``tilt`` and ``dz`` (the boards flat, level with each other)."""
    yb, ya = yaw(b.rot), yaw(a.rot)
    rel = wrap_pi(ya - yb)
    c, s = math.cos(yb), math.sin(yb)
    d = a.pos[:2] - b.pos[:2]
    d = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])  # into B's frame
    u = np.array([math.cos(rel), math.sin(rel)])  # A's long axis in B's frame, toward +x
    v = np.array([-math.sin(rel), math.cos(rel)])  # across A, toward +y
    end = d - (L.A_LEN / 2) * u  # the middle of A's end face
    corner = end - (L.BOARD_W / 2) * v  # its outer corner (the end of the outer face at the joint)
    return {
        "gap": float(end[0] - L.BOARD_W / 2),
        "flush": float(corner[1] + L.B_LEN / 2),
        "square": float(L.A_LEN * abs(math.sin(rel))),
        "rel_deg": math.degrees(rel),
        "side": float(d[0] - L.BOARD_W / 2),
        "tilt": max(tilt_deg(a.rot), tilt_deg(b.rot)),
        "dz": float(abs(a.pos[2] - b.pos[2])),
    }


def joint_ok(g: dict, gap: float, flush: float, square: float) -> bool:
    return bool(-0.002 <= g["gap"] <= gap and abs(g["flush"]) <= flush and g["square"] <= square
                and g["side"] >= L.SIDE_OK and g["tilt"] <= 3.0 and g["dz"] <= L.LEVEL_Z)


def joint_tight(g: dict) -> bool:
    return joint_ok(g, L.GAP_TIGHT, L.FLUSH_TIGHT, L.SQUARE_TIGHT)


def joint_loose(g: dict) -> bool:
    return joint_ok(g, L.GAP_LOOSE, L.FLUSH_LOOSE, L.SQUARE_LOOSE)


# -- the clamp -------------------------------------------------------------------------------------------------------
def to_clamp(c: Pose, p) -> np.ndarray:
    return c.rot.T @ (np.asarray(p, dtype=float) - c.pos)


def board_corners(p: Pose, size_x: float, size_y: float) -> np.ndarray:
    """The four bottom corners of a board in world (table at z 0)."""
    pts = np.array([(sx * size_x / 2, sy * size_y / 2, 0.0) for sx in (1, -1) for sy in (1, -1)])
    return p.pos + pts @ p.rot.T


def clamp_geometry(a: Pose, b: Pose, c: Pose, jaw: float) -> dict:
    """Both boards in the clamp's frame: the pads' contact gaps and how far the pads cover the boards."""
    ca = np.array([to_clamp(c, q) for q in board_corners(a, L.A_LEN, L.BOARD_W)])
    cb = np.array([to_clamp(c, q) for q in board_corners(b, L.BOARD_W, L.B_LEN)])
    fixed_gap = float(cb[:, 0].min())  # B's back face against the fixed pad's face at x 0
    moving_gap = float((L.G0 - jaw) - ca[:, 0].max())  # A's free end against the moving pad's face
    def cover(pts):
        return float(min(pts[:, 1].max(), L.PAD_Y1) - max(pts[:, 1].min(), L.PAD_Y0))
    return {"fixed_gap": fixed_gap, "moving_gap": moving_gap, "cover_a": cover(ca), "cover_b": cover(cb),
            "tilt": tilt_deg(c.rot), "z": float(c.pos[2])}


def on_the_work(cg: dict) -> bool:
    return bool(abs(cg["fixed_gap"]) <= L.PAD_TOUCH and abs(cg["moving_gap"]) <= L.PAD_TOUCH
                and cg["cover_a"] >= L.PAD_OVERLAP_MIN and cg["cover_b"] >= L.PAD_OVERLAP_MIN
                and cg["tilt"] <= 5.0 and abs(cg["z"]) <= 0.008)


def clamp_force(pull: float | None, jaw: float, release: float) -> float:
    """The clamp force (N): the pawl's pull past the jaw's position times the stiffness; none when the pawl is free
    (the release tab turned past half) or the pull is unknown."""
    if pull is None or not math.isfinite(pull) or release >= L.REL_ON * L.REL_UPPER:
        return 0.0
    return float(L.K_JAW * max(0.0, pull - jaw))


def tier(latched: bool, clamped: bool, loose: bool, tight: bool, hands_off: bool, still: bool) -> float:
    """The first rule that holds sets the score."""
    if latched or not clamped:
        return 0.0
    if not loose:
        return 0.25
    if not tight:
        return 0.5
    if not (hands_off and still):
        return 0.75
    return 1.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The latch (FELL), the last 1 s of readings (stillness, hands off) and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"FELL": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev_clamp: Pose | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["FELL"] is not None

    @staticmethod
    def clamp_at_start(c: Pose) -> bool:
        p0 = np.array([L.CLAMP_XF, L.CLAMP_YC0, 0.0])
        return bool(float(np.linalg.norm(c.pos - p0)) <= L.START_TOL and tilt_deg(c.rot) < 3.0)

    def update(self, parts: dict, arms, pull: float | None = None) -> Result:
        self.steps += 1
        poses = {n: read_pose(parts.get(n)) for n in PARTS}
        jv = joint_values(parts.get(CLAMP))
        if any(p is None for p in poses.values()) or jv is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev_clamp = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        a, b, c = poses[A], poses[B], poses[CLAMP]
        if self.prev_clamp is not None and self.clamp_at_start(c) \
                and float(np.linalg.norm(c.pos - self.prev_clamp.pos)) >= L.RESET_JUMP:
            self.reset()  # the clamp back on its starting mark with a jump no trial makes: a reset
            self.steps = 1
            self.resets += 1
        self.prev_clamp = c
        self.hist.append({n: poses[n] for n in PARTS})
        if self.latch["FELL"] is None and min(float(p.pos[2]) for p in poses.values()) < -L.FALL_DROP:
            self.latch["FELL"] = self.steps
        ok, gap = False, None
        try:
            def dist(q):
                return np.min([parts[n].distance(q) for n in PARTS], axis=0)
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or no arms: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(a, b, c, jv, pull, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of any part from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        move = turn = 0.0
        for x in h[1:]:
            for n in PARTS:
                move = max(move, float(np.linalg.norm(x[n].pos - h[0][n].pos)))
                turn = max(turn, turn_deg(h[0][n].rot, x[n].rot))
        return {"move": move, "turn": turn}

    def judge(self, a: Pose, b: Pose, c: Pose, jv: dict, pull, hands_gap) -> Result:
        g = joint_geometry(a, b)
        cg = clamp_geometry(a, b, c, jv["jaw"])
        force = clamp_force(pull, jv["jaw"], jv["release"])
        work = on_the_work(cg)
        clamped = bool(work and force >= L.F_MIN)
        loose, tight = joint_loose(g), joint_tight(g)
        hands_off = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        score = tier(self.latched, clamped, loose, tight, hands_off, still)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch),
            "clamped": clamped, "on_the_work": work, "force_n": round(force, 1), "jaw_mm": _mm(jv["jaw"]),
            "trigger_rad": round(jv["trigger"], 3), "release_rad": round(jv["release"], 3),
            "pull_mm": None if pull is None else _mm(pull),
            "gap_mm": _mm(g["gap"]), "flush_mm": _mm(g["flush"]), "square_mm": _mm(g["square"]),
            "rel_deg": round(g["rel_deg"], 2), "side_mm": _mm(g["side"]), "tilt_deg": round(g["tilt"], 2),
            "dz_mm": _mm(g["dz"]), "loose": loose, "tight": tight,
            "fixed_gap_mm": _mm(cg["fixed_gap"]), "moving_gap_mm": _mm(cg["moving_gap"]),
            "cover_a_mm": _mm(cg["cover_a"]), "cover_b_mm": _mm(cg["cover_b"]),
            "released_now": bool(self.released[-1]), "released_1s": bool(hands_off), "still_1s": still,
            "hands_gap_mm": _mm(hands_gap),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(score, bool(score == 1.0), details)
