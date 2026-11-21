"""The scorer, as plain functions of the parts' states and the arms' state. NumPy only; no simulator.

``Tracker.update(parts, arms)`` -> Result(score, success, details), fed one observation per control step. ``parts``
maps BOARD, SQUARE, PENCIL and BLOCK to mechkit AssemblyStates (frames in layout.py). The arms are read only for the
hands-off check.

A pencil line cannot be drawn in the simulator. What is scored is the PATH of the pencil's lead tip (a declared rule,
the only one): on each step the tip is on the board's top face (within 0.3 mm, or pressed into it), over that face,
with the pencil within 20 degrees of upright, AND the simulator's contact force between pencil and board is 0.3 N or
more (contact.py; unreadable counts as none), the tip's point in the board's frame is added to the line. Nothing else
marks the board. Offline (no forces) the caller says so with geometric=True and the pose rule alone stands in. In a
real trial the pencil's own graphite draws that path.

    SEATED      the head's 90-degree face within 1.0 mm of the board's near edge, turned within 1.5 degrees of it and
                lying within 3 degrees of flat (real: no light under the face along the edge)
    GUIDED      an ink point is guided if, on that step, the square is seated AND the pencil's -x face is on the blade's
                end (gap -1.2 to +0.8 mm) with its middle within 9 mm of the blade's middle (the pencil is against the
                blade's end, so the square's face on the edge sets how far in the line lies)
    GUIDE OK    of the 2 mm stretches along the line, at least 95 percent hold ink and every ink point in them was
                guided (judged by length, so dwelling in one place earns nothing)
    SPANS       the line reaches at least 90 mm along the edge, with no gap over 6 mm along it
    PARALLEL    the straight line fitted through every ink point within 1 degree of parallel to the near edge
    STRAIGHT    every ink point within 1.5 mm of that fitted line (a kink where the face left the edge, a second line
                or a stray mark fails it)
    AT 50 MM    the fitted line 50 mm in from the near edge, within 2 mm, at both its ends
    STOWED      the pencil back in its block, the tip within 3 mm of the slot's axis, at most 5 mm off its floor, axis
                within 10 degrees of upright
    HANDS OFF   on every step of the last 1 s both finger zones 30 mm or more from the square, the pencil and its block
    STILL       over the last 1 s the square and the pencil each moved under 5 mm
    LATCHES     the board moved 20 mm from where it started or turned 3 degrees, or any part more than 3 mm below the
                table top, at any step

    0      a latch fired, or the square was never seated against the edge, or no line at all
    0.25   the square seated against the edge for 0.5 s, but no good line
    0.5    a good line (spans, guided, parallel, straight, 50 mm in), the pencil not back in its block at the last step
    0.75   a good line and stowed, but hands off or still fails on a step of the last 1 s
    1.0    a good line, stowed, hands off and still for the whole last 1 s
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import mechkit as mk
import rigstate

BOARD = "board"
SQUARE = "square"
PENCIL = "pencil"
BLOCK = "holder"
PARTS = (BOARD, SQUARE, PENCIL, BLOCK)
NEAR_B = -L.BOARD_W / 2  # the near edge in the board's frame


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def yaw_of(r: np.ndarray) -> float:
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


@dataclass(frozen=True)
class Pose:
    o: np.ndarray  # origin, world (env frame)
    r: np.ndarray  # 3x3, local to world

    def local(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=float) - self.o) @ self.r


def read(state) -> Pose | None:
    try:
        o = np.asarray(state.origin, dtype=float).ravel()
        r = mk.quat_to_matrix(state.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        return Pose(o, np.asarray(r, dtype=float))
    except (AttributeError, TypeError, ValueError):
        return None


# -- the square ----------------------------------------------------------------------------------------------------------
def face_gap(board: Pose, square: Pose) -> float:
    """How far the head's face is off the near edge (m; negative: into it), measured in the board's frame."""
    return float(NEAR_B - board.local(square.o)[0])


def seated(board: Pose, square: Pose) -> bool:
    p = board.local(square.o)
    rel = board.r.T @ square.r
    return bool(-0.0010 <= face_gap(board, square) <= L.SEAT_TOL + L.EPS
                and abs(float(p[1])) <= L.BOARD_L / 2
                and abs(math.degrees(yaw_of(rel))) <= L.SEAT_YAW_DEG + L.EPS
                and tilt_deg(square.r) <= L.SEAT_TILT_DEG + L.EPS)


# -- the line ----------------------------------------------------------------------------------------------------------
def line_metrics(points) -> dict:
    """``points``: (N, 3) rows (x_b, y_b, guided) in the board's frame. Every check, and the numbers behind it."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    out = {"n": int(len(pts)), "spans": False, "parallel": False, "straight": False, "at_offset": False,
           "guided": False, "angle_deg": None, "max_dev_mm": None, "offset_err_mm": None, "guided_frac": None,
           "y_range_mm": None, "length_mm": None, "max_gap_mm": None}
    if len(pts) < 5:
        return out
    x, y, g = pts[:, 0], pts[:, 1], pts[:, 2]
    ys = np.sort(y)
    gap = float(np.max(np.diff(ys)))
    length = float(ys[-1] - ys[0])
    edges = np.arange(ys[0], ys[-1] + L.GUIDE_BIN, L.GUIDE_BIN)
    idx = np.digitize(y, edges) - 1
    n_bins = max(1, len(edges) - 1)
    good_bins = 0
    for k in range(n_bins):
        sel = idx == k
        if np.any(sel) and np.all(g[sel] > 0.5):
            good_bins += 1
    frac = good_bins / n_bins
    out.update(y_range_mm=[round(float(ys[0]) * 1000, 2), round(float(ys[-1]) * 1000, 2)],
               length_mm=round(length * 1000, 2), max_gap_mm=round(gap * 1000, 2), guided_frac=round(frac, 3),
               guided=bool(frac >= L.GUIDED_FRAC - L.EPS),
               spans=bool(length >= L.LINE_MIN - L.EPS and gap <= L.SPAN_GAP + L.EPS))
    if length < 0.010:
        return out
    b, a = np.polyfit(y, x, 1)
    angle = math.degrees(math.atan(abs(float(b))))
    dev = np.abs(x - (a + b * y)) / math.sqrt(1.0 + float(b) ** 2)
    errs = [float(a + b * yy) - NEAR_B - L.LINE_IN for yy in (float(ys[0]), float(ys[-1]))]
    off = max(errs, key=abs)
    out.update(angle_deg=round(angle, 3), max_dev_mm=round(float(np.max(dev)) * 1000, 3),
               offset_err_mm=round(off * 1000, 3), parallel=bool(angle <= L.PARALLEL_DEG + L.EPS),
               straight=bool(float(np.max(dev)) <= L.STRAIGHT + L.EPS),
               at_offset=bool(abs(off) <= L.OFFSET_TOL + L.EPS))
    return out


def good_line(m: dict) -> bool:
    return bool(m["spans"] and m["guided"] and m["parallel"] and m["straight"] and m["at_offset"])


def ink_point(board: Pose, pencil: Pose) -> np.ndarray | None:
    """The lead's tip in the board's frame if it marks the board this step, else None."""
    if tilt_deg(pencil.r) > L.INK_TILT_DEG:
        return None
    p = board.local(pencil.o)
    if abs(p[0]) > L.BOARD_W / 2 or abs(p[1]) > L.BOARD_L / 2:
        return None
    if p[2] > L.BOARD_H + L.INK_Z or p[2] < L.BOARD_H - 0.010:  # 10 mm under the face: a glitch, not a mark
        return None
    return p


def guided(board: Pose, square: Pose, pencil: Pose) -> bool:
    """The square is seated and the pencil's -x face lies on the blade's end, within the blade's width."""
    if not seated(board, square):
        return False
    s = square.local(pencil.o)
    gap = float(s[0]) + L.PN_X0 - L.SQ_BLADE_END
    return bool(L.GUIDE_GAP_LO - L.EPS <= gap <= L.GUIDE_GAP_HI + L.EPS and abs(float(s[1])) <= L.GUIDE_Y + L.EPS)


def stowed(pencil: Pose, block: Pose) -> bool:
    c = block.local(pencil.o)
    radial = math.hypot(float(c[0]) + L.PN_CENTER_X, float(c[1]))  # the body's middle on the slot's axis
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(pencil.r[:, 2], block.r[:, 2]))))))
    return bool(radial <= L.STOW_RADIAL and L.BL_FLOOR - 0.002 <= c[2] <= L.BL_FLOOR + L.STOW_SEAT
                and ang <= L.STOW_TILT_DEG)


def lowest_z(name: str, p: Pose) -> float:
    """A part's lowest point, table frame (the parts stay near upright; tilted ones are bounded from their frame)."""
    return float(p.o[2] - L.TOP)


def tier(latched: bool, ever_seated: bool, m: dict, is_stowed: bool, settled: bool) -> float:
    if latched:
        return 0.0
    if not good_line(m):
        return 0.25 if ever_seated else 0.0
    if not is_stowed:
        return 0.5
    if not settled:
        return 0.75
    return 1.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The ink ledger, the seat streak, the latches, the last 1 s of hands-off and of positions, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.seat_need = max(1, int(round(L.SEAT_HOLD_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"BOARD_MOVED": None, "BOARD_TURNED": None, "FELL": None}
        self.ink: list = []
        self.board0: tuple | None = None
        self.seat_run = 0
        self.ever_seated = False
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: dict | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(ps: dict) -> bool:
        sq = np.linalg.norm(ps[SQUARE].o - np.array(L.SQ_START)) <= L.START_TOL
        pn = np.linalg.norm(ps[PENCIL].o - np.array(L.PN_START)) <= L.START_TOL
        return bool(sq and pn)

    def update(self, parts: dict, arms, force: float | None = None, geometric: bool = True) -> Result:
        """``force``: the simulator's pencil-on-board contact force this step (N; None: unreadable, counts as none).
        ``geometric=True`` only where there is no simulator (and in the unit tests): the pose rule stands in."""
        self.steps += 1
        ps = {n: read(parts.get(n)) for n in PARTS}
        if any(v is None for v in ps.values()):
            self.bad_steps += 1
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.at_start(ps):
            jump = max(float(np.linalg.norm(ps[n].o - self.prev[n].o)) for n in (SQUARE, PENCIL))
            if jump >= L.RESET_JUMP:
                self.reset()
                self.steps = 1
                self.resets += 1
        if self.board0 is None:
            self.board0 = (ps[BOARD].o.copy(), yaw_of(ps[BOARD].r))
        self.prev = ps
        if self.latch["BOARD_MOVED"] is None and np.linalg.norm(ps[BOARD].o - self.board0[0]) >= L.BOARD_DRAG:
            self.latch["BOARD_MOVED"] = self.steps
        if self.latch["BOARD_TURNED"] is None and abs(math.degrees(yaw_of(ps[BOARD].r) - self.board0[1])) >= L.BOARD_YAW_DEG:
            self.latch["BOARD_TURNED"] = self.steps
        if self.latch["FELL"] is None and min(lowest_z(n, p) for n, p in ps.items()) < -L.FALL_DROP:
            self.latch["FELL"] = self.steps
        is_seated = seated(ps[BOARD], ps[SQUARE])
        self.seat_run = self.seat_run + 1 if is_seated else 0
        if self.seat_run >= self.seat_need:
            self.ever_seated = True
        p = ink_point(ps[BOARD], ps[PENCIL])
        pressed = geometric or (force is not None and math.isfinite(force) and force >= L.MIN_FORCE)
        if not pressed:
            p = None
        g = None
        if p is not None:
            g = guided(ps[BOARD], ps[SQUARE], ps[PENCIL])
            self.ink.append((float(p[0]), float(p[1]), 1.0 if g else 0.0))
        self.hist.append({n: ps[n].o.copy() for n in (SQUARE, PENCIL, BLOCK)})
        ok, gap = False, None
        try:
            dist = _dist_fn(parts)
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or no arms: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(ps, gap, p, g, is_seated)
        self.last.details.update(force_n=None if force is None else round(float(force), 3), geometric=geometric)
        return self.last

    def motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x[n] - h[0][n])) for x in h[1:] for n in (SQUARE, PENCIL))

    def judge(self, ps: dict, gap, p, g, is_seated: bool) -> Result:
        m = line_metrics(self.ink)
        is_stowed = stowed(ps[PENCIL], ps[BLOCK])
        released = len(self.released) == self.window and all(self.released)
        move = self.motion()
        still = bool(move is not None and move <= L.STILL_MOVE + L.EPS)
        score = tier(self.latched, self.ever_seated, m, is_stowed, released and still)
        details = {"valid_state": True, "score": score, "latch": dict(self.latch), "line": m,
                   "good_line": good_line(m), "stowed": is_stowed, "released_1s": released, "still_1s": still,
                   "inking_now": p is not None, "guided_now": g, "seated_now": is_seated,
                   "ever_seated": self.ever_seated,
                   "face_gap_mm": round(face_gap(ps[BOARD], ps[SQUARE]) * 1000, 2),
                   "hands_gap_mm": None if gap is None else round(float(gap) * 1000, 1),
                   "move_1s_mm": None if move is None else round(move * 1000, 2),
                   "square_y_mm": round(float(ps[SQUARE].o[1]) * 1000, 2),
                   "square_yaw_deg": round(math.degrees(yaw_of(ps[SQUARE].r)), 3)}
        return Result(score, bool(score == 1.0), details)


def _dist_fn(parts: dict):
    fns = [parts[n].distance for n in (SQUARE, PENCIL, BLOCK)]

    def dist(q):
        return np.minimum.reduce([np.asarray(f(q), dtype=float) for f in fns])
    return dist
