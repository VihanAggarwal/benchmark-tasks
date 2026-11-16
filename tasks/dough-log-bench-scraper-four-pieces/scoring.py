"""The scorer, as plain functions of the four pieces', the two scrapers' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the four dough pieces (base frame at each body's centre; the pieces p0..p2 carry a
"web" link: the cut strip of dough at the mark after them, sliding down its slot) and of the two scrapers (base frame
at the centre of the blade's edge, blade in its x-z plane). Every reading is a pose or a joint position; the arms are
read only for the let-go check.

    F1 BINNED       latched: a piece lower than 10 mm below the bench top (it went off the bench)
    STRIP DEPTH     each strip's travel down its slot (the cut depth), 0 to 40 mm, read from its joint
    THROUGH         a strip down 37 mm or more, ever (latched: a cut stays cut)
    BLADE IN        a scraper's edge, upright (blade plane within 15 degrees of vertical, its length within 15
                    degrees of along x), with its centre line inside the strip's 9.4 mm (plus 1 mm each side), its
                    middle within 36 mm of the log's centre line, and its edge lower than the log's mid height, ever
                    (latched). A strip pushed down by anything but a scraper's edge in it does not count.
    CUT             THROUGH and BLADE IN for the same mark
    CREASED         a strip pressed in more than 4 mm and not through (dough squashed, not cut)
    DRAGGED         latched: a piece 12 mm or more from where it began, or turned 8 degrees or more (smeared)
    IN LINE         every piece's centre within 30 mm of the log's centre line, in order along y
    TOOLS DOWN      both scrapers' lowest points no lower than 10 mm below the table, and neither within 2 mm of a
                    piece
    RELEASED        on every step of the last 1 s both finger zones are 30 mm or more from all six parts
    STILL           over the last 1 s every part moved under 1 mm and turned under 0.3 degrees

    0      F1 ever fired, or not released and still, or no cut
    1.0    three cuts, no crease, not dragged, in line, tools down
    0.5    three cuts otherwise, or two cuts
    0.25   one cut
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PIECES = L.PIECES
SCRAPERS = L.SCRAPERS
PARTS = PIECES + SCRAPERS


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


def _corners(size, centre=(0.0, 0.0, 0.0)):
    c = np.asarray(centre, dtype=float)
    return np.array([c + np.array([sx, sy, sz]) * np.asarray(size) / 2
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


PIECE_PTS = {f"p{k}": _corners((L.LOG_W, L.BODY_LEN[k], L.LOG_H)) for k in range(4)}
_SC_BLADE = _corners((L.SC_W, L.SC_T, L.SC_H), (0.0, 0.0, L.SC_H / 2))
_SC_BAR = _corners((L.BAR_LX, L.BAR_LY, L.BAR_LZ), (0.0, 0.0, L.BAR_Z))
SCRAPER_PTS = np.vstack([_SC_BLADE, _SC_BAR])


def scraper_samples() -> np.ndarray:
    """Points on a scraper in its own frame for the touching check: its edge and a grid over the blade."""
    pts = [(x, 0.0, z) for x in np.linspace(-L.SC_W / 2, L.SC_W / 2, 11) for z in np.linspace(0.0, L.SC_H, 5)]
    pts += [tuple(p) for p in _SC_BAR]
    return np.array(pts)


SCRAPER_SAMPLES = scraper_samples()


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation
    pts: dict  # part -> (N, 3) corner points
    web: tuple  # the three strips' depths (m), None where a joint is missing


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, pts = {}, {}, {}
    web = []
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            local = PIECE_PTS[name] if name in PIECES else SCRAPER_PTS
            pos[name], rot[name], pts[name] = o, r, o + local @ r.T
        for k in range(3):
            q = float(states[f"p{k}"].joints["web"])
            if not math.isfinite(q):
                return None
            web.append(q)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, pts, tuple(web))


# -- the readings ------------------------------------------------------------------------------------------------------
def fell(r: Reading) -> bool:
    return any(float(np.min(r.pts[n][:, 2])) < L.FALL_Z for n in PIECES)


def upright(r: Reading, name: str) -> bool:
    """The blade's plane within 15 degrees of vertical and its length within 15 degrees of along x."""
    rot = r.rot[name]
    c = math.cos(math.radians(L.BLADE_UPRIGHT_DEG))
    return bool(rot[2, 2] >= c and abs(rot[0, 0]) >= c)


def blade_in(r: Reading, k: int) -> bool:
    """Some scraper's edge is in strip k, upright, below the log's mid height."""
    for name in SCRAPERS:
        o = r.pos[name]
        if (upright(r, name) and abs(o[1] - L.web_y(k)) <= L.WEB_W / 2 + L.CUT_OK_MARGIN
                and abs(o[0] - L.LOG_X) <= L.CUT_X and o[2] <= L.CUT_EDGE_Z):
            return True
    return False


def creased(depths) -> tuple:
    return tuple(L.WEB_CREASE < q < L.WEB_THROUGH for q in depths)


def in_line(r: Reading) -> bool:
    ys = [float(r.pos[n][1]) for n in PIECES]
    return bool(all(abs(float(r.pos[n][0]) - L.LOG_X) <= L.LINE_X for n in PIECES)
                and all(b > a for a, b in zip(ys, ys[1:])))


def tools_down(r: Reading, states) -> bool:
    for name in SCRAPERS:
        if float(np.min(r.pts[name][:, 2])) < L.TOOL_FALL_Z:
            return False
        world = r.pos[name] + SCRAPER_SAMPLES @ r.rot[name].T + np.array([0.0, 0.0, L.TOP])
        for p in PIECES:
            try:
                d = float(np.min(states[p].distance(world)))
            except Exception:  # noqa: BLE001 - unreadable: not down
                return False
            if not math.isfinite(d) or d <= L.TOUCH:
                return False
    return True


def tier(latched: bool, settled: bool, n_cut: int, clean: bool) -> float:
    if latched or not settled or n_cut <= 0:
        return 0.0
    if n_cut >= 3 and clean:
        return 1.0
    if n_cut >= 2:
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


class Tracker:
    """The latches (binned, dragged, blade in, through), the last 1 s of readings (stillness) and of the let-go check,
    and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "dragged": None}
        self.cut_in = [False, False, False]
        self.through = [False, False, False]
        self.max_depth = [0.0, 0.0, 0.0]
        self.start: dict | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @staticmethod
    def starts() -> dict:
        return {f"p{k}": np.array([L.LOG_X, L.BODY_YC[k], L.BENCH_TOP + L.LOG_H / 2]) for k in range(4)}

    def at_start(self, r: Reading) -> bool:
        return (all(float(np.linalg.norm(r.pos[n] - s)) <= L.START_TOL for n, s in self.starts().items())
                and all(q < 0.002 for q in r.web))  # a reset puts every strip back flush

    def teleported(self, prev: Reading, now: Reading) -> bool:
        """The pieces put back on their starting marks, the strips flush again, with a jump no trial makes in one step
        (a reset)."""
        if not self.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in PARTS)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        if self.start is None:
            self.start = {n: (r.pos[n].copy(), r.rot[n].copy()) for n in PIECES}
        self.hist.append(r)
        self.prev = r
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = self.steps
        if self.latch["dragged"] is None and self.dragged(r):
            self.latch["dragged"] = self.steps
        for k in range(3):
            self.max_depth[k] = max(self.max_depth[k], r.web[k])
            if r.web[k] >= L.WEB_THROUGH - L.EPS:
                self.through[k] = True
            if blade_in(r, k):
                self.cut_in[k] = True
        try:
            dists = [states[n].distance for n in PARTS]

            def dist(points):
                return np.min([d(points) for d in dists], axis=0)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, states, gap)
        return self.last

    def dragged(self, r: Reading) -> bool:
        for n in PIECES:
            p0, r0 = self.start[n]
            if (float(np.linalg.norm(r.pos[n][:2] - p0[:2])) >= L.DRAG_MOVE
                    or turn_deg(r0, r.rot[n]) >= L.DRAG_TURN_DEG):
                return True
        return False

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of any part from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in PARTS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        return {"move": move, "turn": turn}

    def cuts(self) -> list:
        return [bool(self.through[k] and self.cut_in[k]) for k in range(3)]

    def judge(self, r: Reading, states, gap) -> Result:
        cuts = self.cuts()
        n_cut = sum(cuts)
        crease = creased(r.web)
        line = in_line(r)
        down = tools_down(r, states)
        dragged = self.latch["dragged"] is not None
        clean = bool(not any(crease) and not dragged and line and down)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, n_cut, clean)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "web_mm": [_mm(q) for q in r.web], "max_web_mm": [_mm(q) for q in self.max_depth],
            "through": list(self.through), "blade_in": list(self.cut_in), "cuts": cuts, "n_cut": n_cut,
            "creased": list(crease), "dragged": dragged, "in_line": line, "tools_down": down, "clean": clean,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "pieces_y_mm": [_mm(r.pos[n][1]) for n in PIECES], "pieces_x_mm": [_mm(r.pos[n][0]) for n in PIECES],
            "scraper_mm": {n: _mmv(r.pos[n]) for n in SCRAPERS},
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
