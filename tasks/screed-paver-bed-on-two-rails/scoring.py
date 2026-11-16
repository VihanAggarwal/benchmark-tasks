"""The scorer, as plain functions of the screed board's pose and the arms' state. NumPy only; no simulator needed.

The sand is a height model kept by the scorer: a grid of 2.5 cm cells, 8 along x by 16 along y (the two rail columns,
y -125..-100 and +100..+125 mm, are steel, not sand: 14 sand columns, 112 sand cells), one depth per cell. It starts
raked to 36 mm with a mound and a hollow (layout.start_heights) and is changed only by the board's measured pose,
once per control step, through ``Tracker.update`` (called from ``task.success()``):

    CUT      every cell whose centre is under the board's sole (25 mm across x, 400 mm along y: one cell column
             wide) and whose sand is above the sole there is cut down to the sole; the volume joins the heap the
             board carries in that cell's y column
    FILL     while the sole is within 3 mm of the rails' tops, a cell under it that is lower than the sole is filled
             up to the sole from that column's heap (no more than the heap holds)
    LIFTED   a column whose sole is more than 3 mm above the rails' tops drops its heap on the cell just ahead of the
             board (toward its direction of travel); ahead of the bed's near edge it falls in the spoil, and the heap
             is dropped there too when the board's front edge reaches 5 mm of the near edge
    CONSERVED  sum(depth) + heap + spoil never changes
    TIPPED   a board tilted more than 30 degrees (on its side or flat) cuts and fills nothing; any point of its surface
             within 2 mm of a finished cell's sand, or in it, presses that cell (see PRESSED)

The sole's height at a cell is the plane of the board's underside from its measured pose, so tilt and skew count.
Between two control steps the pose is interpolated in steps of at most 4 mm, so a fast pull cannot skip cells.

Readings (from the board's pose, the grid and the arms):

    BRIDGED   the sole within 3 mm of both rails' tops, where the board's centre line crosses them (over the rails'
              length); PULL is the longest x travel of the board's centre while bridged on every step
    CUT       at least 84 of the 112 sand cells (three quarters) within 6 mm of 25 mm
    DEV       the largest |depth - 25 mm| over all 112 sand cells
    STOWED    the sole within 3 mm of the rest blocks' tops at both blocks, the centre within 10 mm of their line,
              turned under 3 degrees about z and tilted under 3 degrees
    HANDS OFF every finger zone at least 30 mm from the board, the rails and the bed, on every step of the last 1 s
    STILL     over the last 1 s the board moved under 1 mm and turned under 0.3 degrees
    PRESSED   after the board has passed a cell (the sole was on its sand and has moved on) a pad tip, or a tipped
              board, came to within 2 mm of that cell's sand, or into it (latched until the board passes that cell again)
    F1        latched: the board's lowest point went 20 mm under the table top (knocked off the table)

    0      F1, or no pull of 150 mm bridged, or fewer than 84 cells within 6 mm (the bed is not even cut down)
    0.25   bridged pull and cut
    0.5    every cell within 6 mm of 25 mm
    0.75   every cell within 4 mm and the board stowed
    1.0    every cell within 3 mm, the board stowed, hands off and still for 1 s
    PRESSED caps the score at 0.5.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BOARD = "board"
_SAND_COLS = np.array([j not in L.RAIL_COLS for j in range(L.NY)])
_SAND = np.broadcast_to(_SAND_COLS, (L.NX, L.NY)).copy()
_X = np.array([[L.cell_x(i)] * L.NY for i in range(L.NX)])
_Y = np.array([[L.cell_y(j) for j in range(L.NY)] for _ in range(L.NX)])
_HX, _HY, _HZ = L.BD_X / 2, L.BD_L / 2, L.BD_Z / 2
_CORNERS = np.array([(sx * _HX, sy * _HY, sz * _HZ) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
SUB_STEP = 0.004  # the largest move between two sand updates (m)


def _surface(step=0.005):
    """Points on the board's six faces, about 5 mm apart, in its own frame (for the tipped-board press check)."""
    half = (_HX, _HY, _HZ)
    pts = []
    for ax in range(3):
        a, b = [k for k in range(3) if k != ax]
        ua = np.linspace(-half[a], half[a], max(2, int(round(2 * half[a] / step)) + 1))
        ub = np.linspace(-half[b], half[b], max(2, int(round(2 * half[b] / step)) + 1))
        ga, gb = np.meshgrid(ua, ub, indexing="ij")
        for sign in (-1.0, 1.0):
            p = np.zeros((ga.size, 3))
            p[:, a], p[:, b], p[:, ax] = ga.ravel(), gb.ravel(), sign * half[ax]
            pts.append(p)
    return np.concatenate(pts)


_SURFACE = _surface()


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        return np.full((3, 3), np.nan)
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


def yaw_deg(r: np.ndarray) -> float:
    """How far the board's long axis is turned from world y about z (deg, -90..90; the board is symmetric)."""
    a = r[:, 1]
    n = math.hypot(float(a[0]), float(a[1]))
    if n < 1e-9:
        return 90.0
    return math.degrees(math.asin(max(-1.0, min(1.0, float(a[0]) / n)))) * (1.0 if a[1] >= 0 else -1.0)


@dataclass(frozen=True)
class Reading:
    """The board, one step (m; the table top is z 0)."""
    pos: np.ndarray
    quat: np.ndarray
    rot: np.ndarray
    pts: np.ndarray  # its 8 corners
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # its velocity (m/s)


def read(states) -> Reading | None:
    """A Reading, or None when the board's state is missing, malformed or not finite."""
    try:
        s = states.get(BOARD) if isinstance(states, dict) else None
        if s is None:
            return None
        p = np.asarray(s.pos, dtype=float).ravel()
        q = np.asarray(s.quat_xyzw, dtype=float).ravel()
        if p.size != 3 or q.size != 4 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            return None
        r = rotation(q)
        if not np.all(np.isfinite(r)):
            return None
        v = np.asarray(getattr(s, "lin_vel", np.zeros(3)), dtype=float).ravel()
        v = v if v.size == 3 and np.all(np.isfinite(v)) else np.zeros(3)
        p = p - np.array([0.0, 0.0, L.TOP])
        return Reading(p, q / np.linalg.norm(q), r, p + _CORNERS @ r.T, v)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


# -- the sole -----------------------------------------------------------------------------------------------------------
def sole_plane(pos, rot, x, y):
    """Height (m, above the table top) of the board's underside plane over the horizontal points (x, y)."""
    n = rot[:, 2]
    return pos[2] - (_HZ + n[0] * (x - pos[0]) + n[1] * (y - pos[1])) / n[2]


def under_sole(pos, rot, x, y):
    """Which of the points (x, y), on the sole's plane, lie inside the sole's outline."""
    z = sole_plane(pos, rot, x, y)
    d = np.stack([x - pos[0], y - pos[1], z - pos[2]])
    loc = np.einsum("ji,j...->i...", rot, d)
    return (np.abs(loc[0]) <= _HX + 1e-9) & (np.abs(loc[1]) <= _HY + 1e-9), z


def bridge_heights(r: Reading):
    """The sole's height over each rail's centre line, where the board's centre line crosses it, and the x there;
    None when the board lies too nearly along x to cross them."""
    a = r.rot[:, 1]
    if abs(float(a[1])) < 0.5:
        return None
    out = []
    for side in (-1.0, 1.0):
        t = (side * L.RAIL_Y - r.pos[1]) / a[1]
        if abs(t) > _HY:
            return None
        p = r.pos + a * t
        out.append((float(sole_plane(r.pos, r.rot, p[0], p[1])), float(p[0])))
    return out


def bridged(r: Reading) -> bool:
    b = bridge_heights(r)
    if b is None or tilt_deg(r.rot) > L.TIP_DEG:
        return False
    return all(abs(h - L.RAIL_H) <= L.BRIDGE_TOL + L.EPS and L.RAIL_X0 <= x <= L.RAIL_X1 for h, x in b)


def stowed(r: Reading) -> bool:
    """Resting on both rest blocks: sole 55 mm at each block, centre over their line, square and level."""
    if abs(float(r.rot[1, 1])) < 0.5:
        return False
    for side in (-1.0, 1.0):
        h = float(sole_plane(r.pos, r.rot, L.STOP_X, side * L.STOP_Y))
        if abs(h - L.BD_REST_Z) > L.STOWED_Z_TOL + L.EPS:
            return False
    return bool(abs(float(r.pos[0]) - L.STOP_X) <= L.STOWED_XY_TOL + L.EPS and abs(float(r.pos[1])) <= L.STOWED_XY_TOL
                + L.EPS and abs(yaw_deg(r.rot)) <= L.STOWED_YAW_DEG + L.EPS
                and tilt_deg(r.rot) <= L.STOWED_TILT_DEG + L.EPS)


def fell(r: Reading) -> bool:
    """Knocked off the table: its lowest point 20 mm under the table top. (Tipping over on the table is not a fall:
    a tipped board cuts nothing and presses the finished sand it lies on.)"""
    return bool(float(np.min(r.pts[:, 2])) < -L.OFF_TABLE_DROP)


def hands_zone(r: Reading):
    """The distance function for the hands-off check: the board, both rails and the bed (world points in, metres)."""
    boxes = [(r.pos + np.array([0.0, 0.0, L.TOP]), (L.BD_X, L.BD_L, L.BD_Z), r.rot)]
    for side in (-1.0, 1.0):
        boxes.append((((L.RAIL_X0 + L.RAIL_X1) / 2, side * L.RAIL_Y, L.TOP + L.RAIL_H / 2),
                      (L.RAIL_X1 - L.RAIL_X0, L.RAIL_W, L.RAIL_H)))
    boxes.append((((L.BED_X0 + L.INSIDE_X1) / 2, 0.0, L.TOP + L.BED_TOP / 2),
                  (L.INSIDE_X1 - L.BED_X0, 2 * L.INSIDE_Y, L.BED_TOP)))
    return rigstate.boxes_distance(boxes)


# -- the sand -----------------------------------------------------------------------------------------------------------
class Sand:
    """The height model: ``h`` (NX, NY) depths (m), the per-column ``heap`` the board carries, ``spoil`` lost over the
    near edge. ``touched`` is where the sole has been on the sand, ``touching`` where it is now."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.h = np.array(L.start_heights(), dtype=float)
        self.heap = np.zeros(L.NY)
        self.spoil = 0.0
        self.touched = np.zeros((L.NX, L.NY), dtype=bool)
        self.touching = np.zeros((L.NX, L.NY), dtype=bool)
        self.pressed = np.zeros((L.NX, L.NY), dtype=bool)
        self.heading = -1.0  # the board's direction of travel along x (+1 away from the robot)

    def volume(self) -> float:
        return float(self.h[_SAND].sum() + self.heap.sum() + self.spoil)

    def at(self, pos, rot, heading: float) -> None:
        """One pose of the board on the sand."""
        self.touching[:] = False
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
            return
        if rot[2, 2] <= math.cos(math.radians(L.TIP_DEG)):
            self._tipped_press(pos, rot)
            return
        inside, sole = under_sole(pos, rot, _X, _Y)
        sole = np.maximum(sole, 0.0)
        inside &= _SAND
        cut = inside & (self.h > sole)
        if cut.any():
            vol = np.where(cut, self.h - sole, 0.0)
            self.heap += vol.sum(axis=0)
            self.h = np.where(cut, sole, self.h)
        ride = inside & (sole <= L.RAIL_H + L.RIDE_TOL + L.EPS)
        for j in np.nonzero((self.heap > 0) & ride.any(axis=0))[0]:
            for i in (range(L.NX) if heading < 0 else range(L.NX - 1, -1, -1)):
                if ride[i, j] and self.h[i, j] < sole[i, j] and self.heap[j] > 0:
                    d = min(float(sole[i, j] - self.h[i, j]), float(self.heap[j]))
                    self.h[i, j] += d
                    self.heap[j] -= d
        cols = inside.any(axis=0)
        lifted = cols & (np.where(inside, sole, np.inf).min(axis=0) > L.RAIL_H + L.RIDE_TOL + L.EPS)
        for j in np.nonzero(lifted & (self.heap > 0))[0]:
            x_t = pos[0] + heading * (_HX + L.CELL / 2)
            i_t = int(math.floor((x_t - L.BED_X0) / L.CELL))
            if i_t < 0:
                self.spoil += float(self.heap[j])
            else:
                self.h[min(i_t, L.NX - 1), j] += float(self.heap[j])
            self.heap[j] = 0.0
        if heading < 0 and pos[0] - _HX <= L.BED_X0 + 0.005 + L.EPS:
            self.spoil += float(self.heap.sum())
            self.heap[:] = 0.0
        self.touching = inside & (np.abs(self.h - sole) <= L.CONTACT_TOL + L.EPS)
        self.touched |= self.touching
        self.pressed &= ~self.touching  # passed again: the cell is not 'finished' any more

    def _tipped_press(self, pos, rot) -> None:
        """A board on its side or flat cuts nothing and drops its heap where it is; any of its surface within 2 mm of a
        finished cell's sand presses that cell."""
        if self.heap.any():  # what it carried falls where it tipped (in the bed) or off the near edge (the spoil)
            i_t = int(math.floor((float(pos[0]) - L.BED_X0) / L.CELL))
            if i_t < 0:
                self.spoil += float(self.heap.sum())
            else:
                self.h[min(i_t, L.NX - 1), :] += self.heap  # the rail columns never carry any
            self.heap[:] = 0.0
        fin = self.finished()
        if not fin.any():
            return
        w = pos + _SURFACE @ rot.T
        i = np.floor((w[:, 0] - L.BED_X0) / L.CELL).astype(int)
        j = np.floor((w[:, 1] - L.BED_Y0) / L.CELL).astype(int)
        ok = (i >= 0) & (i < L.NX) & (j >= 0) & (j < L.NY)
        i, j, z = i[ok], j[ok], w[ok, 2]
        hit = fin[i, j] & (z <= self.h[i, j] + L.TOUCH_TOL + L.EPS)
        self.pressed[i[hit], j[hit]] = True

    def within(self, tol: float) -> int:
        """How many sand cells are within ``tol`` of 25 mm."""
        return int((np.abs(self.h[_SAND] - L.BED_TARGET) <= tol + L.EPS).sum())

    def finished(self) -> np.ndarray:
        return self.touched & ~self.touching & _SAND

    def press(self, read_arms) -> None:
        """A pad tip at or in the sand of a finished cell."""
        fin = self.finished()
        if not fin.any():
            return
        for tcp, tips, opening in read_arms:
            rr = rigstate.zone_radius(opening)
            dx = np.maximum(np.abs(_X - tips[0]) - L.CELL / 2, 0.0)
            dy = np.maximum(np.abs(_Y - tips[1]) - L.CELL / 2, 0.0)
            near = np.hypot(dx, dy) <= rr
            self.pressed |= fin & near & ((tips[2] - L.TOP) <= self.h + L.TOUCH_TOL + L.EPS)

    # -- readings
    def max_depth(self) -> float:
        return float(self.h[_SAND].max())

    def min_depth(self) -> float:
        return float(self.h[_SAND].min())

    def deviation(self) -> float:
        return float(np.abs(self.h[_SAND] - L.BED_TARGET).max())


def tier(latched: bool, pull_ok: bool, cut: bool, dev: float, is_stowed: bool, settled: bool, pressed: bool) -> float:
    if latched or not pull_ok or not cut:
        return 0.0
    score = 0.25
    if dev <= L.TOL_LOOSE + L.EPS:
        score = 0.5
        if dev <= L.TOL_MED + L.EPS and is_stowed:
            score = 0.75
            if dev <= L.TOL_TIGHT + L.EPS and settled:
                score = 1.0
    return min(score, 0.5) if pressed else score


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _nlerp(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    if float(q0 @ q1) < 0:
        q1 = -q1
    q = (1 - f) * q0 + f * q1
    return q / np.linalg.norm(q)


class Tracker:
    """The sand, the latches, the pull record, the last 1 s of the board (stillness) and of the let-go check."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.sand = Sand()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.sand.reset()
        self.latch: dict = {"F1": None}
        self.best_pull = 0.0
        self.run: list | None = None  # x range of the current bridged run
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.blind = False  # the arms could not be read while there was sand to press
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return float(np.linalg.norm(r.pos - np.array([L.BOARD_START[0], L.BOARD_START[1], L.BOARD_START[2] - L.TOP]))
                     ) <= L.START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The board put back on its starting mark with a jump no trial makes in one step (a reset): further than its
        last velocity and a free fall could carry it in one control step."""
        dt = 1.0 / L.CONTROL_HZ
        reach = float(np.linalg.norm(prev.vel)) * dt + 0.5 * L.G * dt * dt
        return cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) - reach >= L.RESET_JUMP

    def _advance(self, r: Reading) -> None:
        """Move the board from the last reading to ``r`` through the sand, in steps of at most 4 mm."""
        if self.prev is None:
            self.sand.at(r.pos, r.rot, self.sand.heading)
            return
        dx = float(r.pos[0] - self.prev.pos[0])
        if abs(dx) > 0.0002:
            self.sand.heading = 1.0 if dx > 0 else -1.0
        d = float(np.max(np.abs(r.pos - self.prev.pos)))
        n = int(min(60, max(1, math.ceil(d / SUB_STEP))))
        for k in range(1, n + 1):
            f = k / n
            q = _nlerp(self.prev.quat, r.quat, f)
            self.sand.at(self.prev.pos + (r.pos - self.prev.pos) * f, rotation(q), self.sand.heading)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.run = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self._advance(r)
        self.hist.append(r)
        self.prev = r
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = self.steps
        if bridged(r):
            x = float(r.pos[0])
            self.run = [x, x] if self.run is None else [min(self.run[0], x), max(self.run[1], x)]
            self.best_pull = max(self.best_pull, self.run[1] - self.run[0])
        else:
            self.run = None
        read_arms = rigstate.read_arms(arms)
        if read_arms is None:
            self.blind = self.blind or bool(self.sand.finished().any())
        else:
            self.sand.press(read_arms)
        ok, gap = False, None
        try:
            dist = hands_zone(r)
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the board's largest move (m) and turn (deg) from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos - first.pos)))
            turn = max(turn, turn_deg(first.rot, x.rot))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        s = self.sand
        pull_ok = self.best_pull >= L.BRIDGE_PULL - L.EPS
        tall = s.max_depth() <= L.TALL_MAX + L.EPS
        n_cut = s.within(L.TOL_LOOSE)
        cut = n_cut >= L.CUT_CELLS
        dev = s.deviation()
        is_stowed = stowed(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        pressed = bool(s.pressed.any()) or self.blind
        score = tier(self.latched, pull_ok, cut, dev, is_stowed, released and still, pressed)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch),
            "pull_mm": _mm(self.best_pull), "bridged_now": bridged(r), "cut": cut, "cells_within_6mm": n_cut,
            "tall": tall,
            "max_depth_mm": _mm(s.max_depth()), "min_depth_mm": _mm(s.min_depth()), "dev_mm": _mm(dev),
            "cells_off_3mm": int((np.abs(s.h[_SAND] - L.BED_TARGET) > L.TOL_TIGHT + L.EPS).sum()),
            "heap_mm": _mm(float(s.heap.sum())), "spoil_mm": _mm(s.spoil),
            "stowed": is_stowed, "pressed_cells": int(s.pressed.sum()), "blind": self.blind,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "board_mm": [round(float(v) * 1000.0, 2) for v in r.pos], "board_yaw_deg": round(yaw_deg(r.rot), 2),
            "board_tilt_deg": round(tilt_deg(r.rot), 2),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(score, bool(score == 1.0), details)
