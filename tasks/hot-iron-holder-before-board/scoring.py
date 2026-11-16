"""The scorer, as plain functions of the iron's, the board's, the holder's and the arms' states. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step.

    METAL       the iron's visible metal front: tip, heater barrel and nut, 0 to 95 mm from the tip (layout.py)
    HAZARD      latched 0 for the rest of the trial, any of:
                - a gripper's finger zone (rigstate: the space its fingers can take at their opening) reaches the
                  metal front;
                - the metal front within 1 mm of the board, the cord, the station, the holder's base or posts, or
                  the bench top anywhere off the mat (the mat and the inside of the cage may take it);
                - the iron or the board 5 cm or more below the table top (off the table);
                - the holder tipped more than 20 deg or moved more than 2 cm from its place;
                - the cord pulled: its station end and the iron's rear 95 % of its 1.2 m apart.
    IN HOLDER   the metal front's axis points at 0, 30, 80 and 95 mm from the tip all inside the cage (0 to 125 mm
                along its axis from the mouth, within 15 mm of the axis), the iron's tip direction within 10 deg of
                the cage axis, and no finger zone within 5 mm of the iron (it rests, not held)
    PLACED      all four standoff feet within 2 mm of the mat's top and over the mat, the board's centre within
                40 mm of the mat's centre (NEAR: within 100 mm)
    CLEAR       every finger zone of both grippers 3 cm or more from the iron, the cord and the board
    STILL       over the last 20 steps (1 s) the iron's tip and rear and the board's centre each moved 1 mm or less

    1.0   IN HOLDER, PLACED and CLEAR on each of the last 20 steps, and STILL
    0.75  IN HOLDER and the board standing on the mat (NEAR), not 1.0 (centre 40 to 100 mm out, or a gripper near)
    0.5   IN HOLDER, the board not on the mat
    0     anything else (the iron not in its holder earns nothing, whatever the board does), or a HAZARD at any
          step so far.

Fail closed: a step whose states or arms cannot be read scores 0 and restarts the 20-step window.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

IRON, BOARD, HOLDER = "iron", "board", "holder"
METAL_CHECK_S = (0.0, 0.030, 0.080, 0.095)


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


# -- geometry -----------------------------------------------------------------------------------------------------
def cylinder_distance(points, tip, axis, s0: float, s1: float, r: float) -> np.ndarray:
    """Distance from world points to a solid cylinder on the iron's axis from s0 to s1 (0 inside)."""
    p = np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(tip, dtype=float)
    a = np.asarray(axis, dtype=float)
    s = p @ a
    radial = np.linalg.norm(p - np.outer(s, a), axis=1)
    ds = np.maximum(np.maximum(s0 - s, s - s1), 0.0)
    dr = np.maximum(radial - r, 0.0)
    return np.hypot(ds, dr)


def sections_distance(iron, sections):
    tip, axis = np.asarray(iron.tip, dtype=float), np.asarray(iron.axis, dtype=float)

    def dist(points):
        return np.min([cylinder_distance(points, tip, axis, s0, s1, r) for _, s0, s1, r, _ in sections], axis=0)

    return dist


def metal_distance(iron):
    return sections_distance(iron, L.METAL_SECTIONS)


def iron_distance(iron):
    return sections_distance(iron, L.IRON_SECTIONS)


def metal_samples(iron, step: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Points on the metal front's axis every ``step`` and the radius there."""
    s = np.arange(0.0, L.METAL_END + 1e-9, step)
    radii = np.array([next(r for _, s0, s1, r, _ in L.METAL_SECTIONS if s0 - 1e-9 <= v <= s1 + 1e-9) for v in s])
    pts = np.asarray(iron.tip, dtype=float) + np.outer(s, np.asarray(iron.axis, dtype=float))
    return pts, radii


def cord_samples(iron, step: float = 0.005) -> np.ndarray:
    """Points along the cord's centreline (anchor, segment centres, the iron's rear) about ``step`` apart."""
    line = iron.cord_polyline()
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


def cord_distance(iron):
    pts = cord_samples(iron)

    def dist(points):
        q = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        return np.maximum(d - L.CORD_R, 0.0)

    return dist


def _holder_boxes():
    return [(np.asarray(c, dtype=float), np.asarray(s, dtype=float))
            for kind, c, s, ypr, _, part in L.holder_rows() if part in ("base", "post")]


HOLDER_BOXES = _holder_boxes()
STATION_BOX = (np.array([L.STATION_C[0], L.STATION_C[1], L.TOP + L.STATION_SIZE[2] / 2]), np.asarray(L.STATION_SIZE))
MOUTH_LOCAL = L.MOUTH - np.array([L.HOLDER_ORIGIN[0], L.HOLDER_ORIGIN[1], 0.0])


def in_holder_geometry(iron, holder) -> tuple[bool, dict]:
    """The metal front inside the cage, coaxial within 10 deg (holder frame)."""
    a_world = holder.rotation() @ L.CAGE_AXIS
    tip_dir = -np.asarray(iron.axis, dtype=float)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(tip_dir @ a_world)))))
    us, rs = [], []
    for s in METAL_CHECK_S:
        rel = holder.to_local(iron.at(s)) - MOUTH_LOCAL
        u = float(rel @ L.CAGE_AXIS)
        us.append(u)
        rs.append(float(np.linalg.norm(rel - u * L.CAGE_AXIS)))
    inside = all(-1e-9 <= u <= L.CAGE_LEN + 1e-9 for u in us) and all(r <= L.IN_CAGE_R + 1e-9 for r in rs)
    ok = inside and ang <= L.AXIS_DEG + 1e-9
    return ok, {"axis_deg": round(ang, 1), "u_mm": [round(u * 1000, 1) for u in us],
                "radial_mm": [round(r * 1000, 1) for r in rs]}


def board_feet(board) -> np.ndarray:
    return board.to_world(np.array([(sx, sy, 0.0) for sx, sy in L.STANDOFF_XY]))


def board_centre(board) -> np.ndarray:
    return board.to_world(np.array([0.0, 0.0, L.BOARD_PLATE_Z]))


def board_on_mat(board) -> tuple[bool, float, list]:
    """(all four feet on the mat, centre's distance from the mat centre, feet heights over the mat in mm)."""
    feet = board_feet(board)
    top = L.TOP + L.MAT_TOP
    gaps = [float(f[2] - top) for f in feet]
    on = all(abs(g) <= L.STANDOFF_TOL + 1e-9 for g in gaps) and all(L.over_mat(f[0], f[1]) for f in feet)
    c = board_centre(board)
    off = math.hypot(c[0] - L.MAT_C[0], c[1] - L.MAT_C[1])
    return on, off, [round(g * 1000, 1) for g in gaps]


def hazards(states, arms, holder_spawn) -> list[str]:
    """Every hazard of this step (empty: none). Raises on unreadable states (the caller fails closed)."""
    iron, board, holder = states[IRON], states[BOARD], states[HOLDER]
    out = []
    metal = metal_distance(iron)
    read = rigstate.read_arms(arms)
    if read is None:
        out.append("arms unreadable")
    else:
        for side, a in zip(rigstate.ARMS, read):
            if rigstate.zone_gap(a, metal) < 0.0:
                out.append(f"{side} gripper on the hot metal")
    pts, radii = metal_samples(iron)
    if np.any(board.distance(pts) - radii <= L.CONTACT_M):
        out.append("hot metal on the board")
    if np.any(cord_distance(iron)(pts) - radii <= L.CONTACT_M):
        out.append("hot metal on the cord")
    low = pts[:, 2] - radii
    off_mat = np.array([not L.over_mat(p[0], p[1]) for p in pts])
    if np.any((low <= L.TOP + L.CONTACT_M) & off_mat):
        out.append("hot metal on the bench top")
    if np.any(rigstate.box_distances(pts, *STATION_BOX) - radii <= L.CONTACT_M):
        out.append("hot metal on the station")
    local = holder.to_local(pts)
    hb = np.min([rigstate.box_distances(local, c, s) for c, s in HOLDER_BOXES], axis=0)
    if np.any(hb - radii <= L.CONTACT_M):
        out.append("hot metal on the holder's base")
    for name, z in (("iron", min(float(iron.tip[2]), float(iron.rear[2]))),
                    ("board", float(np.asarray(board.low, dtype=float)[2]))):
        if z < L.TOP - L.OFF_TABLE_M:
            out.append(f"the {name} left the table")
    if math.degrees(holder.tilt()) > L.HOLDER_TILT_DEG:
        out.append("holder tipped")
    o = np.asarray(holder.origin, dtype=float)
    if math.hypot(o[0] - holder_spawn[0], o[1] - holder_spawn[1]) > L.HOLDER_MOVE_M:
        out.append("holder moved")
    if float(np.linalg.norm(np.asarray(iron.rear) - np.asarray(iron.anchor))) >= L.CORD_TAUT * L.CORD_LEN:
        out.append("cord pulled taut")
    return out


class Tracker:
    def __init__(self, holder_spawn=(L.HOLDER_ORIGIN[0], L.HOLDER_ORIGIN[1])) -> None:
        self.holder_spawn = holder_spawn
        self.last = Result()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.latched = ""
        self.last = Result()

    def _unreadable(self, why: str) -> Result:
        self.win.clear()
        self.last = Result(0.0, False, {"unreadable": why, "latched": self.latched})
        return self.last

    def update(self, states, arms) -> Result:
        self.steps += 1
        try:
            iron, board, holder = states[IRON], states[BOARD], states[HOLDER]
            nums = np.concatenate([np.ravel(iron.tip), np.ravel(iron.axis), np.ravel(iron.points),
                                   np.ravel(board.origin), np.ravel(board.quat_xyzw), np.ravel(holder.origin),
                                   np.ravel(holder.quat_xyzw)]).astype(float)
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._unreadable("state")
        if not np.all(np.isfinite(nums)):
            return self._unreadable("numbers")
        try:
            haz = hazards(states, arms, self.holder_spawn)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return self._unreadable(f"hazard check: {exc!r}")
        if "arms unreadable" in haz:
            return self._unreadable("arms")
        if haz and not self.latched:
            self.latched = "; ".join(haz)
        read = rigstate.read_arms(arms)
        held_gap = min(rigstate.zone_gap(a, iron_distance(iron)) for a in read)
        geo, hd = in_holder_geometry(iron, holder)
        in_h = geo and held_gap >= L.HELD_M - 1e-9
        on, off, feet_mm = board_on_mat(board)
        placed, near = on and off <= L.CENTRE_OK + 1e-9, on and off <= L.CENTRE_NEAR + 1e-9
        clear_gap = min(min(rigstate.zone_gap(a, iron_distance(iron)), rigstate.zone_gap(a, cord_distance(iron)),
                            rigstate.zone_gap(a, board.distance)) for a in read)
        clear = clear_gap >= L.CLEAR_M - 1e-9
        ok1 = in_h and placed and clear
        marks = (np.asarray(iron.tip, dtype=float), np.asarray(iron.rear, dtype=float), board_centre(board))
        self.win.append((ok1, marks))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            first = self.win[0][1]
            move = max(float(np.linalg.norm(m - f)) for w in self.win for m, f in zip(w[1], first))
            still = move <= L.STILL_M + 1e-12
        if self.latched:
            score = 0.0
        elif full and still:
            score = 1.0
        elif in_h and near:
            score = 0.75
        elif in_h:
            score = 0.5
        else:
            score = 0.0
        details = {"in_holder": in_h, "cage": hd, "held_gap_mm": round(held_gap * 1000, 1), "board_on_mat": on,
                   "board_off_mm": round(off * 1000, 1), "feet_mm": feet_mm, "placed": placed,
                   "clear_gap_mm": round(clear_gap * 1000, 1), "clear": clear, "still": still, "window_ok": full,
                   "hazards_now": haz, "latched": self.latched}
        self.last = Result(score, score >= 1.0, details)
        return self.last
