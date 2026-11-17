"""The scorer, as plain functions of the seven books' states and the arms. NumPy only; no simulator needed.

The rule is about the end state and about the whole trial: no book may have fallen, the six neighbours must stay in
place and upright, and both grippers must have let go with every book still for the last second. So the scorer keeps
a little state (the largest tilt each book has reached, the new book's counted only while it is not held in a gripper, the last second of poses and of let-go checks) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(books, arms) -> Result(score, success, details)

``books`` maps the seven names in ``layout.BOOKS`` to anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; ``arms`` is ``rigstate.arms()``.
Every book is a plain box whose size is in ``layout``, so its faces follow from ``pos`` and ``quat_xyzw``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

_SIGNS = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(quat_xyzw) -> float:
    """Angle between the book's up axis and the world's vertical (degrees)."""
    return float(np.degrees(np.arccos(np.clip(rotation(quat_xyzw)[2, 2], -1.0, 1.0))))


def yaw_deg(quat_xyzw) -> float:
    """How far the book's spine-to-back axis is turned about the vertical from its start heading, in (-180, 180]."""
    r = rotation(quat_xyzw)
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def turn_deg(q0, q1) -> float:
    """The angle between two orientations (degrees)."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (np.isfinite(na) and np.isfinite(nb)) or na < 1e-9 or nb < 1e-9:
        return float("inf")
    d = abs(float(np.dot(a / na, b / nb)))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0))))


def finite(*states) -> bool:
    """Every number the scorer reads is finite (a simulator that blew up reports NaN or inf)."""
    return all(
        np.all(np.isfinite(np.asarray(getattr(s, k), dtype=float)))
        for s in states for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
    )


def _half(name: str) -> np.ndarray:
    return np.array(L.SIZE[name], dtype=float) / 2


def underside_centre(name: str, s) -> np.ndarray:
    """The centre of the book's underside (world): where it stands on the shelf."""
    return np.asarray(s.pos, dtype=float) - rotation(s.quat_xyzw)[:, 2] * _half(name)[2]


def underside_centre_z(name: str, s) -> float:
    """Height of the centre of the book's underside (z)."""
    return float(underside_centre(name, s)[2])


def underside_corners(name: str, s) -> np.ndarray:
    """The four corners of the book's underside (world), (4, 3)."""
    r, c, h = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float), _half(name)
    return np.array([c + r @ np.array([sx * h[0], sy * h[1], -h[2]]) for sx, sy in _SIGNS])


def spine_x(name: str, s) -> float:
    """The x of the bottom edge of the book's spine (the face toward the robot): where a rule laid along the shelf
    meets it."""
    return float((underside_centre(name, s) - rotation(s.quat_xyzw)[:, 0] * _half(name)[0])[0])


def spine_line(books) -> float:
    """The row's spine line: the mean spine x of the two neighbours beside the gap, C and D."""
    return 0.5 * (spine_x("book_c", books["book_c"]) + spine_x("book_d", books["book_d"]))


def gap_mid_y(books) -> float:
    """The gap's centre line: halfway between the centres of C's and D's undersides (the same thickness)."""
    return 0.5 * (float(underside_centre("book_c", books["book_c"])[1])
                  + float(underside_centre("book_d", books["book_d"])[1]))


def gap_faces(books) -> tuple[float, float]:
    """y of C's face toward the gap and of D's face toward the gap."""
    return (float(underside_centre("book_c", books["book_c"])[1]) + L.NB_T / 2,
            float(underside_centre("book_d", books["book_d"])[1]) - L.NB_T / 2)


def clearances(books) -> tuple[float, float]:
    """The new book's smallest bottom-corner distance to C's face, and to D's face (m; negative: it overlaps)."""
    fc, fd = gap_faces(books)
    ys = underside_corners(L.NEW, books[L.NEW])[:, 1]
    return float(ys.min() - fc), float(fd - ys.max())


def row_shift(name: str, s) -> float:
    """Horizontal distance of a neighbour's centre from its starting place."""
    x, y = L.NB_START[name]
    return float(np.hypot(s.pos[0] - x, s.pos[1] - y))


def on_board(s) -> bool:
    """The book's centre is over the shelf board."""
    return bool(L.SHELF_X0 <= s.pos[0] <= L.SHELF_X1 and abs(s.pos[1] - L.Y0) <= L.ROW_HALF + L.PANEL_GAP)


def off_shelf(name: str, s) -> bool:
    return bool(not on_board(s) or underside_centre_z(name, s) < L.SHELF_TOP - L.OFF_SHELF_DROP)


# -- the rules ---------------------------------------------------------------------------------------------------
def seated(books) -> bool:
    """The new book stands square in the gap: its underside on the shelf, centre within 3 mm of the gap's centre line,
    every bottom corner 2 mm or more clear of C and of D, tilted under 3 degrees, turned under 3 degrees (spine out), its
    spine within 5 mm of the row's spine line."""
    n = books[L.NEW]
    cc, cd = clearances(books)
    return bool(
        cc >= L.SEAT_CLEAR_MIN and cd >= L.SEAT_CLEAR_MIN
        and abs(underside_centre_z(L.NEW, n) - L.SHELF_TOP) <= L.SEAT_Z_TOL
        and abs(float(underside_centre(L.NEW, n)[1]) - gap_mid_y(books)) <= L.SEAT_Y_TOL
        and tilt_deg(n.quat_xyzw) < L.SEAT_TILT_DEG
        and abs(yaw_deg(n.quat_xyzw)) < L.SEAT_YAW_DEG
        and abs(spine_x(L.NEW, n) - spine_line(books)) <= L.SEAT_SPINE_TOL
    )


def in_gap(books) -> bool:
    """The new book stands in the gap: all four corners of its underside between C's and D's faces, the underside on
    the shelf (within 3 mm), tilted under 10 degrees, turned under 15 degrees, its spine within 40 mm of the line."""
    n = books[L.NEW]
    cc, cd = clearances(books)
    return bool(
        cc > 0.0 and cd > 0.0
        and abs(underside_centre_z(L.NEW, n) - L.SHELF_TOP) <= L.SEAT_Z_TOL
        and tilt_deg(n.quat_xyzw) < L.GAP_TILT_DEG
        and abs(yaw_deg(n.quat_xyzw)) < L.GAP_YAW_DEG
        and abs(spine_x(L.NEW, n) - spine_line(books)) <= L.GAP_SPINE_TOL
    )


def row_upright(books, max_tilt: dict) -> bool:
    """Each of A to F tilted under 5 degrees at the end, never over 20 degrees, and within 6 mm of its place."""
    return all(
        tilt_deg(books[n].quat_xyzw) < L.ROW_TILT_END_DEG
        and max_tilt.get(n, 0.0) <= L.ROW_TILT_MAX_DEG
        and row_shift(n, books[n]) <= L.ROW_SHIFT
        for n in L.NEIGHBOURS
    )


def row_leaning(books) -> bool:
    """Each of A to F tilted under 20 degrees at the end and within 20 mm of its place (a neighbour may lean)."""
    return all(
        tilt_deg(books[n].quat_xyzw) < L.ROW_LEAN_END_DEG and row_shift(n, books[n]) <= L.ROW_LEAN_SHIFT
        for n in L.NEIGHBOURS
    )


def book_distance(books):
    """A distance function from world points to the union of the seven books (0 inside), for rigstate.released."""
    return rigstate.boxes_distance(
        [(np.asarray(books[n].pos, dtype=float), np.array(L.SIZE[n], dtype=float), rotation(books[n].quat_xyzw))
         for n in L.BOOKS]
    )


def new_held(books, arms) -> bool:
    """A gripper holds the new book: its finger zone within HOLD_GAP of it, its pads less than HOLD_WIDTH apart.
    Unreadable arms or a distance that cannot be computed: not held (fail closed, a tilted book then counts as fallen)."""
    try:
        read = rigstate.read_arms(arms)
        if read is None:
            return False
        n = books[L.NEW]
        dist = rigstate.boxes_distance(
            [(np.asarray(n.pos, dtype=float), np.array(L.SIZE[L.NEW], dtype=float), rotation(n.quat_xyzw))])
        for a in read:
            if a[2] * rigstate.FULL_GAP < L.HOLD_WIDTH:
                g = rigstate.zone_gap(a, dist)
                if np.isfinite(g) and g <= L.HOLD_GAP:
                    return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _mm(v: float) -> float | None:
    return round(v * 1000, 1) if np.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window_steps = int(round(L.HANDS_S * control_hz))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        # degrees, per book: the neighbours' largest tilt ever; the new book's largest tilt while NOT held
        self.max_tilt = {n: 0.0 for n in L.BOOKS}
        self.max_tilt_held = 0.0  # the new book's largest tilt while held (not scored, kept for the details)
        self.off_table = None  # step at which the new book first went more than OFF_TABLE below the table top
        self.bad_steps = 0  # observations with NaN or inf in them (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) per book] at the previous control step
        self.window: deque = deque(maxlen=self.window_steps)  # per step: (released, {name: (centre, quat)})
        self.resets = 0

    @staticmethod
    def at_start(books) -> bool:
        """The starting layout: every book on its mark, upright and still."""
        for n in L.NEIGHBOURS:
            if row_shift(n, books[n]) > 0.003 or tilt_deg(books[n].quat_xyzw) > 1.0:
                return False
        n = books[L.NEW]
        return bool(
            np.hypot(n.pos[0] - L.NEW_START[0], n.pos[1] - L.NEW_START[1]) < 0.003
            and tilt_deg(n.quat_xyzw) < 1.0
            and all(np.linalg.norm(books[b].lin_vel) < 0.05 for b in L.BOOKS)
        )

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.G * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def update(self, books, arms) -> Result:
        states = [books[n] for n in L.BOOKS]
        if not finite(*states):  # a broken observation: count it, change nothing else, score it 0
            self.bad_steps += 1
            self.window.clear()  # a gap in the record is not a still second
            self.last = Result(0.0, False, {"bad_state": True, "bad_steps": self.bad_steps})
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in states]
        if self.prev is not None and self._teleported(now) and self.at_start(books):
            self.reset()  # the simulator put everything back on its marks: a new episode
            self.resets += 1
        self.prev = now
        self.steps += 1
        for n in L.NEIGHBOURS:
            self.max_tilt[n] = max(self.max_tilt[n], tilt_deg(books[n].quat_xyzw))
        new_tilt = tilt_deg(books[L.NEW].quat_xyzw)
        if new_held(books, arms):
            self.max_tilt_held = max(self.max_tilt_held, new_tilt)
        else:
            self.max_tilt[L.NEW] = max(self.max_tilt[L.NEW], new_tilt)
        if self.off_table is None and float(np.asarray(books[L.NEW].low, dtype=float).ravel()[2]) < L.TOP - L.OFF_TABLE:
            self.off_table = self.steps
        try:
            clear = rigstate.released(arms, book_distance(books), L.HANDS_CLEAR)
        except (ValueError, TypeError, AttributeError):
            clear = False  # fail closed: a gripper that cannot be placed is not clear
        self.window.append((bool(clear), {n: (np.array(books[n].pos, float), np.array(books[n].quat_xyzw, float))
                                          for n in L.BOOKS}))
        self.last = self.result(books)
        return self.last

    # -- the last second --------------------------------------------------------------------------------------
    def hands_off(self) -> bool:
        """Every one of the last second's steps had both grippers' finger zones 30 mm or more from every book."""
        return len(self.window) == self.window_steps and all(c for c, _ in self.window)

    def still(self) -> bool:
        """Over the last second no book's centre moved 5 mm and none turned 5 degrees (from the first step of it): motion a person can see."""
        if len(self.window) < self.window_steps:
            return False
        first = self.window[0][1]
        for _, poses in self.window:
            for n in L.BOOKS:
                p0, q0 = first[n]
                p, q = poses[n]
                if float(np.linalg.norm(p - p0)) > L.STILL_MOVE or turn_deg(q0, q) > L.STILL_TURN_DEG:
                    return False
        return True

    @property
    def fell(self) -> bool:
        """A neighbour was ever tilted over 45 degrees, or the new book was while not held."""
        return any(v > L.FALL_DEG for v in self.max_tilt.values())

    def result(self, books) -> Result:
        states = [books[n] for n in L.BOOKS]
        if not finite(*states):
            return Result(0.0, False, {"bad_state": True, "bad_steps": self.bad_steps})
        n = books[L.NEW]
        cc, cd = clearances(books)
        is_seated, is_in_gap = seated(books), in_gap(books)
        full, leaning = row_upright(books, self.max_tilt), row_leaning(books)
        gone = [b for b in L.NEIGHBOURS if off_shelf(b, books[b])]
        hands, still = self.hands_off(), self.still()
        if self.fell or self.off_table is not None or gone or not (hands and still):
            score = 0.0
        elif is_seated and full:
            score = 1.0
        elif is_in_gap and full:
            score = 0.5
        elif is_in_gap and leaning:
            score = 0.25
        else:
            score = 0.0
        details = {
            "seated": is_seated, "in_gap": is_in_gap, "row_upright": full,
            "row_leaning": leaning, "fell": self.fell, "off_table": self.off_table is not None,
            "new_max_tilt_unheld_deg": round(self.max_tilt[L.NEW], 2),
            "new_max_tilt_held_deg": round(self.max_tilt_held, 2), "neighbours_off_shelf": gone,
            "hands_off": hands, "still": still,
            "new_y_off_mm": _mm(float(underside_centre(L.NEW, n)[1]) - gap_mid_y(books)),
            "clear_c_mm": _mm(cc), "clear_d_mm": _mm(cd),
            "new_tilt_deg": round(tilt_deg(n.quat_xyzw), 2), "new_yaw_deg": round(yaw_deg(n.quat_xyzw), 2),
            "new_spine_off_mm": _mm(spine_x(L.NEW, n) - spine_line(books)),
            "new_underside_mm": _mm(underside_centre_z(L.NEW, n) - L.SHELF_TOP),
            "row_max_tilt_deg": round(max(self.max_tilt[b] for b in L.NEIGHBOURS), 2),
            "row_end_tilt_deg": round(max(tilt_deg(books[b].quat_xyzw) for b in L.NEIGHBOURS), 2),
            "row_max_shift_mm": _mm(max(row_shift(b, books[b]) for b in L.NEIGHBOURS)),
        }
        return Result(score, bool(score == 1.0), details)
