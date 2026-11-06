"""The scorer, as plain functions of the riser's and the print's states and the arms' state. NumPy only; no
simulator needed.

The score is read from the end state: the riser lying flat on the backing board, the print face up on top of it,
the riser set back at least 10 mm from every edge of the print (hidden), the print centred and square on the board,
and both grippers clear. Two things need a little history, kept by a tracker fed one observation per control step
by ``task.success()``: whether each part is at rest (its path over the last 0.5 s, never its reported velocity,
which chatters at 10-45 mm/s while a gripped part stays put) and when a new episode starts.

    Tracker.update(riser, print_, arms) -> Result(score, success, details)

``riser`` and ``print_`` are anything with ``pos`` (bounding-box centre, which for a box is its centre),
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. All the
geometry is computed from ``pos``, ``quat_xyzw`` and the parts' known sizes (layout.py). ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'clear' is taken as met for that
step and the step is counted in ``arms_unavailable`` (a policy cannot cause that; the GPU runs check it stays 0).
An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and
leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is a solid, flat-ended cylinder about the axis from the point between its finger tips 38 mm back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn, and its
lower end is the tips' plane (there is nothing of the fingers below their tips), so 'clear' is measured from the pads
in every direction: finger tips 30 mm straight above the print are 30 mm from it, however far the gripper is open.

Every bound the text states as inclusive ('or more', 'or less', 'within', '3 to 7 mm') is compared with a tolerance
of 1e-9 (metres or degrees), so a value computed exactly on a stated bound lands in the tier the text gives it, not
one lower by floating-point rounding. Bounds stated as strict ('shorter than', 'less than', 'more than') stay strict.
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
EPS = 1e-9  # m or degrees: a value computed exactly on an inclusive bound counts as within it


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
                and all(p.size == 3 for p in parts[2:]))


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


# -- the parts' geometry -----------------------------------------------------------------------------------------
def corners(s, size) -> np.ndarray:
    """The eight corners of a box of ``size`` (world, 8 x 3) from its centre and orientation."""
    half = np.array(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def lowest(s, size) -> float:
    """Height of the part's lowest point above the table top (m)."""
    return float(corners(s, size)[:, 2].min() - L.TOP)


def off_table(s, size) -> bool:
    """The part's lowest point more than 10 mm below the table top: it fell off the table."""
    return bool(lowest(s, size) < -L.OFF_TABLE_DROP)


def riser_tilt_deg(riser) -> float:
    """Angle between the vertical and the riser's 10 mm axis (0..90; either broad face may be down)."""
    zz = rotation(riser.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def print_tilt_deg(print_) -> float:
    """Angle between straight up and the print's face normal, body +z (0..180: face down is 180)."""
    zz = rotation(print_.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def riser_yaw_deg(riser) -> float:
    """Heading of the riser's 30 mm width (body x) seen from above (degrees, -180..180): 0 as it starts (its 70 mm
    length along y), -90 or +90 laid lengthwise along x as the demo leaves it. Reported for the GPU trace only."""
    r = rotation(riser.quat_xyzw)
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def riser_flat(riser) -> bool:
    """The riser's 10 mm axis within 10 degrees of vertical (either broad face down)."""
    return bool(riser_tilt_deg(riser) <= L.RISER_FLAT_DEG + EPS)


def print_face_up(print_) -> bool:
    """The print's face normal within 3 degrees of straight up."""
    return bool(print_tilt_deg(print_) <= L.PRINT_FACE_UP_DEG + EPS)


def riser_on_board(riser) -> bool:
    """Every corner of the riser, seen from above, lies within the backing board's outline."""
    c = corners(riser, L.RISER_SIZE)
    return bool(np.all(np.abs(c[:, 0] - L.BOARD_X) <= L.BOARD_HALF_X + EPS)
                and np.all(np.abs(c[:, 1] - L.BOARD_Y) <= L.BOARD_HALF_Y + EPS))


def print_heading(print_) -> float:
    """Heading of the print's 110 mm sides (body x) seen from above (rad)."""
    r = rotation(print_.quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def in_print_frame(print_, points) -> np.ndarray:
    """Plan coordinates (u along the print's 110 mm sides, v along its 70 mm sides) of world points, (n, 2)."""
    h = print_heading(print_)
    d = np.atleast_2d(np.asarray(points, dtype=float))[:, :2] - np.asarray(print_.pos, dtype=float)[:2]
    ch, sh = math.cos(h), math.sin(h)
    return np.column_stack((d[:, 0] * ch + d[:, 1] * sh, -d[:, 0] * sh + d[:, 1] * ch))


def inset(print_, riser) -> float:
    """In the print's own frame, the smallest signed distance from any corner of the riser's outline seen from
    above to any edge of the print's outline (m): positive when every corner is inside the print, negative when a
    corner sticks out past an edge. Nominal 20 mm; a riser laid across the print, as it starts, gives about 0."""
    uv = in_print_frame(print_, corners(riser, L.RISER_SIZE))
    return float(np.min(np.minimum(L.PRINT_HALF_X - np.abs(uv[:, 0]), L.PRINT_HALF_Y - np.abs(uv[:, 1]))))


def riser_under_print(print_, riser) -> bool:
    """The riser's centre lies inside the print's outline seen from above."""
    u, v = in_print_frame(print_, riser.pos)[0]
    return bool(abs(u) <= L.PRINT_HALF_X + EPS and abs(v) <= L.PRINT_HALF_Y + EPS)


def off_centre(print_) -> float:
    """Plan distance from the print's centre to the board's centre (m)."""
    return float(np.hypot(print_.pos[0] - L.BOARD_X, print_.pos[1] - L.BOARD_Y))


def turned_deg(print_) -> float:
    """Angle between the print's 110 mm sides and the board's 180 mm sides (x), folded to 0..90 degrees: half a
    turn counts as square (the print has no top or bottom edge), a quarter turn is 90."""
    a = math.degrees(print_heading(print_)) % 180.0
    return float(min(a, 180.0 - a))


# -- the arms ----------------------------------------------------------------------------------------------------
def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_axis(arm) -> tuple[np.ndarray, np.ndarray]:
    """The point between the finger tips and the unit vector from there back up the gripper (toward the TCP)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    return tips, (u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0]))


def _perpendiculars(u) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to each other and to the unit vector u."""
    e1 = np.cross(u, np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(u, e1)


_AXIS_S = np.linspace(0.0, L.FINGER_LEN, 9)  # sample points along the zone's axis (from the tips up)
_RING = np.linspace(0.0, 2 * math.pi, 48, endpoint=False)  # and round the rims of its two flat ends
_COS, _SIN = np.cos(_RING), np.sin(_RING)


def _cylinder_gap(tips, u, r, q) -> np.ndarray:
    """Distances from the points q (n x 3) to the solid cylinder of radius r about the axis from ``tips`` along
    ``u`` for FINGER_LEN (negative inside it, by the radial depth). Below the tips' plane the distance is measured
    from that flat end, not from a rounded cap."""
    v = q - tips
    a = v @ u
    rho = np.linalg.norm(v - np.outer(a, u), axis=1)
    da = np.maximum(0.0, np.maximum(-a, a - L.FINGER_LEN))
    dr = rho - r
    return np.where((da == 0.0) & (dr <= 0.0), dr, np.hypot(da, np.maximum(0.0, dr)))


def finger_zone(arm) -> tuple[np.ndarray, float]:
    """Points on the zone's axis and on the rims of its two flat ends (the tips' plane and 38 mm up), and its
    radius."""
    tips, u = finger_axis(arm)
    r = zone_radius(arm)
    e1, e2 = _perpendiculars(u)
    circle = r * (np.outer(_COS, e1) + np.outer(_SIN, e2))
    return np.vstack([tips + np.outer(_AXIS_S, u), tips + circle, tips + L.FINGER_LEN * u + circle]), r


def _closest_on_box(s, size, pts) -> np.ndarray:
    """The point of the part (its real, turned box) nearest to each of the points (n x 3); a point inside it is its
    own nearest point."""
    c, rot = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)
    local = (np.asarray(pts, dtype=float) - c) @ rot  # each row: the point in the part's frame
    h = np.asarray(size, dtype=float) / 2
    return np.clip(local, -h, h) @ rot.T + c


def zone_gap(arm, s, size) -> float:
    """How far a gripper's finger zone is from a part (its real, turned box of ``size``); negative: they overlap.
    The zone is a flat-ended cylinder (the fingers end at their tips), so a gripper whose finger tips are 30 mm
    straight above the print is 30 mm from it whatever its opening. The smallest distance is searched over the
    part's points nearest to the zone's axis and rims, and the part's corners."""
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([_closest_on_box(s, size, pts), corners(s, size)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


def finger_gaps(arms, riser, print_) -> dict[str, float]:
    """Each gripper's finger zone to the nearer of the two parts (m)."""
    return {side: min(zone_gap(arms[side], riser, L.RISER_SIZE), zone_gap(arms[side], print_, L.PRINT_SIZE))
            for side in ("left", "right")}


# -- the trial ---------------------------------------------------------------------------------------------------
def _turn_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    """Angle of the rotation from orientation ra to rb (degrees)."""
    c = (float(np.trace(ra.T @ rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer needs from the past: each part's last 0.5 s of poses, and when a new episode starts."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.window = {"riser": deque(maxlen=L.STILL_STEPS + 1), "print": deque(maxlen=L.STILL_STEPS + 1)}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (riser centre, print centre) at the previous valid control step

    @staticmethod
    def at_start(riser, print_) -> bool:
        """The starting layout: the riser flat on the table on its mark, the print face up on the table on its
        mark (within START_TOL, which covers a seeded reset's jitter)."""
        r_ok = (np.hypot(riser.pos[0] - L.RISER_X, riser.pos[1] - L.RISER_Y) < L.START_TOL
                and abs(lowest(riser, L.RISER_SIZE)) < 0.003 and riser_tilt_deg(riser) < 1.0)
        p_ok = (np.hypot(print_.pos[0] - L.PRINT_X, print_.pos[1] - L.PRINT_Y) < L.START_TOL
                and abs(lowest(print_, L.PRINT_SIZE)) < 0.003 and print_tilt_deg(print_) < 1.0)
        return bool(r_ok and p_ok)

    def motion(self, name: str) -> tuple[float, float]:
        """The part's centre path (m) and summed turn (degrees) over the last 0.5 s; infinite until the window
        has filled (a part is never 'at rest' on less than 0.5 s of history)."""
        win = self.window[name]
        if len(win) < L.STILL_STEPS + 1:
            return math.inf, math.inf
        pts = list(win)
        path = sum(float(np.linalg.norm(b[0] - a[0])) for a, b in zip(pts, pts[1:]))
        turn = sum(_turn_deg(a[1], b[1]) for a, b in zip(pts, pts[1:]))
        return path, turn

    def at_rest(self, name: str) -> bool:
        path, turn = self.motion(name)
        return bool(path < L.STILL_PATH and turn < L.STILL_TURN_DEG)

    def update(self, riser, print_, arms=None) -> Result:
        if not (valid(riser) and valid(print_)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(riser, print_, arms)
            return self.last
        now = (np.array(riser.pos, dtype=float), np.array(print_.pos, dtype=float))
        if self.prev is not None and self.at_start(riser, print_):
            jump = max(float(np.linalg.norm(a - b)) for a, b in zip(now, self.prev))
            if jump > L.RESET_JUMP:  # the simulator put the parts back on their marks: a new episode (rigstate's
                self.reset()  # on_reset hook in task.py normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        self.window["riser"].append((now[0], rotation(riser.quat_xyzw)))
        self.window["print"].append((now[1], rotation(print_.quat_xyzw)))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(riser, print_, arms)
        return self.last

    def result(self, riser, print_, arms=None) -> Result:
        if not (valid(riser) and valid(print_)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        gone = off_table(riser, L.RISER_SIZE) or off_table(print_, L.PRINT_SIZE)
        r_low, p_low = lowest(riser, L.RISER_SIZE), lowest(print_, L.PRINT_SIZE)
        r_still, p_still = self.at_rest("riser"), self.at_rest("print")
        r_flat, r_board = riser_flat(riser), riser_on_board(riser)
        riser_set = bool(r_flat and L.RISER_LOW_MIN - EPS <= r_low <= L.RISER_LOW_MAX + EPS and r_board and r_still)
        face_up, under = print_face_up(print_), riser_under_print(print_, riser)
        floating = bool(face_up and p_still and L.PRINT_LOW_MIN - EPS <= p_low <= L.PRINT_LOW_MAX + EPS and under)
        ins, off, turn = inset(print_, riser), off_centre(print_), turned_deg(print_)
        have_arms = arms_valid(arms)
        gaps = finger_gaps(arms, riser, print_) if have_arms else None
        # unreadable arms: 'clear' taken as met for the step, and the step counted in arms_unavailable
        clear = bool(min(gaps.values()) >= L.CLEAR_GAP - EPS) if have_arms else True
        hidden = bool(ins >= L.INSET_MIN - EPS)
        loose = bool(off <= L.CENTRE_LOOSE + EPS and turn <= L.TURN_LOOSE_DEG + EPS)
        tight = bool(off <= L.CENTRE_TIGHT + EPS and turn <= L.TURN_TIGHT_DEG + EPS)
        if gone or not riser_set:
            score = 0.0
        elif not floating:
            score = 0.25
        elif hidden and loose:
            score = 1.0 if (tight and clear) else 0.75
        else:
            score = 0.5
        r_path, r_turn = self.motion("riser")
        p_path, p_turn = self.motion("print")
        details = {
            "valid_state": True,
            "off_table": gone,
            "riser_set": riser_set,
            "riser_flat": r_flat,
            "riser_tilt_deg": round(riser_tilt_deg(riser), 2),
            "riser_yaw_deg": round(riser_yaw_deg(riser), 2),
            "riser_low_mm": _mm(r_low),
            "riser_on_board": r_board,
            "riser_path_mm": _mm(r_path),
            "riser_turn_deg": round(r_turn, 2) if math.isfinite(r_turn) else None,
            "riser_at_rest": r_still,
            "floating": floating,
            "print_face_up": face_up,
            "print_tilt_deg": round(print_tilt_deg(print_), 2),
            "print_low_mm": _mm(p_low),
            "print_path_mm": _mm(p_path),
            "print_turn_deg": round(p_turn, 2) if math.isfinite(p_turn) else None,
            "print_at_rest": p_still,
            "riser_under_print": under,
            "inset_mm": _mm(ins),
            "hidden": hidden,
            "off_centre_mm": _mm(off),
            "turned_deg": round(turn, 2),
            "clear": clear,
            "left_finger_gap_mm": _mm(gaps["left"]) if gaps else None,
            "right_finger_gap_mm": _mm(gaps["right"]) if gaps else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
