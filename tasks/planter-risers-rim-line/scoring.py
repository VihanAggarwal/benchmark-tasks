"""The scorer, as plain functions of the pots', the risers' and the arms' states. NumPy only; no simulator needed.

The score is read from the end state: each pot standing upright on a riser that stands on the planter floor (on a
disc), over a disc, still and let go, and how far its rim ends below the planter's rim. Two things need a little
history, kept by a tracker fed one observation per control step by ``task.success()``: whether each pot is still
(its centre's path over the last 0.5 s, never its reported velocity, which chatters at 10-45 mm/s while a gripped
or resting part stays put) and when a new episode starts.

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` maps names to anything with ``pos`` (bounding-box centre, which for a cylinder is its centre),
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; only the
eight loose objects (layout.LOOSE) are read, and all the geometry comes from ``pos``, ``quat_xyzw`` and the parts'
known sizes. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` and ``opening``,
or ``None`` when the arms could not be read; then 'let go' is taken as met for that step and the step is counted in
``arms_unavailable`` (a policy cannot cause that; the GPU runs check it stays 0). A missing object, a non-finite
number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Pots and risers are closed cylinders, the same either end up, so 'tilt' is the angle of the axis from vertical folded
to 0..90 degrees, a pot's 'top face' is whichever end is higher and its 'bottom face' whichever is lower.

Every bound the text states as inclusive ('within', 'at least', '4 to 7.5 mm', '10 to 20 mm') is compared with a
tolerance of 1e-9 (metres or degrees), so a value computed exactly on a stated bound lands in the tier the text gives
it; bounds stated as strict ('shorter than', 'closer than') stay strict.

The tolerances (layout.py) are set so that no wrong riser can bring a pot's rim into the band however they stack:
pot A on the 20 mm riser is at least 20.2 mm down, pot B on the 30 mm riser at most 7.5 mm, pot A on the 42 mm
riser at most 6.5 mm, while pot A on the 30 mm riser stays within 10.2 to 18.5 mm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
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


def all_valid(objects) -> bool:
    """All eight loose objects present and valid."""
    try:
        return all(valid(objects[n]) for n in L.LOOSE)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            tcp = np.asarray(arms[side].tcp, dtype=float).ravel()
            if tcp.size != 3 or not np.all(np.isfinite(tcp)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


# -- cylinder geometry ---------------------------------------------------------------------------------------------
def axis_z(s) -> float:
    """|z| of the cylinder's axis (1 upright, either end up; 0 lying on its side)."""
    return float(min(1.0, abs(rotation(s.quat_xyzw)[2, 2])))


def tilt_deg(s) -> float:
    """Angle between the cylinder's axis and the vertical, folded to 0..90 degrees (either end up)."""
    return float(math.degrees(math.acos(axis_z(s))))


def top_face_z(s, height: float) -> float:
    """Height above the table of the centre of the cylinder's upper end face (m)."""
    return float(s.pos[2]) + height / 2 * axis_z(s) - L.TOP


def bottom_face_z(s, height: float) -> float:
    """Height above the table of the centre of the cylinder's lower end face (m)."""
    return float(s.pos[2]) - height / 2 * axis_z(s) - L.TOP


def face_centre(s, height: float, upper: bool) -> np.ndarray:
    """World position of the centre of the cylinder's upper (or lower) end face."""
    a = rotation(s.quat_xyzw)[:, 2]
    if a[2] < 0:
        a = -a  # the end that points up
    return np.asarray(s.pos, dtype=float) + (1.0 if upper else -1.0) * height / 2 * a


def rim_top_z(s, height: float, radius: float = L.POT_R) -> float:
    """Height above the table of the highest point of the cylinder's upper rim: its top face's centre plus
    radius x sin(tilt) (m)."""
    az = axis_z(s)
    return top_face_z(s, height) + radius * math.sqrt(max(0.0, 1.0 - az * az))


def rim_gap(pot, height: float) -> float:
    """How far the pot's highest rim point is below the planter's rim (m; negative: above it)."""
    return L.RIM - rim_top_z(pot, height)


def plan_dist(a, b) -> float:
    """Distance between two points seen from above (m)."""
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def nearest_disc(pot) -> tuple[str, float]:
    """The disc whose centre is nearest the pot's centre seen from above, and that distance (m)."""
    return min(((n, plan_dist(pot.pos, xy)) for n, xy in L.DISCS.items()), key=lambda t: t[1])


def riser_standing(riser, height: float) -> bool:
    """The riser upright (within 2 degrees, either end up) with the centre of its bottom face 4 to 7.5 mm above the
    table: on the planter floor (5 mm) or a disc (6 mm; 7 mm when tilted 2 degrees on its edge); not on the table,
    another riser or a pot."""
    low = bottom_face_z(riser, height)
    return bool(tilt_deg(riser) <= L.RISER_UPRIGHT_DEG + EPS
                and L.RISER_LOW_MIN - EPS <= low <= L.RISER_LOW_MAX + EPS)


def supports(pot, pot_h: float, riser, riser_h: float) -> tuple[bool, float, float]:
    """Does this riser carry the pot? The centre of the pot's bottom face within 1.5 mm (in height) of the centre of
    the riser's top face, and the two centres within 35 mm of each other seen from above. Also returns the height
    step (pot bottom minus riser top, m) and the offset seen from above (m)."""
    step = bottom_face_z(pot, pot_h) - top_face_z(riser, riser_h)
    off = plan_dist(face_centre(pot, pot_h, upper=False), face_centre(riser, riser_h, upper=True))
    return bool(abs(step) <= L.STEP_TOL + EPS and off <= L.AXIS_TOL + EPS), step, off


def in_band(gap: float) -> bool:
    """A rim gap of 10 to 20 mm (both ends inclusive)."""
    return bool(L.GAP_MIN - EPS <= gap <= L.GAP_MAX + EPS)


def tcp_dist(arms, pot) -> float:
    """The nearer gripper's TCP distance from the pot's centre (m)."""
    c = np.asarray(pot.pos, dtype=float)
    return float(min(np.linalg.norm(np.asarray(arms[s].tcp, dtype=float) - c) for s in ("left", "right")))


# -- the trial ----------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer needs from the past: each pot's last 0.5 s of positions, and when a new episode starts."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.window = {p: deque(maxlen=L.STILL_STEPS + 1) for p in L.POTS}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a missing or non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # every loose object's centre at the previous valid control step

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every pot and riser upright on the table on its starting place (within START_TOL,
        which covers a seeded reset's jitter)."""
        for name in L.LOOSE:
            s = objects[name]
            h, _, x, y = (L.POTS.get(name) or L.RISERS[name])
            if not (plan_dist(s.pos, (x, y)) < L.START_TOL and abs(bottom_face_z(s, h)) < 0.003
                    and tilt_deg(s) < 1.0):
                return False
        return True

    def path(self, pot: str) -> float:
        """The pot's centre path over the last 0.5 s (m); infinite until the window has filled (a pot is never
        'still' on less than 0.5 s of history)."""
        win = self.window[pot]
        if len(win) < L.STILL_STEPS + 1:
            return math.inf
        pts = list(win)
        return float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))

    def still(self, pot: str) -> bool:
        return bool(self.path(pot) < L.STILL_PATH)

    def update(self, objects, arms=None) -> Result:
        if not all_valid(objects):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(objects, arms)
            return self.last
        now = {n: np.array(objects[n].pos, dtype=float) for n in L.LOOSE}
        if self.prev is not None and self.at_start(objects):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.LOOSE)
            if jump > L.RESET_JUMP:  # the simulator put everything back on its place: a new episode (rigstate's
                self.reset()  # on_reset hook in task.py normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        for p in L.POTS:
            self.window[p].append(now[p])
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(objects, arms)
        return self.last

    def pot_report(self, name: str, objects, arms) -> dict:
        """Everything the scorer decides about one pot (all heights above the table)."""
        pot = objects[name]
        h = L.POTS[name][0]
        tilt = tilt_deg(pot)
        upright = bool(tilt <= L.POT_UPRIGHT_DEG + EPS)
        disc, disc_off = nearest_disc(pot)
        on_disc = bool(disc_off <= L.DISC_TOL + EPS)
        carrier, best = None, None
        for rname, (rh, _, _, _) in L.RISERS.items():
            riser = objects[rname]
            ok, step, off = supports(pot, h, riser, rh)
            standing = riser_standing(riser, rh)
            if ok and standing and carrier is None:
                carrier = rname
            # for the trace: the carrier, else the riser nearest seen from above (then nearest in height)
            cand = (0 if (ok and standing) else 1, round(off, 4), abs(step), rname, off, step, riser, rh)
            if best is None or cand[:3] < best[:3]:
                best = cand
        b_name, b_off, b_step, b_riser, b_h = best[3:]
        have_arms = arms_valid(arms)
        reach = tcp_dist(arms, pot) if have_arms else None
        let_go = bool(reach >= L.RELEASE_DIST - EPS) if have_arms else True  # unreadable: taken as met, counted
        still = self.still(name)
        gap = rim_gap(pot, h)
        is_set = bool(upright and on_disc and carrier is not None and still and let_go)
        staged = bool(is_set and in_band(gap))
        return {
            "set": is_set,
            "staged": staged,
            "rim_gap_mm": _mm(gap),
            "tilt_deg": round(tilt, 2),
            "upright": upright,
            "disc": disc,
            "disc_off_mm": _mm(disc_off),
            "on_disc": on_disc,
            "riser": carrier,
            "near_riser": b_name,  # the riser that carries it, or else the nearest one seen from above
            "riser_step_mm": _mm(b_step),
            "riser_axis_off_mm": _mm(b_off),
            "riser_low_mm": _mm(bottom_face_z(b_riser, b_h)),
            "riser_tilt_deg": round(tilt_deg(b_riser), 2),
            "bottom_mm": _mm(bottom_face_z(pot, h)),
            "path_mm": _mm(self.path(name)),
            "still": still,
            "tcp_mm": _mm(reach) if reach is not None else None,
            "let_go": let_go,
        }

    def result(self, objects, arms=None) -> Result:
        if not all_valid(objects):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        pots = {p: self.pot_report(p, objects, arms) for p in L.POTS}
        n_staged = sum(r["staged"] for r in pots.values())
        n_set = sum(r["set"] for r in pots.values())
        if n_staged == 2:
            score = 1.0
        elif n_staged == 1:
            score = 0.5
        elif n_set >= 1:
            score = 0.25
        else:
            score = 0.0
        details = {"valid_state": True, "staged": n_staged, "set": n_set}
        for p, rep in pots.items():
            details.update({f"{p}_{k}": v for k, v in rep.items()})
        details.update({
            "arms_available": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        })
        return Result(score, bool(score == 1.0), details)
