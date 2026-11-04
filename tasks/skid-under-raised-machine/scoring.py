"""The scorer, as plain functions of the machine's and the skid's states. NumPy only; no simulator needed.

The rule is about the whole trial, not only its end: the machine may be raised only a little and must not be
dragged out of place while it is up. So the scorer keeps a little state (the worst that has happened so far:
the machine's largest horizontal drift from its start, the highest any part of its base has been above the
table, its largest tilt) and is fed one observation per control step by ``task.success()``.

    Tracker.update(machine, skid) -> Result(score, success, details)

``machine`` and ``skid`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has.
Both are boxes whose sizes are in ``layout``, so their faces follow from ``pos`` and ``quat_xyzw``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import layout as L

_MA, _MB, _MH = (s / 2 for s in L.MACHINE_SIZE)
_SA, _SB, _SH = (s / 2 for s in L.SKID_SIZE)
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it


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
    """Angle between the body's z axis and the world's vertical (degrees)."""
    return float(np.degrees(np.arccos(np.clip(rotation(quat_xyzw)[2, 2], -1.0, 1.0))))


def finite(*states) -> bool:
    """Every number the scorer reads is finite (a simulator that blew up reports NaN or inf)."""
    return all(
        np.all(np.isfinite(np.asarray(getattr(s, k), dtype=float)))
        for s in states for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
    )


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def _face(s, a: float, b: float, dz: float) -> np.ndarray:
    """The four corners (world) of the box face at body height ``dz`` (half sizes a, b)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    return np.array([c + r @ np.array([i * a, j * b, dz]) for i, j in _SIGNS])


def machine_base(machine) -> np.ndarray:
    """The four corners of the machine's underside (world)."""
    return _face(machine, _MA, _MB, -_MH)


def machine_bottom(machine) -> float:
    """Height of the centre of the machine's underside (z)."""
    return float(machine.pos[2] - rotation(machine.quat_xyzw)[2, 2] * _MH)


def base_lift(machine) -> float:
    """How far the highest point of the machine's underside is above the table top: how much it was raised
    (a tilted machine counts by its raised edge)."""
    return float(machine_base(machine)[:, 2].max() - L.TOP)


# -- the skid: a plank has two broad faces; whichever faces up is its top (a plank turned over is still a plank)
def _skid_up(skid) -> np.ndarray:
    n = rotation(skid.quat_xyzw)[:, 2]
    return n if n[2] >= 0 else -n


def skid_tilt_deg(skid) -> float:
    return float(np.degrees(np.arccos(np.clip(_skid_up(skid)[2], -1.0, 1.0))))


def skid_top(skid, at=None) -> float:
    """Height of the skid's upward face (z): at its centre, or, with ``at`` = (x, y), of its plane above that
    point."""
    n = _skid_up(skid)
    c = np.asarray(skid.pos, dtype=float) + n * _SH
    if at is None or n[2] < 1e-6:
        return float(c[2])
    return float(c[2] - (n[0] * (at[0] - c[0]) + n[1] * (at[1] - c[1])) / n[2])


def drift(machine) -> float:
    """Horizontal distance of the machine's centre from its starting mark."""
    return float(np.hypot(machine.pos[0] - L.MACHINE_X, machine.pos[1] - L.MACHINE_Y))


# -- footprints ------------------------------------------------------------------------------------------------
def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex polygon ``clip`` (both CCW)."""
    out = [np.asarray(v, dtype=float) for v in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            return None
        prev = inp[-1]
        for cur in inp:
            sp, sc = side(prev), side(cur)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
            prev = cur
    return np.array(out) if len(out) >= 3 else None


def cover(machine, skid) -> float:
    """Share (0..1) of the machine's footprint (its underside, seen from above) that lies over the skid's top."""
    base = _ccw(machine_base(machine)[:, :2])
    plank = _ccw(_face(skid, _SA, _SB, _SH)[:, :2])
    whole = _area(base)
    if not np.isfinite(whole) or whole <= 1e-12:
        return 0.0
    part = _clip(base, plank)
    if part is None:
        return 0.0
    share = _area(part) / whole
    return float(np.clip(share, 0.0, 1.0)) if np.isfinite(share) else 0.0


# -- the rules ---------------------------------------------------------------------------------------------------
def skid_flat(skid) -> bool:
    """The skid lies flat on the table: its downward face within 3 mm of the table top, tilted under 3 degrees
    (either broad face up)."""
    underside = float(skid.pos[2] - _skid_up(skid)[2] * _SH)
    return bool(abs(underside - L.TOP) <= L.SKID_FLAT_Z_TOL and skid_tilt_deg(skid) < L.SKID_FLAT_TILT_DEG)


def gap_to_skid(machine, skid) -> float:
    """Height of the machine's underside (at its centre) above the skid's top face, right below that point."""
    return machine_bottom(machine) - skid_top(skid, at=np.asarray(machine.pos, dtype=float)[:2])


def on_skid(machine, skid) -> bool:
    """The machine sits on the skid: its underside within 2 mm of the skid's top, tilted under 3 degrees, at
    least 75 % of its footprint over the skid, and the skid lying flat on the table."""
    return bool(
        skid_flat(skid)
        and tilt_deg(machine.quat_xyzw) < L.ON_SKID_TILT_DEG
        and abs(gap_to_skid(machine, skid)) <= L.ON_SKID_Z_TOL
        and cover(machine, skid) >= L.ON_SKID_COVER
    )


def skid_under(machine, skid) -> bool:
    """The skid lies flat on the table under the machine: at least 50 % of the machine's footprint over it, and
    the machine's underside (at its centre) no more than 3 mm below the skid's top, so the skid is beneath the
    machine, not beside it or through it. The machine may be resting on it or not."""
    return bool(
        skid_flat(skid)
        and cover(machine, skid) >= L.UNDER_COVER
        and gap_to_skid(machine, skid) >= -L.UNDER_Z_TOL
    )


def off_table(machine) -> bool:
    return bool(machine.low[2] < L.TOP - L.OFF_TABLE_DROP)


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
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.max_drift = 0.0  # the machine's centre, horizontally from its starting mark
        self.max_lift = 0.0  # the highest point of the machine's underside above the table top
        self.max_tilt = 0.0  # degrees
        self.bad_steps = 0  # observations with NaN or inf in them (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the machine, of the skid] at the previous control step

    @staticmethod
    def at_start(machine, skid) -> bool:
        """The starting layout: machine and skid on their marks, flat on the table and still."""
        m_mark = drift(machine) < 0.003 and abs(machine_bottom(machine) - L.TOP) < 0.003
        s_mark = np.hypot(skid.pos[0] - L.SKID_X, skid.pos[1] - L.SKID_Y) < 0.003 and skid_flat(skid)
        return bool(
            m_mark and tilt_deg(machine.quat_xyzw) < L.ON_SKID_TILT_DEG and is_resting(machine)
            and s_mark and is_resting(skid)
        )

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def update(self, machine, skid) -> Result:
        if not finite(machine, skid):  # a broken observation: count it, change nothing, score it 0
            self.bad_steps += 1
            self.last = Result(0.0, False, {"bad_state": True, "bad_steps": self.bad_steps})
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (machine, skid)]
        if self.prev is not None and self._teleported(now) and self.at_start(machine, skid):
            self.reset()  # the simulator put everything back on its marks: a new episode
        self.prev = now
        self.steps += 1
        self.max_drift = max(self.max_drift, drift(machine))
        self.max_lift = max(self.max_lift, base_lift(machine))
        self.max_tilt = max(self.max_tilt, tilt_deg(machine.quat_xyzw))
        self.last = self.result(machine, skid)
        return self.last

    @property
    def tipped(self) -> bool:
        return self.max_tilt > L.TIP_DEG

    @property
    def moved_out(self) -> bool:
        """The machine's centre left its 30 mm outline at some moment: dragged, or carried to the skid."""
        return self.max_drift > L.DRIFT_MAX

    def result(self, machine, skid) -> Result:
        if not finite(machine, skid):
            return Result(0.0, False, {"bad_state": True, "bad_steps": self.bad_steps})
        seated = on_skid(machine, skid)
        under = skid_under(machine, skid)
        fell = off_table(machine)
        if self.tipped or self.moved_out or fell:
            score = 0.0
        elif seated:
            if self.max_drift <= L.DRIFT_FULL and self.max_lift <= L.LIFT_FULL:
                score = 1.0
            elif self.max_lift <= L.LIFT_PART:  # drift is within DRIFT_PART (= DRIFT_MAX) here
                score = 0.75
            else:
                score = 0.5
        elif under:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(machine) and is_resting(skid))
        details = {
            "on_skid": seated,
            "skid_under": under,
            "cover": round(cover(machine, skid), 3),
            "max_drift_mm": _mm(self.max_drift),
            "max_lift_mm": _mm(self.max_lift),
            "max_tilt_deg": round(self.max_tilt, 2) if np.isfinite(self.max_tilt) else None,
            "moved_out": self.moved_out,
            "off_table": fell,
            "gap_to_skid_mm": _mm(gap_to_skid(machine, skid)),
        }
        return Result(score, success, details)
