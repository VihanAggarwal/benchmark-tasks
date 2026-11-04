"""The scorer, as plain functions of the lid's and the cube's states. NumPy only; no simulator needed.

The rule is about the whole trial, not only its end: the lid must never be set down while it is off the jar,
and the jar may be open for at most 20 s in total. So the scorer keeps a little state (the worst thing that has
happened so far) and is fed one observation per control step by ``task.success()``.

    Tracker.update(lid, cube) -> Result(score, success, details)

``lid`` and ``cube`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import layout as L


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's z axis and the world's vertical (degrees)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    zz = 1.0 - 2.0 * (x * x + y * y)  # R[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def lid_seated(lid) -> bool:
    """Centred on the jar within 5 mm, tilted under 5 degrees, its bottom on the rim (within 4 mm)."""
    off = np.hypot(lid.pos[0] - L.JAR_X, lid.pos[1] - L.JAR_Y)
    return bool(
        off <= L.SEAT_XY_TOL and tilt_deg(lid.quat_xyzw) < L.SEAT_TILT_DEG and abs(lid.low[2] - L.RIM_Z) <= L.SEAT_Z_TOL
    )


def cube_in_jar(cube) -> bool:
    """Wholly inside the 50 x 50 mm opening (2 mm slack for the bounding box of a tilted cube) and below the rim."""
    slack = 0.002
    lo, hi = L.JAR_INNER_HALF + slack, L.JAR_INNER_HALF + slack
    inside_xy = (
        cube.low[0] >= L.JAR_X - lo
        and cube.high[0] <= L.JAR_X + hi
        and cube.low[1] >= L.JAR_Y - lo
        and cube.high[1] <= L.JAR_Y + hi
    )
    return bool(inside_xy and cube.high[2] < L.RIM_Z)


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
        self.open_steps = 0  # control steps with the lid not seated
        self.breach = False  # the lid came within 30 mm of the table while off the jar
        self.lifted = False  # the lid has been 30 mm clear of the rim
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (lid centre, cube centre) at the previous control step

    @staticmethod
    def at_start(lid, cube) -> bool:
        """The starting layout: lid seated and still, cube on its mark."""
        on_mark = np.hypot(cube.pos[0] - L.CUBE_X, cube.pos[1] - L.CUBE_Y) < 0.003 and cube.low[2] < L.TOP + 0.003
        return bool(lid_seated(lid) and is_resting(lid) and on_mark and is_resting(cube))

    def update(self, lid, cube) -> Result:
        now = (np.array(lid.pos, dtype=float), np.array(cube.pos, dtype=float))
        if self.prev is not None and self.at_start(lid, cube):
            jump = max(float(np.linalg.norm(now[i] - self.prev[i])) for i in range(2))
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.prev = now
        self.steps += 1
        seated = lid_seated(lid)
        if not seated:
            self.open_steps += 1
            if lid.low[2] < L.TOP + L.BENCH_CLEARANCE:
                self.breach = True
        if lid.low[2] > L.RIM_Z + L.LIFT_CLEAR:
            self.lifted = True
        self.last = self.result(lid, cube, seated)
        return self.last

    @property
    def open_s(self) -> float:
        return self.open_steps * self.dt

    def result(self, lid, cube, seated: bool | None = None) -> Result:
        seated = lid_seated(lid) if seated is None else seated
        inside = cube_in_jar(cube)
        quick = self.open_s <= L.OPEN_LIMIT_S
        if self.breach:
            score = 0.0
        elif inside and seated:
            score = 1.0 if quick else 0.75
        elif inside:
            score = 0.5
        elif self.lifted:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(lid) and is_resting(cube))
        details = {
            "seated": seated,
            "cube_in_jar": inside,
            "open_s": round(self.open_s, 2),
            "breach": self.breach,
            "lifted": self.lifted,
            "lid_low_z": float(lid.low[2]),
            "lid_tilt_deg": round(tilt_deg(lid.quat_xyzw), 2),
        }
        return Result(score, success, details)
