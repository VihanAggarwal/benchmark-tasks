"""The scorer, as plain functions of the thermometer's, the manikin head's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads the mechkit AssemblyStates of the thermometer (base frame at the centre of its sensing
cup, +x out of the back of the body, the cup facing layout.CUP_DIR; joint ``button``, metres pressed in) and of the head (a fixed base with one link,
``neck``, the head nodding back about a hinge along y, radians). The arms are read for the hands-off-the-head part
of TOUCHING and for the let-go check.

Words used below (every one is a number, see layout.py). Positions on the head are taken in the head's own frame,
so they follow it when it nods back. s is arc length along the forehead's curve from its centre toward the patient's
left temple (the robot's right).

    TOUCHING   the cup's centre within 2 mm of the skin and within 8 mm above or below the scan line, the cup facing
               the skin within 10 degrees of square, the SCAN button pressed 1.5 mm or more, and the head pushed back
               1 to 4 mm at the scan line (0.5 to 2.0 N straight on), with both grippers' finger zones 20 mm or more
               from the head (so the cup, not a hand, is what presses it)
    PASS       one slide along the curve: it begins on a TOUCHING step with s between -10 and +5 mm and ends at the
               first step with s at 50 mm or more (the cup's edge within 6 mm of the hairline). On every step in
               between: within 8 mm of the scan line, facing within 10 degrees, button pressed, the cup's centre no
               more than 6 mm off the skin, s between -10 and 60 mm, and no move back along the curve of more than
               3 mm. Contact may break within those limits.
    CLEAN      a PASS on whose every step TOUCHING holds, taking 1.5 to 4.0 s from s = 5 mm to s = 50 mm
    PUT BACK   the cup's centre within 10 mm (top view) and 4 mm (height) of where it stood in the rest, standing on
               its edge within 10 degrees, turned within 15 degrees of how it stood
    LET GO     every step of the last 1 s: PUT BACK, both grippers' finger zones 30 mm or more from the thermometer,
               it moving under 1 cm/s and 0.5 rad/s, the head back within 1 mm of upright and still
    SHOVED     latched: the head ever pushed back 8 mm or more at the scan line (4 N)

    0      SHOVED ever
    1.0    a CLEAN pass, and LET GO
    0.75   a CLEAN pass, not LET GO
    0.5    a PASS that was not CLEAN (contact broke or left the band, or too fast or too slow)
    0.25   TOUCHING for 0.5 s in all, no PASS
    0      otherwise
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PROBE, HEAD = "probe", "head"
PARTS = (PROBE, HEAD)
NECK = "neck"


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _deg(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


@dataclass(frozen=True)
class Reading:
    """One step. ``lens`` is the cup's centre in the world (z above the table top); ``local`` the same point in the
    head's frame as drawn upright (origin on the table under the forehead's axis)."""
    lens: np.ndarray
    local: np.ndarray
    flush: float  # degrees between the way the cup faces and the skin's inward normal there
    tilt: float  # degrees between the thermometer's z (across its width) and the vertical: 0 standing on its edge
    yaw: float  # degrees: the way the cup faces, about the vertical (0 = +x, toward the head)
    button: float
    push: float  # the head pushed back at the scan line (m)
    head_w: float  # |neck joint velocity| (rad/s)
    speed: float
    spin: float


def _head_frame(h):
    """(rotation, origin) taking the head's drawn frame to the world, its link at its current nod."""
    rb = rotation(h.quat_xyzw)
    ob = np.asarray(h.origin, dtype=float).ravel()
    pose = np.asarray(h.links[NECK], dtype=float).ravel()
    rl = rotation(pose[3:7])
    anchor = np.array([0.0, 0.0, 0.0])
    spec = getattr(h, "spec", None)
    if spec is not None:
        anchor = np.asarray(spec.body_offset(NECK), dtype=float)
    # a drawn point q rides to: p_link + rl @ (q - anchor); so drawn = rl.T @ (world - p_link) + anchor
    return rl, pose[:3], anchor, rb, ob


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        p, h = states.get(PROBE), states.get(HEAD)
        if p is None or h is None:
            return None
        o = np.asarray(p.origin, dtype=float).ravel()
        if o.size != 3:
            return None
        rp = rotation(p.quat_xyzw)
        rl, pl, anchor, _, _ = _head_frame(h)
        btn = float(p.joints["button"])
        theta = float(h.joints[NECK])
        head_w = abs(float(h.joint_vel.get(NECK, 0.0)))
        speed = float(np.linalg.norm(np.asarray(p.lin_vel, dtype=float)))
        spin = float(np.linalg.norm(np.asarray(p.ang_vel, dtype=float)))
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(rp)) and np.all(np.isfinite(rl)) and np.all(np.isfinite(pl))):
        return None
    if not all(math.isfinite(v) for v in (btn, theta, head_w, speed, spin)):
        return None
    local = rl.T @ (o - pl) + anchor  # the head's drawn frame, origin at (FX, FY, table top) in the world as drawn
    face_dir = rp @ np.asarray(L.CUP_DIR, dtype=float)
    phi = math.atan2(-float(local[1]), -float(local[0]))
    normal_world = rl @ np.array([math.cos(phi), math.sin(phi), 0.0])
    flush = _deg(float(face_dir @ normal_world))
    tilt = _deg(float(rp[2, 2]))
    yaw = math.degrees(math.atan2(float(face_dir[1]), float(face_dir[0])))
    lens = o - np.array([0.0, 0.0, L.TOP])
    return Reading(lens, local, flush, tilt, yaw, btn, L.H_ARM * theta, head_w, speed, spin)


# -- the readings ------------------------------------------------------------------------------------------------------
def gap(r: Reading) -> float:
    """The cup's centre off the skin (m): its distance from the forehead's axis less the radius."""
    return math.hypot(float(r.local[0]), float(r.local[1])) - L.RF


def along(r: Reading) -> float:
    """s: arc length round the forehead from its centre toward the patient's left temple (m)."""
    return L.RF * math.atan2(-float(r.local[1]), -float(r.local[0]))


def off_line(r: Reading) -> float:
    return abs(float(r.local[2]) - L.Z_LINE)


def on_skin(r: Reading) -> bool:
    return bool(abs(gap(r)) <= L.FACE_TOL + L.EPS)


def flush_ok(r: Reading) -> bool:
    return bool(r.flush < L.FLUSH_DEG)


def on_line(r: Reading) -> bool:
    return bool(off_line(r) <= L.LINE_TOL + L.EPS)


def button_held(r: Reading) -> bool:
    return bool(r.button >= L.BTN_PRESSED - L.EPS)


def push_in_band(r: Reading) -> bool:
    return bool(L.PUSH_MIN - L.EPS <= r.push <= L.PUSH_MAX + L.EPS)


def touching(r: Reading) -> bool:
    return bool(on_skin(r) and on_line(r) and flush_ok(r) and button_held(r) and push_in_band(r))


def near_skin(r: Reading) -> bool:
    return bool(-L.FACE_TOL - L.EPS <= gap(r) <= L.LIFT_TOL + L.EPS)


def pass_step_ok(r: Reading) -> bool:
    s = along(r)
    return bool(on_line(r) and flush_ok(r) and button_held(r) and near_skin(r) and L.S_MIN - L.EPS <= s <= L.S_MAX + L.EPS)


def starts_pass(r: Reading, touch: bool) -> bool:
    s = along(r)
    return bool(touch and L.S_MIN - L.EPS <= s <= L.START_ZONE + L.EPS)


def shoved(r: Reading) -> bool:
    return bool(r.push >= L.PUSH_SHOVE - L.EPS)


def put_back(r: Reading) -> bool:
    dx, dy = float(r.lens[0]) - L.PR_X0, float(r.lens[1]) - L.PR_Y0
    return bool(math.hypot(dx, dy) <= L.REST_XY + L.EPS and abs(float(r.lens[2]) - L.PR_Z0) <= L.REST_Z + L.EPS
                and r.tilt < L.REST_TILT_DEG
                and abs(math.remainder(r.yaw - math.degrees(L.REST_YAW), 360.0)) < L.REST_YAW_DEG)


def probe_still(r: Reading) -> bool:
    return bool(r.speed <= L.STILL_V + L.EPS and r.spin <= L.STILL_W + L.EPS)


def head_back(r: Reading) -> bool:
    return bool(abs(r.push) <= L.HEAD_BACK + L.EPS and r.head_w <= L.HEAD_STILL_W + L.EPS)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The latches (SHOVED, PASS, CLEAN, touch time), the pass in progress, and the last 1 s of the let-go check."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.shove: int | None = None
        self.touch_steps = 0
        self.passed: int | None = None
        self.clean: int | None = None
        self.attempt: dict | None = None
        self.last_pass: dict | None = None
        self.final: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        """The starting layout: the thermometer standing in its rest, still, the head upright."""
        mark = np.array([L.PR_X0, L.PR_Y0, L.PR_Z0])
        return bool(float(np.linalg.norm(r.lens - mark)) <= L.START_TOL and probe_still(r) and abs(r.push) <= L.HEAD_BACK)

    @staticmethod
    def hands_off_head(states, arms) -> bool:
        """Both grippers' finger zones HANDS_OFF_HEAD or more from the head, so the nod is the cup's doing (unreadable
        arms or head: False)."""
        try:
            return bool(rigstate.released(arms, states[HEAD].distance, L.HANDS_OFF_HEAD))
        except Exception:  # noqa: BLE001 - unreadable: not clear
            return False

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        return bool(cls.at_start(now) and float(np.linalg.norm(now.lens - prev.lens)) >= L.RESET_JUMP)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.attempt = None
            self.final.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        if self.shove is None and shoved(r):
            self.shove = self.steps
        touch = bool(touching(r) and self.hands_off_head(states, arms))
        if touch:
            self.touch_steps += 1
        self._pass(r, touch)
        gap_hands, ok_hands = None, False
        try:
            dist = states[PROBE].distance
            ok_hands = bool(rigstate.released(arms, dist, L.LET_GO))
            gap_hands = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or unreadable arms: not let go
            ok_hands = False
        self.final.append(bool(put_back(r) and ok_hands and probe_still(r) and head_back(r)))
        self.last = self.judge(r, touch, gap_hands)
        return self.last

    def _pass(self, r: Reading, touch: bool) -> None:
        a = self.attempt
        s = along(r)
        if a is not None:
            if not pass_step_ok(r) or s < a["s_max"] - L.BACKSLIP:
                self.attempt = a = None
        if a is None:
            if starts_pass(r, touch):
                self.attempt = {"clean": True, "s_max": s, "begin_step": None, "start_step": self.steps}
            return
        a["s_max"] = max(a["s_max"], s)
        if not touch:
            a["clean"] = False
        if a["begin_step"] is None and s > L.S_BEGIN + L.EPS:
            a["begin_step"] = self.steps - 1
        if s >= L.S_FINISH - L.EPS:
            t = None if a["begin_step"] is None else (self.steps - a["begin_step"]) * self.dt
            timed = bool(t is not None and L.T_MIN - L.EPS <= t <= L.T_MAX + L.EPS)
            if self.passed is None:
                self.passed = self.steps
            if a["clean"] and timed and self.clean is None:
                self.clean = self.steps
            self.last_pass = {"step": self.steps, "clean_contact": a["clean"],
                              "seconds": None if t is None else round(t, 2), "timed": timed}
            self.attempt = None

    @property
    def touched(self) -> bool:
        return self.touch_steps * self.dt >= L.TOUCH_S - L.EPS

    def let_go(self) -> bool:
        return len(self.final) == self.window and all(self.final)

    def tier(self) -> float:
        if self.shove is not None:
            return 0.0
        if self.clean is not None:
            return 1.0 if self.let_go() else 0.75
        if self.passed is not None:
            return 0.5
        if self.touched:
            return 0.25
        return 0.0

    def judge(self, r: Reading, touch: bool, gap_hands) -> Result:
        level = self.tier()
        a = self.attempt
        details = {
            "valid_state": True, "score": level, "shove_step": self.shove, "passed_step": self.passed,
            "clean_step": self.clean, "touch_s": round(self.touch_steps * self.dt, 2),
            "touching": touch, "on_skin": on_skin(r), "on_line": on_line(r), "flush_ok": flush_ok(r),
            "in_band": push_in_band(r), "button_held": button_held(r),
            "lens_mm": [_mm(float(c)) for c in r.lens], "along_mm": _mm(along(r)), "gap_mm": _mm(gap(r)),
            "off_line_mm": _mm(off_line(r)), "flush_deg": round(r.flush, 2), "tilt_deg": round(r.tilt, 2),
            "yaw_deg": round(r.yaw, 2), "push_mm": _mm(r.push), "force_n": round(L.HEAD_K * r.push, 2),
            "button_mm": _mm(r.button), "put_back": put_back(r), "let_go_1s": self.let_go(),
            "hands_gap_mm": _mm(gap_hands),
            "attempt": None if a is None else {"clean": a["clean"], "s_max_mm": _mm(a["s_max"]),
                                                "begin_step": a["begin_step"]},
            "last_pass": self.last_pass, "bad_steps": self.bad_steps,
        }
        return Result(level, bool(level == 1.0), details)
