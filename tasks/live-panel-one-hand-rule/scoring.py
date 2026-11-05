"""The scorer, as plain functions of the two fuses' states and the arms' state. NumPy only; no simulator needed.

The one-hand rule is about the whole trial, not only its end: the first gripper to go inside the live panel is the
panel hand, the other gripper may never go inside it, and whenever the panel hand is inside, the other gripper must
be in its own parking square. So the scorer keeps a little state (which gripper is the panel hand, which rule was
broken and when) and is fed one observation per control step by ``task.success()``.

    Tracker.update(red, green, arms) -> Result(score, success, details)

``red`` and ``green`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the arm rules are taken as kept
for that step and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

rigstate does not give a gripper's heading, so its fingers are taken to fill a round 'finger zone' about its axis: a
flat-ended cylinder from the point between the finger tips to 38 mm back up the gripper, its radius reaching a
finger's outer corner (half the finger gap plus the 8 mm pad thickness, and half the pads' 27 mm width; 57 mm fully
open, 23 mm closed on a fuse). A gripper is 'inside the panel' when any part of its finger zone is over the
panel's 160 x 160 mm outline and less than 15 cm above the table top. It is 'parked' when its TCP is over its own
100 x 100 mm parking square and less than 10 cm above the table top.

The fuses are plain cylinders with two identical ends, so either end may be down: 'tilt' is measured from whichever
end is nearer to facing down.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
# The live zone: the panel's outline, from below the table top (no gripper gets lower) up to 15 cm above it.
ZONE_LO = np.array([L.PANEL_X - L.PANEL_HALF, L.PANEL_Y - L.PANEL_HALF, L.TOP - 0.05])
ZONE_HI = np.array([L.PANEL_X + L.PANEL_HALF, L.PANEL_Y + L.PANEL_HALF, L.ZONE_TOP_Z])


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
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in SIDES:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the fuse's axis (degrees, 0..90): both ends are alike, so whichever
    end is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def _low(s) -> float:
    return float(s.low[2])


def _off(s, x: float, y: float) -> tuple[float, float]:
    return float(s.pos[0]) - x, float(s.pos[1]) - y


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    """How far apart the fingers are (m), from rigstate's opening (0 closed .. 1 fully open, 94 mm)."""
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


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
_RING = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)  # and round the rims of its two flat ends
_COS, _SIN = np.cos(_RING), np.sin(_RING)
_SIGNS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))  # the eight corners of a box, as fractions


def _cylinder_gap(tips, u, r, q) -> np.ndarray:
    """Distances from the points q (n x 3) to the solid cylinder of radius r about the axis from ``tips`` along
    ``u`` for FINGER_LEN (negative inside it, by the radial depth)."""
    v = q - tips
    a = v @ u
    rho = np.linalg.norm(v - np.outer(a, u), axis=1)
    da = np.maximum(0.0, np.maximum(-a, a - L.FINGER_LEN))
    dr = rho - r
    return np.where((da == 0.0) & (dr <= 0.0), dr, np.hypot(da, np.maximum(0.0, dr)))


def finger_zone(arm) -> tuple[np.ndarray, float]:
    """Points on the zone's axis and on the rims of its two flat ends (tips and 38 mm up), and its radius."""
    tips, u = finger_axis(arm)
    r = zone_radius(arm)
    e1, e2 = _perpendiculars(u)
    circle = r * (np.outer(_COS, e1) + np.outer(_SIN, e2))
    return np.vstack([tips + np.outer(_AXIS_S, u), tips + circle, tips + L.FINGER_LEN * u + circle]), r


def zone_box_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone is from a world-aligned box (negative: they overlap). The zone is a
    flat-ended cylinder, so a gripper whose finger tips are 3 cm above a box is 3 cm from it. The smallest distance
    is searched over the box's points nearest to the zone's axis and rims, and the box's corners."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([np.clip(pts, lo, hi), lo + _SIGNS * (hi - lo)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


def in_panel(arm) -> bool:
    """The gripper is inside the panel: some part of its finger zone is over the panel's outline and less than
    15 cm above the table top."""
    return zone_box_gap(arm, ZONE_LO, ZONE_HI) <= 0.0


def panel_gap(arm) -> float:
    """How far the gripper's finger zone is from the panel's live zone (0 inside it)."""
    return max(0.0, zone_box_gap(arm, ZONE_LO, ZONE_HI))


def park_margin(arm, side: str) -> float:
    """How far the gripper's TCP is inside its own parking zone (the square's outline, up to 10 cm above the table
    top): the smallest distance to the zone's sides or top; negative when it is outside."""
    tcp = np.asarray(arm.tcp, dtype=float)
    return float(min(L.PARK_HALF - abs(tcp[0] - L.PARK_X), L.PARK_HALF - abs(tcp[1] - L.PARK_Y[side]),
                     L.TOP + L.PARK_HEIGHT - tcp[2]))


def parked(arm, side: str) -> bool:
    return park_margin(arm, side) >= 0.0


# -- the fuses ----------------------------------------------------------------------------------------------------
def socket_off(s) -> float:
    """Horizontal distance from a fuse's centre to the socket's centre (m)."""
    return float(math.hypot(*_off(s, L.SOCKET_X, L.SOCKET_Y)))


def seated(s) -> bool:
    """Seated in the socket: upright (axis within 10 degrees of vertical, either end down), its lowest point within
    3 mm of the socket's floor and its centre within 8 mm of the socket's centre (inside the 30 mm opening)."""
    return bool(tilt_deg(s.quat_xyzw) < L.SEATED_TILT_DEG and abs(_low(s) - L.SOCKET_FLOOR) <= L.SEATED_Z_TOL
                and socket_off(s) <= L.SEATED_CENTRE_TOL)


def in_socket(s) -> bool:
    """Still (partly) in the socket: centre over its 30 x 30 mm opening and lowest point at least 5 mm below its rim."""
    dx, dy = _off(s, L.SOCKET_X, L.SOCKET_Y)
    half = L.SOCKET_INNER / 2
    return bool(abs(dx) <= half and abs(dy) <= half and _low(s) <= L.SOCKET_RIM - L.IN_SOCKET_DEPTH)


def in_tray(s) -> bool:
    """In the discard tray: centre over its 80 x 80 mm floor, lowest point within 10 mm of the table top."""
    dx, dy = _off(s, L.TRAY_X, L.TRAY_Y)
    return bool(abs(dx) <= L.TRAY_INNER_HALF and abs(dy) <= L.TRAY_INNER_HALF and _low(s) <= L.TOP + L.TRAY_LOW_TOL)


def over_panel(s) -> bool:
    """A fuse inside the panel: its centre over the panel's outline and below the top of the live zone."""
    dx, dy = _off(s, L.PANEL_X, L.PANEL_Y)
    return bool(abs(dx) <= L.PANEL_HALF and abs(dy) <= L.PANEL_HALF and float(s.pos[2]) < L.ZONE_TOP_Z)


def clear_of_panel(s) -> bool:
    """No part of the fuse's outline seen from above (its bounding box) is over the panel's outline."""
    lo, hi = _lo_hi(s)
    return bool(hi[0] <= ZONE_LO[0] or lo[0] >= ZONE_HI[0] or hi[1] <= ZONE_LO[1] or lo[1] >= ZONE_HI[1])


def off_table(s) -> bool:
    return bool(_low(s) < L.TOP - L.OFF_TABLE_DROP)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "second_hand_in_panel": "a second gripper went inside the panel (or both went in on the same step)",
    "other_not_parked": "the panel hand was inside the panel while the other gripper was not in its parking square",
    "fuse_off_table": "a fuse left the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.panel_hand: str | None = None  # 'left', 'right', or 'both' (both went in on the same step)
        self.first_entry_step: int | None = None
        self.steps_inside = {s: 0 for s in SIDES}  # control steps each gripper spent inside the panel
        self.min_panel_gap = {s: math.inf for s in SIDES}  # m: each gripper's closest approach to the panel
        self.min_park_margin = math.inf  # m: the other gripper's smallest margin in its square while the panel
        #                                  hand was inside (negative: it was out)
        self.red_out_step: int | None = None  # first step the red fuse was out of the socket
        self.green_seated_step: int | None = None  # first step the green fuse was seated
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the red fuse, of the green fuse] at the previous valid step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(red, green) -> bool:
        """The starting layout: the red fuse upright on the socket floor at the socket, the green fuse upright on the
        table at its cup, both still (within START_TOL of their places, which covers a seeded reset's jitter)."""
        r_ok = (math.hypot(*_off(red, L.RED_X, L.RED_Y)) < L.START_TOL
                and abs(_low(red) - L.SOCKET_FLOOR) < 0.003 and tilt_deg(red.quat_xyzw) < 2.0)
        g_ok = (math.hypot(*_off(green, L.CUP_X, L.CUP_Y)) < L.START_TOL
                and abs(_low(green) - L.TOP) < 0.003 and tilt_deg(green.quat_xyzw) < 2.0)
        return bool(r_ok and g_ok and is_resting(red) and is_resting(green))

    def _teleported(self, now) -> bool:
        """Some fuse moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def _arm_rules(self, arms) -> None:
        inside = {s: in_panel(arms[s]) for s in SIDES}
        for s in SIDES:
            self.min_panel_gap[s] = min(self.min_panel_gap[s], panel_gap(arms[s]))
            self.steps_inside[s] += int(inside[s])
        if self.panel_hand is None:
            entered = [s for s in SIDES if inside[s]]
            if len(entered) == 2:
                self.panel_hand = "both"
                self.first_entry_step = self.steps
                self._flag("second_hand_in_panel")
            elif entered:
                self.panel_hand = entered[0]
                self.first_entry_step = self.steps
        if self.panel_hand in SIDES:
            hand, other = self.panel_hand, OTHER[self.panel_hand]
            if inside[other]:
                self._flag("second_hand_in_panel")
            if inside[hand]:
                m = park_margin(arms[other], other)
                self.min_park_margin = min(self.min_park_margin, m)
                if m < 0.0:
                    self._flag("other_not_parked")

    def update(self, red, green, arms=None) -> Result:
        if not (valid(red) and valid(green)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(red, green, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (red, green)]
        if self.prev is not None and self._teleported(now) and self.at_start(red, green):
            self.reset()  # the simulator put both fuses back in place: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        if arms_valid(arms):
            self._arm_rules(arms)
        else:
            self.arms_unavailable += 1
        if off_table(red) or off_table(green):
            self._flag("fuse_off_table")
        if self.red_out_step is None and not in_socket(red):
            self.red_out_step = self.steps
        if self.green_seated_step is None and seated(green):
            self.green_seated_step = self.steps
        self.last = self.result(red, green, arms)
        return self.last

    def result(self, red, green, arms=None) -> Result:
        if not (valid(red) and valid(green)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        inside = {s: in_panel(arms[s]) for s in SIDES} if have_arms else None
        grippers_out = (not any(inside.values())) if have_arms else True  # unreadable: taken as met, counted
        g_seated = seated(green)
        r_tray = in_tray(red)
        r_clear = clear_of_panel(red)
        r_out = not in_socket(red)
        g_in = over_panel(green)
        gone = off_table(red) or off_table(green)
        if self.breach or gone:
            score = 0.0
        elif g_seated and r_tray and grippers_out:
            score = 1.0
        elif g_seated and r_clear:
            score = 0.75
        elif r_out and g_in:
            score = 0.5
        elif r_tray:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(red) and is_resting(green))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "panel_hand": self.panel_hand,
            "first_entry_step": self.first_entry_step,
            "left_in_panel": inside["left"] if have_arms else None,
            "right_in_panel": inside["right"] if have_arms else None,
            "left_parked": parked(arms["left"], "left") if have_arms else None,
            "right_parked": parked(arms["right"], "right") if have_arms else None,
            "steps_inside": dict(self.steps_inside),
            "min_left_to_panel_mm": _mm(self.min_panel_gap["left"]),
            "min_right_to_panel_mm": _mm(self.min_panel_gap["right"]),
            "min_park_margin_mm": _mm(self.min_park_margin),
            "grippers_out": grippers_out if have_arms else None,
            "green_seated": g_seated,
            "green_over_panel": g_in,
            "green_socket_off_mm": _mm(socket_off(green)),
            "green_low_mm": _mm(_low(green) - L.TOP),
            "green_tilt_deg": round(tilt_deg(green.quat_xyzw), 2),
            "red_in_socket": not r_out,
            "red_in_tray": r_tray,
            "red_clear_of_panel": r_clear,
            "red_low_mm": _mm(_low(red) - L.TOP),
            "red_tilt_deg": round(tilt_deg(red.quat_xyzw), 2),
            "red_out_step": self.red_out_step,
            "green_seated_step": self.green_seated_step,
            "fuse_off_table": gone,
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "red_resting": is_resting(red),
            "green_resting": is_resting(green),
        }
        return Result(score, success, details)
