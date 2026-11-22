"""The score, from the parts' poses and joints and the arms' fingers (rigstate). NumPy only (plus rigstate).

Read every control step by a Tracker (task.py feeds it from success()):
- live: the NEMA 1-15P plug drawn out less than layout.UNPLUGGED_AT (18 mm = 1.8 cm, blade length);
- latched 0 for the episode:
  - while live: release button pressed >= 2 mm, OR blending wand displaced >= 5 mm relative to motor handle, OR
    either gripper's finger zone within 10 mm of the blade-bearing bell guard;
  - at any time: the blending wand falls off the bench.
- Otherwise, at the latest step:
  - 0.0: still live (untouched start, or plug pushed back in < 18 mm);
  - 0.25: unplugged (plug pulled >= 18 mm free of the outlet block);
  - 0.5: unplugged and blending wand separated from the coupling: its spigot withdrawn >= 18 mm (the full
    engagement length) straight out along the socket axis (tip within 6 mm of the axis as it leaves the socket
    mouth), having passed the 5 mm latch catch with the release button held >= 4 mm (latched once reached;
    displacement alone never counts);
  - 0.75: unplugged, wand separated and resting flat inside the blue sink wash tray (not yet released/settled);
  - 1.0: on every step of the last 1.0 s (20 steps at 20 Hz): unplugged, wand flat inside the sink wash tray, both
    grippers' finger zones >= 40 mm (4 cm) clear of the wand, and wand origin moving < 5 mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MOTOR, WAND, BENCH = "motor", "wand", "bench"
PARTS = (MOTOR, WAND, BENCH)
SEATED_BELL = L.WAND_SEATED_ORIGIN + np.array([0.0, -L.WAND_LEN, 0.0], dtype=float)


def plug_out(bench) -> float:
    return float(bench.joints.get("plug", 0.0))


def live(bench) -> bool:
    return plug_out(bench) < L.UNPLUGGED_AT


def button_press(motor) -> float:
    return float(motor.joints.get("release", 0.0))


def wand_ends(wand) -> tuple[np.ndarray, np.ndarray]:
    """(coupling end, bell end) in world coordinates."""
    o = np.asarray(wand.origin, dtype=float)
    axis_y = np.asarray(wand.rotation(), dtype=float)[:, 1]
    return o, o - L.WAND_LEN * axis_y


def wand_disp(wand) -> float:
    """Maximum displacement (m) of the wand's coupling or bell end from its seated pose on the motor handle."""
    o, bell = wand_ends(wand)
    return float(max(np.linalg.norm(o - L.WAND_SEATED_ORIGIN), np.linalg.norm(bell - SEATED_BELL)))


def spigot_tip(wand) -> np.ndarray:
    """World position of the coupling spigot's tip (seated: SPIGOT_LEN inside the motor's socket)."""
    o = np.asarray(wand.origin, dtype=float)
    return o + np.asarray(wand.rotation(), dtype=float) @ np.array([0.0, L.SPIGOT_LEN, 0.0], dtype=float)


def withdrawal(wand) -> float:
    """Axial withdrawal (m) of the spigot tip out of the socket along the motor axis (-y); 0 when seated."""
    return float(L.SEAM_Y + L.SPIGOT_LEN - spigot_tip(wand)[1])


def tip_off_axis(wand) -> float:
    """Radial distance (m) of the spigot tip from the motor's coupling axis."""
    t = spigot_tip(wand)
    return float(math.hypot(t[0] - L.BLENDER_X, t[2] - (L.TOP + L.AXIS_Z)))


def wand_in_tray(wand) -> bool:
    """Both ends (spigot tip and bell) inside the blue sink wash tray interior, axis horizontal within 15 deg, resting
    flat on tray floor."""
    o, bell = wand_ends(wand)
    tip = spigot_tip(wand)
    axis_y = np.asarray(wand.rotation(), dtype=float)[:, 1]
    if abs(float(axis_y[2])) > math.sin(L.TRAY_FLAT_TILT):
        return False
    for p in (tip, bell):
        if not (L.TRAY_INNER_X[0] <= p[0] <= L.TRAY_INNER_X[1]
                and L.TRAY_INNER_Y[0] <= p[1] <= L.TRAY_INNER_Y[1]):
            return False
    floor_z = L.TOP + L.TRAY_FLOOR
    lowest_z = min(float(o[2]) - L.COLLAR_R, float(bell[2]) - L.BELL_R)
    highest_axis_z = max(float(o[2]), float(bell[2]))
    if lowest_z < L.TOP - 0.005 or lowest_z > floor_z + L.TRAY_FLOOR_TOL:
        return False
    if highest_axis_z > L.TOP + L.TRAY_OUTER[2] + 0.015:
        return False
    return True


def fell(wand) -> bool:
    o, bell = wand_ends(wand)
    return bool(min(float(o[2]), float(bell[2])) < L.TOP + L.FELL_Z)


def hand_gap(arms, wand) -> float | None:
    """Nearest gripper finger zone's gap (m) to the wand's shapes, or None if arms are unreadable."""
    if arms is None or not rigstate.readable(arms):
        return None
    return rigstate.hands_gap(arms, wand.distance)


def bell_gap(arms, wand) -> float | None:
    """Nearest gripper finger zone's gap (m) to the blade-bearing bell guard (lower 46 mm shroud), or None."""
    if arms is None or not rigstate.readable(arms):
        return None
    rot = np.asarray(wand.rotation(), dtype=float)
    o = np.asarray(wand.origin, dtype=float)
    by_mid = (L.BELL_Y[0] + L.BELL_Y[1]) / 2
    blen = L.BELL_Y[1] - L.BELL_Y[0]
    center = o + rot @ np.array([0.0, by_mid, 0.0], dtype=float)
    size = (2 * L.BELL_R, blen, 2 * L.BELL_R)
    dist_fn = rigstate.boxes_distance([(center, size, rot)])
    return rigstate.hands_gap(arms, dist_fn)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.latched: str | None = None
        self.window: list = []  # (done, wand origin, wand bell) for the last HOLD_STEPS steps
        self.separated = False  # latched: spigot pulled out of the socket along its axis with the latch released
        self.catch_button = 0.0  # release-button travel on the last step the wand was still inside the 5 mm catch
        self.last = Result(0.0, False, {})

    def update(self, parts: dict, arms) -> Result:
        self.steps += 1
        motor, wand, bench = parts[MOTOR], parts[WAND], parts[BENCH]
        is_live = live(bench)
        btn = button_press(motor)
        disp = wand_disp(wand)
        gap = hand_gap(arms, wand)
        bgap = bell_gap(arms, wand)
        near_bell = bgap is not None and bgap < L.LIVE_BELL_GAP

        if self.latched is None:
            if is_live and btn >= L.BUTTON_PREMATURE_AT:
                self.latched = f"release button pressed ({btn * 1000:.1f} mm) while plugged in (step {self.steps})"
            elif is_live and disp >= L.WAND_PREMATURE_DISP:
                self.latched = f"wand moved ({disp * 1000:.1f} mm) while plugged in (step {self.steps})"
            elif is_live and near_bell:
                self.latched = (
                    f"gripper within 10 mm of blade bell ({bgap * 1000:.1f} mm) while plugged in (step {self.steps})"
                )
            elif fell(wand):
                self.latched = f"wand fell off the bench (step {self.steps})"

        w_out = withdrawal(wand)
        off_axis = tip_off_axis(wand)
        if not self.separated:
            if w_out < L.WAND_CLEAR_LATCH:
                self.catch_button = btn
            elif (w_out >= L.WAND_SEPARATED_AT and off_axis <= L.SOCKET_AXIS_TOL
                  and self.catch_button >= L.BUTTON_RELEASE_AT):
                self.separated = True
        detached = self.separated
        in_tray = wand_in_tray(wand)
        away = gap is not None and gap >= L.HANDS_AWAY
        done = bool(not is_live and detached and in_tray and away)

        o, bell = wand_ends(wand)
        self.window.append((done, o.copy(), bell.copy()))
        if len(self.window) > L.HOLD_STEPS:
            self.window.pop(0)

        held = (
            len(self.window) == L.HOLD_STEPS
            and all(w[0] for w in self.window)
            and max(np.linalg.norm(w[1] - self.window[0][1]) for w in self.window) < L.STILL
            and max(np.linalg.norm(w[2] - self.window[0][2]) for w in self.window) < L.STILL
        )

        if self.latched is not None or is_live:
            s = 0.0
        elif held:
            s = 1.0
        elif detached and in_tray:
            s = 0.75
        elif detached:
            s = 0.5
        else:
            s = 0.25

        details = {
            "live": is_live,
            "plug_out_mm": round(plug_out(bench) * 1000, 1),
            "button_mm": round(btn * 1000, 2),
            "wand_disp_mm": round(disp * 1000, 1),
            "withdrawal_mm": round(w_out * 1000, 1),
            "tip_off_axis_mm": round(off_axis * 1000, 1),
            "wand_separated": detached,
            "wand_in_tray": in_tray,
            "hand_gap_mm": None if gap is None else round(gap * 1000, 1),
            "bell_gap_mm": None if bgap is None else round(bgap * 1000, 1),
            "latched": self.latched,
        }
        self.last = Result(s, s >= 1.0, details)
        return self.last
