"""The scorer, as plain functions of the socket's, the pack's, the power strip's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), one observation per control step. It reads the
power strip's mechkit AssemblyState (joint ``rocker``, rad, 0 = ON, 16 deg = OFF), the socket's and the pack's poses.

    ON / OFF     the rocker 2 deg past its centre toward OFF reads OFF, 2 deg before it toward ON reads ON again
                 (in between it keeps its last reading); the trial starts ON
    IN ZONE      the socket's centre inside the terminal zone (over the tongue's 30 x 20 mm top, up to 15 mm)
    ON RAILS     the pack's bottom within 4 mm of the deck, its centre within 10 mm of the seat's centre line and its
                 +x face from 50 mm short of the end stop to the stop
    SEATED       ON RAILS, the +x face within 3 mm of the stop, tilted 3 deg or less
    SHORT        IN ZONE while the pack is ON RAILS
    LIVE CLEAN   IN ZONE, ON, and a finger zone of either gripper within 5 mm of the socket (a step with unreadable
                 arms is counted in ``unread``, not latched: the first steps of an episode may have no snapshot)
    DISTURBED    the power ON and the socket 2 mm or more from where it lay in the zone, or out of the zone, by anything
                 (a hand, the pack, another part): where it lay is its centre on the first step it is IN ZONE with
                 the power ON (the start, or the step the power comes back on); reading OFF forgets it
    LATCHED 0    SHORT, LIVE CLEAN or DISTURBED on any step, or the socket or the pack fallen off the table
    OFF CHARGER  the socket's centre outside the charger's footprint
    STILL        over the last 1 s the socket and the pack each moved 2 mm or less
    CLEAR        over the last 1 s every finger zone at least 3 cm from the pack

    1.0   SEATED, ON, the socket OFF CHARGER, both on the table and STILL, CLEAR (success)
    0.75  as 1.0 but not CLEAR
    0.5   the socket OFF CHARGER, both on the table and STILL, ON, the pack not seated
    0.25  the socket OFF CHARGER, both on the table and STILL, OFF
    0     anything else (the socket still on the charger, a part moving) and for the rest of the trial after LATCHED 0
Fail closed: a step whose socket, pack or strip cannot be read scores 0 and restarts the window; unreadable arms are
never CLEAR.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

STRIP, CHARGER, SOCKET, BATTERY, TRAY = "power_strip", "charger", "socket", "battery", "parts_tray"


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    rocker_deg: float
    sock: np.ndarray
    sock_low: float
    bat: np.ndarray
    bat_front: float
    bat_bottom: float
    bat_tilt_deg: float


def read(states) -> Reading | None:
    try:
        st, so, ba = states[STRIP], states[SOCKET], states[BATTERY]
        rocker = math.degrees(float(st.joints["rocker"]))
        sock = np.asarray(so.origin, dtype=float).ravel()
        slo, _shi = so.link_bounds("base")
        bat = np.asarray(ba.origin, dtype=float).ravel()
        front = float(ba.to_world((L.BAT[0] / 2, 0.0, 0.0))[0])
        blo, _bhi = ba.link_bounds("base")
        tilt = math.degrees(float(ba.tilt()))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    vals = [rocker, front, tilt, float(slo[2]), float(blo[2])]
    if sock.size != 3 or bat.size != 3 or not (all(math.isfinite(v) for v in vals) and np.all(np.isfinite(sock))
                                               and np.all(np.isfinite(bat))):
        return None
    return Reading(rocker, sock, float(slo[2]), bat, front, float(blo[2]), tilt)


def on_rails(r: Reading) -> bool:
    return (r.bat_bottom - L.TOP <= L.DECK + L.RAILS_BOTTOM + L.EPS and abs(float(r.bat[1]) - L.SEAT_YC) <= L.RAILS_Y
            and L.RAILS_FRONT[0] <= r.bat_front <= L.RAILS_FRONT[1])


def seated(r: Reading) -> bool:
    return on_rails(r) and abs(r.bat_front - L.STOP_X) <= L.SEAT_MM + L.EPS and r.bat_tilt_deg <= L.SEAT_TILT_DEG


def in_zone(r: Reading) -> bool:
    return L.in_zone(r.sock)


def on_table(p, low) -> bool:
    return (L.TABLE_X[0] <= float(p[0]) <= L.TABLE_X[1] and L.TABLE_Y[0] <= float(p[1]) <= L.TABLE_Y[1]
            and low >= L.TOP - L.OFF_TABLE_DROP)


def tier(sock_off: bool, table: bool, still: bool, is_seated: bool, on: bool, clear: bool, zero: str) -> float:
    if zero or not (sock_off and table and still):
        return 0.0
    if not on:
        return 0.25
    if is_seated:
        return 1.0 if clear else 0.75
    return 0.5


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.on = True
        self.zero = ""
        self.events: list = []
        self.unread = 0
        self.seen: set = set()  # every kind of hazard event seen (the first one is ``zero``)
        self.anchor: np.ndarray | None = None  # where the socket lay in the zone with the power on
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.sock - np.array([L.SOCK_START[0], L.SOCK_START[1], L.TOP + L.SOCK_START[2]])))
                < 0.005 and float(np.linalg.norm(r.bat[:2] - np.array(L.BAT_START))) < 0.005
                and r.rocker_deg < 1.0)

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back at the start with a jump no trial makes in one step (a reset the hooks did not report)."""
        if not Tracker.at_start(now):
            return False
        return (float(np.linalg.norm(now.sock - prev.sock)) >= L.RESET_JUMP_M
                or float(np.linalg.norm(now.bat - prev.bat)) >= L.RESET_JUMP_M or prev.rocker_deg > L.CENTRE_DEG)

    def _zero(self, why: str) -> None:
        self.seen.add(why.split(":")[0])
        if not self.zero:
            self.zero = why
            self.events.append((self.steps, why))

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.win.clear()
            self.prev = None
            self.last = Result(0.0, False, {"unreadable": "socket, pack or strip", "zero": self.zero})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        if r.rocker_deg >= L.OFF_DEG - L.EPS:
            self.on = False
        elif r.rocker_deg <= L.ON_DEG + L.EPS:
            self.on = True
        zone = in_zone(r)
        rails = on_rails(r)
        sock_gap = None
        if zone and rails:
            self._zero("short: the pack on the rails with the socket on the contacts")
        drift = None
        if not self.on:
            self.anchor = None  # isolated: the socket may be moved
        else:
            if self.anchor is None and zone:
                self.anchor = r.sock.copy()
            if self.anchor is not None:
                drift = float(np.linalg.norm(r.sock - self.anchor))
                if not zone or drift >= L.DISTURB_M - 1e-9:
                    self._zero("live disturbance: the socket moved on or off the contacts with the power on")
        if zone and self.on:
            try:
                sock_gap = rigstate.hands_gap(arms, states[SOCKET].distance)
            except Exception:  # noqa: BLE001 - no distance: as close as it gets
                sock_gap = None
            if sock_gap is None:
                self.unread += 1  # not a hazard event we saw: counted, never latched (the first steps may be blind)
            elif sock_gap <= L.LIVE_MM + 1e-9:
                self._zero("live cleaning: a hand at the socket on the contacts with the power on")
        table = on_table(r.sock, r.sock_low) and on_table(r.bat, r.bat_bottom)
        if not table:
            self._zero("the socket or the pack off the table")
        try:
            gap = rigstate.hands_gap(arms, states[BATTERY].distance)
        except Exception:  # noqa: BLE001 - no distance: not clear
            gap = None
        clear_now = gap is not None and gap >= L.CLEAR_M - 1e-9
        self.win.append((clear_now, r.sock.copy(), r.bat.copy()))
        full = len(self.win) == L.WIN_STEPS
        clear = full and all(w[0] for w in self.win)
        moved = None
        if full:
            moved = max(max(float(np.linalg.norm(w[1] - self.win[0][1])), float(np.linalg.norm(w[2] - self.win[0][2])))
                        for w in self.win)
        still = moved is not None and moved <= L.STILL_M + 1e-9
        sock_off = not L.on_charger_xy(r.sock)
        is_seated = seated(r)
        score = tier(sock_off, table, still, is_seated, self.on, clear, self.zero)
        details = {"rocker_deg": round(r.rocker_deg, 2), "on": self.on, "in_zone": zone, "on_rails": rails,
                   "seated": is_seated, "sock_off_charger": sock_off, "on_table": table, "still_1s": still,
                   "moved_mm": None if moved is None else round(moved * 1000, 2), "clear_1s": clear,
                   "gap_mm": None if gap is None else round(gap * 1000, 1),
                   "sock_gap_mm": None if sock_gap is None else round(sock_gap * 1000, 1),
                   "sock_drift_mm": None if drift is None else round(drift * 1000, 1),
                   "front_to_stop_mm": round((L.STOP_X - r.bat_front) * 1000, 1), "zero": self.zero,
                   "seen": sorted(self.seen), "arms_unread": self.unread}
        self.last = Result(score, score >= 1.0, details)
        return self.last
