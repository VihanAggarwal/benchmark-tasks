"""The scorer, as plain functions of the three parts' states and the grippers' state. NumPy only; no simulator needed.

The score is read from where the wire IQI and the two location markers lie on the welded plate, as a radiographer
checks a set-up before the shot (ASME Section V, Article 2): is the IQI flat on the plate's top (source-side) face,
its wires perpendicular to the weld, lying on the weld so every wire crosses it, and its whole footprint between the two
shot-length ticks (T-277.1); and is there a location marker flat on the plate's top at each tick (T-275,
T-275.1(a)(1)). A part counts only once it is released (no gripper's finger zone within 10 mm of it) and settled
(its centre moved under 2 mm in all and its yaw under 1 degree) over the last 0.5 s. A small tracker is fed once per
control step by ``task.success()``:

    Tracker.update(parts, arms) -> Result(score, success, details)

``parts`` maps "iqi", "marker_0", "marker_1" to the states mechkit reports (``AssemblyState``): ``origin`` (the base
frame's origin, the middle of the part's bottom face), ``quat_xyzw`` (that frame's orientation; for the IQI local +x
runs along its wires) and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. A state with a non-finite
number, a zero quaternion or no ``origin`` (a simulator blow-up, or a part not built as an assembly) scores 0 for
that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read. An arm that cannot be read is
skipped by the released rule (the other is still checked) and the step is counted in ``arms_unavailable``. rigstate
gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is a
cylinder round that axis from the point between its pad tips 38 mm back up the gripper, its radius reaching a pad's
outer corner (half the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across:
33 mm with the fingers open 44 mm). A part is released at a step when every readable zone is at least 10 mm from
its bounding box.

Ordered decision list (first match wins):
    1.0  IQI correct, a correct marker at both ticks, all three parts still for 1 s
    0.75 IQI correct, a correct marker at one tick or more
    0.5  IQI correct, no correct marker
    0.25 IQI counted and lying flat on the plate's top, but not correct
    0    otherwise (on the bench or the table, on another part, on its side, still held or moving)
Markers count only when the IQI is correct. The two markers are alike: either may be at either tick, and each tick
counts once.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

SIDES = ("left", "right")
_EPS = 1e-9
_BOX = {"iqi": (L.IQI_LO, L.IQI_HI), "marker_0": (L.MK_LO, L.MK_HI), "marker_1": (L.MK_LO, L.MK_HI)}
_BODY_H = {"iqi": L.IQI_H, "marker_0": L.MK_H, "marker_1": L.MK_H}


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5 and parts[2].size == 3)


def to_world(s, local) -> np.ndarray:
    return np.asarray(s.origin, dtype=float) + rotation(s.quat_xyzw) @ np.asarray(local, dtype=float)


def centre(s, name: str) -> np.ndarray:
    """The middle of the part's body (holder or block), in the world."""
    return to_world(s, (0.0, 0.0, _BODY_H[name] / 2))


def corners(s, name: str) -> np.ndarray:
    """The 8 corners of the part's bounding box (body and paint), in the world."""
    lo, hi = _BOX[name]
    pts = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return np.array([to_world(s, p) for p in pts])


def footprint(s, name: str) -> np.ndarray:
    """The 4 corners of the part's bottom face, seen from above (x, y), in the world."""
    lo, hi = _BOX[name]
    return np.array([to_world(s, (x, y, 0.0))[:2] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])


def tilt_deg(s) -> float:
    """Angle between the part's local z and the vertical, either face up (0..90 degrees)."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(s.quat_xyzw)[2, 2]), 0.0, 1.0))))


def heading_deg(s) -> float:
    """Heading of the part's local x seen from above (degrees, -180..180; for the IQI: which way its wires run)."""
    r = rotation(s.quat_xyzw)
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def wire_err_deg(s) -> float:
    """Angle between the IQI's wires seen from above and x (perpendicular to the weld), either way round (0..90)."""
    a = abs(heading_deg(s)) % 180.0
    return float(min(a, 180.0 - a))


def bottom_gap(s, name: str) -> float:
    """Height of the part's lowest point above the plate's top (m; negative: below it)."""
    return float(corners(s, name)[:, 2].min() - L.PLATE_TOP_W)


def flat(s) -> bool:
    return tilt_deg(s) <= L.FLAT_DEG + _EPS


def over_plate(s, name: str) -> bool:
    """Its centre, seen from above, inside the plate's outline."""
    c = centre(s, name)
    return bool(L.PLATE_X0 - _EPS <= c[0] <= L.PLATE_X1 + _EPS and abs(c[1]) <= L.PLATE_HALF_Y + _EPS)


def on_plate(s, name: str) -> bool:
    """Lying flat on the plate's top (the source side): flat, its centre over the plate, its lowest point within 3 mm
    of the plate's top (so not on the table, on another part, or held up)."""
    return bool(flat(s) and over_plate(s, name) and abs(bottom_gap(s, name)) <= L.SEAT_TOL + _EPS)


def iqi_pose(s) -> dict:
    """The IQI's measures (JSON-clean) and whether its pose is correct (it must also be counted)."""
    c = centre(s, "iqi")
    fp = footprint(s, "iqi")
    d = {"iqi_on_plate": on_plate(s, "iqi"), "iqi_tilt_deg": round(tilt_deg(s), 2),
         "iqi_bottom_mm": round(bottom_gap(s, "iqi") * 1000, 2), "iqi_wire_err_deg": round(wire_err_deg(s), 2),
         "iqi_off_weld_mm": round(float(c[0] - L.WELD_X) * 1000, 2),
         "iqi_max_abs_y_mm": round(float(np.abs(fp[:, 1]).max()) * 1000, 2)}
    d["iqi_pose_ok"] = bool(d["iqi_on_plate"] and wire_err_deg(s) <= L.WIRE_TOL_DEG + _EPS
                            and abs(c[0] - L.WELD_X) <= L.ON_WELD + _EPS
                            and float(np.abs(fp[:, 1]).max()) <= L.SHOT_HALF + _EPS)
    return d


def marker_tick(s, name: str) -> int | None:
    """Which tick (0: y -0.080, 1: y +0.080) a marker's pose marks, or None: lying flat on the plate's top, its centre
    within 10 mm of that tick's centre line along the weld and within 35 mm of the weld centreline."""
    if not on_plate(s, name):
        return None
    c = centre(s, name)
    if abs(c[0] - L.WELD_X) > L.MARK_ACROSS + _EPS:
        return None
    for i, ty in enumerate(L.TICK_Y):
        if abs(c[1] - ty) <= L.MARK_ALONG + _EPS:
            return i
    return None


# -- the grippers ---------------------------------------------------------------------------------------------------
def box_distance(points, s, name: str) -> np.ndarray:
    """Distance from each of the points (N, 3) to the part's bounding box (0 inside it), in the part's own frame."""
    lo, hi = (np.asarray(v, dtype=float) for v in _BOX[name])
    q = (np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(s.origin, dtype=float)) @ rotation(s.quat_xyzw)
    d = np.abs(q - (lo + hi) / 2) - (hi - lo) / 2
    return np.linalg.norm(np.maximum(d, 0.0), axis=1)


def arm_ok(arm) -> bool:
    """One arm with a finite TCP, tip point and opening."""
    try:
        tcp, tips = np.asarray(arm.tcp, dtype=float).ravel(), np.asarray(arm.tips, dtype=float).ravel()
        return bool(tcp.size == 3 and tips.size == 3 and np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))
                    and math.isfinite(float(arm.opening)))
    except (AttributeError, TypeError, ValueError):
        return False


def readable_sides(arms) -> tuple[str, ...]:
    if not isinstance(arms, dict):
        return ()
    return tuple(side for side in SIDES if side in arms and arm_ok(arms[side]))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


_ANGLES = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)


def zone_points(arm, samples: int = 20) -> np.ndarray:
    """Points filling a gripper's finger zone: discs of the zone's radius round its axis, square to it, from the pad
    tips back FINGER_LEN up the gripper (each disc's centre, a ring at half the radius and its rim)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    a = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(u, a)
    r = zone_radius(arm)
    ring = np.concatenate([[np.zeros(3)], *[k * r * (np.outer(np.cos(_ANGLES), a) + np.outer(np.sin(_ANGLES), b))
                                             for k in (0.5, 1.0)]])
    axis = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return (axis[:, None, :] + ring[None, :, :]).reshape(-1, 3)


def zone_gap(arm, s, name: str) -> float:
    """How far a gripper's finger zone is from the part (0 or less: they touch or overlap). The zone runs from the
    pad tips up the gripper, so a gripper straight above a part is its tips' height over the part's top away."""
    d = box_distance(zone_points(arm), s, name)
    return float(d.min()) if float(d.min()) > 0.0 else -1e-3


def released_now(s, name: str, arms) -> bool:
    """No readable gripper's finger zone within 10 mm of the part at this step."""
    return all(zone_gap(arms[side], s, name) >= L.HANDS_OFF - _EPS for side in readable_sides(arms))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each part's centre, heading and released flag over the last 1 s, and
    diagnostics for the trace."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.history: deque = deque(maxlen=2 * self.window + 1)  # {name: (centre, heading, released)} per step
        self.prev = None  # {name: (origin, velocity)} at the previous valid step
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(parts) -> bool:
        """The starting layout: every part flat on the bench on its own mark, at rest."""
        for name in L.NAMES:
            s = parts[name]
            x0, y0, _ = L.IQI_START if name == "iqi" else L.MK_START[name]
            o = np.asarray(s.origin, dtype=float)
            if math.hypot(o[0] - x0, o[1] - y0) >= L.START_TOL or abs(o[2] - L.TOP) >= 0.003 or tilt_deg(s) >= 2.0:
                return False
            try:
                if float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) >= 0.01:
                    return False
            except (AttributeError, TypeError, ValueError):
                pass
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        for name, (o, _) in now.items():
            o0, v0 = self.prev[name]
            if float(np.linalg.norm(o - o0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall:
                return True
        return False

    def _window(self, name: str, entries) -> tuple[float, float, bool]:
        """(centre path m, largest yaw change from the window's last entry deg, released throughout) for one part."""
        path = float(sum(np.linalg.norm(b[name][0] - a[name][0]) for a, b in zip(entries, entries[1:])))
        h_end = entries[-1][name][1]
        yaw = max(abs(_wrap(e[name][1] - h_end)) for e in entries)
        return path, float(yaw), all(e[name][2] for e in entries)

    def motion(self, name: str) -> dict:
        """For one part: counted (released and settled over the last 0.5 s), still_1s (both of the last two 0.5 s
        windows settled, released throughout the last 1 s), and the measures."""
        h = list(self.history)
        w = self.window
        out = {"counted": False, "still_1s": False, "path_mm": None, "yaw_change_deg": None, "released": None}
        if not h:
            return out
        out["released"] = bool(h[-1][name][2])
        if len(h) < w + 1:
            return out
        path, yaw, rel = self._window(name, h[-(w + 1):])
        out.update(path_mm=round(path * 1000, 2), yaw_change_deg=round(yaw, 3))
        settled = path < L.STILL_PATH and yaw < L.STILL_YAW_DEG
        out["counted"] = bool(rel and settled)
        if len(h) >= 2 * w + 1:
            p0, y0, r0 = self._window(name, h[:w + 1])
            out["still_1s"] = bool(out["counted"] and r0 and p0 < L.STILL_PATH and y0 < L.STILL_YAW_DEG)
        return out

    def update(self, parts, arms=None) -> Result:
        self.steps += 1
        if not all(valid(parts.get(n)) for n in L.NAMES):  # a blow-up: score 0, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(parts, arms)
            return self.last
        now = {n: (np.asarray(parts[n].origin, dtype=float), np.asarray(getattr(parts[n], "lin_vel", np.zeros(3)),
                                                                          dtype=float)) for n in L.NAMES}
        if not all(np.all(np.isfinite(v)) and v.size == 3 for _, v in now.values()):
            now = {n: (o, np.zeros(3)) for n, (o, _) in now.items()}
        if self.prev is not None and self._teleported(now) and self.at_start(parts):
            self.reset()  # the simulator put every part back on its mark: a new episode
            self.steps = 1
        self.prev = now
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.history.append({n: (centre(parts[n], n), heading_deg(parts[n]), released_now(parts[n], n, arms))
                             for n in L.NAMES})
        self.last = self.result(parts, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, parts, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable,
                 "arms_readable": list(readable_sides(arms)), "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = [n for n in L.NAMES if not valid(parts.get(n) if isinstance(parts, dict) else None)]
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_parts": bad, "tier_reason": "invalid_state",
                                       **trial})
        mot = {n: self.motion(n) for n in L.NAMES}
        iqi = parts["iqi"]
        d = {"valid_state": True, **iqi_pose(iqi)}
        for n in L.NAMES:
            for k, v in mot[n].items():
                d[f"{n}_{k}"] = v
        ticks: dict[int, str] = {}
        for n in L.MARKERS:
            s = parts[n]
            c = centre(s, n)
            t = marker_tick(s, n)
            d[f"{n}_tick"] = t
            d[f"{n}_on_plate"] = on_plate(s, n)
            d[f"{n}_bottom_mm"] = round(bottom_gap(s, n) * 1000, 2)
            d[f"{n}_tilt_deg"] = round(tilt_deg(s), 2)
            d[f"{n}_from_weld_mm"] = round(float(c[0] - L.WELD_X) * 1000, 2)
            d[f"{n}_y_mm"] = round(float(c[1]) * 1000, 2)
            if t is not None and mot[n]["counted"] and t not in ticks:
                ticks[t] = n
        if readable_sides(arms):
            d["zone_gap_mm"] = {side: {n: round(zone_gap(arms[side], parts[n], n) * 1000, 1) for n in L.NAMES}
                                for side in readable_sides(arms)}
        iqi_counted = mot["iqi"]["counted"]
        iqi_correct = bool(iqi_counted and d["iqi_pose_ok"])
        d.update(iqi_correct=iqi_correct, ticks_marked=sorted(ticks) if iqi_correct else [])
        if iqi_correct:
            if len(ticks) == 2 and mot["iqi"]["still_1s"] and all(mot[n]["still_1s"] for n in ticks.values()):
                score, why = 1.0, "complete"
            elif ticks:
                score, why = 0.75, ("one_tick_marked" if len(ticks) == 1 else "not_still_for_1_s")
            else:
                score, why = 0.5, "no_marker"
        elif iqi_counted and d["iqi_on_plate"]:
            score, why = 0.25, "iqi_on_plate_but_wrong"
        else:
            if not d["iqi_on_plate"]:
                why = "iqi_not_on_plate"
            elif not mot["iqi"]["released"]:
                why = "iqi_held"
            else:
                why = "iqi_not_settled"
            score = 0.0
        d["tier_reason"] = why
        return Result(score, bool(score == 1.0), {**d, **trial})
