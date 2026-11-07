"""The scorer, as plain functions of the middle length's and the shim's states and the arms' state. NumPy only; no
simulator needed.

The score is read from the end state, as a track inspector reads it: a taper gauge in each joint at its narrowest
point, the middle length still between its shoulders on both middle sleepers, the shim off the track and put down on
the table in front of it, and both grippers clear. One whole-trial record is kept, 'gauged': at some control step
the shim stood on the table in the right joint, within 1 mm of both the fixed rail end and the middle length, with
that joint 8.5 to 11.5 mm, so the joint was set on the gauge, as the instruction asks; 1.0 needs it. The rest of the
history kept, by a tracker fed one observation per control step by ``task.success()``, is each part's last 0.5 s of
positions (to judge 'still' from its path, never its reported velocity, which chatters at 10-45 mm/s while a part
stays put) and when a new episode starts.

    Tracker.update(rail, shim, arms) -> Result(score, success, details)

``rail`` (the loose middle length) and ``shim`` are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; every corner, gap and distance is
computed from ``pos``, ``quat_xyzw`` and the parts' known sizes (layout.py). ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening``, or
``None`` when the arms could not be read; then 'hands off' is taken as met for that step and the step is counted in
``arms_unavailable`` (a policy cannot cause that; the GPU runs check that it stays 0). A part state with a missing
field, a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker
(the 'gauged' record included) as it was.

Both parts are plain boxes, so the rail is the same end for end and either way up, and the shim any way round:
tilts are folded to 0..90 degrees and the rail's heading to -90..90.

Every bound the text states as inclusive ('within', 'at least', '8.5 to 11.5 mm') is compared with a tolerance of
1e-9 (metres or degrees), so a value computed exactly on a stated bound lands in the tier the text gives it; bounds
stated as strict ('under', 'below', 'lower than') stay strict.
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
    """Every field present, every number finite and the quaternion not degenerate."""
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


def _deg(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), 2)


# -- geometry of a box --------------------------------------------------------------------------------------------
def corners(s, size) -> np.ndarray:
    """The eight corners of a box of ``size`` (world, 8 x 3) from its centre and orientation."""
    half = np.array(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def lowest(s, size) -> float:
    """Height of the box's lowest corner above the table top (m)."""
    return float(corners(s, size)[:, 2].min() - L.TOP)


def point_box_distance(p, s, size) -> float:
    """Distance from a point to the box (its real, turned outline; 0 inside it)."""
    local = rotation(s.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(size, dtype=float) / 2, 0.0)))


_EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                   if np.count_nonzero(_SIGNS[i] != _SIGNS[j]) == 1])  # the twelve edges, as corner index pairs
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _box_corners(pos, quat, size) -> tuple[np.ndarray, np.ndarray]:
    """(the eight corners (8 x 3), the rotation matrix) of a box."""
    rot = rotation(quat)
    return np.asarray(pos, dtype=float) + (_SIGNS * np.asarray(size, dtype=float) / 2) @ rot.T, rot


def box_distances(pairs) -> list[float]:
    """The gap between each of several pairs of boxes, ((pos, quat_xyzw, size), (pos, quat_xyzw, size)) each: 0 if
    they touch or overlap. Two convex boxes come nearest at an edge (or a corner) of one of them, so the gap is the
    nearest approach of either box's twelve edges to the other box. The distance to a convex box is convex along a
    straight edge, so a golden-section search on every edge at once finds each edge's nearest point (to well under a
    micrometre)."""
    a_pts, b_pts, pos, rot, half = [], [], [], [], []
    for box_a, box_b in pairs:
        ca, ra = _box_corners(*box_a)
        cb, rb = _box_corners(*box_b)
        for cs, (p, r, size) in ((ca, (box_b[0], rb, box_b[2])), (cb, (box_a[0], ra, box_a[2]))):
            a_pts.append(cs[_EDGES[:, 0]])
            b_pts.append(cs[_EDGES[:, 1]])
            pos.append(np.broadcast_to(np.asarray(p, dtype=float), (12, 3)))
            rot.append(np.broadcast_to(r, (12, 3, 3)))
            half.append(np.broadcast_to(np.asarray(size, dtype=float) / 2, (12, 3)))
    a, b = np.concatenate(a_pts), np.concatenate(b_pts)
    pos, rot, half = np.concatenate(pos), np.concatenate(rot), np.concatenate(half)

    def dist(t):
        local = np.einsum("ni,nij->nj", a + (b - a) * t[:, None] - pos, rot)
        return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)

    lo, hi = np.zeros(len(a)), np.ones(len(a))
    for _ in range(36):
        t1, t2 = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        lower = dist(t1) <= dist(t2)
        hi, lo = np.where(lower, t2, hi), np.where(lower, lo, t1)
    best = np.minimum(np.minimum(dist(lo * 0.0), dist(lo * 0.0 + 1.0)), dist((lo + hi) / 2))
    return [float(best[k * 24:(k + 1) * 24].min()) for k in range(len(pairs))]


def box_distance(pos_a, quat_a, size_a, pos_b, quat_b, size_b) -> float:
    """The gap between two boxes (0 if they touch or overlap); see ``box_distances``."""
    return box_distances([((pos_a, quat_a, size_a), (pos_b, quat_b, size_b))])[0]


def axis_tilt_deg(s, axis: int) -> float:
    """Angle between the vertical and the box's body axis ``axis`` (0 x, 1 y, 2 z), folded to 0..90 degrees."""
    zz = rotation(s.quat_xyzw)[2, axis]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


# -- the middle length ----------------------------------------------------------------------------------------------
def rail_tilt_deg(rail) -> float:
    """Angle between the vertical and the middle length's 25 mm (height) axis, 0..90 (either way up)."""
    return axis_tilt_deg(rail, 2)


def rail_yaw_deg(rail) -> float:
    """Heading of the middle length's long axis seen from above, from the track's direction (y), folded to -90..90
    degrees (it is the same end for end)."""
    r = rotation(rail.quat_xyzw)
    a = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
    return float((a + 90.0) % 180.0 - 90.0)


def gaps(rail) -> tuple[float, float]:
    """(left gap, right gap) in metres, each at the joint's narrowest point, as a taper gauge reads it: 0.165 minus
    the largest y of any of the middle length's corners, and the smallest y of any corner plus 0.165. Negative means
    the middle length reaches past a fixed rail end's face."""
    ys = corners(rail, L.MID_SIZE)[:, 1]
    return float(L.FIX_END_Y - ys.max()), float(ys.min() + L.FIX_END_Y)


def seat_x(rail, y_seat: float) -> float:
    """Where the middle length's centre line crosses the line y = ``y_seat`` (x, m); NaN if it does not run along
    the track (more than 60 degrees off y)."""
    r = rotation(rail.quat_xyzw)
    u = r[:, 1]  # the long axis in the world
    if abs(float(u[1])) < 0.5:
        return math.nan
    c = np.asarray(rail.pos, dtype=float)
    s = (y_seat - float(c[1])) / float(u[1])
    return float(c[0] + s * float(u[0]))


def seat_offsets(rail) -> tuple[float, float]:
    """How far the centre line is from x 0.220 at the left seat (y +0.08) and the right seat (y -0.08) (m)."""
    return tuple(abs(seat_x(rail, y) - L.RAIL_X) for y in L.SEAT_YS)


def seated(rail) -> bool:
    """In its seats: lowest corner within 2 mm of the sleeper tops (16 to 20 mm up), its 25 mm axis within 3 degrees
    of vertical (either way up: not on its side, not tipped), and its centre line within 6 mm of x 0.220 at both
    seats (y +0.08 and -0.08)."""
    low = lowest(rail, L.MID_SIZE)
    offs = seat_offsets(rail)
    return bool(abs(low - L.SLEEPER_TOP) <= L.SEAT_Z_TOL + EPS
                and rail_tilt_deg(rail) <= L.TILT_MAX_DEG + EPS
                and all(math.isfinite(o) and o <= L.SEAT_TOL + EPS for o in offs))


def in_band(g: float, band: tuple[float, float]) -> bool:
    return bool(band[0] - EPS <= g <= band[1] + EPS)


# -- the shim -----------------------------------------------------------------------------------------------------
def shim_tilt_deg(shim) -> float:
    """Angle between the vertical and the shim's 45 mm axis, 0..90 (0 standing on its 60 x 10 mm edge as it starts,
    90 lying flat). Reported for the trace only."""
    return axis_tilt_deg(shim, 2)


def shim_away(shim) -> bool:
    """Put away: its lowest corner within 3 mm of the table top, and every corner at x below 0.160 (nearer the arms
    than the sleepers' near ends), any way round."""
    c = corners(shim, L.SHIM_SIZE)
    return bool(abs(float(c[:, 2].min()) - L.TOP) <= L.SHIM_TABLE_TOL + EPS and float(c[:, 0].max()) < L.CESS_X)


def shim_on_track(shim) -> bool:
    """On the track: some corner of the shim at x 0.160 to 0.420 (over the sleepers), |y| below 0.420 (the rails
    end at 0.400) and lower than 45 mm above the table (2 mm above the rail tops): in a joint, on a rail or a sleeper,
    between the rails or under them, or held down among them."""
    c = corners(shim, L.SHIM_SIZE)
    inside = ((c[:, 0] >= L.TRACK_X[0] - EPS) & (c[:, 0] <= L.TRACK_X[1] + EPS) & (np.abs(c[:, 1]) < L.TRACK_Y)
              & (c[:, 2] - L.TOP < L.TRACK_Z))
    return bool(inside.any())


def shim_in_right_joint(rail, shim) -> bool:
    """Standing in the right joint: the shim's lowest corner within 3 mm of the table top, some of it at x 0.210 to
    0.230 (across the rail's line), and its centre between the fixed rail end's face (y -0.165) and the middle
    length's right end (its smallest corner y)."""
    c = corners(shim, L.SHIM_SIZE)
    if abs(float(c[:, 2].min()) - L.TOP) > L.SHIM_TABLE_TOL + EPS:
        return False
    if float(c[:, 0].max()) < L.RAIL_LINE_X[0] - EPS or float(c[:, 0].min()) > L.RAIL_LINE_X[1] + EPS:
        return False
    y = float(np.asarray(shim.pos, dtype=float)[1])
    return bool(-L.FIX_END_Y - EPS <= y <= float(corners(rail, L.MID_SIZE)[:, 1].min()) + EPS)


def shim_contacts(rail, shim) -> tuple[float, float]:
    """(the shim to the fixed rail end, the shim to the middle length): the gaps between their real, turned
    outlines (m; 0 touching)."""
    sh = (shim.pos, shim.quat_xyzw, L.SHIM_SIZE)
    to_fixed, to_rail = box_distances([(sh, (L.FIXED_RIGHT_POS, (0.0, 0.0, 0.0, 1.0), L.FIXED_RIGHT_SIZE)),
                                       (sh, (rail.pos, rail.quat_xyzw, L.MID_SIZE))])
    return to_fixed, to_rail


def gauge_set(rail, shim) -> bool:
    """The joint set on the gauge, on this step: the shim standing in the right joint, within 1 mm of both the
    fixed rail end and the middle length (it fills the joint), with the right gap 8.5 to 11.5 mm at its narrowest
    point (so the 10 mm gauge stands square across the joint, not wedged on a slant or leaning across a wider
    one)."""
    if not shim_in_right_joint(rail, shim) or not in_band(gaps(rail)[1], L.BAND1):
        return False
    return bool(all(d <= L.GAUGE_TOUCH + EPS for d in shim_contacts(rail, shim)))


# -- the arms -----------------------------------------------------------------------------------------------------
def tip_distances(arms, rail, shim) -> dict[str, tuple[float, float]]:
    """Each gripper's pad-tip point (between its finger tips) to the middle length and to the shim (m)."""
    return {side: (point_box_distance(arms[side].tips, rail, L.MID_SIZE),
                   point_box_distance(arms[side].tips, shim, L.SHIM_SIZE)) for side in ("left", "right")}


def hands_off(arms, rail, shim) -> bool:
    """Both grippers' pad-tip points at least 60 mm from the middle length and from the shim."""
    return bool(all(min(d) >= L.HANDS_OFF - EPS for d in tip_distances(arms, rail, shim).values()))


# -- the trial ----------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


TIERS = {
    1.0: "both gaps 8.5 to 11.5 mm, shim put away, both grippers clear, both parts still, gauged during the trial",
    0.5: "both gaps 6 to 14 mm, shim off the track (but not 1.0)",
    0.25: "both gaps at least 3 mm, shim off the track (but not 0.5); or the shim still standing in the right joint",
    0.0: "not seated, a gap under 3 mm, or the shim left anywhere else on the track",
}


def tier(seat_ok: bool, left: float, right: float, in_joint: bool, on_track: bool, away: bool, clear: bool,
         still: bool, gauged: bool) -> float:
    """The score of one end state (the rule in task.json, in code), checked in this order."""
    if not seat_ok or left < L.BAND3_MIN - EPS or right < L.BAND3_MIN - EPS:
        return 0.0
    if in_joint:  # the gauge left in the joint: the joint cannot close, however it is set
        return 0.25
    if on_track:  # the shim dropped or left on the track
        return 0.0
    if in_band(left, L.BAND1) and in_band(right, L.BAND1) and away and clear and still and gauged:
        return 1.0
    if in_band(left, L.BAND2) and in_band(right, L.BAND2):
        return 0.5
    return 0.25


class Tracker:
    """What the scorer needs from the past: whether the joint was set on the gauge at some step ('gauged'), each
    part's last 0.5 s of positions, and when a new episode starts."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.window = {"rail": deque(maxlen=L.STILL_STEPS + 1), "shim": deque(maxlen=L.STILL_STEPS + 1)}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with an unreadable part state (ignored)
        self.gauged = False  # the whole-trial record: the shim filled the right joint on some control step
        self.gauged_step = None  # ... the first such step
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the rail, of the shim] at the previous valid control step

    @staticmethod
    def at_start(rail, shim) -> bool:
        """The starting layout: the middle length seated within 4 mm (in plan) of its crept start, the shim standing
        on the table within 4 mm of its place."""
        r_ok = (math.hypot(float(rail.pos[0]) - L.MID_X0, float(rail.pos[1]) - L.MID_Y0) <= L.START_TOL
                and seated(rail))
        s_ok = (math.hypot(float(shim.pos[0]) - L.SHIM_X0, float(shim.pos[1]) - L.SHIM_Y0) <= L.START_TOL
                and abs(lowest(shim, L.SHIM_SIZE)) <= L.SHIM_TABLE_TOL and shim_tilt_deg(shim) <= 5.0)
        return bool(r_ok and s_ok)

    def _jumped(self, now) -> bool:
        """Some part moved, in one control step and in plan, more than 4 mm further than its last velocity could
        have carried it: the simulator put it there."""
        return any(
            float(np.linalg.norm((p - p0)[:2])) > L.RESET_JUMP + float(np.linalg.norm(v0[:2])) * self.dt
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def motion(self, name: str) -> float:
        """The part's centre path (m) over the last 0.5 s; infinite until the window has filled (a part is never
        'still' on less than 0.5 s of history)."""
        win = self.window[name]
        if len(win) < L.STILL_STEPS + 1:
            return math.inf
        pts = list(win)
        return float(sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])))

    def still(self) -> bool:
        return bool(self.motion("rail") < L.STILL_PATH and self.motion("shim") < L.STILL_PATH)

    def update(self, rail, shim, arms=None) -> Result:
        if not (valid(rail) and valid(shim)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(rail, shim, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (rail, shim)]
        if self.prev is not None and self._jumped(now) and self.at_start(rail, shim):
            self.reset()  # the simulator put both parts back on their starting places: a new episode (rigstate's
            #               on_reset hook in task.py normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        self.window["rail"].append(now[0][0])
        self.window["shim"].append(now[1][0])
        if not self.gauged and gauge_set(rail, shim):
            self.gauged, self.gauged_step = True, self.steps
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(rail, shim, arms)
        return self.last

    def result(self, rail, shim, arms=None) -> Result:
        if not (valid(rail) and valid(shim)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "tier": 0.0})
        left, right = gaps(rail)
        seat_ok, away = seated(rail), shim_away(shim)
        in_joint, on_track = shim_in_right_joint(rail, shim), shim_on_track(shim)
        contacts = shim_contacts(rail, shim) if in_joint else None  # only worked out where they can matter
        have_arms = arms_valid(arms)
        dists = tip_distances(arms, rail, shim) if have_arms else None
        # unreadable arms: 'hands off' taken as met for the step, and the step counted in arms_unavailable
        clear = bool(all(min(d) >= L.HANDS_OFF - EPS for d in dists.values())) if dists else True
        still = self.still()
        score = tier(seat_ok, left, right, in_joint, on_track, away, clear, still, self.gauged)
        x_left, x_right = (seat_x(rail, y) for y in L.SEAT_YS)
        c, sc = np.asarray(rail.pos, dtype=float), np.asarray(shim.pos, dtype=float)
        details = {
            "valid_state": True,
            "tier": score,
            "left_gap_mm": _mm(left),
            "right_gap_mm": _mm(right),
            "rail_x_mm": _mm(c[0]),
            "rail_y_mm": _mm(c[1]),
            "rail_z_mm": _mm(c[2] - L.TOP),
            "rail_low_mm": _mm(lowest(rail, L.MID_SIZE)),
            "rail_yaw_deg": _deg(rail_yaw_deg(rail)),
            "rail_tilt_deg": _deg(rail_tilt_deg(rail)),
            "rail_x_at_left_seat_mm": _mm(x_left),
            "rail_x_at_right_seat_mm": _mm(x_right),
            "seated": seat_ok,
            "shim_x_mm": _mm(sc[0]),
            "shim_y_mm": _mm(sc[1]),
            "shim_low_mm": _mm(lowest(shim, L.SHIM_SIZE)),
            "shim_tilt_deg": _deg(shim_tilt_deg(shim)),
            "shim_far_x_mm": _mm(float(corners(shim, L.SHIM_SIZE)[:, 0].max())),
            "shim_away": away,
            "shim_on_track": on_track,
            "shim_in_joint": in_joint,
            "shim_to_fixed_end_mm": _mm(contacts[0]) if contacts else None,
            "shim_to_rail_mm": _mm(contacts[1]) if contacts else None,
            "gauged": self.gauged,
            "gauged_step": self.gauged_step,
            "left_tip_to_rail_mm": _mm(dists["left"][0]) if dists else None,
            "left_tip_to_shim_mm": _mm(dists["left"][1]) if dists else None,
            "right_tip_to_rail_mm": _mm(dists["right"][0]) if dists else None,
            "right_tip_to_shim_mm": _mm(dists["right"][1]) if dists else None,
            "hands_off": clear,
            "rail_path_mm": _mm(self.motion("rail")),
            "shim_path_mm": _mm(self.motion("shim")),
            "still": still,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
