"""The scorer, as plain functions of the two panels' states and the grippers' state. NumPy only; no simulator needed.

The score is read from where the two rail panels stand at the end, as an inspector reads a guardrail splice from a
top photo: which panel bears on the left and middle posts (the downstream panel, D), whether the other one (the
upstream panel, U) stands against the right post with its end in front of D's end (on the road side, the way
traffic runs) or behind it (lapped against traffic), how far their ends overlap and where, and whether both
grippers have let go and come clear. The scorer keeps only each panel's last half second of poses (to tell a panel
at rest from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update({"panel_left": state, "panel_right": state}, arms) -> Result(score, success, details)

Each state is anything with ``pos`` (the box's centre) and ``quat_xyzw``, as ``yam_arena.task_file.ObjectState``
has; velocities are not used (a gripped or resting body's reported velocity chatters, so rest is judged from
positions). ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp`` (the tool point),
``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could
not be read; then neither gripper counts as released on that step (so the step cannot score 1.0), and the step is
counted in ``arms_unavailable``. A panel state with a non-finite number, a zero quaternion or a missing attribute
(a simulator blow-up), or a missing panel, scores 0 for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is a solid cylinder round the axis from the point between the finger tips 38 mm back up the gripper, out to a
finger's outer corner (half the finger gap plus the finger's 10 mm thickness, and half its 40 mm width across):
60.4 mm wide open. That holds both fingers whatever the gripper's turn, so 'the fingers are 10 mm clear of a panel'
never misses a finger left touching it, while a gripper lifted straight up off a panel is clear as soon as its
finger tips are 10 mm above the panel's top.

The panels are plain boxes, alike end for end, edge for edge and face for face: either 18 x 300 mm edge may be
down, and their heading is folded (a panel turned half a turn is the same panel). They are also interchangeable:
whichever stands in the downstream place is D.

'Outline': a panel's box projected on the table (the convex hull of its eight corners seen from above). At a given
y, its 'front edge' is the outline's smallest x there (toward the arms and the road) and its 'back edge' the
largest x.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

PANELS = ("panel_left", "panel_right")
ARMS = ("left", "right")
_HALF = np.array(L.PANEL_HALF, dtype=float)
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box


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
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        pos = np.asarray(getattr(s, "pos"), dtype=float).ravel()
        quat = np.asarray(getattr(s, "quat_xyzw"), dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    if pos.size != 3 or quat.size != 4:
        return False
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return False
    return bool(np.linalg.norm(quat) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite tool point, finger-tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            a = arms[side]
            tcp = np.asarray(a.tcp, dtype=float).ravel()
            tips = np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- one panel ---------------------------------------------------------------------------------------------------
def corners(p) -> np.ndarray:
    """The panel's eight corners in the world, (8, 3)."""
    return np.asarray(p.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(p.quat_xyzw).T


def tilt_deg(p) -> float:
    """Angle between the vertical and the panel's 45 mm axis (0..90 degrees; either long edge may be down)."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(p.quat_xyzw)[2, 2]), 0.0, 1.0))))


def low(p) -> float:
    """Height of the panel's lowest point above the table top (m)."""
    return float(corners(p)[:, 2].min() - L.TOP)


def heading_err_deg(p) -> float:
    """Angle, seen from above, between the panel's length and the y axis (the line of the posts), folded to
    0..90 degrees (either end may point either way). 90 for a panel standing on its end."""
    a = rotation(p.quat_xyzw)[:, 1]
    ax, ay = abs(float(a[0])), abs(float(a[1]))
    if math.hypot(ax, ay) < 1e-9:
        return 90.0
    return float(math.degrees(math.atan2(ax, ay)))


def length_heading_deg(p) -> float:
    """The signed turn of the panel's length from the y axis seen from above (degrees, -90..90, anticlockwise
    positive: its -y end farther from the arms than its +y end)."""
    a = rotation(p.quat_xyzw)[:, 1]
    if a[1] < 0:
        a = -a
    if math.hypot(float(a[0]), float(a[1])) < 1e-9:
        return 90.0
    return float(math.degrees(math.atan2(-float(a[0]), float(a[1]))))


def outline(p) -> np.ndarray:
    """The panel's outline seen from above: the convex hull of its corners, counter-clockwise, (n, 2)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in corners(p)[:, :2]})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def y_span(hull: np.ndarray) -> tuple[float, float]:
    return float(hull[:, 1].min()), float(hull[:, 1].max())


def slice_x(hull: np.ndarray, y: float, tol: float = 1e-7) -> tuple[float, float] | None:
    """(front edge, back edge): the smallest and largest x of the outline along the line at this y; None if the
    outline does not reach it."""
    y0, y1 = y_span(hull)
    if y < y0 - tol or y > y1 + tol:
        return None
    y = min(max(y, y0), y1)
    xs = []
    n = len(hull)
    for i in range(n):
        (px, py), (qx, qy) = hull[i], hull[(i + 1) % n]
        if abs(py - y) <= 1e-12:
            xs.append(px)
        if (py - y) * (qy - y) < 0:
            xs.append(px + (y - py) / (qy - py) * (qx - px))
    if not xs:  # y on the outline's extreme within float error
        k = int(np.argmin(np.abs(hull[:, 1] - y)))
        xs = [float(hull[k, 0])]
    return float(min(xs)), float(max(xs))


def back_edge(hull, y) -> float | None:
    s = slice_x(hull, y)
    return None if s is None else s[1]


def front_edge(hull, y) -> float | None:
    s = slice_x(hull, y)
    return None if s is None else s[0]


def downstream_gaps(hull) -> list[float | None]:
    """How far the panel's back edge is in front of the posts' faces (x 0.310) at the left and the middle post
    (m; negative = past the faces; None where the outline does not reach)."""
    out = []
    for y in L.D_LINES_Y:
        b = back_edge(hull, y)
        out.append(None if b is None else L.POST_FACE_X - b)
    return out


def point_box_distance(point, p) -> float:
    """Distance from a point to the panel's box (0 inside it), in the panel's own frame."""
    return points_box_distance(np.asarray(point, dtype=float).reshape(1, 3), p)


def points_box_distance(points, p) -> float:
    """The smallest distance from any of these points, (n, 3), to the panel's box (0 if one is inside it)."""
    d = (np.asarray(points, dtype=float) - np.asarray(p.pos, dtype=float)) @ rotation(p.quat_xyzw)
    return float(np.linalg.norm(np.maximum(np.abs(d) - _HALF, 0.0), axis=-1).min())


# -- the grippers --------------------------------------------------------------------------------------------------
# The zone's surface is sampled (a box outside it is nearest to a point of its surface; one that meets it holds a
# sample): every 5 degrees round the axis, its side every 1 mm along the axis, and its two end discs on rings every
# 1/24 of the radius (2.5 mm wide open). Against a ten times finer sampling it reads at most 0.3 mm long.
_ZONE_TH = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
_ZONE_SR = np.array([(s, 1.0) for s in np.linspace(0.0, 1.0, 39)]
                    + [(s, k / 24) for s in (0.0, 1.0) for k in range(24)])  # (along, out) as fractions


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the finger gap plus a finger's thickness, and half
    its width across (m): 60.4 mm wide open."""
    gap = float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm) -> np.ndarray:
    """Points filling the gripper's finger zone (a solid cylinder round its axis, from the point between the finger
    tips FINGER_LEN back up toward and past the tool point), (n, 3)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    ring = np.outer(np.cos(_ZONE_TH), e1) + np.outer(np.sin(_ZONE_TH), e2)  # (72, 3)
    centres = tips + np.outer(_ZONE_SR[:, 0] * L.FINGER_LEN, u)  # (87, 3)
    radii = _ZONE_SR[:, 1] * zone_radius(arm)
    return (centres[:, None, :] + radii[:, None, None] * ring[None, :, :]).reshape(-1, 3)


def finger_clearance(arm, states) -> float:
    """How far the gripper's finger zone is from the nearer of the panels' boxes (m; 0 if they meet)."""
    pts = finger_zone(arm)
    return min(points_box_distance(pts, s) for s in states.values())


# -- the splice ----------------------------------------------------------------------------------------------------
def sample_ys(lo: float, hi: float) -> list[float]:
    """Where the shared stretch [lo, hi] is read: every 2 mm from 5 mm inside one end to 5 mm inside the other
    (both included); a stretch shorter than 10 mm only at its middle."""
    a, b = lo + L.SAMPLE_INSET, hi - L.SAMPLE_INSET
    if b - a < L.EPS:
        return [(lo + hi) / 2]
    n = int(math.floor((b - a) / L.SAMPLE_STEP + 1e-9))
    ys = [a + k * L.SAMPLE_STEP for k in range(n + 1)]
    if b - ys[-1] > 1e-9:
        ys.append(b)
    return ys


def gaps_between(front_hull, back_hull, ys) -> list[float] | None:
    """At each y: the back panel's front edge minus the front panel's back edge (m): the gap between them, along
    x. None if either outline misses one of the readings."""
    out = []
    for y in ys:
        f, b = slice_x(front_hull, y), slice_x(back_hull, y)
        if f is None or b is None:
            return None
        out.append(b[0] - f[1])
    return out


def _within(v, lo, hi) -> bool:
    return lo - L.EPS <= v <= hi + L.EPS


def _all_within(vals, lo, hi) -> bool:
    return vals is not None and len(vals) > 0 and all(_within(v, lo, hi) for v in vals)


def released(arms, states, clearances: dict | None = None) -> bool:
    """Each gripper's finger zone at least 10 mm from both panels' boxes (unreadable arms: not released).
    ``clearances``: each arm's finger_clearance, if already worked out."""
    if not arms_valid(arms):
        return False
    if clearances is None:
        clearances = {a: finger_clearance(arms[a], states) for a in ARMS}
    return all(clearances[a] >= L.RELEASE_CLEAR - L.EPS for a in ARMS)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each panel's centre and orientation over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # {name: (centre, unit quaternion)}: now and 10 before
        self.prev = None  # {name: centre} at the previous valid step
        self.bad_steps = 0  # control steps with an unusable panel state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, p) -> bool:
        """The starting layout for one panel: standing on edge on the table at its starting place."""
        x0, y0 = L.PANEL_START[name]
        near = math.hypot(float(p.pos[0]) - x0, float(p.pos[1]) - y0) < L.START_TOL
        return bool(near and tilt_deg(p) < 5.0 and abs(low(p)) < 0.003)

    @staticmethod
    def _states(panels) -> dict | None:
        try:
            states = {n: panels[n] for n in PANELS}
        except (KeyError, TypeError):
            return None
        return states if all(valid(s) for s in states.values()) else None

    def motion(self, name: str) -> tuple[float | None, float | None]:
        """Over the last 0.5 s: the farthest the panel's centre has been from where it is now (m) and the largest
        turn between its orientation then and now (degrees); (None, None) until 0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None, None
        c_now, q_now = self.history[-1][name]
        past = [h[name] for h in self.history]
        moved = max(float(np.linalg.norm(c - c_now)) for c, _ in past)
        turned = max(math.degrees(2.0 * math.acos(min(1.0, abs(float(q @ q_now))))) for _, q in past)
        return moved, turned

    def update(self, panels, arms=None) -> Result:
        states = self._states(panels)
        if states is None:  # a blow-up or a missing panel: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(panels, arms)
            return self.last
        centres = {n: np.asarray(s.pos, dtype=float) for n, s in states.items()}
        if self.prev is not None and all(self.at_start(n, s) for n, s in states.items()):
            if max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in PANELS) > L.RESET_JUMP:
                self.reset()  # the simulator put them back: a new episode (rigstate's reset hook normally does
                #               this first; this is the fallback)
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        quats = {}
        for n, s in states.items():
            q = np.asarray(s.quat_xyzw, dtype=float)
            quats[n] = q / np.linalg.norm(q)
        self.history.append({n: (centres[n], quats[n]) for n in PANELS})
        self.prev = centres
        self.last = self.result(panels, arms)
        return self.last

    def _panel(self, name: str, p) -> dict:
        """Everything about one panel the ladder needs, each as a fact with its measurement."""
        hull = outline(p)
        moved, turned = self.motion(name)
        tilt, lo, head = tilt_deg(p), low(p), heading_err_deg(p)
        still = moved is not None and moved <= L.STILL_MOVE + L.EPS and turned <= L.STILL_TURN_DEG + L.EPS
        upright = tilt <= L.UPRIGHT_MAX_DEG + L.EPS
        on_table = abs(lo) <= L.ON_TABLE + L.EPS
        gaps = downstream_gaps(hull)
        y0, y1 = y_span(hull)
        standing = bool(upright and on_table and still)
        in_place = bool(standing and all(g is not None and _within(g, -L.D_PAST, L.D_GAP_MAX) for g in gaps)
                        and _within(y0, *L.D_END_Y))
        return {
            "hull": hull, "standing": standing, "upright": upright, "on_table": on_table, "still": still,
            "in_downstream_place": in_place, "tilt_deg": round(tilt, 2), "low_mm": _mm(lo), "moved_mm": _mm(moved),
            "turned_deg": None if turned is None else round(turned, 2), "heading_err_deg": round(head, 2),
            "turn_deg": round(length_heading_deg(p), 2), "gap_left_post_mm": _mm(gaps[0]),
            "gap_middle_post_mm": _mm(gaps[1]), "right_end_y_mm": _mm(y0), "left_end_y_mm": _mm(y1),
            "centre_y_mm": _mm(float(p.pos[1])),
        }

    def result(self, panels, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        states = self._states(panels)
        if states is None:
            return Result(0.0, False, {"valid_state": False, "tier": "invalid_state", "bad_steps": self.bad_steps})
        per = {n: self._panel(n, s) for n, s in states.items()}
        have_arms = arms_valid(arms)
        clear = {a: finger_clearance(arms[a], states) for a in ARMS} if have_arms else None
        rel = released(arms, states, clear)
        details: dict = {"valid_state": True, "released": rel, "arms_available": have_arms,
                         "arms_unavailable": self.arms_unavailable, "bad_steps": self.bad_steps,
                         "D": None, "U": None, "overlap_mm": None, "mid_mm": None, "lap_gap_min_mm": None,
                         "lap_gap_max_mm": None, "wrong_gap_min_mm": None, "wrong_gap_max_mm": None,
                         "wrong_reach_y_mm": None, "right_post_gap_mm": None, "lapped": False, "wrong_lap": False}
        for a in ARMS:
            details[f"{a}_finger_clear_mm"] = _mm(clear[a]) if clear else None
        candidates = [n for n in PANELS if per[n]["in_downstream_place"]]
        score, tier = 0.0, "no_downstream_panel"
        if candidates:
            # Solid panels cannot both stand there; were both reported so, the one farther left is D.
            d = max(candidates, key=lambda n: float(states[n].pos[1]))
            u = PANELS[1] if d == PANELS[0] else PANELS[0]
            dh, uh = per[d]["hull"], per[u]["hull"]
            lo, hi = y_span(dh)[0], y_span(uh)[1]
            overlap = hi - lo
            ys = sample_ys(lo, hi) if overlap > 0 else []
            road = gaps_between(uh, dh, ys) if ys else None  # D's front edge minus U's back edge: U on the road side
            if overlap >= 2 * L.SAMPLE_INSET - L.EPS:
                wrong = gaps_between(dh, uh, ys)  # U's front edge minus D's back edge along the stretch: U behind D
            else:  # a short stretch, or none: each panel read 5 mm inside its own end
                ue, de = slice_x(uh, hi - L.SAMPLE_INSET), slice_x(dh, lo + L.SAMPLE_INSET)
                wrong = None if ue is None or de is None else [ue[0] - de[1]]
            reach = min(lo, L.MIDDLE_POST_RIGHT_Y) - L.WRONG_REACH  # D's end, or the post's side if D stops short
            wrong_lap = bool(hi >= reach - L.EPS and low(states[u]) <= L.WRONG_LOW + L.EPS
                             and _all_within(wrong, L.LAP_GAP_MIN, L.LAP_GAP_MAX))
            # (D's right end at y 0 or less and an overlap of at most 100 mm keep U's centre right of the middle
            # post, y -0.047 or less: no separate rule is needed for it)
            lapped = bool(per[u]["standing"] and per[u]["heading_err_deg"] <= L.LAP_YAW_MAX_DEG + L.EPS
                          and _within(overlap, *L.LAP_OVERLAP) and _all_within(road, L.LAP_GAP_MIN, L.LAP_GAP_MAX))
            ub = back_edge(uh, L.RIGHT_LINE_Y)
            right_gap = None if ub is None else L.POST_FACE_X - ub
            mid = (lo + hi) / 2
            details.update({
                "D": d, "U": u, "overlap_mm": _mm(overlap), "mid_mm": _mm(mid), "readings": len(ys),
                "lap_gap_min_mm": _mm(min(road)) if road else None, "lap_gap_max_mm": _mm(max(road)) if road else None,
                "wrong_gap_min_mm": _mm(min(wrong)) if wrong else None,
                "wrong_gap_max_mm": _mm(max(wrong)) if wrong else None, "wrong_reach_y_mm": _mm(reach),
                "right_post_gap_mm": _mm(right_gap), "u_turn_deg": per[u]["turn_deg"],
                "lapped": lapped, "wrong_lap": wrong_lap,
            })
            if wrong_lap:
                score, tier = 0.0, "wrong_lap"
            elif lapped:
                # (the lap readings and the right-post gap together keep U turned 10.1 degrees or less)
                full = bool(_within(overlap, *L.FULL_OVERLAP) and abs(mid) <= L.FULL_MID + L.EPS
                            and right_gap is not None and _within(right_gap, *L.RIGHT_GAP))
                details["splice_ok"] = full
                score, tier = (1.0, "full") if (full and rel) else (0.5, "lapped")
            else:
                score, tier = 0.25, "downstream_only"
        details["tier"] = tier
        for n in PANELS:
            details[n] = {k: v for k, v in per[n].items() if k != "hull"}
        return Result(score, bool(score == 1.0), details)
