"""The scorer, as plain functions of the three items' states and the grippers' state. NumPy only; no simulator needed.

The score is read from the layout on the sheet at the end, as a curator checks a sheet before it goes for gluing,
with a steel rule and a top photo: is the collection label in the bottom right corner, a few mm (2-12 mm) in from
the right edge and the foot, square and reading upright from the foot; is the determinavit slip directly above it
or directly left of it, 0-10 mm away, lined up with it and reading upright; is the plant wholly on the sheet with
its roots toward the foot, clear of the label, the slip and the barcode. An item counts only once it lies on the
table by itself (let go by both grippers, printed face up and flat, its lowest point within 3 mm of the table top,
still for 1 s).

    Tracker.update(items, arms) -> Result(score, success, details)

``items`` is a dict {"label", "slip", "specimen": state}; each state is what mechkit reports for a compound
(``AssemblyState``): ``origin`` (the middle of the item's bottom face), ``quat_xyzw`` (its frame: local z up through
the printed top; the label's and slip's long side along local y; the specimen's root end along local +x) and the
``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. A state with a non-finite number, a zero quaternion or no
``origin`` (a simulator blow-up, or an item not built as a compound) scores 0 for that step and leaves the tracker
as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the let-go condition
is taken as met for that step and the step is counted in ``arms_unavailable``. A gripper's 'finger zone' is a solid
cylinder round its axis, flat at the point between its pad tips (nothing of the fingers reaches below them) and
running 38 mm back up the gripper; its radius reaches a pad's outer corner (half the current finger gap plus the 8 mm
pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).

Score = 0.4 (label correct) + 0.2 (slip correct; judged only when the label is correct) + 0.4 (specimen correct):
0, 0.4, 0.6, 0.8 or 1.0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_BOTTOM = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=float)  # a face's corners, in order round it


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


def valid(st) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[2].size == 3 and np.linalg.norm(parts[1]) > 0.5)


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


# -- one item ----------------------------------------------------------------------------------------------------
def to_world(st, local) -> np.ndarray:
    return np.asarray(st.origin, dtype=float) + rotation(st.quat_xyzw) @ np.asarray(local, dtype=float)


def body_corners(st, name: str, z: float) -> np.ndarray:
    """The four corners (world, 4 x 3) of the item's main body at height ``z`` in its own frame (0: bottom face)."""
    hx, hy = L.BODY_SIZE[name][0] / 2, L.BODY_SIZE[name][1] / 2
    local = np.column_stack([_BOTTOM[:, 0] * hx, _BOTTOM[:, 1] * hy, np.full(4, z)])
    return np.asarray(st.origin, dtype=float) + local @ rotation(st.quat_xyzw).T


def footprint(st, name: str) -> np.ndarray:
    """The item's outline seen from above (4 x 2, corners in order): its main body's bottom face (the specimen's is
    the 200 x 70 mm board, the label's the 67 x 47 mm tile)."""
    return body_corners(st, name, 0.0)[:, :2]


def extent(st, name: str) -> tuple[float, float, float, float]:
    """(min x, max x, min y, max y) of the footprint: what a rule laid along the sheet's edges reads."""
    f = footprint(st, name)
    return float(f[:, 0].min()), float(f[:, 0].max()), float(f[:, 1].min()), float(f[:, 1].max())


def lowest(st, name: str) -> float:
    """Height of the item's lowest corner above the table top (m)."""
    c = np.vstack([body_corners(st, name, 0.0), body_corners(st, name, L.BODY_SIZE[name][2])])
    return float(c[:, 2].min() - L.TOP)


def tilt_deg(st) -> float:
    """Angle between the item's local z (up through its printed face) and the vertical (0..180 degrees): 0 lying
    flat face up, 90 on an edge, 180 face down."""
    return float(np.degrees(np.arccos(np.clip(rotation(st.quat_xyzw)[2, 2], -1.0, 1.0))))


def heading_deg(st) -> float:
    """Where the item's local x points, seen from above (degrees, -180..180; 0 along +x, away from the arms)."""
    r = rotation(st.quat_xyzw)
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def square_err_deg(st) -> float:
    """Label and slip: how far the long side (local y) is turned from the sheet's width (world y), either end
    (0..90 degrees). A tile standing on end reads 90."""
    r = rotation(st.quat_xyzw)
    hx, hy = float(r[0, 0]), float(r[1, 0])
    if math.hypot(hx, hy) < 0.5:
        return 90.0
    a = math.degrees(math.atan2(hy, hx)) % 180.0
    return float(min(a, 180.0 - a))


def upright_err_deg(st) -> float:
    """Label and slip: angle between the way the item's print reads up (local +x, toward its top text line) and the
    sheet's head (+x), seen from above (0..180 degrees): 0 reading upright from the foot, 180 upside down."""
    r = rotation(st.quat_xyzw)
    hx, hy = float(r[0, 0]), float(r[1, 0])
    if math.hypot(hx, hy) < 0.5:
        return 180.0
    return float(math.degrees(math.acos(np.clip(hx / math.hypot(hx, hy), -1.0, 1.0))))


def root_err_deg(st) -> float:
    """Specimen: angle between its root end (local +x) and the foot (-x), seen from above (0..180 degrees)."""
    r = rotation(st.quat_xyzw)
    hx, hy = float(r[0, 0]), float(r[1, 0])
    if math.hypot(hx, hy) < 0.5:
        return 180.0
    return float(math.degrees(math.acos(np.clip(-hx / math.hypot(hx, hy), -1.0, 1.0))))


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two convex outlines (n x 2, corners in order) share more than an edge (separating-axis test)."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                continue
            n = n / nn
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() + L.OVERLAP_EPS or pb.max() <= pa.min() + L.OVERLAP_EPS:
                return False
    return True


BARCODE = np.array([(L.BARCODE_X1, L.BARCODE_Y1), (L.BARCODE_X0, L.BARCODE_Y1), (L.BARCODE_X0, L.BARCODE_Y0),
                    (L.BARCODE_X1, L.BARCODE_Y0)], dtype=float)


# -- the grippers ---------------------------------------------------------------------------------------------------
def point_distance(p, st, name: str) -> float:
    """Distance from a point to the item's bounding box (body and print; 0 inside it), in the item's own frame."""
    q = rotation(st.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(st.origin, dtype=float))
    d = np.abs(q - np.asarray(L.box_center_local(name))) - np.asarray(L.box_half(name))
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, st, name: str, iters: int = 200) -> float:
    """How far a gripper's finger zone is from the item's bounding box (0 when they touch or overlap). The zone is a
    solid cylinder round the gripper's axis: from the point between the pad tips (its flat bottom: nothing of the
    fingers reaches below the tips) FINGER_LEN back up the gripper, of radius zone_radius(). Both are convex, so
    alternating projections between them converge on their closest pair of points."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    r, length = zone_radius(arm), L.FINGER_LEN
    rot, o = rotation(st.quat_xyzw), np.asarray(st.origin, dtype=float)
    t0, ul = rot.T @ (tips - o), rot.T @ u  # the zone in the item's frame
    c, h = np.asarray(L.box_center_local(name)), np.asarray(L.box_half(name))
    lo, hi = c - h, c + h

    def onto_zone(x):
        w = x - t0
        a = float(w @ ul)
        v = w - a * ul
        rho = float(np.linalg.norm(v))
        if rho > r:
            v = v * (r / rho)
        return t0 + min(max(a, 0.0), length) * ul + v

    q = np.clip(t0, lo, hi)
    for _ in range(iters):
        q_next = np.clip(onto_zone(q), lo, hi)
        if float(np.linalg.norm(q_next - q)) < 1e-10:
            q = q_next
            break
        q = q_next
    return float(np.linalg.norm(onto_zone(q) - q))


# -- the rule -----------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(float(v)) else None


def _within(v: float, lo: float, hi: float) -> bool:
    """lo <= v <= hi, a reading exactly on a limit counting as inside (to within L.EPS)."""
    return lo - L.EPS <= v <= hi + L.EPS


def laid(st, name: str, still: bool, off: bool) -> tuple[bool, str | None, dict]:
    """Whether the item lies on the table by itself: let go, face up and flat within 5 degrees, lowest point within
    3 mm of the table top (not resting on another item, not held up, not fallen below), still for 1 s."""
    tilt, low = tilt_deg(st), lowest(st, name)
    d = {"tilt_deg": round(tilt, 2), "lowest_mm": _mm(low), "still": bool(still), "let_go": bool(off)}
    if not off:
        return False, "not_let_go", d
    if tilt > L.FLAT_DEG + L.ANG_EPS:
        return False, ("face_down" if tilt > 90.0 else "not_flat"), d
    if abs(low) > L.ON_TABLE + L.EPS:
        return False, "not_on_table", d
    if not still:
        return False, "not_still", d
    return True, None, d


def judge_label(st, still: bool, off: bool) -> tuple[bool, dict]:
    ok, why, d = laid(st, "label", still, off)
    x0, x1, y0, y1 = extent(st, "label")
    right_in, foot_in = y0 - L.SHEET_Y0, x0 - L.SHEET_X0
    d.update({"square_err_deg": round(square_err_deg(st), 2), "upright_err_deg": round(upright_err_deg(st), 2),
              "right_edge_in_mm": _mm(right_in),
              "foot_in_mm": _mm(foot_in), "extent_mm": [_mm(v) for v in (x0, x1, y0, y1)]})
    if ok and square_err_deg(st) > L.SQUARE_DEG + L.ANG_EPS:
        ok, why = False, "not_square"
    if ok and upright_err_deg(st) > L.SQUARE_DEG + L.ANG_EPS:
        ok, why = False, "upside_down"
    if ok and not _within(right_in, L.EDGE_IN_MIN, L.EDGE_IN_MAX):
        ok, why = False, "not_at_right_edge"
    if ok and not _within(foot_in, L.EDGE_IN_MIN, L.EDGE_IN_MAX):
        ok, why = False, "not_at_foot"
    return ok, {**d, "correct": ok, "zero_reason": why}


def judge_slip(st, label_st, label_ok: bool, still: bool, off: bool) -> tuple[bool, dict]:
    ok, why, d = laid(st, "slip", still, off)
    x0, x1, y0, y1 = extent(st, "slip")
    lx0, lx1, ly0, ly1 = extent(label_st, "label")
    above_gap, above_align = x0 - lx1, y0 - ly0  # directly above: its bottom edge over the label's top edge
    left_gap, left_align = y0 - ly1, x0 - lx0  # directly left: its right edge beside the label's left edge
    covers = overlap(footprint(st, "slip"), footprint(label_st, "label"))
    on_sheet = min(x0 - L.SHEET_X0, L.SHEET_X1 - x1, y0 - L.SHEET_Y0, L.SHEET_Y1 - y1) >= -L.EPS
    is_above = _within(above_gap, 0.0, L.SLIP_GAP_MAX) and abs(above_align) <= L.SLIP_ALIGN + L.EPS
    is_left = _within(left_gap, 0.0, L.SLIP_GAP_MAX) and abs(left_align) <= L.SLIP_ALIGN + L.EPS
    d.update({"square_err_deg": round(square_err_deg(st), 2), "upright_err_deg": round(upright_err_deg(st), 2),
              "gap_above_mm": _mm(above_gap),
              "align_above_mm": _mm(above_align), "gap_left_mm": _mm(left_gap), "align_left_mm": _mm(left_align),
              "overlaps_label": bool(covers), "on_sheet": bool(on_sheet), "above": bool(is_above),
              "left": bool(is_left), "extent_mm": [_mm(v) for v in (x0, x1, y0, y1)]})
    if ok and not label_ok:
        ok, why = False, "label_not_correct"
    if ok and square_err_deg(st) > L.SQUARE_DEG + L.ANG_EPS:
        ok, why = False, "not_square"
    if ok and upright_err_deg(st) > L.SQUARE_DEG + L.ANG_EPS:
        ok, why = False, "upside_down"
    if ok and covers:
        ok, why = False, "overlaps_label"
    if ok and not on_sheet:
        ok, why = False, "off_sheet"
    if ok and not (is_above or is_left):
        ok, why = False, "not_next_to_label"
    return ok, {**d, "correct": ok, "zero_reason": why}


def judge_specimen(st, label_st, slip_st, still: bool, off: bool) -> tuple[bool, dict]:
    ok, why, d = laid(st, "specimen", still, off)
    x0, x1, y0, y1 = extent(st, "specimen")
    margin = min(x0 - L.SHEET_X0, L.SHEET_X1 - x1, y0 - L.SHEET_Y0, L.SHEET_Y1 - y1)
    f = footprint(st, "specimen")
    covers = {"label": overlap(f, footprint(label_st, "label")), "slip": overlap(f, footprint(slip_st, "slip")),
              "barcode": overlap(f, BARCODE)}
    d.update({"root_err_deg": round(root_err_deg(st), 2), "heading_deg": round(heading_deg(st), 2),
              "margin_mm": _mm(margin), "covers": covers, "extent_mm": [_mm(v) for v in (x0, x1, y0, y1)]})
    if ok and margin < L.SPEC_MARGIN - L.EPS:
        ok, why = False, "not_on_sheet"
    if ok and root_err_deg(st) > L.ROOT_DEG + L.ANG_EPS:
        ok, why = False, ("roots_toward_head" if root_err_deg(st) > 90.0 else "not_along_sheet")
    if ok and any(covers.values()):
        ok, why = False, "covers_" + "_and_".join(k for k, v in covers.items() if v)
    return ok, {**d, "correct": ok, "zero_reason": why}


def evaluate(items: dict, still: dict, off: dict) -> tuple[float, dict]:
    """The score of a state (every item valid) and the details for the trace."""
    lab, dl = judge_label(items["label"], still["label"], off["label"])
    slp, ds = judge_slip(items["slip"], items["label"], lab, still["slip"], off["slip"])
    spc, dp = judge_specimen(items["specimen"], items["label"], items["slip"], still["specimen"], off["specimen"])
    score = L.WEIGHTS["label"] * lab + L.WEIGHTS["slip"] * slp + L.WEIGHTS["specimen"] * spc
    return round(float(score), 6), {"label": dl, "slip": ds, "specimen": dp,
                                    "correct": [n for n, v in (("label", lab), ("slip", slp), ("specimen", spc)) if v]}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each item's top corners over the last second, and how many steps the
    arms could not be read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf, a zero quaternion or a missing origin
        self.arms_unavailable = 0
        self.history: dict[str, deque] = {}  # item -> its top corners at this step and the 20 before it
        self.prev = None  # {item: (origin, velocity)} at the previous valid control step
        self.first_at: dict[str, int] = {}  # each score reached -> the first control step it was reached
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(items: dict) -> bool:
        """The starting layout: every item lying flat on the table on its own starting mark, still."""
        for name in L.ITEMS:
            st = items[name]
            sx, sy, syaw = L.START[name]
            o = np.asarray(st.origin, dtype=float)
            if math.hypot(o[0] - sx, o[1] - sy) > L.START_TOL:
                return False
            dyaw = (heading_deg(st) - syaw + 180.0) % 360.0 - 180.0
            if abs(dyaw) > L.START_YAW_TOL_DEG or tilt_deg(st) > 2.0 or abs(lowest(st, name)) > 0.003:
                return False
            try:
                if float(np.linalg.norm(np.asarray(st.lin_vel, dtype=float))) > 0.05:
                    return False
            except (AttributeError, TypeError, ValueError):
                pass
        return True

    def _teleported(self, now: dict) -> bool:
        """Some item moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for n, (p0, v0) in self.prev.items() if n in now)

    def paths(self, name: str) -> list[float] | None:
        """For each of the last two 0.5 s windows (older first), the longest path any of the item's four top corners
        travelled in it (m); None until 1 s of this trial has been seen."""
        h = self.history.get(name)
        need = self.window * L.STILL_WINDOWS + 1
        if h is None or len(h) < need:
            return None
        pts = list(h)[-need:]
        out = []
        for k in range(L.STILL_WINDOWS):
            seg = pts[k * self.window:(k + 1) * self.window + 1]
            per_corner = np.sum([np.linalg.norm(b - a, axis=1) for a, b in zip(seg, seg[1:])], axis=0)
            out.append(float(np.max(per_corner)))
        return out

    def update(self, items: dict, arms=None) -> Result:
        self.steps += 1
        if not all(valid(items.get(n)) for n in L.ITEMS):  # a blow-up: score 0 for this step, keep everything else
            self.bad_steps += 1
            self.last = self.result(items, arms)
            return self.last
        now = {n: (np.array(items[n].origin, dtype=float), _vel(items[n])) for n in L.ITEMS}
        if self.prev is not None and self._teleported(now) and self.at_start(items):
            bad = self.bad_steps
            self.reset()  # the simulator put every item back on its mark: a new episode
            self.steps, self.bad_steps = 1, bad
        self.prev = now
        for n in L.ITEMS:
            self.history.setdefault(n, deque(maxlen=self.window * L.STILL_WINDOWS + 1)).append(
                body_corners(items[n], n, L.BODY_SIZE[n][2]))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(items, arms)
        if self.last.score > 0 and str(self.last.score) not in self.first_at:
            self.first_at[str(self.last.score)] = self.steps
        return self.last

    def result(self, items: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"step": self.steps, "bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable,
                 "first_step": dict(sorted(self.first_at.items()))}
        bad = sorted(n for n in L.ITEMS if not valid(items.get(n)))
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_items": bad, "zero_reason": "invalid_state", **trial})
        paths = {n: self.paths(n) for n in L.ITEMS}
        still = {n: p is not None and max(p) <= L.STILL_PATH + L.EPS for n, p in paths.items()}
        if arms_valid(arms):
            gaps = {n: min(zone_gap(arms["left"], items[n], n), zone_gap(arms["right"], items[n], n))
                    for n in L.ITEMS}
            off = {n: g >= L.HANDS_OFF - L.EPS for n, g in gaps.items()}
            zones = {n: _mm(g) for n, g in gaps.items()}
        else:  # not readable: taken as met, and counted
            off, zones = {n: True for n in L.ITEMS}, None
        score, details = evaluate(items, still, off)
        details["path_mm"] = {n: None if p is None else [round(v * 1000, 3) for v in p] for n, p in paths.items()}
        details["zone_gap_mm"] = zones
        return Result(score, bool(score >= 1.0 - 1e-9), {"valid_state": True, "zero_reason": None, **details, **trial})


def _vel(st) -> np.ndarray:
    try:
        v = np.asarray(st.lin_vel, dtype=float).ravel()
        return v if v.size == 3 and np.all(np.isfinite(v)) else np.zeros(3)
    except (AttributeError, TypeError, ValueError):
        return np.zeros(3)
