"""On the station table, real size: a bench top (240 x 400 mm, 40 mm over the table, three slots in it at the cut
marks) carries a flat block of bread dough, 256 x 80 x 40 mm, 0.90 kg, lying along y. Its three cut marks, at 64 mm,
128 mm and 192 mm from the right end, divide it into four portions of 64 mm. In front of each arm, on the table, a
6 x 3 in stainless bench scraper (bar handle) stands in a fixed rest.

Goal: four separate pieces: chop straight down through the dough at each mark with a scraper's edge, dragging nothing
sideways, then put the scrapers down clear of the dough and let go of both.

The dough, the bench, the rests and the scrapers are mechkit assemblies (mechkit.py next to this file). Each of the
four pieces is a loose body; the first three carry the cut strip of dough at the mark after them as a link on a
sliding joint (spring and damper: a drive that gives under a pressing edge at a speed that grows with the push, holds
where it is left and does not spring back once pressed 1 mm), running down its slot in the bench top. A fixed joint
(bond) holds each piece to the next until the strip between them is through. Declared rules: ``_dough_rule`` (a
mechkit on_step callback: the strip's drive target, and the bond released the step its strip is through). The arms
are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Divide the dough log into four separate pieces with the bench scrapers, cutting it at the three "
               "marks on the bench's edge, and put the scrapers down clear of the dough.")


# -- the bench, the rests ---------------------------------------------------------------------------------------------
def bench_shapes():
    slots = [(0.0, m - L.LOG_Y, L.SLOT_LX, L.SLOT_LY) for m in L.MARKS]
    shapes = list(mk.plate_with_holes((0.0, 0.0, L.BENCH_TOP - L.BENCH_T / 2), (L.BENCH_LX, L.BENCH_LY), slots,
                                      L.BENCH_T, L.BENCH_COLOR))
    rail_h = L.BENCH_TOP - L.BENCH_T
    for sx in (-1, 1):
        shapes.append(mk.box((sx * (L.BENCH_LX / 2 - L.RAIL_W / 2), 0.0, rail_h / 2), (L.RAIL_W, L.BENCH_LY, rail_h),
                             L.RAIL_COLOR))
    mx = (L.MARK_X0 + L.MARK_X1) / 2 - L.BENCH_X
    for m in L.MARKS:
        shapes.append(mk.box((mx, m, L.BENCH_TOP + L.MARK_H / 2), (L.MARK_X1 - L.MARK_X0, L.MARK_LEN_Y, L.MARK_H),
                             L.MARK_COLOR))
    return shapes


def rest_shapes():
    off = L.REST_GAP / 2 + L.REST_RAIL_Y / 2
    return [mk.box((0.0, s * off, L.REST_H / 2), (L.REST_LX, L.REST_RAIL_Y, L.REST_H), L.REST_COLOR)
            for s in (-1, 1)]


BENCH = mk.assembly("bench", origin=(L.BENCH_X, 0.0, L.TOP), shapes=bench_shapes(), color=L.BENCH_COLOR,
                    friction=L.BENCH_FRICTION, tight_fit=True)
REST_R = mk.assembly("rest_r", origin=(L.SC_X, L.SC_Y["right"], L.TOP), shapes=rest_shapes(), color=L.REST_COLOR,
                     tight_fit=True)
REST_L = mk.assembly("rest_l", origin=(L.SC_X, L.SC_Y["left"], L.TOP), shapes=rest_shapes(), color=L.REST_COLOR,
                     tight_fit=True)


# -- the dough: four pieces, a strip link on the first three ----------------------------------------------------------
def web_link(k: int):
    dy = L.MARKS[k] - L.BODY_YC[k]
    return mk.slider("web", "-Z", [mk.box((0.0, dy, 0.0), (L.LOG_W, L.WEB_W, L.LOG_H), L.DOUGH_COLOR)],
                     upper=L.WEB_TRAVEL, lower=0.0, mass=L.WEB_MASS, spring=L.WEB_SPRING, damping=L.WEB_DAMP)


def piece(k: int):
    joints = [web_link(k)] if k < 3 else []
    return mk.assembly(f"p{k}", origin=(L.LOG_X, L.BODY_YC[k], L.PIECE_Z), mass=L.BODY_MASS[k],
                       shapes=[mk.box((0.0, 0.0, 0.0), (L.LOG_W, L.BODY_LEN[k], L.LOG_H), L.DOUGH_COLOR)],
                       joints=joints, color=L.DOUGH_COLOR, friction=L.DOUGH_FRICTION, tight_fit=True)


PIECE_PARTS = [piece(k) for k in range(4)]
BONDS = [mk.bond(f"cut{k}", PIECE_PARTS[k + 1], PIECE_PARTS[k]) for k in range(3)]


def scraper(side: str):
    return mk.assembly(f"scraper_{side[0]}", origin=(L.SC_X, L.SC_Y[side], L.TOP), mass=L.SC_MASS,
                       shapes=[mk.box((0.0, 0.0, L.SC_H / 2), (L.SC_W, L.SC_T, L.SC_H), L.SC_COLOR),
                               mk.box((0.0, 0.0, L.BAR_Z), (L.BAR_LX, L.BAR_LY, L.BAR_LZ), L.BAR_COLOR)],
                       color=L.SC_COLOR, friction=L.SC_FRICTION, tight_fit=True)


SCRAPER_R, SCRAPER_L = scraper("right"), scraper("left")
OBJECTS = [BENCH, REST_R, REST_L, *PIECE_PARTS, SCRAPER_R, SCRAPER_L]
SPECS = {p.name: p for p in (*PIECE_PARTS, SCRAPER_R, SCRAPER_L)}

# -- the declared rule: the strip's drive target and the bond ----------------------------------------------------------
SAG = L.WEB_MASS * L.G / L.WEB_SPRING  # the drive's give under a strip's own weight (m): held off so it does not creep
_RULE = {"through": [False, False, False]}


def strip_target(q: float, through: bool) -> float:
    """The strip's drive target (m down its slot) for this step: once through, all the way down; flush (less than
    1 mm in), a push of YIELD_F up against its stop (the edge must press harder than that to start it, and it springs
    back if let go); past 1 mm, held where it is, less its own weight."""
    if through:
        return L.WEB_TRAVEL
    if q < L.YIELD_Q:
        return -L.YIELD_F / L.WEB_SPRING
    return q - SAG


@mk.on_step
def _dough_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    for k in range(3):
        st = states.get(f"p{k}")
        try:
            q = float(st.joints["web"])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(q):
            continue
        if q >= L.WEB_THROUGH and not _RULE["through"][k]:
            _RULE["through"][k] = True
            BONDS[k].release()  # the strip is through: the two pieces are no longer one dough
        mk.set_targets(PIECE_PARTS[k], "web", strip_target(q, _RULE["through"][k]))


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped scraper rigidly and drops it when the grip opens; it neither presses a strip
# down nor holds the dough together. Offline only (never in the simulator), this model presses each strip down to the
# edge of an upright scraper whose centre line is in it, as the strip's drive would (down, never back up), and keeps
# the four pieces where they began.
class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.depth = [0.0, 0.0, 0.0]

    def step(self, objects) -> None:
        log_top = L.BENCH_TOP + L.LOG_H
        for name in S.SCRAPERS:
            st = objects.get(name)
            if st is None:
                continue
            o = np.asarray(st.origin, dtype=float) - np.array([0.0, 0.0, L.TOP])
            rot = S.rotation(st.quat_xyzw)
            c = math.cos(math.radians(L.BLADE_UPRIGHT_DEG))
            if not (rot[2, 2] >= c and abs(rot[0, 0]) >= c):
                continue
            for k in range(3):
                if abs(o[1] - L.web_y(k)) <= L.WEB_W / 2 + L.CUT_OK_MARGIN and abs(o[0] - L.LOG_X) <= L.CUT_X:
                    d = min(L.WEB_TRAVEL, max(0.0, log_top - float(o[2])))
                    self.depth[k] = max(self.depth[k], d)

    def states(self, objects) -> dict:
        out = {n: objects.get(n) for n in S.SCRAPERS}
        for k, spec in enumerate(PIECE_PARTS):
            joints = {"web": self.depth[k]} if k < 3 else {}
            out[spec.name] = mk.state_from_poses(spec, (*spec.pos, *mk.yaw_quat(0.0)), joints=joints,
                                                 source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(S.SCRAPERS[0]), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, strip depths mm, cuts]


def _fresh() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()
    _RULE["through"] = [False, False, False]


@rigstate.on_reset
def _new_episode() -> None:
    _fresh()


@mk.on_reset
def _kit_reset() -> None:
    _fresh()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every key part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Three cuts through at the marks by a scraper's edge, no crease, nothing dragged, the pieces in line, both
    scrapers down clear of the dough, both grippers 30 mm clear and every part still for 1 s, no latched fall; held
    1 s. This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _fresh()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(parts)
        parts = _OFFLINE.states(parts)
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get("p0"), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["web_mm"], d["cuts"], d["scraper_mm"]["scraper_r"][1:],
                      d["scraper_mm"]["scraper_l"][1:]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a piece went off the bench, or unless every part is let go and still, or with no cut through; 1.0 for three
    cuts through at the marks by a scraper's edge with no crease, nothing dragged, the pieces in line and both
    scrapers down clear of the dough; 0.5 for three cuts otherwise or two cuts; 0.25 for one cut. The score of the
    latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips
# are 13 mm below. Finger targets: the joint value is minus half the commanded gap (6 mm narrower than the bar).
OPEN = -0.04695  # 94 mm
GRIP = -(L.BAR_LY - 0.006) / 2  # 16 mm commanded across the 22 mm bar
YAW = 0.0  # both arms close along y: across the bar, the blade in the x-z plane
BAR_TCP = L.BAR_Z  # the TCP at the bar's centre with the blade's edge on the table (in its rest)
CARRY_TCP = 0.172  # edge 88 mm up: 8 mm over the dough's top
CHOP_TCP = L.BENCH_TOP + L.BAR_Z - 0.002  # the edge pressed 2 mm into the bench top: 122 mm
HOVER = 0.150
CHOP_SECONDS = 3.0  # the strip's whole 48 mm from the carry height at about 16 mm/s
PARK = {"right": (0.14, -0.40, 0.150), "left": (0.14, 0.40, 0.150)}


def _w(d, x, y, z, finger, label):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, YAW)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def pick_up(side):
    x, y = L.SC_X, L.SC_Y[side]
    return [
        _w(2.0, x, y, HOVER, OPEN, "over the scraper's bar in its rest"),
        _w(1.0, x, y, BAR_TCP + 0.020, OPEN, "down to just above the bar"),
        _w(1.0, x, y, BAR_TCP, OPEN, "down round the bar"),
        _w(0.5, x, y, BAR_TCP, OPEN, "settle"),
        _w(1.5, x, y, BAR_TCP, GRIP, "grip the bar"),
        _w(1.5, x, y, BAR_TCP + 0.050, GRIP, "lift the scraper out of its rest"),
        _w(1.0, x, y, CARRY_TCP, GRIP, "up to carry height"),
    ]


def chop(y_mark, y_aim=None, depth_tcp=CHOP_TCP):
    """From the carry height over the mark: down through the dough, wait, straight back up. ``y_aim`` is where the
    blade is put (a counterfactual puts it off the mark)."""
    x, y = L.SC_X, y_mark if y_aim is None else y_aim
    return [
        _w(CHOP_SECONDS, x, y, depth_tcp, GRIP, f"chop straight down at y {y_mark * 1000:.0f} mm"),
        _w(0.8, x, y, depth_tcp, GRIP, "hold the edge down"),
        _w(2.0, x, y, CARRY_TCP, GRIP, "straight back up"),
    ]


def travel(y_from, y_to, seconds):
    return [_w(seconds, L.SC_X, y_to, CARRY_TCP, GRIP, f"over y {y_to * 1000:.0f} mm")]


def put_back(side, seconds_in=2.0):
    x, y = L.SC_X, L.SC_Y[side]
    return [
        _w(3.0, x, y, CARRY_TCP, GRIP, "back over the rest"),
        _w(seconds_in, x, y, BAR_TCP + 0.020, GRIP, "down toward the rest"),
        _w(1.5, x, y, BAR_TCP + 0.002, GRIP, "slowly into the rest"),
        _w(0.5, x, y, BAR_TCP + 0.002, GRIP, "settle"),
        _w(1.5, x, y, BAR_TCP + 0.002, OPEN, "let go"),
        _w(1.0, x, y, HOVER, OPEN, "straight up"),
        _w(1.5, *PARK[side][:2], PARK[side][2], OPEN, "aside"),
    ]


# Counterfactuals: where each goes wrong. offmark: the right arm's second chop 8 mm off the mark (the edge lands on
# the dough beside the strip: no cut); shallow: the third chop stops 22 mm into the dough (a crease, no cut).
VARIANTS = ("offmark", "shallow")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: each arm lifts its scraper out of its rest. The right arm chops the marks at y -64 mm and 0, then
    puts its scraper back; as it leaves the middle the left arm brings its scraper over y +64 mm, chops, and puts
    it back."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    ym = L.MARKS
    right = pick_up("right") + travel(L.SC_Y["right"], ym[0], 3.0) + chop(ym[0])
    right += travel(ym[0], ym[1], 2.0)
    right += chop(ym[1], y_aim=ym[1] + 0.008 if variant == "offmark" else None)
    right += put_back("right")
    left = pick_up("left")
    left_wait = _duration(right) - _duration(put_back("right")) - _duration(left)
    left.append(_w(left_wait, L.SC_X, L.SC_Y["left"], CARRY_TCP, GRIP, "hold the scraper up while the right arm chops"))
    left += travel(L.SC_Y["left"], ym[2], 3.0)
    left += chop(ym[2], depth_tcp=CHOP_TCP + 0.026 if variant == "shallow" else CHOP_TCP)
    left += put_back("left")
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      offmark  the right arm's second chop put 8 mm off the mark: no cut there (0.5 with the other two)
      shallow  the left arm's chop stops 22 mm into the dough: a crease, no cut (0.5 with the other two)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
