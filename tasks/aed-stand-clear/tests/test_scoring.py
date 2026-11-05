"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic pad states and synthetic arm snapshots
fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v

Most trials are played through ``Sim``, a small kinematic stand-in for the station: the arms follow waypoint plans
(task.demo()'s, or edited and hand-driven ones for the exploits), optionally lagging behind their commands (a
first-order lag, which also shrinks the compression strokes); a gripper that closes on a pad carries it rigidly and a
pad let go drops onto whatever is under it. It cannot push or tip things; those trials move the pads directly with
``Sim.slide`` or ``Sim.put``.
"""

import importlib
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


STUB = _stub_dir()
if STUB is not None:
    sys.path.insert(0, str(STUB))
    task = importlib.import_module("task")
    from yam_arena.demo import Waypoint
else:  # pragma: no cover
    task = Waypoint = None

needs_stub = unittest.skipIf(STUB is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")

HZ = 20.0
DT = 1.0 / HZ
OPEN = -0.04695
FIST = -0.008
TIPS = 0.013
READY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
HALF = np.array(L.PAD_SIZE) / 2
PADS = L.PADS
START = {"pad_a": np.array([L.PAD_A_X, L.PAD_A_Y, L.TOP + HALF[2]]),
         "pad_b": np.array([L.PAD_B_X, L.PAD_B_Y, L.TOP + HALF[2]])}
ON_STERNAL = np.array([L.STERNAL_X, L.STERNAL_Y, L.MARK_TOP + HALF[2]])
ON_APEX = np.array([L.APEX_X, L.APEX_Y, L.MARK_TOP + HALF[2]])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def pad(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like pad: centre, orientation, velocities, world bounding box."""
    c, q = np.array(center, float), np.array(quat, float)
    ext = np.abs(S.rotation(q)) @ HALF
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c - ext, high=c + ext)


def arm(tips, opening):
    tips = np.array(tips, float)
    return rigstate.Arm(tips + [0, 0, TIPS], tips, float(opening))


def fist(tips):
    return arm(tips, 0.016 / L.FULL_GAP)


def tips_over_mark(h, dx=0.0, dy=0.0):
    return (L.CPR_X + dx, L.CPR_Y + dy, L.MARK_TOP + h)


def parked_arms():
    return {s: arm(READY[s] - [0, 0, TIPS], 1.0) for s in READY}


def support_z(x, y):
    """The surface under a pad's centre: a mark on the chest (if its outline overlaps one), the chest, the head, the
    AED, the table, or the floor off the table."""
    lo, hi = S.TORSO_BOX
    if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
        marks = [(L.CPR_X, L.CPR_Y, L.CPR_MARK / 2), (L.STERNAL_X, L.STERNAL_Y, L.SITE_MARK / 2),
                 (L.APEX_X, L.APEX_Y, L.SITE_MARK / 2)]
        if any(abs(x - mx) < h + HALF[0] and abs(y - my) < h + HALF[1] for mx, my, h in marks):
            return L.MARK_TOP
        return L.TORSO_TOP
    lo, hi = S.HEAD_BOX
    if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
        return L.HEAD_TOP
    if abs(x - L.AED_X) <= L.AED_SIZE[0] / 2 and abs(y - L.AED_Y) <= L.AED_SIZE[1] / 2:
        return L.AED_TOP
    if -0.10 <= x <= 0.60 and abs(y) <= 0.60:
        return L.TOP
    return L.TOP - 0.75  # the floor


class Sim:
    """The kinematic stand-in: two arms on waypoint plans, two loose pads, the tracker fed every control step."""

    def __init__(self, lag=None, arms=True, tracker=None):
        self.lag = dict(lag or {})  # side -> first-order time constant (s)
        self.read_arms = arms
        self.t = tracker or Tracker(HZ)
        self.pos = {n: START[n].copy() for n in PADS}
        self.quat = {n: Q0.copy() for n in PADS}
        self.prev = {n: self.pos[n].copy() for n in PADS}
        self.cmd = {s: READY[s].copy() for s in READY}
        self.fin = {s: OPEN for s in READY}
        self.tcp = {s: READY[s].copy() for s in READY}
        self.att = {"left": None, "right": None}  # side -> (pad, offset from the TCP)
        self.results = []
        self.frame()

    # -- one control step ------------------------------------------------------------------------------------
    def _drop(self, n):
        self.pos[n][2] = support_z(*self.pos[n][:2]) + HALF[2]

    def _arm(self, side):
        tau = self.lag.get(side, 0.0)
        a = DT / (tau + DT) if tau > 0 else 1.0
        self.tcp[side] = self.tcp[side] + (self.cmd[side] - self.tcp[side]) * a
        gap = -2 * self.fin[side]
        held = self.att[side]
        if held is not None:
            n, off = held
            if gap > L.PAD_SIZE[0] + 0.002:  # opened: let go, it drops onto what is under it
                self.att[side] = None
                self._drop(n)
            else:
                self.pos[n] = self.tcp[side] + off
            return
        tips = self.tcp[side] - [0, 0, TIPS]
        for n in PADS:
            if any(a is not None and a[0] == n for a in self.att.values()):
                continue
            c = self.pos[n]
            inside_z = c[2] - HALF[2] <= tips[2] <= c[2] + HALF[2]
            near = abs(tips[0] - c[0]) <= HALF[0] - 0.005 and abs(tips[1] - c[1]) <= HALF[1] - 0.005
            if near and inside_z and gap <= L.PAD_SIZE[0]:
                self.att[side] = (n, c - self.tcp[side])
                return

    def snapshot(self):
        out = {}
        for side in READY:
            gap = -2 * self.fin[side]
            if self.att[side] is not None:
                gap = max(gap, L.PAD_SIZE[0])
            out[side] = {"tcp": self.tcp[side].copy(), "tips": self.tcp[side] - np.array([0, 0, TIPS]),
                         "opening": float(np.clip(gap / L.FULL_GAP, 0, 1))}
        return out

    def states(self):
        return [pad(self.pos[n], self.quat[n], (self.pos[n] - self.prev[n]) / DT) for n in PADS]

    def frame(self):
        for side in READY:
            self._arm(side)
        sts = self.states()
        if self.read_arms:
            rigstate.offline_feed(self.snapshot())
            arms = rigstate.arms()
        else:
            arms = None
        r = self.t.update(*sts, arms)
        self.results.append(r)
        self.prev = {n: self.pos[n].copy() for n in PADS}
        return r

    # -- driving it ------------------------------------------------------------------------------------------
    def _track(self, wps, side):
        out, pos, fin = [], self.cmd[side].copy(), self.fin[side]
        for w in wps:
            m = max(1, int(round(w.duration * HZ)))
            for i in range(1, m + 1):
                f = i / m
                out.append((pos + (np.array(w.pos) - pos) * f, fin + (w.finger - fin) * f))
            pos, fin = np.array(w.pos, float), w.finger
        return out

    def play(self, plan, tail_s=4.0, stop_s=None):
        """Run both arms' waypoint plans on one clock, then hold everything for ``tail_s`` (or stop at ``stop_s``
        of plan time, leaving each arm where it is)."""
        tracks = {s: self._track(plan.get(s, []), s) for s in READY}
        n = max(len(v) for v in tracks.values()) + int(round(tail_s * HZ))
        if stop_s is not None:
            n = int(round(stop_s * HZ))
        for k in range(n):
            for s, seq in tracks.items():
                if seq:
                    p, f = seq[min(k, len(seq) - 1)]
                    self.cmd[s], self.fin[s] = np.array(p), f
            self.frame()
        return self

    def drive(self, moves, s):
        """Move one or both arms' commands linearly over ``s`` seconds: ``moves`` maps a side to (pos, finger)."""
        start = {side: (self.cmd[side].copy(), self.fin[side]) for side in moves}
        n = max(1, int(round(s * HZ)))
        for i in range(1, n + 1):
            f = i / n
            for side, (p, fin) in moves.items():
                p0, f0 = start[side]
                self.cmd[side] = p0 + (np.array(p, float) - p0) * f
                if fin is not None:
                    self.fin[side] = f0 + (fin - f0) * f
            self.frame()
        return self

    def hold(self, s):
        for _ in range(int(round(s * HZ))):
            self.frame()
        return self

    def strokes(self, n, half_s=0.5, top=0.040, bottom=0.008):
        """n compressions with the right hand: down to ``bottom`` and up to ``top`` above the mark (tips)."""
        for _ in range(n):
            self.drive({"right": (np.array(tips_over_mark(bottom)) + [0, 0, TIPS], None)}, half_s)
            self.drive({"right": (np.array(tips_over_mark(top)) + [0, 0, TIPS], None)}, half_s)
        return self

    def put(self, name, center, quat=None):
        """Teleport-free placement of a pad over one step (as if pushed into place quickly)."""
        self.pos[name] = np.array(center, float)
        if quat is not None:
            self.quat[name] = np.array(quat, float)
        return self.frame()

    def slide(self, name, to, s=1.0):
        """Move a pad directly (pushed, swept), linearly over ``s`` seconds."""
        a, b = self.pos[name].copy(), np.array(to, float)
        n = max(1, int(round(s * HZ)))
        for i in range(1, n + 1):
            self.pos[name] = a + (b - a) * i / n
            self.frame()
        return self

    @property
    def last(self):
        return self.results[-1]

    @property
    def now_s(self):
        return (self.t.steps - 1) * DT

    def first_success(self):
        return next((i for i, r in enumerate(self.results) if r.success), None)


def demo_plan(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return task.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def W(d, p, f, label="", yaw=0.0):
    return Waypoint(d, tuple(p), f, label, yaw)


def idx(wps, label_part):
    return next(i for i, w in enumerate(wps) if label_part in w.label)


def span(wps):
    return sum(w.duration for w in wps)


def with_duration(wps, label_part, d):
    i = idx(wps, label_part)
    w = wps[i]
    return wps[:i] + [Waypoint(d, w.pos, w.finger, w.label, w.yaw)] + wps[i + 1:]


def right_tcp(h):
    return tuple(np.array(tips_over_mark(h)) + [0, 0, TIPS])


def strokes_wp(n):
    out = []
    for _ in range(n):
        out += [W(0.5, right_tcp(0.008), FIST, "down"), W(0.5, right_tcp(0.040), FIST, "up")]
    return out


def stand_clear_tracker():
    """A tracker with both pads placed during compressions (synthetic arms), then everyone clear: returns it and the
    pads, at the step both grippers became clear."""
    t = Tracker(HZ)
    a, b = pad(START["pad_a"]), pad(START["pad_b"])
    left_away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
    right_up = fist(tips_over_mark(0.040))
    k = 0
    for h in [0.040, 0.008] * 4:  # four compressions with pad A arriving during them
        k += 1
        pa = pad(ON_STERNAL) if k >= 3 else a
        pb = pad(ON_APEX) if k >= 5 else b
        for _ in range(10):
            t.update(pa, pb, {"left": left_away, "right": fist(tips_over_mark(h))})
    del right_up
    return t, pad(ON_STERNAL), pad(ON_APEX)


# ---------------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_sites_and_flatness(self):
        self.assertEqual(S.site_of(pad(ON_STERNAL)), "sternal")
        self.assertEqual(S.site_of(pad(ON_APEX)), "apex")
        self.assertIsNone(S.site_of(pad(START["pad_a"])))  # on the table
        self.assertEqual(S.site_of(pad(ON_APEX + [0.0199, 0, 0])), "apex")
        self.assertIsNone(S.site_of(pad(ON_APEX + [0.0201, 0, 0])))
        self.assertEqual(S.site_of(pad(ON_APEX + [0.014, 0.014, 0])), "apex")  # 19.8 mm diagonally
        self.assertIsNone(S.site_of(pad(ON_APEX + [0.015, 0.015, 0])))  # 21.2 mm
        self.assertEqual(S.site_of(pad(ON_APEX + [0, 0, 0.0019])), "apex")  # 2.9 mm above the torso top
        self.assertIsNone(S.site_of(pad(ON_APEX + [0, 0, 0.0025])))  # hovering 3.5 mm up
        self.assertEqual(S.site_of(pad(ON_APEX, about("z", 37))), "apex")  # turned: still flat
        self.assertEqual(S.site_of(pad(ON_APEX, about("x", 180))), "apex")  # upside down: either face
        tilted = pad(ON_APEX, about("y", 12))
        tilted.low[2] = L.MARK_TOP  # resting on an edge
        self.assertIsNone(S.site_of(tilted))
        self.assertTrue(S.both_on(pad(ON_STERNAL), pad(ON_APEX)))
        self.assertTrue(S.both_on(pad(ON_APEX), pad(ON_STERNAL)))  # swapped: the pads are identical
        self.assertFalse(S.both_on(pad(ON_APEX), pad(ON_APEX + [0, 0, 0.024])))  # stacked on one site

    def test_compression_point_rule(self):
        self.assertTrue(S.on_compression_point(pad((L.CPR_X, L.CPR_Y, L.MARK_TOP + HALF[2]))))
        edge = L.CPR_MARK / 2 + HALF[0]
        self.assertTrue(S.on_compression_point(pad((L.CPR_X + edge - 0.001, L.CPR_Y, L.MARK_TOP + HALF[2]))))
        self.assertFalse(S.on_compression_point(pad((L.CPR_X + edge + 0.001, L.CPR_Y, L.MARK_TOP + HALF[2]))))
        self.assertFalse(S.on_compression_point(pad((L.CPR_X, L.CPR_Y, L.TORSO_TOP + 0.006 + HALF[2]))))  # carried
        self.assertFalse(S.on_compression_point(pad(ON_STERNAL)))
        self.assertFalse(S.on_compression_point(pad(ON_APEX)))

    def test_hand_on_the_chest(self):
        self.assertTrue(S.on_chest(fist(tips_over_mark(0.010))))
        self.assertTrue(S.on_chest(fist(tips_over_mark(0.0449))))
        self.assertFalse(S.on_chest(fist(tips_over_mark(0.046))))
        self.assertTrue(S.on_chest(fist(tips_over_mark(0.010, dx=0.0199))))
        self.assertFalse(S.on_chest(fist(tips_over_mark(0.010, dx=0.0201))))
        self.assertTrue(S.on_chest(arm(tips_over_mark(0.010), 0.030 / L.FULL_GAP)))
        self.assertFalse(S.on_chest(arm(tips_over_mark(0.010), 0.031 / L.FULL_GAP)))  # not closed
        self.assertFalse(S.on_chest(arm(tips_over_mark(0.010), 1.0)))  # open hand

    def test_clear_zone(self):
        self.assertTrue(S.at_patient(fist(tips_over_mark(0.040))))
        # straight above the chest only the height counts: the zone's top is 50 mm over the torso's top
        self.assertTrue(S.at_patient(fist((L.CPR_X, L.CPR_Y, L.TORSO_TOP + 0.0499))))
        self.assertFalse(S.at_patient(fist((L.CPR_X, L.CPR_Y, L.TORSO_TOP + 0.0501))))
        self.assertFalse(S.at_patient(arm((L.CPR_X, L.CPR_Y, L.TORSO_TOP + 0.0501), 1.0)))  # open, still clear
        # beside the torso at table height: 50 mm plus the finger zone's radius
        x_edge = float(S.TORSO_BOX[0][0])
        closed_r = S.zone_radius(fist((0, 0, 0)))
        self.assertAlmostEqual(closed_r, math.hypot(0.008 + 0.008, 0.0135), places=6)
        self.assertTrue(S.at_patient(fist((x_edge - 0.05 - closed_r + 0.001, L.CPR_Y, L.TOP + 0.02))))
        self.assertFalse(S.at_patient(fist((x_edge - 0.05 - closed_r - 0.001, L.CPR_Y, L.TOP + 0.02))))
        open_r = S.zone_radius(arm((0, 0, 0), 1.0))
        self.assertGreater(open_r, 0.056)
        self.assertTrue(S.at_patient(arm((x_edge - 0.05 - open_r + 0.001, L.CPR_Y, L.TOP + 0.02), 1.0)))
        # the head counts too: the pads' starting places are next to it
        self.assertTrue(S.at_patient(arm(START["pad_a"] + [0, 0, -0.008], 1.0)))
        self.assertTrue(S.at_patient(arm(START["pad_b"] + [0, 0, -0.008], 1.0)))
        # the AED's button is clear of the manikin for a closed and for an open hand
        at_button = (L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002)
        self.assertFalse(S.at_patient(fist(at_button)))
        self.assertFalse(S.at_patient(arm(at_button, 1.0)))
        self.assertGreater(S.clear_margin(arm(at_button, 1.0)), 0.02)
        # a tilted gripper is judged along its fingers, not only at its tip point
        tips = np.array([x_edge - 0.05 - closed_r - 0.005, L.CPR_Y, L.TOP + 0.02])
        leaning = rigstate.Arm(tips + [0.030, 0, 0.005], tips, 0.016 / L.FULL_GAP)
        self.assertTrue(S.at_patient(leaning))

    def test_button(self):
        self.assertTrue(S.on_button(fist((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002))))
        self.assertTrue(S.on_button(fist((L.AED_X + 0.0149, L.AED_Y, L.BUTTON_TOP + 0.0059))))
        self.assertFalse(S.on_button(fist((L.AED_X + 0.0151, L.AED_Y, L.BUTTON_TOP + 0.002))))
        self.assertFalse(S.on_button(fist((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.0061))))
        self.assertFalse(S.on_button(arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002), 1.0)))  # open: straddles it

    def test_layout_facts(self):
        # the patient lies face up with the head toward +y: the patient's right (the sternal site's side) is -x
        self.assertLess(L.STERNAL_X, L.CPR_X)
        self.assertGreater(L.STERNAL_Y, L.CPR_Y)
        self.assertGreater(L.APEX_X, L.CPR_X)
        # both sites and the pads' starting places are on the left arm's side; the compression mark on the right's
        for x, y in (*L.SITES.values(), (L.PAD_A_X, L.PAD_A_Y), (L.PAD_B_X, L.PAD_B_Y)):
            self.assertGreater(y, 0.0)
        self.assertLess(L.CPR_Y, 0.0)
        # each pad, on its site, lies wholly on the torso's top
        lo, hi = S.TORSO_BOX
        for x, y in L.SITES.values():
            self.assertTrue(lo[0] + HALF[0] <= x <= hi[0] - HALF[0] and lo[1] + HALF[1] <= y <= hi[1] - HALF[1])


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Demo(unittest.TestCase):
    """The scripted demo through the kinematic stand-in, with and without tracking lag."""

    LAGS = ({}, {"left": 0.1, "right": 0.1}, {"left": 0.3}, {"right": 0.3}, {"left": 0.15, "right": 0.05},
            {"left": 0.3, "right": 0.3})

    def test_demo_scores_one(self):
        for lag in self.LAGS:
            sim = Sim(lag).play(demo_plan())
            r = sim.last
            d = r.details
            self.assertEqual(r.score, 1.0, f"lag {lag}: {d}")
            self.assertTrue(r.success)
            self.assertEqual(d["breaches"], [])
            self.assertEqual(d["charges"], 1)
            self.assertTrue(d["shock_qualified"])
            self.assertLessEqual(d["pre_shock_pause_s"], L.MAX_PAUSE_S - 3.5)  # at least 3.5 s to spare
            self.assertEqual(d["max_pause_s"], d["pre_shock_pause_s"])  # no longer pause than the one before the shock
            self.assertLessEqual(d["post_shock_pause_s"], L.POST_SHOCK_S - 0.75)  # at least 0.75 s to spare
            self.assertLessEqual(d["first_compression_s"], L.MAX_PAUSE_S - 4.0)
            self.assertGreaterEqual(d["min_tip_mm"], 5.0)  # the closed hand never comes within 5 mm of the chest

    def test_demo_timing(self):
        plan = demo_plan()
        sim = Sim().play(plan)
        first = sim.first_success()
        self.assertIsNotNone(first)
        self.assertTrue(all(r.success for r in sim.results[first:]))  # and it holds from then on
        end = max(span(w) for w in plan.values())
        d = sim.last.details
        # the AED charged and shocked when the plan says, within a step or two
        shock_t = (d["shock_step"] - 1) * DT
        self.assertAlmostEqual(shock_t, task.SHOCK_S, delta=0.11)
        charged = next(i for i, r in enumerate(sim.results) if r.details.get("charges", 0) >= 1)
        self.assertAlmostEqual(charged * DT, task.CHARGED_S, delta=0.11)
        # the trial ends (success held 3 s) before the plan and well inside the run limit
        self.assertLess(first * DT + task.SUCCESS_HOLD_S, end)
        self.assertLess((end + 3.0) * HZ, L.RUN_LIMIT_STEPS)
        # each pad arrived on its site while compressions were in progress
        self.assertEqual(d["arrived_in_cpr"], {"pad_a": True, "pad_b": True})
        self.assertEqual(d["sites"], {"pad_a": "sternal", "pad_b": "apex"})
        print(f"\n[demo] plan {end:.1f} s; first compression {d['first_compression_s']} s; charged "
              f"{charged * DT:.2f} s; shock {shock_t:.2f} s; pre-shock pause {d['pre_shock_pause_s']} s; post-shock "
              f"pause {d['post_shock_pause_s']} s; compressions {d['compressions']}; success from step {first} "
              f"({first * DT:.2f} s), trial ends about {first * DT + task.SUCCESS_HOLD_S:.1f} s", file=sys.stderr)

    def test_one_arm_alone(self):
        # The left arm alone puts both pads on and shocks after a clean analysis, but with no compressions: 0.25.
        sim = Sim().play(demo_plan(DEMO_IDLE_ARM="right"))
        d = sim.last.details
        self.assertEqual(sim.last.score, 0.25)
        self.assertEqual(d["breaches"], [])
        self.assertIsNotNone(d["shock_step"])
        self.assertFalse(d["shock_qualified"])
        self.assertEqual(d["compressions"], 0)
        # The right arm alone compresses the whole time, but no pad is ever put on: 0.
        sim = Sim().play(demo_plan(DEMO_IDLE_ARM="left"))
        d = sim.last.details
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(d["breaches"], [])
        self.assertGreater(d["compressions"], 30)
        self.assertFalse(d["ever_both_on"])

    def test_doing_nothing_scores_zero(self):
        sim = Sim().play(demo_plan(DEMO_HOLD_ONLY="1"), tail_s=60.0)
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(sim.last.details["breaches"], [])
        self.assertEqual(sim.last.details["compressions"], 0)

    def test_without_arm_readings(self):
        # rigstate failing for the whole trial: no compression or press can be seen, nothing is breached, and the
        # pads alone give 0.25 (the README says so)
        sim = Sim(arms=False).play(demo_plan())
        self.assertEqual(sim.last.score, 0.25)
        self.assertGreater(sim.last.details["arms_unavailable"], 900)
        self.assertEqual(sim.last.details["compressions"], 0)


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Tiers(unittest.TestCase):
    """Each tier, from edited demo plans."""

    def plan(self):
        return demo_plan()

    def test_pre_shock_pause_too_long_scores_075(self):
        # the left waits 6 s longer before pressing (and the right waits with it): pre-shock pause about 11 s
        p = self.plan()
        p["left"] = with_duration(p["left"], "wait, clear", p["left"][idx(p["left"], "wait, clear")].duration + 6)
        i = idx(p["right"], "stay clear until the shock")
        p["right"] = with_duration(p["right"], "stay clear until the shock", p["right"][i].duration + 6)
        d = Sim().play(p).last.details
        self.assertEqual(Sim().play(p).last.score, 0.75)
        self.assertGreater(d["pre_shock_pause_s"], L.MAX_PAUSE_S)
        self.assertFalse(d["pauses_ok"])
        self.assertTrue(d["post_shock_ok"] and d["kept_up_ok"])

    def test_pre_shock_pause_just_inside_scores_one(self):
        p = self.plan()
        base = Sim().play(self.plan()).last.details["pre_shock_pause_s"]
        extra = L.MAX_PAUSE_S - base - 0.2
        p["left"] = with_duration(p["left"], "wait, clear", p["left"][idx(p["left"], "wait, clear")].duration + extra)
        i = idx(p["right"], "stay clear until the shock")
        p["right"] = with_duration(p["right"], "stay clear until the shock", p["right"][i].duration + extra)
        r = Sim().play(p).last
        self.assertEqual(r.score, 1.0, r.details)
        self.assertLessEqual(r.details["pre_shock_pause_s"], L.MAX_PAUSE_S)

    def test_post_shock_restart_too_slow_scores_075(self):
        p = self.plan()
        i = idx(p["right"], "stay clear until the shock")
        p["right"] = with_duration(p["right"], "stay clear until the shock", p["right"][i].duration + 2.5)
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.75)
        self.assertGreater(r.details["post_shock_pause_s"], L.POST_SHOCK_S)

    def test_compressions_not_kept_up_scores_075(self):
        # back on the chest at once, but only four compressions, then hands off
        p = self.plan()
        i = idx(p["right"], "hands back on the chest")
        p["right"] = p["right"][:i + 1] + strokes_wp(4) + [W(1.0, right_tcp(0.080), FIST, "hands off")]
        r = Sim().play(p, tail_s=10).last
        self.assertEqual(r.score, 0.75)
        self.assertLess(r.details["post_shock_run_s"], L.POST_RUN_S)
        self.assertTrue(r.details["post_shock_ok"])

    def test_a_long_gap_breaks_the_post_shock_run(self):
        p = self.plan()
        i = idx(p["right"], "hands back on the chest")
        p["right"] = (p["right"][:i + 1] + strokes_wp(3) + [W(2.0, right_tcp(0.040), FIST, "rest on the chest")]
                      + strokes_wp(8))
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.75)
        self.assertLess(r.details["post_shock_run_s"], 3.0)

    def test_late_start_scores_075(self):
        # both arms start 8 s late (everything else as in the demo): first compression after 10 s
        p = self.plan()
        p["left"] = [W(8.0, READY["left"], OPEN, "late")] + p["left"]
        p["right"] = [W(8.0, READY["right"], OPEN, "late")] + p["right"]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["pauses_ok"])
        self.assertGreater(r.details["first_compression_s"], L.MAX_PAUSE_S)
        self.assertTrue(r.details["shock_qualified"])

    def test_a_long_pause_before_the_pads_scores_075(self):
        # two compressions at once, then hands resting on the chest for 11 s, then the demo as written (both arms
        # shifted by the same time): every pad still arrives during compressions, but compressions paused 11 s
        p = self.plan()
        i = idx(p["right"], "hands on the chest")
        rest = [W(11.0, right_tcp(0.040), FIST, "hands resting, no compressions")]
        extra = strokes_wp(2) + rest
        p["right"] = p["right"][:i + 2] + extra + p["right"][i + 2:]
        p["left"] = [W(span(extra), READY["left"], OPEN, "shifted")] + p["left"]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.75)
        self.assertTrue(r.details["shock_qualified"])
        self.assertLess(r.details["first_compression_s"], L.MAX_PAUSE_S)
        self.assertLessEqual(r.details["pre_shock_pause_s"], L.MAX_PAUSE_S)
        self.assertGreater(r.details["max_pause_s"], L.MAX_PAUSE_S)
        self.assertFalse(r.details["pauses_ok"])

    def test_left_hand_back_on_the_manikin_scores_075(self):
        p = self.plan()
        p["left"] = p["left"] + [W(1.5, (0.30, 0.15, L.TOP + 0.14), FIST, "back over the chest"),
                                 W(1.0, (0.30, 0.15, L.TORSO_TOP + 0.03 + TIPS), FIST, "hand on the chest")]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["left_off_manikin"])

    def test_a_pad_moved_off_its_site_after_the_shock_scores_075(self):
        sim = Sim().play(self.plan())
        self.assertEqual(sim.last.score, 1.0)
        sim.slide("pad_b", ON_APEX + [0.04, 0.0, 0.0], s=0.5).hold(1.0)
        self.assertEqual(sim.last.score, 0.75)
        self.assertFalse(sim.last.details["both_on"])

    def test_analysis_without_a_shock_scores_half(self):
        # nobody presses; the right hand stays clear: the AED stays charged
        p = self.plan()
        p["left"] = p["left"][:idx(p["left"], "down to 2 cm above the button")]
        p["right"] = p["right"][:idx(p["right"], "hands back on the chest")]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["qualified_analysis"])
        self.assertTrue(r.details["charged"])
        self.assertIsNone(r.details["shock_step"])
        # nobody presses; compressions resume: the charge is cancelled, still 0.5
        p = self.plan()
        p["left"] = p["left"][:idx(p["left"], "down to 2 cm above the button")]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["charged"])
        self.assertEqual(r.details["charges"], 1)

    def test_pads_on_but_never_clear_scores_quarter(self):
        # the right hand never stops compressing: no analysis, no shock; the left's press is 0 (tested in Breaches),
        # so here the left never goes to the button
        p = self.plan()
        p["right"] = p["right"][:idx(p["right"], "stand clear for the analysis")] + strokes_wp(25)
        p["left"] = p["left"][:idx(p["left"], "over the shock button")]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["charges"], 0)

    def test_success_needs_stillness(self):
        sim = Sim().play(self.plan())
        self.assertTrue(sim.last.success)
        a, b = sim.states()
        moving = pad(b.pos, vel=(0.02, 0, 0))
        r = sim.t.update(a, moving, rigstate.arms())
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        spinning = pad(b.pos, spin=(0, 0, 0.6))
        self.assertFalse(sim.t.update(a, spinning, rigstate.arms()).success)

    def test_score_is_idempotent(self):
        sim = Sim().play(self.plan())
        a, b = sim.states()
        arms = rigstate.arms()
        steps = sim.t.steps
        self.assertEqual(sim.t.result(a, b, arms).score, 1.0)
        self.assertEqual(sim.t.result(a, b, arms).score, 1.0)
        self.assertEqual(sim.t.steps, steps)


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Breaches(unittest.TestCase):
    """The rules that score 0 whatever follows, and the AED's behaviour."""

    def test_shock_pressed_while_still_compressing(self):
        p = demo_plan()
        p["right"] = p["right"][:idx(p["right"], "stand clear for the analysis")] + strokes_wp(25)
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.0)
        self.assertIn("pressed_while_not_clear", r.details["breaches"])
        self.assertIsNone(r.details["shock_step"])  # never charged: nothing delivered, still 0

    def test_shock_pressed_with_the_hand_resting_on_the_chest_after_the_analysis(self):
        # clear analysis, the AED charges; the right hand goes back on the chest (no compression) before the press
        p = demo_plan()
        i = idx(p["right"], "stay clear until the shock")
        until = task.CHARGED_S + 0.3 - span(p["right"][:i])
        p["right"] = p["right"][:i] + [W(until, right_tcp(0.080), FIST, "wait"),
                                       W(0.5, right_tcp(0.030), FIST, "hand back on the chest, still"),
                                       W(5.0, right_tcp(0.030), FIST, "rest")]
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.0)
        self.assertIn("pressed_while_not_clear", r.details["breaches"])

    def test_hand_back_on_the_chest_during_the_press(self):
        # the press begins clear, but the right hand comes back down before the press has been held 0.5 s
        p = demo_plan()
        i = idx(p["right"], "stay clear until the shock")
        until = task.PRESS_S - 0.2 - span(p["right"][:i])
        p["right"] = p["right"][:i] + [W(until, right_tcp(0.080), FIST, "wait")] + [
            W(0.5, right_tcp(0.040), FIST, "back too early")] + strokes_wp(10)
        r = Sim().play(p).last
        self.assertEqual(r.score, 0.0)
        self.assertIn("touched_during_shock", r.details["breaches"])
        self.assertNotIn("pressed_while_not_clear", r.details["breaches"])

    def test_pressing_before_the_analysis_does_nothing(self):
        # the left presses 1 s before the AED charges and lets go: no shock, no breach; then it presses again
        t, a, b = stand_clear_tracker()
        away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        press = fist((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002))
        up = fist(tips_over_mark(0.080))
        for _ in range(40):
            t.update(a, b, {"left": away, "right": up})
        for _ in range(12):  # 0.6 s press, before the 3 s analysis is complete
            t.update(a, b, {"left": press, "right": up})
        self.assertIsNone(t.shock_step)
        self.assertEqual(t.breaches, {})
        self.assertFalse(t.charged)
        for _ in range(70):
            t.update(a, b, {"left": away, "right": up})
        self.assertTrue(t.charged)
        for _ in range(10):
            t.update(a, b, {"left": press, "right": up})
        self.assertIsNotNone(t.shock_step)

    def test_a_press_held_through_the_charge_does_nothing(self):
        t, a, b = stand_clear_tracker()
        press = fist((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002))
        away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        up = fist(tips_over_mark(0.080))
        for _ in range(120):  # pressing all through the analysis and 3 s past it
            t.update(a, b, {"left": press, "right": up})
        self.assertTrue(t.charged)
        self.assertIsNone(t.shock_step)
        t.update(a, b, {"left": away, "right": up})  # let go and press again
        for _ in range(9):
            t.update(a, b, {"left": press, "right": up})
        self.assertIsNone(t.shock_step)  # 0.45 s: not yet
        t.update(a, b, {"left": press, "right": up})
        self.assertIsNotNone(t.shock_step)
        self.assertEqual(t.breaches, {})

    def test_touching_during_the_analysis_restarts_it(self):
        t, a, b = stand_clear_tracker()
        away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        up, touch = fist(tips_over_mark(0.080)), fist(tips_over_mark(0.040))
        for _ in range(50):
            t.update(a, b, {"left": away, "right": up})
        self.assertGreater(t.analysis_run, 40)
        t.update(a, b, {"left": away, "right": touch})
        self.assertEqual(t.analysis_run, 0)
        for _ in range(59):
            t.update(a, b, {"left": away, "right": up})
        self.assertFalse(t.charged)
        t.update(a, b, {"left": away, "right": up})
        self.assertTrue(t.charged)

    def test_compressions_resumed_cancel_the_charge(self):
        t, a, b = stand_clear_tracker()
        away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        press = fist((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.002))
        up = fist(tips_over_mark(0.080))
        for _ in range(61):
            t.update(a, b, {"left": away, "right": up})
        self.assertTrue(t.charged)
        for h in (0.040, 0.010):  # one compression
            for _ in range(10):
                t.update(a, b, {"left": away, "right": fist(tips_over_mark(h))})
        self.assertFalse(t.charged)
        for _ in range(10):  # hands up again; the left presses at once: nothing (a new analysis has just begun)
            t.update(a, b, {"left": away, "right": up})
        for _ in range(12):
            t.update(a, b, {"left": press, "right": up})
        self.assertIsNone(t.shock_step)
        self.assertEqual(t.breaches, {})

    def test_a_pad_lifted_cancels_the_charge(self):
        t, a, b = stand_clear_tracker()
        away = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        up = fist(tips_over_mark(0.080))
        for _ in range(61):
            t.update(a, b, {"left": away, "right": up})
        self.assertTrue(t.charged)
        t.update(a, pad(ON_APEX + [0, 0, 0.01]), {"left": away, "right": up})
        self.assertFalse(t.charged)

    def test_one_shock_per_trial(self):
        sim = Sim().play(demo_plan())
        shock = sim.last.details["shock_step"]
        # everyone clear again for a long time: no second analysis, no second charge
        sim.drive({"right": (right_tcp(0.080), None)}, 0.5).hold(8.0)
        self.assertEqual(sim.last.details["charges"], 1)
        self.assertEqual(sim.last.details["shock_step"], shock)
        self.assertFalse(sim.last.details["charged"])

    def test_pressing_again_while_compressing_after_the_shock(self):
        # the shock button is never to be touched while a hand is on the patient, even after the shock
        sim = Sim().play(demo_plan())
        self.assertEqual(sim.last.score, 1.0)
        sim.drive({"right": (right_tcp(0.040), None)}, 0.5)
        sim.drive({"left": ((L.AED_X, L.AED_Y, task.PRESS_Z), None)}, 1.0).hold(0.5)
        self.assertEqual(sim.last.score, 0.0)
        self.assertIn("pressed_while_not_clear", sim.last.details["breaches"])

    def test_pad_on_the_compression_point(self):
        sim = Sim()
        sim.put("pad_a", (L.CPR_X + 0.01, L.CPR_Y, L.MARK_TOP + HALF[2]))
        self.assertEqual(sim.last.score, 0.0)
        self.assertIn("pad_on_compression_point", sim.last.details["breaches"])

    def test_pad_off_the_table(self):
        sim = Sim()
        sim.slide("pad_b", (L.PAD_B_X, 0.62, L.TOP + HALF[2]), s=1.0)
        sim.put("pad_b", (L.PAD_B_X, 0.62, L.TOP - 0.3))
        self.assertEqual(sim.last.score, 0.0)
        self.assertIn("off_table", sim.last.details["breaches"])

    def test_a_breach_is_never_undone(self):
        p = demo_plan()
        sim = Sim()
        sim.put("pad_a", (L.CPR_X, L.CPR_Y, L.MARK_TOP + HALF[2]))
        sim.put("pad_a", START["pad_a"] + [0, 0, 0.002])
        sim.put("pad_a", START["pad_a"])
        sim.play(p)  # then the whole demo, done right
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(sim.last.details["breaches"], ["pad_on_compression_point"])


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Exploits(unittest.TestCase):
    def test_compressing_with_an_open_hand(self):
        p = demo_plan()
        p["right"] = [W(w.duration, w.pos, OPEN, w.label, w.yaw) for w in p["right"]]
        r = Sim().play(p).last
        self.assertEqual(r.details["compressions"], 0)
        self.assertEqual(r.score, 0.25)  # pads on, a clean shock, but no compressions

    def test_shallow_or_off_target_dips_are_not_compressions(self):
        t = Tracker(HZ)
        a, b = pad(START["pad_a"]), pad(START["pad_b"])
        left = arm(READY["left"] - [0, 0, TIPS], 1.0)
        for top, bottom, dx in ((0.029, 0.005, 0.0), (0.040, 0.016, 0.0), (0.040, 0.008, 0.021)):
            for _ in range(3):
                for h in (top, bottom):
                    for _ in range(10):
                        t.update(a, b, {"left": left, "right": fist(tips_over_mark(h, dx=dx))})
        self.assertEqual(len(t.compressions), 0)
        for _ in range(3):  # the real thing, just inside both limits
            for h in (0.0301, 0.0149):
                for _ in range(10):
                    t.update(a, b, {"left": left, "right": fist(tips_over_mark(h))})
        self.assertEqual(len(t.compressions), 3)

    def test_holding_a_pad_in_the_compressing_hand_is_not_compressing(self):
        t = Tracker(HZ)
        a, b = pad(START["pad_a"]), pad(START["pad_b"])
        left = arm(READY["left"] - [0, 0, TIPS], 1.0)
        for h in (0.040, 0.008) * 3:
            for _ in range(10):
                t.update(a, b, {"left": left, "right": arm(tips_over_mark(h), 0.040 / L.FULL_GAP)})
        self.assertEqual(len(t.compressions), 0)

    def test_slow_compressions_do_not_count_as_in_progress(self):
        # dips every 2 s: compressions are counted, but each pad arrives more than 1.5 s after the last one
        p = demo_plan()
        i = idx(p["right"], "hands on the chest")
        j = idx(p["right"], "stand clear for the analysis")
        dur = span(p["right"][i + 1:j])
        slow = []
        for _ in range(int(dur // 2.0)):
            slow += [W(0.5, right_tcp(0.008), FIST, "down"), W(1.5, right_tcp(0.040), FIST, "up, resting")]
        slow.append(W(dur - span(slow), right_tcp(0.040), FIST, "rest"))
        p["right"] = p["right"][:i + 1] + slow + p["right"][j:]
        r = Sim().play(p).last
        self.assertGreater(r.details["compressions"], 10)
        self.assertFalse(all(r.details["arrived_in_cpr"].values()))
        self.assertLessEqual(r.score, 0.25)

    def test_pads_hovering_over_their_sites(self):
        t = Tracker(HZ)
        right = fist(tips_over_mark(0.040))
        left = arm((L.AED_X, L.AED_Y, L.BUTTON_TOP + 0.04), 0.016 / L.FULL_GAP)
        for _ in range(100):
            t.update(pad(ON_STERNAL + [0, 0, 0.006]), pad(ON_APEX + [0, 0, 0.006]), {"left": left, "right": right})
        self.assertFalse(t.ever_both_on)
        self.assertEqual(t.result(pad(ON_STERNAL + [0, 0, 0.006]), pad(ON_APEX + [0, 0, 0.006])).score, 0.0)

    def test_pads_put_on_before_compressions_start(self):
        # the right arm starts only after both pads are on: the pads do not count as put on during compressions,
        # even after a clean shock (0.25)
        p = demo_plan()
        late = span(p["left"][:idx(p["left"], "over the shock button")])
        p["right"] = [W(late, READY["right"], OPEN, "late")] + p["right"]
        i = idx(p["left"], "wait, clear")
        p["left"] = with_duration(p["left"], "wait, clear", p["left"][i].duration + late)
        r = Sim().play(p, tail_s=20).last
        self.assertFalse(any(r.details["arrived_in_cpr"].values()))
        self.assertEqual(r.details["breaches"], [])
        self.assertIsNotNone(r.details["shock_step"])
        self.assertFalse(r.details["shock_qualified"])
        self.assertEqual(r.details["charges"], 2)  # the first charge, before compressions, was cancelled by them
        self.assertEqual(r.score, 0.25)

    def test_a_pad_put_on_early_and_put_on_again_during_compressions_counts(self):
        sim = Sim()
        sim.put("pad_a", ON_STERNAL)  # pushed onto its site before compressions
        sim.put("pad_b", ON_APEX)
        self.assertEqual(sim.t.arrived_in_cpr, {"pad_a": False, "pad_b": False})
        sim.drive({"right": (right_tcp(0.040), FIST)}, 2.0).strokes(3)
        sim.drive({"right": (right_tcp(0.008), None)}, 0.5)
        sim.put("pad_a", ON_STERNAL + [0, 0, 0.01])  # lifted off ...
        sim.drive({"right": (right_tcp(0.040), None)}, 0.5)
        sim.put("pad_a", ON_STERNAL)  # ... and put on again while compressions go on
        sim.put("pad_b", ON_APEX + [0, 0, 0.01])
        sim.put("pad_b", ON_APEX)
        self.assertEqual(sim.t.arrived_in_cpr, {"pad_a": True, "pad_b": True})

    def test_pushing_a_pad_onto_its_site_during_compressions_counts(self):
        sim = Sim()
        sim.drive({"right": (right_tcp(0.040), FIST)}, 2.0).strokes(2)
        sim.pos["pad_a"] = ON_STERNAL + [0, 0.06, 0]  # lying on the chest, off its site
        sim.frame()
        # push it 60 mm along the chest while the compressions go on
        for k in range(1, 21):
            sim.pos["pad_a"] = ON_STERNAL + [0, 0.06 * (1 - k / 20), 0]
            sim.drive({"right": (right_tcp(0.008 if k % 2 else 0.040), None)}, 0.25)
        self.assertEqual(sim.t.site_now["pad_a"], "sternal")
        self.assertTrue(sim.t.arrived_in_cpr["pad_a"])

    def test_swapped_pads_are_fine(self):
        p = demo_plan()
        # pad B to the sternal site, pad A to the apex site: same plan, the pads' starting places swapped
        sim = Sim()
        sim.pos["pad_a"], sim.pos["pad_b"] = START["pad_b"].copy(), START["pad_a"].copy()
        sim.prev = {n: sim.pos[n].copy() for n in PADS}
        sim.t.reset()
        sim.play(p)
        self.assertEqual(sim.last.score, 1.0)
        self.assertEqual(sim.last.details["sites"], {"pad_a": "apex", "pad_b": "sternal"})


# ---------------------------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        t = Tracker(HZ)
        a, b = pad(ON_STERNAL), pad(ON_APEX)
        for _ in range(70):
            r = t.update(a, b, None)
        self.assertEqual(t.arms_unavailable, 70)
        self.assertTrue(t.charged)  # unreadable arms count as clear
        self.assertEqual(r.score, 0.25)
        self.assertEqual(t.breaches, {})

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.array([np.nan, 0, 0]), np.zeros(3), 1.0),
               "right": rigstate.Arm(np.zeros(3), np.zeros(3), 1.0)}
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), bad)
        self.assertEqual(t.arms_unavailable, 1)
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), {"left": bad["right"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_feed({s: {"tcp": READY[s], "tips": READY[s] - [0, 0, TIPS], "opening": 0.5}
                               for s in READY})
        arms = rigstate.arms()
        self.assertAlmostEqual(arms["left"].opening, 0.5)
        self.assertTrue(S.arms_valid(arms))


# ---------------------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t, a, b = stand_clear_tracker()
        before = (t.compressions[:], dict(t.arrived_in_cpr), t.analysis_run, t.breaches.copy())
        bads = [pad((np.nan, 0, 0)), SimpleNamespace(pos=a.pos, quat_xyzw=np.zeros(4), lin_vel=a.lin_vel,
                                                     ang_vel=a.ang_vel, low=a.low, high=a.high),
                pad((np.inf, 0, 0)), SimpleNamespace(pos=a.pos)]
        for bad in bads:
            r = t.update(bad, b, parked_arms())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.bad_steps, len(bads))
        self.assertEqual((t.compressions, t.arrived_in_cpr, t.analysis_run, t.breaches), before)

    def test_good_states_after_a_blow_up_are_scored_as_before(self):
        t, a, b = stand_clear_tracker()
        r0 = t.result(a, b, parked_arms()).score
        t.update(pad((np.nan, 0, 0)), b, parked_arms())
        self.assertEqual(t.result(a, b, parked_arms()).score, r0)

    def test_states_anywhere_do_not_crash(self):
        t = Tracker(HZ)
        rng = np.random.default_rng(3)
        for _ in range(300):
            c = rng.uniform([-0.2, -0.7, L.TOP - 0.8], [0.7, 0.7, L.TOP + 0.4], size=(2, 3))
            q = rng.normal(size=(2, 4))
            arms = {s: arm(rng.uniform([0, -0.4, L.TOP], [0.5, 0.4, L.TOP + 0.3]), rng.uniform(0, 1))
                    for s in READY}
            r = t.update(pad(c[0], q[0]), pad(c[1], q[1]), arms)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


# ---------------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def dirty(self):
        t, a, b = stand_clear_tracker()
        t.update(pad((L.CPR_X, L.CPR_Y, L.MARK_TOP + HALF[2])), b, parked_arms())
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.dirty()
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), parked_arms())
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.compressions, [])

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.dirty()
        t.update(pad(START["pad_a"] + [0.03, -0.02, 0]), pad(START["pad_b"] + [-0.02, 0.03, 0]), parked_arms())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.dirty()
        a0 = np.array(t.prev[0][0])
        for k in range(1, 41):
            c = a0 + (START["pad_a"] - a0) * k / 40
            t.update(pad(c), pad(START["pad_b"]), parked_arms())
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), parked_arms())
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.dirty()
        t.update(pad(START["pad_a"] + [0, 0, 0.2]), pad(START["pad_b"]), parked_arms())
        self.assertTrue(t.breach)

    def test_a_pad_dropped_onto_its_start_is_not_a_teleport(self):
        t = Tracker(HZ)
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), parked_arms())
        t.update(pad(START["pad_a"] + [0, 0, 0.30]), pad(START["pad_b"]), parked_arms())  # (carried up)
        t._flag("off_table")
        fall = START["pad_a"] + [0, 0, 0.30]
        v = np.zeros(3)
        for _ in range(12):  # free fall onto its starting place
            v = v + [0, 0, -L.GRAVITY * DT]
            fall = np.maximum(fall + v * DT, START["pad_a"])
            t.update(pad(fall, vel=v), pad(START["pad_b"]), parked_arms())
        t.update(pad(START["pad_a"]), pad(START["pad_b"]), parked_arms())
        self.assertTrue(t.breach)

    @needs_stub
    def test_rigstate_reset_hook_resets_the_tracker(self):
        tr = task._TRACKER
        tr.reset()
        tr._flag("off_table")
        rigstate.offline_reset()
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook, trace and demo."""

    def objects(self, **over):
        out = {"pad_a": pad(START["pad_a"]), "pad_b": pad(START["pad_b"])}
        out.update(over)
        return out

    def test_objects(self):
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in ("manikin_torso", "manikin_head", "compression_mark", "sternal_site", "apex_site", "aed",
                  "shock_button", "pad_a", "pad_b"):
            self.assertIn(n, names)
        loose = {o.name for o in task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, set(PADS))
        self.assertEqual(task.SCORE_OVER, "end")
        self.assertEqual(task.SCENE, "station")
        by = {o.name: o for o in task.OBJECTS}
        for n in PADS:  # the scorer's starting places match the objects'
            self.assertTrue(np.allclose(by[n].pos, START[n]), n)
        # the scorer's manikin boxes match the objects'
        for name, (lo, hi) in (("manikin_torso", S.TORSO_BOX), ("manikin_head", S.HEAD_BOX)):
            o = by[name]
            self.assertTrue(np.allclose(np.array(o.pos) - np.array(o.size) / 2, lo), name)
            self.assertTrue(np.allclose(np.array(o.pos) + np.array(o.size) / 2, hi), name)
        self.assertAlmostEqual(by["shock_button"].pos[2] + by["shock_button"].height / 2, L.BUTTON_TOP)
        self.assertAlmostEqual(by["compression_mark"].pos[2] + L.MARK_THICK / 2, L.MARK_TOP)
        self.assertGreater(task.SUCCESS_HOLD_S, 1.5 + 1.0)

    def test_hooks_and_reset_hook(self):
        tr = task._TRACKER
        rigstate.offline_reset()
        tr.prev = None
        rigstate.offline_feed({s: {"tcp": READY[s], "tips": READY[s] - [0, 0, TIPS], "opening": 1.0} for s in READY})
        on_cpr = pad((L.CPR_X, L.CPR_Y, L.MARK_TOP + HALF[2]))
        self.assertFalse(task.success(self.objects(pad_a=on_cpr)))
        self.assertIn("pad_on_compression_point", tr.breaches)
        self.assertEqual(task.score(self.objects(pad_a=on_cpr)), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        self.assertEqual(task.score(self.objects()), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.assertFalse(task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_timing(self):
        plan = demo_plan()
        self.assertEqual(set(plan), {"left", "right"})
        self.assertEqual(set(demo_plan(DEMO_IDLE_ARM="left")), {"right"})
        self.assertEqual(set(demo_plan(DEMO_IDLE_ARM="right")), {"left"})
        self.assertEqual(demo_plan(DEMO_HOLD_ONLY="1"), {})
        left, right = plan["left"], plan["right"]
        # the right hand compresses until the left hand is closed over the button, clear of the patient
        lift = span(right[:idx(right, "stand clear for the analysis")])
        self.assertGreaterEqual(lift, task.LEFT_READY_S - 1e-9)
        self.assertLess(lift - task.LEFT_READY_S, 1.0 + 1e-9)
        # the press begins 1 s after the plan's charge; the right hand starts back 0.6 s after the plan's shock
        press_wp = idx(left, "press the shock button")
        self.assertAlmostEqual(span(left[:press_wp]) + task._PRESS_IN, task.CHARGED_S + task.PRESS_MARGIN_S)
        back = span(right[:idx(right, "hands back on the chest")])
        self.assertAlmostEqual(back, task.SHOCK_S + task.RESUME_MARGIN_S)
        self.assertLess((max(span(left), span(right)) + 3.0) * HZ, L.RUN_LIMIT_STEPS)


if __name__ == "__main__":
    unittest.main()
