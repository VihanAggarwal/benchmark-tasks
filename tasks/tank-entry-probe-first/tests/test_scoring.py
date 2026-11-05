"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v

Two kinds of trial are fed to the scorer:
  * ``Film``: hand-written frames (the probe's and the part's centres and orientations, both TCPs and openings),
    moved linearly over n control steps; for things a kinematic model cannot do (knocking, dropping, teleports);
  * ``Proxy``: a kinematic stand-in for the simulator that plays waypoint plans (task.demo() or variants of it) at
    20 Hz: a gripper closing to an object's width with its pad tips inside that object picks it up, the object then
    moves with the TCP (lowered by an optional payload sag), and when the fingers open it drops straight down onto
    the table. It has no contact forces, no swinging and no slip.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, band_of, finger_low_z, gripper_in_tank, holds, let_go, part_in_tank, part_in_tray, part_out,
    probe_in_holder, probe_in_tank, probe_low, probe_tip, rotation, tilt_deg, valid,
)
from test_demo_plan import READY, _stub_dir, sample_plan  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = L.TIPS_BELOW_TCP
OPEN = 1.0
PROBE_W, PART_W = 2 * L.PROBE_RADIUS, L.PART_EDGE
PROBE_HELD = PROBE_W / L.FULL_GAP  # rigstate opening with the fingers stopped on the probe (0.32)
PART_HELD = PART_W / L.FULL_GAP
Q0 = np.array([0.0, 0.0, 0.0, 1.0])

P_START = np.array([L.PROBE_X, L.PROBE_Y, L.TOP + L.PROBE_LEN / 2])  # probe centre in its holder
C_START = np.array([L.PART_X, L.PART_Y, L.TOP + L.PART_EDGE / 2])  # part centre on the tank floor
READY_L, READY_R = np.array(READY["left"]), np.array(READY["right"])
GRIP_ABOVE_TIP = L.PROBE_LEN - L.PROBE_GRIP_DOWN + TIPS  # a held probe's lower end is this far below the TCP
SAMPLE = (L.SAMPLE_X, L.SAMPLE_Y)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def probe_state(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like probe: a cylinder's centre, orientation, velocities and world bounding box."""
    c = np.array(center, float)
    u = rotation(quat)[:, 2]
    ext = np.abs(u) * L.PROBE_LEN / 2 + L.PROBE_RADIUS * np.sqrt(np.clip(1 - u * u, 0, 1))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def part_state(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    c, half = np.array(center, float), np.full(3, L.PART_EDGE / 2)
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def probe_centre_for_tip(x, y, h):
    """Centre of an upright probe whose lower end is at (x, y), h above the floor."""
    return np.array([x, y, L.TOP + h + L.PROBE_LEN / 2])


def held_tcp(probe_centre):
    """TCP of the left gripper holding an upright probe as the demo does (pad tips 30 mm below its top)."""
    return np.array(probe_centre) + [0, 0, L.PROBE_LEN / 2 - L.PROBE_GRIP_DOWN + TIPS]


def part_tcp(part_centre):
    return np.array(part_centre) + [0, 0, -L.PART_EDGE / 2 + 0.004 + TIPS]


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("p", "pq", "c", "cq", "L", "Lo", "R", "Ro")
VEC = ("p", "c", "L", "R")


class Film:
    """A trial as a list of frames. ``go`` moves linearly to new values over n control steps (orientations switch
    half-way); velocities come from the frame-to-frame motion."""

    def __init__(self, start=None, **over):
        self.cur = {"p": P_START, "pq": Q0, "c": C_START, "cq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        if start is not None:
            self.cur.update(start)
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in over.items()})
        self.frames = [dict(self.cur)]

    @classmethod
    def after(cls, frames):
        """A film continuing from the last of these frames (which it includes)."""
        f = cls(start=frames[-1])
        f.frames = list(frames)
        return f

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in ("pq", "cq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def with_probe_tip(self, n, h, xy=SAMPLE, **more):
        """Move the held probe (and the left TCP with it) so its tip is h above the floor at xy."""
        pc = probe_centre_for_tip(*xy, h)
        return self.go(n, p=pc, L=held_tcp(pc), Lo=PROBE_HELD, **more)


def play(frames, tracker=None, arms=True):
    """Feed frames to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r, prev = None, None
    for fr in frames:
        if prev is None:
            vp = vc = np.zeros(3)
        else:
            vp, vc = (fr["p"] - prev["p"]) / DT, (fr["c"] - prev["c"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(probe_state(fr["p"], fr["pq"], vp), part_state(fr["c"], fr["cq"], vc), a)
        prev = fr
    return r, t


def run_until_success(frames, tracker=None):
    """Feed frames; return (result, tracker, the step at which success() first held SUCCESS_HOLD_S)."""
    t = tracker or Tracker(HZ)
    need = int(round(3.0 * HZ))
    streak, done_at, prev, r = 0, None, None, None
    for fr in frames:
        vp = np.zeros(3) if prev is None else (fr["p"] - prev["p"]) / DT
        vc = np.zeros(3) if prev is None else (fr["c"] - prev["c"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
        r = t.update(probe_state(fr["p"], fr["pq"], vp), part_state(fr["c"], fr["cq"], vc), a)
        streak = streak + 1 if r.success else 0
        if streak >= need and done_at is None:
            done_at = t.steps
        prev = fr
    return r, t, done_at


# -- the kinematic proxy ----------------------------------------------------------------------------------------
class Proxy:
    """Plays a waypoint plan at 20 Hz and returns frames. ``sag``: how far each held object pulls its arm down (m)."""

    WIDTH = {"probe": PROBE_W, "part": PART_W}
    REST = {"probe": L.TOP + L.PROBE_LEN / 2, "part": L.TOP + L.PART_EDGE / 2}

    def __init__(self, sag=None):
        self.sag = {"probe": 0.0, "part": 0.0, **(sag or {})}

    @staticmethod
    def _bbox(name, centre):
        s = probe_state(centre) if name == "probe" else part_state(centre)
        return s.low, s.high

    def frames(self, plan, hold_s=6.0):
        s = sample_plan(plan, HZ, hold_s)
        pos = {"probe": P_START.copy(), "part": C_START.copy()}
        held = {"left": None, "right": None}
        offset = {}
        out = []
        for i in range(len(s["t"])):
            fr = {"pq": Q0, "cq": Q0}
            for arm, key in (("left", "L"), ("right", "R")):
                p = s[arm]["pos"][i].copy()
                g = -2.0 * s[arm]["finger"][i]
                k = held[arm]
                if k is not None and g > self.WIDTH[k] + 0.002:  # the fingers open: it drops onto the table
                    held[arm] = None
                    pos[k][2] = self.REST[k]
                    k = None
                if k is None:
                    tips = p - [0, 0, TIPS]
                    for name in ("probe", "part"):
                        if name in held.values() or g > self.WIDTH[name]:
                            continue
                        lo, hi = self._bbox(name, pos[name])
                        if np.all(tips >= lo - 0.003) and np.all(tips <= hi + 0.003):
                            held[arm], k = name, name
                            offset[name] = pos[name] - p
                            break
                if k is not None:
                    tcp = p - [0, 0, self.sag[k]]
                    obj = tcp + offset[k]
                    if obj[2] < self.REST[k]:  # resting on the table: it holds the hand up
                        obj[2] = self.REST[k]
                        tcp[2] = obj[2] - offset[k][2]
                    pos[k] = obj
                    gap = self.WIDTH[k]
                else:
                    tcp, gap = p, max(g, 0.0)
                fr[key], fr[key + "o"] = tcp, gap / L.FULL_GAP
            fr["p"], fr["c"] = pos["probe"].copy(), pos["part"].copy()
            out.append(fr)
        return out


def _load_task():
    stub = _stub_dir()
    if stub is not None and str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    return importlib.import_module("task")


def plan_with(left=None, right=None):
    return {k: v for k, v in (("left", left), ("right", right)) if v is not None}


# == geometry ====================================================================================================
class Geometry(unittest.TestCase):
    def test_tip_of_an_upright_and_a_tilted_probe(self):
        pc = probe_centre_for_tip(*SAMPLE, 0.015)
        self.assertTrue(np.allclose(probe_tip(probe_state(pc)), [*SAMPLE, L.TOP + 0.015]))
        tilted = probe_state(pc, about("x", 20))
        tip = probe_tip(tilted)
        self.assertAlmostEqual(tip[2], pc[2] - L.PROBE_LEN / 2 * math.cos(math.radians(20)), places=6)
        self.assertAlmostEqual(tilt_deg(tilted.quat_xyzw), 20.0, places=4)
        self.assertTrue(np.allclose(probe_tip(probe_state(pc, about("x", 180))), probe_tip(probe_state(pc))))

    def test_bands_and_their_edges(self):
        for h, want in ((0.065, 0), (0.0549, None), (0.075, 0), (0.0751, None), (0.040, 1), (0.0525, None),
                        (0.015, 2), (0.004, None), (0.0, None), (0.027, None)):
            self.assertEqual(band_of(probe_state(probe_centre_for_tip(*SAMPLE, h))), want, h)
        # over the wall, or outside the tank: no band
        self.assertIsNone(band_of(probe_state(probe_centre_for_tip(L.TANK_X, L.TANK_IN_HY + 0.004, 0.065))))
        self.assertIsNone(band_of(probe_state(probe_centre_for_tip(L.PROBE_X, L.PROBE_Y, 0.015))))

    def test_probe_low_and_in_tank(self):
        self.assertTrue(probe_low(probe_state(probe_centre_for_tip(*SAMPLE, 0.039))))
        self.assertFalse(probe_low(probe_state(probe_centre_for_tip(*SAMPLE, 0.041))))
        self.assertTrue(probe_in_tank(probe_state(probe_centre_for_tip(*SAMPLE, 0.079))))
        self.assertFalse(probe_in_tank(probe_state(probe_centre_for_tip(*SAMPLE, 0.081))))
        self.assertFalse(probe_in_tank(probe_state(P_START)))
        self.assertTrue(probe_in_holder(probe_state(P_START)))
        self.assertFalse(probe_in_holder(probe_state(P_START + [0.013, 0, 0])))
        self.assertFalse(probe_in_holder(probe_state(P_START, about("y", 12))))

    def test_gripper_in_tank(self):
        a = snap(right_tcp=[L.PART_X, L.PART_Y, L.RIM_Z + TIPS - 0.001])
        self.assertTrue(gripper_in_tank(a["right"]))
        a = snap(right_tcp=[L.PART_X, L.PART_Y, L.RIM_Z + TIPS + 0.001])
        self.assertFalse(gripper_in_tank(a["right"]))
        # within 10 mm of the outline counts (fingers straddling the wall), further out does not
        edge = L.TANK_Y - L.TANK_OUT_HY
        self.assertTrue(gripper_in_tank(snap(right_tcp=[L.TANK_X, edge - 0.009, L.TOP + 0.05])["right"]))
        self.assertFalse(gripper_in_tank(snap(right_tcp=[L.TANK_X, edge - 0.011, L.TOP + 0.05])["right"]))
        self.assertFalse(gripper_in_tank(snap(right_tcp=[L.TRAY_X, L.TRAY_Y, L.TOP + 0.03])["right"]))

    def test_a_tilted_gripper_reaches_lower_than_its_tip_point(self):
        tcp = np.array([L.PART_X, L.PART_Y, L.RIM_Z + TIPS + 0.005])  # tip point 5 mm above the rim
        tilt = math.radians(30)
        tips = tcp + TIPS * np.array([0.0, math.sin(tilt), -math.cos(tilt)])
        rigstate.offline_feed({"left": {"tcp": READY_L, "tips": READY_L - [0, 0, TIPS], "opening": 1.0},
                               "right": {"tcp": tcp, "tips": tips, "opening": 1.0}})
        arm = rigstate.arms()["right"]
        self.assertLess(finger_low_z(arm), L.RIM_Z)  # 57 mm x sin 30 = 28 mm lower: its pads are in the tank
        self.assertTrue(gripper_in_tank(arm))

    def test_holds_and_let_go(self):
        pc = probe_centre_for_tip(*SAMPLE, 0.015)
        probe = probe_state(pc)
        a = snap(left_tcp=held_tcp(pc), left_open=PROBE_HELD)
        self.assertTrue(holds(a["left"], probe))
        self.assertFalse(let_go(a["left"], probe))
        a = snap(left_tcp=held_tcp(pc), left_open=0.6)  # fingers 56 mm apart: not holding
        self.assertFalse(holds(a["left"], probe))
        a = snap(left_tcp=held_tcp(pc) + [0.03, 0, 0], left_open=PROBE_HELD)  # closed beside it
        self.assertFalse(holds(a["left"], probe))
        a = snap(left_tcp=held_tcp(pc), left_open=1.0)
        self.assertTrue(let_go(a["left"], probe))
        a = snap(left_tcp=held_tcp(pc) + [0, 0, 0.08], left_open=0.2)  # closed, well above it
        self.assertTrue(let_go(a["left"], probe))

    def test_open_fingers_clear_the_walls_inside_the_tank(self):
        """At yaw +-pi/2 the fingers close along x (yaw 0: along y). Open, the pads span 110 mm across their outer
        faces along x and 27 mm along y; on the part's mark they must clear every wall by at least 40 mm."""
        half_span = L.FULL_GAP / 2 + L.FINGER_THICK  # 55 mm along x
        front = (L.TANK_X + L.TANK_IN_HX) - (L.PART_X + half_span)
        back = (L.PART_X - half_span) - (L.TANK_X - L.TANK_IN_HX)
        end = (L.PART_Y - L.FINGER_HALF_WIDTH) - (L.TANK_Y - L.TANK_IN_HY)
        probe_side = (L.SAMPLE_Y - L.PROBE_RADIUS) - (L.PART_Y + L.FINGER_HALF_WIDTH)
        for gap in (front, back, end):
            self.assertGreaterEqual(gap, 0.040 - 1e-9)
        self.assertGreater(probe_side, 0.15)
        self.assertGreaterEqual((L.TANK_Y + L.TANK_IN_HY) - (L.SAMPLE_Y + L.PROBE_RADIUS), 0.025 - 1e-9)

    def test_part_places(self):
        self.assertTrue(part_in_tank(part_state(C_START)))
        self.assertFalse(part_out(part_state(C_START)))
        held_low = part_state([L.PART_X, L.PART_Y, L.RIM_Z + 0.009 + L.PART_EDGE / 2])
        self.assertFalse(part_in_tank(held_low))
        self.assertFalse(part_out(held_low))  # 9 mm above the rim: not yet out
        self.assertTrue(part_out(part_state([L.PART_X, L.PART_Y, L.RIM_Z + 0.011 + L.PART_EDGE / 2])))
        on_table = part_state([L.TANK_X, L.TANK_Y - L.TANK_OUT_HY - 0.03, L.TOP + L.PART_EDGE / 2])
        self.assertTrue(part_out(on_table))
        self.assertFalse(part_in_tray(on_table))
        self.assertTrue(part_in_tray(part_state([L.TRAY_X, L.TRAY_Y, L.TOP + L.PART_EDGE / 2])))
        self.assertFalse(part_in_tray(part_state([L.TRAY_X, L.TRAY_Y, L.TOP + 0.006 + L.PART_EDGE / 2])))


# == the demo through the kinematic proxy ========================================================================
@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = _load_task()
        for key in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(key, None)

    def demo_frames(self, sag=None, hold_s=6.0, **env):
        try:
            os.environ.update(env)
            plan = self.task.demo()
        finally:
            for k in env:
                os.environ.pop(k, None)
        return Proxy(sag).frames(plan, hold_s)

    def test_demo_scores_one(self):
        for sag in ({}, {"probe": 0.003, "part": 0.006}, {"probe": 0.008, "part": 0.008}):
            r, t, done = run_until_success(self.demo_frames(sag))
            with self.subTest(sag=sag):
                self.assertEqual(r.score, 1.0, r.details)
                self.assertTrue(r.success)
                self.assertIsNotNone(done)
                self.assertEqual(t.tests_completed, 1)
                self.assertLess(t.band_steps[2], t.first_entry_step)
                self.assertLess(t.max_tip_in_entry, L.LOWER_HALF)
                self.assertGreater(t.min_left_clear, 0.030)  # the attendant's pads stay 30 mm or more over the rim
                self.assertLess(done, 1200 - 60)

    def test_demo_timing(self):
        frames = self.demo_frames()
        r, t, done = run_until_success(frames)
        plan_s = max(sum(w.duration for w in wps) for wps in self.task.demo().values())
        # the test completes at least 1.5 s before the right gripper's fingers go below the rim
        self.assertGreaterEqual(t.first_entry_step - t.band_steps[2], int(1.5 * HZ))
        self.assertLess(plan_s + 3.0, 60.0)
        self.assertGreater(done * DT, plan_s - 1.5)  # success holds only after the probe is let go in its holder
        print(f"\n  demo {plan_s:.1f} s ({int(round(plan_s * HZ))} steps); bands done at steps {t.band_steps}; "
              f"right in the tank from step {t.first_entry_step}; success held at step {done}", file=sys.stderr)

    def test_right_arm_alone_scores_zero(self):
        r, t = play(self.demo_frames(DEMO_IDLE_ARM="left"))
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_entered_unprotected", r.details["breaches"])

    def test_left_arm_alone_scores_quarter(self):
        r, t = play(self.demo_frames(DEMO_IDLE_ARM="right"))
        self.assertEqual(r.score, 0.25, r.details)
        self.assertEqual(r.details["breaches"], [])
        self.assertTrue(r.details["probe_in_holder"])

    def test_hold_only_scores_zero(self):
        r, t = play(self.demo_frames(DEMO_HOLD_ONLY="1", hold_s=10.0))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["breaches"], [])


# == exploits and rule variants played through the proxy ===============================================================
@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class Exploits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = _load_task()
        cls.W = cls.task.Waypoint
        for key in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(key, None)

    def left_pick(self):
        return self.task._left_test()[:8]  # up to "carry it over the tank"

    def at_tip(self, d, h, label="", finger=None):
        t = self.task
        return self.W(d, (*SAMPLE, L.TOP + h + t.PROBE_TCP_ABOVE_TIP), t.PROBE_GRIP if finger is None else finger,
                      label, t.YAW_L)

    def right_fetch(self, start_s):
        """The demo's right arm, but starting down into the tank at ``start_s``."""
        r = self.task._right()
        r[1] = self.W(start_s - r[0].duration, r[1].pos, r[1].finger, r[1].label, r[1].yaw)
        return r

    def test_entering_as_the_probe_arrives_scores_zero(self):
        left = self.left_pick() + [self.at_tip(2.5, 0.015, "straight to the bottom"), self.at_tip(20.0, 0.015)]
        arrive = sum(w.duration for w in left[:-1])
        r, t = play(Proxy().frames(plan_with(left, self.right_fetch(arrive - 0.8))))
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_entered_unprotected", r.details["breaches"])

    def test_skipping_the_top_band_scores_zero_with_entry_and_nothing_without(self):
        left = self.left_pick() + [self.at_tip(1.5, 0.040), self.at_tip(3.0, 0.040), self.at_tip(1.5, 0.015),
                                   self.at_tip(15.0, 0.015)]
        r, t = play(Proxy().frames(plan_with(left, self.right_fetch(sum(w.duration for w in left[:-1]) + 0.5))))
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_entered_unprotected", r.details["breaches"])
        r, t = play(Proxy().frames(plan_with(left)))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.tests_completed, 0)

    def test_reverse_order_does_not_count(self):
        left = self.left_pick() + [self.at_tip(2.0, 0.015), self.at_tip(3.0, 0.015), self.at_tip(1.5, 0.040),
                                   self.at_tip(3.0, 0.040), self.at_tip(1.5, 0.065), self.at_tip(3.0, 0.065)]
        r, t = play(Proxy().frames(plan_with(left)))
        self.assertEqual(t.tests_completed, 0)
        self.assertEqual(t.stage, 1)  # only the top dwell, which came last, counted
        self.assertEqual(r.score, 0.0)

    def test_a_short_dwell_does_not_count(self):
        # 1 s at the top, with brisk 1 s moves in and out (25 mm/s): at most 1.8 s inside a 10 mm box
        left = self.left_pick() + [self.at_tip(1.0, 0.065), self.at_tip(1.0, 0.065), self.at_tip(1.0, 0.040),
                                   self.at_tip(3.0, 0.040), self.at_tip(1.0, 0.015), self.at_tip(3.0, 0.015)]
        r, t = play(Proxy().frames(plan_with(left)))
        self.assertEqual(t.stage, 0)  # the top did not count, and the later bands need the top first
        self.assertEqual(r.score, 0.0)

    def test_sweeping_down_without_stopping_does_not_count(self):
        # from 75 mm to 5 mm at 1 cm/s: 2 s in a band covers 20 mm, more than the 10 mm box
        left = self.left_pick() + [self.at_tip(1.5, 0.075), self.at_tip(7.0, 0.005), self.at_tip(3.0, 0.005)]
        r, t = play(Proxy().frames(plan_with(left)))
        self.assertEqual(t.stage, 0)
        self.assertEqual(r.score, 0.0)

    def test_full_test_then_probe_withdrawn_voids_it(self):
        test = self.task._left_test()
        left = test + [self.at_tip(1.5, 0.12, "withdraw above the rim"), self.at_tip(1.0, 0.12),
                       self.at_tip(1.5, 0.015, "back to the bottom"), self.at_tip(15.0, 0.015)]
        back = sum(w.duration for w in left[:-1])
        r, t = play(Proxy().frames(plan_with(left, self.right_fetch(back + 0.5))))
        self.assertEqual(t.tests_completed, 1)
        self.assertEqual(t.tests_voided, 1)
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_entered_unprotected", r.details["breaches"])

    def test_probe_raised_while_the_right_is_inside_scores_zero(self):
        # the demo, but the probe is raised to 45 mm (above half depth) while the right gripper is in the tank
        test = self.task._left_test()
        rise_at = self.task.TEST_DONE_S + 3.5  # the right gripper is inside by then
        left = test + [self.at_tip(rise_at - self.task.TEST_DONE_S, 0.015), self.at_tip(1.0, 0.045),
                       self.at_tip(10.0, 0.045)]
        r, t = play(Proxy().frames(plan_with(left, self.task._right())))
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_entered_unprotected", r.details["breaches"])

    def test_probe_dropped_in_the_tank_and_left_there_scores_zero(self):
        test = self.task._left_test()
        left = test + [self.at_tip(1.5, 0.015, "let go", self.task.OPEN), self.at_tip(1.0, 0.10, "up",
                                                                                      self.task.OPEN)]
        r, t = play(Proxy().frames(plan_with(left, self.task._right())))
        self.assertEqual(r.score, 0.0)
        self.assertIn("probe_let_go_in_tank", r.details["breaches"])

    def test_attendant_lowering_the_probe_with_its_own_fingers_in_the_tank_scores_zero(self):
        # gripping the probe lower down, so the pad tips go 10 mm under the rim when the tip is in the bottom band
        t_ = self.task
        lowgrip = L.TOP + 0.060 + TIPS  # tips 60 mm above the probe's lower end: 90 mm below its top
        left = [self.W(1.5, (L.PROBE_X, L.PROBE_Y, lowgrip + 0.08), t_.OPEN, "over", t_.YAW_L),
                self.W(1.5, (L.PROBE_X, L.PROBE_Y, lowgrip), t_.OPEN, "down", t_.YAW_L),
                self.W(1.5, (L.PROBE_X, L.PROBE_Y, lowgrip), t_.PROBE_GRIP, "grip low", t_.YAW_L),
                self.W(1.5, (L.PROBE_X, L.PROBE_Y, lowgrip + 0.10), t_.PROBE_GRIP, "lift", t_.YAW_L),
                self.W(2.0, (*SAMPLE, lowgrip + 0.10), t_.PROBE_GRIP, "over the tank", t_.YAW_L)]
        for h in (0.065, 0.040, 0.015):
            z = lowgrip + h  # pad tips 60 mm above the tip: 5 mm under the rim at the bottom band's 15 mm
            left += [self.W(1.5, (*SAMPLE, z), t_.PROBE_GRIP, "lower", t_.YAW_L),
                     self.W(3.0, (*SAMPLE, z), t_.PROBE_GRIP, "dwell", t_.YAW_L)]
        r, t = play(Proxy().frames(plan_with(left)))
        self.assertIn("left_in_tank", r.details["breaches"])
        self.assertEqual(r.score, 0.0)

    def test_reaching_only_to_just_above_the_rim_gets_nothing_more(self):
        # the right gripper's pad tips stop 2 mm above the rim, so it never enters: it cannot reach the part (its
        # top is 50 mm below the rim) and the part stays on the floor: the test alone, 0.25
        t_ = self.task
        z = L.RIM_Z + 0.002 + TIPS
        right = [self.W(2.0, (L.PART_X, L.PART_Y, t_.PART_WAIT_Z), t_.OPEN, "over", t_.YAW_R),
                 self.W(t_.TEST_DONE_S, (L.PART_X, L.PART_Y, t_.PART_WAIT_Z), t_.OPEN, "wait", t_.YAW_R),
                 self.W(1.5, (L.PART_X, L.PART_Y, z), t_.OPEN, "tips just above the rim", t_.YAW_R),
                 self.W(1.5, (L.PART_X, L.PART_Y, z), t_.PART_GRIP, "close", t_.YAW_R),
                 self.W(2.0, (L.PART_X, L.PART_Y, t_.PART_CARRY_Z), t_.PART_GRIP, "lift", t_.YAW_R)]
        r, t = play(Proxy().frames(plan_with(t_._left(), right)))
        self.assertFalse(t.entered)
        self.assertTrue(r.details["part_in_tank"])
        self.assertEqual(r.score, 0.25)

    def test_part_dropped_back_into_the_tank_scores_quarter(self):
        t_ = self.task
        right = t_._right()[:8] + [self.W(1.5, right_pos := (L.PART_X, L.PART_Y, t_.PART_CARRY_Z), t_.OPEN,
                                          "let it fall back in", t_.YAW_R)]
        del right_pos
        r, t = play(Proxy().frames(plan_with(t_._left(), right)))
        self.assertEqual(r.details["breaches"], [])
        self.assertTrue(r.details["part_in_tank"])
        self.assertEqual(r.score, 0.25)

    def test_part_left_on_the_table_scores_half(self):
        t_ = self.task
        spot = (0.22, -0.23)  # on the bare table between the tank and the tray
        right = t_._right()[:8] + [
            self.W(2.0, (*spot, t_.PART_CARRY_Z), t_.PART_GRIP, "over the table", t_.YAW_R),
            self.W(1.5, (*spot, L.TOP + 0.002 + 0.017), t_.PART_GRIP, "down", t_.YAW_R),
            self.W(1.5, (*spot, L.TOP + 0.002 + 0.017), t_.OPEN, "let go", t_.YAW_R),
            self.W(1.0, (*spot, t_.PART_CARRY_Z), t_.OPEN, "up", t_.YAW_R)]
        r, t = play(Proxy().frames(plan_with(t_._left(), right)))
        self.assertEqual(r.score, 0.5, r.details)


# == tiers and rules on hand-written films =================================================================================
def tested(f):
    """The left gripper takes the probe and runs the full test (3 s per band); the tip ends in the bottom band."""
    grip = held_tcp(P_START)
    f.go(30, L=grip + [0, 0, 0.06]).go(30, L=grip).go(30, Lo=PROBE_HELD)
    carry_c = probe_centre_for_tip(L.PROBE_X, L.PROBE_Y, 0.122)
    f.go(30, p=carry_c, L=held_tcp(carry_c))
    over_c = probe_centre_for_tip(*SAMPLE, 0.122)
    f.go(40, p=over_c, L=held_tcp(over_c))
    for h in (0.065, 0.040, 0.015):
        f.with_probe_tip(30, h).hold(60)
    return f


def fetch(f):
    """With the probe low, the right gripper goes in, grips the part and lifts it 50 mm above the rim."""
    above = part_tcp(C_START) + [0, 0, 0.14]
    f.go(30, R=above).go(40, R=part_tcp(C_START)).go(30, Ro=PART_HELD)
    lifted = C_START + [0, 0, 0.13]
    f.go(40, c=lifted, R=part_tcp(lifted))
    return f


def to_tray(f, open_after=True):
    over = np.array([L.TRAY_X, L.TRAY_Y, f.cur["c"][2]])
    f.go(40, c=over, R=part_tcp(over))
    down = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + L.PART_EDGE / 2])
    f.go(30, c=down, R=part_tcp(down))
    if open_after:
        f.go(30, Ro=OPEN).go(20, R=part_tcp(down) + [0, 0, 0.08])
    return f


def probe_home(f, open_after=True):
    f.with_probe_tip(30, 0.122)
    over = probe_centre_for_tip(L.PROBE_X, L.PROBE_Y, 0.122)
    f.go(40, p=over, L=held_tcp(over))
    f.go(40, p=P_START, L=held_tcp(P_START))
    if open_after:
        f.go(30, Lo=OPEN).go(20, L=held_tcp(P_START) + [0, 0, 0.07])
    return f


def procedure(settle=70):
    return probe_home(to_tray(fetch(tested(Film())))).hold(settle)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().hold(20).frames)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure().frames)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertEqual(r.details["breaches"], [])

    def test_success_needs_stillness(self):
        f = procedure()
        r, t = play(f.frames)
        self.assertTrue(r.success)
        last = f.frames[-1]
        moving = t.update(probe_state(last["p"]), part_state(last["c"], vel=(0.02, 0, 0)),
                          snap(last["L"], last["Lo"], last["R"], last["Ro"]))
        self.assertEqual(moving.score, 1.0)
        self.assertFalse(moving.success)
        spinning = t.update(probe_state(last["p"], spin=(0, 0, 0.8)), part_state(last["c"]),
                            snap(last["L"], last["Lo"], last["R"], last["Ro"]))
        self.assertFalse(spinning.success)

    def test_left_still_closed_on_the_probe_in_its_holder_scores_075(self):
        r, _ = play(probe_home(to_tray(fetch(tested(Film()))), open_after=False).hold(30).frames)
        self.assertEqual(r.score, 0.75)

    def test_right_still_closed_on_the_part_in_the_tray_scores_075(self):
        r, _ = play(probe_home(to_tray(fetch(tested(Film())), open_after=False)).hold(30).frames)
        self.assertEqual(r.score, 0.75)

    def test_probe_held_out_of_the_tank_scores_075(self):
        f = to_tray(fetch(tested(Film())))
        f.with_probe_tip(30, 0.122).hold(30)
        r, _ = play(f.frames)
        self.assertEqual(r.score, 0.75, r.details)

    def test_probe_standing_on_the_table_outside_its_holder_scores_075(self):
        f = to_tray(fetch(tested(Film())))
        f.with_probe_tip(30, 0.122)
        spot = probe_centre_for_tip(0.13, 0.25, 0.122)
        f.go(40, p=spot, L=held_tcp(spot))
        down = probe_centre_for_tip(0.13, 0.25, 0.0)
        f.go(40, p=down, L=held_tcp(down)).go(30, Lo=OPEN).go(20, L=held_tcp(down) + [0, 0, 0.07]).hold(30)
        r, _ = play(f.frames)
        self.assertEqual(r.details["breaches"], [])
        self.assertFalse(r.details["probe_in_holder"])
        self.assertEqual(r.score, 0.75, r.details)

    def test_part_in_the_tray_but_probe_still_in_the_tank_scores_half(self):
        r, _ = play(to_tray(fetch(tested(Film()))).hold(30).frames)
        self.assertEqual(r.score, 0.5, r.details)

    def test_part_held_up_out_of_the_tank_scores_half(self):
        r, _ = play(fetch(tested(Film())).hold(20).frames)
        self.assertEqual(r.score, 0.5)

    def test_part_held_just_over_the_rim_is_not_out(self):
        f = tested(Film())
        above = part_tcp(C_START) + [0, 0, 0.14]
        f.go(30, R=above).go(40, R=part_tcp(C_START)).go(30, Ro=PART_HELD)
        lifted = np.array([L.PART_X, L.PART_Y, L.RIM_Z + 0.005 + L.PART_EDGE / 2])
        f.go(40, c=lifted, R=part_tcp(lifted)).hold(20)
        r, _ = play(f.frames)
        self.assertEqual(r.score, 0.25)

    def test_test_only_scores_quarter(self):
        f = probe_home(tested(Film())).hold(30)
        r, t = play(f.frames)
        self.assertEqual(t.tests_completed, 1)
        self.assertEqual(r.score, 0.25)

    def test_test_only_probe_still_held_low_scores_quarter(self):
        r, t = play(tested(Film()).hold(30).frames)
        self.assertTrue(t.test_valid)
        self.assertEqual(r.score, 0.25)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f.frames)
        last = f.frames[-1]
        a = snap(last["L"], last["Lo"], last["R"], last["Ro"])
        for _ in range(3):
            self.assertEqual(t.result(probe_state(last["p"]), part_state(last["c"]), a).score, r.score)


class Breaches(unittest.TestCase):
    def assert_zero(self, frames, key):
        r, t = play(frames)
        self.assertEqual(r.score, 0.0, r.details)
        self.assertIn(key, r.details["breaches"])
        return r, t

    def test_right_enters_before_any_test(self):
        f = Film()
        f.go(40, R=part_tcp(C_START) + [0, 0, 0.14]).go(40, R=part_tcp(C_START)).hold(10)
        self.assert_zero(f.frames, "right_entered_unprotected")

    def test_right_enters_with_the_probe_low_but_untested(self):
        f = Film()
        grip = held_tcp(P_START)
        f.go(30, L=grip).go(30, Lo=PROBE_HELD)
        f.with_probe_tip(60, 0.015).hold(5)  # straight down to the bottom band, no dwell
        f.go(30, R=part_tcp(C_START) + [0, 0, 0.14]).go(30, R=part_tcp(C_START))
        self.assert_zero(f.frames, "right_entered_unprotected")

    def test_right_enters_after_the_test_with_the_probe_let_go_or_raised(self):
        f = tested(Film())
        f.with_probe_tip(20, 0.060)  # raised to 60 mm: above half depth
        f.go(30, R=part_tcp(C_START) + [0, 0, 0.14]).go(30, R=part_tcp(C_START))
        self.assert_zero(f.frames, "right_entered_unprotected")

    def test_probe_withdrawn_while_the_right_is_inside(self):
        f = fetch(tested(Film()))
        f.go(20, c=C_START + [0, 0, 0.02], R=part_tcp(C_START + [0, 0, 0.02]))  # still in: lower the part again
        f.go(10, c=C_START, R=part_tcp(C_START))
        f.with_probe_tip(30, 0.122)
        self.assert_zero(f.frames, "right_entered_unprotected")

    def test_left_gripper_into_the_tank(self):
        f = Film()
        f.go(40, L=[L.TANK_X, 0.10, L.TOP + 0.12]).go(30, L=[L.TANK_X, 0.10, L.RIM_Z + TIPS - 0.002])
        self.assert_zero(f.frames, "left_in_tank")

    def test_probe_released_inside_the_tank(self):
        f = tested(Film())
        f.go(30, Lo=OPEN)
        self.assert_zero(f.frames, "probe_let_go_in_tank")

    def test_probe_dropped_outside_the_tank(self):
        f = Film()
        grip = held_tcp(P_START)
        f.go(30, L=grip).go(30, Lo=PROBE_HELD)
        up = probe_centre_for_tip(0.15, 0.25, 0.08)
        f.with_probe_tip(30, 0.08, xy=(0.15, 0.25))
        f.go(10, Lo=OPEN)  # let go 80 mm above the table
        f.go(4, p=probe_centre_for_tip(0.15, 0.25, 0.0))
        del up
        self.assert_zero(f.frames, "probe_dropped")

    def test_probe_set_on_the_tank_rim(self):
        f = Film()
        grip = held_tcp(P_START)
        f.go(30, L=grip).go(30, Lo=PROBE_HELD)
        rim = (L.TANK_X, L.TANK_IN_HY + L.TANK_WALL / 2)  # on top of the left wall
        f.with_probe_tip(40, 0.12, xy=rim).with_probe_tip(30, L.TANK_DEPTH, xy=rim)
        f.go(30, Lo=OPEN)
        self.assert_zero(f.frames, "probe_dropped")

    def test_probe_knocked_over(self):
        f = Film()
        knocked = np.array([L.PROBE_X + 0.05, L.PROBE_Y, L.TOP + L.PROBE_RADIUS])
        f.go(6, p=knocked, pq=about("y", 90)).hold(10)
        self.assert_zero(f.frames, "probe_toppled")

    def test_part_flicked_out_without_the_right_gripper(self):
        f = tested(Film())
        f.go(10, c=[L.PART_X, L.PART_Y, L.RIM_Z + 0.03])  # as if poked up and out by the probe
        f.go(10, c=[L.PART_X, L.TANK_Y - L.TANK_OUT_HY - 0.04, L.TOP + L.PART_EDGE / 2])
        self.assert_zero(f.frames, "part_out_unheld")

    def test_part_off_the_table(self):
        f = Film()
        f.go(10, c=[0.6, -0.3, L.TOP - 0.05])
        r, t = play(f.frames)
        self.assertEqual(r.score, 0.0)
        self.assertIn("off_table", r.details["breaches"])

    def test_a_breach_is_never_undone(self):
        f = Film()
        f.go(40, R=part_tcp(C_START) + [0, 0, 0.14]).go(40, R=part_tcp(C_START)).go(30, R=READY_R)
        g = Film.after(f.frames)
        frames = probe_home(to_tray(fetch(tested(g)))).hold(70).frames
        r, t = play(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["breaches"], ["right_entered_unprotected"])
        self.assertTrue(r.details["part_in_tray"] and r.details["probe_in_holder"])


class PokingAndPushing(unittest.TestCase):
    def test_poking_the_part_with_the_probe_changes_nothing(self):
        f = tested(Film())
        f.with_probe_tip(20, 0.010, xy=(L.PART_X, L.PART_Y + 0.05))
        f.go(20, p=probe_centre_for_tip(L.PART_X, L.PART_Y + 0.03, 0.010),
             L=held_tcp(probe_centre_for_tip(L.PART_X, L.PART_Y + 0.03, 0.010)))
        f.go(20, p=probe_centre_for_tip(L.PART_X, L.PART_Y + 0.01, 0.010),
             L=held_tcp(probe_centre_for_tip(L.PART_X, L.PART_Y + 0.01, 0.010)), c=C_START + [0, -0.02, 0])
        f.hold(10)
        r, t = play(f.frames)
        self.assertEqual(r.details["breaches"], [])
        self.assertEqual(r.score, 0.25)  # the test was done; the part is still in the tank

    def test_doing_nothing_scores_zero(self):
        r, _ = play(Film().hold(1200).frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["breaches"], [])

    def test_one_arm_shortcut_left_only_test_and_put_back(self):
        r, _ = play(probe_home(tested(Film())).hold(60).frames)
        self.assertEqual(r.score, 0.25)


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        f = Film()
        f.go(40, R=part_tcp(C_START))  # would be a breach, but the arms are not readable
        r, t = play(f.frames, arms=False)
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertEqual(r.details["breaches"], [])
        self.assertIsNone(r.details["grippers_let_go"])

    def test_the_test_still_counts_without_the_arms(self):
        r, t = play(tested(Film()).frames, arms=False)
        self.assertEqual(t.tests_completed, 1)
        self.assertEqual(r.score, 0.25)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        f = Film()
        t = Tracker(HZ)
        play(f.hold(10).frames, t, arms=False)
        g = Film().go(40, R=part_tcp(C_START))
        r, t = play(g.frames, t)
        self.assertIn("right_entered_unprotected", r.details["breaches"])

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        rigstate.offline_feed({"left": {"tcp": [np.nan, 0, 0], "tips": [0, 0, 0], "opening": 1.0},
                               "right": {"tcp": READY_R, "tips": READY_R, "opening": 1.0}})
        r = t.update(probe_state(P_START), part_state(C_START), rigstate.arms())
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(right_tcp=part_tcp(C_START))
        self.assertTrue(np.allclose(a["right"].tips, part_tcp(C_START) - [0, 0, TIPS]))
        t = Tracker(HZ)
        r = t.update(probe_state(P_START), part_state(C_START), a)
        self.assertIn("right_entered_unprotected", r.details["breaches"])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = tested(Film())
        r, t = play(f.frames)
        stage, steps = t.stage, t.steps
        bad = probe_state(P_START)
        bad.pos = np.array([np.nan, 0.0, 0.0])
        a = snap(f.cur["L"], f.cur["Lo"], f.cur["R"], f.cur["Ro"])
        r2 = t.update(bad, part_state(C_START), a)
        self.assertEqual(r2.score, 0.0)
        self.assertFalse(r2.details["valid_state"])
        self.assertEqual(t.stage, stage)
        self.assertEqual(t.steps, steps + 1)
        self.assertEqual(t.bad_steps, 1)
        zero_q = part_state(C_START)
        zero_q.quat_xyzw = np.zeros(4)
        self.assertFalse(valid(zero_q))
        self.assertEqual(t.update(probe_state(f.cur["p"]), zero_q, a).score, 0.0)
        r3, _ = play([f.frames[-1]], t)
        self.assertEqual(r3.score, 0.25)  # back to normal, the test still on record
        self.assertEqual(t.breaches, {})

    def test_states_anywhere_do_not_crash(self):
        t = Tracker(HZ)
        rng = np.random.default_rng(3)
        for _ in range(300):
            q = rng.normal(size=4)
            pc = rng.uniform([-0.2, -0.7, 0.5], [0.8, 0.7, 1.2])
            cc = rng.uniform([-0.2, -0.7, 0.5], [0.8, 0.7, 1.2])
            a = snap(rng.uniform([0, -0.5, 0.7], [0.6, 0.5, 1.1]), rng.uniform(0, 1),
                     rng.uniform([0, -0.5, 0.7], [0.6, 0.5, 1.1]), rng.uniform(0, 1))
            r = t.update(probe_state(pc, q / np.linalg.norm(q)), part_state(cc), a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def breached(self):
        f = Film()
        f.go(40, R=part_tcp(C_START))
        r, t = play(f.frames)
        self.assertTrue(t.breach)
        return f, t

    def test_teleport_back_to_start_resets(self):
        f, t = self.breached()
        f.go(20, c=C_START + [0, 0.12, 0])  # the part pushed 12 cm along the tank floor
        play(f.frames[-20:], t)
        jump = [dict(f.frames[-1], p=P_START, c=C_START, R=READY_R)]
        r, t = play(jump, t)
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_gradual_return_does_not_reset(self):
        f, t = self.breached()
        f.go(20, c=C_START + [0, 0.05, 0])
        f.go(20, c=C_START)
        play(f.frames[-40:], t)
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        f, t = self.breached()
        play([dict(f.frames[-1], c=C_START + [0, 0.10, 0])], t)
        self.assertTrue(t.breach)

    def test_reset_clears_the_test(self):
        f = tested(Film())
        r, t = play(f.frames)
        self.assertEqual(t.tests_completed, 1)
        play([dict(f.frames[-1], p=P_START, L=READY_L, Lo=OPEN)], t)
        self.assertEqual(t.tests_completed, 0)
        self.assertEqual(t.stage, 0)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        calls = []
        saved = dict(rigstate._RESET_CALLBACKS)
        try:
            t = Tracker(HZ)
            rigstate.on_reset(lambda: (t.reset(), calls.append(1)))
            self.breached()
            play(Film().go(40, R=part_tcp(C_START)).frames, t)
            self.assertTrue(t.breach)
            rigstate.offline_reset()
            self.assertFalse(t.breach)
            self.assertEqual(calls, [1])
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        cls.task = _load_task()

    def objects(self, p=P_START, c=C_START):
        return {"probe": probe_state(p), "steel_part": part_state(c)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("probe", names)
        self.assertIn("steel_part", names)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")
        loose = {o.name for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"probe", "steel_part"})
        probe = next(o for o in self.task.OBJECTS if o.name == "probe")
        self.assertTrue(np.allclose(probe.pos, P_START))
        self.assertGreater(self.task.SUCCESS_HOLD_S, 1.5 + 1.0)  # longer than the last release and retreat

    def test_walls_match_the_layout(self):
        walls = [o for o in self.task.OBJECTS if o.name.startswith("tank_wall")]
        lo = np.min([np.array(o.pos) - np.array(o.size) / 2 for o in walls], axis=0)
        hi = np.max([np.array(o.pos) + np.array(o.size) / 2 for o in walls], axis=0)
        self.assertTrue(np.allclose(lo, [L.TANK_X - L.TANK_OUT_HX, L.TANK_Y - L.TANK_OUT_HY, L.TOP]))
        self.assertTrue(np.allclose(hi, [L.TANK_X + L.TANK_OUT_HX, L.TANK_Y + L.TANK_OUT_HY, L.RIM_Z]))

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=part_tcp(C_START))  # the entrant in the tank, no test
        self.assertFalse(self.task.success(self.objects()))
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertFalse(tr.breach)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        # the right gripper starts down only after the bottom dwell has ended
        right_down = sum(w.duration for w in plan["right"][:2])
        self.assertIn("into the tank", plan["right"][2].label)
        self.assertGreaterEqual(right_down, self.task.TEST_DONE_S + 0.5 - 1e-9)
        # the probe starts up only after the right gripper's lift out of the tank has ended
        up = next(i for i, w in enumerate(plan["left"]) if "out of the tank" in w.label)
        self.assertGreaterEqual(sum(w.duration for w in plan["left"][:up]), self.task.RIGHT_OUT_S + 0.5 - 1e-9)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
