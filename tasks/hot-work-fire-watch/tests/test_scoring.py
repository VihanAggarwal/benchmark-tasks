"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v

Most trials are played through ``Sim``, a small kinematic stand-in for the station: the arms follow waypoint plans
(task.demo()'s, or hand-written ones for the exploits), a gripper that closes on a part carries it rigidly (sagging
by a set amount under its weight) and a part let go drops onto whatever is under it. It cannot push, tip or throw
things; those trials move the objects directly with ``Sim.slide``.
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

HZ = 20.0
DT = 1.0 / HZ
OPEN = -0.04695
TIPS = 0.013
READY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
SAG_PER_KG = 0.008 / 0.3  # about 8 mm down for 0.3 kg at 0.3 m reach
LATERAL_PER_KG = 0.0035 / 0.3  # and 3.5 mm toward the arm's own base
Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


SPEC = {  # name: (kind, half extents (x, y, z) upright, mass, grip width along x, starting centre)
    "torch": ("cyl", (L.TORCH_RADIUS, L.TORCH_RADIUS, L.TORCH_HEIGHT / 2), L.TORCH_MASS, 2 * L.TORCH_RADIUS,
              (L.TORCH_X, L.TORCH_Y, L.TOP + L.TORCH_HEIGHT / 2)),
    "extinguisher": ("cyl", (L.EXT_RADIUS, L.EXT_RADIUS, L.EXT_HEIGHT / 2), L.EXT_MASS, 2 * L.EXT_RADIUS,
                     (L.EXT_X, L.EXT_Y, L.TOP + L.EXT_HEIGHT / 2)),
    "carton": ("box", tuple(v / 2 for v in L.CARTON_SIZE), L.CARTON_MASS, L.CARTON_SIZE[0],
               (L.CARTON_X, L.CARTON_Y, L.MAT_TOP + L.CARTON_SIZE[2] / 2)),
    "rag": ("box", tuple(v / 2 for v in L.RAG_SIZE), L.RAG_MASS, L.RAG_SIZE[0],
            (L.RAG_X, L.RAG_Y, L.MAT_TOP + L.RAG_SIZE[2] / 2)),
}
NAMES = ("torch", "extinguisher", "carton", "rag")


def state(name, center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like object: centre, orientation, velocities, world bounding box."""
    kind, half, *_ = SPEC[name]
    c, q = np.array(center, float), np.array(quat, float)
    R = S.rotation(q)
    if kind == "box":
        ext = np.abs(R) @ np.array(half)
    else:  # an upright cylinder's bounding box, turned
        r, h = half[0], half[2]
        u = R[:, 2]
        ext = np.abs(u) * h + r * np.sqrt(np.clip(1 - u * u, 0, 1))
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c - ext, high=c + ext)


def start_state(name):
    return state(name, SPEC[name][4])


def support_z(x, y):
    """The surface under a point: the plate, the mat or the bare table (the cups' floors are the table)."""
    if abs(x - L.MARK_X) <= L.PLATE_HALF and abs(y - L.MARK_Y) <= L.PLATE_HALF:
        return L.PLATE_TOP
    if math.hypot(x - L.MARK_X, y - L.MARK_Y) <= L.RADIUS:
        return L.MAT_TOP
    return L.TOP


class Sim:
    """The kinematic stand-in: two arms on waypoint plans, four loose parts, the tracker fed every control step."""

    def __init__(self, sag=1.0, lateral=1.0, arms=True, tracker=None):
        self.sag, self.lateral, self.read_arms = sag, lateral, arms
        self.t = tracker or Tracker(HZ)
        self.pos = {n: np.array(SPEC[n][4], float) for n in NAMES}
        self.quat = {n: Q0.copy() for n in NAMES}
        self.prev = {n: self.pos[n].copy() for n in NAMES}
        self.cmd = {s: READY[s].copy() for s in READY}
        self.fin = {s: OPEN for s in READY}
        self.tcp = {s: READY[s].copy() for s in READY}
        self.att = {"left": None, "right": None}  # side -> (name, offset from the commanded TCP)
        self.results = []
        self.frame()

    # -- one control step ------------------------------------------------------------------------------------
    def _drop(self, n):
        half_z = SPEC[n][1][2]
        self.pos[n][2] = support_z(*self.pos[n][:2]) + half_z

    def _arm(self, side):
        gap = -2 * self.fin[side]
        held = self.att[side]
        if held is not None:
            n, off = held
            if gap > SPEC[n][3] + 0.002:  # opened: let go, it drops onto what is under it
                self.att[side] = None
                self.tcp[side] = self.cmd[side].copy()
                self._drop(n)
                return
            kind, half, mass, width, _ = SPEC[n]
            sag = self.sag * SAG_PER_KG * mass
            lat = self.lateral * LATERAL_PER_KG * mass * (1 if side == "left" else -1)
            want = self.cmd[side] + off  # where the part would be with no sag
            floor = support_z(*want[:2]) + half[2]
            lift = want[2] - floor  # how far the commanded grip holds it above its support
            f = min(max(lift / sag, 0.0), 1.0) if sag > 0 else 1.0  # share of the weight the arm carries
            shift = np.array([0.0, lat * f, -sag * f])
            p = want + shift
            p[2] = max(p[2], floor)
            self.pos[n] = p
            self.tcp[side] = p - off
            return
        self.tcp[side] = self.cmd[side].copy()
        tips = self.tcp[side] - [0, 0, TIPS]
        for n in NAMES:
            if any(a is not None and a[0] == n for a in self.att.values()):
                continue
            kind, half, mass, width, _ = SPEC[n]
            c = self.pos[n]
            inside_z = c[2] - half[2] <= tips[2] <= c[2] + half[2]
            if kind == "cyl":
                near = math.hypot(*(tips[:2] - c[:2])) <= 0.010
            else:
                near = abs(tips[0] - c[0]) <= half[0] - 0.005 and abs(tips[1] - c[1]) <= half[1] - 0.005
            if near and inside_z and gap <= width:
                self.att[side] = (n, c - self.cmd[side])
                return

    def snapshot(self):
        out = {}
        for side in READY:
            held = self.att[side]
            gap = -2 * self.fin[side]
            if held is not None:
                gap = max(gap, SPEC[held[0]][3])
            out[side] = {"tcp": self.tcp[side], "tips": self.tcp[side] - np.array([0, 0, TIPS]),
                         "opening": float(np.clip(gap / L.FULL_GAP, 0, 1))}
        return out

    def states(self):
        return [state(n, self.pos[n], self.quat[n], (self.pos[n] - self.prev[n]) / DT) for n in NAMES]

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
        self.prev = {n: self.pos[n].copy() for n in NAMES}
        return r

    # -- driving it ------------------------------------------------------------------------------------------
    def play(self, plan, tail_s=4.0):
        """Run both arms' waypoint plans on one clock, then hold everything still for ``tail_s``."""
        tracks = {s: self._track(plan.get(s, []), s) for s in READY}
        n = max(len(v) for v in tracks.values()) + int(round(tail_s * HZ))
        for k in range(n):
            for s, seq in tracks.items():
                if seq:
                    p, f = seq[min(k, len(seq) - 1)]
                    self.cmd[s], self.fin[s] = np.array(p), f
            self.frame()
        return self

    def _track(self, wps, side):
        out, pos, fin = [], self.cmd[side].copy(), self.fin[side]
        for w in wps:
            m = max(1, int(round(w.duration * HZ)))
            for i in range(1, m + 1):
                f = i / m
                out.append((pos + (np.array(w.pos) - pos) * f, fin + (w.finger - fin) * f))
            pos, fin = np.array(w.pos, float), w.finger
        return out

    def hold(self, s):
        for _ in range(int(round(s * HZ))):
            self.frame()
        return self

    def slide(self, name, to, s=1.0, quat=None):
        """Move a part directly (pushed, swept, knocked), linearly over ``s`` seconds."""
        a, b = self.pos[name].copy(), np.array(to, float)
        n = max(1, int(round(s * HZ)))
        for i in range(1, n + 1):
            self.pos[name] = a + (b - a) * i / n
            if quat is not None and i == n:
                self.quat[name] = np.array(quat, float)
            self.frame()
        return self

    @property
    def last(self):
        return self.results[-1]

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


def replace(wps, label_part, new):
    """A copy of a plan with the first waypoint whose label contains ``label_part`` replaced by ``new`` (a list)."""
    i = next(i for i, w in enumerate(wps) if label_part in w.label)
    return wps[:i] + new + wps[i + 1:]


def cut_at(wps, label_part):
    """The waypoints before the first one whose label contains ``label_part``."""
    i = next(i for i, w in enumerate(wps) if label_part in w.label)
    return wps[:i]


def span(wps):
    return sum(w.duration for w in wps)


needs_stub = unittest.skipIf(STUB is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")


# ---------------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_tilt_and_lower_end(self):
        self.assertAlmostEqual(S.tilt_deg(about("x", 30)), 30.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(about("y", 180)), 180.0, places=4)
        t = start_state("torch")
        self.assertTrue(np.allclose(S.tip(t), (L.TORCH_X, L.TORCH_Y, L.TOP)))
        flipped = state("torch", SPEC["torch"][4], about("x", 180))
        self.assertTrue(np.allclose(S.tip(flipped), (L.TORCH_X, L.TORCH_Y, L.TOP)))  # either end, the lower one

    def test_hot_zone(self):
        def torch_tip_at(x, y, h):
            return state("torch", (x, y, L.PLATE_TOP + h + L.TORCH_HEIGHT / 2))

        self.assertTrue(S.is_hot(torch_tip_at(L.MARK_X, L.MARK_Y, 0.012)))
        self.assertTrue(S.on_mark(torch_tip_at(L.MARK_X, L.MARK_Y, 0.012)))
        self.assertTrue(S.is_hot(torch_tip_at(L.MARK_X, L.MARK_Y, 0.024)))
        self.assertFalse(S.is_hot(torch_tip_at(L.MARK_X, L.MARK_Y, 0.026)))
        self.assertTrue(S.on_mark(torch_tip_at(L.MARK_X + 0.014, L.MARK_Y, 0.01)))
        self.assertFalse(S.on_mark(torch_tip_at(L.MARK_X + 0.016, L.MARK_Y, 0.01)))
        self.assertTrue(S.is_hot(torch_tip_at(L.MARK_X + 0.016, L.MARK_Y, 0.01)))  # hot work, not on the mark
        self.assertTrue(S.is_hot(torch_tip_at(L.MARK_X + 0.059, L.MARK_Y - 0.059, 0.01)))  # over the plate's corner
        self.assertFalse(S.is_hot(torch_tip_at(L.MARK_X + 0.061, L.MARK_Y, 0.0)))  # beside the plate
        self.assertFalse(S.is_hot(start_state("torch")))
        # a torch leaning 30 degrees: its tip is the centre of its lower end, not its bounding box
        lean = about("y", 30)
        u = S.rotation(lean)[:, 2]
        c = np.array([L.MARK_X, L.MARK_Y, L.PLATE_TOP + 0.010]) + u * L.TORCH_HEIGHT / 2
        self.assertTrue(S.on_mark(state("torch", c, lean)))

    def test_outline_gap_and_clear(self):
        self.assertAlmostEqual(S.outline_gap(start_state("carton")), math.hypot(0.04, 0.08), places=6)
        self.assertAlmostEqual(S.outline_gap(start_state("rag")), 0.105, places=6)
        out = state("carton", (task.CARTON_TO[0], task.CARTON_TO[1], L.TOP + 0.02)) if task else None
        if out is not None:
            self.assertGreater(S.outline_gap(out), L.RADIUS + 0.04)
        edge = state("rag", (L.MARK_X, L.MARK_Y - L.RADIUS - 0.025 - 0.001, L.TOP + 0.015))
        self.assertTrue(S.area_clear([edge]))
        edge = state("rag", (L.MARK_X, L.MARK_Y - L.RADIUS - 0.025 + 0.001, L.TOP + 0.015))
        self.assertFalse(S.area_clear([edge]))
        # turned 45 degrees, its outline (bounding box) grows: it must be further out
        turned = state("rag", (L.MARK_X, L.MARK_Y - L.RADIUS - 0.030, L.TOP + 0.015), about("z", 45))
        self.assertFalse(S.area_clear([turned]))

    def test_held_by(self):
        e = state("extinguisher", (0.15, 0.11, L.TOP + 0.10))
        tips = np.array([0.15, 0.11, e.high[2] - 0.029])
        arm = rigstate.Arm(tips + [0, 0, TIPS], tips, 0.044 / L.FULL_GAP)
        self.assertTrue(S.held_by(arm, e, L.EXT_RADIUS, L.EXT_HEIGHT))
        self.assertTrue(S.held_by(rigstate.Arm(arm.tcp, tips, 0.049 / L.FULL_GAP), e, L.EXT_RADIUS, L.EXT_HEIGHT))
        self.assertFalse(S.held_by(rigstate.Arm(arm.tcp, tips, 0.051 / L.FULL_GAP), e, L.EXT_RADIUS, L.EXT_HEIGHT))
        self.assertFalse(S.held_by(rigstate.Arm(arm.tcp, tips, 1.0), e, L.EXT_RADIUS, L.EXT_HEIGHT))  # open
        off = tips + [0.013, 0, 0]
        self.assertFalse(S.held_by(rigstate.Arm(off + [0, 0, TIPS], off, 0.3), e, L.EXT_RADIUS, L.EXT_HEIGHT))
        above = tips + [0, 0, 0.040]  # closed just above its top: not holding it
        self.assertFalse(S.held_by(rigstate.Arm(above + [0, 0, TIPS], above, 0.3), e, L.EXT_RADIUS, L.EXT_HEIGHT))
        # tilted with the gripper: the grip point stays on its axis
        q = about("x", 15)
        u = S.rotation(q)[:, 2]
        c = tips - u * (L.EXT_HEIGHT / 2 - 0.029)
        self.assertTrue(S.held_by(arm, state("extinguisher", c, q), L.EXT_RADIUS, L.EXT_HEIGHT))

    def test_holders(self):
        self.assertTrue(S.torch_parked(start_state("torch")))
        self.assertTrue(S.ext_parked(start_state("extinguisher")))
        lifted = state("torch", (L.TORCH_X, L.TORCH_Y, L.TOP + 0.020 + L.TORCH_HEIGHT / 2))
        self.assertTrue(S.torch_in_holder(lifted))  # bottom 20 mm up: still inside the cup (25 mm rim)
        self.assertFalse(S.torch_parked(lifted))
        out = state("torch", (L.TORCH_X, L.TORCH_Y, L.TOP + 0.030 + L.TORCH_HEIGHT / 2))
        self.assertFalse(S.torch_in_holder(out))
        beside = state("torch", (L.TORCH_X + 0.040, L.TORCH_Y, L.TOP + L.TORCH_HEIGHT / 2))
        self.assertFalse(S.torch_in_holder(beside))
        q = about("x", 15)  # leaning in the cup against a wall: in the holder, not parked
        u = S.rotation(q)[:, 2]
        lean = state("torch", np.array([L.TORCH_X, L.TORCH_Y + 0.004, L.TOP + 0.004]) + u * L.TORCH_HEIGHT / 2, q)
        self.assertTrue(S.torch_in_holder(lean))
        self.assertFalse(S.torch_parked(lean))
        self.assertFalse(S.ext_in_holder(state("extinguisher", (L.TORCH_X, L.TORCH_Y, L.TOP + 0.055))))

    def test_watch_spot(self):
        def ext_at(x, y, q=Q0):
            return state("extinguisher", (x, y, L.TOP + 0.10), q)

        self.assertTrue(S.watch_spot(ext_at(0.15, 0.11)))
        self.assertTrue(S.watch_spot(ext_at(L.MARK_X, L.MARK_Y + 0.209)))
        self.assertFalse(S.watch_spot(ext_at(L.MARK_X, L.MARK_Y + 0.211)))
        self.assertFalse(S.watch_spot(ext_at(L.EXT_X, L.EXT_Y)))  # in its holder: off the mat
        self.assertTrue(S.watch_spot(ext_at(0.15, 0.11, about("y", 19))))
        self.assertFalse(S.watch_spot(ext_at(0.15, 0.11, about("y", 21))))
        self.assertFalse(S.watch_spot(ext_at(0.15, 0.11, about("y", 180))))  # upside down


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Demo(unittest.TestCase):
    """The scripted demo through the kinematic stand-in, with and without the payload sag."""

    def test_demo_scores_one(self):
        for sag, lateral in ((1.0, 1.0), (0.0, 0.0), (2.0, 2.0), (1.5, -1.0)):
            sim = Sim(sag, lateral).play(demo_plan())
            r = sim.last
            self.assertEqual(r.score, 1.0, f"sag x{sag}, lateral x{lateral}: {r.details}")
            self.assertTrue(r.success)
            self.assertEqual(r.details["breaches"], [])
            self.assertGreaterEqual(r.details["weld_s"], L.WELD_S + 0.8)
            self.assertGreaterEqual(r.details["post_watch_s"], L.POST_WATCH_S + 1.0)

    def test_demo_timing(self):
        plan = demo_plan()
        sim = Sim().play(plan)
        left = plan["left"]
        release = span(left[:next(i for i, w in enumerate(left) if "let go of the extinguisher" in w.label)])
        end = max(span(w) for w in plan.values())
        first = sim.first_success()
        self.assertIsNotNone(first)
        self.assertTrue(all(r.success for r in sim.results[first:]))  # and it holds from then on
        # success starts once the left gripper has begun to open, and the trial (success held 3 s) ends only
        # after its retreat: it cannot end while a gripper still holds the extinguisher
        self.assertGreater(first * DT, release)
        self.assertGreater(first * DT + task.SUCCESS_HOLD_S, end)
        self.assertLess(first * DT + task.SUCCESS_HOLD_S + 3.0, L.RUN_LIMIT_STEPS * DT)
        # hot work happens only while the area is clear and the watch posted
        hot = [i for i, r in enumerate(sim.results) if r.details["hot"]]
        self.assertTrue(hot)
        self.assertGreaterEqual(sim.results[hot[0]].details["ready_best_s"], 1.0)
        self.assertAlmostEqual(hot[-1] * DT, task.LAST_HOT_S, delta=0.3)
        print(f"\n[demo] {end:.1f} s plan; hot work {hot[0] * DT:.1f}-{hot[-1] * DT:.1f} s; "
              f"success from step {first} ({first * DT:.2f} s), trial ends about {first * DT + 3:.1f} s",
              file=sys.stderr)

    def test_without_arm_readings_the_demo_still_scores_one(self):
        r = Sim(arms=False).play(demo_plan()).last
        self.assertEqual(r.score, 1.0)
        self.assertGreater(r.details["arms_unavailable"], 1000)

    def test_one_arm_alone(self):
        # The right arm alone clears the rag and welds without a fire watch, with the carton on the mat: 0.
        sim = Sim().play(demo_plan(DEMO_IDLE_ARM="left"))
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(set(sim.last.details["breaches"]),
                         {"hot_not_clear", "hot_unwatched", "combustible_returned"})
        # The left arm alone clears the carton and posts the watch, but the rag stays on the mat: never ready, 0.
        sim = Sim().play(demo_plan(DEMO_IDLE_ARM="right"))
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(sim.last.details["breaches"], [])
        self.assertEqual(sim.last.details["ready_best_s"], 0.0)
        self.assertTrue(any(r.details["watch_spot"] for r in sim.results))  # the watch was over the mat

    def test_doing_nothing_scores_zero(self):
        sim = Sim().play(demo_plan(DEMO_HOLD_ONLY="1"), tail_s=30)
        self.assertEqual(sim.last.score, 0.0)
        self.assertFalse(sim.last.success)
        self.assertEqual(sim.last.details["breaches"], [])


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Tiers(unittest.TestCase):
    """Each tier, from the demo cut short or changed."""

    def setUp(self):
        self.plan = demo_plan()

    def test_extinguisher_still_held_scores_075(self):
        left = cut_at(self.plan["left"], "carry the extinguisher back")
        r = Sim().play({"left": left, "right": self.plan["right"]}).last
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["ext_held_by"], "left")
        self.assertFalse(r.success)

    def test_torch_still_held_scores_075(self):
        right = cut_at(self.plan["right"], "carry the torch back")
        r = Sim().play({"left": self.plan["left"], "right": right}).last
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["torch_held_by"], "right")

    def test_torch_parked_but_still_gripped_scores_075(self):
        right = cut_at(self.plan["right"], "let go of the torch")
        r = Sim().play({"left": self.plan["left"], "right": right}).last
        self.assertTrue(r.details["torch_parked"])
        self.assertEqual(r.score, 0.75)

    def test_extinguisher_leaning_in_its_cup_scores_075(self):
        sim = Sim().play(self.plan)
        self.assertEqual(sim.last.score, 1.0)
        q = about("y", 14)
        u = S.rotation(q)[:, 2]
        sim.slide("extinguisher", np.array([L.EXT_X + 0.005, L.EXT_Y, L.TOP + 0.003]) + u * L.EXT_HEIGHT / 2,
                  0.5, quat=q)
        sim.hold(4)
        self.assertEqual(sim.last.score, 0.75)
        self.assertEqual(sim.last.details["breaches"], [])

    def test_watch_left_early_scores_half(self):
        left = self.plan["left"]
        i = next(i for i, w in enumerate(left) if "keep the watch" in w.label)
        short = left[:i] + [W(left[i].duration - 4.0, left[i].pos, left[i].finger, left[i].label, left[i].yaw)] + \
            left[i + 1:]
        r = Sim().play({"left": short, "right": self.plan["right"]}).last
        self.assertEqual(r.score, 0.5)
        self.assertLess(r.details["post_watch_s"], L.POST_WATCH_S)
        self.assertGreater(r.details["post_watch_s"], 6.0)

    def test_trial_ending_before_the_watch_is_done_scores_half(self):
        sim = Sim()
        sim.play({"left": cut_at(self.plan["left"], "keep the watch"),
                  "right": cut_at(self.plan["right"], "straight up off the plate")}, tail_s=0.0)
        r = sim.last
        self.assertGreaterEqual(r.details["weld_s"], L.WELD_S)
        self.assertEqual(r.score, 0.5)

    def test_short_weld_scores_quarter(self):
        right = self.plan["right"]
        i = next(i for i, w in enumerate(right) if w.label == "weld the mark")
        short = right[:i] + [W(2.5, right[i].pos, right[i].finger, right[i].label, right[i].yaw)] + right[i + 1:]
        r = Sim().play({"left": self.plan["left"], "right": short}).last
        self.assertLess(r.details["weld_s"], L.WELD_S)
        self.assertGreater(r.details["weld_s"], 3.0)
        self.assertEqual(r.score, 0.25)

    def test_ready_without_welding_scores_quarter(self):
        right = cut_at(self.plan["right"], "carry it high over the mark")
        r = Sim().play({"left": self.plan["left"], "right": right}).last
        self.assertEqual(r.score, 0.25)
        self.assertGreaterEqual(r.details["ready_best_s"], L.READY_S)
        self.assertEqual(r.details["weld_s"], 0.0)

    def test_ready_for_under_2_s_scores_zero(self):
        # the watch carried across the mat and straight back: posted with the area clear for well under 2 s
        left = cut_at(self.plan["left"], "carry it over the mat")
        ex = (L.EXT_X, L.EXT_Y, task.EXT_CARRY)
        left += [W(1.2, (0.22, 0.19, task.EXT_CARRY), task.EXT_GRIP, "dip over the mat edge", task.YAW_L),
                 W(1.2, ex, task.EXT_GRIP, "back over the holder", task.YAW_L)]
        left += task._park_tool(L.EXT_X, L.EXT_Y, task.EXT_HANG, task.EXT_CARRY, task.EXT_GRIP, task.YAW_L,
                                "extinguisher", 0.5)
        right = cut_at(self.plan["right"], "hold the torch up")
        r = Sim().play({"left": left, "right": right}).last
        self.assertGreater(r.details["ready_best_s"], 0.0)
        self.assertLess(r.details["ready_best_s"], L.READY_S)
        self.assertEqual(r.score, 0.0)

    def test_success_needs_stillness(self):
        sim = Sim().play(self.plan, tail_s=0.0)
        sim.slide("rag", sim.pos["rag"] + [0.03, 0, 0], 3.0)  # creeping along at 1 cm/s, off the mat
        self.assertEqual(sim.last.score, 1.0)
        self.assertFalse(any(r.success for r in sim.results[-55:]))

    def test_score_is_idempotent(self):
        sim = Sim().play(self.plan)
        a = sim.t.result(*sim.states(), rigstate.arms())
        b = sim.t.result(*sim.states(), rigstate.arms())
        self.assertEqual((a.score, a.success), (b.score, b.success))
        self.assertEqual(a.score, 1.0)


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Breaches(unittest.TestCase):
    """Every zero condition."""

    def setUp(self):
        self.plan = demo_plan()

    def test_weld_before_the_area_is_clear(self):
        # the right arm takes the torch first and welds while the rag is still on the mat (the watch is posted)
        right = self.plan["right"]
        rag = right[:12]
        rest = right[12:]
        r = Sim().play({"left": self.plan["left"], "right": rest + rag}).last
        self.assertIn("hot_not_clear", r.details["breaches"])
        self.assertEqual(r.score, 0.0)

    def test_weld_before_the_watch_is_posted(self):
        # the left arm lifts the extinguisher out but keeps it over its holder, off the mat; the right arm welds
        left = cut_at(self.plan["left"], "carry it over the mat")
        r = Sim().play({"left": left, "right": self.plan["right"]}).last
        self.assertEqual(r.details["breaches"], ["hot_unwatched"])
        self.assertEqual(r.score, 0.0)

    def test_extinguisher_tilted_is_not_a_watch(self):
        sim = Sim()
        left = cut_at(self.plan["left"], "keep the watch")
        sim.play({"left": left, "right": cut_at(self.plan["right"], "carry it high over the mark")}, tail_s=0)
        self.assertTrue(sim.last.details["watch_spot"])
        sim.quat["extinguisher"] = about("y", 25)  # tipped in the grip
        sim.play({"left": [], "right": cut_at(self.plan["right"], "straight up off the plate")[
            len(cut_at(self.plan["right"], "carry it high over the mark")):]}, tail_s=0)
        self.assertEqual(sim.last.details["breaches"], ["hot_unwatched"])  # still held, but not upright
        self.assertEqual(sim.last.details["ext_held_by"], "left")

    def test_combustible_brought_back_during_the_post_watch(self):
        sim2 = Sim()
        right = cut_at(self.plan["right"], "carry the torch back")
        sim2.play({"left": cut_at(self.plan["left"], "carry the extinguisher back"), "right": right}, tail_s=0)
        self.assertEqual(sim2.last.details["breaches"], [])
        sim2.slide("rag", (0.30, -0.15, L.MAT_TOP + 0.015), 2.0)  # pushed back onto the mat ...
        sim2.slide("rag", (0.40, -0.30, L.TOP + 0.015), 2.0)  # ... and off it again
        self.assertEqual(sim2.last.details["breaches"], ["combustible_returned"])
        self.assertEqual(sim2.last.score, 0.0)

    def test_torch_set_down_outside_its_holder(self):
        # the torch put down on the table beside the plate, then picked up again
        right = self.plan["right"]
        i = next(i for i, w in enumerate(right) if "carry it high over the mark" in w.label)
        g, y = task.TORCH_GRIP, task.YAW_R
        spot = (0.26, -0.15)
        down_z = L.TOP + 0.002 + task.TORCH_HANG
        detour = [W(1.5, (*spot, task.TORCH_CARRY), g, "to the table", y), W(1.5, (*spot, down_z), g, "down", y),
                  W(0.5, (*spot, down_z), g, "settle", y), W(1.5, (*spot, down_z), OPEN, "let go", y),
                  W(1.5, (*spot, down_z), g, "grip again", y), W(1.0, (*spot, task.TORCH_CARRY), g, "up", y)]
        r = Sim().play({"left": self.plan["left"], "right": right[:i] + detour + right[i:]}).last
        self.assertEqual(r.details["breaches"], ["torch_set_down"])
        self.assertEqual(r.score, 0.0)

    def test_a_fumble_shorter_than_a_quarter_second_is_not_a_set_down(self):
        t = Tracker(HZ)
        out = state("torch", (0.26, -0.15, L.TOP + L.TORCH_HEIGHT / 2))
        sts = [out, start_state("extinguisher"), start_state("carton"), start_state("rag")]
        rigstate.offline_feed({s: {"tcp": READY[s], "tips": READY[s] - [0, 0, TIPS], "opening": 1.0} for s in READY})
        for _ in range(5):
            t.update(*sts, rigstate.arms())
        self.assertFalse(t.breach)
        t.update(*sts, rigstate.arms())
        self.assertEqual(sorted(t.breaches), ["torch_set_down"])

    def test_extinguisher_set_down_near_the_mark_and_the_same_arm_welds(self):
        # The one-gripper shortcut: the left arm sets the extinguisher down on the mat, then takes the torch.
        left = cut_at(self.plan["left"], "keep the watch")
        g, y = task.EXT_GRIP, task.YAW_L
        down = L.MAT_TOP + 0.002 + task.EXT_HANG
        left += [W(1.5, (*task.POST, down), g, "down to the mat", y), W(0.5, (*task.POST, down), g, "settle", y),
                 W(1.5, (*task.POST, down), OPEN, "let go", y), W(1.0, (*task.POST, task.EXT_CARRY), OPEN, "up", y)]
        right = cut_at(self.plan["right"], "hold the torch up")
        r = Sim().play({"left": left, "right": right}).last
        self.assertIn("extinguisher_set_down", r.details["breaches"])
        self.assertEqual(r.score, 0.0)

    def test_extinguisher_off_the_mat_is_not_a_watch(self):
        left = self.plan["left"]
        far = (0.20, 0.23)  # 24 cm from the mark, beside the mat
        left = [W(w.duration, (*far, w.pos[2]), w.finger, w.label, w.yaw)
                if w.label.startswith(("carry it over the mat", "settle: the fire watch", "keep the watch")) else w
                for w in left]
        r = Sim().play({"left": left, "right": self.plan["right"]}).last
        self.assertEqual(r.details["breaches"], ["hot_unwatched"])

    def test_pushing_a_combustible_off_the_table(self):
        sim = Sim()
        sim.slide("rag", (0.40, -0.56, L.MAT_TOP + 0.015), 2.0)  # swept to the table's edge
        sim.slide("rag", (0.40, -0.60, L.TOP - 0.3), 0.3)  # and over it, onto the floor
        sim.play(self.plan)
        self.assertIn("off_table", sim.last.details["breaches"])
        self.assertEqual(sim.last.score, 0.0)

    def test_pushing_a_combustible_off_the_mat_is_allowed(self):
        sim = Sim()
        sim.slide("rag", (0.40, -0.26, L.TOP + 0.015), 3.0)  # slid along the table, off the mat
        right = self.plan["right"]
        lead = span(right[:12]) - 3.0  # the right arm's rag work, less the 3 s the push took
        sim.play({"left": self.plan["left"], "right": [W(lead, (0.40, -0.26, task.TORCH_CARRY), OPEN, "", task.YAW_R)]
                  + right[12:]})
        self.assertEqual(sim.last.score, 1.0)

    def test_torch_passing_low_over_the_plate_before_clearing(self):
        # The torch, carried low (tip 20 mm over the plate) across it on its way, before the carton is moved
        sim = Sim()
        right = self.plan["right"]
        pick = right[12:19]  # the torch pick
        low = L.PLATE_TOP + 0.020 + task.TORCH_HANG
        g, y = task.TORCH_GRIP, task.YAW_R
        sweep = [W(1.5, (L.MARK_X - 0.05, -0.10, low), g, "low", y), W(2.0, (L.MARK_X - 0.05, 0.05, low), g, "x", y),
                 W(1.5, (L.TORCH_X, L.TORCH_Y, task.TORCH_CARRY), g, "back", y)]
        sim.play({"right": right[:12] + pick + sweep})
        self.assertIn("hot_not_clear", sim.last.details["breaches"])
        self.assertEqual(sim.last.score, 0.0)

    def test_torch_carried_high_over_the_plate_is_not_hot_work(self):
        sim = Sim()
        right = self.plan["right"]
        g, y = task.TORCH_GRIP, task.YAW_R
        over = [W(2.0, (L.MARK_X, L.MARK_Y, task.TORCH_CARRY), g, "over", y),
                W(2.0, (L.TORCH_X, L.TORCH_Y, task.TORCH_CARRY), g, "back", y)]
        sim.play({"right": right[12:19] + over})  # carton and rag still on the mat
        self.assertEqual(sim.last.details["breaches"], [])
        self.assertEqual(sim.last.details["hot_s"], 0.0)

    def test_hot_work_beside_the_mark_counts_as_hot_work(self):
        # welding 40 mm off the mark, over the plate, while the watch has not come over the mat: hot work, a breach
        right = self.plan["right"]
        i = next(i for i, w in enumerate(right) if "carry it high over the mark" in w.label)
        mx, g, y = L.MARK_X + 0.040, task.TORCH_GRIP, task.YAW_R
        beside = right[:i] + [W(2.0, (mx, 0.0, task.TORCH_CARRY), g, "", y), W(1.0, (mx, 0.0, task.WELD_TCP), g, "", y),
                              W(6.0, (mx, 0.0, task.WELD_TCP), g, "", y), W(1.0, (mx, 0.0, task.TORCH_CARRY), g, "", y)]
        sim = Sim().play({"left": cut_at(self.plan["left"], "carry it over the mat"), "right": beside})
        r = sim.last
        self.assertEqual(r.details["breaches"], ["hot_unwatched"])
        self.assertEqual(r.details["weld_s"], 0.0)  # not on the mark: no weld time
        self.assertGreater(r.details["hot_s"], 5.0)
        self.assertEqual(r.score, 0.0)

    def test_a_breach_is_never_undone(self):
        sim = Sim()
        sim.slide("carton", (0.40, 0.56, L.TOP - 0.2), 1.0)
        sim.slide("carton", (0.40, 0.26, L.TOP + 0.02), 2.0)
        sim.play(self.plan)
        self.assertEqual(sim.last.details["breaches"], ["off_table"])
        self.assertEqual(sim.last.score, 0.0)


# ---------------------------------------------------------------------------------------------------------------
@needs_stub
class Exploits(unittest.TestCase):
    def setUp(self):
        self.plan = demo_plan()

    def test_many_tiny_bursts_count_but_every_burst_is_checked(self):
        left, right = self.plan["left"], self.plan["right"]
        i = next(i for i, w in enumerate(right) if w.label == "weld the mark")
        p, g, y = right[i].pos, task.TORCH_GRIP, task.YAW_R
        up = (p[0], p[1], task.WELD_ENTRY)
        bursts = []
        for _ in range(10):  # ten 0.6 s bursts, lifting the tip out of 'hot' between them
            bursts += [W(0.6, p, g, "burst", y), W(0.3, up, g, "lift", y), W(0.3, p, g, "dip", y)]
        extra = span(bursts) - right[i].duration
        j = next(j for j, w in enumerate(left) if "keep the watch" in w.label)
        longer = left[:j] + [W(left[j].duration + extra, left[j].pos, left[j].finger, left[j].label, left[j].yaw)]
        r = Sim().play({"left": longer + left[j + 1:], "right": right[:i] + bursts + right[i + 1:]}).last
        self.assertGreaterEqual(r.details["weld_s"], L.WELD_S)
        self.assertEqual(r.details["breaches"], [])
        self.assertEqual(r.score, 1.0)  # bursts are fine while every one is watched ...
        # ... but one more touch of the plate after the watch has gone is a breach
        wait = span(left) - span(right) + 1.0
        again = [W(wait, right[-1].pos, OPEN, "wait for the watch to leave", y)] + right[12:19] + [
            W(2.0, (L.MARK_X, L.MARK_Y, task.TORCH_CARRY), g, "back over the mark", y),
            W(1.5, p, g, "one more burst", y), W(1.0, (L.MARK_X, L.MARK_Y, task.TORCH_CARRY), g, "up", y)]
        r = Sim().play({"left": left, "right": right + again}).last
        self.assertEqual(r.details["breaches"], ["hot_unwatched"])
        self.assertEqual(r.score, 0.0)

    def test_a_hot_step_restarts_the_post_watch(self):
        # weld, lift, then touch the plate again just before the extinguisher leaves: the 10 s start over
        left, right = self.plan["left"], self.plan["right"]
        i = next(i for i, w in enumerate(right) if "carry the torch back" in w.label)
        g, y = task.TORCH_GRIP, task.YAW_R
        late = span(left[:next(j for j, w in enumerate(left) if "keep the watch" in w.label) + 1]) - span(right[:i]) - 3
        dab = [W(late, (L.MARK_X, L.MARK_Y, task.TORCH_CARRY), g, "wait", y),
               W(1.0, (L.MARK_X, L.MARK_Y, task.WELD_TCP), g, "dab", y),
               W(1.0, (L.MARK_X, L.MARK_Y, task.TORCH_CARRY), g, "up", y)]
        r = Sim().play({"left": left, "right": right[:i] + dab + right[i:]}).last
        self.assertEqual(r.details["breaches"], [])
        self.assertLess(r.details["post_watch_s"], L.POST_WATCH_S)
        self.assertEqual(r.score, 0.5)

    def test_one_gripper_cannot_hold_both(self):
        # the left gripper closed on the extinguisher, the torch hanging next to it: not held, set down
        t = Tracker(HZ)
        e = state("extinguisher", (0.15, 0.11, L.TOP + 0.10))
        torch = state("torch", (0.15, 0.15, L.TOP + 0.08))
        tips = np.array([0.15, 0.11, e.high[2] - 0.029])
        feed = {"left": {"tcp": tips + [0, 0, TIPS], "tips": tips, "opening": 0.044 / L.FULL_GAP},
                "right": {"tcp": READY["right"], "tips": READY["right"] - [0, 0, TIPS], "opening": 1.0}}
        rigstate.offline_feed(feed)
        for _ in range(10):
            r = t.update(torch, e, start_state("carton"), start_state("rag"), rigstate.arms())
        self.assertEqual(r.details["ext_held_by"], "left")
        self.assertIsNone(r.details["torch_held_by"])
        self.assertIn("torch_set_down", r.details["breaches"])

    def test_swapped_roles_are_fine(self):
        # held by either gripper: what matters is that the extinguisher is held while the torch works
        t = Tracker(HZ)
        e = state("extinguisher", (0.15, -0.11, L.TOP + 0.10))
        torch = state("torch", (L.MARK_X, L.MARK_Y, L.PLATE_TOP + 0.012 + L.TORCH_HEIGHT / 2))
        et = np.array([0.15, -0.11, e.high[2] - 0.029])
        tt = np.array([L.MARK_X, L.MARK_Y, torch.high[2] - 0.029])
        feed = {"right": {"tcp": et + [0, 0, TIPS], "tips": et, "opening": 0.044 / L.FULL_GAP},
                "left": {"tcp": tt + [0, 0, TIPS], "tips": tt, "opening": 0.028 / L.FULL_GAP}}
        rigstate.offline_feed(feed)
        far = [state("carton", (0.40, 0.26, L.TOP + 0.02)), state("rag", (0.40, -0.26, L.TOP + 0.015))]
        for _ in range(120):
            r = t.update(torch, e, *far, rigstate.arms())
        self.assertEqual(r.details["breaches"], [])
        self.assertGreaterEqual(r.details["weld_s"], L.WELD_S)
        self.assertEqual(r.score, 0.5)

    def test_extinguisher_standing_on_the_mat_with_a_gripper_beside_it_is_not_held(self):
        t = Tracker(HZ)
        e = state("extinguisher", (0.15, 0.11, L.MAT_TOP + L.EXT_HEIGHT / 2))
        tips = np.array([0.15, 0.11, e.high[2] - 0.029])
        rigstate.offline_feed({"left": {"tcp": tips + [0, 0, TIPS], "tips": tips, "opening": 1.0},
                               "right": {"tcp": READY["right"], "tips": READY["right"] - [0, 0, TIPS], "opening": 1.0}})
        for _ in range(10):
            r = t.update(start_state("torch"), e, start_state("carton"), start_state("rag"), rigstate.arms())
        self.assertIn("extinguisher_set_down", r.details["breaches"])


# ---------------------------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        t = Tracker(HZ)
        out = state("torch", (0.26, -0.15, L.TOP + L.TORCH_HEIGHT / 2))
        for _ in range(20):
            r = t.update(out, start_state("extinguisher"), start_state("carton"), start_state("rag"), None)
        self.assertEqual(t.arms_unavailable, 20)
        self.assertFalse(t.breach)  # 'held' taken as met
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["grippers_let_go"])

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY["right"], READY["right"] - [0, 0, TIPS], float("nan"))}
        sts = [start_state(n) for n in NAMES]
        t.update(*sts, bad)
        t.update(*sts, {"left": bad["left"]})
        t.update(*sts, {"left": object(), "right": object()})
        self.assertEqual(t.arms_unavailable, 3)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_feed({"left": {"tcp": (0.1, 0.2, 0.9), "tips": (0.1, 0.2, 0.887), "opening": 0.25},
                               "right": {"tcp": (0.1, -0.2, 0.9), "tips": (0.1, -0.2, 0.887), "opening": 0.75}})
        r = Tracker(HZ).update(*[start_state(n) for n in NAMES], rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = start_state("torch")
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=np.zeros(3))
        t = Tracker(HZ)
        sts = [start_state(n) for n in NAMES]
        for _ in range(30):
            t.update(*sts, None)
        t._flag("hot_unwatched")
        t.weld_steps, t.ready_best = 50, 10
        before = (dict(t.breaches), t.weld_steps, t.ready_best, t.post_steps, t.arms_unavailable)
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(S.valid(bad))
            for k in range(4):
                objs = list(sts)
                objs[k] = bad
                r = t.update(*objs, None)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(*objs).score, 0.0)
        self.assertEqual((dict(t.breaches), t.weld_steps, t.ready_best, t.post_steps, t.arms_unavailable), before)
        self.assertEqual(t.bad_steps, 16)

    def test_good_states_after_a_blow_up_are_scored_as_before(self):
        if STUB is None:
            self.skipTest("needs the stub")
        sim = Sim().play(demo_plan(), tail_s=0.0)
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        sim.t.update(nan, *sim.states()[1:], rigstate.arms())
        sim.hold(4)
        self.assertEqual(sim.last.score, 1.0)
        self.assertTrue(sim.last.success)

    def test_states_anywhere_do_not_crash(self):
        odd = [state("torch", (5.0, -3.0, -1.0)), state("extinguisher", (0.3, 0.0, 0.0), about("x", 120)),
               state("carton", (-0.2, 0.6, L.TOP), about("y", 90)), state("rag", (0.27, 0, 2.0), vel=(5, -5, 30))]
        t = Tracker(HZ)
        for k in range(8):
            r = t.update(*(odd[(i + k) % 4] for i in range(4)), None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)


class Reset(unittest.TestCase):
    def breached(self):
        t = Tracker(HZ)
        sts = [start_state(n) for n in NAMES]
        hot = state("torch", (L.MARK_X, L.MARK_Y, L.PLATE_TOP + 0.010 + L.TORCH_HEIGHT / 2))
        # carried there gradually
        a, b = sts[0].pos, hot.pos
        for i in range(1, 41):
            p = a + (b - a) * i / 40
            t.update(state("torch", p, vel=(b - a) / 2.0), *sts[1:], None)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(*[start_state(n) for n in NAMES], None)  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.hot_steps, 0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        sts = [start_state(n) for n in NAMES]
        sts[2] = state("carton", sts[2].pos + [0.02, -0.02, 0])
        t.update(*sts, None)
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        sts = [start_state(n) for n in NAMES]
        a, b = t.prev[0][0], sts[0].pos
        for i in range(1, 61):
            t.update(state("torch", a + (b - a) * i / 60, vel=(b - a) / 3.0), *sts[1:], None)
        for _ in range(5):
            t.update(*sts, None)
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        sts = [start_state(n) for n in NAMES]
        t.update(state("torch", (0.4, 0.0, L.TOP + 0.05)), *sts[1:], None)
        self.assertTrue(t.breach)

    def test_a_dropped_torch_is_not_a_teleport(self):
        t = Tracker(HZ)
        sts = [start_state(n) for n in NAMES]
        t.update(*sts, None)
        t._flag("hot_unwatched")
        p = sts[0].pos + [0, 0, 0.3]
        t.update(state("torch", p), *sts[1:], None)
        z, v = p[2], 0.0
        while z > sts[0].pos[2]:
            v += 9.81 * DT
            z = max(sts[0].pos[2], z - v * DT)
            t.update(state("torch", (p[0], p[1], z), vel=(0, 0, -v if z > sts[0].pos[2] else 0)), *sts[1:], None)
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.breach)
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


@needs_stub
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook, trace and demo."""

    def objects(self, **over):
        out = {n: start_state(n) for n in NAMES}
        out.update(over)
        return out

    def test_objects(self):
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in NAMES + ("hot_work_mat", "work_plate", "weld_mark"):
            self.assertIn(n, names)
        loose = {o.name for o in task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, set(NAMES))
        self.assertEqual(task.SCORE_OVER, "end")
        self.assertEqual(task.SCENE, "station")
        by = {o.name: o for o in task.OBJECTS}
        for n in NAMES:  # the scorer's starting places match the objects'
            self.assertTrue(np.allclose(by[n].pos, SPEC[n][4]), n)
        self.assertGreater(task.SUCCESS_HOLD_S, 1.5 + 1.0)

    def test_hooks_and_reset_hook(self):
        tr = task._TRACKER
        rigstate.offline_reset()
        hot = state("torch", (L.MARK_X, L.MARK_Y, L.PLATE_TOP + 0.010 + L.TORCH_HEIGHT / 2))
        rigstate.offline_feed({s: {"tcp": READY[s], "tips": READY[s] - [0, 0, TIPS], "opening": 1.0} for s in READY})
        tr.prev = None
        self.assertFalse(task.success(self.objects(torch=hot)))
        self.assertIn("hot_unwatched", tr.breaches)
        self.assertEqual(task.score(self.objects(torch=hot)), 0.0)
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
        # the torch comes over the plate only after the watch is posted
        posted = span(left[:next(i for i, w in enumerate(left) if "watch is posted" in w.label) + 1])
        carry = span(right[:next(i for i, w in enumerate(right) if "carry it high over the mark" in w.label)])
        self.assertGreaterEqual(carry, posted + 0.5 - 1e-9)
        # the watch is kept 11.5 s past the plan's last hot instant
        leave = span(left[:next(i for i, w in enumerate(left) if "carry the extinguisher back" in w.label)])
        self.assertAlmostEqual(leave - task.LAST_HOT_S, L.POST_WATCH_S + task.WATCH_MARGIN_S, places=6)
        self.assertLess((max(span(left), span(right)) + 3.0) * HZ, L.RUN_LIMIT_STEPS)


if __name__ == "__main__":
    unittest.main()
