"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic probe states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v

Two kinds of trial. ``Sim`` is a small kinematic stand-in for the station: the arms follow waypoint plans (task.demo()'s
or hand-written ones), a gripper that closes on a probe carries it rigidly (sagging by a set amount under its weight,
and resting on a post if the command would push the tip into it) and a probe let go drops onto whatever is under it.
It cannot push or tip things. The tier and exploit tests instead feed hand-made sequences of probe poses ('frames')
with matching arm snapshots, which is quicker and states exactly what each test is about.
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
HALF = np.array(L.PROBE_SIZE) / 2
NAMES = L.PROBES
SIDE_OF = {"red_probe": "left", "black_probe": "right"}
SIGN = {"red_probe": 1.0, "black_probe": -1.0}  # the centre is +140 mm (red) or -140 mm (black) in y from the pin
START = {n: np.array([L.REST_X, L.REST_Y[n], L.PROBE_START_Z]) for n in NAMES}
GRIP_OPENING = L.PROBE_W / L.FULL_GAP  # a gripper closed on a probe (blocked at 26 mm)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like probe: centre, orientation, velocities, world bounding box."""
    c, q = np.array(center, float), np.array(quat, float)
    ext = np.abs(S.rotation(q)) @ HALF
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c - ext, high=c + ext)


def at_rest(name, **kw):
    return state(START[name], **kw)


def centre_for(name, pin_xy, dz=0.001):
    """The centre of a level probe whose contact point is over ``pin_xy`` with its underside ``dz`` above the post
    tops."""
    return np.array([pin_xy[0], pin_xy[1] + SIGN[name] * L.PIN_FROM_CENTRE, L.POST_TOP + dz + L.PROBE_HALF_H])


def on(name, post, dz=0.001, dx=0.0, dy=0.0, **kw):
    x, y = L.POSTS[post]
    return state(centre_for(name, (x + dx, y + dy), dz), **kw)


def over(name, post, **kw):
    return on(name, post, dz=0.015, **kw)


def arm(tcp, opening):
    tcp = np.array(tcp, float)
    return {"tcp": tcp, "tips": tcp - np.array([0, 0, TIPS]), "opening": float(opening)}


def gripping(s):
    """A gripper closed on probe state ``s`` at its middle (TCP 4 mm above its centre)."""
    return arm(np.asarray(s.pos) + [0, 0, 0.004], GRIP_OPENING)


def arms_for(red, black, held=("left", "right")):
    """Arm snapshot: the left gripper on the red probe and the right on the black (each if named in ``held``),
    else that gripper fully open 10 cm above its probe."""
    out = {}
    for side, s in (("left", red), ("right", black)):
        out[side] = gripping(s) if side in held else arm(np.asarray(s.pos) + [0, 0, 0.10], 1.0)
    return out


def feed(t, red, black, n=1, arms="auto"):
    """Feed one pose ``n`` times; arms 'auto': held unless on its rest; None: unreadable."""
    r = None
    for _ in range(n):
        if arms == "auto":
            held = tuple(side for side, s, nm in (("left", red, "red_probe"), ("right", black, "black_probe"))
                         if not S.on_rest(s, S.probe(s, nm), L.REST_Y[nm]))
            a = arms_for(red, black, held)
        else:
            a = arms
        if a is None:
            r = t.update(red, black, None)
        else:
            rigstate.offline_feed(a)
            r = t.update(red, black, rigstate.arms())
    return r


def test_steps(t, red_post, black_post, hold=30):
    """Both tips onto (red_post, black_post) from just above them, held ``hold`` steps, then lifted 15 mm."""
    feed(t, over("red_probe", red_post), over("black_probe", black_post), 2)
    r = feed(t, on("red_probe", red_post), on("black_probe", black_post), hold)
    feed(t, over("red_probe", red_post), over("black_probe", black_post), 2)
    return r


def home(t, n=10):
    """Carry both probes back onto their rests from wherever they are, gradually (6 steps of motion, so it is not
    mistaken for the simulator's reset), then hold them there ``n`` steps."""
    a = [p for p, _ in t.prev] if t.prev else [START[m] for m in NAMES]
    for i in range(1, 7):
        sts = [state(a[k] + (START[m] - a[k]) * i / 6, vel=(START[m] - a[k]) / 6 / DT if i < 6 else (0, 0, 0))
               for k, m in enumerate(NAMES)]
        feed(t, *sts)
    return feed(t, at_rest("red_probe"), at_rest("black_probe"), n)


def pick(t):
    """From the rests to the proving unit, hovering (no test)."""
    feed(t, at_rest("red_probe"), at_rest("black_probe"), 5)
    feed(t, over("red_probe", "P+"), over("black_probe", "P-"), 5)


DEMO_ORDER = [("L1", "L2"), ("L1", "L3"), ("L2", "L3"), ("L2", "N"), ("L3", "N"), ("L3", "PE"), ("N", "PE"),
              ("L2", "PE"), ("L1", "PE"), ("L1", "N")]


def procedure(t, pairs=DEMO_ORDER, prove_first=True, reprove=True, put_back=True, hold=30):
    pick(t)
    if prove_first:
        test_steps(t, "P+", "P-", hold)
    for a, b in pairs:
        test_steps(t, a, b, hold)
    if reprove:
        test_steps(t, "P+", "P-", hold)
    if put_back:
        r = home(t, 10)
    else:
        r = feed(t, over("red_probe", "P+"), over("black_probe", "P-"), 10)
    return r


# -- the kinematic stand-in -------------------------------------------------------------------------------------------
def support_floor(name, centre, quat):
    """The lowest the probe's centre can be: on its rest, on a post under its tip, or on the table."""
    floor = L.TOP + L.PROBE_HALF_H
    x, y = centre[:2]
    for ry in L.REST_Y.values():
        if abs(x - L.REST_X) <= L.REST_SIZE[0] / 2 and abs(y - ry) <= L.REST_SIZE[1] / 2:
            floor = L.REST_TOP + L.PROBE_HALF_H
    p = S.probe(state(centre, quat), name)
    d = np.linalg.norm(S.POST_XY - p.pin[:2], axis=1)
    if d.min() <= L.POST_RADIUS:
        floor = max(floor, centre[2] + (L.POST_TOP - p.pin[2]))
    return floor


class Sim:
    """Two arms on waypoint plans, two probes, the tracker fed every control step."""

    def __init__(self, sag=1.0, lateral=1.0, tilt_deg=0.0, dx=0.0, arms=True, tracker=None):
        self.sag, self.lateral, self.tilt, self.dx, self.read_arms = sag, lateral, tilt_deg, dx, arms
        self.t = tracker or Tracker(HZ)
        self.pos = {n: START[n].copy() for n in NAMES}
        self.quat = {n: Q0.copy() for n in NAMES}
        self.prev = {n: self.pos[n].copy() for n in NAMES}
        self.cmd = {s: READY[s].copy() for s in READY}
        self.fin = {s: OPEN for s in READY}
        self.tcp = {s: READY[s].copy() for s in READY}
        self.att = {"left": None, "right": None}  # side -> (name, offset of the centre from the commanded TCP)
        self.results = []
        self.gaps = []

    def _arm(self, side):
        gap = -2 * self.fin[side]
        held = self.att[side]
        if held is not None:
            n, off = held
            if gap > L.PROBE_W + 0.002:  # opened: let go, it drops onto what is under it
                self.att[side] = None
                self.tcp[side] = self.cmd[side].copy()
                self.quat[n] = Q0.copy()
                self.pos[n][2] = support_floor(n, self.pos[n], Q0)
                return
            # the probe pitches by tilt_deg, tip down (the gripper's tilt under load, magnified by the lever arm)
            q = about("x", self.tilt * (1 if n == "red_probe" else -1))
            sag = self.sag * SAG_PER_KG * L.PROBE_MASS
            lat = self.lateral * LATERAL_PER_KG * L.PROBE_MASS * (1 if side == "left" else -1)
            dx = self.dx * (1 if side == "left" else -1)
            want = self.cmd[side] + off + np.array([dx, lat, -sag])
            floor = support_floor(n, want, q)
            want[2] = max(want[2], floor)
            self.pos[n], self.quat[n] = want, q
            self.tcp[side] = want - off
            return
        self.tcp[side] = self.cmd[side].copy()
        tips = self.tcp[side] - [0, 0, TIPS]
        for n in NAMES:
            if any(a is not None and a[0] == n for a in self.att.values()):
                continue
            c = self.pos[n]
            near = abs(tips[0] - c[0]) <= 0.005 and abs(tips[1] - c[1]) <= 0.10
            inside_z = c[2] - L.PROBE_HALF_H <= tips[2] <= c[2] + L.PROBE_HALF_H
            if near and inside_z and gap <= L.PROBE_W:
                self.att[side] = (n, c - self.cmd[side])
                return

    def snapshot(self):
        out = {}
        for side in READY:
            gap = -2 * self.fin[side]
            if self.att[side] is not None:
                gap = max(gap, L.PROBE_W)
            out[side] = arm(self.tcp[side], float(np.clip(gap / L.FULL_GAP, 0, 1)))
        return out

    def frame(self):
        for side in READY:
            self._arm(side)
        sts = [state(self.pos[n], self.quat[n], (self.pos[n] - self.prev[n]) / DT) for n in NAMES]
        if self.read_arms:
            rigstate.offline_feed(self.snapshot())
            arms = rigstate.arms()
        else:
            arms = None
        r = self.t.update(*sts, arms)
        self.gaps.append(S.tip_gap(S.probe(sts[0], NAMES[0]), S.probe(sts[1], NAMES[1])))
        self.results.append(r)
        self.prev = {n: self.pos[n].copy() for n in NAMES}
        return r

    def play(self, plan, tail_s=4.0):
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


def span(wps):
    return sum(w.duration for w in wps)


needs_stub = unittest.skipIf(STUB is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")


# ---------------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_contact_point_and_start(self):
        for n in NAMES:
            p = S.probe(at_rest(n), n)
            want = START[n] + np.array([0, -SIGN[n] * L.PIN_FROM_CENTRE, -L.PROBE_HALF_H])
            self.assertTrue(np.allclose(p.pin, want), n)
            self.assertAlmostEqual(p.level_deg, 0.0, places=6)
            self.assertIsNone(S.on_post(p))
            self.assertEqual(S.posts_touched(p), [])
            self.assertTrue(S.on_rest(at_rest(n), p, L.REST_Y[n]))
        self.assertTrue(Tracker.at_start(at_rest("red_probe"), at_rest("black_probe")))

    def test_on_post_band(self):
        n = "red_probe"
        self.assertEqual(S.on_post(S.probe(on(n, "L3", dz=0.0), n)), "L3")
        self.assertEqual(S.on_post(S.probe(on(n, "L3", dz=0.0039), n)), "L3")
        self.assertIsNone(S.on_post(S.probe(on(n, "L3", dz=0.0045), n)))
        self.assertEqual(S.on_post(S.probe(on(n, "L3", dz=-0.0029), n)), "L3")
        self.assertIsNone(S.on_post(S.probe(on(n, "L3", dz=-0.0035), n)))
        self.assertEqual(S.on_post(S.probe(on(n, "L3", dx=0.0079), n)), "L3")
        self.assertIsNone(S.on_post(S.probe(on(n, "L3", dx=0.0081), n)))
        self.assertEqual(S.on_post(S.probe(on(n, "P+"), n)), "P+")
        self.assertEqual(S.on_post(S.probe(on("black_probe", "P-"), "black_probe")), "P-")

    def test_any_long_face_down_and_level(self):
        n = "black_probe"
        for roll in (0, 90, 180, -90):  # turned about its own long axis (y): still a square bar, face down
            p = S.probe(on(n, "N", quat=about("y", roll)), n)
            self.assertEqual(S.on_post(p), "N", roll)
            self.assertLess(p.level_deg, 1e-6)
        self.assertTrue(S.is_level(S.probe(on(n, "N", quat=about("x", 9.5)), n)))
        self.assertFalse(S.is_level(S.probe(on(n, "N", quat=about("x", 10.5)), n)))
        self.assertFalse(S.is_level(S.probe(on(n, "N", quat=about("y", 30)), n)))  # rolled onto an edge

    def test_tip_gap(self):
        r, b = S.probe(on("red_probe", "L1"), "red_probe"), S.probe(on("black_probe", "L2"), "black_probe")
        self.assertAlmostEqual(S.tip_gap(r, b), L.POST_PITCH - L.PROBE_W, places=6)  # 24 mm on neighbouring posts
        r, b = S.probe(on("red_probe", "P+"), "red_probe"), S.probe(on("black_probe", "P-"), "black_probe")
        self.assertAlmostEqual(S.tip_gap(r, b), 0.040, places=6)  # 40 mm end to end on the proving unit
        # an edge-to-edge case: the black probe turned 45 degrees about z, corner toward the red tip's corner
        b2 = S.probe(state(centre_for("black_probe", (L.POSTS["L1"][0] + 0.045, 0.0)), about("z", 45)), "black_probe")
        exact_lo = 0.0  # the sampled gap is never below the true gap and at most 1 mm above it
        self.assertGreaterEqual(S.tip_gap(r, b2), exact_lo)

    def test_tip_gap_sampling_error_is_under_1mm(self):
        """Against a dense sampling, the 98-point gap is at most 1 mm too large, over random poses."""
        rng = np.random.default_rng(1)
        g = np.linspace(-1, 1, 21)
        dense = np.array([p for p in np.array(np.meshgrid(g, g, g)).reshape(3, -1).T if np.abs(p).max() == 1.0])
        for _ in range(150):
            qa, qb = (qmul(about("z", rng.uniform(-180, 180)), about("x", rng.uniform(-40, 40))) for _ in range(2))
            ca = np.array([0.25, 0.0, L.POST_TOP + 0.02])
            cb = ca + rng.uniform(-0.04, 0.04, 3) + np.array([0, 0, 0])
            a = S.probe(state(ca + [0, 0.14, 0], qa), "red_probe")
            b = S.probe(state(cb - [0, 0.14, 0], qb), "black_probe")
            fast = S.tip_gap(a, b)
            pa = a.tip_centre + (dense * S.TIP_HALF) @ a.R.T
            pb = b.tip_centre + (dense * S.TIP_HALF) @ b.R.T
            slow = min(S.point_tip_distance(pa, b).min(), S.point_tip_distance(pb, a).min())
            self.assertLessEqual(fast - slow, 0.001)
            self.assertGreaterEqual(fast - slow, -1e-9)

    def test_held(self):
        s = on("red_probe", "L2")
        p = S.probe(s, "red_probe")
        a = SimpleNamespace(**gripping(s))
        self.assertTrue(S.holds(a, p))
        for opening, want in ((0.032 / L.FULL_GAP, True), (0.033 / L.FULL_GAP, False), (1.0, False)):
            self.assertEqual(S.holds(SimpleNamespace(**{**gripping(s), "opening": opening}), p), want)
        # gripping the metal tip end, or 39 mm from it: not on the handle
        for along, want in ((0.10, True), (0.111, False), (0.14, False), (-0.149, True), (-0.16, False)):
            tcp = np.asarray(s.pos) + p.axis * along + [0, 0, 0.004]
            self.assertEqual(S.holds(SimpleNamespace(**arm(tcp, GRIP_OPENING)), p), want, along)
        beside = np.asarray(s.pos) + [0.020, 0, 0.004]  # a closed gripper 20 mm to one side of the bar
        self.assertFalse(S.holds(SimpleNamespace(**arm(beside, GRIP_OPENING)), p))

    def test_rests(self):
        n = "red_probe"
        self.assertFalse(S.on_rest(on(n, "L1"), S.probe(on(n, "L1"), n), L.REST_Y[n]))
        lifted = state(START[n] + [0, 0, 0.005])
        self.assertFalse(S.on_rest(lifted, S.probe(lifted, n), L.REST_Y[n]))
        turned = state(START[n], about("z", 30))  # its ends off the 60 mm board
        self.assertFalse(S.on_rest(turned, S.probe(turned, n), L.REST_Y[n]))
        other = state([L.REST_X, L.REST_Y["black_probe"], L.PROBE_START_Z])
        self.assertTrue(S.on_any_rest(other, S.probe(other, n)))
        self.assertFalse(S.on_rest(other, S.probe(other, n), L.REST_Y[n]))

    def test_a_tip_on_a_post_cannot_reach_another(self):
        """While a tip is on a post (contact point within 8 mm of its axis), its metal is at least 17 mm from every
        other post, whatever its heading: bridging needs a probe off its post."""
        for post in L.POSTS:
            for yaw in range(0, 360, 15):
                for dx, dy in ((0.0079, 0), (-0.0079, 0), (0, 0.0079), (0, -0.0079), (0.0056, 0.0056)):
                    x, y = L.POSTS[post]
                    q = about("z", yaw)
                    p0 = S.probe(state([0, 0, L.POST_TOP + L.PROBE_HALF_H], q), "red_probe")
                    c = np.array([x + dx, y + dy, L.POST_TOP + L.PROBE_HALF_H + 0.001]) - (p0.pin - p0.centre) * [1, 1, 0]
                    p = S.probe(state(c, q), "red_probe")
                    others = [i for i, n in enumerate(S.POST_NAMES) if n != post]
                    d = S.post_distances(S.tip_points(p))[:, others].min()
                    self.assertGreater(d, 0.017, (post, yaw, dx, dy))


@needs_stub
class Demo(unittest.TestCase):
    """The scripted demo, replayed through the kinematic stand-in."""

    @classmethod
    def setUpClass(cls):
        cls.base = Sim().play(demo_plan())

    def test_demo_scores_one(self):
        sim = self.base
        self.assertEqual(sim.last.score, 1.0, sim.last.details)
        self.assertTrue(sim.last.success)
        d = sim.last.details
        self.assertEqual(d["n_pairs"], 10)
        self.assertEqual(d["breaches"], [])
        self.assertEqual(d["early_tests"], [])
        self.assertEqual(d["cancelled_reproves"], 0)
        self.assertEqual(d["n_tests"], 12)  # the prove, the 10 pairs and the re-prove, nothing else
        self.assertGreaterEqual(min(sim.gaps), 0.022)  # the tips never closer than 22 mm (the limit is 15)

    def test_demo_timing(self):
        sim = self.base
        plan = demo_plan()
        total = max(span(w) for w in plan.values())
        first = sim.first_success()
        self.assertIsNotNone(first)
        let_go = span(plan["left"][:-2])  # 'let go' and 'up' are the last two waypoints
        t1 = (first + 1) * DT  # the first step with 1.0 and both probes still
        self.assertGreater(t1, let_go)  # 1.0 only once both grippers have opened past 32 mm
        self.assertGreater(t1 + task.SUCCESS_HOLD_S, total)  # the trial ends after the open and the retreat
        self.assertLess((total + 3.0) * HZ, L.RUN_LIMIT_STEPS)
        steps = [r.details["first_prove_step"] for r in sim.results if r.details.get("first_prove_step")]
        self.assertLess(steps[0] * DT, 12.0)

    def test_sag_tilt_and_offsets(self):
        """No sag, twice the estimated sag, sideways sag either way, the probe pitched 1 degree in the grip, tip down
        or up (2.4 mm at the tip, even with no sag to offset it), and 3 mm of opposite x error on the two arms: all
        still 1.0, and the tips never within 18 mm."""
        for kw in ({"sag": 0.0}, {"sag": 2.0}, {"lateral": -1.0}, {"lateral": 3.0}, {"tilt_deg": 1.0},
                   {"tilt_deg": -1.0}, {"sag": 0.0, "tilt_deg": -1.0}, {"dx": 0.003}, {"dx": -0.003},
                   {"sag": 2.0, "tilt_deg": 1.0}):
            sim = Sim(**kw).play(demo_plan())
            self.assertEqual(sim.last.score, 1.0, (kw, sim.last.details))
            self.assertTrue(sim.last.success, kw)
            self.assertGreaterEqual(min(sim.gaps), L.SHORT_GAP + 0.0029, kw)

    def test_without_arm_readings_the_demo_still_scores_one(self):
        sim = Sim(arms=False).play(demo_plan())
        self.assertEqual(sim.last.score, 1.0)
        self.assertGreater(sim.last.details["arms_unavailable"], 1000)

    def test_one_arm_alone(self):
        for idle in ("left", "right"):
            sim = Sim().play(demo_plan(DEMO_IDLE_ARM=idle))
            self.assertEqual(sim.last.score, 0.0, idle)
            self.assertIsNone(sim.last.details["first_prove_step"], idle)
            self.assertEqual(sim.last.details["breaches"], [], idle)  # no rule broken: nothing was ever tested

    def test_doing_nothing_scores_zero(self):
        sim = Sim().play(demo_plan(DEMO_HOLD_ONLY="1"), tail_s=10.0)
        self.assertEqual(sim.last.score, 0.0)
        self.assertEqual(sim.last.details["breaches"], [])


class Tiers(unittest.TestCase):
    def test_full_procedure_scores_one(self):
        t = Tracker()
        r = procedure(t)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)

    def test_prove_only_scores_quarter(self):
        t = Tracker()
        r = procedure(t, pairs=[], reprove=False)
        self.assertEqual(r.score, 0.25)

    def test_each_pair_adds_005(self):
        for k in range(11):
            t = Tracker()
            r = procedure(t, pairs=DEMO_ORDER[:k], reprove=False)
            self.assertAlmostEqual(r.score, 0.25 + 0.05 * k, places=9, msg=str(k))
        self.assertEqual(r.score, 0.75)

    def test_reprove_without_putting_back_scores_09(self):
        t = Tracker()
        r = procedure(t, put_back=False)
        self.assertEqual(r.score, 0.9)
        self.assertFalse(r.success)

    def test_put_back_but_still_gripped_scores_09(self):
        t = Tracker()
        procedure(t)
        red, black = at_rest("red_probe"), at_rest("black_probe")
        r = feed(t, red, black, 5, arms=arms_for(red, black, held=("right",)))
        self.assertEqual(r.score, 0.9)

    def test_success_needs_stillness(self):
        t = Tracker()
        procedure(t)
        r = feed(t, at_rest("red_probe", vel=(0, 0.02, 0)), at_rest("black_probe"))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)

    def test_tests_before_the_prove_count_nothing(self):
        t = Tracker()
        pick(t)
        for a, b in DEMO_ORDER:
            test_steps(t, a, b)
        r = feed(t, over("red_probe", "P+"), over("black_probe", "P-"))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(r.details["early_tests"]), 10)
        test_steps(t, "P+", "P-")  # proving now does not make them count
        r = home(t, 5)
        self.assertEqual(r.score, 0.25)

    def test_reprove_before_finishing_does_not_count(self):
        t = Tracker()
        r = procedure(t, pairs=DEMO_ORDER[:9])  # nine pairs, re-proved, put back
        self.assertEqual(r.score, 0.70)
        self.assertIsNone(r.details["reprove_step"])
        test_steps(t, *DEMO_ORDER[9])  # the tenth pair, then nothing: 0.75, the earlier re-prove does not count
        r = home(t, 5)
        self.assertEqual(r.score, 0.75)

    def test_a_test_after_the_reprove_cancels_it(self):
        t = Tracker()
        procedure(t, put_back=False)
        test_steps(t, "L1", "L2")
        r = home(t, 5)
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["cancelled_reproves"], 1)
        test_steps(t, "P+", "P-")
        r = home(t, 5)
        self.assertEqual(r.score, 1.0)

    def test_either_probe_on_either_post(self):
        """Pairs are unordered: the black tip on the nearer post counts the same."""
        t = Tracker()
        r = procedure(t, pairs=[(b, a) for a, b in DEMO_ORDER])
        self.assertEqual(r.score, 1.0)

    def test_hold_shorter_than_1s_does_not_count(self):
        t = Tracker()
        r = procedure(t, pairs=[("L1", "L2")], reprove=False, hold=19)
        self.assertEqual(r.score, 0.0)  # even the prove was held only 0.95 s
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-", 20)
        test_steps(t, "L1", "L2", 19)
        r = home(t, 5)
        self.assertEqual(r.score, 0.25)

    def test_a_proving_terminal_with_a_supply_terminal_counts_nothing(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        test_steps(t, "P+", "L2")
        r = test_steps(t, "L1", "P-")
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["n_tests"], 3)

    def test_score_is_idempotent(self):
        t = Tracker()
        procedure(t, pairs=DEMO_ORDER[:4], reprove=False)
        red, black = at_rest("red_probe"), at_rest("black_probe")
        a = t.result(red, black).score
        self.assertEqual(a, t.result(red, black).score)
        self.assertEqual(a, 0.45)


class Rules(unittest.TestCase):
    """The rules for the whole trial: each breach scores 0, whatever follows."""

    def assertZeroForGood(self, t, key):
        self.assertIn(key, t.breaches)
        home(t)
        r = procedure(t)  # a perfect procedure afterwards does not undo it
        self.assertEqual(r.score, 0.0)

    def test_tips_within_15mm(self):
        t = Tracker()
        pick(t)
        feed(t, on("red_probe", "L1", dx=0.004), on("black_probe", "L2", dx=-0.006), 1)  # 14 mm apart
        self.assertZeroForGood(t, "tips_short")

    def test_tips_16mm_apart_is_allowed(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        r = feed(t, on("red_probe", "L1", dx=0.004), on("black_probe", "L2", dx=-0.004), 25)  # 16 mm apart
        self.assertEqual(r.details["breaches"], [])
        self.assertEqual(r.details["n_pairs"], 1)

    def test_both_tips_on_one_post(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        r = feed(t, on("red_probe", "L3", dy=0.007), on("black_probe", "L3", dy=-0.007), 30)  # end to end
        self.assertEqual(r.score, 0.0)
        self.assertIn("tips_short", r.details["breaches"])

    def test_tip_bridging_two_terminals(self):
        """Not reachable with a tip on a post (see Geometry), but a held probe whose tip is turned 45 degrees and
        pushed down between L1 and L2, onto the block, comes within 0.75 mm of both (its 32.5 mm diagonal span in
        the posts' 34 mm gap): a short across two conductors."""
        t = Tracker()
        pick(t)
        q = about("z", 45)
        p0 = S.probe(state([0, 0, 0], q), "red_probe")
        mid = np.array([(L.POSTS["L1"][0] + L.POSTS["L2"][0]) / 2, 0.0, L.BASE_TOP + L.PROBE_HALF_H])
        wedged = state(mid - (p0.tip_centre - p0.centre), q)
        p = S.probe(wedged, "red_probe")
        self.assertEqual(sorted(S.posts_touched(p)), ["L1", "L2"])
        feed(t, wedged, over("black_probe", "N"), 1)
        self.assertNotIn("loose_on_terminal", t.breaches)  # it is held
        self.assertZeroForGood(t, "tip_bridged")

    def test_letting_go_of_a_probe_on_a_post(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        red, black = on("red_probe", "L1"), on("black_probe", "L2")
        feed(t, red, black, 10)
        feed(t, red, black, 1, arms=arms_for(red, black, held=("right",)))  # the left gripper opens
        self.assertZeroForGood(t, "loose_on_terminal")

    def test_probe_laid_across_two_posts(self):
        """The red probe turned along x and laid across L2 and L3, not held: its tip touches L3."""
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        q = about("z", 90)  # body y now along -x: the tip end points to +x for the red probe
        p0 = S.probe(state([0, 0, 0], q), "red_probe")
        c = np.array([L.POSTS["L3"][0], 0.0, 0.0]) - p0.pin + [0, 0, L.POST_TOP]
        laid = state(c, q)
        self.assertEqual(S.on_post(S.probe(laid, "red_probe")), "L3")
        black = on("black_probe", "PE")
        feed(t, laid, black, 1, arms=arms_for(laid, black, held=("right",)))
        self.assertZeroForGood(t, "loose_on_terminal")

    def test_probe_set_down_on_the_table(self):
        t = Tracker()
        pick(t)
        red = state([0.30, 0.30, L.TOP + L.PROBE_HALF_H])  # on the bare table, away from everything
        black = over("black_probe", "P-")
        feed(t, red, black, 5, arms=arms_for(red, black, held=("right",)))
        self.assertNotIn("probe_set_down", t.breaches)  # 0.25 s grace: a fumble
        feed(t, red, black, 1, arms=arms_for(red, black, held=("right",)))
        self.assertZeroForGood(t, "probe_set_down")

    def test_a_brief_fumble_is_not_a_set_down(self):
        t = Tracker()
        pick(t)
        red, black = over("red_probe", "P+"), over("black_probe", "P-")
        feed(t, red, black, 5, arms=arms_for(red, black, held=("right",)))
        r = procedure(t)
        self.assertEqual(r.score, 1.0)

    def test_probe_on_the_other_rest_is_not_a_set_down(self):
        t = Tracker()
        red = state([L.REST_X, L.REST_Y["black_probe"], L.PROBE_START_Z])
        black = state([0.30, -0.05, L.TOP + 0.10])
        feed(t, red, black, 30, arms=arms_for(red, black, held=("right",)))
        self.assertEqual(t.breaches, {})

    def test_off_the_table(self):
        t = Tracker()
        pick(t)
        red = state([0.30, 0.60, L.TOP - 0.05])
        black = over("black_probe", "P-")
        feed(t, red, black, 1, arms=arms_for(red, black))  # (as if still gripped: off the table all the same)
        self.assertZeroForGood(t, "off_table")


class Exploits(unittest.TestCase):
    def test_sweeping_one_tip_along_the_row(self):
        """The red tip rests on L1 while the black tip is swept along the row at 5 cm/s, touching every post top:
        nothing counts (each post is under it for 0.3 s, and it is not steady)."""
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        red = on("red_probe", "L1")
        x0, x1 = L.POSTS["L2"][0] - 0.008, L.POSTS["PE"][0] + 0.02  # from 16 mm short of the red tip
        n = int(round((x1 - x0) / 0.05 * HZ))
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            black = state(centre_for("black_probe", (x, 0.0)), vel=(0.05, 0, 0))
            r = feed(t, red, black)
        self.assertEqual(r.details["n_pairs"], 0)
        self.assertEqual(r.details["breaches"], [])

    def test_a_tip_moving_slowly_but_over_1cm_per_s_does_not_count(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        red = on("red_probe", "L1")
        r = None
        for i in range(30):
            black = state(centre_for("black_probe", (L.POSTS["L2"][0] - 0.0075 + 0.0005 * i, 0.0)), vel=(0.011, 0, 0))
            r = feed(t, red, black)
        self.assertEqual(r.details["n_pairs"], 0)

    def test_tilted_probe_does_not_count(self):
        t = Tracker()
        pick(t)
        test_steps(t, "P+", "P-")
        feed(t, on("red_probe", "L1", quat=about("x", 12)), on("black_probe", "L2"), 30)
        r = home(t, 3)
        self.assertEqual(r.score, 0.25)

    def test_open_grippers_around_the_probes_do_not_hold_them(self):
        t = Tracker()
        pick(t)
        red, black = on("red_probe", "P+"), on("black_probe", "P-")
        a = {s: {**v, "opening": 0.5} for s, v in arms_for(red, black).items()}  # 47 mm: not closed on them
        r = feed(t, red, black, 1, arms=a)
        self.assertEqual(r.score, 0.0)
        self.assertIn("loose_on_terminal", r.details["breaches"])

    def test_one_gripper_cannot_hold_both(self):
        """One gripper closed on the red probe; the black probe's tip on P- with no gripper on it (pushed there): a
        loose probe on a terminal."""
        t = Tracker()
        pick(t)
        red, black = on("red_probe", "P+"), on("black_probe", "P-")
        a = arms_for(red, black, held=("left",))
        a["right"] = gripping(red)  # both grippers on the red probe
        r = feed(t, red, black, 1, arms=a)
        self.assertEqual(r.score, 0.0)
        self.assertIn("loose_on_terminal", r.details["breaches"])

    def test_both_grippers_on_one_probe_is_not_a_test(self):
        red = on("red_probe", "P+")
        a = {k: SimpleNamespace(**gripping(red)) for k in ("left", "right")}
        h_red = S.holders(a, S.probe(red, "red_probe"))
        self.assertEqual(h_red, {"left", "right"})
        self.assertFalse(S.held_apart(h_red, set()))
        self.assertFalse(S.held_apart({"left"}, {"left"}))  # the same single gripper on both is not 'held apart'

    def test_doing_nothing(self):
        t = Tracker()
        r = feed(t, at_rest("red_probe"), at_rest("black_probe"), 200)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["breaches"], [])


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        t = Tracker()
        pick(t)
        r = feed(t, on("red_probe", "P+"), on("black_probe", "P-"), 25, arms=None)
        self.assertEqual(r.details["arms_unavailable"], 25)
        self.assertIsNotNone(r.details["first_prove_step"])  # 'held' taken as met
        self.assertEqual(r.details["breaches"], [])

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker()
        bad = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=0.3),
               "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3), opening=0.3)}
        r = t.update(at_rest("red_probe"), at_rest("black_probe"), bad)
        self.assertEqual(r.details["arms_unavailable"], 1)

    def test_offline_feed_reaches_the_scorer(self):
        red, black = on("red_probe", "P+"), on("black_probe", "P-")
        rigstate.offline_feed(arms_for(red, black))
        a = rigstate.arms()
        self.assertTrue(S.holds(a["left"], S.probe(red, "red_probe")))
        self.assertTrue(S.holds(a["right"], S.probe(black, "black_probe")))


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker()
        procedure(t, pairs=DEMO_ORDER[:3], reprove=False)
        before = (dict(t.tested), t.first_prove, dict(t.breaches))
        for bad in (state([np.nan, 0, 0]), SimpleNamespace(**{**vars(at_rest("red_probe")), "quat_xyzw": np.zeros(4)}),
                    SimpleNamespace(pos=np.zeros(3))):
            r = t.update(bad, at_rest("black_probe"), None)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual((dict(t.tested), t.first_prove, dict(t.breaches)), before)
        self.assertEqual(t.bad_steps, 3)
        r = feed(t, at_rest("red_probe"), at_rest("black_probe"))
        self.assertEqual(r.score, 0.40)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(0)
        t = Tracker()
        for _ in range(300):
            q = rng.normal(size=4)
            s = [state(rng.uniform([-0.2, -0.6, 0.5], [0.8, 0.6, 1.2]), q / np.linalg.norm(q),
                       rng.normal(size=3), rng.normal(size=3)) for _ in range(2)]
            r = t.update(*s, None)
            self.assertIn(r.score, (0.0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.9, 1.0))


class Reset(unittest.TestCase):
    def breached(self):
        t = Tracker()
        pick(t)
        red, black = on("red_probe", "L1"), on("black_probe", "L1", dy=-0.004)
        feed(t, red, black, 3)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = feed(t, at_rest("red_probe"), at_rest("black_probe"))
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        a, b = on("red_probe", "L1").pos, START["red_probe"]
        for i in range(1, 41):
            c = a + (b - a) * i / 40
            feed(t, state(c, vel=(b - a) / 2.0), at_rest("black_probe") if i == 40 else over("black_probe", "P-"))
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        feed(t, state([0.40, 0.40, L.TOP + 0.10]), at_rest("black_probe"))
        self.assertTrue(t.breach)

    @needs_stub
    def test_rigstate_reset_hook_resets_the_tracker(self):
        tr = task._TRACKER
        tr.reset()
        tr.breaches["tips_short"] = 3
        rigstate.offline_reset()
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)


@needs_stub
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook, trace and demo."""

    def objects(self, **over):
        out = {n: at_rest(n) for n in NAMES}
        out.update(over)
        return out

    def test_objects(self):
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in (*NAMES, "proving_unit", "terminal_block", "red_rest", "black_rest", "post_pe_cap", "post_p_pos_cap",
                  "post_p_neg_collar", "post_l1_collar"):
            self.assertIn(n, names)
        loose = {o.name for o in task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, set(NAMES))
        self.assertEqual(task.SCORE_OVER, "end")
        self.assertEqual(task.SCENE, "station")
        by = {o.name: o for o in task.OBJECTS}
        for n in NAMES:  # the scorer's starting places match the objects'
            self.assertTrue(np.allclose(by[n].pos, START[n]), n)
            self.assertEqual(tuple(by[n].size), L.PROBE_SIZE)
        for n, (x, y) in L.POSTS.items():  # every post's cap top is the scorer's post top
            tag = n.lower().replace("+", "_pos").replace("-", "_neg")
            cap = by[f"post_{tag}_cap"]
            self.assertAlmostEqual(cap.pos[2] + cap.height / 2, L.POST_TOP)
            self.assertEqual((cap.pos[0], cap.pos[1]), (x, y))
            col = by[f"post_{tag}_collar"]
            self.assertAlmostEqual(col.pos[2] - col.height / 2, L.BASE_TOP)
        self.assertGreater(task.SUCCESS_HOLD_S, 1.5 + 1.0)
        self.assertIn("10 tests", task.INSTRUCTION)

    def test_hooks_and_reset_hook(self):
        tr = task._TRACKER
        rigstate.offline_reset()
        red, black = on("red_probe", "L1"), on("black_probe", "L1", dy=-0.004)
        rigstate.offline_feed(arms_for(red, black))
        tr.prev = None
        self.assertFalse(task.success(self.objects(red_probe=red, black_probe=black)))
        self.assertIn("tips_short", tr.breaches)
        self.assertEqual(task.score(self.objects(red_probe=red, black_probe=black)), 0.0)
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

    def test_trace_writes_a_line(self):
        import json
        import tempfile
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                rigstate.offline_reset()
                task.success(self.objects())
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
            row = json.loads(open(path).read().splitlines()[-1])
            self.assertIn("min_tip_gap_mm", row)
            self.assertIn("pairs_missing", row)

    def test_demo_switches_and_timing(self):
        plan = demo_plan()
        self.assertEqual(set(plan), {"left", "right"})
        self.assertEqual(set(demo_plan(DEMO_IDLE_ARM="left")), {"right"})
        self.assertEqual(set(demo_plan(DEMO_IDLE_ARM="right")), {"left"})
        self.assertEqual(demo_plan(DEMO_HOLD_ONLY="1"), {})
        self.assertAlmostEqual(span(plan["left"]), span(plan["right"]), places=6)
        self.assertLess((max(span(plan["left"]), span(plan["right"])) + 3.0) * HZ, L.RUN_LIMIT_STEPS)
        # the commanded contact never pushes into a post: the tip's underside is commanded 1 mm above the top
        self.assertAlmostEqual(task.CONTACT_Z - task.TCP_OVER_BOTTOM - L.POST_TOP, 0.001)
        lowest = min(w.pos[2] for w in plan["left"] + plan["right"] if "tip" in w.label)
        self.assertAlmostEqual(lowest, task.CONTACT_Z)


if __name__ == "__main__":
    unittest.main()
