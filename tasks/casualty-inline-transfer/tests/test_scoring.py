"""Offline tests for scoring.py (and the hooks in task.py): synthetic manikin states (trunk pose, neck angle, the head
link's pose by forward kinematics) and synthetic arm snapshots fed through rigstate.offline_feed, no simulator.
    python3 -m unittest discover -s tests -v"""

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

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
START = np.array(L.MANIKIN_ORIGIN, dtype=float)  # trunk frame origin on the mat
PLACED = np.array([L.STRETCHER_X, L.MANIKIN_ORIGIN[1], L.TOP + L.STRETCHER_TOP])  # head on the head zone
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
GRIP_L = np.array([0.0, 0.200, 0.022])  # where the demo holds the head and the trunk, relative to the trunk's origin
GRIP_R = np.array([0.0, -0.030, 0.022])
OPEN = 1.0
HEAD_SHUT = (L.HEAD_SIZE[0] - 0.008) / 0.0939  # rigstate openings when closed on the head / the trunk
TRUNK_SHUT = (L.TRUNK_SIZE[0] - 0.008) / 0.0939
CLEAR_UP = 0.087  # the demo's final TCP height over the manikin's origin


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about 'x', 'y' or 'z'."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def man(o=START, th=0.0, yaw=0.0, roll=0.0, pitch=0.0, vel=(0, 0, 0), spin=(0, 0, 0), thv=0.0):
    """A manikin state as mechkit reports it: trunk origin and orientation (yaw about z, then pitch about the trunk's
    x, roll about its y, degrees), neck angle ``th`` degrees, and the neck link's frame by forward kinematics."""
    qb = qmul(about("z", yaw), qmul(about("x", pitch), about("y", roll)))
    rb = S.rotation(qb)
    qh = qmul(qb, about("x", th))
    o = np.array(o, dtype=float)
    ph = o + rb @ np.array(L.NECK_ANCHOR)
    st = SimpleNamespace(origin=o, quat_xyzw=qb, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                         joints={"neck": math.radians(th)}, joint_vel={"neck": float(thv)},
                         links={"neck": np.concatenate([ph, qh])})
    c = np.vstack([S.corners(st, b) for b in S.BODIES])
    st.low, st.high = c.min(axis=0), c.max(axis=0)
    st.pos = (st.low + st.high) / 2
    return st


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("o", "th", "yaw", "pitch", "L", "Lo", "R", "Ro")
VEC = ("o", "L", "R")


class Film:
    """A trial as a list of frames: the trunk's origin, yaw and pitch, the neck angle, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"o": START, "th": 0.0, "yaw": 0.0, "pitch": 0.0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float) for k in KEYS})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def carry(self, n, **to):
        """Move the manikin with both grippers on it (the TCPs follow the trunk's origin)."""
        o = np.array(to.get("o", self.cur["o"]), float)
        return self.go(n, L=o + GRIP_L, R=o + GRIP_R, **to)


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r, prev = None, None
    frames = film.frames[:until] if until else film.frames
    for fr in frames:
        if prev is None:
            v, w, thv = np.zeros(3), np.zeros(3), 0.0
        else:
            v = (fr["o"] - prev["o"]) / DT
            w = np.array([0.0, 0.0, math.radians(fr["yaw"] - prev["yaw"]) / DT])
            thv = math.radians(fr["th"] - prev["th"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(man(fr["o"], fr["th"], fr["yaw"], pitch=fr["pitch"], vel=v, spin=w, thv=thv), a)
        prev = fr
    return r, t


def procedure(lift=0.050, place=PLACED, carry_th=0.0, final_th=0.0, open_end=True, rise=True):
    """The demo's transfer: both grippers down and closed on head and trunk, both bodies up ``lift``, across, down
    onto the stretcher, let go, grippers up; then still."""
    up = START + [0, 0, lift]
    f = Film()
    f.go(40, L=START + GRIP_L + [0, 0, 0.08], R=START + GRIP_R + [0, 0, 0.08])
    f.go(30, L=START + GRIP_L, R=START + GRIP_R)
    f.go(30, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
    f.carry(40, o=up, th=carry_th)
    f.carry(60, o=[place[0], place[1], up[2]])
    f.carry(40, o=np.array(place) + [0, 0, 0.001], th=final_th)
    f.carry(2, o=place)
    if open_end:
        f.go(30, Lo=OPEN, Ro=OPEN)
        if rise:
            f.go(40, L=np.array(place) + GRIP_L + [0, 0, CLEAR_UP], R=np.array(place) + GRIP_R + [0, 0, CLEAR_UP])
    f.hold(30)
    return f


def head_resting_theta(dz):
    """Neck angle (deg) at which the head's lowest point touches the mat when the level trunk is ``dz`` up."""
    lo, hi = -40.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if S.low(man(START + [0, 0, dz], th=mid), "head") < L.MAT_TOP:
            lo = mid
        else:
            hi = mid
    return hi


class Geometry(unittest.TestCase):
    def test_at_rest_on_the_mat(self):
        m = man()
        for b in S.BODIES:
            self.assertAlmostEqual(S.low(m, b), L.MAT_TOP, places=9)
            self.assertEqual(S.surface_under(m, b), (L.MAT_TOP, "mat"))
            self.assertAlmostEqual(S.lift(m, b), 0.0, places=9)
        self.assertAlmostEqual(S.neck_deg(m), 0.0)
        np.testing.assert_allclose(S.head_centre(m)[:2], [0.200, 0.100], atol=1e-9)
        np.testing.assert_allclose(S.neck_centre(m), [0.200, 0.050, L.TOP + 0.023], atol=1e-9)
        fp = S.footprint(m, "trunk")
        np.testing.assert_allclose(fp, (0.165, 0.235, -0.205, 0.035), atol=1e-9)

    def test_neck_bend_moves_only_the_head(self):
        up, down = man(th=10.0), man(th=-10.0)
        np.testing.assert_allclose(S.corners(up, "trunk"), S.corners(man(), "trunk"), atol=1e-12)
        self.assertGreater(S.head_centre(up)[2], S.head_centre(man())[2] + 0.005)  # positive lifts the crown
        # bent down 10 deg about the hinge (y 0.120, z 0.020 in the base frame): the crown's bottom corner sinks
        a = math.radians(10)
        crown = 0.020 + (0.220 - 0.120) * math.sin(-a) + (0.000 - 0.020) * math.cos(a)
        self.assertAlmostEqual(S.low(down, "head"), L.MAT_TOP + crown, places=9)
        self.assertAlmostEqual(S.neck_deg(down), 10.0)

    def test_surface_under(self):
        self.assertEqual(S.surface_under(man(PLACED), "trunk")[1], "stretcher")
        self.assertEqual(S.surface_under(man(START + [-0.10, 0, 0.05]), "head")[1], "table")  # behind the mat
        self.assertEqual(S.surface_under(man(START + [0.08, 0, 0.05]), "head")[1], "table")  # middle in the gap
        self.assertEqual(S.surface_under(man(START + [0.10, 0, 0.05]), "trunk")[1], "table")  # 25 mm over the edge
        self.assertEqual(S.surface_under(man(START + [0.12, 0, 0.05]), "trunk")[1], "stretcher")  # middle over it
        self.assertEqual(S.surface_under(man(START + [0, 0, 0.05]), "trunk")[1], "mat")
        # half over the stretcher's edge, just above its top, middle over the table: 31 mm up, lifted
        m = man([L.STRETCHER_RECT[0] - 0.005, START[1], L.TOP + 0.031])
        self.assertTrue(S.inside_stretcher(m, "trunk") is False and S.lift(m, "trunk") > 0.030)

    def test_trunk_tilt_and_axis(self):
        self.assertAlmostEqual(S.trunk_tilt_deg(man(roll=180.0)), 180.0, places=6)  # face down is not on its back
        self.assertAlmostEqual(S.trunk_tilt_deg(man(roll=170.0)), 170.0, places=6)
        self.assertAlmostEqual(S.trunk_tilt_deg(man(roll=90.0)), 90.0, places=6)
        self.assertAlmostEqual(S.trunk_tilt_deg(man(roll=12.0)), 12.0, places=6)
        self.assertAlmostEqual(S.trunk_tilt_deg(man(roll=-12.0)), 12.0, places=6)
        self.assertAlmostEqual(S.trunk_axis_deg(man(yaw=12.0)), 12.0, places=6)
        self.assertAlmostEqual(S.trunk_axis_deg(man(yaw=180.0)), 180.0, places=6)  # head toward the hips' side

    def test_neck_zone(self):
        m = man(START + [0, 0, 0.03])
        n = S.neck_centre(m)
        self.assertTrue(S.neck_zone(m, n))
        self.assertTrue(S.neck_zone(m, n + [0.049, 0.024, 0.059]))
        self.assertFalse(S.neck_zone(m, n + [0.0, 0.026, 0.0]))
        self.assertFalse(S.neck_zone(m, n + [0.051, 0.0, 0.0]))
        self.assertFalse(S.neck_zone(m, n + [0.0, 0.0, 0.061]))
        t = man(START + [0, 0, 0.03], yaw=90.0)  # turned: 'along' follows the manikin, not the world's y
        self.assertTrue(S.neck_zone(t, S.neck_centre(t) + [0.02, 0.04, 0.0]))
        self.assertFalse(S.neck_zone(t, S.neck_centre(t) + [0.03, 0.0, 0.0]))
        demo_left = START + GRIP_L  # the demo's head grip is 65 mm from the neck bar's centre
        self.assertFalse(S.neck_zone(man(), demo_left))

    def test_point_to_manikin_distance(self):
        m = man(PLACED)
        top = L.TOP + L.STRETCHER_TOP + L.HEAD_SIZE[2]
        self.assertAlmostEqual(S.distance_to_manikin(m, [0.365, 0.100, top + 0.030]), 0.030, places=9)
        self.assertEqual(S.distance_to_manikin(m, S.head_centre(m)), 0.0)
        side = [0.365 + L.TRUNK_SIZE[0] / 2 + 0.020, -0.085, L.TOP + 0.05]
        self.assertAlmostEqual(S.distance_to_manikin(m, side), 0.020, places=9)

    def test_gripper_clear(self):
        m = man(PLACED)
        over = PLACED + GRIP_L + [0, 0, CLEAR_UP]
        arm = SimpleNamespace(tcp=over, tips=over - [0, 0, TIPS], opening=1.0)
        self.assertTrue(S.gripper_clear(m, arm))
        self.assertFalse(S.gripper_clear(m, SimpleNamespace(tcp=over, tips=over - [0, 0, TIPS], opening=0.85)))
        low_ = PLACED + GRIP_L
        self.assertFalse(S.gripper_clear(m, SimpleNamespace(tcp=low_, tips=low_ - [0, 0, TIPS], opening=1.0)))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(5))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertIsNone(t.breach)

    def test_full_transfer_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertTrue(t.lifted_together)
        self.assertIsNone(t.breach)
        self.assertTrue(r.details["left_clear"] and r.details["right_clear"])

    def test_success_needs_stillness(self):
        f = procedure()
        r, t = play(f)
        self.assertTrue(r.success)
        last = f.frames[-1]
        for kw in ({"vel": (0.02, 0, 0)}, {"thv": 0.2}, {"spin": (0, 0, 0.6)}):
            st = man(last["o"], last["th"], **kw)
            res = t.result(st, snap(last["L"], 1.0, last["R"], 1.0))
            self.assertEqual(res.score, 1.0)
            self.assertFalse(res.success, kw)
        t2 = Tracker(HZ)
        play(procedure(), t2)
        t2.update(man(PLACED + [0.004, 0, 0]), snap(last["L"], 1.0, last["R"], 1.0))  # slid 4 mm in one step
        self.assertFalse(t2.last.success)

    def test_grippers_still_closed_scores_075(self):
        r, _ = play(procedure(open_end=False))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_grippers_open_just_above_it_score_075(self):
        # Open, but the finger tips only 22 mm over the head and 26 mm over the trunk: not yet let go.
        f = procedure(rise=False)
        f.go(20, L=PLACED + GRIP_L + [0, 0, 0.057], R=PLACED + GRIP_R + [0, 0, 0.057]).hold(20)
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["left_clear"] or r.details["right_clear"])

    def test_grippers_open_but_still_around_it_scores_075(self):
        r, _ = play(procedure(rise=False))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["left_clear"])

    def test_worst_neck_tiers(self):
        cases = [  # (worst bend during the carry, final bend) -> score
            (5.9, 2.9, 1.0), (6.5, 0.0, 0.75), (5.0, 3.5, 0.75), (9.9, 4.9, 0.75), (10.5, 0.0, 0.5),
            (9.0, 6.0, 0.5), (14.9, 0.0, 0.5), (15.5, 0.0, 0.25), (19.9, 0.0, 0.25), (20.5, 0.0, 0.0),
        ]
        for worst, final, want in cases:
            with self.subTest(worst=worst, final=final):
                r, t = play(procedure(carry_th=worst, final_th=final))
                self.assertEqual(r.score, want, r.details)
                self.assertAlmostEqual(t.worst_neck, max(worst, final), places=6)

    def test_bend_either_way_counts(self):
        r, _ = play(procedure(carry_th=-8.0))
        self.assertEqual(r.score, 0.75)

    def test_on_stretcher_tolerances(self):
        ok, _ = S.on_stretcher(man(PLACED))
        self.assertTrue(ok)
        cases = {
            "head 24 mm off along": (man(PLACED + [0, 0.024, 0]), True),
            "head 26 mm off along": (man(PLACED + [0, 0.026, 0]), False),
            "18 mm off across": (man(PLACED + [0.018, 0, 0]), True),
            "22 mm off across, over the edge": (man(PLACED + [0.022, 0, 0]), False),
            "turned 5 deg": (man(PLACED, yaw=5.0), True),
            "turned 16 deg": (man(PLACED, yaw=16.0), False),
            "rolled 8 deg onto an edge": (man(PLACED + [0, 0, 0.035 * math.sin(math.radians(8))], roll=8.0), True),
            "rolled 12 deg onto an edge": (man(PLACED + [0, 0, 0.035 * math.sin(math.radians(12))], roll=12.0), False),
            "held 5 mm up": (man(PLACED + [0, 0, 0.005]), False),
            "3.5 mm up": (man(PLACED + [0, 0, 0.0035]), True),
            "trunk on its side": (man(PLACED + [0, 0, 0.015], roll=90.0), False),
            "face down": (man(PLACED + [0, 0, 0.042], roll=180.0), False),  # a casualty goes on its back
            "head first, reversed": (man([L.STRETCHER_X, 0.015, L.TOP + L.STRETCHER_TOP], yaw=180.0), False),
        }
        for name, (m, want) in cases.items():
            with self.subTest(name):
                self.assertEqual(S.on_stretcher(m)[0], want, S.on_stretcher(m)[1])

    def test_lifted_together_then_back_on_the_mat_scores_quarter(self):
        f = Film()
        f.go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(30, o=START + [0, 0, 0.025]).carry(30, o=START).go(20, Lo=OPEN, Ro=OPEN).hold(20)
        r, t = play(f)
        self.assertTrue(t.lifted_together)
        self.assertEqual(r.score, 0.25)

    def test_lifted_together_needs_a_straight_neck(self):
        f = Film()
        f.go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(20, o=START + [0, 0, 0.010], th=12.0).carry(20, o=START + [0, 0, 0.025])  # bent first, then up
        f.carry(30, o=START, th=0.0).go(20, Lo=OPEN, Ro=OPEN).hold(20)
        r, t = play(f)
        self.assertIsNone(t.breach)
        self.assertFalse(t.lifted_together)
        self.assertEqual(r.score, 0.0)

    def test_lift_of_19_mm_is_not_lifted_together(self):
        f = Film()
        f.go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(30, o=START + [0, 0, 0.019]).carry(30, o=START).hold(10)
        r, t = play(f)
        self.assertFalse(t.lifted_together)
        self.assertEqual(r.score, 0.0)

    def test_score_is_idempotent(self):
        f = procedure()
        _, t = play(f)
        last = f.frames[-1]
        st, arms = man(last["o"], last["th"]), snap(last["L"], 1.0, last["R"], 1.0)
        before = (t.steps, t.worst_neck, t.breach, t.lifted_together)
        for _ in range(3):
            self.assertEqual(t.result(st, arms).score, 1.0)
        self.assertEqual((t.steps, t.worst_neck, t.breach, t.lifted_together), before)


class ZeroRules(unittest.TestCase):
    def test_neck_over_20_is_zero_whatever_follows(self):
        f = Film()
        f.go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(20, o=START + [0, 0, 0.04], th=21.0).carry(10, th=0.0)
        r, t = play(f)
        self.assertEqual(t.breach, "neck_bent")
        f2 = procedure()
        f2.frames = f.frames + f2.frames[len(f.frames):]
        r, t = play(f2)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breach, "neck_bent")

    def test_lift_cap(self):
        r, t = play(procedure(lift=0.0765))  # lowest points 79.5 mm above the table
        self.assertEqual(r.score, 1.0)
        r, t = play(procedure(lift=0.0785))  # 81.5 mm
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breach, "lifted_too_high")

    def test_one_body_lifted_alone(self):
        # head 16 mm up (the trunk tilted about its far end, the head level) while the trunk's far end rests: 0
        def head_up(mm):
            a = 0.0
            for _ in range(60):  # trunk pitch that raises the head's underside by mm
                m = man(START + [0, 0, 0.12 * math.sin(a)], th=-math.degrees(a), pitch=math.degrees(a))
                if S.lift(m, "head") < mm / 1000:
                    a += 0.002
            return m
        t = Tracker(HZ)
        snap()
        m = head_up(14)
        t.update(m, rigstate.arms())
        self.assertLessEqual(S.lift(m, "trunk"), L.LIFTED)
        self.assertIsNone(t.breach)
        m = head_up(16)
        t.update(m, rigstate.arms())
        self.assertEqual(t.breach, "head_lifted_alone")
        # the trunk 16 mm up while the head rests on the mat
        t = Tracker(HZ)
        t.update(man(START + [0, 0, 0.016], th=head_resting_theta(0.016)), snap())
        self.assertEqual(t.breach, "trunk_lifted_alone")

    def test_a_body_4_mm_up_is_not_lifted(self):
        # Lifted means more than 5 mm above the surface: the trunk 16 mm up with the head only 4 mm up is the trunk
        # lifted alone; and a tool point at the neck while both are at most 4 mm up is not yet a neck hold.
        dz, lo, hi = 0.016, head_resting_theta(0.016), 0.0
        for _ in range(60):  # the neck angle that leaves the head's underside 4 mm over the mat
            mid = (lo + hi) / 2
            if S.lift(man(START + [0, 0, dz], th=mid), "head") > 0.004:
                hi = mid
            else:
                lo = mid
        m = man(START + [0, 0, dz], th=(lo + hi) / 2)
        self.assertAlmostEqual(S.lift(m, "head"), 0.004, delta=0.0003)
        t = Tracker(HZ)
        t.update(m, snap())
        self.assertEqual(t.breach, "trunk_lifted_alone")
        low4 = man(START + [0, 0, 0.004])
        t = Tracker(HZ)
        t.update(low4, snap(left_tcp=S.neck_centre(low4), left_open=0.3))
        self.assertIsNone(t.breach)

    def test_lifts_slightly_out_of_step_are_allowed(self):
        t = Tracker(HZ)
        m = man(START + [0, 0, 0.012], th=-1.5)  # trunk 12 mm up, head 6-7 mm up: both lifted
        self.assertGreater(S.lift(m, "head"), L.LIFTED)
        t.update(m, snap())
        self.assertIsNone(t.breach)

    def test_held_by_the_neck(self):
        m = man(START + [0, 0, 0.03])
        n = S.neck_centre(m)
        t = Tracker(HZ)
        t.update(m, snap(left_tcp=n, left_open=0.3))
        self.assertEqual(t.breach, "held_by_the_neck")
        t = Tracker(HZ)  # the same tool point while the manikin lies on the mat: not yet a breach
        t.update(man(), snap(left_tcp=S.neck_centre(man()), left_open=1.0))
        self.assertIsNone(t.breach)
        self.assertEqual(t.in_zone, ["left"])
        t = Tracker(HZ)  # the right arm's tool point at the neck counts too
        t.update(m, snap(right_tcp=n + [0, -0.02, 0.01]))
        self.assertEqual(t.breach, "held_by_the_neck")

    def test_a_fall(self):
        t = Tracker(HZ)
        a = snap()
        t.update(man(PLACED + [0, 0, 0.030]), a)
        t.update(man(PLACED + [0, 0, 0.017]), a)  # 13 mm in one step: 0.26 m/s
        self.assertEqual(t.breach, "trunk_fell")
        t = Tracker(HZ)
        t.update(man(PLACED + [0, 0, 0.030]), a)
        t.update(man(PLACED + [0, 0, 0.019]), a)  # 11 mm in one step: 0.22 m/s, a quick set-down
        self.assertIsNone(t.breach)
        t = Tracker(HZ)  # the head swinging down on its own falls too
        t.update(man(START + [0, 0, 0.05], th=0.0), a)
        t.update(man(START + [0, 0, 0.05], th=-10.0), a)  # its underside drops 17 mm in one step
        self.assertEqual(t.breach, "head_fell")

    def test_off_the_table(self):
        t = Tracker(HZ)
        t.update(man([0.60, -0.085, L.TOP - 0.011]), snap())
        self.assertEqual(t.breach, "off_the_table")

    def test_a_breach_is_never_undone(self):
        f = Film()
        f.go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(20, o=START + [0, 0, 0.09])  # too high
        f2 = procedure()
        f2.frames = f.frames + f2.frames[70:]
        r, t = play(f2)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breach, "lifted_too_high")
        self.assertFalse(r.success)


class Exploits(unittest.TestCase):
    def test_one_arm_lifts_the_trunk_and_the_head_hangs(self):
        # The right gripper alone lifts the trunk 50 mm: the head swings down on the free neck (to about 32 deg while
        # its crown still drags on the mat, 40 deg at the stop).
        t = Tracker(HZ)
        for i in range(1, 41):
            dz = 0.05 * i / 40
            t.update(man(START + [0, 0, dz], th=head_resting_theta(dz)), snap(right_tcp=START + GRIP_R + [0, 0, dz],
                                                                                 right_open=TRUNK_SHUT))
        self.assertIn(t.breach, ("trunk_lifted_alone", "neck_bent"))
        self.assertEqual(t.last.score, 0.0)

    def test_one_arm_head_first_then_trunk(self):
        # The head lifted over the stretcher's edge first (the trunk tilted about its far end, which rests on the mat;
        # the neck bent only about 7 deg), then the trunk: the head is up alone, so 0 whatever follows.
        f = Film()
        f.go(20, L=START + GRIP_L).go(20, Lo=HEAD_SHUT)
        a = math.asin(0.031 / 0.24)
        f.go(30, o=START + [0, 0, 0.12 * math.sin(a)], th=-math.degrees(a), pitch=math.degrees(a),
             L=START + GRIP_L + [0, 0, 0.031])
        r, t = play(f)
        self.assertEqual(t.breach, "head_lifted_alone")
        self.assertLess(t.worst_neck, 8.0)
        f2 = procedure()
        f2.frames = f.frames + f2.frames[1:]
        r, t = play(f2)
        self.assertEqual(r.score, 0.0)

    def test_one_body_eased_over_the_stretcher_edge_alone(self):
        # The manikin lies on the table turned 25 deg, head end toward the stretcher; one gripper raises the head
        # (the trunk tilts about its far end, which stays down) until the head reaches over the stretcher's edge just
        # above its top. Part of the head is then over the stretcher, but its middle is still over the table, so it
        # is 31 mm up, lifted alone: 0. (Counted from the highest surface under any part it would not be lifted.)
        t = Tracker(HZ)
        a = snap()
        base = np.array([0.232, -0.085, L.TOP])
        for dz in np.linspace(0.0, 0.031, 20):
            ang = math.asin(dz / 0.24)
            m = man(base + [0, 0, 0.12 * math.sin(ang)], th=-math.degrees(ang), yaw=-25.0, pitch=math.degrees(ang))
            t.update(m, a)
        x0, x1, _, _ = S.footprint(m, "head")
        self.assertGreater(x1, L.STRETCHER_RECT[0] + 0.02)  # well over the edge ...
        self.assertLess((x0 + x1) / 2, L.STRETCHER_RECT[0])  # ... with its middle over the table
        self.assertAlmostEqual(S.low(m, "head"), 0.031, delta=0.002)
        self.assertLessEqual(S.lift(m, "trunk"), L.LIFTED)
        self.assertEqual(t.breach, "head_lifted_alone")

    def test_grip_at_the_neck(self):
        # One gripper on the neck bar lifts both bodies a little: the neck rule fires as soon as either is lifted.
        n = S.neck_centre(man())
        f = Film().go(20, L=n + [0, 0, 0.01]).go(20, Lo=0.2)
        f.go(20, o=START + [0, 0, 0.02], th=-25.0, L=n + [0, 0, 0.03])
        r, t = play(f)
        self.assertEqual(t.breach, "held_by_the_neck")
        self.assertEqual(r.score, 0.0)

    def test_drop_onto_the_stretcher(self):
        f = procedure(place=PLACED + [0, 0, 0.03], open_end=False)
        f.go(1, o=PLACED + [0, 0, 0.012], Lo=OPEN, Ro=OPEN).go(1, o=PLACED).hold(20)
        r, t = play(f)
        self.assertIn(t.breach, ("trunk_fell", "head_fell"))
        self.assertEqual(r.score, 0.0)

    def test_slid_against_the_stretcher_without_lifting(self):
        # Pushed along the mat and table up to the stretcher's side: it cannot climb the 30 mm step; never lifted.
        f = Film().go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(60, o=[L.STRETCHER_RECT[0] - 0.036, START[1], L.TOP])
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.lifted_together)

    def test_straight_at_the_end_does_not_hide_a_bend_on_the_way(self):
        r, t = play(procedure(carry_th=12.0, final_th=0.0))
        self.assertEqual(r.score, 0.5)
        self.assertAlmostEqual(t.worst_neck, 12.0)

    def test_placed_reversed_scores_quarter_at_most(self):
        # Turned end for end (head at the stretcher's foot): not 'on the stretcher' (the head is not on its zone).
        f = Film().go(20, L=START + GRIP_L, R=START + GRIP_R).go(20, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(30, o=START + [0, 0, 0.05])
        f.go(30, o=[L.STRETCHER_X, 0.015, L.TOP + 0.08], yaw=180.0, L=[0.30, -0.18, L.TOP + 0.10],
             R=[0.30, 0.0, L.TOP + 0.10])
        f.go(30, o=[L.STRETCHER_X, 0.015, L.TOP + L.STRETCHER_TOP]).go(20, Lo=OPEN, Ro=OPEN).hold(20)
        r, t = play(f)
        self.assertIn(r.score, (0.0, 0.25))
        self.assertFalse(r.details["on_stretcher"])


class Robustness(unittest.TestCase):
    def test_arms_unavailable(self):
        f = procedure()
        r, t = play(f, arms=False)
        self.assertEqual(r.score, 1.0)  # 'both grippers let go' taken as met
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertIsNone(r.details["left_clear"])
        t = Tracker(HZ)  # the neck rule is not applied on such a step
        m = man(START + [0, 0, 0.03])
        t.update(m, None)
        self.assertIsNone(t.breach)
        t.update(m, snap(left_tcp=S.neck_centre(m)))  # ... and applies again once the arms are back
        self.assertEqual(t.breach, "held_by_the_neck")

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=1.0),
               "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3), opening=1.0)}
        r = t.update(man(START + [0, 0, 0.03]), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertIsNone(t.breach)
        self.assertEqual(r.score, 0.25)
        self.assertEqual(t.update(man(START + [0, 0, 0.03]), {"left": None}).score, 0.25)
        self.assertEqual(t.arms_unavailable, 2)

    def test_offline_feed_reaches_the_scorer(self):
        m = man(START + [0, 0, 0.03])
        rigstate.offline_feed({"left": {"tcp": S.neck_centre(m), "tips": S.neck_centre(m) - [0, 0, TIPS],
                                        "opening": 0.4},
                               "right": {"tcp": READY_R, "tips": READY_R - [0, 0, TIPS], "opening": 1.0}})
        t = Tracker(HZ)
        t.update(m, rigstate.arms())
        self.assertEqual(t.breach, "held_by_the_neck")

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = procedure()
        _, t = play(f)
        before = (t.steps, t.worst_neck, t.breach, t.lifted_together, t.prev[0])
        last = f.frames[-1]
        arms = snap(last["L"], 1.0, last["R"], 1.0)
        good = man(last["o"])
        bads = []
        for key in ("origin", "lin_vel", "ang_vel"):
            b = man(last["o"])
            v = np.array(getattr(b, key), float)
            v[0] = np.nan
            setattr(b, key, v)
            bads.append(b)
        b = man(last["o"])
        b.joints = {"neck": float("inf")}
        bads.append(b)
        b = man(last["o"])
        b.quat_xyzw = np.zeros(4)
        bads.append(b)
        b = man(last["o"])
        b.links = {"neck": np.r_[b.links["neck"][:3], 0, 0, 0, 0]}
        bads.append(b)
        b = man(last["o"])
        b.links = {}
        bads.append(b)
        b = man(last["o"])
        b.joint_vel = {"neck": float("nan")}
        bads.append(b)
        for b in bads:
            r = t.update(b, arms)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual((t.worst_neck, t.breach, t.lifted_together, t.prev[0]), before[1:])
        self.assertEqual(t.steps, before[0] + len(bads))
        r = t.update(good, arms)  # the next good state carries on (no false fall over the gap)
        self.assertEqual(r.score, 1.0)
        self.assertIsNone(t.breach)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(3)
        t = Tracker(HZ)
        for _ in range(300):
            o = START + rng.uniform([-0.4, -0.6, -0.2], [0.6, 0.6, 0.4])
            m = man(o, th=rng.uniform(-60, 60), yaw=rng.uniform(-180, 180), roll=rng.uniform(-180, 180),
                    pitch=rng.uniform(-90, 90))
            r = t.update(m, snap(rng.uniform(-1, 1, 3), rng.uniform(0, 1), rng.uniform(-1, 1, 3), rng.uniform(0, 1)))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Resets(unittest.TestCase):
    def breached(self):
        f = procedure(carry_th=25.0)
        _, t = play(f)
        self.assertEqual(t.breach, "neck_bent")
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(man(), snap())  # the manikin jumps back onto the mat in one step: a new episode
        self.assertIsNone(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.worst_neck, 0.0)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(man(START + [0.03, -0.02, 0]), snap())
        self.assertIsNone(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        o = t.prev[1]
        f = Film(o=o, L=o + GRIP_L, R=o + GRIP_R, Lo=HEAD_SHUT, Ro=TRUNK_SHUT)
        f.carry(40, o=o + [0, 0, 0.03]).carry(60, o=START + [0, 0, 0.03]).carry(40, o=START).hold(10)
        play(f, t)
        self.assertEqual(t.breach, "neck_bent")  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(man(START + [0.15, 0.2, 0]), snap())  # far, not onto the start
        self.assertEqual(t.breach, "neck_bent")

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertIsNone(t.breach)
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the manikin as mechkit builds it, the hooks, the
    reset hooks and the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.mk = importlib.import_module("mechkit")

    def kit_state(self, origin, th=0.0, yaw=0.0):
        spec = self.task.MANIKIN
        q = self.mk.yaw_quat(math.radians(yaw))
        return self.mk.state_from_poses(spec, (*origin, *q), joints={"neck": math.radians(th)}, source="spec")

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        spec = self.task.MANIKIN
        self.assertEqual(spec.kind, "articulation")
        self.assertFalse(spec.fixed_base)
        self.assertEqual(spec.body_names, ("base", "neck"))
        j = spec.link("neck")
        self.assertEqual((j.kind, j.axis, j.lower, j.upper), ("revolute", "X", -L.NECK_LIMIT, L.NECK_LIMIT))
        self.assertAlmostEqual(spec.body_mass("base") + spec.body_mass("neck"), 0.29)
        # densities like a hollow plastic manikin (0.3-0.4 g/cm3) for both bodies
        for body in ("base", "neck"):
            vol = sum(s.volume() for s in spec.body_shapes(body))
            self.assertTrue(300 <= spec.body_mass(body) / vol <= 400, body)

    def test_scorer_geometry_matches_mechkit(self):
        # The scorer computes each body's boxes from layout; mechkit from the declaration: they must agree.
        for origin, th, yaw in ((L.MANIKIN_ORIGIN, 0.0, 0.0), ((0.30, 0.0, L.TOP + 0.05), 17.0, 0.0),
                                ((0.30, 0.0, L.TOP + 0.05), -30.0, 25.0), (tuple(PLACED), 2.0, -8.0)):
            st = self.kit_state(origin, th, yaw)
            for body, kit in (("trunk", "base"), ("head", "neck")):
                lo, hi = st.link_bounds(kit)
                c = S.corners(st, body)
                np.testing.assert_allclose(c.min(axis=0), lo, atol=1e-9)
                np.testing.assert_allclose(c.max(axis=0), hi, atol=1e-9)
            np.testing.assert_allclose(S.head_centre(st), st.link_to_world("neck", L.HEAD_CENTER), atol=1e-9)
            self.assertAlmostEqual(S.neck_deg(st), abs(th), places=9)

    def test_hooks_on_mechkit_states_and_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.mk.offline_reset()
        st = self.kit_state(L.MANIKIN_ORIGIN)
        snap()
        self.assertFalse(self.task.success({"manikin": st}))
        self.assertEqual(self.task.score({"manikin": st}), 0.0)
        bent = self.kit_state(L.MANIKIN_ORIGIN, th=25.0)
        snap()
        self.task.success({"manikin": bent})
        self.assertEqual(tr.breach, "neck_bent")
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertIsNone(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success({"manikin": st}))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.task.success({"manikin": bent})
        self.assertEqual(tr.breach, "neck_bent")
        self.mk.offline_reset()  # mechkit's reset counter: also a new trial at the next success()
        snap()
        self.assertFalse(self.task.success({"manikin": st}))
        self.assertIsNone(tr.breach)
        self.assertEqual(tr.steps, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success({"manikin": self.kit_state(L.MANIKIN_ORIGIN)}))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        self.assertEqual([w.duration for w in left], [w.duration for w in right])  # one clock: up, across, down together
        self.assertEqual([w.finger == self.task.OPEN for w in left], [w.finger == self.task.OPEN for w in right])
        for wps in (left, right):  # every gripper open/close takes >= 1.5 s after a >= 0.5 s settle
            for prev, before, w in zip(wps, wps[1:], wps[2:]):
                if w.finger != before.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
                    self.assertEqual(before.pos, w.pos)
                    self.assertGreaterEqual(before.duration, 0.5, before.label)
            descents = [w for p, w in zip(wps, wps[1:]) if w.pos[2] < p.pos[2] - 1e-9 and w.pos[2] - L.TOP < 0.06]
            for w in descents:  # each final descent near the manikin or the stretcher is slow
                self.assertGreaterEqual(w.duration, 1.5, w.label)
        gap = left[4].pos[1] - right[4].pos[1]
        self.assertAlmostEqual(gap, 0.230, places=9)
        self.assertLessEqual(max(w.pos[2] for w in left + right) - L.TOP, 0.14 + 1e-9)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
