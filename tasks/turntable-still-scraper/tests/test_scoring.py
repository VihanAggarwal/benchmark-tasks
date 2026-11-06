"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic turntable angles, smoother poses
and arm snapshots (fed through rigstate.offline_feed where the hooks are tested), no simulator.
python3 -m unittest discover -s tests -v"""

import copy
import importlib
import json
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
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = L.PAD_TIPS
PHI = math.radians(L.SET_ANGLE_DEG)
YAW_SET = PHI - math.pi / 2  # the blade along the tangent at the contact point
AXIS = np.array(L.AXIS)
OPEN = 1.0
HANDLE_OPEN = L.HANDLE[1] / L.FULL_GAP  # closed on the 32 mm handle (the simulator's reading)
PEG_OPEN = 2 * L.PEG_R / L.FULL_GAP  # closed on a peg
READY = {"left": np.array([0.3234, 0.305, L.TOP + 0.2606]), "right": np.array([0.3234, -0.305, L.TOP + 0.2606])}


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    """Quaternion product a * b (xyzw): turn by b, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def smoother(origin, quat=None, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like smoother (as mechkit gives it: with its frame's origin)."""
    q = about("z", math.degrees(YAW_SET)) if quat is None else np.asarray(quat, float)
    o = np.asarray(origin, float)
    r = S.rotation(q)
    pts = S._CORNERS @ r.T + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(pos=(lo + hi) / 2, quat_xyzw=q, lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=lo, high=hi, origin=o)


def set_origin(gap=0.001, bottom=0.0025, along=0.0, phi=PHI):
    """Where the smoother's origin is when its blade's face is ``gap`` off the side at angle ``phi``, its bottom
    ``bottom`` above the plate, its middle ``along`` along the tangent from the contact point."""
    rm = L.CAKE_R + gap + L.BLADE[1] / 2
    t = np.array([-math.sin(phi), math.cos(phi)])
    xy = AXIS + rm * np.array([math.cos(phi), math.sin(phi)]) + along * t
    return np.array([xy[0], xy[1], L.TOP + L.PLATE_TOP + bottom])


def at_set(gap=0.001, bottom=0.0025, along=0.0, phi=PHI, quat=None, **kw):
    q = about("z", math.degrees(phi - math.pi / 2)) if quat is None else quat
    return smoother(set_origin(gap, bottom, along, phi), q, **kw)


def in_stand(**kw):
    return smoother(L.SMOOTHER_START, about("z", 0.0), **kw)


def arm(tcp, opening=OPEN):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


def holding(s, opening=HANDLE_OPEN):
    """A straight-down gripper closed on the smoother's handle (its tips 9 mm above the handle's underside)."""
    r, o = S.pose(s)
    tips = o + r @ np.array([0.0, 0.0, L.BLADE[2] + L.GRIP_UP])
    return arm(tips + np.array([0.0, 0.0, TIPS]), opening)


def on_peg(theta, k=2, opening=PEG_OPEN):
    """A straight-down gripper closed on peg ``k`` (the plate turned ``theta``)."""
    c = S.peg_centres(theta)[k]
    return arm((c[0], c[1], L.TOP + L.PLATE_TOP + 0.007 + TIPS), opening)


def arms(left=None, right=None):
    return {"left": left if left is not None else arm(READY["left"]),
            "right": right if right is not None else arm(READY["right"])}


class Film:
    """A trial, one control step per frame: (turn (angle, rate), smoother, arms)."""

    def __init__(self):
        self.frames = []
        self.theta = 0.0

    def add(self, n, s, a, theta=None):
        th = self.theta if theta is None else theta
        for _ in range(n):
            self.frames.append(((th, 0.0), s, a))
        self.theta = th
        return self

    def turn(self, deg, n, s, holder="left", k=2, start=None):
        """Turn the plate ``deg`` degrees in ``n`` equal steps with the smoother ``s`` held where it is and the
        other gripper on peg ``k``."""
        a0 = self.theta if start is None else start
        rate = math.radians(deg) / (n * DT)
        for i in range(1, n + 1):
            th = a0 + math.radians(deg) * i / n
            peg = on_peg(th, k)
            hold = holding(s)
            a = arms(left=hold, right=peg) if holder == "left" else arms(left=peg, right=hold)
            self.frames.append(((th, rate), s, a))
        self.theta = a0 + math.radians(deg)
        return self


def play(film, tracker=None, frames=None):
    t = tracker or Tracker(HZ)
    r = None
    for turn, s, a in (film.frames if frames is None else frames):
        r = t.update(turn, s, a)
    return r, t


def finish(f, n=25):
    """Back the smoother 20 mm out and lift both arms clear."""
    away = at_set(gap=0.021)
    return f.add(n, away, arms(left=holding(away), right=arm(READY["right"])))


def procedure(deg=344.0, steps=400, finish_steps=25, stand=True):
    """The whole job: smoother in place, a turn of ``deg`` under it, then the finish. ``stand``: it starts with the
    smoother in its stand (False: already at the cake, so that no teleport back to the start can happen)."""
    s = at_set()
    f = Film().add(5, in_stand(), arms()) if stand else Film()
    f.add(10, s, arms(left=holding(s), right=on_peg(0.0)))
    f.turn(deg, steps, s)
    f.add(5, s, arms(left=holding(s), right=on_peg(f.theta)))
    if finish_steps:
        finish(f, finish_steps)
    return f


def angled(angle_deg, gap=0.0005, phi=PHI, bottom=0.0025):
    """A smoother whose blade is turned ``angle_deg`` from the side's tangent, one vertical edge (its end at -50 mm
    along it) ``gap`` off the side at angle ``phi``: a decorator's angled scraper, edge on."""
    yaw = phi - math.pi / 2 + math.radians(angle_deg)
    t = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2)])
    if np.dot(n, np.array([math.cos(phi), math.sin(phi)])) > 0:
        n = -n  # the face that looks at the axis
    corner = AXIS + (L.CAKE_R + gap) * np.array([math.cos(phi), math.sin(phi)])
    centre = corner + L.BLADE[0] / 2 * t - n * L.BLADE[1] / 2
    return smoother((centre[0], centre[1], L.TOP + L.PLATE_TOP + bottom), about("z", math.degrees(yaw)))


def twisted(psi_deg, bottom=0.0025):
    """The set smoother turned ``psi_deg`` about the vertical through its blade's centre."""
    s0 = at_set(bottom=bottom)
    c = S.blade_centre(s0)
    q = about("z", math.degrees(YAW_SET) + psi_deg)
    r = S.rotation(q)
    return smoother(c - r @ S.BLADE_C, q)


# -- geometry -------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_contact_gap_and_along(self):
        for gap in (-0.001, 0.0, 0.001, 0.003):
            c = S.blade_contact(at_set(gap=gap))
            self.assertAlmostEqual(c.gap, gap, places=6)
            self.assertAlmostEqual(c.along, 0.0, places=6)
            self.assertAlmostEqual(c.face_deg, 0.0, places=4)
            self.assertAlmostEqual(c.phi, PHI, places=6)
            np.testing.assert_allclose(c.point, AXIS + L.CAKE_R * np.array([math.cos(PHI), math.sin(PHI)]), atol=1e-9)
        c = S.blade_contact(at_set(along=0.045))  # the side touching near the blade's end: still the real gap
        self.assertAlmostEqual(abs(c.along), 0.045, places=6)
        self.assertAlmostEqual(c.gap, 0.001, places=6)

    def test_angled_blade_edge_on(self):
        # the real gap and angle for a blade angled to the side, its vertical edge 0.5 mm off it
        for deg in (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0):
            c = S.blade_contact(angled(deg))
            self.assertAlmostEqual(c.gap, 0.0005, places=5, msg=deg)
            self.assertAlmostEqual(c.face_deg, deg, places=3, msg=deg)
            self.assertAlmostEqual(c.phi, PHI, places=3, msg=deg)

    def test_either_face_of_the_blade(self):
        s = at_set(quat=about("z", math.degrees(YAW_SET) + 180))
        c = S.blade_contact(s)
        self.assertAlmostEqual(c.gap, 0.001, places=6)
        r, _ = play(Film().add(30, s, arms(left=holding(s), right=on_peg(0.0))))
        self.assertEqual(r.score, 0.25)
        self.assertAlmostEqual(S.half_turn(S.heading(s) - S.heading(at_set())), 0.0)  # the same heading

    def test_tilt_and_bottom(self):
        s = at_set(quat=qmul(about("z", math.degrees(YAW_SET)), about("x", 4.0)))
        self.assertAlmostEqual(S.tilt_deg(s.quat_xyzw), 4.0, places=4)
        self.assertAlmostEqual(S.blade_bottom(at_set(bottom=0.004)), 0.004, places=6)
        upside_down = smoother(set_origin(), qmul(about("z", math.degrees(YAW_SET)), about("x", 180)))
        self.assertAlmostEqual(S.tilt_deg(upside_down.quat_xyzw), 180.0, places=4)

    def test_sectors_and_mark(self):
        cov = np.zeros(L.N_SECTORS, bool)
        S.mark(cov, 10.2, 12.7)
        self.assertEqual(sorted(np.flatnonzero(cov)), [10, 11, 12])
        cov[:] = False
        S.mark(cov, -1.5, 0.5)  # wraps
        self.assertEqual(sorted(np.flatnonzero(cov)), [0, 358, 359])
        cov[:] = False
        S.mark(cov, 0.0, 400.0)
        self.assertTrue(cov.all())

    def test_one_turn_of_d_passes_d_plus_6(self):
        for deg, want in ((4.0, (8, 10)), (10.0, (16, 18)), (100.0, (106, 108)), (200.0, (206, 208)),
                          (334.0, (340, 342))):
            s = at_set()
            f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
            f.turn(deg, int(deg * 2), s)
            _, t = play(f)
            self.assertTrue(want[0] <= t.n_covered <= want[1], (deg, t.n_covered))

    def test_finger_zone(self):
        a = arm((0.30, -0.10, L.TOP + 0.10), opening=50 / 93.9)
        self.assertAlmostEqual(S.zone_radius(a), math.hypot(0.025 + 0.010, 0.020), places=3)
        pts = S.finger_zone(a)
        self.assertAlmostEqual(float(pts[:, 2].min()), L.TOP + 0.10 - TIPS, places=6)
        self.assertAlmostEqual(float(pts[:, 2].max()), L.TOP + 0.10 - TIPS + L.FINGER_LEN, places=6)
        # straight down over the cake's top, tips 10 mm above it: 10 mm away
        over = arm((*AXIS, L.TOP + L.CAKE_TOP + 0.010 + TIPS), opening=0.2)
        self.assertAlmostEqual(S.zone_cake_gap(over), 0.010, places=4)
        # beside the side, tips at mid height: the zone's edge decides
        r = S.zone_radius(arm((0, 0, 0), opening=0.2))
        beside = arm((AXIS[0] + L.CAKE_R + r + 0.008, AXIS[1], L.TOP + 0.04 + TIPS), opening=0.2)
        self.assertAlmostEqual(S.zone_cake_gap(beside), 0.008, delta=0.0005)
        # closed on a peg, or open round it: on the peg (it is turning the turntable)
        self.assertLess(S.zone_peg_gap(on_peg(0.0), 0.0), 0.0)
        self.assertLess(S.zone_peg_gap(on_peg(0.0, opening=0.53), 0.0), 0.0)

    def test_holder(self):
        s = at_set()
        self.assertEqual(S.holder(arms(left=holding(s)), s), "left")
        self.assertEqual(S.holder(arms(right=holding(s)), s), "right")
        self.assertIsNone(S.holder(arms(left=holding(s, opening=0.51)), s))  # more than half open
        self.assertEqual(S.holder(arms(left=holding(s, opening=0.5)), s), "left")
        h = holding(s)
        top = L.HANDLE[2] - L.GRIP_UP  # the handle's top above the finger tips
        up = arm(h.tcp + [0, 0, top + 0.0099])  # tips 9.9 mm over the handle's top: still at it
        self.assertEqual(S.holder(arms(left=rigstate.Arm(up.tcp, up.tips, HANDLE_OPEN)), s), "left")
        up = arm(h.tcp + [0, 0, top + 0.0102])
        self.assertIsNone(S.holder(arms(left=rigstate.Arm(up.tcp, up.tips, HANDLE_OPEN)), s))

    def test_on_top_and_distances(self):
        flat = smoother((AXIS[0], AXIS[1], L.TOP + L.CAKE_TOP + 0.003), qmul(about("z", 0), about("x", 90)))
        self.assertTrue(S.on_top(flat))  # lying on the top (its 6 mm face 3 mm up, so its lowest face at 0)
        self.assertTrue(S.on_top(smoother((AXIS[0], AXIS[1], L.TOP + L.CAKE_TOP + 0.0029))))  # standing on it
        self.assertFalse(S.on_top(smoother((AXIS[0], AXIS[1], L.TOP + L.CAKE_TOP + 0.0031))))
        self.assertFalse(S.on_top(at_set(gap=-0.001)))  # pressed 1 mm into the side is not on the top
        self.assertAlmostEqual(S.smoother_cake_gap(at_set(gap=0.010)), 0.010, places=4)
        self.assertGreater(S.lowest(in_stand()), L.TOP)


# -- tiers --------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().add(10, in_stand(), arms()))
        self.assertEqual((r.score, r.success), (0.0, False))

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(L.FULL <= t.n_covered <= 352, t.n_covered)
        self.assertIsNotNone(t.first_full_step)

    def test_the_full_score_needs_340_sectors(self):
        r, t = play(procedure(deg=335.0, steps=400))
        self.assertGreaterEqual(t.n_covered, L.FULL)
        self.assertEqual(r.score, 1.0)
        r, t = play(procedure(deg=331.0, steps=400))
        self.assertLess(t.n_covered, L.FULL)
        self.assertEqual(r.score, 0.75)

    def test_full_coverage_without_the_finish_scores_075(self):
        f = procedure(finish_steps=0)
        r, t = play(f)
        self.assertGreaterEqual(t.n_covered, L.FULL)
        self.assertEqual((r.score, r.success), (0.75, False))  # the smoother still at the cake

    def test_finish_boundaries(self):
        f = procedure(finish_steps=0)
        _, base = play(f)
        cases = []
        for gap, ok in ((0.0099, False), (0.0101, True)):
            s = at_set(gap=gap)
            cases.append((s, arms(left=holding(s)), (f.theta, 0.0), ok))
        s = at_set(gap=0.02)
        peg_c = S.peg_centres(f.theta)[0]
        for clear, ok in ((0.0199, False), (0.0201, True)):  # tips straight over a peg, this far above its top
            cases.append((s, arms(left=holding(s), right=arm((*peg_c, L.TOP + L.PEG_TOP + clear + TIPS), PEG_OPEN)),
                          (f.theta, 0.0), ok))
        for rate, ok in ((0.049, True), (0.051, False)):
            cases.append((s, arms(left=holding(s)), (f.theta, rate), ok))
        for s, a, turn, ok in cases:
            t = copy.deepcopy(base)
            for _ in range(20):
                r = t.update(turn, s, a)
            self.assertEqual(r.score, 1.0 if ok else 0.75, (turn, ok))

    def test_the_finish_must_hold_a_second(self):
        f = procedure(finish_steps=0)
        away = at_set(gap=0.0105)
        a = arms(left=holding(away))
        for n, want in ((1, 0.75), (19, 0.75), (20, 1.0)):
            r, t = play(Film().add(n, away, a, theta=f.theta), play(f)[1])
            self.assertEqual(r.score, want, n)
            self.assertTrue(r.success)  # what task.success() reports; the simulator holds it 1 s
        # the reviewer's cut: the turning gripper still on a peg and the plate turning until the very last step
        _, t = play(f)
        play(Film().add(30, at_set(gap=0.0105), arms(left=holding(away), right=on_peg(f.theta))), t)
        r = t.update((f.theta, 0.049), away, a)
        self.assertEqual(r.score, 0.75)

    def test_coverage_tiers(self):
        for deg, want in ((300.0, 0.75), (266.0, 0.75), (255.0, 0.5), (176.0, 0.5), (170.0, 0.25), (10.0, 0.25)):
            r, t = play(procedure(deg=deg, steps=max(20, int(deg * 1.2))))
            self.assertEqual(r.score, want, (deg, t.n_covered))

    def test_set_for_a_second_scores_quarter(self):
        s = at_set()
        away = at_set(gap=0.02)
        r, _ = play(Film().add(20, s, arms(left=holding(s))).add(10, away, arms()))
        self.assertEqual(r.score, 0.25)
        r, _ = play(Film().add(19, s, arms(left=holding(s))).add(10, away, arms()))
        self.assertEqual(r.score, 0.0)  # 0.95 s is not 1 s
        r, _ = play(Film().add(10, s, arms(left=holding(s))).add(1, away, arms())
                    .add(10, s, arms(left=holding(s))))
        self.assertEqual(r.score, 0.0)  # two half seconds are not a second

    def test_set_boundaries(self):
        def scored(s, a=None, n=25):
            return play(Film().add(n, s, a or arms(left=holding(s))))[0].score

        for kw, ok in (({"gap": -0.00099}, True), ({"gap": 0.00299}, True), ({"gap": -0.00101}, False),
                       ({"gap": 0.00301}, False), ({"bottom": -0.00099}, True), ({"bottom": 0.00499}, True),
                       ({"bottom": -0.00101}, False), ({"bottom": 0.00501}, False), ({"along": 0.045}, True)):
            self.assertEqual(scored(at_set(**kw)), 0.25 if ok else 0.0, kw)
        for deg, ok in ((4.9, True), (5.1, False)):
            # rolled in its own plane (about its thickness): the face and its gap unchanged, one bottom corner
            # 4.3-4.4 mm lower, still 0.4-0.5 mm above the plate; only the tilt decides
            s = smoother(set_origin(bottom=0.0048), qmul(about("z", math.degrees(YAW_SET)), about("y", deg)))
            self.assertAlmostEqual(S.blade_contact(s).gap, 0.001, places=5)
            self.assertTrue(0.0 < S.blade_bottom(s) < 0.001)
            self.assertEqual(scored(s), 0.25 if ok else 0.0, deg)
        for deg, ok in ((44.9, True), (45.1, False), (20.0, True)):  # angled to the side, edge on
            self.assertEqual(scored(angled(deg)), 0.25 if ok else 0.0, deg)
        s = at_set()
        self.assertEqual(scored(s, arms(left=holding(s, opening=0.51))), 0.0)  # not holding it
        self.assertEqual(scored(s, arms(right=holding(s))), 0.25)  # either gripper may hold it

    def test_stillness(self):
        t_dir = np.array([-math.sin(PHI), math.cos(PHI), 0.0])
        for speed, want in ((0.0098, 0.25), (0.0102, 0.0)):  # sliding along the side (its own length) for 1 s
            f = Film()
            for i in range(25):
                s = smoother(set_origin() + t_dir * speed * DT * i)
                f.add(1, s, arms(left=holding(s)))
            self.assertEqual(play(f)[0].score, want, speed)
        for rate, want in ((2.9, 0.25), (3.1, 0.0)):  # turning about the vertical through its centre for 1 s
            f = Film()
            for i in range(25):
                s = twisted(rate * DT * i)
                f.add(1, s, arms(left=holding(s)))
            self.assertEqual(play(f)[0].score, want, rate)

    def test_out_of_its_window_until_taken_away(self):
        # the reviewer's case: 0 -> 11.5 mm -> 5 mm without being taken away is not set again
        t_dir = np.array([-math.sin(PHI), math.cos(PHI), 0.0])
        f = Film()
        for d in [min(0.0115, 0.00024 * i) for i in range(60)] + [max(0.005, 0.0115 - 0.00024 * i) for i in range(60)]:
            s = smoother(set_origin() + t_dir * d)
            f.add(1, s, arms(left=holding(s)))
        _, t = play(f)
        self.assertLessEqual(t.max_drift, L.STILL_WINDOW + 1e-9)
        self.assertTrue(t.latched)
        self.assertFalse(t.was_set)
        back = smoother(set_origin() + t_dir * 0.005)
        for way_off, sets_again in ((at_set(gap=0.015), False), (at_set(gap=0.0215), True),
                                    (at_set(bottom=0.0205), True), (back, True)):
            t2 = copy.deepcopy(t)
            hand = arms() if way_off is back else arms(left=holding(way_off))  # let go, or moved/lifted away
            play(Film().add(2, way_off, hand).add(25, back, arms(left=holding(back))), t2)
            self.assertEqual(t2.was_set, sets_again)
        # turned more than 4 deg from its heading when put in place: out of its window too
        f = Film()
        for i in range(60):
            s = twisted(min(4.5, 0.1 * i))  # 2 deg/s
            f.add(1, s, arms(left=holding(s)))
        _, t = play(f)
        self.assertTrue(t.latched)
        self.assertFalse(t.was_set)
        for deg, ok in ((3.9, True), (4.1, False)):
            f = Film()
            for i in range(80):
                s = twisted(min(deg, 0.1 * i))
                f.add(1, s, arms(left=holding(s)))
            self.assertEqual(play(f)[1].was_set, ok, deg)

    def test_a_turn_counts_only_while_set(self):
        s = at_set()
        f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
        f.turn(100.0, 200, s)
        far = at_set(gap=0.01)
        f.turn(100.0, 200, far)  # not set: nothing counts
        f.turn(100.0, 200, s)
        _, t = play(f)
        self.assertTrue(212 <= t.n_covered <= 216, t.n_covered)

    def test_a_turn_counts_only_while_the_other_gripper_drives_a_peg(self):
        s = at_set()
        r0 = S.zone_radius(arm((0, 0, 0), opening=PEG_OPEN))
        for case, want in (("ready", (0, 0)), ("4.9 mm", (106, 108)), ("5.1 mm", (0, 0)), ("holder", (0, 0))):
            f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
            for i in range(1, 201):
                th = math.radians(100.0) * i / 200
                c = S.peg_centres(th)[2]
                out = (c - AXIS) / np.linalg.norm(c - AXIS)
                right = {"ready": arm(READY["right"]),
                         "4.9 mm": arm((*(c + out * (L.PEG_R + r0 + 0.0049)), L.TOP + 0.03 + TIPS), PEG_OPEN),
                         "5.1 mm": arm((*(c + out * (L.PEG_R + r0 + 0.0051)), L.TOP + 0.03 + TIPS), PEG_OPEN),
                         "holder": arm(READY["right"])}[case]
                a = arms(left=holding(s), right=right)
                if case == "holder":  # the holding gripper itself is the one at a peg: it cannot turn and hold
                    a = arms(left=holding(s), right=holding(s))
                f.frames.append(((th, 0.5), s, a))
            _, t = play(f)
            self.assertTrue(want[0] <= t.n_covered <= want[1], (case, t.n_covered))

    def test_a_turn_reported_wrapped_counts_the_same(self):
        s = at_set()
        f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
        f.turn(344.0, 400, s)
        wrapped = Film()
        wrapped.frames = [((S.wrap(th), w), sm, a) for (th, w), sm, a in f.frames]
        _, t1 = play(f)
        _, t2 = play(wrapped)
        self.assertEqual(t1.n_covered, t2.n_covered)
        self.assertGreaterEqual(t2.n_covered, L.FULL)

    def test_turning_the_other_way_counts_too(self):
        s = at_set()
        f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
        f.turn(-100.0, 200, s)
        _, t = play(f)
        self.assertTrue(106 <= t.n_covered <= 108, t.n_covered)

    def test_an_angled_blade_counts(self):
        # a decorator's scraper at 30 deg to the side, its edge 0.5 mm off, while the other gripper turns 100 deg
        s = angled(30.0)
        f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
        f.turn(100.0, 200, s)
        _, t = play(f)
        self.assertTrue(106 <= t.n_covered <= 108, t.n_covered)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        turn, s, a = f.frames[-1]
        before = (t.steps, t.n_covered, dict(t.zeros), t.set_1s, t.finish_steps)
        for _ in range(3):
            self.assertEqual(t.result(turn, s, a).score, r.score)
        self.assertEqual((t.steps, t.n_covered, dict(t.zeros), t.set_1s, t.finish_steps), before)


# -- zero conditions -----------------------------------------------------------------------------------------------
class Zeros(unittest.TestCase):
    def ok_then(self, s, a, n=3):
        f = procedure(finish_steps=0)
        _, t = play(f)
        play(Film().add(n, s, a, theta=f.theta), t)
        r, _ = play(finish(Film(), 25), t)
        return r, t

    def test_fingers_near_the_cake(self):
        r0 = S.zone_radius(arm((0, 0, 0), opening=0.3))
        for clear, zero in ((0.0049, True), (0.0051, False)):
            a = arm((AXIS[0] + L.CAKE_R + r0 + clear, AXIS[1], L.TOP + 0.04 + TIPS), opening=0.3)
            s = at_set(gap=0.02)
            r, t = self.ok_then(s, arms(left=holding(s), right=a))
            self.assertEqual(r.score, 0.0 if zero else 1.0, clear)
            if zero:
                self.assertIn("right_fingers_at_cake", t.zeros)

    def test_fingers_over_the_top(self):
        s = at_set(gap=0.02)
        for clear, zero in ((0.0049, True), (0.0051, False)):
            a = arm((*AXIS, L.TOP + L.CAKE_TOP + clear + TIPS), opening=0.0)
            r, _ = self.ok_then(s, arms(left=a))
            self.assertEqual(r.score, 0.0 if zero else 1.0, clear)

    def test_smoother_on_the_top(self):
        for dz, zero in ((0.0029, True), (0.0031, False)):
            s = smoother((AXIS[0] + 0.02, AXIS[1], L.TOP + L.CAKE_TOP + dz))
            r, _ = self.ok_then(s, arms(left=arm(READY["left"])))
            self.assertEqual(r.score, 0.0 if zero else 1.0, dz)

    def test_smoother_off_the_table(self):
        for dz, zero in ((-0.0200, True), (-0.0199, False)):  # its lowest point 20 mm or more below: fell off
            s = smoother((0.30, 0.66, L.TOP + dz), qmul(about("z", 0), about("x", 90)))
            low = S.lowest(s) - L.TOP
            s = smoother(s.origin - [0, 0, low - dz], s.quat_xyzw)
            r, _ = self.ok_then(s, arms())
            self.assertEqual(r.score, 0.0 if zero else 1.0, dz)

    def test_a_zero_is_never_undone(self):
        s = smoother((AXIS[0], AXIS[1], L.TOP + L.CAKE_TOP))
        f = Film().add(3, s, arms())
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(stand=False), t)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertGreaterEqual(t.n_covered, L.FULL)


# -- exploits -------------------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_one_arm_moves_the_smoother_round_a_still_cake(self):
        f = Film()
        for phi_deg in range(0, 360, 10):  # set it all the way round, the turntable never turns
            s = at_set(phi=math.radians(phi_deg))
            f.add(25, s, arms(left=holding(s)))
            f.add(2, at_set(phi=math.radians(phi_deg), gap=0.02), arms())
        r, t = play(f)
        self.assertEqual(t.n_covered, 0)
        self.assertEqual(r.score, 0.25)

    def test_sliding_the_smoother_round_a_still_cake(self):
        f = Film()
        for i in range(400):  # slow (under 1 cm/s) slide round the side, never lifted
            phi = PHI + i * DT * 0.009 / L.CAKE_R
            s = at_set(phi=phi)
            f.add(1, s, arms(left=holding(s), right=on_peg(0.0)))
        r, t = play(f)
        self.assertEqual(t.n_covered, 0)
        self.assertLessEqual(r.score, 0.25)

    def test_smoother_left_unheld_against_the_cake_while_one_arm_turns(self):
        s = at_set()
        f = Film().add(5, s, arms())
        for i in range(1, 401):
            th = math.radians(344.0) * i / 400
            f.frames.append(((th, 0.7), s, arms(right=on_peg(th))))
        r, t = play(f)
        self.assertEqual((t.n_covered, r.score), (0, 0.0))

    def test_rocking_back_and_forth_over_a_quarter(self):
        s = at_set()
        f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
        for _ in range(8):
            f.turn(90.0, 100, s)
            f.turn(-90.0, 100, s)
        r, t = play(f)
        self.assertLessEqual(t.n_covered, 98)
        self.assertEqual(r.score, 0.25)

    def test_turning_by_pushing_the_cake(self):
        s = at_set()
        f = Film().add(5, s, arms(left=holding(s)))
        for i in range(1, 401):  # the right gripper's closed fingers pushed against the cake's side
            th = math.radians(344.0) * i / 400
            a = AXIS + (L.CAKE_R + 0.012) * np.array([math.cos(th - 2.0), math.sin(th - 2.0)])
            f.frames.append(((th, 0.7), s, arms(left=holding(s), right=arm((*a, L.TOP + 0.05), 0.0))))
        finish(f)
        r, t = play(f)
        self.assertEqual(t.n_covered, 0)  # not at a peg either
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_fingers_at_cake", t.zeros)

    def test_hovering_the_smoother_high_or_far_or_end_on(self):
        for s in (at_set(bottom=0.020), at_set(gap=0.010), angled(50.0)):
            f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
            f.turn(344.0, 400, s)
            r, t = play(finish(f))
            self.assertEqual((t.n_covered, r.score), (0, 0.0))

    def test_turn_first_then_set(self):
        f = Film().add(5, in_stand(), arms())
        for i in range(1, 401):
            th = math.radians(344.0) * i / 400
            f.frames.append(((th, 0.7), in_stand(), arms(right=on_peg(th))))
        f.theta = math.radians(344.0)
        s = at_set()
        f.add(30, s, arms(left=holding(s)))
        r, t = play(finish(f))
        self.assertEqual(t.n_covered, 0)
        self.assertEqual(r.score, 0.25)

    def test_dragging_the_cake_round_with_the_blade(self):
        # The blade pressed on the side drags the cake round by friction, sliding along its own length (its contact
        # point stays put) or swinging round with the side; then it backs off, goes back and drags again. One arm
        # alone (no gripper at a peg) and with the other gripper idling on a peg: the blade follows the turn, so
        # nothing passes it.
        t_dir = np.array([-math.sin(PHI), math.cos(PHI), 0.0])
        for swing in (False, True):
            for other in (False, True):
                f = Film()
                th = 0.0
                for _ in range(20):
                    for i in range(40):  # 9 mm in 2 s (4.5 mm/s), the cake turning with it
                        d = math.radians(8.6) * i / 39
                        s = at_set(gap=-0.0009, phi=PHI + d) if swing else \
                            smoother(set_origin(gap=-0.0009) + t_dir * d * L.CAKE_R)
                        f.frames.append(((th + d, 0.075), s,
                                         arms(left=holding(s), right=on_peg(th + d) if other else None)))
                    th += math.radians(8.6)
                    f.add(3, at_set(gap=0.03), arms(), theta=th)
                r, t = play(f)
                self.assertLessEqual(t.n_covered, 4 * 20, (swing, other))
                self.assertLessEqual(r.score, 0.25, (swing, other))

    def test_sliding_the_blade_against_the_turn_adds_nothing(self):
        # The blade slides along its length against the turn while the cake turns: only the turn counts.
        t_dir = np.array([-math.sin(PHI), math.cos(PHI), 0.0])
        f = Film().add(5, at_set(), arms(left=holding(at_set()), right=on_peg(0.0)))
        for i in range(1, 101):  # the cake turns 50 deg, the blade slides 9 mm back along the tangent
            s = smoother(set_origin() - t_dir * 0.009 * i / 100)
            th = math.radians(50.0) * i / 100
            f.frames.append(((th, 0.17), s, arms(left=holding(s), right=on_peg(th))))
        _, t = play(f)
        self.assertTrue(56 <= t.n_covered <= 58, t.n_covered)

    def test_a_tilted_or_upside_down_smoother_is_never_set(self):
        for q in (qmul(about("z", math.degrees(YAW_SET)), about("x", 12)),
                  qmul(about("z", math.degrees(YAW_SET)), about("x", 180))):
            s = smoother(set_origin(), q)
            f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
            f.turn(344.0, 400, s)
            r, t = play(finish(f))
            self.assertEqual(t.n_covered, 0)

    def test_one_arm_press_and_twist(self):
        # The reviewer's ratchet: the holding gripper presses the blade's bottom edge on the plate and twists 24 deg
        # about the blade's centre (which stays still), lifts 3 mm, twists back, presses again; friction under the
        # edge drags the plate round 8.6 deg a stroke. Its heading leaves its 4 deg window: never set again until
        # taken away; and no other gripper turns the plate. With the other gripper on a peg too, still nothing.
        for other in (False, True):
            f, th = Film(), 0.0
            for _ in range(40):
                for i in range(1, 13):  # press and twist anticlockwise, the plate dragged 8.6 deg
                    x = i / 12
                    sm = 3 * x * x - 2 * x ** 3
                    s = twisted(-12 + 24 * sm, bottom=0.0)
                    f.frames.append(((th + math.radians(8.6) * sm, 0.25), s,
                                     arms(left=holding(s), right=on_peg(th) if other else None)))
                th += math.radians(8.6)
                for i in range(1, 13):  # lift 3 mm, twist back
                    s = twisted(12 - 24 * i / 12, bottom=0.003)
                    f.frames.append(((th, 0.0), s, arms(left=holding(s), right=on_peg(th) if other else None)))
            r, t = play(finish(f))
            self.assertLessEqual(t.n_covered, 10, other)
            self.assertLessEqual(r.score, 0.25, other)

    def test_flick_and_coast(self):
        # The plate flicked to spin, then coasting while the smoother is set: no gripper is at a peg, so the coast
        # counts for nothing (one arm, or the other arm flicking and leaving the pegs).
        for omega0 in (1.0, 3.0, 6.0):
            f, th, w = Film(), 0.0, omega0
            s = at_set()
            f.add(5, s, arms(left=holding(s)))
            for _ in range(200):
                th = min(th + w * DT, L.PLATE_UPPER)
                f.frames.append(((th, w), s, arms(left=holding(s))))
                w *= math.exp(-DT / 3.0)  # even a free-spinning plate (3 s time constant)
            r, t = play(finish(f))
            self.assertEqual(t.n_covered, 0, omega0)
            self.assertEqual(r.score, 0.25, omega0)

    def test_resetting_the_blade_back_each_time(self):
        # The reviewer's fragment plan: 30 deg of turn per set, the blade re-set 30 deg back each time. Each set adds
        # only its own turn plus 6 deg of arc: 6 x 36 = 216 sectors, not the whole side.
        f, th = Film(), 0.0
        for k in range(6):
            phi = PHI - math.radians(30 * k)
            s = at_set(phi=phi)
            f.add(3, s, arms(left=holding(s), right=on_peg(th)))
            f.turn(30.0, 60, s)
            th = f.theta
            f.add(3, at_set(phi=phi, gap=0.03), arms())
        r, t = play(finish(f))
        self.assertLessEqual(t.n_covered, 6 * 38)
        self.assertEqual(r.score, 0.5)

    def test_rolling_the_blade_on_the_side(self):
        # The blade rocked (rolled) +-27 deg round the side while the plate creeps: its heading turns too fast and
        # leaves its window, so it is not set; the roll gains (almost) nothing.
        def rolled(psi):
            a = L.CAKE_R * (psi - PHI)
            u, tt = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
            c = AXIS + (L.CAKE_R + 0.001 + L.BLADE[1] / 2) * u - a * tt
            return smoother((c[0], c[1], L.TOP + L.PLATE_TOP + 0.0025), about("z", math.degrees(psi - math.pi / 2)))

        f = Film()
        for i in range(1000):
            th = math.radians(30.0) * i / 1000
            s = rolled(PHI + math.radians(27) * math.sin(2 * math.pi * i * DT / 12))
            f.frames.append(((th, 0.01), s, arms(left=holding(s), right=on_peg(th))))
        _, t = play(f)
        self.assertLessEqual(t.n_covered, 40)


# -- the arms and the state's health ------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_hold_and_turn_nothing(self):
        # arms that cannot be read hold nothing and turn nothing (no credit), and the finish is not met
        s = at_set()
        f = Film().add(5, s, None)
        f.frames += [((math.radians(344) * i / 400, 0.7), s, None) for i in range(1, 401)]
        f.theta = math.radians(344)
        f.add(25, at_set(gap=0.03), None)
        r, t = play(f)
        self.assertEqual((r.score, r.success, t.n_covered), (0.0, False, 0))
        self.assertEqual(t.arms_unavailable, 430)
        self.assertFalse(r.details["arms_available"])
        # readable while it turned, unreadable at the end: the finish is not met
        f = procedure(finish_steps=0)
        f.add(25, at_set(gap=0.03), None)
        r, t = play(f)
        self.assertGreaterEqual(t.n_covered, L.FULL)
        self.assertEqual(r.score, 0.75)

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.zeros(3), 0.0), "right": arm(READY["right"])}
        self.assertFalse(S.arms_valid(bad))
        self.assertFalse(S.arms_valid({"left": arm(READY["left"])}))
        self.assertFalse(S.arms_valid(None))

    def test_offline_feed_reaches_the_scorer(self):
        s = at_set()
        h = holding(s)
        rigstate.offline_feed({"left": {"tcp": h.tcp, "tips": h.tips, "opening": h.opening},
                               "right": {"tcp": READY["right"], "tips": READY["right"] - [0, 0, TIPS],
                                         "opening": 1.0}})
        self.assertEqual(S.holder(rigstate.arms(), s), "left")


class Robustness(unittest.TestCase):
    def test_sag_is_tolerated(self):
        # the held smoother sags ~1.8 mm down and ~0.2 mm outward from the commanded set pose (our estimate):
        # twice that still leaves it set (and none at all, as offline)
        for k in (0.0, 1.0, 2.0):
            s = at_set(gap=L.SET_GAP + k * 0.0002, bottom=L.SET_LIFT - k * 0.0018)
            f = Film().add(5, s, arms(left=holding(s), right=on_peg(0.0)))
            f.turn(344.0, 400, s)
            r, _ = play(finish(f))
            self.assertEqual(r.score, 1.0, k)

    def test_millimetre_and_degree_wobble_is_tolerated(self):
        # the held smoother wobbling up to 1 mm (gap and height), 2 mm along the side and 1.5 deg in heading
        f = Film()
        for i in range(450):
            th = math.radians(344.0) * min(1.0, i / 400)
            w = 2 * math.pi * i * DT
            s = at_set(gap=0.001 + 0.001 * math.sin(w / 3), bottom=0.0025 + 0.001 * math.sin(w / 2 + 1),
                       along=0.002 * math.sin(w / 5),
                       quat=about("z", math.degrees(YAW_SET) + 1.5 * math.sin(w / 7)))
            f.frames.append(((th, 0.7), s, arms(left=holding(s), right=on_peg(th))))
        f.theta = math.radians(344)
        r, t = play(finish(f))
        self.assertGreaterEqual(t.n_covered, L.FULL)
        self.assertEqual(r.score, 1.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = at_set()
        inf.pos = np.array([0.3, 0.0, np.inf])
        zero_q = at_set()
        zero_q.quat_xyzw = np.zeros(4)
        bad_origin = at_set()
        bad_origin.origin = np.array([np.nan, 0.0, 0.0])
        missing = SimpleNamespace(pos=np.zeros(3))
        f = procedure(finish_steps=0)
        for bad in (nan, inf, zero_q, bad_origin, missing, None):
            self.assertFalse(S.valid(bad))
            _, t = play(f)
            before = (t.n_covered, dict(t.zeros), t.set_1s, t.anchor is None, t.was_set, t.latched)
            a = arms(left=holding(at_set()))
            for turn, s in (((f.theta, 0.0), bad), ((math.nan, 0.0), at_set()), ((f.theta, math.inf), at_set()),
                            (None, at_set())):
                r = t.update(turn, s, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(turn, s, a).score, 0.0)
            self.assertEqual((t.n_covered, dict(t.zeros), t.set_1s, t.anchor is None, t.was_set, t.latched), before)
            self.assertEqual(t.bad_steps, 4)
            r, _ = play(finish(Film()), t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(300):
            o = rng.uniform([-0.5, -0.8, 0.0], [1.0, 0.8, 1.6])
            q = rng.normal(size=4)
            s = smoother(o, q / np.linalg.norm(q), vel=rng.normal(size=3), spin=rng.normal(size=3))
            a = {"left": arm(rng.uniform([-0.2, -0.6, 0.6], [0.6, 0.6, 1.2]), rng.uniform()),
                 "right": arm(rng.uniform([-0.2, -0.6, 0.6], [0.6, 0.6, 1.2]), rng.uniform())}
            r = t.update((rng.uniform(-10, 10), rng.uniform(-5, 5)), s, a if rng.uniform() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def zeroed(self):
        _, t = play(Film().add(3, smoother((AXIS[0], AXIS[1], L.TOP + L.CAKE_TOP)), arms()))
        _, t = play(procedure(deg=200.0, steps=250, finish_steps=0, stand=False), t)
        self.assertTrue(t.zero)
        self.assertGreater(t.n_covered, 0)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.zeroed()
        r = t.update((0.0, 0.0), in_stand(), arms())  # the smoother back in its stand, the plate at 0: new episode
        self.assertFalse(t.zero)
        self.assertEqual((t.steps, t.n_covered), (1, 0))
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(), t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.zeroed()
        jit = smoother(np.asarray(L.SMOOTHER_START) + [0.006, -0.005, 0.0], about("z", 0.5))
        t.update((math.radians(0.5), 0.0), jit, arms())
        self.assertFalse(t.zero)

    def test_gradual_return_does_not_reset(self):
        t = self.zeroed()
        _, o = S.pose(at_set())
        f = Film()
        for i in range(1, 81):
            p = o + (np.asarray(L.SMOOTHER_START) - o) * i / 80
            f.add(1, smoother(p, about("z", math.degrees(YAW_SET) * (1 - i / 80))), arms(), theta=0.0)
        f.add(10, in_stand(), arms(), theta=0.0)
        play(f, t)
        self.assertTrue(t.zero)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.zeroed()
        t.update((0.0, 0.0), smoother((0.40, 0.40, L.TOP)), arms())  # knocked far, not to the stand
        self.assertTrue(t.zero)
        t2 = self.zeroed()
        t2.update((math.radians(200.0), 0.0), in_stand(), arms())  # in the stand but the plate not back at 0
        self.assertTrue(t2.zero)

    def test_rigstate_reset_hook_resets_a_tracker(self):
        t = self.zeroed()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.zero)
            self.assertEqual((t.steps, t.n_covered), (0, 0))
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


# -- the offline stand-in for the turntable's joint ------------------------------------------------------------------
class OfflineModel(unittest.TestCase):
    def tips_arm(self, xy, z, opening):
        return arm((xy[0], xy[1], z + TIPS), opening)

    def grip(self, m, k=2, side="right", z=None):
        """Close a gripper on peg k from 50 mm open, over 30 steps."""
        z = L.TOP + L.PLATE_TOP + 0.007 if z is None else z
        c = S.peg_centres(m.theta)[k]
        for i in range(31):
            o = (0.050 - (0.050 - 0.012) * i / 30) / L.FULL_GAP
            m.update({side: self.tips_arm(c, z, o), ("left" if side == "right" else "right"): arm(READY["left"])})
        return c

    def test_grip_turn_release(self):
        m = offline_model.PlateModel()
        self.grip(m)
        self.assertIsNotNone(m.hold)
        z = L.TOP + L.PLATE_TOP + 0.007
        for i in range(1, 86):  # carry the peg 85 deg round its circle
            a = math.radians(L.PEG_ANGLES_DEG[2] + i)
            m.update({"right": self.tips_arm(AXIS + L.PEG_CIRCLE * np.array([math.cos(a), math.sin(a)]), z,
                                             0.012 / L.FULL_GAP), "left": arm(READY["left"])})
        self.assertAlmostEqual(math.degrees(m.theta), 85.0, places=6)
        self.assertGreater(m.omega, 0)
        c = S.peg_centres(m.theta)[2]
        for i in range(20):  # open
            m.update({"right": self.tips_arm(c, z, (0.012 + 0.002 * i) / L.FULL_GAP), "left": arm(READY["left"])})
        self.assertIsNone(m.hold)
        before = m.theta
        for i in range(30):  # moving away afterwards turns nothing
            m.update({"right": self.tips_arm(c + [0.0, -0.002 * i], z, 0.5), "left": arm(READY["left"])})
        self.assertEqual(m.theta, before)
        self.assertEqual(m.omega, 0.0)

    def test_no_grip_without_closing_on_a_peg(self):
        z = L.TOP + L.PLATE_TOP + 0.007
        for kw in ({"z": L.TOP + L.PEG_TOP + 0.002}, {"z": L.TOP + L.PLATE_TOP - 0.004}):
            m = offline_model.PlateModel()
            self.grip(m, **kw)
            self.assertIsNone(m.hold, kw)
        m = offline_model.PlateModel()  # closing 11 mm off the peg's centre
        c = S.peg_centres(0.0)[2] + [0.011, 0.0]
        for i in range(31):
            m.update({"right": self.tips_arm(c, z, (0.05 - 0.038 * i / 30) / L.FULL_GAP), "left": arm(READY["left"])})
        self.assertIsNone(m.hold)
        m = offline_model.PlateModel()  # a gripper already closed that moves onto the peg
        c = S.peg_centres(0.0)[2]
        for i in range(31):
            m.update({"right": self.tips_arm(c + [0.03 * (1 - i / 30), 0.0], z, 0.0), "left": arm(READY["left"])})
        self.assertIsNone(m.hold)
        for i in range(1, 30):  # and pushes it with closed fingers: not modelled, the plate stays
            m.update({"right": self.tips_arm(c + [-0.002 * i, 0.002 * i], z, 0.0), "left": arm(READY["left"])})
        self.assertEqual(m.theta, 0.0)

    def test_slip_and_stops(self):
        m = offline_model.PlateModel()
        c = self.grip(m)
        z = L.TOP + L.PLATE_TOP + 0.007
        m.update({"right": self.tips_arm(c + (c - AXIS) / np.linalg.norm(c - AXIS) * 0.016, z, 0.128),
                  "left": arm(READY["left"])})
        self.assertIsNone(m.hold)  # drifted 16 mm off the circle: let go
        m = offline_model.PlateModel()
        self.grip(m)
        for i in range(1, 400):  # try to carry it past the joint's stop
            a = math.radians(L.PEG_ANGLES_DEG[2] + i)
            m.update({"right": self.tips_arm(AXIS + L.PEG_CIRCLE * np.array([math.cos(a), math.sin(a)]), z, 0.128),
                      "left": arm(READY["left"])})
        self.assertAlmostEqual(m.theta, L.PLATE_UPPER)

    def test_reset_and_bad_arms(self):
        m = offline_model.PlateModel()
        self.grip(m)
        m.theta = 1.0
        m.update(None)
        self.assertEqual(m.theta, 1.0)
        m.reset()
        self.assertEqual((m.theta, m.hold, m.grips), (0.0, None, 0))


# -- task.py against the offline stand-in ---------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the objects, the hooks, the reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def turntable(self, theta=0.0, rate=0.0, source="sim", origin=None):
        spec = self.task.TURNTABLE
        pose = (*(spec.pos if origin is None else origin), 0.0, 0.0, 0.0, 1.0)
        return self.mk.state_from_poses(spec, pose, joints={"plate": theta}, joint_vel={"plate": rate},
                                        source=source)

    def sm(self, s):
        spec = self.task.SMOOTHER
        return self.mk.state_from_poses(spec, (*s.origin, *s.quat_xyzw), (*s.lin_vel, *s.ang_vel), source="sim")

    def feed(self, a):
        rigstate.offline_feed({k: {"tcp": v.tcp, "tips": v.tips, "opening": v.opening} for k, v in a.items()})

    def test_objects(self):
        objs = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(list(objs), ["turntable", "smoother", "stand_left", "stand_right"])
        tt, sm = objs["turntable"], objs["smoother"]
        self.assertTrue(tt.fixed_base)
        self.assertEqual([(j.name, j.kind, j.axis) for j in tt.joints], [("plate", "revolute", "Z")])
        j = tt.joints[0]
        self.assertTrue(-2 * math.pi + 0.15 < j.lower < 0 < j.upper < 2 * math.pi - 0.15)
        self.assertAlmostEqual(j.damping, L.PLATE_DAMPING)
        self.assertEqual(j.spring, 0.0)
        self.assertEqual(j.friction, 0.0)
        # the full score needs about 334 deg of turn (340 sectors); the stop leaves room to spare
        self.assertGreaterEqual(math.degrees(j.upper) - (L.FULL - 2 * L.ARC_DEG), 15.0)
        self.assertFalse(sm.fixed_base)
        self.assertLessEqual(sm.mass, 0.3)
        self.assertGreaterEqual(min(L.BLADE[1], L.HANDLE[1]), 0.004)  # walls at least 4 mm
        self.assertTrue(0.016 <= L.HANDLE[1] <= 0.080)  # a graspable width
        lo, hi = tt.world_bounds()
        self.assertAlmostEqual(lo[2], L.TOP)
        self.assertAlmostEqual(hi[2], L.TOP + L.CAKE_TOP)
        # the pegs stand on the plate inside its rim; the cake on its centre
        self.assertLess(L.PEG_CIRCLE + L.PEG_R, L.PLATE_R)
        st = self.turntable(theta=1.0)
        plo, phi = st.link_bounds("plate")
        self.assertAlmostEqual(plo[2], L.TOP + L.BASE_H)
        # the stand: the blade hangs in the slot with room each side, the handle rests across both blocks
        a, b = objs["stand_left"], objs["stand_right"]
        slot = (a.pos[1] - a.size[1] / 2) - (b.pos[1] + b.size[1] / 2)
        self.assertAlmostEqual(slot, L.STAND_SLOT)
        play_side = (slot - L.BLADE[1]) / 2
        self.assertGreaterEqual(play_side, 0.006 - 1e-9)  # 6 mm clear each side
        self.assertGreater((L.HANDLE[1] - slot) / 2, play_side)  # the handle stays on both blocks however pushed
        self.assertAlmostEqual(sm.pos[2] + L.BLADE[2], L.TOP + L.STAND_BLOCK[2])  # handle's underside on the tops
        self.assertAlmostEqual(sm.pos[2] - L.TOP, L.HANG)
        self.assertEqual(self.task.SCORE_OVER, "end")

    def test_hooks_read_the_joint_in_the_simulator(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        s = at_set()
        self.feed(arms(left=holding(s)))
        self.assertFalse(self.task.success({"turntable": self.turntable(theta=0.3), "smoother": self.sm(s)}))
        self.assertAlmostEqual(tr.last.details["turn_deg"], math.degrees(0.3), places=2)
        self.assertTrue(tr.last.details["placed"])
        self.assertEqual(self.task.score({"turntable": self.turntable(theta=0.3), "smoother": self.sm(s)}), 0.0)

    def test_offline_states_use_the_model(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(self.task._MODEL.theta, 0.0)
        c = S.peg_centres(0.0)[2]
        z = L.TOP + L.PLATE_TOP + 0.007 + TIPS
        s = in_stand()
        for i in range(31):  # close on peg 2: the joint the offline state reports (0) is replaced by the model's
            self.feed(arms(right=arm((*c, z), (0.05 - 0.038 * i / 30) / L.FULL_GAP)))
            self.task.success({"turntable": self.turntable(theta=0.0, source="offline"), "smoother": self.sm(s)})
        a = math.radians(L.PEG_ANGLES_DEG[2] + 10)
        p = AXIS + L.PEG_CIRCLE * np.array([math.cos(a), math.sin(a)])
        self.feed(arms(right=arm((*p, z), 0.128)))
        self.task.success({"turntable": self.turntable(theta=0.0, source="offline"), "smoother": self.sm(s)})
        self.assertAlmostEqual(tr.last.details["turn_deg"], 10.0, places=3)
        # score() reads the model without moving it
        self.task.score({"turntable": self.turntable(theta=0.0, source="offline"), "smoother": self.sm(s)})
        self.assertAlmostEqual(math.degrees(self.task._MODEL.theta), 10.0, places=3)
        # a simulator state never uses it
        self.task.success({"turntable": self.turntable(theta=0.0, source="sim"), "smoother": self.sm(s)})
        self.assertAlmostEqual(tr.last.details["turn_deg"], 0.0)
        rigstate.offline_reset()  # rigstate's reset resets the model too
        self.assertEqual(self.task._MODEL.theta, 0.0)

    def test_broken_turntable_state_scores_zero(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        bad = self.turntable(origin=(math.nan, 0.0, 0.0))
        self.assertFalse(self.task.success({"turntable": bad, "smoother": self.sm(in_stand())}))
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(self.task.score({"turntable": bad, "smoother": self.sm(in_stand())}), 0.0)
        no_joint = SimpleNamespace(origin=np.zeros(3), joints={}, joint_vel={})
        self.assertEqual(self.task.score({"turntable": no_joint, "smoother": self.sm(in_stand())}), 0.0)

    def test_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        objs = {"turntable": self.turntable(), "smoother": self.sm(in_stand())}
        self.task.success(objs)
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.task.success(objs)  # arms None right after a reset
        self.assertEqual(tr.arms_unavailable, 1)
        self.feed(arms())
        self.task.success(objs)
        self.assertEqual(tr.steps, 2)
        self.task._MODEL.theta = 0.5
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success(objs)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(self.task._MODEL.theta, 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(arms())
            self.assertFalse(self.task.success({"turntable": self.turntable(), "smoother": self.sm(in_stand())}))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches(self):
        task = self.task
        self.assertEqual(set(task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_plan(self):
        task = self.task
        plan = task.demo()

        def times(wps):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                out.append((t, w))
            return out

        left, right = times(plan["left"]), times(plan["right"])
        strokes = [(t - w.duration, t, w) for t, w in right if w.label.startswith("turn:")]
        self.assertEqual(len(strokes), 4 * task.CHORDS)
        self.assertGreaterEqual(strokes[0][0], task.T_SET + 0.3)  # turning starts only once the smoother is set
        hold = [(t - w.duration, t) for t, w in left if w.label.startswith("hold it still")]
        self.assertEqual(len(hold), 1)
        self.assertLessEqual(hold[0][0], task.T_SET + 1e-9)
        self.assertGreaterEqual(hold[0][1], strokes[-1][1] + 0.3)  # held until the last stroke has ended
        total = len(task._strokes()) * task.STROKE_DEG
        self.assertTrue(L.FULL - 2 * L.ARC_DEG + 8 <= total <= math.degrees(L.PLATE_UPPER) - 3, total)
        # each chord stays within 2 mm of the pegs' circle at its middle (a closed grip on the plate is a closed chain)
        step = math.radians(task.STROKE_DEG / task.CHORDS)
        self.assertLess(L.PEG_CIRCLE * (1 - math.cos(step / 2)), 0.002)
        # every open/close takes at least 1.5 s after a settle of at least 0.5 s; every slow descent takes 1.5 s
        for wps in (plan["left"], plan["right"]):
            for a, b in zip(wps, wps[1:]):
                if abs(b.finger - a.finger) > 0.01:
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertTrue(a.label.startswith("settle") and a.duration >= 0.5, (a.label, b.label))
                    self.assertEqual(tuple(a.pos), tuple(b.pos))
                if b.label.endswith("slowly"):
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertLessEqual(a.pos[2] - b.pos[2], 0.035, b.label)
        # the success hold ends after the turning gripper has let go and risen 20 mm clear of the pegs
        let_go = [t for t, w in right if w.label == "let go of the peg"][-1]
        self.assertGreater(len(plan["right"]), 0)
        self.assertGreater(let_go, strokes[-1][1])
        self.assertEqual(task.SUCCESS_HOLD_S, L.FINISH_HOLD_S)

    def test_text_agrees(self):
        """task.py, task.json and both README commands give the model the same instruction; task.json is
        complete."""
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        self.assertEqual(meta["scores"], "TBD after the GPU runs (offline kinematic check predicts demo: 1.0)")
        self.assertNotIn("media", meta)
        for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                    "fidelity", "reset_notes"):
            self.assertTrue(meta[key], key)
        steps = int(meta["run_limit"].split()[0].replace(",", ""))
        self.assertIn(f"--max-steps {steps}", readme)
        demo_steps = math.ceil(max(sum(w.duration for w in v) for v in self.task.demo().values()) * HZ)
        self.assertTrue(3 * demo_steps <= steps <= 6 * demo_steps)
        for text in (json.dumps(meta), readme):
            self.assertNotIn("\u2014", text)  # no em-dashes


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics (the turntable's angle from offline_model)."""

    def test_demo_predicts_one_and_each_arm_alone_less(self):
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 3600)
        self.assertEqual(rep["one_arm"], {"left_only": 0.25, "right_only": 0.0})
        for side in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][side]["worst_ik_mm"], 0.5)
        self.assertGreater(rep["playback"]["min_arm_clearance_m"], 0.02)
        self.assertEqual(proxy["final_hooks"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
