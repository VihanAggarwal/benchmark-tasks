"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, carriage_tilt_deg, grip_point, grippers_done, holder, in_band, plunger_height, plunger_in_sleeve,
    plunger_tilt_deg, rotation, upright, valid,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
OPEN = 1.0
P_GRIP = (2 * L.PLUNGER_RADIUS - 0.008) / L.FULL_GAP  # rigstate openings: commanded on the plunger ...
C_GRIP = (L.CARRIAGE_SIZE - 0.008) / L.FULL_GAP  # ... and on the carriage

Q0 = np.array([0.0, 0.0, 0.0, 1.0])
C_Z = L.PLATE_TOP_Z + L.CARRIAGE_HALF  # the carriage's centre resting on the plate
C_START = np.array([L.HOME[0], L.HOME[1], C_Z])
P_START = np.array([L.SLEEVE_X, L.SLEEVE_Y, L.TOP + L.PLUNGER_HEIGHT / 2])
P_TCP = L.PLUNGER_HEIGHT / 2 - L.PLUNGER_GRIP_BELOW_TOP  # a gripped plunger's centre is this far below the TCP
C_TCP = 0.002  # a gripped carriage's centre is this far below the TCP (tips 9 mm above the plate)
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
BAND = 0.0225  # the demo's lift: the middle of the 15-30 mm band
GUIDE = 0.003  # the demo guides the carriage 3 mm above the plate


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=(L.CARRIAGE_SIZE,) * 3):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def cyl(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like plunger (upright cylinder turned by ``quat``) with its world bounding box."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.abs(a) * L.PLUNGER_HEIGHT / 2 + L.PLUNGER_RADIUS * np.sqrt(np.clip(1 - a * a, 0.0, 1.0))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def at(xy, lift=0.0):
    """The carriage's centre over a mark (x, y), ``lift`` above resting on the plate."""
    return np.array([xy[0], xy[1], C_Z + lift])


def up(h):
    """The plunger's centre lifted ``h`` in its sleeve."""
    return P_START + [0, 0, h]


KEYS = ("c", "cq", "p", "pq", "L", "Lo", "R", "Ro")
VEC = ("c", "p", "L", "R")


class Film:
    """A trial as a list of frames: both objects' centres and orientations, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"c": C_START, "cq": Q0, "p": P_START, "pq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in ("cq", "pq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def then(self, other):
        out = Film()
        out.frames = self.frames + other.frames[1:]
        out.cur = dict(other.cur)
        return out


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vc = vp = np.zeros(3)
        else:
            vc, vp = (fr["c"] - prev["c"]) / DT, (fr["p"] - prev["p"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(box(fr["c"], fr["cq"], vc), cyl(fr["p"], fr["pq"], vp), a)
        prev = fr
    return r, t


# -- the pieces of a trial ------------------------------------------------------------------------------------------
def left_enables(f, h=BAND):
    """The left arm comes down over the plunger, grips it and lifts it ``h`` (about 5 s)."""
    g = P_START + [0, 0, P_TCP]
    f.go(30, L=g + [0, 0, 0.04]).go(20, L=g).go(30, Lo=P_GRIP)
    f.go(30, L=g + [0, 0, h], p=up(h))
    return f


def right_grips(f):
    """The right arm comes down beside the carriage and grips it, then lifts it 3 mm (about 5 s)."""
    g = C_START + [0, 0, C_TCP]
    f.go(30, R=g + [0, 0, 0.09]).go(30, R=g).go(30, Ro=C_GRIP)
    f.go(20, R=g + [0, 0, GUIDE], c=C_START + [0, 0, GUIDE])
    return f


def guide(f, xy, n, lift=GUIDE):
    """The right arm guides the gripped carriage to the mark ``xy`` over n steps, ``lift`` above the plate."""
    c = at(xy, lift)
    return f.go(n, c=c, R=c + [0, 0, C_TCP])


def right_lets_go(f, at_xy=L.HOME):
    c = at(at_xy)
    f.go(20, c=c, R=c + [0, 0, C_TCP]).go(30, Ro=OPEN).go(30, R=c + [0, 0, 0.074])
    return f


def left_lets_down(f):
    g = P_START + [0, 0, P_TCP]
    f.go(30, L=g, p=P_START).go(30, Lo=OPEN).go(40, L=g + [0, 0, 0.07])
    return f


def procedure(settle=30, stop=40, order=("red", "blue", "home"), end_at=L.HOME):
    """The intended trial: enable, grip, guide to each mark in ``order`` at 20-21 mm/s with a ``stop``-step stop
    on each, set the carriage down at ``end_at``, let go, lower the plunger, let go, all still ``settle`` steps."""
    f = Film().hold(10)
    left_enables(f)
    right_grips(f)
    for name in order:
        xy = L.TARGETS[name]
        d = math.hypot(f.cur["c"][0] - xy[0], f.cur["c"][1] - xy[1])
        guide(f, xy, max(1, int(round(d / 0.020 / DT)))).hold(stop)
    right_lets_go(f, end_at)
    left_lets_down(f)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_carriage_tilt_is_folded(self):
        self.assertAlmostEqual(carriage_tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(carriage_tilt_deg(about("y", 90)), 0.0, places=4)  # on its side: still a cube
        self.assertAlmostEqual(carriage_tilt_deg(about("x", 180)), 0.0, places=4)
        self.assertAlmostEqual(carriage_tilt_deg(about("x", 84)), 6.0, places=4)
        self.assertAlmostEqual(carriage_tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt

    def test_plunger_tilt_either_end_up(self):
        self.assertAlmostEqual(plunger_tilt_deg(about("x", 12)), 12.0, places=4)
        self.assertAlmostEqual(plunger_tilt_deg(about("y", 180)), 0.0, places=4)
        self.assertAlmostEqual(plunger_tilt_deg(about("y", 170)), 10.0, places=4)
        self.assertAlmostEqual(plunger_tilt_deg(about("x", 90)), 90.0, places=4)

    def test_height_band_and_grip_point(self):
        self.assertAlmostEqual(plunger_height(cyl(P_START)), 0.0, places=9)
        self.assertFalse(in_band(cyl(up(0.0145))))
        self.assertTrue(in_band(cyl(up(0.0155))))
        self.assertTrue(in_band(cyl(up(0.0295))))
        self.assertFalse(in_band(cyl(up(0.0305))))
        g = grip_point(cyl(up(0.02)))
        self.assertTrue(np.allclose(g, [L.SLEEVE_X, L.SLEEVE_Y, L.TOP + 0.02 + 0.100]))
        self.assertTrue(np.allclose(grip_point(cyl(up(0.02), about("y", 180))), g))  # either end up

    def test_upright(self):
        self.assertTrue(upright(cyl(up(BAND), about("x", 3.9))))
        self.assertFalse(upright(cyl(up(BAND), about("x", 4.1))))
        self.assertFalse(upright(cyl(up(BAND), about("y", 6))))  # what a jam across the sleeve needs
        self.assertTrue(upright(cyl(up(BAND), about("y", 177))))  # either end up
        self.assertTrue(in_band(cyl(up(BAND), about("x", 6))))  # still in its band when jammed

    def test_holder(self):
        p = cyl(up(0.02))
        g = grip_point(p)
        self.assertEqual(holder(p, snap(left_tcp=g, left_open=P_GRIP)), "left")
        self.assertEqual(holder(p, snap(left_tcp=g + [0.0, 0.02, 0.014], left_open=P_GRIP)), "left")  # 24.4 mm
        self.assertIsNone(holder(p, snap(left_tcp=g + [0.0, 0.0, 0.026], left_open=P_GRIP)))  # too far
        self.assertIsNone(holder(p, snap(left_tcp=g, left_open=0.55)))  # open past 0.5: not holding
        self.assertIsNone(holder(p, snap()))

    def test_sleeve(self):
        self.assertTrue(plunger_in_sleeve(cyl(P_START + [0.003, -0.003, 0])))
        self.assertTrue(plunger_in_sleeve(cyl(up(0.079))))
        self.assertFalse(plunger_in_sleeve(cyl(up(0.081))))  # pulled out of the top
        self.assertFalse(plunger_in_sleeve(cyl(up(0.081) + [0.03, 0, 0])))
        self.assertFalse(plunger_in_sleeve(cyl(P_START + [0.016, 0, 0])))  # centre outside the opening

    def test_grippers_done(self):
        c, p = box(C_START), cyl(P_START)
        clear_l = P_START + [0, 0, P_TCP + 0.07]
        clear_r = C_START + [0, 0, 0.074]
        self.assertTrue(grippers_done(snap(clear_l, OPEN, clear_r, OPEN), c, p))
        self.assertFalse(grippers_done(snap(clear_l, OPEN, clear_r, 0.85), c, p))  # right not fully open
        near = C_START + [0, 0, 0.020 + TIPS + 0.029]  # the right tips 29 mm above the carriage's top
        self.assertFalse(grippers_done(snap(clear_l, OPEN, near, OPEN), c, p))
        self.assertTrue(grippers_done(snap(clear_l, OPEN, near + [0, 0, 0.002], OPEN), c, p))
        self.assertFalse(grippers_done(snap(P_START + [0, 0, P_TCP], OPEN, clear_r, OPEN), c, p))  # still round it


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertEqual(t.stage, 0)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.stage, 3)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertLess(t.idle_travel, 0.0005)
        self.assertLess(t.max_speed, 0.025)
        s = t.stop_steps
        self.assertTrue(s["red"] < s["blue"] < s["home"])

    def test_extra_stops_in_between_do_not_matter(self):
        r, t = play(procedure(order=("blue", "red", "home", "blue", "home")))
        self.assertEqual((r.score, r.success), (1.0, True))
        r, t = play(procedure(order=("red", "home", "blue", "home")))
        self.assertEqual(r.score, 1.0)

    def test_not_back_home_scores_075(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50).hold(40)
        guide(f, L.BLUE, 106).hold(40)
        right_lets_go(f, L.BLUE)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertEqual(t.stage, 2)

    def test_home_not_last_scores_075(self):
        # red, blue, home, then guided back to red and left there: it did not finish at home
        r, t = play(procedure(order=("red", "blue", "home", "red"), end_at=L.RED))
        self.assertEqual(r.score, 0.75)
        self.assertEqual(t.stage, 3)

    def test_left_at_home_but_6_mm_off_scores_075(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        off = (L.HOME[0] + 0.006, L.HOME[1])
        guide(f, off, 10)  # nudged 6 mm while still enabled, after the home stop
        right_lets_go(f, off)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(t.breach)

    def test_plunger_not_let_down_scores_075(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        right_lets_go(f)
        r, t = play(f.hold(30))  # still held up in its band
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["plunger_down"])

    def test_plunger_left_up_with_grippers_clear_scores_075(self):
        # Every stop made, both grippers open and clear, but the plunger left wedged up in its band
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        right_lets_go(f)
        f.go(30, Lo=OPEN).go(30, L=P_START + [0, 0, P_TCP + BAND + 0.07])
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertTrue(r.details["grippers_done"])
        self.assertFalse(r.details["plunger_down"])
        self.assertFalse(t.breach)

    def test_gripper_not_open_scores_075(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        c = at(L.HOME)
        f.go(20, c=c, R=c + [0, 0, C_TCP])  # set down but still gripped
        left_lets_down(f)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_done"])

    def test_red_only_scores_half(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50).hold(40)
        guide(f, L.HOME, 50).hold(40)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.stage, 1)

    def test_blue_before_red_scores_half(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, L.BLUE, 60).hold(40)
        guide(f, L.RED, 106).hold(40)
        guide(f, L.HOME, 50).hold(40)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertIsNone(t.stop_steps["blue"])

    def test_guided_without_stopping_scores_quarter(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.HOME[0] + 0.022, L.HOME[1]), 50)  # 22 mm in 2.5 s, no mark reached
        right_lets_go(f, (L.HOME[0] + 0.022, L.HOME[1]))
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.25)
        self.assertIsNotNone(t.guided_at)

    def test_guided_too_little_or_too_briefly_scores_zero(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.HOME[0] + 0.015, L.HOME[1]), 40)  # only 15 mm
        right_lets_go(f, (L.HOME[0] + 0.015, L.HOME[1]))
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        # 22 mm guided, but the plunger held up only 2.5 s in all
        f = Film().hold(10)
        right_grips(f)
        g = P_START + [0, 0, P_TCP]
        f.go(30, L=g).go(30, Lo=P_GRIP).go(6, L=g + [0, 0, BAND], p=up(BAND))
        guide(f, (L.HOME[0] + 0.022, L.HOME[1]), 40)
        f.go(6, L=g, p=P_START)
        r, t = play(f.hold(20))
        self.assertGreaterEqual(t.enabled_travel, 0.020)
        self.assertIsNone(t.guided_at)
        self.assertEqual(r.score, 0.0)

    def test_enabled_time_must_be_one_stretch(self):
        # Two enabled stretches of 2 s, 12 mm guided in each (24 mm, 4 s in all): no single 3 s stretch
        g = P_START + [0, 0, P_TCP]
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        f.go(10, L=g, p=P_START)  # drop out of the band before guiding
        x = L.HOME[0]
        for _ in range(2):
            f.go(5, L=g + [0, 0, BAND], p=up(BAND))
            x += 0.012
            guide(f, (x, L.HOME[1]), 30).hold(5)
            f.go(5, L=g, p=P_START).hold(10)
        r, t = play(f)
        self.assertIsNone(t.guided_at)
        self.assertGreaterEqual(t.enabled_travel, 0.020)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)

    def test_held_tilted_4_degrees_or_more_is_not_enabled(self):
        def trial(tilt):
            f = Film().hold(10)
            left_enables(f)
            f.go(5, pq=about("x", tilt))  # held in its band, leaning in the grip
            right_grips(f)
            for name in ("red", "blue", "home"):
                guide(f, L.TARGETS[name], 60).hold(40)
            right_lets_go(f)
            f.go(5, pq=Q0)
            left_lets_down(f)
            return play(f.hold(30))
        r, t = trial(3.0)
        self.assertEqual((r.score, r.success), (1.0, True))
        r, t = trial(4.5)
        self.assertEqual(r.score, 0.0)
        self.assertIn("moved_while_not_enabled", t.breaches)

    def test_jiggling_in_place_is_not_guiding(self):
        # Enabled for 6 s, the carriage shaken 3 mm either way: 48 mm of travel, but never 20 mm from where it was
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.HOME[0] + 0.003, L.HOME[1]), 5)
        for i in range(8):
            guide(f, (L.HOME[0] + (0.003 if i % 2 else -0.003), L.HOME[1]), 10)
        guide(f, L.HOME, 5)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertGreaterEqual(t.enabled_travel, 0.040)
        self.assertIsNone(t.guided_at)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)

    def test_out_and_back_in_one_stretch_counts(self):
        # 22 mm out and straight back within one enabled stretch of over 3 s: the farthest point counts
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.HOME[0] - 0.022, L.HOME[1]), 30)
        guide(f, L.HOME, 30)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertIsNotNone(t.guided_at)
        self.assertEqual(r.score, 0.25)

    def test_reach_belongs_to_one_stretch(self):
        g = P_START + [0, 0, P_TCP]

        def gripped_both():
            f = Film().hold(10)
            right_grips(f)  # gripped and raised 3 mm before enabling: no horizontal travel
            f.go(30, L=g).go(30, Lo=P_GRIP)
            return f

        def enable(f):
            f.go(5, L=g + [0, 0, BAND], p=up(BAND))

        def disable(f):
            f.go(5, L=g, p=P_START).hold(20)

        # (a) 2 s enabled and still, a break, then 22 mm in a 1.75 s stretch: over 3 s enabled in all, but not in one
        f = gripped_both()
        enable(f)
        f.hold(35)
        disable(f)
        enable(f)
        guide(f, (L.HOME[0] + 0.022, L.HOME[1]), 30)
        disable(f)
        r, t = play(f)
        self.assertIsNone(t.guided_at)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        # (b) 15 mm in each of two 3.7 s stretches, 30 mm from where the first began: no one stretch reaches 20 mm
        f = gripped_both()
        x = L.HOME[0]
        for _ in range(2):
            enable(f)
            x += 0.015
            guide(f, (x, L.HOME[1]), 40).hold(30)
            disable(f)
        r, t = play(f)
        self.assertIsNone(t.guided_at)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        # (c) 22 mm out in 1.25 s right after enabling, back in 1.25 s, held 2 s: the farthest point counts
        f = gripped_both()
        enable(f)
        guide(f, (L.HOME[0] - 0.022, L.HOME[1]), 25)
        guide(f, L.HOME, 25).hold(40)
        disable(f)
        r, t = play(f)
        self.assertIsNotNone(t.guided_at)
        self.assertEqual(r.score, 0.25)

    def test_a_stop_must_last_1_s(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50).hold(17)  # 0.9 s at the mark (the last guiding step counts too)
        guide(f, L.BLUE, 106).hold(40)
        guide(f, L.HOME, 60).hold(40)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.25)  # guided, but no stop counted, so blue and home are not in order either
        self.assertIsNone(t.stop_steps["red"])
        self.assertEqual(t.stage, 0)

    def test_a_stop_must_be_within_3_mm(self):
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.RED[0] - 0.0035, L.RED[1]), 50).hold(40)
        r, t = play(f)
        self.assertIsNone(t.stop_steps["red"])
        self.assertEqual(r.score, 0.25)
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.RED[0] - 0.002, L.RED[1] + 0.002), 50).hold(40)  # 2.8 mm off
        r, t = play(f)
        self.assertEqual(r.score, 0.5)

    def test_a_stop_must_be_enabled_throughout(self):
        # The carriage waits on red while the plunger is let down and lifted again halfway through the second
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50).hold(10)
        g = P_START + [0, 0, P_TCP]
        f.go(4, L=g + [0, 0, 0.010], p=up(0.010)).go(4, L=g + [0, 0, BAND], p=up(BAND)).hold(12)
        r, t = play(f)
        self.assertIsNone(t.stop_steps["red"])
        self.assertFalse(t.breach)
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.5)  # a fresh 1 s once it is held in its band again

    def test_a_crawl_is_not_a_stop(self):
        # 2.5 mm/s through the red mark: never under 2 mm/s there
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        guide(f, (L.RED[0] - 0.004, L.RED[1]), 60)
        guide(f, (L.RED[0] + 0.004, L.RED[1]), 64)
        r, t = play(f)
        self.assertIsNone(t.stop_steps["red"])

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, t = play(f.go(1, c=C_START + [0.0008, 0, 0]).go(1, c=C_START))  # still sliding: 16 mm/s
        self.assertEqual((r.score, r.success), (1.0, False))
        self.assertFalse(t.breach)  # 1.6 mm while not enabled: under the 2 mm limit
        r, _ = play(f.hold(1))
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(P_START + [0, 0, P_TCP + 0.07], OPEN, C_START + [0, 0, 0.076], OPEN)
        r = tr.update(box(C_START), cyl(P_START, spin=(0, 0, 1.0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(C_START, vel=(0, 0, -0.02)), cyl(P_START), a)
        self.assertEqual((r.score, r.success), (1.0, False))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        last = (box(C_START), cyl(P_START), snap(P_START + [0, 0, P_TCP + 0.07], OPEN, C_START + [0, 0, 0.076], OPEN))
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class Rules(unittest.TestCase):
    """Each zero condition, once met, scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn(key, t.breaches)
        return t

    def with_procedure(self, f):
        return f.then(procedure())

    def test_carriage_moved_while_not_enabled(self):
        f = Film().hold(5)
        right_grips(f)
        guide(f, L.RED, 50)  # the right arm alone: nothing holds the plunger
        t = self.assert_zero(f.hold(10), "moved_while_not_enabled")
        self.assertGreater(t.idle_travel, 0.04)

    def test_not_enabled_travel_limit_is_2_mm(self):
        f = Film().hold(5).go(20, c=C_START + [0.0009, 0, 0]).go(20, c=C_START)  # 1.8 mm, not enabled
        r, t = play(self.with_procedure(f))
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(t.idle_travel, 0.0018, delta=0.0002)
        f = Film().hold(5).go(20, c=C_START + [0.0018, 0, 0]).go(20, c=C_START + [0.0018, 0.0012, 0])
        self.assert_zero(f.hold(5), "moved_while_not_enabled")  # 3.0 mm in two nudges

    def test_physics_jitter_adds_nothing(self):
        rng = np.random.default_rng(3)
        f = Film()
        for _ in range(600):
            f.go(1, c=C_START + [*rng.uniform(-0.00009, 0.00009, 2), 0])
        r, t = play(f)
        self.assertFalse(t.breach)
        self.assertEqual(t.idle_travel, 0.0)

    def test_plunger_in_band_but_not_held(self):
        # The plunger left up in its band (wedged), the left gripper open beside it: not enabled
        f = Film().hold(5)
        left_enables(f)
        g = P_START + [0, 0, P_TCP + BAND]
        f.go(30, Lo=OPEN).go(20, L=g + [0, 0, 0.06])
        right_grips(f)
        guide(f, L.RED, 50).hold(40)
        t = self.assert_zero(f, "moved_while_not_enabled")
        self.assertTrue(in_band(cyl(f.cur["p"])))
        self.assertEqual(t.stage, 0)

    def test_plunger_pulled_too_high(self):
        f = Film().hold(5)
        left_enables(f, h=0.035)  # 35 mm: squeezed through, like a panic grip
        right_grips(f)
        guide(f, L.RED, 50).hold(40)
        self.assert_zero(f, "moved_while_not_enabled")

    def test_plunger_not_lifted_enough(self):
        f = Film().hold(5)
        left_enables(f, h=0.012)
        right_grips(f)
        guide(f, L.RED, 50).hold(40)
        self.assert_zero(f, "moved_while_not_enabled")

    def test_too_fast(self):
        f = Film().hold(5)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 18).hold(40)  # 50 mm in 0.9 s: 56 mm/s
        t = self.assert_zero(f, "too_fast")
        self.assertGreater(t.max_speed, L.SPEED_CAP)
        f = Film().hold(5)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 22).hold(40)  # 45 mm/s: allowed
        r, t = play(f)
        self.assertEqual(r.score, 0.5)

    def test_carriage_lifted(self):
        f = Film().hold(5)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50, lift=0.006).hold(40)
        self.assert_zero(f, "carriage_lifted")
        f = Film().hold(5)
        left_enables(f)
        right_grips(f)
        guide(f, L.RED, 50, lift=0.0045).hold(40)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_carriage_tipped(self):
        f = procedure(settle=5)
        f.go(5, cq=about("y", 12)).go(5, cq=Q0).hold(25)
        self.assert_zero(f, "carriage_tipped")

    def test_plunger_tipped(self):
        f = Film().hold(5).go(5, pq=about("x", 16)).go(5, pq=Q0)
        self.assert_zero(self.with_procedure(f), "plunger_tipped")

    def test_plunger_out_of_its_sleeve(self):
        f = Film().hold(5).go(20, p=up(0.09)).go(20, p=P_START)
        self.assert_zero(self.with_procedure(f), "plunger_out_of_sleeve")

    def test_off_the_table(self):
        f = procedure(settle=5)
        f.go(30, c=[0.36, -0.70, L.TOP - 0.4])
        t = self.assert_zero(f, "off_table")
        self.assertIn("moved_while_not_enabled", t.breaches)

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5).go(30, c=C_START + [0.004, 0, 0]).go(30, c=C_START)  # 8 mm while not enabled
        r, t = play(self.with_procedure(f.then(Film().hold(30))))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(t.stage, 3)  # the stops were all made, but the breach stands
        self.assertEqual(r.details["first_breach_step"], t.breaches["moved_while_not_enabled"])


class Exploits(unittest.TestCase):
    def test_right_arm_alone_scores_zero(self):
        # The right arm does the whole carriage job with nothing holding the plunger.
        f = Film().hold(10)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("moved_while_not_enabled", t.breaches)
        self.assertEqual(t.stage, 0)

    def test_wedged_plunger_scores_zero(self):
        # The left lifts the plunger, lets go of it up in the band (wedged against the sleeve), goes away; the
        # right then guides the carriage through every stop.
        f = Film().hold(10)
        left_enables(f)
        f.go(30, Lo=OPEN).go(20, L=READY_L)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        right_lets_go(f)
        f.go(10, p=P_START)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.stage, 0)

    def test_jammed_plunger_with_a_gripper_parked_beside_it_scores_zero(self):
        # The left lifts the plunger into its band, tips it 6 degrees so it jams across the sleeve, lets go, and
        # parks its closed gripper 20 mm beside the grip point (it would count as 'held'); the right then guides the
        # carriage through every stop. A jammed plunger is not upright, so this is not enabled.
        f = Film().hold(10)
        left_enables(f)
        f.go(10, pq=about("x", 6))
        g = grip_point(cyl(f.cur["p"], about("x", 6)))
        f.go(30, Lo=OPEN).go(20, L=g + [0, 0.02, 0.0]).go(30, Lo=0.1)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.details["held_by"], "left")  # the gripper test alone would pass
        self.assertTrue(in_band(cyl(f.cur["p"], about("x", 6))))
        self.assertEqual(r.score, 0.0)
        self.assertIn("moved_while_not_enabled", t.breaches)
        self.assertEqual(t.stage, 0)

    def test_carrying_it_through_the_air_scores_zero(self):
        # Lifting the carriage clear of the plate and carrying it from mark to mark
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60, lift=0.03).hold(10)
            guide(f, L.TARGETS[name], 10).hold(30)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("carriage_lifted", t.breaches)

    def test_flick_between_marks_scores_zero(self):
        # Enabled, but the carriage is flicked from mark to mark in 0.5 s
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 10).hold(40)
        right_lets_go(f)
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("too_fast", t.breaches)

    def test_moving_it_after_letting_the_plunger_down_scores_zero(self):
        # All three stops made, then the plunger let down and the carriage pushed 5 mm
        f = Film().hold(10)
        left_enables(f)
        right_grips(f)
        for name in ("red", "blue", "home"):
            guide(f, L.TARGETS[name], 60).hold(40)
        left_lets_down(f)
        guide(f, (L.HOME[0] + 0.005, L.HOME[1]), 10)
        right_lets_go(f, (L.HOME[0] + 0.005, L.HOME[1]))
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("moved_while_not_enabled", t.breaches)

    def test_pushing_with_open_fingers_is_fine(self):
        # Not an exploit: pushing the carriage with an open finger, slowly and while enabled, is hand-guiding too.
        f = Film().hold(10)
        left_enables(f)
        f.go(30, R=C_START + [-0.03, 0, 0.01])  # behind it, open
        for name in ("red", "blue", "home"):
            c = at(L.TARGETS[name])
            guide_n = 60
            f.go(guide_n, c=c, R=c + [-0.03, 0, -0.01]).hold(40)
        f.go(30, R=f.cur["R"] + [0, 0, 0.08])
        left_lets_down(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (1.0, True))


class Arms(unittest.TestCase):
    def test_arms_unavailable_held_taken_as_met(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: held and 'grippers done' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])

    def test_arms_unavailable_height_still_decides(self):
        f = Film().hold(5)
        right_grips(f)
        guide(f, L.RED, 50).hold(30)  # the plunger never lifted
        r, t = play(f, arms=False)
        self.assertEqual(r.score, 0.0)
        self.assertIn("moved_while_not_enabled", t.breaches)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        p = cyl(up(BAND))
        r = t.update(box(C_START), p, snap())  # in band, but the left gripper is away: not enabled
        self.assertFalse(r.details["enabled"])
        r = t.update(box(C_START), p, snap(left_tcp=grip_point(p), left_open=P_GRIP))
        self.assertTrue(r.details["enabled"])
        self.assertEqual(r.details["held_by"], "left")
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(box(C_START), cyl(P_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(box(C_START), cyl(P_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(box(C_START), cyl(P_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_carriage_on_another_face(self):
        f = procedure(settle=5)
        f.go(1, cq=about("x", 90)).hold(25)  # the same cube, rolled a quarter turn (turned in one step)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(t.breach)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = box(C_START)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=C_START)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), t.stage, t.idle_travel, t.max_speed, t.arms_unavailable, list(t.history))
            a = snap(P_START + [0, 0, P_TCP + 0.07], OPEN, C_START + [0, 0, 0.076], OPEN)
            for c, p in ((box(C_START), bad), (bad, cyl(P_START)), (bad, bad)):
                r = t.update(c, p, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(c, p, a).score, 0.0)
            after = (dict(t.breaches), t.stage, t.idle_travel, t.max_speed, t.arms_unavailable, list(t.history))
            self.assertEqual(after[:5], before[:5])
            self.assertTrue(all(np.allclose(x, y) for x, y in zip(after[5], before[5])))
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(C_START, about("x", 44.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        plungers = [cyl((5.0, -3.0, -1.0)), cyl(P_START, about("x", 89.9)), cyl(up(0.5)), cyl(P_START),
                    cyl((0.2, 0.2, 0.0), about("y", 135))]
        t = Tracker(HZ)
        a = snap()
        for c, p in zip(odd + odd[::-1], plungers + plungers[::-1]):
            r = t.update(c, p, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(box(C_START), cyl(P_START), None)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def breached_at_red(self):
        f = Film().hold(5)
        right_grips(f)
        guide(f, L.RED, 50).hold(10)  # moved with nothing enabled
        _, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached_at_red()
        r = t.update(box(C_START), cyl(P_START), snap())  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.idle_travel, 0.0)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_plunger_snapping_down_with_the_carriage_home_resets(self):
        # The episode ends with the carriage already home and the plunger held up: only the plunger jumps
        f = Film().hold(5)
        left_enables(f)
        f.go(10, c=C_START + [0.004, 0, 0]).go(10, c=C_START)
        f.go(40, c=C_START + [0.004, 0, 0])  # 4 mm while enabled, fine
        _, t = play(f.go(5, Lo=P_GRIP))
        t._flag("too_fast")
        t.update(box(C_START), cyl(P_START), snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached_at_red()
        cur = t.prev
        f = Film(c=cur[0][0], R=READY_R)
        f.go(60, c=C_START).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached_at_red()
        t.update(box(at(L.BLUE)), cyl(P_START), snap())  # knocked far, not to the start
        self.assertTrue(t.breach)

    def test_a_dropped_plunger_is_not_a_teleport(self):
        # Let go of from 28 mm up: it falls back onto the table, fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        t.update(box(C_START), cyl(up(0.028)), snap())
        t._flag("too_fast")
        z, v = 0.028, 0.0
        while z > 0:
            v += 9.81 * DT
            z = max(0.0, z - v * DT)
            t.update(box(C_START), cyl(up(z), vel=(0, 0, -v if z > 0 else 0)), snap())
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached_at_red()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.breach)
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
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, c=C_START, p=P_START):
        return {"carriage": box(c), "enable_plunger": cyl(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("carriage", names)
        self.assertIn("enable_plunger", names)
        self.assertEqual(self.task.SCORE_OVER, "end")
        loose = {o.name for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"carriage", "enable_plunger"})
        self.assertEqual(len([n for n in names if n.startswith("mark_")]), 12)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects()))
        snap()
        self.assertFalse(self.task.success(self.objects(c=C_START + [0.003, 0, 0])))  # 3 mm, not enabled
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects(c=C_START + [0.003, 0, 0])), 0.0)
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
        import os
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
        import os
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        span = lambda wps: sum(w.duration for w in wps)  # noqa: E731
        left, right = plan["left"], plan["right"]
        # the right closes on the carriage only once the plunger is up in its band ...
        self.assertIn("lift it into its middle band", left[6].label)
        self.assertIn("grip the carriage", right[4].label)
        self.assertGreaterEqual(span(right[:4]), span(left[:7]) + 0.5 - 1e-9)
        # ... and the left lowers it only once the right gripper is open and up
        self.assertIn("lower it back down", left[8].label)
        self.assertGreaterEqual(span(left[:8]), span(right) - 1e-9)
        # every carriage move is at most 32 mm/s at its peak (smoothstep: 1.5 x the mean)
        for prev, w in zip(right, right[1:]):
            d = math.hypot(w.pos[0] - prev.pos[0], w.pos[1] - prev.pos[1])
            self.assertLessEqual(1.5 * d / w.duration, 0.033, w.label)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
