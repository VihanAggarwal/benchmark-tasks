"""Offline tests for scoring.py and the hooks in task.py: synthetic desk states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    WINDOW, Tracker, desk_top_centre, facing, heading_rad, lowest, nearest, off_front_deg, on_floor_area, tilt_deg,
    valid,
)

TOP = L.TOP
FLOOR_Z = L.TOP + L.FLOOR
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}
D1, D2, D3, D4 = L.DESKS


def quat(heading=0.0, pitch=0.0, roll=0.0):
    """(x, y, z, w) of Rz(heading) Ry(pitch) Rx(roll), degrees."""
    h, p, r = (math.radians(v) / 2 for v in (heading, pitch, roll))
    qz = np.array([0.0, 0.0, math.sin(h), math.cos(h)])
    qy = np.array([0.0, math.sin(p), 0.0, math.cos(p)])
    qx = np.array([math.sin(r), 0.0, 0.0, math.cos(r)])

    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    return mul(mul(qz, qy), qx)


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def desk(x, y, heading=0.0, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A desk's state as mechkit reports it: frame origin (on the floor under the middle of the desk top) at (x, y),
    the candidate facing `heading` degrees, tilted by `pitch`/`roll`, resting with its lowest point `lift` above the
    floor; low/high from its outline box."""
    back, front, right, left, top = L.OUTLINE
    q = quat(heading, pitch, roll)
    local = np.array([(u, v, z) for u in (-back, front) for v in (-right, left) for z in (0.0, top)])
    pts = local @ rot(q).T
    o = np.array([x, y, FLOOR_Z + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def start(name):
    x, y, h = L.START[name]
    return desk(x, y, math.degrees(h))


def target(name):
    x, y, h = L.TARGET[name]
    return desk(x, y, math.degrees(h))


START = {n: start(n) for n in L.DESKS}
DONE = {n: target(n) for n in L.DESKS}  # the demo's grid


def with_(base, **changes):
    d = dict(base)
    d.update({{"d1": D1, "d2": D2, "d3": D3, "d4": D4}[k]: v for k, v in changes.items()})
    return d


def settled(desks, t=None, n=WINDOW + 1):
    """Feed the same states n times (half a second still); return (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(desks)
    return r, t


def scored(desks) -> float:
    return settled(desks)[0].score


def why(desks, name):
    return settled(desks)[0].details[name]["why"]


class Geometry(unittest.TestCase):
    def test_outline_and_centre(self):
        d = desk(0.3, 0.0, 0.0)  # facing +x: the chair behind the desk, toward the arms
        self.assertAlmostEqual(d.high[0] - d.low[0], 0.077, places=9)  # 45 mm desk + 32 mm chair
        self.assertAlmostEqual(d.high[1] - d.low[1], 0.060, places=9)
        self.assertAlmostEqual(d.high[2] - d.low[2], 0.086, places=9)
        self.assertAlmostEqual(d.high[0], 0.3 + 0.0225, places=9)  # the desk's front edge
        self.assertAlmostEqual(d.low[0], 0.3 - 0.0545, places=9)  # the chair's back
        self.assertAlmostEqual(lowest(d), 0.0, places=12)
        np.testing.assert_allclose(desk_top_centre(d), [0.3, 0.0, FLOOR_Z + 0.069], atol=1e-12)
        # tipped onto its front edge: the desk top's middle moves with it
        tipped = desk(0.3, 0.0, 0.0, pitch=90.0)
        self.assertGreater(desk_top_centre(tipped)[0] - tipped.origin[0], 0.068)

    def test_facing_and_off_front(self):
        for h in (0.0, 30.0, -90.0, 180.0, -150.0):
            self.assertAlmostEqual(math.remainder(math.degrees(heading_rad(desk(0.3, 0.0, h))) - h, 360.0), 0.0)
        for h, off in ((0.0, 0.0), (15.0, 15.0), (-15.0, 15.0), (90.0, 90.0), (-90.0, 90.0), (180.0, 180.0),
                       (-170.0, 170.0)):
            self.assertAlmostEqual(off_front_deg(desk(0.3, 0.0, h)), off, places=6, msg=h)
        np.testing.assert_allclose(facing(desk(0.3, 0.0, 90.0)), (0.0, 1.0), atol=1e-12)
        self.assertAlmostEqual(tilt_deg(desk(0.3, 0.0, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(lowest(desk(0.3, 0.0, 0.0, lift=0.004)), 0.004, places=12)

    def test_nearest(self):
        centres = {n: desk_top_centre(d) for n, d in DONE.items()}
        dist, who = nearest(D1, centres)
        self.assertAlmostEqual(dist, 0.160, places=9)
        self.assertEqual(who, D2)
        dist, who = nearest(D2, {D2: centres[D2]})
        self.assertTrue(math.isinf(dist))
        self.assertIsNone(who)

    def test_start_layout_scores_zero(self):
        for n, d in START.items():
            self.assertTrue(Tracker.at_start(n, d), n)
        r, _ = settled(START)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "no_desk_set_facing_and_spaced")
        for n in L.DESKS:  # every desk sideways (90 degrees off the front), each row's two desks 100 mm apart
            self.assertTrue(r.details[n]["set"], n)
            self.assertEqual(r.details[n]["why"], "not_facing_the_front")
            self.assertAlmostEqual(r.details[n]["off_front_deg"], 90.0)
            self.assertAlmostEqual(r.details[n]["nearest_mm"], 100.0)
            self.assertFalse(r.details[n]["spaced"])
        self.assertEqual(r.details[D1]["nearest_desk"], D2)
        self.assertEqual(r.details[D3]["nearest_desk"], D4)
        for d in START.values():  # clear of the arms' start-pose shadow in the setup screenshot, on the floor
            self.assertTrue(np.all(np.abs([d.low[1], d.high[1]]) < 0.45))
            self.assertTrue(on_floor_area(d))
            self.assertGreaterEqual(d.low[0], L.ROOM_X[0])
            self.assertLessEqual(d.high[0], L.ROOM_X[1])
            self.assertLessEqual(max(abs(d.low[1]), abs(d.high[1])), L.ROOM_Y[1])
        # the two rows face each other across the middle: desks 1 and 2 face -y, desks 3 and 4 face +y
        self.assertAlmostEqual(math.degrees(heading_rad(START[D1])), -90.0)
        self.assertAlmostEqual(math.degrees(heading_rad(START[D4])), 90.0)

    def test_demo_target(self):
        r, _ = settled(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["credited"], list(L.DESKS))
        for n in L.DESKS:
            d = r.details[n]
            self.assertTrue(d["credited"], n)
            self.assertAlmostEqual(d["off_front_deg"], 0.0)
            self.assertAlmostEqual(d["nearest_mm"], 160.0)  # README: 160 mm front to back (35 over the rule)
        dists = [math.hypot(L.TARGET[a][0] - L.TARGET[b][0], L.TARGET[a][1] - L.TARGET[b][1])
                 for a, b in itertools.combinations(L.DESKS, 2)]
        self.assertEqual(sorted(round(v * 1000, 1) for v in dists), [160.0, 160.0, 300.0, 300.0, 340.0, 340.0])
        for d in DONE.values():  # every desk wholly on the floor, clear of the front wall
            self.assertGreaterEqual(d.low[0], L.ROOM_X[0] + 0.020)
            self.assertLessEqual(d.high[0], L.FRONT_FACE - 0.040)


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        self.assertEqual(scored(START), 0.0)  # the start
        self.assertEqual(scored(DONE), 1.0)  # the 2 x 2 grid, all facing the front
        backwards = with_(DONE, d2=desk(0.390, 0.150, 180.0))  # one desk facing the back
        self.assertEqual(scored(backwards), 0.75)
        self.assertEqual(why(backwards, D2), "not_facing_the_front")
        close = with_(DONE, d2=desk(0.330, 0.150, 0.0))  # desks 1 and 2 facing the front but 100 mm apart
        self.assertEqual(scored(close), 0.5)
        self.assertEqual(why(close, D1), "closer_than_125_mm_to_another_desk")
        self.assertEqual(why(close, D2), "closer_than_125_mm_to_another_desk")
        self.assertEqual(scored(with_(DONE, d3=START[D3], d4=START[D4])), 0.5)  # one side done, the other untouched
        self.assertEqual(scored(with_(START, d1=DONE[D1])), 0.25)  # one desk right
        self.assertEqual(scored(with_(START, d1=DONE[D1], d3=DONE[D3])), 0.5)
        self.assertEqual(scored(with_(DONE, d4=desk(0.390, -0.150, 90.0))), 0.75)  # one still sideways
        block = {D1: desk(0.230, 0.050, 0.0), D2: desk(0.330, 0.050, 0.0), D3: desk(0.230, -0.050, 0.0),
                 D4: desk(0.330, -0.050, 0.0)}
        self.assertEqual(scored(block), 0.0)  # all four facing the front, 100 mm apart: a tight block
        sideways = {n: desk(x, y, 90.0) for n, (x, y, _) in L.TARGET.items()}
        self.assertEqual(scored(sideways), 0.0)  # spread out but every one facing a side wall

    def test_facing_band_edges(self):
        for a, want in ((14.9, 1.0), (-14.9, 1.0), (15.1, 0.75), (-15.1, 0.75), (165.0, 0.75)):
            self.assertEqual(scored(with_(DONE, d1=desk(0.230, 0.150, a))), want, a)
            self.assertEqual(scored(with_(DONE, d4=desk(0.390, -0.150, a))), want, a)

    def test_spacing_edges(self):
        # desk 2 straight in front of desk 1: 125 mm is enough, a hair less is not (both lose)
        for gap, want in ((0.1251, 1.0), (0.1250, 1.0), (0.1249, 0.5)):
            self.assertEqual(scored(with_(DONE, d2=desk(0.230 + gap, 0.150, 0.0))), want, gap)
        # in every direction: across the room and on the diagonal too
        across = with_(DONE, d1=desk(0.230, 0.062, 0.0), d3=desk(0.230, -0.062, 0.0))  # 124 mm across
        self.assertEqual(scored(across), 0.5)
        self.assertEqual(why(across, D1), "closer_than_125_mm_to_another_desk")
        diag = with_(DONE, d2=desk(0.230 + 0.088, 0.150 - 0.088, 0.0))  # 124.5 mm on the diagonal
        self.assertEqual(scored(diag), 0.5)

    def test_spacing_counts_desks_that_are_not_set(self):
        # a tipped desk, one off the floor and one still held up all take up their places
        for other in (desk(0.230 + 0.100, 0.150, 0.0, roll=90.0), desk(0.230 + 0.100, 0.150, 0.0, lift=0.030),
                      desk(0.230 + 0.100, 0.150, 90.0)):
            ms = with_(DONE, d2=other)
            self.assertLess(nearest(D1, {n: desk_top_centre(d) for n, d in ms.items()})[0], 0.125)
            self.assertEqual(why(ms, D1), "closer_than_125_mm_to_another_desk")
            self.assertEqual(scored(ms), 0.5)

    def test_only_a_full_score_is_success(self):
        for ms, want in ((with_(START, d1=DONE[D1]), 0.25), (with_(DONE, d3=START[D3], d4=START[D4]), 0.5),
                         (with_(DONE, d2=desk(0.390, 0.150, 180.0)), 0.75)):
            r, _ = settled(ms)
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, want)
        self.assertTrue(settled(DONE)[0].success)

    def test_every_tier_is_reachable(self):
        seen = {scored(ms) for ms in (START, with_(START, d1=DONE[D1]), with_(DONE, d3=START[D3], d4=START[D4]),
                                      with_(DONE, d2=desk(0.390, 0.150, 180.0)), DONE)}
        self.assertEqual(seen, TIERS)

    def test_names_are_labels_only(self):
        swapped = {D1: DONE[D4], D2: DONE[D3], D3: DONE[D2], D4: DONE[D1]}
        self.assertEqual(scored(swapped), 1.0)
        rows = {n: desk(x, y, 0.0) for n, (x, y) in zip(L.DESKS, ((0.20, 0.0), (0.33, 0.0), (0.44, 0.10),
                                                                   (0.44, -0.10)))}
        self.assertEqual(scored(rows), 1.0)  # any layout that keeps the rule: 130 mm apart, 200 mm across
        line = {n: desk(0.300, y, 0.0) for n, y in zip(L.DESKS, (0.195, 0.065, -0.065, -0.195))}
        self.assertEqual(scored(line), 1.0)  # README: four in a line across the room, 130 mm apart

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(DONE)
        self.assertEqual(t.result(DONE).score, 1.0)
        self.assertEqual(t.result(DONE).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(11)
        for _ in range(300):
            ms = {n: desk(rng.uniform(0.10, 0.50), rng.uniform(-0.3, 0.3), rng.uniform(-180, 180),
                          lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)))
                  for n in L.DESKS}
            self.assertIn(scored(ms), TIERS)


class NotSet(unittest.TestCase):
    def test_tilt(self):
        self.assertEqual(scored(with_(DONE, d1=desk(0.230, 0.150, roll=9.9))), 1.0)
        tipped = with_(DONE, d1=desk(0.230, 0.150, roll=10.1))
        self.assertEqual(why(tipped, D1), "not_upright")
        self.assertEqual(scored(tipped), 0.75)
        on_side = with_(DONE, d1=desk(0.230, 0.150, 0.0, roll=90.0))
        self.assertEqual(why(on_side, D1), "not_upright")
        upside_down = with_(DONE, d1=desk(0.230, 0.150, 0.0, roll=180.0))
        self.assertEqual(why(upside_down, D1), "not_upright")
        self.assertEqual(scored(upside_down), 0.75)
        on_its_back = with_(DONE, d3=desk(0.230, -0.150, 0.0, pitch=-90.0))  # chair back down, desk top up
        self.assertEqual(why(on_its_back, D3), "not_upright")

    def test_on_the_floor(self):
        self.assertEqual(scored(with_(DONE, d2=desk(0.390, 0.150, lift=0.0029))), 1.0)
        self.assertEqual(why(with_(DONE, d2=desk(0.390, 0.150, lift=0.0031)), D2), "not_on_the_floor")
        self.assertEqual(why(with_(DONE, d2=desk(0.390, 0.150, lift=0.020)), D2), "not_on_the_floor")  # carried
        self.assertEqual(scored(with_(DONE, d2=desk(0.390, 0.150, lift=-0.0029))), 1.0)
        self.assertEqual(why(with_(DONE, d2=desk(0.390, 0.150, lift=-0.0031)), D2), "not_on_the_floor")

    def test_centre_over_the_floor(self):
        # past the floor's edge on the bare table (1 mm below the floor): the desk top's middle is off the floor
        out = with_(DONE, d1=desk(0.230, 0.290, 0.0, lift=-0.001))
        self.assertEqual(why(out, D1), "centre_off_the_floor")
        self.assertEqual(scored(out), 0.75)
        behind = with_(DONE, d3=desk(0.140, -0.150, 0.0, lift=-0.001))  # behind the floor's near edge
        self.assertEqual(why(behind, D3), "centre_off_the_floor")
        # the centre on the floor's edge counts (a desk half off it tilts under 2 degrees on the 1 mm step)
        self.assertEqual(scored(with_(DONE, d1=desk(0.230, 0.280, 0.0))), 1.0)
        # it is the desk top's middle that counts, not the middle of the whole unit (16 mm toward the chair):
        # the desk top over the floor with the chair hanging off the near edge counts ...
        hang = desk(0.160, 0.150, 0.0)
        self.assertLess(hang.pos[0], L.ROOM_X[0])
        self.assertEqual(scored(with_(DONE, d1=hang)), 1.0)
        # ... and a desk facing the back with its top off the floor does not, though its chair is on it
        back_off = desk(0.140, 0.150, 180.0)
        self.assertGreater(back_off.pos[0], L.ROOM_X[0])
        self.assertEqual(why(with_(DONE, d1=back_off), D1), "centre_off_the_floor")
        # leaning 9.5 degrees out over the side edge: the desk top's middle (seen from above) is past it, though the
        # frame's origin on the floor is not
        lean = desk(0.230, 0.272, 0.0, roll=-9.5)
        self.assertLess(lean.origin[1], L.ROOM_Y[1])
        self.assertGreater(desk_top_centre(lean)[1], L.ROOM_Y[1])
        self.assertEqual(why(with_(DONE, d1=lean), D1), "centre_off_the_floor")

    def test_moving(self):
        t = Tracker()
        for i in range(30):  # creeping 1 mm a step: 10 mm over 0.5 s
            r = t.update(with_(DONE, d3=desk(0.230 + 0.001 * i, -0.150)))
        self.assertEqual(r.details[D3]["why"], "moving")
        self.assertEqual(r.score, 0.75)


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for _ in range(WINDOW):
            self.assertEqual(t.update(DONE).score, 0.0)
        self.assertEqual(t.update(DONE).score, 1.0)

    def test_slow_creep_under_5_mm_is_still(self):
        t = Tracker()
        for i in range(40):  # 0.4 mm a step: 4 mm over 0.5 s
            r = t.update(with_(DONE, d1=desk(0.220 + 0.0004 * i, 0.150)))
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s
            r = t.update(with_(DONE, d1=desk(0.230 + (0.0006 if i % 2 else 0.0), 0.150)))
        self.assertEqual(r.details[D1]["why"], "moving")
        self.assertEqual(r.score, 0.75)

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        for i in range(30):  # 1 degree a step about its origin: 1.1 mm a step at its farthest corner
            a = float(i if i < 15 else 30 - i) - 7.0
            r = t.update(with_(DONE, d2=desk(0.390, 0.150, a)))
        self.assertEqual(r.details[D2]["why"], "moving")


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_floor_counts_as_set(self):
        # The scorer sees only the desks (task.json fidelity): held still 2 mm up at its place, a desk counts.
        self.assertEqual(scored({n: desk(*d.origin[:2], math.degrees(heading_rad(d)), lift=0.002)
                                 for n, d in DONE.items()}), 1.0)

    def test_tipping_a_neighbour_over_does_not_clear_its_place(self):
        # desk 2 knocked on its side 100 mm in front of desk 1: still in the way, so desk 1 is not spaced either
        ms = with_(DONE, d2=desk(0.330, 0.150, 0.0, roll=90.0))
        self.assertEqual(why(ms, D2), "not_upright")
        self.assertEqual(why(ms, D1), "closer_than_125_mm_to_another_desk")
        self.assertEqual(scored(ms), 0.5)

    def test_pushing_a_desk_out_of_the_room_earns_nothing_for_it(self):
        # three desks set and spaced, the fourth shoved off the floor onto the bare table: 0.75 at most
        ms = with_(DONE, d4=desk(0.300, -0.330, 0.0, lift=-0.001))
        self.assertEqual(why(ms, D4), "centre_off_the_floor")
        self.assertEqual(scored(ms), 0.75)

    def test_stacked_on_another_desk(self):
        top = desk(0.390, 0.150, 0.0, lift=L.OUTLINE[4])  # desk 1 lifted onto desk 2's top
        ms = with_(DONE, d1=top)
        self.assertEqual(why(ms, D1), "not_on_the_floor")
        self.assertEqual(why(ms, D2), "closer_than_125_mm_to_another_desk")
        self.assertEqual(scored(ms), 0.5)

    def test_facing_the_back_or_a_side_wall(self):
        for h in (180.0, 90.0, -90.0, 120.0, -45.0):
            self.assertEqual(why(with_(DONE, d1=desk(0.230, 0.150, h)), D1), "not_facing_the_front", h)

    def test_spreading_without_turning_earns_nothing(self):
        # the rows pulled apart (every desk 160 mm or more from the others) but still facing each other
        apart = {D1: desk(0.230, 0.200, -90.0), D2: desk(0.400, 0.200, -90.0), D3: desk(0.230, -0.200, 90.0),
                 D4: desk(0.400, -0.200, 90.0)}
        r, _ = settled(apart)
        self.assertEqual(r.score, 0.0)
        for n in L.DESKS:
            self.assertTrue(r.details[n]["spaced"])
            self.assertEqual(r.details[n]["why"], "not_facing_the_front")


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = settled(DONE)
        self.assertEqual(r.score, 1.0)
        n = {k: len(v) for k, v in t.history.items()}
        bad = []
        for k in ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high"):
            s = target(D1)
            setattr(s, k, np.full_like(np.asarray(getattr(s, k), dtype=float), np.nan))
            bad.append(s)
        z = target(D1)
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = target(D1)
        inf.origin = np.array([np.inf, 0.15, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            for name in L.DESKS:
                res = t.update(dict(DONE, **{name: s}))
                self.assertEqual(res.score, 0.0)
                self.assertEqual(res.details["zero_reason"], "invalid_state")
                self.assertEqual({k: len(v) for k, v in t.history.items()}, n)
        self.assertEqual(t.update(DONE).score, 1.0)  # the record was left as it was

    def test_missing_fields_or_desks(self):
        s = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))  # no origin
        self.assertFalse(valid(s))
        self.assertEqual(Tracker().update(dict(DONE, **{D1: s})).score, 0.0)
        self.assertEqual(Tracker().update({n: d for n, d in DONE.items() if n != D2}).score, 0.0)
        self.assertEqual(Tracker().update(dict(DONE, **{D3: None})).score, 0.0)
        self.assertEqual(Tracker().update({}).score, 0.0)
        self.assertEqual(Tracker().update(None).score, 0.0)
        self.assertEqual(Tracker().result(None).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = target(D1)
        s.quat_xyzw = s.quat_xyzw * 3.0
        self.assertEqual(scored(dict(DONE, **{D1: s})), 1.0)

    def test_facing_straight_up(self):
        s = desk(0.25, 0.10, 0.0, pitch=-90.0)  # standing on the chair's back, the candidate's way pointing up
        self.assertAlmostEqual(tilt_deg(s), 90.0, places=6)
        self.assertIsNone(facing(s))
        self.assertTrue(math.isnan(heading_rad(s)))
        self.assertTrue(math.isnan(off_front_deg(s)))
        r, _ = settled(dict(DONE, **{D1: s}))
        self.assertEqual(r.details[D1]["why"], "not_upright")
        self.assertFalse(r.details[D1]["faces_front"])
        self.assertIsNone(r.details[D1]["heading_deg"])
        self.assertEqual(r.score, 0.75)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = settled(DONE)
        r = t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history[D1]), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(DONE)
        t.update({n: desk(L.START[n][0] + 0.02, L.START[n][1] - 0.02, math.degrees(L.START[n][2]))
                  for n in L.DESKS})
        self.assertEqual(t.steps, 1)

    def test_reset_when_some_desks_had_not_moved(self):
        _, t = settled(with_(START, d1=DONE[D1], d3=DONE[D3]))
        t.update(START)  # two jump home in one step, two never left: a reset
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.START[D1][:2]), np.array([0.23, 0.15])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(with_(START, d1=desk(p[0], p[1], -90.0)))
        self.assertEqual(t.steps, 20)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(START)
        t.update(DONE)
        self.assertEqual(t.steps, WINDOW + 2)

    def test_explicit_reset(self):
        _, t = settled(DONE)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(DONE).score, 0.0)  # half a second of history again


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _boxes_overlap(a, b) -> float:
    """Overlap volume of two axis-aligned boxes given as (centre, size)."""
    (ca, sa), (cb, sb) = a, b
    d = [min(ca[i] + sa[i] / 2, cb[i] + sb[i] / 2) - max(ca[i] - sa[i] / 2, cb[i] - sb[i] / 2) for i in range(3)]
    return max(d[0], 0.0) * max(d[1], 0.0) * max(d[2], 0.0)


def _boxes_touch(a, b) -> bool:
    """Two axis-aligned boxes (centre, size) share at least a face patch (touching, not overlapping)."""
    (ca, sa), (cb, sb) = a, b
    d = [min(ca[i] + sa[i] / 2, cb[i] + sb[i] / 2) - max(ca[i] - sa[i] / 2, cb[i] - sb[i] / 2) for i in range(3)]
    return sum(abs(v) < 1e-12 for v in d) == 1 and sum(v > 1e-9 for v in d) == 2


def _rect_gap(p, q) -> float:
    """Top-view distance between two convex polygons (0 if they overlap) by separating axes: the largest gap."""
    best = -math.inf
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.hypot(*e) or 1.0)
            a, b = p @ n, q @ n
            best = max(best, b.min() - a.max(), a.min() - b.max())
    return max(best, 0.0)


def _footprint(x, y, h):
    """A desk's outline seen from above (4, 2), counter-clockwise, origin at (x, y), facing heading h (rad)."""
    back, front, right, left, _ = L.OUTLINE
    c, s = math.cos(h), math.sin(h)
    pts = [(-back, -right), (front, -right), (front, left), (-back, left)]
    return np.array([(x + c * u - s * v, y + s * u + c * v) for u, v in pts])


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the
    demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        sys.path.insert(0, str(_stub_dir().parent))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit
        cls.specs = {o.name: o for o in cls.task.DESK_OBJECTS}

    def sim(self, desks):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in desks.items()}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], list(L.DESKS))
        self.assertTrue(all(o.mass <= 0.30 for o in loose))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        f = t.FLOOR
        np.testing.assert_allclose([f.pos[0] - f.size[0] / 2, f.pos[0] + f.size[0] / 2], L.ROOM_X, atol=1e-12)
        np.testing.assert_allclose([f.pos[1] - f.size[1] / 2, f.pos[1] + f.size[1] / 2], L.ROOM_Y, atol=1e-12)
        self.assertAlmostEqual(f.pos[2] + f.size[2] / 2, FLOOR_Z, places=12)
        w = t.FRONT_WALL
        self.assertAlmostEqual(w.pos[0] - w.size[0] / 2, L.FRONT_FACE)
        self.assertAlmostEqual(w.size[2], 0.060)
        for o in (f, w):
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, TOP, places=12)
        boxes = [o for o in t.ROOM]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                self.assertLess(_boxes_overlap((a.pos, a.size), (b.pos, b.size)), 1e-15, (a.name, b.name))
        # the whiteboard and the clock hang on the wall's face (touching it), above the floor
        for o in (t.WHITEBOARD, t.CLOCK[0]):
            self.assertAlmostEqual(o.pos[0] + o.size[0] / 2, L.FRONT_FACE)
            self.assertGreater(o.pos[2] - o.size[2] / 2, FLOOR_Z)
            self.assertLess(o.pos[2] + o.size[2] / 2, TOP + L.WALL_H)
        self.assertAlmostEqual(t.CLOCK[1].pos[0] + t.CLOCK[1].size[0] / 2, t.CLOCK[0].pos[0] - t.CLOCK[0].size[0] / 2)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            cen, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None),
                                                                                getattr(o, "yaw", 0.0))
            hx = abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2
            hy = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            if cen[0] - hx < 0.2 - 1e-9:
                self.assertLessEqual(abs(cen[1]) + hy, 0.45 + 1e-9, o.name)

    def test_desk_geometry_matches_the_scorer(self):
        back, front, right, left, top = L.OUTLINE
        boxes = {n: (c, s) for n, c, s, _ in L.DESK_SHAPES}
        for name, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-back, -right, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [front, left, top], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_U), 0.004)  # the grip within 4 mm of the centre of mass
            self.assertLess(abs(com[1]), 0.0005)
            self.assertGreater(com[0], -back)  # over the unit's footprint: it stands on its frames and chair legs
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.START[name][:2], FLOOR_Z], atol=1e-12)
            self.assertTrue(Tracker.at_start(name, rest))
        for i, a in enumerate(L.DESK_SHAPES):
            for b in L.DESK_SHAPES[i + 1:]:
                self.assertLess(_boxes_overlap((a[1], a[2]), (b[1], b[2])), 1e-15, (a[0], b[0]))
        # one solid piece: every shape touches another, and all are connected through touching faces
        names = [s[0] for s in L.DESK_SHAPES]
        links = {n: {m for m in names if m != n and _boxes_touch(boxes[n], boxes[m])} for n in names}
        seen, todo = set(), ["top"]
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(links[n] - seen)
        self.assertEqual(seen, set(names))
        # the desk top: 60 x 45 mm, 72 mm up, its middle the scorer's desk-top centre; the side frames flush with
        # its edges, standing on the floor
        (tc, ts) = boxes["top"]
        np.testing.assert_allclose(tc, L.DESK_TOP_CENTRE, atol=1e-12)
        np.testing.assert_allclose(ts, (0.045, 0.060, 0.006), atol=1e-12)
        self.assertAlmostEqual(tc[2] + ts[2] / 2, 0.072)
        for k in ("frame_left", "frame_right", "chair_left", "chair_right"):
            self.assertAlmostEqual(boxes[k][0][2] - boxes[k][1][2] / 2, 0.0)
        self.assertAlmostEqual(boxes["frame_left"][0][1] + boxes["frame_left"][1][1] / 2, ts[1] / 2)
        # the chair is behind the desk (toward -u), its back at the far end: the chair-to-desk direction is +u
        self.assertLess(boxes["seat"][0][0], -ts[0] / 2)
        self.assertAlmostEqual(boxes["back"][0][0] - boxes["back"][1][0] / 2, -back)
        self.assertAlmostEqual(boxes["back"][0][2] + boxes["back"][1][2] / 2, top)
        # the pads (20 mm wide, centred on the grip) hold the side frames only, within their 45 mm depth
        fc, fs = boxes["frame_left"]
        self.assertLessEqual(self.task.GRIP_U + 0.010, fc[0] + fs[0] / 2 + 1e-12)
        self.assertGreaterEqual(self.task.GRIP_U - 0.010, fc[0] - fs[0] / 2 + 0.002)
        colour = {n: c for n, _, _, c in L.DESK_SHAPES}
        self.assertEqual((colour["top"], colour["seat"], colour["back"]), ("beech", "chair_blue", "chair_blue"))
        sim = self.sim(DONE)
        self.assertAlmostEqual(float(sim[D1].low[2]), FLOOR_Z, places=9)
        self.assertEqual(scored(sim), 1.0)

    def test_offline_state_path(self):
        st = {}
        for name, spec in self.specs.items():
            c, _, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st[name] = spec.offline_state(box)
            self.assertEqual(st[name].source, "offline")
            np.testing.assert_allclose(st[name].origin, [*L.START[name][:2], FLOOR_Z], atol=1e-9)
        self.assertEqual(scored(st), 0.0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start_ = self.sim(START)
        self.assertFalse(self.task.success(start_))
        self.assertEqual(self.task.score(start_), 0.0)
        done = self.sim(DONE)
        for _ in range(WINDOW):
            self.task.success(done)
        self.assertTrue(self.task.success(done))
        self.assertEqual(self.task.score(done), 1.0)
        self.assertEqual(self.task.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(self.task.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertEqual(self.task.score({k: v for k, v in done.items() if k != D4}), 0.0)  # a missing desk: 0

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "zero_reason", "valid_state", "credited", "origins",
                          "quats", "low_z", "vels", "source", *L.DESKS):
                    self.assertIn(k, row)
                for k in ("set", "credited", "why", "tilt_deg", "lowest_mm", "path_mm", "origin_mm", "centre_mm",
                          "heading_deg", "off_front_deg", "faces_front", "nearest_mm", "nearest_desk", "spaced"):
                    self.assertIn(k, row[D1])
                self.assertAlmostEqual(row[D1]["nearest_mm"], 160.0)
                self.assertEqual(row[D1]["centre_mm"], [230.0, 150.0])
                self.assertEqual(row[D1]["why"], "moving")  # one step seen: not yet still
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(DONE)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        approach_z = t.GRIP_Z + t.APPROACH
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and let-go: 1.5 s, standing still
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(tuple(prev.pos), tuple(w.pos), (arm, w.label))
                if w.pos[2] < prev.pos[2] and w.pos[2] < approach_z - 1e-9:  # slow last descents
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.03 / 1.5 + 1e-9, (arm, w.label))
                # no wrist turn of more than 90 degrees in one segment (slerp would be ambiguous at 180)
                self.assertLessEqual(abs(math.remainder(w.yaw - prev.yaw, 2 * math.pi)), math.pi / 2 + 1e-9)
            grips = [w for w in wps if w.label == "grip across the side frames"]
            puts = [w for w in wps if w.label == "let go"]
            self.assertEqual(len(grips), 2)
            self.assertEqual(len(puts), 2)
            for name, grip, put in zip(t.SIDE[arm], grips, puts):
                np.testing.assert_allclose(grip.pos[:2], t.grip_point(L.START[name][:2], L.START[name][2]),
                                           atol=1e-12)
                np.testing.assert_allclose(put.pos[:2], t.grip_point(L.TARGET[name][:2], L.TARGET[name][2]),
                                           atol=1e-12)
                # the grip closes across the desk (yaw = heading mod 180) and the wrist turns it to its place
                for w, h in ((grip, L.START[name][2]), (put, L.TARGET[name][2])):
                    self.assertAlmostEqual(math.remainder(w.yaw - h, math.pi), 0.0, places=12)
                i, j = wps.index(grip), wps.index(put)
                turned = sum(math.remainder(b.yaw - a.yaw, 2 * math.pi) for a, b in zip(wps[i:j], wps[i + 1:j + 1]))
                self.assertAlmostEqual(math.remainder(L.START[name][2] + turned - L.TARGET[name][2], 2 * math.pi), 0)
                # a quarter turn, anticlockwise on the left, clockwise on the right (never round to face the back)
                self.assertAlmostEqual(math.degrees(turned), 90.0 if arm == "left" else -90.0)
                self.assertAlmostEqual(put.pos[2] - grip.pos[2], t.PLACE_UP)
                self.assertAlmostEqual(grip.pos[2], t.GRIP_Z)
        self.assertAlmostEqual(t.GRIP, -0.026)  # 52 mm commanded on the 60 mm frames
        self.assertAlmostEqual(-2 * t.APPROACH_OPEN - 0.060, 0.016)  # 8 mm clear each side going down
        # finger carriages (57 mm above the TCP, +-71 mm along the closing axis open) and the housing (~70 mm above
        # it) over the tallest part of any desk, the chair backs (README: 30 mm)
        self.assertGreaterEqual(round(t.GRIP_Z - TOP + 0.057 - (L.FLOOR + L.OUTLINE[4]), 6), 0.030)
        self.assertGreater(t.GRIP_Z - TOP - 0.013, L.FLOOR + 0.040)  # pad tips on the frames, above the chair seat
        self.assertGreater(t.ABOVE_Z - TOP - 0.013 - (L.FLOOR + L.OUTLINE[4]), 0.019)  # travel over the chair backs

    @staticmethod
    def _sample(wps, dt=0.05):
        """(t, TCP position, yaw, finger, label) of an arm's plan, interpolated as ArmPlan does (smoothstep)."""
        out, t, prev = [], 0.0, None
        for w in wps:
            if prev is not None:
                n = max(1, int(round(w.duration / dt)))
                for i in range(1, n + 1):
                    s = i / n
                    k = 3 * s * s - 2 * s * s * s
                    p = (1 - k) * np.array(prev.pos) + k * np.array(w.pos)
                    yaw = prev.yaw + k * math.remainder(w.yaw - prev.yaw, 2 * math.pi)
                    fing = (1 - s) * prev.finger + s * w.finger
                    out.append((t + s * w.duration, p, yaw, fing, w.label))
            t += w.duration
            prev = w
        return out

    def _desk_track(self, arm):
        """For each sample of the arm's plan: {desk: (x, y, heading, carried)} for its two desks: standing on its
        start until it is lifted, carried rigidly in the grip (turning with the wrist) until it is lowered, then
        standing at its place."""
        t = self.task
        wps = t.demo()[arm]
        ends = np.cumsum([w.duration for w in wps])
        lifts = [i for i, w in enumerate(wps) if w.label == "lift 20 mm"]
        lows = [i for i, w in enumerate(wps) if w.label == "lower it slowly"]
        track = []
        for tt, p, yaw, _, _ in self._sample(wps):
            row = {}
            for k, name in enumerate(t.SIDE[arm]):
                t0, t1 = ends[lifts[k] - 1], ends[lows[k]]
                if tt < t0 - 1e-9:
                    row[name] = (*L.START[name], False)
                elif tt > t1 + 1e-9:
                    row[name] = (*L.TARGET[name], False)
                else:
                    h = L.START[name][2] + math.remainder(yaw - wps[lifts[k]].yaw, 2 * math.pi)
                    row[name] = (p[0] - t.GRIP_U * math.cos(h), p[1] - t.GRIP_U * math.sin(h), h, True)
            track.append(row)
        return track

    def test_carries_stay_clear_of_the_other_desks(self):
        """Replays both arms' plans on one clock and checks every carried desk's outline against every other desk
        (standing or carried) at the same moment; and that each carried desk ends exactly at its place."""
        tl, tr = self._desk_track("left"), self._desk_track("right")
        self.assertEqual(len(tl), len(tr))
        worst, carried_seen = 1.0, set()
        for a, b in zip(tl, tr):
            row = {**a, **b}
            for n, (x, y, h, carried) in row.items():
                if not carried:
                    continue
                carried_seen.add(n)
                fp = _footprint(x, y, h)
                for m, (ox, oy, oh, _) in row.items():
                    if m != n:
                        worst = min(worst, _rect_gap(fp, _footprint(ox, oy, oh)))
                # inside the room and clear of the front wall while carried
                self.assertGreater(fp[:, 0].min(), L.ROOM_X[0])
                self.assertLess(fp[:, 0].max(), L.FRONT_FACE - 0.040)
        self.assertEqual(carried_seen, set(L.DESKS))
        self.assertGreater(worst, 0.035)  # README: more than 35 mm from every other desk (39 mm)
        for row in (tl[-1], tr[-1]):
            for n, (x, y, h, carried) in row.items():
                self.assertFalse(carried)
                np.testing.assert_allclose((x, y), L.TARGET[n][:2], atol=1e-12)

    def test_tcps_stay_apart(self):
        plan = self.task.demo()
        sl, sr = self._sample(plan["left"]), self._sample(plan["right"])
        self.assertEqual(len(sl), len(sr))
        self.assertGreaterEqual(min(abs(a[1][1] - b[1][1]) for a, b in zip(sl, sr)), 0.20)

    def test_pads_stay_clear_of_the_other_desks(self):
        """The pads (8 mm thick, 20 mm wide, tips 13 mm below the TCP): going down beside a desk (76 mm open) and
        letting it go, the pads stay more than 20 mm from every other desk, wherever it stands then (on its start or
        at its place: both are checked), and well short of the front wall; gripping, they sit on the side frames'
        outer faces, within the frames' 45 mm depth."""
        t = self.task

        def pads(xy, yaw, gap):
            d = np.array([-math.sin(yaw), math.cos(yaw)])  # the fingers close along this
            a = np.array([math.cos(yaw), math.sin(yaw)])
            out = []
            for s in (-1.0, 1.0):
                inner, outer = s * gap / 2, s * (gap / 2 + 0.008)
                out.append(np.array([np.asarray(xy) + d * w + a * u for w in (inner, outer) for u in (-0.010, 0.010)]))
            return out

        def local(pt, pose):
            x, y, h = pose
            dx, dy = pt[0] - x, pt[1] - y
            return (dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h))

        worst = 1.0
        for arm in ("left", "right"):
            wps = t.demo()[arm]
            for k, name in enumerate(t.SIDE[arm]):
                down = [w for w in wps if w.label == "down the last 30 mm slowly"][k]
                put = [w for w in wps if w.label == "let go"][k]
                for w in (down, put):
                    for pad in pads(w.pos[:2], w.yaw, -2 * t.APPROACH_OPEN):
                        self.assertLess(pad[:, 0].max(), L.FRONT_FACE - 0.050)
                        for other in L.DESKS:
                            if other == name:
                                continue
                            for (ox, oy, oh) in (L.START[other], L.TARGET[other]):
                                g = _rect_gap(pad[[0, 1, 3, 2]], _footprint(ox, oy, oh))
                                worst = min(worst, g)
                                self.assertGreater(g, 0.020, (arm, name, other, w.label))
                for w, pose in ((down, L.START[name]), (put, L.TARGET[name])):
                    for pad in pads(w.pos[:2], w.yaw, 0.060):  # closed on the 60 mm frames
                        uv = np.array([local(p, pose) for p in pad])
                        self.assertAlmostEqual(float(np.min(np.abs(uv[:, 1]))), 0.030, places=9)
                        self.assertGreaterEqual(uv[:, 0].min(), -0.0225 + 0.002 - 1e-9)
                        self.assertLessEqual(uv[:, 0].max(), 0.0225 + 1e-9)
        self.assertGreater(worst, 0.020)

    def test_one_arm_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_one_arm_alone_scores_half(self):
        # what one arm's moves leave: its own two desks at their places, the other row where it started
        left_only = with_(START, d1=DONE[D1], d2=DONE[D2])
        right_only = with_(START, d3=DONE[D3], d4=DONE[D4])
        self.assertEqual(scored(left_only), 0.5)
        self.assertEqual(scored(right_only), 0.5)
        # half-way through (the first desk placed, the second not yet moved) the placed desk already counts
        self.assertEqual(scored(with_(START, d1=DONE[D1], d3=DONE[D3])), 0.5)
        self.assertGreater(math.hypot(L.TARGET[D1][0] - L.START[D2][0], L.TARGET[D1][1] - L.START[D2][1]), 0.135)

    def test_instruction_states_the_rule_and_matches_everywhere(self):
        ins = self.task.INSTRUCTION
        for s in ("exam", "1:10", "four desks", "chair fixed behind it", "face the front", "whiteboard", "clock",
                  "within 15 degrees", "125 mm (1.25 m) centre to centre", "in any direction", "middles of the desk",
                  "upright on the grey floor"):
            self.assertIn(s, ins)
        self.assertLessEqual(len(ins), 490)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], ins)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{ins}"'), 2)

    def test_task_json_fields(self):
        spec = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                  "fidelity", "reset_notes", "scores"):
            self.assertTrue(spec.get(k), k)
        self.assertEqual(spec["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(spec[k]), n, k)
        self.assertIn("1,200 steps", spec["run_limit"])
        self.assertIn("demo", spec["scores"])

    def test_task_json_states_the_codes_numbers(self):
        spec = json.loads((HERE / "task.json").read_text())
        sc, su = spec["scoring"], spec["setup"]
        for s in ("10 degrees or less", "within 3 mm", "over the grey floor", "5 mm over the last 0.5 s",
                  "within 15 degrees", "at least 125 mm", "set or not", "0.25", "0.5", "0.75", "1.0", "3 s"):
            self.assertIn(s, sc)
        self.assertEqual(L.TILT_MAX_DEG, 10.0)
        self.assertEqual(L.FLOOR_TOL, 0.003)
        self.assertEqual(L.STILL_PATH, 0.005)
        self.assertEqual(L.FACING_DEG, 15.0)
        self.assertEqual(L.SPACING_MIN, 0.125)
        self.assertEqual(L.CREDIT, 0.25)
        for s in ("y +30.5", "x 15.0 to 46.0 cm", "y -28.0 to +28.0 cm", "x 46.0 to 47.0 cm", "60 mm tall",
                  "80 g", "60 mm wide", "45 mm deep", "72 mm", "x 26.0 cm, y +20.0 cm", "x 36.0 cm, y +20.0 cm",
                  "x 26.0 cm, y -20.0 cm", "x 36.0 cm, y -20.0 cm", "100 mm apart"):
            self.assertIn(s, su)
        nums = {float(v) for v in re.findall(r"(\d+(?:\.\d+)?) mm", sc)}
        self.assertTrue({3.0, 5.0, 125.0} <= nums)

    def test_one_arm_reach(self):
        """README 'One arm alone': straight down, at grip and push heights, neither arm reaches any point of the
        other row's desks where they start, at any yaw (joint 6 allows +-120 degrees; 30 degree steps here, 15 in the
        README's wider check); while each arm reaches its own desks' grips at their starts and their places."""
        import yamkin as K
        back, front, right, left, _ = L.OUTLINE
        for arm, others in (("right", (D1, D2)), ("left", (D3, D4))):
            base = K.BASES["station"][arm]
            for other in others:
                x, y, h = L.START[other]
                for u in (-back, 0.0, front):
                    for v in (-right, 0.0, left):
                        px, py = x + u * math.cos(h) - v * math.sin(h), y + u * math.sin(h) + v * math.cos(h)
                        for z in (0.013, 0.050):
                            for yaw in range(-120, 121, 30):
                                cell, _ = K.reach_cell((px, py, TOP + z), math.radians(yaw), base, TOP, None)
                                self.assertEqual(cell, 0, (arm, other, u, v, z, yaw))
            for own in self.task.SIDE[arm]:
                for (ox, oy, oh) in (L.START[own], L.TARGET[own]):
                    gx, gy = self.task.grip_point((ox, oy), oh)
                    cell, _ = K.reach_cell((gx, gy, self.task.GRIP_Z), self.task._yaw(arm, oh), base, TOP, None)
                    self.assertEqual(cell, 2, (arm, own))


if __name__ == "__main__":
    unittest.main()
