"""Offline tests for scoring.py and the hooks in task.py: synthetic air-mover states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
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
    WINDOW, Tracker, airflow, angle_into_wall_deg, heading_rad, in_room, lowest, outlet_gap, sense_of, tilt_deg,
    valid, wall_credit, wall_of,
)

TOP = L.TOP
FLOOR_Z = L.TOP + L.FLOOR
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}
M1, M2, M3, M4 = L.MOVERS


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


def mover(x, y, heading=0.0, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A mover's state as mechkit reports it: frame origin (the middle of its base) at (x, y), airflow turned `heading`
    degrees and tilted by `pitch`/`roll`, resting with its lowest point `lift` above the carpet; low/high from its
    outline box."""
    back, front, right, left, top = L.OUTLINE
    q = quat(heading, pitch, roll)
    local = np.array([(u, v, z) for u in (-back, front) for v in (-right, left) for z in (0.0, top)])
    pts = local @ rot(q).T
    o = np.array([x, y, FLOOR_Z + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def heading_for(wall, sense, angle):
    """The airflow heading (degrees) that blows into `wall` at `angle` degrees going round the room in `sense`."""
    if wall == "left":
        return angle if sense == "clockwise" else 180.0 - angle
    return angle - 180.0 if sense == "clockwise" else -angle


def at_wall(wall="left", x=0.190, sense="clockwise", angle=30.0, gap=0.0077, **kw):
    """A mover blowing into `wall` at `angle` degrees in `sense`, its centre at x, placed in y so that its outlet's
    nearest point is `gap` from the wall's face."""
    h = heading_for(wall, sense, angle)
    probe = mover(x, 0.0, h, **kw)
    c = np.array([probe.origin + rot(probe.quat_xyzw) @ np.array([u, v, z]) for u in L.OUTLET_BOX[0]
                  for v in L.OUTLET_BOX[1] for z in L.OUTLET_BOX[2]])
    y = L.LEFT_FACE - gap - c[:, 1].max() if wall == "left" else L.RIGHT_FACE + gap - c[:, 1].min()
    s = mover(x, y, h, **kw)
    assert abs(outlet_gap(s, wall) - gap) < 1e-9, (outlet_gap(s, wall), gap)
    return s


def start(name):
    x, y, h = L.START[name]
    return mover(x, y, math.degrees(h))


def target(name):
    x, y, h = L.TARGET[name]
    return mover(x, y, math.degrees(h))


START = {n: start(n) for n in L.MOVERS}
DONE = {n: target(n) for n in L.MOVERS}  # the demo's clockwise set-up
ANTI = {  # all four set anticlockwise
    M1: at_wall("left", 0.190, "anticlockwise"), M2: at_wall("left", 0.420, "anticlockwise"),
    M3: at_wall("right", 0.190, "anticlockwise"), M4: at_wall("right", 0.420, "anticlockwise"),
}
PARALLEL = {  # both pairs blowing toward the back wall: two parallel streams, no loop
    M1: at_wall("left", 0.190, "clockwise"), M2: at_wall("left", 0.420, "clockwise"),
    M3: at_wall("right", 0.190, "anticlockwise"), M4: at_wall("right", 0.420, "anticlockwise"),
}


def with_(base, **changes):
    d = dict(base)
    d.update({{"m1": M1, "m2": M2, "m3": M3, "m4": M4}[k]: v for k, v in changes.items()})
    return d


def settled(movers, t=None, n=WINDOW + 1):
    """Feed the same states n times (half a second still); return (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(movers)
    return r, t


def scored(movers) -> float:
    return settled(movers)[0].score


def why(movers, name):
    return settled(movers)[0].details[name]["why"]


class Geometry(unittest.TestCase):
    def test_outline_and_outlet(self):
        m = mover(0.3, 0.0, 0.0)  # airflow +x
        self.assertAlmostEqual(m.high[0] - m.low[0], 0.035, places=9)  # 30 mm body + 5 mm outlet
        self.assertAlmostEqual(m.high[1] - m.low[1], 0.026, places=9)
        self.assertAlmostEqual(m.high[2] - m.low[2], 0.036, places=9)
        self.assertAlmostEqual(m.high[0], 0.3 + 0.020, places=9)  # the outlet is at the front
        self.assertAlmostEqual(lowest(m), 0.0, places=12)

    def test_airflow_wall_angle_sense(self):
        for h in (0.0, 30.0, -90.0, 180.0, -150.0):
            self.assertAlmostEqual(math.remainder(math.degrees(heading_rad(mover(0.3, 0.0, h))) - h, 360.0), 0.0)
        cases = (  # heading -> (wall, angle into it, sense)
            (30.0, ("left", 30.0, "clockwise")), (150.0, ("left", 30.0, "anticlockwise")),
            (-150.0, ("right", 30.0, "clockwise")), (-30.0, ("right", 30.0, "anticlockwise")),
            (60.0, ("left", 60.0, "clockwise")), (-100.0, ("right", 80.0, "clockwise")),
            (-80.0, ("right", 80.0, "anticlockwise")), (165.0, ("left", 15.0, "anticlockwise")),
        )
        for h, (w, a, s) in cases:
            m = mover(0.3, 0.0, h)
            self.assertEqual(wall_of(m), w, h)
            self.assertAlmostEqual(angle_into_wall_deg(m), a, places=6, msg=h)
            self.assertEqual(sense_of(m), s, h)
        self.assertIsNone(wall_of(mover(0.3, 0.0, 0.0)))  # straight along the room
        self.assertIsNone(sense_of(mover(0.3, 0.0, 0.0)))
        self.assertIsNone(sense_of(mover(0.3, 0.0, -90.0)))  # straight across: no way round
        self.assertEqual(wall_of(mover(0.3, 0.0, -90.0)), "right")
        self.assertAlmostEqual(angle_into_wall_deg(mover(0.3, 0.0, -90.0)), 90.0)
        self.assertAlmostEqual(tilt_deg(mover(0.3, 0.0, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(lowest(mover(0.3, 0.0, 0.0, lift=0.004)), 0.004, places=12)

    def test_the_outlet_is_the_nearest_point_to_the_wall(self):
        # the outlet is as wide as the body and in front of it, so at any angle into the wall its wall-side corner is
        # the mover's nearest point: the gap is also the clearance of the whole mover
        for a in (5.0, 15.0, 30.0, 45.0, 60.0, 85.0):
            for wall in ("left", "right"):
                for sense in ("clockwise", "anticlockwise"):
                    m = at_wall(wall, 0.3, sense, a, gap=0.010)
                    nearest = (L.LEFT_FACE - m.high[1]) if wall == "left" else (m.low[1] - L.RIGHT_FACE)
                    self.assertAlmostEqual(nearest, 0.010, places=9, msg=(a, wall, sense))

    def test_start_layout_scores_zero(self):
        for n, m in START.items():
            self.assertTrue(Tracker.at_start(n, m), n)
        r, _ = settled(START)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "no_mover_set")
        for n in L.MOVERS:  # all blowing straight at the right wall (90 degrees), movers 3 and 4 35 mm from it
            self.assertEqual(r.details[n]["why"], "angle_out_of_band")
            self.assertEqual(r.details[n]["wall"], "right")
            self.assertAlmostEqual(r.details[n]["angle_deg"], 90.0)
            self.assertIsNone(r.details[n]["sense"])
        for n in (M3, M4):
            self.assertAlmostEqual(r.details[n]["gap_mm"], 35.0)
        for n in (M1, M2):
            self.assertAlmostEqual(r.details[n]["gap_mm"], 295.0)
        for m in START.values():  # clear of the arms' start-pose shadow in the setup screenshot, inside the room
            self.assertTrue(np.all(np.abs([m.low[1], m.high[1]]) < 0.45))
            self.assertTrue(in_room(m))

    def test_demo_target(self):
        r, _ = settled(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["best_sense"], "clockwise")
        self.assertEqual(r.details["credited"], {"clockwise": 4, "anticlockwise": 0})
        for n in L.MOVERS:
            d = r.details[n]
            self.assertTrue(d["set"], n)
            self.assertAlmostEqual(d["angle_deg"], 30.0)
            self.assertAlmostEqual(d["gap_mm"], 7.7)  # README: 7.7 mm (band 0 to 15)
        for side in ("left", "right"):
            self.assertAlmostEqual(r.details["walls"]["clockwise"][side]["spacing_mm"], 230.0)
            self.assertEqual(r.details["walls"]["clockwise"][side]["credit"], 2)
        # the outlet's wall-side corner is 21.3 mm from the centre toward the wall at 30 degrees
        self.assertAlmostEqual(20 * math.sin(math.radians(30)) + 13 * math.cos(math.radians(30)), 21.26, places=2)


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        self.assertEqual(scored(START), 0.0)  # the start
        self.assertEqual(scored(DONE), 1.0)  # all four set clockwise
        self.assertEqual(scored(ANTI), 1.0)  # ... or anticlockwise
        steep = with_(DONE, m4=at_wall("right", 0.420, "clockwise", angle=60.0))  # one at 60 degrees into its wall
        self.assertEqual(scored(steep), 0.75)
        self.assertEqual(why(steep, M4), "angle_out_of_band")
        self.assertEqual(scored(PARALLEL), 0.5)  # two parallel streams toward the back wall, no loop
        r, _ = settled(PARALLEL)
        self.assertEqual(r.details["credited"], {"clockwise": 2, "anticlockwise": 2})
        bunched = with_(DONE, m3=at_wall("right", 0.320, "clockwise"))  # the right pair 100 mm apart
        self.assertEqual(scored(bunched), 0.75)
        self.assertEqual(settled(bunched)[0].details["walls"]["clockwise"]["right"]["credit"], 1)
        self.assertEqual(scored(with_(DONE, m3=START[M3], m4=START[M4])), 0.5)  # one wall only
        self.assertEqual(scored(with_(START, m1=DONE[M1])), 0.25)  # one mover
        far = with_(DONE, m2=at_wall("left", 0.420, "clockwise", gap=0.020))  # outlet 20 mm off the wall
        self.assertEqual(scored(far), 0.75)
        self.assertEqual(why(far, M2), "outlet_more_than_15_mm_from_the_wall")
        flat = {n: at_wall(w, x, "clockwise", angle=10.0) for n, w, x in
                ((M1, "left", 0.19), (M2, "left", 0.42), (M3, "right", 0.19), (M4, "right", 0.42))}
        self.assertEqual(scored(flat), 0.0)  # all four nearly parallel to their walls
        square = {n: at_wall(w, x, "clockwise", angle=89.0) for n, w, x in
                  ((M1, "left", 0.19), (M2, "left", 0.42), (M3, "right", 0.19), (M4, "right", 0.42))}
        self.assertEqual(scored(square), 0.0)  # pointed straight at the walls

    def test_angle_band_edges(self):
        for a, want in ((14.9, 0.75), (15.1, 1.0), (44.9, 1.0), (45.1, 0.75)):
            self.assertEqual(scored(with_(DONE, m1=at_wall("left", 0.190, "clockwise", angle=a))), want, a)
            self.assertEqual(scored(with_(DONE, m4=at_wall("right", 0.420, "clockwise", angle=a))), want, a)

    def test_gap_edges(self):
        for g, want in ((0.0149, 1.0), (0.0151, 0.75), (0.0, 1.0), (-0.001, 1.0)):  # touching (or 1 mm into) counts
            self.assertEqual(scored(with_(DONE, m2=at_wall("left", 0.420, "clockwise", gap=g))), want, g)
            self.assertEqual(scored(with_(DONE, m3=at_wall("right", 0.190, "clockwise", gap=g))), want, g)

    def test_spacing_edges(self):
        for x2, want in ((0.130 + 0.1899, 0.75), (0.130 + 0.1901, 1.0), (0.130 + 0.3049, 1.0), (0.130 + 0.3051, 0.75)):
            ms = with_(DONE, m1=at_wall("left", 0.130, "clockwise"), m2=at_wall("left", x2, "clockwise"))
            self.assertEqual(scored(ms), want, x2)
            ms = with_(DONE, m3=at_wall("right", 0.130, "clockwise"), m4=at_wall("right", x2, "clockwise"))
            self.assertEqual(scored(ms), want, x2)
        # the band's upper end can be passed in this room: the walls are 350 mm (18.4 ft) long
        wide = with_(DONE, m1=at_wall("left", 0.125, "clockwise"), m2=at_wall("left", 0.440, "clockwise"))
        self.assertEqual(scored(wide), 0.75)
        self.assertEqual(settled(wide)[0].details["walls"]["clockwise"]["left"]["spacing_mm"], 315.0)
        self.assertEqual(wall_credit([]), (0, None))
        self.assertEqual(wall_credit([0.2]), (1, None))
        self.assertEqual(wall_credit([0.2, 0.25])[0], 1)
        self.assertEqual(wall_credit([0.2, 0.25, 0.42])[0], 2)  # three on a wall: a pair in band credits two
        self.assertEqual(wall_credit([0.2, 0.25, 0.30])[0], 1)

    def test_a_wall_credits_at_most_two(self):
        # three set on the left wall, the fourth unmoved: still 0.5
        three = with_(START, m1=at_wall("left", 0.150, "clockwise"), m2=at_wall("left", 0.300, "clockwise"),
                      m3=at_wall("left", 0.440, "clockwise"))
        self.assertEqual(scored(three), 0.5)
        self.assertEqual(settled(three)[0].details["walls"]["clockwise"]["left"]["credit"], 2)
        four = with_(three, m4=at_wall("left", 0.225, "clockwise"))  # all four on one wall
        self.assertEqual(scored(four), 0.5)

    def test_mixed_senses_count_the_larger(self):
        # one clockwise and one anticlockwise on each wall: two agree whichever way is counted
        mixed = with_(DONE, m2=at_wall("left", 0.420, "anticlockwise"), m4=at_wall("right", 0.420, "anticlockwise"))
        self.assertEqual(scored(mixed), 0.5)
        three_one = with_(DONE, m4=at_wall("right", 0.420, "anticlockwise"))  # one blowing against the others
        self.assertEqual(scored(three_one), 0.75)
        r, _ = settled(three_one)
        self.assertEqual(r.details["credited"], {"clockwise": 3, "anticlockwise": 1})

    def test_only_a_full_score_is_success(self):
        for ms, want in ((PARALLEL, 0.5), (with_(START, m1=DONE[M1]), 0.25),
                         (with_(DONE, m4=at_wall("right", 0.420, "clockwise", angle=60.0)), 0.75)):
            r, _ = settled(ms)
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, want)
        self.assertTrue(settled(DONE)[0].success)

    def test_every_tier_is_reachable(self):
        seen = {scored(ms) for ms in (START, with_(START, m1=DONE[M1]), PARALLEL,
                                      with_(DONE, m3=at_wall("right", 0.320, "clockwise")), DONE)}
        self.assertEqual(seen, TIERS)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(DONE)
        self.assertEqual(t.result(DONE).score, 1.0)
        self.assertEqual(t.result(DONE).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(11)
        for _ in range(300):
            ms = {n: mover(rng.uniform(0.10, 0.49), rng.uniform(-0.2, 0.2), rng.uniform(-180, 180),
                           lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)))
                  for n in L.MOVERS}
            self.assertIn(scored(ms), TIERS)


class NotPlaced(unittest.TestCase):
    def test_tilt(self):
        self.assertEqual(scored(with_(DONE, m1=at_wall("left", 0.190, roll=9.9))), 1.0)
        tipped = with_(DONE, m1=at_wall("left", 0.190, roll=10.1))
        self.assertEqual(why(tipped, M1), "not_upright")
        self.assertEqual(scored(tipped), 0.75)
        on_side = with_(DONE, m1=mover(0.19, 0.15, 30.0, roll=90.0))
        self.assertEqual(why(on_side, M1), "not_upright")
        upside_down = with_(DONE, m1=mover(0.19, 0.15, 30.0, roll=180.0))
        self.assertEqual(why(upside_down, M1), "not_upright")
        self.assertEqual(scored(upside_down), 0.75)

    def test_on_the_carpet(self):
        self.assertEqual(scored(with_(DONE, m2=at_wall("left", 0.420, lift=0.0029))), 1.0)
        self.assertEqual(why(with_(DONE, m2=at_wall("left", 0.420, lift=0.0031)), M2), "not_on_the_carpet")
        self.assertEqual(why(with_(DONE, m2=at_wall("left", 0.420, lift=0.030)), M2), "not_on_the_carpet")  # carried
        self.assertEqual(scored(with_(DONE, m2=at_wall("left", 0.420, lift=-0.0029))), 1.0)
        self.assertEqual(why(with_(DONE, m2=at_wall("left", 0.420, lift=-0.0031)), M2), "not_on_the_carpet")

    def test_inside_the_room(self):
        # past the open side (on the bare table, 1 mm below the carpet): not inside the room
        out = with_(DONE, m1=at_wall("left", 0.115, lift=-0.001))
        self.assertEqual(why(out, M1), "not_inside_the_room")
        self.assertEqual(scored(with_(DONE, m1=at_wall("left", 0.121))), 1.0)  # inside (299 mm from mover 2)

    def test_moving(self):
        t = Tracker()
        for i in range(30):  # creeping 1 mm a step: 10 mm over 0.5 s
            r = t.update(with_(DONE, m3=at_wall("right", 0.190 + 0.001 * i)))
        self.assertEqual(r.details[M3]["why"], "moving")
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
            r = t.update(with_(DONE, m1=at_wall("left", 0.180 + 0.0004 * i)))
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s
            r = t.update(with_(DONE, m1=at_wall("left", 0.190 + (0.0006 if i % 2 else 0.0))))
        self.assertEqual(r.details[M1]["why"], "moving")
        self.assertEqual(r.score, 0.75)

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        for i in range(30):  # 1 degree a step about its middle: 0.4 mm a step at its corners
            a = 30.0 + (i if i < 15 else 30 - i)
            r = t.update(with_(DONE, m2=at_wall("left", 0.420, angle=a)))
        self.assertEqual(r.details[M2]["why"], "moving")


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_carpet_counts_as_placed(self):
        # The scorer sees only the movers (task.json fidelity): held still 2 mm up at the place, a mover counts.
        self.assertEqual(scored({n: mover(*m.origin[:2], math.degrees(heading_rad(m)), lift=0.002)
                                 for n, m in DONE.items()}), 1.0)

    def test_outside_the_room_beyond_a_wall(self):
        # beyond the left wall on the bare table, blowing away from the room: the outlet is 'past' the face but the
        # mover is not in the room
        out = mover(0.300, 0.215, 30.0, lift=-0.001)
        self.assertLess(outlet_gap(out, "left"), 0.0)
        self.assertEqual(why(with_(DONE, m1=out), M1), "not_inside_the_room")

    def test_stacked_on_another_mover(self):
        top = at_wall("left", 0.420, lift=0.036)
        ms = with_(DONE, m1=top)  # mover 1 on top of mover 2
        self.assertEqual(why(ms, M1), "not_on_the_carpet")
        self.assertEqual(scored(ms), 0.75)

    def test_set_backward(self):
        # its back against the wall, blowing out into the room at 30 degrees: it blows toward the other wall
        h = heading_for("left", "clockwise", 30.0) + 180.0
        m = mover(0.19, 0.160, h)
        self.assertEqual(wall_of(m), "right")
        self.assertEqual(why(with_(DONE, m1=m), M1), "outlet_more_than_15_mm_from_the_wall")

    def test_outlet_toward_the_back_wall_only(self):
        # in the back corner, blowing along the left wall into the back wall: angle 5 degrees to the side wall
        m = at_wall("left", 0.440, angle=5.0)
        self.assertEqual(why(with_(DONE, m2=m), M2), "angle_out_of_band")

    def test_sitting_on_a_wall(self):
        m = mover(0.300, 0.190, 30.0, lift=L.WALL_H - L.FLOOR)
        self.assertEqual(why(with_(DONE, m1=m), M1), "not_on_the_carpet")

    def test_a_pair_on_one_wall_cannot_count_for_the_other(self):
        # both left movers on the left wall blowing clockwise, both right movers also on the left wall blowing
        # anticlockwise between them: each sense has one pair on one wall only
        ms = {M1: at_wall("left", 0.150, "clockwise"), M2: at_wall("left", 0.440, "clockwise"),
              M3: at_wall("left", 0.250, "anticlockwise"), M4: at_wall("left", 0.350, "anticlockwise")}
        self.assertEqual(scored(ms), 0.5)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = settled(DONE)
        self.assertEqual(r.score, 1.0)
        n = {k: len(v) for k, v in t.history.items()}
        bad = []
        for k in ("origin", "quat_xyzw", "pos", "lin_vel", "low"):
            s = target(M1)
            setattr(s, k, np.full_like(np.asarray(getattr(s, k), dtype=float), np.nan))
            bad.append(s)
        z = target(M1)
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = target(M1)
        inf.origin = np.array([np.inf, 0.15, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            for name in L.MOVERS:
                res = t.update(dict(DONE, **{name: s}))
                self.assertEqual(res.score, 0.0)
                self.assertEqual(res.details["zero_reason"], "invalid_state")
                self.assertEqual({k: len(v) for k, v in t.history.items()}, n)
        self.assertEqual(t.update(DONE).score, 1.0)  # the record was left as it was

    def test_missing_fields_or_movers(self):
        s = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))  # no origin
        self.assertFalse(valid(s))
        self.assertEqual(Tracker().update(dict(DONE, **{M1: s})).score, 0.0)
        self.assertEqual(Tracker().update({n: m for n, m in DONE.items() if n != M2}).score, 0.0)
        self.assertEqual(Tracker().update(dict(DONE, **{M3: None})).score, 0.0)
        self.assertEqual(Tracker().update({}).score, 0.0)
        self.assertEqual(Tracker().update(None).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = target(M1)
        s.quat_xyzw = s.quat_xyzw * 3.0
        self.assertEqual(scored(dict(DONE, **{M1: s})), 1.0)

    def test_outlet_pointing_straight_up(self):
        s = mover(0.25, 0.10, 0.0, pitch=-90.0)
        self.assertAlmostEqual(tilt_deg(s), 90.0, places=6)
        self.assertIsNone(airflow(s))
        self.assertIsNone(wall_of(s))
        self.assertTrue(math.isnan(angle_into_wall_deg(s)))
        self.assertTrue(math.isinf(outlet_gap(s)))
        r, _ = settled(dict(DONE, **{M1: s}))
        self.assertEqual(r.details[M1]["why"], "not_upright")
        self.assertEqual(r.score, 0.75)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = settled(DONE)
        r = t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history[M1]), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(DONE)
        t.update({n: mover(L.START[n][0] + 0.02, L.START[n][1] - 0.02, -90.0) for n in L.MOVERS})
        self.assertEqual(t.steps, 1)

    def test_reset_when_some_movers_had_not_moved(self):
        _, t = settled(with_(START, m1=DONE[M1], m3=DONE[M3]))
        t.update(START)  # two jump home in one step, two never left: a reset
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.START[M1][:2]), np.array([0.26, 0.15])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(with_(START, m1=mover(p[0], p[1], -90.0)))
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
    """A mover's outline seen from above (4, 2), counter-clockwise, origin at (x, y), airflow heading h (rad)."""
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
        cls.specs = {o.name: o for o in cls.task.MOVER_OBJECTS}

    def sim(self, movers):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in movers.items()}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], list(L.MOVERS))
        self.assertTrue(all(o.mass <= 0.30 for o in loose))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        # the carpet fills the room; the walls stand on the table beside it, flush with its edges, not overlapping
        c = t.CARPET
        np.testing.assert_allclose([c.pos[0] - c.size[0] / 2, c.pos[0] + c.size[0] / 2], L.ROOM_X, atol=1e-12)
        np.testing.assert_allclose([c.pos[1] - c.size[1] / 2, c.pos[1] + c.size[1] / 2], L.ROOM_Y, atol=1e-12)
        self.assertAlmostEqual(c.pos[2] + c.size[2] / 2, FLOOR_Z, places=12)
        boxes = [t.CARPET, *t.WALL_BOXES]
        for i, a in enumerate(boxes):
            self.assertAlmostEqual(a.pos[2] - a.size[2] / 2, TOP, places=12)
            for b in boxes[i + 1:]:
                self.assertLess(_boxes_overlap((a.pos, a.size), (b.pos, b.size)), 1e-15, (a.name, b.name))
        by = {o.name: o for o in t.WALL_BOXES}
        self.assertAlmostEqual(by["left_wall"].pos[1] - by["left_wall"].size[1] / 2, L.LEFT_FACE)
        self.assertAlmostEqual(by["right_wall"].pos[1] + by["right_wall"].size[1] / 2, L.RIGHT_FACE)
        self.assertAlmostEqual(by["back_wall"].pos[0] - by["back_wall"].size[0] / 2, L.BACK_FACE)
        for o in t.WALL_BOXES:
            self.assertAlmostEqual(o.size[2], 0.010)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            cen, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None),
                                                                                getattr(o, "yaw", 0.0))
            hx = abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2
            hy = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            if cen[0] - hx < 0.2 - 1e-9:
                self.assertLessEqual(abs(cen[1]) + hy, 0.45 + 1e-9, o.name)

    def test_mover_geometry_matches_the_scorer(self):
        back, front, right, left, top = L.OUTLINE
        for name, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-back, -right, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [front, left, top], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_U), 0.002)  # the grip within 2 mm of the COM
            self.assertLess(abs(com[1]), 0.0005)
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.START[name][:2], FLOOR_Z], atol=1e-12)
            self.assertTrue(Tracker.at_start(name, rest))
        boxes = {n: (c, s) for n, c, s, _ in L.MOVER_SHAPES}
        for i, a in enumerate(L.MOVER_SHAPES):
            for b in L.MOVER_SHAPES[i + 1:]:
                self.assertLess(_boxes_overlap((a[1], a[2]), (b[1], b[2])), 1e-15, (a[0], b[0]))
        # the scorer's outlet box is the outlet shape: in front of the body, as wide as it, at the bottom
        (oc, os_) = boxes["outlet"]
        np.testing.assert_allclose([oc[0] - os_[0] / 2, oc[0] + os_[0] / 2], L.OUTLET_BOX[0], atol=1e-12)
        np.testing.assert_allclose([oc[1] - os_[1] / 2, oc[1] + os_[1] / 2], L.OUTLET_BOX[1], atol=1e-12)
        np.testing.assert_allclose([oc[2] - os_[2] / 2, oc[2] + os_[2] / 2], L.OUTLET_BOX[2], atol=1e-12)
        # the housing: blue behind, black over its front 5 mm, meeting with no gap; its front face is the outlet's back
        (bc, bs), (fc, fs) = boxes["body"], boxes["front"]
        self.assertAlmostEqual(bc[0] + bs[0] / 2, fc[0] - fs[0] / 2)
        self.assertAlmostEqual(bc[0] - bs[0] / 2, -L.OUTLINE[0])
        self.assertAlmostEqual(fc[0] + fs[0] / 2, L.OUTLET_BOX[0][0])
        self.assertAlmostEqual(fs[0], 0.005)
        for k in (1, 2):
            self.assertAlmostEqual(bs[k], fs[k])
            self.assertAlmostEqual(bc[k], fc[k])
        self.assertAlmostEqual(boxes["body"][1][1], os_[1])
        colour = {n: c for n, _, _, c in L.MOVER_SHAPES}
        self.assertEqual((colour["body"], colour["front"], colour["outlet"]), ("mover_blue", "black", "black"))
        # the pads (20 mm wide, centred on the grip) hold the blue box only
        self.assertLessEqual(self.task.GRIP_U + 0.010, bc[0] + bs[0] / 2 + 1e-12)
        self.assertGreaterEqual(self.task.GRIP_U - 0.010, bc[0] - bs[0] / 2 - 1e-12)
        # the handle sits between the pads with room to spare: 10 mm wide against the 26 mm grip
        self.assertLess(boxes["handle"][1][1] + 0.012, boxes["body"][1][1])
        sim = self.sim(DONE)
        self.assertAlmostEqual(float(sim[M1].low[2]), FLOOR_Z, places=9)
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
        self.assertEqual(self.task.score({k: v for k, v in done.items() if k != M4}), 0.0)  # a missing mover: 0

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "valid_state", "credited", "best_sense",
                          "walls", "origins", "quats", "low_z", "vels", "source", *L.MOVERS):
                    self.assertIn(k, row)
                for k in ("set", "why", "tilt_deg", "lowest_mm", "path_mm", "origin_mm", "heading_deg", "wall",
                          "angle_deg", "gap_mm", "sense"):
                    self.assertIn(k, row[M1])
                self.assertAlmostEqual(row[M1]["gap_mm"], 7.7)
                self.assertEqual(row[M1]["why"], "moving")  # one step seen: not yet still
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
                if prev.finger != w.finger and t.GRIP in (prev.finger, w.finger):  # every close and let-go: 1.5 s,
                    # standing still (opening from 50 to 60 mm in the air, away from any mover, is not one)
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(tuple(prev.pos), tuple(w.pos), (arm, w.label))
                if w.pos[2] < prev.pos[2] and w.pos[2] < approach_z - 1e-9:  # slow last descents
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.03 / 1.5 + 1e-9, (arm, w.label))
                # no wrist turn of more than 90 degrees in one segment (slerp would be ambiguous at 180)
                self.assertLessEqual(abs(math.remainder(w.yaw - prev.yaw, 2 * math.pi)), math.pi / 2 + 1e-9)
            grips = [w for w in wps if w.label == "grip across the body"]
            puts = [w for w in wps if w.label == "let go"]
            self.assertEqual(len(grips), 2)
            self.assertEqual(len(puts), 2)
            for name, grip, put in zip(t.SIDE[arm], grips, puts):
                np.testing.assert_allclose(grip.pos[:2], t.grip_point(L.START[name][:2], L.START[name][2]),
                                           atol=1e-12)
                np.testing.assert_allclose(put.pos[:2], t.grip_point(L.TARGET[name][:2], L.TARGET[name][2]),
                                           atol=1e-12)
                # the grip closes across the mover (yaw = heading mod 180) and the wrist turns it to its place
                for w, h in ((grip, L.START[name][2]), (put, L.TARGET[name][2])):
                    self.assertAlmostEqual(math.remainder(w.yaw - h, math.pi), 0.0, places=12)
                i, j = wps.index(grip), wps.index(put)
                turned = sum(math.remainder(b.yaw - a.yaw, 2 * math.pi) for a, b in zip(wps[i:j], wps[i + 1:j + 1]))
                self.assertAlmostEqual(math.remainder(L.START[name][2] + turned - L.TARGET[name][2], 2 * math.pi), 0)
                self.assertAlmostEqual(put.pos[2] - grip.pos[2], t.PLACE_UP)
                self.assertAlmostEqual(grip.pos[2], t.GRIP_Z)
            # the turns: the left arm's movers 120 degrees anticlockwise, the right arm's 60 degrees clockwise
            i, j = wps.index(grips[0]), wps.index(puts[0])
            turned = sum(math.remainder(b.yaw - a.yaw, 2 * math.pi) for a, b in zip(wps[i:j], wps[i + 1:j + 1]))
            self.assertAlmostEqual(math.degrees(turned), 120.0 if arm == "left" else -60.0)
        self.assertAlmostEqual(t.GRIP, -0.009)  # 18 mm commanded on the 26 mm body
        self.assertGreaterEqual(t.GRIP_Z - TOP - 0.013 - L.WALL_H, 0.005)  # pad tips 7 mm over the walls
        self.assertGreater(t.GRIP_Z - TOP + 0.035 - (L.FLOOR + L.OUTLINE[4]), 0.025)  # gripper body over the handle
        self.assertGreater(t.GRIP_Z - TOP - 0.013, L.FLOOR + 0.012)  # pad tips above the outlet's top

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

    def _mover_track(self, arm):
        """For each sample of the arm's plan: {mover: (x, y, heading, carried)} for its two movers: standing on its
        start until it is lifted, carried rigidly in the grip (turning with the wrist) until it is lowered, then
        standing at its place."""
        t = self.task
        wps = t.demo()[arm]
        ends = np.cumsum([w.duration for w in wps])
        lifts = [i for i, w in enumerate(wps) if w.label == "lift 30 mm"]
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

    def test_carries_stay_clear_of_the_other_movers(self):
        """Replays both arms' plans on one clock and checks every carried mover's outline against every other mover
        (standing or carried) at the same moment; and that each carried mover ends exactly at its place."""
        tl, tr = self._mover_track("left"), self._mover_track("right")
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
        self.assertEqual(carried_seen, set(L.MOVERS))
        self.assertGreater(worst, 0.060)  # README: more than 60 mm from every other mover (64 mm)
        for row in (tl[-1], tr[-1]):
            for n, (x, y, h, carried) in row.items():
                self.assertFalse(carried)
                np.testing.assert_allclose((x, y), L.TARGET[n][:2], atol=1e-12)

    def test_tcps_stay_apart(self):
        plan = self.task.demo()
        sl, sr = self._sample(plan["left"]), self._sample(plan["right"])
        self.assertEqual(len(sl), len(sr))
        self.assertGreaterEqual(min(abs(a[1][1] - b[1][1]) for a, b in zip(sl, sr)), 0.20)

    def test_pads_stay_clear_of_the_walls_and_the_other_mover(self):
        """The pads (8 mm thick, 20 mm wide, tips 13 mm below the TCP): at every place the wall-side pad is more than
        5 mm from the wall's face while gripping and its tips 7 mm over the wall's top; going down beside a mover
        (60 mm open) the pads stay clear of the other mover and of the back wall."""
        t = self.task

        def pads(xy, yaw, gap):
            d = np.array([-math.sin(yaw), math.cos(yaw)])  # the fingers close along this
            a = np.array([math.cos(yaw), math.sin(yaw)])
            out = []
            for s in (-1.0, 1.0):
                inner, outer = s * gap / 2, s * (gap / 2 + 0.008)
                out.append(np.array([np.asarray(xy) + d * w + a * u for w in (inner, outer) for u in (-0.010, 0.010)]))
            return out

        for arm in ("left", "right"):
            wps = t.demo()[arm]
            for k, name in enumerate(t.SIDE[arm]):
                put = [w for w in wps if w.label == "let go"][k]
                for gap, min_face in ((0.026, 0.005), (0.050, -0.010)):  # gripping; open to let go (over the wall)
                    for pad in pads(put.pos[:2], put.yaw, gap):
                        face = L.LEFT_FACE - pad[:, 1].max() if arm == "left" else pad[:, 1].min() - L.RIGHT_FACE
                        self.assertGreater(face, min_face, (arm, name, gap))
                down = [w for w in wps if w.label == "down the last 30 mm slowly"][k]
                for pad in pads(down.pos[:2], down.yaw, -2 * t.APPROACH_OPEN):
                    self.assertLess(pad[:, 0].max(), L.BACK_FACE - 0.010)
                    for other in L.MOVERS:
                        if other == name:
                            continue
                        ox, oy, oh = L.START[other]
                        self.assertGreater(_rect_gap(pad[[0, 1, 3, 2]], _footprint(ox, oy, oh)), 0.020,
                                           (arm, name, other))
        self.assertGreaterEqual(t.GRIP_Z - TOP - 0.013 - L.WALL_H, 0.007 - 1e-12)

    def test_one_arm_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule_and_matches_everywhere(self):
        ins = self.task.INSTRUCTION
        for s in ("wet", "1:16", "wall nearest you is left out", "four air movers", "two along each side wall",
                  "black outlet", "no more than 15 mm", "15 to 45 degrees", "190 to 305 mm (10 to 16 ft)",
                  "same way round the room"):
            self.assertIn(s, ins)
        self.assertLessEqual(len(ins), 490)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], ins)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{ins}"'), 2)

    def test_task_json_fields(self):
        spec = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "context", "setup", "scoring", "run_limit", "fidelity",
                  "reset_notes", "scores"):
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
        for s in ("10 degrees or less", "within 3 mm", "inside the room", "5 mm over the last 0.5 s",
                  "15 to 45 degrees", "15 mm or less", "190 to 305 mm", "at most two", "0.25", "0.5", "0.75", "1.0",
                  "3 s", "clockwise", "y +18.5 cm", "y -18.5 cm"):
            self.assertIn(s, sc)
        self.assertEqual(L.TILT_PLACED_DEG, 10.0)
        self.assertEqual(L.FLOOR_TOL, 0.003)
        self.assertEqual(L.STILL_PATH, 0.005)
        self.assertEqual(L.ANGLE_DEG, (15.0, 45.0))
        self.assertEqual(L.GAP_MAX, 0.015)
        self.assertEqual(L.SPACING, (0.190, 0.305))
        self.assertEqual(L.PER_WALL, 2)
        self.assertEqual(L.CREDIT, 0.25)
        for s in ("y +30.5", "x 12.0 to 47.0 cm", "y -18.5 to +18.5 cm", "10 mm thick and 10 mm tall",
                  "x 12.0 to 48.0 cm", "y +18.5 to +19.5", "y -19.5 to -18.5", "x 47.0 to 48.0", "25 g", "30 mm",
                  "26 mm", "5 mm", "12 mm", "20 x 10 x 6", "x 33.0 cm, y +13.0 cm", "x 42.0 cm, y +13.0 cm",
                  "x 33.0 cm, y -13.0 cm", "x 42.0 cm, y -13.0 cm"):
            self.assertIn(s, su)
        nums = {float(v) for v in re.findall(r"(\d+(?:\.\d+)?) mm", sc)}
        self.assertTrue({3.0, 5.0, 15.0, 305.0} <= nums)

    def test_one_arm_reach(self):
        """README 'One arm alone': straight down, at grip heights, neither arm reaches any point of the other side's
        movers where they start, at any yaw (joint 6 allows +-120 degrees; 30 degree steps here, 15 in the README's
        wider check); and on the far wall an arm reaches the place of a set mover only near the open side (not at
        x 0.31 m or more, where a second mover 190 mm along would have to stand)."""
        import yamkin as K
        back, front, right, left, _ = L.OUTLINE
        for arm, others in (("right", (M1, M2)), ("left", (M3, M4))):
            base = K.BASES["station"][arm]
            for other in others:
                x, y, h = L.START[other]
                for u in (-back, 0.0, front):
                    for v in (-right, left):
                        px, py = x + u * math.cos(h) - v * math.sin(h), y + u * math.sin(h) + v * math.cos(h)
                        for z in (0.013, 0.026):
                            for yaw in range(-120, 121, 30):
                                cell, _ = K.reach_cell((px, py, TOP + z), math.radians(yaw), base, TOP, None)
                                self.assertEqual(cell, 0, (arm, other, u, v, z, yaw))
            # ... while each arm reaches its own movers' grips at their starts and their places
            for own in self.task.SIDE[arm]:
                for (ox, oy, oh) in (L.START[own], L.TARGET[own]):
                    gx, gy = self.task.grip_point((ox, oy), oh)
                    cell, _ = K.reach_cell((gx, gy, self.task.GRIP_Z), self.task._yaw(arm, oh), base, TOP, None)
                    self.assertEqual(cell, 2, (arm, own))
        # the far wall: a set mover's centre is at least 146.7 mm out from the room's middle (45 degrees, 15 mm gap),
        # so any point of its housing is at least 127 mm out (a grip across its width at its back, 136 mm); the left
        # arm cannot reach that at x >= 0.31 m
        c45 = L.LEFT_FACE - L.GAP_MAX - (0.020 * math.sin(math.pi / 4) + 0.013 * math.cos(math.pi / 4))
        self.assertAlmostEqual(c45, 0.1467, places=4)
        self.assertAlmostEqual(c45 - 0.015 * math.sin(math.pi / 4) - 0.013 * math.cos(math.pi / 4), 0.127, places=3)
        base = K.BASES["station"]["left"]
        for x in (0.31, 0.35, 0.40, 0.45):
            for y in (-0.127, -0.136, -0.160):
                for z in (0.017, 0.030):
                    for yaw in range(-120, 121, 30):
                        cell, _ = K.reach_cell((x, y, TOP + z), math.radians(yaw), base, TOP, None)
                        self.assertEqual(cell, 0, (x, y, z, yaw))


if __name__ == "__main__":
    unittest.main()
