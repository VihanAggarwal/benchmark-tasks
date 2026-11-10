"""Offline tests for scoring.py and the hooks in task.py: synthetic vehicle states, no simulator.
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
    CAR_POLY, WINDOW, Tracker, ambulance_measures, angle_to_lanes_deg, engine_measures, footprint,
    front_off_downstream_deg, heading_rad, left_side_y, lowest, poly_distance, tilt_deg, valid,
)

TOP = L.TOP
ROAD_Z = L.TOP + L.ROAD_TOP
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}


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


def veh(name, x, y, heading=0.0, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A vehicle state as mechkit reports it: frame origin (the middle of its base) at (x, y), turned `heading`
    degrees and tilted by `pitch`/`roll`, resting with its lowest point `lift` above the road; low/high from its
    outline box."""
    back, front, right, left, top = L.OUTLINE[name]
    q = quat(heading, pitch, roll)
    local = np.array([(u, v, z) for u in (-back, front) for v in (-right, left) for z in (0.0, top)])
    pts = local @ rot(q).T
    o = np.array([x, y, ROAD_Z + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def eng(x, y, heading, **kw):
    return veh("fire_engine", x, y, heading, **kw)


def amb(x, y, heading, **kw):
    return veh("ambulance", x, y, heading, **kw)


def engine_at(buffer=0.0872, heading=-135.0, x=0.232, **kw):
    """The engine turned `heading`, its origin at x, placed in y so that its most downstream point is `buffer`
    upstream of the car's upstream end."""
    probe = eng(x, 0.0, heading, **kw)
    s = eng(x, L.CAR_UP_Y + buffer - footprint("fire_engine", probe)[:, 1].min(), heading, **kw)
    assert abs(engine_measures(s)["buffer"] - buffer) < 1e-9
    return s


def ambulance_at(gap=0.0865, heading=-90.0, x=0.210, **kw):
    """The ambulance turned `heading`, its origin at x, placed in y so that its most upstream point is `gap`
    downstream of the car's downstream end."""
    probe = amb(x, 0.0, heading, **kw)
    s = amb(x, L.CAR_DOWN_Y - gap - footprint("ambulance", probe)[:, 1].max(), heading, **kw)
    assert abs(ambulance_measures(s)["gap"] - gap) < 1e-9
    return s


START_E = eng(*L.START["fire_engine"][:2], math.degrees(L.START["fire_engine"][2]))
START_A = amb(*L.START["ambulance"][:2], math.degrees(L.START["ambulance"][2]))
ENGINE = eng(*L.TARGET["fire_engine"][:2], math.degrees(L.TARGET["fire_engine"][2]))
AMB = amb(*L.TARGET["ambulance"][:2], math.degrees(L.TARGET["ambulance"][2]))
LANE_BLOCK = engine_at(0.090, heading=-90.0, x=0.210)  # straight in lane 1 behind the car: 0.25
NOSE_IN = ambulance_at(0.0865, heading=90.0)  # in the lane, nose to the car: 0.25
ON_SHOULDER = ambulance_at(0.0865, heading=-90.0, x=0.170)  # right gap and heading, but on the shoulder: 0


def settled(engine, ambulance, t=None, n=WINDOW + 1):
    """Feed the same pair n times (half a second still); return (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(engine, ambulance)
    return r, t


def scored(engine, ambulance=AMB) -> float:
    return settled(engine, ambulance)[0].score


def why(engine=ENGINE, ambulance=AMB, which="fire_engine"):
    return settled(engine, ambulance)[0].details[which]["why"]


def zero_reason(engine=ENGINE, ambulance=AMB):
    return settled(engine, ambulance)[0].details["zero_reason"]


class Geometry(unittest.TestCase):
    def test_outlines(self):
        fp = footprint("fire_engine", eng(0.3, 0.1, 0.0))  # heading +x: 125 mm along x, 32 across
        self.assertAlmostEqual(fp[:, 0].max() - fp[:, 0].min(), 0.125, places=9)
        self.assertAlmostEqual(fp[:, 1].max() - fp[:, 1].min(), 0.032, places=9)
        self.assertAlmostEqual(fp[:, 1].max(), 0.1 + 0.0165, places=9)  # the pump panel is on its left
        fp = footprint("ambulance", amb(0.3, 0.1, -90.0))
        self.assertAlmostEqual(fp[:, 1].max() - fp[:, 1].min(), 0.088, places=9)
        self.assertAlmostEqual(fp[:, 0].max() - fp[:, 0].min(), 0.030, places=9)

    def test_heading_angle_panel(self):
        for h in (0.0, 30.0, -90.0, 180.0, -135.0):
            self.assertAlmostEqual(math.remainder(math.degrees(heading_rad(eng(0.3, 0.1, h))) - h, 360.0), 0.0)
        for h, a in ((-90.0, 0.0), (90.0, 0.0), (0.0, 90.0), (-135.0, 45.0), (135.0, 45.0), (-120.0, 30.0),
                     (-150.0, 60.0), (60.0, 30.0)):
            self.assertAlmostEqual(angle_to_lanes_deg(eng(0.3, 0.1, h)), a, places=6, msg=h)
        # the left side faces heading + 90 degrees: downstream (-y) for -135 and +135, upstream for -45 and +45
        self.assertLess(left_side_y(eng(0.3, 0.1, -135.0)), 0.0)
        self.assertLess(left_side_y(eng(0.3, 0.1, 135.0)), 0.0)
        self.assertGreater(left_side_y(eng(0.3, 0.1, -45.0)), 0.0)
        self.assertGreater(left_side_y(eng(0.3, 0.1, 45.0)), 0.0)
        self.assertAlmostEqual(front_off_downstream_deg(amb(0.2, -0.2, -90.0)), 0.0, places=9)
        self.assertAlmostEqual(front_off_downstream_deg(amb(0.2, -0.2, 90.0)), 180.0, places=9)
        self.assertAlmostEqual(front_off_downstream_deg(amb(0.2, -0.2, -70.0)), 20.0, places=9)
        self.assertAlmostEqual(tilt_deg(eng(0.3, 0.1, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(lowest(eng(0.3, 0.1, 0.0, lift=0.004)), 0.004, places=12)

    def test_the_car(self):
        c = np.array(L.car_corners())
        self.assertAlmostEqual(L.CAR_UP_Y, c[:, 1].max())
        self.assertAlmostEqual(L.CAR_UP_Y, -0.0005, places=3)  # task.json: upstream end y -0.05 cm
        self.assertAlmostEqual(L.CAR_DOWN_Y, -0.0595, places=3)  # downstream end y -5.95 cm
        self.assertTrue(np.all((c[:, 0] > L.LANE1[0]) & (c[:, 0] < L.LANE1[1])))  # in lane 1
        self.assertEqual(len(CAR_POLY), 4)

    def test_lane_coverage_needs_an_angle(self):
        # straight, the engine's outline is 32 mm across: it cannot span both lanes, only cover 32 mm of lane 1
        m = engine_measures(LANE_BLOCK)
        self.assertAlmostEqual(m["crash_lane_cover"], 0.032, places=9)
        self.assertGreater(m["x_lo"], L.BOTH_LANES[0])
        # angled 30 and 60 degrees it spans 90 and 124 mm across the road (README); just inside the band, centred,
        # it spans both lanes and scores
        for h, span in ((-120.0, 0.0902), (-150.0, 0.1243)):
            m = engine_measures(engine_at(0.087, heading=h, x=0.2325))
            self.assertAlmostEqual(m["x_hi"] - m["x_lo"], span, places=4)
        for h in (-120.2, -149.8):
            s = engine_at(0.087, heading=h, x=0.2325)
            m = engine_measures(s)
            self.assertLessEqual(m["x_lo"], L.BOTH_LANES[0])
            self.assertGreaterEqual(m["x_hi"], L.BOTH_LANES[1])
            self.assertEqual(scored(s), 1.0, h)

    def test_start_layout_scores_zero(self):
        self.assertTrue(Tracker.at_start("fire_engine", START_E))
        self.assertTrue(Tracker.at_start("ambulance", START_A))
        r, _ = settled(START_E, START_A)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["zero_reason"])
        self.assertEqual(r.details["fire_engine"]["why"], "more_than_250_mm_back")
        self.assertEqual(r.details["ambulance"]["why"], "more_than_250_mm_on")  # and on the shoulder
        self.assertEqual(engine_measures(START_E)["crash_lane_cover"], 0.0)  # in lane 2: not over lane 1 either
        fe, fa = footprint("fire_engine", START_E), footprint("ambulance", START_A)
        self.assertTrue(np.all((fe[:, 0] > L.LANE2[0]) & (fe[:, 0] < L.LANE2[1])))  # in lane 2
        self.assertTrue(np.all((fa[:, 0] > L.SHOULDER_NEAR[0]) & (fa[:, 0] < L.SHOULDER_NEAR[1])))  # on the shoulder
        for fp in (fe, fa):  # clear of the arms' start-pose shadow in the setup screenshot
            self.assertTrue(np.all(np.abs(fp[:, 1]) < 0.45))

    def test_a_sideways_nudge_from_the_start_earns_nothing(self):
        """Both vehicles start beyond the 250 mm partial-credit reach (engine buffer 268 mm, ambulance gap 266.5 mm),
        so shoving one straight across into a lane, without bringing it along the road, scores 0."""
        self.assertAlmostEqual(engine_measures(START_E)["buffer"], 0.268, places=3)
        self.assertAlmostEqual(ambulance_measures(START_A)["gap"], 0.2665, places=4)
        for gap in (engine_measures(START_E)["buffer"], ambulance_measures(START_A)["gap"]):
            self.assertGreater(gap - L.FAR_MAX, 0.015)
        xe, ye, he = L.START["fire_engine"]
        xa, ya, ha = L.START["ambulance"]
        for x in (0.200, 0.210, 0.220):  # the engine shoved into lane 1 (it would cover 28 mm or more of it)
            e = eng(x, ye, math.degrees(he))
            self.assertGreaterEqual(engine_measures(e)["crash_lane_cover"], L.CRASH_LANE_COVER)
            self.assertEqual(scored(e, START_A), 0.0)
            self.assertEqual(why(e, START_A), "more_than_250_mm_back")
        for x in (0.200, 0.210, 0.255):  # the ambulance shoved into lane 1 or 2, still facing upstream
            a = amb(x, ya, math.degrees(ha))
            self.assertEqual(scored(START_E, a), 0.0)
            self.assertEqual(why(START_E, a, "ambulance"), "more_than_250_mm_on")
        # brought 20 mm along the road as well, each earns its partial credit
        self.assertEqual(scored(eng(0.210, ye - 0.020, math.degrees(he)), START_A), 0.25)
        self.assertEqual(scored(START_E, amb(0.210, ya + 0.020, math.degrees(ha))), 0.25)

    def test_demo_target(self):
        r, _ = settled(ENGINE, AMB)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        e, a = engine_measures(ENGINE), ambulance_measures(AMB)
        self.assertAlmostEqual(e["buffer"], 0.0872, places=4)
        self.assertAlmostEqual(e["angle"], 45.0, places=9)
        self.assertLess(e["panel_y"], -0.7)
        self.assertGreater(L.BOTH_LANES[0] - e["x_lo"], 0.0135)  # README: 13.7 and 13.4 mm beyond the marks
        self.assertGreater(e["x_hi"] - L.BOTH_LANES[1], 0.0132)
        self.assertLess(e["x_hi"], L.ROAD_X[1])  # all on the road
        self.assertAlmostEqual(a["gap"], 0.0865, places=4)
        self.assertAlmostEqual(a["centre_x"], 0.210)
        self.assertAlmostEqual(r.details["to_car_mm"]["fire_engine"], 88.9)
        self.assertAlmostEqual(r.details["to_car_mm"]["ambulance"], 86.5)


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        self.assertEqual(scored(START_E, START_A), 0.0)  # the start
        self.assertEqual(scored(ENGINE, AMB), 1.0)  # both right
        self.assertEqual(scored(ENGINE, NOSE_IN), 0.75)  # ambulance in the lane, nose to the car
        self.assertEqual(why(ENGINE, NOSE_IN, "ambulance"), "not_straight_rear_to_the_car")
        square = engine_at(0.087, heading=-165.0, x=0.2325)  # 75 degrees: square across the road
        self.assertEqual(scored(square), 0.75)
        self.assertEqual(why(square), "angle_out_of_band")
        panel_up = engine_at(0.087, heading=-45.0)  # angled right, pump panel toward the traffic
        self.assertEqual(scored(panel_up), 0.75)
        self.assertEqual(why(panel_up), "panel_toward_traffic")
        self.assertEqual(scored(LANE_BLOCK), 0.75)  # straight in lane 1 behind the car: a lane block
        self.assertEqual(why(LANE_BLOCK), "angle_out_of_band")
        self.assertEqual(scored(ENGINE, START_A), 0.5)  # ambulance left on its start
        self.assertEqual(scored(ENGINE, ON_SHOULDER), 0.5)  # ... or set right but on the shoulder
        self.assertEqual(why(ENGINE, ON_SHOULDER, "ambulance"), "not_in_a_lane")
        self.assertEqual(scored(LANE_BLOCK, NOSE_IN), 0.5)  # both 0.25 cases
        self.assertEqual(scored(LANE_BLOCK, START_A), 0.25)  # one 0.25 case, the other not moved
        self.assertEqual(scored(START_E, NOSE_IN), 0.25)
        self.assertEqual(scored(START_E, AMB), 0.5)
        pushed = ambulance_at(0.001)  # pushed against the car
        self.assertEqual(scored(ENGINE, pushed), 0.0)
        self.assertEqual(zero_reason(ENGINE, pushed), "a_vehicle_touching_the_car")

    def test_either_lane_and_either_panel_down_heading(self):
        self.assertEqual(scored(ENGINE, ambulance_at(0.0865, x=0.255)), 1.0)  # ambulance in lane 2
        self.assertEqual(scored(engine_at(0.087, heading=135.0, x=0.2325)), 1.0)  # front toward the near shoulder,
        self.assertLess(left_side_y(engine_at(0.087, heading=135.0)), 0.0)  # upstream: panel still downstream

    def test_engine_angle_band_edges(self):
        for h, want in ((-120.1, 1.0), (-119.9, 0.75), (-149.9, 1.0), (-150.1, 0.75)):
            self.assertEqual(scored(engine_at(0.087, heading=h, x=0.2325)), want, h)
        self.assertAlmostEqual(angle_to_lanes_deg(eng(0.2, 0.1, -120.1)), 30.1, places=6)

    def test_engine_both_lanes_edges(self):
        # slide the angled engine across the road: its outline must reach x 19.05 cm and x 27.45 cm
        base = engine_measures(ENGINE)
        lo_slack = L.BOTH_LANES[0] - base["x_lo"]
        hi_slack = base["x_hi"] - L.BOTH_LANES[1]
        x0 = L.TARGET["fire_engine"][0]
        self.assertEqual(scored(engine_at(0.087, x=x0 + lo_slack - 0.0001)), 1.0)
        out = engine_at(0.087, x=x0 + lo_slack + 0.0001)
        self.assertEqual(scored(out), 0.75)
        self.assertEqual(why(out), "not_across_both_lanes")
        self.assertEqual(scored(engine_at(0.087, x=x0 - hi_slack + 0.0001)), 1.0)
        self.assertEqual(why(engine_at(0.087, x=x0 - hi_slack - 0.0001)), "not_across_both_lanes")

    def test_engine_buffer_edges(self):
        for b, want in ((0.0501, 1.0), (0.0499, 0.75), (0.1249, 1.0), (0.1251, 0.75), (0.2499, 0.75),
                        (0.2501, 0.5), (0.0025, 0.75)):
            self.assertEqual(scored(engine_at(b)), want, b)
        self.assertEqual(why(engine_at(0.0499)), "buffer_under_50_mm")
        self.assertEqual(why(engine_at(0.1251)), "buffer_over_125_mm")
        self.assertEqual(why(engine_at(0.2501)), "more_than_250_mm_back")
        self.assertIsNone(zero_reason(engine_at(0.0025)))
        # straight behind the car in lane 1 the buffer is the gap between the outlines: 2.1 mm counts, 1.9 is touching
        self.assertEqual(scored(engine_at(0.0021, heading=-90.0, x=0.212)), 0.75)
        self.assertEqual(zero_reason(engine_at(0.0019, heading=-90.0, x=0.212)), "a_vehicle_touching_the_car")

    def test_engine_downstream_or_beside(self):
        down = eng(0.232, -0.180, -135.0)  # angled right, but downstream of the car
        self.assertEqual(why(down, ambulance_at(0.20, x=0.255)), "not_upstream")
        beside = eng(0.265, -0.030, -90.0)  # in lane 2 alongside the car
        self.assertLessEqual(engine_measures(beside)["buffer"], 0.0)
        self.assertEqual(why(beside), "not_upstream")
        self.assertEqual(scored(beside), 0.5)
        # angled across lane 1 with its downstream corner 8 mm past the car's upstream end, beside it (5 mm clear):
        # not wholly upstream, so nothing, though it covers lane 1
        corner = eng(0.200, 0.0475, 135.0)
        m = engine_measures(corner)
        self.assertTrue(-0.010 < m["buffer"] < 0.0)
        self.assertGreaterEqual(m["crash_lane_cover"], L.CRASH_LANE_COVER)
        self.assertGreater(poly_distance(footprint("fire_engine", corner), CAR_POLY), L.TOUCH_GAP)
        self.assertEqual(why(corner), "not_upstream")
        self.assertEqual(scored(corner), 0.5)

    def test_ambulance_alongside_the_car(self):
        # in lane 2 beside the car, its rear 20 mm upstream of the car's downstream end (13 mm clear): not downstream
        a = amb(0.255, L.CAR_DOWN_Y + 0.020 - 0.044, -90.0)
        self.assertAlmostEqual(ambulance_measures(a)["gap"], -0.020, places=9)
        self.assertGreater(poly_distance(footprint("ambulance", a), CAR_POLY), L.TOUCH_GAP)
        self.assertEqual(why(ENGINE, a, "ambulance"), "not_downstream")
        self.assertEqual(scored(ENGINE, a), 0.5)

    def test_wholly_upstream_and_downstream_edges(self):
        # 1 mm past the car's end, beside it and clear of it, earns nothing; 1 mm short of it, the partial credit
        for d, want in ((-0.001, 0.5), (0.001, 0.75)):
            a = amb(0.255, L.CAR_DOWN_Y - d - 0.044, -90.0)
            self.assertGreater(poly_distance(footprint("ambulance", a), CAR_POLY), L.TOUCH_GAP)
            self.assertEqual(scored(ENGINE, a), want, d)
        for d, want in ((-0.001, 0.5), (0.001, 0.75)):
            probe = eng(0.200, 0.0, 135.0)
            e = eng(0.200, L.CAR_UP_Y + d - footprint("fire_engine", probe)[:, 1].min(), 135.0)
            self.assertAlmostEqual(engine_measures(e)["buffer"], d, places=9)
            self.assertGreater(poly_distance(footprint("fire_engine", e), CAR_POLY), L.TOUCH_GAP)
            self.assertEqual(scored(e), want, d)

    def test_crash_lane_cover_edge(self):
        # straight engines slid toward lane 2: they cover less and less of lane 1
        for cover, want in ((0.0281, 0.75), (0.0279, 0.5)):
            x = L.LANE1[1] - cover + 0.0155  # its right side (v -15.5 mm) at x = lane 1's left edge - cover
            s = engine_at(0.090, heading=-90.0, x=x)
            self.assertAlmostEqual(engine_measures(s)["crash_lane_cover"], cover, places=9)
            self.assertEqual(scored(s), want, cover)
        self.assertEqual(why(engine_at(0.090, heading=-90.0, x=0.255)), "not_blocking_the_crash_lane")

    def test_ambulance_edges(self):
        for g, want in ((0.0501, 1.0), (0.0499, 0.75), (0.1249, 1.0), (0.1251, 0.75), (0.2499, 0.75),
                        (0.2501, 0.5)):
            self.assertEqual(scored(ENGINE, ambulance_at(g)), want, g)
        self.assertEqual(why(ENGINE, ambulance_at(0.0499), "ambulance"), "gap_under_50_mm")
        self.assertEqual(why(ENGINE, ambulance_at(0.1251), "ambulance"), "gap_over_125_mm")
        self.assertEqual(why(ENGINE, ambulance_at(0.2501), "ambulance"), "more_than_250_mm_on")
        for h, want in ((-75.1, 1.0), (-74.9, 0.75), (-104.9, 1.0), (-105.1, 0.75)):
            self.assertEqual(scored(ENGINE, ambulance_at(0.0865, heading=h)), want, h)
        for x, want in ((0.1876, 1.0), (0.1874, 0.5), (0.2774, 1.0), (0.2776, 0.5)):
            self.assertEqual(scored(ENGINE, ambulance_at(0.0865, x=x)), want, x)
        up = amb(0.210, 0.150, -90.0)  # upstream of the car
        self.assertEqual(why(START_E, up, "ambulance"), "not_downstream")

    def test_only_a_full_score_is_success(self):
        for e, a, want in ((ENGINE, NOSE_IN, 0.75), (ENGINE, START_A, 0.5), (LANE_BLOCK, START_A, 0.25)):
            r, _ = settled(e, a)
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, want)
        self.assertTrue(settled(ENGINE, AMB)[0].success)

    def test_every_tier_is_reachable(self):
        seen = {scored(e, a) for e in (START_E, LANE_BLOCK, ENGINE) for a in (START_A, NOSE_IN, AMB)}
        self.assertEqual(seen, TIERS)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(ENGINE, AMB)
        self.assertEqual(t.result(ENGINE, AMB).score, 1.0)
        self.assertEqual(t.result(ENGINE, AMB).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(7)
        for _ in range(300):
            e = eng(rng.uniform(0.15, 0.30), rng.uniform(-0.3, 0.4), rng.uniform(-180, 180),
                    lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)))
            a = amb(rng.uniform(0.15, 0.30), rng.uniform(-0.4, 0.3), rng.uniform(-180, 180),
                    lift=rng.choice((0.0, 0.0, 0.002, 0.01)), roll=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)))
            self.assertIn(scored(e, a), TIERS)


class ZeroConditions(unittest.TestCase):
    def test_on_its_side_or_roof_zeroes_the_trial(self):
        for r in (45.1, 60.0, 90.0, 180.0):
            self.assertEqual(scored(engine_at(0.087, roll=r)), 0.0, r)
            self.assertEqual(zero_reason(engine_at(0.087, roll=r)), "a_vehicle_on_its_side_or_roof")
            self.assertEqual(scored(ENGINE, ambulance_at(0.0865, roll=r)), 0.0, r)
        self.assertEqual(scored(START_E, amb(*L.START["ambulance"][:2], 90.0, roll=90.0)), 0.0)  # tipped over at its start
        # tilted 10.1 to 45 degrees (leaning on something): that vehicle earns nothing, the trial is not zeroed
        self.assertEqual(scored(engine_at(0.087, roll=9.9)), 1.0)
        self.assertEqual(why(engine_at(0.087, roll=10.1)), "not_upright")
        self.assertEqual(scored(engine_at(0.087, roll=10.1)), 0.5)
        self.assertEqual(scored(ENGINE, ambulance_at(0.0865, pitch=44.9)), 0.5)

    def test_touching_the_car_zeroes_the_trial(self):
        for g, want in ((0.0021, 0.75), (0.0019, 0.0)):
            self.assertEqual(scored(ENGINE, ambulance_at(g)), want, g)
        self.assertEqual(zero_reason(ENGINE, ambulance_at(0.001)), "a_vehicle_touching_the_car")
        # beside the car, against its side (top view distance 1 mm), zero too
        side = eng(L.car_corners()[1][0] + 0.0155 + 0.001, -0.030, -90.0)
        d = poly_distance(footprint("fire_engine", side), CAR_POLY)
        self.assertLess(d, L.TOUCH_GAP)
        self.assertEqual(zero_reason(side), "a_vehicle_touching_the_car")
        # held over the car (seen from above it overlaps): zero
        over = eng(0.211, -0.030, -90.0, lift=0.030)
        self.assertEqual(zero_reason(over), "a_vehicle_touching_the_car")

    def test_vehicles_touching_each_other(self):
        e = engine_at(0.087, heading=-90.0, x=0.210)
        a = amb(0.255, e.origin[1], -90.0)  # side by side, 1 mm apart
        fe, fa = footprint("fire_engine", e), footprint("ambulance", a)
        self.assertLess(poly_distance(fe, fa), 0.020)
        a = amb(0.210 + 0.0165 + 0.015 + 0.001, e.origin[1], -90.0)
        self.assertAlmostEqual(poly_distance(footprint("fire_engine", e), footprint("ambulance", a)), 0.001)
        self.assertEqual(zero_reason(e, a), "the_vehicles_touching")
        self.assertEqual(scored(e, a), 0.0)

    def test_not_parked(self):
        self.assertEqual(scored(engine_at(0.087, lift=0.0029)), 1.0)
        self.assertEqual(why(engine_at(0.087, lift=0.0031)), "not_on_the_road_surface")
        self.assertEqual(why(engine_at(0.087, lift=0.030)), "not_on_the_road_surface")  # carried 30 mm up
        self.assertEqual(scored(engine_at(0.087, lift=-0.0029)), 1.0)
        self.assertEqual(why(engine_at(0.087, lift=-0.0031)), "not_on_the_road_surface")  # sunk: broken state
        self.assertEqual(why(ENGINE, ambulance_at(0.0865, x=0.120), "ambulance"), "not_over_the_road")  # verge
        t = Tracker()
        for i in range(30):  # creeping 1 mm a step: 10 mm over 0.5 s
            r = t.update(engine_at(0.070 + 0.001 * i), AMB)
        self.assertEqual(r.details["fire_engine"]["why"], "moving")
        self.assertEqual(r.score, 0.5)


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for _ in range(WINDOW):
            self.assertEqual(t.update(ENGINE, AMB).score, 0.0)
        self.assertEqual(t.update(ENGINE, AMB).score, 1.0)

    def test_slow_creep_under_5_mm_is_still(self):
        t = Tracker()
        for i in range(40):  # 0.4 mm a step: 4 mm over 0.5 s
            r = t.update(ENGINE, ambulance_at(0.080 + 0.0004 * i))
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s
            r = t.update(ENGINE, ambulance_at(0.0865 + (0.0006 if i % 2 else 0.0)))
        self.assertEqual(r.details["ambulance"]["why"], "moving")
        self.assertEqual(r.score, 0.5)

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        for i in range(30):  # 1 degree a step about its middle: 1.1 mm a step at its corners
            r = t.update(engine_at(0.087, heading=-135.0 + (i if i < 15 else 30 - i)), AMB)
        self.assertEqual(r.details["fire_engine"]["why"], "moving")


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_road_counts_as_parked(self):
        # The scorer sees only the vehicles (task.json fidelity): held still 2 mm up at the place, a vehicle counts.
        self.assertEqual(scored(engine_at(0.087, lift=0.002), ambulance_at(0.0865, lift=0.002)), 1.0)

    def test_engine_stood_on_its_end_to_block(self):
        s = eng(0.232, 0.150, -90.0, pitch=90.0)
        self.assertEqual(scored(s), 0.0)
        self.assertEqual(zero_reason(s), "a_vehicle_on_its_side_or_roof")

    def test_ambulance_on_the_engine(self):
        top = amb(ENGINE.origin[0], ENGINE.origin[1], -90.0, lift=0.043)
        r, _ = settled(ENGINE, top)
        self.assertEqual(r.details["zero_reason"], "the_vehicles_touching")
        self.assertEqual(r.score, 0.0)

    def test_roles_swapped(self):
        # the ambulance angled across both lanes upstream, the engine straight downstream: nothing earned
        a = amb(0.232, 0.120, -135.0)
        e = eng(0.210, L.CAR_DOWN_Y - 0.087 - 0.0625, -90.0)
        r, _ = settled(e, a)
        self.assertEqual(r.details["fire_engine"]["why"], "not_upstream")
        self.assertEqual(r.details["ambulance"]["why"], "not_downstream")
        self.assertEqual(r.score, 0.0)

    def test_ambulance_angled_across_the_lanes(self):
        self.assertEqual(scored(ENGINE, ambulance_at(0.0865, heading=-135.0, x=0.232)), 0.75)

    def test_engine_parked_far_back_or_in_the_median(self):
        self.assertEqual(why(engine_at(0.300)), "more_than_250_mm_back")
        self.assertEqual(why(eng(0.330, 0.150, -135.0)), "not_over_the_road")  # centre on the median

    def test_the_car_cannot_be_used_as_a_stop(self):
        # the angled engine slid downstream until it rests against the car's corner: the trial is zeroed, though
        # its y-buffer alone is still positive (the corners that face each other are at different x)
        x0, y0, h = L.TARGET["fire_engine"]
        y = y0
        while poly_distance(footprint("fire_engine", eng(x0 + 0.015, y, math.degrees(h))), CAR_POLY) > 0.001:
            y -= 0.0005
        s = eng(x0 + 0.015, y, math.degrees(h))
        r, _ = settled(s, AMB)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "a_vehicle_touching_the_car")
        self.assertGreater(engine_measures(s)["buffer"], 0.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = settled(ENGINE, AMB)
        self.assertEqual(r.score, 1.0)
        n = {k: len(v) for k, v in t.history.items()}
        bad = []
        for k in ("origin", "quat_xyzw", "pos", "lin_vel", "low"):
            s = engine_at(0.087)
            setattr(s, k, np.full_like(np.asarray(getattr(s, k), dtype=float), np.nan))
            bad.append(s)
        z = engine_at(0.087)
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = engine_at(0.087)
        inf.origin = np.array([np.inf, 0.15, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            for pair in ((s, AMB), (ENGINE, s)):
                res = t.update(*pair)
                self.assertEqual(res.score, 0.0)
                self.assertEqual(res.details["zero_reason"], "invalid_state")
                self.assertEqual({k: len(v) for k, v in t.history.items()}, n)
        self.assertEqual(t.update(ENGINE, AMB).score, 1.0)  # the record was left as it was

    def test_missing_fields_or_objects(self):
        s = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))  # no origin
        self.assertFalse(valid(s))
        self.assertEqual(Tracker().update(s, AMB).score, 0.0)
        self.assertEqual(Tracker().update(None, AMB).score, 0.0)
        self.assertEqual(Tracker().update(ENGINE, None).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = engine_at(0.087)
        s.quat_xyzw = s.quat_xyzw * 3.0
        self.assertEqual(scored(s), 1.0)

    def test_front_pointing_straight_up(self):
        s = eng(0.23, 0.15, 0.0, pitch=-90.0)
        self.assertAlmostEqual(tilt_deg(s), 90.0, places=6)
        self.assertTrue(math.isnan(angle_to_lanes_deg(s)))
        self.assertEqual(front_off_downstream_deg(s), 180.0)
        self.assertEqual(scored(s), 0.0)
        # on its side, the panel side faces straight up: no panel direction, and the trial is zeroed anyway
        side = eng(0.23, 0.15, -135.0, roll=-90.0)
        self.assertTrue(math.isnan(left_side_y(side)))
        self.assertEqual(scored(side), 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = settled(ENGINE, AMB)
        r = t.update(START_E, START_A)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history["fire_engine"]), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(ENGINE, AMB)
        (xe, ye, _), (xa, ya, _) = L.START["fire_engine"], L.START["ambulance"]
        t.update(eng(xe + 0.0145, ye + 0.020, -90.0), amb(xa - 0.020, ya + 0.020, 90.0))  # within 3 cm of each mark
        self.assertEqual(t.steps, 1)

    def test_reset_when_one_vehicle_had_not_moved(self):
        _, t = settled(ENGINE, START_A)
        t.update(START_E, START_A)  # the engine jumps home in one step, the ambulance never left: a reset
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.START["fire_engine"][:2]), np.array([0.26, 0.25])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(eng(p[0], p[1], -90.0), START_A)
        self.assertEqual(t.steps, 20)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(START_E, START_A)
        t.update(ENGINE, AMB)
        self.assertEqual(t.steps, WINDOW + 2)

    def test_explicit_reset(self):
        _, t = settled(ENGINE, AMB)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(ENGINE, AMB).score, 0.0)  # half a second of history again


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
        cls.specs = {"fire_engine": cls.task.FIRE_ENGINE, "ambulance": cls.task.AMBULANCE}

    def sim(self, engine, ambulance):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in (("fire_engine", engine), ("ambulance", ambulance))}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["fire_engine", "ambulance"])
        self.assertTrue(all(o.mass <= 0.30 for o in loose))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        # the road strips tile the road exactly, flush, side by side, and every line sits on a lane boundary
        strips = sorted((o for o in t.ROAD), key=lambda o: (round(o.pos[0] - o.size[0] / 2, 6), o.pos[1]))
        area = sum(o.size[0] * o.size[1] for o in t.ROAD)
        self.assertAlmostEqual(area, (L.ROAD_X[1] - L.ROAD_X[0]) * (L.ROAD_Y[1] - L.ROAD_Y[0]), places=12)
        for o in strips:
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, ROAD_Z, places=12)
        for i, a in enumerate(t.ROAD):
            for b in t.ROAD[i + 1:]:
                self.assertLess(_boxes_overlap((a.pos, a.size), (b.pos, b.size)), 1e-15, (a.name, b.name))
        by = {o.name: o for o in t.ROAD}
        self.assertAlmostEqual(by["edge_line_near"].pos[0], L.LANE1[0])
        self.assertAlmostEqual(by["edge_line_far"].pos[0], L.LANE2[1])
        self.assertAlmostEqual(by["lane_line_dash_0"].pos[0], L.LANE1[1])
        self.assertEqual(sum(1 for n in by if n.startswith("lane_line_dash")), 6)
        for o in (t.VERGE, t.MEDIAN):
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, ROAD_Z, places=12)
        # the scorer's car outline is the car's
        car = t.CAR
        self.assertEqual((car.pos[0], car.pos[1]), L.CAR_XY)
        self.assertAlmostEqual(car.yaw, math.radians(10.0))
        self.assertEqual(tuple(car.size), L.CAR_SIZE)
        self.assertAlmostEqual(car.pos[2] - car.size[2] / 2, ROAD_Z, places=12)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None),
                                                                              getattr(o, "yaw", 0.0))
            hx = abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2
            hy = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            if c[0] - hx < 0.2 - 1e-9:
                self.assertLessEqual(abs(c[1]) + hy, 0.45 + 1e-9, o.name)

    def test_vehicle_geometry_matches_the_scorer(self):
        for name, spec in self.specs.items():
            back, front, right, left, top = L.OUTLINE[name]
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-back, -right, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [front, left, top], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_U[name]), 0.011)  # the grip within 11 mm of the COM
            self.assertLess(abs(com[1]), 0.0005)
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.START[name][:2], ROAD_Z], atol=1e-12)
            self.assertTrue(Tracker.at_start(name, rest))
        for shapes in (L.ENGINE_SHAPES, L.AMB_SHAPES):
            boxes = [(c, s) for _, c, s, _ in shapes]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    self.assertLess(_boxes_overlap(boxes[i], boxes[j]), 1e-15, (shapes[i][0], shapes[j][0]))
        e = {n: (c, s) for n, c, s, _ in L.ENGINE_SHAPES}
        # the pump panel on the left (+v) side, the windscreen and light bar at the front
        self.assertGreater(e["pump_panel"][0][1], 0.0155)
        self.assertGreater(e["pump_top"][0][1], 0.0)
        self.assertAlmostEqual(e["windscreen"][0][0] + e["windscreen"][1][0] / 2, 0.0625)
        self.assertGreater(e["light_bar"][0][0], 0.04)
        # the engine's pads (27 mm wide, centred at GRIP_U) stay at least 10 mm clear of the pump panel
        panel_back = e["pump_panel"][0][0] - e["pump_panel"][1][0] / 2
        self.assertGreater(panel_back - (self.task.GRIP_U["fire_engine"] + 0.0135), 0.010)
        a = {n: (c, s) for n, c, s, _ in L.AMB_SHAPES}
        self.assertAlmostEqual(a["rear_doors"][0][0] - a["rear_doors"][1][0] / 2, -0.044)
        self.assertAlmostEqual(a["windscreen"][0][0] + a["windscreen"][1][0] / 2, 0.044)
        sim = self.sim(ENGINE, AMB)
        self.assertAlmostEqual(float(sim["fire_engine"].low[2]), ROAD_Z, places=9)
        self.assertEqual(scored(sim["fire_engine"], sim["ambulance"]), 1.0)

    def test_offline_state_path(self):
        st = {}
        for name, spec in self.specs.items():
            c, _, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st[name] = spec.offline_state(box)
            self.assertEqual(st[name].source, "offline")
            np.testing.assert_allclose(st[name].origin, [*L.START[name][:2], ROAD_Z], atol=1e-9)
        self.assertEqual(scored(st["fire_engine"], st["ambulance"]), 0.0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START_E, START_A)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(ENGINE, AMB)
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
        self.assertEqual(self.task.score({"fire_engine": done["fire_engine"]}), 0.0)  # a missing vehicle scores 0

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(ENGINE, AMB))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "valid_state", "to_car_mm", "between_mm",
                          "fire_engine", "ambulance", "origins", "quats", "low_z", "vels", "source"):
                    self.assertIn(k, row)
                for k in ("credit", "why", "tilt_deg", "lowest_mm", "path_mm", "origin_mm", "buffer_mm", "x_lo_mm",
                          "x_hi_mm", "crash_lane_cover_mm", "angle_deg", "panel_y"):
                    self.assertIn(k, row["fire_engine"])
                for k in ("credit", "why", "gap_mm", "centre_x_mm", "front_off_deg"):
                    self.assertIn(k, row["ambulance"])
                self.assertAlmostEqual(row["fire_engine"]["buffer_mm"], 87.2)
                self.assertEqual(row["fire_engine"]["why"], "moving")  # one step seen: not yet still
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(ENGINE, AMB)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            approach_z = ROAD_Z + t.GRIP_H[t.SIDE[arm]] + t.APPROACH
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes 1.5 s, standing still
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(tuple(prev.pos), tuple(w.pos), (arm, w.label))
                if w.pos[2] < prev.pos[2] and w.pos[2] < approach_z - 1e-9:  # slow last descents
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.03 / 1.5 + 1e-9, (arm, w.label))
                # no wrist turn of more than 90 degrees in one segment (slerp would be ambiguous at 180)
                self.assertLessEqual(abs(math.remainder(w.yaw - prev.yaw, 2 * math.pi)), math.pi / 2 + 1e-9)
            name = t.SIDE[arm]
            grip = [w for w in wps if w.label == "grip across the body"][0]
            put = [w for w in wps if w.label == "let go"][0]
            np.testing.assert_allclose(grip.pos[:2], t.grip_point(name, L.START[name][:2], L.START[name][2]),
                                       atol=1e-12)
            np.testing.assert_allclose(put.pos[:2], t.grip_point(name, L.TARGET[name][:2], L.TARGET[name][2]),
                                       atol=1e-12)
            # the grip closes across the vehicle (yaw = heading mod 180) and the wrist turns it to its place
            for w, h in ((grip, L.START[name][2]), (put, L.TARGET[name][2])):
                self.assertAlmostEqual(math.remainder(w.yaw - h, math.pi), 0.0, places=12)
            turned = sum(math.remainder(b.yaw - a.yaw, 2 * math.pi) for a, b in zip(wps, wps[1:])
                         if wps.index(grip) <= wps.index(a) and wps.index(b) <= wps.index(put))
            self.assertAlmostEqual(math.remainder(L.START[name][2] + turned - L.TARGET[name][2], 2 * math.pi), 0.0)
            self.assertAlmostEqual(put.pos[2] - grip.pos[2], t.PLACE_UP)
        self.assertAlmostEqual(t.GRIP["fire_engine"], -0.0115)  # 23 mm commanded on the 31 mm body
        self.assertAlmostEqual(t.GRIP["ambulance"], -0.011)  # 22 mm on 30 mm
        for name, top in (("fire_engine", 0.043), ("ambulance", 0.038)):  # the gripper body clears the light bar
            self.assertGreater(t.GRIP_H[name] + 0.035 - top, 0.015)
            self.assertGreaterEqual(t.GRIP_H[name] - 0.013, 0.011)  # pad tips at least 11 mm above the base

    @staticmethod
    def _sample(wps, dt=0.05):
        """(t, TCP position, yaw) of an arm's plan, interpolated as ArmPlan does (smoothstep in time)."""
        out, t, prev = [], 0.0, None
        for w in wps:
            if prev is not None:
                n = max(1, int(round(w.duration / dt)))
                for i in range(1, n + 1):
                    s = i / n
                    k = 3 * s * s - 2 * s * s * s
                    p = (1 - k) * np.array(prev.pos) + k * np.array(w.pos)
                    yaw = prev.yaw + k * math.remainder(w.yaw - prev.yaw, 2 * math.pi)
                    out.append((t + s * w.duration, p, yaw, w.label))
            t += w.duration
            prev = w
        return out

    def test_carry_paths_stay_clear_of_the_car_and_each_other(self):
        """Replays both arms' carries (the vehicle rigid in the grip, turning with the wrist) and checks the lifted
        vehicles' outlines against the car and each other."""
        t = self.task
        paths = {}
        for arm in ("left", "right"):
            wps = t.demo()[arm]
            a = [i for i, w in enumerate(wps) if w.label == "lift 30 mm"][0]
            b = [i for i, w in enumerate(wps) if w.label == "lower it slowly"][0]
            name = t.SIDE[arm]
            h0, y0 = L.START[name][2], wps[a].yaw
            out = []
            for tt, p, yaw, _ in self._sample(wps[a:b + 1]):
                h = h0 + math.remainder(yaw - y0, 2 * math.pi) if abs(yaw - y0) < math.pi else h0 + (yaw - y0)
                u = t.GRIP_U[name]
                out.append((round(tt, 3), veh(name, p[0] - u * math.cos(h), p[1] - u * math.sin(h), math.degrees(h))))
            self.assertAlmostEqual(math.remainder(h - L.TARGET[name][2], 2 * math.pi), 0.0, places=9)
            paths[arm] = out
        worst_car, worst_between = 1.0, 1.0
        for (tl, e), (tr, a) in zip(paths["left"], paths["right"]):
            self.assertAlmostEqual(tl, tr)  # both arms on one clock: the same moment
            fe, fa = footprint("fire_engine", e), footprint("ambulance", a)
            worst_car = min(worst_car, poly_distance(fe, CAR_POLY), poly_distance(fa, CAR_POLY))
            worst_between = min(worst_between, poly_distance(fe, fa))
        self.assertGreater(worst_car, 0.080)  # README: more than 80 mm from the car
        self.assertGreater(worst_between, 0.20)  # and more than 20 cm from each other

    def test_tcps_stay_apart(self):
        plan = self.task.demo()
        sl, sr = self._sample(plan["left"]), self._sample(plan["right"])
        self.assertEqual(len(sl), len(sr))
        self.assertGreaterEqual(min(abs(a[1][1] - b[1][1]) for a, b in zip(sl, sr)), 0.20)

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
        for s in ("near lane", "one-way, two-lane", "1:80", "traffic comes from your left", "upstream",
                  "30 to 60 degrees", "blocks both lanes", "pump-panel side facing the car", "50 to 125 mm (4 to 10 m)",
                  "downstream", "straight in either lane", "rear doors toward the car",
                  "touch the car or the other vehicle"):
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
        for s in ("10 degrees or less", "within 3 mm", "x 15.0 to 29.5 cm", "5 mm over the last 0.5 s",
                  "y -0.05 cm", "y -5.95 cm", "45 degrees", "within 2 mm", "50 to 125 mm", "30 to 60 degrees",
                  "x 19.05 cm", "x 27.45 cm", "toward -y", "250 mm or less", "at least 28 mm", "x 18.75 to 23.25 cm",
                  "x 18.75 to 27.75 cm", "within 15 degrees", "0.25", "0.5", "0.75", "1.0", "3 s"):
            self.assertIn(s, sc)
        self.assertEqual(L.TILT_PARK_DEG, 10.0)
        self.assertEqual(L.FLOOR_TOL, 0.003)
        self.assertEqual(L.ROAD_CENTRE_X, (0.150, 0.295))
        self.assertEqual(L.STILL_PATH, 0.005)
        self.assertEqual(L.TIPPED_DEG, 45.0)
        self.assertEqual(L.TOUCH_GAP, 0.002)
        self.assertEqual(L.BAND, (0.050, 0.125))
        self.assertEqual(L.ANGLE_DEG, (30.0, 60.0))
        np.testing.assert_allclose(L.BOTH_LANES, (0.1905, 0.2745), atol=1e-12)
        self.assertEqual(L.FAR_MAX, 0.250)
        self.assertEqual(L.CRASH_LANE_COVER, 0.028)
        self.assertEqual(L.LANES, (0.1875, 0.2775))
        self.assertEqual(L.STRAIGHT_DEG, 15.0)
        for s in ("y +30.5", "y -45.0 to +45.0", "x 10.0 to 15.0", "x 15.0 to 18.6", "x 18.6 to 18.9",
                  "x 18.9 to 23.1", "x 23.1 to 23.4", "x 23.4 to 27.6", "x 27.6 to 27.9", "x 27.9 to 29.5",
                  "x 18.75 to 23.25", "x 23.25 to 27.75", "x 29.5 to 36.0", "+41.5", "22 mm", "56 mm",
                  "18 x 30 x 9", "x 21.1 cm, y -3.0 cm", "10 degrees", "80 g", "125 mm", "31 mm", "40 mm",
                  "15 to 30 mm forward", "125 x 32", "x 25.55 cm, y +33.0 cm", "50 g", "88 mm", "30 mm", "35 mm",
                  "x 17.0 cm, y -37.0 cm"):
            self.assertIn(s, su)
        strips = {n: (xs, ys) for n, xs, ys, _ in L.road_strips()}
        self.assertAlmostEqual(strips["shoulder_near"][0][1], 0.186)
        self.assertAlmostEqual(strips["lane_1"][0][1], 0.231)
        self.assertAlmostEqual(strips["shoulder_far"][0][0], 0.279)
        nums = {float(v) for v in re.findall(r"(\d+(?:\.\d+)?) mm", sc)}
        self.assertTrue({3.0, 5.0, 2.0, 125.0, 250.0, 28.0, 45.0} <= nums)

    def test_neither_arm_reaches_the_other_vehicle(self):
        """README 'One arm alone': straight down, at grip heights, neither arm reaches any point of the other
        vehicle where it starts, at any yaw (joint 6 allows +-120 degrees; 30 degree steps here, 15 in the README's
        wider check)."""
        import yamkin as K
        for arm, other in (("right", "fire_engine"), ("left", "ambulance")):
            base = K.BASES["station"][arm]
            x, y, h = L.START[other]
            back, front, right, left, _ = L.OUTLINE[other]
            for u in (-back, 0.0, front):
                for v in (-right, left):
                    px, py = x + u * math.cos(h) - v * math.sin(h), y + u * math.sin(h) + v * math.cos(h)
                    for z in (0.013, 0.026):
                        for yaw in range(-120, 121, 30):
                            cell, _ = K.reach_cell((px, py, TOP + z), math.radians(yaw), base, TOP, None)
                            self.assertEqual(cell, 0, (arm, u, v, z, yaw))
            # ... while each arm reaches its own vehicle's grip at its start and at its place
            own = "fire_engine" if arm == "left" else "ambulance"
            for (ox, oy, oh), yaw in ((L.START[own], self.task.YAWS[own][0]), (L.TARGET[own], self.task.YAWS[own][2])):
                gx, gy = self.task.grip_point(own, (ox, oy), oh)
                cell, _ = K.reach_cell((gx, gy, ROAD_Z + self.task.GRIP_H[own]), yaw, base, TOP, None)
                self.assertEqual(cell, 2, (arm, own))


if __name__ == "__main__":
    unittest.main()
