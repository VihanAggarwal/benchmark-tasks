"""Offline tests for layout.py, scoring.py and task.py: the air rule driven by synthetic bulb joints, the scorer fed
synthetic pressure traces at the 20 Hz control rate; no simulator.

    PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import sys
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

HZ = int(L.CONTROL_HZ)
STATES = {S.BULB: types.SimpleNamespace(joints={S.SQUEEZE: 0.004, S.VALVE: -L.VALVE_START})}
SHUT = -L.VALVE_START  # the valve joint when shut


def trace(peak=180.0, rate=2.5, rise_s=12.0, tail_rate=20.0, end=10.0, pause_at=None, pause_s=0.0, dump=None):
    """A pressure trace: up to ``peak``, down at ``rate`` to 60 (an optional pause, or a ``dump`` = (from, rate)),
    then down at ``tail_rate`` to ``end``."""
    out = [peak * k / (rise_s * HZ) for k in range(int(rise_s * HZ) + 1)]
    p = peak
    while p > 60.0:
        r = rate
        if dump is not None and p <= dump[0]:
            r = dump[1]
        if pause_at is not None and p <= pause_at and pause_s > 0:
            out += [p] * int(pause_s * HZ)
            pause_s = 0.0
        p -= r / HZ
        out.append(p)
    while p > end:
        p -= tail_rate / HZ
        out.append(max(p, end))
    return out + [out[-1]] * 30


def trace_dump_top():
    """Pumped to 195, dumped at 12 mmHg/s to 160, then let down on rate."""
    out = [195.0 * k / 240 for k in range(241)]
    p = 195.0
    while p > 160.0:
        p -= 12.0 / HZ
        out.append(p)
    rest = trace(peak=p, rise_s=0.05)
    return out + rest[2:]


def score(pressures):
    t = S.Tracker()
    r = None
    for p in pressures:
        r = t.update(STATES, None, p)
    return r


class TestAir(unittest.TestCase):
    def stroke(self, cuff, valve=SHUT, lo=0.004, hi=0.020, n=8):
        for s in [lo + (hi - lo) * k / n for k in range(n + 1)] + [hi - (hi - lo) * k / n for k in range(n + 1)]:
            cuff.step(s, valve)

    def test_squeezes_pump_the_cuff_up_and_each_adds_less(self):
        c = S.Cuff()
        c.step(0.004, SHUT)
        self.stroke(c)
        first = c.p
        for _ in range(16):
            self.stroke(c)
        before = c.p
        self.stroke(c)
        self.assertGreater(first, 10.0)
        self.assertLess(c.p - before, first)
        self.assertTrue(170.0 <= T.expected_peak() <= 200.0)

    def test_a_bulb_merely_held_pumps_nothing(self):
        c = S.Cuff()
        for k in range(400):
            c.step(0.004 + 0.0002 * (-1) ** k, SHUT)  # +-0.2 mm (the simulator holds within +-0.05)
        self.assertEqual(c.p, 0.0)

    def test_a_shut_valve_holds_and_an_open_one_lets_down(self):
        c = S.Cuff()
        c.p = 150.0
        for _ in range(100):
            c.step(0.004, SHUT)
        self.assertEqual(c.p, 150.0)
        theta = T.TARGET_RATE / (L.LEAK * math.sqrt(150.0))
        rate = c.leak_rate(theta, 150.0)
        self.assertAlmostEqual(rate, 2.5, places=6)
        c.step(0.004, theta - L.VALVE_START)
        self.assertAlmostEqual(c.p, 150.0 - rate / HZ, places=3)

    def test_wide_open_empties_the_cuff_within_ten_seconds(self):
        c = S.Cuff()
        c.p = 60.0
        for _ in range(int(L.EMPTY_WITHIN_S * HZ)):
            c.step(0.004, L.VALVE_MAX - L.VALVE_START - 0.3)
        self.assertLess(c.p, L.EMPTY_BELOW)

    def test_the_planned_openings_stay_below_the_knee_and_open_one_way(self):
        plan = [th for _, th in T.valve_plan(160.0, 0.0)]
        self.assertTrue(all(b > a for a, b in zip(plan, plan[1:])))
        self.assertLess(plan[-1], L.LEAK_KNEE)


class TestScoring(unittest.TestCase):
    def test_a_clean_reading_is_full(self):
        r = score(trace())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_two_to_three_a_second_is_on_rate_and_four_is_not(self):
        self.assertEqual(score(trace(rate=2.0)).score, 1.0)
        self.assertEqual(score(trace(rate=3.0)).score, 1.0)
        self.assertLess(score(trace(rate=4.2)).score, 1.0)

    def test_a_pause_then_a_quick_fall_is_not_full(self):
        r = score(trace(pause_at=120.0, pause_s=8.0, dump=(119.0, 4.4)))
        self.assertLess(r.score, 1.0)

    def test_a_dump_from_the_peak_to_160_is_not_full(self):
        r = score(trace_dump_top())
        self.assertLess(r.score, 1.0)

    def test_a_quick_last_second_before_60_is_caught(self):
        r = score(trace(dump=(66.0, 9.0)))  # 6 mmHg dumped in the last 0.7 s before 60: a gross last window
        self.assertLess(r.score, 1.0)

    def test_a_dump_is_a_quarter(self):
        self.assertEqual(score(trace(rate=12.0)).score, 0.25)

    def test_over_inflation_scores_nothing(self):
        self.assertEqual(score(trace(peak=230.0)).score, 0.0)

    def test_short_of_170_or_not_emptied_is_half(self):
        self.assertEqual(score(trace(peak=165.0)).score, 0.5)
        self.assertEqual(score(trace(end=40.0)).score, 0.5)

    def test_never_pumped_up_scores_nothing(self):
        self.assertEqual(score([0.0] * 200 + [100.0 - k * 0.1 for k in range(900)]).score, 0.0)

    def test_pumped_up_but_never_let_down_is_a_quarter(self):
        self.assertEqual(score([180.0 * min(1.0, k / 240) for k in range(1200)]).score, 0.25)

    def test_unreadable_state(self):
        r = S.Tracker().update({}, None, 100.0)
        self.assertFalse(r.details["valid_state"])


class TestDemo(unittest.TestCase):
    def test_demo_fits_the_run_limit_and_variants_exist(self):
        for v in ("", *T.VARIANTS):
            p = T.plans(v)
            self.assertEqual(set(p), {"left", "right"})
            self.assertLess(max(sum(w.duration for w in ws) for ws in p.values()), 2400 / L.CONTROL_HZ)
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_the_needle_spans_the_dial(self):
        self.assertAlmostEqual(T.needle_angle(0.0), 0.0)
        self.assertAlmostEqual(T.needle_angle(300.0), L.NEEDLE_SPAN)
        self.assertAlmostEqual(T.needle_angle(400.0), L.NEEDLE_SPAN)


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
