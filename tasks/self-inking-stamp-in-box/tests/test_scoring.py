"""Offline tests for layout.py, scoring.py and task.py: synthetic stamp and sheet states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm); no simulator.

    PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
SHEET_TOP = L.TOP + L.SHEET[2]
BOX = np.array(L.box_world())
HOME = np.array(L.stamp_start())


def yawq(deg, tilt=0.0):
    y, t = math.radians(deg) / 2, math.radians(tilt) / 2
    qz = np.array([0.0, 0.0, math.sin(y), math.cos(y)])
    if not tilt:
        return tuple(qz)
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = (math.sin(t), 0.0, 0.0, math.cos(t))
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def stamp(pos, yaw=0.0, travel=0.0, tilt=0.0):
    return mk.state_from_poses(T.STAMP_SPEC, (*pos, *yawq(yaw, tilt)), np.zeros(6), joints={S.HANDLE: travel})


def sheet(offset=(0.0, 0.0)):
    x, y, z = L.sheet_start()
    return mk.state_from_poses(T.SHEET_SPEC, (x + offset[0], y + offset[1], z, 0.0, 0.0, 0.0, 1.0), np.zeros(6))


class Trial:
    def __init__(self):
        self.t = S.Tracker()

    def step(self, st, sh=None, arms=AWAY):
        return self.t.update({S.STAMP: st, S.SHEET: sh or sheet()}, arms)

    def press(self, xy=BOX, yaw=0.0, depth=L.TRAVEL, n=10, sheet_moves=(0.0, 0.0)):
        pos = (xy[0], xy[1], SHEET_TOP)
        for k in range(n):
            f = k / max(1, n - 1)
            self.step(stamp(pos, yaw, depth), sheet((sheet_moves[0] * f, sheet_moves[1] * f)))
        self.step(stamp(pos, yaw, 0.0), sheet(sheet_moves))
        return self.step(stamp((xy[0], xy[1], SHEET_TOP + 0.05), yaw), sheet(sheet_moves))

    def home(self, n=30):
        r = None
        for _ in range(n):
            r = self.step(stamp(HOME))
        return r


class TestScoring(unittest.TestCase):
    def test_nothing_done_is_zero(self):
        r = Trial().home()
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["prints"], 0)

    def test_square_print_in_box_and_home_is_full(self):
        tr = Trial()
        tr.press()
        r = tr.home()
        self.assertEqual(r.details["prints"], 1)
        self.assertTrue(r.details["print_corners_in"])
        self.assertEqual(r.score, 1.0)

    def test_not_back_is_three_quarters(self):
        tr = Trial()
        r = tr.press()
        self.assertEqual(r.score, 0.75)

    def test_half_press_prints_nothing(self):
        tr = Trial()
        tr.press(depth=L.TRAVEL / 2)
        self.assertEqual(tr.home().score, 0.0)

    def test_short_dab_prints_nothing(self):
        tr = Trial()
        tr.press(n=L.PRINT_STEPS - 1)
        self.assertEqual(tr.home().details["prints"], 0)

    def test_skewed_or_off_box_tiers(self):
        tr = Trial()
        tr.press(yaw=12.0)
        self.assertEqual(tr.home().score, 0.5)
        tr = Trial()
        tr.press(xy=BOX + [0.0, 0.03])  # half over the box's end
        self.assertEqual(tr.home().score, 0.5)
        tr = Trial()
        tr.press(xy=BOX + [0.0, 0.08])  # on the sheet, outside the box
        self.assertEqual(tr.home().score, 0.25)

    def test_upside_down_is_zero(self):
        tr = Trial()
        tr.press(yaw=180.0)
        self.assertEqual(tr.home().score, 0.0)

    def test_two_prints_is_zero(self):
        tr = Trial()
        tr.press()
        tr.press()
        self.assertEqual(tr.home().details["prints"], 2)
        self.assertEqual(tr.home().score, 0.0)

    def test_held_down_does_not_print_twice(self):
        tr = Trial()
        pos = (BOX[0], BOX[1], SHEET_TOP)
        for _ in range(40):
            tr.step(stamp(pos, 0.0, L.TRAVEL))
        self.assertEqual(len(tr.t.prints), 1)

    def test_sheet_slipping_during_the_press_is_not_full(self):
        tr = Trial()
        tr.press(sheet_moves=(0.0, 0.008))
        self.assertLess(tr.home().score, 1.0)

    def test_pressed_tilted_or_off_the_sheet_prints_nothing(self):
        tr = Trial()
        pos = (BOX[0], BOX[1], SHEET_TOP)
        for _ in range(10):
            tr.step(stamp(pos, 0.0, L.TRAVEL, tilt=6.0))
        self.assertEqual(len(tr.t.prints), 0)
        tr = Trial()
        for _ in range(10):
            tr.step(stamp((0.10, -0.25, L.TOP), 0.0, L.TRAVEL))  # pressed on the bare table
        self.assertEqual(len(tr.t.prints), 0)

    def test_dropped_stamp_latches_zero(self):
        tr = Trial()
        tr.step(stamp((0.25, -0.2, L.TOP + 0.02), tilt=80.0))
        tr.press()
        self.assertEqual(tr.home().score, 0.0)

    def test_held_at_home_is_not_full(self):
        tr = Trial()
        tr.press()
        t = T.tcp(L.STAMP_SPOT, L.TOP)
        holding = {"left": AWAY["left"], "right": rigstate.Arm(t, t - np.array([0.0, 0.0, 0.013]), 0.3)}
        r = None
        for _ in range(30):
            r = tr.step(stamp(HOME), arms=holding)
        self.assertEqual(r.score, 0.75)

    def test_unreadable_state_scores_zero(self):
        r = S.Tracker().update({S.STAMP: stamp(HOME)}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_box_on_the_sheet_and_imprint_fits_the_box(self):
        self.assertLess(abs(L.BOX_CENTRE[0]) + L.BOX[0] / 2, L.SHEET[0] / 2)
        self.assertLess(abs(L.BOX_CENTRE[1]) + L.BOX[1] / 2, L.SHEET[1] / 2)
        self.assertLess(L.IMPRINT[1], L.BOX[0])
        self.assertLess(L.IMPRINT[0], L.BOX[1])

    def test_stamp_starts_off_the_sheet(self):
        x0 = L.SHEET_CENTRE[1] - L.SHEET[1] / 2
        self.assertLess(L.STAMP_SPOT[1] + L.HANDLE[1] / 2, x0)


class TestDemo(unittest.TestCase):
    def test_demo_presses_past_the_bottom_and_lets_go(self):
        p = T.plans()["right"]
        press = next(w for w in p if w.label == "push the handle down")
        set_down = next(w for w in p if w.label == "set the base down on the paper")
        self.assertGreater(set_down.pos[2] - press.pos[2], L.TRAVEL)
        self.assertEqual(p[-1].finger, T.OPEN)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertIn("right", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
