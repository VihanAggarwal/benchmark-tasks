"""Offline tests for layout.py, scoring.py and task.py: synthetic states for the hot plate, pan, lid and glass (mechkit
AssemblyStates from the task's own specs) and synthetic arm snapshots (rigstate.Arm); no simulator.

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
PAN0 = np.array(L.pan_start())
SEATED = np.array([PAN0[0], PAN0[1], L.rim_z()])
OFF = L.KNOB_AT_8


def qx(deg):
    a = math.radians(deg) / 2
    return (math.sin(a), 0.0, 0.0, math.cos(a))


def cooktop(front=0.0, rear=0.0):
    return mk.state_from_poses(T.COOKTOP_SPEC, (L.CT_FRONT_X, L.CT_CY, L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                               joints={S.FRONT_KNOB: front, S.REAR_KNOB: rear})


def pan(offset=(0.0, 0.0, 0.0), tilt=0.0):
    yaw = L.PAN_START_YAW / 2
    q = np.array([0.0, 0.0, math.sin(yaw), math.cos(yaw)])
    if tilt:
        t = qx(tilt)
        x1, y1, z1, w1 = t
        x2, y2, z2, w2 = q
        q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    return mk.state_from_poses(T.PAN_SPEC, (*(PAN0 + np.asarray(offset)), *q), np.zeros(6))


def lid(pos=None, tilt=0.0):
    p = np.array(L.lid_start()) if pos is None else np.asarray(pos, dtype=float)
    return mk.state_from_poses(T.LID_SPEC, (*p, *qx(tilt)), np.zeros(6))


def glass(xy=L.GLASS_START, tilt=0.0, lift=0.0):
    return mk.state_from_poses(T.GLASS_SPEC, (xy[0], xy[1], L.TOP + lift, *qx(tilt)), np.zeros(6))


def states(knob=0.0, right=0.0, lid_pos=None, lid_tilt=0.0, pan_off=(0.0, 0.0, 0.0), pan_tilt=0.0, g=None):
    return {S.COOKTOP: cooktop(knob, right), S.PAN: pan(pan_off, pan_tilt), S.LID: lid(lid_pos, lid_tilt),
            S.GLASS: g or glass()}


def run(st, arms=AWAY, n=40, tracker=None):
    t = tracker or S.Tracker()
    r = None
    for _ in range(n):
        r = t.update(st, arms)
    return t, r


class TestScoring(unittest.TestCase):
    def test_start_is_zero(self):
        _, r = run(states())
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["failed"])
        self.assertFalse(r.details["lid_on"])

    def test_lid_on_and_off_and_let_go_is_full(self):
        _, r = run(states(knob=OFF, lid_pos=SEATED))
        self.assertTrue(r.details["lid_on"])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_lid_on_burner_still_on_is_half(self):
        _, r = run(states(knob=0.0, lid_pos=SEATED))
        self.assertEqual(r.score, 0.5)
        _, r = run(states(knob=OFF - math.radians(15), lid_pos=SEATED))
        self.assertEqual(r.score, 0.5)
        _, r = run(states(knob=OFF - math.radians(8), lid_pos=SEATED))
        self.assertEqual(r.score, 1.0)

    def test_wrong_knob_turned_on_is_half(self):
        _, r = run(states(knob=OFF, right=math.radians(90), lid_pos=SEATED))
        self.assertFalse(r.details["other_off"])
        self.assertEqual(r.score, 0.5)

    def test_burner_off_without_lid_is_quarter(self):
        _, r = run(states(knob=OFF))
        self.assertEqual(r.score, 0.25)
        _, r = run(states(knob=OFF, right=math.radians(30)))
        self.assertEqual(r.score, 0.0)

    def test_lid_must_sit_on_the_rim(self):
        for pos, tilt in ((SEATED + [0.006, 0.0, 0.0], 0.0), (SEATED + [0.0, 0.0, 0.005], 0.0), (SEATED, 4.0)):
            _, r = run(states(knob=OFF, lid_pos=pos, lid_tilt=tilt))
            self.assertEqual(r.score, 0.25, (pos, tilt))

    def test_lid_held_is_half(self):
        tcp = T.lid_tcp(SEATED)
        holding = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.4), "right": AWAY["right"]}
        _, r = run(states(knob=OFF, lid_pos=SEATED), arms=holding)
        self.assertEqual(r.score, 0.5)

    def test_unreadable_arms_are_not_clear(self):
        _, r = run(states(knob=OFF, lid_pos=SEATED), arms=None)
        self.assertEqual(r.score, 0.5)

    def test_water_latches_zero(self):
        for g in (glass(tilt=60.0), glass(xy=(L.CT_FRONT_X + 0.05, 0.10), lift=0.10)):
            t = S.Tracker()
            t.update(states(g=g), AWAY)
            _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
            self.assertEqual(r.score, 0.0)
            self.assertIn("water", r.details["failed"])

    def test_the_glass_moved_once_the_lid_is_on_is_no_hazard(self):
        t = S.Tracker()
        t.update(states(knob=OFF, lid_pos=SEATED, g=glass(tilt=60.0)), AWAY)
        _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_moving_the_burning_pan_latches_zero(self):
        for kw in ({"pan_off": (0.0, 0.0, 0.03)}, {"pan_off": (-0.05, 0.0, 0.0)}, {"pan_tilt": 15.0}):
            t = S.Tracker()
            t.update(states(**kw), AWAY)
            _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
            self.assertEqual(r.score, 0.0, kw)
            self.assertIn("pan", r.details["failed"])

    def test_a_hand_over_the_open_pan_latches_zero(self):
        over = PAN0 + np.array([0.02, 0.0, 0.15])
        arms = {"left": rigstate.Arm(over, over - np.array([0.0, 0.0, 0.013]), 1.0), "right": AWAY["right"]}
        t = S.Tracker()
        t.update(states(), arms)
        _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
        self.assertEqual(r.score, 0.0)
        self.assertIn("over the open", r.details["failed"])

    def test_a_hand_carrying_the_lid_over_the_pan_is_shielded(self):
        lid_pos = SEATED + np.array([0.0, 0.03, 0.02])  # the lid sliding on, its knob in the hand
        tcp = T.lid_tcp(lid_pos)
        arms = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.4), "right": AWAY["right"]}
        t = S.Tracker()
        t.update(states(lid_pos=lid_pos), arms)
        self.assertIsNone(t.last.details["failed"])

    def test_the_demo_knob_hand_is_outside_the_rim(self):
        kx, ky = L.knob_xy("front_knob")
        self.assertGreater(math.hypot(PAN0[0] - kx, PAN0[1] - ky), L.PAN_TOP_R + 0.02)

    def test_a_failure_survives_everything_but_a_reset(self):
        t = S.Tracker()
        t.update(states(g=glass(tilt=60.0)), AWAY)
        _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
        self.assertEqual(r.score, 0.0)
        t.reset()
        _, r = run(states(knob=OFF, lid_pos=SEATED), tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_missing_object_scores_zero(self):
        st = states(knob=OFF, lid_pos=SEATED)
        del st[S.LID]
        _, r = run(st)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_start_is_clear_of_failures(self):
        r = S.read(states())
        self.assertIsNone(S.failure(r))

    def test_lid_and_glass_start_off_the_cooktop_and_apart(self):
        x0, x1, y0, y1 = L.cooktop_bounds()
        self.assertGreater(L.LID_START[1] - L.LID_R, y1)
        self.assertLess(L.GLASS_START[0] + L.GLASS_R, x0)
        self.assertGreater(L.LID_START[0] - L.LID_R, L.GLASS_START[0] + L.GLASS_R)

    def test_flames_clear_a_seated_lid(self):
        tallest = max(h for _, h, _ in L.FLAMES) + L.PAN_FLOOR + 0.003
        self.assertLess(tallest, L.PAN_DEPTH - L.LID_SKIRT_H - 0.002)

    def test_skirt_fits_inside_the_pan_and_the_rim_carries_the_lid(self):
        depth = L.PAN_DEPTH - L.LID_SKIRT_H  # where the skirt's lower edge reaches, above the pan's base
        inner = L.PAN_BASE_R + (L.PAN_TOP_R - L.PAN_BASE_R) * depth / L.PAN_DEPTH - L.PAN_WALL / 2
        self.assertLess(L.SKIRT_R + L.SKIRT_T / 2, inner - 0.004)
        self.assertGreater(L.LID_R, L.PAN_TOP_R)


class TestDemo(unittest.TestCase):
    def test_demo_turns_the_knob_to_off_and_seats_the_lid(self):
        p = T.plans()
        swept = sum(abs(b - a) for a, b in T.STROKES)
        self.assertGreaterEqual(swept, math.degrees(L.KNOB_AT_8) + 5.0)
        seat = next(w for w in p["left"] if w.label == "lower it onto the rim")
        self.assertAlmostEqual(seat.pos[2], T.lid_tcp(SEATED)[2] - T.SEAT_PRESS, places=6)
        self.assertEqual(p["left"][-1].finger, T.OPEN)
        self.assertEqual(p["right"][-1].finger, T.OPEN)

    def test_each_stroke_takes_the_bar_along_its_length(self):
        pointer = -math.degrees(L.KNOB_AT_8)  # the bar's direction at the start (seen from above, from +x)
        self.assertAlmostEqual((T.STROKES[0][0] - pointer) % 180.0, 0.0, places=6)
        self.assertAlmostEqual((T.STROKES[1][0] - T.STROKES[0][1]) % 180.0, 0.0, places=6)

    def test_open_fingers_stay_clear_of_the_pan(self):
        kx, ky = L.knob_xy("front_knob")
        gap = math.hypot(PAN0[0] - kx, PAN0[1] - ky) - L.PAN_TOP_R
        finger = -T.KNOB_OPEN + 0.008  # a pad's outer face from the knob's axis
        self.assertGreater(gap - finger, 0.005)

    def test_variants(self):
        self.assertEqual(set(T.plans("knob_only")), {"right"})
        self.assertEqual(set(T.plans("lid_only")), {"left"})
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
