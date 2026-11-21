"""Unit tests (python3 -m unittest discover -s tests -q from the task folder; needs tools/offline on the path, which
the checker's stub provides: run from the repository root with PYTHONPATH set, see README)."""

import json
import math
import os
import pathlib
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = TASK.parents[1]
for p in (TASK, ROOT / "tools" / "offline" / "yam_stub", ROOT / "tools" / "offline"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402

BAL = P.build(P.tune())
ROCK = P.specimen()
STEP100, STEP10 = L.R100_STEP, L.R10_STEP
FAR = {"left": {"tcp": [0.2, 0.27, 0.9075], "tips": [0.2, 0.27, 0.8945], "opening": 0.0},
       "right": {"tcp": [0.14, -0.35, 0.9475], "tips": [0.14, -0.35, 0.9345], "opening": 0.0}}


def bal_state(th=0.0, q100=2 * STEP100, q10=3 * STEP10, qp=L.POISE_ANS):
    return mk.state_from_poses(BAL, (L.X0, L.YP, L.TOP, 0, 0, 0, 1),
                               joints={"beam": th, "r100": q100, "r10": q10, "poise": qp}, source="test")


def rock_on_pan(th=0.0, dx=0.0, dy=0.0):
    c, s = math.cos(th), math.sin(th)
    up, ry = L.PAN_Z - L.HZ, L.LA + dy
    pos = (L.X0 + dx, L.YP + ry * c - up * s, L.TOP + L.HZ + ry * s + up * c)
    return mk.state_from_poses(ROCK, (*pos, math.sin(th / 2), 0, 0, math.cos(th / 2)), source="test")


def rock_at(x, y, z):
    return mk.state_from_poses(ROCK, (x, y, L.TOP + z, 0, 0, 0, 1), source="test")


def run(bal, rock, arms=FAR, steps=45, tracker=None):
    t = tracker or S.Tracker()
    rigstate.offline_feed(arms)
    res = None
    for _ in range(steps):
        rigstate.offline_feed(arms)
        res = t.update(bal, rock, rigstate.arms())
    return res, t


class Parts(unittest.TestCase):
    def test_empty_beam_is_balanced(self):
        self.assertLess(abs(P.tare_torque(BAL)), 1e-4)

    def test_tare_error_in_grams(self):
        grams = abs(P.tare_torque(BAL)) / (L.G * L.LA) * 1000
        self.assertLess(grams, 0.01)

    def test_lever_ratios(self):
        self.assertAlmostEqual(L.R100_MASS * L.R100_STEP, 0.100 * L.LA, places=9)
        self.assertAlmostEqual(L.R10_MASS * L.R10_STEP, 0.010 * L.LA, places=9)
        self.assertAlmostEqual(L.POISE_MASS * L.POISE_SPAN, 0.010 * L.LA, places=9)

    def test_full_scale_is_610_g(self):
        self.assertAlmostEqual(S.reading_g(L.R100_STEP * 5, L.R10_STEP * 10, L.POISE_SPAN), 610.0, places=6)

    def test_answer(self):
        self.assertAlmostEqual(S.reading_g(2 * STEP100, 3 * STEP10, L.POISE_ANS), L.SPEC_G, places=6)
        self.assertAlmostEqual(L.POISE_ANS, 0.064, places=9)

    def test_sensitivity(self):
        # 1 g at the pan turns the beam by 2 degrees against the spring (0.5 g per degree)
        self.assertAlmostEqual(math.degrees(0.001 * L.G * L.LA / L.K_BEAM), 2.0, places=6)

    def test_unique_notch_answer(self):
        sols = [(a, b) for a in range(6) for b in range(11) if 0 <= L.SPEC_G - 100 * a - 10 * b <= 10]
        self.assertEqual(sols, [(2, 3)])

    def test_specimen_mass_and_size(self):
        vol = sum(s.volume() for s in ROCK.shapes)
        self.assertAlmostEqual(L.SPEC_MASS / vol / 1000, 2.7, delta=0.15)  # granite, g/cm3
        self.assertLessEqual(L.SPEC_BODY[1], 0.080)  # fits the gripper
        self.assertLess(L.SPEC_MASS, 0.4)

    def test_pan_is_6_inch(self):
        self.assertAlmostEqual(2 * L.PAN_R, 0.152, places=3)

    def test_weights_reach_inside_the_beams(self):
        self.assertLess(L.ZERO_D + L.R100_STEP * L.R100_N + L.RIDER_L / 2, L.BEAM_LEN)
        self.assertLess(L.ZERO_D + L.R10_STEP * L.R10_N + L.RIDER_L / 2, L.BEAM_LEN)
        self.assertLess(L.ZERO_D + L.POISE_SPAN + L.POISE_L / 2, L.BEAM_LEN)

    def test_grip_and_clearances(self):
        self.assertGreaterEqual(L.SPEC_BODY[1], 0.016)
        # beams 36 mm apart, riders 26 mm across: 10 mm between neighbouring riders
        self.assertGreaterEqual(abs(L.BEAM_DX["b500"] - L.BEAM_DX["poise"]) - L.RIDER_W, 0.009)

    def test_reach_and_height(self):
        self.assertLess(L.HZ + math.sin(math.radians(L.STOP_DEG)) * L.BEAM_LEN + L.RIDER_TOP + 0.02, 0.225)
        self.assertLess(L.PAN_Z + L.SPEC_H + 0.02, 0.225)


class Tiers(unittest.TestCase):
    def test_table(self):
        self.assertEqual(S.tier(False, True, True, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, True, True, True, False), 0.75)
        self.assertEqual(S.tier(False, True, True, True, False, False), 0.5)
        self.assertEqual(S.tier(False, True, True, False, False, False), 0.25)
        self.assertEqual(S.tier(False, True, False, True, True, True), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True, True), 0.0)
        self.assertEqual(S.tier(True, True, True, True, True, True), 0.0)


class Scorer(unittest.TestCase):
    def test_full_score(self):
        res, _ = run(bal_state(), rock_on_pan())
        self.assertEqual(res.score, 1.0, res.details)
        self.assertTrue(res.success)

    def test_full_score_with_the_beam_a_little_off(self):
        res, _ = run(bal_state(th=math.radians(0.9), qp=L.POISE_ANS + 0.004), rock_on_pan(th=math.radians(0.9)))
        self.assertEqual(res.score, 1.0, res.details)

    def test_nothing_done_is_zero(self):
        res, _ = run(bal_state(0, 0, 0, 0), rock_at(L.SPEC_X, L.SPEC_Y, 0.0))
        self.assertEqual(res.score, 0.0)

    def test_riders_set_without_the_specimen_is_zero(self):
        res, _ = run(bal_state(), rock_at(L.SPEC_X, L.SPEC_Y, 0.0))
        self.assertEqual(res.score, 0.0)

    def test_specimen_on_pan_only(self):
        res, _ = run(bal_state(-math.radians(8), 0, 0, 0), rock_on_pan(-math.radians(8)))
        self.assertEqual(res.score, 0.25)

    def test_riders_in_notches_poise_at_zero(self):
        res, _ = run(bal_state(-math.radians(8), qp=0.0), rock_on_pan(-math.radians(8)))
        self.assertEqual(res.score, 0.5)  # 230 g: 6.4 g short, the beam on its stop

    def test_poise_a_little_short_is_the_middle_tier(self):
        # reading 235 g: error -1.4 g (inside 2 g), beam 2.8 degrees
        res, _ = run(bal_state(th=-math.radians(2.8), qp=0.050), rock_on_pan(-math.radians(2.8)))
        self.assertEqual(res.score, 0.75, res.details)

    def test_pointer_off_zero_is_not_full(self):
        res, _ = run(bal_state(th=-math.radians(1.5)), rock_on_pan(-math.radians(1.5)))
        self.assertEqual(res.score, 0.75)

    def test_reading_off_by_a_gram_is_not_full(self):
        res, _ = run(bal_state(qp=L.POISE_ANS - 0.007), rock_on_pan())
        self.assertEqual(res.score, 0.75)

    def test_wrong_notch_is_quarter(self):
        res, _ = run(bal_state(q100=3 * STEP100, qp=0.0), rock_on_pan(math.radians(8)))
        self.assertEqual(res.score, 0.25)

    def test_rider_between_notches_is_not_in_the_notch(self):
        res, _ = run(bal_state(q100=2 * STEP100 + 0.006), rock_on_pan())
        self.assertEqual(res.score, 0.25)

    def test_beam_still_swinging_is_not_at_rest(self):
        t = S.Tracker()
        res = None
        for i in range(45):
            th = math.radians(1.5 * math.sin(i / 3))
            rigstate.offline_feed(FAR)
            res = t.update(bal_state(th=th), rock_on_pan(th), rigstate.arms())
        self.assertLess(res.score, 0.75)

    def test_hand_on_the_poise_is_not_full(self):
        near = {"left": FAR["left"], "right": {"tcp": [L.X0 - 0.036, L.YP - 0.07, L.TOP + 0.13],
                                               "tips": [L.X0 - 0.036, L.YP - 0.07, L.TOP + 0.117], "opening": 0.0}}
        res, _ = run(bal_state(), rock_on_pan(), near)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.details["hands_clear_1s"])

    def test_hand_holding_the_specimen_is_zero(self):
        x, y, z = L.X0, L.YP + L.LA, L.PAN_Z + 0.02
        held = {"left": FAR["left"], "right": {"tcp": [x, y, z + L.TOP + 0.02], "tips": [x, y, z + L.TOP + 0.007],
                                               "opening": 0.4}}
        res, _ = run(bal_state(), rock_on_pan(), held)
        self.assertEqual(res.score, 0.0)

    def test_unreadable_arms_is_zero(self):
        t = S.Tracker()
        rigstate.offline_feed(FAR)
        res = None
        for _ in range(45):
            res = t.update(bal_state(), rock_on_pan(), None)
        self.assertEqual(res.score, 0.0)

    def test_specimen_off_the_pan_beside_it(self):
        res, _ = run(bal_state(), rock_on_pan(dx=0.09))
        self.assertEqual(res.score, 0.0)

    def test_specimen_dropped_off_the_table_latches(self):
        t = S.Tracker()
        run(bal_state(), rock_at(0.5, 0.5, -0.1), tracker=t, steps=2)
        self.assertTrue(t.latched)
        res, _ = run(bal_state(), rock_on_pan(), tracker=t)
        self.assertEqual(res.score, 0.0)

    def test_specimen_on_the_table_under_the_pan_is_zero(self):
        res, _ = run(bal_state(), rock_at(L.X0, L.YP + L.LA, 0.0))
        self.assertEqual(res.score, 0.0)

    def test_specimen_not_still_is_zero(self):
        t = S.Tracker()
        res = None
        for i in range(45):
            rigstate.offline_feed(FAR)
            res = t.update(bal_state(), rock_on_pan(dx=0.0005 * i), rigstate.arms())
        self.assertEqual(res.score, 0.0)

    def test_missing_joint_is_zero(self):
        st = bal_state()
        st.joints.pop("poise")
        res, _ = run(st, rock_on_pan())
        self.assertEqual(res.score, 0.0)

    def test_new_episode_resets_the_latch(self):
        t = S.Tracker()
        run(bal_state(), rock_at(0.5, 0.5, -0.1), tracker=t, steps=2)
        self.assertTrue(t.latched)
        run(bal_state(0, 0, 0, 0), rock_at(L.SPEC_X, L.SPEC_Y, 0.0), tracker=t, steps=1)
        self.assertFalse(t.latched)


class Demo(unittest.TestCase):
    def test_check_task_predicts_full_score_and_one_arm(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(ROOT / "tools" / "offline" / "yam_stub"), str(ROOT / "tools" / "offline")]))
        out = subprocess.run([sys.executable, str(ROOT / "tools/offline/check_task.py"), str(TASK), "--one-arm",
                              "--json", "-"], capture_output=True, text=True, env=env, cwd=ROOT)
        text = out.stdout
        start = text.index("{")
        data = json.loads(text[start:text.rindex("}") + 1])
        self.assertEqual(data["playback"]["proxy"]["predicted_task_score"], 1.0)
        self.assertEqual(data["one_arm"], {"left_only": 0.0, "right_only": 1.0})

    def test_variants(self):
        import task
        for v in task.VARIANTS:
            self.assertIn("right", task.plans(v))
        with self.assertRaises(ValueError):
            task.plans("nonsense")

    def test_waypoints_stay_under_the_tcp_ceiling(self):
        import task
        for side, plan in task.plans().items():
            for w in plan:
                self.assertLessEqual(w.pos[2] - L.TOP, 0.225, (side, w.label))

    def test_demo_idle_and_hold(self):
        import task
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(sorted(task.demo()), ["right"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(task.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]


class GpuRun1Fixes(unittest.TestCase):
    """The physics and demo fixes after GPU run 1 (demo 0.25: unstable beam, rock off the pan's axis, pad over the
    10 g rider and the poise with the beam on its +8 stop, soft detents)."""

    def test_beam_spring_outweighs_the_parts_above_the_edge(self):
        cw = P.tune()
        g = P.gravity_stiffness(cw)
        self.assertGreater(g, L.K_BEAM)  # without the compensation the beam is an inverted pendulum
        self.assertAlmostEqual(P.beam_spring(cw) - g, L.K_BEAM, places=9)
        with open(TASK / "task.json") as f:
            setup = json.load(f)["setup"]
        self.assertIn(f"{P.beam_spring(cw):.3f} N m/rad", setup)

    def test_detent_pull_is_capped_and_holds_in_the_notch(self):
        import task
        notches = [k * STEP100 for k in range(L.R100_N + 1)]
        k = L.RIDER_DETENT_K["r100"]
        t = task.detent_pull_target(0.078, notches, L.RIDER_CAPTURE["r100"], k)
        self.assertAlmostEqual(k * (t - 0.078), L.RIDER_PULL, places=9)  # 2 N toward notch 2
        self.assertAlmostEqual(task.detent_pull_target(0.0798, notches, 0.010, k), 0.080, places=9)
        self.assertAlmostEqual(task.detent_pull_target(0.060, notches, 0.010, k), 0.060, places=9)  # between notches

    def test_roberval_rule(self):
        import task
        bal = bal_state(th=math.radians(-8.0))
        cy = float(task.ROCK_CM[1])  # the rock's centre of mass is 0.3 mm off its frame (the cap)
        self.assertAlmostEqual(task.roberval_target(bal, rock_on_pan(th=math.radians(-8.0), dy=-cy)), 0.0, places=6)
        near = task.roberval_target(bal, rock_on_pan(th=math.radians(-8.0), dy=-0.007 - cy))
        far = task.roberval_target(bal, rock_on_pan(th=math.radians(-8.0), dy=0.005 - cy))
        self.assertLess(near, 0.0)  # toward the edge: the rule adds the missing pan-down torque
        self.assertGreater(far, 0.0)
        expect = -L.SPEC_MASS * L.G * 0.007 * math.cos(math.radians(8.0)) / task.BEAM_SPRING
        self.assertAlmostEqual(near, expect, delta=abs(expect) * 0.05)
        self.assertEqual(task.roberval_target(bal_state(), rock_at(L.SPEC_X, L.SPEC_Y, 0.0)), 0.0)

    def test_pad_stays_on_the_weight_on_either_stop(self):
        import task
        s8 = math.sin(math.radians(8.0))
        for d_face, top, hi in ((0.0535, L.RIDER_TOP, 8.0), (0.076, L.POISE_TOP, 8.0), (0.088, L.RIDER_TOP, 4.0)):
            z = task.pad_tip_z(d_face, top, math.radians(hi)) - L.HZ
            self.assertGreater(z - (0.005 + d_face * s8), 0.002)  # over the bar with the beam on its -8 stop
            self.assertGreater(top - d_face * math.sin(math.radians(hi)) - z, 0.002)  # on the face at theta_hi


if __name__ == "__main__":
    unittest.main()
