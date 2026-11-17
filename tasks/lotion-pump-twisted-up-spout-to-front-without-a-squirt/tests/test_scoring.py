"""Offline tests for layout.py, scoring.py and task.py: synthetic bottle states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
PART = S.PART
FRONT = -math.pi / 2  # head turn that brings the spout from -y to -x


def bottle(x=L.BX, y=L.BY, z=L.TOP + L.BASE_Z, yaw=0.0, roll=0.0, pitch=0.0, twist=0.0, lift=L.TRAVEL,
           source="test"):
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.BOTTLE, (x, y, z, *q), joints={"lift": lift, "twist": twist}, source=source)


def scene(**kw):
    return {PART: bottle(**kw)}


def done(**kw):
    kw.setdefault("twist", FRONT)
    kw.setdefault("lift", 0.0)
    return scene(**kw)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.17, 0.45, L.TOP + 0.15)), "right": arm((0.15, -0.55, L.TOP + 0.15))}


def feed(tr, states, n=1, arms=FAR):
    res = None
    for _ in range(n):
        res = tr.update(states, arms)
    return res


def unlocked_tracker():
    """A tracker that has seen the lugs clear and the head rise, in the middle of a trial."""
    tr = S.Tracker()
    feed(tr, scene(twist=0.0, lift=L.TRAVEL))
    feed(tr, scene(twist=1.6, lift=L.TRAVEL))
    feed(tr, scene(twist=1.6, lift=0.012))
    feed(tr, scene(twist=1.6, lift=0.0))
    return tr


def score_of(states, arms=FAR, tr=None):
    tr = tr or S.Tracker()
    return feed(tr, states, WIN + 3, arms).score


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(L.TRAVEL, 0.023)
        self.assertAlmostEqual(L.HEAD_TOP_LOCKED, 0.148, places=6)
        self.assertAlmostEqual(L.HEAD_TOP_UP, 0.171, places=6)
        self.assertAlmostEqual(L.HEAD_TOP_UP - L.HEAD_TOP_LOCKED, L.TRAVEL, places=9)
        self.assertAlmostEqual(L.BASE_MASS + L.STEM_MASS + L.HEAD_MASS, L.BOTTLE_MASS, places=9)
        ml = math.pi * L.BODY_R ** 2 * L.BODY_H * 1e6
        self.assertTrue(300 < ml < 400, ml)  # a 12 fl oz (355 ml) bottle

    def test_grasp_widths(self):
        self.assertTrue(0.016 <= 2 * L.HEAD_R <= 0.080)
        self.assertTrue(0.016 <= 2 * L.BODY_R <= 0.080)

    def test_spout_sticks_out_past_the_head(self):
        self.assertGreater(L.SPOUT_R1, L.HEAD_R)
        self.assertAlmostEqual(L.SPOUT_R1, 0.048)

    def test_start_is_clear_of_the_arms_and_reachable(self):
        st = T.BOTTLE.rest_state()
        self.assertGreaterEqual(float(st.low[0]), 0.18)
        self.assertLess(L.BX, 0.40)  # the arms reach table x 0.10 to 0.50 in front of their bases
        self.assertLessEqual(abs(L.BY + 0.305), 0.10)  # close to the right arm's base line
        self.assertLess(float(T.SPLASH_WALL.rest_state().low[0]) - float(st.high[0]), 0.2)

    def test_spawn_rests_on_the_liner(self):
        st = T.BOTTLE.rest_state()
        self.assertAlmostEqual(float(st.low[2]) - L.TOP, L.LINER_H, places=9)
        liner = T.LINER.rest_state()
        self.assertAlmostEqual(float(liner.high[2]) - L.TOP, L.LINER_H, places=9)

    def test_start_spout_points_minus_y(self):
        r = S.read(scene())
        self.assertAlmostEqual(S.spout_bearing(r), -90.0, places=5)
        self.assertAlmostEqual(S.spout_off_deg(r), 90.0, places=5)

    def test_joints(self):
        lift, twist = T.BOTTLE.link("lift"), T.BOTTLE.link("twist")
        self.assertEqual((lift.lower, lift.upper, lift.axis), (0.0, L.TRAVEL, "-Z"))
        self.assertEqual(twist.parent, "lift")
        self.assertAlmostEqual(lift.spring * 0.004, 1.6, places=6)
        self.assertAlmostEqual(twist.spring * L.LUG_RAD, 0.035, delta=0.001)
        self.assertGreater(twist.upper, math.pi)

    def test_twist_turns_the_spout(self):
        for tw_deg, want in ((0, -90), (95, 5), (-90, -180), (180, 90)):
            r = S.read(scene(twist=math.radians(tw_deg)))
            self.assertAlmostEqual(L.off_front_deg(S.spout_bearing(r)), L.off_front_deg(want), places=4)

    def test_head_rises_with_lift(self):
        lo, hi = T.BOTTLE.rest_state(), T.BOTTLE.rest_state()
        locked = bottle()
        free = bottle(lift=0.0)
        self.assertAlmostEqual(float(free.high[2] - locked.high[2]), L.TRAVEL, places=6)

    def test_friction_torque_exceeds_the_lock(self):
        mu = (L.LINER_FRICTION + L.BOTTLE_FRICTION) / 2
        cap = 2 / 3 * mu * L.BOTTLE_MASS * 9.81 * L.BODY_R
        self.assertGreater(cap, 1.8 * L.K_TWIST * L.LUG_RAD)


class Tiers(unittest.TestCase):
    def test_full(self):
        self.assertEqual(score_of(done(), tr=self._tr()), 1.0)
        self.assertTrue(feed(self._tr(), done(), WIN + 3).success)

    def _tr(self):
        return unlocked_tracker()

    def test_full_needs_the_lugs_latch(self):
        self.assertEqual(score_of(done(), tr=unlocked_tracker()), 1.0)
        self.assertEqual(score_of(done()), 0.0)  # head up and turned, never through the lugs: not unlocked

    def test_spout_elsewhere_is_05(self):
        self.assertEqual(score_of(scene(twist=math.radians(95), lift=0.0), tr=unlocked_tracker()), 0.5)

    def test_spout_edges(self):
        for off, want in ((14.9, 1.0), (15.0, 1.0), (15.2, 0.5), (-14.9, 1.0), (-15.2, 0.5)):
            self.assertEqual(score_of(done(twist=FRONT + math.radians(off)), tr=unlocked_tracker()), want, off)

    def test_either_way_round_counts(self):
        self.assertEqual(score_of(done(twist=math.radians(270)), tr=unlocked_tracker()), 1.0)

    def test_locked_after_turning_30_is_025(self):
        tr = S.Tracker()
        feed(tr, scene(twist=math.radians(35)), 3)
        self.assertEqual(score_of(scene(), tr=tr), 0.25)

    def test_turn_30_edge(self):
        for deg, want in ((29.0, 0.0), (30.0, 0.25), (45.0, 0.25)):
            tr = S.Tracker()
            feed(tr, scene(twist=math.radians(deg)), 2)
            self.assertEqual(score_of(scene(), tr=tr), want, deg)

    def test_unlocked_edges(self):
        for lift, want in ((0.0, 1.0), (0.005, 1.0), (0.0052, 0.25), (0.012, 0.25)):
            tr = S.Tracker()
            feed(tr, scene(twist=1.6, lift=L.TRAVEL))
            feed(tr, scene(twist=1.6, lift=lift))  # the head rose only this far
            self.assertEqual(score_of(done(lift=lift), tr=tr), want, lift)

    def test_still_locked_untouched_is_0(self):
        self.assertEqual(score_of(scene()), 0.0)

    def test_standing_edges(self):
        ok = math.radians(7.9)
        self.assertEqual(score_of(done(roll=ok), tr=unlocked_tracker()), 1.0)
        self.assertEqual(score_of(done(roll=math.radians(8.5)), tr=unlocked_tracker()), 0.0)
        for dx, want in ((0.029, 1.0), (0.031, 0.0)):
            self.assertEqual(score_of(done(x=L.BX + dx), tr=unlocked_tracker()), want, dx)
        self.assertEqual(score_of(done(y=L.BY - 0.031), tr=unlocked_tracker()), 0.0)

    def test_possible_scores(self):
        self.assertEqual({0.0, 0.25, 0.5, 1.0}, {S.tier(False, True, True, True, True, True),
                                                S.tier(False, True, True, True, False, True),
                                                S.tier(False, True, True, False, False, True),
                                                S.tier(False, True, True, False, False, False)} | {0.0})

    def test_stillness_edges(self):
        tr = unlocked_tracker()
        feed(tr, done(), WIN + 1)
        self.assertEqual(tr.last.score, 1.0)
        # a bottle creeping 4 mm over the second is still; 6 mm is not
        for creep, want in ((0.004, 1.0), (0.006, 0.0)):
            tr = unlocked_tracker()
            for i in range(WIN + 3):
                tr.update(done(x=L.BX + creep * min(i, WIN + 1) / (WIN + 1)), FAR)
            self.assertEqual(tr.last.score, want, creep)

    def test_head_still_moving_at_the_end_is_not_still(self):
        tr = unlocked_tracker()
        for i in range(WIN + 3):  # the head swinging 20 degrees over the last second
            tr.update(done(twist=FRONT + math.radians(20) * min(i, WIN) / WIN), FAR)
        self.assertEqual(tr.last.score, 0.0)
        tr = unlocked_tracker()
        for i in range(WIN + 3):  # the head still rising 8 mm
            tr.update(done(lift=0.008 - 0.008 * min(i, WIN) / WIN), FAR)
        self.assertEqual(tr.last.score, 0.0)


class ZeroConditions(unittest.TestCase):
    def test_tip_latches(self):
        tr = unlocked_tracker()
        feed(tr, done(roll=math.radians(50)))
        self.assertIsNotNone(tr.latch["TIPPED"])
        self.assertEqual(score_of(done(), tr=tr), 0.0)  # stood back up: still 0

    def test_tip_edge(self):
        for deg, latched in ((44.0, False), (46.0, True)):
            tr = unlocked_tracker()
            feed(tr, done(roll=math.radians(deg)))
            self.assertEqual(tr.latch["TIPPED"] is not None, latched, deg)

    def test_fall_latches(self):
        tr = unlocked_tracker()
        feed(tr, done(z=L.TOP + L.BASE_Z - 0.06))
        self.assertIsNotNone(tr.latch["FELL"])
        self.assertEqual(score_of(done(), tr=tr), 0.0)

    def test_stroke_after_free_latches(self):
        tr = unlocked_tracker()
        feed(tr, done(lift=0.0045))
        self.assertIsNotNone(tr.latch["STROKE"])
        self.assertEqual(score_of(done(), tr=tr), 0.0)

    def test_stroke_edge(self):
        for lift, latched in ((0.0039, False), (0.0041, True)):
            tr = unlocked_tracker()
            feed(tr, done(lift=lift))
            self.assertEqual(tr.latch["STROKE"] is not None, latched, lift)

    def test_stroke_is_measured_from_the_highest_the_head_has_been(self):
        tr = S.Tracker()
        feed(tr, scene(twist=1.6, lift=0.0))  # lugs clear, head already at the top
        feed(tr, scene(twist=1.6, lift=0.006))
        self.assertIsNotNone(tr.latch["STROKE"])

    def test_the_rise_itself_is_not_a_stroke(self):
        tr = S.Tracker()
        for lift in (0.023, 0.02, 0.012, 0.004, 0.0):
            feed(tr, scene(twist=1.6, lift=lift))
        self.assertIsNone(tr.latch["STROKE"])

    def test_head_pushed_down_a_little_then_up_again_is_fine(self):
        tr = unlocked_tracker()
        feed(tr, done(lift=0.003))
        feed(tr, done(lift=0.0))
        self.assertIsNone(tr.latch["STROKE"])
        self.assertEqual(score_of(done(), tr=tr), 1.0)

    def test_pressing_the_locked_head_is_no_stroke(self):
        tr = S.Tracker()
        feed(tr, scene(twist=0.2, lift=L.TRAVEL))
        self.assertIsNone(tr.latch["STROKE"])

    def test_hand_on_the_bottle(self):
        tr = unlocked_tracker()
        st = done()
        near = dict(FAR, right=arm((L.BX, L.BY, L.TOP + 0.10), 0.3))
        self.assertEqual(score_of(st, near, tr=tr), 0.0)

    def test_hand_over_the_head(self):
        near = dict(FAR, right=arm((L.BX, L.BY, L.TOP + L.HEAD_TOP_UP + 0.02 + T.PAD_TIP)))
        self.assertEqual(score_of(done(), near, tr=unlocked_tracker()), 0.0)

    def test_parked_hands_are_clear(self):
        self.assertEqual(score_of(done(), tr=unlocked_tracker()), 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(score_of(done(), arms=None, tr=unlocked_tracker()), 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = unlocked_tracker()
        near = dict(FAR, right=arm((L.BX, L.BY + 0.02, L.TOP + 0.10), 0.3))
        for i in range(WIN + 3):
            tr.update(done(), near if i == 3 else FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_moving_bottle(self):
        tr = unlocked_tracker()
        for i in range(WIN + 3):
            tr.update(done(x=L.BX + 0.012 * i / WIN), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_missing_state_scores_zero(self):
        tr = unlocked_tracker()
        self.assertEqual(tr.update({}, FAR).score, 0.0)
        self.assertEqual(tr.update(None, FAR).score, 0.0)

    def test_nan_state(self):
        tr = unlocked_tracker()
        bad = bottle(twist=FRONT, lift=0.0)
        bad.joints["twist"] = float("nan")
        self.assertEqual(tr.update({PART: bad}, FAR).score, 0.0)
        bad2 = bottle(twist=FRONT, lift=0.0)
        bad2.joints["lift"] = float("inf")
        self.assertEqual(tr.update({PART: bad2}, FAR).score, 0.0)
        bad3 = bottle(twist=FRONT, lift=0.0)
        bad3.origin[0] = float("nan")
        self.assertEqual(tr.update({PART: bad3}, FAR).score, 0.0)

    def test_nan_resets_the_windows(self):
        tr = unlocked_tracker()
        feed(tr, done(), WIN + 3)
        self.assertEqual(tr.last.score, 1.0)
        bad = bottle(twist=FRONT, lift=0.0)
        bad.joints["twist"] = float("nan")
        tr.update({PART: bad}, FAR)
        feed(tr, done(), WIN - 2)
        self.assertEqual(tr.last.score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = unlocked_tracker()
        feed(tr, done(roll=math.radians(60)))
        self.assertIsNotNone(tr.latch["TIPPED"])
        feed(tr, scene())  # the reset: bottle on its mark, head locked again
        self.assertIsNone(tr.latch["TIPPED"])
        self.assertIsNone(tr.latch["LUGS"])
        self.assertEqual(tr.resets, 1)

    def test_a_stroke_latch_survives_until_a_reset(self):
        tr = unlocked_tracker()
        feed(tr, done(lift=0.02))
        for _ in range(3):
            feed(tr, done())
        self.assertIsNotNone(tr.latch["STROKE"])

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(3)
        tr = S.Tracker()
        for _ in range(400):
            st = scene(x=L.BX + rng.normal(0, 0.05), y=L.BY + rng.normal(0, 0.05), roll=rng.normal(0, 0.5),
                       pitch=rng.normal(0, 0.5), yaw=rng.uniform(-3, 3), twist=rng.uniform(-3.4, 3.4),
                       lift=rng.uniform(0, L.TRAVEL))
            res = tr.update(st, FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))


class Exploits(unittest.TestCase):
    def test_pulling_the_head_up_without_twisting(self):
        """A straight pull on the head (no turn): up 23 mm, spout where it was. Not unlocked, not turned."""
        self.assertEqual(score_of(scene(twist=0.0, lift=0.0)), 0.0)

    def test_pulling_up_then_turning_the_spout_round_without_the_lugs(self):
        tr = S.Tracker()
        feed(tr, scene(twist=0.0, lift=0.0), 2)
        self.assertEqual(score_of(scene(twist=FRONT, lift=0.0), tr=tr), 0.0)  # clockwise from locked: never the lugs

    def test_turning_the_whole_bottle_instead_of_the_head(self):
        """The bottle itself turned a quarter turn on the liner so the spout faces -x, head still locked."""
        self.assertEqual(score_of(scene(yaw=-math.pi / 2)), 0.0)

    def test_turning_the_whole_bottle_after_unlocking_counts_the_spout_only(self):
        """Unlocked, then the whole bottle turned instead of the head: the spout bearing includes the bottle's yaw."""
        st = scene(yaw=-math.radians(95), twist=math.radians(5), lift=0.0)
        r = S.read(st)
        self.assertLessEqual(S.spout_off_deg(r), 15.0)
        self.assertEqual(score_of(st, tr=unlocked_tracker()), 1.0)  # a legitimate way round on a real bottle too

    def test_unlocking_then_pushing_the_head_down_and_up_again(self):
        tr = unlocked_tracker()
        for lift in (0.0, 0.012, 0.0):
            feed(tr, done(lift=lift))
        self.assertEqual(score_of(done(), tr=tr), 0.0)

    def test_holding_the_head_up_with_the_hand_for_good(self):
        held = dict(FAR, right=arm((L.BX, L.BY, L.TOP + L.HEAD_TOP_UP - 0.01 + T.PAD_TIP), 0.26))
        self.assertEqual(score_of(done(), held, tr=unlocked_tracker()), 0.0)

    def test_carrying_the_bottle_away(self):
        self.assertEqual(score_of(done(x=L.BX + 0.12), tr=unlocked_tracker()), 0.0)

    def test_throwing_the_bottle_over(self):
        self.assertEqual(score_of(done(roll=math.pi / 2, z=L.TOP + 0.03), tr=unlocked_tracker()), 0.0)

    def test_doing_nothing(self):
        self.assertEqual(score_of(scene()), 0.0)

    def test_screwing_the_head_up_half_way_and_leaving_it(self):
        """The head turned 60 degrees and left there by a hand that let go: 0.25 only (not unlocked)."""
        tr = S.Tracker()
        feed(tr, scene(twist=math.radians(60)), 2)
        self.assertEqual(score_of(scene(twist=math.radians(60)), tr=tr), 0.25)


class Rule(unittest.TestCase):
    def test_locked(self):
        self.assertEqual(T.bayonet_targets(0.3, False), (False, L.TRAVEL, 0.0))

    def test_lugs_clear(self):
        free, lift_t, tw_t = T.bayonet_targets(L.LUG_RAD + 0.01, False)
        self.assertTrue(free)
        self.assertEqual(lift_t, 0.0)
        self.assertAlmostEqual(tw_t, L.LUG_RAD + 0.01)

    def test_lugs_stay_clear(self):
        free, lift_t, tw_t = T.bayonet_targets(-1.0, True)
        self.assertTrue(free)
        self.assertEqual((lift_t, tw_t), (0.0, -1.0))

    def test_nan_twist_keeps_the_state(self):
        self.assertEqual(T.bayonet_targets(float("nan"), False), (False, L.TRAVEL, 0.0))
        self.assertEqual(T.bayonet_targets(float("nan"), True)[:2], (True, 0.0))

    def test_rule_writes_targets_and_resets(self):
        calls = []
        saved = T.mk.set_targets
        T.mk.set_targets = lambda spec, joint, value: calls.append((joint, value))
        try:
            T._RULE["free"] = False
            T._bayonet_rule({PART: bottle(twist=0.2)})
            T._bayonet_rule({PART: bottle(twist=1.5)})
            T._bayonet_rule({PART: bottle(twist=0.1)})
            self.assertTrue(T._RULE["free"])
            T._rule_reset()
            self.assertFalse(T._RULE["free"])
        finally:
            T.mk.set_targets = saved
            T._RULE["free"] = False
        self.assertEqual(calls[0], ("lift", L.TRAVEL))
        self.assertEqual(calls[2], ("lift", 0.0))
        self.assertEqual(calls[4], ("lift", 0.0))
        self.assertEqual(calls[5], ("twist", 0.1))


class Hooks(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["liner", "splash_wall", "bottle"])
        self.assertTrue(callable(T.success) and callable(T.score) and callable(T.demo))
        self.assertEqual(T.SCENE, "station")
        self.assertEqual(T.SCORE_OVER, "end")

    def test_instruction_matches_task_json(self):
        tj = json.loads((HERE / "task.json").read_text())
        self.assertEqual(T.INSTRUCTION, tj["instruction"])

    def test_demo_plans(self):
        p = T.plans()
        self.assertEqual(list(p), ["right"])
        self.assertEqual(len(p["right"]), len(T.plan_right()))

    def test_demo_env_switches(self):
        saved = os.environ.copy()
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            os.environ.pop("DEMO_HOLD_ONLY")
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(T.demo(), {})
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(list(T.demo()), ["right"])
            os.environ.pop("DEMO_IDLE_ARM")
            os.environ["DEMO_VARIANT"] = "nonsense"
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_demo_timing(self):
        dur = sum(w.duration for w in T.plans()["right"])
        self.assertLessEqual(dur * HZ, 800)  # within half the 1600-step limit
        self.assertEqual(json.loads((HERE / "task.json").read_text())["run_limit"], "1600 steps (80 s at 20 Hz)")

    def test_pads_close_across_the_spout(self):
        """At each grip the gripper's heading equals the spout's bearing (mod 180): pads along x for a spout along y."""
        twist = 0.0
        plan = T.plan_right()
        grips = [w for w in plan if w.label.startswith("grip the head")]
        self.assertEqual(len(grips), 3)
        for w, turn in zip(grips, (T.UNLOCK_TURN, T.SWING_1, T.SWING_2)):
            bearing = L.spout_bearing_deg(twist)
            d = (math.degrees(w.yaw) - bearing) % 180.0
            self.assertTrue(min(d, 180.0 - d) < 1e-6, (math.degrees(w.yaw), bearing))
            twist += turn
        self.assertAlmostEqual(L.off_front_deg(L.spout_bearing_deg(twist)), 0.0, places=6)

    def test_headings_stay_inside_the_wrist(self):
        for w in T.plan_right():
            self.assertTrue(-1.75 <= w.yaw <= 2.09, math.degrees(w.yaw))

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.GRIP_F, 0.024)
        self.assertLess(-2 * T.GRIP_F, 2 * L.HEAD_R)
        tips_locked = T.GRIP_TCP - T.PAD_TIP
        self.assertGreaterEqual(tips_locked - (L.BASE_Z + L.COLLAR_H + L.BODY_H + L.SHOULDER_H), 0.008)  # off the collar
        self.assertLess(tips_locked, L.BASE_Z + L.HEAD_TOP_LOCKED - 0.010)  # 10 mm or more of the head between the pads
        self.assertGreater(T.HOVER - T.PAD_TIP, L.BASE_Z + L.HEAD_TOP_UP + 0.008)  # the open hand clears the spout
        free_tips = T.FREE_TCP - T.PAD_TIP
        self.assertLess(free_tips, L.BASE_Z + L.HEAD_TOP_UP - 0.010)

    def test_rise_matches_the_travel(self):
        self.assertAlmostEqual(T.GRIP_TCP + T.RISE, T.FREE_TCP, delta=0.002)
        self.assertGreaterEqual(T.RISE, L.UNLOCK_UP)

    def test_no_demo_waypoint_goes_below_the_grip(self):
        for w in T.plan_right():
            self.assertGreaterEqual(w.pos[2] - L.TOP, T.GRIP_TCP - 1e-9, w.label)


def _play(plans, steps=None):
    """Run plans through the offline model and the scorer with the arms at the plans' TCP targets (no IK)."""
    from yam_arena import demo as stub_demo
    from yam_arena import station

    T._OFFLINE.reset()
    T._TRACKER.reset()
    saved = os.environ.copy()
    try:
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)
        T._OFFLINE.plans = {a: stub_demo.ArmPlan(list(w), station.BASES[a]) for a, w in plans.items()}
        end = max([sum(w.duration for w in v) for v in plans.values()] or [1.0]) + 4.0
        n = steps or int(end * HZ)
        idle = {"tcp": np.array([0.1, 0.0, L.TOP + 0.3]), "tips": np.array([0.1, 0.0, L.TOP + 0.287]), "opening": 1.0}
        success_step = None
        for k in range(1, n + 1):
            t = k / HZ
            snap = {}
            for a in ("left", "right"):
                if a in plans:
                    pos, _rot, finger, _label = T._OFFLINE.plans[a].target(t)
                    opening = float(np.clip(finger / rigstate.FINGER_OPEN, 0.0, 1.0))
                    snap[a] = {"tcp": np.array(pos), "tips": np.array(pos) - np.array([0.0, 0.0, T.PAD_TIP]),
                               "opening": opening}
                else:
                    dy = 0.3 if a == "left" else -0.3
                    snap[a] = dict(idle, tcp=idle["tcp"] + np.array([0.0, dy, 0]),
                                   tips=idle["tips"] + np.array([0.0, dy, 0]))
            rigstate.offline_feed(snap)
            T._OFFLINE.step(rigstate.arms())
            res = T._TRACKER.update(T._OFFLINE.states(), rigstate.arms())
            if res.success and success_step is None:
                success_step = k
        return {"score": T._TRACKER.last.score, "success": success_step is not None, "success_step": success_step,
                "details": T._TRACKER.last.details}
    finally:
        os.environ.clear()
        os.environ.update(saved)
        T._OFFLINE.reset()
        T._TRACKER.reset()


class OfflineModelTests(unittest.TestCase):
    def test_states_at_rest_match_the_spawn(self):
        T._OFFLINE.reset()
        st = T._OFFLINE.states()[PART]
        sp = T.BOTTLE.rest_state()
        self.assertTrue(np.allclose(st.origin, sp.origin))
        self.assertAlmostEqual(st.joints["lift"], L.TRAVEL)
        self.assertAlmostEqual(st.joints["twist"], 0.0)

    def test_demo_in_the_model_scores_one(self):
        res = _play(T.plans())
        self.assertEqual(res["score"], 1.0)
        self.assertTrue(res["success"])
        self.assertLessEqual(res["success_step"], 800)
        d = res["details"]
        self.assertAlmostEqual(d["spout_off_front_deg"], 0.0, delta=0.5)
        self.assertGreaterEqual(d["head_up_mm"], 18.0)
        self.assertEqual(d["stroke_depth_mm"], 0.0)

    def test_one_arm_idle_in_the_model(self):
        self.assertEqual(_play({"right": T.plans()["right"]})["score"], 1.0)  # the right arm alone does it
        self.assertEqual(_play({})["score"], 0.0)  # the left arm has no plan here and cannot reach the bottle

    def test_counterfactuals_in_the_model(self):
        self.assertEqual(_play(T.plans("half_turn"))["score"], 0.25)
        self.assertEqual(_play(T.plans("no_swing"))["score"], 0.5)
        self.assertEqual(_play(T.plans("press"))["score"], 0.0)

    def test_press_variant_latches_a_stroke(self):
        res = _play(T.plans("press"))
        self.assertGreater(res["details"]["stroke_depth_mm"], 4.0)

    def test_hold_only_scores_zero(self):
        self.assertEqual(_play({})["score"], 0.0)

    def test_head_springs_back_when_let_go_locked(self):
        """Gripped and turned 40 degrees, then let go: the locked head goes back to 0 and the lock stays down."""
        from yam_arena import demo as stub_demo, station
        T._OFFLINE.reset()
        w = [T.Waypoint(1.0, (L.BX, L.BY, L.TOP + T.GRIP_TCP), T.GRIP_F, "grip", T.PSI0),
             T.Waypoint(1.0, (L.BX, L.BY, L.TOP + T.GRIP_TCP), T.GRIP_F, "turn", T.PSI0 + math.radians(40)),
             T.Waypoint(1.0, (L.BX, L.BY, L.TOP + T.GRIP_TCP), T.OPEN, "let go", T.PSI0 + math.radians(40))]
        res = _play({"right": w}, steps=int(3 * HZ))
        self.assertEqual(res["details"]["twist_deg"], 0.0)
        self.assertGreaterEqual(res["details"]["max_twist_deg"], 39.0)
        self.assertEqual(res["details"]["lift_mm"], 23.0)

    def test_high_gripper_does_not_grip(self):
        T._OFFLINE.reset()
        snap = {"right": {"tcp": np.array([L.BX, L.BY, L.TOP + 0.30]), "tips": np.array([L.BX, L.BY, L.TOP + 0.287]),
                          "opening": 0.26},
                "left": {"tcp": np.array([0.2, 0.3, L.TOP + 0.3]), "tips": np.array([0.2, 0.3, L.TOP + 0.287]),
                         "opening": 1.0}}
        rigstate.offline_feed(snap)
        T._OFFLINE.plans = {}
        T._OFFLINE.step(rigstate.arms())
        self.assertEqual(T._OFFLINE.hold, {})
        T._OFFLINE.reset()


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        tj = json.loads((HERE / "task.json").read_text())
        setup, scoring = tj["setup"], tj["scoring"]
        for token in ("(0.280, -0.250)", "220 x 220 x 3 mm", "x 0.430 to 0.450", "y -0.620 to -0.020", "300 mm tall",
                      "12 fl oz (355 ml)", "66 mm across", "107 mm to its shoulder", "0.42 kg", "148 mm tall",
                      "32 mm across and 26 mm tall", "48 mm from the bottle's axis", "top 148 mm up", "80 degrees",
                      "23 mm", "top 171 mm up", "0.035 N m", "1.6 N for the first 4 mm", "polypropylene", "HDPE",
                      "friction 1.0", "friction 0.8", "dip tube cut off", "sand"):
            self.assertIn(token, setup, token)
        for token in ("under 8 degrees", "within 30 mm", "80 degrees or more", "18 mm or more above its locked height",
                      "within 15 degrees of -x", "30 mm or more", "under 5 mm", "under 5 degrees", "more than 45 degrees",
                      "more than 4 mm below the highest", "50 mm below its start height",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "0.25 not unlocked", "30 degrees or more",
                      "Success: 1.0 held for 1 s", "8 degree angle gauge", "protractor", "60 fps"):
            self.assertIn(token, scoring, token)
        self.assertEqual(L.BX, 0.28)
        self.assertEqual(L.BY, -0.25)
        self.assertEqual(L.UNLOCK_UP, 0.018)
        self.assertEqual(L.STROKE, 0.004)
        self.assertEqual(L.SPOUT_TOL_DEG, 15.0)
        self.assertEqual(L.STAND_TILT_DEG, 8.0)
        self.assertEqual(L.STAND_XY, 0.030)
        self.assertEqual(L.TIP_DEG, 45.0)
        self.assertEqual(L.TURN_PART_DEG, 30.0)
        self.assertEqual(L.FALL_DROP, 0.050)
        self.assertEqual(L.LET_GO, 0.030)
        self.assertEqual(L.STILL_MOVE, 0.005)
        self.assertEqual(L.STILL_TURN_DEG, 5.0)
        self.assertEqual(L.LUG_DEG, 80.0)

    def test_fidelity_states_the_estimates(self):
        tj = json.loads((HERE / "task.json").read_text())
        for token in ("Rigid bottle", "torsion spring", "springs back", "stroke is scored", "estimates"):
            self.assertIn(token, tj["fidelity"])

    def test_scores_field(self):
        tj = json.loads((HERE / "task.json").read_text())
        self.assertRegex(tj["scores"], r"demo: 1\.0")

    def test_field_limits(self):
        tj = json.loads((HERE / "task.json").read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(tj[k]), n, k)

    def test_names_nobody(self):
        for f in ("task.json", "README.md"):
            text = (HERE / f).read_text()
            for bad in ("T-70", "Vihan", "Anthropic", "Claude", "robocurve-team", "/home/user", "~/"):
                self.assertNotIn(bad, text, f"{f}: {bad}")

    def test_text_rules(self):
        for f in ("task.json", "README.md", "task.py", "scoring.py", "layout.py"):
            text = (HERE / f).read_text()
            self.assertNotIn("—", text, f)  # no em-dashes
            self.assertNotIn("Co-Authored", text, f)

    def test_readme_has_the_sections(self):
        text = (HERE / "README.md").read_text()
        for token in ("inspect-robots run", "Runtime", "Seed", "Offline", "Bench", "Third-party assets: none",
                      "Acceptance", "Results"):
            self.assertIn(token, text, token)

    def test_no_pycache(self):
        self.assertTrue(sys.dont_write_bytecode)


if __name__ == "__main__":
    unittest.main()
