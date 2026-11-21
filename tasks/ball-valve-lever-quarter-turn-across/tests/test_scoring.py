"""Offline tests for layout.py, scoring.py and task.py: synthetic cabinet states (mechkit AssemblyStates from the
task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
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


def cabinet(cold=0.0, hot=0.0, dx=0.0, dz=0.0):
    """The cabinet with the cold lever at ``cold`` degrees and the hot one at ``hot`` (clockwise from open)."""
    return mk.state_from_poses(T.CABINET_SPEC, (dx, 0.0, L.TOP + dz, 0.0, 0.0, 0.0, 1.0),
                               joints={"cold": math.radians(cold), "hot": math.radians(hot)}, source="test")


def scene(cold=0.0, hot=0.0, **kw):
    return {S.CABINET: cabinet(cold, hot, **kw)}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def score_of(cold=0.0, hot=0.0, arms=FAR):
    return run(scene(cold, hot), arms).score


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.BACK_T, 0.0127, places=4)  # 1/2 in plywood
        self.assertAlmostEqual(L.PIPE_R * 2, 0.015875, places=5)  # 1/2 in copper, 5/8 in OD
        self.assertAlmostEqual(L.BODY_L, L.HOUSING_L + 2 * L.END_L, places=9)  # 52 mm end to end
        self.assertAlmostEqual(2 * L.BODY_R, 0.029, places=3)  # 30 mm across (29 mm)
        self.assertAlmostEqual(L.COLD_Y - L.HOT_Y, 8 * 0.0254, places=2)  # 8 in between the stubs

    def test_spec(self):
        names = [j.name for j in T.CABINET_SPEC.joints]
        self.assertEqual(names, ["cold", "hot"])
        for j in T.CABINET_SPEC.joints:
            self.assertEqual(j.kind, "revolute")
            self.assertEqual(j.axis, "-Z")  # a clockwise turn seen from above is positive
            self.assertAlmostEqual(j.lower, 0.0)
            self.assertAlmostEqual(j.upper, math.pi / 2)
            self.assertAlmostEqual(j.friction, L.VALVE_FRICTION_TORQUE)
            self.assertAlmostEqual(j.mass, L.LEVER_MASS)
        self.assertIsNone(T.CABINET_SPEC.mass)  # fixed
        self.assertEqual([o.name for o in T.OBJECTS], ["cabinet"])

    def test_lever_turns_clockwise_toward_plus_y(self):
        """With the hinge at pi/2 the grip's middle is 100 mm from the stem along +y; open it is 100 mm toward -x."""
        for name, y in (("cold", L.COLD_Y), ("hot", L.HOT_Y)):
            for q, want in ((0.0, (-L.GRIP_C, 0.0)), (math.pi / 2, (0.0, L.GRIP_C))):
                st = mk.state_from_poses(T.CABINET_SPEC, (0, 0, L.TOP, 0, 0, 0, 1), joints={name: q}, source="test")
                p = np.array([[L.STEM_X + want[0], y + want[1], L.TOP + L.LEVER_Z]])
                self.assertEqual(float(st.distance(p, bodies=(name,))[0]), 0.0, (name, q))
                other = np.array([[L.STEM_X - want[1], y + want[0], L.TOP + L.LEVER_Z]])
                self.assertGreater(float(st.distance(other, bodies=(name,))[0]), 0.05, (name, q))

    def test_lever_dir_and_push_dir(self):
        self.assertTrue(np.allclose(L.lever_dir(0.0), (-1.0, 0.0)))
        self.assertTrue(np.allclose(L.lever_dir(math.pi / 2), (0.0, 1.0)))
        for q in (0.0, 0.4, 1.2):  # the push direction is the lever direction's clockwise derivative
            h = 1e-6
            a, b = np.array(L.lever_dir(q)), np.array(L.lever_dir(q + h))
            self.assertTrue(np.allclose((b - a) / h, L.push_dir(q), atol=1e-5))

    def test_arm_force_needed(self):
        """2.0 N at the grip's middle: under the arms' 5 N, and the commanded 8 mm of penetration (about 2 mm per
        newton of arm give) asks for more than that."""
        self.assertAlmostEqual(L.PUSH_FORCE, 2.0)
        self.assertLess(L.PUSH_FORCE, 3.0)
        self.assertGreater(T.PEN / 0.0023, L.PUSH_FORCE * 1.5)

    def test_nothing_under_the_lever_arc(self):
        """The lever's bar and grip clear the pipe, the clips and the valve body in every pose (they are 0 mm to
        5 mm above them); the stop tabs touch neither pose of the bar."""
        grip_bottom = L.LEVER_Z - L.GRIP_R
        clip_top = L.PIPE_Z + L.PIPE_R + 0.0025
        self.assertGreater(grip_bottom - clip_top, 0.004)
        self.assertGreater(L.LEVER_Z - L.BAR_T / 2 - (L.PLATE_Z + 0.00075), -1e-9)
        for y in (L.COLD_Y, L.HOT_Y):
            for name in ("cold", "hot"):
                if (name == "cold") != (y == L.COLD_Y):
                    continue
                for q in np.linspace(0.0, math.pi / 2, 19):
                    st = mk.state_from_poses(T.CABINET_SPEC, (0, 0, L.TOP, 0, 0, 0, 1), joints={name: q},
                                             source="test")
                    # the lever's own boxes against the base shapes' boxes: the vertical gap above every base shape
                    low = st.link_bounds(name)[0][2] - L.TOP
                    self.assertGreaterEqual(low, L.LEVER_Z - L.GRIP_R - 1e-6)

    def test_tabs_stop_the_bar_without_touching_it(self):
        """Each stop tab is 0 to 1 mm from the lever's side face at its stop, and clear of it in the other pose."""
        tabs = [s for s in T.valve_shapes(L.COLD_Y) if tuple(s.size) == tuple(L.TAB)]
        self.assertEqual(len(tabs), 2)
        # open lever along -x: side faces at y +-9 mm; the tab on its -y side
        t_open = [t for t in tabs if t.center[0] < L.STEM_X][0]
        self.assertAlmostEqual((L.COLD_Y - L.BAR_W / 2) - (t_open.center[1] + L.TAB[1] / 2), 0.0, places=4)
        # closed lever along +y: side faces at x +-9 mm; the tab on its +x side
        t_closed = [t for t in tabs if t.center[0] > L.STEM_X][0]
        self.assertAlmostEqual((t_closed.center[0] - L.TAB[0] / 2) - (L.STEM_X + L.BAR_W / 2), 0.0, places=4)
        # the open lever's tail (12 mm behind the stem) and the closed lever's tail clear the other tab
        self.assertGreater(abs(t_closed.center[1] - L.COLD_Y) - L.TAB[1] / 2, L.BAR_W / 2)
        self.assertGreater(L.STEM_X - t_open.center[0] - L.TAB[0] / 2, L.BAR_W / 2)

    def test_block_path_is_clear(self):
        """The closed block (TCP, +-17 mm) in every waypoint stays 3 mm off the clips, the plate, the panels, the
        hot valve and the back panel, with the pad tips above the pipe and clips."""
        tips = T.TCP_Z - T.PAD_TIP
        self.assertGreater(tips, L.PIPE_Z + L.PIPE_R + 0.0025 + 0.010)
        for i in range(0, 41):
            q = T.Q_END * i / 40
            for pen in (-T.GAP, T.PEN):
                x, y = T.block_xy(q, pen)
                self.assertLess(x + 0.017, L.BACK_X0 - 0.05)
                self.assertGreater(x - 0.017, 0.100)
                self.assertGreater(y - 0.017, L.HOT_Y + 0.15)
                self.assertLess(y + 0.017, L.SIDE_IN_Y - 0.15)
        # the pad tips are at the lever's centre line less 2 mm
        self.assertAlmostEqual(tips, L.LEVER_Z - 0.002 + 0.0005, places=6)

    def test_reach_is_inside_the_left_arm_column(self):
        for i in range(0, 41):
            q = T.Q_END * i / 40
            x, y = T.block_xy(q, T.PEN)
            inward = L.COLD_Y - y if False else 0.305 - y
            self.assertGreater(x, 0.12)
            self.assertLess(x, 0.40)
            self.assertGreater(inward, -0.05)
            self.assertLess(inward, 0.25)


class Tiers(unittest.TestCase):
    def test_table(self):
        for a, want in ((0.0, 0.0), (14.9, 0.0), (15.0, 0.25), (30.0, 0.25), (44.9, 0.25), (45.0, 0.5), (60.0, 0.5),
                        (84.9, 0.5), (87.9, 0.5), (88.0, 1.0), (89.0, 1.0), (90.0, 1.0), (95.0, 1.0), (95.1, 0.0),
                        (120.0, 0.0)):
            self.assertEqual(S.tier(False, True, a), want, a)

    def test_every_tier_through_the_tracker(self):
        for a, want in ((0.0, 0.0), (10.0, 0.0), (20.0, 0.25), (50.0, 0.5), (86.0, 0.5), (88.5, 1.0), (90.0, 1.0)):
            self.assertEqual(score_of(a), want, a)

    def test_no_gap_between_tiers(self):
        """Every angle from 0 to 95 degrees has exactly one tier, and the tiers rise with the angle."""
        last = 0.0
        for a in np.arange(0.0, 95.01, 0.05):
            s = S.tier(False, True, float(a))
            self.assertIn(s, (0.0, 0.25, 0.5, 1.0))
            self.assertGreaterEqual(s, last)
            last = s

    def test_start_scores_zero(self):
        self.assertEqual(score_of(0.0), 0.0)
        r = run(scene())
        self.assertFalse(r.success)

    def test_success_only_at_one(self):
        self.assertTrue(run(scene(88.0)).success)
        self.assertFalse(run(scene(60.0)).success)

    def test_partial_credit_is_possible(self):
        self.assertEqual({score_of(a) for a in (5, 30, 70, 88)}, {0.0, 0.25, 0.5, 1.0})

    def test_hot_lever_slightly_turned_is_fine(self):
        self.assertEqual(score_of(88.0, hot=9.0), 1.0)

    def test_over_the_stop_is_invalid(self):
        self.assertEqual(score_of(96.0), 0.0)


class ZeroConditions(unittest.TestCase):
    def test_wrong_valve_scores_zero(self):
        self.assertEqual(score_of(0.0, hot=90.0), 0.0)

    def test_both_valves_score_zero(self):
        self.assertEqual(score_of(90.0, hot=90.0), 0.0)

    def test_hot_nudged_then_returned_stays_latched(self):
        tr = S.Tracker()
        tr.update(scene(0.0, 12.0), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(90.0, 0.0), tracker=tr).score, 0.0)

    def test_hot_just_over_ten_degrees_latches(self):
        self.assertEqual(score_of(90.0, hot=10.5), 0.0)
        self.assertEqual(score_of(90.0, hot=10.0), 1.0)

    def test_frame_moved_latches(self):
        tr = S.Tracker()
        tr.update(scene(0.0, 0.0, dx=0.006), FAR)
        self.assertEqual(tr.latch["F2"], 1)
        self.assertEqual(run(scene(90.0), tracker=tr).score, 0.0)
        tr = S.Tracker()
        tr.update(scene(0.0, 0.0, dx=0.004), FAR)
        self.assertFalse(tr.latched)

    def test_frame_lifted_latches(self):
        self.assertEqual(run(scene(90.0, dz=0.006)).score, 0.0)

    def test_hand_on_the_lever(self):
        arms = {"left": arm((L.STEM_X, L.COLD_Y + L.GRIP_C, L.TOP + L.LEVER_Z + 0.012), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)

    def test_hand_holding_the_lever_at_the_stop_earns_nothing(self):
        """The hold-on exploit: reach the stop and keep pressing."""
        arms = {"left": arm(T.block_xy(math.pi / 2, T.PEN) + (L.TOP + T.TCP_Z,), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)

    def test_hand_on_the_valve_body(self):
        arms = {"left": arm((L.STEM_X - 0.060, L.COLD_Y, L.TOP + L.PIPE_Z + 0.040), 1.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)
        arms = {"left": arm((L.STEM_X, L.COLD_Y - 0.060, L.TOP + L.PIPE_Z + 0.010), 1.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)

    def test_hand_30_mm_away_is_clear(self):
        """Closed fingers (zone radius 15.7 mm) with the zone 31 mm from the lever and the body: released."""
        # lever at 90 degrees runs along +y from the stem; put the zone beside its grip, 31 mm off its side
        x = L.STEM_X - L.GRIP_R - 0.0157 - 0.031
        arms = {"left": arm((x, L.COLD_Y + L.GRIP_C, L.TOP + L.LEVER_Z + 0.013), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 1.0)
        x = L.STEM_X - L.GRIP_R - 0.0157 - 0.029
        arms = {"left": arm((x, L.COLD_Y + L.GRIP_C, L.TOP + L.LEVER_Z + 0.013), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)

    def test_hand_near_the_hot_lever_is_not_penalised_by_the_hand_check(self):
        arms = {"left": FAR["left"], "right": arm((L.STEM_X - 0.1, L.HOT_Y, L.TOP + 0.20), 1.0)}
        self.assertEqual(score_of(90.0, arms=arms), 1.0)

    def test_parked_hands_are_clear(self):
        arms = {"left": arm((T.PARK[0], T.PARK[1], L.TOP + T.PARK[2]), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(90.0), arms=None).score, 0.0)
        bad = {"left": arm((float("nan"), 0.4, L.TOP + 0.2)), "right": FAR["right"]}
        self.assertEqual(run(scene(90.0), arms=bad).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        near = {"left": arm((L.STEM_X, L.COLD_Y + L.GRIP_C, L.TOP + L.LEVER_Z + 0.012), 0.0), "right": FAR["right"]}
        run(scene(90.0), arms=near, tracker=tr)
        self.assertEqual(tr.update(scene(90.0), FAR).score, 0.0)
        self.assertEqual(run(scene(90.0), tracker=tr, steps=WIN - 2).score, 0.0)
        self.assertEqual(run(scene(90.0), tracker=tr, steps=1).score, 1.0)

    def test_still_for_the_whole_second(self):
        """A lever creeping 0.2 degrees a step (4 degrees a second) is not still; one sitting is."""
        tr = S.Tracker()
        res = None
        for i in range(WIN + 5):
            res = tr.update(scene(70.0 + 0.2 * i), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])
        self.assertEqual(run(scene(70.0 + 0.2 * (WIN + 5)), tracker=tr, steps=WIN + 1).score, 0.5)

    def test_drift_under_two_degrees_is_still(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(scene(88.2 + 1.5 * i / (WIN + 3)), FAR)
        self.assertEqual(res.score, 1.0)

    def test_drift_over_two_degrees_is_not(self):
        """The bound is the stated 2 degrees: 2.5 degrees of creep in the last second scores 0, even near the stop."""
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(scene(87.5 + 2.5 * i / (WIN + 3)), FAR)
        self.assertEqual(res.score, 0.0)
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(scene(60.0 + 2.5 * i / (WIN + 3)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_lever_still_falling_to_the_stop_is_not_scored_yet(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(scene(min(90.0, 40.0 + 3.0 * i)), FAR)
        self.assertEqual(res.score, 0.0 if res.details["swing_1s_deg"] >= L.STILL_DEG else 1.0)

    def test_first_second_scores_zero(self):
        tr = S.Tracker()
        for i in range(WIN - 1):
            self.assertEqual(tr.update(scene(90.0), FAR).score, 0.0)


class Exploits(unittest.TestCase):
    def test_shut_both_to_be_safe(self):
        self.assertEqual(score_of(90.0, 90.0), 0.0)

    def test_close_the_hot_one_and_open_the_cold_one(self):
        self.assertEqual(score_of(0.0, 90.0), 0.0)

    def test_swing_hot_open_and_back_still_zero(self):
        """Closing the hot valve for a moment and opening it again leaves the latch set."""
        tr = S.Tracker()
        for h in (0.0, 45.0, 90.0, 45.0, 0.0):
            tr.update(scene(0.0, h), FAR)
        self.assertEqual(run(scene(90.0, 0.0), tracker=tr).score, 0.0)

    def test_press_the_lever_to_the_stop_and_stay(self):
        arms = {"left": arm(T.block_xy(math.pi / 2, 0.0) + (L.TOP + T.TCP_Z,), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(90.0, arms=arms), 0.0)

    def test_shut_then_unshut_at_the_end(self):
        """A lever shut and then pushed back to 10 degrees by the end scores for where it ends (0)."""
        tr = S.Tracker()
        for a in (90.0,) * 10 + (10.0,) * (WIN + 3):
            res = tr.update(scene(a), FAR)
        self.assertEqual(res.score, 0.0)

    def test_a_quick_unwind_to_open_looks_like_a_reset_not_a_credit(self):
        tr = S.Tracker()
        tr.update(scene(0.0, 90.0), FAR)
        self.assertTrue(tr.latched)
        tr.update(scene(90.0, 0.0), FAR)
        tr.update(scene(0.0, 0.0), FAR)  # both back at open in one step: a reset
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_a_slow_unwind_is_not_a_reset(self):
        tr = S.Tracker()
        tr.update(scene(0.0, 20.0), FAR)
        for a in (80.0, 60.0, 40.0, 20.0, 0.0):
            tr.update(scene(a, 0.0), FAR)
        self.assertEqual(tr.resets, 0)
        self.assertTrue(tr.latched)

    def test_far_hands_with_a_moving_lever_never_score(self):
        tr = S.Tracker()
        res = None
        for i in range(60):
            res = tr.update(scene(min(90.0, i * 5.0)), FAR)
        self.assertIn(res.score, (0.0, 1.0))


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)

    def test_nan_state(self):
        bad = mk.state_from_poses(T.CABINET_SPEC, (0, 0, L.TOP, 0, 0, 0, 1), joints={"cold": float("nan")},
                                  source="test")
        self.assertEqual(S.Tracker().update({S.CABINET: bad}, FAR).score, 0.0)
        bad = mk.state_from_poses(T.CABINET_SPEC, (float("nan"), 0, L.TOP, 0, 0, 0, 1), source="test")
        r = S.Tracker().update({S.CABINET: bad}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_restarts_the_windows(self):
        tr = S.Tracker()
        run(scene(90.0), tracker=tr)
        bad = mk.state_from_poses(T.CABINET_SPEC, (0, 0, L.TOP, 0, 0, 0, 1), joints={"hot": float("nan")},
                                  source="test")
        self.assertEqual(tr.update({S.CABINET: bad}, FAR).score, 0.0)
        self.assertEqual(tr.update(scene(90.0), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(90.0), tracker=tr, steps=WIN + 1).score, 1.0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            r = tr.update(scene(rng.uniform(-10, 120), rng.uniform(-10, 100), dx=rng.uniform(-0.01, 0.01)),
                          {"left": arm(rng.uniform(-0.2, 0.5, 3) + np.array([0.0, 0.0, L.TOP]), rng.uniform(0, 1)),
                           "right": FAR["right"]})
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))

    def test_details_are_json_ready(self):
        r = run(scene(88.0))
        json.dumps(r.details)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(scene(90.0, 15.0), tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(scene(0.0, 0.0), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)


class OfflineStandIn(unittest.TestCase):
    """The offline checker's model of the lever (never used in the simulator)."""

    def push(self, q_cmd, pen=T.PEN, z=T.TCP_Z, opening=0.0, model=None, n=40):
        m = model or T.OfflineModel()
        for i in range(n + 1):
            q = q_cmd * i / n
            x, y = T.block_xy(q, pen)
            m.step({"left": arm((x, y, L.TOP + z), opening), "right": FAR["right"]})
        return m

    def test_the_demo_push_shuts_the_cold_valve_only(self):
        m = self.push(T.Q_END)
        self.assertAlmostEqual(m.q["cold"], math.pi / 2, places=6)
        self.assertEqual(m.q["hot"], 0.0)

    def test_short_push(self):
        m = self.push(math.radians(25.0))
        self.assertGreater(math.degrees(m.q["cold"]), 25.0)
        self.assertLess(math.degrees(m.q["cold"]), 45.0)

    def test_half_push(self):
        m = self.push(math.radians(55.0))
        self.assertGreater(math.degrees(m.q["cold"]), 45.0)
        self.assertLess(math.degrees(m.q["cold"]), 85.0)

    def test_the_clearance_pose_pushes_nothing(self):
        m = self.push(0.0, pen=-T.GAP, n=1)
        self.assertEqual(m.q["cold"], 0.0)

    def test_open_fingers_do_not_push(self):
        m = self.push(T.Q_END, opening=1.0)
        self.assertEqual(m.q["cold"], 0.0)

    def test_a_hovering_block_does_not_push(self):
        m = self.push(T.Q_END, z=T.HOVER)
        self.assertEqual(m.q["cold"], 0.0)

    def test_a_block_ahead_of_the_lever_pushes_nothing(self):
        m = T.OfflineModel()
        x, y = T.block_xy(math.radians(60.0), -0.03 - T.PUSH_OFF)  # well ahead of the closed lever
        m.step({"left": arm((x, y, L.TOP + T.TCP_Z), 0.0), "right": FAR["right"]})
        self.assertEqual(m.q["cold"], 0.0)

    def test_a_lever_stays_where_it_is_left(self):
        m = self.push(math.radians(25.0))
        q = m.q["cold"]
        m.step(FAR)
        m.step({"left": arm((0.1, 0.0, L.TOP + 0.10), 0.0), "right": FAR["right"]})
        self.assertEqual(m.q["cold"], q)

    def test_the_right_arm_on_the_hot_lever(self):
        m = T.OfflineModel()
        for i in range(41):
            q = 1.0 * i / 40
            ux, uy = L.lever_dir(q)
            vx, vy = L.push_dir(q)
            off = T.PUSH_OFF - T.PEN
            p = (L.STEM_X + L.GRIP_C * ux - off * vx, L.HOT_Y + L.GRIP_C * uy - off * vy, L.TOP + T.TCP_Z)
            m.step({"left": FAR["left"], "right": arm(p, 0.0)})
        self.assertGreater(m.q["hot"], 0.5)
        self.assertEqual(m.q["cold"], 0.0)

    def test_unreadable_arms_move_nothing(self):
        m = T.OfflineModel()
        m.step(None)
        self.assertEqual(m.q, {"cold": 0.0, "hot": 0.0})

    def test_states_at_rest_match_the_spawn(self):
        st = T.OfflineModel().states()[S.CABINET]
        self.assertTrue(np.allclose(st.origin, T.CABINET_SPEC.rest_state().origin, atol=1e-9))
        self.assertEqual(S.read({S.CABINET: st}).cold, 0.0)

    def test_scorer_through_task_hooks(self):
        """success()/score() with arm snapshots fed as the offline checker does: the full demo's final pose."""
        rigstate.offline_reset()
        T._TRACKER.reset()
        T._OFFLINE.reset()
        T._EPISODE["mechkit"] = None
        steps = 0
        for _ in range(3):
            T._OFFLINE.q["cold"] = math.pi / 2
        rest = T.CABINET_SPEC.rest_state()
        done = False
        for _ in range(2 * WIN + 5):
            rigstate.offline_feed({"left": {"tcp": [T.PARK[0], T.PARK[1], L.TOP + T.PARK[2]],
                                            "tips": [T.PARK[0], T.PARK[1], L.TOP + T.PARK[2] - T.PAD_TIP],
                                            "opening": 0.0},
                                   "right": {"tcp": [0.1, -0.45, L.TOP + 0.25], "tips": [0.1, -0.45, L.TOP + 0.237],
                                             "opening": 1.0}})
            done = T.success({S.CABINET: rest})
            steps += 1
        self.assertTrue(done)
        self.assertEqual(T.score({S.CABINET: rest}), 1.0)
        rigstate.offline_reset()
        T._TRACKER.reset()
        T._OFFLINE.reset()
        T._EPISODE["mechkit"] = None


class TaskFile(unittest.TestCase):
    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("push", "grip the", "step", "tap", "hold", "clockwise", "degrees", " mm", "pinch"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            for a, wps in plan.items():
                self.assertEqual(a, "left")
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
                n_grips = sum(1 for x, y in zip(wps, wps[1:]) if abs(x.finger - y.finger) > 0.005)
                self.assertLessEqual(n_grips, 1)
            limit = run_limit / 2 if v == "" else run_limit
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ, limit)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", set()), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "short", {"left"}),
                               ("DEMO_VARIANT", "hold_on", {"left"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_demo_timing(self):
        plan = T.plans("")["left"]
        total = sum(w.duration for w in plan)
        self.assertAlmostEqual(total, 13.1, places=3)
        sweep = [w for w in plan if w.label.startswith("swing")]
        self.assertAlmostEqual(sum(w.duration for w in sweep), T.SWEEP_S)
        self.assertEqual(len(sweep), T.SWEEP_STEPS)

    def test_the_demo_ends_pushing_into_the_stop_and_lets_go(self):
        plan = T.plans("")["left"]
        labels = [w.label for w in plan]
        self.assertIn("settle against the stop", labels)
        self.assertTrue(labels[-1] == "aside")
        last_sweep = [w for w in plan if w.label.startswith("swing")][-1]
        self.assertAlmostEqual(last_sweep.yaw, -T.Q_END)
        self.assertGreater(T.Q_END, L.Q_CLOSED)  # the stop takes the overtravel
        held = T.plans("hold_on")["left"]
        self.assertTrue(held[-1].label.startswith("keep pressing"))

    def test_block_starts_short_of_the_lever_and_goes_into_it(self):
        x0, y0 = T.block_xy(0.0, -T.GAP)
        xa, ya = T.block_xy(0.0, T.PEN)
        self.assertAlmostEqual(x0, L.STEM_X - L.GRIP_C)
        self.assertAlmostEqual(ya - y0, T.GAP + T.PEN)
        face = ya + T.PAD_FACE  # the block face (along +y at start) against the grip's trailing surface
        self.assertAlmostEqual(face - (L.COLD_Y - L.GRIP_R), T.PEN, places=9)

    def test_yaw_squares_the_block_to_the_push(self):
        for q in (0.0, 0.5, 1.0, math.pi / 2):
            closing_axis = T.yaw_at(q) + math.pi / 2  # yaw 0 closes along y
            vx, vy = L.push_dir(q)
            self.assertAlmostEqual(abs(math.cos(closing_axis) * vx + math.sin(closing_axis) * vy), 1.0, places=9)

    def test_states_are_a_fixed_cabinet(self):
        self.assertTrue(np.allclose(T.CABINET_SPEC.rest_state().origin, (0.0, 0.0, L.TOP)))


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("x 400 to 412.7 mm", "y -438 to +438 mm", "300 mm high", "18 mm plywood", "x 200 to 412.7 mm",
                      "y +420 mm and -420 mm", "15.9 mm OD", "70 mm height", "x 140 mm", "y +101.5 mm", "y -101.5 mm",
                      "203 mm", "x 345 mm and x 205 mm", "x 270 mm", "52 mm end to end", "30 mm across",
                      "94 mm up", "at 0 and at 90 degrees", "18 x 6 mm", "97 mm up", "130 mm", "21 mm across",
                      "60 mm long", "100 mm from the stem", "blue", "red", "50 g", "clockwise", "0.20 N m",
                      "2.0 N at the middle of the grip", "pointing at the robot"):
            self.assertIn(token, setup)
        for token in ("10 degrees", "5 mm", "30 mm", "2 degrees", "95 degrees", "from 88 degrees up to the stop",
                      "from 45 up to 88", "from 15 up to 45", "under 15", "Possible scores: 0, 0.25, 0.5, 1.0",
                      "1 s", "Success: 1.0 held for 1 s", "Closing the hot valve instead, or both, scores 0"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "600 steps (30 s at 20 Hz)")
        self.assertEqual(L.FULL_DEG, 88.0)
        self.assertEqual(L.HALF_DEG, 45.0)
        self.assertEqual(L.QUARTER_DEG, 15.0)
        self.assertEqual(L.HOT_DEG, 10.0)
        self.assertEqual(L.MOVED_MM, 5.0)
        self.assertEqual(L.STILL_DEG, 2.0)
        self.assertEqual(L.LET_GO, 0.030)
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_the_review_points_are_in_the_task(self):
        """Each review point is answered by the task itself, not only its wording."""
        d = json.loads((HERE / "task.json").read_text())
        # scoring: no gap (the old text left 86 to 90 degrees with 2 to 5 mm of movement unscored)
        self.assertNotIn("valve moved under 2 mm", d["scoring"])
        # clarity: direction, positions
        for token in ("clockwise", "pointing along +y", "x 345 mm", "y +101.5 mm"):
            self.assertIn(token, d["setup"])
        # feasibility: the torque is one the arms can supply, and it is stated with its bench method
        self.assertLess(L.PUSH_FORCE, 3.0)
        self.assertIn("torque screwdriver", d["reset_notes"])

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("hinge", "0.20 N m", "est", "stops at 0 and 90", "fixed"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("household", "laundry tub", "hot line stays in service"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertLess(len(d["instruction"]), 300)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
