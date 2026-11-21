"""Offline tests for scoring.py and task.py: synthetic bed states (mechkit AssemblyStates from the task's own spec) and
synthetic arm snapshots (rigstate.Arm), no simulator.  python3 -m unittest discover -s tests -v"""

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import unittest

import numpy as np

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
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline package

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

TWO_PI = 2 * math.pi
HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
D = math.radians
AWAY = {a: rigstate.Arm(np.array([0.20, s * 0.30, L.TOP + 0.25]), np.array([0.20, s * 0.30, L.TOP + 0.237]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}


# -- states --------------------------------------------------------------------------------------------------------


def bed(deck_deg=0.0, fold_deg=0.0, crank=0.0, w=0.0, knob=0.0):
    origin, quat = T.BED.spawn_pose()
    return mk.state_from_poses(T.BED, (*origin, *quat), source="offline",
                               joints={"deck": D(deck_deg), "fold": D(fold_deg), "crank": crank, "knob": knob},
                               joint_vel={"crank": w})


def arm_at(local, opening=0.2, dz=0.0):
    tcp = np.array([local[0], local[1], L.TOP + local[2] + dz])
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


def away_with(right_local=None, opening=0.2, dz=0.0):
    arms = dict(AWAY)
    if right_local is not None:
        arms["right"] = arm_at(right_local, opening, dz)
    return arms


def feed(tr, wi, st, arms, n):
    """Feed ``n`` identical control steps; returns the last result."""
    res = None
    for _ in range(n):
        res = tr.update({"bed": st}, arms, wi)
    return res


def settled(deck_deg, fold_deg=0.0, arms=None, tr=None, wi=None, n=WIN + 3):
    tr = tr or S.Tracker()
    wi = wi or S.Winder()
    return feed(tr, wi, bed(deck_deg, fold_deg), AWAY if arms is None else arms, n)


def turn(wi, turns, start=0.0, steps_per_turn=48, w=3.0):
    """Turn the crank ``turns`` (negative: backward) from the wrapped angle ``start`` in small steps; returns the end
    angle (wrapped)."""
    n = max(1, int(round(abs(turns) * steps_per_turn)))
    th = start
    wi.update(bed(crank=S.wrap_to_pi(th), w=0.0))  # the first reading only sets the reference angle
    for _ in range(n):
        th += math.copysign(TWO_PI / steps_per_turn, turns)
        wi.update(bed(crank=S.wrap_to_pi(th), w=math.copysign(w, turns)))
    return S.wrap_to_pi(th)


# -- the layout ------------------------------------------------------------------------------------------------------


class Layout(unittest.TestCase):
    def test_folded_and_unfolded_knob_places(self):
        c = L.knob_centre_local(0.0, 0.0)
        self.assertTrue(np.allclose(c, (0.280, -0.145, 0.115), atol=1e-9), c)
        c = L.knob_centre_local(0.0, L.FOLD_OPEN)
        self.assertTrue(np.allclose(c, (0.165, -0.170, 0.115), atol=1e-9), c)
        # a quarter turn on: the knob is at the bottom of its circle, 70 mm below the shaft
        c = L.knob_centre_local(math.pi / 2, L.FOLD_OPEN)
        self.assertTrue(np.allclose(c, (0.165, -0.100, 0.115 - 0.070), atol=1e-9), c)

    def test_knob_circle_is_140_mm_across(self):
        for phi in np.linspace(0, TWO_PI, 9):
            c = np.array(L.knob_centre_local(phi, L.FOLD_OPEN))
            self.assertAlmostEqual(math.hypot(c[1] - L.SHAFT_Y, c[2] - L.SHAFT_Z), 0.070, places=9)

    def test_crank_turns_clockwise_seen_from_the_robot_and_raises(self):
        # from the robot (looking +x, y to the left, z up) the knob goes from 3 o'clock (-y) down to 6 o'clock
        a = L.knob_centre_local(0.0, L.FOLD_OPEN)
        b = L.knob_centre_local(0.3, L.FOLD_OPEN)
        self.assertLess(b[2], a[2])
        self.assertGreater(b[1], a[1])
        self.assertGreater(L.DECK_PER_TURN, 0.0)

    def test_folded_handle_clears_the_gearbox_and_the_table(self):
        self.assertLess(L.PIN_X + L.ARM_LEN + L.ARM_BACK, L.BOX_X[0] - 0.040)
        self.assertLess(L.PIN_X + L.ARM_LEN + L.KNOB_R, L.BOX_X[0] - 0.040)
        self.assertGreater(L.SHAFT_Z - L.ARM_LEN - L.KNOB_R, 0.020)  # the knob's lowest point above the table

    def test_stub_and_tip(self):
        self.assertAlmostEqual(L.STUB_X[1] - L.STUB_X[0], 0.130)
        self.assertAlmostEqual(L.STUB_X[1], L.BOX_X[0])
        self.assertAlmostEqual(2 * L.STUB_R, 0.012)

    def test_masses_agree_with_sizes(self):
        stub = math.pi * (L.STUB_R * 1e2) ** 2 * (0.130 * 1e2) * 7.85 / 1000
        self.assertAlmostEqual(L.CRANK_MASS, stub, delta=0.10 * stub)
        flat = (0.095 * 0.020 * 0.005 * 1e6 + math.pi * 0.4 ** 2 * 3.0) * 7.85 / 1000
        self.assertAlmostEqual(L.ARM_MASS, flat, delta=0.25 * flat)
        knob = math.pi * (L.KNOB_R * 1e2) ** 2 * (L.KNOB_LEN * 1e2) * 1.14 / 1000
        self.assertAlmostEqual(L.KNOB_MASS, knob, delta=0.05 * knob)

    def test_crank_load_follows_from_the_deck(self):
        ratio = TWO_PI / L.DECK_PER_TURN  # crank radians per deck radian: 72
        self.assertAlmostEqual(ratio, 72.0, places=6)
        ideal = L.DECK_MASS * 9.81 * (L.DECK_LEN / 2) / ratio  # N m on the crank with no losses
        eta = 0.46  # a self-locking worm drive is under 50 % efficient
        total = L.BEARING_FRICTION + L.T_LOAD
        self.assertAlmostEqual(total, ideal / eta, delta=0.02)

    def test_deck_is_out_of_reach_and_stiff(self):
        self.assertGreaterEqual(L.DECK_HINGE_X, 0.50)  # the arms reach x 0.45 only near the table
        self.assertGreater(L.DECK_HINGE_Z - L.DECK_T / 2, 0.20)
        # a 50 N push at the deck's far edge moves it by under 2 degrees against its drive
        self.assertLess(math.degrees(50.0 * L.DECK_LEN / L.DECK_K), 2.0)

    def test_deck_drive_is_overdamped(self):
        inertia = L.DECK_MASS * (L.DECK_LEN ** 2) / 3.0
        self.assertGreater(L.DECK_DAMP, 2 * math.sqrt(L.DECK_K * inertia))

    def test_deck_rests_on_its_post(self):
        self.assertAlmostEqual(L.POST_Z[1], L.DECK_HINGE_Z - L.DECK_T / 2)
        self.assertLessEqual(L.POST_X[0], L.DECK_HINGE_X)
        self.assertGreaterEqual(L.POST_X[1], L.DECK_HINGE_X)

    def test_tiers_are_in_the_task_text(self):
        self.assertAlmostEqual(math.degrees(L.ANGLE_OK[0]), 30.0)
        self.assertAlmostEqual(math.degrees(L.ANGLE_OK[1]), 45.0)
        self.assertAlmostEqual(math.degrees(L.ANGLE_GATE), 60.0)
        self.assertAlmostEqual(math.degrees(L.FOLDED_MAX), 15.0)
        self.assertAlmostEqual(math.degrees(L.DECK_PER_TURN), 5.0)
        self.assertEqual(L.TURNS_MIN * 5.0, 30.0)
        self.assertEqual(L.TURNS_MAX * 5.0, 45.0)

    def test_kit_limits(self):
        for j in T.BED.joints:
            self.assertLessEqual(max(-j.lower, j.upper), TWO_PI)
        self.assertEqual({j.name for j in T.BED.joints}, {"crank", "fold", "knob", "deck"})
        self.assertEqual([j.parent for j in T.BED.joints], ["base", "crank", "fold", "base"])

    def test_kit_copies_are_unchanged(self):
        for name in ("mechkit.py", "rigstate.py"):
            kit = HERE.parents[1] / "tools" / "kit" / name
            if kit.exists():
                self.assertEqual((HERE / name).read_bytes(), kit.read_bytes(), name)

    def test_offline_box_and_state(self):
        centre, size, yaw = T.BED.offline_box()
        self.assertEqual(yaw, 0.0)
        self.assertTrue(np.allclose(size, (0.09, 0.09, 0.16)))
        self.assertEqual(T.BED.offline_state(None).joints["deck"], 0.0)

    def test_the_handle_in_the_state(self):
        st = bed(0.0, 90.0)
        k = st.link_to_world("knob", (L.PIN_X + L.ARM_LEN, L.SHAFT_Y - L.KNOB_OFF, L.SHAFT_Z))
        self.assertTrue(np.allclose(k, (0.165, -0.170, L.TOP + 0.115), atol=1e-6), k)
        st = bed(0.0, 0.0)
        k = st.link_to_world("knob", (L.PIN_X + L.ARM_LEN, L.SHAFT_Y - L.KNOB_OFF, L.SHAFT_Z))
        self.assertTrue(np.allclose(k, (0.280, -0.145, L.TOP + 0.115), atol=1e-6), k)


# -- the worm drive --------------------------------------------------------------------------------------------------


class WormDrive(unittest.TestCase):
    def test_nothing_done(self):
        wi = S.Winder()
        wi.update(bed())
        self.assertEqual(wi.turns, 0.0)
        self.assertEqual(wi.target, 0.0)
        self.assertEqual(wi.drag, 0.0)

    def test_five_degrees_a_turn(self):
        wi = S.Winder()
        turn(wi, 1.0)
        self.assertAlmostEqual(wi.turns, 1.0, places=2)
        self.assertAlmostEqual(math.degrees(wi.target), 5.0, delta=0.1)
        turn(wi, 7.0, start=0.0)
        self.assertAlmostEqual(math.degrees(wi.target), 40.0, delta=0.2)

    def test_wrap_counts_the_short_way(self):
        wi = S.Winder()
        wi.update(bed(crank=3.0))
        wi.update(bed(crank=-3.0))  # +0.283 rad forward across the wrap
        self.assertAlmostEqual(wi.th_un, TWO_PI - 6.0, places=6)  # the short way, 0.283 rad, not -6 rad
        wi.update(bed(crank=-2.9))
        self.assertAlmostEqual(wi.th_un, TWO_PI - 6.0 + 0.1, places=6)

    def test_backward_never_goes_below_flat(self):
        wi = S.Winder()
        turn(wi, -3.0)
        self.assertEqual(wi.turns, 0.0)
        self.assertEqual(wi.target, 0.0)
        end = turn(wi, 2.0)
        self.assertAlmostEqual(wi.turns, 2.0, places=1)

    def test_net_turns_not_total(self):
        wi = S.Winder()
        a = turn(wi, 4.0)
        a = turn(wi, -1.5, start=a)
        a = turn(wi, 3.5, start=a)
        self.assertAlmostEqual(wi.turns, 6.0, delta=0.05)
        self.assertAlmostEqual(math.degrees(wi.target), 30.0, delta=0.5)

    def test_rocking_back_and_forth_adds_nothing(self):
        wi = S.Winder()
        a = 0.0
        for _ in range(10):
            a = turn(wi, 0.5, start=a)
            a = turn(wi, -0.5, start=a)
        self.assertLess(wi.turns, 0.6)

    def test_target_stops_below_the_deck_limit(self):
        wi = S.Winder()
        turn(wi, 20.0)
        self.assertLessEqual(wi.target, L.DECK_LIMIT - 0.02 + 1e-9)

    def test_turning_the_knob_alone_does_nothing(self):
        wi = S.Winder()
        for k in range(100):
            wi.update(bed(crank=0.0, knob=S.wrap_to_pi(k * 0.3)))
        self.assertEqual(wi.turns, 0.0)

    def test_load_forward_only_and_saturating(self):
        wi = S.Winder()
        wi.update(bed(crank=0.0, w=3.0))
        self.assertAlmostEqual(wi.drag, L.T_LOAD)
        wi.update(bed(crank=0.0, w=L.W_SAT / 2))
        self.assertAlmostEqual(wi.drag, L.T_LOAD / 2)
        wi.update(bed(crank=0.0, w=0.0))
        self.assertEqual(wi.drag, 0.0)
        wi.update(bed(crank=0.0, w=-3.0))
        self.assertEqual(wi.drag, 0.0)

    def test_broken_states_change_nothing(self):
        wi = S.Winder()
        turn(wi, 2.0)
        before = wi.th_un
        info = wi.update(bed(crank=float("nan")))
        self.assertFalse(info["valid"])
        self.assertEqual(wi.th_un, before)
        info = wi.update(None)
        self.assertFalse(info["valid"])
        self.assertEqual(wi.drag, 0.0)

    def test_reset(self):
        wi = S.Winder()
        turn(wi, 3.0)
        wi.reset()
        self.assertEqual((wi.th_un, wi.target, wi.drag), (0.0, 0.0, 0.0))

    def test_crank_target(self):
        # no load, still: the target is the crank's angle; at speed it leads by w DT / 2 plus the damping term
        self.assertEqual(S.crank_target(1.0, 0.0, 0.0), 1.0)
        t = S.crank_target(1.0, 3.0, 0.0)
        self.assertAlmostEqual(t, 1.0 + 3.0 * L.DT / 2 + L.CRANK_DAMP * 3.0 / L.CRANK_K)
        t = S.crank_target(1.0, 3.0, L.T_LOAD)
        self.assertAlmostEqual(t, 1.0 + 3.0 * L.DT / 2 + (L.CRANK_DAMP * 3.0 - L.T_LOAD) / L.CRANK_K)
        self.assertEqual(S.crank_target(5.9, 6.0, 0.0), L.CRANK_LIMIT)
        self.assertEqual(S.crank_target(-5.9, -6.0, 0.0), -L.CRANK_LIMIT)

    def test_load_is_the_mean_torque_the_drive_spring_gives(self):
        # a steady crank at w: the spring torque K (target - q) - damping w averages the drag over the step
        w, drag = 3.0, L.T_LOAD
        q = 1.0
        tgt = S.crank_target(q, w, drag)
        mid = q + w * L.DT / 2  # the crank's angle at mid-step
        torque_mid = L.CRANK_K * (tgt - mid) - L.CRANK_DAMP * w
        self.assertAlmostEqual(torque_mid, -drag, places=9)

    def test_crank_drive_is_stable_when_folded_and_unfolded(self):
        for inertia in (2e-5, 3e-4):
            self.assertGreater(L.CRANK_DAMP, 0.0)
            self.assertLess(L.CRANK_DAMP * L.DT / inertia, 1e4)  # implicit drive: no step limit, just a sanity bound

    def test_wrap_shift(self):
        self.assertEqual(S.wrap_shift(0.0), 0.0)
        self.assertEqual(S.wrap_shift(math.pi - 0.01), 0.0)
        self.assertAlmostEqual(S.wrap_shift(math.pi + 0.01), -TWO_PI)
        self.assertAlmostEqual(S.wrap_shift(-math.pi - 0.01), TWO_PI)
        self.assertLess(abs(S.wrap_shift(3.2) + 3.2 - 0.0), 3.2)


# -- the tiers ---------------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_doing_nothing_scores_zero(self):
        r = settled(0.0)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        r = settled(0.0, fold_deg=90.0)
        self.assertEqual(r.score, 0.0)

    def test_full_score(self):
        r = settled(40.0)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["folded"])
        self.assertTrue(r.details["released_1s"])
        self.assertTrue(r.details["still_1s"])

    def test_every_tier_is_reachable(self):
        got = {
            "full": settled(37.5).score,
            "in range, handle out": settled(37.5, 90.0).score,
            "folded, a little low": settled(20.0).score,
            "folded, a little high": settled(50.0).score,
            "folded, too low": settled(10.0).score,
            "folded, past the gate": settled(65.0).score,
        }
        self.assertEqual(got, {"full": 1.0, "in range, handle out": 0.5, "folded, a little low": 0.25,
                               "folded, a little high": 0.25, "folded, too low": 0.0, "folded, past the gate": 0.0})

    def test_angle_edges(self):
        self.assertEqual(settled(30.0).score, 1.0)
        self.assertEqual(settled(45.0).score, 1.0)
        self.assertEqual(settled(29.9).score, 0.25)
        self.assertEqual(settled(45.1).score, 0.25)
        self.assertEqual(settled(15.0).score, 0.25)
        self.assertEqual(settled(14.9).score, 0.0)
        self.assertEqual(settled(60.0).score, 0.25)
        self.assertEqual(settled(60.1).score, 0.0)

    def test_fold_edge(self):
        self.assertEqual(settled(40.0, 14.9).score, 1.0)
        self.assertEqual(settled(40.0, 15.0).score, 1.0)
        self.assertEqual(settled(40.0, 15.2).score, 0.5)
        self.assertEqual(settled(20.0, 15.2).score, 0.0)  # unfolded and off range

    def test_unfolded_off_range_is_zero(self):
        self.assertEqual(settled(50.0, 90.0).score, 0.0)
        self.assertEqual(settled(20.0, 45.0).score, 0.0)

    def test_gate_is_latched_for_the_trial(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(61.0), AWAY, 2)
        r = feed(tr, wi, bed(40.0), AWAY, WIN + 5)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["over_60"])
        tr.reset()
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 5).score, 1.0)

    def test_gate_one_step_is_enough(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(20.0), AWAY, 5)
        feed(tr, wi, bed(60.5), AWAY, 1)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 5).score, 0.0)

    def test_gate_at_exactly_60_is_allowed(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(60.0), AWAY, 3)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 5).score, 1.0)

    def test_deck_raised_and_lowered_stays_valid_below_the_gate(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(55.0), AWAY, 3)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 5).score, 1.0)


class HandsOffAndStill(unittest.TestCase):
    knob = L.grip_point_local(0.0, 0.0)

    def test_held_knob_is_half(self):
        arms = away_with(self.knob, opening=0.18)
        r = settled(40.0, 0.0, arms)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["released_1s"])

    def test_gripper_on_the_unfolded_arm_is_half(self):
        arms = away_with(L.grip_point_local(0.0, L.FOLD_OPEN), opening=0.18)
        self.assertEqual(settled(40.0, 90.0, arms).score, 0.5)

    def test_hand_on_the_shaft_is_half(self):
        arms = away_with((0.27, L.SHAFT_Y, L.SHAFT_Z), opening=0.5)
        self.assertEqual(settled(40.0, 0.0, arms).score, 0.5)

    def test_hand_on_the_gearbox_counts_only_if_near_the_shaft_arm_or_knob(self):
        # the gearbox itself is not one of the released bodies: a hand 60 mm from the shaft but on the box is free
        arms = away_with((0.40, L.SHAFT_Y, L.BOX_Z[1] + 0.12), opening=1.0)
        self.assertEqual(settled(40.0, 0.0, arms).score, 1.0)

    def test_finger_zone_29_and_31_mm(self):
        st = bed(40.0, 0.0)
        tcp = np.array([0.20, L.SHAFT_Y, L.TOP + L.SHAFT_Z + 0.20])  # high above the stub; slide it down to the gap
        for gap, want in ((0.029, 0.5), (0.031, 1.0)):
            lo, hi = 0.0, 0.40
            for _ in range(60):  # find a TCP height whose zone gap to the handle bodies is `gap`
                mid = (lo + hi) / 2
                p = np.array([0.30, L.SHAFT_Y, L.TOP + L.SHAFT_Z + mid])
                arm = rigstate.Arm(p, p - np.array([0, 0, 0.013]), 0.0)
                g = rigstate.zone_gap(arm, lambda pts: st.distance(pts, bodies=L.RELEASE_BODIES))
                lo, hi = (lo, mid) if g > gap else (mid, hi)
            p = np.array([0.30, L.SHAFT_Y, L.TOP + L.SHAFT_Z + (lo + hi) / 2])
            arms = dict(AWAY)
            arms["right"] = rigstate.Arm(p, p - np.array([0, 0, 0.013]), 0.0)
            self.assertEqual(settled(40.0, 0.0, arms).score, want, gap)

    def test_left_hand_counts_too(self):
        arms = dict(AWAY)
        arms["left"] = arm_at(self.knob, 0.18)
        self.assertEqual(settled(40.0, 0.0, arms).score, 0.5)

    def test_unreadable_arms_never_release(self):
        tr, wi = S.Tracker(), S.Winder()
        r = feed(tr, wi, bed(40.0), None, WIN + 5)
        self.assertEqual(r.score, 0.5)
        bad = dict(AWAY)
        bad["right"] = rigstate.Arm(np.array([float("nan")] * 3), np.zeros(3), 1.0)
        self.assertEqual(settled(40.0, 0.0, bad).score, 0.5)

    def test_released_less_than_a_second(self):
        tr, wi = S.Tracker(), S.Winder()
        held = away_with(self.knob, 0.18)
        feed(tr, wi, bed(40.0), held, 5)
        r = feed(tr, wi, bed(40.0), AWAY, WIN - 1)
        self.assertEqual(r.score, 0.5)
        r = feed(tr, wi, bed(40.0), AWAY, 2)
        self.assertEqual(r.score, 1.0)

    def test_moving_deck_is_half(self):
        tr, wi = S.Tracker(), S.Winder()
        r = None
        for k in range(WIN + 5):
            r = tr.update({"bed": bed(36.0 + k * 0.3)}, AWAY, wi)  # 6 degrees a second
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["still_1s"])

    def test_creeping_deck_inside_half_a_degree_is_fine(self):
        tr, wi = S.Tracker(), S.Winder()
        r = None
        for k in range(WIN + 5):
            r = tr.update({"bed": bed(40.0 + k * 0.02)}, AWAY, wi)  # under 0.5 degrees over the window
        self.assertEqual(r.score, 1.0)

    def test_stillness_needs_a_full_second_of_states(self):
        tr, wi = S.Tracker(), S.Winder()
        r = feed(tr, wi, bed(40.0), AWAY, WIN - 1)
        self.assertEqual(r.score, 0.5)
        r = feed(tr, wi, bed(40.0), AWAY, 2)
        self.assertEqual(r.score, 1.0)

    def test_dropping_the_hold_for_a_step_restarts_the_second(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(40.0), AWAY, WIN + 3)
        feed(tr, wi, bed(40.0), away_with(self.knob, 0.18), 1)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN - 1).score, 0.5)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, 2).score, 1.0)


class BrokenStates(unittest.TestCase):
    def test_nan_state_scores_zero_and_restarts_windows(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(40.0), AWAY, WIN + 3)
        r = feed(tr, wi, bed(float("nan")), AWAY, 1)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        r = feed(tr, wi, bed(40.0), AWAY, WIN - 1)
        self.assertEqual(r.score, 0.5)
        r = feed(tr, wi, bed(40.0), AWAY, 3)
        self.assertEqual(r.score, 1.0)

    def test_nan_fold_and_missing_bed(self):
        tr, wi = S.Tracker(), S.Winder()
        self.assertEqual(feed(tr, wi, bed(40.0, float("nan")), AWAY, 3).score, 0.0)
        self.assertEqual(tr.update({}, AWAY, wi).score, 0.0)
        self.assertEqual(tr.update({"bed": None}, AWAY, wi).score, 0.0)
        self.assertGreaterEqual(tr.bad_steps, 5)

    def test_nan_deck_does_not_latch_the_gate(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(float("nan")), AWAY, 3)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 3).score, 1.0)

    def test_infinite_deck(self):
        import types
        fake = types.SimpleNamespace(joints={"deck": float("inf"), "fold": 0.0, "crank": 0.0}, joint_vel={},
                                     origin=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]), pos=np.zeros(3),
                                     distance=lambda pts, bodies=None: np.full(len(np.atleast_2d(pts)), 1.0))
        tr, wi = S.Tracker(), S.Winder()
        self.assertEqual(feed(tr, wi, fake, AWAY, 2).score, 0.0)
        self.assertEqual(tr.bad_steps, 2)

    def test_tracker_reset(self):
        tr, wi = S.Tracker(), S.Winder()
        feed(tr, wi, bed(40.0), AWAY, WIN + 3)
        tr.reset()
        self.assertEqual(tr.last.score, 0.0)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, 2).score, 0.5)

    def test_deck_falling_back_to_flat_resets_the_trial(self):
        tr, wi = S.Tracker(), S.Winder()
        turn(wi, 3.0)
        feed(tr, wi, bed(61.0), AWAY, 2)  # over the gate
        feed(tr, wi, bed(0.0), AWAY, 1)  # a new trial the reset hooks missed: a 61 degree drop in one step
        self.assertEqual(tr.resets, 1)
        self.assertFalse(tr.over)
        self.assertEqual(wi.turns, 0.0)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 3).score, 1.0)

    def test_slow_lowering_is_not_a_reset(self):
        tr, wi = S.Tracker(), S.Winder()
        for k in range(60):
            tr.update({"bed": bed(60.0 - k)}, AWAY, wi)
        self.assertEqual(tr.resets, 0)


class Exploits(unittest.TestCase):
    """Attempts to score without doing the work."""

    def test_pushing_the_deck_is_out_of_reach_and_stiff(self):
        # the deck's drive turns the angle by under 2 degrees for 50 N at the far edge, and nothing reaches it
        self.assertLess(math.degrees(50.0 * L.DECK_LEN / L.DECK_K), 2.0)
        self.assertGreater(L.DECK_HINGE_X, 0.45)

    def test_knob_spinning_in_the_fingers_raises_nothing(self):
        wi = S.Winder()
        for k in range(200):
            wi.update(bed(crank=0.0, knob=S.wrap_to_pi(k * 0.2), w=0.0))
        self.assertEqual(wi.target, 0.0)

    def test_folding_and_unfolding_repeatedly_raises_nothing(self):
        wi = S.Winder()
        for k in range(200):
            wi.update(bed(0.0, fold_deg=90.0 * (k % 2), crank=0.0))
        self.assertEqual(wi.turns, 0.0)

    def test_cranking_back_and_forth_does_not_bank_turns(self):
        wi = S.Winder()
        a = 0.0
        for _ in range(6):
            a = turn(wi, -2.0, start=a)
            a = turn(wi, 2.0, start=a)
        self.assertLess(wi.turns, 2.1)

    def test_overshoot_then_back_down_is_still_zero(self):
        tr, wi = S.Tracker(), S.Winder()
        a = turn(wi, 13.0)  # 65 degrees
        for deg in np.linspace(0.0, 65.0, 40):
            tr.update({"bed": bed(deg)}, AWAY, wi)
        a = turn(wi, -5.0, start=a)
        for deg in np.linspace(65.0, 40.0, 40):
            tr.update({"bed": bed(deg)}, AWAY, wi)
        self.assertEqual(feed(tr, wi, bed(40.0), AWAY, WIN + 5).score, 0.0)

    def test_hold_the_handle_and_leave_the_crank_out(self):
        arms = away_with(L.grip_point_local(0.0, L.FOLD_OPEN), 0.18)
        self.assertEqual(settled(40.0, 90.0, arms).score, 0.5)

    def test_holding_a_folded_handle_does_not_pass(self):
        arms = away_with(L.grip_point_local(0.0, 0.0), 0.18)
        self.assertEqual(settled(40.0, 0.0, arms).score, 0.5)

    def test_a_flat_deck_with_the_handle_folded_is_zero(self):
        self.assertEqual(settled(0.0, 0.0).score, 0.0)

    def test_a_deck_just_under_the_range_folded_is_a_quarter_not_more(self):
        self.assertEqual(settled(29.99, 0.0).score, 0.25)


# -- offline model ---------------------------------------------------------------------------------------------------


class OfflineBedModel(unittest.TestCase):
    def setUp(self):
        self.o = T.OfflineBed()

    def hand(self, local, opening=0.18):
        return {"right": arm_at(local, opening), "left": AWAY["left"]}

    def test_open_hand_leaves_the_bed(self):
        self.o.follow(self.hand(L.grip_point_local(0.0, 0.0), 0.44))
        self.assertEqual((self.o.phi, self.o.fold), (0.0, 0.0))

    def test_far_hand_leaves_the_bed(self):
        self.o.follow(self.hand((0.20, 0.0, 0.18)))
        self.assertEqual((self.o.phi, self.o.fold), (0.0, 0.0))

    def test_closed_hand_swings_the_handle_out(self):
        for f in np.linspace(0.0, L.FOLD_OPEN, 30):
            self.o.follow(self.hand(L.grip_point_local(0.0, f)))
        self.assertAlmostEqual(self.o.fold, L.FOLD_OPEN, places=2)
        self.assertAlmostEqual(S.wrap_to_pi(self.o.phi), 0.0, places=2)

    def test_closed_hand_turns_the_crank(self):
        self.o.fold = L.FOLD_OPEN
        for phi in np.linspace(0.0, 4.0, 80):
            self.o.follow(self.hand(L.grip_point_local(phi, L.FOLD_OPEN)))
        self.assertAlmostEqual(S.wrap_to_pi(self.o.phi - 4.0), 0.0, places=1)
        self.assertAlmostEqual(self.o.fold, L.FOLD_OPEN, places=2)

    def test_handle_folds_back_at_the_end(self):
        self.o.fold = L.FOLD_OPEN
        for f in np.linspace(L.FOLD_OPEN, 0.0, 30):
            self.o.follow(self.hand(L.grip_point_local(0.0, f)))
        self.assertAlmostEqual(self.o.fold, 0.0, places=2)

    def test_the_proxy_run_reaches_full_score(self):
        wi = S.Winder()
        tr = S.Tracker()
        for f in np.linspace(0.0, L.FOLD_OPEN, 30):
            tr.update({"bed": self.o.state(self.hand(L.grip_point_local(0.0, f)), wi)}, AWAY, wi)
        for k in range(1, 8 * 48 + 1):
            phi = TWO_PI * 8 * k / (8 * 48)
            st = self.o.state(self.hand(L.grip_point_local(phi, L.FOLD_OPEN)), wi)
            tr.update({"bed": st}, AWAY, wi)
        self.assertAlmostEqual(math.degrees(wi.target), 40.0, delta=1.0)
        for f in np.linspace(L.FOLD_OPEN, 0.0, 30):
            tr.update({"bed": self.o.state(self.hand(L.grip_point_local(0.0, f)), wi)}, AWAY, wi)
        r = None
        for _ in range(WIN + 5):
            r = tr.update({"bed": self.o.state(AWAY, wi)}, AWAY, wi)
        self.assertEqual(r.score, 1.0)


# -- the demo -----------------------------------------------------------------------------------------------------------


def _plans(variant="", idle="", hold=False):
    env = {"DEMO_VARIANT": variant, "DEMO_IDLE_ARM": idle, "DEMO_HOLD_ONLY": "1" if hold else ""}
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        return T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Demo(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(set(_plans()), {"right"})
        self.assertEqual(_plans(hold=True), {})
        self.assertEqual(_plans(idle="right"), {})
        self.assertEqual(set(_plans(idle="left")), {"right"})

    def test_timing_and_run_limit(self):
        plan = _plans()["right"]
        total = sum(w.duration for w in plan)
        self.assertLess(total, 40.0)
        spec = json.loads((HERE / "task.json").read_text())
        limit = int(re.match(r"([\d,]+) steps", spec["run_limit"]).group(1).replace(",", ""))
        steps = int(math.ceil(total * HZ)) + 60
        self.assertLessEqual(steps, limit)
        self.assertGreater(limit, steps)
        self.assertLessEqual(limit, 20 * steps)

    def test_open_and_close_are_slow_and_open_loop(self):
        plan = _plans()["right"]
        for w in plan:
            if "close" in w.label or "let go" in w.label:
                self.assertGreaterEqual(w.duration, 1.5)

    def test_grips(self):
        self.assertAlmostEqual(T.OPEN_KNOB * -2, 0.041)
        self.assertAlmostEqual(T.GRIP_KNOB * -2, 2 * L.KNOB_R - 0.008)
        self.assertLess(-2 * T.OPEN_KNOB, 0.0939)  # inside the gripper's full gap
        self.assertGreater(-2 * T.OPEN_KNOB, 2 * L.KNOB_R + 0.010)  # at least 5 mm a side
        self.assertGreater(-2 * T.GRIP_KNOB, 0.005)

    def test_the_hand_starts_over_the_folded_knob_and_ends_clear(self):
        plan = _plans()["right"]
        first, last = plan[0], plan[-1]
        self.assertAlmostEqual(first.pos[0], 0.280 + 0.0, delta=0.01)
        self.assertAlmostEqual(first.pos[1], -0.145 - L.GRIP_OFF, delta=1e-9)
        st = bed(40.0, 0.0)
        p = np.asarray(last.pos)
        arm = rigstate.Arm(p, p - np.array([0, 0, 0.013]), 1.0)
        gap = rigstate.zone_gap(arm, lambda pts: st.distance(pts, bodies=L.RELEASE_BODIES))
        self.assertGreaterEqual(gap, L.LET_GO + 0.040)  # 40 mm of margin on the 30 mm
        self.assertEqual(last.finger, T.OPEN_KNOB)

    def test_orbit_turns_and_ramps(self):
        total = T.TURNS * T.STEPS_PER_TURN + T.RAMP_STEPS
        self.assertAlmostEqual(T.orbit_angle(total), TWO_PI * T.TURNS, places=9)
        self.assertEqual(T.orbit_angle(0), 0.0)
        steps = [T.orbit_angle(k + 1) - T.orbit_angle(k) for k in range(total)]
        self.assertTrue(all(s >= -1e-12 for s in steps))
        self.assertLessEqual(max(steps), TWO_PI / T.STEPS_PER_TURN + 1e-9)

    def test_orbit_radius_blends_in_and_out(self):
        total = T.TURNS * T.STEPS_PER_TURN + T.RAMP_STEPS
        self.assertAlmostEqual(T.orbit_radius(0), L.ARM_LEN)
        self.assertAlmostEqual(T.orbit_radius(total), L.ARM_LEN)
        self.assertAlmostEqual(T.orbit_radius(100), L.ARM_LEN + T.ORBIT_PULL)

    def test_orbit_starts_and_ends_on_the_knob(self):
        total = T.TURNS * T.STEPS_PER_TURN + T.RAMP_STEPS
        want = np.array(T.grip_world(0.0, L.FOLD_OPEN))
        self.assertTrue(np.allclose(T.orbit_point(0), want, atol=1e-9))
        self.assertTrue(np.allclose(T.orbit_point(total), want, atol=1e-9))

    def test_whole_turns_end_with_the_pin_vertical(self):
        for turns in (4, 8, 11, 13):
            self.assertAlmostEqual(math.sin(T.orbit_angle(turns * T.STEPS_PER_TURN + T.RAMP_STEPS, turns)), 0.0,
                                   places=9)

    def test_swing_ends_on_the_stops_and_follows_the_pin_arc(self):
        p0, y0 = T.swing_point(0, 40, True)
        p1, y1 = T.swing_point(40, 40, True)
        self.assertTrue(np.allclose(p0, T.grip_world(0.0, 0.0)))
        self.assertTrue(np.allclose(p1, T.grip_world(0.0, L.FOLD_OPEN)))
        self.assertAlmostEqual(y0, math.pi / 2)
        self.assertAlmostEqual(y1, 0.0)
        for k in range(41):
            p, _ = T.swing_point(k, 40, True)
            d = math.hypot(p[0] - L.PIN_X, p[1] - L.SHAFT_Y)
            self.assertAlmostEqual(d, math.hypot(L.ARM_LEN, L.KNOB_OFF + L.GRIP_OFF), places=9)
        pe, _ = T.swing_point(40, 40, False)  # folded with the over-travel against the stop
        self.assertGreater(pe[1], T.grip_world(0.0, 0.0)[1])  # a little past the stop, toward +y (into the stop)

    def test_reachable_heights(self):
        # the highest TCP of the demo (the top of the orbit and the approach)
        top = max(w.pos[2] for w in _plans()["right"]) - L.TOP
        self.assertLessEqual(top, 0.205)
        low = min(w.pos[2] for w in _plans()["right"]) - L.TOP
        self.assertGreaterEqual(low, 0.030)

    def test_pads_clear_the_gearbox(self):
        # the open fingers' outer edge at the folded knob is under the gearbox's front face by at least 20 mm
        edge = 0.285 + (-T.OPEN_KNOB) + rigstate.FINGER_THICK
        self.assertLessEqual(edge, L.BOX_X[0] - 0.020)


@unittest.skipIf(OFFLINE is None, "tools/offline not found")
class CounterfactualsThroughTheChecker(unittest.TestCase):
    """The offline checker's kinematic proxy on the demo and its variants."""

    def run_variant(self, variant="", idle=""):
        env = dict(os.environ)
        for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY", "TASK_TRACE"):
            env.pop(k, None)
        if variant:
            env["DEMO_VARIANT"] = variant
        if idle:
            env["DEMO_IDLE_ARM"] = idle
        if variant == "hold":
            env["DEMO_HOLD_ONLY"] = "1"
        out = subprocess.run([sys.executable, str(OFFLINE / "check_task.py"), str(HERE), "--json", "-", "--quiet"],
                             capture_output=True, text=True, env=env, timeout=600)
        text = out.stdout
        start = text.find("{")
        data = json.loads(text[start:text.rfind("}") + 1]) if start >= 0 else {}
        proxy = (data.get("playback") or {}).get("proxy") or {}
        return proxy.get("predicted_task_score")

    def test_demo_full_score(self):
        self.assertEqual(self.run_variant(), 1.0)

    def test_short_turn_is_a_quarter(self):
        self.assertEqual(self.run_variant("short"), 0.25)

    def test_over_turn_is_a_quarter(self):
        self.assertEqual(self.run_variant("over"), 0.25)

    def test_far_over_turn_is_zero(self):
        self.assertEqual(self.run_variant("toomuch"), 0.0)

    def test_handle_left_out_is_half(self):
        self.assertEqual(self.run_variant("unfolded"), 0.5)

    def test_hand_left_on_the_knob_is_half(self):
        self.assertEqual(self.run_variant("held"), 0.5)

    def test_nobody_moving_is_zero(self):
        self.assertEqual(self.run_variant("hold"), 0.0)


# -- the files ----------------------------------------------------------------------------------------------------------


class Files(unittest.TestCase):
    spec = json.loads((HERE / "task.json").read_text())

    def test_fields_and_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "run_limit": 60,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.spec[k]), n, k)
        for k in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                  "fidelity", "reset_notes", "scores"):
            self.assertIn(k, self.spec)
        self.assertEqual(self.spec["category"], "Impact on jobs")

    def test_no_dashes_or_banned_words(self):
        for p in list(HERE.glob("*.md")) + list(HERE.glob("*.json")) + [HERE / "task.py", HERE / "layout.py",
                                                                        HERE / "scoring.py"]:
            text = p.read_text()
            self.assertNotIn("\u2014", text, p.name)
            self.assertNotIn("\u2013", text, p.name)
        blob = json.dumps(self.spec).lower() + (HERE / "README.md").read_text().lower()
        for bad in ("t-70", "vihan", "claude", "anthropic", "robocurve-benchmarking", "/home/"):
            self.assertNotIn(bad, blob)

    def test_numbers_agree(self):
        text = self.spec["setup"] + self.spec["scoring"]
        for s in ("5.0 degrees", "30.0 to 45.0", "60.0 degrees", "15 degrees", "30 mm", "0.5 degrees", "90 x 90 x 160",
                  "12 mm", "130 mm", "450 x 550 x 20", "3.0 kg", "25 mm across x 70 mm", "(165, -170, 115)",
                  "(280, -145, 115)", "0.04 N m", "0.05 bearing", "0.15"):
            self.assertIn(s, text, s)
        self.assertEqual(self.spec["run_limit"], "1800 steps (90 s at 20 Hz)")
        self.assertIn("8-turn", self.spec["context"])
        self.assertEqual(T.TURNS * 5, 40)


if __name__ == "__main__":
    unittest.main()
