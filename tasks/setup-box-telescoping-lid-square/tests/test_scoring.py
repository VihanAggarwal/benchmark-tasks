"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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


def lid_state(x=None, y=None, rim=L.SEAT_RIM, yaw=0.0, pitch=0.0, roll=0.0):
    """The lid with its rim centre at (x, y), ``rim`` over the table, turned yaw/pitch/roll (rad)."""
    x = L.BASE_X if x is None else x
    y = L.BASE_Y if y is None else y
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.LID_PART, (x, y, L.TOP + rim, *q), source="test")


def base_state(dx=0.0, dy=0.0, dz=0.0, yaw=0.0, pitch=0.0, roll=0.0):
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.BASE_PART, (L.BASE_X + dx, L.BASE_Y + dy, L.TOP + dz, *q), source="test")


def product_state(x=None, y=None, z=None):
    x = L.BASE_X if x is None else x
    y = L.BASE_Y if y is None else y
    z = L.PR_Z if z is None else z
    return mk.state_from_poses(T.PRODUCT_PART, (x, y, L.TOP + z, 0.0, 0.0, 0.0, 1.0), source="test")


def scene(lid=None, base=None, product=None):
    return {"base": base or base_state(), "lid": lid or lid_state(), "product": product or product_state()}


def start():
    return {"base": T.BASE_PART.rest_state(), "lid": T.LID_PART.rest_state(), "product": T.PRODUCT_PART.rest_state()}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_lid_fits_the_base_with_the_stated_play(self):
        self.assertAlmostEqual(L.LID_IX - L.BASE_LX, 2 * L.CLR, places=9)
        self.assertAlmostEqual(L.LID_IY - L.BASE_LY, 2 * L.CLR, places=9)
        self.assertAlmostEqual(L.LID_OX, 0.079)
        self.assertAlmostEqual(L.LID_OY, 0.123)
        self.assertAlmostEqual(L.LID_H, 0.036)

    def test_seated_heights(self):
        self.assertAlmostEqual(L.SEAT_TOP, 0.042)
        self.assertAlmostEqual(L.SEAT_RIM, 0.006)
        self.assertAlmostEqual(L.BASE_H - L.SEAT_RIM, L.LID_SKIRT)  # the skirt covers the base's walls by 34 mm

    def test_product_sits_in_the_base_below_the_rim(self):
        self.assertLess(L.PR_LX, L.INNER_LX)
        self.assertLess(L.PR_LY, L.INNER_LY)
        self.assertAlmostEqual(L.FLOOR + L.PR_H, 0.030)
        self.assertGreaterEqual(L.BASE_H - (L.FLOOR + L.PR_H), 0.010)  # 10 mm below the rim: the lid cannot crush it

    def test_parts_spawn_resting(self):
        b, p, d = T.BASE_PART.rest_state(), T.PRODUCT_PART.rest_state(), T.LID_PART.rest_state()
        self.assertAlmostEqual(b.low[2] - L.TOP, 0.0, places=6)
        self.assertAlmostEqual(d.low[2] - L.TOP, 0.0, places=6)
        self.assertAlmostEqual(p.low[2] - L.TOP, L.FLOOR, places=6)  # on the base's floor
        self.assertAlmostEqual(d.high[2] - L.TOP, L.LID_H, places=6)

    def test_yaw_limit_is_a_few_degrees(self):
        """With 2.5 mm play a yawed lid wedges beyond about 2 degrees: the 1.5 degree gate sits inside it."""
        lim = L.lid_yaw_ok_limit_deg()
        self.assertGreater(lim, L.SQUARE_YAW_DEG)
        self.assertLess(lim, 4.0)

    def test_centre_gate_is_wider_than_the_play(self):
        self.assertGreater(L.CENTRE_TOL, L.CLR)

    def test_open_gripper_clears_the_lid(self):
        self.assertGreaterEqual((0.0939 - L.LID_OX) / 2, 0.007)  # 7 mm a side round the lid's 79 mm
        self.assertGreater(0.0939 - 0.008, T.L.LID_OX - 0.0001 - 0.008)

    def test_grip_commands_a_gap_8mm_under_the_lid(self):
        self.assertAlmostEqual(-2 * T.GRIP, L.LID_OX - 0.008, places=6)

    def test_pad_tips_stay_on_the_skirt_and_off_the_table(self):
        self.assertGreaterEqual(T.LID_GRIP_Z - 0.013, 0.004)  # tips over the table
        self.assertLessEqual(T.LID_GRIP_Z + 0.009, L.LID_SKIRT)  # the pad ends under the panel's inner face

    def test_seated_pads_clear_the_base(self):
        """The lid's walls are 4.5 mm outside the base's: the pads, outside the lid, never touch the base."""
        self.assertGreater(L.LID_OX / 2, L.BASE_LX / 2 + L.CLR)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["seated"] and res.details["square"] and res.details["centred"])

    def test_tier_table(self):
        self.assertEqual(S.tier(False, True, True, True, True, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, True, True, True, False, True, True), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, True, False, True), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, True, True, False), 0.5)
        self.assertEqual(S.tier(False, True, True, True, False, False, False, False), 0.25)
        self.assertEqual(S.tier(False, True, True, False, False, False, False, False), 0.0)
        self.assertEqual(S.tier(True, True, True, True, True, True, True, True), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True, True, True, True), 0.0)
        self.assertEqual(S.tier(False, True, False, True, True, True, True, True), 0.0)

    def test_start_scores_zero(self):
        res = run(start())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_lid_on_the_table_beside_the_base_is_zero(self):
        self.assertEqual(run(scene(lid=lid_state(rim=0.0, y=L.BASE_Y - 0.15))).score, 0.0)

    def test_lid_resting_flat_on_the_rim_is_a_quarter(self):
        res = run(scene(lid=lid_state(rim=L.BASE_H)))
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["engaged"])

    def test_lid_perched_on_the_rim_out_of_line_is_a_quarter(self):
        self.assertEqual(run(scene(lid=lid_state(rim=L.BASE_H, x=L.BASE_X + 0.006))).score, 0.25)

    def test_wedged_half_on_is_half(self):
        """Down 20 mm over the walls but tilted 3 degrees: engaged, not seated, not square."""
        res = run(scene(lid=lid_state(rim=0.020, pitch=math.radians(3.0))))
        self.assertEqual(res.score, 0.5)
        self.assertTrue(res.details["engaged"])
        self.assertFalse(res.details["square"])

    def test_engaged_but_not_down_is_half(self):
        res = run(scene(lid=lid_state(rim=0.026)))  # 14 mm over the walls, top face 62 mm
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.details["seated"])

    def test_overlap_just_under_ten_is_a_quarter(self):
        self.assertEqual(run(scene(lid=lid_state(rim=L.BASE_H - 0.0099))).score, 0.25)

    def test_overlap_ten_counts(self):
        self.assertEqual(run(scene(lid=lid_state(rim=L.BASE_H - 0.010))).score, 0.5)

    def test_seat_height_window(self):
        self.assertEqual(run(scene(lid=lid_state(rim=L.SEAT_RIM + 0.0014))).score, 1.0)
        self.assertEqual(run(scene(lid=lid_state(rim=L.SEAT_RIM + 0.0016))).score, 0.5)
        self.assertEqual(run(scene(lid=lid_state(rim=L.SEAT_RIM - 0.0016))).score, 0.5)

    def test_yaw_window(self):
        self.assertEqual(run(scene(lid=lid_state(yaw=math.radians(1.4)))).score, 1.0)
        self.assertEqual(run(scene(lid=lid_state(yaw=math.radians(1.7)))).score, 0.5)
        self.assertEqual(run(scene(lid=lid_state(yaw=math.radians(-1.7)))).score, 0.5)

    def test_yaw_of_the_base_is_the_reference(self):
        """The base turned 20 degrees with the lid turned the same: square."""
        y = math.radians(20.0)
        res = run(scene(base=base_state(yaw=y), lid=lid_state(yaw=y)))
        self.assertEqual(res.score, 1.0)

    def test_yaw_is_modulo_half_a_turn(self):
        self.assertEqual(run(scene(lid=lid_state(yaw=math.pi))).score, 1.0)  # a rectangle turned a half turn

    def test_tilt_window(self):
        self.assertEqual(run(scene(lid=lid_state(pitch=math.radians(0.9)))).score, 1.0)
        self.assertEqual(run(scene(lid=lid_state(pitch=math.radians(1.2)))).score, 0.5)
        self.assertEqual(run(scene(lid=lid_state(roll=math.radians(1.2)))).score, 0.5)

    def test_centre_window(self):
        self.assertEqual(run(scene(lid=lid_state(x=L.BASE_X + 0.003))).score, 1.0)
        self.assertEqual(run(scene(lid=lid_state(y=L.BASE_Y - 0.003))).score, 1.0)
        self.assertEqual(run(scene(lid=lid_state(x=L.BASE_X + 0.0037))).score, 0.5)

    def test_centre_is_taken_against_the_base_not_the_start(self):
        """The base slid 15 mm (allowed) with the lid on it: still 1.0."""
        res = run(scene(base=base_state(dx=0.015), lid=lid_state(x=L.BASE_X + 0.015),
                        product=product_state(x=L.BASE_X + 0.015)))
        self.assertEqual(res.score, 1.0)

    def test_success_needs_the_full_score(self):
        self.assertFalse(run(scene(lid=lid_state(yaw=math.radians(2.0)))).success)


class ZeroConditions(unittest.TestCase):
    def test_base_moved_over_20mm(self):
        res = run(scene(base=base_state(dx=0.021), lid=lid_state(x=L.BASE_X + 0.021),
                        product=product_state(x=L.BASE_X + 0.021)))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["box_still"])

    def test_base_moved_just_under_20mm_is_fine(self):
        res = run(scene(base=base_state(dy=-0.019), lid=lid_state(y=L.BASE_Y - 0.019),
                        product=product_state(y=L.BASE_Y - 0.019)))
        self.assertEqual(res.score, 1.0)

    def test_base_lifted_off_the_table(self):
        res = run(scene(base=base_state(dz=0.005), lid=lid_state(rim=L.SEAT_RIM + 0.005),
                        product=product_state(z=L.PR_Z + 0.005)))
        self.assertEqual(res.score, 0.0)

    def test_base_tilted(self):
        self.assertEqual(run(scene(base=base_state(roll=math.radians(3.0)))).score, 0.0)

    def test_lid_fell_off_the_table_latches(self):
        tr = S.Tracker()
        tr.update(scene(lid=lid_state(rim=-0.05, x=0.6)), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_base_fell_off_the_table_latches(self):
        tr = S.Tracker()
        tr.update(scene(base=base_state(dz=-0.05)), FAR)
        self.assertTrue(tr.latched)

    def test_product_out_of_the_base_latches(self):
        tr = S.Tracker()
        tr.update(scene(product=product_state(x=L.BASE_X + 0.10)), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_product_beside_the_base_on_the_table_is_out(self):
        self.assertEqual(run(scene(product=product_state(y=L.BASE_Y + 0.12, z=L.PR_H / 2))).score, 0.0)

    def test_product_slack_inside_the_base_is_fine(self):
        self.assertEqual(run(scene(product=product_state(x=L.BASE_X + 0.005, y=L.BASE_Y + 0.002))).score, 1.0)

    def test_hand_on_the_lid_is_not_released(self):
        arms = {"left": FAR["left"], "right": arm((L.BASE_X, L.BASE_Y, L.TOP + 0.025), 0.76)}
        res = run(scene(), arms=arms)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_hand_hovering_20mm_over_the_lid_is_not_released(self):
        arms = {"left": FAR["left"], "right": arm((L.BASE_X, L.BASE_Y, L.TOP + L.SEAT_TOP + 0.013 + 0.055 + 0.02), 1.0)}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_parked_hands_are_clear(self):
        self.assertEqual(run(scene()).details["released_1s"], True)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        near = {"left": FAR["left"], "right": arm((L.BASE_X, L.BASE_Y, L.TOP + 0.025), 1.0)}
        run(scene(), arms=near, steps=1, tracker=tr)
        res = run(scene(), steps=WIN - 1, tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(run(scene(), steps=2, tracker=tr).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)
        self.assertEqual(run(scene(), arms={"left": FAR["left"]}).score, 0.0)

    def test_moving_lid(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(scene(lid=lid_state(rim=L.SEAT_RIM + 0.0001 * (i % 40))), FAR)  # up to 3.9 mm of creep
        self.assertEqual(res.score, 0.0)

    def test_turning_lid(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(scene(lid=lid_state(yaw=math.radians(0.02 * i))), FAR)  # drifts 0.46 degrees in the window
        self.assertEqual(res.score, 0.0)

    def test_still_for_the_whole_second_needed(self):
        tr = S.Tracker()
        tr.update(scene(lid=lid_state(rim=L.SEAT_RIM + 0.003)), FAR)  # moved in the first step of the window
        res = run(scene(), steps=WIN - 1, tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(run(scene(), steps=3, tracker=tr).score, 1.0)

    def test_lid_held_in_the_air_over_the_base_scores_zero(self):
        """Held 30 mm up on the base: the hand is on it, and it is moving."""
        arms = {"left": FAR["left"], "right": arm((L.BASE_X, L.BASE_Y, L.TOP + 0.050), 0.76)}
        self.assertEqual(run(scene(lid=lid_state(rim=0.030)), arms=arms).score, 0.0)


class Exploits(unittest.TestCase):
    def test_lid_upside_down_on_the_base_scores_zero(self):
        """Turned over (top panel down, skirt up): resting on its rim on the base's rim is not closing the box."""
        res = run(scene(lid=lid_state(rim=L.BASE_H + L.LID_H, roll=math.pi)))
        self.assertEqual(res.score, 0.0)
        res = run(scene(lid=lid_state(rim=L.BASE_H + L.LID_PANEL + L.LID_SKIRT, roll=math.pi)))
        self.assertEqual(res.score, 0.0)

    def test_lid_laid_over_the_base_on_its_side_scores_zero(self):
        self.assertEqual(run(scene(lid=lid_state(rim=0.045, roll=math.radians(80.0)))).score, 0.0)

    def test_base_carried_into_the_lid_scores_zero(self):
        """The base lifted and pushed into the standing lid: the box has moved far from where it was packed."""
        res = run(scene(base=base_state(dx=L.LID_X - L.BASE_X, dy=L.LID_Y - L.BASE_Y, dz=0.002),
                        lid=lid_state(x=L.LID_X, y=L.LID_Y, rim=0.0),
                        product=product_state(x=L.LID_X, y=L.LID_Y)))
        self.assertEqual(res.score, 0.0)

    def test_base_turned_over_onto_the_standing_lid_scores_zero(self):
        res = run(scene(base=base_state(dx=L.LID_X - L.BASE_X, dy=L.LID_Y - L.BASE_Y, dz=L.LID_H, roll=math.pi),
                        lid=lid_state(x=L.LID_X, y=L.LID_Y, rim=0.0),
                        product=product_state(x=L.LID_X, y=L.LID_Y, z=0.01)))
        self.assertEqual(res.score, 0.0)

    def test_product_taken_out_and_lid_closed_empty_scores_zero(self):
        res = run(scene(product=product_state(x=L.BASE_X + 0.25, y=L.BASE_Y + 0.2, z=L.PR_H / 2)))
        self.assertEqual(res.score, 0.0)

    def test_lid_pushed_off_the_table_then_anything_scores_zero(self):
        tr = S.Tracker()
        tr.update(scene(lid=lid_state(x=0.95, y=0.6, rim=-0.3)), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_base_dragged_away_and_back_is_not_latched_but_final_pose_counts(self):
        tr = S.Tracker()
        tr.update(scene(base=base_state(dx=0.05), lid=lid_state(x=L.BASE_X + 0.05),
                        product=product_state(x=L.BASE_X + 0.05)), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)  # back where it began: the score reads the end state

    def test_base_shoved_and_lid_dropped_on_it_scores_zero(self):
        res = run(scene(base=base_state(dx=0.06), lid=lid_state(x=L.BASE_X + 0.06),
                        product=product_state(x=L.BASE_X + 0.06)))
        self.assertEqual(res.score, 0.0)

    def test_doing_nothing_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_lid_flat_on_top_of_the_base_never_seats(self):
        """Laid flat on the rim 2 mm out of line is 0.25 whatever the arm does next."""
        self.assertEqual(run(scene(lid=lid_state(rim=L.BASE_H, x=L.BASE_X + 0.004))).score, 0.25)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        s = scene()
        del s["lid"]
        res = run(s)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])

    def test_not_a_dict_scores_zero(self):
        self.assertEqual(run(None).score, 0.0)

    def test_nan_state(self):
        bad = mk.state_from_poses(T.LID_PART, (L.BASE_X, L.BASE_Y, float("nan"), 0.0, 0.0, 0.0, 1.0), source="test")
        res = S.Tracker().update(scene(lid=bad), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])

    def test_nan_quaternion(self):
        bad = mk.state_from_poses(T.PRODUCT_PART, (L.BASE_X, L.BASE_Y, L.TOP + L.PR_Z, 0.0, 0.0, 0.0, 1.0),
                                  source="test")
        object.__setattr__(bad, "quat_xyzw", np.array([float("nan"), 0.0, 0.0, 1.0]))
        self.assertEqual(S.Tracker().update(scene(product=bad), FAR).score, 0.0)

    def test_nan_resets_the_windows(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.LID_PART, (L.BASE_X, L.BASE_Y, float("nan"), 0.0, 0.0, 0.0, 1.0), source="test")
        tr.update(scene(lid=bad), FAR)
        res = run(scene(), steps=WIN - 1, tracker=tr)
        self.assertEqual(res.score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(), steps=3, tracker=tr).score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(scene(lid=lid_state(rim=0.0, x=0.6, y=0.3)), FAR)
        tr.latch["F1"] = 3
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(300):
            lid = lid_state(x=L.BASE_X + rng.normal(0, 0.02), y=L.BASE_Y + rng.normal(0, 0.02),
                            rim=float(rng.uniform(-0.02, 0.09)), yaw=float(rng.normal(0, 0.1)),
                            pitch=float(rng.normal(0, 0.1)), roll=float(rng.normal(0, 0.2)))
            res = tr.update(scene(lid=lid), FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))


class DemoPlans(unittest.TestCase):
    def test_demo_is_the_right_arm_alone(self):
        self.assertEqual(set(T.demo()), {"right"})

    def test_hold_only_has_no_plan(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_idle_arm_drops_that_plan(self):
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]

    def test_demo_length_within_run_limit(self):
        steps = max(sum(w.duration for w in plan) for plan in T.plans().values()) * HZ
        self.assertLessEqual(steps, 0.6 * 1200)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_variants_put_the_lid_out_of_line(self):
        for name, (dx, dy) in T.VARIANT_OFFSET.items():
            seat = [w for w in T.plans(name)["right"] if w.label.startswith("lower it over the walls")][0]
            self.assertAlmostEqual(seat.pos[0] - L.BASE_X, dx + T.X_TRIM, places=9)
            self.assertAlmostEqual(seat.pos[1] - L.BASE_Y, dy + T.Y_TRIM, places=9)
            self.assertGreater(max(abs(dx), abs(dy)), L.CLR)

    def test_demo_seats_the_lid_at_the_stated_heights(self):
        plan = T.plans()["right"]
        seat = [w for w in plan if w.label.startswith("lower it over the walls")][0]
        self.assertAlmostEqual(seat.pos[2] - L.TOP, L.SEAT_RIM + T.LID_GRIP_Z - T.PRESS, places=9)
        carry = [w for w in plan if w.label.startswith("over the base")][0]
        self.assertGreaterEqual(carry.pos[2] - L.TOP - T.LID_GRIP_Z, L.BASE_H + 0.008)  # the rim clears the base's rim

    def test_demo_grips_the_lid_on_its_short_axis_with_margin(self):
        plan = T.plans()["right"]
        self.assertGreaterEqual((-2 * T.OPEN - L.LID_OX) / 2, 0.007)
        grip = [w for w in plan if w.label.startswith("grip the lid")][0]
        self.assertAlmostEqual(-2 * grip.finger, L.LID_OX - 0.008, places=6)

    def test_task_text_files_exist(self):
        self.assertTrue((HERE / "task.json").exists())
        self.assertTrue((HERE / "README.md").exists())


class OfflineStandIn(unittest.TestCase):
    def grip_at(self, model, rim=0.0):
        tcp = (L.LID_X, L.LID_Y, L.TOP + rim + T.LID_GRIP_Z)
        model.step({"left": FAR["left"], "right": arm(tcp, 0.76)})

    def test_model_carries_the_lid(self):
        m = T.OfflineModel()
        self.grip_at(m)
        self.assertIn("right", m.grip)
        m.step({"left": FAR["left"], "right": arm((L.LID_X, L.LID_Y + 0.1, L.TOP + 0.07), 0.76)})
        self.assertAlmostEqual(m.lid[1], L.LID_Y + 0.1, places=9)
        self.assertAlmostEqual(m.lid[2], 0.07 - T.LID_GRIP_Z, places=9)

    def test_model_does_not_grip_with_an_open_hand(self):
        m = T.OfflineModel()
        m.step({"left": FAR["left"], "right": arm((L.LID_X, L.LID_Y, L.TOP + T.LID_GRIP_Z), 1.0)})
        self.assertEqual(m.grip, {})

    def test_model_seats_a_lid_let_go_in_line(self):
        m = T.OfflineModel()
        self.grip_at(m)
        m.step({"left": FAR["left"], "right": arm((L.BASE_X + 0.001, L.BASE_Y, L.TOP + 0.05), 0.76)})
        m.step({"left": FAR["left"], "right": arm((L.BASE_X + 0.001, L.BASE_Y, L.TOP + 0.05), 1.0)})
        self.assertAlmostEqual(m.lid[2], L.SEAT_RIM, places=9)

    def test_model_rests_a_lid_let_go_out_of_line_on_the_rim(self):
        m = T.OfflineModel()
        self.grip_at(m)
        m.step({"left": FAR["left"], "right": arm((L.BASE_X + 0.005, L.BASE_Y, L.TOP + 0.05), 0.76)})
        m.step({"left": FAR["left"], "right": arm((L.BASE_X + 0.005, L.BASE_Y, L.TOP + 0.05), 1.0)})
        self.assertAlmostEqual(m.lid[2], L.BASE_H, places=9)

    def test_model_will_not_lower_a_carried_lid_through_the_rim(self):
        m = T.OfflineModel()
        self.grip_at(m)
        m.step({"left": FAR["left"], "right": arm((L.BASE_X + 0.005, L.BASE_Y, L.TOP + 0.01), 0.76)})
        self.assertAlmostEqual(m.lid[2], L.BASE_H, places=9)

    def test_model_states_are_complete(self):
        st = T.OfflineModel().states()
        self.assertEqual(set(st), set(S.PARTS))
        self.assertTrue(S.Tracker().update(st, FAR).details["valid_state"])


class TaskJson(unittest.TestCase):
    def test_fields_and_limits(self):
        d = json.loads((HERE / "task.json").read_text())
        lim = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
               "reset_notes": 580}
        for k, n in lim.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertEqual(d["category"], "Impact on jobs")
        self.assertNotIn(chr(0x2014), json.dumps(d))

    def test_instruction_matches_the_task_file(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)

    def test_thresholds_in_text_match_the_code(self):
        d = json.loads((HERE / "task.json").read_text())
        sc = d["scoring"]
        for token in ("1.5 mm", "42 mm", "1.5 degrees", "1.0 degree", "3.5 mm", "10 degrees", "8 mm", "10 mm",
                      "30 degrees", "25 mm", "4 mm", "20 mm", "3 mm", "2 degrees", "30 mm", "1 mm", "0.3 degrees"):
            self.assertIn(token, sc, token)
        self.assertEqual(L.SEAT_TOL, 0.0015)
        self.assertEqual(L.SQUARE_YAW_DEG, 1.5)
        self.assertEqual(L.SQUARE_TILT_DEG, 1.0)
        self.assertEqual(L.CENTRE_TOL, 0.0035)
        self.assertEqual(L.ENGAGED_OVERLAP, 0.010)
        self.assertEqual(L.ENGAGED_CENTRE, 0.008)
        self.assertEqual(L.ENGAGED_TILT_DEG, 10.0)
        self.assertEqual(L.ON_BASE_CENTRE, 0.025)
        self.assertEqual(L.ON_BASE_TILT_DEG, 30.0)
        self.assertEqual(L.ON_BASE_LOW, 0.004)
        self.assertEqual(L.BASE_MOVE, 0.020)
        self.assertEqual(L.BASE_LIFT, 0.003)
        self.assertEqual(L.BASE_TILT_DEG, 2.0)
        self.assertEqual(L.FALL_Z, -0.010)

    def test_setup_numbers_match_the_code(self):
        su = json.loads((HERE / "task.json").read_text())["setup"]
        for token in ("70 (x) x 114 (y) x 40 (z)", "55 x 105 x 28", "79 x 123 mm outside", "75 x 119 mm inside",
                      "2.5 mm larger", "42 mm", "34 mm", "x 330 mm, y -70 mm", "x 330 mm, y -290 mm", "0.12 kg"):
            self.assertIn(token, su, token)
        self.assertEqual((L.BASE_X, L.BASE_Y, L.LID_X, L.LID_Y), (0.330, -0.070, 0.330, -0.290))


if __name__ == "__main__":
    unittest.main()
