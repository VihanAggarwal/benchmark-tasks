"""Offline tests for layout.py, scoring.py and task.py: synthetic switch states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
SEATED = (L.PX, L.PY, L.SEAT_Z)


def sw(x=L.PX, y=L.PY, z=L.SEAT_Z, yaw=0.0, pitch_deg=0.0, roll_deg=0.0):
    """The switch's frame at (x, y) and height z above the table top (m)."""
    q = mk.quat_from_ypr(yaw, math.radians(pitch_deg), math.radians(roll_deg))
    return mk.state_from_poses(T.SWITCH_PART, (x, y, L.TOP + z, *q), source="test")


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.15, 0.40, L.TOP + 0.20)), "right": arm((0.15, -0.40, L.TOP + 0.20))}


def run(states, arms=FAR, n=2 * WIN, tracker=None):
    """Feed ``n`` identical steps; the tracker and the last result."""
    tr = tracker or S.Tracker()
    res = None
    for _ in range(n):
        res = tr.update({S.SWITCH: states}, arms)
    return tr, res


class LayoutTests(unittest.TestCase):
    def test_fit_and_play(self):
        self.assertAlmostEqual(L.CLEAR_X, 0.00075)
        self.assertAlmostEqual(L.CLEAR_Y, 0.0010)
        self.assertLess(L.BODY_X, L.HOLE_X)
        self.assertLess(L.BODY_Y, L.HOLE_Y)
        self.assertGreater(L.FL_X, L.HOLE_X)
        self.assertGreater(L.FL_Y, L.HOLE_Y)

    def test_standing_on_the_foam(self):
        self.assertAlmostEqual(L.SW_Z0 - L.DEPTH, L.FOAM_H)

    def test_body_clears_the_table_under_the_plate(self):
        self.assertGreater(L.PLATE_TOP - L.DEPTH, 0.010)  # the body's rear face stays well above the table

    def test_reach_of_the_arms_at_the_two_stations(self):
        # the waypoints are checked by tools/offline/check_task.py; here only that both are on the left arm's side
        self.assertGreater(L.PY, 0.0)
        self.assertGreater(L.FOAM_Y, 0.0)

    def test_hole_inside_plate_and_rails_inside_plate(self):
        self.assertLess(L.HOLE_X, L.PL_X)
        self.assertLess(L.HOLE_Y, L.PL_Y)
        self.assertLess(L.BAND_X1, L.PL_X / 2)

    def test_start_pose_is_not_a_success(self):
        tr, res = run(T.SWITCH_PART.rest_state(), n=1)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)


class TierTests(unittest.TestCase):
    def test_seated_top_is_one(self):
        tr, res = run(sw())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["orientation"], "top")

    def test_seated_turned_is_half(self):
        tr, res = run(sw(yaw=math.pi))
        self.assertEqual(res.score, 0.5)
        self.assertEqual(res.details["orientation"], "bottom")

    def test_orientation_window_edges(self):
        self.assertEqual(run(sw(yaw=math.radians(19.0)))[1].score, 1.0)
        self.assertEqual(run(sw(yaw=math.radians(-19.0)))[1].score, 1.0)
        self.assertEqual(run(sw(yaw=math.radians(180 - 19.0)))[1].score, 0.5)
        # neither way (a body turned 45 degrees cannot seat; if a trial reported it, a quarter at most)
        self.assertEqual(run(sw(yaw=math.radians(45.0)))[1].score, 0.25)
        self.assertEqual(run(sw(yaw=math.radians(90.0)))[1].score, 0.25)
        self.assertEqual(run(sw(yaw=math.radians(21.0)))[1].score, 0.25)

    def test_proud_by_1mm_is_quarter(self):
        self.assertEqual(run(sw(z=L.SEAT_Z + 0.001))[1].score, 0.25)

    def test_proud_edges(self):
        self.assertEqual(run(sw(z=L.SEAT_Z + 0.0005))[1].score, 1.0)  # the 0.5 mm limit is in
        self.assertEqual(run(sw(z=L.SEAT_Z + 0.0006))[1].score, 0.25)
        self.assertEqual(run(sw(z=L.SEAT_Z + 0.003))[1].score, 0.25)
        self.assertEqual(run(sw(z=L.SEAT_Z + 0.0035))[1].score, 0.0)

    def test_crooked_tiers(self):
        # tilt about y by t degrees: the flange's high corner rises FL_X/2 sin t over the plate; keep it within 0.5 mm
        # by sinking the origin the same amount, so only the angle decides
        for deg, want in ((2.5, 1.0), (3.5, 0.25), (7.5, 0.25), (9.0, 0.0)):
            rise = L.FL_X / 2 * math.sin(math.radians(deg))
            res = run(sw(z=L.SEAT_Z - rise, pitch_deg=deg))[1]
            self.assertEqual(res.score, want, f"{deg} degrees")

    def test_crooked_about_the_long_side(self):
        rise = L.FL_Y / 2 * math.sin(math.radians(5.0))
        self.assertEqual(run(sw(z=L.SEAT_Z - rise, roll_deg=5.0))[1].score, 0.25)

    def test_not_over_the_cutout_is_zero(self):
        self.assertEqual(run(sw(x=L.PX + 0.004))[1].score, 0.0)
        self.assertEqual(run(sw(y=L.PY + 0.0035))[1].score, 0.0)
        self.assertEqual(run(sw(x=L.PX + 0.002, y=L.PY - 0.002))[1].score, 1.0)

    def test_resting_on_the_plate_on_its_rear_face_is_zero(self):
        self.assertEqual(run(sw(z=L.PLATE_TOP + L.DEPTH))[1].score, 0.0)

    def test_on_the_start_mark_is_zero(self):
        self.assertEqual(run(T.SWITCH_PART.rest_state())[1].score, 0.0)

    def test_possible_scores(self):
        seen = set()
        for st in (sw(), sw(yaw=math.pi), sw(z=L.SEAT_Z + 0.002), sw(x=0.2)):
            seen.add(run(st)[1].score)
        self.assertEqual(seen, {1.0, 0.5, 0.25, 0.0})


class ZeroConditionTests(unittest.TestCase):
    def test_fell_off_the_table_latches_zero(self):
        tr2 = S.Tracker()
        run(sw(x=0.18, y=0.2, z=L.DEPTH - 0.02), tracker=tr2, n=1)  # rear face 20 mm below the table top
        self.assertIsNotNone(tr2.latch["F1"])
        res = run(sw(), tracker=tr2, n=3 * WIN)[1]  # later seated: still zero
        self.assertEqual(res.score, 0.0)

    def test_set_down_on_the_table_then_installed_scores(self):
        tr = S.Tracker()
        run(sw(x=0.18, y=0.2, z=L.DEPTH), tracker=tr, n=3)  # rear face on the table: allowed
        self.assertIsNone(tr.latch["F1"])
        res = run(sw(), tracker=tr, n=3 * WIN)[1]  # later seated, hands off, still
        self.assertEqual(res.score, 1.0)

    def test_on_the_foam_is_not_a_fall(self):
        tr, res = run(T.SWITCH_PART.rest_state(), n=3)
        self.assertIsNone(tr.latch["F1"])

    def test_held_by_a_hand_scores_zero(self):
        near = {"left": arm((L.PX, L.PY, L.TOP + 0.0565)), "right": FAR["right"]}
        tr, res = run(sw(), arms=near)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_finger_zone_29mm_vs_31mm(self):
        # a closed gripper straight over the rocker: its finger zone (radius zone_radius(0) round the 38 mm segment above
        # the pad tips) is just under / just over 30 mm from the rocker's top face
        rocker_top = L.TOP + L.SEAT_Z + L.TOP_OF_ROCKER
        for gap, want in ((0.029, 0.0), (0.031, 1.0)):
            tips_z = rocker_top + gap + rigstate.zone_radius(0.0)
            a = {"left": arm((L.PX, L.PY, tips_z + 0.013), opening=0.0), "right": FAR["right"]}
            self.assertEqual(run(sw(), arms=a)[1].score, want, f"{gap}")

    def test_unreadable_arms_score_zero(self):
        self.assertEqual(run(sw(), arms=None)[1].score, 0.0)
        bad = {"left": arm((float("nan"), 0.0, 0.8)), "right": FAR["right"]}
        self.assertEqual(run(sw(), arms=bad)[1].score, 0.0)
        self.assertEqual(run(sw(), arms={"left": FAR["left"]})[1].score, 0.0)

    def test_not_yet_one_second_of_release(self):
        tr = S.Tracker()
        res = None
        for _ in range(WIN - 1):
            res = tr.update({S.SWITCH: sw()}, FAR)
        self.assertEqual(res.score, 0.0)
        res = tr.update({S.SWITCH: sw()}, FAR)
        self.assertEqual(res.score, 0.0)  # the stillness window needs WIN + 1 readings
        res = tr.update({S.SWITCH: sw()}, FAR)
        self.assertEqual(res.score, 1.0)

    def test_one_step_of_a_hand_resets_the_release_window(self):
        tr, res = run(sw(), n=WIN + 5)
        self.assertEqual(res.score, 1.0)
        near = {"left": arm((L.PX, L.PY, L.TOP + 0.06)), "right": FAR["right"]}
        res = tr.update({S.SWITCH: sw()}, near)
        self.assertEqual(res.score, 0.0)
        for _ in range(WIN - 1):
            res = tr.update({S.SWITCH: sw()}, FAR)
            self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.update({S.SWITCH: sw()}, FAR).score, 1.0)


class StillnessTests(unittest.TestCase):
    def test_creeping_switch_scores_zero(self):
        tr = S.Tracker()
        res = None
        for k in range(2 * WIN):
            res = tr.update({S.SWITCH: sw(x=L.PX + 0.00003 * k)}, FAR)  # 0.6 mm/s... 1.2 mm over the window
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_sub_threshold_drift_is_still(self):
        tr = S.Tracker()
        res = None
        for k in range(2 * WIN):
            res = tr.update({S.SWITCH: sw(x=L.PX + 0.000005 * k)}, FAR)  # 0.1 mm over the window
        self.assertEqual(res.score, 1.0)

    def test_turning_switch_scores_zero(self):
        tr = S.Tracker()
        res = None
        for k in range(2 * WIN):
            res = tr.update({S.SWITCH: sw(yaw=math.radians(0.03 * k))}, FAR)  # 0.6 degrees over the window
        self.assertEqual(res.score, 0.0)

    def test_a_late_movement_resets_the_window(self):
        tr, res = run(sw(), n=2 * WIN)
        self.assertEqual(res.score, 1.0)
        res = tr.update({S.SWITCH: sw(x=L.PX + 0.002)}, FAR)
        self.assertEqual(res.score, 0.0)


class BadStateTests(unittest.TestCase):
    def test_nan_origin(self):
        tr, res = run(sw(), n=2 * WIN)
        self.assertEqual(res.score, 1.0)
        bad = sw()
        bad = type(bad)(**{**bad.__dict__, "origin": np.array([float("nan"), 0.0, 0.0])})
        res = tr.update({S.SWITCH: bad}, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertEqual(tr.update({S.SWITCH: sw()}, FAR).score, 0.0)  # windows start again

    def test_nan_quaternion(self):
        tr = S.Tracker()
        bad = sw()
        bad = type(bad)(**{**bad.__dict__, "quat_xyzw": np.array([float("nan"), 0.0, 0.0, 1.0])})
        self.assertEqual(tr.update({S.SWITCH: bad}, FAR).score, 0.0)

    def test_missing_and_garbage_states(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, FAR).score, 0.0)
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        self.assertEqual(tr.update({S.SWITCH: object()}, FAR).score, 0.0)
        self.assertEqual(tr.bad_steps, 3)

    def test_zero_quaternion_does_not_crash(self):
        tr = S.Tracker()
        bad = sw()
        bad = type(bad)(**{**bad.__dict__, "quat_xyzw": np.zeros(4)})
        tr.update({S.SWITCH: bad}, FAR)


class ExploitTests(unittest.TestCase):
    def test_pressing_it_in_by_hand_and_staying_scores_zero(self):
        # the pads left on the flange (finger zone touching it): in, square, top, still, but not released
        a = {"left": arm((L.PX, L.PY, L.TOP + L.PLATE_TOP + 0.0145), opening=0.1), "right": FAR["right"]}
        self.assertEqual(run(sw(), arms=a)[1].score, 0.0)

    def test_other_arm_holding_it_also_blocks(self):
        a = {"left": FAR["left"], "right": arm((L.PX, L.PY, L.TOP + 0.0565), opening=0.1)}
        self.assertEqual(run(sw(), arms=a)[1].score, 0.0)

    def test_dropping_it_on_the_plate_beside_the_cutout_scores_zero(self):
        self.assertEqual(run(sw(x=L.PX + 0.03, z=L.PLATE_TOP + L.DEPTH))[1].score, 0.0)

    def test_throwing_it_off_the_table_scores_zero(self):
        tr = S.Tracker()
        run(sw(x=0.6, y=0.3, z=L.DEPTH - 0.05), tracker=tr, n=1)  # below the table top
        self.assertEqual(run(sw(), tracker=tr, n=3 * WIN)[1].score, 0.0)

    def test_reset_fallback_clears_the_latch(self):
        tr = S.Tracker()
        run(sw(x=0.18, y=0.2, z=L.DEPTH - 0.02), tracker=tr, n=2)  # fell: latched
        self.assertIsNotNone(tr.latch["F1"])
        tr.update({S.SWITCH: sw(x=0.5, y=0.1, z=0.3)}, FAR)  # in the air, then the parts jump back to the start mark
        tr.update({S.SWITCH: T.SWITCH_PART.rest_state()}, FAR)
        self.assertIsNone(tr.latch["F1"])
        self.assertEqual(tr.resets, 1)

    def test_a_switch_inverted_cannot_score(self):
        # rocker down (turned 180 degrees about x) in the cutout: tilt 180
        self.assertEqual(run(sw(roll_deg=180.0))[1].score, 0.0)


class SceneTests(unittest.TestCase):
    def test_objects_and_names(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["panel", "foam", "switch"])
        self.assertTrue(T.FOAM.mass is None)
        self.assertTrue(T.PANEL_PART.fixed_base)
        self.assertFalse(T.SWITCH_PART.fixed_base)

    def test_switch_mass_and_com_below_the_flange(self):
        com = T.SWITCH_PART.center_of_mass()
        self.assertLess(com[2], 0.0)  # most of the mass is the body, behind the flange
        self.assertAlmostEqual(T.SWITCH_PART.mass, 0.012)

    def test_rest_state_is_on_the_foam(self):
        s = T.SWITCH_PART.rest_state()
        self.assertAlmostEqual(float(s.low[2]) - L.TOP, L.FOAM_H, places=4)

    def test_rest_state_is_wrong_way_round(self):
        r = S.Reading(*(lambda s: (np.asarray(s.origin) - [0, 0, L.TOP], S.rotation(s.quat_xyzw), None, None))(
            T.SWITCH_PART.rest_state()))
        self.assertAlmostEqual(S.orient(r)[0], 180.0, places=3)

    def test_the_cutout_is_open(self):
        # no panel shape covers the cutout and the body's silhouette fits it
        s = T.PANEL_PART.rest_state()
        pts = np.array([[L.PX + dx, L.PY + dy, L.TOP + L.RAIL_H + L.PL_T / 2]
                        for dx in (-0.0105, 0.0105) for dy in (-0.0135, 0.0135)])
        self.assertTrue(np.all(s.distance(pts) > 0.0))

    def test_the_plate_is_solid_outside_the_cutout(self):
        s = T.PANEL_PART.rest_state()
        pts = np.array([[L.PX + 0.02, L.PY, L.TOP + L.RAIL_H + L.PL_T / 2],
                        [L.PX, L.PY + 0.02, L.TOP + L.RAIL_H + L.PL_T / 2]])
        self.assertTrue(np.all(s.distance(pts) <= 1e-9))

    def test_the_switch_body_passes_the_cutout_in_the_scorer_model(self):
        # the body's section is inside the cutout with its play; the flange is wider than the cutout
        self.assertLess(L.BODY_X / 2 + L.CLEAR_X, L.HOLE_X / 2 + 1e-9)
        self.assertGreater(L.FL_X / 2, L.HOLE_X / 2)

    def test_marks_are_on_the_right_halves(self):
        white = [s for s in T.switch_shapes() if tuple(s.color) == L.SW_WHITE]
        xs = sorted(s.center[0] for s in white)
        self.assertLess(xs[0], 0.0)
        self.assertGreater(xs[-1], 0.0)

    def test_instruction_names_no_numbers(self):
        self.assertNotRegex(T.INSTRUCTION, r"\d")


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        self.m = T.OfflineModel()

    def proxy(self, x, y, z, yaw=0.0):
        q = mk.quat_from_ypr(yaw, 0.0, 0.0)
        return {S.SWITCH: mk.state_from_poses(T.SWITCH_PART, (x, y, L.TOP + z, *q), source="offline")}

    def test_released_over_the_cutout_seats(self):
        s = self.m.states(self.proxy(L.PX + 0.001, L.PY, L.PLATE_TOP + L.DEPTH), FAR)[S.SWITCH]
        self.assertAlmostEqual(float(s.origin[2]) - L.TOP, L.SEAT_Z, places=6)

    def test_released_off_the_cutout_rests_on_the_plate(self):
        s = self.m.states(self.proxy(L.PX + 0.02, L.PY, L.PLATE_TOP + L.DEPTH), FAR)[S.SWITCH]
        self.assertAlmostEqual(float(s.origin[2]) - L.TOP, L.PLATE_TOP + L.DEPTH, places=6)

    def test_turned_4_degrees_jams(self):
        s = self.m.states(self.proxy(L.PX, L.PY, L.PLATE_TOP + L.DEPTH, yaw=math.radians(4.0)), FAR)[S.SWITCH]
        self.assertGreater(float(s.origin[2]) - L.TOP, L.PLATE_TOP + L.DEPTH - 1e-6)

    def test_turned_half_a_turn_still_seats(self):
        s = self.m.states(self.proxy(L.PX, L.PY, L.PLATE_TOP + L.DEPTH, yaw=math.pi), FAR)[S.SWITCH]
        self.assertAlmostEqual(float(s.origin[2]) - L.TOP, L.SEAT_Z, places=6)

    def test_held_in_the_air_is_left_alone(self):
        tcp = (L.PX, L.PY, L.TOP + 0.115)
        held = {"left": arm(tcp, opening=0.1), "right": FAR["right"]}
        o = self.proxy(L.PX, L.PY, 0.1)
        z = float(self.m.states(o, held)[S.SWITCH].origin[2]) - L.TOP
        self.assertAlmostEqual(z, 0.1, places=6)

    def test_offset_is_bounded_by_the_play(self):
        s = self.m.states(self.proxy(L.PX + 0.0025, L.PY - 0.0028, L.PLATE_TOP + L.DEPTH), FAR)[S.SWITCH]
        self.assertAlmostEqual(float(s.origin[0]) - L.PX, L.CLEAR_X, places=6)
        self.assertAlmostEqual(float(s.origin[1]) - L.PY, -L.CLEAR_Y, places=6)


class DemoTests(unittest.TestCase):
    def test_demo_is_one_arm_and_fits_the_run_limit(self):
        plan = T.plans()
        self.assertEqual(list(plan), ["left"])
        total = sum(w.duration for w in plan["left"])
        self.assertLess(total * HZ + 80, 1200)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertEqual(list(T.plans(v)), ["left"])
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_idle_arm_and_hold_only(self):
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_the_demo_turns_half_a_turn_between_pick_and_place(self):
        left = T.plans()["left"]
        self.assertAlmostEqual(abs(left[-1].yaw - left[0].yaw), math.pi, places=6)

    def test_pads_end_over_the_flange_top(self):
        self.assertAlmostEqual(T.SEAT_TCP + 0.002 - T.PAD_TIP, L.PLATE_TOP + L.FL_T + 0.0005, places=6)


class FolderTests(unittest.TestCase):
    def test_task_json_numbers_match_layout(self):
        j = json.loads((HERE / "task.json").read_text())
        s = j["setup"] + j["scoring"]
        for needle in ("22.0 (x) x 30.0 (y)", "0.75 mm of play", "1.0 mm along y", "41.5 mm", "x 360 mm, y +220 mm",
                       "x 200 mm, y +220 mm", "28.0 (y) x 20.5 (x) x 25 mm", "within 0.5 mm", "3 degrees", "20 degrees",
                       "30 mm", "0.3 degrees", "10 mm"):
            self.assertIn(needle, s)
        self.assertEqual(j["run_limit"], "1200 steps (60 s at 20 Hz)")
        self.assertIn("2.5 mm (x) and 3 mm (y)", j["scoring"])

    def test_no_dashes_or_team_names(self):
        for name in ("task.json", "README.md", "task.py", "scoring.py", "layout.py"):
            text = (HERE / name).read_text()
            self.assertNotIn("—", text, name)
            self.assertNotIn("–", text, name)
        self.assertIsNone(re.search(r"T-70|Vihan|Aggarwal|Robocurve team", (HERE / "task.json").read_text()))


if __name__ == "__main__":
    unittest.main()
