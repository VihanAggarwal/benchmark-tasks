"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import dataclasses
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
    import parts as P  # noqa: E402
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
Q0 = (0.0, 0.0, 0.0, 1.0)


def unit_state(o=0.0, a=0.0, origin=L.ORIGIN, quat=Q0):
    return mk.state_from_poses(T.UNIT_PART, (*origin, *quat), joints={S.DRAWER: o, S.LATCH: a}, source="test")


def catch_state(origin=L.ORIGIN):
    return mk.state_from_poses(T.CATCH_PART, (*origin, *Q0), source="test")


FB = L.FRONT_BACK  # the slider's travel is the gap behind the inset front plus 15 mm


def scene(o=0.150, a=0.0):
    """``o`` is the slider's travel. Defaults: 150 mm of travel (a 135 mm gap), latch at rest (the demo's end
    state)."""
    return {S.UNIT: unit_state(o, a), S.CATCH: catch_state()}


def with_joint(states, name, value):
    st = dict(states)
    u = st[S.UNIT]
    st[S.UNIT] = dataclasses.replace(u, joints={**dict(u.joints), name: value})
    return st


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def on_bar(o, opening=0.1):
    """The right gripper closed on the handle bar with the drawer ``o`` open."""
    a = dict(FAR)
    a["right"] = arm((T.BAR_X0 - o, T.GRIP_Y, L.TOP + T.Z_BAR), opening)
    return a


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def ramp(tr, o0, o1, a=0.0, steps=WIN + 3, arms=FAR):
    res = None
    for i in range(steps):
        res = tr.update(scene(o0 + (o1 - o0) * i / (steps - 1), a), arms)
    return res


class Geometry(unittest.TestCase):
    def test_the_drawer_stops_at_the_guides_one_and_a_half_inches(self):
        self.assertAlmostEqual(L.ENGAGE_GAP, 0.038, places=6)
        self.assertLess(abs(L.ENGAGE_GAP - 1.5 * 0.0254), 0.001)
        self.assertAlmostEqual(L.ENGAGE_AT, L.ENGAGE_GAP + FB, places=9)
        self.assertAlmostEqual(L.PAST_GAP, 0.050, places=6)
        self.assertAlmostEqual(L.BITE, 0.006, places=6)
        self.assertLess(L.ENGAGE_GAP, L.OPEN_PAST)
        self.assertAlmostEqual(L.OPEN_PAST - L.PAST_GAP, 0.010, places=6)  # 60 mm: the hook 10 mm past the lip

    def test_catch_one_inch_in_and_latch_83_mm(self):
        self.assertLess(abs(L.CATCH_X0 - 0.0254), 0.001)
        self.assertAlmostEqual(L.HOOK_X1 - FB, L.LATCH_LEN, places=6)
        self.assertAlmostEqual(L.TAB_Z1 - L.TAB_Z0, 0.030, places=6)
        self.assertAlmostEqual(2 * L.LATCH_HALF_W, 0.021, places=6)

    def test_the_published_muji_sizes(self):
        self.assertEqual((L.W, L.D, L.H), (0.370, 0.280, 0.185))  # outside, published
        self.assertAlmostEqual(2 * L.BOX_Y0, 0.335, places=6)  # drawer inside width, published
        self.assertAlmostEqual(L.BOX_LEN - L.BOX_SIDE_T, 0.245, places=6)  # drawer inside depth, published
        self.assertAlmostEqual(L.BOX_Z1 - (L.BOTTOM_Z0 + L.BOTTOM_T), 0.155, places=6)  # drawer inside height
        # the estimates fit the published sizes: the drawer inside the carcass, the front in its opening
        self.assertLess(L.BOX_Y0 + L.BOX_SIDE_T, L.SIDE_Y0)
        self.assertLess(FB + L.BOX_LEN, L.D - L.T_BACK)
        self.assertLess(L.BOX_Z1, L.TOP_Z0)
        self.assertLess(L.FRONT_W / 2, L.SIDE_Y0)
        self.assertTrue(L.T < L.FRONT_Z0 and L.FRONT_Z1 < L.TOP_Z0)
        self.assertLess(L.TAB_Z1, L.FRONT_Z1)  # the latch's foot fits on the back of the front
        # the drawer's back never reaches the catch within the slider's travel
        self.assertGreater(FB + L.BOX_LEN - L.BOX_SIDE_T - L.TRAVEL, L.CATCH_X1)

    def test_latch_and_catch_sit_as_on_the_gpu_build(self):
        """Placed from the back of the front and the top's underside exactly as on the unit that scored 1.0."""
        self.assertAlmostEqual(L.HINGE[0] - FB, 0.005, places=6)
        self.assertAlmostEqual(L.TOP_Z0 - L.HINGE[2], 0.004, places=6)
        self.assertAlmostEqual(L.TOP_Z0 - L.ARM_Z1, 0.022, places=6)
        self.assertAlmostEqual(L.TOP_Z0 - L.LIP_Z0, 0.014, places=6)
        self.assertAlmostEqual(L.TOP_Z0 - L.HOOK_Z1, 0.008, places=6)
        self.assertAlmostEqual(L.HOOK_X0 - FB, 0.077, places=6)

    def test_the_handle_is_the_ikea_bar(self):
        self.assertEqual((L.HANDLE_LEN, L.HANDLE_W, L.HANDLE_DEPTH, L.HANDLE_HOLES), (0.143, 0.009, 0.031, 0.128))
        self.assertAlmostEqual(L.BEHIND_BAR, 0.022, places=6)
        self.assertAlmostEqual(L.FRONT_Z1 - L.HANDLE_Z, 0.031, places=6)

    def test_the_handle_is_within_reach_and_clear_of_the_latch(self):
        self.assertLess(abs(T.GRIP_Y), L.HANDLE_HOLES / 2 - 0.010)
        self.assertGreater(L.LATCH_Y - T.GRIP_Y, 0.10)  # the two grip points 190 mm apart
        # the pressing pads (27 mm wide along y) stay 12 mm or more off the drawer's side
        self.assertGreater(L.BOX_Y0 - (L.LATCH_Y + rigstate.FINGER_HALF_WIDTH), 0.012)
        # the gripping pads' upper part (57 mm above the TCP) passes over the chest's top edge
        self.assertGreater(T.Z_BAR + 0.057 - L.H, 0.010)

    def test_clear_angle_puts_the_hook_under_the_lip(self):
        self.assertAlmostEqual(L.corner_z(L.CLEAR_RAD), L.LIP_Z0, places=6)
        self.assertGreater(L.corner_z(0.0), L.LIP_Z0 + 0.005)
        self.assertLess(L.corner_z(L.LATCH_UPPER), L.LIP_Z0 - 0.010)
        self.assertTrue(0.05 < L.CLEAR_RAD < 0.12)

    def test_the_pivot_is_above_the_contact_so_a_pull_turns_the_hook_in(self):
        # the lip pushes the hook's face toward +x below the pivot: the torque turns the hook up (negative angle)
        self.assertGreater(L.HINGE[2], L.HOOK_Z1)
        r = np.array([L.HOOK_X0 - L.HINGE[0], 0.0, (L.LIP_Z0 + L.HOOK_Z1) / 2 - L.HINGE[2]])
        torque_y = np.cross(r, np.array([1.0, 0.0, 0.0]))[1]
        self.assertLess(torque_y, 0.0)  # -y torque about the pivot: toward a negative angle, the hook up

    def test_press_force_is_a_few_newtons(self):
        # the setup's figure: about 2 N pressed 12.5 mm from the root draws the hook clear
        force = L.LATCH_SPRING * L.CLEAR_RAD / 0.0125
        self.assertTrue(1.5 < force < 2.5, force)
        # where the demo's pads meet the arm (their edge nearest the pivot) it takes under 3 N
        r = T.PRESS_X - (L.UX - T.EASE) - T.TIP_HALF - L.HINGE[0]
        self.assertGreaterEqual(r, 0.012)
        self.assertLess(L.LATCH_SPRING * L.CLEAR_RAD / r, 3.0)

    def test_the_demo_press_clears_the_hook_with_a_soft_arm(self):
        """The arms are PD springs: the pads stop where the latch spring's push balances their give. The demo's press
        clears the hook with margin from a stiff arm to twice the modelled give, and 4 mm nearer the pivot."""
        r = T.PRESS_X - (L.UX - T.EASE) - T.TIP_HALF - L.HINGE[0]
        z = T.Z_PRESS - T.TIP
        self.assertGreater(T.pressed_angle(r, z), 1.8 * L.CLEAR_RAD)
        for give in (0.0, 0.0016, T.ARM_GIVE, 2 * T.ARM_GIVE):
            self.assertGreater(T.pressed_angle(r, z, give), 1.1 * L.CLEAR_RAD, give)
        self.assertGreater(T.pressed_angle(r - 0.004, z, 0.003), 1.25 * L.CLEAR_RAD)
        self.assertEqual(T.pressed_angle(r, z, 0.0), L.LATCH_UPPER)  # without give it meets the arm's stop

    def test_the_first_gpu_press_does_not_clear_offline(self):
        """17 Nov GPU run: pads centred in the gap's middle, tips commanded 3 mm into the arm: 2.5 deg on the GPU (the
        hook needs 4.8). The offline model must call that press short too."""
        r = (T.EASE - FB) / 2 - T.TIP_HALF - (L.HINGE[0] - FB)
        self.assertLess(T.pressed_angle(r, L.ARM_Z1 - 0.003, 0.0016), 0.7 * L.CLEAR_RAD)

    def test_pressing_at_the_eased_gap_never_sweeps_the_hook_into_the_lip(self):
        for a in np.linspace(0.0, L.LATCH_UPPER, 26):
            r = S.read(scene(T.EASE, float(a)))
            self.assertEqual(float(S.overlap(r)[0] * S.overlap(r)[2]), 0.0, a)

    def test_the_gap_takes_the_closed_pads(self):
        x = T.PRESS_X - L.UX  # the pads' centre in the base frame
        self.assertGreater(x - T.TIP_HALF, FB - T.EASE + 0.015)  # 18 mm from the back of the front
        self.assertLess(x + T.TIP_HALF, -0.005 + 1e-9)  # and 5 mm from the carcass's front edge
        self.assertGreater(T.Z_PRESS - T.TIP, L.BOX_Z0)

    def test_nothing_overlaps_at_the_start(self):
        r = S.read({S.UNIT: T.UNIT_PART.rest_state(), S.CATCH: T.CATCH_PART.rest_state()})
        self.assertEqual(float(np.prod(S.overlap(r))), 0.0)
        self.assertGreater(L.HOOK_X0 - L.CATCH_X1, 0.03 + FB)

    def test_start_scores_zero(self):
        st = {S.UNIT: T.UNIT_PART.rest_state(), S.CATCH: T.CATCH_PART.rest_state()}
        self.assertEqual(run(st).score, 0.0)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        res = run(scene(0.150))
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_threshold_100_mm(self):
        self.assertEqual(run(scene(FB + 0.1005)).score, 1.0)
        self.assertEqual(run(scene(FB + 0.0995)).score, 0.5)

    def test_half(self):
        self.assertEqual(run(scene(FB + 0.080)).score, 0.5)

    def test_threshold_60_mm(self):
        self.assertEqual(run(scene(FB + 0.0605)).score, 0.5)
        self.assertEqual(run(scene(FB + 0.0595)).score, 0.0)

    def test_on_the_catch_is_zero(self):
        self.assertEqual(run(scene(L.ENGAGE_AT)).score, 0.0)
        self.assertEqual(run(scene(0.0)).score, 0.0)

    def test_quarter_held(self):
        self.assertEqual(run(scene(0.150), arms=on_bar(0.150)).score, 0.25)
        self.assertEqual(run(scene(0.080), arms=on_bar(0.080)).score, 0.25)

    def test_quarter_moving(self):
        tr = S.Tracker()
        res = ramp(tr, 0.100, 0.160)
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["still_1s"])

    def test_possible_scores(self):
        seen = {S.tier(f, o, s) for f in (False, True) for o in (0.0, 0.05, 0.07, 0.12) for s in (False, True)}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})

    def test_tier_table(self):
        self.assertEqual(S.tier(True, 0.2, True), 0.0)
        self.assertEqual(S.tier(False, 0.2, False), 0.25)
        self.assertEqual(S.tier(False, 0.07, False), 0.25)
        self.assertEqual(S.tier(False, 0.07, True), 0.5)
        self.assertEqual(S.tier(False, 0.1, True), 1.0)


class ZeroConditions(unittest.TestCase):
    def test_hook_forced_into_the_lip_latches_zero(self):
        tr = S.Tracker()
        res = run(scene(L.ENGAGE_AT + 0.006, 0.0), tracker=tr, steps=1)  # the hook 6 mm into the lip, not pressed
        self.assertTrue(tr.latched)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(run(scene(0.150), tracker=tr).score, 0.0)  # permanent

    def test_two_mm_into_the_lip_is_not_forced(self):
        tr = S.Tracker()
        run(scene(L.ENGAGE_AT + 0.0019, 0.0), tracker=tr, steps=1)
        self.assertFalse(tr.latched)

    def test_tunnelling_through_the_lip_in_one_step(self):
        tr = S.Tracker()
        tr.update(scene(L.ENGAGE_AT, 0.0), FAR)
        tr.update(scene(FB + 0.090, 0.0), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(0.150), tracker=tr).score, 0.0)

    def test_the_real_way_past_never_latches(self):
        tr = S.Tracker()
        ramp(tr, 0.0, L.ENGAGE_AT, steps=30)
        ramp(tr, L.ENGAGE_AT, T.EASE, steps=20)
        for a in np.linspace(0.0, 0.2, 20):
            tr.update(scene(T.EASE, float(a)), FAR)
        ramp(tr, T.EASE, T.CLEAR, a=0.2, steps=20)
        ramp(tr, T.CLEAR, T.FULL, steps=60)
        res = run(scene(T.FULL), tracker=tr)
        self.assertFalse(tr.latched)
        self.assertEqual(res.score, 1.0)

    def test_riding_the_lips_underside_is_not_forced(self):
        tr = S.Tracker()
        for o in np.linspace(L.ENGAGE_AT + 0.001, L.PAST_AT, 13):
            tr.update(scene(float(o), L.CLEAR_RAD), FAR)
        self.assertFalse(tr.latched)

    def test_finger_zone_gap_20_mm_holds_40_mm_lets_go(self):
        st = scene(0.150)
        x_front = L.UX - 0.150 + L.HANDLE_X0  # the bar's front face, world x
        for gap, expect in ((0.020, 0.25), (0.040, 1.0)):
            x = x_front - gap - rigstate.zone_radius(1.0)
            a = dict(FAR)
            a["right"] = arm((x, 0.0, L.TOP + L.HANDLE_Z + 0.013), 1.0)
            self.assertEqual(run(st, arms=a).score, expect, gap)

    def test_a_finger_resting_on_the_latch_is_not_hands_off(self):
        a = dict(FAR)
        a["left"] = arm((L.UX - 0.150 + 0.03, L.LATCH_Y, L.TOP + L.ARM_Z1 + 0.013), 0.0)
        self.assertEqual(run(scene(0.150), arms=a).score, 0.25)

    def test_a_hand_on_the_catch_side_of_the_top_is_not_hands_off(self):
        a = dict(FAR)
        a["left"] = arm((L.UX + 0.03, L.LATCH_Y, L.TOP + L.H + 0.013 + 0.005), 0.0)
        self.assertEqual(run(scene(0.150), arms=a).score, 0.25)

    def test_one_step_of_contact_resets_the_window(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        tr.update(scene(), on_bar(0.150))
        self.assertEqual(tr.update(scene(), FAR).score, 0.25)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_unreadable_arms_never_released(self):
        self.assertEqual(run(scene(), arms=None).score, 0.25)
        bad = dict(FAR)
        bad["left"] = arm((float("nan"), 0.45, L.TOP + 0.25))
        self.assertEqual(run(scene(), arms=bad).score, 0.25)
        self.assertEqual(run(scene(), arms={"left": FAR["left"]}).score, 0.25)

    def test_still_is_five_mm(self):
        tr = S.Tracker()
        self.assertEqual(ramp(tr, 0.150, 0.154).score, 1.0)
        tr = S.Tracker()
        self.assertEqual(ramp(tr, 0.150, 0.157).score, 0.25)

    def test_needs_a_full_second(self):
        self.assertEqual(run(scene(), steps=WIN).score, 0.25)
        self.assertEqual(run(scene(), steps=WIN + 1).score, 1.0)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        self.assertEqual(tr.update(scene(float("nan")), FAR).score, 0.0)
        self.assertEqual(tr.update(with_joint(scene(), S.LATCH, float("inf")), FAR).score, 0.0)
        self.assertEqual(tr.update({S.UNIT: scene()[S.UNIT]}, FAR).score, 0.0)  # no catch
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        self.assertEqual(tr.bad_steps, 4)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)  # recovers after a clean second

    def test_nan_origin(self):
        st = scene()
        st[S.UNIT] = unit_state(0.15, 0.0, origin=(float("nan"), 0.0, L.TOP))
        self.assertEqual(run(st).score, 0.0)

    def test_infinite_velocity_state(self):
        st = scene()
        st[S.UNIT] = mk.state_from_poses(T.UNIT_PART, (*L.ORIGIN, *Q0), joints={S.DRAWER: 0.15, S.LATCH: 0.0},
                                         base_vel=(float("inf"), 0, 0, 0, 0, 0), source="test")
        self.assertEqual(run(st).score, 1.0)  # the scorer reads poses and joints, never velocities

    def test_distance_failure_is_not_released(self):
        st = scene()
        c = st[S.CATCH]

        class NoDist:
            origin, quat_xyzw, joints, spec = c.origin, c.quat_xyzw, c.joints, c.spec

            def link_pose(self, body):
                return c.link_pose(body)

            def distance(self, pts):
                raise RuntimeError("no distance")

        st[S.CATCH] = NoDist()
        self.assertEqual(run(st).score, 0.25)

    def test_reset_back_to_shut_clears_the_latch(self):
        tr = S.Tracker()
        tr.update(scene(L.ENGAGE_AT, 0.0), FAR)
        tr.update(scene(FB + 0.090, 0.0), FAR)  # tunnelled
        self.assertTrue(tr.latched)
        tr.update(scene(0.0, 0.0), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_a_slow_close_is_not_a_reset(self):
        tr = S.Tracker()
        tr.update(scene(L.ENGAGE_AT + 0.006, 0.0), FAR)
        ramp(tr, L.ENGAGE_AT + 0.006, 0.0, steps=40)
        self.assertTrue(tr.latched)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(scene(0.0)).score, 0.0)

    def test_pulling_on_the_catch_scores_zero(self):
        self.assertEqual(run(scene(L.ENGAGE_AT), arms=on_bar(L.ENGAGE_AT)).score, 0.0)

    def test_the_unit_carried_off_with_the_drawer_shut_scores_zero(self):
        st = {S.UNIT: unit_state(0.0, origin=(L.UX - 0.15, 0.0, L.TOP)), S.CATCH: catch_state()}
        self.assertEqual(run(st).score, 0.0)

    def test_opening_is_the_joint_not_a_pose(self):
        # a drawer box lifted or shifted is not an opening: only the slider's travel counts
        st = {S.UNIT: unit_state(0.0, origin=(L.UX, 0.0, L.TOP + 0.2)), S.CATCH: catch_state()}
        self.assertEqual(run(st).score, 0.0)

    def test_a_latch_pressed_and_left_held_is_not_full(self):
        a = dict(FAR)
        a["left"] = arm((L.UX - 0.150 + 0.0175, L.LATCH_Y, L.TOP + T.Z_PRESS), 0.0)
        self.assertEqual(run(scene(0.150, 0.2), arms=a).score, 0.25)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertIn("left", d)
        self.assertIn("right", d)
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertIn("right", T.demo())
            finally:
                os.environ.pop("DEMO_VARIANT")
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for side, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = side
            try:
                d = T.demo()
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
            self.assertNotIn(side, d)
            self.assertIn(other, d)

    def test_demo_length_is_within_half_the_run_limit(self):
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps, 500)
        self.assertGreater(steps, 400)

    def test_left_waits_for_the_ease_off(self):
        right, left = T.right_plan(), T.left_plan("", T.right_plan())
        t_down = T._until(left, "over the latch") - 1.0
        self.assertGreaterEqual(t_down + 1e-9, T._until(right, "ease off"))
        self.assertLessEqual(T._until(left, "keep it down"), T._until(right, "pull past the catch") + 1e-9)

    def _play(self, plans, lag=False):
        """Step the offline model through timed plans as the robot runs them: from the ready pose (TCP 323 mm ahead
        of each base, 261 mm up, read from the GPU traces), smoothstep in time between waypoints, fingers linear in the
        same progress. ``lag``: x and y follow the square of that progress while z and the fingers keep it, so the
        hand comes down before it gets across, as the right arm did on GPU run 3 (x 62 % of the way when z was 78 %)."""
        m, tr = T.OfflineModel(), S.Tracker()
        starts = {"left": np.array([0.3234, 0.305, L.TOP + 0.2606]),
                  "right": np.array([0.3234, -0.305, L.TOP + 0.2606])}
        total = max(sum(w.duration for w in p) for p in plans.values()) + 3.0
        res = None
        for k in range(int(total * HZ)):
            t = k / HZ
            arms = {}
            for side in ("left", "right"):
                pos, fing = starts[side], T.OPEN
                if side in plans:
                    t0, prev = 0.0, (starts[side], T.OPEN)
                    for w in plans[side]:
                        if t <= t0 + w.duration:
                            u = (t - t0) / w.duration if w.duration else 1.0
                            f = u * u * (3.0 - 2.0 * u)
                            fxy = f * f if lag else f
                            d = np.asarray(w.pos) - prev[0]
                            pos = prev[0] + np.array([fxy * d[0], fxy * d[1], f * d[2]])
                            fing = prev[1] + f * (w.finger - prev[1])
                            break
                        t0, prev = t0 + w.duration, (np.asarray(w.pos, dtype=float), w.finger)
                    else:
                        pos, fing = prev
                arms[side] = arm(pos, fing / T.OPEN if T.OPEN else 0.0)
            m.step(arms)
            res = tr.update(m.states(), arms)
        return res, m

    def _run3_plans(self):
        """The demo as GPU run 3 (17 Nov, MUJI chest, 0.0) ran it: the right hand went from the ready pose straight
        to over the handle (TCP 180 mm up) in 2.5 s, the fingers closing to straddle on the way, on the bar 50 mm
        right of centre; the rest of the plan as now."""
        right = T.right_plan()
        assert right[0].label.startswith("in front of the chest") and right[1].label == "over the handle"
        right = [dataclasses.replace(right[1], duration=2.5)] + right[2:]
        right = [dataclasses.replace(w, pos=(w.pos[0], -0.050, w.pos[2])) if abs(w.pos[1] - T.GRIP_Y) < 1e-9 else w
                 for w in right]
        return {"left": T.left_plan("", right), "right": right}

    def test_demo_scores_full_offline(self):
        for lag in (False, True):
            res, m = self._play(T.plans(), lag=lag)
            self.assertEqual(res.score, 1.0, (lag, res.details, m.blocked))
            self.assertAlmostEqual(m.o, T.FULL, places=3)
            self.assertEqual(m.blocked, {}, lag)

    def test_gpu_run3_right_pads_land_on_the_chest_top(self):
        """GPU run 3 (17 Nov, MUJI chest): the right TCP stalled at z 197 mm (pad tips on the chest's top at 185 mm)
        from step 43 to 127 while commanded down to 143 mm; it then slid off the edge closed on nothing, the drawer
        never opened past -6 mm, and the left pads came down in front of the drawer, not into a gap. Score 0.
        The offline model must see the pads land on the top (it used to pass through it and predict 1.0)."""
        for lag in (False, True):
            res, m = self._play(self._run3_plans(), lag=lag)
            self.assertEqual(res.score, 0.0, lag)
            self.assertIn("right", m.blocked, lag)
            self.assertEqual(m.blocked["right"][0], "chest[1]", (lag, m.blocked))  # the top panel
            self.assertNotIn("right", m.hold)
            self.assertLess(m.o, L.FRONT_BACK, lag)  # the gap never opened
            self.assertLess(m.blocked["right"][2], 50, lag)  # on the way down to the handle (GPU: step 37 to 43)
            self.assertEqual(m.a, 0.0, lag)  # with no gap the left pads never reach the latch

    def test_the_grip_clears_the_handle_legs(self):
        """The rear pad goes down between the bar and the front, where the legs are. 27 mm wide at the TCP, it clears
        the leg at -64 mm from GRIP_Y; at run 3's -50 mm its corner comes down onto the leg."""
        tcp = np.array([T.BAR_X0, T.GRIP_Y, L.TOP + T.Z_BAR])
        tips = tcp - np.array([0.0, 0.0, T.TIP])
        gap = -2 * T.STRADDLE
        self.assertIsNone(T.pad_hit(tcp, tips, gap, 0.0))
        self.assertIsNone(T.pad_hit(tcp + [0.0, -0.002, 0.0], tips + [0.0, -0.002, 0.0], gap, 0.0))  # 2 mm drift
        hit = T.pad_hit(tcp + [0.0, -0.050 - T.GRIP_Y, 0.0], tips + [0.0, -0.050 - T.GRIP_Y, 0.0], gap, 0.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "handle leg[0]")

    def test_the_straddling_pads_fit_in_front_of_the_chest(self):
        """At the bar the rear pad (6 mm) stands 7.5 mm or more in front of the carcass's edge and the drawer front
        (flush, x 300 mm), and it is the chest's top (185 mm) that the run 3 approach came down on."""
        rear_face = T.BAR_X0 + (-2 * T.STRADDLE) / 2 + T.PAD_T
        self.assertLessEqual(rear_face, L.UX - 0.0075 + 1e-9)
        self.assertGreater(L.TOP + L.H, L.TOP + T.APPROACH_R - T.TIP)  # over the bar, the tips are under the top
        self.assertLess(T.FRONT_R[0] + (-2 * T.STRADDLE) / 2 + T.PAD_T, L.UX - 0.040)

    def test_the_pressing_pads_stay_off_the_chest_and_the_drawer(self):
        """The left pads, closed, in the gap at the press and while moving out with the drawer: clear of the carcass's
        edge, the back of the front, the latch's foot and the drawer's side (the latch arm itself is the sprung link
        and is pressed)."""
        for o in (T.EASE, (T.EASE + T.CLEAR) / 2, T.CLEAR):
            tcp = np.array([T.PRESS_X - (o - T.EASE), L.UY + L.LATCH_Y, L.TOP + T.Z_PRESS])
            self.assertIsNone(T.pad_hit(tcp, tcp - [0.0, 0.0, T.TIP], 0.0, o), o)
        tcp = np.array([T.PRESS_X, L.UY + L.LATCH_Y, L.TOP + T.HOVER_L])  # over the latch, before going down
        self.assertIsNone(T.pad_hit(tcp, tcp - [0.0, 0.0, T.TIP], 0.0, T.EASE))
        self.assertLessEqual(T.PRESS_X + T.TIP_HALF, L.UX - 0.005 + 1e-9)  # 5 mm in front of the carcass's edge

    def test_the_hand_cannot_hover_over_the_bar_above_the_chest(self):
        """Why the right hand must come in from the front: over the bar the straight-down reach limit (joint 4) keeps
        the pad tips under the chest's top (offline kinematics)."""
        try:
            import yamkin as K
        except ModuleNotFoundError:
            self.skipTest("tools/offline not on the path")
        base = K.BASES["station"]["right"]
        ok = K.reachable((T.BAR_X0, T.GRIP_Y, L.TOP + L.H + T.TIP + 0.005), yaw=T.YAW_R, base_pos=base)[0]
        self.assertFalse(ok)
        self.assertTrue(K.reachable((T.BAR_X0, T.GRIP_Y, L.TOP + T.APPROACH_R), yaw=T.YAW_R, base_pos=base)[0])
        self.assertTrue(K.reachable((T.FRONT_R[0], T.FRONT_R[1], L.TOP + T.FRONT_R[2]), yaw=T.YAW_R,
                                    base_pos=base)[0])

    def test_both_arms_close_along_world_x(self):
        """pad_hit takes the pads along world x: both demo yaws are +-90 deg."""
        self.assertAlmostEqual(abs(T.YAW_R), math.pi / 2)
        self.assertAlmostEqual(abs(T.YAW_L), math.pi / 2)
        for plan in T.plans().values():
            for w in plan:
                self.assertAlmostEqual(abs(w.yaw), math.pi / 2)

    def test_one_hand_cannot_do_it_offline(self):
        for side in ("left", "right"):
            p = T.plans()
            p.pop(side)
            res, m = self._play(p)
            self.assertEqual(res.score, 0.0, side)
            self.assertLessEqual(m.o, L.ENGAGE_AT + 1e-6)

    def test_variants_offline(self):
        expect = {"no_press": 0.0, "half_open": 0.5, "hold_on": 0.25}
        for v, score in expect.items():
            res, _ = self._play(T.plans(v))
            self.assertEqual(res.score, score, v)


class Folder(unittest.TestCase):
    def test_text_has_no_em_dashes_or_identifying_names(self):
        bad = re.compile(chr(0x2014) + "|" + chr(0x2013) + "|T-70|VihanAggarwal|Claude|Anthropic|co-author", re.I)
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "parts.py"):
            p = HERE / name
            if p.exists():
                self.assertIsNone(bad.search(p.read_text()), name)

    def test_task_json_matches_the_code(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        d = json.loads(p.read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        text = d["setup"] + d["scoring"]
        for needle in ("48390", "83 x 30 x 21 mm", "BAGGANAS 143 mm", "128 mm", "31 mm", "MUJI", "4550512381105",
                       "370 mm wide", "280 mm deep", "185 mm tall", "335 x 245 x 155 mm", "1.4 kg (est)",
                       "1.8 N at 30 mm/s", "200 mm", "140 mm left", "15 mm",
                       "0.30 N m/rad", "about 2 N", "12.5 mm", "25 mm in", "159 mm", "38 mm", "6 mm",
                       "more than 2 mm", "30 mm or more", "under 5 mm", "60 mm", "100 mm", "0.25", "0.5", "1.0",
                       "steel rule", "30 fps"):
            self.assertIn(needle, text, needle)
        self.assertIn(f"{L.LATCH_SPRING:.2f} N m/rad", d["setup"])
        self.assertAlmostEqual(L.DRAWER_DAMPING * T.PULL_SPEED, 1.8, places=6)
        self.assertAlmostEqual(L.LATCH_SPRING * L.CLEAR_RAD / 0.0125, 2.0, delta=0.05)
        self.assertEqual(d["run_limit"].split()[0], "1000")
        self.assertEqual(d["category"], "Societal relatability")

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
