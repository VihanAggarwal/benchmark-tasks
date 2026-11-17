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
MAT_TOP_Z = L.MAT_T


def motor_state(qp=0.0, qn=0.0, pos=None, quat=(0.0, 0.0, 0.0, 1.0)):
    o = L.M_ORIGIN if pos is None else pos
    return mk.state_from_poses(T.MOTOR_PART, (*o, *quat), joints={S.BTN_P: qp, S.BTN_N: qn}, source="test")


def part_state(spec, origin, quat=(0.0, 0.0, 0.0, 1.0)):
    return mk.state_from_poses(spec, (*origin, *quat), source="test")


def at(x, y, z):
    """A world origin from table-relative z."""
    return (x, y, L.TOP + z)


def scene(shaft=None, whisk=None, qp=0.0, qn=0.0, motor_pos=None, shaft_quat=(0, 0, 0, 1), whisk_quat=(0, 0, 0, 1)):
    """Defaults: the shaft lying on the mat, the whisk home on the socket with both buttons out (the 1.0 end state)."""
    sh = shaft if shaft is not None else shaft_on_mat()
    wk = whisk if whisk is not None else L.M_ORIGIN
    return {S.MOTOR: motor_state(qp, qn, motor_pos), S.SHAFT: part_state(T.SHAFT_PART, sh, shaft_quat),
            S.WHISK: part_state(T.WHISK_PART, wk, whisk_quat)}


def shaft_on_mat(dx=0.0, dy=0.0, dz=0.0):
    return at(L.MAT_X + dx, L.MAT_Y - S.SHAFT_MID_U + dy, L.MAT_T + L.SH_BELL_R + dz)


def shaft_on_table(x=0.41, y=-0.40):
    return at(x, y - S.SHAFT_MID_U, L.SH_BELL_R)


def whisk_lying():
    return L.WK_ORIGIN


def start_scene():
    return {S.MOTOR: T.MOTOR_PART.rest_state(), S.SHAFT: T.SHAFT_PART.rest_state(), S.WHISK: T.WHISK_PART.rest_state()}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_hook_geometry_matches_the_text(self):
        self.assertAlmostEqual(L.HOOK_TIP_X, 0.0172)
        self.assertAlmostEqual(L.LIP_HALF - L.HOOK_TIP_X, 0.0018, places=9)  # the hook bites 1.8 mm into the lip
        self.assertAlmostEqual(L.HOOK_CLEARS, 0.0018)  # a button in 1.8 mm draws it clear: the text's figure
        self.assertAlmostEqual(L.LIP_U0 - L.HOOK_Y1, 0.0004, places=9)  # the hook's back face 0.4 mm behind the lip
        self.assertAlmostEqual(L.HOOK_IN_MAX, 0.0004)
        self.assertLess(L.HOOK_CLEARS, L.BTN_TRAVEL)
        self.assertAlmostEqual(L.BTN_Y, 0.045)  # 45 mm from the coupling end, as task.json says

    def test_the_stub_fits_the_socket(self):
        self.assertLess(L.LIP_HALF, L.BORE)  # 38 mm lip in a 41 mm socket
        self.assertLess(L.CORE_HALF_X, L.HOOK_TIP_X)  # the hooks sit in the core's groove
        self.assertAlmostEqual(L.STUB_LEN, 0.021)
        self.assertLess(L.STUB_LEN, L.BORE_DEPTH)  # the shoulder reaches the coupling face before the stub bottoms

    def test_grasp_widths_within_the_gripper(self):
        for w in (L.M_W, 2 * L.SH_BELL_R, 2 * L.WK_BOX_R):
            self.assertGreaterEqual(w, 0.016)
            self.assertLessEqual(w, 0.080)

    def test_start_layout_is_clear_and_reachable(self):
        st = start_scene()
        for n, s in st.items():
            self.assertAlmostEqual(float(s.low[2]), L.TOP, places=3, msg=n)  # lying on the table
        self.assertGreaterEqual(L.M_X, 0.10)
        self.assertLessEqual(L.M_X, 0.50)
        self.assertLess(abs(L.MAT_Y) + L.MAT_SIZE / 2, 0.45)
        self.assertLessEqual(L.MAT_X + L.MAT_SIZE / 2, 0.55)
        self.assertLess(L.WK_X, L.M_X)

    def test_start_scores_zero(self):
        res = run(start_scene())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertFalse(res.details["shaft_off"])

    def test_the_two_buttons_are_on_both_flanks(self):
        names = [j.name for j in T.MOTOR_PART.joints]
        self.assertEqual(sorted(names), sorted([S.BTN_P, S.BTN_N]))

    def test_pressing_a_button_moves_its_hook_out_of_the_lips_way(self):
        """Each button's frame carries its hook on the socket's OPPOSITE wall, so the press (toward the axis) moves the
        hook toward its wall: in by 1.8 mm the hook's tip reaches the lip's 19 mm; at rest it bites 1.8 mm into it.
        A stub pushed in moves the hook out the same way, which the joint's 3 mm travel allows."""
        import parts as P
        for s, name in ((+1, S.BTN_P), (-1, S.BTN_N)):
            j = next(jj for jj in T.MOTOR_PART.joints if jj.name == name)
            press = -1.0 if j.axis == "X" else +1.0  # the world x direction of a negative joint value (in)
            self.assertEqual(press, -s)
            shapes = P.button_shapes(s)
            hook = shapes[-2:]  # the flat and the ramp
            for sh in hook:
                self.assertEqual(np.sign(sh.center[0]), -s, msg=name)  # on the opposite wall
            tip = min(abs(sh.center[0]) - sh.half_extents()[0] for sh in hook)
            self.assertAlmostEqual(tip, L.HOOK_TIP_X, places=4)
            # in by q (q < 0): every point moves press * |q| along x, i.e. away from the axis on the -s side
            self.assertAlmostEqual(tip + L.HOOK_CLEARS, L.LIP_HALF, places=4)
            self.assertGreaterEqual(-j.lower, L.BTN_TRAVEL - 1e-9)

    def test_the_ramps_crest_is_the_hooks_innermost_point(self):
        """The flat block behind the crest stands HOOK_FLAT_RECESS further out than the ramp's crest, so a lip pushed up
        the ramp passes the crest with no front face of the flat block level with its edge (the seam that stopped the
        whisk on the GPU), and the flat's back face still bites well into the lip."""
        import parts as P
        for s in (+1, -1):
            flat, ramp = P.button_shapes(s)[-2:]
            flat_in = abs(flat.center[0]) - flat.half_extents()[0]
            ramp_in = abs(ramp.center[0]) - ramp.half_extents()[0]
            self.assertAlmostEqual(ramp_in, L.HOOK_TIP_X, places=6)
            self.assertAlmostEqual(flat_in - ramp_in, L.HOOK_FLAT_RECESS, places=6)
            self.assertGreaterEqual(L.HOOK_FLAT_RECESS, 0.0004)  # more than the solver's sub-mm penetration under load
            self.assertGreaterEqual(L.LIP_HALF - flat_in, 0.0012)  # the back face still bites over 1 mm into the lip
            self.assertGreater(flat_in, L.CORE_HALF_X)

    def test_only_the_hook_can_touch_a_stub(self):
        """The rest of each frame stays more than the socket's 20.5 mm from the axis (in x or z) at both ends of its
        travel, so a stub (lip 19 mm, 1.5 mm play) never meets it."""
        import parts as P
        for s in (+1, -1):
            for q in (L.HOOK_OUT, -L.BTN_TRAVEL):
                dx = s * q
                for sh in P.button_shapes(s)[:-2]:
                    c, h = np.asarray(sh.center, dtype=float), sh.half_extents()
                    lo_x, hi_x = c[0] + dx - h[0], c[0] + dx + h[0]
                    near_x = 0.0 if lo_x < 0.0 < hi_x else min(abs(lo_x), abs(hi_x))
                    near_z = 0.0 if c[2] - h[2] < 0.0 < c[2] + h[2] else min(abs(c[2] - h[2]), abs(c[2] + h[2]))
                    self.assertGreaterEqual(max(near_x, near_z), L.BORE + 0.0004, msg=(s, q, sh.center))

    def test_the_mat_is_clear_of_the_shaft_at_the_start(self):
        self.assertGreaterEqual(L.MAT_X - L.MAT_SIZE / 2, L.M_X + L.SH_BELL_R + 0.005)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        d = res.details
        self.assertTrue(d["shaft_on_mat"] and d["whisk_home"] and d["whisk_on"] and d["released_1s"] and d["still_1s"])

    def test_full_score_within_the_tolerances(self):
        for dx, dy in ((0.095, 0.0), (-0.095, 0.0), (0.0, 0.095), (0.0, -0.095)):
            self.assertEqual(run(scene(shaft=shaft_on_mat(dx, dy))).score, 1.0, (dx, dy))
        self.assertEqual(run(scene(shaft=shaft_on_mat(dz=0.004))).score, 1.0)  # 4 mm up: within 5 mm
        # whisk 0.9 mm short of the face, 2.5 mm off axis, 2.5 degrees out of square
        w = (L.M_X + 0.0025, L.M_Y0 - 0.0009, L.TOP + L.M_AXIS_Z)
        self.assertEqual(run(scene(whisk=w, whisk_quat=mk.quat_from_ypr(math.radians(2.5)))).score, 1.0)

    def test_half_whisk_on_shaft_on_the_table(self):
        res = run(scene(shaft=shaft_on_table()))
        self.assertEqual(res.score, 0.5)
        self.assertTrue(res.details["shaft_off"] and not res.details["shaft_on_mat"] and res.details["whisk_on"])

    def test_half_whisk_seated_on_one_hook_shaft_on_mat(self):
        for qp, qn in ((-0.002, 0.0), (0.0, -0.002), (-0.003, -0.003)):
            res = run(scene(qp=qp, qn=qn))
            self.assertEqual(res.score, 0.5, (qp, qn))
            self.assertTrue(res.details["whisk_home"] and not res.details["whisk_on"])

    def test_quarter_shaft_off_whisk_not_home(self):
        res = run(scene(whisk=whisk_lying()))
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["whisk_home"])

    def test_quarter_seated_with_the_shaft_elsewhere(self):
        res = run(scene(shaft=shaft_on_table(), qp=-0.002))
        self.assertEqual(res.score, 0.25)

    def test_zero_the_shaft_still_on(self):
        res = run(scene(shaft=L.M_ORIGIN, whisk=whisk_lying()))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["shaft_off"])
        # and with the whisk lying home beside it, a shaft not drawn out never scores
        self.assertEqual(run(scene(shaft=at(L.M_X, L.M_Y0 - 0.010, L.M_AXIS_Z))).score, 0.0)

    def test_shaft_drawn_out_by_just_over_half_a_millimetre_counts_off(self):
        tip_out = L.M_Y0 - L.STUB_LEN - 0.0006  # the stub's tip 0.6 mm outside the face
        self.assertTrue(run(scene(shaft=(L.M_X, tip_out, L.TOP + L.M_AXIS_Z), whisk=whisk_lying())).details["shaft_off"])
        tip_in = L.M_Y0 - L.STUB_LEN + 0.0006  # 0.6 mm inside
        self.assertFalse(run(scene(shaft=(L.M_X, tip_in, L.TOP + L.M_AXIS_Z), whisk=whisk_lying())).details["shaft_off"])

    def test_shaft_over_the_mat_but_not_down(self):
        self.assertEqual(run(scene(shaft=shaft_on_mat(dz=0.0055))).score, 0.5)  # 5.5 mm up: not on the mat
        self.assertEqual(run(scene(shaft=shaft_on_mat(dx=0.105))).score, 0.5)  # centre 105 mm off the mat's centre
        self.assertEqual(run(scene(shaft=shaft_on_mat(dy=-0.105))).score, 0.5)

    def test_whisk_just_short_or_turned_is_not_home(self):
        short = (L.M_X, L.M_Y0 - 0.0012, L.TOP + L.M_AXIS_Z)
        self.assertEqual(run(scene(whisk=short)).score, 0.25)
        off_axis = (L.M_X + 0.0035, L.M_Y0, L.TOP + L.M_AXIS_Z)
        self.assertEqual(run(scene(whisk=off_axis)).score, 0.25)
        self.assertEqual(run(scene(whisk_quat=mk.quat_from_ypr(math.radians(3.6)))).score, 0.25)
        self.assertEqual(run(scene(whisk_quat=mk.quat_from_ypr(0.0, 0.0, math.radians(4.0)))).score, 0.25)

    def test_whisk_pushed_in_past_the_face_is_not_home(self):
        deep = (L.M_X, L.M_Y0 + 0.002, L.TOP + L.M_AXIS_Z)  # 2 mm past the stop: the bore would not allow it
        self.assertEqual(run(scene(whisk=deep)).score, 0.25)

    def test_button_threshold_is_four_tenths_of_a_millimetre(self):
        self.assertEqual(run(scene(qp=-0.0004)).score, 1.0)
        self.assertEqual(run(scene(qp=-0.0006)).score, 0.5)
        self.assertEqual(run(scene(qn=-0.0006)).score, 0.5)

    def test_possible_scores(self):
        got = {run(s).score for s in (scene(), scene(shaft=shaft_on_table()), scene(whisk=whisk_lying()),
                                      start_scene())}
        self.assertEqual(got, {1.0, 0.5, 0.25, 0.0})

    def test_tier_function_table(self):
        t = S.tier
        self.assertEqual(t(False, True, True, True, True, True), 1.0)
        self.assertEqual(t(False, True, True, False, True, True), 0.5)
        self.assertEqual(t(False, True, True, True, True, False), 0.5)
        self.assertEqual(t(False, True, True, False, True, False), 0.25)
        self.assertEqual(t(False, True, True, True, False, False), 0.25)
        self.assertEqual(t(True, True, True, True, True, True), 0.0)
        self.assertEqual(t(False, False, True, True, True, True), 0.0)
        self.assertEqual(t(False, True, False, False, True, True), 0.0)


class ZeroConditions(unittest.TestCase):
    def test_fall_latch_is_permanent(self):
        tr = S.Tracker()
        bad = scene(whisk=(0.20, 0.60, L.TOP - 0.05))  # the whisk off the table's edge
        run(bad, steps=2, tracker=tr)
        self.assertTrue(tr.latched)
        res = run(scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_each_part_falling_latches(self):
        for kw in ({"shaft": at(0.41, -0.5, -0.05)}, {"whisk": at(0.2, 0.6, -0.05)},
                   {"motor_pos": at(0.3, 0.134, -0.05)}):
            tr = S.Tracker()
            run(scene(**kw), steps=1, tracker=tr)
            self.assertTrue(tr.latched, kw)

    def test_a_drop_to_29_mm_below_is_not_a_fall(self):
        tr = S.Tracker()
        run(scene(shaft=at(0.41, -0.1, -0.0 + L.SH_BELL_R - 0.029)), steps=2, tracker=tr)
        self.assertFalse(tr.latched)

    def test_a_hand_on_a_part_is_not_hands_off(self):
        near = dict(FAR)
        near["right"] = arm((L.M_X, L.M_Y0 - 0.045, L.TOP + 0.040))  # the pads on the whisk's gearbox
        res = run(scene(), arms=near)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_a_hand_on_the_motor_part_is_not_hands_off(self):
        near = dict(FAR)
        near["left"] = arm((L.M_X, L.M_Y0 + 0.19, L.TOP + 0.040))
        self.assertEqual(run(scene(), arms=near).score, 0.0)

    def test_finger_zone_gap_20_mm_holds_60_mm_lets_go(self):
        c = scene()[S.MOTOR]
        for gap, expect in ((0.020, 0.0), (0.060, 1.0)):
            x = float(c.high[0]) + gap + rigstate.zone_radius(1.0)
            a = dict(FAR)
            a["left"] = arm((x, L.M_Y0 + 0.12, L.TOP + 0.030 + 0.013))
            res = run(scene(), arms=a)
            self.assertEqual(res.score, expect, (gap, res.details["hands_gap_mm"]))

    def test_one_step_of_contact_resets_the_window(self):
        tr = S.Tracker()
        near = dict(FAR)
        near["right"] = arm((L.M_X, L.M_Y0 - 0.045, L.TOP + 0.040))
        run(scene(), tracker=tr)
        tr.update(scene(), near)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_unreadable_arms_never_released(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)
        bad = dict(FAR)
        bad["left"] = arm((float("nan"), 0.45, L.TOP + 0.25))
        self.assertEqual(run(scene(), arms=bad).score, 0.0)
        self.assertEqual(run(scene(), arms={"left": FAR["left"]}).score, 0.0)

    def test_still_means_visible_motion_only(self):
        tr = S.Tracker()
        for i in range(WIN + 1):  # the whisk creeping 4 mm along its axis over the second: not visible
            res = tr.update(scene(shaft=shaft_on_mat(dy=-0.002 + 0.004 * i / WIN)), FAR)
        self.assertEqual(res.score, 1.0)
        tr = S.Tracker()
        for i in range(WIN + 1):  # the shaft sliding 6 mm: visible
            res = tr.update(scene(shaft=shaft_on_mat(dy=0.006 * i / WIN)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_the_motor_part_moving_is_visible_too(self):
        tr = S.Tracker()
        for i in range(WIN + 1):
            x = L.M_X + 0.008 * i / WIN
            res = tr.update(scene(motor_pos=(x, L.M_Y0, L.TOP + L.M_AXIS_Z), whisk=(x, L.M_Y0, L.TOP + L.M_AXIS_Z)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_turning_5_degrees_is_visible(self):
        tr = S.Tracker()
        for i in range(WIN + 1):
            q = mk.quat_from_ypr(math.radians(7.0 * i / WIN))
            res = tr.update(scene(shaft=shaft_on_mat(), shaft_quat=q), FAR)
        self.assertEqual(res.score, 0.0)

    def test_needs_a_full_second(self):
        tr = S.Tracker()
        res = None
        for _ in range(WIN - 1):
            res = tr.update(scene(), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=4).score, 1.0)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        good = scene()
        tr = S.Tracker()
        run(good, tracker=tr)
        bad = scene()
        bad[S.SHAFT] = mk.state_from_poses(T.SHAFT_PART, (float("nan"), 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0), source="test")
        res = tr.update(bad, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(tr.update({S.MOTOR: good[S.MOTOR], S.SHAFT: good[S.SHAFT]}, FAR).score, 0.0)  # no whisk
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        self.assertEqual(run(good, tracker=tr).score, 1.0)  # recovers after a clean second

    def test_nan_button(self):
        self.assertEqual(run(scene(qp=float("nan"))).score, 0.0)
        self.assertEqual(run(scene(qn=float("inf"))).score, 0.0)

    def test_nan_quaternion(self):
        st = scene()
        st[S.WHISK] = dataclasses.replace(st[S.WHISK], quat_xyzw=np.array([float("nan"), 0.0, 0.0, 1.0]))
        self.assertEqual(run(st).score, 0.0)

    def test_infinite_velocity_state(self):
        st = scene()
        st[S.SHAFT] = mk.state_from_poses(T.SHAFT_PART, (*shaft_on_mat(), 0.0, 0.0, 0.0, 1.0),
                                          base_vel=(float("inf"), 0, 0, 0, 0, 0), source="test")
        self.assertEqual(run(st).score, 1.0)  # the scorer reads poses and joints, never velocities

    def test_distance_failure_is_not_released(self):
        st = scene()
        c = st[S.WHISK]

        class NoDist:
            origin, quat_xyzw, joints, spec = c.origin, c.quat_xyzw, c.joints, c.spec

            def link_pose(self, body):
                return c.link_pose(body)

            def distance(self, pts):
                raise RuntimeError("no distance")

        st[S.WHISK] = NoDist()
        self.assertEqual(run(st).score, 0.0)

    def test_reset_back_on_the_marks_clears_the_latch(self):
        tr = S.Tracker()
        run(scene(whisk=at(0.2, 0.6, -0.05)), steps=2, tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start_scene(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)

    def test_whisk_held_home_by_a_hand_never_scores(self):
        near = dict(FAR)
        near["right"] = arm((L.M_X, L.M_Y0 - 0.045, L.TOP + 0.040), opening=0.2)
        self.assertEqual(run(scene(), arms=near).score, 0.0)

    def test_a_button_held_in_by_a_hand_is_only_seated(self):
        """The whisk pushed home while a hand squeezes a button in: both hooks are not behind the lip, so it is seated
        (0.5 at best), and the hand that squeezes is not 30 mm clear anyway."""
        res = run(scene(qp=-0.003))
        self.assertEqual(res.score, 0.5)
        near = dict(FAR)
        near["left"] = arm((L.M_X, L.M_Y0 + 0.045, L.TOP + 0.040), opening=0.4)
        self.assertEqual(run(scene(qp=-0.003, qn=-0.003), arms=near).score, 0.0)

    def test_whisk_sitting_loose_in_the_socket_and_backed_off_is_not_home(self):
        """A whisk pushed in by a hand that has since let go, backed off over 1 mm by the spring: seated, not on."""
        backed = (L.M_X, L.M_Y0 - 0.0015, L.TOP + L.M_AXIS_Z)
        self.assertEqual(run(scene(whisk=backed)).score, 0.25)

    def test_shaft_left_in_the_socket_whisk_laid_on_top_scores_zero(self):
        res = run(scene(shaft=L.M_ORIGIN, whisk=at(L.M_X, L.M_Y0 + 0.05, L.M_AXIS_Z + 0.06)))
        self.assertEqual(res.score, 0.0)

    def test_shaft_dropped_off_the_table_scores_zero(self):
        tr = S.Tracker()
        run(scene(shaft=at(0.41, -0.60, -0.40)), steps=1, tracker=tr)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_shaft_across_the_mat_edge_is_not_on_the_mat(self):
        """Resting across the mat's edge, its bell on the table beside it and its middle over the mat: the lowest point
        is the table's, 10 mm under the mat's top."""
        res = run(scene(shaft=at(L.MAT_X, L.MAT_Y - S.SHAFT_MID_U, L.SH_BELL_R)))
        self.assertFalse(res.details["shaft_on_mat"])
        self.assertEqual(res.score, 0.5)

    def test_shaft_tipped_on_end_against_the_motor_part_scores_zero_or_half_never_full(self):
        q = mk.quat_from_ypr(0.0, math.radians(80.0), 0.0)
        res = run(scene(shaft=at(L.MAT_X, L.MAT_Y, 0.10), shaft_quat=q))
        self.assertLess(res.score, 1.0)

    def test_whisk_pushed_home_on_the_wrong_axis_height(self):
        low = (L.M_X, L.M_Y0, L.TOP + L.M_AXIS_Z - 0.004)
        self.assertEqual(run(scene(whisk=low)).score, 0.25)

    def test_motor_part_carried_away_with_the_whisk_still_scores_it(self):
        """The motor part's own place is not scored: the pair may be anywhere (the parts' relative pose is the test)."""
        o = at(0.35, 0.10, L.M_AXIS_Z)
        self.assertEqual(run(scene(motor_pos=o, whisk=o)).score, 1.0)

    def test_blender_dragged_with_the_shaft_still_in_is_zero(self):
        o = at(0.35, 0.00, L.M_AXIS_Z)
        self.assertEqual(run(scene(motor_pos=o, shaft=o, whisk=whisk_lying())).score, 0.0)

    def test_bounce_through_the_mat_does_not_pay(self):
        tr = S.Tracker()
        for i in range(WIN + 1):
            tr.update(scene(shaft=shaft_on_mat(dz=0.02 if i % 2 else 0.0)), FAR)
        self.assertEqual(tr.last.score, 0.0)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()

    def test_demo_plans_and_variants(self):
        os.environ.pop("DEMO_VARIANT", None)
        os.environ.pop("DEMO_HOLD_ONLY", None)
        os.environ.pop("DEMO_IDLE_ARM", None)
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
        self.assertLess(steps, 750)
        self.assertGreater(steps, 400)

    def test_one_hand_cannot_do_the_swap_offline(self):
        """The squeezing hand alone never draws the shaft; the pulling hand alone drags the blender with the shaft."""
        m = T.OfflineModel()
        bell = np.array([L.M_X, L.M_Y0 + T.BELL_GRIP_U, L.TOP + T.Z_GRIP])
        right = {"left": FAR["left"], "right": rigstate.Arm(bell, bell - np.array([0, 0, 0.013]), 0.25)}
        m.step(right)  # grips the bell
        for dy in np.linspace(0, -0.04, 10):
            b = bell + np.array([0.0, dy, 0.0])
            m.step({"left": FAR["left"], "right": rigstate.Arm(b, b - np.array([0, 0, 0.013]), 0.25)})
        self.assertTrue(m.inserted(T.SHAFT))  # still in: a hook held it, and the blender went with the hand
        self.assertLess(float(m.org[T.MOTOR][1]), L.M_Y0 - 0.02)


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
        for needle in ("0.62 kg", "41 mm", "24 mm", "1.8 mm", "0.4 mm", "3 mm", "45 mm", "100 mm", "5 mm", "30 mm",
                       "3 degrees", "1 s", "0.25", "0.5", "1.0", "10 N spring scale", "+x button", "-x button",
                       "1 mm feeler"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "1500")

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
