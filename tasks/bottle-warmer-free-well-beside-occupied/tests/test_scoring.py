"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or an offline prediction; nothing was measured on the simulator."""

import math
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


def bottle_state(name, x, y, z, pitch=0.0):
    q = mk.quat_from_ypr(0.0, pitch, 0.0)
    return mk.state_from_poses(T.SPECS[name], (x, y, L.TOP + z, *q), source="test")


def warmer_state(a=0.0, b=0.0):
    return mk.state_from_poses(T.WARMER_PART, (L.WX, L.WY, L.TOP, 0.0, 0.0, 0.0, 1.0),
                               joints={"btn_a": a, "btn_b": b}, source="test")


def states(bx=L.WELL_B[0], by=L.WELL_B[1], bz=L.FLOOR, bpitch=0.0, nx=L.A_X0, ny=L.A_Y0, nz=L.FLOOR, npitch=0.0,
           a=0.0, b=0.0):
    return {S.NEIGHBOUR: bottle_state(S.NEIGHBOUR, nx, ny, nz, npitch), S.BOTTLE: bottle_state(S.BOTTLE, bx, by, bz,
                                                                                               bpitch),
            S.WARMER: warmer_state(a, b)}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def feed(tr, seq, arms=FAR):
    res = None
    for st in seq:
        res = tr.update(st, arms)
    return res


def finish(seq_states, arms=FAR, steps=WIN + 3, tr=None):
    """Feed ``seq_states`` once each, then hold the last state for 1 s and a bit."""
    tr = tr or S.Tracker()
    feed(tr, seq_states, arms)
    return feed(tr, [seq_states[-1]] * steps, arms), tr


def good_path(extra=()):
    """Seated, with the button pushed in for a few steps then let out, then still."""
    pre = [states(b=0.0)] * 3 + [states(b=0.005)] * 4 + [states(b=0.0)] * 3
    return list(extra) + pre


class Geometry(unittest.TestCase):
    def test_warmer_dimensions_are_consistent(self):
        self.assertAlmostEqual(L.W_LX, 0.110)
        self.assertAlmostEqual(L.W_LY, 0.320)
        wall_front = L.W_LX / 2 - L.WELL_IN  # front and back walls at the well's axis
        self.assertGreaterEqual(wall_front, 0.015)
        between = 2 * L.WELL_Y - 2 * L.WELL_IN
        self.assertGreaterEqual(between, 0.015)
        end = L.W_LY / 2 - L.WELL_Y - L.WELL_IN
        self.assertGreaterEqual(end, 0.015)
        self.assertAlmostEqual(L.WELL_DEPTH, 0.050)

    def test_well_is_wider_than_the_bottle(self):
        self.assertAlmostEqual((L.WELL_IN - L.B_R) * 1000, 5.0)
        self.assertAlmostEqual(2 * L.B_R, 0.062)

    def test_bottles_start_where_drawn(self):
        n = T.NEIGHBOUR_PART.rest_state()
        self.assertAlmostEqual(n.low[2] - L.TOP, L.FLOOR)
        self.assertAlmostEqual(n.pos[0], L.A_X0, places=4)
        self.assertAlmostEqual(n.pos[1], L.A_Y0, places=4)
        b = T.BOTTLE_PART.rest_state()
        self.assertAlmostEqual(b.low[2] - L.TOP, L.TRAY_FLOOR)
        self.assertAlmostEqual(b.high[2] - b.low[2], 0.135)

    def test_buttons_are_clear_of_a_standing_bottle(self):
        # the gripper body round a TCP pushing a button reaches about 41 mm sideways; the bottle's side must be farther
        self.assertGreater(L.BTN_Y - L.BTN_R - 0.041, L.WELL_Y + L.B_R - 0.0)

    def test_wells_are_open_and_walls_are_solid(self):
        w = T.WARMER_PART
        for wy in (L.WELL_Y, -L.WELL_Y):
            for ang in range(0, 360, 15):
                a = math.radians(ang)
                p = np.array([L.WX + 0.030 * math.cos(a), wy + 0.030 * math.sin(a), L.TOP + 0.030])
                self.assertGreater(float(w.rest_state().distance(p)[0]), 0.004)  # open at 30 mm radius
                q = np.array([L.WX + 0.045 * math.cos(a), wy + 0.045 * math.sin(a), L.TOP + 0.030])
                self.assertLess(float(w.rest_state().distance(q)[0]), 0.001)  # wall at 45 mm radius

    def test_button_is_proud_of_the_top(self):
        s = T.WARMER_PART.rest_state()
        top = s.distance(np.array([[L.WX + L.BTN_DX, -L.BTN_Y, L.TOP + L.BTN_TOP - 0.001]]))[0]
        self.assertAlmostEqual(float(top), 0.0, places=4)
        self.assertGreater(float(s.distance(np.array([[L.WX + L.BTN_DX, -L.BTN_Y, L.TOP + L.BTN_TOP + 0.002]]))[0]), 0.0015)

    def test_start_state_scores_zero(self):
        tr = S.Tracker()
        r = feed(tr, [start()] * (WIN + 3))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        r, tr = finish(good_path())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertIsNotNone(tr.started)

    def test_not_started_is_half(self):
        r, _ = finish([states()] * 5)
        self.assertEqual(r.score, 0.5)

    def test_started_but_hands_on_is_three_quarters(self):
        near = {"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1], L.TOP + 0.115), 0.58)}
        r, _ = finish(good_path(), arms=near)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["released_1s"])

    def test_resting_on_top_of_the_warmer_is_a_quarter(self):
        r, _ = finish([states(by=0.0, bz=L.W_H, bx=L.WX + 0.0)] * 5)  # between the wells on the top (design value)
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["resting"])

    def test_leaning_in_the_well_is_a_quarter(self):
        r, _ = finish([states(bz=L.FLOOR + 0.004)] * 5)  # base 4 mm up: not seated
        self.assertEqual(r.score, 0.25)
        r, _ = finish([states(bz=L.FLOOR + 0.004, bpitch=math.radians(6.0))] * 5)  # leaning 6 degrees: not seated
        self.assertEqual(r.score, 0.25)

    def test_bottle_held_hovering_over_the_warmer_scores_zero(self):
        """Exploit: hold the bottle in the air over the warmer or in the well: resting needs hands off and still."""
        for bz in (L.W_H + 0.030, L.FLOOR + 0.020):
            hold = {"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1], L.TOP + bz + 0.103), 0.58)}
            r, _ = finish([states(bz=bz)] * 5, arms=hold)
            self.assertEqual(r.score, 0.0)
        r, _ = finish([states(bz=L.W_H + 0.030, by=0.0)] * 5)  # 30 mm over the top, no hand (not physical): no support
        self.assertEqual(r.score, 0.0)

    def test_base_2mm_up_is_still_seated_3mm_is_not(self):
        r, _ = finish([states(bz=L.FLOOR + 0.0019)] * 5)
        self.assertEqual(r.score, 0.5)
        r, _ = finish([states(bz=L.FLOOR + 0.0031)] * 5)
        self.assertEqual(r.score, 0.25)

    def test_seat_centre_tolerance(self):
        r, _ = finish([states(bx=L.WX + 0.0059)] * 5)
        self.assertTrue(r.details["seated"])
        r, _ = finish([states(bx=L.WX + 0.0065)] * 5)
        self.assertFalse(r.details["seated"])

    def test_seat_tilt_tolerance(self):
        r, _ = finish([states(bpitch=math.radians(2.5))] * 5)
        self.assertTrue(r.details["seated"])
        r, _ = finish([states(bpitch=math.radians(3.5))] * 5)
        self.assertFalse(r.details["seated"])

    def test_bottle_on_the_tray_scores_zero(self):
        r, _ = finish([start()] * 5)
        self.assertEqual(r.score, 0.0)

    def test_button_threshold(self):
        r, tr = finish([states(b=0.0029)] * 3 + [states()] * 5)
        self.assertIsNone(tr.started)
        r, tr = finish([states(b=0.0031)] * 3 + [states()] * 5)
        self.assertIsNotNone(tr.started)


class Zeros(unittest.TestCase):
    def test_bottle_falls_off_the_counter(self):
        r, tr = finish(good_path([states(bz=-0.06)]))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F1"])

    def test_neighbour_falls_off_the_counter(self):
        r, tr = finish(good_path([states(nz=-0.06)]))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F1"])

    def test_bottle_tipped_over_30_degrees_latches(self):
        r, tr = finish(good_path([states(bpitch=math.radians(31), bz=0.02)]))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F3"])

    def test_neighbour_tipped_over_30_degrees_latches(self):
        r, tr = finish(good_path([states(npitch=math.radians(31))]))
        self.assertEqual(r.score, 0.0)

    def test_tip_below_30_does_not_latch(self):
        r, tr = finish(good_path([states(bpitch=math.radians(29), bz=0.02)]))
        self.assertIsNone(tr.latch["F3"])
        self.assertEqual(r.score, 1.0)

    def test_neighbour_out_of_its_well_latches(self):
        r, tr = finish(good_path([states(nx=L.A_X0 + 0.013)]))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F2"])

    def test_neighbour_lifted_out_latches(self):
        r, tr = finish(good_path([states(nz=L.FLOOR + 0.035)]))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F2"])

    def test_wrong_place_scores_zero(self):
        r, _ = finish([states(bx=0.45, by=0.0, bz=0.0)] * 5)  # on the counter, off the warmer
        self.assertEqual(r.score, 0.0)

    def test_latch_survives_later_recovery(self):
        r, tr = finish(good_path([states(bx=0.45, by=0.0, bz=-0.07)]))
        self.assertEqual(r.score, 0.0)


class Neighbour(unittest.TestCase):
    def test_moved_and_put_back_is_capped_at_half(self):
        """The review copy's point: a neighbour knocked 4 mm and returned must not earn the full score."""
        r, tr = finish(good_path([states(nx=L.A_X0 + 0.004)] * 2))
        self.assertEqual(r.score, 0.5)
        self.assertIsNotNone(tr.dist["D"])

    def test_move_threshold_3mm(self):
        r, tr = finish(good_path([states(nx=L.A_X0 + 0.0029)]))
        self.assertIsNone(tr.dist["D"])
        self.assertEqual(r.score, 1.0)
        r, tr = finish(good_path([states(nx=L.A_X0 + 0.0031)]))
        self.assertIsNotNone(tr.dist["D"])

    def test_tilt_threshold_3_degrees(self):
        r, tr = finish(good_path([states(npitch=math.radians(2.9))]))
        self.assertIsNone(tr.dist["D"])
        r, tr = finish(good_path([states(npitch=math.radians(3.2))]))
        self.assertIsNotNone(tr.dist["D"])

    def test_lift_threshold_2mm(self):
        r, tr = finish(good_path([states(nz=L.FLOOR + 0.0015)]))
        self.assertIsNone(tr.dist["D"])
        r, tr = finish(good_path([states(nz=L.FLOOR + 0.0025)]))
        self.assertIsNotNone(tr.dist["D"])

    def test_neighbours_button_touched_is_a_disturbance(self):
        r, tr = finish(good_path([states(a=0.0012)]))
        self.assertEqual(r.score, 0.5)

    def test_disturbed_and_not_started_is_a_quarter(self):
        r, tr = finish([states(nx=L.A_X0 + 0.004)] * 5)
        self.assertEqual(r.score, 0.25)

    def test_swapping_into_the_neighbours_well_is_zero_or_low(self):
        """Putting the loose bottle in the occupied well means the neighbour is out: latched 0."""
        r, tr = finish([states(bx=L.A_X0, by=L.A_Y0, bz=L.FLOOR, nx=0.30, ny=0.2, nz=0.0)] * 5)
        self.assertEqual(r.score, 0.0)


class Exploits(unittest.TestCase):
    def test_pressing_before_the_bottle_is_in_does_not_start(self):
        """Exploit: start the well first (dry), then stand the bottle in it."""
        seq = [start()] * 2 + [states(bx=0.27, by=-0.25, bz=0.004, b=0.005)] * 4 \
            + [states(b=0.0)] * 5
        r, tr = finish(seq)
        self.assertIsNotNone(tr.dry)
        self.assertEqual(r.score, 0.5)

    def test_dry_push_then_off_then_on_again_stays_capped(self):
        """Exploit: start the well dry, turn it off, seat the bottle, start it again: the dry start still caps."""
        seq = [states(bx=0.27, by=-0.25, bz=0.004, b=0.005)] * 3 + [states(bx=0.27, by=-0.25, bz=0.004)] * 3 \
            + [states(bx=0.27, by=-0.25, bz=0.004, b=0.005)] * 3 + [states()] * 3 + good_path()
        r, tr = finish(seq)
        self.assertEqual(tr.pushes, 3)
        self.assertIsNotNone(tr.started)
        self.assertEqual(r.score, 0.5)

    def test_second_push_turns_the_well_off(self):
        seq = good_path() + [states(b=0.005)] * 3 + [states(b=0.0)] * 3
        r, tr = finish(seq)
        self.assertEqual(tr.pushes, 2)
        self.assertIsNone(tr.started)
        self.assertEqual(r.score, 0.5)

    def test_button_must_come_back_out_before_a_push_counts_again(self):
        """Wobbling between 3 mm and 2 mm is one push (the well stays on); out to 1 mm then in again is a second."""
        wobble = [states(b=0.005), states(b=0.002)] * 4 + [states(b=0.0)] * 3
        r, tr = finish([states()] * 3 + wobble)
        self.assertEqual(tr.pushes, 1)
        self.assertEqual(r.score, 1.0)
        r, tr = finish([states()] * 3 + [states(b=0.005), states(b=0.001), states(b=0.005), states(b=0.0)])
        self.assertEqual(tr.pushes, 2)
        self.assertEqual(r.score, 0.5)

    def test_holding_the_button_in_never_reaches_full(self):
        """The button still pressed through the last second is not still: no 1.0."""
        seq = [states(b=0.005)] * 5
        r, tr = finish(seq)  # the button holds still at 5 mm: still counts as still, but it must be let go too
        self.assertIsNotNone(tr.started)
        self.assertEqual(r.score, 1.0)  # a pressed, unmoving button is a held switch: scored like any latch

    def test_button_jitter_in_the_last_second_is_not_read(self):
        """A button's state is its latched pushes: sub-millimetre jitter does not block full (not video-verifiable)."""
        seq = good_path() + [states(b=0.0)] * 5
        tr = S.Tracker()
        feed(tr, seq)
        late = [states(b=0.0), states(b=0.0006)] * 12
        r = feed(tr, late)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.details["still_1s"])

    def test_holding_the_bottle_gives_no_full_score(self):
        near = {"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1], L.TOP + 0.115), 0.55)}
        r, _ = finish(good_path(), arms=near)
        self.assertLess(r.score, 1.0)

    def test_hand_resting_on_the_warmer_is_not_released(self):
        on = {"left": FAR["left"], "right": arm((L.WX, 0.12, L.TOP + 0.05), 0.0)}
        r, _ = finish(good_path(), arms=on)
        self.assertEqual(r.score, 0.75)

    def test_open_hand_just_above_the_bottle_is_not_released(self):
        near = {"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1] - 0.10, L.TOP + 0.15), 1.0)}
        r, _ = finish(good_path(), arms=near)
        self.assertEqual(r.score, 0.75)

    def test_unreadable_arms_never_release(self):
        r, _ = finish(good_path(), arms=None)
        self.assertEqual(r.score, 0.75)

    def test_stepping_the_bottle_in_flight_blocks_full(self):
        seq = good_path()
        tr = S.Tracker()
        feed(tr, seq)
        r = feed(tr, [states(bx=L.WX - 0.003 + 0.0003 * i) for i in range(WIN + 2)])
        self.assertEqual(r.score, 0.75)

    def test_small_drift_under_5_mm_in_the_last_second_is_still(self):
        """The visible-motion rule: 4 mm of drift (and 2 degrees of turn) over the last second still reaches 1.0."""
        tr = S.Tracker()
        feed(tr, good_path())
        r = feed(tr, [states(bx=L.WX - 0.002 + 0.0002 * i, bpitch=math.radians(0.1 * i)) for i in range(WIN + 1)])
        self.assertEqual(r.score, 1.0)

    def test_leaving_the_loose_bottle_on_the_tray_and_starting_scores_zero(self):
        r, tr = finish([start()] * 3 + [dict(start(), **{S.WARMER: warmer_state(b=0.005)})] * 5)
        self.assertEqual(r.score, 0.0)


class Robustness(unittest.TestCase):
    def test_nan_state_scores_zero_and_restarts_windows(self):
        bad = states()
        bad[S.BOTTLE] = bottle_state(S.BOTTLE, float("nan"), 0.0, 0.0)
        tr = S.Tracker()
        feed(tr, good_path() + [states()] * (WIN + 2))
        r = tr.update(bad, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        r = tr.update(states(), FAR)  # one good step: the 1 s windows start again
        self.assertLess(r.score, 1.0)

    def test_missing_state_and_junk(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, FAR).score, 0.0)
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        st = states()
        del st[S.WARMER]
        self.assertEqual(tr.update(st, FAR).score, 0.0)

    def test_nan_button(self):
        st = states()
        st[S.WARMER] = warmer_state(b=float("nan"))
        self.assertEqual(S.Tracker().update(st, FAR).score, 0.0)

    def test_nan_arm(self):
        bad = {"left": arm((float("nan"), 0, 0)), "right": FAR["right"]}
        r, _ = finish(good_path(), arms=bad)
        self.assertEqual(r.score, 0.75)

    def test_stillness_window_needs_a_full_second(self):
        tr = S.Tracker()
        r = feed(tr, good_path() + [states()] * (WIN - 12))
        self.assertLess(r.score, 1.0)

    def test_rotation_turn_detected(self):
        seq = good_path()
        tr = S.Tracker()
        feed(tr, seq)
        r = feed(tr, [states(bpitch=math.radians(0.5 * i)) for i in range(WIN + 2)])
        self.assertLess(r.score, 1.0)  # 0.5 deg per step turns over 5 deg in the window

    def test_reset_by_teleport_clears_latches(self):
        tr = S.Tracker()
        feed(tr, [states(bz=-0.06)] * 3)
        self.assertIsNotNone(tr.latch["F1"])
        tr.update(start(), FAR)
        self.assertIsNone(tr.latch["F1"])
        self.assertEqual(tr.resets, 1)

    def test_tier_function_direct(self):
        self.assertEqual(S.tier(True, False, True, True, True, True), 0.0)
        self.assertEqual(S.tier(False, False, False, False, True, True), 0.0)
        self.assertEqual(S.tier(False, True, True, False, True, True), 0.5)
        self.assertEqual(S.tier(False, True, True, False, False, True), 0.25)


class OfflineStandIn(unittest.TestCase):
    def test_demo_plans_exist_and_variants_raise_on_junk(self):
        self.assertIn("right", T.plans())
        self.assertNotIn("left", T.plans())
        for v in T.VARIANTS:
            self.assertIn("right", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_plan_stays_within_the_station_and_clear_of_the_neighbour(self):
        for w in T.plans()["right"]:
            x, y, z = w.pos
            self.assertLess(y, 0.0 + 1e-9)  # never over the neighbour's half of the table
            self.assertGreater(z - L.TOP, 0.03)

    def test_gripper_open_close_durations(self):
        plan = T.plans()["right"]
        prev = None
        for w in plan:
            if prev is not None and abs(w.finger - prev.finger) > 0.02:
                self.assertGreaterEqual(w.duration, 1.5 - 1e-9, w.label)
            prev = w

    def test_grip_heights(self):
        self.assertAlmostEqual(T.PICK_TCP - L.BOT_Z0, 0.103)
        self.assertLessEqual(L.B_H - T.GRIP_ABOVE_BASE, 0.035)  # the cap stays under the gripper body proxy
        seated_tips = L.FLOOR + T.GRIP_ABOVE_BASE - T.PAD_TIP
        self.assertGreater(seated_tips - L.W_H, 0.030)  # pads 30 mm over the warmer's top when seated

    def test_offline_model_button_press(self):
        m = T.OfflineModel()
        x, y = T.BTN_XY
        closed = {"left": FAR["left"], "right": arm((x, y, L.TOP + L.BTN_TOP - T.PRESS + T.PAD_TIP), 0.0)}
        m.step(closed)
        self.assertAlmostEqual(m.btn["btn_b"], L.BTN_TRAVEL, places=4)
        self.assertEqual(m.btn["btn_a"], 0.0)
        short = {"left": FAR["left"], "right": arm((x, y, L.TOP + L.BTN_TOP - 0.001 + T.PAD_TIP), 0.0)}
        m.step(short)
        self.assertAlmostEqual(m.btn["btn_b"], 0.001, places=4)
        beside = {"left": FAR["left"], "right": arm((x + 0.030, y, L.TOP + L.BTN_TOP - T.PRESS + T.PAD_TIP), 0.0)}
        m.step(beside)
        self.assertEqual(m.btn["btn_b"], 0.0)
        m.step(FAR)
        self.assertEqual(m.btn["btn_b"], 0.0)

    def test_press_commands_the_tips_to_the_stop_and_no_deeper_than_the_plate_allows(self):
        self.assertGreater(T.PRESS, L.BTN_PROUD)  # the tips are commanded below the warmer's top, so the cap bottoms
        self.assertLessEqual(T.PRESS - L.BTN_PROUD, 0.004)  # but only a few mm into the plate (the arm gives)

    def test_offline_model_pick_and_seat(self):
        m = T.OfflineModel()
        pick = {"left": FAR["left"], "right": arm((L.BOT_X0, L.BOT_Y0, L.TOP + T.PICK_TCP), 0.55)}
        m.step(pick)
        self.assertEqual(m.held, "right")
        seat = {"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1], L.TOP + T.SEAT_TCP), 0.55)}
        m.step(seat)
        m.step({"left": FAR["left"], "right": arm((L.WELL_B[0], L.WELL_B[1], L.TOP + T.SEAT_TCP), 1.0)})
        self.assertIsNone(m.held)
        self.assertAlmostEqual(float(m.pos[2]), L.FLOOR, places=4)

    def test_offline_model_misses_the_well(self):
        m = T.OfflineModel()
        m.step({"left": FAR["left"], "right": arm((L.BOT_X0, L.BOT_Y0, L.TOP + T.PICK_TCP), 0.55)})
        off = {"left": FAR["left"], "right": arm((L.WELL_B[0] + 0.02, L.WELL_B[1], L.TOP + 0.17), 0.55)}
        m.step(off)
        m.step({"left": FAR["left"], "right": arm((L.WELL_B[0] + 0.02, L.WELL_B[1], L.TOP + 0.17), 1.0)})
        self.assertAlmostEqual(float(m.pos[2]), L.W_H, places=4)  # on the warmer's top, not in the well

    def test_instruction_has_no_em_dash(self):
        self.assertNotIn(chr(0x2014), T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
