"""Offline tests for layout.py, parts.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
Q0 = (0.0, 0.0, 0.0, 1.0)
BIN_ORIGIN = (L.BIN_X, L.BIN_Y, L.TOP)
IN_BIN = (L.BIN_X, L.BIN_Y, L.TOP + L.FLOOR_TOP + L.CAN_H / 2)  # standing on the bucket floor
CAN_START = (L.CAN_X, L.CAN_Y, L.TOP + L.CAN_H / 2)
LID_TOP = L.HINGE_Z + L.LID_T + L.CAP_H


def quat_about(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return (*(a * math.sin(h)), math.cos(h))


def bin_state(lever=0.0, lid=0.0, pos=BIN_ORIGIN, quat=Q0):
    return mk.state_from_poses(T.BIN_PART, (*pos, *quat), joints={"pedal": lever, "lid": lid}, source="test")


def rod_state(rise=0.0, bin_pos=BIN_ORIGIN, quat=Q0):
    r = S.rotation(quat)
    o = np.asarray(bin_pos, dtype=float) + r @ np.array([L.ROD_X, 0.0, L.ROD_BOT + rise])
    return mk.state_from_poses(T.ROD_PART, (*o, *quat), source="test")


def can_state(pos=IN_BIN, quat=Q0):
    return mk.state_from_poses(T.CAN_PART, (*pos, *quat), source="test")


def scene(can=IN_BIN, lever=0.0, lid=0.0, rise=None, bin_pos=BIN_ORIGIN, bin_quat=Q0, can_quat=Q0):
    """Defaults: the can on the bucket floor, the lid shut, the pedal up (the 1.0 end state)."""
    rise = L.rod_from_lever(lever) if rise is None else rise
    return {S.BIN: bin_state(lever, lid, bin_pos, bin_quat), S.ROD: rod_state(rise, bin_pos, bin_quat),
            S.CAN: can_state(can, can_quat)}


def pedal_scene(lever, can=IN_BIN):
    """The pedal at ``lever`` with the rod and the lid where the linkage puts them."""
    rise = L.rod_from_lever(lever)
    return scene(can=can, lever=lever, lid=L.lid_from_rod(rise), rise=rise)


def start_scene():
    return {S.BIN: T.BIN_PART.rest_state(), S.ROD: T.ROD_PART.rest_state(), S.CAN: T.CAN_PART.rest_state()}


def arm(tcp, opening=1.0, approach=(0.0, 0.0, -1.0)):
    tcp = np.asarray(tcp, dtype=float)
    a = np.asarray(approach, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * a / np.linalg.norm(a), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def on_lid(gap):
    """Closed fingers straight down, pad tips ``gap`` over the lid's raised middle. rigstate's finger zone is a
    capsule round the last 38 mm of the fingers, 15.7 mm in radius when closed."""
    return {"left": arm((L.BIN_X, L.BIN_Y, L.TOP + LID_TOP + gap + 0.013), 0.0), "right": FAR["right"]}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def can_lid_gap(lid_deg, drop=T.DROP):
    """The closest the can comes to the lid while it is lowered straight down over ``drop`` from the carry height."""
    st = bin_state(0.0, math.radians(lid_deg))
    top = L.TOP + T.CARRY_Z - (T.GRIP_Z - L.CAN_H / 2)
    best = math.inf
    for dz in np.linspace(0.0, -0.25, 80):
        pts = np.array([drop[0], drop[1], top + dz]) + S.CAN_PTS
        best = min(best, float(np.min(st.distance(pts, bodies=("lid",)))))
    return best


class Geometry(unittest.TestCase):
    def test_published_sizes(self):
        self.assertAlmostEqual(2 * L.OUT_R, 0.168)  # 168 mm across
        self.assertLess(abs(LID_TOP - 0.264), 0.001)  # 264 mm high
        self.assertAlmostEqual(L.BIN_MASS, 0.83)
        self.assertGreater(L.BODY_MASS, 0.5)
        lo, hi = T.BIN_PART.local_bounds()
        self.assertLess(abs((hi[0] - lo[0]) - 0.235), 0.002)  # 235 mm deep with the pedal and the hinge
        a = math.radians(L.LID_OPEN_DEG)  # the lid's front edge, fully open: 411 mm published
        front = L.HINGE_Z + (L.HINGE_X + L.LID_R) * math.sin(a) + L.LID_T * math.cos(a)
        self.assertLess(abs(front - 0.411), 0.005)

    def test_the_linkage_gives_67_degrees_at_the_pedals_stop(self):
        self.assertAlmostEqual(math.degrees(L.LEVER_UPPER), 13.82, places=1)
        self.assertAlmostEqual(math.degrees(L.lid_from_rod(L.rod_from_lever(L.LEVER_UPPER))), L.LID_OPEN_DEG, places=3)
        self.assertAlmostEqual(L.PEDAL_TRAVEL, 0.030, delta=0.001)  # 30 mm at the plate's middle
        self.assertEqual(L.lid_from_rod(0.0), 0.0)
        self.assertEqual(L.lid_from_rod(L.TOP_GAP), 0.0)  # the first 0.5 mm closes the gap under the pin
        prev = -1.0
        for lv in np.linspace(0.0, L.LEVER_UPPER, 30):
            lid = L.lid_from_rod(L.rod_from_lever(lv))
            self.assertGreaterEqual(lid, prev)
            prev = lid
        self.assertLess(L.LID_OPEN_DEG, L.LID_UPPER_DEG)  # the pedal's stop comes before the hinge's

    def test_the_rod_fits_its_slot_and_meets_nose_and_pin(self):
        self.assertLess(L.ROD_TX, L.SLOT_X1 - L.SLOT_X0)
        self.assertLess(L.ROD_TY, 2 * L.SLOT_HY)
        self.assertGreater(L.SLOT_X0, L.IN_R)
        self.assertAlmostEqual(L.ROD_BOT, L.PIVOT_Z + L.ROLLER_R)
        self.assertAlmostEqual(L.ROD_TOP + L.TOP_GAP, L.PIN_Z - L.PIN_R)
        self.assertAlmostEqual(L.ROD_LEN, 0.2139, places=4)
        half = L.ROD_TX / 2
        for deg in np.linspace(0.0, L.LID_OPEN_DEG, 15):  # the pin stays over the rod's flat top as the lid turns
            th = math.radians(deg)
            pin_x = L.HINGE_X - L.PIN_D * math.cos(th + L.PIN_PHI0)
            self.assertLess(abs(pin_x - L.ROD_X), half, deg)
        for lv in np.linspace(0.0, L.LEVER_UPPER, 15):  # the nose stays under the rod's flat foot
            nose_x = L.PIVOT_X + L.LEVER_REAR * math.cos(lv)
            self.assertLess(abs(nose_x - L.ROD_X), half, lv)

    def test_the_rod_is_a_separate_loose_part_and_clear_of_the_bin_at_rest(self):
        self.assertIn(T.ROD_PART, T.OBJECTS)
        self.assertNotIn("rod", T.BIN_PART.body_names)
        b, r = bin_state(), rod_state()
        x0, x1 = L.ROD_X - L.ROD_TX / 2, L.ROD_X + L.ROD_TX / 2
        for z in np.linspace(L.ROD_BOT + 0.001, L.ROD_TOP - 0.001, 25):  # the rod's faces clear the body's slot
            pts = np.array([[x0, 0.0, z], [x1, 0.0, z], [L.ROD_X, L.ROD_TY / 2, z], [L.ROD_X, -L.ROD_TY / 2, z]])
            d = b.distance(pts + np.asarray(BIN_ORIGIN), bodies=("base",))
            self.assertGreater(float(d.min()), 0.0004, z)
        self.assertGreater(float(r.low[2]), L.TOP)

    def test_hinge_signs(self):
        shut, open_ = bin_state(), bin_state(0.0, math.radians(30))
        front = (-L.OUT_R + 0.005, 0.0, L.HINGE_Z + 0.004)
        self.assertGreater(open_.link_to_world("lid", front)[2], shut.link_to_world("lid", front)[2] + 0.05)
        up, down = bin_state(), bin_state(L.LEVER_UPPER)
        mid = (L.PRESS_X, 0.0, L.PLATE_TOP)  # the plate's middle in the bin's frame, as drawn
        self.assertLess(down.link_to_world("pedal", mid)[2], up.link_to_world("pedal", mid)[2] - 0.025)
        self.assertAlmostEqual(L.plate_point(L.PRESS_X, L.LEVER_UPPER)[1], L.PLATE_TOP - L.PEDAL_TRAVEL)

    def test_pedal_force_estimate(self):
        start, stop = L.pedal_force_range()
        self.assertAlmostEqual(start, 4.8, delta=0.05)
        self.assertAlmostEqual(stop, 1.45, delta=0.05)
        self.assertLess(start, 10.0)  # the acceptance check's limit
        com = T.BIN_PART.center_of_mass("pedal")
        self.assertAlmostEqual(L.LEVER_MASS * L.G * (L.PIVOT_X - com[0]), L.LEVER_CG_TORQUE, delta=0.002)

    def test_the_lid_alone_closes_the_linkage(self):
        """No spring: shut, the lid's weight is on the rod's top only after the 0.5 mm gap, and its centre of mass is
        in front of the hinge at every angle up to the stop (it falls shut)."""
        for deg in (0.0, 30.0, L.LID_OPEN_DEG):
            self.assertGreater(T.closing_rate(math.radians(deg)), 0.05)
        com = T.BIN_PART.center_of_mass("lid")
        self.assertLess(com[0], L.HINGE_X - 0.05)

    def test_the_can_fits_and_clears_the_open_lid(self):
        self.assertLess(2 * L.CAN_R, 2 * L.IN_R - 0.06)
        self.assertLess(L.FLOOR_TOP + L.CAN_H, L.RIM - L.IN_DEPTH)
        self.assertGreater(can_lid_gap(L.LID_OPEN_DEG), 0.015)  # 24 mm at 67 degrees
        self.assertGreater(can_lid_gap(60.0), 0.0)
        self.assertEqual(can_lid_gap(33.0), 0.0)  # the half-pressed pedal: the can meets the lid

    def test_the_lever_clears_the_table_down_to_its_stop(self):
        """GPU 17 Nov (job 20261117T104005): the lever's front arm and riser (underside 17.5 mm over the table, 100 mm
        in front of the pivot) meet the table at 10.07 degrees, so the pedal stalls there and the lid opens only 47.8
        degrees, not 67. Expected to fail until the lever is reshaped; then drop the decorator."""
        low, _ = bin_state(L.LEVER_UPPER).link_bounds("pedal")
        self.assertGreater(float(low[2]) - L.TOP, 0.002)

    def test_the_left_hand_and_the_can_clear_a_lid_open_only_60_degrees(self):
        """Margin for a lid short of its 67 degrees: at the drop point, carry height, with the grip as wide as the GPU
        measured it round the can (opening 0.702), the finger zone stays 5 mm and the can 13 mm from a 60-degree lid
        (the zone reaches 1 mm at about 59 degrees)."""
        st = bin_state(0.0, math.radians(60.0))
        tcp = np.array([T.DROP[0], T.DROP[1], L.TOP + T.CARRY_Z])
        tips = tcp + T.PAD_TIP * T.tiltgrip.rotation(0.0, T.PITCH_L)[:, 2]
        zone = rigstate.zone_gap((tcp, tips, 0.702), lambda p: st.distance(p, bodies=("lid",)))
        self.assertGreater(zone, L.TOUCH + 0.003)
        can = tcp + np.array([0.0, 0.0, L.CAN_H / 2 - T.GRIP_Z])
        self.assertGreater(float(st.distance(can + S.CAN_PTS, bodies=("lid",)).min()), L.CAN_TOUCH + 0.010)
        self.assertLess(math.hypot(T.DROP[0] - L.BIN_X, T.DROP[1] - L.BIN_Y) + L.CAN_R, L.IN_R - 0.006)  # rim
        self.assertGreater(T.RELEASE_Z - 0.018, L.RIM + 0.020)  # the fingers' lowest corner over the rim at let-go

    def test_start_layout(self):
        st = start_scene()
        self.assertEqual(run(st).score, 0.0)
        r = S.read(st)
        self.assertFalse(S.can_in_bin(r))
        d = st[S.BIN].distance(S.can_points(r))
        self.assertGreater(float(d.min()), 0.05)  # the can stands well clear of the bin
        self.assertEqual(r.lid_deg, 0.0)
        self.assertAlmostEqual(r.rod_rise, 0.0)
        self.assertAlmostEqual(math.hypot(L.CAN_X - 0.0, L.CAN_Y - 0.305), 0.270, delta=0.01)  # by the left arm

    def test_grip_widths(self):
        self.assertLess(-T.CAN_GRIP * 2, 2 * L.CAN_R)  # squeezes the can
        self.assertGreater(-T.LET_GO * 2, T.HOLD_GAP)  # lets go of it
        self.assertLess(-T.LET_GO * 2, rigstate.FULL_GAP)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["in_bin"])

    def test_can_lying_on_its_side_in_the_bin(self):
        z = L.TOP + L.FLOOR_TOP + L.CAN_R
        self.assertEqual(run(scene(can=(L.BIN_X, L.BIN_Y, z), can_quat=quat_about((1, 0, 0), 90))).score, 1.0)

    def test_can_on_the_table_scores_zero(self):
        self.assertEqual(run(scene(can=CAN_START)).score, 0.0)

    def test_can_on_the_shut_lid_scores_zero(self):
        self.assertEqual(run(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + LID_TOP + L.CAN_H / 2))).score, 0.0)

    def test_can_sticking_out_of_the_top_is_not_in(self):
        hi = L.RIM - L.IN_DEPTH + 0.002  # its top 8 mm under the rim
        res = run(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + hi - L.CAN_H / 2)))
        self.assertFalse(res.details["in_bin"])
        ok = L.RIM - L.IN_DEPTH - 0.001
        self.assertTrue(run(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + ok - L.CAN_H / 2))).details["in_bin"])

    def test_can_under_the_floor_or_outside_the_opening_is_not_in(self):
        low = L.FLOOR_TOP - L.FLOOR_TOL - 0.002
        self.assertFalse(run(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + low + L.CAN_H / 2))).details["in_bin"])
        self.assertFalse(run(scene(can=(L.BIN_X + L.IN_R + 0.003, L.BIN_Y, IN_BIN[2]))).details["in_bin"])

    def test_lid_not_shut_is_half(self):
        res = run(scene(lid=math.radians(6.0), rise=L.pin_rise(math.radians(6.0)) + L.TOP_GAP))
        self.assertEqual(res.score, 0.5)
        res = run(scene(lid=math.radians(4.5), rise=L.pin_rise(math.radians(4.5)) + L.TOP_GAP))
        self.assertEqual(res.score, 1.0)

    def test_bin_moved_or_tilted_is_half(self):
        self.assertEqual(run(scene(bin_pos=(L.BIN_X + 0.035, L.BIN_Y, L.TOP),
                                   can=(L.BIN_X + 0.035, L.BIN_Y, IN_BIN[2]))).score, 0.5)
        self.assertEqual(run(scene(bin_pos=(L.BIN_X + 0.025, L.BIN_Y, L.TOP),
                                   can=(L.BIN_X + 0.025, L.BIN_Y, IN_BIN[2]))).score, 1.0)
        q = quat_about((0, 1, 0), 6.0)
        self.assertEqual(run(scene(bin_quat=q)).details["bin_kept"], False)

    def test_possible_scores(self):
        self.assertEqual({S.tier(*bits) for bits in np.ndindex(2, 2, 2, 2, 2, 2)}, {0.0, 0.25, 0.5, 1.0})

    def test_tier_function_table(self):
        self.assertEqual(S.tier(False, True, False, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, False, False, True, True), 0.5)
        self.assertEqual(S.tier(False, True, False, True, False, True), 0.5)
        self.assertEqual(S.tier(False, True, False, True, True, False), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, True), 0.25)
        self.assertEqual(S.tier(False, False, False, True, True, True), 0.0)
        self.assertEqual(S.tier(True, True, False, True, True, True), 0.0)


class Latches(unittest.TestCase):
    def test_fingers_on_the_lid_latch_pushed(self):
        tr = S.Tracker()
        tr.update(scene(can=CAN_START), on_lid(0.001))
        self.assertTrue(tr.pushed)
        self.assertEqual(run(scene(), tracker=tr).score, 0.25)

    def test_fingers_20_mm_over_the_lid_do_not(self):
        tr = S.Tracker()
        tr.update(scene(can=CAN_START), on_lid(0.015))
        self.assertTrue(tr.pushed)  # the capsule (15.7 mm) reaches within 1 mm
        tr = S.Tracker()
        tr.update(scene(can=CAN_START), on_lid(0.020))
        self.assertFalse(tr.pushed)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_fingers_on_the_pedal_are_fine(self):
        tip = np.array([L.BIN_X + L.PRESS_X, L.BIN_Y, L.TOP + L.PLATE_TOP - L.PEDAL_TRAVEL])
        arms = {"left": FAR["left"], "right": rigstate.Arm(tip + np.array([0, 0, 0.013]), tip, 0.0)}
        tr = S.Tracker()
        for _ in range(5):
            tr.update(pedal_scene(L.LEVER_UPPER, can=CAN_START), arms)
        self.assertFalse(tr.pushed)

    def test_lid_lifted_by_hand_latches_pushed(self):
        """The lid opened 30 degrees with the rod and the pedal down: something else lifted it."""
        tr = S.Tracker()
        for deg in np.linspace(0.0, 30.0, 10):
            tr.update(scene(can=CAN_START, lid=math.radians(deg), rise=0.0), FAR)
        self.assertTrue(tr.pushed)
        self.assertEqual(tr.why["pushed"], "lid lifted above the rod's angle")

    def test_pedal_opening_never_latches(self):
        tr = S.Tracker()
        for lv in np.linspace(0.0, L.LEVER_UPPER, 40):
            tr.update(pedal_scene(lv, can=CAN_START), FAR)
        for lv in np.linspace(L.LEVER_UPPER, 0.0, 40):
            tr.update(pedal_scene(lv, can=CAN_START), FAR)
        self.assertFalse(tr.pushed)

    def test_soft_close_lagging_the_pedal_is_not_pushed(self):
        """Pedal let go at once: the rod drops, the damped lid follows it down slowly (above the rod's angle but
        falling)."""
        tr = S.Tracker()
        tr.update(pedal_scene(L.LEVER_UPPER, can=CAN_START), FAR)
        for deg in np.linspace(L.LID_OPEN_DEG, 0.0, 30):
            tr.update(scene(can=CAN_START, lid=math.radians(deg), rise=0.0), FAR)
        self.assertFalse(tr.pushed)

    def test_can_touching_the_lid_is_half(self):
        tr = S.Tracker()
        tr.update(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + LID_TOP + L.CAN_H / 2 + 0.001)), FAR)
        self.assertFalse(tr.clean)
        self.assertEqual(run(scene(), tracker=tr).score, 0.5)

    def test_can_3_mm_over_the_lid_is_clean(self):
        tr = S.Tracker()
        tr.update(scene(can=(L.BIN_X, L.BIN_Y, L.TOP + LID_TOP + L.CAN_H / 2 + 0.003)), FAR)
        self.assertTrue(tr.clean)

    def test_bin_tipped_or_slid_latches_zero(self):
        tr = S.Tracker()
        tr.update(scene(bin_quat=quat_about((0, 1, 0), 31.0)), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)
        tr = S.Tracker()
        tr.update(scene(bin_pos=(L.BIN_X + 0.11, L.BIN_Y, L.TOP)), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_a_29_degree_rock_does_not_latch(self):
        tr = S.Tracker()
        tr.update(scene(bin_quat=quat_about((0, 1, 0), 29.0)), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)


class Stillness(unittest.TestCase):
    def test_needs_a_full_second(self):
        tr = S.Tracker()
        for i in range(WIN):
            self.assertNotEqual(tr.update(scene(), FAR).score, 1.0, i)
        self.assertEqual(tr.update(scene(), FAR).score, 1.0)

    def test_can_still_rolling_is_half(self):
        tr = S.Tracker()
        for i in range(WIN + 2):
            x = L.BIN_X + 0.0003 * i
            res = tr.update(scene(can=(x, L.BIN_Y, IN_BIN[2])), FAR)
        self.assertEqual(res.score, 0.5)

    def test_4_mm_of_motion_is_still(self):
        tr = S.Tracker()
        for i in range(WIN + 2):
            x = L.BIN_X + (0.004 if i % 2 else 0.0)
            res = tr.update(scene(can=(x, L.BIN_Y, IN_BIN[2])), FAR)
        self.assertEqual(res.score, 1.0)

    def test_lid_bouncing_in_the_last_second_is_half(self):
        tr = S.Tracker()
        for i in range(WIN + 2):
            deg = 8.0 if i == WIN // 2 else 0.0
            res = tr.update(scene(lid=math.radians(deg), rise=L.pin_rise(math.radians(deg)) + L.TOP_GAP), FAR)
        self.assertEqual(res.score, 0.5)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        tr = S.Tracker()
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        self.assertEqual(tr.update({S.BIN: bin_state()}, FAR).score, 0.0)
        st = scene()
        st[S.CAN] = can_state((float("nan"), 0.0, 1.0))
        self.assertEqual(tr.update(st, FAR).score, 0.0)
        st = scene()
        st[S.BIN] = bin_state(lid=float("nan"))
        self.assertEqual(tr.update(st, FAR).score, 0.0)
        self.assertFalse(tr.update(st, FAR).details["valid_state"])

    def test_a_bad_step_restarts_the_window(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        tr.update(None, FAR)
        self.assertNotEqual(tr.update(scene(), FAR).score, 1.0)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_unreadable_arms_never_hold_1(self):
        tr = S.Tracker()
        for _ in range(WIN + 3):
            res = tr.update(scene(), None)
        self.assertNotEqual(res.score, 1.0)
        self.assertFalse(tr.pushed)  # one bad read does not latch for the rest of the trial
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)
        bad = {"left": FAR["left"], "right": rigstate.Arm(np.array([np.nan, 0, 1.0]), np.array([0, 0, 1.0]), 0.5)}
        self.assertNotEqual(tr.update(scene(), bad).score, 1.0)

    def test_distance_failure_counts_as_touching(self):
        class Broken:
            def distance(self, *a, **k):
                raise RuntimeError("boom")
        tr = S.Tracker()
        self.assertTrue(tr._fingers_touch_lid({S.BIN: Broken()}, FAR))

    def test_reset_back_on_the_marks_clears_the_latches(self):
        tr = S.Tracker()
        tr.update(scene(can=CAN_START), on_lid(0.0))
        tr.update(scene(bin_quat=quat_about((0, 1, 0), 40.0)), FAR)
        tr.update(scene(), FAR)
        self.assertTrue(tr.pushed and tr.latched)
        tr.update(start_scene(), FAR)  # the can jumps back to its start: a reset
        self.assertFalse(tr.pushed or tr.latched)
        self.assertEqual(tr.resets, 1)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)

    def test_lid_opened_by_hand_and_the_can_put_in(self):
        tr = S.Tracker()
        for deg in np.linspace(0.0, 70.0, 12):
            tr.update(scene(can=CAN_START, lid=math.radians(deg), rise=0.0), on_lid(0.0))
        self.assertEqual(run(scene(), tracker=tr).score, 0.25)

    def test_can_balanced_on_the_rim_is_not_in(self):
        z = L.TOP + L.RIM + L.CAN_H / 2
        self.assertEqual(run(scene(can=(L.BIN_X + L.IN_R, L.BIN_Y, z))).score, 0.0)

    def test_can_dropped_beside_the_bin_is_zero(self):
        self.assertEqual(run(scene(can=(L.BIN_X, L.BIN_Y + 0.15, L.TOP + L.CAN_R),
                                   can_quat=quat_about((1, 0, 0), 90))).score, 0.0)

    def test_pedal_still_held_down_at_the_end_is_half(self):
        tr = S.Tracker()
        res = run(pedal_scene(L.LEVER_UPPER), tracker=tr)
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.details["shut_1s"])

    def test_bin_knocked_over_with_the_can_in_is_zero(self):
        q = quat_about((0, 1, 0), 90.0)
        self.assertEqual(run(scene(bin_quat=q)).score, 0.0)

    def test_bin_carried_to_the_can_scores_at_most_half(self):
        pos = (L.CAN_X, L.CAN_Y - 0.12, L.TOP)
        res = run(scene(bin_pos=pos, can=(pos[0], pos[1], IN_BIN[2])))
        self.assertEqual(res.score, 0.0)  # slid more than 100 mm: latched


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def press_arms(self, corner):
        w = T._tips_at(1.0, *T.PRESS_XY, corner, T.CLOSED, "test")
        tcp = np.array(w.pos if hasattr(w, "pos") else w[1], dtype=float)
        tips = tcp + T.PAD_TIP * T.A_R
        return {"left": FAR["left"], "right": rigstate.Arm(tcp, tips, 0.0)}

    def test_pressing_the_pedal_opens_the_lid_and_letting_go_closes_it(self):
        m = T.OfflineModel()
        arms = self.press_arms(L.PLATE_TOP - L.PEDAL_TRAVEL - T.PRESS_OVER)
        m.step(arms)
        self.assertAlmostEqual(math.degrees(m.lid), L.LID_OPEN_DEG, places=3)
        self.assertAlmostEqual(m.lever, L.LEVER_UPPER, places=6)
        for i in range(60):
            m.step(FAR)
        self.assertEqual(m.lid, 0.0)
        self.assertEqual(m.lever, 0.0)

    def test_soft_close_takes_about_a_second(self):
        m = T.OfflineModel()
        m.step(self.press_arms(L.PLATE_TOP - L.PEDAL_TRAVEL - T.PRESS_OVER))
        n = 0
        while m.lid > 0.0 and n < 200:
            m.step(FAR)
            n += 1
        self.assertGreater(n / HZ, 0.8)
        self.assertLess(n / HZ, 2.0)

    def test_open_fingers_do_not_press(self):
        m = T.OfflineModel()
        arms = self.press_arms(L.PLATE_TOP - L.PEDAL_TRAVEL - T.PRESS_OVER)
        arms["right"] = rigstate.Arm(arms["right"].tcp, arms["right"].tips, 1.0)
        m.step(arms)
        self.assertEqual(m.lid, 0.0)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertEqual(set(d), {"left", "right"})
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertEqual(set(T.demo()), {"left", "right"})
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

    def test_the_pedal_is_held_until_the_left_hand_is_back(self):
        p = T.plans()
        held = T._until(p["right"], "hold the pedal down")
        self.assertGreaterEqual(held, T._until(p["left"], "back and up"))
        self.assertGreater(T._until(p["left"], "let go") - 1.0, T._until(p["right"], "press the pedal"))

    def test_demo_length_is_within_half_the_run_limit(self):
        limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0]) if (
            HERE / "task.json").exists() else 800
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps, limit / 2)
        self.assertGreater(steps, 250)


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
        self.assertEqual(d["category"], "Societal relatability")
        text = d["setup"] + d["scoring"]
        for needle in ("Brabantia NewIcon 3 L", "264 mm", "168 mm", "0.83 kg", "248 mm", "144 mm", "50 mm",
                       "67 degrees", "30 mm", "4.8 N", "1.5 N", "66 mm", "115 mm", "14 g", "x 260 mm, y 240 mm",
                       "x 330 mm", "10 mm", "72 mm", "4 mm", "5 degrees", "within 1 mm of the lid", "30 degrees", "100 mm",
                       "1 s", "0.25", "0.5", "1.0"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "800")
        self.assertIn("0.10 N m s/rad", d["fidelity"])

    def test_numbers_in_the_text_match_layout(self):
        self.assertAlmostEqual(L.CAN_X, 0.260)
        self.assertAlmostEqual(L.CAN_Y, 0.240)
        self.assertAlmostEqual(L.BIN_X, 0.330)
        self.assertAlmostEqual(L.RIM, 0.248)
        self.assertAlmostEqual(2 * L.IN_R, 0.144)
        self.assertAlmostEqual(L.LID_DAMP, 0.10)
        self.assertEqual((L.IN_DEPTH, L.FLOOR_TOL, L.SHUT_DEG, L.KEPT_DIST, L.KEPT_TILT, L.SLID_DIST, L.TIP_DEG),
                         (0.010, 0.004, 5.0, 0.030, 5.0, 0.100, 30.0))
        self.assertEqual((L.TOUCH, L.CAN_TOUCH, L.FORCED_DEG), (0.001, 0.001, 10.0))

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
