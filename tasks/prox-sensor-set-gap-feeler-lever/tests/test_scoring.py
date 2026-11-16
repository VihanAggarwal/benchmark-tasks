"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
Q_GOOD = L.GAP0 - L.GAP_SET  # the slide that leaves 1.00 mm
Q_SHUT = L.LEVER_MAX


def rig_state(qs=Q_GOOD, ql=Q_SHUT):
    return mk.state_from_poses(T.RIG_PART, (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                               joints={"sensor": qs, "lever": ql}, source="test")


def leaf_state(x=L.LEAF_X0, y=L.LEAF_Y0, z=L.LEAF_Z0, yaw=0.0, pitch=0.0, roll=0.0):
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.LEAF_PART, (x, y, L.TOP + z, *q), source="test")


def in_gap_leaf(dx=0.0):
    return leaf_state(L.X_DOG - L.LF_T / 2 + dx, L.GAP_AIM_TIP_Y, 0.0002)


def scene(qs=Q_GOOD, ql=Q_SHUT, leaf=None):
    return {S.RIG: rig_state(qs, ql), S.LEAF: leaf if leaf is not None else leaf_state()}


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


def witnessed(tr=None):
    """A tracker that has seen the lever come shut with the leaf standing in the gap (the face on it, 1.00 mm)."""
    tr = tr or S.Tracker()
    tr.update(scene(Q_GOOD, 0.0, in_gap_leaf()), FAR)  # open, the leaf in the gap
    tr.update(scene(Q_GOOD, Q_SHUT, in_gap_leaf()), FAR)  # shut on it
    assert tr.witnessed
    return tr


def final(states, arms=FAR, pinch=True):
    tr = witnessed() if pinch else S.Tracker()
    return run(states, arms=arms, tracker=tr)


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.S_LEN, 0.052)  # Turck datasheet "Dimensions 52 mm"
        self.assertAlmostEqual(L.S_MASS, 0.031)  # Turck product page: 31 g
        self.assertAlmostEqual(L.SN, 0.002)
        self.assertAlmostEqual(L.SN_SECURED, 0.00162)  # 0.81 x Sn
        self.assertAlmostEqual(L.CL_X1 - L.CL_X0, 0.022)  # the split block: 22 x 45 x 25 mm
        self.assertAlmostEqual(L.CL_Y, 0.045)
        self.assertAlmostEqual(L.CL_H, 0.025)
        self.assertAlmostEqual(L.LEVER_L, 0.0704)  # KIPP K0006 size 1: A = 70.4 mm
        self.assertAlmostEqual(L.CL_H - (L.AXIS_Z + 0.0061), L.AXIS_Z - 0.0061)  # as much metal over the bore as under
        self.assertAlmostEqual(L.LF_W, 0.0127)  # 1/2 in feeler stock
        self.assertAlmostEqual(L.LF_L, 0.2032, places=3)  # 8 in
        self.assertAlmostEqual(L.LF_T, 0.001)
        self.assertAlmostEqual(L.GAP0, 0.012)
        self.assertAlmostEqual(L.GAP_SET, 0.5 * L.SN)  # half of Sn

    def test_start_positions(self):
        self.assertAlmostEqual(L.S_FACE0, 0.306)
        self.assertAlmostEqual(L.X_DOG, 0.318)
        self.assertAlmostEqual(L.gap_of(0.0), 0.012)
        self.assertAlmostEqual(L.gap_of(Q_GOOD), 0.001)
        self.assertAlmostEqual(L.GAP_X, 0.312)
        self.assertAlmostEqual(L.LEVER_PIVOT[0], 0.291)
        self.assertAlmostEqual(L.S_REAR0, 0.254)
        self.assertAlmostEqual(L.LEAF_Y0 + L.LF_H_YC, -0.313)

    def test_sensor_sits_in_the_clamp(self):
        """The barrel passes through the bore at every slide position; the bore is on the sensor's axis."""
        for q in (0.0, 0.006, L.SLIDE_UP):
            rear, face = L.S_REAR0 + q, L.S_FACE0 + q
            self.assertLess(rear, L.CL_X0 + 0.016)  # the connector end, 10 mm or more outside the block at the start
            self.assertGreater(face, L.CL_X1)
        self.assertLess(L.S_REAR0 + L.CONN_L + L.SLIDE_UP, L.CL_X0 + 1e-9)  # the connector never reaches the block
        st = T.RIG_PART.rest_state()
        axis = st.to_world(np.array([[L.CL_X0 + 0.011, 0.0, L.AXIS_Z]]))
        self.assertGreater(float(st.distance(axis, bodies=("base",))[0]), 0.005)  # the bore is open on the axis

    def test_leaf_clears_the_work(self):
        """The blade at the gap's aim x: 8 mm off the face, 3 mm off the dog, 12 mm from the clamp; its tip is over the
        face's round and the dog's width; 3 mm each way is within the arms' trail."""
        self.assertAlmostEqual(L.GAP_AIM_X - L.LF_T / 2 - L.S_FACE0, 0.008, places=4)
        self.assertAlmostEqual(L.X_DOG - (L.GAP_AIM_X + L.LF_T / 2), 0.003, places=4)
        self.assertGreater(L.GAP_AIM_X - L.LF_T / 2 - L.CL_X1, 0.010)
        self.assertGreater(L.GAP_AIM_TIP_Y, L.S_R)
        self.assertLess(L.GAP_AIM_TIP_Y, L.DOG_Y / 2 + 1e-9)
        self.assertGreater(L.LF_W, 2 * L.S_R)  # the blade is as wide as the sensor's face

    def test_leaf_overlaps_the_face_in_height(self):
        """The blade (z 0 to 12.7 mm on the table) overlaps the face (z 6.5 to 18.5 mm) by 6.2 mm."""
        lo, hi = max(0.0, L.AXIS_Z - L.S_R), min(L.LF_W, L.AXIS_Z + L.S_R)
        self.assertAlmostEqual(hi - lo, 0.0062, places=4)

    def test_pinch_holds_the_leaf(self):
        """The pushed leaf's pinch (the 4 mm overshoot at about 600 N/m of arm stiffness, 2.4 N) holds it by friction on
        both faces (0.6) against its weight (0.29 N) with a factor of 5."""
        weight = L.LF_MASS * 9.81
        pinch = 0.004 * 600.0
        self.assertGreater(2 * 0.6 * pinch, 5 * weight)

    def test_locked_in_pinch_is_small_and_the_leaf_pulls_out(self):
        """After the lock the sensor's target is where it stood against the leaf: a 0.1 mm squeeze left over is
        0.2 N, so the leaf's drag is under 1 N, a fifth of the arm's 5 N."""
        residual = L.SLIDE_K * 0.0001
        drag = 2 * 0.6 * residual + 0.5 * L.LF_MASS * 9.81
        self.assertLess(residual, 0.5)
        self.assertLess(drag, 1.0)

    def test_lock_stiffness(self):
        """A 3 N pull on a locked sensor moves it 1.5 mm (a declared simplification)."""
        self.assertAlmostEqual(3.0 / L.SLIDE_K, 0.0015)

    def test_lever_swing_clears_the_left_hand(self):
        """The lever's every position stays on the -y side of the clamp's edge; the left hand's pad block is within
        y +-13.5 mm: 6 mm or more apart."""
        ys = []
        for q in np.linspace(0.0, L.LEVER_MAX, 25):
            ys.append(L.lever_tip_top(float(q))[0])
        self.assertLess(max(ys) + L.LEVER_T / 2, -0.019)
        self.assertGreater(-0.0135 - (max(ys) + L.LEVER_T / 2), 0.005)

    def test_lever_tip_positions(self):
        y0, z0 = L.lever_tip_top(0.0)
        y1, z1 = L.lever_tip_top(L.LEVER_MAX)
        self.assertAlmostEqual(z1, L.LEVER_PIVOT[2] + L.LEVER_T / 2, places=6)  # shut: horizontal, on the block's top
        self.assertAlmostEqual(y1, L.LEVER_PIVOT[1] - L.LEVER_L, places=6)
        self.assertGreater(z0, 0.075)  # open: 70 degrees up
        self.assertGreater(y0, y1)

    def test_hands_are_apart_at_the_pinch(self):
        """The leaf's 203 mm blade keeps the hand on its handle 19 cm or more from the hand at the sensor's rear."""
        right = np.array([L.GAP_AIM_X, T.HANDLE_Y_GAP])
        left = np.array([T.push_x(T.FULL), 0.0])
        self.assertGreater(float(np.linalg.norm(right - left)), 0.19)

    def test_tray_holds_the_leaf(self):
        r = S.read(scene())
        self.assertTrue(S.leaf_in_tray(r))
        self.assertGreater(L.TRAY_OUT[1], L.LF_L + L.LF_H_Y)  # the leaf (with its handle) is shorter than the tray
        self.assertLess(L.LF_H_X, L.TRAY_OUT[0] - 2 * L.TRAY_WALL)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = final(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_tier_table(self):
        cases = [
            ("full", Q_GOOD, Q_SHUT, leaf_state(), True, 1.0),
            ("leaf on the table, out", Q_GOOD, Q_SHUT, leaf_state(0.38, -0.10, 0.0002), True, 0.75),
            ("leaf still in the gap", Q_GOOD, Q_SHUT, in_gap_leaf(), True, 0.5),
            ("gap 1.4 mm (in range)", L.GAP0 - 0.0014, Q_SHUT, leaf_state(), True, 0.5),
            ("gap 0.6 mm (in range)", L.GAP0 - 0.0006, Q_SHUT, leaf_state(), True, 0.5),
            ("gap 2.5 mm (poor)", L.GAP0 - 0.0025, Q_SHUT, leaf_state(), True, 0.25),
            ("gap 0.4 mm (poor)", L.GAP0 - 0.0004, Q_SHUT, leaf_state(), True, 0.25),
            ("gap 3.9 mm (poor)", L.GAP0 - 0.0039, Q_SHUT, leaf_state(), True, 0.25),
            ("gap 4.5 mm", L.GAP0 - 0.0045, Q_SHUT, leaf_state(), True, 0.0),
            ("good gap, lever open", Q_GOOD, 0.0, leaf_state(), True, 0.25),
            ("bad gap, lever open", L.GAP0 - 0.0025, 0.0, leaf_state(), True, 0.0),
            ("lever 10 degrees short of shut", Q_GOOD, Q_SHUT - math.radians(10), leaf_state(), True, 0.25),
            ("good gap never pinched", Q_GOOD, Q_SHUT, leaf_state(), False, 0.5),
        ]
        for name, qs, ql, leaf, pinch, want in cases:
            res = final(scene(qs, ql, leaf), pinch=pinch)
            self.assertEqual(res.score, want, f"{name}: {res.score} {res.details}")

    def test_window_edges(self):
        for dg, want in ((0.00025, 1.0), (-0.00025, 1.0), (0.00032, 0.5), (-0.00032, 0.5)):
            res = final(scene(Q_GOOD - dg))
            self.assertEqual(res.score, want, dg)

    def test_start_scores_zero(self):
        st = {S.RIG: T.RIG_PART.rest_state(), S.LEAF: T.LEAF_PART.rest_state()}
        self.assertEqual(run(st).score, 0.0)

    def test_open_start_gap_is_not_scored(self):
        self.assertEqual(final(scene(0.0, Q_SHUT)).score, 0.0)  # 12 mm, lever shut

    def test_leaf_standing_in_the_gap_is_not_out(self):
        self.assertFalse(S.leaf_out(S.read(scene(leaf=in_gap_leaf()))))
        self.assertTrue(S.leaf_out(S.read(scene())))

    def test_leaf_lying_in_the_gap_is_not_out(self):
        lying = leaf_state(L.GAP_X, L.GAP_Y, 0.0125, pitch=math.pi / 2)
        self.assertFalse(S.leaf_out(S.read(scene(leaf=lying))))

    def test_leaf_near_the_gap_is_not_out(self):
        self.assertFalse(S.leaf_out(S.read(scene(leaf=leaf_state(0.30, -0.01)))))  # the blade's tip 16 mm from the line
        self.assertTrue(S.leaf_out(S.read(scene(leaf=leaf_state(0.38, -0.30)))))

    def test_leaf_on_the_tray_rim_or_flying_is_not_in_tray(self):
        up = leaf_state(L.LEAF_X0, L.LEAF_Y0, L.LEAF_Z0 + 0.030)  # held 30 mm up over the tray
        self.assertFalse(S.leaf_in_tray(S.read(scene(leaf=up))))
        beside = leaf_state(L.LEAF_X0 + 0.04, L.LEAF_Y0)  # handle half out over the wall
        self.assertFalse(S.leaf_in_tray(S.read(scene(leaf=beside))))
        self.assertEqual(final(scene(leaf=beside)).score, 0.75)

    def test_leaf_lying_in_the_tray_counts(self):
        lying = leaf_state(L.LEAF_X0, L.LEAF_Y0, L.TRAY_FLOOR + 0.0125, pitch=0.0, roll=math.pi / 2)
        self.assertTrue(S.leaf_in_tray(S.read(scene(leaf=lying))))


class Latches(unittest.TestCase):
    def test_crash_is_latched(self):
        tr = witnessed()
        tr.update(scene(L.GAP0 - 0.0001, Q_SHUT), FAR)  # shut with 0.1 mm left
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 3).score, 0.0)

    def test_crash_is_not_forgiven_by_a_later_good_scene(self):
        tr = S.Tracker()
        run(scene(L.GAP0, Q_SHUT), tracker=tr, steps=2)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_gap_under_30_hundredths_with_lever_open_is_not_a_crash(self):
        tr = S.Tracker()
        tr.update(scene(L.GAP0 - 0.0002, 0.0), FAR)
        self.assertFalse(tr.latched)

    def test_crash_edge(self):
        tr = S.Tracker()
        tr.update(scene(L.GAP0 - 0.0004, Q_SHUT), FAR)
        self.assertFalse(tr.latched)
        tr.update(scene(L.GAP0 - 0.00029, Q_SHUT), FAR)
        self.assertTrue(tr.latched)

    def test_lock_on_the_leaf_is_judged_as_the_lever_comes_shut(self):
        tr = S.Tracker()
        self.assertFalse(tr.witnessed)
        tr.update(scene(Q_GOOD, 0.0, in_gap_leaf()), FAR)  # pinched with the lever open: not yet
        self.assertFalse(tr.witnessed)
        tr.update(scene(Q_GOOD, Q_SHUT, in_gap_leaf()), FAR)  # shut on it
        self.assertTrue(tr.witnessed)
        tr.update(scene(), FAR)  # the leaf drawn out afterwards: kept
        self.assertTrue(tr.witnessed)

    def test_lock_needs_the_leaf_in_the_gap_as_the_lever_shuts(self):
        """Pinched, the leaf taken out, then the lever shut on an empty gap: not locked on the leaf."""
        tr = S.Tracker()
        tr.update(scene(Q_GOOD, 0.0, in_gap_leaf()), FAR)
        tr.update(scene(Q_GOOD, 0.0, leaf_state()), FAR)
        tr.update(scene(Q_GOOD, Q_SHUT, leaf_state()), FAR)
        self.assertFalse(tr.witnessed)
        self.assertEqual(run(scene(), tracker=tr).score, 0.5)

    def test_reshutting_without_the_leaf_drops_the_lock(self):
        """Locked on the leaf, re-opened past 20 degrees, shut again on an empty gap: judged on the last shutting."""
        tr = witnessed()
        tr.update(scene(Q_GOOD, Q_SHUT - math.radians(25), leaf_state()), FAR)
        self.assertTrue(tr.witnessed)
        tr.update(scene(Q_GOOD, Q_SHUT, leaf_state()), FAR)
        self.assertFalse(tr.witnessed)

    def test_a_small_bounce_of_the_lever_is_not_a_reshutting(self):
        tr = witnessed()
        tr.update(scene(Q_GOOD, Q_SHUT - math.radians(8), leaf_state()), FAR)
        tr.update(scene(Q_GOOD, Q_SHUT, leaf_state()), FAR)
        self.assertTrue(tr.witnessed)

    def _shut_on(self, leaf, qs=Q_GOOD):
        tr = S.Tracker()
        tr.update(scene(qs, 0.0, leaf), FAR)
        tr.update(scene(qs, Q_SHUT, leaf), FAR)
        return tr.witnessed

    def test_lock_on_an_upright_leaf_against_both_faces(self):
        self.assertTrue(self._shut_on(in_gap_leaf()))
        self.assertTrue(self._shut_on(in_gap_leaf(-0.0001), qs=Q_GOOD - 0.0002))  # 1.2 mm gap, 0.1 mm play each side

    def test_lock_needs_the_blade_square_and_upright(self):
        for yaw, pitch, roll in ((math.radians(15), 0.0, 0.0), (0.0, math.radians(15), 0.0),
                                 (0.0, 0.0, math.radians(15))):
            leaf = leaf_state(L.X_DOG - L.LF_T / 2, L.GAP_AIM_TIP_Y, 0.0002, yaw=yaw, pitch=pitch, roll=roll)
            self.assertFalse(S.blade_on_both_faces(S.read(scene(Q_GOOD, Q_SHUT, leaf))), (yaw, pitch, roll))
            self.assertFalse(self._shut_on(leaf))

    def test_lock_needs_the_blade_over_the_face(self):
        """Only the blade's tip over the face's edge (2 mm of the 12 mm round): not a gauge on the face."""
        leaf = leaf_state(L.X_DOG - L.LF_T / 2, -0.004, 0.0002)
        self.assertTrue(S.leaf_between_faces(S.read(scene(Q_GOOD, Q_SHUT, leaf))))
        self.assertFalse(self._shut_on(leaf))

    def test_lock_needs_both_faces_touched(self):
        """The gap reads 1.00 mm but the blade's pose is 0.5 mm off the dog (and 0.5 mm into the face): not on both."""
        r = S.read(scene(Q_GOOD, Q_SHUT, in_gap_leaf(-0.0005)))
        self.assertFalse(S.blade_on_both_faces(r))

    def test_lock_needs_the_leaf_between_the_faces(self):
        for leaf in (leaf_state(0.36, -0.10), leaf_state(L.X_DOG + 0.01, L.GAP_AIM_TIP_Y, 0.0002)):
            tr = S.Tracker()
            tr.update(scene(Q_GOOD, 0.0, leaf), FAR)
            tr.update(scene(Q_GOOD, Q_SHUT, leaf), FAR)  # gap right, leaf elsewhere or behind the dog
            self.assertFalse(tr.witnessed)

    def test_snapping_parts_back_cannot_erase_a_crash(self):
        """Exploit found by the review copy: crash, then snap the sensor and lever back to their marks in one step.
        The jump is noticed (the timed windows restart) but the crash latch stays: still 0."""
        tr = S.Tracker()
        run(scene(L.GAP0, Q_SHUT), tracker=tr, steps=2)
        self.assertTrue(tr.latched)
        tr.update(scene(0.0, 0.0, leaf_state()), FAR)  # parts snapped back onto their marks
        self.assertTrue(tr.latched)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_reset_hook_clears_the_latches(self):
        """A real trial reset (the framework's reset hook calls Tracker.reset) clears everything."""
        tr = S.Tracker()
        run(scene(L.GAP0, Q_SHUT), tracker=tr, steps=2)
        self.assertTrue(tr.latched)
        tr.reset()
        self.assertFalse(tr.latched)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_sensor(self):
        arms = {"left": arm((L.S_REAR0 - 0.006, 0.0, L.TOP + 0.023), 0.0), "right": FAR["right"]}
        self.assertEqual(final(scene(), arms=arms).score, 0.0)

    def test_hand_on_the_lever(self):
        y, z = L.lever_tip_top(L.LEVER_MAX)
        arms = {"left": FAR["left"], "right": arm((L.LEVER_PIVOT[0], y, L.TOP + z + 0.013), 0.0)}
        self.assertEqual(final(scene(), arms=arms).score, 0.0)

    def test_hand_on_the_leaf(self):
        arms = {"left": FAR["left"], "right": arm((L.LEAF_X0, L.LEAF_Y0 + L.LF_H_YC, L.TOP + 0.027), 0.3)}
        self.assertEqual(final(scene(), arms=arms).score, 0.0)

    def test_hand_clear(self):
        arms = {"left": arm((0.22, 0.30, L.TOP + 0.20)), "right": arm((0.14, -0.20, L.TOP + 0.15))}
        self.assertEqual(final(scene(), arms=arms).score, 1.0)

    def test_parked_hands_are_clear(self):
        arms = {"left": arm((T.PARK_L[0], T.PARK_L[1], L.TOP + T.PARK_L[2])),
                "right": arm((T.PARK_R[0], T.PARK_R[1], L.TOP + T.PARK_R[2]))}
        self.assertEqual(final(scene(), arms=arms).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(final(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = witnessed()
        near = {"left": arm((L.S_REAR0 - 0.006, 0.0, L.TOP + 0.023), 0.0), "right": FAR["right"]}
        run(scene(), arms=near, tracker=tr)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_slide_moving(self):
        tr = witnessed()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(Q_GOOD + 0.00004 * i), FAR)  # 0.04 mm per step: over 0.2 mm in a second
        self.assertEqual(res.score, 0.0)

    def test_slide_still_within_the_window(self):
        tr = witnessed()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(Q_GOOD + 0.00001 * (i % 2)), FAR)  # 0.01 mm jitter
        self.assertEqual(res.score, 1.0)

    def test_lever_moving(self):
        tr = witnessed()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(ql=Q_SHUT - 0.002 * (WIN + 2 - i)), FAR)  # still closing: 2 degrees in a second
        self.assertEqual(res.score, 0.0)

    def test_leaf_moving(self):
        tr = witnessed()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(leaf=leaf_state(L.LEAF_X0 + 0.0001 * i, L.LEAF_Y0)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_leaf_turning(self):
        tr = witnessed()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(leaf=leaf_state(roll=math.radians(0.02) * i)), FAR)
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({S.RIG: rig_state()}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_not_a_dict(self):
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)

    def test_nan_joint(self):
        bad = {S.RIG: rig_state(float("nan"), Q_SHUT), S.LEAF: leaf_state()}
        self.assertEqual(final(bad).score, 0.0)

    def test_nan_leaf(self):
        bad = mk.state_from_poses(T.LEAF_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(leaf=bad), FAR).score, 0.0)

    def test_inf_joint_and_missing_joint(self):
        self.assertEqual(S.Tracker().update(scene(float("inf"), Q_SHUT), FAR).score, 0.0)
        broken = rig_state()
        broken.joints.pop("lever")
        self.assertEqual(S.Tracker().update({S.RIG: broken, S.LEAF: leaf_state()}, FAR).score, 0.0)

    def test_blow_up_restarts_the_windows(self):
        tr = witnessed()
        run(scene(), tracker=tr, steps=WIN + 3)
        tr.update({S.RIG: rig_state(float("nan"), 0.0), S.LEAF: leaf_state()}, FAR)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 1).score, 1.0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            leaf = leaf_state(rng.uniform(0.15, 0.45), rng.uniform(-0.4, 0.1), rng.uniform(0.0, 0.1),
                              yaw=rng.uniform(-3, 3), pitch=rng.uniform(-1.6, 1.6), roll=rng.uniform(-1.6, 1.6))
            r = tr.update(scene(rng.uniform(-0.02, 0.012), rng.uniform(0.0, Q_SHUT), leaf), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Exploits(unittest.TestCase):
    def test_lever_shut_on_the_stop_at_the_dog(self):
        """Shut the clamp with the face pushed to the dog, then (impossible for the arms, but) back to a good gap."""
        tr = witnessed()
        tr.update(scene(L.GAP0, Q_SHUT), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_leaf_dropped_beside_the_work(self):
        """A good gap, the leaf thrown on the table: at most 0.75."""
        self.assertEqual(final(scene(leaf=leaf_state(0.40, -0.05, 0.0002))).score, 0.75)

    def test_good_gap_without_the_leaf_is_capped(self):
        """The slide parked at 1.00 mm by dead reckoning, the leaf never in the gap: no better than 0.5."""
        self.assertEqual(final(scene(), pinch=False).score, 0.5)

    def test_leaf_in_the_gap_but_far_from_the_face(self):
        """The lever shut with the leaf in the gap while the face was 5 mm back: not locked on the leaf."""
        tr = S.Tracker()
        tr.update(scene(L.GAP0 - 0.005, 0.0, in_gap_leaf()), FAR)
        tr.update(scene(L.GAP0 - 0.005, Q_SHUT, in_gap_leaf()), FAR)
        self.assertFalse(tr.witnessed)

    def test_loose_leaf_in_a_wide_gap(self):
        """The lever shut with a 1.4 mm gap and the 1.00 mm leaf loose in it (0.4 mm of play): not locked on it."""
        tr = S.Tracker()
        tr.update(scene(L.GAP0 - 0.0014, 0.0, in_gap_leaf()), FAR)
        tr.update(scene(L.GAP0 - 0.0014, Q_SHUT, in_gap_leaf()), FAR)
        self.assertFalse(tr.witnessed)

    def test_pinch_with_a_thicker_stack(self):
        """Two leaves (2 mm) or a tilted leaf (a 1.8 mm gap) under the lock is not 1.00 mm: not locked on the leaf."""
        tr = S.Tracker()
        tr.update(scene(L.GAP0 - 0.0018, 0.0, in_gap_leaf()), FAR)
        tr.update(scene(L.GAP0 - 0.0018, Q_SHUT, in_gap_leaf()), FAR)
        self.assertFalse(tr.witnessed)

    def test_hand_holding_the_lever_down_scores_nothing(self):
        y, z = L.lever_tip_top(L.LEVER_MAX)
        arms = {"left": FAR["left"], "right": arm((L.LEVER_PIVOT[0], y, L.TOP + z + 0.010), 0.0)}
        self.assertEqual(final(scene(), arms=arms).score, 0.0)

    def test_doing_nothing(self):
        st = {S.RIG: T.RIG_PART.rest_state(), S.LEAF: T.LEAF_PART.rest_state()}
        self.assertEqual(run(st, steps=3 * WIN).score, 0.0)


class DriveRule(unittest.TestCase):
    def test_open_follows_the_sensor(self):
        ts, tl, lock = T.drive_targets(0.004, 0.0, None)
        self.assertAlmostEqual(ts, 0.004)
        self.assertIsNone(lock)
        self.assertAlmostEqual(tl, 0.0)  # the open notch

    def test_shut_holds_where_it_was(self):
        ts, tl, lock = T.drive_targets(0.011, Q_SHUT, None)
        self.assertAlmostEqual(ts, 0.011)
        self.assertAlmostEqual(lock, 0.011)
        self.assertAlmostEqual(tl, Q_SHUT)
        ts, _, lock = T.drive_targets(0.0113, Q_SHUT, lock)  # pushed 0.3 mm further: the target stays
        self.assertAlmostEqual(ts, 0.011)

    def test_lever_within_five_degrees_locks(self):
        _, _, lock = T.drive_targets(0.007, Q_SHUT - math.radians(4.9), None)
        self.assertAlmostEqual(lock, 0.007)
        _, _, lock = T.drive_targets(0.007, Q_SHUT - math.radians(5.1), None)
        self.assertIsNone(lock)

    def test_reopening_unlocks(self):
        _, _, lock = T.drive_targets(0.011, Q_SHUT, None)
        _, _, lock = T.drive_targets(0.011, 0.5, lock)
        self.assertIsNone(lock)

    def test_target_stays_in_the_slide_range(self):
        ts, _, _ = T.drive_targets(0.5, 0.0, None)
        self.assertLessEqual(ts, L.SLIDE_UP)
        ts, _, _ = T.drive_targets(-0.5, 0.0, None)
        self.assertGreaterEqual(ts, L.SLIDE_LO)

    def test_lever_detent(self):
        _, tl, _ = T.drive_targets(0.0, 0.2, None)
        self.assertAlmostEqual(tl, 0.0)  # snaps open within 0.35 rad
        _, tl, _ = T.drive_targets(0.0, 0.7, None)
        self.assertAlmostEqual(tl, 0.7)  # between the notches it holds where it is
        _, tl, _ = T.drive_targets(0.0, Q_SHUT - 0.2, None)
        self.assertAlmostEqual(tl, Q_SHUT)

    def test_detent_torque_beats_gravity_and_loses_to_a_press(self):
        gravity = L.LEVER_MASS * 9.81 * (L.LEVER_L / 2)
        self.assertGreater(L.LEVER_SPRING * L.LEVER_CAPTURE, 2 * gravity)
        press = 2.5 * 0.035  # 2.5 N at 35 mm of the lever
        self.assertLess(L.LEVER_SPRING * L.LEVER_CAPTURE + L.LEVER_FRICTION, press / 2)

    def test_unreadable_state_gives_no_target(self):
        class Bad:
            joints = {}
        T._rig_rule({S.RIG: Bad()})  # must not raise
        T._rig_rule({})


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["rig", "tray", "leaf"])
        self.assertIsNone(T.RIG_PART.mass)
        self.assertIsNone(T.TRAY_PART.mass)
        self.assertAlmostEqual(T.LEAF_PART.mass, L.LF_MASS)
        self.assertEqual({j.name for j in T.RIG_PART.joints}, {"sensor", "lever"})
        self.assertAlmostEqual(T.SENSOR.mass, L.S_MASS)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("push", "pinch", "slide", "hold", "lift", "mm of", "step", "stand"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
                    self.assertGreaterEqual(w.pos[2], L.TOP + 0.012, (v, a, w.label))
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ + 60, run_limit / 2)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "short", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def _span(self, wps, prefix):
        t = 0.0
        for w in wps:
            if w.label.startswith(prefix):
                return t, t + w.duration
            t += w.duration
        raise KeyError(prefix)

    def test_demo_order(self):
        """Leaf lowered in the gap, then pushed and pinched, then the right hand lets go and clears, then the left backs
        off and parks, then the lever, then the leaf out."""
        plan = T.plans("")
        left, right = plan["left"], plan["right"]
        low = self._span(right, "settle in the gap")[1]
        push = self._span(left, "push the sensor")
        pinched = self._span(left, "settle: the leaf is pinched")[1]
        let_go = self._span(right, "let go of the leaf")
        clear = self._span(right, "close the fingers into a block")[1]
        back = self._span(left, "back off")
        parked = self._span(left, "aside")[1]
        trip = self._span(right, "round behind the sensor")
        lever = self._span(right, "in over the clamp")
        leaf_out = self._span(right, "over the leaf's handle at the gap")
        self.assertGreaterEqual(push[0], low - 1e-6)
        self.assertGreaterEqual(let_go[0], pinched - 1e-6)
        self.assertGreaterEqual(back[0], clear - 1e-6)
        self.assertGreaterEqual(trip[0], back[1] - 1e-6)  # the right sets off once the left has backed off the sensor
        self.assertLessEqual(trip[0], parked)
        self.assertGreaterEqual(lever[0], parked - 1e-6)  # and is over the clamp only when the left is parked
        self.assertGreaterEqual(leaf_out[0], self._span(right, "straight up")[1] - 1e-6)

    def test_demo_hands_are_apart_whenever_both_are_low(self):
        """Whenever both TCPs are under 5 cm, they are 17 cm or more apart (the arms' bodies are 20 cm wide)."""
        plan = T.plans("")
        ends = {}
        for a, wps in plan.items():
            t, pts = 0.0, []
            for w in wps:
                t += w.duration
                pts.append((t, np.array(w.pos)))
            ends[a] = pts

        def at(a, t):
            pts = ends[a]
            for te, p in pts:
                if te >= t:
                    return p
            return pts[-1][1]
        worst = 9.0
        for t in np.arange(0.5, 70.0, 0.5):
            lp, rp = at("left", t), at("right", t)
            if lp[2] - L.TOP < 0.05 and rp[2] - L.TOP < 0.05:
                worst = min(worst, float(np.linalg.norm(lp[:2] - rp[:2])))
        self.assertGreater(worst, 0.17)

    def test_push_geometry(self):
        """The pad block (face 5 mm ahead of the TCP) starts 12 mm behind the connector and ends 4 mm past the
        sensor's stop; its pad tips are on the connector's upper half."""
        self.assertAlmostEqual(T.PUSH_START_X + T.PAD_FACE, L.S_REAR0 - 0.012, places=6)
        self.assertAlmostEqual(T.push_x(T.FULL) + T.PAD_FACE - L.S_REAR0, 0.015, places=6)
        self.assertGreater(L.AXIS_Z + L.CONN_R, T.PUSH_TIPS)
        self.assertLess(L.AXIS_Z - L.CONN_R, T.PUSH_TIPS)
        self.assertAlmostEqual(T.FULL, 0.011)

    def test_grip_geometry(self):
        """The pad tips are 12 mm over the leaf's base: on the handle's upper 12 of 24 mm; the handle is 22 mm across
        (the grip's lower limit is 16 mm); commanded 14 mm across."""
        self.assertGreaterEqual(L.LF_H_X, 0.020)
        self.assertAlmostEqual(-2 * T.GRIP_R, 0.014)
        self.assertGreaterEqual(-2 * T.OPEN_R, L.LF_H_X + 0.030)
        self.assertLess(T.HANDLE_TIPS, L.LF_H_Z - 0.006)

    def test_lever_press_waypoints_follow_the_face(self):
        """The block's outboard bottom corner is LEVER_PUSH inside the lever's inboard face, 45 mm up from the pivot, at
        each step: the block pushes the face, so a lagging lever is pushed harder, never run ahead of. The commanded
        y is LEVER_DY further +y (the right arm's measured shortfall)."""
        hw = 0.0135
        for e in T.LEVER_STEPS_DEG:
            y, z = T.lever_tip_target(e)
            fy, fz = L.lever_face_point(L.LEVER_OPEN - math.radians(e), T.LEVER_CONTACT_S)
            ny, nz = math.sin(math.radians(e)), math.cos(math.radians(e))
            self.assertAlmostEqual(y - hw, fy - T.LEVER_PUSH * ny, places=6)
            self.assertAlmostEqual(z - T.PAD_TIP, fz - T.LEVER_PUSH * nz, places=6)
        self.assertEqual(T.LEVER_STEPS_DEG[-1], 0.0)

    def test_lever_block_never_over_the_clamp_low(self):
        """Wherever the block's inboard part is over the clamp block (y > -22.5 mm) its bottom is 30 mm above it."""
        for e in range(0, 71):
            y, z = T.lever_tip_target(float(e))
            if y + 0.0135 > -L.CL_Y / 2:
                self.assertGreater(z - T.PAD_TIP, L.CL_H + 0.030, e)

    def test_reach(self):
        """Every waypoint is 0.10 to 0.50 m in front of its own base (the lead's measured reach)."""
        for a, wps in T.plans("").items():
            for w in wps:
                self.assertGreaterEqual(w.pos[0], 0.10, (a, w.label))
                self.assertLessEqual(w.pos[0], 0.50, (a, w.label))


class OfflineModel(unittest.TestCase):
    def feed(self, m, left=None, right=None):
        a = {"left": left or FAR["left"], "right": right or FAR["right"]}
        m.step(a)

    def test_lever_pressed_down_by_a_block(self):
        """The right arm lands LEVER_DY short of its command (toward -y), so the model is fed the commanded y."""
        m = T.OfflineModel()
        for e in T.LEVER_STEPS_DEG:
            y, z = T.lever_tip_target(e)
            self.feed(m, right=arm((L.LEVER_PIVOT[0], y + T.LEVER_DY, L.TOP + z), 0.0))
        self.assertGreaterEqual(m.q_l, L.LEVER_MAX - L.LEVER_CLOSED_BAND)
        self.assertAlmostEqual(m.lock, 0.0)

    def test_lever_not_pressed_if_the_trail_is_forgotten(self):
        """Commanded without the shortfall, the block lands 31 mm too far out: past the face, the lever stays open."""
        m = T.OfflineModel()
        for e in T.LEVER_STEPS_DEG:
            y, z = T.lever_tip_target(e)
            self.feed(m, right=arm((L.LEVER_PIVOT[0], y, L.TOP + z), 0.0))
        self.assertLess(m.q_l, L.LEVER_MAX - L.LEVER_CLOSED_BAND)

    def test_open_gripper_does_not_press_the_lever(self):
        m = T.OfflineModel()
        y, z = T.lever_tip_target(70.0)
        self.feed(m, right=arm((L.LEVER_PIVOT[0], y + T.LEVER_DY, L.TOP + z), 1.0))
        self.assertEqual(m.q_l, 0.0)

    def test_block_above_or_inboard_of_the_face_does_not_press(self):
        m = T.OfflineModel()
        y, z = T.lever_tip_target(70.0, inboard=T.LEVER_CLEAR)
        self.feed(m, right=arm((L.LEVER_PIVOT[0], y + T.LEVER_DY, L.TOP + z), 0.0))
        y, z = T.lever_tip_target(70.0)
        self.feed(m, right=arm((L.LEVER_PIVOT[0], y + T.LEVER_DY, L.TOP + z + 0.060), 0.0))
        self.assertEqual(m.q_l, 0.0)

    def test_block_that_ran_ahead_does_not_press(self):
        """The old failure: a block already out past the lever's face (outboard of it) does not move the lever."""
        m = T.OfflineModel()
        y, z = T.lever_tip_target(70.0)
        self.feed(m, right=arm((L.LEVER_PIVOT[0], y + T.LEVER_DY - 0.030, L.TOP + z - 0.010), 0.0))
        self.assertEqual(m.q_l, 0.0)

    def test_sensor_pushed_then_held_after_the_lock(self):
        m = T.OfflineModel()
        self.push(m, 0.006)
        self.assertAlmostEqual(m.q_s, 0.006 + T.OVERSHOOT, places=6)
        m.q_l = L.LEVER_MAX
        self.feed(m)
        self.push(m, 0.011)
        self.assertAlmostEqual(m.q_s, 0.006 + T.OVERSHOOT, places=6)  # locked

    def push(self, m, stroke):
        """The left block pushed to ``stroke`` in 1 mm steps from where it came down."""
        x = T.PUSH_START_X
        while x < T.push_x(stroke) - 1e-9:
            x = min(x + 0.001, T.push_x(stroke))
            self.feed(m, left=arm((x, 0.0, L.TOP + T.PUSH_TCP), 0.0))

    def test_sensor_stops_at_the_dog(self):
        m = T.OfflineModel()
        self.push(m, 0.0115)
        self.assertAlmostEqual(m.q_s, L.SLIDE_UP, places=6)

    def test_sensor_pushes_the_leaf_against_the_dog(self):
        m = T.OfflineModel()
        m.leaf = np.array([L.GAP_AIM_X, L.GAP_AIM_TIP_Y, 0.0002])
        self.push(m, T.FULL)
        self.assertAlmostEqual(m.q_s, T.FULL, places=6)
        self.assertAlmostEqual(m.leaf[0], L.X_DOG - L.LF_T / 2, places=6)
        self.assertAlmostEqual(L.gap_of(m.q_s), L.LF_T, places=6)

    def test_open_hand_does_not_push_the_sensor(self):
        m = T.OfflineModel()
        x = T.PUSH_START_X
        for _ in range(15):
            x += 0.001
            self.feed(m, left=arm((x, 0.0, L.TOP + T.PUSH_TCP), 1.0))
        self.assertEqual(m.q_s, 0.0)

    def test_leaf_carried_and_set_down_in_the_tray(self):
        m = T.OfflineModel()
        hy = L.LEAF_Y0 + L.LF_H_YC
        grip = arm((L.LEAF_X0, hy, L.TOP + T.GRIP_TRAY), 0.2)
        self.feed(m, right=grip)
        for i in range(1, 11):  # carried 1 mm per step
            self.feed(m, right=arm((L.LEAF_X0 + 0.001 * i, hy + 0.002 * i, L.TOP + T.GRIP_TRAY + 0.003 * i), 0.2))
        self.assertAlmostEqual(m.leaf[0], L.LEAF_X0 + 0.01, places=6)
        self.assertAlmostEqual(m.leaf[1], L.LEAF_Y0 + 0.02, places=6)
        self.assertAlmostEqual(m.leaf[2], L.LEAF_Z0 + 0.03, places=6)
        self.feed(m, right=arm((L.LEAF_X0 + 0.01, hy + 0.02, L.TOP + T.GRIP_TRAY + 0.03), 1.0))  # opened
        self.assertAlmostEqual(m.leaf[2], L.LEAF_Z0, places=6)  # stands on the tray's floor
        self.assertAlmostEqual(m.leaf[0], L.LEAF_X0 + 0.01, places=6)

    def test_high_gripper_does_not_pick_up_the_leaf(self):
        m = T.OfflineModel()
        hy = L.LEAF_Y0 + L.LF_H_YC
        self.feed(m, right=arm((L.LEAF_X0, hy, L.TOP + 0.10), 0.2))
        self.assertEqual(len(m.grip), 0)

    def test_states_at_rest_match_the_spawn(self):
        st = T.OfflineModel().states()
        r0 = S.read({S.RIG: T.RIG_PART.rest_state(), S.LEAF: T.LEAF_PART.rest_state()})
        r1 = S.read(st)
        self.assertAlmostEqual(r1.q_sensor, r0.q_sensor)
        self.assertTrue(np.allclose(r1.leaf_pos, r0.leaf_pos, atol=1e-9))

    def test_full_demo_through_the_model_scores_one(self):
        """The demo's waypoints fed to the model as TCP snapshots: success, the same path the checker takes."""
        plan = T.plans("")
        tr = S.Tracker()
        m = T.OfflineModel()
        total = max(sum(w.duration for w in wps) for wps in plan.values())

        def pose(wps, t):
            t0 = 0.0
            prev = None
            for w in wps:
                if t <= t0 + w.duration:
                    f = (t - t0) / w.duration
                    f = f * f * (3 - 2 * f)
                    a = np.array(w.pos) if prev is None else np.array(prev.pos)
                    p = a + (np.array(w.pos) - a) * f
                    fa = w.finger if prev is None else prev.finger
                    return p, fa + (w.finger - fa) * f
                t0 += w.duration
                prev = w
            return np.array(wps[-1].pos), wps[-1].finger
        res = None
        for i in range(int((total + 3.0) * HZ)):
            t = i / HZ
            snap = {}
            for a in ("left", "right"):
                p, f = pose(plan[a], t)
                snap[a] = arm(p, float(np.clip(f / -0.04695, 0.0, 1.0)))
            m.step(snap)
            res = tr.update(m.states(), snap)
        self.assertEqual(res.score, 1.0, res.details)

    def test_variants_through_the_model(self):
        want = {"short": 0.25, "crash": 0.0, "leaf_stays": 0.5, "leaf_table": 0.75}
        for v, score in want.items():
            plan = T.plans(v)
            tr, m = S.Tracker(), T.OfflineModel()
            total = max(sum(w.duration for w in wps) for wps in plan.values())

            def pose(wps, t):
                t0, prev = 0.0, None
                for w in wps:
                    if t <= t0 + w.duration:
                        f = (t - t0) / w.duration
                        f = f * f * (3 - 2 * f)
                        a = np.array(w.pos) if prev is None else np.array(prev.pos)
                        fa = w.finger if prev is None else prev.finger
                        return a + (np.array(w.pos) - a) * f, fa + (w.finger - fa) * f
                    t0 += w.duration
                    prev = w
                return np.array(wps[-1].pos), wps[-1].finger
            res = None
            for i in range(int((total + 3.0) * HZ)):
                snap = {}
                for a in ("left", "right"):
                    p, f = pose(plan[a], i / HZ)
                    snap[a] = arm(p, float(np.clip(f / -0.04695, 0.0, 1.0)))
                m.step(snap)
                res = tr.update(m.states(), snap)
            self.assertEqual(res.score, score, (v, res.details))


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("BI2-M12-AP6X-H1141", "52 mm long", "31 g", "1.62 mm", "12.5 mm over the table",
                      "K0006.1501406X20", "22 mm along x", "45 mm across", "25 mm tall", "x 280 to 302 mm", "x 306 mm",
                      "24 x 24 mm", "40 x 40 x 34 mm", "x 318 mm", "12.0 mm", "1.00 mm", "A 70.4 mm", "12.2 mm bore",
                      "x 291 mm", "70 degrees", "12.7 mm x 203 mm", "25 mm into", "22 mm (x) x 40 mm (y) x 24 mm",
                      "30 g", "60 x 270 mm", "x 200 to 260 mm", "y -380 to -110 mm", "y -115 mm", "x 230 mm",
                      "y -313 mm", "20 cm"):
            self.assertIn(token, setup)
        for token in ("5 degrees", "0.25 mm", "0.50 to 1.62 mm", "0.30 to 4.0 mm", "20 degrees", "Locked on the leaf",
                      "30 mm", "60 x 270 mm", "10 mm", "0.2 mm", "1 degree", "1 mm and 0.3 degrees",
                      "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "1 s", "0.75 mm leaf", "1.25 mm leaf", "3 N pull"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "2400 steps (120 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_rules_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("declared", "2 N per mm", "estimates", "jam nuts", "carriage is fixed", "our own part",
                      "clamping force"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Industrial machinery mechanics", "routinely", "538,300", "BLS"):
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
        bad = re.compile("—|–|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f.endswith(".py") else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
