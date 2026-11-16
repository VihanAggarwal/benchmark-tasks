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
LOCKED = L.LEV_Q_LOCK + L.LEV_PRELOAD
SLIP = math.radians(89.0)  # the demo's slip: five clicks
Q0 = (0.0, 0.0, 0.0, 1.0)


def mic_state(theta=L.THETA_C, phi=SLIP, lever=LOCKED, q=None):
    q = L.adv_of(min(theta, L.THETA_C)) if q is None else q
    return mk.state_from_poses(T.MIC_PART, (L.X_A, L.Y_A, L.TOP, *Q0),
                               joints={"spindle": q, "thimble": theta, "ratchet": phi, "lever": lever},
                               source="test")


def block_state(dx=-0.002, dy=0.0, dz=L.BL_Z / 2, quat=Q0):
    return mk.state_from_poses(T.BLOCK_PART, (L.X_A + dx, L.Y_A + dy, L.TOP + L.PL + dz, *quat), source="test")


def states(theta=L.THETA_C, phi=SLIP, lever=LOCKED, q=None, **blk):
    return {S.MIC: mic_state(theta, phi, lever, q), S.BLOCK: block_state(**blk)}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((T.PARK_L[0], T.PARK_L[1], L.TOP + T.PARK_L[2])),
       "right": arm((T.PARK_R[0], T.PARK_R[1], L.TOP + T.PARK_R[2]))}


def run(st, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(st, arms)
    return res


def play(phis, theta=L.THETA_C, lever=LOCKED, hold=WIN + 3, arms=FAR, **blk):
    """The faces touch the block with the knob at the first of ``phis``, which it then turns through with the lever
    off (clicks count only then); then the lever set to ``lever`` and held."""
    tr = S.Tracker()
    for phi in phis:
        tr.update(states(theta, phi, min(lever, 0.0), **blk), arms)
    return run(states(theta, phis[-1], lever, **blk), arms, hold, tr), tr


def quat_x(angle):
    return (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))


def quat_y(angle):
    return (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))


def quat_z(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


class Geometry(unittest.TestCase):
    def test_the_screw(self):
        """0.5 mm per turn: 0.30 mm to touch the 20.000 mm block from 20.30 mm is 216 degrees (0.6 of a turn)."""
        self.assertAlmostEqual(math.degrees(L.THETA_C), 216.0, places=6)
        self.assertAlmostEqual(L.adv_of(2 * math.pi), L.PITCH)
        self.assertAlmostEqual(L.gap_of(L.adv_of(L.THETA_C)), L.T_BLOCK)
        self.assertAlmostEqual(L.theta_of_gap(L.T_BLOCK), L.THETA_C)
        self.assertLess(L.THETA_C + math.radians(L.SQUEEZE_DEG), L.THIM_HI)  # the hinge reaches the squeeze mark

    def test_joint_limits_hold_the_demo(self):
        """One turn at most per hinge (mechkit): the thimble to 216, the knob to 89 more, the lever to its stop."""
        for j in T.MIC_PART.joints:
            if j.kind == "revolute":
                self.assertLessEqual(max(-j.lower, j.upper), 2 * math.pi, j.name)
        self.assertGreater(L.THIM_HI, L.THETA_C + math.radians(30.0))
        self.assertGreater(L.RAT_HI, SLIP + math.radians(60.0))
        self.assertGreater(L.LEV_HI, L.LEV_Q_LOCK)
        self.assertLessEqual(L.SP_LO, L.adv_of(L.THIM_LO) - L.SP_SAG)
        self.assertGreaterEqual(L.SP_HI, L.adv_of(L.THETA_C))

    def test_parts_spawn_at_rest(self):
        st = start()
        self.assertEqual(st[S.MIC].joints, {"spindle": 0.0, "thimble": 0.0, "ratchet": 0.0, "lever": 0.0})
        r = S.read(st)
        self.assertAlmostEqual(S.gap(r), 0.0203)
        self.assertAlmostEqual(float(st[S.BLOCK].origin[2]) - L.TOP - L.PL, L.BL_Z / 2)
        self.assertAlmostEqual(float(st[S.BLOCK].origin[0]) - L.X_A, 0.055)
        self.assertFalse(S.between(r))  # 55 mm in front of the faces
        self.assertAlmostEqual(T.BLOCK_PART.mass, 0.0495)

    def test_the_block_fits_between_the_faces(self):
        """Faces as found 0.30 mm over the block's top; the rails leave 0.75 mm each side; the column 3 mm behind it."""
        m = T.MIC_PART.rest_state()
        top = np.array([[L.X_A, L.Y_A, L.TOP + L.PL + L.T_BLOCK]])
        self.assertAlmostEqual(float(m.distance(top, bodies=("spindle",))[0]), L.G0 - L.T_BLOCK, places=6)
        for sy in (-1, 1):
            side = np.array([[L.X_A, L.Y_A + sy * L.BL_Y / 2, L.TOP + L.PL + 0.001]])
            self.assertAlmostEqual(float(m.distance(side, bodies=("base",))[0]), 0.00075, places=6)
        self.assertAlmostEqual(L.RAIL_GAP, L.BL_Y + 0.0015)
        self.assertAlmostEqual(L.THROAT - L.BL_X / 2, 0.003)
        wall = np.array([[L.X_A - L.THROAT, L.Y_A, L.TOP + L.PL + 0.010]])
        self.assertAlmostEqual(float(m.distance(wall, bodies=("base",))[0]), 0.0, places=6)  # the column's front face
        self.assertAlmostEqual(L.THROAT, 0.0205)

    def test_the_anvil_is_flush_with_the_plate(self):
        m = T.MIC_PART.rest_state()
        on_plate = np.array([[L.X_A + 0.05, L.Y_A + 0.03, L.TOP + L.PL]])
        on_anvil = np.array([[L.X_A, L.Y_A, L.TOP + L.PL]])
        self.assertAlmostEqual(float(m.distance(on_plate, bodies=("base",))[0]), 0.0, places=9)
        self.assertAlmostEqual(float(m.distance(on_anvil, bodies=("base",))[0]), 0.0, places=9)
        above = np.array([[L.X_A, L.Y_A, L.TOP + L.PL + 0.0005]])
        self.assertAlmostEqual(float(m.distance(above, bodies=("base",))[0]), 0.0005, places=9)

    def test_between_window_is_each_face_wholly_on_the_block(self):
        self.assertAlmostEqual(L.BL_X / 2 - L.FACE_R, 0.014325, places=6)
        self.assertAlmostEqual(L.BETWEEN_DX, 0.0143)
        self.assertAlmostEqual(L.BL_Y / 2 - L.FACE_R, 0.001325, places=6)
        self.assertAlmostEqual(L.BETWEEN_DY, 0.0013)
        self.assertLess(0.00075, L.BETWEEN_DY)  # the rails keep it inside the window

    def test_grasp_widths(self):
        """The knob (18 mm) and the thimble (20 mm) are gripped across: over the 16 mm minimum, under 80 mm."""
        for w in (2 * L.RAT_R, 2 * L.THIM_R):
            self.assertGreaterEqual(w, 0.016)
            self.assertLessEqual(w, 0.080)
        self.assertAlmostEqual(-2 * T.GRIP_K, 0.010)  # 8 mm under the knob's width

    def test_pad_tips_clear_the_thimble_when_gripping_the_knob(self):
        self.assertGreaterEqual(T.KNOB_TIPS_Z - L.THIM_Z1, 0.0010)
        self.assertLessEqual(T.KNOB_TIPS_Z, L.RAT_Z1 - 0.010)  # at least 10 mm of the knob between the pads
        self.assertGreater(L.RAT_Z0 - L.THIM_Z1, 0.0)
        self.assertAlmostEqual(T.KNOB_TIPS_Z - L.PL, 0.056, places=6)
        self.assertTrue(T.KNOB_TIPS[0] <= 0.056 <= T.KNOB_TIPS[1])

    def test_the_thimble_variant_takes_the_thimble(self):
        self.assertTrue(T.THIM_TIPS[0] <= T.THIM_TIPS_Z - L.PL < T.THIM_TIPS[1])
        self.assertGreater(L.THIM_Z1, T.THIM_TIPS_Z)
        self.assertGreater(T.THIM_TIPS_Z - L.ARM_Z0 - L.PL, 0.010)  # clear of the frame's arm below

    def test_pusher_clears_everything_tall(self):
        """The closed pad block comes in 3 mm over the rails; the gripper's housing (70 mm over the TCP) and finger
        carriages (57 mm over it) clear the knob's top at the end of the push."""
        self.assertAlmostEqual(T.PUSH_TCP - T.PAD_TIP - L.PL - L.RAIL_H, 0.003)
        self.assertGreaterEqual(T.PUSH_TCP + 0.070 - L.RAT_Z1, 0.020)
        self.assertGreaterEqual(T.PUSH_TCP + 0.057 - L.RAT_Z1, 0.007)
        self.assertGreater(T.PUSH_TCP - T.PAD_TIP, L.PL + L.RAIL_H)

    def test_push_geometry(self):
        self.assertAlmostEqual(T.PUSH_START_X - T.PAD_FACE - T.BLOCK_FRONT0, 0.012)  # in the air, 12 mm in front
        self.assertLess(T.PUSH_START_X, 0.36)  # inside the right arm's reach at the plate's height
        end_front = L.X_A + T.BLOCK_END_U + L.BL_X / 2
        self.assertAlmostEqual(T.PUSH_END_X - T.PAD_FACE, end_front)
        self.assertGreater(T.BLOCK_END_U, -L.THROAT + L.BL_X / 2)  # 1 mm short of the column's face
        self.assertTrue(S.block_between((T.BLOCK_END_U, 0.0, L.BL_Z / 2), np.eye(3)))

    def test_lever_geometry(self):
        a_end = T.lever_angle_for(T.LEV_X1 + T.PAD_FACE - L.X_A)
        self.assertGreater(a_end, L.LEV_SNAP_ON + 0.1)  # the push reaches the cam's snap with room to spare
        a_snap = T.lever_angle_for(0.015)
        self.assertAlmostEqual(T.lever_angle_for(-0.0045), 0.0)
        self.assertGreater(a_snap, 0.5)
        # the pad's face (-x pad block, 16 x 27 mm) still meets the tip at the snap angle
        y_tip = T.lever_tip_y(L.LEV_SNAP_ON)
        self.assertLessEqual(abs(y_tip - T.LEV_Y), 0.010)
        # at the start the pad block is 3 mm or more off the lever's -x face and 17 mm off the thimble
        self.assertLessEqual(T.LEV_X0 + T.PAD_FACE - L.X_A, -L.LEV_W / 2 - 0.003)
        corner = math.hypot(T.LEV_X0 + 0.008 - L.X_A, T.LEV_Y - 0.0135 - L.Y_A)
        self.assertGreater(corner, L.THIM_R + 0.004)
        # and clear of the frame's column (y +-13 mm, insulating pads to +-15 mm)
        self.assertGreater(T.LEV_Y - 0.0135, L.COL_HY + L.PAD_T)
        self.assertGreaterEqual(T.LEV_TIPS_Z, L.LEV_Z0 - 0.001 - 0.0)
        self.assertLessEqual(T.LEV_TIPS_Z, L.LEV_Z1)

    def test_lever_force(self):
        flip = L.LEV_K * (L.LEV_SNAP_OFF + L.LEV_PRELOAD) / L.LEV_L
        self.assertLess(flip, L.LEV_TIP_FORCE)  # the cam's peak at the tip is under 0.5 N
        self.assertLess(L.LEV_K * (L.LEV_SNAP_ON + L.LEV_PRELOAD) / L.LEV_L, 1.0)
        self.assertLess(L.LEV_SNAP_ON, L.LOCK_Q)
        self.assertLess(L.LOCK_Q, L.LEV_Q_LOCK)

    def test_demo_clicks_have_margin(self):
        turn = math.degrees(T.KA0 - T.KA1) + math.degrees(T.KB0 - T.KB1)
        slip = turn - math.degrees(L.THETA_C)
        self.assertAlmostEqual(turn, 305.0)
        self.assertAlmostEqual(slip, 89.0)
        self.assertGreaterEqual(S.clicks_of(math.radians(slip)), L.CLICKS_FULL + 2)  # two clicks to spare
        self.assertLess(turn, 360.0 - 30.0)  # the knob's hinge limit
        self.assertGreater(math.degrees(T.KA0 - T.KA1), 90.0)

    def test_demo_yaws_stay_in_the_wrist_window(self):
        """Reach scan at the knob: the left arm reaches yaws -150 to +50 degrees with joint 6 well inside its limit."""
        for plan in T.plans("").values():
            prev = plan[0].yaw
            for w in plan:
                self.assertLessEqual(abs(w.yaw - prev), math.radians(100.0) + 1e-9, w.label)  # one-axis slerp
                prev = w.yaw
        for w in T.plans("")["left"]:
            self.assertTrue(math.radians(-150.0) <= w.yaw <= math.radians(50.0), w.label)

    def test_task_constants_are_sourced_or_estimated(self):
        self.assertAlmostEqual(L.BLOCK_MASS, 0.0495)
        self.assertAlmostEqual(L.BL_X * L.BL_Y * 1000 ** 2, 315.0)  # the standard 35 x 9 mm section
        self.assertAlmostEqual(L.FACE_R * 2, 0.00635)


class Tiers(unittest.TestCase):
    def test_full(self):
        r, tr = play([0.0, 0.5, 1.0, SLIP])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["clicks"], 5)
        for k in ("between", "touching", "locked", "released_1s", "still_1s"):
            self.assertTrue(r.details[k], k)
        self.assertAlmostEqual(r.details["gap_mm"], 20.0, places=3)

    def test_tier_table(self):
        t = S.tier
        self.assertEqual(t(True, True, 0.0, 5, True), 1.0)
        self.assertEqual(t(True, True, 0.0, 3, True), 1.0)
        self.assertEqual(t(True, True, 0.0, 3, False), 0.75)
        self.assertEqual(t(True, True, 0.00003, 3, True), 1.0)
        self.assertEqual(t(True, True, 0.0, 2, True), 0.5)
        self.assertEqual(t(True, True, 0.0, 1, False), 0.5)
        self.assertEqual(t(True, True, 0.0, 0, True), 0.25)  # right gap, never clicked
        self.assertEqual(t(True, True, 0.0001, 6, True), 0.25)  # gap 0.1 mm off
        self.assertEqual(t(True, True, 0.0002, 0, False), 0.25)
        self.assertEqual(t(True, True, 0.00021, 6, True), 0.0)
        self.assertEqual(t(True, False, 0.0, 6, True), 0.0)  # no block between the faces
        self.assertEqual(t(False, True, 0.0, 6, True), 0.0)  # not clean: latched, held or moving

    def test_not_locked(self):
        r, _ = play([0.0, SLIP], lever=0.0)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["locked"])

    def test_lock_threshold(self):
        r, _ = play([0.0, SLIP], lever=L.LOCK_Q)
        self.assertEqual(r.score, 1.0)
        r, _ = play([0.0, SLIP], lever=L.LOCK_Q - 0.01)
        self.assertEqual(r.score, 0.75)

    def test_click_windows(self):
        for deg, clicks, score in ((0.0, 0, 0.25), (14.9, 0, 0.25), (15.0, 1, 0.5), (29.9, 1, 0.5), (30.0, 2, 0.5),
                                   (44.9, 2, 0.5), (45.0, 3, 1.0), (60.0, 4, 1.0), (89.0, 5, 1.0)):
            r, _ = play([0.0, math.radians(deg)])
            self.assertEqual(r.details["clicks"], clicks, deg)
            self.assertEqual(r.score, score, deg)

    def test_two_clicks_unlocked_is_half(self):
        r, _ = play([0.0, math.radians(35.0)], lever=0.0)
        self.assertEqual(r.score, 0.5)

    def test_gap_windows(self):
        """A spindle that stopped short of touching: the gap off by 0.025 (right gap, no touch yet), 0.1 and 0.25 mm."""
        r = run(states(theta=L.THETA_C - 0.1, phi=0.0, lever=0.0, q=0.0003 - 0.000025))
        self.assertEqual(r.score, 0.25)  # within 0.03 mm but the faces have not touched: no click yet
        r = run(states(theta=1.0, phi=0.0, lever=0.0, q=0.0002))
        self.assertEqual(r.score, 0.25)
        self.assertAlmostEqual(r.details["gap_err_mm"], 0.1, places=3)
        r = run(states(theta=1.0, phi=0.0, lever=0.0, q=0.00005))
        self.assertEqual(r.score, 0.0)

    def test_start_scores_zero(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_block_in_but_spindle_open_scores_zero(self):
        r = run(states(theta=0.0, phi=0.0, lever=0.0, q=0.0))  # 20.30 mm: 0.30 mm off
        self.assertEqual(r.score, 0.0)

    def test_between_windows(self):
        for kw, ok in (({"dx": 0.0143}, True), ({"dx": 0.0145}, False), ({"dx": -0.0143}, True),
                       ({"dy": 0.0013}, True), ({"dy": 0.0015}, False), ({"dy": -0.0015}, False),
                       ({"dz": L.BL_Z / 2 + 0.0014}, True), ({"dz": L.BL_Z / 2 + 0.003}, False),
                       ({"quat": quat_y(math.radians(4.0))}, True), ({"quat": quat_y(math.radians(6.0))}, False),
                       ({"quat": quat_x(math.radians(6.0))}, False), ({"quat": quat_z(math.radians(20.0))}, True)):
            r, _ = play([0.0, SLIP], **kw)
            self.assertEqual(r.details["between"], ok, kw)
            self.assertEqual(r.score, 1.0 if ok else 0.0, kw)

    def test_possible_scores(self):
        seen = set()
        for phis, lever in (([0.0, SLIP], LOCKED), ([0.0, SLIP], 0.0), ([0.0, math.radians(20)], LOCKED),
                            ([0.0], LOCKED)):
            seen.add(play(phis, lever=lever)[0].score)
        seen.add(run(start()).score)
        self.assertEqual(seen, {1.0, 0.75, 0.5, 0.25, 0.0})

    def test_bench_numbers_in_the_scorer_match_the_text(self):
        self.assertEqual((L.GAP_TOL, L.GAP_NEAR, L.CONTACT_TOL), (0.00003, 0.0002, 0.00002))
        self.assertEqual((L.CLICK_DEG, L.CLICKS_FULL, L.SQUEEZE_DEG), (15.0, 3, 10.0))
        self.assertAlmostEqual(math.degrees(L.LOCK_Q), 80.2, places=1)


class Exploits(unittest.TestCase):
    def test_empty_micrometer_closed_to_20_mm_scores_zero(self):
        """The review copy's point: closing the empty micrometer to 20.000 mm earned 0.5. It earns nothing now: every
        tier needs the block between the faces."""
        r = run({S.MIC: mic_state(), S.BLOCK: start()[S.BLOCK]})
        self.assertFalse(r.details["between"])
        self.assertEqual(r.score, 0.0)
        r = run({S.MIC: mic_state(lever=0.0, phi=0.0), S.BLOCK: start()[S.BLOCK]})
        self.assertEqual(r.score, 0.0)
        for lever in (0.0, LOCKED):  # locked or not, with the block left on the plate in front
            self.assertEqual(run({S.MIC: mic_state(theta=L.THETA_C, phi=0.0, lever=lever),
                                  S.BLOCK: start()[S.BLOCK]}).score, 0.0)

    def test_empty_gap_20_with_the_block_beside_the_faces(self):
        r = run(states(dx=0.030))  # 30 mm along the plate: not between
        self.assertEqual(r.score, 0.0)
        r = run(states(dx=0.0, dy=0.003))
        self.assertEqual(r.score, 0.0)

    def test_thimble_squeeze_scores_zero(self):
        """The thimble driven on 11 degrees past touch: about 15 N. Latched: easing off afterwards does not undo it."""
        r, tr = play([0.0, SLIP], theta=L.THETA_C + math.radians(11.0))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F2"])
        r = run(states(), tracker=tr)  # relaxed back to the touch turn
        self.assertEqual(r.score, 0.0)

    def test_thimble_squeeze_just_under_the_mark(self):
        r, tr = play([0.0, SLIP], theta=L.THETA_C + math.radians(9.5))
        self.assertEqual(r.score, 1.0)
        self.assertIsNone(tr.latch["F2"])

    def test_squeeze_needs_the_block(self):
        """Past 216 degrees with no block between the faces is the empty micrometer closing: no squeeze to speak of."""
        r = S.read({S.MIC: mic_state(theta=L.THETA_C + 0.5), S.BLOCK: start()[S.BLOCK]})
        self.assertFalse(S.squeezed_now(r))
        r2 = S.read(states(theta=L.THETA_C + 0.5))
        self.assertTrue(S.squeezed_now(r2))

    def test_squeeze_on_the_way_in_then_correct_still_zero(self):
        tr = S.Tracker()
        tr.update(states(theta=L.THETA_C + math.radians(20.0), phi=0.0), FAR)
        r = run(states(), tracker=tr)
        self.assertEqual(r.score, 0.0)

    def test_clicks_need_the_faces_on_the_block(self):
        """The knob turned 100 degrees while the faces are 0.1 mm short of the block, then the spindle closes by
        itself to the block (a thimble turn): no touch, no clicks."""
        tr = S.Tracker()
        for phi in (0.0, 0.5, 1.2, 1.75):
            tr.update(states(theta=1.0, phi=phi, q=0.0002), FAR)
        r = run(states(phi=1.75), tracker=tr)  # the faces now on the block with the knob at 100 degrees
        self.assertEqual(r.details["clicks"], 0)  # slip is counted from where the knob stood at the first touch
        self.assertEqual(r.score, 0.25)

    def test_knob_turned_ahead_while_locked_then_unlocked(self):
        """Turn the knob while the lever holds the thimble, flip the lever back, close on the block: still no clicks."""
        tr = S.Tracker()
        for phi in (0.0, 1.0, 1.75):
            tr.update(states(theta=0.0, phi=phi, lever=LOCKED, q=0.0), FAR)
        r = run(states(phi=1.75), tracker=tr)
        self.assertEqual(r.score, 0.25)

    def test_knob_turned_back_and_forth_counts_the_most_not_the_sum(self):
        r, _ = play([0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0, 0.4, 0.0])
        self.assertEqual(r.details["clicks"], 1)  # 0.4 rad is 22.9 degrees: one click, however often
        self.assertEqual(r.score, 0.5)

    def test_unlocking_at_the_end_drops_to_three_quarters(self):
        _, tr = play([0.0, SLIP])
        r = run(states(phi=SLIP, lever=0.0), tracker=tr)
        self.assertEqual(r.score, 0.75)

    def test_backing_the_spindle_off_after_the_clicks(self):
        r = run(states(theta=L.THETA_C - math.radians(60.0), phi=SLIP, q=L.adv_of(L.THETA_C - math.radians(60.0))),
                tracker=play([0.0, SLIP])[1])
        self.assertEqual(r.score, 0.25)  # gap 0.083 mm over: within 0.20 mm only

    def test_block_dropped_off_the_plate(self):
        tr = S.Tracker()
        tr.update(states(dz=-0.030), FAR)  # on the table beside the stand
        self.assertIsNotNone(tr.latch["F1"])
        r = run(states(), tracker=tr)  # put back by hand: still latched
        self.assertEqual(r.score, 0.0)

    def test_block_knocked_over_on_the_plate(self):
        r, tr = play([0.0, SLIP], quat=quat_x(math.pi / 2), dz=L.BL_Y / 2)
        self.assertEqual(r.score, 0.0)  # tilted 90 degrees: not between the faces (and not fallen)
        self.assertIsNone(tr.latch["F1"])

    def test_block_lifted_and_held_in_the_air(self):
        r, _ = play([0.0, SLIP], dz=L.BL_Z / 2 + 0.020)
        self.assertEqual(r.score, 0.0)

    def test_hand_holds_the_knob(self):
        knob = arm((L.X_A, L.Y_A, L.TOP + T.KNOB_TCP), 0.1)
        r, _ = play([0.0, SLIP], arms={"left": knob, "right": FAR["right"]})
        self.assertFalse(r.details["released_1s"])
        self.assertEqual(r.score, 0.0)

    def test_hand_holds_the_block_or_rests_open_on_the_lever(self):
        for tcp in ((L.X_A, L.Y_A + 0.02, L.TOP + 0.050), (L.X_A + 0.01, L.Y_A + 0.03, L.TOP + T.LEV_TCP)):
            for opening in (1.0, 0.0):
                r, _ = play([0.0, SLIP], arms={"left": arm(tcp, opening), "right": FAR["right"]})
                self.assertEqual(r.score, 0.0, (tcp, opening))

    def test_hand_30_mm_off_a_control_is_clear(self):
        """The right edge of the let-go rule: a closed gripper whose finger zone is 30 mm or more from every control."""
        tcp = (L.X_A, L.Y_A + 0.12, L.TOP + 0.070)
        r, _ = play([0.0, SLIP], arms={"left": arm(tcp, 0.0), "right": FAR["right"]})
        self.assertTrue(r.details["released_1s"])
        self.assertEqual(r.score, 1.0)

    def test_hand_resting_on_the_stand_does_not_matter(self):
        """The base plate and rails are not controls: a closed gripper over the plate's front edge, 70 mm from the
        block, is clear."""
        tcp = (L.X_A + 0.09, L.Y_A, L.TOP + L.PL + 0.025)
        r, _ = play([0.0, SLIP], arms={"left": FAR["left"], "right": arm(tcp, 0.0)})
        self.assertEqual(r.score, 1.0)

    def test_doing_nothing_scores_zero(self):
        self.assertEqual(run(start(), steps=200).score, 0.0)


class ReleaseAndStill(unittest.TestCase):
    def test_parked_hands_are_clear(self):
        self.assertTrue(run(states()).details["released_1s"])

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(states(), arms=None).score, 0.0)
        self.assertEqual(run(states(), arms={"left": FAR["left"]}).score, 0.0)

    def test_arms_with_nan(self):
        bad = {"left": rigstate.Arm(np.array([np.nan, 0.0, 0.0]), np.zeros(3), 1.0), "right": FAR["right"]}
        self.assertEqual(run(states(), arms=bad).score, 0.0)

    def test_clicks_on_a_locked_spindle_short_of_the_block_earn_nothing(self):
        # the review's case: the spindle locked 0.02 mm short of the block, then the knob turned on the locked thimble
        th = L.THETA_C - math.radians(14.0)
        tr = S.Tracker()
        for phi in np.linspace(0.0, SLIP, 30):
            tr.update(states(theta=th, phi=float(phi), lever=LOCKED, q=L.adv_of(th)), FAR)
        r = run(states(theta=th, phi=SLIP, lever=LOCKED, q=L.adv_of(th)), FAR, WIN + 3, tr)
        self.assertEqual(r.details["clicks"], 0)
        self.assertLess(r.score, 0.5)

    def test_clicks_while_locked_at_touch_earn_nothing(self):
        tr = S.Tracker()
        for phi in np.linspace(0.0, SLIP, 30):
            tr.update(states(phi=float(phi), lever=LOCKED), FAR)
        r = run(states(phi=SLIP, lever=LOCKED), FAR, WIN + 3, tr)
        self.assertEqual(r.details["clicks"], 0)
        self.assertEqual(r.score, 0.25)

    def test_slip_short_of_the_seat_then_unlocked_counts_only_new_slip(self):
        th = L.THETA_C - math.radians(5.0)  # gap within 0.02 mm, thimble 5 degrees short: not seated
        tr = S.Tracker()
        for phi in np.linspace(0.0, SLIP, 30):
            tr.update(states(theta=th, phi=float(phi), lever=0.0, q=L.adv_of(th)), FAR)
        tr.update(states(phi=SLIP, lever=0.0), FAR)  # then seated, the knob not turned further
        r = run(states(phi=SLIP, lever=LOCKED), FAR, WIN + 3, tr)
        self.assertEqual(r.details["clicks"], 0)

    def test_still_window_needs_the_full_second(self):
        r, _ = play([0.0, SLIP], hold=WIN)  # the lever's flip (after the knob's last turn) is inside the window
        self.assertEqual(r.score, 0.0)
        r, _ = play([0.0, SLIP], hold=WIN + 1)
        self.assertEqual(r.score, 1.0)

    def test_moving_block(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(dx=-0.002 + 0.0008 * (i / WIN)), FAR)  # 0.8 mm over the window
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.last.details["still_1s"])

    def test_creeping_thimble(self):
        tr = S.Tracker()
        for i in range(WIN + 3):  # 3 degrees over the second, with the spindle tracking it
            th = L.THETA_C - math.radians(3.0) + math.radians(3.0) * (i / WIN)
            tr.update(states(theta=th, q=L.adv_of(th), phi=SLIP), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_spinning_knob(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(phi=SLIP + math.radians(5.0) * (i / WIN)), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_swinging_lever(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(lever=LOCKED - 0.05 * (i / WIN)), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_turning_block(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(quat=quat_z(math.radians(2.0) * (i / WIN))), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_spindle_drift(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(q=L.adv_of(L.THETA_C) - 0.00001 * (i / WIN)), FAR)
        self.assertEqual(tr.last.score, 0.0)

    def test_small_motion_is_still(self):
        _, tr = play([0.0, SLIP])
        for i in range(WIN + 3):
            f = i / WIN
            tr.update(states(dx=-0.002 + 0.0003 * f, phi=SLIP + math.radians(0.5) * f,
                             quat=quat_z(math.radians(0.5) * f)), FAR)
        self.assertEqual(tr.last.score, 1.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        self.assertEqual(run({}).score, 0.0)
        self.assertEqual(run({S.MIC: mic_state()}).score, 0.0)
        self.assertEqual(run({S.BLOCK: block_state()}).score, 0.0)

    def test_none_states(self):
        self.assertEqual(run(None).score, 0.0)
        self.assertEqual(run({S.MIC: None, S.BLOCK: None}).score, 0.0)

    def test_nan_block(self):
        st = states()
        st[S.BLOCK] = mk.state_from_poses(T.BLOCK_PART, (np.nan, 0.0, L.TOP + 0.04, *Q0), source="test")
        self.assertEqual(run(st).score, 0.0)

    def test_nan_joint(self):
        for name in ("spindle", "thimble", "ratchet", "lever"):
            joints = {"spindle": 0.0003, "thimble": 3.7, "ratchet": 1.5, "lever": 1.6, name: float("nan")}
            st = {S.MIC: mk.state_from_poses(T.MIC_PART, (L.X_A, L.Y_A, L.TOP, *Q0), joints=joints, source="test"),
                  S.BLOCK: block_state()}
            self.assertEqual(run(st).score, 0.0, name)

    def test_infinite_joint(self):
        st = states()
        st[S.MIC].joints["thimble"] = math.inf
        self.assertEqual(run(st).score, 0.0)

    def test_a_blow_up_restarts_the_windows(self):
        _, tr = play([0.0, SLIP])
        tr.update({}, FAR)
        self.assertEqual(tr.last.score, 0.0)
        r = run(states(), tracker=tr, steps=3)
        self.assertEqual(r.score, 0.0)  # the 1 s window starts again
        r = run(states(), tracker=tr, steps=WIN)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(tr.bad_steps, 1)

    def test_degenerate_quaternion(self):
        st = states()
        object.__setattr__(st[S.BLOCK], "quat_xyzw", np.zeros(4))
        r = run(st)
        self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(states(), FAR)
        tr.update(start(), FAR)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(tr.steps, 1)
        self.assertIsNone(tr.latch["F2"])
        self.assertEqual(tr.slip, 0.0)

    def test_pushing_the_block_back_to_the_start_marks_slowly_is_not_a_reset(self):
        tr = S.Tracker()
        dxs = np.linspace(-0.002, L.BL_START_DX, 60)
        for dx in dxs:
            tr.update(states(theta=0.0, phi=0.0, lever=0.0, q=0.0, dx=float(dx)), FAR)
        self.assertEqual(tr.resets, 0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(300):
            joints = {"spindle": float(rng.uniform(-0.001, 0.001)), "thimble": float(rng.uniform(-1, 7)),
                      "ratchet": float(rng.uniform(-1, 7)), "lever": float(rng.uniform(-0.5, 2.0))}
            q = tuple(rng.normal(size=4))
            st = {S.MIC: mk.state_from_poses(T.MIC_PART, (L.X_A, L.Y_A, L.TOP, *Q0), joints=joints, source="test"),
                  S.BLOCK: mk.state_from_poses(T.BLOCK_PART, (*rng.uniform(-0.1, 0.4, 3), *q), source="test")}
            res = tr.update(st, FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["micrometer", "block"])
        self.assertIsNone(T.MIC_PART.mass)
        self.assertAlmostEqual(T.BLOCK_PART.mass, L.BLOCK_MASS)
        self.assertEqual([j.name for j in T.MIC_PART.joints], ["spindle", "thimble", "ratchet", "lever"])
        sp = T.MIC_PART.link("spindle")
        self.assertEqual((sp.kind, sp.axis), ("prismatic", "-Z"))
        for name in ("thimble", "ratchet", "lever"):
            j = T.MIC_PART.link(name)
            self.assertEqual((j.kind, j.axis), ("revolute", "-Z"), name)
            self.assertGreater(j.spring, 0.0, name)  # a drive spring: set_targets needs one
        self.assertEqual(T.MIC_PART.link("ratchet").parent, "thimble")

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("push the block", "yaw", "mm", "step", "degree", "grip"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            self.assertEqual(set(plan), {"left", "right"})
            for wps in plan.values():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "no_lock", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_demo_timing_and_order(self):
        plan = T.plans("")

        def span(wps, prefix):
            t = 0.0
            for w in wps:
                if w.label.startswith(prefix):
                    return t, t + w.duration
                t += w.duration
            raise KeyError(prefix)

        push0, push1 = span(plan["right"], "push the block in")
        aside0, aside1 = span(plan["right"], "aside")
        grip0, _ = span(plan["left"], "grip the knob across")
        over0, _ = span(plan["left"], "over the knob")
        again0, _ = span(plan["left"], "grip the knob again")
        lever0, lever1 = span(plan["left"], "push the lever")
        self.assertGreaterEqual(over0, aside0 - 1e-9)  # the left hand comes to the knob once the right is going aside
        self.assertGreater(grip0, push1)
        self.assertGreater(again0, grip0)
        self.assertGreater(lever0, again0)  # the lever last, after the clicks
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess(total, 46.0)
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
        self.assertLessEqual((total + 3.0) * 20.0, run_limit / 2)  # demo plus the 3 s hold: within half the limit

    def test_arms_are_never_close_in_y_near_the_midline(self):
        """Wrist links hang toward the midline: with both TCPs near it they need about 0.20 m between them in y."""
        plan = T.plans("")
        total = max(sum(w.duration for w in wps) for wps in plan.values())

        def at(wps, t):
            for w in wps:
                if t < w.duration:
                    return np.array(w.pos)
                t -= w.duration
            return np.array(wps[-1].pos)

        worst = 9.0
        for t in np.arange(0.0, total, 0.1):
            pl, pr = at(plan["left"], t), at(plan["right"], t)
            if abs(pl[1]) < 0.12 or abs(pr[1]) < 0.12:
                worst = min(worst, abs(pl[1] - pr[1]))
        self.assertGreater(worst, 0.15)

    def test_geometry_of_the_moves(self):
        self.assertAlmostEqual(T.KNOB_TCP - T.PAD_TIP, T.KNOB_TIPS_Z)
        self.assertAlmostEqual(T.LEV_TCP - T.PAD_TIP, T.LEV_TIPS_Z)
        self.assertGreater(T.HOVER_L - T.PAD_TIP, L.RAT_Z1 + 0.030)  # the hover clears the knob with the pads closed
        self.assertGreater(T.HOVER_R - T.PAD_TIP, L.RAT_Z1)

    def test_retreats_are_open_before_the_hand_moves_aside(self):
        plan = T.plans("")
        for side in ("left", "right"):
            labels = [w.label for w in plan[side]]
            self.assertTrue(any(l.startswith("straight up") for l in labels), side)
            self.assertTrue(plan[side][-1].label == "aside", side)
            self.assertEqual(plan[side][-1].finger, T.OPEN)

    def test_fingers_open_before_turning_back(self):
        for w in T.plans("")["left"]:
            if w.label.startswith("turn the open hand back"):
                self.assertEqual(w.finger, T.OPEN)
            if w.label.startswith("turn the knob"):
                self.assertEqual(w.finger, T.GRIP_K)

    def test_variant_plans_differ(self):
        base = T.plans("")
        self.assertNotEqual([w.yaw for w in T.plans("two_clicks")["left"]], [w.yaw for w in base["left"]])
        self.assertNotEqual([w.yaw for w in T.plans("no_clicks")["left"]], [w.yaw for w in base["left"]])
        self.assertNotEqual([w.pos for w in T.plans("thimble")["left"]], [w.pos for w in base["left"]])
        self.assertLess(len(T.plans("no_lock")["left"]), len(base["left"]))
        self.assertEqual([w.pos for w in T.plans("no_lock")["right"]], [w.pos for w in base["right"]])


def _arm_at(wps, t):
    """The TCP and the opening (0 closed .. 1 open) at ``t`` s into a plan: straight-line smoothstep, no IK."""
    prev_p, prev_f = np.array(wps[0].pos), wps[0].finger
    for w in wps:
        if t <= w.duration:
            s = t / w.duration
            s = s * s * (3 - 2 * s)
            return prev_p + s * (np.array(w.pos) - prev_p), prev_f + s * (w.finger - prev_f)
        t -= w.duration
        prev_p, prev_f = np.array(w.pos), w.finger
    return prev_p, prev_f


def replay(variant="", idle=None):
    """The demo's arms played against the offline model and the tracker, like success() does each control step."""
    if variant:
        os.environ["DEMO_VARIANT"] = variant
    if idle:
        os.environ["DEMO_IDLE_ARM"] = idle
    try:
        m = T.OfflineModel()
        plan = T.demo()
        full = T.plans(variant)
        total = max(sum(w.duration for w in wps) for wps in full.values())
        tr = S.Tracker()
        for k in range(int((total + 3.0) * 20) + 1):
            arms = {}
            for side in ("left", "right"):
                if side in plan:
                    p, f = _arm_at(plan[side], (k + 1) / 20.0)
                    arms[side] = arm(p, min(1.0, max(0.0, f / T.OPEN)))
                else:
                    arms[side] = FAR[side]
            m.step(arms)
            tr.update(m.states(), arms)
        return m, tr
    finally:
        os.environ.pop("DEMO_VARIANT", None)
        os.environ.pop("DEMO_IDLE_ARM", None)


class OfflineModel(unittest.TestCase):
    def _arms(self, left=None, right=None):
        return {"left": left or FAR["left"], "right": right or FAR["right"]}

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)
        self.assertAlmostEqual(S.gap(S.read(st)), L.G0, places=6)

    def test_closed_pad_block_pushes_the_block_to_the_wall(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, 0.20, 100):
            m.step(self._arms(right=arm((x, L.Y_A, L.TOP + T.PUSH_TCP), 0.0)))
        self.assertAlmostEqual(m.u, -L.THROAT + L.BL_X / 2, places=6)  # stopped by the column's face
        self.assertTrue(m.between())

    def test_the_demo_push_ends_between_the_faces_short_of_the_wall(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, T.PUSH_END_X, 100):
            m.step(self._arms(right=arm((x, L.Y_A, L.TOP + T.PUSH_TCP), 0.0)))
        self.assertAlmostEqual(m.u, T.BLOCK_END_U, places=6)
        self.assertGreater(m.u, -L.THROAT + L.BL_X / 2)

    def test_open_gripper_does_not_push(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, 0.20, 60):
            m.step(self._arms(right=arm((x, L.Y_A, L.TOP + T.PUSH_TCP), 1.0)))
        self.assertAlmostEqual(m.u, L.BL_START_DX)

    def test_pad_block_above_the_block_does_not_push(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, 0.20, 60):
            m.step(self._arms(right=arm((x, L.Y_A, L.TOP + 0.090), 0.0)))
        self.assertAlmostEqual(m.u, L.BL_START_DX)

    def test_pad_block_beside_the_channel_does_not_push(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, 0.20, 60):
            m.step(self._arms(right=arm((x, L.Y_A + 0.03, L.TOP + T.PUSH_TCP), 0.0)))
        self.assertAlmostEqual(m.u, L.BL_START_DX)

    def test_a_push_from_behind_does_not_pull_the_block_out(self):
        m = T.OfflineModel()
        m.u = -0.002
        m.step(self._arms(right=arm((L.X_A - 0.1, L.Y_A, L.TOP + T.PUSH_TCP), 0.0)))
        self.assertAlmostEqual(m.u, -0.002)

    def test_turning_the_knob_runs_the_thimble_then_slips(self):
        m = T.OfflineModel()
        m.u = -0.002
        m._turn(math.radians(100.0), False)
        self.assertAlmostEqual(m.theta, math.radians(100.0))
        self.assertEqual(m.phi, 0.0)
        m._turn(math.radians(130.0), False)  # 230 in all: 216 to touch, 14 slipping
        self.assertAlmostEqual(m.theta, L.THETA_C)
        self.assertAlmostEqual(m.phi, math.radians(14.0))
        m._turn(math.radians(30.0), False)
        self.assertAlmostEqual(m.theta, L.THETA_C)
        self.assertAlmostEqual(m.phi, math.radians(44.0))
        m._turn(-math.radians(60.0), False)  # back: the slip first, then the thimble
        self.assertAlmostEqual(m.phi, 0.0)
        self.assertAlmostEqual(m.theta, L.THETA_C - math.radians(16.0))

    def test_turning_the_knob_with_no_block_runs_on_past_20_mm(self):
        m = T.OfflineModel()
        m._turn(math.radians(300.0), False)
        self.assertAlmostEqual(m.theta, math.radians(300.0))
        self.assertEqual(m.phi, 0.0)

    def test_the_thimble_gripped_goes_straight_past_touch(self):
        m = T.OfflineModel()
        m.u = -0.002
        m._turn(math.radians(300.0), True)
        self.assertAlmostEqual(m.theta, math.radians(300.0))
        self.assertTrue(S.squeezed_now(S.read(m.states())))

    def test_locked_means_no_turning(self):
        m = T.OfflineModel()
        m.lock_theta = 1.0
        m._turn(1.0, False)
        m._turn(1.0, True)
        self.assertEqual((m.theta, m.phi), (0.0, 0.0))

    def test_the_lever_is_pushed_and_snapped(self):
        m = T.OfflineModel()
        for x in np.linspace(T.LEV_X0, T.LEV_X1, 40):
            m.step(self._arms(left=arm((x, T.LEV_Y, L.TOP + T.LEV_TCP), 0.0)))
        self.assertAlmostEqual(m.lever, LOCKED)
        self.assertIsNotNone(m.lock_theta)

    def test_a_short_push_lets_the_lever_fall_back(self):
        m = T.OfflineModel()
        for x in np.linspace(T.LEV_X0, L.X_A - 0.003, 20):  # the face reaches only 2 mm: about 11 degrees
            m.step(self._arms(left=arm((x, T.LEV_Y, L.TOP + T.LEV_TCP), 0.0)))
        self.assertEqual(m.lever, 0.0)

    def test_a_push_between_the_notches_holds_where_it_is(self):
        m = T.OfflineModel()
        for x in np.linspace(T.LEV_X0, L.X_A + 0.0, 20):  # the face reaches 5 mm: about 17 degrees
            m.step(self._arms(left=arm((x, T.LEV_Y, L.TOP + T.LEV_TCP), 0.0)))
        self.assertGreater(m.lever, L.LEV_SNAP_OFF)
        self.assertLess(m.lever, L.LEV_SNAP_ON)

    def test_a_pad_block_elsewhere_does_not_move_the_lever(self):
        m = T.OfflineModel()
        for x in np.linspace(T.LEV_X0, T.LEV_X1, 40):
            m.step(self._arms(left=arm((x, 0.0, L.TOP + T.LEV_TCP), 0.0)))  # on the axis line, not at the tip
            m.step(self._arms(right=arm((x, T.LEV_Y, L.TOP + 0.10), 0.0)))  # over it
        self.assertEqual(m.lever, 0.0)

    def test_the_full_demo_scores_one(self):
        m, tr = replay("")
        self.assertEqual(tr.last.score, 1.0)
        self.assertTrue(tr.last.success)
        d = tr.last.details
        self.assertEqual(d["clicks"], 5)
        self.assertAlmostEqual(d["gap_mm"], 20.0, places=3)
        self.assertAlmostEqual(math.degrees(m.theta), 216.0, places=3)
        self.assertAlmostEqual(math.degrees(m.phi), 89.0, places=3)
        self.assertAlmostEqual(m.u, T.BLOCK_END_U, places=6)
        self.assertIsNone(tr.latch["F1"])
        self.assertIsNone(tr.latch["F2"])

    def test_one_arm_idle_scores_zero(self):
        for idle in ("left", "right"):
            _, tr = replay("", idle=idle)
            self.assertEqual(tr.last.score, 0.0, idle)

    def test_variants(self):
        for variant, want in (("no_lock", 0.75), ("two_clicks", 0.5), ("no_clicks", 0.25), ("thimble", 0.0)):
            m, tr = replay(variant)
            self.assertEqual(tr.last.score, want, variant)
        _, tr = replay("thimble")
        self.assertIsNotNone(tr.latch["F2"])
        _, tr = replay("two_clicks")
        self.assertEqual(tr.last.details["clicks"], 2)
        _, tr = replay("no_clicks")
        self.assertEqual(tr.last.details["clicks"], 0)

    def test_hold_only_scores_zero(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            m = T.OfflineModel()
            tr = S.Tracker()
            for _ in range(200):
                m.step(FAR)
                tr.update(m.states(), FAR)
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        self.assertEqual(tr.last.score, 0.0)

    def test_plan_yaw_matches_the_waypoints(self):
        wps = T.plans("")["left"]
        t, end = 0.0, {}
        for w in wps:
            t += w.duration
            end[w.label] = (t, w.yaw)
        for label, (t, yaw) in end.items():
            self.assertAlmostEqual(T.plan_yaw(wps, t - 1e-9), yaw, places=6, msg=label)

    def test_lever_angle_is_monotone(self):
        prev = -1.0
        for x in np.linspace(-0.01, 0.03, 30):
            a = T.lever_angle_for(float(x))
            self.assertGreaterEqual(a, prev)
            prev = a


class Rules(unittest.TestCase):
    def test_lever_cam(self):
        self.assertGreater(T.lever_target(L.LEV_SNAP_ON), L.LEV_Q_LOCK)  # preloaded past the lock stop
        self.assertLess(T.lever_target(L.LEV_SNAP_OFF), 0.0)  # and past the free stop
        self.assertEqual(T.lever_target(0.4), 0.4)  # between the notches it follows the joint
        self.assertGreater(T.lever_target(L.LEV_SNAP_ON + 0.001), L.LEV_Q_LOCK)
        self.assertEqual(T.lever_target(L.LEV_SNAP_ON - 0.001), L.LEV_SNAP_ON - 0.001)
        self.assertLess(T.lever_target(L.LEV_SNAP_OFF - 0.001), 0.0)
        self.assertLessEqual(L.LEV_Q_LOCK + L.LEV_PRELOAD, L.LEV_HI)  # the preload target is inside the limit

    def test_coupling_follows_until_touch(self):
        tt, tp = T.coupling_targets(1.0, 0.03, True, None)  # knob 0.03 rad ahead: the thimble follows, knob eases back
        self.assertAlmostEqual(tt, 1.03)
        self.assertEqual(tp, 0.0)
        tt, tp = T.coupling_targets(1.0, -0.02, True, None)  # thimble ahead: the knob rides on it
        self.assertAlmostEqual(tt, 0.98)
        self.assertEqual(tp, 0.0)
        tt, tp = T.coupling_targets(L.THETA_C - 0.1, 0.5, True, None)  # the thimble stops at touch
        self.assertAlmostEqual(tt, L.THETA_C)
        self.assertEqual(tp, 0.0)

    def test_coupling_slips_at_touch(self):
        tt, tp = T.coupling_targets(L.THETA_C, 0.4, True, None)
        self.assertEqual((tt, tp), (L.THETA_C, 0.4))  # the thimble holds, the knob is free
        tt, tp = T.coupling_targets(L.THETA_C - 0.005, 0.4, True, None)
        self.assertEqual((tt, tp), (L.THETA_C, 0.4))  # held at the touch turn, not at where the thimble is now
        tt, tp = T.coupling_targets(L.THETA_C, -0.1, True, None)  # turned back: the thimble goes with the knob
        self.assertAlmostEqual(tt, L.THETA_C - 0.1)
        self.assertEqual(tp, 0.0)

    def test_coupling_without_a_block_never_stops(self):
        tt, tp = T.coupling_targets(L.THETA_C, 0.4, False, None)
        self.assertAlmostEqual(tt, L.THETA_C + 0.4)
        self.assertEqual(tp, 0.0)
        tt, _ = T.coupling_targets(6.0, 1.0, False, None)
        self.assertEqual(tt, L.THIM_HI)  # clipped to the hinge's limit

    def test_coupling_locked_holds(self):
        tt, tp = T.coupling_targets(3.0, 0.2, True, 2.9)
        self.assertEqual((tt, tp), (2.9, 0.2))

    def test_spindle_target(self):
        self.assertAlmostEqual(T.spindle_target(0.0, True), -L.SP_SAG)
        self.assertAlmostEqual(T.spindle_target(L.THETA_C, True), L.ADV_C - L.SP_SAG)
        self.assertAlmostEqual(T.spindle_target(L.THETA_C + 1.0, True), L.ADV_C - L.SP_SAG)  # stopped at the block
        self.assertAlmostEqual(T.spindle_target(L.THETA_C + 1.0, False), L.adv_of(L.THETA_C + 1.0) - L.SP_SAG)
        self.assertGreaterEqual(T.spindle_target(-5.0, False), L.SP_LO)
        self.assertLessEqual(T.spindle_target(50.0, False), L.SP_HI)
        self.assertAlmostEqual(L.SP_SAG * L.SP_K, L.SP_MASS * 9.81)

    def test_lock_latches_the_thimble_turn(self):
        T._LOCK["theta"] = None
        self.assertIsNone(T._lock_state(3.0, 0.0))
        self.assertEqual(T._lock_state(3.0, 1.5), 3.0)
        self.assertEqual(T._lock_state(3.2, 1.5), 3.0)  # held at the first turn
        self.assertEqual(T._lock_state(3.2, 1.3), 3.0)  # a little back does not release it
        self.assertIsNone(T._lock_state(3.2, 0.1))
        T._LOCK["theta"] = None

    def test_rules_tolerate_bad_states(self):
        T._rules({})
        T._rules({S.MIC: None})
        T._rules({S.MIC: mic_state(), S.BLOCK: block_state()})
        T._rules({S.MIC: mic_state(theta=float("nan")), S.BLOCK: block_state()})
        T._rules({S.MIC: mic_state(), S.BLOCK: block_state(dx=float("inf"))})
        T._LOCK["theta"] = None


class Files(unittest.TestCase):
    def _json(self):
        return json.loads((HERE / "task.json").read_text())

    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = self._json()
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("185 x 90 x 30 mm", "x 150 to 335 mm", "y -45 to +45 mm", "30 mm up", "6.35 mm", "x 240 mm",
                      "4 mm wide and 2.5 mm tall", "x 219.5 to 335 mm", "y +-7.25 mm", "10.5 mm channel",
                      "20 x 36 mm, 45 mm tall", "10 x 32 mm", "14 mm across", "0 to 25 mm", "0.01 mm", "0.5 mm of",
                      "20 x 26 mm, 29 mm tall", "x 199.5 to 219.5 mm", "23.3 mm", "20.5 mm behind the axis",
                      "20.30 mm", "19 mm across", "30 x 8 x 5 mm", "+y", "90 degrees", "0.5 N", "20 mm across",
                      "19.5 mm long", "35 to 54.5 mm", "18 mm across", "12 mm long", "54.8 to 66.8 mm", "216 degrees",
                      "15 degrees", "20.000 mm", "35 x 9 mm", "49.5 g", "friction 0.2", "x 295 mm", "312.5 mm",
                      "55 mm in front of the axis"):
            self.assertIn(token, setup)
        for token in ("14.3 mm", "1.3 mm", "10 mm up, within 1.5 mm", "5 degrees", "20.02 mm", "15 degrees",
                      "80 degrees", "10 degrees", "216 degrees", "14 N", "5 mm below", "30 mm", "0.5 mm", "1 degree",
                      "0.03 mm", "0.20 mm", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "held for 1 s",
                      "load button", "no block between the faces"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "2000 steps (100 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_numbers_in_the_text_match_the_layout(self):
        self.assertEqual(round((L.X_A + L.BASE_X0) * 1000), 150)
        self.assertEqual(round((L.X_A + L.BASE_X1) * 1000), 335)
        self.assertEqual(round((L.BASE_X1 - L.BASE_X0) * 1000), 185)
        self.assertEqual(round(2 * L.BASE_HY * 1000), 90)
        self.assertEqual(round(L.PL * 1000), 30)
        self.assertEqual(round(L.X_A * 1000), 240)
        self.assertEqual(round(L.RAIL_W * 1000), 4)
        self.assertEqual(round(L.RAIL_H * 1000, 1), 2.5)
        self.assertEqual(round((L.X_A + L.RAIL_X0) * 1000, 1), 219.5)
        self.assertEqual(round((L.RAIL_GAP / 2 + L.RAIL_W / 2) * 1000, 2), 7.25)
        self.assertEqual(round(L.RAIL_GAP * 1000, 1), 10.5)
        self.assertEqual(round((L.POST_X1 - L.POST_X0) * 1000), 20)
        self.assertEqual(round(2 * L.POST_HY * 1000), 36)
        self.assertEqual(round(L.POST_H * 1000), 45)
        self.assertEqual(round((L.CLAMP_X1 - L.CLAMP_X0) * 1000), 10)
        self.assertEqual(round(2 * L.CLAMP_HY * 1000), 32)
        self.assertEqual(round((L.CLAMP_Z1 - L.CLAMP_Z0) * 1000), 20)
        self.assertEqual(round(2 * L.SCREW_R * 1000), 14)
        self.assertEqual(round((L.COL_X1 - L.COL_X0) * 1000), 20)
        self.assertEqual(round(2 * L.COL_HY * 1000), 26)
        self.assertEqual(round(L.FRAME_H * 1000), 29)
        self.assertEqual(round((L.X_A + L.COL_X0) * 1000, 1), 199.5)
        self.assertEqual(round((L.X_A + L.COL_X1) * 1000, 1), 219.5)
        self.assertEqual(round(L.ARM_Z0 * 1000, 1), 23.3)
        self.assertEqual(round(L.THROAT * 1000, 1), 20.5)
        self.assertEqual(round(L.G0 * 1000, 2), 20.30)
        self.assertEqual(round(2 * L.COLLAR_R * 1000), 19)
        self.assertEqual((round(L.LEV_L * 1000), round(L.LEV_W * 1000), round((L.LEV_Z1 - L.LEV_Z0) * 1000)),
                         (30, 8, 5))
        self.assertEqual(round(2 * L.THIM_R * 1000), 20)
        self.assertEqual(round((L.THIM_Z1 - L.THIM_Z0) * 1000, 1), 19.5)
        self.assertEqual((round((L.THIM_Z0 - L.PL) * 1000, 1), round((L.THIM_Z1 - L.PL) * 1000, 1)), (35.0, 54.5))
        self.assertEqual(round(2 * L.RAT_R * 1000), 18)
        self.assertEqual(round((L.RAT_Z1 - L.RAT_Z0) * 1000), 12)
        self.assertEqual((round((L.RAT_Z0 - L.PL) * 1000, 1), round((L.RAT_Z1 - L.PL) * 1000, 1)), (54.8, 66.8))
        self.assertEqual(round(math.degrees(L.THETA_C)), 216)
        self.assertEqual(round(L.BLOCK_MASS * 1000, 1), 49.5)
        self.assertEqual(round(L.FRICTION[0], 1), 0.2)
        self.assertEqual(round(L.BL_X0 * 1000), 295)
        self.assertEqual(round((L.BL_X0 + L.BL_X / 2) * 1000, 1), 312.5)
        self.assertEqual(round(L.BL_START_DX * 1000), 55)  # the block's centre 55 mm in front of the axis
        self.assertEqual(self._json()["context"].count("51-9061"), 1)

    def test_fidelity_states_the_rules_and_estimates(self):
        fid = self._json()["fidelity"]
        for token in ("Declared rules", "every control step", "screw", "ratchet", "cam", "1 N per micron", "est",
                      "0.5 mm per thimble turn"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = self._json()["context"]
        for token in ("51-9061", "BLS", "ratchet", "lock", "gauge blocks"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = self._json()
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
                if f == "README.md" and line.startswith("|"):
                    continue  # table rows
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)
        for token in ("--max-steps 2000", "object_jitter=0", "Runtime", "Seed", "Bench check", "Third-party assets"):
            self.assertIn(token, text)

    def test_no_pycache_left(self):
        self.assertFalse(list(HERE.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
