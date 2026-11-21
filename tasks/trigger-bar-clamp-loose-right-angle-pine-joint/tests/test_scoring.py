"""Offline tests for layout.py, parts.py, scoring.py, offline_clamp.py and task.py: synthetic part states (mechkit
AssemblyStates from the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import pathlib
import subprocess
import sys
import types
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
    import offline_clamp  # noqa: E402
    import parts as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

WIN = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))
Q0 = (0.0, 0.0, 0.0, 1.0)
SEATED_XF = L.B_X0  # the fixed pad's face on B's back face
JAW_SEATED = L.JAW_STOP
PULL_OK = L.JAW_STOP + 0.006  # 120 N


def quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def part_states(a=(L.A_CX, L.A_CY, 0.0), b=(L.B_CX, L.B_CY, 0.0), clamp=(SEATED_XF, L.CLAMP_YC, JAW_SEATED),
                trig=0.0, rel=0.0, a_z=0.0, clamp_z=0.0):
    return {
        S.A: mk.state_from_poses(T.A_PART, (a[0], a[1], L.TOP + a_z, *quat(a[2])), source="test"),
        S.B: mk.state_from_poses(T.B_PART, (b[0], b[1], L.TOP, *quat(b[2])), source="test"),
        S.CLAMP: mk.state_from_poses(T.CLAMP_PART, (clamp[0], clamp[1], L.TOP + clamp_z, *Q0),
                                     joints={"jaw": clamp[2], "trigger": trig, "release": rel}, source="test"),
    }


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


AWAY = {"left": arm((0.20, 0.27, L.TOP + 0.16)), "right": arm((0.20, -0.27, L.TOP + 0.16))}


def run(parts, arms=AWAY, pull=PULL_OK, steps=WIN + 5, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(parts, arms, pull)
    return res


class LayoutTests(unittest.TestCase):
    def test_boards_are_1x2_pine(self):
        self.assertAlmostEqual(L.BOARD_T, 0.019, places=4)
        self.assertAlmostEqual(L.BOARD_W, 0.038, places=4)
        self.assertLess(L.B_MASS, 0.4)
        self.assertLess(L.A_MASS, 0.4)
        self.assertLess(L.CLAMP_MASS, 0.4)
        self.assertAlmostEqual(L.B_MASS, 0.108, places=3)

    def test_clamp_fits_the_boards_with_play(self):
        # a 6 in clamp: the boards (138 mm) sit in a 150 mm opening, 12 mm of jaw travel to take up
        self.assertAlmostEqual(L.CAPACITY, 0.150, places=3)
        self.assertLessEqual(L.CAPACITY, 0.1524)
        self.assertAlmostEqual(L.JAW_STOP, 0.012, places=6)
        self.assertEqual(round(L.JAW_STOP / L.STEP), 4)  # four squeezes take the slack, the rest load it
        self.assertAlmostEqual(L.CLAMP_XF + L.G0 - L.A_X1, 0.009, places=6)  # moving pad starts 9 mm off A
        self.assertAlmostEqual(L.B_X0 - L.CLAMP_XF, 0.003, places=6)  # fixed pad 3 mm off B
        self.assertGreaterEqual(L.THROAT, 0.05)

    def test_force_per_squeeze_and_demo_load(self):
        per = L.K_JAW * L.STEP
        self.assertAlmostEqual(per, 60.0, places=6)
        self.assertGreater(per, L.F_MIN)  # one squeeze past contact already clamps
        self.assertEqual(T.SQUEEZES, 6)
        self.assertAlmostEqual(L.K_JAW * (T.SQUEEZES * L.STEP - L.JAW_STOP), 120.0, places=6)

    def test_trigger_travel_matches_the_grip(self):
        self.assertAlmostEqual(L.SPAN_REST, 0.064, places=6)
        self.assertGreaterEqual(L.SPAN_REST, 0.016)
        self.assertLessEqual(L.SPAN_REST + 0.008, 0.080)  # open pads clear the grip
        self.assertLessEqual(T.C_AROUND * -2, 0.080)
        self.assertGreater(-2 * T.C_AROUND, L.SPAN_REST)
        self.assertGreater(-2 * T.C_OPEN, L.SPAN_REST)  # between squeezes the trigger springs back
        self.assertLessEqual(L.SPAN_STOP, 0.080)
        self.assertGreaterEqual(L.SPAN_STOP, 0.016)
        # the spring is weaker than the gripper's 5 N at the grip point
        self.assertLess(L.TRIG_SPRING * L.TRIG_MAX / abs(L.TRIG_ARM), 5.0)

    def test_reach_and_height(self):
        gx, gy = T.grip_xy(0.0, L.CLAMP_YC0)
        self.assertTrue(0.10 <= gx <= 0.50)
        self.assertTrue(0.10 <= L.A_HOLD_X <= 0.50)
        self.assertLessEqual(T.HOVER, 0.225)
        self.assertLess(L.JAW_H, 0.05)
        # the left pads (closed on A) stay clear of the clamp's bar when it is placed
        pad_outer = L.A_CY - L.BOARD_W / 2 - 0.008
        bar_front = L.CLAMP_YC + L.BAR_Y
        self.assertGreaterEqual(pad_outer - bar_front, 0.005)
        # the left gripper's pads clear the moving pad
        self.assertGreaterEqual((L.CLAMP_XF + L.G0) - (L.A_HOLD_X + 0.0135), 0.020)

    def test_start_is_a_rough_dry_fit(self):
        a = S.read_pose(part_states(a=(L.A_CX + L.A_START_DX, L.A_CY + L.A_START_DY, L.A_START_YAW))[S.A])
        b = S.read_pose(part_states()[S.B])
        g = S.joint_geometry(a, b)
        self.assertTrue(0.005 < g["flush"] < 0.007)  # 8 mm in at A's centre, less at the joint corner with the turn
        self.assertGreater(g["gap"], L.GAP_LOOSE * 0.9)  # 4 mm off B's face (a bit more with the turn)
        self.assertGreater(g["square"], L.SQUARE_TIGHT)
        self.assertLess(g["square"], L.SQUARE_LOOSE)
        self.assertFalse(S.joint_loose(g))
        self.assertFalse(S.joint_tight(g))


class GeometryTests(unittest.TestCase):
    def poses(self, **kw):
        p = part_states(**kw)
        return S.read_pose(p[S.A]), S.read_pose(p[S.B]), S.read_pose(p[S.CLAMP])

    def test_ideal_joint(self):
        a, b, c = self.poses()
        g = S.joint_geometry(a, b)
        self.assertAlmostEqual(g["gap"], 0.0, places=6)
        self.assertAlmostEqual(g["flush"], 0.0, places=6)
        self.assertAlmostEqual(g["square"], 0.0, places=6)
        self.assertTrue(S.joint_tight(g))
        cg = S.clamp_geometry(a, b, c, JAW_SEATED)
        self.assertAlmostEqual(cg["fixed_gap"], 0.0, places=6)
        self.assertAlmostEqual(cg["moving_gap"], 0.0, places=6)
        self.assertAlmostEqual(cg["cover_a"], L.PAD_Y1 - (L.A_CY - L.BOARD_W / 2 - L.CLAMP_YC), places=6)
        self.assertGreaterEqual(cg["cover_a"], L.PAD_OVERLAP_MIN)
        self.assertTrue(S.on_the_work(cg))

    def test_flush_is_measured_at_the_joint_with_a_sign(self):
        for dy, want in ((0.001, 0.001), (-0.001, -0.001), (0.0014, 0.0014)):
            a, b, _ = self.poses(a=(L.A_CX, L.A_CY + dy, 0.0))
            self.assertAlmostEqual(S.joint_geometry(a, b)["flush"], want, places=6)

    def test_tight_limits(self):
        edge = L.FLUSH_TIGHT - 1e-5
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY + edge, 0.0))
        self.assertTrue(S.joint_tight(S.joint_geometry(a, b)))
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY + L.FLUSH_TIGHT + 5e-4, 0.0))
        g = S.joint_geometry(a, b)
        self.assertFalse(S.joint_tight(g))
        self.assertTrue(S.joint_loose(g))
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY + L.FLUSH_LOOSE + 5e-4, 0.0))
        self.assertFalse(S.joint_loose(S.joint_geometry(a, b)))

    def test_square_in_mm_over_the_stub(self):
        yaw = math.asin(L.SQUARE_TIGHT / L.A_LEN) - 1e-4
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY, yaw))
        g = S.joint_geometry(a, b)
        self.assertLessEqual(g["square"], L.SQUARE_TIGHT)
        self.assertAlmostEqual(math.degrees(math.asin(L.SQUARE_TIGHT / L.A_LEN)), 1.43, places=2)
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY, math.radians(2.5)))
        g = S.joint_geometry(a, b)
        self.assertFalse(S.joint_tight(g))
        self.assertTrue(abs(g["square"] - 0.0044) < 1e-4)

    def test_board_turned_end_for_end_is_the_same_board(self):
        a, b, _ = self.poses(a=(L.A_CX, L.A_CY, math.pi))
        self.assertTrue(S.joint_tight(S.joint_geometry(a, b)))

    def test_whole_assembly_turned_is_still_square(self):
        # both boards yawed together about the world: the joint is measured in B's frame
        yaw = math.radians(20)
        c, s = math.cos(yaw), math.sin(yaw)
        def turn(p):
            return (c * p[0] - s * p[1], s * p[0] + c * p[1])
        a_xy, b_xy = turn((L.A_CX, L.A_CY)), turn((L.B_CX, L.B_CY))
        a, b, _ = self.poses(a=(*a_xy, yaw), b=(*b_xy, yaw))
        g = S.joint_geometry(a, b)
        self.assertAlmostEqual(g["gap"], 0.0, places=6)
        self.assertAlmostEqual(g["flush"], 0.0, places=6)
        self.assertAlmostEqual(g["square"], 0.0, places=6)


class TierTests(unittest.TestCase):
    def test_perfect_is_one(self):
        res = run(part_states())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.details["force_n"], 120.0, places=3)

    def test_not_clamped_is_zero_whatever_the_joint(self):
        for pull in (0.0, L.JAW_STOP, L.JAW_STOP + 0.001):  # up to 20 N: under 40 N
            self.assertEqual(run(part_states(), pull=pull).score, 0.0, pull)
        self.assertEqual(run(part_states(), pull=None).score, 0.0)
        self.assertEqual(run(part_states(), pull=float("nan")).score, 0.0)
        self.assertEqual(run(part_states(), pull=L.JAW_STOP + 0.002 - 1e-6).score, 0.0)  # 39.98 N
        self.assertEqual(run(part_states(), pull=L.JAW_STOP + 0.002 + 1e-6).score, 1.0)

    def test_clamped_but_crooked_is_quarter(self):
        # the pad pushed on a stub left 8 mm short of flush: clamped, outside the loose limits
        parts = part_states(a=(L.A_CX, L.A_CY + 0.008, 0.0))
        res = run(parts)
        self.assertEqual(res.score, 0.25)
        self.assertTrue(res.details["clamped"])

    def test_loose_is_half(self):
        res = run(part_states(a=(L.A_CX, L.A_CY + 0.002, 0.0)))
        self.assertEqual(res.score, 0.5)

    def test_tight_with_a_hand_on_is_three_quarters(self):
        arms = {"left": arm((L.A_HOLD_X, L.A_CY, L.TOP + 0.021), 0.0), "right": AWAY["right"]}
        res = run(part_states(), arms=arms)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.details["released_1s"])

    def test_hand_on_the_clamp_counts_too(self):
        gx, gy = T.grip_xy(JAW_SEATED, L.CLAMP_YC)
        arms = {"left": AWAY["left"], "right": arm((gx, gy, L.TOP + 0.021), 0.5)}
        self.assertEqual(run(part_states(), arms=arms).score, 0.75)

    def test_moving_for_a_second_is_three_quarters(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 5):
            res = tr.update(part_states(b=(L.B_CX, L.B_CY + 0.0004 * (i % 6), 0.0)), AWAY, PULL_OK)
        self.assertLess(res.score, 1.0)
        res = run(part_states(), tracker=tr, steps=WIN + 2)
        self.assertEqual(res.score, 1.0)

    def test_needs_a_full_second_before_one(self):
        tr = S.Tracker()
        for i in range(WIN - 2):
            res = tr.update(part_states(), AWAY, PULL_OK)
        self.assertEqual(res.score, 0.75)
        for i in range(5):
            res = tr.update(part_states(), AWAY, PULL_OK)
        self.assertEqual(res.score, 1.0)

    def test_unreadable_arms_are_not_released(self):
        self.assertEqual(run(part_states(), arms=None).score, 0.75)
        bad = {"left": AWAY["left"], "right": rigstate.Arm(np.array([np.nan, 0, 0]), np.zeros(3), 1.0)}
        self.assertEqual(run(part_states(), arms=bad).score, 0.75)

    def test_fall_latches_zero(self):
        tr = S.Tracker()
        run(part_states(), tracker=tr)
        tr.update(part_states(a_z=-0.03), AWAY, PULL_OK)
        self.assertEqual(run(part_states(), tracker=tr).score, 0.0)
        self.assertTrue(tr.latched)

    def test_bad_state_is_zero_and_restarts_windows(self):
        tr = S.Tracker()
        run(part_states(), tracker=tr)
        parts = part_states()
        parts[S.CLAMP] = None
        self.assertEqual(tr.update(parts, AWAY, PULL_OK).score, 0.0)
        self.assertEqual(tr.update(part_states(), AWAY, PULL_OK).score, 0.75)

    def test_clamp_back_on_its_start_mark_resets_the_trial(self):
        tr = S.Tracker()
        run(part_states(clamp=(SEATED_XF, L.B_Y0 + L.B_LEN + 0.10, JAW_SEATED)), tracker=tr)  # the clamp far off
        tr.update(part_states(clamp=(L.CLAMP_XF, L.CLAMP_YC0, 0.0)), AWAY, 0.0)
        self.assertEqual(tr.resets, 1)
        self.assertFalse(tr.latched)


class ExploitTests(unittest.TestCase):
    def test_boards_lying_right_without_the_clamp_score_nothing(self):
        far = (L.CLAMP_XF, L.CLAMP_YC0, 0.0)
        self.assertEqual(run(part_states(clamp=far), pull=0.0).score, 0.0)
        self.assertEqual(run(part_states(clamp=far), pull=PULL_OK).score, 0.0)  # pulled, but not on the boards

    def test_clamp_beside_the_work_is_not_on_it(self):
        off = (SEATED_XF, L.B_Y0 + L.B_LEN + 0.10, JAW_SEATED)  # past B's far end
        self.assertEqual(run(part_states(clamp=off)).score, 0.0)

    def test_pads_must_cover_both_boards(self):
        thin = (SEATED_XF, L.B_Y0 + 0.010 - L.PAD_Y1, JAW_SEATED)  # the pads reach only 10 mm over the boards' width
        self.assertEqual(run(part_states(clamp=thin)).score, 0.0)
        ok = (SEATED_XF, L.B_Y0 + L.PAD_OVERLAP_MIN + 0.001 - L.PAD_Y1, JAW_SEATED)  # 16 mm: on the work
        self.assertEqual(run(part_states(clamp=ok)).score, 1.0)

    def test_clamp_must_be_flat(self):
        parts = part_states()
        parts[S.CLAMP] = mk.state_from_poses(T.CLAMP_PART, (SEATED_XF, L.CLAMP_YC, L.TOP, *mk.quat_from_ypr(0.0, 0.0, 0.5)[:4]),
                                             joints={"jaw": JAW_SEATED, "trigger": 0.0, "release": 0.0}, source="test")
        self.assertEqual(run(parts).score, 0.0)

    def test_release_tab_pressed_drops_the_force(self):
        res = run(part_states(rel=L.REL_UPPER), pull=PULL_OK)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["force_n"], 0.0)
        res = run(part_states(rel=0.4 * L.REL_UPPER), pull=PULL_OK)  # under half: still engaged
        self.assertEqual(res.score, 1.0)

    def test_jaw_not_at_the_boards_is_not_clamped_even_if_pulled(self):
        # the pull alone (an open jaw, ratchet run up in the air) gives no contact: the moving pad is 9 mm off
        res = run(part_states(clamp=(SEATED_XF, L.CLAMP_YC, 0.0)), pull=L.JAW_UPPER)
        self.assertEqual(res.score, 0.0)

    def test_stub_on_the_wrong_side_or_off_the_end_does_not_pass(self):
        for a in ((L.B_X0 - L.A_LEN / 2, L.A_CY, 0.0),  # A behind B
                  (L.A_CX, L.B_Y0 - 0.050, 0.0),  # A sticking 69 mm out of B's end
                  (L.A_CX, L.A_CY + 0.100, 0.0),  # A along B's face, far from the end
                  (L.A_CX, L.A_CY, math.pi / 2)):  # A turned across
            res = run(part_states(a=a))
            self.assertLess(res.score, 1.0, a)

    def test_stacked_boards_do_not_pass(self):
        self.assertLess(run(part_states(a_z=0.019)).score, 1.0)

    def test_overlapped_boards_do_not_pass(self):
        self.assertLess(run(part_states(a=(L.A_CX - 0.010, L.A_CY, 0.0))).score, 1.0)

    def test_gap_open_does_not_pass(self):
        parts = part_states(a=(L.A_CX + 0.005, L.A_CY, 0.0))
        self.assertLess(run(parts).score, 1.0)

    def test_squareness_by_one_degree_extra(self):
        res = run(part_states(a=(L.A_CX, L.A_CY, math.radians(1.6))))
        self.assertEqual(res.score, 0.5)

    def test_throwing_a_board_off_the_table_latches(self):
        tr = S.Tracker()
        tr.update(part_states(b=(L.B_CX, L.B_CY, 0.0), a_z=-0.5), AWAY, PULL_OK)
        self.assertTrue(tr.latched)


class RatchetTests(unittest.TestCase):
    def setUp(self):
        T._reset_ratchet()

    def states(self, trig, rel=0.0):
        return {S.CLAMP: types.SimpleNamespace(joints={"jaw": 0.0, "trigger": trig, "release": rel})}

    def squeeze(self, n=1):
        for _ in range(n):
            T._ratchet(self.states(L.TRIG_MAX))  # squeezed to the stop
            T._ratchet(self.states(L.TRIG_MAX))  # held: no second step
            T._ratchet(self.states(0.0))  # let back

    def test_each_full_squeeze_is_one_step(self):
        self.squeeze(1)
        self.assertAlmostEqual(T.RATCHET["pull"], L.STEP)
        self.squeeze(5)
        self.assertAlmostEqual(T.RATCHET["pull"], 6 * L.STEP)
        self.assertEqual(T.RATCHET["steps"], 6)

    def test_holding_the_trigger_down_is_one_step(self):
        for _ in range(100):
            T._ratchet(self.states(L.TRIG_MAX))
        self.assertAlmostEqual(T.RATCHET["pull"], L.STEP)

    def test_half_squeezes_do_nothing(self):
        for _ in range(10):
            T._ratchet(self.states(0.7 * L.TRIG_MAX))
            T._ratchet(self.states(0.45 * L.TRIG_MAX))
        self.assertEqual(T.RATCHET["pull"], 0.0)

    def test_must_come_back_under_30_percent_to_step_again(self):
        T._ratchet(self.states(L.TRIG_MAX))
        T._ratchet(self.states(0.5 * L.TRIG_MAX))  # not far enough back
        T._ratchet(self.states(L.TRIG_MAX))
        self.assertAlmostEqual(T.RATCHET["pull"], L.STEP)
        T._ratchet(self.states(0.2 * L.TRIG_MAX))
        T._ratchet(self.states(L.TRIG_MAX))
        self.assertAlmostEqual(T.RATCHET["pull"], 2 * L.STEP)

    def test_release_tab_frees_the_pawl(self):
        self.squeeze(3)
        T._ratchet(self.states(0.0, rel=L.REL_UPPER))
        self.assertEqual(T.RATCHET["pull"], 0.0)
        self.assertGreater(T.RATCHET["freed"], 0)
        self.squeeze(1)  # squeezing again starts from the open jaw
        self.assertAlmostEqual(T.RATCHET["pull"], L.STEP)

    def test_the_pull_never_passes_the_bar(self):
        self.squeeze(60)
        self.assertLessEqual(T.RATCHET["pull"], L.JAW_UPPER + 1e-12)

    def test_the_gripper_that_holds_the_trigger_gets_the_travel(self):
        # at the grip's stop (46 mm) the trigger has turned TRIG_MAX: the hinge's own limit lies past it
        self.assertGreater(L.TRIG_UPPER, L.TRIG_MAX)
        self.assertAlmostEqual(L.TRIG_MAX * abs(L.TRIG_ARM), L.TRIG_TRAVEL, places=3)


class OfflineModelTests(unittest.TestCase):
    def make(self):
        return offline_clamp.ClampModel(T.A_PART, T.B_PART, T.CLAMP_PART)

    def arms(self, left=None, right=None):
        return {"left": left or AWAY["left"], "right": right or AWAY["right"]}

    def test_pinch_squares_and_flushes_the_stub(self):
        m = self.make()
        tcp = (L.A_HOLD_X + L.A_START_DX, L.A_CY, L.TOP + 0.021)
        for _ in range(3):
            m.step(self.arms(left=arm(tcp, 0.0)))
        self.assertAlmostEqual(m.a[1], L.A_CY, places=6)
        self.assertEqual(m.a[2], 0.0)
        self.assertEqual(m.holds.get("left"), "a")

    def test_open_hand_does_not_move_it(self):
        m = self.make()
        tcp = (L.A_HOLD_X + L.A_START_DX, L.A_CY, L.TOP + 0.021)
        m.step(self.arms(left=arm(tcp, 0.64)))
        self.assertEqual(m.a[1], L.A_CY + L.A_START_DY)

    def test_squeezes_pull_the_jaw_then_load_it(self):
        m = self.make()
        m.a = [L.A_CX, L.A_CY, 0.0]
        m.c = [L.CLAMP_XF, L.CLAMP_YC]
        gx, gy = T.grip_xy(0.0, L.CLAMP_YC)
        m.step(self.arms(right=arm((gx, gy, L.TOP + 0.021), 0.66)))  # takes hold, trigger a little in
        self.assertEqual(m.target, 0.0)
        jaws = []
        for k in range(6):
            x, _ = T.grip_xy(L.STEP * min(k, 4), L.CLAMP_YC)
            m.step(self.arms(right=arm((x, gy, L.TOP + 0.021), 0.0)))  # closed to the stop
            jaws.append(round(m.jaw * 1000, 3))
            m.step(self.arms(right=arm((x, gy, L.TOP + 0.021), 0.75)))  # open: back
        self.assertEqual(jaws, [3.0, 6.0, 9.0, 12.0, 12.0, 12.0])
        self.assertAlmostEqual(m.target, 0.018)
        self.assertAlmostEqual(m.a[0], L.A_CX - 0.003, places=6)  # A pushed up to B and B to the fixed pad
        self.assertAlmostEqual(m.b[0] - L.BOARD_W / 2, L.CLAMP_XF, places=6)
        self.assertEqual(m.a[2], 0.0)

    def test_empty_squeeze_off_the_boards_runs_the_jaw_free(self):
        m = self.make()
        gx, gy = T.grip_xy(0.0, L.CLAMP_YC0)
        m.step(self.arms(right=arm((gx, gy, L.TOP + 0.021), 0.66)))
        for _ in range(3):
            m.step(self.arms(right=arm((gx, gy, L.TOP + 0.021), 0.0)))
            m.step(self.arms(right=arm((gx, gy, L.TOP + 0.021), 0.75)))
        self.assertAlmostEqual(m.jaw, 0.009)
        self.assertEqual(m.a[1], L.A_CY + L.A_START_DY)  # the boards did not move


class DemoTests(unittest.TestCase):
    def test_plans_have_both_arms_and_fit_the_limit(self):
        plan = T.plans()
        self.assertEqual(set(plan), {"left", "right"})
        t = max(sum(w.duration for w in p) for p in plan.values())
        self.assertLess(t * L.CONTROL_HZ + 60, 1500)  # inside the stated run limit with the policy's stop margin
        self.assertEqual(T.plans("five")["right"].__len__() < T.plans()["right"].__len__(), True)

    def test_left_opens_after_the_last_squeeze_and_not_before(self):
        plan = T.plans()
        right = plan["right"]
        idx = [i for i, w in enumerate(right) if w.label.startswith("squeeze")]
        self.assertEqual(len(idx), T.SQUEEZES)
        t_last = sum(w.duration for w in right[: idx[-1] + 3])  # through the last close, hold and open
        left = plan["left"]
        t_open = sum(w.duration for w in left[: [w.label for w in left].index("open")])
        self.assertAlmostEqual(t_open, t_last, places=6)

    def test_every_gap_is_inside_the_gripper_range_or_closed(self):
        for p in T.plans().values():
            for w in p:
                gap = -2 * w.finger
                self.assertTrue(gap == 0.0 or 0.016 <= gap <= 0.094, (w.label, gap))

    def test_hold_only_and_idle_arm(self):
        import os
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_task_file_contract(self):
        self.assertEqual(T.SCENE, "station")
        self.assertEqual([o.name for o in T.OBJECTS], [S.B, S.A, S.CLAMP, "bench_stop"])
        self.assertTrue(all(n.isidentifier() for n in (S.A, S.B, S.CLAMP)))
        self.assertIn("flush and square", T.INSTRUCTION)


class OfflineCheckerTests(unittest.TestCase):
    """The offline checker's report: predicted 1.0, one arm clearly below 1.0, counterfactual squeezes."""

    @classmethod
    def setUpClass(cls):
        if OFFLINE is None:
            raise unittest.SkipTest("tools/offline not found")
        out = subprocess.run([sys.executable, str(OFFLINE / "check_task.py"), str(HERE), "--one-arm", "--json", "-"],
                             capture_output=True, text=True, timeout=600)
        txt = out.stdout
        start = txt.find("{")
        cls.report = json.loads(txt[start:]) if start >= 0 else {}
        cls.rc = out.returncode

    def test_passes(self):
        self.assertEqual(self.rc, 0)

    def test_demo_predicts_one(self):
        proxy = self.report["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertIsNotNone(proxy["success_held_at"])

    def test_one_arm_clearly_below_one(self):
        one = self.report["one_arm"]
        self.assertLessEqual(one["left_only"], 0.25)
        self.assertLessEqual(one["right_only"], 0.25)

    def test_arms_clear_each_other(self):
        self.assertGreater(self.report["playback"]["min_arm_clearance_m"], 0.005)


if __name__ == "__main__":
    unittest.main()
