"""Offline tests for layout.py, parts.py, scoring.py, offline_model.py and task.py: synthetic part states (mechkit
AssemblyStates from the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
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
    import offline_model  # noqa: E402
    import parts as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

Q0 = (0.0, 0.0, 0.0, 1.0)
KIT_DIR = HERE.parents[1] / "tools" / "kit"
WIN = int(L.STILL_WINDOW_S * L.CONTROL_HZ)
X_SEATED = L.NEAR_X - L.SEAT_GAP  # the head face's world x when seated
Y0 = L.RUN_A


def quat_yaw(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def quat_roll(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def far_arms():
    return {"left": rigstate.Arm(np.array([0.05, 0.25, L.TOP + 0.25]), np.array([0.05, 0.25, L.TOP + 0.237]), 1.0),
            "right": rigstate.Arm(np.array([0.2, -0.25, L.TOP + 0.2]), np.array([0.2, -0.25, L.TOP + 0.187]), 1.0)}


def holding_pencil(tip):
    tcp = np.asarray(tip, dtype=float) + np.array([L.PN_CENTER_X, 0, L.PN_GRIP_Z])
    a = far_arms()
    a["right"] = rigstate.Arm(tcp, tcp - np.array([0, 0, 0.013]), 0.15)
    return a


class Scene:
    """Part poses (world) that a test moves about; ``states()`` gives the AssemblyStates the scorer reads."""

    def __init__(self):
        self.board = (np.array(L.BOARD_ORIGIN, float), Q0)
        self.square = (np.array(L.SQ_START, float), Q0)
        self.pencil = (np.array(L.PN_START, float), Q0)
        self.block = (np.array(L.BLOCK_ORIGIN, float), Q0)

    def states(self):
        out = {}
        for name, spec, key in ((S.BOARD, T.BOARD_PART, "board"), (S.SQUARE, T.SQUARE_PART, "square"),
                                (S.PENCIL, T.PENCIL_PART, "pencil"), (S.BLOCK, T.BLOCK_PART, "block")):
            o, q = getattr(self, key)
            out[name] = mk.state_from_poses(spec, (*o, *q), source="test")
        return out


def set_square(sc, x=X_SEATED, y=None, yaw=0.0, roll=0.0):
    y = L.SQ_START[1] if y is None else y
    sc.square = (np.array([x, y, L.TOP]), quat_yaw(yaw) if not roll else quat_roll(roll))


def run(tr, sc, y0=Y0, y1=L.RUN_B, n=150, with_square=True, square_x=X_SEATED, square_yaw=0.0, pencil_dx=0.0,
        z=L.BOARD_H - L.LEAD_DOWN, force=None, geometric=True, pen_y=L.PEN_Y_OFF):
    """Slide the square (if ``with_square``) and the pencil on its blade's end together from y0 to y1."""
    res = None
    for y in np.linspace(y0, y1, n):
        if with_square:
            set_square(sc, x=square_x, y=y - pen_y, yaw=square_yaw)
        sx, sy = float(sc.square[0][0]), float(sc.square[0][1])
        tip_x = sx + L.SQ_BLADE_END - L.PN_X0 - L.PEN_PRESS + pencil_dx
        tip = np.array([tip_x, y if not with_square else y, L.TOP + z])
        sc.pencil = (tip, Q0)
        res = tr.update(sc.states(), holding_pencil(tip), force=force, geometric=geometric)
    return res


def stow_and_let_go(tr, sc, steps=WIN + 3, stowed=True):
    end = np.array(L.PN_START, float) + (np.zeros(3) if stowed else np.array([0.0, 0.05, -0.002]))
    start = np.array(sc.pencil[0], float)
    res = None
    if np.linalg.norm(end - start) > 0.01:  # carried back in small steps, as a trial does (no reset-sized jump)
        for f in np.linspace(0, 1, 40)[1:]:
            p = start + f * (end - start) + np.array([0, 0, 0.05 * math.sin(math.pi * f)])
            sc.pencil = (p, Q0)
            tr.update(sc.states(), holding_pencil(p))
    sc.pencil = (end, Q0)
    for _ in range(steps):
        res = tr.update(sc.states(), far_arms())
    return res


def seat(tr, sc, steps=15):
    set_square(sc, x=L.SQ_START[0], y=Y0 - L.PEN_Y_OFF)
    for _ in range(3):
        tr.update(sc.states(), far_arms())
    for f in np.linspace(0, 1, 20):  # pushed home in small steps
        set_square(sc, x=L.SQ_START[0] + f * (X_SEATED - L.SQ_START[0]), y=Y0 - L.PEN_Y_OFF)
        tr.update(sc.states(), far_arms())
    for _ in range(steps):
        tr.update(sc.states(), far_arms())


def good_run(tr=None, sc=None):
    tr = tr or S.Tracker()
    sc = sc or Scene()
    seat(tr, sc)
    run(tr, sc)
    return tr, sc, stow_and_let_go(tr, sc)


class TestKit(unittest.TestCase):
    def test_kit_files_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).exists():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)


class TestLayout(unittest.TestCase):
    def test_square_starts_30mm_off_the_edge_not_seated(self):
        sc = Scene()
        st = sc.states()
        b, s = S.read(st[S.BOARD]), S.read(st[S.SQUARE])
        self.assertAlmostEqual(S.face_gap(b, s), L.SQ_GAP, places=6)
        self.assertFalse(S.seated(b, s))

    def test_blade_clears_board_and_rests_on_it(self):
        self.assertGreater(L.SQ_BLADE_Z, L.BOARD_H)
        self.assertLess(L.SQ_BLADE_Z - L.BOARD_H, 0.0005)

    def test_blade_end_over_board_and_tail_long(self):
        self.assertAlmostEqual(L.SQ_BLADE_L, 0.3048)
        tail = L.SQ_BLADE_L - L.SQ_EXT
        self.assertAlmostEqual(tail, 0.2558, places=4)
        self.assertLess(L.NEAR_X + L.SQ_EXT, L.NEAR_X + L.BOARD_W)

    def test_line_lands_50mm_in(self):
        # the lead's centre is the blade's end plus the lead's half thickness (the pencil's -x face is on the end)
        self.assertAlmostEqual(L.SQ_EXT + L.PN_STUB / 2 + L.PN_X0 + L.PN_STUB / 2, L.LINE_IN, places=6)

    def test_square_stands_on_its_head(self):
        """The centre of mass (volume weighted, as PhysX derives it) lies inside the head's footprint and the blade's
        front rests on the board, so the tail does not tip it."""
        com = T.SQUARE_PART.center_of_mass()
        self.assertGreater(com[0], -L.SQ_HEAD_DEPTH + 0.010)
        self.assertLess(com[0], 0.0)
        # even with the mass spread by weight (head 0.25, blade 0.10 kg) it stays inside the head
        heavy = (0.25 * (-L.SQ_HEAD_DEPTH / 2) + 0.10 * ((L.SQ_EXT - L.SQ_BLADE_L / 2) )) / 0.35
        self.assertGreater(heavy, -L.SQ_HEAD_DEPTH)

    def test_square_loose_under_limit(self):
        self.assertLess(L.SQ_MASS, 0.4)
        self.assertLess(L.PN_MASS, 0.4)

    def test_block_is_fixed_and_slot_fits_pencil(self):
        self.assertTrue(T.BLOCK_PART.fixed_base)
        self.assertGreater(2 * L.SLOT_HX - L.PN_W, 0.002)
        self.assertGreater(2 * L.SLOT_HY - L.PN_T, 0.002)
        self.assertGreater(L.BL_DEPTH, 0.040)

    def test_pencil_taller_than_block_and_top_low_for_gripper(self):
        self.assertGreater(L.PN_LEN - L.BL_DEPTH - L.BL_FLOOR, 0.080)
        self.assertLessEqual(L.PN_LEN - L.PN_GRIP_Z, 0.040)  # the top stays under the gripper housing

    def test_pencil_fits_gripper(self):
        self.assertGreaterEqual(L.PN_W, 0.016)
        self.assertLessEqual(L.PN_W, 0.080)

    def test_pencil_tip_stub_is_lowest_and_flat_minus_x(self):
        sh = P.pencil_shapes()
        lows = [s.center[2] - s.size[2] / 2 for s in sh] if hasattr(sh[0], "center") else None
        if lows:
            self.assertAlmostEqual(min(lows), 0.0, places=6)

    def test_demo_grips_the_blade_tail(self):
        """The left hand grips the blade's tail behind the head (the first GPU run's press on the head's top moved
        nothing): on the tail, clear of the head, pads reaching under the tail and above the table."""
        self.assertLess(L.GRIP_LOCAL[0] + 0.010 + 0.005, -L.SQ_HEAD_DEPTH)  # pads (20 mm wide) clear the head's back
        self.assertGreater(L.GRIP_LOCAL[0] - 0.010, L.SQ_EXT - L.SQ_BLADE_L)  # on the blade
        self.assertEqual(L.GRIP_LOCAL[1], 0.0)
        self.assertLess(L.GRIP_TIPS_Z, L.SQ_BLADE_Z - 0.005)  # the pads reach under the tail
        self.assertGreater(L.GRIP_TIPS_Z, 0.008)  # and stay off the table
        self.assertLess(-2 * L.BLADE_PINCH, L.SQ_BLADE_W)  # pads commanded under the blade's width
        self.assertGreater(-2 * T.F_OPEN, L.SQ_BLADE_W + 0.020)  # open fingers clear the blade
        self.assertEqual(L.YAW_GRIP, 0.0)  # closing along world y, across the blade


class TestLine(unittest.TestCase):
    def pts(self, x_off=L.LINE_IN, n=120, y0=0.0, y1=0.110, g=1.0, slope=0.0):
        ys = np.linspace(y0, y1, n)
        return np.column_stack([S.NEAR_B + x_off + slope * (ys - y0), ys, np.full(n, g)])

    def test_good_line(self):
        m = S.line_metrics(self.pts())
        self.assertTrue(S.good_line(m), m)

    def test_too_few_points(self):
        self.assertFalse(S.good_line(S.line_metrics(self.pts(n=3))))

    def test_short_line_does_not_span(self):
        m = S.line_metrics(self.pts(y1=0.085))
        self.assertFalse(m["spans"])

    def test_90mm_line_spans(self):
        m = S.line_metrics(self.pts(y1=0.0905, n=200))
        self.assertTrue(m["spans"], m)

    def test_gap_breaks_span(self):
        p = self.pts()
        p = p[(p[:, 1] < 0.045) | (p[:, 1] > 0.060)]
        self.assertFalse(S.line_metrics(p)["spans"])

    def test_not_parallel(self):
        m = S.line_metrics(self.pts(slope=0.03))  # 1.7 degrees
        self.assertFalse(m["parallel"])
        self.assertFalse(S.good_line(m))

    def test_offset(self):
        self.assertFalse(S.line_metrics(self.pts(x_off=L.LINE_IN + 0.004))["at_offset"])
        self.assertFalse(S.line_metrics(self.pts(x_off=L.LINE_IN - 0.004))["at_offset"])
        self.assertTrue(S.line_metrics(self.pts(x_off=L.LINE_IN + 0.0015))["at_offset"])

    def test_kink(self):
        p = self.pts()
        p[60:, 0] += 0.004  # the head left the edge half way
        m = S.line_metrics(p)
        self.assertFalse(m["straight"])

    def test_stray_mark_fails_straight(self):
        p = np.vstack([self.pts(), [[S.NEAR_B + 0.08, 0.05, 1.0]]])
        self.assertFalse(S.line_metrics(p)["straight"])

    def test_unguided(self):
        self.assertFalse(S.line_metrics(self.pts(g=0.0))["guided"])

    def test_partly_unguided_stretch(self):
        p = self.pts(n=300)
        p[100:140, 2] = 0.0  # a 15 mm stretch drawn without the square seated
        self.assertFalse(S.line_metrics(p)["guided"])

    def test_dwell_does_not_pad_guidance(self):
        p = self.pts(n=60, y1=0.060)
        dwell = np.tile(p[-1], (400, 1))
        dwell[:, 2] = 1.0
        q = np.vstack([p, dwell])
        m = S.line_metrics(q)
        self.assertFalse(m["spans"])  # 60 mm only


class TestTracker(unittest.TestCase):
    def test_start_scores_zero(self):
        tr, sc = S.Tracker(), Scene()
        for _ in range(30):
            r = tr.update(sc.states(), far_arms())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_run(self):
        tr, sc, res = good_run()
        self.assertEqual(res.score, 1.0, res.details)
        self.assertTrue(res.success)
        self.assertTrue(res.details["good_line"])

    def test_seated_only_scores_quarter(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        self.assertEqual(tr.last.score, 0.25, tr.last.details)

    def test_seated_for_less_than_half_second_scores_zero(self):
        tr, sc = S.Tracker(), Scene()
        set_square(sc, x=L.SQ_START[0])
        tr.update(sc.states(), far_arms())
        set_square(sc)
        for _ in range(5):
            tr.update(sc.states(), far_arms())
        self.assertEqual(tr.last.score, 0.0)

    def test_hover_does_not_ink(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, z=L.BOARD_H + 0.0020, geometric=False, force=5.0)
        self.assertEqual(res.details["line"]["n"], 0)
        self.assertEqual(res.score, 0.25)

    def test_sim_unreadable_force_no_ink(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, geometric=False, force=None)
        self.assertEqual(res.details["line"]["n"], 0)

    def test_sim_light_touch_below_threshold_no_ink(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, geometric=False, force=0.1)
        self.assertEqual(res.details["line"]["n"], 0)

    def test_sim_pressed_inks(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, geometric=False, force=1.0)
        self.assertGreater(res.details["line"]["n"], 100)
        self.assertTrue(res.details["good_line"])

    def test_sim_force_without_pose_no_ink(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, z=0.040, geometric=False, force=3.0)
        self.assertEqual(res.details["line"]["n"], 0)

    def test_pencil_lying_does_not_ink(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        tip = np.array([L.SQ_START[0] + 0.1, 0.0, L.TOP + L.BOARD_H])
        sc.pencil = (tip, quat_roll(math.radians(80)))
        res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.details["line"]["n"], 0)

    def test_freehand_without_square_on_the_edge_scores_zero(self):
        """The square never brought to the edge: the pencil drawn along a ruler-straight 50 mm line scores nothing."""
        tr, sc = S.Tracker(), Scene()
        for _ in range(5):
            tr.update(sc.states(), far_arms())
        for y in np.linspace(Y0, L.RUN_B, 150):
            tip = np.array([L.NEAR_X + L.LINE_IN, y, L.TOP + L.BOARD_H - L.LEAD_DOWN])
            sc.pencil = (tip, Q0)
            res = tr.update(sc.states(), holding_pencil(tip))
        self.assertEqual(res.score, 0.0, res.details)
        self.assertFalse(res.details["good_line"])

    def test_freehand_with_square_set_down_beside_not_seated(self):
        """Square left 12 mm off the edge, pencil on its blade's end: guided needs seated, so no line."""
        tr, sc = S.Tracker(), Scene()
        res = run(tr, sc, square_x=X_SEATED - 0.012)
        self.assertEqual(res.score, 0.0, res.details)

    def test_pencil_off_the_blade_end_unguided(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, pencil_dx=0.004)  # 4 mm off the blade's end
        self.assertFalse(res.details["line"]["guided"])
        self.assertEqual(res.score, 0.25)

    def test_pencil_beside_the_blade_unguided(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, pen_y=0.020)  # 20 mm along the blade's width
        self.assertFalse(res.details["line"]["guided"])

    def test_head_off_the_edge_during_stroke_fails(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = None
        for i, y in enumerate(np.linspace(Y0, L.RUN_B, 150)):
            x = X_SEATED - (0.004 if i > 75 else 0.0)  # the head leaves the edge half way
            set_square(sc, x=x, y=y - L.PEN_Y_OFF)
            tip = np.array([x + L.SQ_BLADE_END - L.PN_X0 - L.PEN_PRESS, y, L.TOP + L.BOARD_H - L.LEAD_DOWN])
            sc.pencil = (tip, Q0)
            res = tr.update(sc.states(), holding_pencil(tip), geometric=True)
        self.assertFalse(res.details["good_line"])

    def test_skewed_square(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, square_yaw=math.radians(3))
        self.assertFalse(res.details["good_line"])

    def test_two_runs_fail_straight(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        run(tr, sc)
        for _ in range(5):
            tr.update(sc.states(), far_arms())
        res = run(tr, sc, square_x=X_SEATED - 0.0, pencil_dx=0.0015)
        # a second pass 1.5 mm off the first: still guided (gap in range) but the points spread
        self.assertIn(res.score, (0.25, 0.5))

    def test_short_run(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        res = run(tr, sc, y1=Y0 + 0.070)
        self.assertEqual(res.score, 0.25)

    def test_not_stowed(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        run(tr, sc)
        res = stow_and_let_go(tr, sc, stowed=False)
        self.assertEqual(res.score, 0.5, res.details)

    def test_stowed_but_still_held(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        run(tr, sc)
        stow_and_let_go(tr, sc, steps=2)
        pos = np.array(L.PN_START, float)
        res = None
        for _ in range(WIN + 3):
            tcp = pos + np.array([L.PN_CENTER_X, 0, L.PN_GRIP_Z])
            a = far_arms()
            a["right"] = rigstate.Arm(tcp, tcp - np.array([0, 0, 0.013]), 0.15)
            res = tr.update(sc.states(), a)
        self.assertEqual(res.score, 0.75, res.details)

    def test_stowed_moving(self):
        tr, sc = S.Tracker(), Scene()
        seat(tr, sc)
        run(tr, sc)
        stow_and_let_go(tr, sc, steps=2)
        res = None
        for i in range(WIN + 3):
            set_square(sc, x=X_SEATED - 0.0003 * i, y=float(sc.square[0][1]))
            res = tr.update(sc.states(), far_arms())
        self.assertLess(res.score, 1.0)

    def test_pencil_tilted_in_block_not_stowed(self):
        tr, sc, _ = good_run()
        sc.pencil = (np.array(L.PN_START, float), quat_roll(math.radians(15)))
        for _ in range(WIN + 3):
            res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.5)

    def test_pencil_not_seated_in_block(self):
        tr, sc, _ = good_run()
        sc.pencil = (np.array(L.PN_START, float) + np.array([0, 0, 0.02]), Q0)
        for _ in range(WIN + 3):
            res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.5)

    def test_board_dragged_latch(self):
        tr, sc, _ = good_run()
        sc.board = (np.array(L.BOARD_ORIGIN, float) + np.array([0, 0.021, 0]), Q0)
        res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["BOARD_MOVED"])

    def test_board_turned_latch(self):
        tr, sc, _ = good_run()
        sc.board = (np.array(L.BOARD_ORIGIN, float), quat_yaw(math.radians(4)))
        res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["BOARD_TURNED"])

    def test_part_off_table_latch(self):
        tr, sc, _ = good_run()
        sc.square = (np.array(L.SQ_START, float) - np.array([0, 0, 0.05]), Q0)
        res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["FELL"])

    def test_latch_stays(self):
        tr, sc, _ = good_run()
        sc.board = (np.array(L.BOARD_ORIGIN, float) + np.array([0, 0.03, 0]), Q0)
        tr.update(sc.states(), far_arms())
        sc.board = (np.array(L.BOARD_ORIGIN, float), Q0)
        res = tr.update(sc.states(), far_arms())
        self.assertEqual(res.score, 0.0)

    def test_missing_state(self):
        tr, sc = S.Tracker(), Scene()
        st = sc.states()
        st[S.PENCIL] = None
        res = tr.update(st, far_arms())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_unreadable_arms_not_released(self):
        tr, sc, _ = good_run()
        res = None
        for _ in range(WIN + 3):
            res = tr.update(sc.states(), None)
        self.assertEqual(res.score, 0.75)

    def test_reset_clears_ink(self):
        tr, sc, _ = good_run()
        sc2 = Scene()
        tr.update(sc2.states(), far_arms())
        res = tr.update(sc2.states(), far_arms())
        self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.resets, 1)


class TestOffline(unittest.TestCase):
    def mk(self):
        return offline_model.Model(T.BOARD_PART, T.SQUARE_PART, T.PENCIL_PART, T.BLOCK_PART)

    def test_model_pencil_pick_and_stow(self):
        m = self.mk()
        tip = np.array(L.PN_START, float)
        tcp = tip + np.array([L.PN_CENTER_X, 0, L.PN_GRIP_Z])
        m.step({"right": rigstate.Arm(tcp, tcp - [0, 0, 0.013], 0.15), "left": far_arms()["left"]})
        self.assertIsNotNone(m.hold)
        tcp2 = tcp + np.array([0.05, -0.05, 0.06])
        m.step({"right": rigstate.Arm(tcp2, tcp2 - [0, 0, 0.013], 0.15), "left": far_arms()["left"]})
        self.assertTrue(np.allclose(m.pencil, tip + [0.05, -0.05, 0.06]))
        m.step({"right": rigstate.Arm(tcp, tcp - [0, 0, 0.013], 0.15), "left": far_arms()["left"]})
        m.step({"right": rigstate.Arm(tcp, tcp - [0, 0, 0.013], 1.0), "left": far_arms()["left"]})
        self.assertIsNone(m.hold)
        self.assertTrue(np.allclose(m.pencil, L.PN_START))

    def grip_tips(self):
        return np.array([L.SQ_START[0] + L.GRIP_LOCAL[0], L.SQ_START[1] + L.GRIP_LOCAL[1], L.TOP + L.GRIP_TIPS_Z])

    def test_model_square_pushed_home_stops_at_edge(self):
        m = self.mk()
        p = self.grip_tips()
        for dx in np.linspace(0, 0.034, 35):  # 4 mm more than needed
            tips = p + [dx, 0, 0]
            m.step({"right": far_arms()["right"], "left": rigstate.Arm(tips + [0, 0, 0.013], tips, 0.2)})
        self.assertAlmostEqual(m.square[0], L.NEAR_X - L.SEAT_GAP, places=6)

    def test_model_square_drag_along_y(self):
        m = self.mk()
        p = self.grip_tips()
        for dy in np.linspace(0, 0.06, 31):
            tips = p + [0, dy, 0]
            m.step({"right": far_arms()["right"], "left": rigstate.Arm(tips + [0, 0, 0.013], tips, 0.2)})
        self.assertAlmostEqual(m.square[1], L.SQ_START[1] + 0.06, places=6)

    def test_model_press_on_head_top_moves_nothing(self):
        """Closed fingertips pushed over the head's top (the first GPU run's move) do not carry the square."""
        m = self.mk()
        p = np.array([L.SQ_START[0] - 0.052, L.SQ_START[1] - 0.048, L.TOP + L.SQ_HEAD_H - 0.002])
        for dx in np.linspace(0, 0.032, 33):
            tips = p + [dx, 0, 0]
            m.step({"right": far_arms()["right"], "left": rigstate.Arm(tips + [0, 0, 0.013], tips, 0.0)})
        self.assertTrue(np.allclose(m.square, L.SQ_START))

    def test_model_open_fingers_round_blade_hold_nothing(self):
        m = self.mk()
        p = self.grip_tips()
        for dx in np.linspace(0, 0.02, 11):
            tips = p + [dx, 0, 0]
            m.step({"right": far_arms()["right"], "left": rigstate.Arm(tips + [0, 0, 0.013], tips, 1.0)})
        self.assertTrue(np.allclose(m.square, L.SQ_START))

    def test_demo_variants_known(self):
        for v in ("",) + T.VARIANTS:
            self.assertEqual(set(T.plans(v)), {"left", "right"})
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_left_holds_through_stroke(self):
        start, end = T.draw_times()
        t = 0.0
        held_until = None
        for w in T.left_moves():
            t += w.duration
            if w.label.startswith("hold it until"):
                held_until = t
        self.assertGreater(held_until, end)
        self.assertLess(start, end)

    def test_both_hands_move_in_step(self):
        """The two hands' strokes have the same duration, so square and pencil travel together."""
        lw = [w for w in T.left_moves() if w.label.startswith("slide it along")]
        rw = [w for w in T.right_moves() if w.label.startswith("draw: along")]
        self.assertEqual(len(lw), len(rw))
        for a, b in zip(lw, rw):
            self.assertAlmostEqual(a.duration, b.duration)
            self.assertAlmostEqual(a.pos[1] - lw[0].pos[1], b.pos[1] - rw[0].pos[1])

    def test_stroke_speed_even(self):
        """The stroke is cut into short equal waypoints (each is smoothstepped): every 2 mm of the line gets the
        same time, at least 10 control steps (GPU run 3: a single smoothstepped waypoint left 11 percent empty)."""
        steps = T.stroke_steps(L.RUN_A, L.RUN_B)
        dts = {round(dt, 9) for dt, _ in steps}
        self.assertEqual(len(dts), 1)
        dy = np.diff([L.RUN_A] + [y for _, y in steps])
        self.assertLessEqual(float(np.max(dy)), L.GUIDE_BIN / 2 + 1e-9)
        per_bin = L.GUIDE_BIN / float(np.max(dy)) * steps[0][0] * L.CONTROL_HZ
        self.assertGreaterEqual(per_bin, 10)

    def test_stroke_is_longer_than_the_minimum(self):
        self.assertGreater(L.RUN_B - L.RUN_A, L.LINE_MIN + 0.010)

    def test_pencil_lands_off_the_board_then_goes_down(self):
        """The lead comes down beside the blade's end with its tip off the board (no ink), goes in against the end at
        that height (under the blade's top, so it cannot ride up onto the blade), and only then down to draw."""
        self.assertGreater(L.PEN_HOVER, L.INK_Z)
        self.assertLess(L.PEN_HOVER, L.SQ_BLADE_Z + L.SQ_BLADE_T - L.BOARD_H - 0.0007)  # 0.75 mm under its top
        labels = [w.label for w in T.right_moves()]
        i_in = next(i for i, s in enumerate(labels) if s.startswith("in against"))
        i_down = next(i for i, s in enumerate(labels) if s.startswith("lead down onto the board"))
        self.assertLess(i_in, i_down)
        tips = [w.pos[2] - L.TOP - L.PN_GRIP_Z for w in T.right_moves()]
        self.assertAlmostEqual(tips[i_in], L.BOARD_H + L.PEN_HOVER, places=6)

    def test_pencil_on_blade_middle(self):
        self.assertLess(abs(L.RUN_A - L.SQ_START[1]), L.GUIDE_Y - 0.006)

    def test_stow_offset_cancels_measured_slip(self):
        """The return into the block allows for the tip's expected shift from the commanded TCP (GPU runs 1 to 3)."""
        self.assertAlmostEqual(L.STOW_DY + L.PEN_SLIP_Y, 0.0, places=9)
        self.assertLess(abs(L.STOW_DY + L.PEN_SLIP_Y), L.SLOT_HY - L.PN_T / 2)

    def test_model_slip_after_drawing(self):
        m = self.mk()
        tip = np.array(L.PN_START, float)
        tcp = tip + np.array([L.PN_CENTER_X, 0, L.PN_GRIP_Z])
        far = far_arms()["left"]
        m.step({"right": rigstate.Arm(tcp, tcp - [0, 0, 0.013], 0.15), "left": far})
        on_board = np.array([0.25, -0.05, L.TOP + L.BOARD_H]) + [L.PN_CENTER_X, 0, L.PN_GRIP_Z]
        for z in (0.0, 0.0, 0.010):
            t = on_board + [0, 0, z]
            m.step({"right": rigstate.Arm(t, t - [0, 0, 0.013], 0.15), "left": far})
        self.assertTrue(m.slipped)
        self.assertAlmostEqual(float(m.pencil[1]), -0.05 + L.PEN_SLIP_Y, places=9)

    def test_model_grip_survives_press_bias(self):
        self.assertLess(L.PRESS_X_BIAS + 0.001, offline_model.GRIP_R)

    def test_seat_wiggle_before_pencil_down(self):
        t = 0.0
        last_wiggle = None
        for w in T.left_moves():
            t += w.duration
            if w.label.startswith("work the face onto the edge"):
                last_wiggle = t
        t = 0.0
        for w in T.right_moves():
            t += w.duration
            if w.label.startswith("lead down to 1 mm"):
                down = t
                break
        self.assertIsNotNone(last_wiggle)
        self.assertLess(last_wiggle + 1.0, down)


class TestTaskJson(unittest.TestCase):
    def setUp(self):
        self.j = json.loads((HERE / "task.json").read_text())

    def test_fields(self):
        for k in ("title", "category", "instruction", "context", "setup", "scoring", "run_limit", "fidelity",
                  "reset_notes"):
            self.assertIn(k, self.j)
        self.assertEqual(self.j["category"], "Impact on jobs")

    def test_instruction_matches(self):
        self.assertEqual(self.j["instruction"], T.INSTRUCTION)

    def test_no_dashes(self):
        for k, v in self.j.items():
            if isinstance(v, str):
                self.assertNotRegex(v, "[\\u2013\\u2014]", k)

    def test_says_plainly_no_pencil_line_in_sim(self):
        for k in ("scoring", "fidelity"):
            self.assertIn("pencil line cannot be drawn", self.j[k].lower().replace("a pencil line", "a pencil line"), k)

    def test_numbers_in_text(self):
        s = self.j["setup"] + " " + self.j["scoring"]
        for needle in ("19.05", "139.7", "610", "304.8", "25.4", "49 mm", "50 mm", "90 mm", "1.5 mm", "30 mm", "0.3 N",
                       "20 mm", "3 mm", "10 degrees", "2400" if False else "0.73"):
            self.assertIn(needle, s, needle)
        self.assertIn("2400", self.j["run_limit"])

    def test_run_limit_covers_demo(self):
        steps = int(re.match(r"(\d+)", self.j["run_limit"]).group(1))
        d = max(sum(w.duration for w in p) for p in T.plans().values())
        self.assertGreater(steps, d * L.CONTROL_HZ + 100)


if __name__ == "__main__":
    unittest.main()
