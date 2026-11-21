"""Offline tests for layout.py, pendulum.py, parts.py, scoring.py, offline_model.py and task.py: synthetic metronome
states (mechkit AssemblyStates from the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

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
    import clip  # noqa: E402
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import offline_model  # noqa: E402
    import pendulum as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

DT = L.DT
KIT_DIR = HERE.parents[1] / "tools" / "kit"
Q = tuple(mk.yaw_quat(L.M_YAW))
O = np.array(L.M_ORIGIN, dtype=float)


def metro(theta=0.0, slide=0.0, offset=(0.0, 0.0, 0.0), quat=Q):
    return {S.METRONOME: mk.state_from_poses(T.METRONOME_PART, (*(O + np.asarray(offset, float)), *quat),
                                             joints={"rod": theta, "weight": slide}, source="test")}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp + np.array([0.0, 0.0, -0.013]), opening)


FAR = {"left": arm((0.15, 0.40, L.TOP + 0.20)), "right": arm((0.15, -0.30, L.TOP + 0.20))}
GOAL_SLIDE = P.R_GOAL - P.R_START


def swing(bpm_rate, amp, seconds, slide=GOAL_SLIDE, arms=FAR, tracker=None, offset=(0.0, 0.0, 0.0), phase=0.3):
    """A free swing at ``bpm_rate`` ticks per minute (period 120 / bpm), amplitude ``amp``, sampled at 20 Hz."""
    tr = tracker or S.Tracker()
    w = 2 * math.pi * bpm_rate / 120.0
    res = None
    for k in range(int(seconds / DT)):
        th = amp * math.sin(w * k * DT + phase)
        res = tr.update(metro(th, slide, offset), arms(k) if callable(arms) else arms)
    return res, tr


class Kit(unittest.TestCase):
    def test_kit_files_identical(self):
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual((HERE / name).read_bytes(), (KIT_DIR / name).read_bytes(), name)


class Physics(unittest.TestCase):
    def test_scale_marks_ordered_and_on_the_rod(self):
        rs = [P.MARK_R[b] for b in L.MAELZEL]
        self.assertTrue(all(a > b for a, b in zip(rs, rs[1:])), "slower tempos sit higher")
        self.assertGreaterEqual(min(rs), L.W_R_MIN)
        self.assertLessEqual(max(rs), L.W_R_MAX)
        self.assertLess(L.PIVOT_Z + max(rs) + L.W_SIZE[2] / 2, L.CASE_H)

    def test_goal_and_start_beat_their_marks(self):
        self.assertAlmostEqual(P.bpm(P.R_GOAL, L.RUN_AMP), 100.0, delta=0.3)
        self.assertAlmostEqual(P.bpm(P.R_START, L.RUN_AMP), 160.0, delta=0.5)
        # any swing from 15 to 35 deg stays inside the full-marks band (the push holds it out at 27 deg)
        for amp in (0.26, 0.436, 0.61, T.PUSH_AMP):
            self.assertLess(abs(P.bpm(P.R_GOAL, amp) - 100.0), L.TEMPO_TOL)

    def test_one_mark_is_about_1p5_mm(self):
        self.assertAlmostEqual((P.MARK_R[96] - P.MARK_R[104]) * 1000, 3.1, delta=0.4)

    def test_stable_over_the_whole_travel(self):
        self.assertGreater(P.stiffness(L.W_R_MAX), 0.0)

    def test_clip_beats_the_metronome_weight(self):
        # the premise of the hold: dragged up, the clip's friction lifts the whole metronome unless it is held down
        self.assertGreater(L.W_CLIP, L.TOTAL_MASS * L.G * 1.5)
        self.assertAlmostEqual(L.BASE_MASS + L.ROD_LINK_MASS + L.W_MASS, L.TOTAL_MASS)

    def test_rod_above_weight_clears_the_gripper(self):
        # the rod passes up between the open fingers; the housing's lower face is about 70 mm over the TCP
        self.assertLess(L.ROD_Z1 - P.R_START, 0.055)

    def test_offline_swing_period_matches_formula(self):
        m = offline_model.MetronomeModel(T.METRONOME_PART)
        m.s = GOAL_SLIDE
        m.theta = 0.35
        crossings, prev = [], m.theta
        for k in range(200):
            m.step(FAR)
            if prev * m.theta < 0:
                crossings.append(k)
            prev = m.theta
        rate = 60.0 / (np.mean(np.diff(crossings)) * DT)
        self.assertAlmostEqual(rate, P.bpm(P.R_GOAL, 0.35), delta=1.5)


class Tiers(unittest.TestCase):
    def test_still_metronome_scores_zero(self):
        tr = S.Tracker()
        for _ in range(100):
            res = tr.update(metro(), FAR)
        self.assertEqual(res.score, 0.0)

    def test_ticking_at_100_hands_off_is_full(self):
        res, _ = swing(100, 0.35, 8.0)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.details["tempo_bpm"], 100.0, delta=0.5)

    def test_tempo_bands(self):
        for rate, want in ((103.5, 1.0), (107, 0.75), (92, 0.75), (126, 0.25), (160, 0.25), (85, 0.25)):
            slide = P.r_of_bpm(rate, 0.35) - P.R_START
            res, _ = swing(rate, 0.35, 10.0, slide=slide)
            self.assertEqual(res.score, want, rate)

    def test_set_but_not_started_is_half(self):
        tr = S.Tracker()
        for _ in range(60):
            res = tr.update(metro(0.0, GOAL_SLIDE), FAR)
        self.assertEqual(res.score, 0.5)
        res = tr.update(metro(0.0, GOAL_SLIDE + 0.004), FAR)
        self.assertEqual(res.score, 0.0)

    def test_small_swing_not_timed(self):
        res, _ = swing(100, 0.15, 10.0, slide=0.0)
        self.assertEqual(res.score, 0.0)

    def test_stopped_pendulum_not_timed(self):
        res, tr = swing(160, 0.35, 8.0, slide=0.0)
        self.assertEqual(res.score, 0.25)
        for _ in range(40):
            res = tr.update(metro(0.0, 0.0), FAR)
        self.assertEqual(res.score, 0.0)

    def test_hand_pumping_the_rod_does_not_count(self):
        near = {"left": FAR["left"], "right": arm(T.world((L.ROD_X, 0.0, 0.17)), opening=0.0)}
        res, _ = swing(100, 0.35, 10.0, slide=0.0, arms=near)
        self.assertEqual(res.score, 0.0)
        res, _ = swing(100, 0.35, 10.0, arms=near)  # weight set, but the hand is on it: only the set tier
        self.assertEqual(res.score, 0.5)

    def test_a_hand_near_on_one_tick_restarts_timing(self):
        def arms(k):
            if 60 <= k <= 64:
                return {"left": arm(T.world((0.0, 0.0, 0.17)), 0.0), "right": FAR["right"]}
            return FAR
        res, _ = swing(100, 0.35, 5.0, arms=arms)  # under 8 clean ticks after step 64
        self.assertEqual(res.score, 0.5)
        res, _ = swing(100, 0.35, 12.0, arms=arms)  # 8 clean ticks again later: full
        self.assertEqual(res.score, 1.0)

    def test_hand_between_ticks_only_does_not_count(self):
        # a hand that pumps the rod at the ends of its swing and is clear at every crossing
        def arms(k):
            phase = (2 * math.pi * 100 / 120.0 * k * DT + 0.3) % math.pi
            if abs(phase - math.pi / 2) < 0.4:
                return {"left": arm(T.world((0.0, 0.0, 0.17)), 0.0), "right": FAR["right"]}
            return FAR
        res, _ = swing(100, 0.35, 10.0, slide=0.0, arms=arms)
        self.assertEqual(res.score, 0.0)

    def test_moved_case_is_fine_but_held_up_or_fallen_is_not(self):
        res, _ = swing(100, 0.35, 8.0, offset=(0.0, 0.040, 0.0))  # set down 40 mm away: still ticking on its own
        self.assertEqual(res.score, 1.0)
        res, _ = swing(100, 0.35, 8.0, offset=(0.0, 0.0, 0.010))  # held 10 mm off the table the whole time
        self.assertEqual(res.score, 0.5)
        tilted = tuple(mk.matrix_to_quat(mk.axis_angle_matrix((1.0, 0.0, 0.0), 0.3) @ mk.quat_to_matrix(Q)))
        tr = S.Tracker()
        for k in range(int(8.0 / DT)):
            th = 0.35 * math.sin(2 * math.pi * 100 / 120.0 * k * DT + 0.3)
            res = tr.update(metro(th, GOAL_SLIDE, quat=tilted), FAR)
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.details["standing"])

    def test_lifted_then_set_down_counts_once_it_stands(self):
        tr = S.Tracker()
        for _ in range(20):
            tr.update(metro(0.0, 0.0, (0.0, 0.0, 0.020)), FAR)
        res, _ = swing(100, 0.35, 8.0, tracker=tr)
        self.assertEqual(res.score, 1.0)

    def test_unreadable_steps_never_full(self):
        res, tr = swing(100, 0.35, 8.0)
        self.assertEqual(res.score, 1.0)
        res = tr.update({}, FAR)
        self.assertLessEqual(res.score, 0.5)
        res = tr.update(metro(0.2, GOAL_SLIDE), None)  # arms unreadable: not hands off
        self.assertLess(res.score, 1.0)

    def test_every_tier_reachable(self):
        seen = {S.tier(t, b, h, s) for t in (False, True) for b in (None, 100.0, 106.0, 150.0)
                for h in (False, True) for s in (False, True)}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


class Demo(unittest.TestCase):
    def _check(self, **env):
        import check_task as C
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            rep, _, code = C.check(HERE, one_arm=not env, require_json=False)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return rep.data

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_demo_and_one_arm(self):
        d = self._check()
        self.assertEqual(d["playback"]["proxy"]["predicted_task_score"], 1.0)
        # right alone: the weight lifts the unheld case and stays at 160, where the push (made for a weight near the
        # 100 mark) passes under it: not started, 0
        self.assertEqual(d["one_arm"], {"left_only": 0.0, "right_only": 0.0})

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_counterfactuals(self):
        for variant, want in (("nohold", 0.0), ("short", 0.25), ("nostart", 0.5)):
            d = self._check(DEMO_VARIANT=variant)
            self.assertEqual(d["playback"]["proxy"]["predicted_task_score"], want, variant)
        d = self._check(DEMO_HOLD_ONLY="1")
        self.assertEqual(d["playback"]["proxy"]["predicted_task_score"], 0.0)

    def test_demo_points(self):
        # the right grip's pads stay off the case's front, the push's pad tips sit on the set weight's side face and
        # over the middle tier, the clear height passes over the rod's tip and the case
        self.assertGreater(L.TIERS[1][0] - (T.GRIP_X + 0.0135), 0.002)
        self.assertGreater(T.PUSH_TIP, L.PIVOT_Z + P.R_GOAL - L.W_SIZE[2] / 2)
        self.assertLess(T.PUSH_TIP, L.PIVOT_Z + P.R_GOAL + L.W_SIZE[2] / 2 - 0.008)
        self.assertGreaterEqual(T.PUSH_TIP - L.TIERS[1][4], 0.005)
        self.assertGreater(T.PUSH_HIGH - 0.013, L.PIVOT_Z + P.R_GOAL + L.W_SIZE[2] / 2 + 0.006)
        self.assertGreater(T.PUSH_CLEAR, T.PUSH_HIGH)
        self.assertAlmostEqual(T.WEIGHT_TRAVEL, P.R_GOAL - P.R_START)
        self.assertAlmostEqual(T.LIFT_DZ, T.WEIGHT_TRAVEL + L.DRAG_GIVE)

    def test_left_presses_the_case_down(self):
        # GPU 19 Nov: the pinch alone let the case rise 6.5 mm with the dragged weight; the left now presses it down
        # 10 mm (commanded) after the pinch closes and holds it there until it lets go, pads still over the table
        plan = T.left_moves()
        labels = [w.label for w in plan]
        i_pinch = labels.index("settle the pinch")
        i_open = labels.index("open")
        held = plan[i_pinch + 1:i_open + 1]
        self.assertGreaterEqual(T.PRESS_DZ, 0.006)
        for w in held:
            self.assertAlmostEqual(w.pos[2], L.TOP + T.HOLD_Z - T.PRESS_DZ, places=6)
        self.assertGreater(T.HOLD_Z - T.PRESS_DZ - 0.018, 0.006)  # the rolled pads' lowest corner 18 mm under the TCP

    def test_right_eases_off_before_opening(self):
        # the pull eased off inside the clip's stick band for any arm from 1 to 3.5 mm/N, before the pads open
        pull = L.W_CLIP + L.W_MASS * L.G
        for c in (0.001, 0.0022, 0.0035):
            left = pull - T.RELAX_DZ / c  # N still pulling once eased off
            self.assertLess(abs(left), L.W_CLIP - 0.5, c)
        labels = [w.label for w in T.right_moves()]
        self.assertLess(labels.index("ease the pull off (the clip holds)"), labels.index("let go of the weight"))

    def test_drag_trim_from_the_second_gpu_run(self):
        # GPU 19 Nov (second run): commanded 22.7 + 9.2 mm, the weight settled at 61.4 mm over the pivot, 5.0 mm short
        # of 66.4, and then held; the trim adds the measured shortfall
        self.assertAlmostEqual(L.DRAG_GIVE, 0.0092 + (P.R_GOAL - 0.0614), places=4)
        # the commanded top stays well inside the straight-down reach (about 225 mm over the table)
        self.assertLess(T.Z_START + T.LIFT_DZ, 0.16)


def _pad_box(tcp_b, yaw, rot=None):
    """The closed pads of a grip as base-frame sample points (x, y, z over the table): 27 mm wide, 16 mm thick,
    from their tips (13 mm past the TCP along the approach axis) to 9 mm above the TCP."""
    import yam_arena.demo as D
    r = np.asarray(getattr(yaw, "rotation", None) if getattr(yaw, "rotation", None) is not None
                   else D.grip_rotation(float(yaw)), float)
    tcp_w = np.asarray(T.world(tcp_b), float)
    pts = []
    for a in np.linspace(-0.008, 0.008, 3):  # closing axis: the gripper's x
        for b in np.linspace(-0.0135, 0.0135, 7):  # pad width
            for c in np.linspace(-0.009, 0.013, 6):  # along the approach axis (+ toward the tips)
                p = tcp_w + a * r[:, 0] + b * r[:, 1] + c * r[:, 2]
                pts.append(p)
    pts = np.array(pts)
    c, s = math.cos(L.M_YAW), math.sin(L.M_YAW)
    d = pts - np.array([L.M_X, L.M_Y, L.TOP])
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)


def _case_front_x(z):
    """Base x of the case's front (tiers, scale strip and marks) at height z over the table; None over its top."""
    if z > L.CASE_H:
        return None
    tier = next(t for t in L.TIERS if t[3] <= z <= t[4])
    strip = 0.0 if tier is L.TIERS[0] else T.P.SCALE_X + 0.0006
    return tier[0] - strip


class StartPush(unittest.TestCase):
    """GPU 19 Nov, second run: the rolled closed pads came down on the case's top instead of beside the rod (they
    stalled 18 mm over their command, tilted the case 2.5 deg and then tipped it 15 deg; the rod was never pushed, no
    timed ticks). The case had also slid 11.5 mm toward the right arm during the drag."""

    PUSH_LABELS = ("down beside the weight, in front of the case", "push the weight out toward the arms (world -x)",
                   "hold it out")

    def _push_waypoints(self):
        plan = T.right_moves()
        return [w for w in plan if w.label in self.PUSH_LABELS]

    def _base(self, w):
        p = np.asarray(w.pos, float) - np.array([L.M_X, L.M_Y, L.TOP])
        c, s = math.cos(L.M_YAW), math.sin(L.M_YAW)
        return (c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2])

    def test_old_rolled_push_reached_over_the_slid_case(self):
        # the failure, reproduced: the old pose (rolled 0.45 rad, TCP at base x -33 mm, 149 mm up) put the pads'
        # edge into the front of the case as it stood (slid 11.5 mm), under its top
        pts = _pad_box((L.ROD_X - 0.0035, -0.0135, 0.149), T.YAW_R)
        into = [p for p in pts if p[2] <= L.CASE_H and p[0] > _case_front_x(max(p[2], 0.0)) - 0.0115]
        self.assertTrue(into)

    def test_push_pads_clear_the_case_for_any_slide_up_to_13_mm(self):
        self.assertGreaterEqual(T.CASE_SHIFT, 0.0115 + 0.0015)
        for w in self._push_waypoints():
            self.assertEqual(getattr(w.yaw, "rotation", None), None, w.label)  # straight down, not rolled
            for shift in (0.0, -T.CASE_SHIFT):
                for p in _pad_box(self._base(w), w.yaw):
                    front = _case_front_x(p[2])
                    if front is not None:
                        self.assertLess(p[0], front + shift - 0.0025, (w.label, shift, p))

    def test_push_pads_cover_the_weight_not_the_rod(self):
        x_lo, x_hi = T.PUSH_X - T.PAD_HALF, T.PUSH_X + T.PAD_HALF
        for shift in (0.001, 0.0, -0.0115, -T.CASE_SHIFT):
            w0, w1 = L.ROD_X + L.W_X0 + shift, L.ROD_X + L.W_X0 + L.W_SIZE[0] + shift
            self.assertGreaterEqual(min(x_hi, w1) - max(x_lo, w0), 0.010, shift)
        self.assertLess(x_hi, L.ROD_X - L.ROD_D / 2 - 0.002)  # unslid, the pads pass in front of the rod

    def test_push_starts_clear_of_the_weight_and_holds_it_at_20_to_30_deg(self):
        # the pads' face 8 mm off the weight's side before the push (the rod rocked +-1.7 deg there on the GPU)
        self.assertGreaterEqual(-T.PUSH_Y0 - T.FACE - T.W_SIDE, 0.007)
        self.assertTrue(math.radians(20) <= T.PUSH_AMP <= math.radians(30))
        self.assertAlmostEqual(T.push_face(T.PUSH_AMP), T.PUSH_Y1 + T.FACE)
        # the case 2 mm either way along the push (GPU: 1.9 mm) keeps the hold angle within 23 to 31 deg
        for dx in (-0.002, 0.002):
            lo, hi = 0.0, 0.8
            for _ in range(60):
                m = (lo + hi) / 2
                lo, hi = (m, hi) if T.push_face(m) < T.PUSH_Y1 + T.FACE + dx else (lo, m)
            self.assertTrue(math.radians(23) <= lo <= math.radians(31), (dx, math.degrees(lo)))
        # the tip line meets the weight's side wherever it ends from 61.4 (GPU, short) to 68 mm over the pivot
        h = T.PUSH_TIP - L.PIVOT_Z
        s = (h - T.W_SIDE * math.sin(T.PUSH_AMP)) / math.cos(T.PUSH_AMP)
        for r in (0.0614, P.R_GOAL, 0.068):
            self.assertTrue(r - L.W_SIZE[2] / 2 <= s <= r + L.W_SIZE[2] / 2, r)
        # tempo at the hold angle and at 15 deg: inside the band at the mark
        for amp in (0.26, T.PUSH_AMP, T.PUSH_AMP + math.radians(4)):
            self.assertLess(abs(P.bpm(P.R_GOAL, amp) - 100.0), L.TEMPO_TOL - 1.0)

    def test_offline_push_on_a_slid_case_starts_a_full_swing(self):
        # the offline model plays the right hand's start through, the case slid as on the GPU (11.5 mm toward the
        # right arm, 1.9 mm along the push) and as built; the weight at the mark and 5 mm short of it
        plan = T.right_moves()
        i0 = [w.label for w in plan].index("wait for the left hand to clear")
        pts, t = [], 0.0
        prev = np.asarray(plan[i0].pos, float)
        for w in plan[i0 + 1:]:
            n = max(1, int(round(w.duration / DT)))
            for k in range(1, n + 1):
                pts.append((prev + (np.asarray(w.pos, float) - prev) * k / n, w.finger))
            prev = np.asarray(w.pos, float)
        for shift, slide in (((0.0, 0.0), GOAL_SLIDE), ((0.0019, -0.0115), GOAL_SLIDE), ((0.0019, -0.0115),
                                                                                           0.0614 - P.R_START)):
            m = offline_model.MetronomeModel(T.METRONOME_PART)
            m.origin = m.origin + np.array([shift[0], shift[1], 0.0])
            m.s = slide
            left = arm((0.200, 0.300, L.TOP + 0.160))
            peak_held, peak_free, released = 0.0, 0.0, False
            for p, finger in pts:
                opening = 0.0 if finger == T.CLOSED else 1.0
                m.step({"left": left, "right": arm(p, opening)})
                if m.side.get("right"):
                    peak_held = max(peak_held, abs(m.theta))
                elif peak_held > 0:
                    released = True
                    peak_free = max(peak_free, abs(m.theta))
            self.assertTrue(released, shift)
            self.assertTrue(math.radians(24) <= peak_held <= math.radians(31), (shift, math.degrees(peak_held)))
            self.assertTrue(math.radians(24) <= peak_free <= math.radians(31), (shift, math.degrees(peak_free)))


def _slider_run(cmd, k_arm, seconds, load=None):
    """A 1-D stand-in for the weight on its slider (PhysX's implicit drive): the clip's spring and damping toward the
    rule's target, written once per 50 ms control step, and an arm (a spring of stiffness k_arm(t) toward the
    commanded height cmd(t)), integrated implicitly in 12 substeps per step as PhysX does. Returns (t, q, arm pull)."""
    c = clip.Clip()
    q = v = 0.0
    m, K, D, dt = L.W_MASS, L.W_CLIP_K, L.W_CLIP_DAMPING, DT / 12
    load = load or (lambda t: -L.W_MASS * L.G)
    out = []
    for k in range(int(round(seconds / DT))):
        t0 = k * DT
        target = c.update(q)
        pulls = []
        for i in range(12):
            t = t0 + i * dt
            ka = k_arm(t)
            f = K * (target - q) + ka * (cmd(t) - q) + load(t)
            v = (v + dt / m * f) / (1 + dt * D / m + dt * dt * (K + ka) / m)
            q += dt * v
            pulls.append(ka * (cmd(t) - q))
        out.append((t0, q, float(np.mean(pulls))))
    return out


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class ClipRule(unittest.TestCase):
    """The weight's spring clip (clip.py). GPU 19 Nov: as PhysX joint friction the weight crept 0.8 mm down under its
    own 5 g in 11 s and 6.9 mm UP the rod while the pendulum swung, so the tempo ran from 136 to 125."""

    def test_no_physx_joint_friction_and_the_rule_is_registered(self):
        j = T.METRONOME_PART.link("weight")
        self.assertEqual(j.friction, 0.0)
        self.assertEqual(j.spring, L.W_CLIP_K)
        self.assertIn(T._clip_rule, list(mk._hub().step_fns.values()))
        self.assertAlmostEqual(clip.band(), 0.00075)
        self.assertLess(L.W_CLIP_LEAD_MAX, clip.band())  # a sliding weight is never pushed on by its own clip

    def test_free_weight_never_creeps(self):
        # its 5 g and a swing's along-rod loads (well under 0.03 N) for 60 s: it stays within 0.05 mm
        out = _slider_run(lambda t: 0.0, lambda t: 0.0, 60.0,
                          load=lambda t: -L.W_MASS * L.G + 0.03 * math.sin(2 * math.pi * 0.83 * t))
        self.assertLess(max(abs(q) for _, q, _ in out), 0.00005)

    def test_drag_slides_at_3n_and_stays_after_easing_off(self):
        top = 0.031
        for c in (0.001, 0.0022, 0.0035):  # the arm's give, m/N
            ka = 1.0 / c

            def cmd(t, top=top):
                if t < 1.0:
                    return 0.0
                if t < 6.0:
                    return top * _smooth((t - 1.0) / 5.0)
                if t < 6.5:
                    return top
                return top - T.RELAX_DZ * _smooth(t - 6.5)

            out = _slider_run(cmd, lambda t, ka=ka: ka if t < 8.5 else 0.0, 12.0)
            sliding = [f for t, _, f in out if 2.5 < t < 5.5]
            self.assertAlmostEqual(float(np.mean(sliding)), 3.1, delta=0.25, msg=c)
            stall = next(q for t, q, _ in out if t >= 6.45)
            eased = next(q for t, q, _ in out if t >= 8.4)
            end = out[-1][1]
            self.assertAlmostEqual(stall, top - 3.1 * c, delta=0.0005, msg=c)  # stalled by the arm's give
            self.assertAlmostEqual(end, stall - clip.band(), delta=0.0001, msg=c)  # then the clip's set, no more
            self.assertLessEqual(abs(end - eased), clip.band() + 1e-5)

    def test_pushed_down_slides_the_other_way_and_unreadable_keeps_the_target(self):
        c = clip.Clip()
        self.assertEqual(c.update(-0.002), -0.002 + clip.band())
        self.assertEqual(c.update(math.nan), -0.002 + clip.band())
        self.assertEqual(c.update(-0.0016), -0.002 + clip.band())  # back inside the band: held


if __name__ == "__main__":
    unittest.main()
