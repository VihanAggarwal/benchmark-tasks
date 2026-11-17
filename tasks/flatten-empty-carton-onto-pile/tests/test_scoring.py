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


def carton_at(phi=0.0, sign=1.0, dx=0.0, dy=0.0, dz=0.0, joints=None, tilt=0.0):
    """The carton racked ``phi`` rad toward +y (sign +1) or -y, moved by (dx, dy, dz), its base tilted ``tilt`` rad
    about x."""
    q = mk.quat_from_ypr(0.0, 0.0, tilt)
    return mk.state_from_poses(T.CARTON_PART, (L.CARTON_X + dx, L.CARTON_Y + dy, L.TOP + L.CARTON_ORIGIN_Z + dz, *q),
                               joints=joints if joints is not None else L.joints_for(phi, sign), source="test")


def flat_state(i, dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
    q = mk.quat_from_ypr(yaw, 0.0, 0.0)
    return mk.state_from_poses(T.FLAT_PARTS[i], (L.PILE_X + dx, L.PILE_Y + dy, L.TOP + L.flat_z(i) + dz, *q),
                               source="test")


def scene(carton=None, flats=None):
    out = {S.CARTON: carton if carton is not None else carton_at(math.pi / 2)}
    for i, n in enumerate(S.FLATS):
        out[n] = flats[i] if flats is not None else flat_state(i)
    return out


def start():
    return scene(carton_at(0.0))


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.LX, 0.1524, places=4)  # 6 in
        self.assertAlmostEqual(L.WA, 0.3048, places=4)  # 12 in
        self.assertAlmostEqual(L.WB, 0.2032, places=4)  # 8 in
        self.assertAlmostEqual(L.CT, 0.0032, places=4)  # 1/8 in C-flute
        self.assertAlmostEqual(L.FLAT_Y, 0.508, places=4)
        self.assertAlmostEqual(L.PT, 0.0222, places=4)
        self.assertAlmostEqual(L.CARTON_MASS, 0.085, places=3)

    def test_carton_stands_on_the_stack_and_is_203_mm_tall(self):
        st = carton_at(0.0)
        self.assertAlmostEqual(float(st.low[2]) - L.TOP, L.PT - L.CT / 2 - 0.0, delta=0.0017)
        self.assertAlmostEqual(float(st.high[2]) - float(st.low[2]), L.WB + 1.5 * L.CT + 0.0, delta=0.004)
        self.assertGreater(float(st.high[2]) - float(st.low[2]), 0.205)

    def test_stack_flats_rest_on_each_other(self):
        for i in range(L.FLAT_N):
            st = flat_state(i)
            self.assertAlmostEqual(float(st.low[2]) - L.TOP, L.BOARD_T + i * L.FLAT_T, places=6)
        self.assertAlmostEqual(float(flat_state(2).high[2]) - L.TOP, L.PT, places=6)

    def test_fully_racked_is_two_layers_on_the_stack(self):
        r = S.read(scene())
        self.assertAlmostEqual(S.thickness(r), 2 * L.CT, places=5)
        fx, fy = S.footprint(r)
        self.assertAlmostEqual(fx, L.LX, places=4)
        self.assertAlmostEqual(fy, 0.516, delta=0.002)
        c = S.plan_centre(r)
        self.assertLess(float(np.hypot(c[0] - L.PILE_X, c[1] - L.PILE_Y)), 0.005)
        self.assertAlmostEqual(float(np.min(r.pts[:, 2])), L.PT, delta=0.0017)
        self.assertEqual([round(S.interior(r.q[k])) for k in L.CREASES], [180, 0, 180, 0])

    def test_fully_racked_other_way_hangs_off_the_stack(self):
        r = S.read(scene(carton_at(math.pi / 2, sign=-1.0)))
        self.assertTrue(S.is_flat(r))
        self.assertGreater(float(np.hypot(*(S.plan_centre(r) - np.array([L.PILE_X, L.PILE_Y])))), 0.15)

    def test_start_is_standing_not_flat(self):
        r = S.read(start())
        self.assertFalse(S.is_flat(r))
        self.assertGreater(S.thickness(r), 0.2)
        self.assertEqual([round(S.interior(r.q[k])) for k in L.CREASES], [90, 90, 90, 90])

    def test_rack_geometry_is_a_four_bar(self):
        """Racked by phi the end panels stay parallel and the top panel stays level over the underneath one."""
        for phi in (0.3, 0.8, 1.2):
            st = carton_at(phi)
            rb, pb = st.link_pose("ab")
            rd, pd = st.link_pose("cd")
            self.assertAlmostEqual(float(rb[2, 2]), math.cos(phi), places=6)
            self.assertAlmostEqual(float(rd[2, 2]), math.cos(phi), places=6)
            rc, _ = st.link_pose("bc")
            self.assertAlmostEqual(float(rc[2, 2]), 1.0, places=6)

    def test_thickness_by_phi(self):
        self.assertLess(S.thickness(S.read(scene(carton_at(1.2)))), 0.110)  # over half way
        self.assertGreater(S.thickness(S.read(scene(carton_at(0.9)))), 0.110)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_tier_table(self):
        self.assertEqual(S.tier(False, True, True, True, True, True, 0.006), 1.0)
        self.assertEqual(S.tier(False, True, True, False, True, True, 0.006), 0.75)
        self.assertEqual(S.tier(False, True, True, False, False, True, 0.006), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, False, 0.006), 0.5)
        self.assertEqual(S.tier(False, True, False, False, False, True, 0.10), 0.25)
        self.assertEqual(S.tier(False, True, False, False, False, True, 0.20), 0.0)
        self.assertEqual(S.tier(True, True, True, True, True, True, 0.006), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True, True, 0.006), 0.0)

    def test_offset_on_the_stack_is_075(self):
        self.assertEqual(run(scene(carton_at(math.pi / 2, dy=0.09))).score, 0.75)

    def test_flat_on_the_table_beside_the_stack_is_05(self):
        res = run(scene(carton_at(math.pi / 2, dy=0.0, dx=-0.30, dz=-L.PT)))
        self.assertEqual(res.score, 0.5)

    def test_flat_but_high_on_one_side_is_075_or_less(self):
        res = run(scene(carton_at(math.pi / 2, dz=0.012)))  # 12 mm up: past 6, within 20
        self.assertEqual(res.score, 0.75)
        res = run(scene(carton_at(math.pi / 2, dz=0.030)))
        self.assertEqual(res.score, 0.5)

    def test_tilted_carton_is_not_flat(self):
        res = run(scene(carton_at(math.pi / 2, tilt=math.radians(4.0))))  # 516 mm long: 36 mm of rise
        self.assertFalse(res.details["flat"])

    def test_flat_on_a_pushed_stack_is_05(self):
        flats = [flat_state(0), flat_state(1), flat_state(2, dy=0.025)]
        self.assertEqual(run(scene(flats=flats)).score, 0.5)

    def test_wrong_way_racked_is_05(self):
        self.assertEqual(run(scene(carton_at(math.pi / 2, sign=-1.0))).score, 0.5)

    def test_over_half_way_is_025(self):
        self.assertEqual(run(scene(carton_at(1.2))).score, 0.25)

    def test_standing_and_leaning_score_zero(self):
        for phi in (0.0, 0.5, 0.9):
            self.assertEqual(run(scene(carton_at(phi))).score, 0.0, phi)

    def test_one_crease_open_is_not_flat(self):
        j = L.joints_for(math.pi / 2)
        j["bc"] = math.radians(60.0)
        res = run(scene(carton_at(joints=j)))
        self.assertFalse(res.details["flat"])
        self.assertLess(res.score, 0.5)

    def test_crease_window_edges(self):
        for deg, flat in ((14.0, True), (16.0, False)):
            j = L.joints_for(math.pi / 2)
            j["ab"] = -(math.pi / 2 - math.radians(deg))
            r = S.read(scene(carton_at(joints=j)))
            self.assertEqual(all(S.crease_flat(r.q[k]) for k in L.CREASES), flat, deg)

    def test_thickness_and_footprint_edges(self):
        r = S.read(scene())
        self.assertTrue(S.is_flat(r))
        self.assertLessEqual(S.thickness(r), L.THICK_MAX)
        self.assertLess(L.THICK_MAX, 0.016)

    def test_on_the_stack_edges(self):
        for dy, ok in ((0.055, True), (0.065, False)):
            r = S.read(scene(carton_at(math.pi / 2, dy=dy)))
            self.assertEqual(S.on_stack(r), ok, dy)
        for dz, ok in ((0.005, True), (0.008, False)):
            r = S.read(scene(carton_at(math.pi / 2, dz=dz)))
            self.assertEqual(S.on_stack(r), ok, dz)

    def test_stack_kept_edges(self):
        for d, ok in ((0.014, True), (0.016, False)):
            r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, dx=d)]))
            self.assertEqual(S.stack_kept(S.stack_moves(r)), ok, d)
        r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, yaw=math.radians(9.0))]))
        self.assertFalse(S.stack_kept(S.stack_moves(r)))
        r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, dz=0.005)]))
        self.assertFalse(S.stack_kept(S.stack_moves(r)))


class Exploits(unittest.TestCase):
    def test_strip_laid_out_straight_is_not_flat(self):
        """All four creases opened to 180 degrees lays the carton out as a 1 m strip: flat in its creases, but over
        the footprint limit, so never 1.0 and never 'flat'."""
        j = {k: -math.pi / 2 for k in L.CREASES}
        res = run(scene(carton_at(joints=j)))
        self.assertFalse(res.details["flat"])
        self.assertLessEqual(res.score, 0.25)

    def test_carton_carried_in_the_hand_scores_zero(self):
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.02), 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=hand).score, 0.0)

    def test_sweeping_the_stack_away_scores_zero(self):
        flats = [flat_state(i, dy=0.06 + 0.01 * i) for i in range(L.FLAT_N)]
        res = run(scene(flats=flats))
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["KNOCKED"])

    def test_knock_is_latched(self):
        tr = S.Tracker()
        tr.update(scene(flats=[flat_state(0, dy=0.06), flat_state(1), flat_state(2)]), FAR)
        res = run(scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)

    def test_pile_turned_over_30_degrees_latches(self):
        flats = [flat_state(0, yaw=math.radians(35.0)), flat_state(1), flat_state(2)]
        self.assertEqual(run(scene(flats=flats)).score, 0.0)

    def test_stack_slid_out_from_under_the_carton(self):
        """Pulling the flats out and letting the carton drop to the board: flat, but the stack moved: at best 0.5,
        and 0 once any flat moved 40 mm."""
        flats = [flat_state(i, dy=0.30) for i in range(L.FLAT_N)]
        self.assertEqual(run(scene(carton_at(math.pi / 2, dz=-L.PT + L.BOARD_T), flats=flats)).score, 0.0)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_carton(self):
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.04), 1.0), "right": FAR["right"]}
        res = run(scene(), arms=hand)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_hand_clear(self):
        self.assertEqual(run(scene()).score, 1.0)

    def test_parked_hands_are_clear(self):
        parked = {"left": arm((*T.PARK_L[:2], L.TOP + T.PARK_L[2])), "right": arm((*T.PARK_R[:2], L.TOP + T.PARK_R[2]))}
        self.assertEqual(run(scene(), arms=parked).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.04), 1.0), "right": FAR["right"]}
        tr.update(scene(), hand)
        res = run(scene(), tracker=tr, steps=WIN - 1)
        self.assertEqual(res.score, 0.0)
        res = run(scene(), tracker=tr, steps=1)
        self.assertEqual(res.score, 1.0)  # the hand step has left the window

    def test_moving_carton(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(scene(carton_at(math.pi / 2, dy=0.001 * (i % 2) * 2)), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_creases_still_moving(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            j = L.joints_for(math.pi / 2)
            j["bc"] += math.radians(0.5) * (i % 2)
            res = tr.update(scene(carton_at(joints=j)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_still_stack_flat(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(scene(flats=[flat_state(0), flat_state(1), flat_state(2, dy=0.0015 * (i % 2))]), FAR)
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        for name in S.PARTS:
            s = scene()
            del s[name]
            self.assertEqual(run(s).score, 0.0, name)

    def test_nan_state(self):
        st = carton_at(math.pi / 2)
        bad = mk.state_from_poses(T.CARTON_PART, (float("nan"), 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), source="test")
        res = run(scene(carton=bad))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertIsNotNone(st)

    def test_nan_joint(self):
        j = L.joints_for(math.pi / 2)
        j["cd"] = float("nan")
        self.assertEqual(run(scene(carton_at(joints=j))).score, 0.0)

    def test_inf_flat(self):
        flats = [flat_state(0), flat_state(1), flat_state(2, dy=float("inf"))]
        self.assertEqual(run(scene(flats=flats)).score, 0.0)

    def test_blow_up_restarts_the_windows(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.CARTON_PART, (float("nan"), 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), source="test")
        self.assertEqual(tr.update(scene(carton=bad), FAR).score, 0.0)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(scene(flats=[flat_state(0, dy=0.06), flat_state(1), flat_state(2)]), FAR)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(200):
            j = {k: float(rng.uniform(-1.7, 1.7)) for k in L.CREASES}
            c = carton_at(joints=j, dx=float(rng.normal(0, 0.1)), dy=float(rng.normal(0, 0.2)),
                          dz=float(rng.normal(0, 0.05)), tilt=float(rng.normal(0, 0.2)))
            flats = [flat_state(i, dx=float(rng.normal(0, 0.02)), dy=float(rng.normal(0, 0.02))) for i in range(3)]
            res = tr.update(scene(c, flats), FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["board", "flat_1", "flat_2", "flat_3", "carton"])
        self.assertIsNone(T.BOARD.mass)
        self.assertEqual(T.SCORE_OVER, "end")
        self.assertEqual(T.SUCCESS_HOLD_S, 1.0)

    def test_instruction_states_the_goal_only(self):
        t = T.INSTRUCTION.lower()
        for word in ("push", "rack", "crease", "press", "left arm", "right arm", "degrees"):
            self.assertNotIn(word, t)
        self.assertLess(len(T.INSTRUCTION), 300)

    def test_demo_plans(self):
        p = T.plans()
        self.assertEqual(set(p), {"left", "right"})
        for side in p:
            self.assertGreater(len(p[side]), 3)

    def test_demo_env_switches(self):
        old = {k: os.environ.pop(k, None) for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT")}
        try:
            self.assertEqual(set(T.demo()), {"left", "right"})
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(set(T.demo()), {"right"})
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(set(T.demo()), {"left"})
            del os.environ["DEMO_IDLE_ARM"]
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            del os.environ["DEMO_HOLD_ONLY"]
            os.environ["DEMO_VARIANT"] = "shallow"
            self.assertEqual(set(T.demo()), {"left", "right"})
            os.environ["DEMO_VARIANT"] = "nonsense"
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
                os.environ.pop(k, None)
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    def test_demo_timing(self):
        p = T.plans()
        total = max(T._duration(p[s]) for s in p)
        self.assertLess(total, 40.0)
        self.assertLess(total * HZ + 3 * HZ, 1200)  # the demo plus the policy's 3 s hold fits the run limit
        self.assertGreater(T._duration(p["left"]), T._duration(p["right"]))

    def test_push_geometry(self):
        """The closed block starts in the air short of the end panel's foot and the push ends before the carton's
        +y end; the end panel is 203 mm tall and the block pushes at 120 mm."""
        self.assertLess(T.PUSH_Y0 + T.PAD_FACE, T.D_FACE_Y)
        self.assertLess(T.PUSH_Y1, L.A_YMAX)
        self.assertLess(T.PUSH_HZ, L.WB)
        self.assertGreater(T.PUSH_PHI, T.GRAVITY_PHI)
        self.assertLess(T.PUSH_PHI, 1.2)
        self.assertGreater(T.PUSH_Y0 - 0.0, -0.40)  # inside the right arm's own column (base y -305, +-100)

    def test_press_geometry(self):
        """The block's pad tips end 1 mm inside the flat carton's top (6.4 mm over the stack's top)."""
        tips = T.PRESS_Z - T.PAD_TIP
        self.assertAlmostEqual(tips, L.PT + 2 * L.CT - 0.001, places=6)
        for y in T.PRESS_R:
            self.assertLessEqual(y, 0.05)  # the right arm reaches only 0.1 m past the midline, and low
        for y in T.PRESS_L:
            self.assertGreaterEqual(y, 0.05)
        self.assertLess(T.PRESS_Z, 0.065)  # the right arm's table-height reach at x 0.3, y 0 to 0.1 past the midline

    def test_hands_apart(self):
        p = T.plans()
        t_right_done = T._duration(p["right"])
        self.assertGreater(t_right_done, 0.0)
        # the left arm waits parked until the right arm is on its last waypoint
        self.assertGreaterEqual(p["left"][0].duration, t_right_done - 2.0 - 1e-6)


class OfflineModel(unittest.TestCase):
    def feed(self, tcps, opening=0.0, steps=1):
        m = T.OfflineModel()
        for _ in range(steps):
            for tcp in tcps:
                arms = {"left": arm((0.1, 0.5, L.TOP + 0.25)), "right": arm(tcp, opening)}
                rigstate.offline_feed({a: {"tcp": arms[a].tcp, "tips": arms[a].tips, "opening": arms[a].opening}
                                       for a in arms})
                m.step(rigstate.arms())
        return m

    def test_push_leans_the_carton(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.5)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.5, places=3)

    def test_weight_finishes_a_deep_push(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.9)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)], steps=60)
        self.assertAlmostEqual(m.phi, math.pi / 2, places=3)

    def test_shallow_push_stays_leaning(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.5)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)], steps=60)
        self.assertAlmostEqual(m.phi, 0.5, places=3)

    def test_open_gripper_does_not_push(self):
        m = self.feed([(L.CARTON_X, -0.15, L.TOP + L.PT + 0.12)], opening=1.0, steps=5)
        self.assertEqual(m.phi, 0.0)

    def test_high_gripper_does_not_push(self):
        m = self.feed([(L.CARTON_X, -0.15, L.TOP + L.PT + 0.25)], steps=5)
        self.assertEqual(m.phi, 0.0)

    def test_off_to_the_side_does_not_push(self):
        m = self.feed([(L.CARTON_X + 0.15, -0.15, L.TOP + L.PT + 0.12)], steps=5)
        self.assertEqual(m.phi, 0.0)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].pos, T.SPECS[n].rest_state().pos, atol=1e-6), n)
        self.assertEqual(S.Tracker.at_start(S.read(st)), True)

    def test_phi_never_decreases(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.6)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.6, places=3)
        self.feed([(L.CARTON_X, y - 0.05, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.6, places=3)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("508 (y) x 152 (x) x 6.4 mm", "0.085 kg", "x 270 mm", "540 x 175 x 3 mm", "22.2 mm up",
                      "12 x 8 x 6 in (305 x 203 x 152 mm)", "203 mm tall", "y -254 to +51 mm", "0.01 N m",
                      "97 degrees",
                      "glued flat to the underneath panel", "516 mm long", "6.4 mm thick", "flaps already cut off"):
            self.assertIn(token, setup)
        for token in ("15 degrees", "12 mm or less", "540 mm (y) x 180 mm (x)", "60 mm", "6 mm",
                      "120 mm and 20 mm", "15 mm", "4 mm up", "8 degrees", "30 mm", "1 mm",
                      "0.3 degrees", "40 mm", "30 degrees", "110 mm tall", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0",
                      "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        self.assertAlmostEqual(L.PILE_X, 0.270)
        self.assertAlmostEqual(L.A_YMIN, -0.254, places=3)
        self.assertAlmostEqual(L.A_YMAX, 0.0508, places=3)
        self.assertAlmostEqual(L.PT, 0.0222, places=4)
        self.assertEqual(L.THICK_MAX, 0.012)
        self.assertEqual((L.FOOT_Y, L.FOOT_X), (0.540, 0.180))
        scores = d.get("scores", "")  # before the GPU run: empty or TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_simplifications(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("0.01 N m", "never peels", "pass through one another", "est"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Stockers and order fillers", "53-7065", "BLS"):
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
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("Offline prediction", text)
        self.assertIn("Simulator (Isaac Lab-Arena", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
