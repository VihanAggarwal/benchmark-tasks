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
PUSH = 0.0025  # 1.25 N
BUTTON = L.BTN_TRAVEL


def head_at(push=0.0, head_w=0.0):
    return mk.state_from_poses(T.HEAD_PART, (L.FX, L.FY, L.TOP, 0.0, 0.0, 0.0, 1.0), joints={S.NECK: push / L.H_ARM},
                               joint_vel={S.NECK: head_w}, source="test")


def probe_at(x, y, z, yaw=0.0, button=0.0, tilt=0.0, vel=(0.0, 0.0, 0.0), spin=0.0):
    """The scanner with its cup's centre at (x, y, z above the table), its body turned ``yaw`` (radians) and rolled
    ``tilt`` degrees about its own x (off standing on its edge)."""
    q = mk.quat_from_ypr(yaw, 0.0, math.radians(tilt))
    return mk.state_from_poses(T.PROBE_PART, (x, y, L.TOP + z, *q), base_vel=(*vel, 0.0, 0.0, spin),
                               joints={"button": button}, source="test")


def on_forehead(s, push=PUSH, button=BUTTON, dz=0.0, turn=0.0, lift=0.0, head_w=0.0, follow=True):
    """The cup ``s`` m round the forehead's curve from its centre, ``lift`` off the skin, ``dz`` off the scan line,
    facing the skin's normal (``turn`` degrees off it), the head pushed back ``push``."""
    phi = s / L.RF
    h = head_at(push, head_w)
    r = L.RF + lift
    drawn = np.array([-r * math.cos(phi), -r * math.sin(phi), L.Z_LINE + dz])
    p = h.link_to_world(S.NECK, drawn) - np.array([0.0, 0.0, L.TOP])
    yaw = (phi if follow else 0.0) - L.CUP_TILT + math.radians(turn)
    return {S.PROBE: probe_at(p[0], p[1], p[2], yaw, button), S.HEAD: h}


def in_rest(button=0.0, push=0.0, head_w=0.0, vel=(0.0, 0.0, 0.0), spin=0.0, dx=0.0, dz=0.0, yaw=0.0, tilt=0.0):
    return {S.PROBE: probe_at(L.PR_X0 + dx, L.PR_Y0, L.PR_Z0 + dz, yaw, button, tilt, vel, spin), S.HEAD: head_at(push, head_w)}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


class Rig:
    def __init__(self):
        self.tr = S.Tracker()
        self.res = None

    def put(self, states, arms=FAR, n=1):
        for _ in range(n):
            self.res = self.tr.update(states, arms)
        return self.res

    def sweep(self, seconds=2.5, s0=0.0, s1=0.054, **kw):
        n = max(1, int(round(seconds * HZ)))
        for i in range(n + 1):
            self.put(on_forehead(s0 + (s1 - s0) * i / n, **kw))

    def settle_on(self, s=0.0, **kw):
        self.put(on_forehead(s, lift=0.020, push=0.0, button=BUTTON), n=5)
        self.put(on_forehead(s, **kw), n=12)

    def carry_home(self):
        start = on_forehead(0.054, lift=0.030, push=0.0)[S.PROBE].origin - np.array([0.0, 0.0, L.TOP])
        end = np.array([L.PR_X0, L.PR_Y0, L.PR_Z0])
        n = int(math.ceil(float(np.linalg.norm(end - start)) / 0.02))
        for i in range(1, n + 1):
            p = start + (end - start) * i / n
            self.put({S.PROBE: probe_at(*p, 0.0, BUTTON), S.HEAD: head_at(0.0)})
        self.put(in_rest(), n=WIN + 3)

    def full(self, seconds=2.5, **kw):
        self.settle_on()
        self.sweep(seconds, **kw)
        self.carry_home()
        return self.res


class Geometry(unittest.TestCase):
    def test_datasheet_size_and_mass(self):
        self.assertAlmostEqual(L.PR_MASS, 0.213)
        self.assertAlmostEqual(L.BODY_W, 0.050)
        self.assertAlmostEqual(L.BODY_X[1] - L.BODY_X[0], 0.027, places=6)
        self.assertAlmostEqual(-L.BODY_Y[0] + L.CUP_R, 0.200, places=6)  # dome top to the bottom end

    def test_head_gives_half_a_newton_per_mm(self):
        k = (T.NECK_SPRING - L.HEAD_LINK_MASS * 9.81 * (T._COM[2] - L.NECK_Z)) / L.H_ARM ** 2
        self.assertAlmostEqual(k, 500.0, places=6)
        self.assertAlmostEqual(T.NECK_SPRING, 14.0, delta=0.05)

    def test_parts_spawn_where_they_rest(self):
        st = {n: T.SPECS[n].rest_state() for n in S.PARTS}
        r = S.read(st)
        self.assertTrue(S.put_back(r))
        self.assertAlmostEqual(r.yaw, math.degrees(L.CUP_TILT), places=6)
        self.assertAlmostEqual(r.push, 0.0)

    def test_scored_end_is_short_of_the_hairline(self):
        end_phi = L.S_FINISH / L.RF
        self.assertLess(end_phi + L.CUP_R / L.RF, L.HAIR_PHI)  # the cup's edge has not reached the hair
        self.assertLess(T.PHI_END + L.CUP_R / L.RF, L.HAIR_PHI)
        self.assertAlmostEqual(math.degrees(end_phi), 38.2, places=1)

    def test_scan_line_between_brows_and_hairline(self):
        self.assertLess(L.Z_BROW + L.LINE_TOL, L.Z_LINE + 1e-9)
        self.assertLess(L.Z_LINE + L.LINE_TOL, L.Z_HAIR)

    def test_grip_is_on_the_button(self):
        self.assertEqual(L.GRIP_LOCAL[1], L.BTN_Y)
        self.assertAlmostEqual(L.GRIP_LOCAL[0], (L.BODY_X[0] + L.BODY_X[1]) / 2)
        self.assertLess(T.GRIP_PR * -2, L.BODY_X[1] - L.BODY_X[0])  # commanded narrower than the body
        self.assertGreater(T.OPEN_PR * -2, T.GRIP_W + 0.030)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = Rig().full()
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_start_scores_zero(self):
        rig = Rig()
        res = rig.put(in_rest(), n=60)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_speed_window(self):
        self.assertEqual(Rig().full(seconds=3.6).score, 1.0)
        self.assertEqual(Rig().full(seconds=1.0).score, 0.5)  # too fast
        self.assertEqual(Rig().full(seconds=6.0).score, 0.5)  # too slow

    def test_clean_pass_not_put_back(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        res = rig.put(on_forehead(0.054, lift=0.030, push=0.0), n=40)
        self.assertEqual(res.score, 0.75)

    def test_hand_still_on_the_scanner(self):
        rig = Rig()
        rig.full()
        tcp = np.array([L.PR_X0, L.PR_Y0, L.TOP + L.PR_Z0]) + T.rot_z(0.0) @ np.asarray(L.GRIP_LOCAL)
        holding = {"left": FAR["left"], "right": arm(tcp, 0.2)}
        res = rig.put(in_rest(), holding, n=WIN + 2)
        self.assertEqual(res.score, 0.75)

    def test_contact_broken_mid_slide(self):
        rig = Rig()
        rig.settle_on()
        n = 50
        for i in range(n + 1):
            s = 0.054 * i / n
            lift = 0.004 if 20 <= i <= 24 else 0.0
            rig.put(on_forehead(s, lift=lift, push=0.0 if lift else PUSH))
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.5)

    def test_lifted_more_than_6mm_voids_the_pass(self):
        rig = Rig()
        rig.settle_on()
        n = 50
        for i in range(n + 1):
            lift = 0.008 if 20 <= i <= 24 else 0.0
            rig.put(on_forehead(0.054 * i / n, lift=lift, push=0.0 if lift else PUSH))
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_push_band(self):
        for push, touch in ((0.0009, False), (0.0010, True), (0.0040, True), (0.0041, False)):
            r = S.read(on_forehead(0.02, push=push))
            self.assertEqual(S.touching(r), touch, push)
        self.assertEqual(Rig().full(push=0.0045).score, 0.5)  # a pass, out of the band throughout

    def test_flush_limit(self):
        self.assertTrue(S.touching(S.read(on_forehead(0.02, turn=9.5))))
        self.assertFalse(S.touching(S.read(on_forehead(0.02, turn=10.5))))
        self.assertFalse(S.touching(S.read(on_forehead(0.02, turn=-10.5))))

    def test_not_turning_round_the_curve_voids_the_pass(self):
        res = Rig().full(follow=False)
        self.assertEqual(res.score, 0.25)  # flush only near the centre

    def test_button_not_held_scores_zero(self):
        rig = Rig()
        rig.settle_on(button=0.0)
        rig.sweep(button=0.0)
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.0)

    def test_button_let_go_mid_slide(self):
        rig = Rig()
        rig.settle_on()
        n = 50
        for i in range(n + 1):
            rig.put(on_forehead(0.054 * i / n, button=0.0 if i == 25 else BUTTON))
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_button_threshold(self):
        self.assertTrue(S.button_held(S.read(on_forehead(0.0, button=0.0015))))
        self.assertFalse(S.button_held(S.read(on_forehead(0.0, button=0.0014))))

    def test_off_the_line(self):
        self.assertTrue(S.touching(S.read(on_forehead(0.02, dz=0.0079))))
        self.assertFalse(S.touching(S.read(on_forehead(0.02, dz=-0.0081))))
        rig = Rig()
        rig.settle_on(dz=-0.012)
        rig.sweep(dz=-0.012)
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.0)  # never touching: 12 mm below the line, toward the brows

    def test_touch_time_is_a_quarter(self):
        rig = Rig()
        rig.put(on_forehead(0.020), n=int(L.TOUCH_S * HZ))
        self.assertEqual(rig.res.score, 0.25)
        rig2 = Rig()
        rig2.put(on_forehead(0.020), n=int(L.TOUCH_S * HZ) - 1)
        self.assertEqual(rig2.res.score, 0.0)

    def test_shove_latches_zero(self):
        rig = Rig()
        rig.settle_on()
        rig.put(on_forehead(0.0, push=0.008))
        rig.sweep()
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.0)
        self.assertIsNotNone(rig.res.details["shove_step"])

    def test_push_below_shove_is_not_latched(self):
        rig = Rig()
        rig.put(on_forehead(0.0, push=0.0079))
        self.assertIsNone(rig.res.details["shove_step"])

    def test_head_must_be_back_and_still(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        rig.carry_home()
        res = rig.put(in_rest(push=0.0015), n=WIN)
        self.assertEqual(res.score, 0.75)
        res = rig.put(in_rest(head_w=0.05), n=WIN)
        self.assertEqual(res.score, 0.75)
        res = rig.put(in_rest(), n=WIN)
        self.assertEqual(res.score, 1.0)

    def test_release_must_hold_the_whole_second(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        rig.carry_home()
        rig.put(in_rest(vel=(0.0, 0.0, 0.02)))
        res = rig.put(in_rest(), n=WIN - 1)
        self.assertEqual(res.score, 0.75)
        self.assertEqual(rig.put(in_rest()).score, 1.0)

    def test_unreadable_arms_never_let_go(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        rig.carry_home()
        res = rig.put(in_rest(), None, n=WIN + 2)
        self.assertEqual(res.score, 0.75)


class PutBackBounds(unittest.TestCase):
    """The put-back check is bounded on every side and on orientation (a scanner near the rest is not in it)."""

    def test_inside(self):
        self.assertTrue(S.put_back(S.read(in_rest(dx=0.009))))
        self.assertTrue(S.put_back(S.read(in_rest(dz=0.0039))))

    def test_outside_each_side(self):
        for kw in ({"dx": 0.011}, {"dx": -0.011}, {"dz": 0.0045}, {"dz": -0.0045}):
            self.assertFalse(S.put_back(S.read(in_rest(**kw))), kw)

    def test_held_just_above_the_rest_is_not_put_back(self):
        self.assertFalse(S.put_back(S.read(in_rest(dz=0.012))))

    def test_turned_or_lying_is_not_put_back(self):
        self.assertTrue(S.put_back(S.read(in_rest(yaw=math.radians(14.0)))))
        self.assertFalse(S.put_back(S.read(in_rest(yaw=math.radians(16.0)))))
        self.assertFalse(S.put_back(S.read(in_rest(yaw=math.pi))))
        self.assertFalse(S.put_back(S.read(in_rest(tilt=12.0))))
        self.assertFalse(S.put_back(S.read(in_rest(tilt=90.0))))

    def test_hovering_over_the_rest_never_pays(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        res = rig.put(in_rest(dz=0.012), n=WIN + 5)
        self.assertEqual(res.score, 0.75)


class Exploits(unittest.TestCase):
    def test_starting_halfway_is_not_a_pass(self):
        rig = Rig()
        rig.put(on_forehead(0.020), n=12)
        rig.sweep(s0=0.020)
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_sliding_the_wrong_way_is_not_a_pass(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep(s0=0.0, s1=-0.010)
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_back_and_forth_does_not_make_a_pass(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep(1.5, 0.0, 0.030)
        rig.sweep(0.5, 0.030, 0.020)
        rig.sweep(1.5, 0.020, 0.054)
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_a_jump_to_the_end_is_not_clean(self):
        rig = Rig()
        rig.settle_on()
        rig.put(on_forehead(0.004))
        rig.put(on_forehead(0.052))
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.5)  # timed 0.05 s

    def test_pressing_with_the_body_is_not_touching(self):
        """The head pushed in the band while the cup is off the skin (the body leaning on the face) is not touching."""
        r = S.read(on_forehead(0.010, lift=0.004))
        self.assertFalse(S.touching(r))

    def test_pushing_the_head_with_a_hand_is_not_touching(self):
        """The other gripper pushes the head back while the cup hovers 2 mm off the skin: never touching."""
        st = on_forehead(0.0, lift=0.0019)
        self.assertTrue(S.touching(S.read(st)))  # the reading alone would pass
        h = st[S.HEAD]
        brow = h.link_to_world(S.NECK, np.array([-L.RF * math.cos(0.6), L.RF * math.sin(0.6), L.Z_LINE]))
        out = np.array([-math.cos(0.6), math.sin(0.6), 0.0])
        for clear, expect in ((0.0, False), (L.HANDS_OFF_HEAD - 0.001, False), (L.HANDS_OFF_HEAD + 0.001, True)):
            tcp = brow + out * 0.08
            for _ in range(60):  # move the TCP out until the finger zone is ``clear`` from the head
                a = arm(tcp, 0.3)
                g = rigstate.zone_gap(a, h.distance)
                if abs(g - clear) < 2e-4:
                    break
                tcp = tcp + out * (clear - g) * 0.9
            arms = {"left": a, "right": FAR["right"]}
            rig = Rig()
            res = rig.put(st, arms, n=12)
            self.assertEqual(res.details["touching"], expect, clear)
            self.assertEqual(res.score, 0.25 if expect else 0.0, clear)

    def test_unreadable_arms_are_never_touching(self):
        rig = Rig()
        self.assertEqual(rig.put(on_forehead(0.0), None, n=12).score, 0.0)

    def test_down_the_side_of_the_face_is_not_on_the_line(self):
        r = S.read(on_forehead(0.050, dz=-0.030))
        self.assertFalse(S.pass_step_ok(r))


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        rig = Rig()
        self.assertEqual(rig.put({S.PROBE: None, S.HEAD: head_at()}).score, 0.0)
        self.assertEqual(rig.put({}).score, 0.0)
        self.assertEqual(rig.put(None).score, 0.0)

    def test_nan_state(self):
        rig = Rig()
        st = on_forehead(0.0)
        bad = mk.state_from_poses(T.PROBE_PART, (math.nan, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), joints={"button": 0.0}, source="test")
        self.assertEqual(rig.put({S.PROBE: bad, S.HEAD: st[S.HEAD]}).score, 0.0)
        self.assertFalse(rig.res.details["valid_state"])

    def test_nan_mid_slide_restarts_the_pass(self):
        rig = Rig()
        rig.settle_on()
        n = 50
        bad = mk.state_from_poses(T.PROBE_PART, (math.nan, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), joints={"button": 0.0}, source="test")
        for i in range(n + 1):
            if i == 25:
                rig.put({S.PROBE: bad, S.HEAD: head_at()})
            else:
                rig.put(on_forehead(0.054 * i / n))
        rig.carry_home()
        self.assertEqual(rig.res.score, 0.25)

    def test_nan_in_the_last_second_breaks_the_hold(self):
        rig = Rig()
        rig.full()
        rig.put(None)
        self.assertEqual(rig.put(in_rest(), n=WIN - 1).score, 0.75)
        self.assertEqual(rig.put(in_rest()).score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        rig = Rig()
        rig.settle_on()
        rig.sweep()
        rig.put(on_forehead(0.054, lift=0.03, push=0.0))
        res = rig.put(in_rest(), n=WIN + 3)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(rig.tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(3)
        rig = Rig()
        for _ in range(300):
            s = float(rng.uniform(-0.03, 0.08))
            st = on_forehead(s, push=float(rng.uniform(-0.002, 0.012)), button=float(rng.uniform(0, 0.002)),
                             dz=float(rng.uniform(-0.02, 0.02)), turn=float(rng.uniform(-30, 30)),
                             lift=float(rng.uniform(-0.002, 0.02)))
            res = rig.put(st)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_possible_scores_only(self):
        tr = S.Tracker()
        for attrs in ({}, {"touch_steps": 100}, {"passed": 5}, {"clean": 5}):
            tr.reset()
            for k, v in attrs.items():
                setattr(tr, k, v)
            self.assertIn(tr.tier(), (0.0, 0.25, 0.5, 0.75, 1.0))


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(sorted(names), sorted(["head", "rest", "probe"]))
        self.assertTrue(T.HEAD_PART.fixed_base)
        self.assertFalse(T.PROBE_PART.fixed_base)
        self.assertLess(T.PROBE_PART.mass, 0.4)

    def test_instruction_matches_the_json(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(T.INSTRUCTION.count("."), 1)  # one goal sentence

    def test_demo_plans(self):
        for v in ("",) + T.VARIANTS:
            p = T.plans(v)
            self.assertEqual(list(p), ["right"])
            self.assertTrue(all(w.duration > 0 for w in p["right"]))
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        old = dict(os.environ)
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            del os.environ["DEMO_HOLD_ONLY"]
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_demo_fits_the_run_limit(self):
        total = T._duration(T.plans()["right"])
        self.assertLess(total * HZ + 80, 1200)

    def test_sweep_speed_in_the_window(self):
        v = L.RF * T.PHI_END / T.SWEEP_S
        t = (L.S_FINISH - L.S_BEGIN) / v
        self.assertTrue(L.T_MIN < t < L.T_MAX)

    def test_cup_flush_along_the_planned_slide(self):
        """At every sweep waypoint the planned cup faces the skin's normal (the wrist turns with the curve)."""
        for w in T.plans()["right"]:
            if not w.label.startswith("slide round"):
                continue
            psi = w.yaw - math.pi / 2
            lens = np.array(w.pos) - np.array([0.0, 0.0, L.TOP]) - T.rot_z(psi) @ np.asarray(L.GRIP_LOCAL)
            phi = math.atan2(-(lens[1] - L.FY), -(lens[0] - L.FX))
            self.assertAlmostEqual(psi + L.CUP_TILT, phi, places=6)
            self.assertAlmostEqual(math.hypot(lens[0] - L.FX, lens[1] - L.FY), L.RF - T.PEN, places=6)


class OfflineModel(unittest.TestCase):
    def setUp(self):
        self.m = T.OfflineModel()

    def arms(self, right=None):
        return {"left": FAR["left"], "right": arm(right[0], right[1]) if right else FAR["right"]}

    def test_states_at_rest_match_the_spawn(self):
        self.m.step(self.arms())
        st = self.m.states()
        for n in S.PARTS:
            np.testing.assert_allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-9)

    def test_grip_at_the_button_presses_it(self):
        tcp = T.tcp_for((L.PR_X0, L.PR_Y0), L.PR_Z0, 0.0)
        self.m.step(self.arms(((tcp[0], tcp[1], L.TOP + tcp[2]), 0.2)))
        self.assertAlmostEqual(self.m.button, L.BTN_TRAVEL)

    def test_grip_on_the_display_does_not_press_it(self):
        tcp = T.tcp_for((L.PR_X0, L.PR_Y0), L.PR_Z0, 0.0, (L.GRIP_LOCAL[0], L.BTN_Y - 0.040, 0.0))
        self.m.step(self.arms(((tcp[0], tcp[1], L.TOP + tcp[2]), 0.2)))
        self.assertEqual(self.m.button, 0.0)
        self.assertTrue(self.m.grip)  # still carried

    def test_pressing_into_the_forehead_nods_the_head(self):
        cmd = np.array([*L.arc_point(0.0, L.RF - 0.0045), L.Z_LINE])
        lens, theta = self.m.contact(cmd)
        push = theta * L.H_ARM
        self.assertAlmostEqual(push, 0.0045 / (1 / L.ARM_K + 1 / L.HEAD_K) / L.HEAD_K, places=9)
        st = {S.PROBE: probe_at(*lens, -L.CUP_TILT, BUTTON), S.HEAD: head_at(push)}
        self.assertAlmostEqual(S.gap(S.read(st)), 0.0, places=9)

    def test_off_the_forehead_does_not_press(self):
        cmd = np.array([*L.arc_point(0.0, L.RF + 0.010), L.Z_LINE])
        self.assertEqual(self.m.contact(cmd)[1], 0.0)

    def test_released_in_the_rest_stands_there(self):
        self.m.o = np.array([L.PR_X0, L.PR_Y0, L.PR_Z0 + 0.002])
        self.m.step(self.arms())
        self.assertAlmostEqual(float(self.m.o[2]), L.PR_Z0)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("radius 75 mm", "x 360, y -60 mm", "x 285, y -60 mm", "146 mm up", "205 mm up", "170 mm up",
                      "55 degrees", "x 380 mm", "14.0 N m/rad", "0.5 N per mm", "12.8 mm", "0.8 kg", "5 x 20 x 3 cm",
                      "213 g", "27 mm thick", "79 mm below", "30 degrees", "x 240, y -170 mm", "37 mm up",
                      "x 193.5 to 253.5 mm", "-210 to -230", "-270 to -340", "34 mm slot", "12 mm up",
                      "red round SCAN button"):
            self.assertIn(token, setup)
        for token in ("within 2 mm of the skin", "within 8 mm of the scan line", "within 10 degrees",
                      "1.5 mm or more", "1 to 4 mm", "-10 to +5 mm", "50 mm or more", "6 mm off the skin",
                      "-10 to 60 mm", "3 mm", "1.5 to 4.0 s", "within 10 mm (top view) and 4 mm (height)",
                      "within 15 degrees", "30 mm or more", "1 cm/s and 0.5 rad/s", "within 1 mm of upright",
                      "0.02 rad/s", "8 mm or more", "0.5 s", "20 mm or more from the head"):
            self.assertIn(token, scoring)

    def test_layout_values_the_text_states(self):
        self.assertAlmostEqual(L.RF, 0.075)
        self.assertAlmostEqual(L.FACE_TOL, 0.002)
        self.assertAlmostEqual(L.LINE_TOL, 0.008)
        self.assertAlmostEqual(L.FLUSH_DEG, 10.0)
        self.assertAlmostEqual(L.BTN_PRESSED, 0.0015)
        self.assertEqual((L.PUSH_MIN, L.PUSH_MAX, L.PUSH_SHOVE), (0.001, 0.004, 0.008))
        self.assertEqual((L.S_MIN, L.START_ZONE, L.S_BEGIN, L.S_FINISH, L.S_MAX), (-0.010, 0.005, 0.005, 0.050, 0.060))
        self.assertEqual((L.T_MIN, L.T_MAX), (1.5, 4.0))
        self.assertAlmostEqual(L.HANDS_OFF_HEAD, 0.020)
        self.assertEqual((L.REST_XY, L.REST_Z, L.REST_TILT_DEG, L.REST_YAW_DEG), (0.010, 0.004, 10.0, 15.0))
        self.assertAlmostEqual(L.HEAD_UPPER * L.H_ARM, 0.0128, places=6)

    def test_context(self):
        d = json.loads((HERE / "task.json").read_text())
        for token in ("31-1131.00", "1,505,900", "29-2061.00", "666,900", "2 to 3 seconds"):
            self.assertIn(token, d["context"])

    def test_field_limits(self):
        d = json.loads((HERE / "task.json").read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d.get(k, "")), n, k)

    def test_text_rules(self):
        dashes = (chr(0x2013), chr(0x2014))
        for p in [HERE / "task.json", HERE / "README.md", *HERE.glob("*.py"), *(HERE / "tests").glob("*.py")]:
            text = p.read_text()
            for ch in dashes:
                self.assertNotIn(ch, text, p.name)
        for p in (HERE / "task.json", HERE / "README.md"):
            text = p.read_text()
            self.assertIsNone(re.search(r"/home/|/Users/|tools/|research/|pipeline/", text), p.name)
            self.assertIsNone(re.search(r"\bTK-[A-Z0-9]{6}\b", text), p.name)

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline", text.lower())
        self.assertIn("not a measurement", text)


if __name__ == "__main__":
    unittest.main()
