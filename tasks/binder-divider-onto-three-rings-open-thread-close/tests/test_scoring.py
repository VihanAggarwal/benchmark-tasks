"""Offline tests for layout.py, parts.py, scoring.py, offline_binder.py and task.py: synthetic part states (mechkit
AssemblyStates from the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_binder as OB  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

B_SPEC, D_SPEC = T.BINDER_PART, T.DIVIDER_PART
FAR = {"left": rigstate.Arm(np.array([0.1, 0.5, 1.2]), np.array([0.1, 0.5, 1.19]), 1.0),
       "right": rigstate.Arm(np.array([0.1, -0.5, 1.2]), np.array([0.1, -0.5, 1.19]), 1.0)}


def states(q=0.0, origin=None, rot=None):
    o = np.array([L.SHEET_O[0], 0.0, L.TOP + L.SHEET_O[2]]) if origin is None else np.asarray(origin, float)
    r = np.eye(3) if rot is None else rot
    return {S.BINDER: mk.state_from_poses(B_SPEC, (L.XS, 0, L.TOP, 0, 0, 0, 1), joints={"gate": q}, source="offline"),
            S.DIVIDER: mk.state_from_poses(D_SPEC, (*o, *mk.matrix_to_quat(r)), source="offline")}


def rest_states(q, depth=L.REST_DEPTH, tilt=L.REST_TILT):
    m = OB.BinderModel(B_SPEC, D_SPEC)
    m.q = q
    m.mode = "threaded"
    m._rest_pose()
    return m.states((S.BINDER, S.DIVIDER))


def settle(st, arms=FAR, n=25):
    t = S.Tracker()
    res = None
    for _ in range(n):
        res = t.update(st, arms)
    return res


class Geometry(unittest.TestCase):
    def test_pitch_and_normal(self):
        self.assertAlmostEqual(math.sin(L.OPEN), math.cos(L.BETA), places=9)
        self.assertTrue(np.allclose(np.cross(L.U_VEC, L.D_VEC), L.N_VEC))
        self.assertTrue(np.allclose(T.A, -L.D_VEC, atol=1e-9))  # the pitched hands' approach axis lies in the sheet

    def test_letter_and_ring_pitch(self):
        self.assertAlmostEqual(L.SHEET_U, 0.2794, places=4)
        self.assertAlmostEqual(L.SHEET_V, 0.2159, places=4)
        self.assertAlmostEqual(L.RING_Y[2] - L.RING_Y[1], 0.10795, places=5)  # 4.25 in
        self.assertAlmostEqual(L.RING_Y[1] - L.RING_Y[0], 4.25 * 0.0254, places=3)

    def test_tip_gaps(self):
        f = L.half_points("fixed")[-1]
        def gap(q):
            return float(np.linalg.norm(L.rotate_gate(L.half_points("movable")[-1:], q)[0] - f))
        self.assertLess(gap(0.0), 1e-9)
        self.assertLessEqual(gap(L.CLOSED_MAX), 0.0031)
        self.assertGreaterEqual(gap(L.OPEN_MIN), 0.016)
        self.assertAlmostEqual(gap(L.OPEN), 0.0233, delta=0.001)

    def test_start_is_clear(self):
        lo, hi = D_SPEC.world_bounds()
        self.assertGreater(lo[2] - L.TOP, 0.0)  # nothing under the table top
        self.assertLess(hi[0], 0.3407 + 1e-3)
        self.assertGreater(lo[0], 0.18)
        # the divider's low edge rests on the sorter's base, its lip is 2 mm clear
        lip_x = L.SHEET_O[0] + 0.003
        self.assertAlmostEqual(lip_x - hi[0], 0.0023, delta=0.0005)

    def test_holes_line_up_with_rings(self):
        # the three holes are the rings' y positions, the hole row is 12 mm in from the punched edge
        for u, y in zip(L.HOLE_U, L.RING_Y):
            self.assertEqual(u, y)
        self.assertEqual(len(P.divider_shapes()), 6)

    def test_unit_counts(self):
        self.assertEqual(len(P.binder_shapes()), 4 + 3 * (L.SEG + 1))
        self.assertEqual(len(P.gate_shapes()), 3 * L.SEG + 2)

    def test_arm_reach_at_the_lever_and_push(self):
        self.assertAlmostEqual(T.LEVER_XY[0], L.LEVER_PRESS_X)
        self.assertLess(T.PUSH_X0, T.PUSH_X1)


class Tiers(unittest.TestCase):
    def test_reset_scores_zero(self):
        self.assertEqual(settle(states()).score, 0.0)

    def test_open_alone_is_0_3(self):
        self.assertEqual(settle(states(q=L.OPEN)).score, 0.3)

    def test_half_open_is_zero(self):
        # between the notches (0.4 rad): neither open (0.55) nor shut
        self.assertEqual(settle(states(q=0.4)).score, 0.0)

    def test_threaded_open_is_0_7(self):
        r = settle(rest_states(L.OPEN))
        self.assertEqual(r.score, 0.7)
        self.assertTrue(r.details["threaded"])

    def test_threaded_shut_still_hands_off_is_1_0(self):
        r = settle(rest_states(0.0))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_hands_on_caps_at_0_7(self):
        div_pt = np.array([L.SHEET_O[0], 0.0, L.TOP + 0.2])
        st = rest_states(0.0)
        near = {"left": rigstate.Arm(np.array([0.2, 0.1, 0.8]), np.array([0.2, 0.1, 0.77]), 1.0), "right": FAR["right"]}
        d = st[S.DIVIDER]
        pt = d.origin + np.array([0.0, 0.1, 0.0])
        near["left"] = rigstate.Arm(pt + np.array([0, 0, 0.013]), pt, 1.0)
        self.assertEqual(settle(st, near).score, 0.7)

    def test_unreadable_arms_never_hold_1_0(self):
        self.assertLess(settle(rest_states(0.0), arms=None).score, 1.0)

    def test_moving_divider_is_not_resting(self):
        t = S.Tracker()
        st = rest_states(0.0)
        res = None
        for i in range(30):
            d = st[S.DIVIDER]
            moved = dict(st)
            moved[S.DIVIDER] = mk.state_from_poses(D_SPEC, (*(d.origin + np.array([0.0, 0.0, 0.001 * (-1) ** i])),
                                                            *d.quat_xyzw), source="offline")
            res = t.update(moved, FAR)
        self.assertLess(res.score, 1.0)
        self.assertEqual(res.score, 0.7)

    def test_one_step_flap_in_the_last_second(self):
        t = S.Tracker()
        st = rest_states(0.0)
        for i in range(40):
            res = t.update(st if i != 30 else rest_states(L.OPEN), FAR)
        self.assertLess(t.last.score, 1.0 + 1e-9)
        for _ in range(25):
            res = t.update(st, FAR)
        self.assertEqual(res.score, 1.0)

    def test_fall_off_the_table_is_zero(self):
        res = settle(rest_states(0.0), n=2)
        t = S.Tracker()
        low = states(origin=(0.4, 0.0, L.TOP - 0.1))
        self.assertEqual(settle(low).score, 0.0)

    def test_tier_table_is_one_score_per_state(self):
        for is_open in (False, True):
            for thr in (False, True):
                for win in (False, True):
                    s = S.tier(is_open, thr, win)
                    self.assertIn(s, (0.0, 0.3, 0.7, 1.0))
        self.assertEqual(S.tier(True, True, False), 0.7)
        self.assertEqual(S.tier(False, True, False), 0.7)
        self.assertEqual(S.tier(False, False, False), 0.0)


class Exploits(unittest.TestCase):
    def test_divider_lying_on_shut_rings_is_zero(self):
        # flat on top of the rings, no hole round a wire: nothing
        r = settle(states(q=0.0, origin=(0.43, 0.0, L.TOP + 0.040), rot=OB.FLAT_R))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["threaded"])

    def test_divider_beside_open_rings_is_only_open(self):
        r = settle(states(q=L.OPEN, origin=(0.55, 0.0, L.TOP + 0.002), rot=OB.FLAT_R))
        self.assertEqual(r.score, 0.3)

    def test_hooked_on_the_tips_is_not_on_the_rings(self):
        # holes 3 mm along the wire from the open tips: fewer than 8 mm: not threaded
        t = L.wire_at(L.OPEN, 0.003) + np.array([0.0, 0.0, L.TOP])
        d = np.array([-math.cos(L.REST_TILT), 0.0, math.sin(L.REST_TILT)])
        n = np.cross(L.U_VEC, d)
        rot = np.column_stack([L.U_VEC, d, n]) @ L.SHEET_R0.T
        r = settle(states(q=L.OPEN, origin=t - L.HOLE_V * d, rot=rot))
        self.assertFalse(r.details["threaded"])
        self.assertEqual(r.score, 0.3)

    def test_two_of_three_rings_is_not_on_all_three(self):
        st = rest_states(L.OPEN)
        d = st[S.DIVIDER]
        shifted = mk.state_from_poses(D_SPEC, (*(d.origin + np.array([0.0, 0.050, 0.0])), *d.quat_xyzw),
                                      source="offline")  # 50 mm along y: the holes miss all three rings
        r = settle({S.BINDER: st[S.BINDER], S.DIVIDER: shifted})
        self.assertEqual(r.score, 0.3)

    def test_closing_the_rings_without_the_divider_is_zero(self):
        self.assertEqual(settle(states(q=0.0, origin=(0.30, 0.0, L.TOP + 0.0137))).score, 0.0)

    def test_threaded_at_the_foot_with_rings_shut_needs_no_depth(self):
        self.assertTrue(settle(rest_states(0.0)).details["threaded"])

    def test_holes_on_a_ring_but_sheet_off_table_latches(self):
        st = rest_states(0.0)
        d = st[S.DIVIDER]
        low = mk.state_from_poses(D_SPEC, (*(d.origin - np.array([0.0, 0.0, 0.2])), *d.quat_xyzw), source="offline")
        self.assertEqual(settle({S.BINDER: st[S.BINDER], S.DIVIDER: low}).score, 0.0)


class Offline(unittest.TestCase):
    def run_demo(self, variant="", idle=""):
        os.environ["DEMO_VARIANT"] = variant
        os.environ["DEMO_IDLE_ARM"] = idle
        try:
            plan = T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT", None)
            os.environ.pop("DEMO_IDLE_ARM", None)
        return plan

    def test_demo_has_both_arms_and_fits_the_run_limit(self):
        plan = self.run_demo()
        self.assertEqual(set(plan), {"left", "right"})
        steps = max(sum(w.duration for w in p) for p in plan.values()) * 20
        self.assertLess(steps, 1200)

    def test_idle_arm_drops_its_plan(self):
        self.assertEqual(set(self.run_demo(idle="left")), {"right"})
        self.assertEqual(set(self.run_demo(idle="right")), {"left"})

    def test_bad_variant_raises(self):
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_a_single_pinch_swings_the_divider_off_the_rings(self):
        # one hand holding at u = +120 mm: the model hangs the divider under its pad: the holes miss
        m = OB.BinderModel(B_SPEC, D_SPEC)
        m.q = L.OPEN
        tip = m.world_of((0.120, L.GRIP_V, 0.0))
        tcp = tip - 0.013 * T.A
        m.holds["left"] = np.array([0.120, L.GRIP_V, 0.0])
        m.mode = "held"
        g = {"left": (tcp, tip, 0.0), "right": (np.array([0.1, -0.5, 1.2]), np.array([0.1, -0.5, 1.19]), 1.0)}
        m._follow(g)
        self.assertGreater(abs(m.swing), 0.3)

    def test_lever_press_opens_and_detent_snaps(self):
        m = OB.BinderModel(B_SPEC, D_SPEC)
        x, y = T.LEVER_XY
        for z in (0.030, 0.020, 0.012, 0.0095):
            tip = np.array([x, y, L.TOP + z])
            g = {"left": (np.array([0.1, 0.5, 1.2]), np.array([0.1, 0.5, 1.19]), 1.0), "right": (tip + [0, 0, 0.013], tip, 0.0)}
            m._lever(g)
            m._detent(True)
        self.assertEqual(m.q, L.OPEN)
        # released the lever: stays open (the detent holds the notch)
        m._detent(False)
        self.assertEqual(m.q, L.OPEN)

    def test_push_shuts_the_gate(self):
        m = OB.BinderModel(B_SPEC, D_SPEC)
        m.q = L.OPEN
        for x in np.linspace(T.PUSH_X0, T.PUSH_X1, 40):
            tip = np.array([x, T.PUSH_Y, L.TOP + T.PUSH_Z])
            g = {"left": (tip + [0, 0, 0.013], tip, 0.0), "right": (np.array([0.1, -0.5, 1.2]), np.array([0.1, -0.5, 1.19]), 1.0)}
            m._push(g)
            m._detent(False)
        self.assertEqual(m.q, 0.0)


class Text(unittest.TestCase):
    def setUp(self):
        self.j = json.loads((HERE / "task.json").read_text())

    def test_no_em_dashes_or_identity(self):
        for f in HERE.glob("*.py"):
            self.assertNotRegex(f.read_text(), "[\u2013\u2014]", f.name)
        txt = json.dumps(self.j, ensure_ascii=False) + (HERE / "README.md").read_text()
        self.assertNotRegex(txt, "[\u2013\u2014]")
        self.assertNotRegex(txt, r"(?i)T-?70|vihan|aggarwal|claude|anthropic")

    def test_numbers_agree_between_text_and_code(self):
        s = self.j["setup"] + self.j["scoring"]
        for needle in ("0.55 rad", "0.10 rad", "8 mm", "0.80 rad", "16 mm", "30 mm", "5 mm/s", "0.1 rad/s",
                       "-0.108", "0.400 m"):
            self.assertIn(needle, s, needle)
        self.assertEqual(L.OPEN_MIN, 0.55)
        self.assertEqual(L.CLOSED_MAX, 0.10)
        self.assertEqual(S.DEPTH_MIN, 0.008)
        self.assertEqual(L.HANDS_OFF, 0.030)
        self.assertEqual(L.STILL_SPEED, 0.005)
        self.assertEqual(L.OPEN, 0.80)
        self.assertEqual(L.XS, 0.400)

    def test_instruction_agrees(self):
        self.assertEqual(T.INSTRUCTION, self.j["instruction"])

    def test_scores_listed(self):
        self.assertIn("0, 0.3, 0.7, 1.0", self.j["scoring"])


if __name__ == "__main__":
    unittest.main()
