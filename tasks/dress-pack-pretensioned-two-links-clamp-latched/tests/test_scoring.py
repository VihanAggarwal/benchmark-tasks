"""Offline tests for layout.py, parts.py, scoring.py, offline_dresspack.py and task.py: synthetic part states (mechkit
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
    import offline_dresspack as OD  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

Q0 = (0.0, 0.0, 0.0, 1.0)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, L.PAD_TIP]), opening)


FAR = {"left": arm((0.15, 0.30, L.TOP + 0.25)), "right": arm((0.18, -0.28, L.TOP + 0.25))}
GRIP = (L.ARM_X, L.grip_y(), L.TOP + L.HOSE_Z)
HOLDING = {"left": arm((0.15, 0.30, L.TOP + 0.25)), "right": arm((GRIP[0], GRIP[1] - L.CLOSE_AT, GRIP[2]), 0.34)}


def states(q=0.0, phi=L.LID_OPEN, vel=0.0, ring=0.0):
    ro = (L.ORIGIN[0], L.ORIGIN[1] - ring, L.ORIGIN[2])
    return {S.ROBOT: mk.state_from_poses(T.ROBOT_PART, (*L.ORIGIN, *Q0), joints={"pull": q},
                                         joint_vel={"pull": vel}),
            S.RING: mk.state_from_poses(T.RING_PART, (*ro, *Q0), joints={"lid": phi - L.LID_OPEN})}


def run(seq):
    """seq: (states kwargs, arms, steps)"""
    tr = S.Tracker()
    r = None
    for kw, arms, n in seq:
        for _ in range(n):
            r = tr.update(states(**kw), arms)
    return r


SET = {"q": L.MARK, "phi": 0.0, "ring": -L.RING_GAP}  # the hose set: the protector shut and resting on the exit


class LayoutTests(unittest.TestCase):
    def test_waists_line_up_once_a_link(self):
        for k in range(3):
            self.assertTrue(L.aligned(L.CLOSE_AT - k * L.PITCH), k)
        for q in (0.0, L.MARK, L.CLOSE_AT - L.PITCH / 2):
            self.assertFalse(L.aligned(q), q)

    def test_two_links_is_the_tre40_pitch(self):
        self.assertAlmostEqual(L.MARK, 0.0278, places=6)

    def test_set_hose_is_two_links_out(self):
        self.assertAlmostEqual(L.CLOSE_AT - L.RIB_PLAY - L.RING_GAP, L.MARK, places=9)

    def test_band_just_outside_the_set_protector(self):
        crest = L.PITCH - L.GROOVE_W
        near = L.band_rest_y() - L.MARK + crest / 2
        face = L.EXIT_Y - L.RING_W
        self.assertLess(near, face)
        self.assertGreater(near, face - 0.004)

    def test_band_visible_at_the_exit_at_the_start(self):
        crest = L.PITCH - L.GROOVE_W
        self.assertLess(L.band_rest_y() + crest / 2, L.EXIT_Y)
        self.assertGreater(L.band_rest_y() - crest / 2, L.RING_Y)

    def test_pull_force_two_links(self):
        self.assertAlmostEqual(L.PULL_K * L.MARK, 4.0, places=1)

    def test_rib_stop_is_well_above_latch(self):
        self.assertGreater(OD.RIB_STOP, 2 * L.LATCH_AT)

    def test_rib_clears_a_waist_floor_and_sits_on_a_crest(self):
        rib_bottom = L.HINGE_Z - L.LID_T / 2 - L.RIB_DEPTH
        self.assertGreater(rib_bottom, L.HOSE_Z + L.GROOVE_R)
        self.assertLess(rib_bottom, L.HOSE_Z + L.HOSE_R)
        stop = max(math.asin((L.HOSE_Z + math.sqrt(L.HOSE_R ** 2 - x ** 2) - rib_bottom) / (L.HINGE_X - x))
                   for x in np.linspace(-0.009, 0.009, 181))  # the lid's angle with the rib's underside on a crest
        self.assertAlmostEqual(stop, OD.RIB_STOP, delta=math.radians(1.0))

    def test_protector_bigger_than_the_exit(self):
        self.assertGreater(L.RING_OUT, L.UNIT_HALF - L.UNIT_WALL)
        self.assertLess(L.HOSE_R, L.UNIT_HALF - L.UNIT_WALL)

    def test_shell_clear_of_hose(self):
        self.assertGreater(L.RING_IN, L.HOSE_R)

    def test_protector_clear_of_robot(self):
        r = mk.state_from_poses(T.ROBOT_PART, (*L.ORIGIN, *Q0), joints={"pull": L.CLOSE_AT})  # a waist at the rib
        for lid in (0.0, -L.LID_OPEN):
            c = mk.state_from_poses(T.RING_PART, (*L.ORIGIN, *Q0), joints={"lid": lid})
            for body in ("base", "lid"):
                lo, hi = c.link_bounds(body)
                pts = np.array([[a, b, z] for a in np.linspace(lo[0], hi[0], 13) for b in np.linspace(lo[1], hi[1], 5)
                                for z in np.linspace(lo[2], hi[2], 13)])
                pts = pts[c.distance(pts, bodies=(body,)) <= 0]
                self.assertGreater(float(r.distance(pts).min()), 0.0, (lid, body))

    def test_push_tip_touches_lid(self):
        for deg in (95, 60, 30, 0):
            phi = math.radians(deg)
            self.assertFalse(OD._plate_hit(phi, L.push_tip(phi, into=-0.0005)), deg)
            self.assertTrue(OD._plate_hit(phi, L.push_tip(phi, into=0.002)), deg)


class ScoringTests(unittest.TestCase):
    def test_untouched_scores_zero(self):
        self.assertEqual(run([({}, FAR, 40)]).score, 0.0)

    def test_latched_without_pulling_scores_zero(self):
        self.assertEqual(run([({"phi": 0.0}, FAR, 40)]).score, 0.0)

    def test_one_link_set_scores_zero(self):
        self.assertEqual(run([(dict(SET, q=L.MARK - L.PITCH), FAR, 40)]).score, 0.0)

    def test_three_links_set_scores_zero(self):
        self.assertEqual(run([(dict(SET, q=L.MARK + L.PITCH), FAR, 40)]).score, 0.0)

    def test_pulled_but_never_shut(self):
        self.assertEqual(run([({"q": L.CLOSE_AT}, HOLDING, 10), ({}, FAR, 30)]).score, 0.25)

    def test_shut_while_near_then_lost(self):
        self.assertEqual(run([({"q": L.CLOSE_AT, "phi": 0.0}, HOLDING, 5), ({}, FAR, 30)]).score, 0.5)

    def test_shut_and_still_held_by_hand(self):
        self.assertEqual(run([({"q": L.CLOSE_AT - 0.002, "phi": 0.0}, HOLDING, 40)]).score, 0.75)

    def test_full(self):
        r = run([({"q": L.CLOSE_AT}, HOLDING, 10), ({"q": L.CLOSE_AT, "phi": 0.0}, HOLDING, 5), (SET, FAR, L.WINDOW)])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_full_needs_the_whole_window(self):
        r = run([({"q": L.CLOSE_AT}, HOLDING, 10), (SET, FAR, L.WINDOW - 1)])
        self.assertEqual(r.score, 0.75)

    def test_moving_hose_is_not_still(self):
        self.assertEqual(run([(dict(SET, vel=0.01), FAR, 40)]).score, 0.75)

    def test_lid_on_a_crest_is_not_shut(self):
        self.assertEqual(run([({"q": L.MARK, "phi": OD.RIB_STOP}, FAR, 40)]).score, 0.25)

    def test_unreadable_arms_are_not_hands_off(self):
        self.assertEqual(run([(SET, None, 40)]).score, 0.75)

    def test_hand_near_the_protector_is_not_off(self):
        near = dict(FAR, left=arm((L.ARM_X, L.RING_Y + L.RING_GAP, L.TOP + L.HINGE_Z + 0.05)))
        self.assertEqual(run([(SET, near, 40)]).score, 0.75)


class LatchTests(unittest.TestCase):
    def test_latch_rule(self):
        self.assertEqual(T.latch_target(0.0, False), (False, 0.0))
        on, target = T.latch_target(-L.LID_OPEN + math.radians(2.0), False)
        self.assertTrue(on)
        self.assertLess(target, -L.LID_OPEN)
        self.assertTrue(T.latch_target(0.0, True)[0])


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        self.m = OD.DressPackModel(T.ROBOT_PART, T.RING_PART)

    def _pull(self, q):
        held = arm((GRIP[0], GRIP[1] - q * (1 + L.WRIST_C * L.PULL_K), GRIP[2]), 0.3)
        self.m.step({"left": FAR["left"], "right": arm(GRIP, 0.3)})
        self.m.step({"left": FAR["left"], "right": held})
        return held

    def _press(self, held):
        x, z = L.push_tip(0.0, into=0.003)
        self.m.phi = math.radians(20)
        self.m.step({"left": arm((L.ARM_X + x, L.RING_Y, L.TOP + z + L.PAD_TIP), 0.0), "right": held})

    def test_hose_runs_back_when_let_go(self):
        held = self._pull(L.CLOSE_AT)
        self.assertAlmostEqual(self.m.q, L.CLOSE_AT, places=4)
        self.m.step({"left": FAR["left"], "right": arm(held.tcp, 1.0)})
        self.assertEqual(self.m.q, 0.0)

    def test_lid_over_a_crest_stops_on_the_rib(self):
        held = self._pull(L.CLOSE_AT - L.PITCH / 2)
        self._press(held)
        self.assertFalse(self.m.latched)
        self.assertGreaterEqual(self.m.phi, OD.RIB_STOP)

    def test_set_hose_rests_two_links_out(self):
        held = self._pull(L.CLOSE_AT)
        self._press(held)
        self.assertTrue(self.m.latched)
        self.m.step({"left": FAR["left"], "right": arm(held.tcp, 1.0)})
        self.assertAlmostEqual(self.m.q, L.MARK, places=6)
        self.assertAlmostEqual(self.m.ring, -L.RING_GAP, places=6)


class DemoTests(unittest.TestCase):
    def test_plans_and_variants(self):
        for v in ("",) + T.VARIANTS:
            p = T.plans(v)
            self.assertEqual(set(p), {"left", "right"})

    def test_left_waits_for_the_pull(self):
        p = T.plans()
        out = T._until(p["right"], "hold it there")
        first_push = T._until(p["left"], "push the lid over: 95")
        self.assertGreater(first_push, out)

    def test_right_lets_go_after_the_press(self):
        p = T.plans()
        self.assertGreater(T._until(p["right"], "let go"), T._until(p["left"], "keep pressing"))


class TaskJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.j = json.loads((HERE / "task.json").read_text())

    def test_fields(self):
        for k in ("title", "category", "instruction", "context", "setup", "scoring", "run_limit", "fidelity",
                  "reset_notes"):
            self.assertIn(k, self.j)
        self.assertEqual(self.j["category"], "Impact on AI timelines")

    def test_limits(self):
        lim = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
               "reset_notes": 580}
        for k, n in lim.items():
            self.assertLessEqual(len(self.j[k]), n, k)

    def test_no_dashes(self):
        for name in ("task.json", "README.md", "task.py", "layout.py", "parts.py", "scoring.py",
                     "offline_dresspack.py"):
            p = HERE / name
            if p.exists():
                self.assertNotRegex(p.read_text(), "[\u2013\u2014]", name)

    def test_instruction_matches_task(self):
        self.assertEqual(self.j["instruction"], T.INSTRUCTION)

    def test_run_limit_covers_demo(self):
        steps = int(re.match(r"(\d+)", self.j["run_limit"]).group(1))
        t = max(sum(w.duration for w in p) for p in T.plans().values())
        self.assertGreater(steps, t * L.CONTROL_HZ + 200)

    def test_numbers_in_text(self):
        text = self.j["setup"] + self.j["scoring"]
        for s in ("63 mm", "43 mm", "13.9 mm", "6.9 mm", "7 mm", "35 mm", "z 101", "y -120", "y 160", "47 mm",
                  "144 N/m", "4.0 N", "27.8 mm", "60 mm", "TR.40.30", "TR.RSE.40", "TRE.40", "30 g", "55 x 55",
                  "y 129", "4 mm", "x 277.5", "z 126", "95 deg", "0.08 N m", "3 mm", "4.5 mm", "8 deg", "2.5 deg",
                  "33.8 mm", "19.9", "47.7", "6 mm", "2 mm", "y 152.0 to 158.9", "7 mm", "5 mm", "40 mm",
                  "2 mm/s", "35.8 mm"):
            self.assertIn(s, text, s)


if __name__ == "__main__":
    unittest.main()
