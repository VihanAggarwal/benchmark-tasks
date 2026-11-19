"""Offline tests for layout.py, parts.py, scoring.py, offline_headset.py and task.py: synthetic headset states (mechkit
AssemblyStates from the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import filecmp
import json
import os
import pathlib
import re
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
for _p in (ROOT / "tools" / "offline", ROOT / "tools" / "offline" / "yam_stub"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import offline_headset  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

Q0 = (0.0, 0.0, 0.0, 1.0)
FAR = {"left": rigstate.Arm(np.array([0.2, 0.25, L.TOP + 0.17]), np.array([0.2, 0.25, L.TOP + 0.157]), 1.0),
       "right": rigstate.Arm(np.array([0.2, -0.25, L.TOP + 0.17]), np.array([0.2, -0.25, L.TOP + 0.157]), 1.0)}


def hs(origin, cups=0.0, boom=0.0, quat=Q0):
    return {S.HEADSET: mk.state_from_poses(T.HEADSET_PART, (*origin, *quat),
                                           joints={"cup_p": cups, "cup_n": cups, "boom": boom}, source="offline")}


def seated(boom=L.BOOM_DOWN, cups=L.Q_SEAT, dx=0.0, dz=0.0):
    o = np.array(L.HEADSET_ORIGIN_SEATED) + np.array([dx, 0.0, dz])
    return hs(o, cups, boom)


def _box_surface(shape, step=0.001):
    """Points on a box shape's faces about 1 mm apart, base frame at rest."""
    c, h = np.array(shape.center, float), np.array(shape.size, float) / 2
    axes = [np.arange(-h[i], h[i] + 1e-9, step) for i in range(3)]
    out = []
    for i in range(3):
        for sgn in (-1.0, 1.0):
            g = list(axes)
            g[i] = np.array([sgn * h[i]])
            out.append(np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3))
    return np.vstack(out) + c


def run(states, arms=FAR, n=L.WINDOW + 2):
    tr = S.Tracker()
    r = None
    for _ in range(n):
        r = tr.update(states, arms)
    return r


def arms_touching():
    c = np.array(L.HEADSET_ORIGIN_SEATED) + np.array([0.0, offline_headset.SHELL_YC + L.Q_SEAT, L.CUP_ZC])
    near = rigstate.Arm(c + np.array([0, 0, 0.013]), c, 0.6)
    return {"left": near, "right": FAR["right"]}


class Geometry(unittest.TestCase):
    def test_rest_gap_is_narrower_than_the_ears(self):
        self.assertLess(2 * L.REST_INNER, 2 * L.EAR_Y1)
        self.assertAlmostEqual(L.Q_SEAT, 0.012)
        self.assertGreater(L.CLAMP_N, 1.5)
        self.assertLess(L.CLAMP_N, 3.0)

    def test_mass_under_the_gripper_limit(self):
        self.assertAlmostEqual(L.HEADSET_MASS, 0.102, places=3)
        self.assertLess(L.HEADSET_MASS, 0.4)

    def test_boom_down_reaches_the_mouth_and_stays_in_front_of_the_face(self):
        r = S.read(seated())
        self.assertLessEqual(r.mic_gap, L.MIC_NEAR)
        self.assertLess(r.mic[0], L.HX - 0.058 - 0.010)  # in front of the lower face

    def test_boom_swing_clears_the_stand(self):
        # First GPU run (2026-11-19): with a 96 mm boom the stowed microphone sat on the fixed cradle, the push could
        # not swing the boom and yawed the headset 11 degrees instead, so both grips missed their cups.
        stand = mk.state_from_poses(T.STAND_PART, (*T.STAND_PART.pos, *Q0), source="offline")
        pts = np.vstack([_box_surface(s) for s in T.HEADSET_PART.body_shapes("boom")])
        for th in np.linspace(0.0, L.BOOM_DOWN, 80):
            h = hs(L.HEADSET_ORIGIN_START, boom=th)[S.HEADSET]
            gap = float(stand.distance(h.link_to_world("boom", pts)).min())
            self.assertGreater(gap, 0.005, f"boom at {th:.2f} rad is {gap * 1000:.1f} mm from the stand")

    def test_boom_down_clears_the_head(self):
        head = mk.state_from_poses(T.HEAD_PART, (*T.HEAD_PART.pos, *Q0), source="offline")
        pts = np.vstack([_box_surface(s) for s in T.HEADSET_PART.body_shapes("boom")])
        h = seated()[S.HEADSET]
        self.assertGreater(float(head.distance(h.link_to_world("boom", pts)).min()), 0.010)

    def test_push_fingers_clear_the_band_side(self):
        half_width = 0.0135  # a finger's half width at TCP level (27 mm)
        band_side_outer = 0.095  # the band's stepped side runs out to |y| 95 mm
        self.assertGreater(-T.PUSH["y"] - half_width, band_side_outer + 0.002)
        self.assertLess(-T.PUSH["y"] - half_width, -L.BOOM_Y - 0.003)  # still covers the 6 mm arm

    def test_open_fingers_clear_the_shell_on_approach(self):
        self.assertGreaterEqual(L.OPEN_GAP / 2 - L.CUP_R, 0.012)
        self.assertLessEqual(L.OPEN_GAP, 0.090)

    def test_stowed_boom_clears_the_crown(self):
        r = S.read(seated(boom=0.0))
        self.assertGreater(r.mic[2] - L.TOP, L.CROWN_Z)

    def test_grip_fits_the_shell(self):
        self.assertAlmostEqual(L.pinch_force(L.GRIP_GAP, 2 * L.CUP_R), 5.0)

    def test_seated_cushions_are_on_the_ears(self):
        r = S.read(seated())
        for g in r.ear_gap.values():
            self.assertLess(g, L.EAR_NEAR)


class Tiers(unittest.TestCase):
    def test_start_is_zero(self):
        self.assertEqual(run(hs(L.HEADSET_ORIGIN_START)).score, 0.0)

    def test_boom_lowered_on_the_stand_is_zero(self):
        self.assertEqual(run(hs(L.HEADSET_ORIGIN_START, boom=L.BOOM_DOWN)).score, 0.0)

    def test_lifted_off_the_stand_is_quarter(self):
        o = np.array(L.HEADSET_ORIGIN_START) + np.array([0.08, 0, 0.03])
        self.assertEqual(run(hs(o, cups=L.Q_SPREAD, boom=L.BOOM_DOWN)).score, 0.25)

    def test_seated_but_cups_held_apart_is_quarter(self):
        self.assertEqual(run(seated(cups=L.Q_SPREAD)).score, 0.25)

    def test_seated_mic_up_is_half(self):
        self.assertEqual(run(seated(boom=0.0)).score, 0.5)

    def test_mic_half_way_is_half(self):
        self.assertEqual(run(seated(boom=-1.0)).score, 0.5)

    def test_done_is_one(self):
        r = run(seated())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_still_held_is_three_quarters(self):
        self.assertEqual(run(seated(), arms=arms_touching()).score, 0.75)

    def test_unreadable_arms_are_never_hands_off(self):
        self.assertEqual(run(seated(), arms=None).score, 0.75)

    def test_cup_held_4_mm_off_its_ear(self):
        self.assertEqual(run(seated(cups=L.Q_SEAT + 0.004)).score, 0.25)

    def test_slid_along_the_cradle_is_still_on_the_stand(self):
        o = np.array(L.HEADSET_ORIGIN_START) + np.array([0.0, 0.040, 0.0])
        self.assertEqual(run(hs(o, boom=L.BOOM_DOWN)).score, 0.0)

    def test_band_not_down_on_the_crown(self):
        self.assertEqual(run(seated(dz=0.020)).score, 0.25)

    def test_tilted_band(self):
        q = (0.0, np.sin(0.2), 0.0, np.cos(0.2))  # 0.4 rad about y
        o = np.array(L.HEADSET_ORIGIN_SEATED)
        self.assertEqual(run(hs(o, L.Q_SEAT, L.BOOM_DOWN, quat=q)).score, 0.25)

    def test_on_backwards(self):
        q = (0.0, 0.0, 1.0, 0.0)  # turned half round: the boom's cup on the right ear
        o = np.array(L.HEADSET_ORIGIN_SEATED)
        self.assertEqual(run(hs(o, L.Q_SEAT, L.BOOM_DOWN, quat=q)).score, 0.25)

    def test_moving_is_three_quarters(self):
        tr = S.Tracker()
        r = None
        for i in range(L.WINDOW + 2):
            r = tr.update(seated(dx=0.001 * (i % 2)), FAR)  # 20 mm/s jitter
        self.assertEqual(r.score, 0.75)

    def test_part_below_the_table_top_latches(self):
        tr = S.Tracker()
        low = hs(np.array([0.25, 0.0, L.TOP + 0.03]), boom=L.BOOM_DOWN)  # cups 90 mm under the top: it went off
        self.assertLess(low[S.HEADSET].low[2], L.TOP - L.BELOW_TOP)
        r = tr.update(low, FAR)
        self.assertTrue(r.details["latch"])
        for _ in range(L.WINDOW + 2):
            r = tr.update(seated(), FAR)
        self.assertEqual(r.score, 0.0)

    def test_fell_off_the_front_edge_latches(self):
        tr = S.Tracker()
        tr.update(hs(np.array([L.TABLE_X[1] + 0.05, 0.0, L.TOP - 0.20])), FAR)
        for _ in range(L.WINDOW + 2):
            r = tr.update(seated(), FAR)
        self.assertEqual(r.score, 0.0)

    @staticmethod
    def _on_table(x, y, quat=Q0, sink=0.002):
        """The headset resting on the table top at (x, y), its lowest point ``sink`` into the top (contact)."""
        probe = hs(np.array([x, y, L.TOP + 0.3]), quat=quat)[S.HEADSET]
        return np.array([x, y, L.TOP + 0.3 - (probe.low[2] - L.TOP) - sink])

    def test_set_down_on_the_table_then_put_on_is_one(self):
        # Post-GPU gate, 19 Nov: the old latch (the band's origin under 100 mm) zeroed a headset set down on the
        # table on purpose. Upside down on its band, cups up: the band's origin is about 15 mm above the top.
        q = (1.0, 0.0, 0.0, 0.0)
        o = self._on_table(0.25, 0.20, quat=q)
        self.assertLess(o[2] - L.TOP, 0.100)  # what the old rule latched on
        down = hs(o, boom=L.BOOM_DOWN, quat=q)
        tr = S.Tracker()
        for _ in range(30):
            r = tr.update(down, FAR)
            self.assertFalse(r.details["latch"])
        self.assertEqual(r.score, 0.25)  # off the stand, not on the head
        lifted = hs(np.array(L.HEADSET_ORIGIN_SEATED) + np.array([0.0, 0.0, 0.05]), cups=L.Q_SPREAD,
                    boom=L.BOOM_DOWN)
        for _ in range(10):
            tr.update(lifted, FAR)
        for _ in range(L.WINDOW + 2):
            r = tr.update(seated(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_part_past_the_side_edge_latches(self):
        o = self._on_table(0.25, 0.5)
        h = hs(o)[S.HEADSET]
        o[1] += L.TABLE_Y[1] - float(h.high[1]) + 0.003  # upright on its cups, the +y cup 3 mm past the edge
        tr = S.Tracker()
        r = tr.update(hs(o), FAR)
        self.assertTrue(r.details["latch"])
        for _ in range(L.WINDOW + 2):
            r = tr.update(seated(), FAR)
        self.assertEqual(r.score, 0.0)
        o[1] -= 0.008  # 5 mm inside the edge: still on the table, no latch
        self.assertFalse(S.Tracker().update(hs(o), FAR).details["latch"])

    def test_reset_clears_the_latch(self):
        tr = S.Tracker()
        tr.update(hs(np.array([0.25, 0.0, L.TOP + 0.03])), FAR)
        tr.reset()
        for _ in range(L.WINDOW + 2):
            r = tr.update(seated(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_missing_and_nan(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, FAR).score, 0.0)
        for _ in range(L.WINDOW + 2):
            tr.update(seated(), FAR)
        bad = hs(np.array([np.nan, 0.0, L.TOP + 0.2]))
        self.assertLessEqual(tr.update(bad, FAR).score, 0.75)
        self.assertLess(tr.update(seated(), FAR).score, 1.0)  # the window restarted


class OfflineModel(unittest.TestCase):
    def _model(self):
        return offline_headset.HeadsetModel(T.HEADSET_PART, T.HEAD_PART, T.STAND_PART)

    def test_one_hand_carry_into_the_head_falls(self):
        m = self._model()
        c = m.cup_centre("cup_p") + np.array([0, 0, L.GRIP_DZ])
        path = [c + np.array([0, 0, dz]) for dz in np.linspace(0, 0.03, 10)]
        path += [c + np.array([x - L.SX, 0.0, 0.03]) for x in np.linspace(L.SX, L.HX, 60)]
        for tips in path:
            m.step({"left": rigstate.Arm(tips + [0, 0, 0.013], tips, L.GRIP_GAP / rigstate.FULL_GAP),
                    "right": FAR["right"]})
        self.assertEqual(m.mode, "fallen")

    def test_open_fingers_do_not_hold(self):
        m = self._model()
        c = m.cup_centre("cup_p")
        m.step({"left": rigstate.Arm(c + [0, 0, 0.013], c, 1.0), "right": FAR["right"]})
        self.assertEqual(m.holds, {})


class Demo(unittest.TestCase):
    def test_demo_plans_and_variants(self):
        p = T.plans()
        self.assertEqual(set(p), {"left", "right"})
        for v in T.VARIANTS:
            self.assertEqual(set(T.plans(v)), {"left", "right"})
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_hold_only_and_idle_arm(self):
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

    def test_both_arms_on_one_clock(self):
        p = T.plans()
        d = {a: sum(w.duration for w in p[a]) for a in p}
        self.assertAlmostEqual(d["left"], d["right"], places=6)
        t_desc = {}
        for a in p:
            t = 0.0
            for w in p[a]:
                if w.label.startswith("down either side"):
                    t_desc[a] = t
                    break
                t += w.duration
        self.assertAlmostEqual(t_desc["left"], t_desc["right"])

    def test_demo_fits_the_run_limit(self):
        steps = sum(w.duration for w in T.plans()["left"]) * L.CONTROL_HZ
        limit = int(re.match(r"\d+", json.load(open(HERE / "task.json"))["run_limit"]).group())
        self.assertLess(steps + 60, limit)


class Files(unittest.TestCase):
    def test_kit_files_are_the_kits(self):
        for f in ("mechkit.py", "rigstate.py"):
            self.assertTrue(filecmp.cmp(HERE / f, ROOT / "tools" / "kit" / f, shallow=False), f)

    def test_text_has_no_dashes_or_names(self):
        for f in ("task.json", "README.md"):
            t = (HERE / f).read_text()
            self.assertNotIn(chr(0x2014), t)
            self.assertNotIn(chr(0x2013), t)
            for bad in ("T-70", "Vihan", "Aggarwal", "Claude", "Anthropic"):
                self.assertNotIn(bad, t)

    def test_task_json_matches_the_code(self):
        d = json.load(open(HERE / "task.json"))
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["category"], "Societal relatability")
        for s in ("(330, 0)", "(150, 0)", "102 g", "212 mm", "160 mm apart", "136 mm apart", "50 mm", "30 mm",
                  "past the table's edge", "below the table top"):
            self.assertIn(s, d["setup"] + d["scoring"], s)


if __name__ == "__main__":
    unittest.main()
