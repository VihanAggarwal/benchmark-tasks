"""Offline tests for layout.py, shoerule.py, scoring.py, offline_saw.py and task.py: synthetic saw states (mechkit
AssemblyStates from the task's own spec), synthetic arm snapshots (rigstate.Arm) and the demo replayed through the
offline kinematics, no simulator.
    python3 -m unittest discover -s tests -v"""

import hashlib
import json
import math
import os
import pathlib
import re
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE_TOOLS = _offline_dir()
if OFFLINE_TOOLS is not None:
    for _p in (OFFLINE_TOOLS, OFFLINE_TOOLS / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_saw as OS  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import shoerule as R  # noqa: E402
import task as T  # noqa: E402
from yam_arena import demo as D  # noqa: E402

HZ = L.CONTROL_HZ
WIN = L.WIN_STEPS
SPEC = T.SAW_SPEC
AWAY = {a: rigstate.Arm(np.array([0.28, s * 0.25, L.TOP + 0.12]), np.array([0.28, s * 0.25, L.TOP + 0.107]), 0.4)
        for a, s in (("left", 1.0), ("right", -1.0))}
FAR = {a: rigstate.Arm(np.array([0.28, s * 0.25, L.TOP + 0.46]), np.array([0.28, s * 0.25, L.TOP + 0.447]), 0.4)
       for a, s in (("left", 1.0), ("right", -1.0))}  # well clear of a saw stood up or turned over
BASES = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def saw(slide_mm=0.0, bevel_deg=0.0, lever_deg=0.0, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, tilt_deg=0.0, yaw_deg=0.0):
    o = np.array(L.ORIGIN_M, dtype=float) + np.array([dx_mm, dy_mm, dz_mm]) * L.MM
    q = mk.matrix_to_quat(rot_z(math.radians(yaw_deg)) @ L.rot_x(math.radians(tilt_deg)))
    return {S.SAW: mk.state_from_poses(SPEC, (*o, *q), np.zeros(6),
                                       joints={"slide": slide_mm * L.MM, "bevel": math.radians(bevel_deg),
                                               "lever": math.radians(lever_deg)}, source="sim")}


def resting_dz(bevel_deg=45.0, tilt_deg=0.0, yaw_deg=0.0):
    """How far (mm) to raise or lower the saw, turned so, until its lowest point rests on the mat."""
    st = saw(10, bevel_deg, 0, tilt_deg=tilt_deg, yaw_deg=yaw_deg)[S.SAW]
    return (L.TOP + L.MAT_T * L.MM - float(st.low[2])) / L.MM


UPRIGHT_DZ_MM = {b: resting_dz(b, 90.0 - b) for b in (45.0, -45.0)}


def upright(bevel_deg=45.0):
    """The saw set upright on its shoe (the shoe's working face down, on the mat), shoe at ``bevel_deg``, clamped."""
    return saw(10, bevel_deg, 0, tilt_deg=90.0 - bevel_deg, dz_mm=UPRIGHT_DZ_MM[bevel_deg])

def arms_at(point_b_mm, side="right", opening=0.0):
    """Arms with ``side``'s pad tips at a B-frame point (mm), the other away."""
    out = dict(AWAY)
    tips = np.array(L.world_mm(point_b_mm))
    out[side] = rigstate.Arm(tips + np.array([0.0, 0.0, 0.013]), tips, opening)
    return out


def run(tr, states, arms, n):
    res = None
    for _ in range(n):
        res = tr.update(states, arms)
    return res


def replay(variant="", idle=None, hold_s=3.0):
    """The demo's commanded motion through the offline kinematics and the scorer, as the offline checker plays it:
    returns (final score, step at which 1.0 had held SUCCESS_HOLD_S or None, the model, the tracker's last details)."""
    plans = T.plans(variant)
    if idle:
        plans.pop(idle, None)
    arm_plans = {s: D.ArmPlan(w, BASES[s]) for s, w in plans.items()}
    total = max(p.duration for p in arm_plans.values()) + hold_s
    model, tr = OS.OfflineSaw(), S.Tracker()
    start = {s: D.ArmPlan([], BASES[s]).start_pos for s in ("left", "right")}
    held, success_at = 0, None
    n = int(round(total * HZ))
    for k in range(1, n + 1):
        t = k / HZ
        arms = {}
        for s in ("left", "right"):
            if s in arm_plans:
                pos, _rot, finger, _label = arm_plans[s].target(t)
            else:
                pos, finger = start[s], rigstate.FINGER_OPEN
            pos = np.asarray(pos, dtype=float)
            arms[s] = rigstate.Arm(pos, pos - np.array([0.0, 0.0, 0.013]), float(np.clip(finger / rigstate.FINGER_OPEN,
                                                                                            0.0, 1.0)))
        model.step(arms)
        o = np.array(L.ORIGIN_M)
        st = {S.SAW: mk.state_from_poses(SPEC, (*o, 0.0, 0.0, 0.0, 1.0), np.zeros(6), joints=model.joints(),
                                         source="offline")}
        res = tr.update(st, arms)
        held = held + 1 if res.success else 0
        if success_at is None and held >= int(round(T.SUCCESS_HOLD_S * HZ)):
            success_at = k
    return res.score, success_at, model, res.details


# -- geometry and the sourced figures ---------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_sourced_figures(self):
        self.assertAlmostEqual(L.MASS_TOTAL, 2.449, places=3)  # 5.4 lb
        self.assertAlmostEqual(L.BODY_MASS + L.SHOE_MASS + L.CARRIAGE_MASS + L.LEVER_MASS, L.MASS_TOTAL, places=9)
        self.assertAlmostEqual(L.X_SHOE_FRONT - L.X_REAR, L.LENGTH, delta=0.5)
        self.assertAlmostEqual(L.BODY_T, 99.06, places=2)
        self.assertEqual(L.BODY_BOXES["handle"][1][1], L.HEIGHT)  # 215 mm, sole to handle top (maker's datasheet)
        self.assertEqual(L.HEIGHT, 215.0)
        self.assertEqual(L.DETENTS_DEG, (0.0, 15.0, 30.0, 45.0))
        self.assertEqual(L.BEVEL_MAX_DEG, 45.0)

    def test_spec_masses_and_joints(self):
        self.assertAlmostEqual(SPEC.mass, L.BODY_MASS)
        js = {j.name: j for j in SPEC.joints}
        self.assertEqual(list(js), ["slide", "bevel", "lever"])
        self.assertEqual(js["bevel"].parent, "slide")
        self.assertAlmostEqual(js["slide"].upper, 0.010)
        self.assertAlmostEqual(math.degrees(js["bevel"].upper), 45.0)
        self.assertAlmostEqual(math.degrees(js["bevel"].lower), -45.0)
        self.assertAlmostEqual(math.degrees(js["lever"].upper), 70.0)
        self.assertEqual(SPEC.solver_iterations, 32)

    def test_housing_lies_on_the_mat_and_is_the_lowest_part(self):
        lo, hi = SPEC.local_bounds()
        self.assertAlmostEqual(lo[2], 0.0)
        self.assertAlmostEqual(hi[2], L.LEVER_BOXES["tab"][2][1] * L.MM, places=6)
        for n, b in L.BODY_BOXES.items():
            if n != "housing":
                self.assertGreater(b[2][0], 0.0, n)

    def test_shoe_stays_off_the_mat_at_every_bevel_and_slide(self):
        for a in np.arange(-45.0, 45.01, 2.5):
            for s in (0.0, 6.0, 10.0):
                self.assertGreater(L.shoe_points(math.radians(a), s)[:, 2].min(), 3.0, (a, s))

    def test_shoe_tilting_the_demo_way_clears_the_body(self):
        body = [(np.array([b[0][0], b[1][0], b[2][0]]), np.array([b[0][1], b[1][1], b[2][1]]))
                for b in L.BODY_BOXES.values()]
        for a in np.arange(0.0, 45.01, 2.5):
            for s in (0.0, 10.0):
                p = L.shoe_points(math.radians(a), s)
                for lo, hi in body:
                    self.assertFalse(OS.OfflineSaw.overlaps(p, (lo, hi)), (a, s))

    def test_positive_bevel_moves_the_top_edge_out_of_the_body_side(self):
        top = np.array([[0.0, 0.0, L.Z_C + L.SHOE_HALF_W]])
        p = (top - np.array(L.PIVOT)) @ L.rot_x(math.radians(45)).T + np.array(L.PIVOT)
        self.assertLess(p[0, 1], -30.0)

    def test_lever_lifts_its_tab_when_turned_positive(self):
        a = L.lever_points(0.0)[:, 2].max()
        b = L.lever_points(math.radians(60))[:, 2].max()
        self.assertGreater(b, a + 20.0)

    def test_lever_and_shoe_do_not_overlap(self):
        for a in (0.0, 30.0, 60.0):
            for bev in (0.0, 45.0):
                lv = L.lever_points(math.radians(a))
                for b in L.SHOE_BOXES.values():
                    lo = np.array([b[0][0], b[1][0], b[2][0]])
                    hi = np.array([b[0][1], b[1][1], b[2][1]])
                    if bev == 0.0:
                        self.assertFalse(OS.OfflineSaw.overlaps(lv, (lo, hi)), (a, bev))

    def test_world_placement(self):
        lo, hi = SPEC.world_bounds()
        self.assertAlmostEqual(L.ORIGIN_M[1], -0.060)
        self.assertGreater(lo[0], 0.03)
        self.assertLess(hi[0], 0.40)
        mlo, mhi = T.MAT_SPEC.world_bounds()
        self.assertTrue(np.all(mlo[:2] < lo[:2]) and np.all(mhi[:2] > hi[:2]))
        self.assertAlmostEqual(mhi[2], L.ORIGIN_M[2], places=9)


class Kit(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        kit = HERE.parents[1] / "tools" / "kit"
        if not kit.is_dir():
            self.skipTest("tools/kit not found")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             hashlib.sha256((kit / name).read_bytes()).hexdigest(), name)

    def test_objects(self):
        self.assertEqual([o.name for o in T.OBJECTS], ["mat", "saw"])
        self.assertIsNone(T.MAT_SPEC.mass)
        self.assertEqual(T.SCORE_OVER, "end")


# -- the one software rule ------------------------------------------------------------------------------------------
class Rule(unittest.TestCase):
    def setUp(self):
        self.m = R.Memory()

    def test_clamped_holds_everything_where_it_clamped(self):
        t = R.targets(0.0, 0.0, 0.0, self.m)
        self.assertEqual((t["slide"], t["bevel"], t["mode"]), (0.0, 0.0, "clamped"))
        R.targets(0.010, math.radians(45), math.radians(60), self.m)
        t = R.targets(0.010, math.radians(45), math.radians(5), self.m)
        self.assertEqual(t["mode"], "clamped")
        t = R.targets(0.008, math.radians(40), math.radians(0), self.m)  # pushed: the targets do not follow
        self.assertAlmostEqual(t["slide"], 0.010)
        self.assertAlmostEqual(math.degrees(t["bevel"]), 45.0)

    def test_on_the_stop_the_bevel_is_held_at_zero(self):
        for s_mm in (0.0, 3.0, 5.9):
            m = R.Memory()
            t = R.targets(s_mm * L.MM, math.radians(1.5), math.radians(60), m)
            self.assertEqual(t["mode"], "free_on_stop")
            self.assertEqual(t["bevel"], 0.0)

    def test_off_the_stop_the_bevel_follows(self):
        R.targets(0.006, math.radians(20.0), math.radians(60), self.m)
        t = R.targets(0.006, math.radians(21.0), math.radians(60), self.m)
        self.assertEqual(t["mode"], "free")
        self.assertAlmostEqual(math.degrees(t["bevel"]), 21.9, places=6)  # led by 0.9 of the last step

    def test_lead_is_clipped(self):
        R.targets(0.006, math.radians(10.0), math.radians(60), self.m)
        t = R.targets(0.006, math.radians(20.0), math.radians(60), self.m)
        self.assertAlmostEqual(math.degrees(t["bevel"]), 21.0, places=6)

    def test_detent_pull_is_small_and_only_near_a_detent(self):
        self.assertEqual(R.detent_pull(math.radians(22.0)), 0.0)
        p = R.detent_pull(math.radians(43.5))
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p * L.BEVEL_K, 0.03 + 1e-12)
        self.assertLess(R.detent_pull(math.radians(-43.5)), 0.0)
        self.assertLess(R.detent_pull(math.radians(16.0)), 0.0)

    def test_detent_force_at_the_push_height_is_under_8_n(self):
        arm = (T.TILT["z"] - L.Z_C) * L.MM
        self.assertLess(L.DETENT_PULL * L.BEVEL_K / arm, 8.0)
        self.assertLess(L.LEVER_PULL * L.LEVER_K / 0.010, 8.0)

    def test_clamped_shoe_under_a_5_n_push_turns_under_a_degree(self):
        torque = 5.0 * L.SHOE_HALF_W * L.MM
        self.assertLess(math.degrees(torque / L.BEVEL_K), 1.0)

    def test_lever_rests(self):
        self.assertLess(R.lever_target(math.radians(20.0)), math.radians(20.0))
        self.assertGreater(R.lever_target(math.radians(30.0)), math.radians(30.0))
        self.assertAlmostEqual(R.lever_target(math.radians(60.0)), math.radians(60.0))
        self.assertLessEqual(abs(R.lever_target(math.radians(30.0)) - math.radians(30.0)), L.LEVER_PULL + 1e-12)

    def test_tilted_shoe_cannot_slide_back_onto_the_stop(self):
        R.targets(0.008, math.radians(30), math.radians(60), self.m)
        t = R.targets(0.002, math.radians(30), math.radians(60), self.m)
        self.assertGreaterEqual(t["slide"], L.SLIDE_REL * L.MM - 1e-12)

    def test_slide_stays_in_its_slot(self):
        R.targets(0.0095, 0.0, math.radians(60), self.m)
        t = R.targets(0.0105, 0.0, math.radians(60), self.m)
        self.assertLessEqual(t["slide"], 0.010 + 1e-12)

    def test_unclamping_does_not_lead_on_the_clamps_last_pull(self):
        R.targets(0.0, 0.0, 0.0, self.m)
        t = R.targets(0.003, 0.0, math.radians(30), self.m)
        self.assertAlmostEqual(t["slide"], 0.003)

    def test_task_hook_writes_all_three_targets(self):
        calls = []
        orig = mk.set_targets
        mk.set_targets = lambda a, j, v: calls.append((j, v))
        try:
            T.MEM.reset()
            T._shoe_rule(saw(0.0, 0.0, 60.0))
            T._shoe_rule({S.SAW: None})
        finally:
            mk.set_targets = orig
            T.MEM.reset()
        self.assertEqual([c[0] for c in calls], ["slide", "bevel", "lever"] * 2)


# -- the scorer -----------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def setUp(self):
        self.tr = S.Tracker()

    def test_nothing_scores_zero(self):
        self.assertEqual(run(self.tr, saw(), AWAY, WIN + 5).score, 0.0)

    def test_full(self):
        r = run(self.tr, saw(10, 45, 0), AWAY, WIN)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_full_needs_the_whole_window(self):
        self.assertLess(run(self.tr, saw(10, 45, 0), AWAY, WIN - 1).score, 1.0)

    def test_other_direction_counts(self):
        self.assertEqual(run(self.tr, saw(10, -45, 0), AWAY, WIN).score, 1.0)

    def test_band_edges(self):
        for a, want in ((43.0, 1.0), (42.9, 0.5), (47.0, 1.0), (30.0, 0.5), (28.0, 0.5), (27.9, 0.0), (15.0, 0.0)):
            tr = S.Tracker()
            self.assertEqual(run(tr, saw(10, a, 0), AWAY, WIN).score, want, a)

    def test_unclamped_45(self):
        self.assertEqual(run(self.tr, saw(10, 45, 60), AWAY, WIN).score, 0.75)

    def test_lock_edge(self):
        self.assertEqual(run(S.Tracker(), saw(10, 45, 9.9), AWAY, WIN).score, 1.0)
        self.assertEqual(run(S.Tracker(), saw(10, 45, 10.0), AWAY, WIN).score, 0.75)

    def test_milestone_latches(self):
        run(self.tr, saw(6.0, 0, 60), AWAY, 1)
        self.assertEqual(run(self.tr, saw(10, 15, 0), AWAY, WIN).score, 0.25)

    def test_milestone_needs_the_lever_out_and_the_slide(self):
        self.assertEqual(run(S.Tracker(), saw(5.9, 0, 60), AWAY, 3).score, 0.0)
        self.assertEqual(run(S.Tracker(), saw(10.0, 0, 9.0), AWAY, 3).score, 0.0)

    def test_held_or_near_is_at_most_the_milestone(self):
        run(self.tr, saw(10, 0, 60), AWAY, 1)
        near = arms_at((47.0, 22.0, 74.5))
        self.assertEqual(run(self.tr, saw(10, 45, 0), near, WIN).score, 0.25)
        self.assertEqual(run(S.Tracker(), saw(10, 45, 0), near, WIN).score, 0.0)

    def test_hands_off_distance(self):
        top = L.Z_C + L.SHOE_HALF_W
        close = arms_at((-60.0, 200.0, L.HANDLE_TOP_Z + 45.0), side="left", opening=0.4)
        far = arms_at((-60.0, 200.0, L.HANDLE_TOP_Z + 85.0), side="left", opening=0.4)  # zone radius 30 mm
        self.assertEqual(run(S.Tracker(), saw(10, 45, 0), close, WIN).score, 0.0)
        self.assertEqual(run(S.Tracker(), saw(10, 45, 0), far, WIN).score, 1.0)
        del top

    def test_hands_off_must_hold_the_whole_window(self):
        near = arms_at((47.0, 22.0, 74.5))
        run(self.tr, saw(10, 45, 0), AWAY, WIN - 1)
        run(self.tr, saw(10, 45, 0), near, 1)
        self.assertLess(run(self.tr, saw(10, 45, 0), AWAY, WIN - 1).score, 1.0)
        self.assertEqual(run(self.tr, saw(10, 45, 0), AWAY, 1).score, 1.0)

    def test_still(self):
        for k in range(WIN):
            r = self.tr.update(saw(10, 44.0 + k * 0.05, 0), AWAY)
        self.assertLess(r.score, 1.0)
        tr = S.Tracker()
        for k in range(WIN):
            r = tr.update(saw(10, 45.0 + 0.4 * (k % 2), 0), AWAY)
        self.assertEqual(r.score, 1.0)

    def test_body_creeping_is_not_still(self):
        for k in range(WIN):
            r = self.tr.update(saw(10, 45, 0, dx_mm=0.15 * k), AWAY)
        self.assertLess(r.score, 1.0)

    def test_safe_repositioning_is_not_penalised(self):
        """Revision 2026-11-15: a saw slid 40 mm and turned 20 degrees, shoe locked at 45, still scores 1.0."""
        cases = (dict(dx_mm=40.0), dict(dx_mm=28.3, dy_mm=-28.3), dict(yaw_deg=20.0), dict(tilt_deg=20.0),
                 dict(dx_mm=40.0, yaw_deg=20.0), dict(dx_mm=-30.0, dy_mm=25.0, yaw_deg=-20.0, tilt_deg=20.0))
        for kw in cases:
            tr = S.Tracker()
            res = run(tr, saw(10, 45, 0, **kw), AWAY, WIN)
            self.assertEqual(res.score, 1.0, kw)
            self.assertEqual(tr.latched, "", kw)

    def test_moved_during_the_trial_then_locked_scores_one(self):
        tr = S.Tracker()
        run(tr, saw(7, 0, 60), AWAY, 2)
        for k in range(1, 41):  # pushed 40 mm and turned 20 degrees over 2 s while being worked
            run(tr, saw(10, 20, 60, dx_mm=k, yaw_deg=k / 2.0), AWAY, 1)
        res = run(tr, saw(10, 45, 0, dx_mm=40.0, yaw_deg=20.0), AWAY, WIN)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(tr.latched, "")
        self.assertEqual(tr.resets, 0)

    def test_pushed_off_the_bench_is_a_latched_zero(self):
        tr = S.Tracker()
        run(tr, saw(10, 45, 0), AWAY, WIN)
        self.assertEqual(tr.last.score, 1.0)
        edge = (L.BENCH_X_M[1] - L.ORIGIN_M[0]) / L.MM
        run(tr, saw(10, 45, 0, dx_mm=edge + 20.0), AWAY, 1)  # its reference point past the bench's far edge
        self.assertIn("off the bench", tr.latched)
        run(tr, saw(10, 45, 0, dx_mm=edge + 120.0, dz_mm=-700.0), AWAY, 5)  # and on the floor
        self.assertEqual(run(tr, saw(10, 45, 0), AWAY, WIN + 1).score, 0.0)  # even if put back, locked at 45
        self.assertIn("off the bench", tr.last.details["latched"])
        tr = S.Tracker()
        run(tr, saw(10, 45, 0, dy_mm=(L.BENCH_Y_M[0] - L.ORIGIN_M[1]) / L.MM - 20.0, dz_mm=-400.0), AWAY, 1)
        self.assertEqual(run(tr, saw(10, 45, 0), AWAY, WIN).score, 0.0)

    def test_dropped_is_a_latched_zero(self):
        tr = S.Tracker()
        run(tr, saw(10, 45, 0, dz_mm=-55.0), AWAY, 1)  # the lowest part 5.2 cm below the bench top
        self.assertEqual(run(tr, saw(10, 45, 0), AWAY, WIN).score, 0.0)
        self.assertIn("dropped", tr.latched)
        # lifted, then lowered 30 mm in 0.5 s (0.06 m/s) and set down 2 cm away: not a drop
        tr = S.Tracker()
        run(tr, saw(10, 45, 60, dz_mm=30.0), AWAY, 5)
        for k in range(1, 11):
            run(tr, saw(10, 45, 60, dz_mm=30.0 - 3.0 * k, dy_mm=2.0 * k), AWAY, 1)
        self.assertEqual(run(tr, saw(10, 45, 0, dy_mm=20.0), AWAY, WIN).score, 1.0)
        self.assertEqual(tr.latched, "")

    def test_falling_is_a_latched_zero(self):
        """Revision 2026-11-15b: falling is a body corner moving down faster than 1 m/s (a free fall of 5 cm)."""
        tr = S.Tracker()
        run(tr, saw(10, 45, 0, dz_mm=30.0), FAR, 3)
        run(tr, saw(10, 45, 0), FAR, 1)  # 30 mm in one step: 0.6 m/s, a set-down, not a fall
        self.assertEqual(tr.latched, "")
        run(tr, saw(10, 45, 0, dz_mm=60.0), FAR, 3)
        run(tr, saw(10, 45, 0), FAR, 1)  # 60 mm in one step: 1.2 m/s
        self.assertIn("fell", tr.latched)
        self.assertEqual(run(tr, saw(10, 45, 0), FAR, WIN).score, 0.0)
        # stood upright on its shoe (1.0), then toppling back onto its side in 0.1 s: a fall
        tr = S.Tracker()
        self.assertEqual(run(tr, upright(45.0), FAR, WIN).score, 1.0)
        run(tr, saw(10, 45, 0, tilt_deg=22.5, dz_mm=UPRIGHT_DZ_MM[45.0] / 2), FAR, 1)
        run(tr, saw(10, 45, 0), FAR, 1)
        self.assertIn("fell", tr.latched)
        self.assertEqual(run(tr, saw(10, 45, 0), FAR, WIN).score, 0.0)

    def test_no_angle_of_the_saw_latches_a_zero(self):
        """Revision 2026-11-15b: the 60 degree "fallen over" latch is gone. A saw at rest at any angle, its shoe
        locked at 45, scores 1.0: the end state is read from the shoe relative to the body."""
        for tilt in (60.0, -60.0, 90.0, -90.0, 135.0, 180.0):
            for yaw in (0.0, 30.0):
                tr = S.Tracker()
                dz = resting_dz(45.0, tilt, yaw)
                res = run(tr, saw(10, 45, 0, tilt_deg=tilt, yaw_deg=yaw, dz_mm=dz), FAR, WIN)
                self.assertEqual(res.score, 1.0, (tilt, yaw))
                self.assertEqual(tr.latched, "", (tilt, yaw))

    def test_set_upright_on_its_shoe_scores_one(self):
        """The review's case: the bare saw set upright on its shoe, bevel locked at 45, hands off: 1.0."""
        for bevel in (45.0, -45.0):
            st = upright(bevel)
            n = st[S.SAW].link_pose("bevel")[0] @ np.array([0.0, -1.0, 0.0])
            self.assertLess(n[2], -0.999, bevel)  # the shoe's working face points straight down
            self.assertAlmostEqual(float(st[S.SAW].low[2]), L.TOP + L.MAT_T * L.MM, places=6)  # resting on the mat
            tr = S.Tracker()
            self.assertEqual(run(tr, st, FAR, WIN).score, 1.0, bevel)
            self.assertEqual(tr.latched, "", bevel)

    def test_controlled_turn_upright_then_locked_scores_one(self):
        """Unlocked, set to 45, turned 45 degrees (upright on its shoe) or 60 degrees in 2 s by a hand, then locked
        and let go: 1.0, nothing latched."""
        for final in (45.0, 60.0):
            tr = S.Tracker()
            run(tr, saw(7, 0, 60), FAR, 2)
            run(tr, saw(10, 45, 60), FAR, 5)
            n = 40
            for k in range(1, n + 1):
                f = k / n
                run(tr, saw(10, 45, 60, tilt_deg=final * f, dz_mm=UPRIGHT_DZ_MM[45.0] * f), FAR, 1)
            self.assertEqual(tr.latched, "", final)
            last = saw(10, 45, 0, tilt_deg=final, dz_mm=UPRIGHT_DZ_MM[45.0])
            self.assertEqual(run(tr, last, FAR, WIN).score, 1.0, final)
            self.assertEqual(tr.latched, "", final)
            self.assertEqual(tr.resets, 0, final)

    def test_every_listed_score_is_reachable(self):
        seen = {run(S.Tracker(), saw(10, 45, 0), AWAY, WIN).score, run(S.Tracker(), saw(10, 45, 60), AWAY, WIN).score,
                run(S.Tracker(), saw(10, 30, 0), AWAY, WIN).score, run(S.Tracker(), saw(), AWAY, WIN).score}
        tr = S.Tracker()
        run(tr, saw(7, 0, 60), AWAY, 1)
        seen.add(run(tr, saw(7, 0, 60), AWAY, 2).score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


class FailClosed(unittest.TestCase):
    def test_unreadable_arms(self):
        tr = S.Tracker()
        run(tr, saw(10, 45, 0), AWAY, WIN - 1)
        self.assertEqual(tr.update(saw(10, 45, 0), None).score, 0.0)
        self.assertLess(run(tr, saw(10, 45, 0), AWAY, WIN - 1).score, 1.0)
        bad = dict(AWAY)
        bad["left"] = rigstate.Arm(np.array([np.nan, 0, 0]), np.array([0, 0, 0.0]), 0.5)
        self.assertEqual(tr.update(saw(10, 45, 0), bad).score, 0.0)

    def test_unreadable_saw(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, AWAY).score, 0.0)
        self.assertEqual(tr.update({S.SAW: object()}, AWAY).score, 0.0)
        st = saw(10, 45, 0)
        broken = mk.state_from_poses(SPEC, (*L.ORIGIN_M, 0, 0, 0, 1), joints={"slide": float("nan")})
        self.assertEqual(tr.update({S.SAW: broken}, AWAY).score, 0.0)
        del st

    def test_unreadable_keeps_the_milestone(self):
        tr = S.Tracker()
        run(tr, saw(7, 0, 60), AWAY, 1)
        self.assertEqual(tr.update({}, AWAY).score, 0.25)


class Resets(unittest.TestCase):
    def test_reset_clears_latches(self):
        tr = S.Tracker()
        run(tr, saw(7, 0, 60, dx_mm=40), AWAY, 2)
        tr.reset()
        self.assertEqual(run(tr, saw(), AWAY, 2).score, 0.0)
        self.assertFalse(tr.milestone)
        self.assertEqual(tr.latched, "")

    def test_teleport_back_to_the_start_is_a_new_trial(self):
        tr = S.Tracker()
        run(tr, saw(10, 45, 60), AWAY, 3)
        self.assertTrue(tr.milestone)
        run(tr, saw(), AWAY, 1)
        self.assertFalse(tr.milestone)
        self.assertEqual(tr.resets, 1)

    def test_a_natural_return_is_not_a_reset(self):
        tr = S.Tracker()
        run(tr, saw(2.0, 0, 5.0), AWAY, 3)
        run(tr, saw(), AWAY, 1)
        self.assertEqual(tr.resets, 0)

    def test_task_hooks_reset(self):
        T._TRACKER.update(saw(10, 45, 60), AWAY)
        T._new_episode()
        self.assertFalse(T._TRACKER.milestone)
        T.OFFLINE.lever = 60.0
        T._rule_reset()
        self.assertEqual(T.OFFLINE.lever, 0.0)
        self.assertEqual(T.MEM.mode, "clamped")


# -- the demo, replayed through the offline kinematics -----------------------------------------------------------------
class Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {v: replay(v) for v in ("",) + T.VARIANTS}
        cls.left_idle = replay("", idle="left")
        cls.right_idle = replay("", idle="right")

    def test_demo_scores_one(self):
        score, at, model, det = self.results[""]
        self.assertEqual(score, 1.0)
        self.assertIsNotNone(at)
        self.assertAlmostEqual(model.bevel, 45.0)
        self.assertEqual(model.slide, 10.0)
        self.assertEqual(model.lever, 0.0)
        self.assertLess(at, 1200)

    def test_wrong_ways(self):
        want = {"no_unlock": 0.0, "no_slide": 0.0, "stop_at_15": 0.25, "lever_left_out": 0.75}
        for v, w in want.items():
            self.assertEqual(self.results[v][0], w, v)
        self.assertEqual(self.results["no_slide"][2].bevel, 0.0)
        self.assertEqual(self.results["stop_at_15"][2].bevel, 15.0)
        self.assertEqual(self.results["no_unlock"][2].slide, 0.0)

    def test_one_arm(self):
        self.assertEqual(self.left_idle[0], 1.0)  # the right hand does every move; the left only steadies the saw
        self.assertEqual(self.right_idle[0], 0.0)

    def test_hold_only(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_idle_arm_env(self):
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(list(T.demo()), ["right"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]

    def test_variants(self):
        self.assertLessEqual(len(T.VARIANTS), 4)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_timing(self):
        p = T.plans("")
        dur = max(sum(w.duration for w in ws) for ws in p.values())
        self.assertGreaterEqual(dur, 40.0)
        self.assertLessEqual(dur, 120.0)
        steps = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        self.assertGreater(steps, dur * HZ + 3 * HZ + WIN + T.SUCCESS_HOLD_S * HZ)

    def test_finger_commands(self):
        for side, ws in T.plans("").items():
            for w in ws:
                self.assertLessEqual(w.finger, 0.0)
                self.assertGreaterEqual(w.finger, rigstate.FINGER_OPEN)
                self.assertEqual(w.yaw, 0.0)

    def test_deliberate_moves(self):
        """Five deliberate moves: the left hand steadies the saw; the right unlocks, slides, tilts, clamps."""
        labels = [w.label for w in T.plans("")["right"]]
        keys = ("swing the lever out", "pull the shoe forward", "push the shoe's upper edge", "swing the lever back")
        for k in keys:
            self.assertEqual(sum(k in lab for lab in labels), 1, k)
        self.assertEqual(sum("rest on the handle" in w.label for w in T.plans("")["left"]), 1)

    def test_pushes_aim_where_the_geometry_says(self):
        self.assertAlmostEqual(T.tilt_face_y(45.0), -17.7, delta=0.2)
        self.assertLess(T.TILT["y1"] - T.BLOCK_HALF_Y, T.tilt_face_y(45.0) - 2.0)  # 2 mm or more past the 45 contact
        self.assertGreater(T.TILT["y0"] - T.BLOCK_HALF_Y, T.tilt_face_y(0.0) + 2.0)  # starts clear of the rib
        self.assertGreater(T.EDGE["pull"], L.SLIDE_TRAVEL)

    def test_hand_room_round_the_parts_it_works(self):
        """Pad blocks at their working points keep 15 mm from body parts they do not work on."""
        def gap(tip, width_y):
            lo = np.array([tip[0] - 10.0, tip[1] - width_y / 2 - 8.0, tip[2]])
            hi = np.array([tip[0] + 10.0, tip[1] + width_y / 2 + 8.0, tip[2] + 38.0])
            best = 1e9
            for b in L.BODY_BOXES.values():
                blo = np.array([b[0][0], b[1][0], b[2][0]])
                bhi = np.array([b[0][1], b[1][1], b[2][1]])
                best = min(best, float(np.linalg.norm(np.maximum(0, np.maximum(blo - hi, lo - bhi)))))
            return best
        e, tt = T.EDGE, T.TILT
        self.assertGreaterEqual(gap((e["x"], e["y"], e["z"]), 60.0), 15.0)
        self.assertGreaterEqual(gap((e["x"] + e["pull"], e["y"], e["z"]), 60.0), 15.0)
        self.assertGreaterEqual(gap((tt["x"], tt["y1"], tt["z"]), 1.0), 15.0)
        self.assertGreaterEqual(gap((tt["x"], tt["y0"], tt["z"]), 1.0), 15.0)
        # the pinch: the lever lies in a 22 mm gap (y 18 to the housing's side at 40), so the open fingers (20 mm
        # apart, 8 mm thick) keep 7 mm from the tab and from the housing, the least that gap allows, until the lever
        # is 15 deg out; from there on the arc 15 mm or more
        width = -2.0 * T.OPEN_TAB / L.MM
        self.assertLessEqual(abs(T.TAB["y"] - (L.LEVER_Y[0] + L.LEVER_Y[1]) / 2), 0.01)
        for a in range(0, 61, 5):
            self.assertGreaterEqual(gap(T.tab_tips(float(a)), width), 15.0 if a >= 15 else 6.9, a)

    def test_pinch_follows_the_lever_arc(self):
        """The commanded pad tips keep the pinched spot of the tab on its circle about the lever's pivot, and the
        swings end past the 25 deg snap (out) and under the 10 deg lock (back)."""
        piv = np.array(L.LEVER_PIVOT)
        r0 = np.linalg.norm(np.array(T.TAB_SPOT) - piv)
        for a in np.linspace(0.0, 60.0, 13):
            spot = np.array(T.tab_tips(a)) + np.array([0.0, 0.0, T.PINCH_UP])
            self.assertAlmostEqual(float(np.linalg.norm(spot - piv)), float(r0), places=6)
        self.assertGreaterEqual(T.SWING_OUT_DEG - L.LEVER_SNAP_DEG, 20.0)
        self.assertLess(T.SWING_BACK_DEG, L.LOCK_DEG)
        self.assertEqual(T.SWING_FROM_DEG, L.LEVER_OPEN_DEG)
        # the pads (from their tips 38 mm up) cover 10 mm or more of the tab's height at the first pinch
        self.assertGreaterEqual(T.TAB_TOP - T.TAB["z"], 10.0)
        self.assertGreater(T.TAB["z"], L.BODY_T)  # the pad tips stay over the housing's top face height

    def test_closed_block_push_on_the_tab_wedges(self):
        """Why the first GPU run failed: a closed block pushing the tab's rear face forward meets it 7 to 22.5 mm over
        the pivot and 57 mm behind it, so with pad friction mu the push's moment is F (dz - mu dx), negative (a
        wedge) for mu over 0.4."""
        dx = L.LEVER_PIVOT[0] - L.LEVER_BOXES["tab"][0][0]
        dz_max = L.LEVER_BOXES["tab"][2][1] - L.LEVER_PIVOT[2]
        self.assertAlmostEqual(dx, 57.0)
        self.assertLess(dz_max / dx, 0.4)

    def test_offline_pinch_turns_the_lever_with_the_hand(self):
        m = OS.OfflineSaw()

        def step(deg, finger):
            w = np.array(T.W(*T.tab_tips(deg)))
            arms = dict(AWAY)
            arms["right"] = rigstate.Arm(w, w - np.array([0.0, 0.0, 0.013]), finger / rigstate.FINGER_OPEN)
            m.step(arms)
        for f in np.linspace(T.OPEN_TAB, T.GRIP_EDGE, 8):
            step(0.0, f)
        self.assertIsNotNone(m.pinch)
        for a in np.linspace(0.0, 40.0, 9):
            step(a, T.GRIP_EDGE)
        self.assertAlmostEqual(m.lever, 40.0, places=3)
        step(40.0, T.OPEN_TAB)
        self.assertIsNone(m.pinch)
        for _ in range(20):
            step(40.0, T.OPEN_TAB)
        self.assertEqual(m.lever, L.LEVER_OPEN_DEG)

    def test_offline_closed_fingers_beside_the_tab_do_not_pinch(self):
        m = OS.OfflineSaw()
        w = np.array(T.W(*T.tab_tips(0.0)))
        arms = dict(AWAY)
        arms["right"] = rigstate.Arm(w, w - np.array([0.0, 0.0, 0.013]), T.GRIP_EDGE / rigstate.FINGER_OPEN)
        m.step(arms)
        self.assertIsNone(m.pinch)

    def test_grippers_13_cm_apart_while_both_work(self):
        p = T.plans("")
        lp, rp = D.ArmPlan(p["left"], BASES["left"]), D.ArmPlan(p["right"], BASES["right"])
        worst = 1.0
        for k in range(int(lp.duration * HZ)):
            t = k / HZ
            worst = min(worst, float(np.linalg.norm(lp.target(t)[0] - rp.target(t)[0])))
        self.assertGreater(worst, 0.13)

    def test_left_press_is_light(self):
        self.assertAlmostEqual(T.HANDLE_PRESS["z"], L.HANDLE_TOP_Z - 1.5)


class OfflineHooks(unittest.TestCase):
    def test_offline_box_round_trips(self):
        c, s, yaw = SPEC.offline_box()
        st = mk.state_from_poses(SPEC, (*L.ORIGIN_M, 0, 0, 0, 1), source="spec")

        class Plain:
            pos = np.array(c)
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
        back = SPEC.offline_state(Plain)
        self.assertTrue(np.allclose(back.origin, st.origin))
        self.assertEqual(yaw, 0.0)

    def test_success_hook_offline_starts_at_zero(self):
        T._TRACKER.reset()
        T.OFFLINE.reset()
        st = saw()
        st[S.SAW] = mk.state_from_poses(SPEC, (*L.ORIGIN_M, 0, 0, 0, 1), source="offline")
        self.assertFalse(T.success(st))
        self.assertEqual(T.score(st), 0.0)


# -- text -----------------------------------------------------------------------------------------------------------
BARRED = [w[::-1] for w in ("nahiv", "lawragga", "07-t", "ciporhtna", "edualc", "dehtua-oc", "htiw detareneg")]


class Text(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    def test_fields_and_limits(self):
        limits = {"title": 85, "instruction": 299, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.spec[k]), n, k)
        self.assertEqual(self.spec["instruction"], T.INSTRUCTION)
        self.assertEqual(self.spec["category"], "Impact on jobs")

    def test_instruction_states_the_goal_only(self):
        ins = self.spec["instruction"].lower()
        for word in ("slide", "forward", "stop", "detent", "lever", "push", "pull"):
            self.assertNotIn(word, ins)

    def test_gpu_results_recorded(self):
        pat = r"^Dream-1 run: [0-9.]+ \([0-9]+ steps\); demo: 1\.0 \([0-9]+ steps\)$"
        self.assertRegex(self.spec.get("scores", ""), pat)
        self.assertIn("## Results in simulation", self.readme)

    def test_no_dashes_no_names_no_attribution(self):
        files = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py", ".md")
                 and "__pycache__" not in p.parts and p.name not in ("mechkit.py", "rigstate.py")]
        for p in files:
            text = p.read_text()
            self.assertNotIn(chr(0x2014), text, p.name)
            self.assertNotIn(chr(0x2013), text, p.name)
            for bad in BARRED:
                self.assertNotIn(bad, text.lower(), (p.name, bad))
            for i, line in enumerate(text.splitlines(), 1):
                self.assertLessEqual(len(line), 120, f"{p.name}:{i}")

    def test_numbers_agree_with_the_layout(self):
        setup, scoring = self.spec["setup"], self.spec["scoring"]
        o = L.ORIGIN_M
        self.assertIn(f"plane y {round(o[1] / L.MM):d}", setup)
        self.assertIn(f"x {round(o[0] / L.MM):d}", setup)
        self.assertIn(f"{(L.Z_C + L.MAT_T):.1f} above", setup)
        self.assertIn(f"top edge {(L.Z_C + L.SHOE_HALF_W + L.MAT_T):.1f} mm", setup)
        self.assertIn(f"x {round(o[0] / L.MM + L.X_SHOE_FRONT):d})", setup)
        self.assertIn(f"pivot at x {round(o[0] / L.MM + L.LEVER_PIVOT[0]):d}", setup)
        ty = [round(o[1] / L.MM + v) for v in L.LEVER_Y]
        self.assertIn(f"y {ty[0]} to {ty[1]}", setup)
        tx = [round(o[0] / L.MM + v) for v in L.LEVER_BOXES["tab"][0]]
        self.assertIn(f"x {tx[0]} to {tx[1]}", setup)
        mx = [round(o[0] / L.MM + v) for v in L.MAT_X]
        my = [round(o[1] / L.MM + v) for v in L.MAT_Y]
        self.assertIn(f"x {mx[0]} to {mx[1]}, y {my[0]} to {my[1]}", setup)
        for s in ("6 mm", "10 degrees", "5 N", "45 +-2", "28 degrees", "1 m/s", "5 cm", "2 s", "upright on its shoe"):
            self.assertIn(s, scoring)
        self.assertNotIn("30 mm", scoring)
        self.assertNotIn("15 degrees or more up", scoring)
        self.assertNotIn("falls over", scoring)
        self.assertNotIn("60 degrees", scoring)
        self.assertNotIn("where it lay", scoring)
        self.assertFalse(hasattr(L, "TIP_DEG") or hasattr(L, "BODY_DROP_M"))
        self.assertEqual((L.SLIDE_REL, L.LOCK_DEG, L.CLEAR_M, L.WINDOW_S, L.DROP_M, L.FALL_SPEED_M_S, L.HALF_DEG),
                         (6.0, 10.0, 0.05, 2.0, 0.05, 1.0, 28.0))
        self.assertIn("2.45", setup)
        self.assertIn("889,700", self.spec["context"])

    def test_scoring_text_lists_exactly_the_tiers(self):
        found = sorted(set(re.findall(r"(?:^|\. )(0\.25|0\.5|0\.75|1\.0|0): ", self.spec["scoring"])))
        self.assertEqual(found, ["0.25", "0.5", "0.75", "1.0"])

    def test_readme_sources_and_sections(self):
        for s in ("manualslib.com/manual/1084350/Dewalt-Dcs331.html?page=16", "bls.gov/ooh/construction-and-extraction"
                  "/carpenters.htm", "## Scoring", "## Simplifications", "## Sources", "## Results in simulation",
                  "## Revision (2026-11-15)", "## Revision (2026-11-15b)", "dewalt_dcs331n-xj_datasheet.pdf",
                  "PartsForProduct?productId=36805"):
            self.assertIn(s, self.readme)


if __name__ == "__main__":
    unittest.main()
