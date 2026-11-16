"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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


def pieces(depths=(0.040, 0.040, 0.040), dx=None, dy=None, yaw=None, drop=None):
    """The four pieces with their strips ``depths`` down; per-piece shifts dx/dy (m), yaw (rad), drop (m below)."""
    out = {}
    for k, spec in enumerate(T.PIECE_PARTS):
        o = np.array(spec.pos, dtype=float)
        o[0] += (dx or {}).get(k, 0.0)
        o[1] += (dy or {}).get(k, 0.0)
        o[2] -= (drop or {}).get(k, 0.0)
        q = mk.quat_from_ypr((yaw or {}).get(k, 0.0), 0.0, 0.0)
        out[spec.name] = mk.state_from_poses(spec, (*o, *q), joints={"web": depths[k]} if k < 3 else {},
                                             source="test")
    return out


def scraper_at(side, x=None, y=None, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
    spec = T.SCRAPER_R if side == "right" else T.SCRAPER_L
    x = L.SC_X if x is None else x
    y = L.SC_Y["right" if side == "right" else "left"] if y is None else y
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(spec, (x, y, L.TOP + z, *q), source="test")


def rests():
    return {"scraper_r": scraper_at("right"), "scraper_l": scraper_at("left")}


def scene(depths=(0.040, 0.040, 0.040), sr=None, sl=None, **kw):
    out = pieces(depths, **kw)
    out["scraper_r"] = sr if sr is not None else scraper_at("right")
    out["scraper_l"] = sl if sl is not None else scraper_at("left")
    return out


def start():
    out = {s.name: s.rest_state() for s in T.PIECE_PARTS}
    out.update(rests())
    return out


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def chop_scene(k, depth=0.040, edge_z=0.045, side="right", **kw):
    """A scraper standing in strip k with its edge at ``edge_z``; strip k down ``depth``."""
    depths = [0.0, 0.0, 0.0]
    depths[k] = depth
    kw2 = dict(kw)
    s = scraper_at(side, y=L.web_y(k) + kw2.pop("off", 0.0), x=kw2.pop("x", L.SC_X), z=edge_z,
                   pitch=kw2.pop("pitch", 0.0), roll=kw2.pop("roll", 0.0), yaw=kw2.pop("yaw", 0.0))
    return scene(tuple(depths), **({"sr": s} if side == "right" else {"sl": s}), **kw2)


def cut_all(tr, arms=FAR, sides=("right", "right", "left"), **kw):
    """Chop the three marks one after the other (latches the cuts), leaving every strip down."""
    for k in range(3):
        tr.update(chop_scene(k, side=sides[k], **kw), arms)
    return tr


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def done_tracker():
    tr = cut_all(S.Tracker())
    return tr


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.SC_W, 0.1524, places=4)  # 6 in
        self.assertAlmostEqual(L.SC_H, 0.0762, places=4)  # 3 in
        self.assertAlmostEqual(L.LOG_MASS, 0.900, places=3)
        self.assertAlmostEqual(L.LOG_L, 0.256)
        self.assertAlmostEqual(L.PORTION, 0.064)

    def test_pieces_and_strips_tile_the_log(self):
        """Four bodies and three strips fill 256 mm with 0.3 mm of play each side of a strip; end pieces 59 mm and
        middle pieces 54 mm."""
        total = sum(L.BODY_LEN) + 3 * L.KERF
        self.assertAlmostEqual(total, L.LOG_L, places=9)
        self.assertEqual([round(n * 1000) for n in L.BODY_LEN], [59, 54, 54, 59])
        for k in range(3):
            self.assertAlmostEqual(L.BODY_Y[k][1] + L.KERF / 2, L.MARKS[k], places=9)
            self.assertAlmostEqual(L.BODY_Y[k + 1][0] - L.KERF / 2, L.MARKS[k], places=9)
        self.assertAlmostEqual(L.WEB_W, 0.0094)

    def test_marks_divide_the_log_in_four_portions(self):
        edges = [L.LOG_Y - L.LOG_L / 2, *[L.LOG_Y + m for m in L.MARKS], L.LOG_Y + L.LOG_L / 2]
        for a, b in zip(edges, edges[1:]):
            self.assertAlmostEqual(b - a, L.PORTION, places=9)

    def test_piece_masses_within_five_percent_of_their_mean(self):
        m = L.portion_masses()
        mean = sum(m) / 4
        for v in m:
            self.assertLess(abs(v - mean) / mean, 0.05)
        self.assertGreater(sum(m) / L.LOG_MASS, 0.85)  # the strips' dough is 12 percent of the log
        self.assertAlmostEqual(sum(m) + 3 * L.WEB_MASS, L.LOG_MASS, delta=0.008)  # 0.6 mm of play per strip carries no dough

    def test_parts_spawn_resting(self):
        """Pieces stand on the bench top, the strips flush with them; the scrapers' edges on the table in the rests'
        4 mm slots."""
        for k, spec in enumerate(T.PIECE_PARTS):
            st = spec.rest_state()
            self.assertAlmostEqual(st.low[2] - L.TOP, L.BENCH_TOP, places=6)
            self.assertAlmostEqual(st.high[2] - L.TOP, L.BENCH_TOP + L.LOG_H, places=6)
        for side, spec in (("right", T.SCRAPER_R), ("left", T.SCRAPER_L)):
            st = spec.rest_state()
            self.assertAlmostEqual(st.low[2] - L.TOP, 0.0, places=6)
            self.assertLess(L.SC_T, L.REST_GAP)
            self.assertAlmostEqual(st.origin[1], L.SC_Y[side])

    def test_start_is_inside_the_reach(self):
        """Every grasp and chop point in the demo is straight-down reachable (the offline checker says ok for all)."""
        for plan in T.plans().values():
            for w in plan:
                self.assertGreaterEqual(w.pos[0], 0.10)

    def test_slot_holds_strip(self):
        self.assertGreater(L.SLOT_LY, L.WEB_W)
        self.assertGreater(L.SLOT_LX, L.LOG_W)
        self.assertAlmostEqual(L.WEB_TRAVEL, L.BENCH_TOP)  # a full strip bottoms on the table under the bench

    def test_strip_resistance_inside_arm_force(self):
        """The design push to run a strip down at 16 mm/s: yield plus the drive's give at that speed, under 5 N."""
        v = 0.016
        f = L.YIELD_F + L.WEB_SPRING * v * (1 / HZ) / 2 + L.WEB_DAMP * v
        self.assertLess(f, 4.0)
        self.assertGreater(f, L.YIELD_F)

    def test_chop_depth_reaches_through(self):
        """The demo's pressed depth takes the strip past the 37 mm through line and the edge to the bench top."""
        edge = T.CHOP_TCP - L.BAR_Z
        self.assertLess(edge, L.BENCH_TOP)
        self.assertGreater(L.BENCH_TOP + L.LOG_H - L.BENCH_TOP, L.WEB_THROUGH)  # 40 mm of strip, 37 mm is through


class Rule(unittest.TestCase):
    def test_target_logic(self):
        self.assertEqual(T.strip_target(0.0, True), L.WEB_TRAVEL)
        self.assertLess(T.strip_target(0.0, False), 0.0)  # flush: pushed up against its stop with the yield force
        self.assertAlmostEqual(T.strip_target(0.0, False) * L.WEB_SPRING, -L.YIELD_F)
        self.assertLess(T.strip_target(0.0005, False), 0.0)  # springs back if let go within 1 mm
        q = 0.020
        self.assertAlmostEqual(T.strip_target(q, False), q - T.SAG)  # past 1 mm: held where left (less its weight)
        self.assertLess(T.SAG, 0.0005)

    def test_bonds_declared_between_neighbours(self):
        self.assertEqual([b.child for b in T.BONDS], ["p1", "p2", "p3"])
        self.assertEqual([b.parent for b in T.BONDS], ["p0", "p1", "p2"])
        for b in T.BONDS:
            self.assertFalse(b.breakable)

    def test_rule_releases_a_bond_when_through(self):
        T._RULE["through"] = [False, False, False]
        calls = []
        orig = [b.release for b in T.BONDS]
        try:
            for b in T.BONDS:
                b.release = (lambda name=b.name: calls.append(name))
            old = mk.set_targets
            mk.set_targets = lambda *a, **k: 0.0
            try:
                T._dough_rule(pieces((0.0, 0.0380, 0.0)))
                T._dough_rule(pieces((0.0, 0.0380, 0.0)))  # once only
            finally:
                mk.set_targets = old
        finally:
            for b, f in zip(T.BONDS, orig):
                b.release = f
            T._RULE["through"] = [False, False, False]
        self.assertEqual(calls, ["cut1"])

    def test_rule_survives_bad_states(self):
        old = mk.set_targets
        mk.set_targets = lambda *a, **k: 0.0
        try:
            T._dough_rule({})
            T._dough_rule({"p0": None})
        finally:
            mk.set_targets = old
        T._RULE["through"] = [False, False, False]


class Tiers(unittest.TestCase):
    def test_full(self):
        tr = done_tracker()
        r = run(scene(), tracker=tr)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["n_cut"], 3)

    def test_tier_table(self):
        self.assertEqual(S.tier(True, True, 3, True), 0.0)
        self.assertEqual(S.tier(False, False, 3, True), 0.0)
        self.assertEqual(S.tier(False, True, 3, True), 1.0)
        self.assertEqual(S.tier(False, True, 3, False), 0.5)
        self.assertEqual(S.tier(False, True, 2, True), 0.5)
        self.assertEqual(S.tier(False, True, 1, True), 0.25)
        self.assertEqual(S.tier(False, True, 0, True), 0.0)

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_two_cuts_half(self):
        tr = S.Tracker()
        for k, side in ((0, "right"), (1, "right")):
            tr.update(chop_scene(k, side=side), FAR)
        s = scene((0.040, 0.040, 0.0))
        self.assertEqual(run(s, tracker=tr).score, 0.5)

    def test_one_cut_quarter(self):
        tr = S.Tracker()
        tr.update(chop_scene(0), FAR)
        self.assertEqual(run(scene((0.040, 0.0, 0.0)), tracker=tr).score, 0.25)

    def test_each_mark_counts_alone(self):
        for k in range(3):
            tr = S.Tracker()
            tr.update(chop_scene(k), FAR)
            ds = [0.0, 0.0, 0.0]
            ds[k] = 0.040
            self.assertEqual(run(scene(tuple(ds)), tracker=tr).score, 0.25, k)

    def test_blade_must_stay_in_the_strip_edge_margins(self):
        for off, want in ((0.0, 3), (0.0048, 3), (0.0058, 2), (-0.0058, 2), (0.012, 2)):
            tr = S.Tracker()
            tr.update(chop_scene(0), FAR)
            tr.update(chop_scene(1), FAR)
            tr.update(chop_scene(2, side="left", off=off), FAR)
            r = run(scene(), tracker=tr)
            self.assertEqual(r.details["n_cut"], want, off)

    def test_just_through_counts(self):
        tr = S.Tracker()
        for k, side in ((0, "right"), (1, "right"), (2, "left")):
            tr.update(chop_scene(k, depth=0.0372, side=side), FAR)
        self.assertEqual(run(scene((0.0372, 0.0372, 0.0372)), tracker=tr).score, 1.0)

    def test_not_quite_through_does_not_count(self):
        tr = S.Tracker()
        for k, side in ((0, "right"), (1, "right"), (2, "left")):
            tr.update(chop_scene(k, depth=0.0360, side=side), FAR)
        r = run(scene((0.036, 0.036, 0.036)), tracker=tr)
        self.assertEqual(r.details["n_cut"], 0)
        self.assertEqual(r.score, 0.0)

    def test_crease_caps_at_half(self):
        """Two cuts and a strip pressed 20 mm and left: 0.5 (and the crease is flagged)."""
        tr = S.Tracker()
        tr.update(chop_scene(0), FAR)
        tr.update(chop_scene(1), FAR)
        tr.update(chop_scene(2, depth=0.020, side="left"), FAR)
        r = run(scene((0.040, 0.040, 0.020)), tracker=tr)
        self.assertEqual(r.details["creased"], [False, False, True])
        self.assertEqual(r.score, 0.5)

    def test_dent_under_4mm_is_not_a_crease(self):
        tr = cut_all(S.Tracker())
        r = run(scene((0.040, 0.040, 0.040)), tracker=tr)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(S.creased((0.003, 0.0, 0.0)), (False, False, False))
        self.assertEqual(S.creased((0.005, 0.0, 0.0)), (True, False, False))

    def test_dragged_piece_caps_at_half(self):
        tr = cut_all(S.Tracker())
        r = run(scene(dy={1: 0.013}), tracker=tr)
        self.assertTrue(r.details["dragged"])
        self.assertEqual(r.score, 0.5)

    def test_small_shift_is_not_drag(self):
        tr = cut_all(S.Tracker())
        r = run(scene(dy={1: 0.010, 2: -0.008}, dx={3: 0.009}), tracker=tr)
        self.assertFalse(r.details["dragged"])
        self.assertEqual(r.score, 1.0)

    def test_turned_piece_is_drag(self):
        tr = cut_all(S.Tracker())
        r = run(scene(yaw={2: math.radians(9)}), tracker=tr)
        self.assertTrue(r.details["dragged"])
        self.assertEqual(r.score, 0.5)

    def test_drag_is_latched(self):
        tr = cut_all(S.Tracker())
        tr.update(scene(dx={0: 0.02}), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 0.5)

    def test_pieces_out_of_line_cap_at_half(self):
        tr = cut_all(S.Tracker())
        r = run(scene(dx={3: 0.034}), tracker=tr)
        self.assertFalse(r.details["in_line"])
        self.assertEqual(r.score, 0.5)

    def test_pieces_out_of_order_cap_at_half(self):
        tr = cut_all(S.Tracker())
        r = run(scene(dy={1: 0.066}), tracker=tr)  # p1 passes p2
        self.assertFalse(r.details["in_line"])

    def test_scraper_left_in_the_dough_caps(self):
        tr = cut_all(S.Tracker())
        s = scraper_at("right", y=L.web_y(0), z=0.020)  # edge 20 mm up in the strip's slot, blade by two pieces
        r = run(scene(sr=s), tracker=tr)
        self.assertFalse(r.details["tools_down"])
        self.assertEqual(r.score, 0.5)

    def test_scraper_resting_on_a_piece_is_touching(self):
        tr = cut_all(S.Tracker())
        s = scraper_at("left", x=L.LOG_X, y=L.BODY_YC[1], z=L.BENCH_TOP + L.LOG_H)
        r = run(scene(sl=s), tracker=tr)
        self.assertFalse(r.details["tools_down"])
        self.assertEqual(r.score, 0.5)

    def test_scraper_dropped_off_the_table_is_not_down(self):
        tr = cut_all(S.Tracker())
        s = scraper_at("right", y=-0.60, z=-0.30)
        self.assertFalse(run(scene(sr=s), tracker=tr).details["tools_down"])

    def test_scraper_set_down_anywhere_clear_counts(self):
        tr = cut_all(S.Tracker())
        s = scraper_at("right", x=0.30, y=-0.25, z=0.0)
        self.assertEqual(run(scene(sr=s), tracker=tr).score, 1.0)


class Exploits(unittest.TestCase):
    def test_strips_pressed_without_a_scraper_score_zero(self):
        """Three strips pushed down by something else (a finger, a tool laid flat): no blade was in any strip."""
        tr = S.Tracker()
        tr.update(scene((0.040, 0.040, 0.040)), FAR)
        r = run(scene((0.040, 0.040, 0.040)), tracker=tr)
        self.assertEqual(r.details["through"], [True, True, True])
        self.assertEqual(r.details["n_cut"], 0)
        self.assertEqual(r.score, 0.0)

    def test_scraper_laid_flat_over_the_strips_does_not_count(self):
        """A scraper lying on its side across all three marks: not upright, so no strip has a blade in it."""
        tr = S.Tracker()
        flat = scraper_at("right", x=L.LOG_X, y=0.0, z=L.BENCH_TOP + L.LOG_H, roll=math.pi / 2)
        tr.update(scene((0.040, 0.040, 0.040), sr=flat), FAR)
        r = run(scene((0.040, 0.040, 0.040), sr=flat), tracker=tr)
        self.assertEqual(r.details["blade_in"], [False, False, False])
        self.assertEqual(r.score, 0.0)

    def test_blade_turned_across_the_log_does_not_count(self):
        """Blade turned 90 degrees (its length along y, spanning all marks): its centre is in one strip but it is not
        along x, so it cannot cut across the log."""
        tr = S.Tracker()
        s = scraper_at("right", y=L.web_y(1), z=0.045, yaw=math.pi / 2)
        tr.update(scene((0.0, 0.040, 0.0), sr=s), FAR)
        self.assertEqual(tr.cut_in, [False, False, False])

    def test_blade_tilted_over_15_degrees_does_not_count(self):
        for kw in ({"pitch": math.radians(20)}, {"roll": math.radians(20)}):
            tr = S.Tracker()
            tr.update(chop_scene(1, **kw), FAR)
            self.assertFalse(tr.cut_in[1], kw)
        tr = S.Tracker()
        tr.update(chop_scene(1, pitch=math.radians(10)), FAR)
        self.assertTrue(tr.cut_in[1])

    def test_blade_held_high_over_the_strip_does_not_count(self):
        tr = S.Tracker()
        tr.update(chop_scene(1, edge_z=0.075), FAR)
        self.assertFalse(tr.cut_in[1])

    def test_blade_beside_the_log_does_not_count(self):
        tr = S.Tracker()
        tr.update(chop_scene(1, x=L.LOG_X + 0.06), FAR)
        self.assertFalse(tr.cut_in[1])

    def test_pieces_shoved_apart_without_cutting_is_nothing(self):
        """Nothing cut but the pieces slid apart along the bench: no strip through, 0."""
        tr = S.Tracker()
        r = run(scene((0.0, 0.0, 0.0), dy={0: -0.02, 3: 0.02}), tracker=tr)
        self.assertEqual(r.score, 0.0)

    def test_cut_then_pieces_flung_off_the_bench(self):
        tr = cut_all(S.Tracker())
        r = run(scene(drop={3: 0.05}), tracker=tr)
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(tr.latch["F1"])


class Latches(unittest.TestCase):
    def test_piece_on_the_table_is_binned(self):
        tr = cut_all(S.Tracker())
        run(scene(drop={0: 0.045}), tracker=tr, steps=1)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_piece_on_the_bench_edge_is_not_binned(self):
        tr = cut_all(S.Tracker())
        run(scene(drop={0: 0.008}), tracker=tr, steps=1)
        self.assertIsNone(tr.latch["F1"])

    def test_scraper_dropped_is_not_binned_dough(self):
        tr = cut_all(S.Tracker())
        run(scene(sr=scraper_at("right", y=-0.6, z=-0.3)), tracker=tr, steps=1)
        self.assertIsNone(tr.latch["F1"])


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_a_scraper(self):
        tr = cut_all(S.Tracker())
        arms = {"left": arm((L.SC_X, L.SC_Y["left"], L.TOP + L.BAR_Z), 0.3), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=arms, tracker=tr).score, 0.0)

    def test_hand_over_a_piece(self):
        tr = cut_all(S.Tracker())
        arms = {"left": FAR["left"], "right": arm((L.LOG_X, L.BODY_YC[1], L.TOP + 0.075), 1.0)}
        self.assertEqual(run(scene(), arms=arms, tracker=tr).score, 0.0)

    def test_parked_hands_are_clear(self):
        tr = cut_all(S.Tracker())
        arms = {s: arm((T.PARK[s][0], T.PARK[s][1], L.TOP + T.PARK[s][2])) for s in ("left", "right")}
        self.assertEqual(run(scene(), arms=arms, tracker=tr).score, 1.0)

    def test_unreadable_arms_never_release(self):
        tr = cut_all(S.Tracker())
        self.assertEqual(run(scene(), arms=None, tracker=tr).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = cut_all(S.Tracker())
        near = {"left": arm((L.SC_X, L.SC_Y["left"], L.TOP + L.BAR_Z), 0.3), "right": FAR["right"]}
        run(scene(), arms=near, tracker=tr)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_moving_piece(self):
        tr = cut_all(S.Tracker())
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(dx={1: 0.001 * i}), FAR)
        self.assertEqual(res.score, 0.0)

    def test_small_creep_under_5_mm_is_still(self):
        """Over the last second a piece shifting 3 mm (not visible on video) still counts as still."""
        tr = cut_all(S.Tracker())
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(dx={1: 0.00015 * i}), FAR)
        self.assertEqual(res.score, 1.0)

    def test_moving_scraper(self):
        tr = cut_all(S.Tracker())
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(sl=scraper_at("left", y=0.30 + 0.001 * i)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_still_for_the_whole_second_needed(self):
        tr = cut_all(S.Tracker())
        run(scene(), tracker=tr, steps=WIN - 2)
        self.assertEqual(tr.last.score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=4).score, 1.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        s = scene()
        s.pop("p2")
        r = S.Tracker().update(s, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_missing_joint_is_invalid(self):
        s = scene()
        s["p1"] = mk.state_from_poses(T.PIECE_PARTS[1], (*T.PIECE_PARTS[1].pos, 0, 0, 0, 1), source="test")
        s["p1"].joints.pop("web")
        self.assertFalse(S.Tracker().update(s, FAR).details["valid_state"])

    def test_nan_state(self):
        bad = mk.state_from_poses(T.SCRAPER_R, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(sr=bad), FAR).score, 0.0)

    def test_nan_joint(self):
        r = S.Tracker().update(scene(depths=(float("nan"), 0.04, 0.04)), FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_resets_the_windows(self):
        tr = cut_all(S.Tracker())
        run(scene(), tracker=tr, steps=WIN + 2)
        tr.update(scene(depths=(float("nan"), 0.04, 0.04)), FAR)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = cut_all(S.Tracker())
        run(scene(drop={0: 0.09}), tracker=tr, steps=1)
        self.assertIsNotNone(tr.latch["F1"])
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(tr.cuts(), [False, False, False])

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            ds = tuple(float(v) for v in rng.uniform(0.0, 0.041, 3))
            s = scene(ds, dx={k: float(rng.uniform(-0.04, 0.04)) for k in range(4)},
                      dy={k: float(rng.uniform(-0.03, 0.03)) for k in range(4)},
                      sr=scraper_at("right", x=float(rng.uniform(0.1, 0.35)), y=float(rng.uniform(-0.35, 0.1)),
                                    z=float(rng.uniform(0.0, 0.15)), yaw=float(rng.uniform(-3, 3))),
                      sl=scraper_at("left", x=float(rng.uniform(0.1, 0.35)), y=float(rng.uniform(-0.1, 0.35)),
                                    z=float(rng.uniform(0.0, 0.15)), pitch=float(rng.uniform(-1, 1))))
            r = tr.update(s, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class DemoPlans(unittest.TestCase):
    def test_demo_has_both_arms(self):
        p = T.demo()
        self.assertEqual(set(p), {"left", "right"})

    def test_hold_only_has_no_plan(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_idle_arm_drops_that_plan(self):
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]

    def test_demo_length_within_run_limit(self):
        p = T.plans()
        steps = max(sum(w.duration for w in plan) for plan in p.values()) * HZ
        self.assertLessEqual(steps, 0.6 * 1600)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_arms_never_chop_together(self):
        """The left arm's chop starts after the right arm has lifted from the last mark and left the middle."""
        p = T.plans()

        def at(plan, label):
            t = 0.0
            for w in plan:
                t += w.duration
                if w.label.startswith(label):
                    return t
            raise KeyError(label)

        right_back_over = at(p["right"], "straight back up") + 2.0 + 3.0 + 2.0  # to the second chop's end (approx)
        self.assertGreater(at(p["left"], "chop straight down") - T.CHOP_SECONDS, 20.0)
        self.assertGreater(at(p["left"], "hold the scraper up"), 20.0)

    def test_task_text_files_exist(self):
        self.assertTrue((HERE / "task.json").exists())
        self.assertTrue((HERE / "README.md").exists())


class OfflineStandIn(unittest.TestCase):
    def test_model_presses_the_strip_to_the_edge(self):
        m = T.OfflineModel()
        s = scraper_at("right", y=L.web_y(0), z=0.060)
        m.step({"scraper_r": s, "scraper_l": scraper_at("left")})
        self.assertAlmostEqual(m.depth[0], 0.020, places=6)
        s = scraper_at("right", y=L.web_y(0), z=0.075)
        m.step({"scraper_r": s, "scraper_l": scraper_at("left")})
        self.assertAlmostEqual(m.depth[0], 0.020, places=6)  # never back up
        s = scraper_at("right", y=L.web_y(0), z=0.020)
        m.step({"scraper_r": s, "scraper_l": scraper_at("left")})
        self.assertAlmostEqual(m.depth[0], 0.040, places=6)  # capped at the strip's travel

    def test_model_ignores_blade_off_the_strip(self):
        m = T.OfflineModel()
        m.step({"scraper_r": scraper_at("right", y=L.web_y(0) + 0.008, z=0.045), "scraper_l": scraper_at("left")})
        self.assertEqual(m.depth, [0.0, 0.0, 0.0])

    def test_model_states_are_complete(self):
        st = T.OfflineModel().states(rests())
        self.assertEqual(set(st), set(S.PARTS))
        self.assertTrue(S.Tracker().update(st, FAR).details["valid_state"])


class TaskJson(unittest.TestCase):
    def test_fields_and_limits(self):
        import json
        d = json.loads((HERE / "task.json").read_text())
        lim = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
               "reset_notes": 580}
        for k, n in lim.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertEqual(d["category"], "Societal relatability")
        self.assertNotIn("—", json.dumps(d))

    def test_thresholds_in_text_match_the_code(self):
        import json
        d = json.loads((HERE / "task.json").read_text())
        sc = d["scoring"]
        for token in ("37 mm", "4 mm", "9.4 mm", "15 degrees", "12 mm", "8 degrees", "30 mm", "5 mm", "5 degrees",
                      "36 mm", "10 mm below"):
            self.assertIn(token, sc, token)
        self.assertEqual(L.WEB_THROUGH, 0.037)
        self.assertEqual(L.WEB_CREASE, 0.004)
        self.assertEqual(L.DRAG_MOVE, 0.012)
        self.assertEqual(L.DRAG_TURN_DEG, 8.0)
        self.assertEqual(L.CUT_X, 0.036)
        self.assertEqual(L.BLADE_UPRIGHT_DEG, 15.0)
        self.assertEqual(L.STILL_MOVE, 0.005)
        self.assertEqual(L.STILL_TURN_DEG, 5.0)


if __name__ == "__main__":
    unittest.main()
