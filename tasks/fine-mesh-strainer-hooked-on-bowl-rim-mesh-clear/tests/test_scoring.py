"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import random
import re
import subprocess
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
BEAD_Z = L.BEAD_TOP_W  # world height of the rim's top


def strainer_at(s=L.HOOK_AIM_S, dx=0.0, dz=0.0, yaw=L.ST_YAW, pitch=0.0, roll=0.0, z=None, bowl=(0.0, 0.0)):
    """The strainer hooked across the bowl: its ring ``s`` along the axis (toward the handle) and ``dx`` across it from
    the bowl's axis, ``dz`` above the rest plane, turned ``yaw`` about the vertical (round the ring's centre)."""
    c = np.array([L.BOWL_X + bowl[0], L.BOWL_Y + bowl[1]])
    a = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-a[1], a[0]])
    xy = c + a * s + n * dx
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.STRAINER_PART, (xy[0], xy[1], (BEAD_Z + dz) if z is None else z, *q), source="test")


def bowl_at(dx=0.0, dy=0.0, dz=0.0, pitch=0.0):
    q = mk.quat_from_ypr(0.0, pitch, 0.0)
    return mk.state_from_poses(T.BOWL_PART, (L.BOWL_X + dx, L.BOWL_Y + dy, L.BOWL_Z0 + dz, *q), source="test")


def scene(st=None, bw=None):
    return {S.STRAINER: st if st is not None else strainer_at(), S.BOWL: bw if bw is not None else bowl_at()}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.30, 0.40, L.TOP + 0.20)), "right": arm((0.10, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def score_of(states, arms=FAR):
    return run(states, arms).score


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(2 * L.BEAD_OUT, 0.220)  # 220 mm across the rim
        self.assertAlmostEqual(2 * L.R_RIM_IN, 0.200)
        self.assertAlmostEqual(L.BOWL_H, 0.105)
        self.assertAlmostEqual(2 * L.RING_R, 0.130)
        self.assertAlmostEqual(L.RING_DEPTH + L.RING_TOP, 0.057)
        self.assertAlmostEqual(L.STRAINER_LEN, 0.314, places=4)
        self.assertAlmostEqual(L.HK_GAP, 0.030)
        self.assertAlmostEqual(L.BOWL_MASS, 0.82)
        self.assertLess(2 * L.RING_R + 0.01, 2 * L.R_RIM_IN)  # the ring goes into the bowl with 5 mm to spare

    def test_stock_is_half_a_litre(self):
        r0 = L.R_FLOOR_IN + L.WALL_SLOPE * 0.0  # inner radius at the floor
        r1 = L.R_FLOOR_IN + L.WALL_SLOPE * L.STOCK_DEPTH
        vol = math.pi * L.STOCK_DEPTH / 3 * (r0 * r0 + r0 * r1 + r1 * r1)
        self.assertAlmostEqual(vol * 1000, 0.52, delta=0.03)
        self.assertAlmostEqual(L.BOWL_MASS - 0.30, vol * 1000, delta=0.03)  # 1 kg per litre

    def test_hooked_rest_state_gives_one(self):
        r = S.read(scene())
        self.assertTrue(S.hook_on(S.hook_reading(r)))
        self.assertTrue(S.handle_on(S.handle_reading(r)))
        self.assertTrue(S.mesh_clear(r))
        lo, hi = S.mesh_height(r)
        self.assertAlmostEqual(lo, 0.058, places=4)  # 58 mm above the foot, 26 mm over the stock
        self.assertAlmostEqual(lo - L.STOCK_TOP_B, 0.026, places=4)

    def test_hooking_window(self):
        """The scorer's window (the inner leg face inside 100 mm, the outer face outside 110 mm) is s = -10 to +10 mm; the
        flared wall closes it to -6.5 mm for the inner leg's foot."""
        for s, expect in ((-0.0100, True), (-0.0101, False), (0.0100, True), (0.0101, False), (0.0, True)):
            r = S.read(scene(strainer_at(s=s)))
            self.assertEqual(S.hook_on(S.hook_reading(r)), expect, s)

    def test_inner_leg_clears_the_flared_wall(self):
        """With the ring on the aim point the inner leg's foot (12 mm down, 1.5 mm thick) is clear of the wall."""
        foot_z = L.HK_LEG_IN
        wall_at_foot = L.R_RIM_IN - L.WALL_SLOPE * foot_z
        leg_face = -L.HK_IN - L.HOOK_AIM_S
        self.assertGreater(wall_at_foot - leg_face, 0.004)
        self.assertGreater(L.BEAD_OUT - 0.0, 0.0)
        outer = -L.HK_OUT - L.HOOK_AIM_S
        self.assertGreater(outer - L.BEAD_OUT, 0.007)

    def test_start_is_stable_and_clear(self):
        st = T.STRAINER_PART.rest_state()
        self.assertAlmostEqual(st.low[2] - L.TOP, 0.0, places=6)  # the mesh floor lies on the table
        bw = T.BOWL_PART.rest_state()
        self.assertAlmostEqual(bw.low[2] - L.TOP, L.MAT_SIZE[2], places=6)  # the foot on the mat
        com = T.STRAINER_PART.center_of_mass()
        self.assertLess(math.hypot(com[0], com[1]), L.RING_R)  # its centre of mass over the ring's floor
        self.assertLess(com[2], 0.0)
        # the hook end clears the bowl's rim by 18 mm or more (top view); the handle's end is inside |y| 0.45
        self.assertGreater(st.low[1] - bw.high[1], 0.018)
        self.assertLessEqual(st.high[1], 0.445)

    def test_carry_clears_the_rim(self):
        """At carry height the mesh is over the bead's top, the hook's feet too, and the pad tips over the flanges."""
        lip = T.CARRY_LIP
        self.assertGreaterEqual(lip - L.RING_DEPTH, L.BEAD_TOP_W - L.TOP + 0.012)
        self.assertGreaterEqual(lip - L.HK_LEG_OUT, L.BEAD_TOP_W - L.TOP + 0.040)
        self.assertAlmostEqual(T.CARRY_TCP, lip + 0.013, places=6)

    def test_grip_geometry(self):
        """The pads close on the 18 mm handle's flanges next to the ring: their tips (15 mm wide, 6 mm thick fingers) clear
        of the ring's wall and, hung on the bowl, inside the rim's opening; the 94 mm opening wide of the handle."""
        self.assertGreater(L.GRIP_X - 0.0075, L.RING_R + 0.008)  # pad tips 8 mm clear of the ring's wall
        tip_r = math.hypot(L.GRIP_X + 0.0075, L.HD_W / 2 + 0.006)
        self.assertLess(tip_r, L.R_RIM_IN - 0.008)  # 8 mm inside the rim when hung (they are level with the bead's top)
        self.assertLess(L.GRIP_X - L.RING_R, 0.025)  # next to the ring: the twisting load is half that at the end
        self.assertLess(-T.OPEN_H * 2, 0.0939)
        self.assertGreater(-T.OPEN_H * 2, L.HD_W + 0.015)  # open 42 mm: 12 mm each side of the handle
        self.assertLess(-T.GRIP_H * 2, L.HD_W)  # commanded narrower than the handle
        self.assertGreater(-T.GRIP_H * 2, 0.001)  # never fully closed
        self.assertAlmostEqual(T.GRIP_TCP - T.PAD_TIP, L.RING_DEPTH)  # tips level with the flanges' bottoms

    def test_reach(self):
        """Every waypoint's TCP is in the left arm's table-height, hover and carry envelope (yamkin, straight down)."""
        left = T.plans()["left"]
        for w in left:
            x, y, z = w.pos
            self.assertGreater(y, 0.05)
            self.assertLess(x, 0.45)
            self.assertLess(z - L.TOP, 0.2)


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(scene())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["hook_on"] and r.details["handle_on"] and r.details["mesh_clear"])

    def test_start_scores_zero(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_tier_table(self):
        full = (False, True, True, True, True, True, True, True)
        self.assertEqual(S.tier(False, True, True, True, True, True, True, True), 1.0)
        self.assertEqual(S.tier(True, True, True, True, True, True, True, True), 0.0)  # latched
        self.assertEqual(S.tier(False, False, True, True, True, True, True, True), 0.0)  # not still
        self.assertEqual(S.tier(False, True, False, True, True, True, True, True), 0.5)  # no hook
        self.assertEqual(S.tier(False, True, True, False, True, True, True, True), 0.5)  # no handle
        self.assertEqual(S.tier(False, True, True, True, True, True, False, True), 0.5)  # not level
        self.assertEqual(S.tier(False, True, True, True, False, True, True, True), 0.0)  # ring out
        self.assertEqual(S.tier(False, True, True, True, True, False, True, True), 0.0)  # mesh low (in the stock)
        self.assertEqual(S.tier(False, True, False, False, False, False, False, False), 0.0)
        self.assertEqual(full[0], False)

    def test_hook_just_off_the_bead_is_half(self):
        self.assertEqual(score_of(scene(strainer_at(s=0.012))), 0.5)  # outer leg on the bead
        self.assertEqual(score_of(scene(strainer_at(s=-0.012))), 0.5)  # inner leg on the bead

    def test_strap_height_edges(self):
        self.assertEqual(score_of(scene(strainer_at(dz=0.0059))), 1.0)
        self.assertEqual(score_of(scene(strainer_at(dz=0.0061))), 0.5)  # the hook's top not within 6 mm: rests, mesh clear
        self.assertEqual(score_of(scene(strainer_at(dz=-0.0019))), 1.0)
        self.assertEqual(score_of(scene(strainer_at(dz=-0.0021))), 0.5)  # sunk 2 mm into the bead

    def test_handle_off_the_far_bead_is_half(self):
        """Pitched so the handle end is 12 mm over the far bead: hook gap stays small, the handle gap does not."""
        r = S.read(scene(strainer_at(pitch=-math.radians(6.0))))
        self.assertFalse(S.handle_on(S.handle_reading(r)))
        self.assertEqual(score_of(scene(strainer_at(pitch=-math.radians(6.0)))), 0.5)

    def test_tilted_rests_still_half(self):
        self.assertEqual(score_of(scene(strainer_at(dz=0.006, pitch=math.radians(3.3)))), 0.5)

    def test_sitting_in_the_stock_is_zero(self):
        """The mesh down in the stock is the failure the task is about (it clogs): no credit."""
        z = L.BOWL_Z0 + L.STOCK_TOP_B + L.RING_DEPTH
        r = run(scene(strainer_at(z=z)))
        self.assertTrue(r.details["in_stock"])
        self.assertEqual(r.score, 0.0)

    def test_mesh_clear_edges(self):
        base = L.BOWL_Z0 + L.STOCK_TOP_B + L.MESH_ABOVE_STOCK + L.RING_DEPTH
        self.assertEqual(score_of(scene(strainer_at(z=base + 0.0002))), 0.5)  # 15 mm over the stock, not on the rim
        self.assertEqual(score_of(scene(strainer_at(z=base - 0.0005))), 0.0)  # 14.5 mm: too near the stock

    def test_mesh_must_be_below_the_rim_by_20(self):
        z = L.BOWL_Z0 + L.BEAD_TOP_B - L.MESH_BELOW_RIM + L.RING_DEPTH
        self.assertEqual(score_of(scene(strainer_at(z=z + 0.002))), 0.0)  # held up above the rim: the mesh is not 20 mm down
        self.assertEqual(score_of(scene(strainer_at(z=z - 0.002))), 0.5)

    def test_ring_offset_edges(self):
        self.assertEqual(score_of(scene(strainer_at(s=0.0, dx=0.0099))), 1.0)
        self.assertEqual(score_of(scene(strainer_at(s=0.0, dx=0.0199))), 1.0)  # axis misses the bowl's by 20 mm: just level
        self.assertEqual(score_of(scene(strainer_at(s=0.0, dx=0.0215))), 0.5)  # 21.5 mm: not level
        self.assertEqual(score_of(scene(strainer_at(s=0.0, dx=0.0295))), 0.5)  # ring in (30 mm), axis off by 29.5
        self.assertEqual(score_of(scene(strainer_at(s=0.0, dx=0.031))), 0.0)  # ring out, mesh clear, not in the stock

    def test_hook_must_be_on_the_right_rim(self):
        """Any line across the round bowl counts, as long as the hook is on the right rim (within 45 degrees of -y);
        the hook on another part of the rim is 0.5."""
        for deg in (0, 30, -30, 44, -44):
            self.assertEqual(score_of(scene(strainer_at(yaw=L.ST_YAW + math.radians(deg)))), 1.0, deg)
        for deg in (46, -46, 90, 180, -75):
            self.assertEqual(score_of(scene(strainer_at(yaw=L.ST_YAW + math.radians(deg)))), 0.5, deg)
        r = S.read(scene(strainer_at(yaw=L.ST_YAW + math.radians(180))))
        self.assertAlmostEqual(S.hook_reading(r)["side_deg"], 180.0, places=3)

    def test_pitch_and_roll_edges(self):
        self.assertEqual(score_of(scene(strainer_at(pitch=math.radians(3.0)))), 0.5)  # the handle end 5.5 mm low
        r = S.read(scene(strainer_at(roll=math.radians(10.0))))
        self.assertTrue(S.level(r))
        r = S.read(scene(strainer_at(roll=math.radians(16.0))))
        self.assertFalse(S.level(r))
        r = S.read(scene(strainer_at(pitch=math.radians(11.0), z=BEAD_Z + 0.03)))
        self.assertFalse(S.level(r))
        r = S.read(scene(strainer_at(pitch=math.radians(9.0), z=BEAD_Z + 0.03)))
        self.assertTrue(S.level(r))

    def test_upside_down_is_zero(self):
        """Turned over, the mesh is up in the air above the rim: neither clear of the rim nor in the stock."""
        self.assertEqual(score_of(scene(strainer_at(roll=math.pi))), 0.0)

    def test_on_the_table_beside_the_bowl_is_zero(self):
        self.assertEqual(score_of(scene(strainer_at(bowl=(0.0, 0.0), s=0.26, z=L.ST_Z))), 0.0)

    def test_on_the_pot_is_zero(self):
        st = mk.state_from_poses(T.STRAINER_PART, (L.POT_X, L.POT_Y, L.TOP + L.POT_H + L.RING_DEPTH,
                                                   *mk.quat_from_ypr(L.ST_YAW, 0, 0)), source="test")
        self.assertEqual(score_of(scene(st)), 0.0)

    def test_bowl_moved_with_the_strainer_still_hooked_is_judged_in_its_frame(self):
        st = strainer_at(bowl=(0.012, 0.0))
        self.assertEqual(score_of(scene(st, bowl_at(dx=0.012))), 1.0)  # moved 12 mm: allowed
        st = strainer_at(bowl=(0.0, -0.012))
        self.assertEqual(score_of(scene(st, bowl_at(dy=-0.012))), 1.0)


class Latches(unittest.TestCase):
    def test_bowl_ending_over_30mm_from_its_mark_is_zero_even_if_hooked(self):
        for d in ((0.031, 0.0), (0.0, 0.031), (-0.024, 0.024)):
            st = strainer_at(bowl=d)
            r = run(scene(st, bowl_at(dx=d[0], dy=d[1])))
            self.assertEqual(r.score, 0.0, d)
            self.assertFalse(r.details["bowl_ok"])

    def test_bowl_repositioned_within_30mm_is_not_penalised(self):
        for d in ((0.025, 0.0), (0.0, -0.025)):
            r = run(scene(strainer_at(bowl=d), bowl_at(dx=d[0], dy=d[1])))
            self.assertEqual(r.score, 1.0, d)
            self.assertTrue(r.details["bowl_ok"])

    def test_bowl_tipped(self):
        r = run(scene(strainer_at(), bowl_at(pitch=math.radians(11.0))))
        self.assertEqual(r.score, 0.0)
        r = run(scene(strainer_at(), bowl_at(pitch=math.radians(9.0))))
        self.assertEqual(r.score, 1.0)  # (a tipped bowl 9 degrees: the strainer is judged in its frame)

    def test_bowl_lifted(self):
        r = run(scene(strainer_at(), bowl_at(dz=0.03)))
        self.assertEqual(r.score, 0.0)

    def test_bowl_pushed_then_put_back_is_judged_on_its_final_place(self):
        tr = S.Tracker()
        run(scene(strainer_at(bowl=(0.06, 0.0)), bowl_at(dx=0.06)), steps=3, tracker=tr)
        res = run(scene(strainer_at(), bowl_at()), tracker=tr)  # the bowl put back
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.details["bowl_ok"])

    def test_bowl_knocked_off_the_table(self):
        r = run(scene(strainer_at(), bowl_at(dz=-0.05)))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(r.details["latch"]["F2"])

    def test_strainer_fell_off_the_table(self):
        st = strainer_at(z=L.TOP - 0.1)
        r = run(scene(st))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(r.details["latch"]["F2"])

    def test_a_drop_on_the_table_is_not_a_fall(self):
        r = run(start())
        self.assertIsNone(r.details["latch"]["F2"])
        self.assertTrue(r.details["bowl_ok"])


class HandsOffAndStillness(unittest.TestCase):
    def test_hand_on_the_handle(self):
        h = (L.BOWL_X, L.BOWL_Y + 0.15, BEAD_Z + 0.002 + T.PAD_TIP)
        arms = {"left": arm(h, 0.1), "right": FAR["right"]}
        self.assertEqual(score_of(scene(), arms), 0.0)

    def test_hand_on_the_bowl(self):
        h = (L.BOWL_X + 0.125, L.BOWL_Y, L.TOP + 0.08)  # a pad 15 mm from the bowl's outside
        arms = {"left": arm(h, 0.1), "right": FAR["right"]}
        self.assertEqual(score_of(scene(), arms), 0.0)

    def test_hand_holding_the_strainer_up_over_the_bowl(self):
        """Exploit: hold the strainer in the air above the hooked pose: hands on, so 0."""
        st = strainer_at(dz=0.0)
        h = (L.BOWL_X, L.BOWL_Y + L.GRIP_X, BEAD_Z + 0.015)
        self.assertEqual(score_of(scene(st), {"left": arm(h, 0.1), "right": FAR["right"]}), 0.0)

    def test_hand_clear_just_enough(self):
        far = 0.20
        arms = {"left": arm((L.BOWL_X, L.BOWL_Y + 0.19 + far, L.TOP + 0.15)), "right": FAR["right"]}
        self.assertEqual(score_of(scene(), arms), 1.0)

    def test_parked_hands_are_clear(self):
        self.assertEqual(score_of(scene(), {"left": arm((0.30, 0.385, L.TOP + 0.15)), "right": arm((0.05, -0.30, L.TOP + 0.1))}), 1.0)

    def test_unreadable_arms_never_release(self):
        for arms in (None, {}, {"left": FAR["left"]}):
            self.assertEqual(run(scene(), arms).score, 0.0)

    def test_nan_arm(self):
        bad = {"left": rigstate.Arm(np.array([np.nan, 0.0, 0.0]), np.zeros(3), 1.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), bad).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        held = {"left": arm((L.BOWL_X, L.BOWL_Y + 0.15, BEAD_Z + 0.01), 0.1), "right": FAR["right"]}
        run(scene(), held, steps=WIN, tracker=tr)
        for k in range(WIN - 1):
            res = tr.update(scene(), FAR)
            self.assertEqual(res.score, 0.0, k)
        self.assertEqual(tr.update(scene(), FAR).score, 1.0)

    def test_moving_strainer_is_not_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(s=0.0 + 0.0005 * k)), FAR)  # 0.5 mm a step: about 10 mm a second
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_slow_creep_under_5mm_is_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(s=0.0 + 0.0002 * k)), FAR)  # 4 mm over the window
        self.assertEqual(res.score, 1.0)

    def test_slow_turn_under_5_degrees_is_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(yaw=L.ST_YAW + math.radians(0.15 * k))), FAR)  # 3 degrees over the window
        self.assertEqual(res.score, 1.0)

    def test_slow_bowl_creep_under_5mm_is_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(bowl=(0.0002 * k, 0.0)), bowl_at(dx=0.0002 * k)), FAR)  # 4 mm over the window
        self.assertEqual(res.score, 1.0)

    def test_still_limits_are_the_stated_ones(self):
        self.assertAlmostEqual(L.STILL_MOVE, 0.005)
        self.assertAlmostEqual(L.STILL_TURN_DEG, 5.0)

    def test_turning_strainer_is_not_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(yaw=L.ST_YAW + math.radians(0.5 * k))), FAR)  # 10 degrees in 20 steps
        self.assertEqual(res.score, 0.0)

    def test_moving_bowl_is_not_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            res = tr.update(scene(strainer_at(bowl=(0.0005 * k, 0.0)), bowl_at(dx=0.0005 * k)), FAR)  # 10 mm in 1 s
        self.assertEqual(res.score, 0.0)

    def test_success_needs_a_full_second(self):
        tr = S.Tracker()
        for k in range(WIN):
            self.assertFalse(tr.update(scene(), FAR).success, k)
        self.assertTrue(tr.update(scene(), FAR).success)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        self.assertEqual(run({S.STRAINER: strainer_at()}).score, 0.0)
        self.assertEqual(run({}).score, 0.0)
        self.assertEqual(run(None).score, 0.0)

    def test_nan_state(self):
        st = strainer_at()
        bad = mk.AssemblyState(**{**st.__dict__, "origin": np.array([np.nan, 0.0, 0.0])}) if False else None
        o = np.array([L.BOWL_X, np.nan, BEAD_Z])
        bad = mk.state_from_poses(T.STRAINER_PART, (*o, 0.0, 0.0, 0.7071, 0.7071), source="test")
        tr = S.Tracker()
        res = run(scene(bad), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertGreater(tr.bad_steps, 0)
        # and it recovers once the numbers are finite, after a fresh second
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_inf_quaternion(self):
        bad = mk.state_from_poses(T.STRAINER_PART, (0.3, 0.0, BEAD_Z, 0.0, 0.0, 0.0, 1.0), source="test")
        object.__setattr__(bad, "quat_xyzw", np.array([np.inf, 0.0, 0.0, 1.0]))
        self.assertEqual(run(scene(bad)).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(scene(strainer_at(z=L.TOP - 0.1), bowl_at()), steps=3, tracker=tr)
        self.assertTrue(tr.latched)
        run(scene(strainer_at(), bowl_at()), steps=2, tracker=tr)
        res = tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = random.Random(7)
        tr = S.Tracker()
        for _ in range(300):
            q = np.array([rng.gauss(0, 1) for _ in range(4)])
            q /= np.linalg.norm(q)
            st = mk.state_from_poses(T.STRAINER_PART, (rng.uniform(0, 0.6), rng.uniform(-0.5, 0.5),
                                                       L.TOP + rng.uniform(-0.05, 0.3), *q), source="test")
            bw = mk.state_from_poses(T.BOWL_PART, (L.BOWL_X + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02),
                                                   L.BOWL_Z0, 0.0, 0.0, 0.0, 1.0), source="test")
            res = tr.update({S.STRAINER: st, S.BOWL: bw}, FAR)
            self.assertIn(res.score, (0.0, 0.5, 1.0))

    def test_exploit_drop_it_in_the_bowl(self):
        """Letting the strainer fall into the stock earns nothing."""
        z = L.BOWL_Z0 + L.STOCK_TOP_B + L.RING_DEPTH
        self.assertEqual(score_of(scene(strainer_at(z=z, s=0.01, dx=0.01, pitch=math.radians(3)))), 0.0)

    def test_exploit_shove_the_bowl_under_the_strainer(self):
        """Dragging the bowl 40 mm to meet a strainer laid on the table leaves the bowl out of place: 0."""
        st = strainer_at(bowl=(0.0, 0.04))
        self.assertEqual(score_of(scene(st, bowl_at(dy=0.04))), 0.0)

    def test_exploit_lay_it_across_without_the_hook(self):
        """Rested across the rims, hook end 12 mm up on its inner leg: 0.5, never 1.0."""
        self.assertEqual(score_of(scene(strainer_at(s=-0.02, dz=0.006, pitch=math.radians(3.3)))), 0.5)

    def test_exploit_hold_it_and_walk_away_a_little(self):
        """A gripper hovering 40 mm over the handle (inside the 56.6 mm open zone) is still hands on."""
        h = (L.BOWL_X, L.BOWL_Y + L.GRIP_X, BEAD_Z + 0.012 + 0.04 + 0.013)
        self.assertEqual(score_of(scene(), {"left": arm(h, 1.0), "right": FAR["right"]}), 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["mat", "wall", "pot", "bowl", "strainer"])
        self.assertEqual([o.mass is None for o in T.OBJECTS], [True, True, True, False, False])
        for o in T.OBJECTS:
            self.assertTrue(o.name.isidentifier())

    def test_instruction_states_the_goal_only(self):
        t = T.INSTRUCTION
        self.assertLessEqual(len(t), 490)
        for w in ("hook", "handle", "mesh", "let go"):
            self.assertIn(w, t)
        self.assertEqual(t, json.loads((HERE / "task.json").read_text())["instruction"])

    def test_demo_plan(self):
        plan = T.demo()
        self.assertEqual(list(plan), ["left"])
        steps = T._duration(plan["left"]) * L.CONTROL_HZ
        self.assertLessEqual(steps, 1200 / 2)  # within half the run limit
        for w in plan["left"]:
            self.assertGreaterEqual(w.duration, 0.5)

    def test_grip_and_release_take_1_5_s(self):
        for w in T.demo()["left"]:
            if w.label in ("grip the handle across its width", "let go"):
                self.assertGreaterEqual(w.duration, 1.5)

    def test_demo_env_switches(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(T.demo(), {})  # the left arm does the work: idle, nothing moves
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(list(T.demo()), ["left"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_variants_move_the_set_down_point_only(self):
        base = T.plans()["left"]
        for v in T.VARIANTS:
            plan = T.plans(v)["left"]
            self.assertEqual(len(plan), len(base))
            self.assertEqual(plan[:7], base[:7])  # the same pick-up
            self.assertNotEqual(plan[-3], base[-3])

    def test_set_down_points(self):
        plan = T.plans()["left"]
        down = [w for w in plan if w.label.startswith("slowly down: the hook")][0]
        x, y, z = down.pos
        self.assertAlmostEqual(y, L.BOWL_Y + L.HOOK_AIM_S + L.GRIP_X, places=6)
        self.assertAlmostEqual(x, L.BOWL_X, places=6)
        self.assertAlmostEqual(z - L.TOP - 0.013, T.SET_LIP, places=6)  # the rest plane, half a mm under its rest height
        self.assertAlmostEqual(T.SET_LIP, L.BEAD_TOP_W - L.TOP - 0.0005, places=6)

    def test_states_at_rest_match_the_spawn(self):
        for n in S.PARTS:
            s = T.SPECS[n].rest_state()
            self.assertTrue(np.all(np.isfinite(s.origin)))
        r = S.read(start())
        self.assertTrue(S.Tracker.at_start(r))
        self.assertFalse(S.hook_on(S.hook_reading(r)))

    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).exists():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        j = json.loads((HERE / "task.json").read_text())
        setup, scoring = j["setup"], j["scoring"]
        for fragment in ("220 mm across its rolled rim bead", "105 mm tall", "0.30 kg", "x 300 mm, y 0", "y +252 mm",
                         "130 mm across and 57 mm tall", "192.5 mm", "30 mm apart", "121.5 mm", "314 mm long",
                         "58 mm above the bowl's foot", "26 mm above the stock", "0.12 kg", "0.5 L", "30 mm deep",
                         "260 x 260 x 4 mm", "5 in (13 cm)", "125-135 mm", "x 420 mm, y -300 mm", "x 565 mm", "190 mm across, 200 mm tall"):
            self.assertIn(fragment, setup)
        for fragment in ("100 mm or less", "110 mm or more", "within 6 mm above", "no more than 2 mm", "within 5 mm",
                         "within 30 mm of the bowl's axis", "15 mm or more above the stock", "47 mm or more",
                         "at least 20 mm below the rim's top", "within 10 degrees", "within 15 degrees",
                         "within 20 mm of the bowl's axis", "30 mm or more", "under 5 mm", "under 5 degrees",
                         "within 30 mm of its mark", "only where it ends counts", "10 degrees", "20 mm below the table top",
                         "within 25 degrees", "within 45 degrees of straight right", "Possible scores: 0, 0.5, 1.0",
                         "held for 1 s"):
            self.assertIn(fragment, scoring)
        self.assertEqual(j["run_limit"], "1200 steps (60 s at 20 Hz)")
        # thresholds in the text are the layout's
        self.assertAlmostEqual(L.R_RIM_IN, 0.100)
        self.assertAlmostEqual(L.BEAD_OUT, 0.110)
        self.assertAlmostEqual(L.HOOK_GAP_UP, 0.006)
        self.assertAlmostEqual(L.HOOK_GAP_DOWN, 0.002)
        self.assertAlmostEqual(L.HANDLE_OVER, 0.005)
        self.assertAlmostEqual(L.RING_CENTRE_MAX, 0.030)
        self.assertAlmostEqual(L.STOCK_TOP_B + L.MESH_ABOVE_STOCK, 0.047)
        self.assertAlmostEqual(L.BEAD_TOP_B - L.MESH_BELOW_RIM, 0.085)
        self.assertAlmostEqual(L.AXIS_THROUGH, 0.020)
        self.assertAlmostEqual(L.BOWL_MOVED, 0.030)
        self.assertEqual(L.LEVEL_DEG, 10.0)
        self.assertEqual(L.RESTS_TILT_DEG, 25.0)
        self.assertAlmostEqual(L.IN_BOWL_CENTRE, 0.035)
        self.assertEqual(L.RIGHT_RIM_DEG, 45.0)

    def test_scores_field_holds_the_gpu_runs(self):
        j = json.loads((HERE / "task.json").read_text())
        self.assertIn("demo: 1.0", j["scores"])
        self.assertIn("Dream-1 run:", j["scores"])

    def test_fidelity_states_the_rules_and_estimates(self):
        f = json.loads((HERE / "task.json").read_text())["fidelity"]
        for w in ("Rigid bodies", "solid", "estimates", "friction", "fixed"):
            self.assertIn(w.lower(), f.lower())

    def test_context_names_the_work(self):
        c = json.loads((HERE / "task.json").read_text())["context"]
        for w in ("home-kitchen", "hook", "mesh", "clogs"):
            self.assertIn(w, c)

    def test_field_limits(self):
        j = json.loads((HERE / "task.json").read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(j[k]), n, k)

    def test_names_nobody(self):
        text = " ".join(str(v) for v in json.loads((HERE / "task.json").read_text()).values())
        text += (HERE / "README.md").read_text() if (HERE / "README.md").exists() else ""
        for bad in ("T-70", "Aggarwal", "Vihan", "Claude", "Anthropic", "Co-Authored", "/home/", "robocurve"):
            self.assertNotIn(bad.lower(), text.lower().replace("robocurve benchmarking program", ""), bad)

    def test_text_rules(self):
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py"):
            p = HERE / f
            if p.exists():
                t = p.read_text()
                self.assertNotIn("—", t, f)  # no em-dashes
                self.assertNotIn("–", t, f)

    def test_readme_says_offline(self):
        p = HERE / "README.md"
        if p.exists():
            self.assertIn("offline prediction", p.read_text().lower())


class OfflineModel(unittest.TestCase):
    """The stand-in the offline checker uses (never in the simulator)."""

    def _drive(self, variant="", idle=None):
        env = dict(os.environ, DEMO_VARIANT=variant)
        env.pop("DEMO_IDLE_ARM", None)
        if idle:
            env["DEMO_IDLE_ARM"] = idle
        out = subprocess.run([sys.executable, str(OFFLINE / "check_task.py"), str(HERE)], capture_output=True, text=True,
                             env=env, timeout=300).stdout
        m = re.search(r"predicted task_score ([0-9.]+)", out)
        self.assertIsNotNone(m, out[-800:])
        return float(m.group(1))

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_demo_predicts_one(self):
        self.assertEqual(self._drive(), 1.0)

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_variants_predict_half(self):
        for v in T.VARIANTS:
            self.assertEqual(self._drive(v), 0.5, v)

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_idle_left_scores_zero(self):
        self.assertEqual(self._drive(idle="left"), 0.0)

    @unittest.skipIf(OFFLINE is None, "tools/offline not found")
    def test_idle_right_still_scores_one(self):
        self.assertEqual(self._drive(idle="right"), 1.0)

    def test_settle_rules(self):
        m = T.OfflineModel()
        m.origin = np.array([L.BOWL_X, L.BOWL_Y + 0.002, BEAD_Z + 0.02])
        m.grip = np.zeros(3)
        m._settle()
        self.assertAlmostEqual(m.origin[2], BEAD_Z)
        m.origin = np.array([L.BOWL_X, L.BOWL_Y - 0.02, BEAD_Z + 0.02])
        m._settle()
        self.assertGreater(m.pitch, 0.0)
        m.origin = np.array([L.BOWL_X + 0.05, L.BOWL_Y, BEAD_Z + 0.02])
        m._settle()
        self.assertAlmostEqual(m.origin[2], L.BOWL_Z0 + L.STOCK_TOP_B + L.RING_DEPTH)
        m.origin = np.array([0.30, 0.30, 0.9])
        m._settle()
        self.assertAlmostEqual(m.origin[2], L.ST_Z)


if __name__ == "__main__":
    unittest.main()
