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


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    return mk.quat_to_matrix(mk.quat_from_ypr(yaw, pitch, roll))


def pose_for_seat(R, y_c=L.R_Y, qx=L.QX_MID, gap=0.0):
    """The caddy's origin (table frame) for a caddy turned ``R`` whose +y hook centre is at world y ``y_c`` with the
    rail's axis, in the caddy's frame at that hook, at x ``qx`` and the bridge ``gap`` m over the rail's top."""
    dy = float((R @ np.array([0.0, L.HK_Y, 0.0]))[1])
    qz = L.U - L.RAIL_R - gap
    qy = (dy - R[1, 0] * qx - R[1, 2] * qz) / R[1, 1]
    v = R @ np.array([qx, qy, qz])
    return np.array([L.R_X - v[0], y_c - dy, L.RAIL_Z - v[2]])


def caddy_state(origin, R=None):
    R = np.eye(3) if R is None else R
    q = mk.matrix_to_quat(R)
    return mk.state_from_poses(T.CADDY_PART, (origin[0], origin[1], L.TOP + origin[2], *q), source="test")


def fork_state(i, origin, R=None, local=None):
    R = np.eye(3) if R is None else R
    loc = np.array([0.0, L.POCKET_Y[i], L.FORK_Z0]) if local is None else np.asarray(local, dtype=float)
    o = np.asarray(origin, dtype=float) + R @ loc
    q = mk.matrix_to_quat(R)
    return mk.state_from_poses(T.SPECS[L.FORK_NAMES[i]], (o[0], o[1], L.TOP + o[2], *q), source="test")


def scene(origin=None, R=None, forks=(None, None, None)):
    """The caddy at ``origin`` turned ``R`` with its forks in their slots (or at the local offset given)."""
    origin = np.array([L.HUNG_X, L.HUNG_Y, L.HUNG_Z]) if origin is None else np.asarray(origin, dtype=float)
    out = {S.CADDY: caddy_state(origin, R)}
    for i, n in enumerate(L.FORK_NAMES):
        out[n] = fork_state(i, origin, R, forks[i])
    return out


def hung(**kw):
    """Hung as the rest pose lands: both hooks seated, square."""
    return scene(**kw)


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def score_of(states, arms=FAR):
    return run(states, arms).score


class Geometry(unittest.TestCase):
    def test_design_sizes(self):
        self.assertAlmostEqual(L.RAIL_R * 2, 0.010)
        self.assertAlmostEqual(L.HK_GAP, 0.018)
        self.assertAlmostEqual(L.POCKET, 0.022)
        self.assertAlmostEqual(L.INNER_D - L.FORK_T, 0.004)  # 2 mm each side of a fork
        self.assertAlmostEqual((L.POCKET - L.FORK_HEAD_W) / 2, 0.003)
        self.assertAlmostEqual(L.HUNG_Z, 0.014)
        self.assertAlmostEqual(L.U, 0.126)
        self.assertAlmostEqual(L.FORK_TOP_ABOVE_RIM, 0.023)
        self.assertAlmostEqual(L.CADDY_MASS, 0.062, places=3)
        self.assertAlmostEqual(L.X_PLANE, 0.325)

    def test_grip_width_and_clearance(self):
        self.assertLessEqual(L.C_W, 0.080)  # the gripper's 16 to 80 mm
        self.assertAlmostEqual((0.09395 - L.C_W) / 2, 0.009, places=3)  # 9 mm each side at full open
        self.assertAlmostEqual(-2 * T.GRIP_F, L.C_W - 0.008)

    def test_forks_clear_the_dividers_and_walls(self):
        # the middle slot: a 16 mm head in 22 mm between two dividers; the outer slots: 22 mm between wall and divider
        self.assertAlmostEqual(L.POCKET_Y[2] - L.POCKET / 2, L.POCKET / 2 + L.DIV_T, places=6)  # starts 13 mm out
        self.assertAlmostEqual(L.C_W / 2 - L.C_T - (L.POCKET_Y[2] + L.POCKET / 2), 0.0, places=6)  # wall face
        for y in L.POCKET_Y:
            self.assertAlmostEqual((L.POCKET - L.FORK_HEAD_W) / 2, 0.003)
            self.assertLessEqual(abs(y) + L.FORK_HEAD_W / 2, L.C_W / 2 - L.C_T)
        self.assertAlmostEqual((L.INNER_D - L.FORK_T) / 2, 0.002)
        self.assertLess(L.FORK_Z0 + L.FORK_L / 2 - L.C_H, 0.0235)

    def test_forks_clear_the_hooks(self):
        # a fork's +x face (4 mm) is 2 mm from the tab's -x face (6 mm); its top is below the bridge's underside
        self.assertLess(L.FORK_T / 2, L.C_D / 2 - L.C_T)
        self.assertLess(L.FORK_Z0 + L.FORK_L / 2, L.U)

    def test_parts_spawn_where_they_rest(self):
        st = start()
        r = S.read(st)
        self.assertAlmostEqual(float(r.pos[S.CADDY][2]), 0.0, places=9)
        self.assertAlmostEqual(T.CADDY_PART.rest_state().low[2] - L.TOP, 0.0, places=6)
        for i, n in enumerate(L.FORK_NAMES):
            low = T.SPECS[n].rest_state().low[2] - L.TOP
            self.assertAlmostEqual(low, L.C_BT, places=6, msg=n)  # tines on the caddy's floor
        self.assertTrue(all(S.forks_in(r)))
        for s in (-1, 1):
            self.assertFalse(S.hook_reading(r, s)["seated"])

    def test_hung_pose_matches_the_rest_pose(self):
        r = S.read(hung())
        for s in (-1, 1):
            h = S.hook_reading(r, s)
            self.assertTrue(h["seated"], h)
            self.assertAlmostEqual(h["qx"], L.QX_HUNG, places=9)
            self.assertAlmostEqual(h["bridge_gap"], 0.0, places=9)
            self.assertAlmostEqual(h["clear_near"], 0.0, places=9)  # the near leg's face on the wall's plane
            self.assertAlmostEqual(h["clear_rear"], L.HK_GAP - 2 * L.RAIL_R, places=9)
        self.assertAlmostEqual(S.base_height(r), 0.014, places=9)
        self.assertTrue(all(S.forks_in(r)))

    def test_the_window(self):
        self.assertAlmostEqual(L.QX_MAX - L.QX_MIN, 0.008)
        self.assertAlmostEqual(T.DEMO_X, L.R_X - L.QX_MID, places=9)  # 4 mm clear each side
        self.assertTrue(L.QX_MIN < L.R_X - T.DEMO_X < L.QX_MAX)

    def test_lead_in_and_rail_clearances(self):
        self.assertGreater(L.FLARE_DX, 0.008)
        flare_tip_z = L.U - L.HK_LEG_H - L.FLARE_DZ
        carry = L.HUNG_Z + (T.CARRY_Z - L.HUNG_Z) + flare_tip_z  # the lowest hook point when carried over the rail
        self.assertGreater(carry - L.RAIL_TOP, 0.010)  # at least 10 mm over the rail
        self.assertGreater(T.HOVER, T.GRIP_DZ + L.START_Z)

    def test_gripper_housing_clears_the_forks(self):
        """The housing's lower face is 70 mm over the TCP (research/sim_capabilities.md); the forks stand 23 mm over the
        rim and the pads grip 15 mm under it."""
        tcp = L.C_H - T.GRIP_TIPS_DOWN + T.PAD_TIP
        fork_top = L.C_BT + L.FORK_L
        self.assertGreater(tcp + 0.070 - fork_top, 0.020)

    def test_rack_geometry(self):
        self.assertAlmostEqual(L.RAIL_TOP, 0.140)
        self.assertAlmostEqual(L.X_PLANE, L.WIRE_X - L.WIRE_R)
        self.assertGreaterEqual(L.TRAY_X0, L.R_X + L.RAIL_R)
        # the caddy between the end posts, hung anywhere the rail's ends allow
        self.assertLess(L.C_W / 2 + L.HK_Y, L.POST_DY + 0.1)
        # the plates stand behind the gripper's housing (65 mm across x) when hung
        self.assertGreater(L.PLATE_X - L.PLATE_R, T.GRIP_GX + 0.0 + 0.032 + (L.HUNG_X - L.START_X) + 0.005)

    def test_demo_inside_the_reach_rows(self):
        for w in T.plans("")["right"]:
            x, y, z = w.pos[0], w.pos[1], w.pos[2] - L.TOP
            self.assertGreaterEqual(x, 0.10)
            self.assertLessEqual(x, 0.50)
            self.assertLessEqual(abs(y + 0.305), 0.45)  # nearer the right arm than the midline plus 15 cm
            self.assertLessEqual(z, 0.200 + 1e-9)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = run(hung())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_tier_table(self):
        t = S.tier
        self.assertEqual(t(False, True, 2, True, True), 1.0)
        self.assertEqual(t(False, True, 2, True, False), 0.5)
        self.assertEqual(t(False, True, 1, True, True), 0.25)
        self.assertEqual(t(False, True, 2, False, True), 0.25)
        self.assertEqual(t(False, True, 0, True, True), 0.0)
        self.assertEqual(t(True, True, 2, True, True), 0.0)
        self.assertEqual(t(False, False, 2, True, True), 0.0)

    def test_start_scores_zero(self):
        res = run(start())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_a_fork_out(self):
        for i in range(3):
            forks = [None, None, None]
            forks[i] = (0.0, L.POCKET_Y[i], L.C_H + 0.010)  # standing up on the rim: its middle over the rim
            self.assertEqual(score_of(hung(forks=forks)), 0.5, i)
        for i in range(3):
            forks = [None, None, None]
            forks[i] = (0.0, L.POCKET_Y[i] + 0.080, L.FORK_Z0)  # beside the caddy, in the air
            self.assertEqual(score_of(hung(forks=forks)), 0.5, i)

    def test_all_forks_out_is_half(self):
        forks = [(0.2, 0.0, 0.0)] * 3  # on the table in front of the hung caddy
        self.assertEqual(score_of(hung(forks=forks)), 0.5)

    def test_fork_below_the_floor_is_out(self):
        forks = [(0.0, 0.0, L.C_BT - 0.010), None, None]
        self.assertEqual(score_of(hung(forks=forks)), 0.5)

    def test_fork_leaning_but_inside_counts(self):
        # a fork leaning 4 degrees against a wall: its middle moves 5 mm sideways: still inside
        forks = [(0.002, L.POCKET_Y[0], L.FORK_Z0), None, None]
        self.assertEqual(score_of(hung(forks=forks)), 1.0)

    def test_one_hook_only(self):
        # the caddy slid along the rail until its +y hook is past the rail's end: one hook seated
        o = pose_for_seat(np.eye(3), y_c=L.R_Y + 0.124)
        res = run(scene(origin=o))
        self.assertEqual(res.details["n_seated"], 1)
        self.assertEqual(res.details["seated"], {"-1": True, "1": False})
        self.assertEqual(res.score, 0.25)

    def test_both_hooks_near_the_rail_end(self):
        """The +y hook's centre 117 mm from the rail's middle (limit 118): both seated; 119 mm: one seated."""
        res = run(scene(origin=pose_for_seat(np.eye(3), y_c=L.R_Y + 0.117)))
        self.assertEqual((res.details["n_seated"], res.score), (2, 1.0))
        res = run(scene(origin=pose_for_seat(np.eye(3), y_c=L.R_Y + 0.119)))
        self.assertEqual((res.details["n_seated"], res.score), (1, 0.25))

    def test_crooked_in_yaw(self):
        R = rot(yaw=math.radians(8.0))
        res = run(scene(origin=pose_for_seat(R, qx=L.QX_MID + 0.00335), R=R))  # 48 mm span: 3.3 mm each
        self.assertEqual(res.details["n_seated"], 2)  # both hooks still straddle the rail
        self.assertFalse(res.details["square"])
        self.assertEqual(res.score, 0.25)

    def test_yaw_edge(self):
        R = rot(yaw=math.radians(4.5))
        self.assertEqual(score_of(scene(origin=pose_for_seat(R), R=R)), 1.0)
        R = rot(yaw=math.radians(5.5))
        self.assertEqual(score_of(scene(origin=pose_for_seat(R), R=R)), 0.25)

    def test_tilt_about_the_rail(self):
        R = rot(pitch=math.radians(4.0))
        res = run(scene(origin=pose_for_seat(R), R=R))
        self.assertEqual(res.details["n_seated"], 2)
        self.assertEqual(res.score, 1.0)
        R = rot(pitch=math.radians(7.0))
        res = run(scene(origin=pose_for_seat(R), R=R))
        self.assertEqual(res.details["n_seated"], 2)
        self.assertEqual(res.score, 0.25)

    def test_tilt_sideways(self):
        R = rot(roll=math.radians(7.0))
        res = run(scene(origin=pose_for_seat(R), R=R))
        self.assertLessEqual(res.score, 0.25)

    def test_swing_to_the_wires_is_square(self):
        """Released 4 mm from the near leg's contact, the caddy swings until its back wall leans on the wires: tilt
        about 2.5 degrees about the rail, both hooks still seated."""
        R = rot(pitch=math.radians(2.5))
        res = run(scene(origin=pose_for_seat(R, qx=L.QX_MID + 0.002), R=R))
        self.assertEqual(res.score, 1.0)

    def test_caddy_on_the_table_scores_zero(self):
        for x in (0.2, 0.30, 0.316):
            self.assertEqual(score_of(scene(origin=[x, L.R_Y, 0.0])), 0.0, x)

    def test_leg_perched_on_the_rail_scores_zero(self):
        """Lowered 14 mm too far toward the robot: the rear legs' ends stand on the rail's top, the caddy 30 mm high."""
        z = L.RAIL_TOP - (L.U - L.HK_LEG_H)
        o = [L.R_X - (L.REAR_INNER + L.HK_LEG_T / 2), L.R_Y, z]
        res = run(scene(origin=o))
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["n_seated"], 0)

    def test_tab_perched_on_the_rail_scores_zero(self):
        """The back wall's top edge on the rail (the caddy too far toward the rack, the hooks not over it)."""
        o = [L.R_X - (L.C_D / 2 - L.C_T / 2), L.R_Y, L.RAIL_TOP - L.C_H]
        self.assertEqual(score_of(scene(origin=o)), 0.0)

    def test_hovering_over_the_rail_scores_zero(self):
        o = pose_for_seat(np.eye(3), gap=0.006)  # the bridges 6 mm over the rail's top: held up by nothing real
        self.assertEqual(score_of(scene(origin=o)), 0.0)

    def test_sunk_through_the_rail_scores_zero(self):
        o = pose_for_seat(np.eye(3), gap=-0.006)
        self.assertEqual(score_of(scene(origin=o)), 0.0)

    def test_bridge_gap_edges(self):
        for gap, want in ((0.0029, 1.0), (0.0031, 0.0), (-0.0019, 1.0), (-0.0021, 0.0)):
            self.assertEqual(score_of(scene(origin=pose_for_seat(np.eye(3), gap=gap))), want, gap)

    def test_rail_beside_the_legs_scores_zero(self):
        for qx in (L.QX_MIN - 0.004, L.QX_MAX + 0.004):
            self.assertEqual(score_of(scene(origin=pose_for_seat(np.eye(3), qx=qx))), 0.0, qx)

    def test_rail_in_the_gap_edges(self):
        for qx, want in ((L.QX_MIN - 0.0009, 1.0), (L.QX_MAX + 0.0009, 1.0)):
            self.assertEqual(score_of(scene(origin=pose_for_seat(np.eye(3), qx=qx))), want, qx)

    def test_hung_from_the_far_side_scores_zero(self):
        R = rot(yaw=math.pi)
        o = pose_for_seat(R)
        self.assertEqual(score_of(scene(origin=o, R=R)), 0.0)

    def test_upside_down_scores_zero(self):
        R = rot(roll=math.pi)
        self.assertEqual(score_of(scene(origin=[L.HUNG_X, L.R_Y, 0.2], R=R)), 0.0)

    def test_slid_along_the_rail_is_fine(self):
        for dy in (-0.08, 0.0, 0.08):
            self.assertEqual(score_of(scene(origin=[L.HUNG_X, L.R_Y + dy, L.HUNG_Z])), 1.0, dy)

    def test_hung_past_the_rail_end_scores_zero(self):
        o = [L.HUNG_X, L.R_Y + 0.20, L.HUNG_Z]
        self.assertEqual(score_of(scene(origin=o)), 0.0)

    def test_hung_near_the_rail_end_edges(self):
        # the hook centres are 24 mm either side of the middle; the limit is 118 mm for a hook centre
        ok = [L.HUNG_X, L.R_Y + 0.118 - L.HK_Y - 0.001, L.HUNG_Z]
        no = [L.HUNG_X, L.R_Y + 0.118 - L.HK_Y + 0.004, L.HUNG_Z]
        self.assertEqual(score_of(scene(origin=ok)), 1.0)
        self.assertEqual(score_of(scene(origin=no)), 0.25)


class Exploits(unittest.TestCase):
    def test_empty_caddy_hung_is_half(self):
        """Hang the caddy without its cutlery (the forks put aside on the table): at most 0.5."""
        forks = [(0.3, -0.05 * i, 0.0) for i in range(3)]
        self.assertEqual(score_of(hung(forks=forks)), 0.5)

    def test_caddy_held_on_the_rail_scores_zero(self):
        """Hung but a gripper still closed on its side walls: a held caddy earns nothing."""
        tcp = (L.HUNG_X + L.GRIP_X, L.R_Y, L.TOP + T.GRIP_DZ + L.HUNG_Z)
        arms = {"left": FAR["left"], "right": arm(tcp, 0.72)}
        self.assertEqual(run(hung(), arms).score, 0.0)

    def test_hand_resting_on_the_hung_caddy_scores_zero(self):
        tcp = (L.HUNG_X, L.R_Y, L.TOP + 0.20)  # open hand over the caddy, fingers within 30 mm of the forks
        arms = {"left": FAR["left"], "right": arm(tcp, 1.0)}
        self.assertEqual(run(hung(), arms).score, 0.0)

    def test_hand_pressing_the_bridge_scores_zero(self):
        tcp = (L.HUNG_X, L.R_Y + 0.05, L.TOP + L.HUNG_Z + L.U + 0.02)
        arms = {"left": FAR["left"], "right": arm(tcp, 0.0)}
        self.assertEqual(run(hung(), arms).score, 0.0)

    def test_caddy_wedged_between_rail_and_plates_scores_zero(self):
        """Pushed past the rail onto the tray: no hook seated."""
        self.assertEqual(score_of(scene(origin=[0.40, L.R_Y, L.TRAY_H])), 0.0)

    def test_forks_swapped_for_fakes_cannot_hide_a_missing_fork(self):
        """A fork standing beside the caddy (its middle just outside the footprint) is out."""
        forks = [(L.C_D / 2 + 0.003, 0.0, L.FORK_Z0), None, None]
        self.assertEqual(score_of(hung(forks=forks)), 0.5)

    def test_thrown_onto_the_rail_counts(self):
        """The task scores the end state: a caddy that comes to rest hung, hands off, is hung (however it got
        there)."""
        self.assertEqual(score_of(hung()), 1.0)

    def test_both_hands_cannot_prop_the_caddy(self):
        arms = {"left": arm((L.HUNG_X, L.R_Y + 0.07, L.TOP + 0.08), 0.0),
                "right": arm((L.HUNG_X, L.R_Y - 0.07, L.TOP + 0.08), 0.0)}
        self.assertEqual(run(hung(), arms).score, 0.0)


class Latches(unittest.TestCase):
    def test_caddy_off_the_table(self):
        tr = S.Tracker()
        tr.update(scene(origin=[0.5, 0.7, -0.08]), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(hung(), tracker=tr).score, 0.0)

    def test_a_fork_off_the_table(self):
        tr = S.Tracker()
        forks = [(0.0, 0.0, -0.20), None, None]
        tr.update(hung(forks=forks), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(hung(), tracker=tr).score, 0.0)

    def test_a_fork_at_table_level_is_no_fall(self):
        tr = S.Tracker()
        tr.update(scene(forks=[(0.3, 0.0, 0.0), None, None]), FAR)
        self.assertFalse(tr.latched)

    def test_caddy_dropped_on_the_table_is_no_fall(self):
        tr = S.Tracker()
        tr.update(scene(origin=[0.25, L.R_Y, 0.0]), FAR)
        self.assertFalse(tr.latched)

    def test_fall_survives_a_later_hang(self):
        tr = S.Tracker()
        tr.update(scene(origin=[0.5, 0.7, -0.08]), FAR)
        res = run(hung(), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.latch["F1"], 1)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_caddy(self):
        tcp = (L.HUNG_X, L.R_Y, L.TOP + L.HUNG_Z + 0.05)
        arms = {"left": FAR["left"], "right": arm(tcp, 0.7)}
        self.assertEqual(run(hung(), arms).score, 0.0)

    def test_hand_clear(self):
        self.assertEqual(run(hung(), FAR).score, 1.0)

    def test_parked_hands_are_clear(self):
        right = arm(T.PARK[:2] + (L.TOP + T.PARK[2],), 1.0)
        self.assertEqual(run(hung(), {"left": FAR["left"], "right": right}).score, 1.0)

    def test_unreadable_arms_never_release(self):
        for arms in (None, {"left": FAR["left"]}, {"left": FAR["left"], "right": arm((math.nan, 0, 0))}):
            self.assertEqual(run(hung(), arms).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        close = {"left": FAR["left"], "right": arm((L.HUNG_X, L.R_Y, L.TOP + L.HUNG_Z + 0.05), 0.7)}
        for _ in range(5):
            tr.update(hung(), close)
        for _ in range(WIN - 1):
            res = tr.update(hung(), FAR)
        self.assertEqual(res.score, 0.0)
        res = tr.update(hung(), FAR)
        self.assertEqual(res.score, 1.0)

    def test_moving_caddy(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            o = [L.HUNG_X + 0.0005 * k, L.R_Y, L.HUNG_Z]  # 0.5 mm a step: 10 mm a second
            res = tr.update(scene(origin=o), FAR)
        self.assertEqual(res.score, 0.0)

    def test_slow_drift_under_five_millimetres_is_still(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            o = [L.HUNG_X, L.R_Y + 0.0002 * k, L.HUNG_Z]  # 4 mm a second
            res = tr.update(scene(origin=o), FAR)
        self.assertEqual(res.score, 1.0)

    def test_turning_caddy(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            R = rot(yaw=math.radians(0.3 * k))  # 6 degrees in the window
            res = tr.update(scene(origin=pose_for_seat(R), R=R), FAR)
        self.assertEqual(res.score, 0.0)

    def test_forks_rattling_do_not_matter(self):
        tr = S.Tracker()
        res = None
        for k in range(WIN + 3):
            f = [(0.001 * (k % 2), L.POCKET_Y[0], L.FORK_Z0), None, None]
            res = tr.update(hung(forks=f), FAR)
        self.assertEqual(res.score, 1.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        st = hung()
        del st["fork_b"]
        self.assertEqual(run(st).score, 0.0)
        self.assertEqual(run(None).score, 0.0)

    def test_nan_state(self):
        bad = hung()
        bad["fork_a"].origin[:] = math.nan
        res = run(bad)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertGreater(res.details["bad_steps"], 0)

    def test_nan_resets_the_windows(self):
        tr = S.Tracker()
        for _ in range(WIN + 2):
            tr.update(hung(), FAR)
        bad = hung()
        bad["caddy"].origin[:] = math.inf
        tr.update(bad, FAR)
        self.assertEqual(tr.update(hung(), FAR).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(scene(origin=[0.25, L.R_Y, 0.0]), FAR)
        tr.update(scene(origin=[0.5, 0.7, -0.08]), FAR)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            o = rng.uniform([0.0, -0.4, -0.1], [0.6, 0.2, 0.3])
            R = rot(*rng.uniform(-math.pi, math.pi, 3))
            res = S.Tracker().update(scene(origin=o, R=R), FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))
        tr = S.Tracker()
        for _ in range(WIN + 3):
            o = rng.uniform([0.0, -0.4, -0.1], [0.6, 0.2, 0.3])
            res = tr.update(scene(origin=o), FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))

    def test_details_are_json_clean(self):
        res = run(hung())
        json.dumps(res.details)

    def test_nothing_scores_over_one_by_a_trick(self):
        """A caddy turned a quarter about z (its hooks along the rail's normal) is not seated."""
        R = rot(yaw=math.pi / 2)
        self.assertEqual(score_of(scene(origin=pose_for_seat(R), R=R)), 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["wall", "tray", "plates", "caddy", "fork_a", "fork_b", "fork_c"])
        for fixed in (T.WALL, T.TRAY, T.PLATES):
            self.assertIsNone(fixed.mass)
        self.assertAlmostEqual(T.CADDY_PART.mass, L.CADDY_MASS)
        for p in T.FORK_PARTS:
            self.assertAlmostEqual(p.mass, 0.04)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("lift", "lower", "raise", "mm", "step", "tilt", "square", "cutlery", "fork"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            self.assertEqual(set(plan), {"right"})
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
                    self.assertGreaterEqual(w.pos[2], L.TOP + T.GRIP_DZ - 0.0005, (v, a, w.label))
                n_grips = sum(1 for x, y in zip(wps, wps[1:]) if abs(x.finger - y.finger) > 0.005)
                self.assertLessEqual(n_grips, 2)  # close, open
                for x, y in zip(wps, wps[1:]):
                    if abs(x.finger - y.finger) > 0.005:
                        self.assertGreaterEqual(y.duration, 1.5)  # a full open or close needs 1.5 s at 20 Hz
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ + 80, run_limit / 2)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", set()),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "short", {"right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_demo_timing(self):
        plan = T.plans("")["right"]
        self.assertAlmostEqual(sum(w.duration for w in plan), 20.5, places=3)

    def test_grip_geometry(self):
        self.assertAlmostEqual(T.GRIP_DZ, 0.098)
        self.assertAlmostEqual(-T.OPEN * 2 - L.C_W, 0.0179, places=4)  # 94 mm over 76 mm
        # pads (20 mm wide) centred 6 mm behind the caddy's middle cover the 18 mm wall depth less 3 mm
        self.assertLessEqual(L.GRIP_X + 0.010, L.C_D / 2)
        self.assertGreaterEqual(min(L.GRIP_X + 0.010, L.C_D / 2) - max(L.GRIP_X - 0.010, -L.C_D / 2), 0.012)
        # when hung, the pads' back edge is 5 mm short of the rail's tangent plane
        self.assertAlmostEqual(L.X_PLANE - (L.HUNG_X + L.GRIP_X + 0.010), 0.005, places=9)

    def test_lowering_geometry(self):
        pl = T.plans("")["right"]
        lower = [w for w in pl if w.label.startswith("lower")][0]
        self.assertAlmostEqual(lower.pos[2] - L.TOP, T.GRIP_DZ + L.HUNG_Z - T.PRESS)
        self.assertAlmostEqual(lower.pos[0], T.DEMO_X + L.GRIP_X)
        self.assertAlmostEqual(lower.pos[1], L.R_Y)

    def test_variants_miss_the_window(self):
        for v, dx in T.VARIANT_DX.items():
            qx = L.R_X - (T.DEMO_X + dx)
            self.assertFalse(L.QX_MIN - 0.001 <= qx <= L.QX_MAX + 0.001, v)


class OfflineModel(unittest.TestCase):
    def _play(self, m, tcps):
        for tcp, op in tcps:
            m.step({"left": FAR["left"], "right": arm(tcp, op)})

    def _grip(self, m, x=None):
        x = L.START_X + L.GRIP_X if x is None else x
        t = (x, L.START_Y, L.TOP + T.GRIP_DZ)
        self._play(m, [(t, 1.0)] * 3 + [(t, 0.72)] * 3)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)

    def test_grip_carry_and_hang(self):
        m = T.OfflineModel()
        self._grip(m)
        self.assertIsNotNone(m.held)
        gx = T.DEMO_X + L.GRIP_X
        up = L.TOP + T.GRIP_DZ + T.CARRY_Z
        self._play(m, [((gx, L.R_Y, up), 0.72)] * 2)
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + L.HUNG_Z - T.PRESS), 0.72)] * 3)
        self.assertAlmostEqual(float(m.origin[2]), L.HUNG_Z, places=9)  # the rail stops it
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + L.HUNG_Z - T.PRESS), 1.0)] * 3)
        self.assertIsNone(m.held)
        self.assertTrue(np.allclose(m.origin, [L.HUNG_X, L.R_Y, L.HUNG_Z]))
        r = S.read(m.states())
        self.assertTrue(all(S.hook_reading(r, s)["seated"] for s in (-1, 1)))

    def test_release_too_far_back_drops_to_the_table(self):
        m = T.OfflineModel()
        self._grip(m)
        gx = T.DEMO_X - 0.015 + L.GRIP_X
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + L.HUNG_Z), 0.72)] * 2)
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + L.HUNG_Z), 1.0)] * 2)
        self.assertAlmostEqual(float(m.origin[2]), 0.0)
        self.assertAlmostEqual(float(m.origin[0]), T.DEMO_X - 0.015)

    def test_release_in_the_air_drops(self):
        m = T.OfflineModel()
        self._grip(m)
        gx = T.DEMO_X + L.GRIP_X
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + T.CARRY_Z), 0.72)] * 2)
        self._play(m, [((gx, L.R_Y, L.TOP + T.GRIP_DZ + T.CARRY_Z), 1.0)] * 2)
        self.assertAlmostEqual(float(m.origin[2]), 0.0)

    def test_far_gripper_does_not_grab(self):
        m = T.OfflineModel()
        t = (L.START_X + 0.05, L.START_Y, L.TOP + T.GRIP_DZ)
        self._play(m, [(t, 0.72)] * 4)
        self.assertIsNone(m.held)

    def test_open_gripper_does_not_grab(self):
        m = T.OfflineModel()
        t = (L.START_X + L.GRIP_X, L.START_Y, L.TOP + T.GRIP_DZ)
        self._play(m, [(t, 1.0)] * 4)
        self.assertIsNone(m.held)

    def test_forks_ride_with_the_caddy(self):
        m = T.OfflineModel()
        self._grip(m)
        self._play(m, [((L.START_X + L.GRIP_X + 0.05, L.START_Y, L.TOP + T.GRIP_DZ + 0.05), 0.72)])
        r = S.read(m.states())
        self.assertTrue(all(S.forks_in(r)))
        self.assertAlmostEqual(float(r.pos["fork_b"][0]), float(r.pos["caddy"][0]))


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("y -110 mm", "y -240 to +20 mm", "y -235 and +15 mm", "y -260 to +40 mm",
                      "10 mm across", "260 mm long", "x 330 mm and z 135 mm", "z 140 mm", "6 mm across",
                      "z 40 and 85 mm", "x 325 mm", "230 x 300 x 12 mm", "180 mm across", "8 mm thick",
                      "y -155 and -65",
                      "0.062 kg", "18 (x) x 76 (y) x 100 (z) mm", "walls and base 3 mm", "two 2 mm dividers",
                      "22 mm wide", "x 200 mm, y -210 mm", "y -234 and -186 mm", "12 mm wide", "26 mm above the rim",
                      "25 mm toward the rack", "30 mm below the bridge", "18 mm clear", "42 degrees",
                      "14 mm above the table", "0.04 kg", "120 mm long", "16 mm wide", "three 3 mm tines",
                      "23 mm above the rim", "2 mm clear", "y -234, -210 and -186 mm"):
            self.assertIn(token, setup)
        for token in ("1 mm inside", "2 mm below to 3 mm above", "12 mm", "5 degrees", "30 mm", "1 mm",
                      "5 mm and turns under 5 degrees",
                      "50 mm", "Possible scores: 0, 0.25, 0.5, 1.0", "1 s", "0.5 hung with a fork out",
                      "0.25 one hook seated"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_thresholds_in_text_match_layout(self):
        self.assertAlmostEqual(L.SEAT_X_TOL, 0.001)
        self.assertAlmostEqual(L.SEAT_Z_LO, 0.002)
        self.assertAlmostEqual(L.SEAT_Z_HI, 0.003)
        self.assertAlmostEqual(L.END_MARGIN, 0.012)
        self.assertEqual((L.YAW_DEG, L.TILT_DEG), (5.0, 5.0))
        self.assertAlmostEqual(L.FALL_DROP, 0.050)
        self.assertAlmostEqual(L.LET_GO, 0.030)
        self.assertAlmostEqual(L.STILL_MOVE, 0.005)
        self.assertAlmostEqual(L.STILL_TURN_DEG, 5.0)

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Rigid bodies", "18 mm", "4 mm slack", "120 mm", "est", "No software rules"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("cutlery caddy", "rail", "hooks", "hands off"):
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
        bad = re.compile("\\u2014|\\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
