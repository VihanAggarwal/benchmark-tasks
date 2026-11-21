"""Offline tests for layout.py, scoring.py and task.py: synthetic sign states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
KIT_DIR = HERE.parents[1] / "tools" / "kit"
WIN = int(round(L.STILL_WINDOW_S * HZ))
LOCK = L.THETA_LOCK


def sign(hy=L.SY, theta=0.0, hx=L.SX, yaw=0.0):
    """The sign opened symmetrically (``theta`` per leaf, feet on the table), its hinge line at y ``hy`` (middle at x
    ``hx``), turned ``yaw`` about the vertical through its middle."""
    st = T.sign_state(hy, theta, hx)
    if yaw == 0.0:
        return st
    c = np.array([hx, hy, 0.0])
    rz = mk.axis_angle_matrix((0.0, 0.0, 1.0), yaw)
    r = rz @ st.rotation()
    origin = c + rz @ (np.asarray(st.origin) - np.array([0.0, 0.0, L.TOP]) - c) + np.array([0.0, 0.0, L.TOP])
    q = mk.matrix_to_quat(r)
    return mk.state_from_poses(T.SIGN_PART, (*origin, *q), joints=dict(st.joints), source="test")


def free_sign(roll_a, q, origin=None):
    """The sign with leaf A rolled ``roll_a`` (rad, + swings its foot toward -y) about the ridge and the hinge at
    angle ``q``, not necessarily on the table: for lean and fall tests."""
    origin = origin if origin is not None else (L.SX, L.SY, L.TOP + 0.3)
    return mk.state_from_poses(T.SIGN_PART, (*origin, *mk.quat_from_ypr(0.0, 0.0, -roll_a)),
                               joints={"leaf_b": q}, source="test")


def scene(st):
    return {S.SIGN: st}


def start():
    return scene(T.SIGN_PART.rest_state())


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def full(hy=L.SY):
    return sign(hy, LOCK)


def shifted_by(dn):
    """The open sign with its hinge line ``dn`` along y from where it opens in place."""
    return sign(L.SY + dn, LOCK)


class Geometry(unittest.TestCase):
    def test_the_stop_sets_the_spread(self):
        """One sign: a 610 mm leaf at 12.5 degrees each puts the feet 264 mm apart; the spread window holds it and
        ends 1.9 degrees (of hinge) short of it."""
        self.assertAlmostEqual(L.Q_STOP_DEG, 25.0, places=9)
        self.assertAlmostEqual(L.SPREAD_LOCK, 2 * 0.610 * math.sin(math.radians(12.5)), places=12)
        self.assertAlmostEqual(L.SPREAD_LOCK, 0.2641, places=4)
        self.assertLess(L.SPREAD_LO, L.SPREAD_LOCK)
        self.assertGreater(L.SPREAD_HI, L.SPREAD_LOCK)
        self.assertAlmostEqual(L.spread(L.Q_STOP - math.radians(1.9)), 0.24427, places=5)
        self.assertGreater(L.SPREAD_LO, L.spread(L.Q_STOP - math.radians(1.95)))
        self.assertLess(L.SPREAD_LO, L.spread(L.Q_STOP - math.radians(1.9)))
        self.assertLess(L.LEAN_LO_DEG, L.THETA_LOCK_DEG)
        self.assertGreater(L.LEAN_HI_DEG, L.THETA_LOCK_DEG)

    def test_hinge_swings_leaf_b_toward_plus_y(self):
        st = mk.state_from_poses(T.SIGN_PART, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), joints={"leaf_b": 0.3},
                                 source="test")
        foot = st.link_to_world("leaf_b", (0.0, 0.0, 0.0))
        self.assertAlmostEqual(foot[0], 0.0, places=9)
        self.assertAlmostEqual(foot[1], L.H * math.sin(0.3), places=9)
        self.assertAlmostEqual(foot[2], L.H * (1 - math.cos(0.3)), places=9)

    def test_symmetric_opening_keeps_both_feet_on_the_table(self):
        for th in np.linspace(0.0, LOCK, 9):
            st = sign(L.SY, th)
            r = S.read(scene(st))
            self.assertAlmostEqual(r.foot_a[2], 0.0, places=9)
            self.assertAlmostEqual(r.foot_b[2], 0.0, places=9)
            self.assertAlmostEqual(r.foot_a[1], L.SY - L.H * math.sin(th), places=9)
            self.assertAlmostEqual(r.foot_b[1], L.SY + L.H * math.sin(th), places=9)
            self.assertAlmostEqual(float(st.low[2]), L.TOP, places=6)  # nothing below the table top
            self.assertAlmostEqual(S.spread(r), L.spread(2 * th), places=9)
            self.assertAlmostEqual(r.lean_a, math.degrees(th), places=6)
            self.assertAlmostEqual(r.lean_b, math.degrees(th), places=6)
            self.assertAlmostEqual(r.yaw, 0.0, places=6)
            self.assertTrue(np.allclose(r.hinge, r.hinge_a, atol=1e-9))
            self.assertAlmostEqual(r.hinge[0], L.SX, places=9)

    def test_start_is_folded_and_clear_of_everything(self):
        r = S.read(start())
        self.assertAlmostEqual(S.spread(r), 0.0, places=9)
        self.assertFalse(S.over_patch(r))
        self.assertGreater(L.PATCH_Y0 - float(r.high[1]), 0.05)  # 50 mm or more from the mat as it stands folded
        self.assertGreater(L.WALL_X0 - L.PATCH_X1, 0.05)
        self.assertGreater(L.WALL_X0 - float(r.high[0]), 0.10)
        self.assertEqual(S.tier(True, r, True), 0.0)  # do nothing: 0
        self.assertTrue(S.Tracker.at_start(r))
        self.assertAlmostEqual(float(T.SIGN_PART.rest_state().low[2]), L.TOP, places=9)
        self.assertTrue(S.beside(r))

    def test_nothing_under_the_arms_start_pose(self):
        """No part in the strip the arms' start pose hides (|y| over 0.45 m with x under 0.2 m), and the mat starts at
        x 200 mm."""
        r = S.read(start())
        self.assertLess(float(np.max(np.abs([r.low[1], r.high[1]]))), 0.45)
        self.assertGreaterEqual(L.PATCH_X0, 0.200)
        self.assertLess(max(abs(L.PATCH_Y0), abs(L.PATCH_Y1)), 0.45)

    def test_opening_in_place_lands_in_the_middle_of_the_band(self):
        self.assertAlmostEqual(L.GAP_SYMMETRIC, 0.090, places=9)
        self.assertAlmostEqual((L.GAP_LO + L.GAP_HI) / 2, 0.090, places=9)
        r = S.read(scene(full()))
        self.assertAlmostEqual(S.near_gap(r), 0.090, places=6)
        self.assertFalse(S.over_patch(r))
        self.assertGreater(L.PATCH_Y0 - float(np.max(r.pads[:, 1])), 0.050)  # the pads are 50 mm or more from the mat

    def test_one_leaf_pulled_alone_shifts_the_sign_out_of_the_band(self):
        """Drawing only leaf A (its foot B holds the table): the hinge moves toward -y by H sin(theta); leaf B alone,
        toward +y, onto the mat."""
        r = S.read(scene(shifted_by(-L.H * math.sin(LOCK))))
        self.assertGreater(S.near_gap(r), L.GAP_HI)
        r = S.read(scene(shifted_by(L.H * math.sin(LOCK))))
        self.assertLess(S.near_gap(r), L.GAP_LO)
        self.assertTrue(S.over_patch(r))

    def test_masses(self):
        self.assertAlmostEqual(L.MASS_TOTAL, 0.55, places=3)
        self.assertAlmostEqual(T.SIGN_PART.mass, L.MASS_A)

    def test_pull_force_is_within_the_arms(self):
        """Quasi-static: each foot carries half the weight; sliding it out takes friction x that, and a hand 75 mm up
        needs H / (H - 75 mm) of it. Under 3 N each (the arms give about 5 N)."""
        foot = max(L.FRICTION) * L.MASS_TOTAL * L.G / 2
        hand = foot * L.H / (L.H - L.Z_GRIP)
        self.assertLess(hand, 3.0)
        # and the open sign does not splay by itself: the thrust at each foot needs friction tan(theta) / 2
        self.assertLess(math.tan(LOCK) / 2, min(L.FRICTION) / 2)

    def test_grip_geometry(self):
        """The pads (27 mm wide, closing along x across the 20 mm grip) sit inside the grip; the open fingers pass the
        grip with 20 mm each side; the hover keeps the pad tips over the grip's top."""
        self.assertTrue(16e-3 <= L.RIB_W <= 80e-3)
        self.assertGreaterEqual(L.U_GRIP - 0.0135, L.T)
        self.assertLessEqual(L.U_GRIP + 0.0135, L.T + L.RIB_P)
        self.assertLess(abs(T.GRIP_G), (L.RIB_W) / 2)  # commanded 8 mm narrower than the grip
        self.assertAlmostEqual(-2 * T.GRIP_G, L.RIB_W - 0.008, places=9)
        self.assertAlmostEqual(-2 * T.OPEN_G, L.RIB_W + 0.040, places=9)
        self.assertGreater(T.HOVER - 0.013 - L.RIB_Z1, 0.015)
        z0 = L.Z_GRIP - 0.013  # pad tips
        self.assertGreater(z0, L.RIB_Z0)
        self.assertLess(L.Z_GRIP + 0.025, L.RIB_Z1)
        for leaf, yaw in T.YAW.items():
            self.assertAlmostEqual(abs(yaw), math.pi / 2)  # fingers close along x, across the grip

    def test_grip_point_is_on_the_grip_through_the_opening(self):
        """The grip point follows the grip as the leaf swings: its distance from the hinge is constant, and it is the
        real point of the declared leaves (forward kinematics)."""
        r0 = math.hypot(L.U_GRIP, L.H - L.Z_GRIP)
        for leaf in ("A", "B"):
            for th in np.linspace(0.0, LOCK + T.OVERTRAVEL, 7):
                n, z = L.rib_point(leaf, th)
                self.assertAlmostEqual(math.hypot(n, z - L.H * math.cos(th)), r0, places=9)
        for th in (0.0, 0.1, LOCK):
            st = sign(L.SY, th)
            ra, oa = st.link_pose("base")
            pa = oa + ra @ np.array([0.0, -L.U_GRIP, L.Z_GRIP])
            pb = st.link_to_world("leaf_b", (0.0, L.U_GRIP, L.Z_GRIP))
            na, za = L.rib_point("A", th)
            nb, zb = L.rib_point("B", th)
            self.assertAlmostEqual(pa[0], L.SX, places=9)
            self.assertAlmostEqual(pa[1], L.SY + na, places=9)
            self.assertAlmostEqual(pa[2], L.TOP + za, places=9)
            self.assertAlmostEqual(pb[1], L.SY + nb, places=9)
            self.assertAlmostEqual(pb[2], L.TOP + zb, places=9)

    def test_tip_over_margin_folded(self):
        """The folded sign stands on pads 30 mm deep each side of the hinge line (the grab handles shift its centre of
        mass a little): it tips at 4 degrees or more."""
        com = T.SIGN_PART.total_center_of_mass()
        self.assertLess(abs(com[1]), 0.012)
        self.assertGreater(math.degrees(math.atan2(L.PAD_D - abs(com[1]), com[2])), 4.0)

    def test_each_hand_stays_on_its_own_side(self):
        """The right hand works leaf A (y below the hinge line), the left hand leaf B: the hands are 230 mm or more
        apart and never cross the midline toward the other base by more than 0.1 m."""
        for th in (0.0, LOCK + T.OVERTRAVEL):
            _, ya, _ = T.grip_xyz("A", th)
            _, yb, _ = T.grip_xyz("B", th)
            self.assertGreater(yb - ya, 0.23)
            self.assertLess(ya, 0.0)
            self.assertGreater(yb, 0.0)
        self.assertGreaterEqual(L.SX, 0.12)


@unittest.skipIf(OFFLINE is None, "tools/offline (yamkin) is not next to this task")
class ArmClearance(unittest.TestCase):
    """The arms' collision capsules (tools/offline/yamkin.py, fitted to the simulator's hulls) against the leaves' real
    boxes (not the sign's bounding box, which the checker's proxy uses): the wrist is about 52 mm round and sits over
    the grip, so the grip stands well out from the face."""

    @staticmethod
    def clearance(theta, z):
        import yamkin as K
        st = sign(L.SY, theta)
        worst = math.inf
        for leaf, side, base in (("B", "left", (0.0, 0.305, 0.75)), ("A", "right", (0.0, -0.305, 0.75))):
            n, zz = L.rib_point(leaf, theta)
            pos = np.array([L.SX, L.SY + n, L.TOP + (zz if z is None else z)])
            q, err, rerr = K.solve(pos, K.matrix_to_quat_xyzw(K.grip_rotation(T.YAW[leaf])), base_pos=base)
            assert err < 2e-3 and rerr < 0.03, (leaf, theta, err, rerr)
            caps = [c for c in K.arm().capsules(q, base) if c[0] not in K.FINGER_LINKS]
            for body, sgn in (("base", -1.0), ("leaf_b", 1.0)):
                if body == "base":
                    rot, org = st.link_pose("base")
                    to_local = lambda p, rot=rot, org=org: rot.T @ (p - org)
                else:
                    rot, org = st.link_pose("leaf_b")
                    hinge = np.array([0.0, 0.0, L.H])
                    to_local = lambda p, rot=rot, org=org, hinge=hinge: rot.T @ (p - org) + hinge
                lo = np.array([-L.W / 2, min(0.0, sgn * L.T), 0.0])
                hi = np.array([L.W / 2, max(0.0, sgn * L.T), L.H])
                for _, a, b, rad in caps:
                    for t in np.linspace(0.0, 1.0, 21):
                        p = to_local(a + (b - a) * t)
                        d = float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi))) - rad
                        worst = min(worst, d)
        return worst

    def test_wrists_clear_the_leaves(self):
        for th, z in ((0.0, T.HOVER), (0.0, None), (0.5 * LOCK, None), (LOCK + T.OVERTRAVEL, None)):
            self.assertGreater(self.clearance(th, z), 0.008, (th, z))


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(scene(full()))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertTrue(all(d[k] for k in ("spread_ok", "lean_ok", "placed", "beside", "square", "grounded")))
        self.assertTrue(d["released_2s"] and d["still_2s"])

    def test_tier_table(self):
        cases = {
            "do nothing": (scene(T.SIGN_PART.rest_state()), 0.0),
            "full": (scene(full()), 1.0),
            "open part way": (scene(sign(L.SY, 0.5 * LOCK)), 0.25),
            "nearly shut": (scene(sign(L.SY, 0.2 * LOCK)), 0.0),
            "leaf A drawn alone (shifted away from the mat)": (scene(shifted_by(-L.H * math.sin(LOCK))), 0.5),
            "leaf B drawn alone (onto the mat)": (scene(shifted_by(L.H * math.sin(LOCK))), 0.25),
            "ridge 20 degrees off": (scene(sign(L.SY, LOCK, yaw=math.radians(20))), 0.5),
            "ridge 40 degrees off": (scene(sign(L.SY, LOCK, yaw=math.radians(40))), 0.25),
            "ridge 8 degrees off": (scene(sign(L.SY, LOCK, yaw=math.radians(8))), 1.0),
            "slid off the end of the mat": (scene(sign(L.SY, LOCK, hx=L.PATCH_X1 + 0.05)), 0.5),
            "lying over": (scene(free_sign(math.radians(80), L.Q_STOP, (0.3, 0.0, L.TOP + 0.05))), 0.0),
        }
        for name, (st, want) in cases.items():
            self.assertEqual(run(st).score, want, name)

    def test_gripper_within_50_mm_is_075(self):
        n, z = L.rib_point("A", LOCK)
        for tcp_z in (z, 0.2):  # fingers on the grip, and open just above it
            hands = {"left": FAR["left"], "right": arm(np.array([L.SX, L.SY + n, L.TOP + tcp_z]), 0.1)}
            r = run(scene(full()), hands)
            self.assertEqual(r.score, 0.75, tcp_z)
            self.assertFalse(r.success)

    def test_spread_window(self):
        """The feet 244 mm or more apart: the hinge within 1.9 degrees of its stop (25 degrees)."""
        for q_deg, want in ((23.2, 1.0), (23.0, 0.25), (21.0, 0.25), (25.0, 1.0)):
            r = run(scene(sign(L.SY, math.radians(q_deg) / 2)))
            self.assertEqual(r.score, want, q_deg)

    def test_open_short_of_the_stop_is_not_open(self):
        """2.2 degrees short of the stop: a narrower base, not the sign as designed."""
        th = (L.Q_STOP - math.radians(2.2)) / 2
        r = S.read(scene(sign(L.SY, th)))
        self.assertFalse(S.spread_ok(r))
        self.assertTrue(S.lean_ok(r))
        self.assertEqual(run(scene(sign(L.SY, th))).score, 0.25)

    def test_lean_window(self):
        """One leaf upright, the other drawn the whole way (the sign stands on one foot and a corner): not a sign
        standing square on both feet."""
        st = free_sign(0.0, L.Q_STOP, (L.SX, L.SY, L.TOP))
        r = S.read(scene(st))
        self.assertFalse(S.lean_ok(r))
        self.assertEqual(S.tier(True, r, True), 0.5 if S.grounded(r) else 0.0)

    def test_lean_edges(self):
        base = S.read(scene(full()))
        for lo, hi in ((L.LEAN_LO_DEG, L.LEAN_HI_DEG),):
            for a, b, want in ((lo, lo, True), (hi, hi, True), (lo - 0.5, hi, False), (lo, hi + 0.5, False)):
                r = S.Reading(**{**base.__dict__, "lean_a": a, "lean_b": b})
                self.assertEqual(S.lean_ok(r), want, (a, b))

    def test_placement_band_edges(self):
        for gap, want in ((0.036, False), (0.041, True), (0.139, True), (0.145, False)):
            st = sign(L.PATCH_Y0 - gap - L.H * math.sin(LOCK), LOCK)
            r = S.read(scene(st))
            self.assertAlmostEqual(S.near_gap(r), gap, places=6)
            self.assertEqual(S.placed(r), want, gap)

    def test_sign_set_on_the_mat_is_not_placed(self):
        """Exploit: stand the open sign on the mat it should mark: over the patch, 0.25 at best."""
        st = sign(L.PATCH_Y0 + 0.02, LOCK)  # pad B well on the mat
        r = S.read(scene(st))
        self.assertTrue(S.over_patch(r))
        self.assertEqual(run(scene(st)).score, 0.25)

    def test_sign_beside_the_mat_along_its_length(self):
        """The hinge's middle must lie within the mat's 300 mm length (x 200 to 500 mm): a sign slid along the
        corridor's width to the mat's end still counts, one past it does not."""
        for hx, want in ((L.PATCH_X0 + 0.001, True), (L.PATCH_X1 - 0.001, True), (L.PATCH_X0 - 0.01, False),
                         (L.PATCH_X1 + 0.01, False)):
            r = S.read(scene(sign(L.SY, LOCK, hx=hx)))
            self.assertEqual(S.beside(r), want, hx)
            self.assertTrue(S.placed(r))
            self.assertEqual(run(scene(sign(L.SY, LOCK, hx=hx))).score, 1.0 if want else 0.5, hx)

    def test_pads_are_tested_one_by_one(self):
        """The review's point: the open sign standing astride the mat (the mat between its feet, no pad on it) is not
        'a pad over the patch'; a pad that does reach the mat is, even with the sign turned."""
        mid = (L.PATCH_Y0 + L.PATCH_Y1) / 2
        r = S.read(scene(sign(mid, LOCK)))
        self.assertLess(float(np.max(r.pads[:8, 1])), L.PATCH_Y0)
        self.assertGreater(float(np.min(r.pads[8:, 1])), L.PATCH_Y1)
        self.assertFalse(S.over_patch(r))
        self.assertFalse(S.placed(r))
        self.assertEqual(run(scene(sign(mid, LOCK))).score, 0.5)
        r = S.read(scene(sign(L.PATCH_Y0 - L.H * math.sin(LOCK) + 0.010, LOCK)))  # pad B 20 mm onto the mat's edge
        self.assertTrue(S.over_patch(r))
        r = S.read(scene(sign(L.SY, LOCK, yaw=math.radians(25))))
        self.assertFalse(S.over_patch(r))
        st = sign(L.PATCH_Y0 - L.H * math.sin(LOCK) + 0.025, LOCK, yaw=math.radians(25))
        self.assertTrue(S.over_patch(S.read(scene(st))))

    def test_sign_on_the_far_side_of_the_mat_is_not_placed(self):
        """Exploit: the sign beyond the mat (its feet past the near edge): the gap is negative."""
        st = sign(L.PATCH_Y1 + 0.20, LOCK)
        r = S.read(scene(st))
        self.assertLess(S.near_gap(r), 0.0)
        self.assertFalse(S.placed(r))

    def test_sign_held_off_the_table_scores_zero(self):
        """Exploit: lift the open sign (or lean it on something) so its feet are off the table."""
        st = sign(L.SY, LOCK)
        lifted = mk.state_from_poses(T.SIGN_PART, (*np.asarray(st.origin) + np.array([0.0, 0.0, 0.05]),
                                                   *st.quat_xyzw), joints=dict(st.joints), source="test")
        self.assertEqual(run(scene(lifted)).score, 0.0)

    def test_leaf_b_pushed_through_the_stop_is_not_full(self):
        """A hinge reading beyond the stop (a simulator fault): the spread leaves its window."""
        th = LOCK + math.radians(3.0)
        st = mk.state_from_poses(T.SIGN_PART, (L.SX, L.SY - L.H * math.sin(th), L.TOP, *mk.quat_from_ypr(0, 0, -th)),
                                 joints={"leaf_b": 2 * th}, source="test")
        r = S.read(scene(st))
        self.assertFalse(S.spread_ok(r))
        self.assertEqual(run(scene(st)).score, 0.25)

    def test_a_folded_sign_pushed_along_scores_zero(self):
        self.assertEqual(run(scene(sign(L.SY + 0.05, 0.0))).score, 0.0)

    def test_fall_gate(self):
        for deg, want in ((44.0, False), (46.0, True)):
            r = S.read(scene(free_sign(math.radians(deg), 0.4, (L.SX, L.SY, L.TOP + 0.05))))
            self.assertEqual(S.fallen(r), want, deg)
        # the gate looks at either leaf: leaf B over, leaf A upright
        r = S.read(scene(free_sign(0.0, math.radians(50), (L.SX, L.SY, L.TOP + 0.3))))
        self.assertTrue(S.fallen(r))


class ReleaseAndStill(unittest.TestCase):
    def test_hand_clear(self):
        self.assertEqual(run(scene(full())).score, 1.0)

    def test_parked_hands_are_clear(self):
        parked = {"left": arm(T.PARK_L[:2] + (L.TOP + T.PARK_L[2],)),
                  "right": arm(T.PARK_R[:2] + (L.TOP + T.PARK_R[2],))}
        self.assertEqual(run(scene(full()), parked).score, 1.0)

    def test_hands_open_just_above_the_grips_are_not_clear(self):
        n, _ = L.rib_point("B", LOCK)
        hover_over_b = np.array([L.SX, L.SY + n, L.TOP + T.HOVER])
        r = run(scene(full()), {"left": arm(hover_over_b, 1.0), "right": FAR["right"]})
        self.assertEqual(r.score, 0.75)

    def test_unreadable_arms_never_release(self):
        for arms in (None, {"left": FAR["left"]}):
            r = run(scene(full()), arms)
            self.assertEqual(r.score, 0.75)
            self.assertFalse(r.details["released_2s"])

    def test_nan_arm(self):
        bad = {"left": arm((float("nan"), 0.5, L.TOP + 0.2)), "right": FAR["right"]}
        self.assertEqual(run(scene(full()), bad).score, 0.75)

    def test_release_must_hold_the_whole_window(self):
        tr = S.Tracker()
        n, z = L.rib_point("B", LOCK)
        near = {"left": arm((L.SX, L.SY + n, L.TOP + z), 0.1), "right": FAR["right"]}
        for _ in range(WIN + 5):
            tr.update(scene(full()), FAR)
        tr.update(scene(full()), near)  # one step with a hand on the sign
        r = None
        for _ in range(WIN - 1):
            r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 0.75)
        r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 1.0)

    def test_moving_sign_scores_zero(self):
        tr = S.Tracker()
        for i in range(WIN + 5):  # sliding along y 0.4 mm a step: 16 mm over the window
            r = tr.update(scene(sign(L.SY + 0.0004 * i, LOCK)), FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["still_2s"])

    def test_creeping_spread(self):
        """The spread changing 2.5 mm over the window (a leaf creeping) is not standing."""
        tr = S.Tracker()
        for i in range(WIN + 5):
            th = LOCK - math.radians(0.8) * (1 - min(i, WIN) / WIN)
            r = tr.update(scene(sign(L.SY, th)), FAR)
        self.assertEqual(r.score, 0.0)

    def test_still_needs_the_whole_window(self):
        tr = S.Tracker()
        for i in range(WIN - 1):
            r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 0.0)
        r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 0.0)  # WIN readings: the window needs WIN + 1
        r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 1.0)

    def test_small_jitter_is_still(self):
        rng = np.random.default_rng(3)
        tr = S.Tracker()
        for _ in range(WIN + 5):
            r = tr.update(scene(sign(L.SY + float(rng.normal(0, 3e-4)), LOCK)), FAR)
        self.assertEqual(r.score, 1.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        self.assertEqual(run({}).score, 0.0)
        self.assertEqual(run({S.SIGN: None}).score, 0.0)
        self.assertEqual(run(None).score, 0.0)

    def test_nan_state(self):
        st = full()
        bad = mk.AssemblyState(**{**st.__dict__, "origin": np.array([float("nan"), 0.0, L.TOP])})
        r = run(scene(bad))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        bad = mk.AssemblyState(**{**st.__dict__, "joints": {**st.joints, "leaf_b": float("inf")}})
        self.assertEqual(run(scene(bad)).score, 0.0)

    def test_a_blow_up_restarts_the_window(self):
        tr = S.Tracker()
        for _ in range(WIN + 5):
            tr.update(scene(full()), FAR)
        st = full()
        bad = mk.AssemblyState(**{**st.__dict__, "origin": np.array([float("nan"), 0.0, L.TOP])})
        tr.update(scene(bad), FAR)
        r = None
        for _ in range(WIN):
            r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 0.0)
        r = tr.update(scene(full()), FAR)
        self.assertEqual(r.score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        for _ in range(WIN + 5):
            tr.update(scene(full()), FAR)
        r = tr.update(start(), FAR)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.steps, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            roll = float(rng.uniform(-1.6, 1.6))
            q = float(rng.uniform(-0.3, 1.0))
            st = free_sign(roll, min(max(q, 0.0), L.Q_STOP), tuple(rng.uniform([0.0, -0.3, 0.7], [0.5, 0.4, 1.2])))
            tr = S.Tracker()
            r = None
            for _ in range(3):
                r = tr.update(scene(st), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details)

    def test_details_are_json_clean(self):
        json.dumps(run(scene(full())).details)
        json.dumps(run({}).details)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["wall", "patch", "sign"])
        self.assertEqual(len(set(names)), 3)
        self.assertIsNone(T.WALL.mass)
        self.assertIsNone(T.PATCH.mass)
        self.assertEqual(T.SIGN_PART.kind, "articulation")
        self.assertEqual([j.name for j in T.SIGN_PART.joints], ["leaf_b"])
        lb = T.SIGN_PART.link("leaf_b")
        self.assertAlmostEqual(lb.upper, L.Q_STOP)
        self.assertEqual(lb.lower, 0.0)
        self.assertEqual(lb.axis, "X")
        self.assertEqual(lb.spring, 0.0)  # no software rule or spring: the stop alone ends the opening
        self.assertEqual(mk.rules().get("on_step", []) if isinstance(mk.rules(), dict) else [], [])

    def test_instruction_states_the_goal_only(self):
        self.assertNotIn("lock", T.INSTRUCTION.lower())
        self.assertNotIn("mm", T.INSTRUCTION)
        self.assertLess(len(T.INSTRUCTION), 300)

    def test_demo_plans(self):
        plans = T.plans()
        self.assertEqual(set(plans), {"left", "right"})
        for side, plan in plans.items():
            for w in plan:
                self.assertEqual(len(w.pos), 3)
                self.assertTrue(all(math.isfinite(v) for v in w.pos))
                self.assertGreater(w.duration, 0.0)
                self.assertIn(w.yaw, (T.YAW["B" if side == "left" else "A"], T.PARK_YAW))
        self.assertAlmostEqual(T._duration(plans["left"]), T._duration(plans["right"]), places=9)

    def test_demo_env_switches(self):
        old = {k: os.environ.pop(k, None) for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT")}
        try:
            self.assertEqual(set(T.demo()), {"left", "right"})
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            del os.environ["DEMO_HOLD_ONLY"]
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(set(T.demo()), {"right"})
            del os.environ["DEMO_IDLE_ARM"]
            os.environ["DEMO_VARIANT"] = "nonsense"
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
                os.environ.pop(k, None)
                if old[k] is not None:
                    os.environ[k] = old[k]

    def test_demo_timing(self):
        steps = int(round(T._duration(T.plans()["left"]) * HZ))
        limit = int(re.match(r"([\d,]+) steps", json.loads((HERE / "task.json").read_text())["run_limit"])
                    .group(1).replace(",", ""))
        self.assertLessEqual(steps, limit // 2)
        self.assertLessEqual(steps + 3 * int(HZ) + WIN + int(T.SUCCESS_HOLD_S * HZ), limit)

    def test_grasp_and_release_times(self):
        """Every full open or close takes 1.5 s or more (the gripper's delta limit)."""
        for side, plan in T.plans().items():
            prev = None
            for w in plan:
                if prev is not None and abs(w.finger - prev.finger) >= 0.012:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
                prev = w

    def test_draw_starts_together_and_ends_together(self):
        plans = T.plans()

        def start_of_draw(plan):
            t = 0.0
            for w in plan:
                if w.label.startswith("draw the leaf out"):
                    return t
                t += w.duration

        self.assertAlmostEqual(start_of_draw(plans["left"]), start_of_draw(plans["right"]), places=9)

    def test_draw_path_follows_the_rib(self):
        for leaf, side in (("A", "right"), ("B", "left")):
            plan = [w for w in T.plans()[side] if w.label.startswith("draw the leaf out")]
            self.assertEqual(len(plan), T.ARC_STEPS)
            end = plan[-1].pos
            n, z = L.rib_point(leaf, L.THETA_LOCK + T.OVERTRAVEL)
            self.assertAlmostEqual(end[1], L.SY + n, places=9)
            self.assertAlmostEqual(end[2], L.TOP + z, places=9)
            self.assertAlmostEqual(end[0], L.SX)
        # the pull ends 1 degree past the stop: about 10 mm of the arm's give at the grip, and the pads keep 18 mm of
        # the handle beyond them if the pull slides them out along it
        self.assertGreater(T.OVERTRAVEL * math.hypot(L.U_GRIP, L.H - L.Z_GRIP), 0.008)
        self.assertLess(T.OVERTRAVEL * math.hypot(L.U_GRIP, L.H - L.Z_GRIP), L.T + L.RIB_P - (L.U_GRIP + 0.0135))

    def test_pull_needs_the_stop(self):
        """The commanded end is beyond the stop: only the hinge's stop makes the leaves end there."""
        self.assertGreater(L.THETA_LOCK + T.OVERTRAVEL, L.THETA_LOCK)
        self.assertLess(L.THETA_LOCK + T.OVERTRAVEL, math.radians(14.5))

    def test_variants_change_only_the_end(self):
        os.environ["DEMO_VARIANT"] = "short"
        try:
            short = T.demo()
        finally:
            del os.environ["DEMO_VARIANT"]
        full_end = [w for w in T.plans()["left"] if w.label.startswith("draw the leaf out")][-1].pos
        short_end = [w for w in short["left"] if w.label.startswith("draw the leaf out")][-1].pos
        self.assertLess(short_end[1], full_end[1])


class OfflineModel(unittest.TestCase):
    """The checker's stand-in for the physics (never used in the simulator)."""

    @staticmethod
    def rib_tcp(leaf, theta, hy=L.SY, hx=L.SX):
        n, z = L.rib_point(leaf, theta)
        return np.array([hx, hy + n, L.TOP + z])

    def hands(self, ta, tb, oa=0.1, ob=0.1, hy=L.SY):
        return {"left": arm(self.rib_tcp("B", tb, hy), ob), "right": arm(self.rib_tcp("A", ta, hy), oa)}

    def test_start_matches_the_spawn(self):
        st = T.OfflineModel().states()[S.SIGN]
        self.assertTrue(np.allclose(st.origin, T.SIGN_PART.rest_state().origin, atol=1e-9))
        self.assertTrue(np.allclose(st.low, T.SIGN_PART.rest_state().low, atol=1e-9))

    def test_open_hands_do_nothing(self):
        m = T.OfflineModel()
        for th in np.linspace(0.0, LOCK, 10):
            m.step(self.hands(th, th, 1.0, 1.0))
        self.assertEqual(m.theta, 0.0)

    def test_two_hands_open_it_symmetrically_about_the_hinge_line(self):
        m = T.OfflineModel()
        m.step(self.hands(0.0, 0.0))
        self.sweep(m, 0.0, LOCK + T.OVERTRAVEL, 0.0, LOCK + T.OVERTRAVEL, n=60)
        self.assertAlmostEqual(m.theta, LOCK, places=6)  # stopped at the stop
        self.assertAlmostEqual(m.hy, L.SY, places=6)
        r = S.read(m.states())
        self.assertTrue(S.spread_ok(r))
        self.assertAlmostEqual(S.near_gap(r), L.GAP_SYMMETRIC, places=5)

    def sweep(self, m, a0, a1, b0, b1, n=30, oa=0.1, ob=0.1):
        for k in range(n + 1):
            f = k / n
            m.step(self.hands(a0 + (a1 - a0) * f, b0 + (b1 - b0) * f, oa, ob))

    def test_two_hands_part_way(self):
        m = T.OfflineModel()
        m.step(self.hands(0.0, 0.0))
        self.sweep(m, 0.0, 0.1, 0.0, 0.1)
        self.assertAlmostEqual(m.theta, 0.1, places=6)

    def test_released_sign_stays_open(self):
        m = T.OfflineModel()
        m.step(self.hands(0.0, 0.0))
        self.sweep(m, 0.0, LOCK, 0.0, LOCK)
        m.step({"left": arm((0.2, 0.4, L.TOP + 0.2)), "right": arm((0.2, -0.4, L.TOP + 0.2))})
        self.assertAlmostEqual(m.theta, LOCK, places=6)
        self.assertEqual(m.hold, {})

    def test_one_hand_only_shifts_the_sign(self):
        m = T.OfflineModel()
        for k in range(41):  # the right hand draws leaf A out along the path the two-handed demo takes
            th = LOCK * k / 40
            m.step({"left": FAR["left"], "right": arm(self.rib_tcp("A", th), 0.1)})
        self.assertLess(m.hy, L.SY)  # dragged toward the pulling hand's side
        self.assertGreater(S.near_gap(S.read(m.states())), 0.0)
        m2 = T.OfflineModel()
        for k in range(41):
            th = LOCK * k / 40
            m2.step({"right": FAR["right"], "left": arm(self.rib_tcp("B", th), 0.1)})
        self.assertGreater(m2.hy, L.SY)

    def test_hand_must_be_closed_and_on_the_rib(self):
        m = T.OfflineModel()
        m.step({"left": FAR["left"], "right": arm(self.rib_tcp("A", 0.0) + np.array([0.03, 0.0, 0.0]), 0.1)})
        self.assertEqual(m.hold, {})
        m.step({"left": FAR["left"], "right": arm(self.rib_tcp("A", 0.0) + np.array([0.0, 0.0, 0.05]), 0.1)})
        self.assertEqual(m.hold, {})
        m.step({"left": FAR["left"], "right": arm(self.rib_tcp("A", 0.0), 0.7)})
        self.assertEqual(m.hold, {})
        m.step({"left": FAR["left"], "right": arm(self.rib_tcp("A", 0.0), 0.1)})
        self.assertEqual(m.hold, {"right": "A"})

    def test_unreadable_arms_leave_it_alone(self):
        m = T.OfflineModel()
        m.step(None)
        self.assertEqual((m.hy, m.theta), (L.SY, 0.0))

    def test_demo_waypoints_fed_through_the_model_open_the_sign(self):
        """Walk both plans' TCP paths at 20 Hz through the model, as the checker does."""
        plans = T.plans()
        dur = T._duration(plans["left"])
        n = int(round(dur * HZ))

        def pos(plan, t):
            acc, prev = 0.0, None
            for w in plan:
                if t <= acc + w.duration or w is plan[-1]:
                    a = np.array(prev.pos if prev is not None else w.pos, dtype=float)
                    b = np.array(w.pos, dtype=float)
                    f0 = prev.finger if prev is not None else w.finger
                    u = min(max((t - acc) / w.duration, 0.0), 1.0)
                    u = u * u * (3 - 2 * u)
                    return a + (b - a) * u, f0 + (w.finger - f0) * u
                acc += w.duration
                prev = w

        m = T.OfflineModel()
        for k in range(n):
            t = (k + 1) / HZ
            arms = {}
            for side in ("left", "right"):
                p, f = pos(plans[side], t)
                arms[side] = arm(p, min(max(f / rigstate.FINGER_OPEN, 0.0), 1.0))
            m.step(arms)
        self.assertAlmostEqual(m.theta, LOCK, places=3)
        self.assertAlmostEqual(m.hy, L.SY, places=3)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("280 mm wide (along x) and 610 mm tall", "0.55 kg", "0.275 kg", "12 mm thick", "x 330 mm",
                      "y +10 mm", "20 mm across", "140 mm out", "from 30 to 120 mm up", "25 degrees", "12.5 degrees",
                      "264 mm apart", "300 x 200 x 1 mm", "x 200 to 500 mm", "y 232 to 432 mm", "30 mm deep",
                      "260 mm wide", "x 600 to 640 mm", "90 mm from the patch", "70 mm from it", "0.02 N m",
                      "friction 0.6", "200 x 150 mm centred 440 mm up"):
            self.assertIn(token, setup)
        for token in ("244 to 270 mm", "264 mm", "1.9 degrees", "9 to 16 degrees", "40 to 140 mm", "x 200 to 500 mm",
                      "10 degrees",
                      "30 degrees", "2 s", "50 mm", "3 mm", "2 mm", "45 degrees", "70 mm", "1 degree",
                      "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "10 mm", "held 1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1,000 steps (50 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_numbers_in_text_follow_layout(self):
        self.assertEqual(round(L.SPREAD_LOCK * 1000), 264)
        self.assertEqual(round(L.H * 1000), 610)
        self.assertEqual(round(L.W * 1000), 280)
        self.assertEqual(round(L.MASS_TOTAL, 3), 0.55)
        self.assertEqual(round(L.MASS_A, 3), 0.275)
        self.assertEqual(round(L.THETA_LOCK_DEG, 1), 12.5)
        self.assertEqual(round(L.Q_STOP_DEG, 1), 25.0)
        self.assertEqual(round(L.SX * 1000), 330)
        self.assertEqual(round(L.SY * 1000), 10)
        self.assertEqual(round(L.PATCH_Y0 * 1000), 232)
        self.assertEqual(round(L.PATCH_Y1 * 1000), 432)
        self.assertEqual((round(L.PATCH_X0 * 1000), round(L.PATCH_X1 * 1000)), (200, 500))
        self.assertEqual(round(L.RIB_P * 1000), 140)
        self.assertEqual(round((L.PATCH_Y0 - (L.SY + L.T + L.RIB_P)) * 1000), 70)  # folded handle to the mat
        self.assertEqual(round(L.GAP_SYMMETRIC * 1000), 90)
        self.assertEqual(round(L.SPREAD_LO * 1000), 244)

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("stop", "hinge", "estimates", "140 mm", "not modelled"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("37-2011", "BLS"):
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
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f.endswith(".py") else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
