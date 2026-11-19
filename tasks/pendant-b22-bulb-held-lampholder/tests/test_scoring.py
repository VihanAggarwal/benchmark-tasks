"""Offline tests for layout.py, parts.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
    import parts as P  # noqa: E402
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
Q0 = (0.0, 0.0, 0.0, 1.0)
PEND_ORIGIN = (L.PEND_X, L.PEND_Y, L.TOP)


def quat_of(r):
    """A rotation matrix as (x, y, z, w)."""
    r = np.asarray(r, dtype=float)
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (x, y, z, w)


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def pendant_state(rise=0.0, twist_deg=0.0, catch=0.0, c=0.0, stretch=0.0, swing_deg=(0.0, 0.0)):
    joints = {"swing_x": math.radians(swing_deg[0]), "swing_y": math.radians(swing_deg[1]), "slack": rise,
              "stretch": stretch, "holder": math.radians(twist_deg), "plunger_a": c, "plunger_b": c,
              "catch_a": catch, "catch_b": catch}
    return mk.state_from_poses(T.PENDANT_PART, (*PEND_ORIGIN, *Q0), joints=joints, source="test")


def bulb_at(pend, h_end, turn=15.0, lateral=0.0, tilt=0.0):
    """The bulb in the holder's frame: cap's end ``h_end`` over the mouth, turned ``turn`` from the J's leg."""
    rh, mouth = pend.link_pose("holder")
    pos = np.asarray(mouth) + rh @ np.array([lateral, 0.0, h_end])
    rot = rh @ rz(L.ENTRY_DEG + turn) @ rx(tilt)
    return mk.state_from_poses(T.BULB_PART, (*pos, *quat_of(rot)), source="test")


def bulb_world(pos, yaw_deg=L.BULB_START_YAW_DEG, tilt=0.0):
    rot = rz(yaw_deg) @ rx(tilt)
    return mk.state_from_poses(T.BULB_PART, (*pos, *quat_of(rot)), source="test")


def fixed():
    return {S.STAND: T.STAND_PART.rest_state(), S.RING: T.RING_PART.rest_state()}


def scene(h_end=L.H_LOCK_END, turn=15.0, catch=L.CATCH_TRAVEL, lateral=0.0, tilt=0.0, **pend):
    """Defaults: the bulb locked in the hooks of a holder hanging where it started (the 1.0 end state)."""
    p = pendant_state(catch=catch, c=max(0.0, h_end - L.H_PL), **pend)
    return {S.PENDANT: p, S.BULB: bulb_at(p, h_end, turn, lateral, tilt), **fixed()}


def start_scene():
    return {S.PENDANT: T.PENDANT_PART.rest_state(), S.BULB: T.BULB_PART.rest_state(), **fixed()}


def bulb_scene(pos, yaw_deg=L.BULB_START_YAW_DEG, tilt=0.0):
    """The holder hanging at rest and the bulb loose at ``pos`` (world)."""
    return {S.PENDANT: pendant_state(), S.BULB: bulb_world(pos, yaw_deg, tilt), **fixed()}


def arm(tcp, opening=1.0, approach=(0.0, 0.0, -1.0)):
    tcp = np.asarray(tcp, dtype=float)
    a = np.asarray(approach, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * a / np.linalg.norm(a), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def lifted_tracker():
    """A tracker that has seen the bulb lifted 20 mm out of its ring."""
    tr = S.Tracker()
    tr.update(start_scene(), FAR)
    tr.update(bulb_scene(S.BULB_START + np.array([0.0, 0.0, L.TOP + 0.020])), FAR)
    assert tr.lifted is not None
    return tr


class Geometry(unittest.TestCase):
    def test_published_and_estimated_sizes(self):
        self.assertAlmostEqual(2 * L.GLOBE_R, 0.060)  # A60: 60 mm across
        self.assertAlmostEqual(2 * L.CAP_R, 0.022)  # B22d: 22 mm cap
        self.assertAlmostEqual(L.BULB_LEN, 0.104)
        self.assertAlmostEqual(L.BULB_MASS, 0.040)
        lo, hi = T.BULB_PART.local_bounds()
        self.assertAlmostEqual(float(hi[2] - lo[2]), L.BULB_LEN, places=4)
        self.assertAlmostEqual(float(hi[1] - lo[1]), 2 * L.GLOBE_R, places=4)
        self.assertAlmostEqual(2 * L.PIN_OUT, 0.026)  # pin tips 26 mm apart
        self.assertAlmostEqual(L.MOUTH_Z, 0.178)
        self.assertAlmostEqual(L.HOLDER_TOP, 0.230)
        self.assertAlmostEqual(L.FLEX_LEN, 0.240)
        self.assertAlmostEqual(2 * L.HOLDER_R, 0.038)
        self.assertAlmostEqual(L.HANGING_MASS, 0.060)
        self.assertLess(L.HOLDER_MASS, 0.050)  # the acceptance check

    def test_the_cap_fits_the_bore_and_the_catches(self):
        self.assertAlmostEqual(L.GUIDE_IN - L.CAP_R, 0.00175)  # the collar leaves the cap's end 1.75 mm of play
        self.assertLess(L.GUIDE_IN, L.WALL_IN)
        self.assertLess(L.PIN_OUT, L.WALL_IN)  # pins clear the bore
        self.assertGreaterEqual(L.CATCH_R0, L.WALL_IN)  # at rest the catches sit back in the wall
        inner = L.CATCH_R0 - L.CATCH_TRAVEL
        self.assertLess(inner, L.PIN_OUT - 0.001)  # in: 1.5 mm under the pins
        self.assertGreater(inner, L.CAP_R)  # and clear of the cap
        self.assertGreater(L.LIP_C - L.LIP_W / 2, L.PIN_R)  # the pin sits between the lips
        self.assertLess(L.GUIDE_H0, L.H_PL)  # the collar's chamfer is under the plungers' tips
        pin_top = L.H_PL + 0.006 - L.PIN_FROM_END + L.PIN_R
        self.assertLess(pin_top, L.GUIDE_H0)  # and over the pins' path up to 6 mm up the plungers' travel

    def test_the_heights_of_the_j(self):
        """At the lock the pins rest on the hook floor; at C_HOOK they just clear the lips' tops; the J's leg tops out
        at the plungers' stop."""
        pin_low = lambda c: L.H_PL + c - L.PIN_FROM_END - L.PIN_R  # noqa: E731
        self.assertAlmostEqual(pin_low(L.C_LOCK), L.CATCH_H1)
        self.assertAlmostEqual(pin_low(L.C_HOOK), L.CATCH_H1 + L.LIP_H)
        self.assertGreater(pin_low(0.0), L.MOUTH_H)  # the pins are in the bore when the cap meets the plungers
        self.assertLess(L.H_PL + L.C_MAX + 0.001, L.CEIL_H + 0.001)
        self.assertAlmostEqual(L.H_LOCK_END, 0.027)
        self.assertGreaterEqual(L.LOCK_C[0], L.C_LOCK - 0.0005 - 1e-12)
        self.assertLess(L.LOCK_C[1], L.C_HOOK)

    def test_spring_estimates(self):
        self.assertAlmostEqual(L.plunger_force(0.0), 0.40)
        self.assertAlmostEqual(L.plunger_force(L.C_LOCK), 0.42)  # each, at the hooks
        self.assertAlmostEqual(L.plunger_force(L.C_MAX), 0.45)  # each, at full travel
        self.assertLessEqual(2 * L.plunger_force(L.C_MAX), 1.5)  # the acceptance check's push
        self.assertAlmostEqual(L.STRETCH_K * L.STRAIN_DZ, 9.0)
        self.assertAlmostEqual(L.TWIST_K * math.radians(1.0), 0.0005)
        # pushed from below with no hand on it, the holder rides up: the plungers' preload is more than its weight
        self.assertGreater(2 * L.plunger_force(0.0), L.HANGING_MASS * L.G)
        # the locked bulb hangs on the hooks: its weight is under the plungers' push at the lock (the lips hold it)
        self.assertLess(L.BULB_MASS * L.G, 2 * L.plunger_force(L.C_LOCK))

    def test_the_bulb_rests_in_its_ring(self):
        z0, z1, r = L.GLASS[-2]
        self.assertAlmostEqual(L.BULB_START_Z + z0, L.RING_TOP)  # the 38 mm slice sits on the ring's top
        self.assertGreater(r, L.RING_IN)
        self.assertLess(L.GLASS[-1][2], L.RING_IN)  # the globe's tip hangs inside the ring
        self.assertGreater(L.BULB_START_Z - L.BULB_LEN, L.RING_POST_TOP)  # and clear of the post's top
        self.assertAlmostEqual(L.BULB_START_YAW_DEG, L.ENTRY_DEG)

    def test_hinge_and_slide_signs(self):
        rest = pendant_state()
        _, m0 = rest.link_pose("holder")
        self.assertAlmostEqual(float(m0[2]), L.TOP + L.MOUTH_Z)
        _, up = pendant_state(rise=0.010).link_pose("holder")
        self.assertAlmostEqual(float(up[2] - m0[2]), 0.010)
        _, down = pendant_state(stretch=0.010).link_pose("holder")
        self.assertAlmostEqual(float(down[2] - m0[2]), -0.010)
        r, _ = pendant_state(twist_deg=30.0).link_pose("holder")
        self.assertAlmostEqual(math.degrees(math.atan2(r[1, 0], r[0, 0])), 30.0)
        _, sw = pendant_state(swing_deg=(0.0, 10.0)).link_pose("holder")
        self.assertGreater(abs(float(sw[0] - m0[0])), 0.040)  # a 10 degree swing moves the mouth 51 mm
        for name, side in (("catch_a", 1), ("catch_b", -1)):  # catch travel is inward
            out = rest.link_to_world(name, (side * L.CATCH_R0, 0.0, L.MOUTH_Z + L.CATCH_H1))
            inn = pendant_state(catch=L.CATCH_TRAVEL).link_to_world(name, (side * L.CATCH_R0, 0.0,
                                                                            L.MOUTH_Z + L.CATCH_H1))
            self.assertAlmostEqual(abs(float(inn[0] - L.PEND_X)), L.CATCH_R0 - L.CATCH_TRAVEL)
            self.assertAlmostEqual(abs(float(out[0] - L.PEND_X)), L.CATCH_R0)

    def test_the_locked_bulb_sits_on_the_catches_and_clears_the_bore(self):
        st = scene()
        p = st[S.PENDANT]
        b = st[S.BULB]
        pins = [b.link_to_world("base", (s * (L.PIN_OUT - 0.001), 0.0, -L.PIN_FROM_END - L.PIN_R)) for s in (1, -1)]
        d = p.distance(np.array(pins), bodies=S.CATCHES)
        self.assertLess(float(d.max()), 0.0006)  # each pin's underside on its hook floor
        cap = np.array([b.link_to_world("base", (L.CAP_R * math.cos(a), L.CAP_R * math.sin(a), -0.004))
                        for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)])
        self.assertGreater(float(p.distance(cap, bodies=("holder",)).min()), 0.0)

    def test_two_grips_are_90_mm_apart(self):
        hold = L.holder_grip_point()[2]
        lock = L.MOUTH_Z + L.H_LOCK_END - L.GRIP_FROM_END
        self.assertGreaterEqual(hold - lock, 0.090)
        self.assertGreaterEqual(hold - T.LEFT_PULL - T.PUSH_Z, 0.074 - 1e-9)  # as commanded at the push

    def test_the_stand_is_clear_of_the_pendant(self):
        st = T.STAND_PART.rest_state()
        p = pendant_state()
        lo, hi = p.link_bounds("holder")
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        self.assertGreater(float(st.distance(corners).min()), 0.100)
        self.assertAlmostEqual(L.BEAM_Z0, 0.500)
        self.assertLess(L.BEAM_X0, L.PEND_X - L.ROSE_R)  # the arm reaches past the rose

    def test_start_layout(self):
        st = start_scene()
        self.assertEqual(run(st).score, 0.0)
        r = S.read(st)
        self.assertFalse(S.locked(r))
        self.assertGreater(r.lateral, 0.10)
        self.assertAlmostEqual(r.twist_deg, 0.0)
        np.testing.assert_allclose(r.mouth, S.MOUTH_START, atol=1e-9)
        np.testing.assert_allclose(r.bulb_pos, S.BULB_START, atol=1e-9)
        self.assertLess(math.hypot(L.RING_X, L.RING_Y + 0.305), 0.32)  # well within the right arm's reach

    def test_grip_widths(self):
        self.assertLess(-T.HOLDER_GRIP * 2, 2 * L.HOLDER_R)
        self.assertLess(-T.BULB_GRIP * 2, 2 * L.GRIP_R)
        self.assertGreater(-T.OPEN * 2, 2 * L.GLOBE_R + 0.030)
        self.assertLess(-T.OPEN * 2, rigstate.FULL_GAP + 1e-9)


class BayonetRule(unittest.TestCase):
    def feed(self, steps, rule=None):
        rule = rule or S.Bayonet()
        out = None
        for c, turn in steps:
            out = rule.update(0.0, L.H_PL + c, 0.0, turn)
        return rule, out

    def test_up_turn_down_hooks_and_stays(self):
        seq = [(-0.006, 0.0), (0.0, 0.0), (0.002, 0.0), (0.004, 0.0)]
        seq += [(0.004, t) for t in np.linspace(0.0, 15.0, 10)]
        rule, hooked = self.feed(seq)
        self.assertTrue(hooked)
        rule, hooked = self.feed([(0.002, 15.0), (0.002, 14.0)], rule)
        self.assertTrue(hooked)

    def test_turning_at_the_bottom_is_forced(self):
        seq = [(0.0, 0.0), (0.001, 6.0), (0.001, 10.0), (0.004, 15.0)]
        rule, hooked = self.feed(seq)
        self.assertFalse(hooked)
        self.assertFalse(rule.ok)
        self.assertEqual(rule.missed, 1)

    def test_entering_off_the_leg_is_missed(self):
        rule, hooked = self.feed([(0.0, 10.0), (0.004, 10.0), (0.004, 15.0)])
        self.assertFalse(hooked)
        self.assertFalse(rule.ok)

    def test_turned_the_wrong_way_or_too_far_is_forced(self):
        rule, _ = self.feed([(0.004, 0.0), (0.004, -6.0)])
        self.assertFalse(rule.ok)
        rule, _ = self.feed([(0.004, 0.0), (0.004, 21.0)])
        self.assertFalse(rule.ok)

    def test_turned_back_out_of_the_hooks_unhooks(self):
        rule, _ = self.feed([(0.004, 0.0), (0.004, 15.0), (0.002, 15.0)])
        self.assertTrue(rule.hooked)
        rule, hooked = self.feed([(0.002, 8.0)], rule)
        self.assertFalse(hooked)

    def test_out_of_the_bore_resets(self):
        rule, _ = self.feed([(0.004, 0.0), (0.004, 15.0)])
        self.assertTrue(rule.hooked)
        self.assertFalse(rule.update(0.003, L.H_PL, 0.0, 15.0))  # 3 mm off the axis: out of the bore
        self.assertFalse(rule.inside or rule.ok)
        rule, hooked = self.feed([(0.0, 0.0), (0.004, 0.0), (0.004, 15.0)], rule)
        self.assertTrue(hooked)

    def test_unreadable_steps_change_nothing(self):
        rule, _ = self.feed([(0.004, 0.0), (0.004, 15.0)])
        self.assertTrue(rule.update(float("nan"), 0.0, 0.0, 0.0))
        self.assertTrue(rule.hooked)

    def test_tilted_or_low_is_not_in_the_bore(self):
        self.assertFalse(S.Bayonet.in_bore(0.0, L.H_PL, L.INSIDE_TILT_DEG + 1.0))
        self.assertFalse(S.Bayonet.in_bore(0.0, L.PIN_FROM_END - 0.001, 0.0))
        self.assertTrue(S.Bayonet.in_bore(0.0015, L.H_PL, 5.0))

    def test_the_turn_repeats_every_180_degrees(self):
        p = pendant_state()
        for turn in (15.0, 195.0, -165.0):
            r = S.read({S.PENDANT: p, S.BULB: bulb_at(p, L.H_LOCK_END, turn)})
            self.assertAlmostEqual(r.turn_deg, 15.0, places=6)
        self.assertAlmostEqual(S.wrap_half(90.0), -90.0)

    def test_the_turn_is_read_in_the_holders_frame(self):
        p = pendant_state(twist_deg=40.0)
        r = S.read({S.PENDANT: p, S.BULB: bulb_at(p, L.H_LOCK_END, 15.0)})
        self.assertAlmostEqual(r.turn_deg, 15.0, places=6)
        self.assertAlmostEqual(r.twist_deg, 40.0, places=6)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["locked"])

    def test_pins_in_the_leg_are_not_locked(self):
        self.assertEqual(run(scene(turn=0.0, catch=0.0)).score, 0.0)
        self.assertEqual(run(scene(turn=0.0)).score, 0.0)  # catches in but not turned: not locked

    def test_catches_out_is_not_locked(self):
        self.assertEqual(run(scene(catch=0.0)).score, 0.0)
        self.assertEqual(run(scene(catch=0.79 * L.CATCH_TRAVEL)).score, 0.0)
        self.assertEqual(run(scene(catch=0.81 * L.CATCH_TRAVEL)).score, 1.0)

    def test_lock_window_edges(self):
        for turn, want in ((12.5, 0.0), (13.5, 1.0), (16.5, 1.0), (17.5, 0.0)):
            self.assertEqual(run(scene(turn=turn)).score, want, turn)
        for c, want in ((0.0012, 0.0), (0.0017, 1.0), (0.0024, 1.0), (0.0028, 0.0)):
            self.assertEqual(run(scene(h_end=L.H_PL + c)).score, want, c)
        self.assertEqual(run(scene(tilt=4.0)).score, 1.0)
        self.assertEqual(run(scene(tilt=6.0)).score, 0.0)
        self.assertEqual(run(scene(lateral=0.001)).score, 1.0)
        self.assertEqual(run(scene(lateral=0.002)).score, 0.0)

    def test_a_held_bulb_or_holder_is_half(self):
        p = pendant_state(catch=L.CATCH_TRAVEL, c=L.C_LOCK)
        mouth = p.link_pose("holder")[1]
        on_holder = {"left": arm(mouth + np.array([0.0, 0.0, 0.040]), 0.3, (0.5, -0.8, -0.6)),
                     "right": FAR["right"]}
        self.assertEqual(run(scene(), on_holder).score, 0.5)
        under = mouth + np.array([0.0, 0.0, L.H_LOCK_END - L.GRIP_FROM_END])
        on_bulb = {"left": FAR["left"], "right": arm(under, 0.55, (0.1, 0.9, -0.3))}
        self.assertEqual(run(scene(), on_bulb).score, 0.5)

    def test_released_needs_2_mm(self):
        """The left's finger zone just off the holder's side: closed fingers pointing at it from 30 mm out."""
        p = pendant_state()
        mouth = np.asarray(p.link_pose("holder")[1])
        r0 = rigstate.zone_radius(0.0)
        for gap, want in ((0.001, 0.5), (0.003, 1.0)):
            tip = mouth + np.array([0.0, L.HOLDER_R + r0 + gap, 0.030])
            arms = {"left": rigstate.Arm(tip + np.array([0.0, 0.013, 0.0]), tip, 0.0), "right": FAR["right"]}
            self.assertEqual(run(scene(), arms).score, want, gap)

    def test_holder_moved_or_swinging_is_half(self):
        self.assertEqual(run(scene(swing_deg=(0.0, 7.0))).score, 0.5)  # 36 mm off where it hung
        self.assertEqual(run(scene(swing_deg=(0.0, 5.0))).score, 1.0)  # 26 mm
        self.assertEqual(run(scene(rise=0.016)).score, 0.5)  # pushed up 16 mm and left there
        tr = S.Tracker()
        for i in range(WIN + 2):
            res = tr.update(scene(swing_deg=(0.0, 2.5 if i % 2 else 0.0)), FAR)
        self.assertEqual(res.score, 0.5)  # swinging 13 mm

    def test_possible_scores(self):
        self.assertEqual({S.tier(*bits) for bits in np.ndindex(2, 2, 2, 2, 2, 2)}, {0.0, 0.25, 0.5, 1.0})

    def test_tier_function_table(self):
        self.assertEqual(S.tier(False, True, False, False, True, True), 1.0)
        self.assertEqual(S.tier(False, True, False, False, False, True), 0.5)
        self.assertEqual(S.tier(False, True, False, False, True, False), 0.5)
        self.assertEqual(S.tier(False, True, False, True, True, True), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, True), 0.25)
        self.assertEqual(S.tier(False, False, False, False, True, True), 0.0)
        self.assertEqual(S.tier(True, True, False, False, True, True), 0.0)


class Latches(unittest.TestCase):
    def test_pulled_down_16_mm_latches_strained(self):
        tr = S.Tracker()
        tr.update(scene(stretch=0.016), FAR)
        self.assertIsNotNone(tr.latch["strained"])
        self.assertEqual(run(scene(), tracker=tr).score, 0.25)
        tr = S.Tracker()
        tr.update(scene(stretch=0.014), FAR)
        self.assertIsNone(tr.latch["strained"])
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_turned_past_90_latches_wound(self):
        tr = S.Tracker()
        tr.update(scene(twist_deg=95.0), FAR)
        self.assertIsNotNone(tr.latch["wound"])
        self.assertEqual(run(scene(), tracker=tr).score, 0.5)
        tr = S.Tracker()
        tr.update(scene(twist_deg=-85.0), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_bulb_on_the_table_after_lifting_is_dropped(self):
        tr = lifted_tracker()
        tr.update(bulb_scene((0.30, -0.10, L.TOP + L.BULB_LEN + 0.001), yaw_deg=0.0), FAR)
        self.assertEqual(tr.why.get("dropped"), "bulb on the table")
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_bulb_lying_on_the_table_is_dropped(self):
        tr = lifted_tracker()
        tr.update(bulb_scene((0.30, -0.10, L.TOP + L.GLOBE_R + 0.001), tilt=90.0), FAR)
        self.assertTrue(tr.dropped)

    def test_bulb_put_back_on_its_ring_is_dropped(self):
        tr = lifted_tracker()
        for dz in np.linspace(0.020, 0.0, 6):
            tr.update(bulb_scene(S.BULB_START + np.array([0.0, 0.0, L.TOP + dz])), FAR)
        self.assertTrue(tr.dropped)
        self.assertIn("bulb_stand", tr.why["dropped"])
        self.assertEqual(tr.resets, 0)

    def test_globe_against_the_ceiling_stand_is_dropped(self):
        tr = lifted_tracker()
        x = L.POST_X - L.POST_W / 2 - L.GLOBE_R - 0.0005
        tr.update(bulb_scene((x, 0.0, L.TOP + 0.300 - L.GLOBE_C)), FAR)
        self.assertIn("ceiling_stand", tr.why["dropped"])

    def test_nothing_drops_before_the_bulb_leaves_its_ring(self):
        tr = S.Tracker()
        run(start_scene(), tracker=tr)
        self.assertIsNone(tr.lifted)
        self.assertFalse(tr.dropped)

    def test_bulb_carried_and_pushed_in_is_not_dropped(self):
        tr = lifted_tracker()
        for h in np.linspace(-0.030, L.H_PL + L.C_MAX, 12):
            tr.update(scene(h_end=h, turn=0.0, catch=0.0), FAR)
        self.assertFalse(tr.dropped)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        tr = S.Tracker()
        self.assertEqual(tr.update(None, FAR).score, 0.0)
        self.assertEqual(tr.update({S.PENDANT: pendant_state()}, FAR).score, 0.0)
        st = scene()
        st[S.BULB] = bulb_world((float("nan"), 0.0, 1.0))
        self.assertEqual(tr.update(st, FAR).score, 0.0)
        st = scene()
        st[S.PENDANT] = pendant_state(twist_deg=float("nan"))
        res = tr.update(st, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])

    def test_missing_stands_are_skipped(self):
        st = scene()
        del st[S.STAND], st[S.RING]
        self.assertEqual(run(st).score, 1.0)

    def test_a_bad_step_restarts_the_window(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        tr.update(None, FAR)
        self.assertNotEqual(tr.update(scene(), FAR).score, 1.0)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_unreadable_arms_never_hold_1(self):
        tr = S.Tracker()
        for _ in range(WIN + 3):
            res = tr.update(scene(), None)
        self.assertNotEqual(res.score, 1.0)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)
        bad = {"left": FAR["left"], "right": rigstate.Arm(np.array([np.nan, 0, 1.0]), np.array([0, 0, 1.0]), 0.5)}
        self.assertNotEqual(tr.update(scene(), bad).score, 1.0)

    def test_distance_failure_counts_as_not_released(self):
        class Broken:
            def distance(self, *a, **k):
                raise RuntimeError("boom")
        st = scene()
        st[S.BULB] = Broken()
        self.assertFalse(S.Tracker()._released(st, FAR))

    def test_reset_back_on_the_marks_clears_the_latches(self):
        tr = lifted_tracker()
        tr.update(scene(stretch=0.020, twist_deg=100.0), FAR)
        tr.update(bulb_scene((0.30, -0.10, L.TOP + L.BULB_LEN)), FAR)
        self.assertTrue(tr.dropped)
        self.assertIsNotNone(tr.latch["strained"])
        tr.update(start_scene(), FAR)  # everything jumps back to its start: a reset
        self.assertFalse(tr.dropped)
        self.assertEqual(tr.latch, {"dropped": None, "strained": None, "wound": None})
        self.assertEqual(tr.resets, 1)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)

    def test_bulb_just_pushed_up_into_the_bore_is_zero(self):
        self.assertEqual(run(scene(h_end=L.H_PL + L.C_MAX, turn=0.0, catch=0.0)).score, 0.0)

    def test_bulb_balanced_in_the_mouth_turned_but_not_hooked_is_zero(self):
        self.assertEqual(run(scene(h_end=L.H_PL + L.C_LOCK, turn=15.0, catch=0.0)).score, 0.0)

    def test_holder_wound_to_get_the_turn_is_half(self):
        """The bulb pushed in and the holder turned (not the bulb): the flex wound past 90 degrees."""
        tr = S.Tracker()
        for tw in np.linspace(0.0, 120.0, 10):
            tr.update(scene(twist_deg=tw), FAR)
        for tw in np.linspace(120.0, 0.0, 10):
            tr.update(scene(twist_deg=tw), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 0.5)

    def test_holder_pulled_down_to_push_against_is_quarter(self):
        tr = S.Tracker()
        tr.update(scene(stretch=L.STRETCH_MAX), FAR)
        self.assertEqual(run(scene(), tracker=tr).score, 0.25)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    @staticmethod
    def right_on_bulb(m, opening):
        tcp = m.bulb - np.array([0.0, 0.0, L.GRIP_FROM_END])
        a = T.approach(T.HEAD_R0, T.PITCH_R)
        return rigstate.Arm(tcp, tcp + 0.013 * a, opening)

    @staticmethod
    def left_on_holder(m, opening):
        tcp = m.mouth() + np.array([0.0, 0.0, 0.040])
        return rigstate.Arm(tcp, tcp + 0.013 * T.A_L, opening)

    def carry_to(self, m, z_end, held, steps=40):
        """Grip the bulb on its stand and carry it straight under the holder and up to ``z_end`` (cap's end over
        the table)."""
        close = -2 * T.BULB_GRIP / rigstate.FULL_GAP
        hold = -2 * T.HOLDER_GRIP / rigstate.FULL_GAP
        left = (lambda: self.left_on_holder(m, hold)) if held else (lambda: FAR["left"])
        m.step({"left": left(), "right": self.right_on_bulb(m, close)})
        self.assertIsNotNone(m.grip)
        start = m.bulb.copy()
        under = np.array([L.PEND_X, L.PEND_Y, L.TOP + L.MOUTH_Z - 0.010])
        for p in np.linspace(start, under, steps)[1:]:
            tcp = p - m.grip[0]
            m.step({"left": left(), "right": rigstate.Arm(tcp, tcp + 0.013 * T.approach(T.HEAD_R0, T.PITCH_R),
                                                          close)})
        for z in np.linspace(under[2], L.TOP + z_end, steps)[1:]:
            tcp = np.array([L.PEND_X, L.PEND_Y, z]) - m.grip[0]
            m.step({"left": left(), "right": rigstate.Arm(tcp, tcp + 0.013 * T.approach(T.HEAD_R0, T.PITCH_R),
                                                          close)})
        return close, left

    def test_a_held_holder_stops_the_bulb_at_the_plungers_stop(self):
        m = T.OfflineModel()
        self.carry_to(m, L.MOUTH_Z + 0.040, held=True)
        self.assertTrue(m.held)
        self.assertAlmostEqual(m.rise, 0.0)
        self.assertAlmostEqual(m._c(), L.C_MAX)

    def test_a_free_holder_rides_up_on_the_bulb(self):
        m = T.OfflineModel()
        self.carry_to(m, L.MOUTH_Z + 0.040, held=False)
        self.assertFalse(m.held)
        self.assertGreater(m.rise, 0.010)
        self.assertLess(m._c(), L.C_HOOK)

    def test_let_go_unhooked_the_bulb_falls(self):
        m = T.OfflineModel()
        self.carry_to(m, L.MOUTH_Z + L.H_PL + L.C_MAX, held=True)
        m.step({"left": FAR["left"], "right": FAR["right"]})
        self.assertAlmostEqual(float(m.bulb[2]), L.TOP + L.BULB_LEN)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertEqual(set(d), {"left", "right"})
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertEqual(set(T.demo()), {"left", "right"})
            finally:
                os.environ.pop("DEMO_VARIANT")
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for side, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = side
            try:
                d = T.demo()
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
            self.assertNotIn(side, d)
            self.assertIn(other, d)

    def test_the_holder_is_held_until_the_bulb_is_let_go(self):
        p = T.plans()
        held = T._until(p["left"], "hold it steady")
        self.assertGreaterEqual(held, T._until(p["right"], "let go of the bulb"))
        self.assertLess(T._until(p["left"], "grip the holder"), T._until(p["right"], "up into the mouth") - 1.5)

    def test_the_turn_is_15_degrees_about_the_bulbs_axis(self):
        self.assertAlmostEqual(T.HEAD_R1 - T.HEAD_R0, L.HOOK_DEG)
        a0, a1 = T.approach(T.HEAD_R0, T.PITCH_R), T.approach(T.HEAD_R1, T.PITCH_R)
        self.assertAlmostEqual(a0[2], a1[2])

    def test_demo_length_is_well_within_the_run_limit(self):
        limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0]) if (
            HERE / "task.json").exists() else 800
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps + 3 * HZ, 0.65 * limit)  # the plan, plus 3 s of standing still
        self.assertGreater(steps, 250)


# -- clearance of the cap's path (GPU run 1: the cap met the old 23 mm collar at the plungers' rest and stopped) -----
def _sdf(shape, pts):
    """Signed distance (m) from base-frame points to one box or cylinder shape (negative inside)."""
    q = (pts - np.asarray(shape.center, dtype=float)) @ shape.rotation()
    if shape.kind == "box":
        d = np.abs(q) - np.asarray(shape.size, dtype=float) / 2
    else:
        i = "XYZ".index(shape.axis)
        d = np.stack([np.linalg.norm(np.delete(q, i, axis=1), axis=1) - shape.radius,
                      np.abs(q[:, i]) - shape.height / 2], axis=1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def _cap_points(step=0.0005):
    """Surface points of the cap and its pins (bulb frame), 0.5 mm apart or closer."""
    a = np.linspace(0.0, 2 * math.pi, 144, endpoint=False)
    ring = lambda r, z: np.stack([r * np.cos(a), r * np.sin(a), np.full(a.size, z)], axis=1)  # noqa: E731
    pts = [ring(L.CAP_R, z) for z in np.arange(-L.CAP_LEN, 1e-9, step)]
    pts += [ring(r, 0.0) for r in np.arange(0.0, L.CAP_R, 2 * step)]
    b = np.linspace(0.0, 2 * math.pi, 36, endpoint=False)
    for s in (1.0, -1.0):
        for x in np.arange(L.CAP_R, L.PIN_OUT + 1e-9, step):
            pts.append(np.stack([np.full(b.size, s * x), L.PIN_R * np.cos(b), -L.PIN_FROM_END + L.PIN_R * np.sin(b)],
                                axis=1))
        for r in np.linspace(0.0, L.PIN_R, 5):
            pts.append(np.stack([np.full(b.size, s * L.PIN_OUT), r * np.cos(b), -L.PIN_FROM_END + r * np.sin(b)],
                                axis=1))
    return np.concatenate(pts)


CAP_PTS = _cap_points()


def holder_rigid(catch=0.0, body=None):
    """Every rigid shape of the holder (body, hook floors and lips; not the sprung plungers), base frame, the catches
    ``catch`` in."""
    out = list(P.holder_shapes() if body is None else body)
    for side in (1, -1):
        for sh in P.catch_shapes(side):
            out.append(mk.box((sh.center[0] - side * catch, sh.center[1], sh.center[2]), sh.size, quat=sh.quat))
    return out


def _boxes_sdf_min(shapes, pts):
    """The least signed distance from ``pts`` to any of the box ``shapes``, all at once."""
    cen = np.array([sh.center for sh in shapes], dtype=float)
    rot = np.array([sh.rotation() for sh in shapes], dtype=float)
    half = np.array([sh.size for sh in shapes], dtype=float)[:, None, :] / 2
    d = np.abs(np.matmul(pts[None, :, :] - cen[:, None, :], rot)) - half
    out = np.maximum(d, 0.0)
    return float((np.sqrt(np.einsum("npk,npk->np", out, out)) + np.minimum(d.max(axis=2), 0.0)).min())


def cap_clearance(shapes, c, lat=(0.0, 0.0), tilt=0.0, tilt_dir=0.0, turn=0.0):
    """Least distance (m) from the cap and pins to ``shapes``: the cap's end ``c`` over the plungers' rest,
    ``lat`` (x, y) off the holder's axis, tilted ``tilt`` degrees toward ``tilt_dir``, turned ``turn`` from the leg."""
    rot = rz(math.degrees(tilt_dir)) @ rx(tilt) @ rz(-math.degrees(tilt_dir)) @ rz(L.ENTRY_DEG + turn)
    pts = CAP_PTS @ rot.T + np.array([lat[0], lat[1], L.MOUTH_Z + L.H_PL + c])
    boxes = [sh for sh in shapes if sh.kind == "box"]
    others = [float(_sdf(sh, pts).min()) for sh in shapes if sh.kind != "box"]
    return min([_boxes_sdf_min(boxes, pts)] + others)


def old_collar_body():
    """The holder as built for GPU run 1: a 23 mm collar whose chamfer (22.5 to 25 mm) ended at the plungers' tips,
    the roof at 31 mm."""
    m, n = L.MOUTH_Z, L.WALL_SEGS
    out = P._ring(L.WALL_IN, L.HOLDER_R, m + L.MOUTH_H, m + 0.031, n, L.WHITE)
    out += P._funnel(L.HOLDER_R, m, L.HOLDER_R - L.WALL_IN, n, L.WHITE)
    out += P._funnel(L.WALL_IN, m + 0.0225, L.WALL_IN - 0.0115, n, L.WHITE)
    out += P._ring(0.0115, L.WALL_IN, m + 0.025, m + 0.031, n, L.WHITE)
    out.append(P._cyl(m + 0.031, m + L.HOLDER_LEN, L.HOLDER_R, L.WHITE))
    return out


# How far off the cap may come and still go up clear: 0.9 mm any way (the pins' tips have 1 mm to the 28 mm bore),
# 1.5 mm across the pins, or tilted 2 degrees. GPU run 1 had it 0.6 mm off and 1.4 degrees tilted at the plungers.
CS = [i * 0.0005 for i in range(13)]  # c from 0 to 6 mm


def _poses():
    across = math.radians(L.ENTRY_DEG) + math.pi / 2
    yield {}
    for k in range(8):
        a = k * math.pi / 4
        yield {"lat": (0.0009 * math.cos(a), 0.0009 * math.sin(a))}
        if k % 2 == 0:
            yield {"tilt": 2.0, "tilt_dir": a}
    for s in (1, -1):
        yield {"lat": (s * 0.0015 * math.cos(across), s * 0.0015 * math.sin(across))}


class Clearance(unittest.TestCase):
    def test_the_cap_rises_6_mm_against_the_plungers_alone(self):
        """From the plungers' rest to 6 mm up (their stop is at 5), at the J's leg, nothing rigid on the holder
        touches the cap or its pins: the push goes into the plungers' springs, not into the holder."""
        shapes = holder_rigid()
        for pose in _poses():
            for c in CS:
                self.assertGreater(cap_clearance(shapes, c, **pose), 0.0001, (c, pose))

    def test_raised_past_the_lips_the_pins_turn_clear(self):
        """From C_HOOK up, the pins turn from the leg to the hooks clear of the holder (catches out), and with the
        catches in, the hook floors and lips pass under them."""
        out, inn = holder_rigid(), holder_rigid(catch=L.CATCH_TRAVEL)
        for c in [x for x in CS if x >= L.C_HOOK - 1e-9]:
            for turn in (5.0, 10.0, 15.0, 17.0):
                self.assertGreater(cap_clearance(out, c, turn=turn), 0.0001, (c, turn))
            for turn in (13.0, 15.0, 17.0):
                self.assertGreater(cap_clearance(inn, c, turn=turn), 0.0, (c, turn))

    def test_nothing_rigid_in_the_caps_path(self):
        """The volume the cap sweeps from the plungers' rest to 6 mm up, 1.5 mm wider than the cap all round, is
        free of the holder's rigid shapes (the plungers are springs, not counted)."""
        r = np.linspace(0.0, L.CAP_R + 0.0015, 12)
        a = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
        h = np.linspace(L.H_PL - L.CAP_LEN, L.H_PL + 0.006, 49)
        R, A, H = np.meshgrid(r, a, h, indexing="ij")
        pts = np.stack([(R * np.cos(A)).ravel(), (R * np.sin(A)).ravel(), (L.MOUTH_Z + H).ravel()], axis=1)
        for sh in holder_rigid():
            self.assertGreater(float(_sdf(sh, pts).min()), 0.0, sh)

    def test_the_old_collar_fails_this_check(self):
        """The holder of GPU run 1 (23 mm collar) blocks the cap as the GPU saw it (0.6 mm off, 1.4 degrees tilted):
        the cap cannot rise 0.5 mm past the plungers' rest whichever way it is off."""
        old = holder_rigid(body=old_collar_body())
        for k in range(8):
            a = k * math.pi / 4
            lat = (0.0006 * math.cos(a), 0.0006 * math.sin(a))
            self.assertLess(cap_clearance(old, 0.0005, lat=lat), 0.0, k)
            self.assertLess(cap_clearance(old, 0.0005, lat=lat, tilt=1.4, tilt_dir=a), 0.0, k)
        self.assertLess(cap_clearance(old, 0.006), 0.0)  # and its roof stopped the cap at 6 mm

    def test_the_plungers_are_free_to_rise(self):
        """Each plunger slides up the holder's axis from 0 to its 5 mm stop on its spring alone (links of one
        assembly never collide), driven 40 mm below its rest for the preload."""
        for j in T.PLUNGERS:
            self.assertEqual((j.kind, j.axis, j.parent), ("prismatic", "Z", "holder"))
            self.assertEqual((j.lower, j.upper, j.spring), (0.0, L.PL_TRAVEL, L.PL_K))
        self.assertGreaterEqual(L.PL_TRAVEL, 0.005)
        self.assertGreaterEqual(L.C_MAX - L.C_HOOK, 0.002)  # the pins rise 2 mm past the lips before the stop
        self.assertGreater(L.CEIL_H, L.H_PL + 0.006)  # the roof 1 mm over the cap at 6 mm



# -- the arms' give under the push (GPU runs 1 and 2: the cap never got 0.5 mm up the plungers) --------------------
# Quasi-static model. Each arm is a spring at its TCP (the YAM's joint PD: j1-3 80 N m/rad, j4-6 10 N m/rad; the
# payload is not gravity compensated). The right carries the bulb (40 g) and is commanded ``over`` past the cap's end
# meeting the plungers; the cap pushes the plungers (each k (preload + c) plus its 5 g link) up to their stop, then
# the holder. The holder hangs on the stretch spring (600 N/m, 0.9 mm out under its 0.54 N, GPU settle) and above
# that on the free slack slide (40 mm); the left holds it as a spring anchored ``pull`` under where it hung. Units SI.
W_HANG = 0.54  # N on the stretch spring (GPU settle: stretch 0.8 to 0.9 mm at 600 N/m)
ARM_CAP = 20.0  # N: far above anything here (the wrist's 7 N m at a 0.35 m lever)


def push_balance(over, k_right, k_left, pl_k=None, pl_pre=None, pull=0.0, rigid=False, travel=None):
    """(c, rise, force): the cap's end up the plungers' travel, the holder's rise from where it hung (negative:
    down) and the push (N), at equilibrium. ``k_left`` 0: no hand on the holder. ``rigid``: the cap stopped at c = 0
    by something rigid on the holder (GPU run 1's collar)."""
    pl_k = L.PL_K if pl_k is None else pl_k
    pl_pre = L.PL_PRELOAD if pl_pre is None else pl_pre
    travel = L.PL_TRAVEL if travel is None else travel
    e0 = W_HANG / L.STRETCH_K

    def holder(x):  # the push that holds the holder at rise x
        h = W_HANG - L.STRETCH_K * max(0.0, e0 - x) + 0.5 * max(0.0, x - e0) + k_left * (x + pull)
        return h + (1e6 * (x - L.SLACK_UP) if x > L.SLACK_UP else 0.0)

    def plungers(c):
        return 2 * (pl_k * (pl_pre + c) + L.LINK_MASS * L.G)

    def bisect(f, lo, hi):  # f increasing; the root
        for _ in range(100):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if f(m) < 0 else (lo, m)
        return (lo + hi) / 2

    def rise(F):
        return bisect(lambda x: holder(x) - F, -0.05, 0.2)

    def c_of(F):
        if rigid or F <= plungers(0.0):
            return 0.0
        if F >= plungers(travel):
            return travel
        return bisect(lambda c: plungers(c) - F, 0.0, travel)

    def short(F):  # the push the right arm makes at this force's c and rise, less F
        return k_right * (over - c_of(F) - rise(F)) - L.BULB_MASS * L.G - F

    F = ARM_CAP if short(ARM_CAP) > 0 else bisect(lambda f: -short(f), 0.0, ARM_CAP)
    return c_of(F), rise(F), F


OLD = dict(pl_k=500.0, pl_pre=0.001)  # GPU runs 1 and 2: 0.5 N each at rest, 1.5 N at the hooks
K_GPU = ((200.0, 165.0), (150.0, 205.0))  # (right, left) N/m fitted to GPU run 2 (in the turn; at the hold)
K_GRID = (120.0, 150.0, 200.0, 250.0, 300.0)


class ArmGive(unittest.TestCase):
    def test_the_model_reproduces_gpu_run_2(self):
        """Run 2 (500 N/m plungers, commanded 15 mm over): the cap's end 0.05 to 0.31 mm up (plungers 0.43 mm at
        most), the holder 3.8 mm up at the hold and 6.0 mm in the turn, the right 11.5 then 9.1 mm under its
        command."""
        for kr, kl in K_GPU:
            c, x, F = push_balance(0.015, kr, kl, **OLD)
            self.assertLess(c, 0.0005, (kr, kl))
            self.assertTrue(0.003 <= x <= 0.0065, (kr, kl, x))
            self.assertTrue(0.008 <= 0.015 - c - x <= 0.012, (kr, kl))
            self.assertTrue(1.2 <= F <= 1.6, (kr, kl, F))

    def test_the_model_reproduces_gpu_run_1(self):
        """Run 1 (the cap stopped on the old collar at c = 0, commanded 12 mm over): the holder 1.8 mm up at the
        hold and 4.6 mm in the turn, the right 10.5 then 7.7 mm under its command."""
        for kr, kl in K_GPU:
            c, x, _ = push_balance(0.012, kr, kl, rigid=True, **OLD)
            self.assertEqual(c, 0.0)
            self.assertTrue(0.002 <= x <= 0.005, (kr, kl, x))
            self.assertTrue(0.007 <= 0.012 - x <= 0.011, (kr, kl))

    def test_the_push_reaches_the_stop_with_soft_arms(self):
        """Both arms anywhere from 120 to 300 N/m: the cap's end reaches the plungers' 5 mm stop, the holder stays
        within 1.5 mm down and 5.5 mm up of where it hung (strained is 15 mm down), the push under 3.5 N."""
        over = T.PUSH_Z - T.CONTACT_Z
        for kr in K_GRID:
            for kl in K_GRID:
                c, x, F = push_balance(over, kr, kl, pull=T.LEFT_PULL)
                self.assertGreaterEqual(c, L.C_MAX - 1e-6, (kr, kl, c))
                self.assertTrue(-0.0015 <= x <= 0.0055, (kr, kl, x))
                self.assertLess(F, 3.5, (kr, kl))
        c, _, _ = push_balance(over, 100.0, 120.0, pull=T.LEFT_PULL)  # softer than either run showed
        self.assertGreaterEqual(c, 0.004)
        for kr, kl in K_GPU:
            c, x, _ = push_balance(over, kr, kl, pull=T.LEFT_PULL)
            self.assertGreaterEqual(c, L.C_MAX - 1e-6)
            self.assertLess(x, 0.002)

    def test_the_old_plungers_would_still_fail(self):
        """With run 2's 500 N/m plungers this demo would get the cap's end about 1 mm up: under C_HOOK."""
        over = T.PUSH_Z - T.CONTACT_Z
        for kr, kl in K_GPU:
            c, _, _ = push_balance(over, kr, kl, pull=T.LEFT_PULL, **OLD)
            self.assertLess(c, L.C_HOOK - 0.001, (kr, kl))

    def test_with_no_hand_on_the_holder_it_rides_up(self):
        """Left idle: the plungers' preload (0.9 N with their links) is over the holder's hanging weight, so the
        holder slides up the slack instead and the cap's end never leaves the plungers' rest."""
        over = T.PUSH_Z - T.CONTACT_Z
        self.assertGreater(2 * (L.plunger_force(0.0) + L.LINK_MASS * L.G), W_HANG + 0.3)
        for kr in (100.0,) + K_GRID:
            c, x, _ = push_balance(over, kr, 0.0)
            self.assertEqual(c, 0.0, kr)
            self.assertTrue(0.005 < x < L.SLACK_UP, (kr, x))

    def test_the_left_pull_moves_the_holder_little(self):
        """At rest the left's 7 mm (commanded) pull takes the holder 1.2 to 2.3 mm down (strained is 15 mm)."""
        for kl in K_GRID:
            sag = kl * T.LEFT_PULL / (L.STRETCH_K + kl)
            self.assertLess(sag, 0.0025)
            _, x, _ = push_balance(-0.05, 200.0, kl, pull=T.LEFT_PULL)  # the bulb nowhere near
            self.assertAlmostEqual(-x, sag, delta=1e-5)

    def test_the_plan_pulls_while_the_bulb_goes_in_and_eases_off_before_letting_go(self):
        p = T.plans()
        pull = T._until(p["left"], "draw it down")
        self.assertLess(pull, T._until(p["right"], "up into the mouth") - 3.0)
        labels = [w.label for w in p["left"]]
        i = labels.index("ease the pull off")
        self.assertEqual(labels[i + 1], "let go of the holder")
        self.assertAlmostEqual(p["left"][i - 1].pos[2] + T.LEFT_PULL, p["left"][i].pos[2])
        self.assertGreaterEqual(T._until(p["left"], "hold it steady"), T._until(p["right"], "let go of the bulb"))


class Folder(unittest.TestCase):
    def test_text_has_no_em_dashes_or_identifying_names(self):
        bad = re.compile(chr(0x2014) + "|" + chr(0x2013) + "|T-70|VihanAggarwal|Claude|Anthropic|co-author", re.I)
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "parts.py"):
            p = HERE / name
            if p.exists():
                self.assertIsNone(bad.search(p.read_text()), name)

    def test_task_json_matches_the_code(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        d = json.loads(p.read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["category"], "Societal relatability")
        text = d["setup"] + d["scoring"]
        for needle in ("x 450 mm", "500 mm", "x 260 mm", "75 mm", "240 mm", "178 mm", "38 mm", "52 mm", "30 g",
                       "28 mm", "25.5 mm", "5 mm", "0.4 N", "0.45 N", "1.5 N", "15 degrees", "2 mm", "60 mm", "104 mm", "40 g",
                       "50 mm", "x 250 mm, y -130 mm", "13 to 17 degrees", "0.5 mm", "5 degrees", "80%",
                       "within 2 mm", "10 mm", "30 mm", "15 mm", "90 degrees", "1 s", "0.25", "0.5", "1.0"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "800")
        self.assertIn("0.0005 N m per degree", d["setup"])
        self.assertIn("600 N/m", d["fidelity"] + d["setup"])

    def test_numbers_in_the_text_match_layout(self):
        self.assertEqual((L.POST_X, L.BEAM_Z0, L.BEAM_X0), (0.450, 0.500, 0.260))
        self.assertEqual((L.RING_X, L.RING_Y, L.RING_TOP), (0.250, -0.130, 0.050))
        self.assertEqual((2 * L.WALL_IN, 2 * L.GUIDE_IN, L.PL_TRAVEL), (0.028, 0.0255, 0.005))
        self.assertEqual((L.LOCK_TURN, L.LOCK_TILT_DEG, L.LOCK_CATCH), ((13.0, 17.0), 5.0, 0.8))
        self.assertEqual((L.RELEASE_GAP, L.HANG_MOVE, L.HANG_PLAN, L.HANG_DZ), (0.002, 0.010, 0.030, 0.015))
        self.assertEqual((L.STRAIN_DZ, L.WOUND_DEG, L.LEFT_STAND, L.DROP_TABLE, L.DROP_TOUCH),
                         (0.015, 90.0, 0.008, 0.002, 0.001))
        self.assertEqual((L.STRETCH_K, L.TWIST_K_PER_DEG, L.PL_K, L.PL_PRELOAD), (600.0, 0.0005, 10.0, 0.040))
        self.assertEqual(L.HOLDER_FRICTION[0], 0.5)
        self.assertEqual(L.BULB_FRICTION[0], 0.8)

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)

    def test_parts_build(self):
        for fn in (P.stand_shapes, P.rose_shapes, P.holder_shapes, P.bulb_shapes, P.ring_stand_shapes):
            self.assertTrue(fn())
        self.assertEqual(len(P.catch_shapes(1)), 3)


if __name__ == "__main__":
    unittest.main()
