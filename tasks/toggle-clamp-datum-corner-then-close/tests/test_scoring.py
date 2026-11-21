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
SEAT_Z = L.PLATE_H + L.CS_H / 2


def casting_at(x=L.SEAT_X, y=L.SEAT_Y, z=SEAT_Z, yaw=0.0, pitch=0.0, roll=0.0):
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.CASTING_PART, (x, y, L.TOP + z, *q), source="test")


def clamp_at(name, qh, qa=None):
    """A clamp with its handle at ``qh``; the arm where the linkage puts it, but not below flat (the pad resting on a
    seated casting's top), unless ``qa`` is given; the link on the arm's tail."""
    c = L.C1 if name == S.CLAMP1 else L.C2
    if qa is None and math.isfinite(qh):
        qa = min(c.arm_target(qh), L.QA_FLAT)
    link = c.link_joint(qh) if math.isfinite(qh) else 0.0
    return mk.state_from_poses(T.SPECS[name], (c.px, c.py, L.TOP, 0.0, 0.0, 0.0, 1.0),
                               joints={S.HANDLE: qh, S.ARM: qa if qa is not None else 0.0, S.LINK: link},
                               source="test")


def scene(q1=L.QH_UPPER, q2=L.QH_UPPER, cs=None, qa1=None, qa2=None):
    return {S.CASTING: cs if cs is not None else casting_at(), S.CLAMP1: clamp_at(S.CLAMP1, q1, qa1),
            S.CLAMP2: clamp_at(S.CLAMP2, q2, qa2)}


def start():
    return {S.CASTING: T.CASTING_PART.rest_state(), S.CLAMP1: T.CLAMP1_PART.rest_state(),
            S.CLAMP2: T.CLAMP2_PART.rest_state()}


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


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        st = T.CASTING_PART.rest_state()
        self.assertAlmostEqual(st.low[2] - L.TOP, L.ST_H, places=6)  # on the staging block
        self.assertAlmostEqual(st.pos[0], L.ST_X)
        self.assertAlmostEqual(st.pos[1], L.ST_Y)
        self.assertGreater(L.START_Y, 0.17 + L.CS_W / 2 - 0.0001)  # inside the block's 150 mm
        self.assertLess(L.START_Y + L.CS_W / 2, L.ST_Y + L.ST_W / 2)
        self.assertGreaterEqual(L.ST_X - L.ST_L / 2, 0.17 - 1e-9)

    def test_start_is_not_seated_and_scores_zero(self):
        r = S.read(start())
        self.assertFalse(S.seated(r))
        self.assertEqual(run(start()).score, 0.0)

    def test_seat_matches_the_stops(self):
        self.assertAlmostEqual(L.SEAT_X + L.CS_L / 2, L.S1_X0)
        self.assertAlmostEqual(L.SEAT_Y - L.CS_W / 2, L.S2_Y1)
        r = S.read(scene())
        g1, g2 = S.stop_gaps(r)
        for g in g1 + g2:
            self.assertAlmostEqual(g, 0.0, places=9)
        self.assertTrue(S.seated(r))

    def test_stops_lie_within_the_casting_faces(self):
        self.assertGreaterEqual(L.S1_Y0, L.SEAT_Y - L.CS_W / 2)
        self.assertLessEqual(L.S1_Y1, L.SEAT_Y + L.CS_W / 2)
        self.assertGreaterEqual(L.S2_X0, L.SEAT_X - L.CS_L / 2)
        self.assertLessEqual(L.S2_X1, L.SEAT_X + L.CS_L / 2)
        self.assertLess(L.STOP_H, L.CS_H)

    def test_placed_casting_is_short_of_the_stops(self):
        x1 = L.PLACE_X + L.CS_L / 2
        y0 = L.PLACE_Y - L.CS_W / 2
        self.assertGreaterEqual(L.S1_X0 - x1, 0.005)
        self.assertGreaterEqual(y0 - L.S2_Y1, 0.010)
        self.assertGreater(L.PLACE_X - L.CS_L / 2 - (L.C1.px + L.BASE_HALF_ALONG), 0.010)  # clear of C1's base

    def test_pad_sits_on_the_casting_when_flat(self):
        for c in L.CLAMPS:
            p = c.pad_bottom(L.QA_FLAT)
            self.assertAlmostEqual(p[2], L.CS_TOP, places=9)
            self.assertLessEqual(abs(p[0] - L.SEAT_X), L.CS_L / 2 - 0.008)
            self.assertLessEqual(abs(p[1] - L.SEAT_Y), L.CS_W / 2 - 0.008)

    def test_open_clamp_stands_clear_of_the_casting(self):
        for c in L.CLAMPS:
            self.assertAlmostEqual(math.degrees(L.ARM_OPEN), 99.0)
            pad = c.pad_bottom(0.0)
            self.assertGreater(pad[2], L.CS_TOP + 0.06, c.name)  # pad 60 mm or more over the casting's top
            for p in (c.handle_point(c.handle, 0.0, 0.0), c.arm_point(c.arm, 0.0, 0.0)):
                self.assertLess(p[0], 0.0, c.name)  # handle and arm lean back, away from the casting
            near = (c.pivot[:2] - np.array([L.SEAT_X, L.SEAT_Y])) @ c.dir
            self.assertLess(near, -L.CS_L / 2 + 0.03)  # its pivot is behind the pad, over the casting's edge or off it

    def test_parts_follow_the_joints(self):
        """With the joints set as the linkage sets them, mechkit's links put the handle's end, the crank pin, the
        link's far end and the pad's bottom where layout.Clamp says (this checks the hinge axes' signs)."""
        for c in L.CLAMPS:
            spec = T.SPECS[c.name]
            org = np.array([c.px, c.py, 0.0])
            for qh in (0.0, 0.3, 0.7, L.QH_DC - 0.02, L.QH_DC, L.QH_UPPER):
                qa = c.arm_target(qh)
                st = mk.state_from_poses(spec, (c.px, c.py, L.TOP, 0, 0, 0, 1),
                                         joints={S.HANDLE: qh, S.ARM: qa, S.LINK: c.link_joint(qh)})
                top = np.array([0.0, 0.0, L.TOP])
                end0 = c.world(*c.handle_point(c.handle, 0.0, 0.0)) - org
                self.assertTrue(np.allclose(st.link_to_world(S.HANDLE, end0) - top,
                                            c.world(*c.handle_point(c.handle, 0.0, qh)), atol=1e-6), (c.name, qh))
                c0 = c.world(*c.crank_pin(0.0)) - org
                self.assertTrue(np.allclose(st.link_to_world(S.HANDLE, c0) - top, c.world(*c.crank_pin(qh)),
                                            atol=1e-6), (c.name, qh))
                d0 = c.world(*c.tail_pin(0.0)) - org
                d_link = st.link_to_world(S.LINK, d0) - top
                d_arm = st.link_to_world(S.ARM, d0) - top
                self.assertTrue(np.allclose(d_arm, c.world(*c.tail_pin(qa)), atol=1e-6), (c.name, qh))
                self.assertLess(float(np.linalg.norm(d_link - d_arm)), 1e-5, (c.name, qh))  # the loop is closed
                pad = st.link_to_world(S.ARM, S.PAD_DRAWN[c.name]) - top
                self.assertTrue(np.allclose(pad, c.pad_bottom(qa), atol=1e-6), (c.name, qh))

    def test_pad_bottom_face_is_at_the_point(self):
        for c in L.CLAMPS:
            st = clamp_at(c.name, L.QH_UPPER)
            d = float(st.distance(c.pad_bottom(L.QA_FLAT) + np.array([0.0, 0.0, L.TOP + 0.0005]))[0])
            self.assertLess(d, 0.0011)

    def test_shut_clamps_do_not_cross(self):
        sa, sb = clamp_at(S.CLAMP1, L.QH_UPPER), clamp_at(S.CLAMP2, L.QH_UPPER)
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0.15, 0.40, 6000), rng.uniform(-0.12, 0.22, 6000),
                               L.TOP + rng.uniform(0.0, 0.18, 6000)])
        inside_a = sa.distance(pts) == 0.0
        inside_b = sb.distance(pts) == 0.0
        self.assertGreater(inside_a.sum(), 20)
        self.assertGreater(inside_b.sum(), 20)
        self.assertFalse(np.any(inside_a & inside_b))

    def test_joint_shapes_are_positive(self):
        for part in (T.CLAMP1_PART, T.CLAMP2_PART):
            for body in part.body_names:
                for s in part.body_shapes(body):
                    self.assertTrue(all(v > 0 for v in s.size), (part.name, body))


class Linkage(unittest.TestCase):
    def test_open_and_dead_centre_angles(self):
        for c in L.CLAMPS:
            self.assertAlmostEqual(c.arm_target(0.0), 0.0, places=6)  # handle on its open stop: arm on its own
            self.assertAlmostEqual(math.degrees(c.psi(0.0)), -57.0)
            self.assertAlmostEqual(math.degrees(L.QH_UPPER - L.QH_DC), 1.5)
            self.assertGreater(c.arm_target(L.QH_DC), L.QA_FLAT)  # with nothing under the pad: past flat
            self.assertAlmostEqual(c.arm_target(L.QH_DC), L.QA_FLAT + c.delta, places=6)

    def test_ratio_is_zero_at_dead_centre_and_reverses_past_it(self):
        for c in L.CLAMPS:
            self.assertGreater(c.arm_ratio(0.01), 3.0)  # the arm swings fast while the handle starts
            self.assertAlmostEqual(c.arm_ratio(L.QH_DC), 0.0, places=2)
            self.assertLess(c.arm_ratio(L.QH_UPPER - 1e-3), 0.0)
            qs = np.linspace(0.0, L.QH_DC, 200)
            at = [c.arm_target(q) for q in qs]
            self.assertTrue(all(b >= a - 1e-9 for a, b in zip(at, at[1:])))  # closes all the way to dead centre

    def test_press_is_20_n_on_the_stop(self):
        for c in L.CLAMPS:
            self.assertAlmostEqual(c.press_force(L.QH_UPPER), L.PRESS_FORCE, places=6)
            self.assertGreater(c.press_force(L.QH_DC), L.PRESS_FORCE)  # through dead centre the press peaks
            self.assertLess(c.press_force(L.QH_DC), 30.0)
            self.assertEqual(c.press_force(c.touch_qh() - 0.01), 0.0)
        self.assertAlmostEqual(L.PRESS_FORCE, 20.0)
        self.assertEqual(round(L.C1.press_force(L.QH_DC)), 25)
        self.assertEqual(round(L.C2.press_force(L.QH_DC)), 29)

    def test_touch_is_a_few_degrees_short_of_vertical(self):
        for c in L.CLAMPS:
            d = math.degrees(c.touch_qh() - L.QH_DC)
            self.assertLess(d, -2.0)
            self.assertGreater(d, -4.0)

    def test_handle_force_is_within_the_arm(self):
        """The push at the grasp point (15 mm from the handle's end) that holds the handle with the pad on the
        casting, over the whole stroke: never over 3 N, peaking 2.2 N (C1) and 2.7 N (C2); past dead centre it turns
        negative (the lock pulls the handle onto its stop)."""
        for c, peak in ((L.C1, 2.2), (L.C2, 2.7)):
            s = c.handle - L.GRASP_BACK
            fs = [c.handle_force(q, s) for q in np.linspace(0.0, L.QH_DC - 1e-4, 600)]
            self.assertLess(max(fs), 3.0, c.name)
            self.assertAlmostEqual(max(fs), peak, delta=0.06)
            self.assertGreaterEqual(min(fs), -1e-9)
            lock = c.handle_force(L.QH_UPPER, s)
            self.assertLess(lock, -2.0, c.name)

    def test_lock_holds_with_the_weights(self):
        for c in L.CLAMPS:
            lock = c.handle_torque(L.QH_UPPER, L.QA_FLAT, c.arm_target(L.QH_UPPER))
            self.assertGreater(lock, 0.15)  # N m onto the stop
            self.assertGreater(lock, 10 * abs(c.weight_torque(L.QH_UPPER)))

    def test_weights_open_it_near_its_stop_and_close_it_nearer_vertical(self):
        for c in L.CLAMPS:
            self.assertLess(c.weight_torque(0.0), 0.0)  # it rests on its open stop
            b = math.degrees(c.balance_qh() - L.QH_DC)
            self.assertGreater(b, -45.0)
            self.assertLess(b, -35.0)
            self.assertGreater(c.weight_torque(L.QH_DC - math.radians(20)), 0.0)
            self.assertLess(max(abs(c.weight_torque(q)) for q in np.linspace(0, L.QH_UPPER, 100)), 0.06)

    def test_spindle_and_joint_springs(self):
        for c in L.CLAMPS:
            self.assertAlmostEqual(c.k_arm, L.SPINDLE_K * c.pad_s ** 2)
            ke = c.k_arm * c.arm_ratio(c.touch_qh()) ** 2
            self.assertAlmostEqual(c.handle_k, L.HANDLE_K_FRACTION * ke)
            self.assertGreater(L.HANDLE_K_FRACTION, 0.5)


class Rule(unittest.TestCase):
    def test_free_handle_gets_no_torque(self):
        for c in L.CLAMPS:
            for qh in (0.0, 0.4, 0.8):
                t, at = T.linkage_targets(c, qh, c.arm_target(qh), 0.0, c.arm_target(qh))
                self.assertAlmostEqual(t[S.HANDLE], qh, places=9)
                self.assertAlmostEqual(t[S.ARM], c.arm_target(qh))
                self.assertAlmostEqual(t[S.LINK], c.link_joint(qh))
                self.assertEqual(at, t[S.ARM])

    def test_first_step_has_no_torque(self):
        t, _ = T.linkage_targets(L.C1, 0.5, 0.0, 0.0, None)
        self.assertAlmostEqual(t[S.HANDLE], 0.5)

    def test_pad_on_the_casting_pushes_the_handle_open_short_of_dead_centre(self):
        for c in L.CLAMPS:
            qh = L.QH_DC - math.radians(1.5)
            at = c.arm_target(qh)
            t, _ = T.linkage_targets(c, qh, L.QA_FLAT, 0.0, at)
            self.assertLess(t[S.HANDLE], qh)
            want = qh - c.k_arm * (at - L.QA_FLAT) * c.arm_ratio(qh) / c.handle_k
            self.assertAlmostEqual(t[S.HANDLE], want, places=9)

    def test_past_dead_centre_it_pushes_the_handle_onto_its_stop(self):
        for c in L.CLAMPS:
            at = c.arm_target(L.QH_UPPER)
            t, _ = T.linkage_targets(c, L.QH_UPPER, L.QA_FLAT, 0.0, at)
            self.assertGreater(t[S.HANDLE], L.QH_UPPER)

    def test_lead_follows_the_handle_speed_and_is_capped(self):
        c = L.C2
        t, _ = T.linkage_targets(c, 0.5, c.arm_target(0.5), 0.4, c.arm_target(0.5))
        self.assertAlmostEqual(t[S.HANDLE], 0.5 + L.HANDLE_LEAD * 0.4 / L.CONTROL_HZ)
        t, _ = T.linkage_targets(c, 0.5, c.arm_target(0.5), 50.0, c.arm_target(0.5))
        self.assertAlmostEqual(t[S.HANDLE], 0.5 + L.HANDLE_LEAD_MAX)

    def test_unreadable_sets_nothing(self):
        for bad in (None, float("nan"), "x", float("inf")):
            self.assertEqual(T.linkage_targets(L.C1, bad, 0.0, 0.0, 0.3), ({}, 0.3))
            if bad is not None:
                self.assertEqual(T.linkage_targets(L.C1, 0.2, bad, 0.0, 0.3), ({}, 0.3))
        t, _ = T.linkage_targets(L.C1, 0.2, 0.0, float("nan"), None)
        self.assertAlmostEqual(t[S.HANDLE], 0.2)

    def test_targets_stay_within_the_kits_bounds(self):
        for c in L.CLAMPS:
            for qh in np.linspace(-0.05, L.QH_UPPER + 0.05, 50):
                for qa in (0.0, L.QA_FLAT):
                    t, _ = T.linkage_targets(c, qh, qa, 1.0, c.arm_target(min(max(qh, 0), L.QH_UPPER)))
                    for v in t.values():
                        self.assertLess(abs(v), 2 * math.pi)

    def test_joint_declarations(self):
        for part, c in ((T.CLAMP1_PART, L.C1), (T.CLAMP2_PART, L.C2)):
            self.assertIsNone(part.mass)  # fixed base
            h, a, k = part.link(S.HANDLE), part.link(S.ARM), part.link(S.LINK)
            self.assertEqual((h.lower, a.lower), (0.0, 0.0))
            self.assertAlmostEqual(h.upper, L.QH_UPPER)
            self.assertAlmostEqual(h.spring, c.handle_k)
            self.assertAlmostEqual(a.spring, c.k_arm)
            self.assertGreater(a.upper, c.arm_target(L.QH_DC))  # never the arm's stop that holds it, the link does
            self.assertLessEqual(k.lower, min(c.link_joint(q) for q in np.linspace(0, L.QH_UPPER, 50)))
            self.assertGreaterEqual(k.upper, max(c.link_joint(q) for q in np.linspace(0, L.QH_UPPER, 50)))
            self.assertAlmostEqual(c.link_joint(0.0), 0.0, places=9)

    def test_rule_is_registered(self):
        self.assertTrue(any(fn is T._linkage_rule or getattr(fn, "__wrapped__", None) is T._linkage_rule
                            for fn in mk._hub().step_fns.values()))

    def test_handle_target_is_bounded(self):
        c = L.C2
        t, _ = T.linkage_targets(c, L.QH_DC - 0.02, 0.0, 0.0, c.arm_target(L.QH_DC - 0.02))
        self.assertAlmostEqual(abs(t[S.HANDLE] - (L.QH_DC - 0.02)), L.HANDLE_TARGET_MAX)
        self.assertGreater(L.HANDLE_TARGET_MAX * c.handle_k, 3 * c.handle_torque(L.QH_UPPER, L.QA_FLAT,
                                                                                   c.arm_target(L.QH_UPPER)))


class Tiers(unittest.TestCase):
    def test_both_locked_seated_hands_off_is_one(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["locked1"] and res.details["locked2"])

    def test_one_locked_is_half(self):
        self.assertEqual(run(scene(q1=0.0)).score, 0.5)
        self.assertEqual(run(scene(q2=0.0)).score, 0.5)
        self.assertFalse(run(scene(q2=0.0)).success)

    def test_none_locked_is_a_quarter(self):
        self.assertEqual(run(scene(q1=0.0, q2=0.0)).score, 0.25)

    def test_short_of_over_centre_is_not_locked(self):
        for c in L.CLAMPS:
            for deg in (-20.0, -3.0, -1.0, 0.0, 0.4):
                q = L.QH_DC + math.radians(deg)
                kw = {"q1": q} if c is L.C1 else {"q2": q}
                self.assertEqual(run(scene(**kw)).score, 0.5, (c.name, deg))
            q = L.QH_DC + math.radians(0.6)
            self.assertEqual(run(scene(q1=q, q2=q)).score, 1.0)

    def test_not_seated_is_zero_whatever_the_clamps_do(self):
        far = casting_at(x=L.SEAT_X - 0.012)
        self.assertEqual(run(scene(cs=far)).score, 0.0)
        self.assertEqual(run(scene(q1=0.0, q2=0.0, cs=far)).score, 0.0)

    def test_every_zero_condition_of_seating(self):
        tol = L.SEAT_TOL
        for label, cs in (
                ("1.5 mm off S1", casting_at(x=L.SEAT_X - 0.0015)),
                ("1.5 mm off S2", casting_at(y=L.SEAT_Y + 0.0015)),
                ("yawed 2 degrees", casting_at(yaw=math.radians(2.0))),
                ("tilted 2 degrees", casting_at(pitch=math.radians(2.0))),
                ("5 mm up (on the stops' tops)", casting_at(z=SEAT_Z + 0.005)),
                ("upside down", casting_at(roll=math.pi)),
                ("turned a quarter", casting_at(yaw=math.pi / 2)),
                ("slid along S1 off S2", casting_at(y=0.05)),
                ("through S1", casting_at(x=L.SEAT_X + 0.003))):
            for q in (L.QH_UPPER, 0.0):
                self.assertEqual(run(scene(q1=q, q2=q, cs=cs)).score, 0.0, (label, q))
        for dx in (0.0, 0.0009, -0.0009):
            self.assertGreater(run(scene(cs=casting_at(x=L.SEAT_X + dx))).score, 0.0, dx)
        self.assertLessEqual(tol, 0.001)

    def test_shut_on_nothing_is_not_locked(self):
        """The arm where the linkage puts it with nothing under the pad (past flat): the pad is below where the
        casting's top would be; here the casting is lowered 3 mm so nothing touches."""
        for name in (S.CLAMP1, S.CLAMP2):
            c = L.CLAMP_OF[name] if hasattr(L, "CLAMP_OF") else S.CLAMP_OF[name]
            r = S.read(scene(cs=casting_at(z=SEAT_Z - 0.003)))
            self.assertTrue(S.empty(r, name))
            self.assertFalse(S.locked(r, name))
            self.assertTrue(S.shut(c.arm_target(L.QH_UPPER) * 0 + L.QH_UPPER))

    def test_pad_height_window(self):
        r = S.read(scene())
        self.assertTrue(S.pad_touching(r, S.CLAMP1))
        up = S.read(scene(cs=casting_at(z=SEAT_Z - 0.003)))  # casting 3 mm too low: the pad is 3 mm above it
        self.assertFalse(S.pad_touching(up, S.CLAMP1))
        gap = S.read(scene(cs=casting_at(z=SEAT_Z - 0.001)))  # a 1 mm air gap under the pad holds nothing down
        self.assertFalse(S.pad_touching(gap, S.CLAMP1))
        self.assertFalse(S.pad_touching(gap, S.CLAMP2))
        near = S.read(scene(cs=casting_at(z=SEAT_Z - 0.0004)))  # 0.4 mm: within contact slop, pressed
        self.assertTrue(S.pad_touching(near, S.CLAMP1))

    def test_unlocked_clamps_never_raise_the_score(self):
        self.assertLess(run(scene(q1=0.0)).score, run(scene()).score)

    def test_shut_threshold_is_half_a_degree_past_vertical(self):
        self.assertAlmostEqual(math.degrees(L.QH_SHUT - L.QH_DC), 0.5)
        self.assertTrue(S.shut(L.QH_DC + math.radians(0.51)))
        self.assertFalse(S.shut(L.QH_DC + math.radians(0.49)))
        self.assertTrue(S.shut(L.QH_UPPER))


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_casting(self):
        arms = {"left": arm((L.SEAT_X, L.SEAT_Y + 0.02, L.TOP + 0.04), 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_hand_on_a_handle(self):
        c = L.C1
        p = c.grasp_point(c.handle - L.GRASP_BACK, L.QH_UPPER) + np.array([0.0, 0.0, L.TOP + L.GRASP_TCP_UP])
        arms = {"left": FAR["left"], "right": arm(p, 0.13)}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_hand_29_and_31_mm_from_the_casting(self):
        def gap_hand(g):
            # a closed gripper whose pad-tip point is straight above the casting's top, g away (zone: 15.7 mm radius)
            r = rigstate.zone_radius(0.0)
            tips = np.array([0.31, 0.0, L.TOP + L.CS_TOP + g + r])
            tcp = tips + np.array([0.0, 0.0, T.PAD_TIP])
            return rigstate.Arm(tcp, tips, 0.0)
        near = {"left": gap_hand(0.029), "right": FAR["right"]}
        far = {"left": gap_hand(0.031), "right": FAR["right"]}
        self.assertEqual(run(scene(q1=0.0, q2=0.0), arms=near).score, 0.0)
        self.assertEqual(run(scene(q1=0.0, q2=0.0), arms=far).score, 0.25)

    def test_parked_hands_are_clear(self):
        arms = {"left": arm((T.PARK_L[0], T.PARK_L[1], L.TOP + T.PARK_L[2])),
                "right": arm((T.PARK_R[0], T.PARK_R[1], L.TOP + T.PARK_R[2]))}
        self.assertEqual(run(scene(), arms=arms).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        near = {"left": arm((L.SEAT_X, L.SEAT_Y, L.TOP + 0.05), 0.0), "right": FAR["right"]}
        run(scene(), arms=near, tracker=tr)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_moving_casting(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(cs=casting_at(y=L.SEAT_Y + 0.0001 * i)), FAR)  # 0.1 mm a step: 2 mm a second
        self.assertEqual(res.score, 0.0)

    def test_creeping_under_a_mm_is_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(cs=casting_at(z=SEAT_Z + 0.00002 * i)), FAR)  # 0.4 mm in a second
        self.assertEqual(res.score, 1.0)

    def test_turning_casting(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(cs=casting_at(yaw=math.radians(0.06) * i)), FAR)  # 1.2 degrees in a second
        self.assertEqual(res.score, 0.0)

    def test_handle_wobble(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(q1=L.QH_UPPER - math.radians(0.25) * (i % 12)), FAR)  # 2.75 degrees of wobble
        self.assertEqual(res.score, 0.0)


class Latch(unittest.TestCase):
    def test_clamped_off_the_stop_scores_by_the_end_state(self):
        """Clamped 5 mm off S1: not seated, 0; reopened and reseated (a corrected mistake): scored on the end state.
        The CROOKED step is recorded for the trace only."""
        tr = S.Tracker()
        self.assertEqual(run(scene(cs=casting_at(x=L.SEAT_X - 0.005)), tracker=tr, steps=WIN + 3).score, 0.0)
        self.assertTrue(tr.latched)
        run(scene(q1=0.0, q2=0.0, cs=casting_at(x=L.SEAT_X - 0.005)), tracker=tr)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 3).score, 1.0)

    def test_a_clamped_casting_cannot_be_pushed_home_by_an_arm(self):
        """With even one pad pressing 20 N the casting needs about 12 N to slide (friction 0.3 under and over it), well
        beyond the about 5 N an arm pushes sideways, so clamping first leaves it off its stop (not seated)."""
        mu, w = L.CS_FRICTION[0], L.CS_MASS * L.G
        for c in L.CLAMPS:
            f = c.press_force(L.QH_UPPER)
            self.assertGreater(mu * (f + w) + mu * f, 12.0)

    def test_two_pads_hold_the_casting_against_a_20_n_proof_pull(self):
        mu, w = L.CS_FRICTION[0], L.CS_MASS * L.G
        press = sum(c.press_force(L.QH_UPPER) for c in L.CLAMPS)
        self.assertGreater(press + w, 20.0)  # straight up
        self.assertGreater(mu * (press + w) + mu * press, 20.0)  # sideways
        up = side = 20.0 / math.sqrt(2.0)
        self.assertGreater(mu * (press + w - up) + mu * press, side)  # 45 degrees up and sideways
        one = L.C1.press_force(L.QH_UPPER)
        self.assertLess(mu * (one + w) + mu * one, 20.0)

    def test_two_mm_off_is_crooked(self):
        tr = S.Tracker()
        tr.update(scene(cs=casting_at(x=L.SEAT_X - 0.002)), FAR)
        self.assertTrue(tr.latched)

    def test_one_point_two_mm_off_is_not_crooked(self):
        tr = S.Tracker()
        tr.update(scene(cs=casting_at(x=L.SEAT_X - 0.0012)), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 3).score, 1.0)

    def test_open_clamps_never_latch(self):
        tr = S.Tracker()
        for dx in (0.0, -0.005, -0.012, 0.0):
            tr.update(scene(q1=0.0, q2=0.0, cs=casting_at(x=L.SEAT_X + dx)), FAR)
        self.assertFalse(tr.latched)

    def test_one_clamp_locked_off_the_stop_latches(self):
        tr = S.Tracker()
        tr.update(scene(q1=0.0, cs=casting_at(y=L.SEAT_Y + 0.004)), FAR)
        self.assertTrue(tr.latched)

    def test_hand_held_casting_with_clamps_closed_scores_zero(self):
        arms = {"left": arm((L.SEAT_X, L.SEAT_Y + 0.03, L.TOP + 0.05), 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({S.CASTING: casting_at()}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_casting(self):
        bad = mk.state_from_poses(T.CASTING_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(cs=bad), FAR).score, 0.0)

    def test_nan_handle(self):
        s = scene()
        s[S.CLAMP2] = clamp_at(S.CLAMP2, float("nan"))
        self.assertEqual(S.Tracker().update(s, FAR).score, 0.0)

    def test_nan_then_recovery_needs_a_fresh_second(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.CASTING_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(tr.update(scene(cs=bad), FAR).score, 0.0)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 1).score, 1.0)

    def test_non_dict_states(self):
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(scene(cs=casting_at(x=L.SEAT_X - 0.006)), FAR)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(300):
            cs = casting_at(x=L.SEAT_X + rng.uniform(-0.02, 0.02), y=L.SEAT_Y + rng.uniform(-0.02, 0.02),
                            z=SEAT_Z + rng.uniform(-0.002, 0.02), yaw=rng.uniform(-0.1, 0.1),
                            pitch=rng.uniform(-0.1, 0.1), roll=rng.uniform(-0.1, 0.1))
            r = tr.update(scene(q1=rng.uniform(0.0, L.QH_UPPER), q2=rng.uniform(0.0, L.QH_UPPER), cs=cs), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))

    def test_details_line_fits(self):
        d = run(scene()).details
        for k in ("q1_deg", "q2_deg", "psi1_deg", "psi2_deg"):
            self.assertIn(k, d)
        self.assertAlmostEqual(d["psi1_deg"], 1.5, places=2)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["fixture", "guard", "staging", "clamp1", "clamp2", "casting"])
        self.assertEqual([o.mass for o in T.OBJECTS[:5]], [None] * 5)
        self.assertAlmostEqual(T.CASTING_PART.mass, L.CS_MASS)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("push", "mm", "lift", "pick", "carry", "step", "wait"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
                    self.assertGreaterEqual(w.pos[2], L.TOP + 0.0149 + T.PAD_TIP - 0.013, (v, a, w.label))
            # the demo ends within half the run limit
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ * 2, run_limit)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_gripper_commands_are_slow_enough(self):
        """Every open/close change takes 1.5 s or more, except the pinch on a handle and its release (1 s and up)."""
        for a, wps in T.plans("").items():
            for x, y in zip(wps, wps[1:]):
                if abs(x.finger - y.finger) > 0.03:
                    self.assertGreaterEqual(y.duration, 1.0 - 1e-9, (a, y.label))
                    if "clamp" not in y.label:
                        self.assertGreaterEqual(y.duration, 1.5 - 1e-9, (a, y.label))

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "no_clamps", {"left"}),
                               ("DEMO_VARIANT", "left_only", {"left"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)
        self.assertEqual(set(T.demo()), {"left", "right"})

    def test_arms_never_work_close_together(self):
        """The right arm leaves its waiting spot only once the left gripper has let go of C2's handle."""
        plan = T.plans("")
        t = 0.0
        for w in plan["left"]:
            t += w.duration
            if w.label.startswith("straight up off clamp2"):
                break
        self.assertAlmostEqual(plan["right"][0].duration, t)

    def test_carry_clears_the_clamps(self):
        """The casting's underside clears each open clamp's whole box by 5 mm over its footprint while the left arm
        carries it, sampled along every straight segment from the grip to the set-down."""
        wps = T.plans("")["left"]
        i0 = next(i for i, w in enumerate(wps) if w.label.startswith("lift"))
        i1 = next(i for i, w in enumerate(wps) if w.label.startswith("set it down"))
        boxes = [T.SPECS[n].world_bounds() for n in (S.CLAMP1, S.CLAMP2)]
        for a, b in zip(wps[i0 - 1:i1], wps[i0:i1 + 1]):
            for f in np.linspace(0.0, 1.0, 41):
                p = np.array(a.pos) * (1 - f) + np.array(b.pos) * f
                bottom = p[2] - L.TOP - T.GRIP_TO_BOTTOM
                cy = p[1] - T.GRIP_DY
                x0, x1 = p[0] - L.CS_L / 2, p[0] + L.CS_L / 2
                y0, y1 = cy - L.CS_W / 2, cy + L.CS_W / 2
                for lo, hi in boxes:
                    if x1 > lo[0] and x0 < hi[0] and y1 > lo[1] and y0 < hi[1]:
                        self.assertGreaterEqual(bottom, hi[2] - L.TOP + 0.005, (b.label, f))

    def test_set_down_is_over_the_plate_and_short_of_the_stops(self):
        wps = {w.label: w for w in T.plans("")["left"]}
        w = wps["set it down"]
        self.assertAlmostEqual(w.pos[0], L.PLACE_X)
        self.assertAlmostEqual(w.pos[1], L.PLACE_Y + T.GRIP_DY)
        bottom = w.pos[2] - L.TOP - T.GRIP_TO_BOTTOM
        self.assertAlmostEqual(bottom - L.PLATE_H, 0.003, places=6)
        self.assertGreater(T.SHIFT_Z - T.GRIP_TO_BOTTOM, L.PLATE_H + L.STOP_H + 0.010)

    def test_held_slide_and_push_reach_and_overshoot_the_stops(self):
        wps = {w.label: w for w in T.plans("")["left"]}
        slide = wps["slide it -y onto S2, still held"]
        self.assertAlmostEqual(slide.pos[1] - T.GRIP_DY, L.SEAT_Y - T.HELD_OVERTRAVEL)
        self.assertLess(slide.pos[2] - L.TOP - T.GRIP_TO_BOTTOM, L.PLATE_H + L.STOP_H)  # low enough to meet S2
        ease = wps["ease off to a light press"]
        self.assertAlmostEqual(ease.pos[1] - T.GRIP_DY, L.SEAT_Y - 0.001)
        p1 = wps["push it +x into S1"]
        face = p1.pos[0] + L.PAD_FACE
        self.assertGreater(face - (L.SEAT_X - L.CS_L / 2), 0.006)  # past where the near face ends seated
        self.assertLess(abs(T.PUSH1_Y), L.S1_Y1 - L.FIST_HALF + 0.02)
        self.assertGreater(T.PUSH1_Y - L.FIST_HALF, L.C1.py + L.BASE_HALF_ACROSS + 0.015)
        self.assertGreater(T.PUSH_TIPS, L.PLATE_H + 0.010)
        self.assertLess(T.PUSH_TIPS, L.CS_TOP - 0.010)
        self.assertNotIn("push it -y into S2", wps)  # no fist on the left face: C2's open arm stands just beyond it

    def test_handle_strokes_run_from_open_to_past_the_stop(self):
        """The pinch carries each handle from open to just short of where the press starts (never into it, so the
        pinch is never asked to drive the press); a closed block then pushes the grip from behind, its face commanded
        past the grip's back face on the stop."""
        for side, c in (("left", L.C2), ("right", L.C1)):
            wps = T.plans("")[side]
            arc = [w for w in wps if w.label.startswith(("swing " + c.name, "lower " + c.name))]
            self.assertEqual(len(arc), len(T.ARC_FREE) - 1 + len(T.ARC_LAND))
            self.assertAlmostEqual(T.ARC_FREE[0], -57.0)
            for deg, w in zip(T.ARC_FREE[1:] + T.ARC_LAND, arc):
                self.assertTrue(np.allclose(w.pos, np.array(T.grasp_tcp(c, deg)) + np.array([0, 0, L.TOP])))
            self.assertLess(T.ARC_LAND[-1], 0.0)  # the pinch stops short of dead centre
            self.assertGreater(L.QH_DC + math.radians(T.ARC_LAND[-1]), c.touch_qh() - 1e-9)  # pad down by then
            pinch = next(w for w in wps if w.label.startswith("pinch " + c.name))
            g = c.grasp_point(c.handle - L.GRASP_BACK, 0.0)
            self.assertTrue(np.allclose(pinch.pos, g + np.array([0, 0, L.TOP + L.GRASP_TCP_UP]), atol=1e-9))
            self.assertLess(-2 * T.PINCH, L.GRIP_WIDTH)  # squeezed across the grip
            labels = [w.label for w in wps]
            i_let = labels.index(f"let go of {c.name}'s grip")
            i_push = next(i for i, lab in enumerate(labels) if lab.startswith(f"push {c.name}'s handle"))
            self.assertLess(i_let, i_push)
            push = wps[i_push]
            self.assertEqual(push.finger, T.CLOSED)
            self.assertAlmostEqual(push.yaw, T.PUSH_YAW[c.name])
            self.assertGreaterEqual(push.duration, 3.0 - 1e-9)  # slowly through the press
            z = push.pos[2] - L.TOP
            tips = z - T.PAD_TIP
            self.assertGreater(tips, c.A[1] + c.handle - L.GRIP_L + 0.010)  # the block on the grip, 10 mm in from
            self.assertLess(tips + L.FIST_H, c.A[1] + c.handle - 0.010)  # both its ends
            face = float((np.array(push.pos[:2]) - np.array([c.px, c.py])) @ c.dir) + L.PAD_FACE
            on_stop = T.back_face_a(c, math.degrees(L.OVER_CENTRE), tips)
            self.assertGreaterEqual(face - on_stop, 0.015)  # commanded past the stop: the arm trails a pushed load
            start = wps[i_push - 2]
            self.assertTrue(start.label.startswith("down behind"))
            sface = float((np.array(start.pos[:2]) - np.array([c.px, c.py])) @ c.dir) + L.PAD_FACE
            top = c.A[1] + c.handle * math.cos(math.radians(T.PUSH_START_PSI))
            self.assertLess(sface, T.back_face_a(c, -9.0, top) - 0.009)  # comes down well behind the grip
            up = wps[i_push - 3]
            self.assertGreater(up.pos[2] - L.TOP - T.PAD_TIP, c.A[1] + c.handle + 0.010)  # over the handle's end

    def test_block_push_is_square_on_the_grip(self):
        """The block pushes with the face ahead along its fingers' closing direction (the face the S1 push uses) and
        is centred across the grip."""
        for c in (L.C1, L.C2):
            yaw = T.PUSH_YAW[c.name]
            closing = np.array([-math.sin(yaw), math.cos(yaw)])  # yaw 0 closes along y
            self.assertAlmostEqual(abs(float(closing @ c.dir)), 1.0, places=9)
            p = T.push_tcp(c, -0.03, T.push_z(c))
            w = float((np.array(p[:2]) - np.array([c.px, c.py])) @ c.across()[:2])
            self.assertAlmostEqual(w, 0.0, places=9)

    def test_demo_reaches_with_margin(self):
        """Every demo waypoint is where the arm reaches: inboard under 0.40 m, x within 0.11 to 0.35 m."""
        for side, wps in T.plans("").items():
            base_y = 0.305 if side == "left" else -0.305
            for w in wps:
                inboard = abs(w.pos[1] - base_y)
                self.assertLess(inboard, 0.40, (side, w.label))
                self.assertGreaterEqual(w.pos[0], 0.11, (side, w.label))
                self.assertLess(w.pos[0], 0.35, (side, w.label))

    def test_plan_yaws_cover_both_arms(self):
        self.assertEqual(set(T.PLAN_YAWS), {"left", "right"})
        self.assertAlmostEqual(sum(d for d, _ in T.PLAN_YAWS["left"]), sum(w.duration for w in T.plans("")["left"]))


def tcp_at(wps, t):
    acc = 0.0
    prev = None
    for w in wps:
        if t <= acc + w.duration + 1e-9:
            f = (t - acc) / w.duration
            f = f * f * (3 - 2 * f)
            p0 = np.array(prev.pos if prev else w.pos)
            fin0 = prev.finger if prev else w.finger
            return p0 * (1 - f) + np.array(w.pos) * f, fin0 * (1 - f) + w.finger * f
        acc += w.duration
        prev = w
    return np.array(wps[-1].pos), wps[-1].finger


def play(plan):
    """The offline model and the scorer driven from the plan's own TCP positions (the checker does the same)."""
    n = int(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ) + 3 * int(HZ)
    m = T.OfflineModel()
    tr = S.Tracker()
    res = None
    for k in range(1, n + 1):
        snap = dict(FAR)
        for side, wps in plan.items():
            p, fin = tcp_at(wps, k / HZ)
            snap[side] = arm(p, min(1.0, max(0.0, fin / -0.04695)))
        rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in snap.items()})
        m.steps = k - 1
        m.step(snap)
        res = tr.update(m.states(), snap)
    return m, res


class OfflineModel(unittest.TestCase):
    def fresh(self):
        return T.OfflineModel()

    def feed(self, m, left=None, right=None, n=1):
        for _ in range(n):
            snap = {"left": left or FAR["left"], "right": right or FAR["right"]}
            rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in snap.items()})
            m.step(snap)

    def test_states_at_rest_match_the_spawn(self):
        st = self.fresh().states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)
        self.assertEqual(S.Tracker().update(st, FAR).score, 0.0)

    def test_carry_and_set_down(self):
        m = self.fresh()
        z = L.TOP + T.GRIP_Z
        self.feed(m, left=arm((L.START_X, L.START_Y, z), 0.55))
        for f in np.linspace(0.0, 1.0, 30):
            self.feed(m, left=arm((L.START_X + f * (0.25 - L.START_X), L.START_Y + f * (0.10 - L.START_Y),
                                   z + 0.05 * f), 0.55))
        self.assertAlmostEqual(m.cx, 0.25, places=3)
        self.assertAlmostEqual(m.cy, 0.10, places=3)
        self.feed(m, left=arm((L.PLACE_X, L.PLACE_Y, L.TOP + T.PLACE_Z), 0.55))
        self.feed(m, left=arm((L.PLACE_X, L.PLACE_Y, L.TOP + T.PLACE_Z), 0.9))
        self.assertAlmostEqual(m.cz, SEAT_Z, places=4)  # on the plate

    def test_held_casting_stops_at_s2(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.PLACE_X, L.PLACE_Y, L.PLATE_H + 0.003 + L.CS_H / 2
        z = L.TOP + T.PLACE_Z
        self.feed(m, left=arm((L.PLACE_X, L.PLACE_Y, z), 0.55))
        self.assertIn("left", m.carry)
        for y in np.linspace(L.PLACE_Y, L.SEAT_Y - 0.004, 20):
            self.feed(m, left=arm((L.PLACE_X, y, z), 0.55))
        self.assertAlmostEqual(m.cy, L.SEAT_Y)
        self.feed(m, left=arm((L.PLACE_X, L.SEAT_Y - 0.001, z), 0.9))  # let go: drops 3 mm onto the plate
        self.assertAlmostEqual(m.cy, L.SEAT_Y)
        self.assertAlmostEqual(m.cz, SEAT_Z, places=4)

    def test_open_gripper_carries_nothing(self):
        m = self.fresh()
        self.feed(m, left=arm((L.START_X, L.START_Y, L.TOP + T.GRIP_Z), 1.0))
        self.feed(m, left=arm((0.25, 0.10, L.TOP + 0.1), 1.0))
        self.assertAlmostEqual(m.cx, L.START_X)

    def test_push_stops_at_s1(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.PLACE_X, L.SEAT_Y, SEAT_Z
        tips = L.PLATE_H + 0.014
        for x in np.linspace(L.PLACE_NEAR_X - 0.02, L.PLACE_NEAR_X + 0.06, 40):
            m.steps = 490  # inside the x-closing push of the demo
            m.step({"left": arm((x, 0.03, L.TOP + tips + T.PAD_TIP), 0.0), "right": FAR["right"]})
        self.assertAlmostEqual(m.cx, L.SEAT_X)

    def test_high_fist_does_not_push(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.PLACE_X, L.SEAT_Y, SEAT_Z
        for x in np.linspace(0.2, 0.30, 30):
            m.steps = 500
            m.step({"left": arm((x, 0.03, L.TOP + 0.12), 0.0), "right": FAR["right"]})
        self.assertAlmostEqual(m.cx, L.PLACE_X)

    def _stroke(self, m, c, degs, side="right", opening=0.13):
        for deg in degs:
            p = T.grasp_tcp(c, deg)
            snap = {"left": FAR["left"], "right": FAR["right"]}
            snap[side] = arm((p[0], p[1], L.TOP + p[2]), opening)
            rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in snap.items()})
            m.step(snap)

    def test_pinched_handle_follows_the_gripper(self):
        m = self.fresh()
        self._stroke(m, L.C1, (-57.0, -45.0, -30.0))
        self.assertEqual(m.held.get("right"), S.CLAMP1)
        self.assertAlmostEqual(math.degrees(m.qh[S.CLAMP1] - L.QH_DC), -30.0, places=6)

    def test_open_gripper_over_the_grip_holds_nothing(self):
        m = self.fresh()
        self._stroke(m, L.C1, (-57.0, -45.0, -30.0), opening=1.0)
        self.assertNotIn("right", m.held)
        self.assertEqual(m.qh[S.CLAMP1], 0.0)

    def test_let_go_early_falls_open_or_settles_short_and_past_centre_locks(self):
        c = L.C1
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        self._stroke(m, c, (-57.0, -50.0))
        self.feed(m)
        self.assertEqual(m.qh[S.CLAMP1], 0.0)  # near its open stop: back open
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        self._stroke(m, c, (-57.0, -45.0, -30.0, -15.0, -5.0, -1.0))
        self.feed(m)
        self.assertAlmostEqual(m.qh[S.CLAMP1], c.touch_qh())  # short of vertical: the pad rests on the casting
        self.assertFalse(S.shut(m.qh[S.CLAMP1]))
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        self._stroke(m, c, (-57.0, -45.0, -30.0, -15.0, -5.0, -1.0, 0.5, 2.5))
        self.feed(m)
        self.assertAlmostEqual(m.qh[S.CLAMP1], L.QH_UPPER)
        self.assertAlmostEqual(m.qa(S.CLAMP1), L.QA_FLAT)
        self.assertEqual(S.Tracker().update(m.states(), FAR).details["locked1"], True)

    def test_shut_with_nothing_under_the_pad_goes_past_flat(self):
        m = self.fresh()  # the casting still on the staging block
        self._stroke(m, L.C1, (-57.0, -45.0, -30.0, -15.0, -5.0, 0.5, 2.5))
        self.feed(m)
        self.assertAlmostEqual(m.qh[S.CLAMP1], L.QH_UPPER)
        self.assertGreater(m.qa(S.CLAMP1), L.QA_FLAT)

    def _block(self, m, c, faces, side="right", steps=0):
        z = T.push_z(c)
        for f in faces:
            p = T.push_tcp(c, f, z)
            snap = {"left": FAR["left"], "right": FAR["right"]}
            snap[side] = arm((p[0], p[1], L.TOP + p[2]), 0.0)
            rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in snap.items()})
            m.steps = steps
            m.step(snap)

    def test_block_pushes_a_resting_handle_over_centre_and_it_locks(self):
        for c, name, side in ((L.C1, S.CLAMP1, "right"), (L.C2, S.CLAMP2, "left")):
            steps = round(T._until(T.plans("")[side], f"up to {c.name}'s grip") * HZ) + 5  # the push yaw
            m = self.fresh()
            m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
            m.qh[name] = c.touch_qh()
            start, load, end = T.push_points(c)
            self._block(m, c, np.linspace(start, load - 0.005, 10), side, steps)
            self.assertAlmostEqual(m.qh[name], c.touch_qh())  # short of the grip: nothing moves
            self._block(m, c, np.linspace(load, (load + end) / 2, 20), side, steps)
            self.assertGreater(m.qh[name], c.touch_qh())
            self._block(m, c, np.linspace((load + end) / 2, end, 20), side, steps)
            self.assertAlmostEqual(m.qh[name], L.QH_UPPER)
            self._block(m, c, np.linspace(end, start, 10), side, steps)
            self.feed(m)
            self.assertAlmostEqual(m.qh[name], L.QH_UPPER)  # past dead centre: stays on its stop
            self.assertTrue(S.Tracker().update(m.states(), FAR).details["locked1" if c is L.C1 else "locked2"])

    def test_block_stopped_short_of_dead_centre_lets_it_back(self):
        c = L.C1
        steps = round(T._until(T.plans("")["right"], f"up to {c.name}'s grip") * HZ) + 5
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        m.qh[S.CLAMP1] = c.touch_qh()
        start, load, end = T.push_points(c)
        tips = T.push_z(c) - T.PAD_TIP
        short = T.back_face_a(c, -1.0, tips + L.FIST_H)
        self._block(m, c, np.linspace(load, short, 10), "right", steps)
        self.assertLess(m.qh[S.CLAMP1], L.QH_DC)
        self._block(m, c, [start], "right", steps)
        self.feed(m)
        self.assertAlmostEqual(m.qh[S.CLAMP1], c.touch_qh())

    def test_block_in_front_of_the_grip_does_not_pull_it(self):
        c = L.C1
        steps = round(T._until(T.plans("")["right"], f"up to {c.name}'s grip") * HZ) + 5
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        m.qh[S.CLAMP1] = c.touch_qh()
        start, load, end = T.push_points(c)
        self._block(m, c, np.linspace(end + 0.03, end + 0.06, 10), "right", steps)
        self.assertAlmostEqual(m.qh[S.CLAMP1], c.touch_qh())

    def test_hovering_over_the_handle_does_not_move_it(self):
        m = self.fresh()
        p = T.grasp_tcp(L.C2, -57.0)
        for _ in range(5):
            m.steps = 700
            m.step({"left": arm((p[0], p[1], L.TOP + p[2] + 0.05), 0.13), "right": FAR["right"]})
        self.assertEqual(m.qh[S.CLAMP2], 0.0)

    def test_full_demo_by_the_model_scores_one(self):
        m, res = play(T.plans(""))
        self.assertEqual(res.score, 1.0, res.details)
        self.assertAlmostEqual(m.cx, L.SEAT_X)
        self.assertAlmostEqual(m.cy, L.SEAT_Y)

    def test_left_alone_scores_half_and_right_alone_zero(self):
        for idle, want in (("right", 0.5), ("left", 0.0)):
            plan = T.plans("")
            plan.pop(idle)
            _, res = play(plan)
            self.assertEqual(res.score, want, idle)

    def test_variants(self):
        self.assertEqual(play(T.plans("no_clamps"))[1].score, 0.25)
        self.assertEqual(play(T.plans("left_only"))[1].score, 0.5)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("x 200 to 380 mm", "y -160 to +160 mm", "12 mm thick", "x 330 to 345 mm", "y -45 to +45 mm",
                      "x 280 to 320 mm", "y -60 to -50 mm", "x 395 to 403 mm", "140 mm tall", "54 mm above the table",
                      "32 mm behind and 26 mm above the table", "16 mm crank", "link, 32 mm", "7 mm from the arm's",
                      "arm pivot x 236 mm, y -25 mm", "arm 62 mm", "x 288 mm", "handle 110 mm",
                      "arm pivot x 300 mm, y +125 mm", "arm 99 mm", "y +34 mm", "handle 145 mm",
                      "57 degrees", "99 degrees", "1.5 degrees past vertical", "42 mm above the table", "20 N",
                      "25 N (C1) or 29 N (C2)", "2.2 N (C1) or 2.8 N (C2)", "2.2 N (C1) or 2.7 N (C2)", "0.3 N",
                      "about 40 degrees back", "Nothing drives a clamp", "100 (x) x 150 (y) x 20 mm",
                      "x 170 to 270 mm", "y +170 to +320 mm", "60 (x) x 100 (y) x 30 mm", "0.18 kg",
                      "x 220 mm, y +245 mm", "HDPE", "four-bar"):
            self.assertIn(token, setup)
        for token in ("1 degree", "1.5 mm", "1.0 mm", "90 mm face", "40 mm face", "0.5 degree or more past vertical",
                      "0.5 mm above it", "4 mm inside", "30 mm", "at most 1 mm", "at most 1 degree", "over 2 degrees",
                      "20 N", "shut on nothing is not locked", "Possible scores: 0, 0.25, 0.5, 1.0", "1 s",
                      "1 N pull"):
            self.assertIn(token, scoring)
        for text in (setup, scoring, d["fidelity"], d["context"]):
            self.assertNotIn("25 N per pad", text)
            self.assertNotIn("drive closes", text)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "2800 steps (140 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.match(r"Dream-1 run: [^;]+; demo: 1\.0", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_numbers_in_setup_match_the_layout(self):
        self.assertAlmostEqual(L.C1.px, 0.236)
        self.assertAlmostEqual(L.C1.py, -0.025)
        self.assertAlmostEqual(L.C1.arm, 0.062)
        self.assertAlmostEqual(L.C1.pad_s, 0.052)
        self.assertAlmostEqual(L.C1.handle, 0.110)
        self.assertAlmostEqual(L.C2.px, 0.300)
        self.assertAlmostEqual(L.C2.py, 0.125)
        self.assertAlmostEqual(L.C2.arm, 0.099)
        self.assertAlmostEqual(L.C2.pad_s, 0.091)
        self.assertAlmostEqual(L.C2.handle, 0.145)
        self.assertAlmostEqual(L.Z_PIVOT, 0.054)
        self.assertEqual(tuple(L.PIVOT_A), (-0.032, 0.026))
        for c in L.CLAMPS:
            self.assertEqual(round(c.crank * 1000), 16)
            self.assertEqual(round(c.link * 1000), 32)
            self.assertEqual(round(c.rd * 1000), 7)
        self.assertEqual((L.PLATE_X0, L.PLATE_X1), (0.200, 0.380))
        self.assertAlmostEqual(L.ST_X - L.ST_L / 2, 0.170)
        self.assertAlmostEqual(L.START_X, 0.220)
        self.assertAlmostEqual(L.CS_TOP, 0.042)
        self.assertEqual(round(-math.degrees(L.C1.balance_qh() - L.QH_DC) / 5) * 5, 40)

    def test_fidelity_states_the_rule_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("207-U", "57, 99 degree", "jointed bodies", "declared rule", "four-bar", "No drive", "20 N",
                      "est", "0.18 kg"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("51-4041.00", "51-9161.00", "51-4121.06", "288,300", "170,000", "437,700", "896,000", "20 N"):
            self.assertIn(token, ctx)

    def test_context_total_and_names_no_other_task(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        self.assertEqual(288300 + 170000 + 437700, 896000)
        self.assertNotIn("TK-", ctx)

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
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)
        for token in ("inspect-robots run", "Runtime", "Seed", "Bench", "four-bar", "No drive"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
