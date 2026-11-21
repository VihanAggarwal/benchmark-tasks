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


def clamp_at(name, q):
    c = L.C1 if name == S.CLAMP1 else L.C2
    return mk.state_from_poses(T.SPECS[name], (c.px, c.py, L.TOP, 0.0, 0.0, 0.0, 1.0), joints={S.HANDLE: q},
                               source="test")


def scene(q1=L.Q_FLAT, q2=L.Q_FLAT, cs=None):
    return {S.CASTING: cs if cs is not None else casting_at(), S.CLAMP1: clamp_at(S.CLAMP1, q1),
            S.CLAMP2: clamp_at(S.CLAMP2, q2)}


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
        self.assertGreaterEqual(L.ST_X - L.ST_L / 2, 0.2)  # not under the arms' start pose

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

    def test_placed_casting_clears_everything(self):
        x0, x1 = L.PLACE_X - L.CS_L / 2, L.PLACE_X + L.CS_L / 2
        y0, y1 = L.PLACE_Y - L.CS_W / 2, L.PLACE_Y + L.CS_W / 2
        self.assertGreaterEqual(L.S1_X0 - x1, 0.010)
        self.assertGreaterEqual(y0 - L.S2_Y1, 0.010)
        for c in L.CLAMPS:
            lo, hi = T.SPECS[c.name].world_bounds()
            near = x1 > lo[0] - 0.005 and x0 < hi[0] + 0.005 and y1 > lo[1] - 0.005 and y0 < hi[1] + 0.005
            if near:  # only the low foot and post may be this close, never the lever
                self.assertLess(L.Z_PIVOT - 0.014, L.PLATE_H + L.CS_H + 0.03)
        self.assertGreater(x0 - (L.C1.px + L.BASE_HALF_ALONG), 0.005)  # C1's base is 5 mm or more from its near face

    def test_pad_sits_on_the_casting_when_shut(self):
        for c in L.CLAMPS:
            p = c.pad_bottom(L.Q_FLAT)
            self.assertAlmostEqual(p[2], L.CS_TOP, places=9)
            self.assertLessEqual(abs(p[0] - L.SEAT_X), L.CS_L / 2 - 0.008)
            self.assertLessEqual(abs(p[1] - L.SEAT_Y), L.CS_W / 2 - 0.008)

    def test_open_lever_is_tilted_back_and_clear_of_the_casting(self):
        for c in L.CLAMPS:
            tip = c.point(c.handle, L.HANDLE_OFFSET, 0.0)
            end = c.point(c.arm, 0.0, 0.0)
            for p in (tip, end):
                back = (p[:2] - np.array([c.px, c.py])) @ c.dir
                self.assertLess(back, -0.015, c.name)  # leaning away from the casting
            self.assertGreater(c.pad_bottom(0.0)[2], 0.09, c.name)

    def test_lever_shapes_follow_the_hinge(self):
        for c in L.CLAMPS:
            spec = T.SPECS[c.name]
            st = mk.state_from_poses(spec, (c.px, c.py, L.TOP, 0, 0, 0, 1), joints={S.HANDLE: L.Q_FLAT})
            pad = st.link_to_world(S.HANDLE, S.PAD_DRAWN[c.name]) - np.array([0, 0, L.TOP])
            self.assertTrue(np.allclose(pad, c.pad_bottom(L.Q_FLAT), atol=1e-9), c.name)
            d = float(st.distance(c.pad_bottom(L.Q_FLAT) + np.array([0.0, 0.0, L.TOP + 0.0005]))[0])
            self.assertLess(d, 0.0011)  # the pad's bottom face is at the point

    def test_shut_handles_do_not_cross(self):
        a = T.CLAMP1_PART
        b = T.CLAMP2_PART
        sa = mk.state_from_poses(a, (L.C1.px, L.C1.py, L.TOP, 0, 0, 0, 1), joints={S.HANDLE: L.Q_FLAT})
        sb = mk.state_from_poses(b, (L.C2.px, L.C2.py, L.TOP, 0, 0, 0, 1), joints={S.HANDLE: L.Q_FLAT})
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0.18, 0.40, 4000), rng.uniform(-0.12, 0.22, 4000),
                               L.TOP + rng.uniform(0.0, 0.10, 4000)])
        inside_a = sa.distance(pts) == 0.0
        inside_b = sb.distance(pts) == 0.0
        self.assertGreater(inside_a.sum(), 20)
        self.assertGreater(inside_b.sum(), 20)
        self.assertFalse(np.any(inside_a & inside_b))
        # and the handles stay 10 mm or more apart
        lo_a, hi_a = sa.link_bounds(S.HANDLE)
        lo_b, hi_b = sb.link_bounds(S.HANDLE)
        sep = np.maximum(lo_a - hi_b, lo_b - hi_a)
        self.assertGreater(float(sep.max()), 0.010)

    def test_push_force_is_within_the_arm(self):
        """The drive's pull toward open (its target trailing by OPEN_BIAS) plus the lag of a target set once a control
        step while the demo turns the handle (15 degrees in 0.9 s), over the corner's height above the pivot."""
        omega = math.radians(15.0) / T.ARC_STEP_S
        for c in L.CLAMPS:
            for psi in (-L.LEAN, 0.0, L.Q_OC - L.LEAN):
                a, z = L.block_corner(c, psi, T.S_PUSH[c.name])
                h = math.hypot(a, z - L.Z_PIVOT)
                torque = c.press_k * (c.open_bias + omega / L.CONTROL_HZ) + L.LEVER_MASS * L.G * 0.04
                self.assertLess(torque / h, 3.5, (c.name, psi))  # the arm gives about 5 N
            self.assertAlmostEqual(c.press_k * c.open_bias / c.grip_r, L.OPEN_FORCE)

    def test_shut_lever_presses_the_casting_hard(self):
        for c in L.CLAMPS:
            self.assertAlmostEqual(c.press_force(), L.PRESS_FORCE)
            self.assertGreater(c.press_force(L.Q_FLAT + math.radians(1.0)), 0.9 * L.PRESS_FORCE)  # 1.5 mm into it
            # it takes far more than a light lift on the grip to reopen it
            self.assertGreater(c.press_k * (L.Q_TARGET_SHUT - L.Q_FLAT) / c.grip_r, 20.0)
        self.assertGreater(L.Q_TARGET_SHUT, L.Q_UPPER)  # shut on nothing it rests on its far stop
        self.assertLess(L.Q_OC, L.Q_SHUT)
        self.assertLess(L.Q_RELEASE, L.Q_OC)

    def test_two_pads_hold_the_casting_against_a_20_n_proof_pull(self):
        """Friction 0.3: both pads at 25 N hold the casting against 20 N up or sideways away from the stops; one pad
        does not hold it sideways."""
        mu, w = L.CS_FRICTION[0], L.CS_MASS * L.G
        press = sum(c.press_force() for c in L.CLAMPS)
        self.assertGreater(press + w, 20.0)  # straight up
        self.assertGreater(mu * (press + w) + mu * press, 20.0)  # sideways
        up = side = 20.0 / math.sqrt(2.0)
        self.assertGreater(mu * (press + w - up) + mu * press, side)  # 45 degrees up and sideways
        one = L.C1.press_force()
        self.assertLess(mu * (one + w) + mu * one, 20.0)

    def test_contact_psi_inverts_block_corner(self):
        for c in L.CLAMPS:
            for deg in (-30, -10, 0, 20, 45, 60, 75):
                s = T.S_PUSH[c.name]
                a, z = L.block_corner(c, math.radians(deg), s)
                psi, s2 = L.contact_psi(c, a, z)
                self.assertIsNotNone(psi, (c.name, deg))
                self.assertAlmostEqual(math.degrees(psi), deg, places=6)
                self.assertAlmostEqual(s2, s, places=6)

    def test_corner_off_the_handle_is_no_contact(self):
        self.assertIsNone(L.contact_psi(L.C1, 0.2, 0.2)[0])
        self.assertIsNone(L.contact_psi(L.C1, -0.1, 0.30)[0])


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

    def test_half_closed_handle_is_not_locked(self):
        self.assertEqual(run(scene(q1=math.radians(60), q2=math.radians(60))).score, 0.25)
        self.assertEqual(run(scene(q1=math.radians(60))).score, 0.5)

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
            for q in (L.Q_FLAT, 0.0):
                self.assertEqual(run(scene(q1=q, q2=q, cs=cs)).score, 0.0, (label, q))
        for dx in (0.0, 0.0009, -0.0009):
            self.assertGreater(run(scene(cs=casting_at(x=L.SEAT_X + dx))).score, 0.0, dx)
        self.assertLessEqual(tol, 0.001)

    def test_shut_on_nothing_is_not_locked(self):
        self.assertEqual(run(scene(q1=L.Q_UPPER)).score, 0.5)  # past flat, the pad in the air: only C2 locked
        self.assertEqual(run(scene(q1=L.Q_SHUT + 0.001, q2=L.Q_FLAT)).score, 0.5)  # 7 mm over the casting
        r = run(scene(q1=L.Q_UPPER, q2=0.0))
        self.assertTrue(r.details["empty1"])
        self.assertEqual(r.score, 0.25)

    def test_shut_over_the_plate_not_the_casting(self):
        """A pad that is not over the casting's top face (the casting 12 mm along S2) is nothing under it."""
        shifted = casting_at(y=L.SEAT_Y + 0.012)  # not seated either
        self.assertEqual(run(scene(cs=shifted)).score, 0.0)

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

    def test_shut_threshold_is_eight_degrees(self):
        self.assertAlmostEqual(math.degrees(L.Q_FLAT - L.Q_SHUT), 8.0)
        self.assertTrue(S.shut(L.Q_FLAT - math.radians(7.9)))
        self.assertFalse(S.shut(L.Q_FLAT - math.radians(8.1)))


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_casting(self):
        arms = {"left": arm((L.SEAT_X, L.SEAT_Y + 0.02, L.TOP + 0.04), 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_hand_on_a_handle(self):
        c = L.C1
        p = c.point(0.05, 0.02, L.Q_FLAT) + np.array([0.0, 0.0, L.TOP + 0.02])
        arms = {"left": FAR["left"], "right": arm(p, 0.0)}
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
            res = tr.update(scene(cs=casting_at(z=SEAT_Z + 0.00004 * i)), FAR)  # 0.8 mm in a second
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
            res = tr.update(scene(q1=L.Q_FLAT + math.radians(0.25) * (i % 12)), FAR)  # 2.75 degrees of wobble
        self.assertEqual(res.score, 0.0)

    def test_handle_creep_under_two_degrees(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(q1=L.Q_FLAT + math.radians(0.05) * i), FAR)  # 1.1 degrees in a second
        self.assertEqual(res.score, 1.0)


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
        """Why no latch is needed: with even one pad pressing 25 N the casting needs over 15 N to slide (friction 0.3
        under and over it), well beyond the about 5 N an arm pushes sideways, so clamping first leaves it off its
        stop (not seated). This bounds the order loophole the scorer no longer latches."""
        mu, w = L.CS_FRICTION[0], L.CS_MASS * L.G
        for c in L.CLAMPS:
            self.assertGreater(mu * (c.press_force() + w) + mu * c.press_force(), 15.0)

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

    def test_clamp_shut_on_nothing_does_not_latch(self):
        tr = S.Tracker()
        res = run(scene(q1=L.Q_UPPER, q2=0.0), tracker=tr)
        self.assertFalse(tr.latched)
        self.assertEqual(res.score, 0.25)

    def test_reopened_clamp_after_crooked_scores_the_end_state(self):
        tr = S.Tracker()
        tr.update(scene(cs=casting_at(x=L.SEAT_X - 0.006)), FAR)
        run(scene(q1=0.0, q2=0.0), tracker=tr)
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

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
            r = tr.update(scene(q1=rng.uniform(0.0, L.Q_UPPER), q2=rng.uniform(0.0, L.Q_UPPER), cs=cs), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class ToggleRule(unittest.TestCase):
    def test_open_target_trails_the_lever(self):
        for c in L.CLAMPS:
            self.assertEqual(T.toggle_target(c, 0.0, None), (0.0, None))
            q = L.Q_OC - 0.01
            t, ramp = T.toggle_target(c, q, None)
            self.assertIsNone(ramp)
            self.assertAlmostEqual(t, q - c.open_bias)

    def test_past_over_centre_the_target_runs_on_to_its_shut_target(self):
        step = L.SNAP_RATE / L.CONTROL_HZ
        for c in L.CLAMPS:
            t, ramp = T.toggle_target(c, L.Q_OC, None)
            self.assertAlmostEqual(t, L.Q_OC + step)
            n = 0
            while t < L.Q_TARGET_SHUT and n < 100:
                t, ramp = T.toggle_target(c, L.Q_FLAT, ramp)  # the pad on the casting: the lever stays at flat
                n += 1
            self.assertAlmostEqual(t, L.Q_TARGET_SHUT)
            self.assertLess(n / L.CONTROL_HZ, 1.0)  # full press within 1 s of passing over-centre
            # a light disturbance keeps it shut; forced back below Q_RELEASE it is open again
            self.assertEqual(T.toggle_target(c, L.Q_OC - 0.01, ramp)[1], L.Q_TARGET_SHUT)
            self.assertIsNone(T.toggle_target(c, L.Q_RELEASE - 0.01, ramp)[1])

    def test_unreadable_pulls_open(self):
        for bad in (None, float("nan"), "x", float("inf")):
            self.assertEqual(T.toggle_target(L.C1, bad, 1.0), (0.0, None))

    def test_hinge_declaration(self):
        for part in (T.CLAMP1_PART, T.CLAMP2_PART):
            j = part.link(S.HANDLE)
            self.assertEqual(j.lower, 0.0)
            self.assertAlmostEqual(j.upper, L.Q_UPPER)
            self.assertGreater(j.spring, 1.0)  # stiff: the press comes from it
            self.assertGreater(L.Q_TARGET_SHUT, j.upper)
            self.assertIsNone(part.mass)  # fixed base


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
        """Every open/close change takes 1.5 s or more (the gripper moves 0.05 per step)."""
        for a, wps in T.plans("").items():
            for x, y in zip(wps, wps[1:]):
                if abs(x.finger - y.finger) > 0.03:
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
        """The right arm leaves its waiting spot only once the left arm has started to move aside from C2."""
        plan = T.plans("")
        t_left_aside = sum(w.duration for w in plan["left"][:-1])
        t_right_moves = plan["right"][0].duration
        self.assertAlmostEqual(t_left_aside, t_right_moves)

    def test_carry_clears_the_clamps(self):
        """The casting's underside clears each clamp's whole box (levers open) by 5 mm over its footprint while the left
        arm carries it, sampled along every straight segment from the grip to the set-down."""
        wps = T.plans("")["left"]
        i0 = next(i for i, w in enumerate(wps) if w.label.startswith("lift"))
        i1 = next(i for i, w in enumerate(wps) if w.label.startswith("set it down"))
        boxes = [T.SPECS[n].world_bounds() for n in (S.CLAMP1, S.CLAMP2)]
        for a, b in zip(wps[i0 - 1:i1], wps[i0:i1 + 1]):
            for f in np.linspace(0.0, 1.0, 41):
                p = np.array(a.pos) * (1 - f) + np.array(b.pos) * f
                bottom = p[2] - L.TOP - T.GRIP_TO_BOTTOM
                x0, x1 = p[0] - L.CS_L / 2, p[0] + L.CS_L / 2
                y0, y1 = p[1] - L.CS_W / 2, p[1] + L.CS_W / 2
                for lo, hi in boxes:
                    over = x1 > lo[0] and x0 < hi[0] and y1 > lo[1] and y0 < hi[1]
                    if over:
                        self.assertGreaterEqual(bottom, hi[2] - L.TOP + 0.005, (b.label, f))

    def test_set_down_is_over_the_plate_and_short_of_the_stops(self):
        wps = {w.label: w for w in T.plans("")["left"]}
        w = wps["set it down"]
        self.assertAlmostEqual(w.pos[0], L.PLACE_X)
        self.assertAlmostEqual(w.pos[1], L.PLACE_Y)
        bottom = w.pos[2] - L.TOP - T.GRIP_TO_BOTTOM
        self.assertAlmostEqual(bottom - L.PLATE_H, 0.003, places=6)
        self.assertGreater(T.SHIFT_Z - T.GRIP_TO_BOTTOM, L.PLATE_H + L.STOP_H + 0.010)

    def test_pushes_reach_and_overshoot_the_stops(self):
        wps = {w.label: w for w in T.plans("")["left"]}
        p1 = wps["push it +x into S1"]
        face = p1.pos[0] + L.PAD_FACE
        self.assertGreater(face - (L.SEAT_X - L.CS_L / 2), 0.006)  # past where the near face ends seated
        p2 = wps["push it -y into S2"]
        face = p2.pos[1] - L.PAD_FACE
        self.assertGreater((L.SEAT_Y + L.CS_W / 2) - face, 0.006)
        # the first push line is inside S1's span and clear of C1's base
        self.assertLess(abs(T.PUSH1_Y), L.S1_Y1 - L.FIST_HALF + 0.02)
        self.assertGreater(T.PUSH1_Y - L.FIST_HALF, L.C1.py + L.BASE_HALF_ACROSS + 0.015)
        # the fist's pads clear the plate and reach the casting's side face
        self.assertGreater(T.PUSH_TIPS, L.PLATE_H + 0.010)
        self.assertLess(T.PUSH_TIPS, L.CS_TOP - 0.010)

    def test_lever_pushes_start_on_the_face_and_end_past_over_centre(self):
        for c in L.CLAMPS:
            first = T.arc_tcp(c, T.ARC_DEG[0])
            last = T.arc_tcp(c, T.ARC_DEG[-1])
            self.assertLess(T.ARC_DEG[0], 0.0)
            self.assertGreater(T.ARC_DEG[-1], math.degrees(L.Q_OC - L.LEAN) + 10.0)
            self.assertGreater(first[2], 0.08)  # the block is up on the grip, well clear of the plate
            self.assertGreater(last[2] - T.PAD_TIP, 0.07 + 0.01)  # ends above the shut handle's top (0.073 m)
            self.assertLessEqual(max(first[2], last[2]), 0.2)

    def test_demo_reaches_with_margin(self):
        """Every demo waypoint is where the arm reaches: inboard under 0.40 m, x within 0.14 to 0.35 m."""
        for side, wps in T.plans("").items():
            base_y = 0.305 if side == "left" else -0.305
            for w in wps:
                inboard = abs(w.pos[1] - base_y)
                self.assertLess(inboard, 0.40, (side, w.label))
                self.assertGreaterEqual(w.pos[0], 0.14, (side, w.label))
                self.assertLess(w.pos[0], 0.35, (side, w.label))

    def test_plan_yaws_cover_both_arms(self):
        self.assertEqual(set(T.PLAN_YAWS), {"left", "right"})
        self.assertAlmostEqual(sum(d for d, _ in T.PLAN_YAWS["left"]), sum(w.duration for w in T.plans("")["left"]))


class OfflineModel(unittest.TestCase):
    def fresh(self):
        return T.OfflineModel()

    def feed(self, m, left=None, right=None, n=1):
        for _ in range(n):
            m.step({"left": left or FAR["left"], "right": right or FAR["right"]})

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

    def test_open_gripper_carries_nothing(self):
        m = self.fresh()
        self.feed(m, left=arm((L.START_X, L.START_Y, L.TOP + T.GRIP_Z), 1.0))
        self.feed(m, left=arm((0.25, 0.10, L.TOP + 0.1), 1.0))
        self.assertAlmostEqual(m.cx, L.START_X)

    def test_pushes_stop_at_the_stops(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.PLACE_X, L.PLACE_Y, SEAT_Z
        m.steps = 1  # left plan yaw at t ~ 0 is x-closing
        tips = L.PLATE_H + 0.014
        for x in np.linspace(L.PLACE_NEAR_X - 0.02, L.PLACE_NEAR_X + 0.06, 40):
            m.steps = 490  # inside the x-closing push of the demo
            m.step({"left": arm((x, 0.03, L.TOP + tips + T.PAD_TIP), 0.0), "right": FAR["right"]})
        self.assertAlmostEqual(m.cx, L.SEAT_X)

    def test_high_fist_does_not_push(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.PLACE_X, L.PLACE_Y, SEAT_Z
        for x in np.linspace(0.2, 0.30, 30):
            m.steps = 500
            m.step({"left": arm((x, 0.03, L.TOP + 0.12), 0.0), "right": FAR["right"]})
        self.assertAlmostEqual(m.cx, L.PLACE_X)

    def test_lever_pushed_along_its_arc_shuts_and_back_off_early_springs_open(self):
        m = self.fresh()
        m.cx, m.cy, m.cz = L.SEAT_X, L.SEAT_Y, SEAT_Z
        c = L.C1
        right_t = sum(w.duration for w in T.plans("")["right"][:4]) * HZ + 5
        for deg in (-30, -15, 0, 20, 40):
            p = T.arc_tcp(c, deg)
            m.steps = int(right_t)
            m.step({"left": FAR["left"], "right": arm((p[0], p[1], L.TOP + p[2]), 0.0)})
        self.assertFalse(m.shut[S.CLAMP1])
        m.steps = int(right_t)
        m.step(FAR)
        self.assertAlmostEqual(m.psi[S.CLAMP1], -L.LEAN)  # let go short of over-centre: back to open
        for deg in (-30, -15, 0, 20, 40, 60, 72):
            p = T.arc_tcp(c, deg)
            m.steps = int(right_t)
            m.step({"left": FAR["left"], "right": arm((p[0], p[1], L.TOP + p[2]), 0.0)})
        self.assertTrue(m.shut[S.CLAMP1])
        self.assertAlmostEqual(m.q(S.CLAMP1), L.Q_FLAT)

    def test_lever_shut_with_nothing_under_the_pad_goes_to_the_far_stop(self):
        m = self.fresh()  # the casting still on the staging block
        c = L.C1
        right_t = sum(w.duration for w in T.plans("")["right"][:4]) * HZ + 5
        for deg in (-30, -15, 0, 20, 40, 60, 72):
            p = T.arc_tcp(c, deg)
            m.steps = int(right_t)
            m.step({"left": FAR["left"], "right": arm((p[0], p[1], L.TOP + p[2]), 0.0)})
        self.assertAlmostEqual(m.q(S.CLAMP1), L.Q_UPPER)

    def test_hovering_over_the_handle_does_not_shut_it(self):
        m = self.fresh()
        p = T.arc_tcp(L.C2, 72)
        for _ in range(5):
            m.steps = 700
            m.step({"left": arm((p[0], p[1], L.TOP + p[2] + 0.05), 0.0), "right": FAR["right"]})
        self.assertFalse(m.shut[S.CLAMP2])

    def test_full_demo_by_the_model_scores_one(self):
        """The same check the offline checker runs, driven here from the plan's own TCP positions."""
        plan = T.plans("")
        n = int(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ) + 3 * int(HZ)
        m = self.fresh()
        tr = S.Tracker()
        res = None

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

        for k in range(1, n + 1):
            snap = {}
            for side in ("left", "right"):
                if side in plan:
                    p, fin = tcp_at(plan[side], k / HZ)
                    snap[side] = arm(p, min(1.0, max(0.0, fin / -0.04695)))
                else:
                    snap[side] = FAR[side]
            rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in snap.items()})
            m.step(snap)
            res = tr.update(m.states(), snap)
        self.assertEqual(res.score, 1.0, res.details)

    def test_left_alone_scores_half_and_right_alone_zero(self):
        for variant_idle, want in (("right", 0.5), ("left", 0.0)):
            plan = T.plans("")
            plan.pop(variant_idle)
            n = int(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ) + 3 * int(HZ)
            m = self.fresh()
            tr = S.Tracker()
            res = None
            for k in range(1, n + 1):
                snap = dict(FAR)
                for side, wps in plan.items():
                    acc = 0.0
                    prev = None
                    for w in wps:
                        if k / HZ <= acc + w.duration + 1e-9:
                            f = (k / HZ - acc) / w.duration
                            f = f * f * (3 - 2 * f)
                            p0 = np.array(prev.pos if prev else w.pos)
                            fin0 = prev.finger if prev else w.finger
                            snap[side] = arm(p0 * (1 - f) + np.array(w.pos) * f,
                                             min(1.0, max(0.0, (fin0 * (1 - f) + w.finger * f) / -0.04695)))
                            break
                        acc += w.duration
                        prev = w
                    else:
                        snap[side] = arm(wps[-1].pos, min(1.0, max(0.0, wps[-1].finger / -0.04695)))
                rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening}
                                       for s, a in snap.items()})
                m.step(snap)
                res = tr.update(m.states(), snap)
            self.assertEqual(res.score, want, variant_idle)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("x 220 to 380 mm", "y -160 to +160 mm", "12 mm thick", "x 330 to 345 mm", "y -45 to +45 mm",
                      "x 280 to 320 mm", "y -60 to -50 mm", "x 395 to 403 mm", "140 mm tall", "54 mm above the table",
                      "pivot x 236 mm, y -25 mm", "arm 62 mm", "x 288 mm", "handle 70 mm", "pivot x 300 mm, y +125 mm",
                      "arm 99 mm", "y +34 mm", "handle 90 mm", "30 degrees back", "55 degrees from vertical",
                      "15 degrees past flat", "42 mm above the table", "100 (x) x 150 (y) x 20 mm", "x 210 to 310 mm",
                      "y +170 to +320 mm", "60 (x) x 100 (y) x 30 mm", "0.18 kg", "x 260 mm, y +245 mm", "HDPE",
                      "1.5 N", "25 N", "27 N", "34 N"):
            self.assertIn(token, setup)
        for token in ("1 degree", "1.5 mm", "1.0 mm", "90 mm face", "40 mm face", "8 degrees", "0.5 mm above it",
                      "4 mm inside", "30 mm", "0.5 mm", "at most 1 mm", "at most 1 degree", "over 2 degrees", "25 N",
                      "shut on nothing is not locked",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "2800 steps (140 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_numbers_in_setup_match_the_layout(self):
        self.assertAlmostEqual(L.C1.px, 0.236)
        self.assertAlmostEqual(L.C1.py, -0.025)
        self.assertAlmostEqual(L.C1.arm, 0.062)
        self.assertAlmostEqual(L.C1.pad_s, 0.052)
        self.assertAlmostEqual(L.C1.handle, 0.070)
        self.assertAlmostEqual(L.C2.px, 0.300)
        self.assertAlmostEqual(L.C2.py, 0.125)
        self.assertAlmostEqual(L.C2.arm, 0.099)
        self.assertAlmostEqual(L.C2.pad_s, 0.091)
        self.assertAlmostEqual(L.C2.handle, 0.090)
        self.assertAlmostEqual(L.Z_PIVOT, 0.054)
        self.assertAlmostEqual(math.degrees(L.LEAN), 30.0)
        self.assertAlmostEqual(math.degrees(L.Q_OC - L.LEAN), 55.0)
        self.assertAlmostEqual(math.degrees(L.Q_UPPER - L.Q_FLAT), 15.0)
        self.assertAlmostEqual(math.degrees(L.Q_TARGET_SHUT - L.Q_FLAT), 40.0)
        self.assertAlmostEqual(L.PRESS_FORCE, 25.0)
        self.assertAlmostEqual(L.OPEN_FORCE, 1.5)
        self.assertAlmostEqual(L.CS_TOP, 0.042)

    def test_fidelity_states_the_rule_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Declared rule", "55 degrees", "40 degrees past flat", "25 N", "1.5 N", "est", "one lever"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("51-4041.00", "51-9161.00", "51-4121.06", "288,300", "170,000", "437,700", "896,000"):
            self.assertIn(token, ctx)

    def test_context_total_and_names_no_other_task(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        self.assertEqual(288300 + 170000 + 437700, 896000)
        self.assertNotIn("TK-", ctx)

    def test_setup_forces_match_the_layout(self):
        self.assertEqual(round(L.C1.press_k * (L.Q_TARGET_SHUT - L.Q_FLAT) / L.C1.grip_r), 27)
        self.assertEqual(round(L.C2.press_k * (L.Q_TARGET_SHUT - L.Q_FLAT) / L.C2.grip_r), 34)

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
        for token in ("inspect-robots run", "Runtime", "Seed", "Bench"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
