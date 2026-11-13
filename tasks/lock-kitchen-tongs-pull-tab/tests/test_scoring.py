"""Offline tests for layout.py, scoring.py, offline_kinematics.py and task.py: synthetic tongs states (mechkit
AssemblyStates from the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
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


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline stand-in

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_kinematics as OK  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
from yam_arena import demo as D  # noqa: E402

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
SPEC = T.TONGS_SPEC
AWAY = {a: rigstate.Arm(np.array([0.25, s * 0.25, L.TOP + 0.25]), np.array([0.25, s * 0.25, L.TOP + 0.237]), 0.6)
        for a, s in (("left", 1.0), ("right", -1.0))}
LOCK = L.lock_angle()
SHUT_Q = L.OPEN  # hinge reading with the arms fully shut
LOCKED_Q = L.OPEN - LOCK  # resting on the lock


# -- states --------------------------------------------------------------------------------------------------------


def tongs(q_hinge=0.0, q_tab=0.0, origin=None, yaw=None, dz=0.0, r=None):
    o = np.array(L.ORIGIN if origin is None else origin, dtype=float)
    o[2] = L.TOP + dz
    rot = mk.yaw_quat(L.YAW if yaw is None else yaw) if r is None else mk.matrix_to_quat(r)
    return {S.TONGS: mk.state_from_poses(SPEC, (*o, *rot), joints={S.HINGE: q_hinge, S.TAB: q_tab},
                                         source="offline")}


def locked(q_tab=L.TAB_TRAVEL, **kw):
    kw.setdefault("yaw", L.YAW_HELD)
    return tongs(LOCKED_Q, q_tab, **kw)


def arm_at(point, opening=0.3):
    p = np.asarray(point, dtype=float)
    return rigstate.Arm(p + np.array([0.0, 0.0, 0.013]), p, opening)


def with_arm(side, point, opening=0.3):
    arms = dict(AWAY)
    arms[side] = arm_at(point, opening)
    return arms


def run(seq, tracker=None):
    """Feed (states, arms) pairs; return the tracker."""
    tr = tracker or S.Tracker()
    for states, arms in seq:
        tr.update(states, arms)
    return tr


def hold(states, arms, n):
    return [(states, arms)] * n


def squeezed_then(final, arms_final=AWAY, squeeze_steps=WIN, end_steps=2 * WIN + 2):
    """A 1 s squeeze (arms shut, a gripper on them), then ``final`` for 2 s."""
    on = with_arm("right", (*L.HOLD_XY, L.TOP + 0.006), 0.25)
    return hold(tongs(SHUT_Q, 0.0, yaw=L.YAW_HELD), on, squeeze_steps) + hold(final, arms_final, end_steps)


# -- the lock's geometry (the design's numbers) ----------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_tab_travel_against_angle(self):
        for deg, mm in ((27.5, 3.77), (20, 4.68), (10, 5.54), (5, 5.82), (4, 6.25), (3.5, 8.31), (3.0, 11.04),
                        (2.9, 11.70), (2.8, 12.0)):
            self.assertAlmostEqual(L.travel_limit(math.radians(deg)) * 1000, mm, delta=0.03, msg=deg)

    def test_eleven_mm_only_within_three_degrees(self):
        self.assertLess(L.travel_limit(math.radians(3.02)), 0.011)
        self.assertGreaterEqual(L.travel_limit(math.radians(3.0)), 0.011)
        self.assertLessEqual(L.tip_gap(math.radians(3.0)), 0.0093)

    def test_lock_angle_and_locked_tip_gap(self):
        self.assertAlmostEqual(math.degrees(LOCK), 2.86, delta=0.01)
        self.assertAlmostEqual(L.tip_gap(LOCK) * 1000, 8.8, delta=0.05)
        self.assertLess(L.tip_gap(LOCK), L.SHUT_TIP_GAP - 0.005)  # about 5 mm under the limit

    def test_lug_load_on_the_block(self):
        torque = L.SPRING_RATE * (LOCK - L.OPEN - L.SPRING_TARGET)  # spring at the lock angle
        self.assertAlmostEqual(abs(L.SPRING_RATE * (L.OPEN - LOCK - L.SPRING_TARGET)), 0.379, delta=0.002)
        self.assertGreater(abs(torque), 0)
        self.assertAlmostEqual(0.379 / 0.020, 19.0, delta=0.1)

    def test_rest_on_the_block_edge_and_behind_it(self):
        for t, deg in ((0.006, 4.07), (0.008, 3.57), (0.010, 3.17), (0.0109, 3.02)):
            a, on = L.rest_angle(t)
            self.assertTrue(on, t)
            self.assertAlmostEqual(math.degrees(a), deg, delta=0.02)
            self.assertLessEqual(L.tip_gap(a), L.SHUT_TIP_GAP)  # resting on the edge reads shut ...
            self.assertLess(L.tab_gap(t), L.ENGAGED_TAB_GAP)  # ... but not locked
        for t in (0.0, 0.003, 0.005, 0.0058):
            self.assertEqual(L.rest_angle(t), (L.OPEN, False), t)

    def test_spring_numbers(self):
        self.assertAlmostEqual(L.SPRING_RATE, 0.417, delta=0.001)
        self.assertAlmostEqual(L.SPRING_RATE * (0.0 - L.SPRING_TARGET), 0.20, delta=1e-9)  # at the open stop
        self.assertAlmostEqual(L.SPRING_RATE * (L.OPEN - L.SPRING_TARGET), 0.40, delta=1e-9)  # shut
        self.assertAlmostEqual(0.40 / (L.HOLD_U / 1000), 4.44, delta=0.01)  # 4.4 N at 9 cm from the pin
        self.assertAlmostEqual(0.40 / 0.176, 2.27, delta=0.01)  # 2.3 N at the tips
        self.assertAlmostEqual(L.tip_gap(L.OPEN) * 1000, 83.7, delta=0.05)

    def test_heel_clearance_over_the_parked_block(self):
        vertical, normal = L.heel_clearance_open()
        self.assertAlmostEqual(vertical, 1.62, delta=0.03)
        self.assertAlmostEqual(normal, 1.43, delta=0.03)

    def test_tab_clear_of_the_free_arm_at_every_hinge_angle_with_the_tab_in(self):
        """Self-collision is on between the tab and the free arm: with the tab in, nothing of the tab may touch the
        free arm anywhere between open and shut (the lugs only ever meet the block when the tab is drawn)."""
        pts = []
        for body in (S.TAB,):
            for s in SPEC.body_shapes(body):
                c, h = np.asarray(s.center), np.asarray(s.size) / 2
                g = np.stack(np.meshgrid(*[np.linspace(-1, 1, 7)] * 3, indexing="ij"), -1).reshape(-1, 3)
                pts.append(c + g * h)
        pts = np.concatenate(pts)
        for q in np.linspace(0.0, L.OPEN, 25):
            st = tongs(q, 0.0, origin=(0.0, 0.0, 0.0), yaw=L.YAW_HELD)[S.TONGS]
            r, p = st.link_pose(S.TAB)
            world = (pts - SPEC.body_offset(S.TAB)) @ r.T + p
            gap = float(st.distance(world, bodies=(S.HINGE,)).min())
            self.assertGreater(gap, 0.0013, q)

    def test_masses_recompute(self):
        self.assertAlmostEqual(L.PIN_ARM_MASS + L.FREE_ARM_MASS + L.TAB_MASS, L.MASS, delta=1e-9)
        sheet = 200 * 32 * 1 / 1000 * 7.9  # g: 1 mm stainless, 200 x 32 mm
        self.assertAlmostEqual(sheet, 50.6, delta=0.1)
        block = 18 * 10 * 10.6 / 1000 * 7.85
        ring = (24 * 25 * 8 - 12 * 13 * 8) / 1000 * 1.15
        stem = 21 * 6 * 3 / 1000 * 1.15
        self.assertAlmostEqual(block + ring + stem, 19.5, delta=0.1)
        self.assertAlmostEqual(SPEC.body_mass("base") + sum(SPEC.body_mass(j.name) for j in SPEC.joints), L.MASS)

    def test_start_pose_numbers_in_setup(self):
        tips = L.to_world([(176.0, 14.0)])[0]
        free_tip = L.to_world(L.open_free([(176.0, 14.0)], L.OPEN))[0]
        ring = L.to_world([L.RING_CENTRE_UW])[0]
        for got, want in ((tips, (0.020, -0.042)), (free_tip, (0.018, 0.042)), (ring, (0.233, 0.004))):
            np.testing.assert_allclose(got, want, atol=0.0006)
        self.assertEqual(L.ORIGIN[:2], (0.190, 0.005))
        np.testing.assert_allclose(L.RING_XY_HELD, (0.2315, -0.006), atol=1e-9)
        np.testing.assert_allclose(L.RING_GRIP_XY, (0.240, -0.006), atol=1e-9)
        np.testing.assert_allclose(L.HOLD_XY, (0.100, 0.000), atol=1e-9)
        self.assertAlmostEqual(L.HOLD_U - L.RING_CENTRE_UW[0], 131.5)  # "the tab is 13 cm from that hold"

    def test_everything_on_the_worktop(self):
        st = tongs()[S.TONGS]
        self.assertGreater(st.low[0], -0.19)  # the station worktop: x -0.19 to 0.91, y -0.65 to 0.65
        self.assertLess(st.high[0], 0.91)
        self.assertLess(max(abs(st.low[1]), abs(st.high[1])), 0.65)
        self.assertGreater(st.low[0], 0.0)  # in front of the arms' base line

    def test_open_arms_fit_the_open_gripper_at_the_hold(self):
        """Under the 27 mm pads at the hold (pads 27 mm wide along x, closing along y) the open arms' outer faces
        span at most 80 mm (the grasp envelope), with 7 mm or more to each 94 mm open pad (geo5)."""
        us = np.linspace(-24.0, 146.0, 17001)
        pin_out = L.to_world(np.c_[us, np.zeros_like(us)])
        free_out = L.to_world(L.open_free(np.c_[us, np.full_like(us, 28.0)], L.OPEN))
        hx, hy = L.HOLD_XY
        spans = []
        for x in np.linspace(hx - 0.0135, hx + 0.0135, 7):
            a = pin_out[np.argmin(abs(pin_out[:, 0] - x)), 1]  # the pin arm, on the -y side
            b = free_out[np.argmin(abs(free_out[:, 0] - x)), 1]
            spans.append(b - a)
            self.assertLessEqual(b - a, 0.080, x)
            self.assertGreaterEqual(min(a - (hy - 0.047), (hy + 0.047) - b), 0.007, x)
        self.assertAlmostEqual(max(spans), 0.0795, delta=0.0005)
        held = L.to_world([(L.HOLD_U, L.CENTRE_W)], yaw=L.YAW_HELD)[0]
        np.testing.assert_allclose(held, L.HOLD_XY, atol=1e-6)


# -- the kit -----------------------------------------------------------------------------------------------------------


class Kit(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        kit = None
        for p in HERE.parents:
            if (p / "tools" / "kit" / "mechkit.py").is_file():
                kit = p / "tools" / "kit"
                break
        if kit is None:
            self.skipTest("tools/kit not found")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             hashlib.sha256((kit / name).read_bytes()).hexdigest(), name)
        self.assertEqual(mk.VERSION, "2")

    def test_self_collision_on_for_the_tongs_only(self):
        plan = mk.usd_plan(SPEC, "/World/tongs")
        self.assertTrue(plan["self_collisions"])
        other = mk.assembly("other", (0.3, 0.0, L.TOP), [mk.box((0, 0, 0.01), (0.02, 0.02, 0.02))], mass=0.1,
                            joints=[mk.slider("s", "X", [mk.box((0.0, 0.0, 0.03), (0.01, 0.01, 0.01))], upper=0.01)])
        self.assertFalse(mk.usd_plan(other, "/World/other")["self_collisions"])
        # the author step reads usd_plan from mechkit's own module
        self.assertIs(mk._author.__globals__["usd_plan"], T._usd_plan_self_collisions)

    def test_joints(self):
        hinge, tab = SPEC.link(S.HINGE), SPEC.link(S.TAB)
        self.assertEqual((hinge.kind, hinge.axis, hinge.lower), ("revolute", "Z", 0.0))
        self.assertAlmostEqual(hinge.upper, L.OPEN)
        self.assertLess(hinge.upper, 2 * math.pi)
        self.assertLess(abs(L.SPRING_TARGET), 2 * math.pi)
        self.assertEqual((tab.kind, tab.axis, tab.upper, tab.friction), ("prismatic", "-Y", 0.012, 1.0))
        self.assertGreaterEqual(tab.friction, 0.5)
        self.assertEqual(SPEC.surface_friction(), (0.6, 0.6))
        self.assertEqual(SPEC.contact_offset, 0.0005)
        self.assertEqual(SPEC.solver_iterations, 32)

    def test_shut_pose_closes_the_heads(self):
        st = tongs(SHUT_Q, 0.0, origin=(0.0, 0.0, 0.0), yaw=0.0)[S.TONGS]
        lo, hi = st.link_bounds(S.HINGE)
        self.assertAlmostEqual(hi[0], (28 - L.PIN_W) * L.MM, delta=1e-6)  # the free arm's outer face at w 28

    def test_resting_on_the_worktop(self):
        st = tongs()[S.TONGS]
        self.assertAlmostEqual(st.low[2], L.TOP, delta=1e-9)
        self.assertAlmostEqual(st.high[2], L.TOP + 0.030, delta=1e-9)
        ring = [s for s in SPEC.body_shapes(S.TAB)]
        self.assertAlmostEqual(min(np.asarray(s.center)[2] - s.size[2] / 2 for s in ring), 0.003, delta=1e-9)

    def test_spring_preload_is_set_every_step_and_reset(self):
        hub = mk._hub()
        hub.targets.pop(S.TONGS, None)
        T._spring_preload({})
        self.assertAlmostEqual(hub.targets[S.TONGS][S.HINGE], L.SPRING_TARGET)
        hub.targets.pop(S.TONGS, None)
        T._spring_preload_reset()
        self.assertAlmostEqual(hub.targets[S.TONGS][S.HINGE], L.SPRING_TARGET)


# -- the scorer --------------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_nothing_scores_zero(self):
        tr = run(hold(tongs(), AWAY, 100))
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.squeezed)

    def test_full_score(self):
        tr = run(squeezed_then(locked()))
        self.assertEqual(tr.last.score, 1.0)
        self.assertTrue(tr.last.success)

    def test_full_score_needs_no_squeeze_flag(self):
        """The lock itself can only be produced with the arms shut; the 1.0 tier reads the end state."""
        tr = run(hold(locked(), AWAY, 2 * WIN))
        self.assertEqual(tr.last.score, 1.0)

    def test_squeeze_tiers(self):
        open_ = tongs(yaw=L.YAW_HELD)  # let go after a symmetric squeeze: the root stays along y, the free arm opens
        for n, want in ((WIN, 0.25), (WIN + 5, 0.25), (WIN - 1, 0.0), (WIN - 2, 0.0)):
            tr = run(squeezed_then(open_, squeeze_steps=n))
            self.assertEqual(tr.last.score, want, n)

    def test_squeeze_held_to_the_end_tab_in(self):
        on = with_arm("right", (*L.HOLD_XY, L.TOP + 0.006), 0.25)
        tr = run(hold(tongs(SHUT_Q, 0.0, yaw=L.YAW_HELD), on, 3 * WIN))
        self.assertEqual(tr.last.score, 0.25)

    def test_pressed_shut_against_the_worktop_counts_as_a_squeeze(self):
        on = with_arm("left", (0.33, 0.05, L.TOP + 0.030), 0.0)
        tr = run(hold(tongs(SHUT_Q, 0.0, yaw=L.YAW_HELD), on, WIN) + hold(tongs(yaw=L.YAW_HELD), AWAY, 2 * WIN))
        self.assertEqual(tr.last.score, 0.25)

    def test_locked_but_still_pinched(self):
        on = with_arm("right", (*L.HOLD_XY, L.TOP + 0.006), 0.25)
        tr = run(squeezed_then(locked(), on))
        self.assertEqual(tr.last.score, 0.5)

    def test_fingertip_just_inside_and_outside_three_cm(self):
        st = locked()[S.TONGS]
        top = np.array([L.HOLD_XY[0], L.HOLD_XY[1], 0.0])
        # straight above the arms: find the height at which the finger zone is 29 and 31 mm clear
        for want, score in ((0.029, 0.5), (0.031, 1.0)):
            lo, hi = L.TOP, L.TOP + 0.3
            for _ in range(60):
                mid = (lo + hi) / 2
                a = arm_at(top + np.array([0, 0, mid]), 0.6)
                g = rigstate.zone_gap(a, st.distance)
                lo, hi = (mid, hi) if g < want else (lo, mid)
            arms = with_arm("left", top + np.array([0, 0, hi]), 0.6)
            tr = run(squeezed_then(locked(), arms))
            self.assertEqual(tr.last.score, score, want)

    def test_let_go_must_hold_for_the_whole_last_second(self):
        on = with_arm("right", (*L.HOLD_XY, L.TOP + 0.006), 0.25)
        tr = run(hold(locked(), on, WIN) + hold(locked(), AWAY, WIN - 1))
        self.assertEqual(tr.last.score, 0.5)
        tr.update(locked(), AWAY)
        self.assertEqual(tr.last.score, 1.0)

    def test_still_sliding_or_rocking(self):
        seq = [(locked(origin=(0.335, -0.085 - 0.00015 * k, 0)), AWAY) for k in range(3 * WIN)]  # 3 mm / s
        self.assertEqual(run(seq).last.score, 0.5)
        seq = [(locked(yaw=math.radians(0.15 * k)), AWAY) for k in range(3 * WIN)]  # 3 deg / s
        self.assertEqual(run(seq).last.score, 0.5)
        seq = [(locked(origin=(0.335, -0.085 - 0.00009 * k, 0)), AWAY) for k in range(3 * WIN)]  # 1.8 mm / s
        self.assertEqual(run(seq).last.score, 1.0)

    def test_gaps_drifting_in_the_last_second(self):
        """The tip gap drifting 1.1 mm in the last second (locked, let go): 0.5; 0.9 mm: 1.0. The tab cannot drift
        1.1 mm and still be within 1 mm of its stop, so for the tab the case is a 1.0 mm swing (inside)."""
        def tips_drift(total_mm):
            seq = hold(locked(), AWAY, WIN)
            for k in range(WIN + 1):
                gap = L.tip_gap(LOCK) + total_mm / 1000 * k / WIN
                seq.append((tongs(L.OPEN - 2 * math.asin(gap / (2 * 0.176)), L.TAB_TRAVEL, yaw=L.YAW_HELD), AWAY))
            return run(seq).last.score
        self.assertEqual(tips_drift(1.1), 0.5)
        self.assertEqual(tips_drift(0.9), 1.0)
        seq = hold(locked(), AWAY, WIN) + [(locked(q_tab=L.TAB_TRAVEL - (0.001 if k % 2 else 0.0)), AWAY)
                                           for k in range(WIN + 1)]
        self.assertEqual(run(seq).last.score, 1.0)
        seq = hold(locked(), AWAY, WIN) + [(locked(q_tab=L.TAB_TRAVEL - (0.0011 if k % 2 else 0.0)), AWAY)
                                           for k in range(WIN + 1)]
        self.assertEqual(run(seq).last.score, 0.5)  # a 1.1 mm swing of the tab: locked at the end, not still

    def test_held_up_in_the_air(self):
        for dz in (0.0015, 0.005, 0.05):
            self.assertEqual(run(squeezed_then(locked(dz=dz))).last.score, 0.5, dz)
        self.assertEqual(run(squeezed_then(locked(dz=0.0009))).last.score, 1.0)

    def test_lying_tilted_on_something(self):
        r = mk.quat_to_matrix(mk.quat_from_ypr(0.0, math.radians(8.0), 0.0))
        self.assertEqual(run(squeezed_then(locked(r=r, dz=0.004))).last.score, 0.5)

    def test_partly_over_the_worktop_edge_resting_on_it(self):
        self.assertEqual(run(squeezed_then(locked(origin=(0.86, -0.085, 0)))).last.score, 1.0)

    def test_fallen(self):
        self.assertEqual(run(squeezed_then(locked(dz=-0.0101))).last.score, 0.0)
        self.assertEqual(run(squeezed_then(tongs(dz=-0.30, yaw=L.YAW_HELD))).last.score, 0.0)
        self.assertEqual(run(squeezed_then(locked(dz=-0.30))).last.score, 0.0)
        self.assertEqual(run(squeezed_then(locked(dz=-0.009))).last.score, 0.5)  # 9 mm low: not fallen, not on top

    def test_near_misses(self):
        just_short = L.ENGAGED_TAB_GAP - 0.0001 - L.TAB_GAP_IN  # tab gap 15.9 mm
        self.assertEqual(run(squeezed_then(tongs(LOCKED_Q, just_short, yaw=L.YAW_HELD))).last.score, 0.25)
        wide = 2 * math.asin(0.0141 / (2 * 0.176))  # tip gap 14.1 mm with the tab at its stop (scorer only)
        self.assertEqual(run(squeezed_then(tongs(L.OPEN - wide, L.TAB_TRAVEL, yaw=L.YAW_HELD))).last.score, 0.25)
        exact = 2 * math.asin(0.0140 / (2 * 0.176))
        self.assertEqual(run(squeezed_then(tongs(L.OPEN - exact, L.TAB_TRAVEL, yaw=L.YAW_HELD))).last.score, 1.0)
        at16 = tongs(LOCKED_Q, L.ENGAGED_TAB_GAP - L.TAB_GAP_IN, yaw=L.YAW_HELD)
        self.assertEqual(run(squeezed_then(at16)).last.score, 1.0)  # 16.0 mm counts (boundary inside)

    def test_tab_pulled_with_the_arms_open_and_towed(self):
        seq = [(tongs(0.0, min(0.0038, 0.0002 * k), origin=(0.335, -0.085 - 0.0005 * k, 0)),
                with_arm("right", (0.334, -0.128 - 0.0005 * k, L.TOP + 0.004), 0.05)) for k in range(30)]
        seq += hold(tongs(0.0, 0.0038, origin=(0.335, -0.10, 0)), AWAY, 2 * WIN)
        self.assertEqual(run(seq).last.score, 0.0)

    def test_tab_on_the_block_edge(self):
        for t in (0.006, 0.008, 0.0105):
            a, _ = L.rest_angle(t)
            self.assertEqual(run(squeezed_then(tongs(L.OPEN - a, t, yaw=L.YAW_HELD))).last.score, 0.25, t)
            self.assertEqual(run(hold(tongs(L.OPEN - a, t, yaw=L.YAW_HELD), AWAY, 2 * WIN)).last.score, 0.25, t)

    def test_locked_then_pushed_back_in(self):
        tr = run(squeezed_then(locked()))
        self.assertEqual(tr.last.score, 1.0)
        run(hold(tongs(SHUT_Q, 0.0, yaw=L.YAW_HELD), with_arm("right", (0.32, -0.13, L.TOP + 0.004), 0.05), 10)
            + hold(tongs(0.0, 0.0, yaw=L.YAW_HELD), AWAY, 2 * WIN), tr)
        self.assertEqual(tr.last.score, 0.25)

    def test_every_listed_score_is_reachable(self):
        pinch = with_arm("right", (*L.HOLD_XY, L.TOP + 0.006), 0.25)
        seen = {run(squeezed_then(locked())).last.score, run(squeezed_then(locked(), pinch)).last.score,
                run(squeezed_then(tongs(yaw=L.YAW_HELD))).last.score, run(hold(tongs(), AWAY, 50)).last.score}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})


class FailClosed(unittest.TestCase):
    def _final(self, arms):
        return run(squeezed_then(locked(), arms)).last.score

    def test_unreadable_arms_never_let_go(self):
        nan = rigstate.Arm(np.array([np.nan, 0, 1.0]), np.array([0.3, 0, 1.0]), 0.5)
        short = rigstate.Arm(np.array([0.3, 0.0]), np.array([0.3, 0, 1.0]), 0.5)
        for arms in (None, {}, {"left": AWAY["left"]}, {"left": AWAY["left"], "right": nan},
                     {"left": AWAY["left"], "right": short}, "garbage",
                     {"left": AWAY["left"], "right": rigstate.Arm(AWAY["right"].tcp, AWAY["right"].tips, np.nan)}):
            self.assertEqual(self._final(arms), 0.5, repr(arms)[:60])

    def test_distance_that_raises_is_not_let_go(self):
        st = locked()[S.TONGS]

        class Broken:
            def __getattr__(self, name):
                if name == "distance":
                    raise RuntimeError("no distance")
                return getattr(st, name)

        tr = run(hold({S.TONGS: Broken()}, AWAY, 3 * WIN))
        self.assertEqual(tr.last.score, 0.5)

    def test_missing_or_broken_state(self):
        tr = run(squeezed_then(locked()))
        self.assertEqual(tr.last.score, 1.0)
        tr.update({}, AWAY)
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.last.details["valid_state"])
        st = locked()[S.TONGS]
        bad = mk.state_from_poses(SPEC, (np.nan, 0, 0, 0, 0, 0, 1), joints={S.HINGE: 0.1, S.TAB: 0.0})
        tr.update({S.TONGS: bad}, AWAY)
        self.assertEqual(tr.last.score, 0.0)
        tr.update({S.TONGS: st}, AWAY)  # the windows start again: not still yet
        self.assertEqual(tr.last.score, 0.5)
        run(hold({S.TONGS: st}, AWAY, WIN), tr)
        self.assertEqual(tr.last.score, 1.0)

    def test_no_joint_readings(self):
        st = locked()[S.TONGS]

        class NoJoints:
            joints = {}

            def __getattr__(self, name):
                return getattr(st, name)

        self.assertEqual(run(hold({S.TONGS: NoJoints()}, AWAY, 50)).last.score, 0.0)


class Resets(unittest.TestCase):
    def test_reset_clears_the_squeeze(self):
        tr = run(squeezed_then(tongs(yaw=L.YAW_HELD)))
        self.assertTrue(tr.squeezed)
        tr.reset()
        self.assertFalse(tr.squeezed)
        run(hold(tongs(), AWAY, 50), tr)
        self.assertEqual(tr.last.score, 0.0)

    def test_teleport_back_to_the_start_is_a_new_trial(self):
        tr = run(squeezed_then(locked()))
        self.assertEqual(tr.last.score, 1.0)
        run(hold(tongs(), AWAY, 30), tr)
        self.assertEqual(tr.resets, 1)
        self.assertFalse(tr.squeezed)
        self.assertEqual(tr.last.score, 0.0)

    def test_a_natural_release_at_the_start_pose_is_not_a_reset(self):
        """The free arm alone pressed shut (root left at the start pose) and let go: the arms spring open at the
        start pose with no jump of the root or the tab, which must keep the squeeze."""
        on = with_arm("right", (0.10, 0.03, L.TOP + 0.006), 0.0)
        seq = hold(tongs(SHUT_Q, 0.0), on, WIN) + [(tongs(SHUT_Q / 2, 0.0), AWAY)] + hold(tongs(), AWAY, 2 * WIN)
        tr = run(seq)
        self.assertEqual(tr.resets, 0)
        self.assertEqual(tr.last.score, 0.25)

    def test_task_hooks_reset_on_both_signals(self):
        T._TRACKER.reset()
        T._TRACKER.squeezed = True
        T._new_episode()
        self.assertFalse(T._TRACKER.squeezed)
        T.OFFLINE.tab = 0.01
        T._spring_preload_reset()
        self.assertEqual(T.OFFLINE.tab, 0.0)


# -- the offline kinematics and the demo ----------------------------------------------------------------------------


def play(plans, steps=None, kin=None):
    """Play waypoint plans through the stub's ArmPlan targets (no IK) into the offline kinematics and a tracker:
    (final score, first step of a 1 s run of 1.0, kinematics)."""
    bases = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
    ap = {a: D.ArmPlan(list(w), bases[a]) for a, w in plans.items()}
    end = max((p.duration for p in ap.values()), default=0.0)
    n = steps or int(math.floor((end + 3.0) * HZ)) + 1
    kin = kin or OK.OfflineTongs()
    tr = S.Tracker()
    ok_run, success_at = 0, None
    for k in range(n):
        t = (k + 1) / HZ
        arms = {}
        for a in ("left", "right"):
            if a in ap:
                pos, rot, finger, _ = ap[a].target(t)
                tips = np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2]
                arms[a] = rigstate.Arm(np.asarray(pos), tips, float(np.clip(finger / T.FULL_OPEN, 0, 1)))
            else:
                arms[a] = AWAY[a]
        kin.step(arms)
        o, yaw = kin.pose()
        st = mk.state_from_poses(SPEC, (*o, *mk.yaw_quat(yaw)), joints=kin.joints(), source="offline")
        tr.update({S.TONGS: st}, arms)
        ok_run = ok_run + 1 if tr.last.score == 1.0 else 0
        if success_at is None and ok_run >= WIN:
            success_at = k + 1
    return tr.last.score, success_at, kin


def plans(variant="", idle=""):
    old = {k: os.environ.get(k) for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
    try:
        os.environ["DEMO_VARIANT"], os.environ["DEMO_IDLE_ARM"] = variant, idle
        os.environ.pop("DEMO_HOLD_ONLY", None)
        return T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Demo(unittest.TestCase):
    def test_demo_locks_the_tongs(self):
        score, at, kin = play(plans())
        self.assertEqual(score, 1.0)
        self.assertIsNotNone(at)
        self.assertLess(at, 900)
        self.assertAlmostEqual(kin.tab, L.TAB_TRAVEL, delta=1e-9)
        self.assertAlmostEqual(kin.theta, LOCK, delta=1e-4)
        self.assertEqual(kin.towed, 0.0)

    def test_counterfactuals(self):
        self.assertEqual(play(plans(idle="right"))[0], 0.0)  # left arm only: the tongs are towed
        self.assertEqual(play(plans(idle="left"))[0], 0.25)  # right arm only: squeezed, never locked
        score, _, kin = play(plans("tab_only"))
        self.assertEqual(score, 0.0)
        self.assertGreater(kin.towed, 0.010)
        self.assertEqual(kin.tab, 0.0)
        self.assertEqual(play(plans("squeeze_release_pull"))[0], 0.25)
        score, _, kin = play(plans("tab_8mm"))
        self.assertEqual(score, 0.25)
        self.assertAlmostEqual(kin.tab, 0.008, delta=1e-6)
        self.assertEqual(play(plans("no_let_go"))[0], 0.5)
        self.assertEqual(plans_hold_only(), {})

    def test_pull_reaches_the_stop_only_while_held(self):
        kin = OK.OfflineTongs()
        kin.hold, kin.theta = "left", math.radians(5.0)
        self.assertLessEqual(L.travel_limit(kin.theta), 0.0059)

    def test_structure(self):
        p = plans()
        self.assertEqual(set(p), {"left", "right"})
        for side, wps in p.items():
            self.assertEqual({round(w.yaw, 6) for w in wps}, {0.0})  # fingers closing along y, across the tongs
            prev = None
            for w in wps:
                self.assertGreater(w.duration, 0)
                if prev is not None and abs(w.finger - prev.finger) > 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)  # open or close over 1.5 s or more
                    self.assertEqual(tuple(w.pos), tuple(prev.pos), w.label)  # never while moving
                self.assertGreaterEqual(w.finger, T.FULL_OPEN - 1e-9)
                self.assertLessEqual(w.finger, 0.0)
                prev = w
            self.assertEqual(wps[-1].label, "home")
            self.assertGreater(wps[-1].pos[2] - L.TOP, 0.10)
        total = max(sum(w.duration for w in wps) for wps in p.values())
        self.assertLess((total + 3.0) * HZ, 900)

    def test_pull_is_slow_and_ends_on_the_stop(self):
        wps = plans()["left"]
        pull = [w for w in wps if w.label.startswith("draw the tab")][0]
        self.assertLessEqual(T.PULL / pull.duration, 0.010)  # 9 mm/s commanded on average
        self.assertAlmostEqual(T.PULL - L.TAB_TRAVEL, T.PULL_SLACK, delta=1e-9)
        dwell = wps[wps.index(pull) + 1]
        self.assertEqual(dwell.label, "dwell at the stop")
        self.assertGreaterEqual(dwell.duration, 1.0)

    def test_pull_slack_covers_both_arms_give(self):
        """The arms' implicit PD (joints 1 to 3 at 80 N m/rad, 4 to 6 at 10) lets the pad tips give way along x: the
        left ones lag the pull, the right ones (and the held tongs) move with it. With the tab's 1 N of friction the
        commanded pull must exceed 12 mm by more than that give, or the tab stops short of the 11 mm that locks; past
        the stop the remaining slack presses on the stop with only a few newtons."""
        import yamkin as K
        bases = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
        stiff = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
        x_hold, y_hold = L.HOLD_XY
        x_ring, y_ring = L.RING_GRIP_XY
        points = {"right": (x_hold, y_hold, L.TOP + T.HOLD_TIPS + T.PAD_TIPS),
                  "left": (x_ring + L.TAB_TRAVEL, y_ring, L.TOP + T.RING_TIPS + T.PAD_TIPS)}
        give = {}
        for side, pt in points.items():
            q, err, _ = K.arm().solve(pt, K.matrix_to_quat_xyzw(K.grip_rotation(0.0)), base_pos=bases[side])
            self.assertLess(err, 1e-3)
            p, r, jl, ja = K.arm().tcp_pose_jacobian(q, bases[side])
            jt = jl + np.cross(ja.T, T.PAD_TIPS * r[:, 2]).T  # at the pad tips
            give[side] = float((jt @ np.linalg.inv(stiff) @ jt.T)[0, 0])  # m per N along x
        total = give["left"] + give["right"]
        self.assertGreater(total, 0.008)  # the reason for the slack: about 11 mm per newton
        self.assertGreater(T.PULL_SLACK, 1.2 * total * L.TAB_FRICTION)
        self.assertLess(T.PULL_SLACK / total, 4.0)  # newtons on the stop past the 1 N of friction, at most

    def test_hold_lets_go_only_after_the_ring_gripper_has_risen(self):
        """The right (hold) gripper goes down before the left one comes over the ring and rises after the left one
        has risen: the order that keeps the two wrists apart."""
        def times(wps):
            t, out = 0.0, {}
            for w in wps:
                out.setdefault(w.label, (t, t + w.duration))
                t += w.duration
            return out
        p = plans()
        right, left = times(p["right"]), times(p["left"])
        self.assertGreaterEqual(right["open to 60 mm: the arms spring onto the lock"][0],
                                left["straight up, clear of the tongs"][1])
        self.assertLessEqual(right["slowly down, one pad outside each open arm"][1], left["over the ring"][0])

    def test_arm_clearances_along_the_demo(self):
        """The arms' collision capsules (the offline checker's) stay 20 mm or more apart, and every link but the
        fingers 20 mm or more from the tongs (as the offline kinematics move them), all through the demo."""
        import yamkin as K
        p = plans()
        bases = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
        ap = {a: D.ArmPlan(list(w), bases[a]) for a, w in p.items()}
        kin, seeds = OK.OfflineTongs(), {}
        worst_arms, worst_tongs = math.inf, math.inf
        for k in range(int(26.0 * HZ)):
            t = (k + 1) / HZ
            arms, caps = {}, {}
            for a in ap:
                pos, rot, finger, _ = ap[a].target(t)
                tips = np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2]
                arms[a] = rigstate.Arm(np.asarray(pos), tips, float(np.clip(finger / T.FULL_OPEN, 0, 1)))
                if k % 5 == 0 and t > 1.0:
                    ok, q, _, _ = K.reachable(np.asarray(pos), 0.0, base_pos=bases[a],
                                              seeds=[seeds[a]] if a in seeds else None)
                    self.assertTrue(ok, (a, t))
                    seeds[a] = q
                    caps[a] = K.arm().capsules(q, bases[a], finger)
            kin.step(arms)
            if len(caps) < 2:
                continue
            worst_arms = min(worst_arms, K.capsule_clearance(caps["left"], caps["right"])[0])
            o, yaw = kin.pose()
            st = mk.state_from_poses(SPEC, (*o, *mk.yaw_quat(yaw)), joints=kin.joints())
            for a in caps:
                for link, a0, a1, r in caps[a]:
                    if link.startswith("tip"):
                        continue  # the fingers are meant to be at the tongs
                    pts = a0 + np.outer(np.linspace(0.0, 1.0, 30), a1 - a0)
                    worst_tongs = min(worst_tongs, float(st.distance(pts).min()) - r)
        self.assertGreater(worst_arms, 0.020)
        self.assertGreater(worst_tongs, 0.020)


def plans_hold_only():
    os.environ["DEMO_HOLD_ONLY"] = "1"
    try:
        return T.demo()
    finally:
        os.environ.pop("DEMO_HOLD_ONLY", None)


class OfflineHooks(unittest.TestCase):
    def test_offline_box_is_the_head_and_round_trips(self):
        c, size, yaw = SPEC.offline_box()
        self.assertEqual(tuple(round(v, 6) for v in size), (0.014, 0.03, 0.03))
        st = mk.ObjectState(np.array(c), mk.yaw_quat(yaw), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)) \
            if hasattr(mk, "ObjectState") else None
        if st is None:
            from yam_arena.task_file import ObjectState
            st = ObjectState(np.array(c), mk.yaw_quat(yaw), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        back = SPEC.offline_state(st)
        np.testing.assert_allclose(back.origin, L.ORIGIN, atol=1e-9)

    def test_success_hook_offline_starts_at_zero(self):
        rigstate.offline_reset()
        T._TRACKER.reset()
        T.OFFLINE.reset()
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": 0.6} for a in AWAY})
        self.assertFalse(T.success({S.TONGS: SPEC.rest_state()}))
        self.assertEqual(T.score({S.TONGS: SPEC.rest_state()}), 0.0)


# -- the text --------------------------------------------------------------------------------------------------------


class Text(unittest.TestCase):
    def setUp(self):
        self.meta = json.loads((HERE / "task.json").read_text())

    def test_fields_and_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "run_limit": 60,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.meta[k]), n, k)
        self.assertEqual(self.meta["category"], "Societal relatability")
        self.assertEqual(self.meta["instruction"], T.INSTRUCTION)
        self.assertEqual(self.meta["run_limit"], "900 steps (45 s at 20 Hz)")
        self.assertEqual(self.meta["scores"], "Dream-1 run: 0.0; demo: 1.0")

    def test_no_dashes_no_forbidden_words(self):
        texts = {p.name: p.read_text() for p in [HERE / "task.json", HERE / "README.md", HERE / "task.py",
                                                  HERE / "layout.py", HERE / "scoring.py",
                                                  HERE / "offline_kinematics.py"]}
        for name, text in texts.items():
            self.assertNotIn(chr(0x2014), text, name)  # em dash
            self.assertNotIn(chr(0x2013), text, name)  # en dash
        words = ("scale", "model of", "analogue", "analog", "mock", "trainer", "stand-in", "stand in",
                 "to be checked", "bought", "measured later", "placeholder", "claude", "anthropic", "t-70")
        for name in ("task.json", "README.md"):
            low = texts[name].lower()
            for w in words:
                self.assertIsNone(re.search(r"\b" + re.escape(w) + r"\b", low), f"{w!r} in {name}")
        fields = " ".join(str(v) for k, v in self.meta.items() if k != "scores").lower()
        self.assertNotIn("tbd", fields)
        self.assertIsNone(re.search(r"\bm\d{1,2}\b", fields))  # no internal gate labels
        self.assertNotIn("<", fields)

    def test_numbers_agree_with_the_layout(self):
        setup, scoring = self.meta["setup"], self.meta["scoring"]
        for s in ("84 mm between the inner faces", "176 mm from the tips", "27.5 degrees", "61 g", "54 g",
                  "about 20 g", "135 g", "0.20 N m", "0.40 N m", "4.4 N at 9 cm", "12 mm straight out",
                  "5 mm with the tab in and 17 mm", "x 19.0, y +0.5", "x 23.3, y +0.4", "x 2.0, y -4.2",
                  "x 1.8, y +4.2", "1 mm clear", "about 19 N", "within 3 degrees"):
            self.assertIn(s, setup, s)
        for s in ("at most 14 mm for 1 s", "at least 16 mm", "at least 3 cm", "less than 2 mm",
                  "less than 2 degrees", "same to 1 mm", "within 1 mm of its surface", "more than 1 cm below"):
            self.assertIn(s, scoring, s)
        self.assertAlmostEqual(L.tip_gap(L.OPEN) * 1000, 84, delta=0.5)
        self.assertAlmostEqual(L.SHUT_TIP_GAP, 0.014)
        self.assertAlmostEqual(L.ENGAGED_TAB_GAP, 0.016)
        self.assertAlmostEqual(L.tab_gap(L.TAB_TRAVEL), 0.017)
        self.assertIn("0.42 N m/rad", self.meta["fidelity"])
        self.assertIn("1 N friction", self.meta["fidelity"])
        self.assertIn("our estimates", self.meta["fidelity"])
        self.assertIn("4.4 N at 9 cm from the pin; the tab is 13 cm from that hold", self.meta["context"])
        self.assertIn("x 19.0, y +0.5 cm", self.meta["reset_notes"])

    def test_readme(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("Third-party assets: none", text)
        self.assertNotIn("shows only on letting go", text)


if __name__ == "__main__":
    unittest.main()
