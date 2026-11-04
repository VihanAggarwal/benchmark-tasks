"""Offline tests for scoring.py: synthetic object states, no simulator.  python3 -m unittest discover -s tests -v"""

import json
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, base_lift, cover, drift, gap_to_skid, on_skid, rotation, skid_flat, skid_under, tilt_deg,
)

MH = L.MACHINE_SIZE[2]
ST = L.SKID_SIZE[2]


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, size, quat=(0, 0, 0, 1), vel=(0, 0, 0)):
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half  # the world-aligned bounding box of the turned box
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=c - ext, high=c + ext, size=tuple(size))


def machine_at(x=L.MACHINE_X, y=L.MACHINE_Y, bottom=L.TOP, quat=(0, 0, 0, 1), **kw):
    """The machine with the centre of its underside at height ``bottom``."""
    zz = abs(rotation(quat)[2, 2])
    return state((x, y, bottom + zz * MH / 2), L.MACHINE_SIZE, quat, **kw)


def skid_at(x=L.SKID_X, y=L.SKID_Y, bottom=L.TOP, quat=(0, 0, 0, 1), **kw):
    zz = abs(rotation(quat)[2, 2])  # a box turned over still has a face down
    return state((x, y, bottom + zz * ST / 2), L.SKID_SIZE, quat, **kw)


M_HOME = machine_at()
M_UP = machine_at(bottom=L.TOP + 0.020)  # raised 20 mm: 8 mm over the skid
M_ON = machine_at(bottom=L.SKID_TOP)  # resting on the skid, where it started
S_HOME = skid_at()
S_UNDER = skid_at(y=L.MACHINE_Y)  # slid 27 cm: centred under the machine


def glide(a, b, n):
    """n states moving a's centre to b's in equal steps (the simulator moves things gradually)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        out.append(state((1 - f) * a.pos + f * b.pos, a.size, a.quat_xyzw))
    return out


def hold(s, n):
    return [s] * n


def run(machines, skids, hz=20.0):
    """Feed paired machine/skid states (lists of equal length) to a fresh tracker; return the last result."""
    assert len(machines) == len(skids)
    t = Tracker(hz)
    r = None
    for m, s in zip(machines, skids):
        r = t.update(m, s)
    return r, t


def procedure(lift=0.020, set_at=None, drift_xy=(0.0, 0.0), settle=30):
    """The intended sequence: raise the machine by ``lift``, slide the skid under, lower the machine onto it
    (at ``set_at`` = (x, y), default its start), hold still. ``drift_xy`` moves it sideways while it is up."""
    up = machine_at(bottom=L.TOP + lift)
    moved = machine_at(L.MACHINE_X + drift_xy[0], L.MACHINE_Y + drift_xy[1], L.TOP + lift)
    above = machine_at(*(set_at or (L.MACHINE_X, L.MACHINE_Y)), bottom=L.TOP + lift)
    down = machine_at(*(set_at or (L.MACHINE_X, L.MACHINE_Y)), bottom=L.SKID_TOP)
    ms = hold(M_HOME, 10) + glide(M_HOME, up, 40) + glide(up, moved, 20) + glide(moved, above, 20)
    ss = hold(S_HOME, len(ms))
    ms += hold(above, 80)
    ss += glide(S_HOME, S_UNDER, 80)
    ms += glide(above, down, 30) + hold(down, settle)
    ss += hold(S_UNDER, 30 + settle)
    return ms, ss


class Geometry(unittest.TestCase):
    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(about("x", 6)), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt

    def test_cover(self):
        self.assertAlmostEqual(cover(M_ON, S_UNDER), 60 / 70, places=6)  # the skid is 60 mm wide, the machine 70
        self.assertAlmostEqual(cover(M_HOME, S_HOME), 0.0)  # at the start the skid is beside it
        self.assertAlmostEqual(cover(M_ON, skid_at(y=-0.22)), 0.5 * 60 / 70, places=6)  # skid end at y 0
        across = skid_at(y=0.0, quat=about("z", 90))  # lying along x: a 60 mm band of the machine's 90 mm
        self.assertAlmostEqual(cover(M_ON, across), 60 / 90, places=6)
        self.assertFalse(on_skid(M_ON, across))
        self.assertTrue(skid_under(M_ON, across))

    def test_cover_matches_sampling(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            m = machine_at(L.MACHINE_X + rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05), L.SKID_TOP,
                           quat=about("z", rng.uniform(-90, 90)))
            k = skid_at(L.MACHINE_X + rng.uniform(-0.03, 0.03), rng.uniform(-0.25, 0.25),
                        quat=about("z", rng.uniform(-90, 90)))
            u, v = np.meshgrid(np.linspace(-0.5, 0.5, 141), np.linspace(-0.5, 0.5, 181))
            pts = np.stack([u.ravel() * L.MACHINE_SIZE[0], v.ravel() * L.MACHINE_SIZE[1], np.zeros(u.size)], 1)
            world = pts @ rotation(m.quat_xyzw).T + m.pos
            local = (world - k.pos) @ rotation(k.quat_xyzw)  # into the skid's frame
            inside = (np.abs(local[:, 0]) <= L.SKID_SIZE[0] / 2) & (np.abs(local[:, 1]) <= L.SKID_SIZE[1] / 2)
            self.assertAlmostEqual(cover(m, k), inside.mean(), delta=0.015)

    def test_on_skid_tolerances(self):
        self.assertTrue(on_skid(M_ON, S_UNDER))
        self.assertTrue(on_skid(machine_at(bottom=L.SKID_TOP + 0.0015), S_UNDER))
        self.assertFalse(on_skid(machine_at(bottom=L.SKID_TOP + 0.003), S_UNDER))  # held just above it
        self.assertFalse(on_skid(machine_at(bottom=L.SKID_TOP, quat=about("y", 4)), S_UNDER))  # tilted 4 deg
        self.assertTrue(on_skid(machine_at(L.MACHINE_X + 0.012, bottom=L.SKID_TOP), S_UNDER))  # 53/70 of it over
        self.assertFalse(on_skid(machine_at(L.MACHINE_X + 0.015, bottom=L.SKID_TOP), S_UNDER))  # 50/70 < 0.75
        self.assertTrue(on_skid(machine_at(y=0.10, bottom=L.SKID_TOP), S_UNDER))  # anywhere along the skid

    def test_a_skid_turned_over_is_still_flat(self):
        flipped = skid_at(y=0.0, quat=about("x", 180))  # the other broad face up
        self.assertTrue(skid_flat(flipped))
        self.assertTrue(on_skid(M_ON, flipped))
        self.assertTrue(skid_flat(skid_at(y=0.0, quat=about("y", 178.5))))
        self.assertFalse(skid_flat(skid_at(y=0.0, bottom=L.TOP + 0.024, quat=about("y", 90))))  # on its edge
        r, _ = run(*procedure()[:1], [*procedure()[1][:-40], *hold(flipped, 40)])
        self.assertEqual(r.score, 1.0)

    def test_skid_must_lie_flat_on_the_table(self):
        self.assertTrue(skid_flat(S_UNDER))
        held_up = skid_at(y=0.0, bottom=L.TOP + 0.010)
        self.assertFalse(skid_flat(held_up))
        self.assertFalse(on_skid(machine_at(bottom=L.SKID_TOP + 0.010), held_up))
        self.assertFalse(skid_flat(skid_at(y=0.0, bottom=L.TOP + 0.005, quat=about("x", 4))))

    def test_skid_under(self):
        self.assertTrue(skid_under(M_UP, S_UNDER))  # held up over it
        self.assertTrue(skid_under(M_ON, S_UNDER))
        self.assertFalse(skid_under(M_HOME, S_UNDER))  # a level machine on the table can't have the skid under it
        self.assertFalse(skid_under(M_UP, S_HOME))
        self.assertAlmostEqual(gap_to_skid(M_UP, S_UNDER), 0.008, places=6)

    def test_lift_counts_the_raised_edge(self):
        self.assertAlmostEqual(base_lift(M_HOME), 0.0, places=9)
        self.assertAlmostEqual(base_lift(M_UP), 0.020, places=9)
        # tipped 20 deg about x with its low edge on the table: its other edge is 90 sin 20 = 30.8 mm up
        q = about("x", 20)
        r = rotation(q)
        low_edge = min((r @ np.array([0, sy * L.MACHINE_SIZE[1] / 2, -MH / 2]))[2] for sy in (-1, 1))
        tipped = state((L.MACHINE_X, 0.0, L.TOP - low_edge), L.MACHINE_SIZE, q)
        self.assertAlmostEqual(base_lift(tipped), 0.090 * np.sin(np.radians(20)), places=6)

    def test_drift(self):
        self.assertAlmostEqual(drift(machine_at(L.MACHINE_X + 0.006, L.MACHINE_Y - 0.008)), 0.010, places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(hold(M_HOME, 20), hold(S_HOME, 20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run(*procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertAlmostEqual(t.max_lift, 0.020, places=6)
        self.assertLess(t.max_drift, 1e-9)

    def test_just_inside_and_just_outside_the_limits(self):
        r, _ = run(*procedure(lift=0.0299, drift_xy=(0.0, 0.0099)))  # 29.9 mm up, 9.9 mm off, then put back
        self.assertEqual(r.score, 1.0)
        r, _ = run(*procedure(lift=0.0301))
        self.assertEqual(r.score, 0.75)
        r, _ = run(*procedure(drift_xy=(0.0, 0.0101)))
        self.assertEqual(r.score, 0.75)
        r, _ = run(*procedure(lift=0.0499, drift_xy=(0.0299, 0.0)))
        self.assertEqual(r.score, 0.75)
        r, _ = run(*procedure(lift=0.0501))
        self.assertEqual(r.score, 0.5)

    def test_raised_too_high_scores_075(self):
        r, t = run(*procedure(lift=0.040))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_dragged_while_raised_scores_075(self):
        # dragged 20 mm while up and put back where it was: the drift is tracked over the whole trial
        r, t = run(*procedure(drift_xy=(0.0, 0.020)))
        self.assertAlmostEqual(t.max_drift, 0.020, places=6)
        self.assertEqual(r.score, 0.75)

    def test_set_down_20mm_along_the_skid_scores_075(self):
        r, _ = run(*procedure(set_at=(L.MACHINE_X, L.MACHINE_Y + 0.020)))
        self.assertEqual(r.score, 0.75)

    def test_far_too_high_scores_half(self):
        r, _ = run(*procedure(lift=0.080))
        self.assertEqual(r.score, 0.5)

    def test_dragged_out_of_its_30mm_outline_scores_zero(self):
        # dragged 50 mm (or just over 30 mm) while up, even if put back on the skid where it started
        for d in (0.050, 0.0301):
            r, t = run(*procedure(drift_xy=(0.0, d)))
            self.assertTrue(t.moved_out)
            self.assertEqual(r.score, 0.0)
        r, _ = run(*procedure(lift=0.060, drift_xy=(0.0, 0.0299)))  # just inside the outline, raised high
        self.assertEqual(r.score, 0.5)

    def test_skid_under_machine_still_held_scores_quarter(self):
        ms = hold(M_HOME, 10) + glide(M_HOME, M_UP, 40) + hold(M_UP, 110)
        ss = hold(S_HOME, 50) + glide(S_HOME, S_UNDER, 80) + hold(S_UNDER, 30)
        r, _ = run(ms, ss)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_machine_tilted_on_the_skid_end_scores_quarter(self):
        # the skid only partly under (its end at y +0.015), the machine let down with its -y side on the skid's
        # end and its +y edge on the table: tilted about 22 degrees, over half its footprint over the skid
        skid = skid_at(y=0.015 - L.SKID_SIZE[1] / 2)
        theta = np.degrees(np.arctan2(ST, 0.045 - 0.015))
        q = about("x", -theta)
        r_ = rotation(q)
        edge = r_ @ np.array([0, L.MACHINE_SIZE[1] / 2, -MH / 2])  # the +y lower edge, on the table at y 0.045
        centre = np.array([L.MACHINE_X, 0.045, L.TOP]) - edge
        leaning = state(centre, L.MACHINE_SIZE, q)
        self.assertGreater(cover(leaning, skid), 0.5)
        ms = hold(M_HOME, 10) + glide(M_HOME, M_UP, 40) + hold(M_UP, 60) + hold(leaning, 40)
        ss = hold(S_HOME, 50) + glide(S_HOME, skid, 60) + hold(skid, 40)
        r, _ = run(ms, ss)
        self.assertFalse(r.details["on_skid"])
        self.assertEqual(r.score, 0.25)

    def test_machine_set_back_on_the_table_scores_zero(self):
        ms = hold(M_HOME, 10) + glide(M_HOME, M_UP, 40) + glide(M_UP, M_HOME, 40) + hold(M_HOME, 30)
        r, _ = run(ms, hold(S_HOME, len(ms)))
        self.assertEqual(r.score, 0.0)

    def test_tipping_over_30_deg_scores_zero_whatever_follows(self):
        q = about("y", 35)
        tipped = state((L.MACHINE_X, 0.0, L.TOP + 0.045), L.MACHINE_SIZE, q)
        ms, ss = procedure()
        ms = ms[:20] + hold(tipped, 5) + ms[20:]
        ss = ss[:20] + hold(S_HOME, 5) + ss[20:]
        r, t = run(ms, ss)
        self.assertTrue(t.tipped)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_a_lean_under_30_deg_is_not_tipping(self):
        # rocked 25 degrees onto its -x edge for a moment: its +x edge goes 70 sin 25 = 29.6 mm up
        q = about("y", 25)
        r_ = rotation(q)
        low = min((r_ @ np.array([sx * L.MACHINE_SIZE[0] / 2, 0, -MH / 2]))[2] for sx in (-1, 1))
        leaning = state((L.MACHINE_X, 0.0, L.TOP - low), L.MACHINE_SIZE, q)
        ms, ss = procedure(lift=0.020)
        ms = ms[:20] + hold(leaning, 5) + ms[20:]
        ss = ss[:20] + hold(S_HOME, 5) + ss[20:]
        r, t = run(ms, ss)
        self.assertFalse(t.tipped)
        self.assertAlmostEqual(t.max_lift, 0.070 * np.sin(np.radians(25)), places=6)
        self.assertEqual(r.score, 1.0)

    def test_machine_off_the_table_scores_zero(self):
        fallen = machine_at(L.MACHINE_X, 0.70, bottom=0.0)
        r, _ = run(hold(M_HOME, 10) + hold(fallen, 20), hold(S_UNDER, 30))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["off_table"])

    def test_success_needs_stillness(self):
        ms, ss = procedure(settle=0)
        moving = machine_at(bottom=L.SKID_TOP, vel=(0, 0, -0.05))
        r, _ = run(ms + hold(moving, 5), ss + hold(S_UNDER, 5))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        sliding = skid_at(y=0.0, vel=(0, 0.03, 0))
        r, _ = run(ms + hold(M_ON, 5), ss + hold(sliding, 5))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)


class Exploits(unittest.TestCase):
    def test_shoving_the_skid_into_the_unraised_machine_scores_zero(self):
        # one arm only: the skid's end pushes the machine 22 cm along the table
        shoved = machine_at(y=0.22 + L.MACHINE_SIZE[1] / 2)
        ms = hold(M_HOME, 40) + glide(M_HOME, shoved, 60) + hold(shoved, 30)
        ss = hold(S_HOME, 20) + glide(S_HOME, S_UNDER, 80) + hold(S_UNDER, 30)
        r, t = run(ms, ss)
        self.assertEqual(r.score, 0.0)
        self.assertGreater(t.max_drift, 0.2)

    def test_skid_slid_through_a_machine_left_on_the_table_scores_zero(self):
        # the kinematic proxy's picture of the same thing: skid and machine in the same place, machine level
        # on the table. A machine can't be inside the plank, so this is not 'the skid under the machine'.
        ms = hold(M_HOME, 130)
        ss = hold(S_HOME, 20) + glide(S_HOME, S_UNDER, 80) + hold(S_UNDER, 30)
        r, _ = run(ms, ss)
        self.assertEqual(r.score, 0.0)

    def test_carrying_the_machine_to_the_skid_scores_zero(self):
        # one arm alone, instead of sliding the skid under: lift the machine (only 20 mm, within the raise
        # limit) and put it on the skid where it lies, or hold it over the skid there
        target = machine_at(L.SKID_X, L.SKID_Y, bottom=L.SKID_TOP)
        low_home = machine_at(bottom=L.TOP + 0.020)
        low_there = machine_at(L.SKID_X, L.SKID_Y, bottom=L.TOP + 0.020)
        carry = hold(M_HOME, 10) + glide(M_HOME, low_home, 30) + glide(low_home, low_there, 80)
        r, t = run(carry + glide(low_there, target, 20) + hold(target, 70), hold(S_HOME, len(carry) + 90))
        self.assertTrue(t.moved_out)
        self.assertLessEqual(t.max_lift, L.LIFT_FULL)
        self.assertEqual(r.score, 0.0)
        r, _ = run(carry + hold(low_there, 40), hold(S_HOME, len(carry) + 40))  # held over it: 'skid under'
        self.assertEqual(r.score, 0.0)

    def test_carried_and_then_pushed_back_on_the_skid_scores_zero(self):
        # put on the skid where it lies, then the skid (with the machine on it) pushed back to the start
        target = machine_at(L.SKID_X, L.SKID_Y, bottom=L.SKID_TOP)
        up, there = machine_at(bottom=L.TOP + 0.020), machine_at(L.SKID_X, L.SKID_Y, bottom=L.TOP + 0.020)
        ms = hold(M_HOME, 10) + glide(M_HOME, up, 20) + glide(up, there, 60) + glide(there, target, 20)
        ss = hold(S_HOME, len(ms))
        ms += glide(target, M_ON, 80) + hold(M_ON, 70)
        ss += glide(S_HOME, S_UNDER, 80) + hold(S_UNDER, 70)
        r, _ = run(ms, ss)
        self.assertTrue(r.details["on_skid"])
        self.assertEqual(r.score, 0.0)

    def test_one_arm_cannot_reach_half_cover_inside_the_outline(self):
        # one arm moves the machine the most it may (30 mm) toward the skid lying where it is: under a quarter
        # of the footprint is over the skid, so neither 'on' nor 'under'
        shifted = machine_at(y=-0.030, bottom=L.SKID_TOP)
        self.assertLess(cover(shifted, S_HOME), L.UNDER_COVER)
        self.assertFalse(skid_under(shifted, S_HOME))

    def test_tilting_instead_of_raising_counts_the_raised_edge(self):
        # its low edge only 15 mm up, but tipped 20 degrees: the high edge is 46 mm up, so not 'kept low'
        q = about("x", 20)
        r_ = rotation(q)
        low = min((r_ @ np.array([0, s * L.MACHINE_SIZE[1] / 2, -MH / 2]))[2] for s in (-1, 1))
        tilted_up = state((L.MACHINE_X, 0.0, L.TOP + 0.015 - low), L.MACHINE_SIZE, q)
        ms = hold(M_HOME, 10) + hold(tilted_up, 100)
        ss = hold(S_HOME, 30) + glide(S_HOME, S_UNDER, 80)
        ms += glide(machine_at(bottom=L.TOP + 0.015), M_ON, 20) + hold(M_ON, 30)
        ss += hold(S_UNDER, 50)
        r, t = run(ms, ss)
        self.assertGreater(t.max_lift, L.LIFT_FULL)
        self.assertEqual(r.score, 0.75)

    def test_skid_on_top_of_the_machine_scores_zero(self):
        on_top = skid_at(y=0.0, bottom=L.TOP + MH)
        ms = hold(M_HOME, 100)
        ss = hold(S_HOME, 10) + glide(skid_at(bottom=L.TOP + MH), on_top, 60) + hold(on_top, 30)
        r, _ = run(ms, ss)
        self.assertEqual(r.score, 0.0)

    def test_skid_held_up_with_the_machine_on_it_scores_zero(self):
        # the skid held 15 mm off the table under the machine: not 'set down on the skid'
        lifted = skid_at(y=0.0, bottom=L.TOP + 0.015)
        m = machine_at(bottom=L.TOP + 0.015 + ST)
        ms = hold(M_HOME, 10) + glide(M_HOME, machine_at(bottom=L.TOP + 0.027), 30) + glide(
            machine_at(bottom=L.TOP + 0.027), m, 20) + hold(m, 30)
        ss = hold(S_HOME, 10) + glide(skid_at(bottom=L.TOP + 0.015), lifted, 50) + hold(lifted, 30)
        r, _ = run(ms, ss)
        self.assertEqual(r.score, 0.0)


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        far = machine_at(50.0, -40.0, bottom=-10.0, quat=about("y", 123))
        return [
            (machine_at(x=nan), S_HOME),
            (state((L.MACHINE_X, 0, L.TOP + 0.03), L.MACHINE_SIZE, quat=(nan, 0, 0, nan)), S_UNDER),
            (M_ON, skid_at(y=inf)),
            (machine_at(vel=(0, nan, 0)), S_HOME),
            (state((L.MACHINE_X, 0, L.TOP + 0.03), L.MACHINE_SIZE, quat=(0, 0, 0, 0)),
             state((L.SKID_X, 0, L.TOP + 0.006), L.SKID_SIZE, quat=(0, 0, 0, 0))),
            (far, skid_at(-3.0, 8.0, bottom=100.0, quat=about("x", 77))),
            (machine_at(bottom=L.TOP, quat=about("x", 180)), skid_at(quat=about("y", 180))),
        ]

    def test_no_crash_and_clean_details(self):
        for m, s in self.odd():
            t = Tracker()
            t.update(M_HOME, S_HOME)
            r = t.update(m, s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)
            json.dumps(t.result(m, s).details, allow_nan=False)

    def test_a_broken_observation_changes_nothing(self):
        ms, ss = procedure()
        bad = machine_at(x=float("nan"), bottom=L.TOP + 0.5)
        t = Tracker()
        for i, (m, s) in enumerate(zip(ms, ss)):
            r = t.update(bad if i == 60 else m, s)
            if i == 60:
                self.assertEqual(r.score, 0.0)
                self.assertTrue(r.details["bad_state"])
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.bad_steps, 1)
        self.assertAlmostEqual(t.max_lift, 0.020, places=6)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        tipped = state((L.MACHINE_X, 0.12, L.TOP + 0.045), L.MACHINE_SIZE, about("x", 60))
        for _ in range(30):
            t.update(tipped, S_UNDER)
        self.assertTrue(t.tipped)
        self.assertGreater(t.max_drift, 0.1)
        r = t.update(M_HOME, S_HOME)  # everything jumps back in one step: a new episode
        self.assertFalse(t.tipped)
        self.assertEqual(t.max_drift, 0.0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        for m, s in zip(*procedure()):  # and the new episode can score 1.0
            r = t.update(m, s)
        self.assertEqual(r.score, 1.0)

    def test_putting_things_back_slowly_does_not_reset(self):
        t = Tracker()
        high = machine_at(bottom=L.TOP + 0.045)
        ms = hold(M_HOME, 5) + glide(M_HOME, high, 30) + glide(high, M_HOME, 30) + hold(M_HOME, 10)
        ss = hold(S_HOME, 5) + glide(S_HOME, S_UNDER, 30) + glide(S_UNDER, S_HOME, 30) + hold(S_HOME, 10)
        for m, s in zip(ms, ss):
            t.update(m, s)
        self.assertAlmostEqual(t.max_lift, 0.045, places=6)  # back at the start, gradually: same episode
        self.assertEqual(t.steps, len(ms))

    def test_dropping_the_machine_back_onto_its_mark_is_not_a_reset(self):
        # raised 30 cm (far too high), let fall straight back onto its mark: its last sampled step before landing
        # is 10 cm (more than RESET_JUMP), but its speed then (2 m/s) explains it; the raise stays on the record
        t = Tracker()
        high = machine_at(bottom=L.TOP + 0.30)
        for m in hold(M_HOME, 5) + glide(M_HOME, high, 40) + hold(high, 5):
            t.update(m, S_HOME)
        g, dt = L.GRAVITY, 1 / L.CONTROL_HZ
        h, v, last = 0.30, 0.0, None
        while h > 0:  # free fall, sampled every control step
            v, h = v + g * dt, h - v * dt - 0.5 * g * dt * dt
            if h > 0:
                last = machine_at(bottom=L.TOP + h, vel=(0, 0, -v))
                t.update(last, S_HOME)
        self.assertGreater(float(np.linalg.norm(last.pos - M_HOME.pos)), L.RESET_JUMP)
        t.update(M_HOME, S_HOME)  # landed on its mark, still
        self.assertAlmostEqual(t.max_lift, 0.30, places=6)
        r = None
        for m, s in zip(*procedure()):
            r = t.update(m, s)
        self.assertEqual(r.score, 0.5)  # the 30 cm raise still counts

    def test_teleport_after_the_demo_resets(self):
        # the demo's end (skid centred at y -0.05 under the machine on it), then everything back on its marks
        t = Tracker()
        end_skid = skid_at(y=-0.05)
        ms, ss = procedure()
        for m, s in zip(ms, ss[:-1] + [end_skid]):
            t.update(m, s)
        t.max_lift = 0.2  # whatever happened in that episode
        t.update(M_HOME, S_HOME)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.max_lift, 0.0)

    def test_a_small_jump_onto_the_marks_is_not_a_reset(self):
        t = Tracker()
        ms, ss = procedure(lift=0.040)
        near = skid_at(y=L.SKID_Y + 0.05)
        ms += glide(M_ON, M_HOME, 20)
        ss += glide(S_UNDER, near, 20)
        for m, s in zip(ms, ss):
            t.update(m, s)
        t.update(M_HOME, S_HOME)  # the skid jumps the last 5 cm in one step: physics can do that
        self.assertAlmostEqual(t.max_lift, 0.040, places=6)
        self.assertEqual(t.steps, len(ms) + 1)


if __name__ == "__main__":
    unittest.main()
