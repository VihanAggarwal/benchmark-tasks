"""Offline tests for scoring.py: synthetic book states, no simulator.  python3 -m unittest discover -s tests -v"""

import json
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, clearances, gap_mid_y, in_gap, rotation, row_upright, seated, spine_line, spine_x, tilt_deg,
    turn_deg, underside_centre_z, yaw_deg,
)

N = L.NEW


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def state(center, size, quat=(0, 0, 0, 1), vel=(0, 0, 0)):
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=c - ext, high=c + ext)


def neighbour(name, dx=0.0, dy=0.0, quat=(0, 0, 0, 1), lift=0.0):
    x, y = L.NB_START[name]
    # a turned book is held on its bottom edge: lift its centre so its lowest point stays on the shelf
    zz = abs(rotation(quat)[2, 2])
    low = L.SHELF_TOP + lift + zz * L.NB_H / 2 + (1 - zz) * 0.0
    return state((x + dx, y + dy, low), L.SIZE[name], quat)


def new_at(x=L.NEW_END_X, y=L.Y0, bottom=L.SHELF_TOP, quat=(0, 0, 0, 1), vel=(0, 0, 0)):
    """The new book with the centre of its underside at (x, y, bottom)."""
    r = rotation(quat)
    c = np.array([x, y, bottom]) + r[:, 2] * L.NEW_H / 2  # the underside's centre is at (x, y, bottom)
    return state(c, L.SIZE[N], quat, vel)


def row(**kw):
    """The six neighbours at their places, with changes per name: name=dict(dx=, dy=, quat=)."""
    return {n: neighbour(n, **kw.get(n, {})) for n in L.NEIGHBOURS}


def world(new, **kw):
    out = row(**kw)
    out[N] = new
    return out


START_NEW = state((L.NEW_START[0], L.NEW_START[1], L.TOP + L.NEW_H / 2), L.SIZE[N])
CLEAR_ARMS = {  # the left hand backed off in front of the case (axis 95 mm from the spines), the right at ready
    "left": rigstate.Arm(tcp=np.array([L.XS - 0.095, L.Y0, L.SHELF_TOP + 0.18]),
                         tips=np.array([L.XS - 0.095, L.Y0, L.SHELF_TOP + 0.167]), opening=0.47),
    "right": rigstate.Arm(tcp=np.array([0.32, -0.305, L.TOP + 0.26]), tips=np.array([0.32, -0.305, L.TOP + 0.247]),
                          opening=1.0),
}


def holding_arms(books=None):
    """The left hand closed on the new book's top."""
    n = (books or {}).get(N) or new_at()
    tcp = np.array([n.pos[0], n.pos[1], n.pos[2] + L.NEW_H / 2 - 0.015])
    out = dict(CLEAR_ARMS)
    out["left"] = rigstate.Arm(tcp=tcp, tips=tcp - np.array([0, 0, 0.013]), opening=0.23)
    return out


def run(books, arms=CLEAR_ARMS, steps=60, tracker=None, history=None):
    """Feed ``books`` for ``steps`` control steps to a (fresh) tracker; ``history`` is a list of earlier worlds."""
    t = tracker or Tracker()
    for b in history or []:
        t.update(b, arms)
    r = None
    for _ in range(steps):
        r = t.update(books, arms)
    return r, t


class Geometry(unittest.TestCase):
    def test_tilt_and_yaw(self):
        self.assertAlmostEqual(tilt_deg(about("x", 6)), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(yaw_deg(about("z", 7)), 7.0, places=4)
        self.assertAlmostEqual(abs(yaw_deg(about("z", 180))), 180.0, places=3)
        self.assertAlmostEqual(turn_deg(about("x", 1), about("x", 1.5)), 0.5, places=4)

    def test_layout_is_consistent(self):
        # the row (6 x 25 mm + the 40 mm gap) fills the case exactly, 0.5 mm from each panel
        ys = sorted(y for _, y in L.NB_START.values())
        edges = [(y - L.NB_T / 2, y + L.NB_T / 2) for y in ys]
        self.assertAlmostEqual(edges[0][0], L.Y0 - L.ROW_HALF, places=9)
        self.assertAlmostEqual(edges[-1][1], L.Y0 + L.ROW_HALF, places=9)
        self.assertAlmostEqual(edges[3][0] - edges[2][1], L.GAP, places=9)
        for (a0, a1), (b0, b1) in zip(edges[:2], edges[1:3]):
            self.assertAlmostEqual(b0, a1, places=9)  # the packed side touches
        self.assertAlmostEqual(L.GAP - L.NEW_T, 0.010, places=9)  # 5 mm clear each side

    def test_starting_layout(self):
        b = world(START_NEW)
        self.assertFalse(seated(b))
        self.assertFalse(in_gap(b))
        self.assertAlmostEqual(gap_mid_y(b), L.Y0, places=9)
        self.assertAlmostEqual(spine_line(b), L.XS, places=9)
        self.assertTrue(row_upright(b, {}))
        r, _ = run(b)
        self.assertEqual(r.score, 0.0)

    def test_clearances(self):
        cc, cd = clearances(world(new_at()))
        self.assertAlmostEqual(cc, 0.005, places=9)
        self.assertAlmostEqual(cd, 0.005, places=9)
        cc, cd = clearances(world(new_at(y=L.Y0 + 0.004)))
        self.assertAlmostEqual(cc, 0.009, places=9)
        self.assertAlmostEqual(cd, 0.001, places=9)

    def test_spine_x(self):
        n = new_at()
        self.assertAlmostEqual(spine_x(N, n), L.XS, places=9)
        self.assertAlmostEqual(spine_x(N, new_at(x=L.NEW_END_X - 0.02)), L.XS - 0.02, places=9)


class Tiers(unittest.TestCase):
    def test_full_score(self):
        b = world(new_at())
        r, t = run(b)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["seated"] and r.details["hands_off"] and r.details["still"])

    def test_seated_edges(self):
        for dy in (-0.0029, 0.0029):  # 2.1 mm clear on one side: still 1.0
            self.assertEqual(run(world(new_at(y=L.Y0 + dy)))[0].score, 1.0, dy)
        for dy in (-0.0035, 0.0035):  # 1.5 mm clear: in the gap but not seated
            self.assertEqual(run(world(new_at(y=L.Y0 + dy)))[0].score, 0.5, dy)
        self.assertEqual(run(world(new_at(x=L.NEW_END_X - 0.0045)))[0].score, 1.0)  # spine 4.5 mm proud
        self.assertEqual(run(world(new_at(x=L.NEW_END_X - 0.006)))[0].score, 0.5)
        self.assertEqual(run(world(new_at(quat=about("x", 2.5))))[0].score, 1.0)
        self.assertEqual(run(world(new_at(quat=about("z", 2.5))))[0].score, 1.0)  # 2.7 mm clear each side
        self.assertEqual(run(world(new_at(quat=about("z", 3.5))))[0].score, 0.5)  # turned past 3 degrees
        self.assertEqual(run(world(new_at(y=L.Y0 + 0.0025, quat=about("z", 2.5))))[0].score, 0.5)  # 0.2 mm clear of D

    def test_seated_needs_two_mm_clear_each_side(self):
        b = world(new_at())
        cc, cd = clearances(b)
        self.assertAlmostEqual(cc, 0.005, places=9)
        self.assertAlmostEqual(cd, 0.005, places=9)
        b = world(new_at(y=L.Y0 + 0.0029, quat=about("z", 1.0)))  # centre in, but a corner 1.2 mm from D
        self.assertLess(clearances(b)[1], L.SEAT_CLEAR_MIN)
        self.assertFalse(seated(b))
        self.assertEqual(run(b)[0].score, 0.5)
        self.assertEqual(run(world(new_at(bottom=L.SHELF_TOP + 0.0025)))[0].score, 1.0)

    def test_half_credit(self):
        cases = {
            "tilt 6 deg": new_at(quat=about("x", 6)),
            "tilt 9 deg": new_at(quat=about("x", 9)),
            "yaw 5.5 deg": new_at(quat=about("z", 5.5)),  # the gap itself allows no more (30 cos + 105 sin <= 40 mm)
            "spine 20 mm proud": new_at(x=L.NEW_END_X - 0.020),
            "spine 30 mm proud": new_at(x=L.NEW_END_X - 0.030),
        }
        for name, n in cases.items():
            r = run(world(n))[0]
            if name.startswith("tilt"):  # a tilted book leaning in the gap: its lowest corner is on the shelf
                n = new_at(bottom=L.SHELF_TOP, quat=about("x", float(name.split()[1])))
                r = run(world(n))[0]
            self.assertEqual(r.score, 0.5, (name, r.details))

    def test_not_in_gap(self):
        for name, n in {
            "tilt 11 deg": new_at(quat=about("x", 11)),
            "yaw 16 deg": new_at(quat=about("z", 16)),
            "spine 45 mm proud": new_at(x=L.NEW_END_X - 0.045),
            "a corner on C": new_at(y=L.Y0 + 0.0065),
            "on the shelf in front of the row": new_at(x=0.2, y=L.Y0 + 0.08),
            "left on the table": new_at(x=0.17, y=L.Y0 + 0.13, bottom=L.TOP),
        }.items():
            r = run(world(n))[0]
            self.assertEqual(r.score, 0.0, (name, r.details))

    def test_on_the_tops_earns_nothing(self):
        n = new_at(x=L.XS + 0.0525, bottom=L.NB_TOP)  # perched upright on the neighbours' tops: not shelved
        self.assertEqual(run(world(n))[0].score, 0.0)
        turned = new_at(x=L.XS + 0.0525, bottom=L.NB_TOP, quat=about("z", 90))  # turned to bridge C and D
        self.assertEqual(run(world(turned))[0].score, 0.0)

    def test_neighbour_leaning(self):
        # seated, but D leans 10 degrees toward the new book (its lowest edge stays on the shelf)
        d = neighbour("book_d", quat=about("x", 10), lift=0.0)
        low = L.SHELF_TOP + abs(rotation(about("x", 10))[2, 2]) * L.NB_H / 2 + 0.5 * L.NB_T * np.sin(np.radians(10))
        d = state((d.pos[0], d.pos[1], low), L.SIZE["book_d"], about("x", 10))
        b = world(new_at())
        b["book_d"] = d
        r, t = run(b)
        self.assertEqual(r.score, 0.25, r.details)
        self.assertFalse(r.details["row_upright"])
        self.assertTrue(r.details["row_leaning"])
        # a neighbour at 4 degrees is upright enough
        d4 = state((d.pos[0], d.pos[1], L.SHELF_TOP + abs(rotation(about("x", 4))[2, 2]) * L.NB_H / 2
                    + 0.5 * L.NB_T * np.sin(np.radians(4))), L.SIZE["book_d"], about("x", 4))
        b["book_d"] = d4
        self.assertEqual(run(b)[0].score, 1.0)
        # one at 25 degrees: no credit
        d25 = state((d.pos[0], d.pos[1], L.SHELF_TOP + abs(rotation(about("x", 25))[2, 2]) * L.NB_H / 2
                     + 0.5 * L.NB_T * np.sin(np.radians(25))), L.SIZE["book_d"], about("x", 25))
        b["book_d"] = d25
        self.assertEqual(run(b)[0].score, 0.0)

    def test_neighbour_moved(self):
        for dy, expect in ((0.005, 1.0), (0.008, 0.25), (0.019, 0.25), (0.022, 0.0)):
            b = world(new_at(), book_e={"dy": dy})
            self.assertEqual(run(b)[0].score, expect, dy)
        self.assertEqual(run(world(new_at(), book_a={"dx": -0.01}))[0].score, 0.25)  # dragged 10 mm toward the front


class ZeroConditions(unittest.TestCase):
    def test_a_book_fell_at_any_moment(self):
        t = Tracker()
        tipped = world(new_at(), book_b={"quat": about("x", 50), "lift": 0.03})
        t.update(tipped, CLEAR_ARMS)
        r, t = run(world(new_at()), tracker=t)  # stood up again afterwards
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["fell"])

    def test_new_book_fell_in_transit(self):
        t = Tracker()
        t.update(world(new_at(bottom=L.SHELF_TOP + 0.05, quat=about("y", 60))), CLEAR_ARMS)
        r, _ = run(world(new_at()), tracker=t)
        self.assertEqual(r.score, 0.0)

    def test_new_book_tilted_in_the_hand_then_seated_scores_full(self):
        t = Tracker()
        tilted = new_at(bottom=L.SHELF_TOP + 0.05, quat=about("y", 60))  # 60 degrees, a hand closed on it
        for _ in range(5):
            t.update(world(tilted), holding_arms(world(tilted)))
        r, _ = run(world(new_at()), tracker=t, steps=40)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertFalse(r.details["fell"])
        self.assertGreaterEqual(r.details["new_max_tilt_held_deg"], 59.0)

    def test_new_book_tipped_when_not_held_stays_latched(self):
        t = Tracker()
        tipped = new_at(bottom=L.SHELF_TOP + 0.05, quat=about("y", 60))
        t.update(world(tipped), CLEAR_ARMS)
        r, _ = run(world(new_at()), tracker=t, steps=40)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["fell"])

    def test_tilt_with_an_open_hand_beside_it_is_not_held(self):
        t = Tracker()
        tipped = new_at(bottom=L.SHELF_TOP + 0.05, quat=about("y", 60))
        arms = holding_arms(world(tipped))
        arms["left"] = rigstate.Arm(tcp=arms["left"].tcp, tips=arms["left"].tips, opening=1.0)  # pads 94 mm apart
        t.update(world(tipped), arms)
        self.assertTrue(t.fell)

    def test_tilt_with_unreadable_arms_fails_closed(self):
        for arms in (None, {"left": holding_arms()["left"]}):
            t = Tracker()
            t.update(world(new_at(quat=about("y", 60), bottom=L.SHELF_TOP + 0.05)), arms)
            self.assertTrue(t.fell)

    def test_a_neighbour_is_never_held(self):
        t = Tracker()
        b = world(new_at(), book_c={"quat": about("x", 50), "lift": 0.03})
        arms = dict(CLEAR_ARMS)
        c = b["book_c"]
        tcp = np.array([c.pos[0], c.pos[1], c.pos[2] + 0.06])
        arms["left"] = rigstate.Arm(tcp=tcp, tips=tcp - np.array([0, 0, 0.013]), opening=0.23)  # closed on C
        t.update(b, arms)
        self.assertTrue(t.fell)

    def test_new_book_dropped_off_the_table_or_below_the_shelf_latches(self):
        for low in (L.TOP - 0.02, L.TOP - 0.3):
            t = Tracker()
            t.update(world(new_at(bottom=low)), CLEAR_ARMS)
            r, _ = run(world(new_at()), tracker=t, steps=40)
            self.assertEqual(r.score, 0.0, low)
            self.assertTrue(r.details["off_table"])
        t = Tracker()  # on the table (12 mm below the shelf top) is where it starts: fine
        t.update(world(START_NEW), CLEAR_ARMS)
        r, _ = run(world(new_at()), tracker=t, steps=40)
        self.assertEqual(r.score, 1.0)

    def test_neighbour_tilted_twenty_five_then_righted(self):
        t = Tracker()
        t.update(world(new_at(), book_c={"quat": about("x", 25), "lift": 0.02}), CLEAR_ARMS)
        r, _ = run(world(new_at()), tracker=t)
        self.assertEqual(r.score, 0.25, r.details)  # in the gap, row in place now but it tilted over 20: not 'upright'

    def test_neighbour_off_the_shelf(self):
        b = world(new_at())
        b["book_f"] = state((L.NB_X, L.Y0 + 0.2, L.TOP + L.NB_H / 2), L.SIZE["book_f"])  # on the table
        r, _ = run(b)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["neighbours_off_shelf"], ["book_f"])

    def test_nan(self):
        t = Tracker()
        run(world(new_at()), tracker=t, steps=30)
        bad = world(new_at())
        bad["book_a"].pos = np.array([np.nan, 0.0, 0.0])
        r = t.update(bad, CLEAR_ARMS)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["bad_state"])
        r, _ = run(world(new_at()), tracker=t, steps=10)
        self.assertEqual(r.score, 0.0)  # the still second starts over after a broken observation
        r, _ = run(world(new_at()), tracker=t, steps=30)
        self.assertEqual(r.score, 1.0)
        inf = world(new_at())
        inf[N].lin_vel = np.array([np.inf, 0, 0])
        self.assertEqual(Tracker().update(inf, CLEAR_ARMS).score, 0.0)


class HandsOffAndStill(unittest.TestCase):
    def test_still_held_scores_zero(self):
        b = world(new_at())
        r, _ = run(b, arms=holding_arms(b))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["hands_off"])

    def test_let_go_needs_a_whole_second(self):
        b = world(new_at())
        t = Tracker()
        run(b, arms=holding_arms(b), steps=40, tracker=t)
        for k in range(1, 21):
            r = t.update(b, CLEAR_ARMS)
            self.assertEqual(r.score, 1.0 if k == 20 else 0.0, k)
        self.assertTrue(r.success)

    def test_a_hand_still_close(self):
        b = world(new_at())
        arms = dict(CLEAR_ARMS)
        tcp = np.array([L.NB_X, L.Y0 + 0.07, L.SHELF_TOP + 0.2])  # open fingers' zone 30 mm from a neighbour's top
        arms["left"] = rigstate.Arm(tcp=tcp, tips=tcp - np.array([0, 0, 0.013]), opening=1.0)
        self.assertEqual(run(b, arms=arms)[0].score, 0.0)

    def test_unreadable_arms_never_count(self):
        b = world(new_at())
        for arms in (None, {"left": CLEAR_ARMS["left"]}, {"left": CLEAR_ARMS["left"], "right": SimpleNamespace(
                tcp=np.array([np.nan] * 3), tips=np.zeros(3), opening=1.0)}):
            self.assertEqual(run(b, arms=arms)[0].score, 0.0)

    def test_moving_in_the_last_second(self):
        t = Tracker()
        for k in range(20):  # sliding back 0.4 mm a step from 6 mm off: ends seated but moved 7.6 mm over the second, more than 5 mm
            t.update(world(new_at(x=L.NEW_END_X + 0.006 - 0.0004 * k)), CLEAR_ARMS)
        self.assertEqual(t.last.score, 0.0)
        self.assertFalse(t.last.details["still"])
        r, _ = run(world(new_at()), tracker=t, steps=20)
        self.assertEqual(r.score, 1.0)

    def test_turning_in_the_last_second(self):
        t = Tracker()
        for k in range(20):
            t.update(world(new_at(quat=about("z", 6.0 - 0.3 * k))), CLEAR_ARMS)  # ends seated, but turned 5.7 degrees over the second
        self.assertEqual(t.last.score, 0.0)
        t = Tracker()
        for k in range(20):
            t.update(world(new_at(quat=about("z", 2.0 - 0.1 * k))), CLEAR_ARMS)  # 1.9 degrees: not visible
        self.assertEqual(t.last.score, 1.0)

    def test_a_neighbour_creeping(self):
        t = Tracker()
        for k in range(25):
            t.update(world(new_at(), book_a={"dy": 0.0055 - 0.0004 * k}), CLEAR_ARMS)
        self.assertEqual(t.last.score, 0.0)


class Exploits(unittest.TestCase):
    def test_spine_in(self):
        n = new_at(quat=about("z", 180))
        self.assertEqual(run(world(n))[0].score, 0.0)
        self.assertFalse(seated(world(n)))

    def test_shove_the_neighbours_apart_and_stand_in_the_wider_gap(self):
        # C and D pushed 10 mm outward (against B and E they cannot go, but try): the gap's measured from the
        # neighbours as they are, so the book stands square in it, but the row is not in place
        b = world(new_at(), book_c={"dy": -0.010}, book_d={"dy": 0.010})
        r = run(b)[0]
        self.assertLessEqual(r.score, 0.25)

    def test_take_out_a_neighbour_and_use_its_place(self):
        b = world(new_at(y=L.NB_START["book_c"][1]))
        b["book_c"] = state((0.1, 0.5, L.TOP + L.NB_H / 2), L.SIZE["book_c"])  # C removed, left on the table
        self.assertEqual(run(b)[0].score, 0.0)

    def test_leave_the_book_held_in_the_gripper(self):
        b = world(new_at())
        self.assertEqual(run(b, arms=holding_arms(b))[0].score, 0.0)

    def test_hold_the_book_just_above_the_shelf(self):
        b = world(new_at(bottom=L.SHELF_TOP + 0.002))  # within the 3 mm of resting, but a hand still on it
        self.assertEqual(run(b, arms=holding_arms(b))[0].score, 0.0)

    def test_stand_it_on_the_neighbours_without_seating(self):
        n = new_at(bottom=L.NB_TOP, x=L.XS + 0.05)
        self.assertEqual(run(world(n))[0].score, 0.0)

    def test_knock_the_whole_row_back_and_forth(self):
        t = Tracker()
        t.update(world(new_at(), book_a={"quat": about("x", 46), "lift": 0.03}), CLEAR_ARMS)
        self.assertEqual(run(world(new_at()), tracker=t)[0].score, 0.0)

    def test_new_book_on_its_back(self):
        n = state((L.XS + 0.05, L.Y0, L.NB_TOP + L.NEW_T / 2), (L.NEW_H, L.BOOK_D, L.NEW_T))  # lying flat on the row
        n = state((L.XS + 0.05, L.Y0, L.NB_TOP + L.NEW_T / 2), L.SIZE[N], about("y", 90))
        self.assertEqual(run(world(n))[0].score, 0.0)

    def test_a_second_book_ignored_and_new_book_out_of_the_row_ends_zero(self):
        self.assertEqual(run(world(new_at(y=L.Y0 + 0.07)))[0].score, 0.0)  # on top of E and F's place: a neighbour moved


class Episodes(unittest.TestCase):
    def test_reset_clears_the_record(self):
        t = Tracker()
        t.update(world(new_at(), book_b={"quat": about("x", 60), "lift": 0.03}), CLEAR_ARMS)
        self.assertTrue(t.fell)
        t.update(world(new_at(bottom=L.TOP - 0.05)), CLEAR_ARMS)
        self.assertIsNotNone(t.off_table)
        # the simulator puts every book back on its mark in one step
        start = world(START_NEW)
        for _ in range(3):
            t.update(start, CLEAR_ARMS)
        self.assertFalse(t.fell)
        self.assertIsNone(t.off_table)
        self.assertEqual(t.resets, 1)

    def test_a_book_dropped_back_on_its_mark_is_not_a_reset(self):
        t = Tracker()
        t.update(world(new_at(bottom=L.SHELF_TOP + 0.03)), CLEAR_ARMS)
        t.update(world(new_at(), book_b={"quat": about("x", 60), "lift": 0.03}), CLEAR_ARMS)
        t.update(world(START_NEW, book_b={"quat": about("x", 60), "lift": 0.03}), CLEAR_ARMS)  # the row not at its marks
        self.assertTrue(t.fell)


class TaskModule(unittest.TestCase):
    def test_objects_and_json(self):
        import importlib

        sys.path.insert(0, str(HERE.parent.parent / "tools" / "offline" / "yam_stub"))
        import task  # noqa: F401  (the stub yam_arena makes it importable offline)

        importlib.reload(task)
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(L.BOOKS) & set(names), set(L.BOOKS))
        spec = json.loads((HERE / "task.json").read_text())
        for key in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes", "run_limit"):
            self.assertTrue(spec[key])


if __name__ == "__main__":
    unittest.main()
