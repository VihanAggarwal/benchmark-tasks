"""Offline tests for scoring.py and task.py: synthetic tube states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker,
    bearing_err_deg,
    points,
    pose,
    position_of,
    released,
    rotation,
    tilt_deg,
    tube_distance,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
READ = 180.0  # the label yaw that faces the reader slot (local +x along -x)


def arm(tcp, opening=0.38):
    """A gripper pointing straight down with its tool point at ``tcp`` (pad-tip point 13 mm below it)."""
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - [0, 0, 0.013], opening=opening)


# both grippers parked well away (the demo's 'aside' poses): every tube released
FAR = {"left": arm((0.23, 0.20, L.TOP + 0.15)), "right": arm((0.23, -0.20, L.TOP + 0.15))}


def quat(yaw_deg=0.0, tilt_deg_=0.0, tilt_az_deg=0.0):
    """(x, y, z, w) of a tube turned ``yaw_deg`` about its own axis (label bearing, local +x from world +x), then
    tilted ``tilt_deg_`` about a horizontal axis at azimuth ``tilt_az_deg``."""
    qy = np.array([0.0, 0.0, math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)])
    a = math.radians(tilt_az_deg)
    s = math.sin(math.radians(tilt_deg_) / 2)
    qt = np.array([math.cos(a) * s, math.sin(a) * s, 0.0, math.cos(math.radians(tilt_deg_) / 2)])
    return _mul(qt, qy)


def _mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def tube(x, y, z=0.0, yaw=READ, tilt=0.0, az=0.0):
    """A tube state as mechkit reports it (only what the scorer reads): bottom centre at (x, y), z above the table."""
    o = np.array([x, y, L.TOP + z])
    return SimpleNamespace(origin=o, quat_xyzw=quat(yaw, tilt, az), pos=o + [0, 0, 0.05],
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=o, high=o + [0, 0, 0.1])


def in_pos(k, yaw=READ, dx=0.0, dy=0.0, z=0.0, tilt=0.0):
    """A tube standing in analyser rack position k (1..5), offset (dx, dy) from its centre."""
    return tube(L.RACK_X + dx, L.POS_Y[k - 1] + dy, z, yaw, tilt)


def start():
    return {n: tube(*L.TUBE_START[n], yaw=math.degrees(L.TUBE_START_YAW[n])) for n in L.TUBES}


def layout(**over):
    """The four tubes as in the demo's end (P2, P1, P4, P5, all readable), with any tube replaced by ``over``."""
    st = {"tube_1": in_pos(2), "tube_2": in_pos(1), "tube_3": in_pos(4), "tube_4": in_pos(5)}
    st.update(over)
    return st


def settled(objects, steps=WINDOW + 1):
    tr = Tracker(HZ)
    for _ in range(steps):
        r = tr.update(objects, FAR)
    return tr, r


def sc(objects):
    return settled(objects)[1].score


DONE = layout()


class Geometry(unittest.TestCase):
    def test_positions_and_slots(self):
        self.assertEqual(len(L.POS_Y), 5)
        np.testing.assert_allclose(L.POS_Y, [0.056, 0.028, 0.0, -0.028, -0.056], atol=1e-12)
        self.assertAlmostEqual(L.PITCH, 0.028)
        self.assertAlmostEqual(L.NEAR_X1, L.RACK_X - 0.012)
        # neighbouring 16 mm tubes stand 12 mm apart: less than two 8 mm pads need, so beside a standing tube the
        # fingers can only go front and back (across the rack)
        self.assertAlmostEqual(L.PITCH - 2 * L.TUBE_R, 0.012)
        self.assertLess(L.PITCH - 2 * L.TUBE_R, 2 * 0.008)
        self.assertLess(L.NEAR_X0, L.NEAR_X1)  # the slotted wall is the one nearer the arms
        self.assertLess(L.SLOT_W, L.POS_IN)

    def test_rotation_and_quat(self):
        np.testing.assert_allclose(rotation(quat(90.0))[:, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(rotation(quat(READ))[:, 0], [-1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(tilt_deg(tube(0.3, 0, tilt=12.0, az=40.0)), 12.0, places=6)

    def test_points(self):
        b, t, lab = points(in_pos(3))
        np.testing.assert_allclose(b, [L.RACK_X, 0, L.TOP])
        np.testing.assert_allclose(t, [L.RACK_X, 0, L.TOP + 0.100], atol=1e-12)
        np.testing.assert_allclose(lab, [L.RACK_X - L.LABEL_OUT, 0, L.TOP + 0.030], atol=1e-12)

    def test_bearing(self):
        self.assertAlmostEqual(bearing_err_deg(in_pos(1)), 0.0, places=6)
        self.assertAlmostEqual(bearing_err_deg(in_pos(1, yaw=READ + 25)), 25.0, places=6)
        self.assertAlmostEqual(bearing_err_deg(in_pos(1, yaw=READ - 25)), 25.0, places=6)
        self.assertAlmostEqual(bearing_err_deg(in_pos(1, yaw=90)), 90.0, places=6)  # sideways (+y)
        self.assertAlmostEqual(bearing_err_deg(in_pos(1, yaw=-90)), 90.0, places=6)  # sideways (-y)
        self.assertAlmostEqual(bearing_err_deg(in_pos(1, yaw=0)), 180.0, places=6)  # toward the back wall
        # lying on its side with the label straight up: no direction seen from above
        lying = tube(0.3, 0.0, yaw=0.0)
        lying.quat_xyzw = np.array([0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
        np.testing.assert_allclose(rotation(lying.quat_xyzw)[:, 0], [0, 0, 1], atol=1e-12)
        self.assertEqual(bearing_err_deg(lying), 180.0)

    def test_position_of(self):
        for k in range(1, 6):
            self.assertEqual(position_of(in_pos(k)), k)
            self.assertEqual(position_of(in_pos(k, dx=0.0119, dy=-0.0119)), k)
        self.assertIsNone(position_of(in_pos(3, dx=0.0121)))
        self.assertIsNone(position_of(in_pos(3, dx=-0.0121)))
        self.assertIsNone(position_of(tube(L.RACK_X, (L.POS_Y[0] + L.POS_Y[1]) / 2 + 0.0)))  # on a divider
        self.assertIsNone(position_of(start()["tube_1"]))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        tr, r = settled(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(all(t["why"] == "not_in_the_analyser_rack" for t in r.details["tubes"].values()))

    def test_all_four_readable_scores_one(self):
        tr, r = settled(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["readable"], 4)

    def test_every_tier_is_reachable(self):
        away = start()
        for n_ok, want in ((0, 0.0), (1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)):
            st = {n: (DONE[n] if i < n_ok else away[n]) for i, n in enumerate(L.TUBES)}
            tr, r = settled(st)
            self.assertEqual(r.score, want)
            self.assertEqual(r.success, want == 1.0)

    def test_only_four_readable_is_success(self):
        tr, r = settled(layout(tube_4=in_pos(5, yaw=0)))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_any_four_of_the_five_positions(self):
        for empty in range(1, 6):
            ks = [k for k in range(1, 6) if k != empty]
            st = {n: in_pos(k, yaw=READ + 7 * (i - 1.5)) for i, (n, k) in enumerate(zip(L.TUBES, ks))}
            self.assertEqual(sc(st), 1.0)

    def test_which_tube_goes_where_does_not_matter(self):
        st = {"tube_1": in_pos(5), "tube_2": in_pos(3), "tube_3": in_pos(1), "tube_4": in_pos(2)}
        self.assertEqual(sc(st), 1.0)

    def test_bearing_band_edges(self):
        for d, want in ((29.9, 1.0), (-29.9, 1.0), (30.1, 0.75), (-30.1, 0.75)):
            self.assertEqual(sc(layout(tube_1=in_pos(2, yaw=READ + d))), want)

    def test_play_in_the_position_counts(self):
        # the tube can stand 4 mm off centre either way inside the 24 mm opening (16 mm tube)
        for dx, dy in ((0.004, 0.004), (-0.004, 0.004), (0.004, -0.0039), (-0.004, -0.0039)):
            self.assertEqual(sc(layout(tube_2=in_pos(1, dx=dx, dy=dy))), 1.0)

    def test_label_behind_slot_post_earns_nothing(self):
        # an off-centre tube turned part way: the label's edge is behind a post although the bearing is fine
        for sgn in (+1, -1):
            hidden = layout(tube_1=in_pos(2, dy=0.004, yaw=READ + sgn * 25))
            hidden2 = layout(tube_1=in_pos(2, dy=-0.004, yaw=READ + sgn * 25))
            self.assertEqual(sorted([sc(hidden), sc(hidden2)]), [0.75, 1.0])  # the side the label turns toward
            self.assertEqual(sc(layout(tube_1=in_pos(2, yaw=READ + sgn * 25))), 1.0)  # centred: still reads
        bad = hidden if sc(hidden) == 0.75 else hidden2
        self.assertEqual(settled(bad)[1].details["tubes"]["tube_1"]["why"], "label_behind_slot_post")
        self.assertEqual(sc(layout(tube_1=in_pos(2, dy=0.0039, yaw=READ))), 1.0)  # square to its slot: reads

    def test_tilt_edges(self):
        self.assertEqual(sc(layout(tube_3=in_pos(4, tilt=4.9))), 1.0)
        self.assertEqual(sc(layout(tube_3=in_pos(4, tilt=5.1))), 0.75)

    def test_tipped_against_a_wall_never_counts(self):
        # a tube that has tipped over rests on a 40 mm wall's top edge: at most atan(8 mm / 40 mm) = 11.3 degrees in
        # a 24 mm position, and only past its own tipping angle (8.6 degrees or more), so it never reads as upright
        for gap_mm in (4.0, 6.0, 8.0):
            lean = max(8.6, math.degrees(math.atan2(gap_mm, 1000 * L.RACK_H)))
            for az in (0.0, 90.0, 180.0, 270.0):
                st = layout(tube_1=tube(L.RACK_X, L.POS_Y[1], yaw=READ, tilt=lean, az=az))
                self.assertEqual(sc(st), 0.75, (gap_mm, az))
                self.assertEqual(settled(st)[1].details["tubes"]["tube_1"]["why"], "tilted")

    def test_bottom_height_edges(self):
        self.assertEqual(sc(layout(tube_3=in_pos(4, z=0.0049))), 1.0)
        self.assertEqual(sc(layout(tube_3=in_pos(4, z=0.0051))), 0.75)

    def test_score_is_idempotent(self):
        tr, r = settled(DONE)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(DONE, FAR).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def why(self, objects, name):
        return settled(objects)[1].details["tubes"][name]["why"]

    def test_label_sideways_not_turned(self):
        # carried straight across with no turn: every label faces sideways, 90 degrees off the slot
        st = layout(tube_1=in_pos(2, yaw=90), tube_2=in_pos(1, yaw=-90), tube_3=in_pos(4, yaw=-90),
                    tube_4=in_pos(5, yaw=90))
        tr, r = settled(st)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["racked"], 4)
        self.assertTrue(all(t["why"] == "label_turned_away" for t in r.details["tubes"].values()))

    def test_turned_the_wrong_way(self):
        # each quarter turn made the other way: the labels face the back wall
        st = layout(tube_1=in_pos(2, yaw=0), tube_2=in_pos(1, yaw=0), tube_3=in_pos(4, yaw=0),
                    tube_4=in_pos(5, yaw=0))
        self.assertEqual(sc(st), 0.0)

    def test_tilted(self):
        self.assertEqual(self.why(layout(tube_1=in_pos(2, tilt=15.0)), "tube_1"), "tilted")

    def test_upside_down(self):
        st = layout(tube_1=in_pos(2, tilt=180.0, z=0.100))
        self.assertEqual(sc(st), 0.75)
        st = layout(tube_1=in_pos(2, tilt=180.0, z=0.0))
        self.assertEqual(self.why(st, "tube_1"), "tilted")

    def test_held_in_the_air_over_a_position(self):
        st = layout(tube_2=in_pos(1, z=0.030))
        self.assertEqual(sc(st), 0.75)
        self.assertEqual(self.why(st, "tube_2"), "not_down_on_the_rack_floor")

    def test_lying_across_the_rack_top(self):
        # laid over the walls, along y, label toward the arms
        lying = tube(L.RACK_X, 0.0, z=L.RACK_H + 0.008)
        lying.quat_xyzw = _mul(np.array([math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]), quat(READ))
        self.assertEqual(sc(layout(tube_4=lying)), 0.75)

    def test_in_front_of_a_slot_outside_the_rack(self):
        # standing on the table just in front of the slotted wall, label toward the arms
        st = layout(tube_4=tube(L.NEAR_X0 - 0.012, L.POS_Y[4]))
        self.assertEqual(sc(st), 0.75)
        self.assertEqual(self.why(st, "tube_4"), "not_in_the_analyser_rack")

    def test_left_in_its_holding_cell_turned(self):
        s = start()
        st = layout(tube_3=tube(*L.TUBE_START["tube_3"], yaw=READ))
        self.assertEqual(sc(st), 0.75)
        self.assertEqual(sc(s), 0.0)

    def test_on_the_table_elsewhere_or_dropped(self):
        st = layout(tube_1=tube(0.40, 0.0, yaw=READ), tube_2=tube(0.30, 0.25, z=-0.30))  # off the table edge
        self.assertEqual(sc(st), 0.5)

    def test_two_tubes_in_one_position(self):
        # two bottoms inside one opening (a scorer edge case: real tubes would not both fit)
        st = layout(tube_1=in_pos(2, dx=0.007, dy=0.007), tube_2=in_pos(2, dx=-0.007, dy=-0.007))
        tr, r = settled(st)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["tubes"]["tube_1"]["why"], "two_tubes_in_one_position")
        self.assertEqual(r.details["tubes"]["tube_2"]["why"], "two_tubes_in_one_position")

    def test_a_tube_in_the_air_over_an_occupied_position_does_not_cancel_it(self):
        st = layout(tube_2=in_pos(2, z=0.06))  # carried over P2, where tube_1 stands: only tube_2 earns nothing
        self.assertEqual(sc(st), 0.75)
        self.assertEqual(self.why(st, "tube_1"), "")


class Rest(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        for i in range(WINDOW):
            r = tr.update(DONE, FAR)
            self.assertEqual(r.score, 0.0, i)
        self.assertEqual(tr.update(DONE, FAR).score, 1.0)

    def test_moving_scores_zero_until_at_rest(self):
        tr = Tracker(HZ)
        for i in range(30):  # lowered 1 mm a step into P5: moving
            st = layout(tube_4=in_pos(5, z=max(0.0, 0.030 - 0.001 * i)))
            r = tr.update(st, FAR)
        self.assertEqual(r.details["tubes"]["tube_4"]["why"], "still_moving")
        self.assertEqual(r.score, 0.75)
        for _ in range(WINDOW):
            r = tr.update(layout(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        for i in range(3 * WINDOW):
            r = tr.update(layout(tube_1=in_pos(2, dy=0.0009 * (-1) ** i)), FAR)
        self.assertEqual(r.score, 1.0)

    def test_spinning_in_place_counts_as_moving(self):
        tr = Tracker(HZ)
        for i in range(3 * WINDOW):  # turning 3 degrees a step about its own axis, inside the band
            r = tr.update(layout(tube_1=in_pos(2, yaw=READ - 15 + 3 * (i % 10))), FAR)
        self.assertEqual(r.details["tubes"]["tube_1"]["why"], "still_moving")
        self.assertEqual(r.score, 0.75)


class States(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(DONE)
        hist = {n: list(tr.history[n]) for n in L.TUBES}
        n = tr.steps
        bad_states = []
        for bad in (np.nan, np.inf, -np.inf):
            s = in_pos(2)
            s.origin = np.array([bad, 0.0, L.TOP])
            bad_states.append(s)
            s = in_pos(2)
            s.quat_xyzw = np.array([0.0, 0.0, bad, 1.0])
            bad_states.append(s)
        z = in_pos(2)
        z.quat_xyzw = np.zeros(4)
        bad_states.append(z)
        no_origin = SimpleNamespace(quat_xyzw=quat(READ), pos=np.zeros(3))
        bad_states.append(no_origin)
        bad_states.append(None)
        for b in bad_states:
            self.assertFalse(valid(b))
            r = tr.update(layout(tube_1=b), FAR)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + len(bad_states))
        for k in L.TUBES:
            self.assertEqual(len(tr.history[k]), len(hist[k]))
            for a, b in zip(tr.history[k], hist[k]):
                for p, q in zip(a, b):
                    np.testing.assert_array_equal(p, q)
        self.assertEqual(tr.update(DONE, FAR).score, 1.0)  # and it carries on

    def test_unnormalised_quaternion(self):
        s = in_pos(2)
        s.quat_xyzw = 3.0 * s.quat_xyzw
        self.assertTrue(valid(s))
        self.assertEqual(sc(layout(tube_1=s)), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(3)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {}
            for n in L.TUBES:
                q = rng.normal(size=4)
                o = np.array([rng.uniform(0.0, 0.5), rng.uniform(-0.4, 0.4), L.TOP + rng.uniform(-0.1, 0.3)])
                st[n] = SimpleNamespace(origin=o, quat_xyzw=q, pos=o)
            r = tr.update(st, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details)

    def test_pose_details(self):
        p = pose(in_pos(3, z=0.002, yaw=READ + 12, tilt=4))
        self.assertEqual(p["pos"], 3)
        self.assertAlmostEqual(p["bottom_mm"], 2.0)
        self.assertTrue(p["upright"] and p["facing"])


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(DONE)
        n = tr.steps
        tr.update(start(), FAR)
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_reset_after_a_full_trial_then_a_new_trial(self):
        tr, _ = settled(DONE)
        tr.update(start(), FAR)
        for _ in range(WINDOW):
            r = tr.update(DONE, FAR)
        self.assertEqual(r.score, 0.0)  # moving: it just got there (a 0.2 m jump)
        r = tr.update(DONE, FAR)
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(DONE)
        s = {n: tube(L.TUBE_START[n][0] + 0.02, L.TUBE_START[n][1] - 0.02, yaw=33.0) for n in L.TUBES}
        tr.update(s, FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        tr.update(start(), FAR)
        tr.update(start(), FAR)
        self.assertEqual(tr.steps, 2)  # the start seen again, no jump: same trial

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(DONE)
        n = tr.steps
        tr.update(layout(tube_1=tube(0.40, 0.1)), FAR)
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(DONE)
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertTrue(all(len(tr.history[n]) == 0 for n in L.TUBES))
        self.assertEqual(tr.prev_bottom, {})


class Exploits(unittest.TestCase):
    """Strategies that skip the quarter turn or the one-per-position rule earn nothing for those tubes."""

    def test_rack_without_turning(self):
        st = layout(tube_1=in_pos(2, yaw=90), tube_2=in_pos(1, yaw=-90), tube_3=in_pos(4, yaw=-90),
                    tube_4=in_pos(5, yaw=90))
        self.assertEqual(sc(st), 0.0)

    def test_half_turns_and_wrong_turns(self):
        # turning every tube the same way: two of the four end up facing the back wall
        st = layout(tube_1=in_pos(2, yaw=180), tube_2=in_pos(1, yaw=0), tube_3=in_pos(4, yaw=0),
                    tube_4=in_pos(5, yaw=180))
        self.assertEqual(sc(st), 0.5)

    def test_stuff_two_tubes_in_one_position(self):
        st = layout(tube_3=in_pos(4, dx=0.007, dy=0.007), tube_4=in_pos(4, dx=-0.007, dy=-0.007))
        self.assertEqual(sc(st), 0.5)

    def test_lean_a_tube_against_the_slot_wall(self):
        # standing inside a position but propped 11 degrees against its slotted wall, label in the slot
        self.assertEqual(sc(layout(tube_1=in_pos(2, dx=-0.004, tilt=11.0))), 0.75)

    def test_stand_tubes_on_the_holding_rack_turned(self):
        st = {n: tube(*L.TUBE_START[n], yaw=READ) for n in L.TUBES}
        self.assertEqual(sc(st), 0.0)

    def test_hold_one_tube_low_over_a_position(self):
        # a gripper holding a tube 6 mm up: not on the rack's floor
        self.assertEqual(sc(layout(tube_1=in_pos(2, z=0.006))), 0.75)

    def test_hold_a_tube_still_on_the_floor(self):
        # a gripper holding tube_1 still on P2's floor, label in the slot: it earns nothing until let go. The finger
        # zone is a cylinder round the gripper's axis (its heading is not read), 18 mm in radius closed, so it also
        # comes within 20 mm of tube_2 in P1, 28 mm away; tube_3 and tube_4 (P4, P5) still count
        hold = {"left": arm((L.RACK_X, L.POS_Y[1], L.TOP + 0.070), opening=0.08), "right": FAR["right"]}
        tr = Tracker(HZ)
        for _ in range(WINDOW + 1):
            r = tr.update(DONE, hold)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["tubes"]["tube_1"]["why"], "gripper_not_clear")
        self.assertEqual(r.details["tubes"]["tube_2"]["why"], "gripper_not_clear")
        self.assertFalse(r.success)
        for _ in range(WINDOW + 1):
            r = tr.update(DONE, FAR)  # let go and moved away
        self.assertEqual(r.score, 1.0)


class Release(unittest.TestCase):
    """The let-go rule: both grippers' finger zones at least 20 mm from a tube, read fail-closed from rigstate."""

    def test_unreadable_arms_release_nothing(self):
        bad_nan = {"left": arm((np.nan, 0, L.TOP + 0.2)), "right": FAR["right"]}
        for arms in (None, {}, {"left": FAR["left"]}, bad_nan):
            tr = Tracker(HZ)
            for _ in range(WINDOW + 1):
                r = tr.update(DONE, arms)
            self.assertEqual(r.score, 0.0)
            self.assertTrue(all(t["why"] == "gripper_not_clear" for t in r.details["tubes"].values()))

    def test_tube_distance(self):
        d = tube_distance(in_pos(3))
        x, y, z = L.RACK_X, 0.0, L.TOP
        got = d(np.array([[x, y, z + 0.05], [x, y + 0.030, z + 0.05], [x, y, z + 0.130], [x + 0.020, y, z + 0.090],
                          [x - 0.020, y, z + 0.030]]))
        np.testing.assert_allclose(got, [0.0, 0.030 - L.TUBE_R, 0.030, 0.020 - L.CAP_R, 0.020 - L.LABEL_OUT],
                                   atol=1e-9)

    def test_the_demos_moments(self):
        t = in_pos(2)
        z_grip = L.TOP + 0.070 + 0.002
        # gripping it at the place, and just opened (36 mm): not let go
        self.assertFalse(released({"left": arm((L.RACK_X, L.POS_Y[1], z_grip), 0.08), "right": FAR["right"]}, t))
        self.assertFalse(released({"left": arm((L.RACK_X, L.POS_Y[1], z_grip), 0.38), "right": FAR["right"]}, t))
        # risen straight up to the carry height (pad tips 17 mm over the closure): still too close
        self.assertFalse(released({"left": arm((L.RACK_X, L.POS_Y[1], L.TOP + 0.130)), "right": FAR["right"]}, t))
        # moved aside, or 70 mm above the closure: let go
        self.assertTrue(released(FAR, t))
        self.assertTrue(released({"left": arm((L.RACK_X, L.POS_Y[1], L.TOP + 0.183)), "right": FAR["right"]}, t))

    def test_a_gripper_beside_the_tube(self):
        t = in_pos(3)
        # an open gripper (36 mm) beside the tube at its own height, its axis 60 mm away: pads reach 29 mm, so the
        # fingers are 23 mm off the tube (let go); 45 mm away they are 8 mm off (not let go)
        for dy, want in ((0.060, True), (0.045, False)):
            arms = {"left": arm((L.RACK_X, dy, L.TOP + 0.080)), "right": FAR["right"]}
            self.assertEqual(released(arms, t), want, dy)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the
    demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, objects):
        """Synthetic tube states turned into the AssemblyStates mechkit would report from the simulator."""
        spec = {t.name: t for t in self.task.TUBES}
        return {n: self.mk.state_from_poses(spec[n], (*st.origin, *st.quat_xyzw), source="sim")
                for n, st in objects.items()}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], list(L.TUBES))
        for o in t.TUBES:
            self.assertAlmostEqual(o.mass, 0.020)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(len(t.RACK), 14)
        self.assertEqual(len(t.HOLDING), 10)

    def test_rack_walls_leave_five_openings_and_slots(self):
        t = self.task
        walls = [(np.array(w.pos) - np.array(w.size) / 2, np.array(w.pos) + np.array(w.size) / 2) for w in t.RACK]

        def solid(p):
            return any(np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9) for lo, hi in walls)

        for py in L.POS_Y:
            # the 24 x 24 mm opening is free over its full height, and the walls close it in on all four sides
            for dx in (-0.0119, 0.0, 0.0119):
                for dy in (-0.0119, 0.0, 0.0119):
                    for z in (0.001, 0.020, 0.039):
                        self.assertFalse(solid(np.array([L.RACK_X + dx, py + dy, L.TOP + z])))
            for p in ((L.RACK_X + 0.014, py, 0.02), (L.RACK_X - 0.014, py + 0.0095, 0.02), (L.RACK_X, py + 0.014, 0.02),
                      (L.RACK_X, py - 0.014, 0.02)):
                self.assertTrue(solid(np.array([p[0], p[1], L.TOP + p[2]])))
            # the reader slot: open from 6 mm up to the top, 18 mm wide; closed beside it and under it
            mid_x = (L.NEAR_X0 + L.NEAR_X1) / 2
            for dy in (-0.0089, 0.0, 0.0089):
                self.assertFalse(solid(np.array([mid_x, py + dy, L.TOP + 0.020])))
            self.assertTrue(solid(np.array([mid_x, py, L.TOP + 0.003])))
            self.assertTrue(solid(np.array([mid_x, py + 0.0095, L.TOP + 0.020])))
            self.assertTrue(solid(np.array([mid_x, py - 0.0095, L.TOP + 0.020])))
        tops = {round(w.pos[2] + w.size[2] / 2 - L.TOP, 6) for w in t.RACK}
        self.assertEqual(max(tops), L.RACK_H)

    def test_holding_cells(self):
        t = self.task
        walls = [(np.array(w.pos) - np.array(w.size) / 2, np.array(w.pos) + np.array(w.size) / 2) for w in t.HOLDING]

        def solid(p):
            return any(np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9) for lo, hi in walls)

        for n in L.TUBES:
            x, y = L.TUBE_START[n]
            self.assertFalse(solid(np.array([x + 0.0149, y - 0.0149, L.TOP + 0.01])))
            self.assertTrue(solid(np.array([x + 0.016, y, L.TOP + 0.01])))
            self.assertTrue(solid(np.array([x, y + 0.016, L.TOP + 0.01])))
            # out of the screenshot's arm shadow (|y| > 0.45 m with x < 0.20 m)
            self.assertLess(abs(y), 0.45)

    def test_scorer_geometry_matches_mechkit(self):
        spec = self.task.TUBES[0]
        lo, hi = spec.local_bounds()
        np.testing.assert_allclose(lo, [-L.CAP_R, -L.CAP_R, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.LABEL_OUT, L.CAP_R, L.TUBE_H], atol=1e-12)
        # the label is the shape on local +x, between 10 and 50 mm up: the scorer's label direction is local +x
        lab = [s for s in spec.shapes if s.kind == "box"][0]
        self.assertGreater(lab.center[0], 0.0)
        self.assertAlmostEqual(lab.center[0] + lab.size[0] / 2, L.LABEL_OUT)
        self.assertAlmostEqual(lab.center[2] + lab.size[2] / 2, L.LABEL_Z1)
        # its inner edges meet the body: corners exactly on the 8 mm radius
        self.assertAlmostEqual(math.hypot(lab.center[0] - lab.size[0] / 2, lab.size[1] / 2), L.TUBE_R)
        # each tube's spawn pose is where the scorer expects it, with the stated label direction
        for t in self.task.TUBES:
            rest = t.rest_state()
            mx, my = L.TUBE_START[t.name]
            np.testing.assert_allclose(rest.origin, [mx, my, L.TOP], atol=1e-12)
            want = (math.cos(L.TUBE_START_YAW[t.name]), math.sin(L.TUBE_START_YAW[t.name]))
            np.testing.assert_allclose(rotation(rest.quat_xyzw)[:2, 0], want, atol=1e-9)
            self.assertAlmostEqual(bearing_err_deg(rest), 90.0, places=6)  # every label starts a quarter turn off
        rests = {t.name: t.rest_state() for t in self.task.TUBES}
        self.assertTrue(Tracker.at_start(rests))
        # a mechkit state of a racked tube reads the same as the synthetic one
        st = self.sim({"tube_1": in_pos(2)})["tube_1"]
        self.assertEqual(position_of(st), 2)
        self.assertAlmostEqual(bearing_err_deg(st), 0.0, places=6)
        np.testing.assert_allclose(st.low[2], L.TOP, atol=1e-9)

    def test_tube_stands_and_fits(self):
        spec = self.task.TUBES[0]
        com = spec.center_of_mass()
        self.assertLess(abs(com[0]), 0.0015)  # the label shifts the centre of mass under 1.5 mm
        tips = [math.degrees(math.atan2(a, com[2])) for a in (L.TUBE_R - abs(com[0]), L.TUBE_R + abs(com[0]))]
        # every way it can tip is past 8.6 degrees, well past the 5-degree upright limit: a tube resting against a
        # wall never counts as upright; the most a tube can lean in a 24 mm position against a 40 mm wall is 11.3
        self.assertGreater(min(tips), 8.6)
        self.assertLess(max(tips), 10.0)
        self.assertAlmostEqual(math.degrees(math.atan2(L.POS_IN - 2 * L.TUBE_R, L.RACK_H)), 11.31, places=2)
        # 16 mm tube in a 24 mm opening: 4 mm of play each way; the label's outer corners (10.3 mm from the axis)
        # clear the walls, so a tube turns freely in its position
        self.assertAlmostEqual(L.POS_IN / 2 - L.TUBE_R, 0.004)
        self.assertGreater(L.POS_IN / 2 - math.hypot(L.LABEL_OUT, L.LABEL_W / 2), 0.0015)
        # the label shows through its 18 mm slot from 10 to 40 mm up
        self.assertLessEqual(L.LABEL_W, L.SLOT_W)
        self.assertGreater(min(L.LABEL_Z1, L.RACK_H) - max(L.LABEL_Z0, L.SLOT_Z0), 0.025)

    def test_offline_state_path(self):
        t = self.task.TUBES[0]
        c = np.array(t.offline_box()[0])
        box = SimpleNamespace(pos=c, quat_xyzw=quat(math.degrees(L.TUBE_START_YAW[t.name])),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = t.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.TUBE_START[t.name], L.TOP], atol=1e-12)

    @staticmethod
    def feed(arms=None):
        arms = FAR if arms is None else arms
        rigstate.offline_feed({a: {"tcp": v.tcp, "tips": v.tips, "opening": v.opening} for a, v in arms.items()})

    def test_hooks_and_mechkit_reset(self):
        task = self.task
        tr = task._TRACKER
        self.mk.offline_reset()
        self.feed()
        s = self.sim(start())
        self.assertFalse(task.success(s))
        self.assertEqual(task.score(s), 0.0)
        done = self.sim(DONE)
        for _ in range(WINDOW):
            task.success(done)
        self.assertTrue(task.success(done))
        self.assertEqual(task.score(done), 1.0)
        self.assertEqual(task.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(task.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        # a gripper holding tube_1 where it stands: score() and success() see it through rigstate
        self.feed({"left": arm((L.RACK_X, L.POS_Y[1], L.TOP + 0.070), opening=0.08), "right": FAR["right"]})
        for _ in range(WINDOW + 1):
            self.assertFalse(task.success(done))
        self.assertEqual(task.score(done), 0.5)  # tube_1 held, tube_2 (28 mm away) inside the zone too
        rigstate.offline_reset()
        self.assertEqual(task.score(done), 0.0)  # arms unreadable: nothing released
        self.feed()

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.feed()
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "readable", "racked", "tubes", "origins", "quats",
                          "label_faces", "source"):
                    self.assertIn(k, row)
                for k in ("pos", "racked", "readable", "why", "bearing_err_deg", "tilt_deg", "bottom_mm", "at_rest",
                          "released", "moved_mm"):
                    self.assertIn(k, row["tubes"]["tube_1"])
                self.assertEqual(row["tubes"]["tube_1"]["pos"], 2)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(DONE)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
                if prev.pos[2] > w.pos[2] and w.pos[2] < L.TOP + t.GRIP_UP + t.DROP + 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5)  # every final descent is slow
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.PLACE_APPROACH + 1e-9)
            for tube, k, yp, yq in t.PLAN[arm]:
                grip = [w for w in wps if w.label == f"grip {tube}"][0]
                place = [w for w in wps if w.label == f"slowly stand {tube} in P{k}"][0]
                self.assertAlmostEqual(abs(yq - yp), math.pi / 2)  # a quarter turn of the wrist
                end_yaw = L.TUBE_START_YAW[tube] + (place.yaw - grip.yaw)
                self.assertAlmostEqual(math.cos(end_yaw), -1.0, places=9)  # the label ends facing the slot
                self.assertAlmostEqual(grip.yaw, yp)
                np.testing.assert_allclose(grip.pos, [*L.TUBE_START[tube], L.TOP + t.GRIP_UP])
                np.testing.assert_allclose(place.pos, [L.RACK_X, L.POS_Y[k - 1], L.TOP + t.GRIP_UP + t.DROP])
        # the two tubes in each holding rack need opposite turns
        for arm in ("left", "right"):
            (a, _, ap, aq), (b, _, bp, bq) = t.PLAN[arm]
            self.assertLess((aq - ap) * (bq - bp), 0)
        # the pads hold bare tube: their tips above the label's top, their tops below the cap
        self.assertGreaterEqual(t.GRIP_UP - 0.013 - L.LABEL_Z1, 0.005)
        self.assertGreaterEqual(L.BODY_H - (t.GRIP_UP + 0.009), 0.002)
        # carried with the tube's bottom 20 mm over the rack's walls; the slow place starts over them too
        self.assertGreaterEqual(t.TRAVEL - L.TOP - t.GRIP_UP - L.RACK_H, 0.020 - 1e-9)
        self.assertGreater(t.DROP + t.PLACE_APPROACH, L.RACK_H - 0.0)
        # open widths: under the 80 mm grasp limit; 10+ mm clear of the tube; the grip 8 mm narrower than it
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.TUBE_R - 0.008)
        self.assertGreaterEqual(-t.PLACE_OPEN - L.TUBE_R, 0.010 - 1e-9)
        self.assertGreaterEqual(-t.PICK_OPEN - L.TUBE_R, 0.014 - 1e-9)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_only_one_arm_at_the_rack_at_a_time(self):
        t = self.task
        when = t.schedule()
        spans = []
        for arm, i in t.ORDER:
            tube, k, _, yq = t.PLAN[arm][i]
            go = when[(arm, i)]
            spans.append((go, go + t._duration(t._place(tube, k, yq)), arm))
        for (s0, e0, a0), (s1, e1, a1) in zip(spans, spans[1:]):
            self.assertNotEqual(a0, a1)
            self.assertGreaterEqual(s1, e0 + t.RACK_GAP - 1e-9)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("four", "one tube per position", "upright", "barcode label", "reader slot", "nearest you",
                  "let go"):
            self.assertIn(s, ins)

    def test_text_agrees_with_the_code(self):
        """task.json and the README state the numbers the code uses."""
        t = self.task
        spec = json.loads((HERE / "task.json").read_text())
        readme = (HERE / "README.md").read_text()
        self.assertEqual(spec["instruction"], t.INSTRUCTION)
        self.assertEqual(readme.count(f'--instruction "{t.INSTRUCTION}"'), 2)
        self.assertEqual(readme.count("--max-steps 1800"), 2)
        self.assertIn("1,800 steps", spec["run_limit"])
        # scoring: the thresholds in layout.py and task.py
        self.assertEqual((round(L.POS_IN * 1000), round(L.BOTTOM_TOL * 1000), L.TILT_MAX_DEG, L.BEARING_MAX_DEG,
                          round(L.RELEASE_CLEAR * 1000)), (24, 5, 5.0, 30.0, 20))
        self.assertEqual((L.PER_TUBE, round(L.STILL_TOL * 1000), L.STILL_WINDOW_S, t.SUCCESS_HOLD_S),
                         (0.25, 2, 0.5, 1.0))
        for s in ("24 x 24 mm opening and within 5 mm of the table", "within 5 degrees of upright",
                  "more than 8.6 degrees", "within 30 degrees of straight out", "Score 0.25 for each readable tube",
                  "2 mm or less over the last 0.5 s", "held for 1 s", "0, 0.25, 0.5, 0.75, 1.0",
                  "with two tubes in one position, neither counts", "within 20 mm of it", "still held or touched"):
            self.assertIn(s, spec["scoring"])
        self.assertEqual(spec["category"], "Other")
        # setup: every dimension from layout.py
        self.assertEqual([round(y * 100, 1) for y in L.POS_Y], [5.6, 2.8, 0.0, -2.8, -5.6])
        self.assertEqual((round(L.NEAR_X0 * 100, 1), round(L.NEAR_X1 * 100, 1), round(L.FAR_X1 * 100, 1),
                          round(2 * L.RACK_HALF_Y * 1000), round(L.PITCH * 1000), round((L.PITCH - 2 * L.TUBE_R) * 1000)),
                         (28.2, 28.8, 31.8, 148, 28, 12))
        self.assertEqual((round(L.WALL_T * 1000), round(L.DIV_T * 1000)), (6, 4))
        self.assertEqual((round(L.HOLD_X0 * 100, 1), round(L.HOLD_X1 * 100, 1)), (17.9, 28.1))
        self.assertEqual((round(L.TUBE_MASS * 1000), round(2 * L.TUBE_R * 1000), round(L.BODY_H * 1000),
                          round(2 * L.CAP_R * 1000, 1), round(L.CAP_H * 1000)), (20, 16, 82, 16.5, 18))
        for s in ("each 24 x 24 mm inside", "centred at x 30.0 cm and y +5.6, +2.8, 0, -2.8 and -5.6 cm",
                  "28 mm apart", "4 mm dividers inside 6 mm outer walls", "all 40 mm tall", "x 28.2 to 31.8 cm",
                  "148 mm", "x 28.2 to 28.8 cm", "18 mm wide", "from 6 mm above the table", "12 mm apart",
                  "102 x 42 mm outside (x 17.9 to 28.1 cm) and 20 mm tall", "centred at x 20.0 and 26.0 cm",
                  "y +20.0 cm", "20 g each", "16 mm diameter", "82 mm tall", "16.5 mm across and 18 mm tall",
                  "10 mm wide on the side of the tube from 10 to 50 mm above its bottom", "1 mm proud"):
            self.assertIn(s, spec["setup"])
        self.assertIn("4 mm off centre", spec["fidelity"])
        self.assertIn("Pending re-run", spec["scores"])
        self.assertEqual((round(L.SLOT_W * 1000), round(L.SLOT_Z0 * 1000), round(L.RACK_H * 1000),
                          round(L.HOLD_H * 1000), round(L.LABEL_W * 1000), round(L.LABEL_Z0 * 1000),
                          round(L.LABEL_Z1 * 1000), round((L.LABEL_OUT - L.TUBE_R) * 1000)),
                         (18, 6, 40, 20, 10, 10, 50, 1))
        # the starting label directions, in setup and reset_notes
        side = {+1: "left (+y)", -1: "right (-y)"}
        for n, (x, y) in L.TUBE_START.items():
            d = side[round(math.sin(L.TUBE_START_YAW[n]))]
            self.assertIn(f"x {round(x * 100)}, y {'+' if y > 0 else '-'}{round(abs(y) * 100)}", spec["setup"])
            self.assertTrue(d.split()[0] in spec["reset_notes"])
        self.assertIn("x 20, y +20 cm faces its label to the robot's left (+y); the one at x 26, y +20 to the right "
                      "(-y); x 20, y -20 to the right (-y); x 26, y -20 to the left (+y)", spec["setup"])


if __name__ == "__main__":
    unittest.main()
