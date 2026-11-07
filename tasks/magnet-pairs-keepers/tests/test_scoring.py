"""Offline tests for scoring.py and task.py: synthetic magnet and keeper states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    KEEPERS,
    MAGNETS,
    Tracker,
    end_face,
    evaluate,
    keeper_at_end,
    keeper_pose,
    magnet_ends,
    magnet_pose,
    pair_measure,
    red_sign,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FLOOR_W = L.FLOOR_TOP_W


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


MAG_LOCAL = np.array([(sx * 0.05, sy * 0.01, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.015)])
KEEP_LOCAL = np.array([(sx * 0.01, sy * 0.037, sz * 0.0075) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def mag(cx, cy, yaw=0.0, bottom=FLOOR_W, pitch=0.0, roll=0.0):
    """A magnet as mechkit reports it: yaw 0 = red end toward +x (away from the arms), lying flat. Its bottom-face
    middle is over (cx, cy) and its lowest corner at ``bottom`` (world z)."""
    q = quat(yaw, pitch, roll)
    r = _rot(q)
    o = np.array([cx, cy, 0.0])
    o[2] = bottom - float((MAG_LOCAL @ r.T)[:, 2].min())
    c = o + MAG_LOCAL @ r.T
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(c.min(0) + c.max(0)) / 2, lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), low=c.min(0), high=c.max(0), source="test")


def keeper(cx, cy, yaw=0.0, bottom=FLOOR_W, pitch=0.0, roll=0.0):
    """A keeper: yaw 0 = its 74 mm length along y (across the tray), lying flat with its lowest corner at
    ``bottom``. roll turns its length out of level (about its 20 mm width)."""
    q = quat(yaw, pitch, roll)
    r = _rot(q)
    p = np.array([cx, cy, 0.0])
    p[2] = bottom - float((KEEP_LOCAL @ r.T)[:, 2].min())
    c = p + KEEP_LOCAL @ r.T
    return SimpleNamespace(pos=p, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=c.min(0),
                           high=c.max(0))


def state(**kw):
    """The finished layout, with any object replaced by keyword."""
    s = {
        "magnet_a": mag(L.PAIR_X, L.PAIR_DY, 0.0),  # red end away from the arms
        "magnet_b": mag(L.PAIR_X, -L.PAIR_DY, 180.0),  # red end toward the arms: unlike poles side by side
        "keeper_1": keeper(L.KEEP_FAR_X, 0.0),
        "keeper_2": keeper(L.KEEP_NEAR_X, 0.0),
    }
    s.update(kw)
    return s


START = {
    "magnet_a": mag(L.MAG_A_START[0], L.MAG_A_START[1], 90.0, bottom=L.TOP),
    "magnet_b": mag(L.MAG_B_START[0], L.MAG_B_START[1], -90.0, bottom=L.TOP),
    "keeper_1": keeper(*L.KEEP_1_START, bottom=L.TOP),
    "keeper_2": keeper(*L.KEEP_2_START, bottom=L.TOP),
}
OFF_K1 = keeper(*L.KEEP_1_START, bottom=L.TOP)  # a keeper still on its starting mark
OFF_K2 = keeper(*L.KEEP_2_START, bottom=L.TOP)
FAR_X, NEAR_X = L.PAIR_X + L.HALF_LEN, L.PAIR_X - L.HALF_LEN  # the pair's end faces


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(st, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n)


def scored(st):
    return settled(st)[1].score


def detail(st):
    return settled(st)[1].details


def far_keeper_with_gap(gap, dy=0.0, yaw=0.0):
    return keeper(FAR_X + gap + 0.010, dy, yaw)


def near_keeper_with_gap(gap, dy=0.0):
    return keeper(NEAR_X - gap - 0.010, dy)


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(valid(START))
        self.assertTrue(Tracker.at_start(START))
        self.assertFalse(Tracker.at_start(state()))
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(red_sign(START["magnet_a"]), 0)  # lying across the table: neither way along the tray
        # the start is a mirror image: red ends outward on both sides
        ra, _ = magnet_ends(START["magnet_a"])
        rb, _ = magnet_ends(START["magnet_b"])
        np.testing.assert_allclose(ra, [0.22, 0.33, L.TOP + 0.0075], atol=1e-12)
        np.testing.assert_allclose(rb, [0.22, -0.33, L.TOP + 0.0075], atol=1e-12)

    def test_red_end_and_poles(self):
        self.assertEqual(red_sign(mag(0.26, 0.02, 0.0)), 1)
        self.assertEqual(red_sign(mag(0.26, 0.02, 180.0)), -1)
        self.assertEqual(red_sign(mag(0.26, 0.02, 175.0)), -1)
        red, blue = magnet_ends(mag(0.26, 0.0, 0.0))
        np.testing.assert_allclose(red, [0.31, 0.0, FLOOR_W + 0.0075], atol=1e-12)
        np.testing.assert_allclose(blue, [0.21, 0.0, FLOOR_W + 0.0075], atol=1e-12)

    def test_magnet_pose_measures(self):
        p = magnet_pose(mag(0.26, 0.0225, 7.0, pitch=3.0))
        self.assertAlmostEqual(p["heading_deg"], 7.0, places=6)
        self.assertAlmostEqual(p["level_deg"], 3.0, places=6)
        self.assertAlmostEqual(p["low_mm"], 0.0, places=6)
        self.assertTrue(p["stored_pose"] and p["in_tray"] and p["lengthwise"])
        self.assertFalse(magnet_pose(START["magnet_a"])["in_tray"])

    def test_end_faces_pick_the_side_not_the_pole(self):
        for yaw in (0.0, 180.0):
            e, (y0, y1) = end_face(mag(0.26, 0.0225, yaw), 1)
            self.assertAlmostEqual(e[0], FAR_X)
            self.assertAlmostEqual(y1 - y0, 0.020)
            e, _ = end_face(mag(0.26, 0.0225, yaw), -1)
            self.assertAlmostEqual(e[0], NEAR_X)

    def test_pair_measure(self):
        across, along = pair_measure(mag(0.26, 0.0225), mag(0.27, -0.0225, 180.0))
        self.assertAlmostEqual(across, 0.045)
        self.assertAlmostEqual(along, 0.010)

    def test_keeper_measures_at_the_aim(self):
        s = state()
        mags = [s[n] for n in MAGNETS]
        far = keeper_at_end(s["keeper_1"], mags, 1)
        near = keeper_at_end(s["keeper_2"], mags, -1)
        for at in (far, near):
            self.assertTrue(at["ok"])
            np.testing.assert_allclose(at["gaps_mm"], [4.0, 4.0], atol=1e-6)
            np.testing.assert_allclose(at["cover_mm"], [20.0, 20.0], atol=1e-6)
        self.assertFalse(keeper_at_end(s["keeper_1"], mags, -1)["ok"])  # it is not at the near end
        kp = keeper_pose(s["keeper_1"])
        self.assertTrue(kp["on_floor"] and kp["across"])

    def test_tray_holds_the_layout(self):
        # the target layout leaves 11 mm between each keeper and its end rim, and fits across
        self.assertAlmostEqual(L.KEEP_FAR_X + 0.010, L.INNER_X1 - 0.011)
        self.assertAlmostEqual(L.KEEP_NEAR_X - 0.010, L.INNER_X0 + 0.011)
        self.assertLess(L.PAIR_DY + L.MAG_W / 2, L.INNER_Y)
        self.assertLess(L.KEEP_HALF_LEN, L.INNER_Y)


class Tiers(unittest.TestCase):
    def test_finished_pair_scores_one(self):
        r = settled(state())[1]
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["far_end_closed"] and r.details["near_end_closed"])

    def test_the_other_unlike_way_round_also_scores_one(self):
        self.assertEqual(scored(state(magnet_a=mag(L.PAIR_X, L.PAIR_DY, 180.0),
                                      magnet_b=mag(L.PAIR_X, -L.PAIR_DY, 0.0))), 1.0)

    def test_twins_are_interchangeable(self):
        s = state()
        self.assertEqual(scored(state(magnet_a=s["magnet_b"], magnet_b=s["magnet_a"])), 1.0)
        self.assertEqual(scored(state(keeper_1=s["keeper_2"], keeper_2=s["keeper_1"])), 1.0)

    def test_one_end_closed_scores_three_quarters(self):
        self.assertEqual(scored(state(keeper_1=OFF_K1)), 0.75)
        self.assertEqual(scored(state(keeper_2=OFF_K2)), 0.75)

    def test_pair_without_keepers_scores_half(self):
        self.assertEqual(scored(state(keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.5)

    def test_one_magnet_stored_scores_quarter(self):
        self.assertEqual(scored(state(magnet_b=START["magnet_b"], keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.25)
        self.assertEqual(scored(state(magnet_a=START["magnet_a"])), 0.25)  # keepers count for nothing without a pair

    def test_both_stored_not_side_by_side_scores_quarter(self):
        apart = state(magnet_a=mag(0.26, 0.042), magnet_b=mag(0.26, -0.042, 180.0))  # centres 84 mm apart
        self.assertEqual(scored(apart), 0.25)
        stagger = state(magnet_a=mag(0.27, 0.0225), magnet_b=mag(0.25, -0.0225, 180.0))  # 20 mm along
        self.assertEqual(scored(stagger), 0.25)
        self.assertEqual(detail(stagger)["zero_reason"], "stored_not_side_by_side")
        # 61 mm apart, the keepers at the aim still touch both end faces (16.5 mm of each): not a pair, so no end
        # counts as closed
        wide = state(magnet_a=mag(0.26, 0.0305), magnet_b=mag(0.26, -0.0305, 180.0))
        self.assertEqual(scored(wide), 0.25)
        d = detail(wide)
        self.assertTrue(d["keeper_1"]["far"]["ok"])
        self.assertFalse(d["far_end_closed"] or d["near_end_closed"])

    def test_nothing_stored_scores_zero(self):
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(detail(START)["zero_reason"], "no_magnet_stored")

    def test_every_tier_is_reachable(self):
        got = {scored(START), scored(state(magnet_b=START["magnet_b"])), scored(state(keeper_1=OFF_K1, keeper_2=OFF_K2)),
               scored(state(keeper_1=OFF_K1)), scored(state())}
        self.assertEqual(got, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_score_is_idempotent(self):
        tr, r = settled(state())
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(state()).score, r.score)
        self.assertEqual(tr.steps, n)

    def test_random_states_map_to_exactly_one_tier(self):
        rng = random.Random(7)
        for _ in range(300):
            s = {
                "magnet_a": mag(rng.uniform(0.18, 0.34), rng.uniform(-0.06, 0.06), rng.choice([0.0, 180.0, 90.0])
                                + rng.uniform(-12, 12), bottom=FLOOR_W + rng.choice([0.0, 0.0, 0.015])),
                "magnet_b": mag(rng.uniform(0.18, 0.34), rng.uniform(-0.06, 0.06), rng.choice([0.0, 180.0])
                                + rng.uniform(-12, 12)),
                "keeper_1": keeper(rng.uniform(0.18, 0.34), rng.uniform(-0.02, 0.02), rng.uniform(-20, 20)),
                "keeper_2": keeper(rng.uniform(0.18, 0.34), rng.uniform(-0.02, 0.02), rng.uniform(-20, 20)),
            }
            self.assertIn(scored(s), (0.0, 0.25, 0.5, 0.75, 1.0))


class Edges(unittest.TestCase):
    def test_pair_spacing_edges(self):
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.030), magnet_b=mag(0.26, -0.0299, 180.0),
                                      keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.5)  # 59.9 mm across
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0305), magnet_b=mag(0.26, -0.0305, 180.0),
                                      keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.25)  # 61 mm
        self.assertEqual(scored(state(magnet_a=mag(0.2675, 0.0225), magnet_b=mag(0.2526, -0.0225, 180.0),
                                      keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.5)  # 14.9 mm along
        self.assertEqual(scored(state(magnet_a=mag(0.268, 0.0225), magnet_b=mag(0.252, -0.0225, 180.0),
                                      keeper_1=OFF_K1, keeper_2=OFF_K2)), 0.25)  # 16 mm along

    def test_touching_magnets_are_a_pair(self):
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.010), magnet_b=mag(0.26, -0.010, 180.0))), 1.0)

    def test_keeper_gap_edges(self):
        for gap, want in ((-0.0019, 1.0), (0.0, 1.0), (0.0079, 1.0), (-0.0025, 0.75), (0.0085, 0.75)):
            self.assertEqual(scored(state(keeper_1=far_keeper_with_gap(gap))), want, gap)
            self.assertEqual(scored(state(keeper_2=near_keeper_with_gap(gap))), want, gap)

    def test_keeper_cover_edges(self):
        self.assertEqual(scored(state(keeper_1=far_keeper_with_gap(0.004, dy=0.012))), 1.0)  # 12.5 mm of B's face
        self.assertEqual(scored(state(keeper_1=far_keeper_with_gap(0.004, dy=-0.013))), 0.75)  # 11.5 mm of A's
        d = detail(state(keeper_1=far_keeper_with_gap(0.004, dy=-0.013)))
        self.assertAlmostEqual(d["keeper_1"]["far"]["cover_mm"][0], 11.5, places=6)

    def test_small_keeper_skew_counts_and_a_large_one_cannot_close(self):
        self.assertEqual(scored(state(keeper_1=far_keeper_with_gap(0.004, yaw=5.0))), 1.0)
        # skewed 15 degrees the keeper's near edge is 12 mm farther from one magnet than the other: no gap fits
        for x in np.arange(0.315, 0.335, 0.0005):
            self.assertLess(scored(state(keeper_1=keeper(float(x), 0.0, 15.0))), 1.0)
        self.assertFalse(keeper_pose(keeper(0.324, 0.0, 16.0))["across"])

    def test_keeper_lying_on_its_edge_still_closes(self):
        k = keeper(L.KEEP_FAR_X - 0.0025, 0.0, pitch=90.0)  # on its 15 mm side: 15 wide, 20 tall; 4 mm gap
        self.assertEqual(scored(state(keeper_1=k)), 1.0)

    def test_keeper_height_and_level_edges(self):
        self.assertEqual(scored(state(keeper_1=keeper(L.KEEP_FAR_X, 0.0, bottom=FLOOR_W + 0.0029))), 1.0)
        self.assertEqual(scored(state(keeper_1=keeper(L.KEEP_FAR_X, 0.0, bottom=FLOOR_W + 0.0035))), 0.75)
        self.assertEqual(scored(state(keeper_1=keeper(L.KEEP_FAR_X, 0.0, roll=9.0))), 1.0)
        self.assertEqual(scored(state(keeper_1=keeper(L.KEEP_FAR_X, 0.0, roll=11.0))), 0.75)

    def test_stored_magnet_edges(self):
        open_keepers = {"keeper_1": OFF_K1, "keeper_2": OFF_K2}
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, 9.0), **open_keepers)), 0.5)
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, 11.0), **open_keepers)), 0.25)
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, pitch=4.5), **open_keepers)), 0.5)
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, pitch=6.0), **open_keepers)), 0.25)
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, bottom=FLOOR_W + 0.0029), **open_keepers)), 0.5)
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.0225, bottom=FLOOR_W + 0.0035), **open_keepers)), 0.25)

    def test_against_the_rims(self):
        open_keepers = {"keeper_1": OFF_K1, "keeper_2": OFF_K2}
        # pushed against the side rim (outer face on its inner face): stored, but 63 mm from its twin: not a pair
        self.assertEqual(detail(state(magnet_a=mag(0.26, 0.053), **open_keepers))["magnet_a"]["stored"], True)
        self.assertEqual(detail(state(magnet_a=mag(0.26, 0.0545), **open_keepers))["magnet_a"]["stored"], False)
        # against the end rim
        self.assertTrue(detail(state(magnet_a=mag(L.INNER_X1 - 0.050, 0.0225), **open_keepers))["magnet_a"]["stored"])
        self.assertFalse(detail(state(magnet_a=mag(L.INNER_X1 - 0.048, 0.0225), **open_keepers))["magnet_a"]["stored"])

    def test_magnet_on_its_side_counts(self):
        side = mag(L.PAIR_X, L.PAIR_DY + 0.0075, 0.0, roll=90.0)  # 15 wide, 20 tall, centred where it was
        self.assertAlmostEqual(float(np.mean(side.pos[1])), L.PAIR_DY)
        self.assertEqual(scored(state(magnet_a=side)), 1.0)


class ZeroConditions(unittest.TestCase):
    def test_like_poles_side_by_side_is_zero_whatever_else(self):
        like_far = state(magnet_b=mag(L.PAIR_X, -L.PAIR_DY, 0.0))  # both red ends away from the arms
        like_near = state(magnet_a=mag(L.PAIR_X, L.PAIR_DY, 180.0))  # both toward them
        for s in (like_far, like_near):
            self.assertEqual(scored(s), 0.0)  # keepers on both ends do not help
            self.assertEqual(detail(s)["zero_reason"], "like_poles_side_by_side")
            self.assertEqual(scored({**s, "keeper_1": OFF_K1, "keeper_2": OFF_K2}), 0.0)

    def test_the_mirror_image_move_is_the_trap(self):
        """Each arm turning its magnet a quarter turn the mirror image of the other's: like poles."""
        a = mag(L.PAIR_X, L.PAIR_DY, 90.0 - 90.0)  # left magnet turned clockwise: red away
        b = mag(L.PAIR_X, -L.PAIR_DY, -90.0 + 90.0)  # right magnet turned counter-clockwise: red away too
        self.assertEqual(scored(state(magnet_a=a, magnet_b=b)), 0.0)
        b_same = mag(L.PAIR_X, -L.PAIR_DY, -90.0 - 90.0)  # both clockwise
        self.assertEqual(scored(state(magnet_a=a, magnet_b=b_same)), 1.0)

    def test_like_poles_spread_apart_or_crooked_is_still_zero(self):
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.045), magnet_b=mag(0.26, -0.045))), 0.0)
        self.assertEqual(scored(state(magnet_b=mag(0.26, -0.0225, 25.0))), 0.0)  # 25 deg off the tray's length
        self.assertEqual(scored(state(magnet_b=mag(0.26, -0.0225, 0.0, pitch=11.0))), 0.0)  # propped up
        # one stacked on the other the same way round
        self.assertEqual(scored(state(magnet_b=mag(L.PAIR_X, 0.0225, 0.0, bottom=FLOOR_W + 0.015))), 0.0)

    def test_crosswise_magnet_is_not_like_poles_but_not_stored(self):
        s = state(magnet_b=mag(0.26, -0.02, 90.0), keeper_1=OFF_K1, keeper_2=OFF_K2)
        self.assertEqual(scored(s), 0.25)

    def test_magnets_off_the_tray(self):
        self.assertEqual(scored(state(magnet_a=mag(0.26, 0.10, bottom=L.TOP), magnet_b=mag(0.26, -0.10, 180.0,
                                                                                        bottom=L.TOP))), 0.0)


class Exploits(unittest.TestCase):
    def test_keepers_laid_on_top_of_the_magnets(self):
        on_top = state(keeper_1=keeper(FAR_X - 0.008, 0.0, bottom=FLOOR_W + 0.015),
                       keeper_2=keeper(NEAR_X + 0.008, 0.0, bottom=FLOOR_W + 0.015))
        self.assertEqual(scored(on_top), 0.5)

    def test_keepers_lengthwise_beside_the_pair(self):
        along = state(keeper_1=keeper(0.26, 0.050, 90.0), keeper_2=keeper(0.26, -0.050, 90.0))
        self.assertEqual(scored(along), 0.5)

    def test_keeper_standing_on_end(self):
        self.assertEqual(scored(state(keeper_1=keeper(L.KEEP_FAR_X, 0.0, roll=90.0))), 0.75)

    def test_both_keepers_at_one_end(self):
        stacked_out = state(keeper_1=far_keeper_with_gap(0.004), keeper_2=far_keeper_with_gap(0.026))
        self.assertEqual(scored(stacked_out), 0.75)

    def test_keepers_closing_a_lone_magnet(self):
        lone = state(magnet_b=START["magnet_b"], keeper_1=keeper(L.KEEP_FAR_X, L.PAIR_DY),
                     keeper_2=keeper(L.KEEP_NEAR_X, L.PAIR_DY))
        self.assertEqual(scored(lone), 0.25)
        d = detail(lone)  # keepers never count as closing an end without a pair, in the trace either
        self.assertFalse(d["far_end_closed"] or d["near_end_closed"])

    def test_magnet_held_over_the_tray(self):
        held = state(magnet_b=mag(L.PAIR_X, -L.PAIR_DY, 180.0, bottom=FLOOR_W + 0.030))
        self.assertEqual(scored(held), 0.25)

    def test_magnet_standing_on_end(self):
        self.assertEqual(scored(state(magnet_b=mag(L.PAIR_X, -L.PAIR_DY, 180.0, pitch=90.0))), 0.25)

    def test_magnet_propped_on_the_rim(self):
        # its far end on the end rim's top (13 mm up), its near end on the floor: tilted and over the rim
        x0 = L.INNER_X1 - 0.090
        pitch = -math.degrees(math.asin(0.013 / 0.100))
        s = state(magnet_a=mag(x0 + 0.05, 0.0225, 0.0, pitch=pitch))
        self.assertLess(scored(s), 0.5)

    def test_unlike_pair_stacked(self):
        stacked = state(magnet_b=mag(L.PAIR_X, L.PAIR_DY, 180.0, bottom=FLOOR_W + 0.015))
        self.assertEqual(scored(stacked), 0.25)


class Stillness(unittest.TestCase):
    def test_moving_magnet_is_not_stored(self):
        tr = Tracker(HZ)
        feed(tr, [state()] * WINDOW)
        moving = [state(magnet_a=mag(L.PAIR_X + 0.003 * i, L.PAIR_DY)) for i in range(1, 4)]
        r = feed(tr, moving)
        self.assertEqual(r.details["magnet_a"]["stored"], False)
        self.assertLess(r.score, 1.0)
        r = feed(tr, [moving[-1]] * (WINDOW + 1))
        self.assertTrue(r.details["magnet_a"]["at_rest"])

    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [state()] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        rng = random.Random(3)
        states = [state(magnet_a=mag(L.PAIR_X + rng.uniform(-0.0008, 0.0008), L.PAIR_DY + rng.uniform(-0.0008, 0.0008)),
                        keeper_1=keeper(L.KEEP_FAR_X + rng.uniform(-0.0008, 0.0008), 0.0))
                  for _ in range(40)]
        self.assertEqual(feed(tr, states).score, 1.0)

    def test_moving_keeper_does_not_close(self):
        tr = Tracker(HZ)
        feed(tr, [state()] * (WINDOW + 1))
        r = tr.update(state(keeper_1=keeper(L.KEEP_FAR_X + 0.003, 0.0)))
        self.assertEqual(r.score, 0.75)

    def test_a_turn_in_place_counts_as_moving(self):
        tr = Tracker(HZ)
        feed(tr, [state()] * (WINDOW + 1))
        r = tr.update(state(magnet_a=mag(L.PAIR_X, L.PAIR_DY, 3.0)))  # 3 deg about its middle: ends move 2.6 mm
        self.assertFalse(r.details["magnet_a"]["at_rest"])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(state())
        hist = {n: list(tr.history[n]) for n in tr.history}
        prev = dict(tr.prev)
        for name in (*MAGNETS, *KEEPERS):
            for field_, bad in (("pos", np.array([np.nan, 0, 0])), ("quat_xyzw", np.zeros(4)),
                                ("quat_xyzw", np.array([0, 0, np.inf, 1.0]))):
                s = state()
                ns = SimpleNamespace(**vars(s[name]))
                setattr(ns, field_, bad)
                s[name] = ns
                r = tr.update(s)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertFalse(r.details["valid_state"])
        for n in tr.history:
            self.assertEqual(len(tr.history[n]), len(hist[n]))
        for n in prev:
            np.testing.assert_allclose(tr.prev[n], prev[n])
        self.assertEqual(tr.update(state()).score, 1.0)  # carries on as before

    def test_magnet_without_origin_or_missing_object(self):
        s = state()
        plain = SimpleNamespace(**{k: v for k, v in vars(s["magnet_a"]).items() if k != "origin"})
        self.assertFalse(valid({**s, "magnet_a": plain}))
        self.assertFalse(valid({k: v for k, v in s.items() if k != "keeper_2"}))
        self.assertEqual(Tracker(HZ).update({k: v for k, v in s.items() if k != "keeper_2"}).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = state()
        for n in s:
            s[n].quat_xyzw = s[n].quat_xyzw * 3.7
        self.assertEqual(scored(s), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(11)
        tr = Tracker(HZ)
        for _ in range(300):
            s = {}
            for n in MAGNETS:
                q = rng.normal(size=4)
                o = rng.uniform([-0.5, -0.7, 0.6], [0.8, 0.7, 1.2])
                s[n] = SimpleNamespace(origin=o, quat_xyzw=q, pos=o, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                       low=o - 0.05, high=o + 0.05)
            for n in KEEPERS:
                p = rng.uniform([-0.5, -0.7, 0.6], [0.8, 0.7, 1.2])
                s[n] = SimpleNamespace(pos=p, quat_xyzw=rng.normal(size=4), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                       low=p - 0.04, high=p + 0.04)
            self.assertIn(tr.update(s).score, (0.0, 0.25, 0.5, 0.75, 1.0))


def glide(a, b, n):
    """n layouts moving every object in a straight line from layout a to layout b (no turning)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        s = {}
        for k in a:
            if k in MAGNETS:
                o = np.asarray(a[k].origin) + (np.asarray(b[k].origin) - np.asarray(a[k].origin)) * f
                s[k] = SimpleNamespace(**{**vars(b[k]), "origin": o})
            else:
                p = np.asarray(a[k].pos) + (np.asarray(b[k].pos) - np.asarray(a[k].pos)) * f
                s[k] = SimpleNamespace(**{**vars(b[k]), "pos": p})
        out.append(s)
    return out


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(state())
        tr.update(START)
        self.assertEqual(feed(tr, [state()] * WINDOW).score, 0.0)  # history starts over
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(state())
        jit = {
            "magnet_a": mag(L.MAG_A_START[0] + 0.02, L.MAG_A_START[1] - 0.02, 90.0, bottom=L.TOP),
            "magnet_b": mag(L.MAG_B_START[0] - 0.02, L.MAG_B_START[1] + 0.01, -90.0, bottom=L.TOP),
            "keeper_1": keeper(L.KEEP_1_START[0] + 0.02, L.KEEP_1_START[1], bottom=L.TOP),
            "keeper_2": keeper(L.KEEP_2_START[0], L.KEEP_2_START[1] - 0.02, bottom=L.TOP),
        }
        self.assertTrue(Tracker.at_start(jit))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(state())
        feed(tr, glide(state(), START, 60))
        self.assertGreater(tr.steps, 60)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(state())
        n = tr.steps
        tr.update(state(magnet_a=mag(0.40, 0.40, 0.0, bottom=L.TOP)))
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(state())
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.update(state()).score, 0.0)


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
        sys.path.insert(0, str(_stub_dir().parent))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, st):
        """Synthetic states turned into what yam_arena and mechkit would report from the simulator."""
        out = dict(st)
        for n, spec in (("magnet_a", self.task.MAGNET_A), ("magnet_b", self.task.MAGNET_B)):
            out[n] = self.mk.state_from_poses(spec, (*st[n].origin, *st[n].quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted([*MAGNETS, *KEEPERS]))
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 2.5)
        floor = [o for o in t.OBJECTS if o.name == "tray_floor"][0]
        self.assertAlmostEqual(floor.pos[2] + floor.size[2] / 2, FLOOR_W)
        far = [o for o in t.OBJECTS if o.name == "tray_rim_far"][0]
        self.assertAlmostEqual(far.pos[0] - far.size[0] / 2, L.INNER_X1)
        left = [o for o in t.OBJECTS if o.name == "tray_rim_left"][0]
        self.assertAlmostEqual(left.pos[1] - left.size[1] / 2, L.INNER_Y)
        self.assertAlmostEqual(left.pos[2] + left.size[2] / 2, L.TOP + L.RIM_TOP)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, o.size, 0.0)
            half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)

    def test_scorer_geometry_matches_mechkit(self):
        for spec, start, yaw in ((self.task.MAGNET_A, L.MAG_A_START, 90.0), (self.task.MAGNET_B, L.MAG_B_START, -90.0)):
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-0.05, -0.01, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [0.05, 0.01, 0.015], atol=1e-12)
            red = [s for s in spec.shapes if tuple(s.color) == L.NORTH_RED][0]
            self.assertGreater(red.center[0], 0.0)  # the red half is at local +x, the end the scorer calls red
            np.testing.assert_allclose(spec.center_of_mass(), [0.0, 0.0, 0.0075], atol=1e-9)
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [start[0], start[1], L.TOP], atol=1e-12)
        rest = {"magnet_a": self.task.MAGNET_A.rest_state(), "magnet_b": self.task.MAGNET_B.rest_state(),
                "keeper_1": START["keeper_1"], "keeper_2": START["keeper_2"]}
        self.assertTrue(Tracker.at_start(rest))
        sim = self.sim(state())
        np.testing.assert_allclose(sim["magnet_a"].low[2], FLOOR_W, atol=1e-9)
        self.assertEqual(scored(sim), 1.0)

    def test_offline_state_path(self):
        spec = self.task.MAGNET_A
        box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(90.0), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.MAG_A_START[0], L.MAG_A_START[1], L.TOP], atol=1e-12)
        self.assertEqual(red_sign(st), 0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(state())
        for _ in range(WINDOW):
            self.task.success(done)
        self.assertTrue(self.task.success(done))
        self.assertEqual(self.task.score(done), 1.0)
        self.assertEqual(self.task.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(self.task.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(state()))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "moved_mm", "like_poles", "n_stored", "paired",
                          "pair_across_mm", "pair_along_mm", "far_end_closed", "near_end_closed", "magnet_a",
                          "keeper_1", "magnet_a_origin", "magnet_b_red_sign", "keeper_2_pos", "magnet_a_source"):
                    self.assertIn(k, row)
                self.assertEqual(row["keeper_1"]["far"]["gaps_mm"], [4.0, 4.0])
                self.assertEqual(row["magnet_b_red_sign"], -1)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(state())))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    @staticmethod
    def _timeline(wps):
        t, out = 0.0, []
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            # every close and open takes 1.5 s, and so does every last 30 mm down to a grip or a place
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                if prev.pos[2] > w.pos[2] and min(abs(w.pos[2] - z) for z in (t.PICK_Z, t.PLACE_Z, t.KEEP_PICK_Z,
                                                                              t.KEEP_PLACE_Z)) < 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
            # both arms turn their magnet a quarter turn clockwise: unlike poles in the tray
            grip = [w for w in wps if w.label == "grip the magnet"][0]
            place = [w for w in wps if w.label == "slowly lay the magnet on the tray floor"][0]
            self.assertAlmostEqual(math.remainder(place.yaw - grip.yaw, 2 * math.pi), -math.pi / 2)
        start_yaw = {"left": L.MAG_A_YAW, "right": L.MAG_B_YAW}
        ends = {arm: math.cos(start_yaw[arm] - math.pi / 2) for arm in plan}
        self.assertAlmostEqual(ends["left"], 1.0)  # red end away from the arms
        self.assertAlmostEqual(ends["right"], -1.0)  # red end toward them
        # pad tips (13 mm below the TCP) 4 mm above a magnet's underside and 8 mm above a keeper's, at every pick and
        # place; each grip and place uses its part's height
        self.assertAlmostEqual(t.PICK_Z - 0.013 - L.TOP, 0.004)
        self.assertAlmostEqual(t.PLACE_Z - 0.013 - (FLOOR_W + t.DROP), 0.004)
        self.assertAlmostEqual(t.KEEP_PICK_Z - 0.013 - L.TOP, 0.008)
        self.assertAlmostEqual(t.KEEP_PLACE_Z - 0.013 - (FLOOR_W + t.DROP), 0.008)
        for arm, wps in plan.items():
            for w in wps:
                if w.label in ("grip the magnet", "grip the keeper", "slowly lay the magnet on the tray floor",
                               "slowly lay the keeper on the tray floor"):
                    want = {("grip", "magnet"): t.PICK_Z, ("grip", "keeper"): t.KEEP_PICK_Z,
                            ("slowly", "magnet"): t.PLACE_Z, ("slowly", "keeper"): t.KEEP_PLACE_Z}
                    key = (w.label.split()[0], "keeper" if "keeper" in w.label else "magnet")
                    self.assertAlmostEqual(w.pos[2], want[key], msg=(arm, w.label))
        # a keeper's pads (about 27 mm wide at the TCP, 15 mm at the tips 13 mm below it) pass the magnets' top
        # corners (15 mm above the floor) with room to spare at the 4 mm aim, and clear them down to a 1 mm gap
        tcp_up = t.KEEP_GRIP_UP  # above the keeper's underside, i.e. the floor once it rests there
        width_at_corner = 0.015 + 0.012 * (L.MAG_H - (tcp_up - 0.013)) / 0.013
        for gap, room in ((L.KEEP_GAP_AIM, 0.003), (0.001, 0.0)):
            self.assertGreater(gap + L.KEEP_SIZE[0] / 2 - width_at_corner / 2, room, gap)
        # the clock constants are the plan's
        tl = {arm: self._timeline(wps) for arm, wps in plan.items()}

        def end_of(arm, label, k=0):
            return [b for a, b, w in tl[arm] if w.label == label][k]

        self.assertAlmostEqual(end_of("left", "straight up", 0), t.T_A_UP)
        self.assertAlmostEqual(end_of("right", "over the keeper", 0), t.T_B_CLEAR)
        self.assertAlmostEqual(end_of("left", "straight up", 1), t.T_K1_UP)
        # the two hands are never over the tray (with 6 cm to spare) at the same time

        def over_tray(p):
            return (L.INNER_X0 - 0.06 <= p[0] <= L.INNER_X1 + 0.06) and abs(p[1]) <= L.INNER_Y + 0.06

        def tray_spans(arm):  # every move toward, or hold at, a point over the tray
            return [(a, b) for a, b, w in tl[arm] if over_tray(w.pos)]

        for a0, a1 in tray_spans("left"):
            for b0, b1 in tray_spans("right"):
                self.assertTrue(a1 <= b0 + 1e-9 or b1 <= a0 + 1e-9, ((a0, a1), (b0, b1)))
        # the release openings: the finger between the magnets 9 mm clear of the other one; the keeper's pads
        # inside the side rims
        self.assertAlmostEqual(L.PAIR_DY - (-2 * t.MAG_RELEASE) / 2 - 0.008 - (-L.PAIR_DY + L.MAG_W / 2), 0.009)
        self.assertLess(-t.KEEP_RELEASE + 0.008, L.INNER_Y - 0.010)
        self.assertGreater(-2 * t.KEEP_RELEASE, L.KEEP_SIZE[1] + 0.012)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("side by side", "along the tray", "red (north)", "blue (south)", "keeper", "each end",
                  "touching the ends of both magnets"):
            self.assertIn(s, ins)


if __name__ == "__main__":
    unittest.main()
