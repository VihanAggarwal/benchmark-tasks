"""Offline tests for scoring.py and task.py: synthetic bar states, no simulator.
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
from scoring import (  # noqa: E402
    Tracker,
    channel,
    ends,
    heading_err_deg,
    pose_tier,
    stickout,
    tilt_deg,
    tip_toward_chuck,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FLOOR_W = L.TOP + L.FLOOR_Z


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll): the bar turned yaw about the vertical after pitching its
    tip (pitch > 0 tips the tip down) and rolling about its own axis."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def raw(origin, q, vel=(0.0, 0.0, 0.0)):
    """A bar state as mechkit reports it (only what the scorer reads; pos = origin is fine for it)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.08, high=o + 0.08)


def bar(s=0.055, off=0.0, under=0.0, yaw=90.0, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0)):
    """The bar along +y (yaw 90: tip toward the chuck) with stick-out s, its centre line ``off`` m off the channel's
    centre in x and its underside ``under`` m above the channel floor (the origin is the middle of its bottom face).
    Other yaws/pitches/rolls turn it about its tip-face centre's place for the given s."""
    r = _rot(quat(yaw, pitch, roll))
    tip_w = np.array([L.AXIS_X + off, L.FACE_Y + s, FLOOR_W + under + L.BAR_W / 2])
    origin = tip_w - r @ np.array(L.TIP_LOCAL)
    return raw(origin, quat(yaw, pitch, roll), vel)


def bar_backwards(back_out=0.055):
    """Laid in the channel the wrong way round: tip toward the arms' side (-y), the back end ``back_out`` past the
    block's chuck-side face."""
    r = _rot(quat(-90.0))
    back_w = np.array([L.AXIS_X, L.FACE_Y + back_out, FLOOR_W + L.BAR_W / 2])
    return raw(back_w - r @ np.array(L.BACK_LOCAL), quat(-90.0))


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


START = raw((L.BAR_START[0], L.BAR_START[1], L.TOP + L.BAR_START[2]), quat(0.0))
SEATED = bar(0.055)


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(st, n=WINDOW + 1):
    """A tracker that has seen ``st`` at rest for ``n`` steps; returns (tracker, last result)."""
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n)


def scored(st):
    return settled(st)[1].score


def glide(a, b, n):
    """n states moving the origin from state a's to state b's, b's orientation (the simulator moves things
    gradually)."""
    oa, ob = np.asarray(a.origin, float), np.asarray(b.origin, float)
    vel = (ob - oa) * HZ / n
    return [raw(oa + (ob - oa) * (i / n), b.quat_xyzw, vel) for i in range(1, n + 1)]


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(valid(START))
        self.assertTrue(Tracker.at_start(START))
        tip, back, _ = ends(START)
        np.testing.assert_allclose(tip, [L.MAT_X + 0.075, L.MAT_Y, L.TOP + L.MAT_TOP + 0.008], atol=1e-12)
        np.testing.assert_allclose(back, [L.MAT_X - 0.075, L.MAT_Y, L.TOP + L.MAT_TOP + 0.008], atol=1e-12)
        self.assertEqual(pose_tier(START)[0], 0.0)
        self.assertAlmostEqual(heading_err_deg(START), 90.0)
        self.assertFalse(Tracker.at_start(bar(0.055)))

    def test_stickout_and_ends(self):
        for s in (-0.02, 0.0, 0.045, 0.055, 0.064, 0.072):
            b = bar(s)
            self.assertAlmostEqual(stickout(b), s, places=12)
            tip, back, _ = ends(b)
            self.assertAlmostEqual(tip[1] - back[1], L.BAR_LEN, places=12)
            self.assertTrue(tip_toward_chuck(b))
        self.assertFalse(tip_toward_chuck(bar_backwards()))

    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(SEATED), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(bar(pitch=3.0)), 3.0, places=6)
        self.assertAlmostEqual(tilt_deg(bar(roll=4.0)), 4.0, places=6)
        self.assertAlmostEqual(tilt_deg(bar(roll=90.0)), 90.0, places=6)  # on its side
        self.assertAlmostEqual(tilt_deg(bar(roll=180.0)), 180.0, places=6)  # insert down
        self.assertAlmostEqual(tilt_deg(bar(pitch=-90.0)), 90.0, places=6)  # standing on its back end

    def test_heading(self):
        self.assertAlmostEqual(heading_err_deg(SEATED), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar_backwards()), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar(yaw=84.0)), 6.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar(yaw=0.0)), 90.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar(pitch=-90.0)), 90.0, places=6)

    def test_channel_measures(self):
        ch = channel(SEATED)
        self.assertAlmostEqual(ch["over_block_mm"], 80.0, places=6)  # back end 15 mm behind the block
        np.testing.assert_allclose(ch["centre_off_mm"], [0, 0], atol=1e-9)
        np.testing.assert_allclose(ch["underside_mm"], [0, 0], atol=1e-9)
        self.assertAlmostEqual(channel(bar(0.072))["over_block_mm"], 78.0, places=6)  # back end inside the block
        self.assertAlmostEqual(channel(bar(-0.020))["over_block_mm"], 60.0, places=6)  # tip inside the block
        ch = channel(bar(0.055, off=0.004, under=0.002))
        np.testing.assert_allclose(ch["centre_off_mm"], [4, 4], atol=1e-9)
        np.testing.assert_allclose(ch["underside_mm"], [2, 2], atol=1e-9)
        ch = channel(bar(0.055, yaw=95.0))  # skewed 5 degrees about the tip: the offset grows toward the back
        self.assertLess(ch["centre_off_mm"][1], ch["centre_off_mm"][0])
        self.assertEqual(channel(bar(yaw=0.0))["over_block_mm"], 0.0)  # across y entirely: not measured

    def test_play_in_the_channel_is_5_mm(self):
        self.assertAlmostEqual(L.CHANNEL_W, 0.026, places=12)
        self.assertAlmostEqual(L.CENTRE_TOL, 0.006, places=12)
        # a bar against either wall is 5 mm off centre; one on a wall's top is at least 13 mm off
        self.assertAlmostEqual((L.CHANNEL_W - L.BAR_W) / 2, 0.005, places=12)
        self.assertGreater(L.CHANNEL_W / 2, L.CENTRE_TOL + 0.005)
        # the largest skew a bar can have lying between the walls over the block's 80 mm is under 10 degrees
        theta = max(t for t in np.radians(np.arange(0.0, 20.0, 0.01))
                    if L.BAR_W / math.cos(t) + (L.FACE_Y - L.BLOCK_Y0) * math.tan(t) <= L.CHANNEL_W)
        self.assertLess(math.degrees(theta), L.HEADING_MAX_DEG)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START), 0.0)

    def test_set_in_the_band_scores_one(self):
        tr, r = settled(SEATED)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["at_rest"])
        self.assertEqual(r.details["zero_reason"], "")

    def test_band_edges(self):
        for s, want in ((0.045, 1.0), (0.0449, 0.5), (0.064, 1.0), (0.0641, 0.5), (0.038, 0.5), (0.0379, 0.25),
                        (0.072, 0.5), (0.0721, 0.0), (0.050, 1.0), (0.060, 1.0), (0.041, 0.5), (0.068, 0.5),
                        (0.030, 0.25), (0.0, 0.25), (-0.004, 0.25)):
            with self.subTest(s=s):
                self.assertEqual(scored(bar(s)), want)

    def test_every_tier_is_reachable(self):
        got = {scored(bar(s)) for s in np.arange(-0.004, 0.080, 0.0005)}
        self.assertEqual(got, {0.0, 0.25, 0.5, 1.0})

    def test_only_one_scores_success(self):
        for s in (0.030, 0.040, 0.068):
            self.assertFalse(settled(bar(s))[1].success)

    def test_anywhere_across_the_channel_counts(self):
        for off in (-0.005, -0.003, 0.0, 0.003, 0.005, 0.0059, -0.0059):
            self.assertEqual(scored(bar(0.055, off=off)), 1.0)
        for off in (-0.0065, 0.0065, 0.013, -0.019):
            self.assertEqual(scored(bar(0.055, off=off)), 0.0)

    def test_skew_inside_the_channel_counts(self):
        # skewed about the middle of the block (as far as the walls allow, about 7 degrees): the centre line stays
        # within 4.2 mm of the channel's centre at the block's ends
        for yaw in (84.0, 87.0, 93.0, 96.0):
            r = _rot(quat(yaw))
            a = r[:, 0]
            mid = np.array([L.AXIS_X, (L.BLOCK_Y0 + L.FACE_Y) / 2, FLOOR_W + L.BAR_W / 2])
            tip = mid + (L.FACE_Y + 0.055 - mid[1]) / a[1] * a
            b = raw(tip - r @ np.array(L.TIP_LOCAL), quat(yaw))
            self.assertAlmostEqual(stickout(b), 0.055, places=12)
            self.assertLess(max(abs(v) for v in channel(b)["centre_off_mm"]), 4.3)
            self.assertEqual(scored(b), 1.0, msg=f"yaw {yaw}: {pose_tier(b)}")

    def test_underside_within_3_mm(self):
        self.assertEqual(scored(bar(0.055, under=0.0025)), 1.0)
        self.assertEqual(scored(bar(0.055, under=-0.001)), 1.0)  # solver penetration
        self.assertEqual(scored(bar(0.055, under=0.0035)), 0.0)

    def test_small_tilt_counts(self):
        self.assertEqual(scored(bar(0.055, roll=4.5)), 1.0)
        self.assertEqual(scored(bar(0.055, roll=-4.5)), 1.0)

    def test_score_is_idempotent(self):
        tr, _ = settled(SEATED)
        n, h = tr.steps, len(tr.history)
        for _ in range(3):
            self.assertEqual(tr.result(SEATED).score, 1.0)
        self.assertEqual((tr.steps, len(tr.history)), (n, h))


class ZeroConditions(unittest.TestCase):
    def assertZero(self, st, reason):
        tr, r = settled(st)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], reason)

    def test_tip_away_from_the_chuck(self):
        for back_out in (0.045, 0.055, 0.064):
            self.assertZero(bar_backwards(back_out), "tip_away_from_chuck")

    def test_tilted(self):
        self.assertZero(bar(0.055, roll=6.0), "tilted_or_on_its_side")
        self.assertZero(bar(0.055, pitch=6.0), "tilted_or_on_its_side")

    def test_on_its_side_or_insert_down(self):
        self.assertZero(bar(0.055, roll=90.0), "tilted_or_on_its_side")
        self.assertZero(bar(0.055, roll=-90.0), "tilted_or_on_its_side")
        self.assertZero(bar(0.055, roll=180.0), "tilted_or_on_its_side")

    def test_standing_on_an_end(self):
        self.assertZero(bar(0.055, pitch=-90.0), "tilted_or_on_its_side")
        self.assertZero(bar(0.055, pitch=90.0), "tilted_or_on_its_side")

    def test_not_along_the_channel(self):
        self.assertZero(bar(0.055, yaw=0.0), "not_along_the_channel")
        self.assertZero(bar(0.055, yaw=75.0), "not_along_the_channel")

    def test_on_a_wall(self):
        self.assertZero(bar(0.055, off=0.019, under=L.WALL_H), "out_of_the_channel")
        self.assertZero(bar(0.055, off=-0.019, under=L.WALL_H), "out_of_the_channel")

    def test_held_above_the_channel(self):
        self.assertZero(bar(0.055, under=0.010), "not_on_the_channel_floor")
        self.assertZero(bar(0.055, under=0.050), "not_on_the_channel_floor")

    def test_tip_at_the_part(self):
        self.assertZero(bar(0.0725), "tip_at_the_part")
        self.assertZero(bar(0.0745), "tip_at_the_part")  # on the bore mark
        self.assertZero(bar(0.075), "tip_at_the_part")  # on the face

    def test_not_over_the_block(self):
        # lying along y on the table in front of the block, between it and the part
        b = bar(0.230, under=-L.FLOOR_Z)  # back end 80 mm past the face
        self.assertZero(b, "not_over_the_block")
        # only its back 15 mm over the block's front edge
        self.assertZero(bar(0.135), "not_over_the_block")

    def test_on_the_table_or_the_mat(self):
        self.assertZero(START, "not_along_the_channel")
        on_table = raw((0.30, -0.10, L.TOP), quat(90.0))
        self.assertEqual(scored(on_table), 0.0)


class Stillness(unittest.TestCase):
    def test_moving_in_the_band_scores_zero_until_at_rest(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(bar(0.040), bar(0.055), 5))  # slid 15 mm along the channel into the band, 3 mm a step
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "still_moving")
        self.assertEqual(r.details["tier"], 1.0)
        for i in range(WINDOW - 1):
            self.assertEqual(tr.update(SEATED).score, 0.0)
        r = tr.update(SEATED)  # 0.5 s at rest
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [SEATED] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["moved_mm"])
        self.assertEqual(tr.update(SEATED).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        states = [bar(0.055 + (0.0019 if i % 2 else 0.0)) for i in range(WINDOW + 5)]
        self.assertEqual(feed(tr, states).score, 1.0)
        tr = Tracker(HZ)
        states = [bar(0.055 + (0.0021 if i % 2 else 0.0)) for i in range(WINDOW + 5)]
        self.assertEqual(feed(tr, states).score, 0.0)

    def test_a_turn_about_the_middle_counts_as_moving(self):
        tr = Tracker(HZ)
        states = [bar(0.055, roll=(3.0 if i % 2 else 0.0)) for i in range(WINDOW + 5)]  # the ends do not move
        self.assertEqual(feed(tr, states).score, 1.0)
        tr = Tracker(HZ)
        states = [bar(0.055, yaw=(92.0 if i % 2 else 90.0)) for i in range(WINDOW + 5)]  # the back end swings 5 mm
        self.assertEqual(feed(tr, states).score, 0.0)

    def test_partial_tiers_need_rest_too(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(bar(0.020), bar(0.030), 15))
        self.assertEqual((r.score, r.details["tier"]), (0.0, 0.25))


class Exploits(unittest.TestCase):
    def test_backwards_bar_with_the_right_overhang(self):
        # the band read off the wrong end: the bar cannot cut
        self.assertEqual(scored(bar_backwards(0.055)), 0.0)

    def test_laid_across_the_block_on_the_walls(self):
        b = raw((L.AXIS_X, (L.BLOCK_Y0 + L.FACE_Y) / 2, L.TOP + L.WALL_TOP), quat(0.0))
        self.assertEqual(scored(b), 0.0)

    def test_on_its_side_in_the_band(self):
        # a square bar fits the channel on any face; only insert up can cut
        self.assertEqual(scored(bar(0.055, roll=90.0)), 0.0)

    def test_held_in_the_air_over_the_band(self):
        self.assertEqual(scored(bar(0.055, under=0.020)), 0.0)

    def test_pushed_against_the_part(self):
        self.assertEqual(scored(bar(0.0745)), 0.0)
        self.assertEqual(scored(bar(0.080)), 0.0)

    def test_propped_on_a_wall_and_the_floor(self):
        # one edge on the near wall's top, the other on the floor: rolled about 50 degrees
        self.assertEqual(scored(bar(0.055, off=-0.006, under=0.004, roll=-50.0)), 0.0)

    def test_tip_dipped_on_the_block_edge(self):
        # resting across the front edge, tip down toward the table: pitched
        self.assertEqual(scored(bar(0.060, pitch=8.0)), 0.0)

    def test_teleport_shaped_shortcut_is_still_scored_by_pose(self):
        # whatever the path, the end state is what counts: a bar dropped in from above and then at rest scores
        tr = Tracker(HZ)
        feed(tr, glide(bar(0.055, under=0.040), bar(0.055), 4))
        r = feed(tr, [SEATED] * (WINDOW + 1))
        self.assertEqual(r.score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SEATED)
        n, hist = tr.steps, list(tr.history)
        bad = [
            raw((math.nan, 0.0, 0.8), quat(90.0)),
            raw((0.25, math.inf, 0.8), quat(90.0)),
            raw((0.25, 0.0, 0.8), (math.nan, 0, 0, 1)),
            raw((0.25, 0.0, 0.8), (0.0, 0.0, 0.0, 0.0)),
            SimpleNamespace(quat_xyzw=quat(90.0), pos=np.zeros(3), lin_vel=np.zeros(3)),  # no origin
            SimpleNamespace(origin=np.zeros(2), quat_xyzw=quat(90.0), pos=np.zeros(3)),  # malformed
            None,
        ]
        for b in bad:
            with self.subTest(b=b):
                self.assertFalse(valid(b))
                r = tr.update(b)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + len(bad))
        self.assertEqual(len(tr.history), len(hist))
        for (a, b), (c, d) in zip(tr.history, hist):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
        self.assertEqual(tr.update(SEATED).score, 1.0)  # and it carries on

    def test_unnormalised_quaternion(self):
        b = bar(0.055)
        b2 = raw(b.origin, 3.0 * np.asarray(b.quat_xyzw))
        self.assertEqual(scored(b2), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker(HZ)
        for _ in range(500):
            q = rng.normal(size=4)
            o = rng.uniform([-0.2, -0.5, 0.6], [0.8, 0.5, 1.2])
            r = tr.update(raw(o, q))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(SEATED)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history), 1)
        self.assertEqual(r.score, 0.0)

    def test_reset_after_a_full_trial_then_a_new_trial(self):
        tr, _ = settled(SEATED)
        tr.update(START)
        r = feed(tr, [SEATED] * WINDOW)  # 10 steps of the new trial: not yet 0.5 s seen
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.update(SEATED).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SEATED)
        jit = raw((L.BAR_START[0] + 0.03, L.BAR_START[1] - 0.03, L.TOP + L.BAR_START[2]), quat(0.0))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(SEATED)
        feed(tr, glide(SEATED, START, 40))
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(SEATED)
        tr.update(bar(0.020))
        self.assertEqual(tr.steps, WINDOW + 2)
        self.assertEqual(tr.update(bar(0.020)).score, 0.0)  # the jump is in the window: moving

    def test_explicit_reset(self):
        tr, _ = settled(SEATED)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history)), (0, 0))
        self.assertIsNone(tr.prev_origin)


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

    def sim(self, st):
        """A synthetic state turned into the AssemblyState mechkit would report from the simulator."""
        return {"boring_bar": self.mk.state_from_poses(self.task.BAR, (*st.origin, *st.quat_xyzw), source="sim")}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], ["boring_bar"])
        self.assertAlmostEqual(self.task.BAR.mass, 0.30)
        self.assertLessEqual(self.task.BAR.mass, 0.30)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)
        # the channel: 26 mm between the walls, on the spindle axis
        self.assertAlmostEqual(self.task.WALL_FAR.pos[0] - self.task.WALL_NEAR.pos[0], L.CHANNEL_W + L.WALL_T)
        self.assertAlmostEqual((self.task.WALL_FAR.pos[0] + self.task.WALL_NEAR.pos[0]) / 2, L.AXIS_X)
        # the workpiece face is 75 mm past the block's face, the bore mark on it, centred on the axis
        self.assertAlmostEqual(self.task.WORKPIECE.pos[1] - self.task.WORKPIECE.size[1] / 2, L.FACE_Y + 0.075)
        self.assertAlmostEqual(self.task.BORE_MARK.pos[1] + self.task.BORE_MARK.size[1] / 2, L.WORK_FACE_Y)
        self.assertAlmostEqual(self.task.WORKPIECE.pos[2], L.TOP + L.FLOOR_Z + L.BAR_W / 2)

    def test_scorer_geometry_matches_mechkit(self):
        lo, hi = self.task.BAR.local_bounds()
        np.testing.assert_allclose(lo, [-0.075, -0.008, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.075, 0.008, 0.019], atol=1e-12)
        self.assertAlmostEqual(L.TIP_LOCAL[0], hi[0])
        self.assertAlmostEqual(L.BACK_LOCAL[0], lo[0])
        # the insert sits at the tip end: the scorer's tip is the end the reviewer sees as the tip
        ins = [s for s in self.task.BAR.shapes if tuple(s.size) == L.INSERT_SIZE][0]
        self.assertGreater(ins.center[0], 0.06)
        # the spawn pose is the start the scorer expects
        rest = self.task.BAR.rest_state()
        self.assertTrue(Tracker.at_start(rest))
        np.testing.assert_allclose(ends(rest)[0][0], L.MAT_X + 0.075, atol=1e-12)
        # a mechkit state of a seated bar reads the same as the synthetic one
        st = self.sim(SEATED)["boring_bar"]
        self.assertAlmostEqual(stickout(st), 0.055, places=9)
        np.testing.assert_allclose(st.low[2], FLOOR_W, atol=1e-9)

    def test_the_bar_rests_in_every_scored_position(self):
        com = self.task.BAR.center_of_mass()
        self.assertAlmostEqual(com[0], 0.0002, places=4)  # the insert's volume: 0.2 mm toward the tip
        balance = L.TIP_LOCAL[0] - com[0]  # stick-out at which the centre of mass is over the block's face edge
        self.assertAlmostEqual(balance, L.BALANCE_STICKOUT, places=4)
        self.assertGreater(balance, L.LONG_HI + 0.002)  # every pose up to 72 mm out rests on the block
        # tipping backward off the block's back edge happens only for stick-outs below -5 mm
        self.assertLess(L.BLOCK_Y0 - L.FACE_Y + balance, -0.004)

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.BAR.offline_box()[0]), quat_xyzw=quat(0.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.BAR.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.BAR_START[0], L.BAR_START[1], L.TOP + L.BAR_START[2]], atol=1e-12)
        self.assertTrue(Tracker.at_start(st))

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        seated = self.sim(SEATED)
        for _ in range(WINDOW):
            self.task.success(seated)
        self.assertTrue(self.task.success(seated))
        self.assertEqual(self.task.score(seated), 1.0)
        self.assertEqual(self.task.score(seated), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(self.task.success(seated))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(SEATED))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "s_mm", "tilt_deg", "heading_err_deg", "tip_toward_chuck",
                          "centre_off_mm", "underside_mm", "over_block_mm", "at_rest", "moved_mm", "zero_reason",
                          "tier", "bar_origin", "tip", "back", "source"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["s_mm"], 55.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(SEATED)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"right"})
        wps = plan["right"]
        # every close and open takes 1.5 s, and so does every final descent to a grip or place height (the last
        # 30 mm)
        for prev, w in zip(wps, wps[1:]):
            if prev.finger != w.finger:
                self.assertGreaterEqual(w.duration, 1.5)
            if prev.pos[2] > w.pos[2] and min(abs(w.pos[2] - t.PICK[2]), abs(w.pos[2] - t.PLACE[2])) < 1e-9:
                self.assertGreaterEqual(w.duration, 1.5)
                self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
        grip = [w for w in wps if w.label == "grip the bar"][0]
        place = [w for w in wps if w.label == "slowly lay the bar in the channel"][0]
        self.assertEqual(grip.yaw, 0.0)
        self.assertAlmostEqual(place.yaw, math.pi / 2)  # a quarter turn left: the tip swings from +x to +y
        # gripped 35 mm back from the tip; set with the tip 55 mm out, the grip 20 mm past the block's face
        self.assertAlmostEqual(grip.pos[0], L.MAT_X + L.BAR_LEN / 2 - t.GRIP_FROM_TIP)
        self.assertAlmostEqual(place.pos[1] + t.GRIP_FROM_TIP - L.FACE_Y, t.S_CMD)
        self.assertTrue(L.BAND_LO + 0.009 <= t.S_CMD <= L.BAND_HI - 0.009)
        self.assertGreaterEqual(place.pos[1] - 0.010 - L.FACE_Y, 0.010 - 1e-9)  # 20 mm-wide pads 10 mm clear of it
        self.assertAlmostEqual(place.pos[0], L.AXIS_X)
        # the pads (20 mm long) hold the shank behind the black head, and the centre of mass is at most 40 mm
        # behind the grip: under half of the pads' estimated 0.28 N m of friction
        self.assertGreaterEqual(t.GRIP_FROM_TIP - 0.010, L.HEAD_LEN)
        lever = L.TIP_LOCAL[0] - t.BAR.center_of_mass()[0] - t.GRIP_FROM_TIP
        self.assertLessEqual(lever, 0.0405)
        self.assertLess(L.BAR_MASS * 9.81 * lever, 0.28 / 2)
        # pad tips (13 mm below the TCP) 4 mm above the bar's underside at the pick and at the place
        self.assertAlmostEqual(grip.pos[2] - 0.013 - (L.TOP + L.MAT_TOP), 0.004)
        self.assertAlmostEqual(place.pos[2] - 0.013 - (L.TOP + L.FLOOR_Z + t.DROP), 0.004)
        # carried with the bar's underside over the walls' tops
        hover = [w for w in wps if w.label.startswith("carry it over")][0]
        self.assertGreater(hover.pos[2] - t.GRIP_UP - (L.TOP + L.WALL_TOP), 0.025)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", set()),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("tip toward the chuck", "insert up", "45 mm", "64 mm", "16 mm", "40 mm", "off the part"):
            self.assertIn(s, ins)


if __name__ == "__main__":
    unittest.main()
