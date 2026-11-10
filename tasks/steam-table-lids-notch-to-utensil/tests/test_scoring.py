"""Offline tests for scoring.py and task.py: synthetic lid states, no simulator.
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
    LIDS,
    UTENSIL_PARTS,
    Tracker,
    axis_err_deg,
    evaluate,
    heading_deg,
    lid_corners,
    lid_pose,
    nearest_pan,
    notch_err_deg,
    right_lid,
    tilt_deg,
    utensil_moved,
    valid,
)

# Everything below is written out again from the task's description (not read from layout.py or scoring.py).
WINDOW = 10  # 0.5 s at 20 Hz
TOP = 0.7475
RIM = TOP + 0.030  # the pans' rims and the deck: 30 mm above the table
PAN_X = 0.330
PANS = {"pan1": 0.1275, "pan2": 0.0425, "pan3": -0.0425, "pan4": -0.1275}
UTENSIL_PANS = ("pan1", "pan3")
KIND = {"lid_notched_left": "notched", "lid_solid_left": "solid", "lid_solid_right": "solid",
        "lid_notched_right": "notched"}
START = {  # (x, y, heading of the long side's plain end, degrees)
    "lid_notched_left": (0.370, 0.270, -90.0),
    "lid_solid_left": (0.250, 0.270, -90.0),
    "lid_solid_right": (0.370, -0.270, 90.0),
    "lid_notched_right": (0.250, -0.270, 90.0),
}
RIGHT = {"lid_notched_left": "pan1", "lid_solid_left": "pan2", "lid_solid_right": "pan4", "lid_notched_right": "pan3"}
# The utensils' parts (box centres): bowl x 290-314 mm, z 22-26; stem x 284-290, z 22-56; handle x 232-290, z 56-64.
PARTS = {"bowl": (0.302, 0.024), "stem": (0.287, 0.039), "handle": (0.261, 0.060)}
UTENSIL_AT = {f"{pre}_{part}": np.array([x, PANS[pan], TOP + z]) for pre, pan in (("ladle", "pan1"), ("spoon", "pan3"))
              for part, (x, z) in PARTS.items()}


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


def lid(x, y, z=RIM, yaw=0.0, pitch=0.0, roll=0.0):
    """A lid's state: its base frame (the middle of the plate's underside) at (x, y, z), turned as given."""
    q = quat(yaw, pitch, roll)
    o = np.array([x, y, z], dtype=float)
    pos = o + _rot(q) @ np.array([0.0, 0.0, 0.0085])
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=pos, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=pos - 0.06, high=pos + 0.06)


def part(name, offset=(0.0, 0.0, 0.0)):
    p = UTENSIL_AT[name] + np.asarray(offset, dtype=float)
    return SimpleNamespace(pos=p, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=p - 0.005, high=p + 0.005)


def on(pan, dx=0.0, dy=0.0, dz=0.0, yaw=0.0, **kw):
    return lid(PAN_X + dx, PANS[pan] + dy, RIM + dz, yaw, **kw)


def state(**lids):
    """All four right lids seated, centred and square (notches to the front), utensils in place; overrides by name."""
    st = {n: on(pan) for n, pan in RIGHT.items()}
    st.update({n: part(n) for n in UTENSIL_AT})
    st.update(lids)
    return st


def start_state():
    st = {n: lid(x, y, TOP, h) for n, (x, y, h) in START.items()}
    st.update({n: part(n) for n in UTENSIL_AT})
    return st


def settled(st, tracker=None):
    """Feed the same state WINDOW + 1 times (0.5 s at rest)."""
    tr = tracker or Tracker(20.0)
    res = None
    for _ in range(WINDOW + 1):
        res = tr.update(st)
    return tr, res


def scored(st):
    return settled(st)[1].score


def propped(pan, yaw=0.0):
    """A lid propped open on a utensil: the front of its plate on the top of the handle (64 mm up, 290 mm out) and its
    back edge on the back rim; about 20 degrees, front up."""
    slope = math.atan2(0.064 - 0.030, 0.383 - 0.290)
    x = 0.383 - 0.053 * math.cos(slope)
    z = TOP + 0.030 + 0.053 * math.sin(slope)
    return lid(x, PANS[pan], z, yaw, pitch=math.degrees(slope))


class Geometry(unittest.TestCase):
    def test_corners_flat_and_turned(self):
        c = lid_corners(on("pan2"))
        np.testing.assert_allclose(sorted(c[:, 0]), [0.277, 0.277, 0.383, 0.383], atol=1e-12)
        np.testing.assert_allclose(sorted(c[:, 1]), [0.010, 0.010, 0.075, 0.075], atol=1e-12)
        np.testing.assert_allclose(c[:, 2], RIM, atol=1e-12)
        c = lid_corners(lid(0.370, 0.270, TOP, -90.0))  # crosswise on the counter
        np.testing.assert_allclose(sorted(c[:, 1]), [0.217, 0.217, 0.323, 0.323], atol=1e-12)
        np.testing.assert_allclose(sorted(c[:, 0]), [0.3375, 0.3375, 0.4025, 0.4025], atol=1e-12)

    def test_heading_axis_and_notch(self):
        self.assertAlmostEqual(heading_deg(on("pan1")), 0.0)
        self.assertAlmostEqual(heading_deg(on("pan1", yaw=180.0)), 180.0)
        self.assertAlmostEqual(heading_deg(lid(0, 0, TOP, -90.0)), -90.0)
        self.assertIsNone(heading_deg(lid(0, 0, TOP, 0.0, pitch=90.0)))
        self.assertAlmostEqual(axis_err_deg(0.0), 0.0)
        self.assertAlmostEqual(axis_err_deg(180.0), 0.0)
        self.assertAlmostEqual(axis_err_deg(-174.0), 6.0)
        self.assertAlmostEqual(axis_err_deg(90.0), 90.0)
        self.assertAlmostEqual(axis_err_deg(None), 90.0)
        self.assertAlmostEqual(notch_err_deg(0.0), 0.0)
        self.assertAlmostEqual(notch_err_deg(-6.0), 6.0)
        self.assertAlmostEqual(notch_err_deg(180.0), 180.0)
        self.assertAlmostEqual(notch_err_deg(-90.0), 90.0)
        self.assertAlmostEqual(notch_err_deg(None), 180.0)

    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(on("pan1")), 0.0)
        self.assertAlmostEqual(tilt_deg(on("pan1", pitch=3.0)), 3.0)
        self.assertAlmostEqual(tilt_deg(on("pan1", roll=180.0)), 180.0)

    def test_nearest_pan(self):
        for pan in PANS:
            self.assertEqual(nearest_pan(on(pan, dx=0.004, dy=-0.003)), (pan, nearest_pan(on(pan, 0.004, -0.003))[1]))
            self.assertAlmostEqual(nearest_pan(on(pan, dx=0.004, dy=-0.003))[1], 0.005)
        self.assertEqual(nearest_pan(lid(PAN_X, 0.000, RIM))[0] in ("pan2", "pan3"), True)  # on the deck between

    def test_lid_pose_fields(self):
        p = lid_pose(on("pan3", dx=0.003), "notched")
        for k in ("kind", "pan", "centre_off_mm", "heading_deg", "axis_err_deg", "tilt_deg", "corner_gap_mm",
                  "flat_on_pan", "notch_err_deg"):
            self.assertIn(k, p)
        self.assertEqual(p["pan"], "pan3")
        self.assertAlmostEqual(p["centre_off_mm"], 3.0)
        self.assertTrue(p["flat_on_pan"])
        self.assertNotIn("notch_err_deg", lid_pose(on("pan2"), "solid"))

    def test_right_lid(self):
        notch_front = lid_pose(on("pan1"), "notched")
        notch_back = lid_pose(on("pan1", yaw=180.0), "notched")
        solid = lid_pose(on("pan1", yaw=180.0), "solid")
        self.assertTrue(right_lid("pan1", notch_front))
        self.assertTrue(right_lid("pan3", notch_front))
        self.assertFalse(right_lid("pan1", notch_back))
        self.assertFalse(right_lid("pan1", solid))
        self.assertTrue(right_lid("pan2", solid))
        self.assertTrue(right_lid("pan4", lid_pose(on("pan4"), "solid")))
        self.assertFalse(right_lid("pan2", notch_front))
        self.assertFalse(right_lid("pan4", notch_back))


class Tiers(unittest.TestCase):
    def test_start_is_zero(self):
        tr, res = settled(start_state())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertEqual(res.details["zero_reason"].count("not_covered"), 4)

    def test_all_right_is_one(self):
        tr, res = settled(state())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["zero_reason"], "")

    def test_solid_lids_either_way_round(self):
        st = state(lid_solid_left=on("pan2", yaw=180.0), lid_solid_right=on("pan4", yaw=-178.0))
        self.assertEqual(scored(st), 1.0)

    def test_either_notched_lid_on_either_utensil_pan(self):
        st = state(lid_notched_left=on("pan3"), lid_notched_right=on("pan1"))
        self.assertEqual(scored(st), 1.0)
        st = state(lid_solid_left=on("pan4"), lid_solid_right=on("pan2"))
        self.assertEqual(scored(st), 1.0)

    def test_three_quarters_a_utensil_pan_left_uncovered(self):
        st = state(lid_notched_right=lid(0.250, -0.270, TOP, 90.0))
        _, res = settled(st)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)  # success is 1.0 only
        self.assertEqual(res.details["zero_reason"], "pan3_not_covered")

    def test_three_quarters_a_plain_pan_left_uncovered(self):
        st = state(lid_solid_left=lid(0.250, 0.270, TOP, -90.0))
        self.assertEqual(scored(st), 0.75)

    def test_half_one_side_swapped(self):
        # left side swapped: the solid lid propped on the ladle, the notched lid on the rice
        st = state(lid_solid_left=propped("pan1"), lid_notched_left=on("pan2"))
        _, res = settled(st)
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.success)
        self.assertIn("pan1_not_covered", res.details["zero_reason"])
        self.assertIn("pan2_notched_lid_on_a_plain_pan", res.details["zero_reason"])

    def test_half_one_arm_alone(self):
        st = state(lid_solid_right=lid(0.370, -0.270, TOP, 90.0), lid_notched_right=lid(0.250, -0.270, TOP, 90.0))
        self.assertEqual(scored(st), 0.5)
        st = state(lid_solid_left=lid(0.250, 0.270, TOP, -90.0), lid_notched_left=lid(0.370, 0.270, TOP, -90.0))
        self.assertEqual(scored(st), 0.5)

    def test_quarter_one_pan(self):
        st = start_state()
        st["lid_solid_right"] = on("pan4")
        self.assertEqual(scored(st), 0.25)
        st = start_state()
        st["lid_notched_right"] = on("pan3")
        self.assertEqual(scored(st), 0.25)

    def test_every_tier_reachable(self):
        seen = set()
        names = list(RIGHT)
        for k in range(16):
            st = start_state()
            for i, n in enumerate(names):
                if k >> i & 1:
                    st[n] = on(RIGHT[n])
            seen.add(scored(st))
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


class Edges(unittest.TestCase):
    def test_centre_tolerance_8_mm(self):
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dx=0.0079))), 1.0)
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dx=0.0081))), 0.75)
        d = 0.0079 / math.sqrt(2)
        self.assertEqual(scored(state(lid_notched_left=on("pan1", dx=d, dy=d))), 1.0)
        self.assertEqual(scored(state(lid_notched_left=on("pan1", dy=0.0082))), 0.75)

    def test_square_within_6_degrees(self):
        self.assertEqual(scored(state(lid_solid_right=on("pan4", yaw=5.9))), 1.0)
        self.assertEqual(scored(state(lid_solid_right=on("pan4", yaw=6.1))), 0.75)
        self.assertEqual(scored(state(lid_notched_right=on("pan3", yaw=-5.9))), 1.0)
        self.assertEqual(scored(state(lid_notched_right=on("pan3", yaw=-6.1))), 0.75)
        self.assertEqual(scored(state(lid_solid_left=on("pan2", yaw=90.0))), 0.75)  # crosswise on the pan

    def test_flat_within_3_degrees(self):
        # turned about its middle, 2.95 degrees about the long axis moves the corners 1.7 mm up and down (inside 2 mm):
        # seated; 3.1 degrees (1.8 mm) is not, by the tilt alone
        self.assertEqual(scored(state(lid_solid_left=on("pan2", roll=2.95))), 1.0)
        self.assertEqual(scored(state(lid_solid_left=on("pan2", roll=3.1))), 0.75)
        # along the 106 mm length the corners bind first: 2.0 degrees moves them 1.85 mm, 2.2 degrees 2.03 mm
        self.assertEqual(scored(state(lid_notched_left=on("pan1", pitch=-2.0))), 1.0)
        self.assertEqual(scored(state(lid_notched_left=on("pan1", pitch=-2.2))), 0.75)
        self.assertEqual(scored(state(lid_notched_left=on("pan1", pitch=3.1))), 0.75)

    def test_corners_within_2_mm_of_the_rim(self):
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dz=0.0019))), 1.0)
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dz=0.0021))), 0.75)
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dz=-0.0021))), 0.75)  # sunk into the pan: not seated
        # resting on the rim at one end and lifted 2.8 mm at the other (1.5 degrees): flat enough, but a corner is
        # 2.8 mm off the rim
        lift = 0.053 * math.sin(math.radians(1.5))
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dz=lift, pitch=1.5))), 0.75)
        lift = 0.053 * math.sin(math.radians(1.0))  # 1.85 mm at the far end: still seated
        self.assertEqual(scored(state(lid_solid_left=on("pan2", dz=lift, pitch=1.0))), 1.0)

    def test_notched_lid_turned_backwards_is_wrong(self):
        # even lying flat (as if the handle were not there), the notch is at the back, away from the handle
        st = state(lid_notched_left=on("pan1", yaw=180.0))
        _, res = settled(st)
        self.assertEqual(res.score, 0.75)
        self.assertIn("pan1_utensil_without_a_notched_lid_to_the_front", res.details["zero_reason"])
        self.assertEqual(scored(state(lid_notched_right=on("pan3", yaw=174.0))), 0.75)

    def test_propped_lids_are_not_seated(self):
        for n, pan in (("lid_solid_left", "pan1"), ("lid_notched_left", "pan1"), ("lid_solid_right", "pan3")):
            p = lid_pose(propped(pan, 180.0 if KIND[n] == "notched" else 0.0), KIND[n])
            self.assertGreater(p["tilt_deg"], 15.0)
            self.assertGreater(p["corner_gap_mm"], 30.0)
            self.assertFalse(p["flat_on_pan"])
        self.assertEqual(scored(state(lid_notched_left=propped("pan1", 180.0))), 0.75)

    def test_uncovered_pan_reason(self):
        st = state(lid_notched_left=lid(0.370, 0.270, TOP, -90.0))
        self.assertEqual(settled(st)[1].details["pan1"]["seated"], [])


class ZeroConditions(unittest.TestCase):
    @staticmethod
    def after_start(st):
        """The trial starts with everything as set (the tracker notes where the utensils are), then ``st`` holds."""
        tr = Tracker(20.0)
        tr.update(state())
        return settled(st, tr)[1]

    def test_a_utensil_moved_more_than_5_mm(self):
        for n in UTENSIL_AT:
            res = self.after_start(state(**{n: part(n, (0.0, 0.0051, 0.0))}))
            self.assertEqual(res.score, 0.0, n)
            self.assertEqual(res.details["zero_reason"], "utensil_moved")
            self.assertFalse(res.success)

    def test_a_utensil_lifted(self):
        self.assertEqual(self.after_start(state(ladle_stem=part("ladle_stem", (0.0, 0.0, 0.006)))).score, 0.0)
        self.assertEqual(self.after_start(state(spoon_handle=part("spoon_handle", (-0.004, 0.0, 0.004)))).score, 0.0)

    def test_a_utensil_nudged_less_than_5_mm(self):
        self.assertEqual(self.after_start(state(ladle_bowl=part("ladle_bowl", (0.0049, 0.0, 0.0)))).score, 1.0)

    def test_moved_before_anything_is_seen_compares_with_the_layout(self):
        # a scorer asked with no history (result() on a fresh tracker, or evaluate() alone) uses the set positions
        st = state(spoon_bowl=part("spoon_bowl", (0.0, -0.006, 0.0)))
        self.assertEqual(evaluate(st, {n: True for n in LIDS})[0], 0.0)
        self.assertEqual(evaluate(state(), {n: True for n in LIDS})[0], 1.0)

    def test_a_constant_offset_in_the_reported_positions_is_not_a_move(self):
        # fixed parts reported 7 mm off their declared centres from the first step (a frame convention) never move
        offset = {n: part(n, (0.0, 0.0, 0.007)) for n in UTENSIL_AT}
        tr = Tracker(20.0)
        tr.update(state(**offset))
        self.assertEqual(settled(state(**offset), tr)[1].score, 1.0)
        # ... and a reset starts the reference over
        tr.update(start_state())
        tr.reset()
        tr.update(state())
        self.assertEqual(settled(state(**offset), tr)[1].score, 0.0)

    def test_utensil_moved_report(self):
        st = state(spoon_stem=part("spoon_stem", (0.003, 0.004, 0.0)))
        self.assertAlmostEqual(utensil_moved(st)["spoon_stem"], 0.005)
        self.assertEqual(set(utensil_moved(state())), set(UTENSIL_AT))
        ref = {"spoon_stem": UTENSIL_AT["spoon_stem"] + np.array([0.003, 0.004, 0.0])}
        self.assertAlmostEqual(utensil_moved(st, ref)["spoon_stem"], 0.0)

    def test_utensil_names(self):
        self.assertEqual(set(UTENSIL_PARTS), set(UTENSIL_AT))

    def test_invalid_states_score_zero(self):
        bad = []
        st = state()
        st["lid_solid_left"] = lid(np.nan, 0.0425, RIM)
        bad.append(st)
        st = state()
        st["lid_notched_right"].quat_xyzw = np.zeros(4)
        bad.append(st)
        st = state()
        del st["lid_solid_right"].origin
        bad.append(st)
        st = state()
        st["ladle_handle"] = SimpleNamespace(pos=np.array([np.inf, 0.0, 0.0]))
        bad.append(st)
        st = state()
        del st["lid_notched_left"]
        bad.append(st)
        for st in bad:
            self.assertFalse(valid(st))
            self.assertEqual(Tracker(20.0).update(st).score, 0.0)
            self.assertEqual(Tracker(20.0).result(st).details["zero_reason"], "invalid_state")


class Exploits(unittest.TestCase):
    def test_solid_lid_pushed_along_to_clear_the_handle(self):
        # the plate's front must be past the stem (290 mm) to lie flat: 13 mm or more back, off centre
        self.assertEqual(scored(state(lid_notched_left=lid(0.250, 0.270, TOP, -90.0),
                                      lid_solid_left=on("pan1", dx=0.014))), 0.25 + 0.25)
        self.assertEqual(scored(state(lid_solid_right=on("pan3", dx=0.015), lid_notched_right=on("pan4"))), 0.5)

    def test_notched_lid_on_a_plain_pan_earns_nothing(self):
        st = state(lid_notched_left=on("pan2"), lid_solid_left=lid(0.250, 0.270, TOP, -90.0))
        self.assertEqual(scored(st), 0.5)  # pan1 uncovered, pan2 has a slot open
        st = state(lid_notched_right=on("pan4", yaw=180.0), lid_solid_right=lid(0.370, -0.270, TOP, 90.0))
        self.assertEqual(scored(st), 0.5)

    def test_a_lid_stacked_on_another_is_not_seated(self):
        # the solid lid on the rice, and the spare notched lid laid on top of it: at least 3 mm higher (17 mm on the
        # lower lid's handle bar), so 3 mm is the closest case
        st = state(lid_notched_right=on("pan2", dz=0.003))
        _, res = settled(st)
        self.assertEqual(res.score, 0.75)
        self.assertEqual(res.details["pan2"]["seated"], ["lid_solid_left"])

    def test_one_lid_cannot_cover_two_pans(self):
        # a lid on the deck between two pans: 42.5 mm from either centre
        st = state(lid_solid_left=lid(PAN_X, 0.0, RIM), lid_notched_right=lid(0.250, -0.270, TOP, 90.0))
        self.assertEqual(scored(st), 0.5)

    def test_lid_held_hovering_or_upside_down(self):
        self.assertEqual(scored(state(lid_solid_right=on("pan4", dz=0.005))), 0.75)
        flipped = on("pan4", dz=0.003, roll=180.0)  # the plate's underside up, lying on its bar
        self.assertEqual(scored(state(lid_solid_right=flipped)), 0.75)

    def test_random_poses_never_beat_the_rule(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            st = state()
            n = rng.choice(list(RIGHT))
            pan = rng.choice(list(PANS))
            dx, dy = rng.uniform(-0.03, 0.03, 2)
            yaw = rng.uniform(-180, 180)
            st[n] = on(pan, dx, dy, yaw=yaw)
            s = scored(st)
            good = (math.hypot(dx, dy) <= 0.008 and pan == RIGHT[n] and
                    (abs(yaw) <= 6.0 if KIND[n] == "notched" else min(abs(yaw), 180 - abs(yaw)) <= 6.0))
            # the other three lids stay right; this one earns its pan only within the rule (on another pan of its
            # kind it covers nothing new, and its own pan is left uncovered)
            self.assertEqual(s, 1.0 if good else 0.75, (n, pan, dx, dy, yaw))


class Stillness(unittest.TestCase):
    def test_not_at_rest_until_half_a_second_seen(self):
        tr = Tracker(20.0)
        for _ in range(WINDOW):
            self.assertEqual(tr.update(state()).score, 0.0)
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_moving_lid_is_not_seated(self):
        tr = Tracker(20.0)
        for k in range(WINDOW + 1):
            tr.update(state(lid_solid_left=on("pan2", dx=0.0003 * k)))  # 0.3 mm per step: 3 mm in 0.5 s
        res = tr.result(state(lid_solid_left=on("pan2", dx=0.0003 * WINDOW)))
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.details["lid_solid_left"]["at_rest"])

    def test_creep_within_2_mm_is_at_rest(self):
        tr = Tracker(20.0)
        for k in range(WINDOW + 1):
            res = tr.update(state(lid_solid_left=on("pan2", dx=0.00015 * k)))  # 1.5 mm in 0.5 s
        self.assertEqual(res.score, 1.0)

    def test_turning_in_place_is_moving(self):
        tr = Tracker(20.0)
        for k in range(WINDOW + 1):
            res = tr.update(state(lid_solid_right=on("pan4", yaw=0.5 * k)))  # 5 degrees in 0.5 s: corners 5 mm
        self.assertEqual(res.score, 0.75)

    def test_result_is_idempotent(self):
        tr, _ = settled(state())
        a, b = tr.result(state()), tr.result(state())
        self.assertEqual(a.score, b.score)
        self.assertEqual(tr.steps, WINDOW + 1)


class Reset(unittest.TestCase):
    def test_teleport_back_to_the_start_resets(self):
        tr, res = settled(state())
        self.assertEqual(res.score, 1.0)
        res = tr.update(start_state())
        self.assertEqual(tr.steps, 1)
        self.assertEqual(res.score, 0.0)
        res = tr.update(state())  # put back on the pans in one step: not yet at rest
        self.assertEqual(res.score, 0.0)

    def test_slow_return_is_not_a_reset(self):
        tr = Tracker(20.0)
        for _ in range(3):
            tr.update(start_state())
        self.assertEqual(tr.steps, 3)

    def test_start_detection(self):
        self.assertTrue(Tracker.at_start(start_state()))
        st = start_state()
        st["lid_notched_left"] = lid(0.370, 0.270, TOP, 90.0)  # turned half round on its mark
        self.assertFalse(Tracker.at_start(st))
        st = start_state()
        st["lid_solid_left"] = lid(0.250, 0.270, TOP + 0.01, -90.0)
        self.assertFalse(Tracker.at_start(st))
        self.assertFalse(Tracker.at_start(state()))

    def test_explicit_reset(self):
        tr, _ = settled(state())
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.update(state()).score, 0.0)

    def test_invalid_step_keeps_history(self):
        tr, _ = settled(state())
        bad = state()
        bad["lid_solid_left"] = lid(np.nan, 0.0, RIM)
        self.assertEqual(tr.update(bad).score, 0.0)
        self.assertEqual(tr.update(state()).score, 1.0)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _box_corners(c, s):
    return [(c[0] + a * s[0] / 2, c[1] + b * s[1] / 2, c[2] + e * s[2] / 2) for a in (-1, 1) for b in (-1, 1)
            for e in (-1, 1)]


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
        cls.by = {o.name: o for o in cls.task.OBJECTS}

    def sim(self, st):
        """Synthetic states turned into what yam_arena and mechkit would report from the simulator."""
        out = dict(st)
        for n in LIDS:
            out[n] = self.mk.state_from_poses(self.by[n], (*st[n].origin, *st[n].quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t, by = self.task, self.by
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted(LIDS))
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        for pan, y in PANS.items():  # walls 3 mm, 30 mm tall, round a 106 x 65 mm outline
            xs = [by[f"{pan}_{w}"].pos[0] + s * by[f"{pan}_{w}"].size[0] / 2 for w in ("front", "back", "left",
                                                                                        "right") for s in (-1, 1)]
            ys = [by[f"{pan}_{w}"].pos[1] + s * by[f"{pan}_{w}"].size[1] / 2 for w in ("front", "back", "left",
                                                                                        "right") for s in (-1, 1)]
            self.assertAlmostEqual(min(xs), 0.277)
            self.assertAlmostEqual(max(xs), 0.383)
            self.assertAlmostEqual(min(ys), y - 0.0325)
            self.assertAlmostEqual(max(ys), y + 0.0325)
            for w in ("front", "back", "left", "right"):
                self.assertAlmostEqual(by[f"{pan}_{w}"].pos[2] + by[f"{pan}_{w}"].size[2] / 2, RIM)
            food = by[f"{pan}_food"]
            self.assertAlmostEqual(food.pos[2] + food.size[2] / 2, TOP + 0.022)
        for n, at in UTENSIL_AT.items():
            np.testing.assert_allclose(by[n].pos, at, atol=1e-12)
            self.assertIsNone(by[n].mass)
        for n in ("deck_front", "deck_back", "deck_left", "deck_1_2", "deck_2_3", "deck_3_4", "deck_right"):
            self.assertAlmostEqual(by[n].pos[2] + by[n].size[2] / 2, RIM)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, o.size, 0.0)
            half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)

    def test_lid_geometry(self):
        notch = ([(0.0125, 0.0, 0.0015), (0.081, 0.065, 0.003)], [(-0.0405, 0.02275, 0.0015), (0.025, 0.0195, 0.003)],
                 [(-0.0405, -0.02275, 0.0015), (0.025, 0.0195, 0.003)], [(0.0, 0.0, 0.010), (0.050, 0.020, 0.014)])
        solid = ([(0.0, 0.0, 0.0015), (0.106, 0.065, 0.003)], [(0.0, 0.0, 0.010), (0.050, 0.020, 0.014)])
        for n in LIDS:
            spec = self.by[n]
            want = notch if KIND[n] == "notched" else solid
            got = sorted((tuple(np.round(s.center, 6)), tuple(np.round(s.size, 6))) for s in spec.shapes)
            self.assertEqual(got, sorted((tuple(c), tuple(s)) for c, s in want))
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-0.053, -0.0325, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [0.053, 0.0325, 0.017], atol=1e-12)
            rest = spec.rest_state()
            x, y, h = START[n]
            np.testing.assert_allclose(rest.origin, [x, y, TOP], atol=1e-12)
            self.assertAlmostEqual(heading_deg(rest), h)
            self.assertAlmostEqual(spec.mass, 0.046 if KIND[n] == "notched" else 0.048)
        rest = {n: self.by[n].rest_state() for n in LIDS}
        rest.update({n: part(n) for n in UTENSIL_AT})
        self.assertTrue(Tracker.at_start(rest))
        self.assertEqual(scored(rest), 0.0)
        sim = self.sim(state())
        self.assertEqual(scored(sim), 1.0)

    def test_notch_clears_the_utensil_and_a_plain_end_does_not(self):
        """At the demo's target (centred, notch to the front) every part of a utensil above the rim lies over the
        notch: 8 mm either side across, 12 mm to the notch's end along. Turned round, or a solid lid, the plate lies
        over the stem and handle, which stand 26-34 mm above the rim: it is propped."""
        by = self.by
        for pre, pan in (("ladle", "pan1"), ("spoon", "pan3")):
            y = PANS[pan]
            notch = (0.277, 0.277 + 0.025, y - 0.013, y + 0.013)  # x0, x1, y0, y1 of a centred lid's notch
            for p in ("stem", "handle"):
                b = by[f"{pre}_{p}"]
                x0, x1 = b.pos[0] - b.size[0] / 2, b.pos[0] + b.size[0] / 2
                y0, y1 = b.pos[1] - b.size[1] / 2, b.pos[1] + b.size[1] / 2
                self.assertGreater(b.pos[2] + b.size[2] / 2, RIM + 0.025)
                self.assertAlmostEqual(y0 - notch[2], 0.008)
                self.assertAlmostEqual(notch[3] - y1, 0.008)
                self.assertGreaterEqual(notch[1] - x1, 0.012 - 1e-12)
                inside = min(x1, 0.383) - max(x0, 0.277)
                self.assertGreater(inside, 0.0)  # part of it is within the lid's outline: a plain end lands on it
            bowl = by[f"{pre}_bowl"]
            self.assertLess(bowl.pos[2] + bowl.size[2] / 2, RIM)  # under the lid

    def test_offline_state_path(self):
        spec = self.by["lid_notched_right"]
        box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(90.0), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [0.250, -0.270, TOP], atol=1e-12)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(start_state())
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

    def test_hooks_without_utensil_states(self):
        """The hooks read only what they are given: lids alone still score (fixed utensils never move)."""
        self.mk.offline_reset()
        done = {n: v for n, v in self.sim(state()).items() if n in LIDS}
        for _ in range(WINDOW + 1):
            self.task.success(done)
        self.assertEqual(self.task.score(done), 1.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(state()))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "zero_reason", "moved_mm", "utensil_moved_mm",
                          "pan1", "pan4", "lid_notched_left", "lid_solid_right", "lid_notched_left_origin",
                          "lid_solid_right_quat", "lid_notched_right_source"):
                    self.assertIn(k, row)
                for k in ("kind", "pan", "centre_off_mm", "heading_deg", "axis_err_deg", "tilt_deg", "corner_gap_mm",
                          "flat_on_pan", "at_rest", "seated", "notch_err_deg"):
                    self.assertIn(k, row["lid_notched_left"])
                for k in ("seated", "right", "credit"):
                    self.assertIn(k, row["pan3"])
                self.assertEqual(row["lid_notched_right"]["pan"], "pan3")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(state())))  # no crash
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
                if prev.finger != w.finger:  # every close and open takes 1.5 s
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                if prev.pos[2] > w.pos[2] and min(abs(w.pos[2] - t.PICK_Z), abs(w.pos[2] - t.PLACE_Z)) < 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))  # slow final descents
                self.assertLessEqual(abs(w.yaw - prev.yaw), math.pi / 2 + 1e-9, (arm, w.label))
            for w in wps:
                if w.label == "grip the bar":
                    self.assertAlmostEqual(-2 * w.finger, 0.012)  # 8 mm narrower than the 20 mm bar
        # each lid: gripped on its mark with the wrist along its long side, set on its pan's centre with the wrist at
        # yaw 0, so its long side ends along the pan, and a notched lid's notch at the front
        for arm, lids in (("left", ("lid_notched_left", "lid_solid_left")),
                          ("right", ("lid_solid_right", "lid_notched_right"))):
            wps = plan[arm]
            grips = [w for w in wps if w.label == "grip the bar"]
            places = [w for w in wps if w.label == "slowly set the lid on the rim"]
            self.assertEqual(len(grips), 2)
            self.assertEqual(len(places), 2)
            for n, g, p in zip(lids, grips, places):
                x, y, h = START[n]
                np.testing.assert_allclose(g.pos[:2], (x, y), atol=1e-12)
                self.assertAlmostEqual(g.pos[2], TOP + 0.021)
                self.assertAlmostEqual(math.degrees(g.yaw), h)
                heading = h + math.degrees(p.yaw - g.yaw)
                self.assertAlmostEqual(heading, 0.0)
                np.testing.assert_allclose(p.pos[:2], (PAN_X, PANS[RIGHT[n]]), atol=1e-12)
                self.assertAlmostEqual(p.pos[2] - 0.021, RIM + 0.0005)  # the plate let go 0.5 mm above the rim
                self.assertEqual(scored(state(**{n: on(RIGHT[n], yaw=heading)})), 1.0)
        # the two grippers are never both over the middle pans: the right arm waits until the left has left pan 2
        t_left = sum(w.duration for w in plan["left"][:[i for i, w in enumerate(plan["left"])
                                                          if w.label == "straight up"][1] + 1])
        t_right = sum(w.duration for w in plan["right"][:[i for i, w in enumerate(plan["right"])
                                                            if w.label.startswith("wait")][0] + 1])
        self.assertGreaterEqual(t_right, t_left - 1e-9)

    def test_the_wrong_turn_is_as_reachable_as_the_right_one(self):
        """Gripping a notched lid with the wrist turned half round (the same grip on the symmetric bar) and setting it
        down at yaw 0 puts its notch at the back: the arm reaches both, so the turn's direction is a real choice."""
        import yamkin as K

        for arm, n, pan in (("left", "lid_notched_left", "pan1"), ("right", "lid_notched_right", "pan3")):
            x, y, h = START[n]
            base = K.BASES["station"][arm]
            for grip_yaw in (h, h + (180.0 if h < 0 else -180.0)):
                self.assertTrue(K.reachable((x, y, TOP + 0.021), math.radians(grip_yaw), base)[0], (arm, grip_yaw))
            self.assertTrue(K.reachable((PAN_X, PANS[pan], RIM + 0.0215), 0.0, base)[0])
            wrong = h + (0.0 - (h + (180.0 if h < 0 else -180.0)))  # heading after the half-round grip
            self.assertAlmostEqual(abs(math.remainder(wrong, 360.0)), 180.0)
            self.assertEqual(scored(state(**{n: on(pan, yaw=wrong)})), 0.75)

    def test_one_arm_reaches_only_its_own_counter(self):
        """Straight down, neither arm reaches anything at |y| 20 cm or more on the other side: the other counter's lids
        (their nearest edge 21.7 cm out) are out of reach, so one arm has two lids for four pans."""
        import yamkin as K

        for arm, sign in (("left", -1.0), ("right", 1.0)):
            base = K.BASES["station"][arm]
            for x in (0.25, 0.30, 0.37, 0.40):
                for h in (0.005, 0.021, 0.08):
                    for yaw in (-90.0, 0.0, 90.0):
                        self.assertFalse(K.reachable((x, sign * 0.217, TOP + h), math.radians(yaw), base)[0],
                                         (arm, x, h, yaw))

    def test_demo_switches(self):
        old = {k: os.environ.get(k) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(set(self.task.demo()), {"right"})
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(set(self.task.demo()), {"left"})
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_instruction_matches_task_json(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
