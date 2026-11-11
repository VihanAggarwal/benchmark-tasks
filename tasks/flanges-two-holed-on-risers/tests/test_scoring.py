"""Offline tests for scoring.py and task.py: synthetic flange states and gripper snapshots, no simulator.
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
    FLANGES,
    Tracker,
    centre_offset,
    evaluate,
    flange_pose,
    held_by,
    hole_point,
    pattern_yaw_deg,
    seat_gap,
    straddle_off_deg,
    tilt_deg,
    valid,
)

HZ = 20.0
WINDOW = 20  # 1 s at 20 Hz (written out, not from layout.STILL_WINDOW_S)
TOP = 0.7475
STUB_END = TOP + 0.060  # both stub ends (written out, not from layout)
STUB_XY = {"stub_small": (0.330, 0.150), "stub_large": (0.330, -0.150)}
OWN = {"flange_4": "stub_small", "flange_8": "stub_large"}
OTHER = {"flange_4": "stub_large", "flange_8": "stub_small"}
DISC_R = {"flange_4": 0.030, "flange_8": 0.035}
PITCH = {"flange_4": 90.0, "flange_8": 45.0}
ANSWER = {"flange_4": 45.0, "flange_8": 22.5}  # half a pitch off the lines
TOL = {"flange_4": 8.0, "flange_8": 7.0}
STARTS = {"flange_4": (0.210, 0.130), "flange_8": (0.210, -0.130)}


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


def fl(x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
    """A flange as mechkit reports it: base frame (neck's bottom centre) at (x, y, z), hole 1 at heading ``yaw``."""
    o = np.array([x, y, z], dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=quat(yaw, pitch, roll), pos=o + np.array([0.0, 0.0, 0.011]),
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=o - 0.04, high=o + 0.04, source="test")


def seated(name, yaw=None, dx=0.0, dy=0.0, dz=0.0, stub=None, **kw):
    """The flange resting neck down on a stub end (its own by default), its pattern at ``yaw`` (default: the
    answer, half a pitch off the lines), its axis (dx, dy) off the stub's and its neck's bottom dz above the end."""
    sx, sy = STUB_XY[stub or OWN[name]]
    return fl(sx + dx, sy + dy, STUB_END + dz, ANSWER[name] if yaw is None else yaw, **kw)


def start(name, yaw=0.0, dx=0.0, dy=0.0):
    x, y = STARTS[name]
    return fl(x + dx, y + dy, TOP, yaw)


def arm(tips, opening=1.0):
    tips = np.asarray(tips, dtype=float)
    return SimpleNamespace(tcp=tips + np.array([0.0, 0.0, 0.013]), tips=tips, opening=opening)


PARKED = {"left": arm((0.20, 0.25, TOP + 0.147)), "right": arm((0.20, -0.25, TOP + 0.147))}


def holding(name):
    """The arm that carries it, closed on it as the demo grips it (pad tips 7 mm above the neck's bottom)."""
    side = "left" if name == "flange_4" else "right"
    sx, sy = STUB_XY[OWN[name]]
    out = dict(PARKED)
    out[side] = arm((sx, sy, STUB_END + 0.007), (2 * DISC_R[name] - 0.008) / 0.0939)
    return out


DONE = {"flange_4": seated("flange_4"), "flange_8": seated("flange_8")}
START = {"flange_4": start("flange_4"), "flange_8": start("flange_8")}


def state(**kw):
    s = dict(DONE)
    s.update(kw)
    return s


def feed(tr, states, arms=PARKED):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(st, arms=PARKED, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n, arms)


def scored(st, arms=PARKED):
    return settled(st, arms)[1].score


def text_scorer(st, arms=PARKED):
    """A second scorer written from task.json's scoring text alone (states assumed at rest)."""
    total = 0.0
    for n in FLANGES:
        f = st[n]
        o = np.asarray(f.origin, dtype=float)
        sx, sy = STUB_XY[OWN[n]]
        q = f.quat_xyzw / np.linalg.norm(f.quat_xyzw)
        x, y, z, w = q
        up_z = 1 - 2 * (x * x + y * y)
        ax = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
        on = (math.hypot(o[0] - sx, o[1] - sy) <= 0.006 and abs(o[2] - STUB_END) <= 0.003 and tilt <= 5.0)
        # let go: no pad-tip point lower than 10 mm over the face while within finger reach of the axis
        top = o[2] + 0.0225 + DISC_R[n] * math.sin(math.radians(tilt))
        for a in arms.values():
            reach = math.hypot(0.0939 * a.opening / 2 + 0.010, 0.020) + DISC_R[n] + 0.005
            if math.hypot(a.tips[0] - o[0], a.tips[1] - o[1]) < reach and a.tips[2] - top < 0.010:
                on = False
        if not on:
            continue
        total += 0.25
        psi = math.degrees(math.atan2(ax[1], ax[0])) % PITCH[n]
        if abs(psi - ANSWER[n]) <= TOL[n]:
            total += 0.25
    return round(total, 2)


class Geometry(unittest.TestCase):
    def test_straddle_angle_is_read_modulo_the_pitch(self):
        for yaw, want in ((0.0, 45.0), (45.0, 0.0), (-45.0, 0.0), (135.0, 0.0), (90.0, 45.0), (30.0, 15.0),
                          (225.0, 0.0), (-180.0, 45.0), (50.0, 5.0)):
            self.assertAlmostEqual(straddle_off_deg("flange_4", yaw), want, places=9, msg=yaw)
        for yaw, want in ((0.0, 22.5), (22.5, 0.0), (-22.5, 0.0), (67.5, 0.0), (45.0, 22.5), (90.0, 22.5),
                          (157.5, 0.0), (10.0, 12.5)):
            self.assertAlmostEqual(straddle_off_deg("flange_8", yaw), want, places=9, msg=yaw)
        self.assertEqual(straddle_off_deg("flange_4", None), 45.0)

    def test_pose_measures(self):
        f = seated("flange_8", yaw=22.5, dx=0.003, dy=-0.004, dz=0.001)
        self.assertAlmostEqual(centre_offset(f, "stub_large"), 0.005, places=9)
        self.assertAlmostEqual(seat_gap(f), 0.001, places=9)
        self.assertAlmostEqual(pattern_yaw_deg(f), 22.5, places=6)
        self.assertAlmostEqual(tilt_deg(f), 0.0, places=6)
        h = hole_point(f, "flange_8")
        self.assertAlmostEqual(math.hypot(h[0] - f.origin[0], h[1] - f.origin[1]), 0.028, places=9)
        self.assertAlmostEqual(h[2] - f.origin[2], 0.022, places=9)
        self.assertAlmostEqual(tilt_deg(fl(0.3, 0, TOP, pitch=180.0)), 180.0, places=6)
        self.assertAlmostEqual(tilt_deg(fl(0.3, 0, TOP, roll=90.0)), 90.0, places=6)

    def test_start_layout(self):
        self.assertTrue(Tracker.at_start(START))
        for n in FLANGES:
            p = flange_pose(START[n], n)
            self.assertFalse(p["on_own_stub"])
            self.assertFalse(p["pattern_ok"])  # a hole on each centreline's direction
            self.assertAlmostEqual(p["straddle_off_deg"], PITCH[n] / 2)


class Tiers(unittest.TestCase):
    def test_finished_scores_one(self):
        tr, r = settled(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["zero_reason"], "")

    def test_every_pair_maps_to_its_score(self):
        """(seated, two-holed) -> 0.25 s + 0.25 t: every pair, reached by a concrete layout."""
        on_lines = {n: seated(n, yaw=0.0) for n in FLANGES}
        cases = {
            (2, 2): state(),
            (2, 1): state(flange_8=on_lines["flange_8"]),
            (2, 0): dict(on_lines),
            (1, 1): state(flange_8=START["flange_8"]),
            (1, 0): {"flange_4": on_lines["flange_4"], "flange_8": START["flange_8"]},
            (0, 0): dict(START),
        }
        want = {(2, 2): 1.0, (2, 1): 0.75, (2, 0): 0.5, (1, 1): 0.5, (1, 0): 0.25, (0, 0): 0.0}
        for key, st in cases.items():
            tr, r = settled(st)
            self.assertEqual(r.score, want[key], key)
            self.assertEqual((r.details["seated_count"], r.details["two_holed_count"]), key)
            self.assertEqual(r.success, key == (2, 2))
        # either flange can be the one that is seated or two-holed
        self.assertEqual(scored(state(flange_4=seated("flange_4", yaw=0.0))), 0.75)
        self.assertEqual(scored(state(flange_4=START["flange_4"])), 0.5)
        self.assertEqual(scored({"flange_4": START["flange_4"], "flange_8": seated("flange_8", yaw=0.0)}), 0.25)

    def test_two_holed_windows(self):
        for n in FLANGES:
            for d in (TOL[n] - 0.1, -(TOL[n] - 0.1)):
                self.assertEqual(scored(state(**{n: seated(n, yaw=ANSWER[n] + d)})), 1.0, (n, d))
            for d in (TOL[n] + 0.1, -(TOL[n] + 0.1)):
                self.assertEqual(scored(state(**{n: seated(n, yaw=ANSWER[n] + d)})), 0.75, (n, d))
            for k in range(-4, 5):  # every equivalent straddling yaw (the pattern repeats every pitch)
                self.assertEqual(scored(state(**{n: seated(n, yaw=ANSWER[n] + k * PITCH[n])})), 1.0, (n, k))
                self.assertEqual(scored(state(**{n: seated(n, yaw=k * PITCH[n])})), 0.75, (n, k))  # on the lines

    def test_centring_seat_and_tilt_edges(self):
        for n in FLANGES:
            for dx, dy in ((0.0059, 0.0), (0.0, -0.0059), (0.0041, 0.0041)):
                self.assertEqual(scored(state(**{n: seated(n, dx=dx, dy=dy)})), 1.0)
            for dx, dy in ((0.0061, 0.0), (0.0, 0.0061), (0.0044, -0.0044)):
                self.assertEqual(scored(state(**{n: seated(n, dx=dx, dy=dy)})), 0.5)
            self.assertEqual(scored(state(**{n: seated(n, dz=0.0029)})), 1.0)
            self.assertEqual(scored(state(**{n: seated(n, dz=-0.0029)})), 1.0)
            self.assertEqual(scored(state(**{n: seated(n, dz=0.0031)})), 0.5)
            self.assertEqual(scored(state(**{n: seated(n, dz=-0.0031)})), 0.5)
            self.assertEqual(scored(state(**{n: seated(n, pitch=4.9)})), 1.0)
            self.assertEqual(scored(state(**{n: seated(n, roll=-4.9)})), 1.0)
            self.assertEqual(scored(state(**{n: seated(n, pitch=5.1)})), 0.5)

    def test_random_layouts_agree_with_the_text_scorer(self):
        rng = random.Random(7)
        outcomes = set()
        for _ in range(500):
            st = {}
            for n in FLANGES:
                stub = rng.choice((OWN[n], OWN[n], OWN[n], OTHER[n]))
                st[n] = seated(n, yaw=rng.uniform(-180, 180), dx=rng.uniform(-0.0065, 0.0065),
                               dy=rng.uniform(-0.0065, 0.0065), dz=rng.uniform(-0.0035, 0.0035), stub=stub,
                               pitch=rng.uniform(-5.5, 5.5))
                if rng.random() < 0.15:
                    st[n] = start(n, yaw=rng.uniform(-180, 180))
            s = scored(st)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertEqual(s, text_scorer(st))
            outcomes.add(s)
        self.assertEqual(outcomes, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_score_is_idempotent(self):
        tr, r = settled(DONE)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(DONE, PARKED).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def test_nothing_moved(self):
        self.assertEqual(scored(START), 0.0)

    def test_on_the_wrong_stub(self):
        st = state(flange_4=seated("flange_4", stub="stub_large"))
        tr, r = settled(st)
        self.assertEqual(r.score, 0.5)
        self.assertIn("flange_4_on_the_wrong_stub", r.details["zero_reason"])

    def test_held(self):
        for n in FLANGES:
            tr, r = settled(state(), holding(n))
            self.assertEqual(r.score, 0.5, n)
            self.assertIn(f"{n}_still_held", r.details["zero_reason"])
            self.assertTrue(r.details[n]["held_by"])

    def test_open_gripper_hovering_on_it_is_not_let_go(self):
        sx, sy = STUB_XY["stub_small"]
        face = STUB_END + 0.0225
        for above, want in ((0.009, 0.5), (0.011, 1.0)):
            arms = dict(PARKED, left=arm((sx + 0.02, sy, face + above), 1.0))
            self.assertEqual(scored(state(), arms), want, above)
        # beside it, out of a finger's reach (fully open: 57 mm + disc + 5 mm), low down: let go
        reach = math.hypot(0.0939 / 2 + 0.010, 0.020) + 0.030 + 0.005
        self.assertEqual(scored(state(), dict(PARKED, left=arm((sx, sy + reach + 0.001, STUB_END), 1.0))), 1.0)
        self.assertEqual(scored(state(), dict(PARKED, left=arm((sx, sy + reach - 0.001, STUB_END), 1.0))), 0.5)

    def test_the_other_arm_counts_too(self):
        sx, sy = STUB_XY["stub_small"]
        arms = dict(PARKED, right=arm((sx, sy, STUB_END + 0.03), 0.3))
        self.assertEqual(scored(state(), arms), 0.5)

    def test_face_down_or_on_its_rim(self):
        for n in FLANGES:
            self.assertEqual(scored(state(**{n: seated(n, pitch=180.0)})), 0.5)
            self.assertEqual(scored(state(**{n: seated(n, roll=90.0, dz=0.03)})), 0.5)

    def test_on_the_table_beside_its_stub(self):
        for n in FLANGES:
            sx, sy = STUB_XY[OWN[n]]
            self.assertEqual(scored(state(**{n: fl(sx - 0.06, sy, TOP, ANSWER[n])})), 0.5)

    def test_still_moving(self):
        tr = Tracker(HZ)
        moving = [state(flange_8=seated("flange_8", dz=0.04 - 0.002 * k)) for k in range(WINDOW + 1)]
        r = feed(tr, moving)
        self.assertNotEqual(r.score, 1.0)
        self.assertIn("flange_8", r.details["zero_reason"])


class Stillness(unittest.TestCase):
    def test_needs_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [DONE] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.update(DONE, PARKED).score, 1.0)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        falling = [state(flange_4=seated("flange_4", dz=0.002 * (10 - k))) for k in range(10)]
        feed(tr, falling)
        r = feed(tr, [DONE] * WINDOW)
        self.assertLess(r.score, 1.0)
        self.assertEqual(tr.update(DONE, PARKED).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        seq = [state(flange_8=seated("flange_8", dx=0.0009 * (-1) ** k)) for k in range(WINDOW + 1)]
        self.assertEqual(feed(tr, seq).score, 1.0)

    def test_turning_in_place_is_not_still(self):
        tr = Tracker(HZ)
        seq = [state(flange_4=seated("flange_4", yaw=40.0 + 0.5 * k)) for k in range(WINDOW + 1)]
        r = feed(tr, seq)  # 10 degrees in 1 s: hole 1 moves 4 mm
        self.assertEqual(r.score, 0.5)
        self.assertIn("flange_4_still_moving", r.details["zero_reason"])


class Exploits(unittest.TestCase):
    """The spec's shortcut checks, and more."""

    def test_holes_on_the_lines_the_neat_looking_answer(self):
        self.assertEqual(scored({n: seated(n, yaw=0.0) for n in FLANGES}), 0.5)

    def test_the_4_hole_answer_copied_onto_the_8_hole_flange(self):
        self.assertEqual(scored({n: seated(n, yaw=45.0) for n in FLANGES}), 0.75)

    def test_the_8_hole_answer_copied_onto_the_4_hole_flange(self):
        self.assertEqual(scored({n: seated(n, yaw=22.5) for n in FLANGES}), 0.75)  # 22.5 is 22.5 off for 4 holes

    def test_swapped_stubs_turned_correctly(self):
        st = {n: seated(n, stub=OTHER[n]) for n in FLANGES}
        self.assertEqual(scored(st), 0.0)

    def test_one_correct_the_other_still_held(self):
        self.assertEqual(scored(state(), holding("flange_8")), 0.5)

    def test_stacked_on_the_other_flange(self):
        st = state(flange_4=fl(0.330, -0.150, STUB_END + 0.022, 45.0))
        self.assertEqual(scored(st), 0.5)

    def test_pushed_against_its_stub_on_the_table(self):
        sx, sy = STUB_XY["stub_small"]
        st = state(flange_4=fl(sx - 0.048, sy, TOP, 45.0))
        self.assertEqual(scored(st), 0.5)

    def test_leaning_on_its_stub_end(self):
        self.assertEqual(scored(state(flange_8=seated("flange_8", dx=0.02, dz=-0.004, pitch=20.0))), 0.5)

    def test_unreadable_arms_cannot_skip_the_rest_test(self):
        tr = Tracker(HZ)
        r = feed(tr, [state(flange_4=seated("flange_4", dz=0.002 * k)) for k in range(WINDOW + 1)], None)
        self.assertLess(r.score, 1.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_not_holding_and_counted(self):
        tr, r = settled(DONE, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)

    def test_one_unreadable_arm_does_not_hide_the_other(self):
        arms = holding("flange_4")
        arms["right"] = SimpleNamespace(tcp=np.full(3, np.nan), tips=np.full(3, np.nan), opening=float("nan"))
        tr, r = settled(DONE, arms)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["readable_arms"], ["left"])
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)

    def test_held_by_reads_each_arm(self):
        self.assertEqual(held_by(DONE["flange_4"], "flange_4", holding("flange_4")), ["left"])
        self.assertEqual(held_by(DONE["flange_8"], "flange_8", holding("flange_8")), ["right"])
        self.assertEqual(held_by(DONE["flange_8"], "flange_8", PARKED), [])
        self.assertEqual(held_by(DONE["flange_8"], "flange_8", {}), [])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(DONE)
        n = len(tr.history["flange_4"])
        bad = [
            state(flange_4=SimpleNamespace(**{**vars(DONE["flange_4"]), "origin": np.array([np.nan, 0.0, 0.0])})),
            state(flange_8=SimpleNamespace(**{**vars(DONE["flange_8"]), "quat_xyzw": np.zeros(4)})),
            state(flange_8=SimpleNamespace(**{**vars(DONE["flange_8"]), "pos": np.array([np.inf, 0.0, 0.0])})),
            state(flange_4=SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3))),  # no origin
            {"flange_4": DONE["flange_4"]},  # a flange missing
            None,
        ]
        for b in bad:
            self.assertFalse(valid(b))
            r = tr.update(b, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(len(tr.history["flange_4"]), n)
        self.assertEqual(tr.update(DONE, PARKED).score, 1.0)  # the record carried on as it was

    def test_unnormalised_quaternion(self):
        f = DONE["flange_4"]
        g = SimpleNamespace(**{**vars(f), "quat_xyzw": 3.0 * np.asarray(f.quat_xyzw)})
        self.assertEqual(scored(state(flange_4=g)), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(3)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {n: fl(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1.5), rng.uniform(-180, 180),
                        rng.uniform(-180, 180), rng.uniform(-180, 180)) for n in FLANGES}
            self.assertIn(tr.update(st, PARKED).score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(DONE)
        r = tr.update(START, PARKED)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(DONE)
        jit = {"flange_4": start("flange_4", dx=0.03, dy=-0.03), "flange_8": start("flange_8", 90.0, -0.02, 0.03)}
        tr.update(jit, PARKED)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        a, b = DONE, START
        seq = []
        for k in range(41):
            s = k / 40
            seq.append({n: fl(*(np.asarray(a[n].origin) * (1 - s) + np.asarray(b[n].origin) * s)) for n in FLANGES})
        feed(tr, seq)
        self.assertEqual(tr.steps, 41)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(DONE)
        tr.update({n: fl(0.25, 0.0, TOP) for n in FLANGES}, PARKED)
        self.assertEqual(tr.steps, WINDOW + 2)

    def test_explicit_reset(self):
        tr, _ = settled(DONE)
        tr.reset()
        self.assertEqual((tr.steps, tr.arms_unavailable), (0, 0))
        self.assertEqual(tr.update(DONE, PARKED).score, 0.0)  # no history yet


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, st):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        t = self.task
        specs = {"flange_4": t.FLANGE_4, "flange_8": t.FLANGE_8}
        return {n: self.mk.state_from_poses(specs[n], (*st[n].origin, *st[n].quat_xyzw), source="sim")
                for n in FLANGES}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], ["flange_4", "flange_8"])
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        for s in t.STUBS:
            self.assertAlmostEqual(s.pos[2] + s.height / 2, STUB_END)
            self.assertEqual(tuple(s.pos[:2]), STUB_XY[s.name])
        self.assertEqual({s.name: round(2 * s.radius, 4) for s in t.STUBS}, {"stub_small": 0.036, "stub_large": 0.05})
        for o in t.OBJECTS:  # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            elif hasattr(o, "size"):
                lo, hi = np.asarray(o.pos) - np.asarray(o.size) / 2, np.asarray(o.pos) + np.asarray(o.size) / 2
            else:
                lo, hi = np.asarray(o.pos) - o.radius, np.asarray(o.pos) + o.radius
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)
            self.assertGreaterEqual(lo[0], 0.17, o.name)

    def test_centrelines_cross_at_each_stub_axis(self):
        t = self.task
        for tag, (sx, sy) in (("small", STUB_XY["stub_small"]), ("large", STUB_XY["stub_large"])):
            lines = {o.name: o for o in t.LINES if o.name.endswith(tag)}
            self.assertEqual(len(lines), 5)
            for k in ("n", "s"):  # along x, on the stub's y
                self.assertAlmostEqual(lines[f"line_{k}_{tag}"].pos[1], sy)
                self.assertAlmostEqual(lines[f"line_{k}_{tag}"].size[1], 0.004)
            for k in ("w", "e"):  # along y, on the stub's x
                self.assertAlmostEqual(lines[f"line_{k}_{tag}"].pos[0], sx)
            self.assertAlmostEqual(lines[f"line_n_{tag}"].pos[0] + lines[f"line_n_{tag}"].size[0] / 2, sx + 0.080)
            self.assertAlmostEqual(lines[f"line_w_{tag}"].pos[1] + lines[f"line_w_{tag}"].size[1] / 2, sy + 0.080)
            self.assertGreater(lines[f"north_mark_{tag}"].pos[0], sx + 0.080)
        # the starting flanges lie clear of every painted line (seen from above)
        for n in FLANGES:
            x, y = STARTS[n]
            for o in t.LINES:
                dx = max(abs(x - o.pos[0]) - o.size[0] / 2, 0.0)
                dy = max(abs(y - o.pos[1]) - o.size[1] / 2, 0.0)
                self.assertGreater(math.hypot(dx, dy), DISC_R[n] + 0.003, (n, o.name))

    def test_flange_geometry_matches_the_scorer(self):
        t = self.task
        for spec, n, holes in ((t.FLANGE_4, "flange_4", 4), (t.FLANGE_8, "flange_8", 8)):
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-DISC_R[n], -DISC_R[n], 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [DISC_R[n], DISC_R[n], 0.0225], atol=1e-12)
            paint = [s for s in spec.shapes if s.height == 0.0005]
            hole_shapes = [s for s in paint if tuple(s.color) == L.HOLE_BLACK]
            self.assertEqual(len(hole_shapes), holes)
            self.assertEqual(len(paint), holes + 1)  # and the bore
            angles = sorted(round(math.degrees(math.atan2(s.center[1], s.center[0])) % 360, 6) for s in hole_shapes)
            self.assertEqual(angles, [round(360.0 * k / holes, 6) for k in range(holes)])  # hole 1 on local +x
            bore = [s for s in paint if tuple(s.color) == L.BORE_DARK][0]
            for s in hole_shapes:  # holes on the face, clear of the bore and the rim
                r = math.hypot(*s.center[:2])
                self.assertGreater(r - s.radius, bore.radius + 0.003)
                self.assertLessEqual(r + s.radius, DISC_R[n] - 0.003 + 1e-9)
            neck = min((s for s in spec.shapes if s.height == 0.010), key=lambda s: s.center[2])
            self.assertAlmostEqual(neck.radius, {"flange_4": 0.018, "flange_8": 0.025}[n])  # the stub's OD
            vol = sum(math.pi * s.radius ** 2 * s.height for s in spec.shapes if s.height > 0.001)
            self.assertAlmostEqual(vol * 2700.0, spec.mass, delta=0.003)  # solid aluminium
            self.assertLessEqual(spec.mass, 0.30)
        rest = {"flange_4": t.FLANGE_4.rest_state(), "flange_8": t.FLANGE_8.rest_state()}
        self.assertTrue(Tracker.at_start(rest))
        st = self.sim(DONE)
        for n in FLANGES:
            self.assertAlmostEqual(st[n].low[2], STUB_END, places=9)
            self.assertAlmostEqual(pattern_yaw_deg(st[n]), ANSWER[n], places=6)

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.FLANGE_8.offline_box()[0]), quat_xyzw=quat(0.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.FLANGE_8.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [0.210, -0.130, TOP], atol=1e-12)

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(PARKED)
        st0 = self.sim(START)
        self.assertFalse(t.success(st0))
        self.assertEqual(t.score(st0), 0.0)
        done = self.sim(DONE)
        for _ in range(WINDOW):
            self.feed_arms(PARKED)
            t.success(done)
        self.feed_arms(PARKED)
        self.assertTrue(t.success(done))
        self.assertEqual(t.score(done), 1.0)
        self.assertEqual(t.score(done), 1.0)  # score() does not advance it
        self.feed_arms(holding("flange_4"))
        self.assertEqual(t.score(done), 0.5)  # read with the arms of the step at hand
        self.feed_arms(PARKED)
        n = tr.steps
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(PARKED)
        t.success(done)
        self.assertEqual(tr.steps, 1)
        t.success(done)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(PARKED)
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "moved_mm", "arms_unavailable", "readable_arms",
                          "seated_count", "two_holed_count", "flange_4_origin", "flange_8_quat", "flange_4_hole1",
                          "flange_8_source", "left_tips", "openings"):
                    self.assertIn(k, row)
                for n in FLANGES:
                    for k in ("centre_mm", "seat_gap_mm", "tilt_deg", "pattern_yaw_deg", "straddle_off_deg",
                              "on_own_stub", "on_other_stub", "held_by", "at_rest", "seated", "two_holed",
                              "nearest_stub"):
                        self.assertIn(k, row[n])
                self.assertAlmostEqual(row["flange_8"]["straddle_off_deg"], 0.0)
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
        for side, n in (("left", "flange_4"), ("right", "flange_8")):
            wps = plan[side]
            by = {w.label: w for w in wps}
            for prev, w in zip(wps, wps[1:]):  # every close and open takes at least 1.5 s
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)
            self.assertGreaterEqual(by["slowly down around its rim"].duration, 1.5)
            self.assertGreaterEqual(by["slowly onto the stub end"].duration, 1.5)
            self.assertGreaterEqual(by["settle"].duration, 0.5)
            grip, setw, letgo = by["grip the flange"], by["slowly onto the stub end"], by["let go of it"]
            self.assertEqual(grip.pos[:2], STARTS[n])
            self.assertAlmostEqual(-2 * grip.finger, 2 * DISC_R[n] - 0.008)  # 8 mm narrower than the disc
            self.assertEqual(letgo.pos, setw.pos)
            # the pads 7 mm above the table at the pick, under the disc's 10 mm underside
            self.assertAlmostEqual(grip.pos[2] - 0.013 - TOP, 0.007)
            self.assertAlmostEqual(setw.pos[2] - t.GRIP_UP - STUB_END, 0.001)  # neck let go 1 mm above the end
            # carried with the neck's bottom 40 mm over the stub ends
            self.assertAlmostEqual(by["lift it"].pos[2] - t.GRIP_UP - STUB_END, 0.040)
            # the place point leads the stub's axis by 4 mm along the carry
            sx, sy = STUB_XY[OWN[n]]
            self.assertAlmostEqual(math.hypot(setw.pos[0] - sx, setw.pos[1] - sy), 0.004)
            # the pattern is turned half a pitch, commanded 1 degree clockwise for the counter-clockwise bias
            turn = math.degrees(setw.yaw - grip.yaw)
            self.assertAlmostEqual(straddle_off_deg(n, turn + 1.0), 0.0, places=9)
            for bias in (0.0, 0.6, 1.7, 2.5):
                self.assertLessEqual(straddle_off_deg(n, turn + bias), TOL[n] - 5.0)
            # the wrist stays within 90 degrees of its bearing from its base at pick and place (joint 6)
            base_y = 0.305 if side == "left" else -0.305
            for w in (grip, setw):
                bearing = math.degrees(math.atan2(w.pos[1] - base_y, w.pos[0]))
                self.assertLess(abs(math.remainder(math.degrees(w.yaw) - bearing, 180.0)), 90.0)
            # up after letting go: pad tips well over the flange's face, so it counts as let go
            self.assertGreater(by["straight up"].pos[2] - 0.013 - (STUB_END + 0.0225), 0.010 + 0.040)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_landing_error_budget(self):
        """With the 4 mm lead, a flange that lands anywhere from 0 to 9 mm short along the carry is still within the
        6 mm centring window; the open pads clear the stub when the flange is let go."""
        t = self.task
        for n in FLANGES:
            px, py = STARTS[n]
            sx, sy = STUB_XY[OWN[n]]
            qx, qy = t.place_point(n)
            ux, uy = (sx - px), (sy - py)
            d = math.hypot(ux, uy)
            for short in (0.0, 0.004, 0.007, 0.009):
                x, y = qx - short * ux / d, qy - short * uy / d
                self.assertLessEqual(math.hypot(x - sx, y - sy), 0.006, (n, short))
            stub_r = 0.018 if n == "flange_4" else 0.025
            self.assertGreater(DISC_R[n] - 0.004, stub_r + 0.005)  # gripping pads outside the stub

    def test_instruction_withholds_the_answer(self):
        ins = self.task.INSTRUCTION
        for s in ("stub it is bored for", "neck down", "centred", "two-hole", "plant centrelines", "bolt holes",
                  "let go"):
            self.assertIn(s, ins)
        for s in ("45", "22.5", "straddle", "degrees", "between the lines"):
            self.assertNotIn(s, ins)
        j = json.loads((HERE / "task.json").read_text())
        self.assertEqual(j["instruction"], ins)


if __name__ == "__main__":
    unittest.main()
