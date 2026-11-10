"""Offline tests for scoring.py and task.py: synthetic kerb unit states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
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
    NAMES,
    Tracker,
    back_corners,
    face_angle_deg,
    high_end_y,
    rotation,
    slab_corners,
    tilt_deg,
    unit_facts,
    valid,
)

HZ = L.CONTROL_HZ
WINDOW = int(round(L.REST_WINDOW_S * HZ)) + 1  # 11 samples: 0.5 s of stillness
SLOTS = tuple(L.SLOTS)
RIGHT_UNIT = {"S1": "transition_left", "S2": "low_kerb_1", "S3": "low_kerb_2", "S4": "transition_right"}


def quat(yaw_deg=0.0, tilt_deg_=0.0, tilt_axis="x"):
    """Turned yaw about the vertical after tilting about the body's x (or y) axis, as (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qt = np.array([math.sin(b), 0.0, 0.0, math.cos(b)]) if tilt_axis == "x" else \
        np.array([0.0, math.sin(b), 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qt
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def unit(name, x, y, heading=0.0, low=None, tilt=0.0, tilt_axis="x", vel=(0, 0, 0), spin=(0, 0, 0)):
    """A unit's state as mechkit reports it: its frame's origin at (x, y) with its slab's lowest corner at ``low``
    (default: on the bed), ``heading`` the world angle of its own +x (degrees), and the bounding box of every shape."""
    q = quat(heading, tilt, tilt_axis)
    r = rotation(q)
    slab = np.array([(sx * L.UNIT_LEN / 2, sy * L.UNIT_DEPTH / 2, sz * (L.LOW_H - L.LIP_DROP))
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    oz = (L.TOP if low is None else low) - float((slab @ r.T)[:, 2].min())
    origin = np.array([x, y, oz])
    pts = []
    for c, s, _ in L.unit_shapes(name):
        h = np.array(s) / 2
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            pts.append(origin + r @ (np.array(c) + h * np.array([sx, sy, sz])))
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=origin, pos=(lo + hi) / 2, quat_xyzw=q, lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=lo, high=hi)


def shifted(s, dx=0.0, dy=0.0, dz=0.0):
    d = np.array([dx, dy, dz])
    return SimpleNamespace(origin=s.origin + d, pos=s.pos + d, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel,
                           ang_vel=s.ang_vel, low=s.low + d, high=s.high + d)


def laid(name, slot, back=0.0, along=0.0, turn=0.0, reverse=False, **kw):
    """A unit laid in this slot: heading -90 degrees (face to the road; +180 with ``reverse``: face to the footway)
    turned ``turn`` degrees more, the middle of its back face ``back`` in front of the kerb line (+x) and its middle
    ``along`` from the slot's centre."""
    heading = -90.0 + turn + (180.0 if reverse else 0.0)
    s = unit(name, L.LINE_X, L.SLOTS[slot] + along, heading, **kw)
    mid_back = float(np.mean(back_corners(s)[:, 0]))
    return shifted(s, dx=L.BACK_X + back - mid_back)


def start_unit(name):
    x, y, heading = L.UNIT_START[name]
    return unit(name, x, y, math.degrees(heading))


START = {n: start_unit(n) for n in NAMES}
DONE = {RIGHT_UNIT[k]: laid(RIGHT_UNIT[k], k) for k in SLOTS}


def scene(**over):
    """The starting layout with some units replaced (by name)."""
    out = dict(START)
    out.update(over)
    return out


def run(frames, t=None):
    """Feed the frames to a tracker (a fresh one unless given); return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


def score_of(sc):
    """The score of a scene held still for 0.5 s."""
    return run([sc] * WINDOW)[0].score


def details_of(sc):
    return run([sc] * WINDOW)[0].details


def glide(a, b, n):
    """n frames moving every unit from scene a to scene b in equal steps (the orientation of b)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        frame = {}
        for k in NAMES:
            sa, sb = a[k], b[k]
            frame[k] = SimpleNamespace(origin=(1 - f) * sa.origin + f * sb.origin, pos=(1 - f) * sa.pos + f * sb.pos,
                                       quat_xyzw=sb.quat_xyzw, lin_vel=(sb.pos - sa.pos) * HZ / n,
                                       ang_vel=np.zeros(3), low=(1 - f) * sa.low + f * sb.low,
                                       high=(1 - f) * sa.high + f * sb.high)
        out.append(frame)
    return out


class Geometry(unittest.TestCase):
    def test_slots_and_joints(self):
        """Four 60 mm units and five 6 mm joints fill the 270 mm gap exactly."""
        ys = sorted(L.SLOTS.values(), reverse=True)
        self.assertAlmostEqual(ys[0] + L.UNIT_LEN / 2 + L.JOINT, L.GAP_HALF)
        self.assertAlmostEqual(ys[-1] - L.UNIT_LEN / 2 - L.JOINT, -L.GAP_HALF)
        for a, b in zip(ys, ys[1:]):
            self.assertAlmostEqual(a - b, L.UNIT_LEN + L.JOINT)
        self.assertAlmostEqual(4 * L.UNIT_LEN + 5 * L.JOINT, 2 * L.GAP_HALF)
        self.assertAlmostEqual(L.SLOTS["S1"], 0.099)
        self.assertAlmostEqual(L.SLOTS["S2"], 0.033)
        # Slots are far apart: no unit can be within 8 mm of two slot centres.
        self.assertGreater(min(a - b for a, b in zip(ys, ys[1:])), 2 * L.ALONG_TOL)
        # The red paving lies behind the middle two slots and stops short of the kerb backs.
        self.assertAlmostEqual(L.PAVING_Y, L.UNIT_LEN + L.JOINT)  # to the middle of the joints beside S1 and S4
        self.assertLess(L.PAVING_X1, L.BACK_X - 0.005)
        self.assertGreater(L.ROAD_X0, L.FACE_X + L.LINE_TOL - 1e-12)  # a bedded unit never reaches the road strip

    def test_unit_shapes(self):
        for n in NAMES:
            shapes = L.unit_shapes(n)
            for c, s, _ in shapes:  # every box inside the 60 x 25 mm footprint, standing on the bed
                self.assertLessEqual(abs(c[0]) + s[0] / 2, L.UNIT_LEN / 2 + 1e-12)
                self.assertLessEqual(abs(c[1]) + s[1] / 2, L.UNIT_DEPTH / 2 + 1e-12)
                self.assertAlmostEqual(c[2] - s[2] / 2, 0.0)
            # the darker face strip is on the unit's +y side, 4 mm lower than the part behind it
            lips = [(c, s) for c, s, col in shapes if col == L.FACE_GREY]
            backs = [(c, s) for c, s, col in shapes if col == L.CONCRETE]
            self.assertTrue(all(c[1] > 0 for c, _ in lips) and all(c[1] < 0 for c, _ in backs))
            for (cl, sl), (cb, sb) in zip(sorted(lips), sorted(backs)):
                self.assertAlmostEqual(cl[0], cb[0])
                self.assertAlmostEqual(sb[2] - sl[2], L.LIP_DROP)
        self.assertEqual(len(L.unit_shapes("low_kerb_1")), 2)
        self.assertEqual(len(L.unit_shapes("transition_left")), 6)
        # The two transitions are mirror images: one is the other reflected along its length.
        a = sorted((round(c[0], 6), round(c[1], 6), round(s[2], 6)) for c, s, _ in L.unit_shapes("transition_left"))
        b = sorted((round(-c[0], 6), round(c[1], 6), round(s[2], 6)) for c, s, _ in L.unit_shapes("transition_right"))
        self.assertEqual(a, b)
        # ... and not the same part: turned half a turn, one is not the other (its face strip ends up behind).
        c = sorted((round(-c[0], 6), round(-c[1], 6), round(s[2], 6)) for c, s, _ in L.unit_shapes("transition_right"))
        self.assertNotEqual(a, c)
        heights = sorted(s[2] for c, s, col in L.unit_shapes("transition_right") if col == L.CONCRETE)
        self.assertEqual([round(h, 4) for h in heights], [0.012, 0.021, 0.030])

    def test_laid_right_heights_meet(self):
        """All four laid right, the kerb top runs 30 | 30-21-12 | 12 | 12 | 12-21-30 | 30 mm: no step at a joint."""
        def top_at(name, end):  # back-part height at the unit's end nearer +y (end=+1) or -y (end=-1)
            r = rotation(DONE[name].quat_xyzw)
            best = None
            for c, s, col in L.unit_shapes(name):
                if col != L.CONCRETE:
                    continue
                wy = float((r @ np.array(c))[1])
                if best is None or end * wy > end * best[0]:
                    best = (wy, s[2])
            return best[1]
        line = [L.FULL_H]
        for k in SLOTS:
            n = RIGHT_UNIT[k]
            line += [top_at(n, 1), top_at(n, -1)]
        line.append(L.FULL_H)
        steps = [abs(a - b) for a, b in zip(line[0::2], line[1::2])]
        self.assertTrue(all(s < 1e-9 for s in steps), (line, steps))
        self.assertEqual([round(v, 3) for v in line], [0.03, 0.03, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.03, 0.03])

    def test_high_end_and_face(self):
        for k, toward in (("S1", 1.0), ("S4", -1.0)):
            n = RIGHT_UNIT[k]
            r = rotation(DONE[n].quat_xyzw)
            self.assertAlmostEqual(high_end_y(n, r), toward, places=9)
            self.assertAlmostEqual(face_angle_deg(r), 0.0, places=6)
        # At the start both transitions have their tall ends toward the robot (-x) and faces toward each other.
        for n, face_y in (("transition_left", 1.0), ("transition_right", -1.0)):
            r = rotation(START[n].quat_xyzw)
            self.assertAlmostEqual(high_end_y(n, r), 0.0, places=9)
            self.assertAlmostEqual(float(L.HIGH_END[n] * r[0, 0]), -1.0, places=9)
            self.assertAlmostEqual(float(r[1, 1]), face_y, places=9)
        # The low kerbs start faces outward.
        self.assertAlmostEqual(float(rotation(START["low_kerb_1"].quat_xyzw)[1, 1]), 1.0, places=9)
        self.assertAlmostEqual(float(rotation(START["low_kerb_2"].quat_xyzw)[1, 1]), -1.0, places=9)
        self.assertEqual(high_end_y("low_kerb_1", rotation(DONE["low_kerb_1"].quat_xyzw)), 0.0)
        # The wrong transition turned to rise toward the left kerb shows its face to the footway.
        r = rotation(laid("transition_right", "S1", reverse=True).quat_xyzw)
        self.assertGreater(high_end_y("transition_right", r), 0.99)
        self.assertAlmostEqual(face_angle_deg(r), 180.0, places=6)

    def test_face_angle_threshold(self):
        for turn, ok in ((29.9, True), (-29.9, True), (30.1, False), (-30.1, False)):
            f = unit_facts("low_kerb_1", unit("low_kerb_1", 0.3, 0.0, -90.0 + turn))
            self.assertEqual(f["face_to_road"], ok, turn)
        self.assertEqual(face_angle_deg(rotation(quat(0.0, 90.0, "x"))), 90.0)  # on its face or back: no direction

    def test_back_corners_and_slab(self):
        s = DONE["low_kerb_2"]
        self.assertTrue(np.allclose(back_corners(s)[:, 0], L.BACK_X))
        self.assertTrue(np.allclose(sorted(back_corners(s)[:, 1]), [L.SLOTS["S3"] - 0.03, L.SLOTS["S3"] + 0.03]))
        c = slab_corners(s)
        self.assertTrue(np.allclose(sorted(c[:, 2]), [L.TOP] * 4 + [L.TOP + 0.008] * 4))
        f = unit_facts("transition_left", laid("transition_left", "S1", back=0.003, turn=2.0))
        self.assertAlmostEqual(float(np.mean(f["back_off"])), 0.003, places=9)
        self.assertAlmostEqual(f["back_off"][1] - f["back_off"][0], 0.06 * math.sin(math.radians(2.0)), places=9)

    def test_back_corners_built_independently(self):
        """A unit set by its middle on the kerb line's middle (x 38.5 cm), face to the road, has its back corners on
        the line; 6 mm further forward it is out of line (no helper that itself uses back_corners)."""
        s = unit("low_kerb_1", L.LINE_X, L.SLOTS["S2"], -90.0)
        self.assertTrue(np.allclose(back_corners(s)[:, 0], L.BACK_X))
        f = unit_facts("low_kerb_1", s)
        self.assertTrue(f["in_line"] and f["face_to_road"] and f["flat"] and f["on_bed"])
        self.assertEqual(score_of(scene(low_kerb_1=s)), 0.25)
        self.assertEqual(score_of(scene(low_kerb_1=unit("low_kerb_1", L.LINE_X + 0.006, L.SLOTS["S2"], -90.0))), 0.0)
        self.assertEqual(score_of(scene(low_kerb_1=unit("low_kerb_1", L.LINE_X - 0.006, L.SLOTS["S2"], -90.0))), 0.0)

    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(rotation(quat(30.0, 4.0))), 4.0, places=5)
        self.assertAlmostEqual(tilt_deg(rotation(quat(0.0, 180.0))), 180.0, places=5)

    def test_valid(self):
        good = START["low_kerb_1"]
        self.assertTrue(valid(good))
        for key, val in (("origin", np.array([np.nan, 0.0, 0.76])), ("quat_xyzw", np.zeros(4)),
                         ("pos", np.array([np.inf, 0.0, 0.0])), ("lin_vel", np.array([0.0, np.nan, 0.0])),
                         ("origin", np.zeros(2)), ("quat_xyzw", np.zeros(3) + 0.5), ("low", np.array([]))):
            bad = SimpleNamespace(**vars(good))
            setattr(bad, key, val)
            self.assertFalse(valid(bad), key)
        no_origin = SimpleNamespace(**{k: v for k, v in vars(good).items() if k != "origin"})
        self.assertFalse(valid(no_origin))  # built as a plain box, not a compound
        self.assertFalse(valid(None))
        self.assertFalse(valid(SimpleNamespace(**dict(vars(good), origin="x"))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = run([START] * WINDOW)[0]
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["zero"], "no slot holds the right unit laid the right way")

    def test_the_demo_end_scores_one(self):
        r = run([DONE] * WINDOW)[0]
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(all(row["right"] for row in r.details["slots"].values()))

    def test_success_only_with_all_four(self):
        for slot in SLOTS:
            r = run([dict(DONE, **{RIGHT_UNIT[slot]: START[RIGHT_UNIT[slot]]})] * WINDOW)[0]
            self.assertEqual(r.score, 0.75)
            self.assertFalse(r.success, slot)

    def test_every_number_of_slots(self):
        """0.25 per right slot, whichever slots: all 16 combinations; the other units left in the stock area."""
        for k in range(5):
            for chosen in itertools.combinations(SLOTS, k):
                sc = dict(START)
                for slot in chosen:
                    sc[RIGHT_UNIT[slot]] = DONE[RIGHT_UNIT[slot]]
                self.assertEqual(score_of(sc), 0.25 * k, chosen)

    def test_every_outcome_in_the_text(self):
        # low kerbs right, transitions swapped and laid face to the road (each rises toward the middle): 0.5
        sw = dict(DONE, transition_left=laid("transition_left", "S4"), transition_right=laid("transition_right", "S1"))
        self.assertEqual(score_of(sw), 0.5)
        # low kerbs right, transitions swapped and turned to rise toward the full kerbs (faces to the footway): 0.5
        sw2 = dict(DONE, transition_left=laid("transition_left", "S4", reverse=True),
                   transition_right=laid("transition_right", "S1", reverse=True))
        self.assertEqual(score_of(sw2), 0.5)
        # transitions right, low kerbs left in the stock area: 0.5
        self.assertEqual(score_of(dict(START, transition_left=DONE["transition_left"],
                                       transition_right=DONE["transition_right"])), 0.5)
        # a low kerb in S1 and a transition in S2: nothing for either slot; the other two right: 0.5
        mixed = dict(DONE, low_kerb_1=laid("low_kerb_1", "S1"), transition_left=laid("transition_left", "S2"))
        self.assertEqual(score_of(mixed), 0.5)
        # three right and the fourth 12 mm off its slot, or set back 8 mm from the line: 0.75
        self.assertEqual(score_of(dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", along=0.012))), 0.75)
        self.assertEqual(score_of(dict(DONE, transition_right=laid("transition_right", "S4", back=-0.008))), 0.75)
        # one transition right only: 0.25
        self.assertEqual(score_of(scene(transition_right=DONE["transition_right"])), 0.25)
        self.assertEqual(score_of(START), 0.0)

    def test_low_kerbs_are_interchangeable(self):
        sc = dict(DONE, low_kerb_1=laid("low_kerb_1", "S3"), low_kerb_2=laid("low_kerb_2", "S2"))
        self.assertEqual(score_of(sc), 1.0)

    def test_back_line_threshold(self):
        for back in (0.0049, -0.0049):
            self.assertEqual(score_of(dict(DONE, low_kerb_2=laid("low_kerb_2", "S3", back=back))), 1.0, back)
        for back in (0.0051, -0.0051, 0.020):
            self.assertEqual(score_of(dict(DONE, low_kerb_2=laid("low_kerb_2", "S3", back=back))), 0.75, back)

    def test_a_turned_unit_is_judged_at_both_back_corners(self):
        """Turned 4 degrees about its middle, its back corners are 2.1 mm either side of the line: bedded. Turned 10
        degrees, 5.2 mm: not, though its middle is on the line."""
        self.assertEqual(score_of(dict(DONE, transition_left=laid("transition_left", "S1", turn=4.0))), 1.0)
        self.assertEqual(score_of(dict(DONE, transition_left=laid("transition_left", "S1", turn=-4.0))), 1.0)
        self.assertEqual(score_of(dict(DONE, transition_left=laid("transition_left", "S1", turn=10.0))), 0.75)
        # turned 4 degrees and set 3 mm forward: one corner 5.1 mm off: not
        self.assertEqual(score_of(dict(DONE, transition_left=laid("transition_left", "S1", turn=4.0,
                                                                  back=0.003))), 0.75)

    def test_along_threshold(self):
        for along in (0.0079, -0.0079):
            self.assertEqual(score_of(dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", along=along))), 1.0, along)
        for along in (0.0081, -0.0081, 0.033):
            self.assertEqual(score_of(dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", along=along))), 0.75, along)

    def test_tilt_and_bed_thresholds(self):
        ok = laid("transition_right", "S4", tilt=4.9, tilt_axis="x")
        self.assertEqual(score_of(dict(DONE, transition_right=ok)), 1.0)
        bad = laid("transition_right", "S4", tilt=5.1, tilt_axis="x")
        self.assertEqual(score_of(dict(DONE, transition_right=bad)), 0.75)
        for dz, want in ((0.0029, 1.0), (0.0031, 0.75), (-0.0029, 1.0), (-0.0031, 0.75)):
            s = laid("transition_right", "S4", low=L.TOP + dz)
            self.assertEqual(score_of(dict(DONE, transition_right=s)), want, dz)


class ZeroConditions(unittest.TestCase):
    def test_wrong_hand_at_each_end(self):
        """At S1 only the left-end transition laid face to the road counts; every other way round of either
        transition earns nothing there; likewise at S4."""
        for slot, good in (("S1", "transition_left"), ("S4", "transition_right")):
            other = "transition_right" if good == "transition_left" else "transition_left"
            for name, reverse, want in ((good, False, True), (good, True, False), (other, False, False),
                                        (other, True, False)):
                sc = dict(START, **{name: laid(name, slot, reverse=reverse)})
                d = details_of(sc)
                self.assertEqual(d["slots"][slot]["right"], want, (slot, name, reverse))
                self.assertEqual(d["slots_right"], int(want))

    def test_transition_in_a_middle_slot_and_low_kerb_at_an_end(self):
        for slot in ("S2", "S3"):
            for name in ("transition_left", "transition_right"):
                self.assertEqual(score_of(scene(**{name: laid(name, slot)})), 0.0, (slot, name))
        for slot in ("S1", "S4"):
            self.assertEqual(score_of(scene(low_kerb_1=laid("low_kerb_1", slot))), 0.0, slot)

    def test_low_kerb_face_to_the_footway(self):
        self.assertEqual(score_of(scene(low_kerb_1=laid("low_kerb_1", "S2", reverse=True))), 0.0)
        # ... and turned round lying in the slot itself (its back face 25 mm in front of the line)
        s = laid("low_kerb_1", "S2", reverse=True)
        self.assertEqual(score_of(scene(low_kerb_1=shifted(s, dx=L.UNIT_DEPTH))), 0.0)

    def test_on_its_side_end_or_upside_down(self):
        for tilt, axis in ((90.0, "x"), (-90.0, "x"), (90.0, "y"), (180.0, "x"), (180.0, "y")):
            s = laid("transition_left", "S1", tilt=tilt, tilt_axis=axis)
            self.assertEqual(score_of(scene(transition_left=s)), 0.0, (tilt, axis))

    def test_held_up_or_on_something(self):
        self.assertEqual(score_of(scene(low_kerb_2=laid("low_kerb_2", "S3", low=L.TOP + 0.004))), 0.0)
        on_full = laid("transition_right", "S4", along=-0.10, low=L.TOP + L.FULL_H)  # lying on the right full kerb
        self.assertEqual(score_of(scene(transition_right=on_full)), 0.0)
        under = laid("low_kerb_1", "S2")
        over = laid("low_kerb_2", "S2", low=L.TOP + L.LOW_H)
        self.assertEqual(score_of(scene(low_kerb_1=under, low_kerb_2=over)), 0.25)  # the one beneath counts
        self.assertEqual(score_of(scene(low_kerb_2=over)), 0.0)

    def test_in_the_road_or_on_the_footway(self):
        self.assertEqual(score_of(scene(low_kerb_2=laid("low_kerb_2", "S3", back=0.030))), 0.0)
        self.assertEqual(score_of(scene(low_kerb_2=laid("low_kerb_2", "S3", back=-0.030))), 0.0)

    def test_moving(self):
        a = laid("low_kerb_1", "S2")
        frames = [dict(DONE, low_kerb_1=shifted(a, dy=0.001 * i)) for i in range(WINDOW + 5)]  # 20 mm/s along
        r = run(frames)[0]
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["units"]["low_kerb_1"]["at_rest"])

    def test_a_missing_or_broken_unit_scores_zero(self):
        missing = {n: DONE[n] for n in NAMES[:3]}
        r = run([missing] * WINDOW)[0]
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])


class Stillness(unittest.TestCase):
    def test_needs_half_a_second(self):
        self.assertEqual(run([DONE] * (WINDOW - 1))[0].score, 0.0)
        self.assertEqual(run([DONE] * WINDOW)[0].score, 1.0)

    def test_small_jitter_is_still(self):
        frames = [dict(DONE, low_kerb_1=shifted(DONE["low_kerb_1"], dx=0.0009 * (-1) ** i)) for i in range(WINDOW * 3)]
        self.assertEqual(run(frames)[0].score, 1.0)

    def test_slow_creep_is_not(self):
        frames = [dict(DONE, transition_left=shifted(DONE["transition_left"], dy=-0.00025 * i))
                  for i in range(WINDOW * 2)]
        r = run(frames)[0]
        self.assertFalse(r.details["units"]["transition_left"]["at_rest"])
        self.assertEqual(r.score, 0.75)

    def test_a_turn_on_the_spot_is_motion(self):
        """Turning 5 degrees about its own middle moves its end 2.6 mm: not still."""
        frames = [DONE] * WINDOW
        for i in range(1, 6):
            frames.append(dict(DONE, low_kerb_2=laid("low_kerb_2", "S3", turn=1.0 * i)))
        r = run(frames)[0]
        self.assertFalse(r.details["units"]["low_kerb_2"]["at_rest"])
        self.assertEqual(r.score, 0.75)


class BrokenStatesAndResets(unittest.TestCase):
    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        hist = {n: [p.copy() for p in t.hist[n]] for n in NAMES}
        for bad_value in (("origin", np.array([np.nan, 0.0, 0.76])), ("quat_xyzw", np.zeros(4)),
                          ("pos", np.array([np.inf, 0, 0])), ("origin", None)):
            bad = SimpleNamespace(**vars(DONE["transition_right"]))
            setattr(bad, *bad_value)
            r = t.update(dict(DONE, transition_right=bad))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.update({}).score, 0.0)
        self.assertEqual(t.bad_steps, 5)
        for n in NAMES:
            self.assertEqual(len(hist[n]), len(t.hist[n]))
            self.assertTrue(all(np.allclose(a, b) for a, b in zip(hist[n], t.hist[n])))
        self.assertEqual(t.update(DONE).score, 1.0)  # the record carried on: still 1.0 at once

    def test_teleport_back_to_start_resets(self):
        r, t = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.update(DONE).score, 0.0)  # a fresh window: not still yet

    def test_jittered_reset_counts(self):
        r, t = run([DONE] * WINDOW)
        jit = {n: shifted(START[n], dx=0.02, dy=-0.025) for n in NAMES}
        t.update(jit)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        frames = [DONE] * WINDOW + glide(DONE, START, 40)
        r, t = run(frames)
        self.assertEqual(t.steps, WINDOW + 40)
        self.assertEqual(r.score, 0.0)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        r, t = run([DONE] * WINDOW)
        t.update({n: shifted(DONE[n], dx=-0.05) for n in NAMES})
        self.assertEqual(t.steps, WINDOW + 1)

    def test_explicit_reset(self):
        r, t = run([DONE] * WINDOW)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(DONE).score, 0.0)


class Exploits(unittest.TestCase):
    def test_taking_the_nearer_transition(self):
        """Each arm lays the transition that starts on its own side at its own end, turned whichever way puts its
        tall end at the full kerb: both faces end up to the footway, 0 for both ends."""
        sc = dict(DONE, transition_right=laid("transition_right", "S1", reverse=True),
                  transition_left=laid("transition_left", "S4", reverse=True))
        d = details_of(sc)
        self.assertFalse(d["slots"]["S1"]["right"] or d["slots"]["S4"]["right"])
        self.assertEqual(d["units"]["transition_right"]["rises_toward"], "+y")
        self.assertFalse(d["units"]["transition_right"]["face_to_road"])

    def test_butted_together_without_joints(self):
        """Laid from the left full kerb with no joints, the units drift 6 mm a slot: only the first two count."""
        y0 = L.GAP_HALF - L.UNIT_LEN / 2
        sc = {}
        for i, k in enumerate(SLOTS):
            n = RIGHT_UNIT[k]
            sc[n] = laid(n, k, along=(y0 - i * L.UNIT_LEN) - L.SLOTS[k])
        self.assertEqual(score_of(sc), 0.25)  # S1 6 mm off (counts), S2 12 mm, S3 18 mm, S4 24 mm

    def test_standing_units_on_end_in_the_slots(self):
        sc = {}
        for k in SLOTS:
            n = RIGHT_UNIT[k]
            sc[n] = laid(n, k, tilt=90.0, tilt_axis="y")
        self.assertEqual(score_of(sc), 0.0)

    def test_two_low_kerbs_stacked_in_one_slot_count_once(self):
        under = laid("low_kerb_1", "S2")
        over = laid("low_kerb_2", "S2", low=L.TOP + L.LOW_H)
        self.assertEqual(score_of(dict(DONE, low_kerb_1=under, low_kerb_2=over)), 0.75)

    def test_a_unit_off_the_table_changes_nothing(self):
        gone = unit("low_kerb_2", 0.3, 0.1, 0.0, low=0.30)
        self.assertEqual(score_of(dict(DONE, low_kerb_2=gone)), 0.75)

    def test_held_still_just_above(self):
        """The scorer cannot see the grippers: a unit held still within 3 mm of the bed counts as lying there
        (fidelity says so); 3.5 mm up it does not."""
        self.assertEqual(score_of(dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", low=L.TOP + 0.002))), 1.0)
        self.assertEqual(score_of(dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", low=L.TOP + 0.0035))), 0.75)

    def test_upside_down_transition_is_not_the_mirror(self):
        """Turned over, a transition is the mirror of itself; it would rise toward the other end with its face to
        the road, but it is not lying on its base: nothing."""
        flipped = laid("transition_right", "S1", reverse=True, tilt=180.0, tilt_axis="x", low=L.TOP + 0.022)
        f = unit_facts("transition_right", flipped)
        self.assertGreater(f["high_y"], 0.99)
        self.assertTrue(f["face_to_road"])
        self.assertEqual(score_of(scene(transition_right=flipped)), 0.0)


class OneArm(unittest.TestCase):
    def test_each_arm_alone_with_the_demo_moves(self):
        left = scene(transition_left=DONE["transition_left"], low_kerb_1=DONE["low_kerb_1"])
        right = scene(low_kerb_2=DONE["low_kerb_2"], transition_right=DONE["transition_right"])
        self.assertEqual(score_of(left), 0.5)
        self.assertEqual(score_of(right), 0.5)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir() and (d / "yamkin.py").is_file():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline tools (tools/offline: yam_stub, yamkin)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        d = _offline_dir()
        sys.path.insert(0, str(d / "yam_stub"))
        sys.path.insert(0, str(d))
        cls.task = importlib.import_module("task")

    def _times(self, wps):
        """(start, end, waypoint) for each waypoint of a plan."""
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, ["road", "crossover_paving", "full_kerb_left", "full_kerb_left_face",
                                 "full_kerb_right", "full_kerb_right_face", *NAMES])
        self.assertEqual(t.SCORE_OVER, "end")
        for o in (t.ROAD, t.PAVING, *t.FULL_KERBS):
            self.assertIsNone(o.mass)  # fixed
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)
        backs = [o for o in t.FULL_KERBS if not o.name.endswith("face")]
        faces = [o for o in t.FULL_KERBS if o.name.endswith("face")]
        for o in backs:
            self.assertAlmostEqual(o.pos[0] - o.size[0] / 2, L.BACK_X)
            self.assertAlmostEqual(o.size[2], L.FULL_H)
            self.assertAlmostEqual(abs(o.pos[1]) - o.size[1] / 2, L.GAP_HALF)
            self.assertAlmostEqual(o.size[1], L.FULL_LEN)
        for o in faces:
            self.assertAlmostEqual(o.pos[0] + o.size[0] / 2, L.FACE_X)
            self.assertAlmostEqual(o.size[2], L.FULL_H - L.LIP_DROP)
        for o in t.UNITS:
            self.assertEqual(o.mass, L.LOW_MASS if L.KIND[o.name] == "low" else L.TRANSITION_MASS)
            self.assertLessEqual(o.mass, 0.3)
            x, y, heading = L.UNIT_START[o.name]
            self.assertTrue(np.allclose(o.pos, (x, y, L.TOP)))  # the frame's origin, on the bed
            self.assertAlmostEqual(o.yaw, heading)
            self.assertEqual(len(o.shapes), len(L.unit_shapes(o.name)))
            st = o.rest_state()
            self.assertTrue(valid(st))
            self.assertTrue(np.allclose(st.origin, START[o.name].origin))
            self.assertTrue(np.allclose(st.low, START[o.name].low) and np.allclose(st.high, START[o.name].high))
            com = o.center_of_mass()  # PhysX spreads the mass over the colliders by volume
            if L.KIND[o.name] == "low":
                self.assertLess(abs(com[0]), 1e-6)
            else:  # 6 mm toward the tall end of the transition, from the middle where it is gripped
                self.assertAlmostEqual(L.HIGH_END[o.name] * com[0], 0.006, delta=0.0005)
        for o in t.OBJECTS:
            # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
            self.assertLess(abs(o.pos[1]) + o.size[1] / 2, 0.45, o.name)

    def test_rest_state_scores_zero(self):
        states = {o.name: o.rest_state() for o in self.task.UNITS}
        self.assertEqual(score_of(states), 0.0)
        self.assertTrue(Tracker.at_start(states))

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(START))
                self.assertEqual(self.task.score(START), 0.0)
                self.task._TRACKER.reset()
                ok = False
                for _ in range(WINDOW):
                    ok = self.task.success(DONE)
                self.assertTrue(ok)
                self.assertEqual(self.task.score(DONE), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1 + WINDOW)
        last = rows[-1]
        for key in ("step", "score", "success", "valid_state", "slots_right", "zero", "slots", "units", "raw",
                    "mechkit_episode"):
            self.assertIn(key, last)
        row = last["units"]["transition_left"]
        for key in ("kind", "right_in", "bedded", "at_rest", "flat", "tilt_deg", "on_bed", "low_mm", "back_off_mm",
                    "in_line", "face_err_deg", "face_to_road", "rises_toward", "heading_deg", "nearest_slot",
                    "along_mm", "origin_mm"):
            self.assertIn(key, row)
        self.assertEqual(row["right_in"], "S1")
        self.assertEqual(row["back_off_mm"], [0.0, 0.0])
        self.assertEqual(row["along_mm"], 0.0)
        self.assertEqual(row["rises_toward"], "+y")
        self.assertEqual(row["heading_deg"], -90.0)
        self.assertEqual(last["units"]["low_kerb_2"]["rises_toward"], None)
        self.assertEqual(last["slots"]["S4"]["units"], ["transition_right"])
        self.assertEqual(last["slots"]["S2"]["wants"], "low")

    def test_score_hook_reads_the_last_step_not_the_best(self):
        self.task._TRACKER.reset()
        for _ in range(WINDOW):
            self.task.success(DONE)
        self.assertEqual(self.task.score(DONE), 1.0)
        lifted = dict(DONE, low_kerb_1=laid("low_kerb_1", "S2", low=L.TOP + 0.030))
        self.assertFalse(self.task.success(lifted))
        self.assertEqual(self.task.score(lifted), 0.75)

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(START))
        finally:
            os.environ.pop("TASK_TRACE", None)
        self.assertFalse((HERE / "no_such_dir").exists())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                bad = SimpleNamespace(**vars(START["low_kerb_2"]))
                bad.origin = np.array([float("nan"), 0.0, 0.76])
                self.assertFalse(self.task.success(dict(START, low_kerb_2=bad)))
                self.assertFalse(self.task.success({}))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertFalse(rows[0]["valid_state"])
        self.assertFalse(rows[1]["valid_state"])
        self.assertEqual(rows[1]["raw"]["low_kerb_1"], "unreadable")

    def test_a_new_mechkit_episode_resets_the_tracker(self):
        import mechkit as mk
        t = self.task
        t._TRACKER.reset()
        for _ in range(WINDOW):
            t.success(DONE)
        self.assertGreater(t._TRACKER.steps, 1)
        t._EPISODE["mechkit"] = mk.episode() - 1  # as if mechkit had seen a reset since the last step
        t.success(DONE)
        self.assertEqual(t._TRACKER.steps, 1)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_places_score_one(self):
        """Each unit carried from its start, turned by the plan's quarter turn and set where the plan sets it down
        scores 1.0 for the four; with one arm's moves only, 0.5."""
        t = self.task
        sc = dict(START)
        per_arm = {}
        for side in ("left", "right"):
            wps = t.demo()[side]
            one = dict(START)
            for name, slot in t.plan_pairs(side):
                _, _, h0 = L.UNIT_START[name]
                dx, dy = t.place_point(name, slot)
                turn = t.turn(name)
                self.assertAlmostEqual(abs(turn), math.pi / 2)
                self.assertIn((dx, dy), [w.pos[:2] for w in wps])
                self.assertIn(turn, [w.yaw for w in wps if w.pos[:2] == (dx, dy)])
                s = unit(name, dx, dy, math.degrees(h0 + turn))
                sc[name] = one[name] = s
            per_arm[side] = one
        r = run([sc] * WINDOW)[0]
        self.assertEqual(r.score, 1.0)
        for n, row in r.details["units"].items():
            self.assertEqual(row["back_off_mm"], [0.0, 0.0], n)
            self.assertEqual(row["along_mm"], 0.0, n)
        self.assertEqual(score_of(per_arm["left"]), 0.5)
        self.assertEqual(score_of(per_arm["right"]), 0.5)
        self.assertEqual({u for u, _ in t.plan_pairs("left")}, {"transition_left", "low_kerb_1"})
        self.assertEqual({s for _, s in t.plan_pairs("right")}, {"S3", "S4"})

    def test_each_turn_keeps_the_wrist_in_range(self):
        """Every unit is picked with the fingers across its depth at yaw 0 and turned a quarter turn: the left arm's
        turns are -90 degrees and the right arm's +90 (the turns that leave joint 6 far from its limits; the other
        way the left arm could not reach the kerb line)."""
        t = self.task
        for side, sign in (("left", -1.0), ("right", 1.0)):
            for name, _ in t.plan_pairs(side):
                self.assertAlmostEqual(t.turn(name), sign * math.pi / 2, places=9, msg=name)

    def test_arms_take_turns(self):
        """Only one arm is at the kerb line (x > 0.30) at a time, and while one reaches across the midline (its TCP
        on the other side of y = 0) the other is parked beside its own base. The run fits the run limit."""
        plan = self.task.demo()
        segs = {side: self._times(wps) for side, wps in plan.items()}

        def busy(side, test, either=True):
            out, prev = [], None
            for a, b, w in segs[side]:
                p0 = prev if prev is not None else w.pos
                if (either and test(p0)) or test(w.pos):
                    out.append((a, b))
                prev = w.pos
            return out

        def overlap(xs, ys):
            return [(a, b, c, d) for a, b in xs for c, d in ys if min(b, d) - max(a, c) > 1e-9]

        at_line = {s: busy(s, lambda p: p[0] > 0.30) for s in plan}
        self.assertEqual(overlap(at_line["left"], at_line["right"]), [])
        across = {s: busy(s, lambda p, k=self.task.SIDE[s]: k * p[1] < 0.0, either=False) for s in plan}
        self.assertTrue(across["left"] and across["right"])
        parked = {}
        for s in plan:
            spot = (self.task.PARK[0], self.task.SIDE[s] * self.task.PARK[1], self.task.PARK[2])
            parked[s] = [(a, b) for a, b, w in segs[s] if w.pos == spot]
            if segs[s][-1][2].pos == spot:  # after its plan an arm holds its last pose: parked to the end
                parked[s].append((segs[s][-1][1], math.inf))
        for s, other in (("left", "right"), ("right", "left")):
            for a, b in across[s]:
                covered = any(c <= a + 1e-9 and d >= b - 1e-9 for c, d in parked[other])
                self.assertTrue(covered, (s, a, b))
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * HZ, 2000)  # inside the run limit with the policy's 3 s hold

    def test_grips_and_clearances(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.GRIP, L.UNIT_DEPTH - 0.008)  # commanded 8 mm narrower
        self.assertGreaterEqual(-2 * t.OPEN - L.UNIT_DEPTH, 0.016)  # at least 8 mm each side when open
        tips = t.GRIP_Z - t.PAD - L.TOP
        self.assertAlmostEqual(tips, 0.002)  # pad tips 2 mm above the bed
        self.assertGreaterEqual(L.LOW_H - L.LIP_DROP - tips, 0.006 - 1e-9)  # 6 mm of a low kerb's face strip
        self.assertGreaterEqual(L.LOW_H - tips, 0.010 - 1e-9)  # ... and 10 mm of its back
        self.assertGreaterEqual(t.GRIP_Z - L.TOP - 0.015, -1e-9)  # TCP at least 15 mm up
        # The gripper's palm (25 mm above the tool point) clears the tallest unit it holds.
        self.assertGreater(t.GRIP_Z + 0.025 - (L.TOP + L.FULL_H), 0.008)
        # A carried unit's underside passes 20 mm or more over every kerb (the full kerbs are the tallest).
        under = t.HOVER - (t.GRIP_Z - L.TOP)
        self.assertGreaterEqual(under - (L.TOP + L.FULL_H), 0.020)
        # Released 2 mm above the bed, the open pads (12 mm off each face, 8 mm thick) pass over the paving and the
        # road strip with their tips 3 mm above them, and the back pad clears the paving's edge by >= 2 mm.
        release_tips = t.GRIP_Z + t.DROP - t.PAD
        self.assertGreaterEqual(release_tips - (L.TOP + max(L.PAVING_H, L.ROAD_H)), 0.003 - 1e-9)
        gap = (-2 * t.OPEN - L.UNIT_DEPTH) / 2
        self.assertGreater(L.FACE_X + gap, L.ROAD_X0)  # the front pad stands over the road strip, not its edge
        # The two transitions in the stock area leave room for the open fingers between them.
        y_in = L.UNIT_START["transition_right"][1] - L.UNIT_DEPTH / 2
        self.assertGreater(2 * y_in - 2 * (gap + 0.008), 0.03)

    def test_slow_final_descents_and_long_opens(self):
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))
                    self.assertLessEqual(abs(wps[i - 1].pos[2] - w.pos[2]), 0.030 + 1e-9)
                if w.label.startswith(("grip", "let go")):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))
                    self.assertEqual(wps[i - 1].pos, w.pos)  # opened and closed in place

    def test_success_hold_outlasts_every_release(self):
        n = 0
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label == "slowly set the unit down":
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertEqual((settle.label, let_go.label), ("settle", "let go of the unit"))
                    self.assertLessEqual(settle.duration + let_go.duration + L.REST_WINDOW_S,
                                         self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_task_json_numbers_match_the_layout(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["setup"] + " " + spec["scoring"]
        for needle in ("60 mm", "25 mm", "12 mm", "21", "30 mm", "6 mm", "270 mm", "120 mm", "x 37.25 cm",
                       "x 39.75 cm", "y +9.9", "y +3.3", "y -3.3", "y -9.9", "5 degrees", "3 mm", "5 mm", "8 mm",
                       "30 degrees", "0.5 s", "2 mm", "21 g", "37 g", "(x 18.0 cm, y +24.0 cm)",
                       "(x 16.0 cm, y -5.0 cm)", "(x 16.0 cm, y +5.0 cm)", "(x 18.0 cm, y -24.0 cm)"):
            self.assertIn(needle, text, needle)
        self.assertIn("2,000", spec["run_limit"])

    def test_cross_reach(self):
        """Straight down (tool point 13 to 30 mm up, pad tips 0 to 17 mm; headings every 15 degrees), neither arm
        can grip a unit bedded in the far end slot in any pose the rule credits (middle within 8 mm of the slot, back
        within 5 mm of the line) with 5 mm of pad on it, nor the far low kerb where it starts: the README's 'One arm
        alone'."""
        sys.path.insert(0, str(_offline_dir()))
        import yamkin as K
        top = K.TABLE_TOP["station"]
        for side, k in (("left", 1.0), ("right", -1.0)):
            base = K.BASES["station"][side]
            far_slot = "S4" if side == "left" else "S1"
            inner_end = L.SLOTS[far_slot] + k * (L.ALONG_TOL + L.UNIT_LEN / 2)  # the end nearest this arm
            tcp_y = inner_end + k * (0.010 - 0.005)  # pads 20 mm wide, centred 5 mm outside the end: 5 mm on it
            far_low = "low_kerb_2" if side == "left" else "low_kerb_1"
            lx, ly, _ = L.UNIT_START[far_low]
            low_y = ly + k * (L.UNIT_DEPTH / 2 + 0.005)  # fingers along x: 5 mm of pad over its near side
            self.assertAlmostEqual(abs(tcp_y), 0.056)
            for h in (0.013, 0.015, 0.022, 0.030):
                for yaw_deg in range(-180, 180, 15):
                    yaw = math.radians(yaw_deg)
                    for x in (L.LINE_X - L.LINE_TOL, L.LINE_X, L.LINE_X + L.LINE_TOL):
                        ok, q, _, _ = K.reachable((x, tcp_y, top + h), yaw, base)
                        if ok:
                            ok = K.pose_clear(q, base, top)["ok"]
                        self.assertFalse(ok, (side, x, tcp_y, h, yaw_deg))
                    for x in (lx - 0.025, lx, lx + 0.025):
                        ok, q, _, _ = K.reachable((x, low_y, top + h), yaw, base)
                        if ok:
                            ok = K.pose_clear(q, base, top)["ok"]
                        self.assertFalse(ok, (side, x, low_y, h, yaw_deg))
            # ... while each arm reaches its own two slots and both transitions in the stock area.
            for slot in (("S1", "S2") if side == "left" else ("S3", "S4")):
                self.assertTrue(K.reachable((L.LINE_X, L.SLOTS[slot], top + 0.015), -k * math.pi / 2, base)[0])
            for n in ("transition_left", "transition_right"):
                x, y, _ = L.UNIT_START[n]
                self.assertTrue(K.reachable((x, y, top + 0.015), 0.0, base)[0])


if __name__ == "__main__":
    unittest.main()
