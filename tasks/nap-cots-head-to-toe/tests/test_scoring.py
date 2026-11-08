"""Offline tests for scoring.py and task.py: synthetic object states, no simulator.
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
    cot_facts,
    end_marked,
    heading_err_deg,
    long_axis_xy,
    outline,
    poly_distance,
    rotation,
    rug_margin,
    tilt_deg,
    top_height,
    valid,
)

HZ = L.CONTROL_HZ
X0 = L.COT_START["cot_middle"][0]  # 0.2775: every cot's centre x at the start
FAR = L.COT_HALF_LEN - L.END_INSET  # +57.5 mm along the cot: a pillow 25 mm in from the far end
NEAR = -FAR
OUT = 0.194  # where the demo sets the outer cots
WINDOW = int(round(L.REST_WINDOW_S * HZ)) + 1  # 11 samples: 0.5 s of stillness


def quat(yaw_deg=0.0, tilt_deg_=0.0, tilt_axis="x"):
    """Turned yaw about the vertical after tilting about the body's x (or y) axis, as (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qt = np.array([math.sin(b), 0.0, 0.0, math.cos(b)]) if tilt_axis == "x" else \
        np.array([0.0, math.sin(b), 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qt
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def box(size, x, y, low_z, yaw=0.0, tilt=0.0, tilt_axis="x", vel=(0, 0, 0), spin=(0, 0, 0)):
    """A box's state as the simulator reports it: centre (x, y), lowest point at low_z, its bounding box."""
    q = quat(yaw, tilt, tilt_axis)
    half = np.array(size) / 2
    corners = np.array([rotation(q) @ (half * np.array([sx, sy, sz]))
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = np.array([x, y, low_z - corners[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c + corners.min(axis=0), high=c + corners.max(axis=0))


def cot(y, x=X0, low=None, **kw):
    return box(L.COT_SIZE, x, y, L.RUG_TOP if low is None else low, **kw)


def pillow(x, y, low=None, **kw):
    return box(L.PILLOW_SIZE, x, y, L.COT_TOP if low is None else low, **kw)


ALONG = {"far": FAR, "near": NEAR, "mid": 0.0}


def room(cl=0.120, cm=0.0, cr=-0.120, pl="far", pm="far", pr="far", **over):
    """Cots at y cl, cm, cr (x 0.2775, flat), a pillow on each at the named end ('far', 'near', 'mid' or an offset
    along the cot in m); any object can be replaced by keyword."""
    out = {}
    for name, y, end in (("left", cl, pl), ("middle", cm, pm), ("right", cr, pr)):
        out["cot_" + name] = cot(y)
        along = ALONG[end] if isinstance(end, str) else end
        out["pillow_" + name] = pillow(X0 + along, y)
    out.update(over)
    return out


START = room()
DONE = room(cl=OUT, cr=-OUT, pl="near", pr="near")  # the demo's end: near, far, near


def run(frames, t=None):
    """Feed the frames to a tracker (a fresh one unless given); return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


def score_of(scene):
    """The score of a scene held still for 0.5 s."""
    return run([scene] * WINDOW)[0].score


def glide(a, b, n):
    """n frames moving every object from scene a to scene b in equal steps."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        frame = {}
        for k in NAMES:
            sa, sb = a[k], b[k]
            frame[k] = SimpleNamespace(pos=(1 - f) * sa.pos + f * sb.pos, quat_xyzw=sb.quat_xyzw,
                                       lin_vel=(sb.pos - sa.pos) * HZ / n, ang_vel=np.zeros(3),
                                       low=(1 - f) * sa.low + f * sb.low, high=(1 - f) * sa.high + f * sb.high)
        out.append(frame)
    return out


class Geometry(unittest.TestCase):
    def test_angles(self):
        self.assertAlmostEqual(tilt_deg(rotation(quat(30.0, 6.0))), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(rotation(quat(0.0, 176.0))), 4.0, places=4)  # upside down is flat
        self.assertAlmostEqual(heading_err_deg(rotation(quat(10.0))), 10.0, places=4)
        self.assertAlmostEqual(heading_err_deg(rotation(quat(-10.0))), 10.0, places=4)
        self.assertAlmostEqual(heading_err_deg(rotation(quat(170.0))), 10.0, places=4)  # end for end: same cot
        self.assertAlmostEqual(heading_err_deg(rotation(quat(90.0))), 90.0, places=4)
        self.assertEqual(heading_err_deg(rotation(quat(0.0, 90.0, "y"))), 90.0)  # standing on its end: no heading
        u = long_axis_xy(rotation(quat(170.0)))
        self.assertGreater(u[0], 0.9)  # always pointed toward the far end

    def test_gap_is_the_clear_floor_between_outlines(self):
        a = outline(cot(0.0), L.COT_SIZE)
        self.assertAlmostEqual(poly_distance(a, outline(cot(0.183), L.COT_SIZE)), 0.114, places=9)
        self.assertAlmostEqual(poly_distance(a, outline(cot(0.120), L.COT_SIZE)), 0.051, places=9)
        self.assertEqual(poly_distance(a, outline(cot(0.050), L.COT_SIZE)), 0.0)  # overlapping
        self.assertEqual(poly_distance(a, outline(cot(0.069), L.COT_SIZE)), 0.0)  # touching
        # Staggered along x by the most the rug allows (75 mm): still the gap across y.
        self.assertAlmostEqual(poly_distance(a, outline(cot(0.183, x=X0 + 0.075), L.COT_SIZE)), 0.114, places=9)
        # Turned 10 degrees: the nearest corner counts, so the gap shrinks.
        turned = outline(cot(0.183, yaw=10.0), L.COT_SIZE)
        self.assertLess(poly_distance(a, turned), 0.114 - 0.010)

    def test_rug_margin(self):
        edge_y = L.RUG_Y1 - L.COT_SIZE[1] / 2  # 0.2055: the cot's outer side on the rug's edge
        self.assertAlmostEqual(rug_margin(outline(cot(edge_y), L.COT_SIZE)), 0.0, places=9)
        self.assertLess(rug_margin(outline(cot(edge_y + 0.001), L.COT_SIZE)), 0.0)
        self.assertAlmostEqual(rug_margin(outline(cot(OUT), L.COT_SIZE)), 0.0115, places=9)
        self.assertAlmostEqual(rug_margin(outline(cot(0.0), L.COT_SIZE)), 0.035, places=9)  # near edge 35 mm, far 40

    def test_end_bands(self):
        reach = L.COT_HALF_LEN - L.END_BAND  # 32.5 mm from the cot's centre
        self.assertEqual(end_marked(reach + 0.0001), "far")
        self.assertEqual(end_marked(reach - 0.0001), "none")
        self.assertEqual(end_marked(-reach - 0.0001), "near")
        self.assertEqual(end_marked(-reach + 0.0001), "none")
        self.assertEqual(end_marked(0.0), "none")
        self.assertEqual(end_marked(FAR), "far")
        self.assertEqual(end_marked(NEAR), "near")

    def test_valid(self):
        self.assertTrue(valid(cot(0.0)))
        self.assertFalse(valid(None))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))  # fields missing
        bad = cot(0.0)
        bad.pos = np.array([float("nan"), 0.0, 0.8])
        self.assertFalse(valid(bad))
        bad = cot(0.0)
        bad.lin_vel = np.array([0.0, float("inf"), 0.0])
        self.assertFalse(valid(bad))
        bad = cot(0.0)
        bad.quat_xyzw = np.zeros(4)
        self.assertFalse(valid(bad))
        bad = cot(0.0)
        bad.pos = np.zeros(2)
        self.assertFalse(valid(bad))

    def test_room_arithmetic(self):
        need = 3 * L.COT_SIZE[1] + 2 * L.GAP_MIN
        self.assertAlmostEqual(need, 0.435)
        self.assertLess(need, L.RUG_SIZE[1])
        self.assertAlmostEqual(OUT - L.COT_SIZE[1] - L.GAP_MIN, 0.011)  # the demo's gaps: 125 mm, 11 mm spare
        self.assertAlmostEqual(L.COT_START["cot_left"][1] - L.COT_SIZE[1], 0.051)  # 51 mm at the start
        # The cots fit on the rug along x with room to stagger 75 mm at most.
        self.assertAlmostEqual(L.RUG_SIZE[0] - L.COT_SIZE[0], 0.075)
        self.assertGreater(L.COT_START["cot_middle"][0] - L.COT_HALF_LEN, L.RUG_X0)
        self.assertLess(L.COT_START["cot_middle"][0] + L.COT_HALF_LEN, L.RUG_X1)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([START] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["zero"], "")
        self.assertEqual([p["gap_mm"] for p in r.details["pairs"]], [51.0, 51.0])
        self.assertEqual([p["ends"] for p in r.details["pairs"]], [["far", "far"], ["far", "far"]])

    def test_the_demo_end_scores_one(self):
        r, _ = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual([p["gap_mm"] for p in r.details["pairs"]], [125.0, 125.0])
        self.assertEqual(r.details["order"], ["cot_left", "cot_middle", "cot_right"])

    def test_the_other_solution_scores_one(self):
        """Both cots spread and only the middle pillow moved to the near end: far, near, far."""
        self.assertEqual(score_of(room(cl=OUT, cr=-OUT, pm="near")), 1.0)

    def test_every_outcome(self):
        cases = [
            (room(), 0.0),  # the start
            (room(pm="near"), 0.5),  # pillows alternate, cots not spread
            (room(pl="near", pr="near"), 0.5),
            (room(pl="near"), 0.25),  # one outer pillow moved only
            (room(cl=OUT), 0.25),  # one outer cot spread, pillows all far
            (room(cr=-OUT), 0.25),
            (room(cl=OUT, cr=-OUT), 0.5),  # both spread, no pillow moved
            (room(cl=OUT, pr="near"), 0.5),  # one pair spread, the other pair alternates
            (room(cl=OUT, pm="near"), 0.75),  # one spread, both pairs alternate
            (room(cl=OUT, cr=-OUT, pl="near"), 0.75),  # both spread, one pair alternates
            (room(cl=OUT, cr=-OUT, pm="near"), 1.0),
            (room(cl=OUT, cr=-OUT, pl="near", pr="near"), 1.0),
            (room(cl=OUT, cr=-OUT, pl="near", pm="near", pr="near"), 0.5),  # all near: no alternation
            (room(cl=OUT, cr=-OUT, pm="mid"), 0.5),  # a pillow mid-cot: neither of its pairs alternates
            (room(cl=OUT, cr=-OUT, pl="near", pm="mid", pr="near"), 0.5),
            (room(cl=OUT, cr=-OUT, pl="mid", pr="near"), 0.75),  # the left pair loses its head-to-toe credit
        ]
        for scene, want in cases:
            self.assertEqual(score_of(scene), want)
        self.assertEqual({w for _, w in cases}, {0.0, 0.25, 0.5, 0.75, 1.0})  # every tier turns up

    def test_every_combination_of_ends_and_spreads(self):
        """All 3^3 pillow ends x 4 spreads: the score is always 0.25 x (pairs spread + pairs alternating)."""
        for (pl, pm, pr), (sl, sr) in itertools.product(itertools.product(("far", "near", "mid"), repeat=3),
                                                        itertools.product((False, True), repeat=2)):
            scene = room(cl=OUT if sl else 0.12, cr=-OUT if sr else -0.12, pl=pl, pm=pm, pr=pr)
            alt = sum(a != "mid" and b != "mid" and a != b for a, b in ((pl, pm), (pm, pr)))
            self.assertEqual(score_of(scene), 0.25 * (sl + sr + alt), (pl, pm, pr, sl, sr))

    def test_gap_threshold(self):
        y_ok = L.COT_SIZE[1] + L.GAP_MIN + 0.0001  # 114.1 mm of floor
        y_short = L.COT_SIZE[1] + L.GAP_MIN - 0.0001  # 113.9 mm
        self.assertEqual(score_of(room(cl=y_ok)), 0.25)
        self.assertEqual(score_of(room(cl=y_short)), 0.0)

    def test_a_turned_cot_is_measured_at_its_nearest_corner(self):
        """Turned 12 degrees (allowed), the left cot's centre is 189 mm from the middle one's, which would be 120 mm of
        floor if it were square; its near corner leaves only 104 mm, so that pair earns no gap credit."""
        a = math.radians(12.0)
        scene = room(cr=-OUT, pm="near", cot_left=cot(0.189, yaw=12.0),
                     pillow_left=pillow(X0 + FAR * math.cos(a), 0.189 + FAR * math.sin(a)))
        r, _ = run([scene] * WINDOW)
        self.assertEqual(r.details["zero"], "")
        self.assertAlmostEqual(r.details["pairs"][0]["gap_mm"], 103.6, delta=0.2)
        self.assertFalse(r.details["pairs"][0]["gap_ok"])
        self.assertEqual(r.score, 0.75)

    def test_end_threshold(self):
        reach = L.COT_HALF_LEN - L.END_BAND
        self.assertEqual(score_of(room(pm=-(reach + 0.0005))), 0.5)  # 49.5 mm from the near end: marks it
        self.assertEqual(score_of(room(pm=-(reach - 0.0005))), 0.0)  # 50.5 mm: marks no end
        # Overhanging the near end with its centre still over the cot: still on it, still the near end.
        self.assertEqual(score_of(room(pm=-(L.COT_HALF_LEN - 0.001))), 0.5)

    def test_cots_and_pillows_are_interchangeable(self):
        swapped = dict(DONE)
        swapped["cot_left"], swapped["cot_right"] = DONE["cot_right"], DONE["cot_left"]
        swapped["pillow_left"], swapped["pillow_middle"] = DONE["pillow_middle"], DONE["pillow_left"]
        r, _ = run([swapped] * WINDOW)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["order"], ["cot_right", "cot_middle", "cot_left"])

    def test_turned_end_for_end_with_its_pillow(self):
        """A cot turned end for end (180 degrees) carrying its pillow to the near end is a legal way to alternate."""
        scene = room(cl=OUT, cr=-OUT, pm="near", cot_middle=cot(0.0, yaw=180.0))
        self.assertEqual(score_of(scene), 1.0)

    def test_small_turns_and_props_within_tolerance(self):
        self.assertEqual(score_of(room(cl=OUT, cr=-OUT, pm="near", cot_middle=cot(0.0, yaw=4.0))), 1.0)
        for tilt, want in ((4.5, 1.0), (5.5, 0.0)):  # one long side propped up: within 5 degrees of level counts
            tilted = cot(0.0, tilt=tilt)
            top = top_height(cot_facts(tilted), (X0 + NEAR, 0.0))  # the pillow rests on the tilted top
            scene = room(cl=OUT, cr=-OUT, pm="near", cot_middle=tilted, pillow_middle=pillow(X0 + NEAR, 0.0, low=top))
            self.assertEqual(score_of(scene), want, tilt)


class ZeroConditions(unittest.TestCase):
    """Each zero condition, applied to the 1.0 end state."""

    def zero(self, scene, reason):
        r, _ = run([scene] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertIn(reason, r.details["zero"])

    def test_cot_tipped_past_five_degrees(self):
        self.zero(dict(DONE, cot_middle=cot(0.0, tilt=5.5)), "cot not set down")

    def test_cot_on_its_side(self):
        self.zero(dict(DONE, cot_middle=cot(0.0, tilt=90.0)), "cot not set down")

    def test_cot_lifted_or_stacked(self):
        self.zero(dict(DONE, cot_middle=cot(0.0, low=L.RUG_TOP + 0.0035)), "cot not set down")
        self.assertEqual(score_of(dict(DONE, cot_middle=cot(0.0, low=L.RUG_TOP + 0.0025))), 1.0)  # held 2.5 mm up
        self.zero(dict(DONE, cot_left=cot(0.0, low=L.COT_TOP, x=X0)), "cot not set down")  # on the middle cot
        self.zero(dict(DONE, cot_left=cot(OUT, low=L.RUG_TOP + L.PILLOW_SIZE[2])), "cot not set down")  # on a pillow

    def test_cot_turned(self):
        self.zero(dict(DONE, cot_left=cot(OUT, yaw=16.0)), "cot not set down")
        self.zero(dict(DONE, cot_left=cot(OUT, yaw=90.0)), "cot not set down")
        a = math.radians(14.0)
        r, _ = run([dict(DONE, cot_left=cot(0.170, yaw=14.0),
                         pillow_left=pillow(X0 + NEAR * math.cos(a), 0.170 + NEAR * math.sin(a)))] * WINDOW)
        self.assertEqual(r.details["zero"], "")  # 14 degrees still counts as pointing forward
        self.assertTrue(r.details["cots"]["cot_left"]["set_down"])

    def test_cot_turned_while_still_on_the_rug(self):
        """Turned 16 or 20 degrees in the middle of the rug (so the rug-edge rule does not catch it)."""
        for yaw in (16.0, 20.0, -20.0):
            r, _ = run([dict(DONE, cot_middle=cot(0.0, yaw=yaw))] * WINDOW)
            self.assertTrue(r.details["cots"]["cot_middle"]["on_rug"])
            self.assertFalse(r.details["cots"]["cot_middle"]["set_down"])
            self.assertEqual(r.score, 0.0)
            self.assertIn("cot not set down", r.details["zero"])

    def test_cot_sunk_into_the_rug(self):
        """A simulator glitch that sinks a cot 4 mm into the rug is not 'lying flat on it' either."""
        self.zero(dict(DONE, cot_middle=cot(0.0, low=L.RUG_TOP - 0.004)), "cot not set down")

    def test_cot_off_the_rug(self):
        self.zero(dict(DONE, cot_left=cot(0.207)), "cot not set down")  # 1.5 mm over the side edge
        self.zero(dict(DONE, cot_right=cot(-0.207)), "cot not set down")  # ... the other side edge
        self.zero(dict(DONE, cot_middle=cot(0.0, x=L.RUG_X0 + L.COT_HALF_LEN - 0.001)), "cot not set down")  # near
        self.zero(dict(DONE, cot_middle=cot(0.0, x=L.RUG_X1 - L.COT_HALF_LEN + 0.001)), "cot not set down")  # far
        self.zero(dict(DONE, cot_left=cot(0.30, low=L.TOP)), "cot not set down")  # on the table beside the rug

    def test_pillow_not_on_a_cot(self):
        self.zero(dict(DONE, pillow_left=pillow(0.383, 0.13, low=L.RUG_TOP)), "pillow not on a cot")  # on the rug
        self.zero(dict(DONE, pillow_left=pillow(0.25, 0.30, low=L.TOP)), "pillow not on a cot")  # on the table
        self.zero(dict(DONE, pillow_left=pillow(X0 + NEAR, OUT, low=L.COT_TOP + 0.0035)), "pillow not on a cot")
        self.assertEqual(score_of(dict(DONE, pillow_left=pillow(X0 + NEAR, OUT, low=L.COT_TOP + 0.0025))), 1.0)
        self.zero(dict(DONE, pillow_left=pillow(X0 + NEAR, 0.10, low=L.RUG_TOP)), "pillow not on a cot")  # in the gap

    def test_pillow_on_a_pillow(self):
        on_top = pillow(X0 + FAR, 0.0, low=L.COT_TOP + L.PILLOW_SIZE[2])
        self.zero(dict(DONE, pillow_left=on_top), "pillow not on a cot")

    def test_two_pillows_on_one_cot(self):
        self.zero(dict(DONE, pillow_left=pillow(X0 + NEAR, 0.0)), "two pillows on one cot")

    def test_anything_moving(self):
        frames = [DONE] * WINDOW
        r, t = run(frames)
        self.assertEqual(r.score, 1.0)
        nudged = dict(DONE, cot_left=cot(OUT + 0.003))  # a cot shoved 3 mm in one step: not at rest for 0.5 s
        r = t.update(nudged)
        self.assertEqual(r.score, 0.0)
        self.assertIn("cot not set down", r.details["zero"])
        for _ in range(WINDOW - 2):
            r = t.update(nudged)
        self.assertEqual(r.score, 0.0)
        r = t.update(nudged)
        self.assertEqual(r.score, 1.0)  # still again for 0.5 s
        moved = dict(nudged, pillow_middle=pillow(X0 + FAR - 0.003, 0.0))
        self.assertIn("pillow not on a cot", t.update(moved).details["zero"])

    def test_a_missing_object(self):
        for n in NAMES:
            scene = dict(DONE)
            del scene[n]
            r, _ = run([scene] * WINDOW)
            self.assertEqual(r.score, 0.0, n)
            self.assertFalse(r.details["valid_state"])


class Stillness(unittest.TestCase):
    def test_needs_half_a_second(self):
        t = Tracker(HZ)
        for i in range(WINDOW - 1):
            self.assertEqual(t.update(DONE).score, 0.0, i)
        self.assertEqual(t.update(DONE).score, 1.0)

    def test_small_jitter_is_still(self):
        t = Tracker(HZ)
        r = None
        for i in range(30):
            d = 0.0008 * (-1) ** i  # 1.6 mm peak to peak
            r = t.update(dict(DONE, cot_left=cot(OUT + d), pillow_left=pillow(X0 + NEAR, OUT + d)))
        self.assertEqual(r.score, 1.0)

    def test_slow_creep_is_not(self):
        t = Tracker(HZ)
        r = None
        for i in range(30):
            d = 0.0003 * i  # 6 mm/s: 3 mm in 0.5 s
            r = t.update(dict(DONE, cot_left=cot(OUT + 0.005 - d), pillow_left=pillow(X0 + NEAR, OUT + 0.005 - d)))
        self.assertEqual(r.score, 0.0)

    def test_a_turn_on_the_spot_is_motion(self):
        t = Tracker(HZ)
        r = None
        for i in range(30):
            r = t.update(dict(DONE, cot_middle=cot(0.0, yaw=0.5 * i), pillow_middle=pillow(X0 + FAR, 0.0)))
        self.assertEqual(r.score, 0.0)  # its end face moves 0.7 mm a step


class BrokenStatesAndResets(unittest.TestCase):
    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        hist = {n: list(t.hist[n]) for n in NAMES}
        for bad in (float("nan"), float("inf")):
            s = cot(OUT)
            s.pos = np.array([bad, OUT, 0.76])
            r = t.update(dict(DONE, cot_left=s))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
        zq = pillow(X0 + NEAR, OUT)
        zq.quat_xyzw = np.zeros(4)
        self.assertEqual(t.update(dict(DONE, pillow_left=zq)).score, 0.0)
        self.assertEqual(t.update({}).score, 0.0)
        for n in NAMES:
            self.assertEqual(len(t.hist[n]), len(hist[n]))
            self.assertTrue(all(np.allclose(a, b) for a, b in zip(t.hist[n], hist[n])))
        self.assertEqual(t.update(DONE).score, 1.0)  # the record went on as before

    def test_teleport_back_to_start_resets(self):
        r, t = run([DONE] * 30)
        self.assertEqual(r.score, 1.0)
        steps_before = t.steps
        r = t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertLess(t.steps, steps_before)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(t.hist["cot_left"]), 1)

    def test_jittered_reset_counts(self):
        _, t = run([DONE] * 20)
        jittered = room(cl=0.12 + 0.02, cm=-0.015, cr=-0.12 + 0.01)
        t.update(jittered)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        _, t = run([DONE] * 20)
        for f in glide(DONE, START, 40):
            t.update(f)
        self.assertEqual(t.steps, 60)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = run([DONE] * 20)
        t.update(room(cl=0.0, cm=0.12, cr=-0.12))  # a big jump, but not onto the starting marks
        self.assertEqual(t.steps, 21)

    def test_explicit_reset(self):
        _, t = run([DONE] * 20)
        t.reset()
        self.assertEqual((t.steps, t.prev), (0, None))
        self.assertEqual(t.update(DONE).score, 0.0)


class Exploits(unittest.TestCase):
    def test_stacking_cots_to_make_room(self):
        stacked = dict(DONE, cot_middle=cot(OUT, low=L.COT_TOP), pillow_middle=pillow(X0 + FAR, OUT,
                                                                                         low=L.COT_TOP + 0.020))
        self.assertEqual(score_of(stacked), 0.0)

    def test_staggering_along_x_does_not_make_a_gap(self):
        """The rug leaves room to stagger the cots only 75 mm along x, while they are 165 mm long: their outlines
        still face each other across 51 mm."""
        lo, hi = L.RUG_X0 + L.COT_HALF_LEN, L.RUG_X1 - L.COT_HALF_LEN
        scene = room(pm="near", cot_left=cot(0.12, x=lo), cot_middle=cot(0.0, x=hi), cot_right=cot(-0.12, x=lo),
                     pillow_left=pillow(lo + FAR, 0.12), pillow_middle=pillow(hi + NEAR, 0.0),
                     pillow_right=pillow(lo + FAR, -0.12))
        r, _ = run([scene] * WINDOW)
        self.assertEqual([p["gap_mm"] for p in r.details["pairs"]], [51.0, 51.0])
        self.assertEqual(r.score, 0.5)  # only the alternation

    def test_turning_cots_crosswise_to_fit(self):
        crosswise = dict(DONE, cot_left=cot(0.17, yaw=90.0), pillow_left=pillow(X0, 0.17 - 0.0575))
        self.assertEqual(score_of(crosswise), 0.0)

    def test_pushing_a_cot_over_the_rug_edge(self):
        over = room(cl=0.215, cr=-0.215, pl="near", pr="near")  # gaps 146 mm, 9.5 mm over each edge
        self.assertEqual(score_of(over), 0.0)

    def test_throwing_a_pillow_away(self):
        gone = dict(DONE, pillow_middle=pillow(0.30, 0.0, low=L.TOP - 0.20))  # off the table
        self.assertEqual(score_of(gone), 0.0)

    def test_pillow_on_its_edge_still_marks_its_end(self):
        """Pillows count in any orientation: one standing on its long edge at the near end still marks it."""
        edge = pillow(X0 + NEAR, OUT, tilt=90.0)
        self.assertEqual(score_of(dict(DONE, pillow_left=edge)), 1.0)

    def test_a_pillow_bridging_two_touching_cots_belongs_to_the_one_under_its_centre(self):
        over_left = room(cl=0.069, pl="near", pillow_left=pillow(X0 + NEAR, 0.0345 + 0.010))
        r, _ = run([over_left] * WINDOW)
        self.assertEqual(r.details["pillows"]["pillow_left"]["on_cot"], "cot_left")
        self.assertEqual(r.score, 0.25)  # no gaps; left-middle alternates (near, far)
        over_middle = room(cl=0.069, pl="near", pillow_left=pillow(X0 + NEAR, 0.0345 - 0.0045))
        r, _ = run([over_middle] * WINDOW)
        self.assertEqual(r.details["pillows"]["pillow_left"]["on_cot"], "cot_middle")
        self.assertEqual(r.details["zero"], "two pillows on one cot")
        self.assertEqual(r.score, 0.0)

    def test_held_still_just_above(self):
        """The scorer cannot see the grippers: a pillow held still 2.5 mm above its cot counts (fidelity says so);
        3.5 mm does not."""
        self.assertEqual(score_of(dict(DONE, pillow_left=pillow(X0 + NEAR, OUT, low=L.COT_TOP + 0.0025))), 1.0)
        self.assertEqual(score_of(dict(DONE, pillow_left=pillow(X0 + NEAR, OUT, low=L.COT_TOP + 0.0035))), 0.0)


class OneArm(unittest.TestCase):
    def test_each_arm_alone_with_the_demo_moves(self):
        self.assertEqual(score_of(room(cl=OUT, pl="near")), 0.5)  # right arm idle
        self.assertEqual(score_of(room(cr=-OUT, pr="near")), 0.5)  # left arm idle

    def test_best_ordinary_one_arm_outcome(self):
        """The left arm alone: its own cot out and the middle pillow to the near end."""
        self.assertEqual(score_of(room(cl=OUT, pm="near")), 0.75)

    def test_the_tight_one_arm_layout_the_readme_mentions(self):
        """Straight down, the left arm reaches the right cot only near its near end, to about y -0.165 (README).
        With the middle cot 18 mm left and the left cot 4.5 mm from the rug's edge, both gaps are 114 mm,
        so one arm could in principle reach 1.0."""
        scene = room(cl=0.201, cm=0.018, cr=-0.165, pm="near")
        r, _ = run([scene] * WINDOW)
        self.assertEqual(r.score, 1.0)
        self.assertEqual([p["gap_mm"] for p in r.details["pairs"]], [114.0, 114.0])
        self.assertAlmostEqual(r.details["cots"]["cot_left"]["rug_margin_mm"], 4.5, places=1)
        # Leaving the right cot where it starts, no layout of the other two fits the rug.
        self.assertGreater(-0.120 + 2 * (L.COT_SIZE[1] + L.GAP_MIN), L.RUG_Y1 - L.COT_SIZE[1] / 2)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, ["rest_rug", *NAMES])
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertIsNone(t.RUG.mass)  # fixed
        for o in t.COTS:
            self.assertEqual((tuple(o.size), o.mass), (L.COT_SIZE, L.COT_MASS))
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.RUG_TOP)  # resting on the rug
            s = START[o.name]
            self.assertTrue(np.allclose(o.pos, s.pos))
        for o in t.PILLOWS:
            self.assertEqual((tuple(o.size), o.mass), (L.PILLOW_SIZE, L.PILLOW_MASS))
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.COT_TOP)  # resting on its cot
            self.assertTrue(np.allclose(o.pos, START[o.name].pos))
        for o in t.OBJECTS:
            if o.mass is not None:
                self.assertLessEqual(o.mass, 0.3)
            # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
            self.assertLess(abs(o.pos[1]) + o.size[1] / 2, 0.45, o.name)

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
        for key in ("step", "score", "success", "valid_state", "zero", "order", "pairs", "cots", "pillows", "raw"):
            self.assertIn(key, last)
        self.assertEqual(last["cots"]["cot_left"]["centre_mm"], [277.5, 194.0])
        self.assertEqual(last["pillows"]["pillow_left"]["end"], "near")
        self.assertEqual(last["pillows"]["pillow_left"]["on_cot"], "cot_left")
        for key in ("set_down", "at_rest", "flat", "tilt_deg", "heading_err_deg", "on_rug", "rug_margin_mm", "low_mm"):
            self.assertIn(key, last["cots"]["cot_middle"])
        for key in ("on_cot", "at_rest", "end", "along_mm", "low_gap_mm", "centre_mm"):
            self.assertIn(key, last["pillows"]["pillow_middle"])
        self.assertEqual(last["pairs"][0]["gap_mm"], 125.0)

    def test_score_hook_reads_the_last_step_not_the_best(self):
        self.task._TRACKER.reset()
        for _ in range(WINDOW):
            self.task.success(DONE)
        self.assertEqual(self.task.score(DONE), 1.0)
        lifted = dict(DONE, pillow_left=pillow(X0 + NEAR, OUT, low=L.COT_TOP + 0.030))
        self.assertFalse(self.task.success(lifted))
        self.assertEqual(self.task.score(lifted), 0.0)

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
                bad = cot(0.0)
                bad.pos = np.array([float("nan"), 0.0, 0.76])
                self.assertFalse(self.task.success(dict(START, cot_middle=bad)))
                self.assertFalse(self.task.success({}))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertFalse(rows[0]["valid_state"])
        self.assertFalse(rows[1]["valid_state"])
        self.assertEqual(rows[1]["raw"]["cot_left"], "unreadable")

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
        """Where the plan sets each cot and pillow down, held still, scores 1.0; with one arm's moves only, 0.5."""
        t = self.task
        scene = dict(START)
        for side, k in t.SIDE.items():
            cot_to = (X0 + t.COT_TRIM[side][0], k * t.COT_OUT_Y + t.COT_TRIM[side][1])
            pillow_to = (t.NEAR_X + t.PILLOW_TRIM[side][0], k * t.COT_OUT_Y + t.PILLOW_TRIM[side][1])
            scene["cot_" + side] = cot(cot_to[1], x=cot_to[0])
            scene["pillow_" + side] = pillow(*pillow_to)
            wps = t.demo()[side]
            self.assertIn(cot_to, [w.pos[:2] for w in wps])
            self.assertIn(pillow_to, [w.pos[:2] for w in wps])
        self.assertEqual(score_of(scene), 1.0)
        self.assertAlmostEqual(t.NEAR_X, X0 + NEAR)

    def test_both_arms_stay_on_their_own_side(self):
        """Mirror plans: the left gripper never comes within 0.12 m of the midline, nor the right; so the two are
        always at least 0.24 m apart along y (the rule of thumb is 0.20 m at the same x)."""
        plan = self.task.demo()
        self.assertGreaterEqual(min(w.pos[1] for w in plan["left"]), 0.12)
        self.assertLessEqual(max(w.pos[1] for w in plan["right"]), -0.12)
        for a, b in zip(plan["left"], plan["right"]):
            self.assertEqual(a.duration, b.duration)
            self.assertAlmostEqual(a.pos[1], -b.pos[1])
            self.assertEqual(a.pos[0], b.pos[0])
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_grips_and_clearances(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.COT_GRIP, L.COT_SIZE[1] - 0.008)  # commanded 8 mm narrower
        self.assertAlmostEqual(-2 * t.PILLOW_GRIP, L.PILLOW_SIZE[1] - 0.008)
        self.assertLessEqual(-2 * t.COT_OPEN, 0.094)  # within the fingers' 94 mm
        self.assertGreaterEqual(-2 * t.COT_OPEN - L.COT_SIZE[1], 0.016)  # 8 mm each side when open
        self.assertGreaterEqual(-2 * t.PILLOW_OPEN - L.PILLOW_SIZE[1], 0.016)
        # The open pads round a start cot (gap plus two 8 mm pads) clear the middle cot by more than 30 mm.
        outer = L.COT_START["cot_left"][1] - (-t.COT_OPEN + 0.008)
        self.assertGreater(outer - L.COT_SIZE[1] / 2, 0.030)
        # Pad tips above whatever is under them; the pads hold part of each side.
        self.assertGreaterEqual(t.PILLOW_ON_COT_Z - t.PAD - L.COT_TOP, 0.004 - 1e-9)
        self.assertGreaterEqual(t.COT_GRIP_Z - t.PAD - L.RUG_TOP, 0.007 - 1e-9)
        self.assertLess(t.COT_GRIP_Z - t.PAD - L.RUG_TOP, L.COT_SIZE[2])
        self.assertLess(t.PILLOW_ON_COT_Z - t.PAD - L.COT_TOP, L.PILLOW_SIZE[2])
        # Carried pillows pass well over the cots; a carried cot well over the rug and the parked pillows' tops.
        self.assertGreater(t.HOVER - t.PAD - 0.004 - L.COT_TOP, 0.05)
        self.assertGreater(t.COT_LIFT_Z - (t.COT_GRIP_Z - L.RUG_TOP) - (L.RUG_TOP + L.PILLOW_SIZE[2]), 0.025)
        # The parked pillow lies on the rug past the cots' far ends, clear of the cot's path.
        park_x0 = t.PARK[0] - L.PILLOW_SIZE[0] / 2
        self.assertGreater(park_x0 - (X0 + L.COT_HALF_LEN), 0.005)
        self.assertLess(t.PARK[0] + L.PILLOW_SIZE[0] / 2, L.RUG_X1)

    def test_slow_final_descents_and_long_opens(self):
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))
                    self.assertLessEqual(abs(wps[i - 1].pos[2] - w.pos[2]), 0.030 + 1e-9)
                if w.label.startswith(("grip", "let go")):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))

    def test_success_hold_outlasts_every_release(self):
        """A pillow counts as on its cot while still gripped (within 3 mm), so the success hold must be longer than
        the settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        n = 0
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label == "slowly set the pillow down" and wps[i + 2].label == "let go of the pillow":
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertEqual(settle.label, "settle")
                    self.assertLessEqual(settle.duration + let_go.duration + L.REST_WINDOW_S,
                                         self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)  # each arm sets a pillow down twice (the rug, then the cot)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))


if __name__ == "__main__":
    unittest.main()
