"""Offline tests for scoring.py and task.py: synthetic states, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
STUB = HERE.parent.parent / "tools" / "offline" / "yam_stub"
sys.path.insert(0, str(HERE))
if STUB.is_dir() and str(STUB) not in sys.path:
    sys.path.append(str(STUB))

import layout as L  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

A1, A2, B = L.NAMES
SIZE = {n: L.TEMPLATES[n][0] for n in L.NAMES}


def quat(yaw=0.0, tilt_deg=0.0, tilt_axis=(1.0, 0.0, 0.0)):
    """Yaw about z, then a tilt about a horizontal axis (world)."""
    qz = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    if not tilt_deg:
        return qz
    ax = np.asarray(tilt_axis, float) / np.linalg.norm(tilt_axis)
    h = math.radians(tilt_deg) / 2
    qt = np.array([*(ax * math.sin(h)), math.cos(h)])
    # qt * qz (Hamilton product, xyzw)
    x1, y1, z1, w1 = qt
    x2, y2, z2, w2 = qz
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def state(name, x, y, bottom=L.SHEET_TOP, yaw=0.0, tilt=0.0, vel=(0, 0, 0)):
    """A template's state: its long side at ``yaw`` from x, its lowest point at ``bottom`` (a flat block's)."""
    size = np.array(SIZE[name], float)
    q = quat(yaw, tilt)
    c = np.array([x, y, bottom + size[2] / 2])
    s = SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=c, high=c)
    pts = S.corners(s, size)
    c = c + np.array([0.0, 0.0, bottom - pts[:, 2].min()])  # lowest corner exactly at ``bottom``
    s.pos = c
    pts = S.corners(s, size)
    s.low, s.high = pts.min(axis=0), pts.max(axis=0)
    return s


def start_state(name):
    _, _, x, y, yaw = L.TEMPLATES[name]
    return state(name, x, y, bottom=L.TOP, yaw=yaw)


START = {n: start_state(n) for n in L.NAMES}
NOMINAL = {n: state(n, L.ROW_X, L.PLACE_Y[n]) for n in L.NAMES}


def arm(tips, opening=1.0):
    """A gripper pointing straight down with its finger-tip point at ``tips`` (0 closed .. 1 open)."""
    return SimpleNamespace(tcp=np.asarray(tips, float) + [0, 0, 0.013], tips=np.asarray(tips, float), opening=opening)


A_HELD = 0.035 / 0.04695  # the opening while a gripper holds an A template by its 70 mm length


AWAY = {"left": arm((0.240, 0.300, L.TOP + 0.107)), "right": arm((0.240, -0.300, L.TOP + 0.107))}


def run(layout_states, steps=30, arms=AWAY, tracker=None):
    t = tracker or Tracker()
    r = None
    for _ in range(steps):
        r = t.update(layout_states, arms)
    return r, t


def with_(base, **changes):
    out = dict(base)
    out.update(changes)
    return out


class Geometry(unittest.TestCase):
    def test_tilt_and_grain(self):
        self.assertAlmostEqual(S.tilt_deg(quat(0.3, 6.0)), 6.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(quat(0.0, 180.0)), 0.0, places=4)  # face down: alike
        self.assertAlmostEqual(S.grain_deg(quat(math.radians(4.0))), 4.0, places=6)
        self.assertAlmostEqual(S.grain_deg(quat(math.radians(176.0))), 4.0, places=6)  # either way round
        self.assertAlmostEqual(S.grain_deg(quat(math.pi / 2)), 90.0, places=6)
        self.assertAlmostEqual(S.grain_deg(quat(-math.pi / 4)), 45.0, places=6)
        self.assertEqual(S.grain_deg(quat(0.0, 90.0, (0, 1, 0))), 90.0)  # long side straight up

    def test_nominal_numbers(self):
        """The task text's nominal end state: footprints, gaps, edge allowances and the 91 mm strip."""
        a1, a2, b = (NOMINAL[n] for n in L.NAMES)
        self.assertAlmostEqual(S.x_span(a1, SIZE[A1])[0], 0.219, places=9)
        self.assertAlmostEqual(S.x_span(b, SIZE[B])[1], 0.284, places=9)
        self.assertAlmostEqual(S.polygon_distance(S.outline(a1, SIZE[A1]), S.outline(a2, SIZE[A2])), 0.020, places=9)
        self.assertAlmostEqual(S.polygon_distance(S.outline(a2, SIZE[A2]), S.outline(b, SIZE[B])), 0.020, places=9)
        self.assertAlmostEqual(S.edge_inset(a1, SIZE[A1]), 0.019, places=9)
        self.assertAlmostEqual(S.edge_inset(b, SIZE[B]), 0.024, places=9)
        strip, near, far = Tracker.clear_strip(NOMINAL)
        self.assertAlmostEqual(strip, 0.091, places=9)
        self.assertAlmostEqual(near, 0.019, places=9)
        self.assertAlmostEqual(far, 0.091, places=9)

    def test_polygon_distance(self):
        sq = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)
        self.assertEqual(S.polygon_distance(sq, sq + 0.5), 0.0)
        self.assertAlmostEqual(S.polygon_distance(sq, sq + [3, 0]), 2.0)
        self.assertAlmostEqual(S.polygon_distance(sq, sq + [2, 2]), math.sqrt(2))

    def test_finger_zone(self):
        """The zone's radius: half the pad gap + 10 mm, and 20 mm across; judged from its flat bottom straight above."""
        self.assertAlmostEqual(S.zone_radius(arm((0, 0, 1))), math.hypot(0.04695 + 0.010, 0.020), places=6)
        self.assertAlmostEqual(S.zone_radius(arm((0, 0, 1), 0.0)), math.hypot(0.010, 0.020), places=9)
        self.assertAlmostEqual(S.zone_radius(arm((0, 0, 1), 1.7)), S.zone_radius(arm((0, 0, 1))), places=12)
        s = NOMINAL[A2]
        top = s.pos[2] + 0.006
        arms = dict(AWAY, left=arm((L.ROW_X, 0.0, top + 0.050)))
        self.assertAlmostEqual(S.zone_distance(arms, s, SIZE[A2]), 0.050, places=6)  # straight above: its height
        r = S.zone_radius(arms["left"])
        beside = dict(AWAY, left=arm((L.ROW_X, 0.020 + r + 0.030, s.pos[2])))  # level with it, 30 mm off the zone
        self.assertAlmostEqual(S.zone_distance(beside, s, SIZE[A2]), 0.030, places=6)
        self.assertIsNone(S.zone_distance(None, s, SIZE[A2]))

    def test_point_box_distance(self):
        s = NOMINAL[A2]
        self.assertEqual(S.point_box_distance(s.pos, s, SIZE[A2]), 0.0)
        above = s.pos + [0, 0, 0.006 + 0.030]
        self.assertAlmostEqual(S.point_box_distance(above, s, SIZE[A2]), 0.030, places=9)
        beside = s.pos + [0.035 + 0.010, 0, 0]  # 10 mm past an end, A2's long side along x
        self.assertAlmostEqual(S.point_box_distance(beside, s, SIZE[A2]), 0.010, places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(START)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["n_laid"], 0)

    def test_nominal_layout_scores_one(self):
        r, _ = run(NOMINAL)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["strip_mm"], 91.0)

    def test_mirror_row_along_the_far_edge_scores_one(self):
        far = {n: state(n, 0.326, L.PLACE_Y[n]) for n in L.NAMES}
        r, _ = run(far)
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(r.details["near_margin_mm"], 91.0)

    def test_one_and_two_laid(self):
        r, _ = run(with_(START, **{A1: NOMINAL[A1]}))
        self.assertEqual(r.score, 0.2)
        r, _ = run(with_(START, **{A1: NOMINAL[A1], A2: NOMINAL[A2]}))
        self.assertEqual(r.score, 0.4)  # the left arm alone
        r, _ = run(with_(START, **{B: NOMINAL[B]}))
        self.assertEqual(r.score, 0.2)  # the right arm alone

    def test_three_laid_across_the_middle_scores_06(self):
        middle = {n: state(n, L.SHEET_X, L.PLACE_Y[n]) for n in L.NAMES}  # strip 55 mm each side
        r, _ = run(middle)
        self.assertEqual(r.details["n_laid"], 3)
        self.assertEqual(r.score, 0.6)
        self.assertFalse(r.success)

    def test_two_rows_scores_06(self):
        rows = {A1: state(A1, 0.243, 0.0), A2: state(A2, 0.337, 0.0), B: state(B, 0.243, -0.080)}  # 8 mm each edge
        r, _ = run(rows)
        self.assertEqual(r.details["n_laid"], 3)
        self.assertLess(r.details["strip_mm"], 80.0)
        self.assertEqual(r.score, 0.6)

    def test_strip_boundary_is_inclusive(self):
        # A templates' far end at x 0.300 exactly: strip 80 mm. One mm further: 79 mm.
        edge = {n: state(n, 0.300 - SIZE[n][0] / 2, L.PLACE_Y[n]) for n in L.NAMES}
        r, _ = run(edge)
        self.assertAlmostEqual(r.details["strip_mm"], 80.0)
        self.assertEqual(r.score, 1.0)
        over = {n: state(n, 0.301 - SIZE[n][0] / 2, L.PLACE_Y[n]) for n in L.NAMES}
        r, _ = run(over)
        self.assertEqual(r.score, 0.6)

    def test_x_tolerance_band(self):
        """A templates' centres 0.243-0.265 at 0 degrees keep 1.0; at 0.242 both are 7 mm from the near edge (only B
        is laid), at 0.266 the strip is 79 mm."""
        for x, want in ((0.243, 1.0), (0.265, 1.0), (0.242, 0.2), (0.266, 0.6)):
            lay = with_(NOMINAL, **{A1: state(A1, x, 0.060), A2: state(A2, x, 0.0)})
            r, _ = run(lay)
            self.assertEqual(r.score, want, f"x {x}")

    def test_y_tolerance(self):
        """Each template may be 7.5 mm off in y before a 20 mm gap falls under 5 mm."""
        r, _ = run(with_(NOMINAL, **{A1: state(A1, L.ROW_X, 0.060 - 0.0075)}))
        self.assertEqual(r.score, 1.0)
        r, _ = run(with_(NOMINAL, **{A1: state(A1, L.ROW_X, 0.060 - 0.016)}))
        self.assertEqual(r.details["n_laid"], 1)  # A1 and A2 4 mm apart: both lose their credit
        self.assertEqual(r.score, 0.2)


class Terms(unittest.TestCase):
    """Each term of 'laid', one at a time on one template (the others stay laid)."""

    def score_with(self, name, s, arms=AWAY, steps=30):
        r, _ = run(with_(NOMINAL, **{name: s}), steps=steps, arms=arms)
        return r

    def test_grain(self):
        self.assertEqual(self.score_with(A2, state(A2, L.ROW_X, 0.0, yaw=math.radians(5.0))).score, 1.0)
        self.assertEqual(self.score_with(A2, state(A2, L.ROW_X, 0.0, yaw=math.radians(-5.0))).score, 1.0)
        self.assertEqual(self.score_with(A2, state(A2, L.ROW_X, 0.0, yaw=math.pi)).score, 1.0)  # either way round
        r = self.score_with(A2, state(A2, L.ROW_X, 0.0, yaw=math.radians(5.5)))
        self.assertEqual(r.score, 0.4)
        self.assertFalse(r.details["a2_laid"])

    def test_across_the_streaks_earns_nothing(self):
        # Lined up with the sheet's outline (long side along y): across the streaks.
        r = self.score_with(A1, state(A1, 0.240, 0.090, yaw=math.pi / 2))
        self.assertFalse(r.details["a1_laid"])
        self.assertEqual(r.score, 0.4)

    def test_turned_45_degrees_earns_nothing(self):
        r = self.score_with(B, state(B, L.ROW_X, -0.080, yaw=-math.pi / 4))  # still 5 mm clear of A2
        self.assertFalse(r.details["b_laid"])
        self.assertEqual(r.score, 0.4)

    def test_edge_allowance(self):
        # B's near end 8 mm in from the near edge: laid; 7 mm: not (the row's strip is still over 80 mm).
        self.assertEqual(self.score_with(B, state(B, 0.238, -0.060)).score, 1.0)
        r = self.score_with(B, state(B, 0.237, -0.060))
        self.assertEqual(r.score, 0.4)
        # Too near a short end.
        self.assertEqual(self.score_with(B, state(B, L.ROW_X, -0.150 + 0.020 + 0.008)).score, 1.0)
        self.assertEqual(self.score_with(B, state(B, L.ROW_X, -0.150 + 0.020 + 0.007)).score, 0.4)

    def test_gap(self):
        # B 5 mm from A2: laid; 4 mm: B and A2 both lose their credit.
        self.assertEqual(self.score_with(B, state(B, L.ROW_X, -0.045)).score, 1.0)
        r = self.score_with(B, state(B, L.ROW_X, -0.044))
        self.assertEqual(r.details["n_laid"], 1)
        self.assertEqual(r.score, 0.2)

    def test_a_template_carried_overhead_does_not_count_in_the_gap(self):
        # A1 carried by its ends, its underside 45 mm above A2's place (the holding fingers' tips 37 mm above A2's
        # top): A1 is not on the sheet, and A2 keeps its credit (A1 is more than 20 mm above the sheet).
        s = state(A1, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.045)
        held = dict(AWAY, left=arm((L.ROW_X, 0.0, L.SHEET_TOP + 0.049), A_HELD))
        r = self.score_with(A1, s, arms=held)
        self.assertFalse(r.details["a1_laid"])
        self.assertFalse(r.details["a1_low"])
        self.assertTrue(r.details["a2_laid"])
        self.assertTrue(r.details["b_laid"])
        self.assertEqual(r.score, 0.4)

    def test_gap_band_boundary(self):
        # A1 over A2's place with its underside 20 mm above the sheet's top: it lies low and counts (A2 loses its
        # credit); 21 mm: it does not.
        r = self.score_with(A1, state(A1, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.020))
        self.assertTrue(r.details["a1_low"])
        self.assertFalse(r.details["a2_laid"])
        self.assertEqual(r.details["n_laid"], 1)
        r = self.score_with(A1, state(A1, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.021))
        self.assertFalse(r.details["a1_low"])
        self.assertTrue(r.details["a2_laid"])
        self.assertEqual(r.details["n_laid"], 2)

    def test_a_template_on_the_floor_below_another_does_not_count(self):
        r = self.score_with(B, state(B, L.ROW_X, 0.0, bottom=0.0))
        self.assertFalse(r.details["b_low"])
        self.assertTrue(r.details["a2_laid"])
        self.assertEqual(r.score, 0.4)

    def test_tilt(self):
        r = self.score_with(A2, state(A2, L.ROW_X, 0.0, tilt=2.9))
        self.assertTrue(r.details["a2_laid"])
        r = self.score_with(A2, state(A2, L.ROW_X, 0.0, tilt=3.5))
        self.assertFalse(r.details["a2_laid"])
        self.assertEqual(r.score, 0.4)

    def test_height(self):
        self.assertEqual(self.score_with(A2, state(A2, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.002)).score, 1.0)
        self.assertEqual(self.score_with(A2, state(A2, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.0025)).score, 0.4)

    def test_on_the_table_off_the_sheet(self):
        r = self.score_with(A2, state(A2, 0.150, 0.0, bottom=L.TOP))
        self.assertFalse(r.details["a2_laid"])
        self.assertEqual(r.score, 0.4)

    def test_on_the_floor(self):
        r = self.score_with(B, state(B, 0.30, -0.80, bottom=0.0))
        self.assertEqual(r.score, 0.4)

    def test_let_go(self):
        s = NOMINAL[A2]
        top = s.pos[2] + 0.006
        # A closed gripper straight above A2: 29 mm is too near (A2 only), 30 mm counts.
        near = dict(AWAY, left=arm((L.ROW_X, 0.0, top + 0.029), 0.0))
        r = self.score_with(A2, s, arms=near)
        self.assertFalse(r.details["a2_laid"])
        self.assertTrue(r.details["a1_laid"])
        self.assertEqual(r.score, 0.4)
        clear = dict(AWAY, left=arm((L.ROW_X, 0.0, top + 0.030), 0.0))
        self.assertEqual(self.score_with(A2, s, arms=clear).score, 1.0)
        # Wide open over A2, its fingers 47 mm either side reach over A1 and B too: 29 mm up voids all three.
        r = self.score_with(A2, s, arms=dict(AWAY, left=arm((L.ROW_X, 0.0, top + 0.029))))
        self.assertEqual(r.details["n_laid"], 0)
        self.assertEqual(self.score_with(A2, s, arms=dict(AWAY, left=arm((L.ROW_X, 0.0, top + 0.030)))).score, 1.0)
        right_near = dict(AWAY, right=arm((L.ROW_X + 0.035 + 0.020, -0.060, L.SHEET_TOP + 0.006)))
        self.assertFalse(self.score_with(B, NOMINAL[B], arms=right_near).details["b_laid"])

    def test_open_finger_resting_on_a_template_is_not_let_go(self):
        """The review's case: an open gripper whose finger-tip point is 30 mm beyond A1's side (y 0.110) and 2 mm
        below its top, so one pad (47 mm from that point) rests on A1. The point alone is 30 mm away; the finger
        zone overlaps A1."""
        s = NOMINAL[A1]
        tip = (L.ROW_X, 0.080 + 0.030, L.SHEET_TOP + 0.010)
        self.assertAlmostEqual(S.point_box_distance(tip, s, SIZE[A1]), 0.030, places=9)
        r = self.score_with(A1, s, arms=dict(AWAY, left=arm(tip)))
        self.assertEqual(r.details["a1_zone_mm"], 0.0)
        self.assertFalse(r.details["a1_laid"])
        self.assertTrue(r.details["b_laid"])
        # The zone's edge (60.4 mm from that point, at any heading) is also 29.6 mm from A2's side: A2 waits too.
        self.assertAlmostEqual(r.details["a2_zone_mm"], 29.6, places=1)
        self.assertEqual(r.score, 0.2)

    def test_open_gripper_beside_a_template(self):
        # Level with A1, beside its far side (y 0.080): the zone's edge 30 mm off counts, 29 mm does not, at the
        # opening it has (wide open: 60.4 mm radius; closed: 22.4 mm).
        s = NOMINAL[A1]
        for opening in (1.0, 0.5, 0.0):
            r0 = S.zone_radius(arm((0, 0, 0), opening))
            ok = dict(AWAY, left=arm((L.ROW_X, 0.080 + r0 + 0.030, s.pos[2]), opening))
            self.assertEqual(self.score_with(A1, s, arms=ok).score, 1.0, f"opening {opening}")
            bad = dict(AWAY, left=arm((L.ROW_X, 0.080 + r0 + 0.029, s.pos[2]), opening))
            self.assertFalse(self.score_with(A1, s, arms=bad).details["a1_laid"], f"opening {opening}")

    def test_still_held_on_the_sheet_earns_nothing(self):
        # Gripped by its ends, the pads' tips 4 mm above the sheet: the tip point is inside the box. The holding
        # fingers (up to 20 mm across from A1's centre line) are also 20 mm from A2, so A2 is not let go either.
        s = NOMINAL[A1]
        held = dict(AWAY, left=arm((L.ROW_X, 0.060, L.SHEET_TOP + 0.004), A_HELD))
        r = self.score_with(A1, s, arms=held)
        self.assertEqual(r.details["a1_zone_mm"], 0.0)
        self.assertFalse(r.details["a1_laid"])
        self.assertFalse(r.details["a2_laid"])
        self.assertTrue(r.details["b_laid"])
        self.assertEqual(r.score, 0.2)

    def test_moving_earns_nothing(self):
        t = Tracker()
        for i in range(30):  # A2 creeping 0.25 mm a step (5 mm/s)
            t.update(with_(NOMINAL, **{A2: state(A2, L.ROW_X + 0.00025 * i, 0.0)}), AWAY)
        r = t.last
        self.assertGreater(r.details["a2_path_mm"], 2.0)
        self.assertFalse(r.details["a2_laid"])
        self.assertEqual(r.score, 0.4)

    def test_slow_creep_under_the_path_limit_counts(self):
        t = Tracker()
        for i in range(30):  # 0.1 mm a step: 1 mm over 0.5 s
            t.update(with_(NOMINAL, **{A2: state(A2, L.ROW_X + 0.0001 * i, 0.0)}), AWAY)
        self.assertAlmostEqual(t.last.details["a2_path_mm"], 1.0, places=6)
        self.assertEqual(t.last.score, 1.0)

    def test_velocity_is_never_used(self):
        """A still template whose reported velocity chatters at 45 mm/s is still at rest (path, not velocity)."""
        chatter = with_(NOMINAL, **{A2: state(A2, L.ROW_X, 0.0, vel=(0.045, -0.03, 0.02))})
        r, _ = run(chatter)
        self.assertEqual(r.score, 1.0)

    def test_needs_half_a_second_of_history(self):
        r, _ = run(NOMINAL, steps=10)
        self.assertEqual(r.score, 0.0)  # 10 states are 9 steps of path: not yet 0.5 s
        r, _ = run(NOMINAL, steps=11)
        self.assertEqual(r.score, 1.0)


class ExploitsAndEdgeCases(unittest.TestCase):
    def test_stacked_templates_earn_nothing(self):
        stack = with_(NOMINAL, **{A1: state(A1, L.ROW_X, 0.0, bottom=L.SHEET_TOP + 0.012)})  # A1 on A2
        r, _ = run(stack)
        self.assertFalse(r.details["a1_laid"])
        self.assertFalse(r.details["a2_laid"])
        self.assertEqual(r.score, 0.2)

    def test_hovering_just_above_the_glass_earns_nothing(self):
        # Held 1.5 mm above the sheet (within the 2 mm height band) by a gripper closed on its ends.
        s = state(A1, L.ROW_X, 0.060, bottom=L.SHEET_TOP + 0.0015)
        held = dict(AWAY, left=arm((L.ROW_X, 0.060, L.SHEET_TOP + 0.0055), A_HELD))
        r, _ = run(with_(NOMINAL, **{A1: s}), arms=held)
        self.assertFalse(r.details["a1_laid"])
        self.assertTrue(r.details["b_laid"])
        self.assertLess(r.score, 1.0)

    def test_leaning_over_the_sheet_edge_earns_nothing(self):
        # Half on the 3 mm sheet, half on the table: tilted about 4.9 degrees.
        lean = state(B, 0.200, -0.060, bottom=L.TOP, tilt=4.9, yaw=0.0)
        r, _ = run(with_(NOMINAL, **{B: lean}))
        self.assertFalse(r.details["b_laid"])

    def test_standing_on_edge_earns_nothing(self):
        edge = state(A2, L.ROW_X, 0.0, tilt=90.0)
        r, _ = run(with_(NOMINAL, **{A2: edge}))
        self.assertFalse(r.details["a2_laid"])

    def test_turn_everything_90_degrees_is_wrong_for_a2_and_b(self):
        # One fixed routine (every template turned 90 degrees from its start): only A1 ends with the streaks.
        lay = {A1: state(A1, L.ROW_X, 0.060, yaw=0.0), A2: state(A2, L.ROW_X - 0.015, 0.0, yaw=math.pi / 2),
               B: state(B, L.ROW_X, -0.070, yaw=math.pi / 4)}
        r, _ = run(lay)
        self.assertEqual(r.details["n_laid"], 1)
        self.assertEqual(r.score, 0.2)

    def test_lined_up_with_the_sheet_outline_scores_zero(self):
        # Long sides along the sheet's long edges (y): every template across the streaks.
        lay = {A1: state(A1, 0.250, 0.080, yaw=math.pi / 2), A2: state(A2, 0.250, 0.0, yaw=math.pi / 2),
               B: state(B, 0.250, -0.080, yaw=math.pi / 2)}
        r, _ = run(lay)
        self.assertEqual(r.score, 0.0)

    def test_neat_row_on_the_table_beside_the_sheet_scores_zero(self):
        lay = {n: state(n, 0.130, L.PLACE_Y[n], bottom=L.TOP) for n in L.NAMES}  # x 0.095-0.165: off the sheet
        r, _ = run(lay)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(any(r.details[f"{k}_over_sheet"] for k in ("a1", "a2", "b")))

    def test_five_degree_templates_keep_the_band(self):
        # At 5 degrees an A template's outline is 73.2 mm deep: its centre may be 9 mm either side of x 0.254.
        yaw = math.radians(5.0)
        for x in (0.254 - 0.009, 0.254 + 0.009):
            lay = {A1: state(A1, x, 0.060, yaw=yaw), A2: state(A2, x, 0.0, yaw=-yaw), B: state(B, x, -0.060, yaw=yaw)}
            r, _ = run(lay)
            self.assertEqual(r.score, 1.0, f"x {x}")

    def test_drop_from_a_height_counts_only_where_it_lands(self):
        t = Tracker()
        run(with_(NOMINAL, **{B: state(B, L.ROW_X, -0.060, bottom=L.SHEET_TOP + 0.05)}), steps=20, tracker=t)
        self.assertEqual(t.last.score, 0.4)  # B in the air
        for z in (0.03, 0.01, 0.0):  # falls flat onto its place
            t.update(with_(NOMINAL, **{B: state(B, L.ROW_X, -0.060, bottom=L.SHEET_TOP + z)}), AWAY)
        self.assertEqual(t.last.score, 0.4)  # not at rest yet: its path over 0.5 s is 50 mm
        run(NOMINAL, steps=10, tracker=t)
        self.assertEqual(t.last.score, 1.0)  # landed laid: the end state is what counts

    def test_do_nothing_scores_zero(self):
        r, _ = run(START, steps=200)
        self.assertEqual(r.score, 0.0)

    def test_arms_unreadable_fail_closed(self):
        r, t = run(NOMINAL, arms=None)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["arms_ok"])
        self.assertEqual(t.arms_unavailable, 30)
        broken = {"left": arm((0.2, 0.3, 0.9)), "right": SimpleNamespace(tips=np.array([np.nan, 0, 0]))}
        r, _ = run(NOMINAL, arms=broken)
        self.assertEqual(r.score, 0.0)
        no_opening = dict(AWAY, right=arm((0.240, -0.300, L.TOP + 0.107), float("nan")))
        r, _ = run(NOMINAL, arms=no_opening)
        self.assertEqual(r.score, 0.0)
        no_tcp = dict(AWAY, left=SimpleNamespace(tips=AWAY["left"].tips, opening=1.0))
        r, _ = run(NOMINAL, arms=no_tcp)
        self.assertEqual(r.score, 0.0)

    def test_one_template_missing_scores_zero(self):
        r, _ = run({A1: NOMINAL[A1], A2: NOMINAL[A2]})
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])


class BadStatesAndReset(unittest.TestCase):
    def test_nan_state_scores_zero_and_keeps_the_tracker(self):
        t = Tracker()
        run(NOMINAL, steps=20, tracker=t)
        before = {n: list(t.history[n]) for n in L.NAMES}
        bad = state(A2, L.ROW_X, 0.0)
        bad.pos = np.array([np.nan, 0.0, 0.8])
        r = t.update(with_(NOMINAL, **{A2: bad}), AWAY)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual({n: list(t.history[n]) for n in L.NAMES}, before)
        self.assertEqual(t.invalid_steps, 1)
        self.assertEqual(t.update(NOMINAL, AWAY).score, 1.0)  # the next good step scores as before

    def test_zero_quaternion_scores_zero(self):
        t = Tracker()
        run(NOMINAL, steps=20, tracker=t)
        bad = state(B, L.ROW_X, -0.060)
        bad.quat_xyzw = np.zeros(4)
        self.assertEqual(t.update(with_(NOMINAL, **{B: bad}), AWAY).score, 0.0)
        inf = state(B, L.ROW_X, -0.060)
        inf.lin_vel = np.array([np.inf, 0, 0])
        self.assertEqual(t.update(with_(NOMINAL, **{B: inf}), AWAY).score, 0.0)
        self.assertEqual(t.invalid_steps, 2)

    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        run(NOMINAL, steps=30, tracker=t)
        self.assertEqual(t.last.score, 1.0)
        r = t.update(START, AWAY)  # every template jumps back onto its mark in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history[A1]), 1)
        self.assertEqual(r.score, 0.0)

    def test_putting_things_back_slowly_does_not_reset(self):
        t = Tracker()
        run(START, steps=5, tracker=t)
        a1 = L.TEMPLATES[A1]
        for i in range(1, 41):  # A1 lifted away and brought back gradually
            f = math.sin(math.pi * i / 40)
            t.update(with_(START, **{A1: state(A1, a1[2] + 0.1 * f, a1[3], bottom=L.TOP + 0.05 * f, yaw=a1[4])}),
                     AWAY)
        self.assertEqual(t.steps, 45)

    def test_tracker_reset_hook(self):
        t = Tracker()
        run(NOMINAL, steps=30, tracker=t)
        t.reset()
        self.assertEqual(t.update(NOMINAL, AWAY).score, 0.0)  # no 0.5 s of history yet


class TaskFile(unittest.TestCase):
    """task.py through the offline stub: the objects, the hooks and the demo's shape."""

    @classmethod
    def setUpClass(cls):
        if not STUB.is_dir():
            raise unittest.SkipTest("tools/offline/yam_stub not found")
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "TASK_TRACE"):
            os.environ.pop(k, None)
        cls.task = importlib.import_module("task")
        cls.rig = importlib.import_module("rigstate")

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), 15)
        self.assertEqual(len(set(names)), 15)
        strips = [o for o in self.task.OBJECTS if o.name.startswith("glass_strip_")]
        self.assertEqual(len(strips), 12)
        self.assertTrue(all(abs(o.size[1] - 0.025) < 1e-12 for o in strips))  # 25 mm: 4-5 px at 320 x 176
        self.assertTrue(all(o.mass is None for o in strips))
        self.assertAlmostEqual(min(o.pos[1] - o.size[1] / 2 for o in strips), L.SHEET_Y0)
        self.assertAlmostEqual(max(o.pos[1] + o.size[1] / 2 for o in strips), L.SHEET_Y1)
        self.assertEqual(strips[0].color, L.DARK_AMBER)
        self.assertEqual(strips[1].color, L.LIGHT_AMBER)
        loose = {o.name: o for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(set(loose), set(L.NAMES))
        self.assertAlmostEqual(loose[B].yaw, -math.pi / 4)

    def test_hooks_on_the_start_layout(self):
        tf = importlib.import_module("yam_arena.task_file")
        objs = {}
        for o in self.task.OBJECTS:
            q = quat(o.yaw)
            objs[o.name] = tf.ObjectState(np.array(o.pos, float), q, np.zeros(3), np.zeros(3),
                                          np.array(o.pos, float), np.array(o.pos, float))
        self.rig.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": 1.0} for a in S.ARMS})
        self.task._TRACKER.reset()
        for _ in range(15):
            self.assertFalse(self.task.success(objs))
        self.assertEqual(self.task.score(objs), 0.0)

    def test_demo_shape(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left = sum(w.duration for w in plan["left"])
        right = sum(w.duration for w in plan["right"])
        self.assertAlmostEqual(left, self.task.LEFT_DONE_S)
        self.assertAlmostEqual(left, 36.5)
        self.assertAlmostEqual(right, 48.5)
        for w in plan["left"] + plan["right"]:
            self.assertGreater(w.duration, 0.0)
            self.assertTrue(-0.04695 - 1e-9 <= w.finger <= 0.0)
        # the right arm comes over the sheet only once the left arm has parked
        t = 0.0
        for w in plan["right"]:
            t += w.duration
            if w.pos[1] > L.SHEET_Y0 - 0.06:
                self.assertGreaterEqual(t - w.duration, self.task.LEFT_DONE_S - 1e-9)

    def test_text_agrees_with_the_code(self):
        import json
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)
        self.assertIn("2,400 steps", meta["run_limit"])
        self.assertIn("Possible scores: 0, 0.2, 0.4, 0.6, 1.0.", meta["scoring"])
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.task.SCORE_OVER, "end")

    def test_demo_switches(self):
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(set(self.task.demo()), {"left"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(self.task.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")


if __name__ == "__main__":
    unittest.main()
