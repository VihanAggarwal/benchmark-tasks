"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic pin states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
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
    Tracker, at_rail, gap_to_rect, inside_margin, lowest, outline, pin_report, red_gap, released, rotation,
    supports_quarter, tilt_deg, touches_red, valid, zone_gap,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
OPEN = 1.0
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
# The middle of each quarter's clear window (where a pin's centre may stand: 5 mm inside the outline, 3 mm clear
# of red).
CENTRE = {
    "near_left": (0.2630, 0.0570),
    "near_right": (0.2700, -0.0400),
    "far_left": (0.3168, 0.0178),
    "far_right": (0.3300, -0.0570),
}
# Each window's extent for a pin's centre (x0, x1, y0, y1), from the layout (checked by test_windows below).
WINDOW = {
    "near_left": (0.2550, 0.2710, 0.0490, 0.0650),
    "near_right": (0.2630, 0.2770, -0.0480, -0.0320),
    "far_left": (0.3095, 0.3240, 0.0095, 0.0260),
    "far_right": (0.3230, 0.3370, -0.0650, -0.0490),
}
PINS_FOR = {"near_left": "pin_left_1", "far_left": "pin_left_2", "near_right": "pin_right_1",
            "far_right": "pin_right_2"}


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like pin: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.abs(a) * L.PIN_H / 2 + L.PIN_R * np.sqrt(np.clip(1.0 - a ** 2, 0.0, 1.0))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def pin(x, y, base=None, lift=0.0, tilt=0.0, lean_to=0.0, flip=False, vel=(0, 0, 0)):
    """A pin standing with its centre seen from above at (x, y), its lowest point at ``base`` (default: the plate's
    top) plus ``lift``, its axis tilted ``tilt`` degrees, leaning toward the heading ``lean_to`` (degrees from +x);
    ``flip`` stands it upside down."""
    q = qmul(about("z", lean_to), qmul(about("y", tilt), about("x", 180.0 if flip else 0.0)))
    t = math.radians(tilt)
    z0 = (L.PLATE_TOP if base is None else base) + lift
    zc = z0 + (L.PIN_H / 2) * math.cos(t) + L.PIN_R * math.sin(t)
    return state((x, y, zc), q, vel)


def lying(x, y, yaw=0.0, base=None):
    """A pin lying on its side, its axis along the heading ``yaw`` (degrees from +x)."""
    q = qmul(about("z", yaw), about("y", 90.0))
    return state((x, y, (L.PLATE_TOP if base is None else base) + L.PIN_R), q)


def start():
    """Every pin where it starts: upright in its tray."""
    return {name: pin(x, y, base=L.TRAY_TOP) for name, x, y in L.PINS}


def done(quarters=L.QUARTER_NAMES, offsets=None):
    """One pin in the middle of the clear window of each of ``quarters``; the others stay in their trays."""
    s = start()
    for q in quarters:
        dx, dy = (offsets or {}).get(q, (0.0, 0.0))
        x, y = CENTRE[q]
        s[PINS_FOR[q]] = pin(x + dx, y + dy)
    return s


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def feed(t, states, arms=None, n=1):
    r = None
    for _ in range(n):
        r = t.update(states, snap() if arms is None else arms)
    return r


def score_of(states, arms=None, n=12):
    return feed(Tracker(), states, arms, n)


def rect_dist(x, y, rect):
    x0, x1, y0, y1 = rect
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def valid_centre(x, y):
    """An upright pin centred here would count (5 mm inside the outline, 3 mm clear of red), by plain geometry."""
    inside = min(x - L.BOARD_X0, L.BOARD_X1 - x, y - L.BOARD_Y0, L.BOARD_Y1 - y) - L.PIN_R
    red = min(rect_dist(x, y, r[1:]) for r in L.RED) - L.PIN_R
    return inside >= L.EDGE_IN - 1e-9 and red >= L.RED_CLEAR - 1e-9


class Geometry(unittest.TestCase):
    def test_marks_do_not_overlap_and_red_stays_off_the_lines(self):
        strips = [(n, (x0, x1, y0, y1)) for n, x0, x1, y0, y1 in L.lines()] + [(r[0], r[1:]) for r in L.RED]
        for i, (na, a) in enumerate(strips):
            self.assertTrue(L.PLATE_X0 <= a[0] < a[1] <= L.PLATE_X1 and L.PLATE_Y0 <= a[2] < a[3] <= L.PLATE_Y1, na)
            for nb, b in strips[i + 1:]:
                overlap_x = min(a[1], b[1]) - max(a[0], b[0])
                overlap_y = min(a[3], b[3]) - max(a[2], b[2])
                self.assertFalse(overlap_x > 1e-9 and overlap_y > 1e-9, f"{na} overlaps {nb}")
        for name, *rect in L.RED:
            for ln, *line in L.lines():
                gx = max(line[0] - rect[1], rect[0] - line[1], 0.0)
                gy = max(line[2] - rect[3], rect[2] - line[3], 0.0)
                self.assertGreaterEqual(max(gx, gy), 0.002 - 1e-9, f"{name} within 2 mm of {ln}")
            q = L.quarter_of((rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2)
            self.assertIsNotNone(q)
            self.assertTrue(name.startswith("red_" + "".join(w[0] for w in q.split("_"))), name)

    def test_quarter_of(self):
        self.assertEqual(L.quarter_of(0.32, 0.02), "far_left")
        self.assertEqual(L.quarter_of(0.26, 0.02), "near_left")
        self.assertEqual(L.quarter_of(0.26, -0.02), "near_right")
        self.assertEqual(L.quarter_of(0.32, -0.02), "far_right")
        self.assertEqual(L.quarter_of(0.300, 0.000), "far_left")  # on both lines: far, left
        self.assertIsNone(L.quarter_of(0.23, 0.0))
        self.assertIsNone(L.quarter_of(0.30, 0.0801))

    def test_windows(self):
        """Every quarter has exactly one clear window, one connected patch of centres where a pin counts, at least
        14 mm each way, and there is no such spot anywhere else on the board (0.5 mm grid)."""
        step = 0.0005
        xs = np.round(np.arange(L.BOARD_X0, L.BOARD_X1 + 1e-9, step), 6)
        ys = np.round(np.arange(L.BOARD_Y0, L.BOARD_Y1 + 1e-9, step), 6)
        cells = {(i, j) for i, x in enumerate(xs) for j, y in enumerate(ys) if valid_centre(x, y)}
        by_q = {}
        for i, j in cells:
            by_q.setdefault(L.quarter_of(xs[i], ys[j]), set()).add((i, j))
        self.assertEqual(set(by_q), set(L.QUARTER_NAMES))
        for q, pts in by_q.items():
            todo, seen = [next(iter(pts))], set()
            while todo:  # flood fill: one connected patch
                c = todo.pop()
                if c in seen:
                    continue
                seen.add(c)
                todo += [(c[0] + di, c[1] + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if (c[0] + di, c[1] + dj) in pts]
            self.assertEqual(seen, pts, q)
            px = [xs[i] for i, _ in pts]
            py = [ys[j] for _, j in pts]
            self.assertAlmostEqual(min(px), WINDOW[q][0], places=4)
            self.assertAlmostEqual(max(px), WINDOW[q][1], places=4)
            self.assertAlmostEqual(min(py), WINDOW[q][2], places=4)
            self.assertAlmostEqual(max(py), WINDOW[q][3], places=4)
            self.assertGreaterEqual(max(px) - min(px), 0.014 - 1e-9, q)
            self.assertGreaterEqual(max(py) - min(py), 0.014 - 1e-9, q)
            cx, cy = CENTRE[q]
            self.assertTrue(WINDOW[q][0] + 0.007 <= cx + 1e-9 and cx <= WINDOW[q][1] - 0.007 + 1e-9, q)
            self.assertTrue(WINDOW[q][2] + 0.007 <= cy + 1e-9 and cy <= WINDOW[q][3] - 0.007 + 1e-9, q)
        # the windows are in different places in their quarters: an inner corner, an outer corner, the middle, the
        # middle of an outer edge
        rel = {q: ((CENTRE[q][0] - L.QUARTERS[q][0]) / 0.060, (CENTRE[q][1] - L.QUARTERS[q][2]) / 0.080)
               for q in L.QUARTER_NAMES}
        self.assertEqual(len({(round(a, 1), round(b, 1)) for a, b in rel.values()}), 4)

    def test_outline_of_a_pin(self):
        p = outline(pin(0.3, 0.0))
        r = np.hypot(p[:, 0] - 0.3, p[:, 1])
        self.assertGreaterEqual(r.min(), L.PIN_R - 1e-9)  # drawn round the circle, never inside it
        self.assertLess(r.max(), L.PIN_R * 1.003)
        lean = outline(pin(0.3, 0.0, tilt=10.0))  # a lean stretches it along the lean
        self.assertGreater(lean[:, 0].max() - lean[:, 0].min(), 0.020 + 0.0075)
        self.assertAlmostEqual(lean[:, 1].max() - lean[:, 1].min(), 0.020, delta=0.0002)
        flat = outline(lying(0.3, 0.0))
        self.assertAlmostEqual(flat[:, 0].max() - flat[:, 0].min(), L.PIN_H, delta=0.0002)

    def test_tilt_lowest_and_flip(self):
        self.assertAlmostEqual(tilt_deg(pin(0.3, 0.0)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(pin(0.3, 0.0, flip=True)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(pin(0.3, 0.0, tilt=7.0, lean_to=40)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(lying(0.3, 0.0)), 90.0, places=6)
        for s in (pin(0.3, 0.0), pin(0.3, 0.0, tilt=8.0), lying(0.3, 0.0), pin(0.3, 0.0, flip=True)):
            self.assertAlmostEqual(lowest(s), L.PLATE_TOP, places=9)

    def test_gap_to_rect(self):
        rect = (0.0, 0.01, 0.0, 0.01)
        self.assertAlmostEqual(gap_to_rect(outline(pin(0.025, 0.005)), rect), 0.005, delta=1e-4)
        self.assertEqual(gap_to_rect(outline(pin(0.015, 0.005)), rect), 0.0)
        self.assertAlmostEqual(gap_to_rect(outline(pin(0.03, 0.03)), rect), math.hypot(0.02, 0.02) - L.PIN_R,
                               delta=1e-4)
        self.assertEqual(gap_to_rect(outline(pin(0.005, 0.005)), rect), 0.0)  # inside it
        self.assertEqual(gap_to_rect(outline(pin(0.005, 0.005)), (0.0, 0.001, 0.0, 0.001)), 0.0)  # under it

    def test_inside_margin(self):
        self.assertAlmostEqual(inside_margin(outline(pin(0.255, 0.0))), 0.005, delta=1e-4)
        self.assertAlmostEqual(inside_margin(outline(pin(0.300, 0.075))), -0.005, delta=1e-4)

    def test_finger_zone_and_release(self):
        s = pin(*CENTRE["near_left"])
        top = lowest(s) + L.PIN_H
        holding = snap(left_tcp=(CENTRE["near_left"][0], CENTRE["near_left"][1], top - 0.013), left_open=0.21)
        self.assertLess(zone_gap(holding["left"], s), 0.0)
        self.assertFalse(released(holding, s))
        up = snap(left_tcp=(CENTRE["near_left"][0], CENTRE["near_left"][1], top + 0.06), left_open=0.43)
        self.assertGreater(zone_gap(up["left"], s), 0.0)
        self.assertTrue(released(up, s))
        self.assertTrue(released(None, s))  # unreadable arms: taken as let go


class Tiers(unittest.TestCase):
    def test_full_setup_scores_one(self):
        r = score_of(done())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertEqual(r.details["n_supported"], 4)
        self.assertEqual(r.details["touching_red"], [])
        self.assertEqual(r.details["at_rail"], [])

    def test_each_quarter_alone(self):
        for q in L.QUARTER_NAMES:
            with self.subTest(q=q):
                r = score_of(done([q]))
                self.assertEqual(r.score, 0.25, r.details)
                self.assertEqual(r.details["supported"], [q])
                self.assertFalse(r.success)

    def test_start_breaks_no_rule(self):
        r = score_of(start())
        self.assertEqual(r.score, 0.0)
        self.assertEqual((r.details["touching_red"], r.details["at_rail"], r.details["supported"]), ([], [], []))

    def test_every_tier_is_reachable(self):
        cases = {0.0: [], 0.25: ["far_right"], 0.5: ["near_left", "near_right"],
                 0.75: ["near_left", "far_left", "far_right"], 1.0: list(L.QUARTER_NAMES)}
        for want, qs in cases.items():
            with self.subTest(want=want):
                self.assertEqual(score_of(done(qs)).score, want)
        self.assertEqual(score_of(start()).score, 0.0)

    def test_anywhere_in_the_window(self):
        """A pin counts anywhere in its quarter's window, out to 0.1 mm from each corner (the outline is drawn as a
        48-sided polygon round the 20 mm circle, 0.02 mm bigger, so a pin exactly on a limit is just outside it)."""
        e = 0.0001
        for q, (x0, x1, y0, y1) in WINDOW.items():
            for x, y in ((x0 + e, y0 + e), (x0 + e, y1 - e), (x1 - e, y0 + e), (x1 - e, y1 - e),
                         ((x0 + x1) / 2, (y0 + y1) / 2)):
                with self.subTest(q=q, x=x, y=y):
                    self.assertTrue(valid_centre(x, y))
                    s = done([q])
                    s[PINS_FOR[q]] = pin(x, y)
                    self.assertEqual(score_of(s).score, 0.25)
                    self.assertEqual(score_of(s).details["supported"], [q])

    def test_just_outside_the_window(self):
        """0.5 mm past any side of a window the pin no longer counts (too close to red or to the board's edge); it
        does not zero the score either, since its outline is still off the red area."""
        for q, (x0, x1, y0, y1) in WINDOW.items():
            cx, cy = CENTRE[q]
            for x, y in ((x0 - 0.0005, cy), (x1 + 0.0005, cy), (cx, y0 - 0.0005), (cx, y1 + 0.0005)):
                with self.subTest(q=q, x=x, y=y):
                    s = done()
                    s[PINS_FOR[q]] = pin(x, y)
                    r = score_of(s)
                    self.assertEqual(r.score, 0.75, r.details["pins"][PINS_FOR[q]])
                    self.assertNotIn(q, r.details["supported"])

    def test_red_clearance_limit(self):
        # near right window: red resistor column at x 0.2500; a pin 3 mm clear counts, 2.5 mm clear does not (but
        # does not touch: no zero), touching scores 0
        x_clear = 0.2500 + L.PIN_R + 0.0030
        for dx, want in ((0.0001, 1.0), (-0.0005, 0.75), (-(0.0030 + 1e-4), 0.0)):
            with self.subTest(dx=dx):
                s = done()
                s["pin_right_1"] = pin(x_clear + dx, -0.0400)
                self.assertEqual(score_of(s).score, want)

    def test_inside_the_outline_limit(self):
        # near left window: the outline's centre line at x 0.240; the whole pin 5 mm inside counts, 4.5 mm does not
        for x, want in ((0.2551, 1.0), (0.2545, 0.75)):
            with self.subTest(x=x):
                s = done()
                s["pin_left_1"] = pin(x, 0.057)
                self.assertEqual(score_of(s).score, want)

    def test_tilt_limit(self):
        for tilt, want in ((9.5, 1.0), (10.5, 0.75)):
            with self.subTest(tilt=tilt):
                s = done()
                s["pin_right_2"] = pin(*CENTRE["far_right"], tilt=tilt, lean_to=90.0)
                self.assertEqual(score_of(s).score, want)

    def test_bottom_limit(self):
        for lift, want in ((0.0025, 1.0), (0.0035, 0.75)):
            with self.subTest(lift=lift):
                s = done()
                s["pin_left_2"] = pin(*CENTRE["far_left"], lift=lift)
                self.assertEqual(score_of(s).score, want)

    def test_spare_pins_are_allowed_anywhere_harmless(self):
        s = done()
        s["pin_left_3"] = pin(0.300, 0.098)  # on the plate, outside the outline, away from red and rails
        s["pin_right_3"] = lying(0.150, -0.150, base=L.TOP)  # knocked over on the table
        r = score_of(s)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)

    def test_two_pins_in_one_quarter_count_once(self):
        # two 20 mm pins fit in the near-left window only at opposite corners (centres 22 mm apart)
        s = start()
        s["pin_left_1"] = pin(0.2552, 0.0492)
        s["pin_left_3"] = pin(0.2708, 0.0648)
        r = score_of(s)
        self.assertEqual(r.score, 0.25, r.details)
        self.assertEqual(sorted(r.details["supporters"]["near_left"]), ["pin_left_1", "pin_left_3"])
        self.assertEqual(r.details["supported"], ["near_left"])
        s["pin_right_1"] = pin(*CENTRE["near_right"])
        s["pin_right_2"] = pin(*CENTRE["far_right"])
        r = score_of(s)
        self.assertEqual(r.score, 0.75)  # three quarters, whatever the fourth pin did: far left is still empty

    def test_upside_down_counts(self):
        s = done()
        s["pin_left_1"] = pin(*CENTRE["near_left"], flip=True)
        self.assertEqual(score_of(s).score, 1.0)

    def test_success_needs_stillness(self):
        t = Tracker()
        moving = done()
        moving["pin_right_2"] = pin(*CENTRE["far_right"], vel=(0.0, 0.06, 0.0))
        r = feed(t, moving, n=12)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)  # too fast
        t = Tracker()
        for k in range(12):  # creeping 0.4 mm a step: 4 mm over 0.5 s
            s = done(offsets={"far_right": (0.0, 0.0004 * k)})
            r = feed(t, s)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r = feed(t, done(offsets={"far_right": (0.0, 0.0044)}), n=11)
        self.assertTrue(r.success)
        self.assertFalse(feed(Tracker(), done(), n=10).success)  # not yet 0.5 s of path

    def test_score_is_idempotent(self):
        t = Tracker()
        arms = snap()
        feed(t, done(), arms, n=12)
        a = t.result(done(), arms)
        b = t.result(done(), arms)
        self.assertEqual((a.score, a.success), (b.score, b.success))
        self.assertEqual(t.steps, 12)


class ZeroConditions(unittest.TestCase):
    def test_pin_on_red_in_each_quarter(self):
        spots = {"far_left": (0.330, 0.058), "near_left": (0.270, 0.020), "near_right": (0.270, -0.010),
                 "far_right": (0.330, -0.020)}
        for q, (x, y) in spots.items():
            with self.subTest(q=q):
                s = done()
                s[PINS_FOR[q]] = pin(x, y, base=L.MARK_TOP)  # standing on the red area
                r = score_of(s)
                self.assertEqual(r.score, 0.0)
                self.assertEqual(r.details["touching_red"], [PINS_FOR[q]])

    def test_spare_pin_on_red_zeroes_a_full_setup(self):
        s = done()
        s["pin_left_3"] = pin(0.320, 0.060, base=L.MARK_TOP)
        r = score_of(s)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["n_supported"], 4)

    def test_just_touching_red(self):
        x = 0.2500 + L.PIN_R  # its outline touches the near-right resistor column's edge
        s = done()
        s["pin_right_1"] = pin(x, -0.0400)
        self.assertTrue(touches_red(s["pin_right_1"]))
        self.assertEqual(score_of(s).score, 0.0)
        s["pin_right_1"] = pin(x + 0.0005, -0.0400)
        self.assertFalse(touches_red(s["pin_right_1"]))
        self.assertEqual(score_of(s).score, 0.75)

    def test_leaning_over_red(self):
        # standing at the near-left window's edge next to the BGA: upright it counts; leaning 10 degrees over the
        # BGA, its top's shadow reaches 0.5 mm over the red area while its foot is on the plate: 0
        s = done()
        s["pin_left_1"] = pin(0.263, 0.0492)
        self.assertEqual(score_of(s).score, 1.0)
        s["pin_left_1"] = pin(0.263, 0.0492, tilt=10.0, lean_to=-90.0)
        self.assertTrue(touches_red(s["pin_left_1"]))
        r = score_of(s)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["touching_red"], ["pin_left_1"])
        s["pin_left_1"] = pin(0.263, 0.0492, tilt=9.0, lean_to=0.0)  # leaning along the BGA's edge instead
        self.assertFalse(touches_red(s["pin_left_1"]))
        self.assertEqual(score_of(s).score, 1.0)

    def test_fallen_pin_across_red(self):
        s = done()
        s["pin_left_3"] = lying(0.320, 0.060, base=L.MARK_TOP)
        self.assertEqual(score_of(s).score, 0.0)
        s["pin_left_3"] = lying(0.300, 0.097)  # fallen on the plate outside the outline, off red and rails
        self.assertEqual(score_of(s).score, 1.0)

    def test_pin_held_just_over_red(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = pin(0.330, 0.058, base=L.MARK_TOP, lift=0.003)  # 4 mm over the plate
        self.assertEqual(score_of(s).score, 0.0)
        self.assertTrue(touches_red(s["pin_left_2"]))
        s["pin_left_2"] = pin(0.330, 0.058, lift=0.020)  # carried 20 mm over it
        self.assertFalse(touches_red(s["pin_left_2"]))
        self.assertEqual(score_of(s).score, 0.75)

    def test_pin_at_a_rail(self):
        cases = {
            "on the near rail": pin(0.234, 0.050, base=L.RAIL_TOP),
            "on the far rail, leaning": pin(0.366, -0.030, base=L.RAIL_TOP, tilt=20.0),
            "standing on the plate 3 mm from the near rail": pin(0.250, 0.095),
            "leaning against the far rail": pin(0.350, 0.095, tilt=30.0, lean_to=0.0),
            "on the table outside the near rail": pin(0.222, 0.0, base=L.TOP),
            "lying across the far rail": lying(0.366, 0.100, yaw=0.0, base=L.RAIL_TOP),
        }
        for label, p in cases.items():
            with self.subTest(label):
                s = done()
                s["pin_left_3"] = p
                self.assertTrue(at_rail(p))
                r = score_of(s)
                self.assertEqual(r.score, 0.0)
                self.assertEqual(r.details["at_rail"], ["pin_left_3"])

    def test_not_at_a_rail(self):
        cases = {
            "6 mm from the near rail": pin(0.253, 0.095),
            "carried 10 mm over the far rail": pin(0.366, 0.050, base=L.RAIL_TOP, lift=0.010),
            "on the table past the rail's end": pin(0.234, 0.140, base=L.TOP),
        }
        for label, p in cases.items():
            with self.subTest(label):
                s = done()
                s["pin_left_3"] = p
                self.assertFalse(at_rail(p))
                self.assertEqual(score_of(s).score, 1.0)

    def test_board_centre_crosses_red(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = pin(0.300, 0.000, base=L.MARK_TOP)
        r = score_of(s)
        self.assertEqual(r.score, 0.0)  # the near-left BGA's corner is 5 mm from the board's centre


class Exploits(unittest.TestCase):
    def test_pin_still_held_does_not_count(self):
        s = done()
        x, y = CENTRE["far_left"]
        top = lowest(s["pin_left_2"]) + L.PIN_H
        arms = snap(left_tcp=(x, y, top - 0.013), left_open=0.21)
        r = score_of(s, arms)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["pins"]["pin_left_2"]["released"])
        arms = snap(left_tcp=(x, y, top + 0.05), left_open=0.43)  # lifted clear
        self.assertEqual(score_of(s, arms).score, 1.0)

    def test_hovering_open_gripper_next_to_a_pin_still_holds_it(self):
        s = done()
        x, y = CENTRE["near_right"]
        top = lowest(s["pin_right_1"]) + L.PIN_H
        arms = snap(right_tcp=(x - 0.03, y, top + 0.005), right_open=1.0)  # open, its pads beside the pin
        r = score_of(s, arms)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["pins"]["pin_right_1"]["released"])
        arms = snap(right_tcp=(x - 0.03, y, top + 0.075), right_open=1.0)  # 7 cm up: clear of it
        self.assertEqual(score_of(s, arms).score, 1.0)

    def test_open_gripper_hovering_over_a_pin_holds_it_up_to_the_zone_radius(self):
        """The finger zone is round at the tips too: an open gripper straight over a set pin holds it while its tips
        are less than the zone's radius (57 mm fully open, 31 mm at 40 mm) above the pin's top, and not beyond; the
        instruction therefore asks for both grippers to be moved away."""
        s = done()
        x, y = CENTRE["far_right"]
        top = lowest(s["pin_right_2"]) + L.PIN_H
        for opening, radius in ((1.0, 0.0566), (0.040 / L.FULL_GAP, 0.0311)):
            with self.subTest(opening=opening):
                near = snap(right_tcp=(x, y, top + radius - 0.002 + TIPS), right_open=opening)
                r = score_of(s, near)
                self.assertFalse(r.details["pins"]["pin_right_2"]["released"])
                self.assertNotIn("far_right", r.details["supported"])
                far = snap(right_tcp=(x, y, top + radius + 0.002 + TIPS), right_open=opening)
                self.assertEqual(score_of(s, far).score, 1.0)

    def test_stacked_pin_does_not_count_or_zero(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = pin(*CENTRE["near_left"], base=L.PLATE_TOP + L.PIN_H)  # on top of another pin
        r = score_of(s)
        self.assertEqual(r.score, 0.75)

    def test_lying_pin_in_a_quarter_does_not_count(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = lying(0.3168, 0.0178, yaw=90.0)  # in the far-left window but on its side
        r = score_of(s)
        self.assertLessEqual(r.score, 0.75)
        self.assertNotIn("far_left", r.details["supported"])

    def test_pin_straddling_a_quarter_line_counts_for_its_centre(self):
        s = done(["near_left", "near_right", "far_right"])
        # far-left window's near edge: the pin's outline reaches over the 3 mm quarter line at x 0.300, its centre
        # in the far-left quarter, 3.5 mm clear of the near-left BGA
        s["pin_left_2"] = pin(0.3100, 0.0178)
        self.assertLess(outline(s["pin_left_2"])[:, 0].min(), L.MID_X + L.LINE_W / 2)
        r = score_of(s)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertIn("pin_left_2", r.details["supporters"]["far_left"])

    def test_one_pin_cannot_support_two_quarters(self):
        s = start()
        s["pin_left_1"] = pin(0.3097, 0.0100)  # far left, as close to the other quarters as it may be
        r = score_of(s)
        self.assertEqual(r.details["supported"], ["far_left"])

    def test_pin_outside_the_outline_does_not_count(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = pin(0.320, 0.092)  # on the plate beyond the far-left quarter's edge
        r = score_of(s)
        self.assertEqual(r.score, 0.75)
        self.assertIsNone(r.details["pins"]["pin_left_2"]["quarter"])

    def test_pin_on_the_table_does_not_count(self):
        s = done(["near_left", "near_right", "far_right"])
        s["pin_left_2"] = pin(*CENTRE["far_left"], base=L.TOP)  # below the plate's top: not on the plate
        self.assertEqual(score_of(s).score, 0.75)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker()
        for _ in range(200):
            s = {}
            for n in L.PIN_NAMES:
                q = rng.normal(size=4)
                c = rng.uniform([0.1, -0.3, L.TOP - 0.01], [0.5, 0.3, L.TOP + 0.2])
                s[n] = state(c, q / np.linalg.norm(q), vel=rng.normal(size=3) * 0.1)
            r = t.update(s, snap())
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker()
        feed(t, done(), n=12)
        paths = {n: list(t.paths[n]) for n in L.PIN_NAMES}
        for bad in (np.nan, np.inf):
            s = done()
            s["pin_left_1"] = SimpleNamespace(**{**vars(s["pin_left_1"]), "pos": np.array([bad, 0.0, 0.8])})
            self.assertFalse(valid(s["pin_left_1"]))
            r = t.update(s, snap())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
        s = done()
        s["pin_right_2"] = SimpleNamespace(**{**vars(s["pin_right_2"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(t.update(s, snap()).score, 0.0)
        self.assertEqual(t.bad_steps, 3)
        self.assertEqual({n: list(t.paths[n]) for n in L.PIN_NAMES}, paths)
        r = feed(t, done())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_missing_pin_scores_zero(self):
        s = done()
        del s["pin_right_3"]
        r = score_of(s)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(score_of(None).score, 0.0)

    def test_unreadable_arms_are_taken_as_let_go_and_counted(self):
        t = Tracker()
        for _ in range(12):
            r = t.update(done(), None)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(t.arms_unavailable, 12)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker()
        good = snap()
        bad = {"left": rigstate.Arm(np.array([math.nan, 0, 0]), np.zeros(3), 1.0), "right": good["right"]}
        r = feed(t, done(), bad, n=12)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)

    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        feed(t, done(), n=20)
        self.assertEqual(t.steps, 20)
        feed(t, start())
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.paths["pin_left_1"]), 1)
        self.assertTrue(Tracker.at_start(start()))

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        feed(t, done(), n=5)
        a, b = np.array(CENTRE["near_left"]), np.array(L.STARTS["pin_left_1"])
        for k in range(1, 41):
            s = done()
            p = a + (b - a) * k / 40
            s["pin_left_1"] = pin(p[0], p[1], base=L.TRAY_TOP if k == 40 else L.PLATE_TOP + 0.05)
            feed(t, s)
        self.assertEqual(t.steps, 45)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        feed(t, done(), n=5)
        s = done()
        s["pin_left_1"] = pin(0.200, -0.040, base=L.TOP)
        feed(t, s)
        self.assertEqual(t.steps, 6)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = Tracker()
        feed(t, done(), n=12)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.steps, 0)
            self.assertEqual(len(t.paths["pin_left_1"]), 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)

    def test_pin_report_fields(self):
        rep = pin_report(done()["pin_left_1"], snap())
        for k in ("quarter", "supports", "upright", "on_plate", "released", "touches_red", "at_rail", "tilt_deg",
                  "bottom_mm", "inside_mm", "red_mm", "red_nearest", "rail_mm", "rail_nearest"):
            self.assertIn(k, rep)
        self.assertEqual(supports_quarter(done()["pin_left_1"], snap()), "near_left")
        self.assertAlmostEqual(red_gap(done()["pin_left_1"])[0], 0.011, delta=0.0002)  # 3 mm + 8 mm to spare


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted(L.PIN_NAMES))
        for o in loose:
            self.assertAlmostEqual(o.radius, L.PIN_R)
            self.assertAlmostEqual(o.height, L.PIN_H)
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TRAY_TOP)
            self.assertLessEqual(o.mass, 0.3)
        self.assertEqual(self.task.SCORE_OVER, "end")
        reds = [o for o in self.task.OBJECTS if o.name.startswith("red_")]
        self.assertEqual(len(reds), len(L.RED))
        for o in reds:
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.PLATE_TOP)
        self.assertTrue(Tracker.at_start(start()))

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        for _ in range(12):
            self.task.success(done())
        self.assertEqual(self.task.score(done()), 1.0)
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(start()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertEqual(self.task.score(start()), 0.0)

    def test_hooks_ignore_fixed_objects(self):
        rigstate.offline_reset()
        snap()
        s = done()
        s["support_plate"] = SimpleNamespace(pos=np.array([math.nan] * 3))  # fixed objects are not read
        s["red_fl_processor"] = None
        self.assertFalse(self.task.success(s))
        self.assertEqual(self.task.score(s), 1.0)

    def test_trace(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                rigstate.offline_reset()
                snap()
                self.task.success(done())
                os.environ["TASK_TRACE"] = os.path.join(d, "no_such_dir", "trace.jsonl")
                self.assertFalse(self.task.success(start()))  # a bad path never stops a trial
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
            import json
            with open(path) as f:
                row = json.loads(f.readline())
        for k in ("step", "score", "supported", "touching_red", "at_rail", "pins", "zones_mm", "pin_left_1_pos",
                  "left_tcp"):
            self.assertIn(k, row)
        self.assertIn("red_mm", row["pins"]["pin_left_1"])

    def test_demo_places_score_one(self):
        """Where the demo lets go of the four pins scores 1.0 with no sag, with the sag the compliance model
        predicts (about 1 mm toward the arms and 1 mm toward the arm's own side), and with 4 mm off any way."""
        t = self.task
        self.assertEqual({q: tuple(v) for q, v in t.TARGETS.items()}, CENTRE)
        offsets = [(0.0, 0.0), (-0.001, 0.001), (-0.001, -0.001)] + \
            [(0.004 * math.cos(a), 0.004 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        for dx, dy in offsets:
            with self.subTest(dx=dx, dy=dy):
                s = start()
                for q, (x, y) in t.TARGETS.items():
                    side = 1.0 if q.endswith("left") else -1.0  # a left-arm pin sags toward +y, a right-arm one -y
                    s[PINS_FOR[q]] = pin(x + dx, y + dy * side)
                r = score_of(s)
                self.assertEqual(r.score, 1.0, r.details)
                self.assertGreaterEqual(min(p["red_mm"] for n, p in r.details["pins"].items()
                                            if p["supports"]), 3.0)

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger and prev.pos == w.pos:  # every close and open takes 1.5 s
                    self.assertGreaterEqual(w.duration, 1.5)
                if prev.pos[:2] == w.pos[:2] and prev.pos[2] > w.pos[2] and w.pos[2] < L.TOP + 0.06 \
                        and prev.pos[2] - w.pos[2] <= 0.0205:  # every last 2 cm down onto a surface is slow
                    self.assertGreaterEqual(w.duration, 1.0)
        # one gripper over the board at a time: from the start of each carry over a window to the end of the lift
        # after letting go, the two arms' spans never overlap
        spans = {}
        for a in plan:
            t, cur = 0.0, None
            for w in plan[a]:
                if w.label.startswith("carry over the"):
                    cur = t
                t += w.duration
                if w.label == "straight up" and cur is not None:
                    spans.setdefault(a, []).append((cur, t))
                    cur = None
        self.assertEqual(len(spans["left"]), 2)
        self.assertEqual(len(spans["right"]), 2)
        for a0, a1 in spans["left"]:
            for b0, b1 in spans["right"]:
                self.assertTrue(a1 <= b0 + 1e-9 or b1 <= a0 + 1e-9, (spans,))
        # each arm takes pins only from its own tray
        for a in plan:
            for w in plan[a]:
                if w.label.startswith("over pin_"):
                    self.assertIn(f"pin_{a}_", w.label)
        total = max(sum(w.duration for w in plan[a]) for a in plan)
        self.assertLess(total * HZ, 1500 * 0.75)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_ends_clear_of_every_pin(self):
        """Where each arm's plan ends (parked, fingers 40 mm apart), its finger zone is far from every pin as the
        demo leaves them; so is an idle arm at its ready pose (fully open)."""
        t = self.task
        s = start()
        for q, (x, y) in t.TARGETS.items():
            s[PINS_FOR[q]] = pin(x, y)
        plan = t.demo()
        for side, wps in plan.items():
            last = wps[-1]
            tcp = np.array(last.pos, float)
            arms = snap(**{f"{side}_tcp": tcp, f"{side}_open": -last.finger / -t.OPEN})
            for n in L.PIN_NAMES:
                self.assertGreater(zone_gap(arms[side], s[n]), 0.03, (side, n))
        arms = snap()
        for side in ("left", "right"):
            for n in L.PIN_NAMES:
                self.assertGreater(zone_gap(arms[side], s[n]), 0.03, (side, n))
        self.assertEqual(score_of(s, snap()).score, 1.0)

    def test_carried_pin_clears_the_standing_pins_and_rails(self):
        t = self.task
        self.assertGreaterEqual(t.CARRY_Z - t.GRIP_UP - (L.TRAY_TOP + L.PIN_H), 0.03)
        self.assertGreaterEqual(t.CARRY_Z - t.GRIP_UP - (L.PLATE_TOP + L.PIN_H), 0.03)
        self.assertGreater(t.PLACE_Z - 0.013 - L.RAIL_TOP, 0.0)  # pad tips above the rails' tops when placing
        self.assertAlmostEqual(t.PLACE_Z - t.GRIP_UP - L.PLATE_TOP, 0.001)  # let go 1 mm above the plate
        self.assertAlmostEqual(t.PICK_Z - t.GRIP_UP, L.TRAY_TOP)
        self.assertAlmostEqual(-2 * t.PIN_GRIP, 2 * L.PIN_R - 0.008)  # commanded 8 mm narrower than the pin
        self.assertGreaterEqual(-2 * t.LET_GO, 2 * L.PIN_R + 0.015)  # let go: 10 mm or more clear each side
        self.assertAlmostEqual(t.OPEN, rigstate.FINGER_OPEN)
        self.assertGreater(t.GRIP_UP - 0.013, 0.010)  # pad tips well above the plate and the tray
        self.assertLess(t.GRIP_UP - 0.013 + 0.038, L.PIN_H + 0.013)  # pads on the pin's upper part
        # tray neighbours along x: the pads (27 mm across, closing along y) clear the next pin by 15 mm
        self.assertGreaterEqual(min(abs(a - b) for a in L.PIN_XS for b in L.PIN_XS if a != b) - L.PIN_R
                                - L.FINGER_HALF_WIDTH, 0.015)

    def test_instruction_is_the_same_everywhere(self):
        import json
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        # 'let go' needs the finger zones clear (see the hovering tests), so the instruction asks for it
        self.assertTrue(self.task.INSTRUCTION.endswith("let go of them and move both grippers away from the pins."))


class TaskJson(unittest.TestCase):
    """task.json's setup and scoring text says what layout.py and scoring.py do."""

    @classmethod
    def setUpClass(cls):
        import json
        with open(HERE / "task.json") as f:
            cls.spec = json.load(f)

    def test_setup_lists_every_red_area(self):
        def cm(v):
            return f"{v * 100:.2f}"
        setup = self.spec["setup"]
        self.assertIn("eleven red rectangles", setup)
        self.assertEqual(len(L.RED), 11)
        for name, x0, x1, y0, y1 in L.RED:
            ys = f"y {cm(y0)}-{cm(y1)}" if y0 >= 0 else f"y {cm(y0)} to {cm(y1)}"
            self.assertIn(f"x {cm(x0)}-{cm(x1)}, {ys}", setup, name)

    def test_setup_numbers(self):
        setup = self.spec["setup"]
        for text in ("124 x 220 x 5 mm", "centred at x 30.0 cm, y 0", "x 23.1-23.7 cm and x 36.3-36.9 cm",
                     "6 mm wide, 240 mm long (along y) and 20 mm tall", "120 x 160 mm rectangle at x 24.0-36.0 cm",
                     "3 mm wide and 1 mm thick", "140 x 40 x 3 mm", "x 19.0 cm, y +23.0 cm and y -23.0 cm",
                     "20 mm across and 45 mm tall, 110 g", "x 23.0, 19.0 and 15.0 cm"):
            self.assertIn(text, setup)
        self.assertAlmostEqual(L.RAILS[0][1] - L.RAIL_W / 2, 0.231)
        self.assertAlmostEqual(L.RAILS[1][1] + L.RAIL_W / 2, 0.369)
        self.assertEqual(L.PIN_XS, (0.230, 0.190, 0.150))
        self.assertEqual((L.TRAY_X, L.TRAY_Y), (0.190, 0.230))
        for q, (x, y) in CENTRE.items():  # the clear spots named in the setup
            self.assertIn(f"x {x * 100:.1f}, y {y * 100:+.1f}", setup, q)

    def test_scoring_thresholds(self):
        sc = self.spec["scoring"]
        for text, value in (("within 10 degrees of vertical", L.TILT_MAX_DEG == 10.0),
                            ("within 3 mm of the plate's top", L.BOTTOM_TOL == 0.003),
                            ("within 5 mm of the plate's top", L.TOUCH_LOW == 0.005),
                            ("within 5 mm of a rail's", L.RAIL_NEAR == 0.005),
                            ("at most 5 mm above the rail's top", L.RAIL_ABOVE == 0.005),
                            ("at least 5 mm inside the board's edge", L.EDGE_IN == 0.005),
                            ("at least 3 mm from every red area", L.RED_CLEAR == 0.003),
                            ("the 38 mm segment", L.FINGER_LEN == 0.038),
                            ("2 mm or less over the last 0.5 s", L.STILL_PATH == 0.002 and L.STILL_WINDOW_S == 0.5),
                            ("Possible scores: 0, 0.25, 0.5, 0.75, 1.0", True)):
            self.assertIn(text, sc)
            self.assertTrue(value, text)
        self.assertNotIn("Dream", self.spec["instruction"])
        self.assertIn("slower than 5 cm/s", sc)
        self.assertEqual(L.STILL_SPEED, 0.05)
        self.assertIn("round at the finger tips too", sc)

    def test_window_sizes_in_the_text(self):
        """The setup and README give the windows' size: 14 to 16.5 mm on a side (the 0.5 mm grid's extents)."""
        sides = [v for q in WINDOW for v in (WINDOW[q][1] - WINDOW[q][0], WINDOW[q][3] - WINDOW[q][2])]
        self.assertAlmostEqual(min(sides), 0.0140, places=4)
        self.assertAlmostEqual(max(sides), 0.0165, places=4)
        self.assertIn("a rectangle 14 to 16.5 mm on a side", self.spec["setup"])
        readme = " ".join((HERE / "README.md").read_text().split())
        self.assertIn("a rectangle 14 to 16.5 mm on a side", readme)

    def test_text_has_no_internal_references(self):
        """Blind review: the submitted text names no tool path, pipeline file or other task of ours."""
        readme = (HERE / "README.md").read_text()
        text = readme + " ".join(str(v) for v in self.spec.values())
        for bad in ("tools/", "pipeline/", "check_task", "heatsink", "T-70"):
            self.assertNotIn(bad, text)
        self.assertNotRegex(text, r"(?i)\bour\b")


if __name__ == "__main__":
    unittest.main()
