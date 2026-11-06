"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic pin states and synthetic arm snapshots
fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    CHAMBERS, Tracker, centre_in_opening, chamber_id, over_opening, shadow, tilt_deg, top_centre, top_offset, valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 control steps
SETTLE = WINDOW + 2  # enough still steps for the 0.5 s path window to fill
AWAY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis given as a 3-vector."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.array([*(math.sin(h) * a), math.cos(h)])


def axis_of(q):
    x, y, z, w = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def cyl(center, size, quat=Q0, vel=(0, 0, 0)):
    """An ObjectState-like key pin: centre, orientation, velocities, world bounding box of the turned cylinder."""
    c, q = np.array(center, float), np.array(quat, float)
    u = axis_of(q / np.linalg.norm(q)) if np.linalg.norm(q) > 0 else np.array([0.0, 0.0, 1.0])
    h, r = L.PIN_LEN[size] / 2, L.PIN_R
    ext = h * np.abs(u) + r * np.sqrt(np.clip(1 - u * u, 0, 1))
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - ext, high=c + ext)


def size_of(name):
    return L.PINS[name][1]


def at_start(name):
    x, y = L.PIN_START[name]
    return cyl((x, y, L.TOP + L.PIN_LEN[size_of(name)] / 2), size_of(name))


def start_pins():
    return {n: at_start(n) for n in L.PINS}


def in_chamber(name, side, i, *, dx=0.0, dy=0.0, top=0.0, tilt=0.0, tilt_axis=(0, 1, 0), flip=False):
    """Pin ``name`` in chamber i of plug ``side``: its centre (dx, dy) off the chamber's centre, its top-face centre
    ``top`` above the shear line if it stood on the floor (top=None: standing on the floor block), tilted."""
    size = size_of(name)
    q = about(tilt_axis, tilt) if tilt else Q0.copy()
    if flip:  # upside down: half a turn about x first
        qf = about((1, 0, 0), 180.0)
        x1, y1, z1, w1 = q
        x2, y2, z2, w2 = qf
        q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    u = axis_of(q)
    u_up = u if u[2] >= 0 else -u
    if top is None:  # resting on the floor block on its lower rim
        h, r = L.PIN_LEN[size] / 2, L.PIN_R
        bottom = L.TOP + L.FLOOR_H[i] + r * math.sqrt(max(0.0, 1 - u_up[2] ** 2))
        cz = bottom + h * u_up[2]
    else:
        cz = L.PLUG_TOP_Z + top - (L.PIN_LEN[size] / 2) * u_up[2]
    return cyl((L.CHAMBER_X[i] + dx, L.PLUG_Y[side] + dy, cz), size, q)


def correct(side, i, column=None, **kw):
    """The code's pin for chamber i, from the given kit column (default: the plug's own)."""
    return in_chamber(L.pin_name(column or side, L.CODE[i]), side, i, **kw)


TOWARD = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))  # lean directions: away from each of the four walls
MU = (1.0, 1.2)  # the simulator's one friction: dynamic 1.0, static 1.2


def caught(name, side, i, deg, toward):
    """Pin ``name`` in chamber i caught leaning: it stands on one bottom-rim point on the floor block, its opposite
    bottom corner pressed against the chamber wall behind it, leaning ``deg`` degrees away from that wall, toward
    ``toward`` (a horizontal unit 2-vector)."""
    size = size_of(name)
    r, ln, th = L.PIN_R, L.PIN_LEN[size], math.radians(deg)
    d = np.array([toward[0], toward[1], 0.0])
    q = about(np.cross([0.0, 0.0, 1.0], d), deg)
    u = math.sin(th) * d + np.array([0.0, 0.0, math.cos(th)])
    wall = np.array([L.CHAMBER_X[i], L.PLUG_Y[side], 0.0]) - L.CHAMBER_HALF * d  # the wall face behind the pin
    bottom = wall + r * math.cos(th) * d  # bottom-face centre: its corner on the wall, its rim point on the floor
    bottom[2] = L.TOP + L.FLOOR_H[i] + r * math.sin(th)
    return cyl(bottom + (ln / 2) * u, size, q)


def fits_caught(size, deg):
    """The caught pin fits across the chamber: its far top corner stays inside the 34 mm opening."""
    th = math.radians(deg)
    return 2 * L.PIN_R * math.cos(th) + L.PIN_LEN[size] * math.sin(th) <= L.CHAMBER


def caught_lean_holds(size, deg, mu):
    """Static check of a caught lean (one friction mu at the floor and the wall). Gravity turns the pin back
    toward upright about its rim point, which would push its corner into the wall, so it can right itself only if
    the rim slides; it stays when the floor's friction can carry the wall's push."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    a = L.PIN_R * c - (L.PIN_LEN[size] / 2) * s  # its centre of mass on the wall side of the rim point
    return a > 0 and a / (2 * L.PIN_R * (s + mu * c)) <= mu / (1 + mu * mu)


def caught_leans(size, mu):
    """Every caught lean (0.5 degree steps) that fits the chamber and stays."""
    return [d / 2 for d in range(1, 180) if fits_caught(size, d / 2) and caught_lean_holds(size, d / 2, mu)]


def keyed(chambers=CHAMBERS):
    """Start layout with the code's pin standing on the floor of each listed chamber."""
    pins = start_pins()
    for side, i in chambers:
        pins[L.pin_name(side, L.CODE[i])] = correct(side, i, top=None)
    return pins


def snap(left=None, right=None):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp in (("left", left), ("right", right)):
        tcp = AWAY[side] if tcp is None else np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, 0.013]), "opening": 1.0}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def hold(pins, n=SETTLE, t=None, arms="away"):
    """Feed the same pins n steps (arms away unless given); return the last result and the tracker."""
    t = t or Tracker(HZ)
    r = None
    for _ in range(n):
        a = snap() if isinstance(arms, str) else arms
        r = t.update(pins, a)
    return r, t


def score_of(pins, **kw):
    return hold(pins, **kw)[0].score


def row(result, cid):
    return result.details["chambers"][cid]


class Geometry(unittest.TestCase):
    def test_only_the_codes_size_is_flush_in_each_chamber(self):
        for i in range(4):
            for size in L.SIZES:
                off = L.FLOOR_H[i] + L.PIN_LEN[size] - L.PLUG_H
                if size == L.CODE[i]:
                    self.assertAlmostEqual(off, 0.0, places=9)
                else:
                    self.assertGreaterEqual(abs(off), 0.006 - 1e-9)

    def test_seated_pin_reads_flush_and_upright(self):
        s = correct("left", 0, top=None)
        self.assertAlmostEqual(top_offset(s, 3), 0.0, places=9)
        self.assertAlmostEqual(tilt_deg(s), 0.0, places=6)
        self.assertTrue(centre_in_opening(s, "left", 0))
        self.assertFalse(centre_in_opening(s, "left", 1))
        self.assertFalse(centre_in_opening(s, "right", 0))

    def test_either_end_up(self):
        s = correct("left", 2, top=None, flip=True)
        self.assertAlmostEqual(tilt_deg(s), 0.0, places=6)
        self.assertAlmostEqual(top_offset(s, 4), 0.0, places=9)

    def test_lean_moves_the_top_at_most_1_4_mm(self):
        """Resting on its lower rim, an 8 degree lean raises the top-face centre by at most 1.4 mm, so a pin one
        size off (6 mm) stays at least 4.6 mm from the shear line."""
        for i, size in itertools.product(range(4), L.SIZES):
            name = L.pin_name("left", size)
            up = in_chamber(name, "left", i, top=None)
            lean = in_chamber(name, "left", i, top=None, tilt=8.0)
            d = top_offset(lean, size) - top_offset(up, size)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 0.0014 + 1e-6)
            if size != L.CODE[i]:
                self.assertGreater(abs(top_offset(lean, size)), 0.0046 - 1e-6)
                self.assertGreater(abs(top_offset(up, size)), L.FLUSH_TOL)

    def test_lean_cap_is_over_8_deg_and_below_every_tipping_angle(self):
        """In a 34 mm chamber any pin can lean at most 13-27 degrees (its side meets the far wall's face, or its top
        edge for a pin taller than the chamber is deep): more than the 8 degrees rule (a) allows, so that rule is
        needed, and below that pin's tipping angle, so a standing pin cannot fall flat in a chamber. It does NOT
        mean a crooked pin rights itself: one caught with its foot against a wall can stay leaning (see
        Exploits.test_a_pin_caught_leaning_against_a_chamber_wall)."""
        caps = []
        for i, size in itertools.product(range(4), L.SIZES):
            depth, ln, d = L.PLUG_H - L.FLOOR_H[i], L.PIN_LEN[size], 2 * L.PIN_R

            def extent(th):
                return d * math.cos(th) + min(ln, depth / math.cos(th)) * math.sin(th)

            lo, hi = 0.0, math.radians(60.0)
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if extent(mid) < L.CHAMBER else (lo, mid)
            cap = math.degrees(lo)
            caps.append(cap)
            self.assertGreater(cap, L.TILT_MAX_DEG)
            self.assertLess(cap, math.degrees(math.atan2(d, ln)))
        self.assertAlmostEqual(min(caps), 13.0, delta=0.1)
        self.assertAlmostEqual(max(caps), 26.6, delta=0.1)

    def test_chamber_opening_edges(self):
        h = L.CHAMBER_HALF
        self.assertTrue(centre_in_opening(correct("left", 1, dx=h - 1e-4), "left", 1))
        self.assertFalse(centre_in_opening(correct("left", 1, dx=h + 1e-4), "left", 1))
        self.assertTrue(centre_in_opening(correct("right", 3, dy=-(h - 1e-4)), "right", 3))
        self.assertFalse(centre_in_opening(correct("right", 3, dy=-(h + 1e-4)), "right", 3))

    def test_shadow_of_a_standing_and_a_lying_pin(self):
        s = cyl((0.3, 0.0, 0.8), 5)
        poly = shadow(s, 5)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(poly - [0.3, 0.0], axis=1))), L.PIN_R, places=6)
        lying = cyl((0.3, 0.0, 0.8), 5, about((1, 0, 0), 90.0))  # axis along y
        poly = shadow(lying, 5)
        self.assertAlmostEqual(float(poly[:, 1].max() - poly[:, 1].min()), L.PIN_LEN[5] + 0.0, delta=1e-6)
        self.assertAlmostEqual(float(poly[:, 0].max() - poly[:, 0].min()), 2 * L.PIN_R, delta=2e-4)

    def test_neighbouring_chambers_never_overlap(self):
        """A pin anywhere inside one chamber (5 mm play a side) never reaches over the next chamber's opening."""
        for side, i in CHAMBERS:
            for dx, dy in itertools.product((-0.005, 0.0, 0.005), repeat=2):
                s = correct(side, i, dx=dx, dy=dy, top=None)
                size = L.CODE[i]
                self.assertTrue(over_opening(s, size, side, i))
                for side2, j in CHAMBERS:
                    if (side2, j) != (side, i):
                        self.assertFalse(over_opening(s, size, side2, j))

    def test_pins_in_the_kit_are_over_no_chamber(self):
        pins = start_pins()
        for n, s in pins.items():
            for side, i in CHAMBERS:
                self.assertFalse(over_opening(s, size_of(n), side, i))
                self.assertFalse(centre_in_opening(s, side, i))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = hold(start_pins())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["k"], 0)

    def test_every_count_maps_to_k_over_8(self):
        seen = set()
        order = list(CHAMBERS)
        for k in range(9):
            r, _ = hold(keyed(order[:k]))
            self.assertEqual(r.details["k"], k)
            self.assertAlmostEqual(r.score, k / 8)
            self.assertEqual(r.success, k == 8)
            seen.add(r.score)
        self.assertEqual(seen, {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0})

    def test_any_subset_scores_its_size(self):
        for subset in [(("left", 1), ("right", 2)), (("right", 0), ("right", 1), ("right", 2), ("right", 3)),
                       (("left", 3),), tuple(c for c in CHAMBERS if c != ("left", 0))]:
            self.assertAlmostEqual(score_of(keyed(subset)), len(subset) / 8)

    def test_one_plug_alone_is_half(self):
        self.assertEqual(score_of(keyed([c for c in CHAMBERS if c[0] == "left"])), 0.5)
        self.assertEqual(score_of(keyed([c for c in CHAMBERS if c[0] == "right"])), 0.5)


class Rules(unittest.TestCase):
    def one(self, side, i, pin_state, name=None, **kw):
        """Score with only one chamber filled by ``pin_state`` (the code's pin unless ``name``)."""
        pins = start_pins()
        pins[name or L.pin_name(side, L.CODE[i])] = pin_state
        r, _ = hold(pins, **kw)
        return r

    def test_a_centre_must_be_inside_the_opening(self):
        r = self.one("left", 0, correct("left", 0, dx=0.0049, top=None))
        self.assertEqual(r.details["k"], 1)
        # standing on the table beside the plug, its top somewhere: not in a chamber at all
        r = self.one("left", 0, cyl((0.234, 0.180, L.TOP + 0.020), 3))
        self.assertEqual(r.details["k"], 0)

    def test_a_tilt_at_most_8_deg(self):
        for side, i in CHAMBERS:
            ok = self.one(side, i, correct(side, i, top=0.0, tilt=7.9))
            self.assertEqual(ok.details["k"], 1, chamber_id(side, i))
            bad = self.one(side, i, correct(side, i, top=0.0, tilt=8.1, tilt_axis=(1, 0, 0)))
            self.assertEqual(bad.details["k"], 0)
            self.assertIn("upright", row(bad, chamber_id(side, i))["why"])

    def test_b_flush_within_2_5_mm(self):
        for top, k in ((0.0024, 1), (-0.0024, 1), (0.0026, 0), (-0.0026, 0)):
            r = self.one("right", 1, correct("right", 1, top=top))
            self.assertEqual(r.details["k"], k, top)
            if not k:
                self.assertIn("flush", row(r, "R2")["why"])

    def test_c_still_path_over_half_a_second(self):
        t = Tracker(HZ)
        pins = start_pins()
        name = L.pin_name("left", L.CODE[2])
        # it has just arrived: not still until 0.5 s of positions are seen
        pins[name] = correct("left", 2, top=None)
        r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 0)
        self.assertIn("still", row(r, "L3")["why"])
        for _ in range(WINDOW - 1):
            r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 0)
        r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 1)
        # creeping 0.25 mm a step (path 2.5 mm in 0.5 s) does not count, even with a tiny velocity
        for j in range(SETTLE):
            pins[name] = correct("left", 2, top=None, dx=0.00025 * (j % 8), dy=0.0)
            r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 0)
        # jitter of 0.1 mm a step (path 1 mm in 0.5 s) does, whatever the reported velocity
        for j in range(SETTLE):
            s = correct("left", 2, top=None, dx=0.0001 * (j % 2))
            s.lin_vel = np.array([0.04, -0.03, 0.02])  # a chattering velocity reading is ignored
            pins[name] = s
            r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 1)

    def test_c_path_edge_2_mm(self):
        """A steady creep of 0.19 mm a step (1.9 mm in 0.5 s) is still; 0.21 mm a step (2.1 mm) is not."""
        for per_step, k in ((0.00019, 1), (0.00021, 0)):
            t = Tracker(HZ)
            pins = start_pins()
            name = L.pin_name("right", L.CODE[1])
            for j in range(SETTLE):
                pins[name] = correct("right", 1, top=None, dx=-0.003 + per_step * j)
                r = t.update(pins, snap())
            self.assertEqual(r.details["k"], k, per_step)
            self.assertAlmostEqual(row(r, "R2")["path_mm"], per_step * WINDOW * 1000, delta=0.01)
            # the trace-only diagnostics: a steady creep's net displacement is its path, its spread half of it
            self.assertAlmostEqual(row(r, "R2")["net_mm"], per_step * WINDOW * 1000, delta=0.1)
            self.assertAlmostEqual(row(r, "R2")["spread_mm"], per_step * WINDOW * 1000 / 2, delta=0.1)

    def test_c_frame_noise_and_the_drift_diagnostics(self):
        """Review finding (a GPU watch item, not a known bug): (c) sums the path, so frame-to-frame position noise on
        a pin at rest adds up. Gaussian noise of 0.1 mm per axis per step about a fixed point sums to more than 2 mm
        in 0.5 s and the chamber drops out; 0.03 mm stays well under. The trace's net_mm and spread_mm (never
        scored) stay under 1 mm for both: the first GPU demo's trace reads all three for the eight released pins,
        and if resting noise is over about 0.5 mm, (c) becomes net displacement or spread under 2 mm."""
        for sigma, k in ((0.00003, 1), (0.0001, 0)):
            rng = np.random.default_rng(0)
            t = Tracker(HZ)
            pins = start_pins()
            name = L.pin_name("left", L.CODE[0])
            base = correct("left", 0, top=None)
            for _ in range(3 * WINDOW):
                s = correct("left", 0, top=None)
                s.pos = base.pos + rng.normal(0.0, sigma, 3)
                pins[name] = s
                r = t.update(pins, snap())
            c = row(r, "L1")
            self.assertEqual(r.details["k"], k, (sigma, c))
            self.assertEqual(c["still"], k == 1)
            self.assertTrue(c["upright"] and c["flush"] and c["let_go"] and c["clear"])
            self.assertLess(c["net_mm"], 1.0)
            self.assertLess(c["spread_mm"], 1.0)
            if k == 0:
                self.assertGreater(c["path_mm"], 2.0)
            else:
                self.assertLess(c["path_mm"], 1.0)

    def test_c_drift_is_inf_until_half_a_second_is_seen(self):
        t = Tracker(HZ)
        pins = keyed()
        r = t.update(pins, snap())
        self.assertIsNone(row(r, "L1")["net_mm"])  # inf, written as null in the trace
        self.assertIsNone(row(r, "L1")["spread_mm"])
        self.assertEqual(t.drift(L.pin_name("left", L.CODE[0])), (math.inf, math.inf))
        r, _ = hold(pins, n=WINDOW, t=t)
        self.assertEqual(row(r, "L1")["net_mm"], 0.0)
        self.assertEqual(row(r, "L1")["spread_mm"], 0.0)

    def test_d_let_go_50_mm(self):
        side, i = "left", 3
        s = correct(side, i, top=None)
        tc = top_centre(s, L.CODE[i])
        for arm in ("left", "right"):
            near = {arm: tc + [0, 0, 0.049]}
            r = self.one(side, i, s, arms=snap(**near))
            self.assertEqual(r.details["k"], 0)
            self.assertIn("let_go", row(r, "L4")["why"])
            far = {arm: tc + [0, 0, 0.051]}
            self.assertEqual(self.one(side, i, s, arms=snap(**far)).details["k"], 1)
        # sideways counts as much as up: 49 mm off along x
        self.assertEqual(self.one(side, i, s, arms=snap(left=tc + [0.049, 0, 0])).details["k"], 0)

    def test_d_unreadable_arms_give_no_credit(self):
        r, t = hold(keyed(), arms=None)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(t.arms_unavailable, SETTLE)
        broken = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0])), "right": AWAY["right"]}
        self.assertEqual(hold(keyed(), arms=broken)[0].score, 0.0)
        r, _ = hold(keyed(), n=1, t=t)  # readable again: back to the end state's score
        self.assertEqual(r.score, 1.0)

    def test_e_nothing_else_over_the_chamber(self):
        pins = keyed()
        spare = L.pin_name("left", 5)
        # a spare stacked on the flush pin in L1
        pins[spare] = cyl((L.CHAMBER_X[0], L.PLUG_Y["left"], L.PLUG_TOP_Z + L.PIN_LEN[5] / 2), 5)
        r, _ = hold(pins)
        self.assertEqual(r.details["k"], 7)
        self.assertNotIn("L1", r.details["pinned"])
        self.assertEqual(row(r, "L1")["why"], "two pins")
        self.assertFalse(r.success)

    def test_e_a_pin_lying_across_the_plug_top(self):
        pins = keyed()
        spare = L.pin_name("right", 5)
        # lying along x on the plug top, centred over the wall between chambers 1 and 2 (its centre in neither)
        x_wall = L.CROSS_WALL_X[1]
        pins[spare] = cyl((x_wall, L.PLUG_Y["right"], L.PLUG_TOP_Z + L.PIN_R), 5, about((0, 1, 0), 90.0))
        r, _ = hold(pins)
        self.assertEqual(r.details["k"], 6)
        self.assertEqual(set(r.details["pinned"]), {"L1", "L2", "L3", "L4", "R3", "R4"})
        self.assertIn(spare, row(r, "R1")["others_over"])
        # lying along x on the near side wall's top, overlapping all four openings by 8 mm
        pins[spare] = cyl((0.300, L.PLUG_Y["left"] - L.SIDE_WALL_OFF, L.PLUG_TOP_Z + L.PIN_R), 5,
                          about((0, 1, 0), 90.0))
        r, _ = hold(pins)
        self.assertEqual(r.details["k"], 6)  # it spans x 0.274-0.326 and reaches 8 mm in: chambers 2 and 3
        self.assertEqual(set(r.details["pinned"]), {"L1", "L4", "R1", "R2", "R3", "R4"})

    def test_e_a_pin_standing_on_a_wall_top(self):
        pins = keyed()
        spare = L.pin_name("left", 5)
        pins[spare] = cyl((L.CROSS_WALL_X[2], L.PLUG_Y["left"], L.PLUG_TOP_Z + L.PIN_LEN[5] / 2), 5)
        r, _ = hold(pins)
        self.assertNotIn("L2", r.details["pinned"])
        self.assertNotIn("L3", r.details["pinned"])
        self.assertEqual(r.details["k"], 6)

    def test_e_a_pin_held_high_over_a_chamber(self):
        """Seen from above at any height: a pin still held over a chamber at the end hides it from the photo."""
        pins = keyed()
        spare = L.pin_name("left", 5)
        pins[spare] = cyl((L.CHAMBER_X[3], L.PLUG_Y["left"], L.TOP + 0.10), 5)
        r, _ = hold(pins)
        self.assertEqual(r.details["k"], 7)
        self.assertEqual(row(r, "L4")["why"], "two pins")

    def test_pins_elsewhere_neither_add_nor_subtract(self):
        pins = keyed()
        spare_l, spare_r = L.pin_name("left", 5), L.pin_name("right", 5)
        pins[spare_l] = cyl((0.45, 0.10, L.TOP + L.PIN_LEN[5] / 2), 5)  # on the table
        pins[spare_r] = cyl((0.30, -0.24, L.PLUG_TOP_Z + L.PIN_R), 5, about((1, 0, 0), 90.0))  # lying across y
        r, _ = hold(pins)
        self.assertEqual(r.details["k"], 6)  # lying across the right plug, its centre on the wall between R2 and
        self.assertNotIn("R2", r.details["pinned"])  # R3: its 24 mm width reaches 7 mm over both openings
        self.assertNotIn("R3", r.details["pinned"])
        pins[spare_r] = cyl((0.40, -0.10, L.TOP + L.PIN_R), 5, about((1, 0, 0), 90.0))  # lying on the table
        self.assertEqual(hold(pins)[0].score, 1.0)


class Exploits(unittest.TestCase):
    def test_holding_a_pin_at_flush_height(self):
        """A wrong pin held up in a chamber so its top is flush: the gripper is within 50 mm of its top."""
        pins = start_pins()
        name = L.pin_name("left", 5)  # 52 mm in chamber 1 (needs 40)
        pins[name] = in_chamber(name, "left", 0, top=0.0)
        tc = top_centre(pins[name], 5)
        r, _ = hold(pins, arms=snap(left=tc + [0, 0, 0.011]))  # gripping its top
        self.assertEqual(r.details["k"], 0)
        # its pads cannot enter the chamber, so it can only be held above the shear line: grip within 50 mm
        r, _ = hold(pins, arms=snap(left=tc + [0, 0, 0.040]))
        self.assertEqual(r.details["k"], 0)

    def test_last_pin_still_gripped_does_not_finish(self):
        pins = keyed()
        tc = top_centre(pins[L.pin_name("right", L.CODE[3])], L.CODE[3])
        r, _ = hold(pins, arms=snap(right=tc + [0, 0, 0.019]))  # just opened, not yet risen
        # 19 mm above R4's top it is also 48 mm from R3's (44 mm along x): neither counts until it rises
        self.assertEqual(r.details["k"], 6)
        self.assertEqual(set(r.details["pinned"]), {"L1", "L2", "L3", "L4", "R1", "R2"})
        self.assertFalse(r.success)
        r, _ = hold(pins, arms=snap(right=tc + [0, 0, 0.070]))  # risen to 13 cm: let go
        self.assertTrue(r.success)

    def test_two_pins_stacked_are_never_flush(self):
        """Any two pins stacked in any chamber stand at least 10 mm proud, and each is over the other."""
        for side, i in CHAMBERS:
            for a, b in itertools.product(L.SIZES, repeat=2):
                na, nb = L.pin_name("left", a), L.pin_name("right", b)
                pins = start_pins()
                pins[na] = in_chamber(na, side, i, top=None)
                zb = L.TOP + L.FLOOR_H[i] + L.PIN_LEN[a] + L.PIN_LEN[b] / 2
                pins[nb] = cyl((L.CHAMBER_X[i], L.PLUG_Y[side], zb), b)
                self.assertGreaterEqual(L.FLOOR_H[i] + L.PIN_LEN[a] + L.PIN_LEN[b] - L.PLUG_H, 0.010 - 1e-9)
                self.assertEqual(hold(pins, n=SETTLE)[0].details["k"], 0)

    def test_a_short_pin_lying_in_a_chamber(self):
        """Only size 1 (28 mm) fits lying in a 34 mm chamber with clearance; size 2 is exactly 34 mm, a zero-clearance
        fit. Either is caught by the 8 degree rule (and is 4 mm or more low); size 3 and up are longer than the
        chamber (and too long for its diagonal at 24 mm wide)."""
        self.assertLess(L.PIN_LEN[1], L.CHAMBER)
        self.assertAlmostEqual(L.PIN_LEN[2], L.CHAMBER)
        for size in (3, 4, 5):
            self.assertGreater(L.PIN_LEN[size], L.CHAMBER)
        for size in L.SIZES:  # lying diagonally needs (length + width) / sqrt 2 <= 34 mm: none does
            self.assertGreater((L.PIN_LEN[size] + 2 * L.PIN_R) / math.sqrt(2), L.CHAMBER)
        for (side, i), size, turn in itertools.product(CHAMBERS, (1, 2), ((0, 1, 0), (1, 0, 0))):
            name = L.pin_name(side, size)
            pins = start_pins()
            z = L.TOP + L.FLOOR_H[i] + L.PIN_R
            pins[name] = cyl((L.CHAMBER_X[i], L.PLUG_Y[side], z), size, about(turn, 90.0))  # along x, then y
            r, _ = hold(pins)
            self.assertEqual(r.details["k"], 0)
            self.assertIn("upright", row(r, chamber_id(side, i))["why"])
            self.assertLessEqual(z + L.PIN_R - L.PLUG_TOP_Z, -0.004 + 1e-9)  # 4 mm or more low anyway

    def test_a_pin_caught_leaning_against_a_chamber_wall(self):
        """Review finding: a pin can stand on one bottom-rim point with its opposite bottom corner pressed against
        a chamber wall, leaning away from that wall. Righting itself would push that corner into the wall, so it
        can right itself only if its rim slides, and friction (1.0 or 1.2) stops that: the static check finds the
        lean stays from 0.5 degrees up to the chamber's cap for every correct pin (size 1 to 26.5, size 2 to 19.5,
        size 3 to 15.5, size 4 to 13.0 degrees). Rule (a) rejects every such lean over 8 degrees; at 8 or less the
        top moves by at most 1.4 mm, so the code's pin still reads flush and a wrong one stays 4.6 mm or more off."""
        tops = {1: 26.5, 2: 19.5, 3: 15.5, 4: 13.0}
        for mu, (size, top) in itertools.product(MU, tops.items()):
            leans = caught_leans(size, mu)
            self.assertEqual(leans, [d / 2 for d in range(1, int(2 * top) + 1)], (mu, size))  # 0.5 .. top, no gaps
            self.assertFalse(fits_caught(size, top + 0.5) and caught_lean_holds(size, top + 0.5, mu))
        for side, i in CHAMBERS:
            cid, name = chamber_id(side, i), L.pin_name(side, L.CODE[i])
            leans = caught_leans(L.CODE[i], min(MU))
            for toward, deg in itertools.product(TOWARD, leans):
                s = caught(name, side, i, deg, toward)
                self.assertAlmostEqual(tilt_deg(s), deg, places=6)
                self.assertTrue(centre_in_opening(s, side, i))
                if deg < L.TILT_MAX_DEG:
                    self.assertGreater(top_offset(s, L.CODE[i]), 0.0)
                    self.assertLessEqual(top_offset(s, L.CODE[i]), 0.0014 + 1e-6)
                elif deg > L.TILT_MAX_DEG:
                    self.assertGreater(tilt_deg(s), L.TILT_MAX_DEG)
            # through the whole scorer: just inside 8 degrees counts, just over and at the cap does not
            for toward, deg in itertools.product(TOWARD, (0.5, 7.5, 8.5, leans[-1])):
                pins = start_pins()
                pins[name] = caught(name, side, i, deg, toward)
                r, _ = hold(pins)
                self.assertEqual(r.details["k"], 1 if deg < L.TILT_MAX_DEG else 0, (cid, toward, deg))
                if deg > L.TILT_MAX_DEG:
                    self.assertEqual(row(r, cid)["why"], "upright")
        # a wrong pin caught at 8 degrees or less is never flush: 4.6 mm or more off
        for (side, i), size, toward in itertools.product(CHAMBERS, L.SIZES, TOWARD):
            if size == L.CODE[i]:
                continue
            name = L.pin_name(side, size)
            for deg in (d / 2 for d in range(1, 16)):  # 0.5 .. 7.5
                self.assertTrue(fits_caught(size, deg))
                self.assertGreater(abs(top_offset(caught(name, side, i, deg, toward), size)), 0.0046 - 1e-6)
            pins = start_pins()
            pins[name] = caught(name, side, i, 7.5, toward)
            r, _ = hold(pins)
            self.assertEqual(r.details["k"], 0)
            self.assertIn("flush", row(r, chamber_id(side, i))["why"])

    def test_wrong_codes_score_nothing(self):
        for code in [(1, 2, 3, 4), (2, 4, 1, 3), (4, 2, 3, 1), (5, 5, 5, 5), (3, 1, 2, 4)]:
            pins = start_pins()
            for side in L.SIDES:
                for i, size in enumerate(code):
                    if size == L.CODE[i]:
                        continue  # the right size in this chamber: counted (tested elsewhere)
                    name = L.pin_name(side, size)
                    if name in pins and np.allclose(pins[name].pos[:2], L.PIN_START[name]):
                        pins[name] = in_chamber(name, side, i, top=None)
            r, _ = hold(pins)
            self.assertEqual(r.details["k"], 0, code)

    def test_a_wrong_pin_leaning_toward_flush(self):
        """A pin one size too long, as tilted as the scorer allows: still more than 4.6 mm proud."""
        for i in range(4):
            if L.CODE[i] == 5:
                continue
            name = L.pin_name("left", L.CODE[i] + 1)
            pins = start_pins()
            pins[name] = in_chamber(name, "left", i, top=None, tilt=8.0)
            self.assertEqual(hold(pins)[0].details["k"], 0)

    def test_mixed_columns_and_upside_down_pins_still_count(self):
        """The plugs are identical, so which column a pin came from does not matter; nor which end is up."""
        pins = start_pins()
        for i in range(4):
            pins[L.pin_name("right", L.CODE[i])] = correct("left", i, column="right", top=None, flip=(i % 2 == 0))
            pins[L.pin_name("left", L.CODE[i])] = correct("right", i, column="left", top=None)
        r, _ = hold(pins)
        self.assertEqual(r.score, 1.0)

    def test_a_pin_still_falling_or_rocking(self):
        """Dropped in, it counts only once its centre has moved under 2 mm in 0.5 s."""
        t = Tracker(HZ)
        pins = start_pins()
        name = L.pin_name("left", L.CODE[0])
        for z in np.linspace(0.018, 0.0, 4):  # the last 18 mm of a drop over 4 steps
            pins[name] = correct("left", 0, top=float(z))
            r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 0)
        for _ in range(WINDOW - 4):
            r = t.update(pins, snap())
        self.assertEqual(r.details["k"], 0)  # the fall is still inside the window
        r, _ = hold(pins, n=5, t=t)
        self.assertEqual(r.details["k"], 1)

    def test_a_jammed_lean_is_rejected(self):
        """Wedged at the chamber's lean cap (13 deg or more) with its top flush, it would not count either."""
        r, _ = hold({**start_pins(), L.pin_name("left", 4): correct("left", 2, top=0.0, tilt=13.4)})
        self.assertEqual(r.details["k"], 0)


class Robustness(unittest.TestCase):
    def test_invalid_states_score_zero_and_leave_the_tracker(self):
        r, t = hold(keyed())
        self.assertEqual(r.score, 1.0)
        hist = {n: list(h) for n, h in t.hist.items()}
        for bad in ("nan", "zero_quat", "missing", "inf_vel", "short_pos"):
            pins = keyed()
            name = L.pin_name("left", 3)
            s = pins[name]
            if bad == "nan":
                s.pos = np.array([np.nan, 0.0, 0.0])
            elif bad == "zero_quat":
                s.quat_xyzw = np.zeros(4)
            elif bad == "missing":
                del pins[name]
            elif bad == "inf_vel":
                s.lin_vel = np.array([np.inf, 0.0, 0.0])
            else:
                s.pos = np.array([0.1, 0.2])
            r = t.update(pins, snap())
            self.assertEqual(r.score, 0.0, bad)
            self.assertFalse(r.success)
            self.assertEqual({n: list(h) for n, h in t.hist.items()}.keys(), hist.keys())
            for n in hist:
                self.assertEqual(len(t.hist[n]), len(hist[n]))
                self.assertTrue(all(np.array_equal(a, b) for a, b in zip(t.hist[n], hist[n])))
            self.assertEqual(t.result(pins, snap()).score, 0.0)
        self.assertEqual(t.bad_steps, 5)
        r = t.update(keyed(), snap())  # the next good step carries on where it was
        self.assertEqual(r.score, 1.0)
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0, 0])))

    def test_none_objects(self):
        self.assertEqual(Tracker(HZ).update({n: None for n in L.PINS}, snap()).score, 0.0)
        self.assertEqual(Tracker(HZ).update(None, snap()).score, 0.0)

    def test_random_states_stay_in_range(self):
        rng = np.random.default_rng(3)
        t = Tracker(HZ)
        for _ in range(200):
            pins = start_pins()
            for n in rng.choice(list(L.PINS), size=4, replace=False):
                q = rng.normal(size=4)
                c = (rng.uniform(0.15, 0.4), rng.uniform(-0.3, 0.3), L.TOP + rng.uniform(0.0, 0.1))
                pins[n] = cyl(c, size_of(n), q / np.linalg.norm(q))
            r = t.update(pins, snap())
            self.assertIn(r.score, {k / 8 for k in range(9)})
            self.assertEqual(r.success, r.score == 1.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_the_kit_starts_a_new_episode(self):
        r, t = hold(keyed())
        self.assertTrue(r.success)
        r = t.update(start_pins(), snap())  # every pin back on its mark in one step
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(all(len(h) == 1 for h in t.hist.values()))
        r, _ = hold(keyed(), t=t)
        self.assertEqual(r.score, 1.0)

    def test_no_reset_without_a_jump(self):
        r, t = hold(start_pins())
        steps = t.steps
        t.update(start_pins(), snap())
        self.assertEqual(t.steps, steps + 1)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        r, t = hold(keyed())
        pins = keyed()
        pins[L.pin_name("left", 5)] = cyl((0.45, 0.2, L.TOP + 0.026), 5)  # thrown, not back on its mark
        t.update(pins, snap())
        self.assertGreater(t.steps, SETTLE)

    def test_rigstate_reset_hook(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        try:
            t = Tracker(HZ)
            rigstate.on_reset(t.reset)
            hold(keyed(), t=t)
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertEqual(t.steps, 0)
            self.assertIsNone(rigstate.arms())
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _overlap(a, b):
    """Overlap volume of two upright (unturned) boxes, from their specs."""
    lo_a, hi_a = np.array(a.pos) - np.array(a.size) / 2, np.array(a.pos) + np.array(a.size) / 2
    lo_b, hi_b = np.array(b.pos) - np.array(b.size) / 2, np.array(b.pos) + np.array(b.size) / 2
    d = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
    return float(np.prod(np.clip(d, 0, None)))


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: objects, hooks, the reset hook, trace, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.json = json.loads((HERE / "task.json").read_text())

    def objects(self, pins=None):
        pins = pins or start_pins()
        return dict(pins)

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, set(L.PINS))
        for o in self.task.OBJECTS:
            if o.name in L.PINS:
                n = size_of(o.name)
                self.assertAlmostEqual(o.height, L.PIN_LEN[n])
                self.assertAlmostEqual(o.radius, L.PIN_R)
                vol = math.pi * L.PIN_R ** 2 * L.PIN_LEN[n]
                self.assertAlmostEqual(o.mass, 2700 * vol, delta=0.001)  # aluminium
                self.assertLessEqual(o.mass, 0.3)
                self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP)
            else:
                self.assertEqual(type(o).__name__, "Box")
                self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)  # every fixed part stands on the table
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(len(self.task.OBJECTS), 2 * 12 + 15 + 10)

    def test_fixed_boxes_are_butt_jointed(self):
        boxes = [o for o in self.task.OBJECTS if o.mass is None]
        for a, b in itertools.combinations(boxes, 2):
            self.assertLess(_overlap(a, b), 1e-12, (a.name, b.name))

    def test_chambers_and_floors(self):
        by = {o.name: o for o in self.task.OBJECTS}
        for s in ("l", "r"):
            side = "left" if s == "l" else "right"
            for i in range(4):
                cut = by[f"key_{s}_cut_{i + 1}"]
                self.assertAlmostEqual(cut.size[2], L.FLOOR_H[i])
                self.assertAlmostEqual(cut.pos[0], L.CHAMBER_X[i])
                self.assertAlmostEqual(cut.pos[1], L.PLUG_Y[side])
                walls = sorted(by[f"plug_{s}_wall_{j}"].pos[0] for j in (i, i + 1))
                self.assertAlmostEqual(walls[1] - walls[0] - L.CROSS_WALL_T, L.CHAMBER)
                self.assertAlmostEqual(L.FLOOR_H[i] + L.PIN_LEN[L.CODE[i]], L.PLUG_H)
            sides = sorted(by[f"plug_{s}_side_{k}"].pos[1] for k in ("in", "out"))
            self.assertAlmostEqual(sides[1] - sides[0] - L.SIDE_WALL_T, L.CHAMBER)
            bow = by[f"key_{s}_bow"]
            self.assertAlmostEqual(bow.pos[0] + bow.size[0] / 2, L.PLUG_X0)
        for name, (c, n) in L.PINS.items():  # each pin at its cell's centre: row n = size n
            self.assertEqual(L.PIN_START[name], (L.KIT_ROW_X[n - 1], L.KIT_COL_Y[c]))

    def test_instruction_matches_task_json(self):
        self.assertEqual(self.task.INSTRUCTION, self.json["instruction"])
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)
        self.assertIn("3-1-4-2", self.task.INSTRUCTION)
        self.assertEqual("".join(map(str, L.CODE)), "3142")
        self.assertIn("Possible scores: 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 and 1.0",
                      self.json["scoring"])

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        for _ in range(SETTLE):
            snap()
            ok = self.task.success(self.objects(keyed()))
        self.assertTrue(ok)
        self.assertEqual(self.task.score(self.objects(keyed())), 1.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        snap()
        self.assertEqual(self.task.score(self.objects(keyed())), 0.0)  # no history yet: not still

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            try:
                os.environ["TASK_TRACE"] = str(path)
                rigstate.offline_reset()
                for _ in range(SETTLE):
                    snap()
                    self.task.success(self.objects(keyed()))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), SETTLE)
                last = rows[-1]
                self.assertEqual(last["k"], 8)
                for key in ("score", "success", "chambers", "left_tcp", "right_tcp", "pin_pos", "pin_tilt_deg",
                            "arms_unavailable"):
                    self.assertIn(key, last)
                for key in ("pin", "tilt_deg", "top_mm", "path_mm", "net_mm", "spread_mm", "tcp_mm", "others_over",
                            "pinned"):
                    self.assertIn(key, last["chambers"]["L1"])
                self.assertEqual(set(last["pin_tilt_deg"]), set(L.PINS))  # every pin's lean, in a chamber or not
                self.assertTrue(all(v == 0.0 for v in last["pin_tilt_deg"].values()))
                self.assertEqual(last["chambers"]["L1"]["net_mm"], 0.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                snap()
                self.task.success(self.objects())  # must not raise
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        self.assertAlmostEqual(sum(w.duration for w in left), 59.0)
        self.assertAlmostEqual(sum(w.duration for w in right), 66.5)
        self.assertAlmostEqual(sum(w.duration for w in right[:2]), 7.5)
        for wps in (left, right[2:]):
            self.assertEqual(len(wps), 44)
            for p in range(4):
                seg = wps[11 * p: 11 * (p + 1)]
                self.assertAlmostEqual(sum(w.duration for w in seg), 15.5 if p == 0 else 14.5)
                size = L.CODE[p]
                self.assertAlmostEqual(seg[1].pos[2], L.TOP + L.PIN_LEN[size] + 0.001)
                self.assertIn("settle", seg[2].label)  # before the close
                self.assertIn("settle", seg[8].label)  # before the release
                self.assertEqual(seg[3].finger, self.task.CLOSED)
                self.assertEqual(seg[9].finger, self.task.OPEN)
                self.assertGreaterEqual(seg[3].duration, 1.5)
                self.assertGreaterEqual(seg[9].duration, 1.5)
                self.assertAlmostEqual(seg[7].pos[2], L.TOP + 0.079)
                self.assertAlmostEqual(seg[7].pos[0], L.CHAMBER_X[p])
                # the carried pin's bottom clears the plug top by 23 mm or more
                self.assertGreaterEqual(seg[4].pos[2] - 0.001 - L.PIN_LEN[size] - L.PLUG_TOP_Z, 0.023 - 1e-9)
                # released, its top is at 78 mm and it falls 18 mm onto its floor
                drop = (seg[9].pos[2] - 0.001 - L.PIN_LEN[size]) - (L.TOP + L.FLOOR_H[p])
                self.assertAlmostEqual(drop, 0.018)
        self.assertAlmostEqual(self.task.CLOSED, -(2 * L.PIN_R - 0.008) / 2)
        self.assertGreater(-self.task.OPEN * 2, 2 * L.PIN_R + 0.010)  # 40 mm open round a 24 mm pin

    def test_kit_turns_never_overlap(self):
        """Only one arm works low in the kit at a time: from the start of its descent to the end of its lift."""

        def low_spans(wps):
            t, out = 0.0, []
            for w in wps:
                if "down onto" in w.label:
                    start = t
                if "lift it out" in w.label:
                    out.append((start, t + w.duration))
                t += w.duration
            return out

        plan = self.task.demo()
        a, b = low_spans(plan["left"]), low_spans(plan["right"])
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 4)
        for (s0, e0), (s1, e1) in itertools.product(a, b):
            self.assertTrue(e0 <= s1 or e1 <= s0, ((s0, e0), (s1, e1)))

    def test_demo_switches(self):
        for key, value, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                 ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = value
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
