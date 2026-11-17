"""Offline tests for layout.py, scoring.py and task.py: synthetic part states and synthetic arm snapshots
(rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
IDENT = (0.0, 0.0, 0.0, 1.0)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def cup_state(centre, quat=IDENT, vel=(0.0, 0.0, 0.0)):
    c = np.array(centre, float)
    half = np.array([L.CUP_R, L.CUP_R, L.CUP_H / 2])
    ext = np.abs(S.rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=c - ext, high=c + ext)


def carrier_state(dx=0.0, dy=0.0, dz=0.0, quat=IDENT, vel=(0.0, 0.0, 0.0)):
    o = np.array([L.CAR_X + dx, L.CAR_Y + dy, L.TOP + dz], float)
    return SimpleNamespace(origin=o, pos=o + np.array([0.0, 0.0, L.RIM_H / 2]), quat_xyzw=np.array(quat, float),
                           lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=o.copy(),
                           high=o + np.array([0.0, 0.0, L.RIM_H]))


def seat_pos(pocket, dx=0.0, dy=0.0, dz=0.0):
    x, y = L.POCKET_XY[pocket]
    return (x + dx, y + dy, L.TOP + L.FLOOR_TOP + L.CUP_H / 2 + dz)


def start_pos(cup):
    x, y = L.CUP_START[cup]
    return (x, y, L.TOP + L.CUP_H / 2)


def start():
    s = {c: cup_state(start_pos(c)) for c in L.CUP_NAMES}
    s[S.CARRIER] = carrier_state()
    return s


def seated(n=4, **over):
    """The first ``n`` cups seated (in the order a, b, c, d), the rest at their start places."""
    s = start()
    for c in L.CUP_NAMES[:n]:
        s[c] = cup_state(seat_pos(L.ASSIGN[c]))
    s.update(over)
    return s


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def feed(tracker, seq, arms=FAR):
    res = None
    for s in seq:
        res = tracker.update(s, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(2 * L.CUP_R, 0.082)  # the lid's skirt
        self.assertAlmostEqual(L.CUP_BASE_D, 0.056)
        self.assertAlmostEqual(L.CUP_RIM_D, 0.080)
        self.assertAlmostEqual(L.CUP_H, 0.095)
        self.assertAlmostEqual(L.CUP_MASS, 0.25)
        self.assertAlmostEqual(L.PITCH, 0.104)
        self.assertAlmostEqual(L.POCKET_D, 0.036)
        self.assertAlmostEqual(L.PLATE, 0.196, places=6)
        self.assertAlmostEqual(L.SEATED_TOP - L.RIM_H, 0.059, places=6)  # the lid stands 59 mm above the carrier's top

    def test_the_carrier_margin_is_consistent_with_the_pockets(self):
        """The base plate reaches EDGE past the outer faces of the outermost walls (at the mouth)."""
        outer = L.PITCH / 2 + L.POCKET_R_RIM + L.WALL_T
        self.assertAlmostEqual(L.PLATE / 2 - outer, L.EDGE, places=9)

    def test_a_cup_fits_the_pocket_with_play(self):
        base_r = L.cup_parts()[0][2] / 2  # the lowest disc (57.1 mm)
        self.assertGreater(L.POCKET_R_FLOOR - base_r, 0.0015)  # at least 1.5 mm each side on the floor
        self.assertLess(L.POCKET_R_FLOOR - base_r, 0.0035)
        self.assertAlmostEqual(L.POCKET_R_RIM - L.CUP_BASE_D / 2, 0.010)  # 10 mm of lead-in for the base at the mouth
        # a cup on the floor is within 3.6 mm of the pocket's axis (the 12-sided funnel's corners are further out)
        self.assertLessEqual(L.POCKET_R_FLOOR / math.cos(math.pi / L.N_SIDES) - base_r, 0.0036)
        # every disc of a seated cup clears the funnel's flats at its height
        for kind, zc, d, h in L.cup_parts():
            z1 = zc + h / 2
            if kind == "body" and z1 <= L.POCKET_D:
                wall = L.POCKET_R_FLOOR + (L.POCKET_R_RIM - L.POCKET_R_FLOOR) * (zc - h / 2) / L.POCKET_D
                self.assertGreater(wall - d / 2, 0.0015, zc)
        # a cup bottomed but leaning has its underside's centre r x sin(tilt) up: over 2 mm from 5 degrees
        self.assertGreater(base_r * math.sin(math.radians(5.0)), L.SEAT_Z_TOL)

    def test_the_cup_is_a_cone_with_a_lid(self):
        parts = L.cup_parts()
        body = [p for p in parts if p[0] == "body"]
        self.assertEqual(len(body), L.CUP_N_DISCS)
        self.assertAlmostEqual(body[-1][1] + body[-1][3] / 2, L.CUP_BODY_H)
        ds = [p[2] for p in body]
        self.assertEqual(ds, sorted(ds))
        self.assertGreater(ds[0], L.CUP_BASE_D)
        self.assertLess(ds[-1], L.CUP_RIM_D)
        self.assertAlmostEqual(max(p[2] for p in parts), L.LID_SKIRT_D)
        self.assertAlmostEqual(L.cup_radius_at(0.050), L.cup_d(0.052) / 2)

    def test_pockets_do_not_touch(self):
        """The mouths' outer corners of neighbouring pockets stay clear of each other's inner faces."""
        corner_out = (L.POCKET_R_RIM + L.WALL_T) / math.cos(math.pi / L.N_SIDES)
        self.assertGreater(L.PITCH - 2 * corner_out, -0.006)  # the walls may overlap by a few mm at the corners
        self.assertGreater(L.PITCH - (L.POCKET_R_RIM + L.WALL_T) * 2, 0.0)  # but never across the flats

    def test_a_cup_cannot_bridge_two_pockets_low(self):
        """Resting across two mouths, a cup's underside is at the mouths' height, 36 mm above seated."""
        self.assertGreater(L.RIM_H - L.FLOOR_TOP, 10 * L.SEAT_Z_TOL)

    def test_funnel_slabs_make_the_stated_cone(self):
        slabs = L.funnel_slabs(0.0, 0.0, L.POCKET_R_RIM, L.POCKET_R_FLOOR, L.POCKET_D, L.FLOOR_TOP)
        self.assertEqual(len(slabs), L.N_SIDES)
        a = math.atan2(L.POCKET_R_RIM - L.POCKET_R_FLOOR, L.POCKET_D)
        for k, (c, size, ypr) in enumerate(slabs):
            th = 2 * math.pi * k / L.N_SIDES
            self.assertAlmostEqual(ypr[0], th)
            self.assertAlmostEqual(ypr[1], a)
            # the slab's inner face passes through (r, floor) and (R, mouth): measure at its two ends
            n = np.array([math.cos(a), -math.sin(a)])  # outward normal in the (radial, z) plane
            centre = np.array([math.hypot(c[0], c[1]), c[2]])
            inner = centre - n * (L.WALL_T / 2)
            u = np.array([math.sin(a), math.cos(a)])
            lo, hi = inner - u * size[2] / 2, inner + u * size[2] / 2
            self.assertAlmostEqual(lo[0], L.POCKET_R_FLOOR, places=6)
            self.assertAlmostEqual(lo[1], L.FLOOR_TOP, places=6)
            self.assertAlmostEqual(hi[0], L.POCKET_R_RIM, places=6)
            self.assertAlmostEqual(hi[1], L.RIM_H, places=6)

    def test_carrier_shapes_count_and_extent(self):
        shapes = T.carrier_shapes()
        self.assertEqual(len(shapes), 1 + 4 * L.N_SIDES)
        lo, hi = T.CARRIER_PART.rest_state().low, T.CARRIER_PART.rest_state().high
        self.assertAlmostEqual(float(lo[2]), L.TOP, places=6)
        self.assertAlmostEqual(float(hi[2]) - L.TOP, L.RIM_H, delta=0.001)
        self.assertAlmostEqual(float(hi[0] - lo[0]), L.PLATE, delta=0.002)

    def test_start_layout_is_clear(self):
        """Lids at least 15 mm apart and 150 mm from the carrier; the start places are as the setup says."""
        pts = {c: np.array(L.CUP_START[c]) for c in L.CUP_NAMES}
        for i, a in enumerate(L.CUP_NAMES):
            for b in L.CUP_NAMES[i + 1:]:
                self.assertGreater(float(np.linalg.norm(pts[a] - pts[b])) - 2 * L.CUP_R, 0.015, (a, b))
        for c in L.CUP_NAMES:
            gap = abs(pts[c][1]) - L.CUP_R - L.PLATE / 2
            self.assertGreater(gap, 0.15, c)
            self.assertGreater(abs(pts[c][1]), 0.30)  # out beside the arm, not under the ready pose of the other
            self.assertGreaterEqual(pts[c][0], 0.19 - 1e-9)  # clear of the bases (the hidden strip is |y| > 0.45 m)

    def test_each_side_is_one_arms(self):
        for c in L.CUP_NAMES:
            side = 1 if L.CUP_START[c][1] > 0 else -1
            px, py = L.POCKET_XY[L.ASSIGN[c]]
            self.assertEqual(1 if py > 0 else -1, side)
        self.assertEqual(sorted(L.ASSIGN.values()), sorted(L.POCKETS))

    def test_parts_spawn_at_rest(self):
        r = S.read(start())
        self.assertTrue(S.Tracker.at_start(r))
        for c in L.CUP_NAMES:
            st = T._cup(c).rest_state()
            self.assertAlmostEqual(float(st.origin[2]), L.TOP)
            self.assertAlmostEqual(float(st.low[2]), L.TOP)
            self.assertAlmostEqual(float(st.high[2]) - L.TOP, L.CUP_H)
        self.assertAlmostEqual(float(T.CARRIER_PART.rest_state().origin[2]), L.TOP)


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(seated(4))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["seated"], 4)

    def test_tier_table(self):
        for n, want in ((4, 1.0), (3, 0.75), (2, 0.5), (1, 0.25), (0, 0.0)):
            r = run(seated(n))
            self.assertEqual(r.score, want, n)
            self.assertEqual(r.success, want == 1.0)

    def test_any_pockets_count(self):
        """Which cup is in which pocket does not matter: cup_a in the far right pocket, cup_d in the near left."""
        s = seated(0)
        for c, p in (("cup_a", "far_right"), ("cup_b", "near_left"), ("cup_c", "far_left"), ("cup_d", "near_right")):
            s[c] = cup_state(seat_pos(p))
        self.assertEqual(run(s).score, 1.0)

    def test_start_scores_zero(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_two_arms_one_side_each(self):
        s = seated(0)
        s["cup_a"] = cup_state(seat_pos("near_left"))
        s["cup_b"] = cup_state(seat_pos("far_left"))
        self.assertEqual(run(s).score, 0.5)  # the left arm's two cups alone

    def test_seat_edges(self):
        for dz, ok in ((0.0, True), (0.0019, True), (0.0021, False), (-0.0021, False), (0.020, False)):
            s = seated(3, cup_a=cup_state(seat_pos("near_left", dz=dz)))
            self.assertEqual(run(s).score, 0.75 if ok else 0.5, dz)
        for off, ok in ((0.0, True), (0.0075, True), (0.0085, False)):
            s = seated(3, cup_a=cup_state(seat_pos("near_left", dx=off)))
            self.assertEqual(run(s).score, 0.75 if ok else 0.5, off)

    def test_seat_tilt_edges(self):
        for deg, ok in ((0.0, True), (1.9, True), (2.2, False), (6.0, False)):
            s = seated(3, cup_a=cup_state(seat_pos("near_left"), about("x", deg)))
            self.assertEqual(run(s).score, 0.75 if ok else 0.5, deg)

    def test_cup_resting_on_the_mouth_is_not_seated(self):
        """A cup across the rim of two pockets (underside at the mouths' height) earns nothing."""
        s = seated(0)
        s["cup_a"] = cup_state((L.CAR_X, L.POCKET_XY["near_left"][1], L.TOP + L.RIM_H + L.CUP_H / 2))
        self.assertEqual(run(s).score, 0.0)
        s["cup_a"] = cup_state(seat_pos("near_left", dz=L.RIM_H - L.FLOOR_TOP))  # on one mouth, 36 mm up
        self.assertEqual(run(s).score, 0.0)

    def test_cup_on_the_plate_outside_the_pockets_is_not_seated(self):
        """A cup standing on the 7 mm plate past the outer wall (its underside at the floor's height) is over no
        mouth: no credit."""
        s = seated(3)
        edge = L.CAR_X - L.PLATE / 2 + 0.002  # its centre 2 mm inside the plate's edge
        s["cup_d"] = cup_state((edge, L.POCKET_XY["near_right"][1], L.TOP + L.FLOOR_TOP + L.CUP_H / 2))
        r = run(s)
        self.assertEqual(r.details["seated"], 3)
        self.assertEqual(r.score, 0.75)

    def test_two_cups_in_one_pocket_earn_nothing_for_it(self):
        s = seated(0)
        s["cup_a"] = cup_state(seat_pos("near_left"))
        s["cup_b"] = cup_state(seat_pos("near_left", dz=L.CUP_H))  # stacked on the first, lid on its base
        s["cup_c"] = cup_state(seat_pos("near_right"))
        s["cup_d"] = cup_state(seat_pos("far_right"))
        r = run(s)
        self.assertEqual(r.details["seated"], 2)
        self.assertEqual(r.score, 0.5)

    def test_cups_judged_in_the_carriers_frame(self):
        """A carrier turned 20 degrees about its vertical, the cups seated in its pockets: still 1.0."""
        yaw = 20.0
        q = about("z", yaw)
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        st = start()
        st[S.CARRIER] = carrier_state(quat=q)
        for cup in L.CUP_NAMES:
            px, py = L.POCKET_XY[L.ASSIGN[cup]]
            lx, ly = px - L.CAR_X, py - L.CAR_Y
            x, y = L.CAR_X + c * lx - s_ * ly, L.CAR_Y + s_ * lx + c * ly
            st[cup] = cup_state((x, y, L.TOP + L.FLOOR_TOP + L.CUP_H / 2), q)
        self.assertEqual(run(st).score, 1.0)

    def test_carrier_pushed_a_little_still_scores(self):
        dx = 0.025
        st = seated(4, carrier=carrier_state(dx=dx))
        for c in L.CUP_NAMES:
            p = np.array(st[c].pos)
            st[c] = cup_state(p + np.array([dx, 0.0, 0.0]))
        self.assertEqual(run(st).score, 1.0)


class Latches(unittest.TestCase):
    def _then_clean(self, bad):
        tr = S.Tracker()
        feed(tr, [bad])
        return run(seated(4), tracker=tr, steps=WIN + 5)

    def test_spill(self):
        bad = seated(4, cup_a=cup_state(seat_pos("near_left"), about("x", 16.0)))
        r = self._then_clean(bad)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["latch"]["spill"], 1)

    def test_tilt_just_under_the_spill_limit_is_fine(self):
        bad = seated(4, cup_a=cup_state(seat_pos("near_left", dz=0.03), about("x", 14.0)))
        self.assertEqual(self._then_clean(bad).score, 1.0)

    def test_spill_while_carried(self):
        tr = S.Tracker()
        feed(tr, [start()] * 5)
        carried = start()
        carried["cup_a"] = cup_state((0.2, 0.2, L.TOP + 0.2), about("y", 20.0))
        feed(tr, [carried])
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 0.0)

    def test_dropped(self):
        bad = seated(4, cup_a=cup_state(seat_pos("near_left", dz=0.05), vel=(0.0, 0.0, -0.95)))
        r = self._then_clean(bad)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["latch"]["dropped"], 1)

    def test_a_slow_descent_is_not_a_drop(self):
        bad = seated(4, cup_a=cup_state(seat_pos("near_left", dz=0.05), vel=(0.0, 0.0, -0.05)))
        self.assertEqual(self._then_clean(bad).score, 1.0)
        self.assertGreater(L.DROP_SPEED, 0.88)
        self.assertLess(L.DROP_SPEED, 0.89)

    def test_carrier_moved_far(self):
        bad = seated(4, carrier=carrier_state(dx=0.031))
        r = self._then_clean(bad)
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(r.details["latch"]["carrier"])

    def test_carrier_moved_near_the_limit(self):
        bad = seated(4, carrier=carrier_state(dy=0.029))
        self.assertEqual(self._then_clean(bad).score, 1.0)

    def test_carrier_tilted(self):
        bad = seated(4, carrier=carrier_state(quat=about("x", 6.0)))
        self.assertEqual(self._then_clean(bad).score, 0.0)
        ok = seated(4, carrier=carrier_state(quat=about("x", 4.0)))
        self.assertEqual(self._then_clean(ok).score, 1.0)

    def test_a_cup_off_the_table(self):
        bad = seated(4, cup_d=cup_state((0.3, -0.7, L.TOP - 0.20)))
        self.assertEqual(self._then_clean(bad).score, 0.0)

    def test_a_lying_cup_at_the_end(self):
        s = seated(3, cup_d=cup_state((0.3, -0.34, L.TOP + L.CUP_R), about("x", 90.0)))
        r = run(s)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["lying"])

    def test_an_upside_down_cup_in_a_pocket(self):
        s = seated(3, cup_d=cup_state(seat_pos("far_right"), about("x", 180.0)))
        self.assertEqual(run(s).score, 0.0)

    def test_latch_survives_the_rest_of_the_trial(self):
        tr = S.Tracker()
        feed(tr, [seated(4, cup_a=cup_state(seat_pos("near_left"), about("x", 30.0)))])
        for _ in range(3):
            r = run(seated(4), tracker=tr, steps=WIN + 5)
        self.assertEqual(r.score, 0.0)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_a_cup(self):
        x, y, z = seat_pos("near_left")
        near = {"left": arm((x, y + 0.04, L.TOP + 0.09), 0.6), "right": FAR["right"]}
        r = run(seated(4), near)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["released_1s"])

    def test_hand_over_the_carrier(self):
        near = {"left": FAR["left"], "right": arm((L.CAR_X, -0.12, L.TOP + 0.10), 1.0)}
        self.assertEqual(run(seated(4), near).score, 0.0)

    def test_hand_clear(self):
        gapped = {"left": arm((0.17, 0.24, L.TOP + 0.17), 1.0), "right": arm((0.17, -0.24, L.TOP + 0.17), 1.0)}
        self.assertEqual(run(seated(4), gapped).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(seated(4), None).score, 0.0)
        self.assertEqual(run(seated(4), {"left": FAR["left"]}).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        x, y, z = seat_pos("near_left")
        close = {"left": arm((x, y + 0.04, L.TOP + 0.09), 0.6), "right": FAR["right"]}
        feed(tr, [seated(4)] * 5, close)
        r = feed(tr, [seated(4)] * (WIN - 1), FAR)
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [seated(4)] * 3, FAR)
        self.assertEqual(r.score, 1.0)

    def test_moving_cup_is_not_still(self):
        tr = S.Tracker()
        seq = []
        for i in range(WIN + 3):
            seq.append(seated(4, cup_a=cup_state(seat_pos("near_left", dz=0.008 * (i % 2)))))  # 8 mm shimmer
        self.assertEqual(feed(tr, seq).score, 0.0)
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 3).score, 1.0)

    def test_rocking_cup_is_not_still(self):
        tr = S.Tracker()
        seq = [seated(4, cup_a=cup_state(seat_pos("near_left"), about("x", 8.0 * (i % 2)))) for i in range(WIN + 3)]
        self.assertEqual(feed(tr, seq).score, 0.0)

    def test_creeping_carrier_is_not_still(self):
        tr = S.Tracker()
        seq = []
        for i in range(WIN + 3):
            s = seated(4)
            s[S.CARRIER] = carrier_state(dx=0.0008 * i)
            seq.append(s)
        self.assertEqual(feed(tr, seq).score, 0.0)

    def test_needs_a_full_second_of_readings(self):
        tr = S.Tracker()
        self.assertEqual(feed(tr, [seated(4)] * (WIN - 1)).score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        s = seated(4)
        del s["cup_b"]
        r = run(s)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_state(self):
        tr = S.Tracker()
        run(seated(4), tracker=tr)
        bad = seated(4, cup_a=cup_state((float("nan"), 0.0, 0.8)))
        r = tr.update(bad, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(tr.last.success)
        self.assertEqual(tr.bad_steps, 1)
        # the 1 s windows start again: not 1.0 until a full second of good readings
        r = feed(tr, [seated(4)] * (WIN - 1))
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [seated(4)] * 3)
        self.assertEqual(r.score, 1.0)

    def test_inf_velocity_and_zero_quaternion(self):
        tr = S.Tracker()
        self.assertEqual(tr.update(seated(4, cup_a=cup_state(seat_pos("near_left"), vel=(0, 0, float("inf")))),
                                   FAR).score, 0.0)
        bad = seated(4)
        bad["cup_a"].quat_xyzw = np.zeros(4)
        self.assertEqual(tr.update(bad, FAR).score, 0.0)
        self.assertEqual(tr.bad_steps, 2)

    def test_nan_does_not_clear_a_latch(self):
        tr = S.Tracker()
        feed(tr, [seated(4, carrier=carrier_state(dx=0.05))])
        tr.update(seated(4, cup_a=cup_state((float("nan"),) * 3)), FAR)
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        feed(tr, [seated(4, carrier=carrier_state(dx=0.05))])
        self.assertIsNotNone(tr.latch["carrier"])
        tr.update(start(), FAR)
        self.assertEqual(tr.resets, 1)
        self.assertIsNone(tr.latch["carrier"])
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 1.0)

    def test_a_cup_set_back_on_its_mark_is_not_a_reset(self):
        """Everything but one cup stays put, so the layout is not 'at the start': no reset."""
        tr = S.Tracker()
        feed(tr, [seated(1, carrier=carrier_state(dx=0.05))])
        s = start()
        s[S.CARRIER] = carrier_state(dx=0.05)
        tr.update(s, FAR)
        self.assertEqual(tr.resets, 0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(300):
            st = start()
            for c in L.CUP_NAMES:
                q = rng.normal(size=4)
                st[c] = cup_state(rng.uniform(-0.5, 0.5, 3) + np.array([0.2, 0.0, L.TOP]), q / np.linalg.norm(q),
                                  rng.normal(size=3))
            st[S.CARRIER] = carrier_state(*rng.normal(0, 0.01, 3))
            res = tr.update(st, FAR if rng.random() < 0.8 else None)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(res.details)

    def test_details_are_json_clean(self):
        json.dumps(run(seated(4)).details)
        json.dumps(run(seated(4), None).details)


class Exploits(unittest.TestCase):
    def test_throwing_cups_in_from_above(self):
        """Letting the cups fall into the pockets from 5 cm up: fast enough to latch as dropped."""
        v = -math.sqrt(2 * L.G * 0.05)
        tr = S.Tracker()
        feed(tr, [seated(4, cup_a=cup_state(seat_pos("near_left", dz=0.03), vel=(0, 0, v)))])
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 0.0)

    def test_pouring_the_cups_in_by_tipping_the_carrier(self):
        tr = S.Tracker()
        feed(tr, [seated(4, carrier=carrier_state(quat=about("y", 12.0)))])
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 0.0)

    def test_carrying_the_carrier_to_the_cups(self):
        """Dragging the carrier over to a cup is a carrier move over 30 mm: zero, even with all four in at the end."""
        tr = S.Tracker()
        feed(tr, [seated(0, carrier=carrier_state(dy=0.2))])
        self.assertEqual(run(seated(4), tracker=tr, steps=WIN + 5).score, 0.0)

    def test_holding_the_cups_down_at_the_end(self):
        """All four seated but a gripper still pressing on a cup's lid: nothing."""
        x, y, z = seat_pos("far_left")
        press = {"left": arm((x, y, L.TOP + L.SEATED_TOP + 0.02), 0.2), "right": FAR["right"]}
        self.assertEqual(run(seated(4), press).score, 0.0)

    def test_holding_the_carrier_still(self):
        hold = {"left": arm((L.CAR_X, L.CAR_Y + 0.13, L.TOP + 0.03), 0.3), "right": FAR["right"]}
        self.assertEqual(run(seated(4), hold).score, 0.0)

    def test_doing_nothing_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_a_cup_leaned_into_a_pocket(self):
        """Leaning a cup in a pocket with its rim on the mouth and its base on the floor of the next one: no credit."""
        s = seated(3)
        s["cup_d"] = cup_state((L.CAR_X + 0.03, L.POCKET_XY["far_right"][1], L.TOP + L.RIM_H + 0.05),
                               about("y", 25.0))
        self.assertEqual(run(s).details["seated"], 3)

    def test_stacking_cups_beside_the_carrier(self):
        s = seated(0)
        s["cup_a"] = cup_state((0.3, 0.34, L.TOP + L.CUP_H / 2))
        s["cup_b"] = cup_state((0.3, 0.34, L.TOP + L.CUP_H * 1.5))
        self.assertEqual(run(s).score, 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, [S.CARRIER] + list(L.CUP_NAMES))
        for o in T.OBJECTS:
            self.assertIsNotNone(o.mass)  # all loose
        self.assertEqual(T.SCENE, "station")
        self.assertEqual(T.SCORE_OVER, "end")

    def test_instruction_states_the_goal_only(self):
        t = T.INSTRUCTION.lower()
        for word in ("pocket", "upright", "hands", "carrier"):
            self.assertIn(word, t)
        for word in ("grip", "lift", "arm", "left", "right", "slowly"):
            self.assertIsNone(re.search(r"\b" + word + r"\b", t), word)

    def test_demo_plans(self):
        plan = T.demo()
        self.assertEqual(set(plan), {"left", "right"})
        self.assertAlmostEqual(T._duration(plan["left"]), 47.0, delta=2.0)
        self.assertAlmostEqual(T._duration(plan["right"]), 59.0, delta=2.0)
        for arm_name, wps in plan.items():
            grips = [w for w in wps if w.label == "grip the cup"]
            self.assertEqual(len(grips), 2)
            for w in wps:
                self.assertTrue(T.OPEN - 1e-9 <= w.finger <= 0.0)

    def test_demo_env_switches(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        for idle, kept in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = idle
            try:
                self.assertEqual(list(T.demo()), [kept])
            finally:
                del os.environ["DEMO_IDLE_ARM"]

    def test_every_gripper_change_takes_at_least_1p5_s(self):
        for wps in T.demo().values():
            prev = T.OPEN
            for w in wps:
                if abs(w.finger - prev) > 0.01:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
                prev = w.finger

    def test_arms_place_by_turns(self):
        """The four placing windows (carry over the pocket to straight up) run left, right, left, right, and each
        starts no more than OVERLAP before the one before it has ended (the other arm's last move is the aside)."""
        spans = []
        for arm_name, wps in T.demo().items():
            t, t0 = 0.0, None
            for w in wps:
                if w.label == "carry it over the pocket":
                    t0 = t
                t += w.duration
                if w.label == "straight up":
                    spans.append((t0, t, arm_name))
        spans.sort()
        self.assertEqual([s[2] for s in spans], ["left", "right", "left", "right"])
        for (a0, b0, _), (a1, b1, _) in zip(spans, spans[1:]):
            self.assertGreaterEqual(a1, b0 - T.OVERLAP - 1.5 - 1e-6)  # the 1.5 s of 'straight up' is already clear
            self.assertLess(a0, a1)

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.GRIP, T.GRIP_D - 0.008, places=9)  # commanded 8 mm under the cup's width
        self.assertLess(T.GRIP_D, 0.080)  # within the practical grasp widths
        self.assertLess(T.GRIP_ON + 0.038, L.LID_SKIRT_Z0)  # the 38 mm pads stay under the lid
        self.assertGreater(-2 * T.OPEN - 2 * L.CUP_R, 0.010)  # 12 mm to spare over the lid as the hand comes down
        # the pad tips stay above the mouths when the cup is on the floor, with 11 mm to spare (7 mm pressed)
        tips = T.SEAT_Z - T.PAD_TIP
        self.assertGreater(tips - L.RIM_H, 0.0109)
        self.assertGreater(tips - T.PRESS - L.RIM_H, 0.0069)
        # the gripper's housing (its lowest proxy capsule, 35 mm over the TCP) clears the lid at the grip
        self.assertGreaterEqual(T.GRIP_UP + 0.035 - L.CUP_H, -1e-9)
        # and well above the table at the pick
        self.assertGreater(T.PICK_Z - T.PAD_TIP, 0.015)

    def test_carried_cups_clear_the_mouths(self):
        self.assertGreater(T.CARRY_BOTTOM - L.RIM_H, 0.020)
        self.assertGreater(T.CARRY_BOTTOM - 0.010 - L.RIM_H, 0.010)  # still clear after a 10 mm sag
        self.assertGreater(T.ABOVE_MOUTH - T.GRIP_UP - L.RIM_H, 0.010)

    def test_the_gripper_clears_seated_neighbours(self):
        """The finger carriages (57 mm over the TCP) stay above a seated cup's lid while the next cup goes down."""
        carriage = T.SEAT_Z - T.PRESS + 0.057
        self.assertGreater(carriage - L.SEATED_TOP, 0.010)

    def test_pocket_axes_in_reach(self):
        """Each pocket's TCP point is on its own arm's side of the midline and within the arm's reach rows."""
        for cup, pocket in L.ASSIGN.items():
            x, y = L.POCKET_XY[pocket]
            side = 1 if L.CUP_START[cup][1] > 0 else -1
            self.assertGreater(side * y, 0.04)
            self.assertLessEqual(x, 0.30)
            self.assertGreaterEqual(x, 0.15)


class CompoundCups(unittest.TestCase):
    """The scorer reads the simulator's cup compounds (AssemblyState, origin at the underside) as it reads the plain
    test states (pos at the centre)."""

    @staticmethod
    def kit_cup(name, base, quat=IDENT):
        return mk.state_from_poses(T._cup(name), (*base, *quat), source="sim")

    def kit_seated(self, quat=IDENT):
        s = start()
        for c in L.CUP_NAMES:
            x, y = L.POCKET_XY[L.ASSIGN[c]]
            s[c] = self.kit_cup(c, (x, y, L.TOP + L.FLOOR_TOP), quat)
        return s

    def test_rest_states_read_as_the_start(self):
        s = start()
        for c in L.CUP_NAMES:
            s[c] = T._cup(c).rest_state()
        r = S.read(s)
        self.assertTrue(S.Tracker.at_start(r))
        for c in L.CUP_NAMES:
            self.assertAlmostEqual(float(r.pos[c][2]), L.TOP + L.CUP_H / 2)

    def test_four_seated_compounds_score_1(self):
        self.assertEqual(run(self.kit_seated()).score, 1.0)

    def test_a_compound_on_the_mouth_is_not_seated(self):
        s = self.kit_seated()
        x, y = L.POCKET_XY["far_right"]
        s["cup_d"] = self.kit_cup("cup_d", (x, y, L.TOP + L.RIM_H))
        self.assertEqual(run(s).score, 0.75)

    def test_a_tilted_compound_is_not_seated(self):
        s = self.kit_seated()
        x, y = L.POCKET_XY["near_left"]
        s["cup_a"] = self.kit_cup("cup_a", (x, y, L.TOP + L.FLOOR_TOP), about("x", 4.0))
        self.assertEqual(run(s).score, 0.75)

    def test_a_hand_on_a_lid_scores_0(self):
        x, y = L.POCKET_XY["near_left"]
        arms = dict(FAR, left=arm((x, y + 0.02, L.TOP + L.SEATED_TOP + 0.02)))
        self.assertEqual(run(self.kit_seated(), arms=arms).score, 0.0)

    def test_offline_seat_moves_a_compound(self):
        s = start()
        x, y = L.POCKET_XY["near_left"]
        s["cup_a"] = self.kit_cup("cup_a", (x + 0.003, y, L.TOP))
        out = T.offline_seat(s)
        self.assertAlmostEqual(float(out["cup_a"].origin[2]), L.TOP + L.FLOOR_TOP)
        self.assertAlmostEqual(float(out["cup_a"].low[2]), L.TOP + L.FLOOR_TOP)


class OfflineModel(unittest.TestCase):
    def test_a_released_cup_over_a_pocket_goes_to_its_floor(self):
        s = {c: cup_state(start_pos(c)) for c in L.CUP_NAMES}
        x, y = L.POCKET_XY["near_left"]
        s["cup_a"] = cup_state((x + 0.002, y, L.TOP + L.CUP_H / 2))  # dropped to the table inside the carrier's box
        out = T.offline_seat(s)
        self.assertAlmostEqual(out["cup_a"].pos[2], L.TOP + L.FLOOR_TOP + L.CUP_H / 2, places=9)
        self.assertAlmostEqual(out["cup_a"].low[2], L.TOP + L.FLOOR_TOP, places=9)
        self.assertTrue(np.allclose(out["cup_b"].pos, s["cup_b"].pos))

    def test_a_carried_cup_above_the_mouth_is_left_alone(self):
        x, y = L.POCKET_XY["near_left"]
        s = {"cup_a": cup_state((x, y, L.TOP + 0.08 + L.CUP_H / 2))}
        self.assertEqual(float(T.offline_seat(s)["cup_a"].pos[2]), L.TOP + 0.08 + L.CUP_H / 2)

    def test_a_cup_beside_a_pocket_is_left_alone(self):
        x, y = L.POCKET_XY["near_left"]
        s = {"cup_a": cup_state((x + 0.02, y, L.TOP + L.CUP_H / 2))}
        self.assertAlmostEqual(float(T.offline_seat(s)["cup_a"].pos[2]), L.TOP + L.CUP_H / 2)

    def test_the_hooks_run_offline_on_the_rest_state(self):
        states = {o.name: (o.rest_state() if hasattr(o, "rest_state") else cup_state(start_pos(o.name)))
                  for o in T.OBJECTS}
        self.assertFalse(T.success(states))
        self.assertEqual(T.score(states), 0.0)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("196 x 196 mm", "43 mm", "0.10 kg", "x 220 mm, y 0", "7 mm", "104 mm pitch", "x 168 and 272 mm",
                      "y -52 and +52 mm", "12 flat slabs", "4 mm thick", "76 mm across at the mouth",
                      "62 mm at the floor", "36 mm deep", "82 mm across", "95 mm tall", "0.25 kg", "x 190 and 310 mm",
                      "y +340 mm", "y -340 mm", "59 mm above the carrier's top", "80 mm rim", "56 mm base",
                      "88 mm body", "72 mm across"):
            self.assertIn(token, setup)
        for token in ("2 mm", "2 degrees", "8 mm", "30 mm", "15 degrees", "0.886 m/s", "40 mm", "5 degrees",
                      "50 mm", "45 degrees", "5 mm", "5 degrees", "1 s", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0",
                      "number of pockets with a seated cup divided by four"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "3000 steps (150 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_layout_numbers_match_the_text(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertAlmostEqual(L.PLATE * 1000, 196, delta=0.5)
        self.assertEqual(round((L.SEATED_TOP - L.RIM_H) * 1000), 59)
        self.assertEqual(round(2 * L.POCKET_R_RIM * 1000), 76)
        self.assertEqual(round(2 * L.POCKET_R_FLOOR * 1000), 62)
        self.assertEqual(round(L.POCKET_D * 1000), 36)
        self.assertEqual(round(L.RIM_H * 1000), 43)
        self.assertEqual(round(L.CUP_MASS * 1000), 250)
        self.assertIn("within 3.6 mm anyway", d["scoring"])
        self.assertIn("exceeds the cup's base by 4 to 8 mm", d["scoring"])
        self.assertAlmostEqual((2 * L.POCKET_R_FLOOR - L.CUP_BASE_D) * 1000, 6.0, places=6)
        self.assertAlmostEqual((2 * L.POCKET_R_FLOOR - L.cup_parts()[0][2]) * 1000, 4.9, delta=0.05)
        self.assertIn("x 190 and 310 mm", d["reset_notes"])
        self.assertIn("no mat", d["setup"])
        self.assertNotIn("mat", d["reset_notes"].replace("no mat", ""))

    def test_fidelity_states_the_simplifications(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("11 stacked 8 mm discs", "slosh", "crush", "slabs", "est", "reference"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Fast food and counter workers", "3.9 million", "BLS"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertLess(len(d["instruction"]), 300)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f.endswith(".py") else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
