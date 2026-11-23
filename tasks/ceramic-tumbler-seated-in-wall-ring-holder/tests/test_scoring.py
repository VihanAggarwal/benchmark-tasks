"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest

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

try:
    import yamkin as K  # noqa: E402
    from yam_arena import demo as D  # noqa: E402
except ModuleNotFoundError:
    K = D = None

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
SEAT = L.H_SEAT


def tilted(deg, about=(1.0, 0.0, 0.0)):
    return mk.matrix_to_quat(mk.axis_angle_matrix(about, math.radians(deg)))


def tumbler_at(x=L.RING_X, y=L.RING_Y, z=SEAT, tilt=0.0, yaw=0.0, about=(1.0, 0.0, 0.0), source="test"):
    """The tumbler with its base centre at (x, y, z over the table), tilted ``tilt`` degrees about its base centre."""
    q = mk.matrix_to_quat(mk.axis_angle_matrix(about, math.radians(tilt)) @ mk.axis_angle_matrix((0, 0, 1), yaw))
    return mk.state_from_poses(T.TUMBLER_PART, (x, y, L.TOP + z, *q), source=source)


def hung_off(off, tilt=0.0):
    """Hung with its axis ``off`` m from the ring's centre at the ring's height and tilted ``tilt`` degrees (a tilt
    about the base centre moves the axis at that height, so the base is shifted to put the axis where wanted)."""
    run = (L.RING_H - SEAT) * math.tan(math.radians(tilt))  # how far the axis has moved sideways at the ring's height
    return tumbler_at(x=L.RING_X + off - run, tilt=tilt, about=(0.0, 1.0, 0.0))


def holder(dx=0.0):
    o = L.RING_X + dx
    return mk.state_from_poses(T.HOLDER_PART, (o, L.RING_Y, L.TOP, 0, 0, 0, 1), source="spec")


def scene(t=None, h=None):
    return {S.TUMBLER: t if t is not None else tumbler_at(), S.HOLDER: h if h is not None else holder()}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.60, L.TOP + 0.25))}
GRIPPING = {"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + SEAT + T.GRIP_TCP), 0.72)}
BACKED = {"left": FAR["left"], "right": arm((T.RETREAT_X, L.RING_Y, L.TOP + T.CLEAR_TCP), 1.0)}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def surface_radius(state, z, direction=(1.0, 0.0)):
    """Radius at which the tumbler's outer surface lies along ``direction`` from its axis at height ``z`` above its
    base (found from outside: the outermost point still inside the solid)."""
    d = np.asarray(direction, dtype=float)
    lo, hi = 0.0, 0.06  # lo: inside the solid, hi: outside it
    lo = 0.027
    for _ in range(40):
        mid = (lo + hi) / 2
        p = np.array([[state.origin[0] + d[0] * mid, state.origin[1] + d[1] * mid, state.origin[2] + z]])
        if state.distance(p)[0] > 1e-9:
            hi = mid
        else:
            lo = mid
    return lo


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.RING_ID, 0.068)
        self.assertAlmostEqual(L.RING_BAND, 0.008)
        self.assertAlmostEqual(L.RING_T, 0.003)
        self.assertAlmostEqual(L.TUM_H, 0.100)
        self.assertAlmostEqual(L.TUM_D_BASE, 0.060)
        self.assertAlmostEqual(L.TUM_D_RIM, 0.076)
        self.assertAlmostEqual(L.TUM_WALL, 0.005)
        self.assertAlmostEqual(L.TUM_MASS, 0.28)
        self.assertAlmostEqual(L.H_SEAT_ON_TUM, 0.050, places=6)
        self.assertAlmostEqual(L.H_SEAT, 0.040, places=6)
        self.assertAlmostEqual(L.STRAP_LEN, 0.088)

    def test_parts_spawn_where_they_rest(self):
        st = start()
        t, h = st[S.TUMBLER], st[S.HOLDER]
        self.assertAlmostEqual(float(t.low[2]), L.TOP, places=6)
        self.assertAlmostEqual(float(h.low[2]), L.TOP, places=6)
        self.assertAlmostEqual(float(t.high[2]) - L.TOP, L.TUM_H, delta=0.0006)
        self.assertAlmostEqual(float(t.origin[0]), L.TUM_START[0])
        self.assertAlmostEqual(float(t.origin[1]), L.TUM_START[1])
        self.assertIsNone(T.HOLDER_PART.mass)  # the holder is fixed

    def test_tumbler_wall_is_the_stated_cone(self):
        t = tumbler_at(x=0.2, y=0.0, z=0.0)
        for z, want in ((0.010, 0.0308), (0.050, 0.0340), (0.090, 0.0372)):
            r = surface_radius(t, z)  # along a slab's normal: the facet's flat, apothem
            self.assertAlmostEqual(r, want, delta=0.0004, msg=f"z={z}")
        self.assertAlmostEqual(float(t.high[0] - t.low[0]) / 2, 0.038, delta=0.0006)  # rim 76 mm

    def test_tumbler_is_hollow_with_a_floor(self):
        t = tumbler_at(x=0.2, y=0.0, z=0.0)
        inside = lambda z: t.distance(np.array([[0.2, 0.0, L.TOP + z]]))[0]
        self.assertGreater(inside(0.060), 0.02)  # open cup: the middle is air up to the mouth
        self.assertGreater(inside(0.100), 0.02)
        self.assertEqual(inside(0.002), 0.0)  # the floor
        self.assertGreater(inside(0.012), 0.004)  # 5 mm floor

    def test_ring_hole_and_band(self):
        h = holder()
        zr = L.TOP + L.RING_H - L.RING_T / 2
        for k in range(3, 62):  # facet normals and vertices of the 32-gon, not through the strap at 0 degrees
            a = 2 * math.pi * k / 64
            c, s = math.cos(a), math.sin(a)

            def at(r):
                return h.distance(np.array([[L.RING_X + r * c, L.RING_Y + r * s, zr]]))[0]
            self.assertGreater(at(0.0339), 0.0, f"hole clear at {k}")
            self.assertEqual(at(0.0385), 0.0, f"band at {k}")  # inside the 8 mm band
            self.assertGreater(at(0.0425), 0.0, f"outside at {k}")
        self.assertAlmostEqual(h.distance(np.array([[L.RING_X + 0.0345, L.RING_Y, zr]]))[0], 0.0, delta=1e-9)
        top = L.TOP + L.RING_H
        self.assertEqual(h.distance(np.array([[L.RING_X + 0.038, L.RING_Y, top - 0.0005]]))[0], 0.0)
        self.assertAlmostEqual(h.distance(np.array([[L.RING_X + 0.038, L.RING_Y, top + 0.005]]))[0], 0.005, places=6)

    def test_tumbler_base_passes_the_ring_with_4_mm_to_spare(self):
        self.assertAlmostEqual(L.RING_R_IN - L.TUM_R_BASE, 0.004)
        self.assertAlmostEqual(L.seat_floor(0.0), L.H_SEAT, places=6)
        self.assertAlmostEqual(L.seat_floor(0.004), L.RING_H, places=6)
        self.assertIsNone(L.seat_floor(0.0041))

    def test_polygon_error_is_inside_the_hung_window(self):
        flat_to_corner = L.RING_R_IN * (1 / math.cos(math.pi / L.RING_SIDES) - 1) / L.TUM_TAN
        self.assertLess(flat_to_corner, 0.0025)
        self.assertLess(flat_to_corner, L.HUNG_TOL / 3)

    def test_backboard_stands_behind_the_wrist(self):
        board = L.BOARD_X0
        self.assertGreater(board, L.RING_X + 0.10 + 0.02)  # link 4 reaches ~0.10 m past the TCP
        self.assertAlmostEqual(L.FLANGE_X, L.RING_X + L.RING_R_OUT + L.STRAP_LEN)

    def test_mass_and_density(self):
        vol = sum(s.volume() for s in T.TUMBLER_PART.shapes)
        dens = L.TUM_MASS / vol
        self.assertTrue(1500 < dens < 3500, dens)  # ceramic is 2300; overlapping slabs count twice
        com = T.TUMBLER_PART.center_of_mass()
        self.assertTrue(0.035 < com[2] < 0.060, com)


class Tiers(unittest.TestCase):
    def test_hung_scores_full(self):
        r = run(scene())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["hands_gap_mm"] >= 30.0, True)

    def test_start_scores_zero(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_tier_table(self):
        cases = [
            (tumbler_at(tilt=0.0), 1.0),
            (tumbler_at(tilt=5.0), 1.0),
            (hung_off(0.0, 8.0), 0.5),
            (hung_off(0.0, 14.0), 0.5),
            (hung_off(0.0, 20.0), 0.25),
            (tumbler_at(tilt=70.0), 0.0),
            (hung_off(0.004), 1.0),
            (hung_off(0.008), 0.5),
            (hung_off(0.011), 0.5),
            (hung_off(0.016), 0.25),
            (tumbler_at(z=0.055), 0.25),  # caught high on the ring, not hung
            (tumbler_at(z=0.028), 0.25),  # low: past the window but off the table
            (tumbler_at(z=0.012), 0.0),
            (tumbler_at(z=0.0), 0.0),
            (tumbler_at(x=0.45, y=-0.14, z=0.04), 0.0),  # up in the air far from the ring
            (tumbler_at(x=L.RING_X + 0.07, z=0.04), 0.0),
        ]
        for s, want in cases:
            self.assertEqual(run(scene(t=s)).score, want, (s.origin, want))

    def test_window_edges(self):
        for z, want in ((0.0321, 1.0), (0.0479, 1.0), (0.0315, 0.25), (0.0485, 0.25)):
            self.assertEqual(run(scene(t=tumbler_at(z=z))).score, want, z)
        for tilt, want in ((5.9, 1.0), (6.2, 0.5), (14.8, 0.5), (15.3, 0.25)):
            self.assertEqual(run(scene(t=hung_off(0.0, tilt))).score, want, tilt)
        for off, want in ((0.0058, 1.0), (0.0063, 0.5), (0.0118, 0.5), (0.0123, 0.25)):
            self.assertEqual(run(scene(t=hung_off(off))).score, want, off)
        for off, want in ((0.058, 0.25), (0.062, 0.0)):  # caught within 60 mm of the ring's axis
            self.assertEqual(run(scene(t=tumbler_at(x=L.RING_X + off, z=0.05))).score, want, off)
        for z, want in ((0.0205, 0.25), (0.0195, 0.0)):  # caught from 20 mm up
            self.assertEqual(run(scene(t=tumbler_at(z=z))).score, want, z)
        for tilt, want in ((59.0, 0.25), (61.0, 0.0)):  # caught while the mouth is within 60 degrees of up
            self.assertEqual(run(scene(t=tumbler_at(z=0.05, tilt=tilt))).score, want, tilt)

    def test_only_the_stated_scores_exist(self):
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(400):
            s = tumbler_at(x=rng.uniform(0.1, 0.4), y=rng.uniform(-0.3, 0.1), z=rng.uniform(-0.01, 0.16),
                           tilt=rng.uniform(0, 200), yaw=rng.uniform(-3, 3),
                           about=tuple(rng.normal(size=3) * np.array([1, 1, 0.01]) + np.array([0, 0, 0.0001])))
            seen.add(run(scene(t=s), steps=WIN + 1).score)
        self.assertTrue(seen <= {0.0, 0.25, 0.5, 1.0}, seen)

    def test_tumbler_upside_down_on_the_ring_scores_zero(self):
        """Exploit: set the tumbler on the ring mouth down (its 76 mm rim is wider than the hole): not hung."""
        s = tumbler_at(z=L.RING_H + 0.001 + 0.100, tilt=180.0)
        r = run(scene(t=s))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["hung"] or r.details["caught"])

    def test_tumbler_lying_across_the_ring_scores_zero(self):
        s = tumbler_at(z=L.RING_H + 0.04, tilt=90.0)
        self.assertEqual(run(scene(t=s)).score, 0.0)

    def test_tumbler_standing_on_the_table_under_the_ring_scores_zero(self):
        self.assertEqual(run(scene(t=tumbler_at(z=0.0))).score, 0.0)

    def test_tumbler_upright_beside_the_ring_on_the_table_scores_zero(self):
        self.assertEqual(run(scene(t=tumbler_at(x=L.RING_X + 0.1, z=0.0))).score, 0.0)


class Latches(unittest.TestCase):
    def _drop_run(self, z0, z1, steps=1):
        """Held at z0 for a while, then the base goes to z1 over ``steps`` control steps, then hands off and still."""
        tr = S.Tracker()
        hand = {"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + z0 + T.GRIP_TCP), 0.72)}
        for _ in range(5):
            tr.update(scene(t=tumbler_at(z=z0)), hand)
        for z in np.linspace(z0, z1, steps + 1)[1:]:
            tr.update(scene(t=tumbler_at(z=float(z))), hand)
        res = run(scene(t=tumbler_at(z=z1)), arms=BACKED, tracker=tr, steps=WIN + 3)
        return tr, res

    def test_dropped_from_30_mm_is_a_drop(self):
        """Exploit: let go 30 mm over the seat and let it fall onto the ring: seated at the end, still 0 (it chips)."""
        tr, res = self._drop_run(SEAT + 0.030, SEAT)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(res.score, 0.0)

    def test_dropped_from_15_mm_is_not_a_drop(self):
        tr, res = self._drop_run(SEAT + 0.015, SEAT)
        self.assertIsNone(tr.latch["F1"])
        self.assertEqual(res.score, 1.0)

    def test_drop_boundary_is_20_mm(self):
        self.assertIsNotNone(self._drop_run(SEAT + 0.0205, SEAT)[0].latch["F1"])
        self.assertIsNone(self._drop_run(SEAT + 0.0195, SEAT)[0].latch["F1"])

    def test_drop_over_three_steps_counts_but_not_over_four(self):
        self.assertIsNotNone(self._drop_run(SEAT + 0.030, SEAT, steps=3)[0].latch["F1"])  # 10 mm a step
        self.assertIsNone(self._drop_run(SEAT + 0.030, SEAT, steps=6)[0].latch["F1"])  # 5 mm a step: 15 mm in 3

    def test_banged_down_by_a_fast_hand_is_a_drop(self):
        self.assertIsNotNone(self._drop_run(SEAT + 0.080, SEAT, steps=8)[0].latch["F1"])  # 10 mm a step: 200 mm/s

    def test_demo_speed_lowering_is_not_a_drop(self):
        tr = S.Tracker()
        for z in np.linspace(0.12, SEAT, 120):  # 80 mm over 6 s
            tr.update(scene(t=tumbler_at(z=float(z))),
                      {"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + z + T.GRIP_TCP), 0.72)})
        self.assertIsNone(tr.latch["F1"])
        self.assertLess(tr.max_drop, 0.005)

    def _fast_air_run(self, x, z0, z1, steps):
        """Held, the base taken from z0 to z1 at (x, ring y) over ``steps`` steps, stopped dead there, still held."""
        tr = S.Tracker()
        for z in [z0] * 5 + [float(v) for v in np.linspace(z0, z1, steps + 1)[1:]] + [z1] * 5:
            tr.update(scene(t=tumbler_at(x=x, z=z)),
                      {"left": FAR["left"], "right": arm((x, L.RING_Y, L.TOP + z + T.GRIP_TCP), 0.72)})
        return tr

    def test_fast_hand_stopping_in_free_air_beside_the_ring_is_not_a_drop(self):
        tr = self._fast_air_run(L.RING_X - 0.15, 0.20, 0.06, steps=7)  # 20 mm a step, stops 60 mm over the table
        self.assertIsNone(tr.latch["F1"])
        self.assertGreaterEqual(tr.max_drop, 0.059)

    def test_fast_hand_stopping_in_free_air_over_the_ring_is_not_a_drop(self):
        tr = self._fast_air_run(L.RING_X, 0.25, L.RING_H + L.SUPPORT_TOL + 0.003, steps=8)  # stops 8 mm over the top
        self.assertIsNone(tr.latch["F1"])

    def test_fast_hand_banging_it_onto_the_table_is_a_drop(self):
        tr = self._fast_air_run(L.RING_X - 0.15, 0.10, 0.0, steps=5)  # 20 mm a step onto the table
        self.assertIsNotNone(tr.latch["F1"])

    def test_slowing_to_a_stop_on_the_seat_is_not_a_drop(self):
        """A hand that comes down fast but slows (the last step under 6.5 mm) before the seat is not an impact."""
        tr = S.Tracker()
        zs = [SEAT + 0.060, SEAT + 0.045, SEAT + 0.030, SEAT + 0.018, SEAT + 0.009, SEAT + 0.003, SEAT, SEAT]
        for z in zs:
            tr.update(scene(t=tumbler_at(z=z)),
                      {"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + z + T.GRIP_TCP), 0.72)})
        self.assertIsNone(tr.latch["F1"])

    def test_lowest_point_and_support(self):
        self.assertAlmostEqual(S.lowest(S.read(scene(t=tumbler_at(z=0.0)))), 0.0, places=9)
        lying = S.read(scene(t=tumbler_at(x=0.1, z=0.038, tilt=90.0)))  # on its rim circle
        self.assertAlmostEqual(S.lowest(lying), 0.0, places=6)
        self.assertTrue(S.supported(lying))
        self.assertTrue(S.supported(S.read(scene(t=tumbler_at(z=SEAT)))))
        self.assertFalse(S.supported(S.read(scene(t=tumbler_at(x=L.RING_X - 0.15, z=0.06)))))

    def test_knocked_off_the_table_is_a_drop(self):
        tr = S.Tracker()
        run(start(), tracker=tr, steps=3)
        run(scene(t=tumbler_at(x=0.5, y=-0.3, z=-0.30, tilt=90.0)), tracker=tr, steps=2)
        self.assertIsNotNone(tr.latch["F1"])

    def test_drop_stays_latched(self):
        tr, _ = self._drop_run(SEAT + 0.030, SEAT)
        self.assertEqual(run(scene(), arms=BACKED, tracker=tr, steps=3 * WIN).score, 0.0)

    def test_holder_moved_is_zero(self):
        tr = S.Tracker()
        self.assertEqual(run(scene(h=holder(dx=0.008)), tracker=tr).score, 0.0)
        self.assertIsNotNone(tr.latch["F2"])
        self.assertEqual(run(scene(h=holder(dx=0.004))).score, 1.0)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_tumbler_scores_zero(self):
        """Exploit: leave the gripper holding the tumbler in the seat."""
        self.assertEqual(run(scene(), arms=GRIPPING).score, 0.0)

    def test_open_gripper_around_the_tumbler_scores_zero(self):
        arms = {"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + SEAT + T.GRIP_TCP), 1.0)}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_hand_resting_on_the_holder_scores_zero(self):
        arms = {"left": FAR["left"], "right": arm((L.BOARD_X0 - 0.07, L.RING_Y, L.TOP + 0.12), 1.0)}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_backed_away_scores_full(self):
        r = run(scene(), arms=BACKED)
        self.assertEqual(r.score, 1.0)
        self.assertGreaterEqual(r.details["hands_gap_mm"], 30.0)

    def test_retreat_clears_30_mm_with_margin(self):
        d = rigstate.hands_gap(BACKED, T.HOLDER_PART.rest_state().distance)
        self.assertGreater(d, 0.040)
        tum = scene()
        both = lambda p: np.minimum(tum[S.TUMBLER].distance(p), tum[S.HOLDER].distance(p))
        self.assertGreater(rigstate.hands_gap(BACKED, both), 0.040)

    def test_left_arm_at_ready_is_clear(self):
        ready = K.tcp_pose(np.array(K.READY), np.array(K.BASES["station"]["left"], float)) if K else None
        if ready is None:
            self.skipTest("no yamkin")
        p, r = ready
        arms = {"left": arm(p, 1.0), "right": BACKED["right"]}
        self.assertEqual(run(scene(), arms=arms).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        run(scene(), arms=GRIPPING, tracker=tr)
        self.assertEqual(tr.update(scene(), BACKED).score, 0.0)
        self.assertEqual(run(scene(), arms=BACKED, tracker=tr, steps=WIN).score, 1.0)

    def _drift(self, per_step_mm, turn_deg=0.0):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            t = tumbler_at(x=L.RING_X + per_step_mm * 0.001 * i, tilt=turn_deg * i / WIN, about=(0, 0, 1.0),
                           yaw=0.0)
            res = tr.update(scene(t=t), BACKED)
        return res

    def test_moving_tumbler_scores_zero(self):
        self.assertEqual(self._drift(0.5).score, 0.0)

    def test_stillness_window(self):
        self.assertEqual(self._drift(0.04).score, 1.0)  # 0.84 mm over the second
        self.assertEqual(self._drift(0.07).score, 0.0)  # 1.47 mm

    def test_turning_tumbler_scores_zero(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            q = mk.matrix_to_quat(mk.axis_angle_matrix((0, 0, 1.0), math.radians(1.0 * i / WIN * 1.0)))
            t = mk.state_from_poses(T.TUMBLER_PART, (L.RING_X, L.RING_Y, L.TOP + SEAT, *q), source="test")
            res = tr.update(scene(t=t), BACKED)
        self.assertEqual(res.score, 0.0)  # 1 degree over the second
        tr = S.Tracker()
        for i in range(WIN + 2):
            q = mk.matrix_to_quat(mk.axis_angle_matrix((0, 0, 1.0), math.radians(0.4 * i / WIN)))
            t = mk.state_from_poses(T.TUMBLER_PART, (L.RING_X, L.RING_Y, L.TOP + SEAT, *q), source="test")
            res = tr.update(scene(t=t), BACKED)
        self.assertEqual(res.score, 1.0)  # 0.4 degrees


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({S.TUMBLER: tumbler_at()}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)

    def test_nan_state(self):
        bad = mk.state_from_poses(T.TUMBLER_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(t=bad), FAR).score, 0.0)
        inf = mk.state_from_poses(T.TUMBLER_PART, (0.2, 0.0, float("inf"), 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(t=inf), FAR).score, 0.0)

    def test_nan_state_clears_the_windows(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.TUMBLER_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(tr.update(scene(t=bad), FAR).score, 0.0)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_nan_arms_never_release(self):
        nan = {"left": arm((float("nan"), 0.45, 1.0)), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=nan).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(scene(h=holder(dx=0.02)), tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_start_pose_alone_is_not_a_reset(self):
        tr = S.Tracker()
        run(start(), tracker=tr, steps=3)
        self.assertEqual(tr.resets, 0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            s = tumbler_at(x=rng.uniform(0.15, 0.35), y=rng.uniform(-0.3, 0.0), z=rng.uniform(-0.02, 0.2),
                           tilt=rng.uniform(0, 180), about=tuple(rng.normal(size=3)))
            r = tr.update(scene(t=s, h=holder(dx=rng.choice([0.0, 0.0, 0.01]))), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


def replay(variant="", idle="", extra_steps=None):
    """Play the demo through the checker's rules (ArmPlan + yamkin, rigstate fed offline) into task.success()."""
    rigstate.offline_reset()
    mk.offline_reset()
    old = {k: os.environ.get(k) for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
    os.environ.pop("DEMO_HOLD_ONLY", None)
    os.environ["DEMO_VARIANT"] = variant
    os.environ["DEMO_IDLE_ARM"] = idle
    try:
        plans = T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    bases = {a: np.array(K.BASES["station"][a], float) for a in ("left", "right")}
    aps = {a: D.ArmPlan(list(w), bases[a]) for a, w in plans.items()}
    end = max((p.duration for p in aps.values()), default=0.0)
    n = int(math.floor((end + 1.0) * HZ)) + 1 if extra_steps is None else extra_steps
    objs = start()
    first_success, last, hist = None, None, []
    for k in range(n):
        t = (k + 1) / HZ
        snap = {}
        for a in ("left", "right"):
            if a in aps:
                act, _ = aps[a].action7(t)
                q, finger = act[:6], float(act[6]) * K.FINGER_OPEN
            else:
                q, finger = np.array(K.READY, float), K.FINGER_OPEN
            p, r = K.tcp_pose(q, bases[a])
            snap[a] = {"tcp": p, "tips": K.pad_tip(p, r), "opening": float(np.clip(finger / K.FINGER_OPEN, 0, 1))}
        rigstate.offline_feed(snap)
        ok = T.success(objs)
        sc = T.score(objs)
        hist.append(sc)
        if ok and first_success is None:
            first_success = k + 1
        last = sc
    return {"score": last, "first_success": first_success, "steps": n, "hist": hist, "plan_s": end}


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["holder", "tumbler"])
        self.assertIsNone(T.HOLDER_PART.mass)
        self.assertEqual(T.TUMBLER_PART.mass, L.TUM_MASS)
        self.assertEqual(T.SCENE, "station")
        self.assertEqual(T.SCORE_OVER, "end")

    def test_instruction_states_the_goal_only(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertLess(len(T.INSTRUCTION), 300)
        for word in ("32", "48", "6 mm", "degree", "40 mm"):
            self.assertNotIn(word, T.INSTRUCTION)

    def test_demo_plans(self):
        os.environ.pop("DEMO_IDLE_ARM", None)
        os.environ.pop("DEMO_HOLD_ONLY", None)
        os.environ.pop("DEMO_VARIANT", None)
        plans = T.demo()
        self.assertEqual(sorted(plans), ["right"])
        self.assertTrue(all(isinstance(w, T.Waypoint) for w in plans["right"]))

    def test_demo_env_switches(self):
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            os.environ.pop("DEMO_HOLD_ONLY")
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(T.demo(), {})
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(sorted(T.demo()), ["right"])  # the left arm has no plan anyway
            os.environ.pop("DEMO_IDLE_ARM")
            os.environ["DEMO_VARIANT"] = "nonsense"
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
                os.environ.pop(k, None)

    def test_demo_timing(self):
        total = sum(w.duration for w in T.plan())
        self.assertLessEqual(total, 25.0)
        self.assertLessEqual((total + 1.0) * HZ + 2, 0.5 * 1000)  # the demo and its 1 s hold within half the limit
        self.assertTrue(all(w.duration >= 1.5 for w in T.plan() if w.label.startswith(("grip", "let go"))))
        self.assertTrue(all(w.duration >= 0.5 for w in T.plan()))

    def test_grip_geometry(self):
        self.assertAlmostEqual(T.GRIP_TCP, 0.075)
        gap = -2 * T.GRIP
        self.assertAlmostEqual(gap, L.TUM_D_RIM - 0.008)
        self.assertTrue(16 * 0.001 <= gap <= 80 * 0.001 + 1e-9 or gap < 0.094)
        self.assertLessEqual(L.TUM_D_RIM, 0.080)  # within the 16 to 80 mm grasp widths
        # the pad tips end 12 mm over the ring's top face when hung, the pad top at the rim
        self.assertAlmostEqual(T.GRIP_TIPS_UP - L.H_SEAT_ON_TUM, 0.012, places=6)
        self.assertGreaterEqual(T.GRIP_TIPS_UP - L.H_SEAT_ON_TUM - T.PRESS, 0.008)

    def test_heights_inside_the_reach(self):
        for w in T.plan():
            self.assertLessEqual(w.pos[2] - L.TOP, 0.185, w.label)
        self.assertAlmostEqual(T.CARRY_TCP, L.RING_H + 0.015 + T.GRIP_TCP)
        self.assertGreaterEqual(T.CARRY_BASE - L.RING_H, 0.012)

    def test_carry_and_seat_positions(self):
        wps = T.plan()
        over = [w for w in wps if w.label.startswith("carry")][0]
        self.assertEqual((over.pos[0], over.pos[1]), (L.RING_X, L.RING_Y))
        press = [w for w in wps if w.label.startswith("slowly down until")][0]
        self.assertAlmostEqual(press.pos[2] - L.TOP, L.H_SEAT + T.GRIP_TCP - T.PRESS, places=6)

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_every_waypoint_reachable_with_margin(self):
        base = tuple(K.BASES["station"]["right"])
        for w in T.plan():
            ok, q, pe, re = K.reachable(w.pos, yaw=w.yaw, base_pos=base)
            self.assertTrue(ok, w.label)
            self.assertGreater(float(np.degrees(K.joint_margins(q)).min()), 8.0, w.label)

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_arm_clears_the_backboard(self):
        """The right arm's wrist (link 4) and gripper stay 20 mm or more from every real part of the holder."""
        import check_task as C
        import scenegeom as G
        base = np.array(K.BASES["station"]["right"], float)
        solids = []
        for i, s in enumerate(T.HOLDER_PART.shapes):
            x, y, z, w = (0, 0, 0, 1) if s.quat is None else s.quat
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c = np.array(s.center) + np.array([L.RING_X, L.RING_Y, L.TOP])
            solids.append(G.Solid(f"h{i}", "box", tuple(c), tuple(np.array(s.size) / 2), yaw))
        ap = D.ArmPlan(T.plan(), base)
        worst = 1.0
        for k in range(int(ap.duration * HZ) + 1):
            act, _ = ap.action7(k / HZ)
            caps = [c for c in K.arm().capsules(act[:6], base, float(act[6]) * K.FINGER_OPEN)
                    if c[0] not in K.FINGER_LINKS]
            worst = min(worst, min(C.capsules_gap(caps, s)[0] for s in solids))
        self.assertGreater(worst, 0.020)

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_full_demo_replay_scores_one(self):
        r = replay()
        self.assertEqual(r["score"], 1.0)
        self.assertIsNotNone(r["first_success"])
        self.assertLessEqual(r["steps"], 520)
        self.assertLess(max(r["hist"][:300]), 1.0)  # nothing scores early

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_variant_high_drop_scores_zero(self):
        r = replay("high")
        self.assertEqual(r["score"], 0.0)
        self.assertIsNone(r["first_success"])

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_variant_keep_holding_scores_zero(self):
        r = replay("keep")
        self.assertEqual(r["score"], 0.0)

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_hold_only_and_right_arm_idle_score_zero(self):
        self.assertEqual(replay(idle="right", extra_steps=500)["score"], 0.0)
        old = os.environ.get("DEMO_HOLD_ONLY")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
            if old is not None:
                os.environ["DEMO_HOLD_ONLY"] = old

    @unittest.skipIf(K is None, "yamkin not importable")
    def test_left_arm_idle_changes_nothing(self):
        self.assertEqual(replay(idle="left")["score"], 1.0)


class OfflineModel(unittest.TestCase):
    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)

    def test_grip_carry_and_seat(self):
        m = T.OfflineModel()
        tx, ty = L.TUM_START
        m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + T.GRIP_TCP), 0.72)})
        m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + 0.15), 0.72)})
        self.assertAlmostEqual(m.pos[2], 0.15 - T.GRIP_TCP, places=6)
        m.step({"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + 0.05), 0.72)})  # pressed past the seat
        self.assertAlmostEqual(m.pos[2], L.H_SEAT, places=6)
        m.step({"left": FAR["left"], "right": arm((L.RING_X, L.RING_Y, L.TOP + 0.05), 1.0)})
        self.assertAlmostEqual(m.pos[2], L.H_SEAT, places=6)

    def test_open_hand_does_not_carry(self):
        m = T.OfflineModel()
        tx, ty = L.TUM_START
        m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + T.GRIP_TCP), 1.0)})
        m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + 0.15), 1.0)})
        self.assertEqual(m.pos[2], 0.0)

    def test_hand_too_high_or_off_to_the_side_does_not_grip(self):
        tx, ty = L.TUM_START
        for tcp in ((tx, ty, L.TOP + 0.150), (tx + 0.03, ty, L.TOP + T.GRIP_TCP), (tx, ty, L.TOP + 0.020)):
            m = T.OfflineModel()
            m.step({"left": FAR["left"], "right": arm(tcp, 0.72)})
            m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + 0.15), 0.72)})
            self.assertEqual(m.pos[2], 0.0, tcp)

    def test_off_centre_over_the_ring_rests_on_it_then_falls(self):
        m = T.OfflineModel()
        tx, ty = L.TUM_START
        m.step({"left": FAR["left"], "right": arm((tx, ty, L.TOP + T.GRIP_TCP), 0.72)})
        m.step({"left": FAR["left"], "right": arm((L.RING_X + 0.010, L.RING_Y, L.TOP + 0.09), 0.72)})
        self.assertAlmostEqual(m.pos[2], L.RING_H, places=6)  # on the ring's top: more than 4 mm off-centre
        m.step({"left": FAR["left"], "right": arm((L.RING_X + 0.010, L.RING_Y, L.TOP + 0.09), 1.0)})
        self.assertEqual(m.pos[2], 0.0)

    def test_model_is_offline_only(self):
        self.assertTrue(T.no_simulator({S.TUMBLER: T.OfflineModel().states()[S.TUMBLER]}))
        self.assertFalse(T.no_simulator({S.TUMBLER: tumbler_at(source="sim")}))


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("68 mm inside diameter", "8 mm wide band", "3 mm thick", "90 mm above the table", "x 240 mm",
                      "y -140 mm", "88 mm long", "36 x 60 mm", "18 mm thick", "400 mm wide", "300 mm high",
                      "x 373 mm", "0.28 kg", "100 mm tall", "60 mm in diameter", "76 mm", "5 mm wall",
                      "x 220 mm, y -300 mm", "50 mm up", "40 mm above the table", "140 mm", "4 mm to spare",
                      "hollow", "fixed"):
            self.assertIn(token, setup)
        for token in ("32 to 48 mm", "20 mm above the table", "60 mm of the ring", "30 mm or more", "under 1 mm",
                      "0.5 degrees", "5 mm", "20 mm or more", "6 degrees", "15 degrees", "6 mm", "12 mm",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["run_limit"], "1000 steps (50 s at 20 Hz)")
        self.assertAlmostEqual(L.RING_H, 0.090)
        self.assertAlmostEqual(L.BOARD_X0, 0.373, places=6)
        self.assertEqual((L.HUNG_TOL, L.TILT_FULL_DEG, L.TILT_HALF_DEG, L.OFF_FULL, L.OFF_HALF), (0.008, 6.0, 15.0,
                                                                                                    0.006, 0.012))
        self.assertEqual((L.CAUGHT_BASE_MIN, L.CAUGHT_NEAR, L.FALL_DROP, L.HOLDER_MOVE, L.LET_GO),
                         (0.020, 0.060, 0.020, 0.005, 0.030))
        scores = d.get("scores", "")  # before the GPU run: TBD; after it: the real Dream-1 run and demo 1.0
        pat = r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?(; one arm idle: [^;]+)?"
        self.assertTrue(scores == "" or scores.startswith("TBD") or re.fullmatch(pat, scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("32-sided", "hollow", "est", "No software rules"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("SOC 37-2012", "0.28 kg", "wall"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile(chr(0x2014) + "|" + chr(0x2013) + "|co-authored|claude|anthropic|token|password|api[_ ]key",
                         re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)
        self.assertIn("Third-party assets: none", text)


if __name__ == "__main__":
    unittest.main()
