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

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
MM = 0.001


def quat(axis, deg):
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def pot_at(dx=0.0, dy=0.0, z=L.POT_SEATED_Z, base=(L.SAU_ORIGIN[0], L.SAU_ORIGIN[1]), q=(0, 0, 0, 1)):
    """The pot with the centre of its foot over ``base`` + (dx, dy), at height z (default the well floor)."""
    return mk.state_from_poses(T.POT_PART, (base[0] + dx, base[1] + dy, z, *q), source="test")


def pot_start():
    return mk.state_from_poses(T.POT_PART, (*L.POT_ORIGIN, 0, 0, 0, 1), source="test")


def pot_tilted_about_corner(deg, z=L.POT_SEATED_Z):
    """The pot tipped ``deg`` about the foot's -x -y corner, which stays on the well floor, centred as seated."""
    q = quat("y", deg)  # a positive turn about y lifts the -x side... use the corner at +x so it lifts the -x edge
    r = S.rotation(q)
    corner_local = np.array([L.POT_BASE / 2, 0.0, 0.0])
    corner_world = np.array([L.SAU_ORIGIN[0] + L.POT_BASE / 2, L.SAU_ORIGIN[1], z])
    o = corner_world - r @ corner_local
    return mk.state_from_poses(T.POT_PART, (*o, *q), source="test")


def saucer_at(dx=0.0, dy=0.0, dz=0.0, q=(0, 0, 0, 1)):
    o = np.array(L.SAU_ORIGIN) + np.array([dx, dy, dz])
    return mk.state_from_poses(T.SAUCER_PART, (*o, *q), source="test")


def nb_at(dx=0.0, dy=0.0, q=(0, 0, 0, 1), dz=0.0):
    o = np.array(L.NB_ORIGIN) + np.array([dx, dy, dz])
    return mk.state_from_poses(T.NEIGHBOUR_PART, (*o, *q), source="test")


def parts(pot, saucer=None, nb=None):
    return {S.POT: pot, S.SAUCER: saucer or saucer_at(), S.NEIGHBOUR: nb or nb_at()}


def arm(tcp, opening=1.0):
    tcp = np.array(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0, 0, L.PAD_TIP]), opening=opening)


FAR_LEFT = (0.323, 0.305, L.TOP + 0.26)  # the ready pose
FAR_RIGHT = (0.323, -0.305, L.TOP + 0.26)
CLEAR = {"left": arm(FAR_LEFT), "right": arm(FAR_RIGHT)}


def holding(tcp, opening=0.6):
    return {"left": arm(FAR_LEFT), "right": arm(tcp, opening)}


def run(seq, tracker=None):
    """Feed (states, arms) pairs to a tracker; return (last result, tracker)."""
    t = tracker or S.Tracker(HZ)
    r = None
    for st, ar in seq:
        r = t.update(st, ar)
    return r, t


_DEFAULT = object()


def still(states, n=WIN + 5, arms=_DEFAULT):
    return [(states, CLEAR if arms is _DEFAULT else arms)] * n


def full_success_states():
    return parts(pot_at())


class Layout(unittest.TestCase):
    def test_fits(self):
        self.assertGreater(L.SAU_WELL, L.POT_BASE + 2 * 0.006)  # 6 mm or more each side at the floor
        self.assertAlmostEqual((L.SAU_WELL - L.POT_BASE) / 2, 0.007, places=6)
        self.assertGreater(L.BAY, L.SAU_RIM)  # the flared rim clears the stops' inner edge too
        self.assertAlmostEqual((L.BAY - L.SAU_BASE) / 2, 0.005, places=6)
        self.assertGreater(L.SAU_MOUTH, L.POT_LAYERS[0][2])
        self.assertLess(L.POT_RIM, 0.080)  # the gripper opens 94 mm: 8 mm to spare each side over the widest part
        self.assertGreater(L.GRIP_Z1 - L.GRIP_Z0, 0.022)  # the pads (22 mm long) fit inside the second layer
        widths = [w for _, _, w in L.POT_LAYERS]
        self.assertEqual(widths, sorted(widths))
        self.assertAlmostEqual(L.GRIP_W, L.POT_LAYERS[1][2])
        self.assertLessEqual(L.GRIP_W, 0.080)  # practical grasp width
        self.assertAlmostEqual(L.POT_SEATED_Z, L.TOP + 0.006)

    def test_reach_and_clearances(self):
        # the housing's lower face is about 70 mm over the TCP: it clears the succulent's top at the seat
        seat_tcp = T.SET_TCP - L.TOP
        self.assertGreaterEqual(seat_tcp + 0.070 - (0.006 + L.POT_TOP), 0.015)
        # closing along x, the gripper housing is 65 mm across (y): it stays 15 mm clear of the canopy's edge
        canopy_edge = L.BAY_A[1] - L.NB_CANOPY_R
        self.assertGreaterEqual(canopy_edge - (L.BAY_B[1] + 0.0325), 0.015)
        # the carried pot's top passes under the canopy's underside with room
        carry_foot = T.CARRY_TCP - T.GRIP_H - L.TOP
        self.assertGreaterEqual(L.NB_CANOPY_Z - L.NB_CANOPY_T / 2 + L.SAU_FLOOR - (carry_foot + L.POT_TOP), 0.020)
        # the foot clears the saucer's rim (15 mm over the table) and the drainer's edge (15 mm) by 15 mm or more
        self.assertGreaterEqual(carry_foot - (L.TRAY_FLOOR + L.SAU_H), 0.015)
        self.assertGreaterEqual(carry_foot - L.DRAIN_H, 0.015)
        # the pad tips stay above the drainer's edge when gripping
        self.assertGreater(T.GRIP_TCP - L.TOP - L.PAD_TIP, L.DRAIN_H)

    def test_parts_do_not_overlap_at_the_start(self):
        specs = {s.name: s for s in T.OBJECTS}
        boxes = {n: (s.world_bounds()) for n, s in specs.items()}

        def xy_overlap(a, b):
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            return bool(np.all(alo[:2] < bhi[:2]) and np.all(blo[:2] < ahi[:2]))

        self.assertFalse(xy_overlap(S.POT, S.SAUCER))
        self.assertFalse(xy_overlap(S.POT, S.NEIGHBOUR))
        self.assertFalse(xy_overlap("drainer", "tray"))
        self.assertFalse(xy_overlap("window", "tray"))
        # the saucer's rim stays clear of the neighbour's saucer
        self.assertGreater(abs(L.BAY_A[1] - L.BAY_B[1]), L.SAU_RIM + 0.020)

    def test_stability(self):
        for spec, half in ((T.POT_PART, L.POT_BASE / 2), (T.SAUCER_PART, L.SAU_BASE / 2),
                           (T.NEIGHBOUR_PART, L.SAU_BASE / 2)):
            com = spec.total_center_of_mass()
            tip = math.degrees(math.atan2(half, com[2]))
            self.assertGreater(tip, 30.0, f"{spec.name} tips from {tip:.0f} degrees")
            self.assertLess(float(np.hypot(com[0], com[1])), 0.001)  # centred over its base

    def test_masses_and_shapes(self):
        self.assertEqual((T.POT_PART.mass, T.SAUCER_PART.mass, T.NEIGHBOUR_PART.mass), (0.20, 0.045, 0.30))
        for fixed in (T.WINDOW, T.TRAY, T.DRAINER):
            self.assertIsNone(fixed.mass)
        # pot: 3 layers + rim ring (4) + soil + plant
        self.assertEqual(len(T.POT_PART.shapes), 3 + 4 + 1 + 1)
        # the pot's own foot is where the scorer expects it
        lo, hi = T.POT_PART.local_bounds()
        self.assertAlmostEqual(float(lo[2]), 0.0)
        self.assertAlmostEqual(float(hi[2]), L.POT_TOP)
        self.assertAlmostEqual(float(hi[0] - lo[0]), L.POT_RIM)
        # the saucer's own underside
        lo, hi = T.SAUCER_PART.local_bounds()
        self.assertAlmostEqual(float(lo[2]), 0.0)
        self.assertAlmostEqual(float(hi[2]), L.SAU_H)
        self.assertAlmostEqual(float(hi[0] - lo[0]), L.SAU_RIM)

    def test_pot_fits_the_well_in_the_shapes(self):
        """The saucer's shapes leave 72 mm between the lower walls and the pot's foot is 58 mm: no shape of the pot
        (over the first 6 mm that the well's walls reach) touches a wall at the seat."""
        well_half = L.SAU_WELL / 2
        for s in T.SAUCER_PART.shapes[1:5]:  # the lower wall ring
            inner_edge = min(abs(s.center[0]) - s.size[0] / 2 if s.size[0] < s.size[1] else math.inf,
                             abs(s.center[1]) - s.size[1] / 2 if s.size[1] < s.size[0] else math.inf)
            self.assertAlmostEqual(inner_edge, well_half, places=6)


class Seat(unittest.TestCase):
    def read(self, pot, saucer=None):
        return S.seat_reading(S.read(parts(pot, saucer)))

    def test_seated_centre(self):
        s = self.read(pot_at())
        self.assertTrue(S.seated(s))
        self.assertAlmostEqual(s["dz_min"], 0.0, places=9)
        self.assertAlmostEqual(s["centre"], 0.0, places=9)

    def test_height_window(self):
        self.assertTrue(S.seated(self.read(pot_at(z=L.POT_SEATED_Z + 0.0029))))
        self.assertFalse(S.seated(self.read(pot_at(z=L.POT_SEATED_Z + 0.0031))))
        self.assertTrue(S.seated(self.read(pot_at(z=L.POT_SEATED_Z - 0.0019))))
        self.assertFalse(S.seated(self.read(pot_at(z=L.POT_SEATED_Z - 0.0021))))

    def test_on_the_rim_is_not_seated(self):
        s = self.read(pot_at(z=L.TOP + L.TRAY_FLOOR + L.SAU_H))  # foot resting on the rim's top: 9 mm up
        self.assertFalse(S.seated(s))
        self.assertAlmostEqual(s["dz_min"], 0.009, places=6)

    def test_lateral_offset_limits(self):
        self.assertTrue(S.seated(self.read(pot_at(dx=0.0069))))  # 7 mm: the foot touches the well's wall
        self.assertTrue(S.seated(self.read(pot_at(dx=0.0075, dy=0.0))))  # inside 1 mm slack and the 8 mm centre rule
        self.assertFalse(S.seated(self.read(pot_at(dx=0.0085))))  # foot corner 37.5 mm out: over the wall
        self.assertFalse(S.seated(self.read(pot_at(dx=0.0065, dy=0.0065))))  # centre 9.2 mm off

    def test_yaw(self):
        self.assertTrue(S.seated(self.read(pot_at(q=quat("z", 10.0)))))
        self.assertFalse(S.seated(self.read(pot_at(q=quat("z", 11.0)))))
        self.assertTrue(S.seated(self.read(pot_at(q=quat("z", -9.0)))))
        self.assertTrue(S.seated(self.read(pot_at(q=quat("z", 90.0)))))  # a square looks the same a quarter turn on
        self.assertFalse(S.seated(self.read(pot_at(q=quat("z", 45.0)))))

    def test_tilt(self):
        # pot tipped 4 degrees about a foot corner that stays down: the opposite edge ends 4 mm up: too high
        s = self.read(pot_tilted_about_corner(4.0))
        self.assertFalse(S.seated(s))
        self.assertAlmostEqual(s["tilt"], 4.0, places=3)
        s = self.read(pot_tilted_about_corner(1.0))  # 1 mm up at one edge: fine
        self.assertTrue(S.seated(s))
        s = self.read(pot_tilted_about_corner(6.0))
        self.assertFalse(S.seated(s))

    def test_seated_in_a_moved_saucer(self):
        """Seated is read in the saucer's own frame: carried along with it, still seated."""
        sau = saucer_at(dx=0.05, dy=0.02, dz=0.0)
        pot = pot_at(dx=0.05, dy=0.02)
        self.assertTrue(S.seated(self.read(pot, sau)))

    def test_upside_down_pot_not_seated(self):
        s = self.read(pot_at(q=quat("x", 180.0), z=L.POT_SEATED_Z + L.POT_TOP))
        self.assertFalse(S.seated(s))


class Bay(unittest.TestCase):
    def read(self, sau):
        return S.bay_reading(S.read(parts(pot_at(), sau)))

    def test_in_bay_centred(self):
        self.assertTrue(S.in_bay(self.read(saucer_at())))

    def test_offset_limits(self):
        self.assertTrue(S.in_bay(self.read(saucer_at(dx=0.0049))))  # touching the stop
        self.assertTrue(S.in_bay(self.read(saucer_at(dy=-0.0059))))  # 0.9 mm over: within the 1 mm slack
        self.assertFalse(S.in_bay(self.read(saucer_at(dy=0.0061))))
        self.assertFalse(S.in_bay(self.read(saucer_at(dx=0.02))))

    def test_riding_a_stop(self):
        self.assertFalse(S.in_bay(self.read(saucer_at(dz=0.004))))  # up on a stop
        self.assertTrue(S.in_bay(self.read(saucer_at(dz=0.0019))))
        self.assertFalse(S.in_bay(self.read(saucer_at(dz=0.0021))))

    def test_tilt_and_yaw(self):
        self.assertFalse(S.in_bay(self.read(saucer_at(q=quat("x", 4.0)))))
        self.assertTrue(S.in_bay(self.read(saucer_at(q=quat("z", 2.0)))))  # corners 46*(cos+sin) = 47.6: inside 51
        self.assertFalse(S.in_bay(self.read(saucer_at(q=quat("z", 10.0)))))  # 53.3 mm: over the stops

    def test_in_the_other_bay_is_not_in_the_bay(self):
        self.assertFalse(S.in_bay(self.read(saucer_at(dy=L.BAY_PITCH))))


class Tiers(unittest.TestCase):
    def final(self, states, arms=_DEFAULT, n=WIN + 5):
        r, _ = run(still(states, n, arms))
        return r

    def test_full_score(self):
        r = self.final(full_success_states())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_do_nothing_scores_zero(self):
        r = self.final(parts(pot_start()))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_not_in_the_first_second(self):
        """The 1 s windows must fill before anything scores."""
        t = S.Tracker(HZ)
        results = [t.update(*pair) for pair in still(full_success_states(), WIN + 3)]
        self.assertEqual([x.score for x in results[:WIN - 1]], [0.0] * (WIN - 1))
        self.assertEqual(results[WIN].score, 1.0)  # the window of WIN+1 readings (1 s of motion) is full

    def test_half_pot_seated_saucer_out_of_bay(self):
        r = self.final(parts(pot_at(dx=0.02), saucer_at(dx=0.02)))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["seated"] and not r.details["in_bay"])
        self.assertFalse(r.success)

    def test_half_neighbour_disturbed(self):
        r = self.final(parts(pot_at(), nb=nb_at(dx=0.004)))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["nb_disturbed"])

    def test_neighbour_just_inside_limits_is_untouched(self):
        self.assertEqual(self.final(parts(pot_at(), nb=nb_at(dx=0.0029))).score, 1.0)
        self.assertEqual(self.final(parts(pot_at(), nb=nb_at(q=quat("x", 2.9)))).score, 1.0)
        self.assertEqual(self.final(parts(pot_at(), nb=nb_at(q=quat("x", 3.2)))).score, 0.5)

    def test_quarter_pot_upright_on_rim(self):
        r = self.final(parts(pot_at(z=L.TOP + L.TRAY_FLOOR + L.SAU_H)))
        self.assertEqual(r.score, 0.25)

    def test_quarter_pot_upright_on_tray_floor_beside_saucer(self):
        beside = pot_at(base=(L.TRAY_X + 0.045, L.BAY_B[1] + 0.0), z=L.TOP + L.TRAY_FLOOR)
        self.assertTrue(S.upright_in_tray(S.read(parts(beside))))
        r = self.final(parts(beside))
        self.assertEqual(r.score, 0.25)

    def test_pot_upright_beside_saucer_but_neighbour_bumped_is_zero(self):
        beside = pot_at(base=(L.TRAY_X + 0.045, L.BAY_B[1]), z=L.TOP + L.TRAY_FLOOR)
        self.assertEqual(self.final(parts(beside, nb=nb_at(dx=0.004))).score, 0.0)

    def test_saucer_shoved_inside_the_tray_with_pot_seated_is_half(self):
        r = self.final(parts(pot_at(dx=0.06), saucer_at(dx=0.06)))
        self.assertTrue(r.details["saucer_on_tray"])
        self.assertEqual(r.score, 0.5)

    def test_saucer_pushed_past_the_tray_edge_with_pot_seated_is_zero(self):
        r = self.final(parts(pot_at(dx=0.07), saucer_at(dx=0.07)))
        self.assertFalse(r.details["saucer_on_tray"])
        self.assertEqual(r.score, 0.0)

    def test_neighbour_lifted_4_mm_is_disturbed(self):
        r = self.final(parts(pot_at(), nb=nb_at(dz=0.004)))
        self.assertTrue(r.details["nb_disturbed"])
        self.assertEqual(r.score, 0.5)

    def test_seated_in_saucer_just_past_the_tray_edge_is_zero_not_quarter(self):
        dx = L.TRAY_INNER[0] / 2 + 0.001
        r = self.final(parts(pot_at(dx=dx), saucer_at(dx=dx)))
        self.assertTrue(r.details["seated"])
        self.assertFalse(r.details["saucer_on_tray"])
        self.assertEqual(r.score, 0.0)

    def test_pot_upright_on_the_table_outside_the_tray_is_zero(self):
        out = pot_at(base=(0.30, -0.25), z=L.TOP)
        self.assertEqual(self.final(parts(out)).score, 0.0)

    def test_tilted_pot_in_tray_is_zero(self):
        lying = pot_at(q=quat("y", 20.0), base=(L.TRAY_X, 0.0), z=L.TOP + 0.01)
        self.assertEqual(self.final(parts(lying)).score, 0.0)

    def test_outcomes_are_only_the_four_tiers(self):
        for st in (full_success_states(), parts(pot_start()), parts(pot_at(dx=0.02), saucer_at(dx=0.02)),
                   parts(pot_at(z=L.TOP + 0.015))):
            self.assertIn(self.final(st).score, (0.0, 0.25, 0.5, 1.0))


class Release(unittest.TestCase):
    def test_hand_on_the_pot_earns_nothing(self):
        near = holding((L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + 0.0375), opening=0.6)
        r, _ = run(still(full_success_states(), WIN + 5, near))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["released_1s"])

    def test_open_hand_beside_it_earns_nothing(self):
        # the saucer's rim (49 mm out, 15 mm under the zone's lower end) is the nearest part; the open zone's radius is
        # 56.6 mm: the gap is the hypotenuse of ``dist`` and that 15 mm
        for dist, expect in ((0.010, 0.0), (0.020, 0.0), (0.040, 1.0)):
            ar = {"left": arm(FAR_LEFT),
                  "right": arm((L.SAU_ORIGIN[0], L.SAU_ORIGIN[1] - 0.049 - dist - 0.0566, L.POT_SEATED_Z + 0.0375))}
            r, _ = run(still(full_success_states(), WIN + 5, ar))
            self.assertEqual(r.score, expect, dist)

    def test_hand_near_the_neighbour_blocks_it(self):
        ar = {"left": arm((L.NB_ORIGIN[0], L.NB_ORIGIN[1], L.TOP + 0.10), 1.0), "right": arm(FAR_RIGHT)}
        r, _ = run(still(full_success_states(), WIN + 5, ar))
        self.assertEqual(r.score, 0.0)

    def test_unreadable_arms_never_release(self):
        for bad in (None, {"left": arm(FAR_LEFT)}, {"left": arm(FAR_LEFT), "right": arm((math.nan, 0, 0))}):
            r, _ = run(still(full_success_states(), WIN + 5, bad))
            self.assertEqual(r.score, 0.0, bad)

    def test_release_then_the_window_restarts(self):
        near = holding((L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + 0.0375))
        seq = still(full_success_states(), 10, near) + still(full_success_states(), WIN - 2)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)  # clear for only WIN - 2 steps
        r, _ = run(seq + still(full_success_states(), 3))
        self.assertEqual(r.score, 1.0)


class Stillness(unittest.TestCase):
    def test_a_creeping_part_is_not_still(self):
        seq = []
        for i in range(WIN + 5):
            seq.append((parts(pot_at(dx=0.0004 * i)), CLEAR))  # 0.4 mm per step: 8 mm over a second
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["still_1s"])

    def test_a_slow_drift_under_5mm_is_still(self):
        seq = [(parts(pot_at(dx=0.00012 * i)), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)

    def test_a_turning_part_is_not_still(self):
        seq = [(parts(pot_at(q=quat("z", 0.6 * i))), CLEAR) for i in range(WIN + 5)]  # 12 degrees in a second
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_the_saucer_sliding_is_not_still(self):
        seq = [(parts(pot_at(), saucer_at(dx=0.0008 * i)), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_the_neighbour_wobbling_is_not_still(self):
        seq = [(parts(pot_at(), nb=nb_at(dx=0.0008 * (i % 2) * 8)), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)


class Latches(unittest.TestCase):
    def test_pot_tipped_at_any_time_zeroes(self):
        tipped = pot_at(q=quat("y", 40.0), z=L.TOP + 0.05)
        seq = still(parts(pot_start()), 5) + still(parts(tipped), 2) + still(full_success_states(), 2 * WIN)
        r, t = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.latch["F1"], 6)

    def test_pot_tilted_under_30_is_fine(self):
        seq = still(parts(pot_at(q=quat("y", 25.0), z=L.TOP + 0.05)), 3) + still(full_success_states(), 2 * WIN)
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)

    def test_a_part_off_the_sill_zeroes(self):
        for who in (S.POT, S.SAUCER, S.NEIGHBOUR):
            st = full_success_states()
            if who == S.POT:
                fallen = pot_at(base=(0.3, 0.7), z=L.TOP - 0.2)
            elif who == S.SAUCER:
                fallen = saucer_at(dz=-0.2)
            else:
                fallen = nb_at(dz=-0.2)
            bad = dict(st)
            bad[who] = fallen
            seq = still(bad, 2) + still(full_success_states(), 2 * WIN)
            r, t = run(seq)
            self.assertEqual(r.score, 0.0, who)
            self.assertIsNotNone(t.latch["F2"])

    def test_pot_resting_on_the_floor_below_the_sill_edge_but_not_far(self):
        """Down 9 mm (a tilted drainer, a sagging pad) is not 'off the table'."""
        st = full_success_states()
        seq = still(dict(st, **{S.NEIGHBOUR: nb_at(dz=-0.009)}), 2) + still(st, 2 * WIN)
        r, t = run(seq)
        self.assertIsNone(t.latch["F2"])

    def test_neighbour_toppled_zeroes(self):
        seq = still(parts(pot_at(), nb=nb_at(q=quat("x", 35.0))), 2) + still(full_success_states(), 2 * WIN)
        r, t = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.latch["F3"], 1)

    def test_neighbour_bumped_and_pushed_back_stays_disturbed(self):
        seq = still(parts(pot_at(), nb=nb_at(dx=0.008)), 3) + still(full_success_states(), 2 * WIN)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.5)

    def test_neighbour_tilted_4_degrees_and_back_stays_disturbed(self):
        seq = still(parts(pot_at(), nb=nb_at(q=quat("y", 4.0))), 3) + still(full_success_states(), 2 * WIN)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.5)


class BadStates(unittest.TestCase):
    def nan_states(self):
        bad = pot_at()
        object.__setattr__(bad, "origin", np.array([math.nan, 0.0, 0.0]))
        return parts(bad)

    def test_nan_scores_zero_and_restarts_the_window(self):
        t = S.Tracker(HZ)
        for pair in still(full_success_states(), WIN + 5):
            t.update(*pair)
        self.assertEqual(t.last.score, 1.0)
        r = t.update(self.nan_states(), CLEAR)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 1)
        for pair in still(full_success_states(), WIN):
            r = t.update(*pair)
        self.assertEqual(r.score, 0.0)  # the 1 s window has to refill
        r = t.update(full_success_states(), CLEAR)
        r = t.update(full_success_states(), CLEAR)
        self.assertEqual(r.score, 1.0)

    def test_nan_quaternion_and_low(self):
        for field_, value in (("quat_xyzw", np.array([math.nan, 0, 0, 1.0])), ("low", np.array([0, 0, math.nan])),
                              ("quat_xyzw", np.zeros(4))):
            bad = pot_at()
            object.__setattr__(bad, field_, value)
            r, t = run([(parts(bad), CLEAR)])
            self.assertEqual(r.score, 0.0)
            self.assertEqual(t.bad_steps, 1, field_)

    def test_inf_state(self):
        bad = saucer_at()
        object.__setattr__(bad, "origin", np.array([math.inf, 0.0, 0.0]))
        r, t = run([(parts(pot_at(), bad), CLEAR)])
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 1)

    def test_missing_part_and_junk(self):
        t = S.Tracker(HZ)
        self.assertEqual(t.update({S.POT: pot_at()}, CLEAR).score, 0.0)
        self.assertEqual(t.update(None, CLEAR).score, 0.0)
        self.assertEqual(t.update({S.POT: None, S.SAUCER: None, S.NEIGHBOUR: None}, CLEAR).score, 0.0)
        self.assertEqual(t.bad_steps, 3)

    def test_nan_latches_nothing(self):
        t = S.Tracker(HZ)
        t.update(self.nan_states(), CLEAR)
        self.assertEqual(t.latch, {"F1": None, "F2": None, "F3": None, "disturbed": None})

    def test_details_are_json_clean(self):
        r, _ = run(still(full_success_states(), WIN + 2))
        json.dumps(r.details)
        r2, _ = run([(self.nan_states(), CLEAR)])
        json.dumps(r2.details)


class Resets(unittest.TestCase):
    def test_a_new_episode_clears_the_record(self):
        tipped = pot_at(q=quat("y", 40.0), z=L.TOP + 0.05)
        t = S.Tracker(HZ)
        run(still(parts(tipped), 3), t)
        self.assertEqual(t.latch["F1"], 1)
        t.update(parts(pot_start()), CLEAR)  # the simulator puts every part back on its mark in one step
        self.assertEqual(t.resets, 1)
        self.assertIsNone(t.latch["F1"])

    def test_a_pot_set_back_on_its_mark_by_hand_is_not_a_reset(self):
        """Only if the saucer and the neighbour are on their marks too, and a part jumped 8 cm in one step."""
        t = S.Tracker(HZ)
        run(still(parts(pot_at(q=quat("y", 40.0), z=L.TOP + 0.05)), 3), t)
        t.update(parts(pot_start(), nb=nb_at(dx=0.02)), CLEAR)
        self.assertEqual(t.resets, 0)
        self.assertIsNotNone(t.latch["F1"])

    def test_a_slow_return_is_not_a_reset(self):
        t = S.Tracker(HZ)
        run(still(parts(pot_at(q=quat("y", 40.0), z=L.TOP + 0.05)), 3), t)
        for i in range(1, 11):
            f = i / 10
            o = (1 - f) * np.array([L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + 0.05]) + f * np.array(L.POT_ORIGIN)
            t.update(parts(mk.state_from_poses(T.POT_PART, (*o, 0, 0, 0, 1), source="test")), CLEAR)
        self.assertEqual(t.resets, 0)
        self.assertEqual(t.latch["F1"], 1)


class Exploits(unittest.TestCase):
    """Things a robot (or a lazy script) could try for credit it should not get."""

    def test_holding_the_pot_in_the_well_is_not_a_set_down(self):
        near = holding((L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + 0.0375), opening=0.6)
        r, _ = run(still(full_success_states(), 3 * WIN, near))
        self.assertEqual(r.score, 0.0)

    def test_carrying_the_saucer_away_with_the_pot_in_it(self):
        away = parts(pot_at(base=(0.20, -0.235), z=L.TOP + 0.006), saucer_at(dx=-0.13, dy=-0.17, dz=0.0))
        r, _ = run(still(away, WIN + 5))
        self.assertEqual(r.score, 0.0)  # seated in a saucer carried off the tray (to the drainer): nothing

    def test_dropping_the_pot_on_the_saucer_rim_is_a_quarter(self):
        r, _ = run(still(parts(pot_at(dx=0.012, z=L.TOP + L.TRAY_FLOOR + L.SAU_H)), WIN + 5))
        self.assertEqual(r.score, 0.25)

    def test_pot_beside_the_saucer_with_the_saucer_pushed_over_it_is_not_seated(self):
        """Saucer shoved 30 mm to one side, pot where it stood: not seated, not in the bay."""
        r, _ = run(still(parts(pot_at(), saucer_at(dx=0.03)), WIN + 5))
        self.assertLessEqual(r.score, 0.25)

    def test_knocking_the_neighbour_over_to_make_room_scores_zero(self):
        toppled = nb_at(dx=0.03, dy=0.06, q=quat("x", 80.0), dz=0.01)
        seq = still(parts(pot_at(), nb=toppled), WIN + 5)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_tossing_the_pot_in_from_a_height_that_tips_it_scores_zero(self):
        seq = still(parts(pot_start()), 3) + still(parts(pot_at(q=quat("x", 120.0), z=L.TOP + 0.03)), 2) \
            + still(full_success_states(), 2 * WIN)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_keeping_the_pot_hidden_from_the_distance_check(self):
        """The let-go check uses all three parts' shapes: a hand holding the neighbour's canopy up is not clear."""
        ar = {"left": arm((L.NB_ORIGIN[0], L.NB_ORIGIN[1], L.TOP + L.NB_CANOPY_Z + 0.03), 0.3), "right": arm(FAR_RIGHT)}
        r, _ = run(still(full_success_states(), WIN + 5, ar))
        self.assertEqual(r.score, 0.0)

    def test_pot_on_top_of_the_canopy_is_not_seated(self):
        up = pot_at(base=(L.BAY_A[0], L.BAY_A[1]), z=L.TOP + 0.16)
        r, _ = run(still(parts(up), WIN + 5))
        self.assertEqual(r.score, 0.0)  # perched 160 mm up, not on the tray: nothing
        self.assertFalse(r.success)

    def test_success_needs_the_hold_on_top_of_the_score(self):
        t = S.Tracker(HZ)
        first = None
        for i, pair in enumerate(still(full_success_states(), 3 * WIN)):
            r = t.update(*pair)
            if r.success and first is None:
                first = i
        self.assertEqual(first, WIN)  # not before 1 s of readings, then the task file's 1 s hold on top


class OfflineStandIn(unittest.TestCase):
    def test_support_heights(self):
        m = T.OfflineModel()
        sx, sy = L.SAU_ORIGIN[0], L.SAU_ORIGIN[1]
        self.assertAlmostEqual(m.support(sx, sy), L.POT_SEATED_Z)
        self.assertAlmostEqual(m.support(sx + 0.0069, sy - 0.0069), L.POT_SEATED_Z)
        self.assertAlmostEqual(m.support(sx + 0.012, sy), L.TOP + L.TRAY_FLOOR + L.SAU_H)  # on the rim
        self.assertAlmostEqual(m.support(sx, sy - 0.1), L.TOP)  # beyond the tray's edge (and the drainer)
        self.assertAlmostEqual(m.support(L.POT_X0, L.POT_Y0), L.TOP + L.DRAIN_FLOOR)
        self.assertAlmostEqual(m.support(L.TRAY_X + 0.05, L.BAY_B[1] + 0.0), L.TOP + L.TRAY_FLOOR + L.SAU_H)

    def test_grab_carry_release(self):
        m = T.OfflineModel()
        hold = arm((L.POT_X0, L.POT_Y0, L.POT_ORIGIN[2] + T.GRIP_H), 0.6)
        m.step({"left": arm(FAR_LEFT), "right": hold})
        self.assertEqual(m.held, "right")
        up = arm((L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + T.GRIP_H + 0.02), 0.6)
        m.step({"left": arm(FAR_LEFT), "right": up})
        np.testing.assert_allclose(m.pot, [L.SAU_ORIGIN[0], L.SAU_ORIGIN[1], L.POT_SEATED_Z + 0.02], atol=1e-9)
        m.step({"left": arm(FAR_LEFT), "right": arm(up.tcp, 1.0)})
        self.assertIsNone(m.held)
        self.assertAlmostEqual(float(m.pot[2]), L.POT_SEATED_Z)

    def test_a_far_or_open_gripper_takes_nothing(self):
        m = T.OfflineModel()
        m.step({"left": arm(FAR_LEFT), "right": arm((L.POT_X0 + 0.03, L.POT_Y0, L.POT_ORIGIN[2] + T.GRIP_H), 0.6)})
        self.assertIsNone(m.held)
        m.step({"left": arm(FAR_LEFT), "right": arm((L.POT_X0, L.POT_Y0, L.POT_ORIGIN[2] + T.GRIP_H), 1.0)})
        self.assertIsNone(m.held)

    def test_unreadable_arms_change_nothing(self):
        m = T.OfflineModel()
        m.step(None)
        np.testing.assert_allclose(m.pot, L.POT_ORIGIN)


class Demo(unittest.TestCase):
    def tearDown(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def test_demo_plan(self):
        plan = T.demo()
        self.assertEqual(set(plan), {"right"})
        self.assertLess(sum(w.duration for w in plan["right"]) * HZ, 600)  # half the 1,200 step limit

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(T.demo(), {})
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(T.demo()), {"right"})

    def test_every_gripper_change_takes_at_least_1p5_s(self):
        plan = T.right_plan()
        prev = T.OPEN
        for w in plan:
            if abs(w.finger - prev) > 0.01:
                self.assertGreaterEqual(w.duration, 1.5, w.label)
            prev = w.finger

    def test_commanded_gap_is_8mm_narrower_than_the_part(self):
        self.assertAlmostEqual(-2 * T.POT_GRIP, L.GRIP_W - 0.008, places=6)

    def test_pad_tips_never_go_below_the_drainer_floor(self):
        for w in T.right_plan():
            self.assertGreater(w.pos[2] - L.PAD_TIP, L.TOP + L.DRAIN_FLOOR + 0.005, w.label)

    def test_the_hover_clears_the_plant(self):
        self.assertGreaterEqual(T.HOVER_R - L.PAD_TIP, L.POT_ORIGIN[2] + L.POT_TOP + 0.010)

    def test_waypoints_in_reach_box(self):
        for w in T.right_plan():
            self.assertTrue(0.10 <= w.pos[0] <= 0.50 and -0.60 <= w.pos[1] <= 0.0, w.label)


class Files(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], T.INSTRUCTION)
        for key, limit in {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                           "fidelity": 490, "reset_notes": 580}.items():
            self.assertLessEqual(len(spec[key]), limit, key)
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")

    def test_no_em_dashes_or_team_names(self):
        for p in list(HERE.glob("*.py")) + list(HERE.glob("*.md")) + list(HERE.glob("*.json")) + list(
                HERE.glob("tests/*.py")):
            if p.name in ("mechkit.py", "rigstate.py"):
                continue
            text = p.read_text()
            self.assertNotIn(chr(0x2014), text, p.name)
            self.assertNotIn(chr(0x2013), text, p.name)
            self.assertIsNone(re.search("|".join(["T-" + "70", "Vi" + "han", "Aggar" + "wal", "Clau" + "de", "Anthro" + "pic"]), text), p.name)

    def test_kit_files_are_unchanged_copies(self):
        for name in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / name).exists():
                self.assertEqual((HERE / name).read_bytes(), (KIT_DIR / name).read_bytes(), name)



if __name__ == "__main__":
    unittest.main()
