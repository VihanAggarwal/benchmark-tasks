"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import math
import os
import pathlib
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
CLOSED = L.LID_OPEN  # lid joint value, closed
ORDER = L.CELL_PICK_ORDER


def egg_state(k, x, y, z=None, tilt=0.0, speed=0.0, flip=False):
    """Egg k with its frame origin at (x, y) and ``z`` above the table (seated height by default), turned ``tilt`` rad
    about x (``flip``: pointed end up)."""
    z = L.EGG_Z_SEATED if z is None else z
    q = mk.quat_from_ypr(0.0, 0.0, math.pi if flip else tilt)
    return mk.state_from_poses(T.EGG_PARTS[k], (x, y, L.TOP + z, *q), (0.0, 0.0, -speed, 0.0, 0.0, 0.0),
                               source="test")


def carton_state(lid=CLOSED):
    return mk.state_from_poses(T.CARTON, (*T.CARTON.pos, 0, 0, 0, 1), joints={"lid": lid}, source="test")


def packed(lid=CLOSED, cells=None, **kw):
    """Six eggs seated in the six cells (egg k in cell ORDER[k]); ``cells`` maps k to another (x, y) or None."""
    out = {"carton": carton_state(lid)}
    for k in range(6):
        x, y = L.CELLS[ORDER[k]]
        out[f"e{k}"] = egg_state(k, x, y, **kw)
    for k, xy in (cells or {}).items():
        if xy is not None:
            out[f"e{k}"] = egg_state(k, *xy)
    return out


def start():
    out = {f"e{k}": s.rest_state() for k, s in enumerate(T.EGG_PARTS)}
    out["carton"] = T.CARTON.rest_state()
    return out


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def one_off(k, **kw):
    """Packed, with egg k replaced by ``egg_state(k, ...)``."""
    s = packed()
    s[f"e{k}"] = egg_state(k, **kw)
    return s


class Layout(unittest.TestCase):
    def test_seat_heights(self):
        self.assertAlmostEqual(L.C_H0, 0.0035, places=4)
        self.assertAlmostEqual(L.F_H0, 0.0020, places=4)
        self.assertAlmostEqual(L.EGG_Z_SEATED, 0.034, places=4)  # centre at the rim
        top_over_rim = L.C_H0 + L.EGG_L - L.C_D
        self.assertLess(top_over_rim, L.LID_CLEAR - 0.005)  # a seated egg is 7 mm under the lid's plate

    def test_egg_is_44_by_57(self):
        self.assertAlmostEqual(max(L.SLICE_R) * 2, 0.044, delta=0.0006)
        self.assertAlmostEqual(L.SLICE_H * L.EGG_SLICES, 0.057, places=6)
        self.assertLess(L.SLICE_R[0], L.SLICE_R[-1] + 0.001)  # the pointed end is the narrower
        self.assertGreater(max(L.SLICE_R[:6]), 0.0)

    def test_funnel_inner_face_runs_from_floor_to_rim(self):
        for R, r, D in ((L.C_R, L.C_r, L.C_D), (L.F_R, L.F_r, L.F_D)):
            slabs = L.funnel_slabs(0.0, 0.0, R, r, D, L.FLOOR_T)
            (c, size, ypr) = slabs[0]  # the +x wall
            a = ypr[1]
            n_in = np.array([-math.cos(a), 0.0, math.sin(a)])  # inward normal of the inner face
            inner_mid = np.array(c) + n_in * size[0] / 2
            self.assertAlmostEqual(inner_mid[0], (R + r) / 2, places=6)
            self.assertAlmostEqual(inner_mid[2], L.FLOOR_T + D / 2, places=6)
            # the face leans outward going up by (R - r) over D
            t = np.array([math.sin(a), 0.0, math.cos(a)])
            top = inner_mid + t * size[2] / 2
            self.assertAlmostEqual(top[0], R, places=6)
            self.assertAlmostEqual(top[2], L.FLOOR_T + D, places=6)

    def test_y_walls_mirror_x_walls(self):
        slabs = L.funnel_slabs(0.0, 0.0, L.C_R, L.C_r, L.C_D, L.FLOOR_T)
        (cx, sx, yx), (_, _, _), (cy, sy, yy) = slabs[0], slabs[1], slabs[2]
        self.assertAlmostEqual(cx[0], cy[1], places=9)
        self.assertAlmostEqual(cx[2], cy[2], places=9)
        self.assertAlmostEqual(yx[1], -yy[2], places=9)

    def test_lid_drawn_open_with_its_free_edge_up_and_back(self):
        x, z = L.lid_point_world(0.0, L.LID_LEN, 0.0)
        self.assertLess(x, L.CX + L.HINGE_X)  # past vertical, toward the arms
        self.assertGreater(z, L.C_RIM + 0.09)
        xc, zc = L.lid_point_world(L.LID_OPEN, L.LID_LEN, 0.0)
        self.assertAlmostEqual(xc, L.CX + L.HINGE_X + L.LID_LEN, places=6)
        self.assertAlmostEqual(zc, L.C_RIM, places=6)

    def test_lid_boxes_match_lid_point(self):
        boxes = L.lid_boxes()
        plate_centre = np.array(boxes[0][0])
        x, z = L.lid_point_world(0.0, L.LID_LEN / 2, L.LID_CLEAR + L.LID_T / 2)
        self.assertAlmostEqual(plate_centre[0] + L.CX, x, places=6)
        self.assertAlmostEqual(plate_centre[2], z, places=6)

    def test_eggs_start_in_flat_pockets_away_from_each_other_along_y(self):
        for (xa, ya) in L.EGG_START:
            for (xb, yb) in L.EGG_START:
                if (xa, ya) != (xb, yb) and abs(xa - xb) < 0.02:
                    self.assertGreater(abs(ya - yb), 0.15)  # the gripper closes along y: no neighbour in its way

    def test_cells_are_a_2_by_3_grid(self):
        self.assertEqual(len(L.CELLS), 6)
        xs = sorted({round(c[0], 6) for c in L.CELLS})
        ys = sorted({round(c[1], 6) for c in L.CELLS})
        self.assertEqual((len(xs), len(ys)), (2, 3))
        self.assertAlmostEqual(xs[1] - xs[0], L.C_PITCH, places=6)


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(packed())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["n_seated"], 6)

    def test_lid_open_is_three_quarters(self):
        self.assertEqual(run(packed(lid=0.0)).score, 0.75)

    def test_lid_just_open_is_three_quarters(self):
        self.assertEqual(run(packed(lid=CLOSED - 0.060)).score, 0.75)

    def test_lid_within_tolerance_is_closed(self):
        self.assertEqual(run(packed(lid=CLOSED - 0.040)).score, 1.0)

    def test_five_seated_half(self):
        s = packed()
        s["e5"] = egg_state(5, 0.30, 0.20, z=L.EGG_L / 2)  # the sixth on the table
        self.assertEqual(run(s).score, 0.5)

    def test_four_seated_half(self):
        s = packed()
        s["e4"] = egg_state(4, 0.30, 0.20, z=L.EGG_L / 2)
        s["e5"] = egg_state(5, 0.30, 0.25, z=L.EGG_L / 2)
        self.assertEqual(run(s).score, 0.5)

    def test_three_seated_quarter(self):
        s = packed()
        for k in (3, 4, 5):
            s[f"e{k}"] = egg_state(k, 0.30, 0.17 + 0.03 * k, z=L.EGG_L / 2)
        self.assertEqual(run(s).score, 0.25)

    def test_one_seated_quarter(self):
        s = {"carton": carton_state(0.0)}
        for k in range(6):
            s[f"e{k}"] = egg_state(k, 0.30, 0.17 + 0.03 * k, z=L.EGG_L / 2)
        x, y = L.CELLS[0]
        s["e0"] = egg_state(0, x, y)
        self.assertEqual(run(s).score, 0.25)

    def test_no_egg_seated_is_zero(self):
        s = {"carton": carton_state(CLOSED)}
        for k in range(6):
            s[f"e{k}"] = egg_state(k, 0.30, 0.17 + 0.03 * k, z=L.EGG_L / 2)
        self.assertEqual(run(s).score, 0.0)

    def test_start_layout_is_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_tier_function(self):
        self.assertEqual(S.tier(False, True, 6, True), 1.0)
        self.assertEqual(S.tier(False, True, 6, False), 0.75)
        self.assertEqual(S.tier(False, True, 5, True), 0.5)
        self.assertEqual(S.tier(False, True, 4, False), 0.5)
        self.assertEqual(S.tier(False, True, 3, True), 0.25)
        self.assertEqual(S.tier(False, True, 1, True), 0.25)
        self.assertEqual(S.tier(False, True, 0, True), 0.0)
        self.assertEqual(S.tier(True, True, 6, True), 0.0)
        self.assertEqual(S.tier(False, False, 6, True), 0.0)


class SeatedRules(unittest.TestCase):
    def test_egg_12_mm_off_centre_counts(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x + 0.011, y=y)).score, 1.0)

    def test_egg_14_mm_off_centre_does_not(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x + 0.014, y=y)).score, 0.5)

    def test_egg_standing_high_does_not_count(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, z=L.EGG_Z_SEATED + 0.010)).score, 0.5)

    def test_egg_5_mm_high_counts(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, z=L.EGG_Z_SEATED + 0.005)).score, 1.0)

    def test_egg_tipped_30_degrees_does_not_count(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, tilt=math.radians(30))).score, 0.5)

    def test_egg_tipped_20_degrees_counts(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, tilt=math.radians(20))).score, 1.0)

    def test_egg_pointed_end_up_does_not_count(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, flip=True)).score, 0.5)

    def test_egg_lying_on_its_side_in_a_cell_does_not_count(self):
        x, y = L.CELLS[ORDER[2]]
        self.assertEqual(run(one_off(2, x=x, y=y, tilt=math.pi / 2)).score, 0.5)

    def test_two_eggs_in_one_cell_count_once(self):
        s = packed()
        x, y = L.CELLS[ORDER[0]]
        s["e1"] = egg_state(1, x, y, z=L.EGG_Z_SEATED + 0.060)
        self.assertEqual(S.filled(S.read(s)), sorted(set(ORDER) - {ORDER[1]}))
        self.assertEqual(run(s).score, 0.5)

    def test_two_eggs_at_seated_height_in_one_cell_count_once(self):
        s = packed()
        x, y = L.CELLS[ORDER[0]]
        s["e1"] = egg_state(1, x, y)
        self.assertEqual(S.read(s) is not None and len(S.filled(S.read(s))), 5)
        self.assertEqual(run(s).score, 0.5)

    def test_eggs_left_in_the_flat_score_nothing(self):
        s = {"carton": carton_state(CLOSED)}
        for k, (x, y) in enumerate(L.EGG_START):
            s[f"e{k}"] = egg_state(k, x, y, z=L.EGG_START_Z)
        self.assertEqual(run(s).score, 0.0)


class Cracks(unittest.TestCase):
    def test_fast_egg_latches(self):
        tr = S.Tracker()
        tr.update(one_off(0, x=0.2, y=0.2, z=0.04, speed=0.9), FAR)
        self.assertIsNotNone(tr.latch["cracked"])
        self.assertEqual(tr.latch["cracked_egg"], "e0")

    def test_slow_egg_does_not(self):
        tr = S.Tracker()
        tr.update(one_off(0, x=0.2, y=0.2, z=0.04, speed=0.7), FAR)
        self.assertIsNone(tr.latch["cracked"])

    def test_crack_zeroes_a_perfect_end_state_for_good(self):
        tr = S.Tracker()
        tr.update(one_off(0, x=0.2, y=0.2, z=0.04, speed=1.2), FAR)
        self.assertEqual(run(packed(), tracker=tr).score, 0.0)
        self.assertFalse(run(packed(), tracker=tr).success)

    def test_egg_off_the_table_latches(self):
        tr = S.Tracker()
        tr.update(one_off(3, x=0.5, y=0.9, z=-0.3), FAR)
        self.assertIsNotNone(tr.latch["cracked"])

    def test_egg_on_the_table_is_not_cracked(self):
        tr = S.Tracker()
        tr.update(one_off(3, x=0.5, y=0.2, z=L.EGG_L / 2), FAR)
        self.assertIsNone(tr.latch["cracked"])

    def test_thrown_egg_that_lands_seated_still_scores_zero(self):
        """Exploit: fling the egg into its cell. It arrives seated, but it hit at over 0.8 m/s on the way."""
        tr = S.Tracker()
        x, y = L.CELLS[ORDER[0]]
        tr.update(one_off(0, x=x, y=y, z=0.05, speed=1.1), FAR)
        self.assertEqual(run(packed(), tracker=tr).score, 0.0)

    def test_dropping_from_above_the_cell_cracks(self):
        tr = S.Tracker()
        x, y = L.CELLS[ORDER[1]]
        for z, v in ((0.12, 0.0), (0.10, 0.5), (0.06, 1.1), (L.EGG_Z_SEATED, 0.0)):
            tr.update(one_off(1, x=x, y=y, z=z, speed=v), FAR)
        self.assertIsNotNone(tr.latch["cracked"])


class LidExploits(unittest.TestCase):
    def test_lid_closed_over_an_empty_carton_is_nothing(self):
        s = {"carton": carton_state(CLOSED)}
        for k, (x, y) in enumerate(L.EGG_START):
            s[f"e{k}"] = egg_state(k, x, y, z=L.EGG_START_Z)
        self.assertEqual(run(s).score, 0.0)

    def test_lid_held_shut_by_a_hand_is_nothing(self):
        near = {"left": FAR["left"],
                "right": arm((L.CX, L.CY, L.TOP + L.C_RIM + L.LID_CLEAR + 0.020), 0.0)}
        self.assertEqual(run(packed(), arms=near).score, 0.0)

    def test_lid_swinging_is_not_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(packed(lid=CLOSED - 0.25 - 0.01 * i), FAR)  # about 12 degrees over the second
        self.assertEqual(res.score, 0.0)

    def test_lid_drifting_under_5_degrees_is_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(packed(lid=CLOSED - 0.25 - 0.002 * i), FAR)  # about 3 degrees over the second
        self.assertEqual(res.score, 0.75)

    def test_lid_resting_on_a_high_egg_is_not_closed(self):
        s = packed(lid=CLOSED - 0.25)
        self.assertEqual(run(s).score, 0.75)

    def test_cannot_score_with_lid_closed_and_an_egg_standing_high(self):
        x, y = L.CELLS[ORDER[2]]
        s = one_off(2, x=x, y=y, z=L.EGG_Z_SEATED + 0.020)
        s["carton"] = carton_state(CLOSED)
        self.assertEqual(run(s).score, 0.5)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_carton(self):
        arms = {"left": arm((L.CX, L.CY, L.TOP + 0.09), 1.0), "right": FAR["right"]}
        self.assertEqual(run(packed(), arms=arms).score, 0.0)

    def test_hand_on_an_egg(self):
        x, y = L.CELLS[ORDER[0]]
        arms = {"left": FAR["left"], "right": arm((x, y, L.TOP + 0.07), 0.5)}
        self.assertEqual(run(packed(), arms=arms).score, 0.0)

    def test_parked_hands_are_clear(self):
        arms = {s: arm((T.PARK[s][0], T.PARK[s][1], L.TOP + T.PARK[s][2])) for s in ("left", "right")}
        self.assertEqual(run(packed(), arms=arms).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(packed(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        x, y = L.CELLS[ORDER[0]]
        near = {"left": arm((x, y, L.TOP + 0.07), 0.5), "right": FAR["right"]}
        run(packed(), arms=near, tracker=tr)
        self.assertEqual(tr.update(packed(), FAR).score, 0.0)
        self.assertEqual(run(packed(), tracker=tr, steps=WIN).score, 1.0)

    def test_moving_egg(self):
        tr = S.Tracker()
        res = None
        x, y = L.CELLS[ORDER[2]]
        for i in range(WIN + 2):
            res = tr.update(one_off(2, x=x + 0.0005 * i, y=y), FAR)
        self.assertEqual(res.score, 0.0)

    def test_still_for_the_whole_second_needed(self):
        tr = S.Tracker()
        run(packed(), tracker=tr, steps=WIN - 2)
        self.assertEqual(tr.last.score, 0.0)
        self.assertEqual(run(packed(), tracker=tr, steps=4).score, 1.0)

    def test_egg_turning_in_place_is_not_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(one_off(0, x=L.CELLS[ORDER[0]][0], y=L.CELLS[ORDER[0]][1], tilt=0.01 * i), FAR)  # 12 degrees
        self.assertEqual(res.score, 0.0)

    def test_small_creep_under_5_mm_and_5_degrees_is_still(self):
        tr = S.Tracker()
        x, y = L.CELLS[ORDER[2]]
        res = None
        for i in range(WIN + 2):
            res = tr.update(one_off(2, x=x + 0.00015 * i, y=y, tilt=math.radians(0.1) * i), FAR)  # 3 mm, 2 degrees
        self.assertEqual(res.score, 1.0)

    def test_creep_over_5_mm_is_not_still(self):
        tr = S.Tracker()
        x, y = L.CELLS[ORDER[2]]
        res = None
        for i in range(WIN + 2):
            res = tr.update(one_off(2, x=x + 0.0003 * i, y=y), FAR)  # 6.3 mm
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        s = packed()
        s.pop("e2")
        r = S.Tracker().update(s, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_missing_lid_joint_is_invalid(self):
        s = packed()
        s["carton"].joints.pop("lid")
        self.assertFalse(S.Tracker().update(s, FAR).details["valid_state"])

    def test_nan_position(self):
        s = packed()
        s["e1"] = mk.state_from_poses(T.EGG_PARTS[1], (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(s, FAR).score, 0.0)

    def test_nan_lid(self):
        r = S.Tracker().update(packed(lid=float("nan")), FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_velocity(self):
        s = packed()
        x, y = L.CELLS[ORDER[0]]
        s["e0"] = egg_state(0, x, y, speed=float("nan"))
        self.assertFalse(S.Tracker().update(s, FAR).details["valid_state"])

    def test_nan_resets_the_windows(self):
        tr = S.Tracker()
        run(packed(), tracker=tr, steps=WIN + 2)
        tr.update(packed(lid=float("nan")), FAR)
        self.assertEqual(tr.update(packed(), FAR).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(one_off(0, x=0.2, y=0.2, z=0.04, speed=1.5), FAR)
        self.assertIsNotNone(tr.latch["cracked"])
        tr.update(packed(), FAR)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            s = {"carton": carton_state(float(rng.uniform(0.0, CLOSED)))}
            for k in range(6):
                s[f"e{k}"] = egg_state(k, float(rng.uniform(0.15, 0.4)), float(rng.uniform(-0.1, 0.4)),
                                       z=float(rng.uniform(0.0, 0.15)), tilt=float(rng.uniform(-3, 3)),
                                       speed=float(rng.uniform(0.0, 0.5)))
            r = tr.update(s, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class DemoPlans(unittest.TestCase):
    def test_demo_has_both_arms(self):
        self.assertEqual(set(T.demo()), {"left", "right"})

    def test_hold_only_has_no_plan(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_idle_arm_drops_that_plan(self):
        for arm_name, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = arm_name
            try:
                self.assertEqual(set(T.demo()), {other})
            finally:
                del os.environ["DEMO_IDLE_ARM"]

    def test_demo_length_within_run_limit(self):
        p = T.plans()
        steps = max(sum(w.duration for w in plan) for plan in p.values()) * HZ
        self.assertLessEqual(steps, 0.8 * 3600)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_variants_build(self):
        for v in T.VARIANTS:
            self.assertEqual(set(T.plans(v)), {"left", "right"})

    def test_lid_arc_restored_after_a_variant(self):
        before = T.LID_ARC_DEG
        T.plans("lidshort")
        self.assertEqual(T.LID_ARC_DEG, before)

    def test_right_arm_waits_until_the_left_has_parked(self):
        p = T.plans()
        left_total = sum(w.duration for w in p["left"])
        t = 0.0
        for w in p["right"]:
            if w.label.startswith("over the open lid"):
                break
            t += w.duration
        self.assertGreaterEqual(t, left_total - 0.01)

    def test_every_egg_is_picked_once_and_placed_in_its_own_cell(self):
        self.assertEqual(sorted(L.CELL_PICK_ORDER), list(range(6)))
        labels = [w.label for w in T.plans()["left"]]
        self.assertEqual(sum(1 for t in labels if t.startswith("close on the egg")), 6)
        self.assertEqual(sum(1 for t in labels if t.startswith("open the grip")), 6)

    def test_task_text_files_exist(self):
        self.assertTrue((HERE / "task.json").exists())
        self.assertTrue((HERE / "README.md").exists())


def feed_arms(tcp_left, gap_left, tcp_right=None, gap_right=0.0939):
    tr = tcp_right if tcp_right is not None else np.array([0.1, -0.55, L.TOP + 0.25])
    snap = {"left": arm(tcp_left, min(1.0, gap_left / rigstate.FULL_GAP)),
            "right": arm(tr, min(1.0, gap_right / rigstate.FULL_GAP))}
    rigstate.offline_feed({a: {"tcp": s.tcp, "tips": s.tips, "opening": s.opening} for a, s in snap.items()})
    return snap


class OfflineStandIn(unittest.TestCase):
    def setUp(self):
        self.m = T.OfflineModel()

    def test_model_states_are_complete(self):
        st = self.m.states({})
        self.assertEqual(set(st), set(S.PARTS))
        self.assertTrue(S.Tracker().update(st, FAR).details["valid_state"])

    def test_model_starts_at_the_start(self):
        self.assertTrue(S.Tracker().at_start(S.read(self.m.states({}))))

    def test_model_carries_an_egg_and_sets_it_in_a_cell(self):
        x, y = L.EGG_START[0]
        grip_z = L.TOP + T.PICK_TCP
        feed_arms((x, y, L.TOP + 0.08), 0.0939)
        self.m.step({})
        feed_arms((x, y, grip_z), 0.033)
        self.m.step({})
        self.assertIn("left", self.m.held)
        cx, cy = L.CELLS[0]
        feed_arms((cx, cy, L.TOP + 0.135), 0.033)
        self.m.step({})
        feed_arms((cx, cy, L.TOP + T.PLACE_TCP), 0.033)
        self.m.step({})
        feed_arms((cx, cy, L.TOP + T.PLACE_TCP), 0.045)
        self.m.step({})
        self.assertNotIn("left", self.m.held)
        self.assertAlmostEqual(float(self.m.pos["e0"][2]), L.TOP + L.EGG_Z_SEATED, places=6)
        s = S.read(self.m.states({}))
        self.assertEqual(S.seated(s, "e0"), 0)

    def test_model_drop_gives_impact_speed(self):
        x, y = L.EGG_START[0]
        feed_arms((x, y, L.TOP + T.PICK_TCP), 0.033)
        self.m.step({})
        cx, cy = L.CELLS[0]
        feed_arms((cx, cy, L.TOP + T.PLACE_TCP + 0.070), 0.033)
        self.m.step({})
        feed_arms((cx, cy, L.TOP + T.PLACE_TCP + 0.070), 0.045)
        self.m.step({})
        s = S.read(self.m.states({}))
        self.assertGreaterEqual(s.speed["e0"], L.CRACK_SPEED)

    def test_pad_pushes_the_lid_and_it_falls_shut(self):
        face = L.LID_CLEAR + L.LID_T
        for psi in T.LID_ARC_DEG:
            x, z = T.lid_tip(psi, face - T.LID_INTO)
            feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939, tcp_right=np.array([x - T.PAD_FACE, L.CY,
                                                                              L.TOP + z + 0.013]), gap_right=0.0)
            self.m.step({})
        self.assertLess(self.m.psi, 60.0)  # pushed to about 50 degrees
        feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939, tcp_right=np.array([0.14, -0.4, L.TOP + 0.15]), gap_right=0.0939)
        self.m.step({})
        self.assertEqual(self.m.psi, 0.0)  # under its weight

    def test_lid_swings_back_if_only_pushed_to_100(self):
        face = L.LID_CLEAR + L.LID_T
        for psi in (110.0, 105.0, 100.0):
            x, z = T.lid_tip(psi, face - T.LID_INTO)
            feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939, tcp_right=np.array([x - T.PAD_FACE, L.CY,
                                                                              L.TOP + z + 0.013]), gap_right=0.0)
            self.m.step({})
        feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939, tcp_right=np.array([0.14, -0.4, L.TOP + 0.15]), gap_right=0.0939)
        self.m.step({})
        self.assertEqual(self.m.psi, L.LID_OPEN_DEG)

    def test_standing_high_egg_keeps_the_lid_open(self):
        cx, cy = L.CELLS[0]
        self.m.pos["e0"] = np.array([cx, cy, L.TOP + L.EGG_Z_SEATED + 0.015])
        self.m.psi = 60.0
        feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939)
        self.m.step({})
        self.assertGreater(self.m.psi, 5.0)

    def test_model_ignores_a_pad_beside_the_lid(self):
        face = L.LID_CLEAR + L.LID_T
        x, z = T.lid_tip(110.0, face - T.LID_INTO)
        feed_arms((0.1, 0.55, L.TOP + 0.25), 0.0939,
                  tcp_right=np.array([x - T.PAD_FACE, L.CY + 0.2, L.TOP + z + 0.013]), gap_right=0.0)
        self.m.step({})
        self.assertEqual(self.m.psi, L.LID_OPEN_DEG)


class TaskJson(unittest.TestCase):
    def test_fields_and_limits(self):
        import json
        d = json.loads((HERE / "task.json").read_text())
        lim = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
               "reset_notes": 580}
        for k, n in lim.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertEqual(d["category"], "Societal relatability")
        self.assertNotIn("—", json.dumps(d))

    def test_thresholds_in_text_match_the_code(self):
        import json
        d = json.loads((HERE / "task.json").read_text())
        sc = d["scoring"]
        for token in ("12 mm", "6 mm", "25 degrees", "30 mm", "5 mm", "5 degrees"):
            self.assertIn(token, sc, token)
        self.assertEqual(L.CELL_XY, 0.012)
        self.assertEqual(L.SEAT_TOL, 0.006)
        self.assertEqual(L.POINT_DOWN_DEG, 25.0)
        self.assertEqual(L.LID_CLOSED_TOL, 0.005)
        self.assertEqual(L.LET_GO, 0.030)
        self.assertEqual(L.STILL_MOVE, 0.005)
        self.assertEqual(L.STILL_TURN_DEG, 5.0)
        self.assertAlmostEqual(L.STILL_LID, math.radians(5.0))

    def test_no_em_dash_in_task_files(self):
        for name in ("task.py", "scoring.py", "layout.py", "README.md", "task.json"):
            self.assertNotIn("—", (HERE / name).read_text(), name)


if __name__ == "__main__":
    unittest.main()
