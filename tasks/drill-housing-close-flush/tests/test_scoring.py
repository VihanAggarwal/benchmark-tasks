"""Scorer and geometry tests (NumPy only; run from the task folder: python3 -m unittest discover -s tests)."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import scoring as S  # noqa: E402

O = np.array(L.world((0.0, 0.0, 0.0)))
SEAT = np.array(L.world(L.SWITCH_SEAT))


def quat_z(a: float):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def quat_axis(axis, a: float):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    s = math.sin(a / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(a / 2))


def st(origin, quat=(0.0, 0.0, 0.0, 1.0)):
    origin = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=origin, pos=origin.copy(), quat_xyzw=np.asarray(quat, dtype=float),
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=origin, high=origin)


def arm(tips, opening=1.0):
    tips = np.asarray(tips, dtype=float)
    return SimpleNamespace(tcp=tips + np.array([0, 0, 0.013]), tips=tips, opening=opening)


FAR = {"left": arm((0.22, 0.20, L.TOP + 0.1)), "right": arm((0.15, -0.20, L.TOP + 0.1))}


def scene(motor=None, switch=None, cover=None):
    return {"motor": motor or st(O), "switch": switch or st(SEAT), "cover": cover or st(O)}


def settle(tr, objs, arms=FAR, steps=12):
    r = None
    for _ in range(steps):
        r = tr.update(objs, arms)
    return r


def score(objs, arms=FAR):
    return settle(S.Tracker(), objs, arms)


# -- geometry: what fits where (sampled volumes against the fixed half's boxes) ------------------------------------
def _samples(name, origin, rot, step=0.0015):
    pts = []
    for sh in S.SHAPES[name]:
        if sh[0] == "box":
            g = [np.arange(sh[1][i], sh[2][i] + 1e-9, step) for i in range(3)]
            x, y, z = np.meshgrid(*g, indexing="ij")
            pts.append(np.stack([x.ravel(), y.ravel(), z.ravel()], 1))
        else:
            _, x0, x1, r = sh
            th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            pts.append(np.array([(x, q * math.cos(t), q * math.sin(t)) for x in np.arange(x0, x1 + 1e-9, step)
                                 for q in np.arange(0, r + 1e-9, step) for t in th]))
    return np.asarray(origin) + np.concatenate(pts) @ np.asarray(rot).T


def _boxes(d, origin=O):
    return {k: (origin + np.array([b[0][0], b[1][0], b[2][0]]), origin + np.array([b[0][1], b[1][1], b[2][1]]))
            for k, b in d.items()}


FIXED = {**_boxes(L.LOWER), **_boxes(L.NEST)}


def overlaps(pts, boxes, margin=0.0002):
    return sorted(k for k, (lo, hi) in boxes.items() if np.any(np.all((pts > lo + margin) & (pts < hi - margin), 1)))


def rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


class Geometry(unittest.TestCase):
    def test_seated_parts_and_closed_cover_fit(self):
        mot, sw = _samples("motor", O, np.eye(3)), _samples("switch", SEAT, np.eye(3))
        self.assertEqual(overlaps(mot, FIXED), [])
        self.assertEqual(overlaps(sw, FIXED), [])
        cover = _boxes(L.COVER)
        self.assertEqual(overlaps(mot, cover), [])
        self.assertEqual(overlaps(sw, cover), [])
        self.assertEqual(overlaps(_samples("cover", O, np.eye(3), 0.002), FIXED), [])

    def test_unit_rests_on_its_rib_floors(self):
        floor_top = L.LOWER["rib_can_0"][2][1]
        self.assertAlmostEqual(floor_top, -L.R_UNIT, places=9)
        self.assertAlmostEqual(L.LOWER["rib_gearbox_0"][2][1], -L.R_UNIT, places=9)

    def test_turned_round_parts_do_not_fit(self):
        self.assertNotEqual(overlaps(_samples("motor", O, rz(math.pi)), FIXED), [])
        self.assertNotEqual(overlaps(_samples("switch", SEAT, rz(math.pi)), FIXED), [])

    def test_unit_axial_play_is_3_5_mm(self):
        for dx in (0.0034, -0.0034):
            self.assertEqual(overlaps(_samples("motor", O + [dx, 0, 0], np.eye(3)), FIXED), [])
        self.assertNotEqual(overlaps(_samples("motor", O + [0.005, 0, 0], np.eye(3)), FIXED), [])
        self.assertNotEqual(overlaps(_samples("motor", O + [-0.005, 0, 0], np.eye(3)), FIXED), [])

    def test_tongue_is_the_first_contact_of_a_misplaced_cover(self):
        mot, sw = _samples("motor", O, np.eye(3)), _samples("switch", SEAT, np.eye(3))
        for d in ((0.0018, 0, 0), (-0.0018, 0, 0), (0, 0.0018, 0), (0, -0.0018, 0)):
            c = O + np.array(d)
            self.assertEqual(overlaps(_samples("cover", c, np.eye(3), 0.002), FIXED), [], d)
            self.assertEqual(overlaps(mot, _boxes(L.COVER, c)), [], d)
            self.assertEqual(overlaps(sw, _boxes(L.COVER, c)), [], d)
        for d in ((0.0035, 0, 0), (-0.0035, 0, 0), (0, 0.0035, 0), (0, -0.0035, 0)):
            c = O + np.array(d)
            self.assertNotEqual(overlaps(_samples("cover", c, np.eye(3), 0.002), FIXED)
                                + overlaps(mot, _boxes(L.COVER, c)), [], d)

    def test_raised_parts_hold_the_cover_off(self):
        # a unit 3 mm high (on a stop or a step) meets the cover's ribs; so does a switch 3 mm high
        self.assertNotEqual(overlaps(_samples("motor", O + [0, 0, 0.003], np.eye(3)), _boxes(L.COVER)), [])
        self.assertNotEqual(overlaps(_samples("switch", SEAT + [0, 0, 0.003], np.eye(3)), _boxes(L.COVER)), [])

    def test_starts_are_clear_of_everything(self):
        tray = {f"rail{i}": (np.array([x0, y0, L.TOP]), np.array([x1, y1, L.TOP + L.TRAY_H]))
                for i, ((x0, x1), (y0, y1)) in enumerate(L.TRAY_RAILS)}
        allfix = {**FIXED, **tray}
        self.assertEqual(overlaps(_samples("motor", L.UNIT_START, np.eye(3)), allfix), [])
        self.assertEqual(overlaps(_samples("switch", L.SWITCH_START, np.eye(3)), allfix), [])
        self.assertEqual(overlaps(_samples("cover", L.COVER_START, np.eye(3), 0.002), allfix), [])
        lo, hi = S.local_bounds("cover")
        self.assertGreater(L.COVER_START[0] + hi[0], 0.0)
        self.assertLess(L.COVER_START[0] + hi[0], L.OX + L.BARREL_X[0])  # in front of the left half, not under it
        self.assertAlmostEqual(L.COVER_START[2] + lo[2], L.TOP, places=9)  # resting on its tongue

    def test_unit_rests_on_the_tray_rails(self):
        lo = L.UNIT_START[2] - L.TOP
        self.assertAlmostEqual(math.hypot(lo - L.TRAY_H, 0.008), L.R_UNIT, places=9)


class Scoring(unittest.TestCase):
    def test_full_score(self):
        r = score(scene())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["max_rim_gap_mm"], 0.0)

    def test_cover_off_both_seated(self):
        r = score(scene(cover=st(L.COVER_START)))
        self.assertEqual(r.score, 0.5)

    def test_partial_parts(self):
        off_motor = st(L.UNIT_START)
        off_switch = st(L.SWITCH_START)
        self.assertEqual(score(scene(motor=off_motor, cover=st(L.COVER_START))).score, 0.2)
        self.assertEqual(score(scene(switch=off_switch, cover=st(L.COVER_START))).score, 0.3)
        self.assertEqual(score(scene(motor=off_motor, switch=off_switch, cover=st(L.COVER_START))).score, 0.0)

    def test_cover_closed_over_a_missing_part_earns_only_the_parts(self):
        # a part left on its tray is not inside the housing: no zero rule, but no cover credit either
        self.assertEqual(score(scene(motor=st(L.UNIT_START))).score, 0.2)
        self.assertEqual(score(scene(switch=st(L.SWITCH_START))).score, 0.3)

    def test_zero_when_closed_over_an_unseated_part(self):
        high_unit = st(O + [0, 0, 0.004])
        turned_unit = st(O + [0, 0, 0.006], quat_z(math.pi))
        high_switch = st(SEAT + [0, 0, 0.003])
        turned_switch = st(SEAT + [0, 0, 0.003], quat_z(math.pi))
        tilted_cover = st(O + [0, 0, 0.006], quat_axis((1, 0, 0), 0.08))
        for m, s in ((high_unit, None), (turned_unit, None), (None, high_switch), (None, turned_switch)):
            self.assertEqual(score(scene(motor=m, switch=s, cover=tilted_cover)).score, 0.0)
            self.assertEqual(score(scene(motor=m, switch=s, cover=st(O + [0, 0, 0.004]))).score, 0.0)

    def test_cover_on_but_not_closed(self):
        self.assertEqual(score(scene(cover=st(O + [0, 0, 0.003]))).score, 0.7)  # 3 mm up: riding on something
        self.assertEqual(score(scene(cover=st(O + [0.010, 0, 0.003]))).score, 0.7)  # 10 mm off, on the tongue
        self.assertEqual(score(scene(cover=st(O, quat_z(math.radians(10))))).score, 0.7)  # turned 10 degrees

    def test_thresholds_switch_exactly(self):
        self.assertEqual(score(scene(cover=st(O + [0, 0, 0.0014]))).score, 1.0)
        self.assertEqual(score(scene(cover=st(O + [0, 0, 0.0016]))).score, 0.7)
        self.assertEqual(score(scene(cover=st(O + [0.0039, 0, 0]))).score, 1.0)
        self.assertEqual(score(scene(cover=st(O + [0.0041, 0, 0]))).score, 0.7)
        self.assertEqual(score(scene(motor=st(O + [0.0039, 0, 0]))).score, 1.0)
        self.assertEqual(score(scene(motor=st(O + [0, 0.0021, 0]), cover=st(L.COVER_START))).score, 0.2)
        self.assertEqual(score(scene(motor=st(O + [0, 0, 0.0016]), cover=st(L.COVER_START))).score, 0.2)
        self.assertEqual(score(scene(switch=st(SEAT + [0.0029, 0, 0]))).score, 1.0)
        self.assertEqual(score(scene(switch=st(SEAT + [0.0031, 0, 0]), cover=st(L.COVER_START))).score, 0.3)
        self.assertEqual(score(scene(switch=st(SEAT, quat_z(math.radians(9))), cover=st(L.COVER_START))).score, 0.3)

    def test_motor_axis_rules(self):
        tilted = st(O, quat_axis((0, 1, 0), math.radians(4)))
        self.assertFalse(S.motor_in_seat(tilted))
        self.assertTrue(S.motor_in_seat(st(O, quat_axis((0, 1, 0), math.radians(2)))))
        self.assertFalse(S.motor_in_seat(st(O, quat_z(math.pi))))  # spindle toward the rear

    def test_rim_gaps_and_steps_measure_the_corners(self):
        c = st(O + [0.002, -0.001, 0.001])
        np.testing.assert_allclose(S.rim_gaps(c), 0.001, atol=1e-12)
        np.testing.assert_allclose(S.corner_steps(c), math.hypot(0.002, 0.001), atol=1e-12)
        tipped = st(O, quat_axis((1, 0, 0), 0.03))
        self.assertGreater(float(np.max(np.abs(S.rim_gaps(tipped)))), L.RIM_GAP)


class Held(unittest.TestCase):
    def test_a_held_part_earns_nothing(self):
        tip = O + np.array([0.0, L.R_UNIT + 0.002, 0.0015])
        arms = {"left": arm(tip, 0.4), "right": FAR["right"]}
        r = score(scene(cover=st(L.COVER_START)), arms)
        self.assertEqual(r.details["motor_held_by"], "left")
        self.assertEqual(r.score, 0.2)

    def test_open_gripper_holds_nothing(self):
        tip = O + np.array([0.0, L.R_UNIT + 0.002, 0.0015])
        arms = {"left": arm(tip, 0.95), "right": FAR["right"]}
        self.assertEqual(score(scene(), arms).score, 1.0)

    def test_held_cover_cannot_trigger_zero_or_score(self):
        cover_tip = np.array(L.world((0.020, 0.028 + 0.002, 0.003)))
        arms = {"left": arm(cover_tip, 0.5), "right": FAR["right"]}
        r = score(scene(motor=st(O + [0, 0, 0.004]), cover=st(O + [0, 0, 0.006])), arms)
        self.assertEqual(r.details["cover_held_by"], "left")
        self.assertEqual(r.score, 0.2)  # the switch alone; the high unit is not seated, the held cover counts nowhere

    def test_nearest_rule_gives_the_part_to_the_right_object(self):
        # a gripper holding the cover over the housing does not also hold the unit under it
        cover_tip = np.array(L.world((0.020, 0.028 + 0.001, 0.010)))
        arms = {"left": arm(cover_tip, 0.5), "right": FAR["right"]}
        held = S.holders(arms, scene())
        self.assertEqual(held["cover"], "left")
        self.assertIsNone(held["motor"])

    def test_unreadable_arms_fail_closed(self):
        tr = S.Tracker()
        settle(tr, scene())
        r = tr.update(scene(), None)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["arms_unavailable"], 1)
        bad = {"left": arm((math.nan, 0, 0)), "right": FAR["right"]}
        self.assertEqual(tr.update(scene(), bad).score, 0.0)


class Tracking(unittest.TestCase):
    def test_not_at_rest_until_the_window_fills(self):
        tr = S.Tracker()
        r = settle(tr, scene(), steps=5)
        self.assertEqual(r.score, 0.0)
        r = settle(tr, scene(), steps=6)
        self.assertEqual(r.score, 1.0)

    def test_moving_part_is_not_seated(self):
        tr = S.Tracker()
        settle(tr, scene())
        r = None
        for i in range(12):
            r = tr.update(scene(motor=st(O + [0.0005 * (i % 2), 0, 0])), FAR)  # jittering 0.5 mm a step
        self.assertFalse(r.details["motor_seated"])

    def test_blow_up_scores_zero_and_keeps_history(self):
        tr = S.Tracker()
        settle(tr, scene())
        bad = scene(motor=st((math.nan, 0, 0)))
        r = tr.update(bad, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(tr.update(scene(), FAR).score, 1.0)
        zero_q = scene(cover=st(O, (0, 0, 0, 0)))
        self.assertEqual(tr.update(zero_q, FAR).score, 0.0)

    def test_teleport_back_to_start_resets(self):
        tr = S.Tracker()
        settle(tr, scene())
        start = {"motor": st(L.UNIT_START), "switch": st(L.SWITCH_START), "cover": st(L.COVER_START)}
        tr.update(start, FAR)
        self.assertEqual(tr.steps, 1)

    def test_box_state_without_origin_is_understood(self):
        c = st(O)
        lo, hi = S.local_bounds("cover")
        box_state = SimpleNamespace(pos=O + (lo + hi) / 2, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                                    ang_vel=np.zeros(3), low=O + lo, high=O + hi)
        np.testing.assert_allclose(S.frame("cover", box_state)[0], S.frame("cover", c)[0], atol=1e-12)

    def test_every_outcome_maps_to_a_listed_score(self):
        listed = {0.0, 0.2, 0.3, 0.5, 0.7, 1.0}
        options_m = [st(O), st(L.UNIT_START), st(O + [0, 0, 0.004])]
        options_s = [st(SEAT), st(L.SWITCH_START), st(SEAT + [0, 0, 0.003])]
        options_c = [st(O), st(L.COVER_START), st(O + [0, 0, 0.004]), st(O + [0.3, 0, 0])]
        for m in options_m:
            for s in options_s:
                for c in options_c:
                    self.assertIn(score(scene(m, s, c)).score, listed)


class OfflineModel(unittest.TestCase):
    """offline_model.py: the parts' own shapes for the offline checker (never used in the simulator)."""

    def setUp(self):
        import offline_model
        self.m = offline_model.OfflineParts()

    def drop(self, name, origin):
        self.m.o[name] = np.array(origin, dtype=float)
        return self.m.rest_z(name) - O[2]

    def test_starts_are_at_rest(self):
        for n in L.LOOSE:
            self.assertAlmostEqual(self.m.rest_z(n), self.m.o[n][2], places=9)

    def test_parts_let_go_over_their_seats_seat(self):
        self.assertAlmostEqual(self.drop("motor", O + [0, 0, 0.0015]), 0.0, places=9)
        self.assertAlmostEqual(self.drop("switch", SEAT + [0, 0, 0.0015]) + O[2], SEAT[2], places=9)
        self.m.o["motor"], self.m.o["switch"] = O.copy(), SEAT.copy()
        self.assertAlmostEqual(self.drop("cover", O + [0, 0, 0.0005]), 0.0, places=9)

    def test_off_seat_drops_stay_off(self):
        self.assertGreater(self.drop("motor", O + [0.006, 0, 0.010]), 0.003)  # gearbox on the front stop
        self.assertGreater(self.drop("motor", O + [0, 0.003, 0.010]), 0.003)  # on a rib step
        self.m.o["motor"], self.m.o["switch"] = O.copy(), SEAT.copy()
        for d in ((0.0035, 0, 0), (-0.0035, 0, 0), (0, 0.0035, 0), (0, -0.0035, 0)):
            self.assertGreater(self.drop("cover", O + np.array(d) + [0, 0, 0.010]), L.RIM_GAP, d)  # on the tongue
        self.m.o["motor"] = O + [0, 0, 0.003]
        self.assertGreater(self.drop("cover", O + [0, 0, 0.010]), 0.001)  # on a raised unit's top

    def test_grasp_carry_release(self):
        tips = np.array(L.UNIT_START) + [-0.005, 0, 0.0015]
        far = arm((0.15, -0.20, L.TOP + 0.1))
        self.m.step({"left": arm(tips, 1.0), "right": far})
        self.m.step({"left": arm(tips, 0.25), "right": far})  # 23 mm < the can's 36 mm
        self.assertIn("left", self.m.holders["motor"])
        up = tips + [0, 0, 0.05]
        self.m.step({"left": arm(up, 0.25), "right": far})
        self.assertAlmostEqual(self.m.o["motor"][2], L.UNIT_START[2] + 0.05, places=9)
        self.m.step({"left": arm(up, 1.0), "right": far})
        self.assertFalse(self.m.holders["motor"])
        self.assertAlmostEqual(self.m.o["motor"][2], L.UNIT_START[2], places=9)  # back down on the rails


class Demo(unittest.TestCase):
    def test_demo_numbers(self):
        import importlib
        stub = HERE.parents[1] / "tools" / "offline" / "yam_stub"
        for p in (str(stub), str(stub.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            task = importlib.import_module("task")
        except ModuleNotFoundError as e:  # pragma: no cover - only without the offline stub
            self.skipTest(str(e))
        self.assertAlmostEqual(task.UNIT_COM_X, -0.0052, delta=0.0005)
        # pads (20 mm wide) at the unit's centre of mass stay clear of both cradle ribs
        for rib in ("rib_can_0", "rib_gearbox_0"):
            x0, x1 = L.LOWER[rib][0]
            self.assertGreater(min(abs(task.UNIT_COM_X - x0), abs(task.UNIT_COM_X - x1)), 0.010 + 0.005)
        # released pads clear the barrel walls: their tips stay above the rim
        self.assertGreater(task.UNIT_PLACE[2] - task.PAD_TIPS, L.TOP + L.PARTING + 0.002)
        self.assertGreater(task.SWITCH_PLACE[2] - task.PAD_TIPS, L.TOP + L.PARTING + 0.002)
        # cover grips: the right pads (20 mm wide, across the foot's 30 mm) sit on the foot behind the handle, inside
        # its rear end; the two grips stay far enough apart that the arms' wrists clear each other (capsule proxy)
        gx, gy = task.COVER_GRIP_AT["right"]
        self.assertLess(gx + 0.010, L.HANDLE_X[0] - 0.010)
        self.assertGreater(gx - 0.010, L.FOOT_X[0])
        self.assertTrue(L.FOOT_Y[0] < gy < L.FOOT_Y[1])
        lx, ly = task.COVER_GRIP_AT["left"]
        self.assertLess(lx + 0.010, L.BARREL_X[1] - L.WALL)
        self.assertGreater(math.hypot(lx - gx, ly - gy), 0.17)
        # GPU trims (job 20261119T002221): small, and the offline model undoes them exactly at the place points
        self.assertLess(math.hypot(*task.UNIT_TRIM), 0.004)
        self.assertLess(abs(math.degrees(task.UNIT_PLACE_YAW)), 2.0)
        for a, (tx, ty) in task.COVER_TRIM.items():
            self.assertLess(math.hypot(tx, ty), 0.0025)
            p = task._cover_point(a, task.COVER_DOWN_RIM, task.COVER_SHIFT, trim=True)
            self.assertEqual(task._gpu_arm_error(a, "cover", p), (-tx, -ty))
        self.assertEqual(task._gpu_arm_error("left", "motor", task.UNIT_PLACE), tuple(-v for v in task.UNIT_TRIM))
        self.assertEqual(task._gpu_arm_error("left", "motor", task.UNIT_PICK), (0.0, 0.0))
        self.assertEqual(task._gpu_arm_error("right", "motor", task.UNIT_PLACE), (0.0, 0.0))
        # the trimmed motor place, less the GPU arm error, is the nominal one over the unit's centre of mass
        self.assertAlmostEqual(task.UNIT_PLACE[0] - task.UNIT_TRIM[0], L.OX + task.UNIT_COM_X, places=9)
        self.assertAlmostEqual(task.UNIT_PLACE[1] - task.UNIT_TRIM[1], L.OY, places=9)
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        place = [w for w in plan["left"] if w.label in ("lower the last 2 cm slowly", "let go of the motor")]
        self.assertTrue(place and all(abs(w.yaw - task.UNIT_PLACE_YAW) < 1e-12 for w in place))
        span = {a: sum(w.duration for w in wps) for a, wps in plan.items()}
        self.assertAlmostEqual(span["left"], span["right"], places=6)  # the cover phase is on one clock
        self.assertLess(span["left"] * L.CONTROL_HZ, 2400)


if __name__ == "__main__":
    unittest.main()
