"""Offline tests for layout.py, parts.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
    import parts as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
DT = 1.0 / HZ
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
C0 = 0.003  # a good press: every pad 3 mm in (3.6 N)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quat_x(a):
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def bvm(x=L.FACE_X, y=L.FACE_Y, z=None, pads=C0, squeeze=0.0, neck=0.0, tilt=0.0, valve=0.0):
    """The bag and mask: origin at (x, y, z) over the table (default: on the face with ``pads`` in), tilted
    ``tilt`` rad about x."""
    p = list(pads) if isinstance(pads, (list, tuple)) else [pads] * L.PAD_N
    if z is None:
        z = L.FACE_Z - float(np.mean(p))
    joints = {f"pad_{i}": p[i] for i in range(L.PAD_N)}
    joints.update(valve=valve, bag=neck, squeeze_a=squeeze / 2, squeeze_b=squeeze / 2)
    return mk.state_from_poses(T.BVM_PART, (x, y, L.TOP + z, *quat_x(tilt)), joints=joints, source="test")


def fixed():
    return {S.HEAD: T.HEAD_PART.rest_state(), S.BODY: T.BODY_PART.rest_state(), S.TRAY: T.TRAY_PART.rest_state()}


def scene(**kw):
    return {S.BVM: bvm(**kw), **fixed()}


def start_scene():
    return {S.BVM: T.BVM_PART.rest_state(), **fixed()}


def arm(tcp, opening=1.0, approach=(0.0, 0.0, -1.0)):
    tcp = np.asarray(tcp, dtype=float)
    a = np.asarray(approach, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * a / np.linalg.norm(a), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}
HOLD_VALVE = arm((L.FACE_X, L.FACE_Y, L.TOP + L.FACE_Z - C0 + L.VALVE_GRIP_Z), (L.VALVE_W) / L.FULL_GAP)
ON_BAG = arm((L.FACE_X, L.FACE_Y + L.PANEL_Y, L.TOP + L.FACE_Z - C0 + L.HOUSING_Z), 0.07 / L.FULL_GAP, T.A_R)
HOLDING = {"left": HOLD_VALVE, "right": ON_BAG}  # both hands on: the right on the bag
LET_GO = {"left": HOLD_VALVE, "right": FAR["right"]}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def breath_profile(hold=T.HOLD_S, carry=T.CARRY_SQUEEZE, peak=T.BREATH_SQUEEZE):
    """The squeeze (m) per step for one breath as the demo commands it: squeeze, hold, ease, wait."""
    out = []
    for dur, a, b in ((T.SQUEEZE_S, carry, peak), (hold, peak, peak), (T.EASE_S, peak, carry), (T.WAIT_S, carry, carry)):
        n = int(round(dur * HZ))
        for k in range(1, n + 1):
            s = k / n
            out.append(a + (b - a) * (3 * s * s - 2 * s ** 3))
    return out


def give_breaths(tr, n=5, hold=T.HOLD_S, arms=HOLDING, **kw):
    """Feed ``n`` breaths with the mask sealed on the face, the bag held at the carry squeeze before the seal is
    made (as the demo does)."""
    tr.update(scene(z=L.FACE_Z + 0.010, pads=0.0, squeeze=T.CARRY_SQUEEZE, **{k: v for k, v in kw.items()
                                                                              if k not in ("z", "pads")}), arms)
    for _ in range(int(0.5 * HZ)):
        tr.update(scene(squeeze=T.CARRY_SQUEEZE, **kw), arms)
    res = None
    for _ in range(n):
        for sq in breath_profile(hold):
            res = tr.update(scene(squeeze=sq, **kw), arms)
    return res


def let_go(tr, steps=WIN + 3, arms=LET_GO, **kw):
    res = None
    for _ in range(steps):
        res = tr.update(scene(squeeze=0.0, **kw), arms)
    return res


def full_trial(n=5, hold=T.HOLD_S, **kw):
    tr = S.Tracker()
    run(start_scene(), FAR, 3, tr)
    tr.update(scene(z=L.TRAY_Z + 0.050, pads=0.0, x=L.BVM_X0, squeeze=T.CARRY_SQUEEZE), HOLDING)  # lifted
    give_breaths(tr, n, hold, **kw)
    return tr, let_go(tr)


class Geometry(unittest.TestCase):
    def test_published_and_estimated_sizes(self):
        self.assertEqual((L.MASK_OD, L.MASK_MASS), (0.064, 0.037))  # Laerdal Newborn Mask size 1
        mask = L.MASK_ROOT_MASS + L.PAD_N * L.PAD_MASS
        self.assertAlmostEqual(mask, 0.037)
        self.assertAlmostEqual(2 * L.RIM_OUT, L.MASK_OD)
        self.assertAlmostEqual(L.PAD_RC + L.PAD_RADIAL / 2, 0.0315)
        self.assertAlmostEqual(L.BAG_Y[0] - L.BAG_Y[1], 0.150)
        vol = 4 / 3 * math.pi * 0.075 * L.BAG_R ** 2
        self.assertTrue(0.00022 <= vol <= 0.00028, vol)  # the 280 mL bag (published) as an ellipsoid, est
        self.assertAlmostEqual(L.PANEL_Y, -0.148)
        self.assertAlmostEqual(L.BVM_MASS, 0.147)
        self.assertAlmostEqual(L.K_CUSHION if hasattr(L, "K_CUSHION") else T.K_CUSHION, 1200.0)

    def test_the_bag_sits_1_mm_over_the_cushion(self):
        self.assertAlmostEqual(P.bag_low_z(), 0.001)
        s = T.BVM_PART.rest_state()
        self.assertAlmostEqual(float(s.low[2]), L.TOP + L.TRAY_Z, places=6)  # the pads on the tray

    def test_start_layout_on_the_tray(self):
        lo, hi = T.BVM_PART.rest_state().low, T.BVM_PART.rest_state().high
        self.assertTrue(L.TRAY_X[0] < lo[0] and hi[0] < L.TRAY_X[1])
        self.assertTrue(L.TRAY_Y[0] < lo[1] and hi[1] < L.TRAY_Y[1])
        head = T.HEAD_PART.rest_state()
        self.assertGreater(float(head.low[0]), L.TRAY_X[1])  # the tray is in front of the crown
        self.assertGreater(float(T.BODY_PART.rest_state().low[0]), float(head.high[0]))

    def test_slide_and_hinge_signs(self):
        up = bvm(pads=[0.005] + [0.0] * 7, z=0.05)
        rest = bvm(pads=0.0, z=0.05)
        self.assertGreater(float(up.link_pose("pad_0")[1][2]), float(rest.link_pose("pad_0")[1][2]) + 0.004)
        sq = bvm(squeeze=0.010, z=0.05)
        self.assertLess(float(sq.link_pose("squeeze_a")[1][0]), float(rest.link_pose("squeeze_a")[1][0]) - 0.004)
        self.assertGreater(float(sq.link_pose("squeeze_b")[1][0]), float(rest.link_pose("squeeze_b")[1][0]) + 0.004)
        droop = bvm(neck=0.5, z=0.05)
        self.assertLess(float(droop.low[2]), float(rest.low[2]) - 0.05)  # positive: the bag's far end down

    def test_the_face_seat_takes_the_whole_cushion_8_mm_off(self):
        self.assertGreaterEqual(L.FACE_R, L.RIM_OUT + L.SEAL_LAT)
        for dx in np.linspace(-L.FACE_R, L.FACE_R, 41):  # the skull stays under the seat's top
            self.assertLess(L.SKULL_AXIS_Z + P.skull_radius(L.FACE_X + dx), L.FACE_Z)

    def test_nose_and_mouth_fit_in_the_cavity(self):
        (nx0, nx1), nh, nz = L.NOSE
        far = max(math.hypot(nx0, nh), math.hypot(nx1, nh), math.hypot(L.MOUTH[0][1], L.MOUTH[1]))
        self.assertLess(far + L.SEAL_LAT, P.L.PAD_RC - L.PAD_RADIAL / 2)  # inside the pads, 8 mm off
        self.assertGreater(L.DOME_Z[0] - L.PAD_STOP, nz + 0.002)  # the dome over the nose at the pads' stop

    def test_the_eyes_are_clear_of_the_sealed_mask(self):
        for dx, dy in ((0, 0), (-L.SEAL_LAT, 0), (-0.0057, 0.0057), (-0.0057, -0.0057)):
            s = bvm(x=L.FACE_X + dx, y=L.FACE_Y + dy, pads=L.PAD_STOP)
            d = float(np.min(s.distance(S.EYES)))
            self.assertGreater(d, L.EYE_R + L.EYE_TOUCH + 0.002, (dx, dy, d))
        d = float(np.min(bvm(pads=C0).distance(S.EYES)))
        self.assertGreater(d, 0.012)
        eye = np.array(L.EYES[0])
        self.assertAlmostEqual(eye[2], L.TOP * 0 + L.eye_skull_z())
        self.assertLess(eye[2], L.FACE_Z)

    def test_spring_estimates(self):
        self.assertAlmostEqual(T.K_CUSHION, 1200.0)
        self.assertAlmostEqual(L.FINGER_SHARE, 5000.0 / 5400.0)
        self.assertLess(L.PANEL_K * L.PANEL_TRAVEL, 5.0)  # a full panel takes under 5 N: a tenth of a finger's cap
        self.assertLess(T.CHEST_SAG, 0.0003)

    def test_tray_body_head_disjoint(self):
        boxes = [(s.low, s.high) for s in (T.HEAD_PART.rest_state(), T.BODY_PART.rest_state(),
                                           T.TRAY_PART.rest_state())]
        for i in range(3):
            for j in range(i + 1, 3):
                (a0, a1), (b0, b1) = boxes[i], boxes[j]
                self.assertTrue(bool(np.any(a1 <= b0 + 1e-9) or np.any(b1 <= a0 + 1e-9)), (i, j))


class Seal(unittest.TestCase):
    def test_a_good_press_seals(self):
        r = S.read(scene())
        self.assertIsNone(S.seal_fault(r))
        self.assertAlmostEqual(r.force, 3.6)

    def test_each_fault(self):
        cases = {
            "off the target": dict(x=L.FACE_X - 0.0085),
            "tilted": dict(tilt=math.radians(6.5)),
            "not on the face": dict(z=L.FACE_Z + 0.001),
            "below the face": dict(z=L.FACE_Z - 0.011),
            "a pad not pressed": dict(pads=[0.0002] + [0.005] * 7),
            "a pad at its stop": dict(pads=[0.0096] + [0.001] * 7),
            "pressed too lightly": dict(pads=0.0016),
            "pressed too hard: airway shut": dict(pads=0.0085),
        }
        for why, kw in cases.items():
            self.assertEqual(S.seal_fault(S.read(scene(**kw))), why, kw)

    def test_band_edges(self):
        self.assertIsNone(S.seal_fault(S.read(scene(pads=2.0 / T.K_CUSHION))))
        self.assertIsNone(S.seal_fault(S.read(scene(pads=10.0 / T.K_CUSHION))))
        self.assertIsNone(S.seal_fault(S.read(scene(x=L.FACE_X + 0.0079))))
        self.assertIsNone(S.seal_fault(S.read(scene(tilt=math.radians(5.9)))))

    def test_kinked(self):
        self.assertFalse(S.read(scene(neck=0.39)).kinked)
        self.assertTrue(S.read(scene(neck=0.41)).kinked)
        self.assertTrue(S.read(scene(neck=-0.41)).kinked)

    def test_on_the_tray_is_not_sealed(self):
        self.assertIsNotNone(S.seal_fault(S.read(start_scene())))


class LungRule(unittest.TestCase):
    def test_a_10_mm_squeeze_gives_6_mm(self):
        lung = S.Lung()
        lung.update(True, 0.0065)
        lung.update(True, 0.0165)
        self.assertAlmostEqual(lung.rise, 0.006)
        for _ in range(40):
            lung.update(True, 0.0165)
        self.assertAlmostEqual(lung.rise, 0.006)  # held: the valve stays shut

    def test_easing_off_lets_the_chest_fall(self):
        lung = S.Lung()
        lung.update(True, 0.0)
        lung.update(True, 0.010)
        lung.update(True, 0.0085)
        self.assertEqual(lung.mode, "out")
        for _ in range(30):  # 1.5 s: five time constants
            lung.update(True, 0.0)
        self.assertLess(lung.rise, 0.0001)

    def test_no_seal_no_breath(self):
        lung = S.Lung()
        for sq in (0.0, 0.01, 0.02):
            lung.update(False, sq)
        self.assertEqual(lung.rise, 0.0)

    def test_a_squeeze_held_while_sealing_gives_nothing(self):
        lung = S.Lung()
        lung.update(False, 0.010)
        lung.update(True, 0.010)
        self.assertEqual(lung.rise, 0.0)

    def test_relief_valve_cap(self):
        lung = S.Lung()
        lung.update(True, 0.0)
        lung.update(True, 0.024)
        self.assertAlmostEqual(lung.rise, L.RISE_CAP)

    def test_a_leak_mid_breath_lets_the_chest_fall(self):
        lung = S.Lung()
        lung.update(True, 0.0)
        lung.update(True, 0.010)
        lung.update(False, 0.010)
        self.assertLess(lung.rise, 0.006)
        for _ in range(20):
            lung.update(False, 0.010)
        self.assertLess(lung.rise, 0.0001)

    def test_the_demo_breath_lasts_about_2_5_s(self):
        lung, b = S.Lung(), S.Breaths()
        step = 0
        for sq in [T.CARRY_SQUEEZE] * 10 + breath_profile():
            step += 1
            b.update(step, lung.update(True, sq))
        self.assertEqual((b.total, b.good), (1, 1))
        self.assertTrue(2.3 <= b.durations[0] <= 2.65, b.durations)


class BreathCount(unittest.TestCase):
    def feed(self, rises):
        b = S.Breaths()
        for i, r in enumerate(rises, 1):
            b.update(i, r)
        return b

    def test_lengths(self):
        good = self.feed([0.0] + [0.005] * 50 + [0.0])
        self.assertEqual((good.good, good.total), (1, 1))
        short = self.feed([0.0] + [0.005] * 39 + [0.0])
        self.assertEqual((short.good, short.total), (0, 1))
        edge = self.feed([0.0] + [0.005] * 60 + [0.0])
        self.assertEqual(edge.good, 1)
        self.assertIsNone(edge.too_long)
        long = self.feed([0.0] + [0.005] * 61 + [0.0])
        self.assertEqual(long.good, 0)
        self.assertIsNotNone(long.too_long)

    def test_dips_over_1_mm_join(self):
        b = self.feed([0.0] + [0.005] * 20 + [0.002] * 5 + [0.005] * 20 + [0.0])
        self.assertEqual((b.good, b.total), (1, 1))
        self.assertAlmostEqual(b.durations[0], 2.25)

    def test_a_dip_under_1_mm_splits(self):
        b = self.feed([0.0] + [0.005] * 45 + [0.0005] + [0.005] * 45 + [0.0])
        self.assertEqual((b.good, b.total), (2, 2))


class Tiers(unittest.TestCase):
    def test_full_score(self):
        tr, res = full_trial()
        self.assertEqual((tr.breaths.good, tr.breaths.total), (5, 5), res.details)
        self.assertEqual(res.score, 1.0, res.details)
        self.assertTrue(res.success)

    def test_the_bag_still_held_is_three_quarters(self):
        tr, _ = full_trial()
        res = let_go(tr, arms={"left": HOLD_VALVE, "right": arm(ON_BAG.tcp, 1.0, T.A_R)})
        self.assertEqual(res.score, 0.75)

    def test_breath_counts(self):
        for n, want in ((1, 0.25), (2, 0.25), (3, 0.5), (4, 0.5), (6, 0.75)):
            _, res = full_trial(n)
            self.assertEqual(res.score, want, n)

    def test_short_breaths_score_zero(self):
        _, res = full_trial(5, hold=1.2)
        self.assertEqual(res.score, 0.0)

    def test_a_breath_held_too_long_is_half(self):
        tr = S.Tracker()
        run(start_scene(), FAR, 3, tr)
        give_breaths(tr, 4)
        give_breaths(tr, 1, hold=3.0)  # 3.45 s
        res = let_go(tr)
        self.assertEqual((tr.breaths.good, tr.breaths.total), (4, 5))
        self.assertIsNotNone(tr.breaths.too_long)
        self.assertEqual(res.score, 0.5)

    def test_every_breath_too_long_is_zero(self):
        _, res = full_trial(5, hold=3.0)
        self.assertEqual(res.score, 0.0)

    def test_mask_let_go_at_the_end_is_not_at_rest(self):
        tr, _ = full_trial()
        res = let_go(tr, arms=FAR, pads=0.001)  # the mask under its own weight: too light to seal
        self.assertEqual(res.score, 0.75)

    def test_possible_scores_and_table(self):
        self.assertEqual(S.tier(True, 5, 5, False, False, False, True, False), 0.0)
        self.assertEqual(S.tier(False, 0, 3, False, False, False, True, False), 0.0)
        self.assertEqual(S.tier(False, 5, 5, True, False, False, True, False), 0.25)
        self.assertEqual(S.tier(False, 5, 5, False, True, False, True, False), 0.25)
        self.assertEqual(S.tier(False, 5, 5, False, False, True, True, False), 0.5)
        self.assertEqual(S.tier(False, 5, 5, False, False, False, True, False), 1.0)
        self.assertEqual(S.tier(False, 5, 5, False, False, False, True, True), 0.75)
        self.assertEqual(S.tier(False, 5, 6, False, False, False, True, False), 0.75)
        self.assertEqual(S.tier(False, 5, 5, False, False, False, False, False), 0.75)
        self.assertEqual(S.tier(False, 4, 5, False, False, False, True, False), 0.5)
        self.assertEqual(S.tier(False, 2, 2, False, False, False, True, False), 0.25)


class Latches(unittest.TestCase):
    def test_over_inflated(self):
        tr, res = full_trial(peak_hold=None) if False else (None, None)
        tr = S.Tracker()
        run(start_scene(), FAR, 2, tr)
        tr.update(scene(squeeze=0.0), HOLDING)
        for _ in range(45):
            tr.update(scene(squeeze=0.016), HOLDING)
        for _ in range(30):
            tr.update(scene(squeeze=0.0), HOLDING)
        self.assertIsNotNone(tr.latch["over"])
        self.assertEqual(tr.breaths.good, 1)
        self.assertEqual(tr.last.score, 0.25)

    def test_eye_touched_by_the_mask(self):
        tr, _ = full_trial()
        s = bvm(x=L.FACE_X - 0.030, z=L.FACE_Z, pads=0.0)
        tr.update({S.BVM: s, **fixed()}, LET_GO)
        self.assertIsNotNone(tr.latch["eyes"])
        self.assertEqual(let_go(tr).score, 0.0)

    def test_eye_touched_by_a_finger(self):
        tr, _ = full_trial()
        ex, ey, ez = S.EYES[0]
        res = tr.update(scene(), {"left": arm((ex, ey, ez + 0.013 + 0.020), 0.0), "right": FAR["right"]})
        self.assertEqual(tr.why.get("eyes"), "a gripper at an eye", res.details)
        self.assertEqual(let_go(tr).score, 0.0)

    def test_hands_beside_the_eye_are_fine(self):
        tr, res = full_trial()
        self.assertIsNone(tr.latch["eyes"], res.details)

    def test_dropped_on_the_table(self):
        tr, _ = full_trial()
        tr.update(scene(x=0.30, y=-0.20, z=0.0, pads=0.0), FAR)
        self.assertIsNotNone(tr.latch["dropped"])
        self.assertEqual(let_go(tr).score, 0.25)

    def test_nothing_drops_before_it_leaves_the_tray(self):
        tr = S.Tracker()
        run(start_scene(), FAR, 5, tr)
        self.assertIsNone(tr.latch["dropped"])

    def test_lifted_off_between_breaths(self):
        tr = S.Tracker()
        run(start_scene(), FAR, 2, tr)
        give_breaths(tr, 2)
        for _ in range(int(1.2 * HZ)):
            tr.update(scene(z=L.FACE_Z + 0.005, pads=0.0, squeeze=T.CARRY_SQUEEZE), HOLDING)
        self.assertIsNotNone(tr.latch["lifted"])
        give_breaths(tr, 3)
        res = let_go(tr)
        self.assertEqual(tr.breaths.good, 5)
        self.assertEqual(res.score, 0.75)

    def test_a_short_break_is_not_lifted(self):
        tr = S.Tracker()
        run(start_scene(), FAR, 2, tr)
        give_breaths(tr, 2)
        for _ in range(int(0.8 * HZ)):
            tr.update(scene(z=L.FACE_Z + 0.005, pads=0.0, squeeze=T.CARRY_SQUEEZE), HOLDING)
        self.assertIsNone(tr.latch["lifted"])


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        tr = S.Tracker()
        self.assertFalse(tr.update({}, FAR).details["valid_state"])
        self.assertFalse(tr.update(None, FAR).details["valid_state"])
        bad = bvm()
        object.__setattr__(bad, "origin", np.array([np.nan, 0.0, 0.0]))
        self.assertFalse(tr.update({S.BVM: bad}, FAR).details["valid_state"])
        self.assertEqual(tr.bad_steps, 3)

    def test_a_bad_step_restarts_the_window(self):
        tr, res = full_trial()
        self.assertEqual(res.score, 1.0)
        tr.update({}, FAR)
        self.assertFalse(tr.update(scene(squeeze=0.0), LET_GO).success)
        self.assertEqual(let_go(tr).score, 1.0)

    def test_unreadable_arms_never_hold_1(self):
        tr, _ = full_trial()
        for _ in range(WIN + 3):
            res = tr.update(scene(squeeze=0.0), None)
        self.assertEqual(res.score, 0.75)
        self.assertGreater(tr.unread, 0)

    def test_distance_failure_counts_as_not_released(self):
        class Broken:
            def __getitem__(self, k):
                raise RuntimeError("boom")
        self.assertFalse(S.Tracker()._released(Broken(), LET_GO))

    def test_reset_back_on_the_marks_clears_everything(self):
        tr, _ = full_trial()
        tr.update(scene(x=0.30, y=-0.20, z=0.0, pads=0.0), FAR)
        self.assertIsNotNone(tr.latch["dropped"])
        res = tr.update(start_scene(), FAR)
        self.assertEqual(tr.resets, 1)
        self.assertIsNone(tr.latch["dropped"])
        self.assertEqual(tr.breaths.good, 0)
        self.assertEqual(res.score, 0.0)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)

    def test_squeezing_on_the_tray_scores_zero(self):
        tr = S.Tracker()
        for _ in range(5):
            for sq in breath_profile():
                tr.update({S.BVM: bvm(x=L.BVM_X0, z=L.TRAY_Z, pads=0.0, squeeze=sq), **fixed()}, FAR)
        self.assertEqual(tr.breaths.total, 0)
        self.assertEqual(run(start_scene(), tracker=tr).score, 0.0)

    def test_a_light_or_crushing_hold_scores_zero(self):
        for pads in (0.001, 0.009):
            _, res = full_trial(pads=pads)
            self.assertEqual(res.score, 0.0, pads)

    def test_a_kinked_neck_scores_zero(self):
        _, res = full_trial(neck=0.45)
        self.assertEqual(res.score, 0.0)

    def test_tilted_or_off_target_scores_zero(self):
        for kw in (dict(tilt=math.radians(8)), dict(x=L.FACE_X + 0.010)):
            _, res = full_trial(**kw)
            self.assertEqual(res.score, 0.0, kw)

    def test_squeezes_too_small_give_no_breath(self):
        lung, b = S.Lung(), S.Breaths()
        step = 0
        for _ in range(5):
            for sq in breath_profile(peak=T.CARRY_SQUEEZE + 0.006):  # 6 mm more: 3.6 mm of rise
                step += 1
                b.update(step, lung.update(True, sq))
        self.assertEqual(b.total, 0)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    @staticmethod
    def play(plans, extra_s=3.0):
        """Plays plans through the offline model at 20 Hz, as rigid arms (the commanded TCP, the commanded
        opening) and returns the tracker's last result and the model."""
        from yam_arena.demo import ArmPlan
        m, tr = T.OfflineModel(), S.Tracker()
        bases = {"left": (0.0, 0.305, 0.75), "right": (0.0, -0.305, 0.75)}
        aps = {a: ArmPlan(p, bases[a]) for a, p in plans.items()}
        total = max((T._duration(p) for p in plans.values()), default=0.0) + extra_s
        res = None
        for k in range(int(round(total * HZ))):
            t = (k + 1) * DT
            arms = {}
            for a in ("left", "right"):
                if a in aps:
                    pos, rot, finger, _ = aps[a].target(t)
                    arms[a] = rigstate.Arm(np.asarray(pos), np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2],
                                           float(finger) / rigstate.FINGER_OPEN)
                else:
                    arms[a] = FAR[a]
            m.step(arms)
            res = tr.update(m.states(), arms)
        return res, m, tr

    def test_the_demo_scores_1(self):
        res, m, tr = self.play(T.plans())
        self.assertEqual(res.score, 1.0, res.details)
        self.assertEqual(tr.breaths.good, 5)
        for d in tr.breaths.durations:
            self.assertTrue(2.3 <= d <= 2.7, tr.breaths.durations)
        self.assertTrue(2.5 <= res.details["max_sealed_force_n"] <= 4.5, res.details)

    def test_one_arm_scores_zero(self):
        for idle in ("left", "right"):
            p = T.plans()
            p.pop(idle)
            res, _, tr = self.play(p)
            self.assertEqual(res.score, 0.0, (idle, res.details))
            self.assertEqual(tr.breaths.total, 0, idle)

    def test_counterfactuals(self):
        want = {"four": 0.5, "short": 0.0, "light": 0.0, "keep": 0.75}
        for v, s in want.items():
            res, _, _ = self.play(T.plans(v))
            self.assertEqual(res.score, s, (v, res.details))

    def test_left_alone_kinks_the_neck(self):
        p = T.plans()
        p.pop("right")
        _, m, _ = self.play(p)
        self.assertEqual(m.neck, L.NECK_LIMIT)
        self.assertGreater(m.pad, 0.0003)

    def test_right_alone_tips_the_mask(self):
        p = T.plans()
        p.pop("left")
        res, m, _ = self.play(p, extra_s=0.0)
        self.assertNotEqual(res.details["seal_fault"], None)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertEqual(set(d), {"left", "right"})
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertEqual(set(T.demo()), {"left", "right"})
            finally:
                os.environ.pop("DEMO_VARIANT")
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for side, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = side
            try:
                d = T.demo()
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
            self.assertNotIn(side, d)
            self.assertIn(other, d)

    def test_both_hands_move_together(self):
        p = T.plans()
        for label_l, label_r in (("pinch the elbow", "hold the bag"), ("lift the mask", "lift it"),
                                 ("over the face", "carry it beside"), ("down onto the face", "down with the mask"),
                                 ("press the mask on", "level at the neck")):
            self.assertAlmostEqual(T._until(p["left"], label_l), T._until(p["right"], label_r))
        dz_l = p["left"][5].pos[2] - p["left"][4].pos[2]
        dz_r = p["right"][5].pos[2] - p["right"][4].pos[2]
        self.assertAlmostEqual(dz_l, dz_r)
        self.assertGreaterEqual(T._duration(p["left"]), T._duration(p["right"]) + 1.0 - 1e-9)

    def test_finger_commands(self):
        self.assertAlmostEqual(T.VALVE_GRIP, -0.0055)
        for f, sq in ((T.BAG_CARRY, T.CARRY_SQUEEZE), (T.BAG_BREATH, T.BREATH_SQUEEZE)):
            gap = -2 * f
            self.assertAlmostEqual((L.BAG_ACROSS - gap) * L.FINGER_SHARE, sq)
        self.assertGreater(T.BAG_CARRY, T.OPEN)

    def test_demo_length_is_well_within_the_run_limit(self):
        limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0]) if (
            HERE / "task.json").exists() else 1300
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps + 3 * HZ, 0.65 * limit)  # the plan, plus 3 s of standing still
        self.assertGreater(steps, 600)


# -- the arms' give (GPU runs of other tasks: about 150 to 250 N/m at the TCP; a joint-stiffness model of this
# pose gives about 400 N/m vertical). Quasi-static, SI units.
K_GRID = (100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0)
W_BAG = (L.BAG_MASS + 2 * L.PANEL_MASS) * L.G


def press_force(k, over=T.PRESS_OVER, extra=0.0):
    """The cushion's force with the left arm (k N/m) commanded ``over`` under contact, carrying the mask and the
    valve (and ``extra`` N more: the bag hanging off after it is let go)."""
    c = (k * over + T.W_MASK + extra) / (k + T.K_CUSHION)
    return T.K_CUSHION * min(c, L.PAD_TRAVEL), c


def one_arm_max_force(mu, y_valve=-0.025):
    """The most cushion force one hand on the bag can get with the neck free (|angle| under its stops), every pad
    pressed and the mask's weight helping: moments about the neck's hinge. The resultant of 8 pads all pressed lies
    within half the pads' radius of the axis; friction at the face (mu) helps at most."""
    hinge_y, hinge_z = L.NECK_Y, L.HOUSING_Z
    w_mask, w_valve = L.MASK_MASS * L.G, L.VALVE_MASS * L.G
    weight_moment = (0.0 - hinge_y) * w_mask + (y_valve - hinge_y) * w_valve
    arm = (-L.PAD_RC / 2) - hinge_y  # the resultant at its nearest to the hinge
    per_n = arm - hinge_z * mu
    return weight_moment / per_n if per_n > 0 else math.inf


def squeeze_balance(gap, offset, k_bag):
    """Both panels' travel with the fingers commanded ``gap`` apart, centred ``offset`` off the bag's axis (x),
    the bag held in x by ``k_bag`` N/m (the left's arm through the valve)."""
    xa, xb = offset + gap / 2, offset - gap / 2
    share = L.FINGER_K / (L.FINGER_K + L.PANEL_K)

    def travel(x):
        sa = min(L.PANEL_TRAVEL, max(0.0, share * (x + L.PANEL_OUT - xa)))
        sb = min(L.PANEL_TRAVEL, max(0.0, share * (xb - x + L.PANEL_OUT)))
        return sa, sb

    lo, hi = -0.03, 0.03
    for _ in range(100):
        x = (lo + hi) / 2
        sa, sb = travel(x)
        net = L.PANEL_K * (sb - sa) - k_bag * x  # the force on the bag along +x
        lo, hi = (x, hi) if net > 0 else (lo, x)
    x = (lo + hi) / 2
    return travel(x), x


class ArmGive(unittest.TestCase):
    def test_the_press_is_in_the_band_for_any_arm(self):
        for k in K_GRID:
            f, c = press_force(k)
            self.assertTrue(L.SEAL_F[0] + 0.3 <= f <= L.SEAL_F[1] - 1.0, (k, f))
            self.assertTrue(L.PAD_MIN + 0.001 <= c <= L.PAD_STOP - 0.002, (k, c))
            f2, _ = press_force(k, extra=W_BAG)  # the bag let go and hanging from the valve
            self.assertLess(f2, L.SEAL_F[1] - 0.8, k)

    def test_the_light_variant_does_not_seal(self):
        for k in K_GRID:
            f, _ = press_force(k, over=-0.003)
            self.assertLess(f, L.SEAL_F[0], k)

    def test_the_hanging_bag_barely_tilts_the_mask(self):
        """Let go, the bag hangs at its stop: 0.79 N about 137 mm out. The cushion (8 pads) and the wrist (10 N m
        per rad, est from the arm's joints) resist: well under 1 degree, every pad still pressed."""
        lever = -L.NECK_Y + 0.088 * math.cos(L.NECK_LIMIT)
        k_rot = L.PAD_K * L.PAD_RC ** 2 * L.PAD_N / 2 + 10.0
        tilt = W_BAG * lever / k_rot
        self.assertLess(math.degrees(tilt), 1.0)
        _, c = press_force(120.0, extra=W_BAG)
        self.assertGreater(c - L.PAD_RC * tilt, L.PAD_MIN + 0.001)

    def test_one_hand_on_the_bag_cannot_seal(self):
        for mu in (0.6, 0.7, 0.8, 0.9):
            self.assertLess(one_arm_max_force(mu), L.SEAL_F[0] - 0.4, mu)

    def test_the_squeeze_does_not_care_where_the_fingers_close(self):
        # the valve swivels freely on the mask, so sideways (x) only the left's arm through the mask holds the bag:
        # about 0 (the swivel) to 170 N/m; 1000 N/m, a stiff hold, with up to 4 mm off centre
        for k_bag, offsets in ((0.0, (-0.008, 0.0, 0.008)), (170.0, (-0.008, 0.0, 0.008)),
                               (1000.0, (-0.004, 0.0, 0.004))):
            for offset in offsets:
                for f, want in ((T.BAG_CARRY, T.CARRY_SQUEEZE), (T.BAG_BREATH, T.BREATH_SQUEEZE)):
                    (sa, sb), x = squeeze_balance(-2 * f, offset, k_bag)
                    self.assertAlmostEqual(sa + sb, want, delta=0.0002, msg=(k_bag, offset))
                    self.assertGreater(min(sa, sb), 0.0005, (k_bag, offset))

    def test_breaths_reach_4_mm_with_a_weaker_squeeze(self):
        """If the panels or fingers give 30% less squeeze than designed, the rise still passes 4 mm; 50% more and
        it stays under the 9 mm over-inflation latch."""
        extra = T.BREATH_SQUEEZE - T.CARRY_SQUEEZE
        self.assertGreater(L.LUNG_GAIN * extra * 0.7, L.RISE_OK)
        self.assertLess(L.LUNG_GAIN * extra * 1.45, L.RISE_OVER)

    def test_the_bag_holding_hand_cannot_kink_the_neck(self):
        """The right's height error that would kink the neck (0.40 rad at 88 mm) is 34 mm: arms sag a few mm."""
        self.assertGreater(T.NECK_TO_PANEL * math.sin(L.NECK_KINK), 0.030)
        for k in (120.0, 170.0, 250.0):
            sag = W_BAG / k
            self.assertLess(math.asin(sag / T.NECK_TO_PANEL), L.NECK_KINK / 4)


# -- true-geometry clearances (signed distances to the parts' boxes and cylinders) -------------------------------------
def _sdf(shape, pts):
    """Signed distance (m) from base-frame points to one box or cylinder shape (negative inside)."""
    q = (pts - np.asarray(shape.center, dtype=float)) @ shape.rotation()
    if shape.kind == "box":
        d = np.abs(q) - np.asarray(shape.size, dtype=float) / 2
    else:
        i = "XYZ".index(shape.axis)
        d = np.stack([np.linalg.norm(np.delete(q, i, axis=1), axis=1) - shape.radius,
                      np.abs(q[:, i]) - shape.height / 2], axis=1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)


def _min_sdf(shapes, pts):
    return float(min(np.min(_sdf(s, pts)) for s in shapes))


def finger_points(tcp, rot, gap, n=7):
    """Sample points over both finger pads (yam_arena: 27 mm wide, 6 mm thick, from 9 mm behind the TCP to 13 mm past
    it along the approach axis) for the pads ``gap`` apart. ``rot`` columns: closing axis, width axis, approach."""
    c, w, a = rot[:, 0], rot[:, 1], rot[:, 2]
    pts = []
    for side in (1, -1):
        for u in np.linspace(gap / 2, gap / 2 + 0.006, 3):
            for v in np.linspace(-0.0135, 0.0135, n):
                for t in np.linspace(-0.009, 0.013, n):
                    pts.append(tcp + side * u * c + v * w + t * a)
    return np.array(pts)


def housing_points(tcp, rot, n=7):
    """The gripper's carriages and housing (est): a 65 x 50 mm block from 57 mm behind the TCP back to 110 mm."""
    c, w, a = rot[:, 0], rot[:, 1], rot[:, 2]
    pts = []
    for u in np.linspace(-0.0325, 0.0325, n):
        for v in np.linspace(-0.025, 0.025, n):
            for t in np.linspace(-0.110, -0.057, n):
                pts.append(tcp + u * c + v * w + t * a)
    return np.array(pts)


BVM_SHAPES = [s for b in T.BVM_PART.body_names for s in T.BVM_PART.body_shapes(b)]
BAG_SHAPES = [s for b in ("bag", "squeeze_a", "squeeze_b") for s in T.BVM_PART.body_shapes(b)]
VALVE_SHAPES = list(T.BVM_PART.body_shapes("valve"))
MASK_SHAPES = list(T.BVM_PART.body_shapes("base")) + [s for i in range(L.PAD_N)
                                                         for s in T.BVM_PART.body_shapes(f"pad_{i}")]
R_LEFT = __import__("yam_arena.demo", fromlist=["grip_rotation"]).grip_rotation(0.0)
R_RIGHT = T.YAW_R.rotation


class Clearance(unittest.TestCase):
    """In the mask frame at rest (the bag level): where the hands' pads go."""

    def test_the_left_pads_go_down_beside_the_elbow(self):
        tcp = np.array([0.0, 0.0, L.VALVE_GRIP_Z])
        for z in np.linspace(L.VALVE_GRIP_Z + 0.057, L.VALVE_GRIP_Z, 12):
            pts = finger_points(np.array([0.0, 0.0, z]), R_LEFT, rigstate.FULL_GAP)
            self.assertGreater(_min_sdf(BVM_SHAPES, pts), 0.0015, z)
        closed = finger_points(tcp, R_LEFT, L.VALVE_W + 0.0002)
        self.assertGreater(_min_sdf(MASK_SHAPES + BAG_SHAPES, closed), 0.0015)
        housing = VALVE_SHAPES[1]
        self.assertGreater(float(np.min(_sdf(housing, closed))), 0.0015)
        self.assertGreater(float(np.min(_sdf(VALVE_SHAPES[0], closed))), -1e-6)

    def test_the_left_housing_clears_the_elbow_top(self):
        self.assertGreater(L.VALVE_GRIP_Z + 0.057 - L.ELBOW_TOP, 0.050)

    def test_the_right_pads_go_in_along_their_axis_beside_the_panels(self):
        at = np.array([0.0, L.PANEL_Y, L.HOUSING_Z])
        a = R_RIGHT[:, 2]
        for s in np.linspace(0.070, 0.0, 15):
            pts = finger_points(at - s * a, R_RIGHT, rigstate.FULL_GAP)
            self.assertGreater(_min_sdf(BVM_SHAPES, pts), 0.004, s)
            pts = finger_points(at - s * a + np.array([0.008, 0.0, 0.0]), R_RIGHT, rigstate.FULL_GAP)
            self.assertGreater(_min_sdf(BVM_SHAPES, pts), 0.0005, s)  # 8 mm off centre

    def test_the_right_pads_land_on_the_panels(self):
        at = np.array([0.0, L.PANEL_Y, L.HOUSING_Z])
        pts = finger_points(at, R_RIGHT, L.BAG_ACROSS)
        panels = [s for b in ("squeeze_a", "squeeze_b") for s in T.BVM_PART.body_shapes(b)]
        self.assertLess(_min_sdf(panels, pts), 0.0005)
        # every pad point is over a panel face: within the panel's length and height
        rel = pts - at
        self.assertLess(float(np.max(np.abs(rel[:, 1]))), L.PANEL_LEN / 2)
        self.assertLess(float(np.max(np.abs(rel[:, 2]))), L.PANEL_H / 2)

    def test_the_right_housing_clears_the_bag(self):
        at = np.array([0.0, L.PANEL_Y, L.HOUSING_Z])
        self.assertGreater(_min_sdf(BVM_SHAPES, housing_points(at, R_RIGHT)), 0.003)  # 3.7 mm (est housing)

    def test_the_right_pads_clear_the_tray(self):
        at = np.array([L.BVM_X0, L.BVM_Y0 + L.PANEL_Y, T.GRAB_R])
        pts = finger_points(at, R_RIGHT, rigstate.FULL_GAP)
        self.assertGreater(float(np.min(pts[:, 2])), L.TRAY_Z + 0.008)

    def test_the_two_hands_stay_apart(self):
        left = finger_points(np.array([0.0, 0.0, L.VALVE_GRIP_Z]), R_LEFT, rigstate.FULL_GAP)
        right = finger_points(np.array([0.0, L.PANEL_Y, L.HOUSING_Z]), R_RIGHT, rigstate.FULL_GAP)
        d = np.min(np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2))
        self.assertGreater(d, 0.050)

    def test_the_carry_clears_the_head(self):
        head = list(T.HEAD_PART.spec_shapes()) if hasattr(T.HEAD_PART, "spec_shapes") else P.head_shapes()
        for x in np.linspace(L.BVM_X0, L.FACE_X, 30):
            for b in T.BVM_PART.body_names:
                for s in T.BVM_PART.body_shapes(b):
                    pass
            st = bvm(x=x, z=T.LIFT_Z, pads=0.0)
            pts = _bvm_surface(st)
            self.assertGreater(_min_sdf(head, pts), 0.012, x)

    def test_the_way_down_misses_the_eyes(self):
        for z in np.linspace(T.LIFT_Z, L.FACE_Z, 12):
            st = bvm(z=z, pads=0.0)
            self.assertGreater(float(np.min(st.distance(S.EYES))), L.EYE_R + 0.008, z)


def _bvm_surface(state, n=9):
    """Corner and face points of every BVM shape, in the world (table top at 0)."""
    pts = []
    spec = state.spec
    for body in spec.body_names:
        r, p = state.link_pose(body)
        off = spec.body_offset(body)
        for s in spec.body_shapes(body):
            h = s.half_extents()
            c = p + r @ (np.asarray(s.center, dtype=float) - off)
            for u in np.linspace(-1, 1, 3):
                for v in np.linspace(-1, 1, 3):
                    for w in np.linspace(-1, 1, 3):
                        pts.append(c + r @ (np.array([u, v, w]) * h))
    return np.array(pts) - np.array([0.0, 0.0, L.TOP])


class Folder(unittest.TestCase):
    def test_text_has_no_em_dashes_or_identifying_names(self):
        bad = re.compile(chr(0x2014) + "|" + chr(0x2013) + "|T-70|VihanAggarwal|Claude|Anthropic|co-author", re.I)
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "parts.py"):
            p = HERE / name
            if p.exists():
                self.assertIsNone(bad.search(p.read_text()), name)

    def test_task_json_matches_the_code(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        d = json.loads(p.read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["category"], "Other")
        text = d["setup"] + d["scoring"] + d["fidelity"]
        for needle in ("64 mm", "37 g", "280 mL", "x 250 mm", "90 mm", "8 mm", "6 degrees", "0.3", "2 to 10 N",
                       "0.40 rad", "0.6 mm", "1 mm", "0.3 s", "0.2 s", "12 mm", "4 mm", "2.0 to 3.0 s", "9 mm",
                       "3.0 s", "10 mm", "2 mm", "7.5 mm", "5 mm", "1.0 s", "0.25", "0.5", "0.75", "1.0",
                       "150 N/m", "400 N/m", "+-0.5 rad"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "1300")

    def test_numbers_in_the_text_match_layout(self):
        self.assertEqual((L.FACE_X, L.FACE_Y, L.FACE_Z), (0.250, 0.0, 0.090))
        self.assertEqual((L.SEAL_LAT, L.SEAL_TILT_DEG, L.PAD_MIN, L.SEAL_F, L.NECK_KINK),
                         (0.008, 6.0, 0.0003, (2.0, 10.0), 0.40))
        self.assertEqual((L.LUNG_GAIN, L.EASE, L.TAU_OUT, L.TAU_LEAK, L.RISE_CAP), (0.6, 0.001, 0.3, 0.2, 0.012))
        self.assertEqual((L.RISE_OK, L.RISE_LOW, L.RISE_OVER, L.BREATH), (0.004, 0.001, 0.009, (2.0, 3.0)))
        self.assertEqual((L.DROP_LIFT, L.DROP_TABLE, L.EYE_R, L.EYE_TOUCH, L.EYE_HAND, L.BAG_LET_GO, L.LIFT_OFF_S),
                         (0.010, 0.002, 0.0075, 0.001, 0.002, 0.005, 1.0))
        self.assertEqual((L.PAD_K, L.PANEL_K, L.NECK_LIMIT), (150.0, 400.0, 0.5))
        self.assertEqual((L.BVM_X0, L.TRAY_X), (0.110, (0.062, 0.158)))

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)

    def test_parts_build(self):
        for fn in (P.head_shapes, P.body_shapes, P.chest_shapes, P.tray_shapes, P.mask_shapes, P.valve_shapes,
                   P.bag_shapes):
            self.assertTrue(fn())
        self.assertEqual(len(P.pad_shapes(0)), 1)
        self.assertEqual(len(T.BVM_PART.joints), L.PAD_N + 4)


if __name__ == "__main__":
    unittest.main()
