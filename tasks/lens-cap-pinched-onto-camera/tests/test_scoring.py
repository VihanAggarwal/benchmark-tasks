"""Offline tests for layout.py, parts.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the
task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
CAPPED = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # the cap's front toward -y
Q_CAPPED = tuple(float(v) for v in mk.matrix_to_quat(CAPPED))
Q0 = (0.0, 0.0, 0.0, 1.0)


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def quat(r):
    return tuple(float(v) for v in mk.matrix_to_quat(r))


def cam_state(origin=None, rot=None):
    o = L.CAM_ORIGIN if origin is None else origin
    return mk.state_from_poses(T.CAMERA_PART, (*o, *(Q0 if rot is None else quat(rot))), source="test")


def cap_state(origin, rot=CAPPED, qp=0.0, qn=0.0):
    return mk.state_from_poses(T.CAP_PART, (*origin, *quat(rot)), joints={S.LATCH_P: qp, S.LATCH_N: qn},
                               source="test")


def capped_scene(p=0.0, dx=0.0, dz=0.0, qp=0.0, qn=0.0, cam_origin=None, cam_rot=None, cap_turn=None):
    """The cap on the lens (seat face ``p`` out from the lens's face, offset dx/dz across) of a camera at
    ``cam_origin`` turned ``cam_rot``; default: the 1.0 end state."""
    co = np.array(L.CAM_ORIGIN if cam_origin is None else cam_origin, dtype=float)
    cr = np.eye(3) if cam_rot is None else cam_rot
    if cam_origin is None:  # a turned camera resting on the table on its lowest corner
        co[2] += L.TOP - S.lowest_z(cam_state(co, cr))
    cap_rot = cr @ CAPPED if cap_turn is None else cr @ cap_turn @ CAPPED
    cap_o = co + cr @ np.array([dx, -p, dz])
    return {S.CAMERA: cam_state(co, cr), S.CAP: cap_state(cap_o, cap_rot, qp, qn)}


def start_scene():
    return {S.CAMERA: T.CAMERA_PART.rest_state(), S.CAP: T.CAP_PART.rest_state()}


def arm(tcp, opening=1.0, approach=(0.0, 0.0, -1.0)):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * np.asarray(approach, dtype=float), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def overlaps(cap, cam, shrink=5e-5, n=6) -> dict:
    """Sample points inside every cap shape (shrunk by 0.05 mm) that lie inside a camera shape, per cap body."""
    out = {}
    spec = cap.spec
    for body in spec.body_names:
        r, p = cap.link_pose(body)
        off = spec.body_offset(body)
        for s in spec.body_shapes(body):
            rw = r @ s.rotation()
            c = p + r @ (np.asarray(s.center, dtype=float) - off)
            if s.kind == "box":
                h = np.asarray(s.size, dtype=float) / 2 - shrink
                g = np.stack(np.meshgrid(*[np.linspace(-a, a, n) for a in h], indexing="ij"), -1).reshape(-1, 3)
            else:
                k = "XYZ".index(s.axis)
                pts = []
                for rad in (0.5 * s.radius, s.radius - shrink):
                    for a in np.linspace(-s.height / 2 + shrink, s.height / 2 - shrink, 3):
                        for t in np.linspace(0, 2 * math.pi, 48, endpoint=False):
                            v = [rad * math.cos(t), rad * math.sin(t)]
                            v.insert(k, a)
                            pts.append(v)
                g = np.array(pts)
            d = cam.distance(g @ rw.T + c)
            if (d <= 0).any():
                out[body] = out.get(body, 0) + int((d <= 0).sum())
    return out


def fitted(p, q, dx=0.0, dz=0.0) -> dict:
    s = capped_scene(p=p, dx=dx, dz=dz, qp=q, qn=q)
    return overlaps(s[S.CAP], s[S.CAMERA])


class Geometry(unittest.TestCase):
    def test_published_sizes(self):
        self.assertAlmostEqual(2 * L.LENS_R, 0.0685, places=6)
        self.assertAlmostEqual(L.LENS_LEN, 0.070, places=6)
        self.assertAlmostEqual(L.BODY_W, 0.129, places=6)
        self.assertAlmostEqual(L.CAM_AXIS_Z + L.HUMP_Z1, 0.1013, places=6)
        self.assertAlmostEqual(L.BODY_Y1 - L.GRIP_Y0, 0.0776, places=6)  # the published depth, grip included
        self.assertAlmostEqual(L.CAM_MASS, 0.675, places=6)

    def test_camera_stands_on_its_base_and_the_lens_clears_the_table(self):
        s = T.CAMERA_PART.rest_state()
        self.assertAlmostEqual(S.lowest_z(s), L.TOP, places=6)
        self.assertGreater(L.CAM_AXIS_Z - L.LENS_R, 0.008)

    def test_cap_lies_face_up_on_its_teeth(self):
        s = T.CAP_PART.rest_state()
        self.assertAlmostEqual(S.lowest_z(s), L.TOP, places=6)
        self.assertAlmostEqual(P.lowest(P.cap_base_shapes()), L.TEETH_Z, places=6)
        for side in (1, -1):
            self.assertAlmostEqual(P.lowest(P.latch_shapes(side)), L.TEETH_Z, places=6)

    def test_pinched_latches_clear_the_lip_and_free_ones_bite(self):
        self.assertGreater(L.TRAVEL - L.LATCH_CLEARS, 0.0008)  # 0.9 mm to spare when pinched
        self.assertGreaterEqual(L.LATCH_FLAT_R - L.LIP_R, 0.0014 - 1e-9)  # the catch bites 1.4 mm behind the lip
        self.assertLess(L.LATCH_CREST_R + L.LATCH_OUT, L.GROOVE_R)  # and a free latch never touches the groove
        # sliding across until the crest meets the groove leaves the far catch at least 0.8 mm of bite
        self.assertGreaterEqual(L.LATCH_FLAT_R - (L.GROOVE_R - L.LATCH_CREST_R) - L.LIP_R, 0.0008)

    def test_the_ramps_never_rise_above_the_catch_face(self):
        for side in (1, -1):
            for s in P.latch_shapes(side)[1:]:
                top = float(s.center[2]) + float(s.half_extents()[2])
                self.assertLessEqual(top, L.LATCH_CATCH_Z + 1e-9)

    def test_tab_pinch_is_within_the_gripper(self):
        self.assertLess(2 * L.TAB_X1, 0.094 - 0.03)  # 38 mm across the tabs
        self.assertLess(L.HUMP_W, 0.094 - 0.03)  # 58 mm across the hump
        self.assertLess(-2 * T.PINCH, 2 * (L.TAB_X1 - L.TRAVEL))  # the pinch closes past the tabs' stops

    def test_pinched_cap_slides_on_without_touching(self):
        for p in (0.008, 0.006, 0.004, 0.002, 0.001, 0.0):
            self.assertEqual(fitted(p, -L.TRAVEL), {}, p)

    def test_free_latches_cannot_be_out_until_the_cap_is_home(self):
        for p in (0.004, 0.002, 0.001, 0.0005):
            o = fitted(p, 0.0)
            self.assertIn(S.LATCH_P, o, p)
            self.assertIn(S.LATCH_N, o, p)
        self.assertEqual(fitted(0.0, 0.0), {})
        self.assertEqual(fitted(0.0002, 0.0), {})

    def test_the_latches_hold_the_cap_on(self):
        """Latched (seated, both out), the cap cannot come 0.5 mm off: both catches meet the lip's rear face."""
        o = fitted(0.0005, 0.0)
        self.assertTrue(o.get(S.LATCH_P) and o.get(S.LATCH_N))
        self.assertNotIn("base", o)

    def test_lugs_keep_the_cap_centred(self):
        self.assertIn("base", fitted(0.0, -L.TRAVEL, dz=-0.0015))
        self.assertGreater(L.HOME_LAT, L.LIP_R - L.LUG_FLAT_R)

    def test_capture_region_reaches_the_lip(self):
        self.assertGreater(L.LIP_R - (L.LATCH_LEAD_R - L.TRAVEL), 0.0045)
        self.assertGreater(L.LIP_R - L.LUG_LEAD_R, 0.003)

    def test_start_layout_is_clear(self):
        cam, cap = T.CAMERA_PART.rest_state(), T.CAP_PART.rest_state()
        self.assertGreater(float(cam.low[1]) - float(cap.high[1]), 0.020)  # 20 mm or more between them along y
        self.assertTrue(np.all(np.isfinite(cam.low)))
        for s in (cam, cap):
            self.assertLess(abs(float(s.pos[1])), 0.6)

    def test_start_scores_zero(self):
        res = run(start_scene())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["cap_home"])
        self.assertTrue(res.details["camera_down"])


class Tiers(unittest.TestCase):
    def test_full_score(self):
        res = run(capped_scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["cap_latched"])

    def test_full_score_within_the_tolerances(self):
        for kw in ({"p": 0.0009}, {"p": -0.0004}, {"dz": -0.0012}, {"dx": 0.0015}, {"qp": -0.0003},
                   {"cap_turn": rx(2.5)}, {"cap_turn": rz(2.5)}, {"cam_rot": rz(25.0)}, {"cam_rot": rx(8.0)}):
            self.assertEqual(run(capped_scene(**kw)).score, 1.0, kw)

    def test_the_cap_turned_about_its_own_axis_still_counts(self):
        twist = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about the cap's z
        s = capped_scene()
        s[S.CAP] = cap_state(np.array(L.CAM_ORIGIN), CAPPED @ twist)
        self.assertEqual(run(s).score, 1.0)

    def test_half_latched_camera_on_its_side(self):
        c = np.array([L.CAM_X, L.CAM_Y0, L.TOP + 0.0645])
        res = run(capped_scene(cam_origin=c, cam_rot=rx(0.0) @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])))
        self.assertEqual(res.score, 0.5)
        self.assertFalse(res.details["camera_down"])

    def test_half_latched_camera_held_up_off_the_table_still(self):
        c = np.array(L.CAM_ORIGIN) + np.array([0.0, 0.0, 0.020])
        self.assertEqual(run(capped_scene(cam_origin=c)).score, 0.5)

    def test_quarter_home_with_a_latch_held_in(self):
        for kw in ({"qp": -0.0010}, {"qn": -0.0025}, {"qp": -0.0025, "qn": -0.0025}):
            res = run(capped_scene(**kw))
            self.assertEqual(res.score, 0.25, kw)

    def test_zero_cap_not_home(self):
        for kw in ({"p": 0.0011}, {"p": 0.006}, {"p": -0.0006}, {"dx": 0.0021}, {"cap_turn": rx(3.5)}):
            self.assertEqual(run(capped_scene(**kw)).score, 0.0, kw)

    def test_zero_latch_held_in_and_camera_on_its_side(self):
        c = np.array([L.CAM_X, L.CAM_Y0, L.TOP + 0.0645])
        r = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        self.assertEqual(run(capped_scene(qp=-0.001, cam_origin=c, cam_rot=r)).score, 0.0)

    def test_latch_threshold_is_four_tenths_of_a_millimetre(self):
        self.assertEqual(run(capped_scene(qn=-0.00039)).score, 1.0)
        self.assertEqual(run(capped_scene(qn=-0.00041)).score, 0.25)

    def test_camera_tilt_threshold(self):
        self.assertEqual(run(capped_scene(cam_rot=rx(9.5))).score, 1.0)
        self.assertEqual(run(capped_scene(cam_rot=rx(10.5))).score, 0.5)

    def test_possible_scores(self):
        vals = {S.tier(a, b, c, d, e) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1) for e in (0, 1)}
        self.assertEqual(vals, {0.0, 0.25, 0.5, 1.0})

    def test_tier_function_table(self):
        self.assertEqual(S.tier(False, True, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, True, True, False), 0.5)
        self.assertEqual(S.tier(False, True, True, False, True), 0.25)
        self.assertEqual(S.tier(False, True, True, False, False), 0.0)
        self.assertEqual(S.tier(False, True, False, True, True), 0.0)
        self.assertEqual(S.tier(True, True, True, True, True), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True), 0.0)


class ZeroConditions(unittest.TestCase):
    def test_fall_latch_is_permanent(self):
        tr = S.Tracker()
        bad = capped_scene()
        bad[S.CAP] = cap_state((0.25, -0.70, L.TOP - 0.05), np.eye(3))  # the cap off the table's edge
        run(bad, steps=2, tracker=tr)
        self.assertTrue(tr.latched)
        res = run(capped_scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_the_camera_falling_latches(self):
        tr = S.Tracker()
        run(capped_scene(cam_origin=(0.34, 0.70, L.TOP - 0.02)), steps=1, tracker=tr)
        self.assertTrue(tr.latched)

    def test_a_drop_to_29_mm_below_is_not_a_fall(self):
        tr = S.Tracker()
        s = capped_scene()
        s[S.CAP] = cap_state((0.25, -0.13, L.TOP - L.TEETH_Z - 0.029), np.eye(3))
        run(s, steps=2, tracker=tr)
        self.assertFalse(tr.latched)

    def test_a_hand_on_the_cap_is_not_hands_off(self):
        near = dict(FAR)
        near["right"] = arm((L.CAM_X, L.CAM_Y0 - T.CAP_OFF, L.CAM_ORIGIN[2]), 0.29, (0.0, 1.0, 0.0))
        res = run(capped_scene(), arms=near)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_a_hand_on_the_camera_is_not_hands_off(self):
        near = dict(FAR)
        near["left"] = arm((L.CAM_X, T.Y_HUMP, L.TOP + T.Z_HUMP), 0.53)
        self.assertEqual(run(capped_scene(), arms=near).score, 0.0)

    def test_finger_zone_gap_20_mm_holds_40_mm_lets_go(self):
        c = capped_scene()[S.CAMERA]
        for gap, expect in ((0.020, 0.0), (0.040, 1.0)):
            x = float(c.high[0]) + gap + rigstate.zone_radius(1.0)
            a = dict(FAR)
            a["left"] = arm((x, L.CAM_Y0 + 0.09, L.TOP + 0.030 + 0.013))
            res = run(capped_scene(), arms=a)
            self.assertEqual(res.score, expect, (gap, res.details["hands_gap_mm"]))

    def test_one_step_of_contact_resets_the_window(self):
        tr = S.Tracker()
        near = dict(FAR)
        near["left"] = arm((L.CAM_X, T.Y_HUMP, L.TOP + T.Z_HUMP), 0.53)
        run(capped_scene(), tracker=tr)
        tr.update(capped_scene(), near)
        self.assertEqual(tr.update(capped_scene(), FAR).score, 0.0)
        self.assertEqual(run(capped_scene(), tracker=tr).score, 1.0)

    def test_unreadable_arms_never_released(self):
        self.assertEqual(run(capped_scene(), arms=None).score, 0.0)
        bad = dict(FAR)
        bad["left"] = arm((float("nan"), 0.45, L.TOP + 0.25))
        self.assertEqual(run(capped_scene(), arms=bad).score, 0.0)

    def test_still_means_visible_motion_only(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(capped_scene(cam_origin=np.array(L.CAM_ORIGIN) + np.array([0.0002 * i, 0.0, 0.0])), FAR)
        self.assertEqual(res.score, 1.0)  # 4.6 mm in the window

    def test_moving_camera_is_not_still(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(capped_scene(cam_origin=np.array(L.CAM_ORIGIN) + np.array([0.0004 * i, 0.0, 0.0])), FAR)
        self.assertEqual(res.score, 0.0)

    def test_turning_the_camera_5_degrees_is_visible(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            res = tr.update(capped_scene(cam_rot=rz(0.3 * i)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_needs_a_full_second(self):
        self.assertEqual(run(capped_scene(), steps=WIN).score, 0.0)
        self.assertEqual(run(capped_scene(), steps=WIN + 1).score, 1.0)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        s = capped_scene()
        self.assertEqual(run({S.CAMERA: s[S.CAMERA]}).score, 0.0)
        self.assertEqual(run(None).score, 0.0)
        self.assertEqual(run({}).score, 0.0)
        bad = capped_scene(cam_origin=(float("nan"), 0.0, L.TOP + 0.05))
        self.assertFalse(run(bad).details["valid_state"])

    def test_nan_latch(self):
        self.assertEqual(run(capped_scene(qp=float("nan"))).score, 0.0)

    def test_distance_failure_is_not_released(self):
        s = capped_scene()

        class NoDist:
            def __init__(self, st):
                self._s = st

            def __getattr__(self, k):
                if k == "distance":
                    raise RuntimeError("no distance")
                return getattr(self._s, k)

        s[S.CAP] = NoDist(s[S.CAP])
        self.assertEqual(run(s).score, 0.0)

    def test_reset_back_on_the_marks_clears_the_latch(self):
        tr = S.Tracker()
        bad = capped_scene()
        bad[S.CAP] = cap_state((0.25, -0.70, L.TOP - 0.05), np.eye(3))
        run(bad, steps=2, tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start_scene(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)


class Exploits(unittest.TestCase):
    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)

    def test_cap_held_on_by_a_hand_never_scores(self):
        near = dict(FAR)
        near["right"] = arm((L.CAM_X, L.CAM_Y0 - T.CAP_OFF, L.CAM_ORIGIN[2]), 0.29, (0.0, 1.0, 0.0))
        self.assertEqual(run(capped_scene(qp=-L.TRAVEL, qn=-L.TRAVEL), arms=near).score, 0.0)

    def test_cap_laid_against_the_lens_on_the_table_is_not_home(self):
        s = capped_scene(p=0.015)
        self.assertEqual(run(s).score, 0.0)

    def test_cap_on_the_wrong_way_round_is_not_home(self):
        flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # its front into the lens
        self.assertEqual(run(capped_scene(cap_turn=flip)).score, 0.0)

    def test_cap_on_the_camera_s_back_is_not_home(self):
        s = capped_scene()
        back = np.array(L.CAM_ORIGIN) + np.array([0.0, L.BODY_Y1 + L.EYECUP[1], 0.0])
        s[S.CAP] = cap_state(back, rz(180.0) @ CAPPED)
        self.assertEqual(run(s).score, 0.0)

    def test_camera_carried_off_with_the_cap_on_scores_where_it_stands(self):
        c = np.array([0.30, 0.10, L.TOP + L.CAM_AXIS_Z])
        self.assertEqual(run(capped_scene(cam_origin=c, cam_rot=rz(30.0))).score, 1.0)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertIn("left", d)
        self.assertIn("right", d)
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertIn("right", T.demo())
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

    def test_demo_length_is_within_half_the_run_limit(self):
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps, 600)
        self.assertGreater(steps, 300)

    @staticmethod
    def play(plans, extra_s=4.0) -> float:
        """Play the plans' TCP targets (no IK, no lag) through the offline model and the tracker; the last score."""
        from yam_arena.demo import ArmPlan

        model, tr = T.OfflineModel(), S.Tracker()
        bases = {"left": (0.0, 0.305, 0.75), "right": (0.0, -0.305, 0.75)}
        aps = {side: ArmPlan(plans[side], bases[side]) for side in plans}
        dur = max([a.duration for a in aps.values()] + [0.0]) + extra_s
        res = None
        for i in range(int(dur * HZ)):
            t = i / HZ
            arms = {}
            for side in ("left", "right"):
                if side in aps:
                    pos, rot, finger, _ = aps[side].target(t)
                    arms[side] = rigstate.Arm(pos, pos + 0.013 * rot[:, 2], float(np.clip(finger / T.OPEN, 0, 1)))
                else:
                    arms[side] = FAR[side]
            model.step(arms)
            res = tr.update(model.states(), arms)
        return res.score

    def test_demo_scores_one_offline(self):
        self.assertEqual(self.play(T.plans()), 1.0)

    def test_short_push_scores_zero_offline(self):
        self.assertEqual(self.play(T.plans("short_push")), 0.0)

    def test_one_arm_alone_scores_zero_offline(self):
        p = T.plans()
        self.assertEqual(self.play({"right": p["right"]}), 0.0)  # the cap pushed into the air above the camera
        self.assertEqual(self.play({"left": p["left"]}), 0.0)  # the cap left lying

    def test_hang_reproduces_the_first_gpu_run(self):
        # the first GPU run's plan (hand not pre-turned) with the measured hang: the lens's face where the trace had it
        # while the camera was held (336.2, -17.7, 96.0 mm), its axis about 4.4 degrees off -y (the cap's angle there)
        c, r = T.held_lens(0.0, T.HANG_TCP_1, T.HANG_DEG_1)
        seen = np.array([0.33616, -0.01766, L.TOP + 0.09595])
        self.assertLess(float(np.linalg.norm(c - seen)), 0.0005)
        self.assertAlmostEqual(S.angle_deg(-r[:, 1], np.array([0.0, -1.0, 0.0])), 4.4, delta=0.3)
        # where that plan put the cap: 26 mm above the hanging lens's axis, far outside the lugs' capture
        self.assertGreater(L.CAM_AXIS_Z + T.LIFT - 0.002 - (c[2] - L.TOP), 0.020)

    def test_pre_turned_hand_keeps_the_lens_level_and_the_push_ends_on_its_axis(self):
        self.assertLess(abs(math.degrees(math.asin(float(T.LENS_OUT[2])))), 0.5)
        self.assertLess(S.angle_deg(T.LENS_OUT, np.array([0.0, -1.0, 0.0])), 2.0)
        push = next(w for w in T.right_plan() if w.label.startswith("push"))
        seat = np.asarray(push.pos) + T.CAP_OFF * push.yaw.rotation[:, 2] - T.R_TRIM
        rel = T.LENS_ROT.T @ (seat - T.LENS_FACE)
        self.assertAlmostEqual(float(-rel[1]), -T.OVERSHOOT, places=6)
        self.assertLess(math.hypot(float(rel[0]), float(rel[2])), 1e-6)
        self.assertLess(S.angle_deg(push.yaw.rotation[:, 2], -T.LENS_OUT), 1e-6)

    def test_cap_missing_the_lens_stops_on_the_lip(self):
        m = T.OfflineModel()
        m.cam = np.array(L.CAM_ORIGIN, dtype=float)
        m.cap_rot = CAPPED.copy()
        m.cap = m.cam + np.array([0.006, -0.003, 0.0])  # 6 mm off across: outside the capture
        m._meet_lens()
        self.assertAlmostEqual(float(m.cam[1] - m.cap[1]), -L.TEETH_Z, places=6)


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
        text = d["setup"] + d["scoring"]
        for needle in ("0.675 kg", "58 mm", "2.5 mm", "2 N", "1.4 mm", "0.4 mm", "1 mm", "0.5 mm", "2 mm",
                       "3 degrees", "10 degrees", "5 mm", "30 mm", "1 s", "0.25", "0.5", "1.0"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "1200")

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
