"""Unit and exploit tests for immersion-blender-unplugged-before-blade-wand-detached."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

for _p in TASK_DIR.parents:
    _off = _p / "tools" / "offline"
    if (_off / "yam_stub" / "yam_arena").is_dir():
        for _sub in (_off, _off / "yam_stub"):
            if str(_sub) not in sys.path:
                sys.path.insert(0, str(_sub))
        break

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_blender  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402


def _make_specs():
    motor = mk.assembly(S.MOTOR, origin=(L.BLENDER_X, L.SEAM_Y, L.TOP), shapes=P.motor_shapes(),
                        joints=[P.release_button_joint()], color=L.MOTOR_DARK)
    wand = mk.assembly(S.WAND, origin=tuple(L.WAND_SEATED_ORIGIN.tolist()), shapes=P.wand_shapes(),
                       mass=L.WAND_MASS, color=L.MOTOR_SILVER)
    bench = mk.assembly(S.BENCH, origin=(0.0, 0.0, L.TOP), shapes=P.bench_shapes(),
                        joints=[P.plug_joint()], color=L.TRAY_BLUE)
    return motor, wand, bench


def _quat_rz(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def _quat_rx(roll: float):
    return (math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2))


def _arm_snap(xyz, opening=0.64, yaw=0.0):
    p = np.asarray(xyz, dtype=float)
    tips = p + np.array([0.0, 0.0, -L.PAD_TIP])
    return rigstate.Arm(tcp=p, tips=tips, opening=float(opening))


def _clear_arms():
    return {
        "left": _arm_snap((0.18, 0.25, L.TOP + 0.16)),
        "right": _arm_snap((0.14, -0.30, L.TOP + 0.16)),
    }


def _make_parts(motor_spec, wand_spec, bench_spec, *, plug=0.0, button=0.0, wand_o=None, wand_q=(0.0, 0.0, 0.0, 1.0)):
    if wand_o is None:
        wand_o = L.WAND_SEATED_ORIGIN.copy()
    return {
        S.MOTOR: mk.state_from_poses(motor_spec, (L.BLENDER_X, L.SEAM_Y, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                     joints={"release": button}, source="test"),
        S.BENCH: mk.state_from_poses(bench_spec, (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                     joints={"plug": plug}, source="test"),
        S.WAND: mk.state_from_poses(wand_spec, (*np.asarray(wand_o, dtype=float).tolist(), *wand_q), source="test"),
    }


def _wand_in_tray_pose():
    # Oriented along X (yaw = -pi/2 so local -Y points along -X), centred at TRAY_C = (0.260, -0.240)
    origin = np.array([float(L.TRAY_C[0]) + L.WAND_LEN / 2, float(L.TRAY_C[1]), L.TOP + L.TRAY_FLOOR + L.BELL_R])
    return origin, _quat_rz(-math.pi / 2)


def _pull_out(tracker, motor, wand, bench, arms=None):
    """Feed the legal separation: unplugged, release held 5 mm with the wand seated, then the spigot pulled 25 mm
    straight out of the socket along -y with the release still held."""
    arms = _clear_arms() if arms is None else arms
    tracker.update(_make_parts(motor, wand, bench, plug=0.030, button=0.005), arms)
    return tracker.update(_make_parts(motor, wand, bench, plug=0.030, button=0.005,
                                      wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.025, 0.0])), arms)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.motor, self.wand, self.bench = _make_specs()
        self.tracker = S.Tracker()

    def test_rest_live_scores_zero(self):
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.0)
        res = self.tracker.update(parts, _clear_arms())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertIsNone(self.tracker.latched)

    def test_unplugged_only_scores_025(self):
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.025)
        res = self.tracker.update(parts, _clear_arms())
        self.assertEqual(res.score, 0.25)

    def test_axial_pull_out_with_release_held_scores_050(self):
        res = _pull_out(self.tracker, self.motor, self.wand, self.bench)
        self.assertEqual(res.score, 0.5)
        self.assertTrue(self.tracker.separated)
        # Latched: lifting the separated wand clear (off the socket axis) keeps 0.5
        res = self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030,
                                              wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.040, 0.060])),
                                  _clear_arms())
        self.assertEqual(res.score, 0.5)

    def test_partial_pull_short_of_engagement_scores_025(self):
        self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.005), _clear_arms())
        res = self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.005,
                                              wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.015, 0.0])),
                                  _clear_arms())
        self.assertEqual(res.score, 0.25)
        self.assertFalse(self.tracker.separated)

    def test_exploit_dropped_or_displaced_wand_gets_no_detach_credit(self):
        # Release held, then the wand found 40 mm away and 30 mm lower (fell / knocked off, not pulled along the axis)
        self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.005), _clear_arms())
        for _ in range(5):
            res = self.tracker.update(
                _make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.005,
                            wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.040, -0.030])), _clear_arms())
        self.assertEqual(res.score, 0.25)
        self.assertFalse(self.tracker.separated)
        # Even placed flat in the tray afterwards it never earns the tray tiers
        wo, wq = _wand_in_tray_pose()
        for _ in range(L.HOLD_STEPS + 2):
            res = self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq),
                                      _clear_arms())
        self.assertEqual(res.score, 0.25)

    def test_exploit_pull_without_release_held_gets_no_detach_credit(self):
        # Button never held >= 4 mm while the spigot was inside the 5 mm latch catch
        self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.0), _clear_arms())
        res = self.tracker.update(_make_parts(self.motor, self.wand, self.bench, plug=0.030, button=0.005,
                                              wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.025, 0.0])),
                                  _clear_arms())
        self.assertEqual(res.score, 0.25)
        self.assertFalse(self.tracker.separated)

    def test_withdrawal_geometry(self):
        parts = _make_parts(self.motor, self.wand, self.bench)
        self.assertAlmostEqual(S.withdrawal(parts[S.WAND]), 0.0, places=6)
        self.assertAlmostEqual(S.tip_off_axis(parts[S.WAND]), 0.0, places=6)
        parts = _make_parts(self.motor, self.wand, self.bench,
                            wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.018, 0.0]))
        self.assertAlmostEqual(S.withdrawal(parts[S.WAND]), L.SPIGOT_LEN, places=6)

    def test_spigot_fits_socket_with_play(self):
        self.assertGreater(L.SOCKET_APOTHEM, L.SPIGOT_R)
        self.assertGreater(L.SOCKET_DEPTH, L.SPIGOT_LEN)
        self.assertGreaterEqual(L.SPIGOT_LEN, 0.015)
        self.assertLessEqual(L.SPIGOT_LEN, 0.020)

    def test_in_tray_before_hold_scores_075_then_100_after_hold(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        arms = _clear_arms()
        for i in range(L.HOLD_STEPS - 1):
            res = self.tracker.update(parts, arms)
            self.assertEqual(res.score, 0.75, f"step {i + 1}")
            self.assertFalse(res.success)
        res = self.tracker.update(parts, arms)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_latched_zero_on_premature_button_press(self):
        # Press button 2.5 mm while plug is still seated (< 18 mm)
        parts_bad = _make_parts(self.motor, self.wand, self.bench, plug=0.010, button=0.0025)
        res = self.tracker.update(parts_bad, _clear_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(self.tracker.latched)
        # Even if later unplugged and placed cleanly in tray for 25 steps, score stays 0.0
        wo, wq = _wand_in_tray_pose()
        parts_ok = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        for _ in range(25):
            res = self.tracker.update(parts_ok, _clear_arms())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_latched_zero_on_premature_wand_motion(self):
        parts_bad = _make_parts(self.motor, self.wand, self.bench, plug=0.005,
                                wand_o=L.WAND_SEATED_ORIGIN + np.array([0.0, -0.006, 0.0]))
        res = self.tracker.update(parts_bad, _clear_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIn("wand moved", self.tracker.latched)

    def test_latched_zero_on_premature_gripper_near_blade_bell(self):
        parts_live = _make_parts(self.motor, self.wand, self.bench, plug=0.0)
        # Right hand right at the blade-bearing bell guard (y = SEAM_Y - 0.155 = -0.115) while live
        arms_near_bell = {
            "left": _arm_snap((0.18, 0.25, L.TOP + 0.16)),
            "right": _arm_snap((L.BLENDER_X, L.SEAM_Y - 0.155, L.TOP + L.AXIS_Z + L.PAD_TIP), opening=0.2),
        }
        res = self.tracker.update(parts_live, arms_near_bell)
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(self.tracker.latched)
        self.assertIn("blade bell", self.tracker.latched)

    def test_safe_shaft_contact_while_live_does_not_latch(self):
        parts_live = _make_parts(self.motor, self.wand, self.bench, plug=0.0)
        # Right hand touches the upper stainless shaft (y = SEAM_Y - 0.055 = -0.015), well clear of the bell guard
        arms_on_shaft = {
            "left": _arm_snap((0.18, 0.25, L.TOP + 0.16)),
            "right": _arm_snap((L.BLENDER_X, L.SEAM_Y - 0.055, L.TOP + L.AXIS_Z + L.PAD_TIP), opening=0.2),
        }
        res = self.tracker.update(parts_live, arms_on_shaft)
        self.assertEqual(res.score, 0.0)  # still live, so 0.0 now
        self.assertIsNone(self.tracker.latched)  # NOT permanently latched!
        # After unplugging and completing cleanly, still reaches 1.0
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        parts_ok = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        for _ in range(L.HOLD_STEPS):
            res = self.tracker.update(parts_ok, _clear_arms())
        self.assertEqual(res.score, 1.0)

    def test_latched_zero_on_wand_fell_off_bench(self):
        parts_fell = _make_parts(self.motor, self.wand, self.bench, plug=0.030,
                                 wand_o=np.array([0.26, -0.24, L.TOP - 0.05]))
        res = self.tracker.update(parts_fell, _clear_arms())
        self.assertEqual(res.score, 0.0)
        self.assertIn("fell off", self.tracker.latched)

    def test_exploit_wand_tilted_in_tray_rejected(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, _ = _wand_in_tray_pose()
        # Tilt 30 deg upward so axis_y has z component -sin(30 deg) = -0.5, bell end stays above table
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=_quat_rx(math.radians(-30)))
        res = self.tracker.update(parts, _clear_arms())
        self.assertEqual(res.score, 0.5)

    def test_exploit_wand_balanced_on_tray_rim_rejected(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        wo = wo + np.array([0.0, 0.0, 0.040])  # sitting on top of the 40 mm rim
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        res = self.tracker.update(parts, _clear_arms())
        self.assertEqual(res.score, 0.5)

    def test_exploit_wand_overhanging_tray_rejected(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        wo = wo + np.array([0.060, 0.0, 0.0])  # coupling end outside tray interior
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        res = self.tracker.update(parts, _clear_arms())
        self.assertEqual(res.score, 0.5)

    def test_exploit_hand_too_close_never_reaches_100(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        arms_close = {
            "left": _arm_snap((0.18, 0.25, L.TOP + 0.16)),
            "right": _arm_snap((float(L.TRAY_C[0]), float(L.TRAY_C[1]), L.TOP + 0.065), opening=0.5),
        }
        for _ in range(25):
            res = self.tracker.update(parts, arms_close)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)

    def test_exploit_moving_wand_never_reaches_100(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        arms = _clear_arms()
        for k in range(25):
            shift = np.array([0.008 * (k % 2), 0.0, 0.0])  # 8 mm oscillation > 5 mm STILL
            parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo + shift, wand_q=wq)
            res = self.tracker.update(parts, arms)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)

    def test_exploit_unreadable_arms_fail_closed(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        parts = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        for _ in range(25):
            res = self.tracker.update(parts, None)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)

    def test_exploit_replugged_drops_to_zero(self):
        _pull_out(self.tracker, self.motor, self.wand, self.bench)
        wo, wq = _wand_in_tray_pose()
        parts_ok = _make_parts(self.motor, self.wand, self.bench, plug=0.030, wand_o=wo, wand_q=wq)
        for _ in range(L.HOLD_STEPS):
            self.tracker.update(parts_ok, _clear_arms())
        self.assertEqual(self.tracker.last.score, 1.0)
        parts_replugged = _make_parts(self.motor, self.wand, self.bench, plug=0.010, wand_o=wo, wand_q=wq)
        res = self.tracker.update(parts_replugged, _clear_arms())
        self.assertEqual(res.score, 0.0)


class TestOfflineLatchRule(unittest.TestCase):
    def test_button_release_before_pull_reengages_bond(self):
        motor, wand, bench = _make_specs()
        model = offline_blender.BlenderModel(motor, wand, bench)
        self.assertTrue(model.bonded)
        # Press button with closed left hand
        arms_press = {
            "left": _arm_snap((L.BUTTON_X, L.BUTTON_Y, L.TOP + L.BUTTON_TOP_Z - 0.008 + L.PAD_TIP), opening=0.0),
            "right": _arm_snap((0.18, -0.25, L.TOP + 0.16), opening=0.64),
        }
        model.step(arms_press)
        self.assertGreaterEqual(model.button, L.BUTTON_RELEASE_AT)
        self.assertFalse(model.bonded)
        # Lift left hand off button without moving wand -> bond re-engages!
        model.step(_clear_arms())
        self.assertEqual(model.button, 0.0)
        self.assertTrue(model.bonded)

    def test_released_wand_stays_in_socket_until_pulled_along_axis(self):
        motor, wand, bench = _make_specs()
        model = offline_blender.BlenderModel(motor, wand, bench)
        arms_press = {
            "left": _arm_snap((L.BUTTON_X, L.BUTTON_Y, L.TOP + L.BUTTON_TOP_Z - 0.008 + L.PAD_TIP), opening=0.0),
            "right": _arm_snap((0.18, -0.25, L.TOP + 0.16), opening=0.64),
        }
        for _ in range(5):
            model.step(arms_press)
        self.assertFalse(model.bonded)
        np.testing.assert_allclose(model.wand_o, L.WAND_SEATED_ORIGIN)  # unheld: still seated in the socket


if __name__ == "__main__":
    unittest.main()
