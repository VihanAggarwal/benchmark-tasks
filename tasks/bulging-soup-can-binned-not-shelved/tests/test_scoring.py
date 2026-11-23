"""Unit tests (unittest) for the scorer and the offline lid model. No simulator.

    cd tasks/bulging-soup-can-binned-not-shelved && python3 -m unittest discover -s tests -q
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(TASK_DIR))
_YAM_STUB = REPO_ROOT / "tools" / "offline" / "yam_stub"
if (_YAM_STUB / "yam_arena").is_dir() and str(_YAM_STUB) not in sys.path:
    sys.path.insert(0, str(_YAM_STUB))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task  # noqa: E402
from offline_sim import OfflineFlaps  # noqa: E402

LID_SHUT = math.radians(L.LID_OPEN_DEG)


def _arm(tcp, open_frac=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=float(open_frac))


def _clear(open_frac=1.0):
    return {"left": _arm((0.20, 0.34, L.TOP + 0.17), open_frac), "right": _arm((0.12, -0.44, L.TOP + 0.16), open_frac)}


def _can(name, x, y, z_bottom, tilt=0.0):
    q = mk.yaw_quat(0.0) if tilt == 0.0 else np.array([math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)])
    return mk.state_from_poses(task.CAN_SPECS[name], (x, y, L.TOP + z_bottom + L.CAN_H / 2, *q), np.zeros(6),
                               source="test")


def _bin(lid=0.0):
    return mk.state_from_poses(task.BIN_SPEC, (L.BIN_C[0], L.BIN_C[1], L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                               joints={L.LID_NAME: lid}, source="test")


def _start(name):
    x, y = L.CAN_XY0[name]
    return _can(name, x, y, 0.0)


def _shelf(name, spot=None):
    x, y = L.SHELF_SPOTS[spot or name]
    return _can(name, x, y, L.SHELF_TOP)


def _binned(name=L.BULGED):
    return _can(name, L.BIN_C[0], L.BIN_C[1], L.BIN_FLOOR)


def _st(cans=None, lid=0.0):
    out = {n: _start(n) for n in L.CANS}
    out.update(cans or {})
    out[S.BIN] = _bin(lid)
    return out


GOAL = {"can_a": _shelf("can_a"), "can_b": _shelf("can_b"), L.BULGED: _binned()}


class Geometry(unittest.TestCase):
    def test_lid_open_as_drawn_and_shut_flat(self):
        lo, hi = _bin().link_bounds(L.LID_NAME)
        self.assertGreater(hi[2] - L.TOP, L.BIN_RIM + 0.15)
        lo, hi = _bin(LID_SHUT).link_bounds(L.LID_NAME)
        self.assertLess(hi[2] - L.TOP, L.BIN_RIM + 0.012)
        self.assertGreaterEqual(lo[0], L.BIN_X0 - 0.003)

    def test_bulged_can_is_visibly_taller(self):
        a = _start("can_a")
        b = _start(L.BULGED)
        self.assertAlmostEqual((b.high[2] - b.low[2]) - (a.high[2] - a.low[2]), L.DOME_H, delta=0.0006)

    def test_cans_fit(self):
        self.assertLessEqual(L.CAN_MASS, 0.30)
        self.assertLess(L.CAN_H + L.DOME_H + L.BIN_FLOOR, L.BIN_RIM)
        for name in L.CANS:  # 8 cm apart: 12 mm between cans
            for other in L.CANS:
                if name < other:
                    d = math.dist(L.CAN_XY0[name], L.CAN_XY0[other])
                    self.assertGreater(d - 2 * L.CAN_R, 0.010)

    def test_shelf_and_bin_far_apart(self):
        self.assertGreater(L.SHELF_Y[0] - L.BIN_Y1, 0.30)

    def test_reach_ceiling(self):
        for side, wps in task.plans().items():
            for w in wps:
                self.assertLessEqual(w.pos[2] - L.TOP, 0.225 + 1e-9, (side, w.label))


class Ladder(unittest.TestCase):
    def test_start_is_zero(self):
        self.assertEqual(S.Tracker().update(_st(), _clear()).score, 0.0)

    def test_full_ladder(self):
        tr = S.Tracker()
        tr.update(_st(), _clear())
        self.assertEqual(tr.update(_st({L.BULGED: _binned()}), _clear()).score, 0.25)
        self.assertEqual(tr.update(_st({L.BULGED: _binned()}, LID_SHUT), _clear()).score, 0.5)
        self.assertEqual(tr.update(_st({L.BULGED: _binned(), "can_a": _shelf("can_a")}, LID_SHUT), _clear()).score,
                         0.5)
        busy = {"left": _arm((0.38, 0.205, L.TOP + 0.10), 0.4), "right": _clear()["right"]}
        self.assertEqual(tr.update(_st(GOAL, LID_SHUT), busy).score, 0.75)
        for _ in range(L.WIN_STEPS - 1):
            self.assertEqual(tr.update(_st(GOAL, LID_SHUT), _clear()).score, 0.75)
        r = tr.update(_st(GOAL, LID_SHUT), _clear())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_good_cans_shelved_lid_open(self):
        r = S.Tracker().update(_st(GOAL), _clear())
        self.assertEqual(r.score, 0.25)

    def test_good_cans_shelved_bulged_on_counter(self):
        r = S.Tracker().update(_st({"can_a": _shelf("can_a"), "can_b": _shelf("can_b")}, LID_SHUT), _clear())
        self.assertEqual(r.score, 0.0)

    def test_lying_can_is_not_shelved(self):
        x, y = L.SHELF_SPOTS["can_a"]
        lying = _can("can_a", x, y, L.SHELF_TOP, tilt=math.radians(30.0))
        r = S.Tracker().update(_st({**GOAL, "can_a": lying}, LID_SHUT), _clear())
        self.assertFalse(r.details["shelved"]["can_a"])
        self.assertEqual(r.score, 0.5)


class Latches(unittest.TestCase):
    def test_bulged_on_shelf_latches(self):
        tr = S.Tracker()
        r = tr.update(_st({L.BULGED: _shelf(L.BULGED, "can_a")}), _clear())
        self.assertIn("bulging", r.details["latched"])
        for _ in range(30):  # moved to the bin afterwards: still 0
            r = tr.update(_st(GOAL, LID_SHUT), _clear())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_good_can_on_shelf_does_not_latch(self):
        r = S.Tracker().update(_st({"can_a": _shelf("can_a")}), _clear())
        self.assertFalse(r.details["latched"])

    def test_off_table(self):
        r = S.Tracker().update(_st({"can_b": _can("can_b", 0.30, 0.0, -0.10)}), _clear())
        self.assertIn("off the table", r.details["latched"])


class Exploits(unittest.TestCase):
    def test_bin_everything_earns_nothing(self):
        dx = (L.BIN_IN_X[1] - L.BIN_IN_X[0]) / 2 - L.CAN_R - 0.003
        cans = {L.BULGED: _can(L.BULGED, L.BIN_C[0] - dx, L.BIN_C[1], L.BIN_FLOOR),
                "can_a": _can("can_a", L.BIN_C[0] + dx, L.BIN_C[1] + 0.05, L.BIN_FLOOR),
                "can_b": _can("can_b", L.BIN_C[0] + dx, L.BIN_C[1] - 0.05, L.BIN_FLOOR)}
        for lid in (0.0, LID_SHUT):
            tr = S.Tracker()
            for _ in range(L.WIN_STEPS + 2):
                r = tr.update(_st(cans, lid), _clear())
            self.assertTrue(all(r.details["binned"].values()), r.details)
            self.assertTrue(r.details["good_binned"])
            self.assertEqual(r.score, 0.0)

    def test_one_good_can_binned_earns_nothing(self):
        dx = (L.BIN_IN_X[1] - L.BIN_IN_X[0]) / 2 - L.CAN_R - 0.003
        cans = {L.BULGED: _can(L.BULGED, L.BIN_C[0] - dx, L.BIN_C[1], L.BIN_FLOOR),
                "can_a": _shelf("can_a"), "can_b": _can("can_b", L.BIN_C[0] + dx, L.BIN_C[1], L.BIN_FLOOR)}
        r = S.Tracker().update(_st(cans, LID_SHUT), _clear())
        self.assertTrue(r.details["binned"]["can_b"])
        self.assertEqual(r.score, 0.0)
        cans["can_b"] = _shelf("can_b")  # taken back out and shelved: credit returns
        r = S.Tracker().update(_st(cans, LID_SHUT), _clear())
        self.assertEqual(r.score, 0.75)

    def test_can_held_over_the_bin_is_not_binned(self):
        r = S.Tracker().update(_st({L.BULGED: _can(L.BULGED, L.BIN_C[0], L.BIN_C[1], L.BIN_RIM + 0.02)}, LID_SHUT),
                               _clear(0.4))
        self.assertFalse(r.details["binned"][L.BULGED])

    def test_hands_must_be_open_and_clear(self):
        tr = S.Tracker()
        for _ in range(30):
            r = tr.update(_st(GOAL, LID_SHUT), _clear(open_frac=0.3))
        self.assertEqual(r.score, 0.75)

    def test_fail_closed(self):
        self.assertEqual(S.Tracker().update({}, _clear()).score, 0.0)
        tr = S.Tracker()
        for _ in range(30):
            r = tr.update(_st(GOAL, LID_SHUT), None)
        self.assertEqual(r.score, 0.75)  # never 1.0 without readable arms


class Offline(unittest.TestCase):
    def _push(self, sim, x0, x1, z_tips, y=-0.30):
        for x in np.linspace(x0, x1, 25):
            sim.step({"left": _clear()["left"], "right": _arm((float(x), y, L.TOP + z_tips + 0.013), 0.0)})

    def test_push_past_upright_falls_shut(self):
        sim = OfflineFlaps()
        self._push(sim, L.BIN_X1 + 0.040, L.BIN_X1 - 0.045, L.LID_HINGE_Z + 0.035)
        for _ in range(25):
            sim.step(_clear())
        self.assertLessEqual(sim.theta[L.LID_NAME], L.LID_SHUT_DEG)

    def test_pad_coming_down_inside_does_not_close(self):
        sim = OfflineFlaps()
        for z in np.linspace(0.20, L.LID_HINGE_Z + 0.02, 25):
            sim.step({"left": _clear()["left"], "right": _arm((L.BIN_C[0], L.BIN_C[1], L.TOP + z), 0.0)})
        self.assertEqual(sim.theta[L.LID_NAME], L.LID_OPEN_DEG)

    def test_open_gripper_does_not_push(self):
        sim = OfflineFlaps()
        for x in np.linspace(L.BIN_X1 + 0.040, L.BIN_X1 - 0.045, 25):
            sim.step({"left": _clear()["left"],
                      "right": _arm((float(x), -0.30, L.TOP + L.LID_HINGE_Z + 0.048), 1.0)})
        self.assertEqual(sim.theta[L.LID_NAME], L.LID_OPEN_DEG)


class Kit(unittest.TestCase):
    def test_kit_files_byte_identical(self):
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual((TASK_DIR / name).read_bytes(), (REPO_ROOT / "tools" / "kit" / name).read_bytes())

    def test_variants_known(self):
        for v in ("",) + task.VARIANTS:
            self.assertEqual(set(task.plans(v)), {"left", "right"})


if __name__ == "__main__":
    unittest.main()
