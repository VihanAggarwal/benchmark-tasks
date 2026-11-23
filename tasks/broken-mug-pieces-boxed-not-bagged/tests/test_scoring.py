"""Unit tests (unittest) for the scorer and the offline flap model. No simulator.

    cd tasks/broken-mug-pieces-boxed-not-bagged && python3 -m unittest discover -s tests -q
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

SHUT = math.radians(L.LID_OPEN_DEG)
TAB = 0.003


def _arm(tcp, open_frac=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=float(open_frac))


def _clear(open_frac=1.0):
    return {"left": _arm((0.24, 0.34, L.TOP + 0.16), open_frac), "right": _arm((0.14, -0.50, L.TOP + 0.15), open_frac)}


def _piece(name, x, y, z=0.0, yaw=0.0):
    return mk.state_from_poses(task.PIECE_SPECS[name], (x, y, L.TOP + z, *mk.yaw_quat(yaw)), np.zeros(6),
                               source="test")


def _box(lid=0.0, tab=0.0):
    return mk.state_from_poses(task.BOX_SPEC, (L.BOX_C[0], L.BOX_C[1], L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                               joints={L.LID_NAME: lid}, source="test")


def _start():
    return {n: _piece(n, *L.START[n][:2]) for n in S.PIECES}


def _boxed(names=S.PIECES):
    out = _start()
    for n in names:
        out[n] = _piece(n, L.BOX_C[0], L.DROP_Y[n], L.BOX_FLOOR)
    return out


def _st(pieces, box):
    return {**pieces, S.BOX: box}


class Geometry(unittest.TestCase):
    def test_pieces_light_and_graspable(self):
        for n in S.PIECES:
            self.assertLess(task.PIECE_SPECS[n].mass, 0.4)
        self.assertLess(2 * L.BASE_R + 0.008, rigstate.FULL_GAP)  # the base fits the open gripper

    def test_pieces_fit_the_box_without_touching(self):
        st = _boxed()
        lows = sorted((st[n].low[1], st[n].high[1]) for n in S.PIECES)
        for (a0, a1), (b0, b1) in zip(lows, lows[1:]):
            self.assertLess(a1, b0)
        for n in S.PIECES:
            self.assertTrue(S.inside(st[n].low - [0, 0, L.TOP], st[n].high - [0, 0, L.TOP], L.BOX_IN_X, L.BOX_IN_Y,
                                     L.RIM), n)

    def test_lid_open_as_drawn_and_shut_flat(self):
        lo, hi = _box().link_bounds(L.LID_NAME)
        self.assertGreater(hi[2] - L.TOP, L.RIM + 0.12)
        self.assertGreater(lo[0], L.BOX_X1 - 0.01)  # standing behind the far edge, out of the pieces' way
        lo, hi = _box(SHUT).link_bounds(L.LID_NAME)
        self.assertLess(hi[2] - L.TOP, L.RIM + 0.02)
        self.assertGreaterEqual(lo[0], L.BOX_X0 - 0.003)

    def test_reach_ceiling(self):
        for side, wps in task.plans().items():
            for w in wps:
                self.assertLessEqual(w.pos[2] - L.TOP, 0.225 + 1e-9, (side, w.label))


class Ladder(unittest.TestCase):
    def test_start_is_zero(self):
        self.assertEqual(S.Tracker().update(_st(_start(), _box()), _clear()).score, 0.0)

    def test_full_ladder(self):
        tr = S.Tracker()
        self.assertEqual(tr.update(_st(_boxed(["mug_base"]), _box()), _clear()).score, 0.25)
        self.assertEqual(tr.update(_st(_boxed(), _box()), _clear()).score, 0.5)
        self.assertEqual(tr.update(_st(_boxed(), _box(SHUT - math.radians(30))), _clear()).score, 0.5)
        busy = {"left": _clear()["left"], "right": _arm((0.30, -0.22, L.TOP + 0.12), 0.0)}
        self.assertEqual(tr.update(_st(_boxed(), _box(SHUT, TAB)), busy).score, 0.75)
        for _ in range(L.WIN_STEPS - 1):
            self.assertEqual(tr.update(_st(_boxed(), _box(SHUT, TAB)), _clear()).score, 0.75)
        r = tr.update(_st(_boxed(), _box(SHUT, TAB)), _clear())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_lid_must_be_shut_on_its_catch(self):
        tr = S.Tracker()
        self.assertEqual(tr.update(_st(_boxed(), _box(SHUT - math.radians(8))), _clear()).score, 0.5)  # ajar
        self.assertEqual(tr.update(_st(_boxed(), _box(SHUT - math.radians(4))), _clear()).score, 0.75)

    def test_three_of_four_is_a_quarter(self):
        r = S.Tracker().update(_st(_boxed(S.PIECES[:3]), _box(SHUT, TAB)), _clear())
        self.assertEqual(r.score, 0.25)


class Latches(unittest.TestCase):
    def test_a_piece_in_the_bag_is_zero_for_good(self):
        tr = S.Tracker()
        st = _start()
        st["wall_piece_a"] = _piece("wall_piece_a", L.CAN_C[0], L.CAN_C[1], L.CAN_WALL)
        r = tr.update(_st(st, _box()), _clear())
        self.assertEqual(r.score, 0.0)
        self.assertIn("trash bag", r.details["latched"])
        for _ in range(30):
            r = tr.update(_st(_boxed(), _box(SHUT, TAB)), _clear())
        self.assertEqual(r.score, 0.0)

    def test_a_piece_partly_in_the_bag_counts(self):
        st = _start()
        st["wall_piece_a"] = _piece("wall_piece_a", L.CAN_IN_X[0] - 0.02, L.CAN_C[1], L.CAN_RIM - 0.03)
        r = S.Tracker().update(_st(st, _box()), _clear(0.3))
        self.assertIn("trash bag", r.details["latched"])

    def test_carried_over_the_bag_is_not_in_it(self):
        st = _start()
        st["mug_handle"] = _piece("mug_handle", L.CAN_C[0], L.CAN_C[1], L.CAN_RIM + 0.01)
        r = S.Tracker().update(_st(st, _box()), _clear(0.3))
        self.assertFalse(r.details["latched"])

    def test_off_table(self):
        st = _start()
        st["mug_base"] = _piece("mug_base", 0.30, 0.0, -0.05)
        self.assertIn("off the table", S.Tracker().update(_st(st, _box()), _clear()).details["latched"])


class Exploits(unittest.TestCase):
    def test_piece_on_the_rim_is_not_boxed(self):
        st = _boxed()
        st["mug_base"] = _piece("mug_base", L.BOX_C[0], L.DROP_Y["mug_base"], L.RIM)
        r = S.Tracker().update(_st(st, _box(SHUT, TAB)), _clear())
        self.assertLess(r.score, 0.5)

    def test_hands_must_be_open_and_clear(self):
        tr = S.Tracker()
        for _ in range(30):
            r = tr.update(_st(_boxed(), _box(SHUT, TAB)), _clear(open_frac=0.3))
        self.assertEqual(r.score, 0.75)

    def test_fail_closed(self):
        self.assertEqual(S.Tracker().update({}, _clear()).score, 0.0)
        tr = S.Tracker()
        for _ in range(30):
            r = tr.update(_st(_boxed(), _box(SHUT, TAB)), None)
        self.assertEqual(r.score, 0.75)


class Offline(unittest.TestCase):
    def test_push_past_upright_falls_shut(self):
        sim = OfflineFlaps()
        for x in np.linspace(L.BOX_X1 + 0.032, L.BOX_X1 - 0.015, 25):
            sim.step({"left": _clear()["left"],
                      "right": _arm((float(x), L.BOX_C[1], L.TOP + L.LID_HINGE_Z + 0.035 + 0.013), 0.0)})
        for _ in range(25):
            sim.step(_clear())
        self.assertLessEqual(sim.theta[L.LID_NAME], L.LID_SHUT_DEG)

    def test_pad_coming_down_inside_does_not_close(self):
        sim = OfflineFlaps()
        for z in np.linspace(0.20, L.LID_HINGE_Z + 0.02, 25):
            sim.step({"left": _clear()["left"], "right": _arm((L.BOX_C[0], L.BOX_C[1], L.TOP + z), 0.0)})
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
