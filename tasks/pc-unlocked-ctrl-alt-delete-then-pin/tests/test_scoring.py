"""Offline tests for layout.py, scoring.py and task.py: synthetic keyboard states and arm snapshots, no simulator.
    python3 -m unittest discover -s tests -v"""

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
for p in HERE.parents:
    d = p / "tools" / "offline"
    if (d / "yam_stub" / "yam_arena").is_dir():
        for q in (d, d / "yam_stub"):
            if str(q) not in sys.path:
                sys.path.insert(0, str(q))
        break

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT = HERE.parents[1] / "tools" / "kit"
FAR = {"left": rigstate.Arm(np.array([0.20, 0.27, L.TOP + 0.15]), np.array([0.20, 0.27, L.TOP + 0.137]), 1.0),
       "right": rigstate.Arm(np.array([0.20, -0.27, L.TOP + 0.15]), np.array([0.20, -0.27, L.TOP + 0.137]), 1.0)}


def kb(down=(), dx=0.0, dz=0.0, rot=None):
    j = {k: (0.003 if k in down else 0.0) for k in L.KEYS}
    q = (0.0, 0.0, 0.0, 1.0) if rot is None else rot
    return {S.KEYBOARD: mk.state_from_poses(T.KEYBOARD_PART, (L.KB_X + dx, L.KB_Y, L.TOP + dz, *q), joints=j,
                                            source="test")}


def run(seq, arms=FAR):
    tr = S.Tracker(HZ)
    res = None
    for st in seq:
        res = tr.update(st, arms)
    return tr, res


def press(seq_keys, base=(), n=3):
    """States: each key in seq_keys pressed for n steps then released, with base keys held throughout."""
    out = []
    for k in seq_keys:
        out += [kb(set(base) | {k})] * n + [kb(set(base))] * n
    return out


UNLOCK = ([kb()] * 2 + [kb({"lctrl", "lalt"})] * 12 + [kb({"lctrl", "lalt", "delete"})] * 3 + [kb({"lctrl", "lalt"})] * 2
          + [kb()] * 3)
PIN_SEQ = press(["np2", "np5", "np8", "np0"])
ENTER = press(["npenter"])
REST = [kb()] * (WIN + 3)


class LayoutTests(unittest.TestCase):
    def test_pitch_and_clearance(self):
        self.assertAlmostEqual(T.PAIR_PITCH, 2.5 * L.U, places=6)
        self.assertAlmostEqual(T.HOLD_GAP / 2, 2.5 * L.U / 2 - T.FINGER_T / 2, places=6)
        # the Windows cap clears the inner faces of the holding fingers
        win_half = L.key_half("lwin")[1]
        self.assertGreater(T.HOLD_GAP / 2 - win_half, 0.008)

    def test_delete_far_from_ctrl(self):
        self.assertGreater(L.dist(L.key_world("lctrl"), L.key_world("delete")), 0.25)
        self.assertGreater(L.dist(L.key_world("ralt"), L.key_world("delete")), 0.08)

    def test_heights_and_forces(self):
        self.assertLess(T.ABOVE, 0.225)
        self.assertGreater(T.PRESS_TCP - T.PAD_TIP, 0.001)
        self.assertLess(L.KEY_ACT, T.PRESS)
        self.assertLess(L.KEY_ACT, L.KEY_TRAVEL)
        self.assertGreaterEqual(T.PRESS, L.KEY_TRAVEL)  # bottoms out on the stop, as real typing does
        self.assertAlmostEqual(L.ACT_FORCE, 0.70, places=3)

    def test_kit_files_identical(self):
        for n in ("mechkit.py", "rigstate.py"):
            if (KIT / n).exists():
                self.assertEqual((HERE / n).read_bytes(), (KIT / n).read_bytes())


class LogicTests(unittest.TestCase):
    def test_untouched_is_zero(self):
        _, res = run([kb()] * (WIN + 5))
        self.assertEqual(res.score, 0.0)

    def test_full_unlock(self):
        _, res = run(UNLOCK + PIN_SEQ + ENTER + REST)
        self.assertEqual(res.details["screen"], S.DESKTOP)
        self.assertEqual(res.score, 1.0)

    def test_pair_only(self):
        _, res = run([kb({"lctrl", "lalt"})] * 12 + REST)
        self.assertEqual(res.score, 0.25)
        _, res = run([kb({"lctrl", "lalt"})] * 5 + REST)
        self.assertEqual(res.score, 0.0)

    def test_order_matters(self):
        seq = [kb({"delete"})] * 3 + [kb({"delete", "lctrl", "lalt"})] * 12 + REST
        _, res = run(seq)
        self.assertEqual(res.details["screen"], S.LOCK)
        self.assertEqual(res.score, 0.25)

    def test_same_step_is_not_enough(self):
        seq = [kb()] * 2 + [kb({"lctrl", "lalt", "delete"})] * 12 + REST
        _, res = run(seq)
        self.assertEqual(res.details["screen"], S.LOCK)

    def test_right_side_pair_counts(self):
        seq = [kb({"rctrl", "ralt"})] * 12 + [kb({"rctrl", "ralt", "delete"})] * 3 + [kb()] * 3 + PIN_SEQ + ENTER + REST
        _, res = run(seq)
        self.assertEqual(res.score, 1.0)

    def test_windows_key_is_stray(self):
        seq = [kb({"lctrl", "lwin", "lalt"})] * 12 + [kb({"lctrl", "lalt", "delete"})] * 3 + [kb()] + PIN_SEQ + ENTER + REST
        _, res = run(seq)
        self.assertEqual(res.score, 0.5)
        self.assertTrue(res.details["win"])

    def test_pair_released_as_delete_goes_down(self):
        for held in ({"lctrl"}, {"lalt"}, set()):
            seq = [kb({"lctrl", "lalt"})] * 12 + [kb(held | {"delete"})] * 3 + [kb()] * 3 + PIN_SEQ + ENTER + REST
            _, res = run(seq)
            self.assertEqual(res.details["screen"], S.LOCK, held)

    def test_stray_key_caps(self):
        seq = UNLOCK[:2] + press(["end"]) + UNLOCK[2:] + PIN_SEQ + ENTER + REST
        _, res = run(seq)
        self.assertEqual(res.score, 0.5)

    def test_wrong_pin(self):
        seq = UNLOCK + press(["np2", "np5", "np8", "np9"]) + ENTER + REST
        _, res = run(seq)
        self.assertEqual(res.score, 0.5)

    def test_pin_without_enter(self):
        _, res = run(UNLOCK + PIN_SEQ + REST)
        self.assertEqual(res.score, 0.75)

    def test_extra_digit_spoils_pin(self):
        _, res = run(UNLOCK + PIN_SEQ + press(["np1"]) + REST)
        self.assertEqual(res.score, 0.5)

    def test_hands_on_keyboard_not_full(self):
        on = dict(FAR)
        p = np.array([*L.key_world("np5"), L.TOP + 0.035])
        on["right"] = rigstate.Arm(p, p - np.array([0, 0, 0.013]), 0.0)
        _, res = run(UNLOCK + PIN_SEQ + ENTER + REST, arms=on)
        self.assertEqual(res.score, 0.75)

    def test_key_held_at_end_not_full(self):
        _, res = run(UNLOCK + PIN_SEQ + [kb({"npenter"})] * (WIN + 6))
        self.assertEqual(res.score, 0.75)

    def test_slide_harmless_lift_latches(self):
        def shoved(seq, dx):
            out = []
            for st in seq:
                down = {k for k in L.KEYS if st[S.KEYBOARD].joints[k] > 0}
                out.append(kb(down, dx=dx))
            return out
        ramp = [kb(dx=0.005 * i) for i in range(1, 7)]
        _, res = run(UNLOCK[:5] + ramp + shoved(UNLOCK[5:] + PIN_SEQ + ENTER + REST, 0.03))
        self.assertEqual(res.score, 1.0)
        _, res = run(UNLOCK[:5] + [kb(dz=0.008)] * 3 + UNLOCK[5:] + PIN_SEQ + ENTER + REST)
        self.assertEqual(res.score, 0.0)

    def test_bad_state(self):
        tr = S.Tracker(HZ)
        self.assertFalse(tr.update({}, FAR).details["valid_state"])

    def test_screen_targets(self):
        pc = S.PC()
        t = T.screen_targets(pc)
        self.assertGreater(t["show_lock"], 0)
        self.assertEqual(t["show_prompt"], 0)
        pc.screen, pc.typed = S.PROMPT, "25"
        t = T.screen_targets(pc)
        self.assertGreater(t["dot2"], 0)
        self.assertEqual(t["dot3"], 0)


class DemoTests(unittest.TestCase):
    def setUp(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)

    tearDown = setUp

    def _play(self, plan):
        """Both arms on one clock, linear between waypoints at 20 Hz, through the offline model and the tracker."""
        m = T.OfflineModel()
        tr = S.Tracker(HZ)
        tracks = {}
        for side, wps in plan.items():
            pos = np.array(FAR[side].tcp)
            fin = T.OPEN
            pts = []
            for w in wps:
                n = max(1, int(round(w.duration * HZ)))
                tgt = np.array(w.pos)
                for k in range(1, n + 1):
                    pts.append((pos + (tgt - pos) * k / n, fin + (w.finger - fin) * k / n))
                pos, fin = tgt, w.finger
            tracks[side] = pts
        steps = max(len(v) for v in tracks.values()) + 3 * int(HZ)
        res = None
        for i in range(steps):
            arms = {}
            for side in ("left", "right"):
                pts = tracks.get(side)
                if not pts:
                    arms[side] = FAR[side]
                    continue
                p, f = pts[min(i, len(pts) - 1)]
                arms[side] = rigstate.Arm(p, p - np.array([0, 0, 0.013]), -f / 0.04695)
            m.step(arms)
            res = tr.update(m.states(), arms)
        return res

    def test_demo_unlocks(self):
        self.assertEqual(self._play(T.plans()).score, 1.0)

    def test_one_arm(self):
        for idle, want in (("right", 0.25), ("left", 0.0)):
            plan = T.plans()
            plan.pop(idle)
            self.assertEqual(self._play(plan).score, want, idle)

    def test_variants(self):
        want = {"wrong_pin": 0.5, "no_enter": 0.75, "delete_first": 0.25}
        for v, w in want.items():
            self.assertEqual(self._play(T.plans(v)).score, w, v)

    def test_hold_only_and_idle(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(T.demo()), {"right"})

    def test_task_json(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        tj = json.loads(p.read_text())
        text = " ".join(str(v) for v in tj.values())
        self.assertNotRegex(text, "[\\u2013\\u2014]")
        for s in ("K120", "2580", "47.6 mm", "2.0 mm"):
            self.assertIn(s, text)
        self.assertTrue(re.search(r"\d+ steps", tj["run_limit"]))


if __name__ == "__main__":
    unittest.main()
