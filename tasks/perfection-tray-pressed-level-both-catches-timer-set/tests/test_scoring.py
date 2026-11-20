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
WIN = int(round(L.RUN_WINDOW_S * HZ))
Q0 = (0.0, 0.0, 0.0, 1.0)
KIT_DIR = HERE.parents[1] / "tools" / "kit"


def scene(travel=0.0, rock=0.0, dial=0.0, switch=0.0, pawl=0.0, pawl_r=None):
    g = mk.state_from_poses(T.GAME, (*L.ORIGIN, *Q0), joints={S.TRAY: travel, S.ROCK: rock, S.DIAL: dial,
                                                              S.SWITCH: switch}, source="test")
    c = mk.state_from_poses(T.CATCH_PART, (*L.ORIGIN, *Q0),
                            joints={S.PAWL[1]: pawl, S.PAWL[-1]: pawl if pawl_r is None else pawl_r}, source="test")
    return {S.UNIT: g, S.CATCHES: c}


def arm(tcp, opening=0.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


AWAY = {"left": arm(T.PARK["left"] + np.array([0, 0, L.TOP])), "right": arm(T.PARK["right"] + np.array([0, 0, L.TOP]))}
ON_TRAY = {"left": arm((T.PRESS_X, T.PRESS_Y, L.TOP + L.TRAY_TOP0 - L.TRAVEL_HOLD + 0.013)),
           "right": arm((T.PRESS_X, -T.PRESS_Y, L.TOP + L.TRAY_TOP0 - L.TRAVEL_HOLD + 0.013))}
LATCHED = dict(travel=L.TRAVEL_HOLD)


def run(tr, states_fn, arms=AWAY, n=WIN + 2):
    res = None
    for i in range(n):
        res = tr.update(states_fn(i), arms)
    return res


def running(dial0=L.DIAL_FULL * 55 / 60, **kw):
    return lambda i: scene(dial=dial0 - L.DIAL_RATE * L.DT * i, **{**LATCHED, **kw})


class Geometry(unittest.TestCase):
    def test_reach_and_heights(self):
        for x, y in ((T.PRESS_X, T.PRESS_Y), (T.SW_X, T.SW_Y), T.DIAL_C):
            self.assertTrue(0.10 <= x <= 0.50, x)
        self.assertLessEqual(T.SAFE, 0.225)
        self.assertLessEqual(max(w.pos[2] - L.TOP for p in T.plans().values() for w in p), 0.225)

    def test_press_points_200_mm_apart_and_on_the_tray(self):
        self.assertGreaterEqual(2 * T.PRESS_Y, 0.13)
        self.assertLess(T.PRESS_Y + T.PAD_HALF, L.TRAY / 2)

    def test_one_arm_cannot_press_to_the_catch(self):
        f = L.spring_force(L.TRAVEL_SNAP) - T.WEIGHT + 2 * T.PAWL_EXTRA
        self.assertGreater(f, T.F_MAX + 2.0)  # about 11.5 N against about 8 N
        self.assertLess(f / 2, T.F_MAX - 1.5)  # each of two arms about 5.8 N

    def test_rock_stop_keeps_the_far_side_off_its_catch(self):
        # one side pressed to the centre's bottom stop: the far side is still short of the snap
        far = L.TRAVEL_MAX - T.Y_BARB * math.sin(L.ROCK_MAX)
        self.assertLess(far, L.TRAVEL_SNAP)

    def test_tooth_one_mm_over_the_barb_at_flush(self):
        r = S.read(scene(travel=L.TRAVEL_LATCH))
        for s in S.SIDES:
            self.assertAlmostEqual(r.teeth[s][0][2] - r.barbs[s][1][2], 0.001, places=6)
            self.assertAlmostEqual(S._y_overlap(r.barbs[s], r.teeth[s]), 0.004, places=6)

    def test_pawl_swing_clears_the_barb(self):
        need = math.asin((L.BARB_Y_OUT - L.TOOTH_Y_IN + 0.0005) / (L.PAWL_HINGE_Z - (L.TOOTH_Z0 + L.TOOTH_Z1) / 2))
        self.assertLess(need, L.PAWL_UPPER)
        r = S.read(scene(travel=L.TRAVEL_HOLD, pawl=need))
        for s in S.SIDES:
            self.assertLess(S._y_overlap(r.barbs[s], r.teeth[s]), 0.0)

    def test_springs_hold_the_tray_up(self):
        self.assertGreater(L.spring_force(0.0), T.WEIGHT + 1.0)

    def test_dial_and_switch_on_the_front_border(self):
        self.assertLess(L.DIAL_X + L.DIAL_R, L.TRAY_XC - L.TRAY / 2)
        self.assertGreater(L.DIAL_X - L.DIAL_R, -L.CASE / 2)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        res = run(S.Tracker(), lambda i: scene(), arms=AWAY)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["latched"])

    def test_full(self):
        res = run(S.Tracker(), running())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_tier_table(self):
        self.assertEqual(S.tier(True, True, True, True, True, True, True), 0.0)
        self.assertEqual(S.tier(False, True, True, True, False, True, True), 0.7)
        self.assertEqual(S.tier(False, True, False, True, True, True, True), 0.5)
        self.assertEqual(S.tier(False, False, True, True, True, True, True), 0.25)
        self.assertEqual(S.tier(False, False, True, True, True, False, True), 0.1)
        self.assertEqual(S.tier(False, False, True, True, True, False, False), 0.0)

    def test_wound_but_stopped_is_point_seven(self):
        res = run(S.Tracker(), lambda i: scene(dial=L.DIAL_FULL, switch=L.SWITCH_STOP, **LATCHED))
        self.assertEqual(res.score, 0.7)

    def test_dial_under_50_s_is_half(self):
        res = run(S.Tracker(), running(dial0=L.DIAL_FULL * 45 / 60))
        self.assertEqual(res.score, 0.5)

    def test_hands_on_the_tray_is_not_full(self):
        res = run(S.Tracker(), running(), arms=ON_TRAY)
        self.assertEqual(res.score, 0.7)

    def test_needs_a_full_second_of_running(self):
        tr = S.Tracker()
        res = run(tr, running(), n=WIN)
        self.assertLess(res.score, 1.0)

    def test_switch_at_start_but_dial_not_moving_is_not_running(self):
        res = run(S.Tracker(), lambda i: scene(dial=L.DIAL_FULL, **LATCHED))
        self.assertEqual(res.score, 0.7)


class Catches(unittest.TestCase):
    def test_popped_tray_not_caught(self):
        r = S.read(scene())
        self.assertFalse(any(S.caught(r, s) for s in S.SIDES))

    def test_pressed_with_pawls_open_not_caught(self):
        r = S.read(scene(travel=L.TRAVEL_HOLD, pawl=L.PAWL_UPPER))
        self.assertFalse(any(S.caught(r, s) for s in S.SIDES))

    def test_held_near_the_catch_but_above_it_scores_no_latch(self):
        # the tray pressed to just above the snap by hand, pawls resting: the barbs are still above the teeth
        res = run(S.Tracker(), running(travel=L.TRAVEL_SNAP - 0.003, pawl=0.18), arms=ON_TRAY)
        self.assertFalse(res.details["latched"])
        self.assertEqual(res.score, 0.1)

    def test_pressed_far_below_counts_only_while_the_tooth_is_over(self):
        r = S.read(scene(travel=L.TRAVEL_MAX - 0.001))
        self.assertTrue(all(S.caught(r, s) for s in S.SIDES))  # 5 mm under the tooth: still under it
        r = S.read(scene(travel=L.TRAVEL_MAX + 0.002))
        self.assertFalse(any(S.caught(r, s) for s in S.SIDES))

    def test_one_side_caught_is_a_quarter(self):
        tr = S.Tracker()
        run(tr, lambda i: scene(travel=0.0, pawl_r=0.0))
        res = run(tr, lambda i: scene(travel=L.TRAVEL_HOLD, pawl=L.PAWL_UPPER, pawl_r=0.0), n=2)
        res = run(tr, lambda i: scene(), n=3)
        self.assertEqual(res.score, 0.25)

    def test_forced_through_is_zero_for_good(self):
        tr = S.Tracker()
        bad = scene(travel=0.018)  # the barb's block 3 mm into the tooth in z, 4 mm in y
        self.assertTrue(S.forced(S.read(bad)))
        tr.update(bad, AWAY)
        res = run(tr, running())
        self.assertEqual(res.score, 0.0)

    def test_riding_the_bevel_is_not_forced(self):
        for t in np.arange(0.0, T.RAMP[0], 0.0005):
            self.assertFalse(S.forced(S.read(scene(travel=float(t)))))

    def test_nan_and_missing(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, AWAY).score, 0.0)
        bad = scene()
        bad[S.UNIT] = None
        self.assertFalse(tr.update(bad, AWAY).details["valid_state"])

    def test_unreadable_arms_never_released(self):
        res = run(S.Tracker(), running(), arms=None)
        self.assertLess(res.score, 1.0)

    def test_reset_jump_clears_memories(self):
        tr = S.Tracker()
        tr.update(scene(travel=0.018), AWAY)
        tr.update(scene(dial=L.DIAL_FULL, travel=L.TRAVEL_HOLD, switch=L.SWITCH_STOP), AWAY)
        tr.update(scene(), AWAY)
        self.assertIsNone(tr.forced_at)
        self.assertEqual(tr.resets, 1)


class Rules(unittest.TestCase):
    def test_cam_kicks_only_at_start_and_run_out(self):
        self.assertTrue(T.kicked(0.0, 0.0))
        self.assertFalse(T.kicked(L.SWITCH_STOP, 0.0))
        self.assertFalse(T.kicked(0.0, 0.5))

    def test_clockwork(self):
        self.assertAlmostEqual(T.dial_target(0.0, 1.0), 1.0 - L.DIAL_RATE * L.DT)
        self.assertEqual(T.dial_target(L.SWITCH_STOP, 1.0), 1.0)
        self.assertEqual(T.dial_target(0.0, 0.0), 0.0)
        self.assertAlmostEqual(L.dial_seconds(L.DIAL_FULL), 60.0)

    def _sim_targets(self, steps):
        """Run the simulator rule (_rules) on synthetic part states; return the dial drive target of each step."""
        class G:
            def __init__(self, sw, dial):
                self.joints = {S.SWITCH: sw, S.DIAL: dial}
        out = []
        for sw, dial in steps:
            T._rules({S.UNIT: G(sw, dial)})
            out.append(mk._hub().targets[T.GAME.name][S.DIAL])
        return out

    def test_sim_clock_runs_a_dial_held_by_friction(self):
        # GPU run 20261120T082751: a target of (measured dial minus one step) never beat the friction, so the dial
        # stood still at START. The clock's own target must pull away from a dial that has not moved yet.
        T.clock_reset()
        dial = L.DIAL_FULL * 55 / 60
        tg = self._sim_targets([(L.SWITCH_STOP, dial)] + [(0.0, dial)] * 60)
        self.assertEqual(tg[0], dial)  # STOP: held where left
        self.assertAlmostEqual(tg[1], dial - L.DIAL_RATE * L.DT)
        self.assertTrue(all(b <= a for a, b in zip(tg[1:], tg[2:])))
        self.assertAlmostEqual(dial - tg[-1], T.CLOCK_LAG)  # stuck dial: the error grows to the lag cap
        self.assertGreater(L.DIAL_SPRING * T.CLOCK_LAG, 2 * L.DIAL_FRICTION)
        k = next(i for i, t in enumerate(tg[1:], 1) if L.DIAL_SPRING * (dial - t) > L.DIAL_FRICTION)
        self.assertLess(k * L.DT, 1.0)  # the spring beats the friction within a second of START

    def test_sim_clock_follows_a_moving_dial_and_resets(self):
        T.clock_reset()
        dial = 2.0
        steps = [(0.0, dial - 0.6 * L.DIAL_RATE * L.DT * i) for i in range(30)]  # the dial lags the clock a little
        tg = self._sim_targets(steps)
        for i, t in enumerate(tg):
            self.assertAlmostEqual(t, dial - L.DIAL_RATE * L.DT * (i + 1))  # independent of the measured angle
        tg = self._sim_targets([(L.SWITCH_STOP, 1.5), (0.0, 1.5)])
        self.assertEqual(tg, [1.5, 1.5 - L.DIAL_RATE * L.DT])  # STOP clears it; START seeds again from the dial
        T._kit_reset()
        self.assertIsNone(T._CLOCK["target"])
        tg = self._sim_targets([(0.0, 0.0)] * 40)  # run out at START: driven onto the 0 stop, never above it
        self.assertAlmostEqual(tg[-1], T.CLOCK_FLOOR)
        self.assertTrue(T.kicked(0.0, 0.0))
        T.clock_reset()

    def test_offline_stand_in_clock_unchanged(self):
        m = T.OfflineModel()
        m.dial, m.switch = 1.0, 0.0
        m.step(AWAY)
        self.assertAlmostEqual(m.dial, 1.0 - L.DIAL_RATE * L.DT)
        m.switch = L.SWITCH_STOP
        m.step(AWAY)
        self.assertAlmostEqual(m.dial, 1.0 - L.DIAL_RATE * L.DT)

    def test_offline_pinch_turns_with_the_hand(self):
        m = T.OfflineModel()
        m.switch = L.SWITCH_STOP
        for yaw in np.linspace(0.0, -1.0, 11):
            x, y = T.pinch_point(yaw)
            m.step({"left": AWAY["left"], "right": arm((x, y, L.TOP + T.Z_DIAL))})
        self.assertAlmostEqual(m.dial, 1.0, places=6)


def replay(plan: dict, hz=HZ, tail_s=3.0):
    """Interpolate each arm's plan linearly at ``hz`` (no IK: TCP straight lines), feed the offline model and the
    tracker; return the last result and the best score."""
    from yam_arena.constants import FINGER_OPEN
    ready = {"left": np.array([0.30, 0.30, L.TOP + 0.20]), "right": np.array([0.30, -0.30, L.TOP + 0.20])}
    tracks = {}
    for side in ("left", "right"):
        pts = [(0.0, ready[side], FINGER_OPEN)]
        t = 0.0
        for w in plan.get(side, []):
            t += w.duration
            pts.append((t, np.asarray(w.pos, dtype=float), w.finger))
        tracks[side] = pts
    end = max(p[-1][0] for p in tracks.values()) + tail_s
    model, tr = T.OfflineModel(), S.Tracker()
    best, res = 0.0, None
    for k in range(int(end * hz) + 1):
        t = k / hz
        arms = {}
        for side, pts in tracks.items():
            for (t0, p0, f0), (t1, p1, f1) in zip(pts, pts[1:] + [pts[-1]]):
                if t <= t1 or (t1 == pts[-1][0]):
                    a = 0.0 if t1 <= t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
                    p, f = p0 + a * (p1 - p0), f0 + a * (f1 - f0)
                    break
            arms[side] = arm(p, opening=float(np.clip(f / FINGER_OPEN, 0.0, 1.0)))
        model.step(arms)
        res = tr.update(model.states(), arms)
        best = max(best, res.score)
    return res, best


class Demo(unittest.TestCase):
    def test_demo_scores_full(self):
        res, _ = replay(T.plans())
        self.assertEqual(res.score, 1.0, res.details)

    def test_one_arm_idle_is_far_below(self):
        for idle in ("left", "right"):
            p = T.plans()
            p.pop(idle)
            res, _ = replay(p)
            self.assertLessEqual(res.score, 0.25, (idle, res.details))

    def test_variants(self):
        want = {"no_stop": 0.1, "no_wind": 0.25, "no_start": 0.7, "shallow": 0.1, "hold_on": 0.7}
        for v, s in want.items():
            res, _ = replay(T.plans(v))
            self.assertEqual(res.score, s, (v, res.details))

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]

    def test_demo_within_the_run_limit(self):
        steps = max(sum(w.duration for w in p) for p in T.plans().values()) * HZ
        data = json.loads((HERE / "task.json").read_text())
        limit = int(re.match(r"([\d,]+)", data["run_limit"]).group(1).replace(",", ""))
        self.assertLess(steps + 3 * HZ, limit)


class Files(unittest.TestCase):
    def test_kit_files_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).exists():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_text_has_no_dashes_or_names(self):
        for f in ("task.json", "README.md"):
            text = (HERE / f).read_text()
            self.assertNotIn(chr(0x2014), text, f)
            self.assertNotIn(chr(0x2013), text, f)
            self.assertIsNone(re.search(r"T-70|tasks/|tools/|claude|vihan", text, re.I), f)


if __name__ == "__main__":
    unittest.main()
