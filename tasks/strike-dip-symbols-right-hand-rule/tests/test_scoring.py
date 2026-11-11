"""Offline tests for scoring.py and task.py: synthetic token states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    NAMES,
    STATIONS,
    Tracker,
    dir_err_deg,
    bar_yaw_deg,
    bottom_gap,
    errors,
    evaluate,
    held,
    lying,
    outline_distance,
    symbol_sixths,
    tick_yaw_deg,
    tilt_deg,
    valid,
)

HZ = 20.0
FULL = int(round(L.STILL_WINDOW_S * HZ)) * L.STILL_WINDOWS + 1  # 21 steps: 1 s of history


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    raise RuntimeError("tools/offline/yam_stub not found")


def _kin_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    raise RuntimeError("tools/offline/yamkin.py not found")


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def tok(xy, tick_yaw_deg, dz=0.0, pitch=0.0, roll=0.0):
    """A token as mechkit reports it (only what the scorer reads): its bar's centre at xy, its base ``dz`` above the
    map's surface, its tick pointing ``tick_yaw_deg``."""
    o = np.array([xy[0], xy[1], L.MAP_TOP_W + dz])
    return SimpleNamespace(origin=o, quat_xyzw=quat(tick_yaw_deg, pitch, roll), pos=o.copy(), lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), low=o - 0.04, high=o + 0.04)


def at(station, tick_yaw_deg, dx=0.0, dy=0.0, **kw):
    x, y = L.STATIONS[station]
    return tok((x + dx, y + dy), tick_yaw_deg, **kw)


def strike(s):
    return L.STRIKE_YAW_DEG[s]


def dip(s):
    return L.DIP_YAW_DEG[s]


def start_states():
    return {n: tok((x, y), yaw) for n, (x, y, yaw) in L.TOKENS.items()}


def plotted(a=None, b=None, c=None):
    """Tokens 1, 2, 3 on A, B, C with ticks at the given yaws (default: right)."""
    return {"token_1": at("A", dip("A") if a is None else a), "token_2": at("B", dip("B") if b is None else b),
            "token_3": at("C", dip("C") if c is None else c)}


def arm(tips, opening=1.0):
    tips = np.asarray(tips, dtype=float)
    return SimpleNamespace(tcp=tips + np.array([0.0, 0.0, 0.013]), tips=tips, opening=opening)


PARKED = {"left": arm((0.20, 0.25, L.TOP + 0.147)), "right": arm((0.20, -0.25, L.TOP + 0.147))}
CLOSED = L.BAR[1] / L.FULL_GAP  # closed on the bar: the fingers stop at its 60 mm length


def holding(station, side="left"):
    """That gripper closed on the token at the station as the demo holds it (pad tips 4 mm over the map)."""
    x, y = L.STATIONS[station]
    other = "right" if side == "left" else "left"
    return {side: arm((x, y, L.MAP_TOP_W + 0.004), CLOSED), other: PARKED[other]}


def feed(tr, states, arms=PARKED, n=1):
    r = None
    for _ in range(n):
        r = tr.update(states, arms)
    return r


def settled(states, arms=PARKED, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, states, arms, n)


def scored(states, arms=PARKED):
    return settled(states, arms)[1].score


class Rule(unittest.TestCase):
    def test_right_hand_rule_arithmetic(self):
        self.assertEqual({s: L.dip_direction(s) for s in STATIONS}, {"A": 120.0, "B": 340.0, "C": 225.0})
        for s, want in (("A", -30.0), ("B", 110.0), ("C", -135.0)):
            self.assertAlmostEqual(strike(s), want)
        for s, want in (("A", -120.0), ("B", 20.0), ("C", 135.0)):
            self.assertAlmostEqual(dip(s), want)
            # the tick is square to the bar, on the clockwise side seen from above (yaw decreasing)
            self.assertAlmostEqual(math.remainder(math.radians(dip(s) - strike(s)), 2 * math.pi), -math.pi / 2)

    def test_bearings_turn_clockwise_on_the_table(self):
        self.assertAlmostEqual(L.bearing_to_yaw(0.0), 0.0)  # north: +x
        self.assertAlmostEqual(L.bearing_to_yaw(90.0), -90.0)  # east: -y, the arms' right
        self.assertAlmostEqual(L.bearing_to_yaw(270.0), 90.0)  # west: +y

    def test_wrong_answers_are_far_outside_the_windows(self):
        for s in STATIONS:
            b = L.READINGS[s][0]
            lhr = L.bearing_to_yaw(b - 90.0)  # the left-hand habit: tick 180 degrees off
            st, tk = errors(at(s, lhr), s)
            self.assertLess(st, 1e-6)
            self.assertAlmostEqual(tk, 180.0)
            as_dip = L.bearing_to_yaw(b)  # the first number read as the dip direction
            st, tk = errors(at(s, as_dip), s)
            self.assertAlmostEqual(st, 90.0)
        ccw = {s: errors(at(s, L.READINGS[s][0] + 90.0), s)[0] for s in STATIONS}  # yaw +b: bearings CCW
        self.assertEqual({s: round(v, 6) for s, v in ccw.items()}, {"A": 60.0, "B": 40.0, "C": 90.0})


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        st = start_states()
        self.assertTrue(Tracker.at_start(st))
        for n, t in st.items():
            self.assertAlmostEqual(tilt_deg(t), 0.0, places=6)
            self.assertAlmostEqual(bottom_gap(t), 0.0, places=9)
            self.assertTrue(lying(t))

    def test_start_bars_are_15_degrees_or_more_off_every_strike(self):
        for n, (_, _, yaw) in L.TOKENS.items():
            for s in STATIONS:
                st, _ = errors(at(s, yaw), s)
                self.assertGreaterEqual(st, 15.0 - 1e-9, (n, s))

    def test_tick_and_bar_yaw(self):
        t = at("A", dip("A"))
        self.assertAlmostEqual(tick_yaw_deg(t), -120.0)
        self.assertAlmostEqual(bar_yaw_deg(t), -30.0)
        self.assertIsNone(tick_yaw_deg(at("A", 0.0, pitch=90.0)))  # standing on its tick's end

    def test_outline_distance(self):
        t = at("A", 0.0)  # tick toward +x, bar along y
        x, y = L.STATIONS["A"]
        z = L.MAP_TOP_W + 0.02
        self.assertEqual(outline_distance(t, (x, y, z)), 0.0)
        self.assertAlmostEqual(outline_distance(t, (x - 0.030, y, z)), 0.020)  # 20 mm behind the bar's back edge
        self.assertAlmostEqual(outline_distance(t, (x + 0.045, y, z)), 0.010)  # 10 mm past the tick's end
        self.assertAlmostEqual(outline_distance(t, (x, y + 0.050, z)), 0.020)  # 20 mm past the bar's end

    def test_release_reach(self):
        self.assertAlmostEqual(L.release_reach(1.0), math.hypot(0.04695 + 0.010, 0.020) + 0.005, places=6)
        self.assertAlmostEqual(L.release_reach(1.0), 0.0654, places=3)
        self.assertAlmostEqual(L.release_reach(CLOSED), 0.0497, places=3)

    def test_stations_are_far_apart(self):
        pts = list(L.STATIONS.values())
        for i, p in enumerate(pts):
            for q in pts[i + 1:]:
                self.assertGreater(math.dist(p, q), 0.2)  # a token can qualify at one station only


class Tiers(unittest.TestCase):
    def test_plotted_scores_one(self):
        tr, r = settled(plotted())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["sixths"], 6)
        self.assertEqual(r.details["zero_reason"], "")

    def test_every_score_is_reachable(self):
        lhr = {s: dip(s) + 180.0 for s in STATIONS}
        far = (0.27, 0.32)
        cases = {
            6: plotted(),
            5: plotted(c=lhr["C"]),
            4: {**plotted(), "token_3": tok(far, 0.0)},
            3: plotted(a=lhr["A"], b=lhr["B"], c=lhr["C"]),
            2: {**plotted(b=lhr["B"], c=lhr["C"]), "token_1": tok(far, 0.0)},
            1: {"token_1": at("A", lhr["A"]), "token_2": tok(far, 0.0), "token_3": tok((0.27, -0.32), 0.0)},
            0: start_states(),
        }
        for sixths, st in cases.items():
            r = settled(st)[1]
            self.assertEqual(r.details["sixths"], sixths)
            self.assertAlmostEqual(r.score, sixths / 6.0)
            self.assertEqual(r.success, sixths == 6)

    def test_window_edges(self):
        for s in STATIONS:
            self.assertEqual(symbol_sixths(at(s, dip(s) + 5.9), s), 2)
            self.assertEqual(symbol_sixths(at(s, dip(s) - 5.9), s), 2)
            self.assertEqual(symbol_sixths(at(s, dip(s) + 6.1), s), 0)  # the bar is then 6.1 off too
            self.assertEqual(symbol_sixths(at(s, dip(s) + 180.0 + 5.9), s), 1)
            self.assertEqual(symbol_sixths(at(s, dip(s) + 180.0 - 6.1), s), 0)
        st = plotted()
        st["token_1"] = at("A", dip("A"), dx=0.0079)
        self.assertEqual(scored(st), 1.0)
        st["token_1"] = at("A", dip("A"), dx=0.0057, dy=-0.0057)  # 8.06 mm
        self.assertAlmostEqual(scored(st), 4 / 6)

    def test_any_token_may_go_anywhere(self):
        st = plotted()
        swapped = {"token_1": st["token_3"], "token_2": st["token_1"], "token_3": st["token_2"]}
        self.assertEqual(scored(swapped), 1.0)

    def test_small_tilt_and_lying_on_the_ring(self):
        st = plotted()
        st["token_1"] = at("A", dip("A"), pitch=4.0, dz=0.001)
        st["token_2"] = at("B", dip("B"), dz=L.INK_PROUD)
        self.assertEqual(scored(st), 1.0)

    def test_score_is_idempotent(self):
        tr, r = settled(plotted())
        n = tr.steps
        self.assertEqual(tr.result(plotted(), PARKED).score, 1.0)
        self.assertEqual(tr.result(plotted(), PARKED).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def one(self, token_1):
        """Tokens 2 and 3 right; token 1 as given: the score is 2/3 unless token 1 counts at A."""
        st = plotted()
        st["token_1"] = token_1
        return scored(st)

    def test_off_centre(self):
        self.assertAlmostEqual(self.one(at("A", dip("A"), dx=0.009)), 4 / 6)

    def test_tipped(self):
        self.assertAlmostEqual(self.one(at("A", dip("A"), pitch=5.5)), 4 / 6)
        self.assertAlmostEqual(self.one(at("A", dip("A"), roll=90.0)), 4 / 6)  # on its side

    def test_upside_down(self):
        self.assertAlmostEqual(self.one(at("A", dip("A"), roll=180.0, dz=L.TOKEN_H)), 4 / 6)

    def test_held_up_or_stacked(self):
        self.assertAlmostEqual(self.one(at("A", dip("A"), dz=0.0035)), 4 / 6)
        self.assertAlmostEqual(self.one(at("A", dip("A"), dz=L.TOKEN_H)), 4 / 6)  # on another token

    def test_bar_off_the_strike(self):
        self.assertAlmostEqual(self.one(at("A", dip("A") + 90.0)), 4 / 6)
        self.assertAlmostEqual(self.one(at("A", dip("A") + 20.0)), 4 / 6)

    def test_held(self):
        st = plotted()
        self.assertAlmostEqual(settled(st, holding("A"))[1].score, 4 / 6)
        self.assertAlmostEqual(settled(st, holding("B", "right"))[1].score, 4 / 6)

    def test_release_height_and_reach(self):
        st = plotted()
        x, y = L.STATIONS["A"]
        top = L.MAP_TOP_W + L.TOKEN_H
        above = {"left": arm((x, y, top + 0.0099)), "right": PARKED["right"]}
        self.assertAlmostEqual(settled(st, above)[1].score, 4 / 6)  # an open hand on top of it: pressing
        above = {"left": arm((x, y, top + 0.0101)), "right": PARKED["right"]}
        self.assertEqual(settled(st, above)[1].score, 1.0)
        # beside it, low, fully open: the outline (bar end 30 mm from the centre along the strike) within 65 mm
        d = math.radians(strike("A"))
        near = {"left": arm((x + 0.090 * math.cos(d), y + 0.090 * math.sin(d), L.MAP_TOP_W + 0.004)),
                "right": PARKED["right"]}
        self.assertAlmostEqual(settled(st, near)[1].score, 4 / 6)  # 60 mm from its end
        far = {"left": arm((x + 0.100 * math.cos(d), y + 0.100 * math.sin(d), L.MAP_TOP_W + 0.004)),
               "right": PARKED["right"]}
        self.assertEqual(settled(st, far)[1].score, 1.0)  # 70 mm from its end

    def test_moving(self):
        tr = Tracker(HZ)
        st = plotted()
        feed(tr, st, n=FULL)
        moved = plotted()
        moved["token_1"] = at("A", dip("A"), dx=0.002)
        r = feed(tr, moved)  # a 2 mm jump inside the last second
        self.assertAlmostEqual(r.score, 4 / 6)
        r = feed(tr, moved, n=FULL)
        self.assertEqual(r.score, 1.0)

    def test_turning_in_place_is_moving(self):
        tr = Tracker(HZ)
        feed(tr, plotted(), n=FULL)
        r = feed(tr, plotted(a=dip("A") + 4.0))  # the tick's end moves 2.4 mm, the centre not at all
        self.assertAlmostEqual(r.score, 4 / 6)
        self.assertIn("A_no_symbol", r.details["zero_reason"])


class Stillness(unittest.TestCase):
    def test_needs_one_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, plotted(), n=FULL - 1)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["path_mm"])
        r = feed(tr, plotted())
        self.assertEqual(r.score, 1.0)

    def test_jitter_under_1_5_mm_is_still(self):
        tr = Tracker(HZ)
        rng = np.random.default_rng(1)
        r = None
        for _ in range(FULL + 5):
            st = plotted()
            for n in NAMES:
                st[n].origin = st[n].origin + rng.uniform(-3e-5, 3e-5, 3)
            r = tr.update(st, PARKED)
        self.assertEqual(r.score, 1.0)

    def test_every_tier_needs_stillness(self):
        tr = Tracker(HZ)
        st = plotted(a=dip("A") + 180.0)
        feed(tr, st, n=FULL)
        shifted = plotted(a=dip("A") + 180.0)
        shifted["token_1"] = at("A", dip("A") + 180.0, dy=0.002)
        r = feed(tr, shifted)
        self.assertAlmostEqual(r.score, 4 / 6)  # the 1/6 at A needs it still too


class Exploits(unittest.TestCase):
    def test_left_hand_rule_scores_half(self):
        st = plotted(a=dip("A") + 180.0, b=dip("B") + 180.0, c=dip("C") + 180.0)
        self.assertAlmostEqual(scored(st), 0.5)

    def test_readings_taken_as_dip_direction_score_zero(self):
        st = plotted(a=L.bearing_to_yaw(30.0), b=L.bearing_to_yaw(250.0), c=L.bearing_to_yaw(135.0))
        self.assertEqual(scored(st), 0.0)

    def test_bearings_counted_counter_clockwise_score_zero(self):
        for rhr in (True, False):  # either tick side: the bars are 60, 40 and 90 degrees off
            st = {n: at(s, L.READINGS[s][0] + (90.0 if rhr else -90.0) + 0.0)
                  for n, s in zip(NAMES, STATIONS)}
            self.assertEqual(scored(st), 0.0)

    def test_moved_without_turning_scores_zero(self):
        import itertools
        for perm in itertools.permutations(NAMES):
            st = {n: at(s, L.TOKENS[n][2]) for n, s in zip(perm, STATIONS)}
            self.assertEqual(scored(st), 0.0, perm)

    def test_nothing_moved_scores_zero(self):
        self.assertEqual(scored(start_states()), 0.0)

    def test_two_tokens_at_one_station_only_the_nearest_counts(self):
        st = plotted()
        st["token_2"] = at("A", dip("A") + 180.0, dx=0.005)  # 5 mm off, tick reversed
        st["token_1"] = at("A", dip("A"), dx=-0.002)  # 2 mm off, right
        self.assertAlmostEqual(scored(st), 4 / 6)  # A 1/3, B empty, C 1/3
        st["token_2"] = at("A", dip("A") + 180.0, dx=0.002)
        st["token_1"] = at("A", dip("A"), dx=-0.005)
        self.assertAlmostEqual(scored(st), 3 / 6)  # A 1/6, B empty, C 1/3

    def test_hedging_two_orientations_stacked(self):
        st = plotted()
        st["token_2"] = at("A", dip("A") + 180.0, dz=L.TOKEN_H)  # the other answer stacked on top: never counts
        self.assertAlmostEqual(scored(st), 4 / 6)

    def test_last_token_still_gripped(self):
        self.assertAlmostEqual(settled(plotted(), holding("C"))[1].score, 4 / 6)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(plotted())
        n, hist = tr.steps, list(tr.history)
        for bad in ("nan", "inf", "zero_quat", "no_origin", "short"):
            st = plotted()
            t = st["token_2"]
            if bad == "nan":
                t.origin = np.array([np.nan, 0.0, 0.0])
            elif bad == "inf":
                t.quat_xyzw = np.array([0.0, 0.0, np.inf, 1.0])
            elif bad == "zero_quat":
                t.quat_xyzw = np.zeros(4)
            elif bad == "no_origin":
                del t.origin
            else:
                t.origin = np.array([0.3, 0.0])
            self.assertFalse(valid(st))
            r = tr.update(st, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(len(tr.history), len(hist))
        self.assertEqual(tr.bad_steps, 5)
        self.assertGreater(tr.steps, n)
        self.assertEqual(feed(tr, plotted()).score, 1.0)  # the history was kept
        self.assertEqual(tr.update({"token_1": plotted()["token_1"]}, PARKED).score, 0.0)  # a token missing

    def test_unnormalised_quaternion(self):
        st = plotted()
        for n in NAMES:
            st[n].quat_xyzw = st[n].quat_xyzw * 3.0
        self.assertEqual(scored(st), 1.0)

    def test_random_states_give_only_sixths(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(400):
            st = {}
            for n in NAMES:
                s = rng.choice(STATIONS)
                yaw = rng.choice([dip(s), dip(s) + 180.0, rng.uniform(-180, 180)])
                st[n] = at(s, yaw, dx=rng.uniform(-0.012, 0.012), dy=rng.uniform(-0.012, 0.012),
                           dz=rng.choice([0.0, 0.0, 0.004, L.TOKEN_H]), pitch=rng.choice([0.0, 0.0, 7.0]))
            r = settled(st, n=FULL)[1]
            self.assertIn(r.details["sixths"], range(7))
            self.assertAlmostEqual(r.score * 6, r.details["sixths"])
            json.dumps(r.details)
            seen.add(r.details["sixths"])
        self.assertGreaterEqual(len(seen), 5)

    def test_unreadable_arms(self):
        self.assertEqual(settled(plotted(), None)[1].score, 1.0)  # taken as let go
        tr, r = settled(plotted(), {"left": SimpleNamespace(tips=np.array([np.nan, 0, 0]), opening=1.0),
                                    "right": holding("B", "right")["right"]})
        self.assertAlmostEqual(r.score, 4 / 6)  # the readable arm is still checked
        self.assertEqual(tr.arms_unavailable, FULL)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(plotted())
        r = feed(tr, start_states())
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(tr.history), 1)

    def test_jittered_start_resets(self):
        tr, _ = settled(plotted())
        st = {n: tok((x + 0.02, y - 0.02), yaw) for n, (x, y, yaw) in L.TOKENS.items()}
        feed(tr, st)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_or_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker(HZ)
        a = start_states()
        b = {n: tok((x + 0.05, y), yaw) for n, (x, y, yaw) in L.TOKENS.items()}
        feed(tr, b, n=3)
        for k in range(1, 11):  # back in 5 mm steps
            feed(tr, {n: tok((x + 0.05 - 0.005 * k, y), yaw) for n, (x, y, yaw) in L.TOKENS.items()})
        self.assertEqual(tr.steps, 13)
        feed(tr, plotted())
        feed(tr, plotted(a=0.0))  # a jump, but not onto the starting marks
        self.assertEqual(tr.steps, 15)
        self.assertTrue(Tracker.at_start(a))


class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, states):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        spec = {t.name: t for t in self.task.TOKENS}
        return {n: self.mk.state_from_poses(spec[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in states.items()}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], list(NAMES))
        self.assertLessEqual(len(t.OBJECTS), 50)  # no more than the largest scene already run on the GPU
        for tk in t.TOKENS:
            self.assertAlmostEqual(tk.mass, 0.025)
            self.assertLessEqual(tk.mass, 0.30)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 2.0))
        self.assertAlmostEqual(t.MAP.pos[2] + t.MAP.size[2] / 2, L.MAP_TOP_W)
        for m in t.MARKS:
            self.assertAlmostEqual(m.pos[2] + m.size[2] / 2, L.MAP_TOP_W + L.INK_PROUD)
            self.assertGreaterEqual(min(m.size[:2]), L.RING_W - 1e-12)  # every printed line 2 mm or more wide
        # nothing in the screenshot's hidden band (|y| > 0.45 m near x < 0.2 m); every token on the map
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
                self.assertGreaterEqual(lo[0], L.MAP_X0)
            else:
                half = np.hypot(*o.size[:2]) / 2 if o.yaw else max(o.size[:2]) / 2
                lo = np.asarray(o.pos) - half
                hi = np.asarray(o.pos) + half
                if o is not t.MAP:
                    self.assertTrue(L.MAP_X0 < lo[0] and hi[0] < L.MAP_X1 and L.MAP_Y0 < lo[1] and hi[1] < L.MAP_Y1)
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_marks_do_not_overlap_each_other_or_the_tokens(self):
        sys.path.insert(0, str(_kin_dir()))
        import scenegeom as G
        marks = L.ink_marks()
        solids = [G.Solid(n, "box", (cx, cy, L.INK_Z_W), (sx / 2, sy / 2, L.INK_T / 2), yaw)
                  for n, ((cx, cy), (sx, sy), yaw) in marks.items()]
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                self.assertGreater(G.gap(a, b), -0.0005, (a.name, b.name))
        # the letters stay clear of a token on its station, whichever way it is turned (35 mm reach)
        for s, (x, y) in L.STATIONS.items():
            for n, ((cx, cy), (sx, sy), _) in marks.items():
                if n.startswith("label_"):
                    self.assertGreater(math.hypot(cx - x, cy - y) - math.hypot(sx, sy) / 2, 0.035)

    def test_token_geometry_matches_the_scorer(self):
        tk = self.task.TOKENS[0]
        lo, hi = tk.local_bounds()
        np.testing.assert_allclose(lo, [-0.010, -0.030, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.035, 0.030, 0.012], atol=1e-12)
        self.assertAlmostEqual(tk.center_of_mass()[0], 0.0225 * 3.0 / 17.4, places=6)  # 3.9 mm toward the tick
        vol = sum(float(np.prod(s.size)) for s in tk.shapes)
        self.assertAlmostEqual(vol * 1410.0, L.TOKEN_MASS, delta=0.001)  # acetal
        colours = {tuple(s.color) if s.color is not None else tuple(tk.color) for s in tk.shapes}
        self.assertEqual(colours, {L.BAR_RED, L.TICK_YELLOW})
        for t in self.task.TOKENS:
            self.assertTrue(lying(t.rest_state()))
        self.assertTrue(Tracker.at_start({t.name: t.rest_state() for t in self.task.TOKENS}))
        st = self.sim(plotted())
        self.assertAlmostEqual(tick_yaw_deg(st["token_1"]), -120.0)

    def test_offline_state_path(self):
        tk = self.task.TOKENS[2]
        c, _, yaw = tk.offline_box()
        box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = tk.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [0.25, 0.0, L.MAP_TOP_W], atol=1e-12)

    def test_hooks_and_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(PARKED)
        self.assertFalse(t.success(self.sim(start_states())))
        good = self.sim(plotted())
        for _ in range(FULL):
            self.feed_arms(PARKED)
            t.success(good)
        self.feed_arms(PARKED)
        self.assertTrue(t.success(good))
        self.assertEqual(t.score(good), 1.0)
        self.feed_arms(holding("C"))
        self.assertAlmostEqual(t.score(good), 4 / 6)  # read with the arms of the step at hand
        self.feed_arms(PARKED)
        self.rig.offline_reset()
        self.assertEqual(tr.steps, 0)
        self.feed_arms(PARKED)
        t.success(good)
        t.success(good)
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's episode counter alone: a new trial too
        t.success(good)
        self.assertEqual(tr.steps, 1)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(PARKED)
                self.task.success(self.sim(plotted()))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "sixths", "zero_reason", "A", "B", "C", "path_mm",
                          "arms_readable", "arms_unavailable", "left_tips", "right_opening", "token_1_quat"):
                    self.assertIn(k, row)
                for k in ("origin_mm", "tick_yaw_deg", "bar_yaw_deg", "tilt_deg", "bottom_mm", "held", "still",
                          "station", "off_station_mm", "strike_err_deg", "tick_err_deg", "qualifies"):
                    self.assertIn(k, row["token_1"])
                self.assertEqual(row["token_1"]["station"], "A")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(plotted())))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every open and close takes 1.5 s
            labels = [w.label for w in wps]
            for i, lab in enumerate(labels):
                if lab.startswith("grip ") or lab == "let go of it":
                    self.assertEqual(labels[i - 1], "settle")
                    self.assertGreaterEqual(wps[i - 1].duration, 0.5)
                if lab.startswith("slowly"):
                    self.assertGreaterEqual(wps[i].duration, 1.2)
                    self.assertAlmostEqual(wps[i - 1].pos[2] - wps[i].pos[2], t.SLOW)
            self.assertTrue(all(L.TOP + 0.016 < w.pos[2] for w in wps))  # pad tips 3 mm or more over the table
        for arm_side, moves in t.PLAN.items():
            for token, station in moves:
                x0, y0, yaw0 = L.TOKENS[token]
                pick = t.PICK_YAW_DEG[token]
                self.assertAlmostEqual(math.remainder(math.radians(pick - yaw0), math.pi), 0.0)  # closes along the bar
                self.assertAlmostEqual(abs(t.turn_deg(token, station)), 45.0 if token == "token_3" else 90.0)
                bx, by = (0.0, 0.305 if arm_side == "left" else -0.305)
                for (tx, ty), yaw in (((x0, y0), pick), (t.place_point(token, station, arm_side),
                                                         pick + t.turn_deg(token, station) + t.YAW_TRIM_DEG)):
                    bearing = math.degrees(math.atan2(ty - by, tx - bx))  # joint 6: within 90 of the bearing
                    self.assertLessEqual(dir_err_deg(yaw, bearing), 90.0, (token, station, yaw, bearing))
                self.assertAlmostEqual(yaw0 + t.turn_deg(token, station) - dip(station), 0.0, delta=1e-9) \
                    if abs(yaw0 + t.turn_deg(token, station) - dip(station)) < 180 else None
                end = math.degrees(math.remainder(math.radians(yaw0 + t.turn_deg(token, station)), 2 * math.pi))
                self.assertAlmostEqual(end, dip(station))
                px, py = t.place_point(token, station, arm_side)
                sx, sy = L.STATIONS[station]
                self.assertAlmostEqual(math.hypot(px - sx, py - sy), t.LEAD)
                self.assertGreater((px - sx) * (sx - x0) + (py - sy) * (sy - y0), 0)  # leads along the carry
                carry = [w for w in plan[arm_side] if w.label == f"carry it over station {station}, turning it"][0]
                self.assertAlmostEqual(math.degrees(carry.yaw), pick + t.turn_deg(token, station) + t.YAW_TRIM_DEG)
        self.assertAlmostEqual(t.GRIP, -0.026)
        self.assertAlmostEqual(t.YAW_TRIM_DEG, -1.15)
        # the right arm parks before the left arm comes down at the middle (C)
        right_end = sum(w.duration for w in plan["right"])
        left = plan["left"]
        k = [w.label for w in left].index("down to just above it", 10)
        self.assertLess(right_end, sum(w.duration for w in left[:k]))
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule_and_withholds_the_answer(self):
        ins = self.task.INSTRUCTION
        for s in ("right-hand rule", "90 degrees clockwise from strike", "A 030/40", "B 250/25", "C 135/60",
                  "north at the top", "away from the arms", "A left, C middle, B right", "centred", "lying flat",
                  "let go"):
            self.assertIn(s, ins)
        for s in ("down-dip", "points toward", "yaw", "bearing", "counter-clockwise", "120", "340", "225"):
            self.assertNotIn(s, ins)
        self.assertLessEqual(len(ins), 490)

    def test_task_json_matches(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], self.task.INSTRUCTION)
        for s in ("within 8 mm", "under 5 degrees", "within 3 mm", "lower than 10 mm", "65 mm fully open",
                  "50 mm closed", "under 1.5 mm", "two 0.5 s windows", "within 6 degrees", "1/3", "1/6",
                  "bearing 120", "toward 340", "toward 225", "2 s"):
            self.assertIn(s, d["scoring"])
        for s in ("(33.0, +22.0)", "(33.0, -22.0)", "(30.0, 0)", "25 g", "20 x 60 mm", "12 mm tall",
                  "tick yaw 150", "tick yaw 110", "tick yaw 180", "acetal", "1 mm card"):
            self.assertIn(s, d["setup"])
        self.assertEqual(d["scores"], "TBD after the GPU runs (offline kinematic check predicts demo: 1.0)")
        text = json.dumps(d).lower() + (HERE / "README.md").read_text().lower()
        for w in ("\u2013", "\u2014"):  # no en or em dashes
            self.assertNotIn(w, text)


class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, the left arm cannot reach station B and the right arm cannot reach A;
    both reach C and their own side's station."""

    def test_far_station_out_of_reach_straight_down(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        z = L.MAP_TOP_W + 0.018
        for side, far in (("left", "B"), ("right", "A")):
            base = K.BASES["station"][side]
            for yaw in np.linspace(-math.pi, math.pi, 13):
                self.assertFalse(K.reachable((*L.STATIONS[far], z), float(yaw), base)[0], (side, far, yaw))
        for side, own in (("left", "A"), ("right", "B")):
            base = K.BASES["station"][side]
            self.assertTrue(K.reachable((*L.STATIONS[own], z), math.radians(20.0), base)[0])
            self.assertTrue(K.reachable((*L.STATIONS["C"], z), math.radians(-45.0), base)[0])


if __name__ == "__main__":
    unittest.main()
