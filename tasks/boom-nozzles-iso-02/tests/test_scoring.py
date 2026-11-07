"""Offline tests for scoring.py and the hooks in task.py: synthetic nozzle states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import Tracker, lowest, place_of, seated, size_of, tilt_deg, valid  # noqa: E402

FILL = 12  # control steps fed to fill the 0.5 s stillness window (11 positions)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0)):
    """An ObjectState-like nozzle: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    q = np.array(quat, float)
    x, y, z, w = q / np.linalg.norm(q)
    a = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    ext = np.abs(a) * L.NOZZLE_H / 2 + L.NOZZLE_R * np.sqrt(np.clip(1.0 - a ** 2, 0.0, 1.0))
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - ext, high=c + ext)


def nozzle(x, y, bottom, tilt=0.0, lean_to=0.0, vel=(0, 0, 0)):
    """A nozzle with its centre seen from above at (x, y) and its lowest point at ``bottom``, its axis tilted
    ``tilt`` degrees toward the heading ``lean_to`` (degrees from +x)."""
    t = math.radians(tilt)
    zc = bottom + (L.NOZZLE_H / 2) * abs(math.cos(t)) + L.NOZZLE_R * abs(math.sin(t))
    q = about("z", lean_to)
    tq = about("y", tilt)
    # turn about z, then tilt about y in the turned frame (lean toward lean_to)
    qx, qy, qz, qw = q
    bx, by, bz, bw = tq
    quat = np.array([qw * bx + qx * bw + qy * bz - qz * by, qw * by - qx * bz + qy * bw + qz * bx,
                     qw * bz + qx * by - qy * bx + qz * bw, qw * bw - qx * bx - qy * by - qz * bz])
    return state((x, y, zc), quat, vel)


def in_body(body, dx=0.0, dy=0.0, lift=0.0, **kw):
    return nozzle(L.BOOM_X + dx, L.BODIES[body] + dy, L.BOOM_TOP + lift, **kw)


def in_tray(side, dx=0.0, dy=0.0, **kw):
    return nozzle(L.TRAY_X + dx, L.TRAYS[side] + dy, L.TRAY_FLOOR_TOP, **kw)


def on_table(x, y, **kw):
    return nozzle(x, y, L.TOP, **kw)


def start_states():
    return {n: nozzle(x, y, L.start_bottom(n)) for n, (x, y) in L.STARTS.items()}


def solved_states():
    """The job done as the demo does it."""
    s = start_states()
    s["blue_L2"] = in_tray("left")
    s["blue_R3"] = in_tray("right", dx=0.020)
    s["lilac_R1"] = in_tray("right", dx=-0.020)
    s["yellow_left_b"] = in_body("L2")
    s["yellow_left_a"] = in_body("L1")
    s["yellow_right_b"] = in_body("R3")
    s["yellow_right_a"] = in_body("R1")
    return s


def left_arm_done(s=None):
    """What the left arm alone does in the demo: L2's blue to the tray, yellows to L2 and L1."""
    s = dict(s or start_states())
    s["blue_L2"] = in_tray("left")
    s["yellow_left_b"] = in_body("L2")
    s["yellow_left_a"] = in_body("L1")
    return s


def right_arm_done(s=None):
    s = dict(s or start_states())
    s["blue_R3"] = in_tray("right", dx=0.020)
    s["lilac_R1"] = in_tray("right", dx=-0.020)
    s["yellow_right_b"] = in_body("R3")
    s["yellow_right_a"] = in_body("R1")
    return s


def run(states, steps=FILL, tracker=None):
    t = tracker or Tracker()
    r = None
    for _ in range(steps):
        r = t.update(states)
    return r


class Layout(unittest.TestCase):
    def test_iso_colour_code(self):
        """ISO 10625: 01 orange, 015 green, 02 yellow, 025 lilac, 03 blue, 04 red."""
        self.assertEqual(L.SIZE_OF, {"orange": "01", "green": "015", "yellow": "02", "lilac": "025", "blue": "03",
                                     "red": "04"})
        self.assertEqual(L.TARGET_SIZE, "02")

    def test_starting_layout(self):
        start = {b: [n for n, w in L.START_PLACE.items() if w == b] for b in L.BODY_NAMES}
        self.assertEqual({b: [L.COLOUR[n] for n in ns] for b, ns in start.items()},
                         {"L3": ["yellow"], "L2": ["blue"], "L1": [], "R1": ["lilac"], "R2": ["yellow"],
                          "R3": ["blue"]})
        for side in ("left", "right"):
            spares = [n for n, w in L.START_PLACE.items() if w == f"pad_{side}"]
            self.assertEqual(len(spares), 4)
            self.assertEqual(sum(L.COLOUR[n] == "yellow" for n in spares), 2)  # exactly enough for its side
            self.assertFalse(any(L.COLOUR[n] in ("blue", "lilac") for n in spares))
        self.assertEqual(L.COLOUR["orange_right"], "orange")
        self.assertEqual(set(L.OFF_THE_BOOM), {n for n, w in L.START_PLACE.items()
                                               if w in L.BODIES and size_of(n) != "02"})
        self.assertEqual(set(L.KEEP), {w for n, w in L.START_PLACE.items() if w in L.BODIES and size_of(n) == "02"})

    def test_every_start_place_is_where_it_starts(self):
        for n, s in start_states().items():
            w = L.START_PLACE[n]
            expect = ("body", w) if w in L.BODIES else ("pad", w.split("_")[1])
            self.assertEqual(place_of(s), expect, n)

    def test_places_do_not_overlap_in_plan(self):
        """Body openings, tray insides and pads are disjoint seen from above, so a nozzle is in one place only."""
        rects = []
        for b, y in L.BODIES.items():
            h = L.BODY_OPEN / 2
            rects.append((b, L.BOOM_X - h, L.BOOM_X + h, y - h, y + h))
        for side, y in L.TRAYS.items():
            h = L.TRAY_IN / 2
            rects.append((f"tray_{side}", L.TRAY_X - h, L.TRAY_X + h, y - h, y + h))
        for side, y in L.PADS.items():
            hx, hy = L.PAD_SIZE[0] / 2, L.PAD_SIZE[1] / 2
            rects.append((f"pad_{side}", L.PAD_X - hx, L.PAD_X + hx, y - hy, y + hy))
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                apart = a[2] < b[1] or b[2] < a[1] or a[4] < b[3] or b[4] < a[3]
                self.assertTrue(apart, (a[0], b[0]))

    def test_start_pose_strip_is_empty(self):
        """Nothing where the arms' start pose hides the table in the first top-camera frame (|y| > 0.45 m at
        x < 0.2 m), and everything on the table."""
        for n, (x, y) in L.STARTS.items():
            self.assertLess(abs(y) + L.NOZZLE_R, 0.45, n)
        self.assertLess(L.BOOM_SIZE[1] / 2, 0.45)
        self.assertLess(abs(L.PADS["left"]) + L.PAD_SIZE[1] / 2, 0.45)

    def test_nozzle_fits_the_body_with_room(self):
        self.assertAlmostEqual(L.BODY_OPEN - 2 * L.NOZZLE_R, 0.010)  # 5 mm round a centred nozzle
        self.assertGreater(L.NOZZLE_H, L.BODY_WALL_H + 0.015)  # the pads can grip above the walls


class Places(unittest.TestCase):
    def test_body_edges(self):
        h = L.BODY_OPEN / 2
        self.assertEqual(place_of(in_body("L1", dx=h - 0.0005)), ("body", "L1"))
        self.assertEqual(place_of(in_body("L1", dy=-(h - 0.0005))), ("body", "L1"))
        self.assertEqual(place_of(in_body("L1", dx=h + 0.0005, lift=0.008)), ("elsewhere", ""))  # on a wall
        self.assertEqual(place_of(in_body("L1", lift=L.HOLD_ABOVE - 0.001)), ("body", "L1"))
        self.assertEqual(place_of(in_body("L1", lift=L.HOLD_ABOVE + 0.001)), ("elsewhere", ""))  # held up

    def test_tray_and_pad_edges(self):
        h = L.TRAY_IN / 2
        self.assertEqual(place_of(in_tray("right", dx=h - 0.001, tilt=90)), ("tray", "right"))
        self.assertEqual(place_of(on_table(L.TRAY_X + h + 0.012, L.TRAYS["right"])), ("elsewhere", ""))
        self.assertEqual(place_of(nozzle(L.PAD_X, L.PADS["left"], L.PAD_TOP, tilt=90)), ("pad", "left"))
        self.assertEqual(place_of(nozzle(L.PAD_X, L.PADS["left"], L.PAD_TOP + L.HOLD_ABOVE + 0.001)),
                         ("elsewhere", ""))
        held_up = nozzle(L.TRAY_X, L.TRAYS["left"], L.TRAY_FLOOR_TOP + L.HOLD_ABOVE + 0.001)
        self.assertEqual(place_of(held_up), ("elsewhere", ""))  # still held over the tray
        stacked = nozzle(L.TRAY_X, L.TRAYS["left"], L.TRAY_FLOOR_TOP + L.NOZZLE_H)
        self.assertEqual(place_of(stacked), ("tray", "left"))  # standing on another in the tray

    def test_off_the_table(self):
        self.assertEqual(place_of(nozzle(0.30, 0.70, L.TOP - 0.20)), ("off_table", ""))
        self.assertEqual(place_of(nozzle(0.30, 0.10, L.TOP - 0.05)), ("off_table", ""))  # centre below the top
        self.assertEqual(place_of(nozzle(0.95, 0.0, L.TOP)), ("off_table", ""))  # past the far edge

    def test_seated(self):
        self.assertTrue(seated(in_body("R1")))
        self.assertTrue(seated(in_body("R1", lift=L.FIT_BOTTOM - 0.0005)))
        self.assertFalse(seated(in_body("R1", lift=L.FIT_BOTTOM + 0.0005)))
        self.assertTrue(seated(in_body("R1", tilt=L.FIT_TILT_DEG - 0.5)))
        self.assertFalse(seated(in_body("R1", tilt=L.FIT_TILT_DEG + 0.5)))
        self.assertTrue(seated(in_body("R1", tilt=180)))  # upside down: the same cylinder
        self.assertFalse(seated(in_body("R1", lift=-0.005)))  # sunk into the boom (a penetrating contact)

    def test_lowest_and_tilt(self):
        s = nozzle(0.3, 0.0, L.TOP, tilt=90)
        self.assertAlmostEqual(lowest(s), L.TOP, places=6)
        self.assertAlmostEqual(tilt_deg(s), 90.0, places=4)
        s = nozzle(0.3, 0.0, L.TOP, tilt=25, lean_to=40)
        self.assertAlmostEqual(lowest(s), L.TOP, places=6)
        self.assertAlmostEqual(tilt_deg(s), 25.0, places=4)
        self.assertAlmostEqual(float(s.low[2]), L.TOP, places=6)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = run(start_states())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["fitted_02"], ["L3", "R2"])
        self.assertEqual(r.details["wrong_size_in"], ["L2", "R1", "R3"])

    def test_solved_scores_one(self):
        r = run(solved_states())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["done"], ["L2", "L1", "R1", "R3"])

    def test_every_tier_is_reachable(self):
        s = start_states()
        s["yellow_left_a"] = in_body("L1")
        self.assertEqual(run(s).score, 0.25)  # L1 filled, nothing else
        self.assertEqual(run(left_arm_done()).score, 0.5)  # left half done, R1's lilac and R3's blue fitted
        self.assertEqual(run(right_arm_done()).score, 0.5)  # right half done, L2's blue fitted
        s = solved_states()
        s["yellow_right_a"] = start_states()["yellow_right_a"]  # R1 emptied but not refilled
        self.assertEqual(run(s).score, 0.75)
        s = solved_states()
        s["blue_L2"] = on_table(0.25, 0.02, tilt=90)  # every body right, a removed nozzle left on the table
        self.assertEqual(run(s).score, 0.75)
        self.assertEqual(run(solved_states()).score, 1.0)

    def test_each_body_alone_is_a_quarter(self):
        for b, n in (("L2", "yellow_left_b"), ("L1", "yellow_left_a"), ("R1", "yellow_right_a"),
                     ("R3", "yellow_right_b")):
            s = start_states()
            if b == "L2":
                s["blue_L2"] = in_tray("left")
            if b == "R1":
                s["lilac_R1"] = in_tray("right")
            if b == "R3":
                s["blue_R3"] = in_tray("right")
            s[n] = in_body(b)
            self.assertEqual(run(s).score, 0.25, b)

    def test_wrong_size_caps_at_half(self):
        s = solved_states()
        s["lilac_R1"], s["yellow_right_a"] = in_body("R1"), start_states()["yellow_right_a"]
        r = run(s)  # three done (0.75 of points) but the lilac still in R1
        self.assertEqual(r.details["points"], 0.75)
        self.assertEqual(r.score, 0.5)

    def test_orange_for_yellow(self):
        """The 01 (orange) beside the yellows on the right pad, fitted for an 02: that body earns nothing and the
        boom is still mixed."""
        s = solved_states()
        s["orange_right"], s["yellow_right_a"] = in_body("R1"), start_states()["yellow_right_a"]
        r = run(s)
        self.assertEqual(r.details["wrong_size_in"], ["R1"])
        self.assertEqual(r.score, 0.5)

    def test_every_other_size_counts_as_wrong(self):
        for n in ("red_left", "green_left", "orange_right", "red_right", "blue_L2", "lilac_R1"):
            s = solved_states()
            s[n], s["yellow_left_a"] = in_body("L1"), start_states()["yellow_left_a"]
            self.assertEqual(run(s).score, 0.5, n)

    def test_cap_rules_combine(self):
        s = left_arm_done()
        s["green_left"] = on_table(0.25, 0.30)  # loose and mixed: the lower cap
        self.assertEqual(run(s).score, 0.5)
        s = start_states()
        s["yellow_left_a"] = in_body("L1")
        s["red_left"] = on_table(0.25, 0.30)
        self.assertEqual(run(s).score, 0.25)  # points below both caps


class Zero(unittest.TestCase):
    def test_nozzle_off_the_table(self):
        s = solved_states()
        s["red_left"] = nozzle(0.30, 0.70, L.TOP - 0.40)
        self.assertEqual(run(s).score, 0.0)

    def test_two_in_one_body(self):
        s = solved_states()
        s["blue_R3"], s["yellow_right_b"] = in_body("R3"), in_body("R3", lift=L.NOZZLE_H)  # yellow on the blue
        r = run(s)
        self.assertEqual(r.details["two_in_one"], ["R3"])
        self.assertEqual(r.score, 0.0)
        s = solved_states()
        s["red_left"] = in_body("L2", dx=0.004, lift=L.NOZZLE_H, tilt=20)  # a spare leaning on a fitted 02
        self.assertEqual(run(s).score, 0.0)

    def test_kept_02_taken_away(self):
        s = solved_states()
        s["yellow_L3"] = in_tray("left", dx=0.025)
        r = run(s)
        self.assertEqual(r.details["keep_lost"], ["L3"])
        self.assertEqual(r.score, 0.0)
        s = solved_states()
        s["yellow_R2"] = in_body("R2", tilt=25, dy=0.004, lift=0.002)  # knocked over in its body
        self.assertEqual(run(s).score, 0.0)

    def test_kept_02_replaced_by_another_02_counts(self):
        """The scorer reads the end state: an 02 taken out and another put in its place is the same boom."""
        s = solved_states()
        s["yellow_L3"], s["yellow_left_a"] = in_body("L1"), in_body("L3")
        self.assertEqual(run(s).score, 1.0)

    def test_kept_02_used_elsewhere_leaves_its_body_empty(self):
        """Exploit: move L3's yellow to L2 instead of fetching a spare."""
        s = start_states()
        s["blue_L2"], s["yellow_L3"] = in_tray("left"), in_body("L2")
        self.assertEqual(run(s).score, 0.0)


class Loose(unittest.TestCase):
    def test_removed_nozzle_left_on_the_boom(self):
        s = solved_states()
        s["blue_L2"] = nozzle(L.BOOM_X, 0.125, L.BOOM_TOP, tilt=90, lean_to=90)  # lying on the boom between bodies
        self.assertEqual(run(s).details["loose"], ["blue_L2"])
        self.assertEqual(run(s).score, 0.75)

    def test_removed_nozzle_still_held_over_its_tray(self):
        s = solved_states()
        s["blue_L2"] = nozzle(L.TRAY_X, L.TRAYS["left"], L.TRAY_FLOOR_TOP + 0.060)
        self.assertEqual(run(s).score, 0.75)

    def test_last_nozzle_still_held_up(self):
        s = solved_states()
        s["yellow_right_a"] = in_body("R1", lift=0.060)
        self.assertEqual(run(s).score, 0.75)  # R1 empty and a nozzle in the air
        s["yellow_right_a"] = in_body("R1", lift=0.010)  # held low over its body, not standing in it
        self.assertEqual(run(s).score, 0.75)

    def test_removed_nozzle_back_among_the_spares(self):
        for n in L.OFF_THE_BOOM:
            s = solved_states()
            s[n] = nozzle(L.PAD_X, L.PADS["left"], L.PAD_TOP)
            r = run(s)
            self.assertEqual(r.details["removed_on_pad"], [n])
            self.assertEqual(r.score, 0.75, n)

    def test_spares_may_stay_or_go_to_a_tray(self):
        s = solved_states()
        s["red_left"] = in_tray("left", dx=0.030, dy=0.030)
        s["orange_right"] = in_tray("left", dx=-0.030, dy=-0.030, tilt=90)
        self.assertEqual(run(s).score, 1.0)

    def test_either_tray_takes_removed_nozzles(self):
        s = solved_states()
        s["blue_L2"] = in_tray("right", dy=0.030)
        s["lilac_R1"] = in_tray("left", tilt=90, lean_to=30)
        self.assertEqual(run(s).score, 1.0)

    def test_spare_on_the_table_caps(self):
        s = solved_states()
        s["green_left"] = on_table(0.10, 0.20)
        self.assertEqual(run(s).score, 0.75)


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_rest(self):
        t = Tracker()
        s = solved_states()
        for i in range(int(L.CONTROL_HZ) // 2):  # 10 steps: 0.45 s of history
            r = t.update(s)
            self.assertEqual(r.score, 0.0, i)  # L3 and R2 not yet known to be still
        r = t.update(s)
        self.assertEqual(r.score, 1.0)

    def test_moving_nozzle_is_not_fitted(self):
        t = Tracker()
        s = solved_states()
        run(s, tracker=t)
        for k in range(12):  # R1's yellow sliding 0.5 mm per step in its body
            s = dict(s)
            s["yellow_right_a"] = in_body("R1", dx=-0.003 + 0.0005 * k)
            r = t.update(s)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        for _ in range(FILL):
            r = t.update(s)
        self.assertEqual(r.score, 1.0)

    def test_tiny_jitter_is_still(self):
        t = Tracker()
        s = solved_states()
        rng = np.random.default_rng(3)
        for _ in range(30):
            s = dict(s)
            d = rng.uniform(-0.00005, 0.00005, size=2)
            s["yellow_left_a"] = in_body("L1", dx=float(d[0]), dy=float(d[1]))
            r = t.update(s)
        self.assertEqual(r.score, 1.0)

    def test_reported_velocity_is_not_used(self):
        """Gripped parts' reported velocity chatters: only the path counts."""
        s = solved_states()
        s["yellow_right_a"] = in_body("R1", vel=(0.3, -0.2, 0.1))
        self.assertEqual(run(s).score, 1.0)

    def test_score_is_idempotent(self):
        t = Tracker()
        s = solved_states()
        run(s, tracker=t)
        a, b = t.result(s).score, t.result(s).score
        self.assertEqual(a, b)
        self.assertEqual(t.steps, FILL)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker()
        good = solved_states()
        run(good, tracker=t)
        before = [list(map(tuple, t.paths[n])) for n in L.NOZZLE_NAMES]
        for bad in (np.nan, np.inf, -np.inf):
            s = dict(good)
            s["red_right"] = SimpleNamespace(**{**vars(good["red_right"]), "pos": np.array([bad, 0.0, 0.8])})
            r = t.update(s)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        s = dict(good)
        s["blue_L2"] = SimpleNamespace(**{**vars(good["blue_L2"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(t.update(s).score, 0.0)
        s = dict(good)
        s["green_left"] = SimpleNamespace(**{**vars(good["green_left"]), "lin_vel": np.array([0.0, np.nan, 0.0])})
        self.assertEqual(t.update(s).score, 0.0)
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual([list(map(tuple, t.paths[n])) for n in L.NOZZLE_NAMES], before)
        self.assertEqual(t.update(good).score, 1.0)  # carries on as before

    def test_missing_or_malformed_states(self):
        good = solved_states()
        s = dict(good)
        del s["yellow_L3"]
        self.assertEqual(run(s).score, 0.0)
        self.assertEqual(Tracker().update({}).score, 0.0)
        self.assertEqual(Tracker().update(None).score, 0.0)
        s = dict(good)
        s["red_left"] = SimpleNamespace(pos=[0.1, 0.2], quat_xyzw=[0, 0, 0, 1], lin_vel=[0, 0, 0],
                                        ang_vel=[0, 0, 0], low=[0, 0, 0], high=[0, 0, 0])
        self.assertEqual(run(s).score, 0.0)
        self.assertFalse(valid(object()))

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker()
        for _ in range(200):
            s = {}
            for n in L.NOZZLE_NAMES:
                q = rng.normal(size=4)
                s[n] = state(rng.uniform([-0.3, -0.8, 0.0], [1.0, 0.8, 1.2]), q)
            r = t.update(s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        run(solved_states(), tracker=t)
        self.assertEqual(t.last.score, 1.0)
        r = t.update(start_states())
        self.assertEqual(t.episodes, 1)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = start_states(), solved_states()
        run(b, tracker=t)
        for k in range(1, 41):  # every moved nozzle walked back to its mark over 2 s, a few mm per step
            f = k / 40
            s = {n: state(np.array(b[n].pos) * (1 - f) + np.array(a[n].pos) * f) for n in L.NOZZLE_NAMES}
            t.update(s)
        self.assertEqual(t.episodes, 0)
        self.assertEqual(t.steps, FILL + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        run(start_states(), tracker=t)
        s = start_states()
        s["blue_L2"] = on_table(0.25, 0.40)  # flung 0.2 m in a step, not back to the start
        t.update(s)
        self.assertEqual(t.episodes, 0)


class Exploits(unittest.TestCase):
    def test_stack_02s_on_the_wrong_ones(self):
        """Put the yellows on top of the blue and lilac instead of taking those off: two in a body."""
        s = start_states()
        s["yellow_left_b"] = in_body("L2", lift=L.NOZZLE_H)
        s["yellow_left_a"] = in_body("L1")
        s["yellow_right_a"] = in_body("R1", lift=L.NOZZLE_H)
        s["yellow_right_b"] = in_body("R3", lift=L.NOZZLE_H)
        self.assertEqual(run(s).score, 0.0)

    def test_knock_the_wrong_ones_onto_the_table(self):
        s = solved_states()
        for n, x, y in (("blue_L2", 0.26, 0.15), ("lilac_R1", 0.26, -0.03), ("blue_R3", 0.40, -0.25)):
            s[n] = on_table(x, y, tilt=90)
        self.assertEqual(run(s).score, 0.75)

    def test_knock_the_wrong_ones_off_the_table(self):
        s = solved_states()
        s["blue_R3"] = nozzle(0.40, -0.70, L.TOP - 0.70, tilt=90)
        self.assertEqual(run(s).score, 0.0)

    def test_wrong_one_tipped_over_in_its_body(self):
        """A blue pushed over inside L2 (leaning on the wall) is still in L2: the boom is still mixed."""
        s = right_arm_done()
        s["yellow_left_a"] = in_body("L1")
        s["blue_L2"] = in_body("L2", dy=0.004, lift=0.001, tilt=30, lean_to=90)
        r = run(s)
        self.assertEqual(r.details["holds"]["L2"], ["blue_L2"])
        self.assertEqual(r.score, 0.5)

    def test_02s_lying_in_the_bodies_earn_nothing(self):
        s = solved_states()
        s["yellow_left_a"] = in_body("L1", dy=0.003, lift=0.002, tilt=35, lean_to=90)
        self.assertEqual(run(s).score, 0.75)  # L1 holds an 02 but not standing: no point, no cap

    def test_hold_every_wrong_one_up_in_the_air(self):
        s = solved_states()
        s["blue_L2"] = in_body("L2", lift=0.10)  # over L2 but 10 cm up: not in it, and not put away
        s["yellow_left_b"] = start_states()["yellow_left_b"]
        self.assertEqual(run(s).score, 0.75)


class Hooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("task", None)
        stub = HERE.parent.parent / "tools" / "offline" / "yam_stub"
        if stub.is_dir() and str(stub) not in sys.path:
            sys.path.insert(0, str(stub))
        try:
            cls.task = importlib.import_module("task")
        except ImportError as e:  # pragma: no cover - only without the stub
            raise unittest.SkipTest(f"task.py needs yam_arena: {e}")

    def setUp(self):
        self.task._TRACKER.reset()

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted(L.NOZZLE_NAMES))
        for o in loose:
            self.assertAlmostEqual(o.radius, L.NOZZLE_R)
            self.assertAlmostEqual(o.height, L.NOZZLE_H)
            self.assertLessEqual(o.mass, 0.3)
            self.assertEqual(o.color, L.RGB[L.COLOUR[o.name]])
        self.assertEqual(len(names), 1 + 24 + 10 + 2 + 13)

    def test_hooks_on_the_start_and_the_end(self):
        objs = {o.name: None for o in self.task.OBJECTS if o.mass is None}
        for _ in range(FILL):
            ok = self.task.success({**objs, **start_states()})
        self.assertFalse(ok)
        self.assertEqual(self.task.score(start_states()), 0.0)
        self.task._TRACKER.reset()
        for _ in range(FILL):
            ok = self.task.success({**objs, **solved_states()})
        self.assertTrue(ok)
        self.assertEqual(self.task.score(solved_states()), 1.0)

    def test_trace(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                for _ in range(3):
                    self.task.success(start_states())
            finally:
                del os.environ["TASK_TRACE"]
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 3)
        for k in ("step", "score", "success", "points", "cap", "holds", "places", "tilt_deg", "bottom_mm",
                  "path_mm", "wrong_size_in", "loose", "blue_L2_pos"):
            self.assertIn(k, rows[-1])
        self.assertEqual(rows[-1]["places"]["blue_L2"], "body_L2")

    def test_trace_never_stops_a_trial(self):
        os.environ["TASK_TRACE"] = "/nonexistent-dir/trace.jsonl"
        try:
            self.assertFalse(self.task.success(start_states()))
        finally:
            del os.environ["TASK_TRACE"]

    def test_demo_switches(self):
        t = self.task
        self.assertEqual(set(t.demo()), {"left", "right"})
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(t.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(t.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_demo_plan(self):
        """Finger widths, heights over the walls, the place points and the timing between the arms."""
        t = self.task
        plans = t.demo()
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.NOZZLE_R - 0.008)  # 14 mm on the 22 mm nozzle
        self.assertAlmostEqual(-2 * t.APPROACH, 0.042)
        pad_tips = L.BOOM_TOP + t.GRIP_UP - 0.013
        self.assertGreaterEqual(pad_tips - L.WALL_TOP, 0.007 - 1e-9)  # 7 mm over the walls when gripping in a body
        # open pads (outer faces) stay clear of the next spare on a pad (55 mm along x, 60 mm along y)
        outer = -t.APPROACH + 0.008
        self.assertGreater(min(np.diff(L.SPARE_YS)) - L.NOZZLE_R - outer, 0.015)
        # carried nozzles clear the fitted ones' tops
        self.assertGreater(t.CARRY_Z - t.GRIP_UP - (L.BOOM_TOP + L.NOZZLE_H), 0.04)
        for arm, wps in plans.items():
            for w in wps:
                self.assertTrue(-0.04695 - 1e-9 <= w.finger <= 0.0)
                self.assertGreaterEqual(w.duration, 0.3)
                if w.finger == t.GRIP and "grip" in w.label:
                    self.assertGreaterEqual(w.duration, 1.5)
                if "let go" in w.label:
                    self.assertGreaterEqual(w.duration, 1.5)
        # the tray spots: the open pads (13.5 mm either side along x, outer faces 29 mm out along y) stay at least
        # 10 mm inside the tray's walls, and two nozzles in one tray stand 40 mm apart
        for n, (x, y) in t.TRAY_SPOTS.items():
            side = "left" if y > 0 else "right"
            self.assertLessEqual(abs(x - L.TRAY_X) + 0.0135, L.TRAY_IN / 2 - 0.010)
            self.assertLessEqual(abs(y - L.TRAYS[side]) + outer, L.TRAY_IN / 2 - 0.010)
        self.assertAlmostEqual(t.TRAY_SPOTS["blue_R3"][0] - t.TRAY_SPOTS["lilac_R1"][0], 0.040)
        a = sum(w.duration for w in plans["left"])
        b = sum(w.duration for w in plans["right"])
        self.assertLess(max(a, b) * L.CONTROL_HZ, 2400 * 0.65)  # the run limit is 2,400 steps

    def test_middle_is_never_shared(self):
        """Whenever both grippers are low (TCP under 6 cm above the table), they are at least 0.20 m apart across
        y."""
        plans = self.task.demo()

        def at(wps, time):
            t0 = 0.0
            prev = None
            for w in wps:
                if time <= t0 + w.duration:
                    if prev is None:
                        return np.array(w.pos)
                    f = (time - t0) / w.duration
                    return np.array(prev) * (1 - f) + np.array(w.pos) * f
                t0 += w.duration
                prev = w.pos
            return np.array(wps[-1].pos)

        end = max(sum(w.duration for w in p) for p in plans.values())
        for k in range(int(end * 20)):
            tt = k / 20
            pl, pr = at(plans["left"], tt), at(plans["right"], tt)
            low_l, low_r = pl[2] < L.TOP + 0.06, pr[2] < L.TOP + 0.06
            if low_l and low_r:
                self.assertGreaterEqual(pl[1] - pr[1], 0.20, tt)

    def test_demo_end_states_score_one(self):
        """Where the demo leaves each nozzle (the place points with the drop it ends with) scores 1.0."""
        t = self.task
        s = start_states()
        for n, (x, y) in t.TRAY_SPOTS.items():
            s[n] = nozzle(x, y, L.TRAY_FLOOR_TOP)
        for n, b in (("yellow_left_b", "L2"), ("yellow_left_a", "L1"), ("yellow_right_b", "R3"),
                     ("yellow_right_a", "R1")):
            s[n] = in_body(b)
        self.assertEqual(run(s).score, 1.0)
        labels = [w.label for w in t.demo()["left"] + t.demo()["right"]]
        for n in ("blue_L2", "yellow_left_b", "yellow_left_a", "blue_R3", "lilac_R1", "yellow_right_b",
                  "yellow_right_a"):
            self.assertIn(f"grip {n}", labels)
        for n in ("red_left", "green_left", "orange_right", "red_right", "yellow_L3", "yellow_R2"):
            self.assertNotIn(f"grip {n}", labels)


class TaskJson(unittest.TestCase):
    """task.json and the README say what layout.py and scoring.py do."""

    @classmethod
    def setUpClass(cls):
        with open(HERE / "task.json") as f:
            cls.spec = json.load(f)
        cls.readme = (HERE / "README.md").read_text()
        sys.modules.pop("task", None)
        stub = HERE.parent.parent / "tools" / "offline" / "yam_stub"
        if stub.is_dir() and str(stub) not in sys.path:
            sys.path.insert(0, str(stub))
        cls.task = importlib.import_module("task")

    def test_instruction_is_the_same_everywhere(self):
        self.assertEqual(self.spec["instruction"], self.task.INSTRUCTION)
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in self.readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_instruction_never_names_the_colour(self):
        text = self.task.INSTRUCTION.lower()
        for c in L.SIZE_OF:
            self.assertNotIn(c, text)
        self.assertIn("iso colour code", text)

    def test_instruction_sets_no_unscored_rule(self):
        """The scorer reads only the end state, so the instruction asks only for one: the fitted 02s may stay (an
        order to leave them untouched could not be checked), and only other sizes must end in a tray."""
        text = self.task.INSTRUCTION
        self.assertNotIn("where they are", text)
        self.assertIn("already fitted can stay", text)
        self.assertIn("every nozzle of another size that you take off the boom", text)

    def test_fidelity_states_the_mock_up(self):
        fid = self.spec["fidelity"]
        for s in ("125 mm apart", "hang under the boom, 500 mm apart", "about 15 mm across", "No liquid"):
            self.assertIn(s, fid, s)

    def test_setup_numbers(self):
        setup = self.spec["setup"]
        for s in ("680", "44 x 680 x 25 mm", "32 x 32 mm", "8 mm tall", "5 mm thick", "125 mm apart",
                  "x 32.0 cm", "y +31.25, +18.75, +6.25, -6.25, -18.75 and -31.25 cm", "22 mm across",
                  "32 mm tall", "17 g", "100 x 110 x 5 mm", "x 17.75 cm, y +26.5 and -26.5 cm",
                  "x 20.0 cm, y +11.0 and -11.0 cm", "90 x 90 mm", "x 15.0 and 20.5 cm", "y 23.5 and 29.5 cm"):
            self.assertIn(s, setup, s)

    def test_setup_lists_every_nozzle(self):
        setup = self.spec["setup"]
        self.assertIn("L3 yellow, L2 blue, L1 empty, R1 lilac, R2 yellow, R3 blue", setup)
        self.assertIn("left pad: yellow at x 15.0, y 23.5; red at x 15.0, y 29.5; green at x 20.5, y 23.5; "
                      "yellow at x 20.5, y 29.5", setup)
        self.assertIn("right pad: orange at x 15.0, y -23.5; yellow at x 15.0, y -29.5; yellow at x 20.5, "
                      "y -23.5; red at x 20.5, y -29.5", setup)
        for n, colour, x, y, where in L.NOZZLES:
            if where.startswith("pad"):
                self.assertIn(f"{colour} at x {x * 100:.1f}, y {y * 100:.1f}", setup)

    def test_scoring_thresholds(self):
        sc = self.spec["scoring"]
        for s in ("within 10 degrees", "within 3 mm", "2 mm or less", "0.5 s", "less than 40 mm above", "0.25",
                  "at most 0.5", "at most 0.75", "2.5 s", "0, 0.25, 0.5, 0.75 and 1.0",
                  "01 orange, 015 green, 02 yellow, 025 lilac, 03 blue, 04 red"):
            self.assertIn(s, sc, s)
        self.assertEqual(L.FIT_TILT_DEG, 10.0)
        self.assertEqual(L.FIT_BOTTOM, 0.003)
        self.assertEqual(L.STILL_PATH, 0.002)
        self.assertEqual(L.HOLD_ABOVE, 0.040)
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)

    def test_run_limit(self):
        self.assertTrue(self.spec["run_limit"].startswith("2,400 steps"))

    def test_text_has_no_internal_references(self):
        """Blind review: the submitted text names no tool path, pipeline file or other task of ours."""
        text = self.readme + " ".join(str(v) for v in self.spec.values())
        for bad in ("tools/", "pipeline/", "check_task", "heatsink", "T-70", "yamkin"):
            self.assertNotIn(bad, text)
        self.assertNotRegex(text, r"(?i)\bour\b")


if __name__ == "__main__":
    unittest.main()
