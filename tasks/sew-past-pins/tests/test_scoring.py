"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic feeds, pin states and arm
snapshots (fed through rigstate.offline_feed where the task file is involved), no simulator.
    python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from offline_model import OfflineWork  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, handle_box, holds_handle, holds_pin, in_dish, in_work, let_go, needle_gap, off_table, over_plate,
    tilt_deg, under_head,
)

HZ = L.CONTROL_HZ
TIPS = 0.013  # the finger tips below the tool point
GRIP_TCP_UP = 0.009  # the demo's tool point is 9 mm above a gripped pin's centre
OPEN, ON_HANDLE, ON_PIN = 1.0, 0.32, 0.19  # rigstate openings: open, closed on the 30 mm handle, on an 18 mm pin
FAR_RIGHT = np.array([0.20, -0.25, L.TOP + 0.15])  # the right arm out of the way
UP = np.array([0.0, 0.0, 1.0])
S45 = math.sin(math.pi / 4)


def arm(tcp, opening):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


def handle_tcp(feed):
    """The left tool point gripping the handle (the finger tips 10 mm above the plate, inside the handle)."""
    return np.array([L.X_SEAM, L.HANDLE_Y - feed, L.TOP + 0.045])


def pin_state(centre, lying=False, vel=(0.0, 0.0, 0.0), quat=None):
    """An upright pin (or one lying along y) at ``centre``."""
    c = np.array(centre, float)
    if lying:
        half, q = np.array([L.PIN_R, L.PIN_H / 2, L.PIN_R]), np.array([S45, 0.0, 0.0, S45])
    else:
        half, q = np.array([L.PIN_R, L.PIN_R, L.PIN_H / 2]), np.array([0.0, 0.0, 0.0, 1.0])
    if quat is not None:
        q = np.array(quat, float)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=c - half,
                           high=c + half)


def spot(name):
    """The demo's place for a pin in the dish (standing on the floor)."""
    return np.array([L.DISH_X + L.DISH_SPOTS[name], L.DISH_Y, L.TOP + L.DISH_FLOOR_T + L.PIN_H / 2])


class Trial:
    """A trial, one control step per frame: the feed, each pin (riding in the work or free) and both arms."""

    def __init__(self):
        self.feed = 0.0
        self.base = {n: np.array(L.PIN_START[n], float) for n in L.PINS}  # riding pins: centre at feed 0
        self.free: dict[str, np.ndarray] = {}  # pins off the work: their centres
        self.lying: set[str] = set()
        self.left, self.lo = handle_tcp(0.0), ON_HANDLE
        self.right, self.ro = FAR_RIGHT.copy(), OPEN
        self.left_on_handle = True
        self.frames: list = []
        self._last = None

    def centre(self, n):
        return self.free[n] if n in self.free else self.base[n] - np.array([0.0, self.feed, 0.0])

    def step(self, n=1):
        for _ in range(n):
            now = {p: self.centre(p).copy() for p in L.PINS}
            vel = {p: (now[p] - self._last[p]) * HZ if self._last else np.zeros(3) for p in L.PINS}
            self._last = now
            pins = {p: pin_state(now[p], p in self.lying, vel[p]) for p in L.PINS}
            self.frames.append((self.feed, pins, {"left": arm(self.left, self.lo), "right": arm(self.right, self.ro)}))
        return self

    def feed_to(self, target, n=40, hold=True):
        """Slide the work (the left gripper on the handle when ``hold``)."""
        start = self.feed
        for i in range(1, n + 1):
            self.feed = start + (target - start) * i / n
            if hold and self.left_on_handle:
                self.left = handle_tcp(self.feed)
            self.step()
        return self

    def carry(self, name, target, n=20, opening=ON_PIN):
        """The right gripper carries a pin (gripped 9 mm above its centre) to ``target`` (its centre)."""
        if name not in self.free:
            self.free[name] = self.centre(name).copy()
        start = self.free[name].copy()
        self.ro = opening
        for i in range(1, n + 1):
            self.free[name] = start + (np.array(target, float) - start) * i / n
            self.right = self.free[name] + np.array([0.0, 0.0, GRIP_TCP_UP])
            self.step()
        return self

    def pull(self, name, n=20):
        """Grip a pin where it is and lift it straight up 60 mm."""
        self.right, self.ro = self.centre(name) + np.array([0.0, 0.0, GRIP_TCP_UP]), OPEN
        self.step(3)
        self.ro = ON_PIN
        self.step(3)
        return self.carry(name, self.centre(name) + 0.060 * UP, n)

    def to_dish(self, name, n=20):
        """Carry a lifted pin along x to its place's x (clear of the needle head), along y over its place in the
        dish, lower it onto the floor, let go and rise."""
        place = spot(name)
        c = self.centre(name)
        self.carry(name, (place[0], c[1], place[2] + 0.060), n)
        self.carry(name, place + 0.060 * UP, n)
        self.carry(name, place, 10)
        self.step(2)
        self.ro = OPEN
        self.step(3)
        self.right = place + np.array([0.0, 0.0, 0.08])
        return self.step(3)

    def let_go_left(self):
        self.lo = OPEN
        self.step(3)
        self.left = self.left + np.array([0.0, 0.0, 0.07])
        self.left_on_handle = False
        return self.step(3)


def full_trial(pull_d=(0.05, 0.05, 0.05), feed_end=0.178, release=True, dish=True):
    """The demo's procedure: each pin pulled with d at ``pull_d`` (fed there first), put in the dish, the seam fed to
    ``feed_end``, both grippers let go, then still."""
    tr = Trial().step(5)
    for name, d in zip(L.PINS, pull_d):
        need = L.PIN_D0[name] - d
        if need > tr.feed:
            tr.feed_to(need)
        tr.pull(name)
        if dish:
            tr.to_dish(name)
        else:
            tr.carry(name, np.array([0.20, -0.20, L.TOP + L.PIN_H / 2]), 20)  # stood on the table, not in the dish
            tr.ro = OPEN
            tr.step(2)
            tr.right = FAR_RIGHT.copy()
            tr.step()
    tr.feed_to(feed_end, 60)
    tr.step(5)
    if release:
        tr.let_go_left()
    tr.right, tr.ro = FAR_RIGHT.copy(), OPEN if release else tr.ro
    return tr.step(25)


def run(frames, tracker=None):
    t = tracker or Tracker(HZ)
    r = None
    for feed, pins, arms in frames:
        r = t.update(feed, pins, arms)
    return r, t


# -- geometry -----------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_starting_pins_are_in_the_work(self):
        for n in L.PINS:
            s = pin_state(L.PIN_START[n])
            self.assertTrue(in_work(s, 0.0))
            self.assertAlmostEqual(float(s.low[2]), L.TOP + L.POCKET_FLOOR)
            self.assertAlmostEqual(float(s.pos[1]) - L.Y_NEEDLE, L.PIN_D0[n])

    def test_in_work_lift_boundary(self):
        """In the plate while the lowest point is less than 3 mm above the plate's top (22 mm above the table)."""
        c = np.array(L.PIN_START["pin_1"])
        bottom = float(c[2]) - L.PIN_H / 2
        just_in = c + (L.TOP + 0.022 + 0.0025 - bottom) * UP
        just_out = c + (L.TOP + 0.022 + 0.0035 - bottom) * UP
        self.assertTrue(in_work(pin_state(just_in), 0.0))
        self.assertFalse(in_work(pin_state(just_out), 0.0))
        side = c + np.array([L.PLATE_W / 2 + 0.001, 0.0, 0.0])  # off the plate's side
        self.assertFalse(in_work(pin_state(side), 0.0))

    def test_plate_moves_with_the_feed(self):
        p = (L.X_SEAM, L.PLATE_Y1 - 0.01, 0.0)
        self.assertTrue(over_plate(p, 0.0))
        self.assertFalse(over_plate(p, 0.02))
        lo, hi = handle_box(0.10)
        self.assertAlmostEqual((lo[1] + hi[1]) / 2, L.HANDLE_Y - 0.10)
        self.assertAlmostEqual(hi[2] - lo[2], L.HANDLE[2])

    def test_needle_gap_and_head(self):
        z = L.TOP + L.POCKET_FLOOR + L.PIN_H / 2
        s = pin_state((L.X_SEAM, L.Y_NEEDLE + 0.015, z))
        self.assertAlmostEqual(needle_gap(s), 0.006)
        self.assertEqual(needle_gap(pin_state((L.X_SEAM, L.Y_NEEDLE + 0.004, z))), 0.0)  # straddles the line
        self.assertAlmostEqual(needle_gap(pin_state((L.X_SEAM, L.Y_NEEDLE + 0.030, z), lying=True)), 0.010)
        self.assertTrue(under_head(s))
        self.assertFalse(under_head(pin_state((L.X_SEAM + L.HEAD_HALF + 0.001, L.Y_NEEDLE, z))))
        high = L.TOP + L.BAR_Z1 + L.PIN_H / 2
        self.assertFalse(under_head(pin_state((L.X_SEAM, L.Y_NEEDLE, high))))
        self.assertTrue(under_head(pin_state((L.X_SEAM, L.Y_NEEDLE, high - 0.001))))

    def test_in_dish(self):
        for n in L.PINS:
            self.assertTrue(in_dish(pin_state(spot(n))))
        c = spot("pin_2")
        self.assertTrue(in_dish(pin_state(c - np.array([0, 0, L.PIN_H / 2 - L.PIN_R]), lying=True)))  # lying in it
        self.assertFalse(in_dish(pin_state(c + np.array([L.DISH_HALF[0] + 0.001 + L.DISH_SPOTS["pin_3"], 0, 0]))))
        self.assertFalse(in_dish(pin_state(c + np.array([0, L.DISH_HALF[1] + 0.001, 0]))))
        self.assertFalse(in_dish(pin_state(c + np.array([0, 0, L.DISH_RIM]))))  # held above the rim
        self.assertFalse(in_dish(pin_state(c, vel=(0.0, 0.0, -0.05))))  # still falling
        spinning = pin_state(c)
        spinning.ang_vel = np.array([0.0, 0.0, 0.6])
        self.assertFalse(in_dish(spinning))  # still means under 1 cm/s and 0.5 rad/s
        spinning.ang_vel = np.array([0.0, 0.0, 0.4])
        self.assertTrue(in_dish(spinning))

    def test_off_table(self):
        self.assertFalse(off_table(pin_state((0.2, 0.0, L.TOP + L.PIN_H / 2))))
        self.assertFalse(off_table(pin_state((0.2, 0.0, L.TOP + L.PIN_H / 2 - 0.0095))))
        self.assertTrue(off_table(pin_state((0.2, 0.0, L.TOP + L.PIN_H / 2 - 0.0105))))  # 10 mm below the top
        # held in the air beyond the table's edge: not fallen
        self.assertFalse(off_table(pin_state((0.30, L.TABLE_Y[0] - 0.01, L.TOP + 0.10))))

    def test_holds_handle_boundaries(self):
        tcp = handle_tcp(0.05)
        self.assertTrue(holds_handle(arm(tcp, ON_HANDLE), 0.05))
        self.assertTrue(holds_handle(arm(tcp, 0.59), 0.05))  # less open than 0.6
        self.assertFalse(holds_handle(arm(tcp, 0.60), 0.05))
        side = 0.015 + 0.005  # over the handle's footprint grown by 5 mm
        for axis in (0, 1):
            off = np.zeros(3)
            off[axis] = side - 0.0005
            self.assertTrue(holds_handle(arm(tcp + off, 0.0), 0.05))
            off[axis] = side + 0.0005
            self.assertFalse(holds_handle(arm(tcp + off, 0.0), 0.05))
        top = L.TOP + 0.022 + 0.040  # the handle's top
        for z, want in ((top - 0.0005, True), (top + 0.0005, False)):  # the finger tips below its top, not over it
            self.assertEqual(holds_handle(arm(np.array([tcp[0], tcp[1], z + TIPS]), 0.3), 0.05), want, z)
        self.assertFalse(holds_handle(arm(tcp, ON_HANDLE), 0.10))  # the handle has moved on without it

    def test_holds_pin_boundaries(self):
        """A gripper holds a pin when it is less open than 0.30 with its finger-tip point within 4 mm of the pin."""
        s = pin_state(L.PIN_START["pin_2"])
        tcp = np.array(L.PIN_START["pin_2"]) + [0, 0, GRIP_TCP_UP]
        self.assertTrue(holds_pin(arm(tcp, ON_PIN), s))
        self.assertTrue(holds_pin(arm(tcp, 0.29), s))
        self.assertFalse(holds_pin(arm(tcp, 0.30), s))
        for gap, want in ((0.0035, True), (0.0045, False)):  # the tips beside the pin, along x
            self.assertEqual(holds_pin(arm(tcp + [L.PIN_R + gap, 0, 0], 0.0), s), want, gap)
        # closed fingers pushing the pin from the side: the tip point stays 6.4 mm (the tips' thickness) off it
        self.assertFalse(holds_pin(arm(tcp + [L.PIN_R + 0.0064, 0, 0], 0.0), s))

    def test_let_go(self):
        pins = {n: pin_state(spot(n)) for n in L.PINS}
        self.assertTrue(let_go(arm(handle_tcp(0.1), OPEN), 0.1, pins))
        self.assertTrue(let_go(arm(handle_tcp(0.1), 0.90), 0.1, pins))  # open 0.9 or more
        self.assertFalse(let_go(arm(handle_tcp(0.1), 0.85), 0.1, pins))
        self.assertFalse(let_go(arm(handle_tcp(0.1), ON_HANDLE), 0.1, pins))
        top = L.TOP + L.PLATE_TOP + L.HANDLE[2]  # the handle's top
        for up, want in ((0.0195, False), (0.0205, True)):  # closed, its finger tips this far above the handle
            tcp = np.array([L.X_SEAM, L.HANDLE_Y - 0.1, top + up + TIPS])
            self.assertEqual(let_go(arm(tcp, ON_HANDLE), 0.1, pins), want, up)
        on_pin = spot("pin_1") + np.array([0, 0, GRIP_TCP_UP])
        self.assertFalse(let_go(arm(on_pin, ON_PIN), 0.1, pins))
        self.assertTrue(let_go(arm(on_pin + np.array([0, 0, 0.050]), ON_PIN), 0.1, pins))  # closed, 20+ mm clear

    def test_tilt_folded(self):
        self.assertAlmostEqual(tilt_deg((0, 0, 0, 1)), 0.0)
        self.assertAlmostEqual(tilt_deg((1, 0, 0, 0)), 0.0)  # upside down: the same upright pin
        self.assertAlmostEqual(tilt_deg((S45, 0, 0, S45)), 90.0, places=4)


# -- tiers ----------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(Trial().step(10).frames)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run(full_trial().frames)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertEqual(r.details["pulled_in_time"], list(L.PINS))
        self.assertEqual(r.details["in_dish"], list(L.PINS))
        self.assertEqual(r.details["unheld_travel_mm"], 0.0)
        for n in L.PINS:
            self.assertAlmostEqual(t.pulls[n]["d"], 0.050, places=3)

    def test_score_is_idempotent(self):
        tr = full_trial()
        r, t = run(tr.frames)
        feed, pins, arms = tr.frames[-1]
        self.assertEqual(t.result(feed, pins, arms).score, r.score)
        self.assertEqual(t.result(feed, pins, arms).score, r.score)

    def test_feed_thresholds(self):
        """The seam is complete at 160 mm (all three in time, short of it: 0.5); one pin in time counts from 45 mm."""
        for end, want in ((0.1595, 0.5), (0.1605, 1.0)):
            self.assertEqual(run(full_trial(feed_end=end).frames)[0].score, want, end)
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1")
        self.assertEqual(run(tr.feed_to(0.0445).step(5).frames)[0].score, 0.0)
        self.assertEqual(run(tr.feed_to(0.0455).step(5).frames)[0].score, 0.25)
        self.assertEqual(run(tr.feed_to(0.0455 + 0.03).step(5).frames)[0].score, 0.25)  # pin 2 still in the plate

    def test_ladder_rewards_pulls_in_time(self):
        """B1/B2 of the review: the pins are the skill, the last of the feed is not."""
        tr = Trial().step(5)
        for name in L.PINS:
            need = L.PIN_D0[name] - 0.055
            if need > tr.feed:
                tr.feed_to(need)
            tr.pull(name).to_dish(name)
        tr.feed_to(0.165, 40).step(20)  # all three in time and in the dish, stopped at 165 mm, still holding
        self.assertEqual(run(tr.frames)[0].score, 0.75)
        tr.let_go_left().step(25)
        self.assertEqual(run(tr.frames)[0].score, 1.0)
        tr = Trial().step(5)  # all three in time, stood on the table beside the dish
        for name in L.PINS:
            need = L.PIN_D0[name] - 0.055
            if need > tr.feed:
                tr.feed_to(need)
            tr.pull(name)
            tr.carry(name, np.array([0.12, -0.40, L.TOP + L.PIN_H / 2]), 20)
            tr.ro, tr.right = OPEN, FAR_RIGHT.copy()
            tr.step(2)
        tr.feed_to(0.178, 60).let_go_left().step(25)
        self.assertEqual(run(tr.frames)[0].score, 0.75)
        two = Trial().step(5)  # pins 1 and 2 in time, then stopped: 0.5
        two.pull("pin_1").to_dish("pin_1").feed_to(0.050).pull("pin_2").to_dish("pin_2").step(10)
        self.assertEqual(run(two.frames)[0].score, 0.5)

    def test_fed_is_the_furthest_the_plate_went(self):
        """The plate fed 178 mm and then pulled back by its handle to 100 mm: the seam was sewn, 1.0."""
        tr = full_trial()
        frames = tr.frames[:-31]  # before the left gripper lets go
        tr2 = Trial()
        tr2.__dict__.update({k: v for k, v in tr.__dict__.items() if k != "frames"})
        tr2.frames = list(frames)
        tr2.left, tr2.lo, tr2.left_on_handle = handle_tcp(tr2.feed), ON_HANDLE, True
        tr2.feed_to(0.10, 40).let_go_left()
        tr2.right, tr2.ro = FAR_RIGHT.copy(), OPEN
        r, t = run(tr2.step(25).frames)
        self.assertAlmostEqual(t.feed_max, 0.178)
        self.assertEqual(r.score, 1.0)

    def test_grippers_must_let_go(self):
        r, _ = run(full_trial(release=False).frames)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        self.assertFalse(r.details["grippers_let_go"])

    def test_window_boundaries(self):
        lo, hi, e = 0.030, 0.070, 0.0002  # task.json: d 30 to 70 mm
        cases = (((lo + e, 0.05, 0.05), 1.0), ((lo - e, 0.05, 0.05), 0.5), ((0.05, hi - e, 0.05), 1.0),
                 ((0.05, hi + e, 0.05), 0.5), ((0.05, 0.05, hi + e), 0.5), ((0.05, lo - e, 0.05), 0.5),
                 ((0.05, hi + 0.01, lo - e), 0.25))
        for pulls, want in cases:
            self.assertEqual(run(full_trial(pull_d=pulls).frames)[0].score, want, pulls)

    def test_pins_not_in_the_dish(self):
        self.assertEqual(run(full_trial(dish=False).frames)[0].score, 0.75)

    def test_two_pins_in_the_dish_is_not_three(self):
        tr = Trial().step(5)
        for name in L.PINS:
            need = L.PIN_D0[name] - 0.05
            if need > tr.feed:
                tr.feed_to(need)
            tr.pull(name)
            if name == "pin_3":
                tr.carry(name, np.array([0.20, -0.20, L.TOP + L.PIN_H / 2]))
                tr.ro, tr.right = OPEN, FAR_RIGHT.copy()
            else:
                tr.to_dish(name)
        tr.feed_to(0.178, 60).let_go_left().step(25)
        self.assertEqual(run(tr.frames)[0].score, 0.75)

    def test_success_needs_stillness(self):
        tr = full_trial()
        r, t = run(tr.frames)
        feed, pins, arms = tr.frames[-1]
        moving = dict(pins, pin_2=pin_state(pins["pin_2"].pos, vel=(0.0, 0.0, -0.05)))
        r2 = t.update(feed, moving, arms)
        self.assertFalse(r2.success)
        r3 = t.update(feed + 0.0004, pins, arms)  # the plate still sliding at 8 mm/s
        self.assertEqual(r3.score, 1.0)
        self.assertFalse(r3.success)
        r4 = t.update(feed + 0.0004, pins, arms)
        self.assertTrue(r4.success)

    def test_success_waits_for_the_release(self):
        """Success (which ends the trial after SUCCESS_HOLD_S) is never true while a gripper still holds the handle:
        the hold can only begin once both grippers have let go."""
        tr = full_trial(release=False)
        t = Tracker(HZ)
        for feed, pins, arms in tr.frames:
            self.assertFalse(t.update(feed, pins, arms).success)


# -- zero conditions --------------------------------------------------------------------------------------------
class Zeros(unittest.TestCase):
    def test_strike_boundary(self):
        """Pin 1 left in the work while feeding: its centre may come to 15 mm of the needle line, not closer."""
        safe = L.PIN_D0["pin_1"] - 0.015
        tr = Trial().step(5).feed_to(safe).step(5)
        r, t = run(tr.frames)
        self.assertNotIn("strike", r.details["zeros"])
        tr.feed_to(safe + 0.0005).step(2)
        r, t = run(tr.frames)
        self.assertIn("strike", r.details["zeros"])
        self.assertEqual(r.score, 0.0)

    def test_zero_is_never_undone(self):
        tr = Trial().step(5).feed_to(L.PIN_D0["pin_1"] - 0.010)  # pin 1 within 10 mm: struck
        tr.pull("pin_1").to_dish("pin_1")  # then pulled after all and everything else done right
        for name in ("pin_2", "pin_3"):
            tr.feed_to(L.PIN_D0[name] - 0.05).pull(name).to_dish(name)
        tr.feed_to(0.178, 60).let_go_left().step(25)
        r, _ = run(tr.frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zeros"], ["strike"])

    def test_moving_unheld_boundary(self):
        """The work may creep up to 5 mm in all with no gripper on its handle."""
        for creep, zero in ((0.0048, False), (0.0056, True)):
            tr = Trial()
            tr.left, tr.lo, tr.left_on_handle = np.array([0.30, 0.30, L.TOP + 0.20]), OPEN, False
            tr.step(5).feed_to(creep / 2, 4, hold=False).feed_to(0.0, 4, hold=False).step(3)  # there and back
            r, _ = run(tr.frames)
            self.assertEqual("moved_unheld" in r.details["zeros"], zero, creep)

    def test_off_table(self):
        tr = Trial().step(5).pull("pin_1")
        tr.carry("pin_1", np.array([0.30, -0.70, L.TOP - 0.10]), 30)
        r, _ = run(tr.frames)
        self.assertIn("off_table", r.details["zeros"])


# -- unintended strategies ----------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_pull_every_pin_first(self):
        """All three pulled at the start (pins 2 and 3 100 and 150 mm before the needle), then the seam fed: only
        pin 1 was pulled in time: 0.25."""
        r, _ = run(full_trial(pull_d=(0.05, 0.10, 0.15)).frames)
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["pulled_in_time"], ["pin_1"])

    def test_drag_the_work_by_a_pin(self):
        """No gripper on the handle: the right gripper drags the plate by pin 2 in its pocket."""
        tr = Trial()
        tr.left, tr.lo, tr.left_on_handle = np.array([0.30, 0.30, L.TOP + 0.20]), OPEN, False
        tr.step(5).pull("pin_1").to_dish("pin_1")
        tr.right, tr.ro = tr.centre("pin_2") + np.array([0, 0, GRIP_TCP_UP]), ON_PIN
        start = tr.right.copy()
        for i in range(1, 41):
            tr.feed = 0.05 * i / 40
            tr.right = start - np.array([0.0, tr.feed, 0.0])
            tr.step()
        r, _ = run(tr.frames)
        self.assertIn("moved_unheld", r.details["zeros"])
        self.assertEqual(r.score, 0.0)

    def test_push_the_plate(self):
        """Pushing the plate along (a closed gripper on its edge, not the handle) moves it unheld: 0."""
        tr = Trial()
        tr.left, tr.lo, tr.left_on_handle = np.array([L.X_SEAM, 0.0, L.TOP + 0.035]), 0.0, False
        tr.step(5).pull("pin_1").to_dish("pin_1").feed_to(0.10, 60, hold=False)
        self.assertEqual(run(tr.frames)[0].score, 0.0)

    def test_knock_pins_flat_on_the_plate(self):
        """Pins knocked over onto the plate stay in the work and ride into the needle: a strike, 0."""
        tr = Trial().step(5)
        for name in L.PINS:
            c = tr.centre(name)
            tr.base[name] = np.array([c[0] + 0.015, c[1], L.TOP + L.PLATE_TOP + L.PIN_R])  # lying on the plate
            tr.lying.add(name)
        tr.step(2).feed_to(0.178, 80)
        r, _ = run(tr.frames)
        self.assertIn("strike", r.details["zeros"])
        self.assertEqual(r.score, 0.0)

    def test_knock_pins_onto_the_bed(self):
        """B3: all three pins swiped off the plate's side onto the bed, never gripped, the seam fed: a loose pin on
        the bed is knocked loose, 0."""
        tr = Trial().step(5)
        for name in L.PINS:
            c = tr.centre(name)
            tr.free[name] = np.array([L.X_SEAM - L.PLATE_W / 2 - 0.010, c[1], L.TOP + L.BED_TOP + L.PIN_R])
            tr.lying.add(name)
            tr.step(1)
        tr.feed_to(0.178, 80).let_go_left().step(25)
        r, t = run(tr.frames)
        self.assertEqual(r.score, 0.0)
        self.assertIn("knocked_loose", r.details["zeros"])
        self.assertEqual(t.pulls, {})

    def test_swipe_pins_onto_the_table(self):
        """A3 and A3b: all pins swiped off onto the table and the seam fed, or just pin 1 swiped and 50 mm fed: 0."""
        tr = Trial().step(5)
        for name in L.PINS:
            c = tr.centre(name)
            tr.free[name] = np.array([0.30, c[1], L.TOP + L.PIN_R])
            tr.lying.add(name)
            tr.step(1)
        tr.feed_to(0.178, 80).let_go_left().step(25)
        self.assertEqual(run(tr.frames)[0].score, 0.0)
        one = Trial().step(5)
        c = one.centre("pin_1")
        one.free["pin_1"] = np.array([0.30, c[1], L.TOP + L.PIN_R])
        one.lying.add("pin_1")
        one.step(1).feed_to(0.050, 30).step(10)
        r, _ = run(one.frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zeros"], ["knocked_loose"])

    def test_swipe_with_half_open_or_closed_fingers(self):
        """A pin pushed off the plate's side by fingers 47 mm apart beside it, or by closed fingers pushing on its
        side, was not held: knocked loose."""
        for opening, off in ((0.5, L.PIN_R + 0.002), (0.0, L.PIN_R + 0.0065)):
            tr = Trial().step(5)
            c = tr.centre("pin_1")
            start = c + [0.0, 0.0, 0.0]
            tr.free["pin_1"] = start.copy()
            tr.ro = opening
            for i in range(1, 21):
                tr.free["pin_1"] = start + [-0.002 * i, 0.0, 0.0]  # pushed toward -x, off the plate's near side
                tr.right = tr.free["pin_1"] + [off, 0.0, GRIP_TCP_UP]
                tr.step()
            r, _ = run(tr.frames)
            self.assertIn("knocked_loose", r.details["zeros"], opening)

    def test_a_low_or_sideways_pull_counts(self):
        """A1 and A2: each pin gripped and lifted only 8, 25 or 26.5 mm, then carried sideways (low) off the plate
        to the dish, in the window: pulled in time, 1.0."""
        for h in (0.008, 0.025, 0.0265):
            tr = Trial().step(5)
            for name in L.PINS:
                need = L.PIN_D0[name] - 0.055
                if need > tr.feed:
                    tr.feed_to(need)
                tr.right, tr.ro = tr.centre(name) + np.array([0, 0, GRIP_TCP_UP]), OPEN
                tr.step(3)
                tr.ro = ON_PIN
                tr.step(3)
                tr.carry(name, tr.centre(name) + h * UP, 15)
                place, c = spot(name), tr.centre(name)
                tr.carry(name, (place[0], c[1], c[2]), 20)  # sideways along x off the plate, low
                tr.carry(name, (place[0], place[1], max(c[2], place[2] + 0.03)), 20)
                tr.carry(name, place, 10)
                tr.step(2)
                tr.ro = OPEN
                tr.step(3)
                tr.right = place + np.array([0, 0, 0.08])
                tr.step(3)
            tr.feed_to(0.178, 60).let_go_left()
            tr.right, tr.ro = FAR_RIGHT.copy(), OPEN
            r, t = run(tr.step(25).frames)
            self.assertEqual(r.score, 1.0, (h, r.details["zeros"], r.details["pulled_in_time"]))
            self.assertEqual(set(t.pulls), set(L.PINS))

    def test_carry_a_pin_past_the_needle_under_the_bar(self):
        """Pin 1 pulled in time, then carried along the seam past the needle under the bar: a strike."""
        tr = Trial().step(5).pull("pin_1")
        tr.carry("pin_1", tr.centre("pin_1") + np.array([0.0, -0.10, 0.0]), 20)
        self.assertIn("strike", run(tr.frames)[0].details["zeros"])

    def test_carry_a_pin_over_the_bar_or_round_the_head(self):
        """Carried over the needle head's bar, or round its posts, a pin may cross the needle line."""
        over = Trial().step(5).pull("pin_1")
        over.carry("pin_1", over.centre("pin_1") + np.array([0.0, 0.0, 0.05]), 10)  # lowest point 126 mm up
        over.carry("pin_1", over.centre("pin_1") + np.array([0.0, -0.10, 0.0]), 20)
        self.assertNotIn("strike", run(over.frames)[0].details["zeros"])
        side = Trial().step(5).pull("pin_1")
        side.carry("pin_1", side.centre("pin_1") + np.array([-0.08, 0.0, 0.0]), 15)
        side.carry("pin_1", side.centre("pin_1") + np.array([0.0, -0.10, 0.0]), 20)
        self.assertNotIn("strike", run(side.frames)[0].details["zeros"])

    def test_a_fast_jump_across_the_needle_line_is_a_strike(self):
        """Crossing the needle line under the bar between two steps counts even if no step saw it within 6 mm."""
        tr = Trial().step(5).pull("pin_1")
        c = tr.centre("pin_1")
        tr.carry("pin_1", np.array([c[0], L.Y_NEEDLE + 0.016, c[2]]), 10)
        tr.carry("pin_1", np.array([c[0], L.Y_NEEDLE - 0.016, c[2]]), 1)
        self.assertIn("strike", run(tr.frames)[0].details["zeros"])

    def test_feed_back_does_not_undo_a_late_pull(self):
        """Pin 2 let come 20 mm from the needle line, the work fed back so it is 50 mm away, then pulled: late."""
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1")
        tr.feed_to(L.PIN_D0["pin_2"] - 0.020).feed_to(L.PIN_D0["pin_2"] - 0.050)
        tr.pull("pin_2").to_dish("pin_2")
        tr.feed_to(L.PIN_D0["pin_3"] - 0.05).pull("pin_3").to_dish("pin_3")
        tr.feed_to(0.178, 60).let_go_left().step(25)
        r, t = run(tr.frames)
        self.assertAlmostEqual(t.pulls["pin_2"]["d"], 0.050, places=3)
        self.assertNotIn("pin_2", r.details["pulled_in_time"])
        self.assertEqual(r.score, 0.5)

    def test_a_token_lift_put_back_must_still_be_pulled(self):
        """Pin 2 lifted 21 mm in time, put back on the plate and left: it rides into the needle (0)."""
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1").feed_to(0.05)
        c = tr.centre("pin_2")
        tr.carry("pin_2", c + 0.027 * UP, 10)  # lowest point 21 mm over the plate: pulled, d 50 mm
        tr.carry("pin_2", c, 10)
        tr.base["pin_2"] = c + np.array([0.0, tr.feed, 0.0])  # back in its pocket, riding the work again
        del tr.free["pin_2"]
        tr.ro, tr.right = OPEN, FAR_RIGHT.copy()
        tr.feed_to(0.178, 60)
        r, t = run(tr.frames)
        self.assertEqual(t.pulls["pin_2"]["by"], "right")
        self.assertIn("strike", r.details["zeros"])

    def test_a_token_lift_put_back_and_pulled_late(self):
        """A5: pin 2 lifted in time, stood back in its pocket, ridden to 16 mm from the needle line, then pulled and
        put in the dish: it came closer than 30 mm while in the plate, so it was not pulled in time (0.5)."""
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1").feed_to(0.05)
        c = tr.centre("pin_2")
        tr.carry("pin_2", c + 0.027 * UP, 10)
        tr.carry("pin_2", c, 10)
        tr.base["pin_2"] = c + np.array([0.0, tr.feed, 0.0])
        del tr.free["pin_2"]
        tr.ro, tr.right = OPEN, FAR_RIGHT.copy()
        tr.step(2)
        tr.feed_to(L.PIN_D0["pin_2"] - 0.016, 40)
        tr.pull("pin_2").to_dish("pin_2")
        tr.feed_to(L.PIN_D0["pin_3"] - 0.05).pull("pin_3").to_dish("pin_3")
        tr.feed_to(0.178, 60).let_go_left()
        tr.right, tr.ro = FAR_RIGHT.copy(), OPEN
        r, t = run(tr.step(25).frames)
        self.assertAlmostEqual(t.min_d["pin_2"], 0.016, places=3)
        self.assertNotIn("pin_2", r.details["pulled_in_time"])
        self.assertEqual(r.score, 0.5)

    def test_hovering_over_the_handle_is_not_holding_it(self):
        """A4: the left gripper closed to 0.5, 9 mm above the handle's top, following it while something else moves
        the plate: not held, 0."""
        tr = Trial()

        def hover(feed):
            return np.array([L.X_SEAM, L.HANDLE_Y - feed, L.TOP + L.PLATE_TOP + L.HANDLE[2] + 0.009 + TIPS])

        tr.left, tr.lo, tr.left_on_handle = hover(0.0), 0.5, False
        tr.step(5).pull("pin_1").to_dish("pin_1")
        for i in range(1, 41):
            tr.feed = 0.05 * i / 40
            tr.left = hover(tr.feed)
            tr.step()
        r, _ = run(tr.frames)
        self.assertIn("moved_unheld", r.details["zeros"])

    def test_gripped_early_and_lifted_while_feeding(self):
        """Pin 2 gripped at d 75 mm and lifted slowly while the plate keeps feeding: its pull is judged where it
        comes out of its pocket, still before the window: not in time."""
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1").feed_to(0.030)
        tr.right, tr.ro = tr.centre("pin_2") + np.array([0, 0, GRIP_TCP_UP]), OPEN
        tr.step(3)
        tr.ro = ON_PIN
        tr.step(3)
        tr.free["pin_2"] = tr.centre("pin_2").copy()
        for i in range(1, 31):  # the plate fed 15 mm more while the pin rises 15 mm in the gripper
            tr.feed = 0.030 + 0.0005 * i
            tr.free["pin_2"] = tr.free["pin_2"] + [0.0, 0.0, 0.0005]
            tr.right = tr.free["pin_2"] + [0.0, 0.0, GRIP_TCP_UP]
            tr.left = handle_tcp(tr.feed)
            tr.step()
        r, t = run(tr.frames)
        self.assertGreater(t.pulls["pin_2"]["d"], 0.070)
        self.assertNotIn("pin_2", r.details["pulled_in_time"])

    def test_a_pin_held_beyond_the_table_edge_is_not_off_the_table(self):
        """B4: pin 1 carried 10 cm above the table with its centre 1 cm past the table's edge: not fallen."""
        tr = Trial().step(5).pull("pin_1")
        tr.carry("pin_1", np.array([0.30, -0.66, L.TOP + 0.10]), 30)
        self.assertNotIn("off_table", run(tr.frames)[0].details["zeros"])

    def test_the_first_pull_counts(self):
        """Pin 2 lifted out early (d 105 mm), stood back in its pocket, then pulled again in the window: early."""
        tr = Trial().step(5).pull("pin_1").to_dish("pin_1")
        c = tr.centre("pin_2")
        tr.carry("pin_2", c + 0.040 * UP, 10)
        tr.carry("pin_2", c, 10)
        tr.base["pin_2"] = c + np.array([0.0, tr.feed, 0.0])
        del tr.free["pin_2"]
        tr.ro, tr.right = OPEN, FAR_RIGHT.copy()
        tr.step(2)
        for name in ("pin_2", "pin_3"):
            tr.feed_to(L.PIN_D0[name] - 0.055).pull(name).to_dish(name)
        tr.feed_to(0.178, 60).let_go_left().step(25)
        r, t = run(tr.frames)
        self.assertAlmostEqual(t.pulls["pin_2"]["d"], 0.105, places=3)
        self.assertNotIn("pin_2", r.details["pulled_in_time"])
        self.assertEqual(r.score, 0.5)

    def test_holding_a_pin_in_the_dish(self):
        """The last pin held down in the dish instead of let go: the 1.0 conditions want both grippers let go."""
        tr = full_trial()
        frames = tr.frames[:-25]
        feed, pins, arms = frames[-1]
        hold = {"left": arms["left"], "right": arm(spot("pin_3") + np.array([0, 0, GRIP_TCP_UP]), ON_PIN)}
        r, _ = run(frames + [(feed, pins, hold)] * 25)
        self.assertEqual(r.score, 0.75)


# -- arms, robustness, resets ---------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_taken_as_holding_and_counted(self):
        tr = Trial().step(5)
        frames = [(f, p, None) for f, p, _ in tr.pull("pin_1").to_dish("pin_1").feed_to(0.05, 20, hold=False).frames]
        r, t = run(frames)
        self.assertNotIn("moved_unheld", r.details["zeros"])
        self.assertEqual(t.arms_unavailable, len(frames))
        self.assertIsNone(r.details["grippers_let_go"])

    def test_non_finite_arms_are_unavailable(self):
        feed, pins, arms = Trial().step(1).frames[0]
        bad = {"left": arm((math.nan, 0, 0), 0.3), "right": arms["right"]}
        _, t = run([(feed, pins, bad)])
        self.assertEqual(t.arms_unavailable, 1)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_record_alone(self):
        tr = full_trial()
        r, t = run(tr.frames[:-1])
        steps, pulls = t.steps, dict(t.pulls)
        feed, pins, arms = tr.frames[-1]
        good = pins["pin_1"]
        bads = [
            pin_state((math.nan, 0, 0)),
            pin_state(good.pos, vel=(math.inf, 0, 0)),
            pin_state(good.pos, quat=(0, 0, 0, 0)),
            SimpleNamespace(pos=good.pos, quat_xyzw=good.quat_xyzw),  # missing fields
            None,
        ]
        for bad in bads:
            rr = t.update(feed, dict(pins, pin_1=bad), arms)
            self.assertEqual(rr.score, 0.0)
            self.assertFalse(rr.success)
            self.assertEqual(t.pulls, pulls)
        for bad_feed in (math.nan, math.inf, None, "x"):
            self.assertEqual(t.update(bad_feed, pins, arms).score, 0.0)
        self.assertEqual(t.bad_steps, len(bads) + 4)
        self.assertEqual(t.steps, steps + len(bads) + 4)
        self.assertEqual(t.update(feed, pins, arms).score, 1.0)  # a good reading carries on as before

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(7)
        t = Tracker(HZ)
        for _ in range(400):
            pins = {n: pin_state((rng.uniform(-0.5, 1.2), rng.uniform(-0.9, 0.9), rng.uniform(0.0, 1.4)),
                                 lying=rng.random() < 0.3, vel=(rng.uniform(-1, 1), 0, 0)) for n in L.PINS}
            arms = {s: arm((rng.uniform(-0.5, 1.0), rng.uniform(-0.8, 0.8), rng.uniform(0.6, 1.2)), rng.random())
                    for s in ("left", "right")}
            r = t.update(rng.uniform(-0.05, 0.25), pins, arms if rng.random() < 0.9 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def _done_then(self, start_pins, feed=0.0):
        tr = full_trial()
        frames = list(tr.frames)
        feed0, pins, arms = frames[-1]
        frames.append((feed, start_pins, {"left": arm((0.32, 0.30, 1.0), OPEN), "right": arm((0.32, -0.30, 1.0),
                                                                                                 OPEN)}))
        return frames

    def test_teleport_to_the_start_resets(self):
        start = {n: pin_state(L.PIN_START[n]) for n in L.PINS}
        r, t = run(self._done_then(start))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.pulls, {})

    def test_jittered_teleport_resets(self):
        rng = random.Random(3)
        start = {n: pin_state(np.array(L.PIN_START[n]) + [rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), 0])
                 for n in L.PINS}
        r, t = run(self._done_then(start, feed=0.001))
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        tr = full_trial()
        for n in L.PINS:  # each pin carried back to its pocket slowly, the work fed back by its handle
            tr.left, tr.lo, tr.left_on_handle = handle_tcp(tr.feed), ON_HANDLE, True
            tr.carry(n, np.array(L.PIN_START[n]) - np.array([0, tr.feed, 0]) + 0.06 * UP, 30)
        tr.feed_to(0.0, 60)
        r, t = run(tr.frames)
        self.assertGreater(t.steps, 100)
        self.assertEqual(len(t.pulls), 3)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        away = {n: pin_state(np.array(L.PIN_START[n]) + [0.0, 0.0, 0.15]) for n in L.PINS}
        r, t = run(self._done_then(away, feed=0.0))
        self.assertGreater(t.steps, 100)

    def test_rigstate_reset_hook_resets_a_tracker(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        try:
            t = Tracker(HZ)
            rigstate.on_reset(t.reset)
            run(full_trial().frames, t)
            self.assertGreater(t.steps, 0)
            rigstate.offline_reset()
            self.assertEqual(t.steps, 0)
            self.assertEqual(t.pulls, {})
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


# -- the offline model ------------------------------------------------------------------------------------------------
def model_arms(left_tcp, lo, right_tcp=FAR_RIGHT, ro=OPEN):
    return {"left": arm(left_tcp, lo), "right": arm(right_tcp, ro)}


class OfflineModel(unittest.TestCase):
    def test_grip_on_the_handle_drives_the_feed(self):
        m = OfflineWork(HZ)
        m.step(model_arms(handle_tcp(0.0), OPEN))
        m.step(model_arms(handle_tcp(0.0), 0.234))  # the demo's command on the 30 mm handle
        for i in range(1, 21):
            feed, _ = m.step(model_arms(handle_tcp(0.0) - [0, 0.0015 * i, 0], 0.234))
        self.assertAlmostEqual(feed, 0.030)
        _, pins = m.state()
        for n in L.PINS:  # every pin rides along in its pocket
            self.assertAlmostEqual(float(pins[n].pos[1]), L.PIN_START[n][1] - 0.030)

    def test_no_grip_no_feed(self):
        m = OfflineWork(HZ)
        for opening in (OPEN, 0.40):  # open, or not closed onto the handle
            m.reset()
            for i in range(20):
                feed, _ = m.step(model_arms(handle_tcp(0.0) - [0, 0.0025 * i, 0], opening))
            self.assertEqual(feed, 0.0)
        m.reset()  # closed, but beside the handle
        for i in range(20):
            feed, _ = m.step(model_arms(handle_tcp(0.0) + [0.03, -0.0025 * i, 0], 0.234))
        self.assertEqual(feed, 0.0)
        m.reset()  # pushing the plate or dragging a pin: not modelled, nothing moves
        c = np.array(L.PIN_START["pin_2"]) + [0, 0, GRIP_TCP_UP]
        m.step(model_arms(handle_tcp(0.0) + [0, 0, 0.2], OPEN, c, OPEN))
        m.step(model_arms(handle_tcp(0.0) + [0, 0, 0.2], OPEN, c, 0.106))
        for i in range(20):
            feed, pins = m.step(model_arms(handle_tcp(0.0) + [0, 0, 0.2], OPEN, c - [0, 0.0025 * i, 0], 0.106))
        self.assertEqual(feed, 0.0)

    def test_the_needle_stops_the_work_and_the_grip_slips(self):
        m = OfflineWork(HZ)
        m.step(model_arms(handle_tcp(0.0), 0.234))
        for i in range(1, 28):  # 54 mm: 10 mm on past the stop, the finger tips still in the handle
            feed, _ = m.step(model_arms(handle_tcp(0.0) - [0, 0.002 * i, 0], 0.234))
        stop = L.PIN_D0["pin_1"] - (L.PIN_R + L.NEEDLE_R)
        self.assertAlmostEqual(feed, stop)
        for i in range(1, 11):  # back 20 mm: the grip slipped at the stop, so the work comes back 20 mm with it
            feed, _ = m.step(model_arms(handle_tcp(0.0) - [0, 0.054 - 0.002 * i, 0], 0.234))
        self.assertAlmostEqual(feed, stop - 0.020)

    def test_grasp_carry_and_drop_into_the_dish(self):
        m = OfflineWork(HZ)
        c = np.array(L.PIN_START["pin_1"]) + [0, 0, GRIP_TCP_UP]
        left = handle_tcp(0.0) + [0, 0, 0.2]
        m.step(model_arms(left, OPEN, c, OPEN))
        m.step(model_arms(left, OPEN, c, 0.106))
        target = spot("pin_1") + [0, 0, 0.002 + GRIP_TCP_UP]
        for i in range(1, 11):
            _, pins = m.step(model_arms(left, OPEN, c + (target - c) * i / 10, 0.106))
        self.assertTrue(np.allclose(pins["pin_1"].pos, spot("pin_1") + [0, 0, 0.002]))
        _, pins = m.step(model_arms(left, OPEN, target, OPEN))
        self.assertTrue(np.allclose(pins["pin_1"].pos, spot("pin_1")))
        self.assertGreater(np.linalg.norm(pins["pin_1"].lin_vel), 0.1)  # falling, for a moment
        for _ in range(6):
            _, pins = m.step(model_arms(left, OPEN, target + [0, 0, 0.05], OPEN))
        self.assertTrue(in_dish(pins["pin_1"]))

    def test_closing_off_centre_grasps_nothing(self):
        m = OfflineWork(HZ)
        c = np.array(L.PIN_START["pin_1"]) + [0, 0, GRIP_TCP_UP] + [0, 0.009, 0]
        left = handle_tcp(0.0) + [0, 0, 0.2]
        m.step(model_arms(left, OPEN, c, OPEN))
        m.step(model_arms(left, OPEN, c, 0.106))
        _, pins = m.step(model_arms(left, OPEN, c + [0, 0, 0.06], 0.106))
        self.assertTrue(np.allclose(pins["pin_1"].pos, L.PIN_START["pin_1"]))

    def test_a_pin_dropped_back_on_the_plate_rides_again(self):
        m = OfflineWork(HZ)
        c = np.array(L.PIN_START["pin_3"]) + [0, 0, GRIP_TCP_UP]
        left = handle_tcp(0.0)
        m.step(model_arms(left, OPEN, c, OPEN))
        m.step(model_arms(left, 0.234, c, 0.106))
        m.step(model_arms(left, 0.234, c + [0, 0, 0.03], 0.106))
        m.step(model_arms(left, 0.234, c + [0, 0, 0.03], OPEN))  # dropped back into its pocket
        for i in range(1, 11):
            _, pins = m.step(model_arms(left - [0, 0.002 * i, 0], 0.234, FAR_RIGHT, OPEN))
        self.assertAlmostEqual(float(pins["pin_3"].pos[1]), L.PIN_START["pin_3"][1] - 0.020)
        self.assertAlmostEqual(float(pins["pin_3"].low[2]), L.TOP + L.POCKET_FLOOR)

    def test_unreadable_arms_move_nothing_and_reset_starts_over(self):
        m = OfflineWork(HZ)
        m.step(model_arms(handle_tcp(0.0), 0.234))
        m.step(model_arms(handle_tcp(0.0) - [0, 0.01, 0], 0.234))
        feed, _ = m.step(None)
        self.assertAlmostEqual(feed, 0.01)
        m.reset()
        feed, pins = m.state()
        self.assertEqual(feed, 0.0)
        self.assertTrue(all(np.allclose(pins[n].pos, L.PIN_START[n]) for n in L.PINS))


# -- the task file ----------------------------------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the objects, the hooks, the reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def work(self, feed=0.0, source="sim", origin=None):
        spec = self.task.WORK
        pose = (*(spec.pos if origin is None else origin), 0.0, 0.0, 0.0, 1.0)
        return self.mk.state_from_poses(spec, pose, joints={"feed": feed}, source=source)

    def objects(self, feed=0.0, source="sim", pins=None, **kw):
        out = {"work": self.work(feed, source, **kw)}
        out.update(pins or {n: pin_state(np.array(L.PIN_START[n]) - [0, feed, 0]) for n in L.PINS})
        return out

    def feed_arms(self, left=None, lo=OPEN, right=FAR_RIGHT, ro=OPEN):
        left = handle_tcp(0.0) + [0, 0, 0.1] if left is None else left
        rigstate.offline_feed({"left": {"tcp": left, "tips": np.asarray(left) - [0, 0, TIPS], "opening": lo},
                               "right": {"tcp": right, "tips": np.asarray(right) - [0, 0, TIPS], "opening": ro}})

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[:4], ["work", "pin_1", "pin_2", "pin_3"])
        w = self.task.WORK
        self.assertTrue(w.fixed_base)
        self.assertEqual([(j.name, j.kind, j.axis) for j in w.joints], [("feed", "prismatic", "-Y")])
        self.assertAlmostEqual(w.joints[0].upper, L.FEED_MAX)
        self.assertAlmostEqual(w.joints[0].damping, L.FEED_DAMPING)
        self.assertEqual(w.joints[0].spring, 0.0)
        for n in L.PINS:
            o = next(o for o in self.task.OBJECTS if o.name == n)
            self.assertEqual(o.mass, L.PIN_MASS)
        self.assertTrue(all(o.mass is None for o in self.task.OBJECTS if not o.name.startswith("pin_")))
        self.assertEqual(self.task.SCORE_OVER, "end")
        # the plate as drawn: its top 22 mm up, the handle on it, the plate on the bed over its whole travel
        lo, hi = self.work(0.0).link_bounds("feed")
        self.assertAlmostEqual(lo[2], L.TOP + L.BED_TOP)
        self.assertAlmostEqual(hi[2], L.TOP + L.PLATE_TOP + L.HANDLE[2])
        self.assertAlmostEqual(hi[1] - lo[1], L.PLATE_Y1 - L.PLATE_Y0)
        lo2, hi2 = self.work(L.FEED_MAX).link_bounds("feed")
        self.assertAlmostEqual(lo2[1], L.PLATE_Y0 - L.FEED_MAX)
        self.assertGreaterEqual(lo2[1], L.BED_Y0)
        self.assertLessEqual(hi[1], L.BED_Y1)
        # the needle's tip 8 mm over the plate; a pin's top far above it (a pin riding in cannot pass)
        needle = next(o for o in self.task.OBJECTS if o.name == "needle")
        self.assertAlmostEqual(needle.pos[2] - needle.height / 2, L.TOP + L.PLATE_TOP + 0.008)
        self.assertGreater(L.POCKET_FLOOR + L.PIN_H, L.NEEDLE_TIP + 0.02)
        # the pull window is a yellow band 40 mm long on each strip of bed beside the plate, 1 mm high
        bands = [sh for sh in w.shapes if sh.color == L.MARK_YELLOW]
        self.assertEqual(len(bands), 2)
        for sh in bands:
            cy, sy = sh.center[1] + w.pos[1], sh.size[1]
            self.assertAlmostEqual(cy - sy / 2, L.Y_NEEDLE + 0.030)
            self.assertAlmostEqual(cy + sy / 2, L.Y_NEEDLE + 0.070)
            self.assertAlmostEqual(sh.size[0], 0.020)
            self.assertAlmostEqual(sh.size[2], 0.001)
            self.assertGreater(abs(sh.center[0]) - sh.size[0] / 2, L.PLATE_W / 2)  # beside the plate, on the bed
        # the plate passes under the bar and between the posts
        self.assertGreater(L.POST_DX - L.POST[0] / 2, L.BED_W / 2)
        self.assertGreater(L.BAR_Z0, L.PLATE_TOP + L.HANDLE[2])

    def test_offline_signal(self):
        gate = self.task.is_offline_checker_state
        self.assertTrue(gate(self.work(source="offline")))
        for src in ("sim", "spec"):
            self.assertFalse(gate(self.work(source=src)))
        self.assertFalse(gate(None))
        self.assertFalse(gate(SimpleNamespace(pos=(0, 0, 0))))

    def test_hooks_read_the_simulator_and_never_the_model(self):
        tr, model = self.task._TRACKER, self.task._MODEL
        rigstate.offline_reset()
        self.feed_arms(left=handle_tcp(0.06), lo=ON_HANDLE)
        in_dish_now = {n: pin_state(spot(n)) for n in L.PINS}
        self.task.success(self.objects(feed=0.06, pins=in_dish_now))
        self.assertAlmostEqual(tr.feed, 0.06)
        self.assertEqual(model.steps, 0)
        self.assertEqual(self.task.score(self.objects(feed=0.06, pins=in_dish_now)), 0.0)  # no pin pulled
        self.assertEqual(tr.last.details["feed_mm"], 60.0)

    def test_offline_states_use_the_model(self):
        tr, model = self.task._TRACKER, self.task._MODEL
        rigstate.offline_reset()
        self.feed_arms(left=handle_tcp(0.0), lo=0.234)
        self.task.success(self.objects(source="offline"))
        for i in range(1, 11):
            self.feed_arms(left=handle_tcp(0.0) - [0, 0.002 * i, 0], lo=0.234)
            self.task.success(self.objects(source="offline"))  # the checker's work state: feed always 0
        self.assertAlmostEqual(tr.feed, 0.020)
        self.assertEqual(model.steps, 11)
        self.task.score(self.objects(source="offline"))
        self.assertEqual(model.steps, 11)  # score() does not step the model

    def test_broken_work_state_scores_zero(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed_arms()
        bad = self.objects(origin=(math.nan, 0.0, 0.0))
        self.assertFalse(self.task.success(bad))
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(self.task.score(bad), 0.0)

    def test_reset_hooks(self):
        tr, model = self.task._TRACKER, self.task._MODEL
        rigstate.offline_reset()
        self.feed_arms(left=handle_tcp(0.0), lo=0.234)
        self.task.success(self.objects(source="offline"))
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker and the model start over
        self.assertEqual((tr.steps, model.steps), (0, 0))
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset
        self.assertEqual(tr.arms_unavailable, 1)
        self.feed_arms()
        self.task.success(self.objects(source="offline"))
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success(self.objects(source="offline"))
        self.assertEqual((tr.steps, model.steps), (1, 1))

    def test_trace_rows(self):
        path = HERE / "tests" / "_trace_test.jsonl"
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(path)
        try:
            rigstate.offline_reset()
            self.feed_arms()
            self.task.success(self.objects())
            row = json.loads(path.read_text().splitlines()[-1])
            for key in ("score", "feed_mm", "pulls", "pin_d_mm", "min_needle_gap_mm", "handle_held_by",
                        "unheld_travel_mm", "in_dish", "zeros", "left_tips", "source"):
                self.assertIn(key, row)
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(self.objects()))  # an unwritable trace never stops a trial
        finally:
            path.unlink(missing_ok=True)
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo(self):
        task = self.task
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})

        def ends(wps, label):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                if w.label == label:
                    out.append(t)
            return out

        lifts = ends(plan["right"], "pull it straight up")
        self.assertEqual(len(lifts), 3)
        # pin 1 leaves its pocket only 0.5 s or more after the left gripper has closed on the handle and settled
        held = ends(plan["left"], "grip the handle")[0] + 0.5
        lift_dur = next(w.duration for w in plan["right"] if w.label == "pull it straight up")
        self.assertGreaterEqual(lifts[0] - lift_dur, held + 0.5 - 1e-9)
        feeds, t = [], 0.0
        for w in plan["left"]:
            if w.label.startswith("feed the plate"):
                feeds.append((t, t + w.duration))
            t += w.duration
        self.assertEqual(len(feeds), 3)
        grips = ends(plan["right"], "grip the pin")
        for (f0, f1), lift, nxt in zip(feeds, lifts, grips[1:] + [math.inf]):
            self.assertGreaterEqual(f0, lift)  # the work moves only once the pin is out of it
            self.assertLess(f1 + 1.0, nxt)  # and is still before the next pin is gripped
        for wps in plan.values():
            for w in wps:
                if w.finger != task.OPEN and w.duration < 1.5:
                    prev = wps[wps.index(w) - 1]
                    self.assertEqual(prev.finger, w.finger, w.label)  # every open/close takes 1.5 s or more
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(set(task.demo()), {"left"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM", None)

    def test_instruction_is_identical_everywhere(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)


# The right arm's farthest-reaching tilted families toward +y (a full scan: azimuths every 15 deg, yaws every 30 deg;
# the best was azimuth 60, yaw 0), as (azimuth, yaw) in Reach.orient's convention.
RIGHT_FAMILIES = [(az, yaw) for az in (45, 60, 75) for yaw in (330, 0, 30)]


def _yamkin():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    return None


@unittest.skipIf(_yamkin() is None, "needs our offline kinematics (tools/offline/yamkin.py)")
class Reach(unittest.TestCase):
    """The roles are split by reach, also for tilted grips: position and orientation IK with warm starts, marching the
    tool point toward the other arm's side along the plate's line until no pose with the arm's capsules clear of the
    table and its own base exists. The orientation families are the farthest-reaching ones a full scan found (every
    azimuth, every 30 deg of yaw)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_yamkin()))
        import yamkin
        cls.K = yamkin

    def orient(self, tilt, az, yaw):
        a = math.radians(az)
        axis = np.array([math.sin(a), -math.cos(a), 0.0])
        r = self.K.axis_angle_matrix(axis, math.radians(tilt)) if tilt else np.eye(3)
        return r @ self.K.grip_rotation(math.radians(yaw))

    def march(self, base, rot, z, ys):
        """The last y of ``ys`` (in order) the tool point reaches at (0.40, y, z) with rotation ``rot``."""
        K, q, last = self.K, None, None
        quat = K.matrix_to_quat_xyzw(rot)
        for y in ys:
            ok = False
            for seed in ([q] if q is not None else []) + [K.READY, [0, 1.6, 1.8, -0.6, 0, 0], [0.8, 1.2, 1.2, -0.6, 0, 0],
                                                          [-0.8, 1.2, 1.2, -0.6, 0, 0]]:
                qq, pe, re = K.arm().solve((L.X_SEAM, y, L.TOP + z), quat, seed, base, iters=150)
                if pe < 1e-3 and math.degrees(re) < 0.5 and K.pose_clear(qq, base, L.TOP)["ok"]:
                    q, ok = qq, True
                    break
            if not ok:
                if last is not None:
                    break
                continue
            last = y
        return last

    def test_left_arm_tilted_40_reaches_neither_the_window_nor_pin_1(self):
        base = (0.0, 0.305, 0.75)
        ys = np.arange(0.0, -0.40, -0.004)
        far = min(self.march(base, self.orient(40, az, yaw), z, ys)
                  for az in (285, 300, 315) for yaw in (0, 330) for z in (0.03, 0.045))
        self.assertGreater(far, L.Y_NEEDLE + L.WINDOW[1] + 0.03)  # 3 cm or more short of the window's near end
        self.assertGreater(far, L.PIN_START["pin_1"][1] + 0.04)

    def test_right_arm_tilted_60_cannot_hold_the_handle(self):
        """The right arm's finger-tip point stays out of the handle's hold zone (its footprint grown by 5 mm) even
        tilted 60 deg: the review's 55 deg route to the old handle place is closed."""
        base = (0.0, -0.305, 0.75)
        ys = np.arange(0.10, 0.42, 0.004)
        best = -math.inf
        for az, yaw in RIGHT_FAMILIES:
            for z in (0.03, 0.05):
                rot = self.orient(60, az, yaw)
                y = self.march(base, rot, z, ys)
                if y is not None:
                    best = max(best, y + 0.013 * rot[1, 2])
        zone = L.HANDLE_Y - L.HANDLE[1] / 2 - L.HOLD_MARGIN
        self.assertLess(best, zone - 0.010)


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics (the work and pins from offline_model.py)."""

    def test_demo_predicts_one_and_each_arm_alone_zero(self):
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--no-reach", "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertEqual(rep["one_arm"], {"left_only": 0.0, "right_only": 0.0})
        for a in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][a]["worst_ik_mm"], 0.5)


if __name__ == "__main__":
    unittest.main()
