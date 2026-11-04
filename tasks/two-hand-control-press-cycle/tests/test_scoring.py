"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic part states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, button_pressed, button_released, die_gap, finger_zone, grippers_let_go, in_die, let_go, part_in_tray,
    part_seated, rotation, tilt_deg, valid, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
H = L.PART_SIZE[2] / 2  # half the part's 30 mm
TIPS = 0.013  # pad tips below the TCP
HELD = TIPS + L.RAIL_HEIGHT + 0.004 - H  # a part held by its upper 18 mm has its centre this far below the TCP (10 mm)
OPEN = 1.0  # rigstate opening: fully open
GRIP = (L.PART_SIZE[1] - 0.008) / L.FULL_GAP  # closed on the 30 mm part (22 mm gap): 0.234
CLOSED = 0.008 / L.FULL_GAP  # the narrowest commanded gap, 16 mm: 0.085

P_START = np.array([L.PLATE_X, L.PLATE_Y, L.PLATE_TOP_Z + H])
SEATED = np.array([L.NEST_X, L.NEST_Y, L.TOP + H])
IN_TRAY = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + H])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
HOVER_R = L.TOP + 0.12
BTN_L = np.array([L.BUTTON_X, L.BUTTON_Y["left"]])
BTN_R = np.array([L.BUTTON_X, L.BUTTON_Y["right"]])
HOVER_BL = np.array([*BTN_L, L.TOP + 0.10])
HOVER_BR = np.array([*BTN_R, L.TOP + 0.10])
PRESS_L = np.array([*BTN_L, L.BUTTON_TOP_Z + 0.002 + TIPS])  # tips 2 mm above the button's top
PRESS_R = np.array([*BTN_R, L.BUTTON_TOP_Z + 0.002 + TIPS])
FEED_GRIP = P_START + [0, 0, HELD]  # the right TCP closing on the part on the feed mark
NEST_GRIP = SEATED + [0, 0, HELD]  # ... and on the part seated in the nest
LOAD_TCP = NEST_GRIP + [0, 0, 0.002]  # letting it go 2 mm above the nest's floor
TRAY_GRIP = IN_TRAY + [0, 0, HELD + 0.002]  # ... and 2 mm above the tray's floor


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.PART_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def arm(tcp, opening=OPEN):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


KEYS = ("p", "pq", "L", "Lo", "R", "Ro")


class Film:
    """A trial as a list of frames: the part's centre and orientation, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; the part's velocity comes from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"p": P_START, "pq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in ("p", "L", "R") else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in ("p", "L", "R") else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k == "pq":
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    @property
    def t(self):
        return len(self.frames) - 1


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        v = np.zeros(3) if prev is None else (fr["p"] - prev["p"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(box(fr["p"], fr["pq"], v), a)
        prev = fr
    return r, t


# -- the intended procedure, in pieces -------------------------------------------------------------------------
def right_loads(f, to=SEATED, side="R", side_open="Ro"):
    """An arm (the right by default) picks the part off the feed mark by its upper 18 mm, carries it over ``to``
    (the nest), sets it down 2 mm up, lets go (the part drops the 2 mm) and goes up (about 13 s)."""
    kw = lambda **d: {k.replace("R", side).replace("Ro", side_open): v for k, v in d.items()}  # noqa: E731
    f.go(30, **kw(R=[*P_START[:2], HOVER_R])).go(20, **kw(R=FEED_GRIP + [0, 0, 0.03])).go(30, **kw(R=FEED_GRIP))
    f.go(30, **kw(Ro=GRIP))
    f.go(30, **kw(R=[*P_START[:2], HOVER_R]), p=[*P_START[:2], HOVER_R - HELD])
    f.go(30, **kw(R=[*to[:2], HOVER_R]), p=[*to[:2], HOVER_R - HELD])
    set_tcp = to + [0, 0, HELD + 0.002]
    f.go(20, **kw(R=set_tcp + [0, 0, 0.02]), p=set_tcp + [0, 0, 0.02 - HELD]).go(30, **kw(R=set_tcp), p=set_tcp - [0, 0, HELD])
    f.go(8, **kw(Ro=0.4)).go(2, p=to).go(22, **kw(Ro=OPEN))
    f.go(20, **kw(R=[*to[:2], HOVER_R]))
    return f


def both_press(f, gap=0, hold=50, left_first=True, release_gap=10):
    """Both arms go over their buttons, close, and press: the second press begins ``gap`` steps after the first;
    both held ``hold`` steps from the second's start; the first one up releases ``release_gap`` steps before the
    other. Ends with both arms hovering over their buttons, fingers closed (about 9 s)."""
    f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
    f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02])
    first, second = ("L", "R") if left_first else ("R", "L")
    pf, ps = (PRESS_L, PRESS_R) if left_first else (PRESS_R, PRESS_L)
    # both descend the last 2 cm over 30 steps (the tips enter the 'pressed' band, 10 mm over the top, halfway);
    # the second's descent starts ``gap`` steps after the first's
    top = {first: pf + [0, 0, 0.02], second: ps + [0, 0, 0.02]}
    for k in range(1, 30 + gap + 1):
        fr = dict(f.cur)
        for who, k0 in ((first, 0), (second, gap)):
            u = min(max((k - k0) / 30, 0.0), 1.0)
            fr[who] = (1 - u) * top[who] + u * (pf if who == first else ps)
        f.frames.append(fr)
    f.cur = dict(f.frames[-1])
    f.hold(hold)
    f.go(20, **{first: pf + [0, 0, 0.067]})
    if release_gap:
        f.hold(release_gap)
    f.go(20, **{second: ps + [0, 0, 0.067]})
    return f


def right_unloads(f, to=IN_TRAY, open_to=OPEN, drop=True, up=True, frm=SEATED):
    """The right arm opens, goes over the nest, grips the part there by its upper 18 mm, lifts it straight up,
    carries it to ``to`` (the tray), sets it down and opens to ``open_to``, then goes up (about 13 s)."""
    grip = frm + [0, 0, HELD]
    f.go(30, R=[*frm[:2], HOVER_R], Ro=OPEN).go(30, R=grip).go(30, Ro=GRIP)
    f.go(30, R=[*frm[:2], HOVER_R], p=[*frm[:2], HOVER_R - HELD])
    f.go(30, R=[*to[:2], HOVER_R], p=[*to[:2], HOVER_R - HELD])
    set_tcp = to + [0, 0, HELD + 0.002]
    f.go(20, R=set_tcp + [0, 0, 0.02], p=set_tcp + [0, 0, 0.02 - HELD]).go(30, R=set_tcp, p=set_tcp - [0, 0, HELD])
    if drop:
        f.go(8, Ro=0.4).go(2, p=to).go(22, Ro=open_to)
    else:
        f.go(30, Ro=open_to)
    if up:
        f.go(20, R=[*to[:2], HOVER_R])
    return f


def procedure(gap=0, hold=50, to=IN_TRAY, open_to=OPEN, settle=30, drop=True, unload=True, release_gap=10):
    """The intended trial: the right loads the part, both press (synchronous by default, held 2.5 s), the right
    unloads the part to ``to``, everything still for ``settle`` steps."""
    f = Film().hold(10)
    right_loads(f)
    f.hold(10)
    both_press(f, gap=gap, hold=hold, release_gap=release_gap)
    f.hold(10)
    if unload:
        right_unloads(f, to, open_to, drop)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_cube(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 0.0, places=4)  # on another face: the same cube
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("x", 80)), 10.0, places=4)

    def test_finger_zone_covers_the_pads(self):
        a = arm((0.2, 0.2, 0.9), OPEN)
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)  # from the tips ...
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)  # ... 38 mm back up the gripper
        closed = arm((0.2, 0.2, 0.9), 0.0)
        self.assertAlmostEqual(zone_radius(closed), math.hypot(0.008, 0.0135), places=6)

    def test_in_die(self):
        # open fingers over the nest: inside; over the feed mark 12 cm to the left: outside
        self.assertTrue(in_die(arm((L.NEST_X, L.NEST_Y, HOVER_R))))
        self.assertFalse(in_die(arm((L.PLATE_X, L.PLATE_Y, HOVER_R))))
        self.assertAlmostEqual(die_gap(arm((L.PLATE_X, L.PLATE_Y, HOVER_R))),
                               math.hypot(0.25 - 0.24, 0.12 - 0.05) - zone_radius(arm((0, 0, 0))), places=6)  # 14 mm
        # over the tray (x 16 cm, 9 cm before the footprint): outside even fully open (zone 57 mm)
        self.assertFalse(in_die(arm((L.TRAY_X, L.TRAY_Y, L.TOP + 0.03))))
        self.assertTrue(in_die(arm((0.20, 0.0, L.TOP + 0.05))))  # axis 5 cm before the footprint, open: inside
        self.assertFalse(in_die(arm((0.20, 0.0, L.TOP + 0.05), 0.0)))  # closed (zone 16 mm): outside
        self.assertFalse(in_die(arm((0.19, 0.0, L.TOP + 0.05))))  # open, 6 cm before it (zone 56.6 mm): outside
        # the buttons are 21 cm off the midline: a gripper on either is 13 cm clear of the footprint
        for p in (PRESS_L, PRESS_R, HOVER_BL, HOVER_BR):
            self.assertFalse(in_die(arm(p, CLOSED)))
            self.assertGreater(die_gap(arm(p, OPEN)), 0.10)
        # height: the zone runs 38 mm up from the tips; above 20 cm it is outside the space
        self.assertTrue(in_die(arm((L.NEST_X, L.NEST_Y, L.TOP + 0.20 + TIPS - 0.001))))  # tips 1 mm under 20 cm
        self.assertFalse(in_die(arm((L.NEST_X, L.NEST_Y, L.TOP + 0.20 + TIPS + 0.0566))))  # a zone radius over it
        # the ready poses (x 32.3 cm, y +-30.5 cm, 26 cm up) are outside
        self.assertFalse(in_die(arm(READY_L)))
        self.assertFalse(in_die(arm(READY_R)))

    def test_button_pressed_and_released(self):
        self.assertTrue(button_pressed(arm(PRESS_L, CLOSED), "left"))
        self.assertTrue(button_pressed(arm(PRESS_R, CLOSED), "right"))
        self.assertFalse(button_pressed(arm(PRESS_L, CLOSED), "right"))  # the left gripper is not on the right button
        self.assertFalse(button_pressed(arm(PRESS_R, CLOSED), "left"))
        self.assertTrue(button_pressed(arm(PRESS_L, 0.5), "left"))
        self.assertFalse(button_pressed(arm(PRESS_L, 0.51), "left"))  # fingers too open
        self.assertFalse(button_pressed(arm(PRESS_L, OPEN), "left"))
        self.assertTrue(button_pressed(arm(PRESS_L + [0.015, 0, 0.007], CLOSED), "left"))  # 15 mm off, tips 9 mm up
        self.assertFalse(button_pressed(arm(PRESS_L + [0.017, 0, 0], CLOSED), "left"))  # 17 mm off the axis
        self.assertFalse(button_pressed(arm(PRESS_L + [0, 0, 0.009], CLOSED), "left"))  # tips 11 mm above the top
        self.assertTrue(button_pressed(arm(PRESS_L - [0, 0, 0.0065], CLOSED), "left"))  # tips 4.5 mm 'into' it (sag)
        self.assertFalse(button_pressed(arm(PRESS_L - [0, 0, 0.008], CLOSED), "left"))  # 6 mm into it: not a press
        self.assertFalse(button_released(arm(PRESS_L, CLOSED), "left"))
        self.assertFalse(button_released(arm(PRESS_L + [0, 0, 0.027], CLOSED), "left"))  # tips 29 mm up: neither
        self.assertTrue(button_released(arm(PRESS_L + [0, 0, 0.029], CLOSED), "left"))  # 31 mm up
        self.assertTrue(button_released(arm(PRESS_L + [0.031, 0, 0], CLOSED), "left"))  # 31 mm off the axis
        self.assertTrue(button_released(arm(HOVER_BL, CLOSED), "left"))
        self.assertTrue(button_released(arm(READY_R, CLOSED), "right"))

    def test_part_places(self):
        self.assertTrue(part_seated(box(SEATED)))
        self.assertTrue(part_seated(box(SEATED + [0.003, 0.0025, 0.002])))  # 3.9 mm off, 2 mm up: seated
        self.assertFalse(part_seated(box(SEATED + [0.0041, 0, 0])))  # 4.1 mm off the nest's centre
        self.assertFalse(part_seated(box(SEATED + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(part_seated(box(SEATED + [0.003, 0, 0.0005], about("y", 6))))  # tilted 6 degrees
        self.assertTrue(part_seated(box(SEATED, about("z", 90))))  # a quarter turn: still a cube in a square nest
        # up on a rail: its centre 18 mm or more off, and its bottom 8 mm up: not seated
        self.assertFalse(part_seated(box((L.NEST_X + 0.022, L.NEST_Y, L.RAIL_TOP_Z + H))))
        self.assertFalse(part_seated(box(P_START)))
        self.assertFalse(part_seated(box(IN_TRAY)))
        self.assertTrue(part_in_tray(box(IN_TRAY)))
        self.assertTrue(part_in_tray(box(IN_TRAY + [0.03, -0.06, 0.002])))  # off centre, 2 mm up
        self.assertFalse(part_in_tray(box(IN_TRAY + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(part_in_tray(box(IN_TRAY + [0.045, 0, 0])))  # centre over the wall
        self.assertFalse(part_in_tray(box((L.TRAY_X, L.TRAY_Y - 0.08, L.TRAY_TOP_Z + H))))  # on the wall
        self.assertFalse(part_in_tray(box(SEATED)))

    def test_let_go(self):
        lo, hi = box(IN_TRAY).low, box(IN_TRAY).high
        self.assertTrue(let_go(arm(TRAY_GRIP, OPEN), lo, hi))
        self.assertTrue(let_go(arm(TRAY_GRIP, 0.9), lo, hi))
        self.assertFalse(let_go(arm(TRAY_GRIP, 0.85), lo, hi))  # 80 mm apart, still round the part
        self.assertFalse(let_go(arm(TRAY_GRIP, GRIP), lo, hi))
        self.assertTrue(let_go(arm(TRAY_GRIP + [0, 0, 0.10], GRIP), lo, hi))  # closed 10 cm above it: not holding
        arms = {"left": arm(HOVER_BL, CLOSED), "right": arm(TRAY_GRIP + [0, 0, 0.1], OPEN)}
        self.assertTrue(grippers_let_go(arms, box(IN_TRAY)))
        arms["left"] = arm(TRAY_GRIP, GRIP)  # the left holding it counts too
        self.assertFalse(grippers_let_go(arms, box(IN_TRAY)))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertFalse(r.details["part_seated"])

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breach_steps, 0)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertTrue(t.cycled)
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(t.sync_presses, 1)
        self.assertEqual(t.seated_presses, 1)
        self.assertEqual(r.details["double_gap_s"], 0.0)
        self.assertGreaterEqual(r.details["longest_double_s"], 2.5)
        self.assertEqual(r.details["released"], {"left": True, "right": True})
        self.assertTrue(r.details["grippers_let_go"])
        self.assertTrue(r.details["part_in_tray"])

    def pressed_for(self, n):
        """Both buttons pressed on exactly ``n`` consecutive steps (jumped into and out of the band), the part
        seated."""
        f = Film().hold(10)
        right_loads(f)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02])
        f.go(1, L=PRESS_L, R=PRESS_R).hold(n - 1)
        f.go(1, L=HOVER_BL, R=HOVER_BR)
        return play(f.hold(10))

    def test_cycle_completes_on_the_40th_step_of_the_double_press(self):
        r, t = self.pressed_for(1)  # both pressed on one step only
        self.assertFalse(t.cycled)
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(r.score, 0.5)  # a synchronous double press, too short
        r, t = self.pressed_for(L.CYCLE_STEPS - 1)  # 39 steps: no cycle
        self.assertFalse(t.cycled)
        self.assertEqual(r.details["longest_double_s"], 1.95)
        self.assertEqual(r.score, 0.5)
        r, t = self.pressed_for(L.CYCLE_STEPS)  # 40 steps: the cycle completes
        self.assertTrue(t.cycled)
        self.assertEqual(t.cycle_seated_steps, L.CYCLE_STEPS)
        self.assertEqual(r.score, 0.75)  # cycled, not yet unloaded

    def test_cycled_but_not_in_the_tray_scores_075(self):
        # still in the nest after the cycle
        r, t = play(procedure(unload=False))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertTrue(t.cycled)
        self.assertTrue(r.details["part_seated"])
        # held in the air over the tray
        f = procedure(settle=0)
        r, _ = play(f, until=len(f.frames) - 100)
        self.assertFalse(r.details["part_in_tray"])
        self.assertEqual(r.score, 0.75)
        # set down on the table beside the tray
        beside = np.array([0.16, -0.12, L.TOP + H])
        r, _ = play(procedure(to=beside))
        self.assertEqual(r.score, 0.75)
        # on the tray's wall
        r, _ = play(procedure(to=np.array([L.TRAY_X, L.TRAY_Y - 0.073, L.TRAY_TOP_Z + H])))
        self.assertEqual(r.score, 0.75)

    def test_in_the_tray_with_a_button_still_pressed_scores_075(self):
        # the left releases only 2 cm (tips 22 mm over the button): neither pressed nor released: 0.75
        f = Film().hold(10)
        right_loads(f)
        f.hold(10)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02]).go(30, L=PRESS_L, R=PRESS_R).hold(50)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=HOVER_BR)
        f.hold(10)
        right_unloads(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertTrue(t.cycled)
        self.assertTrue(r.details["part_in_tray"])
        self.assertEqual(r.details["released"], {"left": False, "right": True})
        self.assertEqual(t.breach_steps, 0)  # in the band the left is not pressing, so the unload is no breach
        # (the left still fully pressing while the right unloads is a breach: see Breaches)

    def test_seated_with_a_double_press_but_no_cycle_scores_half(self):
        r, t = play(procedure(gap=20, unload=False))  # 1.0 s apart: not synchronous
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.cycled)
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(t.sync_presses, 0)
        self.assertEqual(r.details["double_gap_s"], 1.0)
        r, t = play(procedure(gap=11, unload=False))  # 0.55 s apart: just not synchronous
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.cycled)
        r, t = play(procedure(gap=10, unload=False))  # 0.5 s apart: synchronous
        self.assertEqual(r.score, 0.75)
        self.assertTrue(t.cycled)
        r, t = play(procedure(hold=20, unload=False))  # synchronous, held 1.5 s in all: too short
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.cycled)
        self.assertLess(r.details["longest_double_s"], L.CYCLE_S)

    def test_seated_without_a_double_press_scores_quarter(self):
        f = Film().hold(10)
        right_loads(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertEqual(t.double_presses, 0)
        # one button only, held a long time (the right first gets out of the die space, or it would be a breach)
        f.go(30, R=READY_R)
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L).hold(100)
        f.go(20, L=HOVER_BL)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.25)
        self.assertEqual(t.double_presses, 0)
        self.assertGreaterEqual(r.details["press_s"]["left"], 5.0)
        self.assertEqual(r.details["press_s"]["right"], 0.0)

    def test_part_off_the_table_scores_zero(self):
        f = procedure(settle=5)
        f.go(30, p=[0.30, -0.70, L.TOP - 0.40]).hold(10)  # swept off the table's edge afterwards
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["part_off_table"])

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, p=IN_TRAY + [0.004, 0, 0]).go(3, p=IN_TRAY))  # still sliding in the tray
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))  # stopped
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=HOVER_BL, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
        r = tr.update(box(IN_TRAY, vel=(0.02, 0, 0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_TRAY, spin=(0, 0, 1.0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_TRAY), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        a = snap(left_tcp=HOVER_BL, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
        self.assertEqual(t.result(box(IN_TRAY), a).score, 1.0)
        self.assertEqual(t.result(box(IN_TRAY), a).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))

    def test_the_part_may_land_on_any_face(self):
        f = procedure(settle=0).go(1, pq=about("x", 90)).go(1, pq=about("y", 180)).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_a_gripper_still_holding_the_part_in_the_tray_scores_075(self):
        f = Film().hold(10)
        right_loads(f)
        f.hold(10)
        both_press(f)
        f.hold(10)
        right_unloads(f, open_to=GRIP, drop=False, up=False)  # set down 2 mm up, the fingers never open
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["grippers_let_go"])
        self.assertTrue(t.cycled)

    def test_a_cycle_is_never_undone(self):
        # after the cycle, a second double press that is not synchronous changes nothing
        f = procedure(settle=5)
        both_press(f, gap=30)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.double_presses, 2)
        self.assertTrue(t.cycled)


class Breaches(unittest.TestCase):
    """A button pressed while a gripper's fingers are in the die space, on any step: 0 for the rest of the trial."""

    def assert_breach(self, f, arms=("right",)):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertTrue(t.breach)
        self.assertEqual(t.breach_arms, set(arms))
        self.assertEqual(r.details["first_breach_step"], t.first_breach_step)
        return t

    def test_left_presses_while_the_right_is_still_over_the_nest(self):
        f = Film().hold(10)
        right_loads(f)  # ends with the right hovering 12 cm up over the nest: in the die space
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L)  # the left presses
        f.hold(20)
        f.go(20, L=HOVER_BL)
        f.go(30, R=HOVER_BR).go(30, Ro=CLOSED)
        both_press(f)
        f.hold(10)
        right_unloads(f)
        t = self.assert_breach(f.hold(30))
        self.assertGreater(t.breach_steps, 10)
        self.assertLess(t.first_breach_step, 400)  # as soon as the left's tips are in the 'pressed' band

    def test_pressing_while_loading(self):
        # the left holds its button down from the start while the right loads the part: the moment the right's
        # fingers reach the die space, a breach
        f = Film().hold(10)
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L)
        right_loads(f)
        f.go(20, L=HOVER_BL)
        both_press(f)
        f.hold(10)
        right_unloads(f)
        t = self.assert_breach(f.hold(30))
        self.assertTrue(t.cycled)  # the cycle itself was fine; the breach beforehand still makes the trial 0

    def test_unloading_while_a_button_is_still_pressed(self):
        f = Film().hold(10)
        right_loads(f)
        f.hold(10)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02]).go(30, L=PRESS_L, R=PRESS_R).hold(50)
        f.go(20, R=HOVER_BR)  # only the right releases; the left stays on its button
        right_unloads(f)  # the right goes back to the nest: a breach the moment its fingers are in the die space
        f.go(20, L=HOVER_BL)
        t = self.assert_breach(f.hold(30))
        self.assertTrue(t.cycled)

    def test_one_step_of_press_with_a_gripper_inside_is_a_breach(self):
        f = procedure(settle=0)
        # find a step with the right over the nest unloading (after the cycle), and press the left for that one step
        n = next(i for i, fr in enumerate(f.frames) if fr["Lo"] == CLOSED and np.allclose(fr["R"], NEST_GRIP)
                 and fr["Ro"] == GRIP)
        f.frames[n] = dict(f.frames[n], L=PRESS_L)
        t = self.assert_breach(f.hold(30))
        self.assertEqual(t.breach_steps, 1)
        self.assertEqual(t.first_breach_step, n + 1)

    def test_a_gripper_in_the_die_space_with_no_button_pressed_is_no_breach(self):
        f = Film().hold(10)
        right_loads(f)
        r, t = play(f.hold(30))
        self.assertFalse(t.breach)
        self.assertGreater(t.in_die_steps, 50)

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5)
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L)
        f.go(20, R=[L.NEST_X, L.NEST_Y, HOVER_R]).go(20, R=READY_R)  # a look at the nest, the left pressed
        f.go(20, L=HOVER_BL)
        g = procedure(settle=60)
        f.frames = f.frames + g.frames[1:]
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["breach"], True)
        self.assertTrue(t.cycled)
        self.assertLess(t.first_breach_step, 160)


class Exploits(unittest.TestCase):
    def test_carrying_the_part_straight_to_the_tray_scores_zero(self):
        f = Film().hold(10)
        right_loads(f, to=IN_TRAY)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertTrue(r.details["part_in_tray"])
        self.assertFalse(t.cycled)
        self.assertFalse(t.breach)

    def test_doing_nothing_scores_zero(self):
        r, _ = play(Film().hold(400))
        self.assertEqual((r.score, r.success), (0.0, False))

    def test_pressing_both_buttons_with_the_part_on_the_feed_mark_then_loading_scores_quarter(self):
        f = Film().hold(10)
        both_press(f)  # an 'empty' cycle: both pressed 2.5 s, synchronous, part not seated
        f.hold(10)
        f.go(30, Lo=OPEN, Ro=OPEN)
        right_loads(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(t.seated_presses, 0)
        self.assertFalse(t.cycled)
        self.assertFalse(t.breach)

    def test_part_unseated_during_the_double_press_is_no_cycle(self):
        # the part is seated when the double press begins, then (somehow) nudged 5 mm during it
        f = Film().hold(10)
        right_loads(f)
        f.hold(10)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02]).go(30, L=PRESS_L, R=PRESS_R)
        f.hold(10).go(5, p=SEATED + [0.005, 0, 0]).go(5, p=SEATED).hold(40)
        f.go(20, L=HOVER_BL, R=HOVER_BR)
        r, t = play(f.hold(30))
        self.assertFalse(t.cycled)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.seated_presses, 1)
        # a second, clean double press afterwards completes the cycle
        both_press(f)
        r, t = play(f.hold(30))
        self.assertTrue(t.cycled)
        self.assertEqual(r.score, 0.75)

    def test_knocking_the_part_toward_the_nest_cannot_seat_it(self):
        # shoved along the table from the feed mark, the part stops against the nest's outer rail face (x 0.274 at
        # best, 26 mm from the nest's centre, or y 0.026): never within 4 mm
        f = Film().hold(10)
        f.go(30, R=[L.PLATE_X, L.PLATE_Y + 0.05, L.TOP + 0.03]).go(40, R=[L.NEST_X, L.NEST_Y + 0.06, L.TOP + 0.03],
                                                                   p=[L.NEST_X, L.NEST_Y + 0.026 + H, L.TOP + H])
        f.go(20, R=READY_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["part_seated"])
        self.assertFalse(t.breach)
        # and if it somehow ended up on a rail: not seated either
        f = Film().hold(10).go(30, p=(L.NEST_X + 0.022, L.NEST_Y, L.RAIL_TOP_Z + H)).hold(10)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_left_arm_alone_scores_quarter(self):
        # the left loads the part itself (the feed mark is in its reach) and presses its button, held long
        f = Film().hold(10)
        right_loads(f, side="L", side_open="Lo")
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L).hold(100)
        f.go(20, L=HOVER_BL)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertEqual(t.double_presses, 0)
        self.assertFalse(t.breach)
        # loading with its own button pressed is a breach; it cannot press and load at once anyway

    def test_right_arm_alone_scores_quarter(self):
        f = Film().hold(10)
        right_loads(f)
        f.go(30, R=HOVER_BR).go(30, Ro=CLOSED).go(20, R=PRESS_R + [0, 0, 0.02]).go(30, R=PRESS_R).hold(100)
        f.go(20, R=HOVER_BR)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertEqual(t.double_presses, 0)
        # ... and then unloading it to the tray, uncycled, drops to 0
        right_unloads(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["part_in_tray"])

    def test_one_gripper_cannot_press_both_buttons(self):
        # even if a gripper were somehow over the other arm's button, it would not count
        f = Film().hold(10)
        right_loads(f)
        f.go(30, R=READY_R)
        # the left goes 'onto' the right button (unreachable in reality), high over the die space on the way
        f.go(30, L=READY_L + [0, 0, 0.1]).go(30, L=[*BTN_R, L.TOP + 0.36]).go(30, L=PRESS_R, Lo=CLOSED)
        f.go(30, R=HOVER_BR).go(30, Ro=CLOSED).go(20, R=PRESS_R + [0, 0, 0.02]).go(30, R=PRESS_R)
        r, t = play(f.hold(60))
        self.assertEqual(t.double_presses, 0)
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.25)

    def test_pressing_with_open_fingers_does_not_count(self):
        f = Film().hold(10)
        right_loads(f)
        f.go(30, L=HOVER_BL, R=HOVER_BR)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02]).go(30, L=PRESS_L, R=PRESS_R).hold(50)
        r, t = play(f.hold(30))  # fingers wide open on both
        self.assertEqual(t.double_presses, 0)
        self.assertEqual(r.score, 0.25)

    def test_pressing_off_centre_does_not_count(self):
        f = Film().hold(10)
        right_loads(f)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(50, L=PRESS_L + [0, 0.02, 0], R=PRESS_R).hold(50)  # the left 20 mm off its button's axis
        r, t = play(f.hold(30))
        self.assertEqual(t.double_presses, 0)
        self.assertEqual(r.score, 0.25)

    def test_a_flicker_breaks_the_double_press(self):
        # the left lifts to 14 mm over its button for one step in the middle of a 2.5 s double press: two spans,
        # neither 2 s, and the second begins 1.2 s after the right's: no cycle
        f = Film().hold(10)
        right_loads(f)
        f.hold(10)
        f.go(30, L=HOVER_BL, R=HOVER_BR).go(30, Lo=CLOSED, Ro=CLOSED)
        f.go(20, L=PRESS_L + [0, 0, 0.02], R=PRESS_R + [0, 0, 0.02]).go(30, L=PRESS_L, R=PRESS_R)
        f.hold(24)
        f.frames.append(dict(f.cur, L=PRESS_L + [0, 0, 0.012]))
        f.cur = dict(f.frames[-1])
        f.go(1, L=PRESS_L).hold(25)
        f.go(20, L=HOVER_BL, R=HOVER_BR)
        r, t = play(f.hold(30))
        self.assertFalse(t.cycled)
        self.assertEqual(t.double_presses, 2)
        self.assertEqual(r.score, 0.5)

    def test_part_dropped_from_above_the_tray_still_counts_if_it_lands_in(self):
        f = procedure(to=IN_TRAY + [0, 0, 0.02], settle=0).go(2, p=IN_TRAY).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_loading_with_a_sweep_above_the_die_space(self):
        # A gripper 21 cm up over the nest with a button pressed is outside the space: no breach, but it cannot
        # touch anything there either.
        f = Film().hold(10)
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L)
        f.go(30, R=[L.NEST_X, L.NEST_Y, L.TOP + 0.20 + TIPS + 0.06]).go(30, R=[L.NEST_X, -0.2, L.TOP + 0.20 + TIPS + 0.06])
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)


class Arms(unittest.TestCase):
    def test_arms_unavailable_no_press_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 0.0)  # no gripper reading: no press is ever registered, so no cycle; the part is in
        #                                 the tray uncycled
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertEqual(t.double_presses, 0)
        self.assertIsNone(r.details["opening"])
        self.assertIsNone(r.details["pressed"])
        self.assertFalse(t.breach)
        r, t = play(procedure(unload=False), arms=False)
        self.assertEqual(r.score, 0.25)  # seated, no press seen

    def test_an_unreadable_step_breaks_a_double_press(self):
        f = procedure(unload=False, settle=0)
        n = next(i for i, fr in enumerate(f.frames) if np.allclose(fr["L"], PRESS_L) and np.allclose(fr["R"], PRESS_R))
        t = Tracker(HZ)
        prev = None
        for i, fr in enumerate(f.frames):
            v = np.zeros(3) if prev is None else (fr["p"] - prev["p"]) / DT
            a = None if i == n + 20 else snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
            r = t.update(box(fr["p"], fr["pq"], v), a)
            prev = fr
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(t.double_presses, 2)  # the span restarted after the gap
        self.assertFalse(t.cycled)  # each half of the broken 3.2 s double press is under 2 s
        self.assertLess(t.longest_double, L.CYCLE_STEPS)
        self.assertEqual(r.score, 0.5)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(box(P_START), snap(left_tcp=PRESS_L, left_open=CLOSED, right_tcp=[L.NEST_X, L.NEST_Y, HOVER_R]))
        self.assertTrue(t.breach)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(box(P_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(box(P_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(box(P_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])
        self.assertEqual(r.details["in_die"], {"left": False, "right": False})
        self.assertEqual(r.details["pressed"], {"left": False, "right": False})


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = box(IN_TRAY)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=IN_TRAY)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (t.breach_steps, t.cycled, t.double_presses, t.in_die_steps, t.arms_unavailable)
            a = snap(left_tcp=HOVER_BL, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
            for _ in range(3):
                r = t.update(bad, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(bad, a).score, 0.0)
                self.assertFalse(r.details["valid_state"])
            self.assertEqual((t.breach_steps, t.cycled, t.double_presses, t.in_die_steps, t.arms_unavailable), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(IN_TRAY, about("x", 89.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        a = snap()
        for s in odd + odd[::-1]:
            r = t.update(s, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertEqual(r.success, r.score == 1.0)
        r = t.update(box(P_START), None)
        self.assertIn(r.score, (0.0, 0.25))


class Reset(unittest.TestCase):
    def breached(self):
        f = Film().hold(5)
        f.go(30, L=HOVER_BL).go(30, Lo=CLOSED).go(20, L=PRESS_L + [0, 0, 0.02]).go(30, L=PRESS_L)
        f.go(20, R=[L.NEST_X, L.NEST_Y, HOVER_R])  # the right over the nest, the left pressed
        f.go(20, L=HOVER_BL)
        g = Film().hold(10)
        right_loads(g)
        f.frames = f.frames + g.frames[1:200]  # the part held up over the nest
        _, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        self.assertGreater(float(np.linalg.norm(t.prev[0] - P_START)), 0.08)
        r = t.update(box(P_START), snap())  # the part jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(box(P_START + [0.02, -0.02, 0]), snap())  # a seeded reset puts it near, not on, its mark
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = t.prev[0]
        f = Film(p=cur, R=cur + [0, 0, HELD], Ro=GRIP, L=HOVER_BL, Lo=CLOSED)
        f.go(60, p=[*P_START[:2], HOVER_R - HELD], R=[*P_START[:2], HOVER_R])
        f.go(60, p=P_START + [0, 0, 0.0005], R=FEED_GRIP + [0, 0, 0.0005]).go(2, p=P_START).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(box((0.30, 0.20, L.TOP + H)), snap())  # knocked far, not to the start
        self.assertTrue(t.breach)

    def test_a_dropped_part_is_not_a_teleport(self):
        # Free fall from 0.3 m onto its mark: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = P_START + [0, 0, 0.3]
        t.update(box(p), snap())
        t.breach_steps = 1
        z, v = p[2], 0.0
        while z > P_START[2]:
            v += 9.81 * DT
            z = max(P_START[2], z - v * DT)
            t.update(box((p[0], p[1], z), vel=(0, 0, -v if z > P_START[2] else 0)), snap())
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.breach)
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def sample_plan(plan, hz=HZ, ready=None):
    """The scripted plan sampled on one clock: per arm, (tcp, opening) per control step, interpolating linearly
    between waypoints from the ready pose, finger joint values turned into rigstate openings. Both arms run for
    the longer plan's length; a finished arm holds its last target."""
    ready = ready or {"left": READY_L, "right": READY_R}
    total = max((sum(w.duration for w in wps) for wps in plan.values()), default=0.0)
    n = int(round(total * hz))
    out = {}
    for side in ("left", "right"):
        wps = plan.get(side, [])
        pos, fin = np.array(ready[side], float), -0.04695
        segs = []
        for w in wps:
            segs.append((pos, fin, np.array(w.pos, float), w.finger, w.duration))
            pos, fin = np.array(w.pos, float), w.finger
        rows = []
        for i in range(n + 1):
            t = i / hz
            cur = (pos, fin)
            acc = 0.0
            for p0, f0, p1, f1, d in segs:
                if t <= acc + d:
                    u = (t - acc) / d if d > 0 else 1.0
                    cur = ((1 - u) * p0 + u * p1, (1 - u) * f0 + u * f1)
                    break
                acc += d
            rows.append((cur[0], float(np.clip(cur[1] / -0.04695, 0.0, 1.0))))
        out[side] = rows
    return out, n


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, p=P_START):
        return {"part": box(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("part", names)
        self.assertIn("run_button_left", names)
        self.assertIn("run_button_right", names)
        self.assertIn("press_frame", names)
        self.assertIn("feed_mark", names)
        self.assertEqual(sum(n.startswith("rail_") for n in names), 4)
        self.assertEqual(sum(n.startswith("tray_wall") for n in names), 4)
        loose = [o for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["part"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")
        # the buttons are each other's arm's side, 42 cm apart; the nest floor is 36 mm between the rails' inner faces
        by = {o.name: o.pos[1] for o in self.task.OBJECTS if o.name.startswith("run_button")}
        self.assertGreater(by["run_button_left"], 0.2)
        self.assertLess(by["run_button_right"], -0.2)
        rails = {o.name: o for o in self.task.OBJECTS if o.name.startswith("rail_")}
        self.assertAlmostEqual(rails["rail_left"].pos[1] - rails["rail_left"].size[1] / 2, L.NEST_Y + 0.018)
        self.assertAlmostEqual(rails["rail_front"].pos[0] - rails["rail_front"].size[0] / 2, L.NEST_X + 0.018)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(left_tcp=PRESS_L, left_open=CLOSED, right_tcp=[L.NEST_X, L.NEST_Y, HOVER_R])  # the left pressed, the
        #                                                                                    right over the nest
        self.assertFalse(self.task.success(self.objects()))
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertFalse(tr.breach)
        self.assertEqual(self.task.score(self.objects(SEATED)), 0.25)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        # both press descents (the 'down to 2 cm above the button' moves) start on the same second
        li = next(i for i, w in enumerate(left) if "down to 2 cm above the button" in w.label)
        ri = next(i for i, w in enumerate(right) if "down to 2 cm above the button" in w.label)
        self.assertAlmostEqual(sum(w.duration for w in left[:li]), sum(w.duration for w in right[:ri]))
        # the same two moves, so the tips enter 'pressed' on the same step
        for k in (0, 1):
            self.assertEqual(left[li + k].duration, right[ri + k].duration)
            self.assertAlmostEqual(left[li + k].pos[2], right[ri + k].pos[2])
        # both hold at least the cycle's 2 s
        self.assertIn("hold the button", left[li + 2].label)
        self.assertIn("hold the button", right[ri + 2].label)
        self.assertGreaterEqual(left[li + 2].duration, L.CYCLE_S)
        self.assertGreaterEqual(right[ri + 2].duration, L.CYCLE_S)
        # the right lets go of the part in the nest and is away from it before the press descents start
        let_go = next(i for i, w in enumerate(right) if "let go in the nest" in w.label)
        self.assertLess(sum(w.duration for w in right[:let_go + 2]), sum(w.duration for w in right[:ri]))
        # the left lifts off first (its hold is shorter), the right 0.5 s later ...
        lr = next(i for i, w in enumerate(left) if "release the button" in w.label)
        rr = next(i for i, w in enumerate(right) if "release the button" in w.label)
        self.assertAlmostEqual(sum(w.duration for w in right[:rr]) - sum(w.duration for w in left[:lr]), 0.5)
        # ... and the right heads back toward the nest only once the left's release is complete
        self.assertIn("over the nest", right[rr + 1].label)
        self.assertGreaterEqual(sum(w.duration for w in right[:rr + 1]), sum(w.duration for w in left[:lr + 1]) - 1e-9)
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def run_plan(self, plan, settle_s=3.0):
        """Play the plan through the scorer with a kinematic proxy: the TCPs follow their targets exactly, the part
        follows a TCP while that arm's fingers are closed round it where it stands (the tips between its bottom and
        top) and drops straight down onto the surface below when they open (the feed mark's top, the nest's floor
        between the rails, or the table). Returns (last result, tracker, success step)."""
        rows, n = sample_plan(plan)
        n += int(settle_s * HZ)
        t = Tracker(HZ)
        p = P_START.copy()
        holder = None
        prev = p.copy()
        hold_steps, done = 0, None
        r = None

        def floor_under(p):
            if np.hypot(p[0] - L.PLATE_X, p[1] - L.PLATE_Y) < 0.02:
                return L.PLATE_TOP_Z
            if np.hypot(p[0] - L.NEST_X, p[1] - L.NEST_Y) < L.NEST_INNER / 2 - H + 1e-6:
                return L.TOP  # between the rails
            if abs(p[0] - L.NEST_X) < 0.026 + H and abs(p[1] - L.NEST_Y) < 0.026 + H:
                return L.RAIL_TOP_Z  # on a rail
            return L.TOP

        for i in range(n + 1):
            j = min(i, len(rows["left"]) - 1)
            (ltcp, lo), (rtcp, ro) = rows["left"][j], rows["right"][j]
            tcps = {"left": (ltcp, lo), "right": (rtcp, ro)}
            if holder is None:
                for side, (tcp, o) in tcps.items():
                    gap = o * L.FULL_GAP
                    at_part = np.hypot(*(tcp[:2] - p[:2])) < 0.006 and (p[2] - H) <= tcp[2] - TIPS <= (p[2] + H)
                    if at_part and gap <= L.PART_SIZE[1] + 0.0005:
                        holder = (side, tcp[2] - p[2])
            else:
                side, dz = holder
                if tcps[side][1] * L.FULL_GAP > L.PART_SIZE[1] + 0.002:
                    holder = None
            if holder is not None:
                side, dz = holder
                tcp = tcps[side][0]
                p = np.array([tcp[0], tcp[1], tcp[2] - dz])
            else:
                p = np.array([p[0], p[1], floor_under(p) + H])
            v = (p - prev) / DT
            prev = p.copy()
            r = t.update(box(p, vel=v), snap(ltcp, lo, rtcp, ro))
            hold_steps = hold_steps + 1 if r.success else 0
            if done is None and hold_steps >= int(self.task.SUCCESS_HOLD_S * HZ):
                done = i
        return r, t, done

    def test_demo_plan_scores_one_in_the_kinematic_proxy(self):
        plan = self.task.demo()
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        r, t, done = self.run_plan(plan)
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breach_steps, 0)
        self.assertTrue(t.cycled)
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(t.sync_presses, 1)
        self.assertEqual(r.details["double_gap_s"], 0.0)  # both tips enter the band on the same step
        self.assertGreaterEqual(r.details["longest_double_s"], L.CYCLE_S + 0.4)
        self.assertIsNotNone(done)
        self.assertLess(done, total * HZ)  # success holds before the plan ends (during the last 'up')
        self.assertLess(total + 3.0, 60.0)
        self.assertEqual(r.details["released"], {"left": True, "right": True})
        self.assertTrue(r.details["grippers_let_go"])
        self.assertEqual(t.arms_unavailable, 0)

    def test_one_arm_demos_score_quarter(self):
        plan = self.task.demo()
        r, t, _ = self.run_plan({"right": plan["right"]})  # DEMO_IDLE_ARM=left: the right alone
        self.assertEqual(r.score, 0.0)  # it loads, presses one button, then unloads the uncycled part to the tray
        self.assertEqual(t.double_presses, 0)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["part_in_tray"])
        # ... but up to its unload it was at 0.25 (seated, one button)
        right = plan["right"]
        ri = next(i for i, w in enumerate(right) if "over the nest, opening" in w.label)
        r, t, _ = self.run_plan({"right": right[:ri]}, settle_s=1.0)
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["part_seated"])
        r, t, _ = self.run_plan({"left": plan["left"]})  # DEMO_IDLE_ARM=right: the left alone presses its button
        self.assertEqual(r.score, 0.0)  # the part never left the feed mark
        self.assertFalse(t.breach)
        self.assertEqual(t.double_presses, 0)
        self.assertGreater(r.details["press_s"]["left"], 2.0)

    def test_demo_plan_with_the_presses_1s_apart_would_not_cycle(self):
        # A sanity check of the proxy itself: the left's wait 1 s longer makes the presses 1 s apart: 0.75 lost
        plan = self.task.demo()
        left = list(plan["left"])
        wi = next(i for i, w in enumerate(left) if "wait over the button" in w.label)
        w = left[wi]
        left[wi] = type(w)(w.duration + 1.0, w.pos, w.finger, w.label, w.yaw)
        r, t, _ = self.run_plan({"left": left, "right": plan["right"]})
        self.assertEqual(t.double_presses, 1)
        self.assertEqual(t.sync_presses, 0)
        self.assertFalse(t.cycled)
        self.assertEqual(r.score, 0.0)  # the part is unloaded to the tray uncycled
        self.assertEqual(r.details["double_gap_s"], 1.0)

    def test_demo_plan_with_the_right_unloading_early_would_breach(self):
        # The right's hold shortened by 2.5 s: it heads for the nest while the left is still pressing: 0
        plan = self.task.demo()
        right = list(plan["right"])
        hi = next(i for i, w in enumerate(right) if "hold the button" in w.label)
        w = right[hi]
        right[hi] = type(w)(w.duration - 2.5, w.pos, w.finger, w.label, w.yaw)
        r, t, _ = self.run_plan({"left": plan["left"], "right": right})
        self.assertTrue(t.breach)
        self.assertEqual(r.score, 0.0)


if __name__ == "__main__":
    unittest.main()
