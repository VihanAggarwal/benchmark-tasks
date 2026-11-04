"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
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
    Tracker, board_tilt_deg, cup_tilt_deg, in_box, lowest_bare_corner, on_bare_bench, on_mat, on_stud,
    outline_gap, overhang, rotation, valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
H = L.BOARD_SIZE[2] / 2  # half a board's 20 mm thickness
CH = L.CUP_HEIGHT / 2  # half the cup's height
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.005  # tips this far above a held board's bottom
HELD = TIPS + GRIP_UP - H  # a held board's centre is this far below the TCP
CUP_UP = 0.025  # tips this far above a held cup's bottom
CUP_HELD = TIPS + CUP_UP - CH  # a held cup's centre is this far below the TCP
OPEN = 1.0
GRIP = (L.BOARD_SIZE[1] - 0.008) / L.FULL_GAP  # rigstate opening closed on a 60 mm board (commanded 52 mm)
CUP_GRIP = (2 * L.CUP_RADIUS - 0.008) / L.FULL_GAP
PINCH = 0.016 / L.FULL_GAP  # commanded 16 mm on the 20 mm stud
ON_STUD_SIM = 0.020 / L.FULL_GAP  # what the simulator reads: the fingers stop at the stud's 20 mm

LB0 = np.array([L.BOX_X, L.BOX_Y["left"], L.TOP + H])
RB0 = np.array([L.BOX_X, L.BOX_Y["right"], L.TOP + H])
C0 = np.array([L.CUP_X, L.CUP_Y, L.MAT_TOP + CH])
PARKED = np.array([L.PARK_X, L.PARK_Y, L.PARK_TOP + CH])
MAT_L = np.array([L.MAT_X, 0.05, L.MAT_TOP + H])
MAT_R = np.array([L.MAT_X, -0.05, L.MAT_TOP + H])
STUD_GRIP = np.array([L.STUD_X, L.STUD_Y, L.STUD_TOP - 0.015 + TIPS])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
CARRY = L.TOP + 0.09
HOVER = L.TOP + 0.13
ASIDE_L = np.array([0.30, 0.25, L.TOP + 0.10])
UP_R = np.array([L.MAT_X, -0.05, L.TOP + 0.10])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.BOARD_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def cup(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like upright cylinder (its bounding box from the turned axis and end discs)."""
    c = np.array(center, float)
    ax = rotation(quat)[:, 2]
    ext = np.abs(ax) * CH + L.CUP_RADIUS * np.sqrt(np.clip(1 - ax ** 2, 0, 1))
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


VEC = ("lb", "rb", "c", "L", "R")
QUAT = ("lbq", "rbq", "cq")
KEYS = VEC + QUAT + ("Lo", "Ro")


class Film:
    """A trial as a list of frames: both boards' and the cup's centres and orientations, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"lb": LB0, "rb": RB0, "c": C0, "lbq": Q0, "rbq": Q0, "cq": Q0, "L": READY_L, "R": READY_R,
                    "Lo": OPEN, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in QUAT:
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def then(self, other):
        """Append another film's frames (its first frame, a repeat of where it starts, is dropped)."""
        self.frames = self.frames + other.frames[1:]
        self.cur = dict(other.cur)
        return self


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        v = {k: np.zeros(3) if prev is None else (fr[k] - prev[k]) / DT for k in ("lb", "rb", "c")}
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(box(fr["lb"], fr["lbq"], v["lb"]), box(fr["rb"], fr["rbq"], v["rb"]),
                     cup(fr["c"], fr["cq"], v["c"]), a)
        prev = fr
    return r, t


# -- the steps of a trial -------------------------------------------------------------------------------------
def tcp_key(side):
    return "L" if side == "left" else "R"


def ground(f, side, opening=PINCH, at=STUD_GRIP):
    """The gripper goes down round the stud, closes on it, opens, goes up (about 5 s)."""
    k = tcp_key(side)
    at = np.array(at, float)
    f.go(30, **{k: at + [0, 0, 0.06]}).go(20, **{k: at}).go(20, **{k + "o": opening})
    f.go(20, **{k + "o": OPEN}).go(15, **{k: at + [0, 0, 0.06]})
    return f


def right_moves_cup(f, to=PARKED):
    """The right arm takes the cup off the mat and stands it at ``to`` (the yellow square), then goes up."""
    grip = C0 + [0, 0, CUP_HELD]
    to = np.array(to, float)
    f.go(30, R=[*grip[:2], HOVER]).go(20, R=grip).go(20, Ro=CUP_GRIP)
    f.go(20, R=[*grip[:2], HOVER], c=[*C0[:2], HOVER - CUP_HELD])
    f.go(40, R=[*to[:2], HOVER], c=[*to[:2], HOVER - CUP_HELD])
    f.go(20, R=to + [0, 0, CUP_HELD + 0.002], c=to + [0, 0, 0.002]).go(2, c=to)
    f.go(20, Ro=OPEN).go(15, R=[*to[:2], HOVER])
    return f


def pick(f, side):
    """Down into the box beside the board, grip it, lift it clear of the walls."""
    k = tcp_key(side)
    b = "lb" if side == "left" else "rb"
    start = LB0 if side == "left" else RB0
    grip = start + [0, 0, HELD]
    f.go(30, **{k: grip + [0, 0, 0.08]}).go(20, **{k: grip}).go(20, **{k + "o": GRIP})
    f.go(20, **{k: [*grip[:2], CARRY], b: [*start[:2], CARRY - HELD]})
    return f


def place(f, side, at=None, open_to=OPEN, q=Q0, retreat=None):
    """Carry the held board over ``at`` on the mat, lower it to 2 mm above, let it drop, open, go up."""
    k = tcp_key(side)
    b = "lb" if side == "left" else "rb"
    at = np.array(at if at is not None else (MAT_L if side == "left" else MAT_R), float)
    f.go(30, **{k: [*at[:2], CARRY], b: [*at[:2], CARRY - HELD]})
    f.go(30, **{k: at + [0, 0, HELD + 0.002], b: at + [0, 0, 0.002]})
    f.go(2, **{b: at, b + "q": q}).go(20, **{k + "o": open_to})
    if retreat is not None:
        f.go(20, **{k: retreat})
    return f


def procedure(at_l=None, at_r=None, open_l=OPEN, open_r=OPEN, settle=30, q_l=Q0, q_r=Q0):
    """The intended trial, in sequence: the right arm parks the cup; the left grounds, the right grounds; the left
    unpacks its board onto the mat and moves aside; the right unpacks its board; all still for ``settle`` steps."""
    f = Film().hold(10)
    right_moves_cup(f)
    ground(f, "left")
    ground(f, "right")
    pick(f, "left")
    place(f, "left", at_l, open_l, q_l, retreat=ASIDE_L if open_l >= 0.9 else None)
    pick(f, "right")
    place(f, "right", at_r, open_r, q_r, retreat=UP_R if open_r >= 0.9 else None)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_board_but_not_for_the_cup(self):
        self.assertAlmostEqual(board_tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(board_tilt_deg(about("y", 180)), 0.0, places=4)  # upside down: the same box
        self.assertAlmostEqual(board_tilt_deg(about("x", 170)), 10.0, places=4)
        self.assertAlmostEqual(board_tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(cup_tilt_deg(about("x", 20)), 20.0, places=4)
        self.assertAlmostEqual(cup_tilt_deg(about("y", 180)), 180.0, places=4)  # an upside-down cup is spilled
        self.assertAlmostEqual(cup_tilt_deg(about("z", 90)), 0.0, places=4)

    def test_finger_zone_is_a_flat_ended_cylinder(self):
        b = box(MAT_L)
        top = MAT_L[2] + H
        above = snap(left_tcp=(MAT_L[0], MAT_L[1], top + 0.030 + TIPS))["left"]  # tips 30 mm over the board
        self.assertAlmostEqual(zone_gap(above, b), 0.030, places=6)  # not 30 mm minus the 57 mm radius
        self.assertAlmostEqual(zone_radius(above), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)
        side = snap(left_tcp=(MAT_L[0], MAT_L[1] + 0.03 + 0.07, MAT_L[2] + HELD))["left"]  # axis 70 mm off its side
        self.assertAlmostEqual(zone_gap(side, b), 0.07 - zone_radius(side), places=6)
        diag = snap(left_tcp=(MAT_L[0], MAT_L[1] + 0.03 + 0.08, top + 0.02 + TIPS))["left"]  # 20 mm up, 80 mm out
        self.assertAlmostEqual(zone_gap(diag, b), math.hypot(0.02, 0.08 - zone_radius(diag)), places=6)
        held = snap(left_tcp=MAT_L + [0, 0, HELD], left_open=GRIP)["left"]
        self.assertLess(zone_gap(held, b), 0.0)
        closed = snap(left_tcp=(MAT_L[0], MAT_L[1] + 0.03 + 0.03, MAT_L[2] + HELD), left_open=0.0)["left"]
        self.assertAlmostEqual(zone_gap(closed, b), 0.03 - math.hypot(0.008, 0.0135), places=6)
        # the zone runs 38 mm up from the tips: a box beside the gripper 30 mm above the tips is beside the fingers
        hi = box((MAT_L[0], MAT_L[1] + 0.08, MAT_L[2] + 0.04))
        g = snap(left_tcp=(MAT_L[0], MAT_L[1], MAT_L[2] + 0.04 - 0.03 + TIPS), left_open=OPEN)["left"]
        self.assertAlmostEqual(zone_gap(g, hi), 0.05 - zone_radius(g), places=6)

    def test_closing_on_the_stud_grounds(self):
        on = snap(left_tcp=STUD_GRIP, left_open=PINCH)["left"]
        self.assertTrue(on_stud(on))
        self.assertTrue(on_stud(snap(left_tcp=STUD_GRIP, left_open=ON_STUD_SIM)["left"]))  # as the sim reads it
        self.assertFalse(on_stud(snap(left_tcp=STUD_GRIP, left_open=OPEN)["left"]))  # open round it: no contact
        self.assertFalse(on_stud(snap(left_tcp=STUD_GRIP, left_open=0.5)["left"]))  # 47 mm apart: not on it
        self.assertTrue(on_stud(snap(left_tcp=STUD_GRIP + [0.014, 0, 0], left_open=PINCH)["left"]))
        self.assertFalse(on_stud(snap(left_tcp=STUD_GRIP + [0.016, 0, 0], left_open=PINCH)["left"]))
        # finger tips only 4 mm below the stud's top: not round it
        self.assertFalse(on_stud(snap(left_tcp=(L.STUD_X, L.STUD_Y, L.STUD_TOP - 0.004 + TIPS),
                                      left_open=PINCH)["left"]))
        self.assertTrue(on_stud(snap(left_tcp=(L.STUD_X, L.STUD_Y, L.STUD_TOP - 0.005 + TIPS),
                                     left_open=PINCH)["left"]))

    def test_in_box(self):
        self.assertTrue(in_box(box(LB0)) and in_box(box(RB0)))
        self.assertTrue(in_box(box(RB0 + [0, 0, 0.004])))  # lifted 4 mm inside the walls: still in
        self.assertFalse(in_box(box([*RB0[:2], L.BOX_RIM_Z + H + 0.001])))  # its bottom above the rim: out
        self.assertFalse(in_box(box(MAT_R)))
        self.assertFalse(in_box(box((L.BOX_X, 0.0, L.TOP + H))))

    def test_bare_bench(self):
        self.assertFalse(on_bare_bench(box(LB0)))  # in its box: the box's floor
        self.assertFalse(on_bare_bench(box(MAT_L)))  # flat on the mat
        self.assertIsNone(lowest_bare_corner(box(MAT_L)))  # every corner is over the mat
        self.assertFalse(on_bare_bench(box((MAT_L[0], MAT_L[1], L.TOP + H + 0.0005))))  # sunk 2.5 mm into the mat
        self.assertTrue(on_bare_bench(box((0.30, 0.20, L.TOP + H))))  # on the table between the mat and the box
        self.assertFalse(on_bare_bench(box((0.30, 0.20, L.TOP + H + 0.002))))  # held 2 mm above it
        self.assertTrue(on_bare_bench(box((0.30, 0.20, L.TOP - 0.3))))  # fallen off the table
        # The yellow parking square is bench too: a board resting on it (centred, all corners off the square) counts.
        self.assertTrue(on_bare_bench(box((L.PARK_X, L.PARK_Y + 0.05, L.PARK_TOP + H))))
        self.assertFalse(on_bare_bench(box((L.PARK_X, L.PARK_Y + 0.05, L.PARK_TOP + H + 0.006))))  # held 6 mm above
        # flat on the mat, overhanging its edge by 10 mm: the overhanging corners are 3 mm up in the air
        over = box((L.MAT_X, L.MAT_HALF_Y - 0.03 + 0.01, L.MAT_TOP + H))
        self.assertFalse(on_bare_bench(over))
        self.assertAlmostEqual(lowest_bare_corner(over), L.MAT_SIZE[2], places=6)
        # half off the mat's side edge, tipped so its outer long edge rests on the table: touching the bare bench
        tipped = box((L.MAT_X, L.MAT_HALF_Y, L.TOP + 0.0115), about("x", -2.9))
        self.assertTrue(on_bare_bench(tipped))
        self.assertLess(lowest_bare_corner(tipped), 0.0005)
        # dragged over the box's wall: one end on the table outside the box
        lean = box((L.BOX_X, 0.32 - L.BOX_INNER_HALF - 0.04, L.TOP + 0.014), about("x", 20))
        self.assertTrue(on_bare_bench(lean))

    def test_on_mat_tolerances(self):
        self.assertTrue(on_mat(box(MAT_L), L.OVERHANG_TIGHT))
        self.assertEqual(overhang(box(MAT_L)), 0.0)
        edge = box((L.MAT_X, L.MAT_HALF_Y - 0.03 + 0.010, L.MAT_TOP + H))  # 10 mm past the mat's side edge
        self.assertAlmostEqual(overhang(edge), 0.010, places=6)
        self.assertFalse(on_mat(edge, L.OVERHANG_TIGHT))
        self.assertTrue(on_mat(edge, L.OVERHANG_LOOSE))
        far = box((L.MAT_X + L.MAT_HALF_X - 0.06 + 0.016, 0.0, L.MAT_TOP + H))  # 16 mm past the front edge
        self.assertFalse(on_mat(far, L.OVERHANG_LOOSE))
        self.assertFalse(on_mat(box(MAT_L + [0, 0, 0.004]), L.OVERHANG_LOOSE))  # held 4 mm above it
        self.assertFalse(on_mat(box(MAT_L, about("x", 6)), L.OVERHANG_LOOSE))  # tilted
        self.assertTrue(on_mat(box(MAT_L, about("y", 180)), L.OVERHANG_TIGHT))  # either face down
        self.assertTrue(on_mat(box(MAT_L, about("z", 10)), L.OVERHANG_TIGHT))  # turned 10 deg: still wholly on it
        self.assertFalse(on_mat(box(MAT_L, about("z", 30)), L.OVERHANG_TIGHT))  # turned 30 deg: a corner 6 mm over

    def test_cup_distances(self):
        lifted = box([*RB0[:2], L.BOX_RIM_Z + H + 0.001])  # the right board just out of its box
        self.assertGreater(outline_gap(cup(PARKED), lifted), L.CUP_DISTANCE + 0.03)  # 33 cm
        self.assertLess(outline_gap(cup(C0), lifted), L.CUP_DISTANCE)  # the cup where it starts: 22 cm
        self.assertGreater(outline_gap(cup(PARKED), box(MAT_R)), 0.45)
        self.assertGreater(outline_gap(cup(C0), box(LB0)), L.CUP_DISTANCE)  # 31.6 cm: only the right board is near
        self.assertEqual(outline_gap(cup(MAT_R), box(MAT_R)), 0.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["cup_rule_kept"])

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.breaches, {})
        self.assertIsNone(t.cup_rule_step)
        self.assertEqual(t.groundings, {"left": 1, "right": 1})
        self.assertEqual(t.arms_unavailable, 0)
        self.assertGreater(t.min_cup_to_out_board, L.CUP_DISTANCE + 0.03)
        self.assertGreater(min(t.min_ungrounded_to_board.values()), 0.10)  # no ungrounded gripper near a board
        self.assertLess(t.first_ground_step["left"], t.first_out_step["left"])
        self.assertEqual(r.details["boards_apart_mm"], 40.0)

    def test_ground_first_then_cup_then_ground_again_scores_one(self):
        # The right arm grounds, moves the cup (which charges it again), grounds again, then unpacks: fine.
        f = Film().hold(5)
        ground(f, "right")
        right_moves_cup(f)
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.cup_contacts, {"left": 0, "right": 1})
        self.assertEqual(t.groundings, {"left": 1, "right": 2})

    def test_overhang_scores_075(self):
        r, _ = play(procedure(at_r=MAT_R + [0, -0.03, 0]))  # 10 mm past the mat's right edge
        self.assertEqual((r.score, r.success), (0.75, False))
        r, _ = play(procedure(at_l=MAT_L + [0.064, 0, 0]))  # 14 mm past the front edge
        self.assertEqual(r.score, 0.75)
        r, t = play(procedure(at_l=MAT_L + [0.067, 0, 0]))  # 17 mm past it: on the mat no longer
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_boards_too_close_scores_075(self):
        r, _ = play(procedure(at_l=MAT_L - [0, 0.035, 0], at_r=MAT_R + [0, 0.035, 0]))  # touching
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(at_l=MAT_L - [0, 0.012, 0], at_r=MAT_R + [0, 0.012, 0]))  # 16 mm apart
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(at_l=MAT_L - [0, 0.009, 0], at_r=MAT_R + [0, 0.009, 0]))  # 22 mm apart
        self.assertEqual(r.score, 1.0)

    def test_gripper_still_holding_a_board_scores_075(self):
        r, _ = play(procedure(open_r=GRIP))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["grippers_let_go"])
        r, _ = play(procedure(open_r=0.85))  # 80 mm apart, still round the board: not let go
        self.assertEqual(r.score, 0.75)

    def test_a_gripper_closed_away_from_the_boards_has_let_go(self):
        f = procedure(settle=0)
        f.go(20, Ro=0.0).hold(25)  # the right gripper closes 8 cm above its board
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(r.details["grippers_let_go"])
        self.assertFalse(r.details["grippers_open"])

    def test_one_arm_alone_scores_half(self):
        # The right arm alone: parks the cup, grounds, unpacks its board. The left board stays in its box.
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "right")
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_both_boards_down_with_the_cup_left_on_the_mat_scores_half(self):
        f = Film().hold(10)
        right_moves_cup(f, to=(0.15, -0.20, L.TOP + CH))  # off the mat, but only 20 cm behind it
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        self.assertIsNotNone(t.cup_rule_step)
        self.assertGreater(t.cup_rule_step, t.first_out_step["left"])  # the left board came within 30 cm of it

    def test_one_board_down_with_the_cup_rule_broken_scores_zero(self):
        f = Film().hold(10)
        ground(f, "left")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)  # the cup still on the mat, 5 cm from the left board's spot
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertFalse(r.details["cup_rule_kept"])

    def test_ready_to_unpack_scores_quarter(self):
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "left")
        ground(f, "right")
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["grounded"], {"left": True, "right": True})
        # a board lifted out and held over its box, cup parked: still ready, not yet on the mat
        r, _ = play(pick(f, "left").hold(10))
        self.assertEqual(r.score, 0.25)

    def test_not_ready_scores_zero(self):
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "left")  # only one gripper grounded
        r, _ = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        f = Film().hold(10)
        ground(f, "left")
        ground(f, "right")  # both grounded, the cup still on the mat (22 cm from the right board)
        r, _ = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        f = Film().hold(10)
        ground(f, "right")
        right_moves_cup(f)  # grounded first, then the cup charged it again
        ground(f, "left")
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["grounded"], {"left": True, "right": False})

    def test_ready_after_the_cup_rule_was_broken_scores_zero(self):
        # The left board was taken out toward the mat while the cup still stood there, then put back in its box;
        # the cup is parked and both grippers grounded afterwards: the broken cup rule is not undone.
        f = Film().hold(10)
        ground(f, "left")
        pick(f, "left")
        f.go(30, L=[0.33, 0.12, CARRY], lb=[0.33, 0.12, CARRY - HELD])  # 17 cm from the cup
        f.go(30, L=[*LB0[:2], CARRY], lb=[*LB0[:2], CARRY - HELD]).go(20, L=LB0 + [0, 0, HELD + 0.001], lb=LB0)
        f.go(20, Lo=OPEN).go(20, L=ASIDE_L)
        right_moves_cup(f)
        ground(f, "right")
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertEqual(r.details["grounded"], {"left": True, "right": True})
        self.assertFalse(r.details["cup_rule_kept"])
        self.assertTrue(r.details["in_box"]["left"])

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(Film().then(f).go(3, rb=MAT_R + [0.004, 0, 0]).go(3, rb=MAT_R))  # still sliding
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(Film().then(f).go(3, rb=MAT_R + [0.004, 0, 0]).go(3, rb=MAT_R).hold(1))  # stopped
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=ASIDE_L, right_tcp=UP_R)
        r = tr.update(box(MAT_L, vel=(0.02, 0, 0)), box(MAT_R), cup(PARKED), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(MAT_L), box(MAT_R, spin=(0, 0, 1.0)), cup(PARKED), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(MAT_L), box(MAT_R), cup(PARKED, vel=(0.05, 0, 0)), a)  # the cup need not be still
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        last = (box(MAT_L), box(MAT_R), cup(PARKED), snap(left_tcp=ASIDE_L, right_tcp=UP_R))
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class Breaches(unittest.TestCase):
    """Each rule broken once scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn(key, t.breaches)
        return t

    def test_touching_a_board_before_grounding(self):
        f = Film().hold(5)
        right_moves_cup(f)
        pick(f, "left")  # the left gripper takes its board without grounding first
        f.go(20, L=[*LB0[:2], L.TOP + HELD], lb=LB0).go(20, Lo=OPEN).go(20, L=READY_L)  # puts it back
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        t = self.assert_zero(f.hold(30), "left_ungrounded_touch")
        self.assertLess(t.min_ungrounded_to_board["left"], 0.0)

    def test_touching_a_board_after_the_cup_without_grounding_again(self):
        f = Film().hold(5)
        ground(f, "right")
        right_moves_cup(f)  # charges the right gripper again
        ground(f, "left")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        t = self.assert_zero(f.hold(30), "right_ungrounded_touch")
        self.assertEqual(t.cup_contacts["right"], 1)
        self.assertNotIn("left_ungrounded_touch", t.breaches)

    def test_open_fingers_beside_a_board_count_as_touching(self):
        # ungrounded, its axis 50 mm from the board's side at the board's height: the open pads reach 56.6 mm
        f = Film().hold(5).go(30, L=[L.BOX_X, 0.32 - 0.03 - 0.05, L.TOP + 0.02 + TIPS]).go(20, L=READY_L).hold(5)
        t = self.assert_zero(f, "left_ungrounded_touch")
        self.assertLess(t.min_ungrounded_to_board["left"], 0.0)
        # closed, the same place: 50 mm off its side, zone 15.7 mm: clear
        f = Film().hold(5).go(10, Lo=0.0).go(30, L=[L.BOX_X, 0.32 - 0.03 - 0.05, L.TOP + 0.02 + TIPS])
        r, t = play(f.go(20, L=READY_L).hold(5))
        self.assertNotIn("left_ungrounded_touch", t.breaches)

    def test_hovering_over_a_board_is_not_touching(self):
        # ungrounded and open, finger tips 20 mm above the board's top: 20 mm away (more than 5 mm)
        over = [L.BOX_X, 0.32, L.TOP + 0.02 + 0.02 + TIPS]
        r, t = play(Film().hold(5).go(30, L=over).hold(10).go(20, L=READY_L))
        self.assertFalse(t.breach)
        self.assertAlmostEqual(t.min_ungrounded_to_board["left"], 0.020, places=6)
        # 4 mm above it: touching
        over = [L.BOX_X, 0.32, L.TOP + 0.02 + 0.004 + TIPS]
        self.assert_zero(Film().hold(5).go(30, L=over).go(20, L=READY_L), "left_ungrounded_touch")

    def test_a_board_set_on_the_bench(self):
        f = Film().hold(5)
        right_moves_cup(f)
        ground(f, "left")
        pick(f, "left")
        spot = np.array([0.30, 0.18, L.TOP + H])  # the bare table between the box and the mat
        f.go(20, L=spot + [0, 0, HELD + 0.05], lb=spot + [0, 0, 0.05]).go(20, L=spot + [0, 0, HELD], lb=spot)
        f.go(20, Lo=OPEN).go(20, Lo=GRIP).go(20, L=spot + [0, 0, HELD + 0.06], lb=spot + [0, 0, 0.06])  # regrip
        place(f, "left", retreat=ASIDE_L)
        self.assert_zero(f.hold(30), "left_board_on_bench")

    def test_a_board_dragged_out_over_the_wall(self):
        f = Film().hold(5)
        right_moves_cup(f)
        ground(f, "right")
        grip = RB0 + [0, 0, HELD]
        f.go(30, R=grip).go(20, Ro=GRIP)
        f.go(20, R=grip + [0, 0.09, 0.03], rb=RB0 + [0, 0.09, 0.03])  # pulled up over the wall
        f.go(20, R=grip + [-0.05, 0.16, 0], rb=RB0 + [-0.05, 0.16, 0])  # and slid along the table
        place(f, "right", retreat=UP_R)
        self.assert_zero(f.hold(20), "right_board_on_bench")

    def test_a_board_dropped_on_its_edge(self):
        f = Film().hold(5)
        right_moves_cup(f)
        ground(f, "left")
        pick(f, "left")
        f.go(30, L=[*MAT_L[:2], CARRY], lb=[*MAT_L[:2], CARRY - HELD]).go(10, Lo=OPEN)
        f.go(4, lb=[MAT_L[0], MAT_L[1], L.MAT_TOP + 0.03], lbq=about("x", 90))  # lands on its long edge, on the mat
        t = self.assert_zero(f.hold(20), "left_board_dropped")
        self.assertNotIn("left_board_on_bench", t.breaches)

    def test_cup_knocked_over(self):
        f = Film().hold(5).go(6, c=C0 + [0, -0.01, -0.01], cq=about("x", 90)).hold(10)  # tipped on its side
        f.then(procedure())
        self.assert_zero(f, "cup_knocked_over")

    def test_cup_off_the_table(self):
        f = Film().hold(5)
        f.go(20, R=C0 + [0, 0.06, CUP_HELD])
        f.go(40, R=[L.CUP_X, -0.62, C0[2] + CUP_HELD], c=[L.CUP_X, -0.62 - 0.05, C0[2] - 0.003])  # swept sideways
        f.go(10, c=[L.CUP_X, -0.72, L.TOP - 0.40], R=READY_R)  # over the edge, onto the floor
        f.then(Film(c=[L.CUP_X, -0.72, L.TOP - 0.40]).then(procedure()))
        t = self.assert_zero(f, "cup_off_table")
        self.assertIsNone(t.cup_rule_step)  # the floor is far enough: the zero is from the cup leaving the table

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5).go(20, R=RB0 + [0, 0, HELD]).go(20, R=READY_R)  # the ungrounded right gripper at its board
        f.then(procedure(settle=60))
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(t.breaches["right_ungrounded_touch"], min(t.breaches.values()))
        self.assertEqual(r.details["first_breach_step"], t.breaches["right_ungrounded_touch"])


class Exploits(unittest.TestCase):
    def test_hovering_round_the_stud_without_closing_does_not_ground(self):
        f = Film().hold(5)
        right_moves_cup(f)
        ground(f, "left", opening=OPEN)  # down round the stud, fingers left open
        ground(f, "right", opening=0.5)  # half closed: 47 mm apart round a 20 mm stud
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.groundings, {"left": 0, "right": 0})
        self.assertIn("left_ungrounded_touch", t.breaches)
        self.assertIn("right_ungrounded_touch", t.breaches)

    def test_closing_beside_the_stud_does_not_ground(self):
        f = Film().hold(5)
        right_moves_cup(f)
        ground(f, "left", at=STUD_GRIP + [0.0, 0.03, 0.0])  # closed 30 mm to the side of the stud's axis
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertIn("left_ungrounded_touch", t.breaches)

    def test_shoving_the_cup_off_the_table_scores_zero(self):
        f = Film().hold(5)
        f.go(20, R=C0 + [0.06, 0, CUP_HELD]).go(30, R=[0.30, -0.62, C0[2] + CUP_HELD],
                                                 c=[0.30, -0.66, C0[2]])  # pushed sideways to the edge
        f.go(6, c=[0.30, -0.70, L.TOP - 0.3], cq=about("x", 60))  # and off it
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertTrue({"cup_off_table", "cup_knocked_over"} <= set(t.breaches))

    def test_cup_pushed_just_aside_on_the_mat_is_not_enough(self):
        # The cup slid 10 cm toward the front of the mat instead of parked: the right board comes within 30 cm.
        f = Film().hold(5)
        f.go(20, R=C0 + [-0.06, 0, CUP_HELD]).go(20, R=C0 + [0.04, 0, CUP_HELD], c=C0 + [0.10, 0, 0])
        f.go(20, R=READY_R)
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["cup_rule_kept"])

    def test_carrying_a_board_past_the_parked_cup_breaks_the_cup_rule(self):
        # Cup parked, but the right board is taken the long way round, past the back right corner.
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        f.go(30, R=[0.30, -0.45, CARRY], rb=[0.30, -0.45, CARRY - HELD])  # 7 cm past its box, toward the cup
        place(f, "right", retreat=UP_R)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertLess(t.min_cup_to_out_board, L.CUP_DISTANCE)

    def test_stacking_the_boards_scores_half(self):
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "left")
        ground(f, "right")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        pick(f, "right")
        place(f, "right", at=MAT_L + [0, 0, 2 * H], retreat=UP_R)  # on top of the left board
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        self.assertEqual(r.details["on_mat"], {"left": True, "right": False})

    def test_one_arm_takes_the_cup_to_the_other_corner_and_does_its_own_board(self):
        # Not an exploit: the left arm alone could park the cup in the back left corner and unpack its board: 0.5.
        f = Film().hold(10)
        grip = C0 + [0, 0, CUP_HELD]
        far = np.array([0.08, 0.60, L.TOP + CH])
        f.go(30, L=[*grip[:2], HOVER]).go(20, L=grip).go(20, Lo=CUP_GRIP)
        f.go(20, L=[*grip[:2], HOVER], c=[*C0[:2], HOVER - CUP_HELD])
        f.go(40, L=[*far[:2], HOVER], c=[*far[:2], HOVER - CUP_HELD]).go(20, L=far + [0, 0, CUP_HELD], c=far)
        f.go(20, Lo=OPEN).go(15, L=[*far[:2], HOVER])
        ground(f, "left")
        pick(f, "left")
        place(f, "left", retreat=ASIDE_L)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertIsNone(t.cup_rule_step)


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_not_applied_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'both let go' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        r, _ = play(procedure(open_r=GRIP), arms=False)  # even a closed gripper cannot be seen
        self.assertEqual(r.score, 1.0)
        f = Film().hold(10)
        right_moves_cup(f)
        ground(f, "left")
        ground(f, "right")
        r, t = play(f.hold(10), arms=False)  # groundings cannot be seen either: not ready
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.groundings, {"left": 0, "right": 0})

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(box(LB0), box(RB0), cup(C0), snap(right_tcp=RB0 + [0, 0, HELD]))
        self.assertIn("right_ungrounded_touch", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(box(LB0), box(RB0), cup(C0), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(box(LB0), box(RB0), cup(C0), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(box(LB0), box(RB0), cup(C0), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])

    def test_cup_contact_ungrounds_only_the_gripper_that_touched_it(self):
        t = Tracker(HZ)
        for side in ("left", "right"):
            kw = {f"{side}_tcp": STUD_GRIP, f"{side}_open": PINCH}
            t.update(box(LB0), box(RB0), cup(C0), snap(**kw))
        self.assertEqual(t.grounded, {"left": True, "right": True})
        t.update(box(LB0), box(RB0), cup(C0), snap(right_tcp=C0 + [0, 0, CUP_HELD], right_open=CUP_GRIP))
        self.assertEqual(t.grounded, {"left": True, "right": False})
        self.assertEqual(t.cup_contacts, {"left": 0, "right": 1})
        # an open gripper 2 cm above the cup's lid: not touching it
        t.update(box(LB0), box(RB0), cup(C0), snap(left_tcp=C0 + [0, 0, CH + 0.02 + TIPS]))
        self.assertTrue(t.grounded["left"])


class Robustness(unittest.TestCase):
    def test_either_face_down_and_a_small_turn(self):
        r, _ = play(procedure(q_l=about("y", 180), q_r=about("x", 180)))
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(q_l=about("z", 5), q_r=about("z", -5)))  # turned 5 degrees: outlines 30 mm apart
        self.assertEqual(r.score, 1.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = box(MAT_L)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=MAT_L)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), dict(t.grounded), t.cup_rule_step, t.min_cup_to_out_board, t.arms_unavailable)
            a = snap(left_tcp=ASIDE_L, right_tcp=UP_R)
            for lb, rb, c in ((bad, box(MAT_R), cup(PARKED)), (box(MAT_L), bad, cup(PARKED)),
                              (box(MAT_L), box(MAT_R), bad)):
                r = t.update(lb, rb, c, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(lb, rb, c, a).score, 0.0)
            after = (dict(t.breaches), dict(t.grounded), t.cup_rule_step, t.min_cup_to_out_board, t.arms_unavailable)
            self.assertEqual(after, before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(MAT_L, about("x", 89.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        cups = [cup((5.0, 3.0, -2.0)), cup(C0, about("x", 179)), cup(PARKED, about("y", 90)), cup((0.3, 0, 3.0))]
        t = Tracker(HZ)
        a = snap()
        for i, (lb, rb) in enumerate(zip(odd + odd[::-1], odd[::-1] + odd)):
            r = t.update(lb, rb, cups[i % len(cups)], a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(box(LB0), box(RB0), cup(C0), None)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def breached(self):
        f = Film().hold(5).go(20, L=LB0 + [0, 0, HELD])  # the ungrounded left gripper on its board
        f.then(procedure(settle=5))
        f.frames = f.frames[:300]
        _, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(box(LB0), box(RB0), cup(C0), snap())  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.grounded, {"left": False, "right": False})
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        # (a board has 10 mm of play along x in its box and 40 mm along y)
        t.update(box(LB0 + [0.008, -0.03, 0]), box(RB0 + [-0.009, 0.035, 0]), cup(C0 + [0.02, 0.03, 0]), snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        (lb, _), (rb, _), (c, _) = t.prev
        f = Film(lb=lb, rb=rb, c=c, L=ASIDE_L, R=UP_R)
        f.go(60, lb=[*lb[:2], CARRY]).go(60, lb=[*LB0[:2], CARRY]).go(60, lb=LB0 + [0, 0, 0.0005]).go(2, lb=LB0)
        f.go(60, rb=[*rb[:2], CARRY]).go(60, rb=[*RB0[:2], CARRY]).go(60, rb=RB0)
        f.go(60, c=[*c[:2], HOVER]).go(60, c=[*C0[:2], HOVER]).go(60, c=C0).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(box((0.25, 0.0, L.TOP + H)), box(RB0 + [0, 0.2, 0]), cup(C0), snap())  # knocked far, not home
        self.assertTrue(t.breach)

    def test_a_dropped_board_is_not_a_teleport(self):
        # Free fall from 0.3 m into its box: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = LB0 + [0, 0, 0.3]
        t.update(box(p), box(RB0), cup(C0), snap())
        t._flag("left_ungrounded_touch")
        z, v = p[2], 0.0
        while z > LB0[2]:
            v += 9.81 * DT
            z = max(LB0[2], z - v * DT)
            t.update(box((p[0], p[1], z), vel=(0, 0, -v if z > LB0[2] else 0)), box(RB0), cup(C0), snap())
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


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, lb=LB0, rb=RB0, c=C0):
        return {"left_board": box(lb), "right_board": box(rb), "foam_cup": cup(c)}

    def test_objects_and_text(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in ("left_board", "right_board", "foam_cup", "ground_stud", "esd_mat", "parking_square"):
            self.assertIn(n, names)
        self.assertEqual(self.task.SCORE_OVER, "end")
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        self.assertIn(self.task.INSTRUCTION, (HERE / "README.md").read_text())

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=RB0 + [0, 0, HELD])  # the ungrounded right gripper on its board
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
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_order_and_switches(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})

        def ends(wps):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                out.append((t - w.duration, t, w.label))
            return out

        left, right = ends(plan["left"]), ends(plan["right"])

        def when(seq, text, which=0):
            hits = [(a, b) for a, b, lab in seq if text in lab]
            return hits[which]

        # the left gripper is closed on the stud before it first comes down beside its board
        self.assertLess(when(left, "close on the stud")[1], when(left, "down into the box")[0])
        # the left board leaves its box only after the cup stands on the square (the right gripper has opened)
        self.assertGreaterEqual(when(left, "lift it out")[0], when(right, "let go of the cup")[1] - 1e-9)
        # the right gripper closes on the stud after it let go of the cup, before it goes down to its board
        self.assertGreater(when(right, "close on the stud")[0], when(right, "let go of the cup")[1])
        self.assertLess(when(right, "close on the stud")[1], when(right, "down into the box")[0])
        # one arm at the stud at a time; the left over the mat only once the right is off the stud; the right
        # over the mat only once the left is aside
        self.assertGreaterEqual(when(right, "over the ground stud")[0], when(left, "up off the stud")[1] - 1e-9)
        self.assertGreaterEqual(when(left, "carry it over the mat")[0], when(right, "up off the stud")[1] - 1e-9)
        self.assertGreaterEqual(when(right, "carry it over the mat")[0], when(left, "aside")[1] - 1e-9)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
