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
    Tracker, board_clear, board_fully_out, board_on_deck, board_tilt_deg, finger_zone, let_go, moved,
    patient_flat_on_deck, pulled, raised, rotation, tilt_deg, under_share, valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
PH = L.PATIENT_SIZE[2] / 2  # half the patient's 40 mm height
BH = L.BOARD_SIZE[2] / 2  # half the board's 20 mm thickness
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.004  # tips this far above a held part's underside
HELD_P = TIPS + GRIP_UP - PH  # the TCP is this far above a held patient's centre (-3 mm: just below it)
HELD_B = TIPS + GRIP_UP - BH  # ... and above a held board's centre (+7 mm)
OPEN = 1.0  # rigstate opening: fully open
GRIP_P = (L.PATIENT_SIZE[0] - 0.008) / L.FULL_GAP  # closed on the patient's 70 mm (commanded 62 mm)
GRIP_B = (L.BOARD_SIZE[0] - 0.008) / L.FULL_GAP  # closed on the board's 40 mm (commanded 32 mm)

P_START = np.array([L.PATIENT_X, L.PATIENT_Y, L.PATIENT_BOTTOM_Z + PH])
B_START = np.array([L.BOARD_X, L.BOARD_Y, L.DECK_TOP + BH])
P_DECK = np.array([L.PATIENT_X, L.PATIENT_Y, L.DECK_TOP + PH])  # the patient resting flat on the deck
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
HOVER = L.TOP + 0.12
GRIP_Y0 = L.BOARD_Y - L.BOARD_SIZE[1] / 2 + 0.020  # where the right arm grips the board: 20 mm from its right end
PULL = 0.130
B_OUT = B_START + [0, -PULL, 0]  # the board after the demo's pull: it spans y -0.27..+0.05
UP_R = np.array([L.BOARD_X, GRIP_Y0 - PULL, L.TOP + 0.10])  # where the right hand parks afterwards


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.PATIENT_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def patient(center=P_START, quat=Q0, **kw):
    return box(center, quat, size=L.PATIENT_SIZE, **kw)


def board(center=B_START, quat=Q0, **kw):
    return box(center, quat, size=L.BOARD_SIZE, **kw)


def board_at(pull=0.0, dx=0.0):
    """The board flat on the deck, pulled ``pull`` along -y (and shifted ``dx`` along x) from its start."""
    return board(B_START + [dx, -pull, 0])


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("p", "pq", "b", "bq", "L", "Lo", "R", "Ro")
VEC = ("p", "b", "L", "R")


class Film:
    """A trial as a list of frames: the patient's and the board's centres and orientations, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"p": P_START, "pq": Q0, "b": B_START, "bq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
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
                if k in ("pq", "bq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def copy(self):
        out = Film()
        out.frames, out.cur = list(self.frames), dict(self.cur)
        return out


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vp = vb = np.zeros(3)
        else:
            vp, vb = (fr["p"] - prev["p"]) / DT, (fr["b"] - prev["b"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(patient(fr["p"], fr["pq"], vel=vp), board(fr["b"], fr["bq"], vel=vb), a)
        prev = fr
    return r, t


# -- the intended trial, in pieces -------------------------------------------------------------------------------
def left_grips_patient(f):
    """The left arm comes down over the patient, outside the board, and closes on its x faces (about 6 s)."""
    grip = P_START + [0, 0, HELD_P]
    f.go(30, L=grip + [0, 0, 0.08]).go(30, L=grip).go(30, Lo=GRIP_P)
    return f


def left_raises_patient(f, raise_m=0.012):
    """... and raises it straight up, clear of the board."""
    f.go(30, L=f.cur["L"] + [0, 0, raise_m], p=f.cur["p"] + [0, 0, raise_m])
    return f


def right_grips_board(f):
    """The right arm grips the board's right end across its width, the tips 4 mm above the deck (about 6 s)."""
    grip = np.array([L.BOARD_X, GRIP_Y0, L.DECK_TOP + BH + HELD_B])
    f.go(30, R=grip + [0, 0, 0.08]).go(30, R=grip).go(30, Ro=GRIP_B)
    return f


def right_pulls_board(f, pull=PULL, n=80, drag_patient=False):
    """... pulls it ``pull`` along -y at deck height. With ``drag_patient`` the patient rides along (friction),
    as it does when nothing holds it."""
    to = {"R": f.cur["R"] + [0, -pull, 0], "b": f.cur["b"] + [0, -pull, 0]}
    if drag_patient:
        to["p"] = f.cur["p"] + [0, -pull, 0]
    f.go(n, **to)
    return f


def right_lets_go(f):
    f.hold(10).go(30, Ro=OPEN).go(20, R=UP_R)
    return f


def left_sets_down(f, bottom_z=L.DECK_TOP, pq=Q0, open_to=OPEN, n=60):
    """The left arm lowers the patient until its underside is at ``bottom_z`` (the deck's top) over ``n`` steps,
    settles, opens to ``open_to`` and goes up."""
    p = np.array([f.cur["p"][0], f.cur["p"][1], bottom_z + PH])
    f.go(n, L=p + [0, 0, HELD_P], p=p, pq=pq).hold(10).go(30, Lo=open_to)
    if open_to >= L.GRIPPER_OPEN:
        f.go(20, L=p + [0, 0, 0.12])  # the hand goes up only once it has let go
    return f


def procedure(raise_m=0.012, pull=PULL, open_to=OPEN, settle=70, pq=Q0, bottom_z=L.DECK_TOP):
    """The intended trial: both arms grip, the left raises the patient, the right pulls the board out and lets go,
    the left sets the patient down on the deck and lets go; all still for ``settle`` steps."""
    f = Film().hold(10)
    left_grips_patient(f)
    right_grips_board(f)
    left_raises_patient(f, raise_m)
    f.hold(10)
    right_pulls_board(f, pull)
    right_lets_go(f)
    left_sets_down(f, bottom_z, pq, open_to)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 180.0, places=4)  # the patient has a top and a bottom
        self.assertAlmostEqual(board_tilt_deg(about("y", 180)), 0.0, places=4)  # the board is flat either way up
        self.assertAlmostEqual(board_tilt_deg(about("x", 170)), 10.0, places=4)

    def test_under_share_from_the_outlines(self):
        # at the start the board runs under the patient's whole 60 mm, and its +y end is 50 mm past the patient:
        # the first 50 mm of a straight pull change nothing, the next 60 mm bring the share from 1 to 0
        self.assertAlmostEqual(under_share(patient(), board()), 1.0, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.050)), 1.0, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.070)), 2 / 3, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.080)), 0.5, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.110)), 0.0, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(PULL)), 0.0, places=9)
        # pulled across (along x) instead: the patient overhangs the board by 15 mm each side, so the share falls
        # only past that; half out at 35 mm, out once the board's 40 mm has crossed the patient's 35 mm half-length
        self.assertAlmostEqual(under_share(patient(), board_at(0.0, dx=0.015)), 1.0, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.0, dx=0.035)), 0.5, places=9)
        self.assertAlmostEqual(under_share(patient(), board_at(0.0, dx=0.056)), 0.0, places=9)
        # the patient on the deck beside the board, at the same height: not 'under' either
        self.assertAlmostEqual(under_share(patient(P_DECK + [0.06, 0, 0]), board()), 0.0, places=9)
        self.assertAlmostEqual(pulled(board_at(0.030)), 0.030, places=9)
        self.assertAlmostEqual(pulled(board_at(-0.010)), -0.010, places=9)  # pushed further under

    def test_fully_out_needs_10mm_of_clearance(self):
        self.assertFalse(board_fully_out(patient(), board_at(0.110)))  # just clear, 0 mm
        self.assertAlmostEqual(board_clear(patient(), board_at(0.115)), 0.005, places=9)
        self.assertFalse(board_fully_out(patient(), board_at(0.119)))
        self.assertTrue(board_fully_out(patient(), board_at(0.120)))
        self.assertTrue(board_fully_out(patient(), board_at(PULL)))
        self.assertAlmostEqual(board_clear(patient(), board_at(PULL)), 0.020, places=9)

    def test_board_on_deck(self):
        self.assertTrue(board_on_deck(board()))
        self.assertTrue(board_on_deck(board_at(PULL)))  # its right end 10 mm over the deck's edge: still flat on it
        self.assertTrue(board_on_deck(board(B_OUT, about("x", 180))))  # turned over
        self.assertFalse(board_on_deck(board(B_OUT + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(board_on_deck(board(B_OUT + [0, 0, 0.004], about("x", 4))))  # tilted on the edge
        # its centre past the deck's edge (y -0.26), lying on the table 30 mm lower: not on the deck
        self.assertFalse(board_on_deck(board((L.BOARD_X, -0.30, L.TOP + BH))))

    def test_patient_flat_on_deck(self):
        self.assertTrue(patient_flat_on_deck(patient(P_DECK)))
        self.assertTrue(patient_flat_on_deck(patient(P_DECK + [0, 0, 0.0025])))
        self.assertFalse(patient_flat_on_deck(patient(P_DECK + [0, 0, 0.004])))  # held 4 mm above it
        self.assertFalse(patient_flat_on_deck(patient(P_DECK, about("x", 6))))  # tilted
        self.assertFalse(patient_flat_on_deck(patient()))  # on the board, 20 mm up
        self.assertFalse(patient_flat_on_deck(patient((L.PATIENT_X, 0.25, L.TOP + PH))))  # off the deck's end

    def test_raised_and_moved(self):
        self.assertAlmostEqual(raised(patient()), 0.0, places=9)
        self.assertAlmostEqual(raised(patient(P_START + [0, 0, 0.012])), 0.012, places=9)
        self.assertAlmostEqual(raised(patient(P_DECK)), -0.020, places=9)
        # tilted 5 degrees about its centre: the centre of its underside rises by 20 (1 - cos 5) = 0.08 mm
        self.assertAlmostEqual(raised(patient(P_START, about("x", 5))), PH * (1 - math.cos(math.radians(5))), places=9)
        self.assertAlmostEqual(moved(patient(P_START + [0.009, -0.012, 0])), 0.015, places=9)

    def test_finger_zone_covers_the_pads(self):
        a = snap(left_tcp=(0.2, 0.2, 0.9), left_open=OPEN)["left"]
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)
        closed = snap(left_tcp=(0.2, 0.2, 0.9), left_open=0.0)["left"]
        lo, hi = np.array([0.25, 0.15, 0.88]), np.array([0.30, 0.20, 0.90])
        self.assertAlmostEqual(zone_gap(closed, lo, hi), 0.05 - zone_radius(closed), places=6)

    def test_let_go(self):
        p = patient(P_DECK)
        held = snap(left_tcp=P_DECK + [0, 0, HELD_P], left_open=GRIP_P)["left"]
        self.assertFalse(let_go(held, p.low, p.high))
        nearly = snap(left_tcp=P_DECK + [0, 0, HELD_P], left_open=0.85)["left"]  # 80 mm apart, round a 70 mm block
        self.assertFalse(let_go(nearly, p.low, p.high))
        opened = snap(left_tcp=P_DECK + [0, 0, HELD_P], left_open=0.9)["left"]
        self.assertTrue(let_go(opened, p.low, p.high))
        away = snap(left_tcp=P_DECK + [0, 0.15, 0.0], left_open=0.0)["left"]  # closed, 15 cm away
        self.assertTrue(let_go(away, p.low, p.high))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertEqual(r.details["under_share"], 1.0)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.arms_unavailable, 0)
        self.assertAlmostEqual(t.max_raised, 0.012, places=6)
        self.assertLess(t.max_moved, 1e-9)
        self.assertLess(t.max_descent, L.DESCENT_MAX)
        self.assertTrue(r.details["grippers_let_go"])
        self.assertTrue(r.details["board_fully_out"] and r.details["board_on_deck"])

    def test_raise_within_30mm_scores_one(self):
        r, t = play(procedure(raise_m=0.0299))
        self.assertEqual(r.score, 1.0)
        r, t = play(procedure(raise_m=0.0301))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_raised", t.breaches)

    def test_a_gripper_still_holding_scores_075(self):
        r, _ = play(procedure(open_to=GRIP_P))  # the left still closed on the patient on the deck
        self.assertEqual((r.score, r.success), (0.75, False))
        r, _ = play(procedure(open_to=0.85))  # 80 mm apart, still round the 70 mm block: not let go
        self.assertEqual(r.score, 0.75)
        f = procedure(settle=0)  # the right still closed on the board, out on the deck
        f.go(30, R=B_OUT + [0, GRIP_Y0 - L.BOARD_Y, HELD_B]).go(30, Ro=GRIP_B).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["grippers_let_go"])

    def test_a_gripper_closed_away_from_its_item_has_let_go(self):
        f = procedure()
        f.go(20, Ro=0.0).hold(25)  # the right gripper closes up, parked 10 cm above the deck
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))
        f = procedure(settle=0)
        f.go(20, Lo=0.0).hold(45)  # the left closes 12 cm up over the patient
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_board_out_but_not_flat_on_the_deck_scores_075(self):
        f = procedure(settle=0)  # the board tilted up on the deck's edge afterwards
        f.go(20, b=B_OUT + [0, -0.05, 0.006], bq=about("x", 5)).hold(40)
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        f = procedure(settle=0)  # pulled right off the deck onto the table (30 mm lower): out, but not on the deck
        f.go(40, b=(L.BOARD_X, -0.32, L.TOP + BH)).hold(40)
        r, t = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["board_on_deck"])

    def test_board_out_patient_still_held_up_scores_half(self):
        f = Film().hold(10)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        right_pulls_board(f)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertTrue(r.details["board_fully_out"])
        self.assertFalse(r.details["patient_flat_on_deck"])

    def test_patient_set_down_tilted_scores_half(self):
        r, t = play(procedure(pq=about("x", 7)))  # one edge up on something: tilted 7 degrees on the deck
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_patient_set_down_4mm_above_the_deck_scores_half(self):
        # set down with its underside 4 mm above the deck (the scorer can see only that it is not resting there)
        r, _ = play(procedure(bottom_z=L.DECK_TOP + 0.004))
        self.assertEqual(r.score, 0.5)
        r, _ = play(procedure(bottom_z=L.DECK_TOP + 0.0025))
        self.assertEqual(r.score, 1.0)

    def test_half_out_scores_half(self):
        r, _ = play(procedure(pull=0.080))  # 30 of the patient's 60 mm still over the board
        self.assertEqual(r.score, 0.5)
        r, _ = play(procedure(pull=0.109))
        self.assertEqual(r.score, 0.5)
        r, _ = play(procedure(pull=0.0799))
        self.assertEqual(r.score, 0.25)

    def test_just_out_without_10mm_of_clearance_scores_half(self):
        r, _ = play(procedure(pull=0.115))
        self.assertEqual(r.score, 0.5)
        r, _ = play(procedure(pull=0.120))
        self.assertEqual(r.score, 1.0)

    def test_pulled_2cm_scores_quarter(self):
        r, t = play(procedure(pull=0.020))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertFalse(t.breach)
        self.assertEqual(r.details["under_share"], 1.0)  # the board's end was 50 mm past the patient
        r, _ = play(procedure(pull=0.050))
        self.assertEqual(r.score, 0.25)
        r, _ = play(procedure(pull=0.019))
        self.assertEqual(r.score, 0.0)

    def test_progress_is_read_at_the_end(self):
        # pulled 90 mm (half out), then pushed back under: the end state counts, the whole trial for the rules
        f = procedure(pull=0.090, settle=0)
        f.go(30, b=f.cur["b"] + [0, 0.050, 0]).hold(30)  # 40 mm out now: 0.25
        r, t = play(f)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)
        self.assertAlmostEqual(t.min_under_share, 1 / 3, places=6)
        f.go(30, b=f.cur["b"] + [0, 0.040, 0]).hold(30)  # pushed all the way back: 0
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_patient_ends_off_the_deck_scores_zero(self):
        # the end-state rule on its own (getting there would break the 15 mm rule too): the patient's centre past
        # the deck's +y end, lying on the table
        r = Tracker(HZ).result(patient((L.PATIENT_X, 0.19, L.TOP + PH)), board(B_OUT), snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["patient_over_deck"])
        r = Tracker(HZ).result(patient((L.PATIENT_X, 0.17, L.DECK_TOP + PH)), board(B_OUT), snap())
        self.assertEqual(r.score, 1.0)  # centre over the deck, its +y face 20 mm past the edge: still on it

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, p=P_DECK + [0.004, 0, 0]).go(3, p=P_DECK))  # still sliding on the deck
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=P_DECK + [0, 0, 0.12], right_tcp=UP_R)
        r = tr.update(patient(P_DECK, vel=(0.02, 0, 0)), board(B_OUT), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(patient(P_DECK, spin=(0, 0, 1.0)), board(B_OUT), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(patient(P_DECK), board(B_OUT, vel=(0, -0.03, 0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(patient(P_DECK), board(B_OUT), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        last = (patient(P_DECK), board(B_OUT), snap(left_tcp=P_DECK + [0, 0, 0.12], right_tcp=UP_R))
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class Breaches(unittest.TestCase):
    """Each patient rule broken once scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn(key, t.breaches)
        self.assertEqual(r.details["first_breach_step"], min(t.breaches.values()))
        return t

    def test_patient_moved_over_15mm(self):
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        # the left arm drifts 16 mm along y while holding, then back, and the rest is done right
        f.go(20, L=f.cur["L"] + [0, 0.016, 0], p=f.cur["p"] + [0, 0.016, 0])
        f.go(20, L=f.cur["L"] - [0, 0.016, 0], p=f.cur["p"] - [0, 0.016, 0])
        right_pulls_board(f)
        right_lets_go(f)
        left_sets_down(f)
        t = self.assert_zero(f.hold(60), "patient_moved")
        self.assertAlmostEqual(t.max_moved, 0.016, places=6)
        # 14 mm of drift is within the rule
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        f.go(20, L=f.cur["L"] + [0.014, 0, 0], p=f.cur["p"] + [0.014, 0, 0])
        right_pulls_board(f)
        right_lets_go(f)
        left_sets_down(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(t.breach)

    def test_patient_raised_over_30mm(self):
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f, 0.040)  # lifted 40 mm, then brought back down to 12 mm and the rest done right
        f.go(20, L=f.cur["L"] - [0, 0, 0.028], p=f.cur["p"] - [0, 0, 0.028])
        right_pulls_board(f)
        right_lets_go(f)
        left_sets_down(f)
        t = self.assert_zero(f.hold(60), "patient_raised")
        self.assertAlmostEqual(t.max_raised, 0.040, places=6)

    def test_patient_tilted_over_10deg(self):
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        f.go(10, pq=about("y", 12)).go(10, pq=Q0)  # rocked 12 degrees in the grip for half a second
        right_pulls_board(f)
        right_lets_go(f)
        left_sets_down(f)
        t = self.assert_zero(f.hold(60), "patient_tilted")
        self.assertAlmostEqual(t.max_tilt, 12.0, places=4)
        # a 9 degree rock is within the rule (and the raised edge is still within 30 mm)
        f = procedure(settle=0)
        g = Film()
        g.frames = f.frames[:100] + [dict(fr, pq=about("y", 9)) for fr in f.frames[100:110]] + f.frames[110:]
        g.cur = dict(f.frames[-1])
        r, t = play(g.hold(60))
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(t.max_tilt, 9.0, places=4)

    def test_patient_dropped(self):
        # set down right, then the last 20 mm let fall: 20 mm in one control step is 0.4 m/s
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        right_pulls_board(f)
        right_lets_go(f)
        f.go(30, Lo=OPEN).go(1, p=P_DECK + [0, 0, 0.012]).go(1, p=P_DECK).go(20, L=P_DECK + [0, 0, 0.12])
        t = self.assert_zero(f.hold(60), "patient_dropped")
        self.assertGreater(t.max_descent, 0.3)

    def test_a_fast_drop_is_caught_by_the_displacement_even_with_zero_reported_velocity(self):
        t = Tracker(HZ)
        play(Film().hold(5), t)
        a = snap()
        t.update(patient(P_START + [0, 0, 0.012]), board(), a)
        r = t.update(patient(P_START - [0, 0, 0.008]), board(), a)  # 20 mm lower in one step, velocity reported 0
        self.assertIn("patient_dropped", t.breaches)
        self.assertAlmostEqual(t.max_descent, 0.4, places=6)
        self.assertEqual(r.score, 0.0)

    def test_a_slow_set_down_is_not_a_drop(self):
        t = Tracker(HZ)
        a = snap()
        for z in np.linspace(P_START[2] + 0.012, P_DECK[2], 40):  # 32 mm over 2 s
            r = t.update(patient((P_START[0], P_START[1], z), vel=(0, 0, -0.016)), board(B_OUT), a)
        self.assertNotIn("patient_dropped", t.breaches)
        self.assertAlmostEqual(t.max_descent, 0.0164, places=3)
        self.assertEqual((r.score, r.success), (1.0, False))  # down on the deck, let go, but still moving

    def test_patient_off_the_table(self):
        f = Film().hold(5).go(30, p=[0.30, 0.70, L.TOP - 0.40]).hold(20)  # knocked off the table's edge
        self.assert_zero(f, "patient_off_table")

    def test_board_off_the_table(self):
        f = procedure(settle=5)
        f.go(30, b=[L.BOARD_X, -0.70, L.TOP - 0.40]).hold(20)  # the board pulled right off the table afterwards
        self.assert_zero(f, "board_off_table")

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5)
        left_grips_patient(f)
        left_raises_patient(f, 0.035)  # 35 mm up for a moment, then set back on the board
        f.go(30, L=f.cur["L"] - [0, 0, 0.035], p=f.cur["p"] - [0, 0, 0.035]).go(30, Lo=OPEN).go(20, L=READY_L)
        g = Film()
        g.frames = f.frames + procedure(settle=60).frames[1:]
        r, t = play(g)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(sorted(t.breaches), ["patient_raised"])
        self.assertEqual(r.details["first_breach_step"], t.breaches["patient_raised"])


class Exploits(unittest.TestCase):
    def test_pulling_with_the_patient_on_the_board_drags_it(self):
        # the right arm alone (DEMO_IDLE_ARM=left): the patient rides along on the board (friction) and goes 140 mm
        f = Film().hold(5)
        right_grips_board(f)
        right_pulls_board(f, drag_patient=True)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_moved", t.breaches)
        self.assertAlmostEqual(t.max_moved, PULL, places=6)
        self.assertEqual(r.details["under_share"], 1.0)  # and the board is still under it

    def test_pulling_2cm_with_the_patient_on_the_board_scores_zero(self):
        # a short pull without holding: the patient comes along 20 mm (breach) and the board is still under it
        f = Film().hold(5)
        right_grips_board(f)
        right_pulls_board(f, pull=0.020, n=20, drag_patient=True)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_moved", t.breaches)
        # 10 mm without holding: no breach, but the board is still under the whole patient: 0
        f = Film().hold(5)
        right_grips_board(f)
        right_pulls_board(f, pull=0.010, n=10, drag_patient=True)
        right_lets_go(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertEqual(r.details["under_share"], 1.0)

    def test_raising_the_patient_and_setting_it_back_scores_zero(self):
        # the left arm alone (DEMO_IDLE_ARM=right): raise, hold, set it back down on the board
        f = Film().hold(5)
        left_grips_patient(f)
        left_raises_patient(f)
        f.hold(100)
        left_sets_down(f, bottom_z=L.PATIENT_BOTTOM_Z)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertEqual(r.details["under_share"], 1.0)

    def test_holding_without_pulling_scores_zero(self):
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f)
        r, t = play(f.hold(200))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)

    def test_doing_nothing_scores_zero(self):
        r, t = play(Film().hold(400))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)

    def test_pulling_1cm_scores_zero_2cm_quarter(self):
        r, _ = play(procedure(pull=0.010))
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(pull=0.020))
        self.assertEqual(r.score, 0.25)

    def test_lifting_the_patient_off_and_aside_scores_zero(self):
        # carry the patient 10 cm up and 12 cm aside onto the deck, pull the board freely, set it back
        f = Film().hold(5)
        left_grips_patient(f)
        up = P_START + [0, 0, 0.10]
        f.go(30, L=up + [0, 0, HELD_P], p=up)
        aside = np.array([L.PATIENT_X - 0.06, L.PATIENT_Y - 0.10, L.DECK_TOP + PH])
        f.go(30, L=aside + [0, 0, 0.10 + HELD_P], p=aside + [0, 0, 0.10]).go(30, L=aside + [0, 0, HELD_P], p=aside)
        right_grips_board(f)
        right_pulls_board(f)
        right_lets_go(f)
        f.go(30, L=aside + [0, 0, 0.10 + HELD_P], p=aside + [0, 0, 0.10])
        f.go(30, L=P_DECK + [0, 0, 0.10 + HELD_P], p=P_DECK + [0, 0, 0.10])
        left_sets_down(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_raised", t.breaches)
        self.assertIn("patient_moved", t.breaches)

    def test_lifting_the_patient_within_30mm_and_carrying_it_aside_scores_zero(self):
        # the same within the raise limit: a 25 mm lift carried 8 cm along the deck still moves the patient
        f = Film().hold(5)
        left_grips_patient(f)
        left_raises_patient(f, 0.025)
        f.go(40, L=f.cur["L"] + [0, 0.08, 0], p=f.cur["p"] + [0, 0.08, 0])
        right_grips_board(f)
        right_pulls_board(f)
        right_lets_go(f)
        f.go(40, L=f.cur["L"] - [0, 0.08, 0], p=f.cur["p"] - [0, 0.08, 0])
        left_sets_down(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(sorted(t.breaches), ["patient_moved"])

    def test_pushing_the_patient_off_the_board_scores_zero(self):
        # shoved 50 mm along the board's length onto the deck: it moves and tips off the board's end
        f = Film().hold(5)
        tipped = P_START + [0, 0.050, -0.010]
        f.go(10, L=P_START + [0, -0.05, 0.02]).go(20, p=tipped, pq=about("x", -20))
        f.go(10, p=[L.PATIENT_X, 0.16, L.DECK_TOP + PH], pq=Q0)
        right_grips_board(f)
        right_pulls_board(f)
        right_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_moved", t.breaches)
        self.assertIn("patient_tilted", t.breaches)

    def test_tablecloth_pull_drops_the_patient(self):
        # the board jerked out in one control step: the patient (at most a few mm along) falls 20 mm to the deck
        f = Film().hold(5)
        right_grips_board(f)
        f.go(1, R=f.cur["R"] + [0, -PULL, 0], b=B_OUT, p=P_START + [0, -0.005, 0])
        f.go(1, p=P_DECK + [0, -0.005, 0])
        right_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_dropped", t.breaches)
        self.assertNotIn("patient_moved", t.breaches)

    def test_blocking_the_patient_instead_of_lifting_it(self):
        # the left pad holds the patient back by its +y face while the board slides out under it: it stays put,
        # but as the board's end passes under it, it tips over the edge and comes down to the deck
        f = Film().hold(5)
        f.go(20, L=P_START + [0, 0.04, 0.0])  # the open fingers pressed against the patient's +y face
        right_grips_board(f)
        right_pulls_board(f, pull=0.080, n=60)  # the board's end at the patient's centre line
        # tipping about the board's edge: 20 mm down over a 30 mm lever is 34 degrees, then flat on the deck,
        # all within a few control steps (a 20 mm fall takes 64 ms)
        f.go(2, p=P_START + [0, 0, -0.010], pq=about("x", -34)).go(2, p=P_DECK, pq=Q0)
        right_pulls_board(f, pull=PULL - 0.080, n=40)
        right_lets_go(f)
        f.go(20, L=READY_L)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("patient_tilted", t.breaches)
        self.assertIn("patient_dropped", t.breaches)

    def test_lifting_the_board_out_from_under_a_held_patient_is_allowed(self):
        # not an exploit: with the patient held, the board is lifted out (not slid) and laid on the deck
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f, 0.012)
        f.go(60, R=f.cur["R"] + [0, -0.11, 0], b=f.cur["b"] + [0, -0.11, 0])  # slid 110 mm, just clear
        f.go(20, R=f.cur["R"] + [0, 0, 0.03], b=f.cur["b"] + [0, 0, 0.03])  # lifted 30 mm
        f.go(30, R=f.cur["R"] + [0, -0.03, 0], b=f.cur["b"] + [0, -0.03, 0])  # carried on
        f.go(20, R=f.cur["R"] - [0, 0, 0.03], b=f.cur["b"] - [0, 0, 0.03])  # laid down on the deck
        right_lets_go(f)
        left_sets_down(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(t.breach)

    def test_pulling_the_board_out_across_instead_of_along_is_allowed(self):
        # the board slid along x out from under the held patient (it stays on the 200 mm deck)
        f = Film().hold(5)
        left_grips_patient(f)
        right_grips_board(f)
        left_raises_patient(f, 0.012)
        f.go(60, R=f.cur["R"] + [0.070, 0, 0], b=f.cur["b"] + [0.070, 0, 0])
        right_lets_go(f)
        left_sets_down(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["board_clear_mm"], 15.0)
        self.assertEqual(r.details["pulled_mm"], 0.0)  # not along the board, yet fully out


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'both let go' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        r, _ = play(procedure(open_to=GRIP_P), arms=False)  # even a closed gripper cannot be seen
        self.assertEqual(r.score, 1.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(procedure(settle=5), t, arms=False)
        r = t.update(patient(P_DECK), board(B_OUT), snap(left_tcp=P_DECK + [0, 0, HELD_P], left_open=GRIP_P))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])
        self.assertEqual(t.arms_unavailable, len(procedure(settle=5).frames))

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(patient(), board(), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(patient(), board(), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(patient(), board(), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_board_either_face_up(self):
        f = procedure(settle=0)
        f.go(1, bq=about("y", 180)).hold(60)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = patient(P_DECK)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=P_DECK)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), t.max_moved, t.max_raised, t.max_tilt, t.max_descent, t.min_under_share,
                      t.arms_unavailable)
            a = snap(left_tcp=P_DECK + [0, 0, 0.12], right_tcp=UP_R)
            for p, b in ((patient(P_DECK), bad), (bad, board(B_OUT)), (bad, bad)):
                r = t.update(p, b, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.result(p, b, a).score, 0.0)
            after = (dict(t.breaches), t.max_moved, t.max_raised, t.max_tilt, t.max_descent, t.min_under_share,
                     t.arms_unavailable)
            self.assertEqual(after, before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before, with no false drop
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash_and_details_are_json(self):
        odd = [patient((5.0, -3.0, -1.0)), patient((0.3, 0.0, 0.0), about("x", 120)),
               patient((-0.2, 0.6, L.TOP), about("y", 90)), patient(P_DECK, about("x", 89.9)),
               patient((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0)), patient(P_START, about("x", 180))]
        boards = [board((0.3, 0.0, 2.0)), board(B_OUT, about("y", 90)), board((-1, -1, -1), about("z", 30)),
                  board(B_START, about("x", 180)), board((0.3, 0.3, L.TOP + 0.5), vel=(0, 0, -9)), board()]
        t = Tracker(HZ)
        a = snap()
        for p, b in zip(odd + odd[::-1], boards + boards[::-1]):
            r = t.update(p, b, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
            json.dumps(r.details, allow_nan=False)
        r = t.update(patient(), board(), None)
        self.assertEqual(r.score, 0.0)
        json.dumps(r.details, allow_nan=False)
        json.dumps(Tracker(HZ).result(patient(), board(), None).details, allow_nan=False)

    def test_a_teleport_elsewhere_counts_as_a_move_not_a_drop_of_the_record(self):
        t = Tracker(HZ)
        play(Film().hold(5), t)
        r = t.update(patient((0.25, 0.0, L.DECK_TOP + PH)), board(), snap())  # knocked far: a breach, no reset
        self.assertIn("patient_moved", t.breaches)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def breached(self):
        """A trial with the patient lifted 40 mm and carried 12 cm along the deck (two breaches), still held."""
        f = Film().hold(5)
        left_grips_patient(f)
        left_raises_patient(f, 0.040)
        f.go(40, L=f.cur["L"] + [0, 0.12, 0], p=f.cur["p"] + [0, 0.12, 0])
        _, t = play(f.hold(5))
        self.assertEqual(sorted(t.breaches), ["patient_moved", "patient_raised"])
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(patient(), board(), snap())  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.max_moved, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_after_the_demo_resets(self):
        t = Tracker(HZ)
        play(procedure(settle=5), t)
        t.update(patient(), board(), snap())
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.min_under_share, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        # a seeded reset (object_jitter, +-3 cm) puts things back near, not exactly on, their marks
        t = self.breached()
        r = t.update(patient(P_START + [0.02, -0.02, 0]), board(B_START + [-0.03, 0.01, 0]), snap())
        self.assertFalse(t.breach)
        self.assertEqual(r.details["moved_mm"], 0.0)  # 'moved' counts from where it lay at the first step
        self.assertEqual(r.details["pulled_mm"], 0.0)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        p0 = t.prev[0][0]
        f = Film(p=p0, L=p0 + [0, 0, HELD_P], Lo=GRIP_P)
        f.go(60, L=P_START + [0, 0, HELD_P + 0.04], p=P_START + [0, 0, 0.04])  # carried back over its start
        f.go(80, L=P_START + [0, 0, HELD_P], p=P_START).go(30, Lo=OPEN).go(20, L=READY_L).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(patient((0.25, 0.0, L.DECK_TOP + PH)), board(B_OUT), snap())
        self.assertTrue(t.breach)

    def test_a_dropped_patient_landing_on_its_start_is_not_a_teleport(self):
        # free fall from 0.3 m back onto the board: fast, but its velocity explains it; the raise (and the drop)
        # stay on the record
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = P_START + [0, 0, 0.3]
        t.update(patient(p), board(), snap())
        self.assertIn("patient_raised", t.breaches)
        z, v = p[2], 0.0
        while z > P_START[2]:
            v += 9.81 * DT
            z = max(P_START[2], z - v * DT)
            t.update(patient((p[0], p[1], z), vel=(0, 0, -v if z > P_START[2] else 0)), board(), snap())
        self.assertTrue(t.breach)
        self.assertIn("patient_dropped", t.breaches)

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

    def objects(self, p=P_START, b=B_START):
        return {"bed_deck": box((L.DECK_X, L.DECK_Y, L.TOP + L.DECK_SIZE[2] / 2), size=L.DECK_SIZE),
                "slide_board": board(b), "patient": patient(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {"bed_deck", "slide_board", "patient"})
        self.assertEqual(self.task.SCENE, "station")
        self.assertEqual(self.task.SCORE_OVER, "end")
        by = {o.name: o for o in self.task.OBJECTS}
        self.assertIsNone(by["bed_deck"].mass)
        self.assertAlmostEqual(by["patient"].pos[2] - by["patient"].size[2] / 2, L.PATIENT_BOTTOM_Z)
        self.assertAlmostEqual(by["slide_board"].pos[2] - by["slide_board"].size[2] / 2, L.DECK_TOP)
        self.assertAlmostEqual(by["bed_deck"].pos[2] + by["bed_deck"].size[2] / 2, L.DECK_TOP)
        # the patient lies on the board: its outline inside the board's along y, over it along x
        self.assertGreaterEqual(by["patient"].pos[1] - by["patient"].size[1] / 2,
                                by["slide_board"].pos[1] - by["slide_board"].size[1] / 2)
        self.assertLessEqual(by["patient"].pos[1] + by["patient"].size[1] / 2,
                             by["slide_board"].pos[1] + by["slide_board"].size[1] / 2)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects(p=P_START + [0, 0, 0.04])))  # raised 40 mm
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
        # the end state through the hooks
        rigstate.offline_reset()
        snap(left_tcp=P_DECK + [0, 0, 0.12], right_tcp=UP_R)
        self.assertTrue(self.task.success(self.objects(p=P_DECK, b=B_OUT)))
        self.assertEqual(self.task.score(self.objects(p=P_DECK, b=B_OUT)), 1.0)

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

    def test_trace_writes_json_lines(self):
        import tempfile
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                rigstate.offline_reset()
                snap()
                self.task.success(self.objects())
                self.task.success(self.objects())
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [1, 2])
        self.assertIn("max_raised_mm", rows[0])
        self.assertIn("left_tcp", rows[0])

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        span = lambda wps: sum(w.duration for w in wps)  # noqa: E731
        # the right arm starts its pull only once the patient is up (plus 0.5 s)
        i_pull = next(i for i, w in enumerate(right) if "pull" in w.label)
        i_up = next(i for i, w in enumerate(left) if "raise" in w.label)
        self.assertGreaterEqual(span(right[:i_pull]), span(left[:i_up + 1]) + 0.5 - 1e-9)
        # the left arm starts lowering the patient only once the pull has settled (plus 0.5 s)
        i_lower = next(i for i, w in enumerate(left) if "lower" in w.label)
        i_settled = next(i for i, w in enumerate(right) if "settle" in w.label and i > i_pull)
        self.assertGreaterEqual(span(left[:i_lower]), span(right[:i_settled + 1]) + 0.5 - 1e-9)
        # the patient is held where it lies: every left waypoint is over the patient's start
        for w in left:
            self.assertAlmostEqual(w.pos[0], L.PATIENT_X)
            self.assertAlmostEqual(w.pos[1], L.PATIENT_Y)
        # the raise is commanded under the 30 mm limit, and the pull brings the board 10 mm clear with margin
        self.assertLess(self.task.RAISE, L.RAISE_MAX)
        board_end_y = L.BOARD_Y + L.BOARD_SIZE[1] / 2 - self.task.PULL  # the board's +y end after the pull
        patient_face_y = L.PATIENT_Y - L.PATIENT_SIZE[1] / 2
        self.assertGreaterEqual(patient_face_y - board_end_y, 2 * L.OUT_CLEAR)
        end = B_START + [0, -self.task.PULL, 0]
        self.assertTrue(board_fully_out(patient(), board(end)))
        self.assertTrue(board_on_deck(board(end)))
        self.assertGreaterEqual(board_clear(patient(), board(end)), 2 * L.OUT_CLEAR)
        # success can hold for SUCCESS_HOLD_S within the plan plus 3 s: the left opens to 0.9 about 1.05 s into
        # its release
        i_let_go = next(i for i, w in enumerate(left) if "let go" in w.label)
        gap0, gap1 = -2 * left[i_let_go - 1].finger, -2 * left[i_let_go].finger
        t_open = span(left[:i_let_go]) + left[i_let_go].duration * (0.9 * L.FULL_GAP - gap0) / (gap1 - gap0)
        self.assertLess(t_open + self.task.SUCCESS_HOLD_S, span(left) + 3.0)
        # ... and the hold outlasts the whole release and the rise, so the trial cannot end with the pads still
        # beside the patient
        self.assertGreaterEqual(self.task.SUCCESS_HOLD_S, left[i_let_go].duration + left[i_let_go + 1].duration)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(set(self.task.demo()), {"left"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM", None)


if __name__ == "__main__":
    unittest.main()
