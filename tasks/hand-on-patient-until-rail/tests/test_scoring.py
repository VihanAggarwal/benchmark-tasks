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
    Tracker, centre_off, clearance, face_tilt_deg, flat_on_bed, grippers_released, hand_on, hands_on,
    heading_err_deg, held, let_go, low, outline_meets_deck, on_bed, on_table_off_platforms, rail_up, resting_on_bed, resting_on_trolley, rotation, tilt_deg,
    valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
PH = L.PATIENT_SIZE[2] / 2  # half the patient's 40 mm
RH = L.RAIL_SIZE[2] / 2  # half the rail's 40 mm
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.004  # tips this far above a held part's bottom
HELD_P = TIPS + GRIP_UP - PH  # the TCP is this far below a held patient's centre (-3 mm)
HELD_R = TIPS + GRIP_UP - RH  # ... and a held rail's
OPEN = 1.0  # rigstate opening: fully open
PGRIP = (L.PATIENT_SIZE[0] - 0.008) / L.FULL_GAP  # closed on the 70 mm patient (62 mm gap): 0.66
RGRIP = (L.RAIL_SIZE[1] - 0.008) / L.FULL_GAP  # closed on the 30 mm rail (22 mm gap): 0.234
CLOSED = 0.016 / L.FULL_GAP  # the narrowest commanded gap, 16 mm: 0.17

P_START = np.array([L.PATIENT_X, L.PATIENT_Y, L.TROLLEY_TOP + PH])
R_START = np.array([L.RAIL_X, L.RAIL_Y, L.TOP + RH])
P_ON_MARK = np.array([L.MARK_X, L.MARK_Y, L.MARK_TOP + PH])
R_UP = np.array([L.RAIL_PLACE_X, L.RAIL_PLACE_Y, L.DECK_TOP + RH])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
HOVER_L = L.TOP + 0.12
HOVER_R = L.TOP + 0.10
UP_R = np.array([L.RAIL_X, L.RAIL_Y, L.TOP + 0.12])
SET_GAP = 0.001  # the test films let go with a part this far above its surface (the demo commands 2 mm above the
#                  deck, 1 mm above the mark, for the patient, and 2 mm above the deck for the rail)


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


def patient(center, quat=Q0, **kw):
    return box(center, quat, size=L.PATIENT_SIZE, **kw)


def rail(center, quat=Q0, **kw):
    return box(center, quat, size=L.RAIL_SIZE, **kw)


def tilted_on(surface_z, x, y, quat):
    """The centre of a patient tilted by ``quat`` whose lowest corner rests on the surface at ``surface_z``."""
    ext = np.abs(rotation(quat)) @ (np.array(L.PATIENT_SIZE) / 2)
    return np.array([x, y, surface_z + ext[2]])


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


KEYS = ("p", "pq", "r", "rq", "L", "Lo", "R", "Ro")
VEC = ("p", "r", "L", "R")


class Film:
    """A trial as a list of frames: the patient's and the rail's centres and orientations, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"p": P_START, "pq": Q0, "r": R_START, "rq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
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
                if k in ("pq", "rq"):
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
            vp = vr = np.zeros(3)
        else:
            vp, vr = (fr["p"] - prev["p"]) / DT, (fr["r"] - prev["r"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(patient(fr["p"], fr["pq"], vel=vp), rail(fr["r"], fr["rq"], vel=vr), a)
        prev = fr
    return r, t


# -- the intended trial, piece by piece (both arms on one clock, no sag) -------------------------------------------
def left_picks(f):
    """The left arm comes over the patient, down beside them, closes on them and lifts them to the carry height
    (about 7 s)."""
    g = P_START + [0, 0, HELD_P]
    f.go(30, L=[*P_START[:2], HOVER_L]).go(20, L=g + [0, 0, 0.03]).go(30, L=g).go(10).go(30, Lo=PGRIP).go(10)
    up = np.array([*P_START[:2], HOVER_L])
    f.go(30, L=up, p=up - [0, 0, HELD_P])
    return f


def left_sets_down(f, at=P_ON_MARK):
    """The left arm carries the held patient over ``at`` and lowers them onto it: to 2 cm above, settle, the last
    2 cm slowly, settle (about 5.5 s). The patient ends 1 mm above the surface (2 mm above the deck on the mark),
    still held."""
    over = np.array([at[0], at[1], HOVER_L])
    f.go(40, L=over, p=over - [0, 0, HELD_P])
    entry = at + [0, 0, SET_GAP + 0.020]
    setd = at + [0, 0, SET_GAP]
    f.go(20, L=entry + [0, 0, HELD_P], p=entry).go(10).go(30, L=setd + [0, 0, HELD_P], p=setd).go(10)
    return f


def left_lets_go(f, at=P_ON_MARK, open_to=OPEN, up=True):
    """The left gripper opens (the patient drops its last 1 mm) and goes up."""
    f.go(6, Lo=0.8).go(2, p=at).go(22, Lo=open_to)
    if up:
        f.go(20, L=[at[0], at[1], HOVER_L])
    return f


def right_picks(f):
    """The right arm comes over the rail, down beside it, closes on it and lifts it (about 6.7 s)."""
    g = R_START + [0, 0, HELD_R]
    f.go(30, R=[*R_START[:2], HOVER_R]).go(20, R=g + [0, 0, 0.03]).go(30, R=g).go(10).go(30, Ro=RGRIP).go(10)
    up = np.array([*R_START[:2], HOVER_R])
    f.go(24, R=up, r=up - [0, 0, HELD_R])
    return f


def right_sets_down(f, at=R_UP, rq=None):
    """The right arm carries the held rail over ``at`` and lowers it: to 2 cm above, settle, the last 2 cm, settle
    (5 s). The rail ends 2 mm above the surface, still held."""
    over = np.array([at[0], at[1], HOVER_R])
    f.go(30, R=over, r=over - [0, 0, HELD_R], **({} if rq is None else {"rq": rq}))
    entry = at + [0, 0, 0.002 + 0.020]
    setd = at + [0, 0, 0.002]
    f.go(20, R=entry + [0, 0, HELD_R], r=entry).go(10).go(30, R=setd + [0, 0, HELD_R], r=setd).go(10)
    return f


def right_lets_go(f, at=R_UP, open_to=OPEN, up=True):
    """The right gripper opens (the rail drops its last 2 mm) and goes up and back out."""
    f.go(6, Ro=0.8).go(2, r=at).go(22, Ro=open_to)
    if up:
        f.go(20, R=UP_R)
    return f


def procedure(at=P_ON_MARK, rail_at=R_UP, open_to=OPEN, settle=40, order="patient_first", rq=None):
    """The intended trial. ``patient_first``: both pick, the left sets the patient down and keeps hold, the right
    brings the rail up and lets go, then the left lets go. ``rail_first``: the right brings the rail up and lets
    go first, then the left sets the patient down and lets go at once."""
    f = Film().hold(10)
    left_picks(f)
    right_picks(f)
    if order == "patient_first":
        left_sets_down(f, at)
        f.hold(10)
        right_sets_down(f, rail_at, rq)
        right_lets_go(f, rail_at)
        f.hold(20)  # the margin: the left keeps hold until the rail is up
        left_lets_go(f, at, open_to)
    else:
        right_sets_down(f, rail_at, rq)
        right_lets_go(f, rail_at)
        left_sets_down(f, at)
        left_lets_go(f, at, open_to)
    return f.hold(settle)


# -- geometry ----------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_tilts_and_heading(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 16)), 16.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 180)), 180.0, places=4)  # the patient turned over: not folded
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(face_tilt_deg(about("x", 180)), 0.0, places=4)  # the rail is the same either way up
        self.assertAlmostEqual(face_tilt_deg(about("y", 4)), 4.0, places=4)
        self.assertAlmostEqual(heading_err_deg(about("z", 4)), 4.0, places=4)
        self.assertAlmostEqual(heading_err_deg(about("z", 180)), 0.0, places=4)  # end for end: along x still
        self.assertAlmostEqual(heading_err_deg(about("z", 90)), 90.0, places=4)

    def test_surfaces_and_clearance(self):
        self.assertAlmostEqual(clearance(patient(P_START)), 0.0)
        self.assertAlmostEqual(clearance(patient(P_ON_MARK)), 0.001)  # on the 1 mm mark
        self.assertAlmostEqual(clearance(patient((0.28, 0.14, L.TOP + 0.03 + PH))), 0.030)  # bridging the gap: table
        self.assertAlmostEqual(clearance(patient(P_START + [0, 0, 0.05])), 0.05)
        self.assertTrue(low(patient(P_START)))
        self.assertFalse(low(patient(P_ON_MARK)))  # on the bed: never low (sliding there is allowed)
        self.assertFalse(low(patient(P_START + [0, 0, 0.011])))
        self.assertFalse(low(patient((0.28, 0.14, L.TOP + 0.03 + PH))))
        self.assertTrue(low(patient((0.28, 0.40, L.TOP + PH))))  # on the table

    def test_patient_places(self):
        self.assertTrue(resting_on_trolley(patient(P_START)))
        self.assertFalse(resting_on_trolley(patient(P_START + [0, 0, 0.004])))
        self.assertTrue(on_bed(patient(P_ON_MARK)))
        self.assertTrue(on_bed(patient(P_ON_MARK + [0, 0, 0.008])))  # 9 mm over the deck: on the bed (the rule)
        self.assertFalse(on_bed(patient(P_ON_MARK + [0, 0, 0.011])))
        self.assertTrue(on_bed(patient((0.19, -0.17, L.DECK_TOP + PH))))  # a corner of the deck
        self.assertTrue(on_bed(patient((0.28, 0.14, L.DECK_TOP + PH))))  # bridging the gap: half on the bed
        self.assertTrue(on_bed(patient((0.17, 0.05, L.DECK_TOP + PH))))  # hanging 25 mm over the near edge
        self.assertFalse(on_bed(patient((0.14, 0.05, L.DECK_TOP + PH))))  # outline 5 mm clear of the deck
        self.assertFalse(on_bed(patient((0.28, 0.14, L.DECK_TOP + 0.011 + PH))))  # bridging, 11 mm up
        self.assertTrue(outline_meets_deck(patient((0.28, 0.14, L.TOP + 0.3))))
        self.assertFalse(outline_meets_deck(patient(P_START)))
        self.assertTrue(resting_on_bed(patient(P_ON_MARK)))
        self.assertTrue(resting_on_bed(patient(P_ON_MARK + [0, 0, 0.0015])))  # held 1.5 mm up: resting (the band)
        self.assertFalse(resting_on_bed(patient(P_ON_MARK + [0, 0, 0.004])))  # 4 mm up: held, not resting
        self.assertTrue(resting_on_bed(patient(tilted_on(L.DECK_TOP, 0.35, -0.10, about("x", 10)), about("x", 10))))
        self.assertFalse(resting_on_bed(patient(tilted_on(L.DECK_TOP, 0.35, -0.10, about("x", 16)), about("x", 16))))
        self.assertTrue(flat_on_bed(patient(P_ON_MARK), L.CENTRE_TIGHT))
        self.assertTrue(flat_on_bed(patient(P_ON_MARK + [0.010, 0.010, 0]), L.CENTRE_TIGHT))  # 14 mm off
        self.assertFalse(flat_on_bed(patient(P_ON_MARK + [0.016, 0, 0]), L.CENTRE_TIGHT))
        self.assertTrue(flat_on_bed(patient(P_ON_MARK + [0.016, 0, 0]), L.CENTRE_LOOSE))
        self.assertFalse(flat_on_bed(patient(P_ON_MARK + [0.031, 0, 0]), L.CENTRE_LOOSE))
        self.assertFalse(flat_on_bed(patient(P_ON_MARK, about("x", 6)), L.CENTRE_TIGHT))  # tilted 6: not flat
        self.assertAlmostEqual(centre_off(patient(P_ON_MARK + [0.003, -0.004, 0])), 0.005)
        self.assertTrue(on_table_off_platforms(patient((0.28, 0.40, L.TOP + PH))))  # beside the trolley
        self.assertTrue(on_table_off_platforms(patient((0.45, 0.0, L.TOP + PH))))  # beyond the bed
        self.assertFalse(on_table_off_platforms(patient((0.28, 0.14, L.TOP + 0.03 + PH))))  # bridging the gap
        self.assertFalse(on_table_off_platforms(patient(P_ON_MARK)))
        self.assertTrue(on_table_off_platforms(patient((0.28, 0.40, L.TOP - 0.3))))  # fallen: also off the table

    def test_rail_up(self):
        self.assertTrue(rail_up(rail(R_UP)))
        self.assertTrue(rail_up(rail(R_UP + [0.007, 0.007, 0.002], about("z", 4))))
        self.assertTrue(rail_up(rail(R_UP, about("z", 180))))  # end for end
        self.assertTrue(rail_up(rail(R_UP, about("x", 180))))  # upside down: a plain bar
        self.assertFalse(rail_up(rail(R_UP + [0.011, 0, 0])))  # 11 mm off
        self.assertFalse(rail_up(rail(R_UP, about("z", 6))))  # turned 6 degrees
        self.assertFalse(rail_up(rail(R_UP + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(rail_up(rail(R_UP, about("y", 6))))  # tilted
        self.assertFalse(rail_up(rail(R_START)))
        self.assertFalse(rail_up(rail((L.RAIL_PLACE_X, 0.10, L.DECK_TOP + RH))))  # on the deck's left edge
        self.assertFalse(rail_up(rail((L.RAIL_PLACE_X, L.RAIL_PLACE_Y, L.TOP + RH))))  # on the table beside the bed

    def test_finger_zone_hands_and_let_go(self):
        a = arm((0.2, 0.2, 0.9), OPEN)
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        self.assertAlmostEqual(zone_radius(arm((0.2, 0.2, 0.9), 0.0)), math.hypot(0.008, 0.0135), places=6)
        p = patient(P_ON_MARK)
        lo, hi = p.low, p.high
        g = P_ON_MARK + [0, 0, HELD_P]
        self.assertFalse(let_go(arm(g, PGRIP), lo, hi))
        self.assertLess(zone_gap(arm(g, PGRIP), lo, hi), 0.0)
        self.assertTrue(let_go(arm(g, OPEN), lo, hi))
        self.assertTrue(let_go(arm(g, 0.9), lo, hi))
        self.assertFalse(let_go(arm(g, 0.89), lo, hi))
        self.assertTrue(let_go(arm(g + [0, 0, 0.10], PGRIP), lo, hi))  # closed 10 cm above: not holding
        # a hand on them: fingers closed to 16 mm, tips 5 mm above the patient's top (within 10 mm)
        press = np.array([L.MARK_X, L.MARK_Y, L.MARK_TOP + L.PATIENT_SIZE[2] + 0.005 + TIPS])
        self.assertTrue(hand_on(arm(press, CLOSED), lo, hi))
        self.assertFalse(let_go(arm(press, CLOSED), lo, hi))
        self.assertTrue(hand_on(arm(press + [0, 0, 0.005], CLOSED), lo, hi))  # tips 10 mm up: still a hand on them
        self.assertFalse(hand_on(arm(press + [0, 0, 0.007], CLOSED), lo, hi))  # 12 mm up: hovering, not a hand
        self.assertFalse(let_go(arm(press + [0, 0, 0.007], CLOSED), lo, hi))  # ... but near enough to be 'holding'
        self.assertTrue(hand_on(arm(g, PGRIP), lo, hi))
        self.assertFalse(hand_on(arm(g, 0.9), lo, hi))
        # closed fingers hovering beside the patient, the zone 5 mm clear of them: neither a hand on them nor let go
        beside = np.array([L.MARK_X, L.MARK_Y + 0.030 + 0.005 + zone_radius(arm(g, 0.5)), L.MARK_TOP + PH])
        self.assertFalse(hand_on(arm(beside, 0.5), lo, hi))
        self.assertFalse(let_go(arm(beside, 0.5), lo, hi))
        self.assertTrue(let_go(arm(beside + [0, 0.006, 0], 0.5), lo, hi))  # 11 mm clear: let go
        self.assertFalse(hand_on(arm(press, OPEN), lo, hi))  # open fingers hovering: no hand on them
        self.assertTrue(let_go(arm(press, OPEN), lo, hi))
        arms = {"left": arm(g, PGRIP), "right": arm(READY_R)}
        self.assertEqual(hands_on(arms, p), {"left": True, "right": False})
        self.assertTrue(held(arms, p))
        arms = {"left": arm(READY_L), "right": arm(press, CLOSED)}  # the right's hand counts too
        self.assertEqual(hands_on(arms, p), {"left": False, "right": True})
        self.assertTrue(held(arms, p))
        self.assertFalse(held({"left": arm(READY_L), "right": arm(READY_R)}, p))
        r = rail(R_UP)
        self.assertTrue(grippers_released({"left": arm(READY_L), "right": arm(READY_R)}, p, r))
        self.assertFalse(grippers_released({"left": arm(READY_L), "right": arm(R_UP + [0, 0, HELD_R], RGRIP)}, p, r))
        self.assertFalse(grippers_released({"left": arm(g, PGRIP), "right": arm(READY_R)}, p, r))
        # a gripper closed on the rail counts as holding the rail, not the patient
        arms = {"left": arm(READY_L), "right": arm(R_UP + [0, 0, HELD_R], RGRIP)}
        self.assertFalse(held(arms, p))


# -- tiers ----------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.unattended_steps, 0)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertTrue(t.lifted)
        self.assertTrue(r.details["rail_up"])
        self.assertTrue(r.details["flat_on_mark"])
        self.assertTrue(r.details["grippers_let_go"])
        self.assertGreater(r.details["held_on_bed_s"], 1.0)
        self.assertLess(r.details["max_descent_mps"], L.DROP_SPEED)
        self.assertLess(r.details["drag_mm"], 1.0)
        self.assertGreater(r.details["first_rail_up_step"], r.details["first_on_bed_step"])

    def test_rail_first_then_patient_scores_one(self):
        r, t = play(procedure(order="rail_first"))
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.details["held_on_bed_s"], 0.0)  # never on the bed with the rail down
        self.assertLess(r.details["first_rail_up_step"], r.details["first_on_bed_step"])

    def test_just_inside_the_centre_limit(self):
        r, _ = play(procedure(at=P_ON_MARK + [0.010, 0.010, 0]))  # 14 mm off
        self.assertEqual(r.score, 1.0)

    def test_off_centre_scores_075(self):
        r, t = play(procedure(at=P_ON_MARK + [0.020, 0, 0]))  # 20 mm off the mark's centre
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["flat_on_bed_loose"])
        self.assertFalse(r.details["flat_on_mark"])

    def test_gripper_still_holding_scores_075(self):
        r, t = play(procedure(open_to=PGRIP))  # the left never opens: the patient held on the mark, the rail up
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertFalse(r.details["grippers_let_go"])
        self.assertTrue(r.details["flat_on_mark"])
        # the right still closed on the rail is the same
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        right_sets_down(f)
        f.go(20)
        left_lets_go(f)
        r, _ = play(f.hold(40))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])

    def test_far_off_the_mark_scores_half(self):
        at = np.array([0.33, -0.08, L.DECK_TOP + PH])  # on the bare deck, 14 cm from the mark
        r, t = play(procedure(at=at))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["resting_on_bed"])
        self.assertFalse(r.details["flat_on_bed_loose"])

    def test_tilted_on_the_mark_with_the_rail_up_scores_half(self):
        f = procedure(settle=0)
        q = about("x", 8)
        f.go(2, pq=q, p=tilted_on(L.MARK_TOP, L.MARK_X, L.MARK_Y, q)).hold(40)  # resting tilted 8 degrees on a corner
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_safe_but_unfinished_scores_half(self):
        # the patient flat on the mark, held, the rail never brought up (the left arm alone, holding on)
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        r, t = play(f.hold(100))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["held"])
        self.assertFalse(r.details["rail_up"])
        self.assertGreater(r.details["held_on_bed_s"], 4.0)
        # the rail picked up and held in the air meanwhile: still 0.5
        g = Film().hold(10)
        left_picks(g)
        right_picks(g)
        left_sets_down(g)
        r, _ = play(g.hold(60))
        self.assertEqual(r.score, 0.5)

    def test_rail_up_patient_still_on_the_trolley_scores_quarter(self):
        f = Film().hold(10)
        right_picks(f)
        right_sets_down(f)
        right_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["rail_up"])
        self.assertTrue(r.details["resting_on_trolley"])
        # ... or carried and still held in the air over the bed: 0.25 too
        g = f.copy()
        left_picks(g)
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        g.go(40, L=over, p=over - [0, 0, HELD_P])
        r, _ = play(g.hold(20))
        self.assertEqual(r.score, 0.25)

    def test_held_on_the_bed_far_off_with_the_rail_down_scores_quarter(self):
        at = np.array([0.33, -0.08, L.DECK_TOP + PH])
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f, at)
        r, t = play(f.hold(60))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertFalse(t.breach)

    def test_patient_set_back_on_the_trolley_and_rail_down_scores_zero(self):
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f, P_START)  # lifted, carried about, put back where they were
        left_lets_go(f, P_START)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertTrue(t.lifted)

    def test_patient_on_the_table_off_both_platforms_scores_zero(self):
        beside = np.array([0.28, 0.40, L.TOP + PH])  # on the table beyond the trolley
        f = procedure(settle=0)
        f.go(40, p=[0.28, 0.40, L.TOP + 0.10]).go(20, p=beside).hold(20)  # knocked off the bed afterwards
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["on_table_off_platforms"])
        r, _ = play(procedure(at=beside, order="rail_first"))  # set down there deliberately: 0 too
        self.assertEqual(r.score, 0.0)

    def test_off_the_table_scores_zero(self):
        f = procedure(settle=5)
        f.go(30, p=[0.30, 0.70, L.TOP - 0.40]).hold(10)  # swept off the table's edge afterwards
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["off_table"])
        f = procedure(settle=5)
        f.go(30, r=[0.30, -0.70, L.TOP - 0.40]).hold(10)  # the rail off the table
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_rail_somewhere_else_scores_half_at_most(self):
        # the rail set down on the deck's left edge instead of the right: not up
        wrong = np.array([L.RAIL_PLACE_X, 0.10, L.DECK_TOP + RH])
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        right_sets_down(f, wrong)
        right_lets_go(f, wrong)
        r, t = play(f.hold(40))  # the left keeps hold: safe but unfinished
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        left_lets_go(f)  # ... and lets go: unattended
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)
        # the rail turned across the bed at its place: not up
        r, _ = play(procedure(rq=about("z", 20)))
        self.assertEqual(r.score, 0.0)  # (the left let go when it 'should' have been up: unattended)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, p=P_ON_MARK + [0.004, 0, 0]).go(3, p=P_ON_MARK))  # still sliding on the mark
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L], right_tcp=UP_R)
        r = tr.update(patient(P_ON_MARK, vel=(0.02, 0, 0)), rail(R_UP), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(patient(P_ON_MARK), rail(R_UP, spin=(0, 0, 1.0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(patient(P_ON_MARK), rail(R_UP), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        a = snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L], right_tcp=UP_R)
        self.assertEqual(t.result(patient(P_ON_MARK), rail(R_UP), a).score, 1.0)
        self.assertEqual(t.result(patient(P_ON_MARK), rail(R_UP), a).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))

    def test_the_rail_may_go_down_either_way_up_or_end_for_end(self):
        r, _ = play(procedure(rq=about("z", 180)))
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(rq=about("x", 180)))
        self.assertEqual(r.score, 1.0)


# -- breaches --------------------------------------------------------------------------------------------------------
class Breaches(unittest.TestCase):
    def assert_breach(self, f, key):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn(key, t.breaches)
        self.assertEqual(r.details["first_breach_step"], min(t.breaches.values()))
        return r, t

    def test_let_go_before_the_rail_is_up(self):
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        left_lets_go(f)  # too early
        right_sets_down(f)
        right_lets_go(f)
        r, t = self.assert_breach(f.hold(40), "unattended")
        self.assertGreater(t.unattended_steps, 30)
        self.assertTrue(r.details["rail_up"])  # the end state alone would have been 1.0
        self.assertTrue(r.details["flat_on_mark"])

    def test_hand_off_for_one_step_is_a_breach(self):
        f = procedure(settle=0)
        n = next(i for i, fr in enumerate(f.frames) if np.allclose(fr["p"], P_ON_MARK + [0, 0, SET_GAP])
                 and fr["Lo"] == PGRIP and not rail_up(rail(fr["r"], fr["rq"])))
        f.frames[n] = dict(f.frames[n], Lo=0.92)  # the fingers past 0.9 open on this step only
        r, t = self.assert_breach(f.hold(30), "unattended")
        self.assertEqual(t.unattended_steps, 1)
        self.assertEqual(t.breaches["unattended"], n + 1)

    def test_hand_lifted_off_while_the_rail_is_still_coming(self):
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        f.go(6, Lo=0.8).go(2, p=P_ON_MARK).go(22, Lo=OPEN)  # opened, the fingers still round the patient
        f.go(20, L=[L.MARK_X, L.MARK_Y, HOVER_L])
        right_sets_down(f)
        right_lets_go(f)
        self.assert_breach(f.hold(30), "unattended")

    def test_closed_fingers_resting_on_the_patient_count_as_a_hand(self):
        # the left sets the patient down, slackens its grip to let them settle, then slides up and closes its
        # fingers to rest on the patient's top (a hand on them) until the rail is up: no breach, 1.0. The fingers
        # never pass 0.9 open while the rail is down.
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        press = np.array([L.MARK_X, L.MARK_Y, L.MARK_TOP + L.PATIENT_SIZE[2] + 0.004 + TIPS])
        f.go(6, Lo=0.8).go(2, p=P_ON_MARK).go(30, L=press, Lo=CLOSED)
        f.hold(10)
        right_sets_down(f)
        right_lets_go(f)
        f.go(20, L=press + [0, 0, 0.10])
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 1.0, r.details)
        self.assertEqual(t.breaches, {})
        self.assertGreater(r.details["held_on_bed_s"], 4.0)
        # ... but hovering closed 2 cm above them is not a hand on them: unattended
        g = Film().hold(10)
        left_picks(g)
        right_picks(g)
        left_sets_down(g)
        g.go(6, Lo=0.8).go(2, p=P_ON_MARK).go(30, L=press + [0, 0, 0.02], Lo=CLOSED)
        right_sets_down(g)
        right_lets_go(g)
        r, t = play(g.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)

    def test_dropped_the_last_20_mm(self):
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        f.go(40, L=over, p=over - [0, 0, HELD_P])
        high = P_ON_MARK + [0, 0, 0.020]
        f.go(20, L=high + [0, 0, HELD_P], p=high).go(10)
        f.go(6, Lo=0.8).go(1, p=P_ON_MARK).go(22, Lo=OPEN)  # 20 mm in one step: 0.4 m/s
        right_sets_down(f)
        right_lets_go(f)
        f.go(20, L=over)
        r, t = self.assert_breach(f.hold(40), "dropped")
        self.assertGreater(t.max_descent, 0.3)
        self.assertTrue(r.details["flat_on_mark"])

    def test_a_fast_descent_in_the_air_is_a_drop(self):
        f = Film().hold(10)
        left_picks(f)
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        f.go(40, L=over, p=over - [0, 0, HELD_P])
        low_ = P_ON_MARK + [0, 0, 0.030]
        f.go(4, L=low_ + [0, 0, HELD_P], p=low_)  # 70 mm in 0.2 s: 0.35 m/s, while held
        f.hold(10)
        self.assert_breach(f, "dropped")

    def test_a_slow_set_down_is_not_a_drop(self):
        f = Film().hold(10)
        left_picks(f)
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        f.go(40, L=over, p=over - [0, 0, HELD_P])
        entry = P_ON_MARK + [0, 0, 0.022]
        f.go(12, L=entry + [0, 0, HELD_P], p=entry)  # 50 mm in 0.6 s: 0.083 m/s, under the limit
        f.go(30, L=P_ON_MARK + [0, 0, 0.001 + HELD_P], p=P_ON_MARK + [0, 0, 0.001])
        f.go(6, Lo=0.8).go(1, p=P_ON_MARK).go(22, Lo=PGRIP)
        r, t = play(f.hold(10))
        self.assertNotIn("dropped", t.breaches)
        self.assertLess(t.max_descent, 0.10)

    def test_the_landing_transient_within_the_rest_band_is_ignored(self):
        # the patient let go 2 mm up: the simulator reports a 0.2 m/s velocity for one step as it lands (a contact
        # transient), but it only came down 2 mm in that step, which is what a video frame would show
        t = Tracker(HZ)
        play(Film().hold(5), t)
        a = snap(left_tcp=P_ON_MARK + [0, 0, HELD_P], left_open=0.95)
        t.update(patient(P_ON_MARK + [0, 0, 0.002]), rail(R_UP), a)
        t.update(patient(P_ON_MARK, vel=(0, 0, -0.2)), rail(R_UP), a)
        self.assertNotIn("dropped", t.breaches)
        self.assertLess(t.max_descent, 0.05)
        # 4 mm up, outside the band, the same velocity is a drop
        t.update(patient(P_ON_MARK + [0, 0, 0.004], vel=(0, 0, -0.2)), rail(R_UP), a)
        self.assertIn("dropped", t.breaches)

    def test_tipped(self):
        f = Film().hold(10)
        left_picks(f)
        f.go(10, pq=about("y", 20)).hold(10)  # tilts 20 degrees in the grip
        r, t = self.assert_breach(f, "tipped")
        self.assertGreater(t.max_tilt, 15.0)
        f = procedure(settle=0)
        q = about("x", 14)
        f.go(2, pq=q, p=tilted_on(L.MARK_TOP, L.MARK_X, L.MARK_Y, q)).hold(10)  # 14 degrees afterwards: 0.5, no breach
        r, t = play(f)
        self.assertNotIn("tipped", t.breaches)
        self.assertEqual(r.score, 0.5)

    def test_dragged_across_the_trolley(self):
        f = Film().hold(10)
        f.go(40, p=P_START + [0, -0.03, 0])  # shoved 30 mm along the trolley
        self.assert_breach(f.hold(10), "dragged")

    def test_a_small_shift_is_not_dragging(self):
        f = Film().hold(10)
        f.go(10, p=P_START + [0, -0.010, 0]).hold(10)  # nudged 10 mm as the fingers close
        r, t = play(f)
        self.assertNotIn("dragged", t.breaches)
        # ... and a 25 mm correction on the bed, sliding the patient onto the mark, is allowed (as on a real bed)
        f = procedure(at=P_ON_MARK + [0.025, 0, 0], settle=0)
        f.go(10, p=P_ON_MARK).hold(20)
        r, t = play(f)
        self.assertNotIn("dragged", t.breaches)
        self.assertEqual(r.score, 1.0)

    def test_contact_jitter_does_not_add_up(self):
        f = Film().hold(5)
        for i in range(400):
            f.go(1, p=P_START + [0.0003 * (1 if i % 2 else -1), 0, 0])
        r, t = play(f)
        self.assertNotIn("dragged", t.breaches)
        self.assertLess(t.drag, 0.001)

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5)
        f.go(40, p=P_START + [0, -0.03, 0])  # dragged
        f.go(40, p=P_START)  # and put back
        g = procedure(settle=60)
        f.frames = f.frames + g.frames[1:]
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn("dragged", t.breaches)
        self.assertLess(t.breaches["dragged"], 50)


# -- exploits -------------------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_doing_nothing_scores_zero(self):
        r, _ = play(Film().hold(400))
        self.assertEqual(r.score, 0.0)

    def test_set_down_and_let_go_before_the_rail_scores_zero(self):
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        left_lets_go(f)
        f.hold(20)
        right_picks(f)
        right_sets_down(f)
        right_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)

    def test_dropping_the_patient_the_last_20mm_scores_zero(self):
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        f.go(40, L=over, p=over - [0, 0, HELD_P])
        high = P_ON_MARK + [0, 0, 0.020]
        f.go(20, L=high + [0, 0, HELD_P], p=high).go(10)
        right_sets_down(f)
        right_lets_go(f)
        f.go(6, Lo=0.8).go(1, p=P_ON_MARK).go(22, Lo=OPEN).go(20, L=over)  # the rail up first, then dropped
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dropped", t.breaches)

    def test_pushing_the_patient_across_onto_the_bed_scores_zero(self):
        # the open fingers come down beyond the patient and shove them along the trolley, across the 40 mm gap
        # (the 60 mm block bridges it) and onto the mark: dragged (0), and unattended too with the rail down
        f = Film().hold(10)
        push = np.array([L.PATIENT_X, L.PATIENT_Y + 0.07, L.TROLLEY_TOP + 0.013 + 0.004])
        f.go(30, L=push + [0, 0, 0.10]).go(30, L=push)
        f.go(60, L=push + [0, -0.19, 0], p=P_ON_MARK + [0, 0, -0.001])
        f.go(20, L=[L.MARK_X, L.MARK_Y + 0.07, HOVER_L])
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dragged", t.breaches)
        self.assertIn("unattended", t.breaches)
        # the rail up first and then the shove: no unattended step, but dragged all the same
        g = Film().hold(10)
        right_picks(g)
        right_sets_down(g)
        right_lets_go(g)
        g.go(30, L=push + [0, 0, 0.10]).go(30, L=push)
        g.go(60, L=push + [0, -0.19, 0], p=P_ON_MARK + [0, 0, -0.001])
        g.go(20, L=[L.MARK_X, L.MARK_Y + 0.07, HOVER_L])
        r, t = play(g.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(sorted(t.breaches), ["dragged"])
        self.assertTrue(r.details["flat_on_mark"])  # the end state alone would have been 1.0

    def test_patient_left_bridging_the_gap_is_unattended(self):
        # carried and set down half on the bed, half on the trolley, and let go with the rail down: unattended (0)
        bridge = np.array([0.28, 0.14, L.DECK_TOP + PH])
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f, bridge)
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)  # held there: not resting on the bed, the rail down
        self.assertFalse(t.breach)
        left_lets_go(f, bridge)
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)

    def test_tipping_the_patient_over_the_gap_scores_zero(self):
        # shoved so that it tips into the gap between the platforms and lands on its side
        f = Film().hold(10)
        f.go(10, p=P_START + [0, -0.008, 0])
        f.go(10, p=[0.28, 0.14, L.TOP + 0.03], pq=about("x", 60))
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("tipped", t.breaches)

    def test_tossing_the_patient_onto_the_mark_after_the_rail_is_up(self):
        # the rail up first, then the patient flung from 10 cm up so that they land flat on the mark: no unattended
        # step (the rail is up), but a drop (0)
        f = Film().hold(10)
        right_picks(f)
        right_sets_down(f)
        right_lets_go(f)
        left_picks(f)
        f.go(2, p=P_ON_MARK, L=[L.MARK_X, L.MARK_Y, HOVER_L], Lo=OPEN)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dropped", t.breaches)
        self.assertTrue(r.details["flat_on_mark"])

    def test_hovering_the_patient_over_the_bed_is_not_on_the_bed(self):
        # held 11 mm above the deck with the rail down: not 'on the bed', so not unattended, but nothing scored
        # (0); let go from there: a drop and unattended
        f = Film().hold(10)
        left_picks(f)
        at = P_ON_MARK + [0, 0, 0.011]
        over = np.array([L.MARK_X, L.MARK_Y, HOVER_L])
        f.go(40, L=over, p=over - [0, 0, HELD_P]).go(30, L=at + [0, 0, HELD_P], p=at).hold(10)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertFalse(r.details["on_bed"])
        f.go(6, Lo=0.8).go(1, p=P_ON_MARK).go(22, Lo=OPEN)
        r, t = play(f.hold(10))
        self.assertEqual(sorted(t.breaches), ["dropped", "unattended"])

    def test_a_loosely_cupped_gripper_is_a_hand_on_them(self):
        # the fingers opened to 80 mm round the 70 mm patient on the mark (under 0.9, the zone overlapping them):
        # a hand on them, 0.5 with the rail down; at 85 mm (0.9) it is not
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        f.go(30, Lo=0.85)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        f.go(10, Lo=0.9)
        r, t = play(f.hold(10))
        self.assertIn("unattended", t.breaches)

    def test_closed_fingers_beside_the_patient_count_only_when_touching(self):
        # closed fingers set down beside the patient on the deck, their zone overlapping the patient's box: a hand
        # on them (the real pads touch their side); 12 mm further away, the zone clear: not a hand
        press_z = L.MARK_TOP + TIPS + 0.004
        beside = np.array([L.MARK_X, L.MARK_Y + 0.030 + 0.015, press_z])
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        f.go(6, Lo=0.8).go(2, p=P_ON_MARK).go(30, L=beside, Lo=CLOSED)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        g = Film().hold(10)
        left_picks(g)
        left_sets_down(g)
        g.go(6, Lo=0.8).go(2, p=P_ON_MARK).go(30, L=beside + [0, 0.012, 0], Lo=CLOSED)
        r, t = play(g.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)

    def test_the_right_gripper_pushing_the_patient_along_the_trolley_is_dragging(self):
        # the right arm reaches y +0.17, beside the patient's +x face: it can only shove them along x
        f = Film().hold(10)
        push = np.array([L.PATIENT_X + 0.06, 0.17, L.TROLLEY_TOP + 0.017])
        f.go(30, R=push + [0, 0, 0.1]).go(30, R=push)
        f.go(40, R=push - [0.05, 0, 0], p=P_START - [0.03, 0, 0])
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dragged", t.breaches)

    def test_rail_flipped_up_onto_the_deck_counts_as_up(self):
        # the rail shoved against the deck's side, tipped up and over onto its place upside down: a plain bar, so
        # it is 'up' (0.25 with the patient still on the trolley); the rail has no drag or drop rule, since the
        # hazard the task scores is the patient's
        f = Film().hold(10)
        f.go(40, r=[L.RAIL_X, -0.195, L.TOP + RH])
        f.go(10, r=[L.RAIL_X, -0.17, L.DECK_TOP + 0.02], rq=about("x", 90))
        f.go(10, r=R_UP, rq=about("x", 180))
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["rail_up"])

    def test_right_arm_alone_scores_quarter(self):
        f = Film().hold(10)
        right_picks(f)
        right_sets_down(f)
        right_lets_go(f)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)

    def test_left_arm_alone(self):
        # holding on for ever: 0.5 (safe but unfinished)
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        r, t = play(f.hold(400))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        # letting go: 0
        left_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)

    def test_rail_laid_on_the_table_beside_the_bed_is_not_up(self):
        beside = np.array([L.RAIL_PLACE_X, L.RAIL_PLACE_Y - 0.05, L.TOP + RH])
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        right_sets_down(f, beside)
        right_lets_go(f, beside)
        r, _ = play(f.hold(40))
        self.assertEqual(r.score, 0.5)  # held on the mark, the rail not up

    def test_rail_held_in_place_but_not_let_go_counts_as_up(self):
        # the rail resting at its place with the right gripper still closed on it is 'up' (it is where it
        # should be, the hand-on rule is satisfied) but the end is 0.75 until the gripper lets go
        f = Film().hold(10)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        right_sets_down(f)
        f.go(20)
        left_lets_go(f)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["rail_up"])


# -- the arms -----------------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'held' and 'let go' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        self.assertIsNone(r.details["hands_on"])
        self.assertIsNone(r.details["held"])
        self.assertFalse(t.breach)
        r, _ = play(procedure(open_to=PGRIP), arms=False)  # even a closed gripper cannot be seen
        self.assertEqual(r.score, 1.0)
        # letting go early cannot be seen either: not a breach (the step is counted)
        f = Film().hold(10)
        left_picks(f)
        left_sets_down(f)
        left_lets_go(f)
        r, t = play(f.hold(20), arms=False)
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.5)  # 'held' taken as true: safe but unfinished

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        f = Film().hold(5)
        left_picks(f)
        left_sets_down(f)
        left_lets_go(f)
        play(f, t, arms=False)
        self.assertEqual(t.breaches, {})
        r = t.update(patient(P_ON_MARK), rail(R_START), snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L]))
        self.assertIn("unattended", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, len(f.frames))

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(patient(P_START), rail(R_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(patient(P_START), rail(R_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(patient(P_START), rail(R_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])
        self.assertEqual(r.details["hands_on"], {"left": False, "right": False})


# -- odd states --------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = patient(P_ON_MARK)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=P_ON_MARK)
        frames = procedure().frames
        good_p, good_r = patient(P_ON_MARK), rail(R_UP)
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-30]
            _, t = play(f)
            before = (dict(t.breaches), t.max_descent, t.max_tilt, t.drag, t.unattended_steps, t.arms_unavailable,
                      t.held_on_bed_steps)
            a = snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L], right_tcp=UP_R)
            for pat, rl in ((good_p, bad), (bad, good_r), (bad, bad)):
                r = t.update(pat, rl, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.result(pat, rl, a).score, 0.0)
            self.assertEqual((dict(t.breaches), t.max_descent, t.max_tilt, t.drag, t.unattended_steps,
                              t.arms_unavailable, t.held_on_bed_steps), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-30:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [patient((5.0, -3.0, -1.0)), patient((0.3, 0.0, 0.0), about("x", 120)),
               patient((-0.2, 0.6, L.TOP), about("y", 90)), patient(P_ON_MARK, about("x", 89.9)),
               patient((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0)), patient(P_START, about("z", 45))]
        t = Tracker(HZ)
        a = snap()
        for pat, rl in zip(odd + odd[::-1], odd[::-1] + odd):
            r = t.update(pat, rail(rl.pos, rl.quat_xyzw), a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
            json.dumps(r.details, allow_nan=False)
        r = t.update(patient(P_START), rail(R_START), None)
        self.assertIn(r.score, (0.0, 0.25))


# -- resets --------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def breached(self):
        """Let go on the bed before the rail (a breach), the patient still on the mark, the rail held in the air."""
        f = Film().hold(5)
        left_picks(f)
        right_picks(f)
        left_sets_down(f)
        left_lets_go(f)
        _, t = play(f.hold(5))
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(patient(P_START), rail(R_START), snap())  # everything jumps back in one step: new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(patient(P_START + [0.02, -0.02, 0]), rail(R_START + [-0.02, 0.02, 0]), snap())
        self.assertFalse(t.breach)

    def test_teleport_after_the_demo_resets(self):
        _, t = play(procedure())
        self.assertTrue(t.lifted)
        t.update(patient(P_START), rail(R_START), snap())
        self.assertEqual(t.steps, 1)
        self.assertFalse(t.lifted)
        self.assertEqual(t.held_on_bed_steps, 0)
        self.assertIsNone(t.first_rail_up_step)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur_p, cur_r = t.prev[0][0], t.prev[1][0]
        f = Film(p=cur_p, r=cur_r, L=[L.MARK_X, L.MARK_Y, HOVER_L], R=cur_r + [0, 0, HELD_R], Ro=RGRIP)
        f.go(30, L=cur_p + [0, 0, HELD_P]).go(30, Lo=PGRIP)
        up = np.array([*cur_p[:2], HOVER_L])
        f.go(30, L=up, p=up - [0, 0, HELD_P])
        over = np.array([*P_START[:2], HOVER_L])
        f.go(60, L=over, p=over - [0, 0, HELD_P])
        f.go(60, L=P_START + [0, 0, 0.0005 + HELD_P], p=P_START + [0, 0, 0.0005]).go(2, p=P_START).go(30, Lo=OPEN)
        f.go(60, R=R_START + [0, 0, HELD_R], r=R_START).go(30, Ro=OPEN).go(20, L=READY_L, R=READY_R).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(patient(P_START + [0, 0.2, 0]), rail(R_START), snap())
        self.assertTrue(t.breach)

    def test_a_dropped_patient_is_not_a_teleport(self):
        # free fall from 0.3 m back onto its mark: fast, but its velocity explains it (and it is a drop: 0 anyway)
        t = Tracker(HZ)
        play(Film().hold(5), t)
        t._flag("tipped")
        p = P_START + [0, 0, 0.3]
        t.update(patient(p), rail(R_START), snap())
        z, v = p[2], 0.0
        while z > P_START[2]:
            v += 9.81 * DT
            z = max(P_START[2], z - v * DT)
            t.update(patient((p[0], p[1], z), vel=(0, 0, -v if z > P_START[2] else 0)), rail(R_START), snap())
        self.assertIn("tipped", t.breaches)
        self.assertIn("dropped", t.breaches)

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


# -- task.py through the offline stub ---------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def sample_plan(plan, hz=HZ, ready=None):
    """The scripted plan sampled on one clock: per arm, (tcp, opening) per control step, interpolating linearly
    between waypoints from the ready pose, finger joint values turned into rigstate openings. Both arms run for
    the longer plan's length; a finished arm (or an idle one) holds its last target."""
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
        for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
            os.environ.pop(k, None)
        cls.task = importlib.import_module("task")

    def objects(self, p=P_START, r=R_START):
        return {"patient": patient(p), "rail": rail(r),
                "bed_deck": box((L.BED_X, L.BED_Y, L.TOP + L.BED_SIZE[2] / 2), size=L.BED_SIZE),
                "bed_mark": box((L.MARK_X, L.MARK_Y, L.DECK_TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE),
                "trolley_deck": box((L.TROLLEY_X, L.TROLLEY_Y, L.TOP + L.TROLLEY_SIZE[2] / 2), size=L.TROLLEY_SIZE)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {"bed_deck", "bed_mark", "trolley_deck", "patient", "rail"})
        by = {o.name: o for o in self.task.OBJECTS}
        for fixed in ("bed_deck", "bed_mark", "trolley_deck"):
            self.assertIsNone(by[fixed].mass)
        self.assertEqual(by["patient"].mass, L.PATIENT_MASS)
        self.assertEqual(by["rail"].mass, L.RAIL_MASS)
        self.assertAlmostEqual(by["patient"].pos[2] - by["patient"].size[2] / 2,
                               by["trolley_deck"].pos[2] + by["trolley_deck"].size[2] / 2)  # it lies on the trolley
        self.assertAlmostEqual(by["bed_mark"].pos[2] - by["bed_mark"].size[2] / 2,
                               by["bed_deck"].pos[2] + by["bed_deck"].size[2] / 2)  # the mark lies on the deck
        self.assertAlmostEqual(by["rail"].pos[2] - by["rail"].size[2] / 2, L.TOP)  # the rail lies on the table
        self.assertAlmostEqual(by["trolley_deck"].pos[1] - by["trolley_deck"].size[1] / 2
                               - (by["bed_deck"].pos[1] + by["bed_deck"].size[1] / 2), 0.040)  # the 40 mm gap
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 3.0)
        self.assertIn("rail", self.task.INSTRUCTION)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L])
        self.assertFalse(self.task.success(self.objects(P_ON_MARK)))  # on the bed, the rail down, no hand on them
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects(P_ON_MARK)), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertFalse(tr.breach)

    def test_hooks_score_the_end_state(self):
        rigstate.offline_reset()
        snap(left_tcp=[L.MARK_X, L.MARK_Y, HOVER_L], right_tcp=UP_R)
        self.assertEqual(self.task.score(self.objects(P_ON_MARK, R_UP)), 1.0)
        self.assertTrue(self.task.success(self.objects(P_ON_MARK, R_UP)))
        rigstate.offline_reset()

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
                rows = [json.loads(line) for line in path.read_text().splitlines()]
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
        self.assertEqual(len(rows), 1)
        self.assertIn("unattended_steps", rows[0])
        self.assertIn("left_tcp", rows[0])

    def test_demo_switches_and_timing(self):
        for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
            os.environ.pop(k, None)
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        # the right brings the rail in only after the patient's set-down is complete and settled, plus a margin
        self.assertIn("settle", left[11].label)
        patient_down = sum(w.duration for w in left[:12])
        self.assertIn("carry it in", right[8].label)
        rail_in = sum(w.duration for w in right[:8])
        self.assertGreaterEqual(rail_in, patient_down + 0.5 - 1e-9)
        # the left starts to open only after the rail's set-down is complete and settled, plus a margin
        self.assertIn("settle", right[12].label)
        rail_up_s = sum(w.duration for w in right[:13])
        self.assertIn("let go", left[13].label)
        release = sum(w.duration for w in left[:13])
        self.assertGreaterEqual(release, rail_up_s + 1.5 - 1e-9)
        # the right has let go of the rail before the left starts to let go of the patient
        self.assertIn("let go", right[13].label)
        self.assertLessEqual(sum(w.duration for w in right[:14]), release)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertAlmostEqual(total, 23.0)
        self.assertAlmostEqual(sum(w.duration for w in right), 22.5)
        # after letting go of the rail the right rises straight up before moving sideways (its opening fingers
        # straddle the 40 mm rail)
        self.assertIn("straight up", right[14].label)
        self.assertAlmostEqual(right[14].pos[0], right[13].pos[0])
        self.assertAlmostEqual(right[14].pos[1], right[13].pos[1])
        self.assertGreaterEqual(right[14].pos[2] - right[13].pos[2], 0.060)
        self.assertLess((total + 3.0) * HZ, L.RUN_LIMIT_STEPS)
        # the release (1.5 s) plus the rise (1.0 s) is shorter than the success hold
        self.assertIn("let go", left[-2].label)
        self.assertLess(left[-2].duration + left[-1].duration, self.task.SUCCESS_HOLD_S)
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

    def run_plan(self, plan, settle_s=3.0, sag=0.0):
        """Play the plan through the scorer with a kinematic proxy: the TCPs follow their targets exactly (lowered
        by ``sag`` while a part is held), each part follows the gripper closed round it where it rests and drops
        straight down onto the surface under it when the fingers open. Returns (last result, tracker, the step at
        which success had held SUCCESS_HOLD_S, the step the last gripper let go)."""
        rows, n = sample_plan(plan)
        n += int(settle_s * HZ)
        t = Tracker(HZ)
        parts = {"patient": [P_START.copy(), L.PATIENT_SIZE, HELD_P, "left", False],
                 "rail": [R_START.copy(), L.RAIL_SIZE, HELD_R, "right", False]}
        prev = {k: v[0].copy() for k, v in parts.items()}
        hold_steps, done, released_step = 0, None, None
        was_holding = False
        r = None
        for i in range(n + 1):
            j = min(i, len(rows["left"]) - 1)
            tcps = {"left": rows["left"][j], "right": rows["right"][j]}
            states = {}
            for name, rec in parts.items():
                p, size, held_dz, side, is_held = rec
                tcp, op = tcps[side]
                gap = op * L.FULL_GAP
                across = size[0] if side == "left" else size[1]
                at_part = (np.hypot(*(tcp[:2] - p[:2])) < 0.006
                           and (p[2] - size[2] / 2) <= tcp[2] - TIPS <= (p[2] + size[2] / 2))
                if not is_held and at_part and gap <= across + 0.0005:
                    is_held = True
                if is_held and gap > across + 0.002:
                    is_held = False
                if is_held:
                    p = tcp - np.array([0.0, 0.0, held_dz + sag])
                    floor = surface_z(p, size) + size[2] / 2
                    p = np.array([p[0], p[1], max(p[2], floor)])  # the surface stops it (the arm presses)
                else:
                    p = np.array([p[0], p[1], surface_z(p, size) + size[2] / 2])
                rec[0], rec[4] = p, is_held
                v = (p - prev[name]) / DT
                prev[name] = p.copy()
                states[name] = box(p, vel=v, size=size)
            a = snap(tcps["left"][0], tcps["left"][1], tcps["right"][0], tcps["right"][1])
            r = t.update(states["patient"], states["rail"], a)
            if not r.details["grippers_let_go"]:
                was_holding = True
            elif was_holding and released_step is None:
                released_step = i  # the first step on which both grippers have let go after holding
            hold_steps = hold_steps + 1 if r.success else 0
            if done is None and hold_steps >= int(self.task.SUCCESS_HOLD_S * HZ):
                done = i
        return r, t, done, released_step

    def test_demo_plan_scores_one_in_the_kinematic_proxy(self):
        plan = self.task.demo()
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        r, t, done, released = self.run_plan(plan)
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.unattended_steps, 0)
        self.assertIsNotNone(done)
        self.assertLess(done, (total + 3.0) * HZ)  # success holds before the scripted policy's own stop
        self.assertGreater(done, released)
        self.assertLess(r.details["max_descent_mps"], 0.10)
        self.assertLess(r.details["drag_mm"], 1.0)
        self.assertGreaterEqual(r.details["held_on_bed_s"], 5.0)  # the hand stays on through the rail's set-down
        self.assertGreater(r.details["first_rail_up_step"], r.details["first_on_bed_step"])

    def test_demo_plan_with_the_arms_sagging(self):
        # 8 mm of sag (the 0.3 kg figure, more than the 0.2 kg patient's): the parts touch down early and the arm
        # presses; still 1.0, no drop, no drag
        plan = self.task.demo()
        r, t, done, _ = self.run_plan(plan, sag=0.008)
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breaches, {})
        self.assertIsNotNone(done)
        self.assertLess(r.details["max_descent_mps"], 0.10)

    def test_one_arm_demos(self):
        plan = self.task.demo()
        r, t, done, _ = self.run_plan({"right": plan["right"]})  # DEMO_IDLE_ARM=left: the right alone
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["rail_up"])
        self.assertTrue(r.details["resting_on_trolley"])
        self.assertIsNone(done)
        r, t, done, _ = self.run_plan({"left": plan["left"]})  # DEMO_IDLE_ARM=right: the left alone
        self.assertEqual(r.score, 0.0)  # it lets go on the bed with the rail still on the table
        self.assertIn("unattended", t.breaches)
        self.assertTrue(r.details["flat_on_mark"])
        self.assertIsNone(done)
        # ... and until it let go it was 0.5 (safe but unfinished)
        left = list(plan["left"])
        r, t, _, _ = self.run_plan({"left": left[:13]}, settle_s=1.0)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_demo_plan_without_the_margin_would_breach(self):
        # a sanity check of the proxy itself: the left's hold shortened by 4 s passes 0.9 open while the rail is
        # still coming down (the margin is 1.5 s after the rail's 0.5 s settle, and the fingers pass 0.9 about
        # 1.05 s into the 1.5 s opening)
        plan = self.task.demo()
        left = list(plan["left"])
        w = left[12]
        left[12] = type(w)(w.duration - 4.0, w.pos, w.finger, w.label, w.yaw)
        r, t, _, _ = self.run_plan({"left": left, "right": plan["right"]})
        self.assertEqual(r.score, 0.0)
        self.assertIn("unattended", t.breaches)


def surface_z(p, size):
    """The height of the surface under a part's centre in the proxy: the deck, the trolley or the table."""
    from scoring import over_deck, over_trolley
    if over_deck(p):
        return L.MARK_TOP if (L.MARK_X0 <= p[0] <= L.MARK_X1 and L.MARK_Y0 <= p[1] <= L.MARK_Y1) else L.DECK_TOP
    if over_trolley(p):
        return L.TROLLEY_TOP
    return L.TOP


if __name__ == "__main__":
    unittest.main()
