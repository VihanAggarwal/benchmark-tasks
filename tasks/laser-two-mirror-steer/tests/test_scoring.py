"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker, ray_box, rotation, trace_beam, valid  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
UP = L.TOP + 0.030  # the centre height of a 60 mm part standing on the table
GRIP_UP = 0.020  # a held part's centre is this far below the TCP (TCP 50 mm up, centre 30 mm up)
OPEN = 1.0
SHUT_M = L.MIRROR_SIZE[1] / L.FULL_GAP  # rigstate opening with the fingers stopped on a mirror's 60 mm width
SHUT_B = L.BLOCK_SIZE[1] / L.FULL_GAP  # ... on the block's 40 mm
FACE = L.MIRROR_SIZE[0] / 2 / math.sqrt(2)

ML0 = np.array([L.MIRROR_L_X, L.MIRROR_L_Y, UP, 0.0])  # x, y, z (centre), yaw (deg)
MR0 = np.array([L.MIRROR_R_X, L.MIRROR_R_Y, UP, 0.0])
B0 = np.array([L.BLOCK_X, L.BLOCK_Y, UP, 0.0])
M1T = np.array([0.32 + FACE, L.LASER_Y + FACE, UP, 45.0])  # the demo's places: the beam meets both faces at x 0.32
M2T = np.array([0.32 + FACE, L.DET_Y - FACE, UP, -45.0])
BS = np.array([0.22, L.LASER_Y, UP, 0.0])  # the block in the beam
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
POSE_KEYS = ("ml", "mr", "b")
NAMES = {"ml": "mirror_left", "mr": "mirror_right", "b": "block"}


def quat(yaw_deg=0.0, tilt_deg=0.0):
    """Quaternion (xyzw): a turn of yaw_deg about z, after a tilt of tilt_deg about the body's y axis (for a mirror,
    its 60 mm width: a positive tilt makes the face toward the laser look down, a negative one up)."""
    hz, hy = math.radians(yaw_deg) / 2, -math.radians(tilt_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(hz), math.cos(hz)])
    qy = np.array([0.0, math.sin(hy), 0.0, math.cos(hy)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def state(name, p, vel=(0, 0, 0), spin=(0, 0, 0), tilt=0.0, q=None):
    """An ObjectState-like box for a loose object at p = (x, y, z centre, yaw deg)."""
    p = np.asarray(p, float)
    q = quat(p[3], tilt) if q is None else np.asarray(q, float)
    c, half = p[:3].copy(), np.array(L.SIZES[name], float) / 2
    ext = np.abs(rotation(q)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c - ext, high=c + ext)


def scene(ml=ML0, mr=MR0, b=B0, **kw):
    """The three loose objects' states (plus a fixed one, as the simulator passes every object)."""
    out = {"mirror_left": state("mirror_left", ml, **kw.get("ml_kw", {})),
           "mirror_right": state("mirror_right", mr, **kw.get("mr_kw", {})),
           "block": state("block", b, **kw.get("b_kw", {}))}
    out["laser"] = SimpleNamespace(pos=np.array([L.LASER_X, L.LASER_Y, L.TOP + 0.03]),
                                   quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                   low=np.zeros(3), high=np.zeros(3))
    return out


def beam_of(objs):
    return trace_beam(S.scene_solids({n: objs[n] for n in L.LOOSE}))


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


FINAL = scene(M1T, M2T, B0)


class Film:
    """A trial as a list of frames: the three loose objects (x, y, z, yaw), both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"ml": ML0, "mr": MR0, "b": B0, "L": READY_L, "Lo": OPEN, "R": READY_R, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k not in ("Lo", "Ro") else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k not in ("Lo", "Ro") else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float) for k in a})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def play(film, tracker=None, arms=True):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r, prev = None, None
    for fr in film.frames:
        objs = {}
        for k in POSE_KEYS:
            v = (fr[k][:3] - prev[k][:3]) / DT if prev is not None else np.zeros(3)
            w = (0, 0, math.radians(fr[k][3] - prev[k][3]) / DT) if prev is not None else (0, 0, 0)
            objs[NAMES[k]] = state(NAMES[k], fr[k], v, w)
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(objs, a)
        prev = fr
    return r, t


def tcp_over(p, dz=GRIP_UP):
    return np.array([p[0], p[1], p[2] + dz])


def carry(f, arm, key, to, shut, lift=0.07, turn_at=None):
    """``arm`` ('L' or 'R') takes the part ``key`` from where it is to ``to`` (x, y, z, yaw): down, grip, lift, carry,
    set down, let go, up."""
    o = arm + "o"
    p = np.asarray(f.cur[key], float)
    to = np.asarray(to, float)
    f.go(20, **{arm: tcp_over(p, GRIP_UP + 0.08)}).go(20, **{arm: tcp_over(p)}).go(20, **{o: shut})
    up = p + [0, 0, lift, 0]
    f.go(20, **{arm: tcp_over(up), key: up})
    over = np.array([to[0], to[1], up[2], to[3]])
    f.go(30, **{arm: tcp_over(over), key: over})
    f.go(30, **{arm: tcp_over(to), key: to})
    f.go(20, **{o: OPEN}).go(20, **{arm: tcp_over(to, GRIP_UP + 0.10)})
    return f


def procedure(m1=M1T, m2=M2T, settle=30):
    """The intended trial: the left arm puts the block in the beam, the right arm sets the right mirror, the left
    arm sets the left mirror behind the block, then lifts the block out and puts it back; all still at the end."""
    f = Film().hold(5)
    carry(f, "L", "b", BS, SHUT_B)
    carry(f, "R", "mr", m2, SHUT_M)
    carry(f, "L", "ml", m1, SHUT_M)
    carry(f, "L", "b", B0, SHUT_B)
    return f.hold(settle)


class Beam(unittest.TestCase):
    def test_start_beam_ends_on_the_dump(self):
        b = beam_of(scene())
        self.assertEqual(b.end, "beam_dump")
        self.assertEqual(b.reflections, [])
        self.assertAlmostEqual(b.points[-1][0], L.DUMP_X - L.DUMP_SIZE[0] / 2, places=9)

    def test_demo_path_hits_the_detector_centre(self):
        b = beam_of(FINAL)
        self.assertEqual(b.reflections, ["mirror_left", "mirror_right"])
        self.assertEqual(b.end, "detector")
        self.assertTrue(b.on_detector)
        self.assertLess(b.det_off, 1e-6)
        pts = np.array(b.points)
        np.testing.assert_allclose(pts[1], [0.32, L.LASER_Y, L.BEAM_Z], atol=1e-6)  # the middle of M1's face
        np.testing.assert_allclose(pts[2], [0.32, L.DET_Y, L.BEAM_Z], atol=1e-6)  # the middle of M2's face
        np.testing.assert_allclose(pts[3], [L.DET_FACE_X, L.DET_Y, L.BEAM_Z], atol=1e-6)

    def test_law_of_reflection(self):
        # one mirror at 45 deg in the beam: the beam turns to -y and leaves the table (nothing on that line)
        b = beam_of(scene(ml=M1T))
        self.assertEqual(b.reflections, ["mirror_left"])
        self.assertEqual(b.end, "stray")
        self.assertAlmostEqual(b.points[-1][1], L.TABLE_Y[0], places=6)
        self.assertAlmostEqual(b.points[-1][0], 0.32, places=6)
        # faces square to the beam (yaw 0): straight back into the laser, which stops it
        b = beam_of(scene(ml=[0.33, L.LASER_Y, UP, 0.0]))
        self.assertEqual(b.end, "laser")
        # normal at 60 deg: the beam comes off at -60 deg (incidence 60 deg, reflection 60 deg) and runs off the
        # right edge
        b = beam_of(scene(ml=[0.33, L.LASER_Y, UP, 60.0]))
        d = np.array(b.points[-1]) - np.array(b.points[-2])
        self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])), -60.0, places=6)
        self.assertEqual(b.end, "stray")
        self.assertAlmostEqual(b.points[-1][1], L.TABLE_Y[0], places=6)

    def test_mirror_edge_stops_the_beam(self):
        b = beam_of(scene(ml=[0.33, L.LASER_Y, UP, 90.0]))  # the 20 mm edge faces the laser
        self.assertEqual(b.end, "mirror_left")
        self.assertEqual(b.reflections, [])
        self.assertIn("mirror_left", b.lit)

    def test_only_the_window_reflects_the_rim_stops(self):
        # the left mirror slid along its face (local y, the 60 mm width) by s: the beam meets the face s from the
        # middle; inside 20 mm it reflects off the window, beyond it stops on the 10 mm rim (the mirror is lit)
        u = np.array([-math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        for s_mm, reflects in ((0.0, True), (19.5, True), (-19.5, True), (20.5, False), (-20.5, False),
                               (29.0, False)):
            b = beam_of(scene(M1T + u * s_mm / 1000, M2T, B0))
            self.assertEqual("mirror_left" in b.reflections, reflects, s_mm)
            self.assertIn("mirror_left", b.lit)
            if not reflects:
                self.assertEqual(b.end, "mirror_left")
        # the window is 40 mm tall too (10 to 50 mm up the face): a mirror held 25 mm up has the beam 15 mm up its
        # face, in the window; held 35 mm up, 5 mm up its face, on the rim
        self.assertIn("mirror_left", beam_of(scene(M1T + [0, 0, 0.025, 0], MR0, B0)).reflections)
        self.assertEqual(beam_of(scene(M1T + [0, 0, 0.035, 0], MR0, B0)).end, "mirror_left")
        self.assertTrue(S.in_window(np.array([0.32, L.LASER_Y, L.BEAM_Z]), S.loose_solid("mirror_left",
                                                                                          FINAL["mirror_left"])))

    def test_only_the_detector_front_face_counts(self):
        # the left mirror turning the beam down x 0.20: it meets the detector's side (+y) face, not its front
        objs = scene([0.20 + FACE, L.LASER_Y + FACE, UP, 45.0], MR0, B0)
        b = beam_of(objs)
        self.assertEqual(b.end, "detector")
        self.assertFalse(b.on_detector)
        self.assertIsNone(b.det_off)
        self.assertAlmostEqual(b.points[-1][1], L.DET_Y + L.DET_HALF_W, places=6)
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.25)

    def test_block_stops_the_beam(self):
        b = beam_of(scene(b=BS, ml=M1T, mr=M2T))
        self.assertEqual(b.end, "block")
        self.assertEqual(b.lit, set())

    def test_block_lying_low_does_not_stop_it(self):
        # a part lower than the beam (e.g. a mirror fallen flat, 20 mm tall) lets the beam pass over it
        flat = state("mirror_left", [0.30, L.LASER_Y, L.TOP + 0.01, 0.0], q=np.array([0, math.sin(math.pi / 4), 0,
                                                                                         math.cos(math.pi / 4)]))
        objs = scene()
        objs["mirror_left"] = flat
        self.assertEqual(beam_of(objs).end, "beam_dump")

    def test_tilted_mirror_sends_the_beam_up_or_down(self):
        objs = scene(ml=M1T, mr=M2T, ml_kw={"tilt": -3.0})  # face looks 3 deg up: the beam climbs 6 deg, over M2
        b = beam_of(objs)
        self.assertEqual(b.reflections, ["mirror_left"])
        self.assertEqual(b.end, "stray")
        self.assertGreater(b.points[-1][2], L.TOP + 0.06)
        objs = scene(ml=M1T, mr=M2T, ml_kw={"tilt": 6.0})  # looks down: into the table before the right mirror
        b = beam_of(objs)
        self.assertEqual(b.end, "table")
        self.assertEqual(b.reflections, ["mirror_left"])
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.25)
        self.assertAlmostEqual(b.points[-1][2], L.TOP, places=9)
        # a small tilt keeps it on M2 and the detector but moves the spot down (or up)
        objs = scene(ml=M1T, mr=M2T, ml_kw={"tilt": 0.5})  # 7 mm low: still within 10 mm
        b = beam_of(objs)
        self.assertTrue(b.on_detector)
        self.assertLess(b.det_dz, -0.005)
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 1.0)
        objs = scene(ml=M1T, mr=M2T, ml_kw={"tilt": 1.0})  # 15 mm low: on the face, off centre
        self.assertLess(beam_of(objs).det_dz, -0.010)
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.75)

    def test_trapped_beam_stays_on_the_table(self):
        old = L.MAX_BOUNCES
        try:
            L.MAX_BOUNCES = 1
            b = beam_of(FINAL)
            self.assertEqual(b.end, "trapped")
            self.assertFalse(b.stray)
        finally:
            L.MAX_BOUNCES = old

    def test_ray_box(self):
        box = S.FIXED[2]  # the detector
        hit = ray_box([0.30, L.DET_Y, L.BEAM_Z], [-1, 0, 0], box)
        self.assertAlmostEqual(hit[0], 0.30 - L.DET_FACE_X, places=9)
        self.assertEqual((hit[1], hit[2]), (0, 1.0))  # its +x face
        self.assertIsNone(ray_box([0.30, L.DET_Y, L.BEAM_Z], [1, 0, 0], box))  # pointing away
        self.assertEqual(ray_box([L.DET_X, L.DET_Y, L.BEAM_Z], [1, 0, 0], box), (0.0, -1, 0.0))  # from inside
        self.assertIsNone(ray_box([0.30, 0.0, L.BEAM_Z], [-1, 0, 0], box))  # beside it

    def test_mirror_helpers(self):
        self.assertAlmostEqual(S.mirror_yaw_deg(state("mirror_left", [0, 0, UP, 135.0])), -45.0, places=6)
        self.assertAlmostEqual(S.mirror_yaw_deg(state("mirror_left", [0, 0, UP, -100.0])), 80.0, places=6)
        self.assertAlmostEqual(S.normal_tilt_deg(state("mirror_left", [0, 0, UP, 30.0], tilt=2.0)), 2.0, places=6)
        self.assertAlmostEqual(S.turn_deg(rotation(quat(10)), rotation(quat(12.5))), 2.5, places=6)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = Tracker(HZ).update(scene(), snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.details["beam_end"], "detector")
        self.assertLess(t.max_lit_move, 1e-9)

    def test_spot_off_centre_scores_075(self):
        # the right mirror 12 mm too far forward: the beam meets its face 17 mm along (inside its 40 mm window) and
        # lands 12 mm off centre on the detector
        m2 = M2T + [0.012, 0, 0, 0]
        r, _ = play(procedure(m2=m2))
        self.assertEqual(r.score, 0.75)
        self.assertAlmostEqual(abs(r.details["det_dy_mm"]), 12.0, delta=0.2)

    def test_spot_tolerances(self):
        # the right mirror dx too far forward: the spot moves dx; past 14.1 mm the beam meets that mirror's rim
        for dx, want in ((0.009, 1.0), (0.011, 0.75), (0.014, 0.75), (0.0145, 0.25), (0.019, 0.25)):
            objs = scene(M1T, M2T + [dx, 0, 0, 0], B0)
            self.assertEqual(Tracker(HZ).update(objs, snap()).score, want, dx)

    def test_gripper_still_on_a_mirror_scores_075(self):
        f = procedure(settle=0)
        f.go(20, L=tcp_over(M1T), Lo=SHUT_M).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])

    def test_block_still_gripped_at_the_end_scores_075(self):
        f = procedure(settle=0)
        f.go(20, L=tcp_over(B0), Lo=SHUT_B).hold(30)  # back down and closed round the block on its place
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])

    def test_open_fingers_still_round_a_mirror_have_let_go(self):
        f = procedure(settle=0)
        f.go(20, L=tcp_over(M1T), Lo=OPEN).hold(30)  # open fingers back down round the left mirror
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.details["grippers_let_go"])

    def test_a_gripper_closed_far_away_has_let_go(self):
        f = procedure(settle=0)
        f.go(20, R=[0.20, -0.40, L.TOP + 0.15], Ro=0.0).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_both_mirrors_missing_the_detector_scores_half(self):
        objs = scene(M1T, M2T + [0, 0, 0, 8.0], B0)  # right mirror 8 deg off: the beam ends on the backstop
        b = beam_of(objs)
        self.assertEqual(b.end, "backstop")
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.5)

    def test_block_downstream_scores_half(self):
        # block moved onto the last leg, in front of the detector: off both mirrors, stopped on the table
        objs = scene(M1T, M2T, [0.20, L.DET_Y, UP, 0.0])
        self.assertEqual(beam_of(objs).end, "block")
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.5)

    def test_one_mirror_straight_onto_the_detector_scores_quarter(self):
        # one mirror far out on the beam, turned to send it diagonally onto the detector's face: one reflection
        objs = scene([0.40, L.LASER_Y, UP, 33.0], MR0, B0)
        b = beam_of(objs)
        self.assertEqual(b.reflections, ["mirror_left"])
        self.assertTrue(b.on_detector)
        self.assertEqual(Tracker(HZ).update(objs, snap()).score, 0.25)

    def test_block_left_in_scores_zero(self):
        r = Tracker(HZ).update(scene(M1T, M2T, BS), snap())
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["beam_end"], "block")

    def test_success_needs_stillness(self):
        objs = scene(M1T, M2T, B0, b_kw={"vel": (0.02, 0, 0)})
        r = Tracker(HZ).update(objs, snap())
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        objs = scene(M1T, M2T, B0, ml_kw={"spin": (0, 0, 0.6)})
        self.assertFalse(Tracker(HZ).update(objs, snap()).success)
        self.assertTrue(Tracker(HZ).update(FINAL, snap()).success)

    def test_result_alone_judges_the_current_beam(self):
        # result() (task.score) on a state the tracker was never fed still scores a stray beam or a fallen part 0
        self.assertEqual(Tracker(HZ).result(scene(ml=M1T), snap()).score, 0.0)
        fallen = scene(M1T, M2T, [L.BLOCK_X, 0.70, L.TOP - 0.3, 0.0])
        self.assertTrue(beam_of(fallen).on_detector)
        self.assertEqual(Tracker(HZ).result(fallen, snap()).score, 0.0)
        self.assertEqual(Tracker(HZ).result(FINAL, snap()).score, 1.0)

    def test_score_is_idempotent(self):
        _, t = play(procedure())
        a = t.result(FINAL, snap())
        b = t.result(FINAL, snap())
        self.assertEqual((a.score, a.success), (b.score, b.success))
        self.assertEqual(t.steps, len(procedure().frames))


class Rules(unittest.TestCase):
    def test_stray_beam_scores_zero_for_good(self):
        # the left mirror set in the live beam at 45 deg before the right one is there: stray, then fixed
        f = Film().hold(5).go(10, ml=M1T)  # (also moved while lit)
        f.go(10, mr=M2T).hold(30)
        r, t = play(f)
        self.assertIn("beam_stray", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_stray_alone_is_a_breach(self):
        t = Tracker(HZ)
        t.update(scene(ml=M1T), snap())  # appears lit and still: no movement, but the beam leaves the table
        self.assertEqual(sorted(t.breaches), ["beam_stray"])

    def test_mirror_lowered_into_the_live_beam(self):
        # the right mirror set first; the left one lowered into the live beam at the exact angle: the beam is never
        # stray, but a lit mirror moved (the block was skipped)
        f = Film().hold(5).go(20, mr=M2T)
        high = M1T + [0, 0, 0.06, 0]
        f.go(20, ml=high).go(30, ml=M1T).hold(30)
        r, t = play(f)
        self.assertNotIn("beam_stray", t.breaches)
        self.assertIn("lit_mirror_moved", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["beam_end"], "detector")

    def test_moving_a_mirror_behind_the_block_is_fine(self):
        f = Film(b=BS, ml=M1T + [0.02, 0.01, 0, 5.0], mr=M2T).hold(5)
        f.go(30, ml=M1T).hold(5)  # adjusted while dark
        f.go(20, b=BS + [0, 0, 0.08, 0]).go(20, b=B0).hold(30)  # block out: lit, still
        r, t = play(f)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 1.0)

    def test_turning_a_lit_mirror(self):
        start = scene(M1T, M2T, B0)
        for turn, breach in ((0.5, False), (1.5, True)):
            f = Film(ml=M1T, mr=M2T).hold(5).go(40, ml=M1T + [0, 0, 0, turn]).hold(5)
            r, t = play(f)
            self.assertEqual("lit_mirror_moved" in t.breaches, breach, turn)
        self.assertEqual(Tracker(HZ).update(start, snap()).score, 1.0)

    def test_nudging_a_lit_mirror(self):
        for dx, breach in ((0.002, False), (0.004, True)):
            f = Film(ml=M1T, mr=M2T).hold(5).go(20, mr=M2T + [dx, 0, 0, 0]).hold(5)
            _, t = play(f)
            self.assertEqual("lit_mirror_moved" in t.breaches, breach, dx)

    def test_a_new_lit_spell_starts_afresh(self):
        # lit, block in (dark), moved 10 mm, block out (lit again): no breach
        f = Film(ml=M1T, mr=M2T + [0.01, 0, 0, 0]).hold(5)
        f.go(20, b=BS).go(20, mr=M2T).go(20, b=B0).hold(10)
        r, t = play(f)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 1.0)

    def test_object_off_the_table(self):
        f = procedure(settle=0)
        f.go(10, b=[L.BLOCK_X, 0.70, UP, 0]).go(10, b=[L.BLOCK_X, 0.70, L.TOP - 0.3, 0]).hold(10)
        r, t = play(f)
        self.assertIn("object_off_table", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_a_breach_is_never_undone(self):
        t = Tracker(HZ)
        t.update(scene(ml=M1T), snap())  # stray
        for _ in range(40):
            r = t.update(FINAL, snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["first_breach_step"], 1)

    def test_holding_a_mirror_still_in_the_beam_is_not_moving_it(self):
        # the gripper still closed on the lit right mirror (it is not moving): 0.75, not 0, not 1.0
        t = Tracker(HZ)
        for _ in range(30):
            r = t.update(FINAL, snap(right_tcp=tcp_over(M2T), right_open=SHUT_M))
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 0.75)


class Exploits(unittest.TestCase):
    def test_skip_the_block_with_a_perfect_drop(self):
        # right mirror first, then the left mirror dropped into the beam from above at the exact angle
        f = Film().hold(5).go(20, mr=M2T).go(20, ml=M1T + [0, 0, 0.05, 0])
        f.go(3, ml=M1T).hold(30)  # falls 50 mm in 3 steps
        r, t = play(f)
        self.assertIn("lit_mirror_moved", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_slide_the_mirror_in_sideways(self):
        f = Film(mr=M2T).hold(5).go(10, ml=M1T + [0, 0.06, 0, 0]).go(30, ml=M1T).hold(30)
        r, t = play(f)
        self.assertIn("lit_mirror_moved", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_slide_the_mirror_in_by_its_rim(self):
        # right mirror first, then the left one slid into the live beam along -y at exactly 45 deg and stopped
        # with the beam e mm inside its face's edge: on the rim (e <= 10) the beam reflects off nothing; past it,
        # the mirror has moved over 7 mm in the beam (the rim's 10 mm x cos 45) before the beam reached its window
        c45 = math.sqrt(0.5)
        for e_mm in (1.0, 3.0, 9.0, 10.5, 12.0, 15.0):
            yl = -L.MIRROR_SIZE[1] / 2 + e_mm / 1000
            off = -L.MIRROR_SIZE[0] / 2 * np.array([c45, c45]) + yl * np.array([-c45, c45])
            final = np.array([0.32 - off[0], L.LASER_Y - off[1], UP, 45.0])
            f = Film(mr=M2T).hold(5).go(20, ml=final + [0, 0.02, 0, 0]).go(40, ml=final).go(20, Lo=OPEN).hold(30)
            r, t = play(f)
            self.assertEqual(r.score, 0.0, e_mm)
            if e_mm > 10.0:
                self.assertIn("lit_mirror_moved", t.breaches, e_mm)
                self.assertGreater(t.max_lit_move, 0.007, e_mm)
            else:
                self.assertEqual(r.details["beam_end"], "mirror_left", e_mm)

    def test_swing_a_mirror_in_between_two_steps(self):
        # dark at one control step, in place with the beam on the middle of its window at the next: it moved 40 mm
        # in the live beam (measured from its last place in the dark)
        f = Film(mr=M2T).hold(5).go(20, ml=M1T + [0, 0.04, 0, 0]).go(1, ml=M1T).hold(30)
        r, t = play(f)
        self.assertIn("lit_mirror_moved", t.breaches)
        self.assertAlmostEqual(t.max_lit_move, 0.04, places=6)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["beam_end"], "detector")

    def test_whip_a_lit_mirror_out_between_two_steps(self):
        # both set and lit; the left mirror jerked 50 mm out of the beam in one step (dark at the next), then put
        # back behind the block and the block taken out: it moved in the live beam
        f = Film(ml=M1T, mr=M2T).hold(10).go(1, ml=M1T + [0, 0.05, 0, 0])
        f.go(10, b=BS).go(20, ml=M1T).go(20, b=B0).hold(30)
        r, t = play(f)
        self.assertIn("lit_mirror_moved", t.breaches)
        self.assertEqual(t.breaches["lit_mirror_moved"], 12)
        self.assertEqual(r.score, 0.0)

    def test_block_in_then_move_is_fine(self):
        # both lit; the block goes in (both dark, neither moved), the right mirror is adjusted 10 mm, block out
        f = Film(ml=M1T, mr=M2T + [0.010, 0, 0, 0]).hold(10).go(10, b=BS).go(20, mr=M2T).go(20, b=B0).hold(30)
        r, t = play(f)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 1.0)

    def test_mirrors_placed_but_block_never_removed(self):
        f = Film().hold(5)
        carry(f, "L", "b", BS, SHUT_B)
        carry(f, "R", "mr", M2T, SHUT_M)
        carry(f, "L", "ml", M1T, SHUT_M)
        r, t = play(f.hold(30))
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 0.0)

    def test_left_arm_alone(self):
        # block, left mirror, block out: the beam runs off the right edge of the table
        f = Film().hold(5)
        carry(f, "L", "b", BS, SHUT_B)
        carry(f, "L", "ml", M1T, SHUT_M)
        carry(f, "L", "b", B0, SHUT_B)
        r, t = play(f.hold(30))
        self.assertIn("beam_stray", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_right_arm_alone(self):
        f = Film().hold(5)
        carry(f, "R", "mr", M2T, SHUT_M)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["beam_end"], "beam_dump")


class Arms(unittest.TestCase):
    def test_arms_unavailable_let_go_taken_as_met_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertFalse(r.details["arms_available"])
        self.assertIsNone(r.details["grippers_let_go"])

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = snap(left_tcp=(np.nan, 0.3, 0.9))
        r = t.update(FINAL, bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 1.0)
        t.update(FINAL, {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=tcp_over(M1T), left_open=SHUT_M)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, SHUT_M)
        self.assertFalse(S.let_go(a["left"], {n: FINAL[n] for n in L.LOOSE}))
        self.assertTrue(S.let_go(a["right"], {n: FINAL[n] for n in L.LOOSE}))

    def test_finger_zone(self):
        a = snap(left_tcp=(0.2, 0.2, 0.9), left_open=OPEN)["left"]
        self.assertAlmostEqual(S.zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)
        pts, _ = S.finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)


class States(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        _, t = play(procedure())
        steps, refs = t.steps, dict(t.lit_ref)
        for bad in ({"pos": np.array([np.nan, 0, 0])}, {"quat_xyzw": np.zeros(4)},
                    {"lin_vel": np.array([np.inf, 0, 0])}, {"low": np.array([0, np.nan, 0])}):
            objs = dict(FINAL)
            s = vars(FINAL["mirror_left"]).copy()
            s.update(bad)
            objs["mirror_left"] = SimpleNamespace(**s)
            self.assertFalse(valid(objs["mirror_left"]))
            r = t.update(objs, snap())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.bad_steps, 4)
        self.assertEqual(t.steps, steps + 4)
        self.assertEqual(set(t.lit_ref), set(refs))
        self.assertEqual(t.update(FINAL, snap()).score, 1.0)

    def test_missing_object(self):
        objs = dict(FINAL)
        del objs["block"]
        t = Tracker(HZ)
        self.assertEqual(t.update(objs, snap()).score, 0.0)
        self.assertEqual(t.result(objs).score, 0.0)
        self.assertEqual(t.update(None, None).score, 0.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(300):
            objs = {}
            for n in L.LOOSE:
                q = rng.normal(size=4)
                p = [rng.uniform(-0.3, 1.0), rng.uniform(-0.8, 0.8), L.TOP + rng.uniform(-0.2, 0.3), 0.0]
                objs[n] = state(n, p, q=q / np.linalg.norm(q))
            r = t.update(objs, snap(left_open=rng.uniform(), right_open=rng.uniform()))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Resets(unittest.TestCase):
    def _done(self):
        f = Film().hold(5).go(10, ml=M1T)  # the left mirror swung into the live beam: stray and moved while lit
        r, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self._done()
        r = t.update(scene(), snap())
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self._done()
        t.update(scene(ML0 + [0.02, -0.02, 0, 0], MR0 + [-0.02, 0.01, 0, 0], B0 + [0.01, 0.02, 0, 0]), snap())
        self.assertEqual(t.breaches, {})

    def test_gradual_return_does_not_reset(self):
        t = self._done()
        f = Film(ml=M1T, b=B0).go(40, ml=ML0)
        play(f, t)
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self._done()
        t.update(scene(ml=[0.25, 0.45, UP, 0]), snap())
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        task = _load_task()
        tr = task._TRACKER
        tr.update(scene(ml=M1T), snap())
        self.assertTrue(tr.breach)
        rigstate.offline_reset()  # what the checker (and the wrapped embodiment reset) triggers
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)


def _load_task():
    stub = HERE.parent.parent / "tools" / "offline" / "yam_stub"
    if str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
        sys.path.insert(0, str(stub.parent))
    if "task" in sys.modules:
        return sys.modules["task"]
    return importlib.import_module("task")


@unittest.skipUnless((HERE.parent.parent / "tools" / "offline" / "yam_stub").is_dir(), "offline stub not present")
class TaskFile(unittest.TestCase):
    def test_objects(self):
        task = _load_task()
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in (*L.LOOSE, "laser", "beam_dump", "detector", "backstop"):
            self.assertIn(n, names)
        loose = {o.name for o in task.OBJECTS if getattr(o, "mass", None)}
        self.assertEqual(loose, set(L.LOOSE))

    def test_instruction_matches_task_json(self):
        task = _load_task()
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], task.INSTRUCTION)

    def test_hooks(self):
        task = _load_task()
        rigstate.offline_reset()
        snap()
        self.assertFalse(task.success(scene()))
        self.assertEqual(task.score(scene()), 0.0)
        rigstate.offline_reset()
        for _ in range(3):
            snap()
            ok = task.success(FINAL)
        self.assertTrue(ok)
        self.assertEqual(task.score(FINAL), 1.0)

    def test_trace_rows_and_a_bad_path(self):
        task = _load_task()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                rigstate.offline_reset()
                snap()
                task.success(FINAL)
                row = json.loads(pathlib.Path(path).read_text().splitlines()[-1])
                for k in ("score", "beam_end", "reflections", "det_off_mm", "breaches", "mirror_left_yaw_deg",
                          "max_lit_move_mm", "arms_unavailable", "left_tcp", "mirror_left"):
                    self.assertIn(k, row)
                os.environ["TASK_TRACE"] = os.path.join(d, "no", "such", "dir", "trace.jsonl")
                task.success(FINAL)  # must not raise
            finally:
                os.environ.pop("TASK_TRACE", None)

    def test_demo_switches_and_timing(self):
        task = _load_task()
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(task.demo()), {"right"})
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(set(task.demo()), {"left"})
        os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(task.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        left = plan["left"]
        lift = next(i for i, w in enumerate(left) if i > 24 and w.label == "lift it")  # the block comes out
        t_lift = sum(w.duration for w in left[:lift])
        self.assertGreater(t_lift, task.RIGHT_MIRROR_SET_S + 1.0)
        t_total = max(sum(w.duration for w in p) for p in plan.values())
        self.assertLessEqual((t_total + 3.0) * HZ, 2000)  # the policy stops inside the run limit
        for w in left + plan["right"]:  # gripper changes at least 1.5 s (the CLI's 0.05-per-step limiter)
            self.assertGreater(w.duration, 0.0)


if __name__ == "__main__":
    unittest.main()
