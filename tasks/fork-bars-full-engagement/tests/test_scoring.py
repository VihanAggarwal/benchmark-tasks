"""Offline tests for scoring.py (and the rigstate hooks, the offline ride model and the demo in task.py): synthetic
object states and synthetic arm snapshots fed through rigstate.offline_feed, no simulator.
python3 -m unittest discover -s tests -v"""

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
    Tracker, bars_out, drop_speed, finger_zone, insertion, is_lifted, lowest, on_rack, pocket_insertions, rotation,
    tilt_deg, valid, yaw_off_deg, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
TOP = L.TOP
HC = L.CRATE_HALF[2]  # half the crate's height
BH = L.BAR_HALF_H
TIPS = 0.013  # pad tips below the TCP
AHEAD = L.BAR_HALF_LEN - L.BAR_GRIP_FROM_REAR  # a gripped bar's centre is 65 mm ahead of the TCP ...
BELOW = 0.006  # ... and 6 mm below it
OPEN, GRIP = 1.0, L.BAR_SIZE[1] / L.FULL_GAP  # rigstate openings: fully open, closed on a 24 mm bar
Q0 = np.array([0.0, 0.0, 0.0, 1.0])

C_START = np.array([L.CRATE_X, L.CRATE_Y, TOP + L.FLOOR_TOP + HC])
BL_START = np.array([L.BAR_X, +L.POCKET_Y, TOP + BH])
BR_START = np.array([L.BAR_X, -L.POCKET_Y, TOP + BH])
READY_L = np.array([0.323, 0.305, TOP + 0.26])
READY_R = np.array([0.323, -0.305, TOP + 0.26])
TO_TCP = np.array([-AHEAD, 0.0, BELOW])  # from a gripped bar's centre to the TCP
X = np.array([1.0, 0.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def mul(q1, q2):
    """Quaternion product q1 * q2 (xyzw)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def box(center, size, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def crate(center=C_START, quat=Q0, **kw):
    return box(center, L.CRATE_SIZE, quat, **kw)


def crate_at(x=L.CRATE_X, y=L.CRATE_Y, bottom=L.FLOOR_TOP, quat=Q0, **kw):
    """The crate with the centre of its underside ``bottom`` above the table top."""
    return crate(np.array([x, y, TOP + bottom]) + rotation(quat) @ np.array([0, 0, HC]), quat, **kw)


def bar(center, quat=Q0, **kw):
    return box(center, L.BAR_SIZE, quat, **kw)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("c", "cq", "bl", "br", "L", "Lo", "R", "Ro")
VEC = ("c", "bl", "br", "L", "R")


class Film:
    """A trial as a list of frames: the crate's centre and orientation, both bars' centres, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps (``d`` adds to the current ones); velocities come from
    the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"c": C_START, "cq": Q0, "bl": BL_START, "br": BR_START, "L": READY_L, "Lo": OPEN,
                    "R": READY_R, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, d=None, **to):
        """Over ``n`` steps: to the values in ``to``, and/or by the offsets in ``d`` ({key: offset})."""
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for k, off in (d or {}).items():
            b[k] = np.asarray(b[k], float) + np.asarray(off, float)
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k == "cq":
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def both(self, n, off, crate_too=False):
        """Both arms move their bars together by ``off`` (and the crate with them)."""
        d = {"L": off, "R": off, "bl": off, "br": off}
        if crate_too:
            d["c"] = off
        return self.go(n, d)


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        vel = {k: np.zeros(3) if prev is None else (fr[k] - prev[k]) / DT for k in ("c", "bl", "br")}
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        bars = (bar(fr["bl"], vel=vel["bl"]), bar(fr["br"], vel=vel["br"]))
        r = t.update(crate(fr["c"], fr["cq"], vel=vel["c"]), bars, a)
        prev = fr
    return r, t


def grip_bars(f, bl_x=None, br_x=None):
    """Both arms come down over their bars' rear ends and close on them (about 5 s)."""
    gl, gr = f.cur["bl"] + TO_TCP, f.cur["br"] + TO_TCP
    f.go(40, L=gl + 0.08 * Z, R=gr + 0.08 * Z).go(30, L=gl, R=gr).go(30, Lo=GRIP, Ro=GRIP)
    return f


def lift_and_carry(f, lift=0.051, carry=0.142, insert=0.100):
    """Push both bars ``insert`` in, raise them to the crate's underside, lift it by ``lift``, carry it ``carry``
    toward the rack."""
    f.both(60, insert * X)
    f.both(10, 0.012 * Z)  # the bars' tops reach the crate's underside (0.024 -> 0.036)
    f.both(50, lift * Z, crate_too=True)
    f.both(60, carry * X, crate_too=True)
    return f


def set_down_and_withdraw(f, drop=0.015, back=0.227):
    """Lower the crate onto the rack's bearers, lower the bars 4 mm clear, slide them out and back."""
    f.both(30, -drop * Z, crate_too=True)
    f.both(10, -0.004 * Z)
    f.both(60, -back * X)
    return f


def put_down_and_let_go(f, rise=0.08):
    z_down = TOP + BH - f.cur["bl"][2]
    f.both(40, z_down * Z)
    f.go(30, Lo=OPEN, Ro=OPEN)
    f.go(30, d={"L": rise * Z, "R": rise * Z})
    return f


def procedure(settle=30, **kw):
    """The intended trial: grip both bars, push them fully in, lift the crate 51 mm level, carry it 142 mm, set it
    on the rack's bearers, slide the bars out, put them down, let go, rise; all still for ``settle`` steps."""
    f = Film().hold(10)
    grip_bars(f)
    lift_and_carry(f, **kw)
    set_down_and_withdraw(f, drop=kw.get("lift", 0.051) + L.FLOOR_TOP - L.RACK_TOP)
    put_down_and_let_go(f)
    return f.hold(settle)


# == geometry ======================================================================================================
class Geometry(unittest.TestCase):
    def test_insertion_measures_the_bar_under_the_crate(self):
        c = crate_at()
        self.assertEqual(insertion(c, bar(BL_START)), (None, 0.0))  # tip at the pocket mouth: nothing under it
        for d in (0.073, 0.095, 0.100):
            side, length = insertion(c, bar(BL_START + d * X))
            self.assertEqual(side, +1)
            self.assertAlmostEqual(length, d, places=6)
        self.assertAlmostEqual(insertion(c, bar(BR_START + 0.100 * X))[1], 0.100, places=6)
        self.assertEqual(insertion(c, bar(BR_START + 0.100 * X))[0], -1)
        self.assertAlmostEqual(insertion(c, bar(BL_START + 0.150 * X))[1], L.CRATE_SIZE[0], places=6)  # through

    def test_insertion_needs_the_pocket_line_and_the_bar_just_under_the_crate(self):
        c = crate_at()
        self.assertEqual(insertion(c, bar(BL_START + [0.1, 0.016, 0]))[0], None)  # 16 mm off the pocket line
        self.assertEqual(insertion(c, bar(BL_START + [0.1, 0.014, 0]))[0], +1)
        on_top = BL_START + [0.1, 0, L.FLOOR_TOP + 2 * HC]
        self.assertEqual(insertion(c, bar(on_top))[0], None)  # lying on the crate
        lifted = crate_at(bottom=L.FLOOR_TOP + 0.030)
        self.assertEqual(insertion(lifted, bar(BL_START + 0.1 * X))[0], None)  # on the table, 42 mm under it
        self.assertEqual(insertion(lifted, bar(BL_START + [0.1, 0, 0.030]))[0], +1)  # holding it up
        skew = mul(about("z", 12), Q0)  # turned 12 deg: its tip end leaves the pocket line
        self.assertEqual(insertion(c, bar(BL_START + 0.1 * X, skew))[0], None)

    def test_insertion_is_the_length_of_the_centre_line(self):
        """A bar turned a little in its pocket: the length of its centre line under the crate, not its run along x."""
        c = crate_at()
        a = math.radians(4.0)
        b = bar(BL_START + 0.1 * X, about("z", 4.0))
        tip_x = L.BAR_X + 0.1 + L.BAR_HALF_LEN * math.cos(a)
        side, length = insertion(c, b)
        self.assertEqual(side, +1)
        self.assertAlmostEqual(length, (tip_x - L.FLOOR_X0) / math.cos(a), places=9)
        self.assertGreater(length, tip_x - L.FLOOR_X0)

    def test_insertion_follows_the_crate(self):
        turned = crate_at(quat=about("z", 180))
        self.assertAlmostEqual(pocket_insertions(turned, (bar(BL_START + 0.1 * X), bar(BR_START + 0.1 * X)))[+1],
                               0.100, places=6)
        moved = crate_at(x=L.CRATE_X + 0.05, bottom=0.060)
        bl = bar(BL_START + [0.15, 0, 0.060 - L.FLOOR_TOP + 0.012])
        self.assertAlmostEqual(insertion(moved, bl)[1], 0.100, places=6)

    def test_tilt_is_folded_for_a_plain_box(self):
        self.assertAlmostEqual(tilt_deg(crate_at(quat=about("x", 4))), 4.0, places=6)
        self.assertAlmostEqual(tilt_deg(crate_at(quat=about("x", 180))), 0.0, places=6)
        self.assertAlmostEqual(yaw_off_deg(crate_at(quat=about("z", 178))), 2.0, places=6)

    def test_lowest_and_lifted(self):
        self.assertAlmostEqual(lowest(crate_at()), L.FLOOR_TOP, places=9)
        self.assertFalse(is_lifted(crate_at(bottom=L.FLOOR_TOP + 0.0029)))
        self.assertTrue(is_lifted(crate_at(bottom=L.FLOOR_TOP + 0.0031)))
        self.assertFalse(is_lifted(crate_at(x=0.465, bottom=L.RACK_TOP)))  # on the rack's bearers
        self.assertTrue(is_lifted(crate_at(x=0.465, bottom=L.RACK_TOP + 0.008)))
        tilted = crate_at(quat=about("x", 3))  # rolled about its far-left edge? lowest corner counts
        self.assertLess(lowest(tilted), L.FLOOR_TOP)

    def test_on_rack_tolerances(self):
        self.assertTrue(on_rack(crate_at(x=0.465, bottom=L.RACK_TOP)))
        self.assertTrue(on_rack(crate_at(x=0.474, bottom=L.RACK_TOP)))  # far end at 0.529
        self.assertFalse(on_rack(crate_at(x=0.476, bottom=L.RACK_TOP)))  # 0.531
        self.assertTrue(on_rack(crate_at(x=0.456, bottom=L.RACK_TOP)))  # near end at 0.401
        self.assertFalse(on_rack(crate_at(x=0.465, y=0.016, bottom=L.RACK_TOP)))
        self.assertFalse(on_rack(crate_at(x=0.465, bottom=L.RACK_TOP + 0.0035)))  # still held up
        self.assertFalse(on_rack(crate_at(x=0.465, bottom=L.FLOOR_TOP)))  # on the platform, not the bearers
        self.assertFalse(on_rack(crate_at(x=0.465, bottom=L.RACK_TOP, quat=about("x", 3.5))))
        self.assertFalse(on_rack(crate_at(x=0.465, bottom=L.RACK_TOP, quat=about("z", 6))))
        self.assertTrue(on_rack(crate_at(x=0.465, bottom=L.RACK_TOP, quat=about("z", 180))))

    def test_bars_out(self):
        c = crate_at(x=0.465, bottom=L.RACK_TOP)
        under = (bar([0.430, 0.12, TOP + 0.056]), bar([0.430, -0.12, TOP + 0.056]))
        self.assertFalse(bars_out(c, under))
        out = (bar([0.399 - L.BAR_HALF_LEN, 0.12, TOP + 0.056]), bar([0.399 - L.BAR_HALF_LEN, -0.12, TOP + 0.056]))
        self.assertTrue(bars_out(c, out))
        on_top = (out[0], bar([0.465, 0.0, TOP + L.RACK_TOP + 2 * HC + BH]))
        self.assertFalse(bars_out(c, on_top))
        # 'wholly outside': a tip 0.5 mm under the crate's near edge (x 0.410) is still under it
        tip_in = (out[0], bar([0.4105 - L.BAR_HALF_LEN, -0.12, TOP + 0.056]))
        self.assertFalse(bars_out(c, tip_in))
        tip_out = (out[0], bar([0.4095 - L.BAR_HALF_LEN, -0.12, TOP + 0.056]))
        self.assertTrue(bars_out(c, tip_out))

    def test_finger_zone_covers_the_pads(self):
        a = snap(left_open=GRIP)["left"]
        pts, r = finger_zone(a)
        self.assertEqual(r, 0.024)  # task.json: 'any point within 24 mm of the segment'
        # a finger's outer corner when closed on the bar (6 mm fingers measured): 22.5 mm, inside the zone
        self.assertGreaterEqual(r, math.hypot(0.012 + 0.006, L.FINGER_HALF_WIDTH))
        for opening in (0.0, 0.3, OPEN):  # the same at every opening: wider fingers are not followed further out
            self.assertEqual(zone_radius(snap(left_open=opening)["left"]), r)
        self.assertAlmostEqual(np.linalg.norm(pts[0] - a.tips), 0.0)
        self.assertAlmostEqual(float(pts[-1][2] - a.tips[2]), L.FINGER_LEN, places=6)

    def test_drop_speed(self):
        self.assertAlmostEqual(drop_speed(crate_at(vel=(0, 0, -0.3))), 0.3, places=6)
        spin = crate_at(spin=(2.0, 0, 0))  # rolling: its -y edge goes down at 2 rad/s x 0.15 m
        self.assertGreater(drop_speed(spin), 0.25)

    def test_valid(self):
        self.assertTrue(valid(crate_at()))
        self.assertFalse(valid(crate_at(vel=(math.nan, 0, 0))))
        self.assertFalse(valid(crate(C_START, quat=np.zeros(4))))
        self.assertFalse(valid(SimpleNamespace(pos=C_START)))


# == tiers =========================================================================================================
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertFalse(t.breach)

    def test_full_procedure_scores_one(self):
        f = procedure()
        r, t = play(f)
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertFalse(t.breach)
        self.assertAlmostEqual(t.min_lift_insertion, 0.100, places=6)
        self.assertTrue(t.carried and t.both_full)
        self.assertGreater(t.min_zone_gap, 0.015)
        # never a success while the bars are still held
        gripped = next(i for i, fr in enumerate(f.frames) if fr["Lo"] < 0.5)
        opened = next(i for i, fr in enumerate(f.frames) if i > gripped and fr["Lo"] >= L.GRIPPER_OPEN)
        self.assertFalse(any(play(f, until=k)[0].success for k in range(opened - 40, opened + 1, 10)))

    def test_bars_left_in_the_pockets_scores_075(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f)
        f.both(30, -0.015 * Z, crate_too=True).both(10, -0.004 * Z)
        f.go(30, Lo=OPEN, Ro=OPEN).go(30, d={"L": 0.10 * Z, "R": 0.10 * Z}).hold(30)
        r, _ = play(f)
        self.assertTrue(r.details["on_rack"])
        self.assertFalse(r.details["bars_out"])
        self.assertEqual(r.score, 0.75)

    def test_lifted_on_partial_insertion_scores_075(self):
        r, t = play(procedure(insert=0.080))
        self.assertAlmostEqual(t.min_lift_insertion, 0.080, places=6)
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.75)

    def test_gripper_not_let_go_scores_075(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f)
        set_down_and_withdraw(f)
        f.hold(30)  # still holding both bars, out from under the crate
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        f.go(30, Lo=OPEN, Ro=OPEN).hold(30)  # open but the fingers 0 mm from the bars
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])
        f.go(20, d={"L": 0.05 * Z, "R": 0.05 * Z}).hold(30)  # up: the tips 37 mm over the bars, 30 mm clear
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)

    def test_carried_but_not_on_the_rack_scores_half(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f, carry=0.060).hold(40)  # held up, 60 mm toward the rack
        r, t = play(f)
        self.assertTrue(t.carried)
        self.assertEqual(r.score, 0.5)
        f.both(30, -0.051 * Z, crate_too=True)  # set back down on the floor bearers (its far end now over the gap)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_carried_only_40mm_scores_quarter(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f, carry=0.040).hold(40)
        r, t = play(f)
        self.assertFalse(t.carried)
        self.assertEqual(r.score, 0.25)

    def test_lifted_only_15mm_and_carried_scores_quarter(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f, lift=0.015, carry=0.060).hold(40)
        r, t = play(f)
        self.assertFalse(t.carried)
        self.assertEqual(r.score, 0.25)

    def test_both_bars_fully_in_scores_quarter(self):
        f = Film().hold(10)
        grip_bars(f)
        f.both(60, 0.095 * X).hold(20)
        r, t = play(f)
        self.assertTrue(t.both_full)
        self.assertEqual(r.score, 0.25)

    def test_one_bar_fully_in_or_both_short_scores_zero(self):
        f = Film().hold(10)
        grip_bars(f)
        f.go(60, d={"L": 0.100 * X, "bl": 0.100 * X}).hold(20)
        self.assertEqual(play(f)[0].score, 0.0)
        g = Film().hold(10)
        grip_bars(g)
        g.both(60, 0.094 * X).hold(20)
        self.assertEqual(play(g)[0].score, 0.0)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        f.go(5, d={"c": 0.005 * X})  # still creeping at 2 cm/s
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        f.hold(5)
        r, _ = play(f)
        self.assertTrue(r.success)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        fr = f.frames[-1]
        args = (crate(fr["c"]), (bar(fr["bl"]), bar(fr["br"])), snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]))
        for _ in range(3):
            self.assertEqual(t.result(*args).score, 1.0)
        self.assertEqual(t.steps, len(f.frames))


# == the rules that score 0 =========================================================================================
class Breaches(unittest.TestCase):
    def test_lifted_on_the_tips(self):
        r, t = play(procedure(insert=0.050))
        self.assertIn("lifted_on_tips", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_two_thirds_is_the_limit(self):
        self.assertNotIn("lifted_on_tips", play(procedure(insert=0.074))[1].breaches)
        self.assertIn("lifted_on_tips", play(procedure(insert=0.072))[1].breaches)

    def test_lifting_on_one_bar_tilts_past_5_deg(self):
        f = Film().hold(10)
        grip_bars(f)
        f.both(60, 0.100 * X)
        f.go(10, d={"L": 0.012 * Z, "bl": 0.012 * Z})
        # the left bar alone raises the crate's +y side; it pivots on its -y edge: 25 mm over 270 mm is 5.3 deg
        for k in range(1, 11):
            ang = math.degrees(math.atan2(0.0025 * k, L.POCKET_Y + L.CRATE_HALF[1]))
            q = about("x", ang)
            edge = np.array([L.CRATE_X, -L.CRATE_HALF[1], TOP + L.FLOOR_TOP])
            centre = edge + rotation(q) @ np.array([0, L.CRATE_HALF[1], HC])
            f.go(1, d={"L": 0.0025 * Z, "bl": 0.0025 * Z}, c=centre, cq=q)
        r, t = play(f)
        self.assertIn("tilted", t.breaches)
        self.assertNotIn("lifted_on_tips", t.breaches)  # its low edge never left the bearer
        self.assertEqual(r.score, 0.0)

    def test_tilt_under_5_deg_is_allowed(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f)
        c0 = f.cur["c"]
        f.go(4, cq=about("x", 4.5)).go(4, cq=Q0, c=c0)  # a wobble of 4.5 deg while carried
        set_down_and_withdraw(f)
        put_down_and_let_go(f)
        f.hold(30)
        r, t = play(f)
        self.assertGreater(t.max_tilt, 4.4)
        self.assertEqual(r.score, 1.0)

    def test_too_high(self):
        r, t = play(procedure(lift=0.060))  # lowest point 0.096
        self.assertIn("too_high", t.breaches)
        self.assertEqual(r.score, 0.0)
        r, t = play(procedure(lift=0.058))  # 0.094
        self.assertEqual(r.score, 1.0)

    def test_dropped_onto_the_rack(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f)
        f.go(1, d={"c": -0.015 * Z})  # let fall 15 mm in one step: 0.3 m/s
        f.both(30, -0.019 * Z)
        f.both(60, -0.227 * X)
        put_down_and_let_go(f)
        r, t = play(f.hold(30))
        self.assertIn("fell", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_crate_off_the_table(self):
        f = Film().hold(10)
        f.go(10, c=C_START + [0.0, 0.0, -0.08])
        r, t = play(f)
        self.assertIn("left_table", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_gripper_touching_the_crate(self):
        f = Film().hold(10)
        f.go(30, L=np.array([0.255, 0.05, TOP + 0.05]))  # open fingers against the crate's front face
        f.go(30, L=READY_L)
        grip_bars(f)
        lift_and_carry(f)
        set_down_and_withdraw(f)
        put_down_and_let_go(f)
        r, t = play(f.hold(30))
        self.assertIn("gripper_touched", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertLess(t.min_zone_gap, L.TOUCH_GAP)

    def test_fingers_6mm_from_the_crate_are_not_a_touch(self):
        r = zone_radius(snap()["left"])
        tcp = np.array([L.FLOOR_X0 - r - 0.006, 0.05, TOP + 0.05])
        f = Film().hold(5).go(20, L=tcp + 0.15 * Z).go(20, L=tcp).go(20, L=tcp + 0.15 * Z)
        _, t = play(f)
        self.assertNotIn("gripper_touched", t.breaches)
        self.assertAlmostEqual(t.min_zone_gap, 0.006, places=4)
        tcp = tcp + 0.002 * X
        f = Film().hold(5).go(20, L=tcp + 0.15 * Z).go(20, L=tcp).go(20, L=tcp + 0.15 * Z)
        self.assertIn("gripper_touched", play(f)[1].breaches)

    def test_letting_go_of_a_bar_fully_in_is_not_a_touch(self):
        f = Film().hold(10)
        grip_bars(f)
        f.both(60, 0.100 * X).go(30, Lo=OPEN, Ro=OPEN).go(30, d={"L": 0.08 * Z, "R": 0.08 * Z})
        r, t = play(f)
        self.assertFalse(t.breach)
        self.assertAlmostEqual(t.min_zone_gap, 0.045 - zone_radius(snap(left_open=GRIP)["left"]), places=3)
        self.assertEqual(r.score, 0.25)

    def test_a_breach_is_never_undone(self):
        f = Film().hold(10)
        grip_bars(f)
        f.both(30, 0.050 * X).both(10, 0.012 * Z).both(10, 0.010 * Z, crate_too=True)  # up 10 mm on the tips
        f.both(10, -0.010 * Z, crate_too=True).both(10, -0.012 * Z)  # back down
        f.both(30, 0.050 * X)  # then fully in, and the rest done right
        f.both(10, 0.012 * Z).both(50, 0.051 * Z, crate_too=True).both(60, 0.142 * X, crate_too=True)
        set_down_and_withdraw(f)
        put_down_and_let_go(f)
        r, t = play(f.hold(30))
        self.assertIn("lifted_on_tips", t.breaches)
        self.assertTrue(r.details["on_rack"] and r.details["bars_out"] and r.details["grippers_let_go"])
        self.assertEqual(r.score, 0.0)


# == ways round the rules ===========================================================================================
class Exploits(unittest.TestCase):
    def test_pushing_the_crate_toward_the_rack_with_the_bar_tips(self):
        f = Film().hold(10)
        grip_bars(f)
        # the bars' tips against the crate's face, raised 40 mm, shove it along the bearers onto the platform
        f.both(20, 0.040 * Z)
        f.both(80, 0.030 * X)
        f.go(80, d={"L": 0.110 * X, "R": 0.110 * X, "bl": 0.110 * X, "br": 0.110 * X, "c": 0.080 * X})
        f.hold(20)
        r, t = play(f)
        self.assertFalse(r.details["on_rack"])  # it stops against the rack bearers' faces, on the platform
        self.assertEqual(r.score, 0.0)

    def test_one_arm_inserts_both_bars(self):
        """The best one arm can do: push one bar in, then the other; it cannot lift both."""
        f = Film().hold(10)
        g = f.cur["bl"] + TO_TCP
        f.go(40, R=g + 0.08 * Z).go(30, R=g).go(30, Ro=GRIP).go(60, d={"R": 0.1 * X, "bl": 0.1 * X})
        f.go(30, Ro=OPEN).go(30, d={"R": 0.08 * Z})
        g = f.cur["br"] + TO_TCP
        f.go(40, R=g + 0.08 * Z).go(30, R=g).go(30, Ro=GRIP).go(60, d={"R": 0.1 * X, "br": 0.1 * X})
        r, t = play(f.hold(20))
        self.assertTrue(t.both_full)
        self.assertEqual(r.score, 0.25)

    def test_carrying_it_high_over_the_rack(self):
        r, t = play(procedure(lift=0.070))
        self.assertIn("too_high", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_sliding_the_bars_out_while_it_is_still_up(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f)
        f.both(40, -0.060 * X)  # pulled back 60 mm with the crate still up: 40 mm in
        r, t = play(f)
        self.assertIn("lifted_on_tips", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_gripper_steadying_the_crate_while_it_is_lifted(self):
        f = Film().hold(10)
        grip_bars(f)
        lift_and_carry(f, carry=0.0)
        # the right gripper lets its bar go and reaches over to steady the crate from above
        f.go(20, Ro=OPEN).go(30, R=np.array([L.CRATE_X, -0.05, TOP + 0.087 + 2 * HC + TIPS + 0.002]))
        r, t = play(f)
        self.assertIn("gripper_touched", t.breaches)
        self.assertEqual(r.score, 0.0)


# == the arms =======================================================================================================
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        f = procedure()
        r, t = play(f, arms=False)
        self.assertEqual(r.score, 1.0)  # let-go taken as met
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertFalse(r.details["arms_available"])

    def test_arm_rules_resume_when_the_arms_come_back(self):
        f = Film().hold(10)
        f.go(30, L=np.array([0.255, 0.05, TOP + 0.05]))
        t = Tracker(HZ)
        play(f, tracker=t, arms=False)
        self.assertFalse(t.breach)
        play(f, tracker=t, arms=True)
        self.assertIn("gripper_touched", t.breaches)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(crate_at(), (bar(BL_START), bar(BR_START)), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["opening"])

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=[0.255, 0.05, TOP + 0.05])
        self.assertIs(a, rigstate.arms())
        t = Tracker(HZ)
        t.update(crate_at(), (bar(BL_START), bar(BR_START)), rigstate.arms())
        self.assertIn("gripper_touched", t.breaches)


# == robustness ======================================================================================================
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = procedure()
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        fr = f.frames[-1]
        arms = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
        good_c, good_b = crate(fr["c"]), (bar(fr["bl"]), bar(fr["br"]))
        for bad_c, bad_b in (
            (crate(fr["c"] * np.nan), good_b),
            (crate(fr["c"], vel=(np.inf, 0, 0)), good_b),
            (crate(fr["c"], quat=np.zeros(4)), good_b),
            (good_c, (bar(fr["bl"], quat=[np.nan] * 4), good_b[1])),
            (good_c, (good_b[0], SimpleNamespace(pos=fr["br"]))),
        ):
            before = (dict(t.breaches), t.min_lift_insertion, t.carried, t.both_full, t.max_tilt)
            r = t.update(bad_c, bad_b, arms)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(before, (dict(t.breaches), t.min_lift_insertion, t.carried, t.both_full, t.max_tilt))
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual(t.update(good_c, good_b, arms).score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(3)
        t = Tracker(HZ)
        for _ in range(300):
            qs = [rng.normal(size=4) for _ in range(3)]
            c = crate(rng.uniform(-1, 2, 3), qs[0], vel=rng.normal(size=3), spin=rng.normal(size=3))
            b = (bar(rng.uniform(-1, 2, 3), qs[1]), bar(rng.uniform(-1, 2, 3), qs[2]))
            arms = snap(rng.uniform(-1, 2, 3), rng.uniform(0, 1), rng.uniform(-1, 2, 3), rng.uniform(0, 1))
            r = t.update(c, b, arms)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertIsInstance(r.details, dict)


# == new episodes ====================================================================================================
class Reset(unittest.TestCase):
    def broken(self):
        f = procedure(insert=0.050)
        r, t = play(f)
        self.assertTrue(t.breach)
        return f, t

    def test_teleport_back_to_start_resets(self):
        f, t = self.broken()
        r, _ = play(Film().hold(5), tracker=t)  # the simulator puts everything back in one step
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 6)  # the jump frame started the new episode (and 5 more)
        r, _ = play(procedure(), tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        f, t = self.broken()
        j = np.array([0.02, -0.02, 0.0])
        play(Film(c=C_START + j, bl=BL_START + j, br=BR_START - j).hold(5), tracker=t)
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        f, t = self.broken()
        g = Film(**{k: f.cur[k] for k in ("c", "bl", "br", "L", "R")}, Lo=f.cur["Lo"], Ro=f.cur["Ro"])
        g.go(120, c=C_START, bl=BL_START, br=BR_START).hold(10)
        play(g, tracker=t)
        self.assertTrue(t.breach)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        f, t = self.broken()
        play(Film(c=C_START + [0.0, 0.2, 0.0]).hold(5), tracker=t)
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        f, t = self.broken()
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


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the ride model,
    the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def objects(self, c=None, bl=BL_START, br=BR_START):
        return {"crate": c or crate_at(), "fork_bar_left": bar(bl), "fork_bar_right": bar(br)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in ("crate", "fork_bar_left", "fork_bar_right"):
            self.assertIn(n, names)
        self.assertLess(names.index("fork_bar_right"), names.index("crate"))  # the ride model reads the bars first
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None],
                         ["fork_bar_left", "fork_bar_right", "crate"])

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(left_tcp=[0.255, 0.05, TOP + 0.05])  # a gripper against the crate
        self.assertFalse(self.task.success(self.objects()))
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertIsNone(self.task._RIDE.pose)
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

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):  # one clock, mirrored in y: the bars rise and fall together
            self.assertEqual(a.duration, b.duration)
            self.assertAlmostEqual(a.pos[1], -b.pos[1])
            self.assertEqual((a.pos[0], a.pos[2], a.finger), (b.pos[0], b.pos[2], b.finger))
        lift = next(w for w in left if w.label.startswith("lift"))
        self.assertAlmostEqual(lift.pos[0] - (L.BAR_X - L.BAR_HALF_LEN + L.BAR_GRIP_FROM_REAR), 0.100)
        for prev, w in zip(left, left[1:]):  # every gripper open or close takes at least 1.5 s
            if prev.finger != w.finger:
                self.assertGreaterEqual(w.duration, 1.5)
        for key, value, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                 ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = value
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_offline_specs_carry_the_ride_hooks(self):
        crate_spec = next(o for o in self.task.OBJECTS if o.name == "crate")
        self.assertTrue(callable(getattr(crate_spec, "offline_state", None)))
        centre, size, yaw = crate_spec.offline_box()
        self.assertEqual(tuple(size), L.CRATE_SIZE)

    def test_in_the_simulator_the_specs_are_left_alone(self):
        """With yam_arena's embodiment loaded (a stand-in here), rigstate installs and task.py attaches no offline
        hook to any object spec."""
        import importlib.util
        import types

        class Embodiment:
            def _task_step(self, *a, **k):
                return None

            def reset(self, *a, **k):
                return None

        name = "yam_arena.inspect_embodiment"
        saved_state, saved_mod = dict(rigstate._STATE), sys.modules.get(name)
        saved_hooks = dict(rigstate._RESET_CALLBACKS)
        fake = types.ModuleType(name)
        fake.YamArenaEmbodiment = Embodiment
        sys.modules[name] = fake
        rigstate._STATE["installed"] = False
        try:
            spec = importlib.util.spec_from_file_location("task_in_sim", HERE / "task.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertTrue(mod._IN_SIMULATOR)
            for o in mod.OBJECTS:
                self.assertIsNone(getattr(o, "offline_state", None), o.name)
                self.assertIsNone(getattr(o, "offline_box", None), o.name)
        finally:
            rigstate._STATE.clear()
            rigstate._STATE.update(saved_state)
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved_hooks)
            if saved_mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_mod
            sys.modules.pop("task_in_sim", None)

    def test_ride_model(self):
        """The offline ride model: bars under both pocket lines carry the crate, one bar tilts it, none: it rests."""
        from offline_ride import CrateRide
        ride = CrateRide(("a", "b"))

        def step(bl, br, c=None):
            ride._bar_hook("a")(bar(bl))
            ride._bar_hook("b")(bar(br))
            return ride.crate_state(c or crate_at())

        s = step(BL_START, BR_START)
        self.assertAlmostEqual(lowest(s), L.FLOOR_TOP, places=9)
        bl, br = BL_START + 0.1 * X, BR_START + 0.1 * X
        for k in range(30):  # both bars rise 30 mm: 12 to reach the underside, then 18 with the crate
            bl, br = bl + 0.001 * Z, br + 0.001 * Z
            s = step(bl, br)
        self.assertEqual(ride.mode, "carried")
        self.assertAlmostEqual(lowest(s), L.FLOOR_TOP + 0.018, places=6)
        for k in range(20):
            bl, br = bl + 0.002 * X, br + 0.002 * X
            s = step(bl, br)
        self.assertAlmostEqual(float(s.pos[0]), L.CRATE_X + 0.040, places=6)
        self.assertEqual(ride.warnings, [])
        ride.reset()
        s = step(BL_START, BR_START)
        bl = BL_START + 0.1 * X
        for k in range(40):  # one bar rises 40 mm
            bl = bl + 0.001 * Z
            s = step(bl, BR_START + 0.1 * X)
        self.assertEqual(ride.mode, "pivot")
        self.assertGreater(tilt_deg(s), 5.0)
        ride.reset()
        step(BL_START, BR_START)
        step(BL_START + [0.1, 0, 0.03], BR_START)  # a bar driven into the crate's side
        self.assertTrue(ride.warnings)

    def test_offline_checker_predicts_one_and_the_ride_stays_clean(self):
        """tools/offline/check_task.py --strict --one-arm on this folder, and its playback's ride model: no bar ever
        driven into the crate. Run in-process with sys.modules and sys.path restored afterwards (the checker reloads
        the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.0, "right_only": 0.0})
            task = check_task.load_task(HERE / "task.py", HERE)
            plans = check_task.get_plans(task, check_task.Report())
            solids = [check_task.G.solid_from_spec(o, task.table_top) for o in task.objects]
            out = check_task.Playback(task, plans, solids, hz=HZ, hold_s=3.0).run()
            self.assertEqual(out["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(task.module._RIDE.warnings, [])
            self.assertEqual(task.module._TRACKER.breaches, {})
            self.assertGreater(task.module._TRACKER.min_zone_gap, 0.015)
            check_task._forget_task_modules()
        finally:
            for name in [n for n in sys.modules if n not in saved_modules]:
                del sys.modules[name]
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
