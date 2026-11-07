"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
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
    Tracker, _cylinder_gap, across_line, against_rail, blocks_end, down_axis, end_fit, finger_axis, line_steel,
    part_flat, piece_flat_low, piece_on_chuck, rail_gaps, released, rotation, too_tall, valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
OPEN = 1.0
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
RAIL = L.RAIL_FACE_Y
PY = RAIL - L.PART_HALF[1] - 0.001  # the part 1 mm short of the rail
PX = L.LINES_X[1]  # centred over the middle brass line
BY = RAIL - 0.014 - L.LOW_SIZE[1] / 2  # a low piece covering the end face's 18 mm farthest from the rail


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(name, center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ (np.array(L.SIZES[name], float) / 2)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def box(name, x, y, yaw=0.0, base=None, lift=0.0, extra=None, vel=(0, 0, 0)):
    """A box lying on its broad face (body z up), turned ``yaw`` degrees, its bottom at ``base`` (default: the
    chuck face) plus ``lift``; then turned by the quaternion ``extra`` about its centre."""
    q = about("z", yaw)
    if extra is not None:
        q = qmul(extra, q)
    r = rotation(q)
    zext = float(np.abs(r[2, :]) @ (np.array(L.SIZES[name], float) / 2))
    z0 = (L.FACE_Z if base is None else base) + lift
    return state(name, (x, y, z0 + zext), q, vel)


def on_side(name, x, y, yaw=0.0, base=None):
    """A box resting on one of its long side faces (body y up)."""
    q = qmul(about("z", yaw), about("x", 90))
    zext = L.SIZES[name][1] / 2
    return state(name, (x, y, (L.FACE_Z if base is None else base) + zext), q)


def on_end(name, x, y, base=None):
    """A box standing on one of its end faces (body x up)."""
    q = about("y", 90)
    return state(name, (x, y, (L.FACE_Z if base is None else base) + L.SIZES[name][0] / 2), q)


def start():
    """Everything where it starts."""
    out = {"part": box("part", L.PART_X0, L.PART_Y0, 90.0, base=L.TOP)}
    for name, _, x, y in L.PIECES:
        out[name] = box(name, x, y, base=L.TOP)
    return out


def part_at(x=PX, y=PY, yaw=0.0, **kw):
    return box("part", x, y, yaw, **kw)


def near_x(gap=0.001, px=PX):
    return px - L.PART_HALF[0] - gap - L.LOW_SIZE[0] / 2


def far_x(gap=0.001, px=PX):
    return px + L.PART_HALF[0] + gap + L.LOW_SIZE[0] / 2


def done(px=PX, py=PY, near=0.001, far=0.001):
    """The finished setup: part against the rail across the middle line, a low piece against each end, the tall
    pieces where they started."""
    s = start()
    s["part"] = part_at(px, py)
    s["block_low_2"] = box("block_low_2", near_x(near, px), BY)
    s["block_low_1"] = box("block_low_1", far_x(far, px), BY)
    return s


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def feed(t, states, arms=None, n=1):
    r = None
    for _ in range(n):
        r = t.update(states, snap() if arms is None else arms)
    return r


def score_of(states, arms=None, n=12):
    t = Tracker()
    return feed(t, states, arms, n)


class Geometry(unittest.TestCase):
    def test_chuck_strips_tile_the_face(self):
        spans = L.pole_spans()
        self.assertEqual(len(spans), len(L.LINES_X) + 1)
        edges = sorted([a for a, b in spans] + [b for a, b in spans]
                       + [x - L.LINE_W / 2 for x in L.LINES_X] + [x + L.LINE_W / 2 for x in L.LINES_X])
        self.assertAlmostEqual(edges[0], L.CHUCK_X0)
        self.assertAlmostEqual(edges[-1], L.CHUCK_X1)
        # each pole is wider than the part, so the part can sit on a single pole
        self.assertTrue(all(b - a > L.PART_SIZE[0] for a, b in spans[1:-1]))
        self.assertAlmostEqual(L.LINES_X[1] - L.LINES_X[0], 0.060)

    def test_brass_lines_contrast_with_the_poles(self):
        """Brass and steel of equal brightness blur together in a small, chroma-subsampled frame: the brass lines
        are well darker than the pole steel (relative luminance of the linearised sRGB)."""
        def lum(c):
            lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
            return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        self.assertLess(lum(L.BRASS), 0.6 * lum(L.POLE_STEEL))

    def test_piece_heights(self):
        # low pieces are lower than the part whichever broad or side face they lie on; tall ones are higher anyhow
        self.assertLess(max(sorted(L.LOW_SIZE)[:2]), L.PART_SIZE[2])
        self.assertGreater(min(L.TALL_SIZE), L.PART_SIZE[2] + 0.001)
        self.assertLess(max(L.LOW_MASS, L.TALL_MASS, L.PART_MASS), 0.3)

    def test_start_layout(self):
        s = start()
        self.assertTrue(Tracker.at_start(s))
        for n in L.PIECE_NAMES:
            self.assertFalse(piece_on_chuck(s[n], n))
            self.assertFalse(too_tall(s[n], n))
        r = score_of(s)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_down_axis_is_folded(self):
        self.assertEqual(down_axis(part_at())[0], 2)
        upside = part_at(extra=about("x", 180))
        self.assertEqual(down_axis(upside)[0], 2)
        self.assertLess(down_axis(upside)[1], 1e-6)
        self.assertTrue(part_flat(upside))
        self.assertEqual(down_axis(on_side("part", PX, PY))[0], 1)
        self.assertEqual(down_axis(on_end("part", PX, PY))[0], 0)

    def test_rail_gaps(self):
        g = rail_gaps(part_at(y=RAIL - L.PART_HALF[1] - 0.002))
        self.assertAlmostEqual(g[0], 0.002, places=6)
        self.assertAlmostEqual(g[1], 0.002, places=6)
        turned = part_at(yaw=3.0)  # turned: the far corner nearer the rail than the near one
        a, b = rail_gaps(turned)
        self.assertGreater(a, b)

    def test_against_rail_tolerances(self):
        def at_gap(gap, **kw):
            return against_rail(part_at(y=RAIL - L.PART_HALF[1] - gap, **kw))
        self.assertTrue(at_gap(0.0))
        self.assertTrue(at_gap(0.0029))
        self.assertFalse(at_gap(0.0032))
        self.assertFalse(at_gap(-0.0025))  # past the rail's face: on the rail, not against it
        self.assertTrue(against_rail(part_at(y=RAIL - L.PART_HALF[1] - 0.0005, yaw=4.0)))
        self.assertFalse(against_rail(part_at(y=RAIL - L.PART_HALF[1] - 0.0005, yaw=6.0)))
        self.assertTrue(against_rail(part_at(lift=0.0015)))
        self.assertFalse(against_rail(part_at(lift=0.0025)))
        self.assertTrue(against_rail(part_at(extra=about("x", 2.0))))
        self.assertFalse(against_rail(part_at(extra=about("x", 4.0))))
        self.assertFalse(against_rail(part_at(x=L.CHUCK_X1 - 0.010)))  # hanging off the chuck's end
        self.assertFalse(against_rail(part_at(base=L.TOP)))  # on the table beside the chuck

    def test_crosswise_is_not_lengthwise(self):
        p = box("part", PX, RAIL - L.PART_HALF[0] - 0.001, 90.0)  # its 30 mm end toward the rail
        self.assertTrue(part_flat(p))
        self.assertFalse(against_rail(p))

    def test_on_its_side_is_not_flat(self):
        p = on_side("part", PX, RAIL - L.PART_SIZE[2] / 2 - 0.001)  # 30 mm tall, its 40 x 20 face down
        self.assertFalse(part_flat(p))
        self.assertFalse(against_rail(p))

    def test_across_line(self):
        for line in L.LINES_X:
            self.assertTrue(across_line(part_at(x=line)))
            self.assertTrue(across_line(part_at(x=line + 0.0089)))
            self.assertTrue(across_line(part_at(x=line - 0.0089)))
            self.assertFalse(across_line(part_at(x=line + 0.0095)))
        self.assertFalse(across_line(part_at(x=0.240)))  # wholly on a pole
        x, a, b = line_steel(part_at(x=PX + 0.004))
        self.assertEqual(x, PX)
        self.assertAlmostEqual(a, 0.013, places=6)  # moved 4 mm toward the far end: 13 mm on the near side
        self.assertAlmostEqual(b, 0.021, places=6)

    def test_end_fit(self):
        p = part_at()
        f = end_fit(p, box("block_low_1", far_x(0.002), BY), "block_low_1", 1)
        self.assertAlmostEqual(f["gap"], 0.002, places=6)
        self.assertAlmostEqual(f["angle"], 0.0, places=4)
        self.assertAlmostEqual(f["overlap"], 0.017, places=6)  # the part 1 mm short of the rail, the piece 14 mm
        f = end_fit(p, box("block_low_2", near_x(0.001), BY), "block_low_2", -1)
        self.assertAlmostEqual(f["gap"], 0.001, places=6)
        # a piece across the part's end, long side along y, covers the whole 30 mm depth
        f = end_fit(p, box("block_low_2", PX + L.PART_HALF[0] + 0.001 + 0.009, PY, 90.0), "block_low_2", 1)
        self.assertAlmostEqual(f["gap"], 0.001, places=6)
        self.assertAlmostEqual(f["overlap"], 0.030, places=6)

    def test_finger_zone_and_release(self):
        p = part_at()
        top = p.pos[2] + L.PART_HALF[2]
        over = snap(left_tcp=(PX, PY, top + TIPS + 0.0))
        self.assertLess(zone_gap(over["left"], p, "part"), 0)
        self.assertFalse(released(over, p, "part"))
        self.assertTrue(released(snap(), p, "part"))
        self.assertTrue(released(None, p, "part"))  # unreadable: taken as let go


class Tiers(unittest.TestCase):
    def test_full_setup_scores_one(self):
        r = score_of(done())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual({r.details["near_blocker"], r.details["far_blocker"]}, {"block_low_1", "block_low_2"})

    def test_against_rail_only_quarter(self):
        s = start()
        s["part"] = part_at(x=0.240)  # on one pole
        self.assertEqual(score_of(s).score, 0.25)

    def test_rail_and_line_half(self):
        s = start()
        s["part"] = part_at()
        self.assertEqual(score_of(s).score, 0.5)

    def test_rail_and_one_end_half(self):
        s = done(px=0.240)
        s["block_low_1"] = box("block_low_1", 0.10, -0.15, base=L.TOP)
        self.assertEqual(score_of(s).score, 0.5)

    def test_rail_line_one_end_three_quarters(self):
        for gone in ("block_low_1", "block_low_2"):
            s = done()
            s[gone] = box(gone, 0.10, -0.15, base=L.TOP)
            self.assertEqual(score_of(s).score, 0.75)

    def test_rail_both_ends_no_line_three_quarters(self):
        r = score_of(done(px=0.240))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["across_line"])

    def test_every_tier_is_reachable(self):
        seen = set()
        for px in (PX, 0.240):
            for keep in ((), ("block_low_1",), ("block_low_2",), ("block_low_1", "block_low_2")):
                s = done(px=px)
                for n in ("block_low_1", "block_low_2"):
                    if n not in keep:
                        s[n] = box(n, 0.10 if n.endswith("1") else 0.12, -0.15, base=L.TOP)
                seen.add(score_of(s).score)
        seen.add(score_of(start()).score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_any_line_and_either_piece(self):
        for px in L.LINES_X[::2]:
            s = done(px=px)
            s["block_low_1"], s["block_low_2"] = (box("block_low_1", near_x(0.001, px), BY),
                                                  box("block_low_2", far_x(0.001, px), BY))
            self.assertEqual(score_of(s).score, 1.0)

    def test_nothing_counts_without_the_rail(self):
        for py in (PY - 0.005, PY - 0.030):  # 6 mm and 31 mm short of the rail, still across the line
            s = done(py=py)
            s["block_low_1"] = box("block_low_1", far_x(), py - 0.006)
            s["block_low_2"] = box("block_low_2", near_x(), py - 0.006)
            r = score_of(s)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["against_rail"])

    def test_end_tolerances(self):
        self.assertEqual(score_of(done(far=0.0029)).score, 1.0)
        self.assertEqual(score_of(done(far=0.0032)).score, 0.75)
        self.assertEqual(score_of(done(near=0.0)).score, 1.0)

    def test_end_overlap_and_angle(self):
        s = done()
        s["block_low_1"] = box("block_low_1", far_x(), RAIL - L.PART_SIZE[1] - 0.001 - 0.009 + 0.011)  # 11 mm
        self.assertEqual(score_of(s).score, 1.0)
        s["block_low_1"] = box("block_low_1", far_x(), RAIL - L.PART_SIZE[1] - 0.001 - 0.009 + 0.008)  # 8 mm
        self.assertEqual(score_of(s).score, 0.75)
        s = done()
        x = far_x(0.0) + 0.009 * math.sin(math.radians(8.0)) + 0.0005
        s["block_low_1"] = box("block_low_1", x, BY, 8.0)
        self.assertEqual(score_of(s).score, 1.0)
        s["block_low_1"] = box("block_low_1", x + 0.001, BY, 12.0)
        self.assertEqual(score_of(s).score, 0.75)

    def test_low_piece_on_its_side_still_blocks(self):
        s = done()
        s["block_low_1"] = on_side("block_low_1", far_x(), RAIL - 0.014 - L.LOW_SIZE[2] / 2)  # 18 mm tall
        self.assertTrue(piece_flat_low(s["block_low_1"], "block_low_1"))
        self.assertEqual(score_of(s).score, 1.0)

    def test_success_needs_stillness(self):
        t = Tracker()
        s = done()
        self.assertFalse(feed(t, s, n=10).success)  # 0.5 s of history not yet complete
        self.assertTrue(feed(t, s).success)
        moving = done()
        moving["block_low_1"] = box("block_low_1", far_x(), BY, vel=(0.08, 0, 0))
        self.assertFalse(feed(t, moving).success)
        t = Tracker()
        for k in range(12):  # creeping 0.3 mm a step: 3 mm in 0.5 s
            feed(t, done(far=0.0005 + 0.0002 * k))
        self.assertEqual(t.last.score, 1.0)
        self.assertFalse(t.last.success)

    def test_score_is_idempotent(self):
        t = Tracker()
        s = done()
        r1 = feed(t, s, n=12)
        r2 = t.result(s, snap())
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.success, r2.success)


class ZeroConditions(unittest.TestCase):
    def test_tall_piece_anywhere_on_the_chuck(self):
        s = done()
        s["block_tall_1"] = box("block_tall_1", 0.330, -0.050)
        r = score_of(s)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["too_tall_on_chuck"], ["block_tall_1"])

    def test_tall_piece_as_blocking(self):
        s = done()
        s["block_low_1"] = box("block_low_1", 0.10, -0.15, base=L.TOP)
        s["block_tall_2"] = box("block_tall_2", far_x(), RAIL - 0.014 - L.TALL_SIZE[1] / 2)
        self.assertEqual(score_of(s).score, 0.0)

    def test_tall_piece_on_its_side_is_still_taller(self):
        s = done()
        s["block_tall_1"] = on_side("block_tall_1", 0.330, -0.050)  # 24 mm tall
        self.assertEqual(score_of(s).score, 0.0)

    def test_tall_piece_half_over_the_chuck_edge(self):
        s = done()
        s["block_tall_1"] = box("block_tall_1", 0.200, L.CHUCK_Y0 - 0.005, 90.0)  # its outline 15 mm over the face
        self.assertTrue(too_tall(s["block_tall_1"], "block_tall_1"))
        self.assertEqual(score_of(s).score, 0.0)

    def test_tall_piece_on_the_rail(self):
        s = done()
        s["block_tall_1"] = box("block_tall_1", 0.350, L.RAIL_Y, base=L.FACE_Z + L.RAIL_H)
        self.assertEqual(score_of(s).score, 0.0)

    def test_tall_piece_on_the_table_beside_the_chuck_is_fine(self):
        s = done()
        s["block_tall_1"] = box("block_tall_1", 0.250, L.CHUCK_Y0 - 0.0125, base=L.TOP)
        self.assertFalse(too_tall(s["block_tall_1"], "block_tall_1"))
        self.assertEqual(score_of(s).score, 1.0)

    def test_tall_piece_held_high_over_the_chuck_is_not_on_it(self):
        s = done()
        s["block_tall_1"] = box("block_tall_1", 0.330, -0.050, lift=0.040)
        self.assertFalse(piece_on_chuck(s["block_tall_1"], "block_tall_1"))
        self.assertEqual(score_of(s).score, 1.0)

    def test_low_piece_standing_on_end(self):
        s = done()
        s["block_low_1"] = on_end("block_low_1", far_x(0.001) - L.LOW_SIZE[0] / 2 + L.LOW_SIZE[2] / 2, BY)
        self.assertEqual(score_of(s).score, 0.0)

    def test_low_piece_on_top_of_the_part(self):
        s = done()
        s["block_low_1"] = box("block_low_1", PX, PY, base=L.FACE_Z + L.PART_SIZE[2])
        self.assertEqual(score_of(s).score, 0.0)

    def test_part_not_on_the_chuck(self):
        s = done()
        s["part"] = part_at(base=L.TOP, y=L.CHUCK_Y0 - 0.030)
        self.assertEqual(score_of(s).score, 0.0)

    def test_part_tilted_or_raised(self):
        s = done()
        s["part"] = part_at(extra=about("y", 5.0))
        self.assertEqual(score_of(s).score, 0.0)
        s["part"] = part_at(lift=0.004)
        self.assertEqual(score_of(s).score, 0.0)

    def test_part_still_held(self):
        s = done()
        top = s["part"].pos[2] + L.PART_HALF[2]
        held = snap(left_tcp=(PX, PY, top - 0.007), left_open=0.33)
        r = score_of(s, held)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["part_released"])

    def test_piece_still_held(self):
        s = done()
        b = s["block_low_1"]
        held = snap(right_tcp=(b.pos[0], b.pos[1], b.pos[2] + L.LOW_SIZE[2] / 2 + 0.0045), right_open=0.1)
        r = score_of(s, held)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_hovering_gripper_counts_as_let_go(self):
        """The zone is flat at the finger tips: an open gripper hovering above the part has let go of it once its
        tips are more than 2 mm above it, however wide open; tips resting on it (or within 2 mm) still hold it."""
        s = done()
        top = s["part"].pos[2] + L.PART_HALF[2]
        for up in (0.030, 0.010, 0.0025):
            hover = snap(left_tcp=(PX, PY, top + TIPS + up))
            self.assertAlmostEqual(zone_gap(hover["left"], s["part"], "part"), up, places=4)
            self.assertEqual(score_of(s, hover).score, 1.0, up)
        for up in (0.0015, 0.0, -0.001):
            touching = snap(left_tcp=(PX, PY, top + TIPS + up))
            self.assertEqual(score_of(s, touching).score, 0.0, up)
        # beside the part with the tips at its mid-height: let go only with the round zone 2 mm or more clear of it
        zone_r = math.hypot(L.FULL_GAP / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH)
        mid = s["part"].pos[2]
        side = PY - L.PART_HALF[1]  # the part's side away from the rail
        clear = snap(left_tcp=(PX, side - zone_r - 0.003, mid + TIPS))
        self.assertAlmostEqual(zone_gap(clear["left"], s["part"], "part"), 0.003, places=4)
        self.assertEqual(score_of(s, clear).score, 1.0)
        near = snap(left_tcp=(PX, side - zone_r - 0.001, mid + TIPS))
        self.assertEqual(score_of(s, near).score, 0.0)
        # a gripper closed on a piece narrower than it really is (zone 18.5 mm) at the far end does not reach the
        # part: only that end is lost (test_piece_still_held)
        b = s["block_low_1"]
        holding = snap(right_tcp=(b.pos[0], b.pos[1], b.pos[2] + L.LOW_SIZE[2] / 2 + 0.0045), right_open=0.1)
        self.assertGreaterEqual(zone_gap(holding["right"], s["part"], "part"), L.HANDS_OFF)

    def test_piece_still_held_against_an_end_holds_the_part_too(self):
        """Closed on a real 18 mm piece (opening 18/94) the round zone (21.7 mm) reaches past the piece's end into
        the part 1 mm beyond it, so the part counts as held as well: 0 (the text says so)."""
        s = done()
        b = s["block_low_1"]
        holding = snap(right_tcp=(b.pos[0], b.pos[1], b.pos[2] + L.LOW_SIZE[2] / 2 + 0.0045),
                       right_open=L.LOW_SIZE[1] / L.FULL_GAP)
        self.assertLess(zone_gap(holding["right"], s["part"], "part"), L.HANDS_OFF)
        r = score_of(s, holding)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["part_released"])

    def test_zone_gap_matches_a_dense_search(self):
        """zone_gap against a brute-force search over a dense grid of the box's points, for random grippers round
        the part and a piece: never more than 1 mm above it (an upper bound: both search points on the box)."""
        rng = np.random.default_rng(7)
        s = done()
        g = np.linspace(-1.0, 1.0, 21)
        unit = np.array([(a, b, c) for a in g for b in g for c in g])
        for name in ("part", "block_low_1"):
            st = s[name]
            pts = np.asarray(st.pos) + (unit * (np.array(L.SIZES[name]) / 2)) @ rotation(st.quat_xyzw).T
            for _ in range(60):
                tcp = np.asarray(st.pos) + rng.uniform(-0.09, 0.09, 3) + np.array([0, 0, 0.03])
                tilt = rng.uniform(-0.4, 0.4, 2)
                axis = np.array([tilt[0], tilt[1], -1.0])
                axis = axis / np.linalg.norm(axis)
                opening = float(rng.uniform(0.0, 1.0))
                rigstate.offline_feed({side: {"tcp": tcp + off, "tips": tcp + off + TIPS * axis, "opening": opening}
                                       for side, off in (("left", 0.0), ("right", np.array([0, -0.6, 0])))})
                arm = rigstate.arms()["left"]
                tips, u = finger_axis(arm)
                brute = float(np.min(_cylinder_gap(tips, u, zone_radius(arm), pts)))
                got = zone_gap(arm, st, name)
                if brute > 0:
                    self.assertLessEqual(got, brute + 0.001, (name, tcp, opening))
                    self.assertGreaterEqual(got, -1e-9)
                else:
                    self.assertLess(got, 0.001, (name, tcp, opening))


class Exploits(unittest.TestCase):
    def test_both_low_pieces_at_one_end(self):
        s = done()
        s["block_low_1"] = box("block_low_1", near_x(0.001) - L.LOW_SIZE[0] - 0.001, BY)
        self.assertEqual(score_of(s).score, 0.75)

    def test_piece_leaning_against_the_end(self):
        s = done()
        lean = box("block_low_1", far_x(0.002), BY, extra=about("y", -7.0), lift=0.0005)  # propped 7 deg
        s["block_low_1"] = lean
        self.assertFalse(piece_flat_low(lean, "block_low_1"))
        self.assertFalse(too_tall(lean, "block_low_1"))
        self.assertEqual(score_of(s).score, 0.75)

    def test_piece_rolled_on_its_long_edge(self):
        s = done()
        y = RAIL - 0.014 - L.LOW_SIZE[1] / 2
        s["block_low_1"] = box("block_low_1", far_x(), y, extra=about("x", 4.0))  # 4 deg: still lying flat
        self.assertTrue(piece_flat_low(s["block_low_1"], "block_low_1"))
        self.assertEqual(score_of(s).score, 1.0)
        s["block_low_1"] = box("block_low_1", far_x(), y, extra=about("x", 7.0))  # 7 deg: top 17 mm, not flat
        self.assertFalse(piece_flat_low(s["block_low_1"], "block_low_1"))
        self.assertFalse(too_tall(s["block_low_1"], "block_low_1"))
        self.assertEqual(score_of(s).score, 0.75)

    def test_piece_on_its_side_tipped_up_is_not_lower_than_the_part(self):
        # on its side (18 mm) and tipped 4 deg along its length: its top 20.7 mm up, not lower than the 20 mm part
        q = qmul(about("y", 4.0), about("x", 90))
        zext = float(np.abs(rotation(q)[2, :]) @ (np.array(L.LOW_SIZE) / 2))
        x = far_x(0.0) + 0.0075 * math.sin(math.radians(4.0)) + 0.0015
        s = done()
        s["block_low_1"] = state("block_low_1", (x, RAIL - 0.014 - L.LOW_SIZE[2] / 2, L.FACE_Z + zext), q)
        top = s["block_low_1"].high[2] - L.FACE_Z
        self.assertGreater(top, L.LOW_MAX)
        self.assertLessEqual(top, L.TOO_TALL)
        self.assertFalse(piece_flat_low(s["block_low_1"], "block_low_1"))
        self.assertEqual(score_of(s).score, 0.75)

    def test_piece_level_with_the_face_but_off_the_chuck_does_not_block(self):
        s = start()
        px = L.CHUCK_X1 - L.PART_HALF[0] - 0.001  # the part's far end 1 mm short of the chuck's end
        s["part"] = part_at(x=px)
        self.assertEqual(score_of(s).score, 0.25)
        # a tall piece on its side on the table past the chuck's end, a low piece lying on it level with the face
        s["block_tall_1"] = on_side("block_tall_1", L.CHUCK_X1 + 0.001 + L.TALL_SIZE[0] / 2, BY, base=L.TOP)
        s["block_low_1"] = box("block_low_1", far_x(0.002, px), BY, base=L.TOP + L.TALL_SIZE[1])
        self.assertFalse(piece_on_chuck(s["block_tall_1"], "block_tall_1"))
        self.assertLess(abs(s["block_low_1"].low[2] - L.FACE_Z), L.PIECE_FACE_TOL)
        self.assertEqual(score_of(s).score, 0.25)

    def test_piece_propped_steeply_stands_above_the_part(self):
        s = done()
        prop = box("block_low_1", far_x(0.004), BY, extra=about("y", -15.0), lift=0.002)  # its top 26 mm up
        s["block_low_1"] = prop
        self.assertTrue(too_tall(prop, "block_low_1"))
        self.assertEqual(score_of(s).score, 0.0)

    def test_piece_beside_the_part_not_at_its_end(self):
        s = done()
        s["block_low_1"] = box("block_low_1", PX, PY - L.PART_HALF[1] - 0.001 - L.LOW_SIZE[1] / 2)
        self.assertEqual(score_of(s).score, 0.75)

    def test_part_crosswise_with_pieces(self):
        s = done()
        s["part"] = box("part", PX, RAIL - L.PART_HALF[0] - 0.001, 90.0)
        self.assertEqual(score_of(s).score, 0.0)

    def test_tall_piece_stood_beside_but_off_the_chuck_does_not_block(self):
        s = done()
        s["block_low_1"] = box("block_low_1", 0.10, -0.15, base=L.TOP)
        s["block_tall_2"] = box("block_tall_2", L.CHUCK_X1 + 0.013, BY, 90.0, base=L.TOP)
        r = score_of(s)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["far_end_blocked"])

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker()
        for _ in range(300):
            s = {}
            for n in L.LOOSE:
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
                s[n] = state(n, rng.uniform([0.0, -0.4, L.TOP - 0.05], [0.5, 0.4, L.TOP + 0.2]), q,
                             rng.normal(size=3) * 0.1)
            r = feed(t, s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker()
        feed(t, done(), n=12)
        paths = {n: list(t.paths[n]) for n in L.LOOSE}
        for bad in ("part", "block_low_1", "block_tall_2"):
            for field_, value in (("pos", [math.nan, 0, 0]), ("quat_xyzw", [0, 0, 0, 0]),
                                  ("lin_vel", [math.inf, 0, 0])):
                s = done()
                setattr(s[bad], field_, np.array(value, float))
                self.assertFalse(valid(s[bad]))
                r = t.update(s, snap())
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertEqual({n: list(t.paths[n]) for n in L.LOOSE}, paths)
        self.assertEqual(t.bad_steps, 9)
        self.assertTrue(feed(t, done()).success)  # carries on once the states are good again

    def test_missing_object_scores_zero(self):
        s = done()
        del s["block_low_1"]
        t = Tracker()
        r = feed(t, s)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 1)

    def test_unreadable_arms_are_taken_as_let_go_and_counted(self):
        t = Tracker()
        for _ in range(12):
            r = t.update(done(), None)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(t.arms_unavailable, 12)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker()
        bad = snap()
        bad = {"left": rigstate.Arm(np.array([math.nan, 0, 0]), np.zeros(3), 1.0), "right": bad["right"]}
        r = feed(t, done(), bad, n=12)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)

    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        feed(t, done(), n=20)
        self.assertEqual(t.steps, 20)
        feed(t, start())
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.paths["part"]), 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        feed(t, done(), n=5)
        a, b = np.array([PX, PY]), np.array([L.PART_X0, L.PART_Y0])
        for k in range(1, 41):
            s = start()
            p = a + (b - a) * k / 40
            s["part"] = box("part", p[0], p[1], 90.0, base=L.TOP)
            feed(t, s)
        self.assertEqual(t.steps, 45)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        feed(t, done(), n=5)
        s = done()
        s["part"] = part_at(x=0.200, y=-0.040)
        feed(t, s)
        self.assertEqual(t.steps, 6)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = Tracker()
        feed(t, done(), n=12)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.steps, 0)
            self.assertEqual(len(t.paths["part"]), 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)

    def test_offline_feed_reaches_blocks_end(self):
        s = done()
        self.assertTrue(blocks_end(s["part"], s["block_low_1"], "block_low_1", 1, snap()))
        self.assertFalse(blocks_end(s["part"], s["block_low_1"], "block_low_1", -1, snap()))


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o.name for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(loose), sorted(L.LOOSE))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertAlmostEqual(self.task.PART.yaw, L.PART_YAW0)
        for o in self.task.OBJECTS:
            if o.mass is not None:
                self.assertEqual(tuple(o.size), tuple(L.SIZES[o.name]))
        self.assertTrue(Tracker.at_start(start()))

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        for _ in range(12):
            self.task.success(done())
        self.assertEqual(self.task.score(done()), 1.0)
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(start()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertEqual(self.task.score(start()), 0.0)

    def test_hooks_ignore_fixed_objects(self):
        rigstate.offline_reset()
        snap()
        s = done()
        s["chuck_rail"] = SimpleNamespace(pos=np.array([math.nan] * 3))  # fixed objects are not read
        self.assertFalse(self.task.success(s))
        self.assertEqual(self.task.score(s), 1.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(start()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_places_score_one(self):
        """Where the demo lets go of the part and the two low pieces scores 1.0 (no sag), and still does with the
        sag the compliance model predicts on the GPU (part (-1.7, +1.9) mm, pieces (-0.8, -0.9) mm at their lower
        edges)."""
        t = self.task
        for sag_part, sag_piece in (((0, 0), (0, 0)), ((-0.0017, 0.0019), (-0.0008, -0.0009))):
            s = start()
            s["part"] = part_at(t.PART_X + sag_part[0], t.PART_Y + sag_part[1])
            s["block_low_1"] = box("block_low_1", t.FAR_X + sag_piece[0], t.PIECE_Y + sag_piece[1])
            s["block_low_2"] = box("block_low_2", t.NEAR_X + sag_piece[0], t.PIECE_Y + sag_piece[1])
            r = score_of(s)
            self.assertEqual(r.score, 1.0, r.details)
            self.assertGreater(r.details["near_gap_mm"], 0.5)  # no lower edge lands on the part's top
            self.assertGreater(min(r.details["rail_gap_near_mm"], r.details["rail_gap_far_mm"]), 0.5)

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        # every close and open takes 1.5 s, and every last 2 cm down onto a surface too
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger and prev.pos == w.pos:
                    self.assertGreaterEqual(w.duration, 1.5)
                if prev.pos[2] > w.pos[2] and w.pos[2] < L.FACE_Z + 0.025 and prev.pos[:2] == w.pos[:2]:
                    if prev.pos[2] - w.pos[2] <= 0.0205:
                        self.assertGreaterEqual(w.duration, 1.5)
        # the right arm does not come over the chuck until the left arm has let go and lifted
        t_left = 0.0
        for w in plan["left"]:
            t_left += w.duration
            if w.label == "straight up":
                break
        t_right = 0.0
        for w in plan["right"]:
            if w.label.startswith("carry over the far"):
                break
            t_right += w.duration
        self.assertGreaterEqual(t_right, t_left)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
