"""Offline tests for scoring.py, standin.py and the hooks in task.py: synthetic die, block and part states and
synthetic arm snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
GAP = 0.0939  # fully open finger gap: rigstate opening = gap / GAP
OPEN = 1.0
POST_SHUT = L.POST_SIZE[0] / GAP  # openings when closed on the post, the handle arm, the part
HANDLE_SHUT = L.ARM_SIZE[0] / GAP
PART_SHUT = L.PART_SIZE[1] / GAP
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
POST_TCP = np.array([L.POST_X0, L.POST_Y, L.TOP + 0.098])  # the left tool point on the post with the die shut
PARK_L = np.array([0.200, 0.220, L.TOP + 0.150])
B_START = np.array(L.BLOCK_ORIGIN, dtype=float)
HANDLE = np.array([0.0, -L.GRIP_FROM_BAR, 0.018])  # the right tool point on the handle arm, from the block's origin
B_SEATED = np.array([0.305, -0.080, L.TOP + L.BED_TOP_Z])  # bar x 0.256-0.354, y -0.095..-0.065, on the bed
J_SEATED = L.MOVING_FACE_X0 - (B_SEATED[0] - L.BAR_SIZE[0] / 2)  # 0.055: the platen resting on the bar's near end
P_START = np.array([L.PART_X, L.PART_Y, L.TOP + L.PART_Z])
PART_HELD = np.array([0.0, 0.0, 0.012])  # a held part's centre is 12 mm below the tool point
P_TRAY = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + L.TRAY_FLOOR_Z + L.PART_SIZE[2] / 2])
UP = L.TOP + 0.120


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def about_z_handle(deg):
    """Where the right tool point is on the handle arm, from the block's origin, with the block turned ``deg``
    degrees about z."""
    return S.rotation(about("z", deg)) @ HANDLE


def die(j=0.0, jv=0.0):
    """The die as mechkit reports it (fixed: only the joint moves)."""
    lo = np.array([L.BED_X0, L.BED_Y0, L.TOP])
    hi = np.array([L.BED_X1, L.BED_Y1, L.TOP + L.POST_Z1])
    return SimpleNamespace(pos=(lo + hi) / 2, quat_xyzw=Q0, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=lo, high=hi,
                           origin=np.array(L.DIE_ORIGIN), joints={"platen": float(j)},
                           joint_vel={"platen": float(jv)}, links={}, source="sim")


def block(origin=B_START, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """The block as mechkit reports it: base frame origin and orientation, velocities, bounding box."""
    st = SimpleNamespace(origin=np.array(origin, float), quat_xyzw=np.array(quat, float),
                         lin_vel=np.array(vel, float), ang_vel=np.array(spin, float), source="sim")
    c = S.block_corners(st)
    st.low, st.high = c.min(axis=0), c.max(axis=0)
    st.pos = (st.low + st.high) / 2
    return st


def part(center=P_START, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    c, half = np.array(center, float), np.array(L.PART_SIZE, float) / 2
    ext = np.abs(S.rotation(quat)) @ half
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


VEC = ("b", "p", "L", "R")
KEYS = ("j", "b", "bq", "p", "pq", "L", "Lo", "R", "Ro")


class Film:
    """A trial as a list of frames: the die's opening, the block's and the part's poses, both tool points and
    openings. ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame
    motion."""

    def __init__(self, **start):
        self.cur = {"j": 0.0, "b": B_START, "bq": Q0, "p": P_START, "pq": Q0, "L": READY_L, "Lo": OPEN,
                    "R": READY_R, "Ro": OPEN}
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
                if k in ("bq", "pq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vb = vp = np.zeros(3)
            jv = 0.0
        else:
            vb, vp = (fr["b"] - prev["b"]) / DT, (fr["p"] - prev["p"]) / DT
            jv = (float(fr["j"]) - float(prev["j"])) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(die(fr["j"], jv), block(fr["b"], fr["bq"], vb), part(fr["p"], fr["pq"], vp), a)
        prev = fr
    return r, t


# -- the phases of a trial (each moves the arms and whatever they hold) ---------------------------------------------
def post_at(j):
    return POST_TCP - [j, 0.0, 0.0]


def left_opens(f, j=0.072):
    """Left: down onto the post, grip it, pull the die open to j."""
    f.go(30, L=POST_TCP + [0, 0, 0.032]).go(20, L=POST_TCP).go(30, Lo=POST_SHUT)
    return f.go(60, L=post_at(j), j=j)


def right_takes_block(f):
    g = B_START + HANDLE
    f.go(30, R=g + [0, 0, 0.08]).go(20, R=g).go(30, Ro=HANDLE_SHUT)
    return f.go(20, R=g + [0, 0, 0.04], b=B_START + [0, 0, 0.04])


def right_seats_block(f, seat=B_SEATED):
    """Right: carry the block beside the die, slide the bar in sideways 6 mm over the bed, set it down, let go, go
    up (tool point at y -0.140, outside the zone)."""
    beside = np.array([seat[0], seat[1] - 0.100, L.TOP + L.BED_TOP_Z + 0.006])
    f.go(40, R=beside + HANDLE + [0, 0, 0.018], b=beside + [0, 0, 0.018])
    f.go(20, R=beside + HANDLE, b=beside)
    over = np.array([seat[0], seat[1], beside[2]])
    f.go(60, R=over + HANDLE, b=over)
    f.go(20, R=np.array(seat) + HANDLE, b=seat)
    f.go(30, Ro=OPEN)
    return f.go(30, R=np.array([seat[0], seat[1] - L.GRIP_FROM_BAR, UP]))


def left_blocks(f, j_stop=J_SEATED):
    """Left: ease the platen onto the bar, let go of the post, go up and park."""
    f.go(30, L=post_at(j_stop), j=j_stop).go(30, Lo=OPEN)
    return f.go(30, L=post_at(j_stop) + [0, 0, 0.052]).go(30, L=PARK_L)


def right_takes_part(f, to=P_TRAY):
    """Right: over the part, down, grip, lift it out, carry it, set it down, let go, go up."""
    g = P_START + PART_HELD
    f.go(40, R=g + [0, 0, 0.065]).go(30, R=g).go(30, Ro=PART_SHUT)
    f.go(30, R=g + [0, 0, 0.065], p=P_START + [0, 0, 0.065])
    dst = np.array(to, float)
    f.go(50, R=dst + PART_HELD + [0, 0, 0.065], p=dst + [0, 0, 0.065])
    f.go(40, R=dst + PART_HELD + [0, 0, 0.001], p=dst + [0, 0, 0.001]).go(2, p=dst)
    return f.go(30, Ro=OPEN).go(30, R=dst + PART_HELD + [0, 0, 0.09])


def right_over_handle(f, seat=B_SEATED):
    h = np.array(seat) + HANDLE
    return f.go(40, R=h + [0, 0, 0.10]).go(30, R=h)


def left_relieves(f, j_stop=J_SEATED, j=0.072):
    f.go(40, L=post_at(j_stop) + [0, 0, 0.032]).go(20, L=post_at(j_stop)).go(30, Lo=POST_SHUT)
    return f.go(30, L=post_at(j), j=j)


def right_returns_block(f, seat=B_SEATED):
    f.go(30, Ro=HANDLE_SHUT)
    lifted = np.array(seat) + [0, 0, 0.009]
    f.go(20, R=lifted + HANDLE, b=lifted)
    out = lifted + [0, -0.100, 0]
    f.go(60, R=out + HANDLE, b=out)
    f.go(40, R=B_START + HANDLE + [0, 0, 0.001], b=B_START + [0, 0, 0.001]).go(2, b=B_START)
    return f.go(30, Ro=OPEN).go(30, R=B_START + HANDLE + [0, 0, 0.10])


def left_shuts(f):
    f.go(80, L=POST_TCP, j=0.0).go(30, Lo=OPEN)
    return f.go(30, L=POST_TCP + [0, 0, 0.052]).go(30, L=PARK_L)


def full(f=None):
    """The whole procedure, as the scripted demo does it."""
    f = f or Film()
    left_opens(f)
    right_takes_block(f)
    right_seats_block(f)
    left_blocks(f)
    f.hold(10)
    right_takes_part(f)
    right_over_handle(f)
    left_relieves(f)
    right_returns_block(f)
    left_shuts(f)
    return f.hold(30)


def blocked_with_part_out(f=None):
    """Up to the part in the tray, the die still blocked."""
    f = f or Film()
    left_opens(f)
    right_takes_block(f)
    right_seats_block(f)
    left_blocks(f)
    f.hold(10)
    return right_takes_part(f)


class Geometry(unittest.TestCase):
    def test_block_tilt_is_folded(self):
        self.assertAlmostEqual(S.tilt_deg(about("x", 180)), 0.0, places=6)
        self.assertAlmostEqual(S.tilt_deg(about("x", 30)), 30.0, places=6)
        self.assertAlmostEqual(S.tilt_deg(about("y", 150)), 30.0, places=6)

    def test_bar_yaw_either_end_first(self):
        self.assertAlmostEqual(S.bar_yaw_deg(block(B_SEATED, about("z", 180))), 0.0, places=6)
        self.assertAlmostEqual(S.bar_yaw_deg(block(B_SEATED, about("z", 170))), 10.0, places=5)
        self.assertAlmostEqual(S.bar_yaw_deg(block(B_SEATED, about("z", -95))), 85.0, places=5)

    def test_block_shapes_match_the_layout(self):
        c = S.bar_corners(block(B_SEATED))
        np.testing.assert_allclose(c.min(axis=0), [0.256, -0.095, L.TOP + 0.018], atol=1e-9)
        np.testing.assert_allclose(c.max(axis=0), [0.354, -0.065, L.TOP + 0.048], atol=1e-9)
        b = block()
        np.testing.assert_allclose(b.low, [0.151, -0.255, L.TOP + 0.005], atol=1e-9)
        np.testing.assert_allclose(b.high, [0.249, -0.165, L.TOP + 0.035], atol=1e-9)
        self.assertAlmostEqual(S.bar_seat_offset(block(B_SEATED)), 0.005, places=9)  # 5 mm short of the face
        self.assertAlmostEqual(S.bar_seat_offset(block(B_SEATED + [0.005, 0, 0])), 0.0, places=9)  # pushed home

    def test_pinch_zone_follows_the_moving_face(self):
        y, z = 0.0, L.TOP + 0.05
        self.assertTrue(S.in_pinch_zone((0.302, y, z), 0.0))
        self.assertFalse(S.in_pinch_zone((0.300, y, z), 0.0))
        self.assertTrue(S.in_pinch_zone((0.252, y, z), 0.050))
        self.assertTrue(S.in_pinch_zone((0.368, y, z), 0.050))
        self.assertFalse(S.in_pinch_zone((0.370, y, z), 0.050))
        self.assertTrue(S.in_pinch_zone((0.30, -0.109, z), 0.05))
        self.assertFalse(S.in_pinch_zone((0.30, -0.111, z), 0.05))
        self.assertFalse(S.in_pinch_zone((0.30, 0.0, L.TOP + 0.141), 0.05))
        # the handle post is 28 mm in front of the zone whatever the opening, so a hand on it is never inside
        for j in (0.0, 0.03, 0.05, 0.075):
            self.assertFalse(S.in_pinch_zone(post_at(j), j))
            self.assertTrue(S.in_pinch_zone(post_at(j) + [0.0285, 0, 0], j))

    def test_post_and_platen_boxes_move_with_j(self):
        lo, hi = S.post_box(0.05)
        np.testing.assert_allclose(lo, [0.2105, 0.0475, L.TOP + 0.078], atol=1e-9)
        np.testing.assert_allclose(hi, [0.2355, 0.0725, L.TOP + 0.118], atol=1e-9)
        lo, hi = S.platen_box(0.0)
        np.testing.assert_allclose(lo, [0.2605, -0.100, L.TOP + 0.018], atol=1e-9)
        np.testing.assert_allclose(hi, [0.355, 0.100, L.TOP + 0.118], atol=1e-9)

    def test_seated_conditions(self):
        ok = S.block_seated(block(B_SEATED), J_SEATED)
        self.assertTrue(ok["seated"], ok)
        home = B_SEATED + [0.005, 0, 0]  # pushed home on the fixed face: j 0.050
        self.assertTrue(S.block_seated(block(home), 0.050)["seated"])
        cases = {
            "bar_on_bed": (block(B_SEATED + [0, 0, 0.004]), J_SEATED),
            "bar_in_seat": (block(B_SEATED + [0, -0.009, 0]), J_SEATED),
            "bar_square": (block(B_SEATED, about("z", 11)), J_SEATED),
            "j_in_band": (block(B_SEATED + [-0.012, 0, 0]), J_SEATED + 0.012),
            "face_on_bar": (block(B_SEATED), J_SEATED + 0.005),
        }
        for key, (b, j) in cases.items():
            r = S.block_seated(b, j)
            self.assertFalse(r[key], key)
            self.assertFalse(r["seated"], key)
        self.assertFalse(S.block_seated(block(B_SEATED, about("x", 15)), J_SEATED)["seated"])  # on its side
        self.assertTrue(S.block_seated(block(B_SEATED + [0, 0, 0.002], about("x", 180) * -1), J_SEATED)["bar_square"])

    def test_on_stand_and_in_tray(self):
        self.assertTrue(S.block_on_stand(block()))
        self.assertTrue(S.block_on_stand(block(B_START + [0.010, 0.010, 0])))
        self.assertFalse(S.block_on_stand(block(B_START + [0.020, 0, 0])))  # the bar's end over the pad's edge
        self.assertFalse(S.block_on_stand(block(B_START + [0, 0, 0.004])))
        self.assertFalse(S.block_on_stand(block(B_START, about("y", 20))))
        self.assertTrue(S.part_in_tray(part(P_TRAY)))
        self.assertTrue(S.part_in_tray(part(P_TRAY + [0.045, 0.060, 0.004])))
        self.assertFalse(S.part_in_tray(part(P_TRAY + [0.051, 0, 0])))
        self.assertFalse(S.part_in_tray(part(P_TRAY + [0, 0, 0.006])))
        self.assertFalse(S.part_in_tray(part(P_START)))

    def test_holding_and_near_post(self):
        a = snap(left_tcp=post_at(0.05), left_open=POST_SHUT)["left"]
        self.assertTrue(S.holding_post(a, 0.05))
        self.assertLess(S.near_post(a, 0.05), L.HAND_NEAR_POST)
        a = snap(left_tcp=post_at(0.05), left_open=OPEN)["left"]
        self.assertFalse(S.holding_post(a, 0.05))  # open fingers round it hold nothing ...
        self.assertLess(S.near_post(a, 0.05), L.HAND_NEAR_POST)  # ... but the hand is still at the post
        a = snap(left_tcp=post_at(0.05) + [0, 0, 0.092])["left"]  # finger tips 59 mm over its top: still near
        self.assertLess(S.near_post(a, 0.05), L.HAND_NEAR_POST)
        a = snap(left_tcp=post_at(0.05) + [0, 0, 0.095])["left"]
        self.assertGreater(S.near_post(a, 0.05), L.HAND_NEAR_POST)
        # An open gripper pulling the post with the outer face of one finger: the tool point is 47 mm (half the
        # open gap) plus the finger's 10 mm from the post's face, and still counts as at the post.
        lo, hi = S.post_box(0.05)
        a = snap(left_tcp=[hi[0] + 0.057, L.POST_Y, L.TOP + 0.098])["left"]
        self.assertLess(S.near_post(a, 0.05), L.HAND_NEAR_POST)
        a = snap(left_tcp=[(lo[0] + hi[0]) / 2, hi[1] + 0.047, L.TOP + 0.098])["left"]  # a finger's edge on it
        self.assertLess(S.near_post(a, 0.05), L.HAND_NEAR_POST)

    def test_holding_the_platen_anywhere(self):
        face_top = np.array([L.MOVING_FACE_X0 - 0.010 - 0.05, -0.060, L.TOP + L.FACE_PLATE[2][1] + 0.005])
        a = snap(right_tcp=face_top, right_open=0.2)["right"]  # closed on the face plate's top edge
        self.assertTrue(S.holding_platen(a, 0.05))
        self.assertFalse(S.holding_post(a, 0.05))
        self.assertFalse(S.holding_platen(snap(right_tcp=face_top)["right"], 0.05))  # open: holds nothing
        grasp = P_START + PART_HELD  # closed on the part, cheeks 30 mm away with the die blocked
        self.assertFalse(S.holding_platen(snap(right_tcp=grasp, right_open=PART_SHUT)["right"], J_SEATED))
        # a sloppy grasp at the part's near corner, 5 mm from a cheek's end: the cheeks are inside the pinch zone,
        # beside the part, so a hand there is not taken as holding the die open
        corner = np.array([L.CHEEK_RIGHT[0][1] - J_SEATED + 0.005, -0.020, L.TOP + 0.040])
        self.assertFalse(S.holding_platen(snap(right_tcp=corner, right_open=0.3)["right"], J_SEATED))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(5))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertFalse(t.zeroed)

    def test_full_procedure_scores_one(self):
        r, t = play(full())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertFalse(t.zeroed)
        self.assertIsNotNone(t.blocked_1s_step)
        self.assertIsNone(t.slammed_step)
        self.assertGreaterEqual(r.details["zone_entry_j_mm"], 50.0)

    def test_success_needs_stillness(self):
        f = full()
        f.go(3, p=P_TRAY + [0.003, 0, 0])  # the part still sliding in the tray
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        f.hold(2)
        r, _ = play(f)
        self.assertTrue(r.success)

    def test_success_needs_a_still_platen(self):
        f = full()
        f.go(2, j=0.002)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)

    def test_block_left_in_the_die_scores_075(self):
        r, t = play(blocked_with_part_out().hold(20))
        self.assertEqual(r.score, 0.75, r.details)
        self.assertFalse(r.details["die_closed"])

    def test_block_back_but_die_left_open_by_hand_scores_075(self):
        f = blocked_with_part_out()
        right_over_handle(f)
        left_relieves(f)
        right_returns_block(f)
        r, _ = play(f.hold(20))
        self.assertEqual(r.score, 0.75, r.details)
        self.assertTrue(r.details["post_held"])

    def test_slammed_shut_scores_075(self):
        f = blocked_with_part_out()
        right_over_handle(f)
        left_relieves(f)
        right_returns_block(f)
        f.go(1, Lo=OPEN).go(3, j=0.0)  # let go of the open die: it snaps shut
        f.go(30, L=PARK_L).hold(20)
        r, t = play(f)
        self.assertIsNotNone(t.slammed_step)
        self.assertEqual(r.score, 0.75, r.details)

    def test_easing_shut_while_held_is_not_a_slam(self):
        f = blocked_with_part_out()
        right_over_handle(f)
        left_relieves(f)
        right_returns_block(f)
        f.go(3, L=POST_TCP, j=0.0)  # shut fast, but by the hand on the post
        f.go(30, Lo=OPEN).go(30, L=PARK_L).hold(20)
        r, t = play(f)
        self.assertIsNone(t.slammed_step)
        self.assertEqual(r.score, 1.0, r.details)

    def test_gripper_not_let_go_scores_075(self):
        f = full()
        f.go(20, R=P_TRAY + PART_HELD).go(10, Ro=PART_SHUT).hold(20)  # back down onto the part, closed on it
        r, _ = play(f)
        self.assertEqual(r.score, 0.75, r.details)
        f = full()
        f.go(20, R=P_TRAY + PART_HELD + [0, 0, 0.020]).hold(20)  # open, but its finger tips 14 mm over the part
        r, _ = play(f)
        self.assertEqual(r.score, 0.75, r.details)

    def test_part_out_but_not_in_the_tray_scores_half(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        left_blocks(f)
        f.hold(10)
        right_takes_part(f, to=[0.200, -0.330, L.TOP + 0.020])  # on the table beside the stand
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.5, r.details)
        self.assertIsNotNone(t.removed_step)

    def test_part_lifted_but_put_back_in_the_pocket_scores_quarter(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        left_blocks(f)
        f.hold(10)
        right_takes_part(f, to=P_START)
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.25, r.details)
        self.assertIsNone(t.removed_step)

    def test_blocked_for_a_second_scores_quarter(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        left_blocks(f)
        r, t = play(f.hold(25))
        self.assertEqual(r.score, 0.25, r.details)

    def test_blocked_under_a_second_earns_nothing(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        f.go(30, L=post_at(J_SEATED), j=J_SEATED).go(30, Lo=OPEN)
        f.go(10, L=post_at(J_SEATED) + [0, 0, 0.060])  # clear of the post for a few steps ...
        f.go(10, L=post_at(J_SEATED) + [0, 0, 0.020])  # ... and back near it
        r, t = play(f.hold(10))
        self.assertIsNone(t.blocked_1s_step)
        self.assertEqual(r.score, 0.0)

    def test_score_is_idempotent(self):
        f = full()
        r, t = play(f)
        fr = f.frames[-1]
        args = (die(fr["j"]), block(fr["b"]), part(fr["p"]), rigstate.arms())
        for _ in range(3):
            self.assertEqual(t.result(*args).score, r.score)
        self.assertEqual(t.steps, len(f.frames))


class ZeroRules(unittest.TestCase):
    def test_hold_open_and_grab_scores_zero(self):
        """The method many benchmarks reward: one hand holds the die open, the other reaches in."""
        f = Film()
        left_opens(f)
        right_takes_part(f)
        r, t = play(f.hold(10))
        self.assertIn("reached_in_unblocked", t.zeroes)
        self.assertEqual(r.score, 0.0)

    def test_block_seated_but_hand_still_on_the_handle_scores_zero(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        f.go(30, L=post_at(J_SEATED), j=J_SEATED).hold(25)  # the platen on the block, the hand still holding it
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)
        self.assertEqual(r.score, 0.0)

    def test_hand_let_go_but_still_at_the_post_scores_zero(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        f.go(30, L=post_at(J_SEATED), j=J_SEATED).go(30, Lo=OPEN).hold(25)  # open fingers round the post
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)

    def test_seating_the_block_by_its_bar_scores_zero(self):
        """Holding the bar itself puts the tool point between the faces while the die is held open."""
        f = Film()
        left_opens(f)
        bar_grip = np.array([0.0, 0.0, 0.018])
        g = B_START + bar_grip
        f.go(30, R=g + [0, 0, 0.08]).go(20, R=g).go(30, Ro=HANDLE_SHUT)
        f.go(60, R=B_SEATED + bar_grip + [0, 0, 0.01], b=B_SEATED + [0, 0, 0.01])
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)

    def test_reaching_in_once_the_hand_is_off_is_allowed(self):
        """Blocked now is what the rule asks; the 1 s of blocking the tiers need can come while the part comes out."""
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        left_blocks(f)
        right_takes_part(f)  # straight in, no pause
        r, t = play(f.hold(20))
        self.assertFalse(t.zeroed)
        self.assertEqual(r.score, 0.75, r.details)

    def test_block_skewed_in_the_seat_does_not_block(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        f.go(5, bq=about("z", 12))
        left_blocks(f)
        f.hold(25)
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)

    def test_part_falls(self):
        f = blocked_with_part_out()
        f.go(10, R=P_TRAY + PART_HELD + [0, 0, 0.10], p=P_TRAY + [0, 0, 0.10], Ro=PART_SHUT)
        f.go(1, Ro=OPEN).go(3, p=P_TRAY)  # dropped from 10 cm: 0.33 m/s per step
        r, t = play(f.hold(20))
        self.assertIn("part_fell", t.zeroes)
        self.assertEqual(r.score, 0.0)

    def test_block_falls(self):
        f = Film()
        right_takes_block(f)
        f.go(10, R=B_START + HANDLE + [0, 0, 0.08], b=B_START + [0, 0, 0.08])
        f.go(1, Ro=OPEN).go(3, b=B_START)
        r, t = play(f)
        self.assertIn("block_fell", t.zeroes)

    def test_a_slow_set_down_is_not_a_fall(self):
        f = Film()
        right_takes_block(f)
        f.go(30, R=B_START + HANDLE + [0, 0, 0.001], b=B_START + [0, 0, 0.001]).go(2, b=B_START)
        r, t = play(f)
        self.assertFalse(t.zeroed)
        self.assertLess(t.fastest_drop, 0.1)

    def test_off_the_table(self):
        f = Film()
        f.go(5, p=[0.335, 0.70, L.TOP - 0.05])
        r, t = play(f)
        self.assertIn("part_off_table", t.zeroes)

    def test_die_shut_on_something_at_the_end(self):
        r, t = play(Film().go(10, j=0.020).hold(10))  # part-way shut, nobody holding it
        self.assertTrue(r.details["shut_on_something"])
        self.assertEqual(r.score, 0.0)
        f = full()
        f.go(5, j=0.020).hold(10)  # the 1.0 end, but the die stuck 20 mm open
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        f.go(5, j=0.0).hold(5)  # a current state, not a latch
        r, t = play(f)
        self.assertEqual(r.score, 1.0)

    def test_a_zero_is_never_undone(self):
        f = Film()
        left_opens(f)
        right_takes_part(f)  # hold-open-and-grab: 0 ...
        f.go(30, R=READY_R)
        right_takes_block(f)  # ... then everything else done right
        right_seats_block(f)
        f.go(30, L=post_at(J_SEATED), j=J_SEATED).go(30, Lo=OPEN).go(30, L=PARK_L).hold(30)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(t.blocked_1s_step)


class Exploits(unittest.TestCase):
    def test_one_arm_right_seats_the_block_in_the_shut_die(self):
        """A block set down on the shut die blocks nothing."""
        f = Film()
        right_takes_block(f)
        right_seats_block(f)
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)
        self.assertEqual(r.score, 0.0)

    def test_one_arm_left_opens_and_lets_go(self):
        f = Film()
        left_opens(f)
        f.go(1, Lo=OPEN).go(3, j=0.0).go(30, L=PARK_L)
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(t.slammed_step)

    def test_pushing_the_part_with_the_block(self):
        """The part sits in a pocket: shoved sideways it moves a few mm and stays; nothing is earned."""
        f = Film()
        left_opens(f)
        f.go(10, p=P_START + [0, -0.008, 0])
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(t.removed_step)

    def test_block_left_seated_and_part_dropped_in_the_tray_from_low(self):
        """Releasing the part 1 mm over the tray floor is a set-down, not a fall."""
        r, t = play(blocked_with_part_out())
        self.assertFalse(t.zeroed)

    def test_one_arm_cams_the_die_open_with_the_block(self):
        """One arm alone turning the bar between the faces could push the platen open and leave the bar seated with
        the platen resting on it; nobody pulled the die open by its handle, so it is not blocked: reaching in is 0."""
        f = Film()
        right_takes_block(f)
        mid = np.array([0.335, -0.116, L.TOP + L.BED_TOP_Z])  # the bar along y, 33 mm of it between the shut faces
        f.go(40, R=mid + about_z_handle(90), b=mid, bq=about("z", 90))
        f.go(60, R=B_SEATED + HANDLE, b=B_SEATED, bq=Q0, j=J_SEATED)  # turned home: the bar has pushed it open
        f.go(30, Ro=OPEN).go(30, R=B_SEATED + HANDLE + [0, 0, 0.10]).hold(30)
        r, t = play(f)
        self.assertTrue(r.details["block_seated"], r.details)
        self.assertFalse(t.pulled_open)
        self.assertIsNone(t.blocked_1s_step)
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)

    def test_pulled_open_lapses_when_the_die_shuts(self):
        """Pulling the die open once and letting it slam shut does not count for a later opening by the block."""
        f = Film()
        left_opens(f)
        f.go(1, Lo=OPEN).go(3, j=0.0).go(30, L=PARK_L)
        right_takes_block(f)
        f.go(60, R=B_SEATED + HANDLE, b=B_SEATED, j=J_SEATED)
        f.go(30, Ro=OPEN).go(30, R=B_SEATED + HANDLE + [0, 0, 0.10]).hold(30)
        r, t = play(f)
        self.assertFalse(t.pulled_open)
        self.assertIsNone(t.blocked_1s_step)

    def test_holding_the_face_plate_instead_of_the_post(self):
        """A hand that keeps the platen on (or just off) the bar by its face plate holds the die: not blocked."""
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        plate = np.array([L.MOVING_FACE_X0 - 0.012 - J_SEATED, -0.060, L.TOP + L.FACE_PLATE[2][1] + 0.005])
        f.go(20, L=post_at(J_SEATED), j=J_SEATED)
        f.go(30, R=plate + [0, 0, 0.03]).go(20, R=plate).go(30, Ro=0.2)  # the right closes on the face plate
        f.go(30, Lo=OPEN).go(30, L=PARK_L).hold(30)  # the left lets go of the post and goes away
        r, t = play(f)
        self.assertTrue(r.details["platen_held"], r.details)
        self.assertIsNone(t.blocked_1s_step)

    def test_block_on_its_side_does_not_count(self):
        """The bar is square in section, so on its side it would still hold the die, but the handle points up and
        it is not lying in its seat: not blocked, and reaching in scores 0."""
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        f.go(5, bq=about("x", 90), b=B_SEATED + [0, 0.015, 0.015])
        left_blocks(f)
        f.hold(25)
        right_takes_part(f)
        r, t = play(f)
        self.assertIn("reached_in_unblocked", t.zeroes)
        self.assertIsNone(t.blocked_1s_step)


class Robustness(unittest.TestCase):
    def test_arms_unavailable(self):
        r, t = play(full(), arms=False)
        self.assertEqual(t.arms_unavailable, len(full().frames))
        self.assertEqual(r.score, 1.0)  # the gripper rules are taken as kept
        self.assertIsNone(r.details["grippers_let_go"])

    def test_arm_rules_resume_when_the_arms_come_back(self):
        f = Film()
        left_opens(f)
        right_takes_part(f)
        n = len(f.frames)
        t = Tracker(HZ)
        play(f, t, arms=False, until=n // 2)
        self.assertFalse(t.zeroed)
        f2 = Film()
        f2.frames = f.frames[n // 2:]
        play(f2, t)
        self.assertIn("reached_in_unblocked", t.zeroes)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        a = snap(left_tcp=[np.nan, 0, 0])
        r = t.update(die(), block(), part(), a)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        a = snap(right_open=np.inf)
        t.update(die(), block(), part(), a)
        self.assertEqual(t.arms_unavailable, 2)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_reset()
        a = snap(left_tcp=post_at(0.0), left_open=POST_SHUT)
        self.assertTrue(S.holding_post(a["left"], 0.0))
        np.testing.assert_allclose(a["left"].tips, post_at(0.0) - [0, 0, TIPS])

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = Film()
        left_opens(f)
        right_takes_block(f)
        right_seats_block(f)
        left_blocks(f)
        r, t = play(f.hold(25))
        self.assertEqual(r.score, 0.25)
        before = (t.steps, t.blocked_run, dict(t.zeroes), t.prev)
        a = snap(left_tcp=PARK_L)
        bad = [
            (die(np.nan), block(B_SEATED), part()),
            (die(J_SEATED, np.inf), block(B_SEATED), part()),
            (die(J_SEATED), block(B_SEATED, quat=[0, 0, 0, 0]), part()),
            (die(J_SEATED), block(B_SEATED), part(quat=[np.nan, 0, 0, 1])),
            (die(J_SEATED), block([np.nan, 0, 0]), part()),
            (SimpleNamespace(), block(B_SEATED), part()),
            (die(J_SEATED), block(B_SEATED), SimpleNamespace(pos=None)),
        ]
        for d, b, p in bad:
            r = t.update(d, b, p, a)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.bad_steps, len(bad))
        self.assertEqual((t.steps - len(bad), t.blocked_run, t.zeroes), before[:3])
        r = t.update(die(J_SEATED), block(B_SEATED), part(), a)
        self.assertEqual(r.score, 0.25)

    def test_states_anywhere_do_not_crash(self):
        t = Tracker(HZ)
        rng = np.random.default_rng(3)
        for _ in range(200):
            q = rng.normal(size=4)
            a = snap(rng.uniform(-1, 1, 3) + [0, 0, L.TOP], rng.uniform(0, 1), rng.uniform(-1, 1, 3) + [0, 0, L.TOP],
                     rng.uniform(0, 1))
            r = t.update(die(rng.uniform(-0.01, 0.08), rng.normal()), block(rng.uniform(-1, 1, 3) + [0, 0, L.TOP], q),
                         part(rng.uniform(-1, 1, 3) + [0, 0, L.TOP], rng.normal(size=4)), a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Resets(unittest.TestCase):
    def setUp(self):
        f = Film()
        left_opens(f)
        right_takes_part(f)  # a zero
        self.film = f

    def test_teleport_back_to_start_resets(self):
        r, t = play(self.film)
        self.assertTrue(t.zeroed)
        f = Film()
        f.frames = [dict(self.film.frames[-1], j=0.0, b=B_START, p=P_START, L=READY_L, R=READY_R, Lo=OPEN, Ro=OPEN)]
        r, t = play(f, t)
        self.assertFalse(t.zeroed)
        self.assertEqual(t.steps, 1)

    def test_teleport_to_a_jittered_start_resets(self):
        r, t = play(self.film)
        f = Film()
        f.frames = [dict(self.film.frames[-1], j=0.0, b=B_START + [0.02, -0.02, 0], p=P_START + [0.01, 0.01, 0])]
        play(f, t)
        self.assertFalse(t.zeroed)

    def test_gradual_return_does_not_reset(self):
        r, t = play(self.film)
        f = Film(p=self.film.frames[-1]["p"], j=0.072, L=self.film.frames[-1]["L"], Lo=POST_SHUT)
        f.go(60, p=P_START, j=0.0)
        play(f, t)
        self.assertTrue(t.zeroed)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        r, t = play(self.film)
        f = Film()
        f.frames = [dict(self.film.frames[-1], p=P_START + [0.2, -0.3, 0])]
        play(f, t)
        self.assertTrue(t.zeroed)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = Tracker(HZ)
        saved = dict(rigstate._RESET_CALLBACKS)
        try:
            @rigstate.on_reset
            def _hook():
                t.reset()

            play(self.film, t)
            self.assertTrue(t.zeroed)
            rigstate.offline_reset()
            self.assertFalse(t.zeroed)
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
class StandInModel(unittest.TestCase):
    """standin.py, the offline stand-in for the platen and the block (mechkit needs yam_arena.task_file)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        from standin import StandIn
        cls.StandIn = StandIn

    def test_post_follows_the_hand_and_snaps_shut(self):
        s = self.StandIn(self.task.DIE, self.task.BLOCK)
        s.step(snap(left_tcp=POST_TCP, left_open=0.5))
        self.assertEqual(s.post_holder, "left")
        for x in np.linspace(L.POST_X0, L.POST_X0 - 0.072, 20):
            d, b = s.step(snap(left_tcp=[x, L.POST_Y, POST_TCP[2]], left_open=POST_SHUT))
        self.assertAlmostEqual(d.joints["platen"], 0.072, places=6)
        self.assertEqual(d.source, "offline")
        d, b = s.step(snap(left_tcp=post_at(0.072), left_open=0.7))  # let go: it snaps shut
        self.assertIsNone(s.post_holder)
        self.assertLess(d.joint_vel["platen"], -L.SLAM_SPEED)
        for _ in range(20):
            d, b = s.step(snap(left_tcp=post_at(0.072)))
        self.assertLess(d.joints["platen"], 1e-4)

    def test_block_carried_dropped_and_holding_the_die(self):
        s = self.StandIn(self.task.DIE, self.task.BLOCK)
        s.step(snap(left_tcp=POST_TCP, left_open=0.3, right_tcp=B_START + HANDLE, right_open=0.3))
        self.assertEqual((s.post_holder, s.block_holder), ("left", "right"))
        s.step(snap(left_tcp=post_at(0.072), left_open=0.3, right_tcp=B_SEATED + HANDLE + [0, 0, 0.0005],
                    right_open=0.3))
        d, b = s.step(snap(left_tcp=post_at(0.072), left_open=0.3, right_tcp=B_SEATED + HANDLE + [0, 0, 0.0005],
                           right_open=0.8))
        np.testing.assert_allclose(b.origin, B_SEATED, atol=1e-9)  # dropped 0.5 mm onto the bed
        self.assertEqual(b.lin_vel[2], 0.0)
        for _ in range(3):
            d, b = s.step(snap(left_tcp=post_at(0.045), left_open=0.3, right_tcp=READY_R))
        self.assertAlmostEqual(d.joints["platen"], J_SEATED, places=9)  # pressed on the bar, not past it
        for _ in range(10):
            d, b = s.step(snap(left_tcp=PARK_L, right_tcp=READY_R))
        self.assertAlmostEqual(d.joints["platen"], J_SEATED, places=9)
        self.assertTrue(S.block_seated(b, d.joints["platen"])["seated"])

    def test_a_high_drop_shows_as_a_fall(self):
        s = self.StandIn(self.task.DIE, self.task.BLOCK)
        s.step(snap(right_tcp=B_START + HANDLE, right_open=0.3))
        s.step(snap(right_tcp=B_START + HANDLE + [0, 0, 0.05], right_open=0.3))
        d, b = s.step(snap(right_tcp=B_START + HANDLE + [0, 0, 0.05], right_open=0.9))
        self.assertLess(b.lin_vel[2], -L.FALL_SPEED)
        np.testing.assert_allclose(b.origin, B_START, atol=1e-9)

    def test_a_block_let_go_on_the_shut_die_rests_on_the_platen(self):
        """The right arm alone: the bar set where the shut platen is lands on the platen, and the spring does not
        open the die onto it."""
        s = self.StandIn(self.task.DIE, self.task.BLOCK)
        s.step(snap(right_tcp=B_START + HANDLE, right_open=0.3))
        s.step(snap(right_tcp=B_SEATED + HANDLE + [0, 0, 0.0005], right_open=0.3))
        for _ in range(10):
            d, b = s.step(snap(right_tcp=B_SEATED + HANDLE + [0, 0, 0.0005], right_open=0.9))
        self.assertEqual(d.joints["platen"], 0.0)
        self.assertAlmostEqual(b.low[2], L.TOP + L.FACE_PLATE[2][1], places=9)
        self.assertFalse(S.block_seated(b, 0.0)["seated"])

    def test_arms_unavailable_hold_everything(self):
        s = self.StandIn(self.task.DIE, self.task.BLOCK)
        d, b = s.step(None)
        self.assertEqual(d.joints["platen"], 0.0)
        np.testing.assert_allclose(b.origin, B_START)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim_objects(self, j=0.0, b=B_START, p=P_START):
        """mechkit states marked 'sim' (as the simulator reports them): the stand-in must stay out of it."""
        d = self.mk.state_from_poses(self.task.DIE, (*L.DIE_ORIGIN, 0, 0, 0, 1), joints={"platen": j}, source="sim")
        blk = self.mk.state_from_poses(self.task.BLOCK, (*b, 0, 0, 0, 1), source="sim")
        return {"die": d, "block": blk, "part": part(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in ("die", "block", "part", "die_bed", "block_stand", "tray_floor"):
            self.assertIn(n, names)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertIsNone(self.task.DIE.mass)  # fixed
        self.assertEqual(self.task.BLOCK.mass, L.BLOCK_MASS)
        j = self.task.DIE.link("platen")
        self.assertEqual((j.kind, j.axis, j.upper, j.spring, j.damping), ("prismatic", "-X", L.PLATEN_TRAVEL,
                                                                          L.PLATEN_SPRING, L.PLATEN_DAMPING))

    def test_scorer_geometry_matches_mechkit(self):
        for jj in (0.0, 0.05, 0.075):
            st = self.mk.state_from_poses(self.task.DIE, (*L.DIE_ORIGIN, 0, 0, 0, 1), joints={"platen": jj},
                                          source="sim")
            lo, hi = st.link_bounds("platen")
            elo, ehi = S.platen_box(jj)
            np.testing.assert_allclose(lo, elo, atol=1e-9)
            np.testing.assert_allclose(hi, ehi, atol=1e-9)
        lo, hi = self.task.DIE.world_bounds()
        np.testing.assert_allclose(lo, [L.BED_X0, L.BED_Y0, L.TOP + L.BED_SLAB_Z], atol=1e-9)
        np.testing.assert_allclose(hi, [L.BED_X1, L.BED_Y1, L.TOP + L.POST_Z1], atol=1e-9)
        b = self.mk.state_from_poses(self.task.BLOCK, (*B_SEATED, *about("z", 7)), source="sim")
        np.testing.assert_allclose(S.block_corners(b).min(axis=0), b.low, atol=1e-9)
        np.testing.assert_allclose(S.block_corners(b).max(axis=0), b.high, atol=1e-9)
        # the bed's top layer covers its outline except the pocket (its pieces' areas add up)
        area = sum(s.size[0] * s.size[1] for s in self.task.DIE_BASE[:7])
        pocket = (L.POCKET_X1 - L.POCKET_X0) * (L.POCKET_Y1 - L.POCKET_Y0)
        self.assertAlmostEqual(area + pocket, (L.BED_X1 - L.BED_X0) * (L.BED_Y1 - L.BED_Y0), places=9)

    def test_sim_states_bypass_the_stand_in(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.sim_objects()))
        self.assertEqual(self.task._STANDIN.steps, 0)
        snap(left_tcp=post_at(0.05), left_open=POST_SHUT, right_tcp=P_START + PART_HELD, right_open=PART_SHUT)
        self.task.success(self.sim_objects(j=0.05))  # held open by hand, reaching in: the real joint says so
        self.assertIn("reached_in_unblocked", tr.zeroes)
        self.assertEqual(self.task._STANDIN.steps, 0)
        self.assertEqual(self.task.score(self.sim_objects(j=0.05)), 0.0)

    def test_offline_states_use_the_stand_in_and_reset_hooks(self):
        tr, si = self.task._TRACKER, self.task._STANDIN
        rigstate.offline_reset()
        self.mk.offline_reset()
        objs = {"die": self.task.DIE.offline_state(die()), "block": self.task.BLOCK.rest_state(), "part": part()}
        objs["block"] = self.task.BLOCK.offline_state(
            SimpleNamespace(pos=objs["block"].pos, quat_xyzw=Q0, lin_vel=np.zeros(3), ang_vel=np.zeros(3)))
        self.assertEqual(objs["die"].source, "offline")
        snap(left_tcp=POST_TCP, left_open=0.3)
        self.task.success(objs)
        self.assertEqual(si.post_holder, "left")
        snap(left_tcp=post_at(0.03), left_open=0.3)
        self.task.success(objs)
        self.assertAlmostEqual(tr.last.details["j_mm"], 30.0)
        self.assertEqual(self.task.score(objs), 0.0)
        self.assertEqual(si.steps, 2)  # score() does not advance it
        rigstate.offline_reset()  # rigstate's reset hook: tracker and stand-in start over
        self.assertEqual((tr.steps, si.steps, si.post_holder), (0, 0, None))
        snap()
        self.task.success(objs)
        self.mk.offline_reset()  # mechkit's episode counter: also a new trial
        self.task.success(objs)
        self.assertEqual(tr.steps, 1)

    def test_arms_none_right_after_a_reset(self):
        rigstate.offline_reset()
        self.assertFalse(self.task.success(self.sim_objects()))  # arms None: no crash
        self.assertEqual(self.task._TRACKER.arms_unavailable, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.sim_objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]

        def end_of(wps, label, nth=0):
            t, seen = 0.0, 0
            for w in wps:
                t += w.duration
                if w.label == label:
                    if seen == nth:
                        return t
                    seen += 1
            raise AssertionError(label)

        def start_of(wps, label, nth=0):
            t, seen = 0.0, 0
            for w in wps:
                if w.label == label:
                    if seen == nth:
                        return t
                    seen += 1
                t += w.duration
            raise AssertionError(label)

        # the die is open before the bar goes in; the left lets go only once the right gripper is up and clear
        self.assertLessEqual(end_of(left, "pull the die open"), start_of(right, "slide the bar into the seat"))
        self.assertLessEqual(end_of(right, "up, beside the die"), start_of(left, "ease the platen onto the block"))
        # the right reaches in only after the left has parked (blocked for more than 1 s by then)
        self.assertGreaterEqual(start_of(right, "over the part") - end_of(left, "away from the die"), 0.5)
        # the left goes back to the post only once the right is out of the zone, over the block's handle
        self.assertGreaterEqual(start_of(left, "back over the handle post"), end_of(right, "over the block's handle"))
        # the right takes the block only once the platen is off it; the left shuts the die once the bar is out
        self.assertLessEqual(end_of(left, "pull the platen off the block"), start_of(right, "grip the handle", 1))
        self.assertLessEqual(end_of(right, "slide the bar out of the die"), start_of(left, "ease the die shut"))
        # every grip and release takes 1.5 s; the last 2 cm of every descent onto something take 1.5 s
        for w in left + right:
            if w.label.startswith(("grip", "let go")):
                self.assertGreaterEqual(w.duration, 1.5, w.label)
            if w.label.startswith("slowly down"):
                self.assertGreaterEqual(w.duration, 1.5, w.label)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
