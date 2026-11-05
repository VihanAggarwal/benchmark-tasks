"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
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
    SHAPES, Tracker, cover_back, cover_off, covers_opening, entered, grip_point, grippers_clear, holders, in_column,
    min_clearance, monitored, probe_in_holder, probe_tip, rotation, tilt_deg, tip_band, tip_speed, tool_on_pad,
    tool_out, valid, zone_box_gap,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
CH = L.COVER_SIZE[2] / 2
TH = L.TOOL_SIZE[2] / 2
MX, MYL, MYR = L.MANWAY_X, L.MANWAY_Y["left"], L.MANWAY_Y["right"]
SYL, SYR = L.STAND_Y["left"], L.STAND_Y["right"]

CL_START = np.array([MX, MYL, L.DECK_Z + CH])
CR_START = np.array([MX, MYR, L.DECK_Z + CH])
CL_STAND = np.array([L.STAND_X, SYL, L.STAND_TOP + CH])
CR_STAND = np.array([L.STAND_X, SYR, L.STAND_TOP + CH])
P_START = np.array([L.HOLDER_X, L.HOLDER_Y, L.TOP + L.PROBE_LEN / 2])
T_START = np.array([L.TOOL_X, L.TOOL_Y, L.FLOOR_Z + TH])
T_PAD = np.array([L.PAD_X, L.PAD_Y, L.PAD_TOP + TH])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])

COVER_UP = 0.010  # TCP above a held cover's centre
PROBE_DOWN = L.PROBE_LEN / 2 - L.PROBE_GRIP_DOWN  # 0.025: a held probe's centre is this far below the TCP
TIP_DOWN = L.PROBE_LEN - L.PROBE_GRIP_DOWN  # 0.075: and its tip this far
TOOL_UP = 0.009  # TCP above a held tool's centre
HOVER = L.TOP + 0.130
CARRY_P = L.TOP + 0.155
UP = L.TOP + 0.150
OPEN = 1.0
COVER_O = (L.COVER_SIZE[1] - 0.008) / L.FULL_GAP
PROBE_O = (2 * L.PROBE_RADIUS - 0.008) / L.FULL_GAP
TOOL_O = (L.TOOL_SIZE[0] - 0.008) / L.FULL_GAP
ENTRY_O = 0.074 / L.FULL_GAP
DWELL_TIPS = (0.050, 0.033, 0.016)  # tip heights above the table top
R_OPEN = math.hypot(L.FULL_GAP / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH)  # an open gripper's zone radius (60 mm)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.COVER_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def cyl(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like probe (upright cylinder in its body frame)."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.abs(a) * L.PROBE_LEN / 2 + L.PROBE_RADIUS * np.sqrt(np.clip(1 - a * a, 0.0, 1.0))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def tool(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    return box(center, quat, vel, spin, size=L.TOOL_SIZE)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


POS = ("cl", "cr", "p", "t", "L", "R")
QUATS = ("clq", "crq", "pq", "tq")
KEYS = POS + QUATS + ("Lo", "Ro")


class Film:
    """A trial as a list of frames: the four loose objects' centres and orientations, both TCPs and openings. ``go``
    moves linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"cl": CL_START, "cr": CR_START, "p": P_START, "t": T_START, "L": READY_L, "R": READY_R,
                    "clq": Q0, "crq": Q0, "pq": Q0, "tq": Q0, "Lo": OPEN, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in POS else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in POS else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in QUATS:
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def __add__(self, other):
        out = Film()
        out.frames = self.frames + other.frames
        out.cur = dict(other.cur)
        return out


def states(fr, prev):
    v = {k: np.zeros(3) if prev is None else (fr[k] - prev[k]) / DT for k in ("cl", "cr", "p", "t")}
    return (box(fr["cl"], fr["clq"], v["cl"]), box(fr["cr"], fr["crq"], v["cr"]), cyl(fr["p"], fr["pq"], v["p"]),
            tool(fr["t"], fr["tq"], v["t"]))


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(*states(fr, prev), a)
        prev = fr
    return r, t


# -- the steps of the intended trial ------------------------------------------------------------------------------
def covers_off(f, sides=("left", "right")):
    """Each named arm lifts its cover straight off its manway and sets it on its stand (both at once)."""
    k = {"left": ("L", "Lo", "cl", CL_START, CL_STAND), "right": ("R", "Ro", "cr", CR_START, CR_STAND)}
    use = [k[s] for s in sides]

    def move(n, fn):
        f.go(n, **{kk: v for a in use for kk, v in fn(*a).items()})

    move(30, lambda arm, o, c, s0, s1: {arm: s0 + [0, 0, COVER_UP + 0.05]})
    move(20, lambda arm, o, c, s0, s1: {arm: s0 + [0, 0, COVER_UP]})
    move(30, lambda arm, o, c, s0, s1: {o: COVER_O})
    move(20, lambda arm, o, c, s0, s1: {arm: [*s0[:2], HOVER], c: [*s0[:2], HOVER - COVER_UP]})
    move(40, lambda arm, o, c, s0, s1: {arm: [*s1[:2], HOVER], c: [*s1[:2], HOVER - COVER_UP]})
    move(40, lambda arm, o, c, s0, s1: {arm: s1 + [0, 0, COVER_UP], c: s1})
    move(30, lambda arm, o, c, s0, s1: {o: OPEN})
    move(20, lambda arm, o, c, s0, s1: {arm: [*s1[:2], UP]})
    return f


def left_takes_probe(f):
    g = P_START + [0, 0, PROBE_DOWN]
    f.go(30, L=g + [0, 0, 0.05]).go(20, L=g).go(30, Lo=PROBE_O)
    f.go(30, L=[*g[:2], CARRY_P], p=[*P_START[:2], CARRY_P - PROBE_DOWN])
    f.go(40, L=[MX, MYL, CARRY_P], p=[MX, MYL, CARRY_P - PROBE_DOWN])
    return f


def probe_to(f, n, tip_h, xy=(MX, MYL)):
    """The left gripper moves the held probe so that its tip is at height ``tip_h`` above the table."""
    tcp = L.TOP + tip_h + TIP_DOWN
    return f.go(n, L=[*xy, tcp], p=[*xy, tcp - PROBE_DOWN])


def left_tests(f, tips=DWELL_TIPS, hold=50, xy=(MX, MYL)):
    """Lower the tip to each depth in turn and hold it still ``hold`` steps (50 = 2.5 s)."""
    for h in tips:
        probe_to(f, 30, h, xy).hold(hold)
    return f


def right_takes_tool(f, to=T_PAD, open_at_end=True):
    """The right gripper goes down through the right manway, grips the tool, lifts it out, sets it at ``to``."""
    f.go(30, R=[MX, MYR, HOVER], Ro=ENTRY_O)
    g = T_START + [0, 0, TOOL_UP]
    f.go(40, R=g).go(30, Ro=TOOL_O)
    f.go(40, R=[*g[:2], L.TOP + 0.110], t=[*T_START[:2], L.TOP + 0.110 - TOOL_UP])
    f.go(40, R=[*to[:2], L.TOP + 0.110], t=[*to[:2], L.TOP + 0.110 - TOOL_UP])
    f.go(40, R=np.array(to) + [0, 0, TOOL_UP], t=to)
    if open_at_end:
        f.go(30, Ro=OPEN).go(20, R=[*to[:2], UP])
    return f


def left_returns_probe(f, to=P_START):
    f.go(30, L=[MX, MYL, CARRY_P], p=[MX, MYL, CARRY_P - PROBE_DOWN])
    f.go(40, L=[*to[:2], CARRY_P], p=[*to[:2], CARRY_P - PROBE_DOWN])
    f.go(40, L=np.array(to) + [0, 0, PROBE_DOWN], p=to)
    f.go(30, Lo=OPEN).go(20, L=[*to[:2], UP])
    return f


def covers_on(f, left_at=CL_START, right_at=CR_START, sides=("left", "right")):
    """Each named arm takes its cover from its stand and sets it at ``*_at`` (on the deck), then goes up."""
    k = {"left": ("L", "Lo", "cl", CL_STAND, np.array(left_at, float)),
         "right": ("R", "Ro", "cr", CR_STAND, np.array(right_at, float))}
    use = [k[s] for s in sides]

    def move(n, fn):
        f.go(n, **{kk: v for a in use for kk, v in fn(*a).items()})

    move(30, lambda arm, o, c, s0, s1: {arm: [*s0[:2], HOVER]})
    move(20, lambda arm, o, c, s0, s1: {arm: s0 + [0, 0, COVER_UP]})
    move(30, lambda arm, o, c, s0, s1: {o: COVER_O})
    move(20, lambda arm, o, c, s0, s1: {arm: [*s0[:2], HOVER], c: [*s0[:2], HOVER - COVER_UP]})
    move(40, lambda arm, o, c, s0, s1: {arm: [*s1[:2], HOVER], c: [*s1[:2], HOVER - COVER_UP]})
    move(40, lambda arm, o, c, s0, s1: {arm: s1 + [0, 0, COVER_UP], c: s1})
    move(30, lambda arm, o, c, s0, s1: {o: OPEN})
    move(20, lambda arm, o, c, s0, s1: {arm: [*s1[:2], UP]})
    return f


def up_to_tool(tips=DWELL_TIPS):
    """The trial up to the test being complete, the probe held at the bottom depth."""
    f = Film().hold(5)
    covers_off(f)
    left_takes_probe(f)
    left_tests(f, tips)
    return f


def procedure(settle=30, left_at=CL_START, right_at=CR_START, probe_at=P_START, tool_at=T_PAD):
    """The intended trial: covers off, probe in, three tests top first, tool out onto the pad while the probe is
    held at the bottom depth, probe back in its holder, covers back on; all still for ``settle`` steps."""
    f = up_to_tool()
    right_takes_tool(f, tool_at)
    left_returns_probe(f, probe_at)
    covers_on(f, left_at, right_at)
    return f.hold(settle)


# ==================================================================================================================
class Geometry(unittest.TestCase):
    def test_tilt_is_folded(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)  # upside down: the same plain shape
        self.assertAlmostEqual(tilt_deg(about("x", 170)), 10.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("y", 90)), 90.0, places=4)

    def test_probe_tip_and_grip_point(self):
        p = cyl(P_START)
        np.testing.assert_allclose(probe_tip(p), [L.HOLDER_X, L.HOLDER_Y, L.TOP], atol=1e-9)
        np.testing.assert_allclose(grip_point(p), P_START + [0, 0, PROBE_DOWN], atol=1e-9)
        flipped = cyl(P_START, about("x", 180))  # either end may be the lower one
        np.testing.assert_allclose(probe_tip(flipped), probe_tip(p), atol=1e-9)
        leaning = cyl(P_START, about("y", 20))
        self.assertAlmostEqual(float(probe_tip(leaning)[2]), P_START[2] - 0.05 * math.cos(math.radians(20)))

    def test_tip_speed_includes_spin(self):
        self.assertAlmostEqual(tip_speed(cyl(P_START, vel=(0.003, 0, 0))), 0.003)
        # still centre, spinning about x at 0.2 rad/s: the tip 50 mm below moves at 10 mm/s
        self.assertAlmostEqual(tip_speed(cyl(P_START, spin=(0.2, 0, 0))), 0.010)

    def test_bands_and_column(self):
        for i, h in enumerate(DWELL_TIPS):
            p = cyl([MX, MYL, L.TOP + h + L.PROBE_LEN / 2])
            self.assertEqual(tip_band(p), i)
            self.assertTrue(in_column(probe_tip(p)))
        self.assertIsNone(tip_band(cyl([MX, MYL, L.TOP + 0.060 + 0.05])))  # between the deck and the top band
        self.assertIsNone(tip_band(cyl([MX, MYL, L.TOP + 0.025 + 0.05])))  # between the middle and bottom bands
        self.assertFalse(in_column([MX, MYR, L.TOP + 0.03]))  # the right manway's column is not the test column
        self.assertFalse(in_column([MX + 0.065, MYL, L.TOP + 0.03]))

    def test_covers(self):
        self.assertFalse(cover_off(box(CL_START), "left"))
        self.assertFalse(cover_off(box(CL_START + [0, 0, 0.05]), "left"))  # lifted, still over the opening
        self.assertFalse(cover_off(box(CL_START + [0, 0.055, 0.05]), "left"))  # 5 mm still over it
        self.assertTrue(cover_off(box(CL_START + [0, 0.065, 0.05]), "left"))
        self.assertTrue(cover_off(box(CL_STAND), "left"))
        self.assertTrue(cover_back(box(CL_START), "left"))
        self.assertTrue(cover_back(box(CR_START), "right"))
        self.assertFalse(cover_back(box(CR_START), "left"))
        self.assertTrue(cover_back(box(CL_START + [0.009, 0, 0]), "left"))
        self.assertFalse(cover_back(box(CL_START + [0.012, 0, 0]), "left"))  # centre 12 mm off
        self.assertFalse(cover_back(box(CL_START + [0, 0, 0.005]), "left"))  # held 5 mm above the deck
        self.assertTrue(cover_back(box(CL_START + [0, 0, 0.002]), "left"))  # 2 mm: the arm's sag, contact jitter
        self.assertFalse(cover_back(box(CL_START, about("x", 6)), "left"))  # tilted
        self.assertTrue(cover_back(box(CL_START, about("z", 8)), "left"))  # turned 8 deg: still covers it
        turned = box(CL_START, about("z", 15))  # turned 15 deg: a corner of the opening shows
        self.assertFalse(covers_opening(turned, "left"))
        self.assertFalse(cover_back(turned, "left"))
        self.assertTrue(cover_back(box(CL_START, about("z", 180)), "left"))  # the same plate turned round

    def test_cover_off_uses_its_real_outline(self):
        # A plate turned 45 deg beyond the opening's front-left corner: its world-aligned bounding box still overlaps
        # the opening, but the plate itself is 10 mm clear of it (its short side faces the corner, 35 mm from its
        # centre, which is 45 mm from the corner).
        u = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        corner = np.array([MX + L.MANWAY_HALF[0], MYL + L.MANWAY_HALF[1], CL_START[2]])
        q = about("z", -45)  # long side along (1, -1)
        clear = box(corner + 0.045 * u, q)
        self.assertTrue(clear.low[0] < corner[0] and clear.low[1] < corner[1])  # the box does overlap
        self.assertTrue(cover_off(clear, "left"))
        self.assertFalse(cover_off(box(corner + 0.030 * u, q), "left"))  # 5 mm over the corner
        self.assertFalse(cover_off(box(CL_START + [0, 0, 0.05], about("x", 25)), "left"))  # lifted, tilted, over it
        self.assertTrue(cover_off(box(CL_STAND, about("z", 90)), "left"))  # turned on its stand

    def test_tool(self):
        self.assertFalse(tool_out(tool(T_START)))
        self.assertFalse(tool_out(tool(T_START + [0, 0, 0.09])))  # lifted, but still over the tank
        self.assertTrue(tool_out(tool([0.25, MYR, L.TOP + 0.10])))
        self.assertFalse(tool_out(tool([0.25, MYR, L.TOP + 0.01])))  # beside the tank, low: never lifted out
        self.assertTrue(tool_on_pad(tool(T_PAD)))
        self.assertTrue(tool_on_pad(tool(T_PAD + [0.04, -0.04, 0.002], about("z", 30))))
        self.assertFalse(tool_on_pad(tool(T_PAD + [0.06, 0, 0])))  # centre off the pad
        self.assertFalse(tool_on_pad(tool(T_PAD + [0, 0, 0.005])))  # still held above it
        self.assertFalse(tool_on_pad(tool(T_PAD, about("x", 90))))  # on its side
        self.assertTrue(tool_on_pad(tool(T_PAD, about("x", 180))))  # upside down: the same plain block

    def test_probe_in_holder(self):
        self.assertTrue(probe_in_holder(cyl(P_START)))
        self.assertTrue(probe_in_holder(cyl(P_START + [0.003, -0.003, 0.002])))
        self.assertFalse(probe_in_holder(cyl(P_START, about("y", 12))))
        self.assertFalse(probe_in_holder(cyl(P_START + [0, 0, 0.006])))  # held above the table
        self.assertFalse(probe_in_holder(cyl(P_START + [0.03, 0, 0])))  # standing beside the holder
        self.assertFalse(probe_in_holder(cyl(P_START + [0, 0, 0.025])))  # resting on the holder's walls
        # leaning 12 deg about its tip, which stands at the socket's centre on the table: only the tilt is wrong
        q = about("y", 12)
        up = rotation(q)[:, 2]
        lean = cyl(np.array([L.HOLDER_X, L.HOLDER_Y, L.TOP + 0.0017]) + up * L.PROBE_LEN / 2, q)
        self.assertLess(abs(float(lean.low[2]) - L.TOP), L.PROBE_BOTTOM_TOL)
        self.assertLess(float(np.hypot(*(probe_tip(lean)[:2] - [L.HOLDER_X, L.HOLDER_Y]))), 1e-9)
        self.assertFalse(probe_in_holder(lean))

    def test_entered(self):
        a = snap(right_tcp=[MX, MYR, L.DECK_Z + TIPS - 0.001])["right"]
        self.assertTrue(entered(a))
        a = snap(right_tcp=[MX, MYR, L.DECK_Z + TIPS + 0.001])["right"]
        self.assertFalse(entered(a))
        a = snap(right_tcp=[MX, MYR, L.TOP + 0.027])["right"]
        self.assertTrue(entered(a))
        # beside the tank, low: 4 mm outside the footprint counts (grown 5 mm), 10 mm outside does not
        a = snap(right_tcp=[L.TANK_X1 + 0.004, MYR, L.TOP + 0.03])["right"]
        self.assertTrue(entered(a))
        a = snap(right_tcp=[L.TANK_X1 + 0.010, MYR, L.TOP + 0.03])["right"]
        self.assertFalse(entered(a))
        a = snap(right_tcp=CR_STAND + [0, 0, COVER_UP])["right"]  # setting the cover on its stand
        self.assertFalse(entered(a))

    def test_held_and_monitored(self):
        tcp = np.array([MX, MYL, L.TOP + DWELL_TIPS[2] + TIP_DOWN])
        p = cyl(tcp - [0, 0, PROBE_DOWN])
        arms = snap(left_tcp=tcp, left_open=PROBE_O)
        self.assertEqual(holders(p, arms), ["left"])
        self.assertTrue(monitored(p, arms))
        self.assertEqual(holders(p, snap(left_tcp=tcp, left_open=0.6)), [])  # fingers opening: let go
        self.assertEqual(holders(p, snap(left_tcp=tcp + [0, 0, 0.025], left_open=PROBE_O)), [])  # slid 25 mm
        self.assertEqual(holders(p, snap(left_tcp=tcp + [0, 0, 0.015], left_open=PROBE_O)), ["left"])  # 15 mm
        top = cyl(tcp - [0, 0, PROBE_DOWN - 0.040])  # tip at 56 mm, the top band's top: inside the tank
        self.assertTrue(monitored(top, snap(left_tcp=tcp + [0, 0, 0.040], left_open=PROBE_O)))
        high = cyl(tcp - [0, 0, PROBE_DOWN - 0.044])  # tip at 60 mm: in the deck's opening, above its underside
        self.assertFalse(monitored(high, snap(left_tcp=tcp + [0, 0, 0.044], left_open=PROBE_O)))
        beside = cyl([MX, MYL + 0.04, tcp[2] - PROBE_DOWN])  # right height, outside the column
        self.assertFalse(monitored(beside, snap(left_tcp=[MX, MYL + 0.04, tcp[2]], left_open=PROBE_O)))

    def test_finger_zone_clearance(self):
        c = box(CL_START)
        a = snap(left_tcp=CL_START + [0, 0, UP - CL_START[2]])["left"]  # the demo's end: tips 51 mm over it
        self.assertAlmostEqual(zone_box_gap(a, c.low, c.high), UP - TIPS - c.high[2], places=6)
        self.assertTrue(grippers_clear(snap(left_tcp=CL_START + [0, 0, UP - CL_START[2]],
                                            right_tcp=CR_START + [0, 0, UP - CR_START[2]]), [c]))
        low = snap(left_tcp=[MX, MYL, c.high[2] + TIPS + 0.020])  # tips 20 mm over the cover
        self.assertFalse(grippers_clear(low, [c]))
        shut = snap(left_tcp=[0.2, 0.0, UP], left_open=0.5)
        self.assertFalse(grippers_clear(shut, [c]))  # closed somewhere else: not open

    def test_finger_zone_clearance_uses_real_shapes(self):
        # An open gripper (zone radius 60 mm: 47 mm half gap + 10 mm finger, 20 mm half width) beside a tool turned
        # 45 deg on the pad, 104 mm from its centre across its long side: the tool is 31.6 mm away; its bounding box
        # would be only 1.5 mm away.
        u = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        t = tool(T_PAD, about("z", -45))  # long side along (1, -1)
        tips = T_PAD - 0.104 * u - [0, 0, TH]  # finger tips level with the tool's bottom, on the side away from the tank
        arms = snap(right_tcp=tips + [0, 0, TIPS])
        self.assertAlmostEqual(min_clearance(arms, [t], [SHAPES[3]]), 0.104 - 0.012 - R_OPEN, delta=0.001)
        self.assertLess(min_clearance(arms, [t]), 0.005)  # the bounding box
        self.assertTrue(grippers_clear(arms, [t], shapes=[SHAPES[3]]))
        # and the score uses the real shapes: everything else back, this gripper beside the turned tool is 1.0
        r = Tracker(HZ).result(box(CL_START), box(CR_START), cyl(P_START), t, arms)
        self.assertEqual(r.score, 1.0)
        self.assertGreaterEqual(r.details["min_gripper_clearance_mm"], 30.0)
        # the probe in its holder, a gripper 100 mm from its axis on a diagonal: the round rod is 31.6 mm away (its
        # square bounding box 28.3 mm)
        p = cyl(P_START)
        arms = snap(left_tcp=P_START + 0.100 * u + [0, 0, TIPS])
        self.assertAlmostEqual(min_clearance(arms, [p], [SHAPES[2]]), 0.100 - 0.008 - R_OPEN, delta=0.001)
        self.assertTrue(grippers_clear(arms, [p], shapes=[SHAPES[2]]))
        self.assertLess(min_clearance(arms, [p]), L.CLEAR_GAP)
        # straight above a flat cover the real shape and the box agree
        c = box(CL_START)
        arms = snap(left_tcp=CL_START + [0, 0, UP - CL_START[2]])
        self.assertAlmostEqual(min_clearance(arms, [c], [SHAPES[0]]), min_clearance(arms, [c]), places=6)
        # a gripper whose zone reaches into the tool: negative
        arms = snap(right_tcp=T_PAD + [0, 0, TIPS])
        self.assertLess(min_clearance(arms, [t], [SHAPES[3]]), 0.0)


# ==================================================================================================================
class Tiers(unittest.TestCase):
    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertFalse(t.breach)
        self.assertEqual(t.bands_done, 3)
        self.assertIsNotNone(t.tool_out_step)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertLess(t.ventilated_step, t.band_steps[0])
        self.assertLess(t.test_step, t.first_entry["right"])
        self.assertIsNone(t.first_entry["left"])  # the left gripper never goes in

    def test_not_yet_still(self):
        f = procedure(settle=0)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        tr = Tracker(HZ)
        play(f, tr)
        end = f.frames[-1]
        cl, cr, p, tl = states(end, None)
        arms = snap(end["L"], end["Lo"], end["R"], end["Ro"])
        r = tr.update(cl, cr, p, tool(end["t"], vel=(0.02, 0, 0)), arms)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(end["cl"], spin=(0, 0, 1.0)), cr, p, tl, arms)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(cl, cr, p, tl, arms)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_probe_not_back_is_075(self):
        r, _ = play(procedure(probe_at=P_START + [0.0, -0.06, 0.0]))  # stood on the table beside the holder
        self.assertEqual(r.score, 0.75)

    def test_cover_not_back_is_075(self):
        f = up_to_tool()
        right_takes_tool(f)
        left_returns_probe(f)
        covers_on(f, sides=("left",))  # the right cover stays on its stand
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(right_at=CR_START + [0.015, 0, 0]))  # back on, but 15 mm off centre
        self.assertEqual(r.score, 0.75)

    def test_gripper_not_open_or_not_clear_is_075(self):
        f = procedure(settle=0)
        f.go(20, Ro=0.5).hold(20)  # the right gripper closed again in the air
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        f = procedure(settle=0)
        f.go(20, L=CL_START + [0, 0, 0.040]).hold(20)  # back down: finger tips 17 mm over the cover
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)

    def test_tool_out_not_on_pad_is_05(self):
        f = up_to_tool()
        right_takes_tool(f, to=T_PAD + [0.0, 0.0, 0.03], open_at_end=False)  # still held above the pad
        left_returns_probe(f)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        f = up_to_tool()
        right_takes_tool(f, to=np.array([0.14, -0.06, L.TOP + TH]))  # set on the bare table
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)

    def test_tool_out_step_is_the_first(self):
        f = up_to_tool()
        right_takes_tool(f, to=T_PAD + [0.0, 0.0, 0.10], open_at_end=False)  # held 10 cm over the pad
        f.hold(40)
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertLess(t.tool_out_step, len(f.frames) - 80)  # it latched when the tool first came out

    def test_test_complete_only_is_025(self):
        f = up_to_tool()
        r, t = play(f)
        self.assertEqual(r.score, 0.25)
        self.assertTrue(t.test_complete)
        left_returns_probe(f)
        covers_on(f)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.25)

    def test_nothing_or_covers_only_is_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        r, t = play(covers_off(Film().hold(5)).hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(t.ventilated_step)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        end = f.frames[-1]
        arms = snap(end["L"], end["Lo"], end["R"], end["Ro"])
        objs = states(end, None)
        self.assertEqual(t.result(*objs, arms).score, 1.0)
        self.assertEqual(t.result(*objs, arms).score, 1.0)
        self.assertEqual(t.steps, len(f.frames))


# ==================================================================================================================
class TheTest(unittest.TestCase):
    """The three-depth air test: in order, after both covers are off, each dwell 2 s still in the column."""

    def test_dwells_count_top_first(self):
        r, t = play(up_to_tool())
        self.assertEqual(t.bands_done, 3)
        self.assertEqual(len(t.band_steps), 3)
        self.assertTrue(t.band_steps[0] < t.band_steps[1] < t.band_steps[2])

    def test_short_dwells_do_not_count(self):
        r, t = play(self._film(hold=35))  # 1.75 s at each depth
        self.assertEqual(t.bands_done, 0)
        self.assertEqual(r.score, 0.0)

    def _film(self, tips=DWELL_TIPS, hold=50, xy=(MX, MYL), ventilate=("left", "right")):
        f = Film().hold(5)
        covers_off(f, ventilate)
        left_takes_probe(f)
        left_tests(f, tips, hold, xy)
        return f

    def test_wrong_order_is_not_a_test(self):
        r, t = play(self._film(tips=DWELL_TIPS[::-1]))  # bottom, middle, top
        self.assertEqual(t.bands_done, 1)  # only the top one, done last, counts
        self.assertEqual(r.score, 0.0)

    def test_skipping_a_depth_is_not_a_test(self):
        r, t = play(self._film(tips=(DWELL_TIPS[0], DWELL_TIPS[2])))
        self.assertEqual(t.bands_done, 1)
        self.assertFalse(t.test_complete)

    def test_bottom_first_then_in_order_counts(self):
        r, t = play(self._film(tips=(DWELL_TIPS[2], *DWELL_TIPS)))
        self.assertTrue(t.test_complete)
        self.assertEqual(r.score, 0.25)

    def test_dwell_outside_the_column_does_not_count(self):
        f = Film().hold(5)
        covers_off(f)
        left_takes_probe(f)
        # the same depths, held just behind the left manway's outline (over the deck: only possible in a test
        # film, but it is the column that is scored, not the height alone)
        f.go(20, L=[MX - 0.07, MYL, CARRY_P], p=[MX - 0.07, MYL, CARRY_P - PROBE_DOWN])
        for h in DWELL_TIPS:
            probe_to(f, 30, h, (MX - 0.07, MYL)).hold(50)
        r, t = play(f)
        self.assertEqual(t.bands_done, 0)

    def test_moving_tip_does_not_count(self):
        f = Film().hold(5)
        covers_off(f)
        left_takes_probe(f)
        probe_to(f, 30, DWELL_TIPS[0])
        for _ in range(12):  # bobbing 4 mm up and down inside the top band at 16 mm/s for 2.4 s
            probe_to(f, 5, DWELL_TIPS[0] + 0.004)
            probe_to(f, 5, DWELL_TIPS[0])
        r, t = play(f)
        self.assertEqual(t.bands_done, 0)

    def test_a_break_restarts_the_dwell(self):
        f = Film().hold(5)
        covers_off(f)
        left_takes_probe(f)
        probe_to(f, 30, DWELL_TIPS[0]).hold(30)
        probe_to(f, 4, 0.060).hold(2)  # up out of the band for a moment
        probe_to(f, 4, DWELL_TIPS[0]).hold(30)
        r, t = play(f)
        self.assertEqual(t.bands_done, 0)

    def test_time_in_another_band_does_not_count(self):
        # Fed directly (still, zero velocity): 1.5 s in the middle band, then 0.75 s in the top band. Neither is a
        # 2 s dwell in one band, even though the tip was never moving.
        f = covers_off(Film().hold(5))
        t = Tracker(HZ)
        play(f, t)
        fr = f.frames[-1]
        arms = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
        cl, cr, _, tl = states(fr, None)
        for h, n in ((DWELL_TIPS[1], 30), (DWELL_TIPS[0], 15)):
            for _ in range(n):
                t.update(cl, cr, cyl([MX, MYL, L.TOP + h + L.PROBE_LEN / 2]), tl, arms)
        self.assertEqual(t.bands_done, 0)

    def _fed_dwell(self, tips_and_vel):
        """Covers off, then the probe fed directly, step by step: (tip height, velocity, spin) per step."""
        f = covers_off(Film().hold(5))
        t = Tracker(HZ)
        play(f, t)
        fr = f.frames[-1]
        arms = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
        cl, cr, _, tl = states(fr, None)
        for h, vel, spin in tips_and_vel:
            t.update(cl, cr, cyl([MX, MYL, L.TOP + h + L.PROBE_LEN / 2], vel=vel, spin=spin), tl, arms)
        return t

    def test_still_probe_with_velocity_chatter_counts(self):
        # GPU: a gripped probe that stays put to 0.1 mm reports instantaneous velocities of 10-45 mm/s. The test
        # speed is the tip's path over 0.5 s, so this chatter does not stop the test.
        steps = []
        for h in DWELL_TIPS:
            for i in range(55):
                sgn = 1 if i % 2 else -1
                steps.append((h, (0.03 * sgn, 0.0, 0.02 * sgn), (0.4 * sgn, 0.0, 0.0)))
        t = self._fed_dwell(steps)
        self.assertTrue(t.test_complete)

    def test_slow_creep_counts_fast_creep_does_not(self):
        top_lo = L.BANDS[0][1]
        slow = [(top_lo + 0.001 + 0.003 * i * DT, (0, 0, 0.003), (0, 0, 0)) for i in range(60)]  # 3 mm/s, 3 s
        self.assertEqual(self._fed_dwell(slow).bands_done, 1)
        fast = [(top_lo + 0.0005 + 0.0035 * i * DT, (0, 0, 0.0), (0, 0, 0)) for i in range(60)]  # 3.5 mm/s
        self.assertEqual(self._fed_dwell(fast).bands_done, 1)
        faster = [(top_lo + 0.0005 + 0.006 * i * DT, (0, 0, 0.0), (0, 0, 0)) for i in range(60)]  # 6 mm/s
        self.assertEqual(self._fed_dwell(faster).bands_done, 0)

    def test_speed_needs_half_a_second_of_positions(self):
        # Fewer than 0.5 s of positions: the speed is not known yet, so nothing counts.
        t = Tracker(HZ)
        self.assertEqual(t.path_speed, float("inf"))

    def test_test_before_ventilation_does_not_count(self):
        f = self._film(ventilate=("left",))  # the right cover still on: the space is not ventilated
        r, t = play(f)
        self.assertIsNone(t.ventilated_step)
        self.assertEqual(t.bands_done, 0)
        covers_off(f, ("right",))  # take the right cover off, then test again: now it counts
        probe_to(f, 30, 0.060)
        left_tests(f)
        r, t = play(f)
        self.assertTrue(t.test_complete)

    def test_a_cover_put_back_during_the_test_stops_it(self):
        f = Film().hold(5)
        covers_off(f)
        left_takes_probe(f)
        left_tests(f, DWELL_TIPS[:2])
        f.go(30, cr=CR_START)  # the right cover put back on its manway (the space was ventilated earlier)
        left_tests(f, DWELL_TIPS[2:])
        r, t = play(f)
        self.assertIsNotNone(t.ventilated_step)
        self.assertEqual(t.bands_done, 2)  # the bottom depth, held with a manway shut, is not a test
        f.go(30, cr=CR_STAND)  # off again: now the bottom depth counts
        f.hold(50)
        r, t = play(f)
        self.assertTrue(t.test_complete)

    def test_a_few_mm_either_way_still_counts(self):
        # The demo aims 5 to 6 mm inside each 11 mm band: 4 mm of arm sag (or of slip in the grip) either way
        # still completes the test, and the bottom depth is still a monitoring position.
        for d in (-0.004, 0.004):
            r, t = play(up_to_tool(tips=tuple(h + d for h in DWELL_TIPS)))
            self.assertTrue(t.test_complete, d)
            self.assertTrue(r.details["monitored"], d)
        r, t = play(up_to_tool(tips=tuple(h + 0.007 for h in DWELL_TIPS)))  # 7 mm high: above every band
        self.assertEqual(t.bands_done, 0)


# ==================================================================================================================
class Breaches(unittest.TestCase):
    """Each rule broken once scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key, finish=True):
        r, t = play(f)
        self.assertIn(key, t.breaches)
        if finish:
            left_returns_probe(f)
            covers_on(f)
            f.hold(30)
            r, t = play(f)
        self.assertEqual(r.score, 0.0)
        return t

    def test_entering_before_the_test(self):
        f = Film().hold(5)
        covers_off(f)
        right_takes_tool(f)  # straight in for the tool: no test, no probe
        t = self.assert_zero(f, "entered_before_test", finish=False)
        self.assertIn("entered_unmonitored", t.breaches)
        self.assertEqual(t.breach_objects["entered_before_test"], "right")

    def test_entering_with_a_cover_back_on(self):
        f = up_to_tool()
        f.go(30, cl=CL_START)  # the left cover put back over the manway (the probe still held in the column)
        right_takes_tool(f)
        self.assert_zero(f, "entered_cover_on")

    def test_entering_with_the_probe_lifted_out(self):
        f = up_to_tool()
        probe_to(f, 30, 0.090)  # the probe lifted out of the tank before the right gripper goes in
        right_takes_tool(f)
        self.assert_zero(f, "entered_unmonitored")

    def test_entering_with_the_probe_too_high(self):
        f = up_to_tool()
        probe_to(f, 30, 0.062)  # tip below the deck's top, but in its opening, not under its underside
        right_takes_tool(f)
        self.assert_zero(f, "entered_unmonitored")

    def test_monitoring_from_the_top_depth_is_allowed(self):
        f = up_to_tool()
        probe_to(f, 30, 0.055)  # back up to the top band: the tip is still inside the tank
        right_takes_tool(f)
        r, t = play(f)
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.75)  # tool on the pad, the probe and covers not back yet

    def test_probe_let_go_in_the_tank(self):
        f = up_to_tool()
        f.go(20, Lo=OPEN).go(10, p=[MX, MYL, L.FLOOR_Z + L.PROBE_LEN / 2])  # let go: it slides to the floor
        t = self.assert_zero(f, "probe_let_go_in_tank", finish=False)
        self.assertNotIn("fell", t.breaches)  # it went down slowly; still 0

    def test_probe_let_go_while_the_right_gripper_is_in(self):
        f = up_to_tool()
        f.go(30, R=[MX, MYR, HOVER], Ro=ENTRY_O).go(40, R=T_START + [0, 0, TOOL_UP])
        f.go(20, Lo=OPEN)  # the probe is left standing in the column, unheld, while the right gripper is in
        t = self.assert_zero(f, "probe_let_go_in_tank", finish=False)
        self.assertIn("entered_unmonitored", t.breaches)

    def test_a_dropped_tool(self):
        f = up_to_tool()
        right_takes_tool(f, to=T_PAD + [0, 0, 0.05], open_at_end=False)
        z, v = T_PAD[2] + 0.05, 0.0
        while z > T_PAD[2]:  # let go 5 cm above the pad: free fall
            v += 9.81 * DT
            z = max(T_PAD[2], z - v * DT)
            f.go(1, t=[*T_PAD[:2], z], Ro=OPEN)
        t = self.assert_zero(f, "fell")
        self.assertEqual(t.breach_objects["fell"], "tool")

    def test_a_tipped_probe(self):
        f = up_to_tool()
        right_takes_tool(f)
        f.go(30, L=[MX, MYL, CARRY_P], p=[MX, MYL, CARRY_P - PROBE_DOWN])
        f.go(10, pq=about("x", 35))
        t = self.assert_zero(f, "tipped", finish=False)
        self.assertEqual(t.breach_objects["tipped"], "probe")

    def test_a_cover_off_the_table(self):
        f = Film().hold(5)
        covers_off(f, ("right",))
        f.go(40, cr=[L.STAND_X, -0.70, L.TOP - 0.10])  # pushed off the table's edge
        t = self.assert_zero(f, "off_table", finish=False)
        self.assertEqual(t.breach_objects["off_table"], "cover_right")

    def test_breach_is_permanent(self):
        f = Film().hold(5)
        covers_off(f)
        f.go(30, R=[MX, MYR, HOVER]).go(30, R=[MX, MYR, L.TOP + 0.04]).go(30, R=[MX, MYR, HOVER])  # a quick look
        left_takes_probe(f)
        left_tests(f)
        right_takes_tool(f)
        left_returns_probe(f)
        covers_on(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("entered_before_test", t.breaches)
        self.assertIsNotNone(t.tool_out_step)


# ==================================================================================================================
class Exploits(unittest.TestCase):
    """Strategies that skip the regulated steps earn nothing."""

    def test_one_arm_does_its_half(self):
        # The right arm alone: its cover off, straight in for the tool, onto the pad, cover back.
        f = Film().hold(5)
        covers_off(f, ("right",))
        right_takes_tool(f)
        covers_on(f, sides=("right",))
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("entered_before_test", t.breaches)
        # The left arm alone: its cover off and the full test; never ventilated, so the test never starts
        f = Film().hold(5)
        covers_off(f, ("left",))
        left_takes_probe(f)
        left_tests(f)
        left_returns_probe(f)
        covers_on(f, sides=("left",))
        r, t = play(f.hold(30))
        self.assertEqual((r.score, t.bands_done), (0.0, 0))

    def test_probe_dropped_in_to_monitor(self):
        # Drop the probe into the column, then send the right gripper in while the left goes up: let go, 0
        f = up_to_tool()
        f.go(20, Lo=OPEN).go(10, p=[MX, MYL, L.FLOOR_Z + L.PROBE_LEN / 2]).go(20, L=[MX, MYL, UP])
        right_takes_tool(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("probe_let_go_in_tank", t.breaches)
        self.assertIn("entered_unmonitored", t.breaches)

    def test_rushed_test(self):
        # Dip the probe to each depth for 1 s and go straight in: the test is not complete
        f = Film().hold(5)
        covers_off(f)
        left_takes_probe(f)
        left_tests(f, hold=20)
        right_takes_tool(f)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("entered_before_test", t.breaches)

    def test_entering_the_moment_the_test_completes_is_allowed(self):
        f = Film().hold(5)
        covers_off(f)
        f.go(30, R=[MX, MYR, HOVER], Ro=ENTRY_O)
        left_takes_probe(f)
        for h in DWELL_TIPS[:2]:
            probe_to(f, 30, h).hold(50)
        # the bottom test completes on this hold's last step: 0.5 s for the speed window to fill with still
        # positions (9 steps after the first still one), then 40 steps (2 s)
        probe_to(f, 30, DWELL_TIPS[2]).hold(49)
        r, t = play(f)
        self.assertTrue(t.test_complete)
        f.go(40, R=T_START + [0, 0, TOOL_UP])
        r, t = play(f)
        self.assertFalse(t.breach)


# ==================================================================================================================
class ArmsUnavailable(unittest.TestCase):
    def test_arms_none_rules_off_and_counted(self):
        f = Film().hold(5)
        covers_off(f)
        right_takes_tool(f)  # would be a breach, but the arms cannot be read
        r, t = play(f, arms=False)
        self.assertFalse(t.breach)
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertEqual(r.score, 0.75)  # tool on the pad, covers not back
        self.assertIsNone(r.details["probe_held_by"])

    def test_full_procedure_without_arms(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # 'grippers open and clear' taken as met
        self.assertGreater(t.arms_unavailable, 0)

    def test_non_finite_arms_are_unavailable(self):
        t = Tracker(HZ)
        cl, cr, p, tl = states(Film().frames[0], None)
        rigstate.offline_feed({"left": {"tcp": [np.nan, 0, 0], "tips": [0, 0, 0], "opening": 1.0},
                               "right": {"tcp": READY_R, "tips": READY_R, "opening": 1.0}})
        r = t.update(cl, cr, p, tl, rigstate.arms())
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["arms_available"])
        r = t.update(cl, cr, p, tl, {"left": rigstate.arms()["right"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)


class BadStates(unittest.TestCase):
    def test_nan_inf_zero_quaternion_missing(self):
        f = up_to_tool()
        r, t = play(f)
        self.assertEqual(r.score, 0.25)
        steps, bands = t.steps, list(t.band_steps)
        end = f.frames[-1]
        cl, cr, p, tl = states(end, None)
        arms = snap(end["L"], end["Lo"], end["R"], end["Ro"])
        bad = [
            cyl([np.nan, 0, 0]),
            SimpleNamespace(**{**vars(p), "lin_vel": np.array([np.inf, 0, 0])}),
            SimpleNamespace(**{**vars(p), "quat_xyzw": np.zeros(4)}),
            SimpleNamespace(pos=p.pos),
        ]
        for b in bad:
            r = t.update(cl, cr, b, tl, arms)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
            self.assertFalse(valid(b))
        self.assertEqual(t.bad_steps, 4)
        self.assertEqual(t.steps, steps + 4)
        self.assertEqual(t.band_steps, bands)  # nothing on the record changed
        self.assertFalse(t.breach)
        r = t.update(cl, cr, p, tl, arms)
        self.assertEqual(r.score, 0.25)  # back to normal on the next good step


# ==================================================================================================================
class Resets(unittest.TestCase):
    def breached(self):
        f = Film().hold(5)
        covers_off(f)
        right_takes_tool(f)
        r, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        start = Film().frames[0]
        r = t.update(*states(start, None), snap())  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        fr = dict(Film().frames[0])
        fr["t"] = T_START + [0.02, -0.01, 0]
        fr["p"] = P_START + [-0.01, 0.01, 0]
        t.update(*states(fr, None), snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        f = Film(t=T_PAD, cl=CL_STAND, cr=CR_STAND)
        f.go(60, t=[*T_PAD[:2], HOVER]).go(60, t=[*T_START[:2], HOVER]).go(60, t=T_START)
        f.go(60, cl=[*CL_STAND[:2], HOVER]).go(60, cl=[*CL_START[:2], HOVER]).go(60, cl=CL_START)
        f.go(60, cr=[*CR_STAND[:2], HOVER]).go(60, cr=[*CR_START[:2], HOVER]).go(60, cr=CR_START).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        fr = dict(Film().frames[0])
        fr["t"] = np.array([0.15, 0.0, L.TOP + TH])  # knocked far, not to the start
        t.update(*states(fr, None), snap())
        self.assertTrue(t.breach)

    def test_a_falling_cover_is_not_a_teleport(self):
        t = Tracker(HZ)
        play(Film().hold(5), t)
        t._flag("fell")
        z, v = CL_START[2] + 0.3, 0.0
        while z > CL_START[2]:
            v += 9.81 * DT
            z = max(CL_START[2], z - v * DT)
            fr = dict(Film().frames[0])
            fr["cl"] = np.array([*CL_START[:2], z])
            cl, cr, p, tl = states(fr, None)
            cl.lin_vel = np.array([0, 0, -v if z > CL_START[2] else 0.0])
            t.update(cl, cr, p, tl, snap())
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


# ==================================================================================================================
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

    def objects(self, fr=None):
        cl, cr, p, tl = states(fr or Film().frames[0], None)
        return {"cover_left": cl, "cover_right": cr, "probe": p, "tool": tl}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = sorted(o.name for o in self.task.OBJECTS if o.mass is not None)
        self.assertEqual(loose, ["cover_left", "cover_right", "probe", "tool"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)

    def test_deck_leaves_the_two_manways_open(self):
        deck = [o for o in self.task.OBJECTS if o.name.startswith("deck_")]
        area = sum(o.size[0] * o.size[1] for o in deck)
        full = (L.TANK_X1 - L.TANK_X0) * (L.TANK_Y1 - L.TANK_Y0)
        self.assertAlmostEqual(area, full - 2 * 4 * L.MANWAY_HALF[0] * L.MANWAY_HALF[1], places=9)
        for side in ("left", "right"):  # nothing of the deck over either opening
            x0, x1 = L.MANWAY_X - L.MANWAY_HALF[0], L.MANWAY_X + L.MANWAY_HALF[0]
            y0, y1 = L.MANWAY_Y[side] - L.MANWAY_HALF[1], L.MANWAY_Y[side] + L.MANWAY_HALF[1]
            for o in deck:
                ox0, ox1 = o.pos[0] - o.size[0] / 2, o.pos[0] + o.size[0] / 2
                oy0, oy1 = o.pos[1] - o.size[1] / 2, o.pos[1] + o.size[1] / 2
                self.assertFalse(ox0 < x1 - 1e-9 and x0 < ox1 - 1e-9 and oy0 < y1 - 1e-9 and y0 < oy1 - 1e-9,
                                 f"{o.name} covers part of the {side} manway")
        # each cover is 10 mm larger than its opening on every side
        self.assertAlmostEqual(L.COVER_SIZE[0] / 2 - L.MANWAY_HALF[0], 0.010)
        self.assertAlmostEqual(L.COVER_SIZE[1] / 2 - L.MANWAY_HALF[1], 0.010)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=[MX, MYR, L.TOP + 0.04])  # the right gripper in the tank at the start
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
        # one arm missing from the snapshot: no crash in success() or score(), counted as unavailable
        n = tr.arms_unavailable
        rigstate.offline_feed({"left": {"tcp": READY_L, "tips": READY_L, "opening": 1.0}})
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertEqual(tr.arms_unavailable, n + 1)

    def test_trace_survives_a_bad_path(self):
        import os
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
        import os
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})

        def start_of(wps, label):
            i = next(i for i, w in enumerate(wps) if label in w.label)
            return sum(w.duration for w in wps[:i]), i

        # the right gripper starts down into the manway only after the last test waypoint has ended
        r_in, _ = start_of(plan["right"], "down into the manway")
        self.assertGreaterEqual(r_in, self.task.TEST_DONE_S + 0.5 - 1e-9)
        # the left lifts the probe out only after the right gripper is up out of the tank with the tool
        l_out, _ = start_of(plan["left"], "lift the probe out")
        self.assertGreaterEqual(l_out, self.task.TOOL_OUT_S + 0.5 - 1e-9)
        # three tests of 3 s each, top first
        tests = [w for w in plan["left"] if w.label.startswith("test the")]
        self.assertEqual([w.label for w in tests], ["test the top depth", "test the middle depth",
                                                    "test the bottom depth"])
        self.assertTrue(all(w.duration >= L.DWELL_S + 1.0 for w in tests))
        # every open/close of the fingers takes at least 1.5 s
        for arm, wps in plan.items():
            prev = self.task.OPEN
            for w in wps:
                if abs(w.finger - prev) > 0.005:
                    self.assertGreaterEqual(w.duration, 1.5, f"{arm}: {w.label}")
                prev = w.finger
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
