"""Offline tests for layout.py, parts.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import parts as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.HANDS_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
SEAT = (L.UX, L.UY, L.TOP + L.SEAT_Z)
STORE = (L.TUBE_STORE[0], L.TUBE_STORE[1], L.TOP + L.TUBE_ROTOR_T)
STRIP_ROTOR_START = (L.STRIP_ROTOR_START[0], L.STRIP_ROTOR_START[1], L.TOP + L.STRIP_ROTOR_START[2])


def yq(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quat_about(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return (*(a * math.sin(h)), math.cos(h))


def unit_state(lid=0.0, tab=0.0, plungers=0.0, pos=(L.UX, L.UY, L.TOP), quat=(0.0, 0.0, 0.0, 1.0)):
    return mk.state_from_poses(T.UNIT_PART, (*pos, *quat), source="test",
                               joints={"lid": lid, "tab": tab, "plunger_p": plungers, "plunger_n": plungers})


def part(name, pos, yaw=0.0, quat=None):
    return mk.state_from_poses(T.SPECS[name], (*pos, *(quat or yq(yaw))), source="test")


def in_slot(slot, rotor=SEAT, rotor_yaw=0.0):
    c, s = math.cos(rotor_yaw), math.sin(rotor_yaw)
    y = L.SLOT_Y[slot]
    return (rotor[0] - s * y, rotor[1] + c * y, rotor[2] + L.SEATED_Z)


def scene(lid=0.0, tab=0.0, plungers=0.0, rotor=SEAT, rotor_yaw=0.0, slots=(1, 2), tube=STORE, unit_pos=None,
          unit_quat=None, strip_pos=None):
    """Defaults: the 1.0 end state (strip rotor clicked on, strips in slots 1 and 2, tube rotor on the bench, the
    lid at 0). ``slots`` None for a strip leaves it in the rack."""
    st = {S.UNIT: unit_state(lid, tab, plungers, pos=unit_pos or (L.UX, L.UY, L.TOP),
                             quat=unit_quat or (0.0, 0.0, 0.0, 1.0)),
          S.TUBE_ROTOR: part(S.TUBE_ROTOR, tube),
          S.STRIP_ROTOR: part(S.STRIP_ROTOR, rotor, rotor_yaw),
          S.RACK: part(S.RACK, (L.RACK_X, L.RACK_Y, L.TOP))}
    for i, name in enumerate(S.STRIPS):
        slot = slots[i] if i < len(slots) else None
        if strip_pos and name in strip_pos:
            st[name] = part(name, strip_pos[name])
        elif slot is None:
            x, y, z = L.STRIP_STARTS[i]
            st[name] = part(name, (x, y, L.TOP + z))
        else:
            st[name] = part(name, in_slot(slot, rotor, rotor_yaw), rotor_yaw)
    return st


def start_scene():
    return scene(rotor=STRIP_ROTOR_START, rotor_yaw=L.STRIP_ROTOR_START_YAW, slots=(None, None),
                 tube=SEAT)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.12, 0.40, L.TOP + 0.20)), "right": arm((0.12, -0.40, L.TOP + 0.20))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def opened(tracker=None):
    """A tracker that has seen the tab pressed and the lid open to its stop (the start of every real trial)."""
    tr = tracker or S.Tracker()
    run(start_scene(), steps=3, tracker=tr)
    run(scene(tab=0.003, rotor=STRIP_ROTOR_START, rotor_yaw=L.STRIP_ROTOR_START_YAW, slots=(None, None), tube=SEAT),
        steps=3, tracker=tr)
    run(scene(lid=L.LID_UPPER, rotor=STRIP_ROTOR_START, rotor_yaw=L.STRIP_ROTOR_START_YAW, slots=(None, None),
              tube=SEAT), steps=5, tracker=tr)
    return tr


def finish(states, arms=FAR):
    """Open the lid, then hold ``states`` (with whatever lid angle it has) for the hands-off window and more."""
    tr = opened()
    return run(states, arms=arms, tracker=tr), tr


class Geometry(unittest.TestCase):
    def test_published_sizes(self):
        self.assertAlmostEqual(2 * L.HX, 0.148)
        self.assertAlmostEqual(2 * L.HY, 0.174)
        self.assertAlmostEqual(L.LID_Z1, 0.122)
        self.assertAlmostEqual(L.UNIT_MASS, 1.1)
        lo, hi = T.UNIT_PART.local_bounds()
        self.assertLess(abs((hi[0] - lo[0]) - 0.148), 0.003)
        self.assertLess(abs((hi[1] - lo[1]) - 0.174), 0.003)
        self.assertLess(abs((hi[2] - lo[2]) - 0.122), 0.003)

    def test_strip_rotor_and_strips(self):
        self.assertLess(abs(math.hypot(L.STRIP_ROTOR_X, L.STRIP_ROTOR_Y) - 0.110), 0.001)  # 110 mm across corners
        self.assertGreaterEqual(L.SLOT_LX, L.STRIP_L + 0.002)
        self.assertGreater(L.SLOT_W, L.STRIP_W)
        self.assertAlmostEqual(L.PITCH * 8, L.STRIP_L)
        for a, b in L.SLOT_PAIRS:
            self.assertAlmostEqual(L.SLOT_Y[a], -L.SLOT_Y[b])
        self.assertLess(L.STRIP_MASS, 0.002)

    def test_key_and_snap(self):
        self.assertGreater(L.BORE_HALF, L.SHAFT_R)  # the bore slides over the shaft
        self.assertGreater(L.KEYWAY_HY, L.KEY_HY)  # the pin fits the keyway ...
        self.assertGreaterEqual(L.KEYWAY_X, L.KEY_HX)  # ... along its length
        self.assertGreater(L.KEYWAY_Z, L.KEY_Z1 - L.SEAT_Z)  # and seated, the keyway clears its top
        play = math.degrees(math.atan2(L.KEYWAY_HY - L.KEY_HY, L.KEY_HX))
        self.assertGreater(play, L.KEY_TOL_DEG)  # the pin's width passes within the stated tolerance
        self.assertGreater(L.NOSE_TIP_R, L.BORE_HALF)  # the noses reach into the bore ...
        self.assertLess(L.NOSE_TIP_R - L.PLUNGER_TRAVEL, L.BORE_HALF)  # ... and can be pushed back out of it
        push = L.PLUNGER_SPRING * (L.NOSE_TIP_R - L.BORE_HALF)  # N each, pushed back
        self.assertGreater(push, 0.5)
        self.assertLess(push, 2.0)
        self.assertAlmostEqual(L.SHAFT_TOP, L.SEAT_Z + L.HUB_H)  # seated, the hub's top is flush with the shaft

    def test_lid_torques(self):
        freed_shut = L.lid_spring_torque(0.0, False) + L.lid_gravity_torque(0.0)
        freed_stop = L.lid_spring_torque(L.LID_UPPER, False) + L.lid_gravity_torque(L.LID_UPPER)
        latched = L.lid_spring_torque(0.0, True) + L.lid_gravity_torque(0.0)
        self.assertGreater(freed_shut, 0.15)  # freed, it pops up ...
        self.assertGreater(freed_stop, 0.0)  # ... and stays on its stop
        self.assertLess(latched, -0.3)  # latched, held shut
        self.assertLess(freed_shut / T.CLOSE_RHO, 5.0)  # the push to shut it, 90 mm from the hinge: under 5 N

    def test_open_lid_clears_the_left_hand_and_the_carry(self):
        m = T.OfflineModel()
        q = [L.LID_UPPER]
        # the open fingers over the shaft (yaw 0: closing along y, 36 mm) at every height the left hand uses
        for h in np.linspace(L.SEAT_Z, T.LIFT_H, 30):
            pts = T.pad_points((L.UX, L.UY, L.TOP + h - T.PAD_TIP), -T.HUB_OPEN / -T.OPEN, (0.0, 1.0, 0.0))
            self.assertFalse(m._pressing(pts, q)[0], h)
        # the carry: in front of the open lid, then over the shaft
        for x, y in ((L.UX, L.STRIP_ROTOR_START[1]), (L.UX, 0.09), (L.UX, 0.0)):
            pts = T.pad_points((x, y, L.TOP + T.LIFT_H - T.PAD_TIP), 0.25, (1.0, 0.0, 0.0))
            self.assertFalse(m._pressing(pts, q)[0], (x, y))

    def test_tube_rotor_disc_fits_the_chamber(self):
        """GPU run 2026-11-19: the disc was four turned 96 mm squares, a 16-pointed star reaching 67.9 mm, 6 mm into
        the 62 mm chamber wall all round; the rotor was jammed and the hand could not pull it off. The disc must be a
        16-gon 96 mm across its flats, clear of the chamber wall and of the collar."""
        corners = []
        for sh in P.tube_rotor_disc():
            half = np.asarray(sh.size) / 2
            for sx in (-1, 1):
                for sy in (-1, 1):
                    corners.append(sh.rotation() @ (half * (sx, sy, 1)) + np.asarray(sh.center))
        r = np.hypot(*np.asarray(corners)[:, :2].T)
        self.assertLessEqual(r.max(), L.TUBE_ROTOR_R / math.cos(math.pi / 16) + 1e-6)
        self.assertLessEqual(r.max(), L.BOWL_R - 0.010)
        self.assertGreaterEqual(r.min(), L.COLLAR_R + 0.002)
        lo, hi = T.TUBE_ROTOR_PART.local_bounds()
        self.assertLess(hi[0] - lo[0], 2 * L.BOWL_R - 0.020)

    def test_hands_over_the_open_chamber_stay_clear_of_the_lid(self):
        """GPU run 2026-11-19: the left hand brought the strip rotor over the shaft with its wrist at yaw -90 and the
        gripper housing (102 mm along the closing axis, from 70 mm over the TCP up) ran into the open lid, which
        knocked the unit over. Over the unit, with the lid open, the left hand works at yaw 0 (housing 65 mm across
        x), and whenever the right hand's fingers close along x its far finger carriage (gap/2 + 24 mm out, 57 to 81
        mm over the TCP) stays 2 mm or more in front of the open lid's underside and its side skirts. The full
        capsule check is test_housings_keep_10_mm_from_the_lid_unit_and_each_other."""
        q = L.LID_UPPER

        def plane(w, rho1):
            a = np.array(L.lid_point(0.004, q, w))
            b = np.array(L.lid_point(rho1, q, w))
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([-d[1], d[0]])
            return a, b, d, (-n if n[0] > 0 else n)

        def ahead(pl, x, z):
            a, b, d, n = pl
            p = np.array([x, z]) - a
            if p @ d > np.linalg.norm(b - a):
                return float(np.linalg.norm(np.array([x, z]) - b))
            return float(p @ n)

        plate = plane(L.LID_Z0 - L.HINGE_Z, L.HINGE_X - L.TAB_U0)
        skirt = plane(L.SKIRT_Z0 - L.HINGE_Z, L.HINGE_X - L.LID_X0)
        p = T.plans()
        t_open = T._until(p["right"], "back off forward and up")
        t_shut = T._until(p["right"], "beside the unit")
        for side in ("left", "right"):
            t = 0.0
            prev = p[side][0]
            for w in p[side]:
                t0, t = t, t + w.duration
                if t0 < t_open or t > t_shut:
                    prev = w
                    continue
                for s in np.linspace(0.0, 1.0, 11):
                    x, y, z = (np.asarray(prev.pos) + s * (np.asarray(w.pos) - np.asarray(prev.pos)))
                    hx, hy, hz = x - L.UX, abs(y - L.UY), z - L.TOP
                    if abs(hx) > L.HX + 0.03 or hy > L.HY + 0.03:
                        continue
                    if side == "left":
                        self.assertAlmostEqual(prev.yaw, 0.0, msg=w.label)
                        self.assertAlmostEqual(w.yaw, 0.0, msg=w.label)
                        continue
                    gap = max(-2 * prev.finger, -2 * w.finger, L.STRIP_W if T.STRIP_PINCH in (prev.finger,
                                                                                             w.finger) else 0.0)
                    if abs(math.cos(w.yaw)) > 0.5:  # closing along y: the carriage is not reaching toward the lid
                        continue
                    far = hx + gap / 2 + 0.024
                    planes = [plate] + ([skirt] if hy + 0.0135 >= L.HY - L.SKIRT_T - 0.005 else [])
                    for pl in planes:
                        for up in (0.057, 0.069, 0.081):
                            self.assertGreaterEqual(ahead(pl, far, hz + up), 0.002, (w.label, round(hx, 3),
                                                                                     round(hz, 3)))
                prev = w

    def test_strip_hand_never_goes_over_the_hub(self):
        """Offline, 2026-11-19: backing 25 mm off a placed strip brought the right hand's far pad within 1.5 mm of
        the strip rotor's hub, 9 mm under its top, and the model counted it as pressing an UNpressed rotor home (the
        nopress counterfactual scored 1.0). Holding a strip across its cap row or letting go, neither pad (8 mm
        thick along the closing axis, 27 mm wide across it) may come within 3 mm of the hub's outline below 3 mm over
        the top of a hub resting on its noses (not pressed)."""
        hub_top = L.SEAT_Z + L.NOSE_REST + L.HUB_H + 0.003
        edge = L.HUB_HALF + 0.003
        plan = T.plans()["right"]
        prev = plan[0]
        checked = 0
        for w in plan[1:]:
            fingers = (prev.finger, w.finger)
            if T.STRIP_PINCH not in fingers and T.STRIP_OPEN not in fingers:
                prev = w
                continue
            gap = max(-2 * prev.finger, -2 * w.finger, L.STRIP_W if T.STRIP_PINCH in fingers else 0.0)
            for s in np.linspace(0.0, 1.0, 21):
                x, y, z = np.asarray(prev.pos) + s * (np.asarray(w.pos) - np.asarray(prev.pos))
                yaw = prev.yaw + s * (w.yaw - prev.yaw)
                tip = z - L.TOP - T.PAD_TIP
                if tip >= hub_top:
                    continue
                c = np.array([math.sin(yaw), -math.cos(yaw)])  # the closing axis
                n = np.array([-c[1], c[0]])
                for sign in (-1.0, 1.0):
                    centre = np.array([x - L.UX, y - L.UY]) + sign * (gap / 2 + T.PAD_T / 2) * c
                    reach = np.abs(c) * T.PAD_T / 2 + np.abs(n) * T.PAD_HW  # the pad's half extent in x and y
                    clear = np.any(np.abs(centre) - reach > edge)
                    checked += 1
                    self.assertTrue(clear, (w.label, np.round(centre, 4).tolist(), round(tip, 4)))
            prev = w
        self.assertGreater(checked, 100)

    def test_tab_block_stays_clear_of_the_lid(self):
        front = L.UX + L.LID_X0
        self.assertLessEqual(T.TAB_X + T.BLOCK_HALF, front - 0.002)
        self.assertGreaterEqual(T.TAB_X - T.BLOCK_HALF, L.UX + L.TAB_U0 - 0.008)

    def test_housings_keep_10_mm_from_the_lid_unit_and_each_other(self):
        """GPU run 2026-11-19 and the first fix round: the right gripper's finger carriage went 22.8 mm into the open
        lid's plate over the strips, and 14 to 17 mm into it as the lid sprang up and as it was swung shut. Over the
        whole demo, each arm's gripper housing (the yamkin capsules of the gripper link and of each finger's
        carriage; the pads, which are meant to touch, are not counted) keeps 10 mm or more from the lid and its tab
        at the angle the offline model gives the lid at that step, from the unit's body (the shaft, which the hands
        work on, left out), and from the other arm's housing."""
        try:
            import yamkin as K
        except ModuleNotFoundError:
            self.skipTest("tools/offline/yamkin.py is not on the path")
        hinge = np.array([L.HINGE_X, 0.0, L.HINGE_Z])
        unit = np.array([L.UX, L.UY, L.TOP])

        def boxes(shapes):
            out = []
            for sh in shapes:
                if sh.kind == "box":
                    half = np.asarray(sh.size, dtype=float) / 2
                else:  # a cylinder as its bounding box (more than its true size)
                    half = np.full(3, float(sh.radius))
                    half["XYZ".index(sh.axis)] = sh.height / 2
                out.append((np.asarray(sh.center, dtype=float), sh.rotation(), half))
            return out

        def pack(bs, rot=np.eye(3)):
            return (np.array([unit + hinge + rot @ (c - hinge) for c, _, _ in bs]),
                    np.array([rot @ r for _, r, _ in bs]), np.array([h for _, _, h in bs]))

        def clearance(caps, packed):
            c, r, h = packed
            pts = np.array([a + s * (b - a) for _, a, b, _ in caps for s in np.linspace(0.0, 1.0, 15)])
            rad = np.repeat([cap[3] for cap in caps], 15)
            loc = np.einsum("nji,mnj->mni", r, pts[:, None, :] - c[None])
            q = np.abs(loc) - h[None]
            sd = np.linalg.norm(np.maximum(q, 0.0), axis=2) + np.minimum(q.max(axis=2), 0.0)
            return float((sd - rad[:, None]).min())

        def housing(caps):
            seen, out = set(), []
            for cap in caps:
                if cap[0] == "gripper" or (cap[0] in K.FINGER_LINKS and cap[0] not in seen):
                    out.append(cap)
                seen.add(cap[0])
            return out

        lid = boxes(P.lid_shapes() + P.tab_shapes())
        body = [b for b in boxes(P.body_shapes())
                if np.linalg.norm(b[0][:2]) >= 0.02 or b[2][:2].max() >= 0.02]  # all but the shaft, key and snap
        body = (np.array([unit + c for c, _, _ in body]), np.array([r for _, r, _ in body]),
                np.array([h for _, _, h in body]))
        bases = {a: np.asarray(K.BASES["station"][a], dtype=float) for a in ("left", "right")}
        plans = {a: T.ArmPlan(list(w), bases[a]) for a, w in T.plans().items()}
        model = T.OfflineModel(T._closing_axes)
        worst, top = {}, 0.0
        n = int(max(ap.duration for ap in plans.values()) * HZ)
        for k in range(n):
            t = (k + 1) / HZ
            snap, caps, labels = {}, {}, {}
            for a, ap in plans.items():
                act, labels[a] = ap.action7(t)
                q, finger = act[:6], float(act[6]) * K.FINGER_OPEN
                pos, rot = K.tcp_pose(q, bases[a])
                snap[a] = rigstate.Arm(pos, K.pad_tip(pos, rot), float(np.clip(finger / K.FINGER_OPEN, 0.0, 1.0)))
                caps[a] = housing(K.arm().capsules(q, bases[a], finger))
            model.step(snap)
            top = max(top, model.lid)
            if k % 2:
                continue
            c, s_ = math.cos(model.lid), math.sin(model.lid)
            lid_now = pack(lid, np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]))
            for a in plans:
                for what, d in (("lid", clearance(caps[a], lid_now)), ("unit", clearance(caps[a], body))):
                    if d < worst.get((a, what), (math.inf,))[0]:
                        worst[(a, what)] = (d, round(t, 2), labels[a], round(math.degrees(model.lid), 1))
            d = K.capsule_clearance(caps["left"], caps["right"])[0]
            if d < worst.get(("both", "arms"), (math.inf,))[0]:
                worst[("both", "arms")] = (d, round(t, 2), labels["left"] + " / " + labels["right"], None)
        self.assertEqual(len(worst), 5)
        self.assertAlmostEqual(top, L.LID_UPPER, places=2)  # the lid was open over the chamber work ...
        self.assertLess(model.lid, math.radians(1.0))  # ... and shut at the end
        for key, (d, t, label, q_deg) in worst.items():
            self.assertGreaterEqual(d, 0.010, (key, round(d * 1000, 1), t, label, q_deg))

    def test_start_layout_does_not_score(self):
        res = run(start_scene())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)


class Tiers(unittest.TestCase):
    def test_tier_table(self):
        self.assertEqual(S.tier(False, True, True, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, True, False, True, True), 0.75)
        self.assertEqual(S.tier(False, True, False, False, True, True), 0.5)
        self.assertEqual(S.tier(False, False, False, False, True, True), 0.25)
        self.assertEqual(S.tier(False, False, False, False, True, False), 0.0)
        self.assertEqual(S.tier(False, True, True, False, False, True), 0.0)  # tube rotor not stored
        for seat, bal, store in ((False, True, True), (True, False, True), (True, True, False)):
            self.assertEqual(S.tier(False, seat, bal, True, store, True), 0.0)  # latched on a bad load
        self.assertEqual(S.tier(True, True, True, True, True, True), 0.0)

    def test_full_sequence_scores_one(self):
        res, tr = finish(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["latched"])
        self.assertEqual(tr.catch.latches, 1)

    def test_lid_left_open(self):
        self.assertEqual(finish(scene(lid=L.LID_UPPER))[0].score, 0.75)
        self.assertEqual(finish(scene(lid=L.LID_UPPER, slots=(1, None)))[0].score, 0.5)
        loose = scene(lid=L.LID_UPPER, rotor=(SEAT[0], SEAT[1], SEAT[2] + L.NOSE_REST), slots=(None, None),
                      plungers=0.0)
        self.assertEqual(finish(loose)[0].score, 0.25)
        on_pin = scene(lid=L.LID_UPPER, rotor=(SEAT[0], SEAT[1], SEAT[2] + L.PIN_REST), rotor_yaw=math.pi / 2,
                       slots=(None, None))
        self.assertEqual(finish(on_pin)[0].score, 0.25)

    def test_either_opposite_pair_and_either_strip(self):
        for slots in ((1, 2), (2, 1), (0, 3), (3, 0)):
            self.assertEqual(finish(scene(slots=slots))[0].score, 1.0, slots)
        for slots in ((0, 1), (1, 3), (0, 2)):
            self.assertEqual(finish(scene(slots=slots))[0].score, 0.0, slots)

    def test_rotor_turned_half_a_turn_is_still_seated(self):
        self.assertEqual(finish(scene(rotor_yaw=math.pi))[0].score, 1.0)


class LatchMachine(unittest.TestCase):
    def test_states(self):
        c = L.Latch()
        self.assertTrue(c.latched)
        c.step(0.0, 0.0019)
        self.assertTrue(c.latched)
        c.step(0.0, 0.002)
        self.assertFalse(c.latched)
        self.assertEqual(c.frees, 1)
        c.step(0.0, 0.0)  # not opened yet: shut again with the tab up does not catch
        self.assertFalse(c.latched)
        c.step(math.radians(10.0), 0.0)
        c.step(math.radians(0.5), 0.0015)  # the tab still down: no catch
        self.assertFalse(c.latched)
        c.step(math.radians(0.5), 0.0)
        self.assertTrue(c.latched)
        self.assertFalse(c.reopened)
        c.step(0.0, 0.003)
        self.assertTrue(c.reopened)
        c.reset()
        self.assertTrue(c.latched)
        self.assertEqual((c.frees, c.latches, c.reopened), (0, 0, False))

    def test_targets(self):
        c = L.Latch()
        self.assertEqual(c.target(), L.LID_LATCH_TARGET)
        c.step(0.0, 0.003)
        self.assertEqual(c.target(), L.LID_OPEN_TARGET)

    def test_nan_keeps_state(self):
        c = L.Latch()
        c.step(float("nan"), 0.003)
        self.assertTrue(c.latched)

    def test_reopened_after_latching_scores_zero(self):
        tr = opened()
        run(scene(), tracker=tr)
        run(scene(tab=0.003), steps=2, tracker=tr)
        res = run(scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["reopened"])

    def test_forced_lid_scores_zero(self):
        tr = opened()
        run(scene(), tracker=tr)
        run(scene(lid=math.radians(4.0)), steps=2, tracker=tr)  # pried up while caught
        res = run(scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["lid_forced"])

    def test_latched_start_state_is_not_an_opening(self):
        tr = S.Tracker()
        run(start_scene(), tracker=tr)
        self.assertFalse(tr.catch.opened)
        self.assertEqual(tr.catch.latches, 0)


class HandsOff(unittest.TestCase):
    def test_hand_resting_on_the_lid(self):
        on_lid = {"left": FAR["left"], "right": arm((L.UX, -0.05, L.TOP + L.LID_Z1 + 0.013), 0.0)}
        res, _ = finish(scene(), arms=on_lid)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["hands_off_1s"])

    def test_hand_25_mm_over_the_lid_is_too_close(self):
        near = {"left": FAR["left"], "right": arm((L.UX, -0.05, L.TOP + L.LID_Z1 + 0.025 + 0.013), 0.0)}
        self.assertEqual(finish(scene(), arms=near)[0].score, 0.0)

    def test_hands_must_stay_off_the_whole_second(self):
        tr = opened()
        run(scene(), tracker=tr, steps=WIN + 2)
        on_lid = {"left": FAR["left"], "right": arm((L.UX, -0.05, L.TOP + L.LID_Z1 + 0.013), 0.0)}
        run(scene(), arms=on_lid, tracker=tr, steps=1)
        res = run(scene(), tracker=tr, steps=WIN - 1)
        self.assertEqual(res.score, 0.0)
        res = run(scene(), tracker=tr, steps=2)
        self.assertEqual(res.score, 1.0)


class BadStates(unittest.TestCase):
    def test_nan_and_missing_states(self):
        st = scene()
        st[S.STRIP_ROTOR] = part(S.STRIP_ROTOR, (float("nan"), 0.0, L.TOP))
        self.assertIsNone(S.read(st))
        st = scene()
        del st[S.STRIPS[0]]
        self.assertIsNone(S.read(st))
        self.assertIsNone(S.read(None))
        tr = S.Tracker()
        self.assertEqual(tr.update(st, FAR).score, 0.0)

    def test_a_bad_step_restarts_the_window(self):
        tr = opened()
        run(scene(), tracker=tr)
        tr.update({}, FAR)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)

    def test_unreadable_arms_never_hold_1(self):
        res, _ = finish(scene(), arms=None)
        self.assertEqual(res.score, 0.0)

    def test_teleport_reset_clears_the_latches(self):
        tr = opened()
        run(scene(unit_pos=(L.UX + 0.03, L.UY, L.TOP)), tracker=tr, steps=2)
        self.assertIsNotNone(tr.slid)
        tr.update(start_scene(), FAR)
        self.assertIsNone(tr.slid)
        self.assertEqual(tr.resets, 1)


class Exploits(unittest.TestCase):
    def test_latched_on_the_loose_rotor(self):
        loose = scene(rotor=(SEAT[0], SEAT[1], SEAT[2] + L.NOSE_REST))
        self.assertEqual(finish(loose)[0].score, 0.0)

    def test_rotor_down_but_not_clicked(self):
        self.assertEqual(finish(scene(plungers=-0.0006))[0].score, 0.0)  # a nose still pushed back
        self.assertEqual(finish(scene(lid=L.LID_UPPER, plungers=-0.0006))[0].score, 0.25)

    def test_one_strip_latched(self):
        self.assertEqual(finish(scene(slots=(1, None)))[0].score, 0.0)

    def test_strip_on_the_rotor_top_not_in_its_slot(self):
        lifted = {S.STRIPS[0]: (SEAT[0], SEAT[1] + L.SLOT_Y[1], SEAT[2] + 0.001)}
        self.assertEqual(finish(scene(strip_pos=lifted))[0].score, 0.0)

    def test_tube_rotor_not_stored(self):
        on_unit = (L.UX, L.UY + 0.04, L.TOP + L.LID_Z1 + L.TUBE_ROTOR_T)
        self.assertEqual(finish(scene(tube=on_unit))[0].score, 0.0)
        in_chamber = (L.UX - 0.03, L.UY, L.TOP + L.BOWL_FLOOR + L.TUBE_ROTOR_T)
        self.assertEqual(finish(scene(tube=in_chamber))[0].score, 0.0)
        against = (L.UX - L.HX - L.TUBE_ROTOR_R - 0.001, L.UY, L.TOP + L.TUBE_ROTOR_T)
        self.assertEqual(finish(scene(tube=against))[0].score, 0.0)
        beside = (L.UX - L.HX - L.TUBE_ROTOR_R * 1.5 - 0.01, L.UY, L.TOP + L.TUBE_ROTOR_T)
        self.assertEqual(finish(scene(tube=beside))[0].score, 1.0)

    def test_unit_slid_harmless_tipped_zero(self):
        for dx, want in ((0.014, 1.0), (0.016, 1.0), (0.060, 1.0)):
            tr = opened()
            run(scene(lid=L.LID_UPPER, unit_pos=(L.UX + dx, L.UY, L.TOP)), tracker=tr, steps=2)  # slid, then back
            self.assertEqual(run(scene(), tracker=tr).score, want, dx)
        self.assertEqual(finish(scene(unit_quat=quat_about((0, 1, 0), 11.0)))[0].score, 0.0)
        tr = opened()
        run(scene(lid=L.LID_UPPER, unit_quat=quat_about((0, 1, 0), 11.0)), tracker=tr, steps=2)  # tipped, then back
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_dropped_off_the_table(self):
        tr = opened()
        st = scene(lid=L.LID_UPPER)
        st[S.STRIPS[0]] = part(S.STRIPS[0], (0.2, 0.0, L.TOP - 0.2))
        run(st, tracker=tr, steps=2)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_nothing_done_scores_zero(self):
        self.assertEqual(run(start_scene()).score, 0.0)
        self.assertEqual(finish(start_scene())[0].score, 0.0)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    @staticmethod
    def axes(left=(0.0, 1.0, 0.0), right=(1.0, 0.0, 0.0)):
        return lambda step: {"left": left, "right": right}

    @staticmethod
    def block_on_tab(depth):
        tcp = (T.TAB_X, L.UY, L.TOP + L.LID_Z1 - depth + T.PAD_TIP)
        return {"left": FAR["left"], "right": arm(tcp, 0.0)}

    def test_tab_press_frees_and_the_lid_pops_up_after_the_hand_slides_off(self):
        m = T.OfflineModel(self.axes())
        for _ in range(5):
            m.step(self.block_on_tab(T.PRESS_OVER))
        self.assertFalse(m.catch.latched)
        self.assertEqual(m.lid, 0.0)  # held down by the block on the tab
        n = 0
        while m.lid < L.LID_UPPER - 1e-9 and n < 60:
            m.step(FAR)
            n += 1
        self.assertAlmostEqual(m.lid, L.LID_UPPER, places=6)
        self.assertLess(n / HZ, 1.0)

    def test_a_light_touch_on_the_tab_does_not_free(self):
        m = T.OfflineModel(self.axes())
        m.step(self.block_on_tab(0.001))
        self.assertTrue(m.catch.latched)

    def test_hand_on_the_plate_cannot_push_the_tab(self):
        m = T.OfflineModel(self.axes(left=(1.0, 0.0, 0.0)))
        # open fingers along x over the shaft: one pad over the tab, the other on the plate
        arms = {"left": arm((L.UX, L.UY, L.TOP + L.LID_Z1 - 0.004 + T.PAD_TIP), 1.0), "right": FAR["right"]}
        for _ in range(5):
            m.step(arms)
        self.assertTrue(m.catch.latched)

    def test_hand_through_the_shut_lid_is_ignored(self):
        m = T.OfflineModel(self.axes(left=(1.0, 0.0, 0.0)))
        arms = {"left": arm((L.UX, L.UY, L.TOP + L.SEAT_Z + 0.006 + T.PAD_TIP), 1.0), "right": FAR["right"]}
        for _ in range(5):
            m.step(arms)
        self.assertTrue(m.catch.latched)
        self.assertEqual(m.tab, 0.0)

    def test_rotor_on_the_shaft_cannot_be_taken_with_the_lid_shut(self):
        m = T.OfflineModel(self.axes(left=(0.0, 1.0, 0.0)))
        tcp = (L.UX, L.UY, L.TOP + L.SEAT_Z + T.HUB_TIP + T.PAD_TIP)
        m.step({"left": arm(tcp, (T.HUB_W - 0.002) / rigstate.FULL_GAP), "right": FAR["right"]})
        self.assertIsNone(m.rotors[S.TUBE_ROTOR].holder)

    def _carry_strip_rotor_to(self, m, yaw_axis, lid=L.LID_UPPER):
        m.lid = lid
        m.catch.latched = False
        m.catch.opened = True
        b = m.rotors[S.TUBE_ROTOR]
        b.mode, b.pos = "rest", np.array([L.TUBE_STORE[0], L.TUBE_STORE[1], L.TOP + L.TUBE_ROTOR_T])
        r = m.rotors[S.STRIP_ROTOR]
        sx, sy, sz = L.STRIP_ROTOR_START
        grip = (T.HUB_W - 0.002) / rigstate.FULL_GAP
        m.axes_of = self.axes(left=(0.0, 1.0, 0.0))
        m.step({"left": arm((sx, sy, L.TOP + sz + T.HUB_TIP + T.PAD_TIP), grip), "right": FAR["right"]})
        self.assertEqual(r.holder, "left")
        m.axes_of = self.axes(left=yaw_axis)
        drop = (L.UX, L.UY, L.TOP + L.SEAT_Z + T.DROP_RISE + T.HUB_TIP + T.PAD_TIP)
        m.step({"left": arm(drop, grip), "right": FAR["right"]})
        for _ in range(20):
            m.step({"left": arm(drop, 1.0), "right": FAR["right"]})
        return r

    def test_keyed_drop_rests_on_the_noses_and_a_push_clicks_it(self):
        m = T.OfflineModel()
        r = self._carry_strip_rotor_to(m, (1.0, 0.0, 0.0))  # turned a quarter turn: on the key
        self.assertEqual(r.mode, "shaft")
        self.assertAlmostEqual(r.rise, L.NOSE_REST, places=6)
        top = L.TOP + L.SHAFT_TOP - T.SEAT_OVER + T.PAD_TIP
        for _ in range(5):
            m.step({"left": arm((L.UX, L.UY, top), 0.0), "right": FAR["right"]})
        self.assertEqual(r.rise, 0.0)
        self.assertEqual(m.plungers, 0.0)

    def test_off_key_drop_rests_on_the_pin(self):
        m = T.OfflineModel()
        r = self._carry_strip_rotor_to(m, (0.0, 1.0, 0.0))  # not turned
        self.assertEqual(r.mode, "shaft")
        self.assertAlmostEqual(r.rise, L.PIN_REST, places=6)
        top = L.TOP + L.SHAFT_TOP - T.SEAT_OVER + T.PAD_TIP
        for _ in range(5):
            m.step({"left": arm((L.UX, L.UY, top), 0.0), "right": FAR["right"]})
        self.assertAlmostEqual(r.rise, L.PIN_REST, places=6)  # the pin stops it

    def test_rotor_let_go_over_the_shut_lid_stays_on_it(self):
        m = T.OfflineModel()
        r = self._carry_strip_rotor_to(m, (1.0, 0.0, 0.0), lid=0.0)
        self.assertNotEqual(r.mode, "shaft")
        self.assertGreaterEqual(r.pos[2], L.TOP + L.LID_Z1)

    def test_demo_plans_and_variants(self):
        d = T.demo()
        self.assertEqual(set(d), {"left", "right"})
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertEqual(set(T.demo()), {"left", "right"})
            finally:
                os.environ.pop("DEMO_VARIANT")
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for side, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = side
            try:
                d = T.demo()
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
            self.assertNotIn(side, d)
            self.assertIn(other, d)

    def test_order_of_the_hands(self):
        p = T.plans()
        left, right = p["left"], p["right"]
        self.assertGreaterEqual(T._until(left, "wait for the lid to open"), T._until(right, "back off forward and up"))
        side_start = T._until(right, "wait: the left hand presses")
        self.assertGreaterEqual(side_start, T._until(left, "up"))
        self.assertGreaterEqual(T._until(right, "beside the unit"), T._until(left, "aside"))

    def test_grips_are_within_the_gripper_and_at_5_n(self):
        for g, width in ((T.HUB_GRIP, T.HUB_W), (T.STRIP_PINCH, L.STRIP_W)):
            self.assertAlmostEqual(-2 * g, width - 0.002)  # 2 mm of squeeze: 5 N a finger
            self.assertGreaterEqual(width, 0.004)
            self.assertLessEqual(width, 0.080)
        for opened, width in ((T.HUB_OPEN, T.HUB_W), (T.STRIP_OPEN, L.STRIP_W)):
            self.assertGreaterEqual(-2 * opened - width, 0.012)  # 6 mm or more clear each side going down
            self.assertLessEqual(-2 * opened, 0.040)  # and never wide open over the chamber
        for plan in T.plans().values():
            for w in plan:
                self.assertGreaterEqual(w.duration, 0.3)

    def test_demo_length_is_within_the_run_limit(self):
        limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0]) if (
            HERE / "task.json").exists() else 2400
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps, 0.8 * limit)
        self.assertGreater(steps, 1000)


class Folder(unittest.TestCase):
    def test_text_has_no_em_dashes_or_identifying_names(self):
        names = [w[::-1] for w in ("07-T", "lawraggAnahiV", "edualC", "ciporhtnA", "rohtua-oc")]  # not spelled out
        bad = re.compile("|".join([chr(0x2014), chr(0x2013)] + [re.escape(n) for n in names]), re.I)
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "parts.py",
                     "tests/test_scoring.py"):
            p = HERE / name
            if p.exists():
                self.assertIsNone(bad.search(p.read_text()), name)

    def test_task_json_matches_the_code(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        d = json.loads(p.read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["category"], "Other")
        self.assertIn("demo: 1.0", d["scores"])  # the GPU run of 2026-11-19 (job 20261119T063532)
        self.assertEqual(d["run_limit"].split()[0], "2400")
        text = d["setup"] + d["scoring"]
        for needle in ("6770", "174 x 148 x 122 mm", "1.1 kg", "x 150, y 0", "x 76", "75 degrees", "14 x 20 mm",
                       "2 mm", "96 mm", "40 g", "80 x 76 x 20 mm", "110 mm", "30 g", "75 x 10 mm", "17 mm deep",
                       "19 and 31 mm", "x 250, y +200", "22 mm square", "16 mm tall", "x 200, y -200",
                       "columns 4 and 9", "72 x 7 x 21 mm", "1.66 g", "16 x 3 mm", "10 degrees", "8 mm proud",
                       "4.8 mm proud", "3 N", "0.6 degrees", "3 degrees", "within 1 degree", "1 mm of the shaft",
                       "0.5 mm", "0.2 mm", "2.5 mm", "4 mm", "15 degrees", "5 mm or more", "30 mm", "20 mm",
                       "2 degrees", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0"):
            self.assertIn(needle, text, needle)
        self.assertIn("declared rule", d["fidelity"].lower())
        self.assertIn("75 degree", d["fidelity"])

    def test_numbers_in_the_text_match_layout(self):
        self.assertEqual((L.UX, L.UY), (0.150, 0.0))
        self.assertAlmostEqual(L.UX - L.HX, 0.076)
        self.assertEqual(L.LID_UPPER_DEG, 75.0)
        self.assertAlmostEqual(L.TAB_U1 - L.TAB_U0, 0.014)
        self.assertAlmostEqual(2 * L.TAB_HY, 0.020)
        self.assertEqual((L.TAB_RELEASE, L.CATCH_DEG, L.OPENED_DEG, L.FORCED_DEG), (0.002, 0.6, 3.0, 2.0))
        self.assertAlmostEqual(2 * L.TUBE_ROTOR_R, 0.096)
        self.assertEqual((L.TUBE_ROTOR_MASS, L.STRIP_ROTOR_MASS), (0.040, 0.030))
        self.assertEqual(L.STRIP_ROTOR_START[:2], (0.250, 0.200))
        self.assertEqual((L.RACK_X, L.RACK_Y, L.RACK_COLS), (0.200, -0.200, (4, 9)))
        self.assertAlmostEqual(L.STRIP_STARTS[0][1], -0.2225)
        self.assertAlmostEqual(L.STRIP_STARTS[1][1], -0.1775)
        self.assertAlmostEqual(2 * L.KEY_HX, 0.016)
        self.assertAlmostEqual(L.PIN_REST, 0.008)
        self.assertAlmostEqual(L.NOSE_REST, 0.0048)
        self.assertEqual((L.UNIT_MOVED, L.UNIT_TILT_DEG, L.DROP_Z, L.HANDS_OFF), (0.015, 10.0, -0.020, 0.030))
        self.assertEqual((L.STRIP_TOL_XY, L.STRIP_TOL_X, L.STRIP_TOL_Z), (0.0025, 0.004, 0.002))
        self.assertEqual((L.ROTOR_OFF_AXIS, L.ROTOR_TILT_DEG, L.CLICK_MM, L.PLUNGER_OUT), (0.001, 3.0, 0.5, 0.0002))
        self.assertEqual((L.STORED_LOW, L.STORED_TILT_DEG, L.STORED_CLEAR), (0.004, 10.0, 0.005))

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
