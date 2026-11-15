"""On the station bench, real size: a DeWalt DCS331 20 V MAX jigsaw, bare tool (battery out, no blade, 2.45 kg, 286 mm
long), lying on its left side on a fixed rubber mat, its front away from the robot, the shoe toward the robot's right
and the top handle toward its left. The shoe is square (0 degrees), clamped by the keyless bevel lever, which lies
flush under the body with its finger tab just proud of the housing's upper side. Goal: a 45 degree bevel, locked,
hands off.

The saw is one loose mechkit articulation (layout.py): the body (base, 2.15 kg), a carriage on a slider along the saw
(the shoe's slot, 0 to 10 mm forward), the shoe on a hinge about the blade's line on the carriage (-45 to 45 deg) and
the lever on a hinge on the body (0 to 70 deg). Links of one articulation do not collide with each other. The stop,
the detents and the clamp are ONE rule (shoerule.py, run by ``_shoe_rule``, a mechkit on_step callback) that writes the
three joints' drive targets every control step: clamped (lever under 10 deg) everything holds; unclamped the slide
follows the hand, the bevel is held at 0 until the shoe is 6 mm forward (the positive stop), then follows the hand
with light detents at 0, 15, 30 and 45 either way; the lever rests flush under 25 deg and out at 60 above.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the saw is built as an articulation)
import rigstate  # noqa: E402
import shoerule  # noqa: E402
from offline_saw import OfflineSaw  # noqa: E402
from scoring import SAW, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold 1 s on top of the scorer's own 2 s of hands off and stillness

INSTRUCTION = "Set this jigsaw's shoe to a 45 degree bevel and lock it there, then let go of the saw."

YELLOW = (0.93, 0.72, 0.05)
BLACK = (0.08, 0.08, 0.08)
ALUMINIUM = (0.62, 0.63, 0.65)  # the shoe: aluminium (maker's manual)
RUBBER = (0.18, 0.19, 0.20)
BODY_COLOURS = {"housing": YELLOW, "shoulder": YELLOW, "nose": BLACK, "battery_mount": BLACK, "rear_post": YELLOW,
                "front_post": YELLOW, "handle": BLACK, "lever_boss": BLACK}


def _box(b, colour):
    c, s = L.box_centre_size(b)
    return mk.box(tuple(v * L.MM for v in c), tuple(v * L.MM for v in s), colour)


BODY_SHAPES = [_box(b, BODY_COLOURS[n]) for n, b in L.BODY_BOXES.items()]
CARRIAGE = mk.slider("slide", "X", [_box(L.CARRIAGE_BOX, BLACK)], upper=L.SLIDE_TRAVEL * L.MM, mass=L.CARRIAGE_MASS,
                     spring=L.SLIDE_K, damping=L.SLIDE_D)
SHOE = mk.hinge("bevel", "X", tuple(v * L.MM for v in L.PIVOT), [_box(b, ALUMINIUM) for b in L.SHOE_BOXES.values()],
                lower=-math.radians(L.BEVEL_MAX_DEG), upper=math.radians(L.BEVEL_MAX_DEG), mass=L.SHOE_MASS,
                spring=L.BEVEL_K, damping=L.BEVEL_D, parent="slide")
LEVER = mk.hinge("lever", "Y", tuple(v * L.MM for v in L.LEVER_PIVOT), [_box(b, BLACK) for b in L.LEVER_BOXES.values()],
                 upper=math.radians(L.LEVER_MAX_DEG), mass=L.LEVER_MASS, spring=L.LEVER_K, damping=L.LEVER_D)

# Offline (tools/offline/check_task.py only) the saw is its motor housing's box, which no gripper goes near, so the
# checker's carry proxy leaves it alone; offline_saw.py moves the joints instead.
_HOUSING_C, _HOUSING_S = L.box_centre_size(L.BODY_BOXES["housing"])
_HOUSING_CENTRE = np.array(_HOUSING_C) * L.MM
_HOUSING_SIZE = tuple(v * L.MM for v in _HOUSING_S)


class Saw(mk.Assembly):
    """The saw exactly as mechkit builds and reads it in the simulator; the two methods below are for the offline
    checker only."""

    def offline_box(self):
        origin, quat = self.spawn_pose()
        centre = origin + mk.quat_to_matrix(quat) @ _HOUSING_CENTRE
        return tuple(float(v) for v in centre), _HOUSING_SIZE, float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ _HOUSING_CENTRE
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


SAW_SPEC = Saw(name=SAW, pos=L.ORIGIN_M, color=YELLOW, mass=L.BODY_MASS, yaw=0.0, shapes=tuple(BODY_SHAPES),
               joints=(CARRIAGE, SHOE, LEVER), friction=L.SAW_FRICTION, solver_iterations=32)
SAW_SPEC._check()
_MAT_C, _MAT_S = L.box_centre_size((L.MAT_X, L.MAT_Y, (-L.MAT_T, 0.0)))
MAT_SPEC = mk.assembly("mat", origin=L.ORIGIN_M, shapes=[mk.box(tuple(v * L.MM for v in _MAT_C),
                                                                tuple(v * L.MM for v in _MAT_S), RUBBER)],
                       color=RUBBER, friction=L.MAT_FRICTION)
OBJECTS = [MAT_SPEC, SAW_SPEC]


# -- the rule: stop, detents, clamp ------------------------------------------------------------------------------------
MEM = shoerule.Memory()
RULE = {"mode": "clamped", "targets": None, "skipped": 0}


@mk.on_step
def _shoe_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    s = states.get(SAW)
    try:
        q = {j: float(s.joints[j]) for j in ("slide", "bevel", "lever")}
        ok = all(math.isfinite(v) for v in q.values())
    except (AttributeError, KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        RULE["skipped"] += 1  # unreadable: keep the last targets
        t = RULE["targets"] or {"slide": 0.0, "bevel": 0.0, "lever": 0.0}
        MEM.prev_slide = MEM.prev_bevel = None
    else:
        t = shoerule.targets(q["slide"], q["bevel"], q["lever"], MEM)
        if t["mode"] != RULE["mode"]:
            print(f"[jigsaw] rule {RULE['mode']} -> {t['mode']} at episode {mk.episode()} step {mk.step()}: slide "
                  f"{q['slide'] * 1000:.2f} mm, bevel {math.degrees(q['bevel']):.2f} deg, lever "
                  f"{math.degrees(q['lever']):.1f} deg", file=sys.stderr, flush=True)
        RULE.update(mode=t["mode"], targets=t)
    for j in ("slide", "bevel", "lever"):
        mk.set_targets(SAW_SPEC, j, t[j])


@mk.on_reset
def _rule_reset() -> None:
    MEM.reset()
    RULE.update(mode="clamped", targets=None)
    for j in ("slide", "bevel", "lever"):
        mk.set_targets(SAW_SPEC, j, 0.0)
    OFFLINE.reset()


_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
OFFLINE = OfflineSaw()  # only stepped without a simulator (offline_saw.py)


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    OFFLINE.reset()


def no_simulator(objects) -> bool:
    """True only where nothing simulates the saw: mechkit has no environment AND the state handed in was not read
    from one (mechkit marks those ``source="sim"``)."""
    return mk.env() is None and getattr(objects.get(SAW), "source", "sim") != "sim"


def offline_states(objects, arms) -> dict:
    OFFLINE.step(arms)
    out = dict(objects)
    s = objects[SAW]
    pose = (*np.asarray(s.origin, dtype=float), *np.asarray(s.quat_xyzw, dtype=float))
    out[SAW] = mk.state_from_poses(SAW_SPEC, pose,
                                   np.zeros(6), joints=OFFLINE.joints(), source="offline")
    return out


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only
            pass


def _pose_fields(objects, arms) -> dict:
    """Trace only (GPU tuning): the body's origin offset from where it lay (mm, B frame) and its tilt; the shoe's top
    front corner and the lever tab's top in B (mm); both TCPs in B (mm) and the gripper openings."""
    out = {}
    try:
        s = objects[SAW]
        o = np.asarray(s.origin, dtype=float)
        out["body_mm"] = [round(float(v) / L.MM, 1) for v in (o - np.asarray(L.ORIGIN_M))]
        out["body_tilt_deg"] = round(math.degrees(s.tilt()), 2)
        corner = (L.SHOE_X[1], L.EDGE_W, L.Z_C + L.SHOE_HALF_W)
        tab = (-152.0, 15.0, L.LEVER_BOXES["tab"][2][1])
        for name, body, p in (("shoe_corner_mm", "bevel", corner), ("lever_tab_mm", "lever", tab),
                              ("tab_spot_mm", "lever", TAB_SPOT), ("edge_spot_mm", "bevel", EDGE_SPOT)):
            w = s.link_to_world(body, np.array(p) * L.MM)
            out[name] = [round(float(v), 1) for v in s.to_local(w) / L.MM]
    except Exception:  # noqa: BLE001 - never break scoring for a trace field
        pass
    try:
        for side in ("left", "right"):
            a = arms[side]
            out[f"tcp_{side}_mm"] = [round(float(v), 1) for v in OfflineSaw.to_b(a.tcp)]
            out[f"open_{side}"] = round(float(a.opening), 3)
        # where the spots the right hand pinches (the tab) and grips (the shoe's edge) are from its pad tips: about
        # (0, 0, PINCH_UP) and (0, 0, EDGE_UP) while held; a growing offset is a slip
        tips = np.array(out["tcp_right_mm"]) - np.array([0.0, 0.0, TCP_UP])
        for name in ("tab_spot", "edge_spot"):
            if f"{name}_mm" in out:
                out[f"{name}_from_right_tips_mm"] = [round(float(v), 1) for v in np.array(out[f"{name}_mm"]) - tips]
    except Exception:  # noqa: BLE001
        pass
    return out


def success(objects) -> bool:
    """Runs every control step: feeds the tracker (and, without a simulator, the offline kinematics)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    offline = no_simulator(objects)
    if offline:
        objects = offline_states(objects, arms)
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "rule": {"mode": RULE["mode"], "switches": MEM.switches, "skipped": RULE["skipped"],
                                      "targets": None if RULE["targets"] is None else
                                      {k: round(float(v), 5) for k, v in RULE["targets"].items() if k != "mode"}}}
    row.update(_pose_fields(objects, arms))
    if offline:
        row["offline"] = {"events": [list(map(str, e)) for e in OFFLINE.events[-3:]]}
    _trace(row)
    return result.success


def score(objects) -> float:
    """The score of the latest scored step (see scoring.py)."""
    return _TRACKER.last.score


# -- the scripted demo -------------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout. Points are in the saw's frame B (mm: x along the saw toward
# its front, y from the shoe's working face toward the handle, z over the mat) and are the PAD TIPS; the TCP is 13 mm
# above. Both grippers point straight down with the fingers closing along world y (wrist yaw 0). A closed gripper
# (1 mm commanded) is a block 20 mm along x and 17 mm along y.
#   Left:  rests its closed fingers on the top handle's upper side, 1.5 mm in, for the whole job, then leaves.
#   Right: (1) open astride the lever's finger tab, pinches it (the fingers close along the lever's hinge axis, so the
#              tab turns between the pads like a pin) and carries it up along its arc to 50 deg: past 25 the lever
#              rests out at 60 (the clamp lets go); lets go;
#          (2) open, straddles the shoe's edge where it stands clear in front of the gear housing, closes on it and
#              pulls it 12 mm forward (it stops at 10, 6 takes it off the 0 degree stop), lets go;
#          (3) closed, from the housing side pushes the shoe's upper edge out past the 15 and 30 detents to 45;
#          (4) open astride the swung-out lever, pinches the same spot of its tab and carries it back down along its
#              arc to 5 deg: under 25 it falls flush and clamps; lets go, leaves.
# The first GPU run pushed the tab forward with a closed block instead: the push met the tab's rear face only 7 to
# 22 mm above the pivot and 57 mm behind it, which wedges for pad friction over about 0.4 (the arm stalled 21 mm
# short and the lever turned 1 deg). A pinch moves the tab the way it turns.
OPEN = -0.020  # 40 mm
OPEN_WIDE = -0.030  # 60 mm, approaching the edge
OPEN_TAB = -0.010  # 20 mm astride the 6 mm lever: 7 mm each side of it, the outer finger 7 mm from the housing
GRIP_PUSH = -0.0005  # 1 mm commanded: the pads one block
GRIP_EDGE = -0.001  # 2 mm commanded on the 10 mm edge (and on the 6 mm tab)
TCP_UP = 13.0
HOME = {"left": (0.28, 0.25, 0.12), "right": (0.28, -0.25, 0.12)}  # world TCP points (height over the bench)

TAB = {"x": -147.0, "y": 15.0, "z": 99.5}  # pad tips astride the tab at rest: its middle, 0.4 mm over the housing top
TAB_TOP = L.LEVER_BOXES["tab"][2][1]  # 110: the pads hold the tab's top 10.5 mm
PINCH_UP = (TAB_TOP - TAB["z"]) / 2  # the held spot (the middle of what the pads cover) over the pad tips
SWING_OUT_DEG = 50.0  # let go here: 25 over the snap angle, the rule takes it on to 60
SWING_FROM_DEG = L.LEVER_OPEN_DEG  # where the swung-out lever rests, and is pinched again
SWING_BACK_DEG = 5.0  # let go here: under the 10 deg lock and the 25 deg snap
EDGE = {"x": 45.0, "y": L.EDGE_W / 2, "z": L.Z_C + L.SHOE_HALF_W - 8.5, "pull": 12.0}
TILT = {"x": 47.0, "y0": 22.0, "y1": -12.5, "z": 74.5}  # block centre; face 13.5 -> -21 (the shoe is at 45 at -17.7)
BLOCK_HALF_Y = 8.5  # the closed block's half width along y: its centre is this far from its pushing face
VIA = (-60.0, 175.0)  # x, pad tips: the way back from the shoe to the lever passes above the lever's open tab
HANDLE_PRESS = {"x": -60.0, "y": 200.0, "z": L.HANDLE_TOP_Z - 1.5}
HOVER = 150.0
LEFT_AWAY_S = 3.5  # the left hand's lift and trip home
FRONT_HOVER = 115.0  # pad tips over the mat at the shoe's front (20 mm over the shoe's edge)


EDGE_UP = (L.Z_C + L.SHOE_HALF_W - EDGE["z"]) / 2  # the gripped spot of the edge over the pad tips
TAB_SPOT = (TAB["x"], TAB["y"], TAB["z"] + PINCH_UP)  # B at rest (mm): the spot of the tab the pads pinch (trace)
EDGE_SPOT = (EDGE["x"], EDGE["y"], EDGE["z"] + EDGE_UP)  # the spot of the shoe's edge the pads grip (trace)


def W(x, y, z_tips):
    """World TCP point for pad tips at (x, y, z_tips) in B (mm)."""
    return L.world_mm((x, y, z_tips + TCP_UP))


def tab_tips(deg: float) -> tuple:
    """B (mm): the pad tips that keep the pinched spot of the tab where it is with the lever at ``deg``."""
    piv = np.array(L.LEVER_PIVOT)
    spot = np.array([TAB["x"], TAB["y"], TAB["z"] + PINCH_UP])
    p = L.rot_y(math.radians(deg)) @ (spot - piv) + piv
    return float(p[0]), TAB["y"], float(p[2] - PINCH_UP)


def _home(side):
    return (HOME[side][0], HOME[side][1], L.TOP + HOME[side][2])


class _Plan:
    def __init__(self):
        self.wps: list[Waypoint] = []
        self.t = 0.0

    def add(self, d, pos, finger, label, yaw=0.0):
        self.wps.append(Waypoint(d, pos, finger, label, yaw))
        self.t += d
        return self.t

    def hold_until(self, t, label="hold"):
        if t > self.t + 1e-9:
            w = self.wps[-1]
            self.add(t - self.t, w.pos, w.finger, label, w.yaw)

    def arc(self, a0, a1, leg_s, first_label, label):
        """Carry the pinched tab along its arc from ``a0`` to ``a1`` deg in legs of 10 deg or less, ``leg_s`` each
        (straight chords; the arm's give and the pads' turn take up the 0.2 mm sag of a chord)."""
        n = max(1, int(math.ceil(abs(a1 - a0) / 10.0 - 1e-9)))
        for k in range(1, n + 1):
            a = a0 + (a1 - a0) * k / n
            self.add(leg_s, W(*tab_tips(a)), GRIP_EDGE,
                     f"{first_label} ({a:.0f} deg)" if k == 1 else f"{label} ({a:.0f} deg)")


def tilt_face_y(deg: float, z_tips: float = TILT["z"]) -> float:
    """The block's -y face (B, mm) that has pushed the shoe to ``deg``: the shoe's farthest +y point at or above the
    pad tips (the block reaches 38 mm up)."""
    p = L.shoe_points(math.radians(deg), 0.0, n=21)
    sel = p[(p[:, 2] >= z_tips) & (p[:, 2] <= z_tips + 38.0)]
    return float(sel[:, 1].max())


TILT_15_Y1 = round(tilt_face_y(16.0) + BLOCK_HALF_Y, 1)  # stop_at_15: 1 deg past the 15 detent, inside its capture


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    rp = _Plan()
    tb = TAB
    if variant != "no_unlock":
        rp.add(2.5, W(tb["x"], tb["y"], HOVER), OPEN_TAB, "over the lever's finger tab, fingers open 20 mm")
        rp.add(1.5, W(tb["x"], tb["y"], tb["z"]), OPEN_TAB, "down astride the tab (pad tips at the housing's top)")
        rp.add(0.5, W(tb["x"], tb["y"], tb["z"]), OPEN_TAB, "settle")
        rp.add(1.0, W(tb["x"], tb["y"], tb["z"]), GRIP_EDGE, "pinch the tab (commanded 2 mm)")
        rp.add(0.5, W(tb["x"], tb["y"], tb["z"]), GRIP_EDGE, "settle the pinch")
        rp.arc(0.0, SWING_OUT_DEG, 0.6, "swing the lever out along its arc", "on along the arc")
        rp.add(0.5, W(*tab_tips(SWING_OUT_DEG)), GRIP_EDGE, "hold")
        rp.add(0.75, W(*tab_tips(SWING_OUT_DEG)), OPEN_TAB, "let go: past 25 the lever goes on out to 60")
        x, y, z = tab_tips(SWING_OUT_DEG)
        rp.add(1.0, W(x, y, HOVER + 25.0), OPEN_TAB, "straight up, the tab between the fingers")
    e = EDGE
    t_front = rp.t + 2.5  # the right hand is over the shoe's front, far from the left's way in
    if variant != "no_slide":
        rp.add(2.5, W(e["x"], e["y"], FRONT_HOVER), OPEN_WIDE, "over the shoe's edge in front of the gear housing")
        rp.add(1.5, W(e["x"], e["y"], e["z"]), OPEN_WIDE, "down astride the edge")
        rp.add(0.5, W(e["x"], e["y"], e["z"]), OPEN_WIDE, "settle")
        rp.add(1.5, W(e["x"], e["y"], e["z"]), GRIP_EDGE, "close on the edge (commanded 2 mm)")
        rp.add(0.5, W(e["x"], e["y"], e["z"]), GRIP_EDGE, "settle the grip")
        rp.add(2.5, W(e["x"] + e["pull"], e["y"], e["z"]), GRIP_EDGE,
               f"pull the shoe forward (commanded {e['pull']:.0f} mm; off the stop at 6, end of slot at 10)")
        rp.add(0.5, W(e["x"] + e["pull"], e["y"], e["z"]), GRIP_EDGE, "settle")
        rp.add(1.0, W(e["x"] + e["pull"], e["y"], e["z"]), OPEN_WIDE, "let go")
        rp.add(1.0, W(e["x"] + e["pull"], e["y"], FRONT_HOVER), OPEN_WIDE, "straight up")
    tt = TILT
    y1 = TILT_15_Y1 if variant == "stop_at_15" else tt["y1"]
    rp.add(1.5, W(tt["x"], tt["y0"], FRONT_HOVER), GRIP_PUSH, "over the housing side of the shoe's edge, closed")
    rp.add(1.5, W(tt["x"], tt["y0"], tt["z"]), GRIP_PUSH, "down beside the shoe's inner rib")
    rp.add(0.5, W(tt["x"], tt["y0"], tt["z"]), GRIP_PUSH, "settle")
    rp.add(4.0, W(tt["x"], y1, tt["z"]), GRIP_PUSH,
           "push the shoe's upper edge outward past 15 and 30 into 45" if variant != "stop_at_15"
           else "push the shoe's upper edge outward into the 15 detent only")
    rp.add(0.5, W(tt["x"], y1, tt["z"]), GRIP_PUSH, "hold")
    rp.add(1.5, W(tt["x"], tt["y0"], tt["z"]), GRIP_PUSH, "back off toward the housing side")
    rp.add(1.0, W(tt["x"], tt["y0"], FRONT_HOVER), GRIP_PUSH, "straight up")
    t_tilted = rp.t
    rp.hold_until(t_tilted + LEFT_AWAY_S, "wait over the shoe while the left hand goes home")
    if variant not in ("lever_left_out",):
        x, y, z = tab_tips(SWING_FROM_DEG)
        rp.add(2.0, W(VIA[0], y, VIA[1]), OPEN_TAB, "back along the saw, up over the housing, fingers open 20 mm")
        rp.add(1.5, W(x, y, VIA[1]), OPEN_TAB, "over the swung-out lever's tab")
        rp.add(1.5, W(x, y, z), OPEN_TAB, "down astride the tab")
        rp.add(1.0, W(x, y, z), GRIP_EDGE, "pinch the tab again (commanded 2 mm)")
        rp.add(0.5, W(x, y, z), GRIP_EDGE, "settle the pinch")
        rp.arc(SWING_FROM_DEG, SWING_BACK_DEG, 0.5, "swing the lever back down along its arc", "on down the arc")
        rp.add(0.5, W(*tab_tips(SWING_BACK_DEG)), GRIP_EDGE, "hold: under 10 deg it clamps")
        rp.add(0.75, W(*tab_tips(SWING_BACK_DEG)), OPEN_TAB, "let go: under 25 the lever falls flush")
        x, y, z = tab_tips(SWING_BACK_DEG)
        rp.add(1.0, W(x, y, HOVER), OPEN_TAB, "straight up")
    rp.add(2.5, _home("right"), OPEN, "home")
    # the left hand rests on the handle while the right slides and tilts the shoe (the pushes along and across the
    # saw), arriving once the right is at the shoe's front and leaving before it goes back to the lever (their elbows
    # would come within 2 cm near the lever)
    hp = HANDLE_PRESS
    lpl = _Plan()
    lpl.add(1.0, _home("left"), OPEN, "wait clear while the right hand swings the lever out")
    lpl.hold_until(t_front - 2.5, "wait clear while the right hand swings the lever out")
    lpl.add(2.5, W(hp["x"], hp["y"], hp["z"] + 40.0), GRIP_PUSH, "over the top handle, fingers closed")
    lpl.add(1.5, W(hp["x"], hp["y"], hp["z"]), GRIP_PUSH, "rest on the handle's upper side (1.5 mm in): hold the saw")
    lpl.hold_until(t_tilted, "hold the saw down while the right hand slides and tilts the shoe")
    lpl.add(1.0, W(hp["x"], hp["y"], hp["z"] + 60.0), GRIP_PUSH, "straight up")
    lpl.add(2.5, _home("left"), OPEN, "home")
    rp.hold_until(lpl.t, "home")
    return {"left": lpl.wps, "right": rp.wps}


VARIANTS = ("no_unlock", "no_slide", "stop_at_15", "lever_left_out")


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm checks).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_unlock       the lever is never swung out: the pull and the push meet a clamped shoe; expect 0
      no_slide        the lever out, but the shoe is never slid forward: the push meets the 0 degree stop; the lever
                      swung back; expect 0
      stop_at_15      the push stops at the first detent (15 degrees); the lever swung back; expect 0.25
      lever_left_out  the shoe at 45 but the lever left out (not clamped); expect 0.75"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
