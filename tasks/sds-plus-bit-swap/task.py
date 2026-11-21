"""On the bench, real size: a Bosch 11224VSRC SDS-plus rotary hammer (unplugged) lying on its left side, its chuck
pointing to the robot's right with a 6 x 210 mm SDS-plus bit locked in it; a 10 x 160 mm SDS-plus bit lies on the
bench in front of the chuck, its drive slots turned 45 degrees off the chuck's keys.

Objects (mechkit.py next to this file): the hammer (a loose mechanism: the locking sleeve slides back against its
spring; a stand-in gate for the drive keys slides across the bore) and the two bits (loose bodies). Three rules:
  - the locking balls (welds, mechkit.Weld: the bit's pose is rewritten on the hammer every control step while
    locked): a bit is locked while the sleeve is less than 6 mm back; it locks by itself once pushed home (40 mm in)
    with its slots on the keys and the sleeve let go;
  - the drive keys: a shank gets past 12 mm into the bore only with its slots within 8 degrees of the keys (the gate
    closes across the bore otherwise).
Nothing else is moved by code.

Goal: take the 6 mm bit out (the sleeve held back while it is drawn out) and put the 10 mm bit in, twisting it until
its slots find the keys and pushing it home, as Bosch's manual says. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=no_twist | no_sleeve (see demo() below).
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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402
from scoring import HAMMER, NEW_BIT, OLD_BIT, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Swap the drill bit in the rotary hammer for the 10 mm bit on the bench."

SLEEVE, KEYS = "sleeve", "keys"
BIT_MU = float(os.environ.get("SDS_BIT_MU", 0.25))  # steel bits, lightly greased as Bosch asks (est.)


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def tube(y0: float, y1: float, r_in: float, r_out: float, color) -> list:
    """A tube round the chuck's axis (hammer frame) from y0 to y1: eight flat staves, their inner faces r_in from the
    axis."""
    rm, w = (r_in + r_out) / 2, 2 * r_out * math.tan(math.pi / 8) * 1.02
    out = []
    for k in range(8):
        a = k * math.pi / 4
        out.append(mk.box((rm * math.cos(a), (y0 + y1) / 2, L.AXIS_Z + rm * math.sin(a)), (r_out - r_in, y1 - y0, w),
                          color, ypr=(0.0, -a, 0.0)))
    return out


def funnel(y0: float, y1: float, r0: float, r1: float, color, t: float = 0.002) -> list:
    """A cone round the chuck's axis whose inside narrows from r0 at y0 to r1 at y1: eight staves, each tipped so its
    inner face looks out toward an incoming shank and slides it to the axis."""
    beta = math.atan2(r0 - r1, y1 - y0)
    length = math.hypot(r0 - r1, y1 - y0)
    w = 2 * (r0 + t) * math.tan(math.pi / 8) * 1.05
    out = []
    for k in range(8):
        a = k * math.pi / 4
        rot = tiltgrip.ry(-a) @ np.array([[math.cos(beta), -math.sin(beta), 0.0], [math.sin(beta), math.cos(beta), 0.0],
                                         [0.0, 0.0, 1.0]])
        mid_r, mid_y = (r0 + r1) / 2, (y0 + y1) / 2
        c = np.array([mid_r * math.cos(a), mid_y, L.AXIS_Z + mid_r * math.sin(a)]) + rot @ np.array([t / 2, 0.0, 0.0])
        out.append(mk.box(_f(c), (t, length, w), color, quat=_f(mk.matrix_to_quat(rot))))
    return out


def _hammer():
    # the housing: a round-edged body (all four long edges rounded, a flat strip left under it) with a black gear
    # case at its front and vent slots along its top
    (x0, x1), (z0, z1), r = L.BODY_X, L.BODY_Z, 0.025
    yb, yl = (L.NOSE_Y[1] + L.LENGTH) / 2, L.LENGTH - L.NOSE_Y[1]
    shapes = [mk.box(((x0 + x1) / 2, yb, (z0 + z1) / 2), (x1 - x0 - 2 * r, yl, z1 - z0), L.BODY),
              mk.box(((x0 + x1) / 2, yb, (z0 + z1) / 2), (x1 - x0, yl, z1 - z0 - 2 * r), L.BODY)]
    shapes += [mk.cylinder((xc, yb, zc), r, yl, "Y", L.BODY) for xc in (x0 + r, x1 - r) for zc in (z0 + r, z1 - r)]
    gy0, gy1 = L.NOSE_Y[1], L.NOSE_Y[1] + 0.075
    shapes.append(mk.box(((x0 + x1) / 2, (gy0 + gy1) / 2, (z0 + z1) / 2), (x1 - x0 - 2 * r + 0.002, gy1 - gy0,
                                                                         z1 - z0 + 0.002), L.DARK))
    shapes += [mk.box(((x0 + x1) / 2 + dx, 0.230, z1 + 0.0005), (0.004, 0.060, 0.001), L.DARK)
               for dx in (-0.015, -0.005, 0.005)]  # vents
    shapes.append(mk.cylinder((0.0, (L.NOSE_Y[0] + L.NOSE_Y[1]) / 2, L.AXIS_Z), L.NOSE_R, L.NOSE_Y[1] - L.NOSE_Y[0],
                              "Y", L.DARK))
    shapes += tube(L.BARREL_Y[0], L.BARREL_Y[1], L.BORE_R, L.BARREL_R, L.STEEL)  # the tool holder
    shapes += funnel(L.CAP_Y[0], L.BARREL_Y[0] + 0.001, L.CAP_MOUTH, L.CAP_EXIT, L.DARK)  # the dust cap's cone: it
    # centres a shank before the bore (its exit, corners included, inside the bore's flats)
    shapes.append(mk.box((0.0, L.DEPTH + 0.002, L.AXIS_Z), (2 * L.BORE_R + 0.002, 0.004, 2 * L.BORE_R + 0.002), L.STEEL))
    (gx0, gx1), (gy0, gy1) = L.GRIP_X, L.GRIP_Y
    shapes += [mk.box(((gx0 + gx1) / 2, (gy0 + gy1) / 2, 0.038), (gx1 - gx0, gy1 - gy0, 0.056), L.DARK),
               mk.box((-0.095, gy0 - 0.005, 0.038), (0.030, 0.010, 0.020), L.DARK),  # the trigger
               mk.cylinder((gx0 - 0.020, (gy0 + gy1) / 2, 0.038), 0.006, 0.040, "X", L.DARK)]  # the cord's guard
    sleeve = mk.slider(SLEEVE, "Y", tube(L.SLEEVE_Y[0], L.SLEEVE_Y[1], L.SLEEVE_R[0], L.SLEEVE_R[1], L.DARK),
                       upper=L.SLEEVE_TRAVEL, mass=0.05, spring=L.SLEEVE_SPRING, damping=5.0)
    # The drive keys' stand-in: a plate drawn (joint 0) clear of the bore inside the holder's wall; a rule slides it
    # KEY_TRAVEL across the bore at KEY_DEPTH whenever the shank coming in has its slots off the keys.
    keys = mk.slider(KEYS, "-X", [mk.box((L.KEY_TRAVEL, L.KEY_DEPTH, L.AXIS_Z), (2 * L.BORE_R, 0.002, 2 * L.BORE_R),
                                         L.STEEL)], upper=L.KEY_TRAVEL, mass=0.01, spring=3000.0, damping=5.0)
    return mk.assembly(HAMMER, (L.HAMMER_AT[0], L.HAMMER_AT[1], L.TOP), shapes, joints=[sleeve, keys],
                       mass=L.M_HAMMER, yaw=L.HAMMER_YAW, color=L.BODY, friction=L.HAMMER_FRICTION, contact_offset=0.002)


def _bit(name, flute_r, flute_l, mass, origin, slot_offset, colour):
    """An SDS-plus bit, its frame at the end of its shank with +y from the tip toward it: the 10 mm shank, the fluted
    body, a carbide tip, and the two open drive slots at the shank's end (``slot_offset`` round from the frame's x)."""
    shapes = [mk.cylinder((0.0, -L.SHANK_L / 2, 0.0), L.SHANK_R, L.SHANK_L, "Y", L.STEEL),
              mk.cylinder((0.0, -L.SHANK_L - flute_l / 2, 0.0), flute_r, flute_l, "Y", colour),
              mk.box((0.0, -L.SHANK_L - flute_l - 0.002, 0.0), (2 * flute_r + 0.001, 0.004, 0.002), L.STEEL)]
    for sx in (1, -1):
        a = slot_offset + (0.0 if sx > 0 else math.pi)
        rr = L.SHANK_R - 0.0005
        shapes.append(mk.box((rr * math.cos(a), -0.010, rr * math.sin(a)), (0.0015, 0.020, 0.003), L.DARK,
                             ypr=(0.0, -a, 0.0)))
    return mk.assembly(name, _f(origin), shapes, mass=mass, yaw=L.HAMMER_YAW, color=colour, friction=(BIT_MU, BIT_MU * 0.8),
                       contact_offset=0.0015,
                       angular_damping=3.0)  # rolling resistance: a bit lying on the bench stays put


HAMMER_ORIGIN = np.array([L.HAMMER_AT[0], L.HAMMER_AT[1], L.TOP])
HAMMER_ROT = np.array([[math.cos(L.HAMMER_YAW), -math.sin(L.HAMMER_YAW), 0.0],
                       [math.sin(L.HAMMER_YAW), math.cos(L.HAMMER_YAW), 0.0], [0.0, 0.0, 1.0]])
OLD_HOME = np.array([0.0, L.DEPTH, L.AXIS_Z])
NEW_ORIGIN = L.at(L.NEW_START[0] + (L.SHANK_L + L.NEW_L) / 2, L.NEW_START[1], L.SHANK_R + 0.0005)


def hammer_pt(p_local) -> np.ndarray:
    """A point given in the hammer's frame, in the world."""
    return HAMMER_ORIGIN + HAMMER_ROT @ np.asarray(p_local, dtype=float)
HAMMER_SPEC = _hammer()
OLD_SPEC = _bit(OLD_BIT, L.OLD_R, L.OLD_L, L.M_OLD, hammer_pt(OLD_HOME), 0.0, (0.55, 0.56, 0.60))
NEW_SPEC = _bit(NEW_BIT, L.NEW_R, L.NEW_L, L.M_NEW, NEW_ORIGIN, L.NEW_ROLL, (0.20, 0.20, 0.22))
OBJECTS = [HAMMER_SPEC, OLD_SPEC, NEW_SPEC]
BITS = ((OLD_SPEC, OLD_BIT, 0.0), (NEW_SPEC, NEW_BIT, L.NEW_ROLL))


# -- the rules: locking balls and drive keys -------------------------------------------------------------------------
IDENTITY = (0.0, 0.0, 0.0, 1.0)
LOCKS = {OLD_BIT: mk.weld(OLD_SPEC, HAMMER_SPEC, (_f(OLD_HOME), IDENTITY), name="old_bit_locked"),
         NEW_BIT: mk.weld(NEW_SPEC, HAMMER_SPEC, name="new_bit_locked")}
_RULES = {"episode": None}


def keys_open(seats) -> bool:
    """The gate stays clear of the bore unless a shank at the keys has its slots off them (no shank: closed). A
    shank counts as at the keys if its end is anywhere in the bore's mouth past KEY_DEPTH - 4 mm (judged loosely, so
    a bit sagging in the bore's play never has the gate closed on it)."""
    near = [s for s in seats if s.depth >= L.KEY_DEPTH - 0.004]
    if not near:
        return False
    return all(s.roll <= L.KEY_TOL or s.depth > L.KEY_DEPTH + 0.002 for s in near)


def snapped(hammer, bit, slot_offset) -> tuple:
    """The locked pose for a bit pushed home: on the axis at home depth, its slots turned onto the keys."""
    a = S.slot_angle(hammer, bit, slot_offset)
    resid = a - round(a / math.pi) * math.pi
    rel = tiltgrip.ry(resid) @ (hammer.rot.T @ bit.rot)
    return _f(OLD_HOME), _f(mk.matrix_to_quat(rel))


@mk.on_step
def _chuck(states) -> None:
    if _RULES["episode"] != mk.episode():
        _RULES["episode"] = mk.episode()
        LOCKS[OLD_BIT].fire(states)
        return
    try:
        sleeve = float(states[HAMMER].joints[SLEEVE])
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    hammer = S.body(states[HAMMER])
    if hammer is None:
        return
    seats = []
    for spec, name, off in BITS:
        b = S.body(states[name])
        if b is None:
            continue
        lock = LOCKS[name]
        # a locked bit is home by definition: between the weld's rewrites it sags a little, which must not read as
        # off the axis
        st = S.Seat(True, L.DEPTH, 0.0) if lock.active else S.seat(hammer, b, off)
        p = hammer.local(b.origin)
        if lock.active or math.hypot(p[0], p[2] - L.AXIS_Z) <= 0.010:
            seats.append(st)
        if lock.active and sleeve >= L.SLEEVE_FREE:
            lock.release()
        elif (not lock.active and sleeve < L.SLEEVE_FREE and st.in_bore and abs(st.depth - L.DEPTH) <= L.HOME_TOL
              and st.roll <= L.KEY_TOL):  # home: within 2 mm of 40 mm deep, either side
            lock.fire(states, rel_pose=snapped(hammer, b, off))
    _RULES["keys_target"] = 0.0 if keys_open(seats) else L.KEY_TRAVEL
    mk.set_targets(HAMMER_SPEC, KEYS, _RULES["keys_target"])


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """The 6 mm bit out and put down, the 10 mm bit locked in, hands clear; held 1 s. Runs every control step, so it
    also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "old_locked": LOCKS[OLD_BIT].active, "new_locked": LOCKS[NEW_BIT].active,
           "keys_target_mm": round(1000 * _RULES.get("keys_target", -1.0), 1), **result.details}
    try:
        j = objects[HAMMER].joints
        row["sleeve_mm"], row["keys_mm"] = round(float(j[SLEEVE]) * 1000, 2), round(float(j[KEYS]) * 1000, 2)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if arms is not None:
        try:
            row["tcp"] = {s: [round(float(v), 4) for v in arms[s].tcp] for s in ("left", "right")}
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 old bit out, new bit locked, hands clear; 0.75 locked; 0.5 new bit started in the bore; 0.25 old bit out."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator. Grips with the pads either side along x.
#   1. The left hand's closed fingertips push the locking sleeve's front rim back toward the body (8 mm) and hold it; the
#      right pinches the 6 mm bit near its tip end and draws it 100 mm out, then puts it down on the bench. The sleeve
#      is let go.
#   2. The left hand presses the housing down onto the bench. The right picks the 10 mm bit up near its tip end,
#      brings it onto the chuck's axis, pushes its shank 8 mm in (up to the keys), turns it 45 degrees about its own
#      axis while pushing (its slots find the keys), and pushes it home; it locks. Both let go.
OPEN = -0.04695
CLOSED = 0.0
TIP = 0.013
YAW_R = 0.0  # pads either side along y, across a bit lying along x
YAW_L = float(os.environ.get("SDS_YAW_L", -math.pi / 2))  # the pushing hand: its closed fingers' thin side toward the sleeve
RIM_H = L.CAP_R + 0.0025  # fingertips this far above the axis: over the dust cap, on the sleeve's front rim
RIM_BACK = 0.012  # the fingers' block starts this far in front of the sleeve (about half its thickness and room)
PUSH_SLEEVE = float(os.environ.get("SDS_PUSH", 0.045))  # 8 mm of travel once the fingers meet the rim, and the arm's give against the spring
OLD_PINCH = 0.0  # closed hard on the 6 mm bit
NEW_PINCH = -(L.NEW_R - 0.004)  # a firm squeeze: a pinch only weakly resists the bit pivoting between the pads
PULL_BIT = 0.100
ASIDE = 0.050  # the drawn-out bit is put down this far to the right of the chuck's line
PIN_Y = float(os.environ.get("SDS_PIN_Y", 0.062))  # the left hand presses the gear case down this far behind the nose
OLD_GRIP = float(os.environ.get("SDS_OLD_GRIP", 0.070))  # the 6 mm bit is held this far from its tip
LIFT_PULL = float(os.environ.get("SDS_LIFT", 0.003))  # the pull rises a little: the arm sags as it pulls
H_AXIS = 0.005  # TCP above a pinched bit's or the sleeve's axis (pad tips 8 mm below it)
H_BENCH = 0.010  # TCP above the 10 mm bit's axis when picking it off the bench (tips 2 mm off the bench)
GRIP_FROM_END = 0.080  # the 10 mm bit is carried held at its middle (its centre of mass: it does not pivot in the pinch)
REGRIP_FROM_END = 0.140  # and pushed home held near its tip, clear of the dust cap
MID_DEPTH = 0.025  # pushed this far in before the regrip: the bore holds it
TIP_SAG = 0.008  # how far the tip end then hangs below the axis where it is regripped (worked out on the simulator)
TWIST = float(os.environ.get("SDS_TWIST_DEG", 45.0))  # turns the slots from 45 deg onto the keys
UP = np.array([0.0, 0.0, 0.06])
UP_L = np.array([0.0, 0.0, 0.035])  # the left arm reaches across: short hops keep its wrist inside its limits
PARK = {"left": L.at(0.16, 0.26, 0.10), "right": L.at(0.16, -0.26, 0.10)}
VARIANTS = ("no_twist", "no_sleeve")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, yaw)


def axis_pt(y_rel: float) -> np.ndarray:
    """A point on the chuck's axis, ``y_rel`` from the nose along it (world)."""
    return hammer_pt([0.0, y_rel, L.AXIS_Z])


ALONG = HAMMER_ROT @ np.array([0.0, 1.0, 0.0])  # the chuck's axis, into the tool (world)


AIM = np.array([float(os.environ.get("SDS_AIM_DX", 0.0)), 0.0, float(os.environ.get("SDS_AIM_DZ", 0.0))])  # where the
# held bit sits in the grip, against where the plan puts it (worked out on the simulator)


def holding(end_rel: float, roll: float = 0.0, grip: float = None) -> tuple[np.ndarray, float]:
    """(TCP, yaw) holding the 10 mm bit on the axis with its shank end ``end_rel`` past the nose, turned ``roll``,
    gripped ``grip`` from its shank end."""
    centre = axis_pt(end_rel - (GRIP_FROM_END if grip is None else grip)) + AIM
    return centre + tiltgrip.rx(roll) @ np.array([0.0, 0.0, H_BENCH]), tiltgrip.tilted(YAW_R, roll)


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    # the sleeve: closed fingertips just above the dust cap, against the sleeve's front rim, pushed toward the body
    rim = axis_pt(L.SLEEVE_Y[0] - RIM_BACK) + [0.0, 0.0, RIM_H + TIP]
    old = axis_pt(L.DEPTH - L.SHANK_L - L.OLD_L + OLD_GRIP) + [0.0, 0.0, H_AXIS]  # near the chuck: a short lever
    left = [_w(2.5, rim + UP_L, CLOSED, "over the front of the locking sleeve, fingers closed", YAW_L),
            _w(1.2, rim, CLOSED, "down in front of it, the tips just above the dust cap", YAW_L)]
    if variant != "no_sleeve":
        left.append(_w(1.2, rim + ALONG * PUSH_SLEEVE, CLOSED, "push it back against its spring", YAW_L))
    else:
        left.append(_w(1.2, rim, CLOSED, "(rests there without pushing it back)", YAW_L))
    t_sleeve = sum(w.duration for w in left)
    right = [_w(2.5, old + UP, OPEN, "over the 6 mm bit's tip end", YAW_R),
             _w(1.2, old, OPEN, "down: the pads either side of it", YAW_R),
             _w(0.8, old, OLD_PINCH, "pinch it", YAW_R)]
    right.append(_w(max(0.2, t_sleeve + 0.2 - sum(w.duration for w in right)), old, OLD_PINCH, "wait for the sleeve",
                    YAW_R))
    right += [_w(2.5, old - ALONG * PULL_BIT + [0.0, 0.0, LIFT_PULL], OLD_PINCH, "draw it straight out of the chuck",
                 YAW_R),
              _w(1.2, old - ALONG * PULL_BIT + [0.0, -ASIDE, LIFT_PULL], OLD_PINCH, "slide it aside, at the same height",
                 YAW_R),
              _w(1.0, old - ALONG * PULL_BIT + [0.0, -ASIDE, -L.AXIS_Z + L.SHANK_R + 0.003], OLD_PINCH, "lower it onto the bench",
                 YAW_R),
              _w(0.6, old - ALONG * PULL_BIT + [0.0, -ASIDE, -L.AXIS_Z + L.SHANK_R + 0.003], OPEN, "let go", YAW_R),
              _w(0.8, old - ALONG * PULL_BIT + [0.0, -ASIDE, 0.05], OPEN, "up", YAW_R)]
    t_out = sum(w.duration for w in right[:5])
    left += [_w(max(0.2, t_out - t_sleeve), left[-1].pos, CLOSED, "hold it back while the bit comes out", YAW_L),
             _w(0.6, rim, CLOSED, "let the sleeve come forward", YAW_L),
             _w(0.8, rim + UP_L, CLOSED, "up", YAW_L)]
    pin = hammer_pt([0.0, PIN_Y, L.AXIS_Z + L.NOSE_R + TIP - 0.015])  # pressing the gear case's nose down
    left += [_w(1.5, pin + UP_L, CLOSED, "over the housing, fingers closed"),
             _w(1.0, pin, CLOSED, "press the housing down onto the bench")]
    # the 10 mm bit
    grip = L.at(NEW_ORIGIN[0] - GRIP_FROM_END, L.NEW_START[1], L.SHANK_R + H_BENCH)
    roll = math.radians(TWIST) if variant != "no_twist" else 0.0
    pre, _ = holding(-0.015)
    at_keys, _ = holding(L.KEY_DEPTH - 0.004)
    twisted, yaw_t = holding(L.KEY_DEPTH - 0.002, roll)
    mid, _ = holding(MID_DEPTH, roll)
    up_t = tiltgrip.rx(roll) @ np.array([0.0, 0.0, 0.05])
    regrip, _ = holding(MID_DEPTH, 0.0, REGRIP_FROM_END)
    regrip = regrip - [0.0, 0.0, TIP_SAG]  # part-way in, the bit leans in the bore's play: its tip end hangs lower
    home, _ = holding(L.DEPTH + 0.004, 0.0, REGRIP_FROM_END)
    right += [_w(2.0, grip + UP, OPEN, "over the 10 mm bit's middle", YAW_R),
              _w(1.2, grip, OPEN, "down: the pads either side of it", YAW_R),
              _w(0.8, grip, NEW_PINCH, "pinch it", YAW_R),
              _w(1.5, grip + UP, NEW_PINCH, "lift it", YAW_R),
              _w(3.0, pre + [0.0, 0.0, 0.02], NEW_PINCH, "carry it to the chuck's axis, above", YAW_R),
              _w(1.0, pre, NEW_PINCH, "down onto the axis, shank 15 mm short of the nose", YAW_R),
              _w(1.2, at_keys, NEW_PINCH, "push it in up to the keys", YAW_R),
              _w(1.0, twisted, NEW_PINCH, "twist it while pushing: the slots find the keys", yaw_t),
              _w(1.0, mid, NEW_PINCH, "push it 25 mm in", yaw_t),
              _w(0.6, mid, OPEN, "let go: the bore holds it", yaw_t),
              _w(0.8, mid + up_t, OPEN, "up off it", yaw_t),
              _w(1.2, regrip + [0.0, 0.0, 0.04], OPEN, "over its tip end", YAW_R),
              _w(0.8, regrip, OPEN, "down: the pads either side of it", YAW_R),
              _w(0.6, regrip, NEW_PINCH, "pinch it near the tip", YAW_R),
              _w(1.5, home, NEW_PINCH, "push it home: it locks", YAW_R),
              _w(0.6, home, OPEN, "let go", YAW_R),
              _w(0.8, home + UP, OPEN, "back off", YAW_R),
              _w(1.5, PARK["right"], OPEN, "out to the side", YAW_R)]
    t_let_go = sum(w.duration for w in right[:-2])
    left += [_w(max(0.2, t_let_go - sum(w.duration for w in left)), pin, CLOSED, "hold it while the bit goes in"),
             _w(0.8, pin + UP_L, CLOSED, "up"),
             _w(1.5, PARK["left"], CLOSED, "out to the side")]
    return {"left": left, "right": right}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_twist   the 10 mm bit pushed straight in without the twist: it stops at the keys
      no_sleeve  the sleeve pinched but not drawn back: the 6 mm bit stays locked"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
