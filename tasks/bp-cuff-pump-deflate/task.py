"""On the station table, real size: a training arm lying across the table with an adult blood-pressure cuff wrapped on
its upper arm, the cuff's aneroid gauge facing the robot, and the inflation bulb lying in front of the left arm with
its air-release valve (a knurled thumbscrew on the bulb's tail, a quarter turn open) toward the robot's right.

Objects (mechkit.py next to this file): the training arm, cuff and gauge (one fixed mechanism: the gauge's needle turns
on a hinge) and the bulb (a loose mechanism: one half slides toward the other on a 20 N spring when squeezed; the
thumbscrew turns the valve on a damped hinge). One rule stands in for the air (scoring.Cuff): squeezing the bulb
pumps the cuff up, the open valve lets it down, and the needle shows the pressure. Nothing else is moved by code.

Goal: shut the valve, pump the cuff to 170 to 200 mmHg, then open the valve just enough that the needle falls 2 to 3 mmHg
each second from the peak down to 60, and open it wide to empty the cuff. The valve's flow grows with its opening and with the
pressure, so the hand has to open it a little further as the pressure falls. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=dump | overpump (see demo() below).
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
import tiltgrip  # noqa: E402  (importing it lets a waypoint carry a tilted grip)
from scoring import BULB, GAUGE, NEEDLE, SQUEEZE, VALVE, Cuff, Tracker, bulb_joints  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Take the inflation bulb, shut its valve, and pump the cuff up to between 170 and 200 mmHg on the gauge. "
               "Then open the valve just enough that the needle falls 2 to 3 mmHg each second, all the way down to 60, "
               "and then open it wide to empty the cuff.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _x_quat(angle: float) -> tuple:
    return _f(mk.matrix_to_quat(mk.axis_angle_matrix((1, 0, 0), angle)))


def needle_dir(p_mmhg: float = 0.0) -> np.ndarray:
    """The needle's direction (world, in the dial's y-z plane) at a pressure."""
    a = L.NEEDLE_ZERO + needle_angle(p_mmhg)
    return np.array([0.0, -math.sin(a), math.cos(a)])


def needle_angle(p_mmhg: float) -> float:
    return float(np.clip(p_mmhg, 0.0, 300.0)) / 300.0 * L.NEEDLE_SPAN


def _cuff_and_gauge():
    g = np.array([L.GAUGE["x"], L.GAUGE["y"], L.GAUGE["z"]])
    s = [
        mk.cylinder((L.ARM_X, L.UPPER_ARM["y"], L.UPPER_ARM["r"]), L.UPPER_ARM["r"], L.UPPER_ARM["l"], "Y", L.SKIN),
        mk.cylinder((L.ARM_X, L.FOREARM["y"], L.FOREARM["r"]), L.FOREARM["r"], L.FOREARM["l"], "Y", L.SKIN),
        mk.box((L.ARM_X, L.HAND["y"], L.HAND["size"][2] / 2), L.HAND["size"], L.SKIN),
        mk.cylinder((L.ARM_X, L.CUFF["y"], L.UPPER_ARM["r"]), L.CUFF["r"], L.CUFF["l"], "Y", L.CUFF_COLOUR),
        # the gauge, clipped to the cuff and turned to face the robot: case, dial face, clip
        mk.cylinder(_f(g + [0.010, 0, 0]), L.GAUGE["r"] + 0.004, 0.014, "X", L.DARK),
        mk.cylinder(_f(g + [0.002, 0, 0]), L.GAUGE["r"], 0.003, "X", L.DIAL),
        mk.box(_f((L.ARM_X - 0.030, L.GAUGE["y"], (g[2] + L.UPPER_ARM["r"]) / 2)), (0.050, 0.012, g[2] - L.UPPER_ARM["r"]),
               L.DARK),
    ]
    rr = L.GAUGE["r"]
    for k in range(31):  # ticks every 10 mmHg, longer every 50
        major = k % 5 == 0
        d = needle_dir(10.0 * k)
        a = L.NEEDLE_ZERO + needle_angle(10.0 * k)
        ln = 0.009 if major else 0.004
        s.append(mk.box(_f(g + (rr - 0.003 - ln / 2) * d + [-0.001, 0, 0]), (0.002, 0.0025 if major else 0.0012, ln),
                        L.DARK, quat=_x_quat(a)))
    d0 = needle_dir(0.0)
    needle = mk.hinge(NEEDLE, "X", _f(g + [-0.002, 0, 0]),
                      [mk.box(_f(g + [-0.002, 0, 0] + 0.016 * d0), (0.0015, 0.002, 0.036), L.NEEDLE_COLOUR,
                              quat=_x_quat(L.NEEDLE_ZERO)),
                       mk.cylinder(_f(g + [-0.002, 0, 0]), 0.004, 0.002, "X", L.DARK)],
                      upper=L.NEEDLE_SPAN + 0.1, lower=-0.1, mass=0.002, spring=0.2, damping=0.004)
    return mk.assembly(GAUGE, (0.0, 0.0, L.TOP), s, joints=[needle], color=L.CUFF_COLOUR)


def _bulb():
    w, h, l = L.BULB_W, L.BULB_H, L.BULB_L
    half = 0.020  # each half is 20 mm thick; 20 mm apart at rest, touching when squeezed flat
    s = [mk.box((-(w / 2 - half / 2), 0.0, 0.0), (half, l, h), L.RUBBER),
         mk.cylinder((0.0, -(l / 2 + 0.006), 0.0), 0.009, 0.012, "Y", L.RUBBER),  # the neck
         mk.cylinder((0.0, L.VALVE_STEM_Y + 0.006, 0.0), 0.007, 0.012, "Y", L.CHROME)]  # the valve body
    squeeze = mk.slider(SQUEEZE, "-X", [mk.box((w / 2 - half / 2, 0.0, 0.0), (half, l, h), L.RUBBER)],
                        upper=L.SQUEEZE_TRAVEL, mass=0.012, spring=L.SQUEEZE_SPRING, damping=2.0)
    valve = mk.hinge(VALVE, "Y", (0.0, L.VALVE_STEM_Y, 0.0),
                     [mk.cylinder((0.0, L.VALVE_STEM_Y - 0.005, 0.0), 0.0025, 0.010, "Y", L.CHROME),
                      mk.cylinder((0.0, L.SCREW_Y, 0.0), L.SCREW_R, L.SCREW_T, "Y", L.CHROME),
                      mk.box((0.0, L.SCREW_Y, 0.0), L.WING, L.CHROME)],
                     upper=L.VALVE_MAX - L.VALVE_START, lower=-L.VALVE_START, mass=0.004,
                     damping=L.VALVE_DAMPING)
    return mk.assembly(BULB, _f(L.BULB_START), s, joints=[squeeze, valve], mass=L.M_BULB - 0.016, color=L.RUBBER,
                       friction=L.BULB_FRICTION)


GAUGE_SPEC = _cuff_and_gauge()
BULB_SPEC = _bulb()
OBJECTS = [GAUGE_SPEC, BULB_SPEC]


# -- the air (the one rule) -----------------------------------------------------------------------------------------
_CUFF = Cuff()
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _check_episode() -> None:
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _CUFF.reset()
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()


@mk.on_step
def _air(states) -> None:
    """Squeezes pump the cuff up, the valve lets it down (scoring.Cuff); the needle shows the pressure."""
    _check_episode()
    j = bulb_joints(states)
    if j is None:
        return
    mk.set_targets(GAUGE_SPEC, NEEDLE, needle_angle(_CUFF.step(*j)))


# -- scoring plumbing -----------------------------------------------------------------------------------------------
@rigstate.on_reset
def _new_episode() -> None:
    _CUFF.reset()
    _TRACKER.reset()


def _r(v, n: int = 4) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """Pumped to 170-200, let down at 2-3 mmHg/s from the peak to 60, emptied; held 1 s. Runs every control step, so it
    also feeds the tracker."""
    _check_episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms, _CUFF.p)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **{k: v for k, v in result.details.items() if k != "windows"}}
    if result.details.get("windows"):
        row["last_window"] = result.details["windows"][-1]
    try:
        b = objects[BULB]
        row.update(squeeze_mm=round(float(b.joints[SQUEEZE]) * 1000, 2),
                   valve_deg=round(math.degrees(Cuff.opening(b.joints[VALVE])), 2), bulb=_r(b.origin),
                   valve_vel=round(float(b.joint_vel.get(VALVE, 0.0)), 3), bulb_q=_r(b.quat_xyzw, 3))
    except Exception as exc:  # noqa: BLE001
        row["objects_error"] = str(exc)[:200]
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp))
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 pumped to 170-200 and let down on rate, emptied; 0.5 near enough; 0.25 pumped up; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator. The left hand takes the bulb in a straight-down grip, the pads on its two
# halves, and holds it up in front of the cuff. The right hand comes in level along the bulb's axis from the robot's
# right, pinches the valve's wing thumbscrew on its flats and turns it shut (rolling the wrist), then lets go: each
# squeeze shifts the bulb in the left hand's symmetric grip. The left hand pumps (PUMPS squeezes). The right hand
# pinches the thumbscrew again and opens it to VALVE_PLAN's angles, worked out from the air rule so that the needle
# falls 2.5 mmHg a second: a little further open as the pressure falls. Below 60 it opens the valve wide.
OPEN = -0.04695
HOLD = -0.026  # the left hand's grip on the bulb: about 4 mm of squeeze (the fingers give a little), enough to hold it
SQUEEZE_SHUT = -0.012  # pressing the halves flat: past the 20 mm stroke, as the fingers give under the 20 N spring
PINCH = 0.0  # the right hand's pads closed hard on the thumbscrew's wing flats (8 mm apart)
GRIP_ABOVE = 0.008
LEFT_YAW = -math.pi / 2  # the left hand's pads close along x, on the bulb's halves
WORK = np.array([0.300, 0.060, L.TOP + 0.140])  # the bulb's centre while working
PUMPS = 18
VALVE_LEAD = math.radians(25.0)  # the valve ends this far short of the hand's turn as it opens (the wrist gives under
# the pinch; measured on the simulator), so each opening is commanded this much further
SHUT = -0.40  # turning it shut: well past the seat (rad), so the valve reaches it
PUMP_S = 0.35  # each half of a squeeze
TARGET_RATE = 2.5  # mmHg/s
VARIANTS = ("dump", "overpump")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    yaw = yaw if isinstance(yaw, tiltgrip.TiltedYaw) else float(yaw)
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, yaw)


def right_grip(theta: float) -> float:
    """The right hand's grip turning the valve to opening ``theta`` (rad): level along +y, the pads closing along x
    when the valve is a quarter turn open (wrist joint 6 near 0); rolling about +y turns the thumbscrew with it."""
    roll = theta - math.pi  # a quarter turn open -> roll -90 deg
    return tiltgrip.TiltedYaw(0.0, mk.axis_angle_matrix((0, 1, 0), roll) @ tiltgrip.rotation(0.0, math.pi / 2))


SCREW_DEPTH = 0.002  # the TCP this far short of the thumbscrew's centre: the screw 11 mm in from the pad tips


def screw_tcp(back: float = 0.0) -> np.ndarray:
    """The right TCP pinching the thumbscrew with the bulb at WORK (``back`` m further back along -y)."""
    return WORK + np.array([0.0, L.SCREW_Y - SCREW_DEPTH - back, 0.0])


def valve_plan(start_p: float, t0: float) -> list[tuple[float, float]]:
    """(time, opening) every second from ``t0``: the opening that lets the cuff down at TARGET_RATE from ``start_p``
    to below 60 (the air rule, run forward)."""
    out, p, t = [], start_p, t0
    while p > 55.0:
        theta = TARGET_RATE / (L.LEAK * math.sqrt(p))  # linear below the knee
        out.append((t, theta))
        p -= TARGET_RATE
        t += 1.0
    return out


def expected_peak(pumps: int = PUMPS, stroke_mm: float = 16.0) -> float:
    p = 0.0
    for _ in range(pumps):
        p += L.PUMP_PER_MM * stroke_mm * (1.0 - p / L.PUMP_CEILING)
    return p


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    pumps = 22 if variant == "overpump" else PUMPS
    b, up = L.BULB_REST, np.array([0.0, 0.0, 0.06])
    lt = lambda c: np.asarray(c) + [0.0, 0.0, GRIP_ABOVE]  # noqa: E731
    left = [
        _w(2.5, lt(b) + up, OPEN, "over the bulb", LEFT_YAW),
        _w(1.2, lt(b), OPEN, "down: the pads either side of the bulb", LEFT_YAW),
        _w(0.8, lt(b), HOLD, "take hold of the bulb", LEFT_YAW),
        _w(1.5, lt(WORK), HOLD, "lift it in front of the cuff", LEFT_YAW),
    ]
    right = [
        _w(3.0, screw_tcp(0.08) + [0, 0, 0.05], OPEN, "beside the valve's thumbscrew", right_grip(L.VALVE_START)),
        _w(3.0, screw_tcp(0.08) + [0, 0, 0.05], OPEN, "wait for the bulb", right_grip(L.VALVE_START)),
        _w(1.0, screw_tcp(0.05), OPEN, "level with the thumbscrew", right_grip(L.VALVE_START)),
        _w(1.0, screw_tcp(), OPEN, "in along the bulb's axis, a pad either side", right_grip(L.VALVE_START)),
        _w(0.5, screw_tcp(), PINCH, "pinch the thumbscrew", right_grip(L.VALVE_START)),
        _w(1.2, screw_tcp(), PINCH, "turn it shut", right_grip(SHUT)),
        _w(0.4, screw_tcp(), PINCH, "hold it shut", right_grip(SHUT)),
        _w(0.4, screw_tcp(), OPEN, "let go", right_grip(0.0)),
        _w(0.6, screw_tcp(0.04), OPEN, "back off while the bulb is pumped", right_grip(0.0)),
    ]
    t_shut = sum(w.duration for w in right)
    t = sum(w.duration for w in left)
    left.append(_w(t_shut - t, lt(WORK), HOLD, "hold the bulb up", LEFT_YAW))
    for i in range(pumps):
        left += [_w(PUMP_S, lt(WORK), SQUEEZE_SHUT, f"pump {i + 1}: squeeze", LEFT_YAW),
                 _w(PUMP_S, lt(WORK), HOLD, f"pump {i + 1}: let it fill", LEFT_YAW)]
    t_pumped = sum(w.duration for w in left)
    right += [_w(t_pumped - t_shut + 0.2, screw_tcp(0.04), OPEN, "wait", right_grip(0.0)),
              _w(0.8, screw_tcp(), OPEN, "back in on the thumbscrew", right_grip(0.0)),
              _w(0.5, screw_tcp(), PINCH, "pinch it again", right_grip(0.0))]
    t = sum(w.duration for w in right)
    peak = expected_peak(pumps)
    if variant == "dump":  # open the valve a long way at once: the needle drops far faster than 3 mmHg/s
        right += [_w(0.6, screw_tcp(), PINCH, "open it far at once", right_grip(math.radians(150))),
                  _w(30.0, screw_tcp(), PINCH, "hold", right_grip(math.radians(150)))]
    else:
        # crack it straight to the opening for 2.5 mmHg/s at 160 (turning one way only: the wrist's give depends on the
        # direction), let it down to 160, then open it a little further each second
        start = TARGET_RATE / (L.LEAK * math.sqrt(160.0)) + VALVE_LEAD
        right.append(_w(0.6, screw_tcp(), PINCH, "crack the valve open", right_grip(start)))
        t += 0.6
        t160 = t + max(0.0, (peak - 160.0) / (TARGET_RATE * math.sqrt(peak / 160.0)))
        plan = valve_plan(160.0, t160)
        right.append(_w(t160 - t, screw_tcp(), PINCH, "let it down to 160", right_grip(start)))
        for k, (_, theta) in enumerate(plan):
            right.append(_w(1.0, screw_tcp(), PINCH, f"let down: {160 - 2.5 * k:.0f} mmHg",
                            right_grip(theta + VALVE_LEAD)))
        right += [_w(0.6, screw_tcp(), PINCH, "open it wide to empty the cuff", right_grip(L.VALVE_MAX - 0.05)),
                  _w(6.0, screw_tcp(), PINCH, "hold while the cuff empties", right_grip(L.VALVE_MAX - 0.05))]
    right += [_w(0.4, screw_tcp(), OPEN, "let go of the thumbscrew", right_grip(L.VALVE_MAX - 0.05)),
              _w(1.0, screw_tcp(0.08), OPEN, "back away", right_grip(L.VALVE_MAX - 0.05))]
    left.append(_w(sum(w.duration for w in right) - sum(w.duration for w in left) + 2.0, lt(WORK), HOLD,
                   "hold the bulb", LEFT_YAW))
    return {"left": left, "right": right}


def demo():
    """Both arms: the left holds and pumps the bulb, the right works its valve. DEMO_HOLD_ONLY=1: no plan (a settling
    check). DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      dump      after pumping, the valve opened far at once: the needle drops several times too fast
      overpump  22 squeezes: the cuff goes past 200"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
