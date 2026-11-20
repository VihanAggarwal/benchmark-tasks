"""On the station desk, real size: a Logitech K120 wired keyboard (450 x 155 x 23.5 mm, 550 g, full-size US layout),
loose on its rubber feet, front edge toward the robot, Left Ctrl at the robot's left. Behind it a 24 in monitor shows
the Windows 11 lock screen ("Press Ctrl+Alt+Delete to unlock": the PC's "Interactive logon: Do not require
CTRL+ALT+DEL" policy is Disabled, as many offices set it). A sticky note on the bezel gives the PIN, 2580.

Goal: unlock the PC: hold Ctrl and Alt down together, strike Delete while both are still down, let the keys up, type
the PIN on the number pad and press Enter, so the desktop shows; every key up and both hands off at the end. Left
Ctrl and Left Alt are split by the Windows key and Delete is 29 cm away, so one gripper's two fingertips hold Ctrl
and Alt (47.6 mm apart, the Windows key clear between them) while the other arm strikes Delete.

The keyboard is one mechkit articulation (mechkit.py, next to this file): the case (loose) with one sprung cap on a
slider for every key near the targets (bottom row, navigation cluster, number pad); the rest of the keys are fixed
caps. Keys are pressed by real fingertip contact. Declared rules: the key springs' preload (a drive target above the
rest stop) and the PC's response (scoring.PC: screen, PIN entry), drawn on the monitor by sliding coloured plates
out of the panel. The arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals: DEMO_VARIANT=<name>, see VARIANTS.
Trace:   TASK_TRACE=<file> (one JSON line per control step).
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
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import KEYBOARD  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Sign in to your workstation's test account: press Ctrl+Alt+Delete, type the account's PIN, 2580, and press Enter."



# -- legends: seven-segment strokes (thin white boxes) ---------------------------------------------------------------
SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg", "C": "adef", "t": "defg", "r": "eg", "L": "def", "A": "abcefg", "d": "bcdeg",
       "E": "adefg", "n": "ceg", "P": "abefg", "I": "bc", "+": "gv"}


def glyphs(text, center, h, plane="top", color=L.LEGEND_WHITE, depth=0.0006):
    """Seven-segment legend boxes for ``text`` centred at ``center`` (body frame). Reading frame: u to the reader's
    right, v up. plane "top": a key top read from the robot (u = -y, v = +x); "front": a face toward -x (u = -y,
    v = +z)."""
    w, t = 0.55 * h, 0.12 * h
    pitch = w + 0.35 * h
    u0 = -pitch * (len(text) - 1) / 2
    out = []
    for i, ch in enumerate(text):
        cu = u0 + i * pitch
        segs = {"a": (cu, h / 2, w, t), "d": (cu, -h / 2, w, t), "g": (cu, 0.0, w, t),
                "b": (cu + w / 2, h / 4, t, h / 2), "c": (cu + w / 2, -h / 4, t, h / 2),
                "e": (cu - w / 2, -h / 4, t, h / 2), "f": (cu - w / 2, h / 4, t, h / 2),
                "v": (cu, 0.0, t, w)}
        for k in SEG[ch]:
            su, sv, lu, lv = segs[k]
            if plane == "top":
                out.append(mk.box((center[0] + sv, center[1] - su, center[2]), (lv, lu, depth), color))
            else:
                out.append(mk.box((center[0], center[1] - su, center[2] + sv), (depth, lu, lv), color))
    return out


def win_logo(center, size=0.006, color=L.LEGEND_WHITE):
    q = size * 0.45
    return [mk.box((center[0] + sx * size / 4, center[1] + sy * size / 4, center[2]), (q, q, 0.0006), color)
            for sx in (-1, 1) for sy in (-1, 1)]


LEGENDS = {"lctrl": "CtrL", "rctrl": "CtrL", "lalt": "ALt", "ralt": "ALt", "delete": "dEL", "end": "End",
           "npenter": "Ent", **{k: v for k, v in L.DIGIT_KEYS.items()}}


def key_legend(name, x, y):
    z = L.CAP_TOP - 0.0002  # 0.6 mm strokes, 0.1 mm proud of the cap
    if name in L.WIN_KEYS:
        return win_logo((x, y, z))
    text = LEGENDS.get(name)
    if not text:
        return []
    h = 0.008 if len(text) == 1 else 0.0055
    return glyphs(text, (x, y, z), h)


# -- the keyboard ---------------------------------------------------------------------------------------------------
def _cap(x, y, hx, hy, color=L.CAP_BLACK):
    return mk.box((x, y, L.CASE_Z1 + L.CAP_H / 2), (2 * hx, 2 * hy, L.CAP_H), color)


def case_shapes():
    out = [mk.box((0.0, 0.0, (L.CASE_Z0 + L.CASE_Z1) / 2), (L.KB_D, L.KB_L, L.CASE_Z1 - L.CASE_Z0), L.KB_BLACK)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(mk.cylinder((sx * (L.KB_D / 2 - 0.015), sy * (L.KB_L / 2 - 0.025), L.FOOT_H / 2), 0.006,
                                   L.FOOT_H, color=L.FEET_RUBBER))
    half = (L.U - L.CAP_GAP) / 2
    for row, keys in L.STATIC_ROWS.items():
        for col, w in keys:
            x, y = L.key_xy(col, row)
            out.append(_cap(x, y, half, (w * L.U - L.CAP_GAP) / 2))
    for col, row in L.STATIC_NAV + L.STATIC_NUMPAD:
        x, y = L.key_xy(col, row)
        out.append(_cap(x, y, half, half))
    x, y = L.key_xy(*L.NP_PLUS)
    out.append(_cap(x, y, (2 * L.U - L.CAP_GAP) / 2, half))
    for col in L.FKEYS:
        x, y = L.key_xy(col, 6)
        out.append(_cap(x + L.FKEY_GAP, y, half, half))
    # three white indicator dots above the number pad (Num Lock lit)
    x, y = L.key_xy(19.0, 6)
    out.append(mk.box((x + L.FKEY_GAP, y, L.CASE_Z1 + 0.0005), (0.004, 0.004, 0.001), (0.3, 0.9, 0.4)))
    return out


def key_joints():
    out = []
    for name, (col, row, w, h) in L.KEYS.items():
        x, y = L.key_xy(col, row)
        hx, hy = (h * L.U - L.CAP_GAP) / 2, (w * L.U - L.CAP_GAP) / 2
        out.append(mk.slider(name, "-Z", [_cap(x, y, hx, hy)] + key_legend(name, x, y), upper=L.KEY_TRAVEL, lower=0.0,
                             mass=L.KEY_MASS, spring=L.KEY_SPRING, damping=L.KEY_DAMPING))
    return out


KEYBOARD_PART = mk.assembly(KEYBOARD, origin=(L.KB_X, L.KB_Y, L.TOP), mass=L.KB_MASS, shapes=case_shapes(),
                            joints=key_joints(), color=L.KB_BLACK, friction=L.KB_FRICTION, tight_fit=True)


FRAME_INK = (0.05, 0.05, 0.25)


# -- the monitor -------------------------------------------------------------------------------------------------------
def monitor_shapes():
    cz = L.MON_Z0 + L.MON_H / 2
    return [
        mk.box((0.0, 0.0, 0.005), (0.18, 0.22, 0.010), L.FRAME_BLACK),
        mk.box((0.02, 0.0, (0.010 + L.MON_Z0) / 2 + 0.02), (0.03, 0.05, L.MON_Z0 + 0.03), L.FRAME_BLACK),
        mk.box((-0.040 + L.MON_T / 2, 0.0, cz), (L.MON_T, L.MON_W, L.MON_H), L.FRAME_BLACK),
        # the sticky note with the PIN on the bezel's lower right (yellow)
        mk.box((-0.041, -0.22, L.MON_Z0 + 0.012), (0.001, 0.06, 0.02), (0.98, 0.90, 0.30)),
    ] + glyphs(L.PIN, (-0.0418, -0.22, L.MON_Z0 + 0.012), 0.010, plane="front", color=FRAME_INK)


def _plate(name, color, extra=()):
    cz = L.MON_Z0 + L.MON_H / 2
    shapes = [mk.box((-0.040 + 0.006, 0.0, cz), (0.002, L.MON_W - 0.04, L.MON_H - 0.04), color)] + list(extra)
    return mk.slider(name, "-X", shapes, upper=L.PLATE_SHOW, lower=0.0, mass=0.01, spring=200.0, damping=5.0)


def screen_joints():
    cz = L.MON_Z0 + L.MON_H / 2
    lock = _plate("show_lock", L.SCREEN_LOCK, glyphs("CtrL+ALt+dEL", (-0.040 + 0.0045, 0.0, cz - 0.04), 0.022,
                                                     plane="front", color=L.WHITE, depth=0.002))
    prompt = _plate("show_prompt", L.SCREEN_PROMPT, glyphs("PIn", (-0.040 + 0.0045, 0.0, cz + 0.03), 0.020,
                                                           plane="front", color=L.WHITE, depth=0.002) + [mk.box((-0.040 + 0.0045, 0.0, cz - 0.02), (0.002, 0.16, 0.02),
                                                            L.WHITE)])
    desk = _plate("show_desktop", L.SCREEN_DESKTOP, [mk.box((-0.040 + 0.0045, 0.22 - 0.03 * i, cz + 0.10),
                                                            (0.002, 0.02, 0.02), L.WHITE) for i in range(3)])
    dots = []
    for i in range(4):
        dots.append(mk.slider(f"dot{i + 1}", "-X", [mk.box((-0.040 + 0.002, -0.045 + 0.03 * i, cz - 0.02),
                                                           (0.002, 0.012, 0.012), L.FRAME_BLACK)],
                              upper=L.PLATE_SHOW + 0.004, lower=0.0, mass=0.005, spring=200.0, damping=5.0))
    return [lock, prompt, desk] + dots


MONITOR_PART = mk.assembly("monitor", origin=(L.MON_X, L.MON_Y, L.TOP), shapes=monitor_shapes(),
                           joints=screen_joints(), color=L.FRAME_BLACK)
OBJECTS = [MONITOR_PART, KEYBOARD_PART]


# -- the declared rules --------------------------------------------------------------------------------------------
def screen_targets(pc) -> dict:
    shown = {"show_lock": pc.screen == S.LOCK, "show_prompt": pc.screen == S.PROMPT,
             "show_desktop": pc.screen == S.DESKTOP}
    out = {k: (L.PLATE_SHOW if v else 0.0) for k, v in shown.items()}
    n = len(pc.typed) if pc.screen == S.PROMPT else 0
    for i in range(4):
        out[f"dot{i + 1}"] = L.PLATE_SHOW + 0.004 if i < n else 0.0
    return out


@mk.on_step
def _rules(states) -> None:
    for name in L.KEYS:
        mk.set_targets(KEYBOARD_PART, name, -L.KEY_PRELOAD)  # the dome's preload: pushes the cap up on its stop
    for name, v in screen_targets(_TRACKER.pc).items():
        mk.set_targets(MONITOR_PART, name, v)


# -- the offline stand-in ----------------------------------------------------------------------------------------------
# Offline only: a key's cap goes down by however far a fingertip's pad is below the cap's top while the fingertip's
# footprint overlaps the cap by 3 mm or more each way. Fingertips: 6 mm thick along the closing axis, 15 mm wide across
# it, the two at the TCP +- (gap / 2 + 3 mm) along the closing axis. Each arm's yaw is the demo's (left closes along y,
# right along x): rigstate does not report yaw.
CLOSE_AXIS = {"left": np.array([0.0, 1.0]), "right": np.array([1.0, 0.0])}
FINGER_T, FINGER_W = 0.006, 0.015
OVERLAP = 0.003


def fingertips(side, tcp, tips, opening):
    """[(centre xy, half extents along (x, y)), ...] of the two fingertips, and the pad bottom height (table z 0)."""
    a = CLOSE_AXIS[side]
    gap = max(0.0, float(opening)) * rigstate.FULL_GAP
    c = np.asarray(tcp[:2], dtype=float)
    half = np.where(a > 0.5, FINGER_T / 2, FINGER_W / 2)
    out = [(c + s * a * (gap / 2 + FINGER_T / 2), half) for s in (-1, 1)]
    return out, float(tips[2]) - L.TOP


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.travel = {k: 0.0 for k in L.KEYS}

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        travel = {k: 0.0 for k in L.KEYS}
        if read is not None:
            for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
                prints, z = fingertips(side, tcp, tips, opening)
                depth = L.CAP_TOP - z
                if depth <= 0:
                    continue
                for k in L.KEYS:
                    cx, cy = L.key_world(k)
                    hx, hy = L.key_half(k)
                    for c, h in prints:
                        ox = min(c[0] + h[0], cx + hx) - max(c[0] - h[0], cx - hx)
                        oy = min(c[1] + h[1], cy + hy) - max(c[1] - h[1], cy - hy)
                        if ox >= OVERLAP and oy >= OVERLAP:
                            travel[k] = max(travel[k], min(depth, L.KEY_TRAVEL))
        self.travel = travel

    def states(self) -> dict:
        return {KEYBOARD: mk.state_from_poses(KEYBOARD_PART, (L.KB_X, L.KB_Y, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                              joints=dict(self.travel), source="offline")}


_OFFLINE = OfflineModel()
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(KEYBOARD), "source", "sim") != "sim"


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """The desktop shown after Ctrl+Alt+Delete and the right PIN, no Windows key, no stray key, every key up, both
    grippers 30 mm clear and the keyboard still for 1 s; held 1 s. Feeds the tracker every control step."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {KEYBOARD: objects.get(KEYBOARD)}
    arms = rigstate.arms()
    if no_simulator(parts):
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, **result.details}
    try:
        row["travel_mm"] = {k: round(float(parts[KEYBOARD].joints[k]) * 1000, 2) for k in
                            ("lctrl", "lwin", "lalt", "delete", "np2", "np5", "np8", "np0", "npenter")}
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp)
        except (AttributeError, KeyError, TypeError):
            pass
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 after a lifted or tilted keyboard; 0.25 Ctrl+Alt held together 0.5 s; 0.5 the PIN prompt (or any
    stray key, a Windows key included, or a wrong PIN); 0.75 the PIN typed right; 1.0 the desktop with all keys up and hands off."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
PAD_TIP = 0.013
OPEN = -0.04695
CLOSED = 0.0
PAIR_PITCH = L.key_world("lctrl")[1] - L.key_world("lalt")[1]  # 47.6 mm, Ctrl to Alt centres
HOLD_GAP = PAIR_PITCH - FINGER_T  # 41.6 mm between the pads: fingertip centres on the two cap centres
HOLD_F = -HOLD_GAP / 2
YAW_L, YAW_R = 0.0, math.pi / 2  # left closes along y (the row), right along x
# Commanded 8 mm under the cap tops, as on the GPU-proven conveyor STOP and centrifuge tab (8 and 10 mm): the job
# 20261120T071600 run commanded 3 mm and no key moved at all (every travel 0.0 mm). The cap's 2.8 mm bottom stop
# takes the rest, so the arm's give, not the key, absorbs the overshoot.
PRESS = 0.0080
PRESS_TCP = L.CAP_TOP - PRESS + PAD_TIP
ABOVE = L.CAP_TOP + 0.012 + PAD_TIP  # pad tips 12 mm over the caps
HOVER = 0.080
PARK = {"left": (0.20, 0.27, 0.15), "right": (0.20, -0.27, 0.15)}


def _w(d, xy, z, finger, label, yaw):
    return Waypoint(d, (float(xy[0]), float(xy[1]), L.TOP + z), finger, label, yaw)


def _dur(plan) -> float:
    return sum(w.duration for w in plan)


def press_moves(key, label, yaw):
    xy = L.key_world(key)
    return [
        _w(1.0, xy, ABOVE, CLOSED, f"over {label}", yaw),
        _w(0.8, xy, PRESS_TCP, CLOSED, f"press {label}", yaw),
        _w(0.6, xy, PRESS_TCP, CLOSED, "hold", yaw),
        _w(0.7, xy, ABOVE, CLOSED, f"let {label} up", yaw),
    ]


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    mid = ((L.key_world("lctrl")[0] + L.key_world("lalt")[0]) / 2,
           (L.key_world("lctrl")[1] + L.key_world("lalt")[1]) / 2)  # over the Windows key
    left = [
        _w(2.0, mid, HOVER, OPEN, "over Ctrl and Alt", YAW_L),
        _w(1.5, mid, HOVER, HOLD_F, "open to the Ctrl-Alt pitch", YAW_L),
        _w(1.5, mid, ABOVE, HOLD_F, "down over the caps", YAW_L),
        _w(1.0, mid, PRESS_TCP, HOLD_F, "press Ctrl and Alt", YAW_L),
    ]
    held = _dur(left)
    right = [_w(held - 0.5, L.key_world("delete"), HOVER, CLOSED, "over Delete, fingers closed", YAW_R)]
    order = variant != "delete_first"
    if variant == "delete_first":  # Delete struck before Ctrl and Alt go down: nothing happens
        right = [_w(1.0, L.key_world("delete"), HOVER, CLOSED, "over Delete, fingers closed", YAW_R)]
    right += [_w(0.8 if order else 0.5, L.key_world("delete"), ABOVE, CLOSED, "just above Delete", YAW_R)]
    right += press_moves("delete", "Delete", YAW_R)[1:]
    struck = _dur(right)
    left.append(_w(max(struck - held + 0.3, 1.0), mid, PRESS_TCP, HOLD_F, "hold Ctrl and Alt while Delete goes down",
                   YAW_L))
    left += [
        _w(0.8, mid, ABOVE, HOLD_F, "let Ctrl and Alt up", YAW_L),
        _w(1.2, PARK["left"][:2], PARK["left"][2], OPEN, "aside", YAW_L),
    ]
    pin = {"": L.PIN, "wrong_pin": "2589", "no_enter": L.PIN, "delete_first": L.PIN}.get(variant, L.PIN)
    for d in pin:
        key = [k for k, v in L.DIGIT_KEYS.items() if v == d][0]
        right += press_moves(key, d, YAW_R)
    if variant != "no_enter":
        right += press_moves("npenter", "Enter", YAW_R)
    right += [_w(1.2, PARK["right"][:2], PARK["right"][2], OPEN, "aside", YAW_R)]
    return {"left": left, "right": right}


VARIANTS = ("wrong_pin", "no_enter", "delete_first")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan. DEMO_IDLE_ARM=left|right drops that arm's plan.
    DEMO_VARIANT: wrong_pin (2589 then Enter: 0.5), no_enter (PIN typed, Enter not pressed: 0.75 at most, hands
    off), delete_first (Delete struck before Ctrl and Alt are down: the lock screen stays, 0.25)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
