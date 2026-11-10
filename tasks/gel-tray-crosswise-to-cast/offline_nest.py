"""Offline checker only: where the comb really comes to rest when it is let go into the tray. The simulator never
calls anything in this file (its states say source "sim").

The offline kinematic check (NumPy only) has no physics and sees each mechkit part as the box around all its shapes.
When a gripper lets go of a part, the checker drops that box straight down until its bottom rests on the top of the
highest thing under it (within 2 cm of its bottom). For the comb let go into the tray that is wrong in a known way:
the comb's box bottom is its teeth's bottom, 7 mm below its bar, and the tray's box top is its side walls' top, 5 mm
above the notch bottoms. So the checker sets the comb with its teeth on the tray's wall tops, 12 mm too high: in fact
the teeth hang down inside the tray and the bar rests in the notches of both side walls (or, away from a slot, on the
wall tops).

So when task.py's hooks get states whose ``source`` is "offline", ``correct()`` applies this one rule and the SAME
scorer runs on the result: a comb whose box bottom is within 3 mm of the tray box's top, upright, across the tray
(within 10 degrees), its middle within 6 mm of the tray's centre line (teeth between the side walls), goes down so its
bar rests in a slot's notches if, at both side walls, its bar lies wholly within that slot (within 5 mm of its centre:
the 12 mm bar in the 22 mm notch), and otherwise so its bar rests on the side walls' tops. Its position across the
table and its orientation are kept. No sag, slip, tipping or contact forces: those are for the GPU runs.
"""

from __future__ import annotations

import numpy as np

import layout as L
import mechkit as mk
import scoring as S

FIT = (L.NOTCH_W - L.COMB_BAR[0]) / 2  # 5 mm: the bar lies wholly within the notch


def comb_rest(comb, tray) -> tuple[float | None, str | None]:
    """The height above the tray's underside (in the tray's frame) the comb's origin really rests at, and on what;
    (None, None) if the rule does not apply."""
    if abs(float(comb.low[2]) - float(tray.high[2])) > L.OFFLINE_DROP_TOL:
        return None, None
    w = S.comb_in_tray(comb, tray)
    if (w["tilt_deg"] > L.COMB_TILT_DEG or w["heading_deg"] > L.COMB_HEADING_DEG
            or abs(w["mid_y"]) > L.COMB_Y_TOL):
        return None, None
    for name, sx in L.SLOTS.items():
        if all(abs(x - sx) <= FIT for x in w["seat_x"]):
            return L.NOTCH_BOTTOM, name
    return L.TRAY_H, "wall_tops"


def correct(tray, comb, spec):
    """(comb, info): the offline comb state moved to where it really rests in the tray. ``spec`` is the comb's mechkit
    Assembly. States that are not from the offline checker, or broken, are returned as they are."""
    info = {"comb_on": None}
    if getattr(tray, "source", None) != "offline" or getattr(comb, "source", None) != "offline":
        return comb, info
    try:
        if not (S.valid(tray) and S.valid(comb)):
            return comb, info
        h, on = comb_rest(comb, tray)
        if h is None:
            return comb, info
        local = S.to_local(tray, comb.origin)
        local[2] = h
        o = S.to_world(tray, local)
        q = np.asarray(comb.quat_xyzw, dtype=float).ravel()
        v = np.concatenate([np.asarray(comb.lin_vel, dtype=float).ravel(),
                            np.asarray(comb.ang_vel, dtype=float).ravel()])
        comb = mk.state_from_poses(spec, (*o, *q), v, source="offline")
        info["comb_on"] = on
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        info["error"] = True
    return comb, info
