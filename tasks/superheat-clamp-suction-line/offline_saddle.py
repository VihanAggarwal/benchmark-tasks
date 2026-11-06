"""Offline checker only: where the clamp and the cover really come to rest. The simulator never calls anything in
this file (its states say source "sim").

The offline kinematic check (NumPy only) has no physics and sees each mechkit part as the box
around all its shapes. When a gripper lets go of a part, the checker drops that box straight down until its bottom
rests on the top of the highest thing under it (within 2 cm of its bottom). For these two U-shaped parts that is
wrong in a known way:

- the clamp's box bottom is its legs' bottom, 20 mm below the top block's underside. Let go astride a line, the box
  is set with its bottom on the line's top, 20 mm too high: the legs really pass either side of the line and the
  underside rests on its top;
- the cover's box bottom is its walls' bottom, 36 mm below its ceiling. Let go over the clamp, the box is set on
  the top of the line inside its channel (or on the clamp's box): the walls really pass either side, and the
  ceiling rests on the highest thing under it, normally the clamp's display.

So when task.py's hooks get states whose ``source`` is "offline", ``correct()`` applies these two rules and the SAME
scorer runs on the result:

1. the clamp's box bottom within 3 mm of the top of a line that fits between its legs (the line's half width plus
   the clamp origin's offset from the line's axis at most 16.5 mm, the legs' inner faces), its origin over that
   line's length: its origin goes down onto that line's top;
2. the cover's box bottom within 3 mm of the top of a line inside its channel (half width plus offset at most
   32.5 mm) or of the clamp's box top: its ceiling goes onto the highest of those lines' tops and, if the clamp's
   display lies under its roof, the (corrected) display's top.

No sag, slip, tipping or contact forces: those are for the GPU runs.
"""

from __future__ import annotations

import numpy as np

import layout as L
import mechkit as mk

_TOP = L.TOP


def _over(y: float, y0: float, y1: float) -> bool:
    return y0 <= y <= y1


def _rebuild(spec, s, z_world: float):
    """The same part state with its origin at a new world height."""
    o = np.asarray(s.origin, dtype=float).ravel()
    q = np.asarray(s.quat_xyzw, dtype=float).ravel()
    v = np.concatenate([np.asarray(s.lin_vel, dtype=float).ravel(), np.asarray(s.ang_vel, dtype=float).ravel()])
    return mk.state_from_poses(spec, (o[0], o[1], z_world, *q), v, source="offline")


def clamp_rest(clamp) -> tuple[float | None, str | None]:
    """Rule 1: the height (above the table) the clamp's origin really rests at, and on what; (None, None) if the rule
    does not apply."""
    o = np.asarray(clamp.origin, dtype=float).ravel()
    bottom = float(clamp.low[2]) - _TOP
    for name, (ax, hw, top, y0, y1) in L.STRADDLE_LINES.items():
        if (abs(o[0] - ax) + hw <= L.LEG_INNER + 1e-9 and _over(o[1], y0, y1)
                and abs(bottom - top) <= L.OFFLINE_DROP_TOL):
            return top, name
    return None, None


def cover_rest(cover, clamp_proxy, clamp_now) -> tuple[float | None, str | None]:
    """Rule 2: the height (above the table) the cover's origin really rests at, and on what."""
    o = np.asarray(cover.origin, dtype=float).ravel()
    bottom = float(cover.low[2]) - _TOP
    y0c, y1c = o[1] - L.COVER_L / 2, o[1] + L.COVER_L / 2
    tops = {}
    for name, (ax, hw, top, y0, y1) in L.CHANNEL_LINES.items():
        if abs(o[0] - ax) + hw <= L.CHANNEL_HALF + 1e-9 and y0 < y1c and y0c < y1:
            tops[name] = top
    dropped = any(abs(bottom - t) <= L.OFFLINE_DROP_TOL for t in tops.values())
    if clamp_proxy is not None and abs(bottom - (float(clamp_proxy.high[2]) - _TOP)) <= L.OFFLINE_DROP_TOL:
        dropped = True
    if not dropped:
        return None, None
    if clamp_now is not None:
        c = np.asarray(clamp_now.origin, dtype=float).ravel()
        if (abs(c[0] - o[0]) < (L.COVER_W + L.DISPLAY_S[0]) / 2
                and abs(c[1] - o[1]) < (L.COVER_L + L.DISPLAY_S[1]) / 2):
            tops["clamp_display"] = c[2] - _TOP + L.DISPLAY_TOP
    if not tops:
        return None, None
    name = max(tops, key=tops.get)
    return tops[name] - L.CEILING, name


def correct(clamp, cover, specs: dict):
    """(clamp, cover, info): the two offline states moved to where they really rest. ``specs`` holds the two
    mechkit Assemblies by name. States that are not from the offline checker, or broken, are returned as they are."""
    info = {"clamp_on": None, "cover_on": None}
    if getattr(clamp, "source", None) != "offline" or getattr(cover, "source", None) != "offline":
        return clamp, cover, info
    try:
        proxy_clamp = clamp
        z, on = clamp_rest(clamp)
        if z is not None:
            clamp = _rebuild(specs["clamp"], clamp, _TOP + z)
            info["clamp_on"] = on
        z, on = cover_rest(cover, proxy_clamp, clamp)
        if z is not None:
            cover = _rebuild(specs["cover"], cover, _TOP + z)
            info["cover_on"] = on
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        info["error"] = True
    return clamp, cover, info
