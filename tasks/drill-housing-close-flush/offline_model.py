"""Offline only (tools/offline/check_task.py, no simulator): a kinematic model of the three loose parts, so the
offline checker scores the demo against the parts' real shapes instead of their bounding boxes.

The checker's own proxy sees each compound part as one bounding box, so the unit's box lands on the nose wall
(its spindle passes through the notch) and the switch's box on the handle's front wall (its trigger passes through
the slot). This model replaces those boxes with the parts' own shapes (scoring.SHAPES) and the fixed boxes of the
left half, the nest, the tray and the table (layout.py), and does only three things:

- grasp: a gripper whose pads close from wider than a part to narrower than it, with the part between the pads
  (the pads span from the pad-tip point up 38 mm), takes hold of it; every grip in this demo closes along world y
  (yaw 0), which is the axis used;
- carry: a held part keeps its offset from the pad-tip point (from the mean of both arms' when two hold it), except
  that it cannot be pushed down through a support (it stops on it, as if slipping in the grip); it keeps its
  orientation (every part starts square; the wrist turn the demo adds at the unit's place cancels the turn the GPU
  grip gave it); sideways collisions are not modelled; near a trimmed place point the part is offset by the arm
  error the GPU trace measured there (``arm_error``, set by task.py: minus the demo's trim), so the prediction
  stays on what the GPU arm does with the trimmed command;
- release: a part let go by its last holder drops straight down until one of its shapes meets a fixed box, the
  table or another loose part that is not held (exact for boxes and for cylinders lying along x over a box's top
  face or edge, and boxes over such a cylinder). There is no sliding, tipping or pushing: a part let go off its
  seat stays off it.

It never changes what the simulator does; in the simulator task.py ignores it.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
from scoring import SHAPES

PAD_SPAN_UP = 0.038  # the pads cover the pad-tip point's height up to 38 mm above it (tools/offline/check_task.py)


def _world_box(b, origin=(0.0, 0.0, 0.0)):
    o = np.asarray(origin, dtype=float)
    return (o + np.array([b[0][0], b[1][0], b[2][0]]), o + np.array([b[0][1], b[1][1], b[2][1]]))


_O = np.array(L.world((0.0, 0.0, 0.0)))
FIXED = [_world_box(b, _O) for b in (*L.LOWER.values(), *L.NEST.values())]
FIXED += [(np.array([x0, y0, L.TOP]), np.array([x1, y1, L.TOP + L.TRAY_H])) for (x0, x1), (y0, y1) in L.TRAY_RAILS]


def _bounds(sh):
    if sh[0] == "box":
        return sh[1], sh[2]
    _, x0, x1, r = sh
    return np.array([x0, -r, -r]), np.array([x1, r, r])


def _overlap(a0, a1, b0, b1, eps=1e-7) -> bool:
    return a0 < b1 - eps and b0 < a1 - eps


def _cyl_height_over(yc, r, y0, y1):
    """How far above its axis a cylinder (axis along x, at y = yc, radius r) reaches down to the top face of a box
    spanning y0..y1 (None: they do not meet)."""
    if y0 <= yc <= y1:
        return r
    d = min(abs(yc - y0), abs(yc - y1))
    return math.sqrt(r * r - d * d) if d < r else None


def _need(sh, o, lo2, hi2):
    """The lowest origin z for which the shape ``sh`` (part frame, origin ``o`` in xy) sits on top of the world box
    (lo2, hi2), or None if they do not meet seen from above."""
    lo, hi = _bounds(sh)
    if not (_overlap(o[0] + lo[0], o[0] + hi[0], lo2[0], hi2[0])
            and _overlap(o[1] + lo[1], o[1] + hi[1], lo2[1], hi2[1])):
        return None
    if sh[0] == "box":
        return float(hi2[2] - lo[2])
    h = _cyl_height_over(o[1], sh[3], lo2[1], hi2[1])
    return None if h is None else float(hi2[2] + h)


def _need_over_cyl(sh, o, cyl, oc):
    """The lowest origin z for which the box ``sh`` (part frame, origin ``o``) rests on another part's cylinder
    (``cyl`` in that part's frame, its origin ``oc``), or None."""
    if sh[0] != "box":
        return None  # cylinder on cylinder never happens in this scene
    lo, hi = sh[1], sh[2]
    _, x0, x1, r = cyl
    if not _overlap(o[0] + lo[0], o[0] + hi[0], oc[0] + x0, oc[0] + x1):
        return None
    h = _cyl_height_over(oc[1], r, o[1] + lo[1], o[1] + hi[1])
    return None if h is None else float(oc[2] + h - lo[2])


class OfflineParts:
    def __init__(self, arm_error=None) -> None:
        self.arm_error = arm_error  # (side, part, tips) -> (dx, dy) m, or None: no arm error
        self.reset()

    def reset(self) -> None:
        self.o = {"motor": np.array(L.UNIT_START, dtype=float), "switch": np.array(L.SWITCH_START, dtype=float),
                  "cover": np.array(L.COVER_START, dtype=float)}
        self.holders: dict[str, dict[str, np.ndarray]] = {n: {} for n in L.LOOSE}  # part -> {arm: offset}
        self.widths: dict[str, float] = {}  # arm -> width of the part it holds
        self.prev_gap: dict[str, float] = {}
        self.events: list[tuple] = []

    # -- geometry ---------------------------------------------------------------------------------------------------
    def width_at(self, name: str, tips) -> float | None:
        """The part's width along y between the pads at the pad-tip point (None: nothing of it is between them)."""
        o = self.o[name]
        spans = []
        for sh in SHAPES[name]:
            lo, hi = _bounds(sh)
            x0, x1 = o[0] + lo[0], o[0] + hi[0]
            z0, z1 = o[2] + lo[2], o[2] + hi[2]
            if not (x0 <= tips[0] <= x1 and _overlap(z0, z1, tips[2], tips[2] + PAD_SPAN_UP)):
                continue
            if sh[0] == "box":
                spans.append((o[1] + lo[1], o[1] + hi[1]))
            else:
                r = sh[3]
                dz = max(0.0, tips[2] - o[2]) if o[2] < tips[2] else max(0.0, o[2] - (tips[2] + PAD_SPAN_UP))
                if dz < r:
                    c = math.sqrt(r * r - dz * dz)
                    spans.append((o[1] - c, o[1] + c))
        if not spans:
            return None
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            if a - 0.010 <= tips[1] <= b + 0.010:
                return float(b - a)
        return None

    def rest_z(self, name: str) -> float:
        """Where the part comes to rest moving straight down from where it is: the highest support top below each
        of its shapes (states are kept clear of every support, so a support counts when its top is no higher than
        the shape's underside)."""
        o = self.o[name]
        best = L.TOP - min(_bounds(sh)[0][2] for sh in SHAPES[name])  # the table
        for sh in SHAPES[name]:
            for lo2, hi2 in FIXED:
                z = _need(sh, o, lo2, hi2)
                if z is not None and z <= o[2] + 1e-4:
                    best = max(best, z)
            for other in L.LOOSE:
                if other == name or self.holders[other]:
                    continue
                oc = self.o[other]
                for osh in SHAPES[other]:
                    if osh[0] == "box":
                        z = _need(sh, o, oc + osh[1], oc + osh[2])
                    else:
                        z = _need_over_cyl(sh, o, osh, oc)
                    if z is not None and z <= o[2] + 1e-4:
                        best = max(best, z)
        return float(best)

    # -- one control step -------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        if arms is None:
            return
        for side in ("left", "right"):
            a = arms[side]
            tips, gap = np.asarray(a.tips, dtype=float), float(a.opening) * L.FULL_GAP
            prev = self.prev_gap.get(side, L.FULL_GAP)
            held = next((n for n in L.LOOSE if side in self.holders[n]), None)
            if held is not None:
                if gap > self.widths[side] + 0.001:
                    del self.holders[held][side]
                    del self.widths[side]
                    self.events.append(("release", side, held))
                    if not self.holders[held]:
                        self.o[held][2] = self.rest_z(held)
                        self.events.append(("drop", held, [round(float(v), 4) for v in self.o[held]]))
            elif gap < prev:
                for n in L.LOOSE:
                    w = self.width_at(n, tips)
                    if w is not None and gap < w <= prev + 1e-9:
                        self.holders[n][side] = self.o[n] - tips
                        self.widths[side] = w
                        self.events.append(("grasp", side, n, round(w * 1000, 1)))
                        break
            self.prev_gap[side] = gap
        for n in L.LOOSE:
            h = self.holders[n]
            if h:  # carried; it cannot be pushed down through a support (it slips up in the grip instead)
                target = np.mean([np.asarray(arms[s].tips, dtype=float) + off + self._err(s, n, arms[s].tips)
                                  for s, off in h.items()], axis=0)
                self.o[n] = np.array([target[0], target[1], self.o[n][2]])
                self.o[n][2] = max(float(target[2]), self.rest_z(n))

    def _err(self, side: str, part: str, tips) -> np.ndarray:
        if self.arm_error is None:
            return np.zeros(3)
        dx, dy = self.arm_error(side, part, tips)
        return np.array([dx, dy, 0.0])

    def states(self, objects: dict) -> dict:
        """``objects`` with the three loose parts' states replaced by this model's (square, at rest when let go)."""
        import mechkit as mk

        out = dict(objects)
        specs = {spec.name: spec for spec in _SPECS}
        for n in L.LOOSE:
            out[n] = mk.state_from_poses(specs[n], (*self.o[n], 0.0, 0.0, 0.0, 1.0), np.zeros(6), source="offline")
        return out


_SPECS: list = []  # task.py registers its three assemblies here


def register(*specs) -> None:
    _SPECS[:] = list(specs)
