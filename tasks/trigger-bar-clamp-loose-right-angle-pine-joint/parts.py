"""The shapes: two pine boards and the clamp (a fixed jaw and bar, a moving jaw on a slider with the pistol grip, a trigger
on a hinge, a release tab on a hinge). mechkit shapes in each part's own frame, joints at 0."""

from __future__ import annotations

import layout as L
import mechkit as mk


def _box(lo, hi, color):
    c = tuple((a + b) / 2 for a, b in zip(lo, hi))
    s = tuple(b - a for a, b in zip(lo, hi))
    return mk.box(c, s, color)


def board_shapes(length_x: float, length_y: float):
    """A board lying flat, in its frame: origin on the bench under its centre, size (length_x, length_y)."""
    return [mk.box((0.0, 0.0, L.BOARD_T / 2), (length_x, length_y, L.BOARD_T), L.PINE)]


def stop_shapes():
    """The bench stop: a fixed pine strip along x, in its frame (origin on the bench under its centre)."""
    return [mk.box((0.0, 0.0, L.BOARD_T / 2), (L.STOP_X1 - L.STOP_X0, L.STOP_Y1 - L.STOP_Y0, L.BOARD_T), L.PINE)]


def b_shapes():
    return board_shapes(L.BOARD_W, L.B_LEN)


def a_shapes():
    return board_shapes(L.A_LEN, L.BOARD_W)


def base_shapes():
    """The fixed jaw (yellow body, black pad on its face at x 0), the steel bar and the bar's end stop."""
    h, c = L.JAW_H, L.CLAMP_YELLOW
    return [
        _box((-L.JAW_BODY - L.PAD_T, L.PAD_Y0, 0.0), (-L.PAD_T, L.PAD_Y1, h), c),
        _box((-L.PAD_T, L.PAD_Y0, 0.0), (0.0, L.PAD_Y1, h), L.CLAMP_BLACK),
        _box((L.BAR_X0, -L.BAR_Y, L.BAR_Z0), (L.BAR_X1, L.BAR_Y, L.BAR_Z1), L.STEEL),
        _box((L.BAR_X1 - 0.010, -L.BAR_Y - 0.002, 0.004), (L.BAR_X1, L.BAR_Y + 0.002, 0.022), L.CLAMP_BLACK),
    ]


def jaw_shapes():
    """The moving jaw: black pad (face at x = G0), yellow body, the head sliding on the bar and the pistol grip's handle
    (a rubber-black sleeve over its middle)."""
    h, c = L.JAW_H, L.CLAMP_YELLOW
    g = L.G0
    return [
        _box((g, L.PAD_Y0, 0.0), (g + L.PAD_T, L.PAD_Y1, h), L.CLAMP_BLACK),
        _box((g + L.PAD_T, L.PAD_Y0, 0.0), (g + L.PAD_T + L.JAW_BODY, L.PAD_Y1, h), c),
        _box((L.HEAD_X0, -L.HEAD_Y, 0.0), (L.HEAD_X1, L.HEAD_Y, h), c),
        _box((L.HANDLE_X0, L.HANDLE_Y0, 0.0), (L.HANDLE_X1, L.HANDLE_Y1, h), c),
        _box((L.HANDLE_X0 - 0.002, L.HANDLE_Y0 + 0.010, 0.002), (L.HANDLE_X1 + 0.002, L.HANDLE_Y1 - 0.030, h - 0.002),
             L.CLAMP_BLACK),
    ]


def trigger_shapes():
    """The trigger lever at rest: a black bar 14 mm thick, 110 mm long, hinged at its head end."""
    return [_box((L.TRIG_X0, L.TRIG_Y0, 0.003), (L.TRIG_X1, L.TRIG_Y1, 0.023), L.CLAMP_BLACK)]


def release_shapes():
    return [_box((L.REL_X0, L.REL_Y0, 0.006), (L.REL_X1, L.REL_Y1, 0.020), L.STEEL)]


def clamp_joints():
    jaw = mk.slider("jaw", "-X", jaw_shapes(), upper=L.JAW_UPPER, lower=0.0, anchor=(L.G0, 0.0, 0.0), mass=0.10,
                    spring=L.K_JAW, damping=60.0, friction=0.0)
    trig = mk.hinge("trigger", "-Z", (L.TRIG_PIVOT[0], L.TRIG_PIVOT[1], 0.0), trigger_shapes(), upper=L.TRIG_UPPER,
                    lower=0.0, mass=0.02, spring=L.TRIG_SPRING, damping=0.02, parent="jaw")
    rel = mk.hinge("release", "Z", (L.REL_ANCHOR[0], L.REL_ANCHOR[1], 0.0), release_shapes(), upper=L.REL_UPPER,
                   lower=0.0, mass=0.01, spring=L.REL_SPRING, damping=L.REL_DAMPING, parent="jaw")
    return [jaw, trig, rel]
