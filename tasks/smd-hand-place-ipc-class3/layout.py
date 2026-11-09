"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

Electronics rework at 8:1 scale: a green circuit board on the midline carries three surface-mount footprints (pairs
of tinned pads), and three loose parts lie on two antistatic mats beside it: a 2512 chip resistor and a 1210 ceramic
chip capacitor on the left mat, a D-case (7343) moulded tantalum capacitor on the right mat. Every footprint's axis
is x (its two pads at x -+ from its centre). Each footprint's pads are 2 mm per side wider than its part and reach
6 mm past each of the part's ends, so a centred part has 6 mm of play along its axis before an end overhangs.

Each part is a loose mechkit compound (one rigid body of several boxes): its base frame's origin is the middle of its
bottom face, local +x along its length (for the tantalum: toward its banded + end), local +z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TOP = 0.7475  # station table top (world z)
SCALE = 8  # parts and pads are 8 times real size

# -- the circuit board: a thin green slab fixed to the table, long side along y ---------------------------------------
BOARD_X, BOARD_Y = 0.300, 0.0
BOARD_SIZE = (0.150, 0.220, 0.004)  # x 0.225..0.375, y -0.110..0.110
BOARD_TOP = BOARD_SIZE[2]  # 4 mm above the table
PAD_T = 0.001  # pads are 1 mm proud of the board
PAD_TOP = BOARD_TOP + PAD_T  # 5 mm above the table: where a placed part's underside rests
PAD_TOP_W = TOP + PAD_TOP


@dataclass(frozen=True)
class Footprint:
    """Two equal pads on the board, centred at (x +- pad_offset, y), each pad_len along x and pad_width along y."""

    key: str
    x: float
    y: float
    pad_offset: float  # each pad's centre from the footprint's centre, along x
    pad_len: float  # along x
    pad_width: float  # along y

    @property
    def outer(self) -> float:
        """Distance from the footprint's centre to each pad's outer end, along x."""
        return self.pad_offset + self.pad_len / 2

    @property
    def inner(self) -> float:
        """Distance from the footprint's centre to each pad's inner end, along x."""
        return self.pad_offset - self.pad_len / 2

    def pads(self) -> tuple[tuple[str, float, float], tuple[str, float, float]]:
        """((name, x, y) of the pad nearer the arms (-x), (name, x, y) of the farther pad (+x))."""
        return ((f"{self.key}_pad_near", self.x - self.pad_offset, self.y),
                (f"{self.key}_pad_far", self.x + self.pad_offset, self.y))


# The three footprints, 65 mm apart along y on the board's centre line (x 0.300).
R_FP = Footprint("r", 0.300, +0.065, 0.022, 0.018, 0.029)  # 2512: pads 18 x 29, span 13..31 mm each side
C_FP = Footprint("c", 0.300, 0.000, 0.011, 0.016, 0.024)  # 1210: pads 16 x 24, span 3..19 mm each side
T_FP = Footprint("t", 0.300, -0.065, 0.026, 0.018, 0.038)  # 7343: pads 18 x 38, span 17..35 mm each side
# The tantalum footprint's far (+x) pad is its + pad (anode). Pad names for it are t_pad_minus / t_pad_plus.
PLUS_PAD_SIGN = +1.0

# The + mark: a white square printed on the board beside the + pad (on its -y side, 7 mm from the pad's edge).
PLUS_MARK_SIZE = (0.008, 0.008, 0.001)
PLUS_MARK_X = T_FP.x + T_FP.pad_offset  # 0.326: level with the + pad's centre
PLUS_MARK_Y = T_FP.y - T_FP.pad_width / 2 - 0.007 - PLUS_MARK_SIZE[1] / 2  # -0.095


@dataclass(frozen=True)
class Part:
    """A loose chip part: a box of length (local x) x width (local y) x height, with a termination of length
    ``term`` at each end (the end ``term`` of its length, over its full width)."""

    name: str  # the task-file object's name
    label: str
    length: float
    width: float
    height: float
    term: float
    mass: float
    start: tuple[float, float]  # base-frame origin (x, y) on its mat
    start_yaw: float
    footprint: Footprint
    polarised: bool  # True: local +x is the + (banded) end and must sit on the + pad; local +z must face up

    @property
    def square_section(self) -> bool:
        """Square in section (width = height) with its terminations round all four long faces and nothing marking
        a top: the part is the same, seen from above and on its pads, whichever long face is down (the ceramic
        capacitor)."""
        return not self.polarised and abs(self.width - self.height) < 1e-9


# Parts lie on their mats with their long side along y (yaw 90 degrees); the tantalum's band toward +y. Masses are
# those of solid plastic blocks of these sizes (1.1-1.45 g/cm3): 30, 15 and 50 g.
MAT_SIZE = (0.090, 0.130, 0.003)
MAT_TOP = MAT_SIZE[2]
MAT_L = (0.200, +0.240)  # left mat: x 0.155..0.245, y 0.175..0.305
MAT_R = (0.200, -0.240)  # right mat: y -0.305..-0.175

RESISTOR = Part("resistor", "R1, 2512 chip resistor", 0.050, 0.025, 0.020, 0.008, 0.030,
                (0.200, +0.215), math.pi / 2, R_FP, False)
CAPACITOR = Part("capacitor", "C1, 1210 ceramic chip capacitor", 0.026, 0.020, 0.020, 0.006, 0.015,
                 (0.200, +0.265), math.pi / 2, C_FP, False)
TANTALUM = Part("tantalum", "T1, D-case (7343) moulded tantalum capacitor", 0.058, 0.034, 0.024, 0.008, 0.050,
                (0.200, -0.240), math.pi / 2, T_FP, True)
PARTS = (RESISTOR, CAPACITOR, TANTALUM)
BY_NAME = {p.name: p for p in PARTS}

# The tantalum's build: a moulded body 22 mm tall on two 1 mm silver termination plates under its ends (bottom-only
# terminations, each as long as `term`), and a dark band 1 mm thick and 10 mm wide across its top at the + end.
TA_PLATE_T = 0.001
TA_BODY_H = 0.022
TA_BAND_W = 0.010
TA_BAND_T = 0.001
assert abs(TA_PLATE_T + TA_BODY_H + TA_BAND_T - TANTALUM.height) < 1e-12

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------------
TILT_MAX_DEG = 5.0  # 'flat': lying on a face within this (resistor: either broad face; ceramic: any long face;
#                     tantalum: band face up)
FLAT_TOL = 0.002  # 'flat on the pads': its lowest point within 2 mm of the pads' top (above or below)
AXIS_MAX_DEG = 10.0  # its long axis within this of the footprint's axis, seen from above
END_TOL = 0.0  # end overhang: none permitted (no corner past a pad's outer end)
CLASS3_FRAC = 0.25  # side overhang <= 25 % of the part's width: Class 3
CLASS2_FRAC = 0.50  # side overhang <= 50 % of the part's width: Classes 1 and 2
CLASS3_SIXTHS = 2  # credit in sixths of the full score: 1/3 a part for Class 3
CLASS2_SIXTHS = 1  # 1/6 a part for Classes 1-2 only
OFF_TABLE_DROP = 0.05  # a part whose underside is this far below the table top has fallen off it
STILL_WINDOW_S = 0.5  # 'at rest': neither end of the part moved more than STILL_TOL in the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a part moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def lateral_limit(part: Part, frac: float) -> float:
    """The largest sideways offset of a centred, square part whose side overhang stays within frac x its width."""
    return frac * part.width + part.footprint.pad_width / 2 - part.width / 2


def axial_play(part: Part) -> float:
    """How far a square part can move along its footprint's axis before an end overhangs a pad's outer end."""
    return part.footprint.outer - part.length / 2


# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
PCB_GREEN = (0.10, 0.40, 0.20)
TIN = (0.80, 0.80, 0.78)  # tinned copper pads and the parts' terminations
MARK_WHITE = (0.95, 0.95, 0.95)
MAT_GREY = (0.24, 0.25, 0.27)
R_BLACK = (0.07, 0.07, 0.08)
C_TAN = (0.70, 0.55, 0.36)
TA_AMBER = (0.90, 0.66, 0.16)
TA_BAND = (0.28, 0.15, 0.05)
