"""Every dimension of the task, in one place (metres, kilograms, degrees where named *_DEG, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A geological field map (1:10,000) lies on the table in front of the arms, north at the top: north is +x, away from
the arms, and a north arrow on the map points that way. Three outcrop stations are circled on it and labelled A, B
and C. Their field readings, strike/dip in degrees by the right-hand rule (the dip direction is 90 degrees clockwise
from the strike), are A 030/40, B 250/25 and C 135/60. Three T-shaped tokens, strike-and-dip map symbols (a long
dark red bar along the strike with a short yellow dip tick at its middle, pointing down-dip), lie on the map below
the stations. Plotting the readings means one symbol centred on each station, its bar on that station's strike and
its tick toward that station's dip direction.

Bearings: north is +x and, seen from above, a bearing turns clockwise, toward -y (east is -y, the arms' right). A
bearing of b degrees is therefore the world yaw -b (yaw counted counter-clockwise from +x).
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the readings and the right-hand rule ----------------------------------------------------------------------------
READINGS = {"A": (30.0, 40.0), "B": (250.0, 25.0), "C": (135.0, 60.0)}  # station: (strike bearing, dip angle), deg
RHR_TURN_DEG = 90.0  # right-hand rule: the dip direction is the strike bearing plus 90 degrees (clockwise)


def bearing_to_yaw(bearing_deg: float) -> float:
    """World yaw (degrees, -180..180, counter-clockwise from +x) of a compass bearing on this map (north +x)."""
    return math.degrees(math.remainder(math.radians(-bearing_deg), 2 * math.pi))


def dip_direction(station: str) -> float:
    """The dip direction (bearing, 0..360) of a station's reading under the right-hand rule."""
    return (READINGS[station][0] + RHR_TURN_DEG) % 360.0


STRIKE_YAW_DEG = {s: bearing_to_yaw(b) for s, (b, _) in READINGS.items()}  # A -30, B 110, C -135 (mod 180: 45)
DIP_YAW_DEG = {s: bearing_to_yaw(dip_direction(s)) for s in READINGS}  # A -120, B +20, C +135: where the tick points

# -- the map (fixed): a printed sheet on card, its marks printed in black ---------------------------------------------
MAP_X0, MAP_X1 = 0.200, 0.500  # 300 mm along x (south .. north)
MAP_Y0, MAP_Y1 = -0.400, 0.400  # 800 mm along y (east .. west)
MAP_T = 0.001  # 1 mm card
MAP_TOP_W = TOP + MAP_T  # world z of the map's surface
# Printed marks are modelled as black plastic film 0.5 mm thick set 0.3 mm into the card (standing 0.2 mm proud), so
# they render; a token on a station's ring sits level 0.2 mm up.
INK_T = 0.0005
INK_PROUD = 0.0002
INK_Z_W = MAP_TOP_W + INK_PROUD - INK_T / 2  # every mark's centre (world z)

STATIONS = {"A": (0.330, 0.220), "B": (0.330, -0.220), "C": (0.300, 0.000)}  # station centres (x, y)
RING_R = 0.009  # the ring's centre line: 20 mm across outside, 16 mm inside
RING_W = 0.002
RING_SEGMENTS = 8  # drawn as eight straight strokes
RING_SEG_LEN = 2 * RING_R * math.sin(math.pi / RING_SEGMENTS) - 0.0008  # 6.09 mm: inner corners overlap < 0.3 mm
LABEL_DX = 0.050  # each station's letter is printed due north of it, its centre 50 mm from the station's centre

# Letters, a north arrow and "N": axis-aligned strokes (u, v in the glyph, u to the reader's right = -y, v up = +x),
# each stroke a rectangle (u0, u1, v0, v1) in metres; strokes butt against each other and never overlap.
GLYPH_H = 0.016  # 16 mm tall letters
GLYPH_W = 0.010
STROKE = 0.0025
_H, _W, _S = GLYPH_H, GLYPH_W, STROKE
_INSET = 0.002  # the B's upper bowl is 2 mm narrower than its lower one
GLYPHS = {
    "A": ((0.0, _S, 0.0, _H), (_W - _S, _W, 0.0, _H), (_S, _W - _S, _H - _S, _H),
          (_S, _W - _S, _H / 2 - _S / 2, _H / 2 + _S / 2)),
    "B": ((0.0, _S, 0.0, _H), (_S, _W - _INSET, _H - _S, _H),
          (_W - _INSET - _S, _W - _INSET, _H / 2 + _S / 2, _H - _S),
          (_S, _W - _S, _H / 2 - _S / 2, _H / 2 + _S / 2), (_W - _S, _W, _S, _H / 2 + _S / 2), (_S, _W, 0.0, _S)),
    "C": ((0.0, _S, 0.0, _H), (_S, _W, _H - _S, _H), (_S, _W, 0.0, _S)),
    "N": ((0.0, _S, 0.0, _H), (_W - _S, _W, 0.0, _H)),  # plus a diagonal, NORTH_DIAGONAL
}
NORTH_AT = (0.478, 0.100)  # the "N" glyph's centre (x 0.470 .. 0.486); the arrow runs below it
ARROW_X0, ARROW_X1 = 0.415, 0.455  # the north arrow's shaft: 40 mm along +x, pointing north
ARROW_Y = NORTH_AT[1]
ARROW_HEAD = 0.009  # the arrowhead: a square 9 mm across its diagonals, turned 45 degrees (a diamond)


def _glyph_boxes(ch: str, centre) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """A letter's axis-aligned strokes as world boxes: ((x, y) centre, (size x, size y), yaw 0)."""
    cx, cy = centre
    x0, y0 = cx - GLYPH_H / 2, cy + GLYPH_W / 2  # the glyph's lower-left corner as the reader sees it
    out = []
    for u0, u1, v0, v1 in GLYPHS[ch]:
        out.append(((x0 + (v0 + v1) / 2, y0 - (u0 + u1) / 2), (v1 - v0, u1 - u0), 0.0))
    return out


def _north_diagonal(centre):
    """The N's diagonal from its top left to its bottom right, between (not into) its two uprights."""
    cx, cy = centre
    du, dv = GLYPH_W - 2 * STROKE, -GLYPH_H  # from (S, H) to (W - S, 0) in the glyph
    length = math.hypot(du, dv) - 2 * STROKE  # shortened so its square ends stay clear of the uprights
    yaw = math.atan2(-du, dv)  # glyph (u, v) -> world (-u, v) as (dy, dx)
    return ((cx, cy), (length, STROKE), yaw)


def _arrow():
    """The north arrow: a shaft along +x ending at the back corner of a diamond head (touching, not overlapping)."""
    head_x = ARROW_X1 + ARROW_HEAD / 2  # the diamond's centre; its back corner is the shaft's end, its point north
    side = ARROW_HEAD / math.sqrt(2)
    shaft = (((ARROW_X0 + ARROW_X1) / 2, ARROW_Y), (ARROW_X1 - ARROW_X0, STROKE), 0.0)
    head = ((head_x, ARROW_Y), (side, side), math.pi / 4)
    return [shaft, head]


def ink_marks() -> dict[str, tuple[tuple[float, float], tuple[float, float], float]]:
    """Every printed mark on the map: name -> ((x, y) centre, (length, width) before yaw, yaw in rad)."""
    marks = {}
    for s, (x, y) in STATIONS.items():
        for k in range(RING_SEGMENTS):
            phi = (k + 0.5) * 2 * math.pi / RING_SEGMENTS
            marks[f"ring_{s}_{k}"] = ((x + RING_R * math.cos(phi), y + RING_R * math.sin(phi)), (RING_SEG_LEN, RING_W),
                                      phi + math.pi / 2)
        for i, b in enumerate(_glyph_boxes(s, (x + LABEL_DX, y))):
            marks[f"label_{s}_{i}"] = b
    for i, b in enumerate(_glyph_boxes("N", NORTH_AT)):
        marks[f"north_n_{i}"] = b
    marks["north_n_diagonal"] = _north_diagonal(NORTH_AT)
    for i, b in enumerate(_arrow()):
        marks[f"north_arrow_{i}"] = b
    return marks


# -- the three tokens: each one loose rigid body (mechkit compound) of two boxes ---------------------------------------
# Base frame: origin at the middle of the bar's footprint on its base plane, local +x along the tick (the way it
# points, down-dip), the bar along local y, local +z up. A token's yaw is the world yaw of its local +x: the
# direction its tick points. Its bar's axis is that yaw + 90 degrees.
BAR = (0.020, 0.060, 0.012)  # 20 (across, along the tick) x 60 (long, the strike line) x 12 mm tall
TICK = (0.025, 0.010, 0.012)  # 25 long, 10 wide, 12 tall, from the bar's edge (local x +10 .. +35 mm)
BAR_CENTER = (0.0, 0.0, BAR[2] / 2)
TICK_CENTER = (BAR[0] / 2 + TICK[0] / 2, 0.0, TICK[2] / 2)  # (0.0225, 0, 0.006)
TOKEN_H = BAR[2]
TICK_TIP_LOCAL = (BAR[0] / 2 + TICK[0], 0.0, TOKEN_H / 2)  # the middle of the tick's end face
TOKEN_MASS = 0.025  # acetal (POM, 1.41 g/cm3): 17.4 cm3 is 24.5 g
TOKENS = {  # name: (origin x, y at the start, tick yaw at the start in degrees)
    "token_1": (0.250, 0.150, 150.0),
    "token_2": (0.250, -0.150, 110.0),
    "token_3": (0.250, 0.000, 180.0),
}

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
CENTRE_TOL = 0.008  # the bar's centre within 8 mm of the station's centre
STRIKE_TOL_DEG = 6.0  # the bar within 6 degrees of the strike (mod 180)
TICK_TOL_DEG = 6.0  # the tick within 6 degrees of the right-hand-rule dip direction
FLAT_DEG = 5.0  # lying flat: tilt under 5 degrees
ON_MAP_TOL = 0.003  # its lowest point within 3 mm of the map's surface (not on another token, not held up)
CREDIT_SIXTHS = {"symbol": 2, "strike_only": 1}  # per station: 1/3 and 1/6 of the task score
STILL_WINDOW_S = 0.5  # still: in each of the last two 0.5 s windows (1 s in all) ...
STILL_WINDOWS = 2
STILL_PATH = 0.0015  # ... the bar's centre and the tick's tip each travelled under 1.5 mm (a path, not a velocity)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a token moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# 'Let go' (rigstate): each gripper's pad-tip point (midway between its finger tips) and its opening are known, its
# yaw is not, so a finger may be on any side of that point. With the pads FULL_GAP x opening apart, a finger
# (FINGER_THICK thick outside the gap and 2 x FINGER_HALF_WIDTH wide across it) can touch a token only if the token's
# outline, seen from above, comes within release_reach(opening) of that point. A gripper whose pad-tip point is also
# lower than RELEASE_ABOVE over the token's top may be holding, pressing or propping it: the token does not count.
FULL_GAP = 0.0939
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010


def release_reach(opening: float) -> float:
    """How far (m) from a gripper's pad-tip point, horizontally, a token's outline may be while a finger could touch
    it, over every yaw, plus the margin: 65 mm fully open, 50 mm closed on a token's 60 mm length."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + RELEASE_MARGIN


# -- colours (sRGB) --------------------------------------------------------------------------------------------------
MAP_BUFF = (0.93, 0.89, 0.76)
INK_BLACK = (0.03, 0.03, 0.03)
BAR_RED = (0.62, 0.06, 0.05)
TICK_YELLOW = (1.00, 0.86, 0.05)

assert {s: round(v, 6) for s, v in STRIKE_YAW_DEG.items()} == {"A": -30.0, "B": 110.0, "C": -135.0}
assert {s: round(v, 6) for s, v in DIP_YAW_DEG.items()} == {"A": -120.0, "B": 20.0, "C": 135.0}
