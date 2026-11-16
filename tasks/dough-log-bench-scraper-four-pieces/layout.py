"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A flat block of bread dough lies on a bench top 40 mm over the table, along y, 256 x 80 x 40 mm, 0.90 kg: four
portions of 64 mm marked on the bench's edge. At each of the three marks (y -64, 0, +64 mm) the log has a 10 mm cut
strip of dough over a slot in the bench top: the strip is a link that slides down the slot as a bench scraper's edge
is pressed through it (the dough squeezed down out of the cut). A fixed bond holds each piece to the next until its
strip is through. Two bench scrapers (6 x 3 in blades, bar handles) stand in two fixed rests, one in front of each
arm; each arm chops with its own.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
IN = 0.0254
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the bench top (fixed) -----------------------------------------------------------------------------------------
BENCH_X = 0.22  # the bench's and the log's centre line
BENCH_LX, BENCH_LY = 0.240, 0.400  # 240 mm front to back, 400 mm along y
BENCH_T = 0.012  # top plate thickness (est)
BENCH_TOP = 0.040  # the bench top's top face over the table (the log's underside)
RAIL_W, RAIL_X = 0.020, 0.110  # two rails under the plate's front and back edges, 28 mm tall
SLOT_LX, SLOT_LY = 0.100, 0.012  # a slot per mark through the top plate, over the void under it
BENCH_COLOR = (0.80, 0.62, 0.40)
RAIL_COLOR = (0.55, 0.42, 0.28)
MARK_COLOR = (0.10, 0.10, 0.10)
MARK_X0, MARK_X1 = 0.312, 0.332  # the printed cut marks on the bench top's back edge
MARK_LEN_Y, MARK_H = 0.0015, 0.0006
BENCH_FRICTION = (0.35, 0.30)  # floured dough on a floured wood top (est)

# -- the dough log (loose) ------------------------------------------------------------------------------------------
LOG_X, LOG_Y = BENCH_X, 0.0
LOG_W = 0.080  # across (x)
LOG_H = 0.040  # tall (z)
LOG_L = 0.256  # along y, four 64 mm portions
PORTION = LOG_L / 4  # 64 mm
MARKS = (-0.064, 0.0, 0.064)  # y of the three cut marks (portion boundaries)
KERF = 0.010  # the cut strip's width between two pieces (the dough the scraper squeezes down)
WEB_CLEAR = 0.0003  # a strip is this much narrower than its gap on each side
WEB_W = KERF - 2 * WEB_CLEAR  # 9.4 mm
DOUGH_DENSITY = 1098.6  # kg/m3: 0.90 kg in 256 x 80 x 40 mm (bread dough, about 1.1 g/mL, est)
LOG_MASS = round(DOUGH_DENSITY * LOG_W * LOG_H * LOG_L, 3)  # 0.900 kg
# piece bodies along y: the end pieces 59 mm, the middle ones 54 mm (each loses half a strip per cut it borders)
Y_END0 = LOG_Y - LOG_L / 2  # -0.128
BODY_Y = []  # (y0, y1) per piece
_y = Y_END0
for _k, _len in enumerate((PORTION - KERF / 2, PORTION - KERF, PORTION - KERF, PORTION - KERF / 2)):
    BODY_Y.append((_y, _y + _len))
    _y += _len + KERF
BODY_LEN = [y1 - y0 for y0, y1 in BODY_Y]
BODY_YC = [(y0 + y1) / 2 for y0, y1 in BODY_Y]
BODY_MASS = [round(DOUGH_DENSITY * LOG_W * LOG_H * n, 4) for n in BODY_LEN]
WEB_MASS = round(DOUGH_DENSITY * LOG_W * LOG_H * WEB_W, 4)
PIECE_Z = TOP + BENCH_TOP + LOG_H / 2  # a piece's base frame sits at its centre
DOUGH_COLOR = (0.93, 0.85, 0.66)
DOUGH_FRICTION = (0.35, 0.30)

# the cut strip's dough: a slider link on the piece before the mark, running down its slot
WEB_TRAVEL = LOG_H  # the strip's top goes from the log's top down to the bench top: 40 mm
WEB_THROUGH = 0.037  # a strip pressed down this far (3 mm of its 40 mm left standing) is through: the edge is at the bench top
WEB_CREASE = 0.004  # a strip pressed in more than this and not through is creased (squashed, not cut)
WEB_SPRING = 2000.0  # N/m: the drive stiffness that makes the strip resist a push in proportion to its speed
WEB_DAMP = 15.0  # N s/m
YIELD_F = 1.5  # N: the dough does not give until the edge presses with more than this (design figure)
YIELD_Q = 0.001  # below 1 mm the strip springs back to flush when the edge leaves; past it, it stays where it is left

# -- the bench scrapers (loose) -------------------------------------------------------------------------------------
# 6 in x 3 in stainless blade, bar handle along the back. Base frame at the centre of the blade's edge (on the table
# in the rest), blade in the x-z plane.
SC_W = round(6 * IN, 4)  # 152.4 mm along x
SC_H = round(3 * IN, 4)  # 76.2 mm tall
SC_T = 0.0025  # modelled blade thickness (real 0.9 mm: est, 2.5 mm so it is a stable collider)
BAR_LX, BAR_LY, BAR_LZ = 0.120, 0.022, 0.016  # the handle bar (est)
BAR_Z = SC_H + BAR_LZ / 2  # grip point (TCP) height over the edge: 84.1 mm
SC_MASS = 0.150  # est
SC_COLOR = (0.76, 0.78, 0.82)
BAR_COLOR = (0.14, 0.14, 0.16)
SC_FRICTION = (0.30, 0.25)  # steel on floured wood and on steel (est)
SC_X = LOG_X
SC_Y = {"right": -0.30, "left": 0.30}  # where each scraper stands in its rest
# fixed rests: two rails 28 mm tall with a 4 mm slot between them, on the table
REST_RAIL_Y, REST_GAP, REST_H, REST_LX = 0.012, 0.004, 0.030, 0.150
REST_COLOR = (0.40, 0.40, 0.44)

# -- rule thresholds (scoring.py) -----------------------------------------------------------------------------------
CUT_OK_MARGIN = 0.001  # the blade's centre within the strip's half width plus this
CUT_X = 0.036  # the blade's middle within this of the log's centre line (it spans the log's 80 mm)
CUT_EDGE_Z = BENCH_TOP + LOG_H / 2  # the edge below the log's mid height while in the strip
BLADE_UPRIGHT_DEG = 15.0  # the blade's plane within this of vertical, its length within this of along x
DRAG_MOVE = 0.012  # a piece moved this far from where it began (smeared along the bench), latched
DRAG_TURN_DEG = 8.0
LINE_X = 0.030  # at the end every piece's centre within this of the log's centre line
PIECE_GAP_MIN = 0.004  # at the end every two neighbouring pieces at least this far apart (cut through)
TOUCH = 0.002  # a scraper within this of a piece is touching the dough
FALL_Z = BENCH_TOP - 0.010  # a piece lower than this (off the bench) is binned
TOOL_FALL_Z = -0.010  # a scraper's lowest point lower than this (off the table) is not set down
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005  # a person sees a 5 mm shift on video (was 1 mm)
STILL_TURN_DEG = 5.0  # and a 5 degree turn (was 0.3)
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9

PIECES = ("p0", "p1", "p2", "p3")
SCRAPERS = ("scraper_r", "scraper_l")


def web_y(k: int) -> float:
    """Centre y of the strip at mark k (0..2)."""
    return MARKS[k]


def portion_masses():
    """Mass of each piece once its strips are down (kg): the bodies only."""
    return list(BODY_MASS)
