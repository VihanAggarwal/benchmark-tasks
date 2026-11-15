"""Every dimension of the task, in one place, real size (metres, kilograms, the station's world frame), and the
kinematics of the window: sash hinge, stay swivel and stay lift.

World frame: x forward from the line through the two arm bases, y to the left, z up; the station's table top is at
z 0.7475. Heights named *_H are above the window board's top (the sill), 40 mm over the table.

SOURCES (see README.md):
- The stay and its pins are Samuel Heath P4512-B "Casement stay, 254mm", read from the maker's CAD model
  (p4512-b.igs, Autodesk Inventor, mm) and drawing (p4512-b.pdf): 278.64 mm overall, four holes 7.2 mm across at a
  40.5 mm pitch, the first 64.52 mm from the swivel axis; a bar 13.6 mm wide and 8 mm deep with an 8.1 mm groove
  3.75 mm deep along its underside (the holes go through the 4.25 mm web above it); a swivel 11.11 mm in front of
  the plate's back and a lift pin 20.02 mm along the bar from it; a curled tail ending in a 14 mm ball. Each pin: a
  40 x 15 x 3.5 mm plate, a collar to 6.5 mm and a taper from 7.9 mm across at 6.5 mm to 5.56 mm at 22.04 mm, top at
  22.5 mm; two pins supplied.
- The window size (488 x 600 mm overall) is a common UK window size; the frame and sash sections, the timber and
  glass masses and the hinge friction are ESTIMATES (est below).
- Fitting: "place the stay central to the window ... assembled with the 2 pins" (a maker's fitting guide); where the
  pins sit (under holes 1 and 4 when the window is shut) is our estimate.
"""

from __future__ import annotations

import math

import numpy as np

MM = 0.001
TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0

# -- the window (est unless stated) -------------------------------------------------------------------------------
SILL_H = 0.040  # the window board's top above the table (est)
Z0 = TOP + SILL_H  # world z of the sill top
FRAME_W, FRAME_H = 0.488, 0.600  # overall, a common UK window size
MEMBER = 0.045  # jamb and head face width (est)
FRAME_D = 0.070  # jamb and head depth (est)
XS = 0.420  # world x of the sash's room face when shut
FRAME_X0 = XS - 0.012  # the frame's room face (est)
BOARD_X0, BOARD_X1 = XS - 0.200, XS + 0.090  # window board, room edge to outside edge (est)
SASH_W = 0.396  # 1 mm gap at each jamb
SASH_T = 0.044  # sash thickness (est)
SASH_Z0 = SILL_H + 0.003  # sash underside, 3 mm over the sill
SASH_Z1 = FRAME_H - MEMBER - 0.003  # sash top
STILE = 0.045  # sash stiles and top rail (est)
BOTTOM_RAIL = 0.060  # (est)
GLASS_T = 0.004
TIMBER = 500.0  # kg/m3, softwood (est)
GLASS = 2500.0  # kg/m3, float glass
Y_HINGE = SASH_W / 2  # hinges on the robot's left (+y) stile, at the sash's outside face
HINGE_XY = np.array([XS + SASH_T, Y_HINGE])
SASH_DAMPING = 0.3  # N m s/rad (est)
SASH_FRICTION = 0.2  # N m (est; see fidelity)
SASH_MAX = math.radians(80.0)  # casement hinges open further than any stay setting

# -- the stay: Samuel Heath P4512-B, from the maker's CAD (mm) ------------------------------------------------------
SWIVEL_OFF = 0.01111  # swivel axis in front of the sash face
S_LIFT = 0.02002  # lift pin, along the bar from the swivel axis
HOLE_S = (0.06452, 0.10502, 0.14552, 0.18602)  # hole centres along the bar from the swivel axis
HOLE_R = 0.0036  # 7.2 mm holes
BAR_W = 0.0136  # bar width (13.4 at the top, 13.8 at the foot)
BAR_D = 0.008  # bar depth
WEB_T = 0.00425  # web above the groove (the holes go through it)
GROOVE_W = 0.0081
LEG_W = (BAR_W - GROOVE_W) / 2  # 2.75 mm
BAR_S0, BAR_S1 = 0.0145, 0.2175  # the straight bar, along from the swivel axis
GROOVE_S0, GROOVE_S1 = 0.0386, 0.2105
TAIL_S = 0.26052  # the ball's centre
BALL_R = 0.007  # 14 mm ball
TAIL_END = 0.2675
NECK_W = 0.0075
YS_SHUT = -0.139  # world y of the swivel axis, window shut: the stay central on the sash (est)
BRASS = 8500.0

# Heights with the bar level and seated (a pin's taper meets the 7.2 mm hole at 11.15 mm, CAD)
PIN_TOP_H = 0.0225
PIN_SEAT_H = 0.01115  # where the taper is 7.2 mm across
SEAT_GAP = 0.0002  # drawn 0.2 mm above its seat, so the start is free of overlap
WEB_BOT_H = PIN_SEAT_H + SEAT_GAP  # the web's underside (the groove's ceiling), as drawn
LIFT_H = WEB_BOT_H + 0.00475  # lift pin height (fixed: it rides on the sash): 4.75 mm over the web's underside
WEB_TOP_H = WEB_BOT_H + WEB_T
LEG_BOT_H = WEB_BOT_H - (BAR_D - WEB_T)
BALL_H = LIFT_H + 0.0070  # ball centre 7.0 mm above the lift pin (CAD)
LIFT_Z = SILL_H + LIFT_H  # the lift pin above the table (the base frame's z is above the table)
BAR_MASS = 0.17  # kg, brass, from the shapes (about 20 cm3)
SWIVEL_MASS = 0.03
LIFT_LOWER, LIFT_UPPER = -0.08, 0.45  # rad; the bar droops onto the board long before -0.08
LIFT_DAMPING, SWIVEL_DAMPING = 0.002, 0.002

# -- the pins: as supplied, on the board ---------------------------------------------------------------------------
PIN_PLATE = (0.015, 0.040, 0.0035)  # x (across the bar when shut), y (along it), thickness
COLLAR_R, COLLAR_H = 0.0047, 0.0065
CONE_R0, CONE_Z0 = 0.00395, 0.0065
CONE_R1, CONE_Z1 = 0.00278, 0.02204
TIP_R = 0.0025


def pin_r(h: float) -> float:
    """Radius of a pin's taper at height h over the sill."""
    if h <= CONE_Z0:
        return COLLAR_R
    if h >= CONE_Z1:
        return TIP_R
    return CONE_R0 + (h - CONE_Z0) * (CONE_R1 - CONE_R0) / (CONE_Z1 - CONE_Z0)


PIN_X = XS - SWIVEL_OFF
PINS = {"A": np.array([PIN_X, YS_SHUT + HOLE_S[0]]), "B": np.array([PIN_X, YS_SHUT + HOLE_S[3]])}  # shut: holes 1, 4


# -- kinematics ----------------------------------------------------------------------------------------------------
def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


SWIVEL_SHUT = np.array([PIN_X, YS_SHUT])


def swivel_xy(theta: float) -> np.ndarray:
    """World (x, y) of the swivel axis with the sash open ``theta`` rad (0 shut, positive outward)."""
    return HINGE_XY + rz(theta) @ (SWIVEL_SHUT - HINGE_XY)


def theta_for(rho: float) -> float:
    """The sash angle at which the swivel axis is ``rho`` from pin A in plan (bisection)."""
    lo, hi = 0.0, SASH_MAX
    f = lambda t: float(np.linalg.norm(swivel_xy(t) - PINS["A"])) - rho  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError(f"no sash angle puts the swivel {rho:.4f} m from pin A")
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


THETA_START = theta_for(HOLE_S[1])  # on the second hole (the first open one): about 13 degrees
_B0 = swivel_xy(THETA_START)
_D0 = PINS["A"] - _B0
PHI_START = math.atan2(_D0[1], _D0[0])  # the bar's heading at the start (world)
BASE_YAW = PHI_START - math.pi / 2  # the window's base frame: its +y along the bar at the start
BASE_ORIGIN = (float(_B0[0]), float(_B0[1]), TOP)  # the swivel axis at table height, at the start
ALPHA_START = PHI_START - (math.pi / 2 + THETA_START)  # the bar's angle from the sash (toward the hinge) at the start
SWIVEL_LOWER, SWIVEL_UPPER = -ALPHA_START, math.radians(150.0) - ALPHA_START
SASH_LOWER, SASH_UPPER = -THETA_START, SASH_MAX - THETA_START
_RB = np.array([[math.cos(BASE_YAW), -math.sin(BASE_YAW)], [math.sin(BASE_YAW), math.cos(BASE_YAW)]])


def to_base_xy(p_world) -> np.ndarray:
    """World (x, y) to the base frame's (x, y)."""
    return _RB.T @ (np.asarray(p_world, dtype=float)[:2] - np.array(BASE_ORIGIN[:2]))


def to_world_xy(p_base) -> np.ndarray:
    return _RB @ np.asarray(p_base, dtype=float)[:2] + np.array(BASE_ORIGIN[:2])


HINGE_BASE = to_base_xy(HINGE_XY)
LIFT_ANCHOR = (0.0, S_LIFT, LIFT_Z)
SWIVEL_ANCHOR = (0.0, 0.0, LIFT_Z)
SASH_ANCHOR = (float(HINGE_BASE[0]), float(HINGE_BASE[1]), SASH_Z0)  # above the table


def bar_point(s: float, h: float, n: float = 0.0) -> np.ndarray:
    """A point riding on the stay, base frame at rest: ``s`` along the bar from the swivel, ``h`` over the sill,
    ``n`` across (base x)."""
    return np.array([n, s, SILL_H + h])


def world_of_bar(q, pts) -> np.ndarray:
    """World points of bar points (base frame at rest) with joints q = (sash, swivel, lift)."""
    js, jw, jl = q
    p = np.atleast_2d(np.asarray(pts, dtype=float)).copy()
    c, s = math.cos(jl), math.sin(jl)  # lift about +x through the lift pin
    dy, dz = p[:, 1] - S_LIFT, p[:, 2] - LIFT_Z
    p[:, 1], p[:, 2] = S_LIFT + c * dy - s * dz, LIFT_Z + s * dy + c * dz
    p[:, :2] = p[:, :2] @ rz(jw).T  # swivel about +z through the base origin
    p[:, :2] = (p[:, :2] - HINGE_BASE) @ rz(js).T + HINGE_BASE  # sash about +z through the hinge
    out = np.empty_like(p)
    out[:, :2] = p[:, :2] @ _RB.T + np.array(BASE_ORIGIN[:2])
    out[:, 2] = p[:, 2] + TOP
    return out


def disc_world(q) -> np.ndarray:
    return world_of_bar(q, bar_point(TAIL_S, BALL_H))[0]


def q_for(rho: float, web_h_at_pin: float | None = None, lift: float | None = None) -> tuple[float, float, float]:
    """Joints with pin A ``rho`` along the bar from the swivel (in plan, under the bar's centre line) and either
    the web's underside at the pin at ``web_h_at_pin`` over the sill or the lift joint at ``lift``."""
    theta = theta_for(rho)
    b = swivel_xy(theta)
    d = PINS["A"] - b
    phi = math.atan2(d[1], d[0])
    js = theta - THETA_START
    jw = (phi - PHI_START) - js
    if lift is None:
        lift = lift_for(rho, web_h_at_pin)
    return (js, jw, lift)


def lift_for(rho: float, web_h: float) -> float:
    """The lift angle that puts the web's underside at height ``web_h`` at a plan distance ``rho`` from the swivel."""
    lo, hi = LIFT_LOWER, LIFT_UPPER
    for _ in range(80):
        mid = (lo + hi) / 2
        c, s = math.cos(mid), math.sin(mid)
        # bar point at (s_b, WEB_BOT_H) lifted: plan distance S_LIFT + c*(s_b-S_LIFT) - s*(WEB_BOT_H-LIFT_H) = rho
        sb = S_LIFT + (rho - S_LIFT + s * (WEB_BOT_H - LIFT_H)) / c
        h = LIFT_H + s * (sb - S_LIFT) + c * (WEB_BOT_H - LIFT_H)
        lo, hi = (mid, hi) if h < web_h else (lo, mid)
    return (lo + hi) / 2


def pin_in_bar(q, pin: str = "A") -> tuple[float, float, float]:
    """Where pin ``pin`` is in the bar's own frame: (s along the bar from the swivel, n across, h of the web's
    underside above the sill at that place)."""
    js, jw, jl = q
    p = to_base_xy(PINS[pin])
    p = (p - HINGE_BASE) @ rz(-js).T + HINGE_BASE
    p = p @ rz(-jw).T  # bar frame in plan: the bar along +y, lifted about x
    c, s = math.cos(jl), math.sin(jl)
    sb = S_LIFT + (p[1] - S_LIFT + s * (WEB_BOT_H - LIFT_H)) / c
    h = LIFT_H + s * (sb - S_LIFT) + c * (WEB_BOT_H - LIFT_H)
    return float(sb), float(p[0]), float(h)


# -- shapes --------------------------------------------------------------------------------------------------------
WHITE = (0.93, 0.93, 0.90)  # painted timber
BOARD = (0.86, 0.84, 0.80)
BRASS_COL = (0.78, 0.62, 0.30)
DARK = (0.30, 0.30, 0.32)


def ring(cx, cy, z0, z1, r_in, thick=0.0012, sides=16, color=None):
    """A round hole's wall: ``sides`` box staves whose inner faces are tangent to the circle ``r_in``."""
    import mechkit as mk

    out = []
    tang = 2 * r_in * math.tan(math.pi / sides) + 0.0004
    for k in range(sides):
        a = 2 * math.pi * k / sides
        u = np.array([math.cos(a), math.sin(a)])
        c = np.array([cx, cy]) + u * (r_in + thick / 2)
        out.append(mk.box((c[0], c[1], (z0 + z1) / 2), (thick, tang, z1 - z0), color, ypr=(a, 0.0, 0.0)))
    return out


def cone(cx, cy, z0, r0, z1, r1, thick=0.0008, sides=16, color=None):
    """A convex taper: ``sides`` thin staves whose outer faces are tangent to the cone inscribed in the circles
    (z0, r0)-(z1, r1) (corners on the true cone), each leaning in at the taper's angle."""
    import mechkit as mk

    out = []
    k_in = math.cos(math.pi / sides)
    a_tilt = math.atan2((r0 - r1) * k_in, z1 - z0)
    length = math.hypot((r0 - r1) * k_in, z1 - z0)
    rm, zm = (r0 + r1) / 2 * k_in, (z0 + z1) / 2
    tang = 2 * min(r0, r1) * k_in * math.tan(math.pi / sides) + 0.0001  # no corner stands outside the cone
    for k in range(sides):
        a = 2 * math.pi * k / sides + math.pi / sides
        u = np.array([math.cos(a), math.sin(a)])
        n = np.array([math.cos(a_tilt) * u[0], math.cos(a_tilt) * u[1], math.sin(a_tilt)])
        c = np.array([cx + rm * u[0], cy + rm * u[1], zm]) - n * thick / 2
        out.append(mk.box(tuple(c), (thick, tang, length), color, ypr=(a, -a_tilt, 0.0)))
    return out


def pin_shapes(px, py):
    """One pin in the board's frame (origin on the table under the board's centre), standing on the board."""
    import mechkit as mk

    z = SILL_H
    out = [mk.box((px, py, z + PIN_PLATE[2] / 2), PIN_PLATE, BRASS_COL),
           mk.cylinder((px, py, z + (PIN_PLATE[2] + COLLAR_H) / 2), COLLAR_R, COLLAR_H - PIN_PLATE[2], "Z",
                       BRASS_COL),
           mk.cylinder((px, py, z + (CONE_Z0 + PIN_TOP_H) / 2), TIP_R, PIN_TOP_H - CONE_Z0, "Z", BRASS_COL)]
    zm = (CONE_Z0 + CONE_Z1) / 2  # two segments, so the staves' slits stay under 0.25 mm
    out += cone(px, py, z + CONE_Z0, CONE_R0, z + zm, pin_r(zm), color=BRASS_COL)
    out += cone(px, py, z + zm, pin_r(zm), z + CONE_Z1, CONE_R1, color=BRASS_COL)
    return out


def _b(c_world_rel, size, yaw_world, color):
    """A box given world-aligned (centre relative to the base origin, world yaw) put into the base frame."""
    import mechkit as mk

    c = np.asarray(c_world_rel, dtype=float)
    xy = _RB.T @ c[:2]
    return mk.box((float(xy[0]), float(xy[1]), float(c[2])), size, color, ypr=(yaw_world - BASE_YAW, 0.0, 0.0))


def frame_shapes():
    """Jambs and head (the window's fixed base), base frame. The sill is the separate board object."""
    ox, oy = BASE_ORIGIN[0], BASE_ORIGIN[1]
    cx = FRAME_X0 + FRAME_D / 2 - ox
    zj = (SILL_H + FRAME_H) / 2
    out = []
    for sgn in (1.0, -1.0):
        cy = sgn * (FRAME_W / 2 - MEMBER / 2) - oy
        out.append(_b((cx, cy, zj), (FRAME_D, MEMBER, FRAME_H - SILL_H), 0.0, WHITE))
    out.append(_b((cx, -oy, FRAME_H - MEMBER / 2), (FRAME_D, FRAME_W - 2 * MEMBER, MEMBER), 0.0, WHITE))
    return out


def sash_shapes():
    """The sash at the start (open THETA_START), base frame: stiles, rails, the stay's plate and ear. The pane is
    left out (no transparent material); its mass is in the rails' (task.py)."""
    out = []
    t = THETA_START
    rot = rz(t)
    h = SASH_Z1 - SASH_Z0

    def put(x_shut, y_shut, z, size, color=WHITE):
        p = HINGE_XY + rot @ (np.array([x_shut, y_shut]) - HINGE_XY)
        out.append(_b((p[0] - BASE_ORIGIN[0], p[1] - BASE_ORIGIN[1], z), size, t, color))

    xc = XS + SASH_T / 2
    for sgn in (1.0, -1.0):
        put(xc, sgn * (SASH_W / 2 - STILE / 2), SASH_Z0 + h / 2, (SASH_T, STILE, h))
    inner = SASH_W - 2 * STILE
    put(xc, 0.0, SASH_Z0 + BOTTOM_RAIL / 2, (SASH_T, inner, BOTTOM_RAIL))
    put(xc, 0.0, SASH_Z1 - STILE / 2, (SASH_T, inner, STILE))
    # The stay's plate (45.4 x 13.9 x 3.4 mm) on the bottom rail's room face, and its ear (two lugs) to the swivel.
    plate_z = LIFT_Z - 0.0101 + 0.0139 / 2  # CAD: the plate runs 10.1 mm below the lift pin to 3.75 mm above it
    put(XS - 0.0017, YS_SHUT, plate_z, (0.0034, 0.0454, 0.0139), BRASS_COL)
    for dz in (-0.0082, 0.0019):
        put(XS - 0.0034 - 0.0040, YS_SHUT, LIFT_Z + dz, (0.0080, 0.0159, 0.00375), BRASS_COL)
    return out


def glass_box_shut():
    """The pane's space (between the stiles and rails, through the sash's thickness), window shut, world:
    (centre, size)."""
    z0, z1 = SASH_Z0 + BOTTOM_RAIL, SASH_Z1 - STILE
    return (np.array([XS + SASH_T / 2, 0.0, TOP + (z0 + z1) / 2]),
            np.array([SASH_T, SASH_W - 2 * STILE, z1 - z0]))


def swivel_shapes():
    """The swivel knuckle (between the ear's lugs, carrying the lift pin), base frame at rest."""
    import mechkit as mk

    return [mk.box((0.0, (-0.0028 + 0.0140) / 2, LIFT_Z), (0.0135, 0.0168, 0.0150), BRASS_COL)]


def bar_shapes():
    """The stay's bar at rest (along base +y from the swivel, level, seated), its four holes through the web, the
    groove's two legs, the pivot end, the curled neck and the ball (a 14 mm cylinder across the bar)."""
    import mechkit as mk

    out = []
    web_z = SILL_H + WEB_BOT_H + WEB_T / 2
    cut = 0.0049  # the web is cut to a square 9.8 mm about each hole; the staves fill it
    holes = [(0.0, s - (BAR_S0 + BAR_S1) / 2, 2 * cut, 2 * cut) for s in HOLE_S]
    plate = mk.plate_with_holes((0.0, (BAR_S0 + BAR_S1) / 2, web_z), (BAR_W, BAR_S1 - BAR_S0), holes, WEB_T,
                                BRASS_COL)
    out += list(plate)
    for s in HOLE_S:
        out += ring(0.0, s, SILL_H + WEB_BOT_H, SILL_H + WEB_BOT_H + WEB_T, HOLE_R, color=BRASS_COL)
    leg_h = BAR_D - WEB_T
    leg_z = SILL_H + LEG_BOT_H + leg_h / 2
    for sgn in (1.0, -1.0):
        out.append(mk.box((sgn * (GROOVE_W / 2 + LEG_W / 2), (GROOVE_S0 + GROOVE_S1) / 2, leg_z),
                          (LEG_W, GROOVE_S1 - GROOVE_S0, leg_h), BRASS_COL))
    # solid under the web before and after the groove
    out.append(mk.box((0.0, (BAR_S0 + GROOVE_S0) / 2, leg_z), (BAR_W, GROOVE_S0 - BAR_S0, leg_h),
                      BRASS_COL))
    out.append(mk.box((0.0, (GROOVE_S1 + BAR_S1) / 2, leg_z), (BAR_W, BAR_S1 - GROOVE_S1, leg_h),
                      BRASS_COL))
    # pivot ear above the bar at the lift pin
    out.append(mk.box((0.0, (BAR_S0 + 0.026) / 2, SILL_H + WEB_TOP_H + 0.0032), (0.0066, 0.026 - BAR_S0, 0.0064),
                      BRASS_COL))
    # the neck: from the bar's end up to the ball
    s_a, z_a = BAR_S1 - 0.004, SILL_H + WEB_BOT_H + 0.001
    s_b, z_b = TAIL_S - BALL_R + 0.001, SILL_H + BALL_H - 0.001
    length = math.hypot(s_b - s_a, z_b - z_a)
    pitch = math.atan2(z_b - z_a, s_b - s_a)
    out.append(mk.box((0.0, (s_a + s_b) / 2, (z_a + z_b) / 2), (length, NECK_W, 0.005), BRASS_COL,
                      ypr=(math.pi / 2, -pitch, 0.0)))
    out.append(mk.cylinder((0.0, TAIL_S, SILL_H + BALL_H), BALL_R, 2 * BALL_R, "X", BRASS_COL))
    return out


def board_shapes():
    """The window board (the frame's sill and the room-side board in one) and the two pins, in the board's own
    frame: origin on the table at world (BOARD_CX, 0)."""
    import mechkit as mk

    out = [mk.box((0.0, 0.0, SILL_H / 2), (BOARD_X1 - BOARD_X0, FRAME_W, SILL_H), BOARD)]
    for p in PINS.values():
        out += pin_shapes(p[0] - BOARD_CX, p[1])
    return out


BOARD_CX = (BOARD_X0 + BOARD_X1) / 2
PANE_BOX_SHUT = glass_box_shut()
