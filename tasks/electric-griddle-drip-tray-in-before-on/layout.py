"""Every dimension and threshold of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases (away from the robot), y to the left, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z 0.7475. Heights are above the table top unless named
*_W (world z).

A Presto 22 in electric griddle (07061 class: body 56 x 34 cm, 8 cm tall on its legs, est) stands with its long side
along y and its REAR edge (the grease slot and the drip-tray rails) toward the robot. Its removable drip tray runs on
two rails under that rear edge, along y; it starts pulled out 21 cm toward the robot's right, its inner 7 cm still on
the rails and clear of the drain slot. The rear grease channel ends, on the robot's right, in a drain slot through the
plate with a short spout under it, over the seated tray. The Control Master heat control is plugged into the griddle's end on the robot's
left; its dial (a 45 mm knob with a raised grip bar on top) turns about the vertical, clockwise from OFF.

The griddle's frame: origin at the centre of its footprint on the table, axes the world's. The tray is a slider along
+y (0 = pulled out, as drawn; SEATED_TRAVEL at its stop); the dial a hinge about -z (positive = clockwise seen from
above, 0 = OFF).
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the griddle (fixed): Presto 07061 class, 22 x 14 in cooking surface, body 56 x 34 cm, 8 cm tall (est) ----------
GR_X0, GR_X1 = 0.145, 0.485  # rear edge (toward the robot) .. front edge
GR_Y0, GR_Y1 = -0.200, 0.360  # the end on the robot's right .. the end on the robot's left
GR_CENTRE = ((GR_X0 + GR_X1) / 2, (GR_Y0 + GR_Y1) / 2)  # (0.315, 0.080)
PLATE_Z0, PLATE_Z1 = 0.050, 0.080  # the cast plate's underside and cooking surface (est)
LEG = 0.025
LEGS = [(x, y) for x in (0.215, 0.445) for y in (-0.160, 0.320)]  # inset legs (est), clear of the tray's path
CHANNEL_X0, CHANNEL_X1 = GR_X0 + 0.010, GR_X0 + 0.032  # the rear grease channel along the rear edge (est)
# the drain slot at the channel's end on the robot's right: through the plate, a spout under it over the seated tray
DRAIN_Y0, DRAIN_Y1 = -0.080, -0.040
CHANNEL_Y0, CHANNEL_Y1 = DRAIN_Y0, GR_Y1 - 0.010  # the channel runs from the drain to 1 cm short of the left end
DRAIN_W = CHANNEL_X1 - CHANNEL_X0 - 0.008  # 14 mm across x
SPOUT_Z0 = 0.040  # the spout hangs from the plate's underside to 5 mm over the tray's rim (0.035)
GREASE_T = 0.002  # last use's cooled grease, a 2 mm film in the channel
# rails under the rear edge (open toward -y, the robot's right): two lips hanging from the plate either side of the tray
TRAY_X = 0.175  # the tray's centre line (under the slot)
TRAY_W, TRAY_H, TRAY_L = 0.045, 0.030, 0.300  # 30 x 4.5 x 3 cm (est)
TRAY_Z0 = 0.005  # the tray rides 5 mm over the table on the rails
TRAY_WALL = 0.003
RAIL_T = 0.004
RAIL_Y0, RAIL_Y1 = GR_Y0, GR_Y0 + 0.330
SEATED_Y0 = GR_Y0 - 0.020  # seated, the tray's outer end stands 2 cm proud of the griddle's end (y -0.22 .. 0.08)
TRAVEL = 0.210  # from pulled out (its inner 7 cm on the rails, 5 cm short of the drain slot) to its stop
# its raised pull tab: a stem from the tray's outer end and an upright tab across it (est)
STEM_L, STEM_W, STEM_H = 0.075, 0.020, 0.010
TAB = (TRAY_W, 0.018, 0.055)  # 45 mm across x, 18 mm thick along y, standing 55 mm from the tray's floor level
TRAY_MASS = 0.150
TRAY_DAMPING = 2.0  # N s/m (est): it runs freely on the rails and stays where it is left
TRAY_FRICTION = 0.3  # N (est)


def tray_y_range(travel=0.0):
    """The tray body's y extent at ``travel`` (0 = pulled out)."""
    y0 = SEATED_Y0 - TRAVEL + travel
    return y0, y0 + TRAY_L


def tab_centre_y(travel=0.0):
    return SEATED_Y0 - TRAVEL + travel - STEM_L - TAB[1] / 2


# -- the Control Master heat control (fixed, plugged into the end on the robot's left) -----------------------------
PROBE_X0, PROBE_X1 = 0.280, 0.350
PROBE_Y0, PROBE_Y1 = GR_Y1, GR_Y1 + 0.100
PROBE_Z0, PROBE_Z1 = 0.045, 0.080
DIAL = (0.315, 0.410)  # the dial's axis
DIAL_R, DIAL_SKIRT_H = 0.0225, 0.012  # 45 mm across (est)
FIN = (0.034, 0.010, 0.016)  # the raised grip bar: 34 mm long (along x at OFF), 10 mm wide, 16 mm tall
DIAL_TOP = PROBE_Z1 + DIAL_SKIRT_H + FIN[2]  # 0.108
DIAL_MASS = 0.020
DIAL_UPPER = math.radians(120.0)  # OFF .. 400 F at its stop (est)
DIAL_DAMPING = 0.02  # N m s/rad: it stays where it is left (est)
SET_DEG = 90.0  # the 350 F mark, 90 degrees clockwise from OFF (est)
SET_TOL_DEG = 15.0
ON_DEG = 15.0  # past the OFF position: the element is on
OFF_FLOOR_DEG = -30.0  # readings from -30 to 15 degrees are off

# -- scoring thresholds ------------------------------------------------------------------------------------------------
SEATED_TOL = 0.010  # the tray within 1 cm of its stop
PART_IN = 0.100  # the tray slid 10 cm or more along the rails (0.25)
LET_GO = 0.040  # both grippers' finger zones 40 mm or more from the griddle, tray and control
STILL_WINDOW_S = 1.0
STILL_TRAVEL = 0.002  # over the last 1 s the tray moves under 2 mm ...
STILL_TURN_DEG = 2.0  # ... and the dial under 2 degrees
RESET_JUMP = 0.08
SUCCESS_HOLD_S = 1.0
EPS = 1e-9

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
PLATE = (0.10, 0.10, 0.11)  # non-stick black
BODY = (0.12, 0.12, 0.13)
CHANNEL = (0.04, 0.04, 0.05)
GREASE = (0.86, 0.78, 0.60)  # glossy beige
TRAY = (0.15, 0.15, 0.16)  # black plastic
PROBE = (0.10, 0.10, 0.10)
DIAL_C = (0.85, 0.85, 0.85)
POINTER = (0.85, 0.10, 0.10)
MARK = (0.97, 0.97, 0.97)  # the OFF mark
MARK_SET = (0.98, 0.62, 0.05)  # the 350 mark: orange
DRAIN = (0.01, 0.01, 0.01)  # the open slot
STEEL = (0.62, 0.62, 0.64)  # the slot's rim and the spout
CORD = (0.08, 0.08, 0.08)


def _g(x, y):
    """World (x, y) to the griddle's frame."""
    return x - GR_CENTRE[0], y - GR_CENTRE[1]


def griddle_shapes(mk):
    """The griddle's fixed parts in its own frame: plate, legs, grease channel with its film, rails, the Control
    Master with its OFF and 350 marks, and the cord (a flat strip on the table)."""
    gx, gy = _g((GR_X0 + GR_X1) / 2, (GR_Y0 + GR_Y1) / 2)
    out = [mk.box((gx, gy, (PLATE_Z0 + PLATE_Z1) / 2), (GR_X1 - GR_X0, GR_Y1 - GR_Y0, PLATE_Z1 - PLATE_Z0), PLATE)]
    for lx, ly in LEGS:
        x, y = _g(lx, ly)
        out.append(mk.box((x, y, PLATE_Z0 / 2), (LEG, LEG, PLATE_Z0), BODY))
    cx, cy = _g((CHANNEL_X0 + CHANNEL_X1) / 2, (CHANNEL_Y0 + CHANNEL_Y1) / 2)
    out.append(mk.box((cx, cy, PLATE_Z1 + 0.0005), (CHANNEL_X1 - CHANNEL_X0, CHANNEL_Y1 - CHANNEL_Y0, 0.001), CHANNEL))
    gy0 = DRAIN_Y1 + 0.004  # the film runs from the slot's rim to 2 cm short of the channel's end
    _, gy = _g(0.0, (gy0 + CHANNEL_Y1 - 0.020) / 2)
    out.append(mk.box((cx, gy, PLATE_Z1 + 0.001 + GREASE_T / 2),
                      (CHANNEL_X1 - CHANNEL_X0 - 0.006, CHANNEL_Y1 - 0.020 - gy0, GREASE_T), GREASE))
    # the drain slot: a steel rim and the dark opening, flush with the film; under the plate a spout over the tray
    _, dy = _g(0.0, (DRAIN_Y0 + DRAIN_Y1) / 2)
    out.append(mk.box((cx, dy, PLATE_Z1 + 0.001 + GREASE_T / 2),
                      (DRAIN_W + 0.006, DRAIN_Y1 - DRAIN_Y0 + 0.006, GREASE_T), STEEL))
    out.append(mk.box((cx, dy, PLATE_Z1 + 0.001 + GREASE_T + 0.0005), (DRAIN_W, DRAIN_Y1 - DRAIN_Y0, 0.001), DRAIN))
    out.append(mk.box((cx, dy, (SPOUT_Z0 + PLATE_Z0) / 2), (DRAIN_W, DRAIN_Y1 - DRAIN_Y0, PLATE_Z0 - SPOUT_Z0), STEEL))
    ry = (RAIL_Y0 + RAIL_Y1) / 2
    for rx in (TRAY_X - TRAY_W / 2 - RAIL_T / 2 - 0.0015, TRAY_X + TRAY_W / 2 + RAIL_T / 2 + 0.0015):
        x, y = _g(rx, ry)
        out.append(mk.box((x, y, (0.012 + PLATE_Z0) / 2), (RAIL_T, RAIL_Y1 - RAIL_Y0, PLATE_Z0 - 0.012), BODY))
    px, py = _g((PROBE_X0 + PROBE_X1) / 2, (PROBE_Y0 + PROBE_Y1) / 2)
    out.append(mk.box((px, py, (PROBE_Z0 + PROBE_Z1) / 2), (PROBE_X1 - PROBE_X0, PROBE_Y1 - PROBE_Y0,
                                                            PROBE_Z1 - PROBE_Z0), PROBE))
    dx, dy = _g(*DIAL)
    # OFF: a white bar beside the dial toward the robot (the pointer's side at OFF); 350: an orange bar on the robot's
    # left (90 degrees clockwise seen from above), each 9 x 5 mm, 1.5 mm proud of the control's top
    out.append(mk.box((dx - DIAL_R - 0.0065, dy, PROBE_Z1 + 0.00075), (0.009, 0.005, 0.0015), MARK))  # OFF
    out.append(mk.box((dx, dy + DIAL_R + 0.0065, PROBE_Z1 + 0.00075), (0.005, 0.009, 0.0015), MARK_SET))  # 350
    kx, ky = _g(PROBE_X1 + 0.010, DIAL[1] + 0.020)
    out.append(mk.box((kx, ky, (0.004 + PROBE_Z1 - 0.01) / 2), (0.020, 0.012, PROBE_Z1 - 0.014), CORD))
    sx, sy = _g((PROBE_X1 + 0.65) / 2, DIAL[1] + 0.020)
    out.append(mk.box((sx, sy, 0.002), (0.65 - PROBE_X1, 0.012, 0.004), CORD))
    return out


def dial_shapes(mk):
    """The dial in the griddle's frame at OFF: the skirt, the raised grip bar along x and a red pointer at the bar's
    end toward the robot."""
    cx, cy = _g(*DIAL)
    return [
        mk.cylinder((cx, cy, PROBE_Z1 + DIAL_SKIRT_H / 2), DIAL_R, DIAL_SKIRT_H, "Z", DIAL_C),
        mk.box((cx, cy, PROBE_Z1 + DIAL_SKIRT_H + FIN[2] / 2), FIN, DIAL_C),
        mk.box((cx - FIN[0] / 2 + 0.003, cy, DIAL_TOP + 0.0005), (0.006, 0.006, 0.001), POINTER),
    ]


def tray_shapes(mk):
    """The drip tray as drawn (pulled out), in the griddle's frame: an open tray, its stem and the upright pull tab."""
    y0, y1 = tray_y_range(0.0)
    cx, cy = _g(TRAY_X, (y0 + y1) / 2)
    out = list(mk.open_box((cx, cy, TRAY_Z0 + TRAY_H / 2), (TRAY_W, TRAY_L, TRAY_H), TRAY_WALL, TRAY_WALL, TRAY))
    sx, sy = _g(TRAY_X, y0 - STEM_L / 2)
    out.append(mk.box((sx, sy, TRAY_Z0 + STEM_H / 2), (STEM_W, STEM_L + 0.002, STEM_H), TRAY))
    tx, ty = _g(TRAY_X, tab_centre_y(0.0))
    out.append(mk.box((tx, ty, TRAY_Z0 + TAB[2] / 2), TAB, TRAY))
    return out
