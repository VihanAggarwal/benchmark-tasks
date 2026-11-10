"""Every dimension of the task, in one place (metres, kilograms, radians unless named *_DEG; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W. The
highway is at 1:80 scale.

A one-way, two-lane highway runs across the table along y. Traffic flows toward -y (from the robot's left to its
right), so upstream is +y and downstream is -y. Across the road, from the arms outward: a grass verge, the near
(right-hand) shoulder, the near lane (lane 1, where the car crashed), the far lane (lane 2), the far (left-hand)
shoulder and a grass median. The crashed car stands in the near lane, slightly askew. The fire engine starts upstream
in the far lane, heading downstream; the ambulance starts downstream on the near shoulder, facing upstream.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the road: fixed 1 mm sheets across x, all along y ---------------------------------------------------------------
SHEET_H = 0.001
ROAD_TOP = SHEET_H  # height of the road surface above the table top
ROAD_Y = (-0.450, 0.450)  # 900 mm (72 m) of highway
VERGE_X = (0.100, 0.150)  # grass, nearest the arms
ROAD_X = (0.150, 0.295)  # the asphalt: near shoulder, lane 1, lane 2, far shoulder
MEDIAN_X = (0.295, 0.360)  # grass
SHOULDER_NEAR = (0.150, 0.1875)  # 37.5 mm (3.0 m)
LANE1 = (0.1875, 0.2325)  # the near lane, 45 mm (3.6 m): the crash lane
LANE2 = (0.2325, 0.2775)  # the far lane
SHOULDER_FAR = (0.2775, 0.295)  # 17.5 mm (1.4 m)
LANES = (LANE1[0], LANE2[1])  # both lanes: x 0.1875 .. 0.2775

# The road is laid as strips 1 mm thick, so its painted lines (3 mm wide, wider than true scale so the top camera
# shows them) are flush with the asphalt and nothing parked across them rocks: from the arms outward, the near
# shoulder, the white edge line, lane 1, the dashed white lane line (dashes and asphalt gaps), lane 2, the yellow edge
# line and the far shoulder.
LINE_W = 0.003
EDGE_NEAR_X = LANE1[0]  # white edge line between the near shoulder and lane 1 (centred on the boundary)
EDGE_FAR_X = LANE2[1]  # yellow edge line between lane 2 and the far shoulder
LANE_LINE_X = LANE1[1]  # white dashed line between the lanes
DASH_LEN = 0.0375  # 3 m dashes ...
DASH_YS = (-0.375, -0.225, -0.075, 0.075, 0.225, 0.375)  # ... every 12 m (150 mm)


def road_strips() -> list[tuple[str, tuple[float, float], tuple[float, float], str]]:
    """The road's strips: (name, (x0, x1), (y0, y1), colour key), side by side, together exactly ROAD_X x ROAD_Y."""
    h = LINE_W / 2
    full = ROAD_Y
    strips = [
        ("shoulder_near", (ROAD_X[0], EDGE_NEAR_X - h), full, "asphalt"),
        ("edge_line_near", (EDGE_NEAR_X - h, EDGE_NEAR_X + h), full, "white"),
        ("lane_1", (EDGE_NEAR_X + h, LANE_LINE_X - h), full, "asphalt"),
        ("lane_2", (LANE_LINE_X + h, EDGE_FAR_X - h), full, "asphalt"),
        ("edge_line_far", (EDGE_FAR_X - h, EDGE_FAR_X + h), full, "yellow"),
        ("shoulder_far", (EDGE_FAR_X + h, ROAD_X[1]), full, "asphalt"),
    ]
    lx = (LANE_LINE_X - h, LANE_LINE_X + h)
    y = ROAD_Y[0]
    for i, yc in enumerate(DASH_YS):
        y0, y1 = yc - DASH_LEN / 2, yc + DASH_LEN / 2
        strips.append((f"lane_line_gap_{i}", lx, (y, y0), "asphalt"))
        strips.append((f"lane_line_dash_{i}", lx, (y0, y1), "white"))
        y = y1
    strips.append((f"lane_line_gap_{len(DASH_YS)}", lx, (y, ROAD_Y[1]), "asphalt"))
    return strips


# Lane arrows: painted on top of the lanes (0.5 mm thick) near both ends of the road, far from any parking place.
MARK_T = 0.0005
# Each arrow points downstream (-y): a shaft and two 45-degree head strokes.
ARROW_W = 0.005  # stroke width
ARROW_TIP = 0.020  # the tip is this far downstream of the arrow's centre; the tail this far upstream (40 mm long)
HEAD_LEN = 0.016
_HU, _HN = HEAD_LEN / 2, ARROW_W / 2
HEAD_DX = (_HU + _HN) / math.sqrt(2.0)  # head stroke centres beside the shaft ...
HEAD_DY = (_HU - _HN) / math.sqrt(2.0)  # ... and upstream of the tip
ARROWS = (  # (name, x, y of the centre)
    ("arrow_lane1_up", (LANE1[0] + LANE1[1]) / 2, 0.415),
    ("arrow_lane2_up", (LANE2[0] + LANE2[1]) / 2, 0.415),
    ("arrow_lane1_down", (LANE1[0] + LANE1[1]) / 2, -0.415),
    ("arrow_lane2_down", (LANE2[0] + LANE2[1]) / 2, -0.415),
)

# -- the crashed car (fixed): a 1:80 sedan, 22 x 56 mm (1.8 x 4.5 m), turned 10 degrees, in lane 1 ----------------------
CAR_XY = (0.211, -0.030)
CAR_YAW = math.radians(10.0)
CAR_SIZE = (0.022, 0.056, 0.018)  # x, y, z before the yaw
CAR_CABIN_SIZE = (0.018, 0.030, 0.009)  # on the body's top, same centre and yaw
CAR_LOW_Z = ROAD_TOP  # the car stands on the road


def car_corners() -> tuple[tuple[float, float], ...]:
    """The crashed car's outline seen from above, counter-clockwise (its body; the cabin is inside it)."""
    c, s = math.cos(CAR_YAW), math.sin(CAR_YAW)
    hx, hy = CAR_SIZE[0] / 2, CAR_SIZE[1] / 2
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return tuple((CAR_XY[0] + c * u - s * v, CAR_XY[1] + s * u + c * v) for u, v in pts)


CAR_UP_Y = max(p[1] for p in car_corners())  # -0.000515: the car's upstream end (the buffer is measured from here)
CAR_DOWN_Y = min(p[1] for p in car_corners())  # -0.059485: its downstream end (the ambulance gap from here)

# -- the two vehicles: loose rigid bodies (mechkit compounds) --------------------------------------------------------
# Vehicle frame: origin at the middle of the vehicle's length, on its centreline, at the bottom; +u the front (the
# windscreen and the light bar), +v its own left (the fire engine's pump-panel side), +z up. Each entry:
# (name, centre (u, v, z), size (u, v, z), colour key). No two shapes overlap (mechkit spreads the mass by volume).
ENGINE_MASS = 0.080
ENGINE_SHAPES = (
    ("body", (-0.001, 0.0, 0.020), (0.123, 0.031, 0.040), "engine_red"),  # u -62.5 .. 60.5, v -15.5 .. 15.5
    ("windscreen", (0.0615, 0.0, 0.029), (0.002, 0.027, 0.010), "black"),  # front face, z 24-34
    ("light_bar", (0.055, 0.0, 0.0415), (0.006, 0.024, 0.003), "white"),  # cab roof, at the front
    ("pump_panel", (0.0225, 0.016, 0.022), (0.015, 0.001, 0.028), "silver"),  # left side, u 15-30, z 8-36
    ("pump_top", (0.0225, 0.0105, 0.0405), (0.015, 0.010, 0.001), "silver"),  # roof over the panel, left edge
)
AMB_MASS = 0.050
AMB_SHAPES = (
    ("body_low", (-0.0005, 0.0, 0.010), (0.085, 0.030, 0.020), "amb_white"),  # u -43 .. 42, z 0-20
    ("stripe", (-0.0005, 0.0, 0.022), (0.085, 0.030, 0.004), "amb_red"),  # z 20-24
    ("body_high", (-0.0005, 0.0, 0.0295), (0.085, 0.030, 0.011), "amb_white"),  # z 24-35
    ("rear_doors", (-0.0435, 0.0, 0.017), (0.001, 0.026, 0.026), "door_grey"),  # rear face, u -44 .. -43, z 4-30
    ("windscreen", (0.043, 0.0, 0.027), (0.002, 0.026, 0.010), "black"),  # front face, u 42 .. 44, z 22-32
    ("light_bar", (0.037, 0.0, 0.0365), (0.006, 0.024, 0.003), "amb_red"),  # roof, at the front
)
# Outlines the scorer uses: (back, front, right, left, top) in the vehicle frame, m (front/back along u, left/right
# along v). The engine is 125 mm long and 32 mm across at the pump panel; the ambulance 88 x 30 mm.
OUTLINE = {
    "fire_engine": (0.0625, 0.0625, 0.0155, 0.0165, 0.043),
    "ambulance": (0.044, 0.044, 0.015, 0.015, 0.038),
}
VEHICLES = ("fire_engine", "ambulance")

# Start: the engine upstream in lane 2 heading downstream (pump panel toward the median); the ambulance downstream on
# the near shoulder facing upstream (rear doors away from the car). (origin x, y, heading) Both start beyond the
# 250 mm partial-credit reach (engine buffer 268 mm, ambulance gap 266.5 mm), so a vehicle merely nudged sideways
# into a lane earns nothing: it has to be brought along the road too.
START = {
    "fire_engine": (0.2555, 0.330, -math.pi / 2),
    "ambulance": (0.170, -0.370, math.pi / 2),
}

# -- the rule (scoring.py) ---------------------------------------------------------------------------------------------
TILT_PARK_DEG = 10.0  # 'parked': upright within 10 degrees ...
FLOOR_TOL = 0.003  # ... its lowest point within 3 mm of the road surface ...
ROAD_CENTRE_X = ROAD_X  # ... its centre over the road (x 0.150 .. 0.295) ...
STILL_WINDOW_S = 0.5  # ... and still: its origin's path over the last 0.5 s, turns counted at its corner, under
STILL_PATH = 0.005  # 5 mm (positions, not the simulator's velocities, which chatter while a body is gripped)
TIPPED_DEG = 45.0  # a vehicle tilted more than this is on its side or roof: the trial scores 0
TOUCH_GAP = 0.002  # a vehicle within 2 mm of the car or of the other vehicle (seen from above): the trial scores 0
# Fire engine (the block)
ANGLE_DEG = (30.0, 60.0)  # its long axis to the lanes
BOTH_LANES = (LANE1[0] + 0.003, LANE2[1] - 0.003)  # x 0.1905 .. 0.2745: its outline spans both lanes (3 mm allowed)
CRASH_LANE_COVER = 0.028  # partial: its outline covers at least 28 mm of the crash lane's 45 mm width
BAND = (0.050, 0.125)  # 4 to 10 m: the engine's buffer short of the car, the ambulance's gap beyond it
FAR_MAX = 0.250  # 20 m: partial credit only within this
# Ambulance (in the protected zone)
STRAIGHT_DEG = 15.0  # its front within 15 degrees of downstream (-y): straight in the lane, rear doors to the car
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a vehicle moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)


def turn_radius(name: str) -> float:
    """Turns about the vertical are counted at the outline's farthest corner from the origin."""
    b, f, r, l, _ = OUTLINE[name]
    return math.hypot(max(b, f), max(r, l))


# -- where the demo puts them (the scorer does not use these) ----------------------------------------------------------
TARGET = {
    "fire_engine": (0.232, 0.1425, -3 * math.pi / 4),  # angled 45 deg, front toward the near shoulder and downstream
    "ambulance": (0.210, -0.190, -math.pi / 2),  # lane 1's centre, front downstream
}

# -- colours (sRGB) ----------------------------------------------------------------------------------------------------
COLOURS = {
    "grass": (0.20, 0.42, 0.18),
    "asphalt": (0.16, 0.16, 0.17),
    "white": (0.95, 0.95, 0.95),
    "yellow": (0.95, 0.78, 0.10),
    "car": (0.12, 0.25, 0.60),  # blue
    "car_glass": (0.32, 0.40, 0.50),
    "engine_red": (0.78, 0.06, 0.05),
    "silver": (0.80, 0.81, 0.83),  # aluminium tape
    "black": (0.05, 0.05, 0.05),
    "amb_white": (0.96, 0.96, 0.94),
    "amb_red": (0.85, 0.08, 0.08),
    "door_grey": (0.30, 0.30, 0.32),
}
