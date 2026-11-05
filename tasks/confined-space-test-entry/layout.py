"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are given above the table top and the
absolute value is TOP + that.
"""

TOP = 0.7475  # station table top

# The tank: a closed steel box fixed to the table, built from fixed plates (floor, four walls, a deck with two
# manway openings). Footprint x 0.280-0.440, y -0.240..+0.240.
TANK_X0, TANK_X1 = 0.280, 0.440
TANK_Y0, TANK_Y1 = -0.240, 0.240
FLOOR_H = 0.008  # floor plate, z 0-0.008
WALL_T = 0.008  # walls 8 mm thick, z 0.008-0.058
WALL_TOP_H = 0.058
DECK_T = 0.008  # deck plate, z 0.058-0.066
DECK_H = WALL_TOP_H + DECK_T  # 0.066: the deck's top ('the tank's top', z_d)
FLOOR_Z = TOP + FLOOR_H
DECK_Z = TOP + DECK_H

# The two manway openings in the deck, 120 mm along x by 50 mm along y.
MANWAY_X = 0.360
MANWAY_Y = {"left": 0.180, "right": -0.180}
MANWAY_HALF = (0.060, 0.025)  # x 0.300-0.420; y +0.155..+0.205 and -0.205..-0.155

# Manway covers: loose dark grey polyethylene plates (0.95 g/cm3) lying centred over the openings, 10 mm larger
# than the opening on every side.
COVER_SIZE = (0.140, 0.070, 0.020)
COVER_MASS = 0.19  # 196 cm3 of polyethylene
COVER_HALF = (COVER_SIZE[0] / 2, COVER_SIZE[1] / 2)

# Cover stands: fixed grey rubber mats beside the tank, one in front of each arm.
STAND_X = 0.360
STAND_Y = {"left": 0.310, "right": -0.310}
STAND_SIZE = (0.150, 0.090, 0.005)
STAND_TOP = TOP + STAND_SIZE[2]

# Gas probe: a loose upright aluminium rod with a sensor tip (its lower end), standing in a fixed holder.
PROBE_RADIUS = 0.008  # 16 mm across
PROBE_LEN = 0.100
PROBE_MASS = 0.05  # 20 cm3: a hollow-bored aluminium body (solid would be 54 g)
PROBE_GRIP_DOWN = 0.025  # the grip point: on the probe's axis, this far below its upper end
HOLDER_X, HOLDER_Y = 0.200, 0.180
HOLDER_INNER = 0.024  # square socket, 4 mm round the probe on every side
HOLDER_WALL = 0.006
HOLDER_H = 0.025

# Depth gauge: a fixed white post just outside the tank's front wall, in line with the left manway, with three
# coloured bands at the probe-tip heights to be tested (sections stacked from the table up).
GAUGE_X, GAUGE_Y = 0.447, 0.180  # x 0.441-0.453: 1 mm in front of the front wall
GAUGE_POST = 0.010  # white sections 10 x 10 mm
GAUGE_BAND = 0.012  # coloured sections 12 x 12 mm
BANDS = (  # (name, bottom, top) of each test depth, heights of the probe's tip above the table, top first
    ("top", 0.045, 0.056),
    ("middle", 0.028, 0.039),
    ("bottom", 0.011, 0.022),
)

# The tool: a loose orange-painted aluminium block (2.7 g/cm3) standing in for a hand tool dropped in the tank,
# lying on the floor straight under the right manway, long side along x.
TOOL_SIZE = (0.060, 0.024, 0.020)
TOOL_MASS = 0.08  # 28.8 cm3 of aluminium
TOOL_X, TOOL_Y = 0.360, -0.180

# The tool pad: a fixed white plastic mat in front of the right arm.
PAD_X, PAD_Y = 0.200, -0.180
PAD_SIZE = (0.100, 0.100, 0.003)
PAD_TOP = TOP + PAD_SIZE[2]

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap, lowest 38 mm (6.4 mm at the tips, 10 mm at 38 mm up)
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there (27 mm at the tips, 40 mm higher)
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Rule thresholds (scoring.py).
ENTRY_GROW = 0.005  # 'entered': pad-tip point below the deck's top, over the tank footprint grown by this
MONITOR_TIP_H = WALL_TOP_H  # 'monitored': the probe's tip in the left manway's column, inside the tank: no higher
#                              than the deck's underside (0.058) ...
HELD_DIST = 0.020  # ... and the probe held: a TCP this close to its grip point ...
HELD_OPENING = 0.5  # ... with that gripper less open than this (fingers under 47 mm apart)
DWELL_S = 2.0  # a test at one depth: the tip in the band, in the column and slow, this long without a break
DWELL_SPEED = 0.005  # m/s: 'slow' for the tip, measured as ...
DWELL_WINDOW_S = 0.5  # ... the path the tip moved over the last 0.5 s, divided by 0.5 s (not the simulator's
#                        instantaneous velocity, which chatters at 10-45 mm/s for a gripped probe that is not moving)
FALL_SPEED = 0.25  # m/s downward: a loose object falling (a free drop of more than 3 mm)
TIP_DEG = 30.0  # tilted more than this: tipped over
OFF_TABLE_DROP = 0.010  # an object whose lowest point is this far below the table top is off the table
COVER_CENTRE_TOL = 0.010  # 'back on': centre this close to the manway's centre, horizontally ...
COVER_FLAT_DEG = 5.0  # ... tilted less than this ...
COVER_Z_TOL = 0.003  # ... lowest point within this of the deck's top, and the whole opening under it
TOOL_OUT_RISE = 0.005  # 'out of the tank': the tool's lowest point this far above the deck's top, centre outside
TOOL_FLAT_DEG = 10.0  # 'on the pad': lying on a 60 x 24 mm face within this, ...
TOOL_Z_TOL = 0.003  # ... its lowest point within this of the pad's top, centre over the pad
PROBE_UPRIGHT_DEG = 10.0  # 'in its holder': upright within this, ...
PROBE_BOTTOM_TOL = 0.005  # ... its lowest point within this of the table top, ...
PROBE_HOLDER_TOL = 0.008  # ... its tip within this of the socket's centre (the socket allows 4 mm)
GRIPPER_OPEN = 0.9  # rigstate opening (fingers 84.5 mm apart) for 'open' at the end
CLEAR_GAP = 0.030  # ... and each gripper's finger zone at least this far from every loose object
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
STEEL = (0.50, 0.52, 0.55)
COVER_GREY = (0.20, 0.20, 0.22)
STAND_GREY = (0.62, 0.62, 0.62)
PROBE_BLUE = (0.15, 0.35, 0.80)
HOLDER_BLACK = (0.12, 0.12, 0.12)
WHITE = (0.97, 0.97, 0.97)
BAND_COLOURS = {"top": (0.85, 0.12, 0.10), "middle": (0.95, 0.80, 0.10), "bottom": (0.15, 0.65, 0.25)}
ORANGE = (0.95, 0.45, 0.10)
