"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The window frame's bottom: a fixed aluminium sill, long side along y, centred on the midline.
FRAME_X, FRAME_Y = 0.30, 0.0
SILL_SIZE = (0.060, 0.260, 0.010)  # along x, along y, tall
SILL_TOP = TOP + SILL_SIZE[2]

# Two glazing stops on the sill: walls along y leaving a 40 mm channel centred on x 0.30.
CHANNEL_HALF_W = 0.020  # the channel is 40 mm wide (x 0.28 to 0.32)
STOP_SIZE = (0.008, SILL_SIZE[1], 0.040)  # 8 mm thick, the sill's 260 mm long, 40 mm tall
STOP_XS = (FRAME_X - CHANNEL_HALF_W - STOP_SIZE[0] / 2, FRAME_X + CHANNEL_HALF_W + STOP_SIZE[0] / 2)  # 0.276, 0.324
STOP_TOP = SILL_TOP + STOP_SIZE[2]  # TOP + 0.050

# Two jambs (the frame's sides, cut short) close the channel's ends: the channel is 240 mm long.
CHANNEL_HALF_L = 0.120  # y -0.12 to +0.12
JAMB_SIZE = (2 * CHANNEL_HALF_W, 0.010, 0.060)  # 40 mm along x (fills the channel), 10 mm along y, 60 mm tall
JAMB_YS = (FRAME_Y - CHANNEL_HALF_L - JAMB_SIZE[1] / 2, FRAME_Y + CHANNEL_HALF_L + JAMB_SIZE[1] / 2)  # -/+0.125
JAMB_TOP = SILL_TOP + JAMB_SIZE[2]  # TOP + 0.070

# Two white quarter-point marks on the sill's top in the channel (a 200 mm unit's quarter points when centred).
MARK_SIZE = (0.024, 0.030, 0.001)
MARK_YS = (FRAME_Y - 0.050, FRAME_Y + 0.050)  # right mark, left mark
MARK_TOP = SILL_TOP + MARK_SIZE[2]

# Two loose setting blocks (neoprene), on a fixed tray to the right of the frame, side by side along x.
BLOCK_SIZE = (0.024, 0.030, 0.008)  # 24 mm along x (across the channel), 30 mm along y, 8 mm tall
BLOCK_MASS = 0.007  # neoprene, 1.23 g/cm3: 5.8 cm3 is 7 g
TRAY_SIZE = (0.090, 0.090, 0.004)
TRAY_X, TRAY_Y = 0.20, -0.24
TRAY_TOP = TOP + TRAY_SIZE[2]
BLOCK_STARTS = ((TRAY_X - 0.020, TRAY_Y), (TRAY_X + 0.020, TRAY_Y))  # 40 mm apart, centre to centre
SET_BLOCK_TOP = MARK_TOP + BLOCK_SIZE[2]  # a block lying on its mark: its top is TOP + 0.019

# The glass unit: a loose small double-glazed unit standing on its long edge on the table, left of the frame,
# its faces normal to x. Two 2 mm glass panes on a 20 mm aluminium spacer: 200 g.
UNIT_SIZE = (0.024, 0.200, 0.080)  # thickness (along x), length (along y), height
UNIT_MASS = 0.20
UNIT_X, UNIT_Y = 0.22, 0.29  # its nearer end at y +0.19, about 2 cm past the right arm's top-down reach

# Rule thresholds (scoring.py).
SILL_TOUCH_Z = MARK_TOP + 0.001  # 'touched the sill': a corner of the unit inside the channel below this height
CHANNEL_SLACK = 0.001  # 'in the channel': every corner of the unit within the channel's inner outline + this
TOUCH_SLACK = 0.002  # a corner within the channel's outline + this counts for 'touched the sill'
STAND_TILT_DEG = 3.0  # 'standing': the unit's height axis within this of vertical
DOWN_TILT_DEG = 30.0  # more than this at the end: lying down (or stood on end), 0
ON_BLOCK_GAP = 0.002  # 'resting on a block': the unit's underside within this of the block's top, above or below
ON_BLOCK_SHARE = 0.5  # ... with at least this share of the block's top under the unit
UNIT_CENTRE_TIGHT = 0.008  # the unit's centre along y from the channel's centre: 1.0 (12 mm or more to each jamb)
UNIT_CENTRE_LOOSE = 0.015  # 0.75
BLOCK_TIGHT = 0.008  # a block's centre from its mark's centre: 'set' (1.0, 0.25)
BLOCK_LOOSE = 0.020  # 0.75
BLOCK_YAW_DEG = 15.0  # a block's sides within this of the channel's
BLOCK_FLAT_DEG = 10.0  # a block lies flat: its 8 mm axis within this of vertical (either broad face down)
BLOCK_LOW_BAND = (SILL_TOP - 0.001, SILL_TOP + 0.003)  # a block 'on the sill': its lowest point in this band
OFF_TABLE_DROP = 0.010  # an object's lowest point this far below the table top: it fell off the table
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object landing on its starting spot having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting spot: within this of it (covers a seeded reset's +-3 cm jitter)

# Colours (sRGB).
ALUMINIUM = (0.60, 0.62, 0.65)
WHITE = (0.95, 0.95, 0.95)
RUBBER = (0.08, 0.08, 0.08)
TRAY_GREY = (0.30, 0.30, 0.32)
GLASS = (0.75, 0.85, 0.90)
