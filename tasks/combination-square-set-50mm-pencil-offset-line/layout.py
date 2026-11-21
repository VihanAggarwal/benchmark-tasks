"""Every number of the scene, in metres, kg, N and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP. "Table frame" in
the comments is the world with z measured from the table top.

The board is a 1-by-6 lying flat on the table along y, its near long edge (toward the arms) at x NEAR_X.
The combination square's frame has its origin at the middle of the head's 90-degree face, on the table: local +x runs
from the face out along the blade, over the board (the blade's end is SQ_EXT from the face), local +y along the face
(parallel to the board's edge when seated), +z up.
The pencil's frame has its origin at the centre of the lead's tip, +z up its axis; local +x is across its wide face.
The pencil block's frame has its origin on the table at the middle of the slot.

Figures marked est in task.json are design values the simulation is built to; the setup's checks say how real parts
are matched to them.
"""

from __future__ import annotations

import math

TOP = 0.7475  # the station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
TABLE_X = (-0.19, 0.91)  # the station table (yamkin.TABLE_BOX)
TABLE_Y = (-0.65, 0.65)

# -- the board: a nominal 1-by-6, 3/4 x 5-1/2 in actual, cut to 610 mm (2 ft), lying flat ---------------------------------
BOARD_W = 0.1397  # across (x): 5-1/2 in
BOARD_L = 0.610  # along (y)
BOARD_H = 0.01905  # 3/4 in
BOARD_X = 0.270
BOARD_Y = 0.0
NEAR_X = BOARD_X - BOARD_W / 2  # 0.20015: the edge the square's head goes against
FAR_X = BOARD_X + BOARD_W / 2
BOARD_MASS = 0.73  # est: pine at about 450 kg/m3
BOARD_FRICTION = (0.5, 0.45)  # dry pine on the table, est
PINE = (0.87, 0.74, 0.52)
BOARD_ORIGIN = (BOARD_X, BOARD_Y, TOP)  # the board's frame: the centre of its underside

# -- the job: a line 50 mm in from the near edge, parallel to it, at least 90 mm long ---------------------------------------
LINE_IN = 0.050  # offset of the line's centre from the near edge
LINE_MIN = 0.090  # the line spans at least 90 mm along the edge

# -- the combination square: STANLEY 12 in (300 mm class) combination square, set to a 50 mm gauge ------------------------
# Blade 12 in (304.8 mm) nominal, 1 in (25.4 mm) wide, about 1.6 mm thick, with a head (stock) that slides on it and
# locks with a knurled screw. Head, blade and lock are drawn as ONE body (the screw is locked). The head's 90-degree
# face is the plane local x 0; the blade lies flat on the board's top face, its underside at the head-frame height
# SQ_BLADE_Z (just above the board's 19.05 mm face), and sticks out SQ_EXT from the face. The rest of the blade (the tail)
# runs back out of the head toward the robots.
SQ_BLADE_L = 0.3048
SQ_BLADE_W = 0.0254
SQ_BLADE_T = 0.0016
SQ_EXT = 0.049  # set so that the lead (2 mm thick) lands its centre 50 mm in: 49 mm + 1 mm
SQ_BLADE_Z = 0.01920  # the blade's underside above the table, 0.15 mm above the board's face (est: play in the slot)
SQ_HEAD_DEPTH = 0.062  # along the blade (x), est
SQ_HEAD_LEN = 0.112  # along the face (y): 4-3/8 in, est
SQ_HEAD_H = 0.033  # tall (z), est; stands on the table beside the board
SQ_KNOB_R = 0.009
SQ_KNOB_H = 0.012
SQ_KNOB_X = -0.034
SQ_MASS = 0.35  # est: cast head about 0.25 kg, steel blade 0.10 kg
SQ_FRICTION = (0.35, 0.30)  # painted zinc on dry pine and on the table, est (incline check in setup)
HEAD_COLOR = (0.16, 0.18, 0.22)
STEEL = (0.78, 0.80, 0.83)
INK_BLACK = (0.05, 0.05, 0.06)
KNOB_COLOR = (0.70, 0.52, 0.10)
SQ_GRAD = 0.005  # graduation marks every 5 mm along the blade (long every 10 mm)
SQ_GAP = 0.030  # the square starts set down 30 mm off the edge (face at x NEAR_X - 30 mm)
SQ_START = (NEAR_X - SQ_GAP, -0.113, TOP)  # the head face's midpoint on the table
SEAT_GAP = 0.0003  # the head's face stops 0.3 mm off the edge when pushed home (contact offset)
SQ_BLADE_END = SQ_EXT  # local x of the blade's end face

# -- the pencil: a flat carpenter's pencil, 7 x 18 mm section, sharpened down to 140 mm (5-1/2 in), chisel point --------------------------
# Drawn: body 18 mm across x, 7 mm across y, 140 mm long; the lead's stub 2 mm x 7 mm at the tip, the point
# stepping out to the full width over 8 mm on its +x side only, so its -x face is one flat plane (local x -1 mm).
PN_X0 = -0.001  # the -x face
PN_W = 0.018
PN_T = 0.007
PN_LEN = 0.140
PN_STEP = 0.002
PN_STUB = 0.002  # the lead's thickness across the line (x)
PN_TIP_X = 0.0  # the lead's centre: local x 0
PN_CENTER_X = PN_X0 + PN_W / 2  # 0.008: the body's centre (where the gripper closes)
PN_MASS = 0.014  # est
PN_FRICTION = (0.8, 0.7)  # the lead on pine and the body in the pads, est
PN_RED = (0.78, 0.12, 0.10)
PN_LEAD = (0.12, 0.12, 0.14)

# The pencil block: where the pencil is kept, tip down. A 50 x 36 x 45 mm hardwood block fixed to the table, with a slot
# 21 x 10 mm across and 42 mm deep to a 3 mm floor, so the pencil stands in it with about 1.5 mm to spare each side.
BL_X, BL_Y = 0.400, -0.160
BL_HALF_X, BL_HALF_Y = 0.025, 0.018
SLOT_HX, SLOT_HY = 0.0105, 0.0050
BL_FLOOR = 0.003
BL_DEPTH = 0.042
BL_H = BL_FLOOR + BL_DEPTH  # 45 mm
BLOCK_WOOD = (0.55, 0.36, 0.20)
BLOCK_ORIGIN = (BL_X, BL_Y, TOP)
PN_START = (BL_X - PN_CENTER_X, BL_Y, TOP + BL_FLOOR)  # the lead's tip on the slot's floor

# -- scoring thresholds ----------------------------------------------------------------------------------------------
INK_Z = 0.0003  # an ink point: the lead's tip on the board's top face (0.3 mm for contact offset) ...
MIN_FORCE = 0.3  # ... AND a PhysX contact force of 0.3 N or more between pencil and board (sim; contact.py)
INK_TILT_DEG = 20.0  # and the pencil within 20 degrees of upright
SEAT_TOL = 0.001  # seated: the head's face within 1.0 mm of the edge ...
SEAT_YAW_DEG = 1.5  # ... turned within 1.5 degrees of it ...
SEAT_TILT_DEG = 3.0  # ... and lying within 3 degrees of flat
SEAT_HOLD_S = 0.5  # seated for 0.5 s counts as the first stage
GUIDE_GAP_LO = -0.0012  # guided: the pencil's -x face on the blade's end: gap in [-1.2, +0.8] mm ...
GUIDE_GAP_HI = 0.0008
GUIDE_Y = 0.0090  # ... and the pencil's middle within 9 mm of the blade's middle (3.5 mm of the 12.7 mm half-width spare)
GUIDE_BIN = 0.002  # guidance is judged per 2 mm stretch of the line
GUIDED_FRAC = 0.95  # at least 95 percent of those stretches inked, and every ink point in them guided
SPAN_GAP = 0.006  # no gap over 6 mm along the line
PARALLEL_DEG = 1.0  # the line within 1 degree of parallel to the edge (2 mm over 115 mm)
STRAIGHT = 0.0015  # every ink point within 1.5 mm of the line fitted through them
OFFSET_TOL = 0.002  # the fitted line 50 mm from the edge, within 2 mm, at both ends
STOW_SEAT = 0.005  # stowed (back in its block): the lead's tip within 5 mm of the slot's floor ...
STOW_RADIAL = 0.003  # ... within 3 mm of its axis ...
STOW_TILT_DEG = 10.0  # ... and the pencil within 10 degrees of upright
LET_GO = 0.030  # hands off: finger zones 30 mm or more from the square, the pencil and its block
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005
BOARD_DRAG = 0.020  # latch: the board moved 20 mm
BOARD_YAW_DEG = 3.0  # latch: the board turned 3 degrees
FALL_DROP = 0.003  # latch: any part 3 mm below the table top
EPS = 1e-9
RESET_JUMP = 0.020
START_TOL = 0.003

# -- the demo (table frame unless noted) -----------------------------------------------------------------------------
YAW_R = math.pi / 2  # the right grip closes along world x, on the pencil's wide faces
PN_GRIP_Z = 0.105  # the right TCP on the body, 105 mm over the lead's tip (the top 35 mm above it)
PN_PINCH = -(PN_W - 0.008) / 2  # finger target: 4 mm per side under the body's width
# The left hand GRIPS the blade's tail by its two edges (25.4 mm across), 43 mm behind the head, fingers closing along
# world y. The first GPU run (21 Nov) pressed the head's top with closed fingers and pushed: the fingertips slid over
# the head and the square never moved (face gap 30.0 mm all run), since friction on top can never beat the friction
# under a 0.35 kg part pressed down by the same hand. (A grip on the locking screw's knob was tried offline: the left
# wrist then stands within 89 mm of the pencil hand and the arm capsules overlap by 37 mm.)
GRIP_LOCAL = (-0.105, 0.0)  # the grip's middle on the blade's tail, square frame (the head's back is at x -0.062)
YAW_GRIP = 0.0  # the left fingers close along world y, on the blade's two long edges
BLADE_PINCH = -(SQ_BLADE_W - 0.008) / 2  # finger target: about 4 mm per side under the blade's 25.4 mm
GRIP_TIPS_Z = 0.011  # the pad tips 11 mm over the table, 8 mm under the tail's underside (pads 38 mm tall)
PRESS_X_BIAS = 0.0045  # commanded past the seat toward +x. GPU run 2 (bias 2.5 mm): the left TCP stalled 6.4 mm short of
#                        its x command against the square's static friction (face 4.2 mm off the edge) and seated only
#                        once the stroke's y motion freed it; on the edge it then stood 2.4 mm short of its command
SEAT_WIGGLE = 0.004  # after the push the left hand moves the square 4 mm along +y and back, twice, before the pencil
#                      comes down: sliding friction lets the arm close its x error and the face goes onto the edge
DRAW_SPEED = 0.0034  # both hands together along +y. GPU run 2 at 25 mm/s read 0 N on about 25 percent of the inked
#                      steps, GPU run 3 at 10 mm/s on about 43 percent (130 ink points of about 230 lead-down steps,
#                      2.2 N when read; the sampled steps show isolated zeros, not long runs). Run 3's single 11 s
#                      waypoint was smoothstepped, so mid-stroke the hands ran at 15 mm/s (2.7 steps a 2 mm stretch) and
#                      11 percent of the stretches stayed empty (guided_frac 0.889). Now 3.4 mm/s, cut into DRAW_SEG
#                      waypoints so the speed is even: every 2 mm stretch gets 0.59 s (about 12 steps)
DRAW_SEG = 0.001  # the stroke's waypoint length: any 2 mm of the line then takes the same time
PEN_Y_OFF = -0.002  # the pencil rides 2 mm to -y of the blade's middle: GPU run 3 had it 1.7 to 2.6 mm to -y while the
#                     square stood (the seat wiggle left the square 1.8 mm to +y) and 6.2 to 7.2 mm to +y all stroke
#                     (the left hand lags its command by about 7 mm pushing the square; the pencil hand by 2 mm), against
#                     GUIDE_Y 9 mm; 2 mm back centres both
RUN_A = SQ_START[1] + PEN_Y_OFF  # the stroke starts at this y (pencil) ...
RUN_B = RUN_A + 0.110  # ... and ends at this y: a 110 mm stroke, the line at least 90 mm
LEAD_DOWN = 0.0015  # the lead commanded 1.5 mm into the board's face (1 mm left the TCP only 0.55 mm down on the GPU)
PEN_PRESS = 0.0010  # the pencil's face commanded 1.0 mm into the blade's end (was 0.8; the GPU run put the pencil 0.9
#                     to 1.4 mm beyond its command in +x on the stroke; more than 1.2 mm would leave the rigid offline
#                     proxy, which lets the face pass into the blade, outside the guided band)
PEN_APP_GAP = 0.0015  # the pencil's face comes down this far beyond the blade's end ...
PEN_HOVER = 0.0010  # ... to the lead's tip 1 mm over the board (no contact, no ink; the GPU TCP stood 0.5 mm under its
#                     command there), goes in against the blade's end at that height (the blade spans 0.15 to 1.75 mm
#                     over the board), and only then down
PEN_SLIP_Y = -0.0014  # where the pencil's tip ends up along y over the block, from where the right TCP is commanded:
#   the TCP stops 1.0 mm short at the grasp (approach along +y) with the pencil shoved 0.2 mm on the close, so the
#   pencil is held 1.25 mm to -y of the TCP (runs 1, 2 and 3 alike); the TCP lands 0.4 to 0.7 mm to +y of its command
#   over the block (runs 1, 2, 3); the stroke's drag turns the pencil in the pads by a further 0.5 mm (run 3, seated,
#   10 mm/s), 0.8 mm (run 1) or 2.3 mm (run 2, the head seating during a 25 mm/s stroke); 0.8 assumed:
#   0.55 - 1.25 - 0.8 = -1.5 mm, taken as -1.4 since the slight lean puts the pencil's upper part to +y.
#   The tip must land within 1.5 mm of the slot's axis (a 7 mm pencil in a 10 mm slot). Run 4 (STOW_DY 1.4) hovered
#   it 0.4 mm off the axis in y, which is fine (see STOW_DX for why runs 1 to 4 all missed the slot: x, not y)
STOW_DY = -PEN_SLIP_Y  # so the pencil goes back into its block with the TCP 1.4 mm to +y
# Why the pencil missed its slot in all four GPU runs (traces of 21 Nov): over the block the tip hovered at x 0.3914 to
# 0.3925 (aim 0.392, the 18 mm body centred in the 21 mm slot: 1.5 mm spare each side), y -0.1582 to -0.1617. Then the
# 2 s, 53.5 mm "down into the block" (about 40 mm/s mid-move) and in every run the tip came to rest at x 0.3874 to
# 0.3878, z 0.7925: ON the -x wall's top (block top TOP + 45 mm), the right TCP 4.3 mm to -x of its command and 40 mm
# over it, whatever the y (-0.1603 in run 3, -0.1634 in run 1). The hand trails a fast vertical move in x: the lift out
# of the block (about 40 mm/s) ran it 1.9 mm to +x (run 4, step 153); a joint-lag estimate (yamkin Jacobian, lag =
# joint speed x kd/kp, the wrist kd/kp 0.15 s) gives 1.2 mm to -x per 10 mm/s down over the block (est, not measured).
# Going down at 40 mm/s puts the 1.0 to 1.5 mm margin on the -x side under the wall's top; the opening then let the
# pencil fall over onto the open fingers, and the hand (no longer held up) dropped on it and carried it off.
# Now: aim the tip STOW_DX to +x, stop STOW_OVER over the block's top and wait for the lag to die, cross the wall's
# top slowly (about 4 to 5 mm/s: under 0.6 mm of lag by the estimate), go on down inside the slot, settle, let go,
# wait, then lift straight up along the pencil before moving aside.
STOW_DX = 0.0007  # the tip aimed 0.7 mm to +x of the slot's axis: the hover's mean error (-0.2 mm) plus 0.3 to 0.6 mm
#                   of lag on the slow entry (est); the tip should then cross the top within about 0.8 mm of the axis
STOW_SETTLE_S = 1.0  # a pause over the block before going down: the carry swings the tip by about 0.3 mm
STOW_OVER = 0.003  # the tip stops 3 mm over the block's top ...
STOW_OVER_S = 1.5  # ... (7 mm down from the hover in 1.5 s) ...
STOW_WAIT_S = 0.5  # ... and waits there for the hand's lag to die out (joint time constants 0.06 to 0.15 s)
STOW_ENTRY = 0.015  # then slowly down until the tip is 12 mm into the slot (STOW_OVER + 12 mm) ...
STOW_ENTRY_S = 5.0  # ... 15 mm in 5 s: about 4.5 mm/s as the tip crosses the wall's top
STOW_DOWN_S = 4.0  # then on down inside the slot to the floor (27 mm), the walls now guiding the pencil
STOW_HOLD_S = 1.0  # tip on the floor, the pads still shut, before letting go
STOW_OPEN_WAIT_S = 1.0  # fingers open, nothing moves: the pencil stands free in the slot
STOW_LIFT = 0.045  # then straight up along the pencil, the pad tips to 3 mm under its top (55 mm pins the wrist's
#                   joint4 at its limit, check_task), and aside along -y: the open pads pass 16 mm either side of it
STOW_LIFT_S = 3.0
