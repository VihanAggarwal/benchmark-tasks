"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.

A bench model (about 1:8) of a strong-post W-beam guardrail run along the far edge of a road: three posts in a
row along y, a strip of road in front of them (toward the arms) with its edge line and three white arrows
showing that traffic runs toward +y (from the robot's right to its left), and two loose rail panels.
"""

import math

TOP = 0.7475  # station table top

# Posts: fixed boxes (timber post and blockout as one block), their road-side faces in one line at x 0.310.
POST_SIZE = (0.025, 0.025, 0.060)
POST_FACE_X = 0.310  # the road-side (front) faces
POST_X = POST_FACE_X + POST_SIZE[0] / 2  # 0.3225: centres
POST_SPACING = 0.240
POSTS = {"post_left": POST_SPACING, "post_middle": 0.0, "post_right": -POST_SPACING}  # name -> centre y
POST_HALF_W = POST_SIZE[1] / 2  # 12.5 mm: the middle post covers y -0.0125 .. +0.0125

# Rail panels: loose pine battens standing on an 18 x 300 mm edge. Local axes: x through the two broad faces (the
# 18 mm thickness), y along the 300 mm length, z up the 45 mm height.
PANEL_SIZE = (0.018, 0.300, 0.045)
PANEL_HALF = tuple(s / 2 for s in PANEL_SIZE)
PANEL_MASS = 0.12
PANEL_START = {"panel_left": (0.310, 0.400), "panel_right": (0.310, -0.400)}  # lengths along x at the start
PANEL_START_YAW = math.pi / 2  # local y (the length) along world x: each spans x 0.160 .. 0.460

# The road: a dark board in front of the posts, its edge line and three traffic arrows pointing +y.
ROAD_X0, ROAD_X1 = 0.120, 0.250
ROAD_HALF_Y = 0.350
ROAD_T = 0.002
MARK_T = 0.001  # paint-thick markings on the road (their tops 3 mm above the table)
LINE_X, LINE_W = 0.244, 0.006  # the edge line along y, x 0.241 .. 0.247
ARROW_X = 0.185
ARROW_YS = (-0.100, 0.000, 0.100)  # arrow centres; each points +y, its tip 30 mm past its centre
ARROW_W = 0.010  # stroke width (about 2 px in the 320 x 176 top camera image, 1.7-2 px per cm)
ARROW_TIP = 0.030  # the tip's y beyond the arrow's centre: each arrow is 60 mm long
HEAD_LEN = 0.025  # each of the two head strokes
# The two head strokes (yaw -45 and +45 deg) touch only at the tip, so the markings do not overlap: each stroke's
# centre is (12.4, 5.3) mm from the tip, the shaft runs from 30 mm behind the centre to 5 mm short of the tip,
# where its corners meet the strokes' inner edges. The head spans x 0.160 .. 0.210, on the road.
_HU, _HN = HEAD_LEN / 2, ARROW_W / 2
HEAD_DX = (_HU + _HN) / math.sqrt(2.0)  # 12.37 mm
HEAD_DY = (_HU - _HN) / math.sqrt(2.0)  # 5.30 mm
SHAFT_Y0 = -ARROW_TIP  # from the arrow's centre
SHAFT_Y1 = ARROW_TIP - ARROW_W / 2  # 25 mm

# Rule thresholds (scoring.py).
UPRIGHT_MAX_DEG = 8.0  # 'standing': the 45 mm axis within this of vertical (either edge down)
ON_TABLE = 0.003  # ... its lowest point within this of the table top
STILL_WINDOW_S = 0.5  # ... and still: over the last 0.5 s its centre stayed within STILL_MOVE of where it is now
STILL_MOVE = 0.002
STILL_TURN_DEG = 2.0  # ... and it turned less than this
D_LINES_Y = (POST_SPACING, 0.0)  # the downstream place: the back edge measured at the left and middle posts
D_GAP_MAX = 0.008  # ... within 8 mm in front of the posts' faces
D_PAST = 0.003  # ... or up to 3 mm past them (contact: at the posts' own y a panel cannot really be behind their
#                     faces; the offline kinematic check, which has no contacts, carries the pressed panel 2 mm in)
D_END_Y = (-0.055, 0.0)  # its right end (smallest y of its outline) between these: it reaches the middle post's
#                         centre (the reading at y 0 needs that anyway) and runs at most 55 mm past it
SAMPLE_STEP = 0.002  # the shared stretch is read every 2 mm ...
SAMPLE_INSET = 0.005  # ... from 5 mm inside each end of it (a stretch under 10 mm: at its middle only), so a turned
#                       panel's end face is never read as its long face
LAP_GAP_MIN, LAP_GAP_MAX = -0.001, 0.015  # the lap gap at every reading (-1 mm: contact)
WRONG_REACH = 0.002  # a wrong lap: U's left end at most this short of D's right end or of the middle post's right
#                      side, whichever is farther right (when D's end stops short of that side, U's end can tuck in
#                      behind D's line only as far as the post) ...
MIDDLE_POST_RIGHT_Y = -POST_HALF_W  # -0.0125
WRONG_LOW = 0.005  # ... U's lowest point within this of the table (standing or not) ...
# ... and U's front edge from 1 mm in front of to 15 mm behind D's back edge (LAP_GAP_MIN .. LAP_GAP_MAX) at every
# reading of the shared stretch when it is at least 2 x SAMPLE_INSET long; for a shorter stretch, or none, at one
# pair of readings: U's front edge SAMPLE_INSET inside its left end against D's back edge SAMPLE_INSET inside its
# right end.
LAP_OVERLAP = (0.010, 0.100)  # lapped on the road side (0.5)
LAP_YAW_MAX_DEG = 15.0
FULL_OVERLAP = (0.030, 0.060)  # 1.0
FULL_MID = 0.015  # the stretch's middle within this of y 0 (centred on the middle post)
RIGHT_LINE_Y = -POST_SPACING  # U's back edge read at the right post
RIGHT_GAP = (-0.001, 0.012)
# Released: each gripper's finger zone at least RELEASE_CLEAR from both panels' boxes. rigstate gives no gripper
# heading, so the fingers are taken to be anywhere round the gripper's axis: the zone is a solid cylinder round the
# axis, from the point between the finger tips FINGER_LEN back up the gripper, out to a finger's outer corner
# (half the finger gap plus its thickness, and half its width across): 60.4 mm wide open, 27.6 mm closed on a
# panel's 18 mm.
FULL_GAP = 0.0939  # finger gap fully open (rigstate opening 1.0)
FINGER_THICK = 0.010  # a finger, outside the gap
FINGER_HALF_WIDTH = 0.020  # half a finger's 40 mm width
FINGER_LEN = 0.038  # from the tips up the gripper (13 mm past the TCP to 25 mm behind it)
RELEASE_CLEAR = 0.010  # 10 mm
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a panel moving this far in one control step was teleported ...
START_TOL = 0.045  # ... and a new episode began if every panel is then on its start (covers a seeded reset's jitter)
EPS = 1e-9  # float slack on the edges of every window

# Colours (sRGB).
POST_BROWN = (0.33, 0.22, 0.13)
GALV_GREY = (0.72, 0.74, 0.76)
ROAD_GREY = (0.22, 0.22, 0.23)
WHITE = (0.95, 0.95, 0.95)
