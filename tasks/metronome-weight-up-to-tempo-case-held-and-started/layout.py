"""Every number of the scene, in metres, kg, N, s and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up; the table top is at z TOP. The metronome's
base frame M has its origin on the table under the middle of the case's footprint; the case is drawn with its front
(the face with the pendulum and the tempo scale) toward base -x, and turned M_YAW (+90 deg) about the vertical, so
the front looks toward world -y (the right arm), its back toward the left arm, and the pendulum swings along world x.
Base +x is world +y, base +y is world -x.

The metronome is a Wittner Taktell Piccolo (series 830): a plastic pyramid-shaped clockwork metronome, about
155 x 70 x 45 mm and 171 g (retailer listings). Its inner layout (pivot height, counterweight, the sliding weight,
the scale's spacing) is not published: those are design values (est in task.json), chosen so that the pendulum's own
swing, from the parts' masses and shapes, beats 40 to 208 per minute over the printed Maelzel scale, as the real one
does. The tempo is not a rule: the scorer times the pendulum's swings.
"""

from __future__ import annotations

TOP = 0.7475  # the station table top (world z)
G = 9.81
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
TABLE_X = (-0.19, 0.91)
TABLE_Y = (-0.65, 0.65)

# -- the case (base body, loose) --------------------------------------------------------------------------------------
M_X, M_Y = 0.255, 0.000  # the base frame's origin on the table (world x, y)
M_YAW = 1.5707963267948966  # the case turned 90 deg: its front (base -x) looks toward world -y, the right arm
M_ORIGIN = (M_X, M_Y, TOP)
TOTAL_MASS = 0.171  # the whole metronome, as sold
# Three stacked tiers draw the pyramid (base frame; x depth, y width, z heights).
TIERS = (  # (x0, x1, half width y, z0, z1)
    (-0.0225, 0.0225, 0.035, 0.000, 0.060),
    (-0.0195, 0.0195, 0.029, 0.060, 0.110),
    (-0.0155, 0.0155, 0.022, 0.110, 0.155),
)
CASE_H = 0.155
FRONT_X = TIERS[0][0]  # the lower front face (x)
CASE_FRICTION = (0.5, 0.45)  # ABS feet on a laminate table top, est
BLACK = (0.07, 0.07, 0.08)
WHITE = (0.93, 0.93, 0.90)
INK = (0.08, 0.08, 0.10)
STEEL = (0.80, 0.80, 0.83)
BRASS = (0.83, 0.68, 0.30)
RED = (0.80, 0.08, 0.08)

# -- the pendulum (a link on a hinge about x: it swings in the front's plane) --------------------------------------
PIVOT_Z = 0.055  # the pivot's height over the table
ROD_X = FRONT_X - 0.007  # the rod's centre line: 7 mm in front of the lower front face
ROD_W, ROD_D = 0.003, 0.0015  # across (y) and front to back (x)
ROD_Z0, ROD_Z1 = -0.012, 0.095  # from the pivot: the rod runs from 12 mm below it to its tip 95 mm above it
ROD_TIP_Z = PIVOT_Z + ROD_Z1  # 150 mm over the table, 5 mm under the case's top
CW_R, CW_H, CW_D = 0.011, 0.012, 0.016  # the counterweight inside the case: a disc 22 mm across, 12 mm thick, 16 mm under the pivot
ROD_LINK_MASS = 0.044  # rod and counterweight (spread over their volume, as PhysX does from a mass alone)
SWING_LIMIT = 0.90  # rad either way (the case's slot)
# -- the sliding weight (a link on a slider along the rod) ---------------------------------------------------------
W_SIZE = (0.020, 0.016, 0.014)  # x (it stands 16 mm proud of the rod's front, 4 mm behind it), y across, z tall
W_X0 = -0.016  # its front face from the rod's centre line (x); its back face 4 mm behind the rod
W_MASS = 0.005
W_CLIP = 3.0  # N: the weight's spring clip holds it on the rod until pushed or pulled along it this hard, est
# The clip in the sim (clip.py, a stick-slip spring written every control step; PhysX joint friction let the weight
# creep with no force on it, GPU 19 Nov): a 4000 N/m spring toward a target that stays put while the stretch is under
# 3 N (0.75 mm) and otherwise follows the weight at 3 N, led by its last step's travel (at most 0.5 mm, under the
# 0.75 mm band, so the spring never pushes a sliding weight on). Checked on a 1-D model of the slider (tests): a hand
# dragging it through an arm of 1 to 3.5 mm/N pulls 3.1 to 3.2 N on average (3 N, its 5 g and the damping).
W_CLIP_K = 4000.0  # N/m
W_CLIP_DAMPING = 6.0  # N s/m: about 0.7 of critical for the 5 g weight on that spring
W_CLIP_LEAD = 1.0
W_CLIP_LEAD_MAX = 0.0005  # m
# The scripted demo's trim for the drag (open loop, nothing read from the sim): how far the weight ends under the
# right hand's commanded height. Measured on the GPU (19 Nov, second run, the clip rule and the left's press in
# place), the hand commanded 31.9 mm up (22.7 + 9.2) left the weight 17.7 mm up (61.4 mm over the pivot, 5.0 mm
# short of 66.4), and it then held there to the end: (1) the arm stalled 8.7 mm under its command against the clip's
# 3 N (commanded 878.1, actual 869.4 to 869.6 mm world z); (2) the case rose 1.3 mm and tilted 3 deg about the
# left's pinch, which lifts the rod's front about 2.7 mm, and the pads slipped about 2 mm on the weight (together
# 5.0 mm between the hand and the weight at the stall, 0.6 mm of it the grip settling low); (3) let go, the weight
# settled back 0.56 mm onto the clip's target. The weight follows the hand 1:1 once it slides (steps 271 to 301:
# hand +10.3, weight +10.0 mm), so the shortfall adds: 9.2 + 5.0 = 14.2 mm. The left's press also slid the case
# 11.5 mm toward the right arm while the drag unloaded the table (task.py: CASE_SHIFT); it does not change the trim
# (the slide stopped once the weight slid).
DRAG_GIVE = 0.0142
BASE_MASS = TOTAL_MASS - ROD_LINK_MASS - W_MASS
# The weight's height (its centre above the pivot) for the tempo marks. START: 160 beats per minute, where it was
# left; GOAL: 100. Both from pendulum.r_of_bpm (the physics), rounded to 0.1 mm.
START_BPM = 160
GOAL_BPM = 100
W_R_MIN, W_R_MAX = 0.022, 0.088  # its travel (centre above the pivot)
MAELZEL = (40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 66, 69, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
           116, 120, 126, 132, 138, 144, 152, 160, 168, 176, 184, 192, 200, 208)
SCALE_LABELLED = (40, 60, 80, 100, 120, 144, 160, 208)  # the long marks
RUN_AMP = 0.35  # rad (20 deg): the swing the scale is drawn for, a running metronome's usual swing

# -- scoring -----------------------------------------------------------------------------------------------------------
TEMPO_TOL = 4.0  # beats per minute either side of 100 for full marks (one Maelzel mark: 96 and 104 are the neighbours)
TEMPO_NEAR = 10.0  # 90 to 110: near
TICKS = 8  # the tempo is timed over the last 8 ticks (7 intervals, about 4.2 s at 100)
TICK_GAP_MAX = 1.6  # s: a gap this long between ticks means it stopped (the slowest mark, 40, ticks every 1.5 s)
AMP_MIN = 0.26  # rad (15 deg): each swing between two timed ticks reaches at least this far
SET_TOL_MM = 2.5  # the weight's set height within this of the 100 mark, for the not-started tier
STAND_LIFT = 0.003  # standing: the case's underside within 3 mm of the table top
STAND_TILT = 0.175  # and tilted under 10 deg
HANDS_OFF = 0.030  # both finger zones at least 30 mm from every part of the metronome, while the timed ticks happen
