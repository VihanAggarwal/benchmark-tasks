"""Every number of the scene, in metres, kg, N and rad. NumPy only.

Frames. World x forward from the arm bases, y to the robot's left, z up. Every part's base frame here is the "line"
frame: its origin on the table top under the forearm's axis, at y 0 (world (ARM_X, 0, TOP)), axes as the world's;
so a point (x, y, z) below is (ARM_X + x, y, TOP + z) in the world.

The parked cobot's forearm lies along y across the table in front of both arms. Its dress-pack hose lies on top of
it, out of a linear retraction unit near the forearm's +y (elbow) end, toward the wrist at its -y end. Pulling the
hose out means moving it toward -y. Figures are design values; task.json says which are estimates.
"""

from __future__ import annotations

import math

TOP = 0.7475
CONTROL_HZ = 20.0
WINDOW = 20  # 1 s of control steps

ARM_X = 0.250  # the forearm's axis, world x
ORIGIN = (ARM_X, 0.0, TOP)  # every part's base frame
ARM_Z, ARM_R = 0.040, 0.0315  # forearm axis height over the table, radius (a 63 mm tube)
ARM_Y = (-0.215, 0.300)
WRIST_Y = -0.215
GREY = (0.72, 0.74, 0.76)
DARK = (0.22, 0.24, 0.27)
BLUE = (0.18, 0.42, 0.72)
ORANGE = (0.93, 0.52, 0.14)
BLACK = (0.10, 0.10, 0.11)
YELLOW = (0.98, 0.84, 0.10)

# the hose: igus triflex R TRE.40 (TRE.40.058.0): 43 mm across, 13.9 mm link pitch; modelled as crests and waists
HOSE_R = 0.0215
GROOVE_R = 0.0175  # waist radius: 4 mm under the crests
PITCH = 0.0139  # one link
GROOVE_W = 0.007  # waist length along the hose
SADDLE = 0.008  # the saddles between the forearm and the hose
HOSE_Z = ARM_Z + ARM_R + SADDLE + HOSE_R  # 0.101
HOSE_TOP = HOSE_Z + HOSE_R
HOSE_MASS = 0.12
HOSE_FRICTION = (1.0, 0.9)

# the exit end of a linear retraction unit (igus triflex RSE(C) TR.RSE.40) on the forearm, its exit face at EXIT_Y
EXIT_Y = 0.160
UNIT_Y = (EXIT_Y, 0.270)
UNIT_HALF = 0.0335  # half its width and height round the hose axis: a 47 mm passage
UNIT_WALL = 0.010
PULL_K = 144.0  # N/m: about 4.0 N two links out. Estimate (igus give no force for the unit's elastic band)
PULL_DAMP = 3.0  # N s/m
PULL_MAX = 0.060  # the slide's end stop in the scene

TWO_LINKS = 2 * PITCH  # the maker's pretension: "pull the triflex forward two links"
MARK = TWO_LINKS  # 27.8 mm: the pull left once the protector rests on the exit

# the protector: igus TR.40.30 quick-release protector, 55 mm across and 27 mm along the hose, modelled as a square
# U shell (the lower part) with a flat hinged lid and one rib under the lid; loose, sitting open in a cradle on the
# forearm just in front of the exit, the hose passing through it 1 mm clear.
RING_OUT = 0.0275
RING_IN = 0.0225
RING_W = 0.027
RING_GAP = 0.004  # from its +y face to the exit face at the start
RING_Y = EXIT_Y - RING_GAP - RING_W / 2  # 0.1425, its centre (and the rib's)
RING_MASS = 0.030
RING_FRICTION = (0.3, 0.25)
RIB_W = 0.003
RIB_DEPTH = 0.0045  # 0.5 mm off the waist's floor when shut in a waist
LID_LEN = 2 * RING_OUT
LID_T = 0.005
HINGE_X = RING_OUT  # the lid's hinge along the +x top edge
HINGE_Z = HOSE_Z + RING_IN + LID_T / 2  # the hinge axis at the plate's mid-thickness when shut
LID_OPEN = math.radians(95.0)  # drawn open: stood up just past vertical, on its open stop
LID_SPRING = 0.08  # N m/rad toward open
LID_MASS = 0.012
LATCH_AT = math.radians(2.5)  # within this of shut, the quick-lock catches (the on_step rule)
LATCH_HOLD = 0.30  # rad past shut: the latched lid's drive target (held on its shut stop, 0.024 N m)
SHUT = math.radians(2.5)  # the scorer's "shut"
RIB_PLAY = (GROOVE_W - RIB_W) / 2  # 2 mm either way inside a waist
CRADLE_Y = (RING_Y - RING_W / 2 - 0.030, EXIT_Y)

# the waist that meets the rib when the hose is out CLOSE_AT; let go, the hose runs back RIB_PLAY + RING_GAP to MARK
CLOSE_AT = MARK + RIB_PLAY + RING_GAP  # 33.8 mm
WAIST0_Y = RING_Y + CLOSE_AT  # that waist's place at rest (inside the unit)
HOSE_Y0, HOSE_Y1 = 0.255, -0.120  # the hose's run (its +y end inside the unit, its free moving end)

# the score
NEAR = 0.007  # 0.25 / 0.5: within 7 mm of the mark (half a link)
CLOSE = 0.005  # 0.75 / 1.0: within 5 mm
RELEASE = 0.040  # hands off: every finger zone 4 cm or more from the hose and the protector
STILL = 0.002  # m/s along the slide

# the demo
GRIP_Y = -0.050  # the right hand's grip, on a crest (snapped in grip_y())
# m/N: the right arm's give along y per N of the band's pull, lumped (wrist and arm compliance, the hose's friction
# on its saddles, the hose creeping in the pinch). Fitted to two GPU demo runs (20 Nov): commanded 42.6 mm, the hose
# came out 26.1 mm and held 25.2 mm while the lid was pushed over; commanded 53.8 mm, 32.7 mm and held 31.3 mm. Both
# ratios are 1.64 to 1.72, so 0.0047 (1.68).
WRIST_C = 0.0047
# The demo does not try to stop the hose with a waist under the rib (a 4 mm window: the second GPU run held 31.3 mm,
# 0.5 mm outside it, and the lid stopped 8.6 deg open on a crest). It draws the hose past the waist, to a crest
# centred under the rib (PULL_START, 40.75 mm: 3.45 mm of margin either way), the lid is pressed down onto that
# crest, and then the hose is eased back toward the unit while the lid is still pressed: the rib drops into the waist
# as it passes and the quick-lock catches. Eased back to PULL_END the hose would be under the mark, so the waist's
# wall meets the rib and the protector meets the exit face first: they, not the hand, set the hose at 27.8 mm. Easing
# back, the hose's friction reverses and the arm gives less (on the GPU about 24 mm for this command).
PULL_START = CLOSE_AT + PITCH / 2  # 40.75 mm
PULL_END = 0.0215
PULL_CMD = PULL_START * (1.0 + WRIST_C * PULL_K)  # commanded for PULL_START reached (68.3 mm)
PULL_END_CMD = PULL_END * (1.0 + WRIST_C * PULL_K)  # commanded to ease it back (36.0 mm)
HOSE_GRIP_GAP = 2 * HOSE_R - 0.008
PAD_TIP = 0.013
PUSH_R = 0.034  # the left fingertips' contact on the lid, from the hinge
FINGER_HX = 0.0135  # the shut finger pair's half width along x (closing along y)
FINGER_LEN = 0.038


def waists(q: float):
    """y of every waist centre (base frame) with the hose pulled out ``q``."""
    n0 = math.floor((HOSE_Y1 - WAIST0_Y) / PITCH) - 1
    n1 = math.ceil((HOSE_Y0 - WAIST0_Y) / PITCH) + 1
    return [WAIST0_Y + n * PITCH - q for n in range(n0, n1 + 1) if HOSE_Y1 <= WAIST0_Y + n * PITCH <= HOSE_Y0]


def crests(q: float = 0.0):
    return [w + PITCH / 2 for w in waists(q)]


def aligned(q: float, ring_y: float = RING_Y) -> bool:
    """The rib (at ``ring_y``) fits a waist, within its play, with the hose out ``q``."""
    return any(abs(w - ring_y) <= RIB_PLAY + 1e-9 for w in waists(q))


def band_rest_y() -> float:
    """The yellow crest's centre at rest: the crest that sits just outside the protector's wrist-side face once the
    hose is set (out MARK, the protector on the exit)."""
    face = EXIT_Y - RING_W  # the protector's -y face once it rests on the exit
    return min((c for c in crests(MARK) if c < face - PITCH / 2 + GROOVE_W / 2), key=lambda c: abs(c - face)) + MARK


def grip_y() -> float:
    return min(crests(0.0), key=lambda c: abs(c - GRIP_Y))


def lid_dir(phi: float):
    """Unit vector (x, z) from the hinge along the plate, ``phi`` open from shut (0 shut)."""
    return (-math.cos(phi), math.sin(phi))


def lid_normal(phi: float):
    """The plate's outer (top when shut) face normal (x, z)."""
    return (math.sin(phi), math.cos(phi))


def finger_offset(phi: float) -> float:
    """How far (along the lid's outer normal) the shut finger pair's pad-tip point stands off the outer face when its
    box (x half FINGER_HX, z from the tips up FINGER_LEN) just touches the face."""
    s, c = math.sin(phi), math.cos(phi)
    return FINGER_HX * abs(s) - min(0.0, FINGER_LEN * c)


def push_tip(phi: float, r: float = PUSH_R, into: float = 0.0):
    """Base-frame (x, z) of the left pad-tip point touching the lid's outer face at ``r`` from the hinge with the lid
    ``phi`` open, ``into`` metres pressed past touching."""
    ux, uz = lid_dir(phi)
    nx, nz = lid_normal(phi)
    off = LID_T / 2 + finger_offset(phi) - into
    return (HINGE_X + r * ux + off * nx, HINGE_Z + r * uz + off * nz)
