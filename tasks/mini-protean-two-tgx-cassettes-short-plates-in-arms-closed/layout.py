"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A Bio-Rad Mini-PROTEAN Tetra electrode assembly (the clamping frame of the 2-gel cell, cat. 1658005) stands on the
bench, held in place (fixed), its two green gaskets facing +y and -y. Its core is a U: two legs at its x ends and a
bottom bar, faced with the green gasket; the space between the legs, above the bar, is the inner buffer chamber. A gel
cassette stands on the gel supports at the foot of each gasket, short plate inward, and is held there by a green
clamping arm that swings over its outer face and clicks shut. Two Bio-Rad Mini-PROTEAN TGX cassettes (10 x 8 cm,
4.6 mm thick) stand in a two-slot wet-storage rack on each side, turned a quarter turn from how they go in.

Figures are sourced (README) or estimates (est, said in task.json): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81
CONTROL_HZ = 20.0

# -- the TGX cassette (loose, two of them) ---------------------------------------------------------------------------
# Local frame: origin at the middle of its bottom edge; x along its 100 mm width, z up its 80 mm length, y through its
# thickness. The short plate (with the 1 mm gel behind it) is on the -y side, the long plate on the +y side.
CS_W = 0.100  # Bio-Rad: cassette 10 x 8 x 0.46 cm (W x L x thickness)
CS_H = 0.080
CS_T = 0.0046
CS_TS = 0.0026  # short plate 1.6 mm + gel 1.0 mm (est; the gel is 1.0 mm, Bio-Rad)
CS_TL = CS_T - CS_TS  # long plate 2.0 mm (est)
CS_HS = 0.070  # short plate 70 mm tall: 10 mm under the long plate's top (est)
CS_MASS = 0.030  # styrene copolymer plates and a 1 mm gel, from their volume (est)
CS_LONG_COLOR = (0.82, 0.86, 0.90)
CS_SHORT_COLOR = (0.62, 0.74, 0.88)
CS_BAND_COLOR = (0.15, 0.55, 0.30)  # the green printed band along the long plate's foot (est)
CS_FRICTION = (0.6, 0.5)  # wet styrene on silicone and plastic (est)

# -- the electrode assembly (fixed) -----------------------------------------------------------------------------------
FX = 0.260  # its centre (x); centred on y 0
GY = 0.018  # each gasket's face, 18 mm either side of the centre line (est)
GASKET_T = 0.002  # the green gasket's thickness on the core
LEG_X0, LEG_X1 = 0.040, 0.052  # each leg of the U, from the centre along x (the cassette's ends lie on them)
BOTTOM_Z = 0.018  # the U's bottom bar
L0 = 0.010  # the gel supports: their floor, on which a seated cassette's foot stands
NOTCH_Z = L0 + CS_HS + 0.002  # the notch at the top of each gasket leg: a seated short plate's top is 2 mm under it
GROOVE_W = 0.006  # each support is a groove 6 mm wide (gasket face to lip) and 6 mm deep
LIP_W = 0.003
LIP_TOP = L0 + 0.006
LEAN_DEG = 12.0  # a cassette let go in the groove leans out until its outer face rests on the lip's edge:
# T / cos(lean) + 6 mm * tan(lean) = 6 mm gives about 12 degrees (it tilts away from the centre, as the manual says)
CORE_COLOR = (0.92, 0.92, 0.90)
GASKET_GREEN = (0.20, 0.65, 0.30)
FRAME_FRICTION = (0.8, 0.7)  # silicone gasket (est)

# -- the green clamping arms (hinges on the frame, one per side) ---------------------------------------------------
# Each swings about a vertical pin at the frame's far end, outboard of its gasket, low on the frame. Joint angle q:
# 0 = open (pointing straight out, +y or -y), Q_CLOSED = shut (lying along -x over the foot of the cassette's outer
# face, its handle reaching 85 mm past the cassette's near end).
AW = 0.008  # bar thickness (across, y when shut)
AH = 0.012  # bar height
ARM_Z0, ARM_Z1 = 0.018, 0.030  # it presses the cassette's foot, 8 to 20 mm over the support floor
ARM_ZC = (ARM_Z0 + ARM_Z1) / 2
ARM_L = 0.195  # from the pin to the end of its handle (est)
PIV_X = FX + 0.064
PIV_Y = GY + CS_T + AW / 2  # its inner face, shut, lies on a seated cassette's outer face
HUB_R = 0.007
PILLAR_R = 0.006
ARM_MASS = 0.040  # (est)
ARM_GREEN = (0.15, 0.70, 0.35)
Q_CLOSED = math.pi / 2
Q_UPPER = Q_CLOSED + 0.06  # with nothing in its slot the arm swings on past shut toward the gasket
Q_LATCH = Q_CLOSED - math.radians(2.0)  # the clip catches 2 degrees short of shut
Q_CAM0 = Q_CLOSED - math.radians(12.0)  # its cam rides on the cassette's face over the last 12 degrees
Q_UNLATCH = Q_LATCH - math.radians(5.0)
ARM_K = 2.0  # N m / rad: the hinge spring through which the declared clip rule acts (est)
ARM_DAMPING = 0.02
PRELOAD = 0.05  # rad: latched, the clip presses the arm onto the cassette with ARM_K * PRELOAD = 0.1 N m (est)
CAM_BACK = 0.05  # rad: in the cam zone, an opening torque of ARM_K * CAM_BACK = 0.1 N m (about 0.6 N at the handle)
ARM_LEAD_S = 1.0 / CONTROL_HZ  # free: the spring's anchor leads the arm by one control step (no drag)
PRESENT_TOL = 0.0015  # the clip catches only with a cassette's outer face within 1.5 mm of the arm's line


def arm_sign(side: str) -> float:
    return 1.0 if side == "pos" else -1.0


def arm_dir(side: str, q: float):
    """Unit vector (x, y) along the arm from its pin at joint angle q (0: straight out; pi/2: along -x)."""
    s = arm_sign(side)
    return (-math.sin(q), s * math.cos(q))


def arm_motion(side: str, q: float):
    """The direction the arm's points move as q grows (its leading side; shut: toward the gasket)."""
    s = arm_sign(side)
    return (-math.cos(q), -s * math.sin(q))


def pivot(side: str):
    return (PIV_X, arm_sign(side) * PIV_Y)


# -- the wet-storage racks (fixed, one per cassette) -------------------------------------------------------------------
RACK_X = 0.200
RACK_Y = 0.260
RACK_L, RACK_W, RACK_H = 0.030, 0.120, 0.020  # x, y, z: white HDPE block (est)
SLOT_W, SLOT_L, SLOT_DEPTH = 0.006, 0.104, 0.015  # 0.7 mm of play either side of the 4.6 mm cassette
SLOT_FLOOR = RACK_H - SLOT_DEPTH  # 0.005: a racked cassette's foot
RACK_COLOR = (0.88, 0.88, 0.86)

# Starting poses (origin = the cassette's bottom-edge middle). Gel A: left rack, short plate toward the robot
# (yaw -pi/2: its -y side faces -x); it goes on the +y gasket. Gel B: right rack, short plate away from the robot
# (yaw +pi/2); it goes on the -y gasket. Each needs the same quarter turn (+90 degrees) to go in short plate first.
A_START = (RACK_X, RACK_Y, SLOT_FLOOR, -math.pi / 2)
B_START = (RACK_X, -RACK_Y, SLOT_FLOOR, math.pi / 2)

# -- the demo's numbers ---------------------------------------------------------------------------------------------
TIPS_BELOW = 0.013  # pad tips below the TCP
PAD_FACE = 0.006  # a closed gripper's block: its face this far ahead of the TCP along the closing direction
FIST_H = 0.022  # flat length of the pads above the tips
GRIP_DOWN = 0.022  # carrying: the pad tips 22 mm under the cassette's top (12 mm onto the short plate)
SQUEEZE_X = FX + 0.040  # the squeezing hand straddles both cassettes 40 mm toward their far ends
SQUEEZE_TIPS = 0.078  # its pad tips 12 mm under the cassettes' tops
SQUEEZE_GAP = 2 * (GY + CS_T) - 0.004  # pads commanded 4 mm inside the pair's 45.2 mm: squeezed
PUSH_R = 0.180  # the pushing block on the arm's handle, 180 mm out from its pin
PUSH_BEHIND = AW / 2 + PAD_FACE  # the TCP behind the arm's centre line while pushing
PUSH_Z = ARM_Z0 + TIPS_BELOW  # TCP height pushing: pad tips level with the bar's underside

# -- scoring thresholds (scoring.py) -----------------------------------------------------------------------------------
AGAINST_TOL = 0.0015  # the cassette's inner face within 1.5 mm of its gasket face (gap or overlap)
TILT_MAX_DEG = 3.0
YAW_MAX_DEG = 3.0
X_TOL = 0.006  # its middle within 6 mm of the frame's middle along x (both ends on the gasket legs)
NOTCH_LOW = 0.004  # the short plate's top 0 to 4 mm under the notch
ARM_WIN_DEG = 1.5  # shut: within 1.5 degrees of Q_CLOSED
FLOOR_DROP = 0.020  # a cassette's lowest point 20 mm under the table top: off the bench (latched 0)
LET_GO = 0.020
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
STILL_TURN_DEG = 2.0
STILL_ARM_DEG = 2.0
CREDIT_AGAINST = 0.15
CREDIT_SEATED = 0.15
CREDIT_ARM = 0.20
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
