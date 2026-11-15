# Open a casement window wide on its peg stay, to the last hole

A side-hung timber casement window stands across the table, hinged on the robot's left and opening outward. It rests
open a little on its peg stay: the stay's second hole is on the stay pin. The robot has to open it as wide as the
stay allows, with the stay's last hole on the pin, and let go.

## Why this is everyday work

Peg stays are the standard fitting sold for timber and steel casement windows: a bar hinged on the bottom of the sash,
with a row of holes, dropped over a pin on the window board. Opening the window wider is done many times a year by
anyone who airs a room with such a window. The hand's problem is real: the bar is tied to a swinging sash, so it
moves on the arc the linkage dictates; pushed along the pin it drops into the next hole it meets and the window
opens only one setting; let go off a hole, the sash is free to swing and slam. A person lifts the bar, carries it
clear over the nearer hole while the sash swings out, lowers it onto the pin short of the last hole and pushes on
until the hole drops over the pin.

## Sources

- Stay and pins: Samuel Heath P4512-B "Casement stay, 254mm", product page
  https://www.samuel-heath.com/product/p4512-b/ with the maker's CAD files: the drawing
  (https://d2gu5dwc5p45bw.cloudfront.net/pdf-cad-files/p4512-b.pdf: 278.64 mm overall, pin plates 40 x 15 mm,
  four holes) and the 3D model p4512-b.igs (https://d2gu5dwc5p45bw.cloudfront.net/igs-cad-files/p4512-b.igs,
  Autodesk Inventor, mm). Read from the model's curves: holes 7.2 mm across at a 40.5 mm pitch, 64.52 mm to the
  first from the swivel axis; bar 13.6 mm wide and 8 mm deep with an 8.1 mm groove 3.75 mm deep along its underside;
  swivel 11.11 mm in front of the plate's back, lift pin 20.02 mm along the bar; tail ball 14 mm across, its centre
  260.5 mm from the swivel; pin taper 7.9 mm across at 6.5 mm up, 5.56 mm at 22.04 mm, top at 22.5 mm. Two pins are
  supplied (the product page).
- The mechanism (holes dropped over a pin on the frame set the opening): supplier pages such as
  https://www.doorfurnituredirect.co.uk/saa-casement-stay-10in-254mm.html.
- Fitting: "Place the stay central to the window assembled with the 2 pins" (a maker's guide to fitting a window
  stay, From The Anvil, https://www.graceandgloryhome.co.uk/wp-content/uploads/2025/01/Window_Stay_-_Fitting_Guide.pdf).
- Window size 488 x 600 mm overall: a common UK window size (488 and 630 mm widths with 450 and 600 mm heights appear
  in UK standard size lists, e.g. https://www.everest.co.uk/windows/standard-house-window-size/).

Every figure not listed above is an estimate (stated in task.json "fidelity"): frame and sash sections, softwood
at 500 kg/m3, the sash's mass (3.1 kg with a 4 mm pane), hinge friction 0.2 N m and damping, where the two pins sit
(under holes 1 and 4 when shut), the window board's depth and the start opening (on the second hole, 13.5 degrees).

## Why it starts open on the second hole

With the window shut, the stay lies along the sash, its centre line 11.1 mm in front of the sash's face, so the ball
is 4 mm from the sash. A top-down gripper needs about 15 mm of room around what it holds, so the shut stay cannot be
taken from above. Resting on its first open hole (a common setting for airing a room at night) the stay points well
into the room and the task is the real one: open it wide.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Window board (fixed) | x 0.220 to 0.510, y -0.244 to +0.244, top 40 mm up |
| Sash hinge (outside face) | x 0.464, y +0.198 |
| Pins A, B | x 0.4089; y -0.0745 (A), +0.0470 (B) |
| Stay swivel at the start | x 0.4888, y -0.1426 |
| Ball at the start | x 0.2905, y +0.0263, 63 mm above the table |
| Sash opening on holes 2, 3, 4 | 13.5, 22.4, 30.8 degrees |

The window is one fixed-base articulation: frame (jambs, head), sash on a vertical hinge, the stay's swivel on the
sash (vertical) and its bar on the lift pin (horizontal, across the bar). The board with both pins is a separate fixed
object, so the bar rests on the pins and the board. The pins' tapers are 16 flat staves; the holes are rings of 16
staves through the bar's web; the groove's two legs straddle the pin. Nothing is latched or moved by code.

## Scoring

Read at the end. A hole is on the pin when its centre is within 2.0 mm of pin A's axis in plan and the pin's tip
reaches the bar's top face or stands above it. 1.0: the last hole (4) on the pin, both grippers 30 mm or more from
the stay for the last 20 steps, and the ball and the sash's free corner moved 1 mm or less over that second. 0.5:
hole 3 on the pin, let go. 0.25: hole 3 or 4 on the pin, still held. 0: anything else, including the start (hole 2),
the bar resting on the pin's top or on the board (the sash free), and any run in which a hand entered the space of the
glass pane (latched). In a real trial: look from above for the pin through the hole and lay a straightedge on the bar;
then push and pull 5 N at the sash's free edge with a spring balance.

## The demo (right arm; the task is one-handed, as it is for a person)

Open loop: timed waypoints from the known start layout; every pose of the stay is the linkage's own.

1. Grip the ball from above, the pads on its two faces (across the bar), pad tips 4 mm below its centre.
2. Lift: the second hole comes off the pin, the web 6 mm over the pin's top (the groove's legs 2.25 mm over it).
3. Carry: the pin passes along under the bar from the second hole to 12 mm short of the last, the web kept 6 mm over
   it, so the third hole passes over the pin; the sash swings from 13.5 to 28.3 degrees; the hand turns 23.6 degrees
   with the bar.
4. Lower the bar onto the pin's top (the groove over it), commanded 2 mm lower.
5. Push on 12 mm, still pressing, the hand commanded 3 mm ahead along the bar (the arm's measured lag on the GPU):
   the last hole drops over the pin; then back to centre and down onto the seat, commanded 1 mm below.
6. Open, straight up 60 mm, away; hold.

About 27 s (538 steps); run limit 800 steps (40 s). A dense trace of the key parts is written every control step
with `TASK_TRACE=<file>`: the three joints, the ball, each hole's centre and where pin A is in the bar's frame.

Wrong ways (counterfactual runs, never the submitted demo):

- `DEMO_VARIANT=slide_from_start` (expected 0.5): lifted off the second hole, lowered straight back onto the pin and
  pushed along it; the third hole drops over the pin; let go there.
- `DEMO_VARIANT=release_on_pin_top` (0): carried right, lowered onto the pin's top 12 mm short of the last hole and
  let go: no hole on the pin, the sash free.
- `DEMO_VARIANT=past_last` (0): carried 15 mm past the last hole, lowered onto the pin's top there and let go.
- `DEMO_VARIANT=hold_at_last` (0.25): the demo without letting go.

One arm: `DEMO_IDLE_ARM=left` is the demo itself (the right arm does everything); `DEMO_IDLE_ARM=right` leaves the
window on its second hole (0).

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

## Running

```
inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
DEMO_VARIANT=past_last inspect-robots run ...      DEMO_IDLE_ARM=right inspect-robots run ...
python3 -m unittest discover -s tests
```

Files: `task.py` (objects, hooks, demo), `layout.py` (every dimension and the linkage's kinematics), `scoring.py`,
`offline_stay.py` (the linkage moved from commanded arm poses, used only without a simulator), `mechkit.py` and
`rigstate.py` (unchanged kit files).
