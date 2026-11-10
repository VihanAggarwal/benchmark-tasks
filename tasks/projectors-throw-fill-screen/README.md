# Set two classroom projectors at the throw distance that fills each screen

Audio and video technicians and school IT and AV staff set up portable projectors in classrooms, lecture rooms and
meeting rooms. A fixed-lens projector's throw ratio is printed by its maker: throw distance (from the front of the
lens to the screen) = throw ratio x picture width. To fill a 1.6 m screen, a 1.5:1 projector stands 2.4 m from it and
a 1.0:1 short-throw projector 1.6 m, so two different projectors for two identical screens stand at different
distances. The lens must also be on the screen's centre line and the projector square to the wall, or the picture is
shifted off the screen and keystoned into a trapezoid (digital keystone correction costs sharpness). Working out the
distance from the ratio, setting it with a tape and squaring the projector to the wall is the technician's routine
before a session.

Here the front of a classroom stands on the table at 1:10: the front wall with two matt white screens side by side,
each 160 mm wide (1.6 m), and in front of it the projector table (classroom tables pushed together, 75 mm tall). Two
projectors stand on the table turned sideways, lenses toward the middle, as they were unpacked: a black one with a
1.5:1 throw ratio on the left and a white one with a 1.0:1 throw ratio on the right. Each must face its screen, square
to the wall, its lens on the screen's centre line, with its lens 240 mm (black) or 160 mm (white) from the screen.
The score is what a technician's tape from the wall, a set square and a top photo show once the robot has let go: the
picture width each projector would throw (its distance divided by its own ratio) within 5 % of the screen's width, the
projector square within 5 degrees and its lens within 6 mm of the centre line. Each projector needs a quarter turn, and the direction
of that turn is the policy's choice: the wrong way points the lens at the back of the room.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place (the throw distances are computed there from the ratios) |
| `scoring.py` | The scorer: plain functions of the two projectors' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each projector as one rigid body of two coloured boxes (body and lens) and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads each arm's tool point, finger-tip point and gripper opening for the scorer's 'let go' (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, everything that earns nothing, the exploits we tried, letting go, resets, the demo plan, its carry paths and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (the wall, screens, centre marks and table), the standard `task_score` scorer, and two
helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front or
back. `mechkit.py` (imported by `task.py`) builds each projector from two boxes (the body and a lens barrel of a
contrasting colour on its front face) as one loose rigid body through a hook on `yam_arena.scene`, and reports its
pose (the frame origin at the middle of its body's base, and its orientation) to the scorer. It changes nothing else
the embodiment does, and it stops the run with an error if a projector was not built as a compound body. The
technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the
task file's import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

`yam_arena` hands a task file's `success()` and `score()` only the objects' states. `rigstate.py` (imported by
`task.py`) wraps the embodiment's step and reset once, from the task file's import, and before every control step
reads each arm's tool point, the point between its finger tips and how far its gripper is open (0 closed, 1 fully
open), so the scorer can tell whether a projector has been let go. It reads only state the simulator already has and
changes nothing the embodiment does. Wrapping the embodiment from a task file also follows `tasks/bearing-press`;
`rigstate.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, one projector each): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the two projectors for the two screens on the front wall, one projector per screen. Each screen is 160 mm wide (1.6 m at this 1:10 scale). The black projector has a 1.5:1 throw ratio and the white one 1.0:1 (throw distance, from the front of the lens to the screen = throw ratio x picture width). Stand each projector upright on the table facing its screen, square to the wall, its lens on the screen's centre line, at the distance where its picture just fills the screen." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/projectors-throw-fill-screen/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the two projectors for the two screens on the front wall, one projector per screen. Each screen is 160 mm wide (1.6 m at this 1:10 scale). The black projector has a 1.5:1 throw ratio and the white one 1.0:1 (throw distance, from the front of the lens to the screen = throw ratio x picture width). Stand each projector upright on the table facing its screen, square to the wall, its lens on the screen's centre line, at the distance where its picture just fills the screen." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/projectors-throw-fill-screen/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `success`, `zero_reason`, `valid_state`; for each
screen its `credit` and the `projector` counted; for each projector `credit`, `why`, `screen`, `counted`, `throw_mm`,
`picture_mm`, `width_err_pc`, `square_err_deg`, `nearest_screen`, `centre_off_mm`, `origin_mm`, `tilt_deg`,
`lowest_mm`, `on_table`, `path_mm`, `let_go`, `zone_gap_mm` (each gripper's finger zone to it); `arms_read` and
`openings`; then the projectors' raw `origins`, `quats`, `low_z` and `vels`, and the arms' `left_tcp`, `right_tcp`,
`left_tips`, `right_tips`, or `rigstate_error` when the arms could not be read). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every reset
(mechkit's reset counter, or the simulator putting both projectors back on their marks in one step); the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read at the last step from the projectors' poses and
from each gripper's opening and finger positions.

Seen from above (x up the page, toward the front wall; y to the left):

```
   y +25          y +20     y +12      y +4    y -4      y -12     y -20          y -25
   ================ front wall, x 47.0 to 48.0 (screens' faces at x 46.8) ================
                  [ left screen 160 mm  ]        [ right screen 160 mm ]
                            |  centre line              |  centre line
                            |                           |
        table far edge x 37.0 --------------------------------------------------
                            |                           |
                            |                     [W] white 1.0:1, lens at x 30.8
                            |                           |   (160 mm from the screen)
                 [B]--> lens  at start      lens <--[W]  at start (x 30.0, y -15)
            (x 30.0, y +15) |                           |
                            |                           |
                   [B] black 1.5:1, lens at x 22.8      |
                            |   (240 mm from the screen)
        table near edge x 17.0 -------------------------------------------------

          left arm base (y +30.5)              x 0             right arm base (y -30.5)
```

For each projector: its **throw distance** is how far the centre of its lens's front face is in front of the
screens' faces (x 46.8 cm), measured square to the wall; its **picture width** is that distance divided by its own
throw ratio (1.5 for the black, 1.0 for the white); its **squareness** is the angle, seen from above, between its
lens's axis and straight at the wall; its **centre-line offset** is how far the centre of its lens is, sideways, from
the nearer screen's centre line. **Standing**: upright (tilted 5 degrees or less), on the projector table (the middle
of its base over the table, its lowest point within 3 mm of the table's top), still (the middle of its base moved
under 5 mm over the last 0.5 s, turns counted at its corners: positions, not the simulator's velocities, which
chatter while a part is gripped) and **let go** by both grippers: each gripper is at least 90 % open (fingers
84.5 mm apart: it holds nothing) or its finger zone is more than 10 mm from the projector's body and lens. The arm
state gives no gripper heading, so the finger zone is round: every point within a radius of the 38 mm segment from
the point between the finger tips back up the gripper, the radius reaching a pad's outer corner (half the finger gap
plus the 8 mm pad, and half the pad's 27 mm width across: 16 mm closed, 57 mm fully open). On a step where the
grippers cannot be read, no projector counts as let go. A projector is **set up for a screen** when it is standing,
its lens toward the wall within 10 degrees of square and its lens's centre within 12 mm of that screen's centre
line.

| Outcome, per projector | Credit |
|---|---|
| Set up for a screen, square within 5 degrees, lens within 6 mm of the centre line, picture 152 to 168 mm wide (within 5 %: a throw of 228 to 252 mm for the 1.5:1 projector, 152 to 168 mm for the 1.0:1) | 0.5 |
| Set up for a screen, picture 136 to 184 mm wide (within 15 %: a throw of 204 to 276 mm, or 136 to 184 mm), but off square by 5 to 10 degrees, 6 to 12 mm off the centre line or 5 to 15 % off the width | 0.25 |
| Anything else: not standing (tipped, on its side, lifted, off the table, on the other projector, moving, still held by a gripper), turned more than 10 degrees from square (sideways as at the start, or lens to the back), off both centre lines, or a picture more than 15 % too wide or too narrow | 0 |
| Total: each screen counts its best projector once; the sum for the two screens | 0, 0.25, 0.5, 0.75, 1.0 |

Examples: the start 0; both right 1.0; one right and the other 20 mm too far (picture 173 mm for the black, 180 mm
for the white) 0.75; both 20 mm too far 0.5; one right and the other turned the wrong way, lens to the back, 0.5; one
20 mm too far and the other left at the start 0.25; both on one screen 0.5; each at the other's distance (the black at
160 mm, the white at 240 mm: pictures 107 and 240 mm) 0. Either projector may serve either screen (the screens are
the same width), judged on the distance its own ratio needs. The trial ends early once 1.0 has held for 3 s. A
simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing more than they should, as the tests show: both projectors on one screen count
once (0.5); a projector turned the wrong way round with its body where a correct one would stand points its lens at
the arms and earns nothing; each projector at the other's distance earns nothing; a projector pushed to the table's
far edge (a 100 mm throw) earns nothing; a projector on its side, upside down, stood on the other projector or on the
floor below the table earns nothing; a projector still held by a gripper earns nothing, even set down at the right
place (a gripper less than 90 % open with its fingers within 10 mm of it). A projector resting within 3 mm of the
table's top counts as on it (contact tolerance), once it is let go and still.

## One arm alone

Offline (the kinematic check with `--one-arm`), the scripted demo scores 0.5 with either arm idle: the idle arm's
projector stays where it started, turned sideways (nothing), and the other arm sets its own projector up (0.5). On the
GPU (L40S, one trial each, see 'Results in simulation') it is the same: 0.5 with the left arm idle and 0.5 with the
right arm idle.

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips at tool-point heights
from the table up to the carry height, every yaw in 15 degree steps within joint 6's +-120 degrees): neither arm
reaches the other arm's projector where it starts, at its centre or at its lens's tip. At x 30 cm, 17 mm above the
projector table, the left arm reaches no further than y -7.5 cm, and the white projector's nearest point is at
y -13.2 cm; the right arm, mirrored, is the same. So by pick and place one arm alone has one projector and earns at
most 0.5 (the left arm could even set the black projector up at the right screen: it reaches that screen's centre
line at the black projector's distance). The two arms work at the same time, each on the projector on its own side. We have not tried
a single arm nudging the other projector with an open finger (the open fingers reach 55 mm past the tool point, which
is about enough to touch the white projector's lens) or pushing it within reach, or a tilted grip, and do not claim
those are impossible.

## Demo and run limit

The scripted demo takes 21.6 s (432 control steps), both arms at once. Each arm goes over its projector with the
fingers open (94 mm) and the grip turned to -90 degrees (left) or +90 degrees (right), comes down, takes the last
30 mm over 1.5 s to the tool point 17 mm above the table, and grips across the 34 mm body at its middle (0.8 mm behind
its centre of mass; commanded 26 mm, closing over 1.5 s), the pads on the body from 4 mm above the table to its top.
It lifts the projector 20 mm, carries it over 3.5 s turning the wrist to 0 degrees (the projector turns a quarter turn:
the black one anticlockwise, lens from -y to the wall; the white one clockwise, lens from +y to the wall), lowers the
last 18 mm over 2 s, settles, opens to 94 mm with the projector 2 mm above the table, rises straight up and parks. The
places: the black projector's body centred at x 21.0 cm, y +12.0 cm (lens at x 22.8 cm, 240 mm from the screen), the
white one's at x 29.0 cm, y -12.0 cm (lens at x 30.8 cm, 160 mm from the screen), both square to the wall and on the
centre lines. On the way, both projectors stay over the table, at least 20 mm inside its near edge and 30 mm inside
its sides, and at least 180 mm apart (tested).

Offline (kinematic check): worst inverse-kinematics error 0.08 mm; smallest joint margin 15.3 degrees (joint 4);
arm links at least 6.9 cm apart (capsule proxy), tool points at least 25.3 cm apart; pad tips never lower than 4 mm
above the projector table (79 mm above the station table). The projectors count once the fingers are at least 90 %
open, near the end of 'let go' (17.7 s), and success has held for its 3 s at step 414; the scripted policy stops at
step 493. The run limit is 1,200 steps (1 minute), which leaves a policy room for slower carries, a
set-down to regrip, or a second try.

For the GPU run: read each projector's `throw_mm` (aims 240 and 160; full credit 228-252 and 152-168),
`square_err_deg` (aim 0; full credit 5) and `centre_off_mm` (aim 0; full credit 6) in `TASK_TRACE`, and `openings`
(about 0.36 while a projector is gripped, 1.0 after 'let go'). The arm sags a little under each 50 g projector, so a
projector touches the table slightly before its slow descent ends; it counts only once the grippers have opened, and
the 3 s success hold ends the trial after the grippers have risen clear.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the black projector with its body
centred at x 30.0 cm, y +15.0 cm, its lens toward -y (toward the middle), and the white projector with its body
centred at x 30.0 cm, y -15.0 cm, its lens toward +y, both upright on the projector table. The wall, screens, centre
marks and table are fixed. Nothing is consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 22 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 10 mm MDF board 500 x 190 mm painted pale green, clamped upright across the far end of the table as the front wall
(its face at x 47.0 cm); two 160 x 100 mm matt white PVC sheets 2 mm thick glued flat on its face, their bottoms 80 mm
above the table, centred 120 mm either side of the midline, with a black PVC strip 4 mm wide, 60 mm tall and 2 mm
thick, 15 mm above the table, below each centre; a 200 x 400 mm block 75 mm tall with a beech-effect laminate top (or a box of that size) as the projector table,
at x 17.0 to 37.0 cm. Two 1:10 projector models of 50 g: a 34 x 26 x 14 mm block (black or white plastic, or painted
wood) with a 14 x 10 mm lens block 5 mm deep glued to the middle of one 34 mm face (silver on the black projector,
black on the white one), labelled 1.5:1 and 1.0:1 underneath. Everything is light and blunt, and nothing is powered or
hot, so trials are safe. Score with a tape from the screen's face to the front of each lens, a set square against the
wall for squareness, a top photo for the centre lines and the video for whether the grippers let go.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
