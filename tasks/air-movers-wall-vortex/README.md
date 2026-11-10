# Set air movers to dry a wet room: outlets at the walls, 15-45 degrees, one way round

Water-damage restoration technicians and building maintenance staff dry flooded rooms with air movers, small
snail-shell blowers. Restoration training based on the IICRC S500 standard for water damage restoration places them
along the wet walls with the outlet at the base of the wall, aimed 15 to 45 degrees into it so the air sweeps along the
wall and the carpet edge ("wall washing"), about one for every 10 to 16 ft of wet wall, and all blowing the same way
round the room so the air turns as one vortex. Movers blowing straight at a wall, parallel to it or against each other
leave slow-drying spots. Setting them out is the first job after the water is extracted.

Here a 1:16 room stands on the table: a wet carpet, a left wall, a right wall and a back wall, all cut down to their
bottom 16 cm so the camera sees the floor, and no wall on the side nearest the arms. Four blue air movers, each with a
black front end, a black outlet at the bottom of its front and a grey handle on top, stand in the middle of the room as
the crew wheeled them in, all blowing toward the right wall. Each side wall is 18.4 ft long, so each takes two movers
(the back wall's movers are left out to keep the task to four). The score is what a crew lead checks with a glance, a
protractor and a tape, and what a top photo shows: each outlet within 15 mm (24 cm) of its wall, blowing into it at 15
to 45 degrees; the two on each wall 190 to 305 mm (10 to 16 ft) apart; and all four going the same way round. Whichever
way round is chosen, the left pair turns 120 degrees and the right pair 60, so the start does not suggest one: the robot
has to pick a way and keep to it on both walls. Two pairs angled opposite ways make two parallel streams with no loop
(0.5).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four movers' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each air mover as one rigid body of coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every reason a mover is not set, the exploits we tried, resets, the demo plan, its carry paths and pad clearances, and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (the carpet and the walls), the standard `task_score` scorer, and one helper module next
to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front or
outlet. `mechkit.py` (imported by `task.py`) builds each air mover from four boxes (the housing's blue back and black
front 5 mm, the black outlet, the grey handle) as one loose rigid body through a hook on `yam_arena.scene`, and reports
its pose (the frame origin at the middle of the housing's base, and its orientation) to the scorer. It changes nothing
else the embodiment does, and it stops the run with an error if a mover was not built as a compound body. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, two movers each): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The carpet in this room is wet (1:16 scale; the wall nearest you is left out). Set the four air movers to dry it the way restoration crews do: two along each side wall, each with its black outlet no more than 15 mm (24 cm) from the wall and blowing into the wall at 15 to 45 degrees so the air sweeps along it, the two on each wall 190 to 305 mm (10 to 16 ft) apart, and all four blowing the same way round the room." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/air-movers-wall-vortex/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The carpet in this room is wet (1:16 scale; the wall nearest you is left out). Set the four air movers to dry it the way restoration crews do: two along each side wall, each with its black outlet no more than 15 mm (24 cm) from the wall and blowing into the wall at 15 to 45 degrees so the air sweeps along it, the two on each wall 190 to 305 mm (10 to 16 ft) apart, and all four blowing the same way round the room." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/air-movers-wall-vortex/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `success`, `zero_reason`, `valid_state`, `credited`
per way round, `best_sense`, `walls` (per way round and wall: the movers set, the credit, the spacing used); for each
mover `set`, `why`, `tilt_deg`, `lowest_mm`, `path_mm`, `origin_mm`, `heading_deg`, `wall`, `angle_deg`, `gap_mm`,
`sense`; then the raw `origins`, `quats`, `low_z` and `vels`). Videos: `inspect-robots video <log>.json`. The scorer
keeps one trial's record per process and starts it over at every reset (mechkit's reset counter, or the simulator
putting all four movers back on their marks in one step); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the movers' poses at the last step.
A mover's **airflow** is the direction from the back of its housing to its outlet, seen from above.

Seen from above (x up the page, away from the arms; y to the left); the demo's places, clockwise:

```
                         back wall, face at x 47.0
          +-------------------------------------------------+
          |  [2] place x 42.0, y +15.6   [4] place x 42.0,  |
          |      airflow 30 deg left         y -15.6,       |
   left   |      of +x, into the wall    airflow 30 deg     |   right
   wall   |  [2] start x 42.0, y +13.0   right of -x        |   wall
   face   |                              [4] start x 42.0,  |   face
  y +18.5 |                                  y -13.0        |  y -18.5
          |  [1] start x 33.0, y +13.0   [3] start x 33.0,  |
          |  (all four start blowing         y -13.0        |
          |   toward -y, the right wall)                    |
          |  [1] place x 19.0, y +15.6   [3] place x 19.0,  |
          |      airflow 30 deg left         y -15.6,       |
          |      of +x                   airflow 30 deg     |
          |                              right of -x        |
          +- - - - - - - open side, x 12.0 (no wall) - - - -+
            left arm base (y +30.5)      right arm base (y -30.5)

  clockwise seen from above: away from the arms along the left wall, toward them along the right wall
```

**Placed**: tilted 10 degrees or less, its lowest point within 3 mm of the carpet, its centre (the middle of the
housing's base) inside the room, and still (moved under 5 mm over the last 0.5 s, turns counted at its corners:
positions, not the simulator's velocities, which chatter while a part is gripped). It blows into the **left wall**
(face at y +18.5 cm) if its airflow points to +y at all, into the **right wall** (face at y -18.5 cm) if to -y. Its
**angle into the wall** is between the airflow and the wall seen from above (0 along the wall, 90 straight at it); its
**gap** is measured square to the wall, from the outlet's nearest point to the line of that wall's face (the outlet is
as wide as the housing and in front of it, so at any angle its wall-side corner is the mover's nearest point to the
wall). It goes **clockwise** seen from above if
its airflow points away from the arms (+x) along the left wall or toward them (-x) along the right wall; anticlockwise
the reverse.

| Outcome | Score |
|---|---|
| A mover is **set**: placed, 15 to 45 degrees into its wall, outlet gap 15 mm or less (touching counts) | (counted below) |
| For each way round, each side wall credits the movers set on it going that way: none 0, one 1, two or more 2 if some two have centres 190 to 305 mm apart along the wall (in x), else 1 | at most 2 per wall |
| Score: 0.25 times the larger of the clockwise and anticlockwise totals (left wall plus right wall) | 0, 0.25, 0.5, 0.75, 1.0 |
| No mover set (left in the middle, under 15 degrees or over 45 degrees to the wall, outlet more than 15 mm off, blowing away from the wall, tipped, lifted, outside the room) | 0 |

Examples: the start (all four blowing straight at the right wall, 90 degrees) 0; all four set the same way round, two
per wall 230 mm apart, 1.0 (clockwise or anticlockwise); three right and the fourth at 60 degrees into its wall, or with
its outlet 20 mm off the wall, 0.75; one wall's pair right and the other wall's two set 100 mm apart, 0.75; both walls'
pairs set but all four blowing toward the back wall (two parallel streams, no loop) 0.5; one clockwise and one
anticlockwise on each wall 0.5; one wall's pair right, the other two left in the middle, 0.5; one mover set 0.25; all
four nearly parallel to their walls or pointed straight at them 0. The trial ends early once 1.0 has held for 3 s. A
simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it was.

Why the bands: the angle band (15 to 45 degrees) and the spacing (one mover per 10 to 16 ft of wall) are the figures
restoration training quotes; 10 and 16 ft are 190.5 and 304.8 mm at 1:16, rounded to 190 and 305 mm. The gap (24 cm at
full scale) is this task's reading of "at the base of the wall". The walls are 350 mm (18.4 ft) long, so two movers per
wall is the training's count, and both ends of the spacing band can be missed: two set 100 mm apart, or one at each end
of the wall 315 mm apart, credit one. At 30 degrees the outlet's wall-side corner is 21.3 mm from the mover's centre
toward the wall, so the demo's centres 15.6 cm out leave a 7.7 mm gap (7.3 mm short of the band's 15 mm end).

Unintended strategies earn nothing more than they should, as the tests show: a mover set with its back to the wall,
blowing out into the room, blows toward the other wall and is far from it (not set); a mover outside the room beyond a
wall is not inside the room; one stacked on another, standing on a wall or still held more than 3 mm up is not on the
carpet; a mover on its side or upside down is not upright; three or four set on one wall credit two; movers going
both ways round are counted for whichever way credits more, so a mover blowing against the others adds nothing; a
mover held still 2 mm above the carpet counts as placed (the scorer sees only the movers; `task.json` fidelity says so).

## One arm alone

On the GPU (see 'Results in simulation') the scripted demo with either arm idle scored 0.5, as the offline kinematic
check (`--one-arm`) predicted: the idle arm's two movers stayed in the middle of the room, still blowing straight at the
right wall (not set), and the other arm set its own two along its own wall, 30 to 31 degrees into it with the outlets
6.2 to 7.1 mm off and 231 mm (left arm idle) or 232 mm (right arm idle) apart (0.5).

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips at tool-point heights
13 to 60 mm, every yaw in 15 degree steps within joint 6's +-120 degrees, over a 5 x 3 grid of points on each mover's
outline where it starts: 4,080 poses): neither arm reaches any point of the other side's movers. The right arm reaches
across the middle to y +10.5 cm at x 31.7 cm, and less further back: 5 mm short of the corner of mover 1's outlet
(mover 1 blows toward the middle) and 10 mm short of its housing; the left arm, mirrored, stops 10 mm short of the
back of mover 3's housing. So by pick and place one arm alone has two movers and earns at most 0.5 (both on its own
wall, or one on each wall the same way round). It would need the other side's movers to earn more, and even then it
could set at most one on the far wall: at x 31 cm or more an arm reaches no further than about 10 cm across the
middle, while any grip on the housing of a mover set on the far wall is at least 12.7 cm across it, so the far wall's
two movers (190 mm apart, one of them at x 31 cm or more) cannot both be set by one arm, and it could earn at most
0.75. The two arms work at the same time, each on the movers on its own side. We have not tried a single arm nudging
the far movers over with an open finger (the open fingers reach 55 mm past the tool point), pushing them, or a tilted
grip, and do not claim those are impossible.

## Demo and run limit

The scripted demo takes 41.2 s (824 control steps), both arms at once, clockwise seen from above. Each arm sets the
mover nearer the arms first, at the place nearer the open side, then the other at the place by the back wall. For each
mover the arm goes over it with the fingers 60 mm open, comes down beside it, takes the last 30 mm over 1.5 s and grips
across the housing's 26 mm width (commanded 18 mm, closing over 1.5 s) at its middle (the centre of mass is 1 mm toward
the outlet), the tool point 30 mm above the table: the pads hold the housing from 17 mm up to its top (31 mm), their
tips 7 mm over the 10 mm walls, the 10 mm handle between them with 8 mm to spare each side, the gripper's body 28 mm
over the handle. It lifts the mover 30 mm and carries it over 4 s in two segments, turning the wrist: the left arm turns
its movers 120 degrees anticlockwise (airflow heading from -90 to +30 degrees: away from the arms, 30 degrees into the
left wall; yaw -90 through -30 to +30), the right arm 60 degrees clockwise (heading from -90 to -150 degrees: toward the
arms, 30 degrees into the right wall; yaw +90 through +60 to +30). Then it settles 0.5 s, lowers the last 28 mm over 2
s, settles, opens to 50 mm with the mover 2 mm above the carpet and rises straight up; it turns the wrist back in two
segments on the way to the second mover. Joint 6 stays between about -68 and +67 degrees (the other grip yaws, 180
degrees round, would pass its +-120 degree limit).

The places: movers 1 and 2 at x 19.0 and 42.0 cm, y +15.6 cm, airflow +30 degrees (away from the arms and into the left
wall); movers 3 and 4 at x 19.0 and 42.0 cm, y -15.6 cm, airflow -150 degrees (toward the arms and into the right wall).
Each is 30 degrees into its wall (15 degrees inside each end of the band) with its outlet 7.7 mm from the wall, and each
pair is 230 mm (12.1 ft) apart (40 mm above the band's low end, 75 mm below its high end). Gripping, the wall-side pad
is 5.8 mm from the wall's face; opened to 50 mm it passes 4.6 mm over the wall's face line, its tips 9 mm above the
wall's top (the tool point is 32 mm up when it lets go). Going down beside a mover with the fingers 60 mm open, the pads
stay more than 20 mm from the other movers and 12 mm short of the back wall. On the way, every carried mover stays more
than 60 mm from every other mover (tested).

Offline (kinematic check): worst inverse-kinematics error 0.09 mm; smallest joint margin 11.6 degrees (joint 4);
arm links at least 9.3 cm apart (capsule proxy), tool points at least 26.0 cm apart; pad tips never lower than 17 mm
above the table. Success has held for its 3 s at step 771; the scripted policy stops at step 885. On the GPU the
demo scored 1.0 and the trial ended at step 773, once success had held for its 3 s: all four movers upright, 30.3 to
31.2 degrees into their walls, outlets 6.7 to 7.9 mm off, the pairs 230 and 231 mm apart, all clockwise. The run limit is
1,200 steps (1 minute), which leaves a policy room for slower carries, a set-down to regrip for a bigger turn, or a
second try.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four air movers upright on the carpet,
all blowing toward the right wall (outlets toward -y): air mover 1 centred at x 33.0 cm, y +13.0 cm; air mover 2 at
x 42.0 cm, y +13.0 cm; air mover 3 at x 33.0 cm, y -13.0 cm; air mover 4 at x 42.0 cm, y -13.0 cm (a mover's centre is
the middle of its housing's base). The carpet and walls are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 41 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 350 x 370 mm mid-grey felt sheet 1 mm thick (the carpet) laid on the table from x 12.0 to 47.0 cm, y -18.5 to
+18.5 cm; three strips of off-white painted MDF 10 x 10 mm in section (two 360 mm long for the side walls, one 370 mm
long for the back wall) glued down along its left, right and far edges. Four air mover models: blue blocks 30 x 26 x
30 mm (weighted to 25 g) with their front 5 mm painted black, a 5 x 26 x 12 mm black block glued across the bottom of
that black face for the outlet, and a 20 x 10 x 6 mm grey strip on top along the long axis for the handle (a 1:16
model of a snail-shell air mover of about this size works too). Everything is light and blunt, so trials are safe.
Score from a top photo: a protractor against each wall's face for the angle, a ruler square to the wall from its face
to the outlet's corner for the gap, and a tape between the housings' centres along the wall for the spacing; the black
front ends show which way each mover blows.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The demo's trial ended at step 773 with 1.0 held for 3 s. Dream-1 ran the full 1,200 steps (the instruction above, unchanged) and left all four movers where they started, blowing straight at the right wall (mover 3 moved about 1 mm), so none was set: 0.0.
