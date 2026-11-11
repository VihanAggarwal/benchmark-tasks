# Re-spot the black and the yellow: own spot if free, else the highest-value free spot

Snooker referees, and club players refereeing their own frames, re-spot colours many times a frame: after a colour
is potted while reds remain, forced off the table, or potted in a foul. The WPBSA Official Rules of the Games of
Snooker and English Billiards (May 2022), Section 3 Rule 7, say a colour is spotted on its own spot; if that spot is
occupied, on the highest-value spot available (7(e)); and a spotted colour must not touch another ball (7(i)).
Section 2 Rule 18: a spot is occupied if a ball cannot be placed on it without touching another ball. The spot
values are yellow 2, green 3, brown 4, blue 5, pink 6, black 7.

Here a 1:6 model table lies in front of the arms, its long axis along y with the baulk end at -y (by the right arm)
and the top end at +y. It carries the baulk line, the D and six identical, unlabelled white spots, each where
Section 1 of the rules puts it. Seven discs stand for the balls on the table: reds on the black spot and on the pink
spot, a third red, the pink and the blue in play away from their spots, the green on its spot and the brown away
from its spot. The black waits on a tray beyond the top end and the yellow on a tray beyond the baulk end. The
instruction says only "re-spot both as the rules of snooker require, without moving any other ball": which spot is
which, and that an occupied spot sends a colour to the highest-value free one, is the knowledge being tested. The
black's own spot and the pink spot are taken, so the black goes on the blue spot (the table's centre); the yellow's
own spot is free, so it goes there, the D's right-hand corner as seen from the baulk end (+x). The score is read
from where the discs end up, as a referee or a photo with a ruler would check it: 0.5 for each colour, a quarter
for the black on a free spot that is not the right one, and 0 for the whole trial if any ball already on the table
is moved more than 8 mm.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the full-size numbers they come from |
| `scoring.py` | The scorer: plain functions of the nine discs' states (and the arms' pad-tip points and openings, for 'released'), with a small tracker for the disturbance latch, stillness and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of the table's arithmetic, every scoring tier, every zero condition, the release rule, stillness, the disturbance latch, resets, broken states, the exploits we tried, the task file's hooks and trace, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer;
`rigstate.py`, in this folder, reads the arms only to check that each colour was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "You are the referee. After the foul, the black and the yellow have to be re-spotted; they are on the trays at the two ends of the table. Re-spot both as the rules of snooker require, without moving any other ball on the table, then let go. The discs stand in for the balls." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/snooker-respot-highest-free-spot/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "You are the referee. After the foul, the black and the yellow have to be re-spotted; they are on the trays at the two ends of the table. Re-spot both as the rules of snooker require, without moving any other ball on the table, then let go. The discs stand in for the balls." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/snooker-respot-highest-free-spot/dream
```

Run one trial per process (the default). The tracker in the task module keeps where each table disc stood at the
first step (for the disturbance latch), the colours' paths over the last 1 s (for 'still') and diagnostics for the
trace, and starts a new episode when rigstate sees the embodiment reset or the simulator puts every disc back on its
mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: `black_part`, `yellow_part`, `latched`, `latched_at`,
`latched_by`, `disturbed_now` and `max_disp_mm` (each table disc's largest distance from its start); for each colour
its `nearest_spot`, `off_spot_mm`, `tilt_deg`, `bottom_mm` (above the bed), `nearest_puck` and `nearest_puck_mm`,
`held`, `still` and `on_spot`; every disc's `pos_mm`; the colours' `path_mm` in each of the last two 0.5 s windows;
the first step each score was reached; each arm's `left_tips`/`right_tips` and `left_opening`/`right_opening`,
`arms_readable` and `arms_unavailable`, or `rigstate_error`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the nine discs' poses once per control step
(20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the finger
tips) and how far it is open. Distances are between disc centres, seen from above, as a ruler or a top photo reads
them once the arms have withdrawn.

Disturbance first: each of the seven discs that start on the bed is measured (centre, in 3-D) from where it stood at
the first control step of the trial; if any is ever more than 8 mm from it, the score is 0 for the rest of the trial,
even if it is put back. This is the referee's duty not to disturb the table, and it also blocks pushing a red off the
black spot to free it, or knocking the green off its corner to make room for the yellow.

Otherwise the score is the black part plus the yellow part. A colour counts on a spot only when its centre is within
8 mm of the spot's centre (the closest spots, the yellow and the brown, are 48.7 mm apart, so a colour can count on
only one); it lies flat on the bed (tilt under 10 degrees, its lowest point within 3 mm of the bed top, its centre
over the bed); no other disc's centre is within 31 mm of its centre, seen from above (the disc diameter plus 1 mm, so
it touches nothing and is not stacked; the same test makes a spot 'occupied'); it is released (no gripper's pad-tip
point is lower than 10 mm above its top while within that gripper's finger reach of its axis: hypot(half the finger
gap + 10 mm, 20 mm) + 15 mm + 5 mm, which is 80 mm fully open, 52 mm closed on a disc and 42 mm closed on nothing, at
any yaw, since the gripper's yaw is not read); and it is still (its centre moved under 1.5 mm in each of the last two
0.5 s windows, judged from its path, not from the simulator's velocity, which chatters while a part is gripped).

| Part | Condition | Points |
|---|---|---|
| Zero first | Any table disc ever more than 8 mm from its start | 0 in all, for the rest of the trial |
| Black | Counts on the blue spot (the table's centre, the highest-value free spot) | 0.5 |
| | Counts on any other spot (on this table only the brown or the yellow spot can be used without touching or moving a ball) | 0.25 |
| | Anywhere else: on its tray, held, in front of its own occupied spot, between spots, stacked, touching a ball | 0 |
| Yellow | Counts on the yellow spot (the D's right-hand corner seen from the baulk end) | 0.5 |
| | Anywhere else | 0 |

Every outcome, with its score: both colours on their spots, released and still: 1.0; the black on the brown spot,
the yellow right: 0.75; the yellow right and the black on its tray, still gripped, just in front of its own occupied
spot or off every spot: 0.5; the black on the blue spot and the yellow on the brown spot, against the green at the
D's left corner, or on its tray: 0.5; the black on the brown or yellow spot and the yellow not on its spot: 0.25
(with the black on the yellow spot the yellow cannot count there); any table disc moved over 8 mm (a red pushed off
the black spot, the green pushed off its corner, a ball knocked): 0; nothing re-spotted: 0. So every score 0, 0.25,
0.5, 0.75 and 1.0 can be reached and each end state has exactly one. The green, pink and black spots are occupied
throughout (a colour within 8 mm of one would be within 16 mm of the ball there, so touching it). The trial ends
early once 1.0 has held for 2 s; otherwise it is scored on the last step. A state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: pushing the red off the black spot and putting the black
there (0); shoving the green aside to set the yellow at the D's left corner (0); the black stacked on the red on its
own spot (0 for the black); the black set just in front of its occupied spot, toward the top cushion (0 for the
black); a colour held on its spot (0 for that colour until released); both colours dropped near the right spots but
tilted on a cushion or a ball (0 for them); a table disc lifted 9 mm and put back (0); the colours swapped, the black
on the yellow spot and the yellow on the blue (0.25).

## One arm alone

The demo splits the work by reach: the left arm re-spots the black from the top-end tray, the right arm the yellow
from the baulk-end tray. With one arm idle the other colour stays on its tray, and the idle arm's colour part is 0:
the offline kinematic check predicts 0.5 with either arm idle (the other colour correctly re-spotted). Gripping
straight down at any heading, the right arm cannot reach the black's tray (x 24.0 cm, y +37.0 cm, 67.5 cm across
from its base) and the left arm cannot reach the yellow's (`tests/test_scoring.py`, `OneArmReach`, our IK model of
the arm), and nothing on the table brings a colour within reach of the other arm without a handover, which the demo
does not use. The task does not claim the skill needs two arms: the split follows reach, and each colour's part is
scored on its own. (GPU results: 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 26.5 s (530 control steps). It reads the fixed layout: `free_spot()` in `task.py` walks the
spots from the highest value down and takes the first with no disc centre within 31 mm, which gives the blue spot for
the black and the yellow's own spot for the yellow. Each disc is gripped straight down with the fingers closing along
y, opened to 56 mm (13 mm either side of the 30 mm disc; no other disc's centre is within 73 mm of a pick or a place point) and
commanded to 22 mm to close, with the tool point 17 mm above the disc's bottom so the pad tips stay 4 mm above the
tray or the bed. Every open or close takes 1.5 s, the last 20 mm of each descent take 1.5 s, each colour
is set down after a 0.5 s settle over its spot and let go 1 mm above the bed after another 0.5 s, and it travels
with its bottom 68 mm up (48 mm over the cushions, 38 mm over the discs on the bed). The place points lead the spots
by 4 mm along the carry, since carried parts land a few millimetres short; the 8 mm window covers either case. In
order:

1. Both arms pick their colour from their tray at the same time (the trays are 74 cm apart).
2. The right arm carries the yellow to x 40.12, y -17.15 cm (its spot at 39.87, -17.46 plus the 4 mm lead), sets it
   down and lets go, rises and parks at x 20 cm, y -25 cm, 15 cm up. The left arm holds the black 85 mm over its tray
   meanwhile: with both arms over their spots at once, their elbows (link 4) would pass within 1 cm.
3. The left arm then carries the black to x 35.11, y -0.38 cm (the blue spot plus the lead), sets it down, lets go,
   rises and parks at x 20 cm, y +25 cm, 15 cm up.

Offline, the grippers stay at least 28.3 cm apart and the arms' collision capsules at least 12.0 cm; the worst IK
error is 0.08 mm; the smallest joint margin is 12.1 degrees (joint 4, the left arm over the blue spot at travel
height, 0.47 m from its base: the farthest point of the plan). Success holds from step 529 (the black counts once the
open gripper has risen with its pad tips 10 mm over its top, then 2 s); the scripted policy would stop at step 591.
The run limit is 1,200 steps (60 s), more than twice the demo's time. `PLACE_TRIM` in `task.py` moves an arm's place
point if the GPU run shows a colour landing off by a steady amount.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the black back on its tray at x 24.0 cm,
y +37.0 cm, the yellow on its tray at x 24.0 cm, y -37.0 cm; on the bed a red on each of the spots at (35.0, +24.34)
and (35.0, +14.87), a red at (30.0, +7.0), the pink at (40.0, +9.0), the blue at (30.0, -7.0), the green on its spot
at (30.13, -17.46) and the brown at (42.0, -10.0), all flat (centres, cm). The table, cushions, markings and trays are
fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo ends after about 26 s of simulated time, when success
has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

A 296 x 595 mm board 10 mm thick covered in green baize (a 1:6 table bed), fixed to the table with its centre 35 cm in
front of the arm bases and its long axis across the robot; four 12 mm wide, 20 mm tall wooden strips covered in dark
green cloth round it as cushions. White 0.2 mm vinyl: a 4 mm baulk line across the bed 122.8 mm from the bottom
cushion's face, a D of radius 48.7 mm on it, and six identical 8 mm round spots at the positions in `task.json`'s
`setup` (all 1:6 of the WPBSA table: 3569 x 1778 mm playing area, baulk line 737 mm, D radius 292 mm, black spot
324 mm from the top cushion's face). Two 60 x 60 x 10 mm wooden trays beyond the two ends. Nine plastic discs 30 mm
across and 20 mm tall, about 25 g (turned acrylic or stacked counters glued together), painted black, yellow, three
red, pink, blue, green and brown.

Scoring a real trial: before the run mark each table disc's outline (or photograph the table); afterwards check no
table disc is more than 8 mm from its mark. Then measure each colour's centre from the nearest spot's centre (within
8 mm), check it lies flat on the bed, that no other disc's centre is within 31 mm of it (a 31 mm gauge between
centres), and that it is free of the grippers and at rest.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
