# Space three nap cots 3 feet apart and set the pillows head to toe

Childcare workers and preschool teachers set out the rest area before every nap time (O*NET 39-9011.00 Childcare
Workers, 25-2011.00 Preschool Teachers). Child-care licensing rules and early-care health and safety guidance ask for
cots, mats or cribs to be spaced apart, most often 3 feet (some states accept 2 feet, or less with a solid barrier
between them), and many programs also lay the children head to toe, each child's head beside its neighbours' feet, so
that sleeping children do not breathe into each other's faces; the CDC's COVID-19 guidance for child care programs also
advised spacing nap mats out and placing children head to toe. Staff mark each child's head end with its pillow or
blanket before the children lie down, and check the spacing with a tape or by eye.

Here the room is at 1:8 (114 mm is 3 feet) and the table top is its floor. A green rest rug, 240 x 480 mm, lies in
front of the arms; on it stand three blue cots, 165 x 69 mm, side by side with their long sides pointing away from the
robot and only 51 mm of rug between neighbours, and on each cot a white pillow, all three at the cots' far ends. The
robot has to spread the cots so that neighbours have at least 114 mm of floor between them, every cot still flat on
the rug and pointing the same way, and set the pillows head to toe: one pillow on each cot at one of its ends, no two
neighbours with their pillows at the same end. Three 69 mm cots with two 114 mm gaps fill 435 mm of the rug's 480 mm,
so the outer cots have to go close to the rug's long edges; and the alternation is read along the whole row, so moving
the middle pillow alone fixes both pairs. The score is 0.25 for each neighbouring pair with the gap and 0.25 for each
pair head to toe, read from how things lie at the end, and 0 if a cot is tipped, turned, stacked or off the rug, or a
pillow is not on a cot. The natural split is one side each: each arm spreads the cot on its own side.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three cots' and three pillows' states, with a small tracker for stillness |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets, broken states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (a fixed box and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the nap room ready. Spread the three cots so that there are at least 114 mm (3 feet at this 1:8 scale) of floor between neighbouring cots, keeping each one flat on the green rug and pointing the way it does now, and set the pillows head to toe: one pillow on each cot, at one end of it, and no two neighbouring cots with their pillows at the same end." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nap-cots-head-to-toe/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the nap room ready. Spread the three cots so that there are at least 114 mm (3 feet at this 1:8 scale) of floor between neighbouring cots, keeping each one flat on the green rug and pointing the way it does now, and set the pillows head to toe: one pillow on each cot, at one end of it, and no two neighbouring cots with their pillows at the same end." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/nap-cots-head-to-toe/dream
```

Run one trial per process (the default). The score depends only on where the cots and pillows lie; the small tracker
in the task module keeps each object's path over the last 0.5 s (for stillness), and recognises a new episode in the
same process only when the simulator puts every cot and pillow back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided: `zero` (why the score is 0, if it is), `order` (the cots across the rug),
`pairs` (for each neighbouring pair `gap_mm`, `gap_ok`, the two `ends` and `opposite`), for each cot `set_down`,
`at_rest`, `flat`, `tilt_deg`, `heading_err_deg`, `on_rug`, `rug_margin_mm`, `low_mm`, `centre_mm` and its `pillows`,
for each pillow `on_cot`, `at_rest`, `end`, `along_mm`, `low_gap_mm` and `centre_mm`, and the raw poses. Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the six loose objects' poses, once per
control step (20 Hz), and scores the state at hand: the score is read at the last step. The rug is fixed, so its edges
and top come from `layout.py`. The cots are alike and so are the pillows: neighbours are read from the cots' order
across the rug (by y) and any pillow may end on any cot, which is what a photo of the room shows.

A thing is still when neither its centre nor the centre of one of its end faces moved more than 2 mm over the last
0.5 s (positions, not the simulator's reported velocity, which chatters while a part is gripped). A cot is set down
when it is still, lies flat on the rug (its lowest point, worked out from its pose, within 3 mm of the rug's top and
its broad faces within 5 degrees of level, either face up), its long side is within 15 degrees of x seen from above
(either way round: a cot turned end for end is the same cot) and its whole outline seen from above (the shadow of its
real, turned shape) is on the rug. A pillow is on a cot when it is still, its centre seen from above is over that
cot's outline and its lowest point is within 3 mm of the cot's top face under it, whichever face is down. It marks the
cot's near end when its centre is within 50 mm of the near end, measured along the cot, the far end when within 50 mm
of the far end, and no end in the middle 65 mm. The gap between two cots is the smallest distance between their
outlines seen from above (the narrowest strip of floor, which a tape finds), so a turned cot is judged by its nearest
corner.

| Outcome | Score |
|---|---|
| Any cot not set down (tipped, on its side, on another cot or a pillow, turned over 15 degrees, partly off the rug, held more than 3 mm up or moving), any pillow not on a cot (on the rug or the table, on another pillow, held more than 3 mm up or moving), or two pillows on one cot | 0 |
| Otherwise, for each of the two neighbouring pairs (left-middle, middle-right) with at least 114 mm of floor between its cots | 0.25 |
| ... plus, for each pair whose two pillows mark an end and the ends differ (one near, one far) | 0.25 |

So the score is 0.25 x (pairs spread + pairs head to toe), each count 0, 1 or 2: 0, 0.25, 0.5, 0.75 or 1.0, and every
tier is reached by an ordinary partial job. Every outcome, with numbers: the start 0 (51 mm gaps, all pillows at the
far ends); only one outer pillow moved to its cot's near end (one pair alternates) 0.25; only pillows moved so both
pairs alternate (near-far-near or far-near-far) 0.5; one outer cot spread to 114 mm or more from the middle one,
pillows all at the far ends, 0.25; one pair spread and one pair alternating 0.5; both spread and no alternation 0.5; one pair spread and both alternating, or both spread and one pair alternating, 0.75;
both spread and both alternating 1.0 (by moving the two outer pillows, as the demo does, or only the middle one, or by
turning the middle cot end for end with its pillow); a pillow left in the middle of its cot gives neither of its pairs
head-to-toe credit; any zero condition 0. The trial ends early (success) once 1.0 has held for 2.5 s; otherwise it is
scored as it stands at the run limit. A state with a non-finite number, a zero quaternion or a missing object (a
simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: stacking two cots to make room (0); staggering the cots
along x instead of spreading them (the rug leaves only 75 mm to stagger 165 mm cots, so their outlines still face each
other across 51 mm: alternation only, 0.5); turning cots crosswise to fit (0); pushing the outer cots over the rug's
edges for wider gaps (0); throwing a pillow off the table (0); two pillows on one cot (0). A pillow bridging two
touching cots belongs to the cot under its centre. Pillows count whichever face is down (a pillow on its edge at the
near end still marks it). The scorer cannot see the grippers, so a cot or pillow held still within 3 mm of where it
would lie counts as lying there (see `fidelity`); 3.5 mm up it does not.

## One arm alone

The demo splits the work by side: each arm takes the pillow off the cot on its side, carries that cot out toward the
rug's edge and lays the pillow back on its near end; the middle cot and its pillow (far end) are not touched, which
makes near, far, near. With one arm idle the other still does its own side, one pair spread and head to toe: on the
GPU the demo scored 0.5 with the left arm idle and 0.5 with the right arm idle (gaps of 125.6 and 123.7 mm on the
worked side, 51 mm on the other), as the offline kinematic check predicted. Either arm alone can reach 0.75 with
ordinary moves (not run on the GPU): its own cot out (one gap) and the middle cot's pillow, on the midline within
easy reach, moved to that cot's near end (both pairs then alternate).

We do not claim one arm cannot finish. Left where it starts, the far cot leaves no room for the other two (with 114 mm
gaps the near cot would have to stand 4 cm over the rug's edge), so it has to be moved, and straight down an arm
reaches the far cot only near its near end: in our IK model the left arm's tool point, at the height that grips a cot,
reaches y -17 cm at x 20 cm but only -11 cm at x 30 cm and -6 cm at x 36 cm, where the far cot's far end lies. Gripping
the far cot at its near end and sliding it out to about y -16.5 cm, with the middle cot moved 18 mm toward the left
arm and the left cot 4.5 mm from the rug's edge, would leave both gaps at 114 mm (a test checks that this layout
scores 1.0); an arm could also shove the far cot outward with another cot held as a pusher. Both are tight moves at the edge of the
arm's reach, which the scripted demo does not try. The right arm is the mirror image. (The GPU runs are in
'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 46.9 s (938 control steps), both arms on one clock as mirror images, each working only on its
side of the midline, so the two tool points never come closer than 24.0 cm along y (offline: the arms' collision
capsules at least 4.7 cm apart, the closest being the open fingertips as both come down round their cots). Each arm,
at the same time as the other:

1. lifts the pillow off its cot (gripped across its 44 mm width, fingers opened to 70 mm and commanded to 36 mm, the
   pad tips 4 mm above the cot's top), carries it with its underside 83 mm up and sets it on the rug at x 38.3 cm,
   y +-13.0 cm, 8 mm past the cots' far ends, out of the cot's way;
2. grips its cot at the middle across its 69 mm width (opened fully, to 94 mm: 12 mm to spare each side, and the finger
   on the middle cot's side 32 mm clear of it; commanded to 61 mm; the pad tips 7 mm above the rug), lifts it 43 mm, carries it 74 mm outward over 2 s and
   sets it down at y +-19.4 cm: gaps of 125 mm to the middle cot (11 mm to spare) and 11.5 mm inside the rug's edge;
3. picks the pillow up from the rug and lays it on the cot's near end, its centre at x 22.0 cm, 25 mm in from the
   end (25 mm inside the 50 mm band and 25 mm from the end itself);
4. goes up and out to the side.

Every gripper open or close takes 1.5 s, every final descent is slow (the last 30 mm in 1.5 s) and every part is let
go 2 mm above where it comes to rest after a 0.3 s settle. The cots (60 g) and pillows (12 g) barely load the arms.
`COT_TRIM` and `PILLOW_TRIM` in `task.py` move an arm's place points if the GPU run shows its parts landing off by a
steady amount (compare `centre_mm` in the trace with (277.5, +-194.0) for a cot and (220.0, +-194.0) for a pillow);
both are zero. Offline the smallest joint margin is 18.6 degrees (joint 4), the worst IK error 0.1 mm and the pad tips
stay at least 6 mm above the table. A pillow counts as on its cot while still gripped 2 mm above it, so success must
hold for 2.5 s, longer than the settle and release after the last set-down (0.3 + 1.5 s) plus the 0.5 s stillness
window: the trial cannot end with a pillow still in a closed gripper. Offline, success holds at step 928; the plan
runs to step 938 and the scripted policy stops at step 999. On the GPU the demo trial ended in success at step 910,
the cots set down at y +19.4 and -19.4 cm (gaps 123.7 and 125.3 mm, 9.7 and 11.1 mm inside the rug's edges) and the
outer pillows' centres 25 to 26 mm in from their cots' near ends; no trim was needed. The run limit is 2,400 steps (120 s), which leaves
Dream-1 more than twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the three cots back flat on the rug, long
sides along x, centred at x 27.75 cm and y +12.0, 0 and -12.0 cm (51 mm of rug between neighbours), and one pillow
flat on each cot at its far end, centred at x 33.5 cm and the cot's y. The cots are interchangeable, and so are the
pillows; the rug is fixed.

## Runtime per trial

Measured on an L40S, one trial per process, Isaac Sim's start included: about 2.3 minutes for the scripted demo (it
ends after 45.5 s of simulated time, 910 steps), about 2.4 minutes for each one-arm run (999 steps) and about 5.4
minutes for the Dream-1 trial run to the limit (2,400 steps, 120 s of simulated time); the first start of Isaac Sim
on a new instance takes several minutes more.

## Real-world build

A 240 x 480 mm piece of green felt or thin carpet, about 2 mm thick, fixed to the table 28 cm in front of the arm
bases (x 16.0 to 40.0 cm, y -24.0 to +24.0 cm). Three cots: closed boxes of 3 mm plywood, 165 x 69 x 20 mm, painted
blue, about 60 g each. Three pillows: white-painted hardwood blocks 30 x 44 x 12 mm, about 12 g each. Positions in
`task.json`'s `setup`.

Trials are safe: the heaviest part is a 60 g wooden box, nothing is sharp, hot, powered or under load, and nobody
needs to reach into the robot's space until the trial has ended.

Scoring a real trial, from the room as it stands: tape each gap between neighbouring cots at its narrowest point
(114 mm or more), check by eye that no cot's outline crosses a rug edge and that every cot lies flat and points away
from the robot within 15 degrees (a protractor on a photo from above), and read each pillow's end from a photo from
above, or with a tape from the cot's end (its centre within 50 mm of that end).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

With the arms holding still for 160 steps (the settling check) nothing moved: every cot and pillow stayed on its mark
and the score stayed 0, as at the start. The Dream-1 trial ran to the limit; it shifted the left pillow about 12 mm
on its cot and moved no cot, so the room ended as it began, scored 0.
