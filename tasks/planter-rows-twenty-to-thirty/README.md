# Reset a six-row 20-inch planter to four rows at 30 inches, centred on the hitch

A row-crop planter's row units clamp along its toolbar, and the rows have to match the combine head or sprayer that
will work the field after it. So between seasons farm equipment mechanics and operators (O*NET 49-3041, 45-2091)
re-space the units: they unclamp them, move them along the bar and set each one with a tape measured from the
toolbar's centre (the hitch), so the rows sit symmetric about the tractor, and units that are not needed come off
onto stands. Six rows at 20 in and four rows at 30 in both span 120 in, so the same toolbar carries either. With an
even number of rows no unit sits on the centre line: the two middle rows are half a row spacing either side of it.

Here the planter is at 1:4. A fixed dark-grey toolbar 700 mm long runs across the table in front of both arms, with
a red centre mark 10 mm wide at its middle and a hitch tongue in front of it; six green row units stand square on
it as six rows 20 in (127 mm) apart, three each side of the mark; a grey spare-unit stand sits on the table in front
of the bar on each side. The robot has to reset the planter to four rows at 30 in (190.5 mm): two units each side,
95.25 and 285.75 mm from the mark (half and one and a half spacings), none on the mark, and the two spares on their
stands. Neither new position is marked: they follow from the spacing, the row count and the mark. On each side the
inner unit moves out 31.75 mm, the outer one moves in 31.75 mm and the middle one comes off. The score is read from
where the units are at the end, as a mechanic checks the set-up with a tape from the mark. Each side of the mark is
scored on its own (0.5 each), and each arm naturally resets its own side; one arm alone scores 0.5 (see "One arm
alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the six units' states, with a small tracker for stillness and the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets, broken states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Reset this planter from six rows at 20 inches to four rows at 30 inches. Move the green row units along the toolbar so that four of them stand upright on it, square to it and centred across its width, two on each side of the red centre mark, symmetric about the mark and 30 inches (190.5 mm at this 1:4 scale) apart centre to centre, with no unit on the mark. Stand the two spare units upright, one on each grey stand." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/planter-rows-twenty-to-thirty/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Reset this planter from six rows at 20 inches to four rows at 30 inches. Move the green row units along the toolbar so that four of them stand upright on it, square to it and centred across its width, two on each side of the red centre mark, symmetric about the mark and 30 inches (190.5 mm at this 1:4 scale) apart centre to centre, with no unit on the mark. Stand the two spare units upright, one on each grey stand." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/planter-rows-twenty-to-thirty/dream
```

Run one trial per process (the default). The score depends only on where the units are; the small tracker in the
task module keeps each unit's path over the last 0.5 s (for stillness, which only the early end of the trial needs)
and diagnostics for the trace, and recognises a new episode in the same process only when the simulator puts every
unit back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`left_side` and `right_side` scores; for each side the units
on the toolbar, `all_set`, each row's distance from the mark and how far it is off its target, `stand_filled`;
`covering_mark`, `off_table`, `all_still`, the first step each score was reached; and each unit's centre, distance
from the mark, tilt, squareness, bottom above the toolbar's top, path over the last 0.5 s, `on_toolbar` and `set`).
Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the six units' poses, once per
control step (20 Hz), and scores the state at hand: the score is read at the last step. The toolbar and stands are
fixed, so their faces come from `layout.py`. Distances are measured along the bar (y) from the centre of the red
mark, to each unit's centre.

A unit is on the toolbar when its centre, seen from above, is over the toolbar's top and its lowest point (worked
out from its pose) is within 3 mm of that top, in any attitude: set square, turned across the bar or tipped over.
A unit's tilt is the angle of its 45 mm height from vertical (either way up; its top and bottom are alike). It is
set when it is on the toolbar, tilted at most 5 degrees, its long side within 8 degrees of x (square to the bar, either end forward) and its centre within 12 mm of the bar's
centre line. It stands on a stand when its centre is over that stand's 90 x 90 mm top, its lowest point is within
3 mm of that top and it is tilted at most 5 degrees. A unit belongs to the side of the mark its centre is on.

| Outcome | Score |
|---|---|
| Any unit off the table, or any unit on the toolbar with its centre within 23 mm of the mark's centre (it covers part of the mark: a row on the centre line) | 0 in all |
| Otherwise, for each side of the mark: exactly two units on that side of the toolbar, both set, at 95.25 and 285.75 mm from the mark within 10 mm, and a unit standing on that side's stand | 0.5 for that side |
| Exactly two units on that side of the toolbar, both set, within 25 mm of those distances, and one of the two 0.5 conditions met: both within 10 mm (its stand empty), or a unit standing on its stand (a row 10 to 25 mm off) | 0.25 for that side |
| Anything else on that side (three units still on it, one or none, a unit turned or tipped, a unit more than 25 mm off, or a row 10 to 25 mm off with the stand empty) | 0 for that side |

The score is the sum of the two sides. Every outcome, with numbers: the start (three units each side) 0; only the
spares taken off, the four left at 63.5 and 317.5 mm (31.75 mm off each target) 0; four kept at the old 20 in
positions 63.5 and 190.5 mm 0; three rows at 30 in (0 and +-190.5 mm: a unit on the mark) 0; one side done and the
other as found 0.5; one side at 0.25 (its rows done but its spare not on its stand, or its spare on its stand and a
row 10 to 25 mm off) and the other as found 0.25; one side done and the other at 0.25, 0.75; both sides at 0.25,
0.5; both done 1.0. So every score 0, 0.25, 0.5, 0.75 and 1.0 can be reached. A quarter always needs one of the
side's two jobs done in full (rows within 10 mm, or the spare on its stand): rows nudged part of the way (inner and
outer units moved 7 mm toward their targets) with the spare knocked off the bar earn nothing. The 25 mm band of 0.25 is narrower than the 31.75 mm
between the old and new positions, so a unit left where it was never counts. Any unit may take any place (they are
identical). The trial ends early (success) once 1.0 has held with every unit still (its centre moved under 5 mm in
all over the last 0.5 s) for 3 s, longer than the settle and release after the last set-down (0.5 + 1.5 s), so it cannot end while a unit is still in a closed gripper; otherwise it is scored as it
stands at the run limit. Stillness is read from each unit's path, not from the simulator's reported velocity, which
chatters while a unit is gripped. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores
0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: sweeping the spares off the table instead of standing
them on their stands (0); knocking a spare off the bar and nudging the rows 7 mm (0 for that side); stacking a spare on top of a row unit, or leaving it on the hitch tongue or the table (that
side 0.25, not 0.5); laying a spare on its side on its stand (not standing: 0.25); standing four units on the table
in front of the bar at the right distances (none on the toolbar: 0); setting the rows turned across the bar (not
square: 0); holding units 5 mm above their places (not on the toolbar: 0); carrying a unit across the mark to the
other side (that side then has three: 0); both spares on one stand (the other side 0.25: 0.75 at best). The scorer
cannot see the grippers, so a unit held still with its bottom within 3 mm of the bar counts as set (see
`fidelity`); held 3.5 mm up it does not.

## One arm alone

The demo splits the work by side: each arm resets the three units on its own side of the mark and uses its own
stand. With one arm idle the other still resets its own side, so one arm alone scores 0.5: in simulation the
scripted demo scored 0.5 with the left arm idle (right side 0.5, left side 0 with its three units untouched) and 0.5
with the right arm idle (the mirror image), as the offline kinematic check predicted. Gripping from above, an arm does not reach the far side's work: along the
toolbar's centre line (x 30 cm) the right arm's straight-down reach ends at y +99 mm at grip height (+80 mm with a
unit lifted off the bar, +32 mm at travel height), so it reaches the left inner unit (63.5 mm) and its target
(95.25 mm) but not the left middle unit (190.5 mm), the left outer unit (317.5 mm) or its target (285.75 mm); and at
the stands' line (x 17 cm) it reaches y +176 mm at set-down height, short of the left stand's near edge at +205 mm
(by our IK model of the arm; the left arm is its mirror image). Taking the far inner unit off, or moving it to its
target, leaves that side with units at 190.5 and 317.5 mm, or three units on it: 0 either way. We do not claim one
arm is strictly unable to finish: an arm tipped well over sideways reaches further than straight down, so a policy
able to push units along the bar from the side might reach more. The scripted demo grips from above only.

## Demo and run limit

The scripted demo takes 42.2 s (844 control steps), both arms on one clock as mirror images. Each unit is gripped
across its 36 mm width (yaw 0: the fingers close along the bar, the long side stays along x), opened fully (94 mm:
29 mm either side of the unit, while its neighbours stand 91 mm away at the start) and commanded to 28 mm to close
(about 20 N per finger), with the tool point 28 mm above the unit's bottom so the pad tips stay 15 mm above the
bar's top. Every gripper open or close takes 1.5 s, every final descent onto a unit, the bar or a stand is slow, and
each unit is let go 1 mm above what it is set on after a 0.5 s settle. In order:

1. Each arm lifts its middle unit (y +-190.5 mm) straight up to travel height (its bottom 112 mm up, 37 mm over the
   other units' tops), carries it over its own stand and sets it down there.
2. The left arm lifts its outer unit 31 mm off the bar, moves it in from 317.5 to 285.75 mm and sets it down; at
   the same time the right arm moves its inner unit out from 63.5 to 95.25 mm.
3. The left arm moves its inner unit out from 63.5 to 95.25 mm; the right arm moves its outer unit in from 317.5 to
   285.75 mm.

So in steps 2 and 3 one gripper works near the bar's end while the other works near the mark: the TCPs stay at least
38.1 cm apart and the arms' collision capsules at least 18.6 cm (offline). The smallest joint margins are 14.0
degrees (joint 4, both arms); the pad tips stay at least 26 mm above the table (16 mm above a stand's top). An 80 g
unit held at its middle makes the arm sag about 1 mm, well inside the 10 mm tolerance; `PLACE_TRIM` in `task.py`
moves an arm's place points if the GPU run shows them landing off by a steady amount. In the offline proxy success
is held at step 838; the scripted policy stops at step 905. The run limit is 2,400 steps (120 s), which leaves
Dream-1 almost three times the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the six units back upright and square
on the toolbar's top, long side along x, centred on its centre line (x 30.0 cm), at y +6.35, +19.05, +31.75, -6.35,
-19.05 and -31.75 cm; the toolbar, hitch tongue and stands are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 42 s of simulated time, when
success has held for 3 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 700 x 40 x 30 mm steel bar painted dark grey (painted hardwood also works), fixed to the table (screws, clamps or strong
double-sided tape) 30 cm in front of the arm bases, with a 10 mm red band painted across its top and faces at its
middle; a 100 x 30 x 25 mm dark-grey steel block on the midline butting the bar's face nearest the arms as the hitch tongue; two
90 x 90 x 10 mm light-grey plastic plates fixed 17 cm in front of the arm bases, 25 cm either side of the midline.
Six 70 x 36 x 45 mm wooden blocks painted green (about 80 g each), with a thin line drawn across the top of each at
its centre to read the tape against. Positions in `task.json`'s `setup`.

Scoring a real trial: hook a tape on the centre of the red band and read each unit's centre line along the bar
(85.25 to 105.25 mm and 275.75 to 295.75 mm for full marks on that side; within 25 mm counts for a quarter only
with the spare on its stand); check each
unit square to the bar with a try square against the bar's face (within 8 degrees: its far end within 10 mm of the
square over its 70 mm length) and sitting flat on the bar; check that no unit on the bar has its centre within 23 mm of the red band's
centre (a square unit there covers part of the band) and that one spare stands upright on each stand.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The demo first scored 1.0 at step 771 and ended early at step 840, once 1.0 had held still for 3 s, with all four
rows within 1.2 mm of their targets and both spares standing on their stands. With either arm idle, the working arm's side
scored 0.5 and the idle side 0, its three units still at the old 20 in positions. Dream-1 ran the full 2,400 steps
and ended with three units still on each side of the toolbar (it moved the left outer unit about 12 mm inward): 0.
Settling with no plan (8 s) moved nothing.
