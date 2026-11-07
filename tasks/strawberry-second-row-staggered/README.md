# Hand-plant the second strawberry row, staggered against the machine-set row

Annual strawberries grow on raised beds under black plastic mulch, as a double row either side of the drip tape that
runs along the bed's centre line: the rows about 12 in apart, the plants about 12 in apart in the row, and the two rows
staggered, each plant level with the middle of a gap in the other row, so every plant gets the most room. A two-row
transplanter sets both rows at once. When one of its planting units jams for a stretch, the crew walking behind it
plants that stretch by hand to match the other row (Farmworkers and Laborers, Crop, O*NET 45-2092). Nothing on the bed
marks where the hand-set plants go: along the row they are read from the other row's gaps, and across the bed by
mirroring that row in the drip tape.

Here the bed is at 1:3 (100 mm is about 12 in). A fixed black bed 220 mm across and 600 mm long runs across the
table in front of both arms, with the drip tape as a blue strip along its centre line. The transplanter's far row
stands on it: six green plugs 50 mm beyond the tape, 100 mm apart, with five gaps between them. Three more plugs
stand on a grey plug tray beside each arm's base. The robot has to plant the near row: one plug standing upright
level with the middle of each of the five gaps, and as far on the near side of the tape as the far row is on the
other side, so 50 mm in front of it. That puts the five places at x 25.0 cm, y +20, +10, 0, -10 and -20 cm; none of
them is marked. A near row set in line with the far plants (not staggered), or on the tape, misses every place by
50 mm. The score is 0.2 for each place filled within 15 mm, read from where the plugs stand at the end, and 0 if any
far plant's spot is left without a plug standing on it (one knocked over, shoved, lifted or taken). Each arm
naturally plants from the tray on its own side; one arm alone scores at most 0.6 gripping from above (see "One arm
alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the twelve plugs' states, with a small tracker for stillness and the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets, broken states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and loose cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The transplanter set the far row of strawberry plugs on this bed; plant the near row by hand. Stand one plug from the grey trays upright on the bed level with the middle of each gap in the far row, so the two rows are staggered, as far on the near side of the blue drip tape as the far row is on the other side. Do not disturb the far row." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/strawberry-second-row-staggered/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The transplanter set the far row of strawberry plugs on this bed; plant the near row by hand. Stand one plug from the grey trays upright on the bed level with the middle of each gap in the far row, so the two rows are staggered, as far on the near side of the blue drip tape as the far row is on the other side. Do not disturb the far row." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/strawberry-second-row-staggered/dream
```

Run one trial per process (the default). The score depends only on where the plugs are; the small tracker in the
task module keeps each plug's path over the last 0.5 s (for stillness, which only the early end of the trial needs)
and diagnostics for the trace, and recognises a new episode in the same process only when the simulator puts every
plug back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`filled`, `filled_by` for each place, `nearest_mm` from each place to the
nearest planted plug, `planted` (off the far row's spots), `far_row_kept_by` for each far-row spot, `far_spots_empty`,
`off_table`, `all_still`, the first step each score was reached; for the settle check `far_max_move_mm` and
`far_max_tilt_deg` of the six plugs that started in the far row; and for each plug its centre, tilt from vertical,
bottom above the bed's top, distance from where it started, path over the last 0.5 s and `planted`). Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the twelve plugs' poses, once per
control step (20 Hz), and scores the state at hand: the score is read at the last step. The bed is fixed, so its top
comes from `layout.py`.

The twelve plugs are identical, so the score asks only where plugs stand, never which plug is which: what a photo
of the bed shows. A plug is planted when its centre, seen from above, is over the bed's top, its lowest point (worked
out from its pose) is within 3 mm of that top, and its axis is within 10 degrees of vertical (either end up: the
plug is a plain cylinder). The far row is kept when each of its six spots still has a planted plug with its centre
within 10 mm of the spot, measured flat. A place is filled when a planted plug has its centre within 15 mm of the
place's centre, measured flat. The places are 100 mm apart, so no plug can be within 15 mm of two, and a second plug
crowded onto a filled place adds nothing; the far-row spots are at least 111 mm from every place.

| Outcome | Score |
|---|---|
| A far-row spot left without a planted plug within 10 mm of it (a far plant knocked over, shoved, lifted or taken away), or any plug off the table (its lowest point more than 10 mm below the table top) | 0 |
| Otherwise, for each of the five places with a planted plug within 15 mm of its centre | 0.2 |

So the score is 0, 0.2, 0.4, 0.6, 0.8 or 1.0, and each is reached by filling that many places (six tray plugs for five
places; any plug may fill any place). A plug left on its tray, on the table, lying on the bed, standing on the bed away
from every place and far-row spot, standing on another plug or held above the bed fills nothing and costs nothing. Every
outcome, with numbers: the start 0; a near row set in line with the far plants (y +-5, +-15 and +-25 cm) 0, 50 mm off
every place; a near row on the drip tape (x 30.0 cm) 0, 50 mm off; a staggered row 100 mm in front of the tape (x 20.0
cm) 0; one, two, three or four places filled 0.2, 0.4, 0.6 or 0.8; all five 1.0; any of these with a far-row spot left
empty or a plug off the table 0. The trial ends early (success) once 1.0 has held with every plug still (its centre
moved under 5 mm in all over the last 0.5 s) for 3 s, longer than the settle and release after the last set-down (0.5 +
1.5 s), so it cannot end while a plug is still in a closed gripper; otherwise it is scored as it stands at the run
limit. Stillness is read from each plug's path, not from the simulator's reported velocity, which chatters while a plug
is gripped. A state with a non-finite number, a zero quaternion or a missing plug (a simulator blow-up) scores 0 for
that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: taking plugs from the far row instead of the trays (their
spots are left empty: 0; a far plant moved to a place with a tray plug stood on its spot leaves a bed no photo can tell
from the demo's, and scores the same, since it is the same bed); shoving the whole far row 50 mm along so a row set in
line ends up level with it (0); plugs laid on their sides at the places, leaning 12 or 45 degrees, or stacked two high
(the upper one fills nothing); plugs held 5 mm above their places (nothing filled); the near row stood on the table in
front of the bed (nothing filled); sweeping the spare off the table (0). The scorer cannot see the grippers, so a plug
held still with its bottom within 3 mm of the bed counts as planted (see `fidelity`); held 3.5 mm up it does not.

## One arm alone

The demo splits the work by side: the left arm plants the places at y +20, +10 and 0 cm from the left tray and the
right arm those at -10 and -20 cm from the right tray, leaving the right tray's third plug as the spare. With one arm
idle the other still plants its own places: in simulation the demo scored 0.6 with the right arm idle (the left arm
planted y +20, +10 and 0 cm) and 0.4 with the left arm idle (the right arm planted -10 and -20 cm), as the offline
kinematic check predicted. Either arm alone can fill at most three places gripping from above, 0.6: straight down,
an arm reaches only its own tray (along the trays' centre line, x 20 cm, the left arm's reach ends at y -16.5 cm at
grip height, while the right tray starts at y -33.5 cm), so it has only the three plugs on that tray, and it does not
reach the far end place either (along the near row, x 25 cm, the left arm reaches y -13 cm at set-down height, short
of the -20 cm place). These numbers come from our IK model of the arm; the right arm is the mirror image. We do not
claim one arm is strictly unable to finish: tipped over 40 to 55 degrees, our IK model puts the left arm's tool point
over the right tray's plugs with the arm clear of the table, and tipped 15 degrees over the -20 cm place, so a policy
able to grip and set plugs upright with a steeply tilted gripper might fill more than three. The scripted demo grips
from above only. The far plants are no source of extra plugs: taking one leaves its spot empty (0) unless a tray
plug is stood there instead, which uses up that tray plug.

## Demo and run limit

The scripted demo takes 45.5 s (910 control steps), both arms on one clock. Each plug is gripped across its 30 mm
diameter (yaw 0: the fingers close along y; on a tray the neighbouring plugs stand 60 mm away along x, and on the bed
the planted ones 100 mm away along y), opened fully (94 mm: 32 mm either side of the plug) and commanded to 22 mm to
close, with the tool point 30 mm above the plug's bottom, so the pad tips are 17 mm above it and the plug's top 15 mm
above the tool point. Every gripper open or close takes 1.5 s, every final descent onto a plug or onto the bed is
slow (the last 20 mm in 1.2 to 1.5 s), each plug is carried with its bottom 100 mm up (30 mm over the tops of the
plugs already standing) and let go 1 mm above the bed after a 0.5 s settle. In order:

1. The left arm plants its tray's first plug at y +20 cm; at the same time the right arm plants its first at -10 cm.
2. The left arm plants its second at +10 cm; the right arm its second at -20 cm.
3. The left arm plants its third at y 0; the right arm goes back up over its own tray and waits there.

So whenever both grippers work over the bed they are 30 cm apart along it: the TCPs stay at least 30.0 cm apart and
the arms' collision capsules at least 10.6 cm (offline). The smallest joint margins are 15.3 degrees (joint 4, left
arm); the pad tips stay at least 22 mm above the table (17 mm above a tray's top, 18 mm above the bed's). A 25 g plug
barely loads the arm; `PLACE_TRIM` in `task.py` moves an arm's place points if the GPU run shows its plugs landing off
by a steady amount; none was needed (both are zero). In simulation every plug planted landed within 1.2 mm of its
place, the far row did not move, and success held at step 905, which ended the trial (the offline proxy predicted
904); the scripted plan itself runs to step 971. The run
limit is 2,400 steps (120 s), which leaves Dream-1 more than twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand six plugs back upright on the bed at
x 35.0 cm, y +25, +15, +5, -5, -15 and -25 cm, and three upright on each tray at x 14.0, 20.0 and 26.0 cm, y +36.0 cm
(left) and -36.0 cm (right); the plugs are interchangeable, and the bed, drip tape and trays are fixed.

## Runtime per trial

On an L40S, starting Isaac Sim for each trial: about 2 minutes for the scripted demo (it ends after about 45 s of
simulated time, when success has held for 3 s) and about 5.5 minutes for a Dream-1 trial run to the limit (120 s of
simulated time); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 220 x 600 x 25 mm board covered with black plastic sheet, fixed to the table 30 cm in front of the arm bases, with
a 16 mm blue tape stuck along its centre line for the drip tape; two 160 x 50 x 5 mm light-grey plastic plates fixed
20 cm in front of the arm bases, 36 cm either side of the midline. Twelve 30 mm hardwood dowels 45 mm long painted
green (about 25 g each; plain, the same either end up). Before the first trial, draw a chalk circle 50 mm across round
each far plant's spot (x 35.0 cm, y +25, +15, +5, -5, -15, -25 cm: 160 mm from the bed's near edge and 50, 150, 250,
350, 450 and 550 mm from its left end). Positions in `task.json`'s `setup`.

Scoring a real trial, from the bed as it stands (a photo from above and a tape): find each place with the tape from
the bed's fixed edges (60 mm from its near edge, so 50 mm in front of the drip tape's centre line; 100, 200, 300, 400
and 500 mm from its left end, half-way between two far spots), lay a ring template 60 mm across on it, and count the
place filled if a plug stands upright with its base wholly inside the ring (its centre within 15 mm); check every
chalk circle holds a plug standing upright with its base wholly inside (within 10 mm of the spot), and that no plug
is off the table.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.4 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.6 |
| Dream-1 | 0.0 |

A settle check (`DEMO_HOLD_ONLY=1`, 160 steps) left every plug where it stood: the far row moved 0.0 mm and tilted
0.0 degrees. Dream-1 filled no place and, at about step 1,140, knocked over the far plant at y +25 cm, leaving its
spot empty (0); it ran to the step limit.
