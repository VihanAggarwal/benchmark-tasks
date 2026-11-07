# Set each vine row's end anchor in line, as far out as the row's end post is tall

Vineyard and orchard crews (farmworkers, O*NET 45-2092) build and repair trellis end assemblies. A row's end post
carries the pull of all the row's wires, so it is held back by a brace wire running from its top down to an anchor set
in the ground in line with the row, beyond the end post. A common rule of thumb in trellis guides is to set the
anchor as far from the end post as the post stands above ground, so the brace runs at about 45 degrees: set closer,
the steep brace has to pull much harder and drives the post down; set farther, the anchor takes headland the tractor
needs to turn. Rows on different trellis systems have end posts of different heights, so each row's distance is read
from its own end post.

Here two short rows at about 1:12 stand side by side on the table, parallel, running forward and to the right at
15.1 degrees to x. Each row is a tall end post nearest the arms (100 mm on the left row, 150 mm on the right), a
line post 20 mm lower (80 / 130 mm) 130.5 mm beyond it along the row, and a trellis wire between them. The end post
rises above the line posts as a real one does to carry the brace; the posts are 25 mm square, thicker than scale, so
they show in the top photo. Two grey anchor blocks stand on the
table, one in front of each arm. The robot has to stand one block on each row's line, extended past the end post on
the side away from the line post, with its centre 100 mm (left row) or 150 mm (right row) from the end post's centre.
Nothing is marked: the line has to be extended from the row's two posts, and the distance read from that row's own end
post, not from the shorter line posts or the post spacing. Because the rows are at 15 degrees, setting a block
straight back along x misses the left line by 26 mm and the right by 39 mm. The score is read from where the blocks
are at the end, as a crew checks with a straightedge and a tape. Each row is scored on its own (0.5 each), and each
arm naturally sets the row on its own side; in simulation the scripted demo with either arm idle scored 0.5 (see
"One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, and each row's line and anchor point |
| `scoring.py` | The scorer: plain functions of the two blocks' states, with a small tracker for stillness and the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the wrong rules and exploits we tried, resets, broken states, the demo plan and the one-arm reach claim (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes, some turned about the vertical) and the standard
`task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the brace anchor for each of the two vine rows. Each row has a tall end post nearest you and a shorter line post beyond it, and the row's line runs through the centres of its two posts. Stand one grey anchor block upright on each row's line, past its end post on the side away from the line post, with the block's centre as far from the end post's centre as that end post is tall. The two end posts are different heights." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/vine-row-end-anchor/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the brace anchor for each of the two vine rows. Each row has a tall end post nearest you and a shorter line post beyond it, and the row's line runs through the centres of its two posts. Stand one grey anchor block upright on each row's line, past its end post on the side away from the line post, with the block's centre as far from the end post's centre as that end post is tall. The two end posts are different heights." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/vine-row-end-anchor/dream
```

Run one trial per process (the default). The score depends only on where the blocks are; the small tracker in the
task module keeps each block's path over the last 0.5 s (for stillness, which only the early end of the trial needs)
and diagnostics for the trace, and recognises a new episode in the same process only when the simulator puts both
blocks back on their starting spots in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`left_row` and `right_row` scores; for each row the block
matched to it, `set`, `offset_mm`, `reach_mm` and `reach_err_mm`; `off_table`, `all_still`, the first step each score
was reached; and each block's centre, tilt, bottom above the table, path over the last 0.5 s, `set`, and its offset
from and reach along both rows' lines). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the two blocks' poses, once per
control step (20 Hz), and scores the state at hand: the score is read at the last step. The posts are fixed, so each
row's line (through the centres of its two posts) and its end post's height come from `layout.py`.

A block is set when it stands on the table on one of its 30 x 30 mm faces: its lowest point (worked out from its
pose) within 3 mm of the table top and its 40 mm axis tilted at most 10 degrees from vertical, either end up. The
block is square, so its heading does not matter. For a row, a block's offset is the distance from its centre to the
row's line, and its reach is how far its centre lies beyond the end post's centre along that line, on the side away
from the line post (negative beside or behind the end post). H is that row's end-post height: 100 mm on the left,
150 mm on the right. The two blocks are identical; they are matched one to one to the rows the way that scores more,
so either may serve either row and neither counts twice.

| Outcome | Score |
|---|---|
| Either block off the table (its lowest point more than 10 mm below the table top) | 0 in all |
| Otherwise, for each row: a set block with offset 10 mm or less and reach within 10 mm of H | 0.5 for that row |
| A set block with offset 20 mm or less and reach within 15 mm of H | 0.25 for that row |
| Anything else (no block there; lying on its side, on the other block, held above the table; between the posts, past the line post; farther off) | 0 for that row |

The score is the sum of the two rows: 0, 0.25, 0.5, 0.75 or 1.0. Every outcome, with numbers: the start 0; both
rows right 1.0; one right and the other as found 0.5; one right and the other within the looser band 0.75; both
within the looser band 0.5; one within it and the other as found 0.25. Each wrong rule scores 0 for its row, at
least 4.5 mm outside the looser band: blocks set straight back along x from the end posts instead of along the rows
(26.1 mm off the left line, 39.1 mm off the right); the heights swapped (50 mm off on each row); the line
posts' heights (80 and 130 mm: 20 mm short on each row); the 130.5 mm post spacing (30.5 mm too far on the left,
19.5 mm short on the right). One distance for both rows: 100 or 150 mm gets one row right (0.5); their 125 mm average
is 25 mm off both (0). The looser band is narrower in reach (15 mm) than in offset (20 mm) on purpose, so that every
one of these wrong distances, the nearest of which is 19.5 mm off, scores nothing. The trial ends early (success)
once 1.0 has held with both blocks still (centre moved under 5 mm in all over the last 0.5 s) for 3 s, longer than
the settle and release after each set-down (0.5 + 1.5 s), so it cannot end while a block is still in a closed
gripper; otherwise it is scored as it stands at the run limit. Stillness is read from each block's path, not from the
simulator's reported velocity, which chatters while a block is gripped. A state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: stacking one block on the other at an anchor point (only
the bottom one is on the table: 0.5 at most); laying the blocks on their sides at the points (not set: 0); holding
them 4 mm above the points (not on the table: 0); sweeping a block off the table (0 in all); leaning a block against
an end post (tilted: 0 for that row); putting both blocks at one row's point (a block counts for one row: 0.5). The
posts are fixed, so the rows cannot be moved to suit the blocks. The scorer cannot see the grippers, so a block held
still with its lowest point within 3 mm of the table counts as set (see `fidelity`).

## One arm alone

The demo splits the work by row: the left arm sets the left row's anchor with the block in front of it, and the
right arm the right row's. With one arm idle the other still sets its own row: on the GPU the scripted demo scored
0.5 with the left arm idle and 0.5 with the right arm idle (the offline kinematic check predicted the same). Gripping
from above, an arm does not reach the other arm's
block: at the blocks' line (x 15 cm) the left arm's straight-down reach at grip height ends near y -0.18 m, short of
the right block at y -0.33 m (the right arm is its mirror image; by our IK model of the arm, checked in
`tests/test_scoring.py`). The left arm does reach the right row's anchor point at grip height, but with only its own
block it can still set one row. We do not claim one arm is strictly unable to finish: with the wrist tipped well over
sideways the arm's tool point can reach the far block, so a policy able to grasp from the side (or to push the far
block within reach) might score more. The scripted demo grips from above only.

## Demo and run limit

The scripted demo takes 15.0 s (300 control steps), both arms on one clock. Each block is gripped across its 30 mm
width (yaw 0: the fingers close along y), opened fully (94 mm: 32 mm either side of the block) and commanded to 22 mm
to close (about 20 N per finger), with the tool point 22 mm above the block's bottom so the pad tips stay 9 mm above
the table. Every gripper open or close takes 1.5 s and every final descent is slow (the last 20 mm over 1.2 s). Each
arm: down round its block, grip, lift it straight up to travel height (its bottom 98 mm up), carry it over its row's
anchor point (x 22.3 cm, y +22.6 cm on the left; x 22.5 cm, y -14.1 cm on the right), set it down slowly, let go
1 mm above the table after a 0.5 s settle, and go straight up. The two place points are 36.7 cm apart, the arms'
collision capsules at least 17.2 cm apart (offline), and the open pads over an anchor point stay more than 70 mm from
the nearest post. The smallest joint margins are 18.0 degrees (joint 4, both arms). Under an 80 g block the arm
trails a few millimetres at most (simulated YAM arms trail about 3-7 mm under 1-3 N), inside the 10 mm tolerance;
`PLACE_TRIM` in `task.py` moves an arm's place point if the GPU run shows it landing off by a steady amount (read
`offset_mm` and `reach_mm` in `TASK_TRACE`). In the GPU run the blocks came to rest 0.6 mm (left row) and 0.3 mm
(right row) off their lines, with reaches of 100.5 and 151.1 mm (0.5 and 1.1 mm from H), so `PLACE_TRIM` stays zero;
1.0 was first reached at step 229 and the trial ended in success at step 298 (offline proxy: step 296; the scripted
policy stops at step 361). The run limit is 1,200 steps (60 s), four times the demo's length.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand both blocks back upright on their
starting spots at x 15.0 cm, y +33.0 cm and x 15.0 cm, y -33.0 cm, sides parallel to x and y; the posts and wires are
fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo ends after about 15 s of simulated time, when success
has held for 3 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

Four 25 x 25 mm square wooden dowels or battens stained dark, cut to 100 and 80 mm (left row) and 150 and 130 mm
(right row), glued or screwed upright to the table at the positions in `task.json`'s `setup`, each turned so two of
its sides run along its row (a straightedge against both posts of a row sets this); a length of 2-3 mm galvanised
wire stretched between each row's two posts 70 mm above the table. Two 30 x 30 x 40 mm concrete-grey blocks (cast
mortar, or painted hardwood weighted to about 80 g), each with a cross drawn on its top face through its centre to
read the tape against.

Scoring a real trial: lay a straightedge on the table against the same side face of both of a row's posts; it runs
parallel to the row's line, 12.5 mm from it, and on past the end post. Read the block's centre square off the
straightedge (its offset is that reading minus 12.5 mm, or plus it on the other side) and tape along the straightedge
from the end post's centre to the block's centre: 90 to 110 mm (left row) or 140 to 160 mm (right row) within 10 mm
of the line for full marks on that row; 85 to 115 or 135 to 165 mm within 20 mm for a quarter. Check each block
stands flat on the table on one of its small faces.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
