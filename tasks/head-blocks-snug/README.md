# Set head blocks snug against both sides of a casualty's head without moving it

In spinal motion restriction, once a casualty is on a long spine board or scoop stretcher, a rescuer keeps the
head in line by hand until foam head blocks are placed snug against both sides of it and strapped. The head must
not be pushed sideways, lifted or turned while the blocks go on, and a block left short of the head lets it roll.
Here a yellow spine board is fixed on the table with a tan casualty trainer lying on its back on it: the torso and
neck are fixed to the board, but the head is a separate weighted cylinder (120 mm across, 300 g) resting free on the
board, so any push moves it. Two orange foam head blocks stand on the table beside the board, one in front of each
arm. Each block must end standing upright on the board beside the head, its 70 mm side along the board, with a gap
of 5 mm or less (0.5 for that side; 15 mm or less: 0.25), and be let go; the head must never have moved more than
8 mm from where it started, been lifted or tipped. Each arm sets the block on its own side; in our reach model one
arm alone, gripping from above, can score at most 0.5 (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the head's and blocks' states and the grippers' state, with a small tracker |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, a loose upright cylinder, loose boxes) and the standard
`task_score` scorer.

A block counts only once both grippers have let go of it, and `yam_arena` does not pass the grippers' state to a
task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to
take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each
control step and to start a new trial at each reset. It reads only state the simulator already has and changes
nothing the embodiment does. If it cannot read the arms it prints why once; every block is then taken as let go on
those steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The training casualty lies on a spine board and its head must not move. Stand one orange head block upright on the board at each side of the head, with its long side snug against the head, without pushing the head, then let go of both blocks." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/head-blocks-snug/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The training casualty lies on a spine board and its head must not move. Stand one orange head block upright on the board at each side of the head, with its long side snug against the head, without pushing the head, then let go of both blocks." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/head-blocks-snug/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `left_side`, `right_side`, `head_disturbed`,
`over_head`, `resting`, `head_shift_mm`, `head_lift_mm`, `head_tilt_deg` and their worst values so far
(`worst_shift_mm`, `worst_lift_mm`, `worst_tilt_deg`), and for each block under `blocks`: `side`, `credit`, `snug`,
`beside`, `let_go`, `gap_mm`, `tilt_deg`, `heading_deg`, `rise_mm`, `dx_mm` (its centre's offset from the head's
along x); then `opening`, `arms_unavailable`, `first_disturbed_step`, `first_full_step`, `best_score`, the head's
and blocks' positions and both TCPs). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's
record per process and starts it over at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is the end state, read at the last step, plus
one fact about the whole trial: how far the head has ever been from where it started. The head's start is where its
centre stood at the trial's first step. Seen from above, the head's outline is its 120 mm circle (for a tilted head,
its axis seen from above widened by 60 mm, at most 0.9 mm wider than the true outline at 10 degrees) and a block's
outline is the shadow of its eight corners, so a block that is turned or leaning is measured where it really comes
nearest. A block's gap is the horizontal distance between the two outlines, 0 or less where they touch. A block is
on the side of the head its centre is on; it stands beside the head when both grippers have let go of it, its
80 mm axis is within 10 degrees of vertical (either way up), its 70 mm side is within 15 degrees of the board's long
axis, its lowest point is within 3 mm of the board's top, and its centre is within 25 mm of the head's along x.

| Outcome | Score |
|---|---|
| At any moment the head's centre more than 8 mm from its start seen from above, its lowest point more than 10 mm above the board, or its axis more than 10 degrees from vertical (even if put back) | 0 |
| At the end, any block reaching more than 5 mm inside the head's outline seen from above (on, over or leaning across the head, or held above it) | 0 |
| A block standing beside the head with a gap of 5 mm or less on each side; the head and both blocks still for 1 s (ends the trial) | 1.0 |
| A snug block (5 mm or less) on one side and a block within 15 mm on the other | 0.75 |
| A snug block on one side only, or a block within 15 mm on each side | 0.5 |
| A block within 15 mm on one side only | 0.25 |
| Anything else (no block beside the head, blocks too far off, lying down, turned, off the board, still held) | 0 |

Each side scores its best block, and the score is the sum of the two sides, so every tier is reachable: 0.25 with
one block within 15 mm, 0.5 with one snug block or with a block within 15 mm on each side, 0.75 with one snug and
one within 15 mm, 1.0 with both snug. The 5 mm allowance inside the outline is for contact: a block pressed against
the head, or squeezing it from both sides without moving it, is snug (gap 0), not on the head.

Unintended strategies earn nothing more than they should, as the tests show: both blocks on the same side count as
one side (0.5); a block lying on its side, standing with its narrow 50 mm side toward the head, beside the neck or
the crown rather than the head, or on the table beside the board does not stand beside the head (0 for its side);
setting a block down short and shoving the head more than 8 mm over to it, or pushing the head away and back again,
moves the head (0 for the whole trial; a shift of 8 mm or less is the stated allowance for contact and for reading
the tape marks); a block left on, over or held above the head scores 0 for the whole trial; a block still
held by either gripper does not count (holding both in place is 0); a block held in the air beside the head is not
on the board.

A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts the head and both
blocks back on their starting marks (within 4.5 cm, which covers a seeded reset's jitter) in a single step; because
the head's start is read at the trial's first step, a jittered start is measured from where the head really began.
A simulator state with a non-finite number scores 0 for that step and leaves the scorer's record as it was.

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone sets its block against the left side of the head) and `DEMO_IDLE_ARM=left`
(the right arm alone does the same on the right) each score 0.5, both in the offline kinematic check and in the GPU
runs (see 'Results in simulation': the idle arm's block stays on the table, the head does not move), and in our reach
model 0.5 is the most one arm can do. Each arm can set its own block on either side of the head (yamkin: the left arm
reaches the right-side place pose, x 20 cm, y -9.7 cm, at the same heading), but the other block is out of its reach:
wherever its open fingers would overlap the far block's x range (16.5 to 23.5 cm) at the block's heights, the left
arm's tool point reaches no further than y -19.5 cm and its open fingers, 55 mm beyond the tool point along their
closing axis, no further than y -24.9 cm, 6 mm short of the far block's near face at y -25.5 cm (headings -90 to +90
degrees in 15-degree steps, tool point 13 to 90 mm above the table; the right arm is the mirror image). So, gripping
from above as the simulator's arms do, 1.0 needs both arms, each setting the block on its own side. The margin is
small (6 mm in this model), and a real arm that tilts its gripper might reach a little further; two arms are the
natural way to do the task, not a rule of it.

## Demo and run limit

The scripted demo takes 31.5 s (630 control steps). The left arm comes over its block, down round it (the last
30 mm over 1.5 s), and grips it across its 70 mm length (fingers closing along x on its front and back faces,
commanded 8 mm narrower, over 1.5 s) with the pad tips 25 mm below its top and 10 mm off its centre away from the
head, so no finger is ever between the block and the head and the fingers stay at least 12 mm from the head
(yamkin's capsules). It lifts the block until its bottom is 8 mm above the board's top, carries it 18 cm sideways
over 3 s to 12 mm from the head, closes the last 10 mm over 2 s, lowers it 7 mm straight down over 1.5 s with its
bottom aimed 1 mm above the board and its inner face 2 mm from the head, opens over 1.5 s, goes straight up to 16 cm
and moves out to y +25 cm. The right arm waits above its own block, picks it the same way, and brings it in to the
right side of the head once the left gripper is up and moving away; the two grippers are never within 20 cm of each
other in y while both are low. An overshoot of a few millimetres touches the head and moves it a little (8 mm
allowed; a gap of 0 is snug); a shortfall of up to 3 mm is still snug. The arm carries the 40 g block with well
under 1 N of load. Success has held for its 1 s at step 599, both offline and in the GPU run; the scripted policy
stops at step 691. The run limit is 1,200 steps (60 s).

To retune on the GPU, read each block's `gap_mm` (2 aimed, 3.7 and 3.9 in the GPU run; 5 or less is snug), `tilt_deg` and
`heading_deg` (near 0), `rise_mm` (about 0), `dx_mm` (about 0) and `let_go`, and the head's `worst_shift_mm` (a few
millimetres at most if a block touches it; 8 is the limit) in `TASK_TRACE`, and adjust `PLACE_GAP`, `CARRY_Z`,
`PLACE_Z` or `GRIP_OFF` in `task.py`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the head back on its tape marks on the
board, centred at x 20.0 cm, y 0, axis vertical; stand the two blocks upright on their marks on the table, 70 mm
sides along x, centred at x 20.0 cm, y +28.0 cm and y -28.0 cm. The board, torso and neck stay fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 31.5 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 480 x 300 mm sheet of yellow 15 mm HDPE clamped to the bench as the spine board; a tan-painted wooden torso block
(300 x 200 x 50 mm) and neck block (30 x 60 x 30 mm) screwed to it on its centre line, the neck's near face 150 mm
from the board's near end; and the head, a PLA cylinder 120 mm across and 100 mm tall, printed hollow and weighted
with steel shot to 300 g, standing on its flat end with its centre 80 mm from the board's near end, 10 mm short of
the neck. Two head blocks of dense closed-cell foam (about 140 kg/m3) in an orange vinyl cover, 70 x 50 x 80 mm,
about 40 g. Tape marks on the board around the head's start; marks on the table at the blocks' starts. Score from
the video and the tape marks (did the head move), a 5 mm and a 15 mm feeler gauge or a ruler (each gap), a
protractor (the angles), and the video (both grippers let go).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

In the demo run both blocks ended upright on the board, 3.7 mm (left) and 3.9 mm (right) from the head, and the
head never moved (worst shift 0 mm); 1.0 held for 1 s at step 599. In each one-arm run the working arm's block ended
snug (3.7 or 4.0 mm) and the idle arm's block stayed on the table. In the Dream-1 run neither block reached the
board (the right one was lifted briefly and ended 12 mm from its mark on the table) and the head did not move.
