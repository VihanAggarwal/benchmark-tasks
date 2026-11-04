# Seat a heatsink on a processor straight down, without sliding it across the paste

In server and GPU board assembly, a heatsink is lowered vertically onto freshly applied thermal paste: sliding
or twisting it across the die once it touches smears the paste and leaves voids that overheat the chip, and
dropping it onto a bare die can chip it. Here a green circuit board with a dark grey processor die and four
black retention posts is fixed on the table, and an aluminium heatsink block stands beside it. The heatsink
must end flat, square and centred on the die; the scorer also watches how it got there: how far it slid or
turned while touching the die, and how fast it came down just before it touched. Aligning in the air and then
setting down slowly and straight is the whole skill. This is a single-arm task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the heatsink's state, with a small tracker for the whole trial |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes, fixed cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm; the right arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Place the aluminium heatsink on the processor inside the four posts. Lower it straight down and do not slide it once it touches." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/heatsink-straight-down/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Place the aluminium heatsink on the processor inside the four posts. Lower it straight down and do not slide it once it touches." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/heatsink-straight-down/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`touching`, `bottom_gap_mm`, `centre_off_mm`,
`yaw_err_deg`, `tilt_deg`, `smear_mm`, `setdown_mps`, `on_post`, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the heatsink's pose and velocities,
once per control step (20 Hz), and keeps three facts about the trial: whether the heatsink has touched the die
yet, how far it has slid over the die while touching (its centre's horizontal travel plus any turn about the
vertical, counted at the die's corners: 21 mm times the angle), and its fastest downward speed in the last 10 mm
each time it came down onto the die (every approach, not only the first). "Touching" means its lowest point is
within 2 mm of the die's top while its outline seen from above (the shadow of its real, turned and tilted shape,
not its bounding box) overlaps the die. Smear is followed in steps of 0.2 mm, as a ruler would read it: a move
counts once the heatsink is 0.2 mm from where smear was last counted, and any remainder counts when it stops
touching, so contact jitter that stays within 0.2 mm does not add up over many steps (summing every step's travel
would turn a 0.05 mm shiver into millimetres) while a slow creep still counts in full. The block's two 56 x 56 mm
faces are alike, so either may face down: "tilted" is measured from the nearer one. A new episode is recognised
when the simulator puts the heatsink back on its starting mark (within 4.5 cm of it, which covers a seeded reset's
jitter) in a single step. A simulator state with a non-finite number scores 0 for that step and leaves the
scorer's record of the trial as it was.

| Outcome | Score |
|---|---|
| At the end: centre not over the board, tilted over 30 degrees, or resting on a post | 0 |
| Flat on the die (under 2 degrees), centred within 2 mm, square within 3 degrees, smear 1 mm or less, set-down 0.10 m/s or less; still for 2.5 s | 1.0 |
| The same placement, smear 5 mm or less, but smear over 1 mm or set-down over 0.10 m/s | 0.75 |
| Flat on the die, centred within 5 mm, square within 10 degrees, any smear and speed | 0.5 |
| Touching the die but not meeting 0.5 (tilted, or further off) | 0.25 |
| Anything else (not placed, held above the die, left on the board beside it) | 0 |

Unintended strategies earn less, as the tests show: putting it down roughly and sliding it into place counts
the slide (more than 5 mm: 0.5); twisting it square on the die counts as smear (8 degrees: 0.75; 20 degrees:
0.5); bobbing in and out of the 2 mm band while sliding still counts every step that begins or ends touching;
pushing it along the board and up onto the die counts the whole push (0.5); dropping it from a height is caught
by its speed in the window, and a drop that lands within one control step (from more than about 5 mm) by how far
it fell in that step (0.75 at best); touching the die gently first and then lifting it and dropping it, or rushing
back into the last 10 mm, is caught the same way, because every approach is watched (0.75). At 20 Hz a release
from less than about 5 mm above the die is not reliably told apart from a slow set-down, and the scorer cannot
see the gripper, so a heatsink held still within 2 mm of the die counts as set down (see `fidelity`).

## One arm alone

This task needs one arm, and the demo uses one: the left arm picks the heatsink, carries it over the die and
seats it, while the right arm holds its ready pose. `DEMO_IDLE_ARM=right` is therefore the same run as the demo
(1.0); `DEMO_IDLE_ARM=left` leaves nothing to move (0). The right arm cannot reach the heatsink's starting place,
22 cm left of the midline. No claim is made that the task needs both arms: its difficulty is the contact phase.
(The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 15.0 s (300 control steps): the left arm grips the heatsink across its 56 mm width
(fingers commanded 8 mm narrower, over 1.5 s), lifts it 10 cm (the gripper 12 cm above the table), carries it
over the die, comes down to 2 cm above it, settles for 0.8 s, lowers the last 2 cm straight down over 2.5 s (peak
12 mm/s), opens over 1.5 s with the heatsink's underside aimed 1 mm above the die and goes straight up. The arm's
controller does not carry the heatsink's weight, so the arm sags about 5.6 mm and 1.9 mm sideways each way under
it: the heatsink comes to rest on the die about 0.8 s before the slow descent ends, and the place point is offset
by 1 mm in x and -1 mm in y, half the sideways sag (details in `task.py`). The 2.5 s stillness (rather than the
usual 1 s) is longer than the time from that first rest until the fingers are fully open, so a trial cannot end
while the gripper still holds the heatsink: it ends at about step 284 with the fingers open, and offline (no sag)
success holds for its 2.5 s at step 312. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the heatsink back on its mark at
x 22.0 cm, y +22.0 cm, sides parallel to x and y; the board, die and posts are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 15 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 200 x 160 mm offcut of circuit board (or a 4 mm green plate) clamped to the bench, a 30 x 30 x 3 mm block glued
at its centre as the die, four 8 mm standoffs 14 mm tall at the corners of a 68 mm square around it, and a
56 x 56 x 32 mm flat-faced aluminium block of about 200 g: a solid one would be 270 g, so use a closed hollow
one (walls about 8 mm thick give 200 g). Put a dab of thermal paste (or toothpaste) on the die before each trial: lifting the heatsink off
afterwards shows any slide or twist in the print.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |
