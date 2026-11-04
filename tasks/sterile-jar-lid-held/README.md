# Add a sample to a sterile jar without ever setting its lid down

Aseptic technique, as practised in microbiology, food-safety and clinical labs: the lid of a sterile container
is held while the container is open, never put down on the bench, and the container is open as briefly as
possible. Here one arm lifts the jar's lid and holds it in the air while the other arm drops a sample cube in;
then the lid goes back on. The rule makes the task two-handed: a single arm cannot hold the lid and move the
cube at the same time, and setting the lid down to free the gripper scores 0.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the lid's and cube's states, with a small tracker for the whole trial |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge case (no simulator needed) |

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
  --instruction "Take the lid off the sterile jar and keep holding it in the air, put the yellow cube into the jar, then put the lid back on. Never set the lid down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sterile-jar/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the lid off the sterile jar and keep holding it in the air, put the yellow cube into the jar, then put the lid back on. Never set the lid down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sterile-jar/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the lid's and the cube's poses and
velocities, once per control step (20 Hz), and keeps three facts about the trial: whether the lid was ever set
down while off the jar, whether it was ever lifted clear, and how long the jar has been open in total. A new
episode is recognised when the simulator puts both objects back on their starting marks in a single step.

| Outcome | Score |
|---|---|
| Lid ever came within 30 mm of the table while off the jar (set down, dropped, rested on the cube) | 0 |
| Cube in the jar, lid seated, jar open 20 s or less, all still for 1 s | 1.0 |
| The same, but open longer than 20 s | 0.75 |
| Cube in the jar, lid not seated | 0.5 |
| Lid lifted clear and never set down, cube not in the jar | 0.25 |
| Anything else | 0 |

Unintended strategies earn nothing: knocking the lid off drops it below 30 mm (0); the cube cannot be pushed into
the jar over its 70 mm walls; lifting the lid and putting it back with nothing else done is worth 0.25 at most.

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone lifts and replaces the lid) scores 0.25; `DEMO_IDLE_ARM=left` (the right
arm alone drops the cube onto the closed lid) scores 0. A single arm cannot both hold the lid and move the cube;
setting the lid down to free its gripper scores 0. (The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 19.3 s (386 control steps): the left arm lifts the lid straight up and holds it in the
air 20 cm to the left; the right arm, which has already picked the cube, drops it into the open jar from 18 mm
above the rim and moves away; the left arm puts the lid back. The jar is open for about 12 s. The run limit is
1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lid back centred on the jar, cube back on
its mark at x 18.0 cm, y -24.0 cm.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 19 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A square polypropylene specimen jar (about 66 mm square, 70 mm tall) clamped to the bench, a flat 74 mm square lid
cut from 18 mm acrylic or printed in PLA (40 g), and a 24 mm plastic cube (12 g). Mark 30 mm on the jar's outside
wall so the lid's clearance can be checked on video.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
