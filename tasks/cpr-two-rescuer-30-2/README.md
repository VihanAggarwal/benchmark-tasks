# Give two-rescuer CPR on a training manikin: 30 compressions, 2 breaths, twice

In two-rescuer CPR one rescuer compresses the chest and the other gives the breaths: 30 compressions 5 to 6 cm deep
at 100 to 120 a minute, letting the chest come all the way back up each time, then 2 breaths of about a second each
while the compressor waits with the hands in position on the chest, and back to compressions within 10 seconds.
Here a CPR training manikin lies on the table: its chest is a plate on a spring with a dark mark at its centre, and a
blue spring-return cap by its head stands in for the bag of a bag-valve-mask. The task is two full cycles. The
scorer watches the whole trial, as a CPR feedback manikin does: the depth, recoil and timing of every compression,
the length of every breath, whether the chest was pressed during a breath, and how long compressions stopped. It
needs both arms: during every breath one gripper must wait over the mark while the other presses the bag 39 cm away,
and with a straight-down gripper the bag is out of the right arm's reach and the left arm's tool point cannot come
within 5 cm of the mark.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the manikin, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the chest's and the cap's travel and the grippers' positions, with a tracker that records every compression and breath |
| `mechkit.py` | Builds the manikin as one fixed articulation with two spring sliders (the chest plate and the cap) and reads their joints for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_press.py` | Offline checks only: a small static model of the finger tips holding the chest and the cap down (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook, the press model, the exploits we tried, and the whole demo through our offline checker (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next
to `task.py` and hook `yam_arena` from the task file's import (the manikin is built as an articulation; both arms are
read before each scoring call); they change nothing the embodiment does, and the task uses the standard
`task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Give two cycles of two-rescuer CPR on the manikin. In each cycle press the dark mark on its chest 30 times, 5 to 6 cm deep, 100 to 120 times a minute, letting the chest come all the way back up each time; then give 2 breaths, each by pushing the blue bag by its head about 3 cm down for about 1 second, while the other gripper waits over the mark without pressing. Keep each pause in compressions under 10 seconds. Then lift both grippers 3 cm clear of the chest and the bag." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir ~/yam-eval/cpr-two-rescuer/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Give two cycles of two-rescuer CPR on the manikin. In each cycle press the dark mark on its chest 30 times, 5 to 6 cm deep, 100 to 120 times a minute, letting the chest come all the way back up each time; then give 2 breaths, each by pushing the blue bag by its head about 3 cm down for about 1 second, while the other gripper waits over the mark without pressing. Keep each pause in compressions under 10 seconds. Then lift both grippers 3 cm clear of the chest and the bag." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir ~/yam-eval/cpr-two-rescuer/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when both grippers
jump back to their ready pose, open, in one step).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`chest_mm`, `bag_mm`, `last_compression` with its depth, recoil, interval,
distance from the mark and `pressed_at_mark`, `last_breath`, `cycles` with their counts, rates and pauses, `hand_at_mark`, `clear_mm`,
the tool points and finger tips, ...); `MECHKIT_TRACE=mechkit.jsonl` the manikin's joints and links. Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 30 s; the last test runs our offline
checker on the folder when it is present).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads, once per control step (20 Hz), how far
the chest plate (d) and the bag's cap (b) are pushed down from their drawn positions (mechkit's joint readings) and
both grippers' tool points and finger tips (rigstate), and keeps a record: every compression (d rising through 20 mm
and falling back below 10 mm; its peak depth and time, the recoil before it, and at the peak the distance from the
mark of the nearest tool point low enough to be pressing it) and every breath (b over 15 mm and reaching 25 mm; its
start and end, whether a gripper was pressing the cap at its deepest, the deepest chest meanwhile, whether a gripper
waited over the mark throughout). From that record it
groups compressions into cycles (peaks less than 2 s apart) and breaths into the pauses between cycles, and scores
the trial:

| Outcome | Score |
|---|---|
| The chest pushed past 20 mm while the cap is past 15 mm, at any moment; or fewer than 20 good compressions | 0 |
| Exactly 2 cycles of 28-32 compressions at 100-120 a minute, at least 90 % good, exactly 2 counted breaths after each cycle and none before or during one, both pauses within 10 s, both grippers 30 mm clear of the plate and the cap at the end | 1.0 |
| 2 or more cycles, the first two of 25-35 compressions, at least 80 % good, at least 3 counted breaths after them, their pauses within 10 s | 0.75 |
| A cycle of 25-35 compressions followed by a counted breath, at least 60 % good, every pause within 15 s | 0.5 |
| Anything else with at least 20 good compressions (hands-only CPR, breaths out of sequence) | 0.25 |

A good compression is 50-60 mm deep, follows a recoil to 5 mm or less, has a gripper pressing the mark at its peak
(its tool point within 25 mm of the mark horizontally and at most 40 mm above the mark's top where it then is; with
closed finger tips on the mark the tool point is 13 mm above it), and comes 0.43-0.70 s after the previous one in
its cycle (100-120 a minute is 0.50-0.60 s; the band allows for the 0.05 s sampling). A breath counts if it lasts
0.6-2.0 s, a gripper is pressing the cap at its deepest (tool point within 60 mm of the cap's centre horizontally and
at most 40 mm above the cap's top where it then is), the chest stays within 10 mm meanwhile, and a gripper waits over
the mark (tool point within 25 mm, at most 60 mm above the mark's top as drawn) throughout. The trial ends early once the 1.0 conditions have held for 1 s with
no compression or breath in progress and both sliders still (under 5 mm/s); otherwise it is scored as it stands at
the run limit. A reading with NaN or inf in it scores 0 for that step and
changes nothing on the record. If the grippers cannot be read on a step, the gripper rules are taken as met for that
step and the step is counted (`arms_unavailable` in the trace).

Unintended strategies earn nothing: leaning on the chest (it never comes back below 10 mm) makes all the presses one
long compression (0); holding the bag down while compressing, or pressing the chest during a breath, scores 0 at once;
fast shallow taps are neither deep enough nor at the right interval (0); one arm that compresses and then stretches
over to the bag gives breaths with no gripper waiting over the mark, and one that leaves its gripper over the mark and
leans a link on the cap gives breaths with no gripper pressing the cap: neither kind counts (0.25 at most); a gripper
that only hovers over the mark while the other presses the plate somewhere else is too high to be the one pressing
(more than 40 mm above the mark at the peak), so those presses are never good (0); a third cycle, extra breaths,
breaths before the first cycle or during a cycle, and pauses over 10 s all keep the score below 1.0.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone gives both cycles of compressions; nobody gives the breaths) scores 0.25:
hands-only CPR. `DEMO_IDLE_ARM=right` (the left arm alone presses the bag; nobody compresses) scores 0. (Offline
kinematic predictions; the GPU runs are recorded in `task.json`'s `scores`.)

With the gripper pointing straight down (how the demo, and a person, presses), the bag's near edge (y +0.19) is out
of the right arm's reach even with open fingers (its tool point gets no further left than y +0.11 at the cap's
height), and the left arm's tool point cannot come within 5 cm of the mark.
Our kinematics also show that an arm stretched sideways with its gripper turned far from vertical could put its tool
point over the other station (joint limits only; the simulator lets an arm fold through itself), so reach alone is
not what makes the task two-armed: the breath rule is. Every breath must be given by a gripper pressing the cap while
a gripper waits over the mark, and no gripper can be in both places, 39 cm apart (an arm's link on the cap does not
count as a gripper pressing it).

## Demo and run limit

The scripted demo takes 45.9 s (918 control steps). Both arms close their fingers on the way to their stations
(3 s) and push with the closed finger tips. The right arm, its fingers closing along the diagonal (yaw 45 degrees) so
that the closed finger carriages stand over the corners of the 120 x 120 mm chest opening, comes down to 19 mm above
the mark and from t = 4.0 s gives 30 compressions, each down to a tool point 31 mm above the table in 0.25 s and back
up to 100 mm in 0.30 s (0.55 s each: 109 a minute); it then waits over the mark, not pressing, until t = 24.4 s, gives
the second 30, waits over the mark again while the last breaths are given, and rises clear. The left arm waits 30 mm
above the cap and, 0.4 s after each cycle's last up-stroke, gives 2 breaths: down to a tool point 85 mm above the
table in 0.3 s (the cap about 36 mm down), held 0.7 s, back up in 0.4 s, 0.3 s apart; then it rises clear. The pause
between the cycles is 4.45 s, the one after the second cycle (to the end of its second breath) 3.6 s. Offline (no
lag), 0.65 s pass between the chest coming back under 10 mm and the cap passing 15 mm, and 0.75 s between the cap
coming back under 15 mm and the chest passing 20 mm at the next cycle: room for the arms' lag on the GPU. Offline,
success has held for its 1 s at step 921; the scripted policy stops at step 979. The run limit is 1,400 steps (70 s).

The chest's depth is set by how far the arm goes down, its give against the spring and its lag. The arm's tool point
gives about 1.9 mm per newton downward at the chest (J K^-1 J^T of its joint gains; the shoulder and elbow are
80 N m/rad springs), so the 4 N spring holds it about 7 mm above its target at the bottom of the stroke: offline, with
that give but no lag, the stroke to 31 mm holds the chest 55.4 mm down. On the GPU the arm also lags a stroke of 1.8 a
second, so the depths will come out shallower: read `last_compression.depth_mm` and `recoil_mm` in `TASK_TRACE` and
tune only `STROKE_BOTTOM` in `task.py` (and, if the depths scatter, the 0.25 / 0.30 s split, keeping each compression
within 0.50-0.60 s) until the depths land at 55 +- 3 mm with the recoils under 5 mm. If the recoils come out at 4 mm
or more (the arm not getting back up in time), raise `STROKE_TOP` (also the height at which the right arm waits over
the mark; keep it at or below 125 mm, inside the 60 mm the breath rule allows). The breaths' depth (36 mm against
a 25 mm threshold) and length (about 1.0 s against 0.6-2.0 s) have room for lag. If `last_breath.chest_max_mm` reads
more than 3 mm (the chest not yet back up when a breath starts), raise `BREATH_DELAY_S` (the pauses have over 5 s of
room under 10 s).

Offline (our NumPy-only checker, which has no physics and sees mechkit's joints at 0), the chest's and the cap's
travel come from `offline_press.py`: each slider rests at its sag under its own weight, and finger tips over its top
face and below it hold it down, the slider's spring and the arm's give acting as two springs in series. `task.py`
uses it only when the manikin's state says it came from the offline checker; the simulator never calls it. The
checker's own interference warning for the right gripper is against the manikin's bounding box; against the real
shapes, the gripper's and fingers' collision capsules stay at least 8 mm clear of the deck at the bottom of every
stroke (without the arm's give, which adds about 7 mm).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`; the manikin is fixed in place in any case): there are no seeds. Reset:
nothing is loose or consumed; both springs return the chest plate and the cap to rest, and the grippers start open at
the ready pose.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is about 49 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A CPR feedback training manikin (any that logs compression depth, recoil and rate) with a 30 mm dark sticker on the
compression point, or a 170 x 300 x 80 mm plywood box with a 120 x 120 mm opening and a sprung 115 mm plate in it (a
75 N/m compression spring with 65 mm of travel and a ruler or a linear potentiometer behind the plate), and a 60 mm
sprung cap over a post as the bag (a 40 N/m spring and a second linear potentiometer, logging the cap's travel
for the 15 mm and 25 mm thresholds). The two logs give depth, recoil, rate and the breaths; top and side video show
where the grippers are.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
