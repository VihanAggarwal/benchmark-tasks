# Thin a row of seedlings to one per 10 cm, keeping the stronger of each crowded pair

Thinning direct-sown crops is routine hand work for crop and nursery farmworkers: seed is sown thickly, and once
the seedlings are up the crew pulls out the extras so that the plants left stand a set distance apart, keeping the
strongest plant wherever two are crowded. Here ten green seedlings (upright cylinders 20 mm across, 45 to 90 mm
tall, height standing for strength) stand in a crowded row on a brown seed bed that runs across the table in front
of both arms, and a black compost tray stands beside each arm's base. The instruction gives the rule: going from
the tallest seedling down, keep a seedling if it is at least 10 cm from every seedling kept, pull out all the others
and put them in the trays, and do not disturb the ones kept. The rule keeps four and pulls six. Whether a seedling
goes can depend on a taller one two plants along (the 60 mm one at y -15 cm goes because of the 80 mm one at
-24 cm, with a short one between them), or on one across the middle of the row on the other arm's side (the 65 mm
one at +2 cm goes because of the tallest, at -4 cm), and every one must be pulled from between neighbours 20 to
50 mm away without touching them. The row is longer than either arm reaches, so
each arm thins its own half.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold in one place, and the thinning rule (`keep_set`) |
| `scoring.py` | The scorer: plain functions of the ten seedlings' states, with a tracker for the whole trial |
| `tests/test_scoring.py` | Offline tests of the rule, every tier and zero condition, resets and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, loose upright cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Thin the row of seedlings. Going from the tallest seedling down, keep a seedling if it is at least 10 cm (centre to centre) from every seedling you are keeping; pull out all the others and put them in the black compost trays. Don't disturb the seedlings you keep." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/thin-seedlings-row/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Thin the row of seedlings. Going from the tallest seedling down, keep a seedling if it is at least 10 cm (centre to centre) from every seedling you are keeping; pull out all the others and put them in the black compost trays. Don't disturb the seedlings you keep." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/thin-seedlings-row/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check: nothing should move); `TASK_TRACE=trace.jsonl`
writes one JSON line per control step with what the scorer saw and decided (`score`, `zero_reason`,
`left_on_bed`, `spacing_ok`, `wrong_choice`, `to_pull_in_trays`, `where` for each seedling, `peak_move_mm`,
`peak_tilt_deg`, `max_keeper_move_mm`, `moving`, the positions, ...). Videos: `inspect-robots video <log>.json`.
The scorer keeps one trial's record per process and starts it over when the simulator puts every seedling back on
its spot in one step (a reset), so run one trial per process or rely on that; the run rate is assumed to be 20 Hz
(`-E control_hz=20`). The layout is fixed: always pass `-E object_jitter=0` (a jittered reset would scramble the
row and the spots the scorer measures from).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The rule on the starting row keeps the seedlings at y
+19.5 cm (85 mm tall), +9.0 (75), -4.0 (90) and -24.0 (80), and pulls +24.0 (70, crowded by +19.5),
+13.0 (50), +2.0 (65, crowded by -4.0 across the midline), -9.5 (55), -15.0 (60, crowded by -24.0) and
-19.0 (45). That keep set is the only one in which no two seedlings left stand closer than 10 cm and every pulled
seedling had a taller keeper within 10 cm (the tests check every subset).

The scorer reads all ten seedlings every control step and remembers, for each one, the furthest its centre has
been from its starting spot (in 3D, so lifting counts) and the most it has tilted (either end up). A seedling is on
the bed when its centre is over the bed and its lowest point within 5 mm of the bed's top; in a tray when its centre
is over the tray's floor and its lowest point below the rim. Spacing is judged on where the seedlings left on the
bed were planted.

| Outcome | Score |
|---|---|
| A seedling off the table; a seedling left on the bed that was ever moved more than 15 mm or tilted more than 15 degrees; or fewer than 3 left on the bed | 0 |
| Exactly the keep set left on the bed, none ever moved more than 5 mm, the six others each in a tray | 1.0 |
| Spaced and every other seedling in a tray, with one fault: one keeper moved 5-15 mm, or one keeper replaced by a shorter seedling planted within 10 cm of it | 0.75 |
| The seedlings left on the bed are spaced (no two planted less than 10 cm apart), whatever was pulled and wherever it went | 0.5 |
| At least 3 of the six to pull in a tray and all four keepers on the bed | 0.25 |
| Anything else (nothing pulled, too few pulled, a keeper gone with the spacing still broken) | 0 |

Four "one wrong choice" cases exist on this row: keeping +24.0 instead of +19.5, -15.0 or -19.0 instead of -24.0,
or -9.5 instead of -4.0; each is a misjudged height in a crowded pair. The decisive pairs differ by 15 mm or more
(85 vs 70, 80 vs 60, 90 vs 55 mm).

The score is read at the last step; the trial ends early once the 1.0 conditions have held with every seedling
still for 1 s. A simulator state with a non-finite number (or a missing seedling) scores 0 for that step and leaves
the trial's record as it was.

Unintended strategies earn nothing extra, as the tests show: pulling every seedling leaves none (0); pulling every
other one breaks the spacing either way round (0.25 at most); keeping the most seedlings the 10 cm spacing allows
(five, +24.0, +13.0, +2.0, -9.5 and -24.0, ignoring height) is spaced but keeps weak plants (0.5); knocking the
extras off the bed onto the table instead of composting them is 0.5; knocking a seedling over and leaving it on the
bed, shoving a keeper more than 15 mm, rocking one past 15 degrees, or pulling a keeper and setting it back on its
spot all score 0; standing a pulled seedling back on the bed elsewhere scores 0; holding the last one above its
tray instead of letting go is 0.5.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) pulls the three seedlings on the right half into its tray and leaves
the left half crowded: 0.25. `DEMO_IDLE_ARM=right` (the left arm alone) does the same on the left: 0.25. With
straight-down grips neither arm can finish alone: along the row each arm reaches only about 14 cm past the
midline, at any turn of the gripper (offline kinematics, `tools/offline/yamkin.py`). The right end of the row
holds -15.0, -19.0 and -24.0, 40 to 90 mm apart and all beyond the left arm's reach, and the left end holds +19.5
and +24.0, 45 mm apart and both beyond the right arm's reach. So one arm alone always leaves a crowded pair at the
far end, the spacing fails, and it scores 0.25 at most. Each arm also reaches only the compost tray beside its own
base. (The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 43.4 s (868 control steps); both arms work in parallel, each on its own half. Each
gripper closes along x, across the row, so its open fingers come down in front of and behind the bed, never into
the gaps between neighbours. For each seedling: down beside it with the finger tips 10 mm above the bed's top
(higher for the 65 and 70 mm seedlings, so the stem's top stays clear of the palm), grip the stem (fingers
commanded to 12 mm on the 20 mm stem, over 1.5 s), pull it straight up until its bottom is 108 to 117 mm above
the table (clear of every seedling's top), carry it over the tray with the gripper turned 45 degrees, lower it so
its bottom is 18 to 27 mm above the 40 mm rim, let it drop in (one spot per seedling, 45 to 56 mm apart) and go
back up. The left arm pulls +2.0, +13.0 and +24.0 cm; the right arm -19.0, -15.0 and -9.5 cm. The right arm first
waits up over its tray until the left gripper has lifted the seedling nearest the middle (the only targets less
than 25 cm apart are the two arms' first ones), so the grippers stay at least 36 cm apart. Offline, success has
held for its 1 s at step 854; the policy stops at step 929. The run limit is 1,600 steps (80 s), about twice the
demo, leaving a policy time to look at the row and work out the rule.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand each seedling upright back on its spot
on the bed's centre line (positions in `task.json`'s `setup`); the bed and the trays are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 43 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 520 x 60 mm strip of cork board 10 mm thick, painted brown, with ten shallow 20 mm holes along its centre line
at the positions in `setup`; ten green-painted wooden dowels or foam stems 20 mm across, cut to 45-90 mm and
weighted to the stated masses, standing in the holes; and two open black plastic boxes with 40 mm walls round a
100 x 100 mm floor as compost trays, clamped beside the arms. Lay a ruler along the bed and film from above and from
the side to judge whether a keeper was moved or tipped.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
