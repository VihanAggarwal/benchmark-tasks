# Lift a crate on two fork bars only when both are fully under it, and keep it level

Forklift operators are taught to drive the forks fully under a load before lifting it: a load lifted on the fork
tips overloads them and slides or tips off. The load then travels low and level, is set down before the forks are
withdrawn, and is never pushed or steadied by hand. Here a brown crate, too big for a gripper on every side, rests
on three floor bearers; the two gaps between the bearers are its fork pockets. Each arm takes one loose grey fork
bar, both bars go fully into the pockets (to the white lines), the crate is lifted level on both bars together,
carried onto the bearers of a blue rack, set down, and the bars are slid out, put down and let go. The scorer
watches the whole trial, not only where the crate ends up: every lift must be made with both bars at least
two-thirds in (full marks: fully in), the crate must never tilt more than 5 degrees, rise above 9.5 cm, drop or be
touched by a gripper. The task needs both arms: one bar lifts only one side of the crate and tilts it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the crate's, the bars' and the grippers' states, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_ride.py` | Offline checks only: a small model of the crate riding on the bars (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook, the ride model and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The touch and let-go rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the gripper rules are then not applied on those steps, and the
trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the brown crate from the floor bearers onto the blue rack using only the two grey fork bars: slide both bars into the crate's fork pockets until their rear ends reach the white lines, lift the crate level on both bars together, set it on the rack's bearers and slide the bars out. Never lift with a bar less than two-thirds in, tilt the crate more than 5 degrees, raise its bottom above 9.5 cm, drop it or touch it with a gripper. Then put the bars down and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/fork-bars-full-engagement/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the brown crate from the floor bearers onto the blue rack using only the two grey fork bars: slide both bars into the crate's fork pockets until their rear ends reach the white lines, lift the crate level on both bars together, set it on the rack's bearers and slide the bars out. Never lift with a bar less than two-thirds in, tilt the crate more than 5 degrees, raise its bottom above 9.5 cm, drop it or touch it with a gripper. Then put the bars down and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/fork-bars-full-engagement/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`insertion_mm` per pocket, `lifted`,
`min_lift_insertion_mm`, `crate_low_mm`, `crate_tilt_deg`, `max_low_mm`, `max_drop_mps`, `bar_pitch_deg`,
`zone_gap_mm`, `clearance_mm`, `on_rack`, `bars_out`, `breaches`, `first_breach_step`, `opening`,
`arms_unavailable`, both TCPs, ...). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record
per process and starts it over at every reset, so run one trial per process or rely on the reset; the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 2 minutes; the last test runs our
offline checker on the folder when it is present).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The pockets are the gaps between fixed bearers, not part of
the crate, so a bar's insertion is measured against the crate wherever it is: the length of the bar's centre line
under the crate's outline (in the crate's frame), counted while that stretch is within 15 mm (across) of one of the
crate's two pocket lines, 12 cm either side of its centre, and the bar is just under the crate (its centre below
the crate's underside, its top no more than 20 mm below it: a bar resting on the table in a pocket is 12 mm below).
A pocket's insertion is the larger of the two bars' there; fully in, at the white line, is 100 mm. The crate is
lifted when its lowest point is more than 3 mm above the bearers it stands over (the rack's 72 mm once its centre is
at or past x 0.400, else the floor bearers' 36 mm; no stable resting place lies between).

The scorer cannot see which way a gripper is turned, so each gripper's fingers count as a round zone about its
axis, from the finger tips to 38 mm back up the gripper, 24 mm in radius at every opening: the outer corners of
fingers closed on a bar, whatever the turn. Fingers opened wider spread along an unknown direction and are not
followed further out, so letting go of a bar that is fully in (the open fingers spread along y, 45 mm in front of
the crate) is not a touch. Only the fingers are judged, not the rest of the gripper or arm.
Six rules hold for the whole trial; one broken at any moment scores 0 whatever follows:

1. the crate is lifted while either pocket's insertion is under 73 mm (two-thirds, the yellow line);
2. it tilts more than 5 degrees;
3. its lowest point rises more than 95 mm above the table;
4. its lowest point drops faster than 0.25 m/s (dropped onto something, or fell);
5. it leaves the table (its lowest point more than 10 mm below the table top);
6. a gripper's finger zone comes within 5 mm of it.

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Crate on the rack; every lift with both insertions 95 mm or more; both bars wholly out from under it; both grippers let go | 1.0 |
| Crate on the rack, but lifted at some moment with an insertion of 73 to 95 mm, or a bar left under it, or a gripper not let go | 0.75 |
| Not on the rack, but at some moment lifted 20 mm above the floor bearers on both bars (both 73 mm in) with its centre 50 mm toward the rack | 0.5 |
| Both insertions 95 mm or more at the same moment | 0.25 |
| Anything else | 0 |

'On the rack': the crate's outline seen from above within x 0.400 to 0.530 and within 0.165 of y 0 (10 mm and
15 mm each way for a centred crate), its lowest point within 3 mm of the rack bearers' 72 mm tops, tilted no more
than 3 degrees and turned no more than 5 degrees (either way round: it is a plain box). A gripper has let go when it
is open (opening 0.9 or more, fingers at least 84.5 mm apart) with its tool point and finger-tip point at least
30 mm from both bars and the crate.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with the crate and both
bars still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised from rigstate's reset hook, or, as a
fallback, when the simulator puts the crate and both bars back on their starting places (within 4.5 cm, which
covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number or a zero quaternion
scores 0 for that step and leaves the record of the trial as it was. Tilt is folded (either big face down).

Unintended strategies earn nothing, as the tests show: pushing the crate toward the rack with the bars' tips or a
gripper slides it off the floor bearers onto the platform (same 36 mm height) and against the rack bearers' faces,
not onto them (0); lifting on bars part way in, or sliding the bars out while it is still up, breaks rule 1 (0);
lifting one side at a time tilts it past 5 degrees (rule 2: 25 mm under one side is 5.3 degrees); carrying it high
over the rack breaks rule 3; letting it drop onto the rack breaks rule 4; steadying it with a free gripper breaks
rule 6. Leaving the bars under it on the rack, or a gripper still holding a bar, is 0.75.

## One arm alone

`DEMO_IDLE_ARM=left` and `DEMO_IDLE_ARM=right` (one arm runs its half of the demo) both score 0: one bar goes fully
in and rises, the crate pivots on its far edge and tilts past 5 degrees within 25 mm of lift (offline kinematic
predictions; the GPU runs are recorded in `task.json`'s `scores`). The best a single arm can do is push one bar in,
then the other (each arm reaches both grip points): 0.25. It cannot lift both bars at once, and the crate can only be
lifted on the bars.

## Demo and run limit

The scripted demo takes 34 s (680 control steps); both arms run the same plan mirrored in y, on one clock, so the
bars rise and fall together. Each gripper comes down over its bar's rear end (the last 22 mm over 1.5 s), closes
across the bar's 24 mm width with the pad tips 5 mm above the table (fingers commanded to 16 mm, over 1.5 s) 15 mm
from its rear end, and slides it 100 mm along the table into its pocket, to the white line; the gripper is then
45 mm in front of the crate. The bars rise 63 mm (the crate's underside from 36 mm to 87 mm, commanded), carry it
142 mm forward over the rack, set it down on the rack's bearers (2 s), drop 4 mm clear of it, slide out 112 mm and
back to lie in line with the floor pockets (5 s), go down onto the table, and let go (1.5 s); the grippers rise. The
TCPs stay 0.24 m apart. Offline, success has held for its 1 s at step 682; the scripted policy stops at step 741. The
run limit is 1,200 steps (60 s).

The arms' shoulders and elbows are position-controlled springs (80 N m/rad) with no gravity compensation for a
payload. With a bar and half the crate on each arm they sag about 3 to 4.5 mm down and 4 mm back, and the grippers
pitch about 1.2 degrees (J K^-1 J^T, our kinematics), so the crate's far edge rides about 8 mm lower than
commanded: about 79 mm up, 7 mm over the rack bearers' tops as it passes over them and 16 mm under the 95 mm limit
(8 mm under it if the arms do not sag at all). The 4 mm sag back puts the crate about 4 mm short of the commanded
position, inside the rack window either way. The bars are lowered to 6 mm under the crate and 5 mm over the
platform for the withdrawal (with the unloaded sag). `TASK_TRACE` reports `max_low_mm`, `crate_tilt_deg`,
`bar_pitch_deg` and `min_zone_gap_mm` to check these on the GPU.

Offline (our NumPy-only checker, which carries a gripped object with its gripper but not what rests on it), the
crate's pose comes from `offline_ride.py`: each pocket line of the crate rests on the higher of the fixed support and
a bar under it; both lines on bars carry it (it moves with them), one tilts it about its far edge, none leaves it
resting. The hooks are attributes the checker looks for on the object specs, attached only when `rigstate` finds
no simulator; in the simulator the specs are left as `yam_arena` made them and nothing in `offline_ride.py` runs.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the crate back square on the floor bearers,
centred at x 32.5 cm, y 0 (front face at x 27.0 cm); each bar back flat on the table along x, centred at x 19.0 cm,
y +12.0 cm and y -12.0 cm, tips at the pocket mouths. The bearers, lines and rack are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is about 37 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 110 x 300 x 50 mm closed crate of 3 mm poplar plywood (about 150 g); two 160 mm lengths of 24 mm square aluminium
tube with a 1.5 mm wall (about 60 g each); three 36 mm timbers 110 mm long (one 200 mm wide, two 20 mm wide) fixed
straight to the table top (double-sided tape or glue, so the table stays the pockets' floor) as the floor bearers,
40 mm apart; a 36 mm tall wooden platform 150 x 340 mm, painted blue, carrying three more such bearers as the rack;
white and yellow tape at the lines. A bubble level on the crate, a card behind the rack marked at 95 mm
and top and side video judge the rules.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
