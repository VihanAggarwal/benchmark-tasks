# Plank two scaffold bays: lap the boards over the middle putlog, ends 19-38 mm out

Scaffold erectors and construction laborers (O*NET 47-2061) plank a scaffold's working platform board by board,
and US rule 29 CFR 1926.451(b) sets where each board may end: every board end at least 6 in past the centre line of
its support ((b)(4)); no end of a board 10 ft or shorter more than 12 in past its support ((b)(5)(i)), because a
longer cantilever tips up when someone steps on it (a "trap end"); and where boards overlap to make a long
platform, the overlap only over a support and at least 12 in ((b)(7)). Here a 1:8 model of two scaffold bays
stands on the table: three fixed galvanised putlogs (transoms) 190 mm apart on centre, and two loose orange
scaffold boards, 250 mm long, one in front of each arm. The boards must plank both bays as one straight run: each
across its own bay with both ends 19 to 38 mm (6 to 12 in) past their putlogs' centre lines, the second lapped on
the first over the middle putlog. A 250 mm board on a 190 mm bay has 60 mm of overhang to share between its two
ends, so it must sit within 8 mm of centred along the run; and the lapped board rests on the lower board's end at
one end and on its own outer putlog at the other, sloping about 5.4 degrees. The score is what a scaffold
inspector checks with a tape and a photo; any board end more than 38 mm past its putlog scores 0. The demo uses
both arms, each laying the board in front of it (see 'One arm alone' for what one arm scores).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two boards' poses, with a small tracker of their last half second |
| `offline_settle.py` | Offline only (never runs in the simulator): how the lapped board settles, for our kinematic checker, which has no physics |
| `tests/test_scoring.py` | Offline tests of every scoring tier, the trap-end rule, the exploits we tried, resets, NaN states, the hooks and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

`offline_settle.py` is imported by `task.py` only when the offline stand-in for `yam_arena` is loaded (it carries
`OFFLINE_STUB = True`; the real package does not). A checker without physics drops a released board straight down,
level, so it would leave the lapped board hanging level off the lower board's end; `offline_settle.py` turns it
down about that end onto its putlog, as a real board settles. In the simulator the physics does this and the module
is not loaded.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Plank both scaffold bays with the two orange boards in one straight run: lay one board across each bay on the grey putlogs, lap the two boards over the middle putlog, and leave every board end 19 to 38 mm past the centre line of the last putlog under it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/scaffold-planks-lapped/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Plank both scaffold bays with the two orange boards in one straight run: lay one board across each bay on the grey putlogs, lap the two boards over the middle putlog, and leave every board end 19 to 38 mm past the centre line of the last putlog under it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/scaffold-planks-lapped/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: `score`, `success`, `bays_planked`, `in_line_mm`, `lap_mm`,
`upper`, `lapped_in_line`, `trap_end`, `zero_reason`, `moved_mm`, and for each board (`boards`) `lies_on` (each
putlog it lies on, `direct` or `through` the other board), `on_scaffold`, `overhang_mm` (`low` and `high` end),
`heading_err_deg`, `tilt_deg`, `lowest_mm`, `within_platform`, `still` and `bears_in`, plus each board's pose
(`*_pose`: centre in mm, tilt, heading error, reported speed). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the two boards' poses once per control
step (20 Hz) and keeps only the last half second of each board's end points, for "still" (neither end moved more
than 2 mm in 0.5 s; a gripped board's reported velocity chatters, so its path is used). The putlogs are fixed, so
their tops come from `layout.py`.

- **Lies on.** A board lies on a putlog when its underside comes within 2 mm of the putlog's top where the two
  overlap seen from above (at least 20 square mm of overlap: a real crossing, not a corner touch). The faces are
  the board's real, turned and tilted faces, so a sloping board is found resting on an edge. A board also lies on
  a putlog *through* the other board when it rests on that board (underside within 2 mm of its top where they
  overlap) where that board lies on the putlog, with the putlog's centre line under their overlap: this is the
  lapped board over the middle putlog.
- **On the scaffold.** Lies on at least one putlog, long side within 45 degrees of the run (y), lowest point at
  least 5 mm above the table. Boards not on the scaffold (on the table, held more than 2 mm above it, leaning
  with one end on the table, lying along a putlog, standing on an edge) are ignored.
- **Overhang.** For each end: the distance along the board, toward that end, from where its centre line crosses the
  centre line of the last putlog it lies on, to the end face. A tape from a pencilled centre line on the putlog's
  side to the end face reads the same.
- **Planks a bay.** Lies on both of the bay's putlogs, long side within 5 degrees of the run, tilted at most
  8 degrees (the lapped board slopes about 5.4), outline within the putlogs' length (x 20.0 to 32.0 cm), both
  overhangs 19 to 38 mm, still.

| Outcome | Score |
|---|---|
| Any board on the scaffold with an overhang over 38 mm (a trap end) | 0 |
| Each bay planked by a different board, the two centre lines crossing the middle putlog's centre line within 10 mm of each other, one board resting on the other over the middle putlog | 1.0 |
| Each bay planked by a different board, not lapped in line (side by side, or more than 10 mm out of line) | 0.5 |
| Exactly one bay planked (including both boards across the same bay) | 0.25 |
| Anything else (nothing planked) | 0 |

The lap, the two inner overhangs added, is checked to be at least 38 mm (12 in), but it always is when both boards
plank their bays, since each inner end is at least 19 mm past the middle putlog; the demo's lap is 59 mm. Either
board may go in either bay and either may be the upper one. Where things end up decides every case: one board
across its bay and the other on the table 0.25; both boards across their bays side by side over the middle putlog
0.5; lapped and in line 1.0; a board balanced on one putlog (both ends 125 mm out) 0 even if the other bay is
planked; a flat board 9 mm off centre in its bay (one end 39 mm out) 0; the second board held level on the first
one's end, not let go (its outer end 16 mm above its putlog, so it lies on the middle putlog only and its outer end
is a 220 mm cantilever) 0. The trial ends early, at 1.0, once 1.0 has held for 1 s; otherwise the score is read at
the run limit (a 1.0 state there counts even if it has held less than 1 s). A new episode is recognised when the
simulator puts both boards back on their marks (within 4.5 cm, flat on the table, along x) in a single step. A
simulator state with a non-finite number scores 0 for that step and leaves the scorer's record as it was.

Unintended strategies earn no more than they should, as the tests show: planking the same bay twice (stacked or
side by side) is one bay (0.25); two boards passing each other over the middle putlog are both bays but not one run
(0.5); laying the second board level on top of the first, across the middle putlog, is a cantilever (0); sliding
the upper board inward to make a long lap leaves its inner end over 38 mm out (0); lapping between putlogs instead
of over the middle one leaves the upper board on its outer putlog only, a 180 mm cantilever (0); boards pushed off
the ends of the putlogs plank nothing. The scorer reads only the boards, not the grippers: a board still held is
scored where it lies. Held exactly in place, the lower board would count; the upper board, held level, cannot,
because its outer end is then 16 mm above its putlog.

## One arm alone

With straight-down grips each arm reaches only the board in front of it. The boards start at y +32 cm and -32 cm
(their inner sides at +-30 cm); at grip height the right arm's tool point reaches no further left than about y +21 cm
(at x 10 cm; +16 cm at x 24 cm, +12 cm at x 30 cm), and the left arm is its mirror image (inverse kinematics of the
arm's URDF with joint limits and a collision proxy, every grip heading tried). Either arm could lay its own board
across either bay, but with one board it can plank only one bay: 0.25. That is a limit of where the boards start,
not of the act: laying and lapping two boards does not in itself need two hands, and we do not claim it does. A
tilted wrist or a push toward the far board has not been tried.

The demo's one-arm runs: `DEMO_IDLE_ARM=left` (the right arm alone lays its board across the right bay) and
`DEMO_IDLE_ARM=right` (the left arm alone lowers its board to where the lap would be; with no lower board under it,
it drops 18 mm onto the left and middle putlogs, centred across the left bay) each scored 0.25 on the GPU, as the
offline kinematic check predicted: one bay planked, ends 29.8/30.1 mm (right board) and 29.5/30.5 mm (left board,
lying flat) past their putlogs, the other board untouched on the table (see 'Results in simulation' below).

## Demo and run limit

The scripted demo takes 24.7 s (494 control steps). Every grip closes over 1.5 s with the fingers commanded 8 mm
narrower than the 40 mm board, every release opens over 1.5 s, and every set-down slows for its last 30 mm.

- Both arms at once: each picks the board in front of it where it lies along x, fingers across its width (yaw 0),
  pad tips 4 mm above the table, and lifts it until the tool point is 10 cm above the table.
- Right arm: carries its board over the right bay, turning it a quarter turn (yaw +90 degrees) to lie along the
  run, centred at x 26.0 cm, y -9.5 cm; comes down to 3 cm above; lowers the last 3 cm over 1.5 s to 2 mm above the
  putlogs, settles 0.5 s, opens, goes up (15.2 s) and parks on its own side.
- Left arm: holds its board over its start until the right arm has gone up, then (15.7 s) carries it over the left
  bay, turning it the other way (yaw -90 degrees), centred at x 26.0 cm, y +9.5 cm; lowers it level to 2 mm above
  the right board's top (its inner end over the right board's end, its outer end 18 mm above the left putlog),
  settles, opens and goes up. Let go, the board tips about 5.4 degrees about the right board's end until its outer
  end rests on the left putlog. In the GPU run it came to rest tilted 5.4 degrees with its ends 29.5 mm (inner)
  and 29.7 mm (outer) past their putlogs, a 59.3 mm lap, its centre line 2.5 mm in line with the lower board's at
  the middle putlog; the lower board's ends were 29.8 and 30.1 mm out.

The arms' controllers do not compensate payloads; the static estimate for the 80 g board at either place pose
(J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) is about 1.5 mm down and under 1 mm sideways, inside the 2 mm drop
height and the 8 mm window. The lapped board cannot count until it has been let go and has settled onto its putlog,
and "still" needs 0.5 s of rest, so the trial cannot end while the gripper holds it. In the GPU run 1.0 was first
scored at step 434 and the trial ended early, at 1.0, at step 453 (the offline check also gave step 453); the
scripted plan would run to step 555. The run limit is 1,200 steps
(1 minute), about twice the demo.

Our offline kinematic check (NumPy, no physics; not part of this folder): worst IK error 0.09 mm, smallest joint-4
margin 15.2 degrees, the arms' collision capsules at least 25 cm apart, tool points at least 40 cm apart, pad tips
never lower than 4 mm above the table.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both boards back flat on the table, length
along x, centred at x 24.5 cm, y +32.0 cm and y -32.0 cm. The putlogs are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 25 s of simulated time; a full Dream-1 run is
1 minute); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Three 120 mm lengths of 20 mm square steel bar (or painted hardwood), screwed, clamped or glued to the table top
with their length along x, 190 mm apart on centre, each with its centre line pencilled on both sides; two pine
boards cut 250 x 40 x 16 mm (about 80 g) and painted orange. To score: a tape along each board from the pencilled
centre line to the end face for each overhang, a straightedge across the middle putlog for the line-up, and a top
and a side photo for which board rests on which.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
