# Glaze a window: set two setting blocks, then stand the glass unit on them

Glaziers stand an insulated glass unit on two setting blocks placed at its quarter points, so its weight never bears
on the frame's sill or on one corner: glass whose bottom edge comes down on the metal sill, or that rests on a
single point, can chip, crack or lose its seal. Here the bottom of a window frame (a sill with two glazing stops
forming a 40 mm channel, closed by two short jambs) is fixed on the table. The right arm takes two small rubber
setting blocks from a tray and sets them on two white marks in the channel; the left arm carries a 200 mm
double-glazed unit over, lowers it straight down into the channel and stands it on both blocks. The scorer watches
the whole trial: if any corner of the glass ever comes down to the sill, the trial scores 0, however it ends. The
layout splits the job between the arms: with a top-down grasp only the right arm reaches the blocks and only the
left arm reaches the glass, and glass lowered into the channel without the blocks lands on the sill.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the glass unit's and blocks' states, with a small tracker for the whole trial |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

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
  --instruction "Put the two black setting blocks on the white marks in the frame, then lower the glass unit straight down into the frame so it stands on both blocks and never touches the sill." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/glaze-on-setting-blocks/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the two black setting blocks on the white marks in the frame, then lower the glass unit straight down into the frame so it stands on both blocks and never touches the sill." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/glaze-on-setting-blocks/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module, and recognises a
new episode in the same process only when the simulator puts the glass unit and both blocks back on their
starting spots in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`touched`, `touched_at`, `min_low_in_channel_mm`,
`unit_low_mm`, `unit_tilt_deg`, `unit_off_mm`, `on_block`, `gap_mm`, `share_under`, `block_mark_mm`,
`block_yaw_deg`, positions, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the glass unit's and the two
blocks' poses and velocities, once per control step (20 Hz), and keeps one fact about the trial: whether any
corner of the unit inside the channel's outline has come lower than 2 mm above the sill's top (1 mm above the
marks). A box's lowest point is always at a corner, and a unit standing on a block is at least 8 mm up, so the
rule catches glass lowered too far, dropped, or rocked onto one end, even for a single control step. The unit and
the blocks are plain boxes, the same either way up: the unit may stand on either long edge and a block may lie on
either broad face. "Resting on a block" uses the real outlines seen from above (at least half of the block's top
under the unit's underside), not bounding boxes. A new episode is recognised when the simulator puts everything
back on its starting spot (within 4.5 cm, which covers a seeded reset's jitter), still, in a single step, with
something having moved 8 cm further than its last velocity and a free fall could carry it (so glass dropped back
onto its start is not mistaken for a reset). A simulator state with a non-finite number or a zero quaternion
scores 0 for that step and leaves the record as it was. The score is read at the last step.

| Outcome | Score |
|---|---|
| A corner of the unit came within 2 mm of the sill at any moment; or at the end the unit is tilted over 30 degrees (lying down, on its end) or off the table | 0 |
| Both blocks set (one on each mark, within 8 mm, square within 15 degrees); the unit standing in the channel (tilt under 3 degrees) on both, within 8 mm of centre | 1.0 |
| The unit standing on both blocks, the blocks within 20 mm of their marks and the unit within 15 mm of centre, but outside the 1.0 limits | 0.75 |
| The unit standing in the channel on at least one block, not meeting 0.75 (one block only, a block further off or turned, unit further off centre) | 0.5 |
| Both blocks set, the unit not standing on them (not brought over, still held above, leaning in the channel) | 0.25 |
| Anything else | 0 |

The trial ends early (success) once the 1.0 conditions have held with the unit and both blocks still (under
1 cm/s and 0.5 rad/s) for 3 s, longer than the demo takes to settle and open its fingers once the glass is down
(2 s), so it cannot end while the gripper still holds the unit (see the demo below). Otherwise it is scored as
it stands at the run limit.

Unintended strategies earn less, as the tests show: lowering the glass first and pushing the blocks in under it
afterwards scores 0 (it has already been on the sill); standing it on one block only lets its far end tip down to
the sill (0); a momentary rock onto one end is caught on the step it happens (0); stacking both blocks on one mark
or standing a block on its edge as a taller support is not "set" and the unit is not resting on a block on the
sill (0); standing the unit on the blocks while they are still on the tray is worth nothing (0); blocks moved to
the ends of the channel instead of the quarter points give 0.5 at best; laying the glass flat across the stops is
0. The scorer cannot see the grippers, so glass held perfectly still within 2 mm of the blocks counts as resting
on them (see `fidelity`).

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone sets both blocks) scores 0.25. `DEMO_IDLE_ARM=right` (the left arm
alone lowers the glass where the blocks should be and lets go) scores 0: with no blocks it comes down on the
marks, 1 mm above the sill. With a top-down grasp neither arm can do the other's half. The blocks' nearer sides
are at y -22.5 cm, and near the table the left arm reaches no further right than about y -18 cm at x 18 cm. The
glass unit's nearer end is at y +19 cm, and the right arm reaches no further left than about y +17 cm at x 21 cm
(the unit's near face; less higher up and further forward). So a single arm's best with top-down grasps is 0.25.
An arm stretched across the table with its gripper turned sideways could reach the other side's items, so the
split rests on the top-down reach, not on a hard limit. (Offline kinematic predictions with our model of the arm;
the GPU runs are recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 41.1 s (822 control steps). The right arm grips each block across its 30 mm length
(fingers along the channel, commanded 22 mm, closed over 1.5 s; the pads, 27 mm wide, leave 6.5 mm to each stop),
carries it 15 mm clear of the near stop, lowers it into the channel over 2 s, settles, lets go over 1.5 s with the
block 1.5 mm above its mark, then parks on its own side. Meanwhile the left arm grips the glass unit's top edge
across its 24 mm thickness (fingers commanded 16 mm, closed over 1.5 s, the gripper 16 mm below the top edge),
lifts it and holds it back out of the right arm's way. Once the right arm has parked, it brings the glass beside
the frame, carries it across the near stop with its bottom 18 mm clear, comes down into the channel over 1.5 s to
2 cm above the blocks, settles, lowers the last 2 cm straight down over 2.5 s (peak 12 mm/s), settles, opens to
48 mm (12 mm clear of each face) over 1.5 s and goes straight up. Every grip and release takes 1.5 s, inside the
CLI's 0.05-per-step gripper limiter. `BLOCK_TRIM` and `PLACE_TRIM` in `task.py` (zero) shift the blocks' and the
unit's place points if a GPU run shows them landing consistently off their marks.

The arm's controller does not carry the payload's weight: the left arm sags about 5.6 mm (and 1.3 mm sideways,
0.9 degrees of tilt) under the 0.20 kg unit (static estimate from the arm's joint stiffness). So the glass
crosses the stop about 12 mm clear, and it comes to rest on the blocks about 0.7 s before the slow descent ends,
with no drop when the pads open. The 3 s stillness (rather than the usual 1 s) is longer than the time from that
first rest until the fingers are fully open (2.7 s), so the trial cannot end while the gripper still holds the
glass: it ends at about step 808 with the fingers open and the hand rising. With 15 mm of sag the fingers are at
40 of their 48 mm when it ends, and up to about 20 mm the pads have left the glass. Offline (no sag) success holds
for its 3 s at step 846; the scripted policy stops at step 883. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two blocks back flat on the tray,
centred at x 18.0 and 22.0 cm, y -24.0 cm, their 30 mm sides along y; the glass unit back standing on a long
edge on the table, centred at x 22.0 cm, y +29.0 cm, faces normal to x. The frame, marks and tray are fixed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 44 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 60 x 260 mm sill 10 mm thick (aluminium or hardwood) with two 8 x 40 mm aluminium flat bars screwed along it
40 mm apart as the stops, and two 40 x 10 x 60 mm blocks closing the channel's ends, clamped to the bench; two
24 x 30 mm white labels on the sill at the quarter points; two setting blocks cut from 8 mm neoprene or EPDM to
24 x 30 mm (about 7 g each) on a small tray; and a 24 x 200 x 80 mm double-glazed unit of about 200 g (two 2 mm
panes on a 20 mm aluminium spacer; for safety, acrylic panes on a weighted spacer to the same mass). Chalk the
unit's bottom edge before each trial: a print on the sill outside the blocks shows that it touched.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
