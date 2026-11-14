# Stand a glass unit on two setting blocks in a window frame channel

General maintenance and repair workers (about 1.6 million in the US, BLS occupation 49-9071) keep buildings in
repair, and replacing a broken or misted glazed unit in a window or door is part of that regular work. On the job the worker takes off the beads, lifts out the failed unit, checks or
renews the setting blocks and stands the new unit on them in the existing frame; this task is that last step. The unit
stands on two setting blocks at its quarter points, so its weight never bears on the frame's sill: glass whose
bottom edge comes down on the sill can chip, crack or lose its seal. Here the bottom of a window frame (a sill with two glazing
stops forming a 40 mm channel, closed by two short jambs) is fixed on the table. In the demo the right arm takes two
rubber setting blocks from a tray and sets them on two white marks in the channel; the left arm carries a 200 mm
double-glazed unit over, lowers it straight down into the channel, stands it on both blocks and lets go. The scorer
watches the whole trial: if the glass ever touches the sill (contact, 0.5 mm or less above the surface under it),
the trial scores 0, however it ends.
It also reads both arms: glass still held by a gripper, or hovering above the blocks, is not resting on them. The
layout splits the job between the arms: with a top-down grasp only the right arm reaches the blocks and only the
left arm reaches the glass, and glass lowered into the channel without the blocks lands on the sill.

## Why it is hard

The 24 mm thick, 200 mm long unit is carried over stops 40 mm tall, into a 40 mm channel (8 mm of play per side),
and set down on blocks whose tops are 9 mm above the sill. The glass may not touch the sill at any step
(contact: some part of it inside the channel 0.5 mm or less above the sill's top, or above a mark's top over a
mark): standing on the blocks, its lowest corner is about 9 mm above the sill (8.3 mm above it at the lowest
in the GPU demo), so a unit let down far out of level, onto one block only, or with no blocks, reaches it.
Then both hands must open and move 30 mm or more clear while the glass stays standing on the blocks by itself.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the glass unit's and blocks' states and of the arms, with a small tracker for the whole trial |
| `rigstate.py` | Reads both arms (TCP, pad-tip point, gripper opening) before every control step for the scorer, and its fail-closed let-go helpers |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |
| `REVISION.md` | What changed in answer to the review |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer. `rigstate.py`
wraps the embodiment's step and reset from the task file to read the arms; it changes nothing the embodiment does.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the glass unit centred in the frame on the two black setting blocks, set on the white marks, and let go. The glass must never touch the sill." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/glaze-on-setting-blocks/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the glass unit centred in the frame on the two black setting blocks, set on the white marks, and let go. The glass must never touch the sill." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/glaze-on-setting-blocks/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module, and recognises a
new episode in the same process only when the simulator puts the glass unit and both blocks back on their
starting spots in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`touched`, `touched_at`, `min_low_in_channel_mm`, `min_gap_in_channel_mm`,
`unit_low_mm`, `unit_tilt_deg`, `unit_off_mm`, `on_block`, `gap_mm`, `share_under`, `block_mark_mm`,
`block_yaw_deg`, `hand_gap_mm`, `let_go_steps`, `still`, `settled`, `arms_unreadable`, both TCPs, positions, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Once per control step (20 Hz) the scorer reads the glass
unit's and the two blocks' poses, and both arms (TCP, pad-tip point and gripper opening, through `rigstate.py`).
It keeps a small record of the trial:

- whether the unit has ever touched the sill: at some step some part of it inside the channel's outline (2 mm
  slack) was in contact with the sill's top or a mark's top (0.5 mm or less above the surface under it; a near miss
  is not a touch). The scorer tests the unit's eight corners and points every 5 mm or finer along its twelve edges,
  because a unit lowered with no blocks rests on the two 1 mm marks along its bottom edges while its corners hang
  1 mm above the bare sill. The scorer also tests, exactly, the points where the unit's edges cross the vertical planes
  of each mark's four sides and the unit's lowest point above each mark corner, so contact at a mark's outline between
  two samples is not missed. A unit standing on blocks set on the marks is about 9 mm up, so the rule catches glass lowered too
  far, dropped, or rocked onto one end, even for a single step;
- for the unit and each block, how many steps in a row it has been **let go**: both grippers' finger zones 30 mm
  or more from it. A finger zone is every point within r of the 38 mm of the gripper's axis above the pad tips,
  r = hypot(half the finger gap + 8 mm, 13.5 mm): 35 mm at the demo's 48 mm let-go opening, 57 mm fully open,
  16 mm closed. So a finger anywhere on the object, open or closed, has not let go of it;
- the last 20 poses of each, to judge it **still**: each of them within 2 mm and 2 degrees of the current one.

The rule fails closed. A step on which the arms cannot be read (no snapshot, a side missing, a non-finite TCP,
pad-tip point or opening) or a distance cannot be computed counts as not let go for every object and restarts its
count (`arms_unreadable` counts such steps). A simulator state with a non-finite number or a zero quaternion
scores 0 for that step, keeps the sill record and restarts the let-go and stillness counts.

"Resting" needs all three: let go on each of the last 20 control steps (1 s), still over them, and the support
within 1 mm. A block is **set** on a mark when it lies flat on the sill in the channel (its lowest point within
1 mm of the sill's or a mark's top), is let go and still, is square within 15 degrees and its centre is within
8 mm of the mark's. The unit **rests on** a block only when the unit and that block are both let go and still, at
least half of the block's top (the real outlines seen from above, not bounding boxes) is under the unit's
underside, and the underside is within 1 mm of the block's top. A gripper holding the glass perfectly still on
the blocks, or just above them, therefore earns none of the resting tiers. The unit and the blocks are plain
boxes, the same either way up: the unit may stand on either long edge and a block may lie on either broad face.
A new episode is recognised when rigstate sees the embodiment reset, or when the simulator puts everything back on
its starting spot (within 4.5 cm, which covers a seeded reset's jitter), at rest, in a single step, with
something having moved 8 cm further than its last velocity and a free fall could carry it (so glass dropped back
onto its start is not mistaken for a reset). The score is read at the last step.

| Outcome | Score | Real-bench check |
|---|---|---|
| The unit touched the sill at any moment (in contact with the sill's top or a mark's top, 0.5 mm or less above the surface under it); or at the end the unit is tilted over 30 degrees (lying down, on its end) or off the table | 0 | Blue marking chalk on all four edges of the unit: a print on the sill or on a white mark shows a touch (a near miss leaves none); video for the end pose; read the prints on the top photo after the unit is lifted out |
| Both blocks set (one on each mark, within 8 mm, square within 15 degrees, let go and still); the unit standing in the channel (tilt under 3 degrees) resting on both (let go, still, within 1 mm), within 8 mm of centre | 1.0 | Ruler on the end photo (unit to channel centre), protractor on the photos (unit tilt); after the unit is lifted straight out, ruler and protractor on a top photo of the channel (block to mark, block turn); video (both hands 30 mm clear, nothing moving for 1 s); the unit stands on its own, let go and still for 1 s, with no chalk print on the sill or marks, so only the blocks carry it |
| The unit standing resting on both blocks, the blocks within 20 mm of their marks (let go and still) and the unit within 15 mm of centre, but outside the 1.0 limits | 0.75 | As for 1.0, with the wider bands |
| The unit standing in the channel resting on at least one block, not meeting 0.75 (one block only, a block further off or turned, unit further off centre) | 0.5 | As for 1.0, for one block |
| Both blocks set, the unit not resting on them (not brought over, still held, hovering, leaning in the channel) | 0.25 | Ruler and protractor on a top photo for the blocks; video for the hands |
| Anything else | 0 | |

The 0.5 tier is partial credit for states a glazier would correct: blocks at the channel's ends instead of the
quarter points, the unit far off centre, or the unit slightly out of level and bearing on one block only. Glass let
go on a single block, with nothing under its other end, tips down to the sill and scores 0 (see the tests).

The trial ends early (success) once the 1.0 conditions have held for 1 s. They already include the unit and both
blocks let go and still over the last 1 s, so the trial cannot end while a gripper holds the glass. Otherwise it is
scored as it stands at the run limit.

Unintended strategies earn less, as the tests show: holding the glass still on the blocks, or within a millimetre
or two above them, earns 0.25 at most (the blocks), and holding a block on its mark means it is not set; gripping the glass again
after it has rested, or lifting it 5 mm off the blocks, drops it back to 0.25 until it has been let go again for a
full 1 s;
lowering the glass first and pushing the blocks in under it afterwards scores 0 (it has already been on the sill);
standing it on one block only lets its far end tip down to the sill (0); a momentary rock onto one end is caught on
the step it happens (0); stacking both blocks on one mark or standing a block on its edge as a taller support is
not "set" and the unit is not resting on a block on the sill (0); standing the unit on the blocks while they are
still on the tray is worth nothing (0); blocks moved to the ends of the channel instead of the quarter points give
0.5 at best; laying the glass flat across the stops is 0.

## One arm alone

Measured on the GPU with the revised task (2026-11-14): `DEMO_IDLE_ARM=left` (the right arm alone sets both
blocks) scores 0.25. `DEMO_IDLE_ARM=right` (the left arm alone lowers the glass where the blocks should be and lets
go) scores 0: with no blocks it comes down on the marks, 1 mm above the sill (touch recorded at step 782). With a
top-down grasp neither arm can do the other's half. The blocks' nearer sides
are at y -22.5 cm, and near the table the left arm reaches no further right than about y -18 cm at x 18 cm. The
glass unit's nearer end is at y +19 cm, and the right arm reaches no further left than about y +17 cm at x 21 cm
(the unit's near face; less higher up and further forward). So a single arm's best with top-down grasps is 0.25.
An arm stretched across the table with its gripper turned sideways could reach the other side's items, so the
split rests on the top-down reach, not on a hard limit. (The reach figures are offline kinematic predictions from
the arm's joint geometry; the scores above are the GPU runs.)

## Demo and run limit

The scripted demo takes 43.1 s (862 control steps). The right arm grips each block across its 30 mm length
(fingers along the channel, commanded 22 mm, closed over 1.5 s; the pads, 27 mm wide, leave 6.5 mm to each stop),
carries it 15 mm clear of the near stop, lowers it into the channel over 2 s, settles, lets go over 1.5 s with the
block 1.5 mm above its mark, goes straight up, then parks on its own side. Meanwhile the left arm grips the glass
unit's top edge across its 24 mm thickness (fingers commanded 16 mm, closed over 1.5 s, the gripper 16 mm below the
top edge), lifts it and holds it back out of the right arm's way. Once the right arm has parked, it brings the glass
beside the frame, carries it across the near stop with its bottom 18 mm clear, comes down into the channel over
1.5 s to 2 cm above the blocks, settles, lowers the last 2 cm straight down over 2.5 s (peak 12 mm/s), settles,
opens to 48 mm (12 mm clear of each face) over 1.5 s, goes straight up (1 s) and then back over the unit's empty
starting spot (2 s), where its finger zone ends about 12 cm from the glass. Straight up alone would not count as
let go: with the pad tips 18 mm over the glass, the open finger zone still reaches it. Every grip and release takes
1.5 s, inside the CLI's 0.05-per-step gripper limiter. `BLOCK_TRIM` and `PLACE_TRIM` in `task.py` (zero) shift the
blocks' and the unit's place points if a GPU run shows them landing consistently off their marks.

The arm's controller does not carry the payload's weight: the left arm sags about 5.6 mm (and 1.3 mm sideways,
0.9 degrees of tilt) under the 0.20 kg unit (static estimate from the arm's joint stiffness). So the glass crosses
the stop about 12 mm clear, and it comes to rest on the blocks before the slow descent ends, with no drop when the
pads open. None of this can end the trial early any more: the unit counts as resting only once both grippers have
been 30 mm clear of it for 1 s.

On the GPU (revised task, 2026-11-12) both grippers' finger zones were 30 mm clear of the glass from step 844
(during the left arm's move back), the unit and both blocks were let go and still, meeting the 1.0 conditions,
from step 863, and the trial ended on success at step 882 (44.1 s) of the 1,200-step run limit, with the nearer
finger zone of the two arms 120.7 mm from the glass and no unreadable arm step. The offline prediction (commanded
poses) was one step earlier at each point: 843, 862 and 881; without early success the scripted policy would stop
at step 923 (the plan, then its 3 s hold). With
the arm 5.6 mm or 15 mm lower than commanded throughout, success still comes more than 1 s before the policy stops
(`tests/test_scoring.py`, `DemoTiming`). The run limit is 1,200 steps (60 s).

What the demo reads: no object state. It is open-loop: a fixed list of timed waypoints for each arm, built from
the constants in `layout.py`; no object pose or other simulator state is read while it runs.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two blocks back flat on the tray,
centred at x 18.0 and 22.0 cm, y -24.0 cm, their 30 mm sides along y; the glass unit back standing on a long
edge on the table, centred at x 22.0 cm, y +29.0 cm, faces normal to x. The frame, marks and tray are fixed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 46 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 60 x 260 mm sill 10 mm thick (aluminium or hardwood) with two 8 x 40 mm aluminium flat bars screwed along it
40 mm apart as the stops, and two 40 x 10 x 60 mm blocks closing the channel's ends, clamped to the bench; two
24 x 30 mm white labels on the sill at the quarter points; two setting blocks cut from 8 mm neoprene or EPDM to
24 x 30 mm (about 7 g each) on a small tray; and a 24 x 200 x 80 mm double-glazed unit of about 200 g (two 2 mm
panes on a 20 mm aluminium spacer; for safety, acrylic panes on a weighted spacer to the same mass). Chalk all four
edges of the unit with blue marking chalk before each trial (the marks are white, so a blue print shows): a print on
the sill or on a white mark shows that it touched. Contact is the rule in simulation and on the bench alike (a near
miss leaves no print and is not a touch). After the end photo and video, lift the unit straight out and photograph
the channel from above: the blocks' distance from the marks, their turn and any chalk print on the sill or a mark are
read on that photo. Read the other thresholds as the scoring table says: a ruler and protractor on the end photo for
the unit's centre distance and tilt, the video for the hands' 30 mm release and the 1 s stillness, and, for
"resting", the unit standing on its own, let go and still for 1 s with no chalk print on the sill or marks, so only
the blocks carry it.

Third-party assets: none. The scene is built from the task file's own boxes.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
