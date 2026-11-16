# Stand a glass unit on two setting blocks in a window frame

General maintenance and repair workers (about 1.6 million in the US, BLS occupation 49-9071) keep buildings in
repair, and replacing a broken or misted glazed unit in a window or door is part of that regular work. On the job
the worker takes off the glazing beads, lifts out the failed unit, renews the setting blocks, sets the new unit on
them from the room side and pushes it up against the outside stop; the beads go back on after. This task is that
setting step. The unit stands on two setting blocks at its quarter points, so its weight never bears on the sill:
glass whose edge comes down on the sill can chip, crack or lose its seal.

A small fixed-light window frame stands on the table: a sill, two jambs and a head (20 x 60 mm sections) round a
240 mm wide opening, with a fixed outside stop running round it 15 mm into the opening on the far side. The room
side is open: its glazing bead has been taken off and lies on the table beside the frame. The head is 6 mm above a
unit standing on the blocks and covers the whole rebate, so neither the blocks nor the unit can go in from above.
In the demo the right arm takes two rubber setting blocks from a tray and slides them in under the head onto two
white marks on the rebate floor. The left arm grips the 200 mm double-glazed unit by its top edge, brings it to the
room side of the frame with its top tilted toward the room, sets its bottom edge on the two blocks without touching
the sill and holds it there. The right arm brings a closed fingertip to the unit's room face, the left arm lets go
and withdraws, and the right arm pushes the unit upright under the head until it stands against the outside stop,
then withdraws. The scorer watches the whole trial: if the glass ever touches the sill or a mark (contact: no air
gap between them), the trial scores 0, however it ends. It also reads both arms: glass still held, or leaning on a
finger, is not resting on the blocks.

## Why it is hard

The unit goes in tilted from the room side, as a glazier fits one: its bottom edge comes in over the sill's front
edge and onto blocks whose tops are only 9 mm above the sill, then it is turned upright about that edge, its top
passing 2.5 mm under the head, until it stands against the outside stop. Its bottom edge must never touch the sill
or a mark on the way in, nor while it is turned. The left hand holding the top edge must stay clear of the head (its
top must be well on the room side of the head while the bottom edge is set down), and the leaning unit must be
handed over to the other arm: let go before the fingertip is on it and it falls back toward the room. Both arms work
in the same few centimetres in front of the frame.

## Files

- `task.json`: the submission's details (instruction, setup, scoring, run limit, fidelity, scores, media ID).
- `task.py`: the `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()`.
- `layout.py`: every dimension and threshold, in one place (the frame as named boxes).
- `scoring.py`: the scorer: plain functions of the glass unit's and blocks' states and of the arms, with a small
  tracker for the whole trial.
- `rigstate.py`: reads both arms (TCP, pad-tip point, gripper opening) before every control step for the scorer,
  and its fail-closed let-go helpers.
- `tiltgrip.py`: tilted grips for the scripted demo (pitched and rolled), through `yam_arena.demo.grip_rotation`.
- `offline_unit.py`: offline only: a quasi-static model of the unit for `tools/offline/check_task.py`, which has no
  physics.
- `tests/test_scoring.py`: offline tests of the frame, every scoring tier, the contact rule, the wrong ways and the
  demo's plan.
- `REVISION.md`: what changed in answer to each review.

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer. `rigstate.py`
wraps the embodiment's step and reset from the task file to read the arms; it changes nothing the embodiment does.
`tiltgrip.py` wraps `yam_arena.demo.grip_rotation` so a demo waypoint can carry a tilted rotation (plain waypoints
are unchanged); the idea follows the program repository's tasks that patch `grip_rotation` for tilted grips, and
this is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
INSTR="$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')"
# scripted demo (both arms)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1800 --store-frames \
  --log-dir ~/yam-eval/glaze-on-setting-blocks/demo
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1800 --store-frames \
  --log-dir ~/yam-eval/glaze-on-setting-blocks/dream
```

The instruction is: "Stand the glass unit in the window frame on the two black setting blocks, set on the white
marks, and let go. The glass must never touch the sill."

Run one trial per process (the default). The scorer keeps the trial's record in the task module, and recognises a
new episode in the same process when rigstate sees the embodiment reset, or when the simulator puts the glass unit
and both blocks back on their starting spots in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`touched`, `touched_at`, `sill_gap_mm`, `min_sill_gap_mm`,
`stop_gap_mm`, `in_rebate`, `standing`, `unit_tilt_deg`, `unit_off_mm`, `unit_low_mm`, `on_block`, `gap_mm`,
`share_under`, `block_mark_mm`, `block_yaw_deg`, `hand_gap_mm`, `let_go_steps`, `still`, `settled`,
`arms_unreadable`, both TCPs, positions, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 2 minutes), and from the repository
root `python3 tools/offline/check_task.py tasks/glaze-on-setting-blocks --one-arm --strict`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Once per control step (20 Hz) the scorer reads the glass
unit's and the two blocks' poses, and both arms (TCP, pad-tip point and gripper opening, through `rigstate.py`).
It keeps a small record of the trial:

- whether the unit has ever touched the sill: at some step the unit and the sill (any of its faces: the rebate
  floor, its room-side front) or one of the white marks were in contact, touching or pressed into each other, their
  separation 0.0 mm or less. Contact between the two boxes is decided exactly (the separating-axis test over all 15
  axes of the two boxes), so every face, edge and corner of the unit counts. Any air gap, however small, is not a
  touch: a unit whose bottom edge passes 0.3 mm over the sill has not touched it, exactly as it leaves no chalk on
  the bench. Standing or leaning on blocks set on the marks the unit is 8 mm over them;
- for the unit and each block, how many steps in a row it has been **let go**: both grippers' finger zones 30 mm
  or more from it. A finger zone is every point within r of the 38 mm of the gripper's axis above the pad tips,
  r = hypot(half the finger gap + 8 mm, 13.5 mm): 26.7 mm at the demo's 30 mm let-go opening, 57 mm fully open,
  16 mm closed. So a finger anywhere on the object, open or closed, has not let go of it (the closed fingertip that
  pushed the unit upright included);
- the last 20 poses of each, to judge it **still**: each of them within 2 mm and 2 degrees of the current one.

The rule fails closed. A step on which the arms cannot be read (no snapshot, a side missing, a non-finite TCP,
pad-tip point or opening) or a distance cannot be computed counts as not let go for every object and restarts its
count (`arms_unreadable` counts such steps). A simulator state with a non-finite number or a zero quaternion
scores 0 for that step, keeps the sill record and restarts the let-go and stillness counts.

**Standing in the rebate**: all eight corners of the unit inside the opening seen from above (from the frame's room
face to the outside stop, between the jambs; 1 mm slack), its outside face within 3 mm of the outside stop, and its
height axis under 3 degrees from vertical. A block is **set** on a mark when it lies flat on the sill in the opening
(its lowest point within 1 mm of the sill's or a mark's top), is let go and still, is square within 15 degrees (the
blocks are square, so a quarter turn changes nothing) and its centre is within 8 mm of the mark's. The unit **rests
on** a block only when the unit and that block are both
let go and still, at least half of the block's top (the real outlines seen from above) is under the unit's
underside, and the underside is within 1 mm of the block's top. A unit still leaning on its bottom edge, or held,
rests on nothing. The unit and the blocks are plain boxes, the same either way up.

The tiers, each with how a real trial checks it:

- **0**: the unit touched the sill or a mark at any moment (contact, separation 0.0 mm or less); or at the end it is
  tilted over 30 degrees or off the table. Bench: blue marking chalk rubbed on the sill (top and room-side face) and
  on both marks before the trial; after the end photo and video the unit is lifted out, and any blue on it shows it
  touched (only contact transfers chalk; an air gap leaves none). The video shows the end pose.
- **1.0**: both blocks set (one on each mark, within 8 mm, square within 15 degrees, let go and still); the unit
  standing in the rebate (within 3 mm of the outside stop, tilt under 3 degrees) resting on both (let go, still,
  within 1 mm), within 8 mm of centre. Bench: a top photo of the opening after the unit is lifted out (block to
  mark, block turn); ruler and protractor on the end photo (unit to opening centre, tilt); a 3 mm feeler gauge, put
  in at the unit's end between its outside face and the stop, must not pass; the video (both hands 30 mm clear,
  nothing moving for 1 s); the unit stands on its own, let go and still for 1 s, with no chalk on it.
- **0.75**: the unit standing in the rebate resting on both blocks, the blocks within 20 mm of their marks (let go
  and still) and the unit within 15 mm of centre, but outside the 1.0 limits. Bench: as for 1.0, wider bands.
- **0.5**: the unit standing in the rebate resting on at least one block, not meeting 0.75. Bench: as for 1.0.
- **0.25**: both blocks set, the unit not standing in the rebate on them (not brought in, still held, left leaning,
  short of the stop). Bench: top photo for the blocks; video and feeler gauge for the unit.
- **0**: anything else.

The trial ends early (success) once the 1.0 conditions have held for 1 s. They already include the unit and both
blocks let go and still over the last 1 s, so the trial cannot end while a gripper or the pushing fingertip is on
the glass. Otherwise it is scored as it stands at the run limit.

Unintended strategies and wrong ways earn less, as the tests show (`WrongWays`, `LetGo`): the unit set straight on
the sill or the marks (no blocks), or its bottom edge brought down on the sill on the way in, scores 0 on the step
it happens, whatever follows; a momentary rock onto a corner counts; the unit left leaning on its edge, or still
held, or standing on the blocks more than 3 mm short of the outside stop, earns 0.25 at most (the blocks); blocks
12 mm off their marks give 0.75, 25 mm off 0.5; stacking both blocks on one mark, or standing the unit on the blocks
while they are on the tray, is 0; a fingertip still on the glass is not let go. Lowering the unit in from above is
not possible: the head is in the way (`Frame` tests).

## One arm alone

Predicted offline (the kinematic proxy, `check_task.py --one-arm`): `DEMO_IDLE_ARM=left` (the right arm alone sets
both blocks, then pushes at nothing) 0.25; `DEMO_IDLE_ARM=right` (the left arm alone brings the unit in and lets it
go with no blocks and no fingertip) 0: its bottom edge comes down on the marks. With a top-down grasp neither arm
can do the other's half: near the table the left arm reaches no further right than about y -10 cm at x 30 cm,
short of the blocks at y -22.5 cm, and the right arm no further left than about y +16 cm at x 20 cm, short of the
unit's nearer end at y +19 cm (offline reach maps, `tools/offline/README.md`). The handover needs both: the leaning
unit must be held by one hand while the other takes it. On the GPU (2026-11-14): left arm idle 0.25, right arm idle 0.

## Demo and run limit

The scripted demo is open loop: a fixed list of timed waypoints for each arm, built from `layout.py`; it reads no
object pose or other simulator state. It takes 74.1 s (1,482 control steps); the scripted policy stops at step 1,542
(its 3 s hold). Moves, with control steps:

- Steps 0 to 408, right: grips the far block on the tray across its 30 mm, pitched 60 degrees (pointing forward and
  down), carries it in front of the frame, in under the head 10 mm over the right mark, down over 2 s, lets go
  1.5 mm over the mark, up 3 mm, back out. Left, steps 0 to 170: over the unit, grips its top edge 75 mm left of
  its centre (fingers 16 mm, the TCP 10 mm below the top edge), lifts it 30 mm; then holds it upright out of the way.
- Steps 408 to 890, right: the same for the near block onto the left mark, then parks on its own side and closes
  its fingers. Left: holds.
- Steps 890 to 1080, left: tilts the unit's top 48 degrees toward the room in front of the frame, brings its bottom
  edge in over the blocks 8 mm up, sets it on them over 2 s (commanded 1 mm into them). Right, from step 1020:
  closed fingertip (pitched 55, rolled 35 degrees) to the unit's room face, 75 mm up it and 45 mm right of its
  centre, stopping 0.5 mm off it at step 1110.
- Steps 1110 to 1196, left: holds until step 1116, gives the unit's weight to the blocks (1 s), opens to 30 mm
  (1.5 s), backs out 45 mm along the glass (1.5 s); then goes back over the unit's empty starting spot (to step
  1246). Right: holds the fingertip there until step 1202.
- Steps 1202 to 1352, right: pushes the unit upright about its bottom edge in five 1.5 s steps (48 to 0 degrees),
  the contact sliding from 75 to 45 mm up its face.
- Steps 1352 to 1482, right: down its face to 25 mm up, slides it out against the outside stop (sent 3.5 mm past
  it, so that the stop, not the arm, ends the slide), backs off 45 mm, parks.

On the offline model (`check_task.py`): the 1.0 conditions (everything let go and still for 1 s) hold from
step 1,457, and success holds at step 1,476 of the 1,800-step run limit, 66 steps before the scripted policy
would stop. The run limit is
1,800 steps (90 s).

Why these numbers, from the offline kinematics and capsule model (`tools/offline/yamkin.py`; fingers as 8 x 27 x
38 mm pads; the tests recompute them, `Demo`):

- **Tilt 48 degrees.** The left hand holds the top edge, so the unit's top must be well on the room side of the head
  while its bottom edge goes onto the blocks. At 45 degrees the hand, opening to let go and backing out, comes
  within 16 mm of the head; at 48 it stays 22.6 mm or more clear.
- **Payload sag.** The arm's controller does not carry the unit's weight. The static estimate (J K^-1 J^T of the
  0.20 kg unit's weight and moment at the TCP; joints 80/80/80/10/10/10 N m/rad) with the unit tilted on the blocks
  is large: the TCP 5.6 mm toward the arms, 6.2 mm left, 10.4 mm down, turned 2.7, 2.1 and -0.9 degrees about x, y
  and z. The tilted holding waypoints are commanded that much the other way (`SAG`, `SAG_ROT_DEG`), so that loaded
  the unit is where the plan puts it; once its edge is on the blocks and the fingertip on its face, the command
  returns to the unsagged pose (the blocks take the weight), so the hand does not spring toward the head as it
  opens. Loaded, every pose of the left hand is 23 mm or more from the head and jambs; if there were no sag at all
  the commanded poses would come 10.4 mm from the head (the worst case, while the bottom edge comes in 8 mm over the
  blocks).
- **Set 2.5 mm short of the stop.** The bottom edge is set at x 28.65 cm, 2.5 mm room-side of where it stands
  against the stop: an error of up to 2.5 mm outward in the sag still lets the unit come upright (further out its
  outside bottom corner would come down on the stop's top), and 7.5 mm inward still sets it on the blocks. Pushed
  upright about that edge, the unit stands 2.5 mm from the stop (inside the 3 mm rule) before the last push slides
  it against it.
- **Clearances.** Fingers, gripper body and forearm stay 15 mm or more from the head, the jambs and the outside
  stop at every pose (left hand 22.6 mm, as loaded; right hand 16.9 mm at the right jamb on its way to the frame,
  15.8 mm at the sill's stop setting a block, 27.3 mm from the head). The pads' real gripping faces (27 mm wide at
  the TCP, 15 mm at the tip) setting a block have their lowest corner 2 mm over its bottom, so 3 to 4.5 mm over the
  sill: the blocks sit on it.
- **The arms stay 13 cm apart.** The left hand grips 75 mm left of the unit's centre and the fingertip touches 45 mm
  right of it, the right wrist rolled 35 degrees away: the TCPs are 13.2 cm or more apart at every step, and the
  arms' capsules never overlap (9 mm at the closest, while the fingertip comes in).
- **The leaning unit cannot twist off the fingertip.** On its bottom edge on two blocks and a fingertip, the unit
  leaning 48 degrees could turn about the line through one block's end and the fingertip; with the fingertip 75 mm up
  its face and 45 mm from its centre, every such turn lifts its centre of mass (the test checks all four lines).
- **The push is light.** Holding the unit leaning 48 degrees takes 0.57 N at the fingertip (its weight times its
  centre's 22 mm lean over the edge, over the 75 mm lever), and less as it comes upright; past 17 degrees it falls
  upright by itself onto the blocks. The last push, sent 3.5 mm past the stop's position, adds about 1.4 N through the
  arm's compliance. All well under 8 N.

`check_task.py` warns that the "carried glass_unit runs into head": its own proxy carries a held box upright at the
gripper's offset, which is not where a tilted grip holds it; the task's own model (`offline_unit.py`, which the
checker's scores come from) carries it tilted, and the unit itself stays 7.3 mm or more clear of the frame while
held (commanded poses, no sag).

`BLOCK_TRIM` and `PLACE_TRIM` in `task.py` (zero) shift the blocks' and the unit's set points if a GPU run shows
them landing consistently off; read `block_mark_mm` and `unit_pos` in `TASK_TRACE`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two blocks back flat on the tray,
centred at x 30.0 and 34.0 cm, y -24.0 cm, sides along x and y; the glass unit back standing on a long edge on the
table, centred at x 22.0 cm, y +29.0 cm, faces normal to x. The frame, the marks, the tray and the bead are fixed.

## Runtime per trial

About 2.5 minutes on an L40S once Isaac Sim is running (the demo is about 77 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A small fixed-light frame of 20 x 60 mm aluminium or hardwood sections (sill, two jambs and a head) round a 240 mm
wide, 95 mm tall opening, with a 15 mm outside stop (a fixed bead) screwed round the opening on the far side 37 mm
from the room face, clamped to the bench; a 240 x 12 x 15 mm bead lying beside it; two 30 x 30 mm white labels on
the sill at the quarter points, centred 19 mm from the stop; two setting blocks cut from 8 mm neoprene or EPDM to 30
x 30 mm (about 9 g each) on a small tray; and a 24 x 200 x 80 mm double-glazed unit of about 200 g (for safety,
acrylic panes on a weighted spacer to the same mass). Before each trial rub blue marking chalk on the sill's top and
room face and on the two marks. After the end photo and video, lift the unit out: any blue on it shows it touched
the sill or a mark; contact is the rule in simulation and on the bench alike (an air gap leaves no chalk and is not
a touch). Photograph the opening from above for the blocks' distance from the marks and their turn. Read the other
thresholds as the scoring table says.

Third-party assets: none. The scene is built from the task file's own boxes.

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
