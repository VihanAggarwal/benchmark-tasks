# Second review (2026-11-14)

Review of TK-WB6CLU, "Not accepted" (not final, one more revision allowed). The reviewer's text, verbatim:

> Flags: Prior review feedback for the revision was not supplied; verify that the revision addresses it.
>
> Importance (not met): Setting blocks in a double-glazed unit is specialist work rather than a scenario most people know firsthand, but window repair is regular work for general maintenance and repair workers.
>
> Scoring (not met): A chalk print detects contact, not every approach within the scored 2 mm clearance, and the hidden channel makes that clearance difficult to verify throughout a real trial.
>
> What to change: Use Impact on jobs and explain how placing replacement glazing units fits the regular work of general maintenance and repair workers. Make the real and simulated sill-contact rule agree: for example, score actual contact in both and use the chalk transfer to check it, or provide a practical way to detect the specified 2 mm clearance throughout the trial.

## Category and context

Done. The category is now "Impact on jobs". The context now says how placing a replacement glazing unit fits the
regular work of general maintenance and repair workers (about 1.6 million in the US, BLS occupation 49-9071): they
keep buildings in repair, and replacing a broken or misted glazed unit in a window or door is part of that work. The
README's opening paragraph and the keywords are brought in line with it.

## One sill rule in simulation and on the bench

Done. The scorer used to zero a trial when a corner of the glass came within 2 mm of the sill; the bench check, a
chalk print, only shows contact. Both now use contact:

- Simulation: the unit touched the sill when some part of it inside the channel's outline (2 mm slack) is 0.5 mm or
  less above the surface under it, which is a mark's top over a mark and the sill's top elsewhere. 0.5 mm is the
  scorer's resolution for contact (`SILL_TOUCH_GAP` in `layout.py`). The scorer tests the unit's eight corners and
  points every 5 mm or finer along its twelve edges (`touches_sill` in `scoring.py`), not corners alone: a unit
  lowered with no blocks rests on the two 1 mm marks along its bottom edges while its corners hang 1 mm above the
  bare sill, and that must still be a touch.
- Bench: blue marking chalk on all four edges of the unit (the marks are white, so a blue print shows); a print on the sill or on a white mark shows a touch, and a near miss leaves
  none. No clearance has to be measured through the hidden channel. The `scoring` field and the README table say
  the same thing in the same words.
- Nothing changes for the demo, which stays about 9 mm above the sill, and the scripted demo still predicts 1.0.

## Tests added or changed (`tests/test_scoring.py`)

- A level unit resting on the two marks with no blocks is a touch (its corners are 1 mm above the bare sill).
- A bottom edge 0.4 mm above a mark's top is a touch; 0.6 mm is not.
- A tilted unit whose low corner reaches the sill is a touch; 0.4 mm above the bare sill is a touch, 0.6 mm and 1.0 mm
  are not (near misses). A level unit cannot hover over bare sill only, because its bottom edges always pass over
  the marks, so the bare-sill cases use a unit tipped 3 degrees with its low end clear of the marks.
- A unit on the blocks is not a touch; a unit on the table outside the frame is not.
- Edge samples are at most 5 mm apart.
- The momentary-dip test now rocks the unit down to the sill (it used to stop 1 mm short, which the 2 mm rule counted).
- The task.json test pins the new category and the contact wording, and asserts that the `scoring` field no longer
  contains "less than 2 mm above the sill" or "within 2 mm of the sill".

## Later corrections to the second revision (independent check)

- Contact at a mark's outline: the scorer missed a touch when the unit's lowest point over a mark lay on the mark's
  outline between two edge samples (up to 1 mm of error). `sill_gaps` now also takes, exactly, the points where each
  of the unit's twelve edges crosses the vertical planes of each mark's four sides (inside that side) and the unit's
  lowest point straight above each mark corner (gap = z minus the mark's top), besides the corner and edge samples.
  Tests added: the pose pos (0.3, 0.0045, 0.840757), quaternion (0.241922, 0, 0, 0.970296) (xyzw), true gap 0.038 mm,
  is a touch and the same pose raised 0.7 mm is not; a 2 degree roll about the long axis with the bottom edge at zero
  gap on a mark's edge is a touch.
- A step on which the unit's state is valid but a block's is not now still latches a touch before it is scored 0
  (test added).
- Chalk: blue marking chalk on all four edges of the unit, not the bottom edge, so a print shows on the white marks
  and on the sill from whichever edge touched.
- Bench reading: after the end photo and video the unit is lifted straight out and the channel photographed from above;
  block-to-mark distance, block turn and any chalk print on the sill or a mark are read on that photo.
- Bench "resting": the unit stands on its own, let go and still for 1 s, with no chalk print on the sill or marks, so
  only the blocks carry it. This replaces the 1 mm feeler gauge; the simulated rule (within 1 mm of the block's top)
  is unchanged.
- Context, fidelity and instruction: the context says what the worker does on the job and that this task is the last
  step; the fidelity says the frame is cut down to its bottom rail with both stops fixed, so the unit goes in from
  above; the instruction is "Stand the glass unit centred in the frame on the two black setting blocks, set on the
  white marks, and let go." (110 characters; the never-touch rule stays in the context and the scoring). "A unit on
  a block is at least 8 mm up" is now "about 9 mm up" in the README and left out of the `scoring` field.

## Results on the GPU (after this revision)

Re-run on the L40S on 2026-11-14 with this revision's files, one trial each:

- Scripted demo: 1.0 (success ended the trial at step 882 of 1,200; offline prediction 881)
- Demo with the left arm idle (one-arm): 0.25
- Demo with the right arm idle: 0.0 (the glass came down on the marks; touch recorded at step 782)
- Dream-1: 0.0 (ran the full 1,200 steps)

# First review (2026-11-12): how it was answered

## The reviewer's points

> "Construct validity (not met): A robot can hold the unit stationary within 2 mm of both blocks for three seconds
> and earn full credit without actually supporting it on them."

Fixed. The scorer now reads both arms every control step (TCP, pad-tip point and gripper opening, through
`rigstate.py`). The unit counts as resting on a block only when all of these hold:

1. The unit is **let go**: on each of the last 20 control steps (1 s), both grippers' finger zones were 30 mm or
   more from it. A finger zone is every point within r of the 38 mm of the gripper's axis above the pad tips. r is
   35 mm at the demo's 48 mm opening, 57 mm fully open and 16 mm closed. A finger anywhere on the glass, open or
   closed, has therefore not let go.
2. The unit is **still**: each of its last 20 poses is within 2 mm and 2 degrees of the current one. This is a path
   window, not the reported velocity.
3. The unit is **supported**: its underside is within 1 mm of the block's top, down from 2 mm, with at least half
   of the block's top under it.
4. The **block** itself is set the same way: let go and still, lying flat on the sill with its lowest point within
   1 mm of the sill's or a mark's top. The old band allowed up to 3 mm above the sill.

Once the glass is let go, still and within 1 mm of both block tops, nothing but the blocks can be holding it up. A
gripper holding it perfectly still on the blocks, or 0.5 to 2 mm above them, now earns 0.25 at most (for the
blocks), never 1.0, 0.75 or 0.5. The old test that showed the loophole (`test_held_still_within_2mm_counts_as_resting`,
1.0) is replaced by tests that score these cases 0.25.

> "Scoring (not met): The stated resting test and early-success condition do not distinguish a gripper-supported
> or hovering unit from one resting on the blocks."

Fixed in both places.

- **Resting test:** see above. It applies wherever a tier credits the unit as resting: 1.0 (on both blocks),
  0.75 (on both) and 0.5 (on at least one).
- **Early-success condition:** this is now just "the 1.0 conditions have held for 1 s", and those conditions
  already include the unit and both blocks let go and still for the last 1 s. The old 3 s velocity hold, which
  relied on the demo's timing to cover the grip, is gone (`SUCCESS_HOLD_S` 3.0 to 1.0). Success can no longer start
  while a gripper holds the glass. It also no longer uses the reported velocity. On the earlier GPU run that
  velocity flickered: the early-success signal never fired, and the trial was scored 1.0 when the policy stopped.

The `task.json` scoring text, `fidelity` (the sentence "The scorer cannot see the grippers ..." is removed) and the
README give the same definitions and numbers as `layout.py` and `scoring.py`. A test checks the key phrases in
`task.json` against `layout.py`.

> "What to change: Require the glazing unit to be released and actually supported by the blocks wherever a tier
> credits it as resting; update the scorer and tests so a unit held or hovering within 2 mm cannot earn those tiers."

Done (see above). New tests (`LetGo` in `tests/test_scoring.py`):

- held exactly on both blocks: 0.25;
- held 0.5, 1.5 and 2 mm above them: 0.25;
- held off centre: 0.25, and held on one block: 0;
- let go 1.1 mm above the blocks: 0.25, and 0.9 mm: 1.0;
- let go for 19 steps: 0.25, and for 20: 1.0;
- the 30 mm threshold: 29 mm is not let go, 31 mm is;
- an open hand still round the glass's top edge is not let go;
- a gripper still on a block, or holding it 0.5 mm over its mark: not set;
- glass that rested and was let go (1.0), then gripped again in place or lifted 5 mm: 0.25, and after being put
  back it earns 1.0 again only on the 20th let-go step;
- the success signal stays off while the glass is held.

Stillness is tested both ways: 2.1 mm of creep or a 2.1 degree turn in the last 1 s is not still. Velocity jitter
with an unchanged pose is still.

> "Confirm the full-score demo reaches this released state within the run limit."

The demo had to change: after letting go it only went straight up, to 18 mm above the glass, where the open finger
zone still reaches it. It now has one more waypoint after "straight up": back over the unit's empty starting spot
(`RETREAT_L` = x 22.0 cm, y +29.0 cm, z table + 14 cm, the height and place it first came over the unit), over 2 s,
fingers unchanged at 48 mm. Nothing else in the demo changed: the same grips, heights, timings, trims and right-arm
plan as the GPU run that scored 1.0.

Offline (commanded poses, `tools/offline/check_task.py`):

- the left finger zone is 30 mm clear of the glass from step 843 (42.15 s, during the move back) and ends 117 mm
  clear;
- the unit and both blocks are let go and still from step 862;
- success holds at step 881;
- the scripted policy stops at step 923 (plan 43.1 s, then its 3 s hold), well inside the 1,200-step run limit.

`DemoTiming` checks the same thing with the arm 5.6 mm and 15 mm lower than commanded throughout (payload sag or
tracking error). In each case success comes at least 1 s before the policy stops, and the hand ends more than
60 mm clear.

Confirmed on the GPU with the revised task (2026-11-12): the demo scored 1.0. The glass was let go (both finger
zones 30 mm clear) from step 844, the 1.0 conditions held from step 863, and the trial ended on success at step 882
(44.1 s) of the 1,200-step run limit, with the nearer finger zone of the two arms 120.7 mm from the glass at the
end and no step with unreadable arms. Details under "Results on the GPU" below. The trace records `hand_gap_mm`,
`let_go_steps`, `still`, `settled`, `arms_unreadable` and both TCPs, so the released state can be read off it.

> "The task fits societal relatability better than impact on jobs, since glazing is a familiar niche trade."

Done at the time: the first revision moved the category to "Societal relatability" (the second review then asked for
"Impact on jobs": see the top of this file). Its context was reworded for that: modern home and office windows
stand their glass on setting blocks, and a householder replacing a pane in a uPVC or timber window does what a
glazier does; glass whose edge comes down on the sill chips, cracks or loses its seal and then mists up. The O*NET
reference is dropped. The context also names the hardest physical step with its numbers (over 40 mm stops into a
40 mm channel, 8 mm per side, down onto blocks 9 mm above the sill, no corner within 2 mm of it).

## Fail-closed gripper rules (the new review bar)

Every rule that depends on the grippers is "let go", and it fails closed:

- Unreadable arms on a step count as NOT let go for the unit and both blocks, and restart all three counts. This
  covers a missing snapshot, a missing side, a non-finite TCP, pad-tip point or opening, or a malformed arm.
  `arms_unreadable` counts these steps. A trial whose arms are never readable scores 0, not even 0.25.
- A distance that cannot be computed counts as touching (not let go).
- A simulator state with a non-finite number or a zero quaternion scores 0 on that step and restarts the let-go and
  stillness counts. The sill record is kept.

The task has no keep-out zone, so there is no keep-out rule to make fail closed. The "touched the sill" rule reads
only the glass's pose and is unchanged.

## Instruction and title

The instruction now states the goal only, not the order of work or how to lower the glass: "Stand the glass unit in
the frame on the two black setting blocks, set on the white marks, and let go. The glass must never touch the sill."
(139 characters; the second review changed it, see "Second review" above). The title is now "Stand a glass unit on two setting blocks in a window frame channel": it names
the step the robot completes, since no glazing beads or sealant are fitted. The context says what is withheld. A test checks that neither the instruction nor the title says "then", "first", "before",
"after", "lower" or "straight down".

## Text, so that every threshold can be read in a real trial

- The scoring text ends with how a real trial is read: blue chalk on all four edges of the unit (a print on the sill or on a
  white mark is a touch; glass lowered without blocks lands on the 1 mm marks), the unit lifted out
  after the end photo and the channel photographed from above for the block and chalk readings, ruler and protractor
  on the end photo for centring and tilt, the video for the 30 mm release and the 1 s stillness, and, for "resting",
  the unit standing on its own, let go and still for 1 s with no chalk print (the earlier 1 mm feeler gauge clause is
  gone). The README's scoring table has the same checks in a
  column. The finger-zone radii are given in the README only, to keep the scoring text inside the board's limit;
  the rule itself is unchanged.
- The fidelity is in the order the board's accepted tasks use (physics, then "Simplified:", then deviations with
  their reason, then what is not modelled) and states that the blocks are 30 mm long, shorter than usual, so that
  two fit at the quarter points of this 200 mm unit.
- The context no longer says glass "that rests on a single point" cracks, which the 0.5 tier seemed to contradict.
  The README says what the 0.5 tier credits (states a glazier would correct) and that glass let go on one block
  alone tips to the sill and scores 0.
- The README adds "Why it is hard", "What the demo reads" (open-loop, no object state) and "Third-party assets:
  none".

## Checks

- `python3 -m unittest discover -s tests`: 81 tests pass.
- Mutation spot check of the new rules. Each of these is caught by the tests:
  - let-go ignored;
  - unreadable arms counted as let go;
  - the 2 mm gap restored;
  - let go needed on the current step only;
  - stillness ignored;
  - a blow-up keeping the counts;
  - "set" ignoring let-go;
  - the retreat waypoint removed;
  - a 20 mm clearance;
  - the old 3 mm block band;
  - the old 3 s success hold.
- `tools/offline/check_task.py --require-json --strict --one-arm`: PASS, predicted 1.0 (success held at step 881).
  One arm: left alone 0.0, right alone 0.25.
- `tools/field_limits.py`: ok. Title 66, instruction 139, context 905, setup 2501, scoring 3093, fidelity 473,
  reset_notes 306.

## Results on the GPU

The revised task (this `scoring.py`, `layout.py` and demo) on an L40S, 2026-11-12, one trial each,
`-E object_jitter=0 -E control_hz=20`:

| Run | Score | From the trace |
|---|---|---|
| Scripted demo | 1.0 | Success ended the trial at step 882 of 1,200 (offline prediction 881). Glass let go from step 844; 1.0 conditions from step 863; final `hand_gap_mm` (nearer finger zone) 120.7 for the unit, 168.3 and 224.2 for the blocks; `arms_unreadable` 0; `min_low_in_channel_mm` 8.3; unit 3.8 mm off centre, tilt 0.06 degrees, gap to both blocks 0.0 mm; blocks 0.6 and 0.8 mm from their marks |
| Left arm idle | 0.25 | Both blocks set; the glass not moved |
| Right arm idle | 0.0 | The glass came down on the marks (1.0 mm above the sill) at step 782 |
| Dream-1, 1,200 steps | 0.0 | One block moved into the channel, 42 mm from its mark; the glass not moved |

The README's results table, `task.json`'s `scores` and these figures are the same run.

