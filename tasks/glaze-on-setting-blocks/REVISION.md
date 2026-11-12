# Revision for review TK-GMU4YS (2026-11-12)

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

Done. The category is now "Societal relatability". The context is reworded for it: modern home and office windows
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
(139 characters). The title is now "Stand a glass unit on two setting blocks in a window frame channel": it names
the step the robot completes, since no glazing beads or sealant are fitted. The context says what is withheld. A test checks that neither the instruction nor the title says "then", "first", "before",
"after", "lower" or "straight down".

## Text, so that every threshold can be read in a real trial

- The scoring text ends with how a real trial is read: chalk on the unit's bottom edge (a print on the sill or on a
  white mark is a touch; glass lowered without blocks lands on the 1 mm marks), a ruler and protractor on a top
  photo for the block and centring bands and the tilt, the video for the 30 mm release and the 1 s stillness, and
  a 1 mm feeler gauge between the standing unit and each block. The README's scoring table has the same checks in a
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

- `python3 -m unittest discover -s tests`: 73 tests pass.
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
