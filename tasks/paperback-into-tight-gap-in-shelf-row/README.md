# Shelve a tall paperback into a 40 mm gap in a packed shelf without disturbing the row

Almost everyone has squeezed a book back into a full shelf. The gap is a few millimetres
wider than the book, so it is brought in square by its top, slid in until its spine is level with its neighbours'
spines, and let go of, without leaning on the neighbours. Here the top shelf of a small open bookcase (a shelf board,
two side panels and a back panel, all fixed) holds six upright paperbacks A to F from panel to panel, except for a
40 mm gap between C and D. A taller, thicker red paperback (30 mm thick) stands on the table beside the case, as on
a book truck. The left arm picks it up by its top, above the neighbours, carries it in front of the gap, slides it in
until its back meets the back panel, sets it down, lets go and backs the open hand out. The scorer reads the poses of
all seven books and both grippers: the new book must be seated (square, centred, spine level), the six neighbours must
be upright and in place, no book may ever have fallen (the new book may be tilted while a gripper holds it), and both hands must be off with no book visibly moving (under 5 mm and 5 degrees) for the last
second. The right arm is not needed (see One arm alone).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the books' states, with a small tracker for the whole trial |
| `rigstate.py` | Reads both arms (pad-tip points, openings) for the let-go check; wraps the embodiment's step and reset from `task.py`'s import and changes nothing else |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, visible-motion stillness, NaN states and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (loose and fixed boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm does the job): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the loose red paperback upright into the gap in the row on the shelf, spine out and square, with its spine level with the other spines. Leave every other book upright and where it is, and take both hands off." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/paperback-gap/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the loose red paperback upright into the gap in the row on the shelf, spine out and square, with its spine level with the other spines. Leave every other book upright and where it is, and take both hands off." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/paperback-gap/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module, and recognises a new
episode in the same process when the embodiment resets (`rigstate`) or the simulator puts every book back on its marks
in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check: nothing may move); `TASK_TRACE=trace.jsonl` writes one
JSON line per control step with what the scorer saw and decided (`new_y_off_mm`, `clear_c_mm`, `clear_d_mm`,
`new_spine_off_mm`, `new_tilt_deg`, `new_yaw_deg`, `row_max_tilt_deg`, `row_max_shift_mm`, `hands_gap_mm`,
`book_minus_tcp_y_mm`, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every book is a plain box, so its faces follow from its pose.
Definitions: the *underside* is the centre of a book's bottom face (a tilted book is read where it stands); the *spine
line* is the mean x of the bottom edges of C's and D's spines; the *gap line* is midway between C's and D's
undersides; *clear* is the distance from a neighbour's face to the new book's bottom corners (5.0 mm each side when
the book is centred). The scorer keeps each neighbour's largest tilt over the trial and the new book's largest tilt while it is not held, whether the new book ever dropped off the table, the last second of poses and the last
second of let-go checks (`rigstate.released`: both grippers' finger zones, the 38 mm tip segment with a radius from
15.7 mm closed to 56.6 mm open, at least 30 mm from every book; a gripper that cannot be read is not clear).

| Outcome | Score |
|---|---|
| A neighbour tilted over 45 degrees at any moment, or the new book tilted over 45 degrees at any moment while not held; the new book's lowest point ever more than 10 mm below the table top (dropped off the table, or below the shelf); a neighbour ends off the shelf; or hands not off, or a book moving 5 mm or turning 5 degrees, in the last second | 0 |
| New book seated (underside within 3 mm of the shelf top, centre within 3 mm of the gap line, every bottom corner 2 mm or more clear of C and of D, tilt under 3 degrees, turned under 3 degrees, spine within 5 mm of the line) and A to F upright (end tilt under 5 degrees, never over 20, each within 6 mm of its place) | 1.0 |
| New book in the gap (all four bottom corners between C and D, underside within 3 mm of the shelf, tilt under 10 degrees, turned under 15, spine within 40 mm of the line) and A to F upright | 0.5 |
| New book in the gap (as for 0.5) with the row disturbed: every neighbour under 20 degrees at the end and within 20 mm of its place | 0.25 |
| Anything else | 0 |

'Held' (the new book only): a gripper's finger zone within 10 mm of it with its pads less than 75 mm apart (the 30 mm book fits between them, an open hand at 94 mm does not). A new book tilted over 45 degrees while held does not zero the run (it may be tipped in the hand on the way), as long as it is seated, upright and let go of at the end; a new book tilted that far with no gripper holding it has fallen, and stays latched even if stood up again. Unreadable arms mean not held. The six neighbours are never held: one tilted over 45 degrees at any moment scores 0 for good. The new book starts on the table, 12 mm below the shelf top; only a lowest point more than 10 mm below the table top latches 'off the table'.

Success is 1.0 held a further 1 s (`SUCCESS_HOLD_S`), on top of the scorer's own last second of let-go and visible stillness (no book's centre moves over 5 mm or turns over 5 degrees over that second). The GPU demo's 1.0 was measured under the stricter 0.5 mm / 0.3 degree check; this check is looser, so it holds.
Otherwise the trial is scored as it stands at the run limit.

The 40 mm gap itself limits how far the book can be turned: a 30 mm thick, 105 mm deep book turned by an angle a needs
30 cos a + 105 sin a of width, so it cannot be turned more than about 5.5 degrees inside the gap; the 15 degree
limit of the half-credit tier matters for a spine-in book (turned 180 degrees), which the scorer refuses: a plain
box is symmetric, so the heading is what says the spine faces the robot.

Unintended strategies earn nothing: keeping the book in the gripper (any hand within 30 mm of a book: 0); turning it
round (spine in: not seated, not in the gap: 0); standing it on the tops of C and D, or turned to bridge them (not on the shelf: 0); shoving the
neighbours apart to widen the gap (the gap is measured from the neighbours as they are, but each of A to F must stay
within 6 mm of its place for 1.0 and 0.5, within 20 mm for 0.25); taking a neighbour out and using its place (that
neighbour is off its place: 0 or 0.25 at most); knocking a neighbour past 45 degrees and standing it up again (the
tilt record is kept: 0); letting the new book go while it is tipped past 45 degrees and standing it up again (not held: latched, 0); dropping it off the table or through the shelf (latched, 0); dropping it from above (fell, or not still: 0). The row fills the case from panel to
panel, so there is no other free space on the shelf. A held book that rests exactly on the shelf scores 0 until the
hands are 30 mm clear for a full second.

## One arm alone

Only the left arm works (the gap is on its base line, x 265 to 370 mm, y 285 to 325 mm). `DEMO_IDLE_ARM=right` is the
demo as written: 1.0 on the GPU (offline 1.0). `DEMO_IDLE_ARM=left` (the right arm alone) scores 0.0 on the GPU: the
right arm cannot reach the book or the gap. The task does not claim that it needs two arms.

## Demo and run limit

The scripted demo takes 25.3 s (507 control steps; the scripted policy stops at step 567, success is predicted at
about step 532). Every final descent is a separate 1.5 s move, and every gripper open or close takes 1.5 s:

1. out in front of the case, then over the book (open pads 7 mm over its top), down either side of its top, grip the
   top 27 mm of its 185 mm with the fingers commanded to 22 mm (8 mm under its 30 mm thickness), above the neighbours'
   tops (pad tips 10 mm over them);
2. lift it to shelf height (underside 12 mm over the shelf top), carry it 130 mm across to the gap line at x 170 mm
   (its back 12 mm short of the end panels' front edges), settle;
3. slide it 151 mm in along +x at 35 mm/s until its back meets the back panel (the command goes 4 mm further, so it is
   pushed back against the panel, spine on the line), settle;
4. lower it 15 mm onto the shelf (3 mm more than it needs, so it carries no weight on the arm), settle, open to 44 mm,
   rise 6 mm and back the open hand out 151 mm along -x, its finger zone then about 62 mm clear of the spines (30 mm needed).

The run limit is 1,200 steps (60 s): more than twice the demo including the let-go second.

The pads grip the book's top because the neighbours' tops (148 mm) are 37 mm below it: the fingers (6 mm thick, outer
face 21 mm from the book's side) would otherwise stand in the 5 mm clearance. The arm sags under the 0.26 kg book
(estimated 3 to 8 mm), so the commanded carry height keeps the underside 12 mm over the shelf (its front edge clears the board's front edge even with 7 mm of sag) and the set-down is
3 mm past contact. `TASK_TRACE` reports `new_underside_mm`, `book_minus_tcp_y_mm` (the book's y minus the gripper's y:
a constant offset here moves the book off the gap line, correct it with `Y_TRIM` in `task.py`) and `clear_c_mm`,
`clear_d_mm` (each must stay above 0 on the way in).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: A to F upright from panel to panel (0.5 mm from
each panel) with the 40 mm gap between C and D; the new book upright on the table with its underside at z 0, centred
at x 170 mm, y 435 mm, spine toward the robot. Nothing is consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 26 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A shelf board (12 mm melamine chipboard, 135.5 x 191 mm), two 18 mm pine side panels 240 mm tall and a 12 mm back panel
(clamp or screw them together, the board between the side panels); six used mass-market paperbacks of about
105 x 148 x 25 mm and 0.16 kg (choose by size, or pad thin ones with card), and a taller trade paperback of about
105 x 185 x 30 mm and 0.26 kg. Bench method for every threshold:

| Threshold | Method |
|---|---|
| Clear each side (2 mm or more for 1.0, 5 mm centred; over 0 for the corners in the gap) | feeler gauges or a rule between each neighbour's face and the book's bottom edge |
| Tilt of the new book at the end (under 3 degrees, under 10) | a try square against its side, or the lean of its top over its 185 mm height (3 degrees is 9.7 mm, 10 degrees 32.6 mm) |
| Turned under 3 degrees (spine out; 15 for 0.5) | the spine's two ends against a straightedge laid across the neighbours' spines: 3 degrees is 5.5 mm over its 105 mm depth |
| Spine within 5 mm (40 mm) of the line | the straightedge across the spines of C and D, and a rule |
| A to F upright (5 degrees at the end, never 20) | a try square on each book (5 degrees is 12.9 mm lean over 148 mm); 20 degrees from the side video |
| A to F within 6 mm (20 mm) of their place | a pencil outline of each book's place on the shelf board and a rule |
| Fallen (45 degrees: a neighbour at any time, the new book when no gripper is within 10 mm of it with the pads under 75 mm apart), off the shelf | the side video and a 45 degree wedge; a 75 mm spacer between the pads; a book not standing on the board |
| New book off the table (lowest point 10 mm below the table top) | a 10 mm block at the table edge and the side video |
| Underside within 3 mm of the shelf top | a feeler gauge under the book's front edge (a book held up shows a gap) |
| Hands off for the last second | a 30 mm spacer block between each hand and the nearest book, and a 30 fps video of the last second |
| Still (no book's centre moves over 5 mm or turns over 5 degrees over the last second) | the last second of video: nothing visibly moves; a 5 mm pencil mark beside each spine on the shelf board and a 5 degree wedge held against the new book |

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, both arms hold still) | 0.0 |
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 (`--policy dream`, full limit) | 0.0 |
