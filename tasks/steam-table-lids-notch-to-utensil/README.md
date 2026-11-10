# Cover the steam-table pans: notched lid over each utensil, solid lid on the others

Cooks and food preparation workers keep hot food in steam tables at 57 C (135 F) or above (US FDA Food Code
3-501.16) and cover the pans between servings. A serving utensil in use may stay in the food with its handle above the
food and the container (Food Code 3-304.12(A)), so steam-table pan covers are sold notched: the handle comes out
through the notch at the server's side and the cover sits flat. A pan with no utensil gets a solid cover, since a
notched one leaves an open slot. The two everyday mistakes are both visible and measurable: a solid cover propped
open on a ladle's handle, and a notched cover turned the wrong way so its plain end rests on the handle. Here a 1:5
steam-table section stands in front of the arms: four full-size pans in a grey deck, a ladle (black handle) in the
chilli and a slotted spoon (red handle) in the sweetcorn, the rice and the green beans without. Four lids lie
crosswise on the counters beside it, a notched and a solid one on each side. Each lid needs a quarter turn to lie
along its pan, and which way it turns decides whether a notch ends up at the handle or at the back. The score is what
a top photo and a ruler at each lid's corners show: 0.25 per pan with the right lid lying flat on it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the lids' poses (and the utensils'), with a small tracker of the lids' last half second |
| `mechkit.py` | Helper that builds each lid as one rigid body of boxes (a plate, or a plate with a notch, and a handle bar) and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, invalid states, resets, the exploits we tried, the lids' geometry against the utensils, the demo plan and the arms' reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (the pans, food, deck and utensils), the standard `task_score` scorer, and one helper
module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a single box cannot have a notch.
`mechkit.py` (imported by `task.py`) builds each lid from boxes (a solid lid: a plate and a handle bar; a notched
lid: three plate pieces round the notch and the bar) as one loose rigid body through a hook on `yam_arena.scene`, and
reports its pose (base-frame origin and orientation) to the scorer. It changes nothing else the embodiment does, and
it stops the run with an error if a lid was not built as a compound body. The technique (a `task_file.Box` subclass
the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's import, pxr authoring in a
custom spawner) follows the one first shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an
independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cover the four pans on the steam table. A pan with a serving utensil in it gets a notched lid, turned so the utensil's handle comes out through the notch and the lid sits flat; a pan without a utensil gets a solid lid. Leave the utensils where they are." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/steam-table-lids-notch-to-utensil/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cover the four pans on the steam table. A pan with a serving utensil in it gets a notched lid, turned so the utensil's handle comes out through the notch and the lid sits flat; a pan without a utensil gets a solid lid. Leave the utensils where they are." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/steam-table-lids-notch-to-utensil/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `zero_reason` (why the score is not higher),
`moved_mm` (each lid's corner travel over the last 0.5 s), `utensil_moved_mm`; per lid `kind`, `pan` (the pan it is
over), `centre_off_mm`, `heading_deg`, `axis_err_deg`, `notch_err_deg`, `tilt_deg`, `corner_gap_mm` (its corners'
largest distance from the rim's height), `flat_on_pan`, `at_rest`, `seated`; per pan the lids seated on it, the right
ones among them and its credit; each lid's origin and orientation). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over at every reset (mechkit's reset counter, or the lids
teleported back to the counters), so run one trial per process or rely on the reset; the run rate is assumed to be
20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four lids' poses (and, for the zero
rule, the utensils' parts) once per control step and keeps only each lid's four plate corners over the last 0.5 s
and where each utensil part was at the trial's first step (a utensil's move is measured from there, so a constant
offset in how the simulator reports a fixed part can never read as a move).
A lid is **seated** on a pan when its plate lies flat (tilted 3 degrees or less), every corner of the plate's
underside is within 2 mm of the rim top (30 mm up), the middle of its plate is within 8 mm of the pan's centre seen
from above, its long side is within 6 degrees of the pan's, and no corner moved more than 2 mm over the last 0.5 s.
The **right lid** for pan 1 or 3 (a utensil) is a notched lid with its notch at the front, where the handle comes out
(its plain end's heading within 6 degrees of straight away from the arms); for pan 2 or 4, a solid lid either way
round. Either notched lid may go on either utensil pan and either solid lid on either plain pan.

| Outcome | Score |
|---|---|
| Any part of either utensil more than 5 mm from where it was set (moved, lifted; fixed in the simulation) | 0 |
| Otherwise, for each pan with the right lid seated on it | + 0.25 |
| A pan left uncovered; its lid not seated (propped on the handle, pushed off centre, tilted, hovering, on another lid, moving); a notched lid on a pan without a utensil | + 0 for that pan |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0 (the tests set every one of the 16 subsets of right lids and see all five).
Examples: the start 0; all four right 1.0; one utensil pan left uncovered 0.75; one side's two lids swapped (the solid
lid propped on the ladle, the notched lid on the rice) 0.5; only one pan right 0.25. The trial ends early once 1.0 has
held for 3 s; otherwise the score is read at the last step.

Why the numbers: on a centred lid with its notch at the front, the stem (10 mm wide) rises through the notch (26 mm
wide, 25 mm deep) with 8 mm to spare either side and 12 mm to the notch's end, and the handle runs out above it: the
8 mm centring matches that play. A solid lid, or a notched one turned round, comes down on the handle (34 mm above
the rim) and stands propped about 20 degrees, its front about 35 mm up: far outside the 2 mm. A lid pushed back far
enough to lie flat past the stem has to move 13 mm or more, beyond the 8 mm centring, so no wrong lid can be seated.
A lid on top of another lid is at least 3 mm higher (17 mm where it meets the lower lid's handle bar), beyond the
2 mm. The 6-degree turn allows a slightly skewed but flat lid (a 6-degree skew moves the notch's middle 4 mm
sideways, inside the 8 mm play). The scorer sees only the lids, so a lid still in a gripper counts as it lies.

Unintended strategies earn nothing, as the tests show: a solid lid pushed 14-15 mm back to clear the handle (off
centre), a notched lid on a plain pan (an open slot), a notched lid turned backwards even if it were somehow flat
(the notch must face the front), a lid stacked on another to "cover" a second pan, a lid on the deck between two pans
(42.5 mm from either centre), a lid held hovering 5 mm up, a lid upside down on its bar, a lid still moving or turning,
and moving a utensil out of the way (0 for the whole trial). In 300 random placements of one lid (any pan, up to 30 mm
off centre, any turn) with the other three right, the score is 1.0 exactly when that lid is within the rule on its
pan, and 0.75 otherwise. A simulator state with a non-finite number, a zero quaternion or a lid without its frame scores 0 for that
step and leaves the scorer's record as it was.

## One arm alone

Each arm can cover the two pans on its side with the two lids on its counter, so a single arm scores 0.5: the
offline kinematic check predicted 0.5 for each arm alone, and the GPU runs measured 0.5 with the left arm idle (the
right arm covers pans 4 and 3) and 0.5 with the right arm idle (the left arm covers pans 1 and 2). With straight-down grips, neither arm reaches anything on the
other side at |y| 20 cm or more (inverse kinematics of the arm's URDF with joint limits, x 20-40 cm, 5-80 mm up, every
30 degrees of wrist turn), and the other counter's lids begin at 21.7 cm, so an arm working straight down has two lids
for four pans: 0.5. Leaned over, though, an arm reaches further across: leaned 30 degrees the left arm's pad tips reach
y -28 cm at x 25 cm, leaned 45 degrees y -34 cm (the same IK with a collision proxy, no grasp check), which includes
the right counter's near lid and its handle bar. So one arm with a steeply leaned grip could in principle fetch a lid
from the other side too: the task does not claim to need two arms. Its natural division of work is one arm per side,
both at once, as in the demo, with the two middle pans taken in turn.

## Demo and run limit

The scripted demo takes 45.3 s (906 control steps), both arms at once. Each lid is gripped across its 20 mm handle
bar at the bar's middle (fingers commanded 12 mm, over 1.5 s; the tool point 21 mm above the plate's underside, pad
tips 8 mm up, 5 mm above the plate), so the wrist's yaw is the heading of the lid's long side; it is lifted to carry
height (the plate's underside 104 mm up, 40 mm over the handles), carried over its pan while the wrist turns it a
quarter turn in two halves, lowered to 45 mm above the rim, set down over the last 45 mm in 2.5 s, let go 0.5 mm
above the rim by opening to 70 mm (the pads stay 9 mm short of a neighbouring lid's edge), and the hand goes straight
up. Left arm: the notched lid onto pan 1, turned +90 degrees so its notch comes from facing +y to the front, round
the ladle (t 0-17.0 s); then the solid lid onto pan 2 (t 17.0-34.0 s), and aside. Right arm: the solid lid onto
pan 4 (t 0-17.0 s, the two grippers 25.5 cm apart while both set down); then it picks the notched lid and holds it
over its counter until the left arm has let go on pan 2 and risen (t 34.0 s), and sets it on pan 3, turned
-90 degrees so its notch comes from facing -y to the front, round the slotted spoon (t 34.0-43.8 s), and aside.
Turning the other way is equally reachable (the same grip on the symmetric bar with the wrist turned half round;
tested), and puts the notch at the back.

Offline: worst IK error 0.087 mm, joint limits at least 12.4 degrees away (joint 4), the tool points at least
25.5 cm apart and the arms' links at least 8.5 cm apart (capsule proxy). Success has held for its 3 s at step 879
and the scripted policy stops at step 967. The 3 s stillness (rather than the usual 1 s) is longer than the time
from the last lid's first rest on its pan until the fingers are open, so a trial cannot end while a gripper still
holds a lid. If a lid lands off its pan's centre on the GPU, read its origin in `TASK_TRACE` and adjust `TRIMS` in
`task.py` (8 mm of tolerance; 8 mm of notch play either side of a handle). The run limit is 2,400 steps (2 minutes).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the four lids flat and crosswise (long side
along y) beside the deck: the notched lid centred at x 37.0 cm, y +27.0 cm with its notch toward +y and a solid lid
at x 25.0 cm, y +27.0 cm; a solid lid at x 37.0 cm, y -27.0 cm and the notched lid at x 25.0 cm, y -27.0 cm with its
notch toward -y. The pans, food, deck and utensils are fixed (in a real trial, check each utensil is still clipped
where it was set).

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is 45 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

Four 106 x 65 x 30 mm open steel tins (or 3D-printed pans with 3 mm walls) as 1:5 full-size pans, filled to 8 mm below
the rim with coloured filler (dark red-brown, white, yellow, green), set side by side 20 mm apart in a 30 mm board
with four cut-outs, or with 20 mm wide, 30 mm tall strips between, beside, in front of and behind them, flush with the
rims; fix everything to the bench. Two utensil stand-ins (3D-printed or bent strip, black and red): a 24 x 24 mm bowl on
the food near the front wall, a 6 x 10 mm stem rising to 26 mm above the rim, and a 10 x 8 mm handle from the stem's
top out over the front rim, 45 mm past it; clip each to its pan's front wall below the rim, out of a lid's way. Four
lids cut from 3 mm light grey PVC sheet, 106 x 65 mm, two with a 26 mm wide, 25 mm deep notch in the middle of one
short end, each with a 50 x 20 x 14 mm black PVC bar glued along the middle of its top (about 46 g notched, 48 g
solid). Score from a photo taken straight down (which lid on which pan, which way each notch faces, centring against
marks on the deck) and a ruler from the deck at each lid corner; mark where each utensil's handle sits on the deck's
front edge.

## Nearby tasks

On the board, TK-F6NJUD fills a steam-table well with pans (the same equipment; the act there is packing pans into a
well), TK-2BHG5K puts each round lid on the tin of its own diameter, and TK-XEY5FU puts identical lids on three lunch
boxes. Here the pans are already in, there are two kinds of cover, which one a pan needs depends on whether a utensil
stands in it, and the notched cover only sits flat turned one of two ways, chosen by the turn made while carrying it.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
