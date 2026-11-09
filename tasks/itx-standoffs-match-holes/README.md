# Set case standoffs to a mini-ITX board's four holes, none elsewhere, then mount it

Before a motherboard goes into a PC case, the builder makes the standoffs on the case's tray match the board's
mounting holes. A standoff under the board where it has no hole presses on the board's underside and can short its
circuits; a hole without one leaves the board unsupported, so it flexes when memory or a card is pushed in. Many cases
ship with standoffs fitted for the largest board they take, so a smaller board means moving some of them onto its
own holes and taking out any that would sit under it elsewhere (a loose one left in the case is the same short
waiting to happen), then laying the board on them with its I/O ports toward the case's rear opening. This is
everyday work for computer repairers and PC builders.

Here a grey case tray on the table has seven threaded holes (dark discs). Brass standoffs stand on five of them: on
the mini-ITX board's two rear holes (H1, H2), on two holes that lie under the board's outline where it has no hole
(A1, A2), and on one outside its outline (A3). The board, with a low CPU cooler and its I/O block fitted, lies on an
anti-static mat beside the tray, its four holes marked by gold pads. The robot has to read the board's hole pattern
from the board itself (it is not a rectangle: the front-right hole is 30 mm nearer the rear than the front-left one,
and A2 sits exactly where a rectangular pattern would put it), move the two standoffs that would be under the board
onto its empty holes (H3, H4) or into the parts dish, and lay the board on the four, each hole over its own
standoff. Any standoff left under the board where it has no hole, or loose anywhere on the tray, scores 0, as in the
real job; otherwise each hole with a standoff earns 0.2 and the mounted board 0.2 more. The demo uses both arms,
each moving the standoff on its own side of the tray; two arms are not required (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the five standoffs' and the board's poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds the board (circuit board, cooler, I/O block and hole pads) as one rigid body and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, the hole, stray and mount tolerances, the hole pattern fitting only one way round, the exploits we tried, resets, invalid states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and cylinders and loose cylinders, the standard `task_score` scorer, and one helper
module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres: a circuit board with a cooler and an I/O block on
it is not. `mechkit.py` (imported by `task.py`) builds the board as one loose rigid body from several shapes (three
boxes and four thin cylinders) through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and
orientation) to the scorer. It changes nothing else the embodiment does, and it stops the run with an error if the
board was not built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a
`sys.meta_path` hook patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner)
follows the one first shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent
implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the case tray ready for the mini-ITX board and mount it: a standoff under each of the board's four holes, none anywhere else under the board and none loose on the tray (leave a spare on a hole clear of the board or put it in the parts dish), then lay the board on them with each hole over its standoff and its I/O block at the rear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/itx-standoffs-match-holes/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the case tray ready for the mini-ITX board and mount it: a standoff under each of the board's four holes, none anywhere else under the board and none loose on the tray (leave a spare on a hole clear of the board or put it in the parts dish), then lay the board on them with each hole over its standoff and its I/O block at the rear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/itx-standoffs-match-holes/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`tier`, `zero_reason`, `strays`, `on_hole` (the standoff on
each of H1-H4), `holes_filled`, `mounted`, `board_over_standoffs`, `board_tilt_deg`, `board_yaw_deg`,
`board_x_mm`, `board_y_mm` and `board_z_mm` (its underside's centre from the mounted pose), `board_hole_offset_mm`
and `board_hole_gap_mm` (each hole from its standoff's top, across and up), `standoff_hole` (nearest tray hole),
`standoff_offset_mm`, `standoff_tilt_deg`, `standoff_floor_mm`, `standoff_seated` (the hole it stands on),
`standoff_over_tray`, `standoff_under_board`, `moved_mm`, `off_table`,
the board's pose and holes, the standoffs' bottom centres, `source`). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the five standoffs' and the board's
poses, once per control step (20 Hz), and keeps each part's last half second of positions to tell a part at rest
from one still moving (two points per part: a standoff's bottom and top centres, the board's rear-left and
front-right hole centres; at rest means neither moved more than 2 mm in the last 0.5 s). The score is read at the
last step; the trial ends early once 1.0 has held for 3 s.

- A standoff **stands on a hole** when it is upright (axis within 10 degrees of vertical, either end down) and the
  centre of its bottom face is within 3 mm of the tray's top and within 5 mm of the hole's centre: its 18 mm base
  lies wholly on the hole's 28 mm disc. A **hole has a standoff** when one stands on it at rest.
- A standoff is a **stray** when it is in the case (its centre, seen from above, over the 180 x 200 mm tray or
  less than 9 mm, its radius, outside the tray's edge) and it does not stand on H1-H4 or on A3, the spare hole
  31 mm clear of the board's outline. That covers one left on A1 or A2 (inside the mounted board's 88 x 88 mm
  outline, centred at x 30.0 cm, y 0, where the board has no hole), one shoved off its hole, one lying down or
  standing loose anywhere on the tray, and one left standing on top of the board. One standing on A3, in the dish
  or on the table off the tray is neither credited nor a stray.
- The board is **mounted** when all four holes have a standoff and the board is level within 3 degrees, each of its
  holes is within 7 mm (the radius of its pad) of the axis of the standoff on its own tray hole (rear left over H1,
  rear right over H2, front left over H3, front right over H4) with its underside there within 2 mm of that
  standoff's top, and it is at rest. Matching each hole to its own tray hole is what puts the I/O block at the rear:
  turned a quarter or half turn, wherever it is shifted, one of the board's holes misses every standoff by 15 mm
  or more (21 mm for a quarter turn).

| Outcome | Score |
|---|---|
| Any standoff or the board off the table (centre outside its outline, or lowest point more than 10 mm below its top) | 0 |
| Any stray left (including the start: A1 and A2 are strays; the board laid on the standoffs as they start; A1's or A2's knocked over or slid onto bare tray) | 0 |
| Otherwise, for each of H1, H2, H3 and H4 with a standoff | +0.2 |
| ... and the board mounted (needs all four) | +0.2 |

Possible scores: 0, 0.2, 0.4, 0.6, 0.8, 1.0, each reached by a plain sequence of moves: A1's and A2's standoffs
and H1's taken to the dish, 0.2; A1's and A2's taken to the dish, 0.4; one of them moved onto H3 or H4 and the
other to the dish, 0.6; both moved onto H3 and H4 (either onto either), 0.8; then the board mounted, 1.0. Which
standoff stands where does not matter (A3's may be used too). A new episode is recognised from mechkit's reset
counter, or, as a fallback, when the simulator puts every part back on its starting mark (within 4.5 cm) with at
least one of them jumping more than 8 cm in a single step. A simulator state with a non-finite number, a zero
quaternion or a board without its base frame scores 0 for that step and leaves the record of the trial as it was.

The stray rule makes the score fall to 0 for a standoff that misses its hole by more than 5 mm, rather than
costing it 0.2. That is the job's own rule: a loose standoff in the case, under the board or beside it, is the fault
the check is for, wherever it is. The 5 mm is what a photo shows: the standoff's base wholly on the hole's disc or
not (in the real job, screwed into its hole or not).

Unintended strategies earn no more than they should, as the tests show: laying the board over the strays, or on
standoffs that A3's filled one hole while A1's and A2's stay, is 0; knocking A1's or A2's over, or sliding it off
its hole onto bare tray, is 0 (it is loose in the case), even with the other holes filled and the board mounted;
four standoffs stood in the board's pattern off the tray with the board on them is 0 (no hole has a standoff);
every standoff cleared away and the board laid on the bare tray is 0; a standoff shoved off its hole by the board,
lying across its hole, stacked on another or left standing on the board is a stray (0); throwing strays off the
table is 0; two standoffs crowded onto one hole fill it once; the board turned round, upside down, shifted, held
10 mm up or laid elsewhere is not mounted (0.8 at most); a board swinging about one hole is not at rest. The scorer
cannot see the grippers, so a board held still in its mounted pose (within 2 mm of the standoffs' tops) counts as
mounted; see `scoring` in `task.json`.

## One arm alone

The scripted demo scores 0 with either arm idle, offline (the kinematic check with `--one-arm`) and on the GPU
(0.0 and 0.0, see 'Results in simulation'): each arm moves only the stray on its own side, and the other stray is
left under the board. On the GPU, with the left arm idle the right arm moved A2's standoff onto H4 and laid the
board in its mounted pose, but A1's standoff was still standing under it (three holes filled, one stray: 0); with
the right arm idle the left arm moved A1's standoff onto H3, A2's stayed on A2 and the board stayed on its mat (0).
That is a property of the script, not of the task. Reach (inverse kinematics of the arm's URDF,
straight-down grips): the right arm reaches every hole on the tray (A1 and H3 with at least 0.5 rad of joint
margin), the board on its mat and the mounted pose, so the right arm alone could score 1.0. The left arm reaches
every hole and the mounted pose but not the board's cooler on its mat (straight down it reaches y -17 cm at x 18 cm,
3 cm short of the cooler's centre), so straight down it can score 0.8; it might fetch the board some other way
(pushing it along the mat, or a leaned grip), and we do not claim it cannot. Two arms are not required. Both arms
are used naturally: each clears the stray on its own side of the tray, and the arm nearer the board mounts it.

## Demo and run limit

The scripted demo takes 40.0 s (800 control steps), the arms taking turns over the tray because the two strays are
only 70 mm apart across the midline. Every close and open of the fingers on a part takes 1.5 s in place; fingers
are commanded 8 mm narrower than the part; every final descent is slow (at most 17 mm/s on average). All grips
are straight down at yaw 0 (fingers closing along y).

- Right arm, A2 to H4 (0 to 14.4 s): comes down over A2's standoff with the fingers 50 mm apart (so the open
  fingers stay clear of the neighbouring standoffs), grips it across its 18 mm with the pad tips 6 mm above its
  bottom, lifts it to 8 cm, carries it 30 mm toward the rear onto H4, lowers it the last 25 mm over 1.5 s to
  1.5 mm above the disc, lets go and goes straight up.
- Left arm, A1 to H3 (waits aside over the parts dish until 15.5 s; 15.5 to 31.4 s): the same moves, A1's
  standoff 35 mm toward the front (the robot) onto H3, once the right arm has left the tray; then aside again.
- Right arm, the board (14.4 to 40.0 s): opens fully on the way to the mat, comes down beside the cooler (pad tips
  8 mm above the board's top, clear of the I/O block along the board's far edge), grips the 40 mm cooler, lifts
  the board to 10 cm, holds it there until the left arm has gone back up from H3, carries it over the tray (its
  underside 52 mm over the standoffs), comes down to 25 mm above the mounted pose, lowers it the last 25 mm over
  2.5 s (underside aimed 1 mm above the standoffs' tops), lets go and goes straight up.

Offline (kinematic check): worst inverse-kinematics error 0.10 mm; smallest joint margin 17.2 degrees (right joint
4); arm links at least 7.2 cm apart (capsule proxy), tool points at least 25 cm apart; pad tips never lower than
10 mm above the table. Success has held for its 3 s at step 795; the scripted policy stops at step 861. On the GPU
the demo scored 1.0 by step 739 and the trial ended at step 792 with success held for 3 s. The run
limit is 1,800 steps (90 s): enough for a policy that also has to look at the board, try a hole and move a third
standoff.

For the GPU run: the right arm sags a few millimetres toward its base under the 0.15 kg board, so its place point
carries a small trim (`BOARD_TRIM` in `task.py`, about half the sag measured in earlier runs, scaled to the board's
mass). The board comes to rest on the standoffs slightly before its slow descent ends; the 3 s success hold (rather
than 1 s) lets the gripper open before the trial ends. Read `standoff_offset_mm` (under 5 mm on H3 and H4),
`board_x_mm` and `board_y_mm` (aim 0), `board_hole_offset_mm` (under 7 mm), `board_hole_gap_mm` (within 2 mm) and
`board_tilt_deg` in `TASK_TRACE`; if the board lands off its aim, move its place point by minus the offset with
`BOARD_TRIM`. Measured at the end of the GPU demo run: H3's and H4's standoffs 1.1 mm from their holes' centres and
within 0.1 degree of upright; the board 0.4 mm (x) and 0.8 mm (y) from its mounted pose, level within 0.1 degree,
turned 1.2 degrees, its holes 0.8 to 2.3 mm from their standoffs' axes and resting on their tops.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the five standoffs upright, centred on H1
(x 33.5 cm, y +3.5 cm), H2 (x 33.5 cm, y -3.5 cm), A1 (x 30.0 cm, y +3.5 cm), A2 (x 26.5 cm, y -3.5 cm) and A3
(x 33.5 cm, y -7.5 cm), H3 and H4 empty; the board flat, centred on its mat at x 18.0 cm, y -20.0 cm, its I/O
block toward +x. The tray, its hole discs, the mat and the parts dish are fixed. Nothing is consumed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is 40 s of simulated time); a Dream-1 trial to the
full 1,800 steps took about 4.5 minutes. The first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 180 x 200 mm sheet of 3 mm steel or aluminium (or a painted board) clamped 30 cm in front of the arm bases, with
seven 28 mm dark grey vinyl discs at the hole positions. Five standoffs: 18 mm brass or aluminium tube, 20 mm long,
capped (about 20 g each; a 3D-printed part painted brass works too). The board: an 88 x 88 mm offcut of 3 mm
circuit board or green acrylic with a 14 mm hole drilled through it at each hole position (ringed in gold), a
40 x 40 x 30 mm aluminium or printed block glued at its centre and a 12 x 48 x 15 mm silver block along one edge
(about 150 g in all). A 110 x 110 x 5 mm pink foam mat and an 80 x 80 mm black dish or mat. Score from a photo from
above (each hole's standoff on its disc, a standoff's centre within 5 mm of the disc's centre; each board hole
showing its standoff's top centre within its 14 mm hole), then lift the board off and photograph the tray for
strays (any standoff over the tray that is not standing on H1-H4 or A3). The parts are light and nothing is
powered, so trials are safe.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

Dream-1 ran to the 1,800-step limit: A1's and A2's standoffs were still standing on their holes under the board's
place (strays), so it scored 0; the board was nudged about 1.5 mm on its mat and nothing left the table. The
settling run (no arm moving, 160 steps) moved no part by more than 2 mm.
