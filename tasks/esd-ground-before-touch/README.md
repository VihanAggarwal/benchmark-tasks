# Unpack circuit boards to an ESD mat: ground each gripper, keep the foam cup away

Electronics assemblers handle static-sensitive circuit boards under an electrostatic discharge (ESD) control
programme: whoever handles a board is grounded first, boards are set down only on a static-dissipative mat or kept
in their shielding packaging, and insulators the job does not need, such as a foam coffee cup, are kept at least
30 cm (12 in) from any exposed board. A discharge too small to feel can damage a board without any visible sign, so
the order of the steps matters as much as where the boards end up. Here a green circuit board lies in a silver
shielding box in front of each arm, a grey ESD mat lies between the arms with a white foam cup standing on it, a
gold ground stud stands behind the mat, and a yellow parking square marks the back right corner of the table. Each
gripper must close on the stud before it touches a board, and again after it touches the cup (an insulator, which
can charge it). The cup has to be moved to the parking square before any board comes out, and the boards may rest
only in their boxes or on the mat. The scorer watches both grippers, both boards and the cup at every control step,
not only the end state.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the boards', the cup's and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and upright cylinders) and the standard `task_score`
scorer.

The scorer's gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
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
  --instruction "Move both green circuit boards from their silver shielding boxes onto the grey ESD mat. A gripper may touch a board only after closing on the gold ground stud, and must close on it again after it touches the white foam cup. Before taking a board out, move the cup onto the yellow square at the back right: it must stay at least 30 cm from every board that is out of its box. Boards may rest only in their boxes or on the mat; set them down flat and apart, and do not knock the cup over." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/esd-ground-before-touch/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move both green circuit boards from their silver shielding boxes onto the grey ESD mat. A gripper may touch a board only after closing on the gold ground stud, and must close on it again after it touches the white foam cup. Before taking a board out, move the cup onto the yellow square at the back right: it must stay at least 30 cm from every board that is out of its box. Boards may rest only in their boxes or on the mat; set them down flat and apart, and do not knock the cup over." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/esd-ground-before-touch/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `grounded`,
`first_ground_step`, `groundings`, `cup_contacts`, `cup_rule_kept`, `cup_rule_step`, `min_cup_to_out_board_mm`,
`min_ungrounded_to_board_mm`, `stud_distance_mm`, `on_mat`, `overhang_mm`, `bare_corner_mm`, `boards_apart_mm`,
`cup_tilt_deg`, `opening`, `arms_unavailable`, both TCPs, ...). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over at every reset, so run one trial per process or
rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer cannot see which way a gripper is turned, so
each gripper's fingers count as a round zone about its axis: a flat-ended cylinder from the finger tips to 38 mm
back up the gripper, out to the fingers' outer corners (57 mm fully open, 40 mm closed on a 60 mm board). A gripper
*touches* something when its zone comes within 5 mm of it; a gripper hovering above a board is only as close to it
as its finger tips are.

- **Grounding.** A gripper becomes grounded when it closes on the stud: its tool point within 15 mm of the stud's
  axis (horizontally), its finger tips at least 5 mm below the stud's top, its opening 0.40 or less (fingers at
  most 37.6 mm apart; closed on the 20 mm stud they stop at about 0.21). Fingers left open round the stud, or
  closed beside it, do not ground. It stays grounded until it touches the cup, which charges it again.
- **Out of its box.** A board is out unless its centre is over a box's floor and its lowest point is below the
  box's 25 mm rim.
- **The cup rule** is kept if, at every step, the cup's outline seen from above was at least 30 cm from the
  outline of every board that was out of its box.

Four rules hold for the whole trial; one broken at any moment scores 0 whatever follows:

1. a gripper touches a board while it is not grounded;
2. a board touches the bare bench: a corner less than 1 mm above the table top (or below it, off the table) over
   neither the mat nor a box's floor, or the board rests on the yellow parking square (its lowest point within
   3 mm of the square's top while their outlines overlap: the square is bench, not a place for boards);
3. a board tilts more than 30 degrees (dropped, stood on edge);
4. the cup tilts more than 30 degrees from upright, or leaves the table.

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Both boards flat and wholly on the mat, at least 20 mm apart, both grippers let go, the cup rule kept | 1.0 |
| Both boards flat on the mat within 15 mm of its edge, the cup rule kept (overhanging, too close, or still held) | 0.75 |
| One board flat on the mat (within 15 mm), the cup rule kept; or both, with the cup rule broken | 0.5 |
| No board on the mat, the cup rule kept, both grippers grounded and the cup at least 30 cm from both boards | 0.25 |
| Anything else | 0 |

'Flat on the mat' means tilted less than 5 degrees (either 120 x 60 mm face down) with the lowest point within
3 mm of the mat's top. A gripper has let go when it is open (opening 0.9 or more, fingers at least 84.5 mm apart)
or its zone is more than 10 mm from both boards, so a gripper closed again once it is clear changes nothing.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both boards still
for 1 s. A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts both
boards and the cup back in their starting places (within 4.5 cm) in a single step. A simulator state with a
non-finite number scores 0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing, as the tests show: touching a board before grounding, even to put it back,
breaks rule 1 (0); grounding, moving the cup and then going straight to the board breaks rule 1 (0); hovering open
round the stud, half-closing round it, or closing beside it does not ground; dragging a board over its box's wall
or setting it on the bench to regrip breaks rule 2 (0); a board dropped onto its edge breaks rule 3 (0); shoving
the cup off the table or knocking it over breaks rule 4 (0). Leaving the cup on the mat, pushing it only a few
centimetres aside, parking it 20 cm behind the mat, or carrying a board past the parked cup breaks the cup rule:
at most 0.5. Stacking one board on the other scores 0.5 (only the lower one is on the mat).

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) parks the cup, grounds, and unpacks the right board: 0.5 (the left
board stays in its box). `DEMO_IDLE_ARM=right` (the left arm alone) grounds and carries its board to the mat with
the cup still standing there, which breaks the cup rule with one board on the mat: 0. Each shielding box is reachable
only by the arm in front of it (41 cm forward, 32 cm out to its side; the other arm's reach ends about 10 cm past
the midline), so one arm can never unpack both boards. A one-arm policy that moved the cup to a far corner on its
own side first could still unpack its own board: at most 0.5. (Offline kinematic predictions; the GPU runs are
recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 41.3 s (826 control steps). Every grip closes the fingers along y over 1.5 s, commanded 8
mm narrower than the part (on the stud 4 mm narrower, over 1.2 s). The right arm grips the cup across its 50 mm
with the finger tips 25 mm above its bottom, lifts it, carries it with its bottom 92 mm up to the yellow square,
lowers it to 2 mm above the square and lets go (13.7 s). Meanwhile the left arm waits beside its box, then comes
down round the stud with the finger tips 15 mm below its top, closes on it (grounded at about 11 s), opens, goes up
(14.4 s), and comes down into its box (the open fingers 15 mm inside the side walls) and grips its board across its
60 mm with the finger tips 5 mm above the board's bottom. Once the left arm is off the stud, the right arm, not
grounded yet and fresh from the cup, grounds on the stud in the same way (up again at 23.8 s), then grips its own
board. The left arm holds its board 72 mm up over its box until the right arm is off the stud, carries it to the
mat's left half, lowers the last 2 cm over 1.5 s, lets go 2 mm above the mat and moves aside (33.3 s); only then
does the right arm bring its board to the mat's right half the same way. The boards end 40 mm apart, 20 mm inside
the mat's side edges. Only one arm is near the midline (the cup on the mat, the stud, the mat) at a time: the TCPs
stay at least 25.4 cm apart and the arms' collision capsules at least 7.0 cm apart. Offline, success has held for
its 1 s at step 816. The run limit is 2,000 steps (100 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: each board back flat and centred in its
shielding box, long side along x, and the cup back upright on the mat at x 30.0 cm, y -5.0 cm; the mat, stud,
boxes and parking square are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 41 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 220 x 200 mm grey static-dissipative mat (cut from bench matting) taped to the bench; a 20 mm brass post,
60 mm tall, screwed upright 4 cm behind it and wired to a continuity buzzer whose other lead clips to each
gripper's fingers in turn (the buzzer sounds when a gripper closes on the post); two open silver boxes with 25 mm
walls round a 140 x 140 mm floor (aluminium tape over card), clamped 41 cm in front of each arm; two 120 x 60 mm
green circuit boards each in a 20 mm deep carrier frame (about 80 g with the board); a small lidded foam cup with
a little cold coffee in it (about 30 g, 50 mm across and 70 mm tall); and an 80 x 80 mm yellow plate in the back
right corner. Lay a 30 cm scale bar along the table's right edge and film from above and from the side to judge
the board, cup and distance rules.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
