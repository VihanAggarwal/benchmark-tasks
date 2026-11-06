# Lay out cutting templates on streaky glass: with the streaks, along one edge

Before cutting, a stained-glass maker lays the pattern templates on the sheet and checks the layout, because three
cutting rules decide whether the pieces are usable and how much glass is wasted. Each template's grain arrow must run
parallel to the glass's streaks, or the texture runs the wrong way in the finished panel. Pieces stay clear of the
sheet's rough rolled edge and a few millimetres apart, leaving room for the cutter wheel and the break. And they are
lined up from one edge, so that the rest of the sheet stays one usable strip; pieces cut from the middle of a sheet of
expensive art glass leave two narrow offcuts that are good for little. The check is a top photo and a steel rule.

Here a streaky amber glass sheet (180 x 300 mm, streaks running front to back, across its short side) is taped to the
table, and three white pattern templates lie beside it, each starting at a different turn: A1 across the streaks
(90 degrees off), A2 already with them, and B turned 45 degrees. All three must end flat on the glass, each with its
long side (its grain arrow) within 5 degrees of the streaks, at least 8 mm inside every sheet edge and 5 mm from the
others, lined up along one long edge so that a strip at least 80 mm wide stays clear along the other, with the
grippers let go and away. The streaks run across the sheet's short side, so lining a template up with the sheet's
outline puts it across the grain: the turn has to be read from the streaks, and each template needs a different
turn. A1 and A2 start on the left, B on the right: each arm can reach only its own side's templates, so the right
arm's template has to join the left arm's row. Each arm's work is an ordinary pick, turn and place; the two arms are
needed because of where the templates start (see "One arm alone").

Nearest tasks on the board, and why this one differs. The two closest are TK-9H4KQ6 (three reinforcing bars laid
parallel on bolsters at 100 mm centres with 20 mm end cover) and TK-MGJLVQ (instruments on a sterile drape, kept off
its 1-inch border). First, here the heading is set by a pattern on the surface and is the opposite of what the outline
suggests: the streaks run across the sheet's short side, so a template squared to the sheet's long edges is 90
degrees wrong, while the bars are squared across their bolsters and the instruments have no set heading. Second, the
score depends on where the group sits (the 80 mm clear strip along one long edge, 0.4 of the score), not on a pitch
or an inset alone: the same three templates, turned, inset and spaced correctly, score 0.6 across the middle of the
sheet and 1.0 along one edge. Of the others, TK-89UHVV (biosafety cabinet layout: 10 cm in, rear grille clear, clean
to dirty) has insets and an order, but no orientation and no yield. TK-TM2UVS (filter cartridge with its arrow
downstream) turns one part to a marking on that part; here three parts, each from a different start turn (90, 0 and
45 degrees), turn to the streaks of the surface they lie on. TK-JYVVCS (a row of tiles with 3 mm joints) sets exact
joints against a fixed straightedge; here there is no guide and only minimum gaps. TK-FA22X6 and TK-U3HWFM (keep a
band clear: toddler reach, corridor width) make a band by moving items out of it; here the strip is material yield,
made by packing the pieces to the opposite edge under the grain rule. TK-89FFX3 (kiln pots at least 2 cm apart)
scores spacing only. No stained-glass, grain-direction or material-yield task is on the board.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the sheet's 12 strips and the three templates, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the templates' poses and the grippers' finger zones, with a small tracker (each template's path over the last 0.5 s, the reset) |
| `rigstate.py` | Reads both grippers' finger-tip points, axes and openings for the scorer each control step (copied unchanged from the kit) |
| `tests/test_scoring.py` | Offline tests of every tier and its boundaries, every term of "laid", the exploits we tried, bad states, the reset and the task file's hooks and demo (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed and loose boxes) and the standard `task_score` scorer. `rigstate.py` sits next to `task.py`
and wraps the embodiment's step and reset from the task file's import, to read both grippers' finger-tip points, axes
and openings before each scoring call; it changes nothing the embodiment does.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the three white templates flat on the amber glass sheet, ready to trace. Turn each one so its long side runs along the sheet's streaks. Keep every template at least 8 mm in from the sheet's edges and at least 5 mm from the others, and line them up along one long edge of the sheet so that a strip at least 8 cm wide stays clear along the other long edge. Then let go and move the grippers at least 3 cm away from the templates." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/streaky-glass-template-layout/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the three white templates flat on the amber glass sheet, ready to trace. Turn each one so its long side runs along the sheet's streaks. Keep every template at least 8 mm in from the sheet's edges and at least 5 mm from the others, and line them up along one long edge of the sheet so that a strip at least 8 cm wide stays clear along the other long edge. Then let go and move the grippers at least 3 cm away from the templates." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/streaky-glass-template-layout/dream
```

Run one trial per process (the default). The scorer starts a new trial when the embodiment resets (through
`rigstate`'s reset hook, or, failing that, when the templates jump back onto their start marks in one step).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided: `n_laid`, `strip_mm`, `near_margin_mm`, `far_margin_mm`, `arms_ok`,
`arms_unavailable`, and for each template (`a1_`, `a2_`, `b_`) `laid`, `tilt_deg`, `bottom_mm`, `over_sheet`,
`path_mm`, `zone_mm` (the nearer gripper's finger zone to it), `edge_mm`, `gap_mm`, `low` (counted by the gap rule)
and `grain_deg`, with the templates' poses and both grippers' TCPs, finger-tip points and openings. Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the three
templates' poses and, from `rigstate`, both grippers' finger-tip points (midway between the two pad tips, 13 mm past
the tool point), their axes (tool point to finger-tip point) and how far each is open. Every corner, the outline seen
from above and the lowest point come from a template's pose and its size.

A template is **laid** when (a) it is tilted less than 3 degrees (either face down; they are alike), (b) its lowest
point is within 2 mm of the sheet's top and its centre over the sheet, (c) its centre's path over the last 0.5 s is
under 2 mm, (d) both grippers' fingers are at least 30 mm from its box (3-D), (e) every corner of its outline seen
from above is at least 8 mm inside every sheet edge, (f) its outline is at least 5 mm from every other template's
that lies low (its lowest point within 20 mm of the sheet's top, above or below: on the sheet, stacked on a template
or on the table beside it), and (g) its long side is within 5 degrees of the streaks (the x axis), either way round.
The **clear strip**, once all three are laid, is the larger of the margin from the near long edge (x 0.200) to the
lowest template corner and from the highest template corner to the far long edge (x 0.380). Limits stated as "at
least" or "within" include the limit itself.

For (d) each gripper's fingers are a **finger zone**: a round cylinder about the gripper's axis, from the finger-tip
point 38 mm back up, of radius hypot(half the pad gap + 10 mm, 20 mm), which is where yam_arena's finger collision
meshes reach (6.4 mm thick and 27 mm wide at the tips, 10 mm thick and 40 mm wide 19 to 38 mm up): 60.4 mm fully
open, 22.4 mm closed. It is round because `rigstate` does not report the gripper's heading. So an open finger resting
on a template is never let go, even when the point between the finger tips is 30 mm off; straight above a template
the zone's flat bottom is judged by the finger tips' real height, as the gauge block is; beside a template it can be
stricter than the simulator's fingers (by up to about 40 mm across the closing direction: 60.4 mm against the
fingers' 20 mm half width), never more lenient. The real-trial check judges the fingers the same way (every finger at
least 30 mm from every template, a 30 mm gauge block, with each gripper's fingers swept round its axis to every
heading), so the simulated and the real check pass the same outcomes.

| Outcome | Score |
|---|---|
| All three laid and a clear strip of 80 mm or more along one long edge | 1.0 |
| All three laid but the clear strip under 80 mm (spread across the middle of the sheet, or in two rows) | 0.6 |
| Two laid | 0.4 |
| One laid | 0.2 |
| None laid (nothing done, all left on the table, all across the streaks, all still held, ...) | 0 |

Possible scores: 0, 0.2, 0.4, 0.6, 1.0.

The score is read on the last step (`SCORE_OVER` "end"); the 1.0 condition held for 1 s ends the trial
(`SUCCESS_HOLD_S` 1.0), and a run that reaches its step limit with the 1.0 condition met on that step also scores
1.0. A template earns nothing if it is turned more than 5 degrees off the streaks, less than 8 mm from a sheet edge,
tilted, moving, lying on another, still held or with a gripper's fingers less than 30 mm from it, or off the sheet.
That costs the others nothing except by the gap rule: a template less than 5 mm from another that lies low loses its
credit, and so does the other one. So a template lying on another also takes the credit of the one under it, and so
does one held near the sheet, or resting on the table and leaning on the sheet's edge, less than 5 mm from a laid
one. A template carried more than 20 mm above the sheet takes no credit from the others. Only the end state counts:
there is no whole-trial rule.

For 1.0, every template's x span must lie within x 0.208 to 0.300 (the near band) or 0.280 to 0.372 (the mirror band
along the far edge): an A template's centre has +-11 mm in x at 0 degrees (+-9 mm at 5 degrees) about the demo's row
at x 0.254, B's +-16 mm; along y each template has +-7.5 mm before a 20 mm gap falls under 5 mm, and the row's ends
have 62 mm of slack to the short ends' 8 mm allowance. Two templates one behind the other take at least 135 mm (60
and 70 mm with a 5 mm gap) of the sheet's 180 mm depth, so a layout in two rows leaves a strip of 37 mm at most.

"At rest" is judged by the path the template's centre travelled over the last 10 control steps, never by a velocity:
a gripped or settling part's reported velocity chatters at 10-45 mm/s in the simulator while it stays put. A state
with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it was. If the
grippers cannot be read on a step, no template counts as let go on it (fail closed; `arms_unavailable` counts such
steps).

Unintended strategies earn less, as the tests show: lining the templates up with the sheet's outline (long sides
along its long edges) puts all three across the streaks (0); one fixed routine, turning every template 90 degrees, is
right only for A1 (0.2); laying all three across the middle or in two rows ignores the yield rule (0.6, the strip
rule carries 0.4); a template held on the glass, or hovering 1 to 2 mm above it in a closed gripper, is not let go
(no credit); an open gripper with one finger resting on a template is not let go, even with the point between its
finger tips 30 mm beyond the template's side (the finger zone covers both fingers); a template stacked on another is
12 mm too high and lies low, so the one under it is less than 5 mm from another (neither counts); a template leaning
across the 3 mm sheet edge or standing on its edge is tilted; a template carried over another's place with its
underside more than 20 mm above the sheet does not take that one's credit (20 mm or less does), nor does one on the
floor; a creeping template is not at rest, while a still one whose reported velocity chatters still counts; templates
set in a neat row on the table beside the sheet are off the sheet (0); and doing nothing scores 0, since every
template starts on the table. The sheet is fixed and the templates rigid, so nothing can be modified or destroyed.
Dropping a template from a height counts only if it happens to land laid, which is not a reliable strategy; pushing a
12 mm block up the 3 mm sheet step is impractical, and if it happened the end state would still be a correct layout.

## One arm alone

Measured on the GPU (see "Results in simulation"): with `DEMO_IDLE_ARM=right` the left arm alone lays A1 and A2 in
the row and scores 0.4; with `DEMO_IDLE_ARM=left` the right arm alone lays B and scores 0.2 (both as predicted
offline). Neither arm alone gets past 0.4. The left arm cannot reach B at y -36 cm (a straight-down
grip misses it by 294 mm), nor the right arm A1 or A2 at y +36 cm (misses of 243 and 194 mm); a gripper tilted
about 30 degrees sideways adds roughly 11 cm of reach, far short of that. So two arms are needed only because of
where the templates start: each arm's own work is an ordinary pick, turn and place, and the right arm's template has
to join the left arm's row, because the edge and strip rules leave room for only one row, about 9 cm deep, along one
long edge.

## Demo and run limit

The scripted demo takes 48.5 s (970 control steps). Every grip closes across the template's length, so the pads sit
at its two ends (fingers commanded 8 mm narrower, over 1.5 s) and never in the gaps between templates, which run
along y. Picks are at a tool point 17 mm above the table (pad tips 4 mm above it: an 8 mm grip band on the 12 mm
block), after a 0.5 s settle; carries are at 10 cm. Each place comes down to 15 mm above the place height, lowers the
last 15 mm over 1.0 s, settles 0.5 s and opens over 1.5 s with the template's underside 1 mm above the glass and the
pads' tips 5 mm above it, then goes straight up.

- Left arm: picks A1 at yaw 0 (closing along y, across its 70 mm length), carries it to x 0.254, y +0.060 turning
  it 90 degrees in the air, and lays it; then picks A2 at yaw -90 degrees (closing along x, across its length; joint 6
  about 1.74 rad, 20 degrees inside its limit), carries it to x 0.254, y 0 without turning it, and lays it; then parks
  at x 0.240, y +0.300, 12 cm up (t = 36.5 s).
- Right arm: goes over B and waits there, picks it at yaw +45 degrees (closing along its long side) and lifts it by
  t = 25.5 s, holds it up until the left arm has parked (t = 36.5 s), carries it to x 0.254, y -0.060 turning it 45
  degrees in the air, lays it, goes up and parks at x 0.240, y -0.300, 12 cm up.

The arms are never over the sheet together: the tool points stay at least 36 cm apart, and the arms' collision
capsules at least 15.7 cm apart. Offline, A1 is laid at step 315, A2 at step 655 and B at step 895 (44.75 s), when
the right gripper's finger zone has risen 30 mm above it; the 1.0 condition has then held for its 1 s at step 914
(45.7 s), with a 91 mm clear strip, 19 mm and 24 mm edge allowances and 20 mm gaps. The run limit is 2,400 steps
(2 minutes), 2.5 times the demo; the commands above pass `--max-steps 2400`. A2 carried over A1's place no longer
costs A1 its credit (A2's underside is 80 mm above the sheet, outside the gap rule's 20 mm band). Each neighbour waits
while the next template is set down beside it: the round finger zone of the gripper lowering A2 comes within 30 mm
of A1 (steps 569 to 654), and that of the gripper lowering B within 30 mm of A2 (steps 809 to 894), as the
simulator's fingers themselves do (20 to 27 mm from the neighbour); each is back once that gripper has risen 30 mm
above it. From step 914 to the end every finger zone stays about 72 mm or more from every template.

What the GPU demo's `TASK_TRACE` showed (one trial, L40S):

1. The grips: each template stayed square in the 8 mm grip band and did not twist when turned in the air; the placed
   `grain_deg` was 0.1 (A1), 0.0 (A2) and 0.7 degrees (B), against 5 allowed.
2. The set-downs: no slide or bounce on the 1 mm drop (`bottom_mm` 0 and `tilt_deg` 0 for every template once laid),
   and no template was lifted with a pad on release.
3. In the sampled trace (every 11 steps) the templates count as laid by steps 320 (A1), 661 (A2) and 903 (B); the
   1.0 condition then held for 1 s and ended the trial at step 915 (offline: 914). `arms_unavailable` stayed 0. The
   final edge allowances (`edge_mm`) were 18.4, 18.2 and 23.2 mm, the gaps (`gap_mm`) about 19 mm, the clear strip
   91.5 mm, and every finger zone about 72 mm or more from every template.
4. The top camera shows the 12 stripes as stripes at 320 x 176 (the setup screenshot), not a flat orange.

## Offline checks

`python3 tools/offline/check_task.py <this folder> --require-json --strict --one-arm` (the repository's kinematic
checker: grasp, rigid carry and straight-down drop, no physics) predicts 1.0 for the demo, 0.4 with the right arm
idle and 0.2 with the left arm idle. IK errors are under 0.09 mm and 0.02 degrees, the smallest joint margins 20
degrees (left joint 6) and 24 degrees (right joint 4). Its warnings: "centre not over its support" on the A1 and B
drops, because it sees the sheet as 12 separate 25 mm strips and names one strip as the support, while every template
lies flat across two or three of them on the sheet's flat top; and the demo is longer than the walkthrough's 600
steps (pass `--max-steps 2400`).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: put the templates back at their start
positions and turns (A1 centred at x 20.0 cm, y +36.0 cm, long side along y; A2 at x 32.0 cm, y +36.0 cm, long side
along x; B at x 26.0 cm, y -36.0 cm, long side at 45 degrees with its far end toward -y). The sheet is fixed; nothing
is consumed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is 48.5 s of simulated time); a Dream-1 trial
at the full 2,400 steps took about 5.5 minutes. The first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 180 x 300 mm piece of streaky amber cathedral glass, 3 mm thick, with its edges seamed (sanded) and the streaks
running across its short side, taped to the table; or, safer for trials, a 3 mm acrylic sheet of the same size
printed or painted with 25 mm stripes in two ambers. Three templates cut from 12 mm white acrylic: two of 70 x 40 mm
(about 40 g) and one of 60 x 40 mm (about 34 g), each with a grain arrow drawn along its long side. Mark the start
positions on the table with tape. To score, take a top photo with the grippers open and away and read the edge
allowances, gaps, angles and the clear strip with a steel rule and a protractor; a 30 mm gauge block checks that every
finger is at least 30 mm from every template, with each gripper's fingers swept round its axis to every heading (a
circle about the axis as wide as the fingers reach, 60 mm fully open), as the simulation judges.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.2 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.4 |
| Dream-1 | 0.0 |
