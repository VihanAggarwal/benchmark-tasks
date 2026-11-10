# Set floor monitors in each mic's null: cardioid 180 degrees, supercardioid 126

Audio technicians and sound engineers set floor monitors (wedges) so performers hear themselves without feedback.
A directional vocal mic picks up least from its null: a cardioid's is directly behind it, 180 degrees off its front,
so its wedge goes straight behind the mic, aimed back at the singer; a supercardioid's nulls are about 126 degrees
off its front on either side (it has a small rear lobe), so its wedge goes off to one side. Setting both wedges
straight behind their mics out of habit is the classic mistake. Here a 1:10 stage is set for a seated acoustic duo,
the audience on the arms' side: two stools, and in front of each a mic stand whose mic points upstage at the singer,
a black cardioid on the left and a silver supercardioid on the right. Two identical wedges (a black stepped cabinet
with a grey grille over its low front) wait in the wings beside the stage, each grille facing its own wing. Each
must go into its singer's mic null, 80 to 150 mm (0.8 to 1.5 m) from the stand, its grille aimed at her stool. The
score is what an engineer checks with a tape and a protractor centred on each stand, or a top photo: 0.5 per singer
for a wedge in her mic's null (within 15 degrees) and aimed at her (within 20 degrees), 0.25 for one near it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the wedges' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each wedge as one rigid body of four coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the demo plan and the arms' clearance from the mics (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, cylinders and spheres (the deck, stools, mic stands and mics), the standard
`task_score` scorer, and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a one-colour block shows
neither which side is a wedge's grille nor which way it faces. `mechkit.py` (imported by `task.py`) builds each
wedge from four boxes (a cabinet slab, a taller rear block and two grille plates) as one loose rigid body through a
hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer. It changes
nothing else the embodiment does, and it stops the run with an error if a wedge was not built as a compound body.
The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from
the task file's import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sound check on a 1:10 stage: set one floor wedge for each seated singer in the null of her mic, so it will not feed back. The left singer's black cardioid mic: directly behind it, 180 degrees off the mic's front. The right singer's silver supercardioid: about 126 degrees off its front, either side. Stand each wedge base down 80 to 150 mm from its mic stand, its grey grille aimed at that singer's stool." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wedges-on-mic-nulls/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sound check on a 1:10 stage: set one floor wedge for each seated singer in the null of her mic, so it will not feed back. The left singer's black cardioid mic: directly behind it, 180 degrees off the mic's front. The right singer's silver supercardioid: about 126 degrees off its front, either side. Stand each wedge base down 80 to 150 mm from its mic stand, its grey grille aimed at that singer's stool." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wedges-on-mic-nulls/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `zero_reason`, `moved_mm`; per wedge `singer` (whom
it serves), `r_mm`, `beta_deg`, `null_off_deg`, `aim_deg`, `tilt_deg`, `gap_mm` (its lowest point above the deck),
`deck_margin_mm` (how far inside the deck's edge its outermost corner is), `on_pose`, `fit` (the credit its pose
would earn), `at_rest`, `placed`, `credit`; per singer the wedges serving her, the best one and her credit; each
wedge's origin, orientation and front and back points). Videos: `inspect-robots video <log>.json`. The scorer keeps
one trial's record per process and starts it over at every reset (mechkit's reset counter, or the wedges teleported
back to the wings), so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the two wedges' poses, once per
control step (20 Hz), and keeps the last half second of each wedge's front and back points to tell a wedge at rest
from one still moving. A wedge is placed when its lowest point is within 3 mm of the deck's top, its up axis is
within 10 degrees of vertical (base down: on its back with the grille facing up, on a side, on its front or upside
down it is placed nowhere), its whole outline seen from above (the shadow of all its corners, as it is turned and
tilted) is over the deck, and neither end moved more than 2 mm in the last 0.5 s. It serves the singer whose mic
stand is nearer its middle (the middle of its footprint; the left singer on an exact tie). For her, r is the distance from her stand's centre to the
wedge's middle; beta the angle at the stand between the mic's front (+x, at her stool) and the line to the wedge,
0 to 180 degrees on either side; off the null is |beta - 180| for the cardioid and |beta - 126| for the
supercardioid; aim is the angle between the way the grille faces and the line from the wedge's middle to her stool's
centre. All angles are taken seen from above, as on microphone makers' monitor placement diagrams; the mics' height
above the wedges is not used. The two singers' credits add up.

| Outcome | Score |
|---|---|
| Both singers: a wedge 80-150 mm from her stand, within 15 degrees of her mic's null, aimed within 20 degrees | 1.0 |
| One singer as above; the other's wedge at 60-200 mm, within 30 degrees of the null, aimed within 45 (not full) | 0.75 |
| One singer full and the other nothing (e.g. both wedges straight behind their mics: the supercardioid's is 54 degrees off its null); or both singers at the part level | 0.5 |
| One singer at the part level and the other nothing | 0.25 |
| Neither singer has a placed wedge at the part level or better (left in the wings, facing the wings, beside or in front of a mic, on its back, in the air) | 0 |

Every outcome maps to exactly one score; each tier is reached in the tests by a concrete layout, and 600 random
layouts are checked against a second, independent scorer written from the task's text. Unintended strategies earn
nothing more than they should, as the tests show: carried across unturned, as they were picked up, the wedges face
the wings, aimed 90 and about 130 degrees off (0); both wedges straight behind their mics is 0.5; a wedge facing the
audience, set beside a mic or in front of it (between the mic and the stool) earns nothing; a wedge on its back with
the grille facing up, on a side, on its front or upside down is placed nowhere, nor is one stacked on the other
wedge, held above the deck or hanging over its edge; two wedges at one singer count once (the better); one wedge
between the two singers serves only the nearer; a wedge sliding or turning in place does not count until it has been
still for 0.5 s. Either wedge may serve either singer, and the supercardioid's wedge may go on either side of its
mic. A wedge that earns credit (60 mm or more from its stand and at least 96 degrees round from the mic's front) cannot
touch the stand's base or the stool: its outline reaches at most 37.5 mm from its middle, and its middle is at
least 60 mm from the stand's centre and 75 mm from the stool's. The scorer cannot see the grippers, so a wedge held still but
resting on the deck counts as set down (see `fidelity`). A new trial is recognised from mechkit's reset counter, or,
as a fallback, when the simulator puts both wedges back in the wings (within 4.5 cm of their marks, which covers a
seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0 for that step and
leaves the scorer's record as it was.

## One arm alone

Each arm can set its own wedge in its own singer's null, so a single arm working only on its side scores 0.5: the
scripted one-arm runs on the L40S (`DEMO_IDLE_ARM=left`, `DEMO_IDLE_ARM=right`) scored 0.5 each, the working arm's
wedge set in its singer's null and the idle arm's wedge left in its wing (see 'Results in simulation'). One wedge can serve only one singer, so an arm that reaches only its own wedge cannot score more.
Straight down, neither arm reaches the other arm's wedge: at the wedges' x (22 cm) the left arm's pad tips reach
y -15.5 cm, and the right wedge's nearest edge is at y -36.75 cm; the right arm is the mirror image (inverse
kinematics of the arm's URDF with joint limits and a collision proxy). Leaned over, though, an arm reaches further
across: leaned 45 degrees its pad tips reach y -36 cm there, leaned 60 degrees y -41 cm (the same IK, no grasp
check). So one arm with a steeply leaned grip could in principle fetch the far wedge too: the task does not claim to
need two arms. Its natural division of work is one arm per side, both at once, as in the demo.

## Demo and run limit

The scripted demo takes 17.5 s (350 control steps), both arms at once. Each arm comes down beside its wedge, grips
it across its 60 mm width over its centre of mass (4 mm behind the middle of its footprint; fingers commanded 52 mm,
over 1.5 s, the tool point 20 mm above the wedge's base, pad tips 7 mm up), so the wrist's yaw is the grille's
heading; it lifts it to carry height (tool point 10 cm above the table) and carries it over its place, turning it
on the way in two equal halves. The left arm turns the left wedge from facing the left wing to facing upstage (yaw
+90 to 0 degrees) and sets it 110 mm straight behind the cardioid at x 28.0 cm, y +15.0 cm, aimed at the left
stool. The right arm turns the right wedge from facing the right wing to facing the right stool (yaw -90 to +40.4
degrees) and sets it 110 mm from the supercardioid's stand, 126 degrees round on the outer side, at x 32.5 cm,
y -23.9 cm. Each lowers its wedge the last 3 cm over 2 s, lets go with its base aimed 1 mm above the deck and goes up
and aside. Offline: worst IK error 0.087 mm, joint limits at least 15.4 degrees away (joint 4), the tool points at
least 39 cm apart and the arms' links at least 20.8 cm apart; success has held for its 3 s at step 333 and the
scripted policy stops at step 411. On the L40S the demo's trial ended at step 335, once 1.0 had held for 3 s.

The YAM's wrist housing (link 4) juts about 65 mm ahead of the gripper, 15-30 cm above the table, so the mics are
at a seated singer's mouth height (115 mm above the deck) and the demo sets the wedges 110 mm out (mid-band): over
the whole demo every arm link stays at least 29 mm (left) and 37 mm (right) from every part of the mics and stands
(capsule proxy, which is rounder and larger than the simulator's hulls; tested). The 3 s stillness (rather than the
usual 1 s) is longer than the time from the wedges' first rest on the deck until the fingers are fully open, so a
trial cannot end while a gripper still holds a wedge. If a wedge lands off its target on the GPU, read its origin in
`TASK_TRACE` and adjust `TRIM_LEFT`/`TRIM_RIGHT` in `task.py`: the full band leaves -30/+40 mm in distance and
+-15 degrees round the stand (about +-29 mm) around each target. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the left wedge back in the left wing, on the
table off the deck, base down, its middle at x 22.0 cm, y +39.0 cm, grille facing left (+y); the right wedge at
x 22.0 cm, y -39.0 cm, grille facing right (-y). The deck, stools and mic stands are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 17.5 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 330 x 680 mm sheet of dark grey hardboard taped to the bench as the stage; two short dowels (32 mm across, 45 mm
tall) glued on it as stools, 30 cm apart; two model mic stands glued 40 mm in front of them (a 30 mm disc, an 8 mm rod and a small clip, with a
16 mm bead as the mic's head straight above the rod, 115 mm above the deck, and a 32 mm stick as its handle trailing
away from the stool, so the head points at it; black for the left mic, silver for the right); two 45 x 60 x 30 mm wooden or 3D-printed wedges, stepped as described and weighted to 80 g,
with grey mesh or paint over the step and the riser. Mark each stand's centre on the deck. Score from a photo taken
straight down with a protractor centred on each stand's base (or a tape and a protractor on the deck), and a look
from the side.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
