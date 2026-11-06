# Put both AED pads on a manikin's bare chest in the anterolateral position

Emergency medical technicians, paramedics, firefighters, nurses and trained lay rescuers put the two pads of an
automated external defibrillator (AED) on a patient's bare chest in the anterolateral position of the resuscitation
guidelines: one on the upper right chest just below the collarbone, beside the breastbone; the other on the left side
of the chest below the armpit, so that the shock passes through the heart. The pads must not touch. Here an armless
training torso lies on its back with its head away from the arms, its sternal notch, breastbone, collarbones and
nipples drawn on it, and two white training pads lie on the table beside it, one on each arm's side. Nothing marks
where the pads go: the positions follow from the landmarks, and the manikin's right is the arms' left. The score is
what an instructor reads with a tape measure from the torso's edges: how close each pad's centre is to each of the
two positions, whether the pads lie flat on the chest and were let go, and whether they are apart.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the pads' states and the grippers' state, with a small tracker |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, unreadable states and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed boxes and cylinders, two loose boxes) and the standard `task_score` scorer.

A pad counts only once both grippers are clear of it ("then let go of them"), and `yam_arena` does not pass the
grippers' state to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset
once, at import, to take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects'
frame) before each control step and to start a new trial at each reset. It reads only state the simulator already has
and changes nothing the embodiment does. If it cannot read the arms it prints why once; both pads are then taken as
clear of the grippers on those steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the two AED training pads on the manikin's bare chest in the standard anterolateral position: one on the manikin's upper right chest just below the collarbone, beside the breastbone, and the other on the left side of its chest below the armpit. The pads must not touch. Then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/aed-pads-anterolateral/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the two AED training pads on the manikin's bare chest in the standard anterolateral position: one on the manikin's upper right chest just below the collarbone, beside the breastbone, and the other on the left side of its chest below the armpit. The pads must not touch. Then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/aed-pads-anterolateral/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (for each pad: `pad_left_pos`, `pad_left_to_sternal_mm`,
`pad_left_to_apical_mm`, `pad_left_on_chest`, `pad_left_tilt_deg`, `pad_left_bottom_gap_mm`, `pad_left_released`,
`pad_left_left_zone_mm`, `pad_left_right_zone_mm`, `pad_left_counts`, and the same for `pad_right`; then
`pads_gap_mm`, `pads_touching`, `sternal_credit`, `apical_credit`, `first_full_step`, `opening`, `arms_unavailable`,
`bad_steps`, both TCPs and tip points). Videos: `inspect-robots video <log>.json`. The scorer starts its record over
at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Positions are of a pad's centre seen from above. The two
positions come from the guidelines' anterolateral placement on this torso: sternal at x 40.0 cm, y +8.0 cm, on the
manikin's upper right chest (70 mm below the shoulder line, 80 mm right of the breastbone line, between the
collarbone and the nipple line); apical at x 30.0 cm, y -14.5 cm, on its left side below the armpit (170 mm below the
shoulder line, 145 mm left of the midline, 45 mm in from the chest's left edge, 40 mm below and 60 mm outside the
left nipple: on the mid-axillary line about level with the V6 ECG electrode, as the ERC guidelines put it, and, for a
pad laid along the body, beside the left nipple, as the AHA puts it). A pad laid at that level with its outer edge on
the chest's left edge is 10 mm off, either way round. The two positions are 246 mm apart, so one pad can earn credit
at one of them only. A pad lies on the chest when its centre is over the torso's top, it is tilted at most 10 degrees
(the two large faces are alike, so either may be down) and its lowest point is within 5 mm of the chest top (a pad
resting on a 1 mm landmark line still counts). It counts when it lies on the chest and both grippers' finger zones
are at least 10 mm from it. rigstate gives no gripper heading, so a finger zone is round: every point within a radius
of the 38 mm segment from the point between the finger tips back up the gripper, the radius reaching a finger's outer
corner (57 mm fully open, 16 mm closed). For the upright grippers this means, with a tape, the point between a
gripper's finger tips at least 67 mm from the pad when fully open, 26 mm when closed (an open gripper hovering 5 cm
over a pad keeps it from counting). The pads' outlines seen from above are the shadows of their eight corners, as
they are turned and tilted.

| Outcome | Score |
|---|---|
| The two pads' outlines less than 10 mm apart (touching, overlapping, one on the other) | 0 |
| A pad that counts within 25 mm of each position | 1.0 |
| One position within 25 mm, the other 25-40 mm | 0.75 |
| One position within 25 mm and nothing at the other, or both 25-40 mm | 0.5 |
| One position 25-40 mm, nothing at the other | 0.25 |
| Anything else (pads on the table, held, standing on an edge, more than 40 mm from both positions) | 0 |

Each position's credit is 0.5 within 25 mm, 0.25 within 40 mm; the score is the sum, so every value 0, 0.25, 0.5,
0.75 and 1.0 is reachable (the tests build each one). The score is read at the last step; the trial ends early once
1.0 has held with both pads still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised from rigstate's
reset hook, or, as a fallback, when the simulator puts both pads back on their starting marks (within 4.5 cm, which
covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0 for that step
and leaves the tracker as it was.

Unintended strategies earn nothing more than they should, as the tests show: the mirror-image placement (pads on the
manikin's upper left chest and right side, the arms' right and left) scores 0; both pads at one position, side by
side, score 0.5 (only one fits the rings), and stacked or touching 0; holding the pads in place with the grippers
until the end scores 0 (a held pad does not count), and an open gripper resting on a pad, or a closed one pushing at
its side, keeps that pad from counting; a pad held 15 mm above the chest, or standing on its long edge at a position,
earns nothing there; the side pad put on the front of the chest under the nipple, the training-audit error, is 60 mm
off and earns nothing; one pad cannot bridge both positions.

## One arm alone

The demo uses both arms, each placing the pad on its own side, as two rescuers do: the left arm puts its pad on the
sternal position and the right arm puts its pad on the apical position at the same time. Each pad starts within easy
reach of its own arm only: straight down at the pads' height, each arm's tool point comes no closer than about 35 to
45 mm to the edge of the other arm's pad (inverse kinematics of the arm's URDF with joint limits and a collision
proxy), so one arm alone cannot pick up the other pad where it lies. In simulation each scripted one-arm run scored
0.5, as the offline kinematic check predicted (`DEMO_IDLE_ARM=left`: the right arm's pad on the apical position, 0.6
mm off; `DEMO_IDLE_ARM=right`: the left arm's pad on the sternal position, 0.7 mm off; the other pad did not move).
One arm alone can put its own pad on its own position, and perhaps on the far one with a grip near the pad's end, for
at most 0.5. It could only do better by dragging the other pad within reach with a fingertip pressed on its edge,
which the score does not forbid; so no claim is made that the task cannot be done with one arm.

## Demo and run limit

The scripted demo takes 15.6 s (312 control steps), both arms on one clock. Each arm comes down beside its pad (pad
tips 5 mm above the table), grips it across its 70 mm width (fingers commanded 8 mm narrower, over 1.5 s), lifts it
to 11 cm (gripper height; the pad's underside 32 mm above the chest top), carries it over its position in 2 s, comes
down in 1 s until the pad's underside is 17 mm above the chest top, lowers the last 15 mm over 1.5 s, opens over 1.5
s with the pad's underside aimed 2 mm above the chest top, rises 2.5 cm and moves back and up to 16 cm, clear of the
pads. The left arm grips at yaw -90 degrees (fingers closing along x: its pad lies across), the right at yaw 0;
neither turns on the way, so the sternal pad ends lying across the upper chest and the apical pad along the side of
the chest. The tool points stay at least 24.6 cm apart and the arm links (collision proxy) 8.7 cm apart; the left
arm's smallest joint margin is 9 degrees (joint 4, over the sternal position). The 40 g pad loads each arm with 0.4
N: the compliance model (J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) puts the gripper about 1 mm low at the place
poses, well inside the 25 mm rings, so no trim is applied (`PLACE_TRIM` in `task.py` stays zero). Offline, success
has held for its 1 s at step 322; the scripted policy stops at step 373. In simulation the score first reached 1.0 at
step 304 and the trial ended at step 323: the pads 0.7 mm (sternal) and 0.6 mm (apical) from their positions, flat
(tilt 0.0 degrees), 136 mm apart, the open grippers' finger zones 120 mm and 35 mm clear. The run limit is 1,200
steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two pads back flat on the table, one
centred at x 20.0 cm, y +27.0 cm with its long side along y, the other at x 20.0 cm, y -27.0 cm with its long side
along x; the torso, neck, head, landmarks and AED unit are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 16 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A flat-topped tan PLA shell 320 x 380 mm and 60 mm tall (a tan-painted plywood box works as well) clamped to the
bench with its shoulder line 47 cm out from the arm bases, a tan PLA 20 x 100 x 50 mm neck block and a 150 mm, 90 mm
tall round tan PLA head block beyond it; the landmarks in dark brown 1 mm tape or paint: a 20 mm sternal notch
square, a 120 x 10 mm breastbone line, two 8 x 120 mm collarbone lines and two 15 mm nipple discs at the positions in
`task.json`; a 100 x 140 x 60 mm yellow-green plastic box as the AED unit; and two AED training pads trimmed to 110 x
70 mm (smaller than most adult pads, so a parallel gripper can span one) and glued to white closed-cell foam, 18 mm
thick in all, 40 g each. Score with a tape from the torso's shoulder line and midline to each pad's centre, and a
photo from above for the gap between the pads.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
