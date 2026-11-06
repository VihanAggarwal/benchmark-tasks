# Smooth a cake's side: hold the smoother still as the other arm turns the turntable

Cake decorators in commercial and supermarket bakeries finish an iced cake's side with a bench scraper or cake
smoother: one hand holds it still and upright against the side (flat on it, or angled up to about 45 degrees with its
edge doing the work), its bottom edge on or just above the turntable, while the other hand turns the turntable. The
tool stays put and the cake moves under it, as on a potter's wheel or a lathe; moving the tool round the cake instead
leaves ridges. Here a dummy cake stands on a turntable fixed to the table, the turntable has four pegs to turn it by,
and a smoother hangs in a stand in front of the left arm. One gripper must hold the smoother set against the side (a
gap of -1 to +3 mm, its bottom just above the plate, its blade within 10 mm and 4 degrees of where it was put) while
the other gripper turns the turntable nearly a full turn by its pegs. The score is the share of the side that passed
the still blade while the other gripper was turning the turntable. One gripper cannot do both.

Nearest tasks on the board, and why this one differs: TK-W6W94V (index an off-balance rotary table 90 degrees by its
peg, then pin it) is one static quarter turn while the other arm drops a pin; nothing is held against the work while
it turns and there is no coverage. TK-233JTT (pilot: collect blocks from a rotating lazy susan) picks objects off a
turning surface. TK-UKXJZS (pilot: steady a bowl while loading a spoon) is a static hold. TK-8VAYXV (pilot: wipe
marks off a board) moves the tool over a still surface, the opposite rule. TK-PPW69G (hold a dustpan still while the
other arm sweeps cubes into it) and TK-8HST9N (keep a pump barrel still and work its handle) also hold one thing still
while the other arm works, but there the still object only receives the work; here the still tool does the work and
the moving work is what is scored. The same holds against the popular task sets' archetypes: hold or steady while the
other arm inserts or fills, sweep, wipe and scrub, and simple tool use all move the tool or the parts over still work.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the turntable, the smoother and its stand, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the turntable's angle, the smoother's pose and the grippers, with a tracker for the whole trial |
| `mechkit.py` | Builds the turntable as one fixed articulation with a revolute plate, and the smoother as one rigid compound, and reads them for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: a small kinematic model of a gripper turning the plate by a peg (the simulator never calls it; see "Offline checks") |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition and its boundary, the exploits tried in review, resets, bad states, the offline model, the hooks and a kinematic replay of the whole demo |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next
to `task.py` and hook `yam_arena` from the task file's import (the turntable and the smoother are built as mechkit
bodies; both arms are read before each scoring call); they change nothing the embodiment does, and the task uses the
standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Smooth the side of the cake on the turntable. With one gripper, take the smoother from its stand by the handle and hold it still and upright, its blade against the cake's side, its bottom edge just above the turntable. With the other gripper, turn the turntable anticlockwise (seen from above) by its black pegs almost a full turn. Then move the smoother 1 cm from the cake and both grippers 2 cm from the pegs. Never touch the cake with a gripper or its top with the smoother." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/turntable-still-scraper/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Smooth the side of the cake on the turntable. With one gripper, take the smoother from its stand by the handle and hold it still and upright, its blade against the cake's side, its bottom edge just above the turntable. With the other gripper, turn the turntable anticlockwise (seen from above) by its black pegs almost a full turn. Then move the smoother 1 cm from the cake and both grippers 2 cm from the pegs. Never touch the cake with a gripper or its top with the smoother." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/turntable-still-scraper/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when the smoother
jumps back into its stand in one step with the turntable back at 0).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`covered`, `turn_deg`, `placed`, `set`, `latched`, `holder`, `driver`,
`gap_mm`, `face_deg`, `bottom_mm`, `tilt_deg`, `heading_err_deg`, `drift_mm`, `max_drift_mm`, `turned_while_set_deg`,
`first_full_step`, `smoother_cake_mm`, `min_finger_cake_mm`, `peg_clear_mm`, `finish`, `finish_s`, `zeros`,
`arms_unavailable`, the tool points and finger tips, ...); `MECHKIT_TRACE=mechkit.jsonl` the turntable's joint and the
smoother's pose. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the turntable's
angle (mechkit's joint reading, positive anticlockwise seen from above), the smoother's pose and both grippers.

The smoother is **set** when a gripper at most half open has its finger-tip point within 10 mm of the handle, the
smoother is tilted under 5 degrees, the blade's lowest point is 1 mm below to 5 mm above the plate's top, the blade
comes within -1 to +3 mm of the cake's side where they are nearest (the contact point; the gap is measured from the
blade's real box, a face, an edge or a corner) with its length within 45 degrees of the side's tangent there, and it
is still: the blade's centre moving slower than 1 cm/s and within 10 mm of where it was when the smoother was put in
place, its heading turning slower than 3 degrees a second and within 4 degrees of its heading then. Once out of either
window it is not set again until it has been taken away (its centre 20 mm or more from where it was put, its bottom
20 mm or more above the plate, or let go) and put back.

The side is 360 one-degree sectors fixed to the cake. A step's turn counts only while the smoother stays set **and the
other gripper** (not the one holding the smoother) has its finger zone within 5 mm of a peg: the other hand turns the
turntable. Then the turn, less any part of it that the blade's centre followed round, carries the side across both
edges of the credited arc, 3 degrees either side of the contact point; every sector an edge crosses has passed the
blade, once. One uninterrupted turn of D degrees under a still smoother passes D + 6 sectors. The turntable stops at
349.5 degrees, so the full score asks for 340 of the 360 sectors (a turn of about 334 degrees).

| Outcome | Score |
|---|---|
| At any moment a gripper's finger zone within 5 mm of the cake; any part of the smoother less than 3 mm above the cake's top within 57 mm of its axis; or the smoother's lowest point 20 mm or more below the table top | 0 |
| 340 or more sectors passed the blade, then for 1 s the smoother 10 mm or more from the cake, every finger zone 20 mm or more from every peg and the turntable still (under 0.05 rad/s) | 1.0 |
| 270 or more sectors passed the blade (or 340 without the finish) | 0.75 |
| 180 or more sectors passed the blade | 0.5 |
| The smoother set for 1 s without a break | 0.25 |
| Anything else | 0 |

The finger zone is the round zone about a gripper's axis from its finger tips 38 mm up, its radius reaching the real
fingers' outer corners (10 mm thick outside the gap and 40 mm wide there, from yam_arena's finger collision meshes);
the scorer cannot see a gripper's heading, so the zone is round. The finish's 1 s is timed by the scorer itself, so a
run cut at its limit scores 1.0 only if the finish had already held for 1 s. Grippers that cannot be read on a step
hold nothing, turn nothing and do not meet the finish (the finger zones are then not checked; `arms_unavailable` in
the trace): a run whose arms are never read scores 0.

Unintended strategies earn nothing beyond 0.25 (the smoother set for 1 s), and none of them alone gets coverage:
moving the smoother round a still cake, set again and again or slid along the side, passes nothing (only the
turntable's turn counts); pressing the blade on the side and dragging the cake round with it passes nothing (the
blade follows the side); pressing the blade's bottom edge on the plate and twisting the wrist to ratchet the plate
round turns the blade's heading out of its 4 degree window, and no other gripper is at a peg; flicking the plate and
letting it coast under the smoother counts for nothing (no gripper at a peg while it coasts, and the bearing's drag
stops it within about 5 degrees from 1 rad/s); re-setting the blade further back after each short turn gains only the 6 degree arc
each time (six 30 degree turns pass about 220 sectors: 0.5); rolling the blade round the side turns its heading out of its
window; the smoother left standing or leaning unheld against the cake is never set; rocking the turntable back and
forth over a quarter passes about 97 sectors (0.25); turning first and setting the smoother afterwards passes
nothing; holding the smoother high above the plate, off the side, tilted, upside down or end on (more than 45
degrees to the side) is never set; turning the cake by pushing its side or top with a gripper scores 0.

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone takes the smoother and holds it set against the still cake for 41 s)
scores 0.25; `DEMO_IDLE_ARM=left` (the right arm alone turns the turntable 344 degrees with no tool against the cake)
scores 0. (The GPU results are in 'Results in simulation' below.)

A single arm cannot get any coverage, whichever way its gripper points: a turn counts only while the gripper that is
not holding the smoother is at a peg. Holding the smoother set and turning the plate with the same arm (by the held
blade, by its bottom edge on the plate, by a twist of the wrist, or by a flick before setting it) passes nothing.
With its gripper pointing straight down the right arm cannot reach the stand (21.5 cm past the midline; a
straight-down grip reaches about 10 cm past it), but with its gripper tilted about 30 degrees sideways it can reach
the handle there, so the right arm alone could set the smoother and score 0.25; it still cannot also turn the plate.
The smoother may be held by either gripper.

## Demo and run limit

The scripted demo takes 57.3 s (1,146 control steps). The left arm (yaw 0) closes on the smoother's handle in its
stand, lifts it straight out, carries it to 2 cm outside the cake at 60 degrees (its front left), turning it 30
degrees so the blade lies flat along the side, lowers it slowly to 3.5 mm above the plate and moves it in to a 0.8 mm
gap: set at t = 14.1 s. Meanwhile the right arm closes on the peg at 210 degrees (fingers along the pegs' circle,
12 mm commanded on the 20 mm peg) and waits. From t = 14.6 s it turns the plate in four strokes of 86 degrees (5
straight chords of 17.2 degrees each, 0.8 s apiece, the gripper's yaw following the plate; the chords stay within
1.7 mm of the pegs' circle), between strokes letting go, lifting its tips 10 mm over the pegs and going back 90 degrees
round the circle to the next peg: 344 degrees by t = 54.3 s, 5.5 degrees short of the stop. Offline 340 sectors have
passed the blade at step 1,077 (334 degrees) and 350 by the end. The left arm then backs the smoother 2 cm out and
lifts it, still in hand; the right arm lets go and rises 6.5 cm. Offline, the finish starts at step 1,136 and success
has held for its 1 s at step 1,155 (57.75 s); the scripted policy stops at step 1,207. The run limit is 3,600 steps
(3 minutes), 3.1 times the demo; the commands above pass `--max-steps 3600` (an automatic suggestion of 1,200 counts
the demo alone).

The set pose allows for the held smoother's sag on the soft wrist (our estimate from J K^-1 J^T of the arm's joint
gains, the 67 g payload not compensated: about 1.8 mm down and 0.2 mm away from the cake): on the GPU the gap should
read about 1.0 mm and the bottom about 1.7 mm, both well inside their windows.

What to check first in `TASK_TRACE` on the GPU:

1. `turn_deg` must rise smoothly through 180 degrees to about 344. If it stalls or jumps near 180, PhysX is wrapping
   the revolute joint at +-pi (the scorer's own unwrapping keeps the coverage right, but the plate would not turn
   on): report it, since it breaks the turntable, not the scorer.
2. `covered` at least 340 by the last stroke (`first_full_step`); `driver` is "right" through every stroke.
3. While the plate turns: `set` true from about step 270 to 1,096, `gap_mm` about 1.0 (-1 to 3 allowed), `bottom_mm`
   about 1.7, `tilt_deg` under 5, `heading_err_deg` under 4, `max_drift_mm` under 10, `latched` false. If the gap
   or height drift, tune only `SET_GAP` / `SET_LIFT` in `layout.py` (keeping the commanded values in the windows).
4. If the closed grip on the pegs slips and the plate falls short, add the shortfall to `STROKE_DEG` in `task.py`,
   keeping four strokes under the 349.5 degree stop. Closing on each new peg with the fingers along the circle
   pushes it to the middle of the pads, which takes out a few millimetres of lag left by the stroke before.
5. The finish: `peg_clear_mm` for the right gripper 20 or more after it rises, `smoother_cake_mm` 10 or more,
   `finish_s` reaching 1.0; `min_finger_cake_mm` over 5 throughout (32.5 left, 49.4 right offline); `arms_unavailable`
   must be 0.
6. Settle (`DEMO_HOLD_ONLY=1`): the smoother stays hanging in its stand and the plate stays at 0.

## Offline checks

Without a simulator the turntable's joint never moves, so when the hooks get a turntable state whose `source` is
"offline" (only a kinematic replay says so), `task.py` feeds the scorer the plate's angle from `offline_model.py`
instead of the joint reading: a gripper that closes past a peg's 20 mm width with its finger-tip point within 10 mm of
the peg's centre, between the plate's top and the peg's top, grips the peg; while it stays closed on it and within
15 mm of the pegs' circle, the plate turns with the finger tips' angle about the axis, up to the joint's stops; it
lets go when the gripper opens, rises above the peg or drifts off the circle. Nothing else moves the plate in the
model (open fingers, the smoother or a link pushing it), so an arm that does not grip a peg turns nothing. The model
has no lag, slip, sag, coasting or contact: the arm's tracking, the closed chain through the peg and the soft wrist
on the GPU decide the real angle and the real gap. It resets with rigstate's reset hook and at every new replay.
Through it the demo predicts 1.0 and the two one-arm runs 0.25 and 0.

The replay's warnings for this task, all from the smoother's bounding box (32 mm wide, the handle's width, from the
blade's bottom up): the smoother overlaps both stand blocks by 7 mm at t = 0, and the carried smoother "runs into"
them by 7 mm as it is lifted out. The real blade hangs 6 mm clear of each block and the handle rests on 7 mm of each
block's top, above them. The demo also runs more steps than the walkthrough's 600: pass `--max-steps 3600`. The two
arms' collision capsules stay 3.1 cm apart at their closest (the right arm going back round to the next peg while the
left holds the smoother) and the tool points 20 cm apart.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: turn the plate back to its start (red index
mark straight forward; pegs at 30, 120, 210 and 300 degrees anticlockwise from forward) and hang the smoother back in
its stand by the handle, its length along x, the blade centred at x 27.0 cm, y +21.5 cm. The turntable and the stand
are fixed; nothing is consumed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 58 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more. A Dream-1 trial at the full 3,600 steps takes about 9 to
13 minutes.

## Real-world build

A 340 mm, 10 mm HDPE disc on a lazy-susan bearing on a 100 mm base screwed to the table, with a felt washer under the
plate for drag (let go spinning at about 10 turns a minute, 1 rad/s, it must stop within 10 degrees; adjust the washer
until it does) and a lug under the plate that meets a stop pin in the base at 11.5 degrees clockwise and 349.5
degrees anticlockwise of the start; four 20 mm acetal or wooden pegs 30 mm tall screwed into the plate 155 mm from
its centre at 90 degree spacing, and a degree ring printed round its rim; a 120 mm, 50 mm tall foam cake dummy glued
on its centre (iced or painted, so marks show); a 100 x 70 mm piece of 6 mm acrylic (50 g) with a hollow 70 x 32 x
30 mm polypropylene handle (2 mm walls, 17 g) glued along its top edge; and two 120 x 12 x 72 mm blocks screwed down
18 mm apart as its stand. Nothing is powered. To judge stillness on video, fix two pointers on small stands on the
table beside the turntable, off the plate, one at each end of the blade's set position (or film from a camera fixed
overhead with the set position marked on its image); keep 1 mm, 3 mm and 5 mm shims for the gap and the height.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
