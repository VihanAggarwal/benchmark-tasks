# Line up a motor's belt pulley with a blower's pulley using a straightedge

Belt-drive air handlers, rooftop units and exhaust fans have a motor on an adjustable base driving the fan through
V-belts. After a motor is replaced or its base is moved, HVAC mechanics line the motor's pulley (sheave) up with the
fan's before the belts go on: a straightedge, often a strip of extruded aluminium, laid across both pulleys' outer
faces should touch each pulley at both rims. Gates' white paper "Proper alignment can increase belt drive performance"
gives the limit: the total misalignment of a V-belt drive, angular and parallel "measured and quantified individually,
and then added together", should stay within 1/2 degree (about 1/10 inch per foot).

Here a blower is fixed in front of the right arm with its 90 mm pulley's outer face on the line x 23.0 cm, and 290 mm
to its left a loose 150 g motor model with a 64 mm pulley stands on a fixed steel base plate, turned 3 degrees, its
pulley face 12.1 mm behind the line (total misalignment 5.39 degrees). A 480 mm aluminium straightedge stands on a
felt strip in front of them. The motor must end in line: the score is the total misalignment M = |turn| +
atan(|offset| / 290 mm), with the motor at rest, upright, on its plate, on its line and both grippers clear. The
natural method uses both hands: the straightedge held flat on the blower pulley at both rims, the motor pushed back
against it.

Nearest tasks on the board, and why this one differs: TK-S55UZE (line up two pillow-block bearings with a motor's
shaft) carries a shaft's line out by eye and places bearings on it; nothing is made coplanar, no tool is used and the
measure is centre and bore angle. Here two pulley faces 290 mm apart are made coplanar with a hand-held straightedge,
and the measure is the belt maker's angular-plus-parallel limit, not a position band. TK-UJB387 (register a workpiece
against a corner stop, feeler-gauge gaps at both ends plus square) and TK-DGSHQA (push a fixture plate home against
its locators) end with a part against a fixed stop it touches; here the datum is a remote face 290 mm away, carried to
the motor by a straightedge that the hands hold flat on it, and the motor never touches the blower. TK-PEXMHV (support
a straightedge at its Bessel points) places supports under a straightedge; here the straightedge is the measuring tool
and the motor is what moves. TK-S928WZ (lay retainer bars across a panel) is another two-handed placement with a
different part, rule and measure. Board-wide there is no belt-drive, pulley or sheave task, and none scored on a
published alignment limit.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the blower, base plate, motor, felt and straightedge, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the motor's pose, the straightedge and the grippers, with a small tracker (the motor's last 0.5 s) |
| `mechkit.py` | Builds the blower as one fixed compound and the motor as one loose rigid compound (shafts and pulleys lying along x), and reads the motor's pose for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: a small kinematic model of the straightedge pushing the motor (the simulator never calls it; see "Offline checks") |
| `tests/test_scoring.py` | Offline tests of every tier and gate with its boundary, tier reachability, the exploits tried in review, resets, bad states, the offline model, the hooks, the text and a kinematic replay of the whole demo |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next
to `task.py` and hook `yam_arena` from the task file's import (the blower and the motor are built as mechkit bodies;
both arms are read before each scoring call); they change nothing the embodiment does, and the task uses the standard
`task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Line up the motor's pulley with the blower's pulley. Slide and turn the loose motor on its base plate, keeping it on the plate and the same distance from the blower, until its pulley's outer face is in the same plane as the blower pulley's outer face, so the straightedge laid across both faces touches each pulley at both rims. Then let go of the motor and the straightedge and move both grippers away." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/sheave-straightedge-line-up/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Line up the motor's pulley with the blower's pulley. Slide and turn the loose motor on its base plate, keeping it on the plate and the same distance from the blower, until its pulley's outer face is in the same plane as the blower pulley's outer face, so the straightedge laid across both faces touches each pulley at both rims. Then let go of the motor and the straightedge and move both grippers away." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/sheave-straightedge-line-up/dream
```

Run one trial per process (the default). The scorer starts a new trial when the embodiment resets (through
`rigstate`'s and `mechkit`'s reset hooks, or, failing both, when an object jumps 25 mm or more in one control step and
every loose object is then back near its start).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`score`, `score_tier`, `failed_gates`, `e_mm`, `theta_deg`, `alpha_deg`,
`m_deg`, `min_m_deg`, `face_x_mm`, `tilt_deg`, `plate_margin_mm`, `corner_height_mm`, `y_off_mm`, `path_mm`,
`turn_deg`, `hands_off`, `clear_mm`, `bar_face_at_motor_mm`, `bar_face_at_blower_mm`, `bar_yaw_deg`,
`bar_pitch_deg`, `bar_low_mm`, the tool points and pad tips, the demo's commanded tool points `left_cmd` /
`right_cmd` and `left_cmd_err_mm` / `right_cmd_err_mm`, `arms_unavailable`, ...); `MECHKIT_TRACE=mechkit.jsonl` the
motor's pose. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read at the last step (`SCORE_OVER = "end"`); the
trial ends early once the 1.0 condition has held for 1 s.

- F is the centre of the motor pulley's outer face (88 mm from the body's centre along its shaft, at the shaft's
  height).
- e is how far F is behind (+) or in front of, proud (-), the line x 23.0 cm: the blower pulley's outer face.
- theta is the motor's turn about the vertical from square (its shaft along x), signed and never folded: a motor
  turned round is 180 degrees out.
- alpha = atan(|e| / 290 mm), the offset as an angle over the centre distance.
- The total misalignment M = |theta| + alpha, in degrees: angular plus parallel, added as Gates adds them.

| Outcome | Score |
|---|---|
| Any gate fails: (a) not at rest (F's path 2 mm or more, or the motor's summed turn 0.5 degree or more, over the last 0.5 s); (b) the body's up axis more than 2 degrees from vertical; (c) a bottom corner of the body outside the plate's outline or more than 2 mm from its top; (d) the body's underside centre more than 10 mm from y 14.0 cm; (e) a gripper's tool point or pad-tip point within 60 mm of the motor's or the straightedge's bounding box (a fully open finger's outer face is 57 mm from the tool point) | 0 |
| All gates met, M <= 0.5 degree (an offset alone up to 2.53 mm, or a turn alone up to 0.5 degree: 0.56 mm across the 64 mm pulley) | 1.0 |
| All gates met, 0.5 < M <= 1.0 degree (5.06 mm alone) | 0.5 |
| All gates met, 1.0 < M <= 2.0 degrees (10.1 mm alone) | 0.25 |
| All gates met, M > 2.0 degrees (the start: 5.39) | 0 |

Possible scores: 0, 0.25, 0.5 and 1.0; every end state has exactly one. Stillness is judged by the face centre's path
over the last 0.5 s, never by reported velocity (the simulator's velocities chatter on parts that stay put). If the
grippers cannot be read on a step, gate (e) counts as met there and the step is counted in `arms_unavailable`. A state
with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker alone.

Unintended strategies earn nothing: holding the motor in line, or holding the straightedge against it, fails (e), and
so does leaving open fingers resting against either; sliding the motor along y (toward or away from the blower, which
changes the belt's centre distance) fails (d); turning it round puts theta near 180 degrees; tipping it, standing it
on its pulley or lifting it off the plate (onto the straightedge, say) fails (b) or (c); a motor still sliding fails
(a); the straightedge alone, laid or pushed onto the line, changes nothing; a turn never cancels an offset (|theta| +
alpha). Pushing the motor past the line is scored through e, like any offset. Carrying the motor to the line by hand,
without the straightedge, is allowed: the end state is what counts.

Real trial: tape a line on the table along y at x 20.0 cm, 30 mm in front of the blower pulley's face. Stand a
try-square on the table with its blade against the motor pulley's face at each rim at the shaft's height (64 mm apart)
and read each one's distance from the tape with a steel rule, d1 and d2: e = (d1 + d2)/2 - 30 mm and theta =
asin(|d1 - d2| / 64 mm). With the motor not proud, the straightedge laid flat on the blower pulley's face gives the same
with a feeler gauge: e = (g1 + g2)/2 and theta = asin(|g1 - g2| / 64 mm) from the gaps at the motor pulley's two rims.
A photo shows the motor upright on its plate within 10 mm of a taped y mark and both grippers clear.

## One arm alone

The task does not need two arms, and the scorer does not ask for them: it judges where the motor ends. The left arm
reaches the motor straight down and could carry it to the line, or nudge its body or pulley there, judging the line by
eye, and the right arm reaches its pulley (below): by reach, either arm alone could earn 1.0 (not run). The
straightedge method is what needs two: held by one gripper, the bar cannot lie flat on the blower pulley's face; it
pivots on one of its rims.

The scripted demo with one arm idle, measured on the GPU (the table in "Results in simulation"):

- `DEMO_IDLE_ARM=right` (the left arm alone) scores **0**. The left arm sets the motor down with its face 8.7 mm
  proud, turned 1.35 degrees (M 3.06 degrees), then grips the straightedge alone between the pulleys and presses it.
  With nothing holding the bar's far end, it pivots on the blower pulley's inner rim and its motor end runs on past
  the line: the motor passes through line (M 0.003 degree at about 38.4 s, with the hand still on the bar) and comes
  to rest 4.8 mm behind it, turned 1.1 degrees the other way: M 2.05 degrees, just outside the 0.25 band, so 0 with
  the hands clear. The offline kinematic model predicted 0 with a larger overshoot (e +29.9 mm, M 5.89 degrees), and a
  quasi-static press model gave 0 in 68 of 72 arm-stiffness cases, 0.25 in three and 1.0 in one: a one-handed press
  like this can land in a band by chance, but nothing stops it at the line.
- `DEMO_IDLE_ARM=left` (the right arm alone) scores **0**: the bar, held only outside the blower pulley's outer rim,
  pivots there and its far end swings back, away from the motor, which never moves (M 5.39 degrees on the GPU, as
  offline).
- The right arm alone cannot reach the motor's body straight down (its IK falls 30 to 34 mm short at the body's
  centre), but it does reach the motor's pulley: with the fingers closing along x across the pulley's 18 mm thickness,
  21 mm to the right of its centre (the tool point 62 mm up, the pad tips 3 mm below the shaft), its IK is exact
  (0.06 mm) both at the start (x 25.2 cm, y 11.5 cm) and with the motor in line (x 23.9 cm, y 11.9 cm). So the right
  arm alone can grip the pulley, pull the motor the 12 mm into line and turn it square: 1.0 is within its reach too,
  without tilting the gripper.

## Demo and run limit

The scripted demo takes 50.9 s (1,018 control steps); both arms run on one clock.

1. Left arm, 0 to 17.2 s: over the motor (yaw +3 degrees, fingers across the body), down to the tool point 72 mm up
   (pad tips 59 mm: the body's top 33 mm), close to 62 mm on the 70 mm body (about 20 N per finger), lift it 12 mm,
   turn it square and carry it 18 mm toward the arms in 3 s, set it down slowly with its face 6 mm proud of the line
   (x 22.4 cm: a straightedge only pushes), open, rise. Meanwhile the right arm waits over the straightedge's right
   end.
2. Both arms, from 20.2 s: down round the straightedge (fingers 50 mm open, closing along x: yaw -pi/2 left, +pi/2
   right), the right at y -25.0 cm, outside the blower pulley's outer rim (y -19.5 cm), the left at y 0, between the
   pulleys; close to 12 mm on its 20 mm (24.8 s); lift it 4 mm off the felt (its bottom 6 mm above the table, its top
   46 mm); carry it to x 19.5 cm (its face 19 mm from the motor's, 25 mm from the blower's).
3. 30.3 to 41.3 s, the press: both tool points go from x 19.5 to 25.0 cm at 5 mm/s. Offline the bar meets the
   motor's face at about 34.7 s and brings it to the line at about 35.5 s; on the GPU, with the motor set down 8.7 mm
   proud and the arms giving, it met it at about 35 s and had it in line by about 40 s. The last 30 mm of command
   (`OVERDRIVE`, equal on both arms) builds the press force in the arms' give, holding the bar flat on the blower
   pulley at both rims while it squares the motor against it. Hold 1.5 s.
4. 42.8 to 50.9 s: draw the bar back to x 19.0 cm (its face 30 mm off the pulleys), set it on the table, open to
   50 mm, rise to 15 cm. The pad tips are 60 mm clear of the bar from about 50.2 s. On the GPU the trial ended at
   step 1,024 (51.2 s) once 1.0 had held for 1 s (offline: step 1,023); the scripted policy would stop at step 1,079.

Why 30 mm of overdrive: each arm is a spring at the tool point. By J K^-1 J^T of the joint gains (80/80/80/10/10/10 N
m/rad) at the press, the left arm is about 184 N/m along x and the right 137 N/m with the wrists free, and 780 and
2,060 N/m if the bar's contacts hold the grippers' pitch; GPU arms measured in a comparable reaching posture gave 180
to 440 N/m. Taken as two equal springs with free wrists, the bar pivots on the blower pulley's outer rim and passes
0.418 of each arm's force to the motor, so each arm must push 4.2 N to start it (1.8 N): 141 N/m at 30 mm, 170 N/m at
25 mm. A quasi-static model with each arm's own x and yaw stiffness at its posture (the bar between the two arms, the
blower pulley's two rims and the motor's friction) lines the motor up at 30 mm for joint gains from 0.5 to 8 times
nominal, however the contacts hold the pitch, with the motor set down 3 or 6 mm proud (from 0.55 times at 9 mm); at 25
mm, 6 mm proud, only from 0.6 times. More force cannot push the motor past the line while the bar is held at both
places: the bar would have to lift off the blower pulley's outer rim, which takes the left arm pushing much harder
than the right. So the overdrive is kept equal on both arms (within 3 mm); more on the right than on the left lifts
the bar's far end off the motor. With the motor's centre of mass 40 mm up and 9 mm toward the pulley, a push more than
45 mm above its underside would tip it rather than slide it; the bar's top edge is 34 mm above it.

The run limit is 3,600 steps (3 minutes), 3.5 times the demo; the commands above pass `--max-steps 3600`.

What `TASK_TRACE` showed on the GPU (scripted demo, one trial):

1. Settle (`DEMO_HOLD_ONLY=1`, 160 steps): nothing moved: the motor's `tilt_deg` 0, `path_mm` 0, `e_mm` 12.1,
   `theta_deg` 3.0; the straightedge stayed on its felt.
2. After the motor is set down (from about 15 s): `e_mm` -8.7 (proud, as intended: a straightedge only pushes),
   `theta_deg` 1.35, `y_off_mm` 0.9.
3. During the press: the motor started to move at about 35 s; by about 40 s `e_mm` was 0.0 and `theta_deg` 0.003, and
   they stayed there through the hold and the draw-back. `bar_pitch_deg` rose to about 6.6 (the top edge leading, from
   the soft wrists), the motor's `tilt_deg` stayed under 0.2, and `left_cmd_err_mm` and `right_cmd_err_mm` were 31.2
   and 31.4 at the hold: the arms' give, with the bar stopped on the line. If a re-run leaves the motor proud at the
   hold, `OVERDRIVE` in `task.py` is the setting to raise.
4. The finish: `failed_gates` empty, `clear_mm` 97 mm or more for all four, `path_mm` 0, `arms_unavailable` 0,
   M 0.006 degree: 1.0.

## Offline checks

A kinematic offline check carries a gripped object rigidly and drops it when let go, and nothing pushes anything, so when the
hooks get a motor state whose `source` is "offline" (only a kinematic replay says so), `task.py` feeds the scorer the
motor's pose from `offline_model.py`: the motor keeps its own pose, taking the checker's whenever a gripper carries it;
a gripper at most 30 mm open with its pad-tip point within 15 mm of the straightedge holds it; when the bar's face,
at the motor pulley's face, comes past that face, the bar pushes the motor along x with its face flat on the bar: held
on both sides of the blower pulley, never past the line; held only on the motor's side of the blower's outer rim, on
to the bar's face (the overshoot); held only outside the outer rim, not at all. It never pulls the motor back. It has
no forces, lag, sag, pitch or friction: the GPU decides the real end state. Through it the demo predicts 1.0 and the
two one-arm runs 0 and 0.

Offline the straightedge is carried the full 30 mm past the line (on the GPU the pulleys stop it there). The two arms
stay 5.8 cm apart at their closest (both reaching down to the bar) and the right arm 33 mm from its own base while it
grips the bar 15 cm in front of it.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: put the motor back on its plate, the centre of
its body's underside at x 33.0 cm, y 14.0 cm, turned 3 degrees anticlockwise (seen from above) from square, its
pulley face 12.1 mm behind the line; stand the straightedge on its 20 mm face on the felt, along y, centred at x 15.0
cm, y -4.5 cm. The blower, the plate and the felt are fixed; nothing is consumed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 50 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more. The Dream-1 trial at the full 3,600 steps took about
7 minutes.

## Real-world build

Unpowered, from common materials: an MDF blower box (120 x 100 x 140 mm) screwed to the table with a printed or
turned 90 mm pulley, 20 mm thick, on a fixed 16 mm rod, its axis 52 mm above the table; a 160 x 130 x 12 mm galvanised
steel or aluminium plate taped down beside it as the motor base (x 26.2 to 42.2 cm); a printed 90 x 70 x 80 mm motor
model weighted to 150 g, the weight spread so that its centre of mass sits 40 mm above its underside and 9 mm from the
body's centre toward the pulley, with a 64 mm, 18 mm thick pulley on a 16 mm dowel 40 mm above its underside (the two
pulleys' faces equally far from their grooves); a 480 mm length of 20 x 40 mm aluminium box section (1.2 mm wall) on a
40 x 500 mm felt strip. Tape a y mark for the motor's line and the x 20.0 cm reference line for measuring; keep a
try-square, a steel rule and a feeler gauge.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The demo ended with M 0.006 degree and every gate met. With the left arm alone the motor was pushed 4.8 mm past the
line and turned 1.1 degrees (M 2.05 degrees, just outside the 0.25 band); with the right arm alone it never moved.
Dream-1 ran the full 3,600 steps and moved neither the motor nor the straightedge (M stayed 5.39 degrees).
