# Line up a motor's belt pulley with a blower's pulley 290 mm away

Belt-drive air handlers, rooftop units and exhaust fans have a motor on an adjustable base driving the fan through
V-belts. After a motor is replaced or its base is moved, HVAC mechanics line the motor's pulley (sheave) up with the
fan's before the belts go on: a straightedge, often a strip of extruded aluminium, laid across both pulleys' outer
faces should touch each pulley at both rims. Gates' white paper "Proper alignment can increase belt drive performance"
gives the limit: the total misalignment of a V-belt drive, angular and parallel "measured and quantified individually,
and then added together", should stay within 1/2 degree (about 1/10 inch per foot).

Here a blower is fixed in front of the right arm with its 90 mm pulley's outer face on the line x 23.0 cm, and 290 mm
to its left a loose 150 g motor with a 64 mm pulley stands on a fixed steel base plate, turned 3 degrees, its
pulley face 12.1 mm behind the line (total misalignment 5.39 degrees). A 480 mm aluminium straightedge stands on a
felt strip in front of them. The motor must end in line: the score is the total misalignment M = |turn| +
atan(|offset| / 290 mm), with the motor at rest, upright, on its plate, on its line and both grippers clear, and the
motor must stay on its plate and on its line on every control step of the trial (lifting it more than 2 mm off the
plate, even for one step, scores 0). The 1.0 band is the belt maker's 1/2 degree: 2.53 mm of offset alone, or 0.56 mm
of turn across the motor's 64 mm pulley, against a face 290 mm away. The score judges that alignment; the straightedge
is the trade's tool for reading it, and the natural method uses both hands: the straightedge held flat on the blower
pulley at both rims, the motor slid back against it.

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
| `scoring.py` | The scorer: plain functions of the motor's pose, the straightedge and the grippers, with a small tracker (the motor's and the grippers' last 0.5 s, and the first breach of each whole-trial rule, latched) |
| `mechkit.py` | Builds the blower as one fixed compound and the motor as one loose rigid compound (shafts and pulleys lying along x), and reads the motor's pose for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: a small kinematic model of the straightedge pushing the motor (the simulator never calls it; see "Offline checks") |
| `tests/test_scoring.py` | Offline tests of every tier and gate with its boundary, the whole-trial plate and line rules (the old lift-and-carry demo scores 0), fail-closed grippers, tier reachability, the exploits tried in review, resets, bad states, the offline model, the hooks, the text and a kinematic replay of the whole demo |

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

Run one trial per process (the default). The scorer starts a new trial only when the embodiment resets, through
`rigstate`'s and `mechkit`'s reset hooks. It never infers a reset from where the parts are: a motor lifted off its
plate and then put back on its exact start, with the straightedge at its start, is still the same trial, and the
breach stays latched (`WholeTrial.test_putting_the_parts_back_never_clears_the_latches`).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`score`, `score_tier`, `failed_gates`, `e_mm`, `theta_deg`, `alpha_deg`,
`m_deg`, `min_m_deg`, `face_x_mm`, `tilt_deg`, `plate_margin_mm`, `corner_height_mm`, `lowest_corner_mm`,
`max_lowest_corner_mm`, `y_off_mm`, `max_y_off_mm`, `stayed_on_plate`, `stayed_on_line`, `plate_breach`,
`line_breach` (the first breach: step, why, value), `path_mm`, `turn_deg`, `hands_off`, `hands_off_steps`,
`clear_mm`, `bar_face_at_motor_mm`, `bar_face_at_blower_mm`, `bar_yaw_deg`,
`bar_pitch_deg`, `bar_low_mm`, the tool points and pad tips, the demo's commanded tool points `left_cmd` /
`right_cmd` and `left_cmd_err_mm` / `right_cmd_err_mm`, `arms_unavailable`, ...); `MECHKIT_TRACE=mechkit.jsonl` the
motor's pose. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read at the last step (`SCORE_OVER = "end"`) and
needs the two whole-trial rules, which the tracker checks on every control step from the start; the trial ends early
once the 1.0 condition has held for 1 s.

- F is the centre of the motor pulley's outer face (88 mm from the body's centre along its shaft, at the shaft's
  height).
- e is how far F is behind (+) or in front of, proud (-), the line x 23.0 cm: the blower pulley's outer face.
- theta is the motor's turn about the vertical from square (its shaft along x), signed and never folded: a motor
  turned round is 180 degrees out.
- alpha = atan(|e| / 290 mm), the offset as an angle over the centre distance.
- The total misalignment M = |theta| + alpha, in degrees: angular plus parallel, added as Gates adds them.

Whole trial, on every control step (20 Hz), latched: one step that breaks a rule scores 0 for the rest of the trial.

- (f) On its plate: the lowest bottom corner of the motor's body within 2 mm of the plate's top, the centre of its
  underside over the plate's outline, and its up axis within 10 degrees of vertical. Lifting the motor more than 2 mm
  off the plate, even for a single step, breaks it; so does dropping it off an edge or tipping it up. Rocking onto an
  edge that stays on the plate does not: a motor slid by its body rocks a little onto its leading edge (see the demo).
- (g) On its line: the centre of the body's underside within 10 mm of y 14.0 cm, which keeps the 290 mm centre
  distance that alpha uses.
- A step on which the motor's pose cannot be read (a non-finite number or a zero quaternion) breaks both: they cannot
  be verified there.

| Outcome | Score |
|---|---|
| (f) or (g) broken on any step of the trial | 0 |
| An end gate fails: (a) not at rest (F's path 2 mm or more, or the motor's summed turn 0.5 degree or more, over the last 0.5 s); (b) the body's up axis more than 2 degrees from vertical; (c) a bottom corner of the body outside the plate's outline or more than 2 mm from its top; (d) the body's underside centre more than 10 mm from y 14.0 cm; (e) on any step of the last 0.5 s, a gripper not read, or its tool point or pad-tip point within 60 mm of the motor's or the straightedge's bounding box (a fully open finger's outer face is 57 mm from the tool point) | 0 |
| All met, M <= 0.5 degree (an offset alone up to 2.53 mm, or a turn alone up to 0.5 degree: 0.56 mm across the 64 mm pulley) | 1.0 |
| All met, 0.5 < M <= 1.0 degree (5.06 mm alone) | 0.5 |
| All met, 1.0 < M <= 2.0 degrees (10.1 mm alone) | 0.25 |
| All met, M > 2.0 degrees (the start: 5.39) | 0 |

Possible scores: 0, 0.25, 0.5 and 1.0; every trial has exactly one. Stillness is judged by the face centre's path
over the last 0.5 s, never by reported velocity (the simulator's velocities chatter on parts that stay put).

Everything that depends on the grippers fails closed: a step on which rigstate cannot read both arms (None, a side
missing, a non-finite number) is not hands off, so the score is 0 until 0.5 s of readable, clear steps have passed;
such steps are counted in `arms_unavailable`. A step whose straightedge state is not finite is not hands off either.
A motor state that is not finite scores 0 for that step and, through (f) and (g), for the trial.

Unintended strategies earn nothing: holding the motor in line, or holding the straightedge against it, fails (e), and
so does leaving open fingers resting against either, or letting go for less than 0.5 s; lifting or carrying the
motor, at any moment, fails (f), however well it is set down, even back on its exact start (only an episode
reset clears it); sliding it along y (toward or away from the blower, which changes the belt's centre distance)
beyond 10 mm at any moment fails (g); turning it round puts theta near 180 degrees; tipping it or standing it on its pulley fails (f) or (b); a motor still sliding fails (a); the straightedge
alone, laid or pushed onto the line, changes nothing; a turn never cancels an offset (|theta| + alpha). Pushing the
motor past the line is scored through e, like any offset. The score judges the alignment, not the tool: lining the
motor up without the straightedge, judging the line some other way, also counts if the motor stays on its plate and
its line. What makes it hard by any method is the band: 1/2 degree is 2.53 mm of offset alone, or 0.56 mm of turn
across the 64 mm pulley, read against a face 290 mm away.

Real trial, for each threshold:

| Condition | Real-bench reading |
|---|---|
| M (e and theta) | With the straightedge flat on the blower pulley's face, feeler gauges read the gaps g1 and g2 at the motor pulley's two rims (64 mm apart): e = (g1 + g2)/2, theta = asin(\|g1 - g2\| / 64 mm). With the motor proud of the line: tape a line along y at x 20.0 cm, 30 mm in front of the blower pulley's face, stand a try-square against the motor pulley's face at each rim at the shaft's height, and read each one's distance from the tape with a steel rule, d1 and d2: e = (d1 + d2)/2 - 30 mm, theta = asin(\|d1 - d2\| / 64 mm) |
| (f) on its plate, every step | A side video with the camera level with the plate's top and a card ruled with a line 2 mm above that top standing behind the motor: no bottom corner of the body rises past the line on any frame |
| (g) on its line, every step | A tape mark on the plate at y 14.0 cm with ticks 10 mm either side, seen from above: the body's centre line never passes a tick |
| (a) at rest, (b) upright, (c) flat on its plate | (a): the same videos over the last 0.5 s; (b): a try-square on the plate against the body's side (2 degrees is 2.8 mm over the body's 80 mm height); (c): the side video's 2 mm line and the plate's edges against the body's corners |
| (e) hands off | The top video: both grippers at least 60 mm from the motor and the straightedge over the last 0.5 s (the motor's 70 mm width is the scale in the frame) |

## One arm alone

The task does not need two arms, and the scorer does not ask for them: it judges where the motor ends and that it
never left its plate or its line. The left arm reaches the motor straight down and could slide it to the line, or
nudge its body or pulley there, judging the line some other way, and the right arm reaches its pulley (below): by
reach, either arm alone can attempt the whole task that way, and the score would judge only the result. The
straightedge method is what needs two: held by one gripper, the bar cannot lie flat on the blower pulley's face; it
pivots on one of its rims.

The scripted demo with one arm idle (the same open-loop plan with that arm's waypoints dropped), measured on the GPU
with this version's demo and scorer: both score **0** (offline, by kinematic replay with `offline_model.py`, both 0
too):

- `DEMO_IDLE_ARM=right` (the left arm alone): the left arm slides the motor to 11.1 mm proud, then grips the
  straightedge alone between the pulleys and presses it. With nothing holding the bar's far end, it pivots on the
  blower pulley's inner rim and its motor end runs on past the line. The motor passed through line (M down to 0.004
  degree at about 38 s, with the hand still on the bar) and came to rest 4.9 mm behind it, turned 1.14 degrees the
  other way: M 2.12 degrees, just outside the 0.25 band, every gate met: **0**. It never broke a whole-trial rule
  (`plate_breach` and `line_breach` null, `max_y_off_mm` 4.5). Offline the overshoot is larger (0), and a
  quasi-static press model gave 0 in 68 of 72 arm-stiffness cases, 0.25 in three and 1.0 in one: a one-handed press
  like this can land in a band by chance, but nothing stops it at the line.
- `DEMO_IDLE_ARM=left` (the right arm alone): the bar, held only outside the blower pulley's outer rim, pivots there
  and its far end swings back, away from the motor, which never moves (M stays 5.39 degrees, as offline): **0**.
- The right arm alone cannot reach the motor's body straight down (its IK falls 30 to 34 mm short at the body's
  centre), but it does reach the motor's pulley: with the fingers closing along x across the pulley's 18 mm thickness,
  21 mm to the right of its centre (the tool point 62 mm up, the pad tips 3 mm below the shaft), its IK is exact
  (0.06 mm) both at the start (x 25.2 cm, y 11.5 cm) and with the motor in line (x 23.9 cm, y 11.9 cm). So the right
  arm alone can grip the pulley, pull the motor the 12 mm into line and turn it square: 1.0 is within its reach too,
  without tilting the gripper.

## Demo and run limit

The scripted demo takes 50.9 s (1,018 control steps); both arms run on one clock. It is open-loop: fixed waypoints
for the tool points, wrist yaws and finger openings (`task.py`, `demo()`), with no simulator state read. The motor is
never lifted: it is slid and turned on its plate by its body, then pushed into line by the straightedge.

1. Left arm, 0 to 17.2 s, the motor:
   - over the motor (yaw +3 degrees, fingers across the body), down to the tool point 90 mm up, then slowly to 60 mm
     (5.5 s): the 22 mm pads on the body 47 to 69 mm up, around its centre of mass (52 mm up), their tips 35 mm above
     the plate's top; the finger carriages, 57 mm above the tool point, 25 mm above the body's 92 mm top;
   - close to 62 mm on the 70 mm body (about 20 N per finger, 7.6 s) with the motor resting on the plate;
   - 8.1 to 12.6 s, the slide: at the same height (the arm holds the tool point where it gripped the motor resting on
     the plate; carrying its weight would need the command about 3 mm higher, by the 3.1 mm sag the GPU measured with
     this motor hanging from this arm), the command goes from x 33.0 to 29.7 cm (33 mm, 7.3 mm/s on average) while the
     wrist turns from +3 degrees to square;
   - 12.6 to 13.6 s, ease off: the command goes back to x 31.0 cm (`PRE_X`, the motor's face 8 mm proud of the line,
     x 22.2 cm: a straightedge only pushes), so the arm's pull drops below the plate's static friction before the
     fingers open; settle, open (15.7 s), rise to 15 cm (17.2 s).
   Meanwhile the right arm waits over the straightedge's right end.

   On the GPU the slide left the motor 11.1 mm proud, turned 1.05 degrees, rocked at most about 1.7 degrees onto its
   leading edge (lowest corner 0.0 mm from the plate's top on every step), 1.2 mm off its line. A quasi-static model
   of the arm, the motor and the plate, written before that run, had predicted about 10 mm proud and 2.6 degrees of
   rock: with its wrist free the arm is about 150 N/m along x (J K^-1 J^T of the joint gains; the same model
   predicts the 3.1 mm sag measured on the GPU), so the plate's sliding friction (1.0 x 1.47 N) leaves the motor about
   11 mm behind the command. Through the wrist's pitch the motor rocks about 2.6 degrees onto its leading (pulley-end)
   bottom edge, which stays pressed on the plate: rule (f) reads the lowest bottom corner, and that corner never leaves
   the plate's top (1.3 degrees for arms twice as stiff, 3.3 with the friction at its static value; the trailing edge,
   90 mm back, is then about 4 mm up). It settles flat when the pull eases. The slide is commanded `DRAG_LAG` (13 mm)
   past `PRE_X`, so the motor stops about 10 mm proud (x 22.0 cm, the origin at x 30.8 cm: every bottom corner on the
   plate, the front ones at or just inside its front edge), having slid about 22 mm; that is close to the 8.7 mm proud
   from which the unchanged press lined it up on the GPU in the first version. An arm stiffer
   than modelled follows the ease-off back to `PRE_X` (8 mm proud), arms 0.8 times as stiff leave it 7 mm proud, and
   arms half as stiff about 1 mm behind the line (then the press squares it only if a rim is still proud).
2. Both arms, from 20.2 s: down round the straightedge (fingers 50 mm open, closing along x: yaw -pi/2 left, +pi/2
   right), the right at y -25.0 cm, outside the blower pulley's outer rim (y -19.5 cm), the left at y 0, between the
   pulleys; close to 12 mm on its 20 mm (24.8 s); lift it 4 mm off the felt (its bottom 6 mm above the table, its top
   46 mm); carry it to x 19.5 cm (its face about 15 mm from the motor's, 25 mm from the blower's).
3. 30.3 to 41.3 s, the press (unchanged from the first version): both tool points go from x 19.5 to 25.0 cm at
   5 mm/s. Offline the bar meets the motor's face, 8 mm proud, at about 34.4 s and brings it to the line at about
   35.5 s; on the GPU (the motor 11.1 mm proud and the arms giving) it met it at about 35 s and had it in line by
   about 40 s. The last 30 mm of command (`OVERDRIVE`, equal on both arms) builds the press force in the arms' give, holding the bar flat on the blower
   pulley at both rims while it squares the motor against it. Hold 1.5 s.
4. 42.8 to 50.9 s: draw the bar back to x 19.0 cm (its face 30 mm off the pulleys), set it on the table, open to
   50 mm, rise to 15 cm. The pad tips are 60 mm clear of the bar from about 50.2 s; hands off must then hold for
   0.5 s, and 1.0 for 1 s: on the GPU the score reached 1.0 at step 1,015 and the trial ended at step 1,034
   (51.7 s; offline 1,033); the scripted policy would stop at step 1,079.

Why 30 mm of overdrive: each arm is a spring at the tool point. By J K^-1 J^T of the joint gains (80/80/80/10/10/10 N
m/rad) at the press, the left arm is about 184 N/m along x and the right 137 N/m with the wrists free, and 780 and
2,060 N/m if the bar's contacts hold the grippers' pitch; GPU arms measured in a comparable reaching posture gave 180
to 440 N/m. Taken as two equal springs with free wrists, the bar pivots on the blower pulley's outer rim and passes
0.418 of each arm's force to the motor, so each arm must push 4.2 N to start it (1.8 N): 141 N/m at 30 mm, 170 N/m at
25 mm. A quasi-static model with each arm's own x and yaw stiffness at its posture (the bar between the two arms, the
blower pulley's two rims and the motor's friction) lines the motor up at 30 mm for joint gains from 0.5 to 8 times
nominal, however the contacts hold the pitch, with the motor set down 3 or 6 mm proud (from 0.55 times at 9 mm); at 25
mm, 6 mm proud, only from 0.6 times. The end of that press does not depend on how proud the motor starts (only the
force at full overdrive does), and the first version's GPU run, with nominal-looking gains, lined it up from 8.7 mm.
More force cannot push the motor past the line while the bar is held at both places: the bar would have to lift off the blower pulley's outer rim, which takes the left arm pushing much harder
than the right. So the overdrive is kept equal on both arms (within 3 mm); more on the right than on the left lifts
the bar's far end off the motor. With the motor's centre of mass 40 mm up and 9 mm toward the pulley, a push more than
45 mm above its underside would tip it rather than slide it; the bar's top edge is 34 mm above it.

The run limit is 3,600 steps (3 minutes), 3.5 times the demo; the commands above pass `--max-steps 3600`.

What `TASK_TRACE` showed for each phase on the GPU (this version, job of 2026-11-12):

1. Settle (`DEMO_HOLD_ONLY=1`, 160 steps): nothing moved: the motor's `tilt_deg` 0, `e_mm` 12.1, `theta_deg` 3.0,
   `lowest_corner_mm` 0; the straightedge stayed on its felt.
2. The slide (8.1 to 13.6 s): `stayed_on_plate` and `stayed_on_line` true on every row; `lowest_corner_mm` 0.0 (the
   leading edge on the plate); `tilt_deg` up to about 1.7 while sliding (model 2.6; the tipped-up limit is 10), 0
   once let go; `y_off_mm` -1.2; `left_cmd_err_mm` about 7 to 8 during the slide (model 11). When the motor was let
   go (about 15 s): `e_mm` -11.1 (proud), `theta_deg` 1.05, `plate_margin_mm` -0.8 (the front corners just past the
   plate's front edge, the underside's centre well inside it: the end gate (c) only).
3. The press: the motor started to move at about 35 s; by about 40 s `e_mm` was 0.0 and `theta_deg` 0.01, and they
   stayed there through the hold and the draw-back. `bar_pitch_deg` rose to about 6.6 (the top edge leading, from the
   soft wrists), the motor's `tilt_deg` stayed under 0.2, `y_off_mm` drifted to -2.4, and `left_cmd_err_mm` and
   `right_cmd_err_mm` were 31.2 and 31.4 at the hold: the arms' give, with the bar stopped on the line.
4. The finish: `failed_gates` empty, `clear_mm` 97 mm or more for all four, `arms_unavailable` 0, `hands_off_steps`
   11, `plate_breach` and `line_breach` null, `max_lowest_corner_mm` 0.0, `max_y_off_mm` 2.4, M 0.008 degree
   (`e_mm` 0.0, `theta_deg` 0.004): 1.0.

## Offline checks

A kinematic offline check carries a gripped object rigidly and drops it when let go, and nothing pushes anything, so when the
hooks get a motor state whose `source` is "offline" (only a kinematic replay says so), `task.py` feeds the scorer the
motor's pose from `offline_model.py`: the motor keeps its own pose, taking the checker's whenever a gripper carries it;
a gripper at most 30 mm open with its pad-tip point within 15 mm of the straightedge holds it; when the bar's face,
at the motor pulley's face, comes past that face, the bar pushes the motor along x with its face flat on the bar: held
on both sides of the blower pulley, never past the line; held only on the motor's side of the blower's outer rim, on
to the bar's face (the overshoot); held only outside the outer rim, not at all. It never pulls the motor back. It has
no forces, lag, sag, pitch or friction: the GPU decides the real end state. Through it the demo predicts 1.0 and the
two one-arm runs 0 and 0. Because the checker carries a gripped part rigidly, offline the slide takes the motor all
the way to the command (the face 21 mm proud at x 29.7 cm, its front corners 10 mm past the plate's front edge with
its centre 35 mm inside it) and the ease-off brings it back to x 31.0 cm (8 mm proud); its lowest corner stays on the
plate's top and its y on the line throughout, so neither whole-trial rule is ever broken.

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

Unpowered, from common materials, matching the setup: a 120 x 100 x 140 mm blower box (sheet steel, or MDF, painted
dark grey) screwed to the table, with a 90 mm pulley 20 mm thick (a cast-iron sheave, or one turned from steel) on a
fixed 16 mm steel rod, its axis 52 mm above the table; a 160 x 130 x 12 mm galvanised steel plate taped down beside it
as the motor base (x 26.2 to 42.2 cm); a printed 90 x 70 x 80 mm motor shell weighted to 150 g, the weight placed so
that its centre of mass sits 40 mm above its underside and 9 mm from the body's centre toward the pulley, with a 64 mm,
18 mm thick printed pulley on a 16 mm steel rod 40 mm above its underside (the two pulleys' faces equally far from
their grooves); a 480 mm length of 20 x 40 mm aluminium box section (1.2 mm wall) on a 40 x 500 mm felt strip. Tape a
y mark with ticks 10 mm either side for the motor's line and the x 20.0 cm reference line for measuring; keep a
try-square, a steel rule, a feeler gauge, a card ruled 2 mm above the plate's top for the side camera, and two cameras
(top and side, level with the plate).

Third-party assets: none (every shape is built from boxes and cylinders in `task.py` and `mechkit.py`).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
