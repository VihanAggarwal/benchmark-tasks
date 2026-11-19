# Collapse an empty carton flat on the stack of knocked-down cartons it lies on

Three knocked-down 12 x 8 x 6 in cartons (508 x 356 x 6.4 mm each, flaps out straight) lie stacked on a fixed brown
board. On top of the stack lies an empty regular slotted carton (RSC) of the same size with all eight flaps attached,
on one long side with its open ends along x: a four-panel sleeve of 3.2 mm single-wall C-flute, 203 mm tall, 305 mm
long in y, with a 102 mm flap on each panel at each open end. It was emptied through its near (-x) end, whose flaps
were folded back then and lie out straight. The tape on its far (+x) end has just been slit, and those four flaps hang
loose at uneven angles, as a slit end's flaps do: the top one droops 45 degrees over the opening, the bottom one
stands 30 degrees up off the stack, the -y end panel's hangs 30 degrees out past straight, the +y one's is straight.
Its four corners are creases; one of them is the manufacturer's glue joint, a 38 mm tab glued to the underneath panel.
The robot must collapse it flat (two layers, 6.4 mm, every flap out straight) so it lies on the stack, leave the
stack as found and take its hands away. The instruction names neither the push, the direction nor the flaps.

## Revision (2026-11-18)

The reviewer's feedback on the previous version, verbatim:

> Construct validity (not met): Cutting all the flaps off removes the attached cardboard that workers normally open
> and handle when flattening a carton.
> Scoring (not met): The required 1 mm and 0.3 degree whole-object stillness check cannot be reliably verified for
> every panel and flat using the specified video and single dial indicator.
> What to change: Use a normal carton with its flaps attached and specify how its opened flaps are arranged at reset;
> update the footprint, scorer and fidelity note, and provide a full-score demo of the revised setup. Replace the 1 mm
> and 0.3 degree whole-object stillness test with a practical, video-verifiable no-visible-motion check.

Each point, and its answer:

- **"Cutting all the flaps off removes the attached cardboard that workers normally open and handle."** The carton is
  now a stock 12 x 8 x 6 in RSC with all eight flaps: four 305 x 102 mm on the long panels and four 203 x 102 mm on
  the end panels, each 4 in deep (half the 8 in width, as on every RSC). The robot now handles them: the trial starts
  just after the tape on the far (+x) end is slit (the knife step is its own act, left out for safety), with that
  end's four flaps loose at uneven angles, and every flap must end within 15 degrees of straight out. The robot's
  rack-over brings the top panel down so the stack's top folds its drooping flap out, and the robot then presses the
  two raised flaps (the bottom one and the -y end panel's, which turns up with its panel) out flat. Without that
  press the carton ends racked flat with two flaps standing about 50 mm proud: 0.25 (offline,
  `DEMO_VARIANT=flaps_left`).
- **"Specify how its opened flaps are arranged at reset."** Near (-x) end, the one it was emptied through: its four
  flaps were folded back hard then and lie out straight (within 10 degrees; the sim sets them straight). Far (+x)
  end, tape just slit: top flap drooping 40 to 55 degrees (sim 45), bottom flap 20 to 40 degrees up off the stack
  (sim 30), -y end panel's flap 20 to 40 degrees out past straight (sim 30), +y end panel's within 10 degrees of
  straight (sim straight), each checked with an angle gauge. The carton spans x 92 to 448 mm.
- **Flap creases.** Each flap is a rigid 3.2 mm board plate on its own crease, drawn at its start angle. Its crease
  behaves as a scored crease in corrugated board does: springy for a few degrees about where it rests (its set),
  and, bent further, it yields and takes a new set where it is left. In the sim the spring is a joint drive toward
  the set (0.3 N m per radian, est), and a declared per-step rule moves the set (the drive's target) to the flap's
  angle once the flap is more than 3 degrees from it (est). The spring holds a level long flap's weight (0.0085 N m)
  with 1.6 degrees of sag, inside the 3 degrees, and holds up to 0.016 N m before the crease yields (0.2 N at a pad
  70 mm out). Opened, a flap stays open; pressed flat, it stays flat (springing back at most 3 degrees); nothing in
  the model springs it straight. (Changed on 19 Nov from 0.03 N m of joint friction, which the GPU showed does not
  hold a flap: see below.)
- **"Update the footprint, scorer and fidelity note."** Flattened with its flaps out, the carton is 516 x 356 mm, so
  the footprint limit is 540 x 380 mm; the stack's flats are the same carton knocked down flaps out (508 x 356 x
  6.4 mm, 0.20 kg) on a 540 x 380 mm board. "Flat" also needs every flap within 15 degrees of straight out (read from
  its crease angle plus its start angle), and the 12 mm thickness counts the flaps. The fidelity note states the flap
  model.
- **"Provide a full-score demo of the revised setup."** Done: the scripted demo scores 1.0 on the simulator with
  the flaps attached (19 Nov, below), success held from step 411. The first GPU run of the revised carton (18 Nov)
  had scored 0.25, racked flat with two far flaps standing, because the flaps' joint friction let them creep (see
  the GPU finding below); the flap crease model of 19 Nov fixed that. The screenshot and videos are from the 19 Nov
  run.
- **"Replace the 1 mm and 0.3 degree stillness test with a video-verifiable no-visible-motion check."** In the sim,
  over the last 1 s no panel, flap or flat may move 5 mm or more or turn 5 degrees or more; in the real trial, no
  visible motion of the carton or the stack over the final 1 s in the top video (the station's overhead camera) and
  the side video (a camera on the far side of the table at the stack's height, looking along -x at the carton's full
  length). The dial indicator is gone.

### GPU finding, 18 Nov (demo 0.25), and the demo change

What the trace showed (`trace_sampled_demo`, 830 steps): the push and the weight racked the sleeve flat by step
181 (interior angles 180, 0, 181, 0), the stack's top folded the drooping top flap out (cf 47.7 to 1.2 degrees
from step 161 to 181) and the -y end panel's flap folded itself out and was pressed (df -0.2 at the end). The two
far flaps of the LOWER layer, which lie on the stack, did not stay down: once the carton was flat they crept tip up
by themselves at a steady rate whatever their angle, the underneath panel's (af) about 0.12 degrees a step
(2.4 degrees a second: -31 at step 181, -55 at 341, pressed to -8.4 at 371, -16.9 at 441, -63.1 at the end) and the
+y end panel's (bf, which starts straight) 0.03 to 0.065 a step (0 at 181, -29.8 at the end). The joint velocity
the simulator reports meanwhile is about +0.001 rad/s while af's position falls 0.034 rad/s, against both its weight
and its spring: a position-level drift of the flap creases, not a motion the 0.03 N m friction resists (the
program's other GPU notes found the same, joint friction letting a link creep with no drag). Until a press put them
right, the near flaps crept the same way and propped the underneath panel up 6.8 degrees. The end: af 63 degrees up,
bf 30 up, 92.6 mm thick, so not flat: 0.25. The one flap press (x 410, y -150 mm) had caught af, but 23 s before the
end, and nothing ever pressed bf (y 51 to 254 mm once flat, out of the right arm's reach).

The demo change (task.json unchanged): the right arm's first flap press moved in to 49 mm out from the crease (it
catches af standing up to 61 degrees; at 64 mm it had only just caught it); both arms press along the top at the same
times, 260 mm apart; last, both press the far flaps together 70 mm out (right: af and df; left: bf and cf), pad tips
2 mm into the flat top, and lift in 0.6 s, so the last press ends about 3.5 s before the last step (the policy's own
hold is 3 s). The offline model now includes the creep at the trace's high-end rates (af 2.8, bf 1.3 degrees a
second once flat), and a pad that comes down beyond a steep flap's tip misses it.

Why the demo alone could not reach 1.0: the 12 mm thickness needs af within about 5 degrees of straight at the
last step. At the creep the GPU showed, af passes that within about 2 s of release, and at least 3.5 s must pass
(hands off and still for 1 s inside the policy's 3 s hold), so under the 18 Nov flap model this demo predicts 0.25
offline.

### The flap model change, 19 Nov

A real flap stays where it is put because its crease takes a set, not because of a dry friction torque. The flap
creases now model that directly: no joint friction (PhysX joint friction appears to be what crept here), a joint
spring toward the flap's set (0.3 N m per radian, est) that holds at the position level, and a declared rule, run at
the start of every control step in the simulator, that moves a flap's set to
where the flap is once it is bent more than 3 degrees from it (est). A drifting flap is pulled back by the spring
(up to 0.016 N m before the crease yields, about twice a long flap's weight); a pushed flap takes its new set and
stays, springing back at most 3 degrees, inside the 12 mm thickness's 5 degrees (3.2 for a flap of the upper layer).
The rule does not touch the corner creases, the rack, the scorer or the demo; the setup and fidelity text now say
this instead of the 0.03 N m friction. Offline, the same demo predicts 1.0 (held from step 428).

Two other crease models are kept in the code for comparison, each run with the submitted demo's plan
(`DEMO_VARIANT=crease_set,crease_stiff,crease_ratchet`): `crease_stiff` (1.0 N m per radian, sets past
1.5 degrees: holds 0.026 N m) and `crease_ratchet` (as `crease_set`, but the set follows a flap only toward
straight). On the simulator (19 Nov, variant job 20261119T003430) all three scored 1.0, so the result does not hang
on the estimated spring and yield. `friction_hold` (the 18 Nov model) scored 0.25 on the GPU. The trace carries each
flap's set (`flap_sets_deg`) beside its angle (`flaps_deg`).

### GPU result, 19 Nov (job 20261119T012651, crease set model, the demo below)

2026-11-19, job 20261119T012651-00: settle 0.0, demo 1.0, right arm alone 1.0, left arm alone 0.0, Dream-1 0.0.
Demo trace: the push and the weight racked the sleeve flat by step 176 (interior angles 180, -0.6, 181.5, -0.8 at
the end); the stack's top folded the drooping top flap out (cf 46 to 1.2 degrees, steps 161 to 176); with no creep
(af stayed at -28.6 degrees from step 181 to 356, its set unchanged), the right arm's press at x 395 mm folded the
underneath and -y end panels' far flaps down (af -28.6 to -1.6, df -28.8 to 0.0, steps 356 to 376). End state: every
flap within 1.6 degrees of straight, thickness 6.47 mm, footprint 516.1 x 355.8 mm, centre 1.0 mm off the stack's,
lowest point 0.01 mm under the stack's top, stack moved 0.0 mm (lift 0.01 mm, turn 0.0 degrees), hands 64 mm or
more clear. Score 1.0 from step 411 and held; the trial ended at step 429 with success held 1 s, before the demo's
later presses. Settle (no actions, 160 steps): every flap within 1.6 degrees of its start angle, no set moved,
score 0.

## Why it is hard

1. **Only one family of folds is flat.** A sleeve is knocked down by racking it sideways: the end panels swing over
   and the top panel slides across the bottom one. Pressed straight down, a sleeve buckles one way or the other;
   pressed at the wrong height, the end panel just slides on the stack. Here the four corner creases turn together
   (one angle, 0 to 90 degrees; see `layout.py`) and the carton ends 516 mm long, which is longer than the 305 mm it
   started; the near end's flaps ride along straight out, and the flat carton is 356 mm wide across them.
2. **Where it ends up counts.** Racked toward -y the carton hangs off the stack's end (0.5); racked toward +y it
   lies along the stack (1.0). The stack must stay put (15 mm, 8 degrees): the carton rests on it and a push needs a
   reaction, which here comes only from friction between carton and stack (0.5).
3. **The loose flaps.** The far end's flaps start at uneven angles and every one must end within 15 degrees of
   straight. The drooping top flap is folded out by the stack's top as the top panel comes down onto it (pressing
   the top finishes it if the flap props the panel up); the -y end panel's flap turns up as its panel goes over, and
   it and the raised bottom flap must be pressed flat without dragging the carton or the stack. Racked flat with
   those two left up, the carton scores 0.25.
4. **Past about 45 degrees the weight takes over.** The creases hold the sleeve (0.01 N m each) until the lean is
   enough for its own weight to carry it over (the offline model assumes 46 degrees, an estimate); a hand that stops
   short leaves a leaning sleeve (0), a hand that keeps pushing drags the stack.
5. **Hands off.** The score is read with both grippers 30 mm clear and nothing visibly moving for 1 s: a carton held
   down or held up earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Flat | corners within 15 deg of shut or open; flaps within 15 deg of straight; 12 mm, flaps included; 540 x 380 mm |
| On the stack | footprint centre within 60 mm of the stack's centre (top view); lowest point within 6 mm of its top |
| Offset | the same within 120 mm and 20 mm |
| Stack kept | no flat moved over 15 mm (top view) or 4 mm up, or turned over 8 deg |
| Hands off | last 1 s: fingers 30 mm+ from carton and flats |
| Still | last 1 s: no panel, flap or flat moves 5 mm or more or turns 5 deg or more (no visible motion) |

| Outcome | Score |
|---|---|
| Flat, on the stack, stack kept | 1.0 |
| Flat, offset or leaning on the stack, stack kept | 0.75 |
| Flat anywhere else (or stack moved 15 to 40 mm) | 0.5 |
| Not flat but 110 mm tall or less (over half way) | 0.25 |
| Anything else, or not hands off and still | 0 |
| Any flat moved over 40 mm or turned over 30 deg (latched) | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Flat | 15 degree angle gauge at each corner and flap crease; rule across the top, 12 mm block; tape: 540 x 380 mm |
| On the stack | rule and straight edge across the stack for the centre (60 mm) and the lowest edge (6 mm) |
| Stack kept | pencil mark round each flat before the trial; rule for 15 mm; protractor for 8 degrees |
| Over half way (0.25) | rule on the tallest point of the carton, 110 mm |
| Hands off | 30 mm spacer against each hand, checked on the video of the last 1 s |
| Still | no visible motion of any panel, flap or flat over the final 1 s in the top and side videos (below) |
| Knocked (latched) | pencil mark and rule: 40 mm; protractor: 30 degrees |

Videos for the real trial: the top video is the station's overhead camera, looking down on the stack; the side video
is a camera on the far side of the table, 1 m from the stack at the height of its top, looking along -x at the
carton's full length (it shows the thickness and any flap lifting).

Doing nothing scores 0 (the carton stands). Success: 1.0 held for 1 s.

## Results

Revised carton (flaps attached), flap crease model of 19 Nov. Simulator column: job 20261119T012651 (19 Nov).

| Run | Offline prediction | Simulator (Isaac Lab-Arena, seed 0) |
|---|---|---|
| Scripted demo | 1.0 (held from step 428) | 1.0 (held from step 411) |
| Right arm alone (left idle) | 1.0 | 1.0 |
| Left arm alone (right idle) | 0.0 | 0.0 |
| Settle (no actions) | 0.0 | 0.0 |
| Dream-1, full run limit | n/a | 0.0 |

The previous version (the same carton with its flaps cut off) scored 1.0 with the scripted demo on the simulator
(Dream-1 0.0); those figures do not carry over to the revised carton.

The right arm does the essential work: it racks the carton over and presses its own side's raised far flaps; with
the crease set, the +y end panel's far flap (straight at the start, out of the right arm's reach once flat) stays
straight, so the right arm alone scores 1.0 (offline and on the simulator). The left arm's presses along the top
and over that flap are a margin against the layers and that flap lifting; on the simulator the trial reached
success and ended (step 429) before them. The task does not claim to need two arms. The offline predictions
for the counterfactuals: `DEMO_VARIANT=shallow` (pushed only to 0.55 rad) 0; `DEMO_VARIANT=flaps_left` (the demo
without the flap presses) 0.25.

Runtime per trial: the demo plan is 35.5 s (710 steps at 20 Hz: 32.5 s of waypoints and the scripted policy's 3 s
hold) of the 1200-step limit, and on the simulator it ends at step 429 (21.5 s) once success has held 1 s; a
Dream-1 trial runs the full 1200 steps (60 s of simulated time). Seed: nothing is random (`object_jitter=0`); the
start layout is fixed by `layout.py`.

## Demo

Open loop, timed waypoints, nothing read from the simulator. Wrists straight down, gripper closing axis along y
(yaw 0). No grasp: every contact is the closed pad block's face. The push and the presses along the top are on the
carton's centre line (x 270 mm); the flap presses are 49 and 70 mm out from the far flaps' crease. The push needs
well under 1 N (four 0.01 N m corner creases over a 0.12 m lever, plus damping); a 3 to 7 mm trail of the arm under
load still leaves the lean past 0.9 rad. Offline capsule check of the previous demo (the arms' links, the task's own
offline model posing the carton each step): no link but the fingers came within 34 mm of any flap; not yet rerun for
the new flap presses.

1. Right arm: fingers closed into a block on the way (never open over the carton), to the air 12 mm short of the -y
   end panel's foot at x 270 mm, then straight down outside the panel to 120 mm over the stack's top; push +y for
   about 168 mm at 30 mm/s (until the panel leans 54 degrees); hold 3 s while the weight finishes the rack (the
   top panel comes down onto its drooping flap and the stack's top folds that flap out); lift clear.
2. Right arm: one press at x 395 mm, y -150 mm, pad tips 2 mm into the flat carton's top: it folds the underneath
   panel's raised far flap (and the -y end panel's above it) back down (49 mm out, it still catches that flap
   standing up to 61 degrees).
3. Both arms together, 260 mm apart: presses along the top at x 270 mm, right at y -20 then -140 mm, left at
   y 240 then 120 mm, pad tips 1 mm into the flat carton.
4. Both arms together, last: a press at x 416 mm, right at y -150 mm (the underneath and -y end panels' far flaps),
   left at y 150 mm (the +y end and top panels'), pad tips 2 mm into the flat top; lift in 0.6 s and stay there.

## Simplifications

- Panels are rigid 3.2 mm plates on four revolute joints (0.01 N m friction, 97 degrees of travel either side of
  square). Real corrugated creases spring back and tear.
- Flaps are rigid 3.2 mm plates, one on each panel's edge at each open end, on revolute creases drawn at their start
  angles: a spring (0.3 N m per radian, est, a joint drive) holds each at its set, and a declared rule moves the set
  to the flap's angle once it is bent more than 3 degrees from it (est), as a scored crease takes a set; no joint
  friction; 97 degrees of travel either side of straight. The real crease's yield is gradual, not a step at
  3 degrees, and a real crease also creeps back over minutes and can tear; none of that is modelled. The slots
  between flaps (about 1/4 in) are left out: each flap is drawn as long as its panel.
- The glue joint is a 38 mm tab on a fourth crease, fixed to the underneath panel by a mechkit bond (a FixedJoint,
  same technique as the tape bonds in other tasks in this program) that never breaks; it closes the loop, so the
  sleeve is a four-bar with one degree of freedom. The tab and the end panel overlap in the flat state (no panel
  collides with another).
- The panels and flaps do not collide with one another (one mechkit assembly; they do collide with the stack and the
  grippers), so the flat state is read from the crease angles and the 12 mm thickness limit, not from contact. For
  the same reason the far end's flaps start where none passes through another: the -y end panel's flap hangs out past
  straight rather than folded in (folded in, it would turn down into the stack as the sleeve racks over, and the
  straight-down wrists cannot reach behind it under the top flap to open it first), and the +y one's is straight
  (folded in, it would cross the drooping top flap). Once flat, the raised bottom flap shows up through the top
  layer's flaps until it is pressed down. On a real carton those flaps would lie on it instead, propped up by it
  (the top panel's flap comes down partly onto it, the -y end panel's flap lies over it), and the same press folds
  both layers down together: the act, the press and the end state are the same, and nothing in the score depends on
  the overlap (the 12 mm thickness counts every flap, so a raised flap fails it either way). The model draws each
  crease offset by one board thickness so the two layers come out 6.4 mm thick.
- Friction 0.5 and the 0.55 kg/m2 board are estimates.
- The stack's flats are single slabs.

## Sources

- Stockers and order fillers: BLS, SOC 53-7065 (about 2.8 million US jobs).
- A 12 x 8 x 6 in carton and 1/8 in (3.2 mm) C-flute are catalogue sizes; areal mass 0.55 kg/m2 is an estimate.
- Flap depth: on a regular slotted carton (FEFCO code 0201) every flap is half the carton's width deep, so the long
  flaps meet in the middle when closed: 8 in / 2 = 4 in (101.6 mm). The knocked-down size, 508 x 356 mm, follows
  (two panels plus two flap depths across).
- Masses (0.20 kg a carton) are the board area times the estimated 0.55 kg/m2; the corner crease friction, the flap
  crease spring and its 3 degree yield are estimates (measure with a spring scale on a level panel or flap: the
  torque that bends a flap 3 degrees and leaves it bent).

## Real build and reset

Four empty single-wall C-flute regular slotted cartons 12 x 8 x 6 in, flaps attached, glue joint intact; three of them
knocked down flaps out and stacked on a 540 x 380 mm board. Reset: the three flats on the board, centred, edges
square; lay the fourth carton on its long side on top, glue tab at the -y end, underneath panel from y -254 to
+51 mm. Near (-x) end: fold each flap back hard once and let it lie out straight (within 10 degrees). Far (+x) end:
slit the tape along the seam (a fresh carton) and leave the flaps loose at the start angles above, checked with an
angle gauge: top 40 to 55 degrees drooping, bottom 20 to 40 up, -y end panel's 20 to 40 out past straight, +y end
panel's within 10 of straight. Replace a carton whose creases have gone limp;
single use after about five trials. Cameras for the real trial: the station's overhead camera (top video) and one
camera on the far side of the table, 1 m from the stack at the height of its top, looking along -x (side video).

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the flap set rule, `success()`, `score()`, `demo()`, variants, offline |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and mechanisms; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Run it: `inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...`
(scripted demo); `DEMO_HOLD_ONLY=1` (settle), `DEMO_IDLE_ARM=left|right` (one arm), `TASK_TRACE=<file>` (trace).

Third-party assets: none.
