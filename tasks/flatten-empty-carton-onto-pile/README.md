# Collapse an empty carton flat on the stack of knocked-down cartons it lies on

Three knocked-down 12 x 8 x 6 in cartons (508 x 152 x 6.4 mm each) lie stacked on a fixed brown board. On top of the
stack lies an empty carton of the same size, flaps already cut off, on one long side with its open ends along x: a
four-panel sleeve of 3.2 mm single-wall C-flute, 203 mm tall, 305 mm long in y. Its four corners are creases; one of
them is the manufacturer's glue joint, a 38 mm tab glued to the underneath panel. The robot must collapse it flat
(two layers, 6.4 mm) so it lies on the stack, leave the stack as found and take its hands away. The instruction
names neither the push nor the direction.

## Why it is hard

1. **Only one family of folds is flat.** A sleeve is knocked down by racking it sideways: the end panels swing over
   and the top panel slides across the bottom one. Pressed straight down, a sleeve buckles one way or the other;
   pressed at the wrong height, the end panel just slides on the stack. Here the four creases turn together (one
   angle, 0 to 90 degrees; see the table in `layout.py`) and the carton ends 516 mm long, which is longer than the
   305 mm it started.
2. **Where it ends up counts.** Racked toward -y the carton hangs off the stack's end (0.5); racked toward +y it
   lies along the stack (1.0). The stack must stay put (15 mm, 8 degrees): the carton rests on it and a push needs a
   reaction, which here comes only from friction between carton and stack (0.5).
3. **Past about 45 degrees the weight takes over.** The creases hold the sleeve (0.01 N m each) until the lean is
   enough for its own weight to carry it over (the offline model assumes 46 degrees, an estimate); a hand that stops
   short leaves a leaning sleeve (0), a hand that keeps pushing drags the stack.
4. **Hands off.** The score is read with both grippers 30 mm clear and everything still for 1 s: a carton held down
   or held up earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Flat | each crease within 15 deg of shut (0) or open (180); 12 mm thick at most; footprint within 540 x 180 mm |
| On the stack | footprint centre within 60 mm of the stack's centre (top view); lowest point within 6 mm of its top |
| Offset | the same within 120 mm and 20 mm |
| Stack kept | no flat moved over 15 mm (top view) or 4 mm up, or turned over 8 deg |
| Hands off | last 1 s: fingers 30 mm+ from carton and flats; each moves under 1 mm, turns under 0.3 deg |

| Outcome | Score |
|---|---|
| Flat, on the stack, stack kept | 1.0 |
| Flat, offset or leaning on the stack, stack kept | 0.75 |
| Flat anywhere else (or stack moved 15 to 40 mm) | 0.5 |
| Not flat but 110 mm tall or less (over half way) | 0.25 |
| Anything else, or not hands off | 0 |
| Any flat moved over 40 mm or turned over 30 deg (latched) | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Flat | 15 degree angle gauge at each crease; straight rule and feeler on top (12 mm); tape for 540 x 180 mm |
| On the stack | rule and straight edge across the stack for the centre (60 mm) and the lowest edge (6 mm) |
| Stack kept | pencil mark round each flat before the trial; rule for 15 mm; protractor for 8 degrees |
| Over half way (0.25) | rule on the tallest point of the carton, 110 mm |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s; dial indicator on the carton's top |
| Knocked (latched) | pencil mark and rule: 40 mm; protractor: 30 degrees |

Doing nothing scores 0 (the carton stands). Success: 1.0 held for 1 s.

## Results

| Run | Offline prediction | Simulator (Isaac Lab-Arena, seed 0) |
|---|---|---|
| Scripted demo | 1.0 (held from step 243) | 1.0 (success held at step 235) |
| Right arm alone (left idle) | 1.0 | 1.0 |
| Left arm alone (right idle) | 0.0 | 0.0 |
| Settle (no actions) | 0.0 | 0.0 (carton stands, nothing moves) |
| Dream-1, full run limit | n/a | 0.0 |

One arm is enough: the right arm does the rack-over and both arms only press the flat carton down. The offline
prediction for `DEMO_VARIANT=shallow` is 0.

Runtime per trial: the demo is 37.3 s (746 steps at 20 Hz) of the 1200-step limit; a Dream-1 trial runs the full 1200
steps (60 s of simulated time). Seed: nothing is random (`object_jitter=0`); the start layout is fixed by `layout.py`.

## Demo

Open loop, timed waypoints, nothing read from the simulator. Gripper closing axis along y (yaw 0).

1. Right arm: fingers closed into a block on the way (never open over the carton), to the air 12 mm short of the -y
   end panel's foot at x 270 mm, then straight down outside the panel to 120 mm over the stack's top; push +y for
   about 168 mm at 30 mm/s (until the panel leans 54 degrees); hold 3 s while the weight finishes the rack; lift
   clear.
2. Right arm: two presses along the top, y -140 and -20 mm, pad tips 1 mm into the flat carton; park.
3. Left arm (parked clear until the right arm is done): two presses at y 120 and 240 mm; park.

## Simplifications

- Panels are rigid 3.2 mm slabs on four revolute joints (0.01 N m friction, 97 degrees of travel either side of
  square). Real corrugated creases spring back and tear. The glue joint is a 38 mm tab on a fourth crease, fixed to
  the underneath panel by a mechkit bond (a FixedJoint, same technique as the tape bonds in other tasks in this
  program) that never breaks; it closes the loop, so the sleeve is a four-bar with one degree of freedom. The tab
  and the end panel overlap in the flat state (no panel collides with another).
- The panels do not collide with one another (one mechkit assembly), so the flat state is read from the crease angles
  and the 12 mm thickness limit, not from contact. The model draws each crease offset by one board thickness so the
  two layers come out 6.4 mm thick.
- Friction 0.5 and the 0.55 kg/m2 board are estimates.
- The stack's flats are single slabs.

## Sources

- Stockers and order fillers: BLS, SOC 53-7065 (about 2.8 million US jobs).
- A 12 x 8 x 6 in carton and 1/8 in (3.2 mm) C-flute are catalogue sizes; areal mass 0.55 kg/m2 is an estimate.

## Real build and reset

Four empty single-wall C-flute cartons 12 x 8 x 6 in, flaps cut off, glue joint intact; three of them knocked down and
stacked on a 540 x 175 mm board. Reset: the three flats on the board, centred, the fourth carton on its long side on
top, glue tab at the -y end. Replace a carton whose creases have gone limp; single use after about five trials.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variant, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and mechanisms; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Run it: `inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...`
(scripted demo); `DEMO_HOLD_ONLY=1` (settle), `DEMO_IDLE_ARM=left|right` (one arm), `TASK_TRACE=<file>` (trace).

Third-party assets: none.
