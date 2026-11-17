# Swap a hand blender's shaft for the whisk: both release buttons in, whisk clicked on

A hand (immersion) blender, reference unit Braun MultiQuick 5 Vario (MQ5235 class), lies unplugged on its back on the
table. Its blending shaft is clicked into the motor part's coupling end; a wire whisk with its gearbox lies to one
side; a cork mat lies to the other. The job is the swap done in the middle of a recipe: draw the shaft off, set it down
on the mat, push the whisk on until it clicks, and let go. The instruction does not say that there are two buttons,
that both must be in to draw the shaft, or that the whisk must click on both hooks.

The coupling is a push-on fit. The attachment's stub (a 33 mm core with a 38 mm square lip at its tip) goes into a
41 mm square socket; each flank carries one release button on a spring (2 N, est) that works a latch hook on the
socket's opposite wall, behind the lip: pressing a button moves its hook out of the lip's way. With both buttons in, the shaft slides out; with one hook still in, the lip catches on it and the shaft
binds. Pulled with no button in, the shaft drags the whole blender along the table. The whisk goes on the same way:
the lip rides up both hook ramps (about 2 N, est) and the hooks snap in behind it when the shoulder is on the face.

## Why it is hard

1. **Two buttons, one grip.** The pads of one hand close across the 64 mm body at the buttons and press both flush at
   once (the real two-finger squeeze), while the other hand draws the shaft by its bell. One button alone leaves the
   other hook in: the shaft binds, and the arm's force (about 5 N) cannot overcome it.
2. **Real size, real tolerances.** The lip has 1.5 mm of play in the socket; the whisk's gearbox must be lined up
   within 3 mm and 3 degrees, kept square in its turn (a square lip turned a few degrees jams in the lead-in), and
   pushed 21 mm in on its axis, the last 2 mm against the hook ramps.
3. **Hold the body, then hold it again.** The blender is a free 0.62 kg body: the hand that squeezes must let go
   (the buttons spring out) and re-grip the body away from the buttons, so that the whisk's push does not drag it.
4. **Hands off.** The score is read with both grippers 30 mm clear and every part still for 1 s. A whisk held in
   place does not count.

## Scoring

| Condition | Rule |
|---|---|
| Shaft off | not in the socket, wherever it lies; it counts as in the socket only while the stub's tip is more than 0.5 mm past the coupling face with its axis within 30 mm of the socket's and 30 degrees of its direction (on the mat, on the table or held up, it is off) |
| Shaft on mat | shaft off; its mid-point within 100 mm of the mat's centre in x and y; its lowest point within 5 mm of the mat's top |
| Whisk home | stub in the socket, gearbox shoulder within 1 mm of the coupling face, axis within 3 mm of the socket's, square within 3 degrees |
| Both hooks behind the lip | whisk home and both buttons within 0.4 mm of their rest height |
| Whisk on | home with both hooks behind the lip |
| Whisk seated | home but a button more than 0.4 mm in (a hook rides on the lip) |
| Hands off | last 1 s: both finger zones 30 mm or more from every part; every part moves under 5 mm and turns under 5 degrees |
| Fallen (latched) | any part's lowest point more than 30 mm below the table top |

| Outcome | Score |
|---|---|
| Shaft on the mat, whisk on, hands off | 1.0 |
| Whisk on, shaft elsewhere on the table; or shaft on the mat, whisk seated | 0.5 |
| Shaft off, otherwise | 0.25 |
| Shaft still on, a part fallen, or not hands off | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. Doing nothing scores 0. Success: 1.0 held for 1 s (on top of the scorer's own 1 s
of hands off and stillness).

The scorer reads poses, the two buttons' joint values and both grippers; there is no software rule for the latches.
A hook is behind the lip when its button is out while the attachment is home, because a hook that is not behind the
lip rides on it (the geometry stops the lip against a hook that is in).

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Shaft off | look into the socket from the side and from above: no stub inside |
| Shaft on mat | 100 mm rule from the mat's centre to the shaft's mid-point (from above); 5 mm feeler under its lowest point |
| Whisk home | a 1 mm feeler does not enter between gearbox and coupling face at the top, bottom or either side; protractor, 3 degrees or less between the two axes; a 3 mm rule for the axis offset |
| Whisk on, each hook separately | hold the motor part, press the +x button only and hold it in, pull the whisk straight out with a 10 N spring scale: it stays on (the -x hook holds it); release, press the -x button only and pull again: it stays on (the +x hook holds it). A whisk that comes off in either pull is seated, not on. Do these tests last |
| Seated (one hook or none) | the whisk home by the feeler but it comes off in one of the pulls above, or a button stands more than 0.4 mm in (dial gauge or 0.4 mm feeler under the cap) |
| Hands off | a 30 mm spacer block against each hand; 30 fps video of the last 1 s with a 5 mm ruler and a 5 degree mark in view |
| Fallen | by eye; a rule for 30 mm below the table top |

## Results

Run on an L40S (Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1,500-step limit), one trial each. The offline
prediction column (`tools/offline/check_task.py`, with this task's offline stand-in for the physics,
`task.OfflineModel`) is not a measurement. The GPU column is one run of each; rows marked not run were not run there.
On the GPU the demo's success held from step 713.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run: nothing moves) |
| Scripted demo | below | 1.0 (held from step 712) | 1.0 |
| `DEMO_IDLE_ARM=left` | only the right arm works: it draws the shaft with no button in and drags the blender | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=right` | only the left arm works: the shaft is never drawn | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=shaft_table` | the shaft laid on the table 230 mm off the mat, whisk fitted | 0.5 | not run |
| `DEMO_VARIANT=no_whisk` | the shaft on the mat, the whisk left lying | 0.25 | not run |
| `DEMO_VARIANT=grip_away` | the left hand grips the body away from the buttons: both hooks hold the shaft | 0.0 | not run |

The task needs both arms: one squeezes the buttons and holds the body, the other draws the shaft; without the
squeeze the shaft cannot come out, and without a hand on the shaft nothing comes out either.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both grippers close along x
(left yaw -90 degrees, right +90 degrees); pad tips 24 mm over the table.

| Time (s) | Left gripper (squeeze and hold) | Right gripper (shaft, then whisk) |
|---|---|---|
| 0 to 6.5 | over the buttons (y +65 mm), down, close 56 mm across the 64 mm body: both buttons flush | over the shaft's bell end (y -136 mm), down, grip it 52 mm on the 60 mm bell |
| 6.5 to 9 | hold | draw the shaft 40 mm straight out along its axis |
| 9 to 17 | let go (buttons spring out), up, over the body at y +190 mm, down, grip it away from the buttons | lift, carry to the mat, lower, let go, up |
| 17 to 31.6 | hold the body | over the whisk's gearbox (x 200 mm), grip it, lift 25 mm, carry in line with the socket, lower to the socket's axis, push 30 mm/s until the shoulder is on the face (8 mm of overtravel: about 3 N), let go |
| 31.6 to 34.1 | let go, up, aside | up, aside |

Offline: worst IK 0.09 mm; smallest joint margin 11.5 degrees (right joint 4, at the mat); the two forearms pass 21 mm apart at the
start of the descent (link capsules, t 2.6 s) and the grippers stay 200 mm apart. The demo is 682 steps (34 s), under
half of the 1,500-step limit; the scripted policy stops after 743.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Take the blending shaft off the hand blender and lay it on the cork mat, then fit the whisk to the blender and let go of everything." \
  --scorer task_score --max-steps 1500 --store-frames
```

Runtime: a 1,500-step trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (the demo itself
is 682 steps, 34 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: every part's pose, both
buttons' travel, the shaft's and the whisk's position on the socket's axis, the conditions, both TCPs and openings.
`DEMO_HOLD_ONLY=1` is the settling check (nothing may drift), `DEMO_IDLE_ARM=left|right` the one-arm check,
`DEMO_VARIANT=<name>` the counterfactuals above (never the submitted demo).

## Simplifications

- Rigid bodies. The motor part is one body with two sprung slider links (the buttons); the oval is an octagon
  (10 mm chamfers); the whisk wires are solid hulls; the shaft's bell is built 60 mm across (the unit's 64 mm is an
  estimate) so that it lies level with the body.
- Each button is one sliding frame round the socket (a mechkit slider): the cap on its own flank, bars inside the
  body's walls, and the hook on the socket's opposite wall, so the press moves the hook out of the lip's way. The hook
  is a wedge (25 degree ramp, flat back face 0.4 mm behind the lip), pushed aside by the lip and sprung back by the
  button's spring: joints and shapes, no software rule. The ramp's crest (17.2 mm from the axis) is the hook's tip; the
  flat block behind it stands 0.5 mm further out (its back face bites 1.3 mm into the lip), so the lip slides off the
  crest without meeting the seam between the hook's two collision pieces. The frame does not collide with its own
  body, and only its hook comes within the socket's 20.5 mm (a unit test checks both ends of its travel).
- The motor does not run; the 1.2 m cord is not modelled (in a real trial it is unplugged and taped along the back).
- Estimates, to measure on a bought unit: the socket and stub sizes, button travel and spring force (2 N), the
  coupling's push and pull forces, masses, friction (0.35 plastic, 0.3 steel). The real trial takes any unit that
  passes the acceptance checks under "Real build and reset"; the simulated forces are estimates inside those bounds.

## Sources

- Reference unit: Braun MultiQuick 5 Vario hand blender (MQ5235 class). The manual describes changing between the
  shaft, whisk and chopper by pressing the two buttons and pushing the attachment on until it clicks; the coupling's
  dimensions are our estimates and are to be measured on the unit bought.
- No third-party assets, meshes or scans: every shape is a box or a cylinder, so there are no asset licences to list.

## Real build and reset

A Braun MultiQuick 5 Vario with its shaft and whisk; a 200 x 200 x 10 mm cork mat; the cord unplugged and taped along
the motor part's back. Reset: push the shaft onto the motor part until it clicks (collar face on the coupling face)
and lay the blender on its back along y at x 300 mm, coupling end at y 20 mm; press both buttons and draw the whisk off
if it is on; lay the whisk at x 200 mm, gearbox toward +y, shoulder at y -40 mm. Nothing is consumed; between trials
check that both hooks spring back and that the shaft binds with one button in.

Acceptance checks on the bought unit before the first trial (0-50 N spring scale, calipers); a unit that fails one is
not used, and the measured values replace the estimates above:

| Check | Pass |
|---|---|
| Width across the two buttons (calipers) | 75 mm or less (the gripper's pads must close across them) |
| Squeeze both buttons flush across the body (spring scale on a two-finger clamp) | under 30 N (the fingers manage about 40 N each) |
| Both buttons in, draw the shaft off along its axis | under 5 N |
| One button in (each in turn), pull the shaft | stays on at 10 N |
| Push the whisk on until it clicks | under 15 N |

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `parts.py`, `scoring.py` | Every dimension; the shapes of each part; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and sprung sliders; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |
