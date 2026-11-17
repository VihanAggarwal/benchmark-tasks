# Unlock a new lotion bottle's pump and turn its spout to you, with no squirt

A new 12 fl oz (355 ml) white HDPE lotion bottle stands on a rubber shelf liner on a vanity top, in front of a tiled
splash wall. Its black lock-down pump head sits pressed down on the closure with its spout pointing along -y. The robot
must twist the head a quarter turn anticlockwise (seen from above) so the lugs clear and the pump spring lifts it, then
swing the spout round to point at the robot (-x), and leave the bottle where it was, hands off. The head rides on a
spring: once it is free, any push of more than 4 mm downward is a stroke (lotion squirted over the hand, liner or tiles)
and scores 0. The instruction names neither the follow-up lift nor the swing path.

## Why it is hard

1. **The turn needs a real grip.** The head is 32 mm across and the lugs take about 0.035 N m. A pinch on the cap has to
   carry that torque without the fingers slipping (a flat pad on a round, ribbed cap), while the bottle stays on the
   liner (it resists about 0.08 N m).
2. **The head springs up under the hand.** At 80 degrees the lugs clear and the spring lifts the head 23 mm (9 N at the
   lock, 1.6 N over the first 4 mm). The hand has to go up with it while it is still holding and turning it. Holding the
   height, or pushing down, makes the head a stroke or a jam; the scorer reads the head's travel at every step.
3. **The wrist runs out of turn.** The head must turn about 270 degrees of spout travel (unlock 95 degrees, then round
   to -x), more than one grip allows (the wrist covers about 220 degrees), so the head is turned in several grips, the
   open hand lifted over the 48 mm spout and turned in the air between them. The pads must close across the spout, not
   onto it.
4. **Hands off.** The score is read with both grippers 30 mm clear and the bottle still for 1 s.

One gripper is enough. The left arm is not needed (the liner holds the bottle), which the one-arm runs below confirm:
this task makes no claim to need two arms. Its difficulty is the force and height control, not holding.

## Scoring

| Condition | Rule |
|---|---|
| Standing | tilt under 8 degrees, base centre within 30 mm of its start (top view) |
| Unlocked | head turned 80 degrees or more at some moment (the lugs cleared) and now 18 mm or more above its locked height |
| Spout | the spout's direction within 15 degrees of -x, seen from above |
| Stroke (latched) | once free, the head more than 4 mm below the highest it has been |
| Tip (latched) | tilt over 45 degrees; or the base 50 mm below its start height (off the vanity) |
| Hands off | last 1 s: finger zones 30 mm or more from every part; bottle moves under 5 mm and turns under 5 degrees, head turns under 5 degrees and moves under 5 mm |

| Outcome | Score |
|---|---|
| Standing, unlocked, spout to the front | 1.0 |
| Standing, unlocked, spout elsewhere | 0.5 |
| Standing, not unlocked, head turned 30 degrees or more at some moment | 0.25 |
| Anything else, a stroke or tip, or not hands off | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. Which way the spout was swung is not scored. Doing nothing scores 0. Success: 1.0
held for 1 s.

Bench method for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Standing | 8 degree angle gauge on the bottle; a ring template on the liner for the 30 mm |
| Unlocked | steel rule from the closure's top to the head's top (18 mm more than locked); the video for the 80 degree turn |
| Spout | protractor on the table under the spout against a line along -x through the bottle's start mark (15 degrees) |
| Stroke | 60 fps side video of the head against a rule: any frame with the head 4 mm below its highest |
| Tip | by eye and a rule (base height); 45 degrees from the video |
| Turned 30 degrees (0.25) | the video against a mark on the closure |
| Hands off | 30 mm spacer block against each hand; 30 fps video of the last 1 s (5 mm, 5 degrees, with a mark and a rule) |

## Acceptance checks for the real bottle

The bottle and pump are standard packaging-supplier stock (a 12 fl oz HDPE Boston round with a 24/410 lock-down lotion
pump; many body-lotion and hand-soap bottles use the same pump). Buy three and keep one that passes:

| Check | Range | How |
|---|---|---|
| Locked height / free height | 140 to 155 mm / 163 to 178 mm, travel 20 to 26 mm | steel rule against a block |
| Turn to unlock | quarter turn (80 to 100 degrees), 0.03 to 0.06 N m | a 100 mm arm taped on the head, a spring scale pulled at right angles: 0.3 to 0.6 N |
| Stroke start | 1.5 to 4 N for the first 4 mm | kitchen scale pressed on the free head |
| Free swivel | under 0.02 N m | the same arm and scale: under 0.2 N |
| Body | 60 to 70 mm across, 355 ml | calipers |
| Bottle on the liner | does not turn under a 0.8 N pull on the 100 mm arm (0.08 N m) | spring scale |

Protection from lotion: use the bottle dry. Cut the dip tube off and put a sealed bag of dry sand (about 0.36 kg) inside
to 0.42 kg, so a stroke pumps only air and nothing reaches the pads, the liner or the wall. A fresh bottle, or the same
one pressed down and re-locked (turn the spout to -y, press the head down, a quarter turn clockwise), is used per trial.
If the bottle's own spring differs from the table by more than 30%, change `K_LIFT` in `layout.py` and re-run the demo.

## Results

Offline predictions (`tools/offline/check_task.py`, with this task's offline stand-in for the physics,
`task.OfflineModel`) are not measurements. The GPU column is one run of each on the simulator (Isaac Lab-Arena,
seed 0); the demo's success held from step 783 there too.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run) |
| Scripted demo | below | 1.0 (held from step 783) | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the same demo | 1.0 (the left arm is not used) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | no plan, the left cannot reach the bottle | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=half_turn` | head turned 45 degrees, let go | 0.25 | not run |
| `DEMO_VARIANT=no_swing` | unlocked, spout left at +x | 0.5 | not run |
| `DEMO_VARIANT=press` | head pushed down 30 mm once free | 0.0 | not run |

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. 760 steps (38 s), within half
the 1600-step limit.

| Time (s) | What the right gripper does (fingers across x, then y) |
|---|---|
| 0 to 6.5 | over the head, down round it (pads 134 mm over the table), settle, grip across its 32 mm |
| 6.5 to 11.5 | turn 95 degrees anticlockwise (lugs clear at 80), then follow the head up 22 mm as it springs |
| 11.5 to 14.5 | let go, up to the hover (pad tips 12 mm over the free spout) |
| 14.5 to 23 | grip again (pads across the spout at +5 degrees), turn 90 degrees clockwise, let go, up |
| 23 to 36.5 | turn the open hand 180 degrees in the air, grip, turn 95 degrees clockwise (spout now at -x), let go, up |
| 36.5 to 38 | aside |

Offline: worst IK 0.09 mm; joint 4 margin 6 degrees at the 200 mm hover, joint 6 margin 19 degrees at the heading
extremes; the left arm stays at its ready pose, 56 cm from the right TCP.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1600
```

Runtime: a 1600-step trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (the demo itself is
760 steps, 38 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the head's turn and
travel, the stroke depth, the spout's angle off the front, the bottle's pose, both TCPs and openings.

## Simplifications

- Rigid bodies. The bottle is cylinders (a real one has a moulded shoulder and a ribbed cap); the contents are a fixed
  mass and nothing is dispensed. The head is a cylinder with a box spout and nozzle.
- Two declared software rules (`_bayonet_rule`, a mechkit on_step callback). Until the head has turned 80 degrees: the
  slider's spring target holds the head down at the lock, and the hinge's torsion spring (0.025 N m/rad) pulls it back
  to 0; this is how the lock's turning torque is stated (a head let go before the lugs clear springs back; a real one
  stays where it was left). From 80 degrees on, the target is the free height (the 400 N/m spring lifts the head) and
  the hinge holds where it is; the swivel friction is the joint's damping (0.01 N m s/rad). The scorer reads poses and
  joint positions, not the rule.
- Friction 0.8 bottle, 1.0 liner, 0.4 wall (est). The pump spring is linear; a real one has a preload (the stroke
  start force is checked on the real bottle instead).

## Sources

- Retail listing of a 16 fl oz Boston-round lock-down pump bottle (3 in across, 7.6 in tall locked, 8.5 in free, so
  23 mm of travel): used for the head's travel and the quarter-turn lock; a 12 fl oz bottle's sizes are design values
  inside the acceptance ranges above. 24/410 is the standard neck finish for these pumps.
- Pump listings: pressed all the way down and turned clockwise to lock, counterclockwise to unlock; a laboratory
  supplier's 500 ml HDPE pump bottle with a 24 mm locking closure lists "locks with 1/4 turn"
  (https://www.fishersci.com/shop/products/pump-dispensing-bottle/03406163).

Estimates (stated in fidelity): the lug torque, the free swivel friction, the spring rate, the frictions, the shoulder
and closure sizes.

## Real build and reset

A 12 fl oz HDPE Boston-round bottle with a 24/410 lock-down pump (dip tube cut off, a sealed bag of dry sand inside to
0.42 kg), a 220 x 220 x 3 mm rubber shelf liner, a 300 mm tall tile or board as the splash wall 150 mm behind the
bottle's axis, a steel rule, an 8 degree angle gauge, a protractor, a 30 mm spacer, a spring scale. Reset: turn the
spout to -y, press the head down, a quarter turn clockwise until it locks; stand the bottle on its mark on the liner.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, bayonet rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound bodies with joints; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
