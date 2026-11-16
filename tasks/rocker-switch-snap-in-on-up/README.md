# Press a rocker switch into a panel cutout with its I mark toward the top edge

A grey steel control panel (100 x 160 x 1.5 mm, one 22 x 30 mm cutout) lies flat on two 40 mm steel rails, as it does
on an assembler's bench. Its far edge carries an orange band: that is the panel's top edge. On a foam pad in front of
it stands a KCD1-type rocker switch (body 28 x 20.5 x 25 mm, flange 31 x 25 mm, rocker 25 x 18 mm and 8 mm proud, 12 g),
rocker up, its printed I half toward the robot: wrong way round for this panel. The robot must pick the switch up by its
rocker, turn it half a turn about the vertical, press it into the cutout until its flange lies on the plate, and let go.
The instruction names neither the turn nor the order of moves.

## Why it is hard

1. **A small part in a small hand.** The only part of the switch a top-down gripper can hold is the rocker, 18 mm deep
   and 8 mm proud of the flange. The pads close on its two long faces and rest on the flange beside it.
2. **Half a turn in the hand.** The switch starts with its I half toward the robot. It has to be carried and turned
   180 degrees about the vertical (a joint 6 turn of 3.1 rad) without being dropped or moved off the grip's centre: a
   2 mm grip offset becomes 4 mm after the turn.
3. **A close fit.** The body has 0.75 mm of play each side along x and 1.0 mm along y in the cutout, with its rear edges
   eased 2 mm (est). Crooked by more than about 3 degrees about the vertical it jams on the cutout's edges.
4. **Seated means flush.** The flange's rear face must end within 0.5 mm of the plate, square to it. Proud, or crooked,
   scores a quarter.
5. **Hands off.** The score is read with both grippers 30 mm clear and the switch still for 1 s: a switch held in
   place earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| In | flange's rear face within 0.5 mm of the plate's top (highest corner); switch's centre within 2.5 mm (x) and 3 mm (y) of the cutout's centre |
| Square | switch's axis within 3 degrees of the vertical |
| Orientation | I half within 20 degrees of +x (toward the orange band): top; within 20 degrees of -x: bottom |
| Hands off and held | last 1 s: finger zones 30 mm or more from the switch; it moves under 0.5 mm and turns under 0.3 degrees |
| Fall (latched) | lowest point ever 10 mm below the table top (off the table); set down on it and installed later is fine |

| Outcome | Score |
|---|---|
| In, square, I toward the top edge | 1.0 |
| In, square, I toward the bottom edge (not turned) | 0.5 |
| In the cutout, proud by 0.5 to 3 mm, or crooked 3 to 8 degrees, or turned neither way | 0.25 |
| Anything else, or not hands off and still, or fallen | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| In (0.5 mm) | a 0.5 mm feeler gauge under the flange at its four corners; a look from above at the body centred in the cutout |
| Square (3 degrees) | a try-square on the plate and the switch's flange, or the feeler: 0.5 mm over the 25 mm flange is 1.1 degrees, 1.3 mm is 3 degrees |
| Orientation (20 degrees) | a look at the printed I against the orange band; a 20 degree protractor if in doubt |
| Hands off | a 30 mm spacer block against each hand; 30 fps video of the last 1 s |
| Held (0.5 mm, 0.3 degrees) | a 5 N spring scale pushing up on the body's rear end through the gap between the rails: the switch does not move 0.5 mm (a feeler or dial indicator on the flange). This stands in for the real tabs' 25 N retention |
| Fall (10 mm) | the switch on the floor instead of the table, by eye |

Possible scores: 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

## Results

GPU runs (L40S, one trial each) fill the GPU column. The offline prediction column (`tools/offline/check_task.py`,
with this task's offline stand-in for the physics, `task.OfflineModel`) is a kinematic check, not a measurement.
The nothing, `unturned` and `offset` variants have not been run on the GPU.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run |
| Scripted demo | below | 1.0 (held from step 479) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the same demo: it uses only the left arm | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | nothing happens | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=unturned` | placed without the half turn | 0.5 | not run |
| `DEMO_VARIANT=offset` | put down 6 mm off the cutout's centre line | 0.0 | not run |

One arm is enough here: the left arm reaches both the foam pad and the cutout, so the right arm is not needed. The task
does not claim to need two.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator.

| Time (s) | Left gripper (fingers closing along x) |
|---|---|
| 0 to 6.8 | over the switch at x 200 mm, down beside the rocker (pad tips 0.5 mm over the flange), settle, close on the rocker |
| 6.8 to 11.3 | lift 65 mm, carry 160 mm to x 360 mm over the cutout, turning the gripper half a turn |
| 11.3 to 14.4 | settle, lower until the body's rear face is 4 mm over the plate, settle |
| 14.4 to 18.2 | press slowly: the pad tips come down onto the flange (commanded 2 mm further), settle |
| 18.2 to 23.2 | open, straight up, aside to y 340 mm |

Offline: worst IK 0.09 mm; the smallest joint margin 14 degrees (joint 4) and 17 degrees (joint 6, at the turn);
the right arm holds its ready pose, 36 cm clear.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: a trial of up to 1200 steps takes about 2 to 4 minutes of wall-clock on the L40S including start-up (the demo
itself is about 480 steps, 24 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes
the start layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the flange's gap above the
plate, the tilt, the switch's offset from the cutout's centre, its orientation, both TCPs and openings, and a `hist` of
the last 3 s of those numbers in every row.

## Simplifications

- Rigid bodies. The real switch's spring tabs are not modelled: there is no snap (about 15 N) and no retention (about
  25 N). The switch goes in under its weight and the pads' push and holds by lying flange down on the plate, so the
  held check is "still, with no hand on it", not a measured pull-out. The rocker is flat and rigid; the real one tilts.
  No wires are fitted.
- The panel lies flat. A vertical panel would need a sideways push the top-down demo cannot give.
- Sizes are catalogue-class estimates (est): KCD1-101 class bodies about 28 x 20.5 mm with a 31 x 25 mm flange for a 30 x
  22 mm cutout. Measure the real part and cutout; the 2 mm easing of the body's rear edges (a lead-in) is also est.
- Friction 0.40 plastic on steel (est). The simulator's own grip friction applies to the pads.
- The foam pad is a rigid block.

## Sources

- Team assemblers, O*NET 51-2092.00: well over 500,000 US workers (BLS Occupational Outlook Handbook, assemblers and
  fabricators), who fit panel components by hand.
- KCD1-101 rocker switch (20 A 250 V, 30 x 22 mm panel cutout): catalogue-class dimensions, not measured here.
- Reach figures: `tools/offline/README.md` (the left arm reaches both stations at table height; joint 6 turns 180 degrees
  between them with 17 degrees to spare).
