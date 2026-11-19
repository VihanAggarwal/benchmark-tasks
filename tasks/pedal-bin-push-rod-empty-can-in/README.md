# Throw an empty can into a pedal bin, the lid opened with the pedal only

A Brabantia NewIcon 3 L pedal bin (concrete grey, 264 mm high, 168 mm across, 0.83 kg) stands loose on the table,
pedal toward the robot. An empty 330 ml drink can stands upright beside it. The robot must put the can in the bin and
leave the lid shut, opening the lid only through the pedal, the way the bin is meant to be used: no hand on the lid.

The bin stands on a raised step at the robot's working height (the table stands for it), so a gripper does the foot's
job. The pedal lifts the lid only while it is held down, so one arm holds the pedal down while the other carries the
can over the opening and drops it in; the pedal is let go last, gently, once the other hand is clear.

## Why it is hard

1. **Hold one thing while doing another.** The lid is open only while the pedal is down. One arm must press about
   30 mm and hold for several seconds while the other picks up, carries and drops the can.
2. **Order and timing.** Drop the can before the lid is fully up and it hits the lid's underside; let the pedal go while
   the other hand is still over the opening and the lid shuts on it; let it go with a snap and the bin rocks.
3. **A light, loose bin.** 0.83 kg on a non-slip base. A press that is slanted, off the plate's middle or too hard
   slides or tips it (it tips past 30 degrees: 0).
4. **A light, thin can.** 14 g of aluminium about 0.1 mm thick: it must be held gently on its stiff top band (about 5 N
   a finger, see "The can and the grip"), and dropped from too high it bounces off the rim or the lid.

## Scoring

| Condition | Rule |
|---|---|
| In bin | the can's highest point 10 mm or more below the rim, its centre within 72 mm of the bin's axis (top view), its lowest point no more than 4 mm below the inner bucket's floor |
| Shut | the lid within 5 degrees of closed on every step of the last 1 s |
| Still | over the last 1 s the bin and the can each moved under 5 mm and turned under 5 degrees |
| Bin kept | the bin within 30 mm of where it started and tilted under 5 degrees, at the end |
| Pushed (latched) | the lid moved by anything but the pedal linkage, at any step, by any of: (a) a gripper's finger zone (a capsule round the last 38 mm of its fingers, larger than the fingers) within 1 mm of the lid; (b) the lid rising while more than 5 degrees above the angle the pedal's position and the push rod give it (a hand or arm on the lid, or on the rod); (c) the lid more than 5 degrees above the rod's angle and not closing for 3 steps, 0.15 s (held open: a free lid falls over 20 degrees a second); (d) the can within 1 mm of the lid while the lid rises more than 2 degrees above that angle (the can lifting it) |
| Can hit (latched) | at any step the can within 1 mm of the lid (the simulation's margin for a touch, stricter than the real check) |
| Fall (latched) | the bin tipped over 30 degrees or slid over 100 mm, at any step |

| Outcome | Score |
|---|---|
| In bin, never pushed, shut, still, bin kept, the can never hit the lid | 1.0 |
| In bin, never pushed, but not all of the rest | 0.5 |
| The lid pushed, lifted or held by anything but the pedal (any way, at any time) | 0 |
| Not in bin, or a fall | 0 |

Possible scores: 0, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0 (the can starts on the table). No tier
rewards a lid opened any way but the pedal: the 0.5 tier needs "never pushed" too, and covers only a pedal-opened trial
that ends untidy (the lid not yet shut, the can or bin still moving, the bin moved, or the can grazing the lid).

Other bypasses, and why each scores 0:

| Bypass | What stops it |
|---|---|
| Lift the lid by hand, or with the wrist or forearm (no finger near it) | (a) for the fingers; (b) for any other part: the lid rises above what the pedal gives |
| Press the pedal, then hold the lid up with an arm and let the pedal go | (c): the lid stays above the rod's angle instead of closing |
| Push the rod up in its slot, the pedal untouched | (b): the angle is the lower of the rod's and the pedal's, and the pedal's stays at 0 |
| Push the carried can up through a part-open lid | (d), and (b) beyond 5 degrees |
| Tip the bin so the lid swings open | the lid's weight stays in front of its hinge up to 30 degrees of tip, so it cannot swing open; past 30 degrees the fall latch gives 0, and any opening without the pedal is (b) |
| Wedge the lid open, or leave the can on the pedal | a wedge must first lift the lid ((a) to (d)); the 14 g can is far under the 1.5 N the pedal needs at its stop, so it cannot hold the pedal down |

Bench check for each condition on a real trial (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| In bin | lid held open after the trial: the can rests inside on the bucket floor; a straight-edge across the rim clears the can's top by 10 mm (a 10 mm block fits under it) |
| Shut | a 5 degree mark at the hinge, 30 fps side video of the last 1 s |
| Still | a 5 mm scale beside the bin in the same video |
| Bin kept | tape marks round its base before the trial, a 30 mm rule |
| Pushed | 30 fps side video: no frame shows a gripper or arm on the lid or the can lifting it; the lid is never more than 5 degrees above what the pedal's position gives except while it closes (a chart of pedal travel against lid angle, measured once with a protractor; a 5 degree mark at the hinge) |
| Can hit | the same video; the can seen or heard touching the lid (a near miss that does not touch is not counted) |
| Fall | a 30 degree angle gauge, a 100 mm rule |

## The bin's linkage (simulated, no rule)

The pedal plate is the front end of a lever that pivots under the bucket's floor. The lever's rear end, behind the
bucket, is a round nose under a loose push rod (8 x 7 x 214 mm, 15 g) standing in a slot in the back wall. The rod's
flat top sits 0.5 mm under a round pin on a short arm under the lid's hinge (12 mm in front of the hinge axis, 8 mm
below it). Pressing the pedal raises the rod, the rod raises the pin and the lid swings open; the pedal's stop (14
degrees of lever, 30 mm at the plate's middle) leaves the lid open 67 degrees, its front edge 413 mm over the table
(411 mm published). Let go, the lid's own weight pushes the rod and the lever back, the pedal comes up, and the lid
hinge's damping (0.10 N m s/rad, est) makes the soft close (about 1.4 s from fully open). Every link is a contact
between separate rigid bodies (the rod is its own loose part); no spring, no software rule, no latch.

### Pedal force (est) and the arm

| Lid angle | 0 | 10 | 30 | 50 | 67 (stop) degrees |
|---|---|---|---|---|---|
| Force at the plate's middle (est) | 4.8 | 4.2 | 3.3 | 2.4 | 1.5 N |

From the lid's 0.12 kg (its centre 91.5 mm in front of the hinge, over the bin's axis) through the pin, rod and lever (121 mm from pedal
middle to pivot, 69 mm from pivot to nose), less the lever's own 60 g; friction not counted (`layout.pedal_force`). At
the demo's press pose 5 N needs about 1.7 N m at joint 2 and 1.4 N m at joint 3, a small part of what those joints
give; the arm's give at that pose is about 10 to 15 mm per 5 N, which is why the demo commands the finger corner
15 mm below the pedal's stop. The press point is 27 mm past the base's front edge, so 5 N there tilts the 0.83 kg bin
against about 0.68 N m of its weight: it stays put unless the press is far harder or pulls sideways.

## The can and the grip

| | Value | Source |
|---|---|---|
| Size | 66 mm across, 115 mm tall (330 ml standard can) | Wikipedia, Drink can |
| Mass, empty | 14 g (est; weigh one on a kitchen scale) | estimate |
| Wall | about 0.1 mm aluminium; the top 12 mm necks in to the seamed lid, the base is a domed, thicker cup | estimate |
| Grip | 5 N a finger: the demo commands 64 mm across the 66 mm can, and the simulator's finger drive pushes 5000 N/m times the 1 mm each finger stops short | the program's simulator notes |
| Where | the pads meet the side 74 to 106 mm up: the top band of the straight wall, ending at the necked shoulder (about 103 mm up), which with the seamed top stiffens it against an inward push; the middle of the wall (57 mm up), the easiest place to dent, is 17 mm below the pads | geometry (test_can_grip_force_and_height) |
| Holding margin | 2 pads x 5 N x 0.5 friction = 5 N of friction against the can's 0.14 N weight | |

Published crush figures for empty drink cans are for end-on (axial) loads, 679 to 1193 N for 190 to 500 ml cans (Wang
et al., Materials 17 (2024) 2655); they say nothing about a side squeeze, and we found no published side-squeeze
figure for an empty 330 ml can. So the setup does not rely on one: before trials, an empty can is squeezed across the
same band between two 27 mm wide flat pads with 10 N (twice the grip, a kitchen scale under it) and must show no dent.
The demo's earlier 60 mm grip (about 15 N a finger) was cut to 5 N in the 19 Nov revision. The simulated can is rigid
and cannot dent, which is why the grip force and height are fixed in the demo and checked on the bench instead.

## Results

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each, 19 Nov 2026 (job
20261119T005526, the revised scorer and demo). The offline column is `tools/offline/check_task.py` with this task's offline stand-in for the
physics (`task.OfflineModel`): a prediction, not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`), 160 steps | both arms hold still | 0.0 | 0.0 (nothing moved) |
| Scripted demo | below | 1.0 (held from step 380) | 1.0 (success from about step 369 to the end at step 385; the lid opened to 67.1 degrees, the can came no closer than 32.7 mm to the lid, ended upright on the bucket floor, the lid shut, the bin moved 0.3 mm) |
| `DEMO_IDLE_ARM=left` | the right arm presses and lets go; the can stays on the table | 0.0 | 0.0 (the lid opened to 67.1 degrees and shut; the can never moved) |
| `DEMO_IDLE_ARM=right` | the left arm drops the can onto the shut lid; it falls off | 0.0 | 0.0 (the can hit the shut lid, which counts as pushed; it ended off the bin and the bin moved 15 mm) |
| Dream-1 | `--policy dream`, full 800-step limit | not predicted | 0.0 (the can was nudged 100 mm across the table; the lid never opened) |
| `DEMO_VARIANT=half` | the pedal pressed half way: the lid stops at 33 degrees; the carried can and the fingers meet it | 0.0 (pushed and can hit; 0.25 under the old tiers) | not run on the GPU |
| `DEMO_VARIANT=early` | the pedal let go 0.2 s after the can: the closing lid meets the leaving hand | 0.0 (0.25 under the old tiers) | not run on the GPU |

The 19 Nov run is the revised demo (64 mm grip on the can's top band, about 5 N a finger) scored by the revised
rule: the lid never rose more than 0.01 degrees above what the pedal gives, the fingers never touched it, and pushed
stayed unset. The 17 Nov run (job 20261117T151627, old demo and tiers) gave the same lid and can numbers.

An earlier GPU run (job 20261117T104005) scored 0.25 (0 under the revised scorer: the fingers touched the lid, and the
can and hand lifted it from 47.8 to 57.8 degrees with the pedal fixed). In it the pedal lever's front arm, as first modelled, met the
table at 10.1 degrees of lever turn, before the pedal's stop, so the lid opened only 47.8 degrees and the can and the
fingers met it. The lever's underside was then raised to 30 mm (it now clears the table by 5.7 mm at the stop; the
real lever's shape under the floor is not published) and the demo lowers the can 60 mm into the opening before letting
go.

Two arms are needed: the lid is open only while the pedal is down, the pedal's middle is 48 mm over the table and
the opening 248 mm over it, and nothing on the table is heavy enough to hold the pedal down (the can is 14 g; the pedal
needs about 4.8 N). One gripper cannot hold the pedal down and be over the opening at the same time.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right gripper's fingers
close into a block and lean 25 degrees forward; the left grip leans 60 degrees forward, its fingers reaching down and
round the can.

| Time (s) | Right gripper |
|---|---|
| 0 to 4.5 | in front of the bin with the fingers closed into a block; over the pedal; down to 8 mm over the plate |
| 4.5 to 7.0 | press the plate's middle to the stop (finger corner commanded 15 mm below it): the lid opens 67 degrees |
| 7.0 to 15.0 | hold the pedal down until the left hand is back and up, out of the lid's sweep |
| 15.0 to 17.5 | rise slowly 60 mm: the pedal comes up and the lid closes behind the hand |
| 17.5 to 19.0 | aside |

| Time (s) | Left gripper |
|---|---|
| 0 to 4.3 | over the can, down round it (fingers open, 94 mm), settle |
| 4.3 to 5.5 | grip the can across its top band, the pads 74 to 106 mm up (64 mm commanded across its 66 mm: about 5 N a finger) |
| 5.5 to 10.0 | lift it 15 mm over the rim's height; over the opening, 32 mm in front of the bin's axis (the lid fully open by 7 s) |
| 10.0 to 11.7 | lower it 60 mm into the opening (its bottom 45 mm under the rim, the fingers still over it); settle |
| 11.7 to 12.7 | open to 76 mm: the can drops about 150 mm to the bucket floor, inside the walls |
| 12.7 to 16.0 | back toward the arms and up, out of the lid's sweep; aside |

Offline: worst IK 0.09 mm; minimum joint margin 11.0 degrees (left joint 3); with the lid at 67 degrees the can passes
33 mm from it and the scorer's finger zone (as wide as the GPU measured the grip round the can) 22 mm; at 60 degrees 14
and 5 mm; the zone reaches the 1 mm touch margin at about 59 degrees, so the lid must open that far. Clash check along
the whole demo: closest TCPs 244 mm apart, arm link capsules 28 mm; the right arm passes 8.9 mm from the bin's wall on
its way to the pedal, the lowered left hand 18 mm.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 800
```

Runtime: an 800-step trial takes about 2 to 3 minutes of wall-clock on the L40S including start-up (the demo itself is
380 steps, 19.0 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed. For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes
one line per control step: lid and pedal angles, the rod's rise and the lid angle it allows, the can's position, the
closest the can came to the lid, bin drift and tilt, both TCPs and openings.

## Simplifications

- Rigid bodies. The round steel body and the plastic inner bucket are one wall of 16 flat slabs with a floor 50 mm up;
  the inner bucket does not lift out. The lid is a flat plate with a raised middle.
- The linkage is reduced to its working parts: one lever, one rod, one pin. A real NewIcon's linkage is hidden in the
  body; its parts' sizes here are estimates chosen to give the published open height.
- The soft close is hinge damping, not a separate damper.
- Masses inside the bin, friction (bin base 0.8, can 0.6, rod 0.15) and the pedal force are estimates.
- The pedal is pressed by a gripper, not a foot: the bin stands on the table, which stands for a step at the robot's
  working height. The pedal, rod and lid work as they do underfoot; only the presser differs.
- The touch checks (fingers on the lid, can on the lid) use small margins (1 mm each) in the simulation, so they are
  slightly stricter than a real trial's "seen touching".
- The can is rigid: it cannot dent or crush. The demo keeps its grip to about 5 N on the can's stiff top band, and the
  setup's bench check (10 N, no dent) stands for the dent the simulation cannot show.

## Sources and measurements

- The bin: Brabantia NewIcon pedal bin 3 L, concrete grey, product listing with dimensions (H 26.4 x W 16.8 x D 23.5 cm,
  open height 41.1 cm, 0.83 kg, non-slip base, soft-closing lid):
  https://www.johnlewis.com/brabantia-newicon-pedal-bin-3l/concrete-grey/p110488610
- The can: a standard 330 ml aluminium drink can, 66 mm across and 115 mm tall: https://en.wikipedia.org/wiki/Drink_can
- Empty-can crush loads (axial only, 679 to 1193 N for 190 to 500 ml cans): Wang et al., "On the Crush Behavior and
  Energy Absorption of Sustainable Beverage Cans and Their Polyurethane Foam-Filled Structures", Materials 17 (2024)
  2655, https://pmc.ncbi.nlm.nih.gov/articles/PMC11174009/

Everything else is an estimate (est in task.json). How to measure each on a bought bin:

| Estimate | Sim value | How to measure |
|---|---|---|
| Pedal force at the plate's middle | 4.8 N at the start, 1.5 N at the stop | 0-10 N spring scale hooked under the pedal's middle, pulled down slowly; acceptance: 10 N or less |
| Pedal travel at the plate's middle | 30 mm | ruler beside the pedal, up and fully down |
| Lid angle fully pressed | 67 degrees | protractor at the hinge; acceptance: 60 degrees or more |
| Soft close time | about 1.4 s | phone video at 30 fps from fully open, pedal let go |
| Rim height, opening across | 248 mm, 144 mm | ruler and calipers |
| Inner bucket floor height | 50 mm | ruler down the inside |
| Lid mass | 0.12 kg | kitchen scale (the lid unclipped, or the bin weighed with and without it) |
| Empty can mass | 14 g | kitchen scale |
| Can under the grip | no dent at 10 N (twice the 5 N grip) | two 27 mm flat pads across the band 74 to 106 mm up, pressed on a kitchen scale |
| Friction | 0.8 base, 0.6 can | tilt a board with the bin (the can) on it until it slides: friction is the tangent of that angle |

## Real build and reset

A Brabantia NewIcon 3 L pedal bin that passes the acceptance checks (task.json setup, item 3), an empty 330 ml drink
can, tape for the bin's marks, a protractor, a 0-10 N spring scale, a 10 mm block, a 30 mm and a 100 mm rule. Reset: take
the can out and stand it upright at x 260 mm, y 240 mm; stand the bin on its marks at x 330 mm, y 0, pedal toward the
arms, lid shut, the inner bucket seated. Nothing is consumed.

## Revision (19 Nov)

The review of 19 Nov 2026 said:

> Flags: Note for the team: Check that the real empty can can be grasped without crushing it.
> Construct validity (not met): A robot can lift the lid by hand and place the can inside for 0.25, bypassing the
> required pedal operation.
> Scoring (not met): The explicit 0.25 tier rewards a completion that violates the task's defining pedal-only
> requirement.
> What to change: The pedal-only requirement is the central physical challenge, but the current 0.25 tier rewards
> opening the lid by hand. Change the primary score to zero whenever the lid is pushed or lifted by anything other than
> the pedal linkage, and update the scorer and documentation accordingly.

What changed:

1. **Pushed now scores 0.** The 0.25 tier is gone; the scores are 0, 0.5 and 1.0, and every tier above 0 needs "never
   pushed" (scoring.tier; tests test_pushed_always_scores_zero, test_lid_opened_by_hand_and_the_can_put_in).
2. **Pushed catches every non-pedal lid motion we could find,** not only fingers on the lid: the margin above the
   linkage's angle is 5 degrees, not 10 (GPU run 1 had the can and hand lift the lid 9.9 degrees, just under the old
   margin); the linkage angle now also follows the pedal, so a rod pushed up by hand counts; a lid held open after the
   pedal is let go counts; the can lifting the lid counts. The table of bypasses under Scoring lists each with its
   test. On the 17 Nov GPU demo trace the lid never sat more than 0.12 degrees above the linkage's angle.
3. **The can.** The demo's grip was cut from about 15 N a finger (60 mm commanded) to about 5 N (64 mm) and stays on
   the top band of the wall under the necked shoulder, not its middle; the setup adds a bench check (10 N across that
   band, no dent). See "The can and the grip".
4. task.json (setup, scoring, fidelity), this README, scoring.py, layout.py, task.py and the tests agree; the offline
   check still predicts 1.0 for the demo and 0 for each arm alone, and now 0 (was 0.25) for both wrong-way variants.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `parts.py`, `scoring.py` | Every dimension and the linkage geometry; the parts as shapes; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tiltgrip.py` | Pitched grips for the demo's waypoints |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
