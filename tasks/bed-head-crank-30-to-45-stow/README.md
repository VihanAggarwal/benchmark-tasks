# Crank an empty manual bed's head section to 30-45 degrees and fold the crank away

On a manual bed the head section is raised with a foot-end crank. A nursing assistant (O*NET 31-1131.00, task line
"Turn or reposition bedridden patients.") unfolds the handle, turns it for several turns until the head section
stands at the right angle, and folds the handle away again. On a three-crank bed such as the Drive Medical 15003P
the left crank raises the head and the cranks fold away when not in use. The robot does the same: its gripper takes
the knob of the folded handle, swings the handle out 90 degrees, carries the knob round a 140 mm circle for 8 turns
against a steady load, and swings the handle back into its fold before letting go. The head section's angle is the
score.

**Scope: the bed is empty.** The trial has no patient and no mattress on the head section, so it tests the crank
handling (unfold, a steady multi-turn orbit against a load, counting turns, fold away), not a patient's weight. With a
patient on the bed the crank needs more torque than the 0.2 N m modelled here; that case is outside this task.

## Run

```bash
cd tasks/bed-head-crank-30-to-45-stow
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "<instruction from task.json>" --scorer task_score --max-steps 1800 --policy yam_scripted
```

`--policy yam_scripted` plays the demo (`task.demo()`); a model run uses the same command with its own `--policy`.
`DEMO_IDLE_ARM=left|right` leaves that arm at its ready pose (the one-arm check); `DEMO_HOLD_ONLY=1` leaves both
there (a settling check); `DEMO_VARIANT=short|over|toomuch|unfolded|held` plays a counterfactual (below).

- **Run limit:** 1800 control steps (90 s at 20 Hz). The demo needs 755 steps (the plan is 34.7 s, then the scripted
  policy holds the last pose for 3 s).
- **Runtime per trial:** about 12 minutes of GPU time for a full demo run including start-up, about 38 s of simulated
  time.
- **Seed:** none. The layout is the same in every trial (`object_jitter=0`) and nothing in the task is random.
- **Third-party assets:** none. Every part is built from boxes and cylinders by `mechkit.py` (in this folder).

## Why it is hard

- **A continuous 8-turn circular motion.** The knob runs on a 140 mm circle in the vertical y-z plane round a shaft
  that points at the robot: 440 mm a turn, 3.5 m for the demo's 8 turns, from 185 mm above the table at the top down to 45 mm at the bottom.
  A straight-down grip has to follow it all the way round, the bottom and the top included, with the knob spinning on
  its pin in the fingers.
- **Against a load.** The crank takes about 0.2 N m (2.9 N at the knob) at every moment: 0.05 N m of bearing
  friction and 0.15 N m for lifting the deck through the worm drive. The arm is compliant (a static estimate from the
  YAM's joint stiffness gives roughly 160 N/m at the TCP along x and 400 to 550 N/m along z), so a knob commanded on
  its own circle would run behind the command; the demo commands a circle 12 mm wider so the compliance carries it.
- **A fold with a hard stop.** The handle has to be swung 90 degrees about a vertical pin before the crank can be
  turned (the folded flat lies along the shaft and gives no lever), and swung back to within 15 degrees of the shaft
  at the end. The pin is vertical only when the crank stands at a whole number of turns, so the turns have to be
  counted, not eyeballed.
- **A number to hit.** The deck moves 5.0 degrees a turn: 6 to 9 turns for the 30 to 45 degree range, and past 12
  turns the deck is over 60 degrees, which scores 0 for the rest of the trial.
- **Nothing is chosen.** The layout is the same every trial and the instruction states the method; the difficulty is
  the coordinated physical motion, not a decision.

## Scene

Positions in mm from the midpoint between the arm bases: x forward, y to the robot's left, z up from the table top.

| Object | Real part | Size, mass, material | Start |
|---|---|---|---|
| Gearbox | the foot-end head-crank housing of a manual bed (modelled on a three-crank bed such as the Drive Medical 15003P; not measured on it) | box 90 x 90 x 160 mm, grey steel; fixed | x 340 to 430, y -145 to -55, on the table |
| Shaft | the crank's steel shaft | 12 mm across, 130 mm out of the gearbox's front face (tip at x 210), axis along x at y -100, z 115; bearing friction 0.05 N m | turning freely; crank angle 0 |
| Handle | the fold-away crank arm | steel flat 20 x 5 mm, 95 mm long, black, 0.1 kg; pinned on the shaft's tip, 70 mm from the pin to the knob axis; two stops 90 degrees apart, a spring-ball detent at each (0.04 N m) | folded: lies along the shaft toward the gearbox, tip at x 290 |
| Knob | the handle's grip | red nylon cylinder, 25 mm across x 70 mm long, 39 g, spins on its pin (0.002 N m) | folded: points along -y, centre (280, -145, 115); unfolded: points toward the robot, centre (165, -170, 115) |
| Head deck | the bed's head section | steel panel 450 x 550 x 20 mm, 3.0 kg (a pressed-sheet section, smaller than a real 900 mm wide one), hinged along y at x 500, z 270 | flat, resting on the frame post |
| Frame post | the bed frame under the hinge | steel 40 x 600 x 260 mm; fixed | x 480 to 520, centred on y -100 |

Sizes and torques are our estimates for a manual bed and are meant to be measured on the real bed (`fidelity` says
so). The 15003P's cranks fold down under the bed; this handle folds flat along its shaft instead. The deck sits at x 500 and above, out of the arms' reach (they reach x 450 only near the table).

## How the bed is built

One fixed mechkit assembly (`task.py` `BED`), its base frame on the table at (0, 0), not turned:

| Body | Joint | What it is | Limits, drive |
|---|---|---|---|
| base | fixed | gearbox and frame post | tight-fit contacts (0.5 mm offset, 32 solver iterations), friction 0.5 / 0.4 |
| `crank` | hinge about +x through the shaft | the shaft stub | +-6 rad (rewritten by whole turns, below); friction 0.05 N m; drive spring 2.0 N m/rad, damping 0.04 N m s/rad; the load rule sets its target |
| `fold` (on `crank`) | hinge about -z through the pin | the pin and the flat | 0 to 90 degrees (hard stops); friction 0.04 N m (the detents); damping 0.002 |
| `knob` (on `fold`) | hinge about the knob's own axis | the knob, free on its pin | +-6 rad (rewritten by whole turns); friction 0.002 N m |
| `deck` | hinge about -y through (500, -100, 270) | the head deck panel, 3 kg | 0 to 80 degrees; drive spring 1,000 N m/rad toward the rule's target, damping 40 N m s/rad (overdamped) |

**The worm drive** (`scoring.Winder`, run by a mechkit `on_step` callback at the start of every control step, before
the physics). The crank's angle is integrated unwrapped and clamped at 0 (the flat stop: turning backward at flat does
nothing). The deck's target angle is 5.0 degrees times the net turns, and never more than 78 degrees. The deck's
hinge is driven to that target, so the deck is self-locking by construction: whatever the gripper does to the deck
(it cannot reach it), the angle follows the crank. The score reads the deck hinge's own angle, not the rule's target.
Real counterpart: a worm drive between the crank and the head section, 72 crank radians per deck radian (5 degrees
a turn is our estimate; real beds are of the order of 4 to 6).

**The load.** While the crank turns forward it meets a drag of 0.15 N m (full above 0.5 rad/s, proportional below),
on top of its 0.05 N m bearing friction: it is a drive target on the crank's 2.0 N m/rad spring,
`target = q + w DT / 2 + (damping w - drag) / K`, which at steady speed averages a resisting torque of exactly the
drag over each 50 ms control step. The value: lifting a 3 kg deck by its centre (225 mm from the hinge) takes 6.6 N m
at flat, 0.092 N m at the crank through a 72:1 ratio, 0.2 N m at 46 % efficiency (a self-locking worm is under 50 %
efficient). The deck carries no patient in the simulator (a real one adds 60 to 100 kg of load, so the real crank
force is higher: measure it, the thresholds below do not depend on it). A forward-only load is a simulator choice: a
real friction also resists turning backward.

**Whole-turn rewrite.** The kit's hinges are limited to +-2 pi and the task takes 8 turns. So every control step, once
the crank's or the knob's angle passes +-pi, the rule rewrites that joint's angle by a whole turn (Isaac Lab's
joint-position writer, velocity unchanged, `task._write_joint_positions`); a whole turn leaves every part in the same
pose, so no contact changes, and the rule's unwrapped angle counts the short way across it (unit test
`test_wrap_counts_the_short_way`). If no writer works, the run prints `[bed] ERROR` once and the crank stops at its
6 rad limit.

**Deviations, each in `fidelity`.** The deck is a rule-driven hinge (no gearing is simulated); there is no patient
load; the detents are joint friction with hard stops; the deck panel is smaller than a real head section; the
gearbox is a box on the table.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`, its thresholds `layout.py`.

| Tier | Condition (simulator) | Real-trial check (bench method) |
|---|---|---|
| gate: 0 | the deck's angle exceeded 60.0 degrees at any step (latched for the trial) | the inclinometer's peak hold, or a marked 60 degree template at the deck's edge |
| 1.0 | deck angle 30.0 to 45.0 degrees, fold angle 15 degrees or less, every finger zone of both grippers 30 mm or more from the shaft, handle and knob on every step of the last 1 s (unreadable arms count as holding), deck angle within 0.5 degrees over that second | digital inclinometer on the deck (0.2 degrees, or the bed's own gauge); a 15 degree protractor card between the handle and the shaft; a 30 mm spacer against each hand; 30 fps video of the last second with the inclinometer in view |
| 0.5 | deck angle 30.0 to 45.0 degrees but not the whole top condition (handle left out, a gripper within 30 mm, or the deck still moving) | the same instruments |
| 0.25 | handle folded and the deck at 15.0 up to under 30.0 or over 45.0 up to 60.0 degrees | the same instruments |
| 0 | anything else (a deck near flat, a deck past 60 degrees at any time, a broken state) | |

The trial ends once 1.0 has held for 1 s, otherwise at the run limit; the score is the latest step's. The top tier is
judged on the current state, so a handle swung out again afterwards loses it. No tier depends on a sub-millimetre or a
sub-0.2-degree reading.

## Results

GPU runs (L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each). The offline kinematic check
(`tools/offline/check_task.py --require-json --strict --one-arm`, no physics) predicted the demo at 1.0 held from step 715.
The right arm does the task and the left arm only holds its ready pose, so this task does not need both arms.

| Run | GPU score |
|---|---|
| Scripted demo | 1.0 |
| `DEMO_IDLE_ARM=left` (right arm alone) | 1.0 |
| `DEMO_IDLE_ARM=right` (left arm alone) | 0.0 |
| Dream-1, one trial | 0.0 |

Counterfactuals (offline proxy, `DEMO_VARIANT`): `short` (4 turns, 20 degrees) 0.25; `over` (11 turns, 55 degrees)
0.25; `toomuch` (13 turns, 65 degrees) 0; `unfolded` (the handle left out) 0.5; `held` (the hand stays closed on the
knob) 0.5; nothing moving 0.

## Notes for a real trial

Start every trial with the deck flat (inclinometer 0.0 degrees) and the handle folded in its detent. The head section
of a real bed is heavy: first measure the crank's turning torque with the section loaded and set the load in the
simulator to it, or note the difference. The 5.0 degrees a turn is an estimate: if the real bed differs, count the
turns that give 30 to 45 degrees and keep the angle thresholds.

Sources:

- O*NET OnLine, 31-1131.00 Nursing Assistants (https://www.onetonline.org/link/summary/31-1131.00): employment
  1,505,900 (2025); task line "Turn or reposition bedridden patients."
- BLS Occupational Outlook Handbook, nursing assistants and orderlies
  (https://www.bls.gov/ooh/healthcare/nursing-assistants.htm): "Nursing assistants held about 1.5 million jobs in 2025."
- Critical Care 2011, "An evidence-based recommendation on bed head elevation for mechanically ventilated patients"
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC3219392/): the ATS/IDSA, CDC and Canadian Critical Care Trials Group
  guidelines "recommend the use of a semi-upright position (30 to 45 degrees)".
- Drive Medical 15003P Multi-Height Manual Bed listing (https://www.vitalitymedical.com/drive-multi-height-manual-bed.html):
  "By turning the left crank clockwise and counterclockwise, the head portion of the bed can be raised and lowered";
  "When not in use, the cranks can be folded with ease under the bed".
- Sizes of the gearbox, shaft, handle, knob and deck, the ratio and both torques are our estimates, not measured on
  that bed.
