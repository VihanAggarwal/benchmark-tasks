# Close a stand-mounted micrometer on a 20 mm gauge block with its ratchet and lock it

An outside micrometer (0 to 25 mm class) stands upright in a bench stand, anvil down, its C-frame open toward the robot.
The stand has a lapped work plate flush with the anvil's face and two guide rails; the frame's column clamps in a rear
post. The spindle is open to 20.30 mm. A grade 0 steel gauge block, 20.000 mm thick (35 x 9 mm faces), stands in the
rails' channel 55 mm in front of the faces. The robot must slide the block in between the faces, run the spindle down
by the thimble and finish on the ratchet knob until it clicks at least three times, flip the lock lever, and take its
hands away. The instruction names neither the order nor how far to turn.

## Why it is hard

1. **The block has to be put in.** With the spindle 0.30 mm over the block's top there is no room to carry it between
   the faces. It is pushed along the plate in the rails' channel (0.75 mm of play each side) by a closed pad block until
   the frame's column stops it, as a part is slid onto a micrometer stand's rest. The gripper's housing sits 70 mm over
   its tool point and the knob's top is 67 mm over the plate, so the pusher has to work low, its pad tips 3 mm over the
   rails.
2. **The turns need two or three grips.** 216 degrees of thimble turn bring the faces onto the block (0.5 mm per turn),
   then the knob has to slip at least 45 degrees (three clicks of 15 degrees) on the stopped thimble. The wrist turns
   about 165 degrees at a time, so the hand grips, turns, lets go, turns back open, and grips again.
3. **The thimble must not do the squeezing.** The knob turns the thimble only until the faces touch the block, then it
   slips. Gripped low, on the thimble, the same hand would turn the screw on past touch: more than 10 degrees of
   thimble turn past touch (about 14 N at 1 N per micron, est) scores 0.
4. **The lever is small.** A 30 x 8 x 5 mm lever on the lock collar, just under the thimble, flips 90 degrees under
   about 0.5 N at its tip: a closed pad block pushes its side, from outside the thimble's reach.
5. **Hands off.** The score is read with both grippers 30 mm clear of the block and every control, all still for 1 s.

## Scoring

| Condition | Rule |
|---|---|
| Gap | the distance between the two faces at the last step (20.30 mm less the spindle's travel) |
| Between faces | block's centre within 14.3 mm of the axis along x, 1.3 mm along y; on the plate within 1.5 mm; faces within 5 degrees of square |
| Touch | between faces, gap 20.02 mm or less |
| Clicks | 15 degrees of ratchet slip on the thimble per click (the most it reached), counted only while the faces are seated (touch, thimble within 2 degrees of its 216 degree touch turn) and the lock lever is off (under 34 degrees) |
| Locked | lever turned 80 degrees or more |
| Squeezed (latched) | block between the faces and thimble more than 10 degrees past the 216 degrees of touch |
| Fell (latched) | block's centre more than 5 mm below the plate's top |
| Hands off | last 1 s: fingers 30 mm or more from block and controls; block under 0.5 mm and 1 degree; thimble, knob, lever under 1 degree |

| Outcome | Score |
|---|---|
| Between faces, gap within 0.03 mm of 20.000, 3 or more clicks, locked | 1.0 |
| The same, not locked | 0.75 |
| Between faces, gap within 0.03 mm, 1 or 2 clicks | 0.5 |
| Between faces, gap within 0.20 mm of 20.000, otherwise (no click yet, gap 0.03 to 0.20 mm off) | 0.25 |
| Squeezed, fell, not hands off, no block between the faces, or the gap more than 0.20 mm off | 0 |

Bench check for each condition on a real build (the sim reads joints and poses; a person reads these):

| Condition | Bench check |
|---|---|
| Gap (0.03 / 0.20 mm) | the micrometer's scale to 0.01 mm, block in |
| Between faces | a look from above with a 5 degree gauge, and a 1.3 mm feeler against the block's side |
| Touch (20.02 mm) | the first click, with the scale within 0.02 mm of 20.00 |
| Clicks | count by ear (or a phone microphone), the knob marked in 15 degree steps; only with the lever off and the scale at 20.00 |
| Locked | the lever against an 80 degree mark on the collar |
| Squeezed (10 degrees, 14 N) | a line scribed across thimble and barrel at the first click, a 10 degree mark on the rim; or a 50 N load button under the block with peak hold, under 14 N |
| Fell | a rule 5 mm below the plate's top, from the video |
| Hands off | a 30 mm spacer against each hand; 30 fps video of the last 1 s; a dial indicator on the thimble's rim (1 degree is 0.17 mm) |

Success: 1.0 held for 1 s. Doing nothing scores 0.

Unintended strategies earn nothing. Closing the empty micrometer to 20.000 mm scores 0 (every tier needs the block
between the faces). Clicks are counted only while the faces are seated on the block with the lock off, from where
the knob stood when that began, so turning the knob ahead of time (before the block is in), or locking the spindle
short of the block (say at 20.02 mm) and turning the ratchet against the lock, earns no click. The thimble
driven on past touch is a squeeze (0), latched. A held block or a hand on a control is not hands off. The block off the
plate is latched (0).

## Results

GPU runs: L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each. The offline prediction column
(`tools/offline/check_task.py`, with this task's offline stand-in for the physics, `task.OfflineModel`) is not a
measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run |
| Scripted demo | below | 1.0 (held from step 896) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | turns the knob and flips the lever, no block in | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | pushes the block in, nothing turned | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=no_lock` | the lever left alone | 0.75 | not run |
| `DEMO_VARIANT=two_clicks` | the knob stopped at 39 degrees of slip | 0.5 | not run |
| `DEMO_VARIANT=no_clicks` | the knob stopped 1 degree short of touch | 0.25 | not run |
| `DEMO_VARIANT=thimble` | the pads take the thimble, turned 89 degrees past touch | 0.0 | not run |

With one arm idle the demo scores 0.0 either way: one hand puts the block in, the other turns the knob and flips the
lever.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The turns are computed from
the screw (216 degrees to touch) and the click size (15 degrees).

| Time (s) | Right gripper (closed into a block) | Left gripper |
|---|---|---|
| 0 to 6.5 | in the air in front of the block, close, down to 5.5 mm over the plate | wait over its own side |
| 6.5 to 13.3 | push the block 73 mm at 25 mm/s to 2 mm behind the axis, back off 30 mm, up, aside | wait |
| 11.8 to 18.5 | aside | over the knob, down onto it, grip across its 18 mm |
| 18.5 to 23.4 | aside | turn the knob 165 degrees clockwise (seen from above), let go |
| 23.4 to 33.3 | aside | turn the open hand back, grip again, turn 140 degrees more (305 in all: 216 to touch, 89 slip, five clicks), let go, up |
| 33.3 to 43.9 | aside | closed fingers beside the lever, push it 36 mm to the lock, up, aside |

Offline: worst IK 0.09 mm on both arms; left joint 4 margin 11 degrees, right 10 degrees; the arms' capsules stay 4.3 cm
apart (link 4 against link 4 as the left hand comes down to the knob); TCPs at least 22 cm apart. The gripper's housing
(70 mm over the tool point) clears the knob's top by 20 mm when the right hand pushes.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 2000
```

Runtime: a 2000-step trial is expected to take about 3 to 6 minutes of wall-clock on the L40S including start-up (the
demo itself is 878 steps, 43.9 s of simulated time; the scripted policy stops at step 938, the success hold at about
step 896). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run starts from the same poses
and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the gap, whether the block
is between the faces and touched, the thimble's and knob's turns, the clicks, the squeeze margin, the lever, the block's
pose, both TCPs and openings, and what the scorer saw.

## Simplifications

- Rigid bodies. The frame, stand and controls are boxes and cylinders (a real thimble is knurled and a real ratchet
  has a spring and teeth). The thimble does not travel along the barrel (0.3 mm in all).
- Three declared rules (in `task.py`, applied at the start of every control step): the screw (the spindle is driven to
  0.5 mm per thimble turn and stopped where its face meets the block), the ratchet (the knob, a hinge on the thimble,
  turns the thimble until the faces touch the block, then the thimble holds and the knob slips; its torque limit is not
  modelled, the stop is) and the lever's two-notch cam (it snaps to the lock stop past 34 degrees and back to free
  below 14 degrees; locked, it holds the thimble). The scorer reads joints and poses, not the rules' internals.
- The squeeze (1 N per micron of closing, est) is computed from the thimble's turn past touch, not from a contact force.
- The block starts on the plate in the rails' channel and is slid in; it is not carried or hand-held. The guide rails,
  the flush work plate and the frame's column as a stop are the stand's part rest.
- Friction 0.2 steel on steel (est). Masses from density and size.
- The job ends at the lock. Reading the scale and taking the block out of the faces are not part of the task (the
  lock holds the reading for them); the sim has no scale to read.

## Sources

- Inspectors, testers, sorters, samplers and weighers, SOC 51-9061 (about 0.6 million US jobs): BLS Occupational
  Outlook Handbook, https://www.bls.gov/ooh/production/inspectors-testers-sorters-samplers-and-weighers.htm
- Outside micrometer, 0 to 25 mm, 0.01 mm graduation, 0.5 mm thread pitch, ratchet stop (the Mitutoyo 103-137 class):
  maker's catalogue listing.
- Rectangular gauge blocks, grade 0, 35 x 9 mm section above 10 mm length: ISO 3650.

Estimates (marked est in the task text): the frame's, stand's and controls' sizes other than those the maker lists,
the lever's 0.5 N, the screw's stiffness, the frictions.

## Real build and reset

A bench micrometer stand with a work plate level with the anvil's face, two rails 10.5 mm apart (or a 10.5 mm channel
milled in the plate), a rear post and clamp; an outside micrometer 0 to 25 mm with ratchet stop and lock lever; a 20 mm
grade 0 gauge block; a 5 degree gauge, a 1.3 mm feeler, a 30 mm spacer, a dial indicator. Reset: spindle open to
20.30 mm on the scale, lever off, the block standing in the channel 55 mm in front of the faces' axis, its front end
toward the robot. Wipe the faces and the block between trials. Check the micrometer's zero on its closed setting
once a day.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the three rules, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the stand, micrometer and block as compound rigid bodies and joints; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
