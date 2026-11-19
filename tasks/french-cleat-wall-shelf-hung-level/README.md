# Hang a small tool shelf on the French cleat along a workbench's backboard

A small tool shelf (birch ply, 320 x 100 x 80 mm, open at the top and front, 0.53 kg) stands on the bench with a
12 mm cleat screwed to its back, the lower edge of the cleat cut at 45 degrees. On the workbench's back upstand (backboard,
210 mm tall) 400 mm ahead of the arm bases, a matching 11 mm cleat rail (three screws) has its top edge cut the same way. The robot must hang the shelf: both
bevels face to face, the cleat's back face on the backboard, the shelf level 40 mm over the bench, all 300 mm of its cleat
on the cleat rail, with both hands clear. The instruction names neither the lift nor the order. This is the move
anyone makes when hanging a cabinet, a mirror or a garage tool-wall holder on a French cleat.

## Why it is hard

1. **It is a two-handed carry and a landing.** The shelf is 320 mm wide: each hand pinches one 18 mm end panel (the
   pads close along y), each carries about 2.6 N of the 5.2 N weight. One hand alone lifts one end of it and the other
   end stays on the table.
2. **The shelf has to go over the cleat.** The unit cleat's lowest point is 10 mm over the shelf's underside and the cleat
   rail's front edge stands 61 mm over the bench: the underside must be about 51 mm up (about 11 mm above its seat; the
   eased corners take a little off) before the shelf goes back, or the unit cleat meets the rail's front edge.
3. **It must be carried back until the cleat meets the backboard, then lowered.** The bevels lie on a 45 degree line: a shelf
   carried 7 mm or more short of the board lands on the cleat's edge or in front of it and does not hang (14 mm short:
   it comes down in front of the cleat to the bench; both from the offline model; the GPU did the demo, one-arm and Dream-1 runs only). A shelf lowered
   with the bevels meeting is drawn in by them: in the GPU demo the shelf stopped about 1.5 mm short of the board and
   the bevels drew it the rest of the way as it was lowered.
4. **Both ends must land together.** Lowered skewed (one hand 12 mm short) it hangs from one end only; carried along
   the board it overhangs the rail's end. The bench's end post 110 mm beyond the shelf's end stops the shelf from going
   further left; the backboard stops it going back.
5. **Hands off.** The score is read with both grippers 30 mm clear and the shelf still for 1 s: a shelf held in place
   earns nothing.

## Scoring

Everything is read from the shelf's pose (the backboard is fixed) and both grippers. The unit cleat's two ends are projected on
the backboard; the part over the cleat rail (y -170 to +170 mm) is the engaged length, between points A and B.

| Condition | Rule |
|---|---|
| Bevel contact at A (or B) | the unit cleat's bevel face lies on the rail's bevel plane within 1 mm at two probes 10 mm apart along the slope; the two bevel faces overlap by 4 mm or more along the slope |
| Engaged length | 0 to 300 mm; all 300 mm means the unit cleat's ends are no further out than the rail's (20 mm either way of centred) |
| Seated | at A and B the cleat's back face from 1.5 mm off the backboard to 1 mm into it, and the underside 38 to 42 mm over the bench |
| Level / plumb | long axis within 1 degree of horizontal / back face within 1 degree of vertical |
| Hands off | last 1 s: both finger zones 30 mm or more from the shelf; it moves under 1 mm and turns under 0.3 degrees |

| Outcome | Score |
|---|---|
| Contact at A and B, 300 mm engaged, seated, level and plumb | 1.0 |
| Contact at A and B, 150 mm or more engaged, back face no more than 8 mm off the backboard (or 3 mm into it), within 3 degrees | 0.5 |
| Contact at one of A, B only (one end), or at both with the shelf outside the 0.5 limits | 0.25 |
| Anything else (on the bench, in front of the cleat, on the cleat's edge, leaning, upside down), or not hands off | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Bevel contact (1 mm, 4 mm overlap) | a 1 mm feeler leaf will not enter between the bevel faces within 40 mm of each end of the engaged length; look along the bevel for the faces overlapping by more than a 4 mm strip |
| Engaged length (150 mm / 300 mm) | steel rule along the cleats from the rail's end to the unit cleat's end, read at both ends |
| Seated (1.5 mm, 1 mm) | a 1.5 mm feeler between the unit cleat's back face and the backboard at each end will not enter; a rule from the bench to the underside at each end reads 38 to 42 mm |
| Level (1 / 3 degrees) | torpedo level on the shelf's bottom; 1 degree is about 5 mm over the shelf's 300 mm, 3 degrees about 16 mm |
| Plumb (1 / 3 degrees) | try square or a digital angle gauge on the shelf's back face |
| Back face off the backboard (8 mm) | rule between the unit cleat's back face and the backboard at each end |
| Hands off (30 mm, 1 s) | 30 mm spacer block against each hand; 30 fps video of the last 1 s |
| Still (1 mm, 0.3 degrees, 1 s) | a dial indicator (0.01 mm) on a magnetic base touching the top of each end panel, read over the last 1 s: each changes under 0.8 mm (0.3 degrees of turn is 0.8 mm at the panels, 151 mm from the centre, so this also keeps movement under 1 mm) |
| Success hold (1 s) | the same video |

Success: 1.0 held for 1 s. Doing nothing scores 0. Sliding the shelf along the board is not penalised up to 20 mm; more is
0.5 (150 mm or more still engaged) or 0.25.

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, one trial each; rows marked not run were not run
on the GPU. The offline prediction column (`tools/offline/check_task.py`, with this task's offline stand-in for the
physics, `task.OfflineModel`) is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 394) | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | one end lifted, the far end on the bench | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the same on the other end | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | both hands stop 14 mm short of the backboard | 0.0 | not run |
| `DEMO_VARIANT=skew` | the right hand stops 12 mm short | 0.25 | not run |
| `DEMO_VARIANT=shifted` | carried 60 mm along the board | 0.5 | not run |

With one arm idle the demo scores 0.0 either way, so both hands are needed: one hand cannot lift the shelf's far end.

Read from the GPU demo's trace (simulation, not a real build): the shelf rose to 70 mm (underside), was carried back
until its cleat was 1.5 to 1.9 mm short of the backboard, and was drawn onto the board by the bevels as it was lowered;
hung, the cleat's back face was 0.05 mm off the board, the underside 39.9 mm over the bench, level 0.0
degrees, plumb 0.14 degrees, all 300 mm engaged, both bevel probes within 0.02 mm. 1.0 was first recorded at step 377 (the
trace is sampled every 4 steps) and the run ended on success at step 395. During the carry the shelf tilted up to about
3 degrees about the fingers' axis and settled flat once hung.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both hands mirror each other
(left on the end panel at y +151 mm, right at y -151 mm, fingers closing along y, pad tips 20 mm below the panel's top
edge, pinch point 10 mm in front of the shelf's middle: 32 mm in front of its centre of mass).

| Time (s) | Both hands |
|---|---|
| 0 to 2 | over the end panel, fingers open (94 mm; the panel is 18 mm) |
| 2 to 4.5 | down to just above, then slowly down round the panel |
| 4.5 to 6.5 | settle, then pinch (10 mm commanded, about 20 N a finger) |
| 6.5 to 8.5 | lift 75 mm commanded (the arms trail about 5 mm): the underside about 70 mm up (70.3 mm on the GPU), the unit cleat's lowest edge about 19 mm over the rail's 61 mm front edge |
| 8.5 to 11 | carry the shelf 122 mm toward the backboard: 118 mm to where its cleat meets the board and 4 mm more (pressed back; on the GPU the arms' trail left it 1.5 to 1.9 mm short) |
| 11 to 13.5 | lower it 4 mm past its seat: the bevels meet and draw it in, the weight comes onto the cleat |
| 13.5 to 15.5 | settle, open the fingers (1.5 s) |
| 15.5 to 18.2 | straight up, then 162 mm toward the robot, clear of the shelf |

Four deliberate moves: lift, carry back, lower, let go. Offline: worst IK 0.10 mm; left joint 4 margin 13.6 degrees; TCPs
30 cm apart (302 mm between the pinch points) and arm capsules at least 10.8 cm apart; the gripper housing stays 28 mm
short of the backboard and the wrists (link 4) pass about 25 mm over its 210 mm top edge.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 900
```

Runtime: the demo is 364 steps (18.2 s of simulated time); success holds from step 394 in the offline model and was
first recorded at step 377 on the GPU (the run stopped on success at step 395); a 900-step trial takes about 1.5 to 3
minutes of wall-clock on the L40S including start-up.
Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run starts from the same pose and no seed is
needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the engaged length, both
ends' bevel face gap, back face distance and underside height, the shelf's level and plumb angles, the hands' distance, both
TCPs and openings, and a `hist` of the last 3 s of the key numbers in every row.

## Simplifications

- Rigid bodies. Plywood is solid boxes (680 kg/m3, est). The two 45 degree faces are 2.5 mm slabs with 1 mm stairs behind
  them and arrises eased 2 mm (the rail's top front corner, the unit cleat's lower back corner), so the faces meet
  over a 7 mm run. Screws: heads only (three on the rail, three inside the shelf's back panel); each cleat is one body with its part.
- The cleat rail is on a bench's back upstand, 210 mm tall, as on many repair benches. The arms' wrists pass about 25 mm
  over its top edge at the seat (the YAM's wrist reaches about 70 mm ahead of a straight-down TCP), so the same shelf on a
  full-height wall is out of this robot's reach. The end post is the bench frame's 30 x 30 mm end upright.
- The shelf is a small tool ledge, 0.53 kg. A loaded cabinet is heavier and is lifted by hand and an assistant.
- Friction 0.45 / 0.40 birch ply on birch ply (est); the pads' own friction is the simulator's.
- Nothing else is simulated: no load on the shelf after it hangs.

## Sources

- 45 degree French cleat practice (wall strip screwed to studs or a backboard, unit strip on the back, lift clear, set
  back, lower): common carpentry and home-workshop practice for wall cabinets, mirrors and garage tool walls. No
  figure is quoted from a source; sizes here are the task's own design values.
- Birch plywood about 680 kg/m3 and 18 mm / 12 mm / 6 mm sheet thicknesses: sheet sizes as sold; density an estimate.

Estimates (stated in fidelity): plywood density, friction, the pad-tip pinch depth, the arms' trail under load (2 mm per N,
from the shim task's GPU run).

## Real build and reset

No part is a commercial product: everything is cut from stock sheet and strip with a saw that bevels at 45 degrees.
All sizes are design values (what to cut), not measurements of a built unit. Cut list (thickness x width x length):

| Part | Material | Size (mm) | Notes |
|---|---|---|---|
| Backboard | plywood, painted | 30 x 210 x 700 | two 15 mm sheets glued (est); front face 400 mm ahead of the arm bases |
| Braces (2) | pine | 30 tall x 40 wide x 130 long | behind the board at y -250 and +250 mm, screwed to bench and board |
| End post | pine | 30 x 30 x 210 | against the board, inner face 270 mm left of the midline |
| Cleat rail | birch ply, 12 mm planed to 11 | 11 x 41 x 340 | top edge bevelled 45 degrees: 41 mm tall at the front face, 30 mm at the back; bottom edge 20 mm over the bench; three 8 mm pan-head screws at y -120, 0, +120 mm, 30 mm up |
| End panels (2) | birch ply 18 mm | 18 x 80 x 100 | |
| Bottom | birch ply 6 mm | 6 x 100 x 284 | between the end panels |
| Back | birch ply 6 mm | 6 x 74 x 284 | between the end panels, on the bottom |
| Unit cleat | birch ply 12 mm | 12 x 60 x 300 | lower edge bevelled 45 degrees: 60 mm tall at its back face, 48 mm at its front; top 70 mm over the shelf's underside; three pan-head screws through the back panel at y -100, 0, +100 mm, 50 mm up |

Acceptance check on a built rig, before trials: hang the shelf by hand and confirm with a rule that its underside is 38 to
42 mm over the bench at both ends and that a 1.5 mm feeler will not enter behind its cleat. The hung position is set by
the two 45 degree faces and the backboard, so a bevel cut or a rail fixed 1 mm off moves the hung shelf about 1 mm; if
the check fails, plane the unit cleat's bevel or reset the rail's height. Reset:
stand the shelf on the bench on its underside, centred on y 0, its box at x 170 to 270 mm, cleat toward the board, end panels
square to the board; wipe both bevels.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; backboard, cleat rail and shelf, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
