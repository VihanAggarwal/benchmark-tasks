# Lay a dropped kerb at a driveway, each mirror-image transition kerb at its own end

Kerb layers and groundworkers lower the kerb across a new driveway entrance (a vehicle crossover, or dropped kerb) so
cars can cross the footway (O*NET 47-2061.00 Construction Laborers, 47-4091.00 Segmental Pavers). Low kerb units go
across the entrance, and a transition kerb at each end ramps the kerb top from full height down to the dropped height.
Precast kerb ranges make the transitions in left-hand and right-hand versions, mirror images of each other, because
the face toward the road is battered (sloped back): each fits one end of a crossing only. Turned round to rise the
other way, a transition shows its battered face to the footway; laid at the wrong end with its face to the road, it
rises the wrong way and leaves a step in the kerb top, a trip edge for people on foot and a jolt for wheels.

Here the crossing is a 1:5 model on the table. Two fixed full kerbs (30 mm tall, battered face darker, toward a dark
grey road strip) leave a 270 mm gap in the kerb line, with a red-brown paving strip on the footway behind its middle.
Four loose units lie in a stock area near the robot: two low kerbs (12 mm tall) and two transition kerbs, which step
12, 21 and 30 mm along their 60 mm length and are mirror images. The robot has to lay the low kerbs in the gap's
middle two slots and a transition in each end slot rising to the full kerb beside it, every unit with its back in
line with the full kerbs' backs, its battered face to the road and 6 mm joints. Only one of the transitions can rise
toward each end with its face to the road, and each starts on the side of the end it does not fit, so taking the
nearer one is wrong. The score is 0.25 for each slot holding the right unit laid the right way, read from how the
units lie at the end, as a photo, a straightedge and a ruler would read them. The natural split is one half of the
crossing each, each arm reaching across the midline once for its transition.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four loose units' states, with a small tracker for stillness and the trace |
| `mechkit.py` | Compound bodies for `yam_arena` task files (each loose unit is one rigid body of two or six boxes), hooked from the task file's import; unchanged from our kit |
| `tests/test_scoring.py` | Offline tests of every scoring tier and threshold, every zero condition, the exploits we tried, resets, broken states, the demo plan, its timing and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the fixed parts are
`yam_arena`'s own task-file boxes, the loose units are built by `mechkit.py` (next to `task.py`, imported by it) and
the score comes from the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the dropped kerb for this driveway entrance in the gap in the kerb line, with 6 mm joints: the two low kerbs in the middle, in front of the red paving, and a transition kerb at each end, rising to the full kerb beside it so the kerb top has no step at any joint (the two transitions are mirror images). Every kerb goes battered (darker) face to the road, its back in line with the full kerbs' backs." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/dropped-kerb-transitions/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the dropped kerb for this driveway entrance in the gap in the kerb line, with 6 mm joints: the two low kerbs in the middle, in front of the red paving, and a transition kerb at each end, rising to the full kerb beside it so the kerb top has no step at any joint (the two transitions are mirror images). Every kerb goes battered (darker) face to the road, its back in line with the full kerbs' backs." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/dropped-kerb-transitions/dream
```

Run one trial per process (the default). The score depends only on where the units lie; the small tracker in the
task module keeps each unit's path over the last 0.5 s (for stillness), and starts afresh when mechkit counts a reset
or when the simulator puts every unit back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided: `slots_right`, `zero` (why the score is 0, if it is), for each slot
whether it is `right`, by which `units` and what it `wants`, and for each unit its `kind`, `right_in`, `bedded`,
`at_rest`, `flat`, `tilt_deg`, `on_bed`, `low_mm`, `back_off_mm` (each back corner's offset from the line),
`in_line`, `face_err_deg`, `face_to_road`, `rises_toward` (a transition's tall end: `+y`, `-y`), `heading_deg`,
`nearest_slot`, `along_mm` and `origin_mm`, and the raw poses. mechkit prints, after the first reset, how Isaac Lab
built each unit (one loose rigid body: low kerbs 21 g, transitions 37 g). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the four loose units' poses, once per
control step (20 Hz), and scores the state at hand: the score is read at the last step. Each unit's frame (the centre
of its underside and its orientation) gives every point of it, from its known shape; the road, full kerbs and paving
are fixed, so their places come from `layout.py`. The two low kerbs are alike; each transition is known by its shape
(which end of it is tall), not by where it lies.

The gap holds four slots for 60 mm units with 6 mm joints, centred at y +9.9 cm (S1, beside the left full kerb),
+3.3 cm (S2), -3.3 cm (S3) and -9.9 cm (S4, beside the right full kerb). A unit is still when neither its middle nor
one of its ends moved more than 2 mm over the last 0.5 s (positions, not the simulator's reported velocity, which
chatters while a part is gripped). It is bedded in a slot when it is still and (a) flat: tilted at most 5 degrees,
its lowest point within 3 mm of the table top; (b) in line: both of its back corners within 5 mm of the full kerbs'
back line (x 37.25 cm), along x; (c) face to the road: its battered face strip pointing within 30 degrees of straight
at the road; (d) its middle within 8 mm of the slot's centre, along y (6 mm joints put every middle on its slot
centre; the joints are judged through (d)). A slot is right when S2 or S3 has a low kerb
bedded in it, S1 a transition bedded in it with its 30 mm end toward +y (rising to the left full kerb), and S4 a
transition bedded in it with its 30 mm end toward -y.

| Outcome | Score |
|---|---|
| No slot holds the right unit laid the right way | 0 |
| For each slot that does | 0.25 |

So the score is 0.25 x the number of right slots: 0, 0.25, 0.5, 0.75 or 1.0, and every tier is reached by an ordinary
partial job. A slot counts once, and no unit can be within 8 mm of two slot centres (they are 66 mm apart). Every
outcome, with numbers: all four right 1.0, and the kerb top then runs 30 | 30-21-12 | 12 | 12 | 12-21-30 | 30 mm with
no step at any joint; three right and the fourth 12 mm along from its slot, 8 mm back from the line or left in the
stock area 0.75; the low kerbs right and the transitions swapped, either laid face to the road (each then rises
toward the middle, an 18 mm step at both its ends) or turned to rise toward the full kerbs (faces to the footway),
0.5; the transitions right and the low kerbs left in the stock area 0.5; a low kerb in S1 and a transition in S2
with the other two right 0.5 (nothing for S1 or S2); one transition right only 0.25; the start 0. A unit on its
side, on end, upside down, on another unit or a kerb, held more than 3 mm up, moving, or laid face to the footway
earns its slot nothing. The trial ends early (success) once 1.0 has held for 2.5 s; otherwise it is scored as it
stands at the run limit. A state with a non-finite number, a zero quaternion, no frame origin or a missing unit (a
simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: taking the nearer transition to each end and turning
it to rise toward the full kerb leaves both faces to the footway, 0 for both ends; the units butted together from
the left kerb without joints drift 6 mm a slot, so only S1 counts; units stood on end in the slots, a unit turned
upside down (which would make one transition look like the other) and two low kerbs stacked in one slot (counted
once) earn nothing extra; a unit thrown off the table changes nothing. A unit is judged at both back corners, so
turning it cannot hide an offset. The scorer cannot see the grippers, so a unit held still within 3 mm of the bed
counts as lying there (see `fidelity`); 3.5 mm up it does not.

## One arm alone

The demo splits the work by half: the left arm lays S1 (the transition that starts right of the midline) and S2 (the
low kerb on its side), the right arm S3 (its low kerb) and S4 (the transition that starts left of the midline). With
one arm idle the other still lays its own half: on the GPU the scripted demo scored 0.5 with the left arm idle and 0.5
with the right arm idle, as the offline check predicted. The task is not scored on using both arms.

The other half is out of a straight-down grip's reach. At the kerb line, with the tool point 13 to 30 mm above the
table (pad tips from touching the table to 17 mm up) and any grip heading (5 degree steps), our IK model puts the left
arm's reach at y -4.8 cm at most at x 38.0 cm (-4.2 cm at 38.5, -3.6 cm at 39.0), while a transition bedded in S4 in
any pose the rule credits (middle up to 8 mm toward the middle of the gap, back up to 5 mm off the line, so its
middle at x 38.0 to 39.0 cm) has its nearer end at y -6.1 cm or beyond: a grip with 5 mm of the 20 mm pads on it
needs the tool point at y -5.6 cm, 8 mm or more out of reach. In the stock area the left arm reaches y -18.9 cm at
x 15.5 cm and -17.1 cm at 20.5 cm, while any grip on the right arm's low kerb (lying at y -24.0 cm) needs y -22.3 cm
or beyond, 33 mm or more out of reach. Both transitions are within either arm's reach in
the stock area, but each fits only the other arm's end. The right arm is the mirror image. So one arm with
straight-down grips lays at most one transition and one low kerb: 0.5. We do not claim one arm cannot do more: a
tilted grip, or pushing a unit along the table toward its reach, might lay a third, a move the scripted demo does not
try.

## Demo and run limit

The scripted demo takes 52.7 s (1,054 control steps). The arms take turns: only one is at the kerb line at a time
(left 8.0 to 17.3 s for S1, right 17.3 to 26.6 s for S3, left 26.6 to 35.9 s for S2, right from 43.4 s for S4), and
each reaches across the midline for its transition only while the other waits parked beside its own base (x 18 cm,
y +-33 cm, 10 cm up), the left arm first (0 to 8 s) and the right arm last (35.9 to 43.4 s). Offline the tool
points never come closer than 28.2 cm and the arms' collision capsules stay at least 12.2 cm apart. For each unit
the arm:

1. comes down round it at its middle, fingers open to 49 mm across its 25 mm depth (12 mm to spare each side), the
   pad tips 2 mm above the table, and grips it (commanded to 17 mm) on its back face and its face strip from 2 mm
   up (10 and 6 mm of them on a low kerb, 19 and 15 mm on a transition's middle step);
2. lifts it so its underside is 55 mm up (25 mm over the tallest kerb) and carries it to its slot, turning it a
   quarter turn on the way, the turn that brings its face strip round to the road: every unit starts with the
   fingers at grip heading 0 and the left arm turns its units -90 degrees and the right arm +90 (joint 6 keeps at
   least 22 degrees from its limits; turned the other way it would end within 2 degrees of a limit or past it);
3. sets it down with its middle on the kerb line's middle (x 38.5 cm) at the slot's centre, lowering the last 30 mm
   over 1.5 s and letting go 2 mm above the table after a 0.3 s settle; the open pads pass 3 mm over the paving and
   the road strip;
4. opens over 1.5 s and goes straight up, then on to its next unit or its parking place.

The units weigh 21 and 37 g, so the arms barely sag under them. A transition's centre of mass is 6 mm from where it
is gripped, toward its tall end. `PLACE_TRIM` in `task.py` moves a unit's place point if the GPU run shows it landing
off by a steady amount (compare `back_off_mm` and `along_mm` with 0 in the trace); all trims are zero. Offline the
smallest joint margin is 14.9 degrees (joint 4, over S2 and S3 at travel height), the worst IK error 0.09 mm and the
pad tips stay at least 2 mm above the table. A unit counts as bedded while still gripped 2 mm above the table, so
success must hold for 2.5 s, longer than the settle and release after the last set-down (0.3 + 1.5 s) plus the 0.5 s
stillness window: the trial cannot end with a unit still in a closed gripper. Offline, success holds at step 1,044;
the plan runs to step 1,054 and the scripted policy stops at step 1,115. The run limit is 2,000 steps (100 s), which
leaves Dream-1 nearly twice the demo's time.

Things to look at in the GPU runs: in the settle run nothing moves (`low_mm` 0, `origin_mm` on the start marks).
After each set-down the unit's `back_off_mm` within 2 mm of 0, `along_mm` within 2 mm of 0, `face_err_deg` and
`tilt_deg` under 2, `rises_toward` `+y` for `transition_left` and `-y` for `transition_right`; watch that the units
set between two neighbours (S2 between S1 and S3, S4 between S3 and the right full kerb) touch neither (6 mm
joints).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the four loose units back flat with their
long sides along x: the low kerbs at (x 18.0, y +24.0) cm with the face strip toward +y and (18.0, -24.0) cm with it
toward -y; transition A (`transition_left`, which fits the left end) at (16.0, -5.0) cm and transition B
(`transition_right`) at (16.0, +5.0) cm, each with its tall end toward the robot and its face strip toward the other.
The low kerbs are interchangeable; the transitions are not. The road, full kerbs and paving are fixed. Nothing lies
where the arms' start pose hides the table from the top camera.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo ends after about 52 s of simulated time, when
success has held for 2.5 s; a run to the limit is 100 s); the first start of Isaac Sim on a new instance takes
several minutes more.

## Real-world build

A 1:5 kerb line on the table. Road: a dark grey vinyl sheet about 1 mm thick, 80 x 560 mm, its edge at x 40.25 cm,
taped down. Full kerbs: two 120 mm lengths of 25 x 30 mm section printed in concrete grey PLA, the road-side 6 mm
strip 4 mm lower and darker grey, fixed to the table (double-sided tape) with their backs on the line x 37.25 cm,
from y +-13.5 to +-25.5 cm. Crossover paving: a red-brown vinyl sheet about 1 mm thick, 50 x 132 mm, x 31.5 to
36.5 cm, taped down. Loose units printed in grey PLA with the road-side 6 mm strip darker and 4 mm lower: two low
kerbs 60 x 25 x 12 mm (about 21 g) and two transitions 60 x 25 mm in three 20 mm steps 12, 21 and 30 mm tall (about
37 g), printed as mirror images (looking along it from its low end to its tall end, A has its face strip on the
right, B on the left). Positions in `task.json`'s `setup`.

Trials are safe: the parts are light plastic models, nothing is hot, powered, sharp or under load, and nobody needs to
reach into the robot's space until the trial has ended.

Scoring a real trial, from the kerb line as it stands: a photo from above shows which unit is in which slot, which
way each transition's tall end points and that each unit's dark face strip is on the road side; a straightedge along
the full kerbs' backs and a ruler give each unit's back-corner offsets (5 mm or less), and a tape along the line from
the left full kerb's end gives each unit's middle (slot centres 36, 102, 168 and 234 mm from it, within 8 mm). A unit
tilted, lifted or resting on another one is plain to see.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
