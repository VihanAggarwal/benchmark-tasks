# Move the patient from the trolley to the bed and keep a hand on them until the bed rail is up

Falls from beds and trolleys are among the commonest harms in hospitals and care homes, and a moment is enough: a
patient left on a bed with the rail down and nobody's hand on them is the hazard, whatever happens before or after.
Safe patient handling practice (nursing assistants and orderlies, registered nurses, EMTs) is that after a
transfer the carer keeps a hand on the patient until the rail on the open side is up. Here a skin-pink "patient"
block lies on a dark grey trolley at the left, a white bed deck with a pale green mark stands between the arms, and
a light grey rail lies on the table at the right. Only the left arm reaches the trolley and only the right arm
reaches the rail, so one gripper keeps hold of the patient on the bed while the other brings the rail up along the
bed's right edge: two-handed by reach and by rule. The scorer watches both grippers, the patient and the rail at
every control step, not only the end state. Raising the rail first and then setting the patient over it is also
safe and also scores 1.0: the rule is "never on the bed with the rail down and no hand on them", not an order.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the patient's, the rail's and the grippers' states, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook, the demo played through the scorer, and the exploits we tried (no simulator) |
| `tests/test_demo_plan.py` | Offline checks of the scripted demo against the simulator's and station's limits (reach, fingers, timing, spacing) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (three fixed and two loose boxes) and the standard `task_score` scorer.

The scorer's gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the hand-on rule is then taken as kept on those steps, and the
trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the pink patient block from the dark grey trolley onto the green mark on the white bed, and put the light grey rail up along the bed's right edge, lying along the bed. Once the patient is on the bed, keep a gripper on them until the rail is up: never leave the patient on the bed with the rail down and no gripper on them. Lift the patient, do not slide them off the trolley; lower them gently and do not drop or tip them. Let go of both at the end." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/hand-on-patient-until-rail/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the pink patient block from the dark grey trolley onto the green mark on the white bed, and put the light grey rail up along the bed's right edge, lying along the bed. Once the patient is on the bed, keep a gripper on them until the rail is up: never leave the patient on the bed with the rail down and no gripper on them. Lift the patient, do not slide them off the trolley; lower them gently and do not drop or tip them. Let go of both at the end." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/hand-on-patient-until-rail/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `unattended_steps`,
`rail_up`, `rail_off_mm`, `on_bed`, `resting_on_bed`, `flat_on_mark`, `centre_off_mm`, `patient_low_mm`,
`clearance_mm`, `patient_tilt_deg`, `max_descent_mps`, `drag_mm`, `hands_on`, `held`, `held_on_bed_s`, `opening`,
`zone_to_patient_mm`, `zone_to_rail_mm`, `grippers_let_go`, `first_on_bed_step`, `first_rail_up_step`,
`arms_unavailable`, both TCPs, ...). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record
per process and starts it over at every reset, so run one trial per process or rely on the reset; the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (88 tests; `test_demo_plan.py` stands in
for `tools/offline/check_task.py`, which is not part of this folder, and `test_scoring.py`'s `TaskFile` class
plays the scripted plan through the scorer with a kinematic proxy of the arms and parts).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the patient, the rail and both grippers
once per control step (20 Hz). It cannot see which way a gripper is turned, so each gripper's fingers count as a
round zone about its axis, out to the fingers' outer corners (57 mm fully open, 41 mm closed on the 70 mm patient,
23 mm closed on the 30 mm rail), from the finger tips to 38 mm back up the gripper. A gripper has a hand on the
patient when it is less than 0.9 open, that zone overlaps the patient, and its finger tips are no higher than
10 mm above the patient's top: closed round them, or closed fingers pressed on their top. Hovering above or beside
them is not a hand on them. Either gripper counts. Four rules hold for the whole trial; one broken at any moment
scores 0 whatever follows:

1. the patient is on the bed (any part of their outline over the deck, lowest point within 10 mm of the deck's
   top) with the rail not up and no gripper's hand on them: unattended, even for one control step;
2. the patient's descent rate exceeds 0.15 m/s (dropped; the rate is the step-to-step descent, plus the downward
   velocity while more than 3 mm above the surface under them, so a landing transient within 3 mm is not a drop);
3. the patient tilts more than 15 degrees;
4. the patient slides more than 15 mm while low off the bed (within 10 mm of the trolley's top or the table):
   pushed off the trolley instead of lifted. Sliding on the bed to position them is allowed.

"Rail up": the rail's centre within 10 mm of its place at the deck's right edge (x 28.0 cm, y -16.5 cm), along x
within 5 degrees, level within 5 degrees (either face down), its lowest point within 3 mm of the deck's top. The
rail counts as up while still held in place (it is where it should be, and the patient is protected), but the end
is 0.75 until the gripper lets go.

| Outcome | Score |
|---|---|
| Any rule broken at any moment, or the patient or the rail ends off the table, or the patient ends on the table off both platforms | 0 |
| Rail up, patient flat on the bed (tilt under 5 degrees) within 15 mm of the mark's centre, both grippers let go of both items | 1.0 |
| Rail up, patient flat on the bed within 30 mm of the mark, but 15 to 30 mm off or a gripper still holding the patient or the rail | 0.75 |
| Rail up, patient resting on the bed anywhere (more than 30 mm off the mark, or tilted 5 to 15 degrees); or rail not up, patient flat on the bed within 30 mm with a hand on them (safe but unfinished) | 0.5 |
| Exactly one half done: rail up but the patient not resting on the bed (on the trolley, or in the air); or rail not up, patient resting on the bed more than 30 mm off or tilted 5 to 15 degrees, with a hand on them | 0.25 |
| Anything else (nothing done, the patient put back on the trolley, carried and still in the air with the rail down) | 0 |

The score is read at the last step; the trial ends early once the 1.0 conditions have held with the patient and
the rail still (under 1 cm/s and 0.5 rad/s) for 3 s, longer than the demo's release (1.5 s) plus rise (1.0 s), so
it cannot end while a gripper still holds either item (and the 1.0 conditions need both let go, so the hold starts
counting only once the last gripper has opened past 0.9). A new trial is recognised from rigstate's reset hook, or,
as a fallback, when the simulator puts the patient and the rail back on their starting places (within 4.5 cm, which
covers a seeded reset's jitter) in a single step, further than their last velocity and a free fall could carry them.
A simulator state with a non-finite number scores 0 for that step and leaves the record of the trial as it was.
The rail is a plain bar, so either face may be down and either end may be forward.

Unintended strategies earn nothing, as the tests show: setting the patient on the mark and letting go before the
rail is up breaks rule 1 (0; the end state alone would have been 1.0), even if the fingers pass 0.9 open for a
single step, or the gripper opens and then hovers closed 2 cm above them; dropping the patient the last 20 mm onto
the mark breaks rule 2 (0), whether before or after the rail; shoving the patient along the trolley, across the
40 mm gap (the 60 mm block bridges it) and onto the mark breaks rule 4 (0) and, with the rail down, rule 1 as well;
shoving them so that they tip into the gap breaks rule 3; carrying them and setting them down half on the bed,
half on the trolley, and letting go is unattended (0); putting them back on the trolley after a carry is 0; laying
the rail on the deck's left edge, on the table beside the bed, or turned 20 degrees across its place is not "up",
so the trial is 0.5 at most while the hand stays on and 0 once it lifts off; flinging the patient from 10 cm up
so that they land flat on the mark after the rail is up is a drop (0); holding them 11 mm above the deck with the
rail down is not "on the bed", so not unattended, but scores nothing, and letting go from there is a drop and
unattended; the right gripper, which reaches only the patient's far face on the trolley, can only shove them along
it, which is dragging (0); doing nothing is 0. Leaving a gripper closed on the patient on the mark, or on the rail
in its place, is 0.75. A patient set down 20 mm off the mark's centre is 0.75, 14 cm off on the bare deck is 0.5,
resting tilted 8 degrees on a corner of the mark is 0.5. Raising the rail first and then placing the patient and
letting go at once is legitimate (1.0): no step was unattended. Closed fingers resting on the patient's top
(rather than round them), or a gripper opened to 80 mm round the 70 mm block, while the rail goes up is a hand on
them too (1.0); opened to 85 mm (0.9) it is not. The rail has no drag or drop rule (the hazard scored is the
patient's), so a rail shoved against the deck and tipped over onto its place, even upside down, is up.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) picks the rail up, waits, lays it along the bed's edge and lets go,
with the patient still on the trolley: 0.25. `DEMO_IDLE_ARM=right` (the left arm alone) carries the patient to
the mark, keeps hold of them for 7 s (0.5 meanwhile, safe but unfinished), then opens with the rail still on the
table: unattended, 0. Under the rules no single arm can finish: the right cannot reach the patient on the trolley
(its limit is about y +17 cm, the patient at y +24 cm) and the left cannot reach the rail (its limit about y -18 cm,
the rail at y -27 cm), so only the left can bring the patient and only the right the rail, and the patient may not
be left alone on the bed between the two. The left arm alone could score 0.5 by holding on for ever, which is what
the rule asks of a carer with no second person: safe but not finished. (Offline kinematic predictions, confirmed
by playing each one-arm plan through the scorer in `tests/test_scoring.py`; the GPU runs are recorded in 'Results
in simulation' below.)

## Demo and run limit

The scripted demo takes 23.0 s (460 control steps; the right arm's plan is 22.5 s), both arms on one clock. The left arm goes over the patient at
12 cm up, comes down to 3 cm above the grip, then slowly to the grip height (the finger tips 4 mm above the
trolley's top, holding 36 of the patient's 40 mm), settles 0.5 s, closes to 62 mm over 1.5 s across the patient's
70 mm (the fingers along x, yaw -pi/2), settles 0.5 s, lifts straight up to 12 cm (the patient's underside 10 cm
up, 7 cm over the platforms), carries 19 cm to the right over the bed mark in 2.0 s, comes down to 2 cm above the
set-down in 1.0 s, settles, lowers the last 2 cm over 1.5 s (the patient's underside aimed 2 mm above the deck,
1 mm above the mark), settles, and then keeps its fingers closed on the patient for 7.0 s, until the rail's
set-down is complete and settled plus 1.5 s (the right's release), before opening over 1.5 s and rising 1.0 s.
The right arm meanwhile goes over the rail at 10 cm up, comes down in the same two stages (the tips 4 mm above the
table, holding 36 of the rail's 40 mm), settles, closes to 22 mm across the rail's 30 mm (the fingers along y,
yaw 0) at its centre, settles, lifts to 10 cm and waits in the air 6.3 s until the patient's set-down is complete
and settled plus 0.5 s (14.0 s), then carries the rail 10.5 cm to the left in over the bed's right edge, comes
down to 2 cm above, settles, lowers the last 2 cm over 1.5 s (the rail's underside aimed 2 mm above the deck),
settles, opens over 1.5 s (the rail is up from about 19 s), rises straight up 1.0 s (its opening fingers straddle
the 40 mm rail, so it must not move sideways first) and goes back out over the table 1.0 s. The two TCPs
are never closer than 21.5 cm in y (the patient's grip at y +5 cm, the rail's at y -16.5 cm, both at x 28 cm). The
run limit is 1,200 steps (60 s).

Offline, in the kinematic proxy (the TCPs on their targets, each part following its gripper and dropping its last
millimetre when the fingers open), the patient is on the bed from step 250 (12.5 s), the rail up from step 370
(18.5 s), the hand on the patient with the rail down for 6.0 s, both grippers let go at step 432 (21.6 s), and
success has held its 3 s at step 491 (24.55 s), before the scripted policy's own stop at step 520 (23 s plus its
3 s hold). The fastest descent is 0.05 m/s (the last 2 cm over 1.5 s plus the 1 mm drop), a third of the limit.

Payload sag: the arms' shoulder and elbow joints are position-controlled springs without gravity compensation for a
payload. By the reference tasks' estimate (about 8 mm down and 1 to 2 degrees of gripper tilt for 0.3 kg at 0.3 m
reach) the left arm sags about 5 to 6 mm under the 0.20 kg patient and the right about 3 mm under the 0.10 kg rail.
Each is carried 7 cm or more above anything it passes, so the sag cannot catch an edge, and at each set-down the
sag puts the part on its surface a little before the slow descent ends, the arm still carrying part of its weight:
no drop. With 8 mm of sag in the proxy (more than the patient's), the score is still 1.0 with no drop and no drag.
A gripper tilt of 1 to 2 degrees under the load is well inside the 5 degree 'flat' limit and the 15 degree tip
limit. If the GPU shows the patient landing earlier or later, `TASK_TRACE`'s `patient_low_mm`, `clearance_mm` and
`max_descent_mps` say by how much.

Grip security: the patient (0.20 kg) is held 36 mm deep with 62 mm commanded across 70 mm, its weight straight down
through the pads with no moment about the grip; the rail (0.10 kg) is held at its centre 36 mm deep with 22 mm
across 30 mm, no moment either. Both are well inside the pads' friction by the reference tasks' estimate (two pads
at about 20 N, friction 1.2). The right's open fingers at the rail's pick (110 mm along y about y -27 cm) stay 35 mm
clear of the deck's edge; at the set-down the opening fingers (tips 4 mm above the deck) are over the deck on one
side and over the table 34 mm up on the other.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the patient back flat and centred on the
trolley and the rail back flat on the table at x 28.0 cm, y -27.0 cm, long side along x, both with sides parallel
to x and y; the bed, the mark and the trolley are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 23 s of simulated time plus the 3 s
hold); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 30 mm thick blocks of plywood or MDF clamped to the bench 40 mm apart, one 200 x 300 mm painted white (the
bed) with an 80 x 70 mm pale green card taped on its top centred 80 mm from its left edge, the other 200 x 160 mm
painted dark grey (the trolley); a 70 x 60 x 40 mm block weighted to 200 g and painted skin pink (a wood block, or
a filled plastic case) as the patient; and a 200 x 30 x 40 mm length of rectangular aluminium tube (about 100 g),
light grey, as the rail. Mark a 15 mm and a 30 mm circle round the green card's centre and a 10 mm box round the
rail's place along the bed's right edge for a top photo of the end state; stand a card marked at 3 and 10 mm above
the deck behind the bed, out of the arms' way, for side video; film from above at 50 fps for the hand-on rule
(frame by frame: a hand closed round the block or pressed on its top counts, hovering does not) and the drop (more
than 7.5 mm between two frames).

## Results in simulation

To be measured on the GPU (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each, with the
commands above). The offline plan predicts the first row; the one-arm rows are predicted above.

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline plan: 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (predicted 0.25) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (predicted 0; 0.5 until it lets go) |
| Dream-1 | to be measured on the GPU |
