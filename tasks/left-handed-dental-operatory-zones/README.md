# Left-handed dental operatory: both stools and the cart in the mirrored clock zones

Dental assistants (O*NET 31-9091.00) set up the operatory before each patient. Four-handed dentistry places the
team by the clock concept (zones of activity): seen from above with the patient's face as the clock and the top of
the head at 12, a right-handed dentist works in the operator zone at 7 to 12, the assistant sits in the assistant
zone at 2 to 4, instruments pass over the patient's chest in the transfer zone at 4 to 7, and the static zone at 12
to 2, behind the head, holds the set-up. For a left-handed dentist every zone mirrors about the 12 to 6 axis:
operator 12 to 5, transfer 5 to 8, assistant 8 to 10, static 10 to 12. The assistant sits higher than the dentist to
see into the mouth, so the assistant's stool is the taller one, and nothing stands in the transfer zone, where
instruments are passed. Sources differ on whether the assistant's mobile cart stands in the assistant's zone or in
the static zone, so the scorer accepts either: the cart counts anywhere on the left-handed assistant's side, 8 to
12.

Here the surgery is a table-top model. A fixed light blue patient chair with a headrest lies along y, its headrest
at the -y end with a red mouth mark painted on it, so a patient lying face up would have the top of the head toward
-y and the feet toward +y. The dentist's stool (blue, 40 mm tall) and the assistant's mobile cart (grey box) stand
on the right, the assistant's stool (green, 60 mm tall) beyond the foot of the chair on the left. The robot is told
that the dentist is left-handed and works four-handed, and has to put each piece where it goes round the chair:
nothing in the instruction names a clock position, a side or a zone. The correct set-up is the textbook one
mirrored, cart included: the dentist's stool on the patient's left, the assistant's stool on the patient's right,
the cart on the assistant's side at or behind the assistant, and the transfer zone over the chest kept clear. The
textbook right-handed layout, the habit, earns only a quarter. The score is read from where the objects stand at the
end, as a top photo with a protractor centred on the mouth mark would read it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, window and threshold, in one place, and the clock-angle helpers |
| `scoring.py` | The scorer: plain functions of the three objects' states (and the arms' pad-tip points and openings, for 'released'), with a small tracker for stillness, the release window and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of the clock geometry, every scoring tier and window edge, every zero condition, the release rule, NaN states, resets, the exploits we tried (the spec's shortcut checks among them), the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer;
`rigstate.py`, in this folder, reads the arms only to check that each object was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up this dental surgery for the next patient. The patient will lie in the chair with their head on the headrest and mouth at the painted mark. Today's dentist is left-handed and works four-handed with a chairside assistant. Put the dentist's stool (blue), the assistant's stool (green, the taller one) and the assistant's mobile cart (grey) where they go around the chair for this dentist and assistant, each standing upright on the floor, not on the chair, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/left-handed-dental-operatory-zones/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up this dental surgery for the next patient. The patient will lie in the chair with their head on the headrest and mouth at the painted mark. Today's dentist is left-handed and works four-handed with a chairside assistant. Put the dentist's stool (blue), the assistant's stool (green, the taller one) and the assistant's mobile cart (grey) where they go around the chair for this dentist and assistant, each standing upright on the floor, not on the chair, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/left-handed-dental-operatory-zones/dream
```

Run one trial per process (the default). The score depends on where the objects stand, whether a gripper held one
over the last 0.5 s and whether they have stopped moving; the small tracker in the task module keeps each object's
centres over the last 1 s and whether a gripper held it over the last 0.5 s, plus diagnostics for the trace, and
starts a new episode when rigstate sees the embodiment reset or the simulator puts every object back on its mark in
one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`score_line`, the decision-list line that matched; `released`,
`settled`, `still_1s`, `transfer`; `lh_dentist`, `lh_assistant`, `lh_cart`, `rh_dentist`, `rh_assistant`;
`not_released`, `not_settled`, `in_transfer_zone`; the first step each score was reached; each arm's
`left_tips`/`right_tips` and `left_opening`/`right_opening`, `arms_readable` and `arms_unavailable`, or
`rigstate_error`; and for each object its centre, `clock` (h:mm), `phi_deg`, `r_mm` from the mouth mark, tilt,
bottom above the floor, `standing`, `in_transfer`, `held_now` and its path over the last 0.5 s). Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three objects' poses once per control
step (20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the
finger tips) and how far it is open, and scores the state at hand: the score is read at the last step.

Clock angle: seen from above round the mouth mark's centre, clockwise from 12 o'clock, 30 degrees per hour. 12 is
-y (behind the head), 3 is -x (the supine patient's left, toward the arm bases), 6 is +y (the feet, over the chair)
and 9 is +x (the patient's right). Distance: horizontal, from the mouth mark's centre to the object's centre. An
object is in a window when it stands (its vertical axis within 10 degrees of vertical, either end up, and its lowest
point, worked out from its pose, within 5 mm of the floor: so not on the chair, on another object, held up, or on its
side) with its centre 80 to 200 mm from the mouth at a clock angle inside the window. The windows sit inside every
source's zone limits:

| Window | Clock | Degrees | Zone |
|---|---|---|---|
| Left-handed dentist's stool | 12:30 to 3:30 | 15 to 105 | operator, 12 to 5 |
| Left-handed assistant's stool | 8:15 to 9:45 | 247.5 to 292.5 | assistant, 8 to 10 |
| Cart | 8:15 to 11:45 | 247.5 to 352.5 | the left-handed assistant's side: assistant or static zone, 8 to 12 |
| Right-handed dentist's stool | 8:30 to 11:30 | 255 to 345 | right-handed operator, 7 to 12 |
| Right-handed assistant's stool | 2:15 to 3:45 | 67.5 to 112.5 | right-handed assistant, 2 to 4 |
| Transfer zone, kept clear | 5 to 8 | 150 to 240 | left-handed transfer zone, over the chest |

Released: at no step of the last 0.5 s was a gripper's pad-tip point less than 10 mm above the object's top while
within finger reach of its centre. The gripper's yaw is not read, so the reach is the widest over every yaw,
hypot(half the finger gap + 10 mm, 20 mm) + the object's half width (25 mm for a stool, 37.5 mm, half the diagonal,
for the cart) + 5 mm: 90 mm for a stool and 103 mm for the cart fully open, 52 and 65 mm closed. Each gripper is read
on its own; on a step where one cannot be read, the rule skips it but still checks the other (counted in the trace's
`arms_unavailable`). Settled: the object's centre moved under 2 mm in all over the last 0.5 s (its path, not the
reported velocity, which chatters while a part is gripped).

Decision list, first match wins:

| Line | Condition | Score |
|---|---|---|
| 1 | Any object not released or not settled, or any object's centre within 200 mm of the mouth at 5 to 8 o'clock | 0 |
| 2 | Dentist's stool, assistant's stool and cart each in their left-handed windows, every object settled in each of the last two 0.5 s windows (still for 1 s) | 1.0 |
| 3 | Both stools in their left-handed windows | 0.75 |
| 4 | The dentist's stool in its left-handed window, the assistant's stool not in its | 0.5 |
| 5 | The right-handed layout: dentist's stool 8:30 to 11:30 and assistant's stool 2:15 to 3:45 | 0.25 |
| 6 | Anything else | 0 |

Every final state matches one line, and every score can be reached: the demo's layout (dentist's stool at 1:30 and
assistant's stool at 9:00, both 100 mm out, cart at 11:00, 130 mm out) 1.0; the same with the cart left at its start
or set behind the head at 12:30 (the right-handed static zone) 0.75, and the 1.0 layout before it has been still
for 1 s also 0.75; the assistant's stool left at its start or set at 3:00 0.5; the textbook right-handed layout
(dentist's stool at 11:00, assistant's at 3:00, cart anywhere outside the transfer zone) 0.25; the stools swapped
(dentist's at 9:00, assistant's at 2:00) 0; the cart or a stool at 7:00, 150 mm out (the transfer zone) 0; a stool on
the chair, tipped over, still gripped or still moving 0; nothing moved 0. No latch is needed: the layout is judged as
it is left, and moving equipment through the zones while setting up is normal. The trial ends early (success) once
1.0 has held for 2 s; otherwise it is scored as it stands at the run limit. A state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: the right-handed habit (0.25 at most, whatever the
cart); the stools swapped (0); the cart parked beside the chair over the chest (0, even tipped or lifted); the cart
still in the gripper or an open gripper still round it (0); objects stacked, on the chair or lying on their sides on
the right spots (not counted); the dentist's stool parked on the chair between the two operator windows, or at
12:00 (neither window: 0). One dentist's stool cannot be in both operator windows at once, so there is no hedge.

## One arm alone

The demo splits the work by side: the right arm sets the cart and the dentist's stool, both of which start on its
side; the left arm sets the assistant's stool, which starts beyond the foot of the chair on its side. Gripping
straight down, at any heading (tried every 30 degrees, at grip heights 24 to 60 mm), the left arm cannot reach the
dentist's stool or the cart where they start (x 28.0 and 36.0 cm, y -33.0 and -34.0 cm) and the right arm cannot
reach the assistant's stool (x 40.0 cm, y +33.0 cm), by our IK model of the arm (`tests/test_scoring.py`,
`OneArmReach`). So with the left arm idle the right arm's two placements score 0.5 (the dentist's stool in its
window, the assistant's stool still at the foot end), and with the right arm idle the left arm's assistant's stool
alone scores 0 (the dentist's stool still at its start): the offline kinematic check of the scripted demo predicts
0.5 and 0.0. The task does not claim to need two arms as a skill; the second arm is needed here only because the
equipment starts on both sides. We do not claim one arm is strictly unable: a policy that pushed or tilted the
gripper might reach farther. (GPU one-arm scores: 'Results in simulation' below, once run.)

## Demo and run limit

The scripted demo takes 32.8 s (656 control steps), both arms on one clock, gripping straight down with the fingers
closing along y. Stools are gripped across their 50 mm diameter (commanded 42 mm), the cart across its 45 mm width
(commanded 37 mm); the fingers open fully (94 mm) round each part, with the tool point 24 mm (dentist's stool),
35 mm (assistant's stool) or 26 mm (cart) above its bottom. Every gripper open or close takes 1.5 s, every final
descent is slow (20 mm over 1.2 or 1.5 s), each object is let go with its bottom 2 mm above the floor after a 0.5 s
settle, and each set-down point is commanded 4 mm farther from its arm's base than the target, for the few
millimetres an arm trails under load. In order:

1. The right arm sets the cart at 11:00, 130 mm from the mouth (x 32.5 cm, y -6.3 cm), and rises clear by 16.0 s.
   Meanwhile the left arm moves over the assistant's stool, waits there, and grips it.
2. From 16.5 s the left arm carries the assistant's stool to 9:00, 100 mm from the mouth (x 36.0 cm, y +5.0 cm,
   38 mm clear of the chair's corner), lets go and parks over its emptied start (25.8 s).
3. The right arm fetches the dentist's stool, carries it from 23.5 s to 1:30, 100 mm from the mouth (x 18.9 cm,
   y -2.1 cm, 33 mm clear of the headrest), lets go and parks over its emptied start (32.8 s).

The arms never work beside the chair at once: the closest the two tool points come is 26.5 cm and the arms'
collision capsules 9.5 cm (offline). The left arm carries at a tool-point height of 90 mm (the right at 120 mm), so
that at x 36 cm, near the edge of its straight-down reach, joint 4 keeps 16.5 degrees from its limit (the smallest
joint margin; 15.6 on the right arm); the carried stool's bottom then rides 55 mm up, and its path crosses no other
part. The grippers and fingers stay at least 28 mm from the chair and the headrest throughout. Every target is at least 22.5 degrees (39 mm of arc) inside its window's sides and 20 mm inside the 80 to
200 mm band, far more than the few millimetres a part lands off its command. `PLACE_TRIM` in `task.py` moves an arm's
place points if the GPU run shows them landing off by a steady amount. In the offline check success holds at step
664 (33.2 s) and the scripted policy would stop at step 717. The run limit is 1,200 steps (60 s), nearly twice the
demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the dentist's stool upright centred at
x 28.0 cm, y -33.0 cm, the assistant's stool centred at x 40.0 cm, y +33.0 cm and the cart centred at x 36.0 cm,
y -34.0 cm with its sides square to the table; the chair, headrest and mouth mark are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo ends after about 33 s of simulated time, when success
has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

An 80 x 300 x 40 mm block of MDF painted light blue glued to the table 22 cm in front of the arm bases, along y from
y +7.0 to +37.0 cm (the chair), and a 60 x 50 x 40 mm one against its -y end (the headrest), with a 12 mm red dot
painted on the headrest's top at x 26.0 cm, y +5.0 cm (the mouth). Two 50 mm cylinders 40 and 60 mm tall (3D-printed
PLA weighted with steel shot to 120 and 150 g), painted blue and green, and a 60 x 45 x 45 mm grey block (the same,
200 g). Positions in `task.json`'s `setup`.

Scoring a real trial: take a top photo with a protractor (or a printed clock-face template) centred on the mouth
mark, its 12 toward -y, and a ruler. Read each object's centre's clock angle and distance from the mouth; check
each stands upright on the table, not on the chair or another object, after the arms have let go and withdrawn,
and that nothing stands within 200 mm of the mouth between 5 and 8 o'clock.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | not scored: the policy refused the instruction (its message: 'Instruction requires dropping objects off the table onto the floor'), so the trial errored before any action. The instruction was not changed to get past the filter. |
