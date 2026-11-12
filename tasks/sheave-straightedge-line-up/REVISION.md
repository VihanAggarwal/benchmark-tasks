# Revision for review TK-2XG49A

What changed, answered point by point against the reviewer's text. The scene, the instruction, the tiers and the press
with the straightedge are unchanged. The title now names the alignment rather than the tool (see "Also: the text").

## "Flags: The demo description explicitly reports a 12 mm motor lift despite the instruction to keep it on the plate."

The demo no longer lifts the motor. The left arm grips the motor's body at its centre of mass's height with the motor
resting on the plate (tool point 60 mm up, the pads' tips 35 mm above the plate's top, the finger carriages 25 mm
above the body's top). Then, without changing that height, it commands a 33 mm slide along the plate toward the arms
(x 33.0 to 29.7 cm) while turning the wrist from 3 degrees to square (4.5 s), eases the command back to x 31.0 cm
(1.0 s) and lets go. The motor follows about 11 mm behind the command (predicted), so it slides about 22 mm and stops
with its pulley face about 10 mm proud of the line. Every left-arm target
from the grip to the release is at the grip height; a test checks this (`TaskFile.test_demo_plan`). Carrying the
motor's weight would need the command about 3 mm higher (the GPU measured a 3.1 mm sag with this motor hanging from
this arm). The README's "Demo and run limit" section describes the slide; the lift and the set-down are gone from the
code and from the demo's description. The demo is open-loop (fixed waypoints, no simulator state read), and the
README says so. The README names the lift only as what the first version did (in "Results in simulation" and the
tests' description); every result there is now from the revised demo's GPU runs.

## "Construct validity (not met): A robot can lift the motor off its plate and still receive full credit despite the instruction requiring it to stay on the plate."

It cannot any more. The scorer has a whole-trial rule (f), checked on every control step from the first: the lowest
bottom corner of the motor's body within 2 mm of the plate's top, the centre of its underside over the plate's
outline, and its up axis within 10 degrees of vertical. One step that breaks it latches 0 for the rest of the trial,
however well the motor ends. Lifting the motor more than 2 mm, even for one step, breaks it. Tests:

- `WholeTrial.test_the_old_demo_lift_scores_zero` replays the old demo's lift, carry and set-down: 0, breach "lifted
  off the plate" at the step the motor passed 2 mm. The end state alone would have scored 1.0. The same path slid along
  the plate scores 1.0.
- `test_lift_boundary_on_any_one_step`: a single step at 1.9 mm keeps 1.0; a single step at 2.1 mm, or 2.1 mm below the
  plate's top, scores 0. A 3 mm lift on one step still scores 0 after 2,000 more steps, until a reset.
- `test_rocking_onto_an_edge_is_still_on_the_plate`, `test_off_the_plate_mid_trial`.

## "Scoring (not met): The plate and centre-distance gates check only the end state, so they do not enforce the stated continuous constraints."

Both are now checked continuously as well as at the end:

- (f) on its plate, on every step, as above;
- (g) on its line, on every step: the centre of the body's underside within 10 mm of y 14.0 cm (the 290 mm centre
  distance). One step that breaks it latches 0 (`test_centre_distance_on_every_step`: 9.9 mm on one step keeps 1.0,
  10.1 mm on one step scores 0);
- a step on which the motor's pose cannot be read (a non-finite number, a zero quaternion) breaks both, because
  neither can be verified there (`Robustness.test_blow_up_states_score_zero_and_leave_the_tracker_alone`).

The end gates (a) to (e) are kept: at rest, upright within 2 degrees, flat on the plate (all four corners inside and
within 2 mm), on its line, hands off. The trace reports `stayed_on_plate`, `stayed_on_line`, the first breach of each
(`plate_breach`, `line_breach`: step, reason, value), `lowest_corner_mm`, `max_lowest_corner_mm` and `max_y_off_mm`.

The latches are cleared only by a real episode reset: rigstate's and mechkit's reset hooks, which call
`Tracker.reset()`. The scorer no longer has a position-based reset fallback (see "Also: no reset from positions"), so
no placement of the parts can clear a breach (`WholeTrial.test_putting_the_parts_back_never_clears_the_latches`,
`Robustness.test_no_position_ever_starts_a_new_trial`); a genuine reset still does
(`TaskFile.test_a_genuine_reset_clears_the_latches`, `WholeTrial.test_only_reset_clears_the_latches`).

## "Feasibility (not met): The scripted demo explicitly lifts the motor 12 mm off the plate, so it does not demonstrate completion as instructed."

The revised demo slides and turns the motor on its plate and never lifts it. Offline it scores 1.0 (kinematic replay
through `tools/offline/check_task.py --strict`; success held at step 1,033, before the scripted policy stops at
1,079; worst IK 0.10 mm). The lowest corner stays exactly on the plate's top and the y offset at 0 on every step, so
neither whole-trial rule is ever broken. The press with the straightedge is unchanged in timing, grips and overdrive.

On the GPU (L40S, one trial each, job of 2026-11-12) the revised demo scores **1.0**: first 1.0 at step 1,015, the
trial ended at step 1,034 (51.7 s) with M 0.008 degree (e 0.0 mm, theta 0.004 degree), `failed_gates` empty,
`plate_breach` and `line_breach` null, `max_lowest_corner_mm` 0.0 (the motor never rose off the plate) and
`max_y_off_mm` 2.4 (the 10 mm limit). The slide left the motor 11.1 mm proud and turned 1.05 degrees, rocked at most
about 1.7 degrees onto its leading edge; the press had it in line by about 40 s. Settle: nothing moved. One arm idle:
left idle **0.0** (the motor never moved, M 5.39 degrees); right idle (the left arm alone) **0.0** (it pushed the
motor through the line to 4.9 mm behind it, theta -1.14 degrees, M 2.12, every gate met, no whole-trial breach).
Dream-1: **0.0** over the full 3,600 steps (it moved neither the motor nor the straightedge). `task.json` `scores`
is "Dream-1 run: 0.0; demo: 1.0" (written by `tools/finalize_task.py` from these runs).

What a quasi-static model predicts for the slide (in the README, with the trace fields that show it). With its wrist free the arm is
about 150 N/m along x (J K^-1 J^T of the joint gains; the same model predicts the 3.1 mm sag measured on the GPU). So
sliding friction leaves the motor about 11 mm behind the command. The slide is therefore commanded 13 mm past the end
point (`DRAG_LAG`), then eased back to it. Through the wrist's pitch, the motor rocks about 2.6 degrees onto its
leading bottom edge while sliding. That edge stays pressed on the plate (the arm pushes down slightly), and the motor
settles flat when the pull eases (the trailing edge, 90 mm back, is about 4 mm up while it rocks). Rule (f) uses the lowest corner, so this keeps the motor on the plate; the 10 degree
tipped-up limit is far from the model's 2.6 (6.6 even for arms half as stiff).

## "What to change: Track the motor's plate contact and centre-distance limits throughout the trial, not only at the end, and prevent runs that violate those limits from receiving credit. Provide a full-score demo that slides and turns the motor while keeping it on the plate and within the specified centre-distance tolerance."

Both are done, as above: rules (f) and (g) on every step, latched to 0, and the slide-and-turn demo. The demo's y
command never leaves y 14.0 cm. The model puts the arm's sideways give under the slide's pull at about 2 mm, inside
the 10 mm tolerance.

## Also: every gripper-dependent rule fails closed

- Before, a step on which the grippers could not be read counted as hands off. Now it is never hands off. Hands off
  (e) must hold on every step of the last 0.5 s: both arms read, and every tool centre point and pad-tip point at
  least 60 mm from the motor's and the straightedge's boxes (`test_unreadable_arms_are_never_hands_off`,
  `test_hands_off_holds_over_the_last_half_second`).
- A step with a non-finite straightedge state is not hands off either (`test_a_bad_straightedge_state_is_not_hands_off_but_not_latched`).
- The task has no other gripper-dependent rule or keep-out zone.

## Also: the text

- Title: "Line up a motor's belt pulley with a blower's pulley 290 mm away" (was "... using a straightedge"). The
  scorer judges the alignment, and lining the motor up without the straightedge also counts, as the README says; the
  title now names what is scored. The context says the straightedge is the trade's tool for reading the alignment and
  gives the numbers that make the band hard: 1/2 degree at 290 mm is 2.53 mm of offset, or 0.56 mm of turn across the
  64 mm pulley.
- Fidelity: the motor is "a 150 g printed shell so one gripper holds it", next to a plain statement that a production
  motor of this shaft size is several kilograms and takes more push; the 1.2/1.0 friction is marked as chosen.
- Scoring: a real-trial reading for every threshold: feeler gauges at the motor pulley's rims against the straightedge
  (or a try-square and rule from a tape line), a side video level with the plate's top against a card ruled 2 mm up
  for rule (f), a tape mark with ticks 10 mm either side for rule (g). The README has the full table.
- Real-world build: the same materials as the setup (a steel rod for the motor's shaft, a galvanised steel plate);
  "Third-party assets: none".

## Numbers that changed

- Demo: grip at the tool point 60 mm up (was 72 mm); no lift (was 12 mm); a slide command from x 33.0 to 29.7 cm,
  eased back to 31.0 cm (was a carry to 31.2 cm); total still 50.9 s, 1,018 steps; run limit unchanged (3,600 steps).
- Scorer: new layout constants `PLATE_CONTACT_TOL` 2 mm and `TIP_LIMIT_DEG` 10. `Y_TOL` (10 mm) now applies on every
  step. The hands-off window is 0.5 s. `RESET_JUMP`, `START_TOL` and `START_YAW_TOL_DEG` are gone with the
  position-based reset fallback.
- `task.json`: `scoring` rewritten (2,881 of 3,150 characters); `title`, `context` (938 of 950), `fidelity` (483 of 490)
  and one clause of `setup` (the motor's centre of mass) as above; `scores` from the revised GPU runs. The instruction
  and reset notes are unchanged.
- Tests: 65 (were 56), all passing. A stale assertion that `task.json` has no `media` field now checks the media id's
  form instead.
- The README's "Results in simulation", "One arm alone" and trace notes give the revised version's GPU values; the
  first version's results table is gone (one sentence names its 12 mm lift).

## Also: no reset from positions

A simulated review found that the whole-trial latches could still be cleared. `Tracker.update` used to call
`reset()` when an object jumped 25 mm or more in one control step and every loose part was then at its start. So a
robot could lift the motor (breaking rule (f)), drop it 3 to 4 cm onto its exact start with the straightedge at its
start, have the latch cleared, and then line the motor up for 1.0: the original complaint in another form ("forgets
a violation once the scene is restored").

That fallback is removed. A new trial starts only from the real episode-reset hooks: rigstate's `on_reset` and
mechkit's episode counter, both of which call `Tracker.reset()` in `task.py`. The offline checker runs each trial
through those same hooks, so it needs no fallback. Tests:

- `WholeTrial.test_putting_the_parts_back_never_clears_the_latches`: lift 2.1 mm, 30, 35 or 40 mm, drop onto the
  exact start with the straightedge at its start, slide the motor into line, let go: 0, breach "lifted off the plate"
  (the end state alone scores 1.0). The same with a 30 mm slide along y and back: 0 through rule (g). The same moves
  with no breach: 1.0.
- `Robustness.test_no_position_ever_starts_a_new_trial`: everything jumping back to its exact start, a flung bar, and
  a gradual walk back all continue the trial.
- `TaskFile.test_a_genuine_reset_clears_the_latches` (each hook, through `task.success()`) and
  `WholeTrial.test_only_reset_clears_the_latches`: a real reset clears both latches and the next trial can score 1.0.

No change to the demo, the layout or the tiers. The README's Run paragraph says that only the hooks start a new trial.
