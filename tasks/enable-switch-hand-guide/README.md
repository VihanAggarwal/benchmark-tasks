# Hand-guide a carriage to two points, enabling plunger held in its middle band

Industrial robots are taught by hand-guiding or jogging them while the operator holds a three-position enabling
device in its middle position: letting go, or squeezing it fully in a panic, stops all motion, and teach mode runs
at reduced speed. Here a black enabling plunger stands in a grey sleeve in front of the left arm, with a scale post
beside it, and a white carriage sits on a framed stage in front of the right arm. The carriage must be guided to a
red mark, then a blue mark, stopping on each, then back home, and it may move only while the plunger is held upright
and lifted in its middle band (15 to 30 mm), never faster than 5 cm/s. The scorer watches every control step: where the
plunger is and which gripper holds it, how far and how fast the carriage moves, and where it stops. Only the left
arm reaches the plunger and only the right arm can guide the carriage; and since the gripper that holds the
plunger is tied to the sleeve, whose centre stands 26 cm from the stage's nearest corner, no single arm could do the
job even with more reach.

The carriage in the simulation is a loose block in a fixed frame, not a driven axis: nothing physically locks it
when the plunger is let go. A real enabling device cuts the motors' power, so the carriage could not move; here the
same rule is measured instead (any movement of more than 2 mm in all while not enabled scores 0), which is what
makes it a test of the policy's discipline rather than of the machine's brakes.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the carriage's and plunger's states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes, a loose cylinder) and the standard `task_score` scorer.

The scorer's "held" test needs the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; on those steps "held" and the grippers' end conditions are
taken as met (the plunger's height and tilt alone decide whether the step is enabled; an upright plunger cannot stay
up in its band unless something holds it), and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hand-guide the white carriage, never more than 5 mm above its plate, to the red mark, then to the blue mark, stopping at least 1 s on each, then back to the green home mark. It may move only while the black enabling plunger on the left is held upright in its middle band (lifted 15 to 30 mm: not dropped, not pulled higher), and never faster than 5 cm per second. Finish with the plunger down and both grippers open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/enable-switch-hand-guide/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hand-guide the white carriage, never more than 5 mm above its plate, to the red mark, then to the blue mark, stopping at least 1 s on each, then back to the green home mark. It may move only while the black enabling plunger on the left is held upright in its middle band (lifted 15 to 30 mm: not dropped, not pulled higher), and never faster than 5 cm per second. Finish with the plunger down and both grippers open." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/enable-switch-hand-guide/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`enabled`, `held_by`, `plunger_h_mm`, `plunger_tilt_deg`,
`speed_mm_s`, `max_speed_mm_s`, `idle_travel_mm`, `stage`, `stop_steps`, `off_red_mm` / `off_blue_mm` /
`off_home_mm`, `carriage_lift_mm`, `breaches`, `first_breach_step`, `opening`, `gripper_clearance_mm`,
`arms_unavailable`, both TCPs and tip points, `carriage_minus_right_tcp_mm`, ...). Videos: `inspect-robots video
<log>.json`. The scorer keeps one trial's record per process and starts it over at every reset, so run one trial per
process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Definitions, all read once per control step:

- **h**: the height of the plunger's lowest point above the table top.
- **Held**: a gripper's TCP within 25 mm of the plunger's grip point (on its axis, 20 mm below its top) and that
  gripper's opening under 0.5 (fingers closer than 47 mm).
- **Enabled**: h between 15 and 30 mm, the plunger upright (tilted under 4 degrees) and held. A plunger pulled
  higher than 30 mm (the "panic squeeze") is not enabling, and neither is one left jammed in its band: to jam across
  its 30 mm sleeve and stay up by itself, the 24 mm plunger must tilt at least 5.4 degrees (with its foot 15 mm up;
  more when higher), so an upright plunger in its band is being held.
- **Speed**: the carriage centre's horizontal displacement over the last two control steps, divided by 0.1 s.
- **Travel** is followed in steps of 0.2 mm, as a ruler would read it: a move counts once the carriage is 0.2 mm
  from where travel was last counted, so physics jitter of a resting carriage adds nothing; travel counted on a step
  that is not enabled is travel while not enabled.
- **Stopped at a mark**: the carriage's centre within 3 mm of it, speed under 2 mm/s, every step enabled, for 1 s
  in a row. Stops count only in order (red, then blue, then home); other stops in between do not matter.

Zero conditions, any moment, whatever follows: travel while not enabled over 2 mm in all; speed over 50 mm/s; the
carriage's lowest point more than 5 mm above the plate (carried, not guided) or the carriage tilted over 10 degrees
(any face down); the plunger tilted over 15 degrees or out of its sleeve; the carriage or the plunger off the table.

| Outcome | Score |
|---|---|
| A zero condition at any moment | 0 |
| Stopped at red, then blue, then home; at the end the carriage within 5 mm of home, h under 2 mm, both grippers open (0.9 or more) with tip points and TCPs 30 mm clear of the carriage and plunger | 1.0 |
| Stopped at red, then blue, but not every 1.0 condition (not back home, plunger left up, a gripper still closed or close by) | 0.75 |
| Stopped at red, but not then at blue | 0.5 |
| In one continuous enabled stretch of at least 3 s, the carriage's centre at least 20 mm from where it was when the stretch began; no stop at red | 0.25 |
| Anything else | 0 |

The score is read at the last step; the trial ends early once the 1.0 conditions have held with the carriage and
the plunger still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised from rigstate's reset hook, or,
as a fallback, when the simulator puts things back at their start (carriage within 10 mm of home, plunger down in
its sleeve, both still) with one of them jumping, in one step, 8 mm further than its velocity and a free fall
allow. A simulator state with a non-finite number scores 0 for that step and leaves the record of the trial as it
was. The carriage is a plain cube (any face may be down); the plunger is the same either end up.

Unintended strategies earn nothing, as the tests show: the right arm guiding the carriage with nothing holding the
plunger travels while not enabled (0); lifting the plunger into its band and letting go of it there, then guiding,
is not enabled (0); jamming it 6 degrees across its sleeve and parking the closed left gripper beside it, which the
"held" test alone would accept, is not upright and so not enabled (0); holding it pulled too high (35 mm) or not
high enough (12 mm) is not enabled (0); carrying the carriage through the air from mark to mark lifts it more than 5
mm (0); flicking it from mark to mark in half a second is too fast (0); making all three stops and then nudging the
carriage after the plunger is let down moves it while not enabled (0); shaking the carriage 3 mm either way for 4.5
s while enabled moves it about 50 mm in all but never 20 mm from where it was, so it earns no 0.25. Pushing the
carriage with an open finger, slowly and while enabled, is hand-guiding too and can score 1.0.

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone) lifts the plunger, holds it and lets it down; the carriage never moves:
0. `DEMO_IDLE_ARM=left` (the right arm alone) guides the carriage while nothing holds the plunger, which breaks the
not-enabled rule within the first few millimetres: 0. With a straight-down grip the right arm cannot reach the
plunger (20 cm left of the midline; the IK misses it by 39 mm), and the left arm cannot centre a grip on the
carriage anywhere along its travel (its centre stays 11 to 19 cm right of the midline; the IK misses the blue mark
by 11 mm); the left fingers could at most touch the carriage's near corner while it stands on the blue mark. The
rule decides it beyond reach: the gripper that keeps the plunger in its band is tied to the sleeve, 26 cm from the
stage's nearest corner, so a single arm cannot enable and guide at the same time, and anything the left arm did to
the carriage itself would be movement while not enabled. (Offline kinematic predictions; the GPU runs are recorded
in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 43.0 s (860 control steps). The left arm (yaw -90 degrees, so its open fingers spread along
x and miss the scale post 43 mm to the side) comes down over the plunger, the last 2 cm over 1.5 s with the pad tips
7 mm above the sleeve's rim, grips it 20 mm below its top (fingers commanded to 16 mm on the 24 mm plunger, over
1.5 s) and lifts it 22.5 mm, to the middle of its band, by 10.5 s (if the arm sags a few millimetres under the
0.10 kg plunger, it is still inside the band). Meanwhile the right arm (yaw +90 degrees) comes down round the
carriage with its fingers 70 mm apart (15 mm clear of each side), the pad tips 9 mm above the plate, and waits. At
11.0 s, once the plunger is up, it grips the carriage (32 mm commanded on 40 mm, over 1.5 s), lifts it 3 mm off the
plate and guides it to red (50 mm in 2.5 s), stops 2.5 s, to blue (106 mm in 5 s), stops 2.5 s, and back home
(58 mm in 3 s), stops 2.5 s. Every move peaks at 29 to 32 mm/s, and every stop is 1.5 s longer than the scorer's
1 s, to leave the carriage time to settle. The right arm sets the carriage down, lets go and goes up by 37.0 s, with
the pad tips 41 mm above the carriage; only then does the left arm lower the plunger back onto the table, let go
and go up (pad tips 37 mm above it). Offline, success has held for its 1 s at step 872; the scripted policy
stops at step 921. The run limit is 1,200 steps (60 s).

Two demo constants are for tuning on the GPU, against `TASK_TRACE`: `C_LIFT` (3 mm: we expect the arm to sag
about 2 mm under the 80 g carriage, so it should ride just clear of the plate and never above the 5 mm limit;
`max_lift_mm` shows it) and `CARRIAGE_AIM` (an offset added to every point the carriage is guided to, zero offline,
to cancel a steady difference between the right TCP and the gripped carriage, `carriage_minus_right_tcp_mm`, if it
is more than about 1 mm; the stops allow 3 mm). The gripped plunger hangs plumb below the left fingers, so
`plunger_tilt_deg` should stay near 0 while it is held; it must stay under 4 for the steps to count as enabled.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the plunger back down in its sleeve,
standing upright on the table, and the carriage back on the plate centred on the home mark, sides parallel to x
and y; the sleeve, post, plate, frame and marks are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 46.0 s of simulated time with the
policy's 3 s hold); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A grey PLA square tube (30 x 30 mm inside, 8 mm walls, 80 mm tall) clamped upright to the bench; a black-anodised
aluminium tube 24 mm across and 120 mm long, closed at both ends and weighted to 0.10 kg, standing in it; a white
PLA post 10 x 10 mm and 120 mm tall beside it, topped by red, green and red PLA bands 15 mm tall (or a printed scale
on a card). For the stage, a light grey PLA plate 200 x 200 x 6 mm with a 20 mm PLA frame on its edge, 10 x 10 mm
coloured PLA squares on the frame's top, and a 40 mm cube printed solid in white PLA (about 80 g); all of it except
the plunger and the cube fixed to the bench. Film the plunger from the side against the bands and the stage from
above at 30 fps: at 5 cm/s the carriage moves 1.7 mm per frame.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
