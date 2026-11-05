# Two-rescuer CPR with an AED: pads on during compressions, all clear for the shock

Two-rescuer adult basic life support with an automated external defibrillator (AED), as taught under the AHA and
ERC resuscitation guidelines, splits the work: one rescuer keeps chest compressions going while the other switches on
the AED and attaches its two pads (anterolateral: one below the right collarbone, one on the left side below the
armpit). Compressions stop only when the AED analyses the heart rhythm. Nobody may touch the patient during the
analysis, because movement corrupts it, or when the shock button is pressed, because the shock can pass into a
rescuer. The rescuer who presses the button checks that everyone is clear first. The pause before the shock is kept
under 10 seconds, and compressions restart at once after it. EMTs and paramedics (O*NET 29-2042, 29-2043),
registered nurses (29-1141), lifeguards (33-9092) and workplace first-aiders are trained in it.

Here a CPR manikin stand-in lies face up across the table, head to the robot's left, with a red compression mark at
the centre of the chest and two blue pad squares. The right gripper is the compressing rescuer and the left the AED
operator. An AED with an orange shock button stands beside the head, and two white pad blocks lie beside it. The
scorer watches every control step: whether the pads went on while compressions continued, whether anyone was at the
patient during the analysis or the press, and how long compressions paused. The second gripper's role is the
procedure's: the pads must go on while compressions continue, so one gripper cannot do both, and the gripper that
presses the button is the one that is not on the chest.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, mass, position, threshold and colour, in one place |
| `scoring.py` | The scorer: plain functions of the two pads' states and the grippers' state, with a tracker for the whole trial (compressions, pad arrivals, the AED's state, the shock) |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step (unchanged copy) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, the AED's behaviour, resets, the gripper hook, the demo with tracking lag and the exploits we tried, through a small kinematic stand-in (no simulator) |
| `tests/test_demo_plan.py` | Offline check of the scripted plan against the simulator's and station's limits (generic) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed and loose boxes, one fixed cylinder) and the standard `task_score` scorer.

The rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks. `rigstate.py` (imported
by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool point,
finger-tip point, finger opening, in the objects' frame) before each control step and to start a new trial at each
reset. It reads only state the simulator already has and changes nothing the embodiment does. If it cannot read the
arms it prints why once. On those steps no compression or press can be seen and the area counts as clear; the trace
counts them (`arms_unavailable`). If rigstate failed for a whole trial, the most a run could score is 0.25.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Do two-rescuer CPR with the AED on the manikin. With the right gripper closed, start chest compressions on the red mark within 10 seconds: repeated strokes from at least 3 cm above the mark down to within 1.5 cm of it, without pressing on the chest. Keep compressing while the left gripper puts one white pad flat on each blue square. Then both grippers stand clear, more than 5 cm from the manikin, while the AED analyses for 3 seconds; then press the orange shock button with the closed left gripper. Never touch the button while a gripper is near the manikin. Never pause compressions for more than 10 seconds before the shock; restart them within 3 seconds after it and keep them going." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/aed-stand-clear/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Do two-rescuer CPR with the AED on the manikin. With the right gripper closed, start chest compressions on the red mark within 10 seconds: repeated strokes from at least 3 cm above the mark down to within 1.5 cm of it, without pressing on the chest. Keep compressing while the left gripper puts one white pad flat on each blue square. Then both grippers stand clear, more than 5 cm from the manikin, while the AED analyses for 3 seconds; then press the orange shock button with the closed left gripper. Never touch the button while a gripper is near the manikin. Never pause compressions for more than 10 seconds before the shock; restart them within 3 seconds after it and keep them going." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/aed-stand-clear/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`breaches`, `compressions`, `first_compression_s`,
`in_progress`, `min_tip_mm`, `right_tip_above_mark_mm`, `sites`, `arrived_in_cpr`, `analysis_s`, `charged`,
`charges`, `press_steps`, `shock_step`, `pre_shock_pause_s`, `max_pause_s`, `post_shock_pause_s`,
`post_shock_run_s`, `area_clear`, `clear_margin_mm`, `opening`, `arms_unavailable`, both TCPs, ...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator (from the `work` folder above `tasks/`):
`PYTHONPATH=tools/offline/yam_stub:tasks/aed-stand-clear python3 -m unittest discover -s tasks/aed-stand-clear/tests -v`
(the tests also find `tools/offline/yam_stub` by themselves). 71 tests, all passing.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Heights for the right gripper are of its finger-tip point
(the point between its finger tips) above the compression mark's top.

**Definitions**

- **Closed**: finger gap 30 mm or less.
- **Compression**: the right gripper, closed, with its finger-tip point within 20 mm of the compression mark's centre
  horizontally, comes down from at least 30 mm above the mark to 15 mm or less. Before the next one counts, it must
  go back up to 30 mm, closed over the mark. Opening, or leaving the 20 mm, starts this over. Only the right
  gripper's compressions count.
- **Compressions in progress**: the right gripper closed, within 20 mm of the mark's centre and at most 45 mm above
  it, with a compression counted in the last 1.5 s.
- **Finger zone**: rigstate gives no gripper heading, so each gripper's fingers are a stack of discs along the 38 mm
  from its finger-tip point back up the gripper. Each disc reaches a finger's outer corner: half the gap plus the
  8 mm finger, and 13.5 mm across. That is 57 mm fully open and 21 mm closed.
- **At the patient**: part of a gripper's finger zone is inside the box that extends 50 mm beyond the torso, or
  beyond the head, on every side seen from above, from the table up to 50 mm above that block's top. **Clear**:
  neither gripper is at the patient.
- **On its site**: a pad tilted less than 10 degrees (either 40 x 40 mm face down), its lowest point within 3 mm of
  the torso's top, its centre within 20 mm of the site's centre. The pads are identical; either may go on either
  site. A pad **went on during compressions** if compressions were in progress at the step it last arrived on a site.
- **Pressing**: a closed gripper's finger-tip point within 15 mm of the button's centre horizontally and at most
  6 mm above the button's top.
- **The AED**: it analyses while both pads are on their sites and the area is clear. 3 s of that without a break
  completes an analysis and charges it. The charge is **qualified** if, at that moment, both pads went on during
  compressions. Any break restarts the analysis. A charge is cancelled when a pad leaves its site or a compression is
  counted. A press that began after the AED charged and is held 0.5 s delivers the shock; a press already under way
  when it charges does nothing. One shock per trial: the AED does not analyse again after it (its next analysis would
  come two minutes later, beyond the run).
- **Longest pause** (before the shock): the longest of the time from the trial's start to the first compression,
  every gap between compressions, and the **pre-shock pause** (the last compression to the shock).
- **Post-shock pause**: the shock to the first compression after it. **Kept up**: from that compression to the last
  of an unbroken run of compressions, each at most 1.5 s after the one before.

**Rules for the whole trial**: breaking one at any control step scores 0, whatever follows.

1. A press of the shock button begins while the area is not clear (whether or not the AED is charged, before or
   after the shock).
2. The area is not clear while a press that began on the charged AED is held, before the shock.
3. A pad rests on the chest (lowest point within 5 mm of the torso's top) with its outline seen from above
   overlapping the compression mark.
4. A pad's lowest point is more than 10 mm below the table top (off the table).

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Shock from a qualified charge; longest pause at most 10 s; post-shock pause at most 3 s; kept up at least 5 s; both pads on their sites and the left gripper not at the patient | 1.0 |
| Shock from a qualified charge, but not all of the 1.0 conditions | 0.75 |
| A qualified analysis completed, but no shock from a qualified charge | 0.5 |
| Both pads on their sites at once at some step | 0.25 |
| Anything else (nothing done, compressions only, one pad, pads beside their sites) | 0 |

The score is read at the last step. The trial ends early once the 1.0 conditions have held with both pads still
(under 1 cm/s and 0.5 rad/s) for 3 s. A new trial is recognised from rigstate's reset hook. As a fallback, it is also
recognised when the simulator puts both pads back on their starting places (within 4.5 cm) in a single step. A
simulator state with a non-finite number scores 0 for that step and leaves the trial's record as it was.

### Why these numbers

The manikin is an adult chest at about 0.7 scale (24 cm across, 30 cm from shoulders to waist). Thresholds from the
guidelines' timings are kept at their real values; distances are set from the manikin, the station's gripper, and
what a person checking a video can see.

| Threshold | Value | Why |
|---|---|---|
| Longest pause before the shock | 10 s | The guidelines ask for compressions to start within 10 s, for interruptions to be kept under 10 s, and for the pre-shock pause to be under 10 s. Kept at the real value. The demo's longest pause is its 5.7 s pre-shock pause |
| Post-shock pause | 3 s | The guidelines say to resume compressions immediately after the shock. 3 s is the time a hand needs to come back down (the demo takes 1.7 s), with no waiting |
| Kept up | 5 s, gaps of 1.5 s at most | Stands in for the two minutes of CPR before the next rhythm check. Long enough to show compressions really resumed (five or more), short enough for the run |
| Compression stroke | from 30 mm or more down to 15 mm or less | A real arm must not press on a hard torso, so a compression is a dip that stops short of the chest. 15 mm of travel is a deliberate stroke a side video shows. The demo commands 40 mm to 8 mm, so tracking that halves the stroke still counts. Not to scale: real compressions are 5 to 6 cm deep |
| Over the mark | 20 mm | The mark is 30 mm across: over it, or within 5 mm of its edge. The heel of a hand on the lower half of the breastbone |
| On the chest | at most 45 mm above the mark | The top of the demo's stroke (40 mm) plus 5 mm, and below the 50 mm clear line, so a hand on the chest is never clear |
| Closed | gap 30 mm or less | The two finger pads together make a block at most 46 mm across, a scaled hand heel. An open gripper (94 mm) straddles the mark and touches nothing. A gripper holding a 40 mm pad is not a compressing hand |
| In progress | a compression in the last 1.5 s | Real compressions come every 0.5 to 0.6 s; the demo's every 1.3 s. A longer gap is a pause |
| Clear | fingers more than 50 mm outside the manikin, or more than 50 mm above it | 'Nobody touching the patient', with a gap a video can show: 5 cm here is 7 cm on a full-size chest. The zone covers hands above the chest and beside it, and the head |
| Analysis | 3 s | A real AED analyses for several seconds and then charges. Shortened for the run, but long enough that lifting the hands for a moment does not complete it |
| Charge cancelled | a pad off its site, or a compression | A pad off stops a real AED; compressions after the analysis make it stale |
| Press | tips within 15 mm of the button's centre, at most 6 mm above it, closed, 0.5 s | The button is 30 mm across. 6 mm allows for tracking error while the fingers touch it. Closed, so the finger pads are on the button, not straddling it. 0.5 s makes it a deliberate press, not a brush |
| On its site | centre within 20 mm, lowest point within 3 mm of the chest, tilted under 10 degrees | Within 20 mm the 40 mm pad still covers the site's centre point; real pads are placed to within a few centimetres. Flat on the chest, not held above it or propped on an edge |
| Pad on the compression point | lowest point within 5 mm of the chest, outline over the mark | A pad on the breastbone is where the hands go; never placed there |
| Success hold | 3 s | Longer than a release and retreat (1.5 s + 1.0 s), so the trial cannot end with a gripper still on a pad |

### Exploits we tried (all in `tests/test_scoring.py`)

- **Doing nothing**: 0. **Right arm alone** (`DEMO_IDLE_ARM=left`): compressions the whole time, no pads: 0.
  **Left arm alone** (`DEMO_IDLE_ARM=right`): both pads on and a clean shock, but with no compressions the charge is
  not qualified: 0.25.
- **Shocking while compressions go on** (the right hand never lifts): the AED never charges, and the press begins
  with a hand on the chest: rule 1, 0.
- **Shocking with the right hand resting on the chest after a clean analysis**: rule 1, 0. **The right hand coming
  back during the 0.5 s press**: rule 2, 0. **Pressing again after the shock while compressions go on**: rule 1, 0.
- **Pressing before the analysis completes**: nothing happens. **Holding the button down through the analysis**:
  nothing happens until it is let go and pressed again.
- **Touching the patient during the analysis**: it starts again from zero. **Compressing again after the AED has
  charged**: the charge is cancelled, and a press then does nothing.
- **Putting the pads on before compressions start** (the right arm starting late): even after a clean shock, 0.25. A
  pad lifted and put on again while compressions go on does count.
- **Slow 'compressions'** (a dip every 2 s): compressions are counted, but not in progress when the pads arrive:
  0.25 at most. **Compressing with an open gripper**, **holding a pad in the compressing hand**, **dips of under
  15 mm**, **dips from under 30 mm**, **dips 21 mm off the mark**: no compressions are counted.
- **One early compression, then hands resting for 11 s**, then the demo: everything else right, but the longest pause
  is over 10 s: 0.75. **Starting 8 s late**: 0.75.
- **Hovering the pads 6 mm over their sites**: never on, no analysis: 0.
- **A pad set on the compression mark**, even for one step and then removed: rule 3, 0 whatever follows. **A pad off
  the table**: rule 4, 0.
- **Pushing a pad onto its site along the chest during compressions**: allowed; it counts as on.
- **Swapped pads** (each pad on the other's site): allowed, 1.0.
- **Stopping short**: a pre-shock pause of about 11 s, a post-shock pause of about 4 s, only four compressions after
  the shock, a 2 s rest early in the post-shock compressions, the left hand back on the chest at the end, or a pad
  pushed off its site after the shock: 0.75 each. A pre-shock pause 0.2 s under 10 s still scores 1.0. No press after
  a qualified analysis: 0.5, whether the AED stays charged or compressions resume and cancel the charge.

## One arm alone

Offline kinematic predictions from the demo switches (tested in `tests/test_scoring.py`). The GPU runs will be
recorded in `task.json`'s `scores`.

- `DEMO_IDLE_ARM=right` (the left arm alone) puts both pads on, waits at the button, and shocks after a clean 3 s
  analysis (the idle right arm, at its ready pose, is clear): **0.25**. No compression was ever made, so the charge is
  not qualified.
- `DEMO_IDLE_ARM=left` (the right arm alone) compresses 33 times (24, a pause standing clear, then 9), but no pad
  is ever put on: **0**.

No single arm can score more than 0.25. Every tier above 0.25 needs the pads to arrive while the right gripper is on
the chest and compressing, and only the right gripper's compressions count. Reach also splits the work: the
compression mark is on the right arm's side of the midline, and both pad sites, the pads and the AED are on the
left's.

## Demo and run limit

The scripted demo takes 53.2 s (1,064 control steps); offline, the trial ends at about 50 s.

1. The right arm closes its fingers (16 mm) at its ready pose, comes over the compression mark and starts
   compressing: tips from 40 mm to 8 mm above the mark, 0.3 s down, 0.4 s held, 0.3 s up, 0.3 s held (46 a minute).
   The first compression counts at 4.55 s; 24 are made before the shock.
2. Meanwhile the left arm picks pad A (fingers commanded 32 mm across the 40 mm pad, tips 4 mm above its bottom),
   carries it with its bottom 30 mm above the chest, lowers the last 2 cm slowly onto the sternal site (on at 12.0 s)
   and lets go. It does the same with pad B onto the apex site (on at 27.8 s). Both pads arrive during compressions.
3. The left arm rises away from the chest (clear at 30.5 s), goes over the AED and closes its fingers 76 mm above the
   button (34.6 s). The right arm compresses until then, then lifts its tips to 80 mm above the mark (clear at
   35.6 s).
4. The AED charges at 38.6 s. The left hand comes down 1 s later and presses (2 mm above the button's top) for 1 s:
   shock at 40.1 s. Pre-shock pause 5.7 s.
5. 0.6 s after the shock the right hand comes back down. First compression at 41.75 s (post-shock pause 1.65 s); it
   then compresses eight more times. Success holds from 47.0 s.

Geometry, checked in `tests/test_demo_plan.py`: every waypoint and every sampled point is at least 3.3 cm (left) and
8.0 cm (right) inside the forward reach limits, and 3.0 cm in front of the 10 cm minimum. Whenever the TCPs are
less than 22 cm apart in y they are at least 7.5 cm apart in x, and they are never closer than 9.7 cm. The station
asks for 20 cm in y within 5 cm in x. The references disagree on which way the fingers close at yaw +-pi/2, so the
pads are square, and every finger clearance holds either way. With the left gripper fully open over the apex site
(110 mm across its fingers) and the right compressing, the fingers are at least 24 mm apart.

Tracking: the demo was replayed with first-order lags of 0.1 to 0.3 s on either or both arms, and scored 1.0 each
time. The lag delays the press and shrinks the strokes; the holds at each end of a stroke keep it counting. With a
0.35 s lag on the right arm the first strokes after the shock fall short, the post-shock pause goes to 3.35 s, and
the score drops to 0.75. On the GPU, read `min_tip_mm`, `post_shock_pause_s` and `pre_shock_pause_s` in `TASK_TRACE`.
The pads weigh 15 g, so payload sag is negligible. Both grippers are empty when they compress or press, so they do not
sag either.

The run limit is 1,200 steps (60 s). The procedure has a fixed 3 s analysis and 5 s of kept-up compressions on top of
two pick-and-place moves made while the other arm compresses. The demo uses about 50 s of the 60.

## Novelty

The closest board tasks we were told about:

- **"Place ECG leads on the correct electrodes"**: leads placed on a manikin by colour, with one arm, no timing and no
  clearance rule.
- **"Move a casualty manikin onto a stretcher, keeping its head in line"**: patient handling, scored on the end pose.

This task differs in mechanism, not only in objects:

- One gripper must keep up a rhythmic action (compressions) while the other places the pads; the placement only
  counts if it happens during that action.
- A clearance condition gates a device. The analysis runs only while both grippers are clear, and touching restarts
  it. Pressing the button while a gripper is at the patient scores 0, whatever follows.
- The peri-shock pauses are scored at every step: the longest pause in compressions, the pre-shock pause, the
  post-shock restart and the kept-up compressions.

None of the board tasks, or the common sets (LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench, RoboDojo),
scores a two-rescuer protocol, an all-clear rule, or the pauses around a shock. A run that puts both pads on and
shocks while the other gripper still compresses ends in the same pose as the demo, and scores 0.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both pads back flat on the table, centred at
x 13.0 cm, y +28.5 cm and x 28.0 cm, y +28.5 cm, sides parallel to x and y. The manikin, its marks and the AED are
fixed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is about 50 s of simulated time). The first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

- **Manikin**: a skin-coloured foam block 240 x 300 x 50 mm, with a 100 x 110 x 50 mm head block 10 mm beyond its
  shoulder end, both taped to the bench. A red 30 mm square and two blue 40 mm squares are printed and stuck on its
  top at the stated places.
- **AED**: a 100 x 100 x 45 mm yellow-green box clamped beside the head, with a 30 mm orange cap on top. A
  capacitive touch sensor in the cap logs presses with timestamps.
- **Pads**: two white firm-foam blocks, 40 x 40 x 24 mm, 15 g each.
- **Filming**: tape an outline on the bench 5 cm outside the manikin. Stand a card beside the chest with lines
  15, 30 and 45 mm above the compression mark and 50 mm above the chest. Film from above and from the side.

Nothing presses on the manikin: the compressing hand stops 8 mm above the chest, so a real trial is safe for the arm
and the manikin.

## Results in simulation

To be measured on the GPU (Isaac Lab-Arena, `-E object_jitter=0 -E control_hz=20`, one trial each). Not yet run:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline prediction 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (offline prediction 0) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (offline prediction 0.25) |
| Dream-1 | to be measured on the GPU |
