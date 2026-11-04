# Carry a loaded litter to the trolley: lift and lower both ends together, level

Two-bearer litter and stretcher carries, as EMTs, paramedics and nursing assistants do them: the bearers lift and
lower together on a count and keep the litter level, because a litter raised or set down at one end first tips the
patient, and dragging it along the ground instead of lifting it clear injures the patient and the bearers. Here a
pale grey litter (a 500 mm board) lies across the midline with an orange "casualty" box on its middle, and a fixed
dark grey trolley deck stands 14 cm further forward. Each arm takes one handle; both lift together, carry the litter
level over the trolley, lower it together and let go. The scorer watches both ends' heights, the litter's tilt, its
sliding, the casualty, and both grippers at every control step, not only the end state. The task is two-handed by
physics: the casualty covers the litter's middle, so one gripper can take it only by an end, and lifting one end
tilts it (the ends 40 mm apart in height is 4.6 degrees over 500 mm), which scores 0; the scorer also requires both
grippers to be holding the litter whenever it is in the air.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the litter's, the casualty's and the grippers' states, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |
| `tests/test_demo_plan.py` | Offline checks of the scripted demo against the simulator's and station's limits (reach, fingers, timing, spacing) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (one fixed and two loose boxes) and the standard `task_score` scorer.

The scorer's gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the gripper rules are then not applied on those steps, and the
trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the grey litter by both handles at the same time, keep it level, carry it forward and lower it onto the dark grey trolley, both ends together, then let go. Do not drag the litter and do not lift one end before the other. The orange casualty must stay on the litter." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/litter-lift-level/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the grey litter by both handles at the same time, keep it level, carry it forward and lower it onto the dark grey trolley, both ends together, then let go. Do not drag the litter and do not lift one end before the other. The orange casualty must stay on the litter." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/litter-lift-level/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `end_left_mm`,
`end_right_mm`, `end_diff_mm`, `max_end_diff_mm`, `tilt_deg`, `max_tilt_up_deg`, `clearance_mm`, `drag_mm`,
`setdown_mps`, `carried_level`, `casualty_on`, `casualty_gap_mm`, `centre_dx_mm`, `centre_dy_mm`, `opening`,
`zone_gap_mm`, `grippers_holding`, `grippers_let_go`, `arms_unavailable`, both TCPs, ...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (76 tests; `test_demo_plan.py` stands in
for `tools/offline/check_task.py`, which is not part of this folder).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the litter, the casualty and both
grippers once per control step (20 Hz). The litter's end heights are the heights of the bottom edge of each
40 x 20 mm end face; its tilt is the angle of its long axis from the horizontal. The surface under the litter is
the trolley's top when the litter's centre is over the trolley's 120 x 360 mm footprint, else the table top. The
scorer cannot see which way a gripper is turned, so each gripper's fingers count as a round zone about its axis,
out to the fingers' outer corners (57 mm fully open, 32 mm closed on the 40 mm litter), from the finger tips to
38 mm back up the gripper; a gripper holds the litter when it is less than 0.9 open and that zone is within 10 mm
of the litter. Six rules hold for the whole trial; one broken at any moment scores 0 whatever follows:

1. the litter is dragged: its centre's horizontal travel, summed over the steps on which it is low (less than 10 mm
   above the table top, or above the trolley's top while any part of it is over the trolley), exceeds 15 mm
   (followed in 0.5 mm steps, so a resting litter's contact jitter does not add up);
2. the casualty is not on the litter (its underside more than 6 mm off the litter's top, its centre more than 30 mm
   off, or tilted 10 degrees or more) for 5 control steps in a row (0.25 s; a one-step contact hop as the litter is
   lifted or set down does not count);
3. the casualty tilts more than 15 degrees;
4. the litter's long axis tilts more than 10 degrees;
5. the ends' heights differ by more than 40 mm while an end is up (either end's bottom edge 10 mm or more above the
   table top): lifted or lowered at one end;
6. the litter is 10 mm or more clear of the surface under it while a gripper is not holding it: carried, swung or
   thrown with one hand.

| Outcome | Score |
|---|---|
| Any rule broken at any moment, or the litter or the casualty ends off the table | 0 |
| Litter flat on the trolley (within 3 degrees, along its axis within 10 degrees, centre within 15 mm in x and 20 mm in y), casualty on it, both grippers let go; over the trial the ends never more than 20 mm apart and the tilt never over 5 degrees while an end was up, and it was lowered onto the trolley at 0.10 m/s or less | 1.0 |
| Litter flat on the trolley with its centre within 40 mm in x and y and the casualty on it, but not all of the above (ends 20 to 40 mm apart or tilt 5 to 10 degrees at some moment, dropped the last way, further off centre, or a gripper still holding) | 0.75 |
| Litter resting on the trolley (lowest point within 3 mm of its top, centre over it, within 10 degrees of flat) with the casualty on it, but not meeting 0.75 (turned, rolled on an edge, centre more than 40 mm off) | 0.5 |
| Litter carried level at some moment (lowest point 30 mm up, tilt under 5 degrees, ends within 20 mm, casualty on it) but not on the trolley at the end (set back on the table, or still held) | 0.25 |
| Anything else (never lifted clear, or lifted under 30 mm) | 0 |

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both objects still
(under 1 cm/s and 0.5 rad/s) for 3 s, longer than the demo's release (1.5 s) plus rise (1.0 s), so it cannot end
while a gripper still holds the litter. A new trial is recognised from rigstate's reset hook, or, as a fallback,
when the simulator puts the litter and the casualty back on their starting places (within 4.5 cm, which covers a
seeded reset's jitter) in a single step, further than their last velocity and a free fall could carry them. A
simulator state with a non-finite number scores 0 for that step and leaves the record of the trial as it was. The
litter is a plain board, so either broad face may be down and either end may be to the left.

Unintended strategies earn nothing, as the tests show: lifting one end to the carry height with the other on the
table pivots the litter (ends 63 mm apart, 7 degrees) and breaks rule 5 (0); carrying the litter level with one
gripper, even to a perfect set-down, breaks rule 6 (0; the end state alone would have been 1.0); shoving the litter
along the table up against the trolley and over its edge breaks rule 1 (0); lifting one end 30 mm and swinging the
low end round drags it (0); flinging the litter level onto the trolley breaks rule 6 and throws the casualty off
(0); carrying the casualty to the trolley by itself takes it off the litter (0); skimming the litter forward 8 mm
up and sliding it onto the trolley's deck is dragging whichever surface it is over (0); dragging the litter along the
trolley after setting it down is dragging (0); rolling the litter 12 degrees about its long axis on the way keeps
the ends level but tips the casualty off (0); doing nothing is 0, and lifting the litter less than 30 mm and
putting it back is 0. Lifting the ends unevenly by 20 to 40 mm, tilting 5 to 10 degrees, or letting it fall the last
15 mm onto the trolley caps the score at 0.75, and leaving a gripper closed on the litter on the trolley is 0.75. A
litter set down 45 mm off centre, so that 25 mm of it hangs past the trolley's edge, or turned 15 degrees across
it, is 0.5. Turning the litter end for end or over on the way, with both ends together, changes nothing (1.0).

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) and `DEMO_IDLE_ARM=right` (the left arm alone) each grip one handle and
play the same lift, carry and set-down with the other end untouched. Two things can happen on the GPU, and both
score 0 or at most 0.5:

- If the grasp slips under the load's moment (the far end and the casualty put about 1.0 N m on a 32 mm grip,
  more than the pads' friction can hold on a 20 mm thick board by the estimate below), the far end stays on the
  table and the litter pivots in the fingers: the ends go more than 40 mm apart while the gripped end is up, rule 5
  (0). The offline kinematic picture of this run (the litter pivoting about its far end) scores 0 by rule 5 with
  63 mm between the ends; the casualty most likely slides off the tilted litter as well (rule 2).
- If the grasp holds, the whole litter rises with one gripper, tilted 1 to 2 degrees by the arm's sag, and is
  carried and set on the trolley by its one end: rule 6 (one gripper holding it in the air) scores 0. Should the
  gripper reading be unavailable on those steps, the set-down lands the litter's centre about 21 cm from the
  trolley's centre in y (the held end over it, the rest off the deck), which is not on the trolley: 0.25 at most
  (carried level), or 0.5 only if the deck somehow catches it centred.

Either way a single arm clearly scores less than the demo's 1.0. (Offline kinematic predictions; the GPU runs are
recorded in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 17.5 s (350 control steps), both arms on one clock with identical timings: each arm goes
over its handle (y +-0.21, x 0.24) at 12 cm up, comes down to 3 cm above the grip, then slowly to the grip height
(the finger tips 4 mm above the litter's underside, holding 16 of its 20 mm), settles 0.5 s, closes to 32 mm over
1.5 s across the litter's 40 mm width (the fingers along x: left yaw -pi/2, right +pi/2), settles 0.5 s, lifts
straight up over 2.0 s to the carry height (the litter's underside 63 mm up as commanded), carries 14 cm forward
over the trolley in 3.0 s, comes down to 2 cm above the set-down in 1.0 s, settles 0.5 s, lowers the last 2 cm over
1.5 s (the litter's underside aimed 2 mm above the trolley's deck), settles 0.5 s, opens over 1.5 s and goes up
1.0 s. The TCPs stay 42 cm apart in y throughout. The run limit is 1,200 steps (60 s).

Payload sag: the arms' shoulder and elbow joints are position-controlled springs without gravity compensation for a
payload, and each carries 0.25 kg (half the litter and casualty) at about 0.3 m reach: about 7 mm down and 1 to
2 degrees of gripper tilt, by the same estimate the reference tasks use for 0.2 to 0.3 kg. Both arms sag alike, so
the ends stay level with each other; the litter's underside is then about 56 mm up during the carry, 26 mm clear of
the trolley's 30 mm deck, and 'carried level' needs only 30 mm above the table. At the set-down the sag puts the
litter on the deck about 0.7 s before the slow descent ends, the arms still carrying part of its weight, so there
is no drop; success starts counting only once both grippers have let go (they pass 0.9 open at about 16.3 s,
step 326) and holds for 3 s at about step 386 if the litter is still by then, before the scripted policy's own
stop at step 410 (17.5 s plus its 3 s hold). Without sag (the offline check) the litter drops its last 2 mm when the
fingers open, a step-to-step descent of 0.04 m/s, inside the 0.10 m/s limit. If the GPU shows the litter resting
earlier or later, `TASK_TRACE`'s `clearance_mm`, `end_diff_mm` and `setdown_mps` say by how much.

Grip security: each grip holds a 20 mm thick board 16 mm deep with 32 mm commanded across 40 mm. While both
grippers hold, each carries half the 0.5 kg load straight down through its pads with no moment about the grip, so
the grasp is well inside friction. Only the one-arm runs put a moment on a grip: about 1.0 N m (the litter's and
casualty's weight at 0.21 m from the grip), against roughly 0.4 N m the pads can hold by the reference tasks'
estimate (two pads at 20 N, friction 1.2, about 8 mm of lever on the 16 mm held face), so the litter most likely
pivots in the fingers; whether the simulator's contacts hold it anyway is for the GPU run to show (see above).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the litter back flat on the table centred
at x 24.0 cm, y 0, long side along y, and the casualty back centred on it, sides parallel to x and y; the trolley is
fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 17.5 s of simulated time plus the 3 s
hold); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 500 x 40 x 20 mm board of grey HDPE or PVC (about 200 g; mark the outer 100 mm of each end as the handles), a
300 x 70 x 40 mm orange box weighted to 300 g (a filled plastic case, or a wood block painted orange) set on its
middle, and a 120 x 360 x 30 mm plywood or MDF block painted dark grey, clamped to the bench, as the trolley deck.
Stand cards upright behind the litter's path, out of the arms' way, marked at 10, 20 and 40 mm above the bench and
above the deck, and film each end from the side; tape a 15 mm mark beside the litter's starting place for the drag
and centre lines on the deck (15 and 40 mm either side in x, 20 and 40 mm in y) for a top photo of the end state.

## Results in simulation

To be measured on the GPU (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each, with the
commands above). The offline plan predicts the first row; the one-arm rows are predicted above.

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline plan: 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (predicted 0; 0.5 at most) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (predicted 0; 0.5 at most) |
| Dream-1 | to be measured on the GPU |
