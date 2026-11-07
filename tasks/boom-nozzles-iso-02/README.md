# Fit all six nozzle bodies on a sprayer boom with the 02 size, by the ISO colour code

Before a spraying job the operator checks every nozzle on the crop sprayer's boom: the job's application rate and
forward speed fix one nozzle size, and a boom with mixed sizes sprays unevenly (over-dosing under the larger
nozzles, under-dosed stripes under the smaller). Spray nozzles are colour-coded by size under ISO 10625 (01 orange,
015 green, 02 yellow, 025 lilac, 03 blue, 04 red, ...). Here a bench section of a boom with six black nozzle bodies
stands on the table; at the start the bodies hold, from the robot's left, yellow, blue, nothing, lilac, yellow and
blue, and two grey spares pads hold two yellows each among red, green and orange spares. The job card calls for
02. The robot must know that 02 is yellow (nothing in the scene says so), keep a yellow in the two bodies that
already hold one, take the blue and lilac nozzles off into the white parts trays, and stand a yellow in each of the four bodies that
need one, leaving nothing lying anywhere else. The end state is checked from a top photo: which colour stands in
each body, and where every other nozzle is.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the nozzles' states, with a small tracker (each nozzle's last 0.5 s of path) |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero rule and cap, the exploits we tried, the hooks, the demo plan and the text (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, loose cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The job card calls for size 02 flat-fan nozzles. Make each of the six black nozzle bodies on the sprayer boom hold one 02 nozzle standing upright in it, telling the sizes apart by the ISO colour code. The 02s already fitted can stay. Put every nozzle of another size that you take off the boom into one of the two white parts trays, and fit 02s from the spares on the grey pads. Leave no nozzle anywhere else." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/boom-nozzles-iso-02/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The job card calls for size 02 flat-fan nozzles. Make each of the six black nozzle bodies on the sprayer boom hold one 02 nozzle standing upright in it, telling the sizes apart by the ISO colour code. The 02s already fitted can stay. Put every nozzle of another size that you take off the boom into one of the two white parts trays, and fit 02s from the spares on the grey pads. Leave no nozzle anywhere else." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/boom-nozzles-iso-02/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`points`, `cap`, `holds` per body, `places` per nozzle,
`fitted_02`, `wrong_size_in`, `loose`, `removed_on_pad`, `tilt_deg`, `bottom_mm`, `path_mm`, each nozzle's
position). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the thirteen nozzles' poses, once per
control step (20 Hz), and keeps each nozzle's last 0.5 s of positions to judge "still" from the path (2 mm or
less), not from the reported velocity, which chatters for a gripped part. Each nozzle is in exactly one place,
judged by its centre seen from above and its lowest point: in a body (over the 32 x 32 mm opening, lowest point
less than 40 mm above the boom's top, so a nozzle stacked on a fitted one counts as a second nozzle in that body),
in a tray (over the 90 x 90 mm inside), on a pad (over the 100 x 110 mm top), loose (anywhere else, including still
held up) or off the table. The six body openings, two tray insides and two pad tops do not overlap seen from
above. "Fitted" is in a body, upright within 10 degrees (either end down), lowest point within 3 mm of the boom's
top, and still. Sizes go by colour: yellow is 02; blue (03), lilac (025), red (04), green (015) and orange (01) are
other sizes. A new episode is recognised when the simulator puts every nozzle back on its starting mark (within
4.5 cm of it) in a single step. A simulator state with a non-finite number, a zero quaternion or a missing nozzle
scores 0 for that step and leaves the tracker as it was.

| Outcome | Score |
|---|---|
| A nozzle off the table, a body holding two or more nozzles, or L3 or R2 without a fitted 02 at the end | 0 |
| Otherwise, points: 0.25 for each of L2, L1, R1 and R3 holding one nozzle, a fitted 02 | 0 to 1.0 |
| ... at most, while any body holds a nozzle of another size (fitted, tipped or lying in it) | 0.5 |
| ... at most, while any nozzle is loose, or a blue or lilac (removed) nozzle is on a spares pad | 0.75 |
| Everything right (each body one fitted 02, every other nozzle in a tray or on a pad), held 2.5 s | 1.0 |

The score is the smaller of the points and the caps. Every tier is reachable: the start scores 0 (L3 and R2 are
right, but they earn nothing); only L1 filled 0.25; L2 and L1 done with R1's lilac and R3's blue still fitted 0.5
(the cap); three bodies done and R1 emptied but not refilled 0.75, or every body done with a removed nozzle left on
the table 0.75; everything right 1.0. The trial ends early once 1.0 has held for 2.5 s (rather than the usual 1 s:
the last nozzle counts as fitted once it stands still on the boom's top while the gripper still holds it, and 2.5 s
from then is past the end of the gripper's opening, so a trial does not end with the gripper closed on it).

Unintended strategies earn less, as the tests show: standing the yellows on top of the blue and lilac instead of
taking those off puts two nozzles in a body (0); moving L3's yellow into L2 instead of fetching a spare leaves L3
without its 02 (0); knocking the wrong nozzles onto the table leaves them loose (0.75 at most) and knocking one off
the table scores 0; tipping a blue over inside its body still leaves it in the body (0.5 at most); an orange 01
fitted for a yellow earns nothing and keeps the cap (0.5); a removed blue or lilac put back on a spares pad among
the new spares caps at 0.75; a nozzle held in the air at the end is loose (0.75 at most), and one held low over a
body without standing on the boom earns nothing. The scorer reads only the end state, so an 02 taken out of L3 and
put back, or replaced by another 02, counts as one left alone (the two are indistinguishable in a photo); the
instruction says only that the fitted 02s can stay, so it sets no rule the scorer leaves unchecked.

## One arm alone

The demo uses both arms, each working its own half of the boom from its own pad and tray: the left arm takes L2's
blue to the left tray and fits yellows in L2 and L1; the right arm takes R3's blue and R1's lilac to the right tray
and fits yellows in R3 and R1. With one arm idle the other does its own half and the score is 0.5 either way
(measured in simulation: 0.5 with the left arm idle, 0.5 with the right arm idle): the other half keeps a wrong
size fitted (L2's blue, or R1's lilac and R3's blue), which caps the score at 0.5. Gripping straight down, as the
scripted demo does, one arm alone cannot do better: it reaches its own three bodies, the other side's inner body
(L1 or R1) and both trays, but neither the other side's middle and outer bodies nor the other pad, and its own pad
holds only two 02s, so at most two of the four bodies can get one (0.5). This ceiling holds only for
straight-down grips: kinematically (joint limits, with the arm clear of its own base and the table, not checked
against the nozzles), a gripper tilted about 20 degrees reaches the other side's middle body and the inner column
of the other pad, 30 degrees its outer column and 45 degrees the other side's outer body, so a single arm that
tilts its grip could in principle finish the boom alone. No claim is made that the task needs two arms; with two, each works its own half at once.
(The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 70 s (1,400 control steps): seven pick-and-place moves of 15.2 s each, four by the right
arm and three by the left, the right arm starting 7.2 s (half a move) after the left so that the two grippers are
never both low near the middle of the boom; whenever both are low they are at least 0.25 m apart across y (offline:
closest TCPs 25 cm, arm links 8.5 cm apart at the closest). Every grip is straight down with the fingers closing
along y (the next body along the boom is 125 mm away): the gripper comes down over a nozzle with the fingers 42 mm
apart (clear of the spares 55-60 mm away on a pad), takes the last 20 mm slowly, closes over 1.5 s to a commanded
14 mm on the 22 mm nozzle with the pad tips 15 mm above its bottom (7 mm above a body's walls), lifts it until
the TCP is 13 cm above the table (the nozzle's bottom 102 mm up, 45 mm over the fitted nozzles' tops), carries it
over the target, lowers the last 20 mm over 1.5 s
and opens over 1.5 s with the nozzle's bottom 1 mm above the boom's top (in a body) or 3 mm above the tray floor
(removed nozzles stand in the trays, 40 mm apart in the right one). A 17 g nozzle barely loads the arm, so no
place point is trimmed for sag. Offline (kinematic proxy, no physics) every nozzle lands where planned, success has
held for its 2.5 s at step 1,358, and the predicted score is 1.0. In simulation the demo scored 1.0 and its
trial ended at step 1,360, every fitted nozzle upright (tilt 0 degrees) on the boom's top. The run limit is
2,400 steps (2 minutes), 1.7 times the demo.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the nozzles back upright and centred, in the
bodies L3 yellow, L2 blue, R1 lilac, R2 yellow, R3 blue (L1 empty); on the left pad yellow at x 15.0, y 23.5 cm,
red at 15.0, 29.5, green at 20.5, 23.5 and yellow at 20.5, 29.5; on the right pad orange at 15.0, -23.5, yellow at
15.0, -29.5, yellow at 20.5, -23.5 and red at 20.5, -29.5; both trays empty. The boom, bodies, trays and pads are
fixed.

## Runtime per trial

Measured on an L40S, one trial per process, Isaac Sim's start (about 13 s on a warm instance) included: about
4.3 minutes for the scripted demo (68 s of simulated time, 1,360 steps) and about 6.9 minutes for the Dream-1 trial
at the full 2,400 steps (120 s of simulated time), with 50 objects in the scene. The first start of Isaac Sim on a
new instance takes several minutes more.

## Real-world build

A 680 mm length of 44 x 25 mm aluminium bar clamped to the bench; six square sockets glued on its top, 125 mm
apart, each four strips of 5 x 8 mm black plastic round a 32 x 32 mm opening (or cut from a black 3D print). Two
white plastic trays 100 x 100 x 25 mm (90 x 90 mm inside) and two grey plates 100 x 110 x 5 mm. Thirteen acetal
rod offcuts 22 mm across and 32 mm long (about 17 g each), painted in ISO 10625 colours: six yellow, two blue, one
lilac, two red, one green and one orange (the sim's sRGB values are in `layout.py`). Real nozzle tips are smaller
and lock into their bodies under a bayonet cap; for a check of this end state the cylinders standing in open
sockets give the same photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
