# Load the press, run the cycle with both hands on the run buttons, unload

A two-hand control device starts a hazardous machine cycle (a press, shear or bender) only while both run buttons
are held at once, so that both hands are on the buttons and neither is in the die. The buttons are spaced so one
hand cannot reach both, the cycle starts only when both are pressed within a short time of each other (synchronous
actuation) and runs only while both stay held. Here a small press nest (four steel rails round a 36 x 36 mm floor)
stands on the midline with a press frame behind it, a blue aluminium block waits on a feed mark to the left, two
yellow run buttons stand at the far left and far right of the bench (each in reach of one arm only), and a grey
output tray sits in front of the nest. One arm loads the part into the nest, both arms press their own button within
0.5 s of each other and hold for 2 s, then the part is unloaded into the tray. On no control step may a button be
pressed while any gripper's fingers are in the die space. The buttons, not the physics, make the job two-handed:
one arm alone can load the part and press one button, never run the cycle. The scorer watches both grippers and the
part at every control step, not only the end state. The press is an unpowered bench mock-up: the ram is not modelled
and nothing moves when the cycle runs.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the part's state and the grippers' state, with a tracker for the whole trial (press spans, the double press, the cycle, breaches) |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook, the exploits we tried, and the demo plan played through the scorer with a kinematic proxy (no simulator) |
| `tests/test_demo_plan.py` | Offline checks of the demo and the objects against the station's limits (masses, finger values, reach, TCP separation, timing); generic, reusable |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, two fixed cylinders, one loose box) and the standard `task_score`
scorer.

The scorer's gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; on those steps no press is registered (a running double press
is broken), no breach is recorded, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the blue block into the press nest between the rails, then run the press cycle: hold both yellow run buttons down at the same time, one with each gripper, pressing them within half a second of each other and keeping both held for two seconds. Never have a gripper near the nest while a button is pressed. Then let go of the buttons and move the block into the grey tray." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/two-hand-control-press-cycle/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the blue block into the press nest between the rails, then run the press cycle: hold both yellow run buttons down at the same time, one with each gripper, pressing them within half a second of each other and keeping both held for two seconds. Never have a gripper near the nest while a button is pressed. Then let go of the buttons and move the block into the grey tray." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/two-hand-control-press-cycle/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breach`, `breach_steps`, `first_breach_step`,
`breach_arms`, `in_die`, `die_gap_mm`, `pressed`, `released`, `tips_above_button_mm`, `tips_to_button_axis_mm`,
`press_s`, `double_pressed_now`, `double_now_s`, `double_now_sync`, `double_now_seated`, `double_gap_s`,
`double_presses`, `seated_presses`, `sync_presses`, `longest_double_s`, `cycled`, `cycle_step`, `part_seated`,
`part_in_tray`, `part_to_nest_mm`, `grippers_let_go`, `opening`, `arms_unavailable`, both TCPs, ...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`):
the 0.5 s window and the 2 s cycle are counted in steps (10 and 40).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (71 tests).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer cannot see which way a gripper is turned, so
each gripper's fingers count as a round zone about its axis, out to the fingers' outer corners (57 mm fully open,
23 mm closed on the 30 mm part, 21 mm fully closed), from the finger tips to 38 mm back up the gripper. The die space
is x 25 to 35 cm, y -5 to +5 cm, up to 20 cm above the table. Each button is scored by its own arm's gripper (they
are 42 cm apart, each out of the other arm's reach): 'pressed' is that gripper's finger-tip point within 16 mm of the
button's axis and between 5 mm below and 10 mm above its top with the fingers at most half open. One rule holds for
the whole trial: on every control step on which any point of either finger zone is inside the die space, no button
may be pressed. One step broken, at any moment, scores 0 whatever follows.

The cycle is stateful. Each button's press is a span of consecutive pressed steps. A double press begins on the
first step with both pressed and is synchronous if the two spans began within 10 steps (0.5 s) of each other. A
cycle is a synchronous double press that lasts 40 steps (2.0 s) without a break, the part seated (centre within
4 mm of the nest's centre, bottom within 3 mm of the table, tilt under 5 degrees) on every one of them. Once a cycle
has completed the part is cycled for good.

| Outcome | Score |
|---|---|
| A button pressed on any step with a gripper's fingers in the die space, or the part ends off the table | 0 |
| Cycled part resting in the tray, both buttons released (tips over 30 mm above their tops or 30 mm off their axes), both grippers let go of it, still 1 s | 1.0 |
| Cycled, but the part not resting in the tray (in the nest, held, on the table elsewhere, on a tray wall), or in the tray with a button still pressed (or neither pressed nor released) or a gripper still holding it | 0.75 |
| Part seated in the nest and a double press began with it seated, but no cycle completed (not synchronous; held under 2 s; the part unseated during it) | 0.5 |
| Part seated in the nest, no double press with it seated | 0.25 |
| Anything else (part on the feed mark, beside the nest, up on a rail, in the tray uncycled; nothing done) | 0 |

A gripper has let go of the part when it is open (opening 0.9 or more, fingers at least 84.5 mm apart) or its finger
zone is more than 10 mm from the part. A button has a 20 mm band between 'pressed' (tips up to 10 mm over its top)
and 'released' (over 30 mm): a gripper hovering in that band at the end is not released (0.75), but it is also not
pressing, so a gripper in the die space at that moment is no breach, and a press span ends when the tips enter the
band.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with the part still
for 1 s. Success needs the unloading gripper to have let go, so the trial cannot end while it still holds the part.
A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts the part back on
its feed mark (within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a
non-finite number scores 0 for that step and leaves the record of the trial as it was. The part is a cube, so any
face may be down, and a quarter turn in the square nest is still seated.

Unintended strategies earn nothing, as the tests show: carrying the part straight from the feed mark to the tray
(0: the tray tier needs the cycle); doing nothing (0); one button only, held however long (0.25); both buttons
pressed 1.0 s apart, or 0.55 s apart (0.5: not synchronous), 0.5 s apart (synchronous); both pressed but held
1.5 s, or 1.95 s (0.5), 2.0 s (cycle); pressing both buttons with the part still on the feed mark and loading it
afterwards (0.25: an empty cycle counts for nothing, the double press must begin with the part seated); the part
nudged 5 mm during the double press (0.5: the cycle needs it seated throughout; a clean second double press then
completes it); the left pressing while the right is still over the nest after loading (0, breach the moment the
left's tips reach the pressed band); the left holding its button down from the start while the right loads (0,
breach when the right's fingers enter the die space, even though the cycle afterwards is fine); unloading while a
button is still pressed (0); a button pressed for a single step while the other gripper is in the die space (0);
the left letting its button up for one step in the middle of the hold (two spans, neither 2 s, the second not
synchronous: 0.5); pressing with open fingers, or 20 mm off the button's axis (not a press); knocking the part
toward the nest (it stops against the 8 mm rails, 26 mm from the nest's centre: never seated; on a rail it is 8 mm
up and 18 mm off: not seated); the left gripper somehow over the right button (not counted, and unreachable
anyway); the right arm alone (loads, presses one button: 0.25 up to then; carrying on to the tray uncycled: 0); the
left arm alone (0.25 at best); sweeping over the nest 21 cm up with a button pressed (outside the die space: no
breach, but nothing there can be touched from that height); the part knocked off the table (0). Letting the part
go 2 cm up over the tray is not an exploit: landing flat in the tray and resting is unloading.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) loads the part, presses its own button for 3 s, and goes on to unload the
uncycled part to the tray: 0.25 while the part sits in the nest with one button pressed, 0 at the end (the uncycled
part in the tray earns nothing). `DEMO_IDLE_ARM=right` (the left arm alone) hovers over its button, presses it for
2.5 s and lifts off; the part never leaves the feed mark: 0. The best a single arm can do is 0.25: the left can reach
the feed mark and load the part itself and press its own button (tested), the right the same, but neither can reach
the other's button (y +21 cm is past the right arm's reach at about +17 cm, y -21 cm past the left's at about
-18 cm), and a button pressed while the same gripper is anywhere else is impossible. (Offline kinematic predictions;
the GPU runs are to be recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 38.9 s (778 control steps); both arms grip or press straight down at yaw 0 (fingers
closing along y). The part is held by its upper 18 mm throughout (finger tips 12 mm above its bottom, 22 mm pad
faces), because the pads closed on the 30 mm part stand 8 mm proud of its faces, 1 mm outside the rails' inner
faces, so a part held low could not be set down between the rails without the pads landing on them.

The right arm goes over the part on the feed mark 12 cm up, down to 3 cm above the grip, slowly down (tips 12 mm
above the feed mark), settles, grips the part (fingers commanded to 22 mm) over 1.5 s, settles, lifts it straight
up, carries it over the nest (its fingers in the die space from 8.3 s; the left is at its ready pose, no button
pressed), lowers to 2 cm above the let-go height, settles, lowers the last 2 cm over 1.5 s to the part's bottom
2 mm above the nest's floor (the pad tips 6 mm above the rails), settles, opens over 1.5 s (the part drops 2 mm
between the rails, which leave it 3 mm each way inside the 4 mm seated tolerance), and goes up and out to its run
button at x 20 cm, y -21 cm, 10 cm up (its fingers out of the die space from 16.0 s), where it closes its fingers
(to the 16 mm minimum gap) over 1.5 s, done at 18.2 s. The left arm waits at its ready pose until 9.2 s (while the
right picks from the feed mark at y +12 cm, 12 cm from the left button; the TCPs are never closer than 25 cm in y
when within 5 cm in x), goes over its own button 10 cm up, closes its fingers, and waits there until 18.2 s. Then
both arms make the same press moves on the same clock: down to 2 cm above the button over 1.0 s, the last 2 cm over
1.5 s to a commanded tip height of 2 mm over the button's top ('pressed' allows 5 mm below to 10 mm above, which
covers the unloaded arm's sag either way). In the offline proxy both tip points enter the pressed band on the same
step, 20.1 s (gap 0.0 s, synchronous), with the part seated. The left holds 2.5 s and lifts off (no longer pressed
from 23.35 s, released from 23.65 s); the right holds 3.0 s (the double press lasts 3.25 s, the cycle completes at
22.1 s, step 442) and lifts off 0.5 s after the left (released from 24.15 s). The left then steps 4 cm further left
and stays. Only when its own release is complete does the right go back over the nest (fingers in the die space from
25.6 s, no button pressed since 23.85 s: a 1.75 s margin), opening on the way, lowers to the tips 4 mm above the
rails, settles, grips the part's upper 18 mm, settles, lifts it straight up out of the nest, carries it over the tray
at x 16 cm (out of the die space from 32.2 s), lowers to 2 cm above the set-down, settles, lowers the last 2 cm over
1.5 s, settles, opens over 1.5 s with the part's bottom 2 mm above the tray's floor, and goes up. In the offline
kinematic proxy the 1.0 conditions hold from about step 754 and success (1 s still) is reached at step 774, before
the plan ends at 778; with the 3 s settle the demo needs 838 steps of the 1,200 run limit (60 s).

Timing margins: the only moments a button is pressed are 20.1 to 23.85 s, during which the nearest finger zone is
14.7 cm from the die space (the arms are on their buttons, 21 cm off the midline). The right leaves its button for
the nest 1.75 s after the last press ends, and 0.5 s after the left's release move is complete. The two grasps carry
0.07 kg at 24 and 30 cm reach: the sag is about 2 mm, well inside the 3 mm set-down tolerances (the part is let go
2 mm up in both places) and the 4 mm seated tolerance is guaranteed by the rails, not the arm. `TASK_TRACE` reports
`tips_above_button_mm`, `double_gap_s`, `longest_double_s`, `die_gap_mm` and `breach_steps` so a GPU run shows how
much margin was really there.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the part back flat on the feed mark centred
at x 24.0 cm, y +12.0 cm, sides parallel to x and y; the rails, frame, feed mark, buttons and tray are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 38.9 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Four 8 x 8 mm steel bar offcuts (two of 52 mm, two of 36 mm) glued or screwed to the bench in a square round a
36 x 36 mm floor centred 30 cm in front of the arm-base line; a 10 x 160 x 120 mm steel or aluminium plate stood on
edge 34 mm behind the back rail for the frame (no ram); a 30 mm anodised aluminium cube (about 70 g) for the part; a
40 x 40 mm white card, 2 mm thick, taped at x 24 cm, y +12 cm for the feed mark; two 40 mm diameter, 20 mm tall
yellow dome or mushroom run buttons fixed at x 20 cm, y +21 cm and y -21 cm, each wired to its own lamp and both in
series to a third lamp (the 'cycle' lamp, lit only while both are closed); a grey plastic tray with 15 mm walls
round an 80 x 140 mm floor (or a printed rim, open at the bottom) fixed at x 16 cm on the midline. Tape the die
space's 10 x 10 cm outline on the bench round the nest and stand a card beside it with a 20 cm mark, and film from
above and from the side at 20 frames a second: a frame with a button lamp lit and any finger inside the outline
below the mark is a breach; the two button lamps lighting more than 10 frames apart is not synchronous; the cycle
lamp lit for 40 consecutive frames with the part between the rails is a cycle.

## Results in simulation

To be measured on the GPU with the commands above (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`,
one trial each). The offline predictions are from the kinematic proxy in `tests/test_scoring.py`, not measurements.

| Run | `task_score` | Offline prediction |
|---|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU | 0.0 (0.25 until it unloads the uncycled part) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU | 0.0 (one button pressed, part never loaded) |
| Dream-1 | to be measured on the GPU | |
