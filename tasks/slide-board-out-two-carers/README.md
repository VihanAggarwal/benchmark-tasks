# Withdraw the slide board from under the patient: one hand steadies, the other pulls

After a lateral transfer onto a bed, the slide (transfer) board is withdrawn from under the patient. Safe patient
handling practice has two carers for this: one steadies the patient while the other withdraws the board, because
pulling the board with the patient's weight on it drags the patient along with it (shear on the skin, pressure
injury, a slide toward the bed's edge), and the patient is never lifted away or let drop. Here a weighted
skin-pink block stands in for the patient's torso, lying on a light-blue board on a white bed deck. The left arm
grips the block and takes it a few millimetres off the board; the right arm pulls the board out from under it along
the deck; the left arm lowers the block slowly onto the deck and lets go. The scorer watches the block at every
control step: moved more than 15 mm, raised more than 30 mm, tilted more than 10 degrees or let down faster than
0.15 m/s scores 0, whatever follows. The job needs both arms by physics (the board drags the block, which has the
simulator's friction of 1.2 on it, unless something holds the block) and by rule (a block lifted off and set aside
has moved).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the block's and the board's states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |
| `tests/test_demo_plan.py` | Offline checks of the scripted demo against the station's limits (masses, finger gaps, grip timing, reach, TCP spacing, run limit) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (one fixed and two loose boxes) and the standard `task_score` scorer.

The scorer's "let go" rule needs the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; "both grippers let go" is then taken as met on those steps,
and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the blue slide board out from under the pink patient block and leave it lying on the white bed deck, with the patient block resting flat on the deck where it is. Steady the block with the other gripper while you pull: it must not be dragged, pushed or carried more than 15 mm, lifted more than 3 cm, tipped, or dropped. Let go of both at the end." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-board-out-two-carers/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pull the blue slide board out from under the pink patient block and leave it lying on the white bed deck, with the patient block resting flat on the deck where it is. Steady the block with the other gripper while you pull: it must not be dragged, pushed or carried more than 15 mm, lifted more than 3 cm, tipped, or dropped. Let go of both at the end." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/slide-board-out-two-carers/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `moved_mm`,
`max_moved_mm`, `raised_mm`, `max_raised_mm`, `max_tilt_deg`, `max_descent_m_s`, `under_share`, `pulled_mm`,
`board_clear_mm`, `board_fully_out`, `board_on_deck`, `patient_flat_on_deck`, `opening`, `grippers_let_go`,
`arms_unavailable`, both TCPs, ...). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record
per process and starts it over at every reset, so run one trial per process or rely on the reset; the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (73 tests).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the block's and the board's poses and
velocities once per control step and keeps the worst of four things about the block over the trial: its centre's
largest horizontal distance from where it lay at the first step ("moved"), the greatest height of the centre of
its underside above the board's top ("raised"), its largest tilt, and its fastest descent (the larger of its
reported downward speed and how far its underside came down in one control step divided by the step, so a drop
that lands inside one step is still seen). Six rules hold for the whole trial; one broken at any moment scores 0
whatever follows:

1. the block's centre goes more than 15 mm from where it lay;
2. the centre of its underside goes more than 30 mm above the board's top;
3. it tilts more than 10 degrees;
4. it comes down faster than 0.15 m/s;
5. it leaves the table (lowest point more than 10 mm below the table top);
6. the board leaves the table.

How far the board is out is read from the two outlines seen from above (world-aligned bounding boxes): the share
of the board's starting footprint under the block (40 x 60 mm) that is still under it. The board's left end
starts 50 mm past the block, so a straight pull of 80 mm leaves half of that footprint under, 110 mm leaves none,
and 120 mm puts the outlines the 10 mm apart that "fully out" needs; the demo pulls 130 mm.

| Outcome | Score |
|---|---|
| Any rule broken at any moment; or at the end the block's centre is not over the deck | 0 |
| Board fully out (outlines 10 mm apart) and resting flat on the deck; block resting flat on the deck; both grippers let go | 1.0 |
| Board fully out and the block resting flat on the deck, but a gripper still holds, or the board is not resting flat on the deck (hanging over the deck's edge, tilted, held up) | 0.75 |
| No more than half of the board's starting footprint under the block is still under it (a straight pull of 80 mm or more) | 0.5 |
| The board pulled at least 20 mm along its length, more than half still under | 0.25 |
| Anything else (nothing pulled, the board pushed back under) | 0 |

"Resting flat on the deck": lowest point within 3 mm of the deck's top and the centre over the deck, tilted under
3 degrees for the board (either broad face up) and under 5 degrees for the block. A gripper has let go of the item
it handles (the left the block, the right the board) when it is open (opening 0.9 or more, fingers at least
84.5 mm apart) or its finger zone (a round zone about the gripper's axis out to the fingers' outer corners, 41 mm
closed on the block, 57 mm fully open, from the finger tips to 38 mm back) is more than 10 mm from that item, so a
gripper closed again once it is clear changes nothing, while one still closed round its item scores 0.75.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both objects
still (under 1 cm/s and 0.5 rad/s) for 3 s, longer than letting go of the block and lifting the hand clear
(1.5 s + 1.0 s), so it cannot end with the pads still beside the block. A new trial is recognised from rigstate's
reset hook, or, as a fallback, when the simulator puts both objects back on their marks (within 4.5 cm, which
covers a seeded reset's jitter) in a single step, having moved further than their velocity and a free fall could
carry them (a block dropped back onto the board is not mistaken for a reset). A simulator state with a non-finite
number or a zero quaternion scores 0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing, as the tests show: pulling the board with the block on it drags the block
along (130 mm with the full pull; 20 mm with a short one: rule 1, 0), and a 10 mm pull drags it under the limit
but leaves the board under the whole block (0); lifting the block off and setting it aside to pull the board
freely breaks rules 1 and 2 (0), and carrying it aside within the 30 mm raise limit still breaks rule 1 (0);
pushing the block off the board tips it over the board's edge (rules 1 and 3, 0); jerking the board out from under
it in one step (a tablecloth pull) drops it 20 mm to the deck (rule 4, 0); holding the block back with the open
fingers instead of taking its weight lets it tip over the board's end as the board passes under its centre
(rules 3 and 4, 0); holding the block up but never pulling, raising it and setting it back on the board, and doing
nothing all leave the board under the whole block (0); pulling the board out and leaving the block held in the
air is 0.5; setting it down tilted, or holding it 4 mm above the deck, is 0.5; leaving a gripper closed on the
block or the board is 0.75; pulling the board off the deck onto the table, or leaving it tilted on the deck's
edge, is 0.75. Two things are allowed that are not exploits: lifting the board out, rather than sliding it, once it
is clear of the held block, and sliding it out across (along x) instead of along its length, both 1.0.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone pulls the board) should score 0: with nothing holding it, the block
rides along on the board (its friction on the board, 1.2, holds it against the pull at the pull's 32 mm/s) and
goes about 130 mm, far beyond the 15 mm limit, with the board still under it. `DEMO_IDLE_ARM=right` (the left arm
alone raises the block, holds it, and sets it back down) should score 0: the board ends where it started, under the
whole block. A single arm cannot hold the block and pull the board at the same time, and the one-arm alternative,
lifting the block off and aside, breaks rules 1 and 2. Both arms reach both items (the block is at y +10 cm, the
board's grip point at y -12 cm, both 28 cm out), so the split rests on the physics and the rules, not on reach; the
"let go" rule is checked per gripper and item, so an assignment with the roles swapped is scored the same way.
(Offline predictions; the GPU runs are to be recorded in `task.json`'s `scores` and in the table below.)

## Demo and run limit

The scripted demo takes 20.0 s (400 control steps). The left arm comes down over the block with its open fingers
along x, outside the 40 mm board (the pads 47 mm from the block's centre, the board's edges 20 mm), the pad tips
4 mm above the block's underside (24 mm above the deck), grips it across its 70 mm (fingers commanded 62 mm, over
1.5 s), settles, and raises it 18 mm (commanded) straight up over 1.5 s, then holds it where it is. The right arm
has meanwhile gripped the board 20 mm from its right end across its 40 mm width (fingers commanded 32 mm, the pads
27 mm wide wholly on the board, the pad tips 4 mm above the deck); 0.5 s after the block is up (8.5 s) it pulls
the board 130 mm along -y at deck height over 4 s (32 mm/s), settles, lets go over 1.5 s and goes up. The board
then spans y -27..+5 cm: its left end 20 mm clear of the block (twice the 10 mm "fully out" needs), its right end
10 mm over the deck's edge, its centre 7 cm inside it. The grip points are 22 cm apart in y before the pull and
35 cm after. 0.5 s after the pull has settled (13.5 s) the left arm lowers the block to 1 cm above the deck over
1.5 s, settles, lowers the last 1 cm over 1.5 s (under 7 mm/s) aiming its underside 1 mm above the deck, settles,
lets go over 1.5 s and goes up. Every grip and release takes 1.5 s.

The arm's controller does not carry the payload's weight: the left arm sags about 5.6 mm (and about 1 degree of
tilt) under the 0.20 kg block at this reach (static estimate from the arm's joint stiffness, as in the program's
other tasks), so the block should come up about 12 mm clear of the board, under the 30 mm limit for any sag from 0
and still clear of the board for a sag up to about 16 mm. On the way down, with the sag the block already rests
on the deck before the slow descent ends (the arm then pressing lightly), with no drop when the pads open; with no
sag at all it settles the last 1 mm, which reaches 0.14 m/s at most, under the 0.15 m/s limit. `TASK_TRACE`
reports `max_raised_mm` (expected 6 to 18) and `max_descent_m_s`. Offline, the scorer first sees the left gripper
as let go at 18.6 s (opening 0.9), success holds for its 3 s at 21.6 s (step 432), and the scripted policy stops at
step 460. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the board back flat on the deck centred at
x 28.0 cm, y +2.0 cm, long side along y; the block back flat on the board centred at x 28.0 cm, y +10.0 cm, sides
parallel to x and y. The deck is fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 23 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 200 x 440 x 30 mm plywood platform painted white, clamped to the bench, as the bed deck; a 40 x 320 x 20 mm
hollow polymer bar (or a plastic box section with its ends capped, about 100 g) in light blue as the slide board;
and a 70 x 60 x 40 mm block weighted to 200 g, skin-pink, as the patient. Tape the block's outline and a second
outline 15 mm outside it on the deck, stand a card upright on the deck behind the block (on the side away from the
arms) with a mark 30 mm above the board's top, and film from the side and from above: the moved and raise limits
are read against the tape and the card, the tilt and the set-down against the side video (a 20 mm fall to the
deck takes 64 ms and is plain at 60 fps), and "fully out" and "half out" against the block's taped outline. The
real bed surface is a mattress, so a real trial uses a firm practice pad under the board.

## Results in simulation

To be measured on the GPU with the commands above (Isaac Lab-Arena, `-E object_jitter=0 -E control_hz=20`, one
trial each). The numbers below are offline predictions, not measurements:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline plan predicts 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (predicted 0.0: the block is dragged) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (predicted 0.0: the board stays under) |
| Dream-1 | to be measured on the GPU |

What the GPU run must confirm: that the held block comes up clear of the board (`max_raised_mm` between 6 and
18; if the arm sags more than 16 mm the board drags the block and `patient_moved` appears in `breaches`); that
the board does not slip in the right gripper during the pull (its friction on the deck, about 1.2 N, against the
pads' hold) and does not lift the block's edge as it passes under; that the block's set-down is seen as resting
(`patient_low_mm` within 3 mm) with `max_descent_m_s` under 0.15; that the blocking of the block in the one-arm
run is as predicted (dragged, not left behind); and that success is reached before step 460.
