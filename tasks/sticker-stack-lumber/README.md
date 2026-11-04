# Sticker a two-layer lumber stack with the stickers over the bearers

Lumber is air-dried in stacks: boards lie across bearers in layers with gaps between them, and thin strips called
stickers separate the layers so air can flow through. Every sticker must sit directly above a bearer, so the
weight goes straight down to it; a sticker off its bearer leaves the boards over it bending between supports, and
they dry bowed. Here two fixed bearers stand across the middle of the table, and each arm has two red-cedar boards and
one sticker on its own side. The arms build the stack together: two boards across the bearers with a 6 cm air gap,
a sticker on them right over each bearer (turned a quarter turn from how it lay), and two boards on the stickers
right above the first two. Each arm works on its own side and they take turns at the stack: one places while the
other fetches its next piece. Neither arm can reach the other side's sticker, so the task needs both: one arm
alone can at best lay the first layer.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the boards' and stickers' states, with a small tracker for the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay two of the pale boards across the two fixed bearers with a gap of about 6 cm between them, lay a dark sticker across those boards directly over each bearer, then lay the other two boards on the stickers, each right above one of the first two." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/sticker-stack/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay two of the pale boards across the two fixed bearers with a gap of about 6 cm between them, lay a dark sticker across those boards directly over each bearer, then lay the other two boards on the stickers, each right above one of the first two." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/sticker-stack/dream
```

Run one trial per process (the default). The tracker lives in the task module and recognises a new episode in the
same process only when the simulator puts every piece back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`layer1`, `layer1_gap_mm`, `sticker_off_mm`,
`sticker_bearer`, `layer2_off_mm`, `max_tilt_deg`, `first_step`, and each piece's centre, tilt, heading error and
speed). A trace file that cannot be written is skipped. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score depends only on where the pieces lie at the
last step (the bearers are fixed, so their tops come from `layout.py`). Any board may go in any place and either
sticker over either bearer, and every piece may lie either broad face up and either way round. 'Rests on' is
computed from the pieces' real outlines and faces, not their bounding boxes: seen from above the underside must
overlap the support's top by at least half of a square crossing of their widths, and be within 3 mm of it there.

| Outcome | Score |
|---|---|
| A board or sticker off the table, or standing on an edge or end (tilted over 45 degrees; a sticker rolled onto its 16 mm side counts) | 0 |
| Layer 1 (two boards on both bearers, 30-90 mm apart), stickers on it within 10 mm of different bearers, layer 2 on the stickers within 15 mm of layer 1 | 1.0 |
| The same with the stickers within 25 mm and layer 2 within 30 mm, not all within the 1.0 limits | 0.75 |
| Layer 1 and both stickers within 10 mm; layer 2 missing, held above, turned, on one sticker or further off | 0.5 |
| Layer 1 only | 0.25 |
| Anything else (no two-board layer on the bearers with a 30-90 mm gap) | 0 |

The trial ends early (success) once the 1.0 conditions have held with every piece still (under 1 cm/s and
0.5 rad/s) for 3 s. That is as long as settling, letting go of the last board and lifting the hand (0.5 + 1.5 + 1.0
s), and the board only counts as still from the last moments of its slow descent, so the trial cannot end while the
fingers are still closed on it (they are fully open 2 s after the descent ends). Otherwise it is scored as it stands
at the run limit. An observation with NaN, inf or a zero quaternion in it scores 0 for that step and changes
nothing.

Unintended strategies earn nothing extra: stickers laid straight on the bearers with the boards on them leave no
board on the bearers (0); all four boards side by side on the bearers, or three, is not a two-board layer (0);
two layers without stickers, or stickers laid along the boards, stop at layer 1 (0.25); both stickers over one
bearer stop at layer 1 (0.25); a top board piled on the other top board, or both top boards over one lower board,
stop at 0.5. Knocking a finished stack apart at the end scores what is left.

## One arm alone

The pieces on the other side are out of reach: straight down an arm reaches only 10 cm past the midline (the other
side's boards are centred 38 and 46 cm from it, its sticker 54 cm), and the other side's sticker stays at least 10
cm beyond the TCP in any pose (position-only IK; it lies 85 cm across the table from that arm's base). Tilted about
50 degrees, with the gripper a few mm off the table, an arm can just reach the back end of the other side's inner
board, but never the second sticker, so one arm alone can at best lay layer 1 from its own two boards and add its
one sticker: 0.25 (short of raking the far sticker over with a held board). `DEMO_IDLE_ARM=right` (the left arm
alone lays its inner board on the bearers, its sticker on that single board and its outer board on the sticker)
scores 0, and so does `DEMO_IDLE_ARM=left`: one board on the bearers is not a layer. (Offline kinematic predictions;
the GPU runs are recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 58.4 s (1,168 control steps). Each arm grips its boards across their 40 mm width at the
centre (fingers commanded to 32 mm, pad tips 4 mm above the board's underside) and its sticker across its 20 mm
width (commanded 12 mm) 10 cm from its centre, toward the arm's back; picked with the fingers closing along y, the
sticker is turned a quarter turn on the way and set down along y (left arm yaw -90 degrees, right arm +90). Every
final descent is a separate 1.5 s move over the last 3 cm, every piece is let go with its underside 2 mm above its
support after a 0.5 s settle, and every gripper open or close takes 1.5 s. The payload is not gravity-compensated:
by our stiffness estimate the arm sags about 1.4 mm under a board and about 1.8 mm under a sticker (its far end
about 2.6 mm, as it is held 10 cm off centre), so each piece meets its support at about the end of the descent and
drops less than 1 mm when let go; the 3 mm 'rests on' tolerance is not used up by sag. The order:

1. Both arms pick their inner board at once. The left lays it on the bearers at y +5 cm; as it moves out, the right
   carries its board in and lays it at y -5 cm.
2. Meanwhile the left arm picks its sticker. Once the right arm has left the stack for its own sticker, the left
   lays its sticker across layer 1 over the far bearer (x 36 cm).
3. The left arm goes for its outer board; 0.5 s after it has left the stack the right arm lays its sticker over the
   near bearer (x 20 cm).
4. As the right arm leaves for its outer board, the left lays its outer board on the stickers at y +5 cm and moves
   out; the right lays the last board at y -5 cm and moves out.

Only one arm is ever over the stack: the TCPs stay at least 30 cm apart and the arms' collision capsules at least
11.9 cm (offline). In the offline proxy success is held at step 1,161; the scripted policy stops at step 1,229.
The run limit is 2,400 steps (120 s), the program's common limit, which leaves Dream-1 about twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the four boards back flat along x, centred
at x 28.0 cm and y +38.0, +46.0, -38.0 and -46.0 cm; the stickers back flat along x, centred at x 38.0 cm,
y +54.0 and -54.0 cm. The bearers do not move.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 58-61 s of simulated time, a Dream-1
run up to 120 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 20 x 20 x 320 mm weathered (silver-grey) timber battens screwed or clamped to the bench as bearers, with a
centre-line mark on each end; four 240 x 40 x 16 mm red-cedar boards (about 60 g each); two 280 x 20 x 16 mm poplar
strips stained dark (about 40 g each) as stickers. Tape the starting marks. The offsets and the gap are read from a
top photo with a ruler against the bearers' centre-line marks; a side photo shows whether every piece rests on its
supports.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
