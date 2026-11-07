# Block both lenses for edging, each block 8 mm temporal of the optical centre

Ophthalmic laboratory technicians (O*NET 51-9083.00) lay out and block every spectacle lens before it is edged to
the frame's shape. A lensmeter marks the lens's optical centre (OC) with a dot and its horizontal, the 180 line,
through it. Then a block, which the edger's chuck holds, is stuck on the lens. Many labs block at the frame's boxing
centre, so the edger needs no decentration entered: that point lies (A + DBL - PD) / 2 temporal of the OC along the
180 line, and the block's locating flat must lie along the line, or the edged shape (and a cylinder's axis with it)
comes out turned. For this job (eye size A 54 mm, bridge DBL 20 mm, PD 58 mm) that is (54 + 20 - 58) / 2 = 8 mm.
Blocking on the OC out of habit, with a narrow PD in a wide frame, is the classic remake: by Prentice's rule a
-4.00 D lens 8 mm off carries 3.2 prism dioptres, where ANSI Z80.1 allows about 0.67 of horizontal imbalance.

Here two uncut pale blue lenses lie fixed on white layout pads in front of the two arms, each inked with a black OC
dot, a black 60 mm line through it, turned 25 degrees (left lens) or 30 degrees (right lens) on its pad, and a small
black square at the line's nasal end, toward the midline. Two blue edger blocks (18 x 24 x 16 mm, the 24 mm side
standing for the locating flat) stand on the table in front of the arms. The instruction gives the job's numbers
and the arithmetic (8 mm). The robot must find each lens's temporal side (away from the square), measure 8 mm from
the dot's centre along that lens's own turned line, set one block flat there with its long side turned onto the
line, and let go. The scorer measures each block's centre to its target point and its long side's angle to the
line.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the blocks' states and the grippers' tool points, with a small tracker |
| `rigstate.py` | Reads both grippers' tool points and openings each control step for the let-go rule |
| `tests/test_scoring.py` | Offline tests of every score, every rule and its edge, and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and cylinders, two loose boxes) and the standard `task_score`
scorer. `rigstate.py` wraps the embodiment's step and reset from the task file's import to read the two grippers'
tool points and openings; it changes nothing the embodiment does.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, each blocking the lens in front of it): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Block both lenses for edging at the frame's boxing centre. For this job (eye size 54 mm, bridge 20 mm, PD 58 mm) that is (54 + 20 - 58) / 2 = 8 mm temporal of the optical centre. On each lens, set one blue block flat with its centre 8 mm from the centre of the black dot (the optical centre) along the black ink line, on the side away from the small black square (the nasal mark), with the block's long side along the line. Then let go of both blocks." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir ~/yam-eval/lens-pair-blocking/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Block both lenses for edging at the frame's boxing centre. For this job (eye size 54 mm, bridge 20 mm, PD 58 mm) that is (54 + 20 - 58) / 2 = 8 mm temporal of the optical centre. On each lens, set one blue block flat with its centre 8 mm from the centre of the black dot (the optical centre) along the black ink line, on the side away from the small black square (the nasal mark), with the block's long side along the line. Then let go of both blocks." \
  --scorer task_score --max-steps 1400 --store-frames --log-dir ~/yam-eval/lens-pair-blocking/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: the score and, per lens (`lenses.left`, `lenses.right`),
`n_on` (blocks on it), `block`, `d_mm` (centre to target), `a_deg` (long side to the line), `oc_mm`, `tilt_deg`,
`low_mm` (lowest corner above the lens top), `path_mm` and `turn_deg` (over the last 0.5 s), `tcp_mm` (the nearer
tool point), `still`, `held`, `let_go`, `lens_score` and why it failed; also both tool points and openings, both
blocks' poses and `arms_unavailable`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the two blocks' poses and the two
grippers' tool points (TCP, 13 mm above the finger tips) and openings once per control step (20 Hz) and keeps only
each block's poses over the last 0.5 s. The same rule applies to each lens. A block is **on** a lens when its centre,
seen from above, is within 35 mm of that lens's OC and less than 60 mm above the lens top. A lens is **blocked** when
all of these hold:

- (a) exactly one block is on it;
- (b) the block is flat: its 16 mm axis within 3 degrees of vertical (either 18 x 24 mm face down);
- (c) it is seated: its lowest corner within 2 mm of the lens top;
- (d) it is still: over the last 0.5 s its centre moved at most 2 mm in all and it turned at most 2 degrees in all
  (from positions and orientations; a resting part's reported velocity chatters in the simulator, so velocities are
  never used);
- (e) it is let go: no gripper whose TCP is within 40 mm of the block's centre is less than half open (pads under
  47 mm apart). Half open, the pads are wider apart than the block's longest diagonal (34 mm), so a gripper that
  open cannot be holding it. At a step when the grippers cannot be read, no lens counts as let go (no credit without
  evidence; counted in the trace's `arms_unavailable`).

d is the horizontal distance from the block's centre to the lens's target, the point 8 mm from the OC along the ink
line on the side away from the nasal square: x 29.66 cm, y +24.73 cm (left) and x 29.60 cm, y -24.69 cm (right).
a is the angle seen from above between the block's 24 mm side and the line, 0 to 90 degrees (a block turned half a
turn, or upside down, is the same block).

| Each lens, at the last step | Lens score |
|---|---|
| Blocked, d at most 2 mm and a at most 3 degrees | 0.5 |
| Blocked, d at most 3 mm and a at most 6 degrees, but not 0.5 | 0.25 |
| No block on it, two blocks on it, or its block tilted over 3 degrees, on its side, not seated, still moving, still held, or further off | 0 |

| Both lenses | Score |
|---|---|
| Left lens score plus right lens score | 0, 0.25, 0.5, 0.75 or 1.0 |
| Both 0.5 (success; the trial ends once this has held 1 s, which needs both blocks let go) | 1.0 |
| A missing or non-finite block state (simulator blow-up) | 0 for that step; the tracker keeps its record |

Possible scores: 0, 0.25, 0.5, 0.75 and 1.0. Each lens is in exactly one of its three states, so every outcome maps
to exactly one score, and every sum is reachable (0.75 is one lens at 0.5 and the other at 0.25). The start scores
0. Either block may go on either lens (they are identical).

The task is built to catch the real errors, with these reference values (all in the tests):

| Block set | d or a | Lens score |
|---|---|---|
| Centred on the OC (the habit) | d = 8 mm | 0 |
| 8 mm on the nasal side | d = 16 mm | 0 |
| On the target but not turned (long side still along y) | a = 25 degrees (left), 30 degrees (right) | 0 |
| Turned, but 8 mm straight out along y instead of along the line | d = 3.5 mm (left), 4.1 mm (right) | 0 (both) |
| 8 mm from the dot's edge instead of its centre (12 mm or 4 mm from the OC) | d = 4 mm | 0 |
| Short side along the line | a = 90 degrees | 0 |

Unintended strategies earn nothing more, as the tests show: blocks perfectly placed but still gripped (or pressed down
with a closed gripper, its TCP within 40 mm) score 0 for that lens; a second block on a lens, side by side, stacked or
held up to 60 mm over it, makes that lens 0; a block knocked off onto the pad, tilted on the lens's edge or half on
the pad scores 0; a block still sliding or rocking counts only once its last 0.5 s are still (a creep of 2.1 mm or a
turn of 2.1 degrees in 0.5 s is not still, and a slower creep is scored where it ends); random poses never score
outside the five values. Pushing the lenses or pads does nothing: they are fixed. Sliding a block along the lens into
place is allowed: only the end state counts.

Real-trial reading: take a top photo with the grippers clear. A correct block hides the dot, but the 60 mm line is
centred on the OC, so its two visible ends locate the OC and the target. Measure d with a ruler and a with a
protractor against the line on the photo, check flatness with a square and seating with a 2 mm feeler gauge, and
judge stillness and let-go from the video's last second.

## One arm alone

The task does not need two arms, and it does not claim to. In a lab one technician blocks the two lenses in turn;
the two pads are a layout choice that lets the arms work in parallel. With the straight-down grips of the scripted
demo each arm blocks only its own lens: the other side's block starts 0.31 m past the midline and the other lens's
target is 0.247 m past it, beyond the about 0.10 m that a straight-down grip reaches across the midline.
In the simulated runs (see 'Results in simulation'), `DEMO_IDLE_ARM=right` (the left arm alone) blocked the left
lens on target and left the right lens with no block: 0.5; `DEMO_IDLE_ARM=left` (the right arm alone) blocked the
right lens and left the left lens empty: 0.5 too. The offline kinematic check predicted the same 0.5 for each arm
alone.

To be honest about a capable policy: our IK finds that a grip tilted about its closing axis (the block stays flat
between the fingers) lets one arm reach the far block at its start, beside the idle arm (5 mm of clearance to the
idle arm's collision capsules at a 30 degree tilt, 22 mm at 50 degrees, 51 mm at 70 degrees), and the far lens's
target at 40 to 70 degrees (28 to 117 mm of clearance; at 30 degrees it would hit the idle arm). So one arm could in
principle block both lenses (1.0) with a near-sideways grasp at full stretch. We have not demonstrated it.

## Demo and run limit

Both arms on one clock, mirror plans, 18.2 s (364 control steps). Each arm grips its block across the 18 mm side
with the fingers closing along x (yaw -90 degrees left, +90 degrees right, the signs that keep joint 6 inside its
limits), commanded to a 10 mm gap, 8 mm narrower than the block, so its 24 mm side lies along y between the pads.
Left arm (the right arm is its mirror image):

| Time (s) | Move | Fingers |
|---|---|---|
| 0-3.0 | ready pose to (16.0, 31.0) cm, TCP 10 cm up, yaw -90 degrees | open |
| 3.0-4.0 | down to TCP 4.5 cm | open |
| 4.0-5.5 | down to TCP 1.8 cm (pad tips 5 mm above the table, 11 mm of the block held) | open |
| 5.5-6.0 | settle | open |
| 6.0-7.5 | close | 10 mm gap |
| 7.5-9.0 | up to TCP 8 cm | closed |
| 9.0-11.5 | over the target (29.66, 24.73) cm plus the place trim, TCP at (29.72, 24.61) cm, turning to yaw -65.2 degrees (-65 plus the yaw trim; the block's long side to heading 115 degrees) | closed |
| 11.5-12.5 | down to TCP 6 cm | closed |
| 12.5-14.5 | down to TCP 3.11 cm, the block's bottom 1.0 mm above the marks | closed |
| 14.5-15.2 | settle | closed |
| 15.2-16.7 | open: the block drops 1 mm onto the lens | open |
| 16.7-18.2 | straight up to TCP 10 cm | open |

The right arm picks at (16.0, -31.0) cm with yaw +90 degrees and places over (29.60, -24.69) cm, the TCP at
(29.66, -24.55) cm, with yaw +59.4 degrees (+60 plus the yaw trim; the block's long side to heading 60 degrees).
Turning the gripper by the line's heading minus 90 degrees lays the block's long side on the line; the other yaw
choices put joint 6 at its limit. `PLACE_TRIM` and `PLACE_YAW_TRIM_DEG` in `task.py` are per-arm corrections from
the first simulated run, which left the blocks (-0.6, +1.2) mm (left) and (-0.9, -2.1) mm (right) off their
targets and 0.8 degrees short of (left) and 1.6 degrees past (right) their lines: about 0.9 mm of each offset is
where the fingers held the block, the rest the arm trailing its command at the place. The left position trim takes
back the whole offset, the right one 1.6 of its 2.3 mm. Held, both blocks sat 1.5 to 1.8 degrees past their lines;
opening the fingers turned them back by 2.2 degrees (left) and 0.2 degrees (right), so each yaw trim (-0.2 and -0.6
degrees) centres the result over that range rather than cancelling one sample. The arms' collision capsules stay
at least 36 cm apart. Offline and in the simulator alike, success (1.0) holds for its 1 s at step 337 (16.85 s,
once both grippers are past half open and the blocks have been still for 0.5 s), which ends the trial; in the
simulated run the blocks came to rest 0.03 mm (left) and 0.78 mm (right) from their targets, their long sides
1.0 degree off the lines, flat and seated. Without success the scripted policy stops at step 425. The run limit
is 1,400 steps (70 s at 20 Hz), 3.8 times the demo.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand both blocks back on the table on an
18 x 24 mm face with the 24 mm side along y, centred at x 16.0 cm, y +31.0 cm and x 16.0 cm, y -31.0 cm. The pads,
lenses and marks are fixed; nothing is consumed. The scorer starts over on every episode reset (rigstate's reset
hook; it also spots both blocks being put back on their marks in one step).

## Runtime per trial

Measured on an L40S, each run's start of Isaac Sim included: about 1 minute for the scripted demo (54 s; 337
control steps until success ends it, 17 s of simulated time) and about 3.5 minutes for a trial at the full 1,400
steps (70 s of simulated time; the Dream-1 run took 3 min 21 s). The first start of Isaac Sim on a new instance
takes several minutes more.

## Real-world build

- Lenses: two 70 mm uncut CR-39 lens blanks with a pale blue tint (or 4 mm clear acrylic discs), each taped flat to
  the centre of a white acrylic plate 120 x 120 x 8 mm, and the plates taped to the bench at the listed positions.
- Marks: drawn with a lensmeter marking pen or a fine permanent marker: an 8 mm dot at the centre, a 4 mm wide line
  60 mm long through it, turned 25 degrees (left) or 30 degrees (right) from the y direction, and a 10 mm square on
  the line 21 to 31 mm from the dot, toward the midline.
- Blocks: two blue acetal (or painted wood or printed plastic) blocks 18 x 24 x 16 mm, about 10 g.
- Nothing is powered, hot or sharp; the parts are light. Score with a ruler, a protractor and a 2 mm feeler gauge on
  a top photo, as above.

## How it differs from nearby tasks

The board (875 tasks when last checked) has nothing on optical-lab work, lens layout or blocking (searched lens,
edger, optical centre, lensmeter, blocking, spectacle, ophthalmic, decentration). The nearest by act: TK-56LF8F (set
a pin at the point equally far from three survey marks), TK-T3XFRM (clamp a chip carrier centred on a square and
turned to a pin-1 mark), TK-GQ6FSP (place ballast so a combined centre of mass lands on a mark) and TK-UJB387
(register a workpiece against a corner stop). Each asks for a different point under a different rule. Here the
practice is a lens technician's blocking for edging, and the measured end state is a part's centre a stated distance
along a turned ink line on the workpiece, measured from one mark, on the side a second mark identifies, with the
part's flat turned onto that line, on two lenses whose lines are turned differently.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
