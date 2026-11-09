# Hand-place three SMD parts on their pads to IPC Class 3, tantalum band on the + pad

Rework and prototype technicians hand-place surface-mount parts on a circuit board's pads, and the result is judged
by IPC-A-610 (Acceptability of Electronic Assemblies) from a look straight down: is each part on its own footprint,
does a termination stick out past the outer end of its pad (end overhang, never allowed), and how far does it stick
out past the side of its pad (side overhang: at most half the part's width for Class 1 and 2 products, a quarter of it
for Class 3, high-reliability work). A polarised part placed the wrong way round is a defect in every class; a
reversed tantalum capacitor fails short when the board is powered. Here, at 8 times real size, a green board on the
midline carries three footprints (pairs of silver pads): one for a 2512 chip resistor, one for a 1210 ceramic chip
capacitor and one for a D-case (7343) moulded tantalum capacitor, with a white + mark beside the tantalum's + pad. The
resistor and ceramic capacitor lie on an antistatic mat on the left, the tantalum on a mat on the right. Each part must
go flat and square on the pads that match its size, with no end overhang, the tantalum's dark band on the + pad, and
is scored by its side overhang: 1/3 a part for Class 3, 1/6 for Classes 1-2 only. The pads locate nothing (there is
no paste and no pocket), so the placement accuracy itself is what is measured.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three parts' poses, with a tracker of each part's last half second |
| `mechkit.py` | Builds each part as one loose rigid body of several boxes (body, end caps, plates, band) and reports its pose |
| `tests/test_scoring.py` | Offline tests of every tier and band edge, every zero condition, rest, exploits, resets and the hooks (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no silver ends and no band.
`mechkit.py` (imported by `task.py`) builds each part from three or four boxes (the resistor and ceramic capacitor: a
body between two silver end caps; the tantalum: an amber body on two silver plates under its ends, with a dark band
across its top at the + end) as one loose rigid body through a hook on `yam_arena.scene`, and reports its pose (base
frame origin and orientation) to the scorer. It changes nothing else the embodiment does, and it stops the run with
an error if a part was not built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a
`sys.meta_path` hook patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner)
follows the one first shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent
implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, taking turns at the board): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Place the three parts on the circuit board, each on the pair of silver pads that matches its size (pads a little wider than the part and reaching just past both of its ends): square to its pads and centred on them, with neither end sticking out past a pad. The tantalum capacitor, the largest part, goes with its dark band on the + pad, the one with the small white square (the + mark) beside it." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/smd-hand-place-ipc-class3/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Place the three parts on the circuit board, each on the pair of silver pads that matches its size (pads a little wider than the part and reaching just past both of its ends): square to its pads and centred on them, with neither end sticking out past a pad. The tantalum capacitor, the largest part, goes with its dark band on the + pad, the one with the small white square (the + mark) beside it." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/smd-hand-place-ipc-class3/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`sixths`, `zero_reasons`, and per part `credit_sixths`,
`tier`, `zero_reason`, `at_rest`, `moved_mm`, `along_mm`, `across_mm`, `tilt_deg`, `axis_err_deg`, `low_gap_mm`,
`end_overhang_mm`, `side_overhang_mm` (+ end, - end), `side_pct_w`, the tantalum's `band_toward_plus`; then each
part's `<part>_origin`, `<part>_yaw_deg`, `<part>_offset_mm` from its footprint's centre and `<part>_source`).
`MECHKIT_TRACE=<file>` adds mechkit's own per-step record. Videos: `inspect-robots video <log>.json`. The scorer
keeps one trial's record per process and starts it over at every reset (mechkit's reset counter, or the parts put
back on their marks in one step); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the three parts' poses, once per
control step (20 Hz), and the score is read at the last step. Each part's outline is the shadow of its box seen from
straight above (its eight corners, turned and tilted as it lies), which is what a top photo with a ruler measures.
Its terminations are the last 8 mm of the resistor's and the tantalum's length at each end and the last 6 mm of the
ceramic capacitor's, over the full width. All three footprints lie along x; every pad pair is 2 mm wider than its
own part on each side and reaches 6 mm past its ends, which is how "the pads that match its size" is defined (the
tests check that no other pair fits any part that way).

A part counts only if, at rest (neither end moved more than 2 mm in the last 0.5 s, read from positions because a
gripped part's reported velocity chatters), it lies flat (tilted 5 degrees or less from lying on a face: the resistor
on either broad face; the ceramic capacitor, square in section with its end caps round all four long faces, on any of
them, since it is the same part whichever is down; the tantalum band face up) with its lowest point within 2 mm of the
pads' top, its long axis within 10 degrees of x, no point of its outline past either pad's outer end, and, for the
tantalum, its band at the + pad's end. Its side overhang is how far a termination's outline reaches past the side of
its pad, at the worse of its two terminations (IPC takes a share of the part's width or of the pad's, whichever is
less; these pads are wider than their parts). The resistor's top is unmarked (black all over), so either broad face
up counts; a real chip resistor goes element up (see `fidelity`).

| Outcome | Score |
|---|---|
| At the end, any part off the table (lowest point more than 5 cm below the table top) | 0 for the trial |
| The tantalum on its own pads reversed: flat, band face up, lowest point within 2 mm of the pads' top, axis within 10 degrees of x, centre over its pads, band at the - pad | 0 for the trial |
| Otherwise, per part that counts: side overhang at most 25% of its width (Class 3: R1 6.25 mm, C1 5 mm, T1 8.5 mm) | 1/3 |
| Per part that counts: side overhang over 25% and at most 50% of its width (Classes 1-2: 12.5, 10, 17 mm) | 1/6 |
| Per part: side overhang over 50%; end overhang; on another part's footprint; on top of another part; tilted, the resistor or tantalum on its side, any part on an end, the tantalum upside down; more than 2 mm above its pads; still moving; off the board | 0 |
| Total | the sum: 0, 0.17, 0.33, 0.5, 0.67, 0.83 or 1.0 |

Square and centred along its axis, a part reaches the Class 3 and Class 2 limits when its centre is 8.25 and 14.5 mm
(R1), 7 and 12 mm (C1) or 10.5 and 19 mm (T1) to one side of its footprint's centre line, and it may be 6 mm off
along the axis before an end overhangs. A turn adds to the side overhang at the termination corners, so a resistor
5 mm off to the side is Class 3 square and Class 2 turned 8 degrees. Every total is reachable: each part
independently scores 0, 1/6 or 1/3 (the tests play all 27 combinations). Success (which ends the trial) is 1.0,
all three Class 3, held for 2.5 s; a trial that never reaches it is scored at its last step like any other.

Unintended strategies earn nothing they should not, as the tests show: a part on another part's pads earns nothing
(the resistor fits the tantalum's pads with no overhang, and the tantalum fits the resistor's with 2.5 mm a side,
both 0); a part stacked on another, held 4 mm above its pads, leaning on a neighbour, the resistor on its side, any part on an end or across its
footprint is 0; a reversed tantalum dragged just off its pads avoids the trial zero but earns nothing itself, so it is
never better than leaving it on the mat; pushing a part along the board onto its pads is scored by where it ends, like
any placement (a technician nudges parts with tweezers too); a sweep of resistor poses around its footprint finds no
pose that earns credit while breaking a rule. The scorer cannot see the grippers, so a part held still on its pads
counts as placed where it is held (see `fidelity`); the 2.5 s hold makes the demo's trial end with the fingers open.
A simulator state with a non-finite number, a zero quaternion or a missing part scores 0 for that step and leaves the
scorer's record as it was.

## One arm alone

The demo uses both arms because the parts start on both sides: the left arm places the resistor and the ceramic
capacitor from the left mat, the right arm the tantalum from the right mat, taking turns at the board (the footprints
are only 65 mm apart). The placement itself is one-handed work. Scripted with one arm idle on the GPU, each arm's
own parts score and the others stay on their mat: `DEMO_IDLE_ARM=right` (left arm alone: resistor and ceramic
capacitor, both Class 3) 0.67 and `DEMO_IDLE_ARM=left` (right arm alone: the tantalum, Class 3) 0.33, as the
offline kinematic check predicted.

No claim is made that the task needs two arms. Straight down at grip height, the left arm's tool point reaches y -16
cm at x 20 cm (the tantalum's near end is at y -21.1 cm) and the right arm's y +17 cm (the resistor's near end is at
y +19 cm), by inverse kinematics of the arm's URDF with its joint limits. But leaned about 20 degrees, either arm
reaches the middle of the other mat with its links clear of its own base and the table (same model), so a single arm
able to grip at a lean could place all three parts and score 1.0.

## Demo and run limit

The scripted demo takes 45.6 s (912 control steps). The right arm goes first: it comes down over the tantalum at yaw
+90 degrees (fingers closing along x, across its 34 mm width), lowers the last 30 mm over 1.5 s with its pad tips
9 mm above the part's underside, grips it with the fingers commanded 8 mm narrower over 1.5 s, lifts it to 10 cm,
carries it over its footprint while turning to yaw 0 (the quarter turn swings the band from +y to +x, onto the + pad),
lowers the last 30 mm over 2 s to 1 mm above the pads, opens over 1.5 s only to the part's width plus 16 mm (so the
fingers stay clear of the neighbouring footprints) and goes up and aside. Meanwhile the left arm waits over the
resistor; once the right arm is back up, it does the same with the resistor (yaw -90 to 0) and then the ceramic
capacitor, and goes aside. The closest the two tool points come is 26.9 cm and their links 10 cm (capsule model).
Success has held for its 2.5 s at step 875 (t 43.75 s), with the last part released and the gripper going up, both
offline and in the GPU run; the scripted policy would stop after 973 steps. In the GPU run every part ended within
1.5 mm of its footprint's centre, turned under 1 degree and flat, against 6 mm of play along each footprint and
7-10.5 mm across it for Class 3. To retune, read `<part>_offset_mm` in `TASK_TRACE` and move that part's
`PLACE_TRIM` in `task.py` by minus the offset. The run limit is 1,800 steps (90 s).

The offline checker warns `proxy_drop ... centre not over its support` for each part: its kinematic proxy drops a
part onto a single supporting box, while each part in fact bridges both of its pads (the tests check that both pads
lie under every scored position).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the resistor and ceramic capacitor back on the
left mat, centred at x 20.0 cm, y +21.5 cm and y +26.5 cm, and the tantalum on the right mat at x 20.0 cm, y -24.0
cm, each flat with its long side along y and the tantalum's band toward +y (to the left). The board, pads, + mark and
mats are fixed. Nothing is consumed.

## Runtime per trial

On an L40S, once Isaac Sim is running: about 2 minutes for the scripted demo (875 steps) and about 4 minutes for a
full 1,800-step Dream-1 trial. The first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 150 x 220 mm green plate 4 mm thick (glass-epoxy board or painted plywood) clamped to the bench, with six 1 mm
aluminium plates glued on as the pads (18 x 29, 16 x 24 and 18 x 38 mm, in pairs at the stated spacings) and a white
8 x 8 x 1 mm plastic square glued beside the tantalum's far pad; two 90 x 130 x 3 mm grey rubber mats. The parts are solid plastic
blocks (acrylic, PVC or 3D-printed, about 1.1-1.45 g/cm3): 50 x 25 x 20 mm black with 8 mm of aluminium foil tape round each
end (30 g), 26 x 20 x 20 mm tan with 6 mm of the same tape round each end (15 g), and 58 x 34 x 22 mm amber on two 8 x 34 x 1 mm aluminium
strips under its ends, with a dark brown band 10 mm wide and 1 mm thick across the top at one end (24 mm tall in all,
50 g). Nothing is powered and
nothing is hot, so a trial is safe. Score from a photo taken straight down with a ruler on the board, or measure each
part's offsets from the pad edges with a steel rule or calipers.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.33 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.67 |
| Dream-1 | 0.0 |

The demo's success held from step 875 (all three parts Class 3, side overhang 0, the tantalum's band on the + pad).
In a settling check (`DEMO_HOLD_ONLY=1`, 160 steps) no part moved on its mat. Dream-1 ran all 1,800 steps and left
every part on its mat (0.0).
