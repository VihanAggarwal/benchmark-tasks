# Set a lens and screen on an optical bench for a sharp, same-size image

Physics teachers and school science technicians set up ray-optics demonstrations and class practicals on an optical
bench: a lamp box with an arrow slide as the object, a converging lens in a holder and a white screen, each on a
rider. The result taught at GCSE and A level and in introductory college physics: an object at twice the focal length
(2F) gives a real, inverted image of the same size at 2F on the far side, so the screen stands 4f from the object, the
shortest distance at which a lens forms a real image on a screen. Here a 560 mm bench is fixed along the table 30 cm
in front of the arm bases, with a lamp box on its left end showing a 30 mm arrow, and a scale of ticks beside it every
50 mm from the arrow. The lens (f = 100 mm) starts 50 mm from the arrow, inside its focal length (no real image), and
the screen 280 mm away, nearer than 4f (no lens position can focus on it), so both must move: the lens to 200 mm, the
screen to 400 mm, each left standing upright, square across the bench and on its centre line. Sharp at another size
scores 0.5, one element at its same-size mark 0.25, and either element knocked over or off the bench 0.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, and the lens law |
| `scoring.py` | The scorer: plain functions of the lens's and the screen's poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds the lens and the screen each as one rigid body of boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (bench, lamp box, slide, arrow, scale ticks), the standard `task_score` scorer, and one
helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and each element here is a holder on a rider:
the lens's holder frame and glass on a wider rider, the screen's board on a rider. `mechkit.py` (imported by
`task.py`) builds each as one loose rigid body of boxes (lens: rider, four frame bars and the glass; screen: rider,
white board and black back; no two overlap, so the mass spreads evenly) through a hook on `yam_arena.scene`, and
reports its pose (base-frame origin, the centre of the rider's underside, and orientation) to the scorer. It changes
nothing else the embodiment does, and it stops the run with an error if an element was not built as a compound body.
The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from
the task file's import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the converging lens (focal length 100 mm) and the white screen on the optical bench so that a sharp image of the lamp's arrow, the same size as the arrow, falls on the screen. Leave both standing upright on the bench, square across it and on its centre line, the screen's white face toward the lamp." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lens-bench-same-size-image/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the converging lens (focal length 100 mm) and the white screen on the optical bench so that a sharp image of the lamp's arrow, the same size as the arrow, falls on the screen. Leave both standing upright on the bench, square across it and on its centre line, the screen's white face toward the lamp." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lens-bench-same-size-image/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `down`, `u_mm`, `v_mm`, `v_star_mm`,
`focus_err_mm`, `magnification`, `sharp`, `same_size`, `lens_at_mark`, `screen_at_mark` and, per element, `counts`,
`at_rest`, `on_bench`, `tilt_deg`, `square_err_deg`, `centred`, `tipped`, `off_bench`, `centre_mm`,
`off_centre_mm`, `from_object_mm`, `rider_low_mm`, `rider_high_mm`, plus each element's raw origin, quaternion and
velocity). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it
over at every reset (mechkit counts resets); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the lens's and the screen's poses,
once per control step (20 Hz), and keeps the last half second of where each element's rider corners and top were, to
tell whether it is at rest (from positions, not the reported velocity, which chatters while a part is gripped).
Distances are along the bench (y), from the object plane (the arrow slide's face): u to the lens's centre (45 mm
above the bench top, level with the arrow's centre), d to the screen's white face at the same height, and v = d - u.
An element counts when it is at rest (no rider corner and not its top moved more than 2 mm in 0.5 s), resting on the
bench (its centre over the bench top and the lowest corner of its rider within 3 mm of it), upright (5 degrees),
square across the bench seen from above (5 degrees; the lens either way round, the screen with its white face toward
the lamp) and centred (8 mm of the centre line). The image is sharp when both count, u > f, v > 0 and v is within
10 mm of v* = f u / (u - f), the lens law (1/u + 1/v = 1/f) for the lens's actual position, with f = 100 mm.

| Outcome | Score |
|---|---|
| The lens or the screen tipped (tilted more than 10 degrees: leaning over, lying down, upside down) or off the bench (its centre more than 3 mm below the bench top) | 0 |
| Sharp image, the same size as the arrow (v / u from 0.9 to 1.1) | 1.0 |
| Sharp image at any other size | 0.5 |
| Not sharp, but one counting element at its same-size mark (lens 190 to 210 mm from the arrow, or screen's face 390 to 410 mm) | 0.25 |
| Anything else | 0 |

Examples (checked in the tests): the start (lens 50 mm, screen 280 mm) 0; lens 200 and screen 400 mm 1.0, and with
each anywhere within 6 mm of its mark all 169 combinations at 1 mm steps score 1.0 (at u = 2f the object-to-image
distance u + v* is at its minimum, 4f, and barely changes); lens 150 and screen 450 mm (twice the size) 0.5; lens 300
and screen 450 mm (half the size) 0.5; lens 175 and screen 408 mm (1.33 times) 0.5; lens 200 with the screen left at
280 mm 0.25; screen 400 with the lens left at 50 mm 0.25; lens 200 and screen 370 mm (blurred: 30 mm from focus)
0.25; lens 120 and screen 300 mm 0. The magnification is v / u from where the screen actually stands: with the lens at
190 mm, a screen focused exactly (v* = 211 mm, 1.11 times) scores 0.5, and one 9 mm nearer (still sharp, 1.06 times)
1.0. Possible scores: 0, 0.25, 0.5 and 1.0; every outcome maps to exactly one.

The trial ends early once 1.0 has held for 2.5 s (`SUCCESS_HOLD_S`), otherwise it is scored as it stands at the
run limit. A new trial is recognised from mechkit's reset counter or, as a fallback, when both elements are put back
on their starting marks (within 3 cm) in a single step. A simulator state with a non-finite number or a zero
quaternion scores 0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing they should not, as the tests show: the screen turned round (white face away from
the lamp, the image on its black back) does not count; the order swapped (screen nearer the lamp than the lens) is
never sharp; the lens anywhere inside its focal length never gives a sharp image; elements held 5 mm above the bench,
standing 15 mm off the centre line, or standing on the lamp box do not count; a lens pushed over and slid to its mark
scores 0; a screen leaned back to move its face does not count once it leans more than 5 degrees; an element rocking
or turning on the spot is not at rest. The scorer cannot see the grippers, so an element held still standing on the
bench counts (see `fidelity`): in the simulated demo the lens may count while still gripped, and the trial ends
about when its gripper finishes opening.

## One arm alone

Offline (the kinematic check with `--one-arm`), the scripted demo scores 0.25 with either arm idle: with the right
arm idle the left arm sets the lens at 200 mm and the screen stays at 280 mm, nearer than 4f, so no image is sharp;
with the left arm idle the right arm sets the screen at 400 mm and the lens stays at 50 mm, inside its focal length.
The GPU runs are in 'Results in simulation'.

Reach (inverse kinematics of the arm's URDF with its link collision proxy): gripping straight down, at the screen's
grip height (10.4 cm above the table) the left arm reaches at most y -6.5 cm on the bench's centre line (x 30 cm) and
-9.5 cm at x 26 cm, at the end of the screen's board: just past the screen's start (its grip line at y -9.1 cm),
nowhere near its place (-21.1 cm), and a screen moved within that reach stays nearer than 4f, so no image can be
sharp. The right arm, at the lens's grip height (9.1 cm), reaches at most y +7.5 cm at x 30 cm and +10.5 cm at x 26
cm; the lens starts at +15.0 cm (its rider's near edge at +13.0 cm) and must end at 0. So, gripping straight down,
each arm can set only its own element and one arm earns at most 0.25. Leaning its gripper toward the far side, though,
an arm reaches much further: the left arm leaning 30 degrees reaches y -21.5 cm at x 30 cm, the screen's place, and
the right arm leaning 15 degrees reaches y +14.5 cm at x 30 cm (+17.5 cm at x 26 cm), about the lens's start. So one
arm alone could in principle set both elements with steeply leaned grips; the scripted one-arm runs do not try that,
and we do not claim that one arm cannot score 1.0. Both arms are used naturally: each sets the element on its own side
of the bench.

## Demo and run limit

The scripted demo takes 26.8 s (536 control steps). Each element is held straight down with the fingers closing along
the bench (y, yaw 0) across its 22 mm thickness, commanded 14 mm, closing over 1.5 s: the lens by its holder's top
bar, the tool point 85 mm above the rider's underside (pad tips 8 mm above the glass, 20 mm of pad on the bar; opened
to 50 mm, the pads stay 17 mm clear of the lamp's slide); the screen by the top of its board, the tool point 98 mm up
(27 mm of pad on the board, opened to 60 mm). Each is lifted until its rider is 20 mm above the bench, carried
straight along the bench at that height, lowered over the last 18 mm in 2 s to 2 mm above the bench and let go. The
left gripper then rises until its pad tips are 8 mm above the lens's top (higher, over the lens's place, joint 4 of
the left arm loses its margin), slides 60 mm along -x off the holder's end (the open pads pass beside the holder, 13.5
mm clear of the glass) and goes back to its own side; the right gripper rises until its pad tips are 20 mm above the
screen's top and goes back at the carry height (15 cm above the table), so its open pads stay at least 19 mm from the
screen. The finger pads never come within 13.5 mm of an element they are not holding (checked in the tests). The
clock: the right arm grips the screen at 5.0 to 6.5 s, carries it from 8.3 s, lowers it at 11.3 to 13.3 s (face 400 mm
from the arrow), lets go at 13.8 to 15.3 s and is back at 18.3 s; the left arm waits on its own side for 8 s (so the
two wrists never come close), grips the lens at 13.0 to 14.5 s, carries it from 16.3 s (the right gripper is then
rising from the screen), lowers it at 19.3 to 21.3 s (centre 200 mm from the arrow), lets go at 21.8 to 23.3 s, slides
off the holder at 24.8 to 25.6 s and is back at 26.8 s.

Offline (kinematic check): worst inverse-kinematics error 0.085 mm; smallest joint margin 12.8 degrees (joint 4 of the
left arm, rising over the lens's place), 13.4 degrees for the right arm; arm links at least 14.0 cm apart (capsule
proxy), tool points at least 30.4 cm apart; pad tips never lower than 78 mm above the table. Success has held for its
2.5 s at step 516; the scripted policy stops at step 597. The run limit is 1,200 steps (60 s).

For the GPU run: the arm sags a few millimetres under each element, so its rider may reach the bench before the slow
descent ends; that is expected. If an element lands off its aim (`centre_mm` in `TASK_TRACE` away from x 300 mm, or
`from_object_mm` away from 200 for the lens or 400 for the screen), `PLACE_TRIM` in `task.py` moves its place point
by minus the offset; each element scores full credit within 6 mm of its mark, so a few millimetres need no trim.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the lens back on the bench with its centre at
y +15.0 cm (50 mm from the arrow), the screen with its white face toward the lamp at y -8.0 cm (280 mm from the
arrow), both centred on x 30.0 cm and square across the bench. The bench, lamp box and scale are fixed. Nothing is
consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 26 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A school optical bench kit, or a 560 mm strip of 60 mm aluminium flat bar (or planed wood) clamped to the table along
y, centred 30 cm in front of the arm bases, with a metre rule or tape beside it. The object: a school ray box or lamp
box with an arrow slide (or a printed arrow 30 mm tall on white card, lit from behind), its face at the zero of the
rule. A 38 mm converging lens of focal length 100 mm (10 dioptres; check it: a distant window focuses 98 to 102 mm
behind it) in a plastic holder frame 50 mm wide and 22 mm thick on a plastic rider 50 x 40 mm, and a white screen
(rigid foam board 100 x 100 mm, 22 mm thick, painted white with a black back, or a thin screen in a holder of that
thickness) on a plastic rider 60 x 40 mm, about 80 to 90 g each. Score with the rule (u and d), a set square (tilt
and squareness) and a 3 mm feeler gauge under each rider. The score uses the marked f, not the image seen, so the
lamp may stay off; lit, the image at the full-score positions is sharp and the arrow's size (a lens within 2 mm of
its marked f puts the true focus within the 10 mm sharpness band at the 200 / 400 mm marks). The parts
weigh under 100 g and the lamp is a low-voltage ray box, so trials are safe.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
