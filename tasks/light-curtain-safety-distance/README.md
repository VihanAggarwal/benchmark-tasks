# Mount a press's light curtain at its ISO 13855 safety distance

A safety light curtain only protects a press if its beam is far enough from the die: ISO 13855 sets the minimum
distance S = K x T + C from the machine's stopping time T and the curtain's resolution d, so that a hand breaking
the beam cannot reach the die before the press stops. Here a bench-scale press mock stands at the far end of the
table with a yellow die line taped along its front, and an opening 280 mm wide runs toward the arms between two
side guards. The curtain's emitter and receiver posts stand on mats outside the guards, their lens (broad) faces
toward the press. Each post must go into the opening against its own guard, turned so the lenses face each other
across the opening, with the beam (the line through the posts' centres) at S = 2000 mm/s x 0.06 s + 8 x (20 - 14) mm
= 168 mm from the die line, or up to 20 mm farther. A post closer than 168 mm scores 0. With a straight-down grasp
each arm reaches only its own post, so the work splits naturally: one post per arm.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place (S is computed there from K, T and d) |
| `scoring.py` | The scorer: plain functions of the two posts' states, with a small tracker for the last half second |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge, zero condition and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, the right one 1.5 s behind the left): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the light curtain's two posts in the opening in front of the press, one against each side guard, with their lens faces (the broad faces) facing each other across the opening. The beam runs through the posts' centres: put it at the minimum safety distance from the yellow die line's near edge, S = 2000 mm/s x T + 8 x (d - 14 mm), with stopping time T = 0.06 s and resolution d = 20 mm, or up to 20 mm farther." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/light-curtain-safety-distance/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the light curtain's two posts in the opening in front of the press, one against each side guard, with their lens faces (the broad faces) facing each other across the opening. The beam runs through the posts' centres: put it at the minimum safety distance from the yellow die line's near edge, S = 2000 mm/s x T + 8 x (d - 14 mm), with stopping time T = 0.06 s and resolution d = 20 mm, or up to 20 mm farther." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/light-curtain-safety-distance/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: for each post `d_mm`, `gap_mm`, `tilt_deg`, `bottom_mm`,
`moved_mm`, `at_rest`, `upright`, `on_floor`, `at_guard`, `in_window`, `placed`, `side`, `too_close` and, once
both are placed, `facing_deg`; plus `score`, `zero_reason`, `placed_sides` and both posts' poses. Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the end state, as an installer
checks it with a tape from the die line, a feeler gauge and a look from above. The scorer reads only the two posts'
poses, once per control step (20 Hz), and keeps each post's last half second of positions (its centre, the centre
of its top and the centre of a broad face) to tell a post at rest from one still moving; it does not use the
simulator's velocities, which chatter while a body is gripped.

The safety distance is S = 2000 mm/s x 0.06 s + 8 x (20 - 14) mm = 120 + 48 = 168 mm. For each post, D is how far
its centre line is in front of the die line's near edge (x 44.0 cm), and its gap is the distance from its outer
face (the farthest corner toward the nearer guard, so a turned or tilted post is measured by its real outline) to
that guard's inner face. A post is *placed* when it stands upright (within 5 degrees, either end down) and at rest
on the floor plate (its lowest point within 3 mm of the plate's top), its gap is at most 10 mm (touching counts)
and 168 <= D <= 188 mm. A placed post *faces* the other when the line through its two broad faces is within
5 degrees, seen from above, of the line between the two posts' centres: that covers both a post turned off square
and a beam skewed by the two posts standing at different distances.

| Outcome | Score |
|---|---|
| Either post's centre between the guards (up to their outer faces) less than 168 mm in front of the die line, or past it: standing, lying, held or on the press | 0 |
| Both posts placed, one against each guard, both facing each other (the trial ends as a success once this has held 2.5 s) | 1.0 |
| Both placed, one against each guard, only one facing the other | 0.75 |
| Both placed, one against each guard, neither facing (for example, not turned: lenses toward the press) | 0.5 |
| One guard with a placed post (the other post not placed, or both posts at the same guard) | 0.25 |
| Anything else (posts left on their mats, too far, away from the guards, tilted, lying, held in the air) | 0 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. A post that is not placed earns nothing for itself and leaves the other
post's credit as it is, unless it is closer than S (then the whole trial is 0). Facing is judged against the line
between the two posts, so posts at different distances face when turned toward each other; squared to their guards
5 mm away, posts at the window's opposite ends (168 and 188 mm) are 4.9 degrees off that line and still face. The
score is read from the end state; the 2.5 s hold only ends a trial early as a success. The posts are identical and
the emitter/receiver names are only labels: either may stand at either guard.

Unintended strategies earn less, as the tests show: a post laid on its side at the right place is not upright (it
does not count); a post held in the air over its place (more than 3 mm up) does not count; two posts at the same
guard count once (0.25); a post outside the opening, on top of a guard or on top of the other post does not count;
a post set diagonally into the corner is placed but does not face (0.75 with the other facing); a post dropped
anywhere in the opening closer than 168 mm, or onto the press, makes the trial 0 however well the other post
stands; and a post still sliding or turning (more than 2 mm in the last 0.5 s) does not count until it stops. The
posts are plain boxes with both broad faces alike, so a post turned the wrong way round (lens toward its own
guard) is not told apart from one facing the right way; the scorer cannot see the grippers, so a post held still in
place counts as standing there (both in `fidelity`).

## One arm alone

With a straight-down grasp each arm reaches only its own post: the left arm stops 33 mm short of the right post's
centre on its mat (straight-down inverse kinematics of the YAM arm), and the right arm likewise on the left. The far
guard is only just in reach: the left arm can set a post down at the right guard's farthest corner of the window
(D 188 mm, 10 mm from the guard) but not at the demo's place (D 178 mm, 5 mm from the guard, about 2 mm out of
reach), and with that grasp it cannot fetch the right post. On the L40S the demo with one arm idle scored 0.25 each
way, as the offline kinematic check predicted: with `DEMO_IDLE_ARM=left` the right arm placed its post at the right
guard (D 179.9 mm, gap 4.35 mm) and the other post stayed on its mat; with `DEMO_IDLE_ARM=right` the left arm placed
its post at the left guard (D 180.2 mm, gap 4.05 mm) and the other post stayed on its mat. Facing is only scored once
both guards have a placed post, so one arm running the demo earns 0.25. The task does not claim that two arms are
physically required (a tilted or side grasp might reach farther); it splits naturally by side, one post per arm.

## Demo and run limit

The scripted demo takes 20.0 s (400 control steps); the right arm runs the same plan, mirrored, 1.5 s behind the
left, so the two wrists do not pass the guards' front ends at the same time (the arms stay at least 48 mm apart,
capsule proxy, and the grippers 0.23 m). Each arm comes down over its post, grips it across its 56 mm width 20 mm
below its top (fingers commanded 8 mm narrower, over 1.5 s), lifts it 25 mm off the mat, turns it a quarter turn
(left arm to yaw -90 degrees, right arm to +90) so its broad faces point across the opening, carries it round the
front end of its guard at x 15 cm, forward along the guard with 10 mm between them, then over its place: beam
178 mm from the die line (the middle of the window) and outer face 5 mm from the guard. It settles 10 mm up, lowers
the last 8 mm over 2 s, opens over 1.5 s with the post's bottom aimed 2 mm above the floor and goes straight up. At
the place the fingers close along x, so neither finger is between the post and its guard (the pads stay 9.5 mm
clear of it). The arm's controller does not carry the post's weight: the static estimate of the sag at the place
is (-1.2, +-0.9, -2.8) mm (toward each arm's own guard), so each post comes to rest on the floor shortly before the
slow descent ends, a millimetre or two farther from the die line and about 4 mm from its guard, inside every
tolerance. The 2.5 s stillness (rather than the usual 1 s) outlasts settling, letting go and the first part of the
lift, so the trial ends with both grippers open. On the L40S the demo scored 1.0: both posts placed and facing from
about step 335, success held to step 384 (19.2 s), ending with the posts at D 180.2 and 179.8 mm, gaps 4.05 and
4.17 mm and facing within 1.1 degrees, and no post tilted more than 1.2 degrees at any sampled step (the offline
kinematic check had predicted step 418). The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: each post back upright on its blue mat,
centred, broad faces toward the press and the arms (emitter on the left mat at x 16.0 cm, y +22.0 cm; receiver on
the right at x 16.0 cm, y -22.0 cm). The press, floor plate, die line, guards and mats are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo trial ends at 19.2 s of simulated time, once success
has held 2.5 s; a trial that does not succeed runs the full 60 s); the first start of Isaac Sim on a new instance
takes several minutes more.

## Real-world build

Two 10 mm steel panels 246 mm long and 100 mm tall clamped upright to the bench 280 mm apart as the side guards, a
painted wooden block 80 x 300 x 140 mm at their far end as the press, a 2 mm grey PVC sheet between the guards as the
floor and a strip of 6 mm yellow vinyl tape on it along the press's front as the die line, with a blue 80 x 80 x 3 mm
rubber mat outside each guard's front end as in the setup. The posts are two 36 x 56 x 100 mm boxes of about
150 g (hollow aluminium extrusion with end caps, or a light curtain's housing offcut), with red tape on both broad
faces as lens windows. Measure D with a tape from the die line's near edge to each post's centre line, the gap to
the guard with a feeler gauge, and the facing angle from a top photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |

The demo succeeded at step 384 (19.2 s) with the posts at D 180.2 and 179.8 mm, gaps 4.05 and 4.17 mm, and facing
within 1.1 degrees. Each one-arm run placed one post at its own guard and left the other on its mat. A settling
run (`DEMO_HOLD_ONLY=1`, 160 steps) left both posts upright on their mats, with no measurable drift. Dream-1 moved the
arms but did not move either post in its 1,200 steps: both stayed upright on their mats (0).
