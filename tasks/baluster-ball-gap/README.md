# Stand balusters along a guard so a 51 mm test ball cannot pass through any gap

Residential guards (stairs, landings, balconies) may have no opening that a 4 in (102 mm) sphere can pass, and
building inspectors check with a 4 in ball, the openings next to the newel posts included. The carpenter works out
how many balusters a run needs from the clear distance between the posts and the balusters' width, then lays them
out so the openings are equal. Here the guard is at 1:2: a fixed oak shoe rail runs 356 mm between two fixed oak
newel posts, six white 20 mm balusters stand on the table, three on each arm's side, and the test ball is 51 mm.
The robot has to stand balusters upright on the rail so that no opening passes the ball and the openings are even.
The count is not given: four balusters can never do it (spread perfectly evenly they leave five openings of
55.2 mm), five (42.7 mm) or six (33.7 mm) can. The score is read from where the balusters stand at the end, as an
inspector reads it with the ball and a rule. Gripping from above, each arm reaches only its own three balusters
and not the far end of the rail, so one arm alone sets at most three balusters, which scores 0 (see "One arm
alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the six balusters' states, with a small tracker for the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and loose upright cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the white balusters upright on the oak shoe rail between the two newel posts so that a 51 mm test ball cannot pass through any gap, including the gaps next to the posts, and space them evenly. Use as many of the six balusters as you need." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/baluster-ball-gap/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stand the white balusters upright on the oak shoe rail between the two newel posts so that a 51 mm test ball cannot pass through any gap, including the gaps next to the posts, and space them evenly. Use as many of the six balusters as you need." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/baluster-ball-gap/dream
```

Run one trial per process (the default). The score depends only on where the balusters are; the small tracker in
the task module keeps diagnostics for the trace and recognises a new episode in the same process only when the
simulator puts every baluster back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`standing`, `n_standing`, `down_on_guard`, `openings_mm`
from the left post to the right, `passing`, `spread_mm`, `all_still`, the first step each score was reached, and
each baluster's centre, tilt, bottom above the rail's top and speed). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the six balusters' poses and
velocities, once per control step (20 Hz), and scores the state at hand: the score is read at the last step. The
rail and posts are fixed, so their faces come from `layout.py`.

A baluster stands on the rail when its axis is within 5 degrees of vertical (either end down; its two ends are
alike), its lowest point is within 2 mm of the rail's top, and its centre is over the rail's top (within 20 mm of
its centre line) between the posts. The openings are taken in order along the rail: from the left post's inner
face to the first standing baluster's surface, between each pair of neighbouring standing balusters (the distance
between their centres less one diameter, which is the clear gap a ball must get through at its narrowest, so a
baluster set off the centre line widens its openings), and from the last one to the right post's inner face. An
opening passes the ball when it is 51 mm or more; the spread is the widest opening minus the narrowest. A baluster
is down on the guard when it is tilted more than 5 degrees, does not stand, its outline seen from above (worked out
from its pose, not its bounding box) overlaps or comes within 2 mm of the rail's top or a post's top, and its lowest
point is no more than 20 mm above the rail's top: lying on the rail, or leaning against the rail, a post or another
baluster.

| Outcome | Score |
|---|---|
| Any baluster down on the guard (lying on it, or leaning against the rail, a post or another baluster) | 0 |
| No opening passes the 51 mm ball and the spread is 10 mm or less; every baluster still for 3 s | 1.0 |
| No opening passes the ball, spread over 10 mm | 0.5 |
| Exactly one opening passes the ball, with four or more balusters standing (a guard one baluster short) | 0.25 |
| Anything else (three or fewer standing, four spread evenly, two or more openings passing) | 0 |

Every outcome, with numbers: the start (one 356 mm opening) 0; one to three balusters on the rail 0; four spread
evenly 0 (five openings of 55.2 mm, all passing); four with one wide gap (48, 48, 48, 48, 84 mm) 0.25; five with one
opening of 51 mm or more 0.25; five or six with none passing but uneven (35, 50, 35, 50, 36, 50 mm) 0.5; five evenly
(42.7 mm) or six evenly (33.7 mm) 1.0. However four are spread, at least one opening is 55.2 mm or more, so four
score at most 0.25; five or six are needed for 0.5 or 1.0. Apart from the zero rule for a baluster down on the guard,
balusters left on the table, standing on a post's top, held in the air or knocked off the table are ignored. Any
baluster may go in any place. The trial ends early (success) once 1.0 has held with every baluster still (under 1
cm/s and 0.5 rad/s; one that fell off the table is not waited for) for 3 s, longer than the settle and release after
the last set-down (0.5 + 1.5 s), so it cannot end while the last baluster is still in a closed gripper; otherwise it
is scored as it stands at the run limit. A state with a non-finite number or a zero quaternion (a simulator blow-up)
scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: balusters laid flat across the rail to fill the gaps
(down on the guard: 0); four balusters zigzagged across the rail's width (that only widens the openings: 0); six
standing in three pairs side by side (four openings of 72-78 mm pass: 0); a baluster stood on top of another (not
on the rail: ignored); balusters stood on the table in front of the rail (not on it: 0); a fallen baluster left on
the rail beside a perfect guard (0). The scorer cannot see the grippers, so a baluster held still with its bottom
within 2 mm of the rail counts as standing (see `fidelity`); held 3 mm up it does not.

## One arm alone

The demo splits the work by reach: the left arm sets three balusters (y +12.5, +6.3 and 0 cm), the right arm two
(y -6.3 and -12.5 cm), and the right arm's third baluster stays on the table. Gripping from above, neither arm
reaches the other side's balusters: they stand 27 cm past the midline (their near sides 26 cm), while the
straight-down reach at grip height ends 18.5, 16.4 and 13.4 cm past it at x 14, 20 and 26 cm. Neither arm reaches
the far end of the rail either: at release height over the rail (x 30 cm) the left arm's straight-down reach ends at
y -10.2 cm, 2.3 cm short of the far-end baluster's mark at y -12.5 cm (by our IK model of the arm; the right arm is
its mirror image). So one arm alone sets at most its own three balusters, and three standing score 0 (0.25 needs
four). We do not claim one arm is strictly unable to finish: an arm tipped well over sideways reaches further than
straight down, so a policy able to grip a 20 mm baluster from the side, near the table, could in principle fetch
the other side's balusters and set the far end of the rail itself. The scripted demo grips from above only.

On the GPU both one-arm runs scored 0, as the offline check predicted. With the left arm idle
(`DEMO_IDLE_ARM=left`) the right arm set its two balusters (y -12.7 and -6.35 cm) and nothing else moved: two
standing, a 231.5 mm opening at the left post, 0. With the right arm idle (`DEMO_IDLE_ARM=right`) the left arm set
its first two (y +12.7 and +6.3 cm); its third, set down at y 0 about 2 degrees off plumb, leaned further over the
next few seconds, fell off the front of the rail and rolled away across the table, so two stood at the end (a
230.9 mm opening at the right post): 0. With all three standing it would still have scored 0. (In both full demo
runs the same third baluster stood within 0.01 degrees of plumb.)

## Demo and run limit

The scripted demo takes 65.8 s (1,316 control steps). Each baluster is gripped at mid-height (the TCP 40 mm up, the
pads holding 27 to 49 mm up its 80 mm) with the fingers closing along x, across the rail (left arm yaw -90 degrees,
right arm +90), opened only to 40 mm because its neighbours at the start stand 60 mm away along that axis (fully
open, 94 mm, the pads would hit them), and commanded to 12 mm to close (about 20 N per finger). It is lifted to
carry height (its bottom 100 mm up, 20 mm over the other balusters' tops), carried behind the posts (x 20 cm) to its
own line along y, lowered there to 30 mm, brought forward along that line over the rail between its neighbours
(42.7 mm clear on each side), lowered slowly over the last 19 mm to stand 1 mm above the rail, released after a
0.5 s settle, and the gripper rises straight up and goes back behind the posts. Every gripper open or close takes
1.5 s. The order keeps the two grippers at least 0.25 m apart whenever both are at the rail:

1. Together: the left arm sets a baluster at y +12.5 cm and the right arm one at y -12.5 cm (the two end ones).
2. The right arm sets its second at y -6.3 cm while the left arm picks its second and holds it up on its own side.
3. Once the right arm is back behind the posts (and on its way out of the way), the left arm sets its second at
   y +6.3 cm, then its third at y 0.

The demo's guard is five balusters with every opening 42.7 mm, 8.3 mm under the ball. The TCPs stay at least
25.1 cm apart and the arms' collision capsules at least 5.8 cm (both at the start of round 1, the wrists over the
rail's two halves; offline). The smallest joint margins are 11.0 degrees (right joint 6) and 12.5 degrees (left
joint 4); the pad tips stay at least 27 mm above the table. An 18 g baluster held at its middle makes the arm sag
well under 1 mm. In the offline proxy success is held at step 1,311; the scripted policy stops at step 1,377. The run
limit is 2,400 steps (120 s), which leaves Dream-1 almost twice the demo's time.

On the GPU (L40S, `-E object_jitter=0 -E control_hz=20`) the demo as written scored 1.0 in both runs, with the same
trace each time: the score went from 0 to 0.25 at step 883 (four standing, one 106 mm opening) and to 1.0 at step
1,227, and success ended the trial at step 1,307. The five balusters stood with openings of 41.0, 44.3, 42.2, 43.9,
43.3 and 41.3 mm from the left post (spread 3.4 mm, the widest 6.7 mm under the ball), each within 1.5 mm of the
rail's centre line, tilted at most 0.01 degrees, with nothing down on the guard. The right arm's third baluster never
moved from its mark. Each arm set its balusters 0 to 1.7 mm toward its own side of their marks, which is why the end
openings are the narrowest. `PLACE_TRIM` in `task.py` stays at zero.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the six balusters back upright on the
table, three on the left at x 14.0, 20.0 and 26.0 cm, y +27.0 cm, three on the right at the same x, y -27.0 cm; the
rail and posts are fixed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 66 s of simulated time, when success
has held for 3 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

An oak strip 356 x 40 x 10 mm and two 40 x 40 mm oak blocks 100 mm tall, fixed directly to the table top (screws,
clamps or strong double-sided tape; no base board, so the strip's top is 10 mm above the table), the blocks' inner
faces flush with the strip's ends (356 mm apart), all centred 30 cm in front of the arm bases. Six 20 mm maple dowels
cut to 80 mm with square ends, painted white (about 18 g each). A 51 mm (2 in) wooden craft ball for the test.
Positions in `task.json`'s `setup`.

Scoring a real trial: offer the ball to every opening between the posts (from the front, at any height and angle);
an opening passes it when the ball goes through. Read each opening at its narrowest with a rule or calipers (post
face to baluster, baluster to baluster) and subtract the narrowest from the widest for the spread. A baluster stands
when it rests on the rail plumb within 5 degrees (its top within 7 mm of plumb over its 80 mm height); one lying on
the rail, or leaning against the rail, a post or another baluster, scores 0.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one
trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The demo also scored 1.0 in an earlier GPU run, with the same trace (see "Demo and run limit"). A settling run (no
plan, 160 steps) left all six balusters on their marks. Dream-1 ran to the 2,400-step limit: it picked up one of
the left balusters, dropped it on its side on the table behind the rail and set none on the rail, so the score
stayed 0. The one-arm runs are described under "One arm alone".
