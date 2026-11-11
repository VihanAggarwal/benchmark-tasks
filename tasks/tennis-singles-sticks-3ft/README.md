# Set the singles sticks for singles: centres 3 ft (0.914 m) outside the singles court

Chair umpires, referees and court crews (O*NET 27-2023.00, sports officials) set and check a court before every
match. Most club and tournament courts are combined doubles and singles courts with a doubles net. For a singles
match the ITF Rules of Tennis (Rule 1) have the net held up at 1.07 m by two singles sticks whose centres are
0.914 m (3 ft) outside the singles court on each side; the umpire checks them before play, because sticks set
anywhere else change the net's height over the singles court near the sidelines. Nothing on the court marks the
spot: it has to be known and measured from the lines. Court measurements are made to the outside edges of the
lines: the singles court is 8.23 m wide and the doubles court 10.97 m, so each alley is 1.37 m wide and the stick's
centre lies two thirds of the way across it. The usual mistakes are a stick on the singles sideline, in the middle
of the alley or against the doubles sideline.

Here the middle of a court painted for doubles at 1:15 lies on the table, 150 mm (2.25 m) either side of the net: a
blue court with white singles and doubles sidelines and the centre service line, a green surround, a dark grey band
along the net line where the doubles net hangs, and the two net posts on that line. Two white singles sticks (a
20 mm square post on a 40 mm square foot, 71 mm tall) stand on the near court, one each side. The instruction says
only that the court is painted at 1:15 with the standard ITF lines and that the sticks go on the net line "where the
rules put them"; the 0.914 m offset (61 mm here, |y| 33.53 cm from the centre line) is what the task tests. The score
is what an umpire's tape along the net line reads: 0.5 per side of the court for a stick standing there within
8 mm, 0.25 for a stick elsewhere in that alley. Each arm sets the stick on its own side; one arm alone sets one side
(0.5, see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the ITF arithmetic at 1:15 |
| `scoring.py` | The scorer: plain functions of the two sticks' poses (and the arms' pad-tip points and openings, for 'let go'), with a small tracker of their last second |
| `mechkit.py` | Helper that builds each stick as one rigid body of two boxes (foot and post) and reports its pose (see Requirements) |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does) |
| `tests/test_scoring.py` | Offline tests of the ITF arithmetic, every scoring tier and band edge, every zero condition, the let-go rule, stillness, the exploits we tried, resets, broken states, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and cylinders (the court mats, lines and net posts), the standard `task_score`
scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a stick that stands up on its own needs a
foot wider than its post. `mechkit.py` (imported by `task.py`) builds each stick from two boxes as one loose rigid
body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer. It
changes nothing else the embodiment does, and it stops the run with an error if a stick was not built as a compound
body. `rigstate.py` reads the arms only to check that each stick was let go. The technique (a `task_file.Box`
subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's import, wrapping
the embodiment's step and reset) follows the one first shown in the program repository's `tasks/bearing-press`;
these modules are an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This doubles court is hosting a singles match next, and the net is a doubles net (the dark band across the court marks where it hangs). The court is painted at 1:15 scale with the standard ITF lines. Put the two singles sticks in on the net line, one on each side of the court, standing upright where the rules put them, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tennis-singles-sticks-3ft/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This doubles court is hosting a singles match next, and the net is a doubles net (the dark band across the court marks where it hangs). The court is painted at 1:15 scale with the standard ITF lines. Put the two singles sticks in on the net line, one on each side of the court, standing upright where the rules put them, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tennis-singles-sticks-3ft/dream
```

Run one trial per process (the default). The score depends only on where the sticks stand, whether they have been
still for the last second and whether a gripper still holds one; the small tracker in the task module keeps the middle
of each stick's axis over the last second and diagnostics for the trace, and starts a new episode when rigstate or
mechkit sees the embodiment reset or the simulator puts both sticks back on their marks in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`plus_side` and `minus_side`; per stick its `score`,
`zero_reason`, `side`, centre `x_mm`/`y_mm`, `off_net_mm` (x minus 350 mm), `off_itf_mm` (|y| minus 335.3 mm,
outward positive), `tilt_deg`, `bottom_mm` (lowest point above the court), `held`, `still` and `path_mm` (its
axis midpoint's path in each of the last two 0.5 s windows); each arm's `left_tips`/`right_tips` and
`left_opening`/`right_opening`, `arms_readable` and `arms_unavailable`, or `rigstate_error`; each stick's raw
`origin` and `quat`; the first step each score was reached). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the two sticks' poses once per control step
(20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the finger
tips) and how far it is open, and scores the state at hand: the score is read at the last step. A stick's centre is
the middle of its foot's underside, where the post's axis meets the court; distances across the court are measured
from the centre line (y 0), as court measurements are, and the lines' outside edges are at |y| 27.43 cm (singles)
and 36.57 cm (doubles).

A stick is set when all of these hold:

- let go: no gripper's pad-tip point is less than 10 mm above the stick's top while within that gripper's finger
  reach of its axis. The gripper's yaw is not read, so the reach is the widest over every yaw: hypot(half the finger
  gap + 10 mm, 20 mm) + 28.3 mm (the foot's corner) + 5 mm, which is 94 mm fully open and 62 mm closed on the post.
  Each gripper is read on its own; one that cannot be read is skipped (and counted in `arms_unavailable`);
- still: the middle of its axis (35.5 mm up, not its centre on the court) moved under 1.5 mm along its path in
  each of the last two 0.5 s windows, so under 3 mm over 1 s (a path, not the reported velocity, which chatters
  while a part is gripped);
- upright: its post's axis within 10 degrees of vertical, foot down;
- standing on the court: its lowest point within 3 mm of the court surface (not on the other stick's foot, on a net
  post or held up);
- on the net line: its centre within 6 mm of x 35.0 cm (the 10 mm band's centre line).

| Stick | Condition | Points for its side |
|---|---|---|
| Set, at the ITF point | centre within 8 mm of \|y\| 33.53 cm (0.914 m outside the singles court) | 0.5 |
| Set, elsewhere in its alley | \|y\| from 27.0 to 37.0 cm (either sideline included), not at the ITF point | 0.25 |
| Anything else | not set (left at the start, held, moving, tipped, stacked, off the net line) or outside the alley | 0 |

Each side of the court (y > 0 and y < 0) scores the best of the sticks on it, so at most one stick counts per side,
and the task score is the sum of the two sides: 0, 0.25, 0.5, 0.75 or 1.0. Every outcome, with numbers: both sticks
at their ITF points, 1.0; one there and the other in the middle of its alley (|y| 32.0 cm, 15 mm off), on its
singles sideline (27.3 cm) or against its doubles sideline (36.6 cm), 0.75; both on the singles sidelines, both
mid-alley or both against the doubles sidelines, 0.5; one at its ITF point and the other left at the start, still
gripped, tipped over, upside down, stacked, off the net line or beside the net post (|y| over 37.0 cm), 0.5; both
sticks in one alley with one at the ITF point, 0.5; one in its alley off the point and the other not set, 0.25; both
left at the start, beside the posts, off the net line, tipped or still gripped, 0.

Every wrong guess a non-specialist makes is outside the 8 mm window: the middle of the alley is 15 mm off, the
doubles sideline 31 mm, the singles sideline 62 mm, next to the net post about 60 mm or more. The trial ends early
(success) once 1.0 has held for 2 s; otherwise it is scored as it stands at the run limit. Since a stick counts
only after 1 s of stillness and the hold is 2 s more, the trial cannot end while a closed gripper still holds a
stick even on a step where the arms could not be read (settling and letting go after the set-down take 2 s). A
state with a non-finite number, a zero quaternion or no base-frame origin (a simulator blow-up, or a stick not built
as a compound body) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: both sticks in one alley, even either side of the ITF
point (0.5 at most, as for one stick); a stick held on its spot by a gripper (0 for that side; both held 0); a stick
stood on the other stick's foot or on a net post, or leaning on something past 10 degrees (0 for that side); a
stick upside down (0); sticks at the ITF offset but off the net line (0); a stick still rocking (0 until it has been
still for 1 s).

## One arm alone

The demo splits the work by side: the left arm sets the stick that starts on the left (+y) side in the left alley,
the right arm the other in the right alley, at the same time. With one arm idle its stick stays where it started,
off the net line, and that side scores 0: the offline kinematic check predicts 0.5 with either arm idle (one side
set). Gripping straight down at any heading (tried every 30 degrees), neither arm reaches the other side's stick
(x 24.0 cm, |y| 15.0 cm on the far side of the centre line) or the other side's ITF point (|y| 33.53 cm), by our IK
model of the arm (`tests/test_scoring.py`, `OneArmReach`). We do not claim one arm is strictly unable: a policy that
leans the wrist far over, or pushes a stick across, might reach more. The task does not need two arms for its
skill; they halve the time. (GPU results: "Results in simulation" below.)

## Demo and run limit

The scripted demo takes 18.3 s (366 control steps), both arms on one clock, mirror images. Each stick is gripped
straight down round its post with the fingers closing along y (yaw 0), opened to 56 mm (18 mm either side of the
20 mm post; the pad tips stay 29 mm above the foot) and commanded to 12 mm to close, with the tool point 50 mm above
the foot's underside so the pads hold the post's upper 34 mm. Every gripper open or close takes 1.5 s, every final
descent is slow (30 mm over 1.5 s), each stick is carried with its foot 69 mm above the court, settles 0.5 s over
its place, is lowered and let go with its foot 1 mm above the net band after a 0.5 s settle, and the gripper rises
straight up before parking over its own side of the near court. Each place point is commanded 4 mm past the ITF
point along the carry (`PLACE_LEAD`), because in our GPU runs carried parts landed about 4 mm short of the
commanded point: offline (no lag) the sticks stand 2.0 mm beyond the net line and 3.4 mm outside the ITF point,
inside the 6 and 8 mm windows, and a 4 to 7 mm short landing still scores 1.0. `PLACE_TRIM` in `task.py` moves an
arm's place point if the GPU trace shows a steady offset.

Offline, each stick counts once its gripper, open, has risen with its pad tips 10 mm over the stick's top (about
16.3 s), and success is held at step 365 (18.25 s). The grippers stay at least 30 cm apart and the arms' collision
capsules at least 12 cm; the smallest joint margin is 13.7 degrees (joint 4). A 40 g stick barely loads an arm. The
run limit is 1,200 steps (60 s), more than three times the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the two sticks back upright on the near
court, sides along x and y, centred at x 24.0 cm, y +15.0 cm and at x 24.0 cm, y -15.0 cm; the court mats, lines
and net posts are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo ends after about 18 s of simulated time, when success
has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

A 300 x 900 mm board or table area: a 300 x 731 mm blue vinyl or painted rectangle (the doubles court, its long
side along y) with 84 mm green strips at both ends (the surround, to 450 mm either side of the centre line), a
10 mm dark grey tape band across the middle (the net line, 854 mm long), and white tape lines 3 mm wide along x on
the blue: the centre line, the two singles sidelines with their outside edges 274.3 mm either side of it and the
two doubles sidelines with their outside edges 365.7 mm either side of it, all stopping at the band. Two 25 mm
dark green dowels 71 mm tall glued upright on the band's ends, centred 426.6 mm either side of the centre line.
Two white ABS sticks (a 20 mm square post 63 mm tall glued on a 40 x 40 x 8 mm square foot, 40 g each). Positions
in `task.json`'s `setup`.

Scoring a real trial: after the arms have withdrawn, check each stick stands upright on the court (not on anything)
and has stopped moving; with a rule, measure from the band's centre line to the stick's centre (within 6 mm) and from
the centre line across the court to the stick's centre (335.3 mm within 8 mm for 0.5; 270 to 370 mm for 0.25).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
