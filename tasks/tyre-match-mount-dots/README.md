# Match-mount two tyres: red dot to a wheel's low-point mark, else yellow to the valve

Tyre fitters in car dealers, fleet garages and tyre shops match-mount tyres every day. Many tyres carry two sidewall
dots: red, the tyre's point of maximum radial force variation (its uniformity high point), and yellow, its lightest
point. Tyre makers' match-mounting instructions (for example Yokohama's) say the red dot is lined up with the wheel's
point of minimum radial runout, which many wheels mark with a coloured dot or a notch (steel wheels often carry a
white dot); when the wheel has no such mark, the weight method is used instead and the yellow dot goes to the valve
stem, the wheel's heavy point. A tyre mounted the wrong way round needs more balance weight or causes a vibration,
and the shop has to break the bead and turn it on the rim. Here a 1:10 model of a fitter's bench stands on the table:
two wheels on changer turntables, wheel A (left) with a white low-point mark and wheel B (right) without one, each
with a valve stem, and a loose tyre with a red and a yellow dot lying in front of each wheel. The robot has to drop a
tyre over each wheel, turned so the right dot meets the right mark, and let go. Knowing only "yellow to the valve"
gets wheel B right and wheel A wrong; only the precedence rule gets both. The score is what a fitter reads with a
ruler, a protractor and a photo from above.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two tyres' poses and the grippers' state, with a small tracker of each tyre's last second |
| `mechkit.py` | Helper that builds each tyre as one rigid ring of 36 ribs with two painted dots, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, the shortcuts we tried, unreadable and non-finite states, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed cylinders (turntables, wheels, valve stems, the mark), the standard `task_score` scorer,
and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a solid cylinder has no bore to
drop over a wheel and no dots to read. `mechkit.py` (imported by `task.py`) builds each tyre from 36 overlapping
upright cylinders in a ring (so the bore is a real 46 mm hole) and two thin painted dots, as one loose rigid body,
through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer. It stops
the run with an error if a tyre was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints a warning; on those steps no tyre
counts as let go (so a held tyre can never score), and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount a tyre on each wheel on the changer turntables, turning each tyre so its marks line up with the wheel the way tyre makers require, then let go. The red and yellow dots are on the tyre sidewalls. The small post on each wheel is the valve stem, and the white dot on one wheel is its low-point mark." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tyre-match-mount-dots/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount a tyre on each wheel on the changer turntables, turning each tyre so its marks line up with the wheel the way tyre makers require, then let go. The red and yellow dots are on the tyre sidewalls. The small post on each wheel is the valve stem, and the white dot on one wheel is its low-point mark." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tyre-match-mount-dots/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `n_seated`, `n_correct`, `correct`, `on_wheel`,
`arms_unavailable`, and for each tyre `wheel` (the nearest), `axis_offset_mm`, `bottom_mm` (underside above the
turntable's top), `tilt_deg`, `heading_deg`, `red_deg`, `yellow_deg`, `rule_dot`, `rule_target_deg`,
`rule_error_deg`, `seated_on`, `why_not_seated`, `let_go`, `zone_gap_mm`, `at_rest`, `path_mm`, `yaw_range_deg`;
also each tyre's origin, orientation and dot angles, both TCPs, pad-tip points and openings). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every reset
(rigstate's and mechkit's reset counters, and a teleport of a tyre back to its start), so run one trial per process
or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both tyres' poses and both grippers once
per control step (20 Hz). Each tyre's base frame is the centre of the ring's bottom face, with local +z up through the
dotted sidewall and local +x toward the red dot; the yellow dot is 100 degrees counter-clockwise of it. Angles about
a wheel are seen from above, counter-clockwise from +x (straight away from the arms). A dot's angle is the direction
from the tyre's own centre to the dot's centre: the tyre's turn, which is what match-mounting sets. Wheel A's white
mark is at 120 degrees and both valve stems are at 0 degrees, each 10 mm from its wheel's axis.

A tyre is seated on a wheel when its centre is within 6 mm of that wheel's axis seen from above (its bore clears the
wheel by 5 mm, so any tyre down round the wheel meets this), its underside within 3 mm of the turntable's top, its
axis within 5 degrees of vertical with the dots up, it is let go and it is still. Let go: each gripper's finger zone
at least 10 mm from the ring. rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round
its axis: the zone is everything within reach of a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness out, 13.5 mm across; 57 mm when fully open) around the 38 mm stretch back from the point between the pad
tips. Still: over the last 1 s the ring's centre travelled at most 1.5 mm in each 0.5 s window (path length of
positions; the simulator's reported velocities chatter while a part is gripped) and its heading changed at most
2 degrees. A tyre resting on the other tyre, on a wheel's top edge, upside down, tilted, still held or touched, still
moving or anywhere else is not seated.

Wheel A is correct when exactly one tyre is seated on it with its red dot within 10 degrees of 120 degrees; wheel B
is correct when exactly one tyre is seated on it with its yellow dot within 10 degrees of 0 degrees. Either tyre may
go on either wheel (they are identical).

| Outcome (at the end) | Score |
|---|---|
| Both wheels correct (success once held 1 s) | 1.0 |
| Exactly one wheel correct | 0.5 |
| At least one tyre seated, no wheel correct | 0.25 |
| No tyre seated | 0 |

Every outcome falls in exactly one row, and every row is reachable. The shortcuts we tried score as the rule says
(tests): yellow dot to the valve on both wheels (the rule without its precedence) 0.5; red dot to the valve on both
0.25; the colours crossed (yellow to the mark, red to the valve) 0.25; both tyres mounted at their starting turns
0.25; both right but one still held in the gripper 0.5, and both held in place 0; an open gripper resting on a
correctly turned tyre does not count it; both tyres dropped beside the wheels 0; a tyre stacked on the other one, or
hung on a wheel's top edge, is not seated; a tyre kept spinning on the turntable is not still. A tyre dropped at a
random turn has its rule dot inside the 10 degree window 20/360 = 5.6% of the time, and neither start is inside it
(tyre A's red dot starts 60 degrees from the mark, tyre B's yellow dot 60 degrees from the valve). A simulator state
with a non-finite number, a zero quaternion or no base-frame origin scores 0 for that step and leaves the scorer's
record as it was.

## One arm alone

The demo uses both arms at once, each on its own side: the left arm mounts tyre A on wheel A (red dot from 60 to
120 degrees, a 60 degree turn to the left) and the right arm mounts tyre B on wheel B (yellow dot from 60 to 0
degrees, a 60 degree turn to the right). Each wheel stands 15 cm past the midline from the other arm (34 cm out, about
57 cm from the other arm's base), and gripping straight down an arm reaches only about 7 to 10 cm past the midline at
that depth (our reach maps of the arm), so neither arm can lower a tyre over the far wheel that way. With one arm idle
the scripted demo mounts one tyre correctly, so the offline check predicts 0.5 with either arm idle. Two arms are not
needed for the skill being tested, and we do not claim one arm is strictly unable to score 1.0: with the wrist
tilted from vertical an arm reaches farther, so a policy that can do that might mount both tyres alone. The skill is
reading which dot goes to which mark on each wheel and turning the tyre to it. (GPU results will be added under
'Results in simulation'.)

## Demo and run limit

The scripted demo takes 20.6 s (412 control steps); both arms move at the same time on mirrored plans. Each arm comes
down over its tyre (pad tips 3 mm above the table), grips it across its 64 mm outside (fingers open to 76 mm, 6 mm a
side, then commanded to 56 mm, 8 mm narrower than the tyre, over 1.5 s; each pad rests on two ribs), lifts it to put
its underside 84 mm up (46 mm over the valve stems), carries it over its wheel in 2.5 s while turning it 60 degrees,
settles 0.8 s, comes down to 10 mm over the valve stem, settles 0.5 s, lowers it slowly (2.5 s) over the wheel to
1 mm above the turntable, settles 0.8 s, opens over 1.5 s, goes straight up and moves aside. Every grip yaw keeps
wrist joint 6 at least 66 degrees inside its limit (offline check). In our earlier simulator runs, parts carried in a
gripper ended turned counter-clockwise from the commanded heading by about 0.6 to 1.7 degrees, so each tyre is laid
1 degree clockwise of its answer (`YAW_TRIM_DEG` in `task.py`); with any turn from 0 to 3 degrees the dots end within
2 degrees of their marks (10 allowed). Arms under load have also set parts down a few mm short along the carry, so the
place point leads each wheel's axis by 2.5 mm along +x (`PLACE_TRIM`): the 5 mm radial clearance between bore and
wheel then takes any trail from 2.5 mm the wrong way to 7.5 mm short. Offline (no physics) the tyres drop 1 mm onto
the turntables at the release, their dots 1 degree clockwise of their marks, and success holds for its 1 s at step
391, once both open grippers have risen clear. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both tyres back flat, dots up, tyre A centred
at x 24.0 cm, y +15.0 cm with its red dot toward 60 degrees (yellow at 160), tyre B centred at x 24.0 cm,
y -15.0 cm with its red dot toward -40 degrees (yellow at 60); everything else is fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 21 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two dark grey painted steel discs 54 mm across and 6 mm thick as the turntables, fixed to the bench 30 cm apart; on
each, on the same axis, a silver painted steel cylinder 36 mm across and 24 mm tall as the wheel; on each wheel's top
face a black rubber post 6 mm across and 8 mm tall, centred 10 mm from the axis toward the far side of the bench, as
the valve stem; on the left wheel's top face only, a white painted dot 8 mm across, centred 10 mm from the axis at
120 degrees counter-clockwise from the valve stem. Two black solid rubber rings, 64 mm outside, 46 mm bore, 22 mm
tall, about 40 g, each with a 7 mm red dot and a 7 mm yellow dot painted on one face 27.5 mm from the centre and
100 degrees apart (yellow counter-clockwise of red). Clamp everything but the rings to the bench; nothing moves or is
powered. Score with a protractor (or a photo from above) for each dot's angle about the ring's own centre (as
the scorer measures it; a seated ring sits within 5 mm of the wheel's axis), a ruler for the height, and a photo
showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
