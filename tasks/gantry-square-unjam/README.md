# Square up a racked gantry beam, then pull it in by both ends to its marks

Gantry robots, CNC routers, 3D printers and lab-automation rigs move on a beam that rides on two parallel guides.
Pulled or pushed from one end, a gantry racks: one end leads, the beam turns and its end bearings bind, like a
wide, shallow drawer pulled by one corner. Technicians moving an unpowered gantry for setup or maintenance read
which end lags, square the beam first, then move both ends together. Here a 520 mm aluminium beam lies between two
steel guide rails at the back of the table, out of square: its right end is 8 mm farther from the arms than its
left. It must be squared, pulled 22 cm toward the arms by both ends together until its ends are level with two
yellow marks, and let go. The scorer reads the beam's pose at every control step, not only at the end: one end
leading the other by more than 12 mm at any moment scores 0, and so does moving the beam more than 20 mm without a
gripper at each end. A single grip at one end turns the beam as it drags it (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the beam's state and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and one loose box) and the standard `task_score` scorer.

The score needs a gripper at each end of the beam while it moves and, for 1.0, both grippers clear of it at the
end, and `yam_arena` does not pass the grippers' state to a task file's hooks. `rigstate.py` (imported by
`task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool point,
finger-tip point, finger opening, in the objects' frame) before each control step and to start a new trial at each
reset. It reads only state the simulator already has and changes nothing the embodiment does. If it cannot read the
arms it prints why once; the hands-off and both-ends conditions are then taken as met on those steps, and the trace
counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The grey gantry beam rides between two guide rails and is out of square: its right end is 8 mm farther from you than its left end. First pull the right end toward you until the beam is square, then pull the beam toward you by both ends together until both ends are level with the yellow marks beside the rails. Never let one end lead the other by more than 12 mm, or the beam racks and jams. Then let go of it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/gantry-square-unjam/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The grey gantry beam rides between two guide rails and is out of square: its right end is 8 mm farther from you than its left end. First pull the right end toward you until the beam is square, then pull the beam toward you by both ends together until both ends are level with the yellow marks beside the rails. Never let one end lead the other by more than 12 mm, or the beam racks and jams. Then let go of it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/gantry-square-unjam/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`lead_mm` (signed: + while the right end lags),
`max_lead_mm`, `squared_step`, `squared_travel_mm`, `max_lead_after_square_mm`, `left_end_off_mm`,
`right_end_off_mm`, `centre_off_mm`, `travel`, `rise_mm`, `tilt_deg`, `side_mm`, `breaches`, `left_zone_mm`,
`right_zone_mm`, `opening`, `ends_held`, `unheld_travel_mm`, `arms_unavailable`, the beam's pose, both TCPs,
...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The lead is how much farther from the arms (along x) the
centre of one end face is than the other's: 520 mm times the sine of the beam's turn about the vertical while it
lies flat. It starts at 8 mm. The scorer latches 'squared' at the first step the lead is 2 mm or less, and
'squared first' if the beam's centre had by then moved no more than 25 mm toward the marks (11% of the 224 mm
move), so a beam dragged out of square to the marks and straightened there does not count as squared first. Four
things score 0 the moment they happen, whatever follows: the lead going over 12 mm (racked); the beam popping up
(its lowest point more than 5 mm above the table, which the lips allow only by 4 mm, or tilted more than 5
degrees); the beam leaving the rails (an end face's centre past either end of the rails, or the beam more than
10 mm off the midline sideways); the beam's centre coming more than 20 mm toward the marks, in all, without a
gripper at each end. A gripper is at an end when its finger-tip point is within 10 mm of the beam and at least
130 mm from the beam's middle along its length, toward that end (the demo grips 200 mm out); the scorer adds up
the centre's movement toward the marks on every step without a gripper at each end, signed, so that resting
jitter cancels and only a real move counts. This is the instruction's "by both ends together": squaring the
beam with one arm before the other has gripped moves the centre about 4 mm, well inside the 20 mm.

| Outcome | Score |
|---|---|
| Racked, popped up or off the rails at any step; moved more than 20 mm without a gripper at each end; or never squared | 0 |
| Both end centres within 5 mm of the marks' line (x 22.0 cm), lead now 2 mm or less, squared first and never over 4 mm since, both grippers' finger zones 30 mm or more from the beam | 1.0 |
| Both end centres within 10 mm of the line, squared first and never over 7 mm since (grippers anywhere) | 0.75 |
| The beam's centre within 15 mm of the line | 0.5 |
| The beam's centre moved at least 30% of the way (67 mm) | 0.25 |
| Anything else (not moved far enough) | 0 |

The scorer cannot see which way a gripper is turned, so each gripper's fingers count as a round zone about its
axis, out to the fingers' outer corners (57 mm fully open), from the finger tips to 38 mm back up the gripper; open
and straight above the beam, the zone is 30 mm clear once the tool point is 130 mm above the table. A gripper
closed again well clear of the beam is still clear. The score is read at the last step; the trial ends early once
the 1.0 conditions have held with the beam still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised
from rigstate's reset hook, or, as a fallback, when the simulator puts the beam back at its start (within 4.5 cm,
which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0 for
that step and leaves the record of the trial as it was. The beam is a plain box, so either broad face may be down.

Unintended strategies earn nothing more than they should, as the tests show: pulling the leading (left) end first
racks the beam at once (0); walking it one end at a time in 15 mm steps racks it (0); lifting it to carry it pops
it out from under the lips (0); pulling it out of the front of the rails and pushing it back is 0; dragging it to
the marks still 8 mm out of square is 0 (never squared), and straightening it only once there is 0.5 (not squared
first); one gripper pulling it square from the middle while the other arm stays away is 0 once the beam has come 20
mm. Squaring by pushing the leading end back instead of pulling the lagging one counts as squaring (the scorer
reads only the lead). Each arm moving its own end in turn while the other holds is a two-arm way of keeping the
beam square and is scored by its lead like any other: 5 mm steps 0.75, 3.5 mm steps 1.0. Leaving the grippers on
the beam at the end is 0.75. A gripper only resting against its end (within 10 mm) while the other arm pulls counts
as being at that end; the pull is then still one-ended, and by the compliance model below the beam racks.

## One arm alone

The instruction asks for the beam to be pulled by both ends together, and the score holds a policy to it: once
the beam's centre has come more than 20 mm toward the marks without a gripper at each end, the trial scores 0. One
gripper cannot be at both ends at once, so a single arm can square the beam (a few millimetres of travel) but
never move it far enough for any credit. `DEMO_IDLE_ARM=left` and `DEMO_IDLE_ARM=right` therefore score 0 whatever
the physics does; the offline kinematic check, which carries a gripped object rigidly with its gripper and so
cannot show racking, predicts 0 for each by this rule.

Why the method matters: gripping its own end, 200 mm from the beam's middle, an arm drags the beam's sliding
friction (about 2.45 N at friction 1.0, 2.9 N to start it) through a 0.49 N m moment about its grip. The YAM's
joints are position-controlled springs (taken as 80/80/80/10/10/10 N m/rad, no payload compensation), and the
compliance model (J K^-1 J^T at the grip) turns that into 3.4 degrees of turn, a 30 mm lead: past the 12 mm limit,
and past the 16 mm at which the beam meets both rails at diagonal corners and binds. So the scripted one-arm runs
are also expected to rack; whether PhysX locks the bound beam or lets it slip is recorded here from them. With
both ends held, the moments cancel and each arm carries half the friction: the model gives each grip a 9 mm trail
and a turn of 0.09 degrees.

Reach: straight down at grip height, at any heading, the left arm reaches no point at y -20 cm or beyond anywhere
along the rails, and none at y -10 cm farther out than x 30 cm (inverse kinematics of the arm's URDF with joint
limits and a collision proxy); the right arm is its mirror image. Straight down, the nearest grip to the beam's
middle either arm reaches at the start is 45 mm off it, and the middle itself once the beam has come about 3 cm.
Tilted about 10 degrees, though, either arm reaches the middle at the start, and a grip at the middle drags the
beam with no moment about it: a single arm could square the beam with a twist of its wrist and pull it straight.
Without the 20 mm rule that one-arm move could score 1.0; with it, it scores 0, as the tests show.

## Demo and run limit

The scripted demo takes 26 s (520 control steps). Both arms grip their own end of the beam 200 mm from its middle
(60 mm in from the end), closing the fingers along x across its 40 mm depth (fingers commanded 8 mm narrower,
over 1.5 s), the left at yaw -90 degrees and the right at +90, each turned 0.88 degrees with the beam, the tool
point 18 mm above the table (pad tips 5 mm). The last 22 mm of each descent take 1.5 s. The right arm then pulls
its lagging end square while the left holds and both wrists turn back to square (2 s); both arms pull the beam
22 cm toward the arms over 9 s (peak 38 mm/s), stop 3.5 mm past the marks' line (the right arm 9.6 mm), ease off over
3 s to 0.3 mm (left) and 2.8 mm (right) past it, let go and rise to 150 mm. The TCPs stay 40 cm apart and the arm links at least 23.6 cm apart; the
grippers (collision proxy) stay at least 33 mm from the lips and 40 mm from the rails. Offline, success has held
for its 1 s at step 570; the scripted policy stops at step 621. The run limit is 1,200 steps (60 s).

Trims allow for the arms trailing their commands under the beam's friction, calibrated in six GPU runs (L40S), each
read from `TASK_TRACE`. With the offline model's trims (1.5 mm past square, 9 mm overshoot) the beam was never squared
(0): the right gripper trailed its squaring command by 3.2 mm and the lead only came down to 2.9 mm. So the squaring
move is commanded 4.5 mm past square at the right grip (the lead changes about 1.3 mm per mm of grip difference,
since the end faces are 520 mm apart and the grips 400 mm), which squares it at step 204 with the centre 2.9 mm along.
During the pull the right gripper trails about 3 mm more than the left, so its pull is commanded further; and when
the grippers unload at the end, still gripping, the left end steadily moves about 2 mm further than the right (the
lead rose by 1.8 to 2.5 mm in three runs, whatever the ease-off points), each leaving 0.75. So the right arm's pull
is commanded 6.1 mm further than the left's, leaving the right end about 1.7 mm ahead for the unload to bring square;
the pull stops 3.5 mm past the marks (trailing, the ends stop about 2.5 mm past them); and the grippers ease off over
3 s, to 0.3 mm (left) and 2.8 mm (right) past the line, before letting go. To retune, read `lead_mm`,
`squared_travel_mm`, `max_lead_after_square_mm`, `left_end_off_mm` and `right_end_off_mm` in `TASK_TRACE` and adjust
`SQUARE_TRIM`, `RIGHT_PULL_BIAS`, `OVERSHOOT` and `EASE_X` in `task.py`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. The beam starts 0.22 mm from each rail at its
diagonal corners (0.5 mm of play per end when square). Reset: the beam back flat between the rails at the back,
centred at x 44.4 cm, y 0, its left end face's centre at x 44.0 cm and its right end face's at x 44.8 cm; the rails,
lips and marks are fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 26 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 330 mm lengths of 12 x 40 mm steel flat bar screwed upright to a baseboard with their inner faces 521 mm apart,
each with a 10 x 6 mm steel strip screwed along its top so it overhangs the gap by 10 mm, 34 mm above the board; a
520 mm length of 40 x 30 mm aluminium rectangular tube (about 0.25 kg) as the beam; a rubber mat on the board
between the rails so the beam drags as in the simulation; 1 mm yellow plastic tabs on the board outside the rails
at 22.0 cm; and 1 mm scales along both rails' tops. Film from above to read the lead, and where the grippers are,
frame by frame, and measure the end positions with a tape from each rail's near end.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
