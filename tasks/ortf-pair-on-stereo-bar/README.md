# Rig two cardioid mics on a stereo bar as an ORTF pair aimed at the choir

Location and broadcast recording engineers rig stereo microphone pairs for choirs, orchestras and services. ORTF,
the near-coincident technique of the French broadcaster Office de Radiodiffusion-Television Francaise, is one of the
standard set-ups (Eargle, The Microphone Book; the Schoeps and DPA guides to stereo technique): two cardioid
microphones with their capsules 170 mm apart and their axes splayed outward at an included angle of 110 degrees (55
degrees either side of the pair's centre line), the pair centred on the source and aimed at its centre. The spacing
gives time differences and the angle level differences; together they give a recording angle of about 96 degrees.
An engineer checks a rigged pair with a ruler and a protractor. The obvious alternatives a non-specialist sets up, a
parallel spaced pair, an XY pair at 90 degrees, a NOS pair (300 mm, 90 degrees) or the right angle with the capsules
crossed, each give a different stereo image.

Here a grey stereo-bar plate lies on the midline, a choir riser with six singer figures and a white centre mark
stands beyond it, and two small pencil-condenser microphones (70 x 24 x 24 mm, matte black with a silver grille end:
the capsule end) lie in grey foam case pads, one in front of each arm. The instruction names the technique, not its
numbers: the robot has to know what an ORTF pair is, tell each mic's capsule end, turn each mic 75 degrees, lay both
flat on the plate with the capsules 170 mm apart and the axes 110 degrees apart and splayed outward, centred on the
choir and aimed at its mark, and let go. Spacing and angle are at full scale.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two mics' poses and the grippers' state, with a small tracker of the mics' last second |
| `mechkit.py` | Helper that builds each mic as one rigid body of a black body box and a silver grille box, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and tolerance edge, every zero condition, the shortcuts and exploits we tried, unreadable and non-finite states, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and cylinders, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a one-colour box has no capsule
end. `mechkit.py` (imported by `task.py`) builds each mic from two boxes (a 55 mm black body and a 15 mm silver
grille) as one loose rigid body, through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and
orientation) to the scorer. It stops the run with an error if a mic was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is
then taken as met on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, one after the other): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Rig the two small cardioid microphones on the stereo bar as an ORTF pair for the choir on the riser. The silver grille end of each mic is its capsule. Lay both mics flat on the grey bar plate, set the way an ORTF pair is set, aimed at the centre of the choir (the white mark on the riser), then let go of both mics." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/ortf-pair-on-stereo-bar/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Rig the two small cardioid microphones on the stereo bar as an ORTF pair for the choir on the riser. The silver grille end of each mic is its capsule. Lay both mics flat on the grey bar plate, set the way an ORTF pair is set, aimed at the centre of the choir (the white mark on the riser), then let go of both mics." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/ortf-pair-on-stereo-bar/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: `score`, `success`, `zero_reason`, `gate`; for each mic
(`mics`) its centre, heading, tilt, `bottom_mm` (lowest point above the plate top), `over_plate`, `still` and
`hands_off`; for the pair (`pair`) `separation_mm`, `included_deg`, `splay_outward`, `bisector_deg`,
`facing_err_deg`, `mid_off_axis_mm`, `A`, `S`, `C` and `why_not_a`; `motion` (each mic's path in the last two 0.5 s
windows and its heading's range over 1 s), `zone_gap_mm` (each mic's nearest finger zone), `arms_unavailable`,
`bad_steps`, `first_step` (when each score was first reached), each mic's capsule point, heading and speed, and both
TCPs and openings. Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and
starts it over at every reset (rigstate's and mechkit's reset counters, and a teleport of both mics back to their
starting marks), so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both mics' poses and both grippers once per
control step (20 Hz). From each mic's base frame (the middle of its bottom face; x along the mic toward the grille,
z up) it computes the capsule point c (the grille's centre, 7.5 mm behind the front face, at mid-height), the axis
u (tail to grille) and the mic's eight corners. All pair measures are seen from above.

The gate, for each mic: let go (every gripper's finger zone at least 10 mm from it; rigstate gives no gripper
heading, so the zone is everything within reach of a pad's outer corner, half the current finger gap plus the 8 mm
pad thickness out and 13.5 mm across, around the 38 mm stretch back from the point between the pad tips); still (in
each of the last two 0.5 s windows its centre travelled under 1.5 mm, and its heading changed less than 2 degrees
over the last 1 s: positions, not the simulator's reported velocities, which chatter while a part is gripped); long
axis within 10 degrees of horizontal; centre over the plate; lowest point within 5 mm of the plate's top. A mic
lying on the other mic is 24 mm up and fails; so does a mic on the table (8 mm down) or held over the plate.

Then: A, the axes 110 +/- 8 degrees apart, splayed outward ((cA - cB) . (uA - uB) > 0: the capsules farther apart
than the tails) and the bisector uA + uB within 30 degrees of the direction from the capsules' midpoint to the white
mark; S, the capsule points 170 +/- 10 mm apart; C, the midpoint within 10 mm of the source axis (y 0) and the
bisector within 5 degrees of the direction to the mark.

| Outcome (gate met: both mics let go, still, flat on the plate) | Score |
|---|---|
| A and S and C: 110 degrees, splayed, 170 mm, centred and aimed at the mark (success once held 1 s) | 1.0 |
| A and S, not C: the right pair more than 10 mm off the axis, or aimed more than 5 (up to 30) degrees off the mark | 0.67 |
| A, not S: the right angle facing the choir, spacing outside 160-180 mm (with or without C) | 0.33 |
| Not A: angle outside 102-118 degrees (parallel, XY, NOS, DIN), crossed, aimed away or sideways | 0 |
| Gate not met: a mic held or touched, sliding, standing on an end, on the other mic, on the table, off the plate | 0 |

Each state maps to exactly one line (first match wins). The two mics are alike: either may be either one of the
pair, and every measure is the same with them swapped; a mic rolled about its own axis lies the same and scores the
same. Unintended strategies earn nothing, as the tests show: the right geometry on the table in front of the plate
(0); both mics held just over the plate by the grippers (0); an open gripper pressing a mic down (0); mic B laid
across mic A as an X at 110 degrees (0); both mics stood on their tails (0); the mics left in their pads (0); the
right angle with the capsules crossed (0). A new trial is recognised at rigstate's or mechkit's reset, or when the
simulator puts both mics back on their starting marks (within 4.5 cm, which covers a seeded reset's jitter) in a
single step. A simulator state with a non-finite number, a zero quaternion or no base-frame origin scores 0 for that
step and leaves the scorer's record as it was.

## One arm alone

Offline (the kinematic check with `--one-arm`), the scripted demo scores 0 with either arm idle; the GPU runs are in
'Results in simulation'. With the right arm idle the left arm lays mic A on the plate, mic B stays in its pad, and the
gate fails: 0. With the left arm idle the mirror happens: 0.

Reach (inverse kinematics of the arm's URDF with its link collision proxy): gripping straight down at pad height, the
left arm reaches at most about y -13 cm at x 25 cm, some 17 cm short of mic B (y -29.7 to -34.3 cm); the right arm
is its mirror image. Leaning its gripper 30 to 40 degrees toward the far side, though, an arm does reach mic B's
centre clear of the table and of the other arm at its ready pose. So one arm alone could in principle fetch the far
mic with a steeply leaned grip and then rig both in turn; the scripted one-arm runs do not try that, and we make no
claim that one arm cannot score 1.0. Both arms are used naturally: each fetches the mic on its own side. The task's
difficulty is knowing the ORTF geometry, telling the capsule end, and setting angle, splay, spacing and aim together.

## Demo and run limit

The scripted demo takes 26.5 s (530 control steps). Each mic is gripped at its centre across its two 24 mm side faces,
the fingers closing square to its axis (open 50 mm, commanded 16 mm, 8 mm narrower than the mic, over 1.5 s), with
the pad tips 4 mm above its underside; the gripper's yaw equals the mic's heading, and the mic turns with the
gripper. The left arm comes down round mic A (slowly for the last 30 mm), grips it, lifts it to 10 cm (tool point),
carries it over the plate in 2.5 s while turning it from -20 to +54 degrees, settles 0.5 s, lowers it over 1.5 s until
its underside is 1 mm above the plate, settles 0.5 s, opens over 1.5 s, goes straight up and parks at x 20, y +25 cm,
15 cm up. The right arm waits over mic B and picks it while the left arm works, and starts its own carry (+20 to -56
degrees) only once the left gripper has let go and gone up, so the two grippers are never over the bar together
(their tool points stay at least 32 cm apart). Joint 6 keeps at least 25 degrees to its limit; the tightest joint
margin is joint 4's 15.9 degrees.

Two trims are set from our earlier simulator runs. Parts carried in a gripper ended turned 0.6 to 1.7 degrees
counter-clockwise of the commanded yaw, so each mic is laid 1 degree clockwise of its target (`YAW_TRIM_DEG`); the
error has the same sign on both mics, so it cancels in the included angle and only turns the bisector. Parts also
landed about 4 mm short of their aim, so each place point is set 4 mm past the target along the carry, toward the
midline (`PLACE_LEAD`). Offline (no lag, no turn) the capsules end at (321.2, +80.8) and (320.4, -81.4) mm, 162.2 mm
apart, 110.0 degrees apart, the bisector 1.1 degrees off the mark: 1.0, success held for its 1 s at step 512. Landing
4 mm short and turned 1.2 degrees the pair is 170.0 mm and 110.0 degrees with the bisector 0.2 degrees off the mark;
up to 8 mm short it stays inside the 160-180 mm window (tested). If the GPU trace's `pair.separation_mm` is off 170 by
more than 4 mm, change `PLACE_LEAD` by half the difference; if `pair.bisector_deg` is more than 2 degrees from 0, change
`YAW_TRIM_DEG` by minus that. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: mic A back flat in its grey case pad centred
at x 25.0 cm, y +32.0 cm, along the pad, its grille end pointing 20 degrees clockwise of +x (toward the midline); mic
B in the other pad at x 25.0 cm, y -32.0 cm, its grille end 20 degrees anticlockwise of +x. Everything else is fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 27 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

An 80 x 280 x 8 mm aluminium plate as the stereo-bar plate, clamped on the midline; a 30 x 300 x 40 mm wooden block
painted dark blue as the riser, with a white plastic disc 12 mm across and 0.5 mm thick glued at its centre and six 16 mm wooden pegs 30 mm tall
standing on it as singers; two 100 x 50 mm pieces of 1 mm grey foam as the case pads. For the mics, two 70 x 24 x 24
mm blocks of about 80 g (a hollow aluminium case, or any block weighted to about 80 g), painted matte black with the
front 15 mm silver. Nothing moves or is powered except the mics. Score with a steel rule between the two grille
centres (7.5 mm behind each front face), a protractor on the two mic axes, a rule from the source axis to the
midpoint, and a photo from above showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
