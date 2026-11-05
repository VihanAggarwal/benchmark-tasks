# Steer a laser onto a detector with two mirrors, blocking the beam while they move

Photonics technicians steer laser beams with mirrors every day, and laser-safety practice (ANSI Z136.1 and the lab
procedures built on it) says to block the beam before inserting or moving an optic and to keep every beam
terminated on the table: a mirror swung through a live beam throws a stray reflection across the room. Here a
low-power laser on the left of the table is always on, and its beam runs 4 cm above the table into a beam dump. Two
loose mirrors must steer it, off both, onto the centre of a yellow detector on the right, without the beam ever
leaving the table and without a mirror ever moving while the beam is on it; an orange beam block keeps the beam off
the mirrors while they go in. The scorer traces the beam at every control step, so how the scene is changed matters
as much as where things end up. The beam is not drawn in the simulation: the two red lines mark its starting path
and the two black lines the detector's axis.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: a 3-D ray trace of the beam through the objects' boxes each control step, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of the beam trace, every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The 'let go of everything' condition needs the grippers' state, which `yam_arena` does not pass to a task file's
hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a
snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step
and to start a new trial at each reset. It reads only state the simulator already has and changes nothing the
embodiment does. If it cannot read the arms it prints why once; 'let go' is then taken as met on those steps, and
the trace counts them (`arms_unavailable`). The beam rules use only the objects' states.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The laser is always on: its beam runs 4 cm above the table between the two red lines. Use the two silver mirrors to steer the beam onto the centre of the yellow detector's front face, reflecting it off both mirrors. Never move a mirror while the beam is on it: put the orange beam block in the beam first and take it out once both mirrors are set. Never let the beam leave the table, and let go of everything at the end." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/laser-two-mirror-steer/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The laser is always on: its beam runs 4 cm above the table between the two red lines. Use the two silver mirrors to steer the beam onto the centre of the yellow detector's front face, reflecting it off both mirrors. Never move a mirror while the beam is on it: put the orange beam block in the beam first and take it out once both mirrors are set. Never let the beam leave the table, and let go of everything at the end." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/laser-two-mirror-steer/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`beam_end`, `reflections`, `lit`, `det_off_mm`, `det_dy_mm`,
`det_dz_mm`, `beam_points_mm`, `breaches`, `first_breach_step`, `max_lit_move_mm`, `max_lit_turn_deg`,
`first_lit_step`, each mirror's yaw and tilt, `grippers_let_go`, `opening`, `arms_unavailable`, the three loose
objects' poses, both TCPs, ...). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per
process and starts it over at every reset, so run one trial per process or rely on the reset; the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## The beam, seen from above

The beam is not rendered, so here is the scene with the path the demo sets up (x forward, y to the left; cm):

```
 x
 50 |  [dump]                                         beam at the start: laser -> dump
    |    ^
    |    ^                                    (left arm base y +30.5, right arm base y -30.5, both at x 0)
 32 |  [M1]> > > > > > > > > > > > > > > > > [M2]     M1 at +45 deg turns the beam to -y,
    |    ^      (beam 4 cm above the table)    v      M2 at -45 deg turns it to -x
 22 |  [block]  <- in the beam while the       v
    |    ^         mirrors go in, out after   [det]   detector face at x 21, its centre 4 cm up
 17 |    ^                                    [###]   backstop behind it
 12 |  [laser]                                        mirrors start at (32, +38) and (24, -36),
    +----------------------------------------------- y   the block at (20, +36)
       +24                    0               -16
```

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step the beam is traced in 3-D from the
laser's aperture (x 14 cm, y +24 cm, 40 mm up, along +x) through every object's true box: it reflects off a
mirror's glass, the 40 x 40 mm window in the middle of either 60 x 60 mm face (from the mirror's full orientation, so
a tilted mirror sends it up or down), and stops on anything else it meets (the block, the dump, the backstop, the
laser, the detector, a mirror's 10 mm rim round the glass or its 20 mm edge, the table top). The arms are not part of
the beam model. Three rules hold for the whole trial; one broken at any moment
scores 0 whatever follows:

1. the beam is stray: it leaves the table's outline without being stopped;
2. a mirror the beam is on (its glass, rim or edge) moves more than 3 mm or turns more than 1 degree from where it
   stood at the last control step before the beam came onto it, checked at every step the beam is on it and at the
   first step after the beam leaves it, so a mirror swung into or out of the beam between two control steps counts
   (a mirror kept dark behind the block may be moved freely; when the beam comes back onto it, a new reference is
   taken);
3. a mirror or the block falls off the table.

| Outcome (the beam at the end) | Score |
|---|---|
| A rule broken at any moment | 0 |
| Off both mirrors onto the detector's front face within 10 mm of its centre, both grippers let go | 1.0 |
| Off both mirrors onto the detector's front face, but further than 10 mm or a gripper not let go | 0.75 |
| Off both mirrors, stopped anywhere else on the table (backstop, dump, block, table top) | 0.5 |
| Off one mirror only, stopped on the table | 0.25 |
| Off no mirror (the block still in the beam, or no mirror in it) | 0 |

A gripper has let go when it is open (opening 0.9 or more, fingers at least 84.5 mm apart) or its finger zone
(round about its axis, out to the fingers' outer corners, from the tips to 38 mm up) is more than 10 mm from both
mirrors and the block. The score is read at the last step; the trial ends early once the 1.0 conditions have held
with both mirrors and the block still for 1 s. A new trial is recognised from rigstate's reset hook, or, as a
fallback, when the simulator puts both mirrors and the block back on their starting places (within 4.5 cm, which
covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number, or a missing object,
scores 0 for that step and leaves the record of the trial as it was.

The unintended strategies we tried earn nothing (see the tests): skipping the block and lowering (or dropping, or
sliding) the left mirror into the live beam at exactly the right angle breaks rule 2 even though the beam is never
stray (0); sliding it in only until the beam is just inside its face leaves the beam on the 10 mm rim (no
reflection, 0), and going on to the glass moves the lit mirror at least 7 mm at 45 degrees (10 mm x cos 45, 0);
swinging a mirror into or out of the beam between two control steps counts as moving it in the beam (0); setting the
left mirror before the right one is in place and then taking the block out sends the beam off the right edge of the
table (0); leaving the block in after setting both mirrors is 0 (no reflection); one mirror turned to send the beam
straight onto the detector's face is 0.25, not 1.0 (the path must use both mirrors); moving the block onto the last
leg in front of the detector is 0.5; holding a mirror still in the beam with a closed gripper is 0.75. A gripper
closed again far from everything changes nothing.

## One arm alone

The right arm cannot reach the beam (at table height it reaches at most y +19 cm, at x 10 cm, and less further out;
the beam runs at y +24 cm from x 14 cm), and the left arm cannot reach the right mirror's starting place (y -36 cm;
it reaches y -13 to -15 cm at x 22-26 cm), so a path off both mirrors needs both arms (a hand-over of the right
mirror is also two arms).
`DEMO_IDLE_ARM=right` (the left arm alone) puts the block in, sets the left mirror and takes the block out: the beam
runs off the right edge of the table, 0. `DEMO_IDLE_ARM=left` (the right arm alone) sets the right mirror; the beam
still ends on the dump, 0. The most one arm can earn is 0.25 (the left arm aiming the left mirror at a stop on the
table). (The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 43.6 s (872 control steps). Every part is gripped straight down with the tool point 50 mm
above the table (the pads hold the top 22 mm of the 60 mm parts), fingers commanded 8 mm narrower than the part over
1.5 s; the mirrors are gripped across their 60 mm width, on their 20 mm edges, never on the reflecting faces. Each
pick-and-place lifts the part 70 mm clear, carries it at a tool height of 12 cm, comes down to 2 cm above its place,
lowers the last 2 cm over 1.5 s and lets go with the part 2 mm above the table. The left arm puts the block in the
beam at x 22 cm (in by 8.8 s), then carries the left mirror from its starting place, turning it from 0 to +45
degrees on the way, and sets it with the beam on the middle of its face at x 32 cm (released at 26.4 s). Meanwhile
the right arm sets the right mirror at -45 degrees with the beam's path on the middle of its face at x 32 cm,
y -16 cm (released at 12.2 s) and parks up over its starting place. The mirrors are dark all this time. The left arm
then lifts the block straight out (the beam reaches the detector at 34.1 s), puts it back on its starting place, lets
go and goes up. The TCPs stay at least 42 cm apart. Offline, success has held for its 1 s at step 844; the scripted
policy would stop at step 933. The run limit is 2,000 steps (100 s); pass `--max-steps 2000`.

The detector is 51 cm of beam path from the left mirror, so the spot moves about 9 mm per 0.5 degree of that
mirror's yaw and 2 mm per 0.5 degree of the right mirror's; a mirror shifted 1 mm across the beam moves the spot 1
mm. A 1.0 needs the left mirror within about 0.55 degree; from about 1.0 degree the beam meets the right mirror's
rim instead of its glass (0.25), and at 1.45 degrees one way it misses the right mirror and leaves the table
(stray). The right mirror can be 2.6 degrees off. In the simulator the demo is deterministic, so if the trace shows
a steady offset at the end (`det_dy_mm`, positive toward +y; `mirror_left_yaw_deg`, target +45;
`mirror_right_yaw_deg`, target -45), tune only `M1_PLACE` / `M2_PLACE` in `task.py`: turning the left mirror by +0.1
degree moves the spot +1.8 mm in y, the right mirror by +0.1 degree -0.4 mm; moving the right mirror +1 mm in x
moves it -1 mm; the beam stays on the right mirror's glass while it is within 14 mm of its place across the beam. An
arm sags a millimetre or two under an 80 g mirror; the 3 mm and 1 degree of rule 2 apply only while the beam is on a
mirror, which in the demo is only after both are set and let go.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both mirrors and the block back on their
starting places, upright, sides parallel to x and y (mirror faces toward +x and -x); the laser, dump, detector,
backstop and lines are fixed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is 44 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

A Class 2 (under 1 mW) red alignment laser module in a 40 x 40 x 60 mm housing clamped to the bench with its beam
40 mm up (a visible laser keeps a real trial safe and lets the spot be seen); a black anodized box or a stack of
black card as the beam dump; two 60 x 60 x 20 mm mounts of about 80 g in matte silver plastic, each with a
40 x 40 mm front-surface mirror set in the middle of each 60 x 60 mm face and a 10 mm rim round it; a 40 x 40 x 60 mm
block of wood painted orange as the beam block; a photodiode or a yellow card target with a 20 mm circle on its face
as the detector, with a black board behind it; red and black plastic strips 10 mm wide and 2 mm thick (or a few
layers of tape) for the lines. Fix white cards along the table edges and film from above: any spot on a card is a stray
beam. Wear the eyewear the laser's class calls for.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
