# Change a dressing with one clean hand and one dirty hand

Dressing changes in wound care often use a "clean hand / dirty hand" (non-touch) technique: one hand handles only
the soiled dressing and the other only the sterile supplies, so a contaminated glove never touches the sterile
pad. The soiled dressing goes into the waste bin before the new pad comes over the wound, nothing soiled passes
over anything sterile, and the sterile pad touches nothing but its tray and the wound. Here a stained used
dressing lies on a pink wound mark between the two arms, a white clean pad lies in a blue sterile tray in front of
the left arm, and a yellow waste bin stands in front of the right arm. The right gripper is the dirty hand, the
left the clean hand. The scorer watches both grippers and both pads at every control step, not only the end state,
so the order and the hand that does each step matter as much as where things end up.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two pads' states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The scorer's gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the gripper rules are then not applied on those steps, and the
trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use only the right gripper for the stained used dressing and only the left gripper for the white clean pad, and keep the right gripper away from the clean pad. Put the used dressing in the yellow waste bin, then put the clean pad on the pink wound mark. Apart from the left gripper, the clean pad may touch only its blue tray and the wound mark, and it must never pass over or under the used dressing." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/clean-dirty-hand-dressing/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use only the right gripper for the stained used dressing and only the left gripper for the white clean pad, and keep the right gripper away from the clean pad. Put the used dressing in the yellow waste bin, then put the clean pad on the pink wound mark. Apart from the left gripper, the clean pad may touch only its blue tray and the wound mark, and it must never pass over or under the used dressing." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/clean-dirty-hand-dressing/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `min_left_to_dressing_mm`,
`min_right_to_pad_mm`, `dressing_in_bin`, `pad_on_mark`, `pad_centre_off_mm`, `pad_tilt_deg`,
`pad_bare_corner_mm`, `opening`, `arms_unavailable`, both TCPs, ...). Videos: `inspect-robots video <log>.json`.
The scorer keeps one trial's record per process and starts it over at every reset, so run one trial per process
or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer cannot see which way a gripper is turned, so
each gripper's fingers count as a round zone about its axis, out to the fingers' outer corners (57 mm fully open,
36 mm closed on a 50 mm pad), from the finger tips to 38 mm back up the gripper. Six rules hold for the whole trial; one
broken at any moment scores 0 whatever follows:

1. the left (clean) gripper's finger zone comes within 5 mm of the used dressing;
2. the right (dirty) gripper's finger zone comes within 30 mm of the clean pad;
3. the clean pad touches the used dressing (within 3 mm), or one is carried over the other (their outlines seen
   from above overlap at any height);
4. the clean pad comes over the wound mark (outline seen from above) before the used dressing is in the bin;
5. a corner of the clean pad touches the bare table (less than 1 mm above it, or below it) anywhere but over the
   tray's floor or the wound mark;
6. the clean pad touches the bin.

| Outcome | Score |
|---|---|
| Any rule broken at any moment, or the used dressing ends off the table | 0 |
| Used dressing in the bin, clean pad flat on the wound mark within 10 mm of its centre, both grippers let go | 1.0 |
| Used dressing in the bin, clean pad flat on the mark within 25 mm, but further off than 10 mm or a gripper not let go | 0.75 |
| Used dressing in the bin, the clean pad anywhere else it is allowed (in its tray, held, tilted on the mark) | 0.5 |
| Used dressing off the wound mark but not in the bin | 0.25 |
| Anything else (the used dressing still on the wound mark) | 0 |

A gripper has let go of the item it handles (the left the clean pad, the right the used dressing) when it is open
(opening 0.9 or more, fingers at least 84.5 mm apart) or its finger zone is more than 10 mm from that item, so a
gripper closed again once it is clear changes nothing, while one still closed round its item scores 0.75.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both pads still
for 1 s. A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts both pads
back on their starting places (within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator
state with a non-finite number scores 0 for that step and leaves the record of the trial as it was. Both pads are
plain boxes, so either 50 x 50 mm face may be down.

Unintended strategies earn nothing, as the tests show: the left gripper lifting the used dressing aside to do the
whole job breaks rule 1 (0); the right gripper taking the clean pad, even from a hand-over, breaks rule 2 (0);
shoving the used dressing off the wound with the clean pad, or setting the pad on the dressing, breaks rule 3 (0);
carrying the used dressing over the sterile tray breaks rule 3 (0); bringing the clean pad over the wound while the
dressing is still held above the bin, or balanced on its rim, breaks rule 4 (0); dragging the clean pad across the
table, setting it down to change grip, or letting it bounce off the mark onto the table breaks rule 5 (0); sweeping
the used dressing off the table scores 0. Leaving the clean pad gripped on the mark, or the used dressing gripped
inside the bin, is 0.75.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) bins the used dressing and leaves the clean pad in its tray: 0.5.
`DEMO_IDLE_ARM=right` (the left arm alone) brings the clean pad over the wound mark with the used dressing still on
it: 0. Under the rules no single arm can finish: the left may not touch the used dressing and the right may not
come near the clean pad. Reach alone already splits the ends of the job (only the left arm reaches the tray, only
the right the bin), but both arms reach the wound mark, and without the rules one arm could clear the wound and
place the pad there; the rules decide which gripper may work there on which item. (Offline kinematic predictions;
the GPU runs are recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 18.5 s (370 control steps). Both arms grip across the pads' 50 mm (fingers commanded 8 mm
narrower, over 1.5 s), closing along x, with the finger tips 4 mm above the pad's bottom. The right arm lifts the
used dressing off the wound mark, carries it 10 cm up over the bin, lowers it to 12 mm above the rim and lets it
drop in, then goes up over the bin and stays there, 25 cm from the wound mark. Meanwhile the left arm comes down
into the tray (the open fingers 15 mm inside its end walls), grips the clean pad, lifts it and holds it over the
tray; only when the right gripper has opened over the bin (10.7 s) does it carry the pad over the wound mark, come
down to 2 cm above it, lower the last 2 cm over 1.5 s, let go with the pad 2 mm above the mark and go up. The TCPs
stay at least 22.8 cm apart. Offline, success has held for its 1 s at step 361. The run limit is 1,200 steps
(60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the used dressing back flat and centred on
the wound mark and the clean pad back flat and centred in the tray, sides parallel to x and y; the wound mark, tray
and bin are fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 18.5 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 70 x 70 mm pink silicone wound-care practice pad, 2 mm thick, taped to the bench; two gauze pads folded and
taped into 50 x 50 x 16 mm blocks, one stained with diluted tea and damped to about 20 g, the other dry (about
10 g) and white; a blue plastic tray with 15 mm walls round a 140 x 70 mm floor (or a printed rim, open at the
bottom, clamped to the bench); and an open yellow box with 40 mm walls round a 90 x 90 mm floor as the waste bin,
also clamped. Lay a 30 mm scale bar on the tray's rim and film from above and from the side to judge the rules.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
