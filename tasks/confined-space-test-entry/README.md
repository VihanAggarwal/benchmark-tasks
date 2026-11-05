# Fetch a tool from a tank: covers off, air tested top to bottom, probe kept in

Confined spaces such as tanks, pits and sewers kill workers every year, most often through a hazardous atmosphere,
and many of the dead are would-be rescuers. Before anyone goes in, the manways are opened to ventilate the space and
the air is tested with a detector's probe lowered through a manway, level by level from the top down, pausing at
each level for the detector to respond; while someone is inside, the detector stays in the space. Here a hand tool
has been dropped into a closed grey tank with two manways, each under a loose cover. With straight-down grips the
left arm reaches only the left cover, the gas probe and the left manway, and the right arm only the right cover, the
right manway, the tool and its pad (tilted grips reach further: see One arm alone). The scorer watches the whole trial: no gripper may go into the tank until both covers are off and the probe
has been held still for 2 s at each of three marked depths, top first, and the probe must be held in the tank
whenever a gripper is in it. Then the tool goes onto the white pad, the probe back into its holder and both covers
back on.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the covers', probe's and tool's states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, the test's order and timing, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes, a loose cylinder) and the standard `task_score` scorer.

The scorer's entry and monitoring rules need the grippers' state, which `yam_arena` does not pass to a task file's
hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a
snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step
and to start a new trial at each reset. It reads only state the simulator already has and changes nothing the
embodiment does. If it cannot read the arms it prints why once; the gripper rules are then not applied on those
steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange tool out of the grey tank and put it on the white pad. First lift both manway covers off onto their grey stands. Then lower the blue gas probe through the left manway and hold its tip still for 2 s at each of the three depths marked on the gauge, top first. No gripper may go below the tank's top before that, and the probe must stay held inside the tank whenever a gripper is in it. Finish with the probe in its holder, both covers back on and both grippers open." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/confined-space-test-entry/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange tool out of the grey tank and put it on the white pad. First lift both manway covers off onto their grey stands. Then lower the blue gas probe through the left manway and hold its tip still for 2 s at each of the three depths marked on the gauge, top first. No gripper may go below the tank's top before that, and the probe must stay held inside the tank whenever a gripper is in it. Finish with the probe in its holder, both covers back on and both grippers open." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/confined-space-test-entry/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `breach_objects`, `first_breach_step`,
`ventilated_step`, `band_steps`, `dwell_band`, `dwell_s`, `tip_mm`, `tip_speed_mm_s`, `probe_held_by`, `monitored`,
`entered`, `first_entry_step`, `min_tips_over_tank_mm`, `tool_out_step`, `covers_back`, `cover_centre_off_mm`,
`probe_off_holder_mm`, `min_gripper_clearance_mm`, `arms_unavailable`, both TCPs, ...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Definitions:

- A gripper is **in the tank** when its finger-tip point (between the tips) is below the deck's top (66 mm above
  the table) over the tank's outline grown by 5 mm. The fingers are the lowest part of a gripper, and a cover is
  10 mm larger than its opening on every side, so lifting or replacing a cover never counts as going in.
- A cover is **off** when its outline seen from above no longer overlaps its opening. The space is **ventilated**
  from the first step on which both covers are off.
- The probe's **tip** is the centre of its lower end; it is **in the column** when it is inside the left opening's
  outline seen from above.
- A **test at one depth**: once the space is ventilated, the tip in the column, its height inside that depth's
  band (top 45-56 mm, middle 28-39 mm, bottom 11-22 mm above the table, the gauge's red, yellow and green bands)
  and the tip moving slower than 5 mm/s (the length of the path it moved over the last 0.5 s, divided by 0.5 s:
  a gripped probe's instantaneous velocity in the simulator chatters at 10-45 mm/s while it stays put to 0.1 mm),
  without a break, for 2 s. The **test is complete** when the top, middle
  and bottom depths have been tested in that order; a depth counts only after the one above it.
- The probe is **held** when a gripper's TCP is within 20 mm of the point on the probe's axis 25 mm below its
  upper end and that gripper's opening is under 0.5; it is **monitoring** when it is held with its tip in the column
  no higher than the deck's underside, 58 mm above the table (inside the tank, below the 8 mm deck).

Rules that hold for the whole trial; one broken at any moment scores 0, whatever follows:

1. a gripper is in the tank while the test is not complete;
2. a gripper is in the tank while either cover is not off;
3. a gripper is in the tank while the probe is not monitoring;
4. the probe is not held while its tip is in the tank (below the deck's top over the tank's outline);
5. a cover, the probe or the tool falls (moves down faster than 0.25 m/s: a free drop of more than about 3 mm),
   tips over (more than 30 degrees) or leaves the table (lowest point more than 10 mm below the table top).

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Tool flat on the pad, probe upright in its holder, both covers back on, both grippers open and clear | 1.0 |
| Tool flat on the pad, anything else not as for 1.0 | 0.75 |
| Tool lifted out of the tank at some point, not on the pad at the end | 0.5 |
| Test complete, tool never out of the tank | 0.25 |
| Anything else | 0 |

'Flat on the pad': a 60 x 24 mm face down within 10 degrees, lowest point within 3 mm of the pad's top, centre over
the pad. 'In its holder': within 10 degrees of upright, lowest point within 5 mm of the table, tip within 8 mm of the
socket's centre. 'Back on': tilted under 5 degrees, lowest point within 3 mm of the deck's top, centre within 10 mm
of the opening's centre and the whole opening under the cover's real (turned) outline. 'Open and clear': opening
0.9 or more (fingers at least 84.5 mm apart) and each gripper's finger zone at least 30 mm from both covers, the
probe and the tool. The scorer cannot see which way a gripper is turned, so its finger zone is a flat-ended round
zone about its axis, from the finger-tip point 38 mm up the gripper, out to the fingers' outer corners (half the
finger gap plus the fingers' 10 mm thickness, and half their 40 mm width: 60 mm fully open). 'Out of the tank': the tool's lowest point more than 5 mm above the deck's top with its centre outside the
tank's outline.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with the covers, probe
and tool still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised from rigstate's reset hook, or, as a
fallback, when the simulator puts every loose object back on its starting place (within 4.5 cm, which covers a
seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0 for that step and leaves
the record of the trial as it was. The covers and the tool are plain boxes and the probe a plain cylinder, so every
tilt is folded (either broad face, or either end, down).

Unintended strategies earn nothing, as the tests show: going straight in for the tool after taking the covers off
breaks rule 1 (0); a rushed test (1 s at each depth), a skipped depth, the depths in the wrong order, a probe held
still beside the column, bobbing in the band, or a test begun before the second cover is off does not complete the
test, so going in breaks rule 1 (0); dropping the probe into the tank to "monitor" breaks rule 4 and, once a gripper
goes in, rule 3 (0); lifting the probe's tip above the deck's underside, or out of the left manway's column, while
the right gripper is in breaks rule 3 (0); putting a cover back while a gripper is in breaks rule 2 (0); a tool let go
above the pad falls (0). Going in the very step the bottom test completes is allowed. Leaving a cover on its stand,
the probe beside its holder, or a gripper closed or less than 30 mm above a cover at the end is 0.75.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) takes the right cover off and goes into the tank for the tool with the
left cover still on and no test done: rules 1 to 3 at once, 0. `DEMO_IDLE_ARM=right` (the left arm alone) takes the
left cover off and holds the probe still at the three depths, but the space is never ventilated (the right cover
stays on), so the test never starts: 0. With straight-down grips each arm reaches only about 10 cm past the
midline, and the covers, manways and the probe's holder are 18 cm either side of it. With steeply tilted grips
(15 degrees or more) an arm can reach the near edge of the other side's cover, the probe in its holder and the left
manway's column from the other side (offline kinematics), so one arm alone could in principle slide the far cover
off its opening, then lift its own cover, take the probe and run the test: 0.25 at most. It cannot score more,
whatever it reaches: the tool can only come out with a gripper in the tank, and while one is, the probe must be held
in the left manway's column by a gripper; the gripper holding it there cannot also take the tool, which lies under
the right manway 36 cm away. The fetch therefore needs both arms. (Offline predictions; the GPU runs are recorded in
`task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 88.0 s (1,760 control steps). Every grip closes the fingers over 1.5 s, commanded 8 mm
narrower than the part, after a 0.5 s settle, and every final descent is a separate slow move over the last 2 cm.

- Both arms at once (0-18 s): each comes down over its cover with its fingers closing along y (finger tips 7 mm
  above the deck), grips it across its 70 mm, lifts it straight up to 13 cm, carries it to its stand, lowers the last
  2 cm over 1.5 s, lets go 2 mm above the stand and goes up. Both covers are off from step 204 (10.2 s).
- Left arm (18-49.5 s): comes down over the probe in its holder, grips it 25 mm below its top (its tip then hangs
  75 mm below the tool point), lifts it straight up out of the socket, carries it with its tip 80 mm above the table
  (14 mm over the deck) to the left manway, lowers the tip to 50 mm (top band) and holds 4.5 s, to 33 mm (middle)
  and holds 4.5 s, to 16 mm (bottom, 8 mm above the tank floor) and holds 4.5 s. Each depth is aimed 5 to 6 mm
  inside its 11 mm band, and the finger tips stay 12 mm above the deck. Offline the three tests complete at steps
  702, 822 and 942 (47.1 s); the tip's speed while in a band peaks at 3.4 mm/s as it comes to rest.
- Right arm (18-58.5 s): with its fingers turned to close along x and opened to 74 mm, it waits over the right
  manway until 0.5 s after the left arm's last test waypoint ends, then comes down through the manway (pads 7 mm
  clear of each end of the tool and their outer faces 15 mm inside the opening's 120 mm length, about 8 mm where the
  fingers, which step out about 6.5 mm above the pads, pass the deck; across, the fingers are 27 mm wide at the pads
  and about 35 mm at the deck, about 7 mm clear each side in its 50 mm width; in the tank from step 1,020, 51.0 s,
  3.9 s after the test completes), grips the tool by its ends with its finger tips 6 mm above the tank floor, and lifts it
  straight up and out (its finger tips are above the deck from step 1,150). The left arm holds the probe at the
  bottom depth throughout and for 0.5 s after the right arm's lift ends (monitoring until step 1,200, 50 steps after
  the right gripper is out).
- Then the right arm carries the tool to the pad (out of the tank's outline from step 1,201) and lets go 2 mm above
  it, while the left lifts the probe out, carries it back over its holder, lowers it from 5 mm above the socket's
  walls over 2 s and lets go with the tip 2 mm above the table. Each arm then takes its cover from its stand,
  carries it back over its manway, lowers the last 2 cm over 1.5 s, lets go 2 mm above the deck and goes up to 15 cm
  (finger tips 51 mm above the cover).

The TCPs stay at least 36 cm apart and the arms' collision capsules at least 16 cm apart. In the tank the right
gripper's housing is about 30 mm and its finger carriages about 18 mm above the deck; its offline collision capsule,
which is rounder and larger than the housing, comes within 3.7 mm of the deck at the manway's edges, and its fingers
stay about 7 to 8 mm clear of the opening's edges (worth a look in the simulator). The lowest joint-limit margin is 10.8 degrees (left wrist, joint 4).
Offline, success has held for its 1 s at step 1,832 (the scripted policy stops at step 1,881). The run limit is
2,400 steps (2 minutes).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both covers flat on the deck, centred over
the manways, long side along x; the probe upright in its holder; the tool flat on the tank floor straight under the
right manway, long side along x. The tank, stands, holder, gauge and pad are fixed.

## Runtime per trial

About 3 minutes on an L40S once Isaac Sim is running (the demo is 91 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

Nothing is powered and no gas is used. A tank glued up from 8 mm grey-painted steel or aluminium plate (or MDF),
160 x 480 mm and 66 mm tall, with two 120 x 50 mm manway openings in its top and a clear acrylic front wall, fixed to
the bench; two 140 x 70 x 20 mm dark grey polyethylene plates as covers; two grey 150 x 90 x 5 mm rubber mats as
stands; a 16 mm aluminium rod, 100 mm long, bored to 50 g and painted blue, as the probe, in a black plastic holder
with a 24 x 24 mm square socket and 25 mm walls; a 10 x 10 mm white post, 66 mm tall, with red, yellow and green
bands at 45-56, 28-39 and 11-22 mm, standing against the front wall in line with the left manway; a 60 x 24 x 20 mm
aluminium block painted orange as the tool; and a 100 x 100 mm white plastic mat as the pad, each placed as in
`task.json`'s `setup`. Film from above and through the front wall to judge the depths, the order and when a
gripper goes below the deck.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
