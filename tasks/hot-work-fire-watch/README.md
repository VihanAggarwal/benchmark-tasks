# Weld only after clearing combustibles and posting a fire watch, and keep the watch afterwards

Hot work (welding, cutting, brazing) is governed by a permit procedure: combustibles within 35 ft (10.7 m) of the
work are moved away or shielded before any work starts; a fire watch, a second person with an extinguisher ready,
is posted for the duration of the work; the watch stays for at least half an hour after the work ends, because a
smouldering fire can break out after the torch is put away; and a lit torch is never set down except on its holder.
The US rule is OSHA 29 CFR 1910.252; NFPA 51B is the hot-work standard. Welders, cutters and brazers (O*NET
51-4121), pipefitters (47-2152) and maintenance workers (49-9071) work under it.

Here the right gripper is the welder and the left the fire watch. A pale-yellow hot-work mat, a disc 21 cm in radius
(the 35 ft radius at 1:50), lies on the table with a steel work plate and an orange weld mark at its centre. A
cardboard carton and an oily rag lie on the mat, one in each arm's half of the table. A blue torch stands in a holder
on the right and a red extinguisher in a holder on the left, both outside the mat. The robot must move both
combustibles off the mat, lift the extinguisher and hold it upright over the mat, and only then bring the torch to
the mark for 5 s of welding; it must keep the extinguisher there for 10 s after the torch leaves the plate, then
park both tools. The scorer watches every control step, so the order and timing matter as much as the end state.
The second gripper's role is the procedure's: the fire watch is a separate person whose hands are not on the torch,
and one gripper cannot hold the torch and the extinguisher at once.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, mass, position, threshold and colour, in one place |
| `scoring.py` | The scorer: plain functions of the four loose objects' states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step (unchanged copy) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the gripper hook, the demo and the exploits we tried, through a small kinematic stand-in (no simulator) |
| `tests/test_demo_plan.py` | Offline check of the scripted plan against the simulator's and station's limits (generic) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer.

The 'held' and 'let go' rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; 'held' is then taken as met on those steps, and the trace
counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is hot work. First move the cardboard carton and the rag completely off the yellow mat, and keep them off it. Then, as fire watch, lift the red extinguisher out of its holder with one gripper and hold it upright over the mat. Only while it is held there may the blue torch come near the steel plate: with the other gripper, hold the torch's tip on the orange mark for 5 seconds in total. Keep holding the extinguisher over the mat for 10 seconds after the torch leaves the plate, then put the torch and the extinguisher back in their holders and let go. Never put either one down anywhere else." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/hot-work-fire-watch/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is hot work. First move the cardboard carton and the rag completely off the yellow mat, and keep them off it. Then, as fire watch, lift the red extinguisher out of its holder with one gripper and hold it upright over the mat. Only while it is held there may the blue torch come near the steel plate: with the other gripper, hold the torch's tip on the orange mark for 5 seconds in total. Keep holding the extinguisher over the mat for 10 seconds after the torch leaves the plate, then put the torch and the extinguisher back in their holders and let go. Never put either one down anywhere else." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/hot-work-fire-watch/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `weld_s`, `hot_s`,
`post_watch_s`, `ready_best_s`, `tip_height_mm`, `tip_off_mark_mm`, `carton_gap_mm`, `rag_gap_mm`,
`ext_off_mark_mm`, `ext_tilt_deg`, `torch_held_by`, `ext_held_by`, `opening`, `arms_unavailable`, both TCPs, ...).
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator (from the `work` folder above `tasks/`):
`PYTHONPATH=tools/offline/yam_stub:tasks/hot-work-fire-watch python3 -m unittest discover -s tasks/hot-work-fire-watch/tests -v`
(the tests also find `tools/offline/yam_stub` by themselves).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Definitions:

- **Held**: a gripper's finger gap is no more than 6 mm wider than the part's diameter, and the point between its
  finger tips lies within 12 mm of the part's axis, between its two ends. Either gripper may hold either part.
- **Hot work**: the torch's tip (the centre of its lower end) is over the plate (inside its 120 x 120 mm outline) and
  at most 25 mm above the plate's top. **Welding the mark**: hot work with the tip within 15 mm of the mark's centre.
- **Clear**: no part of either combustible's outline seen from above (its world-aligned bounding box) is within 21 cm
  of the mark's centre: none of it is over the mat.
- **Watch posted**: the extinguisher is held, within 20 degrees of upright, its centre within 21 cm of the mark's
  centre (over the mat).
- **In its holder**: the part's lower end is over its own cup's floor and its lowest point is below the 25 mm rim.
  **Parked**: in its holder, within 10 degrees of upright, its lowest point within 5 mm of the table top.
- **Weld time**: time welding the mark, added up over the trial. **Post-work watch**: the time from the last step of
  hot work during which the watch stays posted without a break; any later hot work starts it again from zero.

Rules for the whole trial; breaking one at any control step scores 0, whatever follows:

1. hot work while the area is not clear;
2. hot work while the watch is not posted;
3. after hot work has begun, a combustible not clear (brought back onto the mat, or never moved);
4. the torch neither in its holder nor held for more than 0.25 s in a row (more than 5 control steps);
5. the same for the extinguisher;
6. a loose object's lowest point more than 10 mm below the table top (off the table).

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Weld time 5 s or more, post-work watch 10 s or more, torch and extinguisher parked, neither gripper holding either, area clear | 1.0 |
| Weld time 5 s or more and post-work watch 10 s or more, but a tool still held or not standing upright in its holder | 0.75 |
| Weld time 5 s or more, post-work watch under 10 s (the watch left early, or the trial ended first) | 0.5 |
| No full weld, but the area was clear and the watch posted together for 2 s without a break | 0.25 |
| Anything else (nothing done; area cleared but no watch; watch posted with a combustible still on the mat) | 0 |

The score is read at the last step; the trial ends early once the 1.0 conditions have held with all four loose
objects still (under 1 cm/s and 0.5 rad/s) for 3 s, longer than the demo's last release and retreat (1.5 s + 1.0 s).
A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts all four loose
objects back on their starting places (within 4.5 cm) in a single step. A simulator state with a non-finite number
scores 0 for that step and leaves the trial's record as it was.

### Why these numbers

| Threshold | Value | Why |
|---|---|---|
| Clearance radius | 21 cm | The standard's 35 ft (10.7 m) at 1:50; the mat's edge shows it, so a real trial needs no measuring |
| Watch within | 21 cm (over the mat) | The fire watch works inside the hot-work area; the same visible edge. The demo holds it 16.3 cm out |
| Hot-work height | 25 mm above the plate | A welding torch works a few mm to about 20 mm from the work; 25 mm also covers the arm's sag. Carried higher, the torch is in transit, not working |
| Hot-work area | the 120 mm plate | The work piece: hot work anywhere on it, not only on the mark, must meet the preconditions |
| On the mark | 15 mm | The mark is 20 x 20 mm: over it or within 5 mm of its edge. Sag and gripper tilt move the tip about 3 mm |
| Weld time | 5 s | A short tack weld; long enough that the watch must be held through a real stretch of work, short enough for the run |
| Post-work watch | 10 s | Stands in for the half hour (not to scale, as the run is 90 s); long enough that leaving at once is clearly different |
| Ready | 2 s | Partial credit for doing the preparation, not for carrying the extinguisher across the mat on the way elsewhere |
| Set-down grace | 0.25 s | A gripper needs more than 1 s to let go and grip again; 0.25 s only absorbs a one-step misreading of 'held' |
| Held: gap | diameter + 6 mm | Closed on the part (3 mm each side at most); a gripper open around the part does not count |
| Held: axis | 12 mm | The grip point of a gripper closed on a 28 or 44 mm cylinder lies on its axis; 12 mm allows the 27 mm pads to grip off centre |
| Watch tilt | 20 degrees | A portable extinguisher is used upright; the held extinguisher tilts 1-2 degrees with the sag |
| Parked tilt / floor | 10 degrees / 5 mm | Standing on the cup's floor, not leaning on a wall or perched on the rim (a 28 mm torch can lean about 20 degrees in its cup) |

### Exploits we tried (all in the tests)

- **Doing nothing**: 0. **Right arm alone** (`DEMO_IDLE_ARM=left`): it clears the rag and welds with the carton on the
  mat and no watch: 0 (rules 1, 2, 3). **Left arm alone** (`DEMO_IDLE_ARM=right`): it clears the carton and posts the
  watch, but the rag stays on the mat, so the area is never clear: 0.
- **Setting the extinguisher down near the mark** so the same gripper can take the torch: rule 5, 0. A gripper beside
  an extinguisher standing on the mat, open, is not holding it (0). One gripper cannot hold both parts.
- **Holding the extinguisher off the mat** (24 cm out) or **tipped 25 degrees** while welding: not a watch, 0.
- **Welding before the area is clear, or before the watch is posted**: 0. **Hot work 40 mm off the mark** (over the
  plate) without a watch: still hot work, 0, and it adds no weld time.
- **The torch in transit**: carried low (tip 20 mm over the plate) across the plate before clearing: hot work, 0.
  Carried 59 mm over it, as in the demo: not hot work.
- **Welding in many tiny bursts**: ten 0.6 s bursts add up to the 5 s and score 1.0 when every burst is watched; one
  more touch of the plate after the watch has gone is 0. A dab late in the post-work watch restarts it (0.5).
- **Pushing combustibles off the table**: rule 6, 0, even if they are put back. **Pushing a combustible off the mat**
  along the table is allowed: it ends clear (1.0 possible). Pushing one back onto the mat after the weld: rule 3, 0.
- **Setting the torch down on the table** and picking it up again: rule 4, 0.
- **Swapping roles** (left welds, right watches): allowed; nothing in the hot-work rule depends on which hand.
- **Stopping short**: the extinguisher or torch still held at the end, or parked but still gripped, or the
  extinguisher leaning in its cup: 0.75. The watch left 4 s early (8.6 s kept): 0.5. A 4.1 s weld: 0.25.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) moves the rag off the mat, takes the torch and welds with the carton still
on the mat and no fire watch: 0 at the first step of hot work (step 576). `DEMO_IDLE_ARM=right` (the left arm alone)
moves the carton off the mat and holds the extinguisher over the mat, but the rag is still on the mat, so the area is
never clear and there is no credit: 0. Under the rules no single arm can score more than 0.25: the weld needs the
torch held near the plate while another gripper holds the extinguisher, and the torch may not be put down. A single
arm that reached both combustibles (each lies 11-13 cm past the midline from the other arm, which the scripted demo
does not attempt) could clear the mat and post the watch for 0.25. (Offline kinematic predictions; the GPU runs go in
`task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 54.8 s (1,096 control steps). Every grip closes the fingers along x (yaw -pi/2 left, +pi/2
right), commanded 8 mm narrower than the part, over 1.5 s, after a 0.5 s settle; every final descent onto a part or a
surface covers the last 2-3 cm over 1.5 s. Both arms first move their combustible off the mat (the carton and rag are
gripped with the pad tips 4 mm above their bottom and set on the bare table, their nearest edge about 25.5 cm from
the mark). The left arm then grips the extinguisher by its top (the TCP 16 mm below the top), lifts it 45 mm, 20 mm
over its holder's rim, and holds it 16.3 cm from the mark at (15, 11) cm: the watch is posted at 24.3 s. The right
arm, which has meanwhile taken the torch by its top the same way, waits until 0.5 s after that, carries the torch
over the mark with its tip 59 mm above the plate, lowers it to 30 mm and then slowly to 12 mm, and holds it there:
hot work from about 28.8 s to 35.3 s, 6.3 to 6.8 s of it on the mark (from no sag to twice the estimated sag). It
lifts the torch straight up and parks it. The left arm holds the watch until 11.5 s after the plan's last hot work
(12.5 to 12.7 s in the offline runs, as the extinguisher takes a moment to cross the mat's edge), then parks the
extinguisher and lets go. Success (all still 3 s) is reached in the
offline stand-in from step 1,054 (52.7 s), so the trial ends at about 55.7 s.

The two TCPs never come within 7 cm of each other in x with less than 27.7 cm between them in y (the station needs
20 cm whenever they are within 5 cm in x), and are never closer than 16.3 cm. Every waypoint is at least 3.6 cm inside
the reach limits and 5 cm in front of the 10 cm minimum. The fire watch is placed 12 cm behind the mark in x so the
two grippers never work side by side at the plate.

Payload sag: the arm does not compensate a held weight (about 8 mm down at 0.3 kg and 0.3 m reach): about 2 mm for
the 80 g torch and 5 mm for the 200 g extinguisher, with 1-2 degrees of tilt. The weld height is commanded with the
tip 12 mm above the plate, so it is 'hot' for any sag from 0 to 12 mm and stays within 15 mm of the mark for any
sideways sag or tilt under 10 mm; the tools are lifted 20 mm over their rims and set back with their bottom 3 mm
above the cup's floor (with the sag they already stand on it). The offline tests replay the demo with no sag, the
estimated sag, twice it, and sideways sag either way: all score 1.0. On the GPU, read `tip_height_mm`,
`tip_off_mark_mm` and `ext_off_mark_mm` in `TASK_TRACE`.

The run limit is 1,800 steps (90 s). The procedure has 15 s of mandated timing (5 s weld, 10 s watch) on top of five
pick-and-place moves; the demo uses 55 s of the 90.

## Novelty

The closest tasks on the board are "Store oxygen apart from fuel-gas cylinders" (separation distances between
stored items: static, no hot work, no second person), "Park the hot iron on its rest" and the hot-plate tasks (a heat
hazard handled with one arm), and "Bring a storage shelf under its sprinkler clearance line" (a fire-code clearance,
checked at the end). None of them has a precondition that gates an action in time (no hot work until the radius is
clear and the watch posted), a second gripper whose job is to stand by with safety equipment during the other's work,
or a duration rule after the work (the post-work watch). The scorer checks the procedure's order and timing at every
control step; the end state alone (everything back in its holders, combustibles off the mat) is reached just as well
by a run that broke every rule, and that run scores 0.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the torch and extinguisher upright and centred
in their holders, the carton on the mat centred at x 33.0 cm, y +11.0 cm and the rag on the mat centred at x 25.0 cm,
y -13.0 cm, sides parallel to x and y; the mat, plate, mark and holders are fixed.

## Runtime per trial

About 2 to 2.5 minutes on an L40S once Isaac Sim is running (the demo is about 56 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A pale-yellow disc of welding blanket 42 cm across (or a painted circle) taped to the bench; a 120 x 120 x 6 mm steel
plate taped on its centre with a 20 x 20 mm orange square painted at its centre; two holder cups printed in PLA (25 mm
walls round 40 x 40 mm and 56 x 56 mm openings), clamped; a 28 mm PVC rod 100 mm long (80 g) as the torch; a 44 mm
plastic tube 110 mm long with end caps, weighted with sand to 200 g, painted red, as the extinguisher; a small closed
cardboard carton (40 x 60 x 40 mm) and a cotton rag folded and stitched into a 50 x 50 x 30 mm block. Nothing is lit.
Stand a card beside the plate with a 25 mm mark for the tip height, and film from above and from the side.

## Results in simulation

To be measured on the GPU (Isaac Lab-Arena, `-E object_jitter=0 -E control_hz=20`, one trial each). Not yet run:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline prediction 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (offline prediction 0) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (offline prediction 0) |
| Dream-1 | to be measured on the GPU |
