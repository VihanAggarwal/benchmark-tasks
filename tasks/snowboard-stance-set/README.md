# Mount snowboard bindings for a 40 cm duck stance and screw them down

Before a rental snowboard goes out, the technician mounts its bindings to the rider's stance: the width from the
shop chart (here 40 cm, a child's), centred on the board's reference mark, with the discs set for a beginner's duck
stance, and each disc screwed through its four slots into the board's threaded inserts. Everything here is real
size. A fixed child's board top lies across the table, nose to the left, heel edge toward the robot, with sixteen
M6 inserts on the standard 4 x 4 pattern (40 mm grid) and a red cross at its middle. Two loose child's bindings lie
parked on the board's ends; their discs are already preset and fixed in the baseplates, +15 degrees on the nose
binding and -15 on the tail binding. Eight M6 binding screws with washers stand in two holders beyond the board's
toe edge.

A disc's four slots sit over four inserts only with the disc square to the board (within about 10 degrees: the slots
are long), and that puts the baseplate at its preset angle (the scorer asks for 3 degrees); the 40 cm stance centred
on the cross puts both discs on the middle insert squares. Each screw has to be fed through a slot into its insert
and turned while it is seated. The score is the end state: both bindings aligned, all eight screws driven, both
bindings clamped, nothing held or moving.

This is the revision of PR #857 asked for in review ("avoid a simple pick and place; include mounting-hole alignment
and fastening"). `REVISION.md` maps each sentence of the review to what changed.

## Why it is hard

- **Carrying and landing the binding.** The 0.35 kg binding is carried by its 10 mm highback, so it hangs off the
  gripper: 3.4 N and 0.25 N m on the arm, which a static calculation from the arm's joint stiffness puts as 12.5 mm
  of sag toward the robot and the toe 2.5 degrees down. It has to land flat (underside within 1.5 mm of the board,
  tilted 3 degrees or less), at its angle within 3 degrees, with each 10 mm slot within 4.0 mm of an 8 mm insert
  across the slot.
- **Feeding the screws.** Each 6 g screw is picked by its 12 mm head, carried over the binding's 58 mm sidewall tops
  and lowered so that its 5 mm shank passes through a 10 mm slot (2.5 mm of play a side) into an 8 mm bore (1.5 mm a
  side) without the head or washer catching.
- **Turning it in regrips.** A full clockwise turn of the 12 mm round head needs at least two 180 degree strokes,
  opening and turning back between them, without lifting the washer 3 mm (which lets go of the thread and restarts the
  count) or tilting the screw past 10 degrees. In each of the first two GPU runs one screw fell short (278 and 353 of
  the 360 degrees): the pads popped up off its head, and once they lifted the screw with them.
- **Nothing is chosen.** The insert pattern, the presets and the parked poses are the same in every trial and the
  instruction states the goal; the difficulty is the placement and the screws, not a decision.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, the thread and clamp rules (`on_step`), `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The rules (`Fastening`: engagement, turn count, drive, clamp) and the scorer (`Tracker`: alignment, seating, release and stillness windows, tiers) as plain NumPy functions |
| `mechkit.py` | Our mechanism kit, version 2, a copy of `tools/kit/mechkit.py`: compound rigid bodies (the board with its bores, the bindings with turned baseplates and slotted discs, the screws, the holders), tight-fit contact settings and the pose-rewrite welds |
| `rigstate.py` | A copy of `tools/kit/rigstate.py`: reads both grippers each control step; its fail-closed `released` is the scorer's release check |
| `tests/test_scoring.py` | Offline tests: geometry, the rules, every tier, the zero conditions, exploits, unreadable arms, resets, the offline stubs, finger clearances, the demo's structure, the text |

Third-party assets: none. Every part is built from boxes and cylinders in `task.py` and `mechkit.py`.

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install.

`yam_arena` task files declare single boxes, cylinders and spheres, so `mechkit.py` (imported by `task.py`) builds
the compound parts and runs the per-step rules; `rigstate.py` wraps the embodiment's step and reset once, at import,
to read both arms. Both read only state the simulator already has. The welds rewrite the welded part's root pose and
velocity every control step (Isaac Lab's `write_root_pose_to_sim_index` / `write_root_velocity_to_sim_index`, as the
accepted tasks snapfit-close and bayonet-mate do); nothing else is written.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount this rental snowboard's bindings for a 40 cm duck stance centred on the red cross, heels on the edge nearest you, and screw both down." \
  --scorer task_score --max-steps 6000 --store-frames --log-dir ~/yam-eval/snowboard-stance-set/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount this rental snowboard's bindings for a 40 cm duck stance centred on the red cross, heels on the edge nearest you, and screw both down." \
  --scorer task_score --max-steps 6000 --store-frames --log-dir ~/yam-eval/snowboard-stance-set/dream
```

Run with `-E object_jitter=0`: the layout is fixed, and the welds hold parts where the rules put them.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check). `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: for each binding its `role`, `origin`, `angle_deg`,
`angle_err_deg`, `tilt_deg`, `z_above_mm`, `slots_over` (the insert under each slot, or null), `square`,
`aligned`, `seated`, `driven`, `clamped`, `fastened`; then `tier`, `score`, `released` (this step), `released_1s`,
`still_1s`, `moving`, `lost`, `zone_mm` (the nearer finger zone to any part), `arms_readable`, `W_mm`, `C_mm`, and per
screw the rule's `binding`, `slot`, `insert`, `turn_deg` and `driven`. Each drive and clamp also prints a
`[snowboard] drive ...` / `clamp ...` line to stderr, and mechkit prints each weld firing. `MECHKIT_TRACE=<file>`
adds mechkit's own per-step record. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`, and from the repository root
`python3 tools/offline/check_task.py tasks/snowboard-stance-set --require-json --strict --one-arm`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`, its thresholds `layout.py`.

**Rules during the trial** (in a mechkit `on_step` callback, before each step's physics). A screw is *engaged* once
its shank has gone through one of a binding's slots into an insert (its tip 5 mm or more down the bore), it is
upright within 10 degrees and its washer is within 1 mm of the disc's top. It stays engaged until the washer is 3 mm
up, the shank leaves that slot or that insert, or it tilts past 10 degrees (the thread lets go); then its count goes
back to 0. While engaged, its turn about its own axis is summed, clockwise seen from above counting up (tightening a
right-hand thread) and counter-clockwise down. At 360 degrees the screw is *driven*: from then on it is welded where
it stands, upright with its washer on the disc. A binding whose four slots hold driven screws, lying flat, is
*clamped*: welded flat where it lies. Nothing is undone within a trial.

**End state, per binding.** *Aligned*: flat (its underside's centre within 1.5 mm of the board's top, tilted
3 degrees or less), its angle within 3 degrees of its target, and each slot over an insert that a 5 mm shank could
reach through it (the insert's centre within 11.0 mm of the slot's centre along the slot and 4.0 mm across it: the
slot's free play plus the bore's). *Seated n*: n slots hold a screw through the slot, 5 mm or more down an insert,
upright within 5 degrees, washer within 1 mm of the disc's top. *Fastened*: aligned, clamped, four driven screws.
The binding nearer the nose is the front binding (target +15 degrees, toe toward the nose), the other the back
binding (-15). A binding's angle is the heading of its heel-to-toe axis seen from above.

**Release and stillness**, needed for every score above 0: on every step of the last 1 s, both grippers' finger
zones 30 mm or more from every part of both bindings and of all eight screws, measured to each part's exact shapes
(`rigstate.released` with the parts' own distance functions). The finger zone is every point within half the finger
gap plus 8 mm (and 13.5 mm across) of the 38 mm of the gripper's axis from the point between its finger tips back
up: 57 mm long fully open, 16 mm closed. An unreadable arm, or a distance that cannot be computed, is not released
(fails closed). And every binding and screw moved 2 mm or less and turned 1 degree or less over that 1 s, judged by
the path, never by the velocity.

| Outcome | Score | Real-trial check |
|---|---|---|
| Both fastened on the middle squares (inserts at \|y\| 18 and 22 cm: discs 400 mm apart, centred on the cross) | 1.0 | Slots over inserts by eye; each washer down on the disc, the final turn shown by a paint witness mark; neither binding slides by hand; angles by protractor against the centre line; disc centres 400 mm apart and 200 mm either side of the cross by tape |
| Both fastened, on other squares (discs 320 to 480 mm apart, or off the cross) | 0.75 | As above, the tape reading 320 or 480 mm, or off the cross |
| Both aligned with four screws seated in each, not both fastened | 0.5 | Slots over inserts by eye; a 1 mm feeler gauge does not pass under any washer; angles by protractor |
| Exactly one aligned with four screws seated (driven or not) | 0.25 | As above, for one binding |
| Anything else: wrong angle, heel forward, presets swapped, three screws, screws resting on a disc or in a bore with no slot over it, a part held, touched or moving, an arm unreadable | 0 | Video: grippers clear of every part, nothing moving in the last second |
| A binding or a screw fell off the table (20 mm or more below its top): for the rest of the trial | 0 | Video |

The trial ends early once 1.0 has held for 1 s; otherwise the score is the end state at the run limit. Each listed
score is reached by a state sequence in the tests (`Tiers.test_every_tier_is_reachable`). A boundary value counts as
inside.

**Reading a real trial.** The real build has threaded M6 inserts (1.0 mm pitch). A real screw is threaded in until a
1 mm feeler gauge no longer passes under its washer (seated, the same 1 mm window as the simulation's), and the next
full clockwise turn advances it the last 1 mm and brings the washer down on the disc (driven, the simulation's
360 degrees). A paint witness mark across each head and the disc beside it shows that last turn. A binding with four
driven screws is clamped when it cannot be slid by hand.

Why the one latched zero: a part that falls to the floor is out of reach of both arms for the rest of the trial and
the set can no longer be finished. A screw dropped on the table or the board, or put into a bore with no slot over
it, is not latched: it can be picked up again and simply does not count. A screw driven through a slot of a binding
that is on the wrong square or at the wrong angle stays driven (the rules do not unscrew it), and that binding then
earns nothing; the scorer does not need a latch for it.

Exploits the tests check: holding a binding or a screw in place at the end (0); open fingers hovering 25 mm over a
head (0); letting go less than 1 s before the end (0); turning a screw that hovers over its slot, or one still in
its holder (no count); turning counter-clockwise (the count goes down); lifting a screw part way out mid-turn (the
count restarts); a binding held 1.6 mm over the board with its screws seated, or tilted 3.2 degrees (not aligned); a
binding turned a quarter or half turn, whose slots also line up (not aligned: angle); the two bindings swapped,
whose presets cannot line up at the other's angle (0); unreadable arms in five forms, and a part whose distance
cannot be computed (never released). A non-finite or missing part state scores 0 for that step and restarts the 1 s
windows. A new trial is recognised from rigstate's and mechkit's reset hooks, or, as a fallback, when both bindings
are put back on their start poses in one step with a jump.

The long slots (24 mm, for heel-to-toe centring) let a disc up to about 10.5 degrees off square still take its
screws by the scorer's 4.0 mm window across the slots (about 7 degrees with the slot's own 2.5 mm play alone); the
3 degree angle rule is what makes a binding aligned.

## Fidelity: what is simplified

Every part is real size in plan. Physics: every part's surfaces have friction 1.2 static and 1.0 dynamic; the
board, bindings, screws and holders use 0.5 mm contact offsets and 32 solver iterations. Simplified, and said in
`fidelity`:

- The inserts are smooth 8 mm square bores (the kit's plate with holes cuts square holes; a real insert is round) and
  the screws' shanks smooth 5 mm cores (the M6 thread's core): no threads. A shank has 1.5 mm of play a side in a
  bore. A screw therefore slides down to its seat, where a real one threads in about 6.5 turns (6.5 mm of its 14.5 mm
  shank goes into the insert once the washer is on the 8 mm disc, at 1.0 mm a turn).
- One clockwise turn of a seated screw counts as driving it home: at the M6 pitch, the last 1 mm of thread a real
  screw advances from the 1 mm seated window to the washer on the disc. The driven screw is then welded where it
  stands, and a binding with four driven screws is welded flat, each by rewriting its pose every control step (as the
  accepted snapfit-close and bayonet-mate do). In the simulation only the washer's contact friction on the disc and the
  screw's angular damping resist the turn; a real screw also meets thread friction, and a technician torques it to
  the maker's figure with a #3 driver afterwards.
- The slots are 10 mm wide, so a shank has 2.5 mm of play a side across a slot; real slots fit the screw closely.
  The extra play takes up the arm's placement error when it sets the hanging binding down (about 1 mm predicted, see
  "Demo and run limit").
- The highback is a flat plate 100 mm tall, lower than a real one: a taller one would put the straight-down carry grip
  above the arms' reach. The binding is one rigid body with its disc preset and fixed, and it has no straps.
- Masses are approximate: the binding is 0.35 kg (real child's bindings weigh about 0.35 to 0.5 kg; their ribbed,
  hollow plates weigh less than solid plates of these outer sizes would), and each screw 6 g (a real M6 x 16 binding
  screw with its washer is about 8 to 10 g).

## One arm alone

Predicted offline (kinematic proxy, no physics): with `DEMO_IDLE_ARM=right` the left arm mounts and fastens the
nose binding with the four nose-side screws and leaves the tail binding where it was parked: 0.25. `DEMO_IDLE_ARM=left`
mirrors it: 0.25. Measured on the GPU (run 3): 0.25 with either arm idle, as predicted.

With the grips the demo uses, one arm cannot do both: the far binding's highback (the carry grip) is at y -0.32 m
for the left arm, and the left arm reaches toward -y only to about y -0.15 m straight down at the board's height;
the far holder is at y -0.215 to -0.305 m and the far binding's middle square at y -0.18 to -0.22 m, beyond its
straight-down reach at the heights the screws need. A contorted one-arm route may exist (the earlier version's
joint-space search found a flat sideways arm reaching y -0.46 m near the table), so two arms help here but are not
shown to be required. The demo, like a technician with two benches' worth of work, fastens one binding with each arm.

## Demo and run limit

The demo is open-loop: it plays fixed waypoints and reads no object state from the simulator. Both arms work at
once, mirrored: the left arm's plan takes 210.3 s (4,206 control steps), the right arm's 206.3 s (it needs fewer
on-the-spot wrist turns).

1. Binding (20.8 s). Come over the highback with the fingers 50 mm apart (20 mm from each face), down until the pad
   tips are 20 mm below its top, close to a 2 mm command (8 mm narrower than the 10 mm highback), lift the binding's
   underside 60 mm over the board, carry it to the middle square while turning the wrist 75 degrees (the binding
   from +90 to +15 degrees; the right arm mirrored), lower it onto the board, open and rise.
2. Four screws (46.0 s each). Over the screw in its holder with the fingers 20 mm apart and closing along x,
   down until the pad tips are 1.5 mm above the washer, close to a 4 mm command on the 12 mm head, lift it to the
   TCP 100 mm over the table (the shank's tip 70 mm up: clear of the 58 mm sidewall tops), carry it in over the
   binding's open toe end to above its insert, lower it slowly through the slot into the insert until the washer is
   0.5 mm above the disc, open (it settles the last 0.5 mm onto the disc), go down 1.5 mm open so that the pad tips
   are 0.5 mm above the washer (4.5 mm of the 5 mm head between the pads), grip the head and hold the grip 0.4 s.
   Then six clockwise strokes of 180 degrees (1,080 in all, for the 360 the rule needs). Each stroke: turn 2 s, let
   go 0.6 s, rise 2 mm open 0.3 s (the pad tips 2.5 mm above the washer: clear of it, still round the head), turn back
   180 degrees open 1.2 s, go down 2 mm round the head 0.3 s, grip 0.8 s and settle the grip 0.3 s. After the sixth:
   let go, rise 2 mm, turn back 90 degrees open and rise (the wrist leaves at the same yaw as before, so the moves
   between screws are unchanged). Once driven the screw is welded and the later strokes slide over its head. Every
   wrist turn over 90 degrees is split (the interpolation takes the short way round); the strokes run 90 degrees
   either side of the yaw at which joint6 is near 0 over these inserts (-20 degrees left, +20 right), inside joint6's
   +-120 degree window with 25 degrees to spare. The strokes are whole multiples of 90 degrees so that, offline,
   every screw's square stub stays square to the next grip (the checker's proxy otherwise grasps a turned neighbour
   40 mm away first).

Clearances in the tests (`Clearances`): through every stroke, at every insert, at the grip, turn-back and release
heights, with the binding at its target and 6 mm / 1.5 mm off it, the pads (as 8 x 27 x 38 mm boxes) stay 11.9 mm or more from the sidewalls and 51.8 mm from
the highback, and the gripper's own collision capsules (the offline checker's) 11.8 mm from the sidewalls and 8.1 mm
from the highback. At the highback grip the open pads are 20 mm from its faces. The finger links' own capsules are
fat fits to plate-shaped fingers (radius 26 to 35 mm) and are not used for objects, as in the offline checker.

Demo knobs, in `task.py` only: `PLACE_TRIM` (per arm dx, dy, dyaw) and `SCREW_TRIM` (per screw dx, dy). Hung by its
highback, the 0.35 kg binding loads the arm with 3.4 N and 0.25 N m. A static calculation of the arm (yam_arena's joint
stiffness: 80 N m/rad for the shoulder and elbow joints, 10 for the wrist) predicted the earlier version's GPU
placement error to within about 0.6 mm; it puts the hanging disc 12.5 mm toward the robot and 2.2 mm outward of the
commanded point, toe 2.5 degrees down. Where the binding ends after its toe touches down lies between that and the
commanded point, so the trim is half of it: +6.2 mm in x (the slots' long axis, 9.5 mm of free play) and 1.1 mm
inward (2.5 mm of play). Read `origin` at the end of a GPU trace and correct the trim by the disc's remaining offset
from (0.300, +-0.200) if a screw misses its slot.

The run limit is 6,000 steps (300 s), 1.43 times the longer arm's plan: that plan ends at step 4,206, 1.0 is held
from step 4,211 offline, and the scripted policy stops by step 4,267 with its end hold, 1,733 steps before the limit.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Nothing in the task is to be found: the inserts,
presets and parked poses are the same in every trial. Reset: each binding back on its parked spot on its own end of
the board (disc on the centre line, toe toward that end and 5 mm from it), the screws back in their holders; the
welds and the turn counts are cleared by mechkit's and the task's reset hooks. The board, the cross and the holders
are fixed. Nothing is consumed.

## Runtime per trial

The first GPU demo of this revision (2,611 steps, 130.6 s of sim time) took about 10 minutes on an L40S, the second
(3,138 steps) about 13. At that rate the demo's 4,267 steps take about 17 minutes and a full 6,000-step Dream-1 trial
about 24.

## Real-world build

A 1100 x 240 x 10 mm plywood or HDPE plank (or a real 110 cm child's board shimmed flat), white, with a 4 mm black
tape centre line, a red tape cross and sixteen M6 threaded inserts on the 4 x 4 pattern at the listed positions
(screw-in M6 inserts for wood or plastic, set flush with the top). Two child's bindings with 4 x 4 discs (real ones
about 0.35 to 0.5 kg; take the straps off or tie them back), each disc set at its preset, +15 or -15 degrees, and
fixed in its baseplate with two drops of cyanoacrylate so that it stays at its preset while the binding is carried.
Eight M6 x 16 binding screws with washers in two drilled blocks, each head with a paint witness mark. Score as in
"Reading a real trial" above: slots over inserts by eye; seated with a 1 mm feeler gauge under each washer; driven
when the next full turn, shown by the witness mark, brings the washer down on the disc; clamped when the binding
cannot be slid by hand; angles with a protractor against the board's centre line; the stance with a tape. Nothing
is powered, sharp or hot.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
