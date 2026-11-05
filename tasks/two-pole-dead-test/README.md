# Prove a supply dead with a two-pole tester: prove the tester, test all 10 pairs, prove it again

Before anyone works on an isolated supply, an electrician proves it dead ("test before touch"; in UK practice "safe
isolation"). A two-pole voltage tester is used: first it is proved on a known source (a proving unit), then each
conductor is tested against every other, then the tester is proved on the known source again, because a tester
that failed part-way reads "dead" on a live conductor. NFPA 70E requires each phase conductor to be tested phase to
phase and phase to ground, with the tester checked on a known source before and after; UK practice, with the HSE's
GS38 guidance on test probes, also tests neutral, so a three-phase supply with neutral and earth takes 10 tests:
L1-L2, L1-L3, L2-L3, L1-N, L2-N, L3-N, L1-PE, L2-PE, L3-PE and N-PE. Electricians (O*NET 47-2111) and electrical
maintenance technicians do it before every job on a circuit.

Here a yellow proving unit (terminals P+ and P-) and a terminal block carrying the five terminals of an isolated
supply (L1, L2, L3, N, PE) stand in a row on the midline. The tester's two probes, a red and a black square bar,
lie on rests to either side. The robot must prove the tester, test all 10 pairs, prove the tester again and put
the probes back, without ever bringing the metal tips close together, bridging two terminals with a tip, leaving a
probe on a terminal or putting a probe down anywhere but a rest. The second gripper's role is the instrument's: a
two-pole tester needs one probe on each of two conductors at the same time, so each gripper holds one probe, and
one arm alone cannot make a single test.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, mass, position, threshold and colour, in one place |
| `scoring.py` | The scorer: plain functions of the two probes' states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step (unchanged copy) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the gripper hook, the demo and the exploits we tried, through a small kinematic stand-in (no simulator) |
| `tests/test_demo_plan.py` | Offline check of the scripted plan against the simulator's and station's limits (generic) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed boxes and cylinders, two loose boxes) and the standard `task_score` scorer.

The 'held' rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks. `rigstate.py`
(imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool
point, finger-tip point, finger opening, in the objects' frame) before each control step and to start a new trial
at each reset. It reads only state the simulator already has and changes nothing the embodiment does. If it cannot
read the arms it prints why once; 'held' is then taken as met on those steps, and the trace counts them
(`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Prove that the five terminals on the dark grey block (L1, L2, L3, N and PE) are dead, using the two-pole tester's red and black probes. Each probe's metal tip is the end that points toward the middle of the table. Hold one probe in each gripper by its handle; a test is both tips resting on the tops of two different terminals at the same time, held still for at least 1 second. First test the tester on the yellow proving unit's two terminals, then test every pair of the five terminals (10 tests), then test the tester on the proving unit again. Keep the two metal tips at least 2 cm apart at all times, never let go of a probe anywhere but its rest, and finish with both probes back on their rests." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/two-pole-dead-test/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Prove that the five terminals on the dark grey block (L1, L2, L3, N and PE) are dead, using the two-pole tester's red and black probes. Each probe's metal tip is the end that points toward the middle of the table. Hold one probe in each gripper by its handle; a test is both tips resting on the tops of two different terminals at the same time, held still for at least 1 second. First test the tester on the yellow proving unit's two terminals, then test every pair of the five terminals (10 tests), then test the tester on the proving unit again. Keep the two metal tips at least 2 cm apart at all times, never let go of a probe anywhere but its rest, and finish with both probes back on their rests." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/two-pole-dead-test/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `first_prove_step`,
`reprove_step`, `pairs_tested`, `pairs_missing`, `holding`, `holding_s`, `red_on_post`, `red_pin_above_post_mm`,
`red_level_deg`, `tip_gap_mm`, `min_tip_gap_mm`, `red_held_by`, `opening`, `arms_unavailable`, both TCPs, ... and
the same for the black probe). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per
process and starts it over at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator (from the `work` folder above `tasks/`):
`PYTHONPATH=tools/offline/yam_stub:tasks/two-pole-dead-test python3 -m unittest discover -s tasks/two-pole-dead-test/tests -v`
(the tests also find `tools/offline/yam_stub` by themselves).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Definitions:

- **Metal tip, contact point**: each probe is a 26 x 26 mm bar 300 mm long; its metal tip is the 20 mm at its tip end
  and its contact point the centre of the tip's underside, 10 mm from that end. The bar is square and may be gripped
  either way up, so the underside is whichever long face points most nearly down.
- **Held**: a gripper's finger gap is at most 32 mm and the point between its finger tips is within 15 mm of the
  probe's axis, between the far end and 40 mm from the tip end (on the handle). A gripper can hold only one probe.
- **On a terminal**: the contact point within 8 mm of a post's axis horizontally (over its 16 mm top) and between
  3 mm below and 4 mm above the post's top.
- **A test**: both probes held by different grippers, each tip on a different terminal, each probe level (underside
  within 10 degrees of facing straight down) and steady (contact point under 1 cm/s), the same pair for 20 control
  steps (1 s) in a row. The **prove** is P+ with P-; a **supply pair** is two of L1, L2, L3, N, PE and counts only
  after the first prove; a proving terminal with a supply terminal counts for nothing. The **re-prove** is a prove
  after all 10 pairs; any supply test after it cancels it.
- **Touching a post**: any part of a metal tip within 2 mm of a post. **On a rest**: both ends of the probe's axis
  over a rest board and its lowest point within -3..+4 mm of the board's top.
- **Gap between the tips**: between the two 26 x 20 x 26 mm metal tips, surface to surface (from 98 points on each
  tip's surface against the other's box: never below the true gap, at most 1 mm above it; tested against a 2,402-point
  sampling over random poses).

Rules for the whole trial; breaking one at any control step scores 0, whatever follows:

1. the two metal tips less than 15 mm apart (a short; this also covers both tips on one post);
2. a metal tip touching two posts at once (a short across two conductors);
3. a probe that is not held touching a post with its metal tip (a probe let go of, or pushed, onto a terminal);
4. a probe neither held nor on a rest for more than 0.25 s (more than 5 control steps: set down, laid across the
   terminals, dropped);
5. a probe's lowest point more than 10 mm below the table top (off the table).

| Outcome | Score |
|---|---|
| Any rule broken at any moment, or the tester never proved | 0 |
| Proved, all 10 pairs tested after it, re-proved after the last of them, both probes on their own rests, neither gripper holding either | 1.0 |
| The same, but a probe still held or not back on its own rest | 0.9 |
| Proved, k supply pairs tested after it (0 to 10), no valid re-prove | 0.25 + 0.05 k (0.25 to 0.75) |

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both probes still
(under 1 cm/s and 0.5 rad/s) for 3 s, longer than the demo's last release and retreat (1.5 s + 1.0 s); and 1.0
itself requires both grippers to have let go, so the trial cannot end with a probe still in a gripper. A new trial
is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts both probes back on their rests
in a single step (one of them moving 5 cm further than its velocity and a free fall allow). A simulator state with a
non-finite number scores 0 for that step and leaves the trial's record as it was.

### Why these numbers

| Threshold | Value | Why |
|---|---|---|
| Pairs | 10, equal weight | Each pair is one path on which a conductor could still be live relative to another (a backfeed between phases, a borrowed neutral, a lost earth); none is optional, which is why the re-prove (0.9) and 1.0 need all 10. N-PE is the test most often skipped in practice; weighting it the same as the others keeps it from being skipped here. 0.05 a pair keeps 10 pairs (0.75) below any re-proved run |
| First prove | 0.25 | The first coordinated two-probe contact; the procedure's own gate (tests before it count for nothing, as a tester not yet proved proves nothing) |
| Re-prove | after all 10 | A prove only vouches for the tests made before it; a supply test after the re-prove is unproven again |
| Test hold | 1 s, both still under 1 cm/s | About the time a two-pole tester takes to give a steady reading; a tip moving at 1 cm/s moves less than the post's 8 mm radius in that time. A probe swept along the row does not count |
| On a terminal: over the top | 8 mm | The post's own radius: the contact point is over the 16 mm post top |
| On a terminal: height | -3 to +4 mm | The real probes carry a 4 mm spring contact pin under the tip, which makes contact anywhere in its travel; 3 mm below covers contact penetration and a slightly tilted tip. The demo's tips sit 0 to 3.8 mm up across every sag case tested |
| Level | 10 degrees | The tip's underside (and its pin) square on the post top; a probe pivoting in the grip because it is being leaned on a terminal tilts far more. Payload sag tilts the 60 g probe about 0.3 degrees |
| Short | 15 mm, tip to tip | A tip sliding off its 16 mm post top moves up to about 16 mm: closer than that, one slip could bring the metal tips together. The instruction asks for 2 cm. Tips on neighbouring posts (50 mm apart) are 24 mm apart, 9 mm clear |
| Touching a post | 2 mm | Metal within 2 mm of a post is treated as touching it (contact or about to be) |
| Held: gap | 26 mm + 6 mm | Closed on the 26 mm bar; a gripper open around a probe, or closed on two, does not count |
| Held: on the handle | 15 mm from the axis, 40 mm or more from the tip end | The grip point of a gripper closed on the bar lies 9 mm below its axis; 40 mm keeps the 27 mm pads wholly behind the 20 mm metal tip, as a hand stays behind a probe's finger guard |
| Set-down grace | 0.25 s | A gripper needs more than 1 s to let go and grip again; 0.25 s only absorbs a brief misreading of 'held' |
| Probe length | 300 mm, gripped at its middle | The grip is 140 mm from the contact point, so the grippers are never closer than 28 cm in y (the station needs 20 cm), and the far post is 3.2 cm inside the reach limit. Gripped at its centre of mass, the probe's weight puts no torque on the grip |
| Post spacing | 50 mm | Neighbouring tips 24 mm apart (above the 15 mm short limit with 9 mm to spare); a tip on a post is at least 17 mm from every other post, so it cannot bridge two |
| Post height | 25 mm above the table | Low enough for reach at the far post, high enough to be read on a side video |

### Exploits we tried (all in the tests)

- **Doing nothing**: 0. **One arm alone**: 0 (below).
- **Testing before proving**: all 10 pairs before the first prove count for nothing (0); proving afterwards gives 0.25.
- **Re-proving before the 10 pairs are done**: does not count (nine pairs, re-proved, put back: 0.70); the tenth pair
  afterwards without another re-prove: 0.75. **Testing again after the re-prove** cancels it (0.75 until re-proved).
- **Both tips on one post** (end to end on L3): rule 1, 0. **Tips 14 mm apart** on neighbouring posts: 0; 16 mm: fine.
- **Sweeping one tip along the row** at 5 cm/s while the other rests on L1, over every post top: no pair counts and no
  rule is broken; a tip creeping at 1.1 cm/s does not count either.
- **A probe tilted 12 degrees** with its tip on a post: not a test.
- **Letting go of a probe with its tip on a post**: rule 3, 0 at once. **Grippers open around the probes** while the
  tips touch the proving unit: not held, rule 3, 0.
- **Laying a probe across two posts**, not held: rule 3 (its tip touches one) and rule 4. A single held probe across
  two posts is not a test: a test needs two probes held by different grippers.
- **One gripper holding the red probe while the black one is pushed onto P-**: rule 3, 0. Both grippers on one probe:
  not 'held apart', no test.
- **A tip wedged between two posts** (turned 45 degrees down onto the block between L1 and L2): within 0.75 mm of
  both, rule 2, 0. With a tip on a post this cannot happen (its metal stays at least 17 mm from every other post).
- **Setting a probe down on the table**: rule 4 after 0.25 s, 0; a fumble of 0.25 s or less is not a breach. A probe on
  the other arm's rest is not a set-down. **A probe off the table**: rule 5, 0.
- **Either probe on either terminal** (black on the nearer post): pairs are unordered, 1.0.
- **Stopping short**: re-proved but still holding a probe on its rest, or not putting the probes back: 0.9.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) and `DEMO_IDLE_ARM=right` (the left arm alone) both score 0 in the offline
replay: the moving arm carries its probe through the whole sequence, but with the other probe still on its rest no
test is ever made, so there is no prove; no rule is broken. No one-arm strategy scores more: 'held' fits only one
26 mm bar in a gripper, a test needs both probes held by different grippers, and a probe pushed onto a terminal
without a gripper on it breaks rule 3. (Offline kinematic predictions; the GPU runs go in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 75.0 s (1,500 control steps). Both grippers close along x across the middle of their probe
(yaw -pi/2 left, +pi/2 right), commanded 18 mm (8 mm narrower than the bar) over 1.5 s after a 0.5 s settle, with the
pad tips 4 mm above the bar's bottom, after a slow 1 s descent over the last 2 cm. Both probes are lifted 15 mm above
the post tops and carried in, tips toward the midline, to just above P+ and P-; both tips go down (1 s for the last
14 mm) and are held for 2 s: the prove completes at step 209 (10.5 s). The black tip then moves over L2, then the red
over L1 (so the proving unit is never paired with a supply terminal), and both go down. The other nine pairs follow
one probe at a time, in the order L1-L2, L1-L3, L2-L3, L2-N, L3-N, L3-PE, N-PE, L2-PE, L1-PE, L1-N: each move lifts
the tip 14 mm (0.6 s), carries it over the next post at 8 cm/s (1 to 1.5 s), lowers it (1 s) and holds it (2 s), a
pair about every 4.6 s, while the other tip stays on its post. The red tip is always on a post nearer the arms than
the black, so neither probe ever passes the other. The red tip then moves over P+, the black over P-, and both go
down: the re-prove completes at step 1,345 (67.3 s). Both probes go back to their rests and are let go 2 mm above
them. In the offline replay the 1.0 conditions hold, still, from step 1,456 (72.8 s), once both grippers have opened
past 32 mm, so the trial ends at about 75.8 s.

Every waypoint is at least 3.0 cm inside the reach limits (the proving unit at x 13 cm, 3.0 cm in front of the 10 cm
minimum; the PE post at x 38 cm, 3.2 cm inside the limit for the TCP's height and its 14 cm from the midline). The two
TCPs are never closer than 28 cm in y (the station needs 20 cm whenever they are within 5 cm in x), and the metal tips
are never closer than 24 mm (on neighbouring posts).

The arm must not press on the fixed posts. Each tip's underside is commanded 1 mm above the post's top, never below
it. The arm does not compensate a held weight (about 8 mm down for 0.3 kg at 0.3 m reach): about 1.6 mm for the 60 g
probe, which is gripped at its centre of mass, so the tip settles onto the post, and the post then carries at most
the probe's own weight (0.6 N) as the arm springs back toward the command above it. With no sag the tip stays 1 mm
up, inside the 4 mm band with 3 mm to spare. The offline tests replay the demo with no sag, the estimated sag, twice
it, sideways sag either way, the probe pitched 1 degree in the grip either way (2.4 mm at the tip, including with no
sag), and 3 mm of opposite x error on the two arms: all score 1.0, and the tips are never closer than 18 mm. On the
GPU, read `red_pin_above_post_mm`, `red_level_deg` and `min_tip_gap_mm` (and the black ones) in `TASK_TRACE`.

The run limit is 2,400 steps (120 s). The procedure needs 12 two-probe contacts, each held 1 s, in a fixed order on
top of two picks and two put-backs; the demo uses 76 s of the 120.

## Novelty

The closest tasks on the board are "Prove a stick-mounted voltage detector on its proof tester before and after
testing a bar" (a one-point, non-contact detector held by one arm, one test), "conductors-series-tester" (bars
placed on a tester's pads, not a two-point test), "Probe three test points in order with a tethered probe" (one probe,
one point at a time) and the board's live-panel one-hand-rule task (the opposite constraint: one hand kept away). None
of them needs two contacts made at the same time by two separately held instruments, and none scores a complete set
of combinations: here every one of the 10 pairs must be covered between a prove and a re-prove, with the order checked
at every control step and with hazards specific to a two-pole test (the tips meeting, a tip bridging two terminals, a
probe left on a terminal). The second gripper is not a holder or a fixture: it is half of the instrument.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the red probe back on the left rest centred at
x 22.0 cm, y +22.0 cm and the black on the right rest at x 22.0 cm, y -22.0 cm, long side along y, tip end toward the
midline; the proving unit, terminal block, posts and rests are fixed.

## Runtime per trial

About 2.5 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 76 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 30 x 100 x 8 mm yellow and a 250 x 40 x 8 mm dark-grey insulating board (PLA or acrylic), clamped to the bench, with
seven 16 mm posts: a 13 mm printed collar in the conductor colour and a 4 mm brass disc on top, 25 mm above the bench
(P+ and P- 60 mm apart across the yellow board; L1, L2, L3, N, PE 50 mm apart along the grey one). Two 26 mm square
PVC tubes 300 mm long, one red and one black, each with a 20 mm brass end cap and a 4 mm spring contact pin under the
cap 10 mm from the end, weighted to 60 g; two light-grey 60 x 320 x 4 mm boards as rests. Nothing is live: for an
objective contact record, wire each brass disc and each pin to a 5 V contact logger; otherwise film the post tops
from the side (with a 4 mm mark) and from above.

## Results in simulation

To be measured on the GPU (Isaac Lab-Arena, `-E object_jitter=0 -E control_hz=20`, one trial each). Not yet run:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline prediction 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (offline prediction 0) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (offline prediction 0) |
| Dream-1 | to be measured on the GPU |
