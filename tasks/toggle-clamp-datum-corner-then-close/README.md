# Seat an aluminium housing on a fixture's datum stops, then lock both toggle clamps

A steel fixture plate on the bench carries two datum stops (S1 across its far side, S2 along its right side) and two
vertical hold-down toggle clamps (C1 on the near side, C2 on the left), both open. An aluminium housing (the casting)
lies on a staging block to the left of the plate, and a splash guard stands behind the plate. The robot must carry the
casting into the fixture, set it hard into the corner against both stops, close both clamps so each locks over centre
with its red pad on the casting's top, and take its hands away. The instruction names neither the order nor the
pushes.

Setting a part against a fixture's stops and clamping it is a listed task of three O*NET occupations (US jobs in
2025, from each O*NET summary page):

- Machinists, O*NET 51-4041.00, 288,300 jobs: "Align and secure holding fixtures, cutting tools, attachments,
  accessories, or materials onto machines."
- Computer numerically controlled tool operators, O*NET 51-9161.00, 170,000 jobs: "Mount, install, align, and secure
  tools, attachments, fixtures, and workpieces on machines."
- Welders, cutters, solderers and brazers, O*NET 51-4121.06, 437,700 jobs: "Align and clamp workpieces together,
  using rules, squares, or hand tools, or position items in fixtures, jigs, or vises."
- Together 896,000 US jobs.

Here the part is an aluminium housing loaded into a machining fixture, the machinist's and CNC operator's case; a weld
fixture is loaded and toggle-clamped the same way.

## Why it is hard (the hard parts, in order)

1. **Carry and set down.** The casting is carried from the staging block over the clamps' bases (40 mm) and the
   stops (24 mm) and set down on the plate short of the stops (the arm's placing error is a few millimetres).
2. **The corner is a contact, not a place.** It has to be pushed home against S1 and S2 with a square push on each
   face; an off-centre push turns it. The score reads 1.0 mm at both ends of both stops, flat on the plate.
3. **Order matters.** Each shut clamp presses the casting with 25 N, so a casting clamped off a stop is held
   there, crooked: friction under and over it (about 16 N with one pad, 31 N with both) is far more than an arm can
   push sideways, and it scores 0 as not seated. The only way out is to reopen the clamp (a 27 to 34 N lift at the
   grip) and reseat the casting; the score is read from the end state, so a corrected mistake is not punished.
4. **The clamps are touched parts that behave.** Each lever rests on its stop 30 degrees back past vertical. The
   handle resists with about 1.5 N at its grip up to over-centre (55 degrees from vertical); a handle pushed only
   part way springs back open. Past over-centre the clamp shuts itself and its pad presses the casting with 25 N; it
   then takes a lift of about 27 N (C1) or 34 N (C2) at the grip to reopen it. A clamp shut on nothing does not count
   as locked.
5. **Two sides.** The clamps stand on two sides of the part and each lever is pushed from its own outer side, past
   the open levers (14 cm tall), the staging block on the left and the guard behind. In the demo the left arm loads
   the casting and shuts C2 and the right arm shuts C1; with either arm idle the demo scores 0.5 or 0 (below).
6. **Hands off.** The score is read with both grippers 30 mm clear of the casting and the clamps and everything
   still for 1 s: a casting held in the corner by a hand earns nothing.

## Scoring

Conditions (all read at the last step, from poses and joint angles; a missing or non-finite reading scores 0):

- **Seated**: flat on the plate (tilt at most 1 degree, lowest corner within 1.5 mm of its top); far face within
  1.0 mm of S1 (gap or overlap) at both ends of S1's 90 mm face (40 mm either side of its centre); right face within
  1.0 mm of S2 at both ends of its 40 mm face (x 285 and 315 mm). A stop point the face does not cover (3 mm in from
  its edges) counts as a miss.
- **Shut** (per clamp): handle within 8 degrees of flat or past it.
- **Locked** (per clamp): shut, with the centre of the pad's underside between 1.5 mm into the casting's top face and
  0.5 mm above it, and at least 4 mm inside the face's edges (the pad then presses it with 25 N). A clamp shut on
  nothing is not locked.
- **Hands off**: on every step of the last 1 s both finger zones are 30 mm or more from the casting and both clamps;
  over that 1 s the casting moves at most 1 mm and turns at most 1 degree, and neither handle turns over 2 degrees.

| Outcome | Score |
|---|---|
| Seated, both clamps locked, hands off | 1.0 |
| Seated, one clamp locked | 0.5 |
| Seated, no clamp locked | 0.25 |
| Not seated (a casting clamped off its stops included), or not hands off | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Seated | 1.0 mm feeler gauge at each end of each stop; a level on the casting's top |
| Locked (each clamp) | 8 degree angle gauge on the handle; a 0.5 mm feeler does not slide under the pad; a 10 N lift |
| | on a spring scale at the grip leaves it shut |
| 1.0 (both locked) | also a proof test: a 20 N pull on a spring scale, straight up and then sideways away from the |
| | stops, does not move the casting (dial indicator) |
| Hands off | a 30 mm spacer against each hand; dial indicator on the casting (1 mm); 30 fps video of the last 1 s |

Possible scores: 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

## Results

The GPU column is from the official L40S launchable, 21 Nov 2026, one trial each, on this build (stiff clamp drive, 25 N per pad).
The offline prediction column is the offline checker with this task's offline stand-in for the physics
(`task.OfflineModel`): a check of the plan, not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Settle (`DEMO_HOLD_ONLY=1`) | arms still: casting stays on its block, levers on their stops | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 1285) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | seats the casting, shuts C2 | 0.5 | 0.5 |
| `DEMO_IDLE_ARM=left` (right alone) | shuts C1 on nothing, casting left on its block | 0.0 | 0.0 |
| `DEMO_VARIANT=no_clamps` | seats the casting, clamps left open | 0.25 | not run |
| `DEMO_VARIANT=left_only` | seats it and shuts C2 only | 0.5 | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

In the GPU demo trace of the earlier build (16 Nov) both levers end at 120 degrees (flat, pads on the casting), with the casting
0.63 mm from S1 and 0.0 mm from S2. On this build, check in the trace that q1_deg and q2_deg reach 120 and stay there
(the pads hold the levers at flat against the drive) and that the snap does not move the casting off S1 or S2.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The pushes and the lever
arcs are computed from the layout.

| Time (s) | Left gripper | Right gripper |
|---|---|---|
| 0 to 6.5 | down round the casting on the block, grip it across its 60 mm | wait aside |
| 6.5 to 21 | lift 70 mm, carry it clear of the clamps, set it down 12 mm short of S1 and S2, let go | wait |
| 21 to 29 | close the fingers into a block, push the near face +x into S1 (8 mm over-travel), back off | wait |
| 29 to 37 | block on the left face, push -y into S2 (8 mm over-travel), back off | wait |
| 37 to 52 | round to C2's outer side, push its handle along an arc past over-centre, move aside | wait |
| 50 to 64.8 | wait | block in front of C1's handle, push it along an arc past over-centre, move aside |

Offline: worst IK 0.1 mm; joint 4 margin 4.6 degrees (a straight-down grip's usual); the TCPs stay 27 cm
apart and the arm capsules 10 cm (the right arm moves in only when the left has started to move aside).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 2800
```

Runtime: a trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (the demo itself is 1296
steps, 64.8 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the gaps to S1 and S2,
the casting's tilt, yaw and height, both handle angles and lock flags, both TCPs and openings, and a `hist` of the
last 3 s of those numbers in every row.

## Simplifications

- The clamps: pad reach close to the catalogue arm lengths of DESTACO 207-U (2.14 in, 54 mm; C1's pad is 52 mm from
  its pivot) and 210-U (3.63 in, 92 mm; C2's pad is 91 mm from its pivot) vertical hold-down clamps. Base, post and
  handle sizes are our own; the handles are 70 and 90 mm.
- Rigid bodies. A clamp's arm and handle are one lever on a hinge (a real one links them by a toggle linkage); the
  handle lies above the arm when shut and the lever stands 30 degrees back when open.
- One declared rule (`task.toggle_target`), acting through a stiff joint drive on each lever (1.86 N m/rad on C1,
  3.26 N m/rad on C2): short of over-centre (55 degrees from vertical) the drive's target trails the lever by about
  2 degrees, a steady pull back toward the open stop of about 1.5 N at the grip (a little more while the handle is
  moving, because the target is set once a control step); past over-centre the target runs on at 1.5 rad/s to
  40 degrees past flat, beyond the lever's far stop. A lever that lands on the casting at flat is held there by it, so
  its pad presses the casting with 25 N (C1: 1.86 x 0.70 / 0.052; C2: 3.26 x 0.70 / 0.091), and it takes about 27 N
  (C1) or 34 N (C2) lifting the grip to reopen it. A lever forced back 5 degrees below over-centre is open again.
- The press: 25 N per pad is a light spindle setting (est) for a thin-walled aluminium part. The catalogue figures,
  the 207-U's 375 lbf (1,670 N) and the 210-U's 600 lbf (2,670 N), are maximum holding capacities, not the press a
  clamp is set to. A real toggle clamp's linkage gives a large press for a small handle force near its dead point,
  which is why a light handle force and a 25 N press go together here. With both pads at 25 N and friction 0.3 the
  casting holds against the 20 N proof pull of the real trial: about 52 N straight up, about 31 N sideways away from
  the stops (plate and pads); with one pad it would slide at about 16 N, so the proof pull is a check of the 1.0 tier
  only. The proof pull itself is not run in the simulation.
- The casting: a 6061 aluminium block 60 x 100 x 30 mm pocketed from below to 5 mm walls and a 5 mm top, standing in
  for a die-cast housing: (60 x 100 x 30 - 50 x 90 x 25) mm3 = 67.5 cm3, at 2.70 g/cm3 0.18 kg. The simulation uses a
  solid box collider of its outer size with that mass (only the outer faces touch anything). Friction 0.3 aluminium on
  steel, HDPE and the pad (est).
- The handle force (1.5 N), the press (25 N), the lever mass (0.04 kg) and the drive's damping are estimates.
- The splash guard and staging block are plain boxes.

## Sources

- Machinists, O*NET 51-4041.00 (task statement and 2025 employment, 288,300):
  https://www.onetonline.org/link/summary/51-4041.00
- Computer numerically controlled tool operators, O*NET 51-9161.00 (task statement and 2025 employment, 170,000):
  https://www.onetonline.org/link/summary/51-9161.00
- Welders, cutters, solderers and brazers, O*NET 51-4121.06 (task statement; 2025 employment 437,700 for
  51-4121.00): https://www.onetonline.org/link/summary/51-4121.06
- DESTACO 207-U (arm 2.14 in, 375 lbf): https://www.destaco.com/hold-down-clamps-207/product/207-U
- DESTACO 210-U (arm 3.63 in, 600 lbf): https://www.destaco.com/hold-down-clamps-210/product/210-U

Estimates (stated in fidelity): the casting's mass and friction, the lever's mass, the handle force, the pad press.

## Real build and reset

A 12 mm steel plate with two hardened stops, a DESTACO 207-U and a 210-U (or clamps of the same reach) bolted at the
positions in `task.json` with their spindle pads set so that each presses the casting's top with about 25 N when shut
(set once with a load cell under a spacer of the casting's height), the pocketed aluminium block above, an HDPE
block, a steel guard plate. Reset: open both clamps (levers back on their stops), clear the plate, wipe chips off the
stops, lay the casting right way up on the block, centred at x 260 mm, y +245 mm, its 100 mm side along y. Check each
pad's press once a session and change the clamp pads every few hundred closings.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the toggle rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the lever geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
