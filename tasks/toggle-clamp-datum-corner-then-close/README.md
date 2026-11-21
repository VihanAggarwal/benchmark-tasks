# Seat an aluminium housing on a fixture's datum stops, then lock both toggle clamps

A steel fixture plate on the bench carries two datum stops (S1 across its far side, S2 along its right side) and two
vertical hold-down toggle clamps (C1 on the near side, C2 on the left), both open. An aluminium housing (the casting)
lies on a staging block to the left of the plate, and a splash guard stands behind the plate. The robot must carry the
casting into the fixture, set it hard into the corner against both stops, close both clamps so each locks over centre
with its red pad on the casting's top, and take its hands away. The instruction names neither the order nor the
moves.

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

## The clamps: an articulated four-bar toggle linkage, closed by the robot

Each clamp is modelled on the DESTACO 207-U vertical hold-down clamp (catalogue: clamp bar opening 99 degrees, handle
opening 57 degrees), with the arm reach of the 207-U (C1) and the 210-U (C2). It is four bars, three of them moving,
each a separate jointed body in the simulation:

- the **base** (fixed): a foot on the plate, a post, two side cheeks; two pivots: the arm's B, 54 mm above the table,
  and the handle's A, 32 mm behind B and 26 mm above the table;
- the **handle** on A: a bar with a black 45 mm grip at its end (C1 110 mm, C2 145 mm) and a 16 mm crank A-C;
- the **connecting link** C-D, 32 mm, on a pin at the crank's end;
- the **clamp arm** on B (C1 62 mm, C2 99 mm), with a red pad under its end and a tail pin D 7 mm from B.

Open, both rest on their stops: the handle 57 degrees back from vertical, the arm 99 degrees up. The robot pushes the
handle forward; the link swings the arm down, fast at first (the arm turns 5 degrees per handle degree) and ever
slower. The pad lands on the casting about 3 degrees before the handle is vertical. Vertical is the linkage's dead
centre (A, C and D in one line): the pad presses hardest there, 25 N (C1) or 29 N (C2), through a spindle with a
rubber cap (40 N/mm, est). The handle's stop is 1.5 degrees past vertical; there the pad still presses 20 N, and
because the linkage is past dead centre that force now pushes the handle onto its stop: the over-centre lock. It
takes about 2.2 N (C1) or 2.8 N (C2) back at the grip to pull it out again.

**No drive.** Nothing closes a clamp but the robot: there is no motor, no spring toward shut and no snap. The force at
the grip, 15 mm from the handle's end, is about 0.3 N (the handle's weight) over the free part of the stroke and
rises to 2.2 N (C1) or 2.7 N (C2) in the last 2 degrees before vertical, where the pad is pressed; an arm gives about
5 N. Let go short of vertical a clamp does not lock: from about 40 degrees back the arm's weight lowers it onto the
casting and the handle stops short of vertical; nearer its open stop it falls back open. With nothing under the pad a
handle can go over centre onto its stop: shut on nothing, which is not locked.

**How the simulation closes the loop.** A tree of joints cannot hold a closed four-bar loop, so one declared rule
(`task.linkage_targets`, run by a mechkit `on_step` callback every 20 Hz control step) does it, through three
joint springs:

- arm: its target is the angle the link gives for the handle's angle (`layout.Clamp.arm_target`, solved from the
  circle intersection of the link and the arm's tail); its spring is the spindle's stiffness (40 N/mm at the pad), so
  an arm held up by the casting presses it with exactly the spindle's force;
- link: its target keeps its far end on the arm's tail pin;
- handle: the arm's spring load is passed back to the handle by virtual work (arm torque times d(arm)/d(handle), the
  linkage's velocity ratio) as a torque, applied through the handle joint's spring (target = handle angle + torque /
  stiffness, plus half a step of the handle's own speed so a moving handle feels no drag). Short of dead centre it
  pushes the handle open; past it, onto its stop. With nothing pressed it is zero.

The handle's spring is 0.6 of the reflected stiffness where the pad first touches: stiff enough that the 20 Hz update
does not chatter against the gripper's own give, soft enough that the once-a-step anchor does not tug a swung handle.
A toy model of one handle (240 Hz physics, the rule at 20 Hz, the gripper as a 1.8 mm/N spring) along the demo's
stroke gives peaks of 1.8 N (C1) and 2.5 N (C2) in the free swing and 2.4 N and 2.9 N in the press, and the handle
ends on its stop; that model is ours, not a simulator run. A gripper pinching the grip across (fingers along the pivot
axis) drives the handle by friction only: enough for the free swing, not for the press (first GPU run of this build:
both handles stalled 2 to 3 degrees short of vertical, the fingers slipping along the grip). The demo therefore lets
go once the pad is down and pushes the grip's back face square on with the gripper closed into a block.

## Why it is hard (the hard parts, in order)

1. **Carry and set down.** The casting is carried from the staging block over the clamps' bases (44 mm) and the
   stops (24 mm) and set down on the plate short of the stops (the arm's placing error is a few millimetres).
2. **The corner is a contact, not a place.** It has to be seated against S1 and S2 square on each face; an
   off-centre push turns it. The score reads 1.0 mm at both ends of both stops, flat on the plate.
3. **Order matters.** Each locked pad presses the casting with 20 N, so a casting clamped off a stop is held there,
   crooked: friction under and over it (about 12 N with one pad, 24 N with both) is more than an arm can push
   sideways, and it scores 0 as not seated. The way out is to reopen the clamp and reseat the casting; the score is
   read from the end state, so a corrected mistake is not punished.
4. **The clamps are articulated parts closed by hand.** Each handle must be carried round its whole stroke, 57
   degrees, and through dead centre onto its stop; let go short of vertical it does not lock. The open arms stand
   99 degrees up beside the casting (C2's 15 cm tall), so a gripper working on the casting's left face meets C2's
   arm.
5. **Two sides.** The clamps stand on two sides of the part, past the staging block on the left and the guard behind.
   In the demo the left arm loads the casting and closes C2 and the right arm closes C1; with either arm idle the
   demo scores 0.5 or 0 (below).
6. **Hands off.** The score is read with both grippers 30 mm clear of the casting and the clamps and everything
   still for 1 s: a casting held in the corner by a hand earns nothing.

## Scoring

Conditions (all read at the last step, from poses and joint angles; a missing or non-finite reading scores 0):

- **Seated**: flat on the plate (tilt at most 1 degree, lowest corner within 1.5 mm of its top); far face within
  1.0 mm of S1 (gap or overlap) at both ends of S1's 90 mm face (40 mm either side of its centre); right face within
  1.0 mm of S2 at both ends of its 40 mm face (x 285 and 315 mm). A stop point the face does not cover (3 mm in from
  its edges) counts as a miss.
- **Shut** (per clamp): the handle over centre, 0.5 degree or more past vertical (its stop is at 1.5).
- **Locked** (per clamp): shut, with the centre of the pad's underside between 1.5 mm into the casting's top face and
  0.5 mm above it, and at least 4 mm inside the face's edges (the pad then presses it with 20 N). A clamp shut on
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
| Locked (each clamp) | the handle against its over-centre stop (past vertical on a square); a 0.5 mm feeler does |
| | not slide under the pad; a 1 N pull back on a spring scale at the grip leaves it shut |
| 1.0 (both locked) | also a proof test: a 20 N pull on a spring scale, straight up and then sideways away from the |
| | stops, does not move the casting (dial indicator) |
| Hands off | a 30 mm spacer against each hand; dial indicator on the casting (1 mm); 30 fps video of the last 1 s |

Possible scores: 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

## Results

This is the revised build (articulated linkage, no drive). Its first GPU run (demo 0.25: pinched handles stalled
short of dead centre) led to the present demo, which pushes the press with a closed block. The GPU column is from the
21 Nov run of that demo. The other column is an offline prediction, from the offline checker with this task's
stand-in for the physics (`task.OfflineModel`). The earlier build's GPU runs (one rigid lever and a closing drive)
do not carry over and are not reported.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Settle (`DEMO_HOLD_ONLY=1`) | arms still: casting stays on its block, clamps on their open stops | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 1394) | 1.0 (held from about step 1378 of 2800) |
| `DEMO_IDLE_ARM=right` (left alone) | seats the casting, closes C2 | 0.5 | 0.5 |
| `DEMO_IDLE_ARM=left` (right alone) | closes C1 on nothing, casting left on its block | 0.0 | 0.0 |
| `DEMO_VARIANT=no_clamps` | seats the casting, clamps left open | 0.25 | not run |
| `DEMO_VARIANT=left_only` | seats it and closes C2 only | 0.5 | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

What to read in the GPU trace: `psi1_deg` and `psi2_deg` (each handle from vertical) end at +1.5 and stay there with
both grippers away; the casting stays within 1 mm of S1 and S2 through both presses.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The handle strokes are
computed from the linkage.

| Time (s) | Left gripper | Right gripper |
|---|---|---|
| 0 to 6.5 | down round the casting on the block, grip it across its 60 mm, 10 mm left of its centre | wait aside |
| 6.5 to 18.5 | lift 70 mm, carry it clear of the clamps, set it down 6 mm short of S1 and 12 mm off S2 | wait |
| 18.5 to 22.8 | still holding it 3 mm over the plate, slide it -y onto S2 (4 mm over-travel), ease off, let go | wait |
| 22.8 to 30.6 | close the fingers into a block, push the near face +x into S1 (8 mm over-travel), back off | wait |
| 30.6 to 42.4 | over C2's open handle, pinch its grip 15 mm from the end, carry it round its pivot: 57 to 11 | wait |
| | degrees back at 12 degrees a second, then slowly on till the pad is down on the casting (2 back) | |
| 42.4 to 48.4 | let go, up over the handle's end, fingers closed into a block along the stroke, down 10 mm or | wait |
| | more behind the grip, up to its back face | |
| 48.4 to 55 | push the grip's back face square on, slowly (3 s), through the press and dead centre onto the | over C1 |
| | over-centre stop, the block's face commanded 20 mm past where the stop puts it (an arm trails | |
| | a pushed load); hold, back off, up, aside | |
| 55 to 68.8 | aside | the same on C1 (from 44.4, once the left gripper has let go of C2's grip) |

The casting is seated against S2 while still held because C2's open arm stands just beyond the casting's left face,
where a gripper pushing -y would run into it. Offline: worst IK 0.1 mm; smallest joint 4 margin 4.1 degrees; the
TCPs stay 23 cm apart and the arm capsules 6.9 cm (the right arm moves in only once the left gripper has let go of
C2's grip; the left then works behind C2's handle, y over +14 cm, the right at C1, y -2.5 cm).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 2800
```

Runtime: a trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (the demo itself is 1376
steps, 68.8 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the gaps to S1 and S2,
the casting's tilt, yaw and height, both handle angles and lock flags, both TCPs and openings, and a `hist` of the
last 3 s of those numbers in every row.

## Simplifications

- The linkage: the 207-U's two opening angles (57 and 99 degrees) are catalogue figures; the link lengths that give
  them (crank 16 mm, link 32 mm, tail 7 mm, the pivots' places) are our own solution, the same for both clamps. Arm
  reach close to the 207-U (2.14 in, 54 mm; C1's pad 52 mm from its pivot) and the 210-U (3.63 in, 92 mm; C2's pad
  91 mm). Handle lengths (110 and 145 mm), base and bar sizes are our own.
- Rigid bodies on hinges; the four-bar's loop is closed by the declared rule above, not by a fourth joint. The handle
  feels the pad's load one control step late (50 ms); the handle's spring is set so that this does not chatter.
- The press: 20 N per pad on the over-centre stop (25 to 29 N through dead centre) is a light spindle setting (est)
  for a thin-walled aluminium part. The catalogue figures, the 207-U's 375 lbf (1,670 N) and the 210-U's 600 lbf
  (2,670 N), are maximum holding capacities, not the press a clamp is set to. With both pads at 20 N and friction
  0.3 the casting holds against the 20 N proof pull of the real trial: about 42 N straight up, about 24 N sideways
  away from the stops (plate and pads); with one pad it would slide at about 12 N, so the proof pull is a check of
  the 1.0 tier only. The proof pull itself is not run in the simulation.
- The casting: a 6061 aluminium block 60 x 100 x 30 mm pocketed from below to 5 mm walls and a 5 mm top, standing in
  for a die-cast housing: (60 x 100 x 30 - 50 x 90 x 25) mm3 = 67.5 cm3, at 2.70 g/cm3 0.18 kg. The simulation uses a
  solid box collider of its outer size with that mass (only the outer faces touch anything). Friction 0.3 aluminium on
  steel, HDPE and the pad (est).
- Estimates: the spindle's stiffness and the press, the handle (0.06 kg), arm (0.05 kg) and link (0.01 kg) masses,
  the joints' damping, the casting's friction.
- The splash guard and staging block are plain boxes.

## Sources

- Machinists, O*NET 51-4041.00 (task statement and 2025 employment, 288,300):
  https://www.onetonline.org/link/summary/51-4041.00
- Computer numerically controlled tool operators, O*NET 51-9161.00 (task statement and 2025 employment, 170,000):
  https://www.onetonline.org/link/summary/51-9161.00
- Welders, cutters, solderers and brazers, O*NET 51-4121.06 (task statement; 2025 employment 437,700 for
  51-4121.00): https://www.onetonline.org/link/summary/51-4121.06
- DESTACO 207-U (arm 2.14 in, 375 lbf; clamp bar opening 99 degrees, handle opening 57 degrees):
  https://www.destaco.com/hold-down-clamps-207/product/207-U
- DESTACO 210-U (arm 3.63 in, 600 lbf): https://www.destaco.com/hold-down-clamps-210/product/210-U

## Real build and reset

A 12 mm steel plate with two hardened stops, a DESTACO 207-U and a 210-U (or clamps of the same reach) bolted at the
positions in `task.json` with their spindle pads set so that each presses the casting's top with about 20 N with the
handle on its stop (set once with a load cell under a spacer of the casting's height), the pocketed aluminium block
above, an HDPE block, a steel guard plate. Reset: open both clamps (handles back on their open stops, arms up), clear
the plate, wipe chips off the stops, lay the casting right way up on the block, centred at x 220 mm, y +245 mm, its
100 mm side along y. Check each pad's press once a session and change the clamp pads every few hundred closings.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | Text; parts, the linkage rule, `success()`, `score()`, `demo()`, variants, offline stand-in |
| `layout.py`, `scoring.py` | Every dimension and the linkage (`Clamp`); the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as jointed rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
