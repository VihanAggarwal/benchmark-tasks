# Slide a quarter sheet pan onto a toaster oven's lower rails, level and all the way in

A small countertop oven stands with its open front face toward the arms, its drop-down door open and lying flat in
front of it (hinged at the front bottom edge, a 170 mm long slab level with the oven floor). Inside, a pair of rails on
each side wall carries a pan; the cavity is 344 mm wide and 245 deep, with a back wall and a roof. On the table, clear
of the open door (6 mm short of its free edge, 176 mm short of the oven's face), lies a quarter sheet pan (9-1/2 x 13 x
1 in, rolled rim, 0.20 kg). The robot must leave the pan on the **lower** rails, level, pushed all the way in so the
oven door could shut, with both hands off it. The instruction names
neither how to hold the pan nor the push.

## Why it is hard

0. **The open door is in the way.** The pan cannot start in front of the face: the open door lies there, so the pan
   starts on the table clear of the door's free edge and is lifted and carried over the door (the carried pan's
   underside is 102 mm over the table, the door's top 20 mm) before it goes onto the rails.
1. **Nothing can go beside or above the pan once it is in.** A sheet pan is carried level by its two short ends, as a
   person carries one, but the side walls stand 7 mm from those ends and the roof is over the rim: the gripper and
   the arm above it cannot follow the pan into the cavity. So the pan is carried by its ends, set on the rails with
   its trailing edge still 95 mm outside the face (its centre 26 mm inside it), let go, and pushed the last 99 mm on
   its near rim with closed fingers, the hand pitched forward so that it reaches in under the roof (`tiltgrip.py`).
2. **A rail under each short end.** The pan's floor overlaps each rail by 11 mm and stands 7 mm from each wall: carried
   27 mm
   over the rails and lowered, it must be within 7 mm of the centre line at the face, or it jams on a wall edge. A
   pan set on one rail tips; on the cavity floor or on the upper rails it scores nothing.
3. **Home means the door could shut.** No part of the pan may stand in front of the face; the cavity is only 3.7 mm
   deeper than the pan, so "all the way in" is: no part of the pan in front of the face (its leading edge is then
   within 3.7 mm of the back wall, touching it or not). A pan left part out scores at most 0.5.
4. **Hands off.** The score is read with both grippers 30 mm clear and the pan still for 1 s: a held pan earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| On the rails | underside corners from 3 mm below to 12 mm above the lower rails' top; over each rail's inner edge by 2 mm along 50 mm; inside the walls |
| In | the pan's centre at least 30 mm inside the oven's front face |
| Level | the pan's floor within 2 degrees of horizontal, face up |
| Home | no part of the pan in front of the oven's front face (the door could shut; the leading edge is then within 3.7 mm of the back wall) |
| Near | the pan's trailing edge no more than 40 mm in front of the face |
| Hands off | last 1 s: finger zones 30 mm or more from the pan; it is still: moves under 5 mm and turns under 5 deg |

| Outcome (the first rule that holds sets the score) | Score |
|---|---|
| On the rails, in, level, home | 1.0 |
| On the rails, in, level, not home, near | 0.5 |
| On the rails and in, otherwise (tilted more than 2 degrees, or more than 40 mm out) | 0.25 |
| Not on the rails, not in (centre under 30 mm inside the face), not hands off, or the pan ever fell off the table | 0 |

Every end state has one score. A pan that is off centre never changes the tier by itself: a wall stops it 7 mm out and
the rails still carry it; off centre and not home is simply 0.5 when it is level and near. Success: 1.0 held for 1 s.
Doing nothing scores 0.

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| On the rails (3 / 12 / 2 mm) | torch along each rail from the front (both short ends on the rails, none dropped below their top); a 12 mm block must not fit under any corner |
| In (30 mm) | a rule from the face to a tape mark on the pan's centre line |
| Level (2 deg) | spirit level on the pan floor, or a 12 mm gauge under one short end (2 degrees over 330 mm) |
| Home (0 mm) | a straightedge laid across the front face: no part of the pan touches or passes it |
| Near (40 mm) | a rule from the face to the trailing edge |
| Hands off (30 mm, 1 s, still) | 30 mm spacer block against each hand; 30 fps video of the last 1 s: no visible motion of the pan against a taped mark on the rail (5 mm and 5 deg are wider than the eye can see); relaxing this never lowers a score |
| Fall (20 mm, latched) | by eye against a rule |

## Results

The offline prediction column is `tools/offline/check_task.py` with this task's offline stand-in for the physics
(`task.OfflineModel`). It is not a measurement. The GPU column is the physics simulator (Isaac Lab, 20 Hz control).
The geometry changed (open drop-down door added, oven moved 20 mm out, pan start moved back to x 0.008 m, stillness
limits relaxed to 5 mm and 5 degrees); the GPU column below is the full re-run on the current geometry.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run) |
| Scripted demo | below | 1.0 (held from step 610) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | one hand on one end cannot lift the pan level | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the same | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | pushed only until the trailing edge is 20 mm out of the face | 0.5 | not run |
| `DEMO_VARIANT=half` | pushed only until the trailing edge is 70 mm out of the face | 0.25 | not run |

With one arm idle the pan is not lifted level, so both hands are needed to carry it: on the GPU it scored 0.0 with
either arm idle.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both hands run the same
plan, mirrored about y 0, on one clock: carrying, the left hand holds the left short end (y +157 mm, fingers closing
along y across the side bead, 40 mm in from the trailing edge) and the right hand the right end; pushing, the hands
are at y +-110 mm on the near rim.

| Time (s) | Both hands |
|---|---|
| 0 to 7.0 | over each short end, down round its side rim (fingers open 94 mm, pad tips 11.5 mm over the underside), grip the 16 mm bead |
| 7.0 to 9.0 | lift the pan straight up until its underside is 27 mm over the lower rails |
| 9.0 to 14.5 | carry it in (slowly: 55 mm/s, so the pan does not pitch nose down) along x until its trailing edge is 95 mm outside the face (the grippers 15 mm or more outside it) |
| 14.5 to 16.5 | lower it onto the rails (2 mm past contact), settle |
| 16.5 to 19.0 | open to 26 mm, up 45 mm |
| 19.0 to 24.5 | close the fingers, turn and pitch the hand forward 55 degrees, in to y +-110 mm, down to the trailing face (corner 12 mm up, 10 mm short) |
| 24.5 to 27.5 | push at 40 mm/s, commanded 10 mm into the back wall (the wall stops the pan after 99 mm) |
| 27.5 to 30.0 | settle, back off 80 mm, away |

Offline: worst IK 0.1 mm; joint 4 margin 11.2 degrees; the arms' capsules stay 4.0 cm apart; TCPs at least 22 cm
apart; the checker's capsule proxy (deliberately fat capsules) overlaps each fingertip with its own arm base at
the first grip, because the pan starts 8 mm in front of the arm-base line. In the physics run the same grip went
through with no fallback reset and a valid state at every step, and the demo first scored 1.0 at step 596.
The plan is 600 steps; the offline success hold starts at step 610 (the arms hold their last pose after the plan
ends; the checker's run is 660 steps), under half the 1200-step limit.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: not measured yet; the demo is 600 steps (30.0 s of simulated time), about 2 to 4 minutes of wall-clock on the
L40S including start-up (estimate from the other mechkit tasks' runs). Seed: nothing is random; `object_jitter=0`
fixes the start layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the pan's pose, its
leading gap and trailing overhang, tilt, rail overlap, both TCPs and openings, and a `hist` of the last 3 s of the key
numbers in every row.

## Simplifications

- Rigid bodies. The pan has straight walls 4 mm thick (a real pan's taper and are about 1 mm) and its rolled rim is a
  solid bead 16 mm
  wide and 10 mm tall (est), so that the grippers (16 mm or more) have something to close on. A real quarter sheet pan
  weighs less than a thick bar; its mass is an estimate (0.20 kg). Friction 0.60 static, 0.47 sliding, aluminium on
  steel (dry, engineering tables).
- The oven is a fixed steel box with solid feet; its rails are 4 mm plates 18 mm deep. The drop-down door is in the
  scene, open, as a fixed flat slab 170 x 374 x 20 mm lying in front of the face with its inner face level with the
  oven floor (hinge and glass not modelled). Heating elements, wire rack and controls are left out. "Home" keeps the
  pan clear of the door's plane, so that the door could shut; its closing is not simulated.
- The oven is a small countertop model; its dimensions are estimates chosen so that a quarter sheet pan fits with
  7 mm each side (the sizes of real units vary). The rails' height (55 mm over the floor) and the upper pair's (65 mm
  higher) are estimates.
- The hands enter the cavity only with the fingers closed and pitched; the gripper housing never goes under the roof.
  That is a limit of the scripted demo's top-down grips, not a property the task depends on: any policy may reach in
  as it likes.

## Sources

- Pan: a standard quarter-size (1/4) sheet pan, bare 18-gauge aluminium, 13 x 9-1/2 x 1 in (330 x 241 x 25 mm) with a
  rolled rim, as sold for commercial and home baking (baking-supply listings). Acceptance check with calipers and a
  scale: 330 and 241 mm each within 3 mm, rim about 16 mm wide (13 to 19 mm), mass 0.2 to 0.3 kg (the simulator
  uses 0.20 kg; record the real one). Its short ends are the grip points. Bearing check with calipers on the underside:
  the pan rests on the rails by its flat floor, whose outside width across the rails must be 320 to 330 mm (wall taper
  at most 5 mm a side, so the floor overlaps each rail's inner edge by 6 mm or more; a pan with a narrower floor is not
  used) and whose depth is 25 mm, 3 mm either way. The simulated pan has straight walls, so its floor is the full 330 x
  241 mm and overlaps each rail by 11 mm.
- Oven: a countertop toaster oven of the "fits a quarter sheet pan (9 x 13 in)" class, about 20 to 25 L. The cavity
  (344 wide x 245 deep x 170 tall, rails 55 mm over the floor, 18 mm deep, 308 mm between their inner edges, upper
  pair 65 mm higher) is an estimate of such an oven, not a model of one product. Acceptance check with a tape and
  calipers: each of these within 3 mm, the pan seated on the lower rails with 11 mm on each rail, 7 mm (3 mm either
  way) from each wall and 3.7 mm (up to 6 mm) of play front to back, and the door open flat 170 mm long. If no oven
  matches, build a steel or wood box to these sizes with rails of steel angle and a hinged flap for the door.

Estimates (stated in fidelity): the pan's mass, wall and bead sizes, the friction, the oven's cavity and rail
dimensions.

## Real build and reset

A countertop oven with its door open and flat, or a box with two pairs of rails (steel angle 18 mm deep) in a 344 mm
wide cavity 245 mm deep, a back wall, a roof 170 mm over the floor and a flap 170 mm long for the open door, and a
quarter sheet pan, plus a ruler and a 12 mm block. Reset: the oven's front face at x 0.425 m, the open
door's free edge at x 0.255 m, the pan flat on the table, centred on y 0, its trailing edge at x 0.008 m (leading edge
6 mm short of the door's free edge). Nothing is consumed.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; the oven and the pan, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the oven and the pan as compound bodies; read both grippers each step |
| `tiltgrip.py` | Wraps `yam_arena.demo.grip_rotation` so the demo's push can be pitched forward |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
