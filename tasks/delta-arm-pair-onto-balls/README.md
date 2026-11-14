# Put a delta 3D printer's sprung arm pair back on a ball-stud pair of its effector

One arm pair of a delta 3D printer lies on the table, its far sockets on a fixed slider's balls, its free sockets
pulled together by two springs. One hand holds the pair open, the other slides the loose effector in, and the
springs close the sockets on one of its ball-stud pairs.

Source: FLSUN's SuperRacer user manual, step 16, "Pull the parallel arm to both sides and install it on the
slider", and its maintenance table, "Apply grease to the contact between the parallel arm and the wide angle ball
head, do it once every 15 days." The arms are 315 mm, the arm length published Klipper configurations for this
printer use. Every other size is this build's, from stock parts.

## Why it is hard

- **The push jams.** The free sockets rest 52 mm apart; the balls are 66 mm over their crowns. Pushed in, each
  ball stops on a tube's end 12 mm short of the socket line.
- **The pair must be held open.** 66 mm between the sockets (88 mm over the tubes) takes about 4 N. The demo
  aims at 92 mm over the tubes (6.2 N on each finger, computed).
- One hand cannot hold the pair open and carry the effector at once: let go, the pair snaps back to 74 mm
  (N17: one hand perches it first, then spreads).
- **Two sockets catch at once on a loose part:** each ball within about 2.5 mm of its socket's axis along the
  arms or in height, 2.5 degrees of yaw (computed). The pair swings sideways as one, 16 mm either way at its free
  ends: the hands must agree on its centre line.

## Scoring

W is the width over the tubes' outer faces at their free ends (74 mm at rest, 84 mm on a stud pair). On: each
ball's centre within 1.5 mm of its socket's axis in 3D (top-view and side-view offsets as a root sum of squares)
and at or inboard of the socket's mouth, and W within 1.5 mm of 84. A gripper is clear when its axis (midway
between its pads), over the last 38 mm of the fingers, is at least R + 30 mm from the arm pair and from the
effector; R is 16 mm closed, 57 mm fully open. An unreadable gripper is not clear.

| Score | At the end of the trial | Bench check |
|---|---|---|
| 0 | not on | caliper |
| 0.25 | on | caliper, top and side video |
| 0.5 | on, one gripper clear | the same |
| 1.0 | on, both clear, no plate corner over 1 mm off the table, all held for the last 1 s, no free socket or effector ball moving over 1 mm | the same, a 1 mm feeler |

## Results in simulation

GPU runs of this build, one trial each, run limit 900 steps (45 s): the demo command below with the variable shown.

| Run | What it does | Score |
|---|---|---|
| `DEMO_HOLD_ONLY=1` | both arms hold still | 0.0 |
| Scripted demo | the phases below | 1.0 |
| Dream-1 | the instruction | 0.0 |
| Right arm alone (`DEMO_IDLE_ARM=left`) | pushes into the closed pair | 0.0 |
| Left arm alone (`DEMO_IDLE_ARM=right`) | opens the pair, closes on nothing | 0.0 |
| `DEMO_VARIANT=N1` | one arm: the right pushes the effector 40 mm in | 0.0 |
| `DEMO_VARIANT=N10` | one arm: turn 45 degrees, slide in, turn square | 0.0 |
| `DEMO_VARIANT=N12a`, `DEMO_VARIANT=N12b` | one arm: a ball in one socket, pull 20 mm sideways | 0.0 |
| `DEMO_VARIANT=N13` | one arm: push to the jam, let go, spread | 0.0 |
| `DEMO_VARIANT=N14` | one arm: twist in as far as it goes, let go, spread | 0.0 |
| `DEMO_VARIANT=N17` | one arm: front balls set on the closed blocks, let go, spread | 0.0 |
| `DEMO_VARIANT=N4` | aimed at 98 mm | 1.0 |
| `DEMO_VARIANT=N5_86`, `DEMO_VARIANT=N5_83`, `DEMO_VARIANT=N5_80` | aimed at only 86, 83, 80 mm | 0.0 |
| `DEMO_VARIANT=N6` | left never closes | 0.0 |
| `DEMO_VARIANT=N7` | left stays at a 40 mm pad gap, right never opens | 0.25 |
| `DEMO_VARIANT=N8` | right never opens | 0.5 |
| `DEMO_VARIANT=N8b` | left closes, stays between the tubes | 0.5 |
| `DEMO_VARIANT=N9` | slide stops 8 mm short | 0.0 |
| `DEMO_VARIANT=N15` | right opens on the post, stays | 0.5 |
| `DEMO_VARIANT=N16` | spread 145 mm from the slider's balls | 0.0 |
| `DEMO_VARIANT=N18` | left at W 84, right holds the effector level 20 mm up | 0.0 |
| `DEMO_VARIANT=N19` | left at W 84, right sets the front balls on the block tops | 0.0 |
| `DEMO_VARIANT=N0` | both arms at ready | 0.0 |

`DEMO_VARIANT=N2` and `DEMO_VARIANT=N3` repeat the one-arm rows. N17 ran once, not re-aimed after calibration.
Rows with several variants: each scored the value shown. Without keeper pins the pair would leave its
slider if spread nearer than about 180 mm to it. Calibration, one arm each, not scored: `DEMO_VARIANT=C4`,
`DEMO_VARIANT=C5`, `DEMO_VARIANT=C7`.

## Demo

Open loop: timed waypoints, nothing read from the simulator, no force. Both grippers close along x,
leaning apart. The left spreads 245 mm from the pivots; the right holds the 20 mm post.

| Time (s) | Left (spreader) | Right (carrier) |
|---|---|---|
| 0 to 2.0 | over the tubes, pads 8 mm apart | over the post, open 60 mm |
| 2.0 to 3.5 | down between the tubes | down either side of the post |
| 3.5 to 5.0 | opening | closes on the post |
| 5.0 to 6.0 | opens to a 67.5 mm pad gap, aiming at W 92 | holds |
| 6.0 to 9.0 | holds the pair open | slides the effector 40 mm in |
| 9.0 to 9.5 | holds | holds |
| 9.5 to 12.0 | closes to 8 mm: sockets onto the balls | holds |
| 12.0 to 13.5 | rises | opens to 60 mm |
| 13.5 to 15.0 | away | rises |
| 15.0 to 16.5 | holds | away |
| 16.5 to 18.0 | holds | holds |

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 900
```

Dream-1: `--policy dream`; `TASK_TRACE=trace.jsonl` logs each step.

## Simplifications

- The far sockets are hinges on the slider's balls (on the bench, keeper pins 0.5 mm off the tubes). One arm
  is hinged on the other, so the pair swings as one rigid piece; a real pair stays a parallelogram.
- So the +x arm's free socket stands further from the slider than the other's: 0.9 mm on a stud pair, 1.6 mm at
  the W 92 hold; closed on a stop pin, 2.6 mm either way (computed). A real pair has none.
- One hinge spring, 0.2 N/mm at the sockets with a 1.2 N preload, gives both springs' pull. The drawn spring
  rods and stop screw carry no force; the stop is the hinge's limit.
- Square 8 mm socket holes; each ball three crossed 10 mm discs; solid tubes of the same mass.
- The effector's 139 g is spread by shape volume: plate 87, post 29, stud pairs 23 g (bench 38, 25, 75 g);
  centre of mass 11.5 mm up, 2.1 mm back (bench 14.2, 1.8).
- The rest bar is built 0.5 mm low: the tubes ride just above it (on the bench they lie on it).
- A tall post to hold the effector by; no hot end or towers. A printer's pairs hang; here one lies level.

## Real build and reset

- Slider: aluminium plate 50 x 30 x 8 mm; keeper pins: two 4 x 30 mm steel dowels, 8 mm into the bench, 44.5 mm
  either side of centre.
- Stud pairs (four): 34 x 10 x 10 mm aluminium bar, two DIN 71803 ball studs (10 mm ball), 56 mm apart.
- Arms: 8 mm square carbon tube, 1 mm wall, 329 mm long; nylon blocks 14 x 14 x 3 mm, 8 mm holes.
- Springs: 0.6 mm music wire, 6.6 mm across, 25 coils, 26.4 mm inside the loops, 0.24 N/mm, 1 N initial
  tension, on M3 eye bolts 30 mm from each socket, each eye 16.3 mm from the tube's inner face.
- Stop screw: M4 nylon, 20 mm from one free socket, set for 52 mm between them.
- Rest bar: 120 x 15 x 9 mm acetal, 175 mm from the slider's balls; 4 x 20 mm steel stop pins, 8 mm into the
  bar, 50 mm either side.

Reset: as `reset_notes` in `task.json`.

## Files

`task.py` (objects, hooks, demo), `layout.py` (sizes), `scoring.py`, `offline_delta.py` (motion offline), kit
files `mechkit.py`, `rigstate.py`, `tiltgrip.py`; tests: `python3 -m unittest discover -s tests`.

Third-party assets: none. Nearest board tasks: TK-DKQNQY, a circlip spread with pliers (one tool, no second
part); TK-GPFKUG, a spring pusher held back while cartons go in (no fit, no seat).
