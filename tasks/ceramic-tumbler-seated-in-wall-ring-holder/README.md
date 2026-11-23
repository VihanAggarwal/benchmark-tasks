# Put a clean tumbler back in its wall ring: lower it upright until the taper wedges

Maids and housekeeping cleaners (O*NET 37-2012.00; 1,332,100 US workers in 2025 per O*NET OnLine,
https://www.onetonline.org/link/summary/37-2012.00) "Replenish supplies, such as drinking glasses, linens, writing
supplies, and bathroom items." Where a bathroom has a wall ring holder, the clean tumbler goes back in the ring.
Seating matters: a cup left on the ring's top or crooked falls at the next bump onto the basin, and a
chipped or broken ceramic cup is a cut hazard, thrown away.

A bathroom tumbler hangs in a wall ring holder: lowered base first, mouth up, through a ring only 4 mm wider (each
side) than its base, it stops where the ring meets its taper and stays there with nobody holding it. Here a hollow
ceramic tumbler (100 mm tall, 60 mm across at the base widening evenly to 76 mm at the rim, 0.28 kg) stands on a
vanity top (the station table) in front of the right arm. A chrome ring (68 mm inside, 8 mm band, 3 mm thick) on an
88 mm strap, screwed through a flange to a plywood backboard standing for the bathroom wall, is fixed 90 mm over the
table. The robot picks the tumbler up, carries it over the ring, lowers it until the ring wedges on the taper (50 mm up
from its base, so the base hangs 40 mm over the table), lets go with both hands 30 mm clear, and backs away. The
instruction names neither the seating height nor the order of the steps.

## Why it is hard

1. **The ring grips a slope, not a ledge.** The wall leans out only 4.6 degrees (8 mm over 100 mm). The base passes the
   ring with 4 mm to spare each side; 50 mm up the ring touches. A tilted or off-centre tumbler is caught by the ring's
   edge higher up (2 mm sideways and the ring touches 25 mm early, leaving the cup caught 25 mm high; 4 mm and the
   base lands on the ring's top) and hangs crooked or tips out.
2. **The seat is a wedge, so the arm has to give.** The ring stops the tumbler wherever it is, and an arm that keeps
   pressing only jams it. The demo lowers 3 mm past the seat and lets the arm yield (the ring stops the cup).
3. **The wall behind limits the hand.** The backboard stands 373 mm out; the placing arm's wrist (link 4) reaches about
   100 mm past its gripper toward it (32 mm to spare at the lowest point). The hand comes in from the front, grips the
   upper wall (pads 62 to 100 mm up it, above the ring when hung), and has to back away without touching the cup.
4. **Hands off.** The score is read with both grippers 30 mm clear of the cup and the holder and the cup still for 1 s:
   a held cup earns nothing, and dropping it from over 20 mm onto the ring (the ceramic chips) scores 0.

## Scoring

| Condition | Rule |
|---|---|
| Hung | underside's centre 32 to 48 mm over the table (seat 40 mm; on the table 0), axis within 60 degrees of up |
| Tilt | the angle between the tumbler's axis and the vertical |
| Off-centre | distance from the tumbler's axis to the ring's centre line, taken at the ring's top face height |
| Caught | underside's centre 20 mm or more up, within 60 mm of the ring's axis, axis within 60 degrees of up |
| Hands off | last 1 s: finger zones 30 mm or more clear; tumbler moves 1 mm or less, turns 0.5 degrees or less |
| Hit (latched) | base down 20 mm or more in 0.15 s, 130 mm/s or faster at the end, then stopped on a support |
| Holder moved (latched) | the holder moves more than 5 mm |

| Outcome | Score |
|---|---|
| Hung, tilt 6 degrees or less, off-centre 6 mm or less | 1.0 |
| Hung, tilt 15 degrees or less, off-centre 12 mm or less (otherwise) | 0.5 |
| Caught on or in the ring, otherwise | 0.25 |
| Anything else (on the table, upside down, lying, away from the ring) | 0 |
| Hit or holder moved (latched), or not hands off and still | 0 |

Bench check for each condition on a real build (the simulator reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Hung (32 and 48 mm) | steel rule from the table to the tumbler's underside |
| Tilt (6 and 15 degrees) | phone level laid across the rim |
| Off-centre (6 and 12 mm) | rule from the ring's edge to the wall at both ends of a diameter; half the difference |
| Caught (20 and 60 mm, 60 degrees) | rule for the base height, tape from the ring's axis, a 60 degree angle gauge |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s; dial indicator on the rim (1 mm, 0.5 deg) |
| Hit (20 mm in 0.15 s, stopped) | 30 fps video against a 20 mm block: base down 20 mm in 4 frames, then still |
| Holder shift (5 mm) | rule from a mark on the table to the backboard |

Success: a score of 1.0 (the score already needs the 1 s hands-off and still window). Doing nothing scores 0. The
two poses the scorer reads are the tumbler's and the holder's; the arms are read only for the let-go check.
Unintended strategies earn nothing: setting the tumbler mouth down on the ring (its 76 mm rim is wider than the
hole) is upside down: 0; leaving it in the gripper fails hands off; dropping it
in from 30 mm is a latched hit (a fast hand that stops in free air is not: the hit needs the stop on the table, its
lowest point within 5 mm of it or below it, or on or in the ring, within 71 mm of its axis and its lowest point no
more than 5 mm over its top); bending or moving the holder is out of reach (it is screwed down) and is 0 if it moved.

## Results

Run on the station (Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1000-step limit), one trial each, on 17 Nov
2026 (L40S). The offline prediction column (`tools/offline/check_task.py` with this task's offline stand-in for the
physics, `task.OfflineModel`, plus the tests) is not a measurement; the GPU column is.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run: the tumbler stays put) |
| Scripted demo | below | 1.0 | 1.0 (first by step 487 in the sampled trace; the run ended at step 501) |
| `DEMO_IDLE_ARM=left` | the right arm alone, as written | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=right` | the left arm alone: no plan, the tumbler never moves | 0.0 | 0.0 |
| `DEMO_VARIANT=high` | lets go 30 mm over the seat | 0.0 (hit) | not run |
| `DEMO_VARIANT=keep` | lowers it into the ring and keeps holding | 0.0 (not hands off) | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 (the tumbler never moved in 1000 steps) |

At the end of the GPU demo the tumbler hung with its base 41.5 mm over the table (seat 40 mm by geometry), tilted
0.18 degrees, 0.01 mm off-centre, with both grippers 45.8 mm clear and nothing latched.

One arm is enough: the right arm does every step; the left stays at its ready pose and only has to keep out of the
way (it is 46 cm from the right gripper at the closest). The task does not claim two arms are needed. The tumbler
stands 30 cm right of the midline, out of the left arm's reach, so the left arm cannot do it alone.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The seating height comes from
the cone's geometry (the ring meets it 50 mm up), not from a sensor.

| Time (s) | Right gripper (fingers close along y) |
|---|---|
| 0 to 7.0 | over the tumbler (x 220 mm, y -300 mm); down; pad tips 62 mm up its wall; close to 68 mm (rim is 76) |
| 7.0 to 8.5 | lift 105 mm straight up: the base ends 15 mm over the ring's top |
| 8.5 to 11.5 | carry it over the ring (x 240 mm, y -140 mm); settle |
| 11.5 to 13.5 | down 10 mm to 5 mm over the ring's top; settle, centred over the hole |
| 13.5 to 18.5 | slowly down 43 mm through the hole, then 15 mm more (12 mm to the seat, 3 mm past it); settle |
| 18.5 to 20.0 | open the fingers: the ring holds the tumbler |
| 20.0 to 23.5 | straight up 58 mm (open pad tips 17 mm over the rim), back 140 mm to x 100 mm: finger zones clear |

Offline: worst IK 0.09 mm; the smallest joint margin is 9.8 degrees (joint 4, at the 180 mm carry height, the
highest the arm goes; the reach there ends near 200 mm, which is why the ring is mounted only 90 mm up); the wrist is 32
mm from the backboard at the lowest point; the demo is 23.5 s (470 steps), and on the GPU the score reached 1.0 by
step 487, once the 1 s hands-off window had filled.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the tumbler in the wall ring, mouth up, until the ring holds it, and let go." \
  --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/tumbler-ring/demo
```

For Dream-1 use `--policy dream` with the same flags. Switches: `DEMO_HOLD_ONLY=1` (a settling check: nothing moves,
the tumbler must stay put), `DEMO_IDLE_ARM=left|right`, `DEMO_VARIANT=high|keep` (counterfactuals, never the submitted
demo), `TASK_TRACE=trace.jsonl` (one JSON line per control step: base height, tilt, off-centre, hands gap, largest
drop, both TCPs and openings, and a `hist` of the last 3 s in every row).

Runtime per trial: about 1.5 to 3 minutes of wall-clock on the L40S including start-up (the demo itself is 23.5 s of
simulated time, the run ended at step 501, 25 s); the first start of Isaac Sim on a new instance takes several
minutes more.
Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed
is needed.

## Simplifications

- Rigid bodies. The tumbler is a hollow 32-sided cone: a 5 mm floor disc and 32 thin slabs leaning 4.57 degrees, so
  the ring meets a real slope (a wedge that centres the cup), within about 2 mm of height from the facets. Glaze and
  the foot ring are left out. The ring is a 32-sided ring of boxes, the strap, flange and backboard boxes. Screw heads
  are left out.
- The ring is 90 mm over the table so the arm's straight-down reach covers it (about 200 mm over the table at this
  distance); a real holder is often higher.
- Not one catalogue product: a generic chrome ring holder and tapered ceramic tumbler at the stated sizes. Catalogue
  ring holders differ (for example HEWI's holder for tumbler insert 477.00.100 has a 74 mm opening).
- Friction 0.40 ceramic, 0.30 chrome (est). Mass 0.28 kg for the tumbler (est: about 109 cm3 of ceramic in a 5 mm
  wall and floor).
- Chipping is not simulated: an impact on the ring or the table is detected from the tumbler's motion and scores 0.
- No software rules: no weld, bond or drive. The one scripted number is the demo's timing.
- The arms are read for the let-go check only (rigstate.py); the offline stand-in in task.py never runs in the
  simulator.

## Sources and estimates

Wall ring tumbler holders and tapered ceramic tumblers are ordinary bathroom fittings; the sizes here are this task's
build sizes, chosen inside the range sold, not copied from one product. Estimates (stated in fidelity): the two
frictions, the tumbler's mass, the strap (16 x 3 mm), the flange (36 x 60 x 3 mm) and the backboard's size (400 x 300
x 18 mm). Occupation, worker count and the quoted task line: O*NET OnLine, 37-2012.00 Maids and Housekeeping Cleaners
(https://www.onetonline.org/link/summary/37-2012.00). The 74 mm HEWI opening: the 477.00.100.98 listing at
https://www.galvinengineering.com.au/hewi-holder-for-soap-dish-insert.

## Real build and reset

A chrome-plated ring holder with a 68 mm hole (or a ring cut from 3 mm sheet on a strap), screwed to an 18 mm plywood
backboard that is clamped upright on the bench, ring top 90 mm over the bench; a glazed ceramic tumbler tapering 60 to
76 mm over 100 mm; a steel rule, a phone level, 30 mm and 20 mm spacer blocks. Reset: stand the tumbler upright on
its mark at x 220 mm, y -300 mm; dry the ring and the tumbler; replace a chipped tumbler.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and threshold; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` (about 1.5 minutes) |

Third-party assets: none.
