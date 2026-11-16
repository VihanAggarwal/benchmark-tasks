# Hang a dish-rack cutlery caddy on the rack's side rail by its two hooks

A slim hanging cutlery caddy (76 x 18 x 100 mm, three 22 mm slots) stands on the table, a fork upright in each slot. A
dish rack's side wall stands 130 mm beyond it: a round 10 mm top rail over two 6 mm wires, with a draining tray and two
plates on edge behind. The robot must lift the caddy, carry it over the rail and lower its two hooks over the rail
together, let go, and leave it hanging still against the wall with every fork still in it. The instruction names
neither the lift, the carry nor the hook alignment.

## Why it is hard

1. **Both hooks must straddle the rail together.** Each hook is an inverted U (the wall's own tab, a bridge, a rear leg)
   with an 18 mm gap over a 10 mm rail: the rail's axis has to come down 14 to 22 mm ahead of the caddy's middle, an
   8 mm window (4 mm each way from the demo's 18 mm). The rear leg's tip is bent outward as a lead-in. Lowered a few
   millimetres too far toward the robot a rear leg lands on top of the rail; too far toward the rack a tab does.
2. **The caddy must be lifted clear of the rail first.** The rear legs hang 30 mm below the bridge and the flared tips
   8 mm lower: the demo lifts the caddy 76 mm (tips 24 mm over the rail's top, room for the loaded arm's 3 to 7 mm sag)
   before it carries it across.
3. **The cutlery rides in a loaded, hanging box.** The forks stand 23 mm over the rim, 2 mm clear of the walls. Tilting
   the caddy in the grip spills them; a gripper carrying it at the rim would hit them (the pads grip 15 mm under the
   rim, 25 mm of housing clearance over the fork tops).
4. **It swings on release.** Let go 4 mm from the near leg's contact, the caddy settles until its back wall leans on the
   lower wires (about 2 to 3 degrees about the rail). Pressed 4 mm past hung (the rail stops it), then opened.
5. **Hands off.** The score is read with both grippers 30 mm clear and the caddy still for 1 s: a held caddy earns
   nothing.

## Scoring

| Condition | Rule |
|---|---|
| Seated (per hook) | rail's axis (at the hook's centre line, in the caddy's frame) in the hook's gap, within 1 mm |
| Seated, also | bridge underside 2 mm below to 3 mm above the rail's top; hook 12 mm or more inside the rail's ends |
| Seated, also | caddy facing the rack (yaw within 30 degrees) and upright within 30 degrees |
| Hung | both hooks seated, yaw within 5 degrees of the wall's normal, tilt within 5 degrees |
| Cutlery kept | all three forks' middles over the caddy's footprint, above its floor and below its rim |
| Hands off | last 1 s: fingers 30 mm or more from caddy and forks; caddy moves under 5 mm, turns under 5 deg |
| Fall (latched) | the caddy's or a fork's middle more than 50 mm below the table top |

| Outcome | Score |
|---|---|
| Hung, cutlery kept, hands off | 1.0 |
| Hung, a fork out of the caddy | 0.5 |
| One hook seated (hooked on one hook only), or both seated with the caddy crooked | 0.25 |
| Anything else (on the table, a leg or tab on the rail, hung on the far side), not hands off, or a fall | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Rail between the legs (1 mm) | look from above; a 1 mm feeler between the rail and each leg's inner face |
| Bridge on the rail (-2 / +3 mm) | a 3 mm feeler under each bridge |
| Hook inside the rail's ends (12 mm) | a rule from the rail's end to the hook's centre line |
| Facing and upright (30 degrees) | a look; a 30 degree angle gauge |
| Yaw and tilt (5 degrees) | a 5 degree angle gauge against the back wall; a spirit-level card on the rim |
| Cutlery kept | a look from above: every fork standing in the caddy |
| Hands off | 30 mm spacer per hand; 30 fps video of the last 1 s with a ruler and 5 degree wedge: no visible shift |
| Fall | a rule from the table top (50 mm) |

Sliding the caddy along the rail is not penalised. Success: 1.0 held for 1 s. Doing nothing scores 0. Hanging the caddy
without its cutlery scores 0.5 at most. A caddy that comes to rest hung with hands off scores as hung,
however it got there.

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, one trial each; rows marked not run were not run
on the GPU. The offline prediction column (`tools/offline/check_task.py`, with this task's offline stand-in for the
physics, `task.OfflineModel`) is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 426) | 1.0 |
| `DEMO_IDLE_ARM=right` | the right arm idles: nothing moves | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` | the left arm idles: the demo is the right arm's alone | 1.0 | 1.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | lowered 15 mm too far toward the robot: legs land on the rail | 0.0 | not run |
| `DEMO_VARIANT=long` | lowered 12 mm too far toward the rack: tabs land on the rail | 0.0 | not run |

One arm is enough: the caddy stands 95 mm from the right arm's base line and the left arm cannot reach it (the left
arm's side of the table is for a second job, not part of this one).

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator.

| Time (s) | Right gripper (fingers across the caddy's width) |
|---|---|
| 0 to 5 | over the caddy, down round its side walls (pad tips 15 mm under the rim, 9 mm clear each side) |
| 5 to 6.5 | grip the two side walls (68 mm commanded across 76 mm) |
| 6.5 to 8.5 | lift the caddy 76 mm (the flared tips 24 mm over the rail's top) |
| 8.5 to 12.5 | carry it 112 mm toward the rack and 100 mm to the rail's middle; settle 1 s |
| 12.5 to 15.5 | lower the hooks over the rail, 4 mm past hung (the rail stops the bridges); settle |
| 15.5 to 17 | open the fingers |
| 17 to 20.5 | straight up, then aside to (x 180, y -320 mm) |

Offline: worst IK 0.10 mm; joint 4 margin 6.4 degrees at the highest waypoints (180 mm hover); the arm's capsules stay
129 mm over its own base and 226 mm from the left arm; the rack's nearest end post is 70 mm and the tray 45 mm
inboard of the right arm's base line, clear of the arm's start pose; no warnings but the bounding-box ones (below).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: the demo is 410 steps (20.5 s of simulated time) plus 3 s of hold; a 1200-step trial is estimated at 3 to 6
minutes of wall-clock on the L40S including start-up (an estimate, not measured). Seed: nothing is random;
`object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: each hook's rail position
in the caddy's frame and its bridge gap, the caddy's yaw, tilt and base height, which forks are in, the released and
still flags, both TCPs and openings, and a `hist` of the last 3 s of those numbers in every row.

Offline warnings, read: the caddy and its forks overlap as bounding boxes (the forks stand inside it), and the
checker's proxy drops the caddy onto the tray (it knows no hooks; the task's offline model hangs it). The checker's
gripper capsule is rounder than the real housing: with the forks 23 mm over the rim the proxy sees no intersection.

## Simplifications

- Rigid bodies. The rack's side wall is a rail, two lower wires and two end posts (a real grid has a wire every 3 to 4
  cm); the tray and the two plates are fixed blocks and discs.
- The hook gap is 18 mm on a 10 mm rail: 4 mm of slack each side where real hooks have 1 to 3 mm, softened by a
  lead-in. The caddy is slim (18 mm deep) so a fork stands in it 2 mm from each wall.
- The forks are 120 mm slabs with three tines, 8 mm thick, 0.04 kg (real forks 2 to 3 mm thick): thicker so they stand
  upright in a 12 mm slot without leaning.
- Friction 0.35 / 0.30 (plastic on coated wire) and 0.30 / 0.25 (steel on plastic) (est). Mass of the caddy from its
  volume at 900 kg/m3 (polypropylene, est).
- No software rules: every part is a plain rigid body (mechkit compounds).

## Sources

No listing or standard was used: every dimension is an estimate for a slim hanging cutlery holder and a dish rack's
wire side wall (rail wire 6 to 12 mm, hooks fitting with a few millimetres of slack, side plates about 180 mm). A real
build uses any hanging holder and rack whose rail and hook gap are measured; `setup` says what to match.

## Real build and reset

**Caddy.** The product type is the dish-rack cutlery caddy that hooks over a drainer's side rail (a plastic basket with
two hook clips on its back wall). Shop ones are usually wider and deeper than ours, so print a matching one: PETG or
PP-like filament, one piece.

| Part | Size |
|---|---|
| Box | 76 (y) x 18 (x) x 100 (z) mm outside, walls and base 3 mm, open top |
| Dividers | two, 2 mm thick, 90 mm tall, giving three slots 22 mm wide and 12 mm deep |
| Hooks | two on the back wall, 12 mm wide, centred 24 mm either side of the middle |
| Each hook | tab 26 mm over the rim; 4 mm bridge, 25 mm toward the rack; 4 mm rear leg 30 mm long, 18 mm clear |
| Lead-in | the rear leg's last 12 mm bent 42 degrees outward |
| Weight | about 0.06 kg |

**Rack.** Any wire dish drainer whose side rail is a 10 mm round wire with a wire or two below it (the back wall leans
on them), or a 10 mm rod on two posts with two 6 mm rods beneath; a tray and two plates behind are optional dressing.

**Forks.** Three ordinary small stainless forks (cake, pastry or child's, 120 to 140 mm long, 2 to 3 mm thick, about
20 g). The 8 mm thickness in the sim is a stand-in so a solid slab stands in a 12 mm slot; thickness does not matter to
the scoring, which only asks that each fork's middle is still over the caddy's footprint, above its floor and below its
rim. Real thin forks lean against the slot walls and that counts as kept.

Tools: a steel rule, 1 mm and 3 mm feeler gauges, a 5 degree angle gauge or printed wedge, a spirit-level card, a phone
for 30 fps video. Reset: the caddy on the counter, back wall (hooks) toward the rack, one fork upright in each slot;
nothing is consumed. Third-party assets: none.

The stillness check (caddy under 5 mm and 5 degrees over the last 1 s) was loosened from 1 mm and 0.3 degrees so that a
person can judge it from video; the GPU demo run was made under the stricter check, so it also meets this one.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |
