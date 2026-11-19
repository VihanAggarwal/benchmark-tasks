# Fit a B22 bulb into a ceiling pendant's lampholder that hangs free on its flex

A plain white pendant set (ceiling rose, 240 mm of round flex, a B22d plastic lampholder, mouth down) hangs from a
rigid stand that stands in for the ceiling, the holder's mouth 178 mm over the table. A new A60 B22 LED bulb (60 mm
across, 104 mm long, 40 g) stands cap-up in a ring stand on the table. The robot must fit the bulb so it locks in,
the way a B22 bayonet bulb goes in: push it up against the two spring plungers, turn it about 15 degrees, let it drop
into the hooks. Then both hands off, the holder hanging where it hung.

## Why it is hard

1. **Nothing to push against.** The holder hangs on a soft flex. Pushed from below it rides up the flex (the
   plungers' preload is more than the 60 g that hangs below the flex's slack), and turned it spins round on the
   flex. One hand must hold the holder steady while the other pushes and turns the bulb.
2. **Order.** The pins must go up the J's vertical legs first, then turn at the top (turned at the bottom they meet
   the J's tooth: the J is missed), then drop into the hooks. Let go before the turn and the plungers push the bulb
   out.
3. **Two grips close together.** The holder is 52 mm long and the bulb is pushed into it from below: the steadying
   grip (40 mm over the mouth) and the bulb grip (on the globe) are under 100 mm apart in height.
4. **Do not over-handle the holder.** Pulled down more than 15 mm the flex is strained at the rose (0.25); turned
   more than 90 degrees it is wound (0.5).

## Scoring

| Condition | Rule |
|---|---|
| Locked | both pins in the hooks of their J-slots (both hook floors 80% or more of their travel in under the pins), the bulb turned 13 to 17 degrees from the J's leg, its cap's end within 0.5 mm of where the pins rest on the hook floor, its axis within 5 degrees and 1.5 mm of the holder's |
| Released | on every step of the last 1 s no gripper's finger zone (a capsule round the last 38 mm of its fingers, larger than the fingers) within 2 mm of the bulb or the holder (its body, plungers and hook floors) |
| Hanging | over the last 1 s the mouth moved under 10 mm; at the end it is within 30 mm (top view) and 15 mm (height) of where it hung at the start |
| Dropped (latched) | after the bulb has left its ring stand (lifted 8 mm): its lowest point within 2 mm of the table, or its globe (a 60 mm sphere) within 1 mm of either stand, set back in its ring included |
| Strained (latched) | the mouth more than 15 mm below where it hung at the start, at any step |
| Wound (latched) | the holder turned more than 90 degrees about its flex, at any step |

| Outcome | Score |
|---|---|
| Locked, not strained or wound, released and hanging | 1.0 |
| Locked, not strained, but wound, or not both of released and hanging | 0.5 |
| Locked, but strained | 0.25 |
| Dropped, or not locked | 0 |

Success: 1.0 held for 1 s. Doing nothing scores 0 (the bulb starts in its ring stand).

Bench check for each condition on a real trial (the pendant is not wired: the rose is a dummy and nothing is live):

| Condition | Bench method |
|---|---|
| Locked | a 5 N spring-scale pull straight down does not bring the bulb out; the paint mark on the cap sits 15 degrees round from the matching mark at the mouth |
| Released, hanging | 30 fps front video of the whole trial, read over its last 1 s, with a 10 mm scale on the stand beside the holder |
| Dropped | the same video: any contact of the bulb with the table or a stand after it leaves its ring stand |
| Strained | a 15 mm mark on the stand at the holder's resting height, checked on every frame |
| Wound | a stripe painted down the holder and flex, checked on every frame (more than a quarter turn counts) |

## The mechanism

The pendant is one fixed-base articulation hung from the rose:

| Joint | What it is | Values (est) |
|---|---|---|
| `swing_x`, `swing_y` | the flex swinging at the rose | free, 1.2 rad each way, light damping |
| `slack` | the flex buckling when the holder is pushed up | free slide, up to 40 mm |
| `stretch` | the flex pulled tight against the rose | 600 N/m spring, 20 mm (15 mm takes 9 N) |
| `holder` | the holder's twist about its axis | 0.0005 N m per degree spring, 172 degrees each way |
| `plunger_a`, `plunger_b` | the two spring plungers in the holder's roof, 11 mm apart | long soft springs: 10 N/m, 5 mm travel, driven 40 mm below their rest stop: 0.4 N each at rest, 0.42 N at the hooks, 0.45 N at full travel |
| `catch_a`, `catch_b` | each J-slot's hook floor with two 1 mm lips | 2.5 mm slide, driven in or out by the J rule |

The holder is a round wall of 12 flat slabs (38 mm across, a 28 mm bore) with a 45-degree funnel at the mouth, a
25.5 mm collar at the top of the bore (chamfered underneath, 1.75 mm of play round the cap) that keeps the cap's end
near the axis, and a roof 7 mm over the plungers' tips. Nothing rigid on the holder is in the cap's path from the
plungers' rest to 6 mm up (a test checks it for a cap up to 0.9 mm off the axis, 1.5 mm across the pins, or tilted
2 degrees), so the push goes into the plungers' springs. The bulb is a 22 mm cap with two pins (26 mm tip to tip,
9 mm from the cap's end), a neck and the globe as stacked cylinders.

**The J rule (scoring.Bayonet).** The J-slot's shape is declared, not drawn: every control step a rule reads the bulb
in the holder's frame (offset, tilt, the cap's end height over the mouth, the turn) and drives the hook floors:

- In the bore: the cap's axis within 2 mm and 6 degrees of the holder's, the pins over the mouth. Entering, the pins
  must be at the leg (turn within 5 degrees), else the J is missed until the bulb comes out again.
- Turned past 5 degrees with the cap's end under 3 mm up the plungers' travel (the pins still under the J's tooth),
  turned back past -5 degrees or on past 20 degrees: forced (missed).
- From the leg, turned 13 to 17 degrees with the cap's end 3 mm or more up (the pins over the lips): hooked. The hook
  floors slide in under the pins (1.5 mm under them) and stay in while the turn stays within 9 to 21 degrees and the
  bulb stays in the bore.

Everything else is contact: the cap's end on the plungers, the pins on the hook floors between the lips (which stop
the bulb turning back out), the bore and the collar round the cap, the fingers on the holder and the globe. Let go in
the hooks, the plungers push the pins down onto the hook floors (2 mm up their travel) and hold the bulb there.

The plungers are softer than a real holder's (est) to suit the simulated arms, which are springs at the TCP: in the
first two GPU runs (stiffer plungers, 1.5 N each at the hooks; in the first also a tighter collar) the right arm
stalled 9 to 11 mm under its command at about 1.4 N and the left gave 4 to 6 mm with the holder, so the cap never got
0.5 mm up the plungers. Their 0.8 N preload is still more than the 0.59 N hanging below the flex's slack, so a free
holder rides up instead.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both grips are pitched side
grips whose closing axis stays level. The left reaches forward and to the right (heading -60 degrees, 40 degrees
down) and pinches the holder 40 mm over its mouth (30 mm commanded across its 38 mm), then draws it down a little
against the flex (commanded 7 mm; the arm's give leaves a pull of about 1 N and the holder 1 to 2.5 mm down) so that
the push does not lift it. The right reaches forward and to
the left (heading 75 degrees, 20 degrees down) and pinches the globe 84 mm under the cap's end (52 mm commanded across
its 57.4 mm); it turns the bulb by turning its heading from 75 to 90 degrees about the bulb's axis. At the lock the
two grips are 97 mm apart in height (74 mm as commanded at the push). The right aims the bulb 0.4 mm behind and
1.5 mm to the right of the holder's axis: where the left's grip holds the holder in the simulator.

| Time (s) | Right gripper |
|---|---|
| 0 to 3.5 | behind the bulb on its stand; fingers either side of the globe; settle |
| 3.5 to 6.2 | grip the globe; lift it 15 mm out of the ring |
| 6.2 to 9.7 | under the holder, the pins lined up with the J's legs; up into the mouth |
| 9.7 to 12.2 | the cap's end onto the plungers; push up to the plungers' stop (commanded 13 mm past it: the arm sags about 2 mm under the bulb and gives 3 to 8 mm more under the push); hold |
| 12.2 to 14.2 | turn it 15 degrees at the top; hold |
| 14.2 to 15.7 | ease down to 1 mm under the hook floor: the pins drop onto it; settle |
| 15.7 to 19.9 | let go of the bulb; back out along the approach; aside |

| Time (s) | Left gripper |
|---|---|
| 0 to 3.5 | beside the holder; fingers either side of it; settle |
| 3.5 to 5.5 | grip the holder; draw it down a little against the flex |
| 5.5 to 17.5 | hold it steady while the bulb goes in, until 0.3 s after the right hand has let go |
| 17.5 to 21.9 | ease the pull off; let go of the holder; back out along the approach; aside |

The plan is 21.9 s (438 steps); the run limit is 800 steps (40 s at 20 Hz).

The push was sized with a quasi-static model of both arms as springs at the TCP, fitted to the first two GPU runs
(right about 150 to 200 N/m, left 165 to 205 N/m; it reproduces both runs' stall). For either arm anywhere from 120 to
300 N/m it puts the cap's end at the plungers' 5 mm stop with the holder between 1.2 mm down and 5.1 mm up from where it
hung, the push under 3.5 N; at the fitted values the holder rises under 2 mm (tests: ArmGive).

Offline checks (`tools/offline/check_task.py --strict --one-arm`): worst IK under 0.1 mm; minimum joint margin 31.9
degrees (left) and 20.2 degrees (right); arm link capsules 12 mm apart at the closest (during the turn, at the
commanded positions: in the simulator the right sits several mm under its command and the left over its). True
clearances from the shapes along the demo: the right gripper 6.8 mm from the holder, 46.6 mm from the ceiling stand;
the left 65.5 mm from the ceiling stand. The checker's straight-down reach and bounding-box interference warnings do
not apply to pitched grips and to a pendant hanging under its stand.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 800
```

Seed: nothing is random; `object_jitter=0` fixes the start layout. For Dream-1 use `--policy dream`.
`TASK_TRACE=trace.jsonl` writes one line per control step: the J rule's state, the bulb in the holder (turn, the cap's
end over the plungers, offset, tilt), the hook floors and plungers, the holder's twist and sag, both TCPs and openings.
`DEMO_HOLD_ONLY=1` holds both arms still (settle check); `DEMO_IDLE_ARM=left|right` drops one arm's plan;
`DEMO_VARIANT=noturn|early` runs a wrong way (never the submitted demo).

## Results

GPU run 3 on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each, beside the offline
prediction (`tools/offline/check_task.py` with `task.OfflineModel`, a rigid stand-in for the physics). Two earlier
builds, with stiffer plungers that the arms could not push in, scored 0.0 in the demo (see The mechanism).

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (settle) | both arms hold still | 0.0 | 0.0 (nothing moved) |
| Scripted demo | above | 1.0 (held from step 444) | 1.0: plungers pushed 5.0 mm in, turned 14.3 deg, hooked, catches fully in (2.5 mm), cap 2.0 mm up at rest, 0.47 mm off axis; holder twist under 0.5 deg |
| `DEMO_IDLE_ARM=left` | the right pushes the bulb up into a free holder: it rides up on the flex and turns with the bulb, the pins never reach the hooks; let go, the bulb falls | 0.0 (dropped) | 0.0: the cap never rose more than 0.1 mm into the plungers; the turn was forced (missed) and the bulb fell when let go |
| `DEMO_IDLE_ARM=right` | the left holds the holder and lets go; the bulb stays on its stand | 0.0 (not locked) | 0.0: the bulb stayed on its stand |
| Dream-1 | `--policy dream`, full 800-step limit | not predicted | 0.0: the bulb stayed on its stand |
| `DEMO_VARIANT=noturn` | pushed up and eased down without the turn: the pins come back down the legs | 0.0 (dropped) | offline check only |
| `DEMO_VARIANT=early` | let go at the top before the turn: the plungers push the bulb out | 0.0 (dropped) | offline check only |

Two arms are needed: a bulb pushed into the free holder lifts it (the plungers' 0.8 N preload is more than the
0.59 N hanging below the flex's slack) and turns it with the bulb, so the pins never go up the J against the plungers.

## Simplifications

- Rigid bodies. The flex is a joint chain (two swing hinges, a slack slide, a stretch spring, a twist spring), not a
  bending cable.
- The J-slot's shape is a declared rule that drives the hook floors; the plungers, the hook floors and their lips, the
  bore and every push and turn are contact.
- The holder is a 12-slab round wall; the bulb's globe is stacked cylinders.
- No wiring: the rose is a dummy and nothing is live.
- Sizes inside the holder, masses, spring rates and friction (holder 0.5, bulb glass 0.8) are estimates.

## Sources and measurements

- The bulb: Philips LED Classic A60 B22, frosted (A60: 60 mm across; B22d cap: 22 mm, two pins).
- The lampholder: a B22d plastic lampholder to BS EN 61184 (bayonet caps and holders).

Everything else is an estimate (est in task.json). How to measure each on a bought pendant:

| Estimate | Sim value | How to measure |
|---|---|---|
| Plunger force | 0.4 N each at rest, 0.45 N each at full travel | 0-2 N spring scale pushing a bulb up with the holder held; acceptance: 1.5 N or less to the stop |
| Plunger travel, hook height | 5 mm, 2 mm up it | calipers on the cap's end against the mouth, at the stop and locked |
| Turn to the hooks | 15 degrees | paint marks on the cap and the mouth, a protractor |
| Holder mass, size | 30 g, 38 mm x 52 mm | kitchen scale, calipers; acceptance: under 50 g |
| Flex length | 240 mm | ruler; acceptance: 220 to 260 mm |
| Flex stretch, twist | 600 N/m; 0.0005 N m per degree | spring scale on the holder and a ruler; a 50 mm lever arm on the holder and the spring scale |
| Bulb length, mass | 104 mm, 40 g | ruler, kitchen scale |
| Friction | 0.5 holder, 0.8 bulb glass | tilt a board with the part on it until it slides: friction is the tangent of that angle |

## Real build and reset

A rigid stand with a top arm 500 mm up (a post on a foot plate fixed to the table), a plain pendant set that passes
the acceptance checks (task.json setup, item 4), a new A60 B22 LED bulb, a 50 mm ring stand, paint for the marks and
the stripe, 0-2 N and 0-10 N spring scales, a 10 mm scale and a 15 mm mark on the stand. Reset: push the bulb up, turn it back
15 degrees and lower it out; stand it cap-up in its ring stand at x 250 mm, y -130 mm with its paint mark on the
stand's; let the holder hang still with its stripe to the front. Nothing is consumed.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the J rule's step hook, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `parts.py`, `scoring.py` | Every dimension; the parts as shapes; the scorer and the J rule |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and articulations; read both grippers each step |
| `tiltgrip.py` | Pitched grips for the demo's waypoints |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
