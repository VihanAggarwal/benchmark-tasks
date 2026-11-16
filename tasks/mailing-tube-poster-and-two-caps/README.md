# Slide a poster roll into a mailing tube and press an end cap onto each end

A kraft mailing tube (51 mm bore, 54 mm outside, 457 mm = 18 in long, open at both ends) lies along y on a packing
board. A paper roll standing for a rolled poster (38 mm across, 380 mm long) lies on the table in line with the
tube's left end. Two black plastic end caps lie on their sides in two cap rests, one in front of each arm, plug
toward the tube. The robot slides the roll in end first and presses a cap onto each end of the tube, then lets go.
The instruction does not say how far in the roll goes or which hand does what.

## Why it is hard

1. **Alignment to a bore with 6.5 mm of play.** The roll has to enter the open end with its axis within a few
   millimetres of the tube's axis, lifted off the table (the bore's floor is 1.5 mm up), and slid 340 mm along y
   without tilting. The tube is loose on its board.
2. **A cap is a plug, not a lid.** Each cap's 18 mm plug has to go straight into the bore (0.5 mm of play at the
   flats, a 3 mm lead-in nose) and the flange has to sit on the tube's end. A cap carried 3 mm high or turned 7
   degrees about the vertical does not seat. The caps lie on their sides and are held across the flange, so the
   hand must lift each one clear of its rails, carry it in line, and press along the axis.
3. **The roll must go far enough.** The plug is 18 mm deep, so a cap pushed against a roll that is nearer than 18
   mm pushes the roll along. The roll's near end is 16.5 mm outside the mouth and the hand's pads cannot follow it
   into the bore (the finger bodies would hit the tube's wall), so the left cap pushes the last 40 mm home.
4. **Two ends, two hands.** The right end of the tube (y -229 mm) is beyond the left arm's reach at this x (it
   reaches y -100 mm at most), and the left end is beyond the right arm's. One hand per end. The tube is free on
   its board, so the two caps are pressed together and the tube is squeezed between them.
5. **Hands off.** The score is read with both grippers 30 mm clear and every part still for 1 s.

## Scoring

| Condition | Rule |
|---|---|
| Roll in | both ends at least 15 mm inside the tube's two ends (along its axis), each within 8 mm of its axis |
| Cap seated at an end | cap axis within 5 degrees of the tube's, pointing out of that end; flange inner face within 1.5 mm outside the end (2 mm in at most); flange centre within 2.5 mm of the tube's axis; plug tip within 3 mm of the axis |
| Ends capped | 0, 1 or 2 (each end counts once) |
| Hands off | last 1 s: finger zones 30 mm or more from all four parts; each moves under 1 mm, turns under 2 deg |
| Lost (latched) | a part's centre more than 30 mm below the table top or off the table |

| Outcome | Score |
|---|---|
| Roll in, both ends capped | 1.0 |
| Roll in, one end capped | 0.5 |
| Roll in, no end capped | 0.25 |
| Roll not in, a part lost, or not hands off | 0 |

Success: 1.0 held for 1 s. Doing nothing scores 0. Caps sealing an empty tube score 0 (the roll must be in).

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Roll in (15 mm) | before the first cap: a depth rod and a rule from each open end to the roll's end. After a cap is seated: its 18 mm plug cannot pass a roll nearer than 18 mm, so a seated cap proves its end |
| Roll on the axis (8 mm) | it lies in the bore, which leaves 6.5 mm: a roll inside is within the limit by construction; a roll lying beside or on top of the tube fails |
| Cap seated (1.5 mm) | 1.5 mm feeler between the flange's inner face and the tube's end, at four points |
| Cap axis (5 deg) | 5 degree angle gauge along the tube against the cap's face |
| Flange centre (2.5 mm) | the flange's overhang past the tube's outside is 3 mm nominal: within 0.5 to 5.5 mm at four points |
| Plug in the bore (3 mm) | the cap does not rock under a light hand, or a 3 mm tip gauge |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s; a 1 mm mark on the board for the tube |
| Lost | by eye |
| Retention (real trial only) | after the run, lift the tube by its middle 10 cm and turn it end-down, once per end: a cap that stays seated counts as capped, a cap that falls out counts as not capped |
| Poster damage (real trial only) | unroll the poster: no crease or tear from the packing, else the score is capped at 0.5 |

The sim reads neither of the last two rows: it scores poses only, and its roll cannot crease.

## Results

GPU runs on an L40S with the commands below fill the GPU column (one trial each). The offline prediction column
(`tools/offline/check_task.py`, with this task's offline stand-in for the physics, `task.OfflineModel`) is not a
measurement. Rows marked not run have no GPU number.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run |
| Scripted demo | below | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | picks and presses cap b only: no roll | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=right` (left alone) | roll and cap a: the right end is out of its reach | 0.5 | 0.5 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | both caps stop 10 mm short of the end: the roll is left 10 mm in | 0.0 | not run |
| `DEMO_VARIANT=right_short` | the right cap stops 10 mm short | 0.5 | not run |

With the right arm idle the left arm alone reaches only the left end (0.5); with the left arm idle nothing is
scored. Both hands are needed for 1.0.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both arms close along x.

| Time (s) | Left gripper | Right gripper |
|---|---|---|
| 0 to 8.5 | down round the roll's middle (y 435 mm, hand at x 282 mm where the roll has rolled to), grip, lift 13 mm to the tube's axis | wait over cap b |
| 8.5 to 16.2 | slide the roll 188 mm along y into the tube at 40 mm/s (its near end 171 mm in), let go, up | wait over cap b |
| 16.2 to 30.8 | over the roll's far end (25 mm in from it), grip, lift 7 mm, push it along the bore floor at 40 mm/s until its far end is 43 mm outside the mouth, let go, up | wait over cap b |
| 30.8 to 42.3 | over cap a, grip it, lift, carry in line, lower | the same for cap b, lowered in line with the tube's right end |
| 42.3 to 51.6 | press cap a on at 25 mm/s (it pushes the roll's end home), let go, up, aside | press cap b on at the same time, let go, up, aside |

The roll is held by its middle for the first slide because a pinch holds no torque about its closing axis: held near
its end it swings about the pads and its near end hangs on the table. It is held by its end only for the second push,
where the bore floor carries it. The roll also rolls sideways on the table at about 4 mm/s before the hand arrives, so
the hand comes down at x 282 mm and the closing pads centre the roll on it.
Both presses command 2 mm past the seat (the arm trails its command), the tube squeezed between them.
Offline: worst IK 0.1 mm; the left hand out at y 0.41 to 0.44 m (0.11 to 0.13 m outboard of its base) needs the yaw
turned half a turn (yaw +90 degrees there); TCPs at least 45 cm apart.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: the demo itself is 1092 steps (55 s of simulated time); a 1200-step trial is expected to take 2 to 3 minutes
of wall-clock on the L40S including start-up. Seed: nothing is random; `object_jitter=0` fixes
the start layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: each part's pose, the
roll's end positions along and off the tube's axis, each cap's distance from its end and from the axis, both TCPs and
openings, and a `hist` of the last 3 s of those numbers in every row.

## Simplifications

- Rigid bodies. The tube is a ring of 16 flat 1.5 mm staves (a round tube is not a native shape, and real kraft board
  dents); the roll is a solid rigid cylinder standing for a poster's paper (it does not bend or telescope).
- The sim plug (49 mm) has clearance in the 51 mm bore (0.5 mm of play at the flats), so retention is not simulated: a
  seated cap stays on by its plug resting in the bore, not by grip. A real stock kraft-tube end cap is a black plastic
  (LDPE) plug with a ribbed, slightly flexible plug and a snug friction fit, needing about 3 N (est) to pull out; the
  real-trial retention check (lift by the middle 10 cm, turn end-down) covers that. Poster damage is not simulated. Caps are 12 g (real about 8 g) for stable
  contact. The plug has a 3 mm lead-in nose (two steps, not a cone).
- No declared software rule: every part is a plain rigid body. The scorer reads poses; the arms are read only for
  the let-go check.
- Friction: tube 0.40 / 0.35 (kraft on board), caps 0.5 / 0.45, board 0.5 / 0.45, roll 0.40 / 0.35 (est). Masses
  are estimates.
- The cap rests (a 5 mm platform under the plug and two 12 mm rails) are the setup's way of holding a cap level on
  its side; a real cap lies on any flat.

## Sources

- Packers and packagers, hand, SOC 53-7064, about 645,000 US jobs in May 2023: BLS OEWS,
  https://www.bls.gov/oes/2023/May/oes537064.htm
- Mailing tubes 2 in diameter, 18 in long, with plastic plug caps: stock sizes in office and shipping supply
  catalogues; kraft board wall 1.5 mm (est).

Estimates (stated in fidelity): the tube's, roll's and caps' masses and frictions, the cap's flange and plug
dimensions, the cap rests, the board.

## Real build and reset

An 18 in x 2 in kraft mailing tube; a pair of 2 in plug caps; a 15 in poster rolled to 38 mm; a 100 x 400 mm board 5
mm thick; a 1.5 mm feeler, a depth rod and a rule. Reset: the tube on the board, the roll in line with its left end
16.5 mm from the mouth, each cap on its rest, plug toward the tube. Use a fresh cap pair every five trials.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
