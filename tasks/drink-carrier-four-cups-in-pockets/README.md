# Load four filled, lidded cups into a drink carrier, each upright in its own pocket

A molded-pulp four-cup drink carrier (196 x 196 mm, 43 mm to the pocket mouths) stands loose on the counter, 22 cm
out between the arms. Beside it stand four filled 8 oz paper hot cups with press-on sip lids, two on each side, 34 cm
out from the midline. The robot must stand every cup upright on the floor of its own pocket, then take both hands off
and leave the carrier where it is. Each pocket is a round tapered funnel (76 mm across at the mouth, 62 mm at the
floor, 36 mm deep) that follows the cup's own taper, so a cup lowered a few millimetres off centre is guided in, and
one set down on the mouth is not seated.

## Why it is hard

Standing cups upright is covered ground; this task adds a flimsy loose carrier that must stay undisturbed while four full, lidded cups are seated in it, with a spill tilt limit and 2.5 mm of play, shared by two arms taking turns. The sim tests the tilt, drop, carrier-move and seating rules; it does not model water or lid failure.

1. **The cups are tapered, heavy and full.** Each tapers from a 56 mm base to an 80 mm rim, 95 mm tall with its lid
   and 0.25 kg filled (2.5 N). Parallel fingers can only hold it high on the cone, just under the 82 mm lid, and must
   carry it level: a lidded hot cup tipped more than 15 degrees spills.
2. **The pockets are tight.** On the floor a cup has 2.5 to 3.6 mm of play round its base; the mouth gives 10 mm of
   lead-in. Arms trail their commands by several millimetres under load, so the cup must be guided down the funnel,
   not dropped in.
3. **The carrier is loose.** It weighs 0.10 kg on a bare counter. A cup pushed against a pocket wall shoves it; more
   than 30 mm, or a 5 degree tilt, scores 0, and so does letting a cup fall more than 40 mm.
4. **Each hand reaches only its own side.** The four cups stand 34 cm out to the left and right; the pockets are 5 cm
   either side of the midline, where both grippers cannot be at once (their fingers must stay 20 cm apart), so the
   arms place by turns.
5. **Hands off.** The score is read with both grippers 30 mm clear and every cup and the carrier visibly still for 1 s (under 5 mm and 5 degrees): a
   cup still pressed or held earns nothing. The GPU demo also met the stricter 0.5 mm and 0.3 degree check it was run under.

## Scoring

| Condition | Rule |
|---|---|
| Seated | underside within 2 mm of the pocket's floor, axis within 2 degrees of the carrier's vertical, centre within 8 mm of the pocket's axis, and the only cup over that pocket's mouth |
| Spill (latched) | any cup tilted more than 15 degrees from the vertical at any step |
| Dropped (latched) | any cup falling faster than 0.886 m/s (a free fall of 40 mm or more) |
| Carrier (latched) | its centre more than 30 mm from its mark, or tilted more than 5 degrees |
| Fell (latched) | a cup or the carrier more than 50 mm below the table top |
| Lying | a cup tilted more than 45 degrees now |
| Hands off | last 1 s: finger zones 30 mm or more from every cup and the carrier; each moves under 5 mm and its axis turns under 5 degrees (visible motion) |

| Outcome | Score |
|---|---|
| Four pockets seated | 1.0 |
| Three, two, one | 0.75, 0.5, 0.25 |
| None seated; any latch; a cup lying; or not hands off | 0 |

Success: 1.0 held for 1 s. Doing nothing scores 0. A carrier turned about its vertical is fine: cups are judged in its
frame. Two cups over one mouth (one stacked on the other) earn nothing for that pocket.

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Seated height (2 mm) | before the trials, press one cup fully down by hand in each pocket and record the lid's height over a straightedge laid across the carrier's top (the model's cup reads 59 mm); a trial cup counts within 2 mm of its pocket's record. Real pulp pockets vary a little, so each pocket gets its own reference |
| Axis (2 degrees) | a digital angle gauge laid on the lid |
| Centre (8 mm) | a rule across the mouth: the gaps between the cup's wall and the mouth's edge at two opposite points differ by 16 mm or less |
| Only cup over the mouth | by eye from above |
| Spill (15 degrees) | a 15 degree protractor held beside the cup in a side video, at every frame |
| Drop (40 mm) | 30 fps video: a cup let go more than 40 mm above where it comes to rest (three frames of free fall) |
| Carrier (30 mm, 5 degrees) | a pencil outline of the carrier on the counter and a rule; a protractor for the tilt |
| Fell (50 mm) | by eye and a rule |
| Hands off | 30 mm spacer blocks against each hand; 30 fps video of the last second in which nothing visibly moves, a 5 mm rule held beside the carrier |
| Spilled or crushed (real trials only) | 0 if water shows on the counter, the carrier or a lid, or a lid is off or lifted: the sim's rigid cup cannot buckle, so its tilt rule stands in |

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 3,000-step limit, one trial each; rows marked not
run were not run on the GPU. The offline prediction column (`tools/offline/check_task.py`, with this task's offline
stand-in for the physics, `task.offline_seat`) only says the plan's geometry is right; it is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 1200) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | seats the two left cups | 0.5 | 0.5 |
| `DEMO_IDLE_ARM=left` (right alone) | seats the two right cups | 0.5 | 0.5 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

One arm alone reaches only its own two cups and pockets, so it cannot score above 0.5: the task needs both arms.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The arms place by turns
(left, right, left, right); each carries on to its next cup while the other places, and waits with its cup in hand
until the carrier is free.

| Step per cup | Time (s) | What |
|---|---|---|
| Over the cup, down | 4.5 to 5.0 | hover 17 cm, down to 3 cm over the grip height, then slowly down beside the cup |
| Grip | 2.5 | settle, close the fingers to 70.9 mm commanded across the cup's 78.9 mm (the 38 mm pads span 47 to 85 mm up its cone, under the lid), settle |
| Lift, carry | 5.5 | straight up to 66 mm commanded underside height (23 mm over the mouths), then 29 to 30 cm to the pocket |
| Down the funnel | 5.0 | settle, down to 16 mm over the mouth, then 3 s slowly to 4 mm below the floor height (it presses about 1 N) |
| Let go | 5.5 | settle, open over 1.5 s, straight up 1.5 s, aside to y 24 cm over 2 s |

The demo takes 59 s (1180 steps; the policy stops at step 1241); the last cup is released at 55.5 s. Offline: worst
IK 0.10 mm; joint 4 margin 8 degrees at the carry height; TCPs at least 29 cm apart at the closest; arm capsules
10.3 cm apart. The run limit is 3,000 steps (150 s): the demo needs 1,200 steps to hold success.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Stand all four cups upright in the four pockets of the carrier, then take both hands off and leave the carrier where it is." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/drink-carrier/demo
```

For Dream-1 use `--policy dream`. Run one trial per process. Runtime: a trial takes about 2 to 4 minutes of wall-clock
on the L40S including start-up (the demo itself is about 62 s of simulated time). Seed: nothing is random;
`object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed is needed.

Environment switches: `DEMO_IDLE_ARM=left|right` leaves that arm at its ready pose (the one-arm check);
`DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control step
(each cup's height over its pocket floor, tilt and offset, the carrier's move and tilt, the grippers' gap, the latches).

## Simplifications

- Rigid bodies. The cup is one rigid compound: its cone as 11 stacked discs 8 mm tall (1.1 mm steps), each as wide
  as the cone at its mid-height, with the lid's 82 mm skirt and 72 mm dome fixed on top. Paper does not buckle and the
  water does not slosh: the 15 degree tilt rule stands in for a spill, and a real trial also checks for water and
  lifted lids.
- Each pocket is a 12-sided funnel of flat 4 mm slabs (real pulp is about 2 mm); the 4 mm is the simulator's contact
  margin. The pockets' 11 degree taper (the cup's is 7.8 degrees) stands in for the pulp's cone.
- Friction: carrier on the table 1.0 / 0.9 (est), cups the simulator's default.
- Offline only: the checker's proxy drops a released cup onto the carrier's bounding box; `task.offline_seat` puts a
  cup released over a pocket on its floor, as the funnel would. It never runs in the simulator.

## Sources and estimates

- Fast food and counter workers, SOC 35-3023, about 3.9 million US jobs (BLS Occupational Outlook Handbook, as listed
  in the program's occupation table).
- Estimates, said in `fidelity`: the 8 oz hot cup (rim 80 mm, base 56 mm, 88 mm body, 95 mm with its lid, 237 g of
  water), the
  carrier's pitch (104 mm), pocket sizes, depth and mass, and all frictions. The real trial's fit check (calipers, a 4 to 8 mm
  floor clearance, see Real build and reset) keeps the cups and carrier inside the modeled fit; the scoring then depends on
  these figures only through the seated reference described above.

## Real build and reset

Products: standard 8 oz paper hot cups with press-on sip lids (typically 80 mm rim, 56 mm base, 88 to 92 mm body) and a four-cup molded-pulp takeaway carrier (typically 60 to 64 mm across a pocket floor, 74 to 78 mm across the mouth, 34 to 38 mm deep, about 195 mm square).

Fit check, before any trial. The model's close fit is 6 mm across (a 62 mm pocket floor round a 56 mm cup base; its lowest disc is 57.1 mm, so 2.5 mm a side). Measure with calipers: each cup's base outside diameter (use cups of 55 to 57 mm), each pocket's inside diameter at the floor, at the mouth and its depth. Use a cup and a pocket together only if the floor diameter exceeds the cup base by 4 to 8 mm (6 mm in the model); swap cups or the carrier until all four pairs are in that range, and reject a carrier whose mouths are under 74 or over 78 mm or whose depth is outside 34 to 38 mm. Then the seated reference (Scoring section) sets each pocket's own floor height. Water: room temperature only, filled to 10 mm under the rim; wipe up any spill at once, replace a wet carrier, and cover each gripper pad with a wipeable film (wipe it dry between trials) so no liquid reaches the arm. A molded-pulp four-cup carrier and four 8 oz paper hot cups with press-on sip lids; room-temperature water (never hot, so a spill cannot scald anyone) to 10 mm under the rim; a
straightedge, a rule, a digital angle gauge, a protractor and 30 mm spacer blocks. Reset: fill fresh cups, press the
lids on all round, stand them at x 190 and 310 mm, y +340 and -340 mm, and wipe the counter dry. Replace a cup that is
dented or leaks. Set a dry carrier centred at x 220 mm, y 0, on the bare
counter (no mat), pockets square to x and y, and pencil its outline. Replace a carrier whose pockets have softened or
whose mouths measure outside 74 to 78 mm.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; the carrier and cup compounds, `success()`, `score()`, `demo()`, the offline stand-in |
| `layout.py`, `scoring.py` | Every dimension and threshold, the funnel's slabs; the scorer |
| `mechkit.py`, `rigstate.py` | Build the carrier and the cups as compound rigid bodies; read both grippers each step (unchanged copies) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
