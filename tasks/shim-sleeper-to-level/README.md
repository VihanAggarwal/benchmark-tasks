# Level a floor sleeper: push a tapered shim under its low end until the bubble centres

A 2x4 floor sleeper (38 x 89 x 610 mm) stands on edge across two fixed 4x4 pads. The floor dips under the near pad,
so its top is 6.0 mm lower than the far pad's and the sleeper slopes 9.9 mm end to end. A 9 in torpedo level lies
on the sleeper's top. On the low pad lies a standard wood shim (8 in long, 1-1/2 in wide, 3/8 in tapering to 1/16
in), thin tip at the sleeper's front face. The robot must leave the sleeper level, the shim under it and the level
on it, with its hands off. The instruction names neither the lift nor how far to push.

## Why it is hard

1. **The shim cannot start under a loaded end.** Where it lies the sleeper's underside is 0.4 to 1.0 mm off the pad
   and the tip is 1.6 mm thick. One hand must lift the low end (about 4.7 N near its end; it pivots on the far pad)
   while the other pushes the shim in by its butt.
2. **The taper sets the height.** The shim's thickness rises 0.039 mm per mm of travel. The sleeper lies flush on
   the sloped shim (it turns about 2.2 degrees across its 38 mm width), so its height is the shim's thickness at its
   rear face: it is level with the tip 151 mm past the front face (6.0 mm thick at the rear face). The window for 1.0
   (within 1 mm end to end) is 137 to 167 mm of travel; for 0.5 (within 2 mm) 122 to 184 mm. Pushed fully home the
   sleeper ends 3.1 mm high; left short it stays low. A shim under only the front edge bears on a corner.
3. **The reading is the level's.** The bubble (a declared rule, below) moves 1.5 mm per milliradian of the level's
   own tilt and sits between its two lines only when the sleeper is within 1 mm over its 610 mm.
4. **Hands off.** The score is read with both grippers 30 mm clear and every part still for 1 s: a held sleeper
   earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Difference | the sleeper's end-to-end height difference: 610 mm x the sine of its long axis's slope |
| On the pads | centre line over each pad's centre line; spans each pad's full 89 mm width; low end <= 5 mm down |
| Shim under | on the low pad (5 mm, tilt under 15 deg), centre within the sleeper's length, crossing under it |
| Full bearing | tip past the rear face, butt short of the front face: across the whole 38 mm |
| Level on | underside centre over the sleeper's top (2 mm allowed), at most 3 mm above it, axis within 10 degrees |
| Hands off | last 1 s: finger zones 30 mm or more from all three parts; each moves under 1 mm, turns under 0.3 deg |

| Outcome | Score |
|---|---|
| On the pads, full bearing, level on, within 1 mm | 1.0 |
| The same within 2 mm | 0.5 |
| On the pads, shim under, otherwise (too far, short, front edge only, level off) | 0.25 |
| Anything else, or not hands off | 0 |
| Sleeper or level 20 mm below the low pad's top, or the sleeper 30 degrees off its edge (latched) | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Difference (1 mm / 2 mm) | steel rule and feeler gauge at each end of the sleeper's top, or the bubble |
| On the pads | rule, pad top to the sleeper's low end (5 mm); from above: centre lines, both ends overhang both pads |
| Shim under, full bearing | rule for the tip past the front and rear faces; 15 degree angle gauge for the shim's tilt |
| Level on | 3 mm feeler under the level; protractor for 10 degrees |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s; dial indicator on the sleeper's top |
| Fall (latched) | sleeper or level 20 mm below the low pad's top, or sleeper turned 30 degrees: by eye and a rule |

Sliding the sleeper along its pads is not penalised. Success: 1.0 held for 1 s. Doing nothing scores 0. On a real
bench: the height of each end of the sleeper's top with a steel rule, or the bubble between its lines.

## Results

Run on an L40S (Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 900-step limit), one trial each. The
offline prediction column (`tools/offline/check_task.py`, with this task's offline stand-in for the physics,
`task.OfflineModel`) is not a measurement. Rows marked not run have no GPU number.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 518) | 1.0 (held from step 508; ended at step 519) |
| `DEMO_IDLE_ARM=right` (left alone) | lifts, lowers: no shim | 0.0 | 0.0 (the shim never moved) |
| `DEMO_IDLE_ARM=left` (right alone) | pushes, no lift | 0.0 (tip jams) | 0.0 (sleeper dragged off pads, step 279) |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 (shim never moved; sleeper 9.9 mm out of level) |
| `DEMO_VARIANT=short` | the shim pushed 95 mm: 3.9 mm low | 0.25 | not run |
| `DEMO_VARIANT=overshoot` | 200 mm: 3.0 mm high | 0.25 | not run |
| `DEMO_VARIANT=home` | fully home: 3.1 mm high | 0.25 | not run |

With one arm idle the demo scores 0.0 either way, so both hands are needed here: one lifts the loaded end, the other
pushes the shim.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The push length is
computed from the shim's taper, not read from the bubble.

| Time (s) | Left gripper (fingers across the sleeper) | Right gripper (closed into a block) |
|---|---|---|
| 0 to 6.5 | down round the low end at y 155 mm, grip it | in front of the shim's butt, close, down to the pad |
| 6.5 to 8.5 | lift the end 30 mm (commanded; it pivots on the far pad) | wait |
| 9.0 to 15.6 | hold it up | push the butt 164 mm at 25 mm/s: the tip ends 151 mm past the front face |
| 16.1 to 19.6 | hold it up, then lower 5 mm past the level height | back off 25 mm, up, aside |
| 20.1 to 24.6 | settle, let go, up, aside | aside |

Four deliberate moves: lift, push, lower, let go. Offline: worst IK 0.09 mm; left joint 4 margin 4.9 degrees at the
195 mm hover (8 degrees while it holds the load); the arms' capsules stay 1.3 cm apart (the left gripper against the
right forearm, as the push ends); TCPs at least 19.6 cm apart. The lift works for any arm give up to about
18 mm (at 12 mm of real lift the underside still clears the front edge of the level shim, 7.5 mm thick, by 1 mm).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 900
```

Runtime: a 900-step trial takes about 1.5 to 3 minutes of wall-clock on the L40S including start-up (the demo itself is
519 steps, 26 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so
every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the difference, the
shim's tip past the front face, its yaw and lowest point, the level's pitch and bubble, the sleeper's low end, both
TCPs and openings, and a `hist` of the last 3 s of those numbers in every row.

## Simplifications

- Rigid bodies. The sleeper is a square box (a real 2x4 has 3 mm eased edges). The shim's taper is one turned slab
  1.4 mm thick over eight flat blocks, so nothing steps up through the sloped top.
- The floor's 6 mm dip is the low pad's block made 6.0 mm shorter.
- One declared rule: the bubble is a small link on a slider along the vial, driven every control step to 1.5 m times
  the level's own measured pitch (toward the high end), stopped at the vial's ends. 1.5 m is a sensitive vial (est);
  it makes 1 mm over 610 mm move the bubble 2.5 mm. The scorer reads the sleeper's pose, not the bubble.
- Friction 0.45 / 0.40 wood on wood and 0.40 / 0.35 the level on wood (est). Masses from density and size.

## Sources

- Carpenters, SOC 47-2031, 889,700 jobs in 2025: BLS Occupational Outlook Handbook,
  https://www.bls.gov/ooh/construction-and-extraction/carpenters.htm
- Shim: 8 in long, 1-1/2 in wide, 3/8 in tapered to 1/16 in (wood shim bundle listings),
  https://www.thegranitegroup.com/Product/DIBPSH8123675 and https://www.truevalue.com/product/wood-shims-8-in-12-pk/
- Torpedo level: Stanley 42-264, 9 in, 1 in thick, 0.2 lb (retailer listing),
  https://eshop.intercityindustrial.com/9-magnetic-torpedo-level--4 and
  https://www.stanleytools.ca/product/42-264/9-225mm-jobmaster-magnetic-torpedo-level
- Dressed lumber sizes 2x4 = 1-1/2 x 3-1/2 in, 4x4 = 3-1/2 x 3-1/2 in: American Softwood Lumber Standard PS 20.
  Spruce-pine-fir specific gravity 0.42: AWC NDS Supplement, https://awc.org/publications/2018-nds-supplement/
- Shimming practice (lift, slide in, check the level):
  https://www.mrhandyman.com/blog/what-is-a-shim-and-how-do-you-use-it-/

Estimates (stated in fidelity): the level's height (45 mm), the vial's sensitivity, frictions, the shim's density,
the pads' 300 mm length.

## Real build and reset

Two 4x4 offcuts 300 mm long, one on a floor (or a plate) 6 mm lower; a 610 mm 2x4; a 9 in torpedo level; a pack
of 8 in wood shims; a steel rule. Reset: the sleeper on edge across the pads from y +170 to -440 mm, the level along
its top, the shim on the low pad along x, tip 2 mm short of the sleeper's front face. Swap a shim that has been
pushed several times.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, bubble rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
