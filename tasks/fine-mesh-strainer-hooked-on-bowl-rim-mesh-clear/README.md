# Hang a hooked strainer across a bowl: hook on the right rim, handle on the left

A person has strained a first ladle of stock into a mixing bowl and is about to pour the next one through the same
fine-mesh strainer. The strainer lies on the counter beside the bowl. They pick it up by the handle, hangs its
hook over one rim of the bowl and rests its handle on the opposite rim, so that the mesh hangs inside the bowl clear
of the stock, and lets go. Here a stainless mixing bowl (220 mm across its rolled rim, 105 mm deep, 0.30 kg) stands
loose on a silicone non-slip mat, holding 0.5 L of strained stock (30 mm deep). Beside it are the stock pot and a tiled
backsplash wall. The strainer (130 mm ring, 57 mm tall, folded 130 mm handle, a hook with a 30 mm gap, 314 mm long)
lies mesh down 25 cm to the bowl's left, handle toward the left arm. The robot must hang it across the bowl, the hook
over the right rim, the handle on the left rim, the mesh 26 mm above the stock, and take its hand away.

## Why it is hard

1. **The hook has to straddle a rolled bead.** The bead (radius 102 to 110 mm) must end up in the hook's 30 mm gap:
   the inner leg inside the wall, the outer leg outside the bead. The strainer has to be lowered so the hook's feet
   (outer 20 mm, inner 12 mm) go down either side of it, on the right rim (the side away from the handle's hand). The window is -6.5 to +10 mm along the strainer's axis (the
   inner leg's foot meets the flared wall at -6.5 mm; the outer leg meets the bead at +10 mm). Set down 20 mm short or
   25 mm past it, a leg's foot lands on top of the bead and the hook end stays 12 or 20 mm up (0.5).
2. **The mesh must not sit in the stock.** The ring is 47 mm deep below the rim plane and the stock is 32 mm up the
   bowl, so there is only 26 mm of margin when it hangs correctly. A strainer let down into the bowl lies in the stock
   and scores 0: that is the failure the cook is avoiding. Carrying it over the left rim needs the mesh above the
   bead's top, so the hand has to lift it 12 cm and then bring it down across the bowl without catching the ring on
   the far rim or the hook on the near one.
3. **The bowl is loose.** It is a 0.82 kg bowl on a mat, not fixed: it may be pushed or knocked on the way, but it must end upright
   (10 degrees or less), on the mat and within 30 mm of its mark, or the score is 0; so the hook cannot be forced on by
   dragging the bowl 4 cm or more under the strainer, while a small safe repositioning costs nothing.
4. **A pinch grip on a thin part.** The handle is 18 mm wide with 10 mm flanges. Held next to the ring, the strainer's
   centre of mass is still 71 mm from the pads, so the grip has to hold a small twisting load (about 0.08 N m) while
   the hand carries it across; held at the handle's end it would be twice that.
5. **Hands off.** The score is read with both grippers 30 mm clear and both parts still for 1 s: a strainer held
   in place earns nothing.

One arm is enough. The left arm picks the strainer up, carries and hangs it, and moves away; the right arm stays at
its ready pose. The task does not claim to need two hands (in a kitchen the other hand only steadies the bowl, and
here the bowl sits on a non-slip mat).

## Scoring

All distances are taken in the bowl's own frame (its axis through the centre of its floor; the bead's centre line at
105 mm radius, its top 105 mm above the bowl's foot; the stock's top 32 mm above it).

| Condition | Rule |
|---|---|
| Hook on | the hook's inner leg face is 100 mm or less from the bowl's axis and its outer leg face 110 mm or more (top view), so the bead lies in the gap; the strap's underside within 6 mm above the bead's top and no more than 2 mm below it; on the right rim: the hook's middle within 45 degrees of straight right (-y) of the bowl's axis (top view) |
| Handle on | where the handle's underside crosses the far bead (within 5 mm of its centre line), within 6 mm above the bead's top and no more than 2 mm below it |
| Ring in | the ring's centre within 30 mm of the bowl's axis (top view): the ring wholly inside the rim's inner edge with 5 mm to spare |
| Mesh clear | the mesh's underside 15 mm or more above the stock's top (47 mm or more above the bowl's foot) and at least 20 mm below the rim's top (85 mm or less) |
| Level | long axis within 10 degrees of horizontal, strainer within 15 degrees of upright, long axis passing within 20 mm of the bowl's axis (any direction across the round bowl counts) |
| Hands off | last 1 s: both finger zones 30 mm or more from the strainer and the bowl; the strainer and the bowl each move under 5 mm and turn under 5 degrees (no visible motion) |
| Bowl in place (final state only) | at the last step the bowl's foot within 30 mm of its mark (top view), not lifted 20 mm, tipped 10 degrees or less; pushing it on the way and putting it right is not penalised |
| Latched zero | either part's lowest point ever more than 20 mm below the table top |

| Outcome | Score |
|---|---|
| Hook on, handle on, ring in, mesh clear, level, hands off | 1.0 |
| Ring in and mesh clear, long axis within 25 degrees of horizontal (across the rims, not hooked, hooked off line or on another part of the rim) | 0.5 |
| Anything else (on the table, on the pot, the mesh down in the stock, held, bowl dislodged) | 0 |

Possible scores: 0, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0. A mesh within 15 mm of the stock earns
nothing, even with the ring in the bowl: it is the outcome the task exists to prevent (the mesh clogs, the next pour
backs up).

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Hook on (100 mm / 110 mm, 6 mm / 2 mm, 45 degrees) | two radius templates (100 and 110 mm from the bowl's centre) laid on the rim; a 6 mm feeler that must not pass between the strap and the bead's top (the strap cannot sit below the bead); a 45 degree set square from the bowl's centre, its edge along the table's y axis, for the right rim |
| Handle on (5 mm, 6 mm) | a rule across the bowl to find the bead under the handle; a 6 mm feeler that must not pass under the handle |
| Ring in (30 mm) | a 30 mm radius circle drawn about the bowl's axis on card, held over the ring's centre |
| Mesh clear (47 mm / 85 mm) | a rule inside the bowl against marks 32 mm (stock), 47 and 85 mm above the foot |
| Level (10, 15 degrees, 20 mm) | a 10 and a 15 degree protractor on the handle; a rule across the bowl for the 20 mm |
| Hands off (30 mm, 5 mm, 5 degrees) | 30 mm spacer against each hand; 30 fps video of the last 1 s against a taped mark on the table at the strainer's handle and the bowl's foot: no visible motion |
| Bowl in place (30 mm, 10 degrees, 20 mm lift) | a taped outline 30 mm outside the bowl's foot, checked at the end; a 10 degree protractor; a rule |
| Fall (20 mm) | a rule against the table |

Unintended strategies earn nothing: dragging the bowl more than 30 mm under the strainer leaves it out of place (0); dropping the strainer
into the bowl lies in the stock (0); laying it across the rims without the hook is 0.5; holding it in place
fails hands off; the pot's rim cannot satisfy ring in.

Stillness rule (relaxed): the earlier limit (1 mm, 0.3 degrees over 1 s) could not be checked on video or one dial
indicator, so it is now the standard visible-motion rule: in sim the strainer and the bowl each move under 5 mm and
turn under 5 degrees over the last 1 s; in a real trial there is no visible motion in 30 fps video against a taped
mark. The GPU demo (1.0, 2026-11-17) already passed the stricter limit, so relaxing it cannot change that score.

Dry run: a real trial uses an empty, dry bowl and no water, so there is no spill risk; a tape mark 32 mm above the
bowl's foot stands for the simulated stock's top.

## Results

Offline predictions (`tools/offline/check_task.py`, with this task's offline stand-in for the physics,
`task.OfflineModel`) are not measurements. The GPU column is the physics simulator (Isaac Lab, 20 Hz control) on the
current demo. Rows marked not run have no GPU number.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run) |
| Scripted demo | below | 1.0 (held from step 476) | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | nothing moves | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the same demo, the right arm idle | 1.0 | 1.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=inner_on_rim` | set down 20 mm short: inner leg on the bead | 0.5 | not run |
| `DEMO_VARIANT=outer_on_rim` | set down 25 mm past: outer leg on the bead | 0.5 | not run |
| `DEMO_VARIANT=off_centre` | set down 25 mm to the side: axis misses by 25 mm | 0.5 | not run |

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Only the left arm moves.

| Time (s) | Left gripper (fingers close along x, across the handle) |
|---|---|
| 0 to 2 | over the handle next to the ring (y 334 mm, 14 cm up) |
| 2 to 5 | down round the handle: pad tips level with the flanges' bottoms |
| 5 to 7 | settle, close on the handle (commanded 2 mm across its 18 mm: about 40 N a pad), settle |
| 7 to 10 | lift 12 cm straight up: the mesh 14 mm over the bead's top, the hook's feet 41 mm over it |
| 10 to 13.75 | carry it 22 cm to the right until the ring's edge is over the far (left) rim's inner edge, then settle |
| 13.75 to 15.25 | down and across: the ring inside the far rim, the hook's feet about 18 mm over the near bead; then across until the hook's outer leg is past the near bead |
| 15.25 to 18.5 | down to 12 mm over the bead, then slowly to half a millimetre under the rest height: the hook's legs straddle the bead, the flanges land on the far one; settle |
| 18.5 to 20 | open (1.5 s) |
| 20 to 22.5 | straight up 5 cm, then aside to the left, clear of the strainer and bowl |

Offline: worst IK 0.08 mm; left joint 4 margin 6.0 degrees (at carry height, over the far rim), joint 6 margin 15.1
degrees; the right arm holds its ready pose, 22.3 cm clear of the left arm's links. Along the path the ring and the
hook stay 5 mm or more clear of the bowl's wall and bead (straight-line check of the commanded path). The hand comes
down across the bowl instead of at full carry height over its middle, where the wrist's joint 4 has little room. The demo is 22.5 s (450 steps), the
run limit 1,200 steps (60 s). The ring is aimed 2 mm toward the handle from the bowl's axis, the middle of the
hooking window (-6.5 to +10 mm), so the arm's trail of a few mm under the strainer's 0.12 kg either way still hooks.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Hang the fine-mesh strainer across the bowl: its hook over the right rim and its handle on the left rim, the mesh hanging in the bowl clear of the stock. Then let go." \
  --scorer task_score --max-steps 1200
```

Runtime: a 1,200-step trial takes about 2 to 4 minutes of wall-clock on the L40S including start-up (the demo itself is
about 450 steps, 22 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every
run starts from the same poses and no seed is needed. For Dream-1 use `--policy dream`. `DEMO_HOLD_ONLY=1` plays no
plan (a settling check); `DEMO_IDLE_ARM=left|right` drops that arm's plan (the one-arm check);
`DEMO_VARIANT=<name>` plays a counterfactual (never the submitted demo). `TASK_TRACE=trace.jsonl` writes one line per
control step: the hook's leg distances and height, the handle's gap over the far bead, the ring's offset, the mesh's
height over the foot, the long axis's slope, the bowl's displacement and tilt, both TCPs and openings, and the
scorer's decision.

## Simplifications

- Rigid bodies. The mesh is a 2 mm solid floor (a real 0.5 mm screen); the ring, the hook and the folded handle are
  boxes; the bowl's flared wall is 24 flat slabs, its rolled bead 24 boxes; the stock is a solid disc (no sloshing,
  no clogging).
- The bowl's wall is a straight flare from 140 mm inside at the floor to 200 mm at the rim (a real bowl curves).
- Friction 0.40 / 0.35 steel on steel and 0.9 / 0.8 silicone mat (est). Masses: bowl 0.30 kg (est) plus 0.52 kg of
  stock; strainer 0.12 kg (est).
- The pot and the backsplash wall are fixed bodies that stand for the kitchen around the bowl; neither is scored.
- No software rules: nothing is welded, bonded or driven. The strainer hangs on the bowl by its own geometry.

## Sources

- Category: Societal relatability. Resting a hooked strainer across a bowl is an everyday home-kitchen skill (straining stock, pasta water, custard); no occupation-count claim is made.
- Strainer: a widely sold stainless 5 in (13 cm) fine-mesh kitchen strainer with a bent-over bowl-rest hook opposite
  the handle (130 mm ring, handle ending 192.5 mm from the ring's centre, hook legs 30 mm apart). Acceptance check on a
  real one, with a ruler and calipers; only a strainer within every range is used: ring 125-135 mm across and 52-62 mm
  tall; handle 16-20 mm wide, ring centre to its end 187-198 mm; hook legs 28-32 mm apart, outer leg 17-23 mm and
  inner leg 10-14 mm long; overall length 305-325 mm; mass 100-200 g. Not a miniature (under 8 cm) strainer. An inexpensive
  kitchenware-store item (typically under 15 USD).
- Bowl: a 3 qt (about 220 mm) stainless mixing bowl; its sizes and all masses are estimates, to be measured.
- Layout and reach limits: our offline kinematics (`tools/offline`), which agrees with yam_arena's reach figures.

Estimates (stated in fidelity): all strainer and bowl dimensions and masses, the stock depth (30 mm, 0.5 L), frictions,
the mat's size, the pot and wall sizes.

## Real build and reset

A 3 qt stainless mixing bowl, a silicone non-slip mat, a 5 in hook strainer (or a flat strap hook and folded handle
made to the sizes in task.json), no water: the bowl stays empty and dry, with a tape mark 32 mm above its foot for the simulated stock's top (so a real trial has no spill risk), a stock pot, a tile or board
backsplash 56.5 cm from the arm line. Reset: the mat and bowl centred at x 300 mm, y 0 empty and dry; the
strainer mesh down, ring centred at x 300 mm, y +252 mm, handle toward +y along y; pot and wall in place; dry the
strainer and the rim. Nothing is consumed.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and threshold; the scorer (plain functions of the parts' states, a small tracker) |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step (unchanged copies of `tools/kit`) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` (no simulator) |

Third-party assets: none.
