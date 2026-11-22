# Put a call headset on a child-size mannequin head: boom down, cups on both ears

A child-size display head with moulded ears (Xigaocun children's manikin head for hats and wigs, Amazon ASIN
B0BM3QNMWF; listed head circumference 48 cm, ears protrude) is taped to the table facing the robot. In front of it a stereo call headset
(Jabra Evolve2 30 SE, 102 g) hangs on its desk stand with its microphone boom stowed straight up. The head stands
in for a child about to join an online class: the robot lowers the boom, puts the headset on the head with each cup
on its own ear (the boom's cup on the head's left) and lets go.

The headset grips the head through its sprung band. Unworn, the cushions rest 136 mm apart, 24 mm less than the
ears (160 mm across), so both cups have to be pulled apart, carried over the head level, and let in onto the ears
together; on the head each cup is pushed out 12 mm and presses its ear with about 2 N. The boom swings forward and
down about 118 degrees from its upper stop to its lower stop, which puts the microphone about 40 mm in front of the
mouth's left corner. The boom is a 104 mm arm from its pivot beside the left cup, then a tip arm running about 88 mm
in toward the face (est; no maker figure), and the setup check below confirms the real boom's reach. On the real headset a stop plus the hinge's friction holds the boom; in the sim gravity holds it
on the stop. The headset's cable is already plugged into a
laptop on the table behind the head (not part of the task) and lies slack, clear of the arms; it counts for no check,
and the sim models it as absent.

## Real parts and setup checks

Only the head's circumference (48 cm) and the headset's weight (102 g) are maker data; every other shape figure is
the sim's design value (est). Jabra publishes no clamp force for this headset; the sim's is 2.0 N per ear (167 N/m
over the 12 mm each cup is pushed out, est). Before the first trial (tape, ruler, calipers, 0 to 10 N spring scale),
and a part that fails a check is replaced: The sim's cranium is a round section 140 mm across (44 cm round); a real 48 cm head is longer
front to back than across, so the checks hold what the headset touches: the width over the ears, the ear height and
the crown.

| Check | Accept |
|---|---|
| Head round just above the ears | 46 to 50 cm |
| Across the ears' outer faces | 150 to 170 mm (sim 160; under 150, glue a 25 x 35 mm craft foam pad on each ear) |
| Ear centres above the neck base; crown height | 105 to 125 mm (sim 115); 195 to 215 mm (sim 205) |
| Cushion inner faces, unworn | 126 to 146 mm (sim 136) |
| Clamp force: headset on the head by hand, spring scale on one cup's outer rim pulled straight out until the cushion lifts off its ear | 1.5 to 3.0 N (sim 2.0); repeated every 20 trials |
| Fit: band on the crown; boom lowered | each cushion covers its ear; boom holds up and down on its own; microphone within 50 mm of the mouth's left corner |

## Why it is hard

1. **Two hands spread one springy part.** Each hand pinches one 60 mm cup front to back and pulls it out 24 mm
   against the band; the hands carry the open headset together. One hand alone cannot open it: the free cup's
   cushion meets the cheek and the ear.
2. **Two targets at once, next to a face.** The cups go down outside the ears (8 to 9 mm clear of them in the GPU
   run), then in onto the ears while the band settles over the crown; a cup let in early or low misses its ear.
3. **A small part set to a state.** The boom (a 6 mm arm) is pushed forward past upright so it drops onto its
   lower stop; it must be on the correct side, so the headset cannot go on backwards.

## Scoring

| Condition | Rule (sim) | Bench method (real trial) |
|---|---|---|
| Latch: left the work surface | any part past the table's edge, or any part 10 mm or more below the table top (fell off), at any step; set down on the table and picked up again does not count | the trial video: any part past the edge or below the table top |
| Off the stand | band underside 15 mm or more over the cradle top, or its centre 30 mm outside the cradle top in plan (sliding along it does not count) | the band no longer touches the cradle |
| Seated | band centre within 15 mm of its seated place (over the head's axis, underside on the crown), tilt under 0.25 rad | band on or within 15 mm above the crown, centred within 15 mm; cup heights within 20 mm; ruler on the end photo |
| On the ears | each cushion face centred within 15 mm of its own ear and no more than 3 mm out from the ear's outer face; the boom's cup on the head's left ear | end photo: each cushion hides a 30 mm ring drawn round its ear's centre (115 mm up), so the cup is centred within 15 mm; a 3 mm card slid straight down against the cushion face, on the ear's front to back centre line, does not pass between cushion and ear (it stops on the ear's top or the cushion's rim) |
| Mic down | boom within 0.25 rad of its lower stop, microphone within 50 mm of the mouth's left corner | boom on its stop; ruler on the end photo |
| Resting | no part moving 5 mm/s or faster | nothing moves 5 mm in the last 1 s of video |
| Hands off | both finger zones 30 mm or more from every part of the headset | no finger touches it in the last 1 s; a 30 mm block passes |

| Outcome | Score |
|---|---|
| Latch fired, or still on the stand | 0 |
| Off the stand, not seated on both ears | 0.25 |
| Seated on both ears, mic not down | 0.5 |
| Seated, on the ears, mic down at the last step; held, touched or moving in the last 1 s | 0.75 |
| All of it on every step of the last 1 s (20 steps) | 1.0 |

| Bypass | What stops it |
|---|---|
| Push it on with one hand, cups unspread | the cushions meet the cheeks and ears: not on the ears, or it falls |
| Leave the cups held apart over the ears | cushion faces 12 mm out from the ears (over 3 mm): 0.25 |
| Put it on backwards | the boom's cup on the right ear: 0.25 |
| Lower the boom and leave the headset on the stand, or slide it along the cradle | 0 |
| Hold it in place | hands off fails: 0.75 |
| Drop it off the table, then pick it up and put it on | the latch: 0 |

## Results

GPU re-run on the current scorer, 22 Nov 2026 (job 20261122T052208): scripted demo 1.0; one arm idle (left) 0.0; one
arm idle (right) 0.0; one Dream-1 trial at the full run limit 0.0. The setup screenshot and the videos are from this run.
The detailed table and trace notes below are from the earlier 19 Nov run (job 20261119T080500, seed 0, object_jitter 0),
which scored the same.

| Run | Offline prediction | GPU |
|---|---|---|
| Scripted demo | 1.0 (success held at step 647 of 691) | 1.0 (1.0 by step 633; run ended at step 649 with it held) |
| One arm idle: left (right arm only) | 0.0 | 0.0 (boom lowered; headset slid 15 mm along the cradle, never off it) |
| One arm idle: right (left arm only) | 0.25 (the proxy lifts it by one cup, knocks it on the head and drops it on the table) | 0.0 (boom still stowed; headset slid 17 mm along the cradle, never off it) |
| Settle (160 steps, no motion) | 0.0 | 0.0 (nothing moved: 0.0 mm/s on every step, boom and cups at 0) |
| Dream-1, one trial at 1200 steps | | 0.0 at the 1200-step limit (job 20261119T061905, run on the current instruction) |

GPU demo, from the per-step trace (sampled every 8 steps):

- The right arm pushes the boom from about step 90; it is on its lower stop (-2.06 rad, 118 degrees) by step 145.
  The headset does not move on the cradle.
- Both cups pinched by step 249 (the fingers stall at opening 0.638 on the 60 mm shells); pulled out 22.3 and
  22.7 mm by step 289.
- Off the stand by step 313; lifted 35 mm over the cradle and carried to over the head's axis by step 409.
- Lowered over the crown (seated by step 449, the cups 8 to 9 mm clear of the ears); the cups let in onto both
  ears by step 481 (score 0.75 while still held).
- The grippers open from step 537 and are 30 mm clear between steps 610 and 617; the score is 1.0 at step 633 and
  holds to the end.

End state (step 649): the band 1.8 mm toward the face from the head's axis and 0.1 mm sideways, its underside
5.1 mm above the crown (5.4 mm from the seated place; the cups' clamp on the ears holds it there), tilt 0.002 rad;
each cushion face 5.4 mm (right) and 5.5 mm (left) from its ear's centre and 0.0 mm out from the ear's outer face,
the cups pushed out 11.9 and 12.1 mm (about 2 N each); boom on its lower stop, microphone 42.8 mm from the mouth's
left corner; nothing moving (0.0 mm/s); the nearest finger zone 93.3 mm from the headset.

One arm idle: neither arm alone takes the headset off its stand. With the left arm idle the right arm still lowers
the boom, then its lone grip and pull slide the headset 15 mm along the cradle; with the right arm idle the boom
stays up and the headset slides 17 mm. Both score 0.

The latch (any part past the table's edge or below its top) never came near in these runs: from the traced band
pose, tilt and joints, the headset's lowest point stayed 68 mm (demo), 55 mm and 73 mm (one arm idle) above the
table top, and every part at least 117 mm inside the table's edges.

Counterfactuals (offline prediction): boom never lowered 0.5; cups never pulled apart 0.0.

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
        -E control_hz=20 --scorer task_score --max-steps 1200

Seed 0, object_jitter 0. The demo plan is 691 steps (34.6 s); on the GPU the run ended at step 649 (32.5 s) with
success held. `DEMO_HOLD_ONLY=1` settles, `DEMO_IDLE_ARM=left` or `right` drops one arm, `DEMO_VARIANT=nopush|nospread`
plays a wrong way, `TASK_TRACE=<file>` writes one JSON line a step. Tests: `python3 -m unittest discover -s tests -q` (offline, NumPy only).

## Files

`task.py` (objects, hooks, demo), `layout.py` (every number), `parts.py` (shapes and joints), `scoring.py` (the
tracker), `offline_headset.py` (offline stand-in for the physics, used only without a simulator), `mechkit.py` and
`rigstate.py` (the kit, unchanged), `tiltgrip.py` (rolled grips for the demo: both wrists lean outward so the
housings clear the band over each cup).
