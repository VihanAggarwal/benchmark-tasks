# Pack two uneven bearings up to the datum, then seat the beam level across them

Before a beam, lintel or machine base is set down, its bearings are brought up to the set level (the datum) with
packers or shims laid flat and fully on each bearing. The member's level comes from the packing, never from forcing
it: a beam laid across bad packing sits out of level and bears on one corner, which is the defect. Carpenters,
structural iron and steel workers and millwrights (O*NET 47-2031, 47-2221, 49-9044) do this on every job. Here two
fixed grey piers stand on the table, 20 mm and 30 mm tall, each with a red-capped datum post behind it whose top is
70 mm up. Each arm has a set of colour-coded beech packers in front of it (yellow 20 mm, blue 30 mm, green 40 mm),
and a 420 mm spruce beam lies on the midline. The left pier needs 50 mm, which from one set only yellow + blue
makes; the right needs 40 mm, the green alone. Then the beam goes on, level across both. The score is what a
site engineer checks with a straightedge from the datum, a feeler gauge, a rule and a level: each packed pier
earns 0.25, a seated beam 0.5 more, and a beam left across bad packing, or a packer stood on its edge on a pier,
scores 0. Both arms are used naturally: each packs the pier in front of it, then one carries the beam.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the packers' and the beam's poses, with a small tracker of their last half second |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets, NaN states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pack each grey pier with the wooden packers (yellow 20 mm, blue 30 mm, green 40 mm thick) until its top is level with the top of the red-capped datum post behind it, every packer lying flat and fully on the pier. Then lay the beam across both packed piers." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/beam-packers-to-datum/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pack each grey pier with the wooden packers (yellow 20 mm, blue 30 mm, green 40 mm thick) until its top is level with the top of the red-capped datum post behind it, every packer lying flat and fully on the pier. Then lay the beam across both packed piers." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/beam-packers-to-datum/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: for each pier `*_packers` (on it), `*_counted`,
`*_level_mm` (top of its highest packer), `*_packed` and `*_rows` (per packer `tilt_deg`, `overhang_mm`,
`rests_on`, `still`, `counts`, `on_edge`); for the beam `beam_level_deg`, `beam_face_deg`, `beam_heading_deg`,
`beam_still`, `beam_across_both`, `beam_seated` and per pier `*_beam_rests`, `*_beam_gap_mm`, `*_bearing_mm`,
`*_centre_line_over`; the zero flags `zero_on_edge`, `zero_beam_tilted_across`; and every loose object's
position. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the six packers' and the beam's
poses, once per control step (20 Hz). Its only memory is where each of them was over the last 0.5 s: 'still'
means moved less than 3 mm and turned less than 2 degrees in that window, read from positions because the
reported velocities of a gripped object chatter. Everything else is the end state:

- A packer is *on* a pier when its centre, seen from above, is over the pier's 60 x 60 mm top or within 15 mm of
  it, and its lowest point is more than 10 mm above the table (a packer lying on the table beside a pier is not on
  it).
- A packer on a pier *counts* when it lies flat (thickness axis within 3 degrees of vertical; either 50 x 50 face
  may be down), its outline seen from above (the shadow of its eight corners, turned as it is) stays within the
  pier's top extended by 5 mm, it rests on the pier's top or on a counted packer (its underside within 2 mm of that
  surface, measured at its centre, and its centre over that packer), and it is still.
- A pier is *packed* when at least one packer is on it, every packer on it counts and the top of its highest
  packer is within 3 mm of the datum (70 mm). A stray packer perched on a correct stack, or lying on the beam over
  the pier, stops it being packed.
- The beam *rests on* a surface when its underside comes within 2 mm of it where they meet. The scorer samples
  each support's top face on a 9 x 9 grid, edges included, and takes the smallest gap to the beam's underside
  plane over the points that lie under the beam, so a tilted beam bearing on a pier's edge is seen resting there
  even though its underside is centimetres above the middle of the pier. A support up to 3 mm into the beam
  (contact penetration) still counts; anything lying on top of the beam is not a support.
- The beam is *seated* when both piers are packed and it lies on one of its four long faces (within 5 degrees),
  is level along its length within 1.25 degrees, rests on each pier's top packer with its centre line crossing
  that packer and at least 30 mm of its length over it, and is still. (A centre line that crosses both packers is
  within 9.2 degrees of the line through the piers, so no separate heading limit is needed.)

| Outcome | Score |
|---|---|
| A packer on a pier stands on an edge or end (thickness axis over 45 degrees from vertical) | 0 |
| The beam rests across both piers (on each, on the pier or a packer on it) out of level by more than 1.25 degrees | 0 |
| Both piers packed and the beam seated; still for 3 s | 1.0 |
| Both piers packed, beam not seated (on the table, held above them, under 30 mm of bearing, off a packer's centre line) | 0.5 |
| One pier packed | 0.25 |
| Nothing packed (start; wrong packing; a packer leaning, overhanging, not resting or still moving) | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. 0.75 cannot occur, because seating the beam needs both piers packed.

Why the two zero rules: a packer stood on its 50 mm edge on the 20 mm pier reaches exactly 70 mm. The flatness
rule alone already keeps it from counting; the zero also takes the other pier's 0.25, because a packer on edge is
the unstable bearing the trade never leaves under a member. And a beam resting across both piers out of level is
the defect the trade guards against: a beam set on bad packing. Packers come in 10 mm steps, so a wrong pier is at
least 10 mm off: across one packed and one bare pier the beam tilts 6 to 8 degrees, across a packed pier and one
10 mm off about 1.6 degrees (1.59 with everything centred), and never less than 1.36 degrees, since its two
contacts are at most its 420 mm length apart. Two packed piers differ by at most 6 mm and the inner edge of a
counted top packer is at least 145 mm from the midline, so their contacts are at least 290 mm apart and the beam
tilts at most 1.19 degrees. The limit, 1.25 degrees (about 9 mm over the beam's length, which a phone level reads
easily), sits between the two: a beam on good packing is never zeroed, and a released beam across a pier 10 mm off
always is, wherever the packers sit on their piers.

Unintended strategies earn less, as the tests show: a packer on edge on the 20 mm pier (70 mm, 0); the beam laid
straight on the bare piers (0); the beam across a packed pier and one made up to 60 mm with the green packer alone
(0, even with the packers placed to flatten it to 1.47 degrees); the beam level on two stacks both 10 mm too high
(0: neither pier packed); a stray packer perched on a correct stack, overhanging (that pier not packed); a packer
laid on the beam over a bearing (that pier not packed); the beam held 10 mm above good packing (0.5); a packer held
just above the stack (not resting: not counted); a packer or beam still sliding (not still). Two yellows (one from
each set) on the right pier also make 40 mm and count as packed, but then the left pier cannot be made up to 50 mm.

## One arm alone

With straight-down grips each arm reaches only its own pier and its own packer set (IK on the URDF): at the right
pier's x (28-30 cm) and the heights a packer is placed at there (45-75 mm), the left arm's tool point gets no
further right than about y -12 cm, or about -13 cm holding a packer by its edge (the pads are 20 mm wide), while a
packer counts on the right pier only with its centre between y -17 and -19 cm and is supported at all only from -15 cm; the packers in front of
the right arm (y -30 cm) are far out of its reach. The same holds for the right arm on the left side. One arm can
still carry the beam to the middle (both reach y 0 at x 29 cm), but laid across one packed and one bare pier the
beam tilts 6 to 8 degrees, which scores 0. So one arm scores 0.25 by any straightforward strategy (pack its own
pier, leave the beam). We do not claim more than that: nudging a packer across the far pier with an open finger,
or a tilted wrist, has not been tried.

The demo's one-arm runs in the simulator (see 'Results in simulation' below): with `DEMO_IDLE_ARM=right` the left
arm alone packs the left pier (yellow + blue, top 69.99 mm) and scores 0.25. With `DEMO_IDLE_ARM=left` the right
arm alone packs the right pier (green, top 70.06 mm), then lays the beam where the demo puts it, across the bare
left pier: the beam tips onto the 20 mm pier, 7.9 degrees out of level, which scores 0.

## Demo and run limit

The scripted demo takes 41.9 s (838 control steps). Every grip closes over 1.5 s with the fingers commanded 8 mm
narrower than the part, every release opens over 1.5 s, and every set-down slows for its last 10-15 mm.

- Left arm (fingers across the packers, along y): picks the yellow 20 mm packer (pad tips 4 mm above the table),
  lifts it to 12 cm, carries it over the left pier, lowers the last 15 mm over 1.5 s and lets go with its underside
  1 mm above the pier; then the blue 30 mm packer onto the yellow the same way (top at 70 mm); then waits over its
  own packer row, 16 cm up, out of the beam's way (28.8 s).
- Right arm: the green 40 mm packer onto the right pier the same way (13.9 s); waits at x 20 cm, y -12 cm, 14 cm
  up while the left arm works (over the beam's middle its forearm would come within 6 cm of the left arm's upper
  arm, capsule model; here 13 cm); once the left arm has parked, turns its wrist 90 degrees over the beam's middle,
  grips the beam across its 40 mm width (pad tips 4 mm above the table), lifts it to an underside height of 93 mm
  (23 mm over the packing), carries it over both piers, lowers it to 10 mm above the packing, settles 0.5 s, lowers
  the last 10 mm over 2 s, settles 0.3 s, opens and goes up.

The arms' controllers do not compensate payloads. The static estimate for the 0.227 kg beam at the place pose
(J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) is (-1.4, -1.5, -6.2) mm and a 1.0 degree tilt: the beam comes to
rest on the packing about 1 s before the slow descent ends, the 1.4 mm sideways sag is far inside the 25 mm the
beam may sit off-centre, and two equal packings level it as the grip unloads. For the packers the estimate is at
most 1.3 mm. The 3 s stillness (rather than the usual 1 s) outlasts the time from that first rest until the
fingers are fully open, so a trial does not end while the gripper still holds the beam. On the GPU the beam
first rested on both packings at about step 765 (tilted under 0.5 degrees, level by step 800), and success held for its
3 s at step 825, 7 steps after the fingers were fully open (offline, with no sag: step 835); the scripted policy
stops at step 899. The run limit is 2,400 steps (2 minutes),
about three times the demo, for a model that has to judge two packings and place four parts.

Offline check (`tools/offline/check_task.py`): worst IK error 0.1 mm, smallest joint-4 margin 13.8 degrees, the
arms' collision capsules at least 8.7 cm apart, tool points at least 30.4 cm apart, pad tips never lower than
4 mm above the table, no part or finger running into anything.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the six packers back flat on their marks,
sides parallel to x and y (each set in a row at y +30 cm or -30 cm: yellow at x 16 cm, blue at 22 cm, green at
28 cm) and the beam flat on a 420 x 40 mm face, its length along y, centred at x 15 cm, y 0. The piers and the
posts are fixed. A new episode is recognised when the simulator puts every loose object back on its mark (within
4.5 cm, flat on the table) in a single step; the scorer then forgets the last half second.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is 42 s of simulated time; a full Dream-1 run is
2 minutes); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two concrete paving offcuts or painted hardwood blocks, 60 x 60 mm, one 20 mm and one 30 mm tall, glued or taped
to the bench 360 mm apart; two 20 mm square steel or hardwood posts, 70 mm tall, painted grey with a 10 mm red band
at the top, fixed 25 mm behind the piers; six 50 x 50 mm beech blocks cut 20, 30 and 40 mm thick and painted
yellow, blue and green (36, 54 and 72 g); a 420 x 40 x 30 mm spruce batten (about 230 g). To score: a straightedge
from a post's top across the packing and a feeler gauge for the datum and the rest gaps, a rule for overhang and
bearing, and a phone level on the beam.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 (ended at step 825: packing tops 69.98 and 70.00 mm, beam level 0.0 degrees, 51 mm of bearing at each end) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 (right pier packed; the beam laid across the bare left pier tilts 7.9 degrees) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 (left pier packed, top 69.99 mm) |
| Dream-1 | 0.0 (ran the full 2,400 steps; no packer or the beam was moved) |
