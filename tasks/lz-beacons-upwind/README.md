# Mark a helicopter landing zone: corner beacons 100 ft or more apart, the fifth upwind

Firefighters and EMTs (O*NET 33-2011.00, 29-2042.00) set up landing zones for air-medical helicopters at road
crashes and rural scenes. Services commonly ask for a zone of about 100 x 100 ft (30 x 30 m) on firm, level, open
ground, marked by a beacon, cone or light at each corner, with a fifth marker on the upwind side so the pilot can
read the wind and land into it. The crew reads the wind from a windsock, smoke or flags.

Here the scene is at 1:110. A green grass field (335 x 360 mm, 37 x 40 m at scale) lies in front of both arms, with
a dark grey road along its left edge and a light grey parking strip along its right edge, a red fire engine parked
on the strip, and a windsock beyond the road whose sleeve streams toward the field. Two orange beacons and the blue
one wait on the road, two orange beacons on the parking strip. The robot has to mark the zone: the four orange
beacons at the corners of a zone at least 100 by 100 ft (277 mm) with square corners, all standing on the grass and
let go, then the blue beacon 1 to 6 m (10 to 55 mm) outside the middle of the zone's upwind side, which the windsock
gives (the sleeve streams toward -y, so the wind comes from +y: the left side). Nothing is marked out: the zone has
to be sized against a minimum, squared, kept on the grass with room left on the upwind side for the marker, and the
marker put on the side the sock says. The score is read from where the beacons stand at the end, as a top photo with
a scale would read it: 0.5 for the zone and 0.5 for the wind marker, with a quarter for each when it is nearly
right. Each arm fetches its own side's beacons: gripping from above, neither reaches the other side's, and one arm
alone scores 0 (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the five beacons' states (and the arms' pad-tip points and openings, for 'let go'), with a small tracker for stillness and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the let-go rule (reach, height, tilt, unreadable arms), the exploits we tried, resets, broken states, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer;
`rigstate.py`, in this folder, reads the arms only to check that each beacon was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A medical helicopter is coming to land on this field (1:110 scale). Mark its landing zone on the grass: an orange beacon at each corner of a zone at least 100 by 100 ft (277 mm) with square corners. Then stand the blue beacon just outside the middle of the zone's upwind side, 1 to 6 m (10 to 55 mm) beyond it. The windsock streams downwind. Every beacon must stand upright on the grass on its own." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lz-beacons-upwind/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A medical helicopter is coming to land on this field (1:110 scale). Mark its landing zone on the grass: an orange beacon at each corner of a zone at least 100 by 100 ft (277 mm) with square corners. Then stand the blue beacon just outside the middle of the zone's upwind side, 1 to 6 m (10 to 55 mm) beyond it. The windsock streams downwind. Every beacon must stand upright on the grass on its own." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/lz-beacons-upwind/dream
```

Run one trial per process (the default). The score depends only on where the beacons are and whether a gripper
still holds one; the small tracker in the task module keeps each beacon's path over the last 0.5 s (for stillness,
which only the early end of the trial needs) and diagnostics for the trace, and starts a new episode when rigstate
sees the embodiment reset or the simulator puts every beacon back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`square` and `marker` parts; `orange_not_placed`,
`blue_placed`, `held`; the zone's `sides_mm`, `angles_deg` and `convex`; `upwind_side`, `upwind_normal_deg`,
`blue_out_mm` (d) and `blue_along` (t); `all_still`, the first step each score was reached; each arm's
`left_tips`/`right_tips` and `left_opening`/`right_opening`, `arms_readable`, `reach_mm` and `arms_unavailable`, or
`rigstate_error`; and each beacon's centre, tilt, bottom above the grass, `in_outline`, path over the last 0.5 s,
`held` and `placed`). Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the five beacons' poses once per control
step (20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the
finger tips) and how far it is open, and scores the state at hand: the score is read at the last step. Distances are
between beacon centres, seen from above, as a tape between beacons or a top photo with a scale reads them once the
arms have withdrawn.

A beacon's tilt is the angle of its axis from vertical (either end up: the simulated beacon's ends are alike). A
beacon is placed when it is tilted at most 10 degrees, its lowest point (worked out from its pose) is within 3 mm of
the grass top, its centre is inside the grass outline (x 13.0 to 46.5 cm, y -17.0 to +19.0 cm), and it is let go: no
gripper's pad-tip point is less than 10 mm above its top while within that gripper's finger reach of its axis. The
gripper's yaw is not read, so the reach is the widest over every yaw, hypot(half the finger gap + 10 mm, 20 mm) + 14
mm + 5 mm: 79 mm fully open, 50 mm closed on a beacon, 41 mm closed on nothing. So a beacon on the road or the
strip, tipped over, stacked on another, held above the grass, or still gripped, pressed or propped by a finger is
not placed. Each gripper is read on its own: on a step where one cannot be read, the let-go rule skips it but still
checks the other (such steps are counted in the trace's `arms_unavailable`). The four orange centres are joined in
order of angle round their centroid into the zone (whichever beacon is at whichever corner): four sides, centre to
centre, and four corner angles. Within the angle limits below the zone is always a convex quadrilateral. The upwind
side is the zone's side whose outward direction (square to it, away from the zone) is nearest to +y, where the wind
comes from; if two sides are equally near (a zone turned 45 degrees, which this field cannot hold), either counts. d
is the blue centre's distance outside that side's line (negative inside it) and t its position along the side, 0 at
one corner and 1 at the other.

| Part | Condition | Points |
|---|---|---|
| Zero first | Any orange beacon not placed (off the grass, tipped, stacked, held up, still held) | 0 in all |
| Square | Every side at least 277 mm (100 ft) and every corner 80 to 100 degrees | 0.5 |
| | Every side at least 249 mm (90 ft) and every corner 70 to 110 degrees | 0.25 |
| | Anything else (smaller, out of square) | 0 in all |
| Wind marker | Blue placed, d 10 to 55 mm (1 to 6 m), t 1/3 to 2/3 (the middle third) | 0.5 |
| | Blue placed, d over 0 and at most 100 mm, t 0 to 1, not all of the above | 0.25 |
| | Anything else (inside the zone, beside another side, past a corner, more than 100 mm out, not placed) | 0 |

The score is square + wind marker. Every outcome, with numbers: the demo's 292 mm zone square to the field with the
blue 30 mm outside the middle of its left (upwind) side, 1.0; the same with the blue beside a corner of that side (t
under 1/3) or only 5 mm out, 0.75; a 280 mm zone with the blue 70 mm out, 0.75; a 260 mm zone (90 to 100 ft) with
the blue 30 mm out at the middle of the upwind side, 0.75; the 292 mm zone with the blue outside its right
(downwind) side (the sock misread), inside it, or still gripped on its spot, 0.5; a 260 mm zone with the blue beside
a corner of the upwind side, 0.5; a 260 mm zone with the blue downwind, 0.25; a 166 mm (60 ft) zone, a 250 mm zone
with 65 degree corners, an orange beacon used as the wind marker (the blue at a corner), an orange beacon left on
the road, or one still gripped at its corner, 0. So every score 0, 0.25, 0.5, 0.75 and 1.0 can be reached. Any
orange beacon may be any corner, and a zone larger than 277 mm on a side (or a rectangle, say 320 x 300 mm) counts
as long as it has square corners and stays on the grass.

The blue beacon must stand on the grass as well, so the zone's upwind side must be at least 10 mm inside the field's
left edge (at most y +18.0 cm): the zone needs room for its marker, as on a real field, and a zone spread over the
whole field leaves none (0.5 at best). A 277 mm zone fits on this field turned at most about 14 degrees (a 249 mm
one about 27), so the upwind side is always the zone's left side, within that turn. The trial ends early (success)
once 1.0 has held with every beacon still (its centre moved under 5 mm in all over the last 0.5 s) for 3 s;
otherwise it is scored as it stands at the run limit. A beacon counts only once let go; the 3 s hold is also longer
than the settle and release after the last set-down (0.5 + 1.5 s), so even on a step where the arms could not be
read the trial cannot end while a beacon is still in a closed gripper. Stillness is read from each beacon's path, not
from the simulator's reported velocity, which chatters while a part is gripped. A state with a non-finite number or
a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: an orange beacon as the wind marker with the blue at a
corner (0); the blue stacked on an orange beacon or held above the grass (not placed: 0.5 at most); a beacon still
gripped on its spot at the end (not placed: the blue still gripped 0.5, two orange beacons gripped at their corners
0); a zone with its upwind side on the field's edge and the blue on the road beyond it (0.5); a corner on the road
(0); beacons laid on their sides on the right spots (0); two orange beacons side by side at one corner (0).

## One arm alone

The demo splits the work by side: the left arm fetches the two orange beacons and the blue one from the road and
sets the zone's two left corners and the wind marker; the right arm fetches the two orange beacons from the parking
strip and sets the two right corners. With one arm idle, the other's side is done but the idle side's two orange
beacons stay where they started, off the grass, and the score is 0: in simulation the scripted demo scored 0.0 with
the left arm idle (the road's two orange beacons and the blue one still on the road; the right arm's two corners
set) and 0.0 with the right arm idle (the strip's two orange beacons still on the strip; the left arm's two corners
and the blue marker set).

Gripping straight down, at any heading (tried every 30 degrees), the left arm cannot reach the parking strip's
beacons (x 20.0 and 24.0 cm, y -21.5 cm) and the right arm cannot reach the road's (y +22.5 cm), by our IK model of
the arm (`tests/test_scoring.py`, `OneArmReach`). Even given all four orange beacons on its own side, the largest
square zone one arm could set on the grass straight down is about 240 mm on a side for the left arm and about 230 mm
for the right (exact IK at the four corners at set-down height, centres on a 5 mm grid, the zone turned -20 to +20
degrees, grip headings 0, +-45 and 90 degrees; a 1 cm reach-map search over the whole field agrees), under the 249
mm of the 0.25 tier; only a 249 mm rhombus with 70 degree corners just touches the edge of the left arm's 1 cm reach
map (none for the right), and a 277 mm zone fits neither. We do not claim one arm is strictly unable: with the wrist
tilted 60 to 70 degrees from vertical the model's tool point does reach the far side's beacons, so a policy able to
grip or push beacons from the side might do more. The scripted demo grips from above only. (GPU results: 'Results in
simulation' below.)

## Demo and run limit

The scripted demo takes 48.1 s (962 control steps), both arms on one clock, mirror images for their first two
beacons. Each beacon is gripped straight down with the fingers closing along y, opened fully (94 mm: 33 mm either
side of a 28 mm beacon, its neighbours on the road 60 mm away) and commanded to 20 mm to close, with the tool point
20 mm above the beacon's bottom so the pads hold its lower 7 to 28 mm and their tips stay 8 mm above the table.
Every gripper open or close takes 1.5 s, every final descent (onto a beacon and onto the grass) is slow (20 mm over
1.2 or 1.5 s), each beacon is let go 1 mm above the grass after a 0.5 s settle, and beacons travel with their bottoms
60 mm up (31 mm over any standing beacon, 32 mm over the fire engine). In order:

1. The left arm sets an orange beacon from the road at the zone's near left corner (x 14.8 cm, y +14.6 cm); at the
   same time the right arm sets one from the strip at the near right corner (x 14.8 cm, y -14.6 cm).
2. The left arm sets the other road orange at the far left corner (x 44.0 cm, y +14.6 cm), the right arm the other
   strip orange at the far right corner (x 44.0 cm, y -14.6 cm). The zone is 292 x 292 mm, 15 mm over the minimum.
3. The left arm stands the blue beacon at x 29.4 cm, y +17.6 cm: 30 mm outside the middle of the left (upwind)
   side, 14 mm inside the grass. The right arm parks over the empty strip.

The grippers stay at least 29.2 cm apart and the arms' collision capsules at least 9.0 cm (offline). The smallest
joint margin is 13.4 degrees (joint 4, both arms, over the far corners at travel height). A 40 g beacon barely loads
an arm; the zone has 15 mm to spare on every side, the marker 20 mm on its distance band and 49 mm along the side,
and every beacon stands 14 to 25 mm inside the grass's nearest edge, all far more than the few millimetres an arm
trails its command. `PLACE_TRIM` in `task.py` moves an arm's place points if the GPU run shows them landing off by a
steady amount. The blue beacon counts once the left gripper, open, has risen with its pad tips 10 mm over the
beacon's top (about 46.1 s; step 923 in simulation); success is then held at step 982 (49.1 s), after both arms
have parked, where the simulated run ended; the scripted policy would stop at step 1,023. The grippers pass at least 53 mm from the windsock and 41 mm
from the fire engine (capsules). The run limit is 2,400 steps (120 s), two and a half times the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the orange beacons back upright on the
road centred at x 20.0 and 26.0 cm, y +22.5 cm, and on the parking strip centred at x 20.0 and 24.0 cm,
y -21.5 cm, and the blue beacon on the road centred at x 32.0 cm, y +22.5 cm; the mats, fire engine and windsock are
fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 49 s of simulated time, when
success has held for 3 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 335 x 360 mm piece of green artificial turf or felt (1 mm thick or so) taped to the table 13 cm in front of the
arm bases; a 380 x 70 mm dark grey card or vinyl strip along its left edge (the road) and a 380 x 80 mm light grey
one along its right edge (the parking strip), their tops flush with the grass; a 90 x 24 x 27 mm red-painted wooden
block glued on the strip (the fire engine); a 10 mm white dowel 90 mm tall fixed upright beyond the road with a
five-band sleeve of stiff card at its top (orange, white, orange, white, orange; 18, 16, 14, 12 and 10 mm square, 12
mm each, its axis 80 mm up), pointing toward the field. Five 28 mm round plastic cylinders 28 mm tall weighted to
about 40 g (short lengths of pipe with a plug, or turned plastic), four painted orange and one blue. Positions in
`task.json`'s `setup`.

Scoring a real trial: take a top photo with a ruler on the grass for scale (or measure directly). Tape between the
centres of neighbouring orange beacons (each side at least 277 mm for full marks, 249 mm for a quarter); measure each
corner with a protractor on the photo, or check it against a set square (80 to 100 degrees: within 10 degrees of
square); then square off the left side's line and measure the blue beacon's centre out from it (10 to 55 mm) and
along it (within the middle third). Check every beacon stands upright with its centre on the grass, after the
arms have let go and withdrawn.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
