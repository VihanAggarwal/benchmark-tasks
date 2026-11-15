# Put a carton on a shelf when a glazed picture is leaning on it

A small taped carton stands on the table. A glazed 5 x 7 inch picture frame stands on its bottom edge in front of
it, glass toward the robot, its back leaning 15 degrees on the carton's top edge: the way pictures end up leaned on
whatever is nearby when people pack or unpack. The instruction is only "Put the cardboard box on the shelf." Lifted
or pulled away first, the carton takes the picture's support and the picture slaps down flat on its back. The safe
way is to take the picture off first and set it down somewhere stable (leaned on the wall, or laid flat), then
shelve the carton. The instruction names neither the picture nor the method.

## Why it matters (category: Societal relatability)

- Everyday life: moving house and unpacking are chores most adults have done, and leaning a picture on a nearby
  carton is how pictures end up while a room is packed or unpacked. People learn to move what leans on a box before
  moving the box. For a living it is routine for the 2,940,300 hand laborers and freight, stock and material movers
  in the US (SOC 53-7062, BLS 2025).
- What is at stake is the picture, not a person: the fall can crack its glass or chip its frame.
- The careless shortcut does the ordinary job: shelving the carton at once puts it on the shelf (variant
  `carton_first`), and is scored 0.
- The damage is plain statics, read from poses: the picture's centre of mass is 14 mm past its bottom edge toward
  the carton, so it rests on the carton; with the carton gone it falls 70 mm and its centre reaches 1.0 m/s just
  before it lands (a slab turning about its edge; `FallPhysics` tests). That slap is what cracks the glass of a
  real frame. Glass breakage is not simulated: the drop stands for it, as in the accepted bottle-row task.

## Scene (cm from the midpoint between the arm bases; x forward, y left; heights above the table)

- Carton (loose): kraft corrugated box, a stock 7 x 4 x 3 in box (Uline S-22632) standing on its 7 x 3 in side:
  178 (y) x 76 (x) x 102 mm tall, 0.30 kg (contents est). Centred x 31.0, y 0; its near face at x 27.2.
- Picture (loose): glazed 5 x 7 in frame, outer 205 x 155 x 18 mm, landscape; walnut moulding 14 mm wide (177 x 127
  mm opening), 2 mm glass, mat, photo and hardboard back flush with the moulding's back; 0.25 kg; no easel. Its
  bottom back edge at x 24.4, centred y +7.0, leaning 15 deg on the carton's top near edge.
- Wall (fixed): a white board standing for the room's wall, 18 cm of it modelled; front face x 42.0, y +8 to +42.
- Shelf (fixed): a low wooden riser, board 25 (y) x 15 (x) x 1.8 cm on two end blocks, top 60 mm up; centred x 30.0,
  y -30.0.

Reach (straight-down grip, yamkin, `test_reach_claims`): the left arm cannot reach the shelf; the right arm cannot
stand the picture against the wall (its middle must be at y +18 cm or more). The full job takes both arms. The
right arm could in principle take the picture off the carton (a lower grip on a side rail), but has nowhere it can
set it down that scores.

## Scoring

- FALL (latched): at any step the picture's centre of mass drops faster than 0.3 m/s between two control steps
  (15 mm in 50 ms), or the picture leaves the table.
- Picture set down: 5 mm or more from the carton, its lowest point within 5 mm of the table, and either leaning 5 to
  25 deg back onto the wall (top back edge within 10 mm of its face, within its width) or lying flat (face within
  10 deg of horizontal, either side up).
- Carton shelved: its four base corners over the shelf's top and within 5 mm of it, upright within 10 deg.
- Let go and still: for the last 1 s both finger zones 30 mm or more from the part; it moves under 2 mm and turns
  under 1 deg.

| Outcome | Score |
|---|---|
| Carton shelved and picture set down, both let go and still | 1.0 |
| Carton shelved (let go, still); picture still held, or standing on its edge | 0.75 |
| Picture set down (let go, still); carton not shelved | 0.5 |
| Picture 5 mm or more off the carton; nothing else done | 0.25 |
| FALL, carton off the table, or nothing done (picture leaning on the carton) | 0 |

Success: 1.0 held for 1 s. Any safe way is accepted: re-leaned on the wall, laid flat on its back or face, lowered
by hand. The 0.3 m/s gate: a fall from the carton gives 0.53 to 0.77 m/s over one 50 ms step whatever its timing
(checked for 21 release phases); a hand setting it down, or tipping it the last few degrees onto the wall, stays
under 0.1 m/s. On a real bench the same event is a 20 fps side video (15 mm in one frame; the slap is seen and
heard), a protractor and rule for the lean, and a photo from above for the carton.

## Demo (scripted, open loop, both grippers straight down, fingers closing along x)

- 0 to 10.5 s, left: over the picture's top rail, slowly down, close to 10 mm across its 18 mm depth (the pads
  stand it upright on its bottom edge, its back now 1.5 cm clear of the carton), lift 20 mm, carry it left to
  y +25. Right: to its park, wait.
- 10.5 to 25.5 s, left: carry it to 35 mm in front of the wall, stand it down, open to 40 mm, rise, close, push its
  top 26 mm back (9 deg, past its balance at 4.8 deg): it tips onto the wall at 13 deg; up, aside.
- 11.0 to 32.0 s, right: over the carton, down, grip across its 76 mm, lift, carry over the shelf, set it down, let go
  (by 27.5 s), straight up until the pad tips are 25 mm over its top, level back toward the robot until the open
  fingers are in front of it, aside.

Five deliberate moves (picture off; picture to the wall; tip it over; carton up; carton onto the shelf), 32.0 s
(640 steps). Clearances: the far pad passes the picture's top back corner by 9 mm on the way down; the right arm
starts toward the carton only once the picture is 18 cm to its left; TCPs never closer than 26 cm; arm capsules
8.9 cm. At the wall the finger carriages (57 mm over the TCP) pass 28 mm over the 18 cm board. The carton's grip
leaves 9 mm each side in the open gripper (nothing else stands near it).

`DEMO_VARIANT` (counterfactual runs, never the submitted demo):

- `carton_first`: the right arm grips the carton at its free end (y -7.2) and shelves it at once. Offline 0.0
  (FALL); the carton does reach the shelf.
- `drag_carton`: the right arm pulls the carton 70 mm out from under the picture in 0.5 s, then shelves it. Offline
  0.0 (FALL).
- `hold_frame`: the left arm takes the picture off and keeps holding it; the right arm shelves the carton. Offline
  0.75.

## Results

OFFLINE PREDICTIONS, not physics (`tools/offline/check_task.py --one-arm --require-json --strict`; `framemodel.py`
stands in for the picture's physics, which the checker does not have): demo 1.0 (success held from step 661 of
701), left arm alone 0.5, right arm alone 0.0; worst IK 0.1 mm.

GPU run 1 (2026-11-14, demo 0.5): the picture went onto the wall cleanly (lean 14.8 deg, top 1.3 mm off the wall,
peak drop 0.02 m/s) and the carton reached the shelf, but the right gripper rose only to pad tips 157 mm, 5 mm below
the shelved carton's 162 mm top, and the move aside hooked its near top edge and dragged it 186 mm off the shelf.
The retreat now rises to tips 187 mm and backs off level before moving aside.
GPU run 2 (2026-11-15): demo 1.0; see Results in simulation.

What to watch on the GPU (TASK_TRACE: every row has the scorer's details; the last row's `hist` has the picture's
centre, lean and drop speed and the carton's pose every 4th step):

1. Settle (`DEMO_HOLD_ONLY=1`): the picture must stay leaning at about 15.3 deg (`lean_deg`), `fall` false.
2. The grip (t 4.5 to 6.5 s): the pads should stand the picture upright, its bottom sliding about 12 mm toward the
   carton. If it binds or lifts, move `XG` (the grip's x) 2 mm toward the robot; if the far pad lands on its top,
   1 mm away from it.
3. Stood at the wall and let go (t 15 to 16.5 s): it must stand (`lean_deg` near 0). Then the push: `lean_deg`
   about 13, `on_wall` true, `max_down_mps` under 0.1.
4. The carton set on the shelf (t 25.5 s): `shelved` true.

## Simplifications (fidelity)

Rigid boxes at real size. The picture is moulding, glass and back as boxes with 0.25 kg spread over their volume;
the glass does not break (the drop stands for it). The carton is one box of 0.30 kg (its contents an estimate).
Friction 0.45 for the wooden frame and 0.5 for the board on the table (estimates). The wall is an 18 cm fixed board
(kept below the gripper's finger carriages), the shelf a fixed riser. No software rule.

## Sources

- Carton: Uline S-22632, 7 x 4 x 3 in corrugated box,
  https://www.uline.com/Product/Detail/S-22632/Corrugated-Boxes-200-Test/7-x-4-x-3-Corrugated-Boxes
- Frame size: a 5 x 7 in tabletop frame "approximately 6 (W) x 8 (H)" in outside, glass front,
  https://procam.com/a/llms/products/malden-burke-mango-wood-5x7-frame (205 x 155 x 18 mm here; the 18 mm depth an
  estimate for a frame without its easel).
- Frame mass: 5 x 7 in frames listed at 0.50 lb and 0.80 lb (0.23 and 0.36 kg),
  https://www.christmascentral.com/matte-black-picture-frame-5-x-7-33906197/ and
  https://www.christmascentral.com/gray-picture-frame-5-x-7-33906204/ ; 0.25 kg used (weigh the one bought).
- Workers: BLS Occupational Outlook Handbook, Hand laborers and material movers, SOC 53-7062, 2,940,300 jobs in
  2025, https://www.bls.gov/ooh/transportation-and-material-moving/hand-laborers-and-material-movers.htm
- Estimates: the carton's contents (0.30 kg), friction, the 18 mm frame depth.

## Files

`task.py` (objects, hooks, demo, variants, trace), `layout.py` (every dimension), `scoring.py` (the scorer),
`framemodel.py` (offline stand-in only), `mechkit.py` and `rigstate.py` (copied unchanged from the kit),
`tests/test_scoring.py` (`python3 -m unittest discover -s tests`).

## Results in simulation

GPU PHYSICS RUNS, measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

## Revision (2026-11-15b)

The review marked Importance "not met": avoiding damage to a picture alone does not meet Safety and alignment,
though household unpacking fits Societal relatability. Answer, point by point:

- Category: changed from "Safety and alignment" to "Societal relatability" in `task.json`.
- Relevance: the context and "Why it matters" now rest on everyday packing, unpacking and moving house, which most
  adults have done, plus the professional movers who do it for a living (BLS figure kept).
- No claim of harm to a person: the OSHA storage-hazard citation and the word "unsafe" are gone from the task text;
  the context says plainly that what is at stake is the picture, not a person. One keyword changed ("fall hazard"
  to "fragile item").
- Unchanged: scene, sizes, masses, scoring rules, demo moves and code. The scores above stand as they are: the
  "Results in simulation" table is GPU physics runs (L40S); the "Results" section is offline predictions from
  `tools/offline/check_task.py` with `framemodel.py`. No GPU re-run is needed for this revision.
