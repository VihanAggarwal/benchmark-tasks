# Hang up a desk phone: set the handset back so each end sits in its own cradle cup

Every call ends with the handset going back on the phone. A receptionist or office clerk who has laid the handset on the
desk while looking something up picks it up, turns it, and sets it down so its two ends drop into the cradle's two cups;
its weight presses the hook switch and the line closes. A handset left across the cradle rocks off, keeps the line busy and
lands on the desk. Here a loose 1.10 kg desk phone (charcoal, 200 x 240 x 55 mm) stands on the desk with two cups sunk 10 mm
in its top and a spring plunger (the hook switch) between them, standing 8 mm proud. A 0.17 kg handset (two square pads joined by a round handle) lies face down on the
desk to its right, long axis pointing away from the robot. A monitor stands behind the phone and a pen pot beside it. The
robot picks the handset up by its handle, turns it a quarter turn, lowers its two pads into the cups (the mouthpiece pad is
longer than the earpiece cup, so it only seats one way round), lets go and takes its hand away, without shoving the phone.
`setup_drawing.svg` is a dimensioned drawing of the handset, the phone and the start layout.

## Why it is hard

1. **The fit.** Each pad has 8 mm to spare each side across it (x) and 6 mm at each end (y) in its cup. The turn must be a clean quarter turn and the handset
   lowered level: a pad that lands on a cup's rim rests on the top, 10 mm too high, and the handle never reaches the switch.
2. **One way round.** The 70 mm mouthpiece pad cannot enter the 58 mm earpiece cup. Turned end for end it lies on the phone and
   scores at most 0.25.
3. **The hook switch.** The line closes only if the handle presses the plunger down 2 mm or more. The seated handle's underside is
   4 mm over the phone's top and presses the plunger 4 mm (it stands 8 mm proud: 63 mm over the desk, handle underside 59 mm).
4. **A phone that moves.** The phone is loose (1.10 kg on rubber feet, friction 1.0): shoving it more than 8 mm, or tilting it more
   than 3 degrees, scores 0, so the cups cannot be brought to the handset.
5. **Hands off.** The score is read with both grippers 30 mm clear and both parts still for 1 s: a held handset earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Pad in its cup | the pad's four bottom corners inside its own cup (0.5 mm allowed) and 2 mm below to 6 mm above the cup floor (earpiece pad in the earpiece cup, mouthpiece pad in the mouthpiece cup); read in the phone's own frame |
| Pad resting on the phone | the four bottom corners over the phone's top outline and 2 mm below to 14 mm above the cup floors (in a cup, or on the flat top 10 mm higher) |
| Seated | both pads in their own cups with every corner 2 mm below to 3 mm above the floor; long axis within 6 degrees of the phone's (plan view); underside within 3 degrees of level against the phone; hook plunger down 2 mm or more |
| Hands off | last 1 s: both finger zones 30 mm or more from the handset and the phone; each moves under 5 mm and turns under 5 degrees (visible motion) |
| Faults (latched) | handset or phone more than 10 mm below the desk top; phone more than 8 mm from its place or tilted more than 3 degrees |

| Outcome | Score |
|---|---|
| Seated and hands off | 1.0 |
| One or both pads in their own cups, both pads resting on the phone, not seated (an end up on a rim, plunger not down, turned a little) | 0.5 |
| Both pads resting on the phone, neither in its own cup (lying across the top, or end for end) | 0.25 |
| Anything else, a fault, or not hands off | 0 |

Doing nothing scores 0 (the handset is beside the phone). How the handset is tilted on the way is not judged, only where it ends. Success: 1.0 held for 1 s.

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| Pad height (2 mm / 3 mm / 6 mm) | feeler gauges under the pad bottoms at the corners, against the cup floor |
| Pad inside its cup (0.5 mm) | a 0.5 mm feeler between the pad and the cup wall all round |
| Heading (6 degrees) | try-square on the phone's long side: the handset's two ends within 20 mm of parallel over its 197 mm |
| Level (3 degrees) | phone level on the handle, 10 mm rise over its 197 mm is 3 degrees |
| Hook switch (2 mm) | depth gauge on the plunger, or the phone's line lamp / dial tone as the closed line |
| Hands off (30 mm / 5 mm / 5 degrees) | 30 mm spacer block against each hand; 30 fps video of the last 1 s; a ruler or grid sheet under the handset and phone (5 mm), eye or a protractor against the phone's edge (5 degrees) |
| Phone shift (8 mm / 3 degrees) | an 8 mm outline taped round its feet; phone level on its top |
| Fall | by eye and a ruler for 10 mm below the desk |

## Demo, run limit and runtime

The right arm alone does it: down beside the handle (pads 22 mm in its 34 mm height, fingers commanded to 26 mm across the
34 mm handle), grip, lift 75 mm, carry 21 cm over the cradle turning a quarter turn to the left (yaw 0 to +pi/2), lower 2 mm
past the cups' floor (2 s), let go (1.5 s), up, away. 20 s (400 steps); success predicted offline at step 429; the policy stops
at 461. The run limit is 1,200 steps (1 minute). The left arm stays at its ready pose. One arm is enough, as in the real job:
`DEMO_IDLE_ARM=left` is the same demo (offline prediction 1.0, GPU 1.0); `DEMO_IDLE_ARM=right` scores 0 (nothing moves; GPU
0.0). This task does not claim to need two arms.

Switches: `DEMO_HOLD_ONLY=1` (settle check), `DEMO_IDLE_ARM=left|right`, `TASK_TRACE=trace.jsonl` (one JSON line per control
step: each pad's height over the cup floor, its offset from the cup wall, the handset's heading and tilt against the phone, the
hook's travel, the phone's shift). Trial runtime on an L40S once Isaac Sim is running: about 1 to 1.5 minutes; the first start
on a new instance takes several minutes more. The layout is fixed (`-E object_jitter=0`): there is no seed.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The call is over. Hang up: put the handset back on the phone so each end sits in its own cup of the cradle and the hook switch is pressed down, then let go. Don't shove the phone about." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/handset-cradle/demo
```

Use `--policy dream` for Dream-1. One trial per process. Offline checks, no simulator:
`python3 -m unittest discover -s tests -v`, then `python3 tools/offline/check_task.py tasks/<name> --require-json --strict`.

## Simplifications and what to read on the GPU

Rigid bodies. The coiled cord is left out: a real cord adds a tether and drag, so this task isolates the seating, the quarter turn and the switch press, not cord handling. The handset is two square-edged pads and a round handle; the phone is a slab with
two vertical-walled cups and the hook switch, no keypad or display. The hook switch is a 200 N/m spring plunger with 8 mm of
travel (weaker than the handset's 1.67 N weight over its whole travel). Friction 0.45 on the handset and 1.0 on the phone's
feet are estimates. The offline checker's carry-and-drop stand-in (`task.py`, `OfflineModel`) predicted 1.0 before the GPU runs, which
are in Results in simulation below. Things to read in `TASK_TRACE` on the GPU: that the phone and handset settle at start
(`DEMO_HOLD_ONLY=1`: the handset must not creep, the plunger must stay at 0); `ear_dz_mm` and `mouth_dz_mm` (0 to 3 at the end);
`ear_over_cup_mm` and `mouth_over_cup_mm` (about -6 centred, must stay under 0.5); `handset_yaw_vs_phone_deg` after the release
(the place yaw is trimmed 1.2 degrees clockwise for the usual counter-clockwise turn of a carried part: change `YAW_TRIM` in
`task.py` if it is more than 2 degrees from 0); `hook_mm` (about 4 at the end); `phone_move_mm` (under 3 all trial). The demo,
not the scoring, is tuned on the GPU.

## Real build and sources

A desk phone with a two-cup cradle and a spring hook switch, or a 3D-printed or wood stand-in with the same cups and a
spring plunger; a handset or a printed handset of the stated shape; a flat monitor and a pen pot standing where drawn. The
occupations are General Office Clerks (SOC 43-9061) and Receptionists and Information Clerks (SOC 43-4171), US Bureau of Labor
Statistics Occupational Outlook Handbook. All sizes are design values for a corded desk phone (handsets are typically 190 to
210 mm long with a handle about 30 to 35 mm across; bases about 0.7 to 1.2 kg), not measurements of one product: measure your
phone and keep the 8 mm and 6 mm clearances round each pad and the mouthpiece pad longer than the earpiece cup. No third-party assets or
meshes: every part is a box or cylinder.

## Files

`task.json` (submission text), `task.py` (objects, hooks, demo, offline stand-in), `layout.py` (every dimension),
`scoring.py` (the scorer), `mechkit.py` and `rigstate.py` (the program's compound-body and arm-reading helpers, unchanged),
`setup_drawing.svg` (dimensioned drawing), `tests/test_scoring.py` (offline tests: every tier, boundary, latch, NaN, reset and
exploit).

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control, 1,200-step limit), one trial each. The settle run
(`DEMO_HOLD_ONLY=1`, nothing commanded) scored 0.0.

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
