# Widen tongue-and-groove pliers to the last groove: open wide, slide, close

A loose pair of tongue-and-groove pliers lies flat on the bench, set to its narrowest jaw (groove 1 of 5), handles
30 degrees apart. The robot resets it to the widest jaw: open the handles past the angle at which the tongue
leaves its groove, carry the joint 24 mm along the slot toward the handles, close until the tongue drops into
groove 5, and leave the pliers flat with nothing touching them.

## The tool

Channellock 420 straight-jaw pliers, 9.5 in, blue dipped handles. Maker's page: 241.3 mm long, 0.47 in joint,
1.57 in (39.9 mm) handle span, 0.87 lb (394.6 g); retailers list 5 positions. Ours: 242 mm long, 391 g, a 12 mm
joint, handles 40 mm across their outer edges when closed in groove 1.

Laid out from a photograph of the maker's 12 in pliers of the same pattern (Wikimedia Commons,
"Channellock tongue-and-groove pliers.jpg", by user Typhoon, CC BY-SA 3.0; proportions only); the joint follows the
usual design (US 7,017,458; Wikipedia, "Tongue-and-groove pliers"). The two halves cross at the rivet. The slotted half
carries the slot, the five arc grooves and the outer jaw; the rivet half, on top, carries the rivet, the tongue and the
inner jaw. The slot leans 32 degrees off the handle line and the jaws stand ahead of the rivet, 58 degrees to that
line. Groove 1, the narrowest jaw, is the slot's end nearest the jaws; sliding the rivet back toward the handles opens
the jaws, and groove 5, the widest, is the slot's end nearest the handles.

## Why it is hard

1. **The obvious push does nothing.** At the start the tongue is in groove 1: pushing the rivet half along the slot
   only drags the whole loose tool.
2. **The window is narrow and at the end of travel.** The joint slides only while the handles are 72 to 85 degrees
   apart. Closing below 72 too early drops the tongue into the nearest groove, 1 to 4.
3. **A loose tool on a long lever.** The rivet half is swung about the rivet from a grip 162 mm out while the slotted
   half lies loose on the bench, and the slide pulls the joint toward the robot. A hand that strays off the arc drags
   the whole tool.
4. **A clean finish.** Locked, both halves flat, both grippers 25 mm clear, still for a full second.

## Scoring

| Condition | Rule |
|---|---|
| Groove n | the rivet's centre within 1.5 mm of that groove's centre along the slot |
| Locked | slotted half held, a 5 N push on the rivet half along the slot moves the rivet under 1 mm (handles closed below the 72 degree release angle) |
| At rest | both halves flat (tilt under 3 degrees, lowest point within 2 mm of the bench top); through the last 1 s both grippers 25 mm clear of the pliers (from the fingers' full sweep about the gripper's axis), neither half nor the rivet moving more than 1 mm, neither half turning more than 1 degree |

| Outcome | Score |
|---|---|
| Locked in groove 5 and at rest | 1.0 |
| Locked in groove 2, 3 or 4 and at rest | 0.5 |
| The rivet slid 3 mm or more out of groove 1 with the handles open (72 degrees or more) | 0.2 |
| Anything else, or any part of the pliers 5 cm or more below the bench top at the end | 0 |

Doing nothing scores 0.

## Results in simulation

L40S, Isaac Lab-Arena, `-E object_jitter=0 -E control_hz=20`, one trial each, run limit 1,200 steps (60 s).

| Run | What it does | Score |
|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, 160 steps) | both arms hold still | 0.0 |
| Scripted demo | both arms, the phases below | 1.0 |
| Right arm alone (`DEMO_IDLE_ARM=left`) | the same plan without the left arm | 0.5 |
| Left arm alone (`DEMO_IDLE_ARM=right`) | the left grips the slotted handle, then lets go | 0.0 |
| Dream-1 | the task's instruction, full run limit | 0.0 |
| Push without opening (`DEMO_VARIANT=naive`) | the right arm alone grips the rivet half at the start and pushes it 45 mm along the slot | 0.0 |
| No slide (`DEMO_VARIANT=no_slide`) | the demo without the slide: handles opened to 80 degrees and closed again | 0.0 |
| Short slide (`DEMO_VARIANT=early_close`) | the demo with the slide commanded 24 mm, not 48, and eased back 6 mm, not 21 | 0.5 |
| Left open (`DEMO_VARIANT=slide_open`) | the demo up to the slide's end, then both grippers let go with the handles still past 72 degrees | 0.2 |

## Demo

Open loop: timed waypoints from the known start layout. Both grippers point straight down and close across a
handle: the right 162 mm out from the rivet on the rivet half's handle, the left 170 mm along the slotted one. The
second arm steadies the slotted half against the pull; without it the pull drags the loose tool and the joint
stops a groove short (right arm alone, table above).

| Time (s) | Right gripper (rivet half) | Left gripper (slotted half) |
|---|---|---|
| 0 to 6.5 | down onto its handle and close | waits clear of the right arm |
| 6.5 to 11.5 | swing the handles open about the rivet, 30 to 80 degrees | wait |
| 11.5 to 18.0 | hold at 80 degrees | down onto the slotted handle and close |
| 18.0 to 26.0 | pull the joint along the slot toward the handles (48 mm commanded for 24 mm of travel: the arms give), dwell 2 s, ease back 21 mm | hold |
| 26.0 to 31.0 | hold at 80 degrees | let go, lift, move aside |
| 31.0 to 37.0 | swing the handles in to 50 degrees: the tongue drops into groove 5 at 72 | home |
| 37.0 to 43.5 | let go, lift, home | |

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

For Dream-1 use `--policy dream`. `DEMO_IDLE_ARM` (left or right) drops one arm's plan, `DEMO_VARIANT` runs a
counterfactual, `TASK_TRACE=trace.jsonl` writes one line per control step.

## Simplifications

- The joint is a slider (0 to 24 mm) and a hinge (handles 0 to 85 degrees). Below 72 degrees a stiff spring,
  set each control step from the handle spread, holds the rivet at the centre of the groove it dropped into;
  at 72 and above it slides against joint friction set to 1.5 N and a light drag (stiff past 22 mm/s). Closing
  between grooves pulls it into the nearest groove, where a real tongue could land on a ridge.
- The two halves do not collide with each other, so the lock is this rule; the tongue is a rib and the grooves
  are the gaps between ridges, and no other part of one half overlaps the other at any spread or slide.
- Forged shapes are boxes: straight handles, flat toothless jaws that close parallel (real ones meet at the tips
  first). One friction value, 0.4, for the pliers on the bench.
- The 72 degree release, the 85 degree stop, the slide force, the 6 mm groove spacing and every size the maker's
  page does not list are our estimates of the 420.

Third-party assets: none (the photograph was measured, not copied).

## Real build and reset

One Channellock 420 as sold, a top and a side camera, a caliper, a 5 N push gauge. Nothing is made, used up,
powered, sharp or hot. Groove: read which groove the tongue sits in. Locked: hold the slotted handle, push the
other half along the slot at 5 N with the gauge and read the rivet's movement with the caliper.

Reset: open the handles wide, slide the joint to the slot's end nearest the jaws (groove 1, the narrowest), close to
30 degrees and lay the pliers flat, rivet half on top, with the rivet at (0.380, 0.000) m, the slotted half's handle
straight back toward the arms and the other 30 degrees to its right. Check the joint will not slide.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The task text; the pliers, the groove lock, `success()`, `score()`, `demo()` |
| `layout.py`, `scoring.py` | Every dimension, threshold and shape; the scorer in plain NumPy |
| `mechkit.py`, `rigstate.py` | Build the pliers as one loose articulation; read both grippers each step |
| `offline_pliers.py`, `tests/` | The pliers without a simulator; offline tests: `python3 -m unittest discover -s tests` |
