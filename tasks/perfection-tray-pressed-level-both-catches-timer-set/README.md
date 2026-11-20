# Reset a Perfection game: wind the timer round, press the pop-up tray down level

**Instruction:** Get the Perfection game ready for the next player: tray down and held by its catches, timer set to
60 seconds and running, hands off.

Category: Societal relatability. Run limit 1,600 steps (80 s at 20 Hz).

## The act

Perfection (Hasbro Gaming) is the family game whose shape tray pops up when its 60 s timer runs out. Before every turn
someone resets it, in the game's own order: switch to STOP, turn the timer dial clockwise to its stop, press the tray
down until it latches, slide the switch to START. The trial starts where the last game ended: tray popped 25 mm proud,
dial at 0, switch still at START (so the run-out cam holds both catches open).

What makes it a robot test:

1. **Order with a visible reason.** At START with the dial run out the cam keeps the catches open: the tray springs
   back up however hard it is pressed. Wound with the switch still at START, the clock runs while the tray is pressed
   and is under 50 s by the time the hands are off.
2. **A level two-hand press.** Four springs push the tray up with about 11.7 N at the catch line, and each catch needs
   about 0.7 N more to ride its pawl out: about 11.6 N in all once the tray's own weight is taken off, against the
   roughly 8 N one arm pushes straight down. Pushed on one side, the tray rocks on its guides (4 degree stop) and only
   that side's barb reaches its tooth. Two hands pressing near the left and right edges at once, about 5.8 N each,
   bring both barbs past both teeth.
3. **Wind and start, then let go.** The dial is turned 300 degrees by its grip bar, more than one wrist turn, so
   it is pinched, turned, released and regripped without knocking the switch; the switch is slid to START; both
   hands clear the game.

The demo: the right arm slides the switch to STOP; both hands press the tray down until both catches hold (at STOP
the cam is off, so they hold with the dial still at 0, and the tray is then out of the way of the hand at the dial);
the right hand pinches the grip bar 4 mm out from the dial's axis and turns its wrist clockwise about that axis, in
three grasps (62, 180 and 58 degrees, the open hand lifted over the bar and turned back between them); then it slides
the switch to START and both arms move aside.

## Results

| Run | Score | Notes |
|---|---|---|
| Offline model, demo | 1.0 | success held from step 829 (41.5 s); 806 demo steps |
| Offline model, left arm idle | 0.1 | the right arm alone: switch, a one-side press (rocks, no catch at the 2 mm overlap bar), dial, START |
| Offline model, right arm idle | 0.1 | the left arm alone presses its side: the cam holds the catches open (switch never moved) |
| GPU demo | 1.0 | job 20261120T000008, 830 steps |
| GPU one arm idle (left / right) | 0.0 / 0.0 | |
| GPU settle (no motion) | 0.0 | nothing moved |
| Dream-1 (full run limit) | 0.0 | |

Wrong ways (DEMO_VARIANT, offline model):

| Variant | What it does | Score |
|---|---|---|
| no_stop | switch left at START: with the dial at 0 the cam holds the catches open, the tray pops back | 0.1 |
| no_wind | dial left at 0: at START the cam opens the catches and the tray pops | 0.25 |
| no_start | latched and wound, clock never started | 0.7 |
| shallow | pressed only to 6 mm above flush: no catch | 0.1 |
| hold_on | all done, right gripper left on the switch | 0.7 |

## Scoring

Caught (a side): the pawl's tooth reaches 2 mm or more in over that side's barb and its underside sits between 0.5 mm
below and 6 mm above the barb's top. Latched: both sides caught. Forced: a barb more than 2 mm into a tooth in both
y and z (the solver let it through; a broken catch in a real trial): 0 for the rest of the trial. Set: the dial shows
50 s or more. Running: over the last 1 s the switch at START on every step and the dial turned back 2 degrees or more.
Released: both finger zones 50 mm or more from the game on every step of the last 1 s.

1.0 latched, set, running, released (held 1 s more); 0.7 latched and set; 0.5 latched; 0.25 one side caught at some
step; 0.1 the tray pressed 10 mm or more down at some step; 0 otherwise. The untouched start scores exactly 0.

## The model

- Scenery: the box lid with the 25 shapes, handles up, behind the unit (fixed, never touched).
- Unit: one fixed articulation. Case (deck 285 x 285 with the tray's opening, four walls, printed dial ring and
  STOP/START marks); tray on a slide down (0 to 30 mm) with a rock hinge about x (plus or minus 4 degrees) carrying the
  plate, 25 drawn wells, two side skirts and the barbs; dial on a hinge about z (0 to 300 degrees); switch knob on a
  slide (0 to 12 mm).
- Catches: a separate fixed part (links of one assembly never collide with each other, and the teeth must meet the
  barbs): two pawls, each on a sprung hinge along x (0.15 N m/rad, stop at 0.30 rad).
- Springs: the slide's drive at 320 N/m, its target held 12 mm above the top stop (the preload). Rock: 0.5 N m/rad.
- Declared rules (per control step): the preload target; the clockwork (at STOP the dial's target is where it is; at
  START the clock keeps its own target, seeded from the dial when the switch arrives and lowered 5 degrees per second
  whatever the dial reads, at most 0.2 rad behind it, so the spring always beats the dial's friction); the switch's two detents (snap within 4 mm); the run-out cam (at START with the
  dial at 0, both pawls driven toward 0.6 rad, against their 0.30 rad stop). Nothing holds the tray down but the barbs
  against the teeth.
- Offline (no simulator) a stand-in model gives the scorer the press (each arm at most 8 N, 1.8 mm/N of give), the rock,
  the catches, the dial turned by closed pads pinching its grip bar (it follows the pads' angle about its axis) or
  pushed round by closed pads beside it, and the switch pushed by closed pads.

## Bench checks for the real trial (estimates to measure)

| Figure | Value used | How to measure on the bought game |
|---|---|---|
| Case and tray sizes | 285 x 285 x 70; tray 230 x 230, popped 25 mm | steel rule and calipers |
| Spring push | 3.8 N popped, 11.7 N at the catch line | spring scale on a flat block at the tray's centre, read at 0 and 24 mm down (a depth gauge on the deck) |
| Catch force | about 0.7 N per side | the scale's reading just before each click, minus the spring reading at that depth |
| Rock | up to 4 degrees | press one edge with the scale, measure each edge's height with the rule |
| Dial | 300 degrees for 60 s, 0.015 N m to turn, 1 s per second at START | protractor ring under the dial; a spring scale on the grip bar 14 mm out; stopwatch |
| Switch | 12 mm travel, two detents | calipers |
| Tray weight | 150 g | kitchen scale (with the springs, by subtraction) |

Real-trial readings: latched is the tray within 2 mm of flush at both side edges with the hands off (rule on the deck);
the dial reading and the switch position from a photo at the end; running from the video (the dial's grip bar moves
over the last second); hands off from the overhead video.

## Run

    inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...

Settle: DEMO_HOLD_ONLY=1. One arm: DEMO_IDLE_ARM=left or right. Wrong ways: DEMO_VARIANT=no_stop, no_wind, no_start,
shallow, hold_on. Trace: TASK_TRACE=file (one JSON line per control step). Seed fixed by the layout (no jitter).
Tests: `python3 -m unittest discover -s tests -q` (NumPy only).
