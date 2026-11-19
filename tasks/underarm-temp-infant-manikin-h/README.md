# Take a baby's temperature under its arm with a digital stick thermometer

An infant care manikin of newborn size (52 cm, 2.3 kg, soft vinyl, jointed arms, the kind used in parenting and baby
care classes) lies on its back on a blue changing mat, head to the robot's left, so its right arm lies along its side
nearest the robot. It has a newborn's proportions: a head 120 mm across (about a quarter of its length), short
limbs and no painted face; in the render the head, body and limbs are simple stacked, rounded and box shapes. A white digital stick
thermometer, switched off and display up, lies on the mat beside the manikin's right hand. A thin tan force-sensor
disc is taped on the skin of the armpit, under the upper arm. The manikin's shoulder joint swings the arm forward and
up, so the armpit is opened by raising the arm. The instruction (task.json) is:

> Take the infant care manikin's temperature under its right arm with the digital stick thermometer, leave the reading on show with the thermometer display up on the mat, put the arm back at its side and switch the thermometer off.

This is what a parent or carer does with a feverish baby, step by step as the thermometer's manual says:

1. Press On/Off. The unit runs a self-test and is ready about 2 s later.
2. Lift the baby's arm forward and up off its side. This opens the armpit.
3. Put the tip in the armpit so that it touches the skin.
4. Lower the arm onto the thermometer. The arm then holds it in place.
5. Wait for the peak: "usually 11 seconds" underarm. Any break in the contact restarts the count.
6. Lift the arm, take the thermometer out and lay it display up, so the reading can be read.
7. Put the arm back at the baby's side.
8. Press On/Off again. The display goes blank.

## Why it is hard

- **The arm falls back.** The manikin's shoulder is a disc joint that turns about its left-right axis. Its
  friction (est 0.02 N m) is less than the arm's weight torque (about 0.07 N m when the arm is flat). Let go below
  about 70 deg, the arm sinks back onto the manikin's side, or onto whatever is under it. Something has to hold the
  arm up while the tip goes in.
- **The armpit is closed until the arm is up.** At rest the upper arm's inner face is 3.5 mm from the armpit sensor
  film. The tip is 4 mm across. The stick cannot be pushed in from outside with the arm down.
- **The stick is light and loose.** It weighs about 15 g and is 22 mm wide. A pinch holds no torque about its own
  closing axis, so a stick lifted by its tail swings out of the fingers. The demo carries it at its centre of mass
  and slides it by its tail only while it lies on the mat.
- **The arm has to come down onto the stick, not beside it.** The tip must stay in contact with the sensor disc
  (0.5 mm or less) on every step of the 11 s count, with the arm resting on the stick (0.5 mm or less) at 22 deg or
  less.
  It rests about 13 to 15 deg up (13.2 to 13.4 deg in the GPU demo). An arm dropped hard or off to one side pushes the stick
  away from the film. The count then restarts, or never finishes.
- **The device's own rules.** A press switches the unit on or off. A 5 s hold opens the C/F setting instead and does
  not switch. The unit turns itself off 90 s after a reading. "Off" only counts when the button switched it off after
  the reading was shown.
- **Handle the baby safely.** Taking the manikin off the mat, lifting it more than 30 mm or rolling it over more than
  45 deg scores 0. Shifting it a little on the mat does not (the armpit sensor moves with it).

## Scoring

The scorer reads every 50 ms step (`scoring.py`): the manikin's pose and its shoulder angle, the thermometer's pose
and button travel, the mat's pose and both grippers. `SCORE_OVER = "end"`: the score is the final step's score. Success
is 1.0 held for 1 s.

| Score | When |
|---|---|
| 1.0 | Reading, Shown, Arm back and Off, with both hands off and everything still over the last 1 s |
| 0.75 | Reading, Shown and Arm back, but not all of the rest (left on, off by itself, or still touched) |
| 0.5 | a Reading (not shown for 2 s, arm not back, or switched off before Shown) |
| 0.25 | Closed at some step, but no Reading |
| 0 | otherwise; and for the whole run after unsafe handling (the manikin off the mat, lifted more than 30 mm or rolled over more than 45 deg) or if any part of the thermometer was ever 5 mm or more below the mat top |

| Term | Simulation rule | Bench check on a real trial |
|---|---|---|
| Switching | button in 0.3 mm or more, released within 5 s; a 5 s hold opens the setting instead | video of each press; the display lights up or goes blank |
| Ready | 2 s after switching on | the self-test flash ends |
| Closed | on and ready; the tip's front 8 mm within 0.5 mm of the sensor disc (contact); the thermometer within 0.5 mm of the manikin's right arm (the arm resting on it); the shoulder at 22 deg or less | FSR 402 calibrated with a 20 g weight reads 0.2 N or more; side video: the hand's centre 76 mm or less above the mat; top video: the upper arm resting on the stick |
| Reading | Closed on every step for 11.0 s unbroken (a break restarts the count) | the sensor log shows 0.2 N or more for 11.0 s unbroken; the display shows Lo (the unit's completed reading below 32.0 C) or a temperature (Err does not count) |
| Shown | after the Reading, while on: the thermometer on the mat (lowest point within 4 mm of the mat top), tip 30 mm or more from the disc, no part within 10 mm of the manikin's arm, display within 30 deg of straight up, every finger of both grippers 30 mm or more from it, held 2.0 s | video: the display face up for 2 s, hands away (a 30 mm block) |
| Arm back | shoulder at 5 deg or less | side video: the hand's centre 30 mm or less above the mat |
| Off | switched off with its button after Shown, and off now | video of the press; the display goes blank |
| Hands off | every finger of both grippers 30 mm or more from manikin and thermometer on every step of the last 1 s | a 30 mm block |
| Still | the thermometer, the manikin and its hand each moved 5 mm or less over the last 1 s | video |
| Zero (unsafe handling) | the manikin's back midline point off the mat top, more than 30 mm over it, or the manikin rolled more than 45 deg from lying on its back; thermometer 5 mm or more below the mat top | the mat's edge; a 30 mm block by its back; side video |

Why these thresholds:

- **0.5 mm touch, on every step of the count.** The manual says the tip must touch the skin, so nearness earns
  nothing. The engine's rest offset is 0, so surfaces resting on each other read 0 to 0.25 mm in the scorer (GPU
  demo: tip 0.02 mm after the slide and 0.10 to 0.22 mm through the count, arm 0.00 mm). 0.5 mm allows for that and no more: a tip 1 mm short of the film, or an arm
  hovering 1.5 mm over the stick, never counts (`tests`, `test_no_reading_without_contact`). There is no contact
  sensor in the simulation, so the rule is geometric; on the bench it is the FSR 402 reading 0.2 N or more.
- **22 deg.** The arm rests about 15 deg up on the stick in the model (`parts.support_angle`) and 13.2 to 13.4 deg in the GPU
  demo. 22 deg allows for the arm's give and settling. Lifted to 30 deg it is clearly off the stick, and the tip is no
  longer held.
- **11 s, 5 s, 2 s and 90 s** are the manual's figures.
- **30 mm hands off** is the review bar for "let go".
- **Unsafe handling only.** A carer may shift a baby a little on the mat; that is not a fault, and the armpit sensor
  moves with the manikin. Taking it off the mat, lifting it more than 30 mm or rolling it more than 45 deg is.

The real value on the display is not scored. The manikin is at room temperature, so the real thermometer shows Lo, the
unit's completed reading below 32.0 C. The rule only checks that a full reading was taken.

## Mechanisms and parts (H2)

- **Infant care manikin** (`parts.py`, the doll in the code; mechkit articulation, loose, 2.2 kg body plus a 0.10 kg
  arm, friction 0.5): one rigid body for the head, torso, diaper, legs, left arm and sensor film, plus the right arm
  as one rigid link on a revolute at the shoulder. The revolute turns about the manikin's left-right axis (world x) at
  y 135, 20.5 mm above the mat. Its range is 0 to 180 deg, friction 0.02 N m, damping 0.005 N m s, no spring. 0 deg is
  the arm lying along the body on its stop. The joint swings the arm forward and up, so raising the arm opens the
  armpit. The real manikin's arm is also one moulded piece on a disc joint about that axis. The upper arm is 35 mm
  thick, the forearm 30 mm, the hand 26 mm; the forearm angles 8 deg out at the elbow. The body lies flat on its back
  on the mat: no shape goes under its back (tested). Newborn proportions: the head a ball 120 mm across (twelve
  stacked slices), crown at y 284, soles at y -238 (522 mm); the legs are capsules (thigh radius 22 mm, shin 19, a
  rounded knee, a rounded foot toes up), the left hand and elbow are rounded; no face is drawn. These shapes are look
  only and sit off every path: the head, the far leg, the knees, shins, feet and the left arm stay 29.5 mm or more
  from both arms' collision capsules, fingers included, through the whole plan; the near thigh keeps the old leg's
  radius and line (its near side at x 254), so its clearance to the left hand's pinch is unchanged. The torso, its
  armpit side, the sensor film and the right arm are unchanged. Mass is unchanged; the body's centre of mass moved
  12.7 mm toward the head and 2.2 mm up (base frame (4.2, 68.2, 38.3) mm, was (3.8, 55.4, 36.2)).
- **Armpit sensor:** an Interlink FSR 402 disc (14.68 mm, 0.46 mm thick) on the torso's side at x 238, centred at
  y 90 (45 mm toward the feet from the shoulder axis), 10 mm above the mat. Its face points at the robot. Its tail
  runs down the torso side to the mat. In simulation it is part of the manikin's body.
- **Thermometer** (mechkit articulation, loose, 15 g in all, friction 0.4): a 114 mm white stick, 22 mm wide and
  13 mm thick, tapering over its front 30 mm to a 4 mm metal tip. Two links:
  - The grey-blue On/Off button, 10 mm across and 28 mm from the tail: a slider with 0.8 mm of travel and a
    2500 N/m spring (2 N at full travel).
  - The display face: a slider under a 22 x 10 mm window (36 to 58 mm from the tail). The device rule moves it:
    the dark blank half sits under the window when the unit is off; the lit green half, with "Lo" in black, slides
    under the window when it is on. It has a soft 2 N/m spring so that its push never moves the stick.
- **Device rule** (`scoring.Unit`): switching, self-test, 11 s peak and auto-off, all from the manual. The count runs
  only while Closed (contact) holds on every step. `task._display` (a mechkit on_step callback) sets the display
  face's target from the rule every step.
- **Mat:** loose, 460 x 600 x 15 mm, 0.6 kg, friction 0.8 on the table.

## Demo

The demo is open loop: timed waypoints from the known start layout (`task.plans()`). Both grippers point straight
down. The right arm handles the thermometer. The left arm holds the manikin's arm, pinching the hand across x, so the
pinch's closing axis is parallel to the shoulder axis and the arm turns freely in it. The two arms reach across each
other's side, so they take turns in the shared space next to the armpit. Times are approximate.

| Time (s) | Right arm (thermometer) | Left arm (manikin's arm) |
|---|---|---|
| 0 to 5 | presses On/Off with closed fingers | waits at home |
| 5 to 22 | grips the stick at its centre of mass, carries it to the manikin's right side, lays it along x 45 mm short of the armpit, lets go, backs off behind its tail and goes home | waits at home |
| 22 to 30 | at home | pinches the hand and lifts the arm in an arc to 40 deg |
| 30 to 48 | comes in low from the robot's right, grips near the tail, slides the stick 45 mm along the mat until the tip touches the film, lets go and leaves low | holds the arm at 40 deg |
| 48 to 56 | at home | lowers the arm onto the stick (it rests about 13 deg up on the GPU), lets go, goes home |
| 49 to 60 | | the 11 s count runs (Closed) |
| 60 to 68 | at home | once the count is over, pinches the hand again and lifts the arm to 40 deg |
| 68 to 85 | comes in, grips near the tail, slides the stick out 45 mm, lets go, leaves; the stick lies display up out of the armpit, so Shown completes its 2 s at about 82.7 s | holds the arm at 40 deg (about 37 on the GPU) |
| 85 to 93 | at home | lowers the arm to the manikin's side, lets go, goes home |
| 93 to 115 | grips the stick at its centre of mass, lays it back where it started, display up, lifts well clear, waits 2.8 s (Shown), presses On/Off (Off) | at home |
| 115 to 121 | goes home; success holds from 117.5 s | at home |

Why 40 deg: higher, the left wrist comes down over the right's slide; lower, the stick has less room under the arm.
The arms' collision capsules stay apart through the whole plan (tested in `tests/test_scoring.py`, `Reach`). The left
pinches the hand 6 mm above its centre, so that the gripper's housing clears the manikin's diaper and legs by 9 mm.

Two hands are the natural way to do it: one holds the arm up at about 40 deg while the other slides the stick in, as
a carer does. The model's joint friction does hold the arm above about 70 deg, so a single hand could first raise it
that high and let go; the demo does not do that. With either arm idle the scripted demo scores 0.0, offline and on
the GPU.

## Running it

```
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
```

- `DEMO_HOLD_ONLY=1`: no plan (settle check).
- `DEMO_IDLE_ARM=left` or `right`: one arm idle.
- `TASK_TRACE=<file>`: one JSON line per control step (score, shoulder angle, tip gap, arm gap, unit state, poses).

Run limit: 3200 steps (160 s at 20 Hz). The layout is fixed (`object_jitter=0`); there is no random seed.

## Results

GPU run of this version (Isaac Sim, `object_jitter=0`, 3200-step limit, 19 Nov 2026): the manikin, the 0.5 mm
contact rule for Closed and the unsafe-handling zero exactly as described above. The screenshot and videos are of this
run. The scorer ran live on every step: the 11 s count ran unbroken from about step 973 to the reading near step 1192.
The torso moved at most 0.03 mm and turned 0.02 deg, and the manikin stayed flat on the mat (lift -0.01 mm, roll
0 deg), so the unsafe-handling zero never applied.

| Run | GPU | Offline prediction |
|---|---|---|
| Settle (no plan, 160 steps) | 0.0, nothing moved | 0.0 |
| Scripted demo | 1.0, success held from step 2350 (117.5 s) of the 3200-step limit | 1.0, success held from step 2350 |
| Left arm idle | 0.0 | 0.0 |
| Right arm idle | 0.0 | 0.0 |
| Dream-1, one trial at the full limit | 0.0 | |
| Demo without the Off press | | 0.75 |
| Demo without the slide into the armpit | | 0.0 |
| Demo without the On press | | 0.0 |

The offline figures come from `tools/offline/check_task.py` and `offline_model.py`: rigid kinematics with no forces,
friction or arm give, so they are predictions of the plan's geometry, not measurements. Two GPU demo runs of
earlier versions scored 0.0 and 0.25; the fixes were in the scripted demo (the right's way out backs behind the tail) and in one
collider (the diaper's rounded part stuck 2 mm under the manikin's back, so the manikin rocked when the arm came down
on the stick).

## Simplifications (fidelity)

- Rigid bodies, 20 Hz control. The manikin is one rigid body; its right arm is one rigid piece. Vinyl does not give.
- No temperature is simulated. The display is a drawn face that the device rule slides under the window.
- No beeps. The 11 s count is the declared rule.
- The FSR film is part of the manikin's body in simulation; its disc is the named contact patch.

## Measure before a real trial

- The manikin's shoulder friction and the arm's mass and centre of mass: tilt the arm and find the angle where it
  stays.
  If the real joint holds the arm up at 40 deg, the second hand is natural but not needed.
- The head, torso, arm and hand sizes, and the 3.5 mm gap between the upper arm's inner face and the armpit sensor
  film at rest.
- The thermometer's width, tip, mass, button position, travel and force (a kitchen scale under the button).
- Friction by incline test: the manikin's vinyl on the mat, the thermometer on the mat.
- The FSR 402 calibration with a 20 g weight, and its reading with the arm resting on the stick.

## GPU results

From the traces of the GPU run of this version (19 Nov 2026):

- Settle: nothing moved; the manikin lay flat at z 15.0 mm, the torso moved 0 and the thermometer stayed still.
- The unit was switched on by its button at 4.05 s and ready 2 s later.
- The right laid the stick down 45 mm short of the armpit square to it (yaw 0.5 deg). The left lifted the arm to
  about 40.5 deg and held it there.
- The slide put the tip on the film (tip gap 0.02 mm, step 842), and it stayed on after the right let go.
- The arm came down onto the stick and rested 13.2 to 13.4 deg up; the stick tipped about 5 deg nose down (tail up
  about 7 mm). Closed held unbroken from about step 973 (tip gap 0.10 to 0.22 mm, arm gap 0.00 mm on every sampled
  step) and the reading was taken 11 s later, near step 1192.
- The second lift held the arm at about 36.7 deg. After the slide out the stick lay display up (0 deg) out of the
  armpit (tip 46 mm from the disc) and Shown completed its 2 s at about 82.7 s; the arm was laid back at its side
  (0 deg) by about 85.6 s.
- The right laid the stick back near its start (tail 3 mm from its start mark, yaw 1.1 deg) and switched the unit
  off by its button at 115.15 s. Score 1.0 from about step 2330, success held from step 2350 (117.5 s).
- Over the whole demo the torso moved at most 0.03 mm and turned 0.02 deg, the manikin did not lift (-0.01 mm) or
  roll (0 deg); at the end both hands were off and everything was still.
- With either arm idle Closed never happened (0.0): with the left idle nobody held the arm up (it rose at most
  10.8 deg) and the tip never touched the film, and with the right idle the unit was never switched on and the stick
  never moved. Dream-1 scored 0.0 at the full 3200-step limit; it
  never switched the unit on or reached Closed.

## Sources

- Vicks SpeedRead V912US manual (operation, 11 s underarm peak, Lo below 89.6 F, auto-off after about 1.5 min):
  https://manualslib.mx/manual/375599/Vicks-Speedread-V912Us.html
- Thermometer size, about 114 mm long and 13 mm thick: manuals.plus listing (ASIN B000GRXHIE)
- Nasco newborn baby care model SB17159, 21 in (53 cm), 5 lb (2.3 kg): https://mms.mckesson.com/product/1209872
- The same line, "soft vinyl body with jointed, posable arms and legs":
  https://www.schoolspecialty.com/newborn-baby-doll-baby-girl-with-medium-brown-skin-tone-2218338
- Interlink FSR 402 (14.68 mm active disc, 0.46 mm thick, 0.2 to 20 N): Interlink FSR 400 series datasheet
- Estimates (stated in the setup and fidelity): the manikin's body and arm sizes, joint friction and arm mass; the
  thermometer's width, tip, mass, button and window; the mat's size; all friction values.
