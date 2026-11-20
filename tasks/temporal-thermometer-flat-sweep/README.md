# Take a manikin's temperature with an Exergen temporal artery thermometer

An adult care-training manikin head stands upright on the table, facing the robot, on a sprung neck: push its
forehead and the head nods back, 0.5 N per mm at the scan line. An Exergen TemporalScanner TAT-5000S (datasheet size
and mass: 5 x 20 x 3 cm, 213 g) stands on its long edge in a foam rest to the robot's right of the head. The robot
must take the scanner out of the rest, hold its SCAN button in, put the sensing cup flush on the centre of the
forehead, slide it round the forehead's curve toward the hairline lightly and at the maker's pace, lift it off, and
stand the scanner back in its rest with both hands off.

Who does it: nursing assistants (O*NET 31-1131.00, 1,505,900 US jobs in 2025, BLS figures on O*NET; task line
"Record vital signs, such as temperature, blood pressure, pulse, or respiration rate, as directed by medical or nursing
staff") and licensed practical nurses (29-2061.00, 666,900; "Provide basic patient care or treatments, such as taking
temperatures or blood pressures"). The TAT-5000S manual names its intended users: "physicians, nurses, and nursing
assistants". The maker's sales aid gives the steps this task follows: "Place the probe flush against the center of
the forehead", "Press and hold the SCAN button throughout the entire measurement", "Slowly slide the probe along the
forehead midline to the hairline over approximately 2-3 seconds", "Release the SCAN button and read the displayed
temperature"; and "Scanning down the side of the face may result in lower readings". It also lists a brief touch
behind the earlobe; the in-service sheet says that touch is "to assure the correct reading if the patient is sweaty".
This task scores the forehead scan only (the manikin is dry); see Simplifications.

## Why it is hard

1. **The hand that carries the scanner also holds SCAN.** The button is on the front of the body, 79 mm below the
   cup's centre. The pads must close across it and the back of the body, and keep it at least 1.5 mm of its 2 mm in
   for the whole slide. A grip anywhere else on the body carries the scanner but never presses it.
2. **The forehead curves.** In top view it is an arc of radius 75 mm; from its centre to the scored end (50 mm of arc)
   it turns 38 degrees. The cup must stay within 10 degrees of the skin's normal, so the wrist turns as the cup slides,
   within 8 mm of the mid-forehead line (not down toward the brows or the side of the face).
3. **Light and steady.** The head must be pushed back 1 to 4 mm at the scan line (0.5 to 2.0 N) on every step, and the
   45 mm timed stretch must take 1.5 to 4.0 s (11 to 30 mm/s). A position-stiff arm pressed into the forehead shoves the
   head (8 mm, 4 N, latched zero); a cup that drifts off the skin leaves the band.
4. **Back in its rest, hands off.** The scanner must stand in the two slots again (within 10 mm, 4 mm in height, on its
   edge, turned within 15 degrees), both hands 30 mm clear, still for 1 s, with the head back upright.

## Scoring

s is arc length round the forehead from its centre toward the patient's left temple (the robot's right). Points on
the head are read in the head's own frame, so the nod does not move the line.

| Condition | Rule |
|---|---|
| Touching | cup's centre within 2 mm of the skin and within 8 mm of the scan line (170 mm up); cup facing within 10 degrees of the skin's normal; SCAN in 1.5 mm or more; head pushed back 1 to 4 mm at the scan line; both grippers' finger zones 20 mm or more from the head (the cup, not a hand, presses it) |
| Pass | starts on a touching step with s from -10 to +5 mm; ends at the first step with s 50 mm or more; every step between: on the line (8 mm), facing within 10 degrees, SCAN held, cup's centre no more than 6 mm off the skin, s from -10 to 60 mm, never back more than 3 mm |
| Clean | a pass touching on every step, 1.5 to 4.0 s from s 5 mm to s 50 mm |
| Put back | cup's centre within 10 mm (top view) and 4 mm (height) of where it stood; on its edge within 10 degrees; turned within 15 degrees |
| Let go | every step of the last 1 s: put back, both finger zones 30 mm or more from the scanner, scanner under 1 cm/s and 0.5 rad/s, head within 1 mm of upright and under 0.02 rad/s |
| Shoved (latched) | the head ever pushed back 8 mm or more (4 N) |

| Outcome | Score |
|---|---|
| Clean pass and let go | 1.0 |
| Clean pass, not let go | 0.75 |
| A pass that was not clean (contact broke, left the band, too fast, too slow) | 0.5 |
| Touching for 0.5 s in all, no pass | 0.25 |
| Anything else, doing nothing, or shoved | 0 |

Success: 1.0 held for 1 s. A missing or non-finite state scores 0 on that step and restarts the pass and the 1 s
window. A jump back to the starting layout (a reset) starts the episode's latches afresh.

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Push band and shove | dial indicator behind the head at the scan line (marks at 1, 4 and 8 mm) |
| Flush (10 degrees) | top-view and side-view video against a 10 degree gauge |
| Line and arc | washable ink dots 8 mm above and below the scan line at 0, 5, 50 and 60 mm of arc; the skin the cup slides on stays bare |
| Speed | stopwatch between the 5 and 50 mm dots |
| SCAN held | the scanner's clicking on the video's sound: the manual says it scans "as long as the button is depressed" and clicks while it scans |
| Hands off the head (20 mm) | a 20 mm spacer between each hand and the head on video |
| Put back | outline marks on the rest |
| Hands off | a 30 mm spacer against each hand; video of the last 1 s |

## Results

The offline column comes from the offline demo check with this task's stand-in for the physics (`task.OfflineModel`:
the scanner carried by a hand closed across it, SCAN pressed when the pads close on the button, the cup stopped by the
skin and the head nodding through the arm's sideways give and the neck spring in series). It is not a measurement:
it is a prediction. The GPU scores of this version are in the last column.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run yet |
| Scripted demo | below (right arm only) | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` | the same demo, left arm idle (it has no part in the plan) | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=right` | the right arm idle: nothing is done | 0.0 | 0.0 |
| `DEMO_VARIANT=no_button` | gripped 40 mm further down, on the display: SCAN never pressed | 0.0 | not run |
| `DEMO_VARIANT=fast` | the slide in 0.9 s | 0.5 | not run |
| `DEMO_VARIANT=light` | cup commanded 1 mm into the skin (out of the band) | 0.0 | not run |
| `DEMO_VARIANT=hard` | cup commanded 18 mm into the skin (head shoved) | 0.0 | not run |
| `DEMO_VARIANT=no_follow` | slid round the curve without turning the scanner | 0.25 | not run |
| `DEMO_VARIANT=low` | slid 12 mm below the scan line | 0.0 | not run |
| `DEMO_VARIANT=keep_probe` | never put back | 0.75 | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

The task needs only one arm: the instruction is a one-handed job in real practice. `DEMO_IDLE_ARM=right` leaves no plan
(score 0); idling the left arm changes nothing.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right gripper closes
across the body at the SCAN button; the left arm holds its ready pose.

| Time (s) | Right gripper |
|---|---|
| 0 to 6.5 | over the scanner, down beside it in the gap between the rest's blocks, close across the button and the back (SCAN pressed) |
| 6.5 to 10 | lift it out of the slots, carry it in front of the forehead centre, cup 30 mm off the skin, body turned so the cup faces the head |
| 10 to 12.8 | cup onto the forehead centre, commanded 6.5 mm in and 4 mm above the line (on the GPU 4.5 mm nodded the head only 0.6 to 2.2 mm, and the scanner rides about 6 mm low), settle |
| 12.8 to 15.8 | slide round the curve to 54 mm of arc in 3.0 s, the wrist turning 41.5 degrees with it (the timed 45 mm takes about 2.4 s) |
| 15.8 to 18.1 | hold, lift the cup straight off the skin, back away from the head |
| 18.1 to 24.9 | carry it back over the rest, lower it into the slots, settle |
| 24.9 to 34.9 | open slowly (SCAN comes out), hold, straight up, aside, hands off and still |

Offline: worst IK 0.09 mm; right joint 4 margin 4.4 degrees, joint 6 margin 16 degrees; the gripper's body stays 5 mm
clear of the head's real shapes during the slide, and its finger zones (the 20 mm hands-off rule) 36 mm or more. The offline check (`--require-json --strict --one-arm`) passes, predicted
1.0, held from step 638 (demo 698 steps, 34.9 s; run limit 1200). Along the slide the offline model gives 2.3 to 2.8 mm
of push and the cup within 1.1 degrees of flush. GPU risks: the scanner may turn in the grip (the skin's push acts
79 mm from the pads), and the gripper's clearance to the head is small.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Seed: nothing is random; `object_jitter=0` fixes the start layout. For Dream-1 use `--policy dream`.
`TASK_TRACE=trace.jsonl` writes one line per control step: s, gap, push and force, flush and tilt, button travel, the
touching flags, put back, the hands' gap.

## Simplifications

- Rigid bodies. The head is built from cylinders and boxes to a 75 mm forehead radius (a design value, not a scan);
  the skin does not give: the head nods on a sprung neck hinge instead (0.5 N per mm at the scan line, estimated), so
  pressing too hard or too lightly shows as a nod outside the band.
- The scanner's outer size and mass are the datasheet's; its shape, the 30 degree cup angle, the button's place,
  travel and force are estimates from the maker's photos.
- No temperature is simulated: a scan counts from pose, force, speed and the button.
- The behind-the-ear touch the maker adds is not scored: the manikin is dry, and the in-service sheet gives that
  touch for a sweaty patient.
- Friction 0.5 on the scanner, 0.45 on skin and foam (estimated).

## Sources

- Nursing assistants: https://www.onetonline.org/link/summary/31-1131.00 (task line; 1,505,900 jobs, 2025)
- Licensed practical nurses: https://www.onetonline.org/link/summary/29-2061.00 (task line; 666,900 jobs, 2025)
- TAT-5000S sales aid (flush on the centre, SCAN held, slide to the hairline over 2 to 3 seconds, release and read,
  side of the face reads low, behind the ear):
  https://www.exergen.com/wp-content/uploads/2026/08/TAT-5000S_Sales_Aid-draft-850603-r1-1-1.pdf
- TemporalScanner in-service sheet (the touch behind the ear is for a sweaty patient; cleaning with alcohol):
  https://www.exergen.com/wp-content/uploads/2024/09/Exergen-TemporalScanner-Important-Points-for-Inservicing.pdf
- TAT-5000S instructions for use (intended users: physicians, nurses and nursing assistants; the red button; scans while
  it is held, clicking):
  https://www.exergen.com/wp-content/uploads/2023/09/818694r1-TA5KS-US-IFU-NoCE-111122.pdf
- TAT-5000S datasheet (5 x 20 x 3 cm, 213 g):
  https://www.exergen.com/wp-content/uploads/2026/04/818063r1-sht2-TAT-5000S-EC-Product-Datasheet.pdf

## Real build and reset

An adult care-training manikin head whose forehead fits a 75 mm radius gauge at the scan line, its neck bolted to a
steel hinge with a torsion spring on a small base, set with a scale and a dial indicator at the scan line to 0.5 N per
mm; a TemporalScanner TAT-5000S; a two-block foam rest with slots; a
washable ink dots beside the scan line for the bench checks (never on the scan path). Reset: the scanner back on its edge in the slots, the head
upright (dial at zero), the cup and forehead wiped with an alcohol pad. Nothing is consumed.
