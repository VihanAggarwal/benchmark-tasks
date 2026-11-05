# Test a tank's air top to bottom before entry, and keep monitoring while the other arm is inside

Permit-required confined-space entry (US OSHA 29 CFR 1910.146) is built around one order of events. Nobody enters
a tank until its air has been tested with a direct-reading gas monitor. Gases layer: hydrogen sulfide and many
solvent vapours sink, and methane rises. So the sampling probe is lowered from the top down and held at each level
in turn (Appendix B, on testing stratified atmospheres in the direction of travel). Monitoring continues for as long
as the entrant is inside (paragraph (d)(5)), and the attendant stays outside the space the whole time (paragraph
(i)(4)).

Here the left gripper is the attendant. It holds an orange probe wand (the monitor's sampling probe on its rigid
extension) and only the probe goes into the tank. The right gripper is the entrant. It reaches into the tank for a
dropped steel part and puts it in a tray. Fetching the part is easy. What is scored, at every control step, is the
order of events. First a complete top-middle-bottom test, with the tip held still for 2 s in each band. Then the
right gripper may go in, but only while the probe is held low in the tank by the left gripper. The left gripper
never goes in. The task needs both arms because the probe must stay in the tank, held, for the whole time the
right gripper is inside.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, mass, position and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the probe's and the part's states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point, opening) for the scorer each control step (unchanged from the program's shared copy) |
| `tests/test_scoring.py` | Offline tests: every tier, every rule, the exploits we tried, resets, the gripper hook, and the demo played through a kinematic proxy |
| `tests/test_demo_plan.py` | Offline checks of the scripted plan against the station's limits (reach, grips, timing, TCP separation) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, a loose box and a loose upright cylinder) and the standard
`task_score` scorer.

The gripper rules need the grippers' state, which `yam_arena` does not pass to a task file's hooks. `rigstate.py`
(imported by `task.py`) wraps the embodiment's step and reset once, at import. Before each control step it takes a
snapshot of both arms (tool point, pad-tip point and finger opening, in the objects' frame), and it starts a new
trial at each reset. It reads only state the simulator already has and changes nothing the embodiment does. If it
cannot read the arms it prints why, once. The gripper rules are then not applied on those steps, and the trace
counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Before the right gripper reaches into the grey tank, test the tank's air with the orange probe held in the left gripper: hold the probe's tip still for 2 seconds just below the rim, then at mid-depth, then just above the floor, in that order. Keep the probe in the lower half of the tank, held by the left gripper, the whole time the right gripper is inside, and never put the left gripper into the tank. With the right gripper, take the steel part out of the tank and put it in the blue tray; then put the probe back in its holder." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tank-entry-probe-first/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Before the right gripper reaches into the grey tank, test the tank's air with the orange probe held in the left gripper: hold the probe's tip still for 2 seconds just below the rim, then at mid-depth, then just above the floor, in that order. Keep the probe in the lower half of the tank, held by the left gripper, the whole time the right gripper is inside, and never put the left gripper into the tank. With the right gripper, take the steel part out of the tank and put it in the blue tray; then put the probe back in its holder." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tank-entry-probe-first/dream
```

Environment switches for the demo:

- `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm check).
- `DEMO_HOLD_ONLY=1` plays no plan (a settling check).
- `TASK_TRACE=trace.jsonl` writes one JSON line per control step with what the scorer saw and decided:
  `test_stage`, `band_done_steps`, `tip_band`, `tip_height_mm`, `left_holds_probe`, `first_entry_step`,
  `max_tip_during_entry_mm`, `min_left_above_rim_mm`, `breaches`, `part_low_mm`, `opening`, both TCPs, and so on.

Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset, so run one trial per process or rely on the reset. The run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (the tests find the offline `yam_arena`
stub by walking up to `tools/offline/yam_stub`).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The probe's tip is the lower end of its axis, worked out
from its centre and orientation, so a tilted probe is measured where its end really is. Heights are measured from
the tank floor (the table top), and the rim is at 80 mm.

**The test.** There are three bands for the tip, which must be over the tank's 120 x 360 mm opening:

- top: 55 to 75 mm, which is 5 to 25 mm below the rim;
- middle: 30 to 50 mm;
- bottom: 5 to 25 mm.

A band counts once the tip has stayed in it, with the probe held by the left gripper, for 2 s (40 consecutive
control steps), with its height, x and y each varying by 10 mm or less over those 2 s. Bands count only in the
order top, middle, bottom. A dwell in a band out of turn counts for nothing. When the bottom band counts, the test
is complete. The test stays valid only while the probe stays in the tank (tip below the rim), held by the left
gripper. If the probe comes out or is let go, a new full test is needed before the right gripper may go in again.

**Holding, letting go and being in the tank.**

- A gripper *holds* an object when its fingers are closed to 47 mm or less (opening 0.5) with the point between
  its pad tips inside the object's bounding box grown by 5 mm.
- It has *let go* when it is open to 84.5 mm or more (opening 0.9), or its fingers are more than 10 mm from the
  object.
- A gripper is *in the tank* when its fingers' lowest point is below the rim, with the point between its pad
  tips within 10 mm of the tank's outline seen from above. For a gripper pointing straight down, the lowest point
  is the pad tips. For a tilted gripper it is lower by the reach of its pad corners times the sine of the tilt.

rigstate gives no gripper heading, so the fingers are taken to be anywhere round the gripper's axis. Rules 1 to 7
apply at every moment, and breaking one scores 0 whatever follows:

1. the right gripper is in the tank while the test is not complete and valid, or while the probe is not held by
   the left gripper with its tip over the opening and less than 40 mm above the floor (the lower half);
2. the left gripper is in the tank;
3. the probe is in the tank (tip below the rim, over the tank's outline) without the left gripper holding it;
4. the probe is out of the tank, held by neither gripper, with its lowest point more than 10 mm above the table
   (dropped, or set down on the rim, a wall, the holder's collar or the tray);
5. the probe tilts more than 45 degrees (it fell over or was laid down);
6. the part stops being in the tank (its centre over the outline and its lowest point below the rim) while the
   right gripper is not holding it;
7. the probe or the part goes off the table.

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Part in the tray, probe standing in its holder, both grippers let go | 1.0 |
| Part in the tray; probe out of the tank, held by the left gripper or standing upright on the table clear of the tank; not all of the 1.0 conditions | 0.75 |
| Part out of the tank (lowest point 10 mm or more above the rim, or clear of the tank's outline), not as above | 0.5 |
| A complete top-middle-bottom test made, part still in the tank | 0.25 |
| Anything else (no complete test) | 0 |

The score is read at the last step. The trial ends early once the 1.0 conditions have held for 3 s with both loose
objects still (under 1 cm/s and 0.5 rad/s). The 3 s is longer than the demo's last release and retreat
(1.5 s + 1.0 s), so a trial cannot end with the left gripper still round the probe. A new trial is recognised from
rigstate's reset hook. As a fallback, it is also recognised when the simulator puts the probe back in its holder and
the part back on its mark in a single step. A simulator state with a non-finite number scores 0 for that step and
leaves the trial's record as it was.

### Why these numbers

The station's tank is a scale model of a small process tank or sump about 2 m deep (roughly 1:25). Each threshold
is set from that scale, from the station's gripper, or from what a person checking the video can see.

| Threshold | Value | Why |
|---|---|---|
| Tank depth | 80 mm | Deep enough for three separate 20 mm bands with 5 mm gaps between them. Shallow enough that a gripper working on the floor has only 63 mm of itself below the rim. |
| Bands | 5-25 mm below the rim, 30-50 mm (half depth +-10 mm), 5-25 mm above the floor | The top, middle and bottom thirds of the depth, as Appendix B's "test as you descend" becomes at this scale: just under the opening, where lighter gases such as methane collect; mid-depth; and just above the floor, where heavier ones such as H2S collect. The bottom band stops 5 mm above the floor, so a probe resting on the floor does not count (a real probe is kept out of sludge). Each band is 20 mm tall, so an arm sagging up to 10 mm under the probe still holds the tip inside a band aimed at its centre. |
| Dwell | 2 s inside a 10 mm box | A real monitor needs a held sampling point: the sample has to travel up the tube and the sensors have to respond, which takes many seconds per level. At this scale and run length that becomes 2 s (40 control steps). The 10 mm box makes it a deliberate stop. A probe swept through a band at more than 5 mm/s never counts. |
| Order | top, then middle, then bottom | The order the procedure requires (testing in the direction of travel). It also matches how the probe descends, so a skipped or reversed level is visible on video. |
| Lower half during entry | tip more than 40 mm below the rim | Monitoring continues where the entrant works, at the floor, where heavy gases collect. This is half the depth, the coarsest rule a video can check. |
| Test voided when the probe comes out | | A pre-entry test says nothing once the monitor has left the space. |
| "In the tank" | fingers below the rim, within 10 mm of the outline | The rim is the boundary of the space. The 10 mm covers fingers straddling the 8 mm wall. |
| Attendant's clearance | (demo: pads 55 mm above the rim) | The attendant stays outside. The probe is long enough (150 mm, gripped 30 mm below its top) that the attendant never needs to break the rim. |
| Hold | closed to 47 mm, pad-tip point within 5 mm of the object | Fingers closed on a 30 mm object read about 30 mm. 47 mm is half the full opening, so a gripper at 47 mm or less with its tips inside the object's outline has the object between its fingers. |
| Let go | open to 84.5 mm, or fingers 10 mm away | As in the program's other gripper tasks: open wider than anything here, or clearly clear of it. |
| Part out | lowest point 10 mm above the rim | A part held just at the rim is still in the tank. |
| Probe dropped | unheld, more than 10 mm up, out of the tank | A monitor that is let go above the table falls, or was left on something. 10 mm allows for a probe set back in its collar from 2 mm up. |
| Probe toppled | more than 45 degrees | A 30 x 150 mm cylinder past about 11 degrees falls over. 45 degrees leaves a probe leaning in its collar (at most 28 degrees) unpunished. |
| Standing / in holder | under 10 degrees, within 5 mm of the table, centre within 12 mm of the holder's | The collar allows 10 mm of play; 2 mm more is slack. |
| Still | under 1 cm/s and 0.5 rad/s, for 3 s | As in the program's other tasks, and longer than the last release and retreat. |

### Exploits tried (all in `tests/test_scoring.py`)

- Doing nothing scores 0, and so does the scripted demo with both arms idle.
- The right gripper alone (going in without a test) breaks rule 1: 0.
- The left gripper alone can make the test and put the probe back: 0.25, the intended credit for the test alone.
- Going in as the probe arrives at the bottom, before any dwell: rule 1, 0.
- Skipping the top band, going in reverse order, dwelling only 1 s, or sweeping slowly through all three bands
  without stopping all leave the test incomplete. Going in then breaks rule 1 (0); without going in the score is 0.
- A complete test, then the probe raised out of the tank and put back to the bottom at once: the test is void, so
  going in breaks rule 1 (0).
- Raising the probe to 45 mm (above half depth) while the right gripper is inside, or withdrawing it before the
  right gripper is out: rule 1, 0.
- Dropping the probe into the tank and leaving it there (to free the left gripper): rule 3, 0.
- Gripping the probe lower down, so the left pads go below the rim during the bottom dwell: rule 2, 0.
- Dropping the probe outside the tank, or standing it on the tank's rim: rule 4, 0. Knocking it over: rule 5, 0.
- Poking or pushing the part with the probe changes nothing: the part stays in the tank, and the score is 0.25 if
  the test was made. A part flicked out of the tank by anything but the right gripper breaks rule 6: 0.
- Lifting the part out with the pad tips just above the rim, so as never to count as "in the tank", is impossible.
  The part's top is 50 mm below the rim, and the pads' flat faces are 22 mm tall. The pads close on nothing and the
  score stays 0.25. A part held up to just under 10 mm above the rim is still not out (0.25, not 0.5).
- A breach is never undone. A run that goes in early, then does everything right, still scores 0.
- A tilted gripper is judged by its pad corners, not its tip point, so tipping the hand cannot slip the pads under
  the rim unseen.

## One arm alone

Offline kinematic predictions from the demo switches (tested in `tests/test_scoring.py`). The GPU runs will be
recorded in `task.json`'s `scores`.

- `DEMO_IDLE_ARM=right` (the left arm alone) takes the probe, makes the full test, holds the probe low, and puts it
  back in its holder: **0.25**. The part stays in the tank.
- `DEMO_IDLE_ARM=left` (the right arm alone) waits, then goes into the tank with no probe in it: rule 1, **0**.

No single arm can finish. The probe must be held in the tank for the whole time a gripper works inside, and the
gripper holding it may not go in.

## Novelty

The closest tasks we were told about on the board:

- **"Open the vent before draining the tank"**: an order of valve operations on a tank, done by one arm, with no
  entry and no monitoring.
- **The hold-while-the-other-works tasks** (a sterile lid held in the air; a skid slid under a raised machine). One
  arm holds something while the other works, and the scored constraint is on the held object's pose.

This task differs in mechanism:

- The pre-entry test is ordered and banded by height, and it only counts if each level is held still long enough.
- The test is a precondition that gates the other arm's access to a region, and it is voided when the probe leaves.
- Continuous in-space monitoring is checked at every step of the entry, not only at the end.
- The roles split: the attendant gripper may never break the plane of the opening, while its probe must stay
  inside.

None of the board tasks or the common sets (LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench, RoboDojo)
score entry permits, atmospheric testing order or attendant rules.

## Demo and run limit

The scripted demo takes 41.0 s (820 control steps). Both arms close their fingers along y: the left at yaw -pi/2,
the right at +pi/2. Every grip is commanded 8 mm narrower than the object (22 mm on the 30 mm probe and on the
30 mm part), and every open or close takes 1.5 s.

1. The left arm comes down over the probe. It lowers the last 2 cm slowly, grips the probe with the pad tips 30 mm
   below its top, and lifts it straight up out of the collar. The probe's lower end is then 42 mm above the rim.
2. It carries the probe over the tank, to x 28 cm, y +14 cm, and lowers the tip to the centre of each band in turn
   (65, 40 and 15 mm), holding 3 s at each. Offline, the bands count at steps 265, 349 and 439. Throughout, the
   left pads stay at least 55 mm above the rim.
3. The right arm has waited over the part with its pad tips 67 mm above the rim. It starts down 0.5 s after the
   bottom dwell ends, and its fingers pass the rim at step 498, 3 s after the test completed.
4. The right arm lowers the last 2 cm slowly, grips the part with the pad tips 4 mm above the floor, and lifts it
   to 53 mm above the rim. It carries the part to the tray, lets it drop from 15 mm above the tray's rim, and goes
   up.
5. 0.5 s after the right arm's lift has ended, the left arm lifts the probe out, carries it back, and lowers it into
   the collar with the last 2 cm slow. It lets go with the probe 2 mm above the table and goes up.

Offline, success has held its 3 s at step 857 (42.9 s). The run limit is 1,200 steps (60 s).

Geometry checked in `tests/test_demo_plan.py`:

- The TCPs are never closer than 22.5 cm in y.
- The demo keeps 9.1 cm (left) and 4.7 cm (right) inside the reach limits, including the midline limit at height.
- Inside the tank, the right gripper's open pads (110 mm across their outer faces, 27 mm wide) have 46 mm to each
  long wall and 40 mm to the end wall.

The arms' controller does not carry a payload's weight. The left arm should sag about 3 mm under the 0.12 kg probe,
and the right about 6 mm under the 0.21 kg part. The proxy test plays the demo with no sag, with that sag, and with
8 mm on both arms: it scores 1.0 every time, and the tip stays at least 7 mm above the floor and inside each band.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the probe back upright in the centre
of its holder at x 20.0 cm, y +25.0 cm, and put the part back on its mark on the tank floor at x 28.0 cm,
y -8.5 cm, sides parallel to x and y. The tank, holder and tray are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is about 43 s of simulated time). The first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

- **Tank:** an open-top steel or aluminium box, 120 x 360 mm inside, 80 mm deep, with 8 mm walls (or a printed
  shell on a thin base), clamped to the bench.
- **Part:** a 30 mm steel cube (about 0.21 kg).
- **Probe:** a 30 mm aluminium tube, 150 mm long, with a 50 g plug at its lower end (about 0.12 kg), painted
  orange. Paint rings on it at 5, 25, 30, 40, 50, 55 and 75 mm from its lower end.
- **Holder:** a black collar with 30 mm walls round a 50 x 50 mm opening, clamped to the bench.
- **Tray:** a blue tray with 25 mm walls round an 80 x 80 mm floor, clamped to the bench.

Film from the side, level with the rim. The rim's position between the probe's rings gives the tip's depth below
the rim: 5 to 25 mm for the top band, 30 to 50 mm for the middle, 55 to 75 mm for the bottom, and beyond the 40 mm
ring for the lower half. The video's timestamps give the dwell times. The same view shows whether either
gripper's pads go below the rim. The arms are never near anything that could hurt them: the tank is open, its
walls are 8 mm steel, and nothing is under load.

## Assumptions a GPU run must confirm

- **Gripper housing.** A gripper working on the tank floor has 63 mm of itself below the rim. The fingers are
  known to clear the walls with 40 to 46 mm to spare, but the size of the housing above them was not available
  offline. Watch the first demo run for contact between the right wrist or housing and the walls. If there is
  contact, the tank can be made wider along x without changing any rule.
- **The probe grip.** A 30 mm cylinder held between two flat pads 30 mm below its top, with 120 mm hanging below,
  is assumed not to slip or swing. In `TASK_TRACE`, `tip_height_mm` during the dwells should stay within a few mm.
- **The opening reading.** With the fingers stopped on a 30 mm object, rigstate's opening is assumed to read about
  0.32, well under the 0.5 'hold' limit. Check `opening` and `left_holds_probe` in `TASK_TRACE`.
- **Payload sag.** Sag was taken from the program's figures (about 8 mm for 0.3 kg at 0.3 m). The bands tolerate
  10 mm of it. Check `tip_height_mm` at each dwell, and `min_left_above_rim_mm` (about 55 mm expected).
- **Settling.** Each dwell is 3 s, against the 2 s the scorer needs, leaving about 1 s for the arm to settle. If
  `band_done_steps` shows a band counting late, lengthen `DWELL` in `task.py`.
- **Releases and joints.** The probe is assumed to stand in its collar after being let go 2 mm up, and the part to
  stay in the tray after a 15 mm drop. The right wrist at yaw +pi/2, at x 28 cm and y -8.5 cm with the TCP 17 mm
  above the table, is assumed to be inside its joint limits. The program's reference tasks use the same yaw at
  the midline.

## Results in simulation

To be measured on the GPU (not yet run). Measure with the commands above, one trial each, and fill in:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | to be measured on the GPU (offline prediction 1.0) |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | to be measured on the GPU (offline prediction 0) |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | to be measured on the GPU (offline prediction 0.25) |
| Dream-1 | to be measured on the GPU |
