# Clamp a thermometer on the suction line near its service valve and insulate it

To check an air conditioner's refrigerant charge by superheat, an HVAC technician reads the suction pressure at
the service port and the suction-line temperature with a clamp-on pipe thermometer. The makers' charging
procedures say where: on the vapour (suction) line near its service valve, insulated (Bryant's split-system
instructions: "Measure vapor-line temperature by attaching a service thermometer to unit vapor line near vapor
valve." and "Insulate thermometer for accurate readings."). The suction line is the larger of the two copper lines
and is normally insulated; the small bare one is the liquid line. A clamp on the liquid line, on a fitting, on the
foam or far down the line, or left uncovered, reads the wrong temperature, and the charge is then set wrong.

Here a bench mock-up of a condensing unit's service corner is fixed on the table: the unit's grey corner, two brass
service valves on its face and two copper lines running from them to the robot's right. The 19 mm suction line has
a brazed coupling at its valve, 24.5 cm of bare copper, then black foam insulation; the 9.5 mm liquid line is bare.
A blue pipe-clamp thermometer (a rigid saddle whose legs pass either side of the line) and a black foam cover (a
half-shell) stand on two pads. The clamp must end astride the suction line on bare copper, its middle within 15 cm
of the suction valve's body, with the cover over it and both grippers let go. The task needs one arm (see "One arm
alone").

The instruction names the line as a technician would, "the suction line", and does not say which of the two copper
lines that is: knowing that it is the larger one, insulated further along, is part of the trade knowledge the task
tests (the clamp's legs would also fit over the small liquid line, which scores 0).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the clamp's and the cover's poses and the grippers' positions |
| `mechkit.py` | Builds the clamp and the cover as compound rigid bodies (several boxes each) and reads their poses |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point) for the scorer each control step |
| `offline_saddle.py` | Used only by an offline kinematic check (no physics): never called in the simulator |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the demo and the exploits we tried |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. The fixed rig is made of `yam_arena`'s
own fixed boxes and cylinders. The clamp and the cover are U-shaped, which one box cannot be, so `mechkit.py`
(imported by `task.py`) builds each as one loose rigid body made of several boxes and reports its pose to the
scorer; it uses no joints. The score also needs the grippers to be clear at the end, and `yam_arena` does not pass
the grippers' state to a task file's hooks, so `rigstate.py` (imported by `task.py`) wraps the embodiment's step and
reset once, at import, to take a snapshot of both arms before each control step and to start a new trial at each
reset. Both read only state the simulator already has and change nothing the embodiment does. If `rigstate.py`
cannot read the arms it prints why once, and the grippers then count as not clear (score 0) on those steps.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm; the right arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the blue pipe-clamp thermometer on the suction line: set it astride the line, on bare copper with its middle within 15 cm of the suction service valve's body and clear of the brazed joint next to the valve. Then set the black foam cover over the clamp to insulate it, and let go of both." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/superheat-clamp-suction-line/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the blue pipe-clamp thermometer on the suction line: set it astride the line, on bare copper with its middle within 15 cm of the suction service valve's body and clear of the brazed joint next to the valve. Then set the black foam cover over the clamp to insulate it, and let go of both." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/superheat-clamp-suction-line/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`seat_checks`, `clamp_x_off_mm`, `clamp_above_copper_mm`,
`clamp_tilt_deg`, `clamp_heading_deg`, `clamp_ends_y_mm`, `d_mm`, `clamp_path_mm`, `cover_checks`,
`ceiling_gap_mm`, `cover_tilt_deg`, `cover_off_mm`, `cover_path_mm`, `clear_margin_mm` (per part, the worse
gripper's margin over the clear rule: 0 or more is clear), `clear_of_clamp`, `clear_of_both`, `arms_unavailable`,
both parts' origins and orientations, both TCPs and finger-tip points). `MECHKIT_TRACE=<file>` adds mechkit's own
per-step record. Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and
starts it over at every reset, so run one trial per process or rely on the reset; the run rate is assumed to be
20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (the tests that load `task.py` need an
offline `yam_arena` stand-in and are skipped without it).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is judged at the end of the trial from the two
parts' poses and the grippers' positions. "At rest" means the part's frame origin travelled less than 3 mm in all
over the last 10 control steps (0.5 s); resting parts' reported velocities chatter in the simulator, so the scorer
never reads a velocity. The clamp's centre is the centre of its top block's underside, so "from 1 mm below to 2 mm
above the copper's top" means the block rests on the copper, its legs either side. Resting astride, the legs leave
7 mm of play a side and cannot turn past about 18 degrees before their inner corners meet the copper, so the 8 mm and
20 degree limits never fail a clamp that rests astride: only where it rests decides. The bare-copper rule takes all
four corners of the block's underside: a block that reaches over the coupling's end (y 16.5 cm) or the foam's end
(y -8 cm) is not on bare copper, whatever its height; the coupling's top is also 3 mm higher than the bare copper,
and the 45 mm foam is too wide for the 33 mm gap between the legs. d is measured along the line from the suction
valve's body face (y 19 cm) to the clamp's centre (its middle, as the instruction says, not its near end), so a seated
clamp has d from 4.5 cm (its block's end at the coupling's end) to 25 cm (at the foam).

A gripper is clear of a part when its tool point and the point between its finger tips are both at least 4 cm above
the top of the part's bounding box, or both at least 6.5 cm from that box seen from above. Two points on the
gripper's axis are not enough on their own: an open finger's outer face is 5.7 cm from the tool point (4.7 cm half
opening plus a 1 cm finger) and its corners about 6 cm, so a finger can rest on a part while both points are more
than 4 cm from it. Below 4 cm above a part the rule therefore asks for 6.5 cm to the side, which no finger of a
downward-pointing gripper, open or closed and at any yaw, can bridge.

| Outcome | Score |
|---|---|
| Clamp seated on the bare suction line, d at most 15 cm, insulated (block inside the cover's outline, ceiling resting on the display, cover tilted at most 10 degrees, at rest), both grippers clear of both parts | 1.0 |
| Clamp seated, d at most 15 cm, both grippers clear of the clamp; not insulated, or a gripper not yet clear of the cover | 0.5 |
| Clamp seated, d over 15 cm (insulated or not), both grippers clear of the clamp | 0.25 |
| Anything else: a gripper not clear of the clamp (holding it, touching it or just beside it); the clamp on the liquid line, the coupling, the foam, a valve, the table or a pad; tilted over 4 degrees, turned over 20 degrees, off the axis by more than 8 mm, or moving; the cover alone; nothing done | 0 |

The run ends early once 1.0 has held for 1 s, which needs both grippers clear of both parts, so the run never ends
early with a downward-pointing gripper still on either part. A run that reaches the run limit with the clamp seated
and let go but the cover still in the gripper (or not yet placed) scores 0.5 (0.25 with d over 15 cm) as long as both
grippers are clear of the clamp: the clamp's step is done and the cover's is not. A new trial is recognised from rigstate's reset hook, from mechkit's reset counter, or, as a fallback,
when the simulator puts a part back at its start (a jump of more than 5 cm in one control step, with both parts then
upright on their pads within 4.5 cm of their starts, which covers a seeded reset's jitter). A part state with a
missing field or a non-finite number scores 0 for that step and leaves the scorer's record as it was.

Unintended strategies earn nothing more than they should, as the tests show: the clamp on the liquid line under
the cover (0); the clamp turned across the line, standing on its legs on the copper (0); a leg on the copper with
the clamp beside the line (0); the clamp on the coupling, where it rides 3 mm high and overlaps the fitting (0);
the cover set on the clamp while it is still on its pad (0); a gripper holding the cover down on the clamp (0, not
clear of the clamp); an open finger resting on the cover's roof, or on the clamp's block, with the tool point and
finger-tip point more than 4 cm from the part (0); the clamp seated 15.2 cm from the valve, its near end at 13.2 cm
(0.25); the cover's ceiling sunk onto the block beside the display (0.5); the cover upside down on the clamp or
tossed on the table (0.5); the clamp knocked off the line after covering (0). The scorer does not check the cover's
heading: what counts is that the whole block is under it, and a cover turned more than about 30 degrees on the clamp
would have its walls across the copper, 16 mm too high to rest on the display. Dragging the clamp along the copper
into place is allowed: only where it ends counts.

## One arm alone

This task needs one arm, and the demo uses one: the left arm sets the clamp and then the cover, while the right arm
holds its ready pose. In simulation, `DEMO_IDLE_ARM=right` (the left arm alone) is the same run as the demo and
scored 1.0, and `DEMO_IDLE_ARM=left` (the right arm alone) leaves the scripted demo nothing to move and scored 0.0
(see "Results in simulation"). So one arm is enough: the left arm alone completes the task. The right arm alone, straight down, cannot reach the clamp's centre
on its pad (50.5 cm left of its base; the inverse kinematics falls 11 mm short at grip height), though it reaches the
clamp's near end (y 18 cm), the cover's pad and the place point at grip height; a grip on the clamp's end corners or
a tilted grip could bring it over, so a right-arm policy could score up to 1.0. No claim is made that the task needs
both arms: its difficulty is knowing which line, which stretch of it, and seating two U-shaped parts over it.

## Demo and run limit

The scripted demo takes 39.9 s (798 control steps). The left arm grips the clamp's top block across its 40 mm
(fingers along y, commanded 8 mm narrower, over 1.5 s) with the pad tips 4 mm above the block's underside, lifts it to
15 cm, carries it over the bare suction line at y 9 cm (d = 10 cm, 5.5 cm clear of the coupling's end), comes down to
3 cm above the place point, settles 0.5 s, lowers the last 3 cm over 2 s with the legs passing 7 mm either side of
the copper, settles and opens over 1.5 s with the block's underside 2 mm above the copper (the finger tips 6 mm above
it), and rises. It then turns the fingers along x (yaw -90 degrees), grips the cover by its two walls across its
77 mm (pad tips 17 mm up the walls), lifts it to 17 cm, carries it over the clamp, lowers it over the last 4 cm in
2.5 s with the walls passing 8 mm either side of the clamp, opens with the ceiling 2 mm above the display and rises
to 18 cm (pad tips 5.75 cm above the cover's top). Offline, success has held for its 1 s at step 806 (40.3 s): the
grippers are clear of the cover from about 39.3 s. The scripted policy stops at step 859. In simulation the demo
scored 1.0 and the run ended early at step 807, once success had held for 1 s. The run limit is 3,000 steps
(2.5 minutes, 3.8 times the demo).

An offline kinematic check (no physics) of the demo predicts 1.0. It sees each mechkit part as its bounding box, so
the carried clamp's and the carried cover's boxes "run into" the suction line as they come down (by 18 and 14 mm):
those are the legs and the walls passing either side of the copper, as designed; against the real shapes the legs
clear the copper by 7 mm a side and the walls clear the clamp by 8 mm a side. Its box model would also set each
part, when let go, on the copper's top (the clamp's box by its legs' bottom, the cover's box by its walls' bottom);
`offline_saddle.py` lowers them to where they really rest (the block's underside on the copper, the ceiling on the
display) for the prediction only. The legs' 7 mm a side is close to how far a loaded arm trails its command (3 to
7 mm), so `task.py` has two trims for the place points: if the GPU trace shows the clamp landing off the line's axis
or off d = 10 cm (`clamp_x_off_mm`, `d_mm`), set `CLAMP_TRIM`; if the cover lands off the clamp (`cover_off_mm`), set
`COVER_TRIM` (the comment next to them gives the signs). In simulation the clamp landed 0.4 mm off the line's axis at
d = 10.0 cm, resting on the copper, and the cover 0.5 mm across the line and 2 mm along it from the clamp's centre, its
ceiling on the display, so both trims are left at zero.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the clamp on its legs on its pad at
x 15 cm, y 20 cm, legs along y, display up; stand the cover on its walls on its pad at x 15 cm, y 8.5 cm, channel
along y. Everything else is fixed and nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 40 s of simulated time, a full Dream-1 run
2.5 minutes); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

All unpowered and at room temperature. Glue or screw to a baseboard: a 12 x 6 x 16 cm painted block as the unit's
corner; a 3 cm and a 2.4 cm brass (or brass-painted) block as the valves, each with a short dowel as its port cap;
a 2.5 cm length of 25 mm square section in a darker copper colour as the brazed coupling; a 24.5 cm length of 19 mm
square copper or copper-coloured aluminium bar as the bare suction line, continued by a 7 cm length of 45 mm square
black closed-cell foam; a 35 cm length of 9.5 mm square bar as the liquid line; two grey painted wooden blocks as the
supports; and two 3 mm grey plastic plates as the pads. The clamp: a 49 x 40 x 16 mm aluminium block with two
8 x 40 x 20 mm aluminium legs screwed under its edges 33 mm apart, painted blue, a 4 mm black plastic square glued on
top (about 60 g; drill pockets to lighten it). The cover: closed-cell foam board glued into a 77 x 70 mm roof on two
36 mm walls (about 20 g). Score with a ruler from the valve face and a top photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |
