# Plot three strike/dip readings by the right-hand rule as geological map symbols

Field geologists, geology students and mapping technicians plot structural readings on geological maps every day.
At each outcrop the strike and dip of the bedding is measured with a compass-clinometer and booked as strike/dip; on
the map it becomes a T symbol: the long bar along the strike, the short tick at its middle pointing down-dip. A
strike value names a line (030 and 210 are the same line), so a convention fixes which side the plane dips to. Under
the right-hand rule the plane dips to the right when you face along the strike: the dip direction is the strike plus
90 degrees, clockwise. (Wikipedia, "Strike and dip"; the GMT geology symbol documentation states the same: strike
"clockwise from the north (azimuth)", the plane "dipping orthogonally to the right when looking in the direction of
the strike azimuth". In the UK the counter-clockwise variant has sometimes been used, which is why the instruction
names the convention.)

Here a 1:10,000 field map lies in front of the arms, north at the top (+x, away from the arms, with a north arrow),
with three outcrop stations circled and lettered: A on the left, C in the middle, B on the right. Three identical
T-shaped tokens (a dark red bar with a yellow tick) lie on the map below them. The instruction gives the three
readings, A 030/40, B 250/25, C 135/60, and the convention, and asks for one symbol centred on each station, turned
as a geological map shows the reading. It never says which way bearings run on the table or which way each tick
must point: reading the north arrow, turning three bearings into three headings and putting each tick on the
down-dip side is what the task tests. The score is read from where the symbols end up, as a protractor on the
finished map reads them: 1/3 per station for the full symbol, 1/6 for the bar on the strike with the tick reversed.

| Station | Reading (RHR) | Strike line (bearings) | Tick toward (RHR) | Tick, left-hand habit (1/6) | Bar yaw | Tick yaw |
|---|---|---|---|---|---|---|
| A | 030/40 | 030-210 | 120 | 300 | -30 (150) | -120 |
| B | 250/25 | 250-070 | 340 | 160 | 110 (-70) | +20 |
| C | 135/60 | 135-315 | 225 | 045 | -135 (45) | +135 |

North is +x and, seen from above, a bearing turns clockwise, toward -y, so a bearing b is the world yaw -b
(counter-clockwise from +x). The dip angles are not plotted (on a map they are written beside the symbol).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, threshold and printed mark, in one place, with the readings and the right-hand rule |
| `scoring.py` | The scorer: plain functions of the three tokens' poses (and the arms' pad-tip points and openings, for 'let go'), with a small tracker of each token's last second |
| `mechkit.py` | Helper that builds each token as one rigid body of two boxes (bar and tick) and reports its pose |
| `rigstate.py` | Helper that reads both grippers (pad-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of the rule's arithmetic, every tier and window edge, every zero condition, stillness, the let-go rule, the exploits we tried, broken states, resets, the task file's hooks and trace, the demo plan and one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a T needs two boxes. `mechkit.py` (imported
by `task.py`) builds each token from its bar and its tick as one loose rigid body, through a hook on
`yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer; it stops the run with an
error if a token was not built as a compound body. The rule includes letting go, and `yam_arena` does not pass the
grippers' state to a task file's hooks: `rigstate.py` wraps the embodiment's step and reset once, at import, to read
both arms' pad-tip points and openings before each control step and to start a new trial at each reset. If it
cannot read an arm, the let-go test is skipped for that arm on that step and the trace counts it
(`arms_unavailable`). Both helpers read only state the simulator already has and change nothing else the
embodiment does; both are the same files as in our other tasks.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Geological field map, 1:10,000, north at the top (its arrow points away from the arms). Stations are circled: A left, C middle, B right. Readings, strike/dip in degrees by the right-hand rule (dip direction 90 degrees clockwise from strike): A 030/40, B 250/25, C 135/60. Each red token is a strike-and-dip symbol: a long bar with a short yellow dip tick at its middle. Plot them: one symbol centred on each circle, turned as a geological map shows its reading, lying flat; then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/strike-dip-symbols-right-hand-rule/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Geological field map, 1:10,000, north at the top (its arrow points away from the arms). Stations are circled: A left, C middle, B right. Readings, strike/dip in degrees by the right-hand rule (dip direction 90 degrees clockwise from strike): A 030/40, B 250/25, C 135/60. Each red token is a strike-and-dip symbol: a long bar with a short yellow dip tick at its middle. Plot them: one symbol centred on each circle, turned as a geological map shows its reading, lying flat; then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/strike-dip-symbols-right-hand-rule/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: `score`, `sixths`, `zero_reason`; for each station (`A`, `B`,
`C`) the token that counts, the candidates and its sixths; for each token its `origin_mm`, `tick_yaw_deg`,
`bar_yaw_deg`, `tilt_deg`, `bottom_mm` (above the map), `lying`, `held`, `still`, nearest `station`,
`off_station_mm`, `strike_err_deg`, `tick_err_deg` and `qualifies`, and its quaternion; `path_mm` (each token's
path in the last two 0.5 s windows); the first step each sixth was reached; each arm's pad-tip point and opening,
`arms_readable` and `arms_unavailable`, or `rigstate_error`. Videos: `inspect-robots video <log>.json`. The scorer
keeps one trial's record per process and starts it over at every reset (rigstate's and mechkit's reset counters,
and a teleport of every token back to its starting mark), so run one trial per process or rely on the reset; the
run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three tokens' poses and both grippers
once per control step (20 Hz). From a token's base frame (origin at the middle of the bar's footprint, local +x the
way the tick points) it reads the tick's heading seen from above, the bar's axis (the tick's heading plus 90
degrees, a line), its tilt, its lowest and highest points and the tick's end.

A token qualifies at a station when its bar's centre is within 8 mm of the station's centre (the ring is 20 mm
across, so a protractor centred on the ring reads it); it lies flat on the map (tilted under 5 degrees, its lowest
point within 3 mm of the map's surface, so a token resting on another token, held up, tipped or upside down does
not); it has been let go (no gripper's pad-tip point is lower than 10 mm above the token's top while within that
gripper's finger reach of the token's outline seen from above: hypot(half the finger gap + 10 mm, 20 mm) + 5 mm,
65 mm fully open and 50 mm closed on a token, at any yaw, since rigstate gives no gripper heading); and it is still
(its bar's centre and its tick's end each travelled under 1.5 mm in each of the last two 0.5 s windows, read from
the path and not from the reported velocity, which chatters while a part is gripped; the tick's end catches a token
turning in place). At each station the qualifying token nearest its centre counts; the tokens are alike and any may
go to any station.

| Station outcome (the counting token) | Credit |
|---|---|
| Bar within 6 degrees of the strike (mod 180) and tick within 6 degrees of the right-hand-rule dip direction | 1/3 |
| Bar within 6 degrees of the strike, tick the other way (the left-hand or old UK habit) | 1/6 |
| Bar more than 6 degrees off the strike, or no qualifying token (none there, held, tipped, stacked, more than 8 mm off centre, moving) | 0 |

Task score = A + B + C, in sixths: 0, 1/6, 1/3, 1/2, 2/3, 5/6 or 1.0; each end state has exactly one. A tick within
6 degrees of the dip direction puts the bar within 6 degrees of the strike, so the two windows cannot disagree. The
trial ends early once 1.0 has held for 2 s; otherwise it is scored on the last step. A state with a non-finite
number, a zero quaternion or no base-frame origin scores 0 for that step and leaves the record as it was.

The wrong answers a non-geologist gives are all far outside the windows, as the tests show: the left-hand habit (all
three ticks reversed) scores 1/2; the first number read as the dip direction (each symbol a quarter turn off) 0;
bearings counted counter-clockwise on the table (strike errors 60, 40 and 90 degrees, either tick side) 0; tokens
moved onto the stations without being turned (any assignment) 0, since every starting bar is 15 degrees or more off
every strike; nothing moved 0. Other strategies we tried: hedging with the second answer stacked on the first (the
stacked token never lies on the map: the station keeps the bottom one's credit only); two tokens at one station
(only the nearest counts, and another station is then empty); a symbol still gripped, or pressed by an open hand
from above (it does not count until let go).

## One arm alone

The demo splits the work by reach: the left arm plots A and then C, the right arm plots B and parks before the left
arm comes down at the middle. With the right arm idle the left arm still plots A and C: 2/3 (offline prediction);
with the left arm idle the right arm plots only B: 1/3. Gripping straight down at any heading, the left arm cannot
reach station B and the right arm cannot reach station A (`tests/test_scoring.py`, `OneArmReach`, our IK model of
the arm); C, on the midline, is in reach of both. So straight down either arm alone can plot at most two stations
(2/3). With a strongly tilted gripper (45 to 75 degrees from vertical) our kinematics do reach the far station from
either base, so no claim is made that one arm alone cannot score 1.0: the task's difficulty is the convention, not
the reach. (GPU results: 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 35.0 s of plan (700 control steps). Each token is gripped straight down at its bar's centre
with the fingers closing along the bar, on its two 20 mm end faces (opened fully to 94 mm, 17 mm either side;
commanded to 52 mm to close, 8 mm under the bar's 60 mm length), so the 20 mm pads stay clear of the tick and centre
the bar between them; the tool point is 17 mm above the token's bottom (pad tips 4 mm over the map). Each token is
lifted to 53 mm, carried to its station while the gripper turns (a quarter turn for tokens 1 and 2, an eighth for
token 3; the token turns with it, so the gripper turns exactly as far as the tick must), settled 0.5 s, lowered the last 20 mm over 1.5 s, settled 0.5 s and
let go 1 mm above the map over 1.5 s; the gripper then rises straight up. Every open and close takes 1.5 s.

| Arm | Token (start, tick yaw) | Gripper yaw | To | Place point (cm) | Gripper yaw there |
|---|---|---|---|---|---|
| left | token 1 (25.0, +15.0), 150 | -30 | A | (33.33, +22.26) | +58.85 |
| right | token 2 (25.0, -15.0), 110 | +110 | B | (33.33, -22.26) | +18.85 |
| left | token 3 (25.0, 0), 180 | 0 | C | (30.40, 0) | -46.15 |

Both arms run on one clock: the left and right arms plot A and B at the same time (their grippers stay 30 cm or more
apart), the right arm parks at x 20, y -25 cm, 16 cm up, and the left arm then plots C and parks at x 20, y +25 cm.
Two trims for the GPU, both inside the windows offline: the place yaw is set 1.15 degrees clockwise of the dip
direction (`YAW_TRIM_DEG`; in our runs gripped parts ended 0.6 to 1.7 degrees counter-clockwise of the commanded
yaw), and each place point leads its station by 4 mm along the carry (`LEAD`; carried parts landed about 4 mm
short). Offline, with neither effect, each symbol lands 4 mm off centre and 1.15 degrees off (windows 8 mm and 6
degrees). Every gripper yaw is within 90 degrees of the
target's bearing from that arm's base, and joint 6 keeps 41 degrees or more from its limits throughout (yamkin). Offline, success holds for its 2 s
at step 697 (the last symbol counts once the open gripper's pad tips have risen 10 mm over it and it has been still
for 1 s); the scripted policy would stop at step 761. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the three tokens back flat on the map,
token 1 centred at x 25.0 cm, y +15.0 cm with its tick pointing 150 degrees counter-clockwise from +x seen from
above, token 2 at x 25.0, y -15.0 with its tick at 110 degrees, token 3 at x 25.0, y 0 with its tick at 180
degrees (pointing at the arms). The map and its marks are fixed. Nothing is consumed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 35 s of simulated time, when
success has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 300 x 800 mm sheet of pale buff paper or card, fixed to the table with its long side across the robot and its near
edge 20 cm in front of the arm bases, printed in black with three 20 mm rings (2 mm lines) at the positions in
`task.json`'s `setup`, the letters A, B and C (16 mm capitals) 50 mm due north of them, and a 40 mm north arrow with
an N pointing away from the robot. Three T-shaped tokens of acetal (POM) or any rigid plastic, about 25 g: a 20 x 60
x 12 mm bar painted dark red and, from the middle of one long side, a 25 x 10 x 12 mm tick painted yellow. Nothing
moves or is powered. Score with a protractor and a rule centred on each ring (or a top photo): the bar's centre
within 8 mm of the ring's centre, the bar's line within 6 degrees of the strike, the tick within 6 degrees of the
dip direction, the token flat on the map and free of the grippers.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.33 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.67 |
| Dream-1 | 0.0 |
