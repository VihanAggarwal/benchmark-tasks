# Set four drift-fence arms tight against a pitfall trap

Ecological consultants and wildlife biologists survey ground-dwelling reptiles, amphibians and invertebrates with
pitfall traps (buckets sunk flush with the ground) fed by drift fences: low barriers that animals follow until they
drop in. In Y and cross arrays several fence arms radiate from one central hub trap. Each arm is staked out, then
set straight along its line with its end butted to the hub's rim: a gap lets animals walk round the trap, and an arm
end pushed out over the mouth narrows the opening the other arms feed. This is a tabletop model of a cross array:
a black disc stands for the hub trap's mouth, four orange pegs mark the far ends of four arms, and four pine boards
are the fence. Each board must end standing on its long narrow edge along the line from the trap's centre to one
peg, its inner end 0 to 10 mm from the rim and not over the opening, with the grippers let go and lifted clear. The
result is checked as a field crew checks it: a feeler gauge or gauge block at the rim and a sight along each arm.
Practice varies between protocols (straight-line fences often run across the middle of an end bucket, and some crews
lap a fence a little over a bucket's lip). On a hub trap fed by four arms, an arm end over the mouth blocks the other
arms' entry, so this protocol treats any overlap as a failed set-up: the instruction says "right up against the
trap's rim but not over its opening", task.json's context gives the reason, and the score is 0.

Nearest tasks on the board: TK-DV2K3W (six blocks set in a ring round a post) and TK-TMBFRA (toe-boards stood on
edge along a platform's straight edges, gaps of 6 mm or less). Here long parts are aimed at a point along lines that
pegs give, with the end gap measured to a round rim and any overhang scoring 0, as opposed to point objects set at a
radius or boards laid along straight edges.

Methods: Corn 1994, "Straight-line drift fences and pitfall traps", in Heyer et al. (eds.), *Measuring and
Monitoring Biological Diversity: Standard Methods for Amphibians* (Smithsonian Institution Press); USGS Techniques
and Methods 2-A5 (pitfall-trap arrays for herpetofauna). Workers: Zoologists and Wildlife Biologists (O*NET
19-1023.00), Biological Technicians (19-4021.00), Environmental Science and Protection Technicians (19-4042.00).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain NumPy functions of the boards' poses and the grippers' TCPs, with a small tracker |
| `rigstate.py` | Reads both arms' TCPs for the scorer (wraps the embodiment's step and reset; changes nothing it does) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, the exploits we tried and the task file's hooks (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed cylinders, loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the drift fence around the black pitfall trap: for each of the four orange pegs, stand one wooden board on its long narrow edge along the line from the trap's centre to that peg, with the board's inner end right up against the trap's rim but not over its opening. Then let go of every board and lift the grippers clear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/drift-fence-pitfall-arms/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the drift fence around the black pitfall trap: for each of the four orange pegs, stand one wooden board on its long narrow edge along the line from the trap's centre to that peg, with the board's inner end right up against the trap's rim but not over its opening. Then let go of every board and lift the grippers clear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/drift-fence-pitfall-arms/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: for each board its `line` (sector), `fenced`, `standing`,
`tilt_deg`, `bottom_mm`, `at_rest`, `path_mm`, `free`, `tcp_clear_mm`, `bearing_deg`, `axis_err_deg`, `offset_mm`,
`gap_mm` and `over_rim`; then `lines_fenced`, `zero_rule`, `arms_available`, `arms_unavailable`, the boards' poses and
both TCPs and openings. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the four boards'
poses and, through `rigstate.py`, both grippers' TCPs. It keeps no rule about the whole trial, only each board's last
0.5 s of centre positions (for "at rest") and a count of steps on which the arms could not be read.

- **Footprint**: a board's whole outline seen from above (the convex outline of its eight corners), standing or not.
- **Gap**: the plan distance from the trap's centre to the nearest point of the footprint, minus the 35 mm rim
  radius. Measured to the nearest point, not to the end face's middle, so a board set off to one side or turned is
  judged by the corner that is really nearest the rim.
- **Standing**: the 50 mm axis within 10 degrees of vertical (either long edge down), the lowest point within 3 mm
  of the table top, and at rest: every centre position recorded over the last 0.5 s (the current control step and
  the 10 before it) within 2 mm of the current one. Speeds are never read: a gripped part's reported velocity
  chatters at 10-45 mm/s while it stays put.
- **Free**: no gripper TCP that is less than 90 mm above the table is within 60 mm, in plan, of the footprint. With
  the fingers fully open (94 mm) a pad's outer corner is 56 mm from the TCP in plan (47 mm half gap, 8 mm pad,
  10 mm half width), so a pad resting on a board always makes it not free; with its TCP 90 mm up a straight-down
  gripper's pad tips are 27 mm above the board's top. If the arms cannot be read on a step, every board counts as
  free on that step (and `arms_unavailable` counts it).
- **Sector**: each board belongs to the peg whose bearing from the trap's centre is nearest the bearing of the
  board's centre: four 90 degree sectors centred on 45, 135, 225 and 315 degrees. Lines 1 and 3, and 2 and 4,
  continue each other through the trap, so without sectors a board on line 3 would also lie along line 1.
- **Fenced**: line k is fenced when a board in sector k is standing and free, its long axis is within 8 degrees of
  the trap-to-peg-k direction (either end may face the trap), the trap's centre is within 10 mm of its long centre
  line, and its gap is -1 to +10 mm (the 1 mm below zero allows for contact against the disc's edge). Each board
  fences at most its own sector's line; each line counts once. A board turned about its middle misses the trap's
  centre by 10 mm at about 5.5 degrees; one swung about its inner end can turn the full 8 degrees.
- **Zero rule**: the score is 0 if, at the scored step, any board (standing, lying, held or not) has a gap below
  -1 mm: its outline is more than 1 mm inside the rim, on or over the opening. On this four-arm hub an end over the
  mouth blocks the other arms' entry, so the protocol counts the set-up as failed; the 1 mm allows for contact
  against the disc's edge.
- Otherwise the score is 0.25 per fenced line. Success, which ends the trial, is all four lines fenced with no board
  over the rim, held for 1 s; otherwise the score is read at the last step.

| Outcome | Score |
|---|---|
| Boards at their starts | 0 |
| One board on its line | 0.25 |
| One arm's two boards on its two lines (1 and 2, or 3 and 4) | 0.5 |
| Two boards on lines 3 and 4 only (lines 1 and 2, which continue them, are not credited) | 0.5 |
| Three good boards, the fourth turned 12 degrees | 0.75 |
| All four on their lines with 15 mm gaps | 0 |
| Four good boards, one end 3 mm over the rim | 0 |
| Four good boards, one still gripped (fingers closed, TCP 35 mm up) | 0.75, until it is let go and the gripper's TCP rises to 90 mm above the table or is more than 60 mm, in plan, from the board's outline |
| Four good boards, an open gripper left low with one finger resting on a board's face (TCP 47 to 56 mm from it) | 0.75 |
| Boards laid round the trap tangentially | 0 |
| A board lying flat on its line | that line is not fenced |
| A board lying across the trap | 0 |
| Two boards in one sector | that line counts once |
| Four good boards, let go, grippers clear, still for 1 s | 1.0 |

A simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it
was. A new episode (rigstate's reset hook, or every board put back on its own starting mark in one step) clears the
position history and the count.

Unintended strategies the tests try, and what they earn: one board along line 3 fences only line 3 (0.25, not 0.5);
one arm's two boards on lines 1 and 3 earn 0.5; two boards end to end through the trap put a joint over the opening
(0); a board stood on its 20 x 50 mm end along a line, stacked on another board, leaning 35 degrees after a drop, or
set beyond the peg or 20 mm short of the rim fences nothing (a board standing on its other long edge counts: the two
long edges are alike); a board in one sector aimed at the neighbouring peg, or between two pegs aimed at the trap,
fences nothing; a gripper left hovering 60 mm over a finished board loses that line (0.75), and so does an open
gripper left low with one finger pressed on a board's face or corner (0.75); a held board hovering over the mouth at
the end, or a board riding up on the 3 mm disc, scores 0; a board still creeping (6 mm/s) is not at rest, and one
that falls over after release loses its line; reported velocity chatter on still boards changes nothing; pushing or
sliding boards into place is allowed and is scored only by the end state. Random poses never score outside 0, 0.25,
0.5, 0.75 or 1.0.

## One arm alone

Measured in simulation (L40S, the same demo with one arm left at its ready pose): `DEMO_IDLE_ARM=right` (the left
arm alone sets its two boards on lines 1 and 2) scored 0.5, and `DEMO_IDLE_ARM=left` (the right arm alone, lines 3
and 4) scored 0.5, as predicted offline (`check_task.py --one-arm`). We do not expect one arm to score more than
0.5: it reaches three of the four line positions (its own two and the near one on the other side, at x 19.6 cm,
y -/+7.4 cm), but it can grip only its own two boards. The other arm's boards start 67.5-75.5 cm across from its
base (y), beyond the reach of a straight-down or tilted (up to 40 degree) grip in our kinematics checks; and with the
sector rule its two boards cannot be credited twice on the collinear lines. Nothing in the act itself needs two hands: each board is a one-arm pick and
place. The two-arm need comes from the layout (each arm can reach only its own side's boards; task.json's setup says
so), and the two arms work in parallel. (The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 34.0 s (680 control steps), in two simultaneous 16 s phases and a 2 s park:

- Phase A (0-16 s): the left arm sets board 1 (y +37 cm) on line 1 and the right arm board 3 (y -37 cm) on line 3.
- Phase B (16-32 s): the left arm sets board 2 on line 2 and the right arm board 4 on line 4.
- 32-34 s: both arms park at x 22 cm, y +/-30 cm, TCP 16 cm up, yaw 0.

The inner board of each pair goes first, so the open outer finger clears the outer board by 15 mm. Each board is
gripped 20 mm from its centre towards the end that will face the peg (`GRIP_OFFSET`), so its TCP ends 125 mm out
along the line while its centre ends 105 mm out. Each board: 2.0 s to above that grip point (TCP 12 cm up, fingers
open, yaw 0), 1.5 s down to TCP 35 mm (pad tips 22 mm up the 50 mm board), 0.5 s settle, 1.5 s close to 12 mm (8 mm
under its 20 mm thickness), 1.5 s lift to TCP 12 cm (board bottom 85 mm up, clear of the 60 mm pegs and 50 mm
boards), 2.5 s carry to above its line while turning the grip +45 degrees (lines 1 and 3) or -45 degrees (lines 2
and 4), 0.5 s settle, 1.2 s down to 2 cm above the place, 1.3 s for the last 2 cm (board bottom 1 mm up), 0.5 s
settle, 1.5 s open, 1.5 s straight up to 12 cm. Each board's centre goes 105 mm from the trap's centre along its
line, at x 34.42 / 19.58 / 19.58 / 34.42 cm, y +7.42 / +7.42 / -7.42 / -7.42 cm for lines 1-4: its inner end face is
40 mm out (gap 5 mm), its inner corners 41.2 mm out (6.2 mm clear of the rim), its outer end 170 mm out (24 mm short
of the peg's surface); adjacent inner ends are 42 mm apart.

Offline (no physics) the last two boards are let go at 29.35 s, the grippers rise past 90 mm at about 31.4 s, and
success has held for its 1 s at step 647 (32.35 s); the scripted policy stops at step 741. The run limit is 1,800
steps (90 s), 2.6 times the demo.

Offline kinematics: worst IK error 0.083 mm / 0.013 degrees; smallest joint margins joint 4 at 14.1 degrees (the
left arm arriving above board 2, the right arm above line 4) and joint 6 at 43.9 degrees; the TCPs never come
closer than 25.0 cm; the arms' collision capsules stay 7.4 cm apart at worst (both lowering in phase B, the two
link4 capsules); links at least 64 mm above the table; no finger pad or link runs into a peg or board. With grips
at the boards' centres the capsules came within 3.7 cm (the left arm on line 2, the right on line 4), and no
timing helped: the closeness comes from that pair of reach poses, and with the left arm risen above line 2 while
the right lowers onto line 4 it is 3.1 cm. Gripping 20 mm out puts each TCP 20 mm further from the trap along its
line and gives 7.4 cm (15 mm out would give 6.5 cm). The phases are therefore simultaneous. Carrying at TCP 16 cm
over line 1 would leave joint 4 only 0.06 rad, so the carry is at 12 cm.

What to check first on the GPU (`TASK_TRACE`):

1. A 200-step Dream-1 run: the API accepts the instruction as worded.
2. Settle (`DEMO_HOLD_ONLY=1`): all four boards stay standing at their starts.
3. Each released board stays standing (`tilt_deg` under 3), slides under 3 mm along its line and turns under 3
   degrees after the fingers open; `gap_mm` lands in 0-10 mm (aim 5-6; if it does not, change `PLACE_RADIUS` in
   `task.py` by the difference) and `offset_mm` under 10 (if a board turns in the grip, trim its `LINE_YAW`).
4. `free` turns true for each board as its gripper rises; `arms_unavailable` stays 0; success at about step 650.
5. The board's centre hangs 20 mm off the grip axis (65 g: 0.013 N m). It must not pitch in the grip: in the
   video the board stays level while carried, and `tilt_deg` and `bottom_mm` stay small after release. If it
   pitches, lower `GRIP_OFFSET` in `task.py` to 15 mm (6.5 cm between the arms' capsules) and re-run check_task.

What the GPU runs showed (L40S, `TASK_TRACE`): in the settling run the four boards did not move from their starts.
In the demo no board tilted more than 1.8 degrees at any point, carried or set (so the off-centre grip did not pitch
them, and `GRIP_OFFSET` stays 20 mm); at the end the four gaps were 5.3-6.1 mm, the centre-line offsets 1.1-3.7 mm and
the axis errors 1.1-1.2 degrees; no board ever went over the rim; `arms_unavailable` stayed 0; lines 1 and 3 were
fenced from step 313, lines 2 and 4 from step 633, and success ended the trial at step 648. The full 1,800-step
Dream-1 run was accepted as worded and ran to the step limit.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four boards back on their 130 x 20 mm
edges at x 24.0 cm, y +37.0, +45.0, -37.0 and -45.0 cm, long sides along x. The trap and pegs are fixed. Nothing is
consumed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is 34 s of simulated time, a full Dream-1 run
90 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 70 mm disc of 3 mm black-painted plywood (or rubber sheet) fixed to the bench as the trap; four 12 mm wooden dowels,
60 mm long, painted orange and glued upright 200 mm from the disc's centre at 45, 135, 225 and 315 degrees (drill a
template, or mark two diagonals through the centre); four planed pine blocks 130 x 20 x 50 mm (about 65 g each).
Check each trial after the arms withdraw: a 10 mm gauge block (and a feeler gauge for the 1 mm overlap) between each
board's inner end and the rim, a string from the disc's centre to each peg's centre or a top photo for the angle and
the centre-line offset, and a small square or phone level for standing.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 (full 1,800 steps; all four boards were left at their starts) | 0.0 |
