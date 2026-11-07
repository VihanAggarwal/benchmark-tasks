# Set a small part on a surface grinder's magnetic chuck and block it in

A surface grinder holds its work on a magnetic chuck whose face is steel pole pieces separated by non-magnetic
brass lines. A part is held only where it bridges a brass line from one pole to the next, so a small part sitting on
a single pole is barely held. Small parts are set against the chuck's back rail and "blocked in": steel pieces butted
against their ends stop the wheel pushing them along the table, and the blocking must be lower than the part so the
wheel passes over it. Here a chuck with a 5 mm rail along its left edge and three brass lines 60 mm apart is fixed on
the table, a blue 40 x 30 x 20 mm steel part lies to its left, and four black steel blocking pieces lie to its right:
two lower than the part and two taller than it whichever way they lie. The part must end flat and lengthwise against
the rail across a brass line, with a low piece against each end; a piece taller than the part left anywhere on the
chuck scores 0. The scorer reads only the end state, the way a setter checks it before starting the table.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the five loose objects' states and the grippers' state, with a small tracker |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, blow-ups and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The score counts an item only once no gripper is holding it, and `yam_arena` does not pass the grippers' state to a
task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to
take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each
control step and to start a new trial at each reset. It reads only state the simulator already has and changes
nothing the embodiment does. If it cannot read the arms it prints why once; nothing then counts as held on those
steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the blue steel part flat on the magnetic chuck for grinding: lengthwise against the rail along the chuck's left edge, straddling one of the brass lines. Then block it in: put a blocking piece against each end of the part, using only pieces lower than the part so the grinding wheel passes over them. Let go of everything when done." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/grinder-chuck-block-in/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the blue steel part flat on the magnetic chuck for grinding: lengthwise against the rail along the chuck's left edge, straddling one of the brass lines. Then block it in: put a blocking piece against each end of the part, using only pieces lower than the part so the grinding wheel passes over them. Let go of everything when done." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/grinder-chuck-block-in/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`against_rail`, `across_line`, `near_end_blocked`,
`far_end_blocked`, `too_tall_on_chuck`, `part_released`, `part_tilt_deg`, `part_bottom_mm`, `part_square_deg`,
`rail_gap_near_mm`, `rail_gap_far_mm`, `line_steel_near_mm`, `line_steel_far_mm`, `near_gap_mm`, `far_gap_mm`,
`near_angle_deg`, `near_overlap_mm`, ..., every piece's height above the face, each finger zone's distance to each
object, the openings, every loose object's position, both TCPs). Videos: `inspect-robots video <log>.json`. The scorer
keeps one trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from where the objects rest at the end.
The blocking pieces are told apart only by their size, never by their names: any piece lying flat on the face with
its top at least 1 mm below the part's top can block an end, and any piece on the chuck with its top more than 21 mm
above the face (the part is 20 mm tall) scores 0, however it lies. The low pieces are 15 mm tall lying flat and
18 mm on a side; the tall ones are at least 24 mm whichever way they lie, so a tall piece cannot be laid down to pass
as a low one. A piece is on the chuck when some of its outline seen from above is over the chuck and its lowest
point is at most 21 mm above the face, which covers lying on the face, on the rail, propped against the part or
stacked on it.

| Outcome | Score |
|---|---|
| Any blocking piece on the chuck with its top more than 21 mm above the face (higher than the part) | 0 |
| (R) The part lies on a 40 x 30 mm face (tilted under 3 degrees, lowest point within 2 mm of the face), wholly over the chuck, its length within 5 degrees of the rail, both lower corners of its rail side within 3 mm of the rail's face, not held | 0.25 |
| (B) With R: a brass line under the part with at least 8 mm of the part's length on steel on each side | +0.25 |
| (E) With R, for each end: a piece flat on the face, top at most 19 mm up, not held, its side facing the part parallel to the end face within 10 degrees, its nearest point within 3 mm of the end face, covering at least 10 mm of the end face's 30 mm depth | +0.25 each |

Possible scores: 0; 0.25 (R alone: the part against the rail but on one pole); 0.5 (R and B, or R and one blocked
end); 0.75 (R, B and one end, or R and both ends off a line); 1.0 (all four). Without R nothing else counts: the
rail is the reference the whole setup is built from. The 8 mm in B is measured along the rail from the line's edge
to the centre of each end face, so a part 9 mm or less off a line's centre bridges it; the inner poles are 54 mm
wide, so a 40 mm part can also sit wholly on one. "Held" means the item is within 2 mm of a gripper's finger zone: a
solid cylinder round the gripper's axis from the finger tips 38 mm back up, flat at the tips, its radius reaching a
pad's outer corner at any heading (57 mm fully open, 25 mm at the 26 mm the demo opens to, 22 mm closed on an 18 mm
piece). Because the zone ends flat at the finger tips, a gripper hovering open above an item has let go of it once
its tips are more than 2 mm above it; a finger resting on an item still holds it. Because the zone is round (the
scorer does not read the gripper's heading), a gripper still closed on the middle of a piece against an end of the
part also reaches the part: the part then counts as held, so the score is 0 until that gripper lets go and moves
off. The trial ends early once 1.0 has held for 1 s with the part and both blocking pieces still (each moved under 2
mm, path length, over the last 0.5 s, and slower than 5 cm/s); otherwise it is scored as it stands at the run limit.
A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts the part back on
its starting mark (within 4.5 cm) in a single step. A simulator state with a non-finite number, or a missing object,
scores 0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing more than they should, as the tests show: a tall piece used as blocking, or left
anywhere on the chuck (lying flat, on its side, on the rail, or half over the chuck's edge), scores 0; so does a low
piece stood on end, propped steeply against the part, or stacked on it (each stands above the part). Both low pieces
at one end block one end (0.75 with B). A piece leaning against an end, more than 3 mm away, turned more than 10
degrees, covering less than 10 mm of the end face, or set beside the part instead of at its end does not count. The
part laid crosswise, on its 40 x 20 mm side (30 mm tall), tilted, raised, short of the rail, or still held scores 0;
a piece still held at the end loses that end, and when it is held by its middle against the part the part counts as
held too (0). A tall piece on the table beside the chuck, or held high above it, is not on the chuck. A low piece
lying on its side (18 mm tall) counts: it is still lower than the part, and real blocking is any steel lower than
the work. A piece whose top ends between 19 and 21 mm above the face (a low piece on its side tipped up 4 degrees)
neither blocks an end nor zeroes the score.

## One arm alone

The demo uses both arms: the left arm places the part and the right arm the two low pieces. The part starts within
easy reach of the left arm and the blocking pieces within easy reach of the right arm. With a straight-down grip at
any heading, the right arm's open fingers reach to within about 2 mm of the part's nearest corner where it starts
(too little to grip it) and the left arm's stay about 18 mm short of the nearer low piece and about 50 mm short of
the other (inverse kinematics of the arm's URDF with joint limits and a collision proxy). So the scripted demo with
one arm idle does only that arm's half: the left arm alone sets the part against the rail across a line for 0.5, and
the right arm alone blocks an empty chuck for 0. The GPU runs (L40S, see 'Results in simulation' below) gave exactly
that, as the offline kinematic check predicted: `DEMO_IDLE_ARM=right` 0.5 (the part against the rail, 1.3 and
0.5 mm from it, across the middle line, both ends open) and `DEMO_IDLE_ARM=left` 0 (both low pieces set on the
chuck, the part still at its start). These numbers show what this scripted demo does with one arm; they are not a
limit on one-arm policies. Two arms are not a rule of the task, and nothing in it needs two arms physically: a
one-arm policy that tilts its gripper, or nudges the part toward itself before gripping it, could bring everything
within its reach and score up to 1.0.

## Demo and run limit

The scripted demo takes 51.6 s (1,032 control steps). The left arm grips the part by its two 30 x 20 mm ends
(fingers closing along y, commanded 32 mm, over 1.5 s; pad tips 7 mm above its bottom), lifts it to 10 cm, carries it
over the chuck while turning the wrist 90 degrees so the part lies along x, comes down to 2 cm above the face,
settles, lowers the last 2 cm over 2 s and lets go with the part's bottom 0.5 mm above the face, centred over the
middle brass line (x 27.0 cm) and 2.8 mm short of the rail; it goes straight up and parks to the left. Meanwhile the
right arm picks the nearer low piece across its 18 mm width (fingers 44 mm apart on the way down, commanded 10 mm),
waits 26 cm right of the midline until the left arm has let go and lifted (t 20 s), sets the piece against the
part's far end (gap aimed at 1.0 mm), opens to 26 mm and goes up, then picks the other low piece and sets it against
the near end (gap aimed at 2.3 mm) the same way, and parks. The pieces cover the 18 mm of the end faces farthest from
the rail, 14 mm short of it, and the pad tips are 6.5 mm above a piece's bottom, above the 5 mm rail, so the gripper's
rail-side pad never meets the rail. The TCPs stay at least 27 cm apart and the arm links 9.9 cm (capsule proxy).

The arms are joint springs and the controller does not carry a payload's weight. The compliance model (J K^-1 J^T,
joints 80/80/80/10/10/10 N m/rad) puts the 188 g part's lower edge, at the place point, 1.7 mm nearer the arms and
1.9 mm nearer the rail than commanded, and an 85 g piece's 0.7-0.8 mm nearer the arms and 0.8-0.9 mm to the right.
So the place points leave room for it: the offline check, which has no sag, sees the part 2.8 mm from the rail and
the near and far gaps at 2.3 and 1.0 mm. In the GPU runs of this demo (L40S) the part landed 1.8 and 1.0 mm from the
rail at its two corners and turned 1.1 degrees, and the near and far pieces 1.7 and 2.0 mm from its ends, all inside
the 3 mm limits; the part lay flat (0.0 mm off the face) with 18.4 and 15.6 mm of steel either side of the middle
line. On the GPU the trial ended with 1.0 held for its 1 s at step 993 (offline, step 992); the scripted policy
would stop at step 1,093. The run limit is
1,800 steps (90 s).

The offline check warns that the part's centre is not over "its support": the part straddles a brass line, which
is its own fixed strip, so it rests on three strips of the chuck's face at one height.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the part back on the table at x 20.0 cm,
y +26.0 cm, its 40 mm length along y; the low pieces at (25.0, -25.0) and (17.0, -31.0) cm and the tall pieces at
(17.0, -25.0) and (25.0, -31.0) cm, each lying on its largest face, length along x. The chuck and the rail are fixed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is 52 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

No magnetic chuck is needed (nothing is energised): a 200 x 100 x 25 mm steel or aluminium plate clamped to the
bench, with three 6 mm brass strips (or brass-coloured tape) across it 60 mm apart, and a 10 x 5 mm steel bar fixed
along its left long edge as the rail. The part is a 40 x 30 x 20 mm steel block painted blue (188 g); the low pieces
are two 40 x 18 x 15 mm steel blocks (85 g each) and the tall pieces two 40 x 24 x 28 mm steel blocks (211 g each),
all four blackened alike. Check the end state with a steel rule, a 3 mm feeler gauge and a straight edge laid across
the part's top, with the grippers open and every finger more than 2 mm clear of the part and the pieces (the
simulated zone, which takes in both pads at any heading, is stricter beside an item than a ruler on the real
fingers).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The demo ended at step 993 with 1.0 held for 1 s. The Dream-1 trial ran the full 1,800 steps and ended with every
object still on its starting mark (0). With the left arm idle the right arm set both low pieces on the chuck but the
part never reached it (0); with the right arm idle the left arm set the part against the rail across the middle
brass line with both ends open (0.5).
