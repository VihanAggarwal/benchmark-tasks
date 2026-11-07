# Load a CNC tool magazine: the face mill gets empty pockets on both sides

Before a job, a CNC machine operator or tool setter loads the new tools into the machine's tool magazine. Tool-changer
manuals single out large (oversize) tools, typically face mills wider than the pocket spacing: a large tool needs the
pockets on both sides of it left empty, or it strikes its neighbours when the magazine indexes or the changer arm
swings. So the setter looks for a pocket whose two neighbours are free, keeps them free while loading the rest, and
leaves the tools already loaded where they are (in a fixed-pocket magazine the tool table records which pocket holds
which tool). Here a bench-scale magazine stands on the table: a round plate with eight square sockets on a ring,
numbered P1 (straight ahead, red tab) to P8 counter-clockwise, three of them (P1, P2 and P5) already holding steel
tools. Two new small tools wait on a tray on the left and a face mill, a 40 mm shank under a 76 mm cutter body, on
a tray on the right. Only one pocket, P7, has two empty neighbours; once the face mill is there, P6 and P8 must stay
empty, so the small tools belong in P3 and P4. The score is what a setter sees in a photo from above: which pockets
hold a tool, and whether every tool is down and upright in its socket.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the six tools' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds the face mill as one rigid body of two cylinders and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, fixed and loose cylinders, the standard `task_score` scorer, and one helper module
next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain cylinder cannot show a face mill's
wide cutter over its narrow shank. `mechkit.py` (imported by `task.py`) builds the face mill from two cylinders as one
loose rigid body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the
scorer. It changes nothing else the embodiment does, and it stops the run with an error if the face mill was not
built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook
patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first
shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the three new tools from the trays into empty pockets of the tool magazine. The face mill is a large tool: both pockets next to it must stay empty, or it will hit its neighbours at the next tool change. Leave the tools already in the magazine where they are, and let go of the tools." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/magazine-large-tool-pocket/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the three new tools from the trays into empty pockets of the tool magazine. The face mill is a large tool: both pockets next to it must stay empty, or it will hit its neighbours at the next tool change. Leave the tools already in the magazine where they are, and let go of the tools." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/magazine-large-tool-pocket/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`pockets`, `seated`, `tilt_deg`, `floor_mm`,
`centre_off_mm`, `low_mm` and `moved_mm` per tool, `face_mill_pocket`, `face_mill_neighbours_with_tools`,
`small_tools_counted`, `loaded_pockets_missing`, `loose_on_magazine`, `tier`, `zero_reason`, the face mill's origin
and every small tool's position). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the six tools' poses, once per control
step (20 Hz). For each tool it computes the centres of its two end faces: a small tool is the same either end down,
so its bottom is its lower end; the face mill's bottom is its shank's end face, and it is upside down when the
cutter body is underneath (tilt 180 degrees). A tool is seated in pocket k when it is upright within 5 degrees, its
bottom face's centre is inside Pk's 52 x 52 mm opening seen from above and within 3 mm of the plate's top, and it is
at rest: over the last 0.5 s neither end moved more than 2 mm, measured on positions (the simulator's reported
velocities chatter while a part is gripped). A tool is loose on the magazine when some part of it is over the plate
seen from above (its outline from above, end faces and sides, computed exactly from its pose, overlaps the plate's
circle by more than 1 mm; the face mill is its shank and its cutter body), its lowest point is less than 75 mm above
the plate's top or below it (a tool on the table leaning on the plate counts), and it is not seated. Neighbours are
the two pockets next to each one around the ring (P8 and P1 are neighbours). A new episode is recognised when
mechkit counts a reset, or when the simulator puts the three new tools back on their starting marks (within 4.5 cm,
which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number, a zero
quaternion or no face-mill origin scores 0 for that step and leaves the scorer's record as it was.

| Outcome (at the last step) | Score |
|---|---|
| P1, P2 or P5 not holding one of the three steel-grey tools loaded there, seated (they look alike, so a swap among those three pockets is not scored) | 0 |
| Any tool loose on the magazine: lying on the plate or the walls, leaning on the plate from the table, standing on the plate outside a pocket, leaning in a pocket, upside down, on another tool, held low over it, or still moving | 0 |
| The face mill seated in a pocket next to a pocket holding a seated tool (P3, P4, P6 or P8 next to a loaded pocket; or P7 with a small tool in P6 or P8) | 0 |
| Otherwise: 0.5 for the face mill seated with both neighbours empty (only P7 can be), plus 0.25 for each new small tool seated in P3 or P4 | 0 to 1.0 |
| All three loaded so (face mill in P7, small tools in P3 and P4), success once held 2.5 s | 1.0 |

The possible scores are 0, 0.25 (one small tool in P3 or P4), 0.5 (both small tools, or the face mill alone), 0.75
(the face mill and one small tool) and 1.0, and each layout maps to exactly one of them. A small tool seated in P6,
P7 or P8 earns nothing: it takes the face mill's only pocket or one of its neighbours (and with the face mill in P7
it is a zero condition). A tool neither seated nor loose on the magazine (on its tray, on the table with no part
over the plate, on the floor, or with its lowest point 75 mm or more above the plate) earns nothing and does not by
itself zero the score. Unintended strategies earn nothing, as the tests show: taking a loaded tool out (or moving it
to another pocket) to make room next to the face mill (0); standing a new tool on top of a loaded one (0); laying a
tool across a socket's walls or on the plate (0); leaving a tool held a few millimetres above its pocket (0);
standing a small tool upright on the plate between pockets (0). The face mill's cutter would in fact clear a seated
neighbour by 11 mm here (68.9 mm pitch, 38 + 20 mm radii): the rule is the manual's, scored by which pockets are
filled, as a setter checks it. The scorer cannot see the grippers, so a tool held still in its pocket is scored
where it stands (as `scoring` in `task.json` says).

## One arm alone

The rule itself does not need two arms; the layout splits the work by reach. Gripping straight down, both arms
reach every pocket they need (P3, P4 and P7), but each reaches only its own tray: the left arm cannot reach the face
mill, 22 cm right of the midline (it falls 35 mm short at the mill's centre and still 15 mm short gripping the
cutter body's edge), and the right arm cannot reach the small tools, 23 cm left of it (32 and 53 mm short), as
checked with the offline kinematics. On the GPU, the demo with `DEMO_IDLE_ARM=right` (left arm alone) seated both
small tools in P3 and P4 and scored 0.5; with `DEMO_IDLE_ARM=left` (right arm alone) it seated the face mill in P7
and scored 0.5, as the offline check predicted. Gripping straight down, either arm alone can earn at most 0.5: the
left arm's two small tools, or the right arm's face mill. No claim is made that the rule needs both arms: its
difficulty is working out the one pocket the large tool can take and keeping its neighbours free. (All GPU results
are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 35.1 s (702 control steps); the arms take turns at the magazine's middle. The left arm grips
the first small tool on its tray across its 40 mm (fingers commanded 8 mm narrower, over 1.5 s) with the pad tips
36 mm above its bottom, lifts it to carry its bottom 115 mm above the table (35 mm over the loaded tools' tops),
carries it over P3, comes down to 3 cm above the place height, lowers the last 3 cm over 2 s, waits 0.5 s, opens
over 1.5 s with the tool's bottom aimed 2 mm above the plate and goes straight up; it then loads the second small
tool into P4 the same way and moves aside. At P3 and P4 the fingers close along the line from the ring's centre
through the pocket (yaw 0 at P3, 45 degrees at P4), so the open pads swing toward the empty middle of the ring and
past the plate's rim, never toward the loaded tools next door; at P7, whose neighbours are empty, they close
around the ring (along x), which keeps the right arm's open pads away from the left arm. The right arm moves over
the face mill and waits
there 8 s, so that it reaches in over P7 only after the left arm has left P3 (both reaching in at x 0.30 together
would bring the two elbows within a few millimetres); it grips the face mill's 76 mm cutter body (68 mm commanded)
with the pad tips 4 mm above the body's underside, the centre of mass 2 mm above the pad tips, carries it with the
shank's bottom 87 mm up, lowers the shank into P7 the same way (the cutter body stays 15 mm above the socket walls)
and moves aside before the left arm comes in to P4. The arm's controller does not carry a tool's weight, so the
last tool comes to rest on the plate a little before its slow descent ends; success then needs 0.5 s at rest and
2.5 s held, so the trial ends with the fingers open and the hand going up. Offline (no sag) the last tool drops
2 mm at the release and success holds for its 2.5 s at step 689. On the GPU the last tool came to rest just before
its descent ended, and success held for its 2.5 s at step 665 (33.3 s), where the trial ended, with the
fingers open (they finish opening at step 652) and the hand going up. `TRIM` in `task.py` can move a place point if
a tool lands far off its pocket's centre (the sockets give 6 mm of play either side); on the GPU the small tools
landed 2.9 and 1.8 mm from their pockets' centres and the face mill 6.1 mm (4.4 mm along x, 4.2 mm along y, its
shank still clear of the walls), all upright and seated, so `TRIM` is left at zero. The run limit is 1,200 steps
(60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the three steel tools upright in P1, P2 and
P5; the two blue small tools upright on the left tray at x 26.0 cm, y +23.0 cm and x 16.0 cm, y +23.0 cm; the face
mill shank down on the right tray at x 20.0 cm, y -22.0 cm; P3, P4, P6, P7 and P8 empty. The plate, sockets, tab and
trays are fixed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is about 35 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 270 mm round steel (or aluminium) plate 10 mm thick, clamped to the bench; on it eight square sockets of 4 mm flat
bar 25 mm tall, 52 mm inside, glued on a 90 mm radius circle every 45 degrees with their sides square to the bench,
and a red tab beyond the far one (P1). Five 40 mm aluminium tubes 70 mm long with end caps, 150 g each (three left
bare as the loaded tools, two painted blue); and a face mill proxy: a blue 40 mm tube 40 mm long with a smooth black
76 mm disc 30 mm thick fixed on top (no inserts or cutting edges), 280 g in all. Two thin plastic sheets as trays.
Score from a photo from above with the grippers open: which sockets hold a tool, and whether each tool is down and
upright (a ruler against the socket wall shows a tool sitting high).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The demo seated the small tools in P3 and P4 and the face mill in P7, with the three loaded tools unmoved, and
success held at step 665. The left-arm-idle run seated only the face mill (P7); the right-arm-idle run seated only
the two small tools (P3, P4). Dream-1 ran the full 1,200 steps and left every tool where it started.
