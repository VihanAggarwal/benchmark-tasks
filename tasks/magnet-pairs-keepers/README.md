# Store two bar magnets as a pair: unlike poles side by side, a keeper across each end

School and college science technicians put the bar magnets away after every magnetism lesson. Steel and Alnico bar
magnets slowly lose strength when they are stored loose, or side by side with like poles together. Stored as a pair,
side by side with the north end of each beside the south end of the other and a soft-iron keeper laid across each
end, magnets and keepers form one closed magnetic loop and the magnets keep their strength; that is why school bar
magnets are sold in pairs with keepers. Here a wooden storage tray (white floor, dark rims) stands on the midline,
two two-colour bar magnets (north half red, south half blue) lie on the table on either side of it, red ends pointing
outward, and two dark grey iron keepers lie nearer the arms. The start is a mirror image about the midline, so each
magnet has to be turned a quarter turn as it goes in, and the two must be turned the same way in the world: turning
them the mirror-image way, the natural symmetric move, puts both red ends at the same end of the tray (like poles
side by side), which scores 0. The score is what the technician checks with a rule and a look from above: both
magnets lying in the tray along its length, side by side, red beside blue, and a keeper touching both magnets at each
end.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the magnets' and keepers' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each magnet as one rigid body of two coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and loose boxes (the keepers), the standard `task_score` scorer, and one helper module
next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a one-colour bar has no
visible poles. `mechkit.py` (imported by `task.py`) builds each magnet from two boxes (the red north half and the
blue south half) as one loose rigid body through a hook on `yam_arena.scene`, and reports its pose (base-frame
origin and orientation) to the scorer, which reads from it which way the red end points. It changes nothing else the
embodiment does, and it stops the run with an error if a magnet was not built as a compound body. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the two bar magnets away in the wooden tray as a pair: lying side by side along the tray, one end of each toward you and the other away from you, with the red (north) end of each next to the blue (south) end of the other. Then lay one grey iron keeper across each end of the pair, touching the ends of both magnets." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/magnet-pairs-keepers/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the two bar magnets away in the wooden tray as a pair: lying side by side along the tray, one end of each toward you and the other away from you, with the red (north) end of each next to the blue (south) end of the other. Then lay one grey iron keeper across each end of the pair, touching the ends of both magnets." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/magnet-pairs-keepers/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`zero_reason`, `like_poles`, `n_stored`, `paired`,
`pair_across_mm`, `pair_along_mm`, `far_end_closed`, `near_end_closed`, `moved_mm` per part; per magnet `stored`,
`at_rest`, `low_mm`, `level_deg`, `heading_deg`, `inside_rims`, `in_tray`, `red`; per keeper `on_floor`,
`across_err_deg`, and at each end `gaps_mm` and `cover_mm` against both magnets; and every part's pose). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset (mechkit's reset counter, or the simulator putting everything back on its marks in one step); the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from where the four parts lie at the last
step, and a part counts only when it is at rest: none of its points (a magnet's or keeper's two end-face centres)
has moved more than 2 mm in the last 0.5 s (positions, not the simulator's velocities, which chatter while a part is
gripped).

A magnet is **stored** when it lies on the tray floor (lowest point at most 3 mm above it), its length within 5
degrees of level and within 10 degrees of the tray's length seen from above, its whole outline inside the rims (any
face may be down: a magnet on its side is as well stored, magnetically, as one lying flat). Its **red end** points
along the tray either away from the arms or toward them. The magnets are **side by side** when both are stored with
their centres at most 60 mm apart across the tray and within 15 mm of each other along it; touching counts (a real
unlike pair pulls together unless a spacer is fitted). A keeper **closes** an end of a side-by-side pair when it is
at rest on the floor (lowest point at most 3 mm above it, its length within 10 degrees of level), lies across the
tray (its length within 15 degrees of across seen from above), and sits beyond that end: on the line along the tray
through each magnet's end-face centre, its near edge is from 2 mm inside the end face (contact) to 8 mm beyond it,
and it spans at least 12 mm of each end face's width. With the magnets' end faces 45 mm apart across the tray, the
8 mm band also limits how crooked a closing keeper can be (about 12 degrees).

| Outcome | Score |
|---|---|
| Both magnets lengthwise in the tray (centre inside the rims, lowest point under 20 mm above the floor, within 30 degrees of the tray's length) with their red ends pointing the same way: like poles side by side, whatever the keepers do | 0 |
| No magnet stored | 0 |
| One magnet stored, or both stored but not side by side (keepers count for nothing without a pair) | 0.25 |
| Both stored side by side (their red ends then point opposite ways: unlike poles adjacent), no end closed | 0.5 |
| One end closed by a keeper | 0.75 |
| Both ends closed | 1.0 |

The like-poles zero uses a looser "lengthwise in the tray" than "stored", so a like pair cannot escape it by being
left slightly crooked, spread apart, propped up or one on top of the other. Either magnet may point either way,
provided the other points the opposite way (both unlike arrangements score 1.0), and either keeper may close either
end. The trial ends early once 1.0 has held for 2.5 s. A simulator state with a non-finite number scores 0 for that
step and leaves the record of the trial as it was.

Unintended strategies earn nothing more than they should, as the tests show: the mirror-image move (like poles) is 0
with or without keepers; keepers laid on top of the magnets' ends, along the tray beside the pair, or standing on end
close nothing (0.5); both keepers at one end close one end (0.75); keepers around a lone magnet score 0.25; a magnet
held above the tray, standing on end, across the tray or propped on a rim is not stored; an unlike pair stacked one
on the other is one stored magnet (0.25); a part still moving does not count.

## One arm alone

In simulation (the GPU runs in 'Results in simulation', one trial each) the scripted demo scores 0.25 with either arm
idle, as the offline kinematic check with `--one-arm` predicted. With the right arm idle (`DEMO_IDLE_ARM=right`) the
left arm stores magnet A and lays keeper 1 across the far end of a lone magnet: one magnet stored, 0.25. With the left
arm idle (`DEMO_IDLE_ARM=left`) the right arm stores magnet B and lays keeper 2 across its near end: 0.25. In both
runs the idle arm's magnet and keeper stay on their starting marks.

Reach (inverse kinematics of the arm's URDF with its link collision proxy; the point between the finger tips 9 mm
above the table): gripping straight down, the left arm reaches at most y -16.1 cm at x 22 cm, 69 mm short of the
other magnet's near end (y -23 cm), and the right arm is its mirror image. Leaning its gripper 20 degrees toward the
far side, though, with the fingers closing across the magnet, an arm does reach a grip 10 mm in from the far magnet's
near end. So one arm alone could in principle pick the far magnet by its end with a tilted grip, then do everything
in turn; the scripted one-arm runs do not try that, and we do not claim that one arm cannot score 1.0. Both arms are
used naturally: each brings in the magnet and the keeper on its own side.

The pole decision is the policy's, not the kinematics': each arm can turn its magnet either way. Picked across its
width (gripper yaw +90 or -90 degrees) and laid along the tray (yaw 0), the left or the right magnet ends with its red
end either away from the arms or toward them; all four moves keep the interpolated path's inverse-kinematics error
under 0.1 mm with at least 18.9 degrees to every joint limit.

## Demo and run limit

The scripted demo takes 46.9 s (938 control steps). Both arms grip their magnet at its middle across its 20 mm width
(fingers along x, commanded 12 mm, closing over 1.5 s), the tool point 17 mm above the magnet's underside (pad tips 4
mm), lift it to 12 cm and, while carrying it to the tray over 2.5 s, turn it a quarter turn clockwise seen from
above: the left magnet's red end swings from +y to away from the arms, the right one's from -y to toward them. Each
magnet is lowered the last 30 mm over 2 s to 2 mm above the floor, centred 22.5 mm either side of the midline at x
26.0 cm (a 25 mm gap), and let go by opening only to 36 mm, so the finger between the two magnets stays 9 mm clear of
the one already there. The keepers are gripped across their 74 mm length (commanded 66 mm) with the tool point 21 mm
above their underside (pad tips 8 mm), set down across the tray, keeper 2 aimed 4 mm off the near end faces (centre
at x 19.6 cm) and keeper 1 centred at x 32.05 cm (aimed 4 mm off the far end faces, at x 32.4 cm, then moved 3.5 mm
nearer with `TRIM_K_FAR` because on the GPU the magnets come to rest about 4 mm short of their aim; see below), and
let go by opening to 88 mm (the pads 11 mm inside the side rims). A keeper overhangs the pair by
only 4.5 mm a side, so a few millimetres of sideways sag puts a pad in line with a magnet's end; gripping this high
brings the pads' narrow tips (about 15 mm wide, against 27 mm at the tool point) level with the magnets' top corners,
and a pad clears them unless the keeper lands within 0.7 mm of the end faces (2.6 mm with a 17 mm grip). The clock:
left stores magnet A (0-16.5 s); right, holding magnet B since 7.2 s, stores it once the left hand is up (16.5-25.8
s); left fetches keeper 1 meanwhile and lays it across the far end once the right hand is back over keeper 2
(27.3-36.6 s); right lays keeper 2 across the near end (36.6-45.9 s); both hands go aside. The two hands are never
over the tray at the same time.

Offline (kinematic check): worst inverse-kinematics error 0.095 mm; smallest joint margin 14.6 degrees (the left
arm's joint 4, as it leaves keeper 1 for its aside pose); arm links at least 8.5 cm apart (capsule proxy), tool points at least
27.5 cm apart; pad tips never lower than 4 mm above the table. Success has held for its 2.5 s at step 941; the
scripted policy stops at step 999. The run limit is 1,800 steps (90 s).

On the GPU (L40S, one trial): the arm sags a few millimetres under each 0.17-0.23 kg part, so each part touches the
floor slightly before its slow descent ends. Both magnets came to rest on the floor about 4 mm short of their aim
along the tray (centres at x 25.6 cm, y +2.5 cm and -2.5 cm, within 1.1 degrees of the tray's length, magnet A's
red end away from the arms and magnet B's toward them); keeper 1 lay across the far end with gaps of 3.2 and 3.6 mm,
keeper 2 across the near end with gaps of 1.5 and 0.5 mm (band -2 to 8 mm), each spanning the full 20 mm of every end
face. (In an earlier run, before keeper 1's place point was trimmed 3.5 mm nearer, its far gaps were 7.4 and 7.8 mm,
too close to the 8 mm edge; the magnets landed the same way in both runs.) Score 1.0 was reached at about step 860
and the trial ended at step 910 (45.5 s), once it had held for 2.5 s, with the last keeper's gripper open and rising. To
rerun elsewhere, read `gaps_mm` and the magnets' `centre` in `TASK_TRACE`; if a part lands off its aim, move its place
point by minus the offset with `TRIM_A`, `TRIM_B`, `TRIM_K_FAR` and `TRIM_K_NEAR` in `task.py`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: magnet A back on the table centred at x 22.0
cm, y +28.0 cm, along y with its red end toward +y; magnet B at x 22.0 cm, y -28.0 cm, red end toward -y; the
keepers flat at x 13.0 cm, y +22.0 cm and y -22.0 cm, length along y. The tray is fixed. Nothing is consumed.

## Runtime per trial

Measured on an L40S, from launch to score: about 2 minutes for the scripted demo (910 control steps) and about 3.5
minutes for a Dream-1 trial that runs the full 1,800 steps, each including the simulator's start on a warm instance
(under 30 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A tray of 3 mm plywood painted white, 182 x 138 mm, with 6 x 13 mm dark-stained beech strips glued around its edges
(170 x 126 mm inside), screwed or clamped to the table centred 26 cm in front of the arm bases. A matched pair of 100
x 20 x 15 mm steel school bar magnets with red north and blue south halves (as the instruction names them), and two
soft-iron keepers about 20 x 74 x 15 mm; other sizes work if the scoring distances are scaled to them. Real magnets
differ from the simulation in three ways, none of which changes what scores: a keeper set down within a few
millimetres is pulled onto the pair (touching counts); an unlike pair 25 mm apart may pull together (touching still
counts as side by side); a like pair pushes apart (still 0: the like-poles rule needs only both magnets lengthwise in
the tray, not side by side). If the grippers' fingers or screws are steel, fit non-magnetic pads so the magnets do
not cling to them. The parts weigh 0.17 to 0.23 kg, so trials are safe. Score from a top photo over a rule
(positions, directions, which end is red) and a feeler gauge or rule for the keeper gaps.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |

Settling check (`DEMO_HOLD_ONLY=1`, 160 steps, no plan): no part moved (0.0 mm). In the Dream-1 trial (1,800
steps, no policy error) none of the four parts moved from its starting mark (at most 0.1 mm).
