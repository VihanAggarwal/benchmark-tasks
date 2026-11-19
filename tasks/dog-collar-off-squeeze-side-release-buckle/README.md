# Take the collar off the dog manikin

A life-size CPR dog manikin (6.8 kg, loose) lies on its side on the table, head toward the robot. It wears a red
3/4 in nylon collar closed with a plastic side-release buckle on top of its neck. The robot must unclip the buckle,
draw the open collar out from under the neck, lay it on the table beside the manikin and let go.

Instruction (task.json and `task.py` `INSTRUCTION`, identical): "Take the collar off the dog manikin: unclip the buckle, draw the collar off its neck, lay it on the table beside the manikin and let go of it."

The manikin is a Simulaids CasPeR CPR dog, item 100-5000 (about US$460 from medical-training suppliers). A cheaper
stand-in is any firm life-size lying-dog figure, for example a Douglas Sadie yellow Lab plush (about 760 mm, about
US$130), laid on its side and weighted to 5 to 9 kg with sandbags on its flank, if it passes the setup's checks (neck
round, the 25 mm gap under the neck, the head too big for the buckled loop). If a 25 mm block does not pass under the
neck, the head and the shoulder go on two 20 to 25 mm rubber pads outside the collar's path (one under the head at x
under 250 mm, one under the chest at x over 400 mm) and the check is repeated.

Status (GPU simulator, 19 Nov 2026, this version of the scene, with the manikin's ears, nose, eye, cheeks, shoulder
and tail and the dark grey female): the scripted demo scores 1.0, first reached at step 712 of the 1600-step limit and
held to the end of the run at step 723; with either arm idle it scores 0.0; in the settle check nothing moves
(manikin 0.1 mm); one Dream-1 trial at the full run limit scored 0.0. The setup screenshot and the videos are of this
version. Figures under "GPU results" below.

The buckle opens the way a real one does: two sprung prongs on the male half hook behind ribs inside the female, and
the male slides out only while both side tabs are pressed in at once. One hand pinches the female across its 29 mm
width, which presses both tabs flush; the other hand grips the male's exposed 22 mm and pulls it out. The buckled
loop (381 mm) is 54 mm longer than the neck's round and 34 mm shorter than the head's, so the collar can only come off
open, drawn round and out from under the neck.

## Why it is hard

1. **Two hands on one 56 mm part.** The pinch on the female and the grip on the male sit side by side on top of the
   neck; both hands work within a few centimetres of each other and of the manikin.
2. **Hold while the other pulls.** The tabs must stay in for the whole pull: pinch, let go, then pull, and both hooks
   catch again. Pulling the male alone drags the whole collar round the neck (the hand slips off it).
3. **A loose strap round a body.** Once open, the collar hangs round the neck; it has to be drawn out from under a
   neck 25 mm off the table without snagging the female half, then laid flat and let go.
4. **A loose, heavy manikin.** It is not fixed: nudging it 20 mm caps the score at 0.75, and dragging it 100 mm
   scores 0.

## Scoring

| Condition | Rule (sim) | Bench method (real trial) |
|---|---|---|
| Latch A: off the table | any part of the collar more than 3 mm below the table top or past its edge, at any step | the trial video |
| Latch B: manikin dragged | the manikin's nose tip or rump 100 mm or more (3-D) from where it started, at any step | tape marks at the nose and rump; ruler on the end photo and video |
| Nudged (caps at 0.75) | the nose tip or rump 20 mm or more from where it started, at any step | the same tape marks and ruler |
| Unlatched | the male 3 mm or more out of the female along her axis | pen line on the male at the mouth; ruler on the end photo |
| Apart | every part of the male 5 mm or more outside the female's outer box | no part of the male in the mouth; its prong tips 5 mm or more out |
| Off the manikin | every part of the collar (webbing, both halves, D-ring, tri-glide) 25 mm or more from the manikin's shapes | a 25 mm block passes between collar and manikin along the whole collar without moving either |
| Resting | no part of the collar over 40 mm above the table top, no body moving 5 mm/s or faster | ruler; no part moves 5 mm in the last 1 s of video |
| Hands off | both grippers' finger zones (a capsule round the last 38 mm of the fingers, larger than the fingers) 30 mm or more from every part of the collar | no finger touches the collar in the last 1 s; a 30 mm block passes between each gripper and the collar |

| Outcome | Score |
|---|---|
| A latch fired, or the male under 3 mm out (still buckled) | 0 |
| Unlatched at the last step, not apart | 0.25 |
| Apart at the last step, some part of the collar within 25 mm of the manikin | 0.5 |
| Apart and off the manikin at the last step, but resting or hands off fails on a step of the last 1 s, or the manikin was nudged | 0.75 |
| Apart, off the manikin, resting and hands off on every one of the last 20 steps (1 s), manikin never nudged | 1.0 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0; every end state falls in exactly one. Success: 1.0 held for 1 s. The score
reads poses only; nothing reads the tabs or the order of moves. A step whose states cannot be read restarts the 1 s
window and caps the score at 0.75; unreadable arms are never "hands off".

| Bypass | What stops it |
|---|---|
| Pull the male without pressing the tabs | the hooks stay behind their ribs: still buckled, 0 |
| Press the tabs, let go, then pull | the springs push the prongs back out; the hooks catch again |
| Slide the closed collar off over the head | the head (415 mm round) is wider than the loop (381 mm) |
| Leave the open collar round the neck or under it | some part within 25 mm of the manikin: 0.5 (a link lying on the table under the 25 mm neck gap is under 23 mm from it) |
| Lift the manikin's head to slip the collar off, or drag the manikin | nose or rump 20 mm off its mark: at most 0.75; 100 mm: 0 |
| Hold the collar up in the air, or keep hold of it | not resting or not hands off: 0.75 |

## The model

All parts are rigid bodies built with `mechkit.py` (copied unchanged from the team kit; contact only, no software
rule for the buckle):

* **Manikin** (`parts.dog_shapes`): one loose 6.8 kg body of cylinders and boxes to the maker's outline (711 x 584 x
  254 mm): muzzle, head (132 mm across behind the ears), neck (along x from 265 to 380 mm, 104 mm across, axis 77 mm
  up, underside 25 mm off the table), chest, flank, rump, four legs. So it reads as a dog in the top camera it also
  has look-only parts (all of them colliders, like every mechkit shape): a floppy upper ear lying over the skull from
  its back and hanging down its +y side (x 206 to 257 mm), the lower ear flipped out flat on the table behind the head
  (x 166 to 218, y -55 to -124, 1 to 4 mm up), a 48 mm black nose with its leather over the muzzle tip (front still at
  x 88, the tape mark), a 20 mm eye toward the back of the skull, cheeks between muzzle and skull (x 165 to 190), a
  deeper shoulder on the chest's leg side (x 420 to 570, to y 132) so the body tapers toward the rump, and a tail
  lying on the table out of the rump and along behind the hind legs (x 782 to 880, y 3 to 227). Every one of them is
  at x 257 mm or less or at x 420 mm or more: at least 48 mm from the collar's side (x 305.5) and clear of the
  neck (x 265 to 380), where the collar, the buckle and both grippers work. Drawn a darker tan than the table, with
  dark brown ears and a black nose and eye; the colour is not the maker's (no colour data found).
* **Collar** (`COLLAR_PART`): one loose articulation. Its base is the female housing (floor, roof, back block, two side
  walls each with a slot open from the mouth to 19 mm in and a catch rib inside behind it, 6 g), drawn dark grey so
  its seam with the black male shows. Thirteen red webbing
  links, 24.5 x 19 x 2 mm on a 25 mm pitch, hang from it on hinges along x (damping 0.0002 N m s/rad, limits 2.6 rad),
  drawn round the neck on the circle that makes the loop 381 mm. The tri-glide (black, 2.5 g) rides on link 2, the
  steel D-ring (25 x 20 mm, 3 g, drawn light silver) on link 13.
* **Male** (`MALE_PART`): a second small articulation (body and centre tongue, 4 g) with two prongs on vertical hinges
  (0.7 g each). Each prong's spring gives 2.5 N at its tab pressed flush (0.54 N m/rad). Tab 18 mm from the hinge,
  hook 26.3 mm: a tab 0.89 mm in turns the hook clear of its 1.3 mm rib; flush it turns 2.2 mm.
* **Bar bond** (`BAR_BOND`): the last link is glued to the male's webbing bar by an unbreakable mechkit bond (a
  FixedJoint across the two articulations).

Every figure marked est in `task.json` is a design value the simulation is built to; the setup's checks say how real
parts are matched to them.

## The demo (scripted, open loop)

`demo()` returns timed waypoints for both arms from the known start; nothing is read from the simulator. Both grips
close along world x and are rolled about x (`tiltgrip.py`, the same patch of `yam_arena.demo.grip_rotation` as the
cord-lock task): the left wrist 0.35 rad, the right 0.85 rad, leaning apart so the two hands fit side by side over the
56 mm buckle.

| t (s) | Left | Right |
|---|---|---|
| 0 to 6.5 | over the female, down either side (44 mm open), pinch to 27 mm across the 29 mm housing (5 N a finger: tabs flush), settle | over the male (36 mm open), wait |
| 6.5 to 13 | hold the pinch | down either side of the male, grip to 23 mm across 25 mm (5 N a finger), pull 40 mm along -y (2 s) |
| 13 to 17.5 | open, up 50 mm, aside | hold |
| 15.5 to 36 | parked | out past the neck's side, down to just over the table, draw the collar out from under the neck along -y at about 40 mm/s to y -445 (the wrist keeps its yaw, so the strap stays square to the neck and the female follows it straight under; it rolls upright between y -130 and -280), lower the male to the table, open, up, aside |

Pads are kept off the manikin: the left pads' lowest corner 2.6 mm over the neck, the right's 3.5 mm. The demo is 718
steps (35.9 s); the run limit is 1600 steps (80 s).

Environment switches: `DEMO_HOLD_ONLY=1` (no plan: the settle check), `DEMO_IDLE_ARM=left|right` (the one-arm check),
`DEMO_VARIANT=early|short` (counterfactuals, never the submitted demo: `early` lets go of the tabs before the pull,
`short` stops the draw at y -250), `TASK_TRACE=<file>` (one JSON line per control step).

```
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 --scorer task_score
```

`object_jitter` must be 0: the bar bond is authored at the declared spawn poses.

## Results

| Run | GPU (sim, 19 Nov 2026, this version) | Offline model (prediction) |
|---|---|---|
| Settle (no plan) | 0.0, nothing moved (still buckled, as it should be) | 0.0 |
| Demo | 1.0 (from step 712; run ended at step 723) | 1.0 (held at step 713) |
| One arm idle: left idle / right idle | 0.0 / 0.0 | 0.0 / 0.0 |
| Dream-1 (1600 steps) | 0.0 | not modelled |
| `early` / `short` counterfactuals | not run | 0.0 / 0.5 |

### GPU results

From the full GPU run of 19 Nov 2026 on this version of the scene (the manikin with its look-only parts and the
present colours); the setup screenshot and the videos come from the same run.

* **Settle** (160 steps, no plan): score 0.0, as expected for a buckled collar. The buckle stayed clicked (slide
  -0.32 mm, prongs at 0), the bond held, the manikin's nose and rump moved 0.1 mm, the lowest collar point stayed
  3.1 mm or more above the table.
* **Demo** (1.0): the left pinch pressed both tabs (prongs 0.072 and 0.079 rad, tabs flush) by step 118; the male came
  out 37.5 mm by step 253 with the female still held; when the left let go the female dropped onto the table on the
  manikin's +y side, then followed the strap straight under the neck (its x stayed between 317 and 327 mm), crossed
  y 0 at about step 505 and ended at y -143 mm, the male at y -423 mm. The score reached 0.75 at step 523 (collar
  25 mm clear of the manikin) and 1.0 at step 712, once both hands had been away for the whole last second. Last
  row of the trace (step 723): buckle apart, male 227 mm clear of the female's box, collar off the manikin
  (59.2 mm from its shapes) and resting on the table (highest point 18.1 mm, lowest 0.0 mm, speed 0.0 mm/s), finger
  zones 141.5 mm from the collar (hands off), manikin moved 0.3 mm (never nudged), prongs back at 0, bar bond intact,
  no latch.
* **One arm idle**: left idle 0.0 (the right pulled the male with the tabs out; it never came out, slide under
  0.3 mm); right idle 0.0 (the left pressed the tabs, prongs 0.073 and 0.079 rad, but nothing pulled the male: slide
  0.23 mm, under the 3 mm unlatch line). The manikin moved 0.2 mm at most in either run.
* **Dream-1** (one trial, 1600 steps, the manikin instruction accepted by the Dream service): 0.0. Both grippers
  closed and drifted toward the robot; neither came within 228 mm of the collar, which stayed buckled where it
  started; the manikin moved 0.2 mm.

The offline numbers come from `offline_collar.py`, a stand-in used only without a simulator (`tools/offline`
checker): grips detected from the commanded pad tips and openings, tabs pressed from the pinch gap, the buckle as
clicked / sliding / apart, and the webbing as a rope of 14 nodes 25 mm apart plus the two halves as rigid ends (no friction, no inertia) pushed out of the neck, head and
chest and up onto the table. It is a prediction, not a measurement.

## Tests

```
python3 -m unittest discover -s tests -v
```

47 tests (geometry and the catch, the look-only parts kept out of the collar's region, every tier, latches, bad states, bypasses, the offline model and full replays of
the demo, one-arm runs and counterfactuals, the right wrist's constant yaw while it holds the male, and the task text against the code). NumPy only; `yam_arena` from the
simulator or the offline stand-in.

## Assets

None: every shape is a mechkit box or cylinder defined in `parts.py`. No third-party meshes or textures.
