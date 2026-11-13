# Thread a shock cord through a spring cord lock, pressing its button to open it

Spring cord locks close the drawcords of hoods, jacket hems, backpacks and stuff sacks, and they come with no-tie
elastic shoelaces. The Lock Laces fitting guide says: "Press down on the top of the lock to open the passages, and
thread the ends of the laces through the passages with your other hand". When a cord slips out of its lock, threading
it back in is that act: one hand holds the button in against its spring while the other feeds a soft cord through a
hole that is open only while the button is held.

Here the robot does it with a real-size lock and cord on the table: press the button of a loose Ravenox Extra Large
cord lock, push one end of a 320 mm length of 5/16 in. shock cord through the lock until at least 40 mm stands out of
the far side (with at least 40 mm left on the near side), then let go of both, so that the lock lies on the table
gripping the cord by its own spring.

## Why it is hard

1. **The passage exists only while the button is held in.** The passage is the overlap of the body's hole and the
   plunger's hole: 4.4 mm + q, where q is how far the button is pressed in. With the button out it is 4.4 mm against
   a 7.9 mm cord, so the cord's round end gets at most 2.7 mm in (the 2 mm face plus 0.67 mm of its cap): below the
   5 mm first tier. The passage admits the 7.9 mm cord only while the button is held at least 3.5 mm in (5.25 N),
   and leaves 1.75 mm per side only at flush (7 mm, 10.5 N). Pushed at the unpressed lock, the end only shoves the
   loose lock away (C1). With the lock braced, a round end aimed at the 4.4 mm slot meets the plunger's edge 34 to
   70 degrees off its axis and could wedge it open with about 3 N of push (friction 0.5); but a straight cord end
   can push the plunger only 3.5 mm in, where the passage equals the cord, and the scorer pays nothing unless the
   button was pressed at least 5 mm in before a cord end was 5 mm in and stayed that far in until the cord was
   through (Pressed and Press held, below). Wedging also leaves the
   5.25 N pinch dragging on every millimetre that follows, which buckles any free length over 22 mm. Pressing the
   button is the way through.
2. **The squeeze that opens the lock also holds it.** The lock weighs 8.8 g and lies loose; it slides on the table at
   about 0.04 N, so an uneven squeeze shoves it away. Pressing the button flush
   takes 10.5 N over 7 mm, and between 3.5 and 7 mm the spring pushes back 5.25 to 10.5 N for as long as the end is in
   the passage. The YAM gripper's finger drive is 5 N per mm of position error per finger (50 N cap): the demo
   commands 26 mm across the 34 mm pressed body, 20 N a finger at the stop. The lock may be squeezed on the table or in
   the air; nothing requires a lift.
3. **The fit.** Pressed flush, the passage is 11.4 x 13 mm and 13 mm deep: 1.75 mm per side along the plunger's
   travel and 2.55 mm across it for the 7.9 mm cord. A straight 20 mm length of cord passes only within about 14
   degrees of the hole's axis (7.9 / cos t + 13 tan t <= 11.4).
4. **The cord is a cantilever when it is pushed.** For the end to come 40 mm out of the far face in one push, at least
   13 + 40 = 53 mm of cord must stand free of the hand, whatever the gripper: bending stiffness EI about 1e-3 N m2,
   so it buckles at pi^2 EI / (4 L^2) = 0.88 N and droops 0.4 mm. Two straight-down YAM grippers cannot come close
   enough to each other for that (their wrists collide below about 170 mm apart), so the demo tilts both; the cord
   then leaves the right pads 5.4 mm ahead of the TCP, and the demo's single push leaves 75.6 mm free: droop 1.8 mm
   (against 2.55 mm of clearance across the passage), buckling at 0.43 N. A catch on any edge folds the cord. A
   shorter free length needs a regrasp, or the pressed lock slid along the cord; every method still has to hold the
   passage at 3.5 mm or more of travel while the end passes the 13 mm depth.
5. **Order and release.** Let go before the cord is through, the lock springs back to its 4.4 mm passage. With the
   cord through and both grippers clear, the plunger rests on the cord: in simulation 3.5 mm in (button 3.5 mm out),
   the spring pressing the cord with 5.25 N, a friction hold of about 5 N against the lock's 0.086 N weight.
6. **Two hands, and why one arm is predicted to fail.** Each arm can reach both parts with a tilted grip, so reach does
   not make the task two-armed. The only one-arm route is to squeeze the lock and slide it, on the table, onto the end
   of the cord lying there. The lying end meets the face hole's lower edge (1.75 mm up) 2.2 mm below its 3.95 mm
   centre, a contact 34 degrees above horizontal, and the plunger's lower rail (3.5 mm up) at the same angle: the
   edges try to lift the end into the passage. Lifting the end of the lying cord by 1.75 to 3.5 mm takes about
   0.024 N; through a 34 degree contact at friction 0.5 the push along the table must reach about 0.19 N, while the
   whole loose cord slides on the table at about 0.08 N. So in simulation the cord is predicted to be pushed along,
   not threaded (C2, C3); the one-arm runs in the table below test it.

None of this goes away with a fully explicit instruction: the instruction states the method.

## Scoring

Every step the scorer reads the cord's centre line (its two ends and 16 segment centres), the lock's pose and the
plunger's travel, in the lock's frame. The scoring window is the body hole's outline (11.4 x 16.5 mm), which contains
the 11.4 x 13 mm passage, across the lock's 13 mm depth. Through: the centre line crosses the lock's mid-plane inside
the window exactly once. Out is measured along the cord, on both sides, from where it first reaches that side's face
to its end; the cord's out is the smaller side. Either end of the cord may be threaded.

Released: every point of both grippers' finger zones (pad thickness and width included, as rigstate.py defines them)
is at least 30 mm from the lock and from the cord; an arm that cannot be read counts as not released, and so does a
part whose distance cannot be computed. Let go: released on every step of the last 1 s (20 steps). Pressed
(latched): at some step while no cord end was 5 mm in and the cord was not through, the plunger was at least 5 mm in
(the button at most 2 mm out); a cord end wedged into the 4.4 mm slot can push it only 3.5 mm in, so the instruction's
"press in the button" is scored, and every line above 0 needs Pressed. Press held: from the first step a cord end is
5 mm in until the first step the cord is through, the plunger must stay at least 5 mm in on every step (the button at
most 2 mm out). A press let go in that window (press lost, kept until a reset) caps the score at 0.25: pressing,
releasing and then wedging the cord in earns nothing past the first stage. The score is the highest line that holds;
a held cord or lock earns the first stage at most.

| Score | Rule | Real-bench check |
|---|---|---|
| 1.0 | let go, and on every step of that second, all at once: through with out at least 40 mm; the lock's lowest point within 2 mm of the table; the plunger at least 1 mm in (button at most 6 mm out); nothing moved more than 2 mm. The trial ends once this has held 1 s | the 53 mm pen ring (13 + 40) at or past the entry face (side video); a steel rule along the cord on the near side; the depth rod of a caliper on the button (6.0 mm or less out; the empty lock reads 7.0); the top video for the grippers 30 mm clear (a 30 mm card for reference) and 1 s still; the side video for the lock on the table |
| 0.75 | let go, and through with out at least 40 mm | the 53 mm ring and the steel rule, grippers clear |
| 0.5 | let go, and through with out at least 10 mm | the 23 mm ring (13 + 10), grippers clear |
| 0.25 | at some point in the trial a cord end was at least 5 mm into the window (latched) | the 5 mm ring at or past the entry face |
| (all above 0) | Pressed: the plunger at least 5 mm in while no cord end was 5 mm in and the cord was not through (latched) | the top video: the button at most 2 mm out before the 5 mm ring reaches the entry face (it stands 7 mm out at rest) |
| (0.5 and up) | Press held: the plunger at least 5 mm in on every step from the first cord end 5 mm in until the cord is first through; a press lost caps the score at 0.25 | the top video: the button at most 2 mm out for as long as the cord end is in the lock, until the end is past the lock's middle (the 5 mm ring 1.5 mm past the entry face) |
| 0 | doing nothing, or none of the above | |

There is no whole-trial zero: nothing in the scene can be harmed.

Why the plunger reading is the grip: with both grippers clear, only the cord in the passage can hold the plunger in.
Held 1 mm or more in, the spring presses the cord with 1.5 N or more: a friction hold of 1.5 N or more at the pair's
friction of 0.5 (the cord's 0.6 and the lock's 0.4, averaged), 17 times the lock's 0.086 N weight. The real trial
reads the same plunger with the caliper.

## Shortcut and robustness checks

Each run is the demo with one change (`DEMO_VARIANT=...` or `DEMO_IDLE_ARM=...`). "Predicted" is what
`offline_cordlock.py` gives (the design's own geometry, no physics). Rows C8 and the two idle-arm runs were also
run on the GPU and give the measured score; the other variants were not run on the GPU.

| # | Run | Predicted | Why |
|---|---|---|---|
| C1 | `naive`: the right arm pushes the cord straight at the unpressed lock where it lies | 0 | the end stops 2.7 mm in at most; the loose lock slides away |
| C2 | `slide_right`: left idle; the right squeezes the lock and slides it along the table onto the lying cord's end | 0 | point 6: the cord is pushed along |
| C3 | `slide_left`: right idle; the left does the same | 0 | as C2 |
| C4 | `short`: both arms, the cord gripped 30 mm from its end, one push to the closest pose | 0.25 | the end reaches 10.5 mm in, short of the far face |
| C5 | `keep_lock`: the lock is never let go | 0.25 | a held part earns the first stage only |
| C6 | `keep_cord`: the cord is never let go | 0.25 | as C5 |
| C7 | `early_lock`: the lock is set down and let go before the push | 0 | the passage closes to 4.4 mm (as C1) |
| C8 | the demo (both arms) | 1.0, success held at step 535 (GPU: 1.0) | |
| C9 | `in_place`: the lock squeezed where it lies; the cord carried to it and pushed in | 1.0 | the task needs no lift and no carry |
| - | `DEMO_IDLE_ARM=left` / `right`: the demo with one arm at its ready pose | 0 / 0 (GPU: 0.0 / 0.0) | the other arm alone neither presses the lock nor moves the cord |

## Scripted demo (open loop)

`demo()` reads nothing but the fixed start layout. Both grippers close along world x and are rolled about world x for
the whole demo (`tiltgrip.py`): the left 0.5 rad with its tips toward -y, the right 0.6 rad with its tips toward +y,
so the wrists lean apart. At the push's end the two arms' links and housings are 33 mm apart (offline capsules); the
fingers are plates in different x bands, 5 and 9 mm apart.

| Time (s) | Left arm (lock) | Right arm (cord) |
|---|---|---|
| 0 to 7.0 | over the lock, slowly down either side of its ends (TCP 18 mm up, 8 mm above the hole's centre), squeeze to a 26 mm command: the button goes flush and the holes line up | over the cord 81 mm from its end, slowly down, grip it (closed on the 7.9 mm cord; its centre 12 mm below the TCP) |
| 7.0 to 13.0 | lift 30 mm, carry the squeezed lock to body centre (302, 40), set it down on the table, still squeezed: hole centre x 300, near face y 33.5 | raise the cord to the hole's height (cord at 10.45 mm, hole centre 10 mm), bring its end 26.5 mm short of the near face, wait |
| 12.5 to 19.0 | hold | push once along +y, 88 mm in 6 s: the end comes 48.5 mm out of the far face and about 258 mm stays on the near side; the right pad's front corner stops 6 mm short of the near face |
| 19.0 to 24.0 | open to 60 mm over 1.5 s (the plunger follows the pad out until the cord stops it 3.5 mm in), rise leaning back along +y, move away | hold |
| 22.0 to 29.0 | hold clear | open to 30 mm over 1.5 s (the cord drops to the table either side of the lock), rise leaning back along -y, move away, hold clear |

The demo ends in the instructed state and breaks no instruction clause. Offline, success holds from step 535 of the
580-step plan (run limit 900).

## Fidelity

PhysX at 240 Hz, control at 20 Hz. Simplified: the lock's spring is a prismatic joint, 1.5 N/mm from the free
position with no preload (10.5 N at flush); the cord is 16 rigid 20 mm capsules (16-sided colliders, radius 3.95 mm)
on bending joints of 0.05 N m/rad, so it neither stretches nor flattens and holds the button 3.5 mm out, against the
1.0 tier's 6 mm limit: a real cord flattened by the pinch sits further out, and the limit passes one squashed from
7.9 to 5.4 mm. The lock's edges are square instead of rounded; masses are those of solid parts at 1.41 g/cm3. The
scorer measures the cord from its centre line's end points, which sit about 2.4 mm inside the colliders' rounded ends, so it
reads the cord slightly short and never long. The lock's walls and hole position, the plunger's size, hole and
travel, the spring rate and the cord's bending stiffness are our estimates of the named parts. Not modelled: the cord's braid texture and black tracer.

## Real-world build and reset

| Part | Real | Simulated |
|---|---|---|
| Cord lock | Ravenox Extra Large spring cord lock, black Delrin, stainless spring: 1.6 x 0.8 x 0.5 in. (41 x 20 x 13 mm with the button out), hole 0.65 x 0.45 in. (16.5 x 11.4 mm) | the same outside and hole; body 34 x 20 x 13 mm with 2 mm walls; plunger 28.5 x 16 x 9 mm with an 11.4 x 13 mm hole; travel 7 mm to flush; 1.5 N/mm; body 4.9 g, plunger 3.9 g |
| Shock cord | Tarpsnow TA-UC-SC3125, 5/16 in. polyester-sleeve synthetic-rubber shock cord, white with a black tracer, about 3.1 lb per 100 ft (46 g/m); 320 mm cut with a hot knife so both ends are sealed round | 16 capsules of 20 mm, radius 3.95 mm, 14.7 g, EI 1e-3 N m2 (60 mm held level droops 0.7 mm) |
| Pen rings | a fine permanent marker, 5, 23 and 53 mm from each end | (read from the centre line) |
| Reading | steel rule, caliper with a depth rod, side and top video | |

Reset: press the button, pull the cord back out and let the button spring out (7 mm out). Lay the lock on a long side
with its body centre at x 302, y 130, button toward +x, hole axis along y. Lay the cord straight along y at x 300
with one end at y -60 pointing to +y. Check that the pen rings can be read. Nothing is consumed.

## Files

| File | What |
|---|---|
| `task.py` | the objects, the hooks (success, score), the demo and its variants |
| `layout.py` | every dimension and threshold |
| `scoring.py` | the scorer (latched tiers, the end state, its 1 s window) as plain functions |
| `offline_cordlock.py` | the parts' kinematics for the offline checker and the tests only (never used in the simulator) |
| `tiltgrip.py` | tilted waypoints for the scripted demo |
| `mechkit.py`, `rigstate.py` | the lock as a body with one sprung slider; the arms' state for the release rule |
| `cablekit.py` | the cord as a capsule chain |
| `tests/test_scoring.py` | offline tests |

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20
DEMO_IDLE_ARM=left ...            # one arm idle
DEMO_VARIANT=naive ...            # a counterfactual (naive, short, keep_lock, keep_cord, early_lock, in_place,
                                  # slide_right, slide_left)
TASK_TRACE=/tmp/trace.jsonl ...   # one JSON line per scored step: out, in, travel, release, stillness
```

## Offline tests

`python3 -m unittest discover -s tests -v` (NumPy only). They recompute the geometry (passage 4.4 mm + q, masses from
volumes, the cord's mass and stiffness), check every tier and its near misses (an end 4.9 / 5.0 mm in; out 9.9 /
10.0 / 39.9 / 40.0 mm; 39 mm left on the near side; plunger 0.9 / 1.0 mm in; the lock 2.1 / 2.0 mm off the table;
motion 2.1 / 1.9 mm in the last second; a finger zone 29 / 31 mm from the part; the press 4.9 / 5.0 mm before the end
goes in), the held and hovering states, the exploits (a cord wedged through without the press scores 0; press, release, then wedge the cord in scores 0.25 at most; a press dropped mid-insertion caps at 0.25; and a press
made only once the cord is in does not count; a 7 mm partial threading with the tail on the entry side scores 0.25; the cord round the outside of the
lock; a doubled cord through the hole; a curled tail and a far end wrapped back round the lock, both measured along
the cord; the far side pulled back after 0.75; either end threaded), five unreadable-arm forms, a raising distance,
unreadable part states, resets, and the demo and every variant played through `offline_cordlock.py`.

Third-party assets: none. `cablekit.py` is copied unchanged from the program repository's accepted cable tasks, with
its own header.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
