# Turn a frying pan's handle in on a cooktop, clear of the edge and the hot elements

A frying pan with hot oil in it sits on the front left element of a four-element electric cooktop, its handle
pointing at the cook's side and sticking out about 15 cm past the cooktop's front edge, right where a toddler standing
at the stove can reach up and grab it. A toddler is about to come into the kitchen. The robot (standing where the cook
stands) has to turn the pan so its handle no longer sticks out, keep the pan on its burner, and let go. The left rear
element was just switched off and still glows; the right front element is lit under a saucepan. Turned straight back,
the handle would rest over the hot left rear element; turned toward the middle, over the lit right front one. Only
diagonally back to the right (41 to 68 degrees right of straight back, the handle's full width counted) is it inside
the cooktop, turned toward the back and clear of both. The short way round, past the front, runs the handle into the
saucepan, so the pan has to be swung the long way, 210 degrees, through the left and the back.

**Revision 2 (after the last review):** the reviewer saw that the handle at the start, pointing out past the
cooktop's left side edge, looked safe and out of a toddler's reach. The handle now starts pointing toward the cook's
side, 30 degrees to the left of straight toward the robot, its end 149 mm past the cooktop's FRONT edge (the edge
nearest the robot, where a child stands). To keep that start reachable and clear of the left arm's base, the cooktop
moved 3 cm farther from the robot (front edge at x 0.100 instead of 0.070) and 17 cm to the robot's left (its middle
at y +0.170); the elements, pan and saucepan moved with it, so the safe sector and every scoring threshold are
unchanged. The scorer also gained one rule, which only makes it stricter: the handle must be turned toward the back
(its end farther from the front edge than where it leaves the pan). It closes a 3 degree sliver (112 to 114 degrees
to the right of straight back) where a handle still pointing at the cook's side ended 12 mm inside the front edge.
The demo is the same open-loop swing, now from +150 degrees instead of +90.

## Why it matters

Pulling a pan down by a handle that sticks out past the edge of the stove is a classic way young children are scalded;
fire-safety and children's-hospital guidance is to turn handles inward, away from the edge (NFPA cooking safety tips;
Nationwide Children's Hospital, cooking safety). Turned in, a handle must not rest over another element that is on or
still hot: it heats up and burns the next hand that grabs it. The task tests whether a robot applies the rule
without creating the second hazard, and whether it can swing a pan of hot oil round on its burner by the handle,
without tipping it or sliding it off.

## Sources

- Cooktop: Frigidaire FFEC3025UB 30 in electric smoothtop. 30 5/8 x 21 3/8 in; elements: left front 6 in, left rear
  9 in, right rear 6 in, right front 6/9 in expandable (retailer specification pages, e.g.
  https://www.ajmadison.com/cgi-bin/ajmadison/FFEC3025UB.html,
  https://www.canadianappliance.ca/product/Frigidaire_FFEC3025UB_Electric_Cooktop_Frigidaire_FFEC3025UB.html).
- Pan: Tramontina Professional 10 in fry pan, 80114/535DS: 10 in across the top, 18 3/4 in overall, 2.05 lb
  (e.g. https://www.acehardware.com/departments/home-and-decor/kitchen-utensils-and-gadgets/cookware/6239909).
- Guidance: NFPA cooking safety tips ("turn pot handles toward the back of the stove", reprinted at
  https://www.ishn.com/articles/87803-nfpa-offers-cooking-safety-tips-for-the-thanksgiving-holiday-11-25);
  Nationwide Children's Hospital, cooking safety
  (https://www.nationwidechildrens.org/research/areas-of-research/center-for-injury-research-and-policy/injury-topics/home-safety/cooking-safety).

Every figure not listed above is an estimate (task.json "fidelity"): the element centres, the pan's base diameter
(192 mm), depth (48 mm), wall angle, handle rise (12 degrees) and section (24 x 16 mm), the saucepan, and the
pan's friction on the glass (0.3).

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Arm bases | x 0, y +0.305 (left) and -0.305 (right); each base reaches about 46 mm in front of x 0 |
| Cooktop glass | x 0.100 to 0.643, y -0.219 to +0.559, 6 mm proud of the counter (front edge = the cook's side) |
| Left front element (the pan's) | centre x 0.250, y +0.370, 6 in |
| Left rear element (hot, just off) | centre x 0.480, y +0.365, 9 in |
| Right front element (lit, saucepan) | centre x 0.250, y -0.025, 9 in |
| Right rear element (off) | centre x 0.480, y -0.030, 6 in |
| Pan at the start | base centre on the left front element; handle at +150 degrees (toward the robot, 30 degrees to its left), its end at x -0.043, y +0.539: 149 mm past the front edge, 10 mm inside the left edge, about 73 mm clear of the left arm's base (capsule model) |
| Safe handle directions (pan centred) | -68 to -41 degrees (0 = straight back, +x; negative = to the robot's right); no other direction |

The pan is one loose rigid body (mechkit compound): base, 20 flat wall segments, a 3 mm layer of oil and the handle
(in a real trial the oil is an amber silicone disc 184 mm across and 3 mm thick, of the same 92 g, bonded to the
pan's floor: no liquid);
its centre of mass sits 48 mm toward the handle, as on a real pan. The cooktop (glass and the four rings) and the
saucepan are fixed. Nothing is moved or latched by code.

## Scoring (scoring.py)

Read at the end, except the failures, which are read every step.

- Safe handle: seen from above, every part of the handle (its full 24 mm width, rim to end) inside the cooktop's
  outline, the handle turned toward the back (its end farther from the front edge than where it leaves the pan), and
  every part at least 20 mm outside the rings of the two hot elements.
- 1.0: safe handle, pan centred (base centre within 30 mm of the element's), flat (3 degrees, on the glass), both
  grippers 30 mm or more from it for the last 20 steps, and still (2 mm over the last second).
- 0.5: safe handle and flat, but still held, 30 to 60 mm off centre, or not yet still.
- 0: anything else, including a handle past an edge or over a hot element, and the whole run if at any moment the
  pan tilted past 10 degrees (the oil spills), lifted more than 25 mm, or went more than 60 mm off its element.

Partial credit stops at 0.5 on purpose: a handle past the edge or over a hot element is the hazard itself, so no
state with it earns anything. In a real trial: a top-down photo with the cooktop's outline, the two hot rings and
their 20 mm borders marked, and a mark at the element's centre; tilt and lift from the side video.

## Real trials (no cooktop, no heat, no liquid)

The real Frigidaire cooktop costs about US$1,000, so real trials use a cold stand-in that keeps what the task depends
on: the cooktop's outline, where its hot elements are, and a smooth glass surface for the pan to slide on.

- Surface: a Quartet Infinity G3624B black tempered-glass dry-erase board (frameless, 36 x 24 in = 914 x 610 mm,
  about US$100; https://quartet.com/p/dry-erase-boards/glass-boards/infinity-glass-dry-erase-boards-black-surface),
  laid flat on the counter. The pan's aluminium base slides on its tempered glass as on the cooktop's glass-ceramic:
  both are smooth, hard glass, and the simulation uses a static friction of 0.30 (est.). Check before trials: with
  the board tilted up, the empty pan (disc fitted) starts to slide at 15 to 19 degrees (static friction 0.27 to
  0.34); if it does not, clean the glass and the pan's base.
- Outline: the cooktop's 778 x 543 mm outline, centred on the board (front edge at x 0.100, middle on y +0.170), marked
  with 6 mm white vinyl tape laid outside the line. The glass beyond it (68 mm at each side, 33 mm front and back)
  does not count: the handle must stay inside the tape.
- Elements: vinyl rings at the centres in the geometry table (6 in = 152 mm, 9 in = 229 mm across). The two hot ones
  (left rear, right front) are solid red discs with a thin red line drawn 20 mm outside each, the scorer's clearance;
  the other two are 6 mm white ring outlines. The left front ring carries the pan; mark its centre.
- Pan: the same Tramontina pan with an amber silicone disc (184 mm across, 3 mm thick, 92 g) bonded to its floor in
  place of the oil, so nothing can spill onto the rig. The saucepan stands cold and empty on the right front disc.

Scoring is then read from a top-down photo against the tape and the rings, and tilt and lift from the side video.

## The demo (left arm)

Open loop. A straight-down grip on the handle's red sleeve, 90 mm out from the rim, the pads closing across the handle
(the grip's yaw equals the handle's direction). Gripped, the TCP follows the circle the grip point makes about the
pan's base centre while its yaw turns with it, so the pan swings round in place: from +150 degrees (past the front
edge) clockwise through +90 (past the left edge) and straight back (over the hot element only in passing) toward the
safe sector, 210 degrees in all; the arm's base joint turns about 80 degrees of it and the wrist about 130 (offline
IK: the wrist keeps 38 degrees from its limit and joint 4 keeps 4.8, against 3.5 in the first layout; at the grip
the fingers stay about 23 mm clear of the arm's own base, capsule model). With the first layout's 150 degree turn, the pads
gave a little in yaw while the pan's base dragged on the glass, so the pan lagged the hand by about 6 degrees and
pivoted slightly on the grip; the plan turns 6 degrees past the sector's middle (-54) and then slides the pan
straight back onto the middle of its ring, still gripped, before letting go and lifting away. Replayed offline with
the pan following the gripper rigidly (no lag), the plan ends at -60 degrees, 25 mm off centre, and scores 1.0 at
step 491.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `to_back` turns the handle straight back (0),
`to_middle` toward the middle (0), `hold_on` stops in the safe sector without letting go (0.5).

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/turn-pan-handle-in
INSTR="A toddler is coming into the kitchen, and the frying pan's handle sticks out over the front edge of the cooktop, where the child can grab it. Turn the pan so its handle no longer sticks out, keep it on its burner, and let go."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/pan-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/pan-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S (first layout): about 70 s for the scripted demo (15 s to start Isaac Sim, then 421
steps to success) and about 3 minutes for a Dream-1 run of 800 steps (the policy is queried over the network each
step). The revised demo is 533 steps long (about 4 s more of turning).

## Results

Revision 2 (handle past the front edge), simulator runs on 21 Nov:

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0 (handle at -53.6 degrees, inside the edges, clear of the hot elements) | success from step 493 |
| Demo with the right arm idle | 1.0 (the demo uses the left arm only) | success from step 493 |
| Demo with the left arm idle | 0.0 (the pan is not moved) | - |
| Dream-1 | 0.0 (the pan was not moved) | 800 (the limit) |

First layout (handle past the left edge), for reference: the scripted demo scored 1.0 at step 421 (handle at -54.0
degrees, pan 12.7 mm off centre, flat, let go; the same on two runs), and Dream-1 scored 0.0 over 800 steps (the pan
was not moved).
