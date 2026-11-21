# Mark a line 50 mm from a board's edge with a combination square and pencil

A 610 mm offcut of a 1-by-6 pine board (3/4 x 5-1/2 in actual) lies along the table. A STANLEY 46-012 12 in
combination square, its blade already locked so that it sticks 49 mm out from the head's face, lies on the table with
the head 30 mm off the board's near edge and the blade over the board. A flat carpenter's pencil stands tip down in a
wooden pencil block beside the board. The robot must seat the head's face on the edge, put the pencil's lead on the board
against the end of the blade, and slide square and pencil along together, so the lead draws a line 50 mm in from the
edge, parallel to it, at least 90 mm long. Then the pencil goes back in its block and both hands let go.

Marking a line a set distance in from an edge with a combination square is a routine layout step for carpenters
(rip lines, shelf dadoes, the back of a notch); with the blade locked to the distance the square serves as the gauge.

**A pencil line cannot be drawn in the simulator.** The scorer therefore reads the PATH of the pencil's lead tip (its
offset from the board's edge, straightness and length) while the square's head is held against the edge, under a measured
pencil-on-board contact force. In a real trial the pencil's own graphite draws that path and a ruler and a straightedge
check it.

| Part | Product | Figures |
|---|---|---|
| Board | 610 mm offcut of a nominal 1-by-6 pine board (about US$10 for 6 ft) | 19.05 x 139.7 x 610 mm, 0.73 kg (est) |
| Square | STANLEY 46-012 12 in combination square (about US$15) | listing: 12 in blade; est: blade 304.8 x 25.4 x 1.6 mm, cast head 112 x 62 x 33 mm, locking screw, 0.35 kg |
| Pencil | flat red carpenter's pencil, the common 7 in kind (about US$1) | est: 7 x 18 mm section, worn to 140 mm, chisel point 2 mm thick, 14 g |
| Pencil block | 50 x 36 x 45 mm hardwood block fixed to the table (est) | slot 21 x 10 mm, 42 mm deep, 3 mm floor |

## Why it is hard

1. **The head must stay flat on the edge while it moves.** The pencil pushes the end of the blade back toward the robots,
   which pulls the head off the edge. The head hand grips the blade's tail just behind the head, pushes the square
   along the table until the head's face is on the edge (30 mm), and keeps it pushed toward the edge while it slides
   110 mm along it, with the face within 1 mm and 1.5 degrees of the edge on every ink step. (Pressing on the head's
   top and pushing does not move it: in the first GPU run the fingertips slid over the head.) If the face leaves the
   edge the line steps off its 50 mm.
2. **The pencil hand keeps its own contact.** The lead must stay on the board (0.3 N or more) and its flat face against
   the end of a 25.4 mm wide blade, within 9 mm of its middle, over the whole 110 mm path, at the same pace as the head
   hand (3.4 mm/s). Too little pressure and the lead hovers; too much and it pushes the head off the edge.
3. **One hand is not enough.** With the head not held, the pencil's push drags the 0.35 kg square (friction about 1.2 N)
   away from the edge. With only the head hand, nothing draws.
4. **The arms must not collide.** The two hands work on the same line about 160 mm apart (the blade's end and its tail),
   the arms crossed, and the right arm waits aside while the left brings the head to the edge.
5. **The whole act.** The pencil comes out of its block first and goes back into it after, then both hands let go.

## Scoring

| Condition | Rule (sim) | Bench method (real trial) |
|---|---|---|
| Ink (declared rule) | each step the lead's tip is on the board's top face (within 0.3 mm, or into it), pencil within 20 degrees of upright, AND a PhysX contact force of 0.3 N or more (contact.py; unreadable counts as none): a point of the line | the pencil's own line |
| Seated | head's face within 1.0 mm of the edge, within 1.5 degrees of parallel, lying flat within 3 degrees | no light under the face, along the edge |
| Guided | a line point is guided if the square is seated AND the pencil's flat face is on the blade's end (gap -1.2 to +0.8 mm), within 9 mm of its middle | the pencil seen against the blade's end |
| Guided OK | of the 2 mm stretches of the line, 95 percent hold ink and every ink point in them was guided | the video |
| Spans | at least 90 mm long, no gap over 6 mm | ruler |
| Parallel | the fitted line within 1 degree of parallel to the edge | its distance from the edge at both ends differs by under 2 mm |
| Straight | every point within 1.5 mm of the fitted line | a straightedge along the line |
| At 50 mm | the fitted line 50 mm in from the edge within 2 mm at both ends | ruler |
| Stowed | the pencil back in its block: tip within 3 mm of the slot's axis, 5 mm of its floor, axis within 10 degrees | the pencil standing in its block |
| Hands off | both finger zones 30 mm or more from square, pencil and block on every step of the last 1 s | no finger touches them in the last 1 s of video |
| Still | square and pencil each move under 5 mm over the last 1 s | no visible motion in the last 1 s |
| Latches | the board moved 20 mm or turned 3 degrees, or any part over 3 mm below the table top, at any step | the video |

| Outcome | Score |
|---|---|
| A latch fired, or the head never seated against the edge for 0.5 s | 0 |
| The head seated, but no good line | 0.25 |
| A good line, the pencil not back in its block at the last step | 0.5 |
| A good line, stowed, but hands off or still fails on a step of the last 1 s | 0.75 |
| A good line, stowed, hands off and still for the whole last 1 s (20 steps) | 1.0 |

| Bypass | What stops it |
|---|---|
| Draw along a straight 50 mm path without the square on the edge | not guided: 0 |
| Leave the head 12 mm off the edge and draw against the blade | not seated, not guided: 0 |
| Hover the lead (no contact force) | no ink: no line, 0.25 at most |
| Let the head drift off the edge mid stroke | straight or 50 mm fails: 0.25 |
| Draw twice, or leave a stray mark | straight fails: 0.25 |
| Stop short of 90 mm | does not span: 0.25 |
| Leave the pencil out of its block | 0.5 |

## How it differs from nearby kinds of task

- Drawing one line along a tool that stays put on the board: here the whole square slides along the board's edge, its
  head pressed on the edge, and the line's distance comes from the blade's setting.
- Scribing with a compass run along a wall, or dabbing one mark off a tape measure: here it is a continuous gauged line,
  50 mm from the edge along 110 mm.

## Run it

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/combination-square-set-50mm-pencil-offset-line/task.py -E object_jitter=0 -E control_hz=20

About 87 s of demo (1806 steps at 20 Hz with the policy's closing hold) of the 2400 steps allowed; no randomness (`object_jitter=0`). Settle check:
`DEMO_HOLD_ONLY=1`. One arm idle: `DEMO_IDLE_ARM=left` or `right`. Counterfactual demos (never the submitted one):
`DEMO_VARIANT=short|unseated|nostow`. Trace: `TASK_TRACE=<file>`.

## Results

| Run | Offline prediction (tools/offline) | GPU |
|---|---|---|
| Demo (both arms) | 1.0 (success held at step 1775) | run 1 0.0 (square never moved, force not read); run 2 0.25 (seated late, 0 N on about 30 percent of ink steps, pencil missed the slot); run 3 0.25 (seated and guided, but 11 percent of the 2 mm stretches empty at a smoothstepped 15 mm/s mid-stroke; pencil caught the slot's wall); run 4 (21 Nov, 18:33 job) 0.5 at the end of the run: the line good (guided 1.0, offset error -0.61 mm, 109 mm long), the pencil not stowed; stow fixed (below); run 5 (21 Nov, 19:27 job) 1.0, held from step 1761 to the end |
| One arm idle: left (head hand) only | 0.25 (seated, no line) | 0.25 (run 5) |
| One arm idle: right (pencil hand) only | 0.0 (the head never reaches the edge) | 0.0 (run 5) |
| Variant short (45 mm short) | 0.25 | not run on the GPU |
| Variant unseated (head left 15 mm off the edge) | 0.0 | not run on the GPU |
| Variant nostow (pencil put down beside the block) | 0.5 | not run on the GPU |
| Dream-1 at the full run limit | n/a | 0.0 (run 5) |

**GPU run 4: why the pencil was not stowed, and the fix.** The line was good; the score stayed at 0.5 because the pencil
never went into its block. Over the block the lead's tip hovered at x 0.3915, y -0.1596 (aim 0.392, -0.160: the 18 mm
body has 1.5 mm to spare each side of the 21 mm slot). The 2 s, 53.5 mm move down (about 40 mm/s mid-move) then left
the tip sitting on the top of the block's -x wall (x 0.3873, z 0.7925, the block's top), the hand 4.3 mm to -x of its
command and 40 mm above it. The fingers opened there, the pencil fell over onto the open fingers, the hand (no longer
held up) dropped onto it, and the move out and up carried the pencil away jammed between the fingers. Runs 1 to 3 ended
the same way, at x 0.3874 to 0.3878 on the same wall's top, with the tip's y anywhere from -0.1603 to -0.1634. So the
miss was in x, not in y as the earlier notes said. The hand lags behind a fast vertical move: on the way up out of the
block it ran 1.9 mm to +x, and on a fast move down it lags toward -x. Our estimate, not a measurement, is about 1.2 mm
to -x per 10 mm/s of descent over the block. The demo's stow now hovers as before, then: aims the tip 0.7 mm to +x of
the slot's axis; goes down to 3 mm over the block's top and waits 0.5 s for the lag to die out; goes past the walls'
top slowly (15 mm in 5 s, about 4.5 mm/s at the top, under 0.6 mm of lag by the same estimate); goes on down inside the
slot to the floor (4 s); holds 1 s; opens; waits 1 s; lifts straight up along the pencil (45 mm in 3 s); then moves
aside. The line drawing is unchanged. Offline the demo still predicts 1.0. GPU run 5 (21 Nov) scored 1.0 with this fix.
