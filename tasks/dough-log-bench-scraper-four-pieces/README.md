# Divide a 0.9 kg dough log into four pieces with bench scrapers, chopping at the marks

A 256 x 80 x 40 mm block of bread dough (0.90 kg) lies along y on a bench top 40 mm over the table. Three marks on the
bench's edge, at 64, 128 and 192 mm from the right end, divide it into four portions. In front of each arm a 6 x 3 in
stainless bench scraper stands in a fixed rest. The robot must chop straight down through the dough at each mark with a
scraper's edge, drag nothing along the bench, put the scrapers down clear of the dough and let go.

## Why it is hard

1. **The edge must be exactly on the mark.** Each cut is a 9.4 mm strip of dough over a slot; the 2.5 mm blade's centre
   line must be within 5.7 mm of the mark and the blade upright (15 degrees), spanning the log's 80 mm. An edge that lands on
   the dough beside the strip meets rigid dough and cuts nothing.
2. **The dough gives only to a firm, steady push.** A strip does not move under about 1.5 N, then goes down about 2.5 N for
   16 mm/s (design figures, est), so the arm has to press at a steady rate through 40 mm and not stop short: a strip left
   20 mm down is a crease (two pieces still joined), not a cut. The arm's push limit is about 5 N.
3. **No drag.** Sideways force on the edge shoves the pieces along the bench (floured dough, friction 0.35): 12 mm or 8 degrees
   caps the score at 0.5.
4. **Two hands.** Each arm reaches the table 0.10 to 0.50 m in front of its base and only about 10 cm past the midline at
   low height. With the edge 8 mm over the dough the right arm cannot reach the mark at y +64 mm (IK: not reachable at
   a TCP height of 164 mm, reachable at 158 mm), so the left arm's scraper chops that one.
5. **Hands off, tools down.** Scored with both grippers 30 mm clear, nothing visibly moving for 1 s (under 5 mm and 5 degrees), scrapers clear of the dough.

## Scoring

| Condition | Rule |
|---|---|
| Cut at a mark | strip pressed 37 mm or more down its slot, and a scraper's edge in it (upright within 15 deg, centre line within 4.7 + 1 mm of the mark, middle within 36 mm of the log's centre line, edge below the log's mid height) |
| Creased | a strip pressed more than 4 mm and not through |
| Dragged (latched) | a piece 12 mm or more from its start, or turned 8 degrees or more |
| In line | every piece's centre within 30 mm of the centre line, in order along y |
| Tools down | each scraper's lowest point no more than 10 mm below the table, none within 2 mm of a piece |
| Hands off | last 1 s: finger zones 30 mm or more from every part; each part moves under 5 mm, turns under 5 deg (nothing visibly moves) |
| Binned (latched) | a piece more than 10 mm below the bench top |

| Outcome | Score |
|---|---|
| Three cuts, no crease, not dragged, in line, tools down | 1.0 |
| Three cuts otherwise, or two cuts | 0.5 |
| One cut | 0.25 |
| No cut, binned, or not hands off | 0 |

Possible scores 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| Cut | pieces fully parted, cut faces flat (side photo); video shows the edge inside the 9.4 mm of the mark, marks 1 mm rule |
| Creased | two pieces still joined by dough |
| Dragged | 12 mm rule from each piece's start line; 8 degree angle gauge |
| In line | 30 mm rule from the centre line, order by eye |
| Tools down | 2 mm feeler gauge against each piece |
| Hands off | 30 mm spacer against each hand; video of the last 1 s shows no piece or scraper visibly move (5 mm mark on the bench) |
| Binned | 10 mm drop by eye |
| Portion weights | 1 g scale: each piece within 8 percent of its expected weight (n portions), together at least 85 percent of the weigh-in; see Portion weights below |

## Portion weights (real trial)

The log is 900 g over 256 mm, 3.5 g/mm, so a portion (a quarter of the pieces' combined weight, 225 g if nothing is lost) is 225 g and 8 percent is 18 g, which is
5.1 mm of cut position. Only the pieces that really exist are weighed: after one or two cuts some pieces are still joined,
and a piece joined across n portions is expected to weigh n portions, within 8 percent of that. A piece's length is
64 mm per portion plus its far cut's offset from its mark minus its near cut's (an end has no cut). The sim window (centre
line within 4.7 + 1 = 5.7 mm of the mark) is 0.6 mm wider than the weight tolerance, so on the bench the weigh-in is the
stricter, final check.

- A cut counts only if the video shows the edge within 5.7 mm of the mark and the pieces on both sides of it are within
  8 percent of their expected weight. A piece outside tolerance voids the cuts at its two ends; a cut bordering no failing
  piece stays. The tier follows the cuts left: three 1.0 (if the other conditions hold), two 0.5, one 0.25, none 0.
  Partial credit therefore needs no four equal portions, only correct weights for the pieces that exist.
- Pieces weighing together under 85 percent of the weigh-in (dough lost or binned): 0.
- In the sim the pieces are fixed at 207, 190, 190 and 207 g (each portion's expected weight is then the quarter, 198.5 g, not 225 g): the largest deviation is 8.5 g, 4.3 percent,
  and the total of 794 g is 88 percent of 900 g, so the sim's pieces satisfy both checks; the 12 percent is the strips'
  dough squeezed out of the cut. Sim cut positions therefore need no weight rule; the sim's 5.7 mm window is the whole test.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. All waypoints are straight-down
reachable (worst IK 0.09 mm). Each arm: open over its scraper's bar, down, close on the 22 mm bar, lift 50 mm out of the rest,
up to carry height (edge 8 mm over the dough).

| Time (s) | Right gripper | Left gripper |
|---|---|---|
| 0 to 8.5 | take its scraper from the rest | take its scraper from the rest |
| 8.5 to 25.1 | carry to y -64, chop (3 s down, 0.8 s hold, 2 s up), to y 0, chop | hold the scraper up |
| 25.1 to 36 | back over its rest, set the scraper in, let go, aside | to y +64, chop |
| 28 to 45 | | back over its rest, set it in, let go, aside |

Demo 44.9 s (898 steps, 56 percent of the limit); the arms are 30 cm or more apart in TCP and 13.5 cm in capsules.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1600
```

Runtime: a trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (estimate; the demo is 898 steps).
Seed: nothing is random; `object_jitter=0` fixes the start layout. For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl`
writes one line per control step: the strips' depths, cuts, scraper positions, and a 3 s `hist`.

## Simplifications

- The dough is four rigid pieces and three strips (links on a drive that resists in proportion to speed, with a 1.5 N yield
  and a hold where left) sliding down slots in the bench top; the strips' dough, 12 percent of the log, is squeezed out of
  the cut. A real bench has no slots and a real cut loses almost none. Piece sizes are fixed by the strips, so equal
  portions come from landing the edge on the marks.
- A bond holds each piece to the next until its strip is through (the only rule: `task._dough_rule`, which also sets each
  strip's drive target). Press figures are design values at the arm's limit, not dough's.
- Blade 2.5 mm thick (real 0.9 mm), friction and masses estimated.

## Sources

- Bench scraper 6 x 3 in blade with a bar handle: catalogue size of common stainless bench scrapers (handle, thickness and
  mass are our estimates).
- Bread dough about 1.1 g/mL (est).

## Real build and reset

Why it matters: home bakers divide a log of dough into equal pieces for rolls and buns with a bench scraper; a ten-second
clip shows at once what is being done.

Equipment: a bench board, two 6 x 3 in stainless bench scrapers, a 256 x 80 x 40 mm tray or frame (3D-printed or cut from
board) to shape the log, a ruler, a marker, a 1 g scale. The real trial cuts real dough, with no slots.

Dough (yeast-free, so it does not rise): 560 g plain flour, 330 g water, 10 g salt (900 g, 59 percent water); knead 5 min,
rest covered 20 min, at a room temperature of 20 to 22 C. Press into the tray, weigh 900 g, turn out. A named
store-bought refrigerated dough, brought to 20 to 22 C and shaped the same way, is an equal stand-in. Use within 30 min of
shaping; make a fresh batch per trial (or re-knead the leftovers once).

Bench: wipe, then dust with 5 g flour through a sieve. Marks: with the ruler, mark the bench edge with a marker at 64, 128
and 192 mm from the right end of the log's start line, 20 mm long.

Reset (a few minutes): re-form the log in the tray, check 900 g, turn out centred at x 220 mm, redo the flour; stand each
scraper in its rest, blade along x.

Rig protection: the grippers hold only the scrapers' bar handles, never the dough. Fit food-safe nitrile or silicone covers
on the finger pads, wipe flour and dough off the handles and pads between trials, and wash the scrapers each session.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound bodies with sliders and bonds; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.

## Results

Run on the GPU VM (Isaac Lab-Arena, `object_jitter=0`, 20 Hz, 1600-step limit), one trial each:

| Run | What it does | `task_score` |
|---|---|---|
| Scripted demo (`demo()` as written) | both arms chop as in the Demo table | 1.0 |
| `DEMO_IDLE_ARM=left` (right arm alone) | two marks chopped | 0.5 |
| `DEMO_IDLE_ARM=right` (left arm alone) | one mark chopped | 0.25 |
| Dream-1 (`--policy dream`, full limit) | | 0.0 |

With one arm idle the demo scores below 1.0 (0.5 and 0.25), so both arms are needed here (point 4 above). Earlier offline
predictions (`tools/offline/check_task.py`, not measurements) were 1.0, 0.5 and 0.25 for the same three runs.

The demo's 1.0 was measured under the earlier, stricter hands-off check (each part moves under 1 mm and turns under
0.3 degrees over the last 1 s). That threshold is now 5 mm and 5 degrees, a condition a person can check from ordinary
trial video; relaxing it cannot lower the demo's score, and the GPU demo has not been re-run under it.
