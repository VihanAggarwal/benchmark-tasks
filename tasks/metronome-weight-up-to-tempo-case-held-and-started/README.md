# Set the practice metronome to 100 for the next piece and start it ticking

A Wittner Taktell Piccolo 836 (series 830), a small black plastic clockwork pyramid metronome (155 mm tall, base
70 x 45 mm, 171 g, tempo 40 to 208), stands loose on the table, its front with the pendulum and the tempo scale toward
the right arm. Its sliding weight sits at the 160 mark, where the last piece left it, and the pendulum hangs still.
The robot must set it to 100 for the next piece and start it: the weight slid up the rod to the 100 mark, the
pendulum swung out and let go so that it beats on its own with both hands away. The instruction names no grip, no
hold and no way to start it.

Why it starts wound: the spring is wound once when practice begins, and between pieces a player only changes the
tempo and restarts the pendulum. That between-pieces reset is the act here, so the winding key is not used.

## Why people know it

Learning an instrument is common: "two-thirds of Americans (66%) have learned to play a musical instrument at some
point in their lives" (YouGov, 2022). The metronome is the standard practice aid for it: metronomes "are thus
commonly used at all skill levels, from beginners to professional musicians, and are often recommended to music
students without reservation" (Wikipedia, Metronome). The clockwork pyramid (Maelzel's design, patented 1815) is set
the only way it can be: "The weight slides up the pendulum rod to decrease tempo, or down to increase tempo"
(Wikipedia), then the rod is swung out and let go. Anyone who has practised with one, or sat beside a child, a
teacher or a classmate who did, has seen exactly these two moves, and the ticking rod is known on sight. We found no
survey of how many people have set a mechanical metronome themselves, and many students now also use electronic or
phone metronomes.

## Why it is hard

1. **The case must be held down while the weight is pulled up.** The weight's spring clip grips the rod with about
   3 N (a design value; the pre-trial check tests it), more than the whole metronome weighs (1.68 N). Pulled up with one hand, the weight does not slide: the
   metronome comes off the table with it. One hand must hold the case down on the table while the other drags the
   weight 22.7 mm up the rod, as a person does.
2. **The weight must stop on its mark.** 96 and 104 are 1.5 mm either side of 100 on this scale, so the weight has to
   stop within about 1.5 mm of the mark after a 22.7 mm drag against the clip.
3. **Then it must be started and left alone.** The pendulum is swung out 20 to 30 degrees and let go so it swings on
   its own; a swing under 15 degrees, a stopped pendulum or a hand near it while it is timed does not count.
4. **The tempo is physics, not a rule.** The pendulum (rod, counterweight inside the case, sliding weight) swings at
   its own rate, which depends on where the weight sits; the scorer times the swings. `pendulum.py` gives the rate
   from the same masses and shapes the simulator uses, and the scale marks are drawn where those rates fall.
   The one rule is the weight's spring clip (`clip.py`): a stiff spring whose anchor stays put while the weight is
   pushed or pulled along the rod with under 3 N and follows it at 3 N. It knows no mark. (PhysX joint friction, the
   first build's clip, let the weight creep along the rod with no force on it.)

The case is light and the two hands work within 10 cm of each other, so both wrists are rolled 0.45 rad about world x
while the case is held and the weight dragged (`tiltgrip.py`): the pads still pinch across x while the wrists lean apart
(straight-down grips put the left wrist into the right gripper). The start is made straight down, with the left hand
parked: the closed pads come down in front of the case beside the weight, push it out (27 deg aimed, about 24 on the GPU) and slide off its
front.

## Scoring

The sim and the real trial use the same windows and tolerances.

| Condition | Rule (sim, every 0.05 s step) | Bench method (real trial: 30 fps video with sound, end photo) |
|---|---|---|
| Tick | the pendulum's angle changes sign between two steps (time interpolated) | each audible tick, timed on the sound track |
| Timed | the last 8 ticks: none 1.6 s or more apart, the last under 1.6 s before the end, every swing between them 15 deg or more each side, both finger zones 30 mm or more from the metronome and the case standing (tilt under 10 deg, underside within 3 mm of the table) on every step from just before the first of them | the rod's tip 25 mm or more either side of upright; no finger within 30 mm over that stretch (a 30 mm block between each hand and the metronome in the end pose); the case flat on the table against the table line |
| Tempo | 60 / mean interval of the 8 timed ticks | the time from the first to the eighth tick on the sound track, divided by 7 |
| Set | the weight's pointer (top front edge) within 2.5 mm of the 100 mark, along the rod (every mark is drawn level with the pointer of a weight set to it) | calipers or a ruler on the end photo, pointer to the 100 mark |

| Outcome (at the last step; the first that holds) | Score |
|---|---|
| Timed at 96 to 104 per minute | 1.0 |
| Timed at 90 to 110 | 0.75 |
| The weight set, but not timed (not started, stopped, held up, or a hand still near) | 0.5 |
| Timed at any other tempo, the weight not set | 0.25 |
| Otherwise | 0 |

Every end state has one score. Success: 1.0 held for 1 s. Doing nothing scores 0. Where the case ends up on the table
does not matter. The score never reads the order of moves or which hand did what: pulled up with the case not held,
the weight lifts the whole metronome instead of sliding, so it stays at 160.

## Results

GPU run of this version (19 Nov 2026, job 20261119T061907), at the full 1200-step limit; the setup screenshot and
the videos are of this run and this version.

- Settle (`DEMO_HOLD_ONLY=1`, 160 steps): 0.0. Nothing moved: the pendulum stayed upright (under 0.01 deg), the
  weight at 43.69 mm over the pivot (the 160 mark), the case's lift, tilt and slide 0.
- Scripted demo: 1.0. The drag took the weight from 43.69 to 67.69 mm over the pivot (step 325); after the right hand
  eased off and let go it settled at 66.93 mm (step 352) and stayed there to the end (66.94 mm, the pointer 0.54 mm
  over the 100 mark, inside the 2.5 mm Set band). During the drag the case slid 12 mm toward the right arm (11.2 mm
  from its start at the end) and rose at most 1.3 mm with 2.9 deg of tilt under the left's pinch; it stayed
  standing. The right TCP reached its push height (876.8 to 878 mm world z, steps 568 to 586) without stalling on
  the case. The push held the rod out at about 24 deg (steps 613 to 622; 27 aimed); while it pressed, the case
  rocked up 6.7 mm and 10.6 deg (step 604) and settled back flat by step 613, before any timed tick. Let go, the
  pendulum swung free at about 23 to 26 deg each side; both finger zones were 30 mm or more from the metronome from
  about step 667 (210 mm at the end) and the case stood flat. The last 8 ticks timed 97.9 per minute: score 1.0
  from step 757, held 1 s, and the run stopped at step 776.
- One arm, `DEMO_IDLE_ARM=right` (left alone, the case held): 0.0. Nothing else happened; the weight stayed at
  43.69 mm, the pendulum still, the case moved 2.4 mm.
- One arm, `DEMO_IDLE_ARM=left` (right alone): 0.0. The clip held: pulled up with the case not held, the weight moved
  under 0.5 mm along the rod (43.7 to 44.1 mm) while the whole metronome came off the table (up to 48.5 mm) and then fell
  on its side (tilt 90 deg, 107 mm from its start), still at 160.
- Dream-1 (`--policy dream`, 1200 steps): 0.0. The weight ended 1.4 mm lower (42.3 mm, away from 100), the
  pendulum never swung past 18 deg, the case moved 1.6 mm.

Earlier GPU runs (19 Nov 2026, earlier builds), kept as history: the first build (the clip as PhysX joint friction)
scored 0.25: the weight crept along the rod with no force on it and never reached the mark, and the swing started at
only 17 deg. The second build (the clip rule of `clip.py`) scored 0.0: the weight stopped 5.0 mm short (61.4 mm) and
the rolled start push came down on the case's top and tipped it. The fixes since: the drag commanded 5.0 mm farther,
and the start pushes the weight's side straight down from in front of the case.

The offline column is a numpy-only offline check of the scripted demo (no GPU, no simulator) with this task's offline
stand-in for the physics (`offline_model.py`: grips from the commanded pads, the clip as a dead band of the arm's
give, the clip lifting the unheld case, the exact free pendulum equation, a quasi-static push on the rod or the
weight). It is a prediction, not a measurement.

| Run | What it does | Offline prediction | GPU score (job 20261119T061907) |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | left pinches the case's back half and presses it down; right drags the weight 160 to 100, eases off, lets go, pushes the weight out from in front of the case, slides off it | 1.0 | 1.0 (97.9 per minute; 1.0 from step 757, stopped at step 776) |
| One arm: `DEMO_IDLE_ARM=right` (left alone) | the case is held; nothing else happens | 0.0 | 0.0 |
| One arm: `DEMO_IDLE_ARM=left` (right alone) | the pulled weight lifts the unheld case instead of sliding and stays at 160 | 0.0 | 0.0 (the case lifted and fell over) |
| `DEMO_VARIANT=nohold` | as the right alone | 0.0 | not run |
| `DEMO_VARIANT=short` | the weight dragged only 12 mm (about 126) | 0.25 | not run |
| `DEMO_VARIANT=nostart` | the weight set, the pendulum never started | 0.5 | not run |
| Dream-1 | `--policy dream`, full limit (1200 steps) | not predicted | 0.0 |

Both hands are needed: either arm alone scored 0.0 on the GPU.

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
        -E control_hz=20 --scorer task_score --max-steps 1200 --epochs 1

Runtime per trial: about 60 s of sim time at most (1200 steps); the scripted moves end at step 837 (on the GPU the run stopped at step 776, success held). Seed: any (no randomness
with `object_jitter=0`). Trace: `TASK_TRACE=<file>`. One arm: `DEMO_IDLE_ARM=left|right`. Settle:
`DEMO_HOLD_ONLY=1`. Tests: `python3 -m unittest discover -s tests` (36 tests, offline).

## Files

`task.py` (scene, hooks, demo), `layout.py` (every number), `pendulum.py` (the swing rate from the parts), `parts.py`
(shapes and joints), `clip.py` (the weight's spring clip, the one rule), `scoring.py` (the tracker), `offline_model.py` (offline only), `tiltgrip.py` (rolled grips for
the demo), `mechkit.py` and `rigstate.py` (a shared mechanism kit, copied unchanged), `tests/`.

## GPU checks (job 20261119T061907)

- Settle: the pendulum stayed upright, the weight at 43.69 mm, the case still. Passed.
- The clip: the weight did not move while nobody held it (66.93 to 66.94 mm from step 352 to the end, through the
  push and the free swing). Passed.
- The drag: the weight settled at 66.93 mm over the pivot (64.9 to 67.9 wanted), with `layout.DRAG_GIVE` at 14.2 mm.
  The case slid 12 mm toward the right arm (the push allows 13 mm). Passed.
- The start: the right TCP reached its push height without stalling on the case; the rod was held at about 24 deg and
  swung free at 23 to 26 deg; the case rocked 10.6 deg during the push and settled before the timed ticks. Passed.
- The swing rate: 97.9 per minute at 66.94 mm (96 to 104 wanted). Passed.

## Sources

- YouGov, "Younger Americans are increasingly exposed to playing music and a wider range of instruments" (L. Sanders, 23 Aug 2022; 3,000 US adult citizens,
  June to July 2022): "two-thirds of Americans (66%) have learned to play a musical instrument at some point in
  their lives".
  https://yougov.com/en-us/articles/43512-young-americans-increasingly-exposed-music
- Gallup for NAMM, "American Attitudes Toward Making Music" (2003): 54 percent of US households surveyed have a
  member who plays an instrument; 64 percent of players began musical training between ages 5 and 11.
  https://www.namm.org/news/press-releases/gallup-organization-reveals-findings-american-attitudes-toward-making-music
- ABRSM, Making Music 2014 (with Trinity College London): 76% of UK children aged 5 to 14 say they know how to play a
  musical instrument (41% in 1999).
  https://www.trinitycollege.com/news/viewarticle/more-children-than-ever-in-the-uk-are-learning-to-play-a-musical-instrument
- ABRSM, Making Music 2021 (1,503 children, 1,505 adults): 59% of children and 27% of adults currently play an
  instrument (as reported by Arts Professional and The Herald; we could not open the report itself).
  https://www.artsprofessional.co.uk/node/196789
  https://www.pressreader.com/uk/the-herald-1130/20211130/281925956290737
- Wikipedia, "Metronome": Maelzel patented the mechanical wind-up metronome in 1815; "The weight slides up the
  pendulum rod to decrease tempo, or down to increase tempo"; metronomes "are thus commonly used at all skill levels,
  from beginners to professional musicians, and are often recommended to music students without reservation".
  https://en.wikipedia.org/wiki/Metronome
- Wittner W836 Piccolo (retailer listing): "Wittner's most popular compact pendulum metronome", "easily adjusted via a
  sliding counterweight", "full tempo range"; black, US$79.95. https://www.billsmusic.com/w836/
