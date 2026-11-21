"""The weight's spring clip on the rod, as a stick-slip spring: the one declared software model of the task. Plain
Python; task.py writes its target through mechkit every control step (a mechkit on_step rule), the tests and the
offline checks call the same function.

Why a rule. The first GPU run (19 Nov) modelled the clip as PhysX joint friction (3 N). PhysX let the weight creep
with no real force on it: 0.8 mm down the rod under its own 5 g in 11 s with nobody touching it, and 6.9 mm UP the
rod while the pendulum swung (the weight's real loads along the rod are under 0.06 N there), so the tempo ran away
from 136 to 125 per minute within 9 s. The same creep is noted for every other task that leaned on PhysX joint
friction. A real metronome's weight is held by a spring clip: it does not move until pushed or pulled along the rod
with about 3 N, and then slides against about 3 N.

The model. The slider has a stiff spring (``layout.W_CLIP_K``, 4000 N/m) toward a target. While the weight sits
within ``layout.W_CLIP / layout.W_CLIP_K`` (0.75 mm) of the target, the target stays put: the clip holds against
any force up to 3 N (the weight's own 5 g moves it 0.012 mm). Pulled or pushed farther, the target is moved so the
spring pulls back with 3 N, led by the weight's travel over the last control step (the target is written once per
50 ms step; the lead keeps the step's mean pull near 3 N instead of 3 N plus the spring's rise over the step), at
most 0.5 mm, under the 0.75 mm band, so the spring never pushes a sliding weight on. So: held below 3 N, sliding at
about 3 N, never creeping. Let go, a weight that was sliding settles back onto the target, 0.75 mm short of where it
stalled (the clip's elastic set). Nothing reads where the weight is meant to go; there is no mark, notch or snap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import layout as L
import pendulum as P

LOWER = L.W_R_MIN - P.R_START  # the slider's limits (m, 0 at the 160 mark)
UPPER = L.W_R_MAX - P.R_START


def band() -> float:
    """The stick band (m): the spring's stretch at the clip's 3 N."""
    return L.W_CLIP / L.W_CLIP_K


@dataclass
class Clip:
    target: float = 0.0
    prev: float | None = None
    slips: int = 0

    def reset(self) -> None:
        self.target, self.prev, self.slips = 0.0, None, 0

    def update(self, q: float) -> float:
        """The drive target for this control step from the weight's slide ``q`` (m) now."""
        q = float(q)
        if not math.isfinite(q):
            self.prev = None
            return self.target
        b = band()
        e = q - self.target
        if abs(e) > b:
            s = 1.0 if e > 0 else -1.0
            lead = 0.0
            if self.prev is not None and (q - self.prev) * s > 0:
                lead = max(-L.W_CLIP_LEAD_MAX, min(L.W_CLIP_LEAD_MAX, L.W_CLIP_LEAD * (q - self.prev)))
            self.target = max(LOWER, min(UPPER, q - s * b + lead))
            self.slips += 1
        self.prev = q
        return self.target
