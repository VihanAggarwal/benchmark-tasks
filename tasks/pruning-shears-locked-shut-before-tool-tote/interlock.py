"""Declared mechanical interlock rule for the bypass pruning shears' thumb safety catch.

Shared between the simulator's @mk.on_step hook (task.py) and the offline kinematics (offline_shears.py).
Pure Python, no simulator imports.

Rule:
- The handle hinge ("jaw") has a return spring pulling it toward open (q_jaw = 0.0 rad = 22.0 deg open).
- The thumb safety catch ("catch") has a detent spring pulling it toward unlocked (q_catch = 0.0 m).
- Interlock engages (locked = True) ONLY when the handles are squeezed shut (open_deg <= 5.0 deg) AND
  the thumb safety catch is slid >= 9.0 mm (of its 12.0 mm travel).
- While locked, the catch's drive target switches to 12.0 mm (locked stop) and the handle hinge's drive
  target switches to JAW_UPPER (0.0 deg closed), holding the shears shut after the handles are released.
- If the catch is slid back below 6.0 mm (CATCH_RELEASE_M), the interlock disengages (locked = False)
  and both targets return to 0.0, so the return spring opens the handles again.
- Sliding the catch >= 9.0 mm while the handles are still open (open_deg > 5.0 deg) does NOT lock:
  the catch target stays 0.0 and springs back when released.
"""

from __future__ import annotations

import math

import layout as L


class Interlock:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.locked = False
        self.switches = 0
        self.skipped = 0

    def step(self, q_jaw: float, q_catch: float) -> tuple[bool, float, float]:
        """Update interlock from (q_jaw_rad, q_catch_m). Returns (locked, jaw_target_rad, catch_target_m)."""
        if not (math.isfinite(q_jaw) and math.isfinite(q_catch)):
            self.skipped += 1
            return self.targets()
        open_deg = L.open_deg_of(q_jaw)
        was = self.locked
        if not self.locked:
            if open_deg <= L.CLOSED_MAX_DEG + L.EPS and q_catch >= L.CATCH_ENGAGE_M - L.EPS:
                self.locked = True
        else:
            if q_catch < L.CATCH_RELEASE_M - L.EPS:
                self.locked = False
        if self.locked != was:
            self.switches += 1
        return self.targets()

    def targets(self) -> tuple[bool, float, float]:
        if self.locked:
            return True, L.JAW_UPPER, L.CATCH_TRAVEL
        return False, 0.0, 0.0
