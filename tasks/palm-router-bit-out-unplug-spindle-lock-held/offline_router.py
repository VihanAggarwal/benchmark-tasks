"""The offline stand-in for the physics (tools/offline/check_task.py and the tests only; never used in the simulator).

The checker's proxy knows no plug, button, spindle or nut. Without a simulator, task.py steps this model from the
arms' TCPs, pad tips and openings and scores what it gives:

1. The plug. A gripper whose pad tips lie within 6 mm across (y) of the plug body's centre and within the body's
   length along x (the pads are 27 mm wide, so they still cover the body), between its bottom and its top, with its
   pads commanded under the body's 25 mm width plus 2 mm, holds it until they open past the width plus 4 mm; held,
   the plug follows the TCP back along -x (out 0 to 30 mm; it never goes back in).
2. The lock button (on the router frame's -x side). A nearly closed gripper (gap under 10 mm) whose pads cover the
   button's height, within 12 mm of the axis across, and whose shut pads' +x edge (TCP + L.PRESS_EDGE along the
   router frame's +x, wrist straight down at yaw 0) is past the button's rest face pushes it in by that much, up to
   its 5 mm stop; otherwise the spring has it out. Locked: in 3.5 mm or more.
3. The wrench. Held like the plug at its grip point on the handle (pads under 18 mm plus 2 mm, tips within 6 mm of
   the grip point in xy and 10 mm under to 2 mm over its mid-plane); held, it follows the TCP and, at nut height
   with its grip point within 10 mm of the grip's radius to that radius plus the jaw's 25 mm slide plus 10 mm from
   the spindle axis, points from the axis to the grip point (the demo only ever holds it radially there, both strokes
   and the swing back between them); back over its rest it lies as it was picked up (yaw 0); else it keeps its last
   heading. Its jaw is on the nut when its centre is within 6 mm of the axis
   at nut height and its heading fits the flats (within 4 deg modulo 60); then a turn of the wrench turns the nut on
   the spindle if the lock is in, else it turns the spindle, nut and bit together (the nut does not loosen). With the
   lock in, each engagement first loses L.SEAT_TRAIL_LOSS (jaw seating and wrist trail) and the spindle turns through
   its L.LOCK_PLAY before the nut moves: both measured on the GPU (20 Nov), so the old 51 deg stroke predicts the
   0.644 rad the GPU gave.
   Let go, it drops onto the rest block (over it) or the bench.
4. The bit. Glued in the collet until the nut is 0.7 rad (40 deg) loose. Held at its cutter (pads under 12.7 mm plus 2 mm,
   tips within 5 mm of the axis, between the cutter's bottom and its top) and free, it follows the TCP. Let go, it
   drops: into a block hole when its shank is within 4 mm of the hole's centre and its end already below the
   block's top (shank on the bench, standing), back into the collet within 4 mm of the axis, else lying on its side
   on the bench.

No friction, inertia or contact forces; every number is a design value.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

FULL_GAP = 0.0939
CAPTURE = 0.006


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_ry(a: float):
    return (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))


def _quat_rz(a: float):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


class RouterModel:
    def __init__(self, router_spec, lock_spec, bit_spec, wrench_spec, bench_spec) -> None:
        self.specs = {"router": router_spec, "lock": lock_spec, "bit": bit_spec, "wrench": wrench_spec,
                      "bench": bench_spec}
        self.reset()

    def reset(self) -> None:
        self.plug_out = 0.0
        self.button = 0.0
        self.spindle = 0.0
        self.nut = 0.0
        self.bonded = True
        self.bit_o = np.array([L.RX, L.RY, L.TOP + L.SHANK_Z[0]])
        self.bit_q = (0.0, 0.0, 0.0, 1.0)
        self.w_o = np.array([L.WRENCH0[0], L.WRENCH0[1], L.TOP + L.WRENCH_Z_REST])
        self.w_yaw = 0.0
        self.holds: dict = {}  # arm -> (part, offset)
        self.engaged_yaw: float | None = None
        self.lock_ref: float | None = None  # the spindle's angle when the lock went in (its play is about this)
        self.slip = 0.0  # what is left of this engagement's seating and wrist-trail loss (rad)
        self.lag = 0.0  # the loss taken so far in this engagement (the hand's commanded turn the nut has not had)
        self.was_on = False  # the jaw has been on the nut this episode (then off: its heading is no longer guessed)
        self.mode = "rest"

    # -- geometry -------------------------------------------------------------------------------------------------
    def plug_centre(self) -> np.ndarray:
        return np.array([L.STRIP_X[0] - L.PLUG_SIZE[0] / 2 - self.plug_out, L.PLUG_Y, L.TOP + L.PLUG_Z])

    def grip_point(self) -> np.ndarray:
        return self.w_o + np.array([*(L.GRIP_X * L.unit(self.w_yaw)), 0.0])

    def locked(self) -> bool:
        return self.button >= L.LOCKED_AT

    def _flat_offset(self, yaw: float) -> float:
        """The jaw's heading less the nearest heading that fits the flats, in (-30, 30] deg."""
        nut_abs = self.spindle + self.nut + self.lag  # the lost turn: the held jaw is still on the flats
        fit = L.NUT_FLAT_PHASE + L.ROUTER_YAW - math.pi / 2  # a heading that fits the flats at nut_abs 0
        d = (yaw - fit - nut_abs) % (math.pi / 3)
        return d - math.pi / 3 if d > math.pi / 6 else d

    def _fits_flats(self, yaw: float) -> bool:
        """Within 4 deg, or up to the spindle's play behind the flats: the jaw sliding on turns the nut and the
        spindle back into line (seen on the GPU: the spindle jumped 0.14 rad as the jaw went on)."""
        d = self._flat_offset(yaw)
        return -(L.LOCK_PLAY + math.radians(4.0)) <= d <= math.radians(4.0)

    def _jaw_on(self) -> bool:
        zmid = L.TOP + (L.NUT_Z[0] + L.NUT_Z[1]) / 2
        r = math.hypot(self.w_o[0] - L.RX, self.w_o[1] - L.RY)
        return r <= CAPTURE and abs(self.w_o[2] - zmid) <= 0.004 and self._fits_flats(self.w_yaw)

    # -- one step -------------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        if arms is None:
            return
        pressing = False
        for side in ("left", "right"):
            a = arms.get(side)
            if a is None:
                continue
            gap = float(a.opening) * FULL_GAP
            tips, tcp = np.asarray(a.tips, dtype=float), np.asarray(a.tcp, dtype=float)
            if side in self.holds:
                part, off = self.holds[side]
                width = {"plug": L.PLUG_SIZE[1], "wrench": L.HANDLE_W, "bit": 2 * L.CUTTER_R}[part]
                if gap > width + 0.004:
                    del self.holds[side]
                    self._let_go(part)
                else:
                    self._carry(part, tcp - off)
                continue
            # a new grasp?
            pc = self.plug_centre()
            if (gap <= L.PLUG_SIZE[1] + 0.002 and abs(tips[1] - pc[1]) <= CAPTURE
                    and abs(tips[0] - pc[0]) <= L.PLUG_SIZE[0] / 2 and pc[2] - L.PLUG_SIZE[2] / 2 - 0.002 <= tips[2] <= pc[2] + L.PLUG_SIZE[2] / 2):
                self.holds[side] = ("plug", tcp - pc)
                continue
            gp = self.grip_point()
            if (gap <= L.HANDLE_W + 0.002 and np.hypot(*(tips[:2] - gp[:2])) <= CAPTURE
                    and gp[2] - 0.010 <= tips[2] <= gp[2] + 0.002 and not self._jaw_on()):
                self.holds[side] = ("wrench", tcp - gp)
                continue
            cut0 = self.bit_o[2] + (L.CUTTER_Z[0] - L.SHANK_Z[0])
            cut1 = self.bit_o[2] + (L.CUTTER_Z[1] - L.SHANK_Z[0])
            if (gap <= 2 * L.CUTTER_R + 0.002 and np.hypot(*(tips[:2] - self.bit_o[:2])) <= 0.005
                    and cut0 - 0.001 <= tips[2] <= cut1 and self.bit_q == (0.0, 0.0, 0.0, 1.0)):
                self.holds[side] = ("bit", tcp - self.bit_o)
                continue
            # the lock button (on R's -x side, pressed along R's +x)
            if gap < 0.010:
                local = L.to_router(tcp)
                face = local[0] + L.PRESS_EDGE
                rest = L.BUTTON_FACE_X
                covers = tips[2] <= L.TOP + L.BUTTON_Z + 0.004 and tcp[2] + 0.025 >= L.TOP + L.BUTTON_Z
                if covers and abs(local[1]) <= 0.012 and face > rest and local[0] < 0.0:
                    self.button = float(np.clip(face - rest, 0.0, L.BUTTON_TRAVEL))
                    pressing = True
        if not pressing:
            self.button = 0.0
        if self.bonded and self.nut >= L.BIT_FREE_AT:
            self.bonded = False
        self.mode = ",".join(f"{s}:{p}" for s, (p, _) in sorted(self.holds.items())) or "rest"

    def _carry(self, part: str, point: np.ndarray) -> None:
        if part == "plug":
            rest_x = L.STRIP_X[0] - L.PLUG_SIZE[0] / 2
            self.plug_out = float(np.clip(max(self.plug_out, rest_x - point[0]), 0.0, L.PLUG_UPPER))
        elif part == "wrench":
            ax = np.array([L.RX, L.RY])
            d = point[:2] - ax
            at_nut = (abs(point[2] - (L.TOP + sum(L.NUT_Z) / 2)) <= 0.020
                      and L.GRIP_X - 0.010 <= np.linalg.norm(d) <= L.GRIP_X + L.SLIDE + 0.010)
            rest_grip = np.asarray(L.WRENCH0, dtype=float) + np.array([L.GRIP_X, 0.0])
            if at_nut:
                yaw = math.atan2(d[1], d[0])
            elif np.linalg.norm(point[:2] - rest_grip) <= 0.006:
                yaw = 0.0  # being laid back on its rest, as it was picked up
            else:
                yaw = self.w_yaw
            was_on = self._jaw_on()
            self.w_yaw = yaw
            self.w_o = point - np.array([*(L.GRIP_X * L.unit(yaw)), 0.0])
            on = self._jaw_on() if not was_on else (math.hypot(self.w_o[0] - L.RX, self.w_o[1] - L.RY) <= CAPTURE
                                                    and abs(self.w_o[2] - (L.TOP + sum(L.NUT_Z) / 2)) <= 0.004)
            if on and self.engaged_yaw is None:
                d = self._flat_offset(yaw)
                if d < -math.radians(4.0):
                    self.spindle += d  # the jaw going on turns nut and spindle back into line
                self.slip = L.SEAT_TRAIL_LOSS
            if not self.locked():
                self.lock_ref = None
            elif self.lock_ref is None:
                self.lock_ref = self.spindle
            if on and self.engaged_yaw is not None:
                turn = _angle_wrap(yaw - self.engaged_yaw)
                if self.locked():
                    # GPU-calibrated losses (jobs 142744, 145559): the jaw seating and the wrist trailing eat
                    # SEAT_TRAIL_LOSS of each engagement's forward turn, then the spindle turns through its play
                    # (LOCK_PLAY either way of where the pin went in) before the nut turns on it.
                    if turn > 0.0 and self.slip > 0.0:
                        used = min(turn, self.slip)
                        self.slip -= used
                        self.lag += used
                        turn -= used
                    sp = float(np.clip(self.spindle + turn, self.lock_ref - L.LOCK_PLAY, self.lock_ref + L.LOCK_PLAY))
                    turn -= sp - self.spindle
                    self.spindle = sp
                    self.nut = float(np.clip(self.nut + turn, 0.0, L.NUT_UPPER))
                else:
                    self.spindle += turn
            self.engaged_yaw = yaw if on else None
            if not on:
                self.lag = 0.0
            self.was_on = self.was_on or on
        elif part == "bit":
            if not self.bonded:
                self.bit_o = point.copy()

    def _let_go(self, part: str) -> None:
        if part == "wrench":
            self.engaged_yaw = None
            over_rest = (L.REST_X[0] <= self.w_o[0] <= L.REST_X[1] and abs(self.w_o[1] - L.WRENCH0[1]) <= 0.017
                         and abs(self.w_yaw) < math.radians(10.0))
            if not self._jaw_on():
                self.w_o = self.w_o.copy()
                self.w_o[2] = L.TOP + (L.WRENCH_Z_REST if over_rest else L.WRENCH_T / 2)
        elif part == "bit" and not self.bonded:
            o = self.bit_o
            # Only a shank already down a hole stands in it: let go higher (GPU 20261121T192134: 55 mm over the
            # block's top) the bit hits the block and falls over.
            in_hole = o[2] <= L.TOP + L.BLOCK_SIZE[2]
            for h in L.HOLES:
                if in_hole and math.hypot(o[0] - h[0], o[1] - h[1]) <= 0.004:
                    self.bit_o = np.array([h[0], h[1], L.TOP])
                    return
            if math.hypot(o[0] - L.RX, o[1] - L.RY) <= 0.004:
                self.bit_o = np.array([L.RX, L.RY, L.TOP + L.SHANK_Z[0]])
                return
            self.bit_o = np.array([o[0], o[1], L.TOP + L.CUTTER_R])
            self.bit_q = _quat_ry(math.pi / 2)

    # -- states for the scorer ------------------------------------------------------------------------------------
    def states(self) -> dict:
        s = self.specs
        q = _quat_rz(L.ROUTER_YAW)
        router = mk.state_from_poses(s["router"], (L.RX, L.RY, L.TOP, *q),
                                     joints={"spindle": self.spindle, "nut": self.nut}, source="offline")
        lock = mk.state_from_poses(s["lock"], (L.RX, L.RY, L.TOP, *q), joints={"button": self.button},
                                   source="offline")
        bench = mk.state_from_poses(s["bench"], (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                    joints={"plug": self.plug_out}, source="offline")
        bit = mk.state_from_poses(s["bit"], (*self.bit_o, *self.bit_q), source="offline")
        wrench = mk.state_from_poses(s["wrench"], (*self.w_o, *_quat_rz(self.w_yaw)), source="offline")
        return {"router": router, "lock": lock, "bit": bit, "wrench": wrench, "bench": bench}
