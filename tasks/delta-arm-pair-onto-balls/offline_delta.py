"""Kinematics of the arm pair and the effector for use ONLY where there is no simulator: tools/offline/check_task.py
(which by itself can carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result
from here: there the hinges, the spring, the stop screw, the pins and every contact are PhysX's.

What this file assumes is what the design claims (spec 2, 3.2, 5), so its prediction is the design's prediction, not
a measurement. It moves the parts from the arms' COMMANDED pad-tip points, TCPs and openings (rigstate's offline
feed):

- the pair stays centred on x 0.300: arm_b turns phi against arm_a and arm_a turns -phi / 2 (the symmetric spread of
  one hand; no other sideways swing is modelled); the spring closes it to the stop screw (phi 0) unless something
  holds it open;
- a gripper spreads the pair when it entered between the tubes with its fingers' outer faces inside the tubes' inner
  faces (tips at most 25 mm above the table, its TCP 40 to 300 mm along the arms from the pivots); the tubes' inner
  faces at its TCP's station are then its pad gap + 2 x 8 mm apart less the fingers' give under the spring's force
  (5,000 N/m a finger, spec 7 O4), never wider than both tubes on their rest-bar pins;
- the sockets catch a stud pair when the pair closes from 66 mm or more between the sockets with that pair's balls
  within 2.5 mm of the socket line along the arms and of socket height (layout.CATCH_ALONG), its midpoint within
  2.0 mm of the pair's centre line, its yaw within 2.5 deg: the pair then stops at 62 mm between the sockets (W 84)
  and, if no gripper holds the effector, pulls it onto the sockets' axes (spec 5.5, "slide under the sockets'
  pull"); balls
  between the blocks outside that window hold the pair at 66 mm (on the crowns, W 88);
- the effector is held by a gripper whose pads straddled its post open (more than 20 mm) and then closed below the
  post's 20 mm; it lets go once open wider than 21 mm; a held effector translates with the gripper's pad tips
  (no turn), never below the table;
- pushed at a pair whose sockets are less than 66 mm apart, the effector stops with its front balls 12 mm short of
  the socket line (spec 2, the jam), level on the table;
- released in the air: if its front balls are over the closed pair's free socket blocks it perches there with the
  rear edge of its plate on the table (tilt about 11 deg, V8); otherwise it drops flat. A perched effector drops
  flat once the sockets are 66 mm or more apart, its ball line moving 3.9 mm toward +y (V8);
- nothing tips, slides or bounces otherwise; friction and forces are not modelled.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

POST_GRIP_GAP = 0.020  # m: the post's width across the pads
POST_RELEASE_GAP = 0.021
SPREAD_V_RANGE = (40.0, 300.0)  # mm along the arms from the pivots
SPREAD_TIP_MAX_H = 0.025  # m above the table
PERCH_BALL_Z = L.BALL_Z + L.BLOCK_SIZE / 2 + L.BALL_R  # 25 mm: block tops at z 20 plus the ball's radius (V8)
REAR_EDGE = L.HEX_FLATS / 2  # mm: the plate's rear flat from its centre


def rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def perch_tilt() -> float:
    """The tilt (rad) at which the front balls' centres stand at 25 mm with the rear edge on the table: solves
    61 sin t + 13 cos t = 25 (V8)."""
    a, b = L.STUD_PAIR_OUT + REAR_EDGE, L.BALL_Z
    r = math.hypot(a, b)
    return math.asin(PERCH_BALL_Z / r) - math.atan2(b, a)


class OfflineDelta:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phi = 0.0
        self.origin = np.array([*L.EFFECTOR_ORIGIN, L.TOP])  # world, m
        self.tilt = 0.0  # rad about world x, front up (perched)
        self.held_by: str | None = None
        self.grip_rel = np.zeros(3)
        self.around = {"left": False, "right": False}
        self.spreading = {"left": False, "right": False}
        self.seated: int | None = None
        self.perched = False
        self.jammed = False
        self.events: list[str] = []

    # -- frames -------------------------------------------------------------------------------------------------
    def rotation(self) -> np.ndarray:
        return rx(self.tilt)

    def balls(self) -> list[tuple[np.ndarray, np.ndarray]]:
        r = self.rotation()
        return [(self.origin + r @ L.m(a), self.origin + r @ L.m(b)) for a, b in L.effector_balls_local()]

    @staticmethod
    def socket_y() -> float:
        return L.SLIDER_ORIGIN[1] - L.ARM_LEN * math.cos(L.ARM_TURN) * L.MM

    def socket_gap(self, phi: float | None = None) -> float:
        return L.socket_distance(self.phi if phi is None else phi)

    # -- one control step ---------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        hands = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            hands[side] = (np.asarray(tcp, dtype=float), np.asarray(tips, dtype=float), gap)
        prev_tips = getattr(self, "_prev_tips", None)
        self._update_hold(hands)
        if self.held_by is not None and prev_tips is not None:
            self._carry(hands[self.held_by][1])
        target = self._spread_target(hands)
        self._close_or_open(target)
        self._prev_tips = {s: h[1].copy() for s, h in hands.items()}

    def _post_local(self, tips) -> np.ndarray:
        return self.rotation().T @ ((np.asarray(tips) - self.origin) / L.MM)

    def _on_post(self, tips) -> bool:
        p = self._post_local(tips)
        (sx, sy, sz), (cx, cy, cz) = L.POST
        return abs(p[0] - cx) <= 4.0 and abs(p[1] - cy) <= sy / 2 + 3.0 and cz - sz / 2 <= p[2] <= cz + sz / 2

    def _update_hold(self, hands) -> None:
        for side, (tcp, tips, gap) in hands.items():
            if self.held_by == side:
                if gap > POST_RELEASE_GAP + 1e-9:
                    self.held_by = None
                    self._release()
                continue
            on = self._on_post(tips)
            if self.held_by is None and on and self.around[side] and gap < POST_GRIP_GAP - 1e-9:
                self.held_by = side
                self.grip_rel = self.origin - tips
                self.events.append(f"{side} holds the post")
            self.around[side] = bool(on and gap > POST_GRIP_GAP)

    def _carry(self, tips) -> None:
        want = tips + self.grip_rel
        if self.perched:
            self.perched, self.tilt = False, 0.0
        want[2] = max(want[2], L.TOP)
        if self.seated is not None:
            fy = self.balls()[self.seated][0][1]
            if abs((want[1] - self.origin[1]) + (fy - self.socket_y())) > L.CATCH_ALONG * L.MM:
                self.seated = None
                self.events.append("pulled off the sockets")
        low = want[2] - L.TOP <= 0.002
        if self.seated is None and low and self.socket_gap() < L.WIDTH_PASS - L.WIDTH_EXTRA - 0.5:
            limit = self.socket_y() - L.JAM_SHORT * L.MM - (self.balls()[0][0][1] - self.origin[1])
            if want[1] > limit:
                want[1] = limit
                if not self.jammed:
                    self.events.append("jammed 12 mm short")
                self.jammed = True
        self.origin = want

    def _release(self) -> None:
        if self.origin[2] - L.TOP <= 0.002:
            self.origin[2] = L.TOP
            return
        front = self.balls()[0]
        d = self.socket_gap()
        over = all(abs(abs(b[0] - L.SLIDER_ORIGIN[0]) / L.MM - (d / 2 + L.BLOCK_PROUD / 2)) <= 5.0 for b in front)
        along = all(abs(b[1] - self.socket_y()) / L.MM <= L.BLOCK_SIZE / 2 + 2.0 for b in front)
        ball_y = float(front[0][1])
        if over and along and d < L.WIDTH_PASS - L.WIDTH_EXTRA:
            t = perch_tilt()
            pivot_y = ball_y - (L.STUD_PAIR_OUT + REAR_EDGE) * L.MM * math.cos(t) + L.BALL_Z * L.MM * math.sin(t)
            self.tilt, self.perched = t, True
            self.origin = np.array([self.origin[0], pivot_y + REAR_EDGE * L.MM * math.cos(t),
                                    L.TOP + REAR_EDGE * L.MM * math.sin(t)])
            self.events.append("perched on the blocks")
        else:
            self.origin[2] = L.TOP
            self.events.append("dropped flat")

    def _spread_target(self, hands) -> float | None:
        """phi held by a spreading gripper (None: no gripper spreads)."""
        best = None
        for side, (tcp, tips, gap) in hands.items():
            v = (L.SLIDER_ORIGIN[1] - tcp[1]) / L.MM
            h = tips[2] - L.TOP
            in_region = SPREAD_V_RANGE[0] <= v <= SPREAD_V_RANGE[1] and -0.002 <= h <= SPREAD_TIP_MAX_H
            outer = gap / L.MM + 2 * L.FINGER_THICK
            off = abs(tcp[0] - L.SLIDER_ORIGIN[0]) / L.MM
            inner = L.inner_gap_at(v, self.phi) if in_region else 0.0
            if not in_region:
                self.spreading[side] = False
                continue
            if not self.spreading[side] and outer / 2 + off <= inner / 2 + 0.5:
                self.spreading[side] = True  # entered between the tubes
            if not self.spreading[side]:
                continue
            w = L.width_for_pad_gap(gap / L.MM - L.GPU_GAP_LOSS_MM, v)
            phi = min(L.phi_for_width(w), 2 * L.pin_stop_angle())
            best = phi if best is None else max(best, phi)
        return best

    def _close_or_open(self, target: float | None) -> None:
        want = 0.0 if target is None else target
        phi_on = L.phi_for_width(L.WIDTH_ON)
        phi_pass = L.phi_for_width(L.WIDTH_PASS)
        if self.perched and L.socket_distance(max(want, self.phi)) >= L.WIDTH_PASS - L.WIDTH_EXTRA - 1e-6:
            t = self.tilt
            pivot_y = self.origin[1] - REAR_EDGE * L.MM * math.cos(t)
            self.tilt, self.perched = 0.0, False
            self.origin = np.array([self.origin[0], pivot_y + REAR_EDGE * L.MM, L.TOP])
            self.events.append("dropped from the perch between the open sockets")
        if self.seated is not None:
            if want > phi_on + 1e-6:
                self.phi = want  # opened again off the balls
                if L.socket_distance(want) >= L.WIDTH_PASS - L.WIDTH_EXTRA:
                    self.seated = None
            else:
                self.phi = phi_on
            return
        if want >= self.phi - 1e-12:
            self.phi = want
            return
        # closing: the sockets meet whatever balls are in their way
        was_open = L.socket_distance(self.phi) >= L.WIDTH_PASS - L.WIDTH_EXTRA - 1e-6
        for i, pair in enumerate(self.balls()):
            if was_open and self._catchable(pair) and want <= phi_on:
                self.phi, self.seated = phi_on, i
                self.events.append(f"sockets caught stud pair {i}")
                if self.held_by is None:
                    self._centre_on_sockets(i)
                return
            if was_open and self._in_way(pair):
                self.phi = max(want, phi_pass)
                return
        self.phi = want

    def _catchable(self, pair) -> bool:
        b1, b2 = pair
        mid = (b1 + b2) / 2
        dy = (np.array([b1[1], b2[1]]) - self.socket_y()) / L.MM
        dz = (np.array([b1[2], b2[2]]) - L.TOP) / L.MM - L.BALL_Z
        yaw = math.degrees(math.atan2(b2[1] - b1[1], b2[0] - b1[0]))
        yaw = min(abs(yaw), abs(abs(yaw) - 180.0))
        return bool(np.all(np.abs(dy) <= L.CATCH_ALONG) and np.all(np.abs(dz) <= L.CATCH_ALONG)
                    and abs(mid[0] - L.SLIDER_ORIGIN[0]) / L.MM <= L.CATCH_X and yaw <= L.CATCH_YAW_DEG)

    def _in_way(self, pair) -> bool:
        for b in pair:
            dy = abs(b[1] - self.socket_y()) / L.MM
            dz = abs((b[2] - L.TOP) / L.MM - L.BALL_Z)
            dx = abs(b[0] - L.SLIDER_ORIGIN[0]) / L.MM
            if dy <= L.BLOCK_SIZE / 2 + L.BALL_R and dz <= 10.0 and dx <= L.WIDTH_PASS / 2:
                return True
        return False

    def _centre_on_sockets(self, i: int) -> None:
        b1, b2 = self.balls()[i]
        mid = (b1 + b2) / 2
        self.origin = self.origin + np.array([L.SLIDER_ORIGIN[0] - mid[0], self.socket_y() - mid[1], 0.0])

    # -- what the scorer sees -----------------------------------------------------------------------------------
    def joints(self) -> dict:
        return {L.ARM_A: -self.phi / 2.0, L.ARM_B: self.phi}

    def effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(origin, quat xyzw) of the effector."""
        t = self.tilt
        return self.origin.copy(), np.array([math.sin(t / 2), 0.0, 0.0, math.cos(t / 2)])
