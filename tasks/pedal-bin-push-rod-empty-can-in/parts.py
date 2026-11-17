"""The parts as lists of mechkit shapes, in the bin's frame (origin on the table under its axis, x toward the hinge)
or in each loose part's own frame. No simulator needed.

The bin's body is a round wall of 16 flat slabs (168 mm across outside, 144 mm inside) with the inner bucket's floor
50 mm up. The back slab is split round a slot (10 x 9 mm, the table to the rim) that holds the push rod, closed
behind by a narrow spine. The lid is a plate and a raised middle, a lug that reaches the hinge 7.5 mm behind the body,
and under the hinge a short arm with a round pin (r 3 mm) that hangs into the top of the slot. The pedal lever is a
plate in front of the body, an arm under the floor to its pivot and a rear arm to a round nose (r 3 mm) under the rod.
The rod (8 x 7 x 214 mm) is a separate loose part: it stands on the nose, its flat top 0.5 mm under the pin.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


def body_shapes() -> list:
    out = [mk.cylinder((0.0, 0.0, L.FLOOR_TOP - L.FLOOR_T / 2), L.IN_R, L.FLOOR_T, "Z", L.GREY)]
    radial = L.OUT_R - L.IN_R
    centre_r = (L.OUT_R + L.IN_R) / 2
    width = 2.0 * L.OUT_R * math.tan(math.pi / L.SEGMENTS) + 0.0007  # closes the outside corners
    for i in range(1, L.SEGMENTS):
        phi = 2.0 * math.pi * i / L.SEGMENTS
        out.append(mk.box((centre_r * math.cos(phi), centre_r * math.sin(phi), L.RIM / 2),
                          (radial, width, L.RIM), L.GREY, ypr=(phi, 0.0, 0.0)))
    out += back_slab_shapes(width)
    return out


def back_slab_shapes(width: float) -> list:
    """The slab at the back (phi 0) split round the rod's slot: an inner skin, two side fills and a spine behind."""
    zc, h = L.RIM / 2, L.RIM
    side = width / 2 - L.SLOT_HY
    return [
        mk.box(((L.IN_R + L.SLOT_X0) / 2, 0.0, zc), (L.SLOT_X0 - L.IN_R, width, h), L.GREY),
        mk.box(((L.SLOT_X0 + L.SLOT_X1) / 2, (L.SLOT_HY + width / 2) / 2, zc), (L.SLOT_X1 - L.SLOT_X0, side, h), L.GREY),
        mk.box(((L.SLOT_X0 + L.SLOT_X1) / 2, -(L.SLOT_HY + width / 2) / 2, zc), (L.SLOT_X1 - L.SLOT_X0, side, h),
               L.GREY),
        mk.box(((L.SLOT_X1 + L.OUT_R) / 2, 0.0, zc), (L.OUT_R - L.SLOT_X1, width, h), L.GREY),
        mk.box((L.OUT_R + 0.002, 0.0, zc), (0.004 + 2 * (L.OUT_R - L.SLOT_X1), 0.018, h), L.GREY),  # the spine
    ]


def lid_shapes() -> list:
    z0 = L.HINGE_Z
    lug_x0, lug_x1 = L.OUT_R - 0.004, L.HINGE_X + 0.004
    return [
        mk.cylinder((0.0, 0.0, z0 + L.LID_T / 2), L.LID_R, L.LID_T, "Z", L.LID_GREY),
        mk.cylinder((0.0, 0.0, z0 + L.LID_T + L.CAP_H / 2), L.CAP_R, L.CAP_H, "Z", L.LID_GREY),
        mk.box(((lug_x0 + lug_x1) / 2, 0.0, z0 + L.LID_T / 2), (lug_x1 - lug_x0, 0.040, L.LID_T), L.LID_GREY),
        mk.box((L.PIN_X, 0.0, (L.PIN_Z + z0) / 2), (0.004, L.PIN_LEN, z0 - L.PIN_Z), L.LID_GREY),  # the pin's arm
        mk.cylinder((L.PIN_X, 0.0, L.PIN_Z), L.PIN_R, L.PIN_LEN, "Y", L.LID_GREY),  # the pin
    ]


def lever_shapes() -> list:
    """The pedal lever as drawn at rest (pedal up): the plate, a riser behind it, the front arm under the floor to the
    pivot, the rear arm (6 mm wide, it runs up the slot's foot) and the round nose under the rod."""
    pz = L.PIVOT_Z
    arm_z0 = 0.030  # the front arm's and the riser's underside: 5.7 mm clear of the table at the pedal's stop
    arm_z1 = arm_z0 + 0.005
    plate_c = ((L.PLATE_X0 + L.PLATE_X1) / 2, 0.0, L.PLATE_TOP - L.PLATE_T / 2)
    riser_x0, riser_x1 = L.PLATE_X1 - 0.006, L.PLATE_X1
    riser_z0, riser_z1 = arm_z0, L.PLATE_TOP - L.PLATE_T
    return [
        mk.box(plate_c, (L.PLATE_X1 - L.PLATE_X0, 2 * L.PLATE_HY, L.PLATE_T), L.PEDAL_GREY),
        mk.box(((riser_x0 + riser_x1) / 2, 0.0, (riser_z0 + riser_z1) / 2), (0.006, 0.020, riser_z1 - riser_z0),
               L.PEDAL_GREY),
        mk.box(((riser_x0 + L.PIVOT_X) / 2, 0.0, (arm_z0 + arm_z1) / 2), (L.PIVOT_X - riser_x0, 0.020, 0.005),
               L.PEDAL_GREY),
        mk.box((L.PIVOT_X, 0.0, (pz + arm_z1) / 2), (0.006, 0.020, arm_z1 - pz), L.PEDAL_GREY),  # post to the hub
        mk.cylinder((L.PIVOT_X, 0.0, pz), 0.005, 0.024, "Y", L.PEDAL_GREY),  # the pivot's hub
        mk.box(((L.PIVOT_X + L.ROLLER_X) / 2, 0.0, pz), (L.ROLLER_X - L.PIVOT_X, 0.006, 0.005), L.PEDAL_GREY),
        mk.cylinder((L.ROLLER_X, 0.0, pz), L.ROLLER_R, 0.006, "Y", L.PEDAL_GREY),  # the nose
    ]


def rod_shapes() -> list:
    """The push rod in its own frame: origin at the middle of its foot."""
    return [mk.box((0.0, 0.0, L.ROD_LEN / 2), (L.ROD_TX, L.ROD_TY, L.ROD_LEN), L.ROD_COLOR)]


def can_shapes() -> list:
    """The can in its own frame: origin at its centre."""
    return [mk.cylinder((0.0, 0.0, 0.0), L.CAN_R, L.CAN_H, "Z", L.CAN_COLOR)]
