"""The scene's shapes and joints (mechkit), from layout.py. NumPy and mechkit only.

Three task objects:
- ``motor`` (fixed): the Cuisinart Smart Stick CSB-175 motor handle resting in a low countertop cradle (base) plus
  the spring-loaded release button on top (prismatic slider along -Z, 6 mm stroke, 3.5 N spring);
- ``wand`` (loose): the stainless steel blending wand (18 mm coupling spigot seated in the motor's socket, coupling
  collar, 18 mm shaft, octagonal 64 mm bell guard with blunt recessed blade hub), latched to ``motor`` at spawn by a
  mechkit bond that releases while the release button is pressed >= 4 mm and re-engages if the button is released
  before the wand is withdrawn 5 mm; once released the spigot is still held in the socket and the wand only comes off
  by being pulled 18 mm straight out along its axis;
- ``bench`` (fixed): the unpowered benchtop outlet block at (34, 20) cm with the molded NEMA 1-15P plug on a prismatic
  slider along +Y (4.0 N friction, 35 mm travel, 18 mm blades) and the blue sink wash tray (24 x 14 x 4 cm) at
  (26, -24) cm.
"""

from __future__ import annotations

import math

import layout as L
import mechkit as mk


# -- motor body in countertop cradle + release button -------------------------------------------------------------
def motor_shapes() -> list:
    """Table-frame shapes relative to motor origin (BLENDER_X, SEAM_Y, TOP)."""
    s = []
    # Main cylindrical motor handle along Y behind the coupling socket: local y in [SOCKET_DEPTH, MOTOR_LEN]
    body_len = L.MOTOR_LEN - L.SOCKET_DEPTH
    s.append(mk.cylinder((0.0, L.SOCKET_DEPTH + body_len / 2, L.AXIS_Z), L.MOTOR_R, body_len, axis="Y",
                         color=L.MOTOR_DARK))
    # Coupling socket (brushed stainless front ring): 12 wall boxes around the Y axis leaving a 12-sided bore of
    # inscribed radius SOCKET_APOTHEM, SOCKET_DEPTH deep, open toward -y. The wand's spigot sits in it.
    s.extend(socket_shapes())
    # Top grip cap at +y end
    s.append(mk.cylinder((0.0, L.MOTOR_LEN - 0.010, L.AXIS_Z), L.MOTOR_R + 0.001, 0.020, axis="Y", color=L.MOTOR_SILVER))
    # Low countertop cradle supporting the motor body (local y in [0.015, MOTOR_LEN])
    cy0 = L.CRADLE_Y[0] - L.SEAM_Y
    cy1 = L.CRADLE_Y[1] - L.SEAM_Y
    clen = cy1 - cy0
    cy_mid = (cy0 + cy1) / 2
    # Base slab under motor
    s.append(mk.box((0.0, cy_mid, L.CRADLE_BASE_H / 2), (2 * L.CRADLE_HALF_X, clen, L.CRADLE_BASE_H), L.CRADLE_WOOD))
    # Left and right cradle side cheeks
    cheek_w = L.CRADLE_HALF_X - L.MOTOR_R
    cheek_x = L.MOTOR_R + cheek_w / 2
    cheek_h = L.CRADLE_WALL_H - L.CRADLE_BASE_H
    cheek_z = L.CRADLE_BASE_H + cheek_h / 2
    s.append(mk.box((cheek_x, cy_mid, cheek_z), (cheek_w, clen, cheek_h), L.CRADLE_WOOD))
    s.append(mk.box((-cheek_x, cy_mid, cheek_z), (cheek_w, clen, cheek_h), L.CRADLE_WOOD))
    # Rear stop block behind the motor handle (plain rest; no strap, clamp or lip over or around the handle)
    s.append(mk.box((0.0, cy1 + 0.006, L.CRADLE_WALL_H / 2), (2 * L.CRADLE_HALF_X, 0.012, L.CRADLE_WALL_H),
                    L.CRADLE_WOOD))
    return s


def socket_shapes() -> list:
    """The motor's coupling socket: SOCKET_SIDES wall boxes, local y in [0, SOCKET_DEPTH], around the motor axis."""
    n = L.SOCKET_SIDES
    thick = L.MOTOR_R - L.SOCKET_APOTHEM
    width = 2 * L.MOTOR_R * math.tan(math.pi / n)
    r_mid = L.SOCKET_APOTHEM + thick / 2
    s = []
    for k in range(n):
        a = 2 * math.pi * k / n
        cx, cz = r_mid * math.cos(a), r_mid * math.sin(a)
        # box local x is radial, local z tangential; rotate about Y by -a (pitch) so local x points at angle a in xz
        s.append(mk.box((cx, L.SOCKET_DEPTH / 2, L.AXIS_Z + cz), (thick, L.SOCKET_DEPTH, width), L.MOTOR_SILVER,
                        ypr=(0.0, -a, 0.0)))
    return s


def release_button_joint() -> mk.Joint:
    """Spring-loaded release button on top of the motor handle, sliding along -Z."""
    bx, by, bz = L.BUTTON_SIZE
    ly = L.BUTTON_Y - L.SEAM_Y
    cap = mk.box((0.0, ly, L.MOTOR_TOP_Z + bz / 2), (bx, by, bz), L.BUTTON_RED)
    return mk.slider("release", "-Z", [cap], upper=L.BUTTON_TRAVEL, mass=0.015,
                     spring=L.BUTTON_SPRING, damping=L.BUTTON_DAMPING)


# -- stainless blending wand (loose) ------------------------------------------------------------------------------
def wand_shapes() -> list:
    """Wand frame: origin on the centerline at the coupling seam (0, 0, 0); wand extends along -Y."""
    s = []
    # Coupling spigot: y in [0.0, SPIGOT_LEN], seated in the motor's socket (the 18 mm axial engagement)
    s.append(mk.cylinder((0.0, L.SPIGOT_LEN / 2, 0.0), L.SPIGOT_R, L.SPIGOT_LEN, axis="Y", color=L.STEEL_DARK))
    # Coupling collar: y in [-COLLAR_LEN, 0.0]
    s.append(mk.cylinder((0.0, -L.COLLAR_LEN / 2, 0.0), L.COLLAR_R, L.COLLAR_LEN, axis="Y", color=L.STEEL_DARK))
    # Stainless shaft: y in [SHAFT_Y[0], SHAFT_Y[1]]
    sy0, sy1 = L.SHAFT_Y
    s.append(mk.cylinder((0.0, (sy0 + sy1) / 2, 0.0), L.SHAFT_R, sy1 - sy0, axis="Y", color=L.MOTOR_SILVER))
    # Plain 18 mm stainless shaft: pinched as is (no grip sleeve, flange or other add-on)
    # Octagonal bell guard: y in [BELL_Y[0], BELL_Y[1]] (4 rotated boxes about Y so it rests flat without rolling)
    by0, by1 = L.BELL_Y
    blen = by1 - by0
    by_mid = (by0 + by1) / 2
    apothem = L.BELL_R * math.cos(math.pi / 8)
    chord = 2 * L.BELL_R * math.sin(math.pi / 8)
    for k in range(4):
        pitch = k * math.pi / 4
        s.append(mk.box((0.0, by_mid, 0.0), (chord, blen, 2 * apothem), L.MOTOR_SILVER, ypr=(0.0, pitch, 0.0)))
    # Recessed blunt blade hub disc inside bell end (visual stand-in)
    s.append(mk.cylinder((0.0, by0 + 0.008, 0.0), 0.018, 0.006, axis="Y", color=L.STEEL_DARK))
    return s


# -- bench: unpowered outlet block, NEMA 1-15P plug slider, blue sink wash tray -----------------------------------
def outlet_shapes() -> list:
    """Fixed unpowered benchtop outlet block at (34, 20) cm, face pointing +Y."""
    ox, oy = float(L.OUTLET_C[0]), float(L.OUTLET_C[1])
    sx, sy, sz = L.OUTLET_SIZE
    s = [mk.box((ox, oy, sz / 2), (sx, sy, sz), L.OUTLET_WHITE)]
    # Dark receptacle slot face on +y side
    s.append(mk.box((ox, L.OUTLET_FACE_Y + 0.0005, L.PLUG_Z), (0.022, 0.001, 0.026), (0.25, 0.25, 0.26)))
    return s


def plug_joint() -> mk.Joint:
    """Molded NEMA 1-15P plug body in front of the +Y outlet face, sliding along +Y with 4.0 N friction."""
    ox = float(L.OUTLET_C[0])
    px, py, pz = L.PLUG_SIZE
    y0 = L.OUTLET_FACE_Y
    body = mk.box((ox, y0 + py / 2, L.PLUG_Z), (px, py, pz), L.PLUG_BLACK)
    # Strain relief tail and drop toward the cord run on the table
    stub = mk.cylinder((ox, y0 + py + 0.008, L.PLUG_Z), L.CORD_R, 0.016, axis="Y", color=L.PLUG_BLACK)
    drop = mk.box((ox, y0 + py + 0.016, (L.PLUG_Z + L.CORD_R) / 2), (2 * L.CORD_R, 2 * L.CORD_R, L.PLUG_Z - L.CORD_R),
                  L.PLUG_BLACK)
    # Two brass NEMA 1-15P prongs extending into the outlet block along -y (length BLADE_LEN = 18 mm)
    prong_y = y0 - L.BLADE_LEN / 2
    p_left = mk.box((ox - 0.006, prong_y, L.PLUG_Z), (0.002, L.BLADE_LEN, 0.006), L.BRASS_BLADE)
    p_right = mk.box((ox + 0.006, prong_y, L.PLUG_Z), (0.002, L.BLADE_LEN, 0.006), L.BRASS_BLADE)
    return mk.slider("plug", "Y", [body, stub, drop, p_left, p_right], upper=L.PLUG_UPPER, mass=0.035,
                     damping=L.PLUG_DAMPING, friction=L.PLUG_FRICTION)


def cord_shapes() -> list:
    """Visible black power cord routed on the table from beside the plug's tail around behind the cradle into the
    rear (+y) end of the immersion blender motor handle (fixed visual run, matching palm-router)."""
    r = L.CORD_R
    z = r + 0.0005
    # Start just inside the plug tail's -x side (x = 0.324, y = 0.285) so it never blocks the plug slider or gripper
    x_plug = float(L.OUTLET_C[0]) - 0.016
    y_start = L.OUTLET_FACE_Y + L.PLUG_SIZE[1] + 0.018
    y_back = 0.330
    x_motor = L.BLENDER_X
    y_motor_rear = L.SEAM_Y + L.MOTOR_LEN + 0.010
    return [
        # Short stub from plug tail line to cord run
        mk.box(((x_plug + float(L.OUTLET_C[0])) / 2, y_start, z),
               (float(L.OUTLET_C[0]) - x_plug + 2 * r, 2 * r, 2 * r), L.PLUG_BLACK),
        # Run back along +Y clear of the plug's pull path
        mk.box((x_plug, (y_start + y_back) / 2, z), (2 * r, y_back - y_start + 2 * r, 2 * r), L.PLUG_BLACK),
        # Run across along -X behind the motor cradle
        mk.box(((x_plug + x_motor) / 2, y_back, z), (x_plug - x_motor + 2 * r, 2 * r, 2 * r), L.PLUG_BLACK),
        # Run forward along -Y into the rear of the immersion blender motor handle
        mk.box((x_motor, (y_back + y_motor_rear) / 2, z), (2 * r, y_back - y_motor_rear + 2 * r, 2 * r), L.PLUG_BLACK),
        # Short vertical riser into the rear end of the motor handle centerline
        mk.box((x_motor, y_motor_rear + 0.003, (z + L.AXIS_Z) / 2), (2 * r, 2 * r, L.AXIS_Z - z), L.PLUG_BLACK),
    ]


def tray_shapes() -> list:
    """Blue sink wash tray (24 x 14 x 4 cm) at (26, -24) cm."""
    cx, cy = float(L.TRAY_C[0]), float(L.TRAY_C[1])
    ox, oy, oz = L.TRAY_OUTER
    return list(mk.open_box((cx, cy, oz / 2), (ox, oy, oz), wall=L.TRAY_WALL, floor=L.TRAY_FLOOR, color=L.TRAY_BLUE))


def bench_shapes() -> list:
    return outlet_shapes() + cord_shapes() + tray_shapes()
