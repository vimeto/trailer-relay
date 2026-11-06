// Procedural low-poly vehicles, built from the simulator's vehicle specs.
// Local frame: +x = forward, +y = up, +z = right (so sim "left" = -z).
import * as THREE from 'three';

const P = (f, u, l) => new THREE.Vector3(f, u, -l);

function std(color, o = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.06, ...o });
}

class Kit {
  constructor(team) {
    this.team = new THREE.Color(team);
    this.m = {
      team: std(team, { roughness: 0.45, metalness: 0.15 }),
      teamDark: std(new THREE.Color(team).multiplyScalar(0.55), { roughness: 0.5 }),
      dark: std(0x2a2d2f, { roughness: 0.8 }),
      tyre: std(0x1b1c1d, { roughness: 0.95 }),
      rim: std(0xc9ccd1, { roughness: 0.35, metalness: 0.6 }),
      glass: std(0x3d5a70, { roughness: 0.08, metalness: 0.3, transparent: true, opacity: 0.72 }),
      chrome: std(0xdfe3e8, { roughness: 0.25, metalness: 0.85 }),
      white: std(0xf1efe9, { roughness: 0.55 }),
      grey: std(0xa7adb3, { roughness: 0.5, metalness: 0.35 }),
      rust: std(0xa8432c, { roughness: 0.7 }),
      wood: std(0x8a6a45, { roughness: 0.9 }),
      seat: std(0x1f2122, { roughness: 0.7 }),
      head: new THREE.MeshStandardMaterial({ color: 0xfff6d8, emissive: 0xfff1c0, emissiveIntensity: 0.9 }),
      tail: new THREE.MeshStandardMaterial({ color: 0xff3b2f, emissive: 0xff2a1a, emissiveIntensity: 0.8 }),
    };
    this.group = new THREE.Group();
    this.wheels = [];
  }
  box(dims, pos, mat, parent = this.group) {
    const [len, h, w] = dims;
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(len, h, w), mat);
    mesh.position.copy(P(...pos));
    mesh.castShadow = true; mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }
  cyl(r, len, pos, mat, axis = 'y', parent = this.group, seg = 18) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, seg), mat);
    if (axis === 'x') mesh.rotation.z = Math.PI / 2;
    if (axis === 'z') mesh.rotation.x = Math.PI / 2;
    mesh.position.copy(P(...pos));
    mesh.castShadow = true;
    parent.add(mesh);
    return mesh;
  }
  wheel(f, l, r, w, steer = false) {
    const pivot = new THREE.Group();
    pivot.position.copy(P(f, r, l));
    const spin = new THREE.Group();
    pivot.add(spin);
    const tyre = new THREE.Mesh(new THREE.CylinderGeometry(r, r, w, 22), this.m.tyre);
    tyre.rotation.x = Math.PI / 2; tyre.castShadow = true;
    spin.add(tyre);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.55, r * 0.55, w * 1.04, 10), this.m.rim);
    hub.rotation.x = Math.PI / 2;
    spin.add(hub);
    // a spoke so rotation is visible
    const spoke = new THREE.Mesh(new THREE.BoxGeometry(r * 1.1, r * 0.16, w * 1.06), this.m.dark);
    spin.add(spoke);
    this.group.add(pivot);
    this.wheels.push({ pivot, spin, r, steer });
  }
  bar(a, b, thick, mat, u) {
    // a, b = [f, l] at height u
    const dx = b[0] - a[0], dl = b[1] - a[1];
    const len = Math.hypot(dx, dl);
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(len, thick, thick), mat);
    mesh.position.copy(P((a[0] + b[0]) / 2, u, (a[1] + b[1]) / 2));
    mesh.rotation.y = Math.atan2(dl, dx);
    mesh.castShadow = true;
    this.group.add(mesh);
  }
  done() { return { group: this.group, wheels: this.wheels, kit: this }; }
}

// ------------------------------------------------------------------ vehicles
function car(s, team) {
  const k = new Kit(team);
  const L = s.body_front + s.body_rear, cf = (s.body_front - s.body_rear) / 2, W = s.body_width;
  k.box([L, 0.55, W], [cf, 0.62, 0], k.m.team);
  k.box([L * 0.98, 0.16, W * 0.98], [cf, 0.3, 0], k.m.dark);
  k.box([L * 0.5, 0.5, W * 0.86], [cf - 0.35, 1.13, 0], k.m.glass);
  k.box([L * 0.48, 0.06, W * 0.84], [cf - 0.35, 1.41, 0], k.m.team);
  k.box([0.08, 0.12, 0.35], [s.body_front, 0.72, W / 2 - 0.3], k.m.head);
  k.box([0.08, 0.12, 0.35], [s.body_front, 0.72, -W / 2 + 0.3], k.m.head);
  k.box([0.06, 0.12, 0.3], [-s.body_rear, 0.75, W / 2 - 0.25], k.m.tail);
  k.box([0.06, 0.12, 0.3], [-s.body_rear, 0.75, -W / 2 + 0.25], k.m.tail);
  k.bar([-s.body_rear, 0], [-s.hitch_offset, 0], 0.08, k.m.dark, 0.45);
  k.cyl(0.06, 0.12, [-s.hitch_offset, 0.52, 0], k.m.chrome);
  const r = 0.33, w = 0.22, l = W / 2 - 0.14;
  k.wheel(0, l, r, w); k.wheel(0, -l, r, w);
  k.wheel(s.wheelbase, l, r, w, true); k.wheel(s.wheelbase, -l, r, w, true);
  return k.done();
}

function mower(s, team) {
  const k = new Kit(team);
  const W = s.body_width;
  k.box([1.55, 0.22, 0.82], [0.62, 0.36, 0], k.m.team);
  k.box([0.62, 0.34, 0.74], [1.22, 0.56, 0], k.m.team);
  k.box([0.7, 0.1, W], [0.66, 0.16, 0], k.m.dark);
  k.box([0.34, 0.08, 0.46], [0.1, 0.6, 0], k.m.seat);
  k.box([0.08, 0.36, 0.46], [-0.08, 0.8, 0], k.m.seat);
  k.bar([0.55, 0], [0.75, 0], 0.05, k.m.dark, 0.85);
  k.cyl(0.16, 0.03, [0.72, 0.98, 0], k.m.dark);
  k.box([0.04, 0.08, 0.2], [s.body_front, 0.56, 0.2], k.m.head);
  k.box([0.04, 0.08, 0.2], [s.body_front, 0.56, -0.2], k.m.head);
  k.bar([-s.body_rear, 0], [-s.hitch_offset, 0], 0.05, k.m.dark, 0.26);
  k.wheel(0, W / 2 - 0.1, 0.3, 0.2); k.wheel(0, -W / 2 + 0.1, 0.3, 0.2);
  k.wheel(s.wheelbase, 0.38, 0.18, 0.12, true); k.wheel(s.wheelbase, -0.38, 0.18, 0.12, true);
  return k.done();
}

function tractor(s, team) {
  const k = new Kit(team);
  k.box([2.2, 0.42, 0.62], [1.45, 0.72, 0], k.m.dark);
  k.box([1.9, 0.86, 0.84], [1.95, 1.2, 0], k.m.team);
  k.box([0.1, 0.5, 0.7], [s.body_front - 0.05, 1.15, 0], k.m.dark);
  k.box([0.06, 0.12, 0.2], [s.body_front, 1.32, 0.25], k.m.head);
  k.box([0.06, 0.12, 0.2], [s.body_front, 1.32, -0.25], k.m.head);
  k.box([1.45, 1.2, 1.38], [0.12, 2.12, 0], k.m.glass);
  k.box([1.6, 0.1, 1.55], [0.12, 2.77, 0], k.m.team);
  for (const [f, l] of [[0.82, 0.68], [0.82, -0.68], [-0.58, 0.68], [-0.58, -0.68]]) k.box([0.07, 1.25, 0.07], [f, 2.1, l], k.m.dark);
  k.box([1.15, 0.1, 0.55], [0, 1.66, 0.92], k.m.team);
  k.box([1.15, 0.1, 0.55], [0, 1.66, -0.92], k.m.team);
  k.cyl(0.06, 1.0, [1.55, 2.2, 0.28], k.m.dark);
  k.bar([-s.body_rear, 0], [-s.hitch_offset, 0], 0.12, k.m.dark, 0.6);
  k.wheel(0, 0.9, 0.8, 0.5); k.wheel(0, -0.9, 0.8, 0.5);
  k.wheel(s.wheelbase, 0.78, 0.47, 0.3, true); k.wheel(s.wheelbase, -0.78, 0.47, 0.3, true);
  return k.done();
}

function truck(s, team) {
  const k = new Kit(team);
  const W = s.body_width;
  k.box([6.0, 0.32, 1.0], [2.0, 0.78, 0], k.m.dark);
  k.box([2.25, 2.55, W], [s.body_front - 1.12, 2.25, 0], k.m.team);
  k.box([0.06, 1.0, W * 0.86], [s.body_front + 0.01, 2.75, 0], k.m.glass);
  k.box([0.1, 0.5, W], [s.body_front, 0.9, 0], k.m.chrome);
  k.box([0.06, 0.16, 0.32], [s.body_front + 0.05, 1.2, W / 2 - 0.3], k.m.head);
  k.box([0.06, 0.16, 0.32], [s.body_front + 0.05, 1.2, -W / 2 + 0.3], k.m.head);
  k.box([2.25, 0.45, W * 0.96], [s.body_front - 1.12, 3.72, 0], k.m.teamDark);
  k.cyl(0.32, 1.3, [2.6, 0.9, W / 2 - 0.2], k.m.chrome, 'x');
  k.cyl(0.08, 2.2, [s.body_front - 2.35, 3.1, W / 2 - 0.15], k.m.chrome);
  k.cyl(0.55, 0.12, [-s.hitch_offset, 1.0, 0], k.m.dark);
  const r = 0.52;
  k.wheel(s.wheelbase, W / 2 - 0.18, r, 0.32, true); k.wheel(s.wheelbase, -W / 2 + 0.18, r, 0.32, true);
  for (const f of [0.66, -0.66]) { k.wheel(f, W / 2 - 0.28, r, 0.52); k.wheel(f, -W / 2 + 0.28, r, 0.52); }
  return k.done();
}

// ------------------------------------------------------------------ trailers (origin = trailer axle)
function openBox(k, s, floorU, wallH, wallMat, floorMat, rim) {
  const L = s.trailer_front + s.trailer_rear, cf = (s.trailer_front - s.trailer_rear) / 2, W = s.trailer_width, t = 0.06;
  k.box([L, 0.1, W], [cf, floorU, 0], floorMat);
  k.box([L, wallH, t], [cf, floorU + wallH / 2, W / 2 - t / 2], wallMat);
  k.box([L, wallH, t], [cf, floorU + wallH / 2, -W / 2 + t / 2], wallMat);
  k.box([t, wallH, W], [s.trailer_front - t / 2, floorU + wallH / 2, 0], wallMat);
  k.box([t, wallH, W], [-s.trailer_rear + t / 2, floorU + wallH / 2, 0], wallMat);
  if (rim) {
    k.box([L, 0.07, t * 1.3], [cf, floorU + wallH, W / 2 - t / 2], rim);
    k.box([L, 0.07, t * 1.3], [cf, floorU + wallH, -W / 2 + t / 2], rim);
  }
  k.box([0.04, 0.08, 0.22], [-s.trailer_rear - 0.01, floorU + 0.08, W / 2 - 0.2], k.m.tail);
  k.box([0.04, 0.08, 0.22], [-s.trailer_rear - 0.01, floorU + 0.08, -W / 2 + 0.2], k.m.tail);
}

function boxTrailer(s, team) {
  const k = new Kit(team);
  openBox(k, s, 0.62, 0.45, k.m.grey, k.m.dark, k.m.team);
  k.bar([s.trailer_front, 0.55], [s.trailer_length, 0], 0.07, k.m.dark, 0.55);
  k.bar([s.trailer_front, -0.55], [s.trailer_length, 0], 0.07, k.m.dark, 0.55);
  for (const l of [s.trailer_width / 2 + 0.05, -s.trailer_width / 2 - 0.05]) k.box([0.8, 0.05, 0.26], [0, 0.72, l], k.m.dark);
  k.wheel(0, s.trailer_width / 2 + 0.05, 0.3, 0.2); k.wheel(0, -s.trailer_width / 2 - 0.05, 0.3, 0.2);
  return k.done();
}

function gardenCart(s, team) {
  const k = new Kit(team);
  openBox(k, s, 0.38, 0.32, k.m.teamDark, k.m.dark, null);
  k.bar([s.trailer_front, 0], [s.trailer_length, 0], 0.05, k.m.dark, 0.3);
  k.wheel(0, s.trailer_width / 2 + 0.06, 0.2, 0.12); k.wheel(0, -s.trailer_width / 2 - 0.06, 0.2, 0.12);
  return k.done();
}

function tippingTrailer(s, team) {
  const k = new Kit(team);
  openBox(k, s, 1.08, 0.95, k.m.rust, k.m.dark, k.m.team);
  k.box([s.trailer_front + s.trailer_rear - 0.6, 0.25, 0.5], [(s.trailer_front - s.trailer_rear) / 2, 0.85, 0], k.m.dark);
  k.bar([s.trailer_front, 0], [s.trailer_length, 0], 0.14, k.m.dark, 0.62);
  k.wheel(0, 0.95, 0.55, 0.4); k.wheel(0, -0.95, 0.55, 0.4);
  return k.done();
}

function semiTrailer(s, team) {
  const k = new Kit(team);
  const L = s.trailer_front + s.trailer_rear, cf = (s.trailer_front - s.trailer_rear) / 2, W = s.trailer_width;
  k.box([L, 2.75, W], [cf, 2.68, 0], k.m.white);
  k.box([L * 0.97, 0.32, 0.02], [cf, 2.0, W / 2 + 0.012], k.m.team);
  k.box([L * 0.97, 0.32, 0.02], [cf, 2.0, -W / 2 - 0.012], k.m.team);
  k.box([L * 0.97, 0.08, 0.02], [cf, 3.45, W / 2 + 0.012], k.m.team);
  k.box([L * 0.97, 0.08, 0.02], [cf, 3.45, -W / 2 - 0.012], k.m.team);
  k.box([L, 0.25, 1.0], [cf, 1.18, 0], k.m.dark);
  for (const l of [0.8, -0.8]) k.box([0.12, 1.1, 0.12], [s.trailer_front - 2.2, 0.7, l], k.m.grey);
  k.box([0.04, 0.12, 0.3], [-s.trailer_rear - 0.01, 1.35, W / 2 - 0.25], k.m.tail);
  k.box([0.04, 0.12, 0.3], [-s.trailer_rear - 0.01, 1.35, -W / 2 + 0.25], k.m.tail);
  for (const f of [-1.3, 0, 1.3]) { k.wheel(f, W / 2 - 0.3, 0.5, 0.5); k.wheel(f, -W / 2 + 0.3, 0.5, 0.5); }
  return k.done();
}

// ------------------------------------------------------------------ mirrors and the reversing-camera screen
// Mirror heads sit at the rig's mirror positions (relay/perception.py rig_for), taller than wide (the truck's clearly
// tall), with the glass turned to bisect the line to the driver's eye and the mirror's view axis.
// h/w/d = glass housing height / width / depth, u = centre height, arms = [from [f, l, u] -> to the head] in the vehicle frame
export const MIRROR_STYLE = {
  car: { h: 0.19, w: 0.12, d: 0.06, u: 1.0, housing: 'team', screenU: 1.02, head: 1.15 },
  tractor: { h: 0.32, w: 0.17, d: 0.06, u: 2.35, housing: 'dark', screenU: 2.0, head: 2.35 },
  truck: { h: 0.62, w: 0.22, d: 0.08, u: 2.75, housing: 'dark', screenU: 2.4, head: 2.8 },
  mower: { h: 0.2, w: 0.12, d: 0.05, u: 1.0, housing: 'dark', screenU: 0.9, head: 1.05 },
};
const GLOW = new THREE.Color('#fff4d6');

function addMirrors(k, s, rig, key) {
  const st = MIRROR_STYLE[key] || MIRROR_STYLE.car;
  const out = {};
  for (const m of rig?.mirrors || []) {
    const [mx, my] = m.pos, side = Math.sign(my) || 1;
    // glass normal: halfway between "towards the eye" and the mirror's view axis
    const toEye = Math.atan2(rig.eye[1] - my, rig.eye[0] - mx), ax = m.axis_deg * Math.PI / 180;
    const nrm = Math.atan2(Math.sin(toEye) + Math.sin(ax), Math.cos(toEye) + Math.cos(ax));
    const head = new THREE.Group();
    head.position.copy(P(mx, st.u, my));
    head.rotation.y = nrm;
    const housing = new THREE.Mesh(new THREE.BoxGeometry(st.d, st.h, st.w), st.housing === 'team' ? k.m.team : k.m.dark);
    housing.castShadow = true;
    const glassMat = std(0xcfdde6, { roughness: 0.12, metalness: 0.75, emissive: 0x000000 });
    const glass = new THREE.Mesh(new THREE.BoxGeometry(0.012, st.h * 0.84, st.w * 0.8), glassMat);
    glass.position.x = st.d / 2 + 0.004;
    head.add(housing, glass);
    k.group.add(head);
    // arms from the body side to the inner edge of the head
    const inner = my - side * st.w * 0.45;
    if (key === 'truck') {
      const body = side * (s.body_width / 2 - 0.02);
      for (const du of [st.h * 0.38, -st.h * 0.38]) k.bar([mx - 0.05, body], [mx, inner], 0.035, k.m.dark, st.u + du);
    } else if (key === 'tractor') {
      // from the cab's front corner post, out and forward on a long arm
      const post = [0.82, side * 0.68], u0 = 2.55;
      const len = Math.hypot(mx - post[0], inner - post[1], st.u - u0);
      const arm = new THREE.Mesh(new THREE.BoxGeometry(len, 0.035, 0.035), k.m.dark);
      arm.position.copy(P((mx + post[0]) / 2, (st.u + u0) / 2, (inner + post[1]) / 2));
      arm.lookAt(arm.position.clone().add(P(mx - post[0], st.u - u0, inner - post[1])));
      arm.rotateY(-Math.PI / 2);
      k.group.add(arm);
    } else {
      k.bar([mx - 0.04, side * (s.body_width / 2 - 0.04)], [mx, inner], 0.05, st.housing === 'team' ? k.m.team : k.m.dark, st.u - st.h * 0.3);
    }
    out[m.name] = { head, glass, mat: glassMat, u: st.u, pos: m.pos };
  }
  return out;
}

// a small glow on the dashboard screen (seen through the cab) while the driver watches the reversing camera
function addScreen(k, s, rig, key) {
  if (!rig?.screen) return null;
  const st = MIRROR_STYLE[key] || MIRROR_STYLE.car;
  // the low-poly car's glasshouse ends before the rig's screen position: keep the glow on the dashboard, inside
  const x = key === 'car' ? Math.min(rig.screen[0], (s.body_front - s.body_rear) / 2 - 0.35 + (s.body_front + s.body_rear) * 0.25 - 0.15) : rig.screen[0];
  const mat = new THREE.MeshBasicMaterial({ color: GLOW, transparent: true, opacity: 0.95, depthTest: false, depthWrite: false });
  const scr = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.16, 0.26), mat);
  scr.position.copy(P(x, st.screenU, rig.screen[1]));
  scr.renderOrder = 9;
  scr.visible = false;
  scr.userData.u = st.screenU;
  k.group.add(scr);
  return scr;
}

/** Glow the mirror the driver is looking at, or the screen ('camera'); null = nothing. */
export function setLook(combo, at) {
  if (combo._look === at) return;
  combo._look = at;
  for (const [name, m] of Object.entries(combo.mirrors)) {
    const on = name === at;
    m.mat.emissive.copy(on ? GLOW : new THREE.Color(0));
    m.mat.emissiveIntensity = on ? 1.6 : 0;
    m.head.scale.setScalar(on ? 1.18 : 1);
  }
  if (combo.screen) combo.screen.visible = at === 'camera';
}

const BUILDERS = {
  car: [car, boxTrailer], mower: [mower, gardenCart], tractor: [tractor, tippingTrailer], truck: [truck, semiTrailer],
};

export function buildCombo(spec, team, rig = null) {
  const [v, t] = BUILDERS[spec.key] || BUILDERS.car;
  const vehicle = v(spec, team);
  const trailer = t(spec, team);
  const mirrors = addMirrors(vehicle.kit, spec, rig, spec.key);
  const screen = addScreen(vehicle.kit, spec, rig, spec.key);
  const root = new THREE.Group();
  root.add(vehicle.group, trailer.group);
  const mats = new Set();
  root.traverse((o) => { if (o.material) mats.add(o.material); });
  for (const m of mats) { m.transparent = true; m.userData.baseOpacity = m.opacity; }
  return { root, vehicle, trailer, mirrors, screen, style: MIRROR_STYLE[spec.key] || MIRROR_STYLE.car, mats: [...mats], height: { car: 1.5, mower: 1.0, tractor: 2.9, truck: 4.0 }[spec.key] || 2 };
}

export function setOpacity(combo, a) {
  if (combo._a === a) return;
  combo._a = a;
  for (const m of combo.mats) {
    m.opacity = (m.userData.baseOpacity ?? 1) * a;
    m.depthWrite = true;
  }
  combo.root.traverse((o) => { if (o.isMesh) o.castShadow = a > 0.5; });
}

// Pose the combination from a sim state [x, y, th1, th2], steering and odometer.
export function poseCombo(combo, spec, st) {
  const { x, y, th1, th2, steer, odo } = st;
  const hx = x - spec.hitch_offset * Math.cos(th1), hy = y - spec.hitch_offset * Math.sin(th1);
  const ax = hx - spec.trailer_length * Math.cos(th2), ay = hy - spec.trailer_length * Math.sin(th2);
  combo.vehicle.group.position.set(x, 0, -y);
  combo.vehicle.group.rotation.y = th1;
  combo.trailer.group.position.set(ax, 0, -ay);
  combo.trailer.group.rotation.y = th2;
  for (const w of combo.vehicle.wheels) {
    if (w.steer) w.pivot.rotation.y = steer;
    w.spin.rotation.z = -odo / w.r;
  }
  for (const w of combo.trailer.wheels) w.spin.rotation.z = -odo / w.r;
  return { hx, hy, ax, ay };
}
