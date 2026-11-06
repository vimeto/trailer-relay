// Themed surroundings per vehicle: the courses keep their shapes, only the place around them changes.
//   car     -> parking lot   (asphalt, painted bays, kerbs, lamp posts)
//   tractor -> farmyard      (dirt, hay bales, barn, fence)
//   mower   -> garden        (striped lawn, hedges, flower beds, a shed)
//   truck   -> sand quarry   (sand, sand and gravel piles, an excavator, a conveyor)
// Low-poly and instanced, so it stays cheap on software GL (swiftshader).
import * as THREE from 'three';
import { corridorDiscs } from './geom2d.js';

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}
function noise(g, w, h, base, specks, n, seed) {
  const r = rng(seed);
  g.fillStyle = base; g.fillRect(0, 0, w, h);
  for (let i = 0; i < n; i++) {
    g.fillStyle = specks[Math.floor(r() * specks.length)];
    const s = 1 + r() * 2.2;
    g.fillRect(r() * w, r() * h, s, s);
  }
  return r;
}

// ground textures; size = metres covered by one texture tile
const TEX = {
  lot: () => ({ size: 9, tex: canvasTex(256, 256, (g, w, h) => {
    const r = noise(g, w, h, '#45484b', ['rgba(30,32,34,0.5)', 'rgba(120,122,124,0.35)', 'rgba(80,82,85,0.5)'], 5200, 3);
    g.strokeStyle = 'rgba(25,26,27,0.35)'; g.lineWidth = 1.2;
    for (let i = 0; i < 4; i++) { g.beginPath(); let x = r() * w, y = r() * h; g.moveTo(x, y); for (let k = 0; k < 5; k++) { x += (r() - 0.5) * 50; y += (r() - 0.5) * 50; g.lineTo(x, y); } g.stroke(); }
  }) }),
  dirt: () => ({ size: 10, tex: canvasTex(256, 256, (g, w, h) => {
    const r = noise(g, w, h, '#8a7150', ['rgba(110,88,60,0.6)', 'rgba(160,136,100,0.5)', 'rgba(96,78,52,0.5)', 'rgba(120,128,70,0.25)'], 6500, 5);
    for (let i = 0; i < 26; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '70,56,38' : '150,128,92'},0.12)`; g.beginPath(); g.ellipse(r() * w, r() * h, 10 + r() * 30, 5 + r() * 12, r() * 3, 0, 7); g.fill(); }
  }) }),
  lawn: () => ({ size: 4, tex: canvasTex(128, 128, (g, w, h) => {
    g.fillStyle = '#6e9a44'; g.fillRect(0, 0, w / 2, h);
    g.fillStyle = '#5c873a'; g.fillRect(w / 2, 0, w / 2, h);
    const r = rng(9);
    for (let i = 0; i < 1600; i++) { g.fillStyle = r() < 0.5 ? 'rgba(40,70,30,0.25)' : 'rgba(150,190,100,0.2)'; g.fillRect(r() * w, r() * h, 1, 2 + r() * 2); }
  }) }),
  sand: () => ({ size: 12, tex: canvasTex(256, 256, (g, w, h) => {
    const r = noise(g, w, h, '#d4b98a', ['rgba(170,140,95,0.5)', 'rgba(236,220,186,0.55)', 'rgba(150,130,100,0.35)'], 7000, 13);
    g.strokeStyle = 'rgba(160,130,90,0.22)'; g.lineWidth = 2;
    for (let k = 0; k < 9; k++) { g.beginPath(); const y = r() * h; g.moveTo(0, y); for (let x = 0; x <= w; x += 16) g.lineTo(x, y + Math.sin(x / 30 + k) * 5); g.stroke(); }
  }) }),
};
// the course surface and the apron strip just outside the rails, per theme
export const SURFACES = {
  car: { base: '#6c6f71', specks: ['rgba(40,42,44,0.45)', 'rgba(150,152,154,0.35)'], apron: '#b8b5ad' },
  tractor: { base: '#a88f68', specks: ['rgba(120,98,66,0.5)', 'rgba(196,176,140,0.45)'], apron: '#6f5d42' },
  mower: { base: '#cdbf9c', specks: ['rgba(150,135,105,0.5)', 'rgba(236,228,206,0.5)'], apron: '#7b5f40' },
  truck: { base: '#a39785', specks: ['rgba(110,100,86,0.5)', 'rgba(196,188,172,0.45)'], apron: '#bfa678' },
};
const THEME = { car: 'lot', tractor: 'dirt', mower: 'lawn', truck: 'sand' };
const texCache = {};
export function surfaceTexture(vehicle) {
  const s = SURFACES[vehicle] || SURFACES.car;
  return (texCache[vehicle] ||= canvasTex(256, 256, (g, w, h) => noise(g, w, h, s.base, s.specks, 6000, 11)));
}

const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.9, flatShading: true, ...o });
const V = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function inPoly(x, y, P) {
  let inside = false;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
    const [xi, yi] = P[i], [xj, yj] = P[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
class Placer {
  // free space test against the course (centreline corridor, walls) and earlier props
  constructor(t, rect, others) {
    const g = t.tr.geometry;
    this.rect = rect; this.others = others; this.taken = [];
    this.corr = [];
    for (const [x, y, r] of corridorDiscs(g, 1.5)) this.corr.push([...t.toWorld(x, y), r]);
    for (const w of g.walls || []) for (const [x, y] of resampleLine(w, 1.0)) this.corr.push([...t.toWorld(x, y), 0.3]);
    for (const z of g.zones || []) for (const [x, y] of z.polygon || []) this.corr.push([...t.toWorld(x, y), 0.5]);
    const dc = g.dock_corners || [];
    for (const [x, y] of dc) this.corr.push([...t.toWorld(x, y), 1.0]);
    // a walls-only yard: keep the whole enclosed box clear
    this.box = null;
    // closed wall polygons (turnaround kerbs): nothing inside
    this.polys = (g.walls || []).filter((w) => w?.length > 3 && Math.hypot(w[0][0] - w[w.length - 1][0], w[0][1] - w[w.length - 1][1]) < 1e-3)
      .map((w) => w.map(([x, y]) => t.toWorld(x, y)));
    if (!g.centerline?.length && g.walls?.length) {
      const pts = g.walls.flat().map(([x, y]) => t.toWorld(x, y));
      this.box = [Math.min(...pts.map((p) => p[0])) - 1.5, Math.max(...pts.map((p) => p[0])) + 1.5, Math.min(...pts.map((p) => p[1])) - 1.5, Math.max(...pts.map((p) => p[1])) + 1.5];
    }
  }
  clearance(x, y) {
    let c = 1e9;
    for (const [px, py, hw] of this.corr) c = Math.min(c, Math.hypot(px - x, py - y) - hw);
    if (this.box && x > this.box[0] && x < this.box[1] && y > this.box[2] && y < this.box[3]) c = -1;
    for (const P of this.polys) if (inPoly(x, y, P)) c = -1;
    return c;
  }
  free(x, y, r, pad = 1.2) {
    if (this.clearance(x, y) < r + pad) return false;
    for (const [px, py, pr] of this.taken) if (Math.hypot(px - x, py - y) < pr + r + 0.4) return false;
    for (const o of this.others) if (x > o[0] - r && x < o[1] + r && y > o[2] - r && y < o[3] + r) return false;
    return true;
  }
  inside(x, y, r) { const [x0, x1, y0, y1] = this.rect; return x - r > x0 && x + r < x1 && y - r > y0 && y + r < y1; }
  take(x, y, r) { this.taken.push([x, y, r]); }
  // random free spots of radius r (inside the slab unless outside=true allows the ring around it)
  spots(n, r, rand, { tries = 400, pad = 1.2, outside = 0 } = {}) {
    const [x0, x1, y0, y1] = this.rect, out = [];
    for (let k = 0; k < tries && out.length < n; k++) {
      const x = x0 - outside + rand() * (x1 - x0 + 2 * outside), y = y0 - outside + rand() * (y1 - y0 + 2 * outside);
      if (!outside && !this.inside(x, y, r)) continue;
      if (!this.free(x, y, r, pad)) continue;
      this.take(x, y, r);
      out.push([x, y]);
    }
    return out;
  }
}
function resampleLine(points, step) {
  const out = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i];
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / step));
    for (let k = 1; k <= n; k++) out.push([x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n]);
  }
  return out;
}

// instanced boxes from a list of {x, y, h(eight base), sx, sy, sz, rot, color?}
function boxes(list, material, { shadow = true } = {}) {
  if (!list.length) return null;
  const im = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), material, list.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), c = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
  list.forEach((b, k) => {
    q.setFromAxisAngle(up, b.rot || 0);
    s.set(b.sx, b.sz, b.sy);
    m.compose(V(b.x, b.y, (b.h || 0) + b.sz / 2), q, s);
    im.setMatrixAt(k, m);
    if (b.color) im.setColorAt(k, c.set(b.color));
  });
  im.castShadow = shadow; im.receiveShadow = true;
  return im;
}
// boxes along the slab perimeter (kerbs, fence rails, hedges)
function perimeter(rect, inset, fn) {
  const [x0, x1, y0, y1] = rect;
  const a = [[x0 + inset, y0 + inset], [x1 - inset, y0 + inset], [x1 - inset, y1 - inset], [x0 + inset, y1 - inset]];
  for (let k = 0; k < 4; k++) fn(a[k], a[(k + 1) % 4]);
}

function slab(rect, kind, y = 0.006) {
  const [x0, x1, y0, y1] = rect, w = x1 - x0, d = y1 - y0;
  const { tex, size } = TEX[kind]();
  tex.repeat.set(w / size, d / size);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
  m.rotation.x = -Math.PI / 2;
  m.position.copy(V((x0 + x1) / 2, (y0 + y1) / 2, y));
  m.receiveShadow = true;
  m.userData.slab = true;
  return m;
}

// ------------------------------------------------------------------ the four places
function parkingLot(G, P, rand, scale) {
  // painted bays in rows wherever they fit
  const lines = [], depth = 5.0, width = 2.6;
  const [x0, x1, y0, y1] = P.rect;
  for (let y = y0 + 2; y + depth < y1 - 1; y += depth * 2 + 7) {
    for (const [yy, dir] of [[y, 1], [y + depth * 2, -1]]) {
      for (let x = x0 + 2; x + width < x1 - 1.5; x += width) {
        const cx = x + width / 2, cy = yy + dir * depth / 2;
        if (!P.free(cx, cy, 2.8, 0.6) || P.clearance(x, yy) < 1.4 || P.clearance(x + width, yy) < 1.4) continue;
        lines.push({ x, y: cy, sx: 0.12, sy: depth, sz: 0.012, h: 0.006 }, { x: x + width, y: cy, sx: 0.12, sy: depth, sz: 0.012, h: 0.006 });
        lines.push({ x: cx, y: yy + dir * depth, sx: width, sy: 0.12, sz: 0.012, h: 0.006 });
        if (lines.length > 420) break;
      }
    }
  }
  const paint = boxes(lines, new THREE.MeshStandardMaterial({ color: '#ecebe4', roughness: 0.8 }), { shadow: false });
  if (paint) { paint.receiveShadow = true; G.add(paint); }
  // a few parked cars, muted and without trailers so they never read as competitors
  const parked = [];
  for (let k = 0; k < lines.length && parked.length < 8; k += 9) {
    const b = lines[k];
    if (b.sy !== depth || rand() < 0.45) continue;
    const cx = b.x + width / 2, cy = b.y;
    if (!P.free(cx, cy, 2.2, 0.8)) continue;
    const col = ['#6d7378', '#8b8f93', '#4f5a66', '#9a8f7e', '#5d6b5c'][Math.floor(rand() * 5)];
    parked.push({ x: cx, y: cy, sx: 1.75, sy: 4.2, sz: 0.75, h: 0.18, color: col }, { x: cx, y: cy - 0.2, sx: 1.55, sy: 2.1, sz: 0.55, h: 0.93, color: '#2f3b45' });
  }
  const pc = boxes(parked, mat('#ffffff', { roughness: 0.6, flatShading: false }));
  if (pc) G.add(pc);
  // kerbs around the lot
  const kerbs = [];
  perimeter(P.rect, 0.2, ([ax, ay], [bx, by]) => kerbs.push({ x: (ax + bx) / 2, y: (ay + by) / 2, sx: Math.abs(bx - ax) + 0.3, sy: Math.abs(by - ay) + 0.3, sz: 0.16 }));
  G.add(boxes(kerbs, mat('#c9c6bd'), { shadow: false }));
  // lamp posts along the edge
  const poles = [], heads = [];
  const H = 6.5;
  perimeter(P.rect, 1.0, ([ax, ay], [bx, by]) => {
    const L = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.round(L / 22));
    for (let k = 0; k <= n; k++) {
      const x = ax + (bx - ax) * k / n, y = ay + (by - ay) * k / n;
      if (!P.free(x, y, 0.4, 1.0)) continue;
      poles.push({ x, y, sx: 0.16, sy: 0.16, sz: H });
      heads.push({ x, y, sx: 0.9, sy: 0.4, sz: 0.18, h: H - 0.1 });
    }
  });
  G.add(boxes(poles, mat('#5b6066', { roughness: 0.5 })));
  if (heads.length) G.add(boxes(heads, new THREE.MeshStandardMaterial({ color: '#fff4d2', emissive: '#ffe9b0', emissiveIntensity: 0.6 })));
}

function barnAt(G, x, y, rot, w, d, h, wallCol, roofCol) {
  const grp = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(wallCol));
  body.position.y = h / 2;
  const roof = new THREE.Mesh(new THREE.CylinderGeometry(0.01, w * 0.62, h * 0.55, 4, 1), mat(roofCol));
  roof.rotation.y = Math.PI / 4;
  roof.scale.set(1, 1, d / w);
  roof.position.y = h + h * 0.27;
  const door = new THREE.Mesh(new THREE.BoxGeometry(w * 0.36, h * 0.7, 0.05), mat('#f1ebe0'));
  door.position.set(0, h * 0.35, d / 2 + 0.03);
  grp.add(body, roof, door);
  grp.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  grp.position.copy(V(x, y));
  grp.rotation.y = rot;
  G.add(grp);
}

function farmyard(G, P, rand) {
  // barn: the roomiest spot in the yard or just outside it
  let best = null;
  for (let k = 0; k < 600; k++) {
    const [x0, x1, y0, y1] = P.rect;
    const x = x0 - 6 + rand() * (x1 - x0 + 12), y = y0 - 6 + rand() * (y1 - y0 + 12);
    if (!P.free(x, y, 6.5, 1.5)) continue;
    const c = P.clearance(x, y);
    if (!best || c > best[2]) best = [x, y, c];
  }
  if (best) { P.take(best[0], best[1], 6.5); barnAt(G, best[0], best[1], Math.round(rand() * 3) * Math.PI / 2, 9, 7, 4.4, '#9e3426', '#4a4f55'); }
  // round hay bales in a few clusters
  const bales = [];
  for (const [cx, cy] of P.spots(4, 2.6, rand, { pad: 1.5 })) {
    for (let k = 0; k < 4 + Math.floor(rand() * 3); k++) {
      const x = cx + (rand() - 0.5) * 3.2, y = cy + (rand() - 0.5) * 3.2;
      if (P.clearance(x, y) > 1.6) bales.push([x, y, rand() * Math.PI]);
    }
  }
  if (bales.length) {
    const im = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.75, 0.75, 1.2, 10), mat('#d9b85e'), bales.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    bales.forEach(([x, y, a], k) => { e.set(Math.PI / 2, 0, a, 'YXZ'); q.setFromEuler(e); m.compose(V(x, y, 0.75), q, new THREE.Vector3(1, 1, 1)); im.setMatrixAt(k, m); });
    im.castShadow = true; im.receiveShadow = true;
    G.add(im);
  }
  // fence: posts + two rails around the yard
  const posts = [], rails = [];
  perimeter(P.rect, 0.6, ([ax, ay], [bx, by]) => {
    const L = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.round(L / 2.6));
    for (let k = 0; k <= n; k++) posts.push({ x: ax + (bx - ax) * k / n, y: ay + (by - ay) * k / n, sx: 0.14, sy: 0.14, sz: 1.25 });
    for (const h of [0.55, 1.0]) rails.push({ x: (ax + bx) / 2, y: (ay + by) / 2, sx: Math.abs(bx - ax) + 0.1, sy: Math.abs(by - ay) + 0.1, sz: 0.09, h });
  });
  G.add(boxes(posts, mat('#7a5a3a')));
  G.add(boxes(rails, mat('#8d6a45'), { shadow: false }));
}

function garden(G, P, rand) {
  // hedges in segments around the garden
  const hedges = [];
  perimeter(P.rect, 0.7, ([ax, ay], [bx, by]) => {
    const L = Math.hypot(bx - ax, by - ay);
    let s = 0;
    while (s < L - 1) {
      const len = Math.min(L - s, 3 + rand() * 4);
      const f0 = s / L, f1 = (s + len) / L;
      const x = ax + (bx - ax) * (f0 + f1) / 2, y = ay + (by - ay) * (f0 + f1) / 2;
      hedges.push({ x, y, sx: Math.max(0.8, Math.abs(bx - ax) * (f1 - f0)), sy: Math.max(0.8, Math.abs(by - ay) * (f1 - f0)), sz: 1.0 + rand() * 0.3, color: rand() < 0.5 ? '#3f6a2e' : '#365f29' });
      s += len + 1.2 + rand() * 1.5;
    }
  });
  G.add(boxes(hedges, mat('#ffffff')));
  // flower beds: soil boxes with flowers on top
  const beds = [], flowers = [];
  const cols = ['#e34b5f', '#f2c84b', '#f08bc0', '#9d6ce0', '#f6f2e8', '#ff8c42'];
  for (const [x, y] of P.spots(7, 1.5, rand, { pad: 0.9 })) {
    const rot = rand() * Math.PI, w = 2.2 + rand() * 1.2, d = 0.9 + rand() * 0.4;
    beds.push({ x, y, sx: w, sy: d, sz: 0.22, rot });
    for (let k = 0; k < 14; k++) {
      const u = (rand() - 0.5) * w * 0.85, v = (rand() - 0.5) * d * 0.7;
      flowers.push([x + u * Math.cos(rot) + v * Math.sin(rot), y - u * Math.sin(rot) + v * Math.cos(rot), cols[Math.floor(rand() * cols.length)]]);
    }
  }
  G.add(boxes(beds, mat('#5a3d27')));
  if (flowers.length) {
    const im = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.13, 0), mat('#ffffff', { roughness: 0.7 }), flowers.length);
    const m = new THREE.Matrix4(), c = new THREE.Color();
    flowers.forEach(([x, y, col], k) => { m.makeTranslation(x, 0.32, -y); im.setMatrixAt(k, m); im.setColorAt(k, c.set(col)); });
    G.add(im);
  }
  // round shrubs and a small shed
  const shrubs = P.spots(6, 0.7, rand, { pad: 0.8 });
  if (shrubs.length) {
    const im = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.7, 0), mat('#4c7a34'), shrubs.length);
    const m = new THREE.Matrix4();
    shrubs.forEach(([x, y], k) => { const s = 0.7 + rand() * 0.6; m.makeScale(s, s * 0.85, s).setPosition(x, 0.5 * s, -y); im.setMatrixAt(k, m); });
    im.castShadow = true;
    G.add(im);
  }
  const shed = P.spots(1, 2.4, rand, { pad: 1.0, outside: 3 });
  if (shed.length) barnAt(G, shed[0][0], shed[0][1], rand() * Math.PI, 3.0, 2.4, 2.1, '#9b7b55', '#5b4636');
}

function excavatorAt(G, x, y, rot) {
  const grp = new THREE.Group();
  const yellow = mat('#e8b423', { roughness: 0.6 }), dark = mat('#2c2f31'), glass = mat('#3e5566', { roughness: 0.2 });
  const add = (geo, m, px, py, pz, rx = 0, rz = 0) => { const o = new THREE.Mesh(geo, m); o.position.set(px, py, pz); o.rotation.set(rx, 0, rz); o.castShadow = true; o.receiveShadow = true; grp.add(o); return o; };
  add(new THREE.BoxGeometry(4.6, 0.9, 0.9), dark, 0, 0.45, -1.35);
  add(new THREE.BoxGeometry(4.6, 0.9, 0.9), dark, 0, 0.45, 1.35);
  add(new THREE.BoxGeometry(3.6, 1.3, 2.8), yellow, -0.3, 1.6, 0);
  add(new THREE.BoxGeometry(1.4, 1.5, 1.2), glass, 0.8, 3.0, -0.75);
  add(new THREE.BoxGeometry(5.2, 0.55, 0.6), yellow, 3.3, 3.4, 0.6, 0, 0.55);
  add(new THREE.BoxGeometry(3.6, 0.45, 0.5), yellow, 6.0, 3.2, 0.6, 0, -0.95);
  add(new THREE.BoxGeometry(1.2, 0.9, 1.3), dark, 6.9, 1.4, 0.6, 0, -0.3);
  grp.position.copy(V(x, y));
  grp.rotation.y = rot;
  G.add(grp);
}

function quarry(G, P, rand) {
  // sand and gravel piles
  const piles = [];
  for (const [x, y] of P.spots(7, 7, rand, { pad: 2.5, tries: 600 })) piles.push([x, y, 5 + rand() * 4, 3 + rand() * 3, rand() < 0.7 ? '#d6b679' : '#9a958c']);
  for (const [x, y] of P.spots(5, 4, rand, { pad: 2, tries: 400 })) piles.push([x, y, 3 + rand() * 2, 2 + rand() * 1.5, rand() < 0.6 ? '#cdae74' : '#8f8a82']);
  if (piles.length) {
    const im = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 8, 1), mat('#ffffff'), piles.length);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
    piles.forEach(([x, y, r, h, col], k) => { q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rand() * 6); m.compose(V(x, y, h / 2), q, new THREE.Vector3(r, h, r)); im.setMatrixAt(k, m); im.setColorAt(k, c.set(col)); });
    im.castShadow = true; im.receiveShadow = true;
    G.add(im);
  }
  const ex = P.spots(1, 6, rand, { pad: 2, tries: 600, outside: 6 });
  if (ex.length) excavatorAt(G, ex[0][0], ex[0][1], rand() * Math.PI * 2);
  // a conveyor ramping up to a pile
  const cv = P.spots(1, 9, rand, { pad: 2, tries: 600, outside: 8 });
  if (cv.length) {
    const [x, y] = cv[0], a = rand() * Math.PI * 2, L = 16, rise = 6;
    const grp = new THREE.Group();
    const belt = new THREE.Mesh(new THREE.BoxGeometry(L, 0.35, 1.1), mat('#3a3d40'));
    belt.rotation.z = Math.atan2(rise, L);
    belt.position.set(0, rise / 2 + 0.8, 0);
    grp.add(belt);
    for (const f of [-0.35, 0.05, 0.42]) {
      const h = rise / 2 + 0.8 + f * rise;
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.25, h, 0.9), mat('#c9a227'));
      leg.position.set(f * L, h / 2, 0);
      grp.add(leg);
    }
    const pile = new THREE.Mesh(new THREE.ConeGeometry(4.5, 4.2, 8), mat('#d2b27a'));
    pile.position.set(L / 2 + 1.5, 2.1, 0);
    grp.add(pile);
    grp.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    grp.position.copy(V(x, y));
    grp.rotation.y = a;
    G.add(grp);
  }
  // boulders
  const rocks = P.spots(10, 0.9, rand, { pad: 1.0 });
  if (rocks.length) {
    const im = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(0.9, 0), mat('#8c877e'), rocks.length);
    const m = new THREE.Matrix4();
    rocks.forEach(([x, y], k) => { const s = 0.6 + rand() * 0.9; m.makeScale(s, s * 0.7, s).setPosition(x, 0.3 * s, -y); im.setMatrixAt(k, m); });
    im.castShadow = true;
    G.add(im);
  }
}

const BUILD = { car: parkingLot, tractor: farmyard, mower: garden, truck: quarry };

/**
 * tracks: laid-out tracks ({ tr, bb (rotated frame), ox, oy, toWorld }); gap = the layout gap between courses.
 * Returns { group, rects } with rects = [x0, x1, y0, y1] world rectangles that trees should keep out of.
 */
export function buildScenery(tracks, gap = 24) {
  const group = new THREE.Group();
  const M = Math.max(4, Math.min(10, gap / 2 - 2.5));
  const rects = tracks.map((t) => [t.bb.minx + t.ox - M, t.bb.maxx + t.ox + M, t.bb.miny + t.oy - M, t.bb.maxy + t.oy + M]);
  tracks.forEach((t, i) => {
    const veh = t.tr.vehicle, kind = THEME[veh] || 'lot';
    group.add(slab(rects[i], kind));
    const P = new Placer(t, rects[i], rects.filter((_, j) => j !== i));
    (BUILD[veh] || parkingLot)(group, P, rng(100 + i * 17));
  });
  return { group, rects };
}

// painted ground markings for geometry.zones ({name, kind, polygon}); returns meshes in track-local coordinates
const ZONE_COLORS = { phase1: '#7cc4ff', phase2: '#f6cf3a', goal: '#f6cf3a', target: '#f6cf3a', bay: '#f6cf3a', turn: '#7cc4ff', turnaround: '#7cc4ff', manoeuvre: '#7cc4ff', maneuver: '#7cc4ff',
  no_go: '#ff6b5e', forbidden: '#ff6b5e', keep_out: '#ff6b5e', start: '#f4f1ea' };
export const zoneColor = (kind) => ZONE_COLORS[String(kind || '').toLowerCase()] || '#f4f1ea';
export function zoneMeshes(zones) {
  const out = [];
  for (const z of zones || []) {
    const poly = z.polygon || [];
    if (poly.length < 3) continue;
    const col = zoneColor(z.kind);
    const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
    fill.rotation.x = -Math.PI / 2;
    fill.position.y = 0.03;
    fill.renderOrder = 1;
    const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(poly.map(([x, y]) => V(x, y, 0.04))), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9 }));
    const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length, cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
    out.push({ fill, line, center: [cx, cy], name: z.name || z.kind || 'zone', color: col });
  }
  return out;
}
