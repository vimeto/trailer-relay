// Trailer Relay viewer: replays recorded runs (no physics in the browser).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { buildCombo, poseCombo, setOpacity, setLook } from './models.js';
import { headAt, lookWords, viewGeometry } from './driver.js';
import {
  fetchJSON, prepareRun, applyClock, sampleLeg, runAt, elapsedOf, legStatusAt, callAt, progressOf, fmtClock,
  gazeAt, variantOf, assignColors, phaseInfo, phaseText, thinkingAt,
} from './data.js';
import { hitchDeg, critOf, coursePoints, railLines, hasWallBoundary, corridorDiscs, pointInPoly } from './geom2d.js';
import { buildScenery, surfaceTexture, SURFACES, zoneMeshes } from './scenery.js';
import { createGrid, createInspect } from './panels.js';
import { createChips } from './chips.js';
import {
  describe, uniqueLabels, modelColor, modelShort, modelName, learningTag, runLabel as runLabelOf, OBS_NAMES, LEVEL_SHORT, MODE_NAMES,
  CONTROL_NAMES, CLOCK_NAMES, CONDITION_NAMES, VEHICLE_NAMES, TEMPLATE_NAMES, courseSet, relayName, obsName,
} from './labels.js';
import { createResults } from './results.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// ?dir= points at another runs directory (?runs= is the run selection; a value ending in '/' is still read as a directory)
const RUNS_DIR = params.get('dir') || (params.get('runs')?.endsWith('/') ? params.get('runs') : '../runs/');
const POLL_MS = Number(params.get('poll')) || 3000;
// A static export (python -m relay site) marks its page with <meta name="relay-static">: the data never changes, so
// there is no polling and no directory listing to probe; every file the viewer asks for is in the export.
const STATIC = !!document.querySelector('meta[name="relay-static"]') || params.get('static') === '1';
// ?leg=<n> (1-based) in Broadcast/Grid/Inspect: open in heats on that leg; Results' notes take ?leg=<vehicle>
const LEG_PARAM = /^\d+$/.test(params.get('leg') || '') ? Math.max(1, Number(params.get('leg'))) : null;
// ?clean=1: the 3D scene only, for stills (no panels, no labels; the warning rings on the rigs stay);
// ?clean=chips keeps the in-scene callouts and name flags too
const CLEAN = params.has('clean');
if (CLEAN) document.body.classList.add('clean', ...(params.get('clean') === 'chips' ? ['clean-chips'] : []));
const SPEEDS = [1, 2, 5, 10, 30];
const RAIL_H = { mower: 0.5, car: 0.8, tractor: 0.95, truck: 1.15 };
const VIEWS = ['results', 'broadcast', 'grid', 'inspect'];
const SCENE_VIEWS = ['broadcast', 'grid', 'inspect'];
const LAYOUT_GAP = 24;
const STATUS_TEXT = {
  READY: 'Ready', REVERSING: 'Reversing', PULLING_FORWARD: 'Pulling fwd', STOPPED: 'Stopped', COMMITTED: 'Committed',
  JACKKNIFED: 'Jackknifed', PINNED: 'Pinned', FINISHED: 'Parked', TIME_UP: 'Time up', DNF: 'DNF', WAITING: 'Waiting',
  LOOKING: 'Looking', THINKING: 'Thinking',
};

// ------------------------------------------------------------------ renderer & scene
const canvas = $('scene');
// logarithmic depth: the ground is a stack of flat layers a few mm apart (grass 0, theme slab 6 mm, course 25 mm,
// zone fills / dashes 30-35 mm); a linear depth buffer (near 0.3, far 4000) loses mm precision past ~150 m -> z-fighting flicker
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: params.has('shot') });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const HORIZON = new THREE.Color('#e9dcc2');
scene.fog = new THREE.Fog(HORIZON, 160, 900);

const camera = new THREE.PerspectiveCamera(42, 1, 0.3, 4000);
camera.position.set(0, 80, 120);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 4;
controls.maxDistance = 900;

// sky dome
{
  const geo = new THREE.SphereGeometry(2000, 32, 16);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color('#5d8fc9') }, mid: { value: new THREE.Color('#a9c8e4') }, bottom: { value: HORIZON } },
    vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader: `uniform vec3 top; uniform vec3 mid; uniform vec3 bottom; varying vec3 vP;
      void main(){ float h = vP.y; vec3 c = h > 0.0 ? mix(mix(bottom, mid, smoothstep(0.0, 0.18, h)), top, smoothstep(0.18, 0.75, h)) : bottom;
      gl_FragColor = vec4(c, 1.0); }`,
  });
  scene.add(new THREE.Mesh(geo, mat));
}

const hemi = new THREE.HemisphereLight('#dbe8ff', '#6b5b3e', 1.15);
scene.add(hemi);
const sun = new THREE.DirectionalLight('#fff0d6', 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.03;
scene.add(sun, sun.target);
const SUN_DIR = new THREE.Vector3(-0.55, 1.0, 0.42).normalize();

function setShadowFrame(target, radius) {
  const r = Math.max(14, radius);
  const cam = sun.shadow.camera;
  cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r; cam.near = 1; cam.far = r * 4 + 120;
  cam.updateProjectionMatrix();
  sun.target.position.copy(target);
  sun.position.copy(target).addScaledVector(SUN_DIR, r * 2 + 60);
}

// ------------------------------------------------------------------ textures
function canvasTex(w, h, draw, repeat = true) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

const grassTex = canvasTex(512, 512, (g, w, h) => {
  const r = rng(7);
  g.fillStyle = '#7f9a58'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 9000; i++) {
    const v = r();
    g.fillStyle = v < 0.33 ? 'rgba(98,128,64,0.55)' : v < 0.66 ? 'rgba(150,170,98,0.45)' : 'rgba(118,140,74,0.6)';
    const x = r() * w, y = r() * h, s = 1 + r() * 3;
    g.fillRect(x, y, s, s * (1 + r() * 2));
  }
  for (let i = 0; i < 40; i++) {
    g.fillStyle = `rgba(${r() < 0.5 ? '170,160,100' : '90,118,60'},0.10)`;
    g.beginPath(); g.arc(r() * w, r() * h, 20 + r() * 50, 0, Math.PI * 2); g.fill();
  }
});

const stripeTex = canvasTex(128, 16, (g, w, h) => {
  g.fillStyle = '#f3f1ec'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#d6352b'; g.fillRect(0, 0, w / 2, h);
  g.fillStyle = 'rgba(0,0,0,0.12)'; g.fillRect(0, h - 3, w, 3);
});

const bayTex = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = 'rgba(244, 196, 48, 0.20)'; g.fillRect(0, 0, w, h);
  g.strokeStyle = 'rgba(250, 205, 50, 0.95)'; g.lineWidth = 14;
  for (let k = -h; k < w + h; k += 44) { g.beginPath(); g.moveTo(k, 0); g.lineTo(k + h, h); g.stroke(); }
}, true);

// ------------------------------------------------------------------ world helpers
const toV = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

function ribbon(left, right, h, uScale) {
  // triangle strip between two equally long polylines (sim coords)
  const n = Math.min(left.length, right.length);
  const pos = new Float32Array(n * 2 * 3), uv = new Float32Array(n * 2 * 2), idx = [];
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (i) {
      const mx = (left[i][0] + right[i][0] - left[i - 1][0] - right[i - 1][0]) / 2;
      const my = (left[i][1] + right[i][1] - left[i - 1][1] - right[i - 1][1]) / 2;
      s += Math.hypot(mx, my);
    }
    pos.set([left[i][0], h, -left[i][1], right[i][0], h, -right[i][1]], i * 6);
    uv.set([s * uScale, 0, s * uScale, 1], i * 4);
    if (i) { const a = (i - 1) * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function wall(points, h0, h1, stripeLen) {
  // vertical strip along a polyline
  const n = points.length;
  const pos = new Float32Array(n * 6), uv = new Float32Array(n * 4), idx = [];
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (i) s += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    pos.set([points[i][0], h0, -points[i][1], points[i][0], h1, -points[i][1]], i * 6);
    uv.set([s / stripeLen, 0, s / stripeLen, 1], i * 4);
    if (i) { const a = (i - 1) * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function resample(points, step) {
  const out = [points[0]];
  let acc = 0;
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i];
    const d = Math.hypot(x1 - x0, y1 - y0);
    let pos = step - acc;
    while (pos <= d) { const f = pos / d; out.push([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]); pos += step; }
    acc = d - (pos - step);
  }
  out.push(points[points.length - 1]);
  return out;
}

function textSprite(lines, { color = '#f4f1ea', bg = 'rgba(16,20,18,0.78)', accent = null, worldH = 2, screen = false } = {}) {
  const pad = 26, fs = [56, 34];
  const c = document.createElement('canvas');
  const g = c.getContext('2d');
  const fonts = [`700 ${fs[0]}px Inter, -apple-system, Helvetica, Arial, sans-serif`, `600 ${fs[1]}px Inter, -apple-system, Helvetica, Arial, sans-serif`];
  let w = 0;
  lines.forEach((t, i) => { g.font = fonts[Math.min(i, 1)]; w = Math.max(w, g.measureText(t).width); });
  c.width = Math.ceil(w + pad * 2 + (accent ? 14 : 0));
  c.height = Math.ceil(pad * 1.6 + fs[0] + (lines.length > 1 ? fs[1] + 10 : 0));
  g.fillStyle = bg;
  const r = 22;
  g.beginPath(); g.roundRect(0, 0, c.width, c.height, r); g.fill();
  if (accent) { g.fillStyle = accent; g.beginPath(); g.roundRect(0, 0, 14, c.height, [r, 0, 0, r]); g.fill(); }
  g.textBaseline = 'top';
  let y = pad * 0.8;
  lines.forEach((t, i) => {
    g.font = fonts[Math.min(i, 1)];
    g.fillStyle = i ? 'rgba(244,241,234,0.7)' : color;
    g.fillText(t, pad + (accent ? 14 : 0), y);
    y += fs[Math.min(i, 1)] + 10;
  });
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, depthTest: !screen, fog: false, sizeAttenuation: !screen }));
  sp.center.set(0.5, 0);
  sp.scale.set(worldH * c.width / c.height, worldH, 1);
  sp.renderOrder = 10;
  return sp;
}

// ------------------------------------------------------------------ app state
const state = {
  runs: [], groups: {}, relayId: null, tracks: [], T: 0, Tmax: 1, playing: !params.has('paused'),
  speed: Number(params.get('speed')) || 10, cam: params.get('cam') || 'follow', focus: 0, dirtyTicks: true,
  layoutCenter: new THREE.Vector3(), layoutRadius: 50,
  clockMode: 'relay', clockPinned: params.has('clock'), heatLeg: (LEG_PARAM ?? 1) - 1,
  pool: new Map(), // run_id -> { item (index entry), rec, file }
  sel: null, // Set of run ids, or null = everything in the group
  filters: { model: new Set(), level: new Set(), study: new Set(), prompt: new Set(), obs: new Set(), mode: new Set(), control: new Set() },
  q: '', // picker search
  pickerOpen: params.has('picker'), compareOpen: params.has('compare'), anyLive: false, polls: 0,
  view: VIEWS.includes(params.get('view')) ? params.get('view') : 'broadcast',
  lazy: false, // true while only the Results view has been opened: the 3D scene and run records are not loaded yet
};
const world = new THREE.Group();
scene.add(world);

// ground
{
  grassTex.repeat.set(220, 220);
  const g = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ map: grassTex, roughness: 1 }));
  g.rotation.x = -Math.PI / 2;
  g.receiveShadow = true;
  scene.add(g);
}

// turnaround kerb polygon -> runs of segments: label 4 = kerb (low stone); 5-7 = side-road / end walls (barriers)
const KERB_LABEL = 4;
function splitByLabel(W, labels) {
  const out = [];
  let cur = null, kind = null;
  for (let k = 0; k < labels.length; k++) {
    const kb = labels[k] === KERB_LABEL;
    if (kb !== kind || !cur) { cur = [W[k]]; cur.kerb = kb; out.push(cur); kind = kb; }
    cur.push(W[k + 1]);
  }
  return out;
}

function buildTrack(tr, i, nLegs) {
  const geo = tr.geometry;
  const grp = new THREE.Group();
  const all = coursePoints(geo);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const bb = { minx: Math.min(...xs), maxx: Math.max(...xs), miny: Math.min(...ys), maxy: Math.max(...ys) };
  const theme = SURFACES[tr.vehicle] || SURFACES.car;
  // turnaround courses: one closed kerb polygon is the whole boundary (its left/right rails are just its halves)
  const walled = hasWallBoundary(geo);
  const hasRails = !walled && geo.left_rail?.length > 1 && geo.right_rail?.length > 1;

  if (walled) {
    const sTex = surfaceTexture(tr.vehicle).clone(); sTex.needsUpdate = true;
    sTex.wrapS = sTex.wrapT = THREE.RepeatWrapping; sTex.repeat.set(1 / 4.5, 1 / 4.5);
    for (const w of geo.walls) {
      if (!(w.length > 3)) continue;
      const shape = new THREE.Shape(w.map(([x, y]) => new THREE.Vector2(x, y)));
      const surf = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ map: sTex, roughness: 0.95 }));
      surf.rotation.x = -Math.PI / 2; surf.position.y = 0.025; surf.receiveShadow = true;
      grp.add(surf);
    }
  } else if (hasRails) {
    // apron strip just outside the rails, then the driving surface (both themed per place)
    const apronL = [], apronR = [];
    const n = Math.min(geo.left_rail.length, geo.right_rail.length);
    for (let k = 0; k < n; k++) {
      const [lx, ly] = geo.left_rail[k], [rx, ry] = geo.right_rail[k];
      const dx = lx - rx, dy = ly - ry, d = Math.hypot(dx, dy) || 1, e = 0.9;
      apronL.push([lx + dx / d * e, ly + dy / d * e]); apronR.push([rx - dx / d * e, ry - dy / d * e]);
    }
    const apron = new THREE.Mesh(ribbon(apronL, apronR, 0.012, 0.25), new THREE.MeshStandardMaterial({ color: theme.apron, roughness: 1 }));
    apron.receiveShadow = true;
    grp.add(apron);
    const sTex = surfaceTexture(tr.vehicle).clone(); sTex.needsUpdate = true;
    // the surface follows the real rails (asymmetric courses: wider on one side where there is a yard or lay-by)
    const surf = new THREE.Mesh(ribbon(geo.left_rail, geo.right_rail, 0.025, 0.22), new THREE.MeshStandardMaterial({ map: sTex, roughness: 0.95 }));
    surf.receiveShadow = true;
    surf.userData.surface = true;
    grp.add(surf);
  }

  // centre guide dashes (turnaround: the two phase guide routes, fainter)
  const guides = walled && geo.routes?.length ? geo.routes : geo.centerline?.length > 2 ? [geo.centerline] : [];
  for (const guide of guides) {
    if (!(guide?.length > 2)) continue;
    const pts = resample(guide, 0.5);
    const dash = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.9, 0.09), new THREE.MeshStandardMaterial({ color: '#f3efe2', roughness: 0.8, transparent: true, opacity: walled ? 0.3 : 0.55 }), Math.ceil(pts.length / 4));
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    let c = 0;
    for (let k = 1; k < pts.length - 1; k += 4) {
      const [x, y] = pts[k], [x2, y2] = pts[k + 1];
      e.set(-Math.PI / 2, 0, Math.atan2(y2 - y, x2 - x), 'XYZ');
      q.setFromEuler(e);
      m.compose(toV(x, y, 0.035), q, new THREE.Vector3(1, 1, 1));
      dash.setMatrixAt(c++, m);
    }
    dash.count = c;
    dash.receiveShadow = true;
    grp.add(dash);
  }

  // painted zones (multi-phase courses): a tinted marking, an outline and a small label
  const zoneSigns = [];
  const dcq = geo.dock_corners || [];
  const isBay = (z) => z.polygon?.length === dcq.length && z.polygon.every((p, k) => Math.hypot(p[0] - dcq[k][0], p[1] - dcq[k][1]) < 1e-3);
  for (const z of zoneMeshes((geo.zones || []).filter((z) => !isBay(z)))) {
    grp.add(z.fill, z.line);
    const lab = textSprite([z.name], { worldH: 0.018, color: z.color, screen: true });
    lab.position.copy(toV(z.center[0], z.center[1], 0.6));
    grp.add(lab);
    zoneSigns.push(lab);
  }

  // handoff bay, plus the green outline shown when the focused rig is parked in it
  const dc = geo.dock_corners || [];
  let parked = null, bay = null;
  if (dc.length === 4) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(dc.flatMap(([x, y]) => [x, 0.035, -y])), 3));
    const L = Math.hypot(dc[1][0] - dc[2][0], dc[1][1] - dc[2][1]), W = Math.hypot(dc[0][0] - dc[1][0], dc[0][1] - dc[1][1]);
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, W / 2.2, 0, W / 2.2, L / 2.2, 0, L / 2.2]), 2));
    g.setIndex([0, 2, 1, 0, 3, 2]);
    g.computeVertexNormals();
    bay = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: bayTex, transparent: true, roughness: 0.8, depthWrite: false, side: THREE.DoubleSide }));
    bay.receiveShadow = true;
    grp.add(bay);
    const frame = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(dc.map(([x, y]) => toV(x, y, 0.045))), new THREE.LineBasicMaterial({ color: '#f6cf3a' }));
    grp.add(frame);
    const cx = dc.reduce((a, p) => a + p[0], 0) / 4, cy = dc.reduce((a, p) => a + p[1], 0) / 4;
    const pos = [];
    for (const [x, y] of [...dc, dc[0]]) { const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1, e = 0.25; pos.push(x + dx / d * e, 0.07, -(y + dy / d * e)); }
    const lg = new LineGeometry();
    lg.setPositions(pos);
    const pm = new LineMaterial({ color: new THREE.Color('#4fd18b'), linewidth: 5, transparent: true, opacity: 0.95, worldUnits: false });
    parked = new Line2(lg, pm);
    parked.renderOrder = 4;
    parked.visible = false;
    parked.userData.mat = pm;
    grp.add(parked);
  }

  // barriers: posts + striped rail band, along every rail (route rails, start/dock walls, extra walls)
  const H = RAIL_H[tr.vehicle] || 0.8;
  let lines = railLines(geo);
  if (walled && geo.walls.length === 1 && geo.wall_labels?.length === geo.walls[0].length - 1) lines = splitByLabel(geo.walls[0], geo.wall_labels);
  else if (!hasRails && !walled && dc.length === 4) lines.push([dc[0], dc[1]]);
  const stripeMat = new THREE.MeshStandardMaterial({ map: stripeTex, roughness: 0.55, side: THREE.DoubleSide });
  const lowMat = new THREE.MeshStandardMaterial({ color: '#d9d5cc', side: THREE.DoubleSide, roughness: 0.6 });
  const postGeo = new THREE.BoxGeometry(0.09, H, 0.09);
  const postMat = new THREE.MeshStandardMaterial({ color: '#e9e6df', roughness: 0.6 });
  const posts = [];
  const kerbMat = new THREE.MeshStandardMaterial({ color: '#cfcac0', roughness: 0.85, side: THREE.DoubleSide });
  for (const line of lines) {
    if (line.kerb) {
      // a low kerb stone, no posts
      const k = new THREE.Mesh(wall(line, 0, Math.max(0.16, H * 0.28), 1e6), kerbMat);
      k.castShadow = true; k.receiveShadow = true;
      grp.add(k);
      continue;
    }
    const band = new THREE.Mesh(wall(line, H * 0.62, H, 1.0), stripeMat);
    band.castShadow = true; band.receiveShadow = true;
    grp.add(band);
    const low = new THREE.Mesh(wall(line, H * 0.2, H * 0.3, 1e6), lowMat);
    low.castShadow = true;
    grp.add(low);
    for (const p of resample(line, Math.max(1.2, H * 2.2))) posts.push(p);
  }
  const inst = new THREE.InstancedMesh(postGeo, postMat, Math.max(1, posts.length));
  const m = new THREE.Matrix4();
  posts.forEach(([x, y], k) => { m.makeTranslation(x, H / 2, -y); inst.setMatrixAt(k, m); });
  inst.count = posts.length;
  inst.castShadow = true;
  grp.add(inst);

  // signs
  const spec = state.runs[0]?.legs.find((l) => l.i === i)?.spec;
  const scale = Math.max(1.1, (spec?.combo_length || 8) * 0.17);
  const sign = textSprite([`LEG ${i + 1}`, `${spec?.name || tr.vehicle} · ${TEMPLATE_NAMES[tr.template] || tr.template}`], { worldH: 0.036, accent: '#f6cf3a', screen: true });
  const c0 = geo.centerline?.[0] || all[0] || [0, 0];
  sign.position.copy(toV(c0[0], c0[1], H + scale * 1.5));
  sign.userData.base = sign.position.clone();
  grp.add(sign);
  const bc = dc.length ? [dc.reduce((a, p) => a + p[0], 0) / dc.length, dc.reduce((a, p) => a + p[1], 0) / dc.length] : c0;
  const bayZone = (geo.zones || []).find(isBay);
  const baySign = textSprite([bayZone ? String(bayZone.name || 'exit zone').toUpperCase() : i === nLegs - 1 ? 'FINISH BAY' : 'HANDOFF BAY'], { worldH: 0.022, color: '#f6cf3a', screen: true });
  baySign.position.copy(toV(bc[0], bc[1], H + 0.5));
  grp.add(baySign);

  const fx = new THREE.Group();
  grp.add(fx);
  return { grp, bb, tr, fx, sign, baySign, H, parked, zoneSigns, bay };
}

// Rotate each course so it packs tightly (a diagonal L-turn wastes most of its bounding box).
function orientTrack(t) {
  const pts = coursePoints(t.tr.geometry);
  let best = null;
  for (let k = 0; k < 24; k++) {
    const a = (k * Math.PI) / 12, c = Math.cos(a), s = Math.sin(a);
    let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
    for (const [x, y] of pts) {
      const X = c * x - s * y, Y = s * x + c * y;
      minx = Math.min(minx, X); maxx = Math.max(maxx, X); miny = Math.min(miny, Y); maxy = Math.max(maxy, Y);
    }
    // small area, prefer landscape, and prefer staying near the authored orientation
    const w = maxx - minx, d = maxy - miny;
    const cost = w * d * (d > w ? 1.25 : 1) * (1 + 0.04 * Math.min(k, 24 - k));
    if (!best || cost < best.cost) best = { cost, a, bb: { minx, maxx, miny, maxy } };
  }
  t.rot = best.a;
  t.bb = best.bb;
  t.grp.rotation.y = best.a;
}

function layoutTracks(tracks) {
  const gap = LAYOUT_GAP;
  const n = tracks.length;
  for (const t of tracks) orientTrack(t);
  const arrange = (rows) => {
    const per = Math.ceil(n / rows);
    const rowsOut = [];
    for (let r = 0; r < rows; r++) rowsOut.push(tracks.slice(r * per, (r + 1) * per));
    const rowW = rowsOut.map((row) => row.reduce((a, t) => a + t.bb.maxx - t.bb.minx, 0) + gap * Math.max(0, row.length - 1));
    const rowH = rowsOut.map((row) => Math.max(0, ...row.map((t) => t.bb.maxy - t.bb.miny)));
    const W = Math.max(...rowW), D = rowH.reduce((a, b) => a + b, 0) + gap * (rows - 1);
    return { rowsOut, rowW, rowH, W, D };
  };
  const aspect = Math.max(1.2, (window.innerWidth - 380) / Math.max(1, window.innerHeight - 260)) * 1.25;
  const cands = [1, 2].filter((r) => r <= n).map(arrange);
  // the camera is pitched, so depth costs about 1/sin(pitch) more screen than width
  const score = (c) => Math.max(c.W / aspect, c.D * 1.3);
  const best = cands.reduce((a, b) => (score(b) < score(a) ? b : a));
  let y = best.D / 2;
  best.rowsOut.forEach((row, r) => {
    let x = -best.rowW[r] / 2;
    const cy = y - best.rowH[r] / 2;
    for (const t of row) {
      t.ox = x - t.bb.minx;
      t.oy = cy - (t.bb.miny + t.bb.maxy) / 2;
      t.grp.position.set(t.ox, 0, -t.oy);
      x += t.bb.maxx - t.bb.minx + gap;
    }
    y -= best.rowH[r] + gap;
  });
  for (const t of tracks) {
    t.grp.updateMatrixWorld(true);
    const c = Math.cos(t.rot), s = Math.sin(t.rot);
    t.toWorld = (x, y) => [c * x - s * y + t.ox, s * x + c * y + t.oy];
  }
  state.layoutCenter.set(0, 0, 0);
  state.layoutRadius = Math.max(best.W, best.D) / 2;
  state.layoutSize = { w: best.W, d: best.D };
  return { total: best.W, maxh: best.D };
}

function plantTrees(tracks, extent, rects = []) {
  // keep trees off the courses and out of the themed places around them
  const r = rng(42);
  const pts = [];
  for (const t of tracks) {
    const g = t.tr.geometry;
    if (g.centerline?.length || g.left_rail?.length) for (const [x, y, r] of corridorDiscs(g, 2)) pts.push([...t.toWorld(x, y), r + 11]);
    else for (const [x, y] of resample(coursePoints(g), 2)) pts.push([...t.toWorld(x, y), 3 + 11]);
  }
  const ok = (x, y) => pts.every(([px, py, rr]) => (px - x) ** 2 + (py - y) ** 2 > rr * rr)
    && rects.every(([x0, x1, y0, y1]) => x < x0 - 2.5 || x > x1 + 2.5 || y < y0 - 2.5 || y > y1 + 2.5);
  const trunkGeo = new THREE.CylinderGeometry(0.18, 0.28, 1.6, 6);
  const crownGeo = new THREE.IcosahedronGeometry(1.6, 0);
  const coneGeo = new THREE.ConeGeometry(1.5, 4.2, 7);
  const N = 340;
  const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: '#6b4f36', roughness: 1 }), N);
  const crowns = new THREE.InstancedMesh(crownGeo, new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.9, flatShading: true }), N);
  const cones = new THREE.InstancedMesh(coneGeo, new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.9, flatShading: true }), N);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), col = new THREE.Color();
  let a = 0, b = 0, tries = 0;
  const W = extent.total / 2 + 90, Hh = extent.maxh / 2 + 70;
  while (a + b < N && tries++ < 8000) {
    const x = (r() * 2 - 1) * W, y = (r() * 2 - 1) * Hh;
    // denser near the edges of the field, sparse in the middle
    if (Math.abs(y) < extent.maxh / 2 + 10 && Math.abs(x) < extent.total / 2 + 10 && r() < 0.93) continue;
    if (!ok(x, y)) continue;
    const k = 0.7 + r() * 0.9;
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), r() * 6.28);
    s.set(k, k, k);
    m.compose(new THREE.Vector3(x, 0.8 * k, -y), q, s);
    trunks.setMatrixAt(a + b, m);
    if (r() < 0.55) {
      m.compose(new THREE.Vector3(x, (1.6 + 1.2) * k, -y), q, s);
      crowns.setMatrixAt(a, m);
      crowns.setColorAt(a, col.setHSL(0.24 + r() * 0.06, 0.38, 0.30 + r() * 0.1));
      a++;
    } else {
      m.compose(new THREE.Vector3(x, (1.6 + 1.8) * k, -y), q, s);
      cones.setMatrixAt(b, m);
      cones.setColorAt(b, col.setHSL(0.30 + r() * 0.05, 0.35, 0.22 + r() * 0.08));
      b++;
    }
  }
  trunks.count = a + b; crowns.count = a; cones.count = b;
  for (const im of [trunks, crowns, cones]) { im.castShadow = true; im.receiveShadow = true; world.add(im); }
}

// ------------------------------------------------------------------ per-run actors
function makeTrail(leg, color) {
  const tr = leg.trace, spec = leg.spec, pos = [];
  for (let k = 0; k < tr.t.length; k++) {
    const hx = tr.x[k] - spec.hitch_offset * Math.cos(tr.th1[k]), hy = tr.y[k] - spec.hitch_offset * Math.sin(tr.th1[k]);
    const ax = hx - spec.trailer_length * Math.cos(tr.th2[k]), ay = hy - spec.trailer_length * Math.sin(tr.th2[k]);
    pos.push(ax, 0.06, -ay);
  }
  if (pos.length < 6) pos.push(pos[0], pos[1], pos[2] + 0.001);
  const g = new LineGeometry();
  g.setPositions(pos);
  const mat = new LineMaterial({ color: new THREE.Color(color), linewidth: 3, transparent: true, opacity: 0.9, worldUnits: false });
  const line = new Line2(g, mat);
  line.computeLineDistances();
  line.renderOrder = 2;
  return { line, mat, n: tr.t.length };
}

function makeMarker(color, size) {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.RingGeometry(size * 0.75, size, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.07;
  g.add(ring);
  const dot = new THREE.Mesh(new THREE.CircleGeometry(size * 0.3, 20), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false }));
  dot.rotation.x = -Math.PI / 2;
  dot.position.y = 0.075;
  g.add(dot);
  g.visible = false;
  return { g, ring };
}

// translucent ground fan from its origin between angles a0..a1 (radians, in the fan's frame), fading out with distance;
// fan.userData.set(a0, a1, R) reshapes it (driver views change every frame)
function makeFan(color, { alpha = 0.46, edge = 0.85, rings = 6, n = 40, h = 0.09 } = {}) {
  const c = new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.12);
  const col = [], idx = [];
  for (let r = 0; r <= rings; r++) {
    const a = alpha * (1 - r / rings) ** 1.2 + (r === rings ? 0.0 : 0.05 * alpha / 0.46);
    for (let k = 0; k <= n; k++) col.push(c.r, c.g, c.b, a);
  }
  for (let r = 0; r < rings; r++) for (let k = 0; k < n; k++) {
    const a = r * (n + 1) + k, b = a + n + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array((rings + 1) * (n + 1) * 3), 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 4));
  g.setIndex(idx);
  const fanMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const fan = new THREE.Mesh(g, fanMat);
  fan.renderOrder = 3;
  // crisp edges of the view
  const og = new THREE.BufferGeometry();
  og.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array((n + 3) * 3), 3));
  const lineMat = new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: edge, depthWrite: false });
  const outline = new THREE.Line(og, lineMat);
  outline.renderOrder = 3;
  const grp = new THREE.Group();
  grp.add(fan, outline);
  grp.visible = false;
  let key = '';
  grp.userData.set = (a0, a1, R) => {
    const k2 = `${a0.toFixed(3)}|${a1.toFixed(3)}|${R.toFixed(2)}`;
    if (k2 === key) return;
    key = k2;
    const P = g.attributes.position.array;
    let o = 0;
    for (let r = 0; r <= rings; r++) {
      const rr = (r / rings) * R;
      for (let k = 0; k <= n; k++) { const ph = a0 + (a1 - a0) * k / n; P[o++] = rr * Math.cos(ph); P[o++] = h; P[o++] = -rr * Math.sin(ph); }
    }
    g.attributes.position.needsUpdate = true;
    g.computeBoundingSphere();
    const Q = og.attributes.position.array;
    o = 0;
    Q[o++] = 0; Q[o++] = h + 0.01; Q[o++] = 0;
    for (let k = 0; k <= n; k++) { const ph = a0 + (a1 - a0) * k / n; Q[o++] = R * Math.cos(ph); Q[o++] = h + 0.01; Q[o++] = -R * Math.sin(ph); }
    Q[o++] = 0; Q[o++] = h + 0.01; Q[o++] = 0;
    og.attributes.position.needsUpdate = true;
    og.computeBoundingSphere();
  };
  // overall strength (dimmed while the head turns)
  grp.userData.fade = (f) => { fanMat.opacity = f; lineMat.opacity = edge * f; };
  return grp;
}

// the driver's 120-degree view (cone observation)
function makeCone(spec, color) {
  const R = spec.view_range ?? 1.6 * (spec.view_radius ?? 14);
  const grp = makeFan(color);
  grp.userData.set(-Math.PI / 3, Math.PI / 3, R);
  grp.userData.R = R;
  return grp;
}

// driver's-seat view (obs "driver"): the direct cone around the gaze, the active mirror's / reversing camera's view,
// and a slim head needle from the eye (to the mirror or screen while looking at one). All in track coordinates.
const AID_COLOR = '#fff4d6';
function makeDriverFx(color) {
  const grp = new THREE.Group();
  const direct = makeFan(color, { alpha: 0.34, edge: 0.7 });
  const aid = makeFan(AID_COLOR, { alpha: 0.6, edge: 0.95, h: 0.1 });
  const periph = [makeFan(color, { alpha: 0.16, edge: 0.3, rings: 3, n: 16, h: 0.085 }), makeFan(color, { alpha: 0.16, edge: 0.3, rings: 3, n: 16, h: 0.085 })];
  direct.visible = aid.visible = true;
  const ng = new THREE.BufferGeometry();
  ng.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
  const needle = new THREE.Line(ng, new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9, depthTest: false, depthWrite: false }));
  needle.renderOrder = 8;
  needle.frustumCulled = false;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.075, 12, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.5), depthTest: false, transparent: true, opacity: 0.9 }));
  head.renderOrder = 8;
  grp.add(direct, aid, ...periph, needle, head);
  grp.visible = false;
  return { grp, direct, aid, periph, needle, head };
}

// pose the driver overlay for leg pose st and head state hs; returns the view geometry
function poseDriverFx(fx, leg, st, hs, combo) {
  const geo = viewGeometry(leg.rig, leg.spec, st, hs);
  const sty = combo.style;
  const fade = hs.turning ? 0.3 : 1;
  fx.direct.visible = !!geo.direct;
  if (geo.direct) {
    fx.direct.position.set(geo.direct.origin[0], 0, -geo.direct.origin[1]);
    fx.direct.userData.set(geo.direct.a0, geo.direct.a1, geo.direct.range);
    fx.direct.userData.fade(fade);
  }
  fx.periph.forEach((f, k) => {
    const v = geo.periph[k];
    f.visible = !!v && !hs.turning;
    if (f.visible) { f.position.set(v.origin[0], 0, -v.origin[1]); f.userData.set(v.a0, v.a1, v.range); }
  });
  fx.aid.visible = !!geo.aid && !hs.turning && geo.aid.range > 0;
  if (fx.aid.visible) {
    fx.aid.position.set(geo.aid.origin[0], 0, -geo.aid.origin[1]);
    fx.aid.userData.set(geo.aid.a0, geo.aid.a1, geo.aid.range);
  }
  // head needle: to the mirror / screen being watched, else a short stroke along the gaze
  const [ex, ey] = geo.eye, H = sty.head;
  let to;
  if (geo.aid && !hs.turning && geo.aid.kind === 'camera' && combo.screen) {
    // the glow drawn on the dashboard (vehicle-local x forward, -z left)
    const sx = combo.screen.position.x, sy = -combo.screen.position.z;
    to = [st.x + sx * Math.cos(st.th1) - sy * Math.sin(st.th1), st.y + sx * Math.sin(st.th1) + sy * Math.cos(st.th1), combo.screen.position.y];
  } else if (geo.aid && !hs.turning) to = [...geo.aid.via, sty.u];
  else { const L = Math.max(1.2, leg.spec.body_width * 0.75); to = [ex + L * Math.cos(geo.gaze), ey + L * Math.sin(geo.gaze), H]; }
  const p = fx.needle.geometry.attributes.position.array;
  p[0] = ex; p[1] = H; p[2] = -ey; p[3] = to[0]; p[4] = to[2]; p[5] = -to[1];
  fx.needle.geometry.attributes.position.needsUpdate = true;
  fx.needle.material.opacity = hs.turning ? 0.5 : 0.9;
  fx.head.position.set(ex, H, -ey);
  return geo;
}

function disposeObject(o) {
  o.parent?.remove(o);
  o.traverse((x) => {
    x.geometry?.dispose?.();
    const m = x.material;
    if (Array.isArray(m)) m.forEach((mm) => mm.dispose?.()); else m?.dispose?.();
  });
}

function disposeActors() {
  for (const run of state.runs) for (const a of run.actors || []) {
    disposeObject(a.combo.root);
    disposeObject(a.trail.line);
    for (const mk of a.markers) disposeObject(mk.g);
    if (a.cone) disposeObject(a.cone);
    if (a.dfx) disposeObject(a.dfx.grp);
  }
}

function buildActors() {
  for (const run of state.runs) {
    run.actors = run.legs.map((leg) => {
      const T = state.tracks[leg.i];
      const combo = buildCombo(leg.spec, run.color, leg.rig);
      T.grp.add(combo.root);
      const trail = makeTrail(leg, run.color);
      T.grp.add(trail.line);
      let cone = null;
      if (run.cone) { cone = makeCone(leg.spec, run.color); T.grp.add(cone); }
      let dfx = null;
      if (run.driver && leg.trace.gaze?.length) { dfx = makeDriverFx(run.color); T.grp.add(dfx.grp); }
      const markers = [];
      for (const e of leg.events) {
        if (e.type === 'contact') {
          const mk = makeMarker('#ff4d3f', Math.max(0.35, leg.spec.combo_length * 0.05));
          mk.g.position.copy(toV(e.x, e.y));
          markers.push({ ...mk, t: e.t });
        } else if (e.type === 'jackknife') {
          const st = sampleLeg(leg, e.t);
          const mk = makeMarker('#ff9a3d', Math.max(0.45, leg.spec.combo_length * 0.06));
          mk.g.position.copy(toV(st.x - leg.spec.hitch_offset * Math.cos(st.th1), st.y - leg.spec.hitch_offset * Math.sin(st.th1)));
          markers.push({ ...mk, t: e.t });
        }
      }
      for (const mk of markers) T.fx.add(mk.g);
      return { combo, trail, markers, cone, dfx };
    });
  }
}

// ------------------------------------------------------------------ HUD
const BOARD_MAX = 5;
function buildHud() {
  state.sceneRev = (state.sceneRev || 0) + 1;
  const sp = $('speeds');
  sp.innerHTML = '';
  for (const s of SPEEDS) {
    const b = document.createElement('button');
    b.className = 'btn sm'; b.textContent = `${s}×`; b.dataset.speed = s;
    b.onclick = () => { state.speed = s; syncButtons(); };
    sp.appendChild(b);
  }
  $('play').onclick = () => setPlaying(!state.playing);
  $('scrub').oninput = (e) => { seek(Number(e.target.value) / 1000 * state.Tmax); };
  for (const b of $('cams').querySelectorAll('[data-cam]')) b.onclick = () => setCam(b.dataset.cam);
  for (const b of $('clockmode').querySelectorAll('[data-clock]')) b.onclick = () => setClockMode(b.dataset.clock, true);
  const hl = $('heatlegs');
  hl.innerHTML = '';
  state.tracks.forEach((t, i) => {
    const b = document.createElement('button');
    b.className = 'btn sm'; b.dataset.leg = i; b.textContent = `L${i + 1}`;
    b.title = `Follow leg ${i + 1} (${VEHICLE_NAMES[t.tr.vehicle] || t.tr.vehicle})`;
    b.onclick = () => setHeatLeg(i);
    hl.appendChild(b);
  });

  // standings: compact rows, re-ordered each frame; only the top few are shown
  const rows = $('board-rows');
  rows.innerHTML = '';
  state.runs.forEach((run, k) => {
    const li = document.createElement('li');
    li.className = 'row';
    li.innerHTML = `<span class="pos"></span><i class="dot" style="background:${run.color}"></i><div class="who"><div class="name"></div><div class="meta"><span class="legs">${state.tracks.map(() => '<b></b>').join('')}</span><span class="var"></span></div></div><span class="chip"></span><span class="time"></span>`;
    li.querySelector('.name').textContent = run.lab?.model || shortLabel(run.label);
    li.querySelector('.var').textContent = run.variantTag || '';
    li.title = run.lab?.full || run.label;
    li.onclick = () => setFocus(k);
    rows.appendChild(li);
    run.row = { li, pos: li.querySelector('.pos'), legs: [...li.querySelectorAll('.legs b')], chip: li.querySelector('.chip'), time: li.querySelector('.time') };
  });
  state.dirtyTicks = true;
  syncButtons();
}

function syncButtons() {
  $('play').textContent = state.playing ? '❚❚' : '▶';
  $('play').setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
  for (const b of $('speeds').children) b.classList.toggle('on', Number(b.dataset.speed) === state.speed);
  for (const b of $('cams').querySelectorAll('[data-cam]')) b.classList.toggle('on', b.dataset.cam === state.cam);
  for (const b of $('clockmode').querySelectorAll('[data-clock]')) b.classList.toggle('on', b.dataset.clock === state.clockMode);
  for (const b of $('heatlegs').children) b.classList.toggle('on', Number(b.dataset.leg) === state.heatLeg);
  for (const b of $('views').children) b.classList.toggle('on', b.dataset.view === state.view);
  $('heatlegs').hidden = state.clockMode !== 'heats';
  $('cams').hidden = state.view !== 'broadcast';
  $('compare-btn').classList.toggle('on', state.compareOpen);
  $('runs-btn').classList.toggle('on', state.pickerOpen);
  $('clock-speed').textContent = `${state.speed}×`;
  state.runs.forEach((r, k) => r.row?.li.classList.toggle('focus', k === state.focus));
}

// transport event strip: one lane per run (the focused one brightest), ticks for what happened
const STRIP_COL = { contact: '#ff6b5e', jackknife: '#ff8a3d', controller: '#f4f1ea', controller_error: '#ff6b5e', wake: '#c9a7ff', phase: '#7cc4ff' };
function buildTicks() {
  const ticks = $('ticks');
  ticks.innerHTML = '';
  state.dirtyTicks = false;
  const fr = state.runs[state.focus];
  if (!fr) return;
  const pct = (t) => `${(100 * t / state.Tmax).toFixed(3)}%`;
  const X = (t) => (1000 * t / state.Tmax).toFixed(2);
  const heats = state.clockMode === 'heats';
  const lanes = [fr, ...state.runs.filter((r) => r !== fr)].slice(0, 15);
  const LH = Math.max(1.6, Math.min(4.5, 22 / lanes.length));
  let svg = '';
  lanes.forEach((run, k) => {
    const y = (k + 0.5) * LH, focus = run === fr;
    for (const leg of run.legs) {
      if (heats && leg.i !== Math.min(state.heatLeg, run.legs.length - 1)) continue;
      svg += `<line x1="${X(leg.start)}" x2="${X(leg.end)}" y1="${y.toFixed(2)}" y2="${y.toFixed(2)}" stroke="${run.color}" stroke-width="${focus ? Math.min(3, LH) : Math.max(1, LH - 1.2)}" stroke-opacity="${focus ? 0.95 : 0.5}" vector-effect="non-scaling-stroke"/>`;
      if (focus && run.variant.control.includes('realtime')) for (const u of leg.turns) if (u.t_reply > u.t + 0.05) svg += `<rect x="${X(leg.start + u.t)}" y="0" width="${(1000 * (Math.min(u.t_reply, leg.time) - u.t) / state.Tmax).toFixed(2)}" height="${LH.toFixed(2)}" fill="rgba(201,167,255,.45)"/>`;
      for (const e of leg.events) {
        const c = STRIP_COL[e.type];
        if (!c || (!focus && e.type !== 'contact' && e.type !== 'jackknife' && e.type !== 'phase')) continue;
        const tall = e.type === 'phase' ? LH * lanes.length : LH;
        svg += `<line x1="${X(leg.start + e.t)}" x2="${X(leg.start + e.t)}" y1="${(e.type === 'phase' ? 0 : y - LH / 2).toFixed(2)}" y2="${(e.type === 'phase' ? tall : y + LH / 2).toFixed(2)}" stroke="${c}" stroke-width="${focus ? 2 : 1.5}" vector-effect="non-scaling-stroke"/>`;
      }
    }
  });
  ticks.innerHTML = `<svg viewBox="0 0 1000 ${(LH * lanes.length).toFixed(1)}" preserveAspectRatio="none">${svg}</svg>`;
  ticks.querySelector('svg').style.height = `${Math.min(22, LH * lanes.length).toFixed(1)}px`;
  // the focused run's legs: separators and labels
  for (const leg of fr.legs) {
    if (heats && leg.i !== Math.min(state.heatLeg, fr.legs.length - 1)) continue;
    const s = document.createElement('span');
    s.textContent = `L${leg.i + 1} ${leg.spec.key}`;
    s.style.left = pct(leg.start);
    s.className = 'focusleg';
    ticks.appendChild(s);
  }
  // legs that are short on this clock would print on top of each other: shorten, then drop
  const labs = [...ticks.querySelectorAll('.focusleg')];
  for (const pass of [0, 1]) {
    let right = -Infinity;
    for (const s of labs) {
      if (s.hidden) continue;
      const l = s.offsetLeft;
      if (l < right + 6) { if (pass === 0) s.textContent = s.textContent.split(' ')[0]; else s.hidden = true; }
      else right = l + s.offsetWidth;
    }
  }
}

const GA = { cx: 60, cy: 62, r: 50 };
function arcPath(a0, a1) {
  // angles in degrees from vertical; negative = right side of the gauge
  const p = (a) => [GA.cx + GA.r * Math.sin(a * Math.PI / 180), GA.cy - GA.r * Math.cos(a * Math.PI / 180)];
  const [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${GA.r} ${GA.r} 0 0 ${a1 > a0 ? 1 : 0} ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
let gaugeFor = null;
function setGauge(spec, hitchDeg) {
  const crit = Math.min(spec.critical_hitch_deg ?? spec.max_hitch_deg, spec.max_hitch_deg);
  const mech = spec.max_hitch_deg;
  const map = (d) => Math.max(-90, Math.min(90, d)) * (90 / 90);
  if (gaugeFor !== spec.key) {
    gaugeFor = spec.key;
    // driver's left (positive hitch) drawn on the left of the dial
    $('g-crit-l').setAttribute('d', crit < mech - 0.5 ? arcPath(-map(mech), -map(crit)) : '');
    $('g-crit-r').setAttribute('d', crit < mech - 0.5 ? arcPath(map(crit), map(mech)) : '');
    $('g-mech-l').setAttribute('d', arcPath(-90, -map(mech)));
    $('g-mech-r').setAttribute('d', arcPath(map(mech), 90));
  }
  const a = -map(hitchDeg);
  $('g-needle').setAttribute('transform', `rotate(${a.toFixed(2)} ${GA.cx} ${GA.cy})`);
  const h = Math.abs(hitchDeg);
  $('hitch-val').style.color = h >= mech - 0.5 ? 'var(--bad)' : h >= crit ? 'var(--warn)' : '';
}

function setText(el, txt) { if (el.textContent !== txt) el.textContent = txt; }
function setHTML(el, html) { if (el.innerHTML !== html) el.innerHTML = html; }
const escHTML = (s) => s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

// Reasoning summaries arrive as markdown sections ("**Title**\n\nbody"). Show the latest section.
function thoughtHTML(raw) {
  const secs = raw.split(/\n\s*\n(?=\*\*)/).map((x) => x.trim()).filter(Boolean);
  const last = secs[secs.length - 1] || raw;
  const m = last.match(/^\*\*(.+?)\*\*\s*/);
  const body = (m ? last.slice(m[0].length) : last).replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();
  return (m ? `<span class="thought-title">${escHTML(m[1])}</span>` : '') + escHTML(body);
}
function thoughtTitle(raw) {
  const secs = raw.split(/\n\s*\n(?=\*\*)/).map((x) => x.trim()).filter(Boolean);
  const last = secs[secs.length - 1] || raw;
  const m = last.match(/^\*\*(.+?)\*\*/);
  return m ? m[1] : last.replace(/\*\*/g, '').split(/[.\n]/)[0];
}
function gazeWords(g) {
  let d = ((g % 360) + 540) % 360 - 180; // (-180, 180]
  const a = Math.abs(d);
  const where = a < 20 ? 'ahead' : a > 160 ? 'behind' : a < 70 ? (d > 0 ? 'ahead-left' : 'ahead-right') : a < 110 ? (d > 0 ? 'left' : 'right') : (d > 0 ? 'behind-left' : 'behind-right');
  return `${d >= 0 ? '+' : ''}${d.toFixed(0)}° ${where}`;
}
function setChip(el, status) {
  const cls = `chip ${status}`;
  if (el.className !== cls) el.className = cls;
  setText(el, STATUS_TEXT[status] || status);
}

function legResult(L) {
  const m = L.L.metrics;
  if (L.outcome === 'FINISHED') return `Leg ${L.i + 1} parked in ${m.time.toFixed(1)} s`;
  const why = { TIME_UP: 'Time up', DNF: 'DNF' }[L.outcome] || L.outcome;
  return `Leg ${L.i + 1} · ${why} · ${(m.short_m ?? 0).toFixed(1)} m short`;
}

const STOP_WORD = { completed: 'done', finished: 'parked', contact: 'CONTACT', blocked: 'blocked', jackknife: 'JACKKNIFE', time_up: 'time up', rejected: 'rejected' };
function resultLine(c) {
  if (!c) return '';
  const r = c.result || {};
  const a = c.args || {};
  if (c.name === 'look') return `${c.command != null ? `<b>#${c.command}</b> ` : ''}<b>look</b>(${Number(a.yaw_deg ?? 0).toFixed(0)}°) → view ${r.yaw_deg !== undefined ? `${Number(r.yaw_deg).toFixed(0)}°` : 'updated'}`;
  const woke = c.woke?.length ? ` · <span class="err">${escHTML(c.woke.join(', '))}</span>` : '';
  const span = c.t1 != null && c.t0 != null ? ` · ${(c.t1 - c.t0).toFixed(1)} s` : '';
  if (c.name === 'set_controller') return `<b>set_controller</b> → ${c.error && !c.woke ? '<span class="err">failed to start</span>' : 'driving'}${span}${woke}`;
  if (c.name === 'keep_driving') return `<b>keep_driving</b>(${Number(a.check_in_s ?? 0).toFixed(0)} s)${span}${woke}`;
  if (c.name === 'drive' && a.duration_s !== undefined) {
    return `<b>#${c.command ?? '?'}</b> drive(steer ${Number(a.steer_deg ?? 0).toFixed(0)}°, ${Number(a.speed_mps ?? 0).toFixed(1)} m/s, ${Number(a.duration_s).toFixed(1)} s)${span}${woke}`;
  }
  if (c.name === 'run_program') {
    const n = (c.commands || []).length;
    const ok = c.error ? '<span class="err">error</span>' : 'ok';
    return `<b>run_program</b> → ${n} command${n === 1 ? '' : 's'} · ${ok}${c.t1 !== undefined ? ` · ${(c.t1 - c.t0).toFixed(1)} s` : ''}`;
  }
  const sp = a.speed_mps != null ? `, ${Number(a.speed_mps).toFixed(1)} m/s` : '';
  const cmd = `drive(steer ${Number(a.steer_deg ?? 0).toFixed(0)}°, ${Number(a.distance_m ?? 0) >= 0 ? '+' : ''}${Number(a.distance_m ?? 0).toFixed(1)} m${sp})`;
  return `<b>#${c.command ?? '?'}</b> ${cmd} → ${(r.traveled ?? 0).toFixed(1)} m ${STOP_WORD[r.stop_reason] || ''}`;
}

const clip = (txt, n) => (txt.length > n ? `${txt.slice(0, n - 1)}…` : txt);
// compact code preview for the lower-third: the first few meaningful lines (no imports or blanks), then the output
function programHTML(c, tl = Infinity) {
  const code = String(c.args?.code ?? '').replace(/\t/g, '  ').split('\n');
  const body = code.filter((l) => l.trim() && !/^\s*(import|from)\s/.test(l));
  const N = 4;
  const shown = body.slice(0, N).map((l) => escHTML(clip(l, 62))).join('\n') + (body.length > N ? `\n<span class="more">… ${body.length - N} more lines · full code in Inspect</span>` : '');
  let out = '';
  const so = String(c.stdout ?? '').trim();
  const done = c.t1 === undefined || c.t1 === null ? !!(so || c.error) : tl >= c.t1 - 1e-6;
  if (!done) {
    out = `<div class="prog-out dim">running… ${c.t1 != null ? `(${Math.max(0, c.t1 - tl).toFixed(1)} s left)` : ''}</div>`;
  } else if (c.error) {
    const lines = String(c.error).trim().split('\n');
    out = `<div class="prog-out err">${escHTML(lines.slice(-2).map((l) => clip(l, 80)).join('\n'))}</div>`;
  } else if (so) {
    const lines = so.split('\n');
    out = `<div class="prog-out">${escHTML(lines.slice(-2).map((l) => clip(l, 80)).join('\n'))}</div>`;
  }
  return `<pre class="prog">${shown}</pre>${out}`;
}

// ------------------------------------------------------------------ frame update
const tmpV = new THREE.Vector3();
const focusInfo = { pos: new THREE.Vector3(), front: new THREE.Vector3(), rear: new THREE.Vector3(), C: 8 };
const isRealtime = (run) => String(run.variant.control).includes('realtime');
const fmtDeg = (d) => `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(0)}°`;
let chips = null, grid = null, inspect = null, results = null, lastPanel = 0;

// what the in-scene chips need from one rig this frame (positions are track-local; projected later)
function chipCandidates(run, k, info, T) {
  const { leg, st, ls, tl, ended, p } = info;
  const sp = leg.spec, focus = k === state.focus;
  // the focused run carries its full name; the others a compact one ("Sol #2"), full name on hover in the standings
  const name = escHTML(focus ? run.name : run.mini);
  const mini = escHTML(run.mini);
  const vt = '';
  const hitch = [st.x - sp.hitch_offset * Math.cos(st.th1), st.y - sp.hitch_offset * Math.sin(st.th1)];
  const combo = run.actors[run.legs.indexOf(leg)].combo;
  const top = [(st.x + p.ax) / 2, (st.y + p.ay) / 2, combo.height + 0.3];
  const base = { key: `${run.runId}:${leg.i}`, ti: leg.i, color: run.color, focus, info: { run: run.runId, leg: leg.i } };
  const out = [];
  const think = !ended && isRealtime(run) ? thinkingAt(leg, tl) : null;
  const meanwhile = think ? Math.max(0, leg.dist[st.idx] - leg.dist[sampleLeg(leg, think.t).idx]) : 0;
  const thinkTxt = think ? `thinking ${(tl - think.t).toFixed(0)} s` : '';
  if (ended) {
    if (leg.outcome === 'FINISHED' && T - leg.end < 25) out.push({ ...base, prio: 3, cls: 'ok', at: top, html: `<em>Parked</em> · ${name}<small>${vt}leg ${leg.i + 1} in ${fmtClock(leg.time)}</small>`, short: `<em>Parked</em> · ${mini}`, kind: 'PARKED' });
    return out;
  }
  const hd = hitchDeg(st), crit = critOf(sp);
  if (ls.status === 'PINNED') {
    const c = ls.contact;
    const at = c && c.x !== undefined ? [c.x, c.y, 0.25] : [...hitch, 0.4];
    out.push({ ...base, prio: 9, cls: 'bad', at, ring: 'bad', kind: 'PINNED', short: `<em>Pinned</em> · ${mini}`,
      html: `<em>Pinned</em> · ${name}<small>${vt}on the rail ${(tl - ls.since).toFixed(0)} s · touch #${ls.touches}${think ? ` · ${thinkTxt}` : ''}</small>` });
    return out;
  }
  if (ls.status === 'JACKKNIFED') {
    out.push({ ...base, prio: 8, cls: 'bad', at: [...hitch, 0.5], ring: 'jk', kind: 'JACKKNIFED', short: `<em>Jackknifed</em> · ${mini}`, html: `<em>Jackknifed</em> · ${name}<small>${vt}hitch ${fmtDeg(hd)} · limit ${sp.max_hitch_deg}°</small>` });
    return out;
  }
  if (ls.status === 'COMMITTED' || Math.abs(hd) >= crit - 3) {
    out.push({ ...base, prio: 6, cls: 'warn', at: [...hitch, 0.5], ring: 'warn', kind: 'COMMITTED', short: `<em>${fmtDeg(hd)}</em> · ${mini}`,
      html: `<em>${ls.status === 'COMMITTED' ? 'Committed' : 'Hitch'} ${fmtDeg(hd)}</em> · ${name}<small>${vt}critical ${crit.toFixed(0)}°</small>` });
  }
  if (think) {
    const who = run.code ? `controller #${Math.max(1, ls.ctrl)} keeps driving` : 'last command keeps running';
    out.push({ ...base, key: `${base.key}:think`, prio: 5, cls: 'think', at: top, kind: 'THINKING', short: `<em>Thinking ${(tl - think.t).toFixed(0)} s</em> · ${mini}`,
      html: `<em>Thinking ${(tl - think.t).toFixed(0)} s</em> · ${name}<small>${who} · ${meanwhile.toFixed(1)} m meanwhile</small>` });
  }
  if (!out.length) {
    out.push(focus ? { ...base, prio: 4, cls: 'flag', at: top, kind: 'FLAG', html: `${escHTML(run.lab?.model || run.mini)}${run.lab?.tag ? ` <small class="v">${escHTML(clip(run.lab.tag, 34))}</small>` : ''}` }
      : { ...base, prio: 1, cls: 'flag mini', at: top, kind: 'FLAG', html: mini, short: mini });
  }
  return out;
}

// identical replays (study attempts, repeated trials) often sit in the same state at the same spot: one callout and one
// ring per group, "4 pinned here · naive camera" with the runs listed under it (full names on hover)
const MERGE_M = 1.5;
const MERGE_WORD = { PINNED: 'pinned here', JACKKNIFED: 'jackknifed here', COMMITTED: 'committed here', THINKING: 'thinking here', PARKED: 'parked here', FLAG: 'here' };
function mergeStacked(list) {
  const groups = [];
  for (const c of list) { // list is sorted by priority: the focused run, then the most urgent, heads its group
    const g = groups.find((x) => x.head.kind === c.kind && x.head.ti === c.ti && Math.hypot(x.head.at[0] - c.at[0], x.head.at[1] - c.at[1]) < MERGE_M);
    if (g) g.rest.push(c); else groups.push({ head: c, rest: [] });
  }
  const byId = new Map(state.runs.map((r) => [r.runId, r]));
  for (const { head, rest } of groups) {
    if (!rest.length) continue;
    const runs = [head, ...rest].map((c) => byId.get(c.info?.run)).filter(Boolean);
    const models = [...new Set(runs.map((r) => r.lab?.model || r.mini))];
    const tags = runs.map((r) => (models.length === 1 ? (r.lab?.tag || r.mini) : r.mini));
    const shown = tags.slice(0, 4).map(escHTML).join(' · ') + (tags.length > 4 ? ` · +${tags.length - 4}` : '');
    const n = runs.length;
    const word = MERGE_WORD[head.kind] || 'here';
    const title = escHTML(runs.map((r) => r.name).join('\n'));
    const lead = head.kind === 'FLAG' ? `${n} runs here` : `<em>${n} ${word}</em>`;
    head.html = `<span class="stack" title="${title}">${lead} · ${escHTML(models.length === 1 ? models[0] : `${models.length} models`)}<small>${shown}</small></span>`;
    head.short = head.html;
    head.merged = n;
    head.prio += 0.25 * Math.min(4, n - 1); // a stack is more newsworthy than one rig
    for (const c of rest) { c.html = null; c.ring = null; c.mergedInto = head.key; }
  }
}

function hudRects() {
  const out = [];
  for (const id of ['clock', 'board', 'radio', 'runs-panel']) {
    const e = $(id);
    if (!e || e.hidden) continue;
    const r = e.getBoundingClientRect();
    if (r.width) out.push({ x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 });
  }
  for (const sel of ['.dock', '.viewseg', '.relay-title', '.bc-hint']) {
    const el = document.querySelector(sel);
    if (!el || el.hidden) continue;
    const r = el.getBoundingClientRect();
    if (!r.width) continue;
    out.push({ x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 });
  }
  return out;
}

function update(dtReal) {
  const T = state.T;
  const nLegs = state.tracks.length;
  const heats = state.clockMode === 'heats';
  const broadcast = state.view === 'broadcast';
  const standings = [];
  const cands = [];
  const parkedOn = new Set();

  for (const [k, run] of state.runs.entries()) {
    const { active, finishedAll } = runAt(run, T, state.heatLeg);
    let score = 0, legsDone = 0, driving = 0;
    let liveStatus = 'READY', liveInfo = null;
    for (const [j, leg] of run.legs.entries()) {
      const A = run.actors[j]; // actors are aligned with run.legs (a record may skip courses)
      const { combo, trail, markers, cone, dfx } = A;
      const started = T >= leg.start;
      const ended = !leg.running && T >= leg.end;
      let tl = 0, opacity = 1;
      if (!started) {
        // relay mode: only the first run shows a faint waiting rig on later courses
        combo.root.visible = !heats && k === 0 && leg.i > 0;
        opacity = 0.28;
        trail.line.visible = false;
      } else {
        combo.root.visible = true;
        trail.line.visible = true;
        tl = Math.min(T - leg.start, leg.time);
        if (ended) opacity = heats ? 0.55 : 0.32;
      }
      let st = null, p = null;
      if (combo.root.visible) {
        st = sampleLeg(leg, tl);
        p = poseCombo(combo, leg.spec, st);
        setOpacity(combo, opacity);
        if (trail.line.visible) {
          trail.line.geometry.instanceCount = Math.max(0, st.idx);
          trail.mat.opacity = ended ? 0.4 : k === state.focus ? 0.95 : 0.7;
        }
      }
      if (ended && leg.outcome === 'FINISHED' && (k === state.focus || T - leg.end < 25)) parkedOn.add(leg.i);
      if (started && st) {
        const ls = legStatusAt(leg, tl);
        const prog = ended && leg.outcome === 'FINISHED' ? 1 : progressOf(leg, st);
        if (ended) { score += leg.outcome === 'FINISHED' ? 1 : (leg.L.metrics.progress ?? prog); if (leg.outcome === 'FINISHED') legsDone++; }
        else { score += prog * 0.999; driving++; }
        if (leg.i === active || (heats && leg === run.legs[Math.min(state.heatLeg, run.legs.length - 1)])) {
          liveStatus = ended ? leg.outcome : ls.status;
          liveInfo = { leg, st, ls, prog, tl, p, ended };
        }
      }
      // the driver's view (cone observation): focused run only, while the leg is being driven
      if (cone) {
        const on = !!st && started && !ended && k === state.focus;
        cone.visible = on && broadcast;
        if (on) {
          const yaw = gazeAt(leg, tl) * Math.PI / 180;
          const seat = leg.spec.body_front * 0.4;
          cone.position.set(st.x + seat * Math.cos(st.th1), 0, -(st.y + seat * Math.sin(st.th1)));
          cone.rotation.y = st.th1 + yaw;
          A.gaze = gazeAt(leg, tl);
          A.coneOn = true;
        } else A.coneOn = false;
      }
      // driver's-seat view: focused run only, while the leg is being driven; the watched mirror / screen glows
      if (dfx) {
        const on = !!st && started && !ended && k === state.focus;
        const hs = st ? headAt(leg, tl, st.idx) : null;
        A.head = on ? hs : null;
        dfx.grp.visible = on && broadcast && !!hs;
        setLook(combo, on && hs ? hs.at : null);
        A.view = dfx.grp.visible ? poseDriverFx(dfx, leg, st, hs, combo) : null;
      }
      // past rail touches stay marked on the ground for the focused run only
      for (const mk of markers) {
        const on = started && tl >= mk.t && k === state.focus;
        mk.g.visible = on;
        if (on) {
          const age = tl - mk.t;
          mk.g.scale.setScalar(age < 2 ? 1 + 0.6 * Math.sin(age * 9) * (1 - age / 2) : 1);
          mk.ring.material.opacity = ended ? 0.4 : 0.8;
        }
      }
    }
    const hl = run.legs[Math.min(state.heatLeg, run.legs.length - 1)];
    const clock = heats ? (hl ? Math.max(0, Math.min(T - hl.start, hl.time)) : 0) : elapsedOf(run, T);
    run.live = { active, finishedAll, liveStatus, liveInfo, score, clock, legsDone, driving };
    standings.push(run);
    if (broadcast && liveInfo && !finishedAll) for (const c of chipCandidates(run, k, liveInfo, T)) cands.push(c);

    const endChip = legsDone === nLegs ? 'FINISHED' : run.rec.partial || run.isLive ? 'WAITING' : 'DNF';
    run.row.legs.forEach((b, i) => {
      const L = run.legs.find((l) => l.i === i);
      if (!L) { b.className = ''; return; }
      const ended = !L.running && T >= L.end;
      b.className = ended ? (L.outcome === 'FINISHED' ? 'done' : 'fail') : (T >= L.start && (heats || L.i === run.legs[active]?.i) ? 'live' : '');
    });
    setChip(run.row.chip, finishedAll ? endChip : liveStatus);
    setText(run.row.time, fmtClock(clock));
  }
  for (const tk of state.tracks) if (tk.parked) tk.parked.visible = parkedOn.has(state.tracks.indexOf(tk));

  // standings: order by score; show the top few (and always the focused run)
  standings.sort((a, b) => (b.live.score - a.live.score) || (a.live.clock - b.live.clock));
  const rows = $('board-rows');
  const fpos = standings.indexOf(state.runs[state.focus]);
  standings.forEach((run, i) => {
    setText(run.row.pos, String(i + 1));
    if (rows.children[i] !== run.row.li) rows.insertBefore(run.row.li, rows.children[i] || null);
    const show = i < BOARD_MAX - (fpos >= BOARD_MAX ? 1 : 0) || i === fpos;
    if (run.row.li.hidden === show) run.row.li.hidden = !show;
  });
  const hiddenN = Math.max(0, standings.length - BOARD_MAX);
  const more = $('board-more');
  more.hidden = !hiddenN;
  if (hiddenN) {
    const rest = standings.filter((r) => r.row.li.hidden);
    const pinned = rest.filter((r) => r.live.liveStatus === 'PINNED' && !r.live.finishedAll).length;
    setHTML(more, `+${hiddenN} more${pinned ? ` · <span class="bad">${pinned} pinned</span>` : ''} · G for the grid`);
  }
  const lead = standings[0];
  setText($('board-leg'), !lead ? '' : heats ? `heats · leg ${state.heatLeg + 1}` : `leader on leg ${lead.live.active + 1}/${nLegs}`);

  // clock
  setText($('clock-time'), fmtClock(T));
  setText($('clock-eyebrow'), heats ? `Heats · leg ${state.heatLeg + 1}` : 'Relay clock');
  setText($('clock-of'), state.anyLive ? `live · ${fmtClock(state.Tmax)} so far` : `of ${fmtClock(state.Tmax)}`);
  const atEdge = state.anyLive && T >= state.Tmax - 0.75;
  $('live-pill').hidden = !state.anyLive;
  $('live-pill').classList.toggle('edge', atEdge);
  const sc = $('scrub');
  const v = Math.round(T / state.Tmax * 1000);
  if (Number(sc.value) !== v && document.activeElement !== sc) sc.value = v;
  const pc = `${(T / state.Tmax * 100).toFixed(2)}%`;
  sc.style.setProperty('--p', pc);
  $('playhead').style.left = pc;
  if (state.dirtyTicks) buildTicks();

  // focused run → lower-third
  const run = state.runs[state.focus];
  $('radio').hidden = !run;
  if (run) {
    const { liveInfo, active } = run.live;
    const leg = heats ? run.legs[Math.min(state.heatLeg, run.legs.length - 1)] : run.legs[active];
    $('radio-swatch').style.background = run.color;
    setText($('radio-name'), run.name || run.label);
    $('radio-name').title = run.lab?.full || '';
    const ph = leg ? phaseInfo(leg, liveInfo ? liveInfo.tl : 0) : null;
    $('radio-leg').title = ph?.label || '';
    setHTML($('radio-leg'), leg ? `Leg ${leg.i + 1}/${nLegs} · ${escHTML(leg.spec.name)} + ${escHTML(leg.spec.trailer_name || 'trailer')}${ph ? ` · <span class="phase">${phaseText(ph)}</span>${ph.label ? ` · ${escHTML(ph.label)}` : ''}` : ''}` : 'Waiting to start');
    setChip($('radio-status'), run.live.liveStatus);
    const think = liveInfo && !liveInfo.ended && isRealtime(run) ? thinkingAt(liveInfo.leg, liveInfo.tl) : null;
    $('radio-think').hidden = !think;
    if (think) setText($('radio-think'), `Thinking ${(liveInfo.tl - think.t).toFixed(0)} s`);
    if (leg) {
      const st = liveInfo ? liveInfo.st : sampleLeg(leg, 0);
      const hd = hitchDeg(st);
      setGauge(leg.spec, hd);
      setText($('hitch-val'), `${hd >= 0 ? '+' : ''}${hd.toFixed(1)}°`);
      setText($('steer-val'), `${(st.steer * 180 / Math.PI).toFixed(1)}°`);
      const hasV = st.v !== null && st.v !== undefined;
      $('speed-read').hidden = !hasV;
      if (hasV) {
        setText($('speed-val'), `${st.v < -0.02 ? '−' : ''}${Math.abs(st.v).toFixed(1)} m/s`);
        setText($('speed-dir'), st.v < -0.02 ? 'Reversing' : 'Speed');
      }
      const A = run.actors[run.legs.indexOf(leg)];
      const gz = A?.gaze;
      if (run.driver) {
        $('gaze-read').hidden = !(liveInfo && !liveInfo.ended && A?.head);
        if (A?.head) setText($('gaze-val'), lookWords(A.head));
      } else {
        $('gaze-read').hidden = !(run.cone && liveInfo && gz !== undefined);
        if (run.cone && gz !== undefined) setText($('gaze-val'), gazeWords(gz));
      }
    }
    const ls = liveInfo ? liveInfo.ls : { touches: 0, shuttles: 0 };
    setText($('touch-val'), String(ls.touches));
    setText($('shuttle-val'), String(ls.shuttles));
    const rt = $('radio-text');
    if (run.isAuto) {
      setText(rt, 'Privileged planner (full map, sampling MPC). Sets the reference pace for each course.');
      rt.className = 'radio-text quiet';
      setText($('radio-eyebrow'), 'Driver radio');
      setHTML($('radio-cmd'), '');
    } else {
      const c = liveInfo ? callAt(liveInfo.leg, liveInfo.tl) : null;
      const txt = c?.text?.trim(), th = c?.thought?.trim();
      if (c?.name === 'run_program' || (c?.args?.code && (c.name === 'set_controller' || c.name === 'keep_driving'))) {
        setHTML(rt, (th ? `<span class="thought-title">${escHTML(clip(thoughtTitle(th), 70))}</span>` : '') + programHTML(c, liveInfo.tl));
        rt.className = 'radio-text program';
        setText($('radio-eyebrow'), c.name === 'run_program' ? `Program · turn ${c.turn ?? '?'}` : `Controller · turn ${c.turn ?? '?'}${c.kept ? ' · still driving' : ''}`);
      } else {
        if (txt) setText(rt, txt);
        else if (th) setHTML(rt, thoughtHTML(th));
        else setText(rt, c ? '(no message; drove silently)' : run.legs.length ? 'Reading the briefing…' : 'Waiting for the first report…');
        rt.className = `radio-text${!txt && !th ? ' quiet' : ''}${!txt && th ? ' thought' : ''}`;
        setText($('radio-eyebrow'), !txt && th ? 'Thinking · model summary' : 'Driver radio');
      }
      setHTML($('radio-cmd'), resultLine(c));
    }
    if (liveInfo) {
      const { p, st: s } = liveInfo;
      const lg = liveInfo.leg;
      focusInfo.pos.set((s.x + p.ax) / 2, 0, -(s.y + p.ay) / 2);
      focusInfo.front.set(s.x + lg.spec.body_front * Math.cos(s.th1), 0, -(s.y + lg.spec.body_front * Math.sin(s.th1)));
      focusInfo.rear.set(p.ax - lg.spec.trailer_rear * Math.cos(s.th2), 0, -(p.ay - lg.spec.trailer_rear * Math.sin(s.th2)));
      focusInfo.C = lg.spec.combo_length;
      const g = state.tracks[lg.i].grp;
      g.localToWorld(focusInfo.pos); g.localToWorld(focusInfo.front); g.localToWorld(focusInfo.rear);
    }
  }

  updateCamera(dtReal);
  for (const tk of state.tracks) {
    tk.baySign.getWorldPosition(tmpV);
    tk.baySign.visible = tmpV.distanceTo(camera.position) < 90 && state.cam !== 'overview' && !CLEAN;
    // course name boards are for the bird's-eye views; up close they crowd the top of the screen
    tk.sign.visible = (state.cam === 'overview' || state.cam === 'free') && !CLEAN;
    for (const z of tk.zoneSigns) z.visible = state.cam !== 'overview' && !CLEAN;
  }

  // in-scene chips, projected through the camera as it is this frame
  if (broadcast) {
    camera.updateMatrixWorld();
    const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
    const proj = (ti, x, y, z) => {
      state.tracks[ti].grp.localToWorld(tmpV.set(x, z, -y)).project(camera);
      return tmpV.z > 1 ? [-1e4, -1e4] : [(tmpV.x * 0.5 + 0.5) * w, (-tmpV.y * 0.5 + 0.5) * h];
    };
    const list = cands.map((c) => {
      const a = proj(c.ti, c.at[0], c.at[1], c.at[2] ?? 0);
      return { ...c, anchor: a, ring: c.ring ? { cls: c.ring, at: a } : null, prio: c.prio + (c.focus ? 0.5 : 0) };
    });
    // labels lift off the rig: rings sit on the hitch/contact point, the callout stands above it
    // declutter: full callouts only for the focused run, the leader and the two most urgent other runs; the rest wear a
    // compact tag (state + "Sol #2"), and with a crowded screen the plain name tags of the others go
    const leader = standings[0]?.runId;
    const onScreen = list.filter((c) => c.anchor[0] > 0 && c.anchor[0] < w && c.anchor[1] > 0 && c.anchor[1] < h).length;
    const urgent = new Set();
    list.sort((a, b) => b.prio - a.prio);
    mergeStacked(list);
    for (const c of list) {
      const lead = c.info?.run === leader;
      if (c.mergedInto) continue; // folded into its group's callout: no label, no ring
      if (c.merged) { /* a merged group keeps its summary callout */ }
      else if (!c.focus && !lead && c.kind !== 'FLAG') {
        // the two most urgent other runs keep all their callouts in full
        if (c.prio >= 5 && (urgent.has(c.info?.run) || urgent.size < 2)) urgent.add(c.info?.run);
        else { c.html = c.short; c.cls += ' compact'; }
      }
      if (c.kind === 'FLAG' && !c.focus && onScreen > 8) c.html = null;
      if (c.ring) c.anchor = [c.anchor[0], c.anchor[1] - 10];
    }
    // course boards (overview) are part of the picture: labels keep off them too
    const occ = hudRects();
    for (const tk of state.tracks) {
      for (const sg of [tk.sign, tk.baySign]) {
        if (!sg.visible) continue;
        sg.getWorldPosition(tmpV).project(camera);
        if (tmpV.z > 1) continue;
        const sh = sg.scale.y * h * 1.5, sw = sh * sg.scale.x / sg.scale.y;
        const px = (tmpV.x * 0.5 + 0.5) * w, py = (-tmpV.y * 0.5 + 0.5) * h;
        occ.push({ x: px - sw / 2, y: py - sh, w: sw, h: sh });
      }
    }
    chips.render(list, occ, w, h);
  }

  // Grid / Inspect redraw at ~12 Hz
  const now = performance.now();
  if (!broadcast && state.view !== 'results' && now - lastPanel > 80) {
    lastPanel = now;
    const ctx = panelCtx();
    if (state.view === 'grid') grid.render(ctx);
    else inspect.render(ctx);
  }
}

function panelCtx() {
  return {
    runs: state.runs, T: state.T, focus: state.focus, heats: state.clockMode === 'heats', heatLeg: state.heatLeg, nLegs: state.tracks.length,
    rotOf: (i) => state.tracks[i]?.rot ?? 0, shortLabel, setText, setChip, progressOf, realtime: isRealtime, resultLine,
  };
}

// ------------------------------------------------------------------ cameras
const camGoal = { pos: new THREE.Vector3(), target: new THREE.Vector3() };
let camSnap = true;
const _ovCam = new THREE.PerspectiveCamera();
// candidate screen areas the HUD leaves free (from the cards' real rects); the overview uses whichever frames the layout largest
function freeAreas() {
  const W = window.innerWidth, H = window.innerHeight;
  const R = (id) => { const e = $(id); if (!e || e.hidden) return null; const r = e.getBoundingClientRect(); return r.width ? r : null; };
  const nav = document.querySelector('.topnav').getBoundingClientRect(), dock = document.querySelector('.dock').getBoundingClientRect();
  const clock = R('clock'), board = R('board'), lower = R('radio'), rp = state.pickerOpen ? R('runs-panel') : null;
  const left = rp ? rp.right + 12 : 12, bottom = dock.top - 10;
  const topAll = Math.max(nav.bottom, clock?.bottom ?? 0, board?.bottom ?? 0) + 10;
  const out = [
    { x0: left, x1: W - 12, y0: topAll, y1: Math.min(bottom, (lower?.top ?? bottom) - 10) }, // band between the top cards and the lower-third
    { x0: Math.max(left, (clock?.right ?? 0) + 12), x1: (board?.left ?? W) - 12, y0: nav.bottom + 8, y1: (lower?.top ?? bottom) - 10 }, // centre column
    { x0: Math.max(left, (lower?.right ?? 0) + 12), x1: W - 12, y0: (board?.bottom ?? nav.bottom) + 10, y1: bottom }, // right of the lower-third
  ];
  return out.filter((f) => f.x1 - f.x0 > 160 && f.y1 - f.y0 > 120);
}
function overviewGoal() {
  // fit the layout's bounding box into a screen area the HUD leaves free, by projecting its corners
  const { w, d } = state.layoutSize;
  const W = window.innerWidth, H = window.innerHeight;
  const pitch = 56 * Math.PI / 180;
  const corners = [];
  for (const sx of [-0.5, 0.5]) for (const sz of [-0.5, 0.5]) for (const h of [0, 3]) corners.push(new THREE.Vector3(sx * w, h, sz * d).add(state.layoutCenter));
  _ovCam.fov = camera.fov; _ovCam.aspect = camera.aspect; _ovCam.near = 0.5; _ovCam.far = 5000; _ovCam.updateProjectionMatrix();
  const v = new THREE.Vector3();
  let best = null;
  const areas = freeAreas();
  for (const free of areas.length ? areas : [{ x0: 0, x1: W, y0: 0, y1: H }]) {
    const target = state.layoutCenter.clone();
    let dist = Math.max(w, d) * 1.2 + 20;
    for (let it = 0; it < 7; it++) {
      _ovCam.position.copy(target).add(v.set(0, Math.sin(pitch) * dist, Math.cos(pitch) * dist));
      _ovCam.lookAt(target);
      _ovCam.updateMatrixWorld();
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const c of corners) {
        v.copy(c).project(_ovCam);
        const px = (v.x * 0.5 + 0.5) * W, py = (-v.y * 0.5 + 0.5) * H;
        x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
      }
      const k = Math.max((x1 - x0) / (free.x1 - free.x0), (y1 - y0) / (free.y1 - free.y0));
      // re-centre: move the target so the box centre lands on the free-area centre
      const mPerPx = 2 * dist * Math.tan(camera.fov * Math.PI / 360) / H;
      const dxPx = (x0 + x1) / 2 - (free.x0 + free.x1) / 2, dyPx = (y0 + y1) / 2 - (free.y0 + free.y1) / 2;
      target.x += dxPx * mPerPx;
      target.z += dyPx * mPerPx / Math.sin(pitch);
      dist *= 0.35 + 0.65 * k;
    }
    if (!best || dist < best.dist) best = { dist, target };
  }
  camGoal.target.copy(best.target);
  camGoal.pos.copy(best.target).add(v.set(0, Math.sin(pitch) * best.dist, Math.cos(pitch) * best.dist));
}
function updateCamera(dt) {
  if (state.cam === 'free') { setShadowFrame(controls.target, Math.max(30, camera.position.distanceTo(controls.target) * 0.6)); return; }
  const C = focusInfo.C;
  if (state.cam === 'overview') {
    overviewGoal();
  } else if (state.cam === 'follow') {
    const d = Math.max(11, C * 1.55 + 7);
    camGoal.target.copy(focusInfo.pos);
    camGoal.pos.copy(focusInfo.pos).add(tmpV.set(0.42, 0.78, 0.62).normalize().multiplyScalar(d));
  } else if (state.cam === 'chase') {
    // over the cab, looking back down the rig the way the driver does when reversing
    const dir = tmpV.copy(focusInfo.front).sub(focusInfo.rear).setY(0);
    const len = dir.length() || 1;
    dir.divideScalar(len);
    camGoal.pos.copy(focusInfo.front).addScaledVector(dir, C * 0.32 + 2.5).setY(C * 0.55 + 3.5);
    camGoal.target.copy(focusInfo.rear).addScaledVector(dir, -C * 0.45);
  }
  const k = camSnap ? 1 : 1 - Math.exp(-dt * 2.6);
  camera.position.lerp(camGoal.pos, k);
  controls.target.lerp(camGoal.target, k);
  camSnap = false;
  const shadowR = state.cam === 'overview' ? state.layoutRadius + 20 : Math.max(18, C * 1.6 + 10);
  setShadowFrame(state.cam === 'overview' ? state.layoutCenter : controls.target, shadowR);
}

function setCam(c) { state.cam = c; camSnap = false; syncButtons(); }
function setFocus(k) { state.focus = k; state.dirtyTicks = true; syncButtons(); }
function setPlaying(p) { state.playing = p; if (!p) state.followLive = false; if (p && !state.anyLive && state.T >= state.Tmax - 1e-6) state.T = 0; syncButtons(); }
function seek(T) { state.T = Math.max(0, Math.min(state.Tmax, T)); state.followLive = state.anyLive && state.T >= state.Tmax - 0.5; }

// Relay clock (legs back to back) vs heats (every leg starts at 0:00). Keep the focused rig where it is.
function setClockMode(mode, user = false) {
  if (user) state.clockUser = true;
  const run = state.runs[state.focus];
  if (mode !== state.clockMode && run) {
    if (mode === 'heats') {
      const { active, leg } = runAt(run, state.T, state.heatLeg);
      if (leg) { state.heatLeg = active; state.T = Math.max(0, state.T - leg.start); }
    } else {
      const leg = run.legs[Math.min(state.heatLeg, run.legs.length - 1)];
      if (leg) state.T = leg.relayStart + Math.min(state.T, leg.time);
    }
  }
  state.clockMode = mode;
  for (const r of state.runs) applyClock(r, mode);
  state.Tmax = Math.max(1, ...state.runs.map((r) => r.total));
  state.T = Math.min(state.T, state.Tmax);
  state.dirtyTicks = true;
  camSnap = false;
  syncButtons();
  writeURL();
}
function setHeatLeg(i) {
  state.heatLeg = Math.max(0, Math.min(state.tracks.length - 1, i));
  state.dirtyTicks = true;
  syncButtons();
  writeURL();
}
function goLive() { state.followLive = true; state.T = state.Tmax; setPlaying(true); }

controls.addEventListener('start', () => { if (state.cam !== 'free') { state.cam = 'free'; syncButtons(); } });

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'Escape') { if (state.compareOpen) setCompare(false); else if (state.pickerOpen) setPicker(false); else if (state.view === 'grid' || state.view === 'inspect') setView('broadcast'); return; }
  if (e.key === 'r' || e.key === 'R') { setView('results'); return; }
  if (state.view === 'results') {
    // the Results page scrolls; only the view keys work here
    if (e.key === 'b' || e.key === 'B') setView('broadcast');
    else if (e.key === 'g' || e.key === 'G') setView('grid');
    else if (e.key === 'i' || e.key === 'I') setView('inspect');
    else if (e.key === 'p' || e.key === 'P') { setView('broadcast'); setPicker(true); }
    return;
  }
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.key === 'ArrowRight') seek(state.T + (e.shiftKey ? 30 : 5));
  else if (e.key === 'ArrowLeft') seek(state.T - (e.shiftKey ? 30 : 5));
  else if (/^[1-5]$/.test(e.key)) { state.speed = SPEEDS[Number(e.key) - 1]; syncButtons(); }
  else if (e.key === 'b' || e.key === 'B') setView('broadcast');
  else if (e.key === 'g' || e.key === 'G') setView('grid');
  else if (e.key === 'i' || e.key === 'I') setView('inspect');
  else if (e.key === 'f' || e.key === 'F') { setView('broadcast'); setCam('follow'); }
  else if (e.key === 'c' || e.key === 'C') setCam('chase');
  else if (e.key === 'o' || e.key === 'O') setCam('overview');
  else if (e.key === 'p' || e.key === 'P') setPicker(!state.pickerOpen);
  else if (e.key === 't' || e.key === 'T') setCompare(!state.compareOpen);
  else if (e.key === 'h' || e.key === 'H') setClockMode(state.clockMode === 'heats' ? 'relay' : 'heats', true);
  else if (e.key === ']') setFocus((state.focus + 1) % state.runs.length);
  else if (e.key === '[') setFocus((state.focus + state.runs.length - 1) % state.runs.length);
});

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const run of state.runs) for (const a of run.actors || []) a.trail.mat.resolution.set(w, h);
  for (const t of state.tracks) t.parked?.userData.mat?.resolution.set(w, h);
  if (state.view === 'grid') grid?.reset();
}
window.addEventListener('resize', resize);

// ------------------------------------------------------------------ views: Broadcast (3D) · Grid (small multiples) · Inspect (one run)
function setView(v, { focus } = {}) {
  if (!VIEWS.includes(v)) return state.view;
  if (v !== 'results' && state.lazy) ensureScene(); // first visit to a 3D / map view: load the runs now
  if (focus !== undefined && focus !== state.focus) setFocus(focus);
  state.view = v;
  document.body.dataset.view = v;
  $('grid-view').hidden = v !== 'grid';
  $('inspect-view').hidden = v !== 'inspect';
  $('results-view').hidden = v !== 'results';
  if (v === 'results') { results?.show(params.get('sec')); if (state.pickerOpen) setPicker(false); if (state.compareOpen) setCompare(false); }
  if (v !== 'broadcast') chips?.clear();
  else camSnap = true;
  showHint();
  lastPanel = 0;
  grid?.reset(); inspect?.reset();
  syncButtons();
  writeURL();
  return v;
}
function initViews() {
  for (const b of $('views').querySelectorAll('[data-view]')) b.onclick = () => setView(b.dataset.view);
  results = createResults($('results-view'), {
    fetchJSON, runsDir: RUNS_DIR, static: STATIC, openRuns, getIndex: () => state.index, indexSig: () => String(state.indexTxt?.length ?? 0),
    onSection: (sec) => { state.section = sec; writeURL(); }, onState: () => writeURL(),
  });
  $('to-inspect').onclick = () => setView('inspect');
  $('board-more').onclick = () => setView('grid');
  chips = createChips($('chips'));
  grid = createGrid($('grid-tiles'), { onOpen: (k) => setView('inspect', { focus: k }) });
  inspect = createInspect({
    swatch: $('insp-swatch'), name: $('insp-name'), vari: $('insp-var'), chips: $('insp-chips'), zoom: $('insp-zoom'), run: $('insp-run'),
    svg: $('insp-svg'), legend: $('insp-legend'), logTitle: $('insp-log-title'), log: $('insp-log'), charts: $('insp-charts'), notes: $('insp-notes'),
  }, { onPick: (k) => { setFocus(k); inspect.reset(); lastPanel = 0; } });
}

// ------------------------------------------------------------------ loading, selection, live polling
// A run is identified by its run_id, which survives the move from runs/live/<id>.json to runs/<id>.json.
const runIdOf = (it) => it.run_id || it.file.replace(/^live\//, '').replace(/\.json$/, '');
const simvOf = (r) => r.config?.sim_version ?? 1;
const isAutoItem = (it) => it.driver?.spec === 'autopilot';
const studyTag = (L) => {
  const s = String(L.study || '');
  return `${CONDITION_NAMES[L.condition] || L.condition} · ${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(9, 11)}:${s.slice(11, 13)}`;
};
// variantOf().control is 'discrete', 'continuous · paused' or 'continuous · realtime'
const ctrlName = (c) => (String(c).startsWith('continuous') ? (String(c).endsWith('realtime') ? `${CONTROL_NAMES.continuous} · ${CLOCK_NAMES.realtime.toLowerCase()}` : CONTROL_NAMES.continuous) : CONTROL_NAMES[c] || c);
// picker facets: the chip text is the value (and the f_<key> URL value), in the readers' vocabulary
const FACETS = [
  ['model', 'Model', (v, it) => modelName(describe(it))],
  ['level', 'Difficulty', (v, it) => { const d = describe(it); return d.obs === 'driver' ? LEVEL_SHORT[d.level || 'v1'] : 'Not driver\'s seat'; }],
  ['study', 'Notes study', (v, it) => (it.learning ? studyTag(it.learning) : 'No notes')],
  ['prompt', 'Prompt', (v) => `Prompt version ${v.prompt}`],
  ['obs', 'Driver sees', (v) => OBS_NAMES[v.obs] || v.obs],
  ['mode', 'How it drives', (v) => (MODE_NAMES[v.mode] || v.mode) + (v.route === 'api' || v.route === 'local' ? '' : ` · via ${v.route}`)],
  ['control', 'Between decisions', (v) => ctrlName(v.control)],
];
const facetVal = (it, key) => FACETS.find((f) => f[0] === key)[2](variantOf(it), it);

function regroup() {
  const groups = {};
  for (const r of state.index.runs || []) (groups[r.relay.id] ||= []).push(r);
  state.groups = groups;
  state.groupIds = Object.keys(groups).sort((a, b) => {
    const last = (g) => g.map((r) => r.updated ? new Date(r.updated * 1000).toISOString() : r.created || '').sort().pop() || '';
    return last(groups[b]).localeCompare(last(groups[a]));
  });
  const sel = $('relay-select');
  const sig = state.groupIds.map((x) => `${x}:${groups[x].length}`).join('|');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.innerHTML = '';
    for (const x of state.groupIds) {
      const o = document.createElement('option');
      const g = groups[x];
      o.value = x; o.textContent = `${relayName(g[0].relay.name)} · ${courseSet(g[0].relay.seed)} · ${g.length} run${g.length === 1 ? '' : 's'}`;
      sel.appendChild(o);
    }
  }
  sel.value = state.relayId || '';
  $('picker').hidden = state.groupIds.length < 2;
}

// the runs of a relay, newest physics only, a finished record superseding its live file
function groupItems(relayId) {
  const all = state.groups[relayId] || [];
  if (!all.length) return [];
  const physics = params.has('physics') ? Number(params.get('physics')) : Math.max(...all.map(simvOf));
  let items = all.filter((r) => simvOf(r) === physics);
  if (!items.length) items = all;
  const by = new Map();
  for (const it of items) {
    const id = runIdOf(it), prev = by.get(id);
    if (!prev || (prev.live && !it.live)) by.set(id, it);
  }
  items = [...by.values()];
  items.sort((a, b) => isAutoItem(a) - isAutoItem(b) || a.driver.label.localeCompare(b.driver.label)
    || (a.config?.prompt_version ?? 1) - (b.config?.prompt_version ?? 1) || String(variantOf(a).obs).localeCompare(variantOf(b).obs)
    || variantOf(a).mode.localeCompare(variantOf(b).mode) || (a.driver.trial ?? 0) - (b.driver.trial ?? 0) || (a.created || '').localeCompare(b.created || ''));
  return items;
}

const labelOf = (it) => { const l = (state.itemLabels || new Map()).get(runIdOf(it)); return l || null; };
const matchesQuery = (it) => {
  const q = state.q.trim().toLowerCase();
  if (!q) return true;
  const l = labelOf(it);
  return q.split(/\s+/).every((w) => `${l?.text || ''} ${l?.full || ''} ${runIdOf(it)} ${it.driver?.spec || ''} ${it.learning?.study || ''}`.toLowerCase().includes(w));
};
const passesFilters = (it) => FACETS.every(([key]) => !state.filters[key].size || state.filters[key].has(facetVal(it, key)));
const isSelected = (it) => !state.sel || state.sel.has(runIdOf(it));
const isShown = (it) => isSelected(it) && passesFilters(it);

// Colours: a hue per model, a shade per run of that model. Assigned over the runs that pass the facet filters
// (the comparison being made), so ticking runs on and off never repaints the others.
function colorFor(items, it) {
  const sig = items.map(runIdOf).join('|') + FACETS.map(([k]) => [...state.filters[k]].join(',')).join(';');
  if (state.colorSig !== sig) {
    state.colorSig = sig;
    const entry = (x) => { const d = describe(x); return { id: runIdOf(x), model: modelShort(d), base: modelColor(d), auto: isAutoItem(x) }; };
    const m = assignColors(items.filter(passesFilters).map(entry));
    const rest = assignColors(items.map(entry));
    for (const [k, v] of rest) if (!m.has(k)) m.set(k, v);
    state.colorMap = m;
  }
  return state.colorMap.get(runIdOf(it)) || '#e9e4d8';
}

function selectionFromURL(items) {
  const raw = params.get('runs');
  const only = params.get('only');
  if (raw && !raw.endsWith('/')) {
    const want = raw.split(',').map((x) => x.trim()).filter(Boolean);
    const hit = (it) => want.some((w) => w === runIdOf(it) || w === it.file || w === it.file.replace(/\.json$/, '') || it.file.endsWith(`/${w}`));
    return new Set(items.filter(hit).map(runIdOf).concat(want.filter((w) => !items.some((it) => runIdOf(it) === w))));
  }
  if (only) return new Set(items.filter((r) => only.split(',').some((w) => r.driver.spec.includes(w) || r.file.includes(w))).map(runIdOf));
  return null;
}

function writeURL() {
  const u = new URL(location.href);
  if (state.view === 'results') u.searchParams.set('view', 'results');
  if (state.view === 'results' && state.section && state.section !== 'overview') u.searchParams.set('sec', state.section); else u.searchParams.delete('sec');
  // the notes reader's own state (study, compare, leg, driver) is part of a Results link
  const rp = state.view === 'results' ? results?.urlParams?.() || {} : {};
  for (const k of ['study', 'compare', 'driver']) if (rp[k] != null) u.searchParams.set(k, rp[k]); else u.searchParams.delete(k);
  const done = () => { if (rp.leg != null) u.searchParams.set('leg', rp.leg); try { history.replaceState(null, '', u.toString().replace(/=(?=&|$)/g, '')); } catch (e) { /* sandboxed */ } };
  if (!state.relayId) { if (rp.leg == null && !LEG_PARAM) u.searchParams.delete('leg'); done(); return; }
  u.searchParams.set('relay', state.relayId);
  u.searchParams.delete('only');
  if (state.sel) u.searchParams.set('runs', [...state.sel].join(',')); else if (!params.get('runs')?.endsWith('/')) u.searchParams.delete('runs');
  for (const [key] of FACETS) {
    if (state.filters[key].size) u.searchParams.set(`f_${key}`, [...state.filters[key]].join(',')); else u.searchParams.delete(`f_${key}`);
  }
  if (state.clockUser || params.has('clock')) u.searchParams.set('clock', state.clockMode);
  if (state.clockMode === 'heats') u.searchParams.set('leg', String(state.heatLeg + 1)); else u.searchParams.delete('leg');
  if (state.pickerOpen) u.searchParams.set('picker', ''); else u.searchParams.delete('picker');
  if (state.compareOpen) u.searchParams.set('compare', ''); else u.searchParams.delete('compare');
  if (state.view !== 'broadcast') u.searchParams.set('view', state.view); else u.searchParams.delete('view');
  done();
}

function buildWorld(relay) {
  world.clear();
  state.relay = relay;
  state.tracks = relay.tracks.map((tr, i) => buildTrack(tr, i, relay.tracks.length));
  for (const t of state.tracks) world.add(t.grp);
  const ext = layoutTracks(state.tracks);
  const scenery = buildScenery(state.tracks, LAYOUT_GAP);
  world.add(scenery.group);
  state.scenery = scenery.group;
  plantTrees(state.tracks, ext, scenery.rects);
  state.heatLeg = Math.min(state.heatLeg, state.tracks.length - 1);
  state.worldFor = relay.id;
  camSnap = true;
}

// tiles are narrow: drop the vendor prefix ("GPT-6.1 Sol" -> "6.1 Sol"); run names come from labels.js
const shortLabel = (s) => s.replace(/^GPT-/, '');

// one name per run, from labels.js ("GPT-6.1 Sol · Driver's seat, Medium · Course set B · run 2"); twins in the same set get their start time
function trialTags(runs) {
  const labs = uniqueLabels(runs.map((r) => r.rec));
  runs.forEach((r, i) => {
    const l = labs[i];
    r.lab = l;
    r.name = l.text + (r.rec.partial && !r.isLive ? ' (partial)' : '');
    r.mini = l.mini;
    r.variantTag = r.isAuto ? 'reference planner' : l.tag;
    r.trialTag = '';
  });
}

// fetch what changed, then rebuild runs/actors/HUD, keeping time, focus and camera
async function syncRuns({ force = false } = {}) {
  const items = groupItems(state.relayId);
  state.items = items;
  const shown = items.filter(isShown);
  const stale = shown.filter((it) => {
    const p = state.pool.get(runIdOf(it));
    return !p || p.file !== it.file || (it.updated ?? 0) !== p.updated;
  });
  await Promise.all(stale.map(async (it) => {
    try {
      const rec = await fetchJSON(RUNS_DIR + it.file);
      state.pool.set(runIdOf(it), { rec, file: it.file, updated: it.updated ?? 0 });
    } catch (e) {
      // a live file can vanish between the index and the fetch (the run just finished): keep what we had
      console.warn(`skip ${it.file}: ${e.message}`);
    }
  }));
  const sig = shown.filter((it) => state.pool.has(runIdOf(it))).map((it) => `${runIdOf(it)}@${state.pool.get(runIdOf(it)).file}@${state.pool.get(runIdOf(it)).updated}`).join('|');
  renderPicker();
  if (state.compareOpen) renderCompare();
  if (!force && sig === state.runsSig) return false;
  state.runsSig = sig;

  // remember the view
  const focusId = state.runs[state.focus]?.runId ?? null;
  const atEdge = state.anyLive && (state.followLive || (state.playing && state.T >= state.Tmax - 0.75));
  const firstBuild = !state.built;

  disposeActors();
  const recs = shown.filter((it) => state.pool.has(runIdOf(it))).map((it) => ({ it, ...state.pool.get(runIdOf(it)) }));
  const relay = recs[0]?.rec.relay || items[0]?.relay;
  if (recs.length && state.worldFor !== recs[0].rec.relay.id) buildWorld(recs[0].rec.relay);
  state.runs = recs.map(({ it, rec, file }) => prepareRun(rec, colorFor(items, it), file));
  trialTags(state.runs);
  state.anyLive = state.runs.some((r) => r.isLive);
  if (!state.clockUser && (firstBuild || state.anyLive !== state.wasLive)) {
    state.clockMode = params.get('clock') === 'heats' || params.get('clock') === 'relay' ? params.get('clock') : state.anyLive || LEG_PARAM ? 'heats' : 'relay';
    if (params.has('clock')) state.clockUser = true;
  }
  state.wasLive = state.anyLive;
  for (const r of state.runs) applyClock(r, state.clockMode);
  state.Tmax = Math.max(1, ...state.runs.map((r) => r.total));
  if (firstBuild) {
    const fq = params.get('focus');
    state.focus = Math.max(0, fq !== null ? state.runs.findIndex((r) => r.label.toLowerCase().includes(fq.toLowerCase()) || r.spec.includes(fq)) : state.runs.findIndex((r) => !r.isAuto));
    if (params.has('t')) state.T = Number(params.get('t'));
    else if (state.anyLive && !params.has('paused')) { state.T = state.Tmax; state.followLive = true; }
  } else {
    const k = state.runs.findIndex((r) => r.runId === focusId);
    state.focus = k >= 0 ? k : Math.max(0, state.runs.findIndex((r) => !r.isAuto));
    if (atEdge && state.anyLive) state.T = state.Tmax;
  }
  state.T = Math.min(Math.max(0, state.T), state.Tmax);
  state.built = true;

  if (state.runs.length) {
    buildActors();
  }
  buildHud();
  resize();
  $('radio').hidden = !state.runs.length;
  if (relay) {
    const names = relay.tracks ? relay.tracks.map((t) => t.vehicle).join(' → ') : '';
    setText($('relay-title'), `${relayName(relay.name)}${names ? ` · ${names}` : ''}`);
    const labels = [...new Set(state.runs.map((x) => x.lab?.model || x.label))];
    const nl = state.relay?.tracks.length ?? 0;
    const sub = !state.runs.length ? 'No runs selected — open Runs to pick some'
      : `${state.runs.length} run${state.runs.length === 1 ? '' : 's'} on ${nl === 1 ? 'one course' : `the same ${nl} courses`} (${courseSet(relay.seed)})${state.anyLive ? ', some still driving' : ''} · ${labels.join(' vs ')}`;
    setText($('relay-sub'), sub);
  }
  state.dirtyTicks = true;
  if (firstBuild) camSnap = true;
  return true;
}

async function switchRelay(id, sel = undefined) {
  state.relayId = id;
  for (const [key] of FACETS) state.filters[key].clear();
  state.sel = sel !== undefined ? sel : defaultSelection(groupItems(id));
  state.built = false;
  state.runsSig = null;
  disposeActors();
  state.runs = [];
  const items = groupItems(id);
  if (items[0]) {
    const rec0 = await fetchJSON(RUNS_DIR + items[0].file).catch(() => null);
    if (rec0) { state.pool.set(runIdOf(items[0]), { rec: rec0, file: items[0].file, updated: items[0].updated ?? 0 }); buildWorld(rec0.relay); }
  }
  await syncRuns({ force: true });
  writeURL();
}

// With many runs on one course set, Broadcast opens on the newest experiment only (at most 6 runs): the newest
// learning study, or the newest runs of the same observation / level / control started within three hours.
const DEFAULT_MAX = 6;
function defaultSelection(items) {
  state.hint = null;
  if (items.length <= 10) return null;
  const by = items.slice().sort((a, b) => (b.created || '').localeCompare(a.created || ''));
  const top = by[0], d0 = describe(top), t0 = Date.parse(top.created || '') || 0;
  const same = by.filter((it) => {
    const d = describe(it);
    if (top.learning) return it.learning?.study === top.learning.study;
    return !it.learning && d.obs === d0.obs && d.level === d0.level && d.control === d0.control && d.mode === d0.mode
      && Math.abs(t0 - (Date.parse(it.created || '') || 0)) < 3 * 3600e3;
  });
  const pick = same.slice(0, DEFAULT_MAX);
  const what = top.learning ? `notes study (${(CONDITION_NAMES[top.learning.condition] || top.learning.condition).toLowerCase()})` : `${obsName(d0.obs, d0.obs === 'driver' ? d0.level || 'v1' : null)} runs`;
  state.hint = { n: pick.length, of: items.length, what };
  return new Set(pick.map(runIdOf));
}
function showHint() {
  const h = $('bc-hint');
  const on = !!state.hint && state.view === 'broadcast' && !state.lazy;
  h.hidden = !on;
  if (on) h.innerHTML = `Showing the newest ${escHTML(state.hint.what)}: <b>${state.hint.n} of ${state.hint.of}</b> runs on these courses · <button class="lnk" id="bc-pick">choose runs (P)</button> · <button class="lnk" id="bc-res">all results (R)</button>`;
  if (on) { $('bc-pick').onclick = () => setPicker(true); $('bc-res').onclick = () => setView('results'); }
}

function toast(msg) {
  let t = $('toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._h); t._h = setTimeout(() => { t.hidden = true; }, 5000);
}

// load the first course set and its runs (deferred while only Results has been shown)
let scenePromise = null;
function ensureScene() {
  if (!state.lazy && scenePromise) return scenePromise;
  state.lazy = false;
  scenePromise ||= initScene();
  return scenePromise;
}
async function initScene() {
  regroup();
  const ids = state.groupIds;
  if (!ids.length) { $('empty').hidden = false; $('radio').hidden = true; return; }
  const want = params.get('relay');
  let id = ids.find((x) => x === want || x.startsWith(want || '\0'));
  // ?runs=<id>,… without ?relay=: the course set that holds most of the named runs
  const named = (params.get('runs') || '').endsWith('/') ? [] : (params.get('runs') || '').split(',').map((x) => x.trim()).filter(Boolean);
  if (!id && named.length) {
    let bn = 0;
    for (const g of ids) {
      const n = groupItems(g).filter((it) => named.some((w) => w === runIdOf(it) || w === it.file || it.file.endsWith(`/${w}`))).length;
      if (n > bn) { bn = n; id = g; }
    }
  }
  id ||= ids[0];
  state.relayId = id;
  regroup();
  state.items = groupItems(id);
  state.sel = selectionFromURL(state.items) ?? defaultSelection(state.items);
  await syncRuns({ force: true });
  if (!state.runs.length && state.items[0]) {
    const it = state.items[0];
    const rec0 = await fetchJSON(RUNS_DIR + it.file).catch(() => null);
    if (rec0) buildWorld(rec0.relay);
  }
  showHint();
}

// open exactly these runs (from Results): the course set holding most of them, in the given view
async function openRuns(ids, view = 'broadcast', opts = {}) {
  ids = [...new Set(ids)].filter(Boolean);
  if (!ids.length) return;
  await ensureScene();
  regroup();
  let best = null, bn = 0;
  for (const g of state.groupIds) {
    const n = groupItems(g).filter((it) => ids.includes(runIdOf(it))).length;
    if (n > bn) { bn = n; best = g; }
  }
  if (!best) { toast('These runs are not in the run list (an older simulator version, or still being written).'); return; }
  const inGroup = new Set(groupItems(best).map(runIdOf).filter((x) => ids.includes(x)));
  state.hint = null;
  if (best !== state.relayId) await switchRelay(best, inGroup);
  else { for (const [key] of FACETS) state.filters[key].clear(); state.sel = inGroup; await syncRuns({ force: true }); }
  state.hint = null;
  if (bn < ids.length) toast(`Showing ${bn} of ${ids.length} runs: the others drove other courses (${relayName(state.relay?.name)}, ${courseSet(state.relay?.seed)} shown). Open those from Results separately.`);
  const k = Math.max(0, state.runs.findIndex((r) => r.runId === ids[0]));
  setView(view, { focus: k });
  if (opts.leg != null && state.tracks[opts.leg]) {
    setClockMode('heats', true); setHeatLeg(opts.leg);
    const L = state.runs[k]?.legs.find((l) => l.i === opts.leg);
    seek(L ? L.time : 0);
    setPlaying(false);
  }
  showHint();
  writeURL();
  settle();
}

let pollBusy = false;
async function poll() {
  if (pollBusy) return false;
  pollBusy = true;
  try {
    const idx = await fetchJSON(RUNS_DIR + 'index.json');
    state.polls++;
    if (results && state.view === 'results' && state.polls % 4 === 0) results.refresh();
    const txt = JSON.stringify(idx);
    if (txt === state.indexTxt) return false;
    state.indexTxt = txt;
    state.index = idx;
    regroup();
    if (state.lazy) return false;
    if (!state.relayId || !state.groups[state.relayId]) {
      if (!state.groupIds.length) return false;
      $('empty').hidden = true;
      await switchRelay(state.groupIds[0]);
      return true;
    }
    return await syncRuns();
  } catch (e) {
    console.warn('poll failed', e.message);
    return false;
  } finally {
    pollBusy = false;
  }
}

// ---------- run picker panel
function variantTags(it) {
  const v = variantOf(it);
  const t = [];
  const auto = isAutoItem(it); // the reference driver has no prompt, harness or observation level
  if (auto) t.push('<span class="vt">reference</span>');
  else {
    const d = describe(it);
    if (it.learning) t.push(`<span class="vt learn">${escHTML(learningTag(it.learning))}</span>`);
    t.push(`<span class="vt obs">${escHTML(obsName(v.obs, v.obs === 'driver' ? d.level || 'v1' : null))}</span>`);
    t.push(`<span class="vt mode">${escHTML(MODE_NAMES[v.mode] || v.mode)}</span>`);
    t.push(`<span class="vt mode">${escHTML(ctrlName(v.control))}</span>`);
    if (v.route !== 'api' && v.route !== 'local') t.push(`<span class="vt route">via ${escHTML(v.route)}</span>`);
    t.push(`<span class="vt" title="prompt version ${v.prompt}${v.harness !== 1 ? `, harness ${v.harness}` : ''}">prompt v${v.prompt}</span>`);
  }
  if (it.live) t.push('<span class="vt live">driving now</span>');
  else if (it.partial) t.push('<span class="vt partial">unfinished</span>');
  return t.join('');
}
function itemResult(it) {
  const T = it.totals || {};
  const n = T.legs ?? it.legs?.length ?? 0;
  const done = T.legs_finished ?? (it.legs || []).filter((l) => l.outcome === 'FINISHED').length;
  const tm = T.adjusted_time ?? T.relay_time;
  return `<b>${done}/${n}</b>${tm != null ? ` · ${fmtClock(tm)}` : ''}`;
}
function renderPicker() {
  const items = state.items || [];
  if (state.itemLabelsFor !== items) {
    state.itemLabelsFor = items;
    const labs = uniqueLabels(items);
    state.itemLabels = new Map(items.map((it, i) => [runIdOf(it), labs[i]]));
  }
  const shownN = items.filter(isShown).length;
  setText($('runs-count'), `${shownN}/${items.length}`);
  setText($('rp-count'), `${shownN} of ${items.length} shown`);
  if (!state.pickerOpen) return;
  // facet chips (only facets that actually vary)
  const fx = $('facets');
  fx.innerHTML = '';
  for (const [key, name] of FACETS) {
    const vals = new Map();
    for (const it of items) { const v = facetVal(it, key); vals.set(v, (vals.get(v) || 0) + 1); }
    if (vals.size < 2 && !state.filters[key].size) continue;
    const row = document.createElement('div');
    row.className = 'facet'; row.dataset.facet = key;
    row.innerHTML = `<span class="eyebrow">${name}</span>`;
    for (const [v, n] of vals) {
      const b = document.createElement('button');
      b.className = `fchip${state.filters[key].has(v) ? ' on' : ''}`;
      b.dataset.value = v;
      b.innerHTML = `${escHTML(String(v))}<span class="n">${n}</span>`;
      b.onclick = () => toggleFilter(key, v);
      row.appendChild(b);
    }
    fx.appendChild(row);
  }
  $('sel-clear').hidden = !FACETS.some(([k]) => state.filters[k].size);
  const ul = $('run-list');
  ul.innerHTML = '';
  let hiddenQ = 0;
  for (const it of items) {
    const id = runIdOf(it);
    if (!matchesQuery(it)) { hiddenQ++; continue; }
    const lab = labelOf(it);
    const li = document.createElement('li');
    li.className = `run-item${isSelected(it) ? '' : ' off'}${passesFilters(it) ? '' : ' filtered'}`;
    li.dataset.run = id;
    li.title = `${lab?.full || it.driver.label}\n${it.file}`;
    li.innerHTML = `<input type="checkbox" ${isSelected(it) ? 'checked' : ''} aria-label="show ${escHTML(lab?.text || it.driver.label)}"><i class="sw" style="background:${colorFor(items, it)}"></i>`
      + `<div class="ri-main"><div class="ri-name">${escHTML(lab?.model || it.driver.label)} <span class="ri-tag">${escHTML(lab?.tag || '')}</span></div><div class="ri-tags">${variantTags(it)}</div></div>`
      + `<div class="ri-res">${itemResult(it)}</div>`;
    li.onclick = (e) => { if (e.target.tagName !== 'INPUT') e.preventDefault(); toggleRun(id); };
    ul.appendChild(li);
  }
  if (hiddenQ) ul.insertAdjacentHTML('beforeend', `<li class="ri-more">${hiddenQ} run${hiddenQ === 1 ? '' : 's'} hidden by the search</li>`);
}
function toggleRun(id) {
  const items = state.items || [];
  state.hint = null; showHint();
  if (!state.sel) state.sel = new Set(items.map(runIdOf));
  if (state.sel.has(id)) state.sel.delete(id); else state.sel.add(id);
  if (items.every((it) => state.sel.has(runIdOf(it)))) state.sel = null;
  writeURL();
  return syncRuns();
}
function toggleFilter(key, v) {
  const f = state.filters[key];
  if (f.has(v)) f.delete(v); else f.add(v);
  writeURL();
  return syncRuns();
}
function selectRuns(ids) {
  state.sel = ids ? new Set(ids) : null;
  state.hint = null; showHint();
  writeURL();
  return syncRuns();
}
function setPicker(open) {
  state.pickerOpen = open;
  $('runs-panel').hidden = !open;
  $('runs-btn').classList.toggle('on', open);
  renderPicker();
  writeURL();
}

// ---------- compare table (aggregated over the shown runs, by variant)
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const sdev = (xs) => { if (xs.length < 2) return null; const m = mean(xs); return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)); };
function variantKey(it) { const v = variantOf(it); return [v.model, v.prompt, v.harness, v.obs, v.mode, v.route].join('|'); }
function renderCompare() {
  const items = (state.items || []).filter(isShown);
  const nLegs = state.relay?.tracks.length ?? Math.max(0, ...items.map((it) => it.totals?.legs ?? 0));
  const groups = new Map();
  for (const it of items) { const k = variantKey(it); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(it); }
  const rows = [...groups.values()].map((its) => {
    const tot = (f) => its.map((it) => f(it.totals || {}, it)).filter((x) => x !== null && x !== undefined && !Number.isNaN(x));
    const finished = its.filter((it) => !it.live && !it.partial);
    const fullyParked = finished.filter((it) => (it.totals?.legs_finished ?? 0) === (it.totals?.legs ?? nLegs));
    return {
      its, color: colorFor(state.items, its[0]), live: its.some((it) => it.live),
      parked: tot((t) => t.legs_finished),
      progress: tot((t) => t.relay_progress),
      adj: fullyParked.map((it) => it.totals.adjusted_time ?? it.totals.relay_time),
      touches: tot((t) => t.touches),
      shuttles: tot((t) => t.shuttles),
      cost: tot((t) => t.cost_usd ?? t.usage?.cost_usd),
    };
  });
  for (const r of rows) for (const k of ['parked', 'progress', 'adj', 'touches', 'shuttles', 'cost']) { r[`${k}M`] = mean(r[k]); r[`${k}S`] = sdev(r[k]); }
  rows.sort((a, b) => (b.parkedM ?? -1) - (a.parkedM ?? -1) || (b.progressM ?? -1) - (a.progressM ?? -1) || (a.adjM ?? 1e9) - (b.adjM ?? 1e9));
  const best = {};
  const cols = { parked: 1, progress: 1, adj: -1, touches: -1, shuttles: -1, cost: -1 };
  for (const [k, dir] of Object.entries(cols)) {
    const vals = rows.map((r) => r[`${k}M`]).filter((x) => x !== null);
    if (vals.length > 1 && Math.max(...vals) - Math.min(...vals) > 1e-9) best[k] = dir > 0 ? Math.max(...vals) : Math.min(...vals);
  }
  const cell = (r, k, fmt, cls = '') => {
    const m = r[`${k}M`];
    if (m === null) return `<td class="dim ${cls}">—</td>`;
    const isBest = best[k] !== undefined && Math.abs(m - best[k]) < 1e-9;
    const sd = r[`${k}S`];
    return `<td class="${isBest ? 'best' : ''} ${cls}" data-col="${k}">${fmt(m)}${sd !== null && sd > 0 ? `<span class="sd">±${fmt(sd)}</span>` : ''}</td>`;
  };
  const legCells = (r) => {
    let h = '<span class="legcells">';
    for (let i = 0; i < nLegs; i++) {
      h += '<span class="legcell">';
      for (const it of r.its.slice(0, 9)) {
        const L = it.legs?.[i];
        const o = L?.outcome;
        const c = !L ? '' : o === 'FINISHED' ? 'ok' : o == null ? (it.live ? 'run' : '') : (L.jackknifes ? 'jk' : 'bad');
        h += `<b class="${c}" title="Leg ${i + 1} · ${escHTML(labelOf(it)?.mini || modelName(describe(it)))}: ${escHTML(STATUS_TEXT[o] || o || (L ? 'driving' : 'not started'))}"></b>`;
      }
      h += '</span>';
    }
    return `${h}</span>`;
  };
  const head = `<thead><tr><th>Driver and set-up</th><th>Runs</th><th>Legs parked</th><th>Progress</th><th title="time including penalties, parked runs only">Adjusted time</th><th>Rail touches</th><th title="forward/reverse changes">Gear changes</th><th>Cost</th><th>Per leg</th></tr></thead>`;
  const body = rows.map((r) => {
    const it = r.its[0];
    return `<tr data-variant="${escHTML(variantKey(it))}"><td class="var"><div class="vn"><i style="background:${r.color}"></i>${escHTML(modelName(describe(it)))}</div><div class="ri-tags">${variantTags({ ...it, live: r.live, partial: false })}</div></td>`
      + `<td data-col="runs">${r.its.length}${r.live ? '<span class="sd">live</span>' : ''}</td>`
      + cell(r, 'parked', (x) => `${x.toFixed(x % 1 ? 1 : 0)}/${nLegs}`)
      + cell(r, 'progress', (x) => `${(100 * x / Math.max(1, nLegs)).toFixed(0)}%`)
      + cell(r, 'adj', (x) => fmtClock(x))
      + cell(r, 'touches', (x) => x.toFixed(x % 1 ? 1 : 0))
      + cell(r, 'shuttles', (x) => x.toFixed(x % 1 ? 1 : 0))
      + cell(r, 'cost', (x) => `$${x < 0.1 ? x.toFixed(3) : x.toFixed(2)}`)
      + `<td>${legCells(r)}</td></tr>`;
  }).join('');
  setHTML($('compare-table'), head + `<tbody>${body || '<tr><td class="dim" colspan="9">No runs selected</td></tr>'}</tbody>`);
  // progress bars after the number look nicer than a bare percentage
  for (const td of $('compare-table').querySelectorAll('td[data-col="progress"]')) {
    const pct = parseFloat(td.textContent);
    if (!Number.isNaN(pct)) td.insertAdjacentHTML('afterbegin', `<span class="pbar"><b style="width:${Math.min(100, pct)}%"></b></span>`);
  }
  const r = state.relay;
  setText($('cmp-title'), r ? `${relayName(r.name)} · ${courseSet(r.seed)}` : '—');
  setText($('cmp-sub'), `${rows.length} set-up${rows.length === 1 ? '' : 's'} · ${items.length} run${items.length === 1 ? '' : 's'} · ${nLegs} legs${rows.some((x) => x.live) ? ' · live runs update every few seconds' : ''}`);
}
function setCompare(open) {
  state.compareOpen = open;
  $('compare').hidden = !open;
  if (open) renderCompare();
  syncButtons();
  writeURL();
}

async function main() {
  initViews();
  $('runs-btn').onclick = () => setPicker(!state.pickerOpen);
  $('rp-close').onclick = () => setPicker(false);
  $('compare-btn').onclick = () => setCompare(!state.compareOpen);
  $('cmp-close').onclick = () => setCompare(false);
  $('compare').onclick = (e) => { if (e.target === $('compare')) setCompare(false); };
  $('live-pill').onclick = goLive;
  $('sel-all').onclick = () => selectRuns(null);
  $('sel-none').onclick = () => selectRuns([]);
  $('sel-clear').onclick = () => { for (const [k] of FACETS) state.filters[k].clear(); writeURL(); syncRuns(); };
  $('relay-select').onchange = async (e) => { await switchRelay(e.target.value); showHint(); };
  $('rp-search').oninput = (e) => { state.q = e.target.value; renderPicker(); };
  $('sel-match').onclick = () => selectRuns((state.items || []).filter((it) => passesFilters(it) && matchesQuery(it)).map(runIdOf));
  for (const [key] of FACETS) for (const v of (params.get(`f_${key}`) || '').split(',').filter(Boolean)) state.filters[key].add(v);

  try {
    state.index = await fetchJSON(RUNS_DIR + 'index.json');
    state.indexTxt = JSON.stringify(state.index);
  } catch (e) {
    state.index = { runs: [] };
  }
  regroup();
  // Landing: with no runs named in the URL, open Results (when the digest exists) and load the 3D scene on demand
  const runsParam = params.get('runs');
  const named = (runsParam && !runsParam.endsWith('/')) || params.has('only') || params.has('relay') || (params.has('view') && params.get('view') !== 'results');
  if (params.get('view') === 'results' || !named) {
    await results.load(true);
    if (params.get('view') === 'results' || results.available()) state.view = 'results';
  }
  if (state.view === 'results') state.lazy = true;
  else await ensureScene();
  setView(state.view);
  if (!state.lazy) { setPicker(state.pickerOpen); setCompare(state.compareOpen); }
  if (!STATIC) setInterval(poll, POLL_MS);

  let last = performance.now();
  let frames = 0, lastSig = '';
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (state.playing && state.runs.length) {
      state.T += dt * state.speed;
      if (state.T >= state.Tmax) {
        state.T = state.Tmax;
        // live: wait at the edge for the next poll instead of stopping
        if (state.anyLive) state.followLive = true;
        else { state.playing = false; syncButtons(); }
      }
    }
    if (state.runs.length) update(dt);
    else updateCamera(dt);
    controls.update();
    // the 3D scene is only on screen in Broadcast (Grid and Inspect are SVG), and a still frame is not redrawn
    const p = camera.position, q = controls.target;
    const sig = [state.T, state.view, state.focus, state.cam, state.clockMode, state.heatLeg, state.sceneRev, window.innerWidth, window.innerHeight,
      p.x, p.y, p.z, q.x, q.y, q.z].map((v) => (typeof v === 'number' ? v.toFixed(3) : v)).join('|');
    if (frames < 2 || (state.view === 'broadcast' && sig !== lastSig)) { renderer.render(scene, camera); lastSig = sig; }
    if (++frames === 2) window.__relayReady = true;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// test & automation hooks
const settle = () => { camSnap = true; lastPanel = 0; if (state.runs.length) update(0); controls.update(); renderer.render(scene, camera); };
window.__relay = {
  seek: (T) => { seek(T); settle(); return state.T; },
  play: (p = true) => setPlaying(p),
  cam: (c) => { setCam(c); settle(); },
  focus: (k) => { setFocus(k); settle(); },
  setClock: (m) => { setClockMode(m, true); settle(); return state.clockMode; },
  heatLeg: (i) => { setHeatLeg(i); settle(); return state.heatLeg; },
  select: async (ids) => { await selectRuns(ids); settle(); return state.runs.map((r) => r.runId); },
  filter: async (key, v) => { await toggleFilter(key, v); settle(); return state.runs.map((r) => r.runId); },
  picker: (open = true) => { if (open && state.view === 'results') setView('broadcast'); setPicker(open); },
  // Results view: state, sections, and opening runs from it
  results: () => results.state(),
  section: (sec) => { if (state.view !== 'results') setView('results'); results.go(sec); return results.section(); },
  openRuns: async (ids, view = 'broadcast', leg) => { await openRuns(ids, view, { leg }); settle(); return state.runs.map((r) => r.runId); },
  // the shared run label (labels.js) for any index item / record
  label: (x) => { const { desc, ...l } = runLabelOf(x); return l; },
  compare: (open = true) => setCompare(open),
  refresh: async () => { const changed = await poll(); settle(); return changed; },
  goLive: () => { goLive(); settle(); return state.T; },
  gaze: () => {
    const run = state.runs[state.focus];
    const a = run?.actors?.[run.live?.active];
    const out = { cone: !!run?.cone, visible: !!a?.cone?.visible, yaw: a?.gaze ?? null, R: a?.cone?.userData.R ?? null, cones: state.runs.reduce((n, r) => n + (r.actors || []).filter((x) => x.cone?.visible).length, 0) };
    if (run?.driver) {
      // driver's-seat view of the focused run's active leg: head state, what is drawn, which mirror / screen glows
      const h = a?.head, v = a?.view, fx = a?.dfx;
      Object.assign(out, {
        driver: true, yaw: h ? h.gaze : null, at: h?.at ?? null, turning: !!h?.turning, words: h ? lookWords(h) : null,
        visible: !!fx?.grp.visible, direct: !!(fx?.grp.visible && fx.direct.visible), aid: fx?.grp.visible && fx.aid.visible ? v?.aid?.kind ?? null : null,
        aidRange: v?.aid?.range ?? null, aidFov: v?.aid ? (v.aid.a1 - v.aid.a0) * 180 / Math.PI : null, aidOrigin: v?.aid?.origin ?? null,
        directFov: v?.direct ? (v.direct.a1 - v.direct.a0) * 180 / Math.PI : null, R: v?.direct?.range ?? null,
        glow: a ? Object.entries(a.combo.mirrors).filter(([, m]) => m.mat.emissiveIntensity > 0).map(([n]) => n) : [],
        screenGlow: !!a?.combo.screen?.visible,
        views: state.runs.reduce((n, r) => n + (r.actors || []).filter((x) => x.dfx?.grp.visible).length, 0),
      });
    }
    return out;
  },
  // the vehicle models of every shown run: which mirrors each carries (and where), and whether it has a camera screen
  models: () => state.runs.map((r) => ({ runId: r.runId, legs: (r.actors || []).map((a, j) => ({ leg: r.legs[j].i, vehicle: r.legs[j].spec.key,
    mirrors: Object.entries(a.combo.mirrors).map(([name, m]) => { const p = m.head.getWorldPosition(new THREE.Vector3()); const s = new THREE.Box3().setFromObject(m.head).getSize(new THREE.Vector3()); return { name, pos: m.pos, u: m.u, height: s.y, world: p.toArray() }; }),
    screen: !!a.combo.screen })) })),
  // world (track-local x, y, height) -> CSS pixels; used by tooling that overlays the 3D view (viewer/mockup)
  project: (legIdx, x, y, h = 0) => {
    const v = state.tracks[legIdx].grp.localToWorld(new THREE.Vector3(x, h, -y)).project(camera);
    const r = renderer.domElement.getBoundingClientRect();
    return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height, v.z];
  },
  // free camera framing one whole leg (dist = multiple of the course size, elev = pitch in degrees)
  frameLeg: (i, dist = 0.95, elev = 48, yaw = 0) => {
    const box = new THREE.Box3().setFromObject(state.tracks[i].grp), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    const R = Math.max(s.x, s.z) * dist, e = elev * Math.PI / 180, a = yaw * Math.PI / 180;
    setCam('free');
    controls.target.copy(c);
    camera.position.set(c.x + R * Math.cos(e) * Math.sin(a), c.y + R * Math.sin(e), c.z + R * Math.cos(e) * Math.cos(a));
    controls.update(); settle();
  },
  // free camera close to the focused run's rig on a course (dist in metres, elev / yaw in degrees; yaw 0 = from behind the vehicle)
  closeup: (legIdx, dist = 9, elev = 30, yaw = 0, along = 0.5) => {
    const run = state.runs[state.focus];
    const j = run?.legs.findIndex((l) => l.i === legIdx);
    if (!(j >= 0)) return false;
    const leg = run.legs[j], st = sampleLeg(leg, Math.max(0, Math.min(state.T - leg.start, leg.time)));
    const [hx, hy] = [st.x - leg.spec.hitch_offset * Math.cos(st.th1), st.y - leg.spec.hitch_offset * Math.sin(st.th1)];
    const tx = st.x + (hx - st.x) * along * 2, ty = st.y + (hy - st.y) * along * 2;
    const g = state.tracks[legIdx].grp, c = g.localToWorld(new THREE.Vector3(tx, 1, -ty));
    const a = st.th1 + Math.PI + yaw * Math.PI / 180, e = elev * Math.PI / 180;
    const p = g.localToWorld(new THREE.Vector3(tx + dist * Math.cos(e) * Math.cos(a), 1 + dist * Math.sin(e), -(ty + dist * Math.cos(e) * Math.sin(a))));
    setCam('free');
    controls.target.copy(c); camera.position.copy(p);
    controls.update(); settle();
    return true;
  },
  // a course as drawn: the bay quad (track coordinates, from the mesh), the surface's end cross-section, and how many
  // scenery props stand inside the driving corridor (left rail + right rail polygon); for asymmetric-course checks
  course: (i) => {
    const t = state.tracks[i];
    if (!t) return null;
    const geo = t.tr.geometry, L = geo.left_rail || [], R = geo.right_rail || [];
    const bp = t.bay?.geometry.attributes.position.array;
    const bay = bp ? [0, 1, 2, 3].map((k) => [bp[k * 3], -bp[k * 3 + 2]]) : null;
    let surface = null;
    t.grp.traverse((o) => { if (o.userData.surface) surface = o; });
    let endWidth = null;
    if (surface) { const a = surface.geometry.attributes.position.array, n = a.length; endWidth = Math.hypot(a[n - 6] - a[n - 3], a[n - 4] - a[n - 1]); }
    const road = L.length && R.length ? [...L, ...R.slice().reverse()] : null;
    let inside = 0, checked = 0;
    if (road && state.scenery) {
      const v = new THREE.Vector3(), m = new THREE.Matrix4();
      const test = (wp) => { checked++; t.grp.worldToLocal(v.copy(wp)); if (pointInPoly([v.x, -v.z], road)) inside++; };
      state.scenery.updateMatrixWorld(true);
      for (const o of state.scenery.children) {
        if (o.userData.slab) continue;
        if (o.isInstancedMesh) for (let k = 0; k < o.count; k++) { o.getMatrixAt(k, m); test(new THREE.Vector3().setFromMatrixPosition(m.premultiply(o.matrixWorld))); }
        else test(o.getWorldPosition(new THREE.Vector3()));
      }
    }
    return { bay, dock: geo.dock_corners || null, endWidth, propsInside: inside, propsChecked: checked, asym: t.tr.meta?.asym ?? null };
  },
  colors: () => state.runs.map((r) => ({ file: r.file, label: r.label, color: r.color })),
  // Broadcast / Grid / Inspect; with no argument, returns the current view
  view: (v, focus) => { if (v) { setView(v, { focus }); settle(); } return state.view; },
  // the in-scene chips currently placed (labels and rings), for tests and tooling
  chips: () => chips.shown().map((c) => ({ ...c })),
  debug: () => ({ cam: camera.position.toArray(), target: controls.target.toArray(), pos: focusInfo.pos.toArray(), front: focusInfo.front.toArray(), rear: focusInfo.rear.toArray(), C: focusInfo.C }),
  state: () => ({
    T: state.T, Tmax: state.Tmax, view: state.view, cam: state.cam, focus: state.focus, relay: state.relay?.id, clockMode: state.clockMode, heatLeg: state.heatLeg,
    anyLive: state.anyLive, followLive: !!state.followLive, polls: state.polls, playing: state.playing,
    shown: (state.items || []).filter(isShown).map(runIdOf), groupRuns: (state.items || []).map(runIdOf),
    lazy: state.lazy, hint: state.hint,
    runs: state.runs.map((r) => ({ label: r.label, name: r.name, mini: r.mini, color: r.color, spec: r.spec, runId: r.runId, file: r.file, live: r.isLive, total: r.total, active: r.live?.active, status: r.live?.liveStatus, score: r.live?.score, legs: r.legs.map((l) => ({ start: l.start, time: l.time, running: l.running })), variant: r.variant })),
  }),
};

main().catch((e) => { console.error(e); window.__relayError = String(e); });
