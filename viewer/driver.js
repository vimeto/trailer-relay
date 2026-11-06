// Driver's-seat view (observation level "driver", relay/perception.py): the rig (eye, head limits, mirrors, reversing
// camera), where the driver looks at a given time and the geometry of what that look covers. Shared by the 3D view
// (app.js, models.js) and Inspect (panels.js). Vehicle frame: x metres ahead of the rear axle, y metres to the left;
// angles in degrees, 0 = vehicle forward, + = left, +-180 = straight back.

export const GLANCE_TOL_DEG = 7.0; // perception.GLANCE_TOL_DEG: gaze this close to a mirror / the screen = looking at it
const TURN_EPS_DEG = 0.5;
// relay/vehicles.py values, for records that predate them in rec.vehicles
const VIEW_RANGE = { mower: 10, car: 22, tractor: 25, truck: 40 };
const SENSOR_RANGE = { mower: 4, car: 6, tractor: 7, truck: 10 };

const deg = (a) => a * 180 / Math.PI;
const rad = (a) => a * Math.PI / 180;
const r1 = (v) => Math.round(v * 10) / 10;
export const wrapDeg = (a) => ((a + 180) % 360 + 360) % 360 - 180;
const dirDeg = (a, b) => deg(Math.atan2(b[1] - a[1], b[0] - a[0]));

/** relay/perception.py rig_for(), ported exactly (used when a record carries no rig). */
export function rigFor(spec) {
  const bf = spec.body_front, w = spec.body_width;
  const fov = spec.view_fov_deg ?? 120, range = spec.view_range ?? VIEW_RANGE[spec.key] ?? 20;
  const sensor = spec.sensor_range ?? SENSOR_RANGE[spec.key] ?? Math.max(4, spec.combo_length * 0.6);
  let eye, lim, block, mirrors, screen, camRange;
  const M = (name, pos, axis, mfov, noise) => ({ name, pos, axis_deg: axis, fov_deg: mfov, noise });
  if (spec.key === 'car') {
    eye = [0.45 * bf, 0.22 * w]; lim = [-160, 160]; block = null;
    const mx = bf - 1.3;
    mirrors = [M('left_mirror', [mx, w / 2 + 0.12], 180 - 11 + 2, 22, 1.4), M('right_mirror', [mx, -(w / 2 + 0.12)], -(180 - 14 + 2), 28, 1.8)];
    screen = [bf - 1.0, 0.0]; camRange = sensor;
  } else if (spec.key === 'truck') {
    eye = [bf - 1.6, w / 2 - 0.67]; lim = [-120, 120]; block = 100;
    const mx = bf - 0.9;
    mirrors = [M('left_mirror', [mx, w / 2 + 0.25], 180 - 18 + 2, 36, 1.5), M('right_mirror', [mx, -(w / 2 + 0.25)], -(180 - 18 + 2), 36, 1.8)];
    screen = [bf - 0.9, 0.0]; camRange = sensor;
  } else if (spec.key === 'tractor') {
    eye = [0.25 * bf, 0.0]; lim = [-170, 170]; block = null;
    const mx = bf - 1.6;
    mirrors = [M('left_mirror', [mx, w / 2 + 0.2], 180 - 15 + 2, 30, 1.5), M('right_mirror', [mx, -(w / 2 + 0.2)], -(180 - 15 + 2), 30, 1.5)];
    screen = null; camRange = 0.0;
  } else {
    eye = [0.1, 0.0]; lim = [-170, 170]; block = null;
    mirrors = []; screen = null; camRange = 0.0;
  }
  return {
    eye, gaze_min: lim[0], gaze_max: lim[1], fov_deg: fov, range, cab_block_deg: block, mirrors, screen,
    camera_fov_deg: 130.0, camera_range: camRange,
    mirror_gaze: Object.fromEntries(mirrors.map((m) => [m.name, r1(dirDeg(eye, m.pos))])),
    screen_gaze: screen ? r1(dirDeg(eye, screen)) : null,
  };
}

/** The rig of a vehicle in a record: rec.vehicles[v].rig, else config.perception.rigs[v], else rigFor(spec). */
export function rigOf(rec, key, spec) {
  return rec?.vehicles?.[key]?.rig || rec?.config?.perception?.rigs?.[key] || rigFor(spec);
}

/** What a gaze direction is aimed at: 'left_mirror' | 'right_mirror' | 'camera' | null (perception.perceive order). */
export function lookingAt(rig, gaze) {
  let at = null;
  for (const m of rig.mirrors || []) if (Math.abs(gaze - rig.mirror_gaze[m.name]) <= GLANCE_TOL_DEG) at = m.name;
  if (rig.screen_gaze !== null && rig.screen_gaze !== undefined && Math.abs(gaze - rig.screen_gaze) <= GLANCE_TOL_DEG) at = 'camera';
  return at;
}

/**
 * Head state at leg time t from trace.gaze: { gaze (interpolated, deg), turning, target (where the turn ends), at }.
 * The head is "turning" while the gaze changes between samples (and one sample after: the saccade) - nothing is seen then.
 */
export function headAt(leg, t, idx) {
  const g = leg.trace.gaze;
  if (!g?.length) return null;
  const T = leg.trace.t, n = g.length;
  const i = Math.max(0, Math.min(n - 1, idx));
  const j = Math.min(i + 1, n - 1);
  const span = T[j] - T[i];
  const f = span > 1e-9 ? Math.min(1, Math.max(0, (t - T[i]) / span)) : 0;
  const gaze = g[i] + (g[j] - g[i]) * f;
  const turning = Math.abs(g[j] - g[i]) > TURN_EPS_DEG || (i > 0 && Math.abs(g[i] - g[i - 1]) > TURN_EPS_DEG);
  let target = g[j];
  if (turning) {
    // where the head comes to rest: the first pair of equal samples ahead
    for (let k = j; k < Math.min(n - 1, j + 80); k++) if (Math.abs(g[k + 1] - g[k]) <= TURN_EPS_DEG) { target = g[k]; break; }
  }
  const rig = leg.rig;
  return { gaze, turning, target, at: turning || !rig ? null : lookingAt(rig, gaze), targetAt: rig ? lookingAt(rig, target) : null };
}

const AT_WORDS = { left_mirror: 'left mirror', right_mirror: 'right mirror', camera: 'camera' };
function dirWords(g) {
  const d = wrapDeg(g), a = Math.abs(d), side = d > 0 ? 'left' : 'right';
  if (a < 15) return 'ahead';
  if (a < 70) return `ahead-${side} ${a.toFixed(0)}°`;
  if (a < 110) return `${side} ${a.toFixed(0)}°`;
  return `over ${side} shoulder ${a.toFixed(0)}°`;
}
/** "left mirror" / "camera" / "over left shoulder 152°" / "ahead" / "turning → right mirror" */
export function lookWords(h) {
  if (!h) return '—';
  if (h.turning) return `turning → ${AT_WORDS[h.targetAt] || dirWords(h.target)}`;
  return AT_WORDS[h.at] || dirWords(h.gaze);
}

// vehicle frame -> world (sim) coordinates for pose st
const toWorld = (st, [x, y]) => [st.x + x * Math.cos(st.th1) - y * Math.sin(st.th1), st.y + x * Math.sin(st.th1) + y * Math.cos(st.th1)];

/**
 * What the look covers at pose st (world coordinates, angles in radians, world frame):
 *   eye, gaze (heading), direct {origin, a0, a1, range} | null (cab-blocked or turning -> dim),
 *   periph [{origin, a0, a1, range}] (corner of the eye, perception v2; empty for older records),
 *   aid {kind: 'left_mirror'|'right_mirror'|'camera', origin, a0, a1, range, via (mirror / screen point)} | null,
 *   mirrors [{name, pos}], screen (world point) | null
 */
export function viewGeometry(rig, sp, st, h) {
  const eye = toWorld(st, rig.eye);
  const out = { eye, gaze: st.th1 + rad(h.gaze), direct: null, aid: null, turning: h.turning,
    mirrors: (rig.mirrors || []).map((m) => ({ name: m.name, pos: toWorld(st, m.pos) })), screen: rig.screen ? toWorld(st, rig.screen) : null };
  let lo = h.gaze - rig.fov_deg / 2, hi = h.gaze + rig.fov_deg / 2;
  if (rig.cab_block_deg !== null && rig.cab_block_deg !== undefined) { lo = Math.max(lo, -rig.cab_block_deg); hi = Math.min(hi, rig.cab_block_deg); }
  if (hi > lo) out.direct = { origin: eye, a0: st.th1 + rad(lo), a1: st.th1 + rad(hi), range: rig.range };
  // corner of the eye (perception v2): two side bands beyond the direct cone, close by only
  out.periph = [];
  if (rig.periphery_range > 0) {
    const P = rig.periphery_deg ?? 100, cb = rig.cab_block_deg ?? 180;
    for (let [a, b] of [[h.gaze + rig.fov_deg / 2, h.gaze + P], [h.gaze - P, h.gaze - rig.fov_deg / 2]]) {
      a = Math.max(a, -cb); b = Math.min(b, cb);
      if (b > a) out.periph.push({ origin: eye, a0: st.th1 + rad(a), a1: st.th1 + rad(b), range: rig.periphery_range });
    }
  }
  if (h.at === 'camera') {
    const ax = st.x - sp.hitch_offset * Math.cos(st.th1) - sp.trailer_length * Math.cos(st.th2);
    const ay = st.y - sp.hitch_offset * Math.sin(st.th1) - sp.trailer_length * Math.sin(st.th2);
    const origin = [ax - sp.trailer_rear * Math.cos(st.th2), ay - sp.trailer_rear * Math.sin(st.th2)];
    const axis = st.th2 + Math.PI, half = rad(rig.camera_fov_deg / 2);
    out.aid = { kind: 'camera', origin, a0: axis - half, a1: axis + half, range: rig.camera_range, via: out.screen };
  } else if (h.at) {
    const m = rig.mirrors.find((x) => x.name === h.at);
    if (m) {
      const axis = st.th1 + rad(m.axis_deg), half = rad(m.fov_deg / 2);
      const origin = toWorld(st, m.pos);
      out.aid = { kind: m.name, origin, a0: axis - half, a1: axis + half, range: rig.range, via: origin };
    }
  }
  return out;
}
