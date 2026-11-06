// 2D geometry in sim coordinates (x, y; headings in radians), shared by the Grid, Inspect and chips.
// The perception overlay re-derives the driver's report from the recorded pose with the same rules as
// relay/observe.py (records store poses, not reports), so it is a reconstruction, not a replay.

const cos = Math.cos, sin = Math.sin;

export function hitchPoint(st, sp) { return [st.x - sp.hitch_offset * cos(st.th1), st.y - sp.hitch_offset * sin(st.th1)]; }
export function trailerAxle(st, sp) {
  const [hx, hy] = hitchPoint(st, sp);
  return [hx - sp.trailer_length * cos(st.th2), hy - sp.trailer_length * sin(st.th2)];
}
export function rearCenter(st, sp) {
  const [ax, ay] = trailerAxle(st, sp);
  return [ax - sp.trailer_rear * cos(st.th2), ay - sp.trailer_rear * sin(st.th2)];
}
// corners [FL, FR, RR, RL] (relay/geometry.py rect_corners)
export function rect(cx, cy, h, front, rear, hw) {
  const ux = cos(h), uy = sin(h), nx = -uy, ny = ux;
  return [[cx + front * ux + hw * nx, cy + front * uy + hw * ny], [cx + front * ux - hw * nx, cy + front * uy - hw * ny],
    [cx - rear * ux - hw * nx, cy - rear * uy - hw * ny], [cx - rear * ux + hw * nx, cy - rear * uy + hw * ny]];
}
export function rigShape(st, sp) {
  const ax = trailerAxle(st, sp);
  return {
    vehicle: rect(st.x, st.y, st.th1, sp.body_front, sp.body_rear, sp.body_width / 2),
    trailer: rect(ax[0], ax[1], st.th2, sp.trailer_front, sp.trailer_rear, sp.trailer_width / 2),
    hitch: hitchPoint(st, sp), axle: ax,
  };
}
export function hitchDeg(st) {
  const h = ((st.th1 - st.th2 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return h * 180 / Math.PI;
}
export const critOf = (sp) => Math.min(sp.critical_hitch_deg ?? sp.max_hitch_deg, sp.max_hitch_deg);

export function resample(points, step) {
  if (!points?.length) return [];
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

// a closed wall polygon is the whole boundary (turnaround courses); left/right rails are then only its two halves
const isClosed = (w) => w?.length > 3 && Math.hypot(w[0][0] - w[w.length - 1][0], w[0][1] - w[w.length - 1][1]) < 1e-3;
export const hasWallBoundary = (geo) => (geo.walls || []).some(isClosed);

// every polyline that acts as a rail: the two route rails, the start and dock walls, plus geometry.walls
export function railLines(geo) {
  const L = geo.left_rail, R = geo.right_rail, lines = [];
  if (L?.length && R?.length && !hasWallBoundary(geo)) lines.push(L, R, [R[0], L[0]], [R[R.length - 1], L[L.length - 1]]);
  for (const w of geo.walls || []) if (w?.length > 1) lines.push(w);
  return lines;
}

// ---- multi-phase courses (turnaround): geometry.routes = one guide route per phase
// polyline arclength, projection and lookup
export function polyS(P) {
  const S = [0];
  for (let k = 1; k < P.length; k++) S.push(S[k - 1] + Math.hypot(P[k][0] - P[k - 1][0], P[k][1] - P[k - 1][1]));
  return S;
}
export function projectS(P, S, x, y) {
  let best = 1e18, bs = 0;
  for (let k = 0; k < P.length - 1; k++) {
    const x0 = P[k][0], y0 = P[k][1], dx = P[k + 1][0] - x0, dy = P[k + 1][1] - y0;
    const u = Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy || 1e-9)));
    const ex = x0 + u * dx - x, ey = y0 + u * dy - y, d = ex * ex + ey * ey;
    if (d < best) { best = d; bs = S[k] + u * (S[k + 1] - S[k]); }
  }
  return bs;
}
export function pointOn(P, S, s) {
  if (s <= 0) return P[0];
  for (let k = 1; k < S.length; k++) {
    if (S[k] >= s) { const f = (s - S[k - 1]) / Math.max(1e-9, S[k] - S[k - 1]); return [P[k - 1][0] + f * (P[k][0] - P[k - 1][0]), P[k - 1][1] + f * (P[k][1] - P[k - 1][1])]; }
  }
  return P[P.length - 1];
}
/**
 * Phase routes of a multi-phase track, as relay/turnaround.py builds them: [{P, S, off, dir, ref}], or null.
 * Progress along the manoeuvre = off + (arclength of the phase's reference point on its route); phase 1 follows
 * the trailer axle, phase 2 the vehicle's rear axle. Phase 2's offset joins the two routes where the rig stands
 * straight in the side road with its trailer axle on the phase-1 goal: off = s1_goal - s2(ref2), with ref2 one
 * hitch + trailer length back along route 1.
 */
export function phaseRoutes(track, sp) {
  const g = track.geometry, ph = track.phases;
  if (!(g.routes?.length > 1 && ph?.length > 1)) return null;
  const key = `_routes_${sp.key}`;
  if (track[key]) return track[key];
  const out = g.routes.map((P, i) => ({ P, S: polyS(P), off: 0, dir: ph[i]?.direction ?? (i ? 1 : -1), ref: ph[i]?.ref ?? (i ? 'vehicle' : 'trailer') }));
  const s1goal = ph[0].goal_s ?? out[0].S[out[0].S.length - 1];
  const ref2 = pointOn(out[0].P, out[0].S, s1goal - (sp.hitch_offset + sp.trailer_length));
  out[1].off = s1goal - projectS(out[1].P, out[1].S, ref2[0], ref2[1]);
  Object.defineProperty(track, key, { value: out, enumerable: false });
  return out;
}
// manoeuvre arclength of the pose in phase index i (0-based)
export function phaseProgressS(track, sp, st, i) {
  const R = phaseRoutes(track, sp);
  if (!R) return null;
  const r = R[Math.min(i, R.length - 1)];
  const p = r.ref === 'trailer' ? trailerAxle(st, sp) : [st.x, st.y];
  return r.off + projectS(r.P, r.S, p[0], p[1]);
}
function segments(geo) {
  if (geo._segs) return geo._segs;
  const segs = [];
  for (const line of railLines(geo)) for (let k = 0; k + 1 < line.length; k++) segs.push([line[k][0], line[k][1], line[k + 1][0], line[k + 1][1]]);
  Object.defineProperty(geo, '_segs', { value: segs, enumerable: false });
  return segs;
}
// distance along a ray to the nearest rail, capped at rng
export function rayDistance(px, py, dx, dy, geo, rng) {
  let best = rng;
  for (const [x0, y0, x1, y1] of segments(geo)) {
    const ex = x1 - x0, ey = y1 - y0;
    const den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-12) continue;
    const wx = x0 - px, wy = y0 - py;
    const t = (wx * ey - wy * ex) / den, u = (wx * dy - wy * dx) / den;
    if (t >= 0 && u >= 0 && u <= 1 && t < best) best = t;
  }
  return best;
}

function railSamples(track, sp) {
  const key = `_samples_${sp.key}`;
  if (track[key]) return track[key];
  const g = track.geometry;
  const step = Math.min(2, Math.max(0.3, sp.combo_length / 10));
  const lines = [];
  if (!hasWallBoundary(g)) {
    if (g.right_rail?.length) lines.push(resample(g.right_rail, step));
    if (g.left_rail?.length) lines.push(resample(g.left_rail, step));
  }
  for (const w of g.walls || []) if (w?.length > 1) lines.push(resample(w, step));
  Object.defineProperty(track, key, { value: lines, enumerable: false });
  return lines;
}

export const eyePoint = (st, sp) => [st.x + 0.4 * sp.body_front * cos(st.th1), st.y + 0.4 * sp.body_front * sin(st.th1)];

function inCone(p, eye, heading, fovDeg, range) {
  const dx = p[0] - eye[0], dy = p[1] - eye[1];
  if (Math.hypot(dx, dy) > range) return false;
  let a = Math.atan2(dy, dx) - heading;
  a = ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return Math.abs(a) <= fovDeg * Math.PI / 360;
}

// sensor layout (relay/observe.py _sensor_layout): [name, [f, l] in the body frame, [df, dl]]
function sensorLayout(sp) {
  const th = sp.trailer_width / 2, vh = sp.body_width / 2, r2 = Math.SQRT1_2;
  const tmid = (sp.trailer_front - sp.trailer_rear) / 2, vmid = (sp.body_front - sp.body_rear) / 2;
  return {
    trailer: [['rear-left', [-sp.trailer_rear, th], [-r2, r2]], ['rear', [-sp.trailer_rear, 0], [-1, 0]], ['rear-right', [-sp.trailer_rear, -th], [-r2, -r2]],
      ['left', [tmid, th], [0, 1]], ['right', [tmid, -th], [0, -1]]],
    vehicle: [['front-left', [sp.body_front, vh], [r2, r2]], ['front', [sp.body_front, 0], [1, 0]], ['front-right', [sp.body_front, -vh], [r2, -r2]],
      ['left', [vmid, vh], [0, 1]], ['right', [vmid, -vh], [0, -1]]],
  };
}

// centreline point at arclength s
function pointAtS(geo, s) {
  const C = geo.centerline, S = geo.s;
  if (s <= S[0]) return C[0];
  for (let k = 1; k < S.length; k++) {
    if (S[k] >= s) { const f = (s - S[k - 1]) / Math.max(1e-9, S[k] - S[k - 1]); return [C[k - 1][0] + f * (C[k][0] - C[k - 1][0]), C[k - 1][1] + f * (C[k][1] - C[k - 1][1])]; }
  }
  return C[C.length - 1];
}

/**
 * What the driver's report contained at this pose, by observation level:
 *   points  rail samples in the local map (all levels except "sensors"; in-cone only for cone levels)
 *   radius  the local-map radius around the trailer rear (non-cone)
 *   rays    proximity sensors (non-cone levels)
 *   route   the route arrow (non-cone levels): from the trailer rear to the centreline 0.6 C ahead
 *   cone    {eye, heading, fov, range} for cone levels
 */
export function perception(leg, st, { obs = 'local', gazeDeg = 180, progressS = null, phase = 0 } = {}) {
  const sp = leg.spec, g = leg.track.geometry;
  const cone = String(obs).startsWith('cone');
  const routes = phaseRoutes(leg.track, sp);
  const out = { obs, cone: null, points: [], radius: null, center: rearCenter(st, sp), rays: [], route: null };
  const o = out.center;
  if (cone) {
    const eye = eyePoint(st, sp), heading = st.th1 + gazeDeg * Math.PI / 180;
    const fov = sp.view_fov_deg ?? 120, range = sp.view_range ?? 1.6 * (sp.view_radius ?? 14);
    out.cone = { eye, heading, fov, range };
    for (const line of railSamples(leg.track, sp)) for (const p of line) if (inCone(p, eye, heading, fov, range)) out.points.push(p);
    return out;
  }
  if (obs !== 'sensors') {
    const R = sp.view_radius ?? 14;
    out.radius = R;
    for (const line of railSamples(leg.track, sp)) {
      // driving forward (turnaround phase 2) the driver watches ahead of the vehicle as much as behind the trailer
      const vr = (routes && routes[Math.min(phase, routes.length - 1)].dir > 0 ? 1 : 0.5) * R;
      for (const p of line) if (Math.hypot(p[0] - o[0], p[1] - o[1]) <= R || Math.hypot(p[0] - st.x, p[1] - st.y) <= vr) out.points.push(p);
    }
  }
  const rng = sp.sensor_range ?? Math.max(4, sp.combo_length * 0.6);
  const lay = sensorLayout(sp);
  const ax = trailerAxle(st, sp);
  for (const [grp, origin, h] of [['trailer', ax, st.th2], ['vehicle', [st.x, st.y], st.th1]]) {
    const c = cos(h), s = sin(h);
    for (const [name, [pf, pl], [df, dl]] of lay[grp]) {
      const px = origin[0] + pf * c - pl * s, py = origin[1] + pf * s + pl * c;
      const dx = df * c - dl * s, dy = df * s + dl * c;
      const d = rayDistance(px, py, dx, dy, g, rng);
      out.rays.push({ name: `${grp} ${name}`, from: [px, py], to: [px + dx * d, py + dy * d], d, hit: d < rng - 1e-6 });
    }
  }
  if (routes) {
    // phase-aware arrow (observe._route_arrow_turn): reversing, from the trailer rear; forward, from the vehicle front
    const r = routes[Math.min(phase, routes.length - 1)], fwd = r.dir > 0;
    const sl = phaseProgressS(leg.track, sp, st, phase) - r.off;
    const to = pointOn(r.P, r.S, Math.min(r.S[r.S.length - 1], sl + (fwd ? sp.body_front : sp.trailer_rear) + 0.6 * sp.combo_length));
    const from = fwd ? [st.x + sp.body_front * cos(st.th1), st.y + sp.body_front * sin(st.th1)] : o;
    out.route = { from, to };
  } else if (g.centerline?.length && g.s?.length && progressS !== null) {
    const length = leg.track.length ?? g.s[g.s.length - 1];
    out.route = { from: o, to: pointAtS(g, Math.min(length, progressS + sp.trailer_rear + 0.6 * sp.combo_length)) };
  }
  return out;
}

export function pointInPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// all points that describe a course (for bounding boxes)
export function coursePoints(geo) {
  const pts = [...(geo.left_rail || []), ...(geo.right_rail || []), ...(geo.dock_corners || [])];
  for (const w of geo.walls || []) pts.push(...w);
  for (const z of geo.zones || []) pts.push(...(z.polygon || []));
  if (!pts.length && geo.centerline) pts.push(...geo.centerline);
  return pts;
}

// The driving corridor as discs [x, y, r] every <= step metres: one per rail cross-section (midpoint of the paired rail
// points, radius half the gap), so asymmetric courses (relay asym: a yard or lay-by on one side) are covered where they
// really are; courses without rails fall back to the centreline and halfwidth.
export function corridorDiscs(geo, step = 1.5) {
  const L = geo.left_rail, R = geo.right_rail, out = [];
  if (L?.length > 1 && R?.length === L.length && !hasWallBoundary(geo)) {
    let prev = null;
    for (let k = 0; k < L.length; k++) {
      const cur = [(L[k][0] + R[k][0]) / 2, (L[k][1] + R[k][1]) / 2, Math.hypot(L[k][0] - R[k][0], L[k][1] - R[k][1]) / 2];
      if (prev) {
        const n = Math.max(1, Math.ceil(Math.hypot(cur[0] - prev[0], cur[1] - prev[1]) / step));
        for (let j = 1; j < n; j++) out.push(prev.map((v, i) => v + (cur[i] - v) * j / n));
      }
      out.push(cur);
      prev = cur;
    }
    return out;
  }
  const C = geo.centerline || [];
  const hw = Math.max(1.5, ...(geo.halfwidth || [3]));
  for (const [x, y] of resample(C, step)) out.push([x, y, hw]);
  return out;
}
