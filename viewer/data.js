// Loading run records and sampling them at a given race time.
import { phaseProgressS } from './geom2d.js';
import { rigOf } from './driver.js';

export const VEHICLE_FALLBACK = {
  mower: { key: 'mower', name: 'Ride-on mower', trailer_name: 'garden cart', wheelbase: 1.25, hitch_offset: 0.3, trailer_length: 0.85, max_steer_deg: 45, max_hitch_deg: 85, body_front: 1.55, body_rear: 0.25, body_width: 1.05, trailer_front: 0.55, trailer_rear: 0.45, trailer_width: 0.95, critical_hitch_deg: 54.9, combo_length: 3.15, view_radius: 7 },
  car: { key: 'car', name: 'Car', trailer_name: 'box trailer', wheelbase: 2.75, hitch_offset: 1.05, trailer_length: 2.6, max_steer_deg: 34, max_hitch_deg: 75, body_front: 3.6, body_rear: 1.0, body_width: 1.85, trailer_front: 2.0, trailer_rear: 0.9, trailer_width: 1.7, critical_hitch_deg: 52.6, combo_length: 8.15, view_radius: 14 },
  tractor: { key: 'tractor', name: 'Tractor', trailer_name: 'tipping trailer', wheelbase: 2.5, hitch_offset: 1.3, trailer_length: 3.6, max_steer_deg: 45, max_hitch_deg: 80, body_front: 3.0, body_rear: 0.9, body_width: 2.3, trailer_front: 3.2, trailer_rear: 1.4, trailer_width: 2.4, critical_hitch_deg: 80, combo_length: 9.3, view_radius: 16 },
  truck: { key: 'truck', name: 'Semi truck', trailer_name: 'semi-trailer', wheelbase: 3.9, hitch_offset: -0.35, trailer_length: 10.2, max_steer_deg: 42, max_hitch_deg: 85, body_front: 5.2, body_rear: 1.0, body_width: 2.55, trailer_front: 11.6, trailer_rear: 1.9, trailer_width: 2.55, critical_hitch_deg: 85, combo_length: 16.95, view_radius: 30 },
};

export const PALETTE = ['#4c8dff', '#f5a524', '#b07cff', '#35c983', '#ff6b6b', '#29c6d6', '#ff8fc7'];
export const AUTOPILOT_COLOR = '#e9e4d8';

// Colours: one hue per model (fixed: labels.js MODEL_COLORS, the same in every view), a distinct shade per run of
// that model (variants first, then trials). entries: [{ id, model, base, order, auto }] in display order, where `base` is
// the model's colour; returns Map id -> '#rrggbb'.
const MODEL_HUES = [214, 30, 278, 148, 352, 188, 56, 322];
function hsl2hex(h, s, l) {
  s /= 100; l /= 100;
  const k = (n) => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)))).toString(16).padStart(2, '0');
  return `#${f(0)}${f(8)}${f(4)}`;
}
function hex2hsl(hex) {
  const n = parseInt(hex.slice(1), 16), r = (n >> 16) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  let h = 0;
  if (d) h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, d ? 100 * d / (1 - Math.abs(2 * l - 1)) : 0, 100 * l];
}
export function assignColors(entries) {
  const out = new Map();
  const models = [];
  for (const e of entries) if (!e.auto && !models.includes(e.model)) models.push(e.model);
  models.forEach((model, mi) => {
    const mine = entries.filter((e) => !e.auto && e.model === model);
    const base = mine[0].base;
    const [bh, bs] = base ? hex2hsl(base) : [MODEL_HUES[mi % MODEL_HUES.length], 88];
    const sat = Math.min(1, bs / 80); // a grey baseline stays greyish
    const n = mine.length, hue = bh;
    // shades alternate light/dark so neighbours in the list differ most; hue drifts a little within ±14°
    const order = [...Array(n).keys()].sort((a, b) => (a % 2) - (b % 2) || a - b);
    if (n === 1 && base) { out.set(mine[0].id, base); return; }
    mine.forEach((e, k) => {
      const r = n === 1 ? 0.5 : order.indexOf(k) / (n - 1);
      const L = n === 1 ? 60 : 42 + 36 * r, S = (88 - 22 * r) * sat;
      const H = hue + (n === 1 ? 0 : (k / (n - 1) - 0.5) * Math.min(28, 6 * n));
      out.set(e.id, hsl2hex((H + 360) % 360, S, L));
    });
  });
  for (const e of entries) if (e.auto) out.set(e.id, AUTOPILOT_COLOR);
  return out;
}

export async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function bsearch(arr, t) {
  let lo = 0, hi = arr.length - 1;
  if (t <= arr[0]) return 0;
  if (t >= arr[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] <= t) lo = mid; else hi = mid;
  }
  return lo;
}

const lerpAngle = (a, b, f) => {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
};

// Variant facets of a run record (missing fields = defaults of older records).
export function variantOf(rec) {
  const c = rec.config || {};
  return {
    model: rec.driver?.label || rec.driver?.spec || '?',
    prompt: c.prompt_version ?? 1,
    harness: c.harness_version ?? 1,
    obs: c.agent?.obs_level ?? (rec.driver?.spec === 'autopilot' ? 'full' : 'local'),
    mode: c.mode || 'tools',
    route: c.route || 'api',
    physics: c.sim_version ?? 1,
    // discrete = every command ends standing still; continuous = the rig keeps rolling (paused or realtime clock)
    control: c.control === 'continuous' ? `continuous · ${c.clock || 'paused'}` : 'discrete',
  };
}

export function prepareRun(rec, color, file) {
  const vehicles = { ...VEHICLE_FALLBACK, ...(rec.vehicles || {}) };
  let cum = 0;
  const legs = rec.legs.map((L, i) => {
    let tr = L.trace;
    if (!tr || !tr.t?.length) {
      // a live leg that has not produced a trace sample yet: park the rig at its start pose
      const ss = rec.relay.tracks[L.index ?? i]?.start_state || [0, 0, 0, 0];
      tr = { t: [0], x: [ss[0]], y: [ss[1]], th1: [ss[2]], th2: [ss[3]], steer: [0] };
    }
    const n = tr.t.length;
    const odo = new Float64Array(n);
    for (let k = 1; k < n; k++) {
      const dx = tr.x[k] - tr.x[k - 1], dy = tr.y[k] - tr.y[k - 1];
      const fwd = dx * Math.cos(tr.th1[k]) + dy * Math.sin(tr.th1[k]);
      odo[k] = odo[k - 1] + Math.sign(fwd) * Math.hypot(dx, dy);
    }
    const ev = (L.events || []).slice().sort((a, b) => a.t - b.t);
    const calls = [];
    for (const e of L.transcript || []) for (const c of e.calls || []) if (c.t0 !== undefined) calls.push({ ...c, text: e.text, thought: e.reasoning, turn: e.turn });
    calls.sort((a, b) => a.t0 - b.t0);
    // continuous code mode: keep_driving shows the controller that is still driving
    let ctrl = null;
    for (const c of calls) {
      if (c.name === 'set_controller' && !c.error && c.args?.code) ctrl = c.args.code;
      if (c.name === 'keep_driving' && ctrl) { c.args = { ...c.args, code: ctrl }; c.kept = true; }
    }
    const looks = ev.filter((e) => e.type === 'look');
    // cumulative path length (|ds|) and stop count, for "metres driven meanwhile" and the stops counter
    const dist = new Float64Array(n), stopsCum = new Int32Array(n);
    for (let k = 1; k < n; k++) {
      dist[k] = dist[k - 1] + Math.hypot(tr.x[k] - tr.x[k - 1], tr.y[k] - tr.y[k - 1]);
      stopsCum[k] = stopsCum[k - 1] + (tr.v && Math.abs(tr.v[k - 1]) >= 0.02 && Math.abs(tr.v[k]) < 0.02 ? 1 : 0);
    }
    // model turns: asked at t, answer applied at t_reply (realtime clock: the rig keeps driving in between)
    const turns = (L.transcript || []).filter((e) => e.t !== undefined && e.t !== null).map((e) => ({
      turn: e.turn, t: e.t, t_reply: e.t_reply ?? e.t, think: e.think_sim_s ?? Math.max(0, (e.t_reply ?? e.t) - e.t), latency: e.latency_s,
      text: e.text || '', reasoning: e.reasoning || '', calls: e.calls || [], late: !!e.late, usage: e.usage,
    }));
    const phases = ev.filter((e) => e.type === 'phase');
    const time = L.metrics?.time ?? (n ? tr.t[n - 1] : 0);
    const running = !!L.running || L.metrics?.outcome == null;
    const relayStart = L.start_clock ?? cum;
    cum = relayStart + time;
    const spec = vehicles[L.vehicle] || VEHICLE_FALLBACK.car;
    return {
      // rig: the driver's eye, head limits, mirrors and reversing camera (every run's models carry the mirrors)
      i: L.index ?? i, L, spec, rig: rigOf(rec, L.vehicle, spec), track: rec.relay.tracks[L.index ?? i], trace: tr, odo, events: ev, calls,
      looks, dist, stopsCum, turns, phases, gaze0: L.gaze0_deg ?? 180, time, running, relayStart,
      start: relayStart, end: relayStart + time, outcome: running ? null : L.metrics.outcome,
    };
  });
  const run = {
    file, rec, color, legs, label: rec.driver.label, spec: rec.driver.spec, isAuto: rec.driver.spec === 'autopilot',
    runId: rec.run_id || file, isLive: !!rec.live, variant: variantOf(rec), total: 0,
  };
  run.cone = String(run.variant.obs).startsWith('cone');
  run.driver = run.variant.obs === 'driver'; // driver's-seat perception: head, mirrors, reversing camera (trace.gaze)
  run.code = run.variant.mode === 'code';
  applyClock(run, 'relay');
  return run;
}

// "relay": legs stitched onto one clock (leg i starts when leg i-1 ends).
// "heats": every leg starts at T = 0 (they are driven concurrently anyway).
export function applyClock(run, mode) {
  for (const leg of run.legs) {
    leg.start = mode === 'heats' ? 0 : leg.relayStart;
    leg.end = leg.start + leg.time;
  }
  run.total = run.legs.length ? Math.max(...run.legs.map((l) => l.end)) : 0;
  run.clockMode = mode;
}

// gaze yaw (deg, relative to the vehicle heading, + left) at leg time t
export function gazeAt(leg, t) {
  let g = leg.gaze0, prev = leg.gaze0, tLook = -1e9;
  for (const e of leg.looks) { if (e.t <= t + 1e-6) { prev = g; g = e.yaw_deg; tLook = e.t; } else break; }
  // a quick head turn rather than a jump
  const f = Math.min(1, Math.max(0, (t - tLook) / 0.35));
  let d = g - prev;
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return prev + d * (f * f * (3 - 2 * f));
}

export function sampleLeg(leg, t) {
  const tr = leg.trace;
  const i = bsearch(tr.t, t);
  const j = Math.min(i + 1, tr.t.length - 1);
  const span = tr.t[j] - tr.t[i];
  const f = span > 1e-9 ? Math.min(1, Math.max(0, (t - tr.t[i]) / span)) : 0;
  return {
    x: tr.x[i] + (tr.x[j] - tr.x[i]) * f,
    y: tr.y[i] + (tr.y[j] - tr.y[i]) * f,
    th1: lerpAngle(tr.th1[i], tr.th1[j], f),
    th2: lerpAngle(tr.th2[i], tr.th2[j], f),
    steer: tr.steer[i] + (tr.steer[j] - tr.steer[i]) * f,
    v: tr.v ? tr.v[i] + (tr.v[j] - tr.v[i]) * f : null,
    odo: leg.odo[i] + (leg.odo[j] - leg.odo[i]) * f,
    idx: i,
  };
}

// Where is this run at race time T? -> the leg in focus ("active") and its local time.
// In heats mode every leg is live at once; `heatLeg` picks which one is in focus.
export function runAt(run, T, heatLeg = 0) {
  let active = run.legs.length - 1;
  if (run.clockMode === 'heats') {
    active = Math.min(heatLeg, run.legs.length - 1);
  } else {
    for (let k = 0; k < run.legs.length; k++) {
      if (T < run.legs[k].end || run.legs[k].running) { active = k; break; }
    }
  }
  const leg = run.legs[active];
  const t = leg ? Math.max(0, Math.min(T - leg.start, leg.time)) : 0;
  const finishedAll = !run.isLive && run.legs.length > 0 && T >= run.total;
  return { active, leg, t, finishedAll };
}

// elapsed race time of a run: the sum of each leg's elapsed time (equals the relay clock in relay mode)
export function elapsedOf(run, T) {
  let s = 0;
  for (const l of run.legs) s += Math.max(0, Math.min(T - l.start, l.time));
  return s;
}

export function legStatusAt(leg, t) {
  let status = 'READY', touches = 0, shuttles = 0, jk = 0, cmd = null, outcome = null, since = 0, contact = null, ctrl = 0;
  const set = (s, at) => { if (s !== status) { status = s; since = at; } };
  for (const e of leg.events) {
    if (e.t > t + 1e-6) break;
    if (e.type === 'status') set(e.status, e.t);
    else if (e.type === 'contact') { touches++; contact = e; set('PINNED', e.t); }
    else if (e.type === 'shuttle') shuttles++;
    else if (e.type === 'jackknife') { jk++; set('JACKKNIFED', e.t); }
    else if (e.type === 'command') cmd = e;
    else if (e.type === 'controller') ctrl++;
    else if (e.type === 'finish') { outcome = e.outcome; set(e.outcome, e.t); }
  }
  return { status, touches, shuttles, jk, cmd, outcome, since, contact, ctrl };
}

// the current phase of a multi-phase course (phase events), or null
// the phase a multi-phase leg is in at time t: {k, n, dir, label}, from the phase events and track.phases; null otherwise
export function phaseInfo(leg, t) {
  const tp = leg.track.phases || [];
  const ev = phaseAt(leg, t);
  if (!ev && tp.length < 2) return null;
  const k = ev ? (ev.phase ?? 1) : 1, p = tp[k - 1] || {};
  return { k, n: ev?.of ?? tp.length, dir: p.direction > 0 ? 'forward' : p.direction < 0 ? 'reverse' : '', label: ev?.label || p.name || '' };
}
export const phaseText = (ph) => `Phase ${ph.k}${ph.n ? `/${ph.n}` : ''}${ph.dir ? ` · ${ph.dir}` : ''}`;

// 0-based phase index at leg time t: a {type: "phase", phase: k} event starts phase k (1-based)
export function phaseIndexAt(leg, t) { const p = phaseAt(leg, t); return p ? Math.max(0, (p.phase ?? 1) - 1) : 0; }

export function phaseAt(leg, t) {
  let p = null;
  for (const e of leg.phases || []) { if (e.t <= t + 1e-6) p = e; else break; }
  return p;
}

// the model turn in progress at leg time t (asked, answer not applied yet), or null
export function thinkingAt(leg, t) {
  for (const u of leg.turns || []) if (u.t_reply > u.t + 0.05 && u.t <= t + 1e-6 && t < u.t_reply - 1e-6) return u;
  return null;
}

export function callAt(leg, t) {
  let best = null;
  for (const c of leg.calls) { if (c.t0 <= t + 1e-6) best = c; else break; }
  return best;
}

// arclength of the trailer axle along the centreline
export function axleS(leg, st) {
  const g = leg.track.geometry, spec = leg.spec;
  if (!g.centerline?.length || !g.s?.length) return 0;
  const hx = st.x - spec.hitch_offset * Math.cos(st.th1), hy = st.y - spec.hitch_offset * Math.sin(st.th1);
  const ax = hx - spec.trailer_length * Math.cos(st.th2), ay = hy - spec.trailer_length * Math.sin(st.th2);
  const C = g.centerline, S = g.s;
  let best = 1e18, bs = 0;
  for (let k = 0; k < C.length - 1; k++) {
    const x0 = C[k][0], y0 = C[k][1], dx = C[k + 1][0] - x0, dy = C[k + 1][1] - y0;
    const L2 = dx * dx + dy * dy || 1e-9;
    const u = Math.max(0, Math.min(1, ((ax - x0) * dx + (ay - y0) * dy) / L2));
    const px = x0 + u * dx - ax, py = y0 + u * dy - ay, d = px * px + py * py;
    if (d < best) { best = d; bs = S[k] + u * (S[k + 1] - S[k]); }
  }
  return bs;
}

// progress (0..1) from the start pose to the bay centre
export function progressOf(leg, st) {
  // multi-phase courses (turnaround): along the phase routes, the same measure as relay/turnaround.py
  const pt = (s) => phaseProgressS(leg.track, leg.spec, s, phaseIndexAt(leg, leg.trace.t[s.idx] ?? 0));
  if (pt(st) !== null) {
    if (leg._s0 === undefined) leg._s0 = pt(sampleLeg(leg, 0));
    return Math.max(0, Math.min(1, (pt(st) - leg._s0) / Math.max(1e-6, leg.track.goal_s - leg._s0)));
  }
  if (!leg.track.geometry.centerline?.length) return leg.L.metrics?.progress ?? 0;
  if (leg._s0 === undefined) leg._s0 = axleS(leg, sampleLeg(leg, 0));
  const s = axleS(leg, st);
  return Math.max(0, Math.min(1, (s - leg._s0) / Math.max(1e-6, leg.track.goal_s - leg._s0)));
}

export function fmtClock(t) {
  t = Math.max(0, t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}
