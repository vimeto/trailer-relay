// Grid (small multiples on the shared clock) and Inspect (one run: perception, decision log, charts).
// Pure DOM/SVG; the app passes in each run's live state (computed once per frame in app.js).
import { sampleLeg, legStatusAt, thinkingAt, phaseInfo, phaseText, phaseIndexAt, gazeAt, axleS, fmtClock } from './data.js';
import { rigShape, hitchDeg, critOf, perception, railLines, coursePoints, hasWallBoundary } from './geom2d.js';
import { headAt, lookWords, viewGeometry } from './driver.js';
import { MODE_NAMES, CONTROL_NAMES, CLOCK_NAMES, TEMPLATE_NAMES } from './labels.js';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
const f2 = (v) => (Math.round(v * 100) / 100).toString();
const STATUS_TEXT = {
  READY: 'Ready', REVERSING: 'Reversing', PULLING_FORWARD: 'Pulling fwd', STOPPED: 'Stopped', COMMITTED: 'Committed',
  JACKKNIFED: 'Jackknifed', PINNED: 'Pinned', FINISHED: 'Parked', TIME_UP: 'Time up', DNF: 'DNF', WAITING: 'Waiting',
  LOOKING: 'Looking', THINKING: 'Thinking',
};
export const chipHTML = (status, text) => `<span class="chip ${esc(status)}">${esc(text ?? STATUS_TEXT[status] ?? status)}</span>`;

// ------------------------------------------------------------------ course drawing (shared)
// A track is drawn in its layout orientation (same as the 3D view), y up: svg = (X, -Y).
const mappers = new WeakMap();
export function mapperFor(track, rot = 0) {
  let m = mappers.get(track);
  if (m && m.rot === rot) return m;
  const c = Math.cos(rot), s = Math.sin(rot);
  const P = (x, y) => [c * x - s * y, -(s * x + c * y)];
  const pts = coursePoints(track.geometry).map(([x, y]) => P(x, y));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const box = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  const pt = (x, y) => { const [a, b] = P(x, y); return `${f2(a)},${f2(b)}`; };
  const pts2 = (arr) => arr.map(([x, y]) => pt(x, y)).join(' ');
  m = { rot, P, pt, pts: pts2, box, size: Math.max(box.x1 - box.x0, box.y1 - box.y0) };
  m.course = courseSVG(track, m);
  mappers.set(track, m);
  return m;
}
// the exit zone of a turnaround course is also its dock_corners: draw it once, as the bay
const sameQuad = (a, b) => !!(a && b && a.length === b.length && a.every((p, k) => Math.hypot(p[0] - b[k][0], p[1] - b[k][1]) < 1e-3));
function courseSVG(track, m) {
  const g = track.geometry;
  let h = '';
  const walled = hasWallBoundary(g);
  if (walled) for (const w of g.walls) h += `<polygon class="road" points="${m.pts(w)}"/>`;
  else if (g.left_rail?.length && g.right_rail?.length) h += `<polygon class="road" points="${m.pts([...g.left_rail, ...g.right_rail.slice().reverse()])}"/>`;
  // phase guide routes (turnaround): faint dashes
  for (const r of g.routes || []) if (r?.length > 1) h += `<polyline class="guide" points="${m.pts(r)}"/>`;
  for (const z of g.zones || []) if (z.polygon?.length > 2 && !sameQuad(z.polygon, g.dock_corners)) h += `<polygon class="zone ${esc(String(z.kind || ''))}" points="${m.pts(z.polygon)}"/>`;
  if (g.dock_corners?.length) h += `<polygon class="bay" points="${m.pts(g.dock_corners)}"/>`;
  for (const line of railLines(g)) h += `<polyline class="rail" points="${m.pts(line)}"/>`;
  return h;
}
function zoneLabels(track, m, size) {
  let h = '';
  for (const z of track.geometry.zones || []) {
    const p = z.polygon || [];
    if (p.length < 3) continue;
    const cx = p.reduce((a, q) => a + q[0], 0) / p.length, cy = p.reduce((a, q) => a + q[1], 0) / p.length;
    const [x, y] = m.P(cx, cy);
    h += `<text class="zone-label" x="${f2(x)}" y="${f2(y)}" text-anchor="middle" style="font-size:${f2(size)}px">${esc(z.name || z.kind)}</text>`;
  }
  return h;
}
// trailer-axle path of a leg, mapped, one point per `stride` trace samples (cached on the leg)
function pathPts(leg, m, stride) {
  const key = `${m.rot}|${stride}`;
  leg._paths ||= {};
  const hit = leg._paths[key];
  if (hit && hit.n === leg.trace.t.length) return hit.pts;
  const tr = leg.trace, sp = leg.spec, out = [];
  for (let k = 0; k < tr.t.length; k += stride) {
    const hx = tr.x[k] - sp.hitch_offset * Math.cos(tr.th1[k]), hy = tr.y[k] - sp.hitch_offset * Math.sin(tr.th1[k]);
    out.push(m.pt(hx - sp.trailer_length * Math.cos(tr.th2[k]), hy - sp.trailer_length * Math.sin(tr.th2[k])));
  }
  leg._paths[key] = { n: tr.t.length, pts: out };
  return out;
}
function rigSVG(m, leg, st, ls, ended) {
  const r = rigShape(st, leg.spec);
  let h = `<polygon class="trailer" points="${m.pts(r.trailer)}"/><polygon class="vehicle" points="${m.pts(r.vehicle)}"/>`;
  const hd = Math.abs(hitchDeg(st)), crit = critOf(leg.spec);
  const st8 = ls.status;
  if (!ended && (st8 === 'JACKKNIFED' || st8 === 'COMMITTED' || hd >= crit - 3)) {
    const [x, y] = m.pt(...r.hitch).split(',');
    h += `<circle class="hring ${st8 === 'JACKKNIFED' ? 'bad' : 'warn'}" cx="${x}" cy="${y}" r="${f2(Math.max(0.8, leg.spec.combo_length * 0.12))}"/>`;
  }
  return h;
}
const legOf = (run, heats, heatLeg) => run.legs[heats ? Math.min(heatLeg, run.legs.length - 1) : run.live?.active ?? 0];
function legState(run, leg, T) {
  const tl = Math.max(0, Math.min(T - leg.start, leg.time));
  const started = T >= leg.start, ended = !leg.running && T >= leg.end;
  const st = sampleLeg(leg, tl);
  return { tl, started, ended, st, ls: legStatusAt(leg, tl) };
}

// ------------------------------------------------------------------ GRID
export function createGrid(root, { onOpen }) {
  let sig = null;
  const tiles = new Map();
  function layout(n, aspect) {
    const W = root.clientWidth || 1200, H = root.clientHeight || 700, gap = 10;
    let best = null;
    const maxCols = Math.max(1, Math.min(n, Math.floor((W + gap) / (250 + gap))));
    for (let cols = 1; cols <= maxCols; cols++) {
      const rows = Math.ceil(n / cols);
      const tw = (W - gap * (cols - 1)) / cols, th = (H - gap * (rows - 1)) / rows;
      const mw = tw - 20, mh = th - 72;
      if (mw <= 0) break;
      const sc = mh > 60 ? Math.min(mw / aspect, mh) : -1e3 + mh;
      if (!best || sc > best.sc) best = { cols, rows, sc, th };
    }
    const fixed = best.th < 170;
    root.style.setProperty('--cols', String(best.cols));
    root.style.setProperty('--rowh', fixed ? '190px' : `${Math.floor(best.th)}px`);
  }
  function build(ctx) {
    root.innerHTML = '';
    tiles.clear();
    ctx.runs.forEach((run, k) => {
      const el = document.createElement('div');
      el.className = 'gtile';
      el.dataset.run = run.runId;
      el.style.setProperty('--c', run.color);
      el.innerHTML = `<div class="gt-head"><i class="dot"></i><span class="gt-name"></span><span class="gt-var"></span><span class="chip"></span></div>
        <svg class="gt-map" preserveAspectRatio="xMidYMid meet"><g class="course"></g><polyline class="path${run.variant.control === 'discrete' ? ' dashed' : ''}"/><g class="touches"></g><g class="rig"></g></svg>
        <div class="gt-bar"><b></b></div>
        <div class="gt-foot"><span class="leg"></span><span>Progress<b class="prog"></b></span><span>Stops<b class="stops"></b></span><span>Touch<b class="touch"></b></span><span class="fin"><span class="finl">Time</span><b class="final"></b></span><span class="legs">${run.legs.map(() => '<b></b>').join('')}</span></div>`;
      el.querySelector('.gt-name').textContent = run.lab?.model || ctx.shortLabel(run.label);
      el.querySelector('.gt-var').textContent = run.variantTag || '';
      el.title = `${run.lab?.full || run.label} — click to inspect`;
      el.onclick = () => onOpen(k);
      root.appendChild(el);
      const q = (s) => el.querySelector(s);
      tiles.set(run.runId, { el, svg: q('svg'), course: q('.course'), path: q('.path'), touches: q('.touches'), rig: q('.rig'), chip: q('.chip'), bar: q('.gt-bar b'),
        leg: q('.leg'), prog: q('.prog'), stops: q('.stops'), touch: q('.touch'), final: q('.final'), finl: q('.finl'), pips: [...el.querySelectorAll('.legs b')], legIdx: -1 });
    });
  }
  function render(ctx) {
    const s = ctx.runs.map((r) => r.runId).join('|') + `|${ctx.runs.length}`;
    if (s !== sig) { sig = s; build(ctx); }
    if (!ctx.runs.length) return;
    const fr = ctx.runs[ctx.focus] || ctx.runs[0];
    const fl = legOf(fr, ctx.heats, ctx.heatLeg);
    const mF = fl ? mapperFor(fl.track, ctx.rotOf(fl.i)) : null;
    const lk = `${root.clientWidth}x${root.clientHeight}|${ctx.runs.length}|${mF?.size}`;
    if (mF && root.dataset.lk !== lk) { root.dataset.lk = lk; layout(ctx.runs.length, (mF.box.x1 - mF.box.x0 + 6) / (mF.box.y1 - mF.box.y0 + 6)); }
    ctx.runs.forEach((run, k) => {
      const t = tiles.get(run.runId);
      if (!t) return;
      t.el.classList.toggle('focus', k === ctx.focus);
      const leg = legOf(run, ctx.heats, ctx.heatLeg);
      if (!leg) { ctx.setChip(t.chip, 'WAITING'); return; }
      const m = mapperFor(leg.track, ctx.rotOf(leg.i));
      if (t.legIdx !== leg.i) {
        t.legIdx = leg.i;
        const pad = m.size * 0.04;
        t.svg.setAttribute('viewBox', `${f2(m.box.x0 - pad)} ${f2(m.box.y0 - pad)} ${f2(m.box.x1 - m.box.x0 + 2 * pad)} ${f2(m.box.y1 - m.box.y0 + 2 * pad)}`);
        t.course.innerHTML = m.course;
        t.bay = t.course.querySelector('.bay');
      }
      const { tl, started, ended, st, ls } = legState(run, leg, ctx.T);
      const stride = leg.trace.t.length > 1500 ? 6 : 3;
      const pts = pathPts(leg, m, stride);
      t.path.setAttribute('points', started ? pts.slice(0, Math.floor(st.idx / stride) + 1).join(' ') : '');
      let th = '';
      if (started) for (const e of leg.events) {
        if (e.t > tl) break;
        if (e.type === 'contact' && e.x !== undefined) { const [x, y] = m.pt(e.x, e.y).split(','); th += `<circle class="touch" cx="${x}" cy="${y}" r="${f2(m.size * 0.012)}"/>`; }
      }
      if (t.touches.innerHTML !== th) t.touches.innerHTML = th;
      t.rig.innerHTML = rigSVG(m, leg, st, ls, ended);
      t.rig.style.opacity = started ? (ended ? 0.6 : 1) : 0.35;
      const parked = ended && leg.outcome === 'FINISHED';
      t.bay?.classList.toggle('parked', parked);
      const think = !ended && thinkingAt(leg, tl);
      const status = !started ? 'WAITING' : ended ? leg.outcome : ls.status;
      if (think && ctx.realtime(run)) ctx.setChip(t.chip, 'THINKING', `Thinking ${(tl - think.t).toFixed(0)} s`);
      else ctx.setChip(t.chip, status);
      const prog = ended && leg.outcome === 'FINISHED' ? 1 : started ? ctx.progressOf(leg, st) : 0;
      t.bar.style.width = `${(prog * 100).toFixed(1)}%`;
      const ph = started ? phaseInfo(leg, tl) : null;
      ctx.setText(t.leg, `Leg ${leg.i + 1} · ${leg.spec.name || leg.spec.key}${ph ? ` · phase ${ph.k}/${ph.n}` : ''}`);
      ctx.setText(t.prog, `${Math.round(prog * 100)}%`);
      ctx.setText(t.stops, leg.trace.v ? String(leg.stopsCum[st.idx] ?? 0) : '–');
      ctx.setText(t.touch, String(ls.touches));
      const fin = ended ? (leg.outcome === 'FINISHED' ? fmtClock(leg.time) : (STATUS_TEXT[leg.outcome] || leg.outcome)) : started ? fmtClock(tl) : '–';
      ctx.setText(t.final, fin);
      ctx.setText(t.finl, ended ? 'Final' : 'Time');
      t.final.className = `final ${ended ? (leg.outcome === 'FINISHED' ? 'ok' : 'bad') : ''}`;
      t.pips.forEach((b, i) => {
        const L = run.legs[i];
        if (!L) { b.className = ''; return; }
        const done = !L.running && ctx.T >= L.end;
        b.className = done ? (L.outcome === 'FINISHED' ? 'done' : 'fail') : (ctx.T >= L.start && L === leg ? 'live' : '');
      });
    });
  }
  return { render, reset: () => { sig = null; } };
}

// ------------------------------------------------------------------ INSPECT
export function createInspect(els, { onPick }) {
  let zoom = 'rig', staticKey = null, logKey = null, chartKey = null, notesKey = null;
  els.zoom.onclick = () => { zoom = zoom === 'rig' ? 'course' : 'rig'; els.zoom.textContent = zoom === 'rig' ? 'Whole course' : 'Zoom to rig'; staticKey = null; };
  els.run.onchange = () => onPick(Number(els.run.value));
  let cursor = null, mask = null, nowTurn = null;

  function header(ctx, run, leg, info) {
    els.swatch.style.background = run.color;
    ctx.setText(els.name, run.name || run.label);
    els.name.title = run.lab?.full || '';
    const v = run.variant;
    const ph = leg ? phaseInfo(leg, info.tl) : null;
    const tpl = TEMPLATE_NAMES[leg?.track?.template] || '';
    ctx.setText(els.vari, !leg ? 'not started' : `Leg ${leg.i + 1} of ${ctx.nLegs} · ${leg.spec.name}${leg.spec.trailer_name ? ` + ${leg.spec.trailer_name}` : ''}${tpl ? `, ${tpl.toLowerCase()}` : ''}${ph ? ` · ${phaseText(ph)}${ph.label ? ` (${ph.label})` : ''}` : ''} · ${info.tl.toFixed(1)} s into the leg`);
    const ctl = String(v.control || '');
    els.vari.title = run.isAuto ? 'Reference planner (plans on the full course map)'
      : `${MODE_NAMES[v.mode] || v.mode} · ${ctl.startsWith('continuous') ? `${CONTROL_NAMES.continuous} · ${CLOCK_NAMES[ctl.endsWith('realtime') ? 'realtime' : 'paused']}` : CONTROL_NAMES.discrete}`;
    const sel = els.run;
    const sig = ctx.runs.map((r) => r.runId).join('|');
    if (sel.dataset.sig !== sig) {
      sel.dataset.sig = sig;
      sel.innerHTML = ctx.runs.map((r, k) => `<option value="${k}">${esc(r.name || ctx.shortLabel(r.label))}</option>`).join('');
    }
    if (sel.value !== String(ctx.focus)) sel.value = String(ctx.focus);
  }

  function legend(run) {
    const v = String(run.variant.obs);
    const items = [];
    if (run.isAuto) items.push('<span class="note">Reference planner: it plans on the full course map, so there is no driver report to show.</span>');
    else if (run.driver) {
      items.push(`<span><i style="background:${run.color}"></i>direct view (gaze ±60°)</span><span><i style="background:${run.color};opacity:.35"></i>corner of the eye (close by)</span><span><i style="background:#fff4d6"></i>mirror / reversing-camera view, while looked at</span><span><i class="dotl" style="background:#fff"></i>eye · mirrors</span><span><i style="background:rgba(255,255,255,.45)"></i>head turning: sees nothing</span>`);
      if (run.variant.control.includes('realtime')) items.push('<span><i style="background:var(--think)"></i>driven while the model was thinking</span>');
      items.push('<span class="note">View geometry from the recorded gaze and the rig in relay/perception.py. The driver\'s noisy readings are not recorded, so they are not shown.</span>');
    } else {
      if (v !== 'sensors') items.push(`<span><i class="dotl" style="background:#fff"></i>rail points in the report${v.startsWith('cone') ? ' (inside the view cone)' : ''}</span>`);
      if (!v.startsWith('cone')) items.push('<span><i style="background:var(--ok)"></i>proximity ray</span><span><i style="background:var(--bad)"></i>ray on a rail</span><span><i style="background:var(--info)"></i>route arrow</span>');
      else items.push('<span><i style="background:rgba(255,255,255,.6)"></i>view cone (look)</span>');
      if (run.variant.control.includes('realtime')) items.push('<span><i style="background:var(--think)"></i>driven while the model was thinking</span>');
      items.push('<span class="note">Rebuilt from the recorded pose with the rules of relay/observe.py; the record stores poses, not the reports.</span>');
    }
    return items.join('');
  }

  function mapRender(ctx, run, leg, info) {
    const svg = els.svg;
    const m = mapperFor(leg.track, ctx.rotOf(leg.i));
    const key = `${run.runId}|${leg.i}|${zoom}|${m.rot}`;
    if (staticKey !== key) {
      staticKey = key;
      svg.innerHTML = `<g class="course">${m.course}</g><g class="zl"></g><g class="dyn"></g>`;
      els.legend.innerHTML = legend(run);
      svg.style.setProperty('--c', run.color);
      svg._dyn = svg.querySelector('.dyn');
      svg._zl = svg.querySelector('.zl');
    }
    const { st, tl, ls, ended, started } = info;
    const sp = leg.spec, rig = rigShape(st, sp);
    // window: around the rig, or the whole course
    const W = svg.clientWidth || 800, H = svg.clientHeight || 500, asp = W / Math.max(1, H);
    let cx, cy, half;
    if (zoom === 'rig') {
      const c = m.P((rig.hitch[0] + rig.axle[0]) / 2, (rig.hitch[1] + rig.axle[1]) / 2);
      cx = c[0]; cy = c[1];
      half = Math.max((sp.view_radius ?? 14) + sp.combo_length * 0.6, sp.combo_length * 1.6);
    } else {
      cx = (m.box.x0 + m.box.x1) / 2; cy = (m.box.y0 + m.box.y1) / 2;
      half = Math.max((m.box.x1 - m.box.x0) / 2 / asp, (m.box.y1 - m.box.y0) / 2) * 1.06;
    }
    const vw = 2 * half * asp, vh = 2 * half;
    svg.setAttribute('viewBox', `${f2(cx - vw / 2)} ${f2(cy - vh / 2)} ${f2(vw)} ${f2(vh)}`);
    const u = vh / H; // metres per pixel
    if (svg._zlu !== Math.round(u * 1000)) { svg._zlu = Math.round(u * 1000); svg._zl.innerHTML = zoneLabels(leg.track, m, 11 * u); }
    let h = '';
    if (started) {
      const stride = leg.trace.t.length > 1500 ? 4 : 2;
      const pts = pathPts(leg, m, stride);
      // driven while thinking (realtime clock): the rig kept rolling between the question and the answer
      if (ctx.realtime(run)) {
        for (const tu of leg.turns) {
          if (tu.t > tl || tu.t_reply <= tu.t + 0.05) continue;
          const i0 = idxAt(leg, tu.t), i1 = idxAt(leg, Math.min(tu.t_reply, tl));
          if (i1 > i0) h += `<polyline class="thinkpath${tu.t_reply > tl ? ' now' : ''}" points="${pts.slice(Math.floor(i0 / stride), Math.floor(i1 / stride) + 1).join(' ')}"/>`;
        }
      }
      h += `<polyline class="path" style="stroke:${run.color}" points="${pts.slice(0, Math.floor(st.idx / stride) + 1).join(' ')}"/>`;
      for (const e of leg.events) {
        if (e.t > tl) break;
        if (e.type === 'contact' && e.x !== undefined) { const [x, y] = m.pt(e.x, e.y).split(','); h += `<circle class="touch" cx="${x}" cy="${y}" r="${f2(4 * u)}"/>`; }
      }
      if (!run.isAuto && !ended && run.driver) h += driverSVG(m, leg, st, tl, u);
      else if (!run.isAuto && !ended) {
        const P = perception(leg, st, { obs: run.variant.obs, gazeDeg: gazeAt(leg, tl), progressS: axleS(leg, st), phase: phaseIndexAt(leg, tl) });
        if (P.radius) { const [x, y] = m.pt(...P.center).split(','); h += `<circle class="radius" cx="${x}" cy="${y}" r="${f2(P.radius)}"/>`; }
        if (P.cone) {
          const { eye, heading, fov, range } = P.cone, n = 18, a0 = heading - fov * Math.PI / 360;
          const arc = [eye];
          for (let k = 0; k <= n; k++) { const a = a0 + (fov * Math.PI / 180) * k / n; arc.push([eye[0] + range * Math.cos(a), eye[1] + range * Math.sin(a)]); }
          h += `<polygon class="cone" points="${m.pts(arc)}"/>`;
        }
        // distances only on the three closest hits; labels are sized in screen pixels (u = metres per pixel)
        const near = new Set(P.rays.filter((r) => r.hit).sort((p, q) => p.d - q.d).slice(0, 3));
        let lab = '';
        const placedL = [];
        for (const r of P.rays) {
          const [x1, y1] = m.pt(...r.from).split(','), [x2, y2] = m.pt(...r.to).split(',');
          h += `<line class="ray ${r.hit ? 'hit' : 'clear'}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
          if (near.has(r) && !placedL.some(([px, py]) => Math.hypot(px - x2, py - y2) < 26 * u)) placedL.push([+x2, +y2]), lab += `<text class="raylabel" x="${x2}" y="${f2(+y2 - 7 * u)}" text-anchor="middle" style="font-size:${f2(11 * u)}px;stroke-width:${f2(3 * u)}px">${r.d.toFixed(1)} m</text>`;
        }
        const pr = f2(Math.max(2.2 * u, 0.06));
        for (const p of P.points) { const [x, y] = m.pt(...p).split(','); h += `<circle class="mapdot" cx="${x}" cy="${y}" r="${pr}"/>`; }
        if (P.route) {
          const [x1, y1] = m.pt(...P.route.from).split(','), [x2, y2] = m.pt(...P.route.to).split(',');
          h += `<defs><marker id="ahead" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L6,3 L0,6 z" fill="#7cc4ff"/></marker></defs><line class="route" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#ahead)"/>`;
        }
        h += lab;
      }
    }
    h += `<g style="opacity:${started ? (ended ? 0.6 : 1) : 0.4}"><polygon class="trailer" points="${m.pts(rig.trailer)}"/><polygon class="vehicle" style="fill:${run.color}" points="${m.pts(rig.vehicle)}"/></g>`;
    if (run.driver && started && !ended) h += driverDots(m, leg, st, tl, u);
    const hd = Math.abs(hitchDeg(st));
    if (started && !ended && (ls.status === 'JACKKNIFED' || ls.status === 'COMMITTED' || hd >= critOf(sp) - 3)) {
      const [x, y] = m.pt(...rig.hitch).split(',');
      h += `<circle class="hring ${ls.status === 'JACKKNIFED' ? 'bad' : 'warn'}" cx="${x}" cy="${y}" r="${f2(Math.max(10 * u, sp.combo_length * 0.1))}"/>`;
    }
    svg._dyn.innerHTML = h;
  }

  // driver's-seat view geometry (obs "driver"): direct cone, the watched mirror's / camera's view, eye, mirrors
  function driverSVG(m, leg, st, tl, u) {
    const hs = headAt(leg, tl, st.idx);
    if (!hs) return '';
    const geo = viewGeometry(leg.rig, leg.spec, st, hs);
    const wedge = (v) => {
      const arc = [v.origin];
      for (let k = 0; k <= 24; k++) { const a = v.a0 + (v.a1 - v.a0) * k / 24; arc.push([v.origin[0] + v.range * Math.cos(a), v.origin[1] + v.range * Math.sin(a)]); }
      return m.pts(arc);
    };
    let h = '';
    if (!hs.turning) for (const v of geo.periph || []) h += `<polygon class="vcone periph" points="${wedge(v)}"/>`;
    if (geo.direct) h += `<polygon class="vcone direct${hs.turning ? ' dim' : ''}" points="${wedge(geo.direct)}"/>`;
    if (geo.aid && geo.aid.range > 0) h += `<polygon class="vcone aid ${geo.aid.kind}" points="${wedge(geo.aid)}"/>`;
    const [ex, ey] = m.pt(...geo.eye).split(',');
    if (geo.aid?.via) { const [x, y] = m.pt(...geo.aid.via).split(','); h += `<line class="sight" x1="${ex}" y1="${ey}" x2="${x}" y2="${y}"/>`; }
    else { const L = Math.max(1.5, 12 * u), g = geo.gaze; const [x, y] = m.pt(geo.eye[0] + L * Math.cos(g), geo.eye[1] + L * Math.sin(g)).split(','); h += `<line class="sight${hs.turning ? ' dim' : ''}" x1="${ex}" y1="${ey}" x2="${x}" y2="${y}"/>`; }
    return h;
  }
  // eye and mirror dots go on top of the rig
  function driverDots(m, leg, st, tl, u) {
    const hs = headAt(leg, tl, st.idx);
    if (!hs) return '';
    const geo = viewGeometry(leg.rig, leg.spec, st, hs);
    const r = f2(Math.max(2.4 * u, 0.07));
    let h = '';
    for (const mm of geo.mirrors) { const [x, y] = m.pt(...mm.pos).split(','); h += `<circle class="mirror${hs.at === mm.name ? ' on' : ''}" cx="${x}" cy="${y}" r="${r}"/>`; }
    if (geo.screen && hs.at === 'camera') { const [x, y] = m.pt(...geo.screen).split(','); h += `<circle class="mirror on" cx="${x}" cy="${y}" r="${r}"/>`; }
    const [ex, ey] = m.pt(...geo.eye).split(',');
    return `${h}<circle class="eye" cx="${ex}" cy="${ey}" r="${f2(Math.max(3 * u, 0.09))}"/>`;
  }

  function idxAt(leg, t) {
    const T = leg.trace.t;
    let lo = 0, hi = T.length - 1;
    if (t <= T[0]) return 0;
    if (t >= T[hi]) return hi;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid] <= t) lo = mid; else hi = mid; }
    return lo;
  }
  const distBetween = (leg, a, b) => Math.max(0, leg.dist[idxAt(leg, b)] - leg.dist[idxAt(leg, a)]);

  function turnHTML(ctx, run, leg, tu, tl) {
    const realtime = ctx.realtime(run);
    const thinking = tu.t_reply > tl;
    const think = Math.max(0, Math.min(tl, tu.t_reply) - tu.t);
    const wall = tu.latency != null ? `${tu.latency.toFixed(1)} s wall` : '';
    let bar;
    if (tu.turn === 1 && tu.t === 0 && !realtime) bar = `<div class="thinkbar paused"><i style="width:100%"></i><span>planning before the start${wall ? ` · ${wall}` : ''} · clock paused</span></div>`;
    else if (realtime && tu.t_reply > tu.t + 0.05) {
      const meanwhile = distBetween(leg, tu.t, Math.min(tu.t_reply, tl));
      bar = `<div class="thinkbar"><i style="width:${Math.min(100, think / Math.max(10, tu.think || think) * 100).toFixed(0)}%"></i><span data-live="${thinking ? 1 : 0}">${thinking ? 'thinking' : 'thought'} ${think.toFixed(1)} s · rig drove ${meanwhile.toFixed(1)} m meanwhile${tu.late ? ' · answer came after the leg ended' : ''}</span></div>`;
    } else bar = `<div class="thinkbar paused"><i style="width:100%"></i><span>clock paused while thinking${wall ? ` · ${wall}` : ''}</span></div>`;
    const calls = thinking ? [] : tu.calls;
    const chips = calls.map((c) => `<span class="chip ctrl">${esc(c.name)}</span>`).join('')
      + calls.flatMap((c) => c.woke || []).map((w) => `<span class="chip ${/leg over/.test(w) ? 'FINISHED' : /contact|jackknife|error|committed/.test(w) ? 'PINNED' : 'READY'}">woke: ${esc(w)}</span>`).join('');
    let say = '';
    const text = tu.text.trim(), th = tu.reasoning.trim();
    if (!thinking && text) say = `<div class="say">${esc(text)}</div>`;
    else if (!thinking && th) {
      const secs = th.split(/\n\s*\n(?=\*\*)/).map((x) => x.trim()).filter(Boolean);
      const last = secs[secs.length - 1] || th, mm = last.match(/^\*\*(.+?)\*\*\s*/);
      say = `<div class="say thought">${mm ? `<b>${esc(mm[1])}</b> ` : ''}${esc((mm ? last.slice(mm[0].length) : last).replace(/\*\*/g, ''))}</div>`;
    }
    let body = '';
    for (const c of calls) {
      body += `<div class="call">${ctx.resultLine(c)}</div>`;
      const code = String(c.args?.code ?? '').replace(/\t/g, '  ').replace(/^\s*\n/, '');
      if (code.trim()) body += `<details><summary>${c.kept ? 'controller still driving' : c.name === 'run_program' ? 'program' : 'controller code'} · ${code.trimEnd().split('\n').length} lines</summary><pre class="code">${esc(code.trimEnd())}</pre></details>`;
      if (c.stdout?.trim()) body += `<pre class="code out">${esc(c.stdout.trim().split('\n').slice(-8).join('\n'))}</pre>`;
      if (c.error) body += `<pre class="code out err">${esc(String(c.error).trim().split('\n').slice(-6).join('\n'))}</pre>`;
    }
    return `<div class="turn${thinking ? ' now' : ''}" data-turn="${tu.turn}"><div class="meta"><span class="num">turn ${tu.turn}</span><span class="num">t ${tu.t.toFixed(1)} s${tu.t_reply > tu.t + 0.05 ? ` → ${tu.t_reply.toFixed(1)} s` : ''}</span><span class="sp"></span>${thinking ? '<span class="chip THINKING">thinking</span>' : chips}</div>${bar}${say}${body}</div>`;
  }

  // learning studies: the notes this leg read before driving, and the note it left (accepted, or rejected with reasons)
  function notesHTML(run, leg) {
    const L = (run.rec.legs || []).find((x, j) => (x.index ?? j) === leg.i);
    if (!L || (!L.notes_in && !L.note)) return '';
    const nin = L.notes_in || [], n = L.note;
    const read = nin.length ? `<details class="ln-read"><summary>Read ${nin.length} note${nin.length === 1 ? '' : 's'} before driving</summary>${nin.map((x) => `<blockquote><cite>${esc(x.from)}</cite>${esc(x.text)}</blockquote>`).join('')}</details>`
      : '<div class="ln-none">No notes to read before this leg.</div>';
    let wrote = '<div class="ln-none">Wrote no note (a transfer attempt, or the study does not ask).</div>';
    if (n) {
      const rej = (n.rejections || []).map((why, i) => `<div class="ln-draft"><span class="nbadge bad">✕ draft ${i + 1}</span> ${esc((why || []).join('; '))}${n.drafts?.[i] ? `<p>${esc(n.drafts[i])}</p>` : ''}</div>`).join('');
      wrote = n.accepted
        ? `<div class="ln-note ok"><span class="nbadge ok">✓ note accepted</span><p>${esc(n.text)}</p>${rej ? `<details><summary>${n.rejections.length} rejected draft${n.rejections.length === 1 ? '' : 's'} first</summary>${rej}</details>` : ''}</div>`
        : `<div class="ln-note bad"><span class="nbadge bad">✕ every draft rejected: nothing passed on</span>${rej}</div>`;
    }
    const tag = run.rec.learning ? `${run.rec.learning.condition} · ${run.rec.learning.role === 'transfer' ? 'transfer' : `step ${run.rec.learning.step}`}` : '';
    return `<section class="lnotes"><div class="eyebrow">Notes${tag ? ` · ${esc(tag)}` : ''}</div>${read}${wrote}</section>`;
  }

  function logRender(ctx, run, leg, info) {
    const tl = info.tl;
    const vis = leg ? leg.turns.filter((u) => u.t <= tl + 1e-6) : [];
    const cur = vis.find((u) => u.t_reply > tl);
    const key = `${run.runId}|${leg?.i}|${vis.length}|${cur?.turn ?? ''}|${leg?.turns.length}`;
    ctx.setText(els.logTitle, run.isAuto ? 'Decision log · reference planner' : `Decision log · ${vis.length} turn${vis.length === 1 ? '' : 's'} so far${leg ? ` of ${leg.turns.length}` : ''}`);
    if (key !== logKey) {
      logKey = key;
      els.log.innerHTML = run.isAuto ? '<div class="log-empty">The reference planner has no model turns: it replans every step on the full map.</div>'
        : !vis.length ? '<div class="log-empty">No turns yet at this time.</div>'
          : vis.slice().reverse().map((u) => turnHTML(ctx, run, leg, u, tl)).join('');
      nowTurn = cur ? { el: els.log.querySelector('.turn.now .thinkbar span'), bar: els.log.querySelector('.turn.now .thinkbar i'), tu: cur } : null;
    } else if (nowTurn?.el) {
      const think = Math.max(0, tl - nowTurn.tu.t);
      nowTurn.el.textContent = `thinking ${think.toFixed(1)} s · rig drove ${distBetween(leg, nowTurn.tu.t, tl).toFixed(1)} m meanwhile`;
      nowTurn.bar.style.width = `${Math.min(100, think / Math.max(10, nowTurn.tu.think || think) * 100).toFixed(0)}%`;
    }
  }

  function chartsRender(ctx, run, leg, info) {
    const svg = els.charts;
    const W = Math.max(300, svg.clientWidth || 1200), H = Math.max(120, svg.clientHeight || 170);
    const key = `${run.runId}|${leg.i}|${W}x${H}|${leg.time}|${leg.trace.t.length}`;
    const T = Math.max(1, leg.time), x0 = 64, x1 = W - 10;
    const x = (t) => x0 + (t / T) * (x1 - x0);
    if (key !== chartKey) {
      chartKey = key;
      const tr = leg.trace, n = tr.t.length, step = Math.max(1, Math.floor(n / 900));
      const hs = [], vs = [];
      for (let k = 0; k < n; k += step) {
        hs.push([tr.t[k], hitchDeg({ th1: tr.th1[k], th2: tr.th2[k] })]);
        if (tr.v) vs.push([tr.t[k], tr.v[k]]);
      }
      const crit = critOf(leg.spec), mech = leg.spec.max_hitch_deg;
      const hmax = Math.min(mech + 5, Math.max(30, crit + 5, ...hs.map((p) => Math.abs(p[1]) + 5)));
      const vmax = Math.max(0.5, ...vs.map((p) => Math.abs(p[1]))) * 1.1;
      const top = 8, mid = Math.round(H * 0.56), bot = H - 18;
      const yh = (v) => top + (1 - (v + hmax) / (2 * hmax)) * (mid - top - 8);
      const yv = (v) => mid + 4 + (1 - (v + vmax) / (2 * vmax)) * (bot - mid - 4);
      let g = '';
      // think windows (realtime) and stops
      if (ctx.realtime(run)) for (const tu of leg.turns) if (tu.t_reply > tu.t + 0.05) g += `<rect x="${f2(x(tu.t))}" y="${top}" width="${f2(Math.max(1, x(Math.min(T, tu.t_reply)) - x(tu.t)))}" height="${bot - top}" fill="rgba(201,167,255,.14)"/>`;
      if (tr.v) {
        let s0 = null;
        for (let k = 0; k < n; k++) {
          const stopped = Math.abs(tr.v[k]) < 0.02;
          if (stopped && s0 === null) s0 = tr.t[k];
          if ((!stopped || k === n - 1) && s0 !== null) { if (tr.t[k] - s0 > 0.25) g += `<rect x="${f2(x(s0))}" y="${mid + 4}" width="${f2(Math.max(1, x(tr.t[k]) - x(s0)))}" height="${bot - mid - 4}" fill="rgba(255,255,255,.07)"/>`; s0 = null; }
        }
      }
      g += `<text x="0" y="${f2(yh(0) + 4)}">hitch °</text><text x="0" y="${f2((mid + bot) / 2 + 6)}">speed m/s</text>`;
      g += `<line x1="${x0}" x2="${x1}" y1="${f2(yh(0))}" y2="${f2(yh(0))}" stroke="rgba(255,255,255,.16)"/>`;
      for (const s of [1, -1]) g += `<line x1="${x0}" x2="${x1}" y1="${f2(yh(s * crit))}" y2="${f2(yh(s * crit))}" stroke="rgba(245,176,65,.5)" stroke-dasharray="4 4"/>`;
      g += `<text class="mono" x="${x0 - 6}" y="${f2(yh(crit) + 4)}" text-anchor="end">+${crit.toFixed(0)}</text><text class="mono" x="${x0 - 6}" y="${f2(yh(-crit) + 4)}" text-anchor="end">−${crit.toFixed(0)}</text>`;
      g += `<polyline fill="none" stroke="${run.color}" stroke-width="1.6" points="${hs.map(([t, v]) => `${f2(x(t))},${f2(yh(v))}`).join(' ')}"/>`;
      if (vs.length) {
        g += `<line x1="${x0}" x2="${x1}" y1="${f2(yv(0))}" y2="${f2(yv(0))}" stroke="rgba(255,255,255,.16)"/>`;
        g += `<polyline fill="none" stroke="#7cc4ff" stroke-width="1.5" points="${vs.map(([t, v]) => `${f2(x(t))},${f2(yv(v))}`).join(' ')}"/>`;
        g += `<text class="mono" x="${x0 - 6}" y="${f2(yv(vmax / 1.1) + 8)}" text-anchor="end">+${(vmax / 1.1).toFixed(1)}</text><text class="mono" x="${x0 - 6}" y="${f2(yv(-vmax / 1.1))}" text-anchor="end">−${(vmax / 1.1).toFixed(1)}</text>`;
      } else g += `<text x="${x0 + 8}" y="${f2((mid + bot) / 2 + 4)}">no speed trace in this record (discrete commands)</text>`;
      // events
      const COL = { contact: '#ff6b5e', jackknife: '#ff8a3d', shuttle: '#f5b041', controller: '#f4f1ea', controller_error: '#ff6b5e', wake: '#c9a7ff', phase: '#7cc4ff', finish: '#4fd18b' };
      for (const e of leg.events) {
        const c = COL[e.type];
        if (!c) continue;
        const xe = f2(x(e.t));
        if (e.type === 'phase') g += `<line x1="${xe}" x2="${xe}" y1="${top}" y2="${bot}" stroke="${c}" stroke-dasharray="2 3"/><text x="${f2(+xe + 3)}" y="${top + 9}" style="fill:${c}">${esc(e.label || `phase ${e.phase}`)}</text>`;
        else if (e.type === 'controller') g += `<line x1="${xe}" x2="${xe}" y1="${bot - 6}" y2="${bot + 2}" stroke="${c}" stroke-width="1.5"/>`;
        else if (e.type === 'shuttle') g += `<line x1="${xe}" x2="${xe}" y1="${mid}" y2="${mid + 6}" stroke="${c}" stroke-width="1.5"/>`;
        else g += `<line x1="${xe}" x2="${xe}" y1="${top}" y2="${bot}" stroke="${c}" stroke-width="1.2" stroke-opacity=".8"/>`;
      }
      g += `<text class="mono" x="${x0}" y="${H - 3}">0:00</text><text class="mono" x="${x1}" y="${H - 3}" text-anchor="end">${fmtClock(T)}</text>`;
      g += `<text x="${f2((x0 + x1) / 2)}" y="${H - 3}" text-anchor="middle">▮ contact · ▮ jackknife · ┃ controller install · ┃ gear change${ctx.realtime(run) ? ' · ░ model thinking' : ''} · ░ stopped</text>`;
      g += `<rect class="mask" y="0" height="${H - 14}" fill="rgba(13,16,14,.72)"/><line class="cursor" y1="0" y2="${H - 14}" stroke="#fff" stroke-width="1.5"/>`;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.innerHTML = g;
      cursor = svg.querySelector('.cursor'); mask = svg.querySelector('.mask');
    }
    const xc = f2(x(info.tl));
    cursor.setAttribute('x1', xc); cursor.setAttribute('x2', xc);
    mask.setAttribute('x', xc); mask.setAttribute('width', f2(Math.max(0, x1 - +xc + 10)));
  }

  function render(ctx) {
    const run = ctx.runs[ctx.focus];
    if (!run) { els.log.innerHTML = '<div class="log-empty">No runs selected.</div>'; return; }
    const leg = legOf(run, ctx.heats, ctx.heatLeg);
    const info = leg ? legState(run, leg, ctx.T) : { tl: 0, started: false, ended: false };
    header(ctx, run, leg, info);
    const chips = [];
    if (leg) {
      const think = !info.ended && thinkingAt(leg, info.tl);
      if (think && ctx.realtime(run)) chips.push(chipHTML('THINKING', `Thinking ${(info.tl - think.t).toFixed(0)} s`));
      chips.push(chipHTML(!info.started ? 'WAITING' : info.ended ? leg.outcome : info.ls.status));
      chips.push(`<span class="chip ctrl">hitch ${hitchDeg(info.st).toFixed(0)}°</span>`);
      if (run.driver && info.started && !info.ended) { const hs = headAt(leg, info.tl, info.st.idx); if (hs) chips.push(`<span class="chip ctrl look">looking: ${esc(lookWords(hs))}</span>`); }
    }
    // notes card: redrawn only when the run or leg changes (keeps <details> open while playing)
    const nk = `${run.runId}|${leg?.i}|${run.rec.legs?.length}`;
    if (els.notes && nk !== notesKey) { notesKey = nk; const h = leg ? notesHTML(run, leg) : ''; els.notes.innerHTML = h; els.notes.hidden = !h; }
    const ch = chips.join('');
    if (els.chips.innerHTML !== ch) els.chips.innerHTML = ch;
    if (!leg) return;
    mapRender(ctx, run, leg, info);
    logRender(ctx, run, leg, info);
    chartsRender(ctx, run, leg, info);
  }
  return { render, reset: () => { staticKey = logKey = chartKey = notesKey = null; } };
}
