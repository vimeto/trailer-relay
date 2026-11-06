// Results view: the review tool for the whole study, read from runs/results.json (relay/results.py) and the learning
// studies (runs/studies/*.json). Five sections: Overview (the calibration matrix), Failure modes, Learning, Notes, Runs.
// Every number links through to the runs behind it; runs open in Broadcast / Grid / Inspect via ctx.openRuns.
// Plain DOM + SVG. Colours: one per model (labels.js), status colours for outcomes (style.css --ok / --bad / --warn).
import { MODE_NAMES, CONTROL_NAMES, CLOCK_NAMES, runLabel, uniqueLabels, describe, modelShort, modelName, modelColor, modelRank, learningTag, LEVELS, LEVEL_DEFAULT, LEVEL_NAMES, LEVEL_SHORT,
  LEVEL_NOTES, CONDITION_NAMES, STEP_NAMES, NUMBERS_NAMES, SECTION_NAMES, VEHICLE_NAMES, courseSet, relayName, obsName } from './labels.js';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const VEHICLES = ['car', 'tractor', 'mower', 'truck'];
const OUTCOMES = [
  { key: 'parked', label: 'Parked', color: 'var(--ok)', glyph: '✓' },
  { key: 'rail', label: 'Stuck on the rail', color: 'var(--bad)', glyph: '✕' },
  { key: 'time', label: 'Out of time', color: 'var(--warn)', glyph: '◷' },
  { key: 'other', label: 'Other', color: 'var(--other)', glyph: '–' },
];
const OUT = Object.fromEntries(OUTCOMES.map((o) => [o.key, o]));
const PARTS = [
  { key: 'own front corner', label: 'Own front corner', color: '#9085e9' },
  { key: 'own rear corner', label: 'Own rear corner', color: '#c3bcf3' },
  { key: 'trailer front', label: 'Trailer front', color: '#efa7c2' },
  { key: 'trailer rear', label: 'Trailer rear', color: '#d55181' },
];
const GAZE = [
  { key: 'camera', label: 'Camera screen', color: '#c98500' },
  { key: 'mirror', label: 'Mirror', color: '#9085e9' },
  { key: 'ahead', label: 'Ahead', color: '#7cc4ff' },
  { key: 'side', label: 'Side', color: '#d55181' },
  { key: 'back', label: 'Over the shoulder', color: '#a7aba3' },
];
const CONDITIONS = {
  self: { name: CONDITION_NAMES.self, step: STEP_NAMES.self, what: 'One driver drives the same course again and again and reads only its own earlier notes.' },
  chain: { name: CONDITION_NAMES.chain, step: STEP_NAMES.chain, what: 'A line of fresh drivers shares one notebook: each reads it, drives, and rewrites it.' },
  pool: { name: CONDITION_NAMES.pool, step: STEP_NAMES.pool, what: 'Rounds of drivers drive at the same time; the next round reads every note the previous one left.' },
};
// ?sec= keys stay stable; the names are the reader's
const SECTIONS = [
  ['overview', SECTION_NAMES.overview, 'Parked legs by model and difficulty'],
  ['failures', SECTION_NAMES.failures, 'Rail hits, timeouts, blind spots'],
  ['learning', SECTION_NAMES.learning, 'Does a note help the next try?'],
  ['notes', SECTION_NAMES.notes, 'The notes, kept and rejected'],
  ['runs', SECTION_NAMES.runs, 'Filter, compare, replay'],
];

export function outcomeKey(L) {
  if (!L) return null;
  if (L.outcome === 'FINISHED') return 'parked';
  if (L.outcome == null) return null; // still driving
  const m = L.mode || (/(touch)/i.test(L.reason || '') ? 'rail-hit loop' : L.outcome === 'TIME_UP' ? 'out of time' : null);
  return m === 'rail-hit loop' ? 'rail' : m === 'out of time' ? 'time' : 'other';
}
const outcomeText = (L) => {
  const k = outcomeKey(L);
  if (!k) return 'driving';
  if (k === 'parked') return `Parked · ${Number(L.time).toFixed(0)} s`;
  return `${k === 'other' ? (L.mode || L.outcome || 'other') : OUT[k].label} · ${Math.round(100 * (L.progress ?? 0))}%`;
};
export function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), w = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, (c - w) / d), Math.min(1, (c + w) / d)];
}
const pct = (x, d = 0) => (x == null || Number.isNaN(x) ? '—' : `${(100 * x).toFixed(d)}%`);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const uniq = (xs) => [...new Set(xs)];
const legScore = (L) => (L.outcome === 'FINISHED' ? 1 : Math.max(0, Math.min(1, L.progress ?? 0)));
const fmtWhen = (c) => (c ? c.replace('T', ' ').slice(5, 16) : '');
const studyTagOf = (L) => { const s = String(L.study || ''); return `${CONDITION_NAMES[L.condition] || L.condition} · ${s.slice(9, 11)}:${s.slice(11, 13)}`; };
const setShort = (seed) => courseSet(seed).replace('Course set ', 'set ');
const fmtTime = (t) => (t == null ? '—' : `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, '0')}`);

// ------------------------------------------------------------------ data
function buildModel(R, index) {
  const legsBy = new Map();
  for (const L of R?.legs || []) { if (!legsBy.has(L.run_id)) legsBy.set(L.run_id, []); legsBy.get(L.run_id).push(L); }
  for (const ls of legsBy.values()) ls.sort((a, b) => a.index - b.index);
  const rows = new Map();
  const add = (src, kind) => {
    const d = describe(src);
    const id = d.runId;
    if (!id || id === '?') return;
    const prev = rows.get(id);
    if (prev && !(prev.live && !src.live)) { if (kind === 'results') prev.res = src; return; }
    rows.set(id, { id, d, kind, src, live: !!src.live, partial: !!src.partial, res: prev?.res || (kind === 'results' ? src : null) });
  };
  for (const it of index?.runs || []) add(it, 'index');
  for (const r of R?.runs || []) add(r, 'results');
  const all = [...rows.values()].map((row) => {
    const { d, src, res } = row;
    const legs = res ? (legsBy.get(row.id) || []) : (src.legs || []).map((l, i) => ({ index: i, ...l }));
    const T = src.totals || {};
    const nLegs = res ? res.legs : T.legs ?? legs.length;
    const parked = res ? res.parked : T.legs_finished ?? legs.filter((l) => l.outcome === 'FINISHED').length;
    return {
      ...row, legs, nLegs, parked, file: res?.file || src.file,
      progress: res ? res.progress : T.relay_progress, touches: res ? res.touches : T.touches, time: res ? res.time : T.relay_time,
      model: modelName(d), short: modelShort(d), color: modelColor(d), rank: modelRank(d), scripted: d.scripted, auto: d.auto,
      relay: d.relayId, relayName: d.relayName, seed: d.seed, obs: d.obs, level: d.obs === 'driver' ? d.level || 'v1' : null,
      learning: d.learning, created: d.created, continuous: d.control === 'continuous',
    };
  });
  const labs = uniqueLabels(all.map((r) => r.src));
  all.forEach((r, i) => { r.lab = labs[i]; });
  all.sort((a, b) => (b.created || '').localeCompare(a.created || ''));
  const byId = new Map(all.map((r) => [r.id, r]));
  // results rows only (finished continuous harness>=4 runs, with failure modes)
  const res = all.filter((r) => r.res);
  const studies = new Map();
  for (const s of R?.studies || []) studies.set(s.study, s);
  return { all, byId, res, studies: [...studies.values()].sort((a, b) => b.study.localeCompare(a.study)), generated: R?.generated };
}

// ------------------------------------------------------------------ the view
// Deep links into the notes reader: ?view=results&sec=notes&study=<id>[&compare=<id>][&leg=<vehicle|1-4>][&driver=<n>]
function fromURL() {
  const p = new URLSearchParams(location.search);
  const lg = (p.get('leg') || '').toLowerCase();
  const leg = VEHICLES.includes(lg) ? VEHICLES.indexOf(lg) : /^[1-4]$/.test(lg) ? Number(lg) - 1 : 0;
  const drv = /^\d+$/.test(p.get('driver') || '') ? Number(p.get('driver')) - 1 : null;
  return { a: p.get('study') || null, b: p.get('compare') || null, leg, member: drv };
}

export function createResults(root, ctx) {
  const st = {
    section: 'overview', R: null, M: null, sig: '', err: null,
    split: 'none',
    fm: { obs: 'driver', relay: 'standard', levels: new Set(), models: new Set(), learning: false, scripted: false },
    ln: { level: null, numbers: 'all', seed: null },
    notes: fromURL(),
    runs: { q: '', f: {}, ids: null, idsTitle: '', sort: ['created', -1], sel: new Set() },
  };
  root.innerHTML = `<nav class="res-nav" aria-label="Results sections">
      <div class="res-brand"><div class="eyebrow">Results</div><div class="res-gen" id="res-gen">—</div></div>
      ${SECTIONS.map(([k, n, s]) => `<button data-sec="${k}"><b>${n}</b><span>${s}</span></button>`).join('')}
      ${ctx.static ? '' : '<div class="res-navfoot">Refresh the data with<br><code>python -m relay results</code></div>'}
    </nav>
    <div class="res-main" id="res-main" tabindex="-1"></div>
    <div class="res-tip" id="res-tip" hidden></div>`;
  const main = root.querySelector('#res-main');
  const tip = root.querySelector('#res-tip');
  // the nav's Runs tab lists every run; a chart's drill-down (showRuns) is what narrows it
  for (const b of root.querySelectorAll('[data-sec]')) b.onclick = () => { if (b.dataset.sec === 'runs') st.runs.ids = null; go(b.dataset.sec); };

  // one tooltip for every chart: elements carry data-tip (HTML built from escaped text)
  root.addEventListener('pointermove', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (!t || !root.contains(t)) { tip.hidden = true; return; }
    if (tip._for !== t) { tip._for = t; tip.innerHTML = t.dataset.tip; }
    tip.hidden = false;
    const r = root.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
    if (x + w > r.width - 8) x = e.clientX - r.left - w - 14;
    if (y + h > r.height - 8) y = e.clientY - r.top - h - 14;
    tip.style.transform = `translate(${x}px, ${y}px)`;
  });
  root.addEventListener('pointerleave', () => { tip.hidden = true; });
  // drill-down: anything with data-ids lists its runs
  main.addEventListener('click', (e) => {
    const a = e.target.closest('[data-open]');
    if (a) { e.preventDefault(); e.stopPropagation(); ctx.openRuns(a.dataset.open.split(','), a.dataset.view || 'broadcast', { leg: a.dataset.leg != null ? Number(a.dataset.leg) : undefined }); return; }
    const t = e.target.closest('[data-ids]');
    if (t && main.contains(t)) { e.preventDefault(); showRuns(t.dataset.ids.split(',').filter(Boolean), t.dataset.title || 'Selected runs'); }
  });

  function go(sec, opts = {}) {
    if (!SECTIONS.some(([k]) => k === sec)) sec = 'overview';
    st.section = sec;
    for (const b of root.querySelectorAll('[data-sec]')) b.classList.toggle('on', b.dataset.sec === sec);
    Object.assign(st.notes, opts.notes || {});
    render();
    main.scrollTop = 0;
    ctx.onSection?.(sec);
  }
  function showRuns(ids, title) {
    st.runs.ids = ids; st.runs.idsTitle = title; st.runs.sel = new Set();
    go('runs');
  }

  // The runs directory listing (python -m http.server / relay serve) says whether the digest and the studies exist,
  // so a site without them does not log a 404; without a listing (static host) the files are simply tried.
  async function listing() {
    if (ctx.static) return null; // a static host has no directory listing: the exported files are all there
    try {
      const r = await fetch(ctx.runsDir, { cache: 'no-store' });
      if (!r.ok) return null;
      const t = await r.text();
      return /<a href=/i.test(t) ? t : null;
    } catch (e) { return null; }
  }
  async function load(force = false) {
    const dir = await listing();
    if (dir !== null && !dir.includes('href="results.json"')) {
      st.err = 'no runs/results.json yet';
      if (!st.M) st.M = buildModel(null, ctx.getIndex?.());
      return !st.R;
    }
    try {
      const R = await ctx.fetchJSON(`${ctx.runsDir}results.json`);
      // studies may be newer than the digest (written as each attempt finishes)
      if (dir === null || dir.includes('href="studies/"')) try {
        const names = await ctx.fetchJSON(`${ctx.runsDir}studies/index.json`);
        const fresh = await Promise.all((names || []).map((n) => ctx.fetchJSON(`${ctx.runsDir}studies/${n}`).catch(() => null)));
        const by = new Map((R.studies || []).map((s) => [s.study, s]));
        for (const s of fresh) if (s && (!by.has(s.study) || (s.runs?.length || 0) >= (by.get(s.study).runs?.length || 0))) by.set(s.study, s);
        R.studies = [...by.values()];
      } catch (e) { /* no studies dir */ }
      const sig = `${R.generated}|${(R.studies || []).map((s) => `${s.study}:${s.runs?.length}`).join(',')}|${ctx.indexSig?.() || ''}`;
      if (!force && sig === st.sig) return false;
      st.sig = sig; st.R = R; st.err = null;
      st.M = buildModel(R, ctx.getIndex?.());
      return true;
    } catch (e) {
      st.err = e.message;
      if (!st.M) st.M = buildModel(null, ctx.getIndex?.());
      return true;
    }
  }
  async function refresh() { if (await load()) render(); }

  // ------------------------------------------------------------------ rendering helpers
  const card = (title, caption, body, extra = '') => `<section class="rcard ${extra}"><header><h3>${title}</h3>${caption ? `<p class="cap">${caption}</p>` : ''}</header>${body}</section>`;
  const dot = (c) => `<i class="mdot" style="background:${c}"></i>`;
  const legend = (items, kind = 'rect') => `<div class="rlegend">${items.map((o) => `<span><i class="${kind}" style="background:${o.color}"></i>${esc(o.label)}</span>`).join('')}</div>`;
  const idsOf = (rows) => uniq(rows.map((r) => r.run_id || r.id)).join(',');
  const segBar = (segs, total, { ids = {}, title = '', h = 14, labels = false } = {}) => {
    // 100% stacked bar with 2px surface gaps; each segment is its own hit target
    if (!total) return '<div class="sbar empty"><span>no legs</span></div>';
    let x = 0;
    const parts = segs.filter((s) => s.n > 0).map((s) => {
      const w = 100 * s.n / total;
      const out = `<b style="left:${x}%;width:calc(${w}% - 2px);background:${s.color}" ${ids[s.key] ? `data-ids="${ids[s.key]}" data-title="${esc(`${title} · ${s.label}`)}"` : ''} data-tip="${esc(`<b>${s.n}</b> of ${total} (${pct(s.n / total)})<br>${esc(s.label)}${title ? `<br><span class=dim>${esc(title)}</span>` : ''}`)}">${labels && w >= 12 ? `<span>${pct(s.n / total)}</span>` : ''}</b>`;
      x += w;
      return out;
    });
    return `<div class="sbar${labels ? ' lab' : ''}" style="height:${h}px">${parts.join('')}</div>`;
  };

  function header(title, sub, controls = '') {
    return `<div class="res-head"><div><h2>${title}</h2>${sub ? `<p class="sub">${sub}</p>` : ''}</div>${controls ? `<div class="res-controls">${controls}</div>` : ''}</div>`;
  }
  const segCtl = (name, opts, cur) => `<div class="rseg" data-ctl="${name}">${opts.map(([v, l]) => `<button data-v="${esc(v)}" class="${String(cur) === String(v) ? 'on' : ''}">${esc(l)}</button>`).join('')}</div>`;
  const chips = (name, vals, set, lab = (v) => v) => `<div class="rchips" data-ctl="${name}">${vals.map((v) => `<button data-v="${esc(v)}" class="${set.has(v) ? 'on' : ''}">${esc(lab(v))}</button>`).join('')}</div>`;
  const selectCtl = (name, opts, cur, label) => `<label class="rsel"><span>${esc(label)}</span><select data-ctl="${name}">${opts.map(([v, l]) => `<option value="${esc(v)}" ${String(cur) === String(v) ? 'selected' : ''}>${esc(l)}</option>`).join('')}</select></label>`;

  function render() {
    const M = st.M;
    root.querySelector('#res-gen').textContent = M?.generated ? `data as of ${fmtWhen(M.generated)}` : 'no results yet';
    if (!M) { main.innerHTML = '<div class="res-empty">Loading results…</div>'; return; }
    let html = '';
    try {
      html = { overview, failures, learning, notes, runs: runsTable }[st.section]();
    } catch (e) {
      console.error(e);
      html = `<div class="res-empty">Could not draw this section: ${esc(e.message)}</div>`;
    }
    if (st.err && st.section !== 'runs') html = `<div class="res-warn">The results file could not be read (${esc(st.err)}).${ctx.static ? '' : ' Run <code>python -m relay results</code>.'} The run list below still works.</div>${html}`;
    main.innerHTML = html;
    wire();
  }

  // control wiring (after each render)
  function wire() {
    for (const el of main.querySelectorAll('.rseg[data-ctl], .rchips[data-ctl]')) {
      for (const b of el.querySelectorAll('button')) b.onclick = () => control(el.dataset.ctl, b.dataset.v, el.classList.contains('rchips'));
    }
    for (const el of main.querySelectorAll('select[data-ctl], input[data-ctl]')) {
      const ev = el.tagName === 'INPUT' && el.type === 'search' ? 'input' : 'change';
      el.addEventListener(ev, () => control(el.dataset.ctl, el.type === 'checkbox' ? el.checked : el.value));
    }
    if (st.section === 'runs') wireRuns();
  }
  function control(name, v, multi) {
    const toggle = (set) => { if (set.has(v)) set.delete(v); else set.add(v); };
    if (name === 'split') st.split = v;
    else if (name === 'fm-obs') st.fm.obs = v;
    else if (name === 'fm-relay') st.fm.relay = v;
    else if (name === 'fm-levels') toggle(st.fm.levels);
    else if (name === 'fm-models') toggle(st.fm.models);
    else if (name === 'fm-learning') st.fm.learning = !!v;
    else if (name === 'fm-scripted') st.fm.scripted = !!v;
    else if (name === 'ln-level') st.ln.level = v;
    else if (name === 'ln-numbers') st.ln.numbers = v;
    else if (name === 'ln-seed') st.ln.seed = v === '' ? null : Number(v);
    else if (name === 'nt-a') st.notes.a = v;
    else if (name === 'nt-b') st.notes.b = v || null;
    else if (name === 'nt-leg') st.notes.leg = Number(v);
    else if (name === 'nt-member') st.notes.member = v === '' ? null : Number(v);
    else if (name === 'rt-q') { st.runs.q = v; renderRunsBody(); return; }
    else if (name.startsWith('rt-f-')) { const k = name.slice(5); if (v === '') delete st.runs.f[k]; else st.runs.f[k] = v; }
    else if (multi) return;
    const y = main.scrollTop;
    render();
    main.scrollTop = y;
    ctx.onState?.();
  }

  // ------------------------------------------------------------------ 1. overview: the calibration matrix
  function matrixRows(rows) {
    return rows.filter((r) => r.res && r.obs === 'driver' && r.relayName === 'standard' && !r.learning);
  }
  function cellStats(rows) {
    const legs = rows.flatMap((r) => r.legs.filter((l) => l.outcome != null));
    const k = legs.filter((l) => l.outcome === 'FINISHED').length;
    return { k, n: legs.length, runs: rows.length, legs };
  }
  function rateBar(k, n, color, { h = 8 } = {}) {
    const [lo, hi] = wilson(k, n), p = n ? k / n : 0;
    return `<svg class="rate" viewBox="0 0 100 ${h + 6}" preserveAspectRatio="none" aria-hidden="true"><rect class="track" x="0" y="3" width="100" height="${h}" rx="2"/>`
      + (n ? `<rect x="0" y="3" width="${Math.max(0.01, 100 * p)}" height="${h}" rx="2" fill="${color}"/><line class="ci" x1="${100 * lo}" x2="${100 * hi}" y1="${3 + h / 2}" y2="${3 + h / 2}"/><line class="ci" x1="${100 * lo}" x2="${100 * lo}" y1="1" y2="${h + 5}"/><line class="ci" x1="${100 * hi}" x2="${100 * hi}" y1="1" y2="${h + 5}"/>` : '')
      + '</svg>';
  }
  function overview() {
    const M = st.M;
    const mx = matrixRows(M.res);
    const models = uniq(mx.filter((r) => !r.scripted).map((r) => r.model)).sort((a, b) => modelRank({ spec: '', model: a }) - modelRank({ spec: '', model: b }) || a.localeCompare(b));
    const rankOf = (m) => mx.find((r) => r.model === m)?.rank ?? 99;
    models.sort((a, b) => rankOf(a) - rankOf(b));
    const bases = uniq(mx.filter((r) => r.scripted).map((r) => r.model)).sort();
    const seeds = uniq(mx.map((r) => r.seed)).sort();
    const cell = (model, level) => {
      const rows = mx.filter((r) => r.model === model && r.level === level);
      const s = cellStats(rows);
      const color = rows[0]?.color || '#888';
      if (!rows.length) return '<td class="mcell none"><span>not run</span></td>';
      const title = `${model} · ${obsName('driver', level)}`;
      const p = s.n ? s.k / s.n : 0, [lo, hi] = wilson(s.k, s.n);
      let sub = '';
      if (st.split === 'vehicle' || st.split === 'seed') {
        const keys = st.split === 'vehicle' ? VEHICLES : seeds;
        sub = `<div class="msplit">${keys.map((key) => {
          const legs = st.split === 'vehicle' ? s.legs.filter((l) => l.vehicle === key) : rows.filter((r) => r.seed === key).flatMap((r) => r.legs.filter((l) => l.outcome != null));
          const kk = legs.filter((l) => l.outcome === 'FINISHED').length;
          const ids = st.split === 'vehicle' ? idsOf(rows) : idsOf(rows.filter((r) => r.seed === key));
          return `<div class="msrow" data-ids="${ids}" data-title="${esc(`${title} · ${st.split === 'vehicle' ? key : courseSet(key)}`)}" data-tip="${esc(`<b>${kk}/${legs.length}</b> legs parked<br>${esc(st.split === 'vehicle' ? VEHICLE_NAMES[key] || key : courseSet(key))}`)}"><span class="msk">${st.split === 'vehicle' ? key : setShort(key)}</span>${rateBar(kk, legs.length, color, { h: 5 })}<span class="msn">${legs.length ? `${kk}/${legs.length}` : '—'}</span></div>`;
        }).join('')}</div>`;
      }
      const runsBySeed = seeds.map((sd) => [sd, rows.filter((r) => r.seed === sd).length]).filter(([, n]) => n).map(([sd, n]) => `${setShort(sd)} ×${n}`).join(' · ');
      return `<td class="mcell"><div class="mc" data-ids="${idsOf(rows)}" data-title="${esc(title)}" data-tip="${esc(`<b>${s.k} of ${s.n}</b> legs parked (${pct(p)})<br>95% interval ${pct(lo)}–${pct(hi)}<br>${s.runs} run${s.runs === 1 ? '' : 's'} · ${runsBySeed}<br><span class=dim>click to list the runs</span>`)}">
        <div class="mc-top"><b>${pct(p)}</b><span>${s.k}/${s.n} legs</span></div>${rateBar(s.k, s.n, color)}
        <div class="mc-sub">${s.runs} run${s.runs === 1 ? '' : 's'} · ${runsBySeed}</div></div>${sub}</td>`;
    };
    const row = (m, muted) => {
      const r0 = mx.find((r) => r.model === m);
      return `<tr class="${muted ? 'muted' : ''}"><th scope="row">${dot(r0.color)}<span>${esc(m)}</span></th>${LEVELS.map((lv) => cell(m, lv)).join('')}</tr>`;
    };
    const table = !mx.length ? '<div class="res-empty">No driver\'s-seat runs on the standard courses yet.</div>'
      : `<table class="matrix"><thead><tr><th></th>${LEVELS.map((lv) => `<th scope="col" class="${lv === LEVEL_DEFAULT ? 'def' : ''}"><b>${LEVEL_NAMES[lv]}</b>${lv === LEVEL_DEFAULT ? '<em class="deftag">default</em>' : ''}<span>${LEVEL_NOTES[lv]}</span></th>`).join('')}</tr></thead>
        <tbody>${models.map((m) => row(m, false)).join('')}${bases.length ? `<tr class="grp"><td colspan="${LEVELS.length + 1}">Baselines: simple scripted controllers, no language model</td></tr>${bases.map((m) => row(m, true)).join('')}` : ''}</tbody></table>`;
    const ctl = segCtl('split', [['none', 'All legs'], ['vehicle', 'By vehicle'], ['seed', 'By course set']], st.split);
    // summary strip
    const nRuns = M.all.length, live = M.all.filter((r) => r.live).length;
    const legsN = M.res.reduce((a, r) => a + r.legs.length, 0);
    const strip = `<div class="rstats"><div><b>${nRuns}</b><span>runs${live ? ` · ${live} still driving` : ''}</span></div><div><b>${M.res.length}</b><span>runs analysed (current simulator)</span></div><div><b>${legsN}</b><span>legs driven (one course each)</span></div><div><b>${M.studies.length}</b><span>learning-from-notes stud${M.studies.length === 1 ? 'y' : 'ies'}</span></div></div>`;
    return header('Scoreboard', 'Each run is a relay of four courses (legs): reverse a car, a tractor, a ride-on mower and a truck, each with its trailer, into a bay. The model drives from the driver\'s seat (mirrors, a reversing camera, rough depth). Columns: how hard the perception is. Runs that read notes from earlier drivers are left out here.', ctl)
      + strip
      + card('Legs parked, by model and difficulty', 'Each cell: the share of legs parked over all runs of that model at that difficulty, with a 95% interval as a whisker (few runs = wide whisker). Under the bar: runs per course set (A and B are two sets of the same four course shapes). Click a cell to list its runs.', table, 'wide')
      + otherVariants();
  }
  function otherVariants() {
    const M = st.M;
    const inMatrix = new Set(matrixRows(M.res).map((r) => r.id));
    const rest = M.all.filter((r) => !inMatrix.has(r.id) && !r.learning && !r.auto);
    const groups = new Map();
    for (const r of rest) {
      const k = [r.model, r.relayName, r.seed, r.obs, r.level, r.d.mode, r.d.control, r.d.clock].join('|');
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(r);
    }
    const gs = [...groups.values()].sort((a, b) => a[0].rank - b[0].rank || a[0].model.localeCompare(b[0].model) || String(a[0].relayName).localeCompare(String(b[0].relayName)) || (a[0].seed ?? 0) - (b[0].seed ?? 0));
    if (!gs.length) return '';
    const body = gs.map((rs) => {
      const r = rs[0];
      const legs = rs.flatMap((x) => x.legs.filter((l) => l.outcome != null));
      const k = legs.filter((l) => l.outcome === 'FINISHED').length;
      const prog = mean(rs.filter((x) => x.progress != null).map((x) => x.progress / Math.max(1, x.nLegs)));
      const tch = mean(rs.filter((x) => x.touches != null).map((x) => x.touches));
      const variant = [obsName(r.obs, r.level), MODE_NAMES[r.d.mode] || r.d.mode, r.d.control === 'continuous' ? CLOCK_NAMES[r.d.clock] || CONTROL_NAMES.continuous : CONTROL_NAMES.discrete].filter(Boolean).join(' · ');
      const where = `${relayName(r.relayName)} · ${courseSet(r.seed)}`;
      return `<tr data-ids="${idsOf(rs)}" data-title="${esc(`${r.model} · ${where} · ${variant}`)}" class="click"><th scope="row">${dot(r.color)}${esc(r.model)}</th><td>${esc(where)}</td><td>${esc(variant)}</td><td class="num">${rs.length}${rs.some((x) => x.live) ? ' <span class="dim">driving</span>' : ''}</td>`
        + `<td class="num"><span class="minibar">${rateBar(k, legs.length, r.color, { h: 5 })}</span>${legs.length ? `${k}/${legs.length}` : '—'}</td><td class="num">${pct(prog)}</td><td class="num">${tch == null ? '—' : tch.toFixed(1)}</td></tr>`;
    }).join('');
    return card('Other set-ups', 'Runs outside the scoreboard: other views of the course (an exact map instead of the driver\'s seat, a vision cone), other courses and earlier simulator versions. Legs parked, mean progress per leg and rail hits per run. Click a row for its runs.',
      `<div class="tscroll"><table class="rtable"><thead><tr><th>Model</th><th>Courses</th><th>Set-up</th><th class="num">Runs</th><th class="num">Legs parked</th><th class="num">Progress</th><th class="num">Rail hits</th></tr></thead><tbody>${body}</tbody></table></div>`, 'wide');
  }

  // ------------------------------------------------------------------ 2. failure modes
  function fmSlice() {
    const f = st.fm;
    return st.M.res.filter((r) => (f.obs === 'all' || r.obs === f.obs) && (f.relay === 'all' || r.relayName === f.relay)
      && (!f.levels.size || f.levels.has(r.level || '—')) && (!f.models.size || f.models.has(r.model))
      && (f.learning || !r.learning) && (f.scripted || !r.scripted));
  }
  function failures() {
    const M = st.M, f = st.fm;
    const base = M.res;
    const obsOpts = uniq(base.map((r) => r.obs)).sort();
    const relayOpts = uniq(base.map((r) => r.relayName)).sort();
    const levels = uniq(base.filter((r) => f.obs === 'all' || r.obs === f.obs).map((r) => r.level || '—')).sort((a, b) => LEVELS.indexOf(a) - LEVELS.indexOf(b));
    const modelsAll = uniq(base.filter((r) => f.scripted || !r.scripted).sort((a, b) => a.rank - b.rank || a.model.localeCompare(b.model)).map((r) => r.model));
    const ctl = `${selectCtl('fm-obs', [['all', 'any'], ...obsOpts.map((o) => [o, obsName(o)])], f.obs, 'Driver sees')}${selectCtl('fm-relay', [['all', 'any'], ...relayOpts.map((o) => [o, relayName(o)])], f.relay, 'Courses')}`
      + `<label class="rcheck"><input type="checkbox" data-ctl="fm-scripted" ${f.scripted ? 'checked' : ''}>baselines</label><label class="rcheck"><input type="checkbox" data-ctl="fm-learning" ${f.learning ? 'checked' : ''}>runs with notes</label>`;
    const ctl2 = `<div class="res-filters"><span class="eyebrow">Difficulty</span>${chips('fm-levels', levels, f.levels, (v) => LEVEL_SHORT[v] || v)}<span class="eyebrow">Model</span>${chips('fm-models', modelsAll, f.models)}<span class="dim">none ticked = all</span></div>`;
    const rows = fmSlice();
    const legs = rows.flatMap((r) => r.legs.filter((l) => l.outcome != null).map((l) => ({ ...l, r })));
    const head = header('How they fail', `How legs end and what goes wrong, for the runs picked below: ${rows.length} runs, ${legs.length} legs.`, ctl) + ctl2;
    if (!legs.length) return `${head}<div class="res-empty">No legs in this slice.</div>`;
    const models = uniq(rows.map((r) => r.model)).sort((a, b) => (rows.find((r) => r.model === a).rank - rows.find((r) => r.model === b).rank) || a.localeCompare(b));
    const colorOf = (m) => rows.find((r) => r.model === m).color;

    // A: outcome mix per model x level
    const groups = [];
    for (const m of models) for (const lv of uniq(legs.filter((l) => l.r.model === m).map((l) => l.r.level || '—')).sort((a, b) => LEVELS.indexOf(a) - LEVELS.indexOf(b))) groups.push([m, lv]);
    const mix = groups.map(([m, lv]) => {
      const ls = legs.filter((l) => l.r.model === m && (l.r.level || '—') === lv);
      const segs = OUTCOMES.map((o) => ({ ...o, n: ls.filter((l) => outcomeKey(l) === o.key).length }));
      const ids = Object.fromEntries(OUTCOMES.map((o) => [o.key, idsOf(ls.filter((l) => outcomeKey(l) === o.key))]));
      const title = `${m}${lv !== '—' ? ` · ${LEVEL_SHORT[lv] || lv}` : ''}`;
      return `<div class="hrow"><div class="hlab">${dot(colorOf(m))}<span>${esc(m)}</span><span class="dim">${esc(LEVEL_SHORT[lv] || lv)}</span></div>${segBar(segs, ls.length, { ids, title })}<div class="hn">${ls.length} legs</div></div>`;
    }).join('');
    const cardA = card('How legs end', '100% bars per model and difficulty. Green: parked. Red: stuck on the rail, the leg stopped after more than 8 rail touches (touch, pull forward, reverse, touch again). Amber: the leg\'s clock ran out. Grey: gave up, hit the command limit, jackknifed or errored. Click a segment for its runs.',
      legend(OUTCOMES.map((o) => ({ ...o, label: o.key === 'other' ? 'Other (gave up, command limit, jackknife, error)' : o.label }))) + `<div class="hrows">${mix}</div>`);

    // B: which part touched
    const contacts = legs.flatMap((l) => (l.contacts || []).map((c) => ({ ...c, l })));
    const partRows = models.map((m) => {
      const cs = contacts.filter((c) => c.l.r.model === m);
      const segs = PARTS.map((p) => ({ ...p, n: cs.filter((c) => c.part === p.key).length }));
      const failed = legs.filter((l) => l.r.model === m && l.outcome !== 'FINISHED' && l.culprit);
      const top = PARTS.map((p) => [p.label, failed.filter((l) => l.culprit === p.key).length]).sort((a, b) => b[1] - a[1])[0];
      const ids = Object.fromEntries(PARTS.map((p) => [p.key, idsOf(cs.filter((c) => c.part === p.key).map((c) => c.l.r))]));
      return `<div class="hrow"><div class="hlab">${dot(colorOf(m))}<span>${esc(m)}</span></div><div class="hbar">${segBar(segs, cs.length, { ids, title: `${m} · rail contacts` })}${top && top[1] ? `<small class="hnote">failed legs: mostly the ${esc(top[0].toLowerCase())} (${top[1]})</small>` : ''}</div><div class="hn">${cs.length} contacts</div></div>`;
    }).join('');
    const cardB = card('What touched the rail', 'Every rail contact in the slice by the part that touched: the tow vehicle\'s own corners (violet) or the trailer (pink). The note under each bar names the part that most often hit hardest on the legs that failed.',
      legend(PARTS) + `<div class="hrows">${partRows}</div>`);

    // C: where along the course
    const BINS = 20;
    const hist = models.map((m) => {
      const cs = contacts.filter((c) => c.l.r.model === m && c.frac != null);
      const b = Array.from({ length: BINS }, () => [0, 0]);
      for (const c of cs) b[Math.min(BINS - 1, Math.floor(c.frac * BINS))][c.bend ? 0 : 1]++;
      return { m, cs, b };
    });
    const ymax = Math.max(1, ...hist.flatMap((h) => h.b.map(([x, y]) => x + y)));
    const W = 520, H = 74, bw = W / BINS;
    const histSvg = hist.map(({ m, cs, b }) => {
      let g = `<line class="axis" x1="0" x2="${W}" y1="${H}" y2="${H}"/>`;
      b.forEach(([bend, str], i) => {
        const x = i * bw + 1, w = bw - 2;
        const hb = H * bend / ymax, hs = H * str / ymax;
        const tipTxt = `<b>${bend + str}</b> contacts at ${i * 5}–${i * 5 + 5}% of the course<br>${bend} on a bend · ${str} on a straight<br><span class=dim>${esc(m)}</span>`;
        if (bend) g += `<rect x="${x}" y="${H - hb}" width="${w}" height="${hb}" fill="var(--bend)" rx="1.5"/>`;
        if (str) g += `<rect x="${x}" y="${H - hb - hs - (bend ? 2 : 0)}" width="${w}" height="${hs}" fill="var(--straight)" rx="1.5"/>`;
        g += `<rect class="hit" x="${i * bw}" y="0" width="${bw}" height="${H}" data-tip="${esc(tipTxt)}" ${bend + str ? `data-ids="${idsOf(cs.filter((c) => Math.min(BINS - 1, Math.floor(c.frac * BINS)) === i).map((c) => c.l.r))}" data-title="${esc(`${m} · contacts at ${i * 5}–${i * 5 + 5}%`)}"` : ''}/>`;
      });
      return `<div class="hist"><div class="hlab">${dot(colorOf(m))}<span>${esc(m)}</span><span class="dim">${cs.length}</span></div><svg viewBox="0 0 ${W} ${H + 1}" preserveAspectRatio="none">${g}</svg></div>`;
    }).join('');
    const cardC = card('Where along the course', `Rail contacts by position along the course (start → bay, 5% bins; all legs pooled), stacked on a bend (blue) or on a straight (grey). Bars share one scale: the tallest bin holds ${ymax}.`,
      legend([{ label: 'on a bend', color: 'var(--bend)' }, { label: 'on a straight', color: 'var(--straight)' }]) + `<div class="hists">${histSvg}<div class="hist axis-row"><div class="hlab"></div><div class="xaxis"><span>start</span><span>25%</span><span>50%</span><span>75%</span><span>bay</span></div></div></div>`);

    // D: gaze at contacts vs overall
    const dl = legs.filter((l) => l.gaze);
    let cardD = '';
    if (dl.length) {
      const gz = models.map((m) => {
        const ls = dl.filter((l) => l.r.model === m);
        if (!ls.length) return '';
        const overall = GAZE.map((g) => ({ ...g, n: mean(ls.map((l) => l.gaze[g.key] || 0)) * 1000 }));
        const cs = ls.flatMap((l) => (l.contacts || []).filter((c) => c.gaze));
        const atC = GAZE.map((g) => ({ ...g, n: cs.filter((c) => c.gaze === g.key).length }));
        const tot = overall.reduce((a, s) => a + s.n, 0);
        return `<div class="gpair"><div class="hlab">${dot(colorOf(m))}<span>${esc(m)}</span></div><div class="gbars">
          <div class="hrow tight"><div class="glab">whole leg</div>${segBar(overall.map((s) => ({ ...s, n: Math.round(s.n) })), Math.round(tot), { title: `${m} · share of time looking`, h: 16, labels: true })}<div class="hn">${ls.length} legs</div></div>
          <div class="hrow tight"><div class="glab">at a contact</div>${segBar(atC, cs.length, { title: `${m} · gaze at the moment of a contact`, h: 16, labels: true })}<div class="hn">${cs.length} contacts</div></div></div></div>`;
      }).join('');
      cardD = card('Where the driver was looking', 'Driver\'s-seat legs only. Top bar: share of the leg spent looking at each place (mean over legs). Bottom bar: what the driver was looking at at the moment of each rail contact. A contact bar that differs from the time bar points at a blind spot.',
        legend(GAZE) + `<div class="gpairs">${gz}</div>`);
    }
    return head + `<div class="rgrid">${cardA}${cardB}</div><div class="rgrid">${cardC}${cardD}</div>`;
  }

  // ------------------------------------------------------------------ 3. learning
  function studyRuns(s) {
    // study rows enriched with the digest's run labels; order: train by step then member, transfer last
    return [...(s.runs || [])].sort((a, b) => (a.role === 'transfer') - (b.role === 'transfer') || a.step - b.step || a.member - b.member);
  }
  // a study names its model by spec ("codex:gpt-6.1-sol"): take the name and colour from a run of that model
  const SPEC_NAME = { 'codex:gpt-6.1-sol': 'GPT-6.1 Sol', 'codex:gpt-6.1-terra': 'GPT-6.1 Terra', 'codex:gpt-6-luna': 'GPT-6 Luna' };
  function mInfo(spec) {
    const r = st.M.all.find((x) => x.d.spec === spec);
    if (r) return { name: r.model, color: r.color };
    const d = { spec, model: SPEC_NAME[spec] || spec };
    return { name: modelName(d), color: modelColor(d) };
  }
  const relayScore = (run) => ({ parked: run.legs.filter((l) => l.outcome === 'FINISHED').length / Math.max(1, run.legs.length), progress: mean(run.legs.map(legScore)) });
  function baselineFor(model, level, seed) {
    const rows = st.M.res.filter((r) => r.d.spec === model && r.obs === 'driver' && r.level === level && r.seed === seed && r.relayName === 'standard' && !r.learning);
    if (!rows.length) return null;
    const legs = rows.flatMap((r) => r.legs);
    const k = legs.filter((l) => l.outcome === 'FINISHED').length;
    const per = rows.map((r) => ({ parked: r.legs.filter((l) => l.outcome === 'FINISHED').length / Math.max(1, r.legs.length), progress: mean(r.legs.map(legScore)) }));
    return { rows, n: rows.length, parked: k / Math.max(1, legs.length), parkedCI: wilson(k, legs.length), progress: mean(per.map((p) => p.progress)),
      progressMin: Math.min(...per.map((p) => p.progress)), progressMax: Math.max(...per.map((p) => p.progress)) };
  }
  function lnStudies() {
    const L = st.ln;
    return st.M.studies.filter((s) => (!L.level || s.level === L.level) && (L.numbers === 'all' || s.policy?.numbers === L.numbers) && (L.seed == null || s.seed === L.seed));
  }
  // one series per model, split by the numbers policy when a card holds both (e.g. self-allow and self-none):
  // the no-numbers series is a lighter tint of the model colour with hollow markers
  const tint = (hex, f = 0.45) => { const n = parseInt(hex.slice(1), 16); const c = [n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.round(v + (255 - v) * f)); return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`; };
  function seriesOf(studies) {
    const mixed = uniq(studies.map((s) => s.policy?.numbers || '')).length > 1;
    const out = new Map();
    for (const s of studies) {
      const num = s.policy?.numbers || '';
      const key = mixed ? `${s.model}|${num}` : s.model;
      if (out.has(key)) continue;
      const mi = mInfo(s.model), light = mixed && num === 'none';
      out.set(key, { key, model: s.model, numbers: num, hollow: light, color: /^#[0-9a-f]{6}$/i.test(mi.color) && light ? tint(mi.color) : mi.color,
        name: mixed ? `${mi.name} · ${NUMBERS_NAMES[num] || num}` : mi.name, rank: modelRank({ spec: s.model, model: mi.name }) * 2 + (light ? 1 : 0) });
    }
    return [...out.values()].sort((a, b) => a.rank - b.rank || a.key.localeCompare(b.key));
  }
  const seriesKey = (ser, s) => (ser.some((x) => x.key.includes('|')) ? `${s.model}|${s.policy?.numbers || ''}` : s.model);
  function curveChart(cond, studies, metric) {
    // per series: values per step (members), transfer values; baseline from the matrix
    const ser = seriesOf(studies);
    const byModel = new Map(ser.map((x) => [x.key, { x, steps: new Map(), transfer: [], study: [], seed: null, tseed: null, level: null, maxStep: 0 }]));
    for (const s of studies) {
      const e = byModel.get(seriesKey(ser, s));
      if (e.seed == null) Object.assign(e, { seed: s.seed, tseed: s.transfer_seed, level: s.level });
      e.study.push(s.study);
      e.maxStep = Math.max(e.maxStep, s.steps || 0);
      for (const run of s.runs || []) {
        const v = relayScore(run)[metric];
        if (run.role === 'transfer') e.transfer.push({ v, run });
        else { if (!e.steps.has(run.step)) e.steps.set(run.step, []); e.steps.get(run.step).push({ v, run }); }
        if (run.role !== 'transfer') e.maxStep = Math.max(e.maxStep, run.step);
      }
    }
    if (!byModel.size) return '';
    const S = Math.max(1, ...[...byModel.values()].map((e) => e.maxStep));
    const W = 560, H = 230, l = 40, r = 10, t = 12, b = 32;
    const xs = (k) => l + (W - l - r - 46) * (S === 1 ? 0.5 : (k - 1) / (S - 1));
    const xT = W - r - 14;
    const y = (v) => t + (H - t - b) * (1 - v);
    let g = '';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${l - 6}" y="${y(v) + 3.5}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    for (let k = 1; k <= S; k++) g += `<text class="tick" x="${xs(k)}" y="${H - b + 16}" text-anchor="middle">${k}</text>`;
    g += `<text class="tick" x="${xT}" y="${H - b + 16}" text-anchor="middle">T</text><line class="axis" x1="${xT - 22}" x2="${xT - 22}" y1="${t}" y2="${H - b}"/>`;
    g += `<text class="tick lab" x="${(l + xs(S)) / 2}" y="${H - 2}" text-anchor="middle">${CONDITIONS[cond]?.step || 'Step'}</text><text class="tick lab" x="${xT}" y="${H - 2}" text-anchor="middle">new courses</text>`;
    const tips = Array.from({ length: S + 1 }, () => []);
    const marks = [];
    const baseDrawn = new Set();
    // with more than two series the bands (min–max over drivers, the reference's interval) pile up into a wash: lines only,
    // the spread stays in the hover tooltip
    const bands = [...byModel.values()].filter((e) => e.study.length).length <= 2;
    for (const [, e] of byModel) {
      if (!e.study.length) continue;
      const model = e.x.model, { color, name: mname, hollow } = e.x;
      const base = baselineFor(model, e.level, e.seed), tbase = e.tseed != null ? baselineFor(model, e.level, e.tseed) : null;
      const first = !baseDrawn.has(model); // the no-notes reference is the same for both number policies: draw it once
      baseDrawn.add(model);
      if (base && first) {
        const bv = base[metric], [blo, bhi] = metric === 'parked' ? base.parkedCI : [base.progressMin, base.progressMax];
        if (bands) g += `<rect x="${xs(1) - 8}" y="${y(bhi)}" width="${xs(S) - xs(1) + 16}" height="${Math.max(1, y(blo) - y(bhi))}" fill="${color}" opacity="0.07"/>`;
        g += `<line class="ref" x1="${xs(1) - 8}" x2="${xs(S) + 8}" y1="${y(bv)}" y2="${y(bv)}" stroke="${color}"/>`;
      }
      if (tbase && first) g += `<line class="ref" x1="${xT - 12}" x2="${xT + 12}" y1="${y(tbase[metric])}" y2="${y(tbase[metric])}" stroke="${color}"/>`;
      const pts = [];
      for (let k = 1; k <= S; k++) {
        const vs = (e.steps.get(k) || []).map((p) => p.v);
        if (!vs.length) continue;
        pts.push([k, mean(vs), Math.min(...vs), Math.max(...vs), e.steps.get(k)]);
      }
      if (pts.length > 1) {
        if (bands) g += `<path d="M${pts.map(([k, , lo]) => `${xs(k)},${y(lo)}`).join('L')}L${pts.slice().reverse().map(([k, , , hi]) => `${xs(k)},${y(hi)}`).join('L')}Z" fill="${color}" opacity="0.12"/>`;
        g += `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map(([k, m]) => `${xs(k)},${y(m)}`).join(' ')}"/>`;
      } else if (pts.length === 1) {
        const [k, , lo, hi] = pts[0];
        g += `<line x1="${xs(k)}" x2="${xs(k)}" y1="${y(hi)}" y2="${y(lo)}" stroke="${color}" stroke-width="2" opacity="0.5"/>`;
      }
      for (const [k, m, lo, hi, items] of pts) {
        marks.push(hollow ? `<circle class="mk hollow" cx="${xs(k)}" cy="${y(m)}" r="4" style="fill:#141814;stroke:${color}"/>` : `<circle class="mk" cx="${xs(k)}" cy="${y(m)}" r="4.5" fill="${color}"/>`);
        tips[k - 1].push(`<span class="trow"><i style="background:${color}"></i><b>${pct(m)}</b> ${esc(mname)} <span class=dim>(${items.length} driver${items.length === 1 ? '' : 's'}${items.length > 1 ? `, ${pct(lo)}–${pct(hi)}` : ''})</span></span>`);
      }
      if (base && first) for (let k = 0; k < S; k++) tips[k].push(`<span class="trow dim"><i class="dash" style="border-color:${color}"></i>${pct(base[metric])} ${esc(mname)} without notes (${base.n} run${base.n === 1 ? '' : 's'})</span>`);
      if (e.transfer.length) {
        const vs = e.transfer.map((p) => p.v), m = mean(vs), lo = Math.min(...vs), hi = Math.max(...vs);
        if (hi > lo) g += `<line x1="${xT}" x2="${xT}" y1="${y(hi)}" y2="${y(lo)}" stroke="${color}" stroke-width="2" opacity="0.6"/>`;
        marks.push(`<rect class="mk" x="${xT - 5}" y="${y(m) - 5}" width="10" height="10" transform="rotate(45 ${xT} ${y(m)})" ${hollow ? `style="fill:#141814;stroke:${color}"` : `fill="${color}"`}/>`);
        tips[S].push(`<span class="trow"><i style="background:${color}"></i><b>${pct(m)}</b> ${esc(mname)} on new courses (${courseSet(e.tseed)})${tbase ? ` · ${pct(tbase[metric])} without notes` : ''}</span>`);
      }
    }
    // hover columns (crosshair snaps to a step)
    let hits = '';
    for (let k = 1; k <= S + 1; k++) {
      const x = k <= S ? xs(k) : xT, w = k <= S ? Math.max(24, (xs(2) - xs(1)) || 60) : 40;
      if (!tips[k - 1].length) continue;
      hits += `<g class="col"><line class="xhair" x1="${x}" x2="${x}" y1="${t}" y2="${H - b}"/><rect class="hit" x="${x - w / 2}" y="${t}" width="${w}" height="${H - b - t}" data-tip="${esc(`<b>${k <= S ? `${CONDITIONS[cond]?.step || 'Step'} ${k}` : 'New courses (transfer)'}</b><br>${tips[k - 1].join('<br>')}`)}"/></g>`;
    }
    return `<svg class="curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(`${cond} ${metric}`)}">${g}${hits}${marks.join('')}</svg>`;
  }
  function monotonicity(cond, studies) {
    const ser = seriesOf(studies);
    const rows = ser.map(({ key, color, name: mname }) => {
      const ss = studies.filter((s) => seriesKey(ser, s) === key);
      const cells = VEHICLES.map((veh) => {
        let seqs = [];
        for (const s of ss) {
          const train = (s.runs || []).filter((r) => r.role !== 'transfer');
          const val = (run) => { const L = run.legs.find((l) => l.vehicle === veh); return L ? legScore(L) : null; };
          if (cond === 'pool') {
            const steps = uniq(train.map((r) => r.step)).sort((a, b) => a - b);
            seqs.push(steps.map((k) => mean(train.filter((r) => r.step === k).map(val).filter((v) => v != null))));
          } else {
            for (const m of uniq(train.map((r) => r.member))) seqs.push(train.filter((r) => r.member === m).sort((a, b) => a.step - b.step).map(val).filter((v) => v != null));
          }
        }
        seqs = seqs.filter((q) => q.length > 1);
        const ok = seqs.filter((q) => q.every((v, i) => !i || v >= q[i - 1] - 0.02)).length;
        return seqs.length ? `<td class="num" data-tip="${esc(`<b>${ok} of ${seqs.length}</b> sequences never got worse<br>${esc(veh)} · ${esc(mname)}<br><span class=dim>score = 1 if parked, else progress; a drop of more than 2 points counts as worse</span>`)}"><span class="mono-bar"><b style="width:${100 * ok / seqs.length}%"></b></span>${ok}/${seqs.length}</td>` : '<td class="num dim">—</td>';
      }).join('');
      return `<tr><th scope="row">${dot(color)}${esc(mname)}</th>${cells}</tr>`;
    }).join('');
    const what = cond === 'pool' ? 'the round means of each leg' : cond === 'chain' ? 'the chain of generations' : 'each driver\'s attempts';
    return `<div class="mono"><div class="eyebrow">Never got worse</div><p class="cap">Share of sequences (${what}) in which the leg's score never dropped from one step to the next.</p><div class="tscroll"><table class="rtable compact mono-t"><thead><tr><th></th>${VEHICLES.map((v) => `<th class="num">${v}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }
  function learning() {
    const M = st.M;
    if (!M.studies.length) {
      return header(SECTION_NAMES.learning, 'After each leg a driver may leave a short note for the next driver.')
        + `<div class="res-empty big"><b>No learning studies yet.</b><p>Start one with<br><code>python -m relay learn --model codex:gpt-6.1-sol --condition self --members 2 --steps 3</code><br>then <code>python -m relay results</code>. Studies appear here as each attempt finishes.</p></div>`;
    }
    const levels = uniq(M.studies.map((s) => s.level)).sort((a, b) => LEVELS.indexOf(a) - LEVELS.indexOf(b));
    if (!st.ln.level || !levels.includes(st.ln.level)) st.ln.level = levels.includes('medium') ? 'medium' : levels[0];
    const seeds = uniq(M.studies.map((s) => s.seed)).sort();
    const ctl = `${selectCtl('ln-level', levels.map((v) => [v, LEVEL_SHORT[v] || v]), st.ln.level, 'Difficulty')}${selectCtl('ln-numbers', [['all', 'any'], ['allow', 'may use numbers'], ['none', 'without numbers']], st.ln.numbers, 'Notes')}${seeds.length > 1 ? selectCtl('ln-seed', [['', 'any'], ...seeds.map((v) => [v, courseSet(v)])], st.ln.seed ?? '', 'Courses') : ''}`;
    const ss = lnStudies();
    let body = '';
    for (const cond of Object.keys(CONDITIONS)) {
      const cs = ss.filter((s) => s.condition === cond);
      if (!cs.length) continue;
      const leg = seriesOf(cs).map((x) => ({ label: x.name, color: x.color, hollow: x.hollow }));
      const nMembers = cs.reduce((a, s) => a + (s.members || 1), 0);
      body += card(`${CONDITIONS[cond].name} <span class="dim">· ${cs.length} stud${cs.length === 1 ? 'y' : 'ies'}, ${nMembers} driver line${nMembers === 1 ? '' : 's'}</span>`,
        `${CONDITIONS[cond].what} Solid line: mean over drivers per ${CONDITIONS[cond].step.toLowerCase()}${leg.length <= 2 ? ', shaded band: min–max over drivers' : ' (the spread over drivers is in the hover)'}. Dashed: the same model without notes on the same course and level. ◆ at T: one more drive on an unseen course set with the final notes (transfer), with that set's no-notes level as a short dash. Hover a column for the numbers.`,
        `${legend(leg, 'line')}${leg.some((x) => x.hollow) ? '<div class="rlegend"><span><i class="ring"></i>hollow markers, lighter line: notes without numbers</span></div>' : ''}<div class="rlegend"><span><i class="dashl"></i>same model, no notes${leg.length <= 2 ? ' (band: 95% interval for parked, min–max for progress)' : ''}</span><span>◆ new courses (transfer: the notes taken to an unseen course set)</span></div><div class="curves"><div><div class="eyebrow">Legs parked</div>${curveChart(cond, cs, 'parked')}</div><div><div class="eyebrow">Mean progress per leg</div>${curveChart(cond, cs, 'progress')}</div>${monotonicity(cond, cs)}</div>`, 'wide');
    }
    if (!body) body = '<div class="res-empty">No studies match these filters.</div>';
    // studies table
    const list = M.studies.map((s) => {
      const mi = mInfo(s.model);
      const expected = (s.condition === 'chain' ? 1 : s.members || 1) * (s.steps || 0) + (s.transfer_seed != null ? (s.condition === 'self' ? s.members || 1 : 1) : 0);
      const notesW = (s.runs || []).flatMap((r) => r.legs.map((l) => l.note)).filter(Boolean);
      const acc = notesW.filter((n) => n.accepted).length, rej = notesW.length - acc;
      return `<tr class="click" data-study="${esc(s.study)}"><th scope="row">${dot(mi.color)}${esc(mi.name)}</th><td>${esc(CONDITIONS[s.condition]?.name || s.condition)}</td><td>${esc(LEVEL_SHORT[s.level] || s.level)} · ${esc(setShort(s.seed))}${s.transfer_seed != null ? ` → ${esc(setShort(s.transfer_seed))}` : ''}</td><td>${esc(NUMBERS_NAMES[s.policy?.numbers] || '')}</td><td class="num">${s.condition === 'chain' ? 1 : s.members}×${s.steps}</td><td class="num">${(s.runs || []).length}/${expected}</td><td class="num">${acc} <span class="dim">/ ${rej} rejected</span></td><td class="mono-id">${esc(s.study)}</td><td><button class="lnk" data-notes="${esc(s.study)}">Read notes →</button></td></tr>`;
    }).join('');
    body += card('All studies', 'Every learning-from-notes study. Runs: tries finished of those planned (practice plus the try on new courses). Notes: kept / rejected by the note check (too long, code, or numbers where none are allowed).', `<div class="tscroll"><table class="rtable"><thead><tr><th>Model</th><th>How notes travel</th><th>Difficulty · courses</th><th>Notes</th><th class="num">Drivers × tries</th><th class="num">Runs</th><th class="num">Notes</th><th>Study</th><th></th></tr></thead><tbody>${list}</tbody></table></div>`, 'wide');
    setTimeout(() => {
      for (const b of main.querySelectorAll('[data-notes]')) b.onclick = (e) => { e.stopPropagation(); go('notes', { notes: { a: b.dataset.notes } }); };
      for (const tr of main.querySelectorAll('tr[data-study]')) tr.onclick = () => { const s = M.studies.find((x) => x.study === tr.dataset.study); if (s) showRuns(s.runs.map((r) => r.run_id), `Study ${s.study}`); };
    });
    return header(SECTION_NAMES.learning, 'After each leg a driver may write a short note (plain prose, at most 600 characters) for whoever drives that course next. Do the notes help? Parked share and progress per try, against the same model without notes.', ctl) + body;
  }

  // ------------------------------------------------------------------ 4. notes browser
  function stepName(s, run) {
    const c = s.condition;
    if (run.role === 'transfer') return `New courses (transfer)${(c === 'self' && (s.members || 1) > 1) ? ` · driver ${run.member + 1}` : ''}`;
    if (c === 'self') return `Attempt ${run.step}${(s.members || 1) > 1 ? ` · driver ${run.member + 1}` : ''}`;
    if (c === 'chain') return `Generation ${run.step}`;
    return `Round ${run.step} · driver ${run.member + 1}`;
  }
  function studyOption(s) {
    return `${CONDITIONS[s.condition]?.name || s.condition} · ${mInfo(s.model).name} · ${LEVEL_SHORT[s.level] || s.level} · ${NUMBERS_NAMES[s.policy?.numbers] || ''} · ${fmtWhen(s.study.slice(0, 13).replace(/^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)/, '$1-$2-$3T$4:$5'))}`;
  }
  function noteColumn(s, legIdx) {
    if (!s) return '<div class="res-empty">Pick a study.</div>';
    const runs = studyRuns(s);
    const veh = VEHICLES[legIdx];
    const color = mInfo(s.model).color;
    // who read what: a note's text appears in later notes_in
    if (s.condition === 'self') runs.sort((a, b) => a.member - b.member || (a.role === 'transfer') - (b.role === 'transfer') || a.step - b.step);
    const entries = runs.map((run) => ({ run, L: run.legs.find((l) => l.vehicle === veh) || run.legs[legIdx] })).filter((e) => e.L);
    // who received a note: the reader's notes_in carries the same text under the writer's label (relay/learning.py)
    const fromLabel = (run) => (s.condition === 'self' ? `you, attempt ${run.step}` : s.condition === 'chain' ? `notebook as rewritten by driver ${run.step}` : `round ${run.step}, driver ${run.member + 1}`);
    const readers = (text, w) => entries.filter((e) => e !== w && (s.condition !== 'self' || e.run.member === w.run.member)
      && (e.L.notes_in || []).some((n) => n.text === text && n.from === fromLabel(w.run)));
    const chip = (L) => { const k = outcomeKey(L); return `<span class="ochip ${k || 'run'}">${k ? OUT[k].glyph : '…'} ${esc(outcomeText(L))}</span>`; };
    const mem = st.notes.member, multi = s.condition !== 'chain' && (s.members || 1) > 1;
    const items = entries.filter((e) => mem == null || !multi || e.run.member === mem).map((e) => {
      const { run, L } = e;
      const nin = L.notes_in || [];
      const read = nin.length ? `<details class="nin"${nin.length <= 2 ? ' open' : ''}><summary>Read ${nin.length} note${nin.length === 1 ? '' : 's'}${s.condition === 'chain' ? ' (the notebook)' : ''}</summary>${nin.map((n) => `<blockquote><cite>${esc(n.from)}</cite>${esc(n.text)}</blockquote>`).join('')}</details>`
        : `<div class="nin none">${run.role === 'transfer' ? 'Read no notes for this leg' : 'No notes to read yet'}</div>`;
      let wrote = '';
      const n = L.note;
      if (!n) wrote = run.role === 'transfer' ? '<div class="nnone">Transfer attempts read notes but do not write one.</div>' : '<div class="nnone">No note was written.</div>';
      else {
        const rej = n.rejections || [], drafts = n.drafts || [];
        const rejected = rej.map((why, i) => `<div class="draft"><div class="dhead"><span class="nbadge bad">✕ rejected draft ${i + 1}</span></div><p>${esc(drafts[i] ?? '')}</p><ul>${(why || []).map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>`).join('');
        if (n.accepted) {
          const by = readers(n.text, e);
          wrote = `<div class="note ok" style="--mc:${color}"><div class="nhead"><span class="nbadge ok">✓ accepted</span>${rej.length ? `<span class="dim">after ${rej.length} rejected draft${rej.length === 1 ? '' : 's'}</span>` : ''}<span class="nlen">${(n.text || '').length} chars</span></div><p class="ntext">${esc(n.text)}</p>`
            + (by.length ? `<div class="readby"><span class="eyebrow">Read by</span>${by.map((b) => `<span class="rb">${esc(stepName(s, b.run))} ${chip(b.L)}</span>`).join('')}</div>` : '<div class="readby none">Not read by any later attempt yet.</div>')
            + `</div>${rejected ? `<details class="drafts"><summary>${rej.length} rejected draft${rej.length === 1 ? '' : 's'} before this</summary>${rejected}</details>` : ''}`;
        } else {
          wrote = `<div class="note bad"><div class="nhead"><span class="nbadge bad">✕ rejected — nothing passed on</span></div>${rejected}</div>`;
        }
      }
      return `<article class="nentry${run.role === 'transfer' ? ' transfer' : ''}"><header><div class="ntitle"><b>${esc(stepName(s, run))}</b>${chip(L)}<span class="dim">${L.touches ?? 0} rail hit${L.touches === 1 ? '' : 's'}</span></div>
        <div class="nopen"><button class="lnk" data-open="${esc(run.run_id)}" data-view="inspect" data-leg="${L.index}">Inspect leg</button><button class="lnk" data-open="${esc(run.run_id)}">Broadcast</button></div></header>${read}<div class="wrote">${wrote}</div></article>`;
    }).join('');
    const pol = s.policy || {};
    return `<div class="ncol-head">${dot(color)}<div><b>${esc(studyOption(s))}</b><div class="dim">${esc(CONDITIONS[s.condition]?.what || '')} Notes: at most ${pol.max_chars} characters, ${pol.numbers === 'none' ? 'no numbers' : `at most ${pol.max_numbers} numbers`}, prose only.</div></div></div>${items || '<div class="res-empty">No attempts yet.</div>'}`;
  }
  function notes() {
    const M = st.M;
    if (!M.studies.length) return header(SECTION_NAMES.notes, 'What drivers wrote for each other.') + '<div class="res-empty big"><b>No learning studies yet.</b><p>Notes appear here once a study (<code>python -m relay learn …</code>) has run.</p></div>';
    if (!st.notes.a || !M.studies.some((s) => s.study === st.notes.a)) st.notes.a = M.studies[0].study;
    const A = M.studies.find((s) => s.study === st.notes.a), B = M.studies.find((s) => s.study === st.notes.b);
    const opts = (cur, none) => `${none ? '<option value="">— none —</option>' : ''}${Object.keys(CONDITIONS).map((c) => {
      const ss = M.studies.filter((s) => s.condition === c);
      return ss.length ? `<optgroup label="${CONDITIONS[c].name}">${ss.map((s) => `<option value="${esc(s.study)}" ${s.study === cur ? 'selected' : ''}>${esc(studyOption(s))}</option>`).join('')}</optgroup>` : '';
    }).join('')}`;
    const count = (s, i) => (s ? (s.runs || []).filter((r) => r.legs.find((l) => l.vehicle === VEHICLES[i])?.note?.text).length : 0);
    // with several drivers per step (self, pool) one driver's line can be followed on its own
    const drivers = Math.max(...[A, B].filter((x) => x && x.condition !== 'chain').map((x) => x.members || 1), 1);
    if (st.notes.member != null && st.notes.member >= drivers) st.notes.member = null;
    const legBtns = VEHICLES.map((v, i) => [String(i), `${v[0].toUpperCase()}${v.slice(1)} · ${count(A, i)}${B ? `/${count(B, i)}` : ''}`]);
    const ctl = `<label class="rsel wide"><span>Study</span><select data-ctl="nt-a">${opts(st.notes.a)}</select></label><label class="rsel wide"><span>Compare with</span><select data-ctl="nt-b">${opts(st.notes.b, true)}</select></label>`;
    return header(SECTION_NAMES.notes, 'Every try at one course, in order: the notes the driver read, how it drove, and the note it left (kept, or rejected with the reasons). "Read by" follows a note to the tries that received it.', ctl)
      + `<div class="res-filters"><span class="eyebrow">Leg</span>${segCtl('nt-leg', legBtns, String(st.notes.leg))}<span class="dim">numbers = accepted notes per leg</span>${drivers > 1 ? `<span class="eyebrow">Driver</span>${segCtl('nt-member', [['', 'all'], ...Array.from({ length: drivers }, (_, i) => [String(i), String(i + 1)])], st.notes.member == null ? '' : String(st.notes.member))}` : ''}</div>`
      + `<div class="ncols${B ? ' two' : ''}"><div class="ncol">${noteColumn(A, st.notes.leg)}</div>${B ? `<div class="ncol">${noteColumn(B, st.notes.leg)}</div>` : ''}</div>`;
  }

  // ------------------------------------------------------------------ 5. runs table
  const FILTERS = [
    ['model', 'Model', (r) => r.model],
    ['obs', 'Driver sees', (r) => obsName(r.obs)],
    ['level', 'Difficulty', (r) => (r.level ? LEVEL_SHORT[r.level] || r.level : '—')],
    ['relay', 'Courses', (r) => relayName(r.relayName)],
    ['seed', 'Course set', (r) => courseSet(r.seed)],
    ['study', 'Notes study', (r) => (r.learning ? `${CONDITION_NAMES[r.learning.condition] || r.learning.condition} · ${r.short} · ${String(r.learning.study).slice(9, 11)}:${String(r.learning.study).slice(11, 13)}` : 'no notes')],
    ['role', 'Try', (r) => (r.learning ? (r.learning.role === 'transfer' ? 'new courses' : 'practice') : '—')],
  ];
  const SORTS = {
    created: (r) => r.created || '', label: (r) => r.lab.text, relay: (r) => `${r.relayName}${r.seed}`, parked: (r) => (r.nLegs ? r.parked / r.nLegs : -1),
    progress: (r) => (r.progress ?? -1) / Math.max(1, r.nLegs || 1), touches: (r) => r.touches ?? -1, time: (r) => r.time ?? -1,
  };
  function runsFiltered() {
    const R = st.runs, q = R.q.trim().toLowerCase();
    let rows = st.M.all;
    if (R.ids) { const want = new Set(R.ids); rows = rows.filter((r) => want.has(r.id)); }
    for (const [k, , f] of FILTERS) if (R.f[k] != null) rows = rows.filter((r) => String(f(r)) === R.f[k]);
    if (q) rows = rows.filter((r) => `${r.lab.text} ${r.lab.full} ${r.id} ${r.d.spec} ${r.learning?.study || ''}`.toLowerCase().includes(q));
    const [key, dir] = R.sort;
    const f = SORTS[key] || SORTS.created;
    return rows.slice().sort((a, b) => { const x = f(a), y = f(b); return (x < y ? -1 : x > y ? 1 : 0) * dir || (b.created || '').localeCompare(a.created || ''); });
  }
  function legChips(r) {
    const n = Math.max(r.nLegs || 0, r.legs.length);
    let h = '<span class="lchips">';
    for (let i = 0; i < n; i++) {
      const L = r.legs.find((l) => (l.index ?? -1) === i) || r.legs[i];
      const k = L ? outcomeKey(L) : null;
      const txt = L ? `${L.vehicle}: ${k ? outcomeText(L) : r.live ? 'driving' : 'not run'}${L.touches ? ` · ${L.touches} rail hits` : ''}` : 'not run';
      h += `<b class="${k || (r.live ? 'run' : 'none')}" data-tip="${esc(txt)}">${k ? OUT[k].glyph : r.live ? '…' : ''}</b>`;
    }
    return `${h}</span>`;
  }
  function runsRowsHTML(rows) {
    return rows.map((r) => {
      const sel = st.runs.sel.has(r.id);
      const learn = r.learning ? `<span class="vt learn" title="${esc(r.learning.study)}">${esc(studyTagOf(r.learning))}</span>${r.learning.numbers === 'none' ? ' <span class="vt">without numbers</span>' : ''}` : '';
      return `<tr data-id="${esc(r.id)}" class="${sel ? 'sel' : ''}"><td class="ck"><input type="checkbox" ${sel ? 'checked' : ''} aria-label="select ${esc(r.lab.text)}"></td>`
        + `<td class="who" title="${esc(r.lab.full)}"><div class="wn">${dot(r.color)}<b>${esc(r.model)}</b><span>${esc(r.lab.tag)}</span></div><div class="wsub">${esc(fmtWhen(r.created))}${r.live ? ' · <span class="live">driving now</span>' : r.partial ? ' · unfinished' : ''}${!r.continuous ? ` · ${CONTROL_NAMES.discrete.toLowerCase()}` : ''}</div></td>`
        + `<td>${esc(relayName(r.relayName))} · ${esc(setShort(r.seed))}</td><td>${esc(obsName(r.obs, r.level))}</td><td>${learn || '<span class="dim">—</span>'}</td>`
        + `<td>${legChips(r)}</td><td class="num">${r.nLegs ? `${r.parked}/${r.nLegs}` : '—'}</td><td class="num">${r.progress != null ? pct(r.progress / Math.max(1, r.nLegs)) : '—'}</td><td class="num">${r.touches ?? '—'}</td><td class="num">${fmtTime(r.time)}</td>`
        + `<td class="acts"><button class="lnk" data-open="${esc(r.id)}">Broadcast</button><button class="lnk" data-open="${esc(r.id)}" data-view="inspect">Inspect</button></td></tr>`;
    }).join('');
  }
  function selBar() {
    const n = st.runs.sel.size;
    const relays = uniq([...st.runs.sel].map((id) => st.M.byId.get(id)?.relay).filter(Boolean));
    return `<div class="selbar${n ? ' on' : ''}"><b>${n}</b> selected${relays.length > 1 ? ` <span class="warn">· ${relays.length} different course sets: Broadcast shows one at a time</span>` : ''}<span class="sp"></span><button class="btn sm" data-act="clear" ${n ? '' : 'disabled'}>Clear</button><button class="btn sm" data-act="grid" ${n ? '' : 'disabled'}>Open in Grid</button><button class="btn sm primary" data-act="broadcast" ${n ? '' : 'disabled'}>Open ${n || ''} in Broadcast</button></div>`;
  }
  function runsTable() {
    const R = st.runs, all = st.M.all;
    const flt = FILTERS.map(([k, name, f]) => {
      const vals = uniq(all.map((r) => String(f(r)))).sort();
      return selectCtl(`rt-f-${k}`, [['', 'all'], ...vals.map((v) => [v, v])], R.f[k] ?? '', name);
    }).join('');
    const ids = R.ids ? `<div class="idschip">${esc(R.idsTitle)} <b>${R.ids.length}</b><button class="x" data-act="ids-clear" aria-label="Show all runs">×</button></div>` : '';
    const cols = [['', ''], ['label', 'Run'], ['relay', 'Courses'], ['', 'Driver sees'], ['', 'Notes study'], ['', 'Legs'], ['parked', 'Parked'], ['progress', 'Progress'], ['touches', 'Rail hits'], ['time', 'Time'], ['', '']];
    const th = cols.map(([k, n], i) => `<th ${k ? `data-sort="${k}" class="sortable${R.sort[0] === k ? ` on ${R.sort[1] > 0 ? 'asc' : 'desc'}` : ''}"` : ''} ${i >= 6 && i <= 9 ? 'style="text-align:right"' : ''}>${i === 0 ? '<input type="checkbox" data-act="all" aria-label="select all shown">' : n}</th>`).join('');
    return header(SECTION_NAMES.runs, `Every run, newest first. Tick runs to replay them side by side (one course set at a time), or open one in detail.`)
      + `<div class="res-filters runs"><input type="search" data-ctl="rt-q" placeholder="Search label, run id, study…" value="${esc(R.q)}">${flt}${ids}</div>`
      + `<div class="rtable-wrap"><table class="rtable runs"><thead><tr>${th}</tr></thead><tbody id="rt-body"></tbody></table><div class="rt-count" id="rt-count"></div></div>${selBar()}`;
  }
  function renderRunsBody() {
    const body = main.querySelector('#rt-body');
    if (!body) return;
    const rows = runsFiltered();
    body.innerHTML = runsRowsHTML(rows) || '<tr><td colspan="11" class="dim" style="padding:24px">No runs match.</td></tr>';
    main.querySelector('#rt-count').textContent = `${rows.length} of ${st.M.all.length} runs`;
    const all = main.querySelector('[data-act=all]');
    if (all) all.checked = rows.length > 0 && rows.every((r) => st.runs.sel.has(r.id));
    st.runs.shown = rows.map((r) => r.id);
  }
  function wireRuns() {
    renderRunsBody();
    const body = main.querySelector('#rt-body');
    body.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-id]');
      if (!tr || e.target.closest('[data-open]')) return;
      const id = tr.dataset.id;
      if (st.runs.sel.has(id)) st.runs.sel.delete(id); else st.runs.sel.add(id);
      tr.classList.toggle('sel', st.runs.sel.has(id));
      tr.querySelector('input').checked = st.runs.sel.has(id);
      updSel();
    });
    for (const h of main.querySelectorAll('th[data-sort]')) h.onclick = () => {
      const k = h.dataset.sort;
      st.runs.sort = st.runs.sort[0] === k ? [k, -st.runs.sort[1]] : [k, k === 'label' || k === 'relay' ? 1 : -1];
      const y = main.scrollTop; render(); main.scrollTop = y;
    };
    main.querySelector('[data-act=all]').onchange = (e) => {
      for (const id of st.runs.shown || []) { if (e.target.checked) st.runs.sel.add(id); else st.runs.sel.delete(id); }
      renderRunsBody(); updSel();
    };
    main.querySelector('[data-act=ids-clear]')?.addEventListener('click', () => { st.runs.ids = null; render(); });
    wireSel();
  }
  function updSel() {
    const old = main.querySelector('.selbar');
    old.outerHTML = selBar();
    wireSel();
  }
  function wireSel() {
    const bar = main.querySelector('.selbar');
    bar.querySelector('[data-act=clear]').onclick = () => { st.runs.sel.clear(); renderRunsBody(); updSel(); };
    bar.querySelector('[data-act=broadcast]').onclick = () => ctx.openRuns([...st.runs.sel], 'broadcast');
    bar.querySelector('[data-act=grid]').onclick = () => ctx.openRuns([...st.runs.sel], 'grid');
  }

  return {
    // the notes reader's state for the page URL (only while it is shown)
    urlParams: () => (st.section !== 'notes' ? {} : { study: st.notes.a, compare: st.notes.b, leg: VEHICLES[st.notes.leg], driver: st.notes.member == null ? null : st.notes.member + 1 }),
    async show(sec) { if (!st.M || !st.R) await load(true); go(sec || st.section); },
    refresh,
    reload: async () => { await load(true); render(); },
    section: () => st.section,
    showRuns,
    go,
    state: () => ({ section: st.section, generated: st.M?.generated ?? null, runs: st.M?.all.length ?? 0, digest: st.M?.res.length ?? 0, studies: st.M?.studies.length ?? 0,
      shown: st.runs.shown || [], selected: [...st.runs.sel], ids: st.runs.ids, notes: { ...st.notes }, split: st.split, error: st.err }),
    select: (ids) => { st.runs.sel = new Set(ids); if (st.section === 'runs') { renderRunsBody(); updSel(); } },
    available: () => !!st.R,
    load,
  };
}
