// The viewer's vocabulary: every name a reader sees (models, observation, perception level, mode, clock, course
// sets, learning roles, sections) comes from here, and one label per run is used everywhere (standings, 3D tags,
// Grid, Inspect, picker, compare, Results). Written for readers outside the project:
//   "GPT-6.1 Sol · Driver's seat, Medium · Course set B · run 2"
//   "GPT-6.1 Sol · Attempt 3 · driver 2" / "GPT-6.1 Terra · Notebook relay · generation 5" / "… · Team round 2 · driver 3"
// The compact form ("Sol · run 2", "Sol · attempt 3, driver 2") is for tags in the 3D scene.
// It accepts a full run record, a runs/index.json item, or a runs/results.json run row.

// ---------------------------------------------------------------- vocabulary
export const OBS_NAMES = {
  driver: "Driver's seat", local: 'Map view (exact sensors)', cone: 'Vision cone', 'cone-sketch': 'Vision cone + sketch',
  sketch: 'Sketch map', full: 'Full map', sensors: 'Range sensors only',
};
export const OBS_NOTES = {
  driver: 'what a human driver sees: head, mirrors, reversing camera, rough depth', local: 'exact distances to the rails around the rig',
  cone: 'only what lies in front of the eyes', sensors: 'distance sensors, no map',
};
export const LEVELS = ['easy', 'medium', 'hard', 'v1']; // difficulty increases left to right; v1 is the first, legacy level
export const LEVEL_DEFAULT = 'medium';
export const LEVEL_NAMES = { easy: 'Easy', medium: 'Medium', hard: 'Hard', v1: 'No peripheral vision (v1)' };
export const LEVEL_SHORT = { easy: 'Easy', medium: 'Medium', hard: 'Hard', v1: 'No periphery' };
// relay/perception.py LEVELS: depth-noise scale, reversing-camera range multiplier, peripheral-vision range
export const LEVEL_NOTES = {
  easy: 'depth noise ×0.35 · camera ×2 · peripheral 5 m', medium: 'depth noise ×0.6 · camera ×1.5 · peripheral 4 m',
  hard: 'depth noise ×1 · camera ×1 · peripheral 3 m', v1: 'depth noise ×1 · camera ×1 · no corner of the eye',
};
export const MODE_NAMES = { code: 'Writes its own controller', tools: 'Command by command' };
export const CONTROL_NAMES = { continuous: 'Keeps rolling', discrete: 'Stops after each command' };
export const CLOCK_NAMES = { paused: 'Clock paused while thinking', realtime: 'Real-time clock' };
export const RELAY_NAMES = { standard: 'Standard courses', asym: 'Off-centre bays', turnaround: 'Turnaround', gauntlet: 'Gauntlet', warmup: 'Warm-up', mini: 'Mini' };
export const VEHICLE_NAMES = { car: 'Car + trailer', tractor: 'Tractor + tipping trailer', mower: 'Ride-on mower + cart', truck: 'Truck + semi-trailer' };
export const TEMPLATE_NAMES = { turn: 'Turnaround', straight: 'Straight', lturn: 'L-turn', sbend: 'S-bend', alley: 'Alley dock', hairpin: 'Hairpin', chicane: 'Chicane' };
export const CONDITION_NAMES = { self: 'Retries with own notes', chain: 'Notebook relay', pool: 'Team rounds' };
export const STEP_NAMES = { self: 'Attempt', chain: 'Generation', pool: 'Round' };
export const NUMBERS_NAMES = { allow: 'notes may use numbers', none: 'notes without numbers' };
export const SECTION_NAMES = { overview: 'Scoreboard', failures: 'How they fail', learning: 'Learning from notes', notes: 'What they wrote', runs: 'All runs' };
export const courseSet = (seed) => (seed == null || seed === '' ? '—' : `Course set ${Number.isInteger(+seed) && +seed >= 0 && +seed < 26 ? String.fromCharCode(65 + +seed) : seed}`);
export const relayName = (name) => RELAY_NAMES[name] || (name ? String(name)[0].toUpperCase() + String(name).slice(1) : '—');
export const obsName = (obs, level) => (obs === 'driver' ? `${OBS_NAMES.driver}${level ? `, ${LEVEL_SHORT[level] || level}` : ''}` : OBS_NAMES[obs] || obs || '—');
export const levelName = (lv) => LEVEL_NAMES[lv] || lv || '—';

// fixed colours per model (validated as a categorical set on the dark surface: dataviz validate_palette.js)
export const MODEL_COLORS = { Sol: '#3987e5', Terra: '#d95926', Luna: '#199e70' };
export const BASELINE_COLOR = '#9b9a92';
export const AUTOPILOT_COLOR = '#e9e4d8';
// scripted baselines: muted tints (a grey family), one per policy
const BASELINE_TINTS = { 'ctrl-camera': '#b3aa8a', 'ctrl-shoulder': '#8ea3b0', 'ctrl-local': '#ad93a8' };
const EXTRA_COLORS = ['#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];
export const MODEL_ORDER = ['Sol', 'Terra', 'Luna'];

const SCRIPTED = { 'ctrl-camera': 'Naive controller (camera)', 'ctrl-shoulder': 'Naive controller (shoulder)', 'ctrl-local': 'Naive controller (map)' };
const SCRIPTED_SHORT = { 'ctrl-camera': 'Naive (camera)', 'ctrl-shoulder': 'Naive (shoulder)', 'ctrl-local': 'Naive (map)' };
const CANON = { 'GPT-6.1 Sol': 'Sol', 'GPT-6.1 Terra': 'Terra', 'GPT-6 Luna': 'Luna' };

// Normalise any run-like object to the facts a label needs.
export function describe(x) {
  if (!x) return null;
  if (x._desc) return x._desc;
  const rec = x.rec || x; // a prepared run carries its record
  const drv = rec.driver || {};
  const cfg = rec.config || {};
  const ag = cfg.agent || {};
  const spec = drv.spec ?? rec.model ?? rec.spec ?? '?';
  const obs = rec.obs !== undefined && !rec.config ? rec.obs : ag.obs_level ?? (spec === 'autopilot' ? 'full' : 'local');
  const level = rec.level !== undefined && !rec.config ? rec.level
    : obs === 'driver' ? (cfg.perception?.level || ag.perception || 'v1') : null;
  const relay = rec.relay && typeof rec.relay === 'object' ? rec.relay : { id: rec.relay, name: rec.relay_name, seed: rec.seed };
  const d = {
    runId: rec.run_id || x.runId || x.file || '?',
    spec, model: drv.label ?? rec.label ?? spec, obs, level,
    relayId: relay.id ?? null, relayName: relay.name ?? (String(relay.id || '').replace(/-s\d+$/, '') || null), seed: relay.seed ?? null,
    trial: drv.trial ?? rec.trial ?? 0,
    mode: rec.config ? (cfg.mode || 'tools') : rec.mode ?? null,
    control: rec.config ? (cfg.control || 'discrete') : (rec.clock ? 'continuous' : null),
    clock: rec.config ? (cfg.clock || null) : rec.clock ?? null,
    route: cfg.route || (spec === 'autopilot' || String(spec).startsWith('scripted:') ? 'local' : String(spec).startsWith('codex:') ? 'codex' : 'api'),
    learning: rec.learning || null, created: rec.created || '',
    auto: spec === 'autopilot', scripted: String(spec).startsWith('scripted:'),
  };
  try { Object.defineProperty(x, '_desc', { value: d, enumerable: false, configurable: true }); } catch (e) { /* frozen */ }
  return d;
}

// "GPT-6.1 Sol" -> "Sol"; scripted specs -> "Naive (camera)", "replay 5e3296"
export function modelShort(d) {
  if (typeof d === 'string') d = { spec: d, model: d };
  if (d.spec === 'autopilot') return 'Autopilot';
  if (String(d.spec).startsWith('scripted:')) {
    const p = d.spec.slice(9);
    if (SCRIPTED_SHORT[p]) return SCRIPTED_SHORT[p];
    const m = p.match(/^replay=.*?(\w{6})(?:\.json)?$/);
    if (m) return `replay ${m[1]}`;
    return p;
  }
  const lab = String(d.model || d.spec);
  // the study's three models by name alone; anything else keeps its version ("GPT-6 Sol" -> "6 Sol")
  return CANON[lab] || lab.replace(/^GPT-/i, '').replace(/^[a-z-]+:/i, '');
}

// the full name where there is room: "GPT-6.1 Sol", "Naive controller (camera)", "Autopilot (reference planner)"
const FULL = { Sol: 'GPT-6.1 Sol', Terra: 'GPT-6.1 Terra', Luna: 'GPT-6 Luna' };
export function modelName(d) {
  if (typeof d === 'string') d = { spec: d, model: d };
  if (d.spec === 'autopilot') return 'Autopilot (reference planner)';
  if (String(d.spec).startsWith('scripted:')) return SCRIPTED[String(d.spec).slice(9)] || modelShort(d);
  const s = modelShort(d);
  return FULL[s] || String(d.model || d.spec).replace(/^[a-z-]+:/i, '');
}

// stable colour of a model (hue identity across every view)
const extra = new Map();
export function modelColor(d) {
  if (typeof d === 'string') d = { spec: d, model: d };
  if (d.spec === 'autopilot') return AUTOPILOT_COLOR;
  if (String(d.spec).startsWith('scripted:')) return BASELINE_TINTS[String(d.spec).slice(9)] || BASELINE_COLOR;
  const s = modelShort(d);
  if (MODEL_COLORS[s]) return MODEL_COLORS[s];
  if (!extra.has(s)) extra.set(s, EXTRA_COLORS[extra.size % EXTRA_COLORS.length]);
  return extra.get(s);
}
export const modelRank = (d) => { const s = modelShort(d); const k = MODEL_ORDER.indexOf(s); return k >= 0 ? k : d.scripted ? 50 : d.auto ? 60 : 10; };

// "Attempt 3 · driver 2" (self), "Notebook relay · generation 5" (chain), "Team round 2 · driver 3" (pool),
// "New courses (transfer) · driver 2"; short: "attempt 3, driver 2", "generation 5", "round 2, driver 3", "new courses"
export function learningTag(L, short = false) {
  if (!L) return '';
  const c = L.condition || 'notes';
  const drv = c === 'chain' ? '' : `driver ${(L.member ?? 0) + 1}`;
  if (L.role === 'transfer') return short ? `new courses${drv ? `, ${drv}` : ''}` : `New courses (transfer)${drv ? ` · ${drv}` : ''}`;
  const k = L.step ?? '?';
  if (short) return c === 'self' ? `attempt ${k}, ${drv}` : c === 'chain' ? `generation ${k}` : c === 'pool' ? `round ${k}, ${drv}` : `step ${k}`;
  return c === 'self' ? `Attempt ${k} · ${drv}` : c === 'chain' ? `Notebook relay · generation ${k}` : c === 'pool' ? `Team round ${k} · ${drv}` : `Step ${k}`;
}

// The facts after the model name, in order. `full` adds everything (for tooltips).
export function labelParts(d, { full = false } = {}) {
  const out = [];
  if (d.auto) return out;
  if (d.learning) {
    if (d.learning.condition === 'self' && !full) out.push(learningTag(d.learning));
    else out.push(full && d.learning.condition === 'self' ? `${CONDITION_NAMES.self} · ${learningTag(d.learning)}` : learningTag(d.learning));
    if (d.learning.numbers === 'none') out.push(NUMBERS_NAMES.none);
    if (full) { out.push(obsName(d.obs, d.obs === 'driver' ? d.level || 'v1' : null)); out.push(courseSet(d.seed)); }
    return out;
  }
  out.push(obsName(d.obs, d.obs === 'driver' ? d.level || 'v1' : null));
  if (d.relayName && d.relayName !== 'standard') out.push(relayName(d.relayName));
  if (d.seed !== null && d.seed !== undefined) out.push(courseSet(d.seed));
  if (d.mode && d.mode !== 'code') out.push(MODE_NAMES[d.mode] || d.mode);
  if (d.control === 'discrete') out.push(CONTROL_NAMES.discrete);
  if (d.clock === 'realtime') out.push(CLOCK_NAMES.realtime);
  if (full && d.route === 'api') out.push('via the API');
  out.push(`run ${(d.trial ?? 0) + 1}`);
  return out.filter(Boolean);
}

// { model: "GPT-6.1 Sol", short: "Sol", tag: "Driver's seat, Medium · Course set B · run 2", text: model + tag,
//   mini: "Sol · run 2", full (tooltips: everything, with the start time), color }
export function runLabel(x, opts = {}) {
  const d = describe(x);
  const model = modelName(d), short = modelShort(d);
  const parts = labelParts(d);
  const tag = parts.join(' · ');
  const mini = d.auto ? short : d.learning ? `${short} · ${learningTag(d.learning, true)}` : `${short} · run ${(d.trial ?? 0) + 1}`;
  const fullParts = labelParts(d, { full: true });
  const full = `${model}${fullParts.length ? ` · ${fullParts.join(' · ')}` : ''}${d.created ? ` · started ${d.created.replace('T', ' ').slice(0, 16)}` : ''}`;
  return { model, short, tag, text: tag ? `${model} · ${tag}` : model, mini, full, color: modelColor(d), desc: d, ...opts };
}

// Labels for a set shown together: no two read the same. A duplicate full label gets the start time ("· 08:08",
// or the run id's tail when two started the same minute); the compact name ("Sol · run 2") adds the level, then the
// course set, then the whole tag, and finally the same suffix, until it is unique too.
export function uniqueLabels(xs) {
  const labs = xs.map((x) => runLabel(x));
  const count = (arr) => arr.reduce((m, x) => m.set(x, (m.get(x) || 0) + 1), new Map());
  const suffix = (i, idx) => {
    const c = labs[i].desc.created || '';
    const sameTime = idx.filter((j) => (labs[j].desc.created || '').slice(11, 16) === c.slice(11, 16)).length > 1;
    return sameTime || !c ? labs[i].desc.runId.slice(-4) : c.slice(11, 16);
  };
  const groups = (arr) => { const by = new Map(); arr.forEach((k, i) => { if (!by.has(k)) by.set(k, []); by.get(k).push(i); }); return by; };
  const texts = labs.map((l) => l.text);
  const suf = new Array(labs.length).fill(null);
  for (const idx of groups(texts).values()) if (idx.length > 1) for (const i of idx) suf[i] = suffix(i, idx);
  const mini = (l, k) => {
    const d = l.desc;
    if (k === 0 || d.auto) return l.mini;
    const what = d.learning ? (CONDITION_NAMES[d.learning.condition] || d.learning.condition).toLowerCase() : d.obs === 'driver' ? LEVEL_SHORT[d.level || 'v1'] : OBS_NAMES[d.obs] || d.obs;
    const tail = d.learning ? learningTag(d.learning, true) : `run ${(d.trial ?? 0) + 1}`;
    const set = courseSet(d.seed).replace('Course set ', 'set ');
    return k === 1 ? `${l.short} · ${what} · ${tail}` : k === 2 ? `${l.short} · ${what} · ${set} · ${tail}` : `${l.short} · ${l.tag}`;
  };
  const level = new Array(labs.length).fill(0);
  let pick = labs.map((l) => mini(l, 0));
  for (let k = 1; k <= 3; k++) {
    const c = count(pick);
    if ([...c.values()].every((n) => n < 2)) break;
    // escalate every run of a model that has a clash, so one model's runs read alike ("Sol hard #1", "Sol medium #1")
    const clash = new Set(labs.filter((l, i) => c.get(pick[i]) > 1).map((l) => l.short));
    pick = pick.map((m, i) => (clash.has(labs[i].short) && !labs[i].desc.auto ? mini(labs[i], (level[i] = k)) : m));
  }
  const c = count(pick);
  return labs.map((l, i) => {
    const s2 = suf[i];
    const m = c.get(pick[i]) > 1 ? `${pick[i]} · ${s2 || l.desc.runId.slice(-4)}` : pick[i];
    return s2 ? { ...l, tag: `${l.tag} · ${s2}`, text: `${l.text} · ${s2}`, mini: m } : { ...l, mini: m };
  });
}

export const studyShort = (L) => (L ? `${CONDITION_NAMES[L.condition] || L.condition} · ${String(L.study || '').replace(/^\d{8}-\d{4}-/, '')}` : '');
