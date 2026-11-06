// In-scene chips (Broadcast): rings on the rig and callouts above it, re-projected every frame.
// Only exceptional states get a marker; labels are placed by priority, stacked when they collide,
// and dropped (ring kept) when there is no room. Name flags are the lowest priority.
const RING = { bad: [44, 24], jk: [40, 22], warn: [58, 30] };
const MAX_LABELS = 7, MAX_RINGS = 10, STEM = 22;

export function createChips(root) {
  const pool = new Map();
  let shown = [];
  const el = (key, cls) => {
    let e = pool.get(key);
    if (!e) {
      const d = document.createElement('div');
      d.innerHTML = '<div class="tag"></div><div class="stem"></div>';
      root.appendChild(d);
      e = { d, tag: d.firstChild, stem: d.lastChild, html: null, cls: null, used: false };
      pool.set(key, e);
    }
    if (e.cls !== cls) { e.cls = cls; e.d.className = cls; }
    return e;
  };
  const ringEl = (key, cls) => {
    let e = pool.get(key);
    if (!e) { const d = document.createElement('div'); root.appendChild(d); e = { d, cls: null, used: false }; pool.set(key, e); }
    const c = `ring ${cls}`;
    if (e.cls !== c) { e.cls = c; e.d.className = c; }
    return e;
  };
  const overlaps = (a, list) => list.some((b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y);

  /**
   * cands: [{ key, prio, cls ('bad'|'jk'|'warn'|'think'|'ok'|'flag'), html, anchor: [x, y], ring: {cls, at: [x, y]} | null, color, info }]
   * occupied: screen rects the labels must avoid (HUD cards); W, H: viewport size.
   */
  function render(cands, occupied, W, H) {
    for (const e of pool.values()) e.used = false;
    shown = [];
    const placed = occupied.map((r) => ({ ...r }));
    let nl = 0, nr = 0;
    cands.sort((a, b) => b.prio - a.prio);
    for (const c of cands) {
      if (c.ring && nr < MAX_RINGS) {
        const [w, h] = RING[c.ring.cls] || RING.bad;
        const [x, y] = c.ring.at;
        if (x > -w && x < W + w && y > -h && y < H + h) {
          const r = ringEl(`${c.key}:ring`, c.ring.cls);
          r.used = true;
          r.d.style.transform = `translate(${(x - w / 2).toFixed(1)}px, ${(y - h / 2).toFixed(1)}px)`;
          nr++;
        }
      }
      if (!c.html || nl >= MAX_LABELS) continue;
      const [ax, ay] = c.anchor;
      if (ax < 0 || ax > W || ay < 0 || ay > H) continue;
      const e = el(`${c.key}:label`, `callout ${c.cls}`);
      if (e.html !== c.html) { e.html = c.html; e.tag.innerHTML = c.html; }
      if (c.color) e.d.style.setProperty('--c', c.color);
      e.d.style.display = '';
      const w = e.tag.offsetWidth, h = e.tag.offsetHeight;
      let ok = null;
      // above the anchor first (stacking higher, then sideways); the focused/critical ones may also hang below it
      const below = c.prio >= 5;
      // compact tags stay close to their rig (or are dropped) instead of climbing into a tower
      const maxK = /compact|mini/.test(c.cls) ? 2 : 4;
      for (let k = 0; k < maxK && !ok; k++) {
        const stem = STEM + k * (h + 6);
        for (const dx of [0, -0.35 * w, 0.35 * w, -0.7 * w, 0.7 * w]) {
          const ups = { x: ax - w / 2 + dx, y: ay - stem - h, w, h };
          const downs = { x: ax - w / 2 + dx, y: ay + stem, w, h };
          for (const [box, down] of below ? [[ups, false], [downs, true]] : [[ups, false]]) {
            if (box.x < 4 || box.x + w > W - 4 || box.y < 4 || box.y + h > H - 4) continue;
            if (overlaps(box, placed)) continue;
            ok = { box, stem, dx, down };
            break;
          }
          if (ok) break;
        }
      }
      if (!ok) { e.d.style.display = 'none'; continue; }
      e.used = true;
      placed.push(ok.box);
      e.stem.style.height = `${ok.stem}px`;
      e.d.classList.toggle('below', !!ok.down);
      e.stem.style.transform = ok.dx ? `translateX(${(-ok.dx).toFixed(1)}px)` : '';
      e.d.style.transform = `translate(${ok.box.x.toFixed(1)}px, ${(ok.down ? ok.box.y - ok.stem : ok.box.y).toFixed(1)}px)`;
      e.d.style.width = `${w}px`;
      nl++;
      shown.push({ key: c.key, cls: c.cls, text: e.tag.textContent, x: ax, y: ay, ...(c.info || {}) });
    }
    for (const e of pool.values()) if (!e.used) e.d.style.display = 'none'; else if (e.d.style.display === 'none') e.d.style.display = '';
    // rings never report; record them for the hooks
    for (const [k, e] of pool) if (e.used && k.endsWith(':ring')) shown.push({ key: k, cls: e.cls, ring: true });
  }
  function clear() { for (const e of pool.values()) { e.d.style.display = 'none'; e.used = false; } shown = []; }
  return { render, clear, shown: () => shown };
}
