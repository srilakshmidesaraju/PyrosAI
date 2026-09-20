/* Pyros-AI — Thermal Source Operations Dashboard
 *
 * Three linked views over the pipeline output: a risk-ranked queue, a map, and
 * a detail panel. No framework, no build step.
 *
 * This dashboard is driven live in front of an audience, so the render path is
 * built around a few rules:
 *
 *   - every per-event string (queue row, SVG chart path, histogram bars, score
 *     bars, evidence list) is built ONCE in prepare() and cached on the event
 *     object. Selecting an event is then string concatenation plus a single
 *     innerHTML assignment, never numeric work.
 *   - markers are clustered above CLUSTER_THRESHOLD events, with chunked
 *     loading so adding several hundred of them never blocks the main thread.
 *   - filter clicks are debounced; map-driven DOM updates go through rAF.
 *   - the queue uses one delegated listener, not one per row.
 */
'use strict';

const TYPE_COLOR = {
  'Industrial Fire':      '#ef4444',
  'Gas Flare':            '#3b82f6',
  'Wildfire':             '#f97316',
  'Agricultural Burn':    '#eab308',
  'Insufficient Evidence':'#64748b',
};
const TYPE_ORDER = ['Industrial Fire', 'Gas Flare', 'Wildfire',
                    'Agricultural Burn', 'Insufficient Evidence'];
const DOW = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
const NIGHT = new Set([0,1,2,3,4,5,6,20,21,22,23]);

const CLUSTER_THRESHOLD = 100;   // above this, cluster markers
const FILTER_DEBOUNCE_MS = 150;
const TILE_TIMEOUT_MS = 4500;
const TILE_ERROR_LIMIT = 5;

const state = {
  data: null, events: [], byId: new Map(), markers: new Map(),
  selected: null, hidden: new Set(),
  map: null, layer: null, clustered: false,
  basemap: null, activeSpec: null, tiles: null, labels: null,
  vectorBase: null, tileGen: 0, filterTimer: 0, rafPending: 0,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
  (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const colorOf = (t) => TYPE_COLOR[t] || TYPE_COLOR['Insufficient Evidence'];
const pad2 = (n) => String(n).padStart(2, '0');

/* ==================================================================== boot */

function load() {
  // 1. offline payload injected by results.data.js
  if (window.PYROS_DATA) return Promise.resolve(window.PYROS_DATA);
  // 2. results.json served over http
  return fetch('./results.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    // 3. the FastAPI backend
    .catch(() => fetch('/api/events?limit=5000')
      .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then((j) => ({ meta: { event_count: j.count }, events: j.events })));
}

load().then(init).catch((err) => {
  $('detail-body').innerHTML =
    '<div id="load-error"><strong>Could not load results.</strong><br><br>' +
    'Run the pipeline first:<br><code>python run_pipeline.py --fetch --osm</code>' +
    '<br><br>That writes <code>frontend/results.json</code> and ' +
    '<code>frontend/results.data.js</code>.<br><br>' +
    '<span style="color:var(--text-faint)">' + esc(err.message) + '</span></div>';
});

function init(data) {
  const t0 = performance.now();
  state.data = data;
  state.events = data.events || [];
  state.events.forEach((e) => state.byId.set(e.id, e));

  prepare(state.events);                 // one pass, everything cached
  renderHeader(data);
  buildFilters();
  renderQueue();
  buildMap(data);
  wireEvents();

  const wanted = new URLSearchParams(location.search).get('event')
              || decodeURIComponent(location.hash.replace(/^#/, ''));
  if (wanted && state.byId.has(wanted)) select(wanted);

  console.info(`[pyros] ${state.events.length} events ready in ` +
               `${(performance.now() - t0).toFixed(0)} ms`);
}

/* ============================================ pre-computation (once, at load) */

function prepare(events) {
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    const c = colorOf(e.classification);
    e._c = c;
    e._chart = frpChart(e, c);           // SVG path strings, never recomputed
    e._hours = barsHtml(e.hour_histogram || [], 24, (h) => NIGHT.has(h));
    e._dow = barsHtml(e.dow_histogram || [], 7, (d) => d >= 5);
    e._scores = scoreRows(e);
    e._evidence = (e.evidence || [])
      .map((x) => `<li>${esc(x)}</li>`).join('') || '<li>No criteria satisfied.</li>';
    e._row = queueRow(e, c);
  }
}

function queueRow(e, c) {
  // An isolation-forest-only flag has no meaningful deviation multiple (it is
  // a population comparison, not a departure from the site's own baseline), so
  // showing "1.0x" next to an ANOMALY badge would read as a contradiction.
  let badge = e.anomaly_flag
    ? `<span class="badge">${+e.deviation_mult >= 1.5
         ? 'Anomaly ' + (+e.deviation_mult).toFixed(1) + '×' : 'Outlier'}</span>`
    : (e.classification === 'Insufficient Evidence'
        ? '<span class="badge review">Review</span>' : '');
  if (e.flare_validated === true) badge += '<span class="badge verified">Verified</span>';
  return `<div class="row" data-id="${esc(e.id)}">
    <div class="rank mono">${e.rank}</div>
    <div class="body">
      <div class="name" title="${esc(e.name)}">${esc(e.name)}</div>
      <div class="meta">
        <span class="tdot" style="background:${c}"></span>
        <span class="tname">${esc(e.classification)}</span>
        <span class="id mono">${esc(e.id)}</span>
      </div>${badge}
    </div>
    <div class="risk">
      <div class="n mono" style="color:${c}">${Math.round(e.risk_score)}</div>
      <div class="bar"><i style="width:${Math.min(100, e.risk_score)}%;background:${c}"></i></div>
    </div>
  </div>`;
}

function barsHtml(hist, n, isAlt) {
  let max = 0;
  for (let i = 0; i < n; i++) max = Math.max(max, hist[i] || 0);
  if (max <= 0) max = 1;
  let out = '<div class="bars">';
  for (let i = 0; i < n; i++) {
    const h = Math.max(1, ((hist[i] || 0) / max) * 100);
    out += `<div class="b${isAlt(i) ? ' night' : ' hot'}" style="height:${h.toFixed(1)}%"></div>`;
  }
  return out + '</div>';
}

function scoreRows(e) {
  const scores = e.class_scores || {};
  return Object.keys(scores)
    .sort((a, b) => scores[b] - scores[a])
    .map((name) => {
      const v = scores[name] || 0;
      const win = name === e.classification;
      return `<div class="score-row${win ? ' win' : ''}">
        <span class="sn">${esc(name)}</span>
        <span class="st"><i style="width:${(v * 100).toFixed(0)}%;background:${colorOf(name)}"></i></span>
        <span class="sv">${v.toFixed(2)}</span>
      </div>`;
    }).join('');
}

function frpChart(e, color) {
  const h = e.frp_history || [];
  if (!h.length) return '<div class="chart-note">No FRP history.</div>';

  const W = 356, H = 104, PL = 30, PR = 6, PT = 8, PB = 14;
  const iw = W - PL - PR, ih = H - PT - PB;
  let peak = 0;
  for (let i = 0; i < h.length; i++) peak = Math.max(peak, h[i]);
  const yMax = Math.max(peak * 1.12, 0.5);

  const x = (i) => PL + (h.length === 1 ? iw / 2 : (i / (h.length - 1)) * iw);
  const y = (v) => PT + ih - (v / yMax) * ih;

  let line = '';
  for (let i = 0; i < h.length; i++) line += `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(h[i]).toFixed(1)}`;

  const single = h.length === 1
    ? `<circle cx="${x(0).toFixed(1)}" cy="${y(h[0]).toFixed(1)}" r="3.5" fill="${color}"/>` : '';
  const area = h.length > 1
    ? `<path class="area" d="${line}L${x(h.length-1).toFixed(1)},${(PT+ih).toFixed(1)}L${x(0).toFixed(1)},${(PT+ih).toFixed(1)}Z"/>` : '';

  // Mark the readings that actually tripped the z-score test.
  const mean = +e.baseline_mean || 0, sd = Math.max(+e.baseline_std || 0, 0.01);
  let spikes = '';
  if (e.baseline_established) {
    for (let i = 0; i < h.length; i++) {
      if (Math.abs((h[i] - mean) / sd) > 3) {
        spikes += `<circle class="spike" cx="${x(i).toFixed(1)}" cy="${y(h[i]).toFixed(1)}" r="3"/>`;
      }
    }
  }
  const base = e.baseline_established
    ? `<line class="base" x1="${PL}" y1="${y(mean).toFixed(1)}" x2="${W-PR}" y2="${y(mean).toFixed(1)}"/>
       <text x="${W-PR}" y="${(y(mean)-3).toFixed(1)}" text-anchor="end">baseline ${mean.toFixed(1)}</text>` : '';

  let ticks = '';
  for (const v of [0, yMax / 2, yMax]) {
    ticks += `<line class="grid" x1="${PL}" y1="${y(v).toFixed(1)}" x2="${W-PR}" y2="${y(v).toFixed(1)}"/>
              <text x="${PL-5}" y="${(y(v)+3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`;
  }

  return `<div style="color:${color}"><svg class="chart" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Fire radiative power history, peak ${peak.toFixed(0)} megawatts">
    ${ticks}${area}<path class="line" d="${line}"/>${single}${base}${spikes}
    <text x="${PL-5}" y="${H-2}" text-anchor="end">MW</text>
  </svg></div>`;
}

/* ================================================================== header */

function renderHeader(d) {
  const m = d.meta || {};
  $('region-sub').textContent = [m.region, m.phase].filter(Boolean).join(' · ');
  $('s-events').textContent = (m.event_count ?? state.events.length).toLocaleString();
  $('s-dets').textContent = (m.detection_count ?? 0).toLocaleString();
  $('s-anom').textContent = (m.anomaly_count
    ?? state.events.filter((e) => e.anomaly_flag).length).toLocaleString();
  $('s-verified').textContent = (m.flares_validated ?? 0).toLocaleString();
  $('s-verified').title = `${m.flares_validated ?? 0} gas flare calls confirmed by the ` +
    `World Bank registry; ${m.flares_unvalidated ?? 0} not listed`;
  $('s-window').textContent = (m.window_days ? m.window_days + 'd' : '—');
  if (m.window_start) {
    $('s-window').title = `${m.window_start} → ${m.window_end} (${m.timezone || 'IST'})`;
  }
}

/* =================================================================== queue */

function buildFilters() {
  const present = TYPE_ORDER.filter((t) => state.events.some((e) => e.classification === t));
  $('filters').innerHTML = present.map((t) =>
    `<button class="chip on" data-type="${esc(t)}">
       <span class="sw" style="background:${colorOf(t)}"></span>${esc(t)}
     </button>`).join('');
}

function visibleEvents() {
  return state.hidden.size
    ? state.events.filter((e) => !state.hidden.has(e.classification))
    : state.events;
}

function renderQueue() {
  const rows = visibleEvents();
  $('queue-count').textContent = `${rows.length.toLocaleString()} / ${state.events.length.toLocaleString()}`;

  // Events arrive already sorted by risk from export.py, and every row string
  // was built in prepare(), so this is a join and one innerHTML assignment.
  let html = '';
  for (let i = 0; i < rows.length; i++) html += rows[i]._row;
  $('queue').innerHTML = html;
  markQueueSelection();
}

function markQueueSelection() {
  const prev = $('queue').querySelector('.row.sel');
  if (prev) prev.classList.remove('sel');
  if (!state.selected) return;
  const row = $('queue').querySelector(`.row[data-id="${CSS.escape(state.selected)}"]`);
  if (row) { row.classList.add('sel'); row.scrollIntoView({ block: 'nearest' }); }
}

function wireEvents() {
  // One delegated listener for all rows, however many there are.
  $('queue').addEventListener('click', (ev) => {
    const row = ev.target.closest('.row');
    if (row) select(row.dataset.id);
  });

  $('filters').addEventListener('click', (ev) => {
    const chip = ev.target.closest('.chip');
    if (!chip) return;
    const type = chip.dataset.type;
    if (state.hidden.has(type)) { state.hidden.delete(type); chip.classList.add('on'); chip.classList.remove('off'); }
    else { state.hidden.add(type); chip.classList.remove('on'); chip.classList.add('off'); }

    // Debounced: rapid chip toggling rebuilds the queue and the marker layer
    // once, after the user stops clicking, instead of on every click.
    clearTimeout(state.filterTimer);
    state.filterTimer = setTimeout(() => {
      renderQueue();
      refreshMarkers();
    }, FILTER_DEBOUNCE_MS);
  });

  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') select(null); });
}

/* ===================================================================== map */

const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services';
const ESRI_ATTR = 'Tiles &copy; <a href="https://www.esri.com/">Esri</a>';

/* Basemap registry — the switcher buttons, the status label, the tile filter
 * and the fallback chain all read this one array.
 *
 * The Dark layer is Esri's Dark Gray Canvas, NOT CartoDB dark_all. CartoDB
 * still answers HTTP 200 without an API key, so it looks fine in a network
 * log, but the PNG it returns is stamped "API KEY REQUIRED" across every tile.
 * Verified by fetching z7/x92/y58 directly. If you obtain a Carto key, add it
 * as an entry here — nothing else needs to change. */
const BASEMAPS = [
  { key: 'dark', label: 'Dark', status: 'Dark Ops', filter: 'dim',
    url: `${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    labels: `${ESRI}/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`,
    maxNativeZoom: 16, attribution: `${ESRI_ATTR} &mdash; Dark Gray Canvas` },
  { key: 'satellite', label: 'Satellite', status: 'Satellite', filter: null,
    url: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
    labels: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
    maxNativeZoom: 18, attribution: `${ESRI_ATTR} &mdash; World Imagery` },
  { key: 'terrain', label: 'Terrain', status: 'Terrain', filter: null,
    url: `${ESRI}/World_Topo_Map/MapServer/tile/{z}/{y}/{x}`,
    maxNativeZoom: 18, attribution: `${ESRI_ATTR} &mdash; World Topo` },
];
const OSM_FALLBACK = {
  key: 'osm', label: 'OSM', status: 'OSM fallback', filter: 'invert',
  url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  maxNativeZoom: 18, attribution: '&copy; OpenStreetMap contributors',
};

function buildMap(d) {
  const map = L.map('map', {
    center: [15.5, 80.5], zoom: 7, maxZoom: 18,
    preferCanvas: true, zoomControl: true, worldCopyJump: false,
  });
  state.map = map;

  map.createPane('basemapLabels');
  map.getPane('basemapLabels').style.zIndex = 250;
  map.getPane('basemapLabels').style.pointerEvents = 'none';

  state.vectorBase = fallbackGrid().addTo(map);

  // Pausing the anomaly pulse for the duration of a pan/zoom is the single
  // biggest frame-time win available here. Toggled via a class so the work is
  // one classList call, not a walk over every marker.
  const container = map.getContainer();
  map.on('movestart zoomstart', () => container.classList.add('map-moving'));
  map.on('moveend zoomend', () => container.classList.remove('map-moving'));

  state.clustered = state.events.length > CLUSTER_THRESHOLD;
  state.layer = state.clustered ? makeClusterGroup() : L.layerGroup();
  state.layer.addTo(map);

  for (const e of state.events) state.markers.set(e.id, makeMarker(e));
  refreshMarkers();

  renderBasemapSwitch();
  const wanted = new URLSearchParams(location.search).get('basemap');
  selectBasemap(BASEMAPS.some((b) => b.key === wanted) ? wanted : BASEMAPS[0].key);
}

function makeClusterGroup() {
  return L.markerClusterGroup({
    // chunkedLoading adds markers across animation frames instead of in one
    // synchronous burst, which is what keeps several hundred of them from
    // freezing the map on first paint.
    chunkedLoading: true, chunkInterval: 50, chunkDelay: 10,
    maxClusterRadius: 55, disableClusteringAtZoom: 12,
    removeOutsideVisibleBounds: true, showCoverageOnHover: false,
    spiderfyOnMaxZoom: true,
    // Cluster split/merge animation re-lays-out every icon on each zoom. It is
    // decorative, and switching it off is the difference between a smooth zoom
    // and a visible hitch with several hundred markers on screen.
    animate: false, animateAddingMarkers: false,
    iconCreateFunction(cluster) {
      const kids = cluster.getAllChildMarkers();
      let anom = false;
      for (let i = 0; i < kids.length; i++) if (kids[i].options.pyrosAnom) { anom = true; break; }
      const n = cluster.getChildCount();
      const size = n < 10 ? 30 : n < 50 ? 36 : 44;
      return L.divIcon({
        className: '',
        html: `<div class="pyros-cluster${anom ? ' has-anom' : ''}"
                    style="width:${size}px;height:${size}px">${n}</div>`,
        iconSize: [size, size],
      });
    },
  });
}

function makeMarker(e) {
  const icon = L.divIcon({
    className: '',
    html: `<div class="pin${e.anomaly_flag ? ' anom' : ''}" style="color:${e._c}"><span class="dot"></span></div>`,
    iconSize: [18, 18], iconAnchor: [9, 9],
  });
  // riseOnHover:false — reordering z-index on hover forces restacking of every
  // marker in the pane, which is exactly the kind of jitter to avoid here.
  return L.marker([e.lat, e.lon], { icon, riseOnHover: false, pyrosAnom: !!e.anomaly_flag })
    .bindTooltip(
      `<strong>${esc(e.name)}</strong><br>${esc(e.classification)} · risk ${Math.round(e.risk_score)}` +
      (e.anomaly_flag ? `<br><span style="color:#fca5a5">${+e.deviation_mult >= 1.5
        ? 'ANOMALY ' + (+e.deviation_mult).toFixed(1) + '×' : 'OUTLIER'}</span>` : ''),
      { direction: 'top', offset: [0, -8] })
    .on('click', () => select(e.id));
}

function refreshMarkers() {
  const layer = state.layer;
  layer.clearLayers();
  const shown = visibleEvents();
  const batch = [];
  for (let i = 0; i < shown.length; i++) {
    const m = state.markers.get(shown[i].id);
    if (m) batch.push(m);
  }
  if (state.clustered) layer.addLayers(batch);     // chunked internally
  else for (const m of batch) layer.addLayer(m);
}

function fallbackGrid() {
  const style = { color: '#18232f', weight: 1, fill: false, interactive: false };
  const parts = [];
  for (let lat = 8; lat <= 24; lat++) parts.push(L.polyline([[lat, 70], [lat, 92]], style));
  for (let lon = 70; lon <= 92; lon++) parts.push(L.polyline([[8, lon], [24, lon]], style));
  parts.push(L.rectangle([[12.6, 76.7], [19.2, 84.8]], {
    color: '#2a5573', weight: 1, dashArray: '5 4', fill: true,
    fillColor: '#0c1720', fillOpacity: .55, interactive: false,
  }));
  return L.layerGroup(parts);
}

/* -------------------------------------------------------- basemap switch */

function renderBasemapSwitch() {
  $('basemap-switch').innerHTML = BASEMAPS.map((b) =>
    `<button class="bm-btn" data-key="${b.key}">${esc(b.label)}</button>`).join('');
  $('basemap-switch').addEventListener('click', (ev) => {
    const btn = ev.target.closest('.bm-btn');
    if (btn) selectBasemap(btn.dataset.key);
  });
}

function selectBasemap(key) {
  const spec = BASEMAPS.find((b) => b.key === key) || BASEMAPS[0];
  state.basemap = spec.key;
  $('basemap-switch').querySelectorAll('.bm-btn').forEach((b) =>
    b.classList.toggle('on', b.dataset.key === spec.key));
  mountBasemap(spec);
}

function mountBasemap(spec) {
  const map = state.map;
  const gen = ++state.tileGen;                  // stale timers from earlier mounts no-op

  for (const layer of [state.tiles, state.labels]) {
    if (layer && map.hasLayer(layer)) map.removeLayer(layer);
  }
  state.tiles = state.labels = null;
  state.activeSpec = spec;
  applyTileFilter(spec.filter);

  let tiles;
  try {
    tiles = L.tileLayer(spec.url, {
      maxZoom: 18, maxNativeZoom: spec.maxNativeZoom,
      attribution: spec.attribution, updateWhenIdle: false, keepBuffer: 2,
    });
    if (spec.labels) {
      state.labels = L.tileLayer(spec.labels, {
        maxZoom: 18, maxNativeZoom: spec.maxNativeZoom, pane: 'basemapLabels',
      }).addTo(map);
    }
  } catch (err) {
    console.warn('basemap construction failed', err);
    degrade(gen, spec);
    return;
  }

  // Leaflet surfaces a dead tile server as a 'tileerror' event rather than a
  // thrown exception, so this handler is the real equivalent of a catch here.
  let ok = false, errors = 0;
  tiles.on('tileload', () => {
    if (ok || gen !== state.tileGen) return;
    ok = true;
    if (map.hasLayer(state.vectorBase)) map.removeLayer(state.vectorBase);
    setStatus('online', spec.status);
  });
  tiles.on('tileerror', () => {
    if (ok || gen !== state.tileGen) return;
    if (++errors >= TILE_ERROR_LIMIT) degrade(gen, spec);
  });

  state.tiles = tiles.addTo(map);
  setStatus(spec.key === 'osm' ? 'offline' : 'pending', spec.status + '…');
  setTimeout(() => { if (!ok && gen === state.tileGen) degrade(gen, spec); }, TILE_TIMEOUT_MS);
}

function degrade(gen, spec) {
  if (gen !== state.tileGen) return;
  if (spec.key !== OSM_FALLBACK.key) { mountBasemap(OSM_FALLBACK); return; }
  const map = state.map;
  for (const layer of [state.tiles, state.labels]) {
    if (layer && map.hasLayer(layer)) map.removeLayer(layer);
  }
  state.tiles = state.labels = null;
  if (!map.hasLayer(state.vectorBase)) state.vectorBase.addTo(map);
  applyTileFilter(null);
  setStatus('offline', 'Offline basemap');
}

function applyTileFilter(name) {
  const el = state.map.getContainer();
  el.classList.remove('tf-dim', 'tf-invert');
  if (name) el.classList.add('tf-' + name);
}

function setStatus(cls, text) {
  $('basemap-status').className = cls;
  $('basemap-text').textContent = text;
}

/* ================================================================== select */

function select(id) {
  state.selected = id;
  const hash = id ? '#' + id : '';
  if (location.hash !== hash) history.replaceState(null, '', location.pathname + location.search + hash);

  // Marker highlight and queue highlight are pure class toggles, batched into
  // one frame so a click never lands mid-paint.
  cancelAnimationFrame(state.rafPending);
  state.rafPending = requestAnimationFrame(() => {
    state.markers.forEach((m, mid) => {
      const el = m.getElement();
      if (el) {
        const pin = el.firstElementChild;
        if (pin) pin.classList.toggle('sel', mid === id);
      }
    });
    markQueueSelection();
  });

  if (!id) { renderPlaceholder(); return; }
  const e = state.byId.get(id);
  if (!e) return;
  state.map.panTo([e.lat, e.lon], { animate: true, duration: .4 });
  renderDetail(e);
}

function renderPlaceholder() {
  $('detail-id').textContent = '';
  $('detail-body').innerHTML =
    '<div id="placeholder"><div class="glyph">◎</div>' +
    '<p>Select an event from the priority queue<br>or click a marker on the map.</p></div>';
}

/* ================================================================== detail */

function renderDetail(e) {
  const c = e._c;
  $('detail-id').textContent = e.id;

  // Single innerHTML assignment. Everything numeric was precomputed in
  // prepare(); sections marked `lazy` get content-visibility:auto so the
  // browser defers their layout until they scroll into view.
  $('detail-body').innerHTML = `
    <div class="d-head">
      <div class="eid mono">${esc(e.id)} · RISK RANK ${e.rank} OF ${state.events.length}</div>
      <h2>${esc(e.name)}</h2>
      <div class="d-type"><span class="sw" style="background:${c}"></span>
        <span class="label" style="color:${c}">${esc(e.classification)}</span></div>
      <div class="meter">
        <div class="meter-top"><span>Classification confidence</span>
          <span class="val mono" style="color:${c}">${(e.confidence * 100).toFixed(0)}%</span></div>
        <div class="meter-bar"><i style="width:${(e.confidence * 100).toFixed(0)}%;background:${c}"></i></div>
      </div>
      <div class="meter">
        <div class="meter-top"><span>Risk score</span>
          <span class="val mono" style="color:${c}">${e.risk_score.toFixed(1)}</span></div>
        <div class="meter-bar"><i style="width:${Math.min(100, e.risk_score)}%;background:${c}"></i></div>
      </div>
    </div>

    <section class="d-sec"><h3>Why this classification</h3>
      <p class="reason">${esc(e.reason)}</p></section>

    ${anomalyBlock(e)}
    ${validationBlock(e)}

    <section class="d-sec"><h3>FRP history · ${(e.frp_history || []).length} detections</h3>
      ${e._chart}
      <div class="chart-note"><span>${esc((e.frp_dates || [])[0] || '')}</span>
        <span>peak ${(+e.frp_peak).toFixed(1)} MW</span>
        <span>${esc((e.frp_dates || [])[(e.frp_dates || []).length - 1] || '')}</span></div>
    </section>

    <section class="d-sec lazy"><h3>Thermal rhythm · hour of day (IST)</h3>
      ${e._hours}
      <div class="bars-axis"><span>00</span><span>06</span><span>12</span><span>18</span><span>23</span></div>
      <div class="chart-note" style="margin-top:8px">
        <span>Night share ${(e.night_share * 100).toFixed(0)}%</span>
        <span>Weekend share ${(e.weekend_share * 100).toFixed(0)}%</span></div>
    </section>

    <section class="d-sec lazy"><h3>Day of week</h3>${e._dow}
      <div class="bars-axis">${DOW.map((d) => `<span>${d}</span>`).join('')}</div></section>

    <section class="d-sec lazy"><h3>Behavioural features</h3>
      <dl class="kv">
        <dt>Persistence</dt><dd>${(e.persistence * 100).toFixed(0)}%</dd>
        <dt>Active span</dt><dd>${e.active_days} / ${e.date_range} days</dd>
        <dt>Duty cycle</dt><dd>${(e.duty_cycle * 100).toFixed(0)}%</dd>
        <dt>FRP mean / peak</dt><dd>${(+e.frp_mean).toFixed(1)} / ${(+e.frp_peak).toFixed(1)} MW</dd>
        <dt>FRP variance</dt><dd>${(+e.frp_variance).toFixed(2)}</dd>
        <dt>Robust variation</dt><dd>${(+e.frp_robust_var).toFixed(3)}</dd>
        <dt>Spread rate</dt><dd>${e.spread_rate >= 0 ? '+' : ''}${(+e.spread_rate).toFixed(2)} km/day</dd>
        <dt>Footprint</dt><dd>${(+e.footprint_km).toFixed(2)} km</dd>
        <dt>Neighbours &lt; 5 km</dt><dd>${e.neighbour_count ?? 0}</dd>
        <dt>Detections</dt><dd>${e.detection_count}</dd>
        <dt>Nearest facility</dt><dd>${(e.dist_to_facility_m / 1000).toFixed(2)} km</dd>
        <dt>Facility type</dt><dd>${esc((e.facility_type || '—').replace(/_/g, ' '))}</dd>
        <dt>Coordinates</dt><dd>${(+e.lat).toFixed(4)}, ${(+e.lon).toFixed(4)}</dd>
      </dl>
      <div style="margin-top:9px">
        ${(e.sensors || []).map((s) => `<span class="tag">${esc(s)}</span>`).join('')}
        ${(e.satellites || []).map((s) => `<span class="tag">${esc(s)}</span>`).join('')}
      </div>
    </section>

    <section class="d-sec lazy"><h3>Class scores</h3>${e._scores}
      <div class="chart-note" style="margin-top:8px">
        <span>Margin over runner-up</span><span>${(+e.margin).toFixed(3)}</span></div></section>

    <section class="d-sec lazy"><h3>Supporting evidence</h3>
      <ul class="evidence">${e._evidence}</ul></section>

    ${cnnBlock(e)}
  `;
  $('detail-body').scrollTop = 0;
}

function anomalyBlock(e) {
  if (e.anomaly_flag) {
    const kinds = [];
    if (Math.abs(+e.z_score) > 3) kinds.push('FRP spike');
    if ((e.offhours || []).length) kinds.push('off-hours');
    if (+e.iso_score > 0.75) kinds.push('profile outlier');
    return `<section class="d-sec"><h3>Baseline deviation</h3>
      <div class="alert-box">
        <div class="alert-head">
          <span class="t">⚠ ${esc(kinds.join(' + ') || 'anomaly')}</span>
          <span class="dev mono">${+e.deviation_mult >= 1.5
            ? (+e.deviation_mult).toFixed(1) + '×'
            : 'iso ' + (+e.iso_score).toFixed(2)}</span>
        </div>
        <p>${esc(e.anomaly_desc)}</p>
      </div>
      <dl class="kv" style="margin-top:10px">
        <dt>Site baseline</dt><dd>${(+e.baseline_mean).toFixed(1)} MW</dd>
        <dt>Baseline σ</dt><dd>${(+e.baseline_std).toFixed(2)}</dd>
        <dt>z-score</dt><dd>${(+e.z_score).toFixed(2)}</dd>
        <dt>Isolation score</dt><dd>${(+e.iso_score).toFixed(2)}</dd>
        ${(e.offhours || []).length ? `<dt>Off-hours</dt><dd>${e.offhours.map(pad2).join(', ')}:00</dd>` : ''}
      </dl></section>`;
  }
  return `<section class="d-sec"><h3>Baseline deviation</h3>
    <div class="alert-box clear">
      <div class="alert-head"><span class="t">${e.baseline_established ? 'Nominal' : 'No baseline'}</span></div>
      <p>${esc(e.anomaly_desc)}</p>
    </div></section>`;
}

/* World Bank Global Gas Flaring registry cross-check. Four states: confirmed,
 * classified-but-unlisted, a registry flare nearby under a different label, and
 * nothing to say — in which case the section is omitted entirely rather than
 * rendered empty. */
function validationBlock(e) {
  const m = e.wb_match;

  if (e.flare_validated === true && m) {
    const vol = m.volume_m3 ? (m.volume_m3 / 1e6).toFixed(1) + ' Mm³ total' : '—';
    return `<section class="d-sec"><h3>Registry validation</h3>
      <div class="val-box ok">
        <div class="val-head"><span class="t">✓ Confirmed by World Bank flare registry</span></div>
        <p><strong>${esc(m.field_name)}</strong> — ${esc(m.operator)}</p>
      </div>
      <dl class="kv" style="margin-top:10px">
        <dt>Distance</dt><dd>${Math.round(m.distance_m)} m</dd>
        <dt>Years active</dt><dd>${m.years_active ?? '—'}${m.year_last ? ' (to ' + m.year_last + ')' : ''}</dd>
        <dt>Flare level</dt><dd>${esc(m.flare_level || '—')}</dd>
        <dt>Recorded volume</dt><dd>${vol}</dd>
      </dl></section>`;
  }

  if (e.flare_validated === false) {
    return `<section class="d-sec"><h3>Registry validation</h3>
      <div class="val-box warn">
        <div class="val-head"><span class="t">⚠ Not listed in World Bank registry</span></div>
        <p>Classified as a gas flare, but no documented flare site within 2 km.
           The registry covers oil and gas sector flaring only — small, industrial
           or recently commissioned flares may be unlisted, so this is a prompt
           for review rather than evidence of error.</p>
      </div></section>`;
  }

  if (e.flare_validated === null && m) {
    return `<section class="d-sec"><h3>Registry note</h3>
      <div class="val-box note">
        <div class="val-head"><span class="t">Registry flare nearby</span></div>
        <p>The World Bank registry lists a documented flare
           <strong>${Math.round(m.distance_m)} m</strong> away
           (<strong>${esc(m.field_name)}</strong>, ${esc(m.operator)}).
           Our classification differs — this warrants review.</p>
      </div></section>`;
  }

  return '';                       // nothing known either way: omit the section
}

function cnnBlock(e) {
  if (!e.cnn_status || e.cnn_status === 'no_image' || e.cnn_status === 'unavailable') return '';
  const img = e.cnn_image
    ? `<img class="cnn-img" src="${esc(e.cnn_image)}" alt="Satellite imagery at this coordinate" loading="lazy">` : '';
  const result = e.cnn_result
    ? `<dl class="kv"><dt>Land cover</dt><dd>${esc(e.cnn_result)}</dd>
         <dt>CNN confidence</dt><dd>${((+e.cnn_confidence) * 100).toFixed(0)}%</dd></dl>`
    : '';
  return `<section class="d-sec lazy"><h3>Imagery verification</h3>
    ${img}${result}
    <p class="reason" style="margin-top:8px">${esc(e.cnn_context || '')}</p>
    <div class="chart-note" style="margin-top:6px"><span>source</span><span>${esc(e.cnn_status)}</span></div>
  </section>`;
}
