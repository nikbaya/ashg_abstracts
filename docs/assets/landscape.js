// Landscape page: every ASHG abstract from 2022 to 2026 as a dot at its UMAP position (landscape.py), on one canvas.
// Search an author to highlight their abstracts, fly to their weighted centroid and draw their trajectory through the
// years they published in, with a web of co-authors or similar abstracts; paint keywords and institutions as density or
// dots; toggle years to show only some of them. Data has no titles or abstract text.
(async function () {
  const root = document.getElementById("landscape");
  if (!root || typeof d3 === "undefined") return;
  const $ = id => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // Data and the inverted index: author name -> [{abstract_id, weight}]
  // ---------------------------------------------------------------------------
  const WEIGHTS = { first: 1.5, middle: 0.5, last: 1.0 };
  const SIM_FLOOR = 10; // text matches below this percent are not drawn (same floor as Find talks)

  let abstracts, synonyms = {};
  try {
    const [rows, syn] = await Promise.all([
      fetch(root.dataset.src).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch(root.dataset.syn).then(r => r.ok ? r.json() : {}).catch(() => ({})),
    ]);
    abstracts = rows;
    synonyms = syn;
  } catch (err) {
    $("ls-info").textContent = "The landscape data could not be loaded.";
    return;
  }

  function buildInvertedIndex(rows) {
    const index = new Map();
    for (const a of rows) {
      for (const { name, position } of a.authors) {
        const w = WEIGHTS[position];
        let list = index.get(name);
        if (!list) index.set(name, list = []);
        const last = list[list.length - 1];
        if (last && last.abstract_id === a.id) last.weight = Math.max(last.weight, w); // a name listed twice on one abstract
        else list.push({ abstract_id: a.id, weight: w });
      }
    }
    return index;
  }

  const N = abstracts.length;
  const index = buildInvertedIndex(abstracts);
  const ALL_NAMES = [...index.keys()].sort((a, b) => a.localeCompare(b));
  const idToIdx = new Map(abstracts.map((a, i) => [a.id, i]));
  const YEARS = [...new Set(abstracts.map(a => a.year))].sort((a, b) => a - b);
  const VIS = new Uint8Array(N).fill(1); // 1 if the abstract's year is toggled on; everything below reads this
  const alts = new Map(Object.entries(synonyms).map(([label, s]) => [label.toLowerCase(), s.toLowerCase().split("; ")]));
  for (const a of abstracts) {  // every spelling grouped under a keyword, so free text finds an abstract by any of them
    a.kwAll = a.keywords.flatMap(k => [k.toLowerCase(), ...(alts.get(k.toLowerCase()) || [])]);
  }

  const X = new Float64Array(N), Y = new Float64Array(N);   // world coordinates
  const SX = new Float32Array(N), SY = new Float32Array(N); // screen coordinates (per frame)
  for (let i = 0; i < N; i++) { X[i] = abstracts[i].x; Y[i] = abstracts[i].y; }

  // ---------------------------------------------------------------------------
  // Constants, state, canvas
  // ---------------------------------------------------------------------------
  const TAU = Math.PI * 2;
  const BG = "#f0efec"; // the site's --surface-2, like the Find talks stage
  const BASE_FILL = "rgba(55, 62, 74, 0.30)";
  const DIM_FILL = "rgba(55, 62, 74, 0.09)";
  const HIT_FILL = "#111827";
  const AUTHOR_SEARCH_SHOW = { marker: true, trail: true, firsts: false };
  const BASE_R = 1.6, HIT_R = 3.6, LINK_R = 2.6, PAINT_R = 3.4;
  const PALETTE = d3.schemeTableau10.filter(c => !["#bab0ab", "#edc949", "#ff9da7"].includes(c)); // too faint on a light page
  const CLUSTER_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2", "#17becf", "#e31a1c", "#6a3d9a", "#33a02c", "#ff1493", "#00a6d6"];
  const MAX_EDGES = 6000;
  const N_SHADES = 6;
  const KDE_SIGMA = 4;   // Gaussian bandwidth in map units (the map spans about 200)
  const KDE_CELLS = 256; // density grid resolution along the longer side
  const shadeColor = d3.interpolateRgb("#cfd3d9", "#4b5563"); // text match 10% .. 100%

  const stage = $("ls-stage"), canvas = $("ls-canvas"), tip = $("ls-tip");
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, dpr = 1;
  let transform = d3.zoomIdentity;
  let hover = -1;
  let pinned = false; // a clicked dot keeps its tooltip (with the Label button) until the map is panned or clicked elsewhere

  const state = {
    author: null,      // { name, mask, hits, firstHits, cx, cy, count, firstCount, trail, edges, linkPct }
    mode: "none",      // web around the author: "coauthors" | "similar" | "none"
    labels: [],        // { name, cx, cy, count } people marked on the map by name
    layers: [],        // { id, type, term, exact, color, count, mask, style, density, peak, img }
    clusterLayers: [], // locally dense cores of the selected preloaded topic groups
    clustersEnabled: false,
    hoveredClusterId: null,
    focusedClusterId: null,
    hoveredClusterSection: null,
    focusedClusterSection: null,
    hiddenClusterIds: new Set(),
    isolatedClusterId: null,
    years: new Set(YEARS), // years shown; the others are hidden everywhere
    animate: $("ls-animate").checked,
  };
  let nextLayerId = 1;
  let trailAnimation = null;
  const TRAIL_ANIMATION_MS = 1400;

  const BOUNDS = (() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < N; i++) {
      minX = Math.min(minX, X[i]); maxX = Math.max(maxX, X[i]);
      minY = Math.min(minY, Y[i]); maxY = Math.max(maxY, Y[i]);
    }
    return { minX, maxX, minY, maxY };
  })();
  // Fixed world-space grid every layer's density is computed on; drawing just scales the result with the zoom.
  const GRID = (() => {
    const pad = 3 * KDE_SIGMA;
    const x0 = BOUNDS.minX - pad, y0 = BOUNDS.minY - pad;
    const cell = Math.max(BOUNDS.maxX - BOUNDS.minX, BOUNDS.maxY - BOUNDS.minY) / KDE_CELLS;
    const gx = Math.ceil((BOUNDS.maxX + pad - x0) / cell), gy = Math.ceil((BOUNDS.maxY + pad - y0) / cell);
    return { x0, y0, cell, gx, gy, w: gx * cell, h: gy * cell };
  })();

  // Paint suggestions: every keyword group and institution with the number of abstracts that have it
  function vocabOf(pick) {
    const m = new Map();
    for (const a of abstracts) for (const label of pick(a)) {
      const key = label.toLowerCase(), v = m.get(key);
      if (v) v.count++; else m.set(key, { label, count: 1, alts: alts.get(key) || [] });
    }
    return m;
  }
  const keywordVocab = vocabOf(a => a.keywords);
  const instVocab = vocabOf(a => a.institutions);

  // ---------------------------------------------------------------------------
  // Rendering: one canvas, repainted on zoom via requestAnimationFrame
  // ---------------------------------------------------------------------------
  let drawQueued = false;
  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => { drawQueued = false; draw(); });
  }

  function fillNodes(color, r, test) {
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      if (!test(i)) continue;
      const x = SX[i], y = SY[i];
      if (x < -r || x > W + r || y < -r || y > H + r) continue; // cull off-screen dots
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, TAU);
    }
    ctx.fill();
  }

  const shadeOf = pct => Math.min(N_SHADES - 1, Math.max(0, Math.floor((pct - SIM_FLOOR) / (100 - SIM_FLOOR) * N_SHADES)));
  const shadeT = b => b / (N_SHADES - 1);

  function drawWeb(a, k, tx, ty) {
    const e = a.edges;
    for (let b = 0; b < N_SHADES; b++) {
      ctx.strokeStyle = d3.color(shadeColor(shadeT(b))).copy({ opacity: 0.18 + 0.5 * shadeT(b) }).formatRgb();
      ctx.lineWidth = 0.6 + 0.9 * shadeT(b);
      ctx.beginPath();
      for (let j = 0; j < e.length; j += 3) {
        if (shadeOf(e[j + 2]) !== b) continue;
        ctx.moveTo(k * X[e[j]] + tx, k * Y[e[j]] + ty);
        ctx.lineTo(k * X[e[j + 1]] + tx, k * Y[e[j + 1]] + ty);
      }
      ctx.stroke();
    }
    const lp = a.linkPct;
    if (state.mode !== "coauthors" || a.showCoauthorNodes) {
      for (let b = 0; b < N_SHADES; b++) {
        fillNodes(shadeColor(shadeT(b)), LINK_R, i => lp[i] > 0 && shadeOf(lp[i]) === b);
      }
    }
  }

  // The author's path through time: their weighted centroid for each year, joined in year order, labelled by year
  function drawTrail(trail, k, tx, ty, color, progress = 1) {
    const reached = progress * (trail.length - 1);
    const fullSegments = Math.floor(reached);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(k * trail[0].cx + tx, k * trail[0].cy + ty);
    for (let j = 1; j <= fullSegments; j++) {
      ctx.lineTo(k * trail[j].cx + tx, k * trail[j].cy + ty);
    }
    if (fullSegments < trail.length - 1) {
      const from = trail[fullSegments], to = trail[fullSegments + 1];
      const fraction = reached - fullSegments;
      ctx.lineTo(k * (from.cx + fraction * (to.cx - from.cx)) + tx,
        k * (from.cy + fraction * (to.cy - from.cy)) + ty);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (let j = 0; j <= fullSegments; j++) {
      const p = trail[j];
      const x = k * p.cx + tx, y = k * p.cy + ty;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, TAU);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.stroke();
      ctx.lineJoin = "round";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
      ctx.strokeText(String(p.year), x + 6, y - 6);
      ctx.fillStyle = color;
      ctx.fillText(String(p.year), x + 6, y - 6);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
    }
  }

  // Target on each abstract in idx (the abstracts someone leads)
  function drawTargets(idx, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const i of idx) {
      const x = SX[i], y = SY[i];
      if (x < -14 || x > W + 14 || y < -14 || y > H + 14) continue;
      ctx.moveTo(x + 7, y);
      ctx.arc(x, y, 7, 0, TAU);
      ctx.moveTo(x - 13, y); ctx.lineTo(x - 4, y);
      ctx.moveTo(x + 4, y);  ctx.lineTo(x + 13, y);
      ctx.moveTo(x, y - 13); ctx.lineTo(x, y - 4);
      ctx.moveTo(x, y + 4);  ctx.lineTo(x, y + 13);
    }
    ctx.stroke();
  }

  function draw() {
    const k = transform.k, tx = transform.x, ty = transform.y;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < N; i++) {
      SX[i] = k * X[i] + tx;
      SY[i] = k * Y[i] + ty;
    }

    const a = state.author;
    // A selected author's annotations are the ones on their label card, so unchecking one removes it from the map
    const selLabel = a ? state.labels.find(l => l.name === a.name) : null;
    const show = a ? (selLabel?.show ?? AUTHOR_SEARCH_SHOW) : null;
    const annoColor = selLabel?.color ?? HIT_FILL; // the selected author's annotations use their label color
    if (a) {
      const am = a.mask;
      fillNodes(DIM_FILL, BASE_R, i => am[i] === 0 && VIS[i] === 1);
    } else {
      fillNodes(BASE_FILL, BASE_R, i => VIS[i] === 1);
    }

    if (state.clusterLayers.length) drawClusterContours(k, tx, ty);

    for (const layer of state.layers) {
      if (layer.style === "density") {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(layer.img, k * GRID.x0 + tx, k * GRID.y0 + ty, k * GRID.w, k * GRID.h);
      } else {
        const m = layer.mask;
        fillNodes(layer.color, PAINT_R, i => m[i] === 1);
      }
    }

    if (a) {
      if (a.edges.length) drawWeb(a, k, tx, ty);
      if (a.showCoauthorNodes) fillNodes(HIT_FILL, HIT_R, i => a.mask[i] === 1);
      if (show.trail && a.trail.length > 1) {
        const progress = trailAnimation?.name === a.name
          ? Math.min(1, (performance.now() - trailAnimation.start) / TRAIL_ANIMATION_MS)
          : 1;
        drawTrail(a.trail, k, tx, ty, annoColor, progress);
        if (progress >= 1) trailAnimation = null;
      }
      if (show.firsts) drawTargets(a.firstHits, annoColor);
    }

    if (state.labels.length) {
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.lineJoin = "round";
      for (const l of state.labels) {
        const x = k * l.cx + tx, y = k * l.cy + ty;
        // The selected author's own annotations are drawn above in HIT_FILL, not twice
        const selected = a && a.name === l.name;
        if (!selected && l.show.trail && l.trail.length > 1) drawTrail(l.trail, k, tx, ty, l.color);
        if (!selected && l.show.firsts && l.firstIdx.length) drawTargets(l.firstIdx, l.color);
        if (!l.show.marker) continue;
        if (x < -200 || x > W + 200 || y < -20 || y > H + 20) continue;
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.textBaseline = "middle";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(x - 4, y);
        ctx.lineTo(x + 4, y);
        ctx.moveTo(x, y - 4);
        ctx.lineTo(x, y + 4);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.strokeStyle = l.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        ctx.strokeText(l.name, x + 8, y);
        ctx.fillStyle = HIT_FILL;
        ctx.fillText(l.name, x + 8, y);
      }
    }

    if (hover >= 0) {
      ctx.strokeStyle = HIT_FILL;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(SX[hover], SY[hover], 6, 0, TAU);
      ctx.stroke();
    }
    if (trailAnimation) requestDraw();
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const first = W === 0;
    W = stage.clientWidth;
    H = stage.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    if (first) fitAll(); else requestDraw();
  }

  // ---------------------------------------------------------------------------
  // d3-zoom: pan and zoom drive the transform; the canvas repaints on zoom
  // ---------------------------------------------------------------------------
  const zoom = d3.zoom()
    .scaleExtent([0.4, 80])
    .on("zoom", (event) => {
      transform = event.transform;
      if (event.sourceEvent) hideTip();
      requestDraw();
    });
  const canvasSel = d3.select(canvas).call(zoom).on("dblclick.zoom", null);

  function fitAll() {
    const { minX, maxX, minY, maxY } = BOUNDS;
    const k = 0.92 * Math.min(W / (maxX - minX), H / (maxY - minY));
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    canvasSel.call(zoom.transform, d3.zoomIdentity.translate(W / 2 - k * cx, H / 2 - k * cy).scale(k));
  }

  function flyTo(cx, cy, radius) {
    // Zoom so the spread fills roughly a third of the smaller side of the stage.
    const k = Math.min(40, Math.max(1.5, 0.35 * Math.min(W, H) / Math.max(radius, 3)));
    const target = d3.zoomIdentity.translate(W / 2 - k * cx, H / 2 - k * cy).scale(k);
    canvasSel.transition().duration(1100).ease(d3.easeCubicInOut).call(zoom.transform, target);
  }

  // ---------------------------------------------------------------------------
  // Autocomplete (shared by the author box and the paint box)
  // ---------------------------------------------------------------------------
  function makeAutocomplete({ input, list, search, onSelect, onEnter, autoFirst }) {
    let items = [], opts = [], active = -1;
    const hide = () => {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      items = []; opts = []; active = -1;
    };
    const mark = () => opts.forEach((li, j) => li.setAttribute("aria-selected", String(j === active)));

    function render() {
      const q = input.value.trim();
      if (!q) { hide(); return; }
      items = search(q);
      active = autoFirst && items.length ? 0 : -1;
      const kids = [];
      opts = [];
      let group = null;
      items.forEach(item => {
        if (item.group && item.group !== group) {
          group = item.group;
          const h = document.createElement("li");
          h.className = "group";
          h.setAttribute("role", "presentation");
          h.textContent = group;
          kids.push(h);
        }
        const li = document.createElement("li");
        li.setAttribute("role", "option");
        const a = document.createElement("span"), b = document.createElement("span");
        a.textContent = item.label;
        if (item.hint) {
          const h = document.createElement("small");
          h.textContent = ` also: ${item.hint}`;
          a.append(h);
        }
        b.textContent = item.meta;
        li.append(a, b);
        li.addEventListener("mousedown", e => { e.preventDefault(); hide(); onSelect(item); });
        opts.push(li);
        kids.push(li);
      });
      mark();
      list.replaceChildren(...kids);
      list.hidden = items.length === 0;
      input.setAttribute("aria-expanded", String(!list.hidden));
    }

    input.addEventListener("input", render);
    input.addEventListener("keydown", e => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!items.length) return;
        const down = e.key === "ArrowDown", n = items.length;
        active = active < 0 ? (down ? 0 : n - 1) : (active + (down ? 1 : -1) + n) % n;
        mark();
        opts[active].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") {
        e.preventDefault();
        const item = items[active];
        hide();
        if (item) onSelect(item); else onEnter(input.value);
      } else if (e.key === "Escape") hide();
    });
    input.addEventListener("blur", hide);
    return { hide };
  }

  // ---------------------------------------------------------------------------
  // Author search: highlight, weighted centroid, web of connections, camera
  // ---------------------------------------------------------------------------
  const INFO_DEFAULT = "Search an author to highlight their abstracts and trace their path through the years.";

  // Edges are flat triples [from, to, match %] over node indices; linkPct is the best match reaching each node.
  function computeConnections(a) {
    const edges = [];
    const linkPct = new Uint8Array(N);
    if (state.mode === "coauthors") {
      const sourcesByCoauthor = new Map();
      for (const s of a.hits) {
        for (const au of abstracts[s].authors) {
          if (au.name === a.name) continue;
          let sources = sourcesByCoauthor.get(au.name);
          if (!sources) sourcesByCoauthor.set(au.name, sources = new Set());
          sources.add(s);
        }
      }
      const seenTargets = new Set();
      for (const [name, sources] of sourcesByCoauthor) {
        let target = -1, targetDistance = Infinity;
        for (const { abstract_id } of index.get(name)) {
          const t = idToIdx.get(abstract_id);
          if (a.mask[t] || !VIS[t]) continue;
          const distance = (X[t] - a.cx) ** 2 + (Y[t] - a.cy) ** 2;
          if (distance < targetDistance) { target = t; targetDistance = distance; }
        }
        if (target < 0 || seenTargets.has(target)) continue;
        let source = -1, sourceDistance = Infinity;
        for (const s of sources) {
          const distance = (X[s] - X[target]) ** 2 + (Y[s] - Y[target]) ** 2;
          if (distance < sourceDistance) { source = s; sourceDistance = distance; }
        }
        seenTargets.add(target);
        edges.push(source, target, 55);
        linkPct[target] = 55;
        if (edges.length >= 3 * MAX_EDGES) break;
      }
    } else if (state.mode === "similar") {
      const best = new Map(); // undirected pair -> edge offset, so a mutual match is drawn once at its higher score
      for (const s of a.hits) {
        for (const [id, pct] of abstracts[s].similar || []) {
          if (pct < SIM_FLOOR) continue;
          const t = idToIdx.get(id);
          if (!VIS[t]) continue;
          const key = Math.min(s, t) * N + Math.max(s, t);
          if (best.has(key)) { edges[best.get(key) + 2] = Math.max(edges[best.get(key) + 2], pct); continue; }
          best.set(key, edges.length);
          edges.push(s, t, pct);
        }
      }
      for (let j = 0; j < edges.length; j += 3) {
        const t = edges[j + 1];
        linkPct[t] = Math.max(linkPct[t], edges[j + 2]);
      }
    }
    a.edges = Int32Array.from(edges);
    a.linkPct = linkPct;
    $("ls-legend").hidden = state.mode !== "similar";
  }

  // The author's abstracts in the years shown, as index entries
  function visibleEntries(name) {
    return index.get(name).filter(({ abstract_id }) => VIS[idToIdx.get(abstract_id)] === 1);
  }

  // Centroid of the author's visible abstracts (by author-position weight) together with the abstracts linked in the
  // current mode, so it moves when the web is switched between co-authors and similar text.
  function updateCentroid(a) {
    let sw = 0, sx = 0, sy = 0;
    for (const { abstract_id, weight } of visibleEntries(a.name)) {
      const i = idToIdx.get(abstract_id);
      sw += weight; sx += weight * X[i]; sy += weight * Y[i];
    }
    for (let i = 0; i < N; i++) {
      if (!a.linkPct[i] || a.mask[i]) continue;
      sw += WEIGHTS.middle; sx += WEIGHTS.middle * X[i]; sy += WEIGHTS.middle * Y[i];
    }
    a.cx = sx / sw; a.cy = sy / sw;
    const l = state.labels.find(l => l.name === a.name);
    if (l) Object.assign(l, labelPos(a.name));
  }

  function authorRadius(a) {
    let r = 0;
    for (const i of a.hits) r = Math.max(r, Math.hypot(X[i] - a.cx, Y[i] - a.cy));
    for (let i = 0; i < N; i++) if (a.linkPct[i]) r = Math.max(r, Math.hypot(X[i] - a.cx, Y[i] - a.cy));
    return Math.min(r, 70); // a consortium's web can span the map; keep the zoom usable
  }

  // Per-year centroids of a person's visible abstracts in year order: the trajectory drawn on the map
  // One point per year: the first-author abstracts that year if there are any, otherwise all of their abstracts
  function trailOf(name) {
    const byYear = new Map();
    for (const { abstract_id, weight } of visibleEntries(name)) {
      const i = idToIdx.get(abstract_id), y = abstracts[i].year;
      const p = byYear.get(y) || { year: y, first: { sw: 0, sx: 0, sy: 0 }, all: { sw: 0, sx: 0, sy: 0 } };
      const groups = weight === WEIGHTS.first ? [p.first, p.all] : [p.all];
      for (const g of groups) { g.sw += weight; g.sx += weight * X[i]; g.sy += weight * Y[i]; }
      byYear.set(y, p);
    }
    return [...byYear.values()].sort((u, v) => u.year - v.year).map(p => {
      const g = p.first.sw > 0 ? p.first : p.all;
      return { year: p.year, cx: g.sx / g.sw, cy: g.sy / g.sw };
    });
  }

  function computeTrail(a) {
    a.trail = trailOf(a.name);
  }

  // The selected author's state over the years shown, or null if none of their abstracts are shown
  function buildAuthor(name) {
    const entries = visibleEntries(name);
    if (!entries.length) return null;
    const mask = new Uint8Array(N);
    const hits = [];
    let sw = 0, sx = 0, sy = 0;
    const firstHits = [];
    for (const { abstract_id, weight } of entries) {
      const i = idToIdx.get(abstract_id);
      mask[i] = 1;
      hits.push(i);
      sw += weight; sx += weight * X[i]; sy += weight * Y[i];
      if (weight === WEIGHTS.first) firstHits.push(i);
    }
    const a = {
      name, mask, hits, firstHits, cx: sx / sw, cy: sy / sw, count: entries.length, firstCount: firstHits.length,
      showCoauthorNodes: state.author?.name === name ? state.author.showCoauthorNodes : false,
    };
    computeTrail(a);
    computeConnections(a);
    return a;
  }

  function authorInfo(a) {
    const years = a.trail.length > 1 ? `, ${a.trail.length} years (${a.trail.map(p => p.year).join(", ")})` : "";
    $("ls-info").textContent = `${a.count} abstract${a.count === 1 ? "" : "s"}, ${a.firstCount} as first author${years}.`;
  }

  function selectAuthor(name) {
    if (!index.has(name)) return;
    if (state.author?.name !== name) state.mode = "none";
    const a = buildAuthor(name);
    if (!a) { $("ls-info").textContent = `${name} has no abstracts in the years shown.`; return; }
    state.author = a;
    startTrailAnimation();
    addLabel(name);
    state.labels.find(l => l.name === name).show = { ...AUTHOR_SEARCH_SHOW };
    renderLabels(); // the card shows them as on
    updateCentroid(a);
    $("ls-author").value = name;
    authorInfo(a);
    $("ls-author-clear").hidden = false;
    flyTo(a.cx, a.cy, authorRadius(a));
    requestDraw();
  }

  // Recomputes the selected author after the years shown change, without moving the view
  function refreshAuthor() {
    if (!state.author) return;
    const a = buildAuthor(state.author.name);
    if (!a) { clearAuthor(); return; }
    state.author = a;
    startTrailAnimation();
    updateCentroid(a);
    authorInfo(a);
  }

  // Years toggled: hide or show their abstracts everywhere (dots, webs, centroids, labels, paint layers)
  function applyYears() {
    for (let i = 0; i < N; i++) VIS[i] = state.years.has(abstracts[i].year) ? 1 : 0;
    state.labels = state.labels.filter(l => visibleEntries(l.name).length);
    refreshAuthor();
    for (const l of state.labels) Object.assign(l, labelPos(l.name));
    updatePresetClusters();
    for (const layer of state.layers) {
      const { mask, count } = matchAbstracts(layer.type, layer.term, layer.exact);
      layer.mask = mask;
      layer.count = count;
      buildDensity(layer);
    }
    hideTip();
    renderLabels();
    renderLayers();
    updateCount();
    requestDraw();
  }

  function updateCount() {
    const shown = VIS.reduce((s, v) => s + v, 0);
    $("ls-count").textContent = shown === N
      ? `${N.toLocaleString()} abstracts, ${index.size.toLocaleString()} unique authors`
      : `${shown.toLocaleString()} of ${N.toLocaleString()} abstracts shown, ${index.size.toLocaleString()} unique authors`;
  }

  function clearAuthor() {
    state.author = null;
    $("ls-author").value = "";
    $("ls-info").textContent = INFO_DEFAULT;
    $("ls-author-clear").hidden = true;
    renderLabels();
    $("ls-legend").hidden = true;
    requestDraw();
  }

  // Matching ignores case, apostrophes (straight or curly), hyphens and spaces: "odonnellluria" finds "Anne O'Donnell-Luria".
  // "Jane M. Smith" also answers to "Jane Smith": the first + last form drops any middle names or initials.
  const norm = s => s.toLowerCase().replace(/['\u2019\u2018-]/g, "");
  const key = s => norm(s).replace(/\s+/g, "");
  const firstLast = n => { const p = n.split(/\s+/); return p.length > 2 ? `${p[0]} ${p[p.length - 1]}` : n; };

  makeAutocomplete({
    input: $("ls-author"), list: $("ls-author-list"), autoFirst: true,
    search(q) {
      const l = norm(q).trim(), k = key(q), tiers = [[], [], [], []]; // exact name, exact last name, prefix, substring
      for (const name of ALL_NAMES) {
        const n = norm(name), fl = firstLast(n);
        const last = n.split(/\s+/).pop();
        if (key(name) === k || key(fl) === k) tiers[0].push(name);
        else if (last === l) tiers[1].push(name);
        else if (key(name).startsWith(k) || key(fl).startsWith(k)) tiers[2].push(name);
        else if (key(name).includes(k) || key(fl).includes(k)) tiers[3].push(name);
      }
      const byCount = (a, b) => index.get(b).length - index.get(a).length;
      tiers[1].sort(byCount);
      return tiers.flat().slice(0, 25).map(name => ({ label: name, meta: index.get(name).length, value: name }));
    },
    onSelect: item => selectAuthor(item.value),
    onEnter: text => {
      const t = key(text);
      const name = ALL_NAMES.find(n => key(n) === t) ?? ALL_NAMES.find(n => key(firstLast(n)) === t);
      if (name) selectAuthor(name);
    },
  });
  $("ls-author").addEventListener("input", () => { if (!$("ls-author").value.trim() && state.author) clearAuthor(); });
  $("ls-author-clear").addEventListener("click", clearAuthor);

  $("ls-labels").addEventListener("click", e => {
    const btn = e.target.closest("#ls-web button");
    if (!btn) return;
    state.mode = btn.dataset.value;
    for (const b of $("ls-web").children) b.setAttribute("aria-pressed", String(b === btn));
    if (state.author) {
      computeConnections(state.author);
      updateCentroid(state.author);
      flyTo(state.author.cx, state.author.cy, authorRadius(state.author));
      requestDraw();
    }
  });
  $("ls-legend-bar").style.background =
    `linear-gradient(to right, ${d3.range(N_SHADES).map(b => shadeColor(shadeT(b))).join(",")})`;

  // ---------------------------------------------------------------------------
  // Labels: name any number of people to mark on their first-author abstract (or the centroid of their abstracts)
  // ---------------------------------------------------------------------------

  // First-author abstracts if the person has any (the label sits on that node); otherwise the weighted centroid of all
  // their abstracts. The selected author's centroid also follows the web mode, see updateCentroid.
  // Also returns what the label's annotations draw: the per-year trail and the indices of the abstracts they lead.
  function labelPos(name) {
    const entries = visibleEntries(name);
    const firsts = entries.filter(e => e.weight === WEIGHTS.first);
    const extras = { trail: trailOf(name), firstIdx: firsts.map(e => idToIdx.get(e.abstract_id)) };
    const au = state.author;
    if (!firsts.length && au && au.name === name) return { cx: au.cx, cy: au.cy, count: entries.length, ...extras };
    const use = firsts.length ? firsts : entries;
    let sw = 0, sx = 0, sy = 0;
    for (const { abstract_id, weight } of use) {
      const i = idToIdx.get(abstract_id);
      sw += weight; sx += weight * X[i]; sy += weight * Y[i];
    }
    return { cx: sx / sw, cy: sy / sw, count: entries.length, ...extras };
  }

  // Annotations a label can show, with the card's checkbox text. The marker is on by default; the rest are opt-in.
  const ANNOTATIONS = [
    { key: "marker", text: "Label marker" },
    { key: "trail", text: "Path across years" },
    { key: "firsts", text: "First-author abstracts" },
  ];

  // A palette color not yet used by a label or a paint layer, so each new label stands apart
  function pickLabelColor() {
    const used = new Set([...state.labels.map(l => l.color), ...state.layers.map(l => l.color)]);
    const free = PALETTE.filter(c => !used.has(c));
    const pool = free.length ? free : PALETTE;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function addLabel(name) {
    if (!index.has(name) || !visibleEntries(name).length || state.labels.some(l => l.name === name)) return;
    state.labels.push({ name, color: pickLabelColor(), show: { marker: true, trail: false, firsts: false }, ...labelPos(name) });
    renderLabels();
    requestDraw();
  }

  function removeLabel(name) {
    state.labels = state.labels.filter(l => l.name !== name);
    renderLabels();
    requestDraw();
  }

  function clearLabels() {
    state.labels = [];
    renderLabels();
    requestDraw();
  }

  function renderLabels() {
    $("ls-label-clear").hidden = state.labels.length === 0;
    $("ls-labels").replaceChildren(...state.labels.map(l => {
      const li = document.createElement("li");
      li.style.setProperty("--author-color", l.color);

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = l.name;
      name.title = `${l.name}: ${l.count} abstract${l.count === 1 ? "" : "s"}`;

      const color = document.createElement("input");
      color.type = "color";
      color.value = l.color;
      color.setAttribute("aria-label", `Annotation color for ${l.name}`);
      color.title = "Annotation color";
      color.addEventListener("input", () => {
        l.color = color.value;
        li.style.setProperty("--author-color", l.color);
        requestDraw();
      });

      const n = document.createElement("span");
      n.className = "n";
      n.textContent = l.count;

      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "x";
      rm.textContent = "×";
      rm.setAttribute("aria-label", `Remove label for ${l.name}`);
      rm.addEventListener("click", () => removeLabel(l.name));

      // Annotations menu: opens as a popover over the map, so the card's row does not change; options that do not apply here are disabled
      const anno = document.createElement("details");
      anno.className = "ls-anno";
      const sum = document.createElement("summary");
      sum.setAttribute("aria-label", `Options for ${l.name}`);
      const panel = document.createElement("div");
      panel.className = "ls-anno-panel";
      const available = {
        marker: [true, ""],
        trail: [l.trail.length > 1, l.trail.length > 1 ? "" : " (one year shown)"],
        firsts: [l.firstIdx.length > 0, l.firstIdx.length ? ` (${l.firstIdx.length})` : " (none)"],
      };
      for (const { key, text } of ANNOTATIONS) {
        const [enabled, note] = available[key];
        const lab = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = l.show[key] && enabled;
        cb.disabled = !enabled;
        cb.addEventListener("change", () => { l.show[key] = cb.checked; requestDraw(); });
        lab.append(cb, `${text}${note}`);
        panel.append(lab);
      }
      anno.append(sum, panel);

      if (state.author?.name === l.name) {
        const web = document.createElement("div");
        web.className = "ls-card-web";
        const heading = document.createElement("span");
        heading.className = "ls-card-web-label";
        heading.textContent = "Web";
        const controls = document.createElement("span");
        controls.id = "ls-web";
        controls.className = "seg";
        controls.setAttribute("role", "group");
        controls.setAttribute("aria-label", `Web around ${l.name}`);
        for (const [value, text] of [["coauthors", "Co-authors"], ["similar", "Similar text"], ["none", "Off"]]) {
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.value = value;
          button.setAttribute("aria-pressed", String(state.mode === value));
          button.textContent = text;
          controls.append(button);
        }
        web.append(heading, controls);
        panel.append(web);

        const coauthorNodes = document.createElement("label");
        const coauthorNodesToggle = document.createElement("input");
        coauthorNodesToggle.type = "checkbox";
        coauthorNodesToggle.checked = state.author.showCoauthorNodes;
        coauthorNodesToggle.setAttribute("aria-label", "Show co-author abstract nodes");
        coauthorNodesToggle.addEventListener("change", () => {
          state.author.showCoauthorNodes = coauthorNodesToggle.checked;
          requestDraw();
        });
        coauthorNodes.append(coauthorNodesToggle, "Show co-author abstracts");
        panel.append(coauthorNodes);
      }

      li.append(anno, color, name, n, rm);
      return li;
    }));
  }

  $("ls-label-clear").addEventListener("click", clearLabels);

  // Annotation menus close on an outside click or Escape, like a menu
  const closeAnnoMenus = except => document.querySelectorAll(".ls-anno[open]").forEach(d => { if (d !== except) d.open = false; });
  document.addEventListener("click", e => closeAnnoMenus(e.target.closest(".ls-anno")));
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeAnnoMenus(null); });

  // ---------------------------------------------------------------------------
  // Hover tooltip
  // ---------------------------------------------------------------------------
  function hideTip() {
    pinned = false;
    tip.classList.remove("pinned");
    tip.hidden = true;
    if (hover >= 0) { hover = -1; requestDraw(); }
  }

  function fillTip(i) {
    const a = abstracts[i];
    const line = (text, cls) => {
      const d = document.createElement("div");
      if (cls) d.className = cls;
      d.textContent = text;
      return d;
    };
    const who = line(a.authors[0] ? a.authors[0].name : "No named speaker", "who");
    const meta = [a.type, a.year].filter(Boolean).join(" · ");
    if (meta) {
      const t = document.createElement("span");
      t.textContent = ` · ${meta}`;
      who.append(t);
    }
    const kids = [who];
    if (a.institutions.length) {
      const shown = a.institutions.slice(0, 3).join("; ");
      kids.push(line(a.institutions.length > 3 ? `${shown} +${a.institutions.length - 3}` : shown));
    }
    if (a.keywords.length) kids.push(line(a.keywords.slice(0, 6).join(" · "), "dim"));
    if (a.authors.length > 1) kids.push(line(`${a.authors.length} authors`, "dim"));
    const au = state.author;
    if (au && state.mode === "similar" && au.linkPct[i] && !au.mask[i]) {
      kids.push(line(`${au.linkPct[i]}% text match to ${au.name}'s abstract`));
    }
    if (pinned && a.authors[0]) {
      const btn = document.createElement("button");
      btn.type = "button";
      const name = a.authors[0].name;
      const done = state.author && state.author.name === name;
      btn.textContent = done ? `${name} is highlighted` : `Highlight and label ${name}`;
      btn.disabled = done;
      btn.addEventListener("click", () => { selectAuthor(name); hideTip(); });
      kids.push(btn);
    }
    tip.replaceChildren(...kids);
  }

  function nearestDot(mx, my) {
    let best = -1, bd = 100; // within 10 px
    for (let i = 0; i < N; i++) {
      if (!VIS[i]) continue;
      const dx = SX[i] - mx, dy = SY[i] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  function placeTip(mx, my) {
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${mx + 14 + w > W - 8 ? Math.max(8, mx - w - 14) : mx + 14}px`;
    tip.style.top = `${my + 14 + h > H - 8 ? Math.max(8, my - h - 14) : my + 14}px`;
  }

  canvas.addEventListener("mousemove", e => {
    if (pinned && !e.buttons) return;
    if (e.buttons) { hideTip(); return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const best = nearestDot(mx, my);
    if (best !== hover) { hover = best; requestDraw(); if (best >= 0) fillTip(best); }
    if (best < 0) { tip.hidden = true; return; }
    placeTip(mx, my);
  });
  canvas.addEventListener("click", e => {
    const wasPinned = pinned;
    if (wasPinned) hideTip();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const best = nearestDot(mx, my);
    if (best < 0) return;
    hover = best;
    pinned = true;
    tip.classList.add("pinned");
    fillTip(best);
    placeTip(mx, my);
    requestDraw();
  });
  canvas.addEventListener("mouseleave", () => { if (!pinned) hideTip(); });

  // ---------------------------------------------------------------------------
  // Paint layers: keywords and institutions, stacked, recolorable; density or dots
  // ---------------------------------------------------------------------------
  const paintMsg = $("ls-paint-msg");
  const say = text => { paintMsg.textContent = text; };
  const TYPE_LABEL = { keyword: "Keyword", institution: "Institution", text: "Contains (keyword, spelling, or institution)" };

  // exact: a suggestion that was picked matches that keyword group or institution only; free text matches substrings
  // in the keywords, their grouped spellings and the institutions.
  function matchAbstracts(type, term, exact) {
    const q = term.toLowerCase();
    const eq = exact ? s => s.toLowerCase() === q : s => s.toLowerCase().includes(q);
    const inKeywords = a => exact ? a.keywords.some(eq) : a.kwAll.some(eq);
    const inInst = a => a.institutions.some(eq);
    const test = type === "institution" ? inInst : type === "keyword" ? inKeywords : a => inKeywords(a) || inInst(a);
    const mask = new Uint8Array(N);
    let count = 0;
    for (let i = 0; i < N; i++) if (VIS[i] && test(abstracts[i])) { mask[i] = 1; count++; }
    return { mask, count };
  }

  function pickColor() {
    const used = new Set(state.layers.map(l => l.color));
    const free = PALETTE.filter(c => !used.has(c));
    const pool = free.length ? free : PALETTE;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  // Kernel density estimate of a layer's matching dots (Gaussian, fixed bandwidth), scaled by the layer's own peak.
  function buildDensity(layer) {
    const { x0, y0, cell, gx, gy } = GRID;
    const d = new Float32Array(gx * gy);
    const R = Math.ceil(3 * KDE_SIGMA / cell), inv = 1 / (2 * KDE_SIGMA * KDE_SIGMA);
    for (let i = 0; i < N; i++) {
      if (!layer.mask[i]) continue;
      const ix = Math.round((X[i] - x0) / cell - 0.5), iy = Math.round((Y[i] - y0) / cell - 0.5);
      for (let yy = Math.max(0, iy - R); yy <= Math.min(gy - 1, iy + R); yy++) {
        const dy = y0 + (yy + 0.5) * cell - Y[i];
        for (let xx = Math.max(0, ix - R); xx <= Math.min(gx - 1, ix + R); xx++) {
          const dx = x0 + (xx + 0.5) * cell - X[i];
          d[yy * gx + xx] += Math.exp(-(dx * dx + dy * dy) * inv);
        }
      }
    }
    layer.density = d;
    layer.peak = d.reduce((m, v) => Math.max(m, v), 0) || 1;
    layer.img = document.createElement("canvas");
    layer.img.width = gx;
    layer.img.height = gy;
    tintDensity(layer);
  }

  // The layer's color at an opacity that rises smoothly with density (faint tails, no hard edge)
  function tintDensity(layer) {
    const { gx, gy } = GRID;
    const c = d3.color(layer.color);
    const g = layer.img.getContext("2d");
    const img = g.createImageData(gx, gy);
    for (let j = 0; j < layer.density.length; j++) {
      const t = layer.density[j] / layer.peak;
      if (t < 0.005) continue;
      img.data[4 * j] = c.r; img.data[4 * j + 1] = c.g; img.data[4 * j + 2] = c.b;
      img.data[4 * j + 3] = Math.round(255 * 0.85 * (1 - Math.exp(-3.2 * t)));
    }
    g.putImageData(img, 0, 0);
  }

  function addLayer(type, term, exact) {
    term = term.trim();
    if (!term) return;
    const dup = state.layers.some(l => l.type === type && l.exact === exact && l.term.toLowerCase() === term.toLowerCase());
    if (dup) { say(`"${term}" is already painted.`); return; }
    const { mask, count } = matchAbstracts(type, term, exact);
    if (count === 0) { say(`No abstracts match "${term}".`); return; }
    const layer = { id: nextLayerId++, type, term, exact, color: pickColor(), count, mask, style: "density" };
    buildDensity(layer);
    state.layers.push(layer);
    say("");
    renderLayers();
    requestDraw();
  }

  function removeLayer(id) {
    state.layers = state.layers.filter(l => l.id !== id);
    renderLayers();
    requestDraw();
  }

  function clearAllPaint() {
    state.layers = [];
    say("");
    renderLayers();
    requestDraw();
  }

  const CLUSTER_RADIUS = 8;
  const CLUSTER_NEIGHBORS = 5;
  const CLUSTER_MIN_CORE = 30;
  const CLUSTER_CONTOUR_FRACTION = 0.4;
  const CLUSTER_THEMES = [
    {
      label: "Research areas",
      groups: [
        { name: "Genetic testing & precision medicine", terms: ["genetic testing", "variant interpretation", "variant classification", "american college of medical genetics", "variants of uncertain significance", "precision medicine", "genetic counselors", "clinical sequencing", "genetic screening", "carrier screening", "genomic screening", "germline genetic testing"] },
        { name: "Pharmacogenetics", terms: ["pharmacogen", "pharmacogenomics", "pharmacogenetic", "pharmacogenomic", "pharmacogenes"] },
        { name: "Ancient DNA & human evolution", terms: ["ancient", "archaic", "neanderthal", "denisovan", "human evolution"] },
        { name: "GWAS & polygenic traits", terms: ["genome-wide association", "polygenic", "genome-wide significant", "summary statistics", "fine mapping", "genetic correlation", "credible set", "mendelian randomization"] },
        { name: "Molecular & cellular genomics", terms: ["gene expression", "single cell", "differential expression", "regulatory element", "spatial transcriptomic", "transcriptome-wide association", "gene regulatory network", "epigenetic"] },
        { name: "Machine learning & AI", terms: ["machine learning", "deep learning", "artificial intelligence", "language model", "neural network"] },
      ],
    },
    {
      label: "Diseases",
      groups: [
        { name: "Coronary artery disease", terms: ["coronary artery disease"] },
        { name: "Metabolic syndrome", terms: ["obesity", "body mass index", "insulin resistance", "metabolic syndrome", "blood pressure", "systolic blood pressure", "type 2 diabetes", "t2dm"], boundaryTerms: ["t2d"] },
        { name: "Breast cancer", terms: ["breast cancer"] },
        { name: "Alzheimer's disease", terms: ["alzheimer"] },
        { name: "Autism spectrum disorder", terms: ["autism spectrum disorder"] },
        { name: "Schizophrenia", terms: ["schizophrenia", "schizophrenic"] },
        { name: "Alcohol use", terms: ["alcohol use disorder", "problematic alcohol use", "alcohol consumption", "alcohol intake"] },
        { name: "Eye disease", terms: ["glaucoma", "inherited retinal disease", "retinal disease", "retinal dystrophy", "diabetic retinopathy", "macular dystrophy", "macular degeneration"] },
        { name: "Inflammatory bowel disease", terms: ["inflammatory bowel disease", "ulcerative colitis", "crohn disease", "crohn's disease"] },
        { name: "Rare disease", terms: ["rare disease", "genetic testing", "diagnostic yield", "human phenotype ontology", "online mendelian inheritance in man"] },
      ],
    },
  ];

  function denseCore(mask) {
    const cell = CLUSTER_RADIUS;
    const buckets = new Map();
    const keyOf = (x, y) => `${Math.floor(x / cell)},${Math.floor(y / cell)}`;
    for (let i = 0; i < N; i++) {
      if (!mask[i]) continue;
      const key = keyOf(X[i], Y[i]);
      let bucket = buckets.get(key);
      if (!bucket) buckets.set(key, bucket = []);
      bucket.push(i);
    }
    const core = new Uint8Array(N);
    let count = 0;
    for (let i = 0; i < N; i++) {
      if (!mask[i]) continue;
      const gx = Math.floor(X[i] / cell), gy = Math.floor(Y[i] / cell);
      let neighbors = 0;
      for (let y = gy - 1; y <= gy + 1 && neighbors < CLUSTER_NEIGHBORS; y++) {
        for (let x = gx - 1; x <= gx + 1 && neighbors < CLUSTER_NEIGHBORS; x++) {
          for (const j of buckets.get(`${x},${y}`) || []) {
            if (i !== j && (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 <= cell ** 2) neighbors++;
            if (neighbors >= CLUSTER_NEIGHBORS) break;
          }
        }
      }
      if (neighbors >= CLUSTER_NEIGHBORS) { core[i] = 1; count++; }
    }
    return { mask: core, count };
  }

  function peakComponent(density) {
    const { gx, gy } = GRID;
    let peakIndex = 0;
    for (let i = 1; i < density.length; i++) if (density[i] > density[peakIndex]) peakIndex = i;
    const peak = density[peakIndex];
    const minimum = peak * CLUSTER_CONTOUR_FRACTION;
    if (!peak) return new Uint8Array(density.length);

    const region = new Uint8Array(density.length);
    const queue = new Int32Array(density.length);
    let head = 0, tail = 0;
    queue[tail++] = peakIndex;
    region[peakIndex] = 1;
    while (head < tail) {
      const index = queue[head++];
      const x = index % gx, y = Math.floor(index / gx);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (Math.abs(dx) + Math.abs(dy) !== 1) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= gx || ny < 0 || ny >= gy) continue;
          const next = ny * gx + nx;
          if (!region[next] && density[next] >= minimum) {
            region[next] = 1;
            queue[tail++] = next;
          }
        }
      }
    }
    return region;
  }

  function interiorCenter(component, gx, gy, fallback) {
    const size = component.length;
    const distance = new Uint16Array(size).fill(65535);
    const queue = new Int32Array(size);
    let head = 0, tail = 0, sumX = 0, sumY = 0, count = 0;
    for (let index = 0; index < size; index++) {
      if (!component[index]) continue;
      const x = index % gx, y = Math.floor(index / gx);
      sumX += x;
      sumY += y;
      count++;
      const boundary = x === 0 || x === gx - 1 || y === 0 || y === gy - 1
        || !component[index - 1] || !component[index + 1]
        || !component[index - gx] || !component[index + gx];
      if (boundary) {
        distance[index] = 0;
        queue[tail++] = index;
      }
    }
    if (!count || !tail) return fallback;
    while (head < tail) {
      const index = queue[head++];
      const x = index % gx, y = Math.floor(index / gx);
      for (const next of [x > 0 ? index - 1 : -1, x + 1 < gx ? index + 1 : -1,
        y > 0 ? index - gx : -1, y + 1 < gy ? index + gx : -1]) {
        if (next < 0 || !component[next] || distance[next] <= distance[index] + 1) continue;
        distance[next] = distance[index] + 1;
        queue[tail++] = next;
      }
    }
    const centerX = sumX / count, centerY = sumY / count;
    let best = fallback, bestDistance = -1, bestCentroidDistance = Infinity;
    for (let index = 0; index < size; index++) {
      if (!component[index]) continue;
      const x = index % gx, y = Math.floor(index / gx);
      const centroidDistance = (x - centerX) ** 2 + (y - centerY) ** 2;
      if (distance[index] > bestDistance
        || (distance[index] === bestDistance && centroidDistance < bestCentroidDistance)) {
        best = index;
        bestDistance = distance[index];
        bestCentroidDistance = centroidDistance;
      }
    }
    return best;
  }

  function makeNonOverlappingContours(layers) {
    const { gx, gy } = GRID;
    const candidates = layers.map(layer => peakComponent(layer.density));
    let active = layers.map((_, i) => i);
    const contour = d3.contours().size([gx, gy]).thresholds([0.5]);
    for (let attempt = 0; active.length && attempt <= layers.length; attempt++) {
      const owner = new Int16Array(gx * gy).fill(-1);
      const strength = new Float32Array(gx * gy);
      for (const i of active) {
        const { density, peak } = layers[i];
        for (let j = 0; j < owner.length; j++) {
          if (!candidates[i][j]) continue;
          const relativeDensity = density[j] / peak;
          if (relativeDensity > strength[j]) {
            strength[j] = relativeDensity;
            owner[j] = i;
          }
        }
      }

      for (const i of active) {
        const layer = layers[i];
        let seed = -1, bestDensity = 0;
        for (let j = 0; j < owner.length; j++) {
          if (owner[j] !== i) continue;
          const relativeDensity = layer.density[j] / layer.peak;
          if (relativeDensity > bestDensity) { bestDensity = relativeDensity; seed = j; }
        }
        if (seed < 0) {
          layer.contourPath = null;
          layer.count = 0;
          continue;
        }

        const component = new Uint8Array(owner.length);
        const queue = new Int32Array(owner.length);
        let head = 0, tail = 0;
        queue[tail++] = seed;
        component[seed] = 1;
        while (head < tail) {
          const index = queue[head++];
          const x = index % gx, y = Math.floor(index / gx);
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (Math.abs(dx) + Math.abs(dy) !== 1) continue;
              const nx = x + dx, ny = y + dy;
              if (nx < 0 || nx >= gx || ny < 0 || ny >= gy) continue;
              const next = ny * gx + nx;
              if (!component[next] && owner[next] === i) {
                component[next] = 1;
                queue[tail++] = next;
              }
            }
          }
        }
        const owned = new Float32Array(owner.length);
        for (let j = 0; j < owner.length; j++) if (component[j]) owned[j] = 1;

        const shape = contour(owned).find(feature => feature.coordinates.length);
        if (!shape) {
          layer.contourPath = null;
          layer.count = 0;
          continue;
        }
        const polygon = shape.coordinates.reduce((best, coords) => {
          const ring = coords[0];
          const area = Math.abs(d3.polygonArea(ring));
          return !best || area > best.area ? { ring, area } : best;
        }, null);
        if (!polygon) {
          layer.contourPath = null;
          layer.count = 0;
          continue;
        }
        const geometry = { type: "Polygon", coordinates: [polygon.ring] };
        layer.contourPath = new Path2D(d3.geoPath(null)(geometry));
        const labelIndex = interiorCenter(component, gx, gy, seed);
        layer.labelX = GRID.x0 + ((labelIndex % gx) + 0.5) * GRID.cell;
        layer.labelY = GRID.y0 + (Math.floor(labelIndex / gx) + 0.5) * GRID.cell;
        layer.count = 0;
        for (let j = 0; j < N; j++) {
          if (!layer.mask[j]) continue;
          const x = Math.round((X[j] - GRID.x0) / GRID.cell - 0.5);
          const y = Math.round((Y[j] - GRID.y0) / GRID.cell - 0.5);
          if (x >= 0 && x < gx && y >= 0 && y < gy && component[y * gx + x]) layer.count++;
        }
      }
      const survivors = active.filter(i => layers[i].contourPath && layers[i].count >= CLUSTER_MIN_CORE);
      if (survivors.length === active.length) break;
      for (const i of active) if (!survivors.includes(i)) {
        layers[i].contourPath = null;
        layers[i].count = 0;
      }
      active = survivors;
    }
  }

  function drawClusterContours(k, tx, ty) {
    const hoveredId = state.hoveredClusterId && !state.hiddenClusterIds.has(state.hoveredClusterId) ? state.hoveredClusterId : null;
    const focusedId = state.focusedClusterId && !state.hiddenClusterIds.has(state.focusedClusterId) ? state.focusedClusterId : null;
    const activeClusterId = hoveredId || focusedId;
    const activeClusterSection = state.hoveredClusterSection || state.focusedClusterSection;
    ctx.save();
    ctx.translate(k * GRID.x0 + tx, k * GRID.y0 + ty);
    ctx.scale(k * GRID.cell, k * GRID.cell);
    for (const layer of state.clusterLayers) {
      if (!layer.contourPath || !isClusterVisible(layer)) continue;
      const muted = activeClusterId ? activeClusterId !== layer.id
        : activeClusterSection && activeClusterSection !== layer.section;
      const color = muted ? "#aeb4bd" : layer.color;
      ctx.fillStyle = d3.color(color).copy({ opacity: muted ? 0.15 : 0.24 }).formatRgb();
      ctx.fill(layer.contourPath);
      ctx.strokeStyle = d3.color(color).copy({ opacity: muted ? 0.45 : 0.9 }).formatRgb();
      ctx.lineWidth = 0.8 / (k * GRID.cell);
      ctx.stroke(layer.contourPath);
    }
    ctx.restore();
    drawClusterLabels(k, tx, ty);
  }

  function drawClusterLabels(k, tx, ty) {
    const hoveredId = state.hoveredClusterId && !state.hiddenClusterIds.has(state.hoveredClusterId) ? state.hoveredClusterId : null;
    const focusedId = state.focusedClusterId && !state.hiddenClusterIds.has(state.focusedClusterId) ? state.focusedClusterId : null;
    const activeClusterId = hoveredId || focusedId;
    const activeClusterSection = state.hoveredClusterSection || state.focusedClusterSection;
    const shown = [];
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const layer of [...state.clusterLayers].sort((a, b) => b.count - a.count)) {
      if (!layer.contourPath || !isClusterVisible(layer)) continue;
      const x = k * layer.labelX + tx, y = k * layer.labelY + ty;
      const width = ctx.measureText(layer.term).width, height = 15;
      const box = { left: x - width / 2, right: x + width / 2, top: y - height / 2, bottom: y + height / 2 };
      if (box.right < 0 || box.left > W || box.bottom < 0 || box.top > H) continue;
      if (shown.some(other => box.left < other.right + 6 && box.right + 6 > other.left
        && box.top < other.bottom + 3 && box.bottom + 3 > other.top)) continue;
      shown.push(box);
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.strokeText(layer.term, x, y);
      const muted = activeClusterId ? activeClusterId !== layer.id
        : activeClusterSection && activeClusterSection !== layer.section;
      ctx.fillStyle = muted ? "#aeb4bd" : layer.color;
      ctx.fillText(layer.term, x, y);
    }
    ctx.restore();
  }

  function isClusterVisible(layer) {
    return !state.hiddenClusterIds.has(layer.id)
      && (!state.isolatedClusterId || state.isolatedClusterId === layer.id);
  }

  function syncClusterControls() {
    for (const layer of state.clusterLayers) {
      const li = layer.listItem;
      if (!li) continue;
      const visible = isClusterVisible(layer);
      li.setAttribute("aria-pressed", String(visible));
      li.setAttribute("aria-label", `${visible ? "Hide" : "Show"} ${layer.term} cluster, ${layer.count.toLocaleString()} abstracts`);
      li.classList.toggle("is-hidden", !visible);
      li.classList.toggle("is-isolated", state.isolatedClusterId === layer.id);
    }
    for (const section of CLUSTER_THEMES) {
      const members = state.clusterLayers.filter(layer => layer.section === section.label);
      const visible = members.some(isClusterVisible);
      const heading = document.querySelector(`.ls-cluster-section[data-cluster-section="${section.label}"] .ls-cluster-section-title`);
      if (heading) {
        heading.setAttribute("aria-pressed", String(visible));
        heading.setAttribute("aria-label", `${visible ? "Hide" : "Show"} all ${section.label.toLowerCase()} clusters`);
        heading.classList.toggle("is-hidden", !visible);
      }
    }
  }

  function updatePresetClusters() {
    state.hoveredClusterId = null;
    state.focusedClusterId = null;
    state.hoveredClusterSection = null;
    state.focusedClusterSection = null;
    if (!state.clustersEnabled) {
      state.clusterLayers = [];
      $("ls-cluster-research").replaceChildren();
      $("ls-cluster-diseases").replaceChildren();
      state.hiddenClusterIds.clear();
      state.isolatedClusterId = null;
      requestDraw();
      return;
    }
    const groups = CLUSTER_THEMES.flatMap(section =>
      section.groups.map(group => ({ ...group, section: section.label })));
    const layers = [];
    for (const [groupIndex, group] of groups.entries()) {
      const mask = new Uint8Array(N);
      const terms = group.terms;
      const boundaryTerms = group.boundaryTerms || [];
      for (let i = 0; i < N; i++) {
        if (VIS[i] && abstracts[i].keywords.some(k => {
          const keyword = k.toLowerCase();
          return terms.some(term => keyword.includes(term))
            || boundaryTerms.some(term => new RegExp(`(^|[^a-z0-9])${term}($|[^a-z0-9])`).test(keyword));
        })) mask[i] = 1;
      }
      const support = denseCore(mask);
      if (support.count < CLUSTER_MIN_CORE) continue;
      const layer = {
        id: `cluster-${groupIndex}`,
        section: group.section,
        term: group.name,
        color: group.name === "Rare disease"
          ? "#00a6d6"
          : CLUSTER_PALETTE[(groupIndex + 2) % CLUSTER_PALETTE.length],
        count: support.count,
        mask: support.mask,
      };
      buildDensity(layer);
      layers.push(layer);
    }
    makeNonOverlappingContours(layers);
    state.clusterLayers = layers;
    const visibleLayers = layers.filter(layer => layer.contourPath && layer.count >= CLUSTER_MIN_CORE);
    state.clusterLayers = visibleLayers;
    for (const section of CLUSTER_THEMES) {
      const list = $(section.label === "Research areas" ? "ls-cluster-research" : "ls-cluster-diseases");
      list.replaceChildren(...visibleLayers.filter(layer => layer.section === section.label).map(layer => {
        const li = document.createElement("li");
        const swatch = document.createElement("span");
        swatch.className = "ls-cluster-swatch";
        swatch.style.backgroundColor = layer.color;
        const name = document.createElement("span");
        name.textContent = layer.term;
        const count = document.createElement("span");
        count.className = "n";
        count.textContent = layer.count.toLocaleString();
        li.tabIndex = 0;
        li.setAttribute("role", "button");
        layer.listItem = li;
        const toggle = () => {
          state.isolatedClusterId = null;
          if (state.hiddenClusterIds.has(layer.id)) state.hiddenClusterIds.delete(layer.id);
          else state.hiddenClusterIds.add(layer.id);
          syncClusterControls();
          requestDraw();
        };
        li.addEventListener("click", toggle);
        li.addEventListener("keydown", e => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
          }
        });
        li.addEventListener("contextmenu", e => {
          e.preventDefault();
          state.isolatedClusterId = state.isolatedClusterId === layer.id ? null : layer.id;
          if (state.isolatedClusterId) state.hiddenClusterIds.delete(layer.id);
          syncClusterControls();
          requestDraw();
        });
        li.addEventListener("mouseenter", () => { state.hoveredClusterId = layer.id; requestDraw(); });
        li.addEventListener("mouseleave", () => { state.hoveredClusterId = null; requestDraw(); });
        li.addEventListener("focus", () => { state.focusedClusterId = layer.id; requestDraw(); });
        li.addEventListener("blur", () => { state.focusedClusterId = null; requestDraw(); });
        li.append(swatch, name, count);
        return li;
      }));
    }
    syncClusterControls();
    requestDraw();
  }

  for (const section of document.querySelectorAll(".ls-cluster-section")) {
    const name = section.dataset.clusterSection;
    const heading = section.querySelector(".ls-cluster-section-title");
    heading.addEventListener("mouseenter", () => { state.hoveredClusterSection = name; requestDraw(); });
    heading.addEventListener("mouseleave", () => { state.hoveredClusterSection = null; requestDraw(); });
    heading.addEventListener("focus", () => { state.focusedClusterSection = name; requestDraw(); });
    heading.addEventListener("blur", () => { state.focusedClusterSection = null; requestDraw(); });
    heading.addEventListener("click", () => {
      const members = state.clusterLayers.filter(layer => layer.section === name);
      const hasVisible = members.some(isClusterVisible);
      state.isolatedClusterId = null;
      for (const layer of members) {
        if (hasVisible) state.hiddenClusterIds.add(layer.id);
        else state.hiddenClusterIds.delete(layer.id);
      }
      syncClusterControls();
      requestDraw();
    });
  }
  for (const [selector, id] of [["#ls-cluster-research", "Research areas"], ["#ls-cluster-diseases", "Diseases"]]) {
    const list = document.querySelector(selector);
    list.addEventListener("mouseenter", () => { state.hoveredClusterSection = id; requestDraw(); });
    list.addEventListener("mouseleave", () => { state.hoveredClusterSection = null; requestDraw(); });
  }
  $("ls-clusters-toggle").addEventListener("change", e => {
    state.clustersEnabled = e.target.checked;
    $("ls-cluster-settings").hidden = !state.clustersEnabled;
    updatePresetClusters();
  });

  function renderLayers() {
    $("ls-paint-clear").hidden = state.layers.length === 0;
    $("ls-layers").replaceChildren(...state.layers.map(layer => {
      const li = document.createElement("li");

      const color = document.createElement("input");
      color.type = "color";
      color.value = layer.color;
      color.setAttribute("aria-label", `Color for ${layer.term}`);
      color.addEventListener("input", () => { layer.color = color.value; tintDensity(layer); requestDraw(); });

      const options = document.createElement("details");
      options.className = "ls-anno ls-paint-options";
      const summary = document.createElement("summary");
      summary.setAttribute("aria-label", `Display options for ${layer.term}`);
      summary.title = "Display options";
      const panel = document.createElement("div");
      panel.className = "ls-anno-panel";
      const styleLabel = document.createElement("span");
      styleLabel.className = "ls-card-web-label";
      styleLabel.textContent = "Display as";
      const styles = document.createElement("span");
      styles.className = "seg";
      styles.setAttribute("role", "group");
      styles.setAttribute("aria-label", `Display style for ${layer.term}`);
      for (const [value, text] of [["density", "Density"], ["dots", "Dots"]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.value = value;
        button.setAttribute("aria-pressed", String(layer.style === value));
        button.textContent = text;
        button.addEventListener("click", () => {
          layer.style = value;
          for (const sibling of styles.children) sibling.setAttribute("aria-pressed", String(sibling === button));
          requestDraw();
        });
        styles.append(button);
      }
      panel.append(styleLabel, styles);
      options.append(summary, panel);

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = layer.term;
      name.title = `${TYPE_LABEL[layer.type]}: ${layer.term} (${layer.count} abstracts)`;

      const n = document.createElement("span");
      n.className = "n";
      n.textContent = layer.count;

      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "x";
      rm.textContent = "×";
      rm.setAttribute("aria-label", `Remove ${layer.term}`);
      rm.addEventListener("click", () => removeLayer(layer.id));

      li.append(color, options, name, n, rm);
      return li;
    }));
  }

  const paintBox = makeAutocomplete({
    input: $("ls-paint"), list: $("ls-paint-list"), autoFirst: false,
    search(q) {
      const l = q.toLowerCase();
      const wordStart = new RegExp("(^|[^a-z0-9])" + l.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "([^a-z0-9]|$)");
      const rank = (vocab, type, group) => {
        const hits = [];
        for (const [key, v] of vocab) {
          const p = key.indexOf(l);
          let hint = "", first;
          if (p === 0 || (p > 0 && wordStart.test(key))) first = 0;  // prefix, or a whole word such as the "(prs)" acronym
          else if (p > 0) first = 1;
          else {
            hint = v.alts.find(a => a === l) || v.alts.find(a => a.includes(l)) || "";
            if (!hint) continue;
            first = hint === l ? 0 : 2;                                // an exact synonym outranks substring hits
          }
          hits.push({ label: v.label, meta: v.count, value: v.label, type, group, hint, first, count: v.count });
        }
        return hits.sort((a, b) => a.first - b.first || b.count - a.count).slice(0, 8);
      };
      return rank(keywordVocab, "keyword", "Keywords").concat(rank(instVocab, "institution", "Institutions"));
    },
    onSelect: item => { addLayer(item.type, item.value, true); $("ls-paint").value = ""; },
    onEnter: text => { addLayer("text", text, false); $("ls-paint").value = ""; },
  });

  $("ls-paint-form").addEventListener("submit", e => {
    e.preventDefault();
    paintBox.hide();
    addLayer("text", $("ls-paint").value, false);
    $("ls-paint").value = "";
  });
  $("ls-paint-clear").addEventListener("click", clearAllPaint);

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  const yearOptions = $("ls-year-options");
  yearOptions.append(...YEARS.map(y => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(y);
    input.checked = true;
    label.append(input, ` ${y}`);
    return label;
  }));
  function updateYearControls() {
    const all = state.years.size === YEARS.length;
    $("ls-years-summary").textContent = all ? "All years" : [...state.years].sort((a, b) => a - b).join(", ");
    const selectAll = $("ls-years-all");
    selectAll.checked = all;
    selectAll.indeterminate = !all && state.years.size > 0;
    for (const input of yearOptions.querySelectorAll("input")) {
      const year = Number(input.value);
      input.checked = state.years.has(year);
      input.disabled = input.checked && state.years.size === 1;
    }
  }
  yearOptions.addEventListener("change", e => {
    const input = e.target.closest("input");
    if (!input) return;
    const year = Number(input.value);
    if (!input.checked && state.years.size === 1) {
      input.checked = true;
      return;
    }
    if (input.checked) state.years.add(year); else state.years.delete(year);
    updateYearControls();
    applyYears();
  });
  $("ls-years-all").addEventListener("change", e => {
    state.years = new Set(e.target.checked ? YEARS : [YEARS[YEARS.length - 1]]);
    updateYearControls();
    applyYears();
  });
  function startTrailAnimation() {
    if (!state.animate || !state.author || state.author.trail.length < 2) {
      trailAnimation = null;
      return;
    }
    trailAnimation = { name: state.author.name, start: performance.now() };
    requestDraw();
  }
  $("ls-animate").addEventListener("change", e => {
    state.animate = e.target.checked;
    startTrailAnimation();
    requestDraw();
  });
  updateYearControls();
  updatePresetClusters();
  updateCount();
  new ResizeObserver(resize).observe(stage);
  resize();
})();
