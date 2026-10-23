// Landscape page: every ASHG 2026 abstract as a dot at its t-SNE position (landscape.py), on one canvas.
// Search an author to highlight their abstracts and fly to their weighted centroid, with a web of co-authors or
// similar abstracts; paint keywords and institutions as density or dots. Data has no titles or abstract text.
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
  const BASE_R = 1.6, HIT_R = 3.6, LINK_R = 2.6, PAINT_R = 3.4;
  const PALETTE = d3.schemeTableau10.filter(c => !["#bab0ab", "#edc949", "#ff9da7"].includes(c)); // too faint on a light page
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
    author: null,      // { name, mask, hits, firstHits, cx, cy, count, firstCount, edges, linkPct }
    mode: "coauthors", // web around the author: "coauthors" | "similar" | "none"
    style: "density",  // how paint layers are drawn: "density" | "dots"
    labels: [],        // { name, cx, cy, count } people marked on the map by name
    layers: [],        // { id, type, term, exact, color, count, mask, density, peak, img }
  };
  let nextLayerId = 1;

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
    for (let b = 0; b < N_SHADES; b++) {
      fillNodes(shadeColor(shadeT(b)), LINK_R, i => lp[i] > 0 && shadeOf(lp[i]) === b);
    }
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
    if (a) {
      const am = a.mask;
      fillNodes(DIM_FILL, BASE_R, i => am[i] === 0);
      if (a.edges.length) drawWeb(a, k, tx, ty);
      fillNodes(HIT_FILL, HIT_R, i => am[i] === 1);
    } else {
      fillNodes(BASE_FILL, BASE_R, () => true);
    }

    if (state.style === "density") {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      for (const layer of state.layers) {
        ctx.drawImage(layer.img, k * GRID.x0 + tx, k * GRID.y0 + ty, k * GRID.w, k * GRID.h);
      }
    } else {
      for (const layer of state.layers) {
        const m = layer.mask;
        fillNodes(layer.color, PAINT_R, i => m[i] === 1);
      }
    }

    if (a) {
      // Target on each abstract where they are first author
      ctx.strokeStyle = HIT_FILL;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const i of a.firstHits) {
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

      // Weighted centroid: a hollow diamond
      const cx = k * a.cx + tx, cy = k * a.cy + ty, d = 7;
      ctx.beginPath();
      ctx.moveTo(cx, cy - d); ctx.lineTo(cx + d, cy); ctx.lineTo(cx, cy + d); ctx.lineTo(cx - d, cy);
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.fill();
      ctx.stroke();
    }

    if (state.labels.length) {
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.lineJoin = "round";
      for (const l of state.labels) {
        const x = k * l.cx + tx, y = k * l.cy + ty;
        if (x < -200 || x > W + 200 || y < -20 || y > H + 20) continue;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, TAU);
        ctx.fillStyle = "#b91c1c";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
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
  const INFO_DEFAULT = "Highlights their abstracts and flies to their weighted centroid.";

  // Edges are flat triples [from, to, match %] over node indices; linkPct is the best match reaching each node.
  function computeConnections(a) {
    const edges = [];
    const linkPct = new Uint8Array(N);
    if (state.mode === "coauthors") {
      const seen = new Set();
      outer:
      for (const s of a.hits) {
        for (const au of abstracts[s].authors) {
          if (au.name === a.name) continue;
          for (const { abstract_id } of index.get(au.name)) {
            const t = idToIdx.get(abstract_id);
            if (a.mask[t]) continue;
            const key = s * N + t;
            if (seen.has(key)) continue;
            seen.add(key);
            edges.push(s, t, 55);
            linkPct[t] = 55;
            if (edges.length >= 3 * MAX_EDGES) break outer;
          }
        }
      }
    } else if (state.mode === "similar") {
      const best = new Map(); // undirected pair -> edge offset, so a mutual match is drawn once at its higher score
      for (const s of a.hits) {
        for (const [id, pct] of abstracts[s].similar || []) {
          if (pct < SIM_FLOOR) continue;
          const t = idToIdx.get(id);
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

  // Centroid of the author's abstracts (by author-position weight) together with the abstracts linked in the current
  // mode, so it moves when the web is switched between co-authors and similar text.
  function updateCentroid(a) {
    let sw = 0, sx = 0, sy = 0;
    for (const { abstract_id, weight } of index.get(a.name)) {
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

  function selectAuthor(name) {
    const entries = index.get(name);
    if (!entries) return;
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
    const firstCount = firstHits.length;
    const a = state.author = { name, mask, hits, firstHits, cx: sx / sw, cy: sy / sw, count: entries.length, firstCount };
    computeConnections(a);
    addLabel(name);
    updateCentroid(a);
    $("ls-author").value = name;
    $("ls-info").textContent = `${entries.length} abstract${entries.length === 1 ? "" : "s"}, ${firstCount} as first author.`;
    $("ls-author-clear").hidden = false;
    flyTo(a.cx, a.cy, authorRadius(a));
    requestDraw();
  }

  function clearAuthor() {
    state.author = null;
    $("ls-author").value = "";
    $("ls-info").textContent = INFO_DEFAULT;
    $("ls-author-clear").hidden = true;
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

  $("ls-web").addEventListener("click", e => {
    const btn = e.target.closest("button");
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
  function labelPos(name) {
    const entries = index.get(name);
    const firsts = entries.filter(e => e.weight === WEIGHTS.first);
    const au = state.author;
    if (!firsts.length && au && au.name === name) return { cx: au.cx, cy: au.cy, count: entries.length };
    const use = firsts.length ? firsts : entries;
    let sw = 0, sx = 0, sy = 0;
    for (const { abstract_id, weight } of use) {
      const i = idToIdx.get(abstract_id);
      sw += weight; sx += weight * X[i]; sy += weight * Y[i];
    }
    return { cx: sx / sw, cy: sy / sw, count: entries.length };
  }

  function addLabel(name) {
    if (!index.has(name) || state.labels.some(l => l.name === name)) return;
    state.labels.push({ name, ...labelPos(name) });
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

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = l.name;
      name.title = `${l.name}: ${l.count} abstract${l.count === 1 ? "" : "s"}`;

      const n = document.createElement("span");
      n.className = "n";
      n.textContent = l.count;

      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "x";
      rm.textContent = "×";
      rm.setAttribute("aria-label", `Remove label for ${l.name}`);
      rm.addEventListener("click", () => removeLabel(l.name));

      li.append(name, n, rm);
      return li;
    }));
  }

  $("ls-label-clear").addEventListener("click", clearLabels);

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
    if (a.type) {
      const t = document.createElement("span");
      t.textContent = ` · ${a.type}`;
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
    for (let i = 0; i < N; i++) if (test(abstracts[i])) { mask[i] = 1; count++; }
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
    const layer = { id: nextLayerId++, type, term, exact, color: pickColor(), count, mask };
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

  function renderLayers() {
    $("ls-paint-clear").hidden = state.layers.length === 0;
    $("ls-layers").replaceChildren(...state.layers.map(layer => {
      const li = document.createElement("li");

      const color = document.createElement("input");
      color.type = "color";
      color.value = layer.color;
      color.setAttribute("aria-label", `Color for ${layer.term}`);
      color.addEventListener("input", () => { layer.color = color.value; tintDensity(layer); requestDraw(); });

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

      li.append(color, name, n, rm);
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
  $("ls-style").addEventListener("click", e => {
    const btn = e.target.closest("button");
    if (!btn) return;
    state.style = btn.dataset.value;
    for (const b of $("ls-style").children) b.setAttribute("aria-pressed", String(b === btn));
    requestDraw();
  });

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  $("ls-count").textContent = `${N.toLocaleString()} abstracts, ${index.size.toLocaleString()} unique authors`;
  new ResizeObserver(resize).observe(stage);
  resize();
})();
