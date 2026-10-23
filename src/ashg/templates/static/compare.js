// "Compare": text match, mutual rank and shared authors for two abstracts picked by first author.
// Data: network.json (nodes, names of authors on 2+ abstracts) and network_sim.bin.gz (every pair's text match, 0 = under 10%).
(() => {
  const root = document.getElementById("compare");
  if (!root) return;
  const TYPES = { poster: "Poster", platform: "Platform talk", plenary: "Plenary", symposium: "Symposium talk", session: "Session" };
  const slots = { a: root.querySelector("[data-slot=a]"), b: root.querySelector("[data-slot=b]") };
  const out = root.querySelector("#cmp-out");
  const state = { a: null, b: null };
  let data, nodes, tracks, pairs = null, below = null, pairsLoading = false, vecs = null; // vecs: {terms, rows}, rows[i] = [term, weight, term, weight, ...]
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const author = (n) => n.a || n.n || "Unknown author";
  const pct = (m) => (m ? `${m}%` : "under 10%");
  const toks = (s) => s.toLowerCase().split(/[\s,.]+/).filter(Boolean);

  function matches(q) { // names are stored as "Surname, I."; every typed word must start a name word, or be a first name with that initial
    const words = toks(q), board = /^(?:board\s*)?(\d{2,4})([wtf]?)$/i.exec(q.trim()); // a poster board number, with or without the day letter
    if (board) return nodes.map((n, i) => i).filter((i) => nodes[i].b && nodes[i].b.toLowerCase().startsWith((board[1] + board[2]).toLowerCase())).sort((x, y) => nodes[x].b.localeCompare(nodes[y].b)).slice(0, 8);
    if (q.trim().length < 2 || !words.length) return [];
    const ok = (name) => { const t = toks(name); return words.every((w) => t.some((x) => x.startsWith(w)) || (w.length > 1 && t.some((x) => x.length === 1 && x === w[0]))); };
    const rank = (name) => { const t = toks(name), sur = name.toLowerCase().split(",")[0].trim();
      return t.length === words.length && t.every((x) => words.includes(x)) ? -1 : words.includes(sur) ? 0 : words.some((w) => sur.startsWith(w)) ? 1 : 2; };
    const hits = [];
    for (let i = 0; i < nodes.length; i++) if (nodes[i].a && ok(nodes[i].a)) hits.push(i);
    return hits.sort((x, y) => rank(nodes[x].a) - rank(nodes[y].a)).slice(0, 8);
  }

  function pick(slot, id) {
    state[slot] = id;
    const box = slots[slot], input = box.querySelector("input");
    input.value = ""; box.querySelector(".sim-results").hidden = true;
    renderChosen(slot); show();
    if (id != null) loadPairs();
  }

  function renderChosen(slot) {
    const id = state[slot], box = slots[slot].querySelector(".cmp-chosen");
    box.innerHTML = id == null ? "" : `<b>${esc(author(nodes[id]))}</b> <span>${TYPES[nodes[id].t]} · ${esc(nodes[id].w)}</span> <button type="button" aria-label="Clear">×</button>`;
    if (id != null) box.querySelector("button").addEventListener("click", () => pick(slot, null));
  }

  const gz = (url) => fetch(url).then((r) => new Response(r.body.pipeThrough(new DecompressionStream("gzip"))));
  function loadPairs() {
    if (pairs || pairsLoading || !root.dataset.pairs) return;
    pairsLoading = true;
    gz(root.dataset.pairs).then((r) => r.arrayBuffer()).then((buf) => { pairs = new Uint8Array(buf); below = cumulative(pairs, nodes.length); show(); }).catch(() => { pairsLoading = false; });
    gz(root.dataset.terms).then((r) => r.json()).then((v) => { vecs = v; show(); }).catch(() => {});
  }

  function cumulative(p, n) { // below[v] = share of distinct abstract pairs (upper triangle) scoring strictly under v
    const h = new Array(102).fill(0); let t = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { h[p[i * n + j]]++; t++; }
    const c = new Array(102).fill(0); for (let v = 1; v < 102; v++) c[v] = c[v - 1] + h[v - 1] / t;
    return c;
  }
  function percentile(m) {
    if (!below) return "";
    const lo = 100 * below[m];
    if (!m) return `<p class="sub">Under 10% is the bottom ${(100 * below[1]).toFixed(0)}% of all abstract pairs; scores that low aren't ranked more finely.</p>`;
    const top = 100 - lo;
    return `<p class="sub">Higher than <b>${lo >= 99.95 ? "99.9+" : lo.toFixed(lo >= 99 ? 1 : 0)}%</b> of all abstract pairs (top ${top < 0.1 ? "0.1" : top.toFixed(top < 10 ? 1 : 0)}%).</p>`;
  }

  const rankOf = (from, to) => { const k = nodes[from].nt.findIndex(([j]) => j === to); return k < 0 ? null : k + 1; };

  function show() {
    const { a, b } = state;
    history.replaceState(null, "", a != null || b != null ? `#${[a != null ? `a=${a}` : "", b != null ? `b=${b}` : ""].filter(Boolean).join("&")}` : location.pathname);
    if (a == null || b == null) { out.innerHTML = "<p class=\"sub\">Search for a first author in each box.</p>"; return; }
    if (a === b) { out.innerHTML = "<p class=\"sub\">That is the same abstract twice. Pick a different second abstract.</p>"; return; }
    const A = nodes[a], B = nodes[b];
    const m = pairs ? pairs[a * nodes.length + b] : null;
    const ra = rankOf(a, b), rb = rankOf(b, a);
    const rankText = (from, to, r) => r ? `${esc(author(nodes[to]))} is the <b>#${r}</b> closest text match to ${esc(author(nodes[from]))}` : `${esc(author(nodes[to]))} is not among ${esc(author(nodes[from]))}'s ${nodes[from].nt.length} closest`;
    const shared = A.au.filter((x) => B.au.includes(x));
    const union = new Set([...A.au, ...B.au]).size;
    const who = shared.map((id) => data.names[id]).filter(Boolean);
    // why: terms both abstracts weight heavily (each keeps only its heaviest terms), ranked by weight * weight
    const pairsOf = (r) => { const o = []; for (let k = 0; k < r.length; k += 2) o.push([r[k], r[k + 1]]); return o; };
    const wb = vecs ? new Map(pairsOf(vecs.rows[b])) : null;
    const overlap = wb ? pairsOf(vecs.rows[a]).filter(([t]) => wb.has(t)).map(([t, w]) => [vecs.terms[t], w * wb.get(t)]).sort((x, y) => y[1] - x[1]) : [];
    const total = overlap.reduce((s, [, p]) => s + p, 0);
    const why = !wb ? `<h2>Why</h2><p class="sub">Loading…</p>` : overlap.length ? `<h2>Why</h2><p class="sub">Words and two-word phrases both abstracts use, ranked by how much each adds to the match (percent of the total):</p>` +
      `<ul class="cmp-why">${overlap.slice(0, 10).map(([term, p]) => `<li><span class="bar" style="width:${Math.round(100 * p / overlap[0][1])}%"></span><b>${esc(term)}</b> ${Math.round(100 * p / total)}%</li>`).join("")}</ul>` +
      `<p class="sub">${overlap.length} shared term${overlap.length > 1 ? "s" : ""} in all.</p>` : `<h2>Why</h2><p class="sub">These two abstracts share no scored words or phrases.</p>`;
    const row = (k, f) => `<tr><th>${k}</th><td>${f(A)}</td><td>${f(B)}</td></tr>`;
    out.innerHTML =
      `<div class="cmp-score"><span class="big">${m == null ? "…" : pct(m)}</span><span>text match${m == null ? " (loading)" : ""}</span></div>` + (m == null ? "" : percentile(m)) +
      `<ul class="cmp-facts"><li>${rankText(a, b, ra)}.</li><li>${rankText(b, a, rb)}.</li>` +
      `<li>${shared.length ? `<b>${shared.length}</b> shared author${shared.length > 1 ? "s" : ""} (${Math.round(100 * shared.length / union)}% of all their authors): ${esc(who.join(", "))}` : "No shared authors."}</li></ul>` +
      `<table class="cmp-table"><thead><tr><th></th><th>${esc(author(A))}</th><th>${esc(author(B))}</th></tr></thead><tbody>` +
      row("Type", (n) => TYPES[n.t]) + row("Track", (n) => esc(tracks[n.k]) + (n.inf ? " (inferred)" : "")) + row("Time", (n) => esc(n.w)) +
      row("Location", (n) => (n.b ? `Board ${esc(n.b)}` : "Room not in the data")) +
      row("", (n) => `<a href="similar.html#n=${nodes.indexOf(n)}">Closest abstracts →</a>`) + `</tbody></table>` + why;
  }

  for (const slot of ["a", "b"]) {
    const box = slots[slot], input = box.querySelector("input"), res = box.querySelector(".sim-results");
    input.addEventListener("input", () => {
      const hits = matches(input.value);
      res.innerHTML = hits.length ? hits.map((i) => `<button type="button" data-id="${i}">${esc(nodes[i].a)}<span>${TYPES[nodes[i].t]} · ${esc(nodes[i].w)}${nodes[i].b ? ` · ${esc(nodes[i].b)}` : ""}</span></button>`).join("")
        : input.value.trim().length < 2 ? "" : "<p class=\"sub\">No first-author matches.</p>";
      res.hidden = !res.innerHTML;
    });
    res.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) pick(slot, Number(b.dataset.id)); });
  }
  document.addEventListener("click", (e) => { for (const s of Object.values(slots)) if (!s.contains(e.target)) s.querySelector(".sim-results").hidden = true; });

  fetch(root.dataset.src).then((r) => r.json()).then((d) => {
    data = d; nodes = d.nodes; tracks = d.tracks;
    const q = new URLSearchParams(location.hash.slice(1));
    for (const s of ["a", "b"]) { const v = q.get(s); if (v != null && Number(v) < nodes.length) { state[s] = Number(v); renderChosen(s); } }
    if (state.a != null || state.b != null) loadPairs();
    show();
  }).catch(() => { out.textContent = "Couldn't load the abstract network. Try reloading."; });
})();
