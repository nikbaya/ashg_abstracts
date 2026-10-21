// "Find talks": a local ego-network of the abstracts most similar to one abstract.
// Data is outputs/network.json from network.py: first author, track, type, board, time and neighbor ids. No titles.
(() => {
  const root = document.getElementById("similar");
  if (!root) return;
  const $ = (sel) => root.querySelector(sel);
  const PALETTE = [...d3.schemeTableau10, ...d3.schemeDark2, "#7a5195"];
  const TYPES = { poster: "Poster", platform: "Platform talk", plenary: "Plenary", symposium: "Symposium talk", session: "Session" };
  const KINDS = { poster: "Poster", symposium: "Symposium", session: "Session" }; // list badge; everything else is a "Talk"
  const params = new URLSearchParams(location.search);
  const now = Number(params.get("now")) || Date.now() / 1000; // ?now=<unix seconds> previews another moment

  const FANOUT = [15, 4, 2]; // new neighbors per node at each degree: 15 + 15*4 + 60*2 = 195 bubbles at most
  const state = { id: null, by: "nt", sort: "match", hidePast: false, selected: null, degrees: 1, history: [] }; // history: abstracts visited before the current one, oldest first
  let data, nodes, tracks, color;

  const svg = d3.select("#sim-graph");
  const linkLayer = svg.append("g").attr("class", "links");
  const nodeLayer = svg.append("g").attr("class", "nodes");
  const tip = $("#sim-tip");
  const stage = $("#sim-stage");
  let width = 0, height = 0, sy = 1, vh = 0; // the layout runs in width x vh space and is stretched vertically by sy to fill tall (phone) panes
  const pos = new Map(); // id -> {x, y}, kept between states so nodes glide instead of jumping

  const isPast = (n) => n.e < now;
  const author = (n) => n.a || n.n || "Unknown author"; // symposium talks without a named speaker fall back to their session
  const short = (s) => (s.length > 26 ? s.slice(0, 25) + "…" : s);
  const PLANNER = "https://ashg.floq.live/event/ashg-2026/dailyprogramme?objectClass=person&type=detail&objectId=";
  const ENTRY = "https://ashg.floq.live/event/ashg-2026/search?objectClass=timeslot&type=detail&objectId=";
  const planner = (n) => n.u ? `<a class="ext" href="${ENTRY}${n.u}" target="_blank" rel="noopener" title="This entry in the online program">Program ↗</a>`
    : n.p.length ? `<a class="ext" href="${PLANNER}${n.p[0]}" target="_blank" rel="noopener" title="Presenter in the online planner">Planner ↗</a>` : "";
  const touch = () => matchMedia("(pointer: coarse)").matches;

  function details(n, match, depth) {
    const rows = [[n.t === "symposium" || n.t === "session" ? "Speaker" : "Author", author(n)]];
    if (n.n && n.n !== n.a) rows.push(["Session", n.n]);
    rows.push(["Track", tracks[n.k] + (n.inf ? " (inferred from text)" : "")], ["Type", TYPES[n.t]],
      ["Location", n.b ? `Board ${n.b}` : "Room not in the data"], ["Time", n.w]);
    if (n.lw) rows.push(["Lightning talk", n.lw]);
    if (match != null) rows.push(["Match", `${match}%${depth > 1 ? " to the one it links from" : ""}`]);
    return rows.map(([k, v]) => `<div><span>${k}</span>${v.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("");
  }

  function showTip(event, d) {
    const n = nodes[d.id];
    tip.innerHTML = details(n, d.match, d.depth) + (touch() && d.id !== state.id ? "<em>Tap again to center on this one</em>" : "");
    tip.hidden = false;
    const box = stage.getBoundingClientRect();
    const x = Math.min(Math.max(d.x - tip.offsetWidth / 2, 4), box.width - tip.offsetWidth - 4), dy = d.y * sy;
    const y = dy + d.r + 8 + tip.offsetHeight > box.height ? dy - d.r - 8 - tip.offsetHeight : dy + d.r + 8;
    tip.style.transform = `translate(${x}px, ${Math.max(y, 4)}px)`;
  }

  function hideTip() { tip.hidden = true; state.selected = null; }

  const usable = (id) => nodes[id][state.by].filter(([j]) => !state.hidePast || !isPast(nodes[j]));

  // breadth-first from the center, each node adding its closest not-yet-seen neighbors; links run parent -> child
  function expand() {
    const list = [{ id: state.id, r: 22, center: true, depth: 0 }], seen = new Set([state.id]), links = [];
    let ring = [list[0]];
    for (let depth = 1; depth <= state.degrees; depth++) {
      const next = [];
      for (const parent of ring) {
        let added = 0;
        for (const [id, match] of usable(parent.id)) {
          if (added === FANOUT[depth - 1]) break;
          if (seen.has(id)) continue;
          seen.add(id); added++;
          const d = { id, match, depth, r: d3.scaleSqrt().domain([0, 100]).range([4, 19 - 3 * depth])(match) };
          next.push(d); list.push(d); links.push({ source: parent, target: d });
        }
      }
      ring = next;
    }
    return { list, links };
  }

  const sim = d3.forceSimulation().alphaDecay(0.04).velocityDecay(0.35);

  function render() {
    const { list: data_, links } = expand();
    const center = nodes[state.id];
    const total = nodes[state.id][state.by].length, hidden = total - usable(state.id).length;
    for (const d of data_) {
      const p = pos.get(d.id);
      d.x = p ? p.x : width / 2; d.y = p ? p.y : vh / 2;
    }
    const reach = Math.min(width, vh) / 2 - 40;

    // First-ring neighbors sit at evenly spaced angles (keeping their current order around the center, so a
    // re-center glides) with the radius set by match, so the ring never has big radial gaps.
    const cx = width / 2, cy = vh / 2, ring = data_.filter((d) => d.depth === 1), known = new Set(ring.filter((d) => pos.has(d.id)));
    const angle = (d) => Math.atan2(d.y - cy, d.x - cx);
    const order = [...[...known].sort((a, b) => angle(a) - angle(b)), ...ring.filter((d) => !known.has(d))];
    const a0 = order.length && known.size ? angle(order[0]) : -Math.PI / 2;
    order.forEach((d, i) => {
      const a = a0 + (2 * Math.PI * i) / order.length, R = reach * Math.min(1, 1.05 - 0.6 * d.match / 100);
      d.tx = cx + R * Math.cos(a); d.ty = cy + R * Math.sin(a);
      if (!known.has(d)) { d.x = d.tx; d.y = d.ty; }
    });
    for (const l of links) { // deeper new nodes start just outside their parent
      const d = l.target, p = l.source;
      if (d.depth > 1 && !pos.has(d.id)) {
        const dx = p.x - cx, dy = p.y - cy, n = Math.hypot(dx, dy) || 1;
        d.x = p.x + (dx / n) * 30 + (Math.random() - 0.5) * 20; d.y = p.y + (dy / n) * 30 + (Math.random() - 0.5) * 20;
      }
    }
    const pinRing = () => { for (const d of ring) { d.vx += (d.tx - d.x) * 0.25; d.vy += (d.ty - d.y) * 0.25; } };

    const g = nodeLayer.selectAll("g.node").data(data_, (d) => d.id);
    g.exit().transition().duration(250).style("opacity", 0).remove();
    const enter = g.enter().append("g").attr("class", "node").attr("tabindex", 0).attr("role", "button").style("opacity", 0);
    enter.append("circle").attr("r", 0);
    enter.append("text").attr("text-anchor", "middle");
    const all = enter.merge(g);
    const visited = new Set(state.history);
    all.classed("center", (d) => d.center).classed("visited", (d) => visited.has(d.id) && !d.center)
      .attr("aria-label", (d) => `${author(nodes[d.id])}, ${tracks[nodes[d.id].k]}${d.match != null ? `, ${d.match}% match` : ""}`)
      .on("pointerenter", (e, d) => { if (e.pointerType === "mouse") showTip(e, d); })
      .on("pointerleave", (e) => { if (e.pointerType === "mouse") hideTip(); })
      .on("focus", (e, d) => showTip(e, d))
      .on("blur", hideTip)
      .on("keydown", (e, d) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(d.id); } })
      .on("click", (e, d) => {
        if (d.center) return;
        if (e.pointerType === "mouse" || state.selected === d.id) return go(d.id);
        state.selected = d.id; showTip(e, d);
      });
    all.transition().duration(400).style("opacity", (d) => (isPast(nodes[d.id]) && !d.center ? 0.55 : 1));
    all.select("circle").attr("fill", (d) => color(nodes[d.id].k)).transition().duration(500).attr("r", (d) => d.r);
    all.select("text").text((d) => (d.depth < 2 ? short(author(nodes[d.id])) : "")).attr("dy", (d) => d.r + 12);

    const link = linkLayer.selectAll("line").data(links, (d) => d.target.id);
    link.exit().remove();
    const lines = link.enter().append("line").merge(link);

    sim.nodes(data_)
      .force("link", d3.forceLink(links).distance((l) => reach / l.target.depth * (1.05 - 0.6 * l.target.match / 100)).strength((l) => (l.target.depth === 1 ? 0.05 : 0.6)))
      .force("ring", pinRing)
      .force("charge", d3.forceManyBody().strength(state.degrees > 1 ? -35 : -60))
      .force("collide", d3.forceCollide((d) => d.r + (d.depth > 1 ? 2 : 10)))
      .force("x", d3.forceX(width / 2).strength((d) => (d.center ? 0.5 : 0.03)))
      .force("y", d3.forceY(vh / 2).strength((d) => (d.center ? 0.5 : 0.03)))
      .on("tick", () => {
        for (const d of data_) {
          d.x = Math.max(d.r + 6, Math.min(width - d.r - 6, d.x));
          d.y = Math.max(d.r + 6, Math.min(vh - d.r - 18 / sy, d.y));
          pos.set(d.id, { x: d.x, y: d.y });
        }
        all.attr("transform", (d) => `translate(${d.x},${d.y * sy})`);
        lines.attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y * sy).attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y * sy);
      })
      .alpha(0.9).restart();

    const by = state.by === "nt" ? "text" : "co-authors";
    $("#sim-card").innerHTML = `<h2>${author(center)} ${planner(center)}</h2><dl>${details(center)}</dl>` +
      `<p class="sub">${data_.length > 1 ? `Showing ${data_.length - 1} abstracts within ${state.degrees} degree${state.degrees > 1 ? "s" : ""} by ${by}` : `No abstracts to show by ${by}`}` +
      `${state.by === "na" && !total ? ": no other abstract shares an author with this one" : ""}` +
      `${hidden && state.hidePast ? ` (${hidden} already past are hidden)` : ""}.</p>`;
    // upcoming: chronological, scrolled so the current or next session is at the top; earlier sessions sit above
    const ranked = data_.slice(1).sort(state.sort === "time"
      ? (a, b) => nodes[a.id].s - nodes[b.id].s
      : (a, b) => b.match - a.match);
    $("#sim-list").innerHTML = ranked.map((d) => { const n = nodes[d.id];
      return `<li>${planner(n)}<button type="button" data-id="${d.id}"${isPast(n) ? ' class="past"' : n.s <= now ? ' class="live"' : ""}><b>${d.match}%</b> ${author(n)}<em class="kind ${n.t === "poster" ? "poster" : "talk"}">${KINDS[n.t] || "Talk"}</em>` +
        `<span>${tracks[n.k]}${n.inf ? " (inferred)" : ""}<br>${n.w}${n.lw ? ` · lightning ${n.lw}` : ""}${n.b ? ` · ${n.b}` : ""}${d.depth > 1 ? ` · ${d.depth} degrees` : ""}</span></button></li>`; }).join("");
    const side = $("#sim-list"), first = state.sort === "time" && ranked.findIndex((d) => !isPast(nodes[d.id]));
    side.scrollTop = first > 0 ? side.children[first].getBoundingClientRect().top - side.getBoundingClientRect().top + side.scrollTop : 0;
    renderTrail();
    history.replaceState(null, "", `#n=${state.id}`);
  }

  function go(id, fresh) {
    hideTip();
    if (fresh) { state.history = []; state.id = id; } else if (id !== state.id) { if (state.history.at(-1) !== state.id) state.history.push(state.id); state.id = id; }
    render();
  }

  const TRAIL_SHOWN = 6;
  function renderTrail() {
    const h = state.history;
    $("#sim-trail").hidden = !h.length;
    $("#sim-trail-list").innerHTML = (h.length > TRAIL_SHOWN ? "<i>…</i>" : "") +
      h.map((id, i) => [id, i]).slice(-TRAIL_SHOWN).map(([id, i]) => `<button type="button" data-id="${id}" data-i="${i}">${short(author(nodes[id]))}</button>`).join("<i>›</i>") + "<i>›</i><b>now</b>";
    $("#sim-trail-list").scrollLeft = 1e6; // keep the current end in view
  }

  function resize() {
    const box = stage.getBoundingClientRect();
    width = box.width; height = box.height;
    sy = Math.min(1.8, Math.max(1, height / width * 0.9)); vh = height / sy;
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    if (data) { pos.clear(); render(); }
  }

  function search(q) {
    const out = $("#sim-results");
    q = q.trim().toLowerCase();
    if (q.length < 2) { out.hidden = true; return; }
    // names are stored as "Surname, I."; every typed word must start a name word, or be a full first name with that initial
    const words = q.split(/[\s,.]+/).filter(Boolean);
    const ok = (name) => { const t = name.toLowerCase().split(/[\s,.]+/).filter(Boolean);
      return words.every((w) => t.some((x) => x.startsWith(w)) || (w.length > 1 && t.some((x) => x.length === 1 && x === w[0]))); };
    // exact full-name match first, then surname matches (exact, then prefix), then matches on the initial or a first name
    const toks = (name) => name.toLowerCase().split(/[\s,.]+/).filter(Boolean);
    const rank = (name) => { const t = toks(name), sur = name.toLowerCase().split(",")[0].trim();
      return t.length === words.length && t.every((x) => words.includes(x)) ? -1 : words.includes(sur) ? 0 : words.some((w) => sur.startsWith(w)) ? 1 : 2; };
    const hits = [];
    for (let i = 0; i < nodes.length; i++) if (ok(author(nodes[i]))) hits.push(i);
    hits.sort((a, b) => rank(author(nodes[a])) - rank(author(nodes[b]))).splice(8);
    const row = (i, last) => `<button type="button" data-id="${i}">${author(nodes[i])}${last ? '<em class="kind last">Last author</em>' : ""}<span>${TYPES[nodes[i].t]} · ${nodes[i].w}${nodes[i].b ? ` · ${nodes[i].b}` : ""}</span></button>`;
    // co-authors (names are "First M. Last"): each match lists the first authors of the abstracts they appear on
    const sn = (name) => name.replace(/^(.*)\s(\S+)$/, "$2, $1");
    const cos = data.co.filter(([name]) => ok(sn(name))).sort((a, b) => rank(sn(a[0])) - rank(sn(b[0]))).slice(0, 4);
    // last-author abstracts come first in ids: show those, or the first three if there are none; the rest sit behind a button
    const shown = (ids, nl) => { const k = nl || Math.min(3, ids.length);
      return ids.slice(0, k).map((i, j) => row(i, j < nl)).join("") + (ids.length > k ? `<button type="button" class="more" data-more>Show ${ids.length - k} more</button><div hidden>${ids.slice(k).map((i) => row(i)).join("")}</div>` : ""); };
    // sections in order of their best match, first author before co-author on a tie, so an exact co-author
    // match is not buried under looser first-author hits
    const secs = cos.map(([name, ids, nl]) => ({ r: rank(sn(name)), k: 1, html: `<p class="sub">${name}, co-author on ${ids.length === 1 ? "1 abstract" : ids.length + " abstracts"}${nl ? ` (last author on ${nl})` : ""}</p>` +
      shown(ids, nl) }));
    if (hits.length) secs.push({ r: Math.min(...hits.map((i) => rank(author(nodes[i])))), k: 0, html: `<p class="sub">First author</p>` + hits.map((i) => row(i)).join("") });
    secs.sort((x, y) => x.r - y.r || x.k - y.k);
    out.innerHTML = secs.length ? secs.map((x) => x.html).join("") : "<p class=\"sub\">No author matches.</p>";
    out.hidden = false;
  }

  function bind() {
    root.querySelectorAll("[data-sim=by] button").forEach((b) => b.addEventListener("click", () => {
      state.by = b.dataset.value;
      root.querySelectorAll("[data-sim=by] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      hideTip(); render();
    }));
    root.querySelectorAll("[data-sim=sort] button").forEach((b) => b.addEventListener("click", () => {
      state.sort = b.dataset.value;
      root.querySelectorAll("[data-sim=sort] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      render();
    }));
    root.querySelectorAll("[data-sim=degrees] button").forEach((b) => b.addEventListener("click", () => {
      state.degrees = Number(b.dataset.value);
      root.querySelectorAll("[data-sim=degrees] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      hideTip(); render();
    }));
    const list = $("#sim-list"), mark = (id, on) => nodeLayer.selectAll("g.node").classed("hl", (d) => on && d.id === id);
    list.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) go(Number(b.dataset.id)); });
    list.addEventListener("pointerover", (e) => { const b = e.target.closest("button"); if (b) mark(Number(b.dataset.id), true); });
    list.addEventListener("pointerleave", () => mark(-1, false));
    $("#sim-past").addEventListener("change", (e) => { state.hidePast = e.target.checked; hideTip(); render(); });
    $("#sim-q").addEventListener("input", (e) => search(e.target.value));
    $("#sim-results").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b && b.dataset.more !== undefined) { b.nextElementSibling.hidden = false; b.remove(); return; }
      if (b) { $("#sim-results").hidden = true; $("#sim-q").value = ""; go(Number(b.dataset.id), true); } // a new search starts a fresh path
    });
    $("#sim-random").addEventListener("click", () => go(Math.floor(Math.random() * nodes.length)));
    stage.addEventListener("click", (e) => { if (!e.target.closest(".node")) hideTip(); });
    new ResizeObserver(() => { const b = stage.getBoundingClientRect(); if (b.width !== width || b.height !== height) resize(); }).observe(stage);
    $("#sim-trail-list").addEventListener("click", (e) => { // unravel: drop the clicked name and everything after it from the path
      const b = e.target.closest("button");
      if (!b) return;
      hideTip(); state.history.length = Number(b.dataset.i); state.id = Number(b.dataset.id); render();
    });
    $("#sim-clear").addEventListener("click", () => { state.history = []; renderTrail(); render(); });
  }

  fetch(root.dataset.src).then((r) => r.json()).then((d) => {
    data = d; nodes = d.nodes; tracks = d.tracks;
    color = (k) => PALETTE[k % PALETTE.length];
    $("#sim-legend").innerHTML = tracks.map((t, k) => `<li><i style="background:${color(k)}"></i>${t}</li>`).join("");
    const m = /^#n=(\d+)$/.exec(location.hash);
    state.id = m && Number(m[1]) < nodes.length ? Number(m[1]) : Math.floor(Math.random() * nodes.length);
    root.classList.add("ready");
    bind(); resize(); render();
  }).catch(() => { $("#sim-card").textContent = "Couldn't load the abstract network. Try reloading."; });
})();
