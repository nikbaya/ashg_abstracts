// "Find talks": a local ego-network of the abstracts most similar to one abstract, plus a minimap of all of them.
// Data is outputs/network.json from network.py: first author, track, type, board, time and neighbor ids. No titles.
(() => {
  const root = document.getElementById("similar");
  if (!root) return;
  const $ = (sel) => root.querySelector(sel);
  const PALETTE = [...d3.schemeTableau10, ...d3.schemeDark2, "#7a5195"];
  const TYPES = { poster: "Poster", platform: "Platform talk", plenary: "Plenary" };
  const params = new URLSearchParams(location.search);
  const now = Number(params.get("now")) || Date.now() / 1000; // ?now=<unix seconds> previews another moment

  const FANOUT = [15, 4, 2]; // new neighbors per node at each degree: 15 + 15*4 + 60*2 = 195 bubbles at most
  const state = { id: null, by: "nt", hidePast: false, selected: null, degrees: 1, history: [] }; // history: abstracts visited before the current one, oldest first
  let data, nodes, tracks, color;

  const svg = d3.select("#sim-graph");
  const linkLayer = svg.append("g").attr("class", "links");
  const nodeLayer = svg.append("g").attr("class", "nodes");
  const tip = $("#sim-tip");
  const stage = $("#sim-stage");
  const mini = $("#sim-mini canvas.dots"), pathCanvas = $("#sim-mini canvas.path");
  const dot = $("#sim-here");
  let width = 0, height = 0, sy = 1, vh = 0; // the layout runs in width x vh space and is stretched vertically by sy to fill tall (phone) panes
  const pos = new Map(); // id -> {x, y}, kept between states so nodes glide instead of jumping

  const isPast = (n) => n.e < now;
  const author = (n) => n.a || "Unknown author";
  const PLANNER = "https://ashg.floq.live/event/ashg-2026/dailyprogramme?objectClass=person&type=detail&objectId=";
  const planner = (n) => n.p.length ? `<a class="ext" href="${PLANNER}${n.p[0]}" target="_blank" rel="noopener" title="Presenter in the online planner">Planner ↗</a>` : "";
  const touch = () => matchMedia("(pointer: coarse)").matches;

  function details(n, match, depth) {
    const rows = [["Author", author(n)], ["Track", tracks[n.k] + (n.inf ? " (inferred from text)" : "")],
      ["Type", TYPES[n.t]], ["Location", n.b ? `Board ${n.b}` : "Room not in the data"], ["Time", n.w]];
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
    all.select("text").text((d) => (d.depth < 2 ? nodes[d.id].a : "")).attr("dy", (d) => d.r + 12);

    const link = linkLayer.selectAll("line").data(links, (d) => d.target.id);
    link.exit().remove();
    const lines = link.enter().append("line").merge(link);

    sim.nodes(data_)
      .force("link", d3.forceLink(links).distance((l) => reach / l.target.depth * (1.05 - 0.6 * l.target.match / 100)).strength(0.6))
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
    const ranked = data_.slice(1).sort((a, b) => b.match - a.match);
    $("#sim-list").innerHTML = ranked.map((d) => { const n = nodes[d.id];
      return `<li>${planner(n)}<button type="button" data-id="${d.id}"${isPast(n) ? ' class="past"' : ""}><b>${d.match}%</b> ${author(n)}` +
        `<span>${tracks[n.k]}${n.inf ? " (inferred)" : ""}<br>${TYPES[n.t]} · ${n.w}${n.b ? ` · ${n.b}` : ""}${d.depth > 1 ? ` · ${d.depth} degrees` : ""}</span></button></li>`; }).join("");
    renderTrail();
    const dotPos = (n) => ({ left: `${100 * n.x}%`, top: `${100 * n.y}%` });
    Object.assign(dot.style, dotPos(center));
    history.replaceState(null, "", `#n=${state.id}`);
  }

  function go(id) {
    hideTip();
    if (id !== state.id) { if (state.history.at(-1) !== state.id) state.history.push(state.id); state.id = id; }
    render();
  }

  const TRAIL_SHOWN = 10;
  function renderTrail() {
    const h = state.history;
    $("#sim-trail").hidden = !h.length;
    $("#sim-trail-list").innerHTML = (h.length > TRAIL_SHOWN ? "<i>…</i>" : "") +
      h.slice(-TRAIL_SHOWN).map((id) => `<button type="button" data-id="${id}">${author(nodes[id])}</button>`).join("<i>›</i>") + "<i>›</i><b>now</b>";
    drawPath();
  }

  function drawPath() { // grey line through the visited abstracts on the minimap, ending at the current one
    const size = pathCanvas.clientWidth, dpr = window.devicePixelRatio || 1;
    pathCanvas.width = pathCanvas.height = size * dpr;
    const c = pathCanvas.getContext("2d");
    c.scale(dpr, dpr);
    const pts = [...state.history, state.id].map((id) => [nodes[id].x * (size - 2), nodes[id].y * (size - 2)]);
    c.strokeStyle = "rgb(60 60 60 / 0.6)"; c.lineWidth = 1;
    c.beginPath(); pts.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke();
    c.fillStyle = "rgb(60 60 60 / 0.8)"; pts.slice(0, -1).forEach(([x, y]) => c.fillRect(x - 1.5, y - 1.5, 3, 3));
  }

  function resize() {
    const box = stage.getBoundingClientRect();
    width = box.width; height = box.height;
    sy = Math.min(1.8, Math.max(1, height / width * 0.9)); vh = height / sy;
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    if (data) { pos.clear(); render(); }
  }

  function drawMini() {
    const size = mini.clientWidth, dpr = window.devicePixelRatio || 1;
    mini.width = mini.height = size * dpr;
    const c = mini.getContext("2d");
    c.scale(dpr, dpr);
    nodes.forEach((n) => {
      c.globalAlpha = state.hidePast && isPast(n) ? 0.12 : 0.8;
      c.fillStyle = color(n.k);
      c.fillRect(n.x * (size - 2), n.y * (size - 2), 2, 2);
    });
  }

  function search(q) {
    const out = $("#sim-results");
    q = q.trim().toLowerCase();
    if (q.length < 2) { out.hidden = true; return; }
    // names are stored as "Surname, I."; every typed word must start a name word, or be a full first name with that initial
    const words = q.split(/[\s,.]+/).filter(Boolean);
    const ok = (name) => { const t = name.toLowerCase().split(/[\s,.]+/).filter(Boolean);
      return words.every((w) => t.some((x) => x.startsWith(w)) || (w.length > 1 && t.some((x) => x.length === 1 && x === w[0]))); };
    // surname matches first (exact, then prefix), then matches on the initial or a first name
    const rank = (name) => { const sur = name.toLowerCase().split(",")[0].trim();
      return words.includes(sur) ? 0 : words.some((w) => sur.startsWith(w)) ? 1 : 2; };
    const hits = [];
    for (let i = 0; i < nodes.length; i++) if (ok(nodes[i].a)) hits.push(i);
    hits.sort((a, b) => rank(nodes[a].a) - rank(nodes[b].a)).splice(8);
    out.innerHTML = hits.length ? hits.map((i) => `<button type="button" data-id="${i}">${author(nodes[i])}<span>${TYPES[nodes[i].t]} · ${nodes[i].w}${nodes[i].b ? ` · ${nodes[i].b}` : ""}</span></button>`).join("")
      : "<p class=\"sub\">No first author matches.</p>";
    out.hidden = false;
  }

  function bind() {
    root.querySelectorAll("[data-sim=by] button").forEach((b) => b.addEventListener("click", () => {
      state.by = b.dataset.value;
      root.querySelectorAll("[data-sim=by] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      hideTip(); render();
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
    $("#sim-past").addEventListener("change", (e) => { state.hidePast = e.target.checked; drawMini(); hideTip(); render(); });
    $("#sim-q").addEventListener("input", (e) => search(e.target.value));
    $("#sim-results").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (b) { $("#sim-results").hidden = true; $("#sim-q").value = ""; go(Number(b.dataset.id)); }
    });
    $("#sim-random").addEventListener("click", () => go(Math.floor(Math.random() * nodes.length)));
    stage.addEventListener("click", (e) => { if (!e.target.closest(".node")) hideTip(); });
    $("#sim-mini").addEventListener("click", (e) => { // jump to the abstract nearest the tapped spot
      const r = mini.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      let best = 0, bd = Infinity;
      nodes.forEach((n, i) => { const d = (n.x - x) ** 2 + (n.y - y) ** 2; if (d < bd) { bd = d; best = i; } });
      go(best);
    });
    new ResizeObserver(() => { const b = stage.getBoundingClientRect(); if (b.width !== width || b.height !== height) resize(); }).observe(stage);
    window.addEventListener("resize", () => { drawMini(); drawPath(); });
    $("#sim-trail-list").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) go(Number(b.dataset.id)); });
    $("#sim-clear").addEventListener("click", () => { state.history = []; renderTrail(); render(); });
  }

  fetch(root.dataset.src).then((r) => r.json()).then((d) => {
    data = d; nodes = d.nodes; tracks = d.tracks;
    color = (k) => PALETTE[k % PALETTE.length];
    $("#sim-legend").innerHTML = tracks.map((t, k) => `<li><i style="background:${color(k)}"></i>${t}</li>`).join("");
    const m = /^#n=(\d+)$/.exec(location.hash);
    state.id = m && Number(m[1]) < nodes.length ? Number(m[1]) : Math.floor(Math.random() * nodes.length);
    root.classList.add("ready");
    bind(); resize(); drawMini(); render();
  }).catch(() => { $("#sim-card").textContent = "Couldn't load the abstract network. Try reloading."; });
})();
