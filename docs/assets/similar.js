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
  const state = { id: null, by: "nt", sort: "match", hidePast: false, hideCo: false, open: new Set(), selected: null, degrees: 1, colorBy: "track", history: [] }; // history: abstracts visited before the current one, oldest first
  let data, nodes, tracks, color;
  let pairs = null, pairsLoading = false; // pairs: n*n bytes, any pair's text match (0 = under 10%); fetched the first time a deeper degree needs it
  let vecs = null, vecsLoading = false; // vecs: {terms, rows}, rows[i] = [term, weight, term, weight, ...]; fetched the first time "Keyword scores" is opened
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  function sharedTerms(a, b, k = 6) { // terms both abstracts use, by share of the match between them
    const w = new Map(); for (let i = 0; i < vecs.rows[b].length; i += 2) w.set(vecs.rows[b][i], vecs.rows[b][i + 1]);
    const r = vecs.rows[a], hits = [];
    for (let i = 0; i < r.length; i += 2) if (w.has(r[i])) hits.push([vecs.terms[r[i]], r[i + 1] * w.get(r[i])]);
    const total = hits.reduce((s, h) => s + h[1], 0);
    return { n: hits.length, top: hits.sort((x, y) => y[1] - x[1]).slice(0, k).map(([t, p]) => [t, Math.round(100 * p / total)]) };
  }
  function whyHtml(id) {
    if (!vecs) return "<div class=\"why-terms\">Loading…</div>";
    const { n, top } = sharedTerms(state.id, id);
    return `<div class="why-terms">${n ? `${top.map(([t, p]) => `<span>${esc(t)} <i>${p}%</i></span>`).join("")}${n > top.length ? `<em>+${n - top.length} more</em>` : ""}` : "No shared words or phrases with the center."}</div>`;
  }
  function paintWhy(li) { // sync one row's open/closed state without re-rendering the graph or list
    const id = Number(li.dataset.id), open = state.open.has(id);
    li.querySelector("button.why").setAttribute("aria-expanded", open);
    const old = li.querySelector(".why-terms");
    if (old) old.remove();
    if (open) li.insertAdjacentHTML("beforeend", whyHtml(id));
  }
  function loadVecs() {
    if (vecs || vecsLoading || !root.dataset.terms) return;
    vecsLoading = true;
    fetch(root.dataset.terms).then((r) => new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).json())
      .then((v) => { vecs = v; $("#sim-list").querySelectorAll("li").forEach((li) => paintWhy(li)); }).catch(() => { vecsLoading = false; });
  }
  const pct = (m) => (m ? `${m}%` : "<10%");

  const svg = d3.select("#sim-graph");
  const view = svg.append("g").attr("class", "view"); // pan/zoom happens on this group
  const linkLayer = view.append("g").attr("class", "links");
  const nodeLayer = view.append("g").attr("class", "nodes");
  let zt = d3.zoomIdentity; // current pan/zoom, so the tooltip can follow its bubble
  const zoom = d3.zoom().scaleExtent([0.4, 6]).on("zoom", (e) => { zt = e.transform; view.attr("transform", zt); hideTip(); });
  svg.call(zoom).on("dblclick.zoom", null);
  const recenter = (ms = 0) => svg.transition().duration(ms).call(zoom.transform, d3.zoomIdentity);
  const kindOf = (id) => (nodes[id].t === "poster" ? "poster" : "talk");
  const fillOf = (id) => (state.colorBy === "type" ? `var(--${kindOf(id)})` : color(nodes[id].k));
  function paintLegend() {
    $("#sim-legend-title").textContent = state.colorBy === "type" ? "Talk and poster colors" : "Track colors";
    $("#sim-legend").innerHTML = state.colorBy === "type"
      ? [["poster", "Poster"], ["talk", "Talk (platform, plenary, symposium, session)"]].map(([k, t]) => `<li><i style="background:var(--${k})"></i>${t}</li>`).join("")
      : tracks.map((t, k) => `<li><i style="background:${color(k)}"></i>${t}</li>`).join("");
  }
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
    : n.p.length ? `<a class="ext" href="${PLANNER}${n.p[0]}" target="_blank" rel="noopener" title="Presenter in the online planner">ASHG Planner ↗</a>` : "";
  const CAL = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="11" rx="1.5"/><path d="M2 6.5h12M5.5 1.5v3M10.5 1.5v3M8 8.5v3.5M6.25 10.25h3.5"/></svg>';
  const gcal = (n, match, to, short) => { // Google Calendar "create event" link; times are unix seconds, written as UTC
    const t = (x) => new Date(x * 1000).toISOString().replace(/[-:]|\.\d+/g, "");
    const q = new URLSearchParams({ action: "TEMPLATE", text: `ASHG 2026: ${author(n)} (${TYPES[n.t]})`, dates: `${t(n.s)}/${t(n.e)}` });
    if (n.b) q.set("location", `Board ${n.b}`);
    const url = n.u ? ENTRY + n.u : n.p.length ? PLANNER + n.p[0] : "";
    q.set("details", [match != null ? `Similarity to ${author(to)}: ${match}%` : "", url].filter(Boolean).join("\n"));
    return `<a class="cal" href="https://calendar.google.com/calendar/render?${q}" target="_blank" rel="noopener" title="Add this to Google Calendar" aria-label="Add this to Google Calendar">${CAL}${short ? "Add" : "Google Calendar"}</a>`;
  };
  const hallLink = (n) => n.t === "poster" && n.b && window.hallMap
    ? `<details class="hall" open><summary>Poster hall map</summary>${window.hallMap(n.b)}</details>` : "";
  function centerHall() { // wide blocks scroll sideways: bring the board into view
    const box = document.querySelector("details.hall[open] .hall-scroll"), on = box && box.querySelector(".hl-s.on");
    if (on) { const r = on.getBoundingClientRect(); box.scrollLeft = r.left + r.width / 2 - box.getBoundingClientRect().left + box.scrollLeft - box.clientWidth / 2; }
  }
  document.addEventListener("toggle", centerHall, true); // also when the reader reopens the map
  const touch = () => matchMedia("(pointer: coarse)").matches;

  function details(n, match, depth, via) {
    const rows = [[n.t === "symposium" || n.t === "session" ? "Speaker" : "Author", author(n)]];
    if (n.n && n.n !== n.a) rows.push(["Session", n.n]);
    rows.push(["Track", tracks[n.k] + (n.inf ? " (inferred from text)" : "")], ["Type", TYPES[n.t]],
      ["Location", n.b ? `Board ${n.b}` : "Room not in the data"], ["Time", n.w]);
    if (n.lw) rows.push(["Lightning talk", n.lw]);
    if (match != null) rows.push(["Match", depth > 1 && via != null ? `${pct(match)} to the center · ${via}% to the one it links from` : `${pct(match)}${depth > 1 ? " to the one it links from" : ""}`]);
    return rows.map(([k, v]) => `<div><span>${k}</span>${v.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("");
  }

  function showTip(event, d) {
    const n = nodes[d.id];
    tip.innerHTML = details(n, d.match, d.depth, d.via) + (touch() && d.id !== state.id ? "<em>Tap again to center on this one</em>" : "");
    tip.hidden = false;
    const box = stage.getBoundingClientRect();
    const r = d.r * zt.k, px = zt.applyX(d.x), dy = zt.applyY(d.y * sy);
    const x = Math.min(Math.max(px - tip.offsetWidth / 2, 4), box.width - tip.offsetWidth - 4);
    const y = dy + r + 8 + tip.offsetHeight > box.height ? dy - r - 8 - tip.offsetHeight : dy + r + 8;
    tip.style.transform = `translate(${x}px, ${Math.max(y, 4)}px)`;
  }

  function hideTip() { tip.hidden = true; state.selected = null; }

  const shares = (j) => nodes[state.id].fa != null && nodes[j].au.includes(nodes[state.id].fa); // the center's first author is on j
  const usable = (id) => nodes[id][state.by].filter(([j]) => (!state.hidePast || !isPast(nodes[j])) && (!state.hideCo || !shares(j)));

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
          const toCenter = depth > 1 && state.by === "nt" && pairs ? pairs[state.id * nodes.length + id] : null; // text match to the center, not to the parent
          const d = { id, match: toCenter ?? match, via: toCenter != null ? match : null, depth, r: d3.scaleSqrt().domain([0, 100]).range([4, 19 - 3 * depth])(match) };
          next.push(d); list.push(d); links.push({ source: parent, target: d });
        }
      }
      ring = next;
    }
    return { list, links };
  }

  function loadPairs() { // center-to-node matches for degrees 2-3 (text only); until they arrive the page shows parent matches
    if (pairs || pairsLoading || state.degrees < 2 || !root.dataset.pairs) return;
    pairsLoading = true;
    fetch(root.dataset.pairs).then((r) => new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer())
      .then((buf) => { pairs = new Uint8Array(buf); render(); })
      .catch(() => { pairsLoading = false; });
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
    const ox = state.degrees > 1 ? width * 0.4 : 0, oy = state.degrees > 1 ? vh * 0.4 : 0; // deeper graphs may spill past the pane; pan to see them
    recenter();

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
      .attr("aria-label", (d) => `${author(nodes[d.id])}, ${tracks[nodes[d.id].k]}${d.match != null ? `, ${pct(d.match)} match` : ""}`)
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
    all.select("circle").style("fill", (d) => fillOf(d.id)).transition().duration(500).attr("r", (d) => d.r);
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
          d.x = Math.max(d.r + 6 - ox, Math.min(width - d.r - 6 + ox, d.x));
          d.y = Math.max(d.r + 6 - oy, Math.min(vh - d.r - 18 / sy + oy, d.y));
          pos.set(d.id, { x: d.x, y: d.y });
        }
        all.attr("transform", (d) => `translate(${d.x},${d.y * sy})`);
        lines.attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y * sy).attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y * sy);
      })
      .alpha(0.9).restart();

    const by = state.by === "nt" ? "text" : "co-authors";
    $("#sim-card").innerHTML = `<h2>${author(center)} ${planner(center)}</h2><dl>${details(center)}</dl>${hallLink(center)}` +
      `<p class="sub">${data_.length > 1 ? `Showing ${data_.length - 1} abstracts within ${state.degrees} degree${state.degrees > 1 ? "s" : ""} by ${by}` : `No abstracts to show by ${by}`}` +
      `${state.by === "na" && !total ? ": no other abstract shares an author with this one" : ""}` +
      `${hidden && state.hidePast ? ` (${hidden} already past are hidden)` : ""}.</p>`;
    centerHall();
    // upcoming: chronological, scrolled so the current or next session is at the top; earlier sessions sit above
    const ranked = data_.slice(1).sort(state.sort === "time"
      ? (a, b) => nodes[a.id].s - nodes[b.id].s
      : (a, b) => b.match - a.match);
    $("#sim-list").innerHTML = ranked.map((d) => { const n = nodes[d.id];
      return `<li data-id="${d.id}"${isPast(n) ? ' class="past"' : n.s <= now ? ' class="live"' : ""}>${planner(n)}<button type="button" data-id="${d.id}"><b>${pct(d.match)}</b> ${author(n)}<em class="kind ${n.t === "poster" ? "poster" : "talk"}">${KINDS[n.t] || "Talk"}</em>` +
        `<span>${tracks[n.k]}${n.inf ? " (inferred)" : ""}</span></button>` +
        `<div class="when">${n.w}${n.lw ? ` · lightning ${n.lw}` : ""}${n.b ? ` · ${n.b}` : ""}${d.depth > 1 ? ` · ${d.depth} degrees` : ""} · <button type="button" class="why" data-why="${d.id}" aria-expanded="${state.open.has(d.id)}" title="Show the words and phrases this abstract shares with the center">Keyword scores</button></div>${state.open.has(d.id) ? whyHtml(d.id) : ""}</li>`; }).join("");
    const side = $("#sim-list"), first = state.sort === "time" && ranked.findIndex((d) => !isPast(nodes[d.id]));
    side.scrollTop = first > 0 ? side.children[first].getBoundingClientRect().top - side.getBoundingClientRect().top + side.scrollTop : 0;
    renderTrail();
    history.replaceState(null, "", `#n=${state.id}`);
  }

  function go(id, fresh) {
    hideTip(); state.open.clear(); // shared words are relative to the center
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
    if (!box.width) return; // graph hidden
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
    const board = /^(?:board\s*)?(\d{2,4})([wtf]?)$/.exec(q); // a poster board number, with or without the day letter
    if (board) {
      const ids = []; for (let i = 0; i < nodes.length; i++) if (nodes[i].b && nodes[i].b.toLowerCase().startsWith(board[1] + board[2])) ids.push(i);
      ids.sort((a, b) => nodes[a].b.localeCompare(nodes[b].b));
      out.innerHTML = ids.length ? `<p class="sub">Poster board</p>` + ids.slice(0, 8).map((i) => `<button type="button" data-id="${i}">${author(nodes[i])}<span>${TYPES[nodes[i].t]} · ${nodes[i].w} · ${nodes[i].b}</span></button>`).join("")
        + (ids.length > 8 ? `<p class="sub">${ids.length - 8} more: type more of the number</p>` : "") : "<p class=\"sub\">No poster board matches.</p>";
      out.hidden = false; return;
    }
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
      hideTip(); render(); loadPairs();
    }));
    root.querySelectorAll("[data-sim=sort] button").forEach((b) => b.addEventListener("click", () => {
      state.sort = b.dataset.value;
      root.querySelectorAll("[data-sim=sort] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      render();
    }));
    root.querySelectorAll("[data-sim=degrees] button").forEach((b) => b.addEventListener("click", () => {
      state.degrees = Number(b.dataset.value);
      root.querySelectorAll("[data-sim=degrees] button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      hideTip(); render(); loadPairs();
    }));
    $("#sim-color").addEventListener("change", (e) => {
      state.colorBy = e.target.value;
      paintLegend(); nodeLayer.selectAll("g.node circle").style("fill", (d) => fillOf(d.id));
    });
    $("#sim-recenter").addEventListener("click", () => recenter(300));
    $("#sim-hide-graph").addEventListener("change", (e) => { root.classList.toggle("no-graph", e.target.checked); hideTip(); }); // showing it again resizes via the observer
    const list = $("#sim-list"), mark = (id, on) => nodeLayer.selectAll("g.node").classed("hl", (d) => on && d.id === id);
    list.addEventListener("click", (e) => { const b = e.target.closest("li"); if (!b || e.target.closest("a")) return;
      const w = e.target.closest("button.why");
      if (!w) return go(Number(b.dataset.id));
      const id = Number(w.dataset.why); // toggle: shared words with the center
      if (state.open.has(id)) state.open.delete(id); else { state.open.add(id); loadVecs(); }
      paintWhy(b); });
    list.addEventListener("pointerover", (e) => { const b = e.target.closest("li"); if (b) mark(Number(b.dataset.id), true); });
    list.addEventListener("pointerleave", () => mark(-1, false));
    $("#sim-past").addEventListener("change", (e) => { state.hidePast = e.target.checked; hideTip(); render(); });
    $("#sim-co").addEventListener("change", (e) => { state.hideCo = e.target.checked; hideTip(); render(); });
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
    paintLegend();
    const m = /^#n=(\d+)$/.exec(location.hash);
    state.id = m && Number(m[1]) < nodes.length ? Number(m[1]) : Math.floor(Math.random() * nodes.length);
    root.classList.add("ready");
    bind(); resize(); render();
  }).catch(() => { $("#sim-card").textContent = "Couldn't load the abstract network. Try reloading."; });
})();
