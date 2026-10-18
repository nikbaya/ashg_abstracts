// Page toggles: a [data-control=K] button group shows the elements whose data-show-K matches the pressed
// button. The year is remembered across pages and can be linked as #2024. Without JS everything shows.
(() => {
  const controls = [...document.querySelectorAll("[data-control]")];
  if (!controls.length) return;
  const state = {};
  const values = c => [...c.querySelectorAll("button")].map(b => b.dataset.value);
  let saved = null;
  try { saved = localStorage.getItem("ashg-year"); } catch (e) { /* storage blocked */ }
  controls.forEach(c => {
    const key = c.dataset.control, vals = values(c);
    const wanted = key === "year" ? [location.hash.slice(1), saved] : [];
    state[key] = wanted.find(v => vals.includes(v)) ?? c.dataset.default ?? vals[0];
  });

  function apply() {
    Object.entries(state).forEach(([key, v]) => {
      document.querySelectorAll(`[data-show-${key}]`).forEach(el => { el.hidden = el.getAttribute(`data-show-${key}`) !== v; });
      document.querySelectorAll(`[data-control="${key}"] button`).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
    });
  }
  controls.forEach(c => c.addEventListener("click", ev => {
    const b = ev.target.closest("button");
    if (!b) return;
    state[c.dataset.control] = b.dataset.value;
    if (c.dataset.control === "year") {
      try { localStorage.setItem("ashg-year", b.dataset.value); } catch (e) { /* storage blocked */ }
      history.replaceState(null, "", "#" + b.dataset.value);
    }
    apply();
  }));
  addEventListener("hashchange", () => {
    const c = document.querySelector('[data-control="year"]'), v = location.hash.slice(1);
    if (c && values(c).includes(v)) { state.year = v; apply(); }
  });
  apply();
})();

// Sortable tables: click a header; cells keep the raw value in data-v so numbers sort numerically.
document.querySelectorAll("table.sortable").forEach(table => {
  table.querySelectorAll("th").forEach((th, col) => th.addEventListener("click", () => {
    const asc = th.getAttribute("aria-sort") !== "ascending";
    table.querySelectorAll("th").forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", asc ? "ascending" : "descending");
    const body = table.tBodies[0];
    const key = r => { const t = r.cells[col].dataset.v ?? r.cells[col].textContent; const n = parseFloat(t); return isNaN(n) ? t : n; };
    [...body.rows].sort((a, b) => { const x = key(a), y = key(b); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); })
      .forEach(r => body.appendChild(r));
  }));
});

// Phrase explorer (overview page): search box filters the phrase list; per-year talk/poster mention
// shares as a small line chart; institutions and metros writing about the phrase.
(() => {
  const el = document.getElementById("drv-data");
  if (!el) return;
  // The overview page fetches the data (data-src); the phrases page inlines it.
  if (el.dataset.src) fetch(el.dataset.src).then(r => r.json()).then(start).catch(() => {});
  else start(JSON.parse(el.textContent));
  function start(data) {
  const phraseSel = document.getElementById("drv-phrase"), yearSeg = document.getElementById("drv-year");
  const search = document.getElementById("drv-search"), chart = document.getElementById("drv-chart");
  const tip = document.getElementById("drv-tip");
  let year = "all";
  const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const pct = v => (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%";
  // 95% CIs. oddsRatio: (a/b) ÷ (c/d), Woolf log interval, +0.5 to every cell if any is zero.
  // wilson: proportion k/n, the interval the site uses for talk rates.
  const oddsRatio = (a, b, c, d) => {
    if (!(a && b && c && d)) [a, b, c, d] = [a + 0.5, b + 0.5, c + 0.5, d + 0.5];
    const lr = Math.log(a * d / (b * c)), se = Math.sqrt(1 / a + 1 / b + 1 / c + 1 / d);
    return [Math.exp(lr), Math.exp(lr - 1.96 * se), Math.exp(lr + 1.96 * se)];
  };
  const wilson = (k, n, z = 1.96) => {
    const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  };
  // every other phrase in 5+ talks, searchable but without the institution/metro breakdown:
  // [phrase, [talks, posters] per year flattened, or, lo, hi], most-mentioned first
  const MORE = "More phrases (abstract text only)", MORE_MAX = 150;
  // a phrase in both lists keeps its listed entry (titles and plurals counted)
  const listed = new Set(data.phrases.map(p => p.phrase));
  const more = (data.more || []).filter(([phrase]) => !listed.has(phrase)).map(([phrase, flat, ...or]) =>
    ({ phrase, group: MORE, counts: data.years.map((_, i) => [flat[2 * i], flat[2 * i + 1]]), or }));
  const byPhrase = Object.fromEntries([...data.phrases, ...more].map(p => [p.phrase, p]));

  // best matches first: the exact phrase, then phrases starting with the query, then the rest
  const rank = (name, q) => { const l = name.toLowerCase(); return l === q ? 0 : l.startsWith(q) ? 1 : 2; };
  const byRank = q => (a, b) => rank(a, q) - rank(b, q);
  const suggest = document.getElementById("drv-suggest");

  // type-ahead popup under the search box: best matches first, matched text in bold
  let active = -1;
  function showSuggestions(names, q) {
    suggest.replaceChildren(...names.map(name => {
      const li = node("li"), i = name.toLowerCase().indexOf(q);
      li.setAttribute("role", "option");
      li.dataset.value = name;
      li.append(name.slice(0, i), node("b", null, name.slice(i, i + q.length)), name.slice(i + q.length));
      return li;
    }));
    active = -1;
    suggest.hidden = !names.length || document.activeElement !== search;
    search.setAttribute("aria-expanded", String(!suggest.hidden));
  }
  function highlight(i) {
    const items = [...suggest.children];
    if (!items.length) return;
    active = (i + items.length) % items.length;
    items.forEach((li, k) => li.setAttribute("aria-selected", String(k === active)));
    items[active].scrollIntoView({ block: "nearest" });
  }
  function choose(name) {
    search.value = name;
    fillOptions(name.toLowerCase());
    phraseSel.value = name;
    suggest.hidden = true;
    search.setAttribute("aria-expanded", "false");
    render();
  }
  search?.addEventListener("keydown", ev => {
    if (ev.key === "ArrowDown") { ev.preventDefault(); highlight(active + 1); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); highlight(active - 1); }
    else if (ev.key === "Enter" && active >= 0) { ev.preventDefault(); choose(suggest.children[active].dataset.value); }
    else if (ev.key === "Escape") { suggest.hidden = true; }
  });
  suggest?.addEventListener("mousedown", ev => {  // mousedown, so it lands before the input loses focus
    const li = ev.target.closest("li");
    if (li) { ev.preventDefault(); choose(li.dataset.value); }
  });
  search?.addEventListener("blur", () => { suggest.hidden = true; });
  search?.addEventListener("focus", () => { if (suggest.children.length) suggest.hidden = false; });

  function fillOptions(q) {
    const keep = phraseSel.value;
    phraseSel.replaceChildren();
    const match = p => p.phrase.toLowerCase().includes(q);
    const hits = q ? [...data.phrases.filter(match), ...more.filter(match)] : [];
    const exact = q && byPhrase[hits.find(p => p.phrase.toLowerCase() === q)?.phrase];
    const groups = {};
    if (exact) groups["Exact match"] = [exact.phrase];
    data.phrases.filter(p => (!q || match(p)) && p !== exact).sort((a, b) => byRank(q)(a.phrase, b.phrase))
      .forEach(p => (groups[p.group] ??= []).push(p.phrase));
    const extra = q ? more.filter(p => match(p) && p !== exact).sort((a, b) => byRank(q)(a.phrase, b.phrase)) : [];
    if (extra.length) groups[MORE] = extra.slice(0, MORE_MAX).map(p => p.phrase);
    Object.entries(groups).forEach(([g, ps]) => {
      const og = document.createElement("optgroup"); og.label = g;
      ps.forEach(p => { const o = node("option", null, p); o.value = p; og.appendChild(o); });
      phraseSel.appendChild(og);
    });
    if (extra.length > MORE_MAX)
      phraseSel.lastChild.appendChild(node("option", null, `…${extra.length - MORE_MAX} more; keep typing`)).disabled = true;
    if (!phraseSel.options.length) phraseSel.appendChild(node("option", null, "No matching phrase")).disabled = true;
    else if (q) phraseSel.selectedIndex = 0;  // the best match, not whatever was chosen before
    else if ([...phraseSel.options].some(o => o.value === keep)) phraseSel.value = keep;
    showSuggestions(q ? hits.map(p => p.phrase).sort(byRank(q)).slice(0, 8) : [], q);
  }
  fillOptions("");
  search?.addEventListener("input", () => {
    fillOptions(search.value.trim().toLowerCase());
    if (phraseSel.value && byPhrase[phraseSel.value]) render();
  });

  const SVG = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, text) => {
    const n = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text != null) n.textContent = text;
    return n;
  };
  let mode = "abs";  // "abs": share of talks and of posters (two lines); "rel": odds of a talk with vs without (one line)
  const modeSeg = document.getElementById("drv-mode");
  modeSeg?.addEventListener("click", ev => {
    const b = ev.target.closest("button");
    if (!b) return;
    mode = b.dataset.mode;
    modeSeg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    if (byPhrase[phraseSel.value]) drawChart(byPhrase[phraseSel.value]);
  });

  function drawChart(p) {
    const W = 360, H = 150, L = 40, R = 18, T = 12, B = 24;
    const shares = [0, 1].map(i => p.counts.map((c, y) => c[i] / data.totals[y][i]));
    // in "rel" mode a year with no poster mentions has no odds ratio and is skipped
    const ci = p.counts.map(([a, c], y) => {
      const [nt, np] = data.totals[y];
      return c ? oddsRatio(a, c, nt - a, np - c) : null;
    });
    const series = mode === "abs" ? [["talk", shares[0]], ["poster", shares[1]]] : [["ratio", ci.map(r => r && r[0])]];
    const vals = series.flatMap(([, v]) => v).filter(v => v != null);
    // intervals set the scale, but an extreme one is cut off rather than flattening the line
    const his = mode === "rel" ? ci.filter(Boolean).map(r => Math.min(r[2], 3 * Math.max(...vals, 1))) : [];
    const top = Math.max(...vals, ...his, mode === "rel" ? 1 : 0) * 1.1 || 0.01;
    const steps = mode === "abs" ? [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] : [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10];
    const step = steps.find(s => top / s <= 4) ?? steps[steps.length - 1] * 2;
    const tick = v => mode === "abs" ? (100 * v).toFixed(step < 0.01 ? 1 : 0) + "%" : +v.toFixed(2) + "×";
    const x = i => L + (W - L - R) * (data.years.length > 1 ? i / (data.years.length - 1) : 0.5);
    const y = v => T + (H - T - B) * (1 - v / top);
    chart.replaceChildren();
    for (let v = 0; v <= top + 1e-9; v += step) {
      chart.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }));
      chart.appendChild(svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "axis" }, tick(v)));
    }
    if (mode === "rel") {
      chart.appendChild(svg("line", { x1: L, x2: W - R, y1: y(1), y2: y(1), class: "ref" }));
      ci.forEach((r, i) => r && chart.appendChild(svg("line", { x1: x(i), x2: x(i), y1: y(r[1]), y2: y(Math.min(r[2], top)), class: "ci" })));
    }
    data.years.forEach((yr, i) => chart.appendChild(svg("text", { x: x(i), y: H - 6, "text-anchor": "middle", class: "axis" }, yr)));
    series.forEach(([k, v]) => {
      const pts = v.map((s, i) => [i, s]).filter(([, s]) => s != null);
      chart.appendChild(svg("polyline", { points: pts.map(([i, s]) => `${x(i)},${y(s)}`).join(" "), class: "line " + k }));
      pts.forEach(([i, s]) => {
        const dot = svg("circle", { cx: x(i), cy: y(s), r: 4, class: "dot " + k });
        const [t, q] = p.counts[i];
        dot.appendChild(svg("title", {}, k === "ratio"
          ? `${data.years[i]}: ${s.toFixed(2)}× odds of a talk (95% CI ${ci[i][1].toFixed(2)}–${ci[i][2].toFixed(2)}; ` +
            `${pct(shares[0][i])} of talks vs ${pct(shares[1][i])} of posters)`
          : `${data.years[i]}: ${pct(s)} of ${k}s (${k === "talk" ? t : q})`));
        chart.appendChild(dot);
      });
    });
    document.getElementById("drv-legend").hidden = mode !== "abs";
    document.getElementById("drv-ratio-note").hidden = mode !== "rel";
    chart.setAttribute("aria-label", mode === "abs"
      ? `Share of talks and of posters mentioning “${p.phrase}”, by year`
      : `Odds of a talk with vs without “${p.phrase}”, by year`);
    const [or, lo, hi] = p.or, f = v => v.toFixed(2);
    const or_ = document.getElementById("drv-or");
    or_.replaceChildren(node("strong", null, `${f(or)}×`),
      ` odds of a talk with vs without “${p.phrase}”, all years (95% CI ${f(lo)}–${f(hi)}; combined across years).`);
  }
  ["all", ...data.years.map(String)].forEach(y => {
    const b = node("button", null, y === "all" ? "All years" : y);
    b.type = "button"; b.setAttribute("aria-pressed", String(y === year));
    b.addEventListener("click", () => { year = y; yearSeg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); render(); });
    yearSeg.appendChild(b);
  });
  phraseSel.addEventListener("change", render);

  // "abs": bars count abstracts mentioning the phrase, split talk/poster; "rel": odds of a talk for each
  // entity's abstracts with vs without the phrase, like the phrase's own odds ratio above
  // row: [name, talks, posters, entity abstracts, entity talks] for the period
  const entityOR = ([, a, b, n, t]) => oddsRatio(a, b, t - a, n - t - b);
  const fx = v => v.toFixed(v < 1 ? 2 : 1);
  let barMode = "abs";
  const barSeg = document.getElementById("drv-bar-mode");
  barSeg?.addEventListener("click", ev => {
    const b = ev.target.closest("button");
    if (!b) return;
    barMode = b.dataset.mode;
    barSeg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    render();
  });

  function showTip(ev, row, phrase, base) {
    const [name, talks, posters, total] = row, n = talks + posters;
    tip.replaceChildren();
    tip.appendChild(node("strong", null, `${n} abstract${n === 1 ? "" : "s"}`));
    tip.appendChild(node("div", null, `${name} · mention “${phrase}”`));
    [["talk", talks, "talks"], ["poster", posters, "posters"]].forEach(([k, v, label]) => {
      const d = node("div"); const key = node("span", "k"); key.style.background = `var(--${k})`;
      d.append(key, `${v} ${label}`); tip.appendChild(d);
    });
    tip.appendChild(node("div", null, `${pct(n / total)} of its ${total} abstracts${year === "all" ? "" : " in " + year}`));
    const [lo, hi] = wilson(n, total).map(v => v / base);
    tip.appendChild(node("div", null, `${(n / total / base).toFixed(1)}× the rate over all abstracts (${pct(base)}); ` +
      `95% CI ${lo.toFixed(1)}–${hi.toFixed(1)}×`));
    const [or, orLo, orHi] = entityOR(row);
    tip.appendChild(node("div", null, `${fx(or)}× odds of a talk with vs without it (95% CI ${fx(orLo)}–${fx(orHi)}); ` +
      `${row[4] - talks} talks of ${total - n} abstracts without it`));
    tip.hidden = false;
    const r = ev.type === "focus" ? ev.target.getBoundingClientRect() : { left: ev.clientX, bottom: ev.clientY };
    tip.style.left = Math.min(r.left + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(r.bottom + 12, innerHeight - tip.offsetHeight - 8) + "px";
  }

  function list(container, rows, phrase, base) {
    container.replaceChildren();
    if (!rows.length) { container.appendChild(node("p", "empty", "None with a talk mentioning it.")); return; }
    const ratio = r => entityOR(r)[0], ci = r => entityOR(r).slice(1);
    const rel = barMode === "rel";
    if (rel) rows = [...rows].sort((a, b) => ratio(b) - ratio(a));
    const max = Math.max(...rows.map(r => r[1] + r[2]));
    // "rel" is a forest plot on a log scale spanning every interval and 1×, but an interval past 1/50× or
    // 50× (a handful of abstracts) is cut off rather than squashing the rest
    const lo0 = Math.max(0.02, Math.min(1, ...rows.map(r => ci(r)[0]))) / 1.15;
    const hi0 = Math.min(50, Math.max(1, ...rows.map(r => ci(r)[1]))) * 1.15;
    const pos = v => 100 * Math.min(1, Math.max(0, Math.log(v / lo0) / Math.log(hi0 / lo0))) + "%";
    rows.forEach(row => {
      const [name, talks, posters] = row, n = talks + posters;
      const r = node("div", "bar-row"); r.tabIndex = 0;
      r.appendChild(node("span", "name", name));
      const cell = node("span", "cell");
      if (rel) {
        const [lo, hi] = ci(row), ref = node("span", "ref"), line = node("span", "ci"), dot = node("span", "pt");
        ref.style.left = pos(1);
        line.style.left = pos(lo); line.style.width = `calc(${pos(hi)} - ${pos(lo)})`;
        dot.style.left = pos(ratio(row));
        cell.append(ref, line, dot);
      } else {
        const track = node("span", "track");
        track.style.width = (100 * n / max) + "%";
        [["seg-talk", talks], ["seg-poster", posters]].forEach(([cls, v]) => {
          if (v) { const seg = node("span", cls); seg.style.flex = String(v); track.appendChild(seg); }
        });
        cell.appendChild(track);
      }
      r.appendChild(cell);
      r.appendChild(node("span", "val", rel ? `${fx(ratio(row))}× (${ci(row).map(fx).join("–")})`
        : `${n} (${talks} talk${talks === 1 ? "" : "s"})`));
      r.addEventListener("pointermove", ev => showTip(ev, row, phrase, base));
      r.addEventListener("focus", ev => showTip(ev, row, phrase, base));
      ["pointerleave", "blur"].forEach(t => r.addEventListener(t, () => { tip.hidden = true; }));
      container.appendChild(r);
    });
    if (!rel) return;
    // tick axis under the forest plot, aligned with the plot column
    const axis = node("div", "bar-row axis"), cell = node("span", "cell");
    // candidates in priority order (1×, then decades, then 2/5 steps); one is skipped if it would sit within
    // 16% of the plot width of a tick already placed, so labels never overlap
    const at = v => parseFloat(pos(v)), kept = [];
    [1, 10, 0.1, 2, 5, 0.5, 0.2, 20, 50, 0.05, 0.02].filter(v => v >= lo0 && v <= hi0)
      .forEach(v => { if (kept.every(k => Math.abs(at(k) - at(v)) >= 16)) kept.push(v); });
    kept.forEach(v => {
      const t = node("span", "tick", v + "×"); t.style.left = pos(v); cell.appendChild(t);
    });
    axis.append(node("span"), cell, node("span"));
    container.appendChild(axis);
  }

  function render() {
    const phrase = phraseSel.value, p = byPhrase[phrase];
    if (!p) return;
    drawChart(p);
    const who = document.getElementById("drv-who"), none = document.getElementById("drv-none");
    who.hidden = !data.drivers[phrase];
    none.hidden = !who.hidden;
    if (who.hidden) return;
    const d = data.drivers[phrase][year], n = d.talks + d.posters;
    document.getElementById("drv-summary").textContent =
      `${n} abstracts mention “${phrase}” ${year === "all" ? "across all years" : "in " + year} ` +
      `(${d.talks} talks, ${d.posters} posters); ${d.unmatched} have no matched first-author institution.`;
    // mention rate over every complete abstract in the period: the 1× of the ratio bars
    const totals = year === "all" ? data.totals : [data.totals[data.years.indexOf(+year)]];
    const base = n / totals.reduce((s, [t, q]) => s + t + q, 0);
    document.getElementById("drv-bar-legend").hidden = barMode !== "abs";
    document.getElementById("drv-bar-note").hidden = barMode !== "rel";
    list(document.getElementById("drv-institutes"), d.institutes, phrase, base);
    list(document.getElementById("drv-metros"), d.metros, phrase, base);
  }
  // Deep links: phrases.html?phrase=X#explorer opens with X selected (any case; a phrase outside the list goes in
  // the search box). On this page such links select the phrase in place rather than reloading.
  function select(want) {
    const key = Object.keys(byPhrase).find(k => k.toLowerCase() === want.toLowerCase());
    const listed = key && data.phrases.some(p => p.phrase === key);
    search.value = listed ? "" : (key ?? want);
    fillOptions(search.value.trim().toLowerCase());
    if (key) phraseSel.value = key;
    render();
  }
  const wanted = new URLSearchParams(location.search).get("phrase");
  if (wanted) select(wanted);
  document.addEventListener("click", ev => {
    const a = ev.target.closest("a[href*='?phrase=']");
    if (!a || ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button) return;
    const url = new URL(a.href);
    if (url.pathname !== location.pathname) return;
    ev.preventDefault();
    select(url.searchParams.get("phrase"));
    history.replaceState(null, "", url.search + "#explorer");
    document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
  });
  document.addEventListener("lookup-phrase", ev => {  // from the word Manhattan on the overview page
    select(ev.detail);
    document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
  });
  render();
  }
})();

// Center-of-mass map (geography page): a zoomed basemap from center.py and the yearly centers for the chosen
// map, group and host-city setting. With host-city first authors removed, rings mark the full-meeting centers.
(() => {
  const el = document.getElementById("com-data");
  if (!el) return;
  const data = JSON.parse(el.textContent);
  const root = document.getElementById("com"), map = document.getElementById("com-map");
  const tip = document.getElementById("com-tip"), tbody = document.querySelector("#com-table tbody");
  root.hidden = false;
  const state = { map: "na", group: "abstracts", host: root.dataset.host || "with", na: "with" };
  const SHORT = { "Los Angeles": "LA", "Washington, DC": "DC" };
  const NAMES = { na: "North America (US and Canada)", world: "Worldwide", world_no_na: "Worldwide without North America" };
  const SVG = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text != null) n.textContent = text;
    return n;
  };
  const node = (tag, text) => { const n = document.createElement(tag); if (text != null) n.textContent = text; return n; };
  const num = v => v.toLocaleString("en-US");
  const deg = (v, pos, neg) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? pos : neg}`;
  const host = y => data.hosts[y];

  root.querySelectorAll("[data-com]").forEach(seg => seg.addEventListener("click", ev => {
    const b = ev.target.closest("button");
    if (!b || b.disabled) return;
    state[seg.dataset.com] = b.dataset.value;
    render();
  }));

  let drawn = null, layer = null;
  function basemap(key, v) {
    map.setAttribute("viewBox", `0 0 ${v.width} ${v.height}`);
    map.setAttribute("aria-label", `Map: ${NAMES[key]}, yearly centers of mass of first authors`);
    const parts = [svg("rect", { width: v.width, height: v.height, class: "water" }),
                   svg("path", { d: v.land, class: v.borders == null ? "land outlined" : "land" }),
                   svg("path", { d: v.lakes, class: "water" })];
    if (v.borders) parts.push(svg("path", { d: v.borders, class: "border" }));
    v.places.forEach(([name, x, y]) => parts.push(svg("circle", { cx: x, cy: y, r: 2.5, class: "place" }),
      svg("text", { x: x + 6, y: y + 4, class: "place" }, name)));
    v.water.forEach(([name, x, y]) => parts.push(svg("text", { x, y, class: "sea", "text-anchor": "middle" }, name)));
    layer = svg("g", { class: "overlay" });
    map.replaceChildren(...parts, layer);
    drawn = key;
  }

  // year labels: the first spot clear of every point, place name and earlier label, else the least crowded
  const SPOTS = [[10, -8, "start"], [-10, -8, "end"], [10, 15, "start"], [-10, 15, "end"], [12, 4, "start"], [-12, 4, "end"]];
  function placeLabels(rows, v, extra) {
    const w = v.width, h = v.height;
    const boxes = [...rows, ...extra].map(r => [r.x - 7, r.y - 7, r.x + 7, r.y + 7])
      .concat(v.places.map(([name, x, y]) => [x - 3, y - 7, x + 6 + 6.2 * name.length, y + 6]));
    const hit = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
    return rows.map(r => {
      const width = 7.2 * r.label.length;
      const box = ([dx, dy, anchor]) => {
        const x0 = anchor === "start" ? r.x + dx : r.x + dx - width;
        return [x0, r.y + dy - 11, x0 + width, r.y + dy + 3];
      };
      const clashes = s => { const b = box(s); return (b[0] > 2 && b[2] < w - 2 && b[1] > 2 && b[3] < h - 2 ? 0 : 99) + boxes.filter(o => hit(b, o)).length; };
      const spot = SPOTS.reduce((best, s) => (clashes(s) < clashes(best) ? s : best));
      boxes.push(box(spot));
      return spot;
    });
  }

  function showTip(ev, r, key) {
    const talks = state.group === "talks", h = host(r.year);
    tip.replaceChildren(node("strong", `${r.year} · ${h}`),
      node("div", `Center ${deg(r.lat, "N", "S")}, ${deg(r.lon, "E", "W")}`),
      node("div", `${num(r.n)} ${talks ? "talks" : "abstracts"}${key === "na" ? " at US or Canadian metros" : ""}`),
      node("div", `${num(r.host_km)} km from ${h}`));
    if (key !== "world_no_na")
      tip.appendChild(node("div", state.host === "with" ? `Host-city first authors move it ${num(r.pull_km)} km`
        : `${num(r.pull_km)} km from the center with them`));
    tip.appendChild(node("div", `95% range about ±${num(Math.round(1.96 * r.sd_e))} km east–west, ` +
      `±${num(Math.round(1.96 * r.sd_n))} km north–south (bootstrap)`));
    tip.hidden = false;
    const b = ev.type === "focus" ? ev.target.getBoundingClientRect() : { left: ev.clientX, bottom: ev.clientY };
    tip.style.left = Math.min(b.left + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(b.bottom + 12, innerHeight - tip.offsetHeight - 8) + "px";
  }

  function render() {
    const key = state.map === "world" && state.na === "without" ? "world_no_na" : state.map;
    const noHosts = key === "world_no_na";  // every host so far is North American: removing them changes nothing
    root.querySelectorAll("[data-com]").forEach(seg => seg.querySelectorAll("button").forEach(b => {
      b.setAttribute("aria-pressed", String(state[seg.dataset.com] === b.dataset.value));
      b.disabled = seg.dataset.com === "host" && noHosts;
    }));
    document.getElementById("com-na-ctl").hidden = state.map !== "world";
    const v = data.views[key], removed = state.host === "without" && !noHosts;
    const rows = v.variants[`${state.group}|${removed ? "without" : "with"}`]
      .map(r => ({ ...r, label: `${r.year} · ${SHORT[host(r.year)] ?? host(r.year)}` }));
    const ghosts = removed ? v.variants[`${state.group}|with`] : [];
    if (drawn !== key) basemap(key, v);
    map.dataset.group = root.dataset.group = state.group;

    // points persist across toggles within a map, so a change of setting glides them to their new places
    const old = Object.fromEntries([...layer.querySelectorAll("g.pt")].map(g => [g.dataset.year, g]));
    layer.querySelectorAll(":scope > :not(g.pt)").forEach(n => n.remove());
    const under = svg("g");
    ghosts.forEach((g, i) => {
      under.appendChild(svg("line", { x1: g.x, y1: g.y, x2: rows[i].x, y2: rows[i].y, class: "pull" }));
      under.appendChild(svg("circle", { cx: g.x, cy: g.y, r: 5.5, class: "ghost" }));
    });
    if (!removed)  // with the pull lines drawn, the path between years is one line too many
      under.appendChild(svg("polyline", { points: rows.map(r => `${r.x},${r.y}`).join(" "), class: "path" }));
    layer.prepend(under);
    const spots = placeLabels(rows, v, ghosts);
    rows.forEach((r, i) => {
      let g = old[r.year];
      if (!g) {
        g = svg("g", { class: "pt", tabindex: 0, "data-year": r.year });
        g.append(svg("circle", { r: 14, class: "hit" }), svg("circle", { r: 6, class: "dot" }), svg("text", { class: "yr" }));
        ["pointerleave", "blur"].forEach(t => g.addEventListener(t, () => { tip.hidden = true; }));
        layer.appendChild(g);
      }
      g.onpointermove = g.onfocus = ev => showTip(ev, r, key);
      g.style.transform = `translate(${r.x}px, ${r.y}px)`;
      const [dx, dy, anchor] = spots[i], t = g.querySelector("text");
      t.textContent = r.label;
      Object.entries({ x: dx, y: dy, "text-anchor": anchor }).forEach(([k, val]) => t.setAttribute(k, val));
      g.setAttribute("aria-label", `${r.label}: ${deg(r.lat, "N", "S")}, ${deg(r.lon, "E", "W")}`);
    });
    document.getElementById("com-ghost-key").hidden = !removed;
    document.getElementById("com-dot-key").textContent = removed ? "center of mass, each year"
      : "center of mass, each year (line joins the years in order)";

    const talks = state.group === "talks", ns = rows.map(r => r.n);
    const span = `${num(Math.min(...ns))}–${num(Math.max(...ns))} ${talks ? "talks" : "abstracts"} a year`;
    const what = { na: "First authors at US and Canadian metro areas", world: "First authors worldwide",
                   world_no_na: "Abstracts with no first author in the US or Canada" }[key];
    const notes = [`${NAMES[key]}. ${what}, ${span}.`];
    if (removed) notes.push("Abstracts with a first author in that year's host metro are left out; rings show the centers with them.");
    if (noHosts) notes.push("Every host city so far is in North America, so removing host-city first authors changes nothing here.");
    if (talks) notes.push("Talks are few, so their centers move a lot by chance: hover a point for its 95% range.");
    document.getElementById("com-note").textContent = notes.join(" ");

    document.getElementById("com-n-head").textContent = talks ? "Talks" : "Abstracts";
    tbody.replaceChildren(...rows.map(r => {
      const tr = document.createElement("tr");
      [[r.year, "text"], [host(r.year), "text"], [`${deg(r.lat, "N", "S")}, ${deg(r.lon, "E", "W")}`, "text"],
       [num(r.n)], [`${num(r.host_km)} km`], [noHosts ? "—" : `${num(r.pull_km)} km`]].forEach(([val, cls]) => {
        const td = node("td", val); if (cls) td.className = cls; tr.appendChild(td);
      });
      return tr;
    }));
  }
  render();
})();

// Overrepresented-phrase score: tokens and 1-3-word n-grams as in features.py (tokenize, ngrams), so a text scores
// here as it would in talk_score.py. Top-level so it can be checked against the Python outside a browser.
const talkScore = (() => {
  const TOKEN = /\p{Nd}+(?:[.,]\p{Nd}+)*(?![\p{L}\p{N}_-])|[\p{L}\p{N}_]+(?:[-'’][\p{L}\p{N}_]+)*/gu;
  const NUM = /^[\p{Nd}.,]+$/u;
  const tokenize = text => (text.toLowerCase().match(TOKEN) || []).map(t => NUM.test(t) ? "<num>" : t.replace(/’/g, "'"));
  const grams = tokens => {
    const out = new Set();
    for (let n = 1; n <= 3; n++) for (let i = 0; i + n <= tokens.length; i++) out.add(tokens.slice(i, i + n).join(" "));
    return out;
  };
  // groups: [label, [other variants], or, talks, posters]; returns the groups the text contains, with the variant found
  const match = (text, groups) => {
    const g = grams(tokenize(text));
    return groups.map(([label, also, or]) => ({ label, or, found: [label, ...also].find(v => g.has(v)) })).filter(m => m.found);
  };
  // whole-number percentile among abstracts (counts per score), ties counted half
  const percentile = (s, talks, posters) => {
    const counts = talks.map((t, i) => t + posters[i]), n = counts.reduce((a, b) => a + b, 0);
    const below = counts.slice(0, s).reduce((a, b) => a + b, 0);
    return Math.round(100 * (below + 0.5 * (counts[s] || 0)) / n);
  };
  return { tokenize, match, percentile };
})();

// Score page: the pasted or opened text is scored here and never leaves the browser.
(() => {
  const el = typeof document !== "undefined" && document.getElementById("sc-data");
  if (!el) return;
  const data = JSON.parse(el.textContent);
  const text = document.getElementById("sc-text"), file = document.getElementById("sc-file");
  const result = document.getElementById("sc-result"), chart = document.getElementById("sc-chart");
  const SVG = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs, label) => {
    const n = document.createElementNS(SVG, tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (label != null) n.textContent = label;
    return n;
  };
  const nT = data.talks.reduce((a, b) => a + b, 0), nP = data.posters.reduce((a, b) => a + b, 0);
  const pct = v => (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%";

  function draw(s) {
    // histogram: share of talks and of posters at each score, side by side; your score marked
    const W = 560, H = 190, L = 40, R = 14, T = 16, B = 30, max = data.talks.length - 1;
    const shares = [data.talks.map(c => c / nT), data.posters.map(c => c / nP)];
    const top = Math.max(...shares.flat()) * 1.12, step = top > 0.1 ? 0.05 : 0.02;
    const slot = (W - L - R) / (max + 1), bar = Math.max(1, slot / 2 - 1);
    const x = i => L + slot * (Math.min(i, max) + 0.5), y = v => T + (H - T - B) * (1 - v / top);
    chart.replaceChildren();
    for (let v = 0; v <= top + 1e-9; v += step) {
      chart.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }));
      chart.appendChild(svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "axis" }, Math.round(100 * v) + "%"));
    }
    for (let v = 0; v <= max; v += 5)
      chart.appendChild(svg("text", { x: x(v), y: H - 14, "text-anchor": "middle", class: "axis" }, v));
    chart.appendChild(svg("text", { x: (L + W - R) / 2, y: H - 1, "text-anchor": "middle", class: "axis" },
      "overrepresented phrases found"));
    for (let i = 0; i <= max; i++) {
      [["talk", shares[0][i], -bar - 0.5], ["poster", shares[1][i], 0.5]].forEach(([k, v, dx]) => v &&
        chart.appendChild(svg("rect", { x: x(i) + dx, y: y(v), width: bar, height: y(0) - y(v), class: "bar " + k })));
    }
    if (s != null) {
      chart.appendChild(svg("line", { x1: x(s), x2: x(s), y1: T - 4, y2: y(0), class: "you" }));
      const right = s > max * 0.7;
      chart.appendChild(svg("text", { x: x(s) + (right ? -5 : 5), y: T + 6, class: "you-label",
        "text-anchor": right ? "end" : "start" }, s > max ? `your text (${s})` : "your text"));
    }
    // a hover target per score: share of talks and of posters with that many
    for (let i = 0; i <= max; i++) {
      const hit = svg("rect", { x: x(i) - slot / 2, y: T, width: slot, height: y(0) - T, class: "hit" });
      hit.appendChild(svg("title", {}, `${i} phrase${i === 1 ? "" : "s"}: ${pct(shares[0][i])} of talks (${data.talks[i]}), ` +
        `${pct(shares[1][i])} of posters (${data.posters[i]})`));
      chart.appendChild(hit);
    }
  }

  function render() {
    const t = text.value.trim();
    const words = t ? talkScore.tokenize(t).length : 0;
    result.hidden = !t;
    draw(t ? talkScore.match(t, data.groups).length : null);
    if (!t) return;
    const found = talkScore.match(t, data.groups), s = found.length;
    const p = Math.min(99, Math.max(1, talkScore.percentile(s, data.talks, data.posters)));
    document.getElementById("sc-score").textContent = s;
    document.getElementById("sc-pct").textContent = (p === 99 ? "over 99" : p === 1 ? "under 1" : p) + "%";
    document.getElementById("sc-short").hidden = words >= 150;
    document.getElementById("sc-words").textContent = words;
    const list = document.getElementById("sc-found");
    list.replaceChildren(...found.sort((a, b) => b.or - a.or).map(m => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = m.found.replace(/<num>/g, "N");
      chip.title = `${m.label.replace(/<num>/g, "N")}: odds ratio ${m.or.toFixed(2)} (talks vs posters, all years)`;
      return chip;
    }));
    document.getElementById("sc-none").hidden = s > 0;
  }
  text.addEventListener("input", render);
  file.addEventListener("change", () => {
    const f = file.files[0];
    if (!f) return;
    f.text().then(t => { text.value = t; render(); });
  });
  document.getElementById("sc-clear").addEventListener("click", () => { text.value = ""; file.value = ""; render(); text.focus(); });
  render();
})();

// Words more common in the latest year (phrases page): search every word, sort by any column, show the top rows.
(() => {
  const el = typeof document !== "undefined" && document.getElementById("we-data");
  if (!el) return;
  // The overview page fetches the data (data-src); the phrases page inlines it.
  if (el.dataset.src) fetch(el.dataset.src).then(r => r.json()).then(start).catch(() => {});
  else start(JSON.parse(el.textContent));
  function start(data) {
  const SHOW = +el.dataset.show || 50;
  const search = document.getElementById("we-search"), body = document.getElementById("we-body");
  const count = document.getElementById("we-count"), heads = [...document.querySelectorAll("#we-table th")];
  const sig = data.rows.filter(r => r[4] < data.alpha).length;
  let key = 4, asc = true;
  // abstracts columns sort by share, so the two periods compare
  const value = (r, k) => k === 1 ? r[1] / data.n1 : k === 2 ? r[2] / data.n0 : r[k];
  const share = (k, n) => `${k.toLocaleString()} (${(100 * k / n).toFixed(k / n < 0.1 ? 1 : 0)}%)`;
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const fmtP = p => {
    if (p === 0) return "< 10⁻³⁰⁰";
    if (p >= 0.001) return String(+p.toPrecision(2));
    const [m, e] = p.toExponential(1).split("e");
    return `${m} × 10${[...String(+e)].map(c => SUP[c]).join("")}`;
  };
  const cell = (text, cls) => { const td = document.createElement("td"); if (cls) td.className = cls; td.textContent = text; return td; };

  function render() {
    const q = search.value.trim().toLowerCase();
    const rows = data.rows.filter(r => !q || r[0].includes(q))
      .sort((a, b) => { const x = value(a, key), y = value(b, key); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); });
    body.replaceChildren(...rows.slice(0, SHOW).map(r => {
      const tr = document.createElement("tr"), p = cell(fmtP(r[4]));
      if (r[4] < data.alpha) p.style.fontWeight = "700";
      tr.append(cell(r[0], "text"), cell(share(r[1], data.n1)), cell(share(r[2], data.n0)), cell(r[3].toFixed(2)), p);
      return tr;
    }));
    count.textContent = (q ? `${rows.length.toLocaleString()} matching word${rows.length === 1 ? "" : "s"}`
      : `${data.rows.length.toLocaleString()} words, ${sig} significant`) + (rows.length > SHOW ? ` · showing ${SHOW}` : "");
  }
  heads.forEach(th => th.addEventListener("click", () => {
    const k = +th.dataset.k;
    asc = k === key ? !asc : k === 0 || k === 4;  // words A to Z and p smallest first; the rest largest first
    key = k;
    heads.forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", asc ? "ascending" : "descending");
    render();
  }));
  search.addEventListener("input", render);
  render();
  }
})();

// Interactive Manhattan plots (phrases page): words A to Z per year, words overrepresented in the latest year, and
// genes along the genome. Points on a canvas; hover (or tap) for the word or gene; the lead point of each tower (no
// taller point within a few pixels) is labelled, biggest first, where its label fits. The per-year charts follow
// the page's year buttons. Without the data (or JS) the static PNG inside the holder stays.
(() => {
  const holders = typeof document !== "undefined" ? [...document.querySelectorAll("[data-manhattan]")] : [];
  if (!holders.length) return;
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  const INK = color("--text-primary"), INK_2 = color("--text-secondary"), RULE = color("--rule"), TALK = color("--talk");
  const LIGHT = "#9fc0ea", FONT = getComputedStyle(document.body).fontFamily;
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const pct = v => (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%";
  const count = (k, n) => `${k.toLocaleString()} of ${n.toLocaleString()} abstracts (${pct(k / n)})`;
  const yearNow = () => document.querySelector('[data-control="year"] button[aria-pressed="true"]')?.dataset.value ?? "all";
  const charts = [];

  function model(kind, data) {
    const w = data.words, g = data.genes, n = data.n;
    if (kind === "genes") {
      const ymax = Math.max(...Object.entries(g.counts).map(([y, c]) => Math.max(...c) / n[y])) * 1.25;
      return {
        xs: g.x, band: g.chrom.map(c => c % 2), x0: -20, x1: g.chroms[g.chroms.length - 1][2] + 20, ymax, fmt: pct,
        ticks: g.chroms.filter(c => c[3]).map(c => [(c[1] + c[2]) / 2, c[0]]), labelMin: 0.2, follows: true,
        leadPx: 2,  // neighbouring genes are separate signals (TP53 and BRCA1 sit 8px apart), not one tower
        ys: y => g.counts[y].map(c => c / n[y]), skipZero: true,
        title: y => y === "all" ? `All years, ${n.all.toLocaleString()} abstracts` : `${y}, ${n[y].toLocaleString()} abstracts`,
        tip: (i, y) => [g.genes[i], `chromosome ${g.chroms[g.chrom[i]][0]}`, count(g.counts[y][i], n[y])],
      };
    }
    const letters = w.words.map(s => s[0]), firsts = [...new Set(letters)];
    const band = letters.map(l => firsts.indexOf(l) % 2);
    const ticks = firsts.map(l => { const a = letters.indexOf(l), b = letters.lastIndexOf(l); return [(a + b) / 2, b - a > w.words.length / 120 ? l : ""]; });
    const base = { xs: w.words.map((_, i) => i), band, x0: -w.words.length * 0.01, x1: w.words.length * 1.01, ticks };
    if (kind === "words") {
      const ymax = Math.max(...Object.entries(w.counts).map(([y, c]) => Math.max(...c) / n[y])) * 1.15;
      return { ...base, ymax, fmt: pct, labelMin: 0.15, follows: true, ys: y => w.counts[y].map(c => c / n[y]),
        title: y => y === "all" ? `All years, ${n.all.toLocaleString()} abstracts` : `${y}, ${n[y].toLocaleString()} abstracts`,
        tip: (i, y) => [w.words[i], count(w.counts[y][i], n[y])] };
    }
    const e = w.enriched, last = String(e.year), nb = n.all - n[last];
    return { ...base, ymax: Math.max(...e.logp) * 1.1, fmt: v => String(v), line: e.line, labelMin: 0, minY: 4 * e.line,
      ys: () => e.logp, follows: false,
      tip: i => {
        const k = w.counts[last][i], before = w.counts.all[i] - k, lp = e.logp[i];
        return [w.words[i], `${last}: ${count(k, n[last])}`, `${e.before}: ${count(before, nb)}`,
          lp >= 1 ? `p = 10${[...String(-Math.round(lp))].map(c => SUP[c]).join("")}${lp >= e.line ? ", significant" : ""}`
            : "not overrepresented"];
      } };
  }

  function setup(holder, data) {
    const kind = holder.dataset.manhattan, m = model(kind, data);
    const canvas = document.createElement("canvas"), tip = document.createElement("div"), title = holder.querySelector(".mh-title");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", holder.dataset.label || "");
    tip.className = "tip"; tip.hidden = true; tip.setAttribute("role", "tooltip");
    holder.querySelector(".mh-fallback")?.remove();
    holder.append(canvas, tip);
    const ctx = canvas.getContext("2d");
    let pts = [], year = m.follows ? yearNow() : null, hover = -1;
    const L = 44, R = 12, T = 30, B = 26;

    function draw() {
      const W = holder.clientWidth, H = Math.round(Math.min(340, Math.max(240, W * 0.42))), dpr = devicePixelRatio || 1;
      canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.width = W + "px"; canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (title) title.textContent = m.title ? m.title(year) : "";
      const X = x => L + (W - L - R) * (x - m.x0) / (m.x1 - m.x0), Y = v => T + (H - T - B) * (1 - v / m.ymax);
      // y grid: about four round steps
      const raw = m.ymax / 4, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(s => s * mag).find(s => s >= raw);
      ctx.font = `11px ${FONT}`; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let v = 0; v <= m.ymax; v += step) {
        ctx.strokeStyle = RULE; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, Math.round(Y(v)) + 0.5); ctx.lineTo(W - R, Math.round(Y(v)) + 0.5); ctx.stroke();
        ctx.fillStyle = INK_2; ctx.fillText(m.fmt === pct ? `${+(100 * v).toFixed(2)}%` : m.fmt(+v.toFixed(6)), L - 6, Y(v));
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      m.ticks.forEach(([x, label]) => label && ctx.fillText(label, X(x), H - B + 8));
      if (m.line) {
        ctx.strokeStyle = INK_2; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(L, Y(m.line)); ctx.lineTo(W - R, Y(m.line)); ctx.stroke(); ctx.setLineDash([]);
        ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText("Bonferroni 5%", W - R, Y(m.line) - 2);
      }
      const ys = m.ys(year);
      pts = [];
      ys.forEach((v, i) => { if (!(m.skipZero && !v)) pts.push([X(m.xs[i]), Y(v), i]); });
      [0, 1].forEach(b => {
        ctx.fillStyle = b ? LIGHT : TALK;
        ctx.beginPath();
        pts.forEach(([x, y, i]) => { if (m.band[i] === b) { ctx.moveTo(x + 2, y); ctx.arc(x, y, 2, 0, 2 * Math.PI); } });
        ctx.fill();
      });
      // tower leads, tallest first: no taller point within leadPx (10) either side; then a label where it fits
      const order = pts.filter(([, , i]) => ys[i] >= Math.max(m.labelMin * m.ymax, m.minY ?? 0)).sort((a, b) => a[1] - b[1]);
      const taller = [], boxes = [], maxLabels = Math.floor((W - L - R) / 55);
      ctx.font = `bold 11px ${FONT}`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      for (const [x, y, i] of order) {
        const lead = !taller.some(tx => Math.abs(tx - x) < (m.leadPx ?? 10));
        taller.push(x);
        if (!lead || boxes.length >= maxLabels) continue;
        const text = kind === "genes" ? data.genes.genes[i] : data.words.words[i], w = ctx.measureText(text).width + 4;
        for (const [dx, dy] of [[0, 4], [-w / 2 - 4, 4], [w / 2 + 4, 4], [0, 16], [-w / 2 - 4, 16], [w / 2 + 4, 16], [0, 28]]) {
          const box = [x + dx - w / 2, y - dy - 13, x + dx + w / 2, y - dy];
          if (box[0] < 2 || box[2] > W - 2 || box[1] < 2) continue;
          if (boxes.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
          boxes.push(box);
          if (dx || dy > 4) { ctx.strokeStyle = INK_2; ctx.lineWidth = 0.75; ctx.beginPath(); ctx.moveTo(x, y - 2); ctx.lineTo(x + dx, y - dy); ctx.stroke(); }
          ctx.fillStyle = INK; ctx.fillText(text, x + dx, y - dy);
          break;
        }
      }
      if (hover >= 0) {
        const p = pts.find(q => q[2] === hover);
        if (p) { ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI); ctx.stroke(); }
      }
    }

    function nearest(ev) {
      const r = canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = -1, bd = 100;  // within 10px
      for (const [x, y, i] of pts) { const d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = i; } }
      return best;
    }
    function show(ev) {
      const i = nearest(ev);
      if (i !== hover) { hover = i; draw(); }
      if (i < 0) { tip.hidden = true; canvas.style.cursor = ""; return; }
      const [head, ...lines] = m.tip(i, year);
      tip.replaceChildren(Object.assign(document.createElement("strong"), { textContent: head }),
        ...lines.map(t => Object.assign(document.createElement("div"), { textContent: t })));
      tip.hidden = false;
      canvas.style.cursor = kind === "enriched" ? "pointer" : "";
      tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = Math.min(ev.clientY + 12, innerHeight - tip.offsetHeight - 8) + "px";
    }
    canvas.addEventListener("pointermove", show);
    canvas.addEventListener("pointerdown", show);
    canvas.addEventListener("pointerleave", () => { tip.hidden = true; if (hover >= 0) { hover = -1; draw(); } });
    if (kind === "enriched") canvas.addEventListener("click", ev => {  // fill the table's search with that word
      const i = nearest(ev), q = document.getElementById("we-search");
      if (i < 0) return;
      if (!q) { document.dispatchEvent(new CustomEvent("lookup-phrase", { detail: data.words.words[i] })); return; }
      q.value = data.words.words[i]; q.dispatchEvent(new Event("input")); q.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    new ResizeObserver(() => draw()).observe(holder);
    charts.push(() => { if (m.follows && yearNow() !== year) { year = yearNow(); hover = -1; tip.hidden = true; draw(); } });
  }

  const sync = () => charts.forEach(f => f());
  document.querySelector('[data-control="year"]')?.addEventListener("click", () => setTimeout(sync));
  addEventListener("hashchange", () => setTimeout(sync));
  fetch(holders[0].dataset.src).then(r => r.json()).then(data => holders.forEach(h => setup(h, data))).catch(() => {});
})();

// What's hot (overview): a sticky line chart of the yearly share of abstracts mentioning each topic. The step
// nearest the middle of the screen picks the topics to color (in its data-phrases order) and the y-axis eases to
// fit them; the other topics are hidden. Without JS the steps read as text and the table stays.
(() => {
  const root = document.querySelector("[data-story]");
  if (!root) return;
  const { years, series } = JSON.parse(root.querySelector("script[type='application/json']").textContent);
  const holder = root.querySelector(".story-chart"), steps = [...root.querySelectorAll(".story-step")];
  const NS = "http://www.w3.org/2000/svg", L = 42, T = 14, B = 28, GAP = 34;  // GAP: between two-line labels
  const make = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    return parent ? parent.appendChild(e) : e;
  };
  const pct = v => `${+(100 * v).toFixed(v < 0.1 ? 1 : 0)}%`;
  holder.hidden = false;
  root.classList.add("live");
  const svg = make("svg", { role: "img", "aria-label": holder.getAttribute("aria-label") }, holder);
  const clip = make("rect", {}, make("clipPath", { id: "story-clip" }, make("defs", {}, svg)));
  const grid = make("g", { class: "grid" }, svg), axis = make("g", { class: "axis" }, svg);
  const plot = make("g", { "clip-path": "url(#story-clip)" }, svg), labels = make("g", { class: "labels" }, svg);
  const cross = make("line", { class: "cross" }, svg);
  const tip = Object.assign(document.createElement("div"), { className: "tip", hidden: true });
  tip.setAttribute("role", "tooltip");
  holder.append(tip);
  const lines = Object.keys(series).map(p => {
    const g = make("g", { class: "topic" }, plot);
    return { p, g, path: make("path", {}, g), dots: years.map(() => make("circle", { r: 3.5 }, g)) };
  });

  let W = 0, H = 0, R = 128, rFrom = R, rTo = R, ymax = 0.15, from = ymax, to = ymax, t0 = 0, active = [], ticks = [], hoverYear = -1;
  const X = i => L + (W - L - R) * i / (years.length - 1), Y = v => T + (H - T - B) * (1 - v / ymax);
  const still = matchMedia("(prefers-reduced-motion: reduce)");

  function niceTicks(max) {  // about four round steps, and a top that clears the highest point
    const raw = max / 4, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw);
    const top = Math.ceil(max * 1.08 / step) * step;
    return { top, ticks: Array.from({ length: Math.round(top / step) + 1 }, (_, i) => +(i * step).toFixed(6)) };
  }

  function draw() {
    if (!W) return;  // not laid out yet; the ResizeObserver draws once it is
    clip.setAttribute("x", L - 6); clip.setAttribute("y", T - 6);
    clip.setAttribute("width", W - L - R + 12); clip.setAttribute("height", H - T - B + 6);
    grid.replaceChildren(...ticks.filter(v => v <= ymax * 1.001).map(v => make("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) })));
    axis.replaceChildren(
      ...ticks.filter(v => v <= ymax * 1.001).map(v => Object.assign(make("text", { x: L - 6, y: Y(v), dy: "0.32em", "text-anchor": "end" }), { textContent: pct(v) })),
      ...years.map((y, i) => Object.assign(make("text", { x: X(i), y: H - B + 18, "text-anchor": "middle" }), { textContent: y })));
    lines.forEach(({ p, path, dots }) => {
      const s = series[p].share;
      path.setAttribute("d", s.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(""));
      dots.forEach((c, i) => { c.setAttribute("cx", X(i)); c.setAttribute("cy", Y(s[i])); });
    });
    // direct labels at the right end, nudged apart so they don't overlap, kept inside the plot
    const placed = active.map((p, k) => ({ p, k, y: Y(series[p].share.at(-1)) })).sort((a, b) => a.y - b.y);
    placed.forEach((d, j) => { if (j) d.y = Math.max(d.y, placed[j - 1].y + GAP); });
    for (let j = placed.length - 1; j >= 0; j--)  // then back up from the bottom edge if the nudging ran past it
      placed[j].y = Math.min(placed[j].y, j === placed.length - 1 ? H - B - 6 : placed[j + 1].y - GAP);
    labels.replaceChildren(...placed.map(({ p, k, y }) => {
      const g = make("g", { class: `lab hl-${k + 1}`, transform: `translate(${W - R + 10},${y})` });
      Object.assign(make("text", { y: -2 }, g), { textContent: p });
      Object.assign(make("text", { y: 12, class: "val" }, g), { textContent: `${pct(series[p].share.at(-1))} in ${years.at(-1)}` });
      return g;
    }));
    if (hoverYear >= 0) { cross.setAttribute("x1", X(hoverYear)); cross.setAttribute("x2", X(hoverYear)); cross.setAttribute("y1", T); cross.setAttribute("y2", H - B); }
    cross.style.display = hoverYear >= 0 ? "" : "none";
  }

  function frame(now) {
    const k = still.matches ? 1 : Math.min(1, (now - t0) / 650), e = 1 - (1 - k) ** 3;  // ease-out cubic
    ymax = from + (to - from) * e; R = rFrom + (rTo - rFrom) * e;
    draw();
    if (k < 1) requestAnimationFrame(frame);
  }
  function show(step) {
    steps.forEach(s => s.classList.toggle("active", s === step));
    active = JSON.parse(step.dataset.phrases);
    lines.forEach(({ p, g }) => {
      const k = active.indexOf(p);
      g.setAttribute("class", k < 0 ? "topic" : `topic on hl-${k + 1}`);
      if (k >= 0) plot.appendChild(g);  // colored lines on top
    });
    const n = niceTicks(Math.max(...active.flatMap(p => series[p].share)));
    ticks = n.ticks; from = ymax; to = n.top; rFrom = R; rTo = labelRoom(); t0 = performance.now();
    requestAnimationFrame(frame);
  }

  function labelRoom() {  // right margin wide enough for the longest label of this step
    const probe = make("g", { class: "lab" }, labels), text = make("text", {}, probe);
    const widths = active.flatMap(p => [p, `${pct(series[p].share.at(-1))} in ${years.at(-1)}`])
      .map(s => { text.textContent = s; return text.getComputedTextLength(); });
    probe.remove();
    return Math.max(80, ...widths.map(w => w + 18));
  }
  function size() {
    W = holder.clientWidth; H = holder.clientHeight;
    R = rTo = labelRoom();
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    draw();
  }
  new ResizeObserver(size).observe(holder);

  // the step crossing the middle of the screen (below the sticky chart on narrow screens) is the active one
  const io = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && show(e.target)),
    { rootMargin: "-50% 0px -49% 0px" });
  steps.forEach(s => io.observe(s));
  show(steps[0]);

  // hover: crosshair on the nearest year, the colored topics' shares and counts in a tooltip
  svg.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left;
    const i = Math.max(0, Math.min(years.length - 1, Math.round((mx - L) / (W - L - R) * (years.length - 1))));
    if (mx < L - 20 || mx > W - R + 20) { svg.dispatchEvent(new Event("pointerleave")); return; }
    hoverYear = i; draw();
    tip.replaceChildren(Object.assign(document.createElement("strong"), { textContent: `ASHG ${years[i]}` }),
      ...active.map(p => Object.assign(document.createElement("div"), {
        textContent: `${p}: ${pct(series[p].share[i])} · ${series[p].docs[i]} abstracts, ${series[p].talks[i]} talks` })));
    tip.hidden = false;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 14, innerHeight - tip.offsetHeight - 8) + "px";
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; hoverYear = -1; draw(); });
})();

// The header is sticky; the page toolbar sticks just below it, wherever the header wraps to.
(() => {
  const header = document.querySelector(".site-header");
  if (!header) return;
  const set = () => document.documentElement.style.setProperty("--header-h", header.offsetHeight + "px");
  set();
  new ResizeObserver(set).observe(header);
})();
