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
  const data = JSON.parse(el.textContent);
  const phraseSel = document.getElementById("drv-phrase"), yearSeg = document.getElementById("drv-year");
  const search = document.getElementById("drv-search"), chart = document.getElementById("drv-chart");
  const tip = document.getElementById("drv-tip");
  let year = "all";
  const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const pct = v => (100 * v).toFixed(v < 0.1 ? 1 : 0) + "%";
  const x2 = v => v.toFixed(v < 10 ? 1 : 0) + "×";
  // 95% CIs. rateRatio: (a/n1) ÷ (c/n2), Katz log interval, +0.5 to a zero count.
  // wilson: proportion k/n, the interval the site uses for talk rates.
  const rateRatio = (a, n1, c, n2) => {
    const r = (a / n1) / (c / n2), [aa, cc] = [a || 0.5, c || 0.5];
    const se = Math.sqrt(1 / aa - 1 / n1 + 1 / cc - 1 / n2), lr = Math.log((aa / n1) / (cc / n2));
    return [r, a ? Math.exp(lr - 1.96 * se) : 0, Math.exp(lr + 1.96 * se)];
  };
  const wilson = (k, n, z = 1.96) => {
    const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  };
  // every other phrase in 5+ talks, searchable but without the institution/metro breakdown:
  // [phrase, [talks, posters] per year flattened, or, lo, hi], most-mentioned first
  const MORE = "More phrases (abstract text only)", MORE_MAX = 150;
  const more = (data.more || []).map(([phrase, flat, ...or]) =>
    ({ phrase, group: MORE, counts: data.years.map((_, i) => [flat[2 * i], flat[2 * i + 1]]), or }));
  const byPhrase = Object.fromEntries([...data.phrases, ...more].map(p => [p.phrase, p]));

  function fillOptions(q) {
    const keep = phraseSel.value;
    phraseSel.replaceChildren();
    const groups = {};
    data.phrases.filter(p => !q || p.phrase.toLowerCase().includes(q)).forEach(p => (groups[p.group] ??= []).push(p.phrase));
    const extra = q ? more.filter(p => p.phrase.toLowerCase().includes(q)) : [];
    if (extra.length) groups[MORE] = extra.slice(0, MORE_MAX).map(p => p.phrase);
    Object.entries(groups).forEach(([g, ps]) => {
      const og = document.createElement("optgroup"); og.label = g;
      ps.forEach(p => { const o = node("option", null, p); o.value = p; og.appendChild(o); });
      phraseSel.appendChild(og);
    });
    if (extra.length > MORE_MAX)
      phraseSel.lastChild.appendChild(node("option", null, `…${extra.length - MORE_MAX} more; keep typing`)).disabled = true;
    if (!phraseSel.options.length) phraseSel.appendChild(node("option", null, "No matching phrase")).disabled = true;
    else if ([...phraseSel.options].some(o => o.value === keep)) phraseSel.value = keep;
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
  let mode = "abs";  // "abs": share of talks and of posters (two lines); "rel": talk share ÷ poster share (one line)
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
    // in "rel" mode a year with no poster mentions has no ratio and is skipped
    const ci = p.counts.map(([a, c], y) => (c ? rateRatio(a, data.totals[y][0], c, data.totals[y][1]) : null));
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
          ? `${data.years[i]}: ${s.toFixed(2)}× (95% CI ${ci[i][1].toFixed(2)}–${ci[i][2].toFixed(2)}; ` +
            `${pct(shares[0][i])} of talks vs ${pct(shares[1][i])} of posters)`
          : `${data.years[i]}: ${pct(s)} of ${k}s (${k === "talk" ? t : q})`));
        chart.appendChild(dot);
      });
    });
    document.getElementById("drv-legend").hidden = mode !== "abs";
    document.getElementById("drv-ratio-note").hidden = mode !== "rel";
    chart.setAttribute("aria-label", mode === "abs"
      ? `Share of talks and of posters mentioning “${p.phrase}”, by year`
      : `Share of talks divided by share of posters mentioning “${p.phrase}”, by year`);
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

  // "abs": bars count abstracts mentioning the phrase, split talk/poster; "rel": each entity's mention rate
  // ÷ the mention rate over all abstracts in the period (1× = average)
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
    tip.hidden = false;
    const r = ev.type === "focus" ? ev.target.getBoundingClientRect() : { left: ev.clientX, bottom: ev.clientY };
    tip.style.left = Math.min(r.left + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(r.bottom + 12, innerHeight - tip.offsetHeight - 8) + "px";
  }

  function list(container, rows, phrase, base) {
    container.replaceChildren();
    if (!rows.length) { container.appendChild(node("p", "empty", "None with a talk mentioning it.")); return; }
    const ratio = r => (r[1] + r[2]) / r[3] / base;
    const ci = r => wilson(r[1] + r[2], r[3]).map(v => v / base);  // the overall rate is treated as fixed
    const rel = barMode === "rel";
    if (rel) rows = [...rows].sort((a, b) => ratio(b) - ratio(a));
    const max = Math.max(...rows.map(r => r[1] + r[2]));
    // "rel" is a forest plot on a log scale spanning every interval and 1×; rows have a talk mentioning
    // the phrase, so every lower bound is above 0
    const lo0 = Math.min(1, ...rows.map(r => ci(r)[0])) / 1.15, hi0 = Math.max(1, ...rows.map(r => ci(r)[1])) * 1.15;
    const pos = v => 100 * Math.log(v / lo0) / Math.log(hi0 / lo0) + "%";
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
      r.appendChild(node("span", "val", rel ? `${x2(ratio(row))} (${ci(row).map(v => v.toFixed(1)).join("–")})`
        : `${n} (${talks} talk${talks === 1 ? "" : "s"})`));
      r.addEventListener("pointermove", ev => showTip(ev, row, phrase, base));
      r.addEventListener("focus", ev => showTip(ev, row, phrase, base));
      ["pointerleave", "blur"].forEach(t => r.addEventListener(t, () => { tip.hidden = true; }));
      container.appendChild(r);
    });
    if (!rel) return;
    // tick axis under the forest plot, aligned with the plot column
    const axis = node("div", "bar-row axis"), cell = node("span", "cell");
    [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].filter(v => v >= lo0 && v <= hi0).forEach(v => {
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
  render();
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
  const state = { map: "na", group: "abstracts", host: "with", na: "with" };
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
