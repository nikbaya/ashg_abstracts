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

// Filter box: hides table rows whose text doesn't contain the query.
document.querySelectorAll("[data-filter]").forEach(input => input.addEventListener("input", () => {
  const q = input.value.trim().toLowerCase();
  document.querySelectorAll("table.filterable").forEach(table => {
    let shown = 0;
    [...table.tBodies[0].rows].forEach(r => { r.hidden = q && !r.textContent.toLowerCase().includes(q); shown += !r.hidden; });
    const note = table.parentElement.querySelector(".empty");
    if (note) note.hidden = shown > 0;
  });
}));

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
  const byPhrase = Object.fromEntries(data.phrases.map(p => [p.phrase, p]));

  function fillOptions(q) {
    const keep = phraseSel.value;
    phraseSel.replaceChildren();
    const groups = {};
    data.phrases.filter(p => !q || p.phrase.toLowerCase().includes(q)).forEach(p => (groups[p.group] ??= []).push(p.phrase));
    Object.entries(groups).forEach(([g, ps]) => {
      const og = document.createElement("optgroup"); og.label = g;
      ps.forEach(p => { const o = node("option", null, p); o.value = p; og.appendChild(o); });
      phraseSel.appendChild(og);
    });
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
  function drawChart(p) {
    // share of talks and of posters mentioning the phrase, per year
    const W = 360, H = 150, L = 40, R = 18, T = 12, B = 24;
    const series = [["talk", 0], ["poster", 1]].map(([k, i]) =>
      [k, p.counts.map((c, y) => c[i] / data.totals[y][i])]);
    const top = Math.max(...series.flatMap(([, v]) => v)) * 1.15 || 0.01;
    const step = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2].find(s => top / s <= 4) ?? 0.25;
    const x = i => L + (W - L - R) * (data.years.length > 1 ? i / (data.years.length - 1) : 0.5);
    const y = v => T + (H - T - B) * (1 - v / top);
    chart.replaceChildren();
    for (let v = 0; v <= top + 1e-9; v += step) {
      chart.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }));
      chart.appendChild(svg("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "axis" }, (100 * v).toFixed(step < 0.01 ? 1 : 0) + "%"));
    }
    data.years.forEach((yr, i) => chart.appendChild(svg("text", { x: x(i), y: H - 6, "text-anchor": "middle", class: "axis" }, yr)));
    series.forEach(([k, v]) => {
      chart.appendChild(svg("polyline", { points: v.map((s, i) => `${x(i)},${y(s)}`).join(" "), class: "line " + k }));
      v.forEach((s, i) => {
        const dot = svg("circle", { cx: x(i), cy: y(s), r: 4, class: "dot " + k });
        dot.appendChild(svg("title", {}, `${data.years[i]}: ${pct(s)} of ${k}s (${p.counts[i][k === "talk" ? 0 : 1]})`));
        chart.appendChild(dot);
      });
    });
    chart.setAttribute("aria-label", `Share of talks and of posters mentioning “${p.phrase}”, by year`);
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

  function showTip(ev, row, phrase) {
    const [name, talks, posters, total] = row, n = talks + posters;
    tip.replaceChildren();
    tip.appendChild(node("strong", null, `${n} abstract${n === 1 ? "" : "s"}`));
    tip.appendChild(node("div", null, `${name} · mention “${phrase}”`));
    [["talk", talks, "talks"], ["poster", posters, "posters"]].forEach(([k, v, label]) => {
      const d = node("div"); const key = node("span", "k"); key.style.background = `var(--${k})`;
      d.append(key, `${v} ${label}`); tip.appendChild(d);
    });
    tip.appendChild(node("div", null, `${pct(n / total)} of its ${total} abstracts${year === "all" ? "" : " in " + year}`));
    tip.hidden = false;
    const r = ev.type === "focus" ? ev.target.getBoundingClientRect() : { left: ev.clientX, bottom: ev.clientY };
    tip.style.left = Math.min(r.left + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(r.bottom + 12, innerHeight - tip.offsetHeight - 8) + "px";
  }

  function list(container, rows, phrase) {
    container.replaceChildren();
    if (!rows.length) { container.appendChild(node("p", "empty", "None with a talk mentioning it.")); return; }
    const max = Math.max(...rows.map(r => r[1] + r[2]));
    rows.forEach(row => {
      const [name, talks, posters] = row, n = talks + posters;
      const r = node("div", "bar-row"); r.tabIndex = 0;
      r.appendChild(node("span", "name", name));
      const track = node("span", "track"); track.style.width = (100 * n / max) + "%";
      [["seg-talk", talks], ["seg-poster", posters]].forEach(([cls, v]) => {
        if (v) { const seg = node("span", cls); seg.style.flex = String(v); track.appendChild(seg); }
      });
      const cell = node("span"); cell.appendChild(track); r.appendChild(cell);
      r.appendChild(node("span", "val", `${n} (${talks} talk${talks === 1 ? "" : "s"})`));
      r.addEventListener("pointermove", ev => showTip(ev, row, phrase));
      r.addEventListener("focus", ev => showTip(ev, row, phrase));
      ["pointerleave", "blur"].forEach(t => r.addEventListener(t, () => { tip.hidden = true; }));
      container.appendChild(r);
    });
  }

  function render() {
    const phrase = phraseSel.value, d = data.drivers[phrase][year], n = d.talks + d.posters;
    drawChart(byPhrase[phrase]);
    document.getElementById("drv-summary").textContent =
      `${n} abstracts mention “${phrase}” ${year === "all" ? "across all years" : "in " + year} ` +
      `(${d.talks} talks, ${d.posters} posters); ${d.unmatched} have no matched first-author institution.`;
    list(document.getElementById("drv-institutes"), d.institutes, phrase);
    list(document.getElementById("drv-metros"), d.metros, phrase);
  }
  render();
})();
