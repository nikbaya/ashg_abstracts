"""Render the GitHub Pages site (docs/) from the aggregate CSVs in outputs/.

One page per topic; every year with a {year}_summary.csv in outputs/ gets a panel on each topic
page (oldest first), switched with a year toggle that opens on the newest. Everything across years
(the all_*.csv pooled tables, the trends_* figures and the phrase explorer) is on the overview page.

The public site never names an institution, metro area, country or person for having no talks:
those rows are dropped here (the CSVs in outputs/ keep them). World regions are shown in full.

Usage: python src/ashg/build_site.py [--outputs outputs] [--docs docs] [--repo-url URL] [--site-url URL]
"""

import argparse
import hashlib
import html
import json
import pathlib
import shutil

import jinja2
import numpy as np
import pandas as pd

from center import great_circle_km
from figures import HOST_METRO

TEMPLATES = pathlib.Path(__file__).resolve().parent / "templates"
PAGES = [  # (file, nav label, page title)
    ("index.html", "Overview", "Talks vs posters at ASHG"),
    ("phrases.html", "Phrases", "Phrases that predict a talk"),
    ("institutions.html", "Institutions", "Institutions"),
    ("geography.html", "Geography", "Regions, countries and metro areas"),
    ("authors.html", "Senior authors", "Senior authors with the highest talk rate"),
    ("methods.html", "Methods", "Methods and caveats"),
]
YEAR_FIGURES = ("headline", "institutions", "metros", "regions", "senior_authors")
TREND_FIGURES = ("trends_talk_rate", "trends_phrases", "trends_biobanks", "trends_shift", "trends_sample_size", "trends_regions",
                 "trends_metros", "trends_center_na", "trends_center_world")
EXPLORER_ROWS = 10


def pct(v):
    return f"{100 * v:.1f}%"


def with_talks(df):
    """The site never lists an institution, place or person for having no talks."""
    return df[df.talks > 0]


def table(df, columns, sorted_by=None):
    """columns: [(header, column, formatter or None, kind)] -> sortable HTML table. kind is "num",
    "text", or "html" (pre-escaped markup, sorted by the column `{column}_sort`). Cells keep the raw
    value in data-v so sorting is numeric."""
    def th(h, col, kind):
        sort = ' aria-sort="descending"' if col == sorted_by else ""
        return f'<th class="{"" if kind == "num" else "text"}"{sort}>{html.escape(h)}</th>'

    head = "".join(th(h, col, kind) for h, col, _, kind in columns)
    rows = []
    for r in df.to_dict("records"):
        cells = []
        for _, col, fmt, kind in columns:
            v = r[col]
            if kind == "html":
                cells.append(f'<td class="text" data-v="{html.escape(str(r[col + "_sort"]))}">{v}</td>')
                continue
            shown = fmt(v) if fmt and pd.notna(v) else ("" if pd.isna(v) else v)
            cls = "" if kind == "num" else "text"
            cells.append(f'<td class="{cls}" data-v="{html.escape(str(v))}">{html.escape(str(shown))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def ci(r):
    return f"{pct(r.ci_lo)}–{pct(r.ci_hi)}"


SHARE_COLS = [("Abstracts", "abstracts", None, "num"), ("Talks", "talks", None, "num"),
              ("Talk rate", "talk_rate", pct, "num"), ("95% CI", "ci", None, "num")]


def first_author(df, columns):
    """Table of first-author credit (the outputs CSVs also hold senior-author rows, not shown on the site)."""
    df = df[df.basis == "first"]
    return table(df.assign(ci=df.apply(ci, axis=1)), columns, sorted_by="abstracts")


def phrase_cell(r):
    shown = html.escape(r["ngram"])
    if isinstance(r["related"], str) and r["related"]:
        shown += f'<span class="related">also {html.escape(", ".join(r["related"].split("; ")))}</span>'
    return shown


def year_context(out, year, label=None):
    """Tables and numbers for one year, or for year "all" (every year pooled; `label` e.g. "2022–2025")."""
    summary = pd.read_csv(out / f"{year}_summary.csv").iloc[0].to_dict()
    for col in ("abstracts", "talks", "truncated", "truncated_talks"):
        summary[col] = int(summary[col])

    premium = pd.read_csv(out / f"{year}_premium.csv")
    for col in ("ngram", "related"):
        premium[col] = premium[col].str.replace("<num>", "N", regex=False)
    premium["phrase"] = premium.apply(phrase_cell, axis=1)
    premium["phrase_sort"] = premium.ngram
    premium["or_ci"] = premium.apply(lambda r: f"{r.or_lo:.2f}–{r.or_hi:.2f}", axis=1)
    prem_cols = [("Phrase", "phrase", None, "html"), ("Talks", "talk_docs", None, "num"),
                 ("Posters", "poster_docs", None, "num"), ("Odds ratio", "or", "{:.2f}".format, "num"),
                 ("95% CI", "or_ci", None, "num")]

    inst = with_talks(pd.read_csv(out / f"{year}_institutions.csv"))
    metros = with_talks(pd.read_csv(out / f"{year}_metros.csv"))
    metros["institutes"] = metros.institutes.str.replace("; ", ", ")
    regions = pd.read_csv(out / f"{year}_regions.csv")
    regions["countries"] = regions.countries.str.replace("; ", ", ")
    countries = with_talks(pd.read_csv(out / f"{year}_countries.csv"))
    authors = with_talks(pd.read_csv(out / f"{year}_senior_authors.csv"))
    authors = authors.assign(ci=authors.apply(ci, axis=1))  # no rank column: the intervals overlap too much to rank people

    top = {d: list(premium[premium.direction == d].ngram.head(3)) for d in ("talk", "poster")}
    return {
        "year": year,
        "label": label or year,
        "host": HOST_METRO.get(year),
        "summary": summary,
        "top_phrases": top,
        "premium": {d: table(premium[premium.direction == d], prem_cols) for d in ("talk", "poster")},
        "institutions": first_author(inst, [("Institution", "institute", None, "text")] + SHARE_COLS),
        "metros": first_author(metros, [("Metro area", "metro", None, "text"),
                                        ("Largest institutions", "institutes", None, "text")] + SHARE_COLS),
        "regions": first_author(regions, [("Region", "region", None, "text"),
                                          ("Largest countries", "countries", None, "text")] + SHARE_COLS),
        "countries": first_author(countries, [("Country", "country", None, "text"),
                                              ("Region", "region", None, "text")] + SHARE_COLS),
        "senior_authors": table(authors, [("Senior author", "name", None, "text"),
                                          ("Main institution", "institute", None, "text")] + SHARE_COLS),
        "min_abstracts": int(authors.min_abstracts.iloc[0]) if len(authors) else 5,
    }


def sparkline(values, w=96, h=24, pad=3):
    """Inline SVG line of yearly values from 0 (bottom) to the series max, last point dotted."""
    top = max(values) or 1
    pts = [(pad + (w - 2 * pad) * i / (len(values) - 1), h - pad - (h - 2 * pad) * v / top)
           for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
            f'<polyline points="{line}"/><circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.5"/></svg>')


def sparkline_pair(talks, posters, w=120, h=40, pad=4):
    """Inline SVG of the yearly share of talks and of posters on one scale from 0 to the larger max."""
    top = max(talks + posters) or 1
    svg = f'<svg class="spark pair" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
    for cls, values in (("poster", posters), ("talk", talks)):
        pts = [(pad + (w - 2 * pad) * i / (len(values) - 1), h - pad - (h - 2 * pad) * v / top)
               for i, v in enumerate(values)]
        svg += (f'<polyline class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'
                f'<circle class="{cls}" cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.5"/>')
    return svg + "</svg>"


def highlights(out, years, n_rising=6, n_hot=6, min_hot_talks=10, n_rows=6):
    """Numbers for the overview's vignettes: rising topics and 2026's talk-leaning phrases, the clinical
    shift, the institutions with the highest pooled talk rate, and the host-city pull on the center of mass."""
    latest = years[-1]
    t = pd.read_csv(out / "trends_phrases.csv")
    rising = []
    for phrase, g in t[t.group == "Topics to watch"].groupby("phrase", sort=False):
        g = g.set_index("year").loc[years]
        rising.append({"phrase": phrase, "first": g.share.iloc[0], "last": g.share.iloc[-1],
                       "peak_year": int(g.share.idxmax()), "peak": g.share.max(),  # for topics that peaked earlier
                       "spark": sparkline(list(g.share)), "talks_last": int(g.talk_docs.iloc[-1])})
    rising.sort(key=lambda r: r["last"] - r["first"], reverse=True)  # biggest gain first

    prem = pd.read_csv(out / f"{latest}_premium.csv")
    hot = prem[(prem.direction == "talk") & (prem.talk_docs >= min_hot_talks)].head(n_hot)
    shift = pd.read_csv(out / "trends_shift.csv")
    prespecified = set(t[t.group.str.startswith("Poster-leaning")].phrase)  # picked before looking at the latest year
    shift = shift[shift.phrase.isin(prespecified)].to_dict("records")
    for r in shift:
        g = t[t.phrase == r["phrase"]].set_index("year").loc[years]
        r["spark"] = sparkline_pair(list(g.talk_share), list(g.poster_share))

    inst = pd.read_csv(out / "all_institutions.csv")
    pool = inst[inst.basis == "first"]
    top = pool[pool.talks > 0].sort_values(["talk_rate", "abstracts"], ascending=False).head(n_rows)
    latest_inst = pd.read_csv(out / f"{latest}_institutions.csv")
    latest_inst = latest_inst[(latest_inst.basis == "first") & (latest_inst.talks > 0)]
    latest_top = list(latest_inst.sort_values(["talk_rate", "abstracts"], ascending=False).head(3).institute)

    center = None
    if (out / "trends_center.csv").exists():
        c = pd.read_csv(out / "trends_center.csv")
        c = c[(c.view == "na") & (c.group == "abstracts")].pivot(index="year", columns="host_city",
                                                                    values=["lat", "lon"])
        km = great_circle_km(c.lat["with"], c.lon["with"], c.lat["without"], c.lon["without"])
        center = [{"year": int(y), "host": HOST_METRO.get(int(y)), "km": round(float(k))} for y, k in zip(c.index, km)]
    baseline = pool.baseline.iloc[0]
    return {"rising": rising[:n_rising], "hot": odds_rows(hot.sort_values("or", ascending=False)), "shift": shift,
            "inst": rate_rows(top, "institute", baseline), "inst_pool": len(pool), "baseline": baseline,
            "latest_top": latest_top, "center": center, "latest": latest}


def rate_rows(df, key, baseline):
    """Rows for an overview talk-rate table: bar, 95% CI and the all-abstracts line, as % of the widest interval."""
    scale = df.ci_hi.max()
    rows = [{**r, "label": r[key], "count": f"{r['talks']} / {r['abstracts']}", "val": f"{100 * r['talk_rate']:.1f}%",
             "bar": 100 * r["talk_rate"] / scale, "ci_left": 100 * r["ci_lo"] / scale,
             "ci_width": 100 * (r["ci_hi"] - r["ci_lo"]) / scale} for r in df.to_dict("records")]
    return {"rows": rows, "base": 100 * baseline / scale}


def odds_rows(df):
    """Rows for an overview odds-ratio table, like rate_rows but on a log scale with the line at 1×."""
    lo, hi = np.log(min(df.or_lo.min(), 1) / 1.1), np.log(df.or_hi.max() * 1.05)
    pos = lambda v: 100 * (np.log(v) - lo) / (hi - lo)
    rows = [{**r, "label": r["ngram"], "count": f"{r['talk_docs']} / {r['poster_docs']}", "val": f"{r['or']:.1f}×",
             "bar": pos(r["or"]), "ci_left": pos(r["or_lo"]), "ci_width": pos(r["or_hi"]) - pos(r["or_lo"])}
            for r in df.to_dict("records")]
    return {"rows": rows, "base": pos(1)}


def explorer_json(path):
    """Phrase-explorer data with zero-talk rows dropped and lists cut to EXPLORER_ROWS; embedded in a
    <script type="application/json">, so "</" must not close the tag."""
    data = json.loads(path.read_text())
    for periods in data["drivers"].values():
        for entry in periods.values():
            for kind in ("institutes", "metros"):
                entry[kind] = [r for r in entry[kind] if r[1] > 0][:EXPLORER_ROWS]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--repo-url", default="https://github.com/nikbaya/ashg_abstracts")
    ap.add_argument("--site-url", default="https://nikbaya.github.io/ashg_abstracts/")
    ap.add_argument("--aliases", default="aliases.csv")
    args = ap.parse_args()
    out, docs = pathlib.Path(args.outputs), pathlib.Path(args.docs)
    (docs / "img").mkdir(parents=True, exist_ok=True)
    (docs / "assets").mkdir(exist_ok=True)

    years = sorted((int(p.name[:4]) for p in out.glob("[0-9][0-9][0-9][0-9]_summary.csv")))  # ascending everywhere on the site
    for y in years:
        for fig in YEAR_FIGURES:
            shutil.copy(out / f"{y}_{fig}.png", docs / "img" / f"{y}_{fig}.png")
    trends = [f for f in TREND_FIGURES if (out / f"{f}.png").exists()]
    for f in trends:
        shutil.copy(out / f"{f}.png", docs / "img" / f"{f}.png")
    pooled = (out / "all_summary.csv").exists()
    if pooled:
        for fig in YEAR_FIGURES:
            shutil.copy(out / f"all_{fig}.png", docs / "img" / f"all_{fig}.png")
    stale = docs / "trends.html"  # merged into the overview
    stale.unlink(missing_ok=True)
    for f in (docs / "img").glob("*_senior.png"):  # senior-author credit figures are no longer on the site
        f.unlink()
    for f in (TEMPLATES / "static").iterdir():
        shutil.copy(f, docs / "assets" / f.name)
    # content hash in the asset URLs, so browsers don't pair new HTML with a cached old site.css
    asset_v = hashlib.sha1(b"".join(f.read_bytes() for f in sorted((TEMPLATES / "static").iterdir()))).hexdigest()[:8]
    drivers = out / "trends_drivers.json"
    center = out / "trends_center.json"  # center-of-mass map (geography page), from center.py

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=False,
                             trim_blocks=True, lstrip_blocks=True)
    pages = [{"file": f, "label": label, "title": title} for f, label, title in PAGES]
    shared = {"years": [year_context(out, y) for y in years], "trends": trends, "pages": pages,
              "pooled": year_context(out, "all", f"{years[0]}–{years[-1]}") if pooled else None,
              "drivers_json": explorer_json(drivers) if drivers.exists() else None,
              "center_json": center.read_text().replace("</", "<\\/") if center.exists() else None,
              "highlights": highlights(out, years) if pooled and (out / "trends_phrases.csv").exists() else None,
              "repo_url": args.repo_url, "site_url": args.site_url, "asset_v": asset_v,
              "n_institutes": pd.read_csv(args.aliases, comment="#").institute.nunique()}
    for i, page in enumerate(pages):
        html_out = env.get_template(page["file"] + ".j2").render(
            page=page, prev=pages[i - 1] if i else None, next=pages[i + 1] if i + 1 < len(pages) else None, **shared)
        (docs / page["file"]).write_text(html_out)
    (docs / ".nojekyll").touch()
    print(f"wrote {len(pages)} pages to {docs}/ ({', '.join(map(str, years))})")


if __name__ == "__main__":
    main()
