"""Render docs/index.html (GitHub Pages) from the aggregate CSVs in outputs/.

Every year with a {year}_summary.csv in outputs/ gets a section, newest first.

Usage: python src/ashg/build_site.py [--outputs outputs] [--docs docs] [--repo-url URL]
"""

import argparse
import html
import pathlib
import shutil

import jinja2
import pandas as pd

TEMPLATES = pathlib.Path(__file__).resolve().parent / "templates"


def pct(v):
    return f"{100 * v:.1f}%"


def table(df, columns):
    """columns: [(header, column, formatter or None, is_text)] -> sortable HTML table.
    Cells keep the raw value in data-v so sorting is numeric."""
    head = "".join(f'<th class="{"text" if text else ""}">{html.escape(h)}</th>' for h, _, _, text in columns)
    rows = []
    for r in df.to_dict("records"):
        cells = []
        for _, col, fmt, text in columns:
            v = r[col]
            shown = fmt(v) if fmt and pd.notna(v) else ("" if pd.isna(v) else v)
            cells.append(f'<td class="{"text" if text else ""}" data-v="{html.escape(str(v))}">{html.escape(str(shown))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def ci(r):
    return f"{pct(r.ci_lo)}–{pct(r.ci_hi)}"


def year_context(out, year, data):
    summary = pd.read_csv(out / f"{year}_summary.csv").iloc[0].to_dict()
    summary["abstracts"], summary["talks"] = int(summary["abstracts"]), int(summary["talks"])
    inst = pd.read_csv(out / f"{year}_institutions.csv")
    inst["ci"] = inst.apply(ci, axis=1)
    inst_cols = [("#", "rank", None, False), ("Institution", "institute", None, True),
                 ("Abstracts", "abstracts", None, False), ("Talks", "talks", None, False),
                 ("Talk share", "talk_share", pct, False), ("95% CI", "ci", None, False)]
    authors = pd.read_csv(out / f"{year}_senior_authors.csv")
    authors["ci"] = authors.apply(ci, axis=1)
    premium = pd.read_csv(out / f"{year}_premium.csv")
    premium["ngram"] = premium.ngram.str.replace("<num>", "N", regex=False)
    premium["or_ci"] = premium.apply(lambda r: f"{r.or_lo:.2f}–{r.or_hi:.2f}", axis=1)
    prem_cols = [("Phrase", "ngram", None, True), ("Talks", "talk_docs", None, False),
                 ("Posters", "poster_docs", None, False), ("Odds ratio", "or", "{:.2f}".format, False),
                 ("95% CI", "or_ci", None, False)]
    ss = pd.read_csv(out / f"{year}_sample_size.csv")
    n_truncated = 0
    abstracts = sorted((data / "abstracts").glob(f"{year}_*.parquet"))
    if abstracts:
        n_truncated = int(sum(pd.read_parquet(p, columns=["truncated"]).truncated.sum() for p in abstracts))
    return {
        "year": year,
        "summary": summary,
        "n_truncated": n_truncated,
        "premium": {d: table(premium[premium.direction == d], prem_cols) for d in ("talk", "poster")},
        "institutions": {b: table(inst[inst.basis == b], inst_cols) for b in ("senior", "first")},
        "senior_authors": table(authors, [("#", "rank", None, False), ("Senior author", "name", None, True),
                                          ("Main institution", "institute", None, True),
                                          ("Abstracts", "abstracts", None, False), ("Talks", "talks", None, False),
                                          ("Talk share", "talk_share", pct, False), ("95% CI", "ci", None, False)]),
        "min_abstracts": int(authors.min_abstracts.iloc[0]) if len(authors) else 5,
        "sample_size": table(ss, [("Type", "type", None, True), ("Abstracts", "abstracts", "{:,}".format, False),
                                  ("With an N", "coverage", pct, False), ("Median N", "median_max_n", "{:,.0f}".format, False),
                                  ("25th pct", "q25", "{:,.0f}".format, False), ("75th pct", "q75", "{:,.0f}".format, False)]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--repo-url", default="https://github.com/nikbaya/ashg_abstracts")
    args = ap.parse_args()
    out, docs = pathlib.Path(args.outputs), pathlib.Path(args.docs)
    (docs / "img").mkdir(parents=True, exist_ok=True)

    years = sorted((int(p.name.split("_")[0]) for p in out.glob("*_summary.csv")), reverse=True)
    for y in years:
        shutil.copy(out / f"{y}_headline.png", docs / "img" / f"{y}_headline.png")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=False)
    page = env.get_template("index.html.j2").render(
        years=[year_context(out, y, pathlib.Path(args.data)) for y in years], repo_url=args.repo_url)
    (docs / "index.html").write_text(page)
    (docs / ".nojekyll").touch()
    print(f"wrote {docs / 'index.html'} ({', '.join(map(str, years))})")


if __name__ == "__main__":
    main()
