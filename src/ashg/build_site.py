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


def year_context(out, year):
    summary = pd.read_csv(out / f"{year}_summary.csv").iloc[0].to_dict()
    for col in ("abstracts", "talks", "truncated", "truncated_talks"):
        summary[col] = int(summary[col])
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
    return {
        "year": year,
        "summary": summary,
        "premium": {d: table(premium[premium.direction == d], prem_cols) for d in ("talk", "poster")},
        "institutions": {b: table(inst[inst.basis == b], inst_cols) for b in ("senior", "first")},
        "senior_authors": table(authors, [("#", "rank", None, False), ("Senior author", "name", None, True),
                                          ("Main institution", "institute", None, True),
                                          ("Abstracts", "abstracts", None, False), ("Talks", "talks", None, False),
                                          ("Talk share", "talk_share", pct, False), ("95% CI", "ci", None, False)]),
        "min_abstracts": int(authors.min_abstracts.iloc[0]) if len(authors) else 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--repo-url", default="https://github.com/nikbaya/ashg_abstracts")
    args = ap.parse_args()
    out, docs = pathlib.Path(args.outputs), pathlib.Path(args.docs)
    (docs / "img").mkdir(parents=True, exist_ok=True)

    years = sorted((int(p.name.split("_")[0]) for p in out.glob("*_summary.csv")), reverse=True)
    for y in years:
        for fig in ("headline", "institutions", "senior_authors"):
            shutil.copy(out / f"{y}_{fig}.png", docs / "img" / f"{y}_{fig}.png")
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=False)
    page = env.get_template("index.html.j2").render(
        years=[year_context(out, y) for y in years], repo_url=args.repo_url)
    (docs / "index.html").write_text(page)
    (docs / ".nojekyll").touch()
    print(f"wrote {docs / 'index.html'} ({', '.join(map(str, years))})")


if __name__ == "__main__":
    main()
