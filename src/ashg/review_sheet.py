"""Write a compact review sheet of randomly sampled parsed abstracts.

A human checks each row against the PDF page and marks it OK or describes the
error; checked rows become golden-set test fixtures.

Usage: python src/ashg/review_sheet.py OUT.md ABSTRACTS.parquet [...] [--n 20] [--seed 0]
"""

import argparse

import pandas as pd

from authors import parse_authors, senior_author


def clip(s, n):
    s = (s or "").replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for path in args.abstracts:
        df = pd.read_parquet(path)
        df = df[~df.truncated]
        for r in df.sample(min(args.n, len(df)), random_state=args.seed).sort_values("page_start").itertuples():
            authors = parse_authors(r.authors_raw)
            senior = senior_author(authors) or {"name": "", "affils": []}
            rows.append(
                f"| {r.abstract_id} | {r.page_start} | {clip(r.title, 70)} | {len(authors)} "
                f"| {clip(senior['name'], 30)} | {clip('; '.join(senior['affils']), 60)} "
                f"| {clip(r.body[:40], 40)} … {clip(r.body[-40:], 40)} | |"
            )
    header = (
        "# Golden-set review (2025)\n\n"
        "Check each row against the PDF page. In the last column write `ok`, or what is wrong\n"
        "(e.g. `senior should be X`, `title cut`, `body includes next title`).\n\n"
        "| id | page | title | #authors | senior author | senior affils | body start … end | check |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    with open(args.out, "w") as f:
        f.write(header + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
