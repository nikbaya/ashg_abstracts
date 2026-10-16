"""List the most frequent first/senior-author affiliation segments.

Used to seed the hand-curated aliases.csv. Department/division segments are
skipped, since they never name an institution on their own.

Usage: python src/ashg/seed_institutions.py ABSTRACTS.parquet [...] [--top 250]
"""

import argparse
import collections
import re

import pandas as pd

from authors import first_author, parse_authors, senior_author

SUBUNIT_RE = re.compile(r"^(dep(artmen)?t\.?|div(ision)?|program|lab(oratory)? of|section|unit|school of|faculty|graduate)\b", re.I)


def affil_segments(author):
    for affil in author["affils"] if author else []:
        for seg in re.split(r"\s*;\s*", affil):
            seg = seg.strip(" .")
            if seg and not SUBUNIT_RE.match(seg):
                yield seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("abstracts", nargs="+")
    ap.add_argument("--top", type=int, default=250)
    args = ap.parse_args()

    counts = collections.Counter()
    for path in args.abstracts:
        df = pd.read_parquet(path)
        for raw in df.loc[~df.truncated, "authors_raw"]:
            authors = parse_authors(raw)
            segs = set(affil_segments(first_author(authors))) | set(affil_segments(senior_author(authors)))
            counts.update(segs)  # once per abstract
    for seg, n in counts.most_common(args.top):
        print(f"{n:4d} {seg[:70]}")


if __name__ == "__main__":
    main()
